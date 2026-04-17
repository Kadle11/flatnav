#!/usr/bin/env python3
"""Generate exact k-NN ground truth (.ivecs) for large .fvecs datasets with FAISS.

This script is designed for memory-constrained environments where loading all
100M base vectors into one in-memory index is risky. It computes exact results by:
1) Iterating over the base vectors in chunks.
2) Running brute-force FAISS IndexFlatL2 search per chunk.
3) Merging each chunk's top-k with global top-k for every query.

The final output is standard .ivecs format where each row is:
[int32 k][int32 neighbor_0]...[int32 neighbor_{k-1}]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import numpy as np


def read_fvecs_header(path: str) -> tuple[int, int]:
    """Return (num_vectors, dim) for an .fvecs file."""
    with open(path, "rb") as f:
        dim = int(np.fromfile(f, dtype=np.int32, count=1)[0])
    vec_bytes = (dim + 1) * np.dtype(np.float32).itemsize
    file_size = os.path.getsize(path)
    if file_size % vec_bytes != 0:
        raise ValueError(f"Invalid .fvecs file size for {path}")
    num_vectors = file_size // vec_bytes
    return int(num_vectors), dim


class FvecsMemmapReader:
    """Persistent memmap-backed reader for efficient repeated range reads."""

    def __init__(self, path: str):
        self.path = path
        self.num_vectors, self.dim = read_fvecs_header(path)
        self._raw = np.memmap(
            path,
            dtype=np.float32,
            mode="r",
            shape=(self.num_vectors, self.dim + 1),
        )

    def read_range(self, start: int, end: int) -> np.ndarray:
        """Read vectors in [start, end) as contiguous float32 array."""
        if start < 0 or end < start or end > self.num_vectors:
            raise ValueError(f"Invalid range: start={start}, end={end}")
        if end == start:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.ascontiguousarray(self._raw[start:end, 1:], dtype=np.float32)


def merge_topk(
    running_dists: np.ndarray,
    running_ids: np.ndarray,
    cand_dists: np.ndarray,
    cand_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge current global top-k with candidate top-k and keep exact best k.

    This is a key/value merge where key=distance and value=id. We keep the k
    smallest keys and use id as deterministic tie-break for equal distances.
    """
    k = running_dists.shape[1]

    all_dists = np.concatenate((running_dists, cand_dists), axis=1)
    all_ids = np.concatenate((running_ids, cand_ids), axis=1)

    # Partial select first to avoid full sort cost.
    part = np.argpartition(all_dists, kth=k - 1, axis=1)[:, :k]
    merged_d = np.take_along_axis(all_dists, part, axis=1)
    merged_i = np.take_along_axis(all_ids, part, axis=1)

    # Final deterministic ordering: primary key distance, secondary key id.
    out_d = np.empty_like(merged_d)
    out_i = np.empty_like(merged_i)
    for r in range(merged_d.shape[0]):
        order = np.lexsort((merged_i[r], merged_d[r]))
        out_d[r] = merged_d[r, order]
        out_i[r] = merged_i[r, order]

    return out_d, out_i


def write_ivecs_chunk(file_obj, indices: np.ndarray) -> None:
    """Write one chunk of int32 indices matrix (batch_nq, k) in .ivecs row format."""
    if indices.dtype != np.int32:
        indices = indices.astype(np.int32, copy=False)

    batch_nq, k = indices.shape
    out = np.empty((batch_nq, k + 1), dtype=np.int32)
    out[:, 0] = k
    out[:, 1:] = indices
    out.tofile(file_obj)


def exact_search_single_query_batch(
    faiss,
    base_reader: FvecsMemmapReader,
    xq: np.ndarray,
    k: int,
    base_batch_size: int,
    log_prefix: str,
) -> np.ndarray:
    """Compute exact top-k ids for one query batch by scanning all base chunks."""
    nb = base_reader.num_vectors
    dim = base_reader.dim
    total_base_batches = (nb + base_batch_size - 1) // base_batch_size

    running_dists = np.full((xq.shape[0], k), np.inf, dtype=np.float32)
    running_ids = np.full((xq.shape[0], k), -1, dtype=np.int32)

    for base_batch_idx, base_start in enumerate(range(0, nb, base_batch_size), start=1):
        base_end = min(base_start + base_batch_size, nb)
        xb = base_reader.read_range(base_start, base_end)

        index = faiss.IndexFlatL2(dim)
        index.add(xb)

        local_k = min(k, xb.shape[0])
        cand_d, cand_i = index.search(xq, local_k)
        cand_i = (cand_i + base_start).astype(np.int32, copy=False)

        if local_k < k:
            pad_cols = k - local_k
            dpad = np.full((cand_d.shape[0], pad_cols), np.inf, dtype=np.float32)
            ipad = np.full((cand_i.shape[0], pad_cols), -1, dtype=np.int32)
            cand_d = np.concatenate((cand_d, dpad), axis=1)
            cand_i = np.concatenate((cand_i, ipad), axis=1)

        if base_batch_idx == 1:
            running_dists[:] = cand_d
            running_ids[:] = cand_i
        else:
            running_dists, running_ids = merge_topk(
                running_dists,
                running_ids,
                cand_d,
                cand_i,
            )

        if base_batch_idx == total_base_batches or base_batch_idx % 10 == 0:
            print(
                f"{log_prefix} merged base batch "
                f"{base_batch_idx}/{total_base_batches}"
            )

    return running_ids


def load_ivecs_ids(path: str, nq: int, k: int) -> np.ndarray:
    """Memory-map output .ivecs and return only id matrix (nq, k)."""
    row_width = k + 1
    expected_bytes = nq * row_width * np.dtype(np.int32).itemsize
    actual_bytes = os.path.getsize(path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Unexpected output size for {path}: got {actual_bytes} bytes, "
            f"expected {expected_bytes}"
        )

    mm = np.memmap(path, dtype=np.int32, mode="r", shape=(nq, row_width))
    return np.ascontiguousarray(mm[:, 1:])


def main() -> None:
    try:
        import faiss  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "FAISS is required. Install with `pip install faiss-cpu` "
            "(or a GPU FAISS package for your environment)."
        ) from exc

    parser = argparse.ArgumentParser(
        description="Generate exact batched FAISS ground truth (.ivecs) for .fvecs files"
    )
    parser.add_argument("--base-fvecs", required=True, help="Path to base vectors .fvecs")
    parser.add_argument("--queries-fvecs", required=True, help="Path to query vectors .fvecs")
    parser.add_argument("--output-ivecs", required=True, help="Output .ivecs path")
    parser.add_argument("--k", type=int, default=100, help="Number of nearest neighbors")
    parser.add_argument(
        "--base-batch-size",
        type=int,
        default=1_000_000,
        help="Base vectors per chunk (default: 1,000,000)",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=20_000,
        help="Queries per chunk (default: 20,000)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="FAISS CPU threads, 0 means FAISS default",
    )
    parser.add_argument(
        "--query-start",
        type=int,
        default=0,
        help="Global query start index (inclusive, default: 0)",
    )
    parser.add_argument(
        "--query-end",
        type=int,
        default=-1,
        help="Global query end index (exclusive, default: all queries)",
    )
    parser.add_argument(
        "--preload-queries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Preload all queries into RAM once and reuse across base batches "
            "(enabled by default)."
        ),
    )
    parser.add_argument(
        "--validate-after-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run an exact post-run validation pass and print validation accuracy",
    )
    parser.add_argument(
        "--validation-queries",
        type=int,
        default=200,
        help="Number of queries to validate after run (default: 200)",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=42,
        help="RNG seed for validation query sampling (default: 42)",
    )
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.base_batch_size <= 0:
        raise ValueError("--base-batch-size must be > 0")
    if args.query_batch_size <= 0:
        raise ValueError("--query-batch-size must be > 0")

    base_reader = FvecsMemmapReader(args.base_fvecs)
    query_reader = FvecsMemmapReader(args.queries_fvecs)

    nb, dim_b = base_reader.num_vectors, base_reader.dim
    nq_total, dim_q = query_reader.num_vectors, query_reader.dim
    if dim_b != dim_q:
        raise ValueError(f"Dimension mismatch: base={dim_b}, queries={dim_q}")

    q_start = args.query_start
    q_end = nq_total if args.query_end < 0 else min(args.query_end, nq_total)
    if q_start < 0 or q_start > q_end:
        raise ValueError(
            f"Invalid query range: query-start={q_start}, query-end={q_end}, total={nq_total}"
        )
    nq = q_end - q_start
    if nq == 0:
        raise ValueError("Selected query range is empty")

    if args.threads > 0:
        faiss.omp_set_num_threads(args.threads)

    k = min(args.k, nb)

    print(f"Base vectors   : {nb:,} x {dim_b}")
    print(f"Query vectors  : {nq_total:,} x {dim_q}")
    print(f"Query range    : [{q_start}, {q_end}) => {nq:,} queries")
    print(f"k              : {k}")
    print(f"Base batch     : {args.base_batch_size:,}")
    print(f"Query batch    : {args.query_batch_size:,}")
    print(f"Preload queries: {args.preload_queries}")
    if args.threads > 0:
        print(f"FAISS threads  : {args.threads}")

    all_queries: Optional[np.ndarray] = None
    if args.preload_queries:
        tq = time.time()
        all_queries = query_reader.read_range(q_start, q_end)
        print(
            f"Preloaded queries: {all_queries.shape[0]:,} vectors "
            f"in {time.time() - tq:.2f} sec"
        )

    t0 = time.time()
    next_write_log = 1000
    written_queries = 0
    with open(args.output_ivecs, "wb") as out_f:
        for q_start in range(0, nq, args.query_batch_size):
            q_end = min(q_start + args.query_batch_size, nq)
            if all_queries is None:
                xq = query_reader.read_range(
                    args.query_start + q_start,
                    args.query_start + q_end,
                )
            else:
                xq = all_queries[q_start:q_end]

            merged_ids = exact_search_single_query_batch(
                faiss,
                base_reader,
                xq,
                k,
                args.base_batch_size,
                log_prefix=(
                    f"Query batch [{args.query_start + q_start}:{args.query_start + q_end})"
                ),
            )
            write_ivecs_chunk(out_f, merged_ids)

            written_queries += q_end - q_start
            while written_queries >= next_write_log:
                print(f"Write progress  : {next_write_log:,}/{nq:,} queries")
                next_write_log += 1000

    total_elapsed = time.time() - t0
    print(f"Wrote ground truth to: {args.output_ivecs}")
    print(f"Total elapsed: {total_elapsed/60.0:.2f} min")

    if args.validate_after_run:
        val_count = min(max(1, args.validation_queries), nq)
        rng = np.random.default_rng(args.validation_seed)
        val_local_ids = np.sort(rng.choice(nq, size=val_count, replace=False))
        val_global_ids = args.query_start + val_local_ids

        print(
            f"Starting post-run validation on {val_count} queries "
            f"(seed={args.validation_seed})"
        )
        tv = time.time()

        if all_queries is None:
            xq_val = np.vstack(
                [query_reader.read_range(i, i + 1) for i in val_global_ids]
            ).astype(np.float32, copy=False)
        else:
            xq_val = all_queries[val_local_ids]

        recomputed_ids = exact_search_single_query_batch(
            faiss,
            base_reader,
            xq_val,
            k,
            args.base_batch_size,
            log_prefix="Validation",
        )
        output_ids = load_ivecs_ids(args.output_ivecs, nq, k)[val_local_ids]

        exact_match = np.all(recomputed_ids == output_ids, axis=1)
        set_recall = np.array(
            [
                len(set(recomputed_ids[i].tolist()) & set(output_ids[i].tolist())) / float(k)
                for i in range(val_count)
            ],
            dtype=np.float64,
        )

        print(
            "Validation accuracy: "
            f"exact-order={exact_match.mean() * 100.0:.3f}% | "
            f"set-recall@{k}={set_recall.mean() * 100.0:.3f}%"
        )
        print(f"Validation elapsed: {(time.time() - tv)/60.0:.2f} min")


if __name__ == "__main__":
    main()
