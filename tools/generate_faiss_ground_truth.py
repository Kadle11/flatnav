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


def read_fvecs_range(path: str, dim: int, start: int, end: int) -> np.ndarray:
    """Read vectors in [start, end) from .fvecs as contiguous float32 array."""
    if start < 0 or end < start:
        raise ValueError(f"Invalid range: start={start}, end={end}")
    count = end - start
    if count == 0:
        return np.empty((0, dim), dtype=np.float32)

    vec_bytes = (dim + 1) * np.dtype(np.float32).itemsize
    offset = start * vec_bytes
    raw = np.memmap(
        path,
        dtype=np.float32,
        mode="r",
        offset=offset,
        shape=(count, dim + 1),
    )
    return np.ascontiguousarray(raw[:, 1:], dtype=np.float32)


def merge_topk(
    running_dists: np.ndarray,
    running_ids: np.ndarray,
    cand_dists: np.ndarray,
    cand_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge current global top-k with candidate top-k and keep exact best k."""
    k = running_dists.shape[1]

    all_dists = np.concatenate((running_dists, cand_dists), axis=1)
    all_ids = np.concatenate((running_ids, cand_ids), axis=1)

    # Partial select first, then sort the kept k for deterministic ascending L2 output.
    part = np.argpartition(all_dists, kth=k - 1, axis=1)[:, :k]
    merged_d = np.take_along_axis(all_dists, part, axis=1)
    merged_i = np.take_along_axis(all_ids, part, axis=1)

    order = np.argsort(merged_d, axis=1)
    merged_d = np.take_along_axis(merged_d, order, axis=1)
    merged_i = np.take_along_axis(merged_i, order, axis=1)

    return merged_d, merged_i


def write_ivecs(path: str, indices: np.ndarray) -> None:
    """Write int32 indices matrix (nq, k) to standard .ivecs file."""
    if indices.dtype != np.int32:
        indices = indices.astype(np.int32, copy=False)

    nq, k = indices.shape
    out = np.empty((nq, k + 1), dtype=np.int32)
    out[:, 0] = k
    out[:, 1:] = indices

    with open(path, "wb") as f:
        out.tofile(f)


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
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.base_batch_size <= 0:
        raise ValueError("--base-batch-size must be > 0")
    if args.query_batch_size <= 0:
        raise ValueError("--query-batch-size must be > 0")

    nb, dim_b = read_fvecs_header(args.base_fvecs)
    nq, dim_q = read_fvecs_header(args.queries_fvecs)
    if dim_b != dim_q:
        raise ValueError(f"Dimension mismatch: base={dim_b}, queries={dim_q}")

    if args.threads > 0:
        faiss.omp_set_num_threads(args.threads)

    k = min(args.k, nb)

    print(f"Base vectors   : {nb:,} x {dim_b}")
    print(f"Query vectors  : {nq:,} x {dim_q}")
    print(f"k              : {k}")
    print(f"Base batch     : {args.base_batch_size:,}")
    print(f"Query batch    : {args.query_batch_size:,}")
    if args.threads > 0:
        print(f"FAISS threads  : {args.threads}")

    # Global running top-k per query. Initialized with +inf and invalid IDs.
    running_dists = np.full((nq, k), np.inf, dtype=np.float32)
    running_ids = np.full((nq, k), -1, dtype=np.int32)

    t0 = time.time()
    total_base_batches = (nb + args.base_batch_size - 1) // args.base_batch_size

    for base_batch_idx, base_start in enumerate(range(0, nb, args.base_batch_size), start=1):
        base_end = min(base_start + args.base_batch_size, nb)
        xb = read_fvecs_range(args.base_fvecs, dim_b, base_start, base_end)

        index = faiss.IndexFlatL2(dim_b)
        index.add(xb)

        local_k = min(k, xb.shape[0])
        for q_start in range(0, nq, args.query_batch_size):
            q_end = min(q_start + args.query_batch_size, nq)
            xq = read_fvecs_range(args.queries_fvecs, dim_q, q_start, q_end)

            cand_d, cand_i = index.search(xq, local_k)
            cand_i = (cand_i + base_start).astype(np.int32, copy=False)

            # If local_k < k, pad candidate arrays so merge function can stay generic.
            if local_k < k:
                pad_cols = k - local_k
                dpad = np.full((cand_d.shape[0], pad_cols), np.inf, dtype=np.float32)
                ipad = np.full((cand_i.shape[0], pad_cols), -1, dtype=np.int32)
                cand_d = np.concatenate((cand_d, dpad), axis=1)
                cand_i = np.concatenate((cand_i, ipad), axis=1)

            merged_d, merged_i = merge_topk(
                running_dists[q_start:q_end],
                running_ids[q_start:q_end],
                cand_d,
                cand_i,
            )
            running_dists[q_start:q_end] = merged_d
            running_ids[q_start:q_end] = merged_i

        elapsed = time.time() - t0
        print(
            f"Processed base batch {base_batch_idx}/{total_base_batches} "
            f"({base_end - base_start:,} vectors, global end {base_end:,}) "
            f"elapsed={elapsed/60.0:.2f} min"
        )

    write_ivecs(args.output_ivecs, running_ids)
    total_elapsed = time.time() - t0
    print(f"Wrote ground truth to: {args.output_ivecs}")
    print(f"Total elapsed: {total_elapsed/60.0:.2f} min")


if __name__ == "__main__":
    main()
