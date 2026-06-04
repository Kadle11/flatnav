import argparse
import gc
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import flatnav
import hnswlib
import numpy as np
from flatnav.data_type import DataType

from data_loader import get_data_loader


FLATNAV_DATA_TYPES = {
    "float32": DataType.float32,
    "uint8": DataType.uint8,
    "int8": DataType.int8,
}


def compute_recall_at_k(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    found_set = set(found[:k].tolist())
    truth_set = set(truth[:k].tolist())
    if not truth_set:
        return 0.0
    return len(found_set.intersection(truth_set)) / float(k)


def _to_float32_contiguous(batch: np.ndarray) -> np.ndarray:
    if batch.dtype == np.float32 and batch.flags.c_contiguous:
        return batch
    return np.ascontiguousarray(batch, dtype=np.float32)


def _sorted_row_fraction(values: np.ndarray) -> float:
    if values.ndim != 2 or values.shape[1] < 2:
        return 0.0
    deltas = np.diff(values, axis=1)
    return float(np.mean(np.all(deltas >= 0, axis=1)))


def build_flatnav_index_from_graph_file(
    train_data: np.ndarray,
    metric: str,
    num_node_links: int,
    mtx_filename: str,
) -> Tuple[Any, float]:
    dataset_size, dim = train_data.shape

    logging.info("Creating FlatNav index from graph file %s", mtx_filename)
    build_start = time.time()
    index = flatnav.index.create(
        distance_type=metric,
        index_data_type=FLATNAV_DATA_TYPES["float32"],
        dim=dim,
        dataset_size=dataset_size,
        max_edges_per_node=num_node_links,
        verbose=False,
        collect_stats=False,
    )

    graph_load_start = time.time()
    index.allocate_nodes(data=train_data).build_graph_links(mtx_filename)
    graph_load_sec = time.time() - graph_load_start
    logging.info("Loaded graph links from %s in %.2f sec", mtx_filename, graph_load_sec)

    return index, time.time() - build_start


def build_flatnav_index_from_hnsw_graph(
    train_data: np.ndarray,
    metric: str,
    num_node_links: int,
    ef_construction: int,
    num_build_threads: int,
    build_batch_size: int,
    graph_tmp_dir: str,
    save_mtx: str,
) -> Tuple[Any, float]:
    dataset_size, dim = train_data.shape
    hnsw_space = metric if metric == "l2" else "ip"

    hnsw_index = hnswlib.Index(space=hnsw_space, dim=dim)
    hnsw_index.init_index(
        max_elements=dataset_size,
        ef_construction=ef_construction,
        M=max(2, num_node_links // 2),
    )
    hnsw_index.set_num_threads(max(1, num_build_threads))

    logging.info("Building HNSW base layer graph in batches")
    build_start = time.time()
    for start in range(0, dataset_size, build_batch_size):
        end = min(start + build_batch_size, dataset_size)
        batch = _to_float32_contiguous(train_data[start:end])
        labels = np.arange(start, end, dtype=np.int32)
        hnsw_index.add_items(batch, labels)
        if (end == dataset_size) or ((end // build_batch_size) % 5 == 0):
            logging.info("HNSW added %d/%d vectors", end, dataset_size)

    if save_mtx:
        save_mtx_path = Path(save_mtx).expanduser().resolve()
        save_mtx_path.parent.mkdir(parents=True, exist_ok=True)
        mtx_filename = str(save_mtx_path)
        remove_mtx_after_load = False
    else:
        graph_tmp_dir_path = Path(graph_tmp_dir).expanduser().resolve()
        graph_tmp_dir_path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".mtx", delete=False, dir=graph_tmp_dir_path
        ) as tmp:
            mtx_filename = tmp.name
        remove_mtx_after_load = True

    save_start = time.time()
    hnsw_index.save_base_layer_graph(filename=mtx_filename)
    save_sec = time.time() - save_start
    mtx_size_bytes = os.path.getsize(mtx_filename)
    mtx_size_gib = mtx_size_bytes / float(1024**3)
    save_mib_per_sec = (mtx_size_bytes / float(1024**2)) / max(save_sec, 1e-9)
    logging.info(
        "Saved base-layer graph to %s (%.2f GiB) in %.2f sec (%.2f MiB/s)",
        mtx_filename,
        mtx_size_gib,
        save_sec,
        save_mib_per_sec,
    )

    # Release HNSW memory before building FlatNav to reduce memory pressure.
    del hnsw_index
    gc.collect()

    try:
        index, _ = build_flatnav_index_from_graph_file(
            train_data=train_data,
            metric=metric,
            num_node_links=num_node_links,
            mtx_filename=mtx_filename,
        )
    finally:
        if remove_mtx_after_load:
            try:
                os.remove(mtx_filename)
            except OSError:
                pass

    return index, time.time() - build_start


def run_recall_only_batched(
    dataset_path: str,
    queries_path: str,
    gtruth_path: str,
    metric: str,
    num_node_links: int,
    ef_construction: int,
    ef_search_values: List[int],
    num_build_threads: int,
    num_search_threads: int,
    build_batch_size: int,
    graph_tmp_dir: str,
    save_mtx: str,
    existing_mtx: str,
    num_queries: int,
    k: int,
    search_batch_size: int = 128,
    search_only: bool = False,
    exp_id: str = "",
) -> Dict[str, object]:
    loader = get_data_loader(
        train_dataset_path=dataset_path,
        queries_path=queries_path,
        ground_truth_path=gtruth_path,
    )
    train_data, queries, ground_truth = loader.load_data()

    if num_queries > 0:
        limit = min(num_queries, queries.shape[0], ground_truth.shape[0])
        queries = queries[:limit]
        ground_truth = ground_truth[:limit]

    train_data = train_data.astype(np.float32, copy=False)
    queries = _to_float32_contiguous(queries)
    ground_truth = np.ascontiguousarray(ground_truth, dtype=np.int32)

    dataset_size = train_data.shape[0]
    dim = train_data.shape[1]
    effective_k = min(k, ground_truth.shape[1])
    if effective_k <= 0:
        raise ValueError("Ground truth has no neighbors to evaluate recall.")

    if existing_mtx:
        mtx_path = Path(existing_mtx).expanduser().resolve()
        if not mtx_path.is_file():
            raise FileNotFoundError(f"Existing MTX file not found: {mtx_path}")
        index, build_time_sec = build_flatnav_index_from_graph_file(
            train_data=train_data,
            metric=metric,
            num_node_links=num_node_links,
            mtx_filename=str(mtx_path),
        )
    else:
        index, build_time_sec = build_flatnav_index_from_hnsw_graph(
            train_data=train_data,
            metric=metric,
            num_node_links=num_node_links,
            ef_construction=ef_construction,
            num_build_threads=num_build_threads,
            build_batch_size=build_batch_size,
            graph_tmp_dir=graph_tmp_dir,
            save_mtx=save_mtx,
        )
    logging.info("Build complete in %.2f sec", build_time_sec)

    gc.collect()

    index.set_num_threads(num_search_threads)
    logging.info("Set FlatNav search threads to %d", num_search_threads)

    if search_only and exp_id:
        os.makedirs("/tmp/measurement", exist_ok=True)
        ready_file = f"/tmp/measurement/{exp_id}.ready"
        logging.info("Signaling ready for measurement at %s", ready_file)
        Path(ready_file).touch()
        time.sleep(1)
    else:
        logging.info("Running batched queries without signaling for search-only benchmarking.")

    results: Dict[str, Dict[str, float]] = {}

    for ef_search in ef_search_values:
        logging.info("Running batched queries with ef_search=%d", ef_search)
        start = time.time()
        recalls: List[float] = []
        times_ms: List[float] = []
        failed_batches = 0
        num_queries_total = len(queries)

        for start_idx in range(0, num_queries_total, search_batch_size):
            end_idx = min(start_idx + search_batch_size, num_queries_total)
            batch = queries[start_idx:end_idx]
            try:
                t0 = time.perf_counter()
                distances, labels = index.search(queries=batch, K=effective_k, ef_search=ef_search, num_initializations=100)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                per_query_ms = dt_ms / float(end_idx - start_idx)
                times_ms.extend([per_query_ms] * (end_idx - start_idx))

                batch_labels = labels.astype(np.int32)

                # batch_labels shape should be (batch_size, K)
                for j in range(end_idx - start_idx):
                    recalls.append(compute_recall_at_k(batch_labels[j], ground_truth[start_idx + j], effective_k))
    
            except RuntimeError:
                failed_batches += 1
                continue

        total_sec = time.time() - start
        avg_recall = float(np.mean(recalls)) if recalls else 0.0
        results[str(ef_search)] = {
            "recall": avg_recall,
            "total_time_sec": total_sec,
            "avg_time_ms_per_query": float(np.mean(times_ms)) if times_ms else 0.0,
            "p99_time_ms": float(np.percentile(times_ms, 99)) if times_ms else 0.0,
            "failed_batches": failed_batches,
            "qps": num_queries_total / total_sec if total_sec > 0 else 0.0,
        }
        logging.info(
            "ef_search=%d recall@%d=%.6f total_time=%.2fs failed_batches=%d p99_ms=%.3f qps=%.2f",
            ef_search,
            effective_k,
            avg_recall,
            total_sec,
            failed_batches,
            results[str(ef_search)]["p99_time_ms"],
            num_queries_total / total_sec if total_sec > 0 else 0.0,
        )

    del index
    gc.collect()

    return {
        "dataset": dataset_path,
        "queries": queries_path,
        "ground_truth": gtruth_path,
        "metric": metric,
        "dataset_size": int(dataset_size),
        "dim": int(dim),
        "num_queries": int(len(queries)),
        "num_node_links": num_node_links,
        "ef_construction": ef_construction,
        "num_build_threads": num_build_threads,
        "num_search_threads": num_search_threads,
        "k": effective_k,
        "build_time_sec": build_time_sec,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FlatNav and run batched query recall with p99 timings."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--gtruth", required=True)
    parser.add_argument("--metric", default="l2", choices=["l2", "angular"]) 
    parser.add_argument("--num-node-links", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef-search", nargs="+", type=int, default=[100, 200])
    parser.add_argument("--num-build-threads", type=int, default=1)
    parser.add_argument("--num-search-threads", type=int, default=1)
    parser.add_argument("--build-batch-size", type=int, default=250000)
    parser.add_argument("--graph-tmp-dir", default=str(Path(tempfile.gettempdir()).resolve()))
    parser.add_argument("--existing-mtx", default="")
    parser.add_argument("--save-mtx", default="")
    parser.add_argument("--num-queries", type=int, default=0)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--search-batch-size", type=int, default=128)
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Write to /tmp/measurement/ to signal ready for search benchmarking.",
    )
    parser.add_argument(
        "--exp-id",
        default="",
        help="Experiment ID for the /tmp/measurement/ file if --search-only is used.",
    )
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    summary = run_recall_only_batched(
        dataset_path=args.dataset,
        queries_path=args.queries,
        gtruth_path=args.gtruth,
        metric=args.metric,
        num_node_links=args.num_node_links,
        ef_construction=args.ef_construction,
        ef_search_values=args.ef_search,
        num_build_threads=args.num_build_threads,
        num_search_threads=args.num_search_threads,
        build_batch_size=args.build_batch_size,
        graph_tmp_dir=args.graph_tmp_dir,
        save_mtx=args.save_mtx,
        existing_mtx=args.existing_mtx,
        num_queries=args.num_queries,
        k=args.k,
        search_batch_size=args.search_batch_size,
        search_only=args.search_only,
        exp_id=args.exp_id,
    )

    print(json.dumps(summary, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logging.info("Saved results to %s", output_path)


if __name__ == "__main__":
    main()
