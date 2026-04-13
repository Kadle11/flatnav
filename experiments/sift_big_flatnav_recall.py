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


def build_flatnav_index_from_hnsw_graph(
    train_data: np.ndarray,
    metric: str,
    num_node_links: int,
    ef_construction: int,
    num_build_threads: int,
    build_batch_size: int,
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

    with tempfile.NamedTemporaryFile(suffix=".mtx", delete=False) as tmp:
        mtx_filename = tmp.name
    hnsw_index.save_base_layer_graph(filename=mtx_filename)

    # Release HNSW memory before building FlatNav to reduce memory pressure.
    del hnsw_index
    gc.collect()

    logging.info("Creating FlatNav index from HNSW graph")
    index = flatnav.index.create(
        distance_type=metric,
        index_data_type=FLATNAV_DATA_TYPES["float32"],
        dim=dim,
        dataset_size=dataset_size,
        max_edges_per_node=num_node_links,
        verbose=False,
        collect_stats=False,
    )

    index.allocate_nodes(data=train_data).build_graph_links(mtx_filename)

    try:
        os.remove(mtx_filename)
    except OSError:
        pass

    return index, time.time() - build_start


def run_recall_only(
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
    num_queries: int,
    k: int,
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

    if num_build_threads != 1:
        logging.warning(
            "num_build_threads=%d requested. If you still hit native crashes, retry with --num-build-threads 1.",
            num_build_threads,
        )

    index, build_time_sec = build_flatnav_index_from_hnsw_graph(
        train_data=train_data,
        metric=metric,
        num_node_links=num_node_links,
        ef_construction=ef_construction,
        num_build_threads=num_build_threads,
        build_batch_size=build_batch_size,
    )
    logging.info("Build complete in %.2f sec", build_time_sec)

    index.set_num_threads(num_search_threads)
    results: Dict[str, Dict[str, float]] = {}

    for ef_search in ef_search_values:
        logging.info("Running sequential queries with ef_search=%d", ef_search)
        start = time.time()
        recalls = []
        failed_queries = 0

        for i, query in enumerate(queries):
            try:
                _, neighbors = index.search_single(
                    query=query,
                    ef_search=ef_search,
                    K=effective_k,
                    num_initializations=100,
                )
            except RuntimeError:
                failed_queries += 1
                continue
            recalls.append(compute_recall_at_k(neighbors, ground_truth[i], effective_k))

            if (i + 1) % 1000 == 0:
                logging.info("Processed %d/%d queries", i + 1, len(queries))

        total_sec = time.time() - start
        avg_recall = float(np.mean(recalls)) if recalls else 0.0
        results[str(ef_search)] = {
            "recall": avg_recall,
            "total_time_sec": total_sec,
            "avg_time_ms_per_query": (total_sec * 1000.0) / max(1, len(queries)),
            "failed_queries": failed_queries,
        }
        logging.info(
            "ef_search=%d recall@%d=%.6f total_time=%.2fs failed=%d",
            ef_search,
            effective_k,
            avg_recall,
            total_sec,
            failed_queries,
        )

    # Explicitly release native resources to reduce teardown-related native errors.
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
        description="Build FlatNav and run sequential query recall only."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to training dataset (.fvecs/.npy/.bin supported by data_loader).",
    )
    parser.add_argument("--queries", required=True, help="Path to queries file.")
    parser.add_argument("--gtruth", required=True, help="Path to ground-truth file.")
    parser.add_argument("--metric", default="l2", choices=["l2", "angular"])
    parser.add_argument("--num-node-links", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef-search", nargs="+", type=int, default=[100, 200])
    parser.add_argument("--num-build-threads", type=int, default=1)
    parser.add_argument("--num-search-threads", type=int, default=1)
    parser.add_argument(
        "--build-batch-size",
        type=int,
        default=250000,
        help="Number of vectors per HNSW add batch.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=0,
        help="Optional query limit for faster runs (0 means all).",
    )
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output path to save results as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    summary = run_recall_only(
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
        num_queries=args.num_queries,
        k=args.k,
    )

    print(json.dumps(summary, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logging.info("Saved results to %s", output_path)


if __name__ == "__main__":
    main()
