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


def _compare_query_results(
    serial_query_results: List[Dict[str, object]],
    threaded_query_results: List[Dict[str, object]],
    k: int,
    ground_truth: np.ndarray,
) -> List[Dict[str, object]]:
    comparisons: List[Dict[str, object]] = []

    for serial_entry, threaded_entry in zip(serial_query_results, threaded_query_results):
        query_index = int(serial_entry["query_index"])
        serial_status = str(serial_entry["status"])
        threaded_status = str(threaded_entry["status"])

        if serial_status != "ok" or threaded_status != "ok":
            comparisons.append(
                {
                    "query_index": query_index,
                    "status": "skipped",
                    "serial_status": serial_status,
                    "threaded_status": threaded_status,
                    "label_differences": [],
                    "distance_differences": [],
                }
            )
            continue

        serial_labels = np.asarray(serial_entry["labels"], dtype=np.int64)
        threaded_labels = np.asarray(threaded_entry["labels"], dtype=np.int64)
        serial_distances = np.asarray(serial_entry["distances"], dtype=np.float64)
        threaded_distances = np.asarray(threaded_entry["distances"], dtype=np.float64)
        serial_label_set = set(serial_labels[:k].tolist())
        threaded_label_set = set(threaded_labels[:k].tolist())
        label_set_intersection = serial_label_set.intersection(threaded_label_set)
        label_set_only_serial = sorted(serial_label_set.difference(threaded_label_set))
        label_set_only_threaded = sorted(threaded_label_set.difference(serial_label_set))

        # Determine which non-overlapping labels contribute positively to recall
        gt_set = set(ground_truth[query_index][:k].tolist()) if ground_truth is not None else set()
        label_set_only_serial_positive = [l for l in label_set_only_serial if l in gt_set]
        label_set_only_serial_negative = [l for l in label_set_only_serial if l not in gt_set]
        label_set_only_threaded_positive = [l for l in label_set_only_threaded if l in gt_set]
        label_set_only_threaded_negative = [l for l in label_set_only_threaded if l not in gt_set]
        # Among overlapping labels, see which contribute to recall
        label_set_overlap_positive = sorted([l for l in label_set_intersection if l in gt_set])
        label_set_overlap_negative = sorted([l for l in label_set_intersection if l not in gt_set])
        label_set_overlap_positive_count = int(len(label_set_overlap_positive))
        label_set_overlap_positive_fraction = float(label_set_overlap_positive_count) / float(max(1, k))

        compare_k = min(
            k,
            serial_labels.shape[0],
            threaded_labels.shape[0],
            serial_distances.shape[0],
            threaded_distances.shape[0],
        )

        label_differences = [
            {
                "rank": int(rank),
                "serial": int(serial_labels[rank]),
                "threaded": int(threaded_labels[rank]),
                "same": bool(serial_labels[rank] == threaded_labels[rank]),
            }
            for rank in range(compare_k)
        ]
        distance_differences = [
            {
                "rank": int(rank),
                "serial": float(serial_distances[rank]),
                "threaded": float(threaded_distances[rank]),
                "difference": float(threaded_distances[rank] - serial_distances[rank]),
            }
            for rank in range(compare_k)
        ]

        comparisons.append(
            {
                "query_index": query_index,
                "status": "ok",
                "label_differences": label_differences,
                "distance_differences": distance_differences,
                "label_set_overlap_count": int(len(label_set_intersection)),
                "label_set_overlap_fraction": float(len(label_set_intersection) / float(max(1, k))),
                "label_set_overlap_positive": label_set_overlap_positive,
                "label_set_overlap_negative": label_set_overlap_negative,
                "label_set_overlap_positive_count": label_set_overlap_positive_count,
                "label_set_overlap_positive_fraction": label_set_overlap_positive_fraction,
                "label_set_only_serial": label_set_only_serial,
                "label_set_only_serial_positive": label_set_only_serial_positive,
                "label_set_only_serial_negative": label_set_only_serial_negative,
                "label_set_only_threaded": label_set_only_threaded,
                "label_set_only_threaded_positive": label_set_only_threaded_positive,
                "label_set_only_threaded_negative": label_set_only_threaded_negative,
                "label_mismatch_count": int(sum(not item["same"] for item in label_differences)),
                "distance_max_abs_diff": float(
                    max(
                        (abs(item["difference"]) for item in distance_differences),
                        default=0.0,
                    )
                ),
            }
        )

    return comparisons


def _run_search_mode(
    index: Any,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    ef_search: int,
    k: int,
    num_threads: int,
    mode_name: str,
) -> Dict[str, object]:
    index.set_num_threads(max(1, num_threads))
    logging.info("Set FlatNav search threads to %d for %s run", num_threads, mode_name)

    query_results: List[Dict[str, object]] = []
    recalls: List[float] = []
    failed_queries = 0
    start = time.perf_counter()

    if num_threads <= 1:
        for query_index, query in enumerate(queries):
            try:
                query_start = time.perf_counter()
                distances, labels = index.search_single(
                    query=query,
                    ef_search=ef_search,
                    K=k,
                    num_initializations=100,
                )
                query_time_ms = (time.perf_counter() - query_start) * 1000.0
            except RuntimeError as exc:
                failed_queries += 1
                query_results.append(
                    {
                        "query_index": int(query_index),
                        "status": "failed",
                        "error": str(exc),
                        "labels": [],
                        "distances": [],
                        "recall": None,
                        "query_time_ms": None,
                    }
                )
                continue

            label_array = np.asarray(labels, dtype=np.int64)
            distance_array = np.asarray(distances, dtype=np.float64)
            recall = compute_recall_at_k(label_array, ground_truth[query_index], k)
            recalls.append(recall)

            query_results.append(
                {
                    "query_index": int(query_index),
                    "status": "ok",
                    "recall": float(recall),
                    "query_time_ms": float(query_time_ms),
                    "labels": label_array[:k].astype(np.int64).tolist(),
                    "distances": distance_array[:k].astype(float).tolist(),
                }
            )
    else:
        try:
            query_start = time.perf_counter()
            labels, distances = index.search(
                queries=queries,
                K=k,
                ef_search=ef_search,
                num_initializations=100,
            )
            total_query_time_ms = (time.perf_counter() - query_start) * 1000.0
        except RuntimeError as exc:
            failed_queries = len(queries)
            for query_index in range(len(queries)):
                query_results.append(
                    {
                        "query_index": int(query_index),
                        "status": "failed",
                        "error": str(exc),
                        "labels": [],
                        "distances": [],
                        "recall": None,
                        "query_time_ms": None,
                    }
                )
        else:
            label_matrix = np.asarray(labels, dtype=np.int64)
            distance_matrix = np.asarray(distances, dtype=np.float64)
            per_query_time_ms = total_query_time_ms / float(max(1, len(queries)))

            for query_index in range(len(queries)):
                label_array = label_matrix[query_index]
                distance_array = distance_matrix[query_index]
                recall = compute_recall_at_k(label_array, ground_truth[query_index], k)
                recalls.append(recall)

                query_results.append(
                    {
                        "query_index": int(query_index),
                        "status": "ok",
                        "recall": float(recall),
                        "query_time_ms": float(per_query_time_ms),
                        "labels": label_array[:k].astype(np.int64).tolist(),
                        "distances": distance_array[:k].astype(float).tolist(),
                    }
                )

    total_sec = time.perf_counter() - start
    avg_recall = float(np.mean(recalls)) if recalls else 0.0

    return {
        "mode": mode_name,
        "num_threads": int(max(1, num_threads)),
        "recall": avg_recall,
        "total_time_sec": total_sec,
        "avg_time_ms_per_query": (total_sec * 1000.0) / max(1, len(queries)),
        "failed_queries": failed_queries,
        "query_results": query_results,
    }


def run_recall_thread_comparison(
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
            graph_tmp_dir=graph_tmp_dir,
            save_mtx=save_mtx,
        )

    logging.info("Build complete in %.2f sec", build_time_sec)
    gc.collect()

    results: Dict[str, Dict[str, object]] = {}

    for ef_search in ef_search_values:
        logging.info("Running serial comparison pass with ef_search=%d", ef_search)
        serial_results = _run_search_mode(
            index=index,
            queries=queries,
            ground_truth=ground_truth,
            ef_search=ef_search,
            k=effective_k,
            num_threads=1,
            mode_name="serial",
        )

        logging.info(
            "Running threaded comparison pass with ef_search=%d and %d thread(s)",
            ef_search,
            num_search_threads,
        )
        threaded_results = _run_search_mode(
            index=index,
            queries=queries,
            ground_truth=ground_truth,
            ef_search=ef_search,
            k=effective_k,
            num_threads=num_search_threads,
            mode_name="threaded",
        )

        results[str(ef_search)] = {
            "serial": serial_results,
            "threaded": threaded_results,
            "comparison": {
                "query_differences": _compare_query_results(
                    serial_query_results=serial_results["query_results"],
                    threaded_query_results=threaded_results["query_results"],
                    k=effective_k,
                    ground_truth=ground_truth,
                )
            },
        }

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
        description="Build FlatNav and compare serial and threaded search recall on the first N queries."
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
        "--graph-tmp-dir",
        default=str(Path(tempfile.gettempdir()).resolve()),
        help="Directory where temporary .mtx graph is written. Use a fast local SSD path.",
    )
    parser.add_argument(
        "--existing-mtx",
        default="",
        help="Optional path to a prebuilt HNSW base-layer .mtx graph. If set, HNSW build/dump is skipped.",
    )
    parser.add_argument(
        "--save-mtx",
        default="",
        help="Optional path to persist generated HNSW base-layer .mtx for reuse in future runs.",
    )
    parser.add_argument(
        "-n",
        "--num-queries",
        type=int,
        default=10,
        help="Only evaluate the first N queries (0 means all).",
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

    summary = run_recall_thread_comparison(
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
    )

    print(json.dumps(summary, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logging.info("Saved results to %s", output_path)


if __name__ == "__main__":
    main()