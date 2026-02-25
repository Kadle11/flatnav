import logging
import json
import numpy as np
import os
from typing import List, Tuple
import time
import argparse
from experiments.run_benchmark import train_index

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hubness speed tests.")
    parser.add_argument(
        "--hubness-percentile-threshold",
        type=float,
        default=99,
        help="The percentile threshold to consider for hub nodes.",
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=100,
        help="Ef-construction parameter for the index.",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=100,
        help="Ef-search parameter for the index.",
    )
    parser.add_argument(
        "--hub-selection-method",
        type=str,
        choices=["access-count", "degree"],
        default="access-count",
        help="Method for hub classification: 'access-count' (query-dependent) or 'degree' (static, query-independent)",
    )
    args = parser.parse_args()
    return args


# This should be a persistent volume mount.
DISTRIBUTIONS_SAVE_PATH = "/root/node-access-distributions"
SPEED_TESTS_SAVE_PATH = "/root/data/speed-tests"
DATASETS_BASE_PATH = "/root/data"
METRICS_DIR = "/root/metrics"

os.makedirs(SPEED_TESTS_SAVE_PATH, exist_ok=True)

SYNTHETIC_DATASETS = [
    # "normal-16-angular",
    # "normal-16-euclidean",
    # "normal-32-angular",
    # "normal-32-euclidean",
    # "normal-64-angular",
    "normal-64-euclidean",
    # "normal-128-angular",
    "normal-128-euclidean",
    # "normal-256-angular",
    "normal-256-euclidean",
    "normal-512-euclidean",
    # "normal-1024-angular",
    # "normal-1024-euclidean",
    # "normal-1536-angular",
    # "normal-1536-euclidean",
]

ANN_DATASETS = [
    # "glove-100-angular",
    # "nytimes-256-angular",
    "gist-960-euclidean",
    # "yandex-deep-10m-euclidean",
    # "spacev-10m-euclidean",
]


def select_hub_nodes(percentile: float, node_access_counts_path: str) -> List[int]:
    """
    Select the nodes that fall above the given percentile based on access counts.
    :param percentile: The percentile to consider.
    :param node_access_counts_path: Path to JSON file with node access counts.
    :return selected_nodes: The subset of nodes that fall above the given percentile.
    """
    if not os.path.exists(node_access_counts_path):
        raise FileNotFoundError(
            f"Node access counts not found at {node_access_counts_path}"
        )

    with open(node_access_counts_path, "r") as f:
        data = json.load(f)

    data = {int(k): int(v) for k, v in data.items()}
    node_access_counts = list(data.values())

    threshold = np.percentile(node_access_counts, percentile)
    selected_nodes = [node for node, count in data.items() if count >= threshold]
    num_hub_nodes = len(selected_nodes)
    hub_nodes_percentage = (num_hub_nodes / len(data)) * 100
    logger.info(f"Selected {num_hub_nodes} hub nodes ({hub_nodes_percentage}%)")

    return selected_nodes


def select_hub_nodes_by_degree(index, percentile: float = 90) -> List[int]:
    """
    Select hub nodes based on total graph degree (in-degree + out-degree) percentile.
    This is query-INDEPENDENT - degree is a property of the graph structure only.
    
    Args:
        index: FlatNav index
        percentile: Percentile threshold (default 90 = top 10% highest degree nodes)
    
    Returns:
        List of node IDs that are classified as hubs by degree
    """
    # Get the outdegree table: list of neighbor lists for each node
    outdegree_table = index.get_graph_outdegree_table()
    num_nodes = len(outdegree_table)
    
    # Calculate out-degrees
    out_degrees = {node_id: len(neighbors) for node_id, neighbors in enumerate(outdegree_table)}
    
    # Calculate in-degrees by counting how many times each node appears as a neighbor
    in_degrees = {node_id: 0 for node_id in range(num_nodes)}
    for node_id, neighbors in enumerate(outdegree_table):
        for neighbor in neighbors:
            in_degrees[neighbor] += 1
    
    # Calculate total degree (in + out) for each node
    total_degrees = {
        node_id: in_degrees[node_id] + out_degrees[node_id]
        for node_id in range(num_nodes)
    }
    
    # Find threshold total degree
    degree_values = list(total_degrees.values())
    threshold = np.percentile(degree_values, percentile)
    
    # Select hub nodes with total degree >= threshold
    hub_nodes = [
        node_id for node_id, degree in total_degrees.items()
        if degree >= threshold
    ]
    
    # Compute statistics
    avg_in_degree = sum(in_degrees.values()) / num_nodes if num_nodes > 0 else 0
    avg_out_degree = sum(out_degrees.values()) / num_nodes if num_nodes > 0 else 0
    avg_total_degree = sum(total_degrees.values()) / num_nodes if num_nodes > 0 else 0
    
    logger.info(f"Selected {len(hub_nodes)} hub nodes by total degree (top {100-percentile}% of {num_nodes} nodes)")
    logger.info(f"  Degree threshold: {threshold:.1f}")
    logger.info(f"  In-degree  - Min: {min(in_degrees.values())}, Max: {max(in_degrees.values())}, Avg: {avg_in_degree:.1f}")
    logger.info(f"  Out-degree - Min: {min(out_degrees.values())}, Max: {max(out_degrees.values())}, Avg: {avg_out_degree:.1f}")
    logger.info(f"  Total deg  - Min: {min(degree_values)}, Max: {max(degree_values)}, Avg: {avg_total_degree:.1f}")
    
    return hub_nodes


def run_test(
    train_dataset: np.ndarray,
    queries: np.ndarray,
    distance_type: str,
    dim: int,
    dataset_size: int,
    max_edges_per_node: int,
    ef_construction: int,
    ef_search: int,
    hub_nodes: list[int],
    dataset_name: str,
) -> List[List[bool]]:
    flatnav_index = train_index(
        train_dataset=train_dataset,
        distance_type=distance_type,
        dim=dim,
        dataset_size=dataset_size,
        max_edges_per_node=max_edges_per_node,
        ef_construction=ef_construction,
        index_type="flatnav",
        data_type="float32",
        use_hnsw_base_layer=True,
        hnsw_base_layer_filename=f"{dataset_name}_hnsw_base_layer.mtx",
        num_build_threads=16,
    )

    # Now set the hub nodes in the index
    flatnav_index.set_hub_nodes(hub_nodes=hub_nodes)

    # Now run search to record the sequence of visited nodes for each query
    flatnav_index.set_num_threads(1)

    for query in queries:
        _, indices = flatnav_index.search_single(
            query=query,
            ef_search=ef_search,
            K=100,
            num_initializations=100,
        )

    # Retrieve the list of visited nodes flag. This is a List[List[bool]]
    # where each element is a list of booleans indicating whether the node
    # that was visited at that position was a hub node or not.
    visited_nodes_flags = flatnav_index.get_visited_nodes_sequence()
    return visited_nodes_flags


def main() -> None:

    args = parse_args()
    percentile_threshold = args.hubness_percentile_threshold
    logger.info(
        f"Running speed tests for hubness percentile threshold {percentile_threshold}"
    )

    for dataset_name in SYNTHETIC_DATASETS + ANN_DATASETS:
        train_dataset_path = os.path.join(
            DATASETS_BASE_PATH, dataset_name, f"{dataset_name}.train.npy"
        )
        queries_path = os.path.join(
            DATASETS_BASE_PATH, dataset_name, f"{dataset_name}.test.npy"
        )
        train_dataset = np.load(train_dataset_path)
        queries = np.load(queries_path)
        distance_type = "angular" if "angular" in dataset_name else "l2"
        dataset_size, dim = train_dataset.shape

        # Select hub nodes based on chosen method
        if args.hub_selection_method == "degree":
            # Build index first for degree-based selection
            logger.info(f"Building index for {dataset_name} to compute node degrees...")
            flatnav_index = train_index(
                train_dataset=train_dataset,
                distance_type=distance_type,
                dim=dim,
                dataset_size=dataset_size,
                max_edges_per_node=32,
                ef_construction=args.ef_construction,
                index_type="flatnav",
                data_type="float32",
                use_hnsw_base_layer=True,
                hnsw_base_layer_filename=f"{dataset_name}_hnsw_base_layer.mtx",
                num_build_threads=16,
            )
            hub_nodes: list[int] = select_hub_nodes_by_degree(
                flatnav_index, percentile=100 - (100 - args.hubness_percentile_threshold)
            )
        else:
            # Access-count based selection
            node_access_counts_path = os.path.join(
                DISTRIBUTIONS_SAVE_PATH, f"{dataset_name}_node_access_counts.json"
            )
            hub_nodes: list[int] = select_hub_nodes(
                args.hubness_percentile_threshold, node_access_counts_path
            )

        logger.info(f"Running test for {dataset_name}")
        start = time.time()
        visited_nodes_flags = run_test(
            train_dataset=train_dataset,
            queries=queries,
            distance_type=distance_type,
            dim=dim,
            dataset_size=dataset_size,
            max_edges_per_node=32,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
            hub_nodes=hub_nodes,
            dataset_name=dataset_name,
        )
        end = time.time()
        logger.info(f"Test for {dataset_name} completed in {end - start} seconds.")

        # Save this result as a numpy file with name SPEED_TESTS_SAVE_PATH/dataset_name.npy

        save_path = os.path.join(
            SPEED_TESTS_SAVE_PATH,
            f"{dataset_name}.{args.hubness_percentile_threshold}.npy",
        )
        np.save(save_path, np.array(visited_nodes_flags, dtype=object))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
