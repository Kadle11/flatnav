"""
Neighborhood Profiling for Search Navigation

This script profiles how graph search navigates different neighborhoods during search.
It uses K-means clustering to define neighborhoods and tracks which neighborhoods
are visited during search, along with hub/non-hub classification.

Based on hubness_speed_test.py template.
"""

import logging
import json
import numpy as np
import os
from typing import List, Tuple, Optional
import time
import argparse
from sklearn.cluster import KMeans
from experiments.run_benchmark import train_index

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neighborhood profiling for search navigation.")
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
        "--num-neighborhoods",
        type=int,
        default=None,
        help="Number of neighborhoods (clusters). Default: sqrt(dataset_size), bounded [10, 500].",
    )
    args = parser.parse_args()
    return args


# Paths - should be persistent volume mounts
DISTRIBUTIONS_SAVE_PATH = "/root/node-access-distributions"
NEIGHBORHOOD_TESTS_SAVE_PATH = "/root/data/neighborhood-tests"
DATASETS_BASE_PATH = "/root/data/hubness/data"
METRICS_DIR = "/root/metrics"

os.makedirs(NEIGHBORHOOD_TESTS_SAVE_PATH, exist_ok=True)

SYNTHETIC_DATASETS = [
    "normal-16-angular",
    "normal-16-euclidean",
    "normal-32-angular",
    "normal-32-euclidean",
    "normal-64-angular",
    "normal-64-euclidean",
    "normal-128-angular",
    "normal-128-euclidean",
    # "normal-1536-angular",
    # "normal-1536-euclidean",
]

ANN_DATASETS = [
    "glove-100-angular",
    "nytimes-256-angular",
    "gist-960-euclidean",
]


def get_num_neighborhoods(dataset_size: int, user_override: Optional[int] = None) -> int:
    """
    Determine the number of neighborhoods for K-means clustering.
    
    Args:
        dataset_size: Number of nodes in the dataset
        user_override: Optional user-specified value
        
    Returns:
        Number of neighborhoods (clusters)
    """
    if user_override is not None:
        return user_override
    # Auto: sqrt(N) with bounds
    return max(10, min(int(np.sqrt(dataset_size)), 500))


def compute_neighborhood_assignments(
    train_dataset: np.ndarray, 
    num_neighborhoods: int
) -> np.ndarray:
    """
    Assign each node to a neighborhood using K-means clustering.
    
    Args:
        train_dataset: Array of shape (num_nodes, dim) with node vectors
        num_neighborhoods: Number of clusters/neighborhoods
        
    Returns:
        Array of shape (num_nodes,) with neighborhood IDs (0 to num_neighborhoods-1)
    """
    logger.info(f"Computing K-means clustering with {num_neighborhoods} neighborhoods...")
    kmeans = KMeans(n_clusters=num_neighborhoods, random_state=42, n_init=10)
    neighborhood_assignments = kmeans.fit_predict(train_dataset)
    logger.info(f"Clustering complete. Cluster sizes: min={np.bincount(neighborhood_assignments).min()}, "
                f"max={np.bincount(neighborhood_assignments).max()}, "
                f"mean={np.bincount(neighborhood_assignments).mean():.1f}")
    return neighborhood_assignments


def select_hub_nodes(percentile: float, node_access_counts_path: str) -> List[int]:
    """
    Select the nodes that fall above the given percentile of access counts.
    
    Args:
        percentile: The percentile threshold to consider
        node_access_counts_path: Path to JSON file with node access counts
        
    Returns:
        List of node IDs classified as hubs
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
    logger.info(f"Selected {num_hub_nodes} hub nodes ({hub_nodes_percentage:.2f}%)")

    return selected_nodes


def run_neighborhood_profiling(
    train_dataset: np.ndarray,
    queries: np.ndarray,
    distance_type: str,
    dim: int,
    dataset_size: int,
    max_edges_per_node: int,
    ef_construction: int,
    ef_search: int,
    hub_nodes: List[int],
    dataset_name: str,
) -> List[List[Tuple[int, bool]]]:
    """
    Run search queries and collect visited node IDs with hub flags.
    
    Args:
        train_dataset: Training vectors
        queries: Query vectors
        distance_type: "angular" or "l2"
        dim: Vector dimension
        dataset_size: Number of vectors
        max_edges_per_node: Graph connectivity parameter
        ef_construction: Construction-time search width
        ef_search: Query-time search width
        hub_nodes: List of hub node IDs
        dataset_name: Name of the dataset
        
    Returns:
        List of visited node sequences, each containing (node_id, is_hub) tuples
    """
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

    # Set hub nodes in the index
    flatnav_index.set_hub_nodes(hub_nodes=hub_nodes)
    logger.info(f"Set {flatnav_index.count_hub_nodes()} hub nodes in index")

    # Clear any previous visited nodes data
    flatnav_index.clear_visited_nodes_with_ids()

    # Run search with single thread to ensure consistent ordering
    flatnav_index.set_num_threads(1)

    for i, query in enumerate(queries):
        if (i + 1) % 100 == 0:
            logger.debug(f"Processing query {i + 1}/{len(queries)}")
        # Use the new search method that tracks node IDs
        _, _ = flatnav_index.search_single_with_node_ids(
            query=query,
            ef_search=ef_search,
            K=100,
            num_initializations=100,
        )

    # Retrieve the list of visited nodes with IDs and hub flags
    # This is List[List[Tuple[node_id, is_hub]]]
    visited_nodes_with_ids = flatnav_index.get_visited_nodes_with_ids()
    return visited_nodes_with_ids


def main() -> None:
    args = parse_args()
    percentile_threshold = args.hubness_percentile_threshold
    logger.info(
        f"Running neighborhood profiling with hubness percentile threshold {percentile_threshold}"
    )

    for dataset_name in SYNTHETIC_DATASETS + ANN_DATASETS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'='*60}")

        # Load hub node information
        node_access_counts_path = os.path.join(
            DISTRIBUTIONS_SAVE_PATH, f"{dataset_name}_node_access_counts.json"
        )

        try:
            hub_nodes: List[int] = select_hub_nodes(
                percentile_threshold, node_access_counts_path
            )
        except FileNotFoundError as e:
            logger.warning(f"Skipping {dataset_name}: {e}")
            continue

        # Load dataset
        train_dataset_path = os.path.join(
            DATASETS_BASE_PATH, dataset_name, f"{dataset_name}.train.npy"
        )
        queries_path = os.path.join(
            DATASETS_BASE_PATH, dataset_name, f"{dataset_name}.test.npy"
        )

        if not os.path.exists(train_dataset_path):
            logger.warning(f"Skipping {dataset_name}: train dataset not found")
            continue

        train_dataset = np.load(train_dataset_path)
        queries = np.load(queries_path)
        distance_type = "angular" if "angular" in dataset_name else "l2"
        dataset_size, dim = train_dataset.shape

        # Compute neighborhood assignments
        num_neighborhoods = get_num_neighborhoods(dataset_size, args.num_neighborhoods)
        neighborhood_assignments = compute_neighborhood_assignments(
            train_dataset, num_neighborhoods
        )

        # Run profiling
        logger.info(f"Running neighborhood profiling for {dataset_name}")
        start = time.time()
        visited_nodes_with_ids = run_neighborhood_profiling(
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
        logger.info(f"Profiling for {dataset_name} completed in {end - start:.2f} seconds.")

        # Save results
        # 1. Save visited nodes with IDs
        visited_save_path = os.path.join(
            NEIGHBORHOOD_TESTS_SAVE_PATH,
            f"{dataset_name}.{percentile_threshold}.visited_nodes.npy",
        )
        np.save(visited_save_path, np.array(visited_nodes_with_ids, dtype=object))
        logger.info(f"Saved visited nodes to {visited_save_path}")

        # 2. Save neighborhood assignments
        assignments_save_path = os.path.join(
            NEIGHBORHOOD_TESTS_SAVE_PATH,
            f"{dataset_name}.neighborhood_assignments.npy",
        )
        np.save(assignments_save_path, neighborhood_assignments)
        logger.info(f"Saved neighborhood assignments to {assignments_save_path}")

        # 3. Save metadata
        metadata = {
            "dataset_name": dataset_name,
            "dataset_size": dataset_size,
            "dim": dim,
            "num_neighborhoods": num_neighborhoods,
            "num_hub_nodes": len(hub_nodes),
            "hubness_percentile": percentile_threshold,
            "ef_construction": args.ef_construction,
            "ef_search": args.ef_search,
            "num_queries": len(queries),
        }
        metadata_save_path = os.path.join(
            NEIGHBORHOOD_TESTS_SAVE_PATH,
            f"{dataset_name}.{percentile_threshold}.metadata.json",
        )
        with open(metadata_save_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_save_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
