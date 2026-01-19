"""
Plot Neighborhood Profiling Results

This script visualizes how search navigates through graph neighborhoods,
showing the number of unique neighborhoods accessed in each bin of the 
search sequence, split by hub vs non-hub nodes.

Based on plot_speed_test_results.py template.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging
from typing import List, Tuple, Dict
import argparse

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

BASE_PATH = "/root/data/neighborhood-tests"
METRICS_PATH = "/root/metrics"

os.makedirs(METRICS_PATH, exist_ok=True)

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
    # "glove-100-angular",
    # "nytimes-256-angular",
    # "gist-960-euclidean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot neighborhood profiling results.")
    parser.add_argument(
        "--hubness-percentile",
        type=float,
        default=99.0,
        help="Hubness percentile threshold used during profiling.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=30,
        help="Number of bins to divide the search sequence into.",
    )
    return parser.parse_args()


def bin_visited_nodes(
    query_visited_nodes: List[Tuple[int, bool]], 
    num_bins: int
) -> List[List[Tuple[int, bool]]]:
    """
    Divide visited nodes into bins based on position in search sequence.
    
    Args:
        query_visited_nodes: List of (node_id, is_hub) tuples
        num_bins: Number of bins
        
    Returns:
        List of binned data, each bin containing (node_id, is_hub) tuples
    """
    query_len = len(query_visited_nodes)
    if query_len == 0:
        return [[] for _ in range(num_bins)]
    
    bin_size = query_len // num_bins
    binned_data = []
    
    for i in range(num_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < num_bins - 1 else query_len
        binned_data.append(query_visited_nodes[start:end])
    
    return binned_data


def count_unique_neighborhoods_in_bin(
    bin_data: List[Tuple[int, bool]],
    neighborhood_assignments: np.ndarray
) -> Tuple[int, int]:
    """
    Count unique neighborhoods accessed in a bin, split by hub/non-hub.
    
    Args:
        bin_data: List of (node_id, is_hub) tuples for this bin
        neighborhood_assignments: Array mapping node_id -> neighborhood_id
        
    Returns:
        (num_hub_neighborhoods, num_nonhub_neighborhoods)
    """
    hub_neighborhoods = set()
    nonhub_neighborhoods = set()
    
    for node_id, is_hub in bin_data:
        if node_id < len(neighborhood_assignments):
            neighborhood_id = neighborhood_assignments[node_id]
            if is_hub:
                hub_neighborhoods.add(neighborhood_id)
            else:
                nonhub_neighborhoods.add(neighborhood_id)
    
    return len(hub_neighborhoods), len(nonhub_neighborhoods)


def plot_neighborhood_access(
    dataset_name: str,
    hub_neighborhoods_per_bin: np.ndarray,
    nonhub_neighborhoods_per_bin: np.ndarray,
    num_bins: int,
    metadata: Dict,
) -> None:
    """
    Create a stacked bar chart showing neighborhood access patterns.
    
    Args:
        dataset_name: Name of the dataset
        hub_neighborhoods_per_bin: Average unique hub neighborhoods per bin
        nonhub_neighborhoods_per_bin: Average unique non-hub neighborhoods per bin
        num_bins: Number of bins
        metadata: Dataset metadata dict
    """
    bins = np.arange(1, num_bins + 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Stacked bar chart
    ax.bar(bins, hub_neighborhoods_per_bin, label="Hub Node Neighborhoods", color="royalblue")
    ax.bar(
        bins,
        nonhub_neighborhoods_per_bin,
        bottom=hub_neighborhoods_per_bin,
        label="Non-Hub Node Neighborhoods",
        color="coral",
    )

    # Labels and titles
    ax.set_xlabel("Search Progress (Binned)", fontsize=12)
    ax.set_ylabel("Avg. Unique Neighborhoods Accessed", fontsize=12)
    
    title = f"Neighborhood Access During Search\n"
    title += f"Dataset: {dataset_name} | "
    title += f"K={metadata.get('num_neighborhoods', 'N/A')} neighborhoods | "
    title += f"Hub percentile: {metadata.get('hubness_percentile', 'N/A')}%"
    ax.set_title(title, fontsize=11)

    # X-axis: show only some tick labels to avoid crowding
    tick_positions = [1, num_bins // 4, num_bins // 2, 3 * num_bins // 4, num_bins]
    tick_labels = ["Start", "25%", "50%", "75%", "End"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Save figure
    fig_name = os.path.join(METRICS_PATH, f"{dataset_name}_neighborhood_access_plot.png")
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)
    plt.close()
    logging.info(f"Saved plot to {fig_name}")


def plot_total_neighborhoods_over_search(
    dataset_name: str,
    hub_neighborhoods_cumulative: np.ndarray,
    nonhub_neighborhoods_cumulative: np.ndarray,
    num_bins: int,
    metadata: Dict,
) -> None:
    """
    Create a line plot showing cumulative unique neighborhoods discovered.
    
    Args:
        dataset_name: Name of the dataset
        hub_neighborhoods_cumulative: Cumulative hub neighborhoods at each bin
        nonhub_neighborhoods_cumulative: Cumulative non-hub neighborhoods at each bin
        num_bins: Number of bins
        metadata: Dataset metadata dict
    """
    bins = np.arange(1, num_bins + 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    total_cumulative = hub_neighborhoods_cumulative + nonhub_neighborhoods_cumulative
    
    ax.plot(bins, total_cumulative, label="Total", color="black", linewidth=2)
    ax.plot(bins, hub_neighborhoods_cumulative, label="Hub Neighborhoods", 
            color="royalblue", linewidth=1.5, linestyle="--")
    ax.plot(bins, nonhub_neighborhoods_cumulative, label="Non-Hub Neighborhoods", 
            color="coral", linewidth=1.5, linestyle="--")

    ax.set_xlabel("Search Progress (Binned)", fontsize=12)
    ax.set_ylabel("Cumulative Unique Neighborhoods", fontsize=12)
    
    title = f"Cumulative Neighborhoods Discovered During Search\n"
    title += f"Dataset: {dataset_name} | Total neighborhoods: {metadata.get('num_neighborhoods', 'N/A')}"
    ax.set_title(title, fontsize=11)

    tick_positions = [1, num_bins // 4, num_bins // 2, 3 * num_bins // 4, num_bins]
    tick_labels = ["Start", "25%", "50%", "75%", "End"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    fig_name = os.path.join(METRICS_PATH, f"{dataset_name}_neighborhood_cumulative_plot.png")
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)
    plt.close()
    logging.info(f"Saved cumulative plot to {fig_name}")


def process_dataset(dataset_name: str, hubness_percentile: float, num_bins: int) -> bool:
    """
    Process and plot results for a single dataset.
    
    Args:
        dataset_name: Name of the dataset
        hubness_percentile: Hubness percentile threshold
        num_bins: Number of bins
        
    Returns:
        True if successful, False otherwise
    """
    # Load visited nodes data
    visited_path = os.path.join(
        BASE_PATH, f"{dataset_name}.{hubness_percentile}.visited_nodes.npy"
    )
    assignments_path = os.path.join(
        BASE_PATH, f"{dataset_name}.neighborhood_assignments.npy"
    )
    metadata_path = os.path.join(
        BASE_PATH, f"{dataset_name}.{hubness_percentile}.metadata.json"
    )

    if not os.path.exists(visited_path):
        logging.warning(f"Skipping {dataset_name}: visited nodes file not found at {visited_path}")
        return False

    if not os.path.exists(assignments_path):
        logging.warning(f"Skipping {dataset_name}: neighborhood assignments not found at {assignments_path}")
        return False

    logging.info(f"Processing {dataset_name}...")

    # Load data
    visited_nodes_data = np.load(visited_path, allow_pickle=True)
    neighborhood_assignments = np.load(assignments_path)
    
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    num_queries = len(visited_nodes_data)
    logging.info(f"Loaded {num_queries} queries for {dataset_name}")

    # Initialize accumulators
    hub_neighborhoods_per_bin = np.zeros(num_bins)
    nonhub_neighborhoods_per_bin = np.zeros(num_bins)
    
    # For cumulative tracking (average across queries)
    hub_neighborhoods_cumulative = np.zeros(num_bins)
    nonhub_neighborhoods_cumulative = np.zeros(num_bins)

    # Process each query
    for query_idx, query_visited_nodes in enumerate(visited_nodes_data):
        # Convert to list of tuples if needed
        if isinstance(query_visited_nodes, np.ndarray):
            query_visited_nodes = [tuple(x) for x in query_visited_nodes]
        
        if len(query_visited_nodes) == 0:
            continue

        # Bin the visited nodes
        binned_data = bin_visited_nodes(query_visited_nodes, num_bins)

        # Track unique neighborhoods for this query
        query_hub_seen = set()
        query_nonhub_seen = set()

        for bin_idx, bin_data in enumerate(binned_data):
            # Count unique neighborhoods in this bin
            hub_count, nonhub_count = count_unique_neighborhoods_in_bin(
                bin_data, neighborhood_assignments
            )
            hub_neighborhoods_per_bin[bin_idx] += hub_count
            nonhub_neighborhoods_per_bin[bin_idx] += nonhub_count

            # Update cumulative tracking for this query
            for node_id, is_hub in bin_data:
                if node_id < len(neighborhood_assignments):
                    neighborhood_id = neighborhood_assignments[node_id]
                    if is_hub:
                        query_hub_seen.add(neighborhood_id)
                    else:
                        query_nonhub_seen.add(neighborhood_id)
            
            hub_neighborhoods_cumulative[bin_idx] += len(query_hub_seen)
            nonhub_neighborhoods_cumulative[bin_idx] += len(query_nonhub_seen)

    # Average over queries
    hub_neighborhoods_per_bin /= num_queries
    nonhub_neighborhoods_per_bin /= num_queries
    hub_neighborhoods_cumulative /= num_queries
    nonhub_neighborhoods_cumulative /= num_queries

    # Generate plots
    plot_neighborhood_access(
        dataset_name,
        hub_neighborhoods_per_bin,
        nonhub_neighborhoods_per_bin,
        num_bins,
        metadata,
    )

    plot_total_neighborhoods_over_search(
        dataset_name,
        hub_neighborhoods_cumulative,
        nonhub_neighborhoods_cumulative,
        num_bins,
        metadata,
    )

    return True


def main():
    args = parse_args()
    num_bins = args.num_bins
    hubness_percentile = args.hubness_percentile

    logging.info(f"Plotting neighborhood profiling results")
    logging.info(f"  Hubness percentile: {hubness_percentile}")
    logging.info(f"  Number of bins: {num_bins}")

    successful = 0
    for dataset in SYNTHETIC_DATASETS + ANN_DATASETS:
        if process_dataset(dataset, hubness_percentile, num_bins):
            successful += 1

    logging.info(f"\nCompleted. Processed {successful} datasets successfully.")


if __name__ == "__main__":
    main()
