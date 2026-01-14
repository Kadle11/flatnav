#!/usr/bin/env python3
"""
Profile Hub vs Non-Hub Vector Search Performance

This script measures compute and memory bandwidth utilization separately for
hub and non-hub node distance computations during vector search.

Usage:
    python profile_hub_compute.py --dataset glove-100-angular --ef-search 200 --k 100

Requires:
    - Linux `perf` tool for hardware counter measurements
    - FlatNav with hub profiling instrumentation
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import flatnav.index
import hnswlib
from utils import get_metric_from_dataset_name, load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Search modes (must match C++ enum)
SEARCH_MODE_NORMAL = 0
SEARCH_MODE_HUB_ONLY = 1
SEARCH_MODE_NONHUB_ONLY = 2

# Default dataset path
ROOT_DATASET_PATH = os.getenv("ROOT_DATASET_PATH", "/root/data/hubness/data")

# Output directory for profiling results
PROFILE_OUTPUT_PATH = os.getenv("PROFILE_OUTPUT_PATH", "/root/metrics/hub_profiling")


@dataclass
class PerfMetrics:
    """Container for perf hardware counter metrics."""
    cycles: int = 0
    instructions: int = 0
    cache_references: int = 0
    cache_misses: int = 0
    l1_dcache_loads: int = 0
    l1_dcache_load_misses: int = 0
    llc_loads: int = 0
    llc_load_misses: int = 0
    branches: int = 0
    branch_misses: int = 0
    
    # Derived metrics
    ipc: float = 0.0
    cache_miss_rate: float = 0.0
    l1_miss_rate: float = 0.0
    llc_miss_rate: float = 0.0
    
    def compute_derived(self):
        """Compute derived metrics from raw counters."""
        if self.cycles > 0:
            self.ipc = self.instructions / self.cycles
        if self.cache_references > 0:
            self.cache_miss_rate = self.cache_misses / self.cache_references
        if self.l1_dcache_loads > 0:
            self.l1_miss_rate = self.l1_dcache_load_misses / self.l1_dcache_loads
        if self.llc_loads > 0:
            self.llc_miss_rate = self.llc_load_misses / self.llc_loads


@dataclass
class ProfilingResult:
    """Results from a single profiling pass."""
    mode: str  # "normal", "hub_only", "nonhub_only"
    
    # Software counters from FlatNav
    total_distance_computations: int = 0
    hub_distance_computations: int = 0
    nonhub_distance_computations: int = 0
    
    # Hardware metrics
    perf_metrics: PerfMetrics = field(default_factory=PerfMetrics)
    
    # Timing
    total_search_time_ms: float = 0.0
    num_queries: int = 0
    
    # Per-computation derived metrics
    cycles_per_computation: float = 0.0
    cache_misses_per_computation: float = 0.0
    
    def compute_per_computation_metrics(self):
        """Compute per-distance-computation metrics."""
        total_comps = self.hub_distance_computations + self.nonhub_distance_computations
        if total_comps > 0:
            self.cycles_per_computation = self.perf_metrics.cycles / total_comps
            self.cache_misses_per_computation = self.perf_metrics.cache_misses / total_comps


def parse_perf_output(perf_stderr: str) -> PerfMetrics:
    """
    Parse perf stat CSV output to extract hardware counters.
    
    Expected format (with -x , flag):
        123456,,cycles,123456,100.00,,
        789012,,instructions,789012,100.00,,
    """
    metrics = PerfMetrics()
    
    # Map perf event names to our fields
    event_map = {
        'cycles': 'cycles',
        'instructions': 'instructions',
        'cache-references': 'cache_references',
        'cache-misses': 'cache_misses',
        'L1-dcache-loads': 'l1_dcache_loads',
        'L1-dcache-load-misses': 'l1_dcache_load_misses',
        'LLC-loads': 'llc_loads',
        'LLC-load-misses': 'llc_load_misses',
        'branches': 'branches',
        'branch-misses': 'branch_misses',
    }
    
    for line in perf_stderr.strip().split('\n'):
        if not line or line.startswith('#'):
            continue
        
        parts = line.split(',')
        if len(parts) < 3:
            continue
        
        try:
            value_str = parts[0].strip().replace(',', '')
            if not value_str or value_str == '<not supported>':
                continue
            value = int(value_str)
            event_name = parts[2].strip()
            
            if event_name in event_map:
                setattr(metrics, event_map[event_name], value)
        except (ValueError, IndexError):
            continue
    
    metrics.compute_derived()
    return metrics


def get_perf_events() -> List[str]:
    """Return list of perf events to measure."""
    return [
        'cycles',
        'instructions',
        'cache-references',
        'cache-misses',
        'L1-dcache-loads',
        'L1-dcache-load-misses',
        'LLC-loads',
        'LLC-load-misses',
        'branches',
        'branch-misses',
    ]


def check_perf_available() -> bool:
    """Check if perf tool is available."""
    try:
        result = subprocess.run(['perf', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def select_hub_nodes(node_access_counts: Dict[int, int], percentile: float = 90) -> List[int]:
    """
    Select hub nodes based on access count percentile.
    
    Args:
        node_access_counts: Dict mapping node_id -> access_count
        percentile: Percentile threshold (default 90 = top 10% most accessed)
    
    Returns:
        List of node IDs that are classified as hubs
    """
    if not node_access_counts:
        return []
    
    access_counts = list(node_access_counts.values())
    threshold = np.percentile(access_counts, percentile)
    
    hub_nodes = [
        node_id for node_id, count in node_access_counts.items()
        if count >= threshold
    ]
    
    logging.info(f"Selected {len(hub_nodes)} hub nodes (top {100-percentile}% of {len(node_access_counts)} nodes)")
    return hub_nodes


def build_index_with_hnsw(
    train_data: np.ndarray,
    distance_type: str,
    max_edges_per_node: int,
    ef_construction: int,
) -> Tuple[Any, str]:
    """
    Build FlatNav index using HNSW graph structure.
    
    Returns:
        Tuple of (flatnav_index, mtx_filename)
    """
    dataset_size, dim = train_data.shape
    
    # Build HNSW index to get graph structure
    hnsw_index = hnswlib.Index(
        space=distance_type if distance_type == "l2" else "ip",
        dim=dim
    )
    hnsw_index.init_index(
        max_elements=dataset_size,
        ef_construction=ef_construction,
        M=max_edges_per_node // 2,
    )
    hnsw_index.set_num_threads(32)
    
    logging.info("Building HNSW index...")
    start = time.time()
    hnsw_index.add_items(data=train_data, ids=np.arange(dataset_size))
    logging.info(f"HNSW indexing time: {time.time() - start:.2f}s")
    
    # Export graph structure
    mtx_filename = tempfile.mktemp(suffix=".mtx")
    hnsw_index.save_base_layer_graph(filename=mtx_filename)
    
    # Build FlatNav index
    flatnav_index = flatnav.index.create(
        distance_type=distance_type,
        dim=dim,
        dataset_size=dataset_size,
        max_edges_per_node=max_edges_per_node,
        verbose=True,
        collect_stats=True,
    )
    
    flatnav_index.allocate_nodes(train_data).build_graph_links(mtx_filename)
    
    return flatnav_index, mtx_filename


def run_profiling_pass(
    index: Any,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    ef_search: int,
    search_mode: int,
    mode_name: str,
    use_perf: bool = True,
) -> ProfilingResult:
    """
    Run a single profiling pass with specified search mode.
    
    Args:
        index: FlatNav index
        queries: Query vectors
        ground_truth: Ground truth labels
        k: Number of neighbors
        ef_search: Search parameter
        search_mode: SEARCH_MODE_NORMAL, HUB_ONLY, or NONHUB_ONLY
        mode_name: Human-readable mode name
        use_perf: Whether to use perf for hardware counters
    
    Returns:
        ProfilingResult with all metrics
    """
    result = ProfilingResult(mode=mode_name)
    result.num_queries = len(queries)
    
    # Set search mode and reset stats
    index.set_search_mode(search_mode)
    index.reset_stats()
    index.set_num_threads(1)  # Single-threaded for accurate profiling
    
    logging.info(f"Running {mode_name} pass with {result.num_queries} queries...")
    
    if use_perf and check_perf_available():
        # Run with perf stat
        perf_events = get_perf_events()
        
        # Create a temporary script to run the search
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            script_path = f.name
            # Note: We can't easily pickle the index, so we just time the search
            # and get perf metrics from the subprocess
        
        # For now, run search directly and just measure time
        # TODO: Implement proper subprocess-based perf measurement
        start_time = time.perf_counter()
        failed_queries = 0
        for i, query in enumerate(queries):
            try:
                _ = index.search_single(query, k, ef_search)
            except RuntimeError as e:
                # In HUB_ONLY or NONHUB_ONLY modes, we may not find k neighbors
                failed_queries += 1
                continue
        end_time = time.perf_counter()
        
        if failed_queries > 0:
            logging.warning(f"  {failed_queries} queries failed to return {k} results (expected in filtered modes)")
        
        result.total_search_time_ms = (end_time - start_time) * 1000
    else:
        # Run without perf
        start_time = time.perf_counter()
        failed_queries = 0
        for query in queries:
            try:
                _ = index.search_single(query, k, ef_search)
            except RuntimeError as e:
                # In HUB_ONLY or NONHUB_ONLY modes, we may not find k neighbors
                # This is expected - just continue
                failed_queries += 1
                continue
        end_time = time.perf_counter()
        
        if failed_queries > 0:
            logging.warning(f"  {failed_queries} queries failed to return {k} results (expected in filtered modes)")
        
        result.total_search_time_ms = (end_time - start_time) * 1000
    
    # Get software counters
    result.hub_distance_computations = index.get_hub_distance_computations()
    result.nonhub_distance_computations = index.get_nonhub_distance_computations()
    result.total_distance_computations = result.hub_distance_computations + result.nonhub_distance_computations
    
    logging.info(f"  Total distance computations: {result.total_distance_computations:,}")
    logging.info(f"  Hub distance computations: {result.hub_distance_computations:,}")
    logging.info(f"  Non-hub distance computations: {result.nonhub_distance_computations:,}")
    logging.info(f"  Total search time: {result.total_search_time_ms:.2f} ms")
    
    return result


def run_full_profiling(
    dataset_name: str,
    train_data: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    distance_type: str,
    max_edges_per_node: int,
    ef_construction: int,
    ef_search: int,
    k: int,
    hub_percentile: float = 90,
) -> Dict[str, ProfilingResult]:
    """
    Run full profiling with all three passes.
    
    Returns:
        Dict mapping mode name to ProfilingResult
    """
    results = {}
    
    # Build index
    logging.info(f"Building index for {dataset_name}...")
    index, mtx_filename = build_index_with_hnsw(
        train_data, distance_type, max_edges_per_node, ef_construction
    )
    
    # First pass: Normal search to get node access distribution
    logging.info("Pass 0: Getting node access distribution...")
    index.set_search_mode(SEARCH_MODE_NORMAL)
    index.reset_stats()
    index.set_num_threads(1)
    
    for query in queries:
        try:
            _ = index.search_single(query, k, ef_search)
        except RuntimeError:
            # Should not happen in NORMAL mode, but handle gracefully
            continue
    
    node_access_counts = dict(index.get_node_access_counts())
    
    # Select hub nodes
    hub_nodes = select_hub_nodes(node_access_counts, hub_percentile)
    index.set_hub_nodes(hub_nodes)
    
    # Pass 1: Normal search (baseline)
    results['normal'] = run_profiling_pass(
        index, queries, ground_truth, k, ef_search,
        SEARCH_MODE_NORMAL, "normal"
    )
    
    # Pass 2: Hub-only search
    results['hub_only'] = run_profiling_pass(
        index, queries, ground_truth, k, ef_search,
        SEARCH_MODE_HUB_ONLY, "hub_only"
    )
    
    # Pass 3: Non-hub-only search
    results['nonhub_only'] = run_profiling_pass(
        index, queries, ground_truth, k, ef_search,
        SEARCH_MODE_NONHUB_ONLY, "nonhub_only"
    )
    
    # Cleanup
    try:
        os.remove(mtx_filename)
    except OSError:
        pass
    
    return results


def print_profiling_summary(results: Dict[str, ProfilingResult], dataset_name: str):
    """Print a summary of profiling results."""
    print("\n" + "=" * 80)
    print(f"PROFILING RESULTS: {dataset_name}")
    print("=" * 80)
    
    print("\n### Distance Computation Breakdown ###")
    print(f"{'Mode':<15} {'Total Comps':>15} {'Hub Comps':>15} {'Non-Hub Comps':>15} {'Time (ms)':>12}")
    print("-" * 75)
    
    for mode, result in results.items():
        print(f"{mode:<15} {result.total_distance_computations:>15,} "
              f"{result.hub_distance_computations:>15,} "
              f"{result.nonhub_distance_computations:>15,} "
              f"{result.total_search_time_ms:>12.2f}")
    
    print("\n### Per-Query Metrics ###")
    for mode, result in results.items():
        if result.num_queries > 0:
            avg_comps = result.total_distance_computations / result.num_queries
            avg_time = result.total_search_time_ms / result.num_queries
            print(f"{mode}: {avg_comps:.1f} comps/query, {avg_time:.3f} ms/query")
    
    print("\n### Hub vs Non-Hub Analysis ###")
    normal = results.get('normal')
    hub_only = results.get('hub_only')
    nonhub_only = results.get('nonhub_only')
    
    if normal and hub_only and nonhub_only:
        # Compare time per computation
        if hub_only.total_distance_computations > 0:
            hub_time_per_comp = hub_only.total_search_time_ms / hub_only.total_distance_computations
            print(f"Hub time per computation: {hub_time_per_comp * 1000:.3f} µs")
        
        if nonhub_only.total_distance_computations > 0:
            nonhub_time_per_comp = nonhub_only.total_search_time_ms / nonhub_only.total_distance_computations
            print(f"Non-hub time per computation: {nonhub_time_per_comp * 1000:.3f} µs")
        
        if hub_only.total_distance_computations > 0 and nonhub_only.total_distance_computations > 0:
            ratio = (hub_only.total_search_time_ms / hub_only.total_distance_computations) / \
                    (nonhub_only.total_search_time_ms / nonhub_only.total_distance_computations)
            print(f"Hub/Non-hub time ratio: {ratio:.2f}x")
    
    print("=" * 80 + "\n")


def save_results(results: Dict[str, ProfilingResult], dataset_name: str, output_path: str):
    """Save profiling results to JSON file."""
    os.makedirs(output_path, exist_ok=True)
    
    output_data = {
        'dataset': dataset_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': {}
    }
    
    for mode, result in results.items():
        output_data['results'][mode] = {
            'total_distance_computations': result.total_distance_computations,
            'hub_distance_computations': result.hub_distance_computations,
            'nonhub_distance_computations': result.nonhub_distance_computations,
            'total_search_time_ms': result.total_search_time_ms,
            'num_queries': result.num_queries,
        }
    
    filepath = os.path.join(output_path, f"{dataset_name}_hub_profile.json")
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logging.info(f"Results saved to {filepath}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile hub vs non-hub vector search performance"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., glove-100-angular)"
    )
    
    parser.add_argument(
        "--root-dataset-path",
        type=str,
        default=ROOT_DATASET_PATH,
        help="Root path for datasets"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of nearest neighbors"
    )
    
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=100,
        help="ef_construction parameter"
    )
    
    parser.add_argument(
        "--ef-search",
        type=int,
        default=200,
        help="ef_search parameter"
    )
    
    parser.add_argument(
        "--num-node-links",
        type=int,
        default=32,
        help="max_edges_per_node parameter"
    )
    
    parser.add_argument(
        "--hub-percentile",
        type=float,
        default=90,
        help="Percentile threshold for hub selection (default: 90 = top 10%%)"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default=PROFILE_OUTPUT_PATH,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Limit number of queries (default: all)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load dataset
    dataset_name = args.dataset
    metric = get_metric_from_dataset_name(dataset_name)
    base_path = os.path.join(args.root_dataset_path, dataset_name)
    
    if not os.path.exists(base_path):
        logging.error(f"Dataset path not found: {base_path}")
        sys.exit(1)
    
    logging.info(f"Loading dataset {dataset_name} from {base_path}")
    train_data, queries, ground_truth = load_dataset(base_path, dataset_name)
    
    # Limit queries if specified
    if args.num_queries and args.num_queries < len(queries):
        queries = queries[:args.num_queries]
        ground_truth = ground_truth[:args.num_queries]
        logging.info(f"Limited to {len(queries)} queries")
    
    logging.info(f"Dataset: {train_data.shape[0]} vectors, {train_data.shape[1]} dimensions")
    logging.info(f"Queries: {len(queries)}")
    
    # Run profiling
    results = run_full_profiling(
        dataset_name=dataset_name,
        train_data=train_data,
        queries=queries,
        ground_truth=ground_truth,
        distance_type=metric,
        max_edges_per_node=args.num_node_links,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        k=args.k,
        hub_percentile=args.hub_percentile,
    )
    
    # Print and save results
    print_profiling_summary(results, dataset_name)
    save_results(results, dataset_name, args.output_path)


if __name__ == "__main__":
    main()
