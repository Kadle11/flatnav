#!/usr/bin/env python3
"""
Profile Vector Search Performance with Hub/Non-Hub Breakdown

This script measures compute and memory bandwidth utilization during vector search,
providing a breakdown of distance computations between hub and non-hub nodes.

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

# Search mode (must match C++ enum)
SEARCH_MODE_NORMAL = 0

# Default dataset path
ROOT_DATASET_PATH = os.getenv("ROOT_DATASET_PATH", "/root/data/hubness/data")

# Output directory for profiling results
PROFILE_OUTPUT_PATH = os.getenv("PROFILE_OUTPUT_PATH", "/root/metrics/hub_profiling")

SYNTHETIC_DATASETS = [
    "normal-16-angular",
    "normal-16-euclidean",
    "normal-32-angular",
    "normal-32-euclidean",
    "normal-64-angular",
    "normal-64-euclidean",
    "normal-128-angular",
    "normal-128-euclidean",
    # "normal-256-angular",
    # "normal-256-euclidean",
    # "normal-1024-angular",
    # "normal-1024-euclidean",
    "normal-1536-angular",
    "normal-1536-euclidean",
]

ANN_DATASETS = [
    "glove-100-angular",
    "nytimes-256-angular",
    "gist-960-euclidean",
    # "yandex-deep-10m-euclidean",
    # "spacev-10m-euclidean",
]


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


def check_perf_available() -> Tuple[bool, str]:
    """
    Check if perf tool is available and usable.
    
    Returns:
        (is_available, message)
    """
    try:
        result = subprocess.run(['perf', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            return False, "perf command not found"
        
        # Check perf_event_paranoid level
        try:
            with open('/proc/sys/kernel/perf_event_paranoid', 'r') as f:
                paranoid_level = int(f.read().strip())
                
            if paranoid_level > 2:
                return False, (
                    f"perf_event_paranoid={paranoid_level} is too restrictive. "
                    f"Run: sudo sysctl -w kernel.perf_event_paranoid=1 (or -1 for full access)"
                )
            elif paranoid_level > 0:
                return True, f"perf available with restricted access (paranoid={paranoid_level})"
            else:
                return True, f"perf available with full access (paranoid={paranoid_level})"
        except:
            # If we can't read paranoid level, try running perf stat anyway
            test_result = subprocess.run(
                ['perf', 'stat', '-e', 'cycles', '--', 'ls'],
                capture_output=True, text=True
            )
            if test_result.returncode == 0:
                return True, "perf available (paranoid level unknown)"
            else:
                return False, f"perf test failed: {test_result.stderr[:200]}"
                
    except FileNotFoundError:
        return False, "perf command not installed"


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
    hub_nodes: List[int],
    distance_type: str,
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
    
    perf_available, perf_msg = check_perf_available()
    if use_perf and not perf_available:
        logging.warning(f"perf requested but not available: {perf_msg}")
        logging.warning("Falling back to software-only metrics")
        use_perf = False
    elif use_perf:
        logging.info(f"Using perf: {perf_msg}")
    
    if use_perf:
        # Create a temporary script that will be wrapped by perf
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            script_path = f.name
            # Write a self-contained search script
            # Determine the correct index class based on distance type
            if distance_type == 'l2':
                index_class = 'IndexL2Float'
            else:
                index_class = 'IndexIPFloat'
            
            f.write(f"""#!/usr/bin/env python3
import sys
sys.path.insert(0, '{Path(__file__).parent}')
import numpy as np
import flatnav.index

# Load the serialized index using the correct class
IndexClass = getattr(flatnav.index, '{index_class}')
index = IndexClass.load_index('{tempfile.gettempdir()}/profiling_index.flatnav')
queries = np.load('{tempfile.gettempdir()}/profiling_queries.npy')
hub_nodes = np.load('{tempfile.gettempdir()}/profiling_hub_nodes.npy').tolist()

# Debug output
print(f"DEBUG: Loaded {{len(hub_nodes)}} hub nodes", file=sys.stderr)
print(f"DEBUG: Loaded {{len(queries)}} queries", file=sys.stderr)
sys.stderr.flush()

# Enable stats collection (not serialized with index)
index.set_collect_stats(True)

# Set hub nodes before running searches
print("DEBUG: About to set hub nodes...", file=sys.stderr)
sys.stderr.flush()
index.set_hub_nodes(hub_nodes)
hub_count = index.count_hub_nodes()
print(f"DEBUG: Set hub nodes, count_hub_nodes()={{hub_count}}", file=sys.stderr)
sys.stderr.flush()

print("DEBUG: About to set search mode...", file=sys.stderr)
sys.stderr.flush()
index.set_search_mode({search_mode})
print("DEBUG: Set search mode", file=sys.stderr)
sys.stderr.flush()

print("DEBUG: About to reset stats...", file=sys.stderr)
sys.stderr.flush()
index.reset_stats()
print("DEBUG: Reset stats", file=sys.stderr)
sys.stderr.flush()

index.set_num_threads(1)

print(f"DEBUG: Starting search with mode={search_mode}", file=sys.stderr)
sys.stderr.flush()

failed_count = 0
for i, query in enumerate(queries):
    try:
        _ = index.search_single(query, {k}, {ef_search})
    except RuntimeError as e:
        failed_count += 1
        if failed_count <= 5:
            print(f"DEBUG: Query {{i}} failed: {{e}}", file=sys.stderr)
        continue

print(f"DEBUG: Finished search loop, failed_count={{failed_count}}", file=sys.stderr)
sys.stderr.flush()

try:
    total_dist = index.get_query_distance_computations()
    hub_dist = index.get_hub_distance_computations()
    nonhub_dist = index.get_nonhub_distance_computations()
    print(f"DEBUG: Stats: total={{total_dist}}, hub={{hub_dist}}, nonhub={{nonhub_dist}}", file=sys.stderr)
except Exception as e:
    print(f"DEBUG: Error getting stats: {{e}}", file=sys.stderr)
    total_dist = 0
    hub_dist = 0
    nonhub_dist = 0

# Print stats for parent process to parse
print(f"FAILED:{{failed_count}}")
print(f"TOTAL_DIST_COMPS:{{total_dist}}")
print(f"HUB_DIST_COMPS:{{hub_dist}}")
print(f"NONHUB_DIST_COMPS:{{nonhub_dist}}")
""")
        
        # Save index, queries, and hub_nodes to temp files
        index_path = f'{tempfile.gettempdir()}/profiling_index.flatnav'
        queries_path = f'{tempfile.gettempdir()}/profiling_queries.npy'
        hub_nodes_path = f'{tempfile.gettempdir()}/profiling_hub_nodes.npy'
        index.save(index_path)
        np.save(queries_path, queries)
        np.save(hub_nodes_path, np.array(hub_nodes, dtype=np.uint32))
        
        # Run with perf stat
        perf_events = ','.join(get_perf_events())
        perf_cmd = [
            'perf', 'stat',
            '-e', perf_events,
            '-x', ',',  # CSV output
            'python3', script_path
        ]
        
        start_time = time.perf_counter()
        proc_result = subprocess.run(perf_cmd, capture_output=True, text=True)
        end_time = time.perf_counter()
        
        result.total_search_time_ms = (end_time - start_time) * 1000
        
        # Log subprocess output for debugging - ALWAYS show this
        logging.info(f"  Subprocess return code: {proc_result.returncode}")
        logging.info(f"  Subprocess stdout (first 500): {proc_result.stdout[:500] if proc_result.stdout else 'EMPTY'}")
        logging.info(f"  Subprocess stderr (first 500): {proc_result.stderr[:500] if proc_result.stderr else 'EMPTY'}")
        
        # Parse perf output from stderr
        result.perf_metrics = parse_perf_output(proc_result.stderr)
        
        # Parse stats from stdout
        failed_queries = 0
        for line in proc_result.stdout.split('\n'):
            if line.startswith('FAILED:'):
                failed_queries = int(line.split(':')[1])
            elif line.startswith('TOTAL_DIST_COMPS:'):
                result.total_distance_computations = int(line.split(':')[1])
            elif line.startswith('HUB_DIST_COMPS:'):
                result.hub_distance_computations = int(line.split(':')[1])
            elif line.startswith('NONHUB_DIST_COMPS:'):
                result.nonhub_distance_computations = int(line.split(':')[1])
        
        # Cleanup
        try:
            os.unlink(script_path)
            os.unlink(index_path)
            os.unlink(queries_path)
            os.unlink(hub_nodes_path)
        except:
            pass
        
        if failed_queries > 0:
            logging.warning(f"  {failed_queries} queries failed to return {k} results (expected in filtered modes)")
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
        
        # Only get software counters from parent index when NOT using perf subprocess
        # (when using perf, we already parsed them from subprocess stdout above)
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
) -> ProfilingResult:
    """
    Run profiling with normal search pass.
    
    Returns:
        ProfilingResult with all metrics
    """
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
            _ = index.search_single(query, k, ef_search, num_initializations=100)
        except RuntimeError:
            # Should not happen in NORMAL mode, but handle gracefully
            continue
    
    node_access_counts = dict(index.get_node_access_counts())
    
    # Select hub nodes
    hub_nodes = select_hub_nodes(node_access_counts, hub_percentile)
    index.set_hub_nodes(hub_nodes)
    
    # Run normal search pass with profiling
    result = run_profiling_pass(
        index, queries, ground_truth, k, ef_search,
        SEARCH_MODE_NORMAL, "normal", hub_nodes, distance_type, use_perf=True
    )
    
    # Cleanup
    try:
        os.remove(mtx_filename)
    except OSError:
        pass
    
    return result


def print_profiling_summary(result: ProfilingResult, dataset_name: str):
    """Print a summary of profiling results."""
    print("\n" + "=" * 80)
    print(f"PROFILING RESULTS: {dataset_name}")
    print("=" * 80)
    
    print("\n### Distance Computation Breakdown ###")
    print(f"{'Total Comps':>15} {'Hub Comps':>15} {'Non-Hub Comps':>15} {'Time (ms)':>12}")
    print("-" * 60)
    print(f"{result.total_distance_computations:>15,} "
          f"{result.hub_distance_computations:>15,} "
          f"{result.nonhub_distance_computations:>15,} "
          f"{result.total_search_time_ms:>12.2f}")
    
    print("\n### Hardware Metrics (perf) ###")
    
    # Check if perf metrics were collected
    has_perf_data = result.perf_metrics.cycles > 0
    
    if not has_perf_data:
        print("⚠️  WARNING: No hardware metrics collected!")
        print("   Perf counters are not available. This can happen due to:")
        print("   1. perf_event_paranoid level too high (current: check /proc/sys/kernel/perf_event_paranoid)")
        print("   2. Running in a container/VM without perf access")
        print("   3. Kernel compiled without perf support")
        print("\n   To enable perf on the host:")
        print("   sudo sysctl -w kernel.perf_event_paranoid=1")
        print("   (or -1 for full access, or 2 for user-space only)")
        print("\n   Then rebuild and rerun the Docker container.")
        print("\n   Software-only metrics (timing, computation counts) are still valid below.\n")
    
    pm = result.perf_metrics
    cache_miss_pct = pm.cache_miss_rate * 100 if pm.cache_miss_rate > 0 else 0
    l1_miss_pct = pm.l1_miss_rate * 100 if pm.l1_miss_rate > 0 else 0
    
    print(f"{'Cycles':>15} {'Instructions':>15} {'IPC':>8} {'Cache Miss %':>12} {'L1 Miss %':>10}")
    print("-" * 65)
    print(f"{pm.cycles:>15,} {pm.instructions:>15,} "
          f"{pm.ipc:>8.3f} {cache_miss_pct:>11.2f}% {l1_miss_pct:>9.2f}%")
    
    print("\n### Per-Distance-Computation Hardware Metrics ###")
    print(f"{'Cycles/Comp':>15} {'Cache Misses/Comp':>20} {'L1 Misses/Comp':>18}")
    print("-" * 55)
    if result.total_distance_computations > 0:
        cycles_per = result.perf_metrics.cycles / result.total_distance_computations
        cache_miss_per = result.perf_metrics.cache_misses / result.total_distance_computations
        l1_miss_per = result.perf_metrics.l1_dcache_load_misses / result.total_distance_computations
        print(f"{cycles_per:>15.2f} {cache_miss_per:>20.4f} {l1_miss_per:>18.4f}")
    
    print("\n### Per-Query Metrics ###")
    if result.num_queries > 0:
        avg_comps = result.total_distance_computations / result.num_queries
        avg_time = result.total_search_time_ms / result.num_queries
        print(f"{avg_comps:.1f} comps/query, {avg_time:.3f} ms/query")
    
    print("\n### Hub vs Non-Hub Breakdown ###")
    total_comps = result.hub_distance_computations + result.nonhub_distance_computations
    if total_comps > 0:
        hub_pct = result.hub_distance_computations / total_comps * 100
        nonhub_pct = result.nonhub_distance_computations / total_comps * 100
        print(f"Hub computations: {result.hub_distance_computations:,} ({hub_pct:.1f}%)")
        print(f"Non-hub computations: {result.nonhub_distance_computations:,} ({nonhub_pct:.1f}%)")
    
    print("=" * 80 + "\n")


def save_results(result: ProfilingResult, dataset_name: str, output_path: str):
    """Save profiling results to JSON file."""
    os.makedirs(output_path, exist_ok=True)
    
    pm = result.perf_metrics
    total_comps = result.hub_distance_computations + result.nonhub_distance_computations
    hub_pct = result.hub_distance_computations / total_comps * 100 if total_comps > 0 else 0
    nonhub_pct = result.nonhub_distance_computations / total_comps * 100 if total_comps > 0 else 0
    
    output_data = {
        'dataset': dataset_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'distance_computations': {
            'total': result.total_distance_computations,
            'hub': result.hub_distance_computations,
            'nonhub': result.nonhub_distance_computations,
            'hub_percentage': hub_pct,
            'nonhub_percentage': nonhub_pct
        },
        'timing': {
            'total_search_time_ms': result.total_search_time_ms,
            'num_queries': result.num_queries,
            'avg_time_per_query_ms': result.total_search_time_ms / result.num_queries if result.num_queries > 0 else 0,
            'avg_comps_per_query': result.total_distance_computations / result.num_queries if result.num_queries > 0 else 0
        },
        'perf_metrics': {
            'cycles': pm.cycles,
            'instructions': pm.instructions,
            'IPC': pm.ipc,
            'cache_miss_rate': pm.cache_miss_rate,
            'l1_miss_rate': pm.l1_miss_rate,
            'cycles_per_computation': pm.cycles / result.total_distance_computations if result.total_distance_computations > 0 else 0,
            'cache_misses_per_computation': pm.cache_misses / result.total_distance_computations if result.total_distance_computations > 0 else 0
        }
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
        default=100,
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
    
    for dataset_name in SYNTHETIC_DATASETS + ANN_DATASETS:
    
        # Load dataset
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
        result = run_full_profiling(
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
        print_profiling_summary(result, dataset_name)
        save_results(result, dataset_name, args.output_path)


if __name__ == "__main__":
    main()
