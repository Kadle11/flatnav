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
import random

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
ROOT_DATASET_PATH = os.getenv("ROOT_DATASET_PATH", "/root/data")

# Output directory for profiling results
PROFILE_OUTPUT_PATH = os.getenv("PROFILE_OUTPUT_PATH", "/root/metrics/hub_profiling")

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
    # "normal-512-euclidean",
    # "normal-1024-angular",
    # "normal-1024-euclidean",
    # "normal-1536-angular",
    # "normal-1536-euclidean",
]

ANN_DATASETS = [
    # "glove-100-angular",
    # "nytimes-256-angular",
    # "gist-960-euclidean",
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

    # Degree stats for accessed nodes
    accessed_avg_in_degree: float = 0.0
    accessed_avg_out_degree: float = 0.0
    
    # Accuracy (fraction [0,1])
    recall_at_k: float = 0.0
    
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


def select_hub_nodes_by_degree(index: Any, percentile: float = 90) -> Tuple[List[int], Dict[str, float]]:
    """
    Select hub nodes based on total graph degree (in-degree + out-degree) percentile.
    This is query-INDEPENDENT - degree is a property of the graph structure only.
    
    Args:
        index: FlatNav index
        percentile: Percentile threshold (default 90 = top 10% highest degree nodes)
    
    Returns:
        Tuple of (hub_nodes, degree_stats) where degree_stats contains avg in/out degrees for hubs and non-hubs
    """
    # Get the outdegree table: list of neighbor lists for each node
    outdegree_table = index.get_graph_outdegree_table()
    num_nodes = len(outdegree_table)
    
    # Calculate out-degrees
    out_degrees = {node_id: len(neighbors) for node_id, neighbors in enumerate(outdegree_table)}
    
    # Calculate in-degrees by counting how many times each node appears as a neighbor
    in_degrees = {node_id: 0 for node_id in range(num_nodes)}
    for node_id, neighbors in enumerate(outdegree_table):
        for neighbor_id in neighbors:
            in_degrees[neighbor_id] += 1
    
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
    hub_nodes_set = set(hub_nodes)
    
    # Compute statistics
    avg_in_degree = sum(in_degrees.values()) / num_nodes if num_nodes > 0 else 0
    avg_out_degree = sum(out_degrees.values()) / num_nodes if num_nodes > 0 else 0
    avg_total_degree = sum(total_degrees.values()) / num_nodes if num_nodes > 0 else 0
    
    # Compute separate stats for hubs and non-hubs
    hub_in_degrees = [in_degrees[n] for n in hub_nodes]
    hub_out_degrees = [out_degrees[n] for n in hub_nodes]
    nonhub_in_degrees = [in_degrees[n] for n in range(num_nodes) if n not in hub_nodes_set]
    nonhub_out_degrees = [out_degrees[n] for n in range(num_nodes) if n not in hub_nodes_set]
    
    avg_hub_in_degree = sum(hub_in_degrees) / len(hub_in_degrees) if hub_in_degrees else 0
    avg_hub_out_degree = sum(hub_out_degrees) / len(hub_out_degrees) if hub_out_degrees else 0
    avg_nonhub_in_degree = sum(nonhub_in_degrees) / len(nonhub_in_degrees) if nonhub_in_degrees else 0
    avg_nonhub_out_degree = sum(nonhub_out_degrees) / len(nonhub_out_degrees) if nonhub_out_degrees else 0
    
    degree_stats = {
        'avg_hub_in_degree': avg_hub_in_degree,
        'avg_hub_out_degree': avg_hub_out_degree,
        'avg_nonhub_in_degree': avg_nonhub_in_degree,
        'avg_nonhub_out_degree': avg_nonhub_out_degree,
    }
    
    logging.info(f"Selected {len(hub_nodes)} hub nodes by total degree (top {100-percentile}% of {num_nodes} nodes)")
    logging.info(f"  Degree threshold: {threshold:.1f}")
    logging.info(f"  In-degree  - Min: {min(in_degrees.values())}, Max: {max(in_degrees.values())}, Avg: {avg_in_degree:.1f}")
    logging.info(f"  Out-degree - Min: {min(out_degrees.values())}, Max: {max(out_degrees.values())}, Avg: {avg_out_degree:.1f}")
    logging.info(f"  Total deg  - Min: {min(degree_values)}, Max: {max(degree_values)}, Avg: {avg_total_degree:.1f}")
    logging.info(f"  Hub avg in-degree: {avg_hub_in_degree:.1f}, Hub avg out-degree: {avg_hub_out_degree:.1f}")
    logging.info(f"  Non-hub avg in-degree: {avg_nonhub_in_degree:.1f}, Non-hub avg out-degree: {avg_nonhub_out_degree:.1f}")
    
    return hub_nodes, degree_stats


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
    hnsw_index.set_num_threads(16)
    
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

    # --- Compute average in/out degree for accessed nodes ---
    # Get outdegree table
    outdegree_table = index.get_graph_outdegree_table()
    num_nodes = len(outdegree_table)
    out_degrees = {node_id: len(neighbors) for node_id, neighbors in enumerate(outdegree_table)}
    in_degrees = {node_id: 0 for node_id in range(num_nodes)}
    for node_id, neighbors in enumerate(outdegree_table):
        for neighbor_id in neighbors:
            in_degrees[neighbor_id] += 1

    # Snapshot node access counts BEFORE running queries for this pass
    node_access_counts_before = dict(index.get_node_access_counts())
    # Run queries to collect visitation
    perf_available, perf_msg = check_perf_available()
    if use_perf and not perf_available:
        logging.warning(f"perf requested but not available: {perf_msg}")
        logging.warning("Falling back to software-only metrics")
        use_perf = False
    elif use_perf:
        logging.info(f"Using perf: {perf_msg}")
    
    # Helper for recall
    def _extract_ids(res_obj) -> List[int]:
        """Robustly extract neighbor IDs from various return formats.
        Handles FlatNav `(distances, indices)` and HNSW `(indices, distances)`.
        """
        import numpy as np
        # list/tuple container
        if isinstance(res_obj, (list, tuple)):
            # tuple of two ndarrays: choose the integer one
            if (
                len(res_obj) == 2
                and isinstance(res_obj[0], np.ndarray)
                and isinstance(res_obj[1], np.ndarray)
            ):
                a0, a1 = res_obj
                if np.issubdtype(a1.dtype, np.integer):
                    return [int(x) for x in a1.ravel()]
                if np.issubdtype(a0.dtype, np.integer):
                    return [int(x) for x in a0.ravel()]
                # Fallback: prefer second array
                try:
                    return [int(x) for x in a1.astype(np.int64).ravel()]
                except Exception:
                    return []
            # list/tuple of (id, dist)
            if len(res_obj) > 0 and isinstance(res_obj[0], (list, tuple)) and len(res_obj[0]) >= 1:
                return [int(x[0]) for x in res_obj]
            # list/tuple of ids or mixed
            out: List[int] = []
            for x in res_obj:
                if isinstance(x, (int, np.integer)):
                    out.append(int(x))
                elif isinstance(x, np.ndarray):
                    if x.size == 1:
                        out.append(int(x.item()))
                    else:
                        out.extend([int(v) for v in x.ravel()])
                else:
                    try:
                        out.append(int(x))
                    except Exception:
                        pass
            return out
        # numpy array
        if isinstance(res_obj, np.ndarray):
            if res_obj.ndim == 1 and np.issubdtype(res_obj.dtype, np.integer):
                return [int(x) for x in res_obj]
            if res_obj.ndim == 2 and res_obj.shape[1] >= 1:
                # Prefer integer columns if present; else first column
                for col in range(res_obj.shape[1]):
                    col_arr = res_obj[:, col]
                    if np.issubdtype(col_arr.dtype, np.integer):
                        return [int(x) for x in col_arr]
                col0 = res_obj[:, 0]
                try:
                    return [int(x) for x in col0]
                except Exception:
                    return [int(x[0]) for x in res_obj]
            if res_obj.dtype == object and res_obj.size > 0:
                first = res_obj.flat[0]
                if isinstance(first, (list, tuple)) and len(first) >= 1:
                    return [int(x[0]) for x in res_obj.flat]
            return []
        # scalar fallback
        try:
            return [int(res_obj)]
        except Exception:
            return []

    total_hits = 0.0
    evaluated = 0

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
        _ = index.search_single(query=query, K={k}, ef_search={ef_search})
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
        
        # Re-run queries in parent process to collect node access stats (perf subprocess doesn't update parent index)
        logging.info(f"  Re-running queries in parent process to collect node access statistics...")
        # Important: reset stats to clear any accumulated counts from previous runs
        index.reset_stats()
        index.set_num_threads(1)
        for qi, query in enumerate(queries):
            try:
                res = index.search_single(query=query, K=k, ef_search=ef_search)
                result_ids = _extract_ids(res)
                if ground_truth is not None:
                    gt_row = ground_truth[qi]
                    gt_set = set(int(x) for x in np.asarray(gt_row)[:k])
                    hits = sum(1 for rid in result_ids[:k] if rid in gt_set)
                    total_hits += (hits / float(k)) if k > 0 else 0.0
                    evaluated += 1
            except RuntimeError:
                continue
        logging.info(f"  Finished re-running {len(queries)} queries for node access stats")
        if evaluated > 0:
            result.recall_at_k = total_hits / evaluated
    else:
        # Run without perf
        start_time = time.perf_counter()
        failed_queries = 0
        for qi, query in enumerate(queries):
            try:
                res = index.search_single(query=query, K=k, ef_search=ef_search)
                result_ids = _extract_ids(res)
                if ground_truth is not None:
                    gt_row = ground_truth[qi]
                    gt_set = set(int(x) for x in np.asarray(gt_row)[:k])
                    hits = sum(1 for rid in result_ids[:k] if rid in gt_set)
                    total_hits += (hits / float(k)) if k > 0 else 0.0
                    evaluated += 1
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
        if evaluated > 0:
            result.recall_at_k = total_hits / evaluated
    
    logging.info(f"  Total distance computations: {result.total_distance_computations:,}")
    logging.info(f"  Hub distance computations: {result.hub_distance_computations:,}")
    logging.info(f"  Non-hub distance computations: {result.nonhub_distance_computations:,}")
    logging.info(f"  Total search time: {result.total_search_time_ms:.2f} ms")
    logging.info(f"  Recall@{k}: {result.recall_at_k*100:.2f}%")

    # After queries, compute delta-accessed nodes for THIS pass
    node_access_counts_after = dict(index.get_node_access_counts())
    # Build union of keys to compute deltas robustly
    all_keys = set(node_access_counts_before.keys()) | set(node_access_counts_after.keys())
    accessed_nodes = set()
    total_delta_accesses = 0
    for nid in all_keys:
        before = node_access_counts_before.get(nid, 0)
        after = node_access_counts_after.get(nid, 0)
        delta = after - before
        if delta > 0:
            accessed_nodes.add(nid)
            total_delta_accesses += delta

    logging.info(f"  Delta node accesses (this pass): {total_delta_accesses}, Unique nodes accessed (delta): {len(accessed_nodes)}")
    
    if accessed_nodes:
        accessed_in_degrees = [in_degrees[n] for n in accessed_nodes]
        accessed_out_degrees = [out_degrees[n] for n in accessed_nodes]
        result.accessed_avg_in_degree = float(np.mean(accessed_in_degrees))
        result.accessed_avg_out_degree = float(np.mean(accessed_out_degrees))
        logging.info(f"  Avg in-degree: {result.accessed_avg_in_degree:.2f}, Avg out-degree: {result.accessed_avg_out_degree:.2f}")
        logging.info(f"  Sample accessed nodes (first 5): {sorted(accessed_nodes)[:5]}")
    else:
        result.accessed_avg_in_degree = 0.0
        result.accessed_avg_out_degree = 0.0
        logging.warning("  No accessed nodes detected for this pass - degree stats will be 0")
    
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
    hub_selection_method: str = "access-count",
    all_queries: Optional[np.ndarray] = None,
    all_ground_truth: Optional[np.ndarray] = None,
    num_queries_limit: Optional[int] = None,
    query_ratio_extremes: bool = False,
) -> Tuple[ProfilingResult, Optional[Dict[str, ProfilingResult]], Optional[Dict[str, float]]]:
    """
    Run profiling with normal search pass (and optional ratio-extremes subsets).
    
    Args:
        hub_selection_method: 'access-count' (query-dependent) or 'degree' (static)
        all_queries: full query set (untrimmed) used to compute ratio extremes
        all_ground_truth: full ground truth aligned with all_queries
        num_queries_limit: how many queries to keep for top/bottom subsets
        query_ratio_extremes: if True, also profile top/bottom queries by hub/nonhub ratio
    
    Returns:
        (base ProfilingResult, extra_results dict or None, degree_stats dict or None)
    """
    # Build index
    logging.info(f"Building index for {dataset_name}...")
    index, mtx_filename = build_index_with_hnsw(
        train_data, distance_type, max_edges_per_node, ef_construction
    )
    
    degree_stats = None
    # Select hub nodes based on chosen method
    if hub_selection_method == "degree":
        logging.info("Using DEGREE-BASED hub classification (static, query-independent)")
        hub_nodes, degree_stats = select_hub_nodes_by_degree(index, hub_percentile)
    else:
        # Default: access-count based (query-dependent)
        logging.info("Using ACCESS-COUNT-BASED hub classification (query-dependent)")
        
        # First pass: Normal search to get node access distribution
        logging.info("Pass 0: Getting node access distribution...")
        index.set_search_mode(SEARCH_MODE_NORMAL)
        index.reset_stats()
        index.set_num_threads(1)
        
        for query in queries:
            try:
                _ = index.search_single(query=query, K=k, ef_search=ef_search, num_initializations=100)
            except RuntimeError:
                # Should not happen in NORMAL mode, but handle gracefully
                continue
        
        node_access_counts = dict(index.get_node_access_counts())
        hub_nodes = select_hub_nodes(node_access_counts, hub_percentile)
    
    # Set hub nodes on the index
    index.set_hub_nodes(hub_nodes)
    
    # Run normal search pass with profiling on the provided query slice
    result = run_profiling_pass(
        index, queries, ground_truth, k, ef_search,
        SEARCH_MODE_NORMAL, "normal", hub_nodes, distance_type, use_perf=True
    )
    
    extra_results: Optional[Dict[str, ProfilingResult]] = None
    
    if query_ratio_extremes and all_queries is not None and all_ground_truth is not None and num_queries_limit:
        logging.info("Computing hub/nonhub ratio per query over full query set to select extremes...")
        # Reset stats and collect per-query hub/nonhub counts
        ratios: List[Tuple[float, int]] = []
        index.reset_stats()
        index.set_collect_stats(True)
        index.set_search_mode(SEARCH_MODE_NORMAL)
        index.set_num_threads(1)
        per_query_stats: List[Tuple[int, int]] = []
        for q in all_queries:
            try:
                _ = index.search_single(query=q, K=k, ef_search=ef_search, num_initializations=100)
            except RuntimeError:
                per_query_stats.append((0, 0))
                continue
            per_query_stats.append((index.get_hub_distance_computations(), index.get_nonhub_distance_computations()))
            index.reset_stats()
        
        # Compute ratios (hub / nonhub, handle zero)
        for idx, (hub_c, nonhub_c) in enumerate(per_query_stats):
            ratio = hub_c / nonhub_c if nonhub_c > 0 else float('inf') if hub_c > 0 else 0.0
            ratios.append((ratio, idx))
        
        ratios_sorted = sorted(ratios, key=lambda x: x[0])
        bottom_indices = [idx for _, idx in ratios_sorted[:num_queries_limit]]
        top_indices = [idx for _, idx in ratios_sorted[-num_queries_limit:]]
        
        def subset(arr: np.ndarray, indices: List[int]) -> np.ndarray:
            return np.asarray(arr)[indices]
        
        top_queries = subset(all_queries, top_indices)
        top_gt = subset(all_ground_truth, top_indices)
        bottom_queries = subset(all_queries, bottom_indices)
        bottom_gt = subset(all_ground_truth, bottom_indices)
        
        extra_results = {}
        logging.info(f"Profiling TOP ratio queries (count={len(top_queries)})")
        extra_results["top_ratio"] = run_profiling_pass(
            index, top_queries, top_gt, k, ef_search,
            SEARCH_MODE_NORMAL, "top_ratio", hub_nodes, distance_type, use_perf=True
        )
        logging.info(f"Profiling BOTTOM ratio queries (count={len(bottom_queries)})")
        extra_results["bottom_ratio"] = run_profiling_pass(
            index, bottom_queries, bottom_gt, k, ef_search,
            SEARCH_MODE_NORMAL, "bottom_ratio", hub_nodes, distance_type, use_perf=True
        )
    
    # Cleanup
    try:
        os.remove(mtx_filename)
    except OSError:
        pass
    
    return result, extra_results, degree_stats


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


def save_results(result: ProfilingResult, dataset_name: str, output_path: str, 
                 extra_results: Optional[Dict[str, ProfilingResult]] = None,
                 degree_stats: Optional[Dict[str, float]] = None,
                 hub_selection_method: str = "access-count"):
    """Save profiling results to JSON file (optionally with extra subsets and degree stats)."""
    os.makedirs(output_path, exist_ok=True)
    
    def to_dict(res: ProfilingResult) -> Dict[str, Any]:
        pm = res.perf_metrics
        total_comps = res.hub_distance_computations + res.nonhub_distance_computations
        hub_pct = res.hub_distance_computations / total_comps * 100 if total_comps > 0 else 0
        nonhub_pct = res.nonhub_distance_computations / total_comps * 100 if total_comps > 0 else 0
        return {
            'distance_computations': {
                'total': res.total_distance_computations,
                'hub': res.hub_distance_computations,
                'nonhub': res.nonhub_distance_computations,
                'hub_percentage': hub_pct,
                'nonhub_percentage': nonhub_pct
            },
            'timing': {
                'total_search_time_ms': res.total_search_time_ms,
                'num_queries': res.num_queries,
                'avg_time_per_query_ms': res.total_search_time_ms / res.num_queries if res.num_queries > 0 else 0,
                'avg_comps_per_query': res.total_distance_computations / res.num_queries if res.num_queries > 0 else 0
            },
            'accuracy': {
                'recall_at_k_percent': res.recall_at_k * 100.0
            },
            'perf_metrics': {
                'cycles': pm.cycles,
                'instructions': pm.instructions,
                'IPC': pm.ipc,
                'cache_miss_rate': pm.cache_miss_rate,
                'l1_miss_rate': pm.l1_miss_rate,
                'cycles_per_computation': pm.cycles / res.total_distance_computations if res.total_distance_computations > 0 else 0,
                'cache_misses_per_computation': pm.cache_misses / res.total_distance_computations if res.total_distance_computations > 0 else 0
            },
            'accessed_node_degrees': {
                'avg_in_degree': res.accessed_avg_in_degree,
                'avg_out_degree': res.accessed_avg_out_degree
            }
        }
    
    output_data: Dict[str, Any] = {
        'dataset': dataset_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hub_selection_method': hub_selection_method,
        'baseline': to_dict(result)
    }
    
    # Add degree statistics if available (only for degree-based hub selection)
    if degree_stats is not None:
        output_data['degree_statistics'] = degree_stats
    
    if extra_results:
        output_data['query_ratio_extremes'] = {
            key: to_dict(val) for key, val in extra_results.items()
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
        "--hub-selection-method",
        type=str,
        choices=["access-count", "degree"],
        default="access-count",
        help="Method for hub classification: 'access-count' (query-dependent) or 'degree' (static, query-independent)"
    )

    parser.add_argument(
        "--query-ratio-extremes",
        action="store_true",
        help="If set, identify top/bottom num-queries by hub/nonhub access ratio and profile them separately"
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
        
        # Shuffle queries and ground_truth together
        combined = list(zip(queries, ground_truth))
        random.shuffle(combined)
        all_queries, all_ground_truth = zip(*combined)
        
        # Select num_queries after shuffling for baseline runs
        queries = all_queries[:args.num_queries]
        ground_truth = all_ground_truth[:args.num_queries]
        
        logging.info(f"Dataset: {train_data.shape[0]} vectors, {train_data.shape[1]} dimensions")
        logging.info(f"Queries (baseline slice): {len(queries)}; full queries available: {len(all_queries)}")
        
        # Run profiling (with optional ratio extremes)
        result, extra_results, degree_stats = run_full_profiling(
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
            hub_selection_method=args.hub_selection_method,
            all_queries=np.array(all_queries),
            all_ground_truth=np.array(all_ground_truth),
            num_queries_limit=args.num_queries,
            query_ratio_extremes=args.query_ratio_extremes,
        )
        
        # Print and save results
        print_profiling_summary(result, dataset_name)
        save_results(result, dataset_name, args.output_path, extra_results, 
                     degree_stats, args.hub_selection_method)


if __name__ == "__main__":
    main()
