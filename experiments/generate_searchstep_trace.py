#!/usr/bin/env python3
"""
Generate detailed search step traces for vector search queries.

This script analyzes the beam search algorithm's exploration patterns, recording:
- Which nodes were explored at each step
- The distance threshold filtering (explored vs traversed lists)
- Hub/non-hub classification based on graph degree

Output format includes search steps with explored and traversed node lists.

Usage:
    python generate_searchstep_trace.py --dataset gist-960-euclidean --k 100 --ef-search 100
"""

import argparse
import logging
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Add python-build directory to use locally built _core module
python_build_path = str(Path(__file__).parent.parent / "python-build")
if python_build_path not in sys.path:
    sys.path.insert(0, python_build_path)

import _core
import hnswlib
from utils import get_metric_from_dataset_name, load_dataset

# Use _core.index instead of flatnav.index
flatnav_index = _core.index
DataType = _core.data_type.DataType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default dataset path
ROOT_DATASET_PATH = os.getenv("ROOT_DATASET_PATH", "data/")

# Output directory for results
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/root/metrics/search_step_traces")


def select_hub_nodes_by_degree(index: Any, percentile: float = 90) -> List[int]:
    """
    Select hub nodes based on total graph degree (in-degree + out-degree) percentile.
    This is query-INDEPENDENT - degree is a property of the graph structure only.
    
    Args:
        index: FlatNav index
        percentile: Percentile threshold (default 90 = top 10% highest degree nodes)
    
    Returns:
        List of hub node IDs
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
            if neighbor != node_id:  # Skip self-loops
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
    
    logging.info(f"Selected {len(hub_nodes)} hub nodes by total degree (top {100-percentile}% of {num_nodes} nodes)")
    logging.info(f"  Degree threshold: {threshold:.1f}")
    logging.info(f"  In-degree  - Min: {min(in_degrees.values())}, Max: {max(in_degrees.values())}, Avg: {np.mean(list(in_degrees.values())):.1f}")
    logging.info(f"  Out-degree - Min: {min(out_degrees.values())}, Max: {max(out_degrees.values())}, Avg: {np.mean(list(out_degrees.values())):.1f}")
    logging.info(f"  Total deg  - Min: {min(degree_values)}, Max: {max(degree_values)}, Avg: {np.mean(degree_values):.1f}")
    
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
    hnsw_index.set_num_threads(16)
    
    logging.info("Building HNSW index...")
    start = time.time()
    hnsw_index.add_items(data=train_data, ids=np.arange(dataset_size))
    logging.info(f"HNSW indexing time: {time.time() - start:.2f}s")
    
    # Export graph structure
    mtx_filename = tempfile.mktemp(suffix=".mtx")
    hnsw_index.save_base_layer_graph(filename=mtx_filename)
    
    # Build FlatNav index
    flatnav_idx = flatnav_index.create(
        distance_type=distance_type,
        dim=dim,
        dataset_size=dataset_size,
        max_edges_per_node=max_edges_per_node,
        verbose=True,
        collect_stats=False,
        index_data_type=DataType.float32,
    )
    
    flatnav_idx.allocate_nodes(train_data).build_graph_links(mtx_filename)
    
    return flatnav_idx, mtx_filename


def generate_search_step_traces(
    index: Any,
    queries: np.ndarray,
    k: int,
    ef_search: int,
    hub_nodes: List[int],
    ground_truth: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Generate search step traces for all queries.
    
    Args:
        index: FlatNav index
        queries: Query vectors (num_queries x dim)
        k: Number of neighbors
        ef_search: Search parameter
        hub_nodes: List of node IDs classified as hubs
        ground_truth: Ground truth nearest-neighbor labels per query
    
    Returns:
        List of query trace dicts with search steps
    """
    hub_nodes_set = set(hub_nodes)
    
    traces = []
    num_queries = queries.shape[0] if queries.ndim > 1 else 1
    
    # Handle single query case
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    
    logging.info(f"Generating search step traces for {num_queries} queries...")

    def extract_ids(search_output: Any) -> List[int]:
        """Robustly extract neighbor IDs from search output formats."""
        if isinstance(search_output, (list, tuple)):
            if (
                len(search_output) == 2
                and isinstance(search_output[0], np.ndarray)
                and isinstance(search_output[1], np.ndarray)
            ):
                first, second = search_output
                if np.issubdtype(second.dtype, np.integer):
                    return [int(x) for x in second.ravel() if int(x) >= 0]
                if np.issubdtype(first.dtype, np.integer):
                    return [int(x) for x in first.ravel() if int(x) >= 0]

            extracted: List[int] = []
            for value in search_output:
                if isinstance(value, np.ndarray):
                    if np.issubdtype(value.dtype, np.integer):
                        extracted.extend([int(x) for x in value.ravel() if int(x) >= 0])
                elif isinstance(value, (int, np.integer)):
                    extracted.append(int(value))
            return extracted

        if isinstance(search_output, np.ndarray):
            if np.issubdtype(search_output.dtype, np.integer):
                return [int(x) for x in search_output.ravel() if int(x) >= 0]
            return []

        try:
            candidate = int(search_output)
            return [candidate] if candidate >= 0 else []
        except Exception:
            return []

    def compute_recall_at_k(predicted: Any, ground_truth_labels: Any, k_value: int) -> Optional[float]:
        if ground_truth_labels is None:
            return None

        pred = extract_ids(predicted)
        gt = [int(label) for label in np.asarray(ground_truth_labels).reshape(-1) if int(label) >= 0]

        if not gt:
            return 0.0

        pred_topk = pred[:k_value]
        gt_topk = gt[:k_value]

        if len(pred_topk) < len(gt_topk):
            logging.debug(
                "Recall uses fewer predicted labels than requested K "
                f"(pred={len(pred_topk)}, gt={len(gt_topk)}, k={k_value})"
            )

        hit_count = len(set(pred_topk) & set(gt_topk))
        return hit_count / len(gt_topk)
    
    for query_idx in range(num_queries):
        if query_idx % max(1, num_queries // 10) == 0:
            logging.info(f"  Processing query {query_idx}/{num_queries}...")
        
        # Clear previous trace
        index.clear_visited_nodes_by_search_step()
        
        # Run search
        query = queries[query_idx]
        distances, labels = index.search_single_with_node_ids(
            query=query,
            K=k,
            ef_search=ef_search,
            num_initializations=5
        )
        
        # Get search steps
        all_search_steps = index.get_visited_nodes_by_search_step()

        recall_accuracy = None
        if ground_truth is not None and query_idx < len(ground_truth):
            recall_accuracy = compute_recall_at_k((distances, labels), ground_truth[query_idx], k)
        
        # Extract search steps for this query (should be only 1 since we call search_single)
        if all_search_steps and len(all_search_steps) > 0:
            query_steps = all_search_steps[0]
            
            # Convert search steps to output format
            steps_output = []
            for step in query_steps:
                step_dict = {
                    "node_id": step["node_id"],
                    "level": step["level"],
                    "explored_list": sorted(step["explored_list"]),
                    "candidate_list": sorted(step["traversed_list"]),
                }
                steps_output.append(step_dict)
            
            trace = {
                "query_id": query_idx,
                "search_steps": steps_output,
                "recall_accuracy": recall_accuracy,
            }
            traces.append(trace)
    
    logging.info(f"Generated traces for {len(traces)} queries")
    
    return traces


def build_is_hub_mapping(traces: List[Dict[str, Any]], hub_nodes: List[int], num_total_nodes: int) -> Dict[int, bool]:
    """
    Build a complete is_hub mapping for all nodes seen in the traces.
    Maps node_id -> True/False based on hub classification.
    
    Args:
        traces: List of query traces
        hub_nodes: List of hub node IDs
        num_total_nodes: Total number of nodes in the index
    
    Returns:
        Dictionary mapping node_id -> is_hub (bool)
    """
    hub_nodes_set = set(hub_nodes)
    
    # Collect all nodes seen across all traces
    all_seen_nodes: Set[int] = set()
    for trace in traces:
        for step in trace["search_steps"]:
            all_seen_nodes.update(step["explored_list"])
            all_seen_nodes.update(step["candidate_list"])
    
    # Build is_hub mapping for all seen nodes
    is_hub_mapping = {
        node_id: node_id in hub_nodes_set
        for node_id in sorted(all_seen_nodes)
    }
    
    logging.info(f"Built is_hub mapping for {len(is_hub_mapping)} nodes")
    hub_count = sum(1 for v in is_hub_mapping.values() if v)
    logging.info(f"  Hub nodes: {hub_count}, Non-hub nodes: {len(is_hub_mapping) - hub_count}")
    
    return is_hub_mapping


def build_node_degree_mapping(index: Any, node_ids: Set[int]) -> Dict[int, Dict[str, int]]:
    """
    Build in-degree and out-degree mapping for a set of node IDs.

    Args:
        index: FlatNav index
        node_ids: Node IDs to include in degree mapping

    Returns:
        Dictionary mapping node_id -> {"in_degree": int, "out_degree": int}
    """
    outdegree_table = index.get_graph_outdegree_table()
    num_nodes = len(outdegree_table)

    out_degrees = {node_id: len(neighbors) for node_id, neighbors in enumerate(outdegree_table)}

    in_degrees = {node_id: 0 for node_id in range(num_nodes)}
    for node_id, neighbors in enumerate(outdegree_table):
        for neighbor in neighbors:
            if neighbor != node_id:
                in_degrees[neighbor] += 1

    degree_mapping: Dict[int, Dict[str, int]] = {}
    for node_id in sorted(node_ids):
        if 0 <= node_id < num_nodes:
            degree_mapping[node_id] = {
                "in_degree": int(in_degrees[node_id]),
                "out_degree": int(out_degrees[node_id]),
            }

    logging.info(f"Built node degree mapping for {len(degree_mapping)} nodes")
    return degree_mapping


def save_search_step_traces(
    traces: List[Dict[str, Any]],
    is_hub_mapping: Dict[int, bool],
    node_degree_mapping: Dict[int, Dict[str, int]],
    hub_percentile: float,
    dataset_name: str,
    output_path: str,
    trace_params: Optional[Dict[str, Any]] = None,
):
    """Save search step traces to pickle file."""
    os.makedirs(output_path, exist_ok=True)
    
    # Convert is_hub mapping to list of {node_id: is_hub} objects
    is_hub_list = [
        {node_id: is_hub}
        for node_id, is_hub in is_hub_mapping.items()
    ]
    
    output_data = {
        "hub_percentile": hub_percentile,
        "is_hub": is_hub_list,
        "node_degrees": [
            {node_id: degree_counts}
            for node_id, degree_counts in node_degree_mapping.items()
        ],
        "queries": traces,
        "trace_params": trace_params or {},
    }

    params = trace_params or {}
    hub_pct = params.get("hub_percentile", hub_percentile)
    num_queries = params.get("num_queries_processed", params.get("num_queries_requested", "all"))

    def _format_value(value: Any) -> str:
        if isinstance(value, float):
            if value.is_integer():
                return str(int(value))
            return str(value).replace('.', 'p')
        return str(value)

    filename = (
        f"{dataset_name}"
        f"_k{_format_value(params.get('k', 'na'))}"
        f"_efc{_format_value(params.get('ef_construction', 'na'))}"
        f"_efs{_format_value(params.get('ef_search', 'na'))}"
        f"_m{_format_value(params.get('num_node_links', 'na'))}"
        f"_hubp{_format_value(hub_pct)}"
        f"_nq{_format_value(num_queries)}"
        f"_searchstep_trace.pkl"
    )

    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logging.info(f"Saved search step traces to {filepath}")
    
    # Print summary stats
    print("\n" + "=" * 80)
    print(f"SEARCH STEP TRACE SUMMARY: {dataset_name}")
    print("=" * 80)
    print(f"Hub percentile: {hub_percentile}")
    if trace_params:
        print("Trace parameters:")
        print(f"  k: {trace_params.get('k')}")
        print(f"  ef_construction: {trace_params.get('ef_construction')}")
        print(f"  ef_search: {trace_params.get('ef_search')}")
        print(f"  num_node_links: {trace_params.get('num_node_links')}")
        print(f"  distance_type: {trace_params.get('distance_type')}")
        print(f"  num_queries_requested: {trace_params.get('num_queries_requested')}")
        print(f"  num_queries_processed: {trace_params.get('num_queries_processed')}")
    print(f"Total queries processed: {len(traces)}")
    print(f"Total unique nodes seen: {len(is_hub_mapping)}")
    hub_count = sum(1 for v in is_hub_mapping.values() if v)
    print(f"Hub nodes: {hub_count} ({hub_count/len(is_hub_mapping)*100:.1f}%)")
    print(f"Non-hub nodes: {len(is_hub_mapping) - hub_count} ({(len(is_hub_mapping)-hub_count)/len(is_hub_mapping)*100:.1f}%)")
    
    # Stats about search steps
    total_steps = sum(len(trace["search_steps"]) for trace in traces)
    avg_steps = total_steps / len(traces) if traces else 0
    print(f"Total search steps across all queries: {total_steps}")
    print(f"Average steps per query: {avg_steps:.1f}")
    
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate detailed search step traces for vector search queries"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., gist-960-euclidean, mnist-784-euclidean)"
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
        help="Number of nearest neighbors to return"
    )
    
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=100,
        help="ef_construction parameter for HNSW graph construction"
    )
    
    parser.add_argument(
        "--ef-search",
        type=int,
        default=100,
        help="ef_search parameter for beam search"
    )
    
    parser.add_argument(
        "--num-node-links",
        type=int,
        default=32,
        help="max_edges_per_node (M) parameter"
    )
    
    parser.add_argument(
        "--hub-percentile",
        type=float,
        default=90,
        help="Percentile threshold for hub selection based on graph degree (default: 90 = top 10%%)"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default=OUTPUT_PATH,
        help="Output directory for search step traces"
    )
    
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Limit number of queries (default: use all available test queries)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logging.info(f"Loading dataset: {args.dataset}")
    
    # Load dataset
    dataset_config = get_metric_from_dataset_name(args.dataset)
    base_path = os.path.join(args.root_dataset_path, args.dataset)
    train_data, test_data, ground_truth = load_dataset(base_path, args.dataset)
    
    logging.info(f"Train data shape: {train_data.shape}")
    logging.info(f"Test data shape: {test_data.shape}")
    
    # Limit queries if specified
    if args.num_queries is not None:
        test_data = test_data[:args.num_queries]
        ground_truth = ground_truth[:args.num_queries]
        logging.info(f"Limited to {args.num_queries} test queries")
    
    # Determine distance type from dataset name
    distance_type = "l2" if "euclidean" in args.dataset else "angular"
    
    logging.info(f"Building FlatNav index with distance_type={distance_type}...")
    
    # Build index
    index, mtx_filename = build_index_with_hnsw(
        train_data, 
        distance_type,
        args.num_node_links,
        args.ef_construction
    )
    
    # Select hub nodes based on graph degree
    hub_nodes = select_hub_nodes_by_degree(index, percentile=args.hub_percentile)
    
    # Generate search step traces
    logging.info(f"Generating search step traces for {len(test_data)} queries...")
    traces = generate_search_step_traces(
        index,
        test_data,
        args.k,
        args.ef_search,
        hub_nodes,
        ground_truth,
    )
    
    # Build is_hub mapping for all nodes seen in traces
    is_hub_mapping = build_is_hub_mapping(traces, hub_nodes, train_data.shape[0])

    # Build in/out degree mapping for nodes included in is_hub mapping
    node_degree_mapping = build_node_degree_mapping(index, set(is_hub_mapping.keys()))
    
    # Save results
    save_search_step_traces(
        traces,
        is_hub_mapping,
        node_degree_mapping,
        args.hub_percentile,
        args.dataset,
        args.output_path,
        trace_params={
            "dataset": args.dataset,
            "k": args.k,
            "ef_construction": args.ef_construction,
            "ef_search": args.ef_search,
            "num_node_links": args.num_node_links,
            "hub_percentile": args.hub_percentile,
            "distance_type": distance_type,
            "num_queries_requested": args.num_queries,
            "num_queries_processed": len(test_data),
        },
    )
    
    # Cleanup
    try:
        os.remove(mtx_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
