#!/usr/bin/env python3
"""
SIFT-100M Benchmark Profiler with Recall and Memory Tracking

Measures:
- Index build time and memory
- Query execution time at various ef-search values
- Recall@100
- Peak memory usage during search

Usage:
    python sift_benchmark_profiler.py --local
    python sift_benchmark_profiler.py --docker

Works alongside pcm-memory for bandwidth measurement:
    sudo pcm-memory 0.5 -csv=system_bandwidth.csv &
    poetry run python sift_benchmark_profiler.py --local
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import flatnav
from flatnav.data_type import DataType

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


FLATNAV_DATA_TYPES = {
    "float32": DataType.float32,
    "uint8": DataType.uint8,
    "int8": DataType.int8,
}


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmark execution"""
    dataset_name: str
    num_vectors: int
    dimension: int
    num_queries: int
    num_node_links: int
    num_build_threads: int
    ef_construction: int
    metric: str
    build_time_sec: float = 0.0
    initial_rss_mb: float = 0.0
    peak_rss_mb: float = 0.0
    memory_used_mb: float = 0.0
    peak_bandwidth_mb_s: float = 0.0
    bandwidth_csv: str = ""
    search_results: Dict[str, Dict[str, float]] = field(default_factory=dict)  # ef_search -> {time, recall}
    
    def to_dict(self):
        """Convert to dict for JSON serialization"""
        return asdict(self)
    
    def print_summary(self):
        """Print human-readable summary"""
        print("\n" + "="*70)
        print(f"SIFT-100M Benchmark Results ({self.metric.upper()})")
        print("="*70)
        print(f"Dataset:          {self.dataset_name}")
        print(f"Vectors:          {self.num_vectors:,}")
        print(f"Dimension:        {self.dimension}")
        print(f"Query Set Size:   {self.num_queries}")
        print(f"Node Links:       {self.num_node_links}")
        print(f"Build Threads:     {self.num_build_threads}")
        print(f"EF Construction:  {self.ef_construction}")
        print("-"*70)
        print(f"Build Time:       {self.build_time_sec:.2f}s")
        print(f"Initial RSS:      {self.initial_rss_mb:.1f} MB")
        print(f"Peak RSS:         {self.peak_rss_mb:.1f} MB")
        print(f"Memory Used:      {self.memory_used_mb:.1f} MB")
        if self.bandwidth_csv:
            print(f"Bandwidth CSV:    {self.bandwidth_csv}")
            print(f"Peak Bandwidth:   {self.peak_bandwidth_mb_s:.1f} MB/s")
        print("\nSearch Performance (EF-Search, Time, Recall@100):")
        print("-"*70)
        for ef_search in sorted(self.search_results.keys(), key=lambda x: int(x)):
            result = self.search_results[ef_search]
            time_ms = result.get("time_ms", 0.0)
            recall = result.get("recall", 0.0)
            queries_per_sec = (self.num_queries * 1000.0 / time_ms) if time_ms > 0 else 0
            print(f"  EF={ef_search:>3}  |  {time_ms:>8.2f}ms  |  {recall*100:>6.2f}%  |  {queries_per_sec:>7.0f} q/s")
        print("="*70 + "\n")


def get_memory_stats_mb() -> Tuple[float, float]:
    """Get current RSS and peak RSS in MB."""
    try:
        with open(f'/proc/{os.getpid()}/status') as f:
            rss_mb = 0.0
            peak_rss_mb = 0.0
            for line in f:
                if line.startswith('VmRSS:'):
                    rss_mb = float(line.split()[1]) / 1024.0
                elif line.startswith('VmHWM:'):
                    peak_rss_mb = float(line.split()[1]) / 1024.0
            return rss_mb, peak_rss_mb
    except:
        return 0.0, 0.0


def summarize_bandwidth_csv(csv_path: str) -> float:
    """Best-effort peak bandwidth summary from a pcm-memory CSV file."""
    if not csv_path or not os.path.exists(csv_path):
        return 0.0

    peak_bandwidth = 0.0
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_total = 0.0
                for key, value in row.items():
                    if not key:
                        continue
                    lowered = key.lower()
                    if "time" in lowered or "timestamp" in lowered or "date" in lowered:
                        continue
                    try:
                        row_total += float(value)
                    except (TypeError, ValueError):
                        continue
                peak_bandwidth = max(peak_bandwidth, row_total)
    except Exception:
        return 0.0

    return peak_bandwidth


def load_benchmark_data(
    data_path: str,
    queries_path: str,
    gtruth_path: str,
    max_queries: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load benchmark dataset using appropriate loader"""
    logger.info(f"Loading training data from: {data_path}")
    logger.info(f"Loading queries from: {queries_path}")
    logger.info(f"Loading ground truth from: {gtruth_path}")
    
    # Load training data directly based on the training file extension.
    if data_path.endswith(".fvecs"):
        # Keep the 100M SIFT base as a memmap-backed view to avoid a massive in-memory copy.
        with open(data_path, "rb") as f:
            dim = int(np.fromfile(f, dtype=np.int32, count=1)[0])
            f.seek(0, os.SEEK_END)
            total_vectors = f.tell() // ((dim + 1) * np.dtype(np.float32).itemsize)

        train_data = np.memmap(
            data_path,
            dtype=np.float32,
            mode="r",
            shape=(total_vectors, dim + 1),
        )[:, 1:]
    else:
        train_data = np.load(data_path, mmap_mode="r")

    # Load queries and ground truth directly so `.npy` shapes stay intact.
    queries = np.load(queries_path, mmap_mode="r").astype(np.float32, copy=False)
    ground_truth = np.load(gtruth_path, mmap_mode="r").astype(np.int32, copy=False)
    
    # Evaluate on at most 10k queries and keep strict alignment with ground truth.
    if max_queries > 0:
        eval_queries = min(max_queries, queries.shape[0], ground_truth.shape[0])
        queries = queries[:eval_queries]
        ground_truth = ground_truth[:eval_queries]

    logger.info(f"Training set shape: {train_data.shape}, dtype: {train_data.dtype}")
    logger.info(f"Queries shape (eval): {queries.shape}, dtype: {queries.dtype}")
    logger.info(f"Ground truth shape (eval): {ground_truth.shape}, dtype: {ground_truth.dtype}")
    
    return train_data, queries, ground_truth


def build_index(
    train_data: np.ndarray,
    num_node_links: int,
    ef_construction: int,
    num_build_threads: int,
    build_batch_size: int,
    metric: str
) -> Tuple[any, float, float, float]:
    """Build FlatNav index and measure time/memory"""
    logger.info(
        f"Building index with M={num_node_links}, ef_construction={ef_construction}, "
        f"num_build_threads={num_build_threads}"
    )
    initial_rss_mb, _ = get_memory_stats_mb()
    
    build_start = time.time()
    index = flatnav.index.create(
        distance_type=metric,
        index_data_type=FLATNAV_DATA_TYPES["float32"],
        dim=train_data.shape[1],
        dataset_size=train_data.shape[0],
        max_edges_per_node=num_node_links,
        verbose=False,
        collect_stats=False,
    )
    index.set_num_threads(num_build_threads)

    # Build in contiguous chunks so large fvecs-backed views do not trigger huge copies.
    dataset_size = train_data.shape[0]
    for start in range(0, dataset_size, build_batch_size):
        end = min(start + build_batch_size, dataset_size)
        batch = np.ascontiguousarray(train_data[start:end], dtype=np.float32)
        index.add(
            data=batch,
            ef_construction=ef_construction,
            num_initializations=100,
        )

        if end == dataset_size or (end // build_batch_size) % 5 == 0:
            logger.info(f"  Indexed {end:,}/{dataset_size:,} vectors")
    
    build_time = time.time() - build_start
    current_rss_mb, peak_rss_mb = get_memory_stats_mb()
    memory_used = max(0.0, peak_rss_mb - initial_rss_mb)
    
    logger.info(
        f"Build completed in {build_time:.2f}s, RSS={current_rss_mb:.1f}MB, peak RSS={peak_rss_mb:.1f}MB"
    )
    
    return index, build_time, initial_rss_mb, peak_rss_mb, memory_used


def run_searches(
    index: any,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    ef_search_values: List[int],
    k: int = 100
) -> Dict[str, Dict[str, float]]:
    """Run searches at various ef_search values and compute recall"""
    results = {}
    
    for ef_search in ef_search_values:
        logger.info(f"Running search with ef_search={ef_search}")
        search_start = time.time()
        recalls = []
        
        for i, query in enumerate(queries):
            # FlatNav search API returns (distances, indices).
            _, neighbors = index.search_single(
                query=query,
                ef_search=ef_search,
                K=k,
                num_initializations=100,
            )
            
            # Compute recall@k
            if len(neighbors) > 0:
                found_neighbors = set(neighbors.tolist())
                true_neighbors = set(ground_truth[i, :k])
                recall = len(found_neighbors & true_neighbors) / k if len(true_neighbors) > 0 else 0.0
                recalls.append(recall)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1:,}/{len(queries)} queries")
        
        search_time = time.time() - search_start
        avg_recall = np.mean(recalls) if recalls else 0.0
        time_per_query_ms = (search_time * 1000.0) / len(queries) if len(queries) > 0 else 0.0
        
        results[str(ef_search)] = {
            "time_ms": time_per_query_ms,
            "recall": avg_recall,
            "total_time_sec": search_time,
        }
        
        logger.info(f"  EF={ef_search}: {time_per_query_ms:.2f}ms/query, recall@100={avg_recall*100:.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="SIFT-100M Benchmark Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sift_benchmark_profiler.py --local
  python sift_benchmark_profiler.py --docker
  python sift_benchmark_profiler.py --docker --output /root/metrics/sift-benchmark.json
        """
    )
    
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local paths (/mydata/flatnav/data)"
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Use Docker paths (/root/data)"
    )
    parser.add_argument(
        "--num-node-links",
        type=int,
        default=32,
        help="Number of bi-directional links per node (default: 32)"
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=100,
        help="Construction ef parameter (default: 100)"
    )
    parser.add_argument(
        "--num-build-threads",
        type=int,
        default=1,
        help="Number of threads to use during index construction (default: 1)"
    )
    parser.add_argument(
        "--allow-unsafe-parallel-build",
        action="store_true",
        help=(
            "Allow multi-threaded build for very large .fvecs datasets. "
            "By default, the profiler falls back to 1 thread to avoid native crashes."
        ),
    )
    parser.add_argument(
        "--build-batch-size",
        type=int,
        default=1_000_000,
        help="Number of vectors per add() batch during index construction (default: 1000000)"
    )
    parser.add_argument(
        "--ef-search-values",
        type=str,
        default="100,200",
        help="Comma-separated ef-search values to test (default: 100,200)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="l2",
        help="Distance metric: l2 or angular (default: l2)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of neighbors to retrieve (default: 100)"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=10000,
        help="Maximum number of queries to evaluate, aligned with ground truth rows (default: 10000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (default: stdout only)"
    )
    parser.add_argument(
        "--bandwidth-csv",
        type=str,
        default="system_bandwidth.csv",
        help="pcm-memory CSV file to summarize bandwidth from (default: system_bandwidth.csv)"
    )
    
    args = parser.parse_args()
    
    # Validate environment selection
    if not (args.local or args.docker):
        parser.error("Must specify either --local or --docker")
    if args.local and args.docker:
        parser.error("Cannot specify both --local and --docker")
    if args.build_batch_size <= 0:
        parser.error("--build-batch-size must be > 0")
    
    # Set paths based on environment
    if args.local:
        base_path = "/mydata/flatnav/data/sift-128-euclidean"
        large_base_path = "/mydata/flatnav/data/sift-128-euclidean"
        data_path = f"{large_base_path}/sift100m_base.fvecs"
        queries_path = f"{base_path}/sift-128-euclidean.test.npy"
        gtruth_path = f"{base_path}/sift-128-euclidean.gtruth.npy"
        env_type = "LOCAL"
    else:
        base_path = "/root/data/sift-128-euclidean"
        large_base_path = "/root/data/sift-128-euclidean-big"
        data_path = f"{large_base_path}/sift100m_base.fvecs"
        queries_path = f"{base_path}/sift-128-euclidean.test.npy"
        gtruth_path = f"{base_path}/sift-128-euclidean.gtruth.npy"
        env_type = "DOCKER"
    
    logger.info(f"Starting SIFT-100M benchmark ({env_type} environment)")
    logger.info(f"Data path: {data_path}")
    
    # Load data
    train_data, queries, ground_truth = load_benchmark_data(
        data_path, queries_path, gtruth_path, max_queries=args.max_queries
    )

    effective_build_threads = args.num_build_threads
    large_fvecs_dataset = data_path.endswith(".fvecs") and train_data.shape[0] >= 50_000_000
    if large_fvecs_dataset and args.num_build_threads > 1 and not args.allow_unsafe_parallel_build:
        logger.warning(
            "Detected very large .fvecs training set with multi-threaded build requested. "
            "Falling back to 1 build thread to avoid known native crashes. "
            "Pass --allow-unsafe-parallel-build to override."
        )
        effective_build_threads = 1
    
    # Build index
    index, build_time, initial_rss_mb, peak_rss_mb, memory_used_mb = build_index(
        train_data,
        args.num_node_links,
        args.ef_construction,
        effective_build_threads,
        args.build_batch_size,
        args.metric
    )
    
    # Parse ef_search values
    ef_search_values = [int(x.strip()) for x in args.ef_search_values.split(",")]
    
    # Run searches
    search_results = run_searches(
        index, queries, ground_truth, ef_search_values, args.k
    )

    peak_bandwidth_mb_s = summarize_bandwidth_csv(args.bandwidth_csv)
    
    # Compile metrics
    metrics = BenchmarkMetrics(
        dataset_name="sift-100m",
        num_vectors=train_data.shape[0],
        dimension=train_data.shape[1],
        num_queries=queries.shape[0],
        num_node_links=args.num_node_links,
        num_build_threads=effective_build_threads,
        ef_construction=args.ef_construction,
        metric=args.metric,
        build_time_sec=build_time,
        initial_rss_mb=initial_rss_mb,
        peak_rss_mb=peak_rss_mb,
        memory_used_mb=memory_used_mb,
        peak_bandwidth_mb_s=peak_bandwidth_mb_s,
        bandwidth_csv=args.bandwidth_csv,
        search_results=search_results,
    )
    
    # Print summary
    metrics.print_summary()
    
    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
