import argparse
import json
import os
import pickle
import re
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


S_VERTEX_BYTES = 8
S_SCALAR_BYTES = 8
S_DISTANCE_BYTES = 8


def load_trace(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def build_hub_map(is_hub_entries: Iterable) -> Dict[int, bool]:
    hub_map: Dict[int, bool] = {}
    for entry in is_hub_entries:
        if isinstance(entry, dict):
            for k, v in entry.items():
                hub_map[int(k)] = bool(v)
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            hub_map[int(entry[0])] = bool(entry[1])
    return hub_map


def parse_dataset_label(filename: str) -> Tuple[str, int]:
    base = os.path.splitext(filename)[0]
    if "_searchstep_trace_" in base:
        dataset_part, percentile_part = base.split("_searchstep_trace_", 1)
        try:
            percentile = int(percentile_part)
        except ValueError:
            percentile = 0
    else:
        dataset_part = base
        percentile = 0
    label = f"{dataset_part}_p{percentile}" if percentile else dataset_part
    return label, percentile


def parse_dimension_from_label(label: str) -> int:
    match = re.search(r"-(\d+)-", label)
    if not match:
        raise ValueError(f"Unable to infer dimension from label: {label}")
    return int(match.group(1))


def summarize_trace(trace: dict) -> dict:
    hub_map = build_hub_map(trace.get("is_hub", []))

    total_explored = 0
    total_traversed = 0
    total_explored_hub = 0
    total_explored_nonhub = 0
    total_traversed_hub = 0
    total_traversed_nonhub = 0

    unique_explored = set()
    unique_traversed = set()
    unique_explored_hub = set()
    unique_explored_nonhub = set()
    unique_traversed_hub = set()
    unique_traversed_nonhub = set()

    for query in trace.get("queries", []):
        for step in query.get("search_steps", []):
            explored_list = step.get("explored_list", []) or []
            traversed_list = step.get("traversed_list", []) or []

            total_explored += len(explored_list)
            total_traversed += len(traversed_list)

            for node_id in explored_list:
                unique_explored.add(node_id)
                if hub_map.get(node_id, False):
                    total_explored_hub += 1
                    unique_explored_hub.add(node_id)
                else:
                    total_explored_nonhub += 1
                    unique_explored_nonhub.add(node_id)

            for node_id in traversed_list:
                unique_traversed.add(node_id)
                if hub_map.get(node_id, False):
                    total_traversed_hub += 1
                    unique_traversed_hub.add(node_id)
                else:
                    total_traversed_nonhub += 1
                    unique_traversed_nonhub.add(node_id)

    return {
        "total_explored": total_explored,
        "total_traversed": total_traversed,
        "total_explored_hub": total_explored_hub,
        "total_explored_nonhub": total_explored_nonhub,
        "total_traversed_hub": total_traversed_hub,
        "total_traversed_nonhub": total_traversed_nonhub,
        "unique_explored": len(unique_explored),
        "unique_traversed": len(unique_traversed),
        "unique_explored_hub": len(unique_explored_hub),
        "unique_explored_nonhub": len(unique_explored_nonhub),
        "unique_traversed_hub": len(unique_traversed_hub),
        "unique_traversed_nonhub": len(unique_traversed_nonhub),
    }


def bytes_trivial_offload() -> int:
    return S_VERTEX_BYTES + S_DISTANCE_BYTES


def bytes_reuse(dim: int) -> int:
    s_embedding = S_SCALAR_BYTES * dim
    return S_VERTEX_BYTES + s_embedding


def bytes_to_gb(value_bytes: float) -> float:
    return value_bytes / (1024.0 ** 3)


def plot_bar(values: List[Tuple[str, float, float]], title: str, out_path: str) -> None:
    labels = [v[0] for v in values]
    trivial_vals = np.array([v[1] for v in values])
    reuse_vals = np.array([v[2] for v in values])

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(max(10, len(labels) * 1.2), 8))
    plt.bar(x - width / 2, trivial_vals, width, label="DM trivial offload")
    plt.bar(x + width / 2, reuse_vals, width, label="DM reuse")

    plt.xticks(x, labels, rotation=35, ha="right")
    plt.xlabel("Dataset")
    plt.ylabel("Memory used (GB)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze search step traces and plot distance memory metrics")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.path.join(repo_root, "metrics", "search_step_trace"),
        help="Directory containing *_searchstep_trace_*.pkl files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(repo_root, "metrics", "search_step_trace", "plots"),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    if not os.path.isabs(args.input_dir):
        args.input_dir = os.path.join(repo_root, args.input_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(repo_root, args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    pkl_files = sorted(
        f for f in os.listdir(args.input_dir) if f.endswith(".pkl") and "searchstep_trace" in f
    )
    if not pkl_files:
        raise FileNotFoundError(f"No searchstep_trace .pkl files found in {args.input_dir}")

    overall_points = []
    hub_points = []
    nonhub_points = []
    summary_rows = []

    for filename in pkl_files:
        label, _ = parse_dataset_label(filename)
        dim = parse_dimension_from_label(label)
        size_trivial = bytes_trivial_offload()
        size_reuse = bytes_reuse(dim)

        trace = load_trace(os.path.join(args.input_dir, filename))
        stats = summarize_trace(trace)

        dm_trivial = bytes_to_gb(stats["total_explored"] * size_trivial)
        dm_reuse = bytes_to_gb(stats["unique_explored"] * size_reuse)
        overall_points.append((label, dm_trivial, dm_reuse))

        dm_trivial_hub = bytes_to_gb(stats["total_explored_hub"] * size_trivial)
        dm_reuse_hub = bytes_to_gb(stats["unique_explored_hub"] * size_reuse)
        hub_points.append((label, dm_trivial_hub, dm_reuse_hub))

        dm_trivial_nonhub = bytes_to_gb(stats["total_explored_nonhub"] * size_trivial)
        dm_reuse_nonhub = bytes_to_gb(stats["unique_explored_nonhub"] * size_reuse)
        nonhub_points.append((label, dm_trivial_nonhub, dm_reuse_nonhub))

        summary_rows.append(
            {
                "dataset": label,
                "dimension": dim,
                "total_explored": stats["total_explored"],
                "total_explored_hubs": stats["total_explored_hub"],
                "total_explored_nonhubs": stats["total_explored_nonhub"],
                "total_traversed": stats["total_traversed"],
                "total_traversed_hubs": stats["total_traversed_hub"],
                "total_traversed_nonhubs": stats["total_traversed_nonhub"],
                "unique_explored": stats["unique_explored"],
                "unique_explored_hubs": stats["unique_explored_hub"],
                "unique_explored_nonhubs": stats["unique_explored_nonhub"],
                "unique_traversed": stats["unique_traversed"],
                "unique_traversed_hubs": stats["unique_traversed_hub"],
                "unique_traversed_nonhubs": stats["unique_traversed_nonhub"],
                "dm_trivial_offload_gb": dm_trivial,
                "dm_reuse_gb": dm_reuse,
                "dm_trivial_offload_hubs_gb": dm_trivial_hub,
                "dm_reuse_hubs_gb": dm_reuse_hub,
                "dm_trivial_offload_nonhubs_gb": dm_trivial_nonhub,
                "dm_reuse_nonhubs_gb": dm_reuse_nonhub,
            }
        )

        print(f"Dataset: {label}")
        print(f"  Total explored computations: {stats['total_explored']}")
        print(f"    Hubs: {stats['total_explored_hub']} | Non-hubs: {stats['total_explored_nonhub']}")
        print(f"  Total traversed computations: {stats['total_traversed']}")
        print(f"    Hubs: {stats['total_traversed_hub']} | Non-hubs: {stats['total_traversed_nonhub']}")
        print(f"  Unique explored computations: {stats['unique_explored']}")
        print(f"    Hubs: {stats['unique_explored_hub']} | Non-hubs: {stats['unique_explored_nonhub']}")
        print(f"  Unique traversed computations: {stats['unique_traversed']}")
        print(f"    Hubs: {stats['unique_traversed_hub']} | Non-hubs: {stats['unique_traversed_nonhub']}")

    plot_bar(
        overall_points,
        "DM trivial offload vs DM reuse (All explored computations)",
        os.path.join(args.output_dir, "dm_trivial_vs_reuse_all.png"),
    )
    plot_bar(
        hub_points,
        "DM trivial offload vs DM reuse (Explored hubs)",
        os.path.join(args.output_dir, "dm_trivial_vs_reuse_hubs.png"),
    )
    plot_bar(
        nonhub_points,
        "DM trivial offload vs DM reuse (Explored non-hubs)",
        os.path.join(args.output_dir, "dm_trivial_vs_reuse_nonhubs.png"),
    )

    summary_path = os.path.join(args.output_dir, "dm_trivial_reuse_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)


if __name__ == "__main__":
    main()
