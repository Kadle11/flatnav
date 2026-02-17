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
            traversed_list = step.get("candidate_list", []) or []

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


def compute_reuse_stats(trace: dict) -> dict:
    hub_map = build_hub_map(trace.get("is_hub", []))
    per_query = []
    for query in trace.get("queries", []):
        search_steps = query.get("search_steps", []) or []
        reuse_beyond_prev_count = 0
        reuse_beyond_prev_hub_count = 0
        reuse_beyond_prev_nonhub_count = 0
        comparable_steps = 0
        comparable_steps_hub = 0
        comparable_steps_nonhub = 0

        for idx in range(1, len(search_steps)):
            prev_step = search_steps[idx - 1]
            prev_traversed = prev_step.get("candidate_list", []) or []


            node_id = search_steps[idx].get("node_id", None)
            if node_id is None:
                continue

            is_hub = hub_map.get(node_id, False)
            comparable_steps += 1
            if is_hub:
                comparable_steps_hub += 1
            else:
                comparable_steps_nonhub += 1
            in_prev = node_id in prev_traversed
            if not in_prev:
                reuse_beyond_prev_count += 1
                if is_hub:
                    reuse_beyond_prev_hub_count += 1
                else:
                    reuse_beyond_prev_nonhub_count += 1

        total_steps = len(search_steps)
        reuse_beyond_prev_pct = (100.0 * reuse_beyond_prev_count / comparable_steps) if comparable_steps > 0 else 0.0
        reuse_beyond_prev_hub_pct = (100.0 * reuse_beyond_prev_hub_count / comparable_steps) if comparable_steps > 0 else 0.0
        reuse_beyond_prev_nonhub_pct = (100.0 * reuse_beyond_prev_nonhub_count / comparable_steps) if comparable_steps > 0 else 0.0

        per_query.append(
            {
                "query_id": query.get("query_id"),
                "total_steps": total_steps,
                "comparable_steps": comparable_steps,
                "comparable_steps_hub": comparable_steps_hub,
                "comparable_steps_nonhub": comparable_steps_nonhub,
                "reuse_beyond_prev_count": reuse_beyond_prev_count,
                "reuse_beyond_prev_pct": reuse_beyond_prev_pct,
                "reuse_beyond_prev_hub_count": reuse_beyond_prev_hub_count,
                "reuse_beyond_prev_hub_pct": reuse_beyond_prev_hub_pct,
                "reuse_beyond_prev_nonhub_count": reuse_beyond_prev_nonhub_count,
                "reuse_beyond_prev_nonhub_pct": reuse_beyond_prev_nonhub_pct,
            }
        )

    if per_query:
        avg_total_steps = float(np.mean([q["total_steps"] for q in per_query]))
        avg_comparable_steps = float(np.mean([q["comparable_steps"] for q in per_query]))
        avg_reuse_beyond_prev_count = float(np.mean([q["reuse_beyond_prev_count"] for q in per_query]))
        avg_reuse_beyond_prev_pct = float(np.mean([q["reuse_beyond_prev_pct"] for q in per_query]))
        avg_reuse_beyond_prev_hub_count = float(np.mean([q["reuse_beyond_prev_hub_count"] for q in per_query]))
        avg_reuse_beyond_prev_hub_pct = float(np.mean([q["reuse_beyond_prev_hub_pct"] for q in per_query]))
        avg_reuse_beyond_prev_nonhub_count = float(np.mean([q["reuse_beyond_prev_nonhub_count"] for q in per_query]))
        avg_reuse_beyond_prev_nonhub_pct = float(np.mean([q["reuse_beyond_prev_nonhub_pct"] for q in per_query]))
    else:
        avg_total_steps = 0.0
        avg_comparable_steps = 0.0
        avg_reuse_beyond_prev_count = 0.0
        avg_reuse_beyond_prev_pct = 0.0
        avg_reuse_beyond_prev_hub_count = 0.0
        avg_reuse_beyond_prev_hub_pct = 0.0
        avg_reuse_beyond_prev_nonhub_count = 0.0
        avg_reuse_beyond_prev_nonhub_pct = 0.0

    return {
        "overall": {
            "total_queries": len(per_query),
            "avg_total_steps": avg_total_steps,
            "avg_comparable_steps": avg_comparable_steps,
            "avg_reuse_beyond_prev_count": avg_reuse_beyond_prev_count,
            "avg_reuse_beyond_prev_pct": avg_reuse_beyond_prev_pct,
            "avg_reuse_beyond_prev_hub_count": avg_reuse_beyond_prev_hub_count,
            "avg_reuse_beyond_prev_hub_pct": avg_reuse_beyond_prev_hub_pct,
            "avg_reuse_beyond_prev_nonhub_count": avg_reuse_beyond_prev_nonhub_count,
            "avg_reuse_beyond_prev_nonhub_pct": avg_reuse_beyond_prev_nonhub_pct,
        },
        "queries": per_query,
    }


def bytes_trivial_offload() -> int:
    return 2*S_VERTEX_BYTES + S_DISTANCE_BYTES


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
    plt.bar(x - width / 2, trivial_vals, width, label="DM offload disabled")
    plt.bar(x + width / 2, reuse_vals, width, label="DM offload enabled")

    plt.xticks(x, labels, rotation=35, ha="right")
    plt.xlabel("Dataset")
    plt.ylabel("Memory used (GB)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_relative_bar(values: List[Tuple[str, float, float]], title: str, out_path: str) -> None:
    labels = [v[0] for v in values]
    trivial_vals = np.array([v[1] for v in values])
    reuse_vals = np.array([v[2] for v in values])

    scaled_trivial = np.ones_like(trivial_vals)
    scaled_reuse = np.zeros_like(reuse_vals)
    for idx, trivial_val in enumerate(trivial_vals):
        if trivial_val > 0:
            scaled_reuse[idx] = reuse_vals[idx] / trivial_val
        else:
            scaled_trivial[idx] = 0.0
            scaled_reuse[idx] = 0.0

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(max(10, len(labels) * 1.2), 8))
    bars_trivial = plt.bar(x - width / 2, scaled_trivial, width, label="DM offload disabled (baseline)")
    bars_reuse = plt.bar(x + width / 2, scaled_reuse, width, label="DM offload enabled")

    for bar, value in zip(bars_trivial, scaled_trivial):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}",
                 ha="center", va="bottom", fontsize=8)
    for bar, value in zip(bars_reuse, scaled_reuse):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}",
                 ha="center", va="bottom", fontsize=8)

    plt.xticks(x, labels, rotation=35, ha="right")
    plt.xlabel("Dataset")
    plt.ylabel("Relative memory (DM offload disabled = 1.0)")
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
    relative_points = []
    summary_rows = []

    for filename in pkl_files:
        label, _ = parse_dataset_label(filename)
        dim = parse_dimension_from_label(label)
        size_trivial = bytes_trivial_offload()
        size_reuse = bytes_reuse(dim)

        trace = load_trace(os.path.join(args.input_dir, filename))
        stats = summarize_trace(trace)
        reuse_stats = compute_reuse_stats(trace)
        reuse_stats["dataset"] = label

        dm_trivial = bytes_to_gb(stats["total_explored"] * size_trivial)
        dm_reuse = bytes_to_gb(stats["unique_explored"] * size_reuse)
        overall_points.append((label, dm_trivial, dm_reuse))

        dm_trivial_hub = bytes_to_gb(stats["total_explored_hub"] * size_trivial)
        dm_reuse_hubs_only = bytes_to_gb(stats["unique_explored_hub"] * size_reuse)
        relative_points.append((label, dm_trivial_hub, dm_reuse_hubs_only))

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

        reuse_stats_path = os.path.join(args.output_dir, f"{label}_reuse_stats.json")
        with open(reuse_stats_path, "w", encoding="utf-8") as f:
            json.dump(reuse_stats, f, indent=2)

        print(f"Dataset: {label}")
        print(f"  Total explored computations: {stats['total_explored']}")
        print(f"    Hubs: {stats['total_explored_hub']} | Non-hubs: {stats['total_explored_nonhub']}")
        print(f"  Total candidate list computations: {stats['total_traversed']}")
        print(f"    Hubs: {stats['total_traversed_hub']} | Non-hubs: {stats['total_traversed_nonhub']}")
        print(f"  Unique explored computations: {stats['unique_explored']}")
        print(f"    Hubs: {stats['unique_explored_hub']} | Non-hubs: {stats['unique_explored_nonhub']}")
        print(f"  Unique candidate list computations: {stats['unique_traversed']}")
        print(f"    Hubs: {stats['unique_traversed_hub']} | Non-hubs: {stats['unique_traversed_nonhub']}")

    plot_relative_bar(
        overall_points,
        "Relative DM offload disabled vs enabled (All explored computations)",
        os.path.join(args.output_dir, "dm_trivial_vs_reuse_all.png"),
    )
    plot_relative_bar(
        hub_points,
        "Relative DM offload disabled vs enabled (Explored hubs)",
        os.path.join(args.output_dir, "dm_trivial_vs_reuse_hubs.png"),
    )
    plot_relative_bar(
        nonhub_points,
        "Relative DM offload disabled vs enabled (Explored non-hubs)",
        os.path.join(args.output_dir, "dm_trivial_vs_reuse_nonhubs.png"),
    )
    plot_relative_bar(
        relative_points,
        "Relative DM offload disabled vs enabled (Hubs only)",
        os.path.join(args.output_dir, "dm_trivial_vs_reuse_relative_hubs_baseline.png"),
    )
    plot_bar(
        overall_points,
        "Absolute DM offload disabled vs enabled (All explored computations)",
        os.path.join(args.output_dir, "dm_offload_disabled_vs_enabled_absolute.png"),
    )

    summary_path = os.path.join(args.output_dir, "dm_trivial_reuse_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)


if __name__ == "__main__":
    main()
