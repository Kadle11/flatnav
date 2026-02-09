#!/usr/bin/env python3
"""
Plot hub profile metrics from JSON results.

Visualizes hub vs non-hub performance breakdown including:
- Distance computations and percentages
- Average computations per node type
- Estimated time split
- Estimated cycles split
- Node counts and per-node time estimates
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default paths
DEFAULT_PROFILE_PATH = "/root/metrics/hub_profiling"
DEFAULT_OUTPUT_PATH = "/root/metrics/hub_profiling/plots"


def load_profile(filepath: str) -> Dict[str, Any]:
    """Load a hub profile JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_distance_computation_breakdown(result: Dict[str, Any], dataset_name: str, output_path: str) -> None:
    """Plot distance computation breakdown: total, hub, non-hub."""
    if 'baseline' not in result:
        logging.warning(f"Missing 'baseline' key in {dataset_name}, skipping distance computation breakdown plot")
        return
    
    baseline = result['baseline']
    if 'distance_computations' not in baseline:
        logging.warning(f"Missing 'distance_computations' key in {dataset_name}, skipping distance computation breakdown plot")
        return
    
    if 'hub_nonhub_breakdown' not in baseline:
        logging.warning(f"Missing 'hub_nonhub_breakdown' key in {dataset_name}, skipping distance computation breakdown plot")
        return
    
    dist_comps = baseline['distance_computations']
    hub_nonhub = baseline['hub_nonhub_breakdown']
    
    required_keys = ['hub', 'nonhub', 'hub_percentage', 'nonhub_percentage']
    if not all(k in dist_comps for k in required_keys):
        logging.warning(f"Missing required distance computation keys in {dataset_name}, skipping plot")
        return
    
    required_hub_keys = ['hub_node_count', 'nonhub_node_count']
    if not all(k in hub_nonhub for k in required_hub_keys):
        logging.warning(f"Missing required hub breakdown keys in {dataset_name}, skipping plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart: Hub vs Non-Hub
    ax = axes[0]
    sizes = [dist_comps['hub'], dist_comps['nonhub']]
    labels = [f"Hub\n({dist_comps['hub_percentage']:.1f}%)", 
              f"Non-Hub\n({dist_comps['nonhub_percentage']:.1f}%)"]
    colors = ['#FF6B6B', '#4ECDC4']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f', startangle=90)
    ax.set_title(f"Distance Computations: Hub vs Non-Hub\n{dataset_name}")
    
    # Bar chart: Node counts and computations
    ax = axes[1]
    x = np.arange(2)
    width = 0.35
    
    node_counts = [hub_nonhub['hub_node_count'], hub_nonhub['nonhub_node_count']]
    comp_counts = [dist_comps['hub'], dist_comps['nonhub']]
    
    # Normalize for visualization (use log scale if needed)
    ax.bar(x - width/2, [c/1e6 for c in node_counts], width, label='Node Count', color='#95E1D3')
    ax.bar(x + width/2, [c/1e6 for c in comp_counts], width, label='Distance Computations (millions)', color='#F38181')
    
    ax.set_ylabel('Count (millions)')
    ax.set_title(f"Node Counts vs Distance Computations\n{dataset_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(['Hub Nodes', 'Non-Hub Nodes'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_path, f"{dataset_name}_distance_breakdown.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    logging.info(f"Saved {filepath}")
    plt.close()


def plot_per_node_averages(result: Dict[str, Any], dataset_name: str, output_path: str) -> None:
    """Plot average computations and time per node type."""
    if 'baseline' not in result:
        logging.warning(f"Missing 'baseline' key in {dataset_name}, skipping per-node averages plot")
        return
    
    baseline = result['baseline']
    if 'hub_nonhub_breakdown' not in baseline:
        logging.warning(f"Missing 'hub_nonhub_breakdown' key in {dataset_name}, skipping per-node averages plot")
        return
    
    hub_nonhub = baseline['hub_nonhub_breakdown']
    
    required_keys = ['avg_comps_per_hub_node', 'avg_comps_per_nonhub_node', 
                     'avg_time_per_hub_node_ms', 'avg_time_per_nonhub_node_ms']
    if not all(k in hub_nonhub for k in required_keys):
        logging.warning(f"Missing required per-node average keys in {dataset_name}, skipping plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart: Average computations per node
    ax = axes[0]
    categories = ['Hub Nodes', 'Non-Hub Nodes']
    avg_comps = [hub_nonhub['avg_comps_per_hub_node'], hub_nonhub['avg_comps_per_nonhub_node']]
    colors = ['#FF6B6B', '#4ECDC4']
    bars = ax.bar(categories, avg_comps, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Average Computations per Node', fontsize=11)
    ax.set_title(f"Average Distance Computations per Node\n{dataset_name}", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_comps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Bar chart: Average estimated time per node (ms)
    ax = axes[1]
    avg_times = [hub_nonhub['avg_time_per_hub_node_ms'], hub_nonhub['avg_time_per_nonhub_node_ms']]
    bars = ax.bar(categories, avg_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Average Estimated Time (ms)', fontsize=11)
    ax.set_title(f"Average Estimated Time per Node\n{dataset_name}", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_times):
        height = bar.get_height()
        label_text = f'{val:.6f}' if val < 0.001 else f'{val:.4f}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label_text, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(output_path, f"{dataset_name}_per_node_averages.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    logging.info(f"Saved {filepath}")
    plt.close()


def plot_estimated_time_split(result: Dict[str, Any], dataset_name: str, output_path: str) -> None:
    """Plot estimated time split between hub and non-hub computations."""
    if 'baseline' not in result:
        logging.warning(f"Missing 'baseline' key in {dataset_name}, skipping estimated time split plot")
        return
    
    baseline = result['baseline']
    if 'hub_nonhub_breakdown' not in baseline:
        logging.warning(f"Missing 'hub_nonhub_breakdown' key in {dataset_name}, skipping estimated time split plot")
        return
    
    hub_nonhub = baseline['hub_nonhub_breakdown']
    
    required_keys = ['estimated_time_ms_hubs', 'estimated_time_ms_nonhubs']
    if not all(k in hub_nonhub for k in required_keys):
        logging.warning(f"Missing required estimated time keys in {dataset_name}, skipping plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart: Time split
    ax = axes[0]
    times = [hub_nonhub['estimated_time_ms_hubs'], hub_nonhub['estimated_time_ms_nonhubs']]
    total_time = sum(times)
    if total_time == 0:
        logging.warning(f"Total estimated time is zero for {dataset_name}, skipping time split plot")
        plt.close()
        return
    
    percentages = [t / total_time * 100 for t in times]
    labels = [f"Hub\n({percentages[0]:.1f}%)\n{times[0]:.2f} ms", 
              f"Non-Hub\n({percentages[1]:.1f}%)\n{times[1]:.2f} ms"]
    colors = ['#FF6B6B', '#4ECDC4']
    ax.pie(times, labels=labels, colors=colors, autopct='%1.0f', startangle=90)
    ax.set_title(f"Estimated Total Time Split\n{dataset_name}")
    
    # Bar chart: Estimated time comparison
    ax = axes[1]
    x = np.arange(2)
    width = 0.6
    bars = ax.bar(x, times, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Estimated Time (ms)', fontsize=11)
    ax.set_title(f"Estimated Computation Time: Hub vs Non-Hub\n{dataset_name}", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Hub Nodes', 'Non-Hub Nodes'])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(output_path, f"{dataset_name}_estimated_time_split.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    logging.info(f"Saved {filepath}")
    plt.close()


def plot_estimated_cycles_split(result: Dict[str, Any], dataset_name: str, output_path: str) -> None:
    """Plot estimated cycles split between hub and non-hub computations."""
    if 'baseline' not in result:
        logging.warning(f"Missing 'baseline' key in {dataset_name}, skipping estimated cycles split plot")
        return
    
    baseline = result['baseline']
    if 'hub_nonhub_breakdown' not in baseline:
        logging.warning(f"Missing 'hub_nonhub_breakdown' key in {dataset_name}, skipping estimated cycles split plot")
        return
    
    hub_nonhub = baseline['hub_nonhub_breakdown']
    
    required_keys = ['estimated_cycles_hubs', 'estimated_cycles_nonhubs']
    if not all(k in hub_nonhub for k in required_keys):
        logging.warning(f"Missing required estimated cycles keys in {dataset_name}, skipping plot")
        return
    
    cycles_hub = hub_nonhub['estimated_cycles_hubs']
    cycles_nonhub = hub_nonhub['estimated_cycles_nonhubs']
    
    if cycles_hub == 0 and cycles_nonhub == 0:
        logging.warning(f"No cycle data for {dataset_name}, skipping cycles plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart: Cycles split
    ax = axes[0]
    cycles = [cycles_hub, cycles_nonhub]
    total_cycles = sum(cycles)
    percentages = [c / total_cycles * 100 for c in cycles]
    labels = [f"Hub\n({percentages[0]:.1f}%)\n{cycles[0]/1e9:.2f}B", 
              f"Non-Hub\n({percentages[1]:.1f}%)\n{cycles[1]/1e9:.2f}B"]
    colors = ['#FF6B6B', '#4ECDC4']
    ax.pie(cycles, labels=labels, colors=colors, autopct='%1.0f', startangle=90)
    ax.set_title(f"Estimated Total Cycles Split\n{dataset_name}")
    
    # Bar chart: Estimated cycles comparison
    ax = axes[1]
    x = np.arange(2)
    width = 0.6
    cycles_billions = [c / 1e9 for c in cycles]
    bars = ax.bar(x, cycles_billions, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Estimated Cycles (Billions)', fontsize=11)
    ax.set_title(f"Estimated Cycles: Hub vs Non-Hub\n{dataset_name}", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Hub Nodes', 'Non-Hub Nodes'])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, cycles_billions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}B', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(output_path, f"{dataset_name}_estimated_cycles_split.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    logging.info(f"Saved {filepath}")
    plt.close()


def plot_node_counts_comparison(result: Dict[str, Any], dataset_name: str, output_path: str) -> None:
    """Plot node counts for hub vs non-hub."""
    if 'baseline' not in result:
        logging.warning(f"Missing 'baseline' key in {dataset_name}, skipping node counts plot")
        return
    
    baseline = result['baseline']
    if 'hub_nonhub_breakdown' not in baseline:
        logging.warning(f"Missing 'hub_nonhub_breakdown' key in {dataset_name}, skipping node counts plot")
        return
    
    hub_nonhub = baseline['hub_nonhub_breakdown']
    
    required_keys = ['hub_node_count', 'nonhub_node_count']
    if not all(k in hub_nonhub for k in required_keys):
        logging.warning(f"Missing required node count keys in {dataset_name}, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(2)
    width = 0.6
    node_counts = [hub_nonhub['hub_node_count'], hub_nonhub['nonhub_node_count']]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax.bar(x, node_counts, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Node Count', fontsize=11)
    ax.set_title(f"Hub vs Non-Hub Node Counts\n{dataset_name}", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Hub Nodes', 'Non-Hub Nodes'])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels and percentages
    total_nodes = sum(node_counts)
    for bar, count in zip(bars, node_counts):
        height = bar.get_height()
        pct = count / total_nodes * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(output_path, f"{dataset_name}_node_counts.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    logging.info(f"Saved {filepath}")
    plt.close()


def plot_comprehensive_summary(result: Dict[str, Any], dataset_name: str, output_path: str) -> None:
    """Create a comprehensive summary figure with multiple subplots."""
    if 'baseline' not in result:
        logging.warning(f"Missing 'baseline' key in {dataset_name}, skipping comprehensive summary plot")
        return
    
    baseline = result['baseline']
    
    # Check for required sections
    if 'distance_computations' not in baseline:
        logging.warning(f"Missing 'distance_computations' key in {dataset_name}, skipping comprehensive summary plot")
        return
    
    if 'hub_nonhub_breakdown' not in baseline:
        logging.warning(f"Missing 'hub_nonhub_breakdown' key in {dataset_name}, skipping comprehensive summary plot")
        return
    
    if 'timing' not in baseline:
        logging.warning(f"Missing 'timing' key in {dataset_name}, skipping comprehensive summary plot")
        return
    
    if 'accuracy' not in baseline:
        logging.warning(f"Missing 'accuracy' key in {dataset_name}, skipping comprehensive summary plot")
        return
    
    dist_comps = baseline['distance_computations']
    hub_nonhub = baseline['hub_nonhub_breakdown']
    timing = baseline['timing']
    
    # Check for required keys in each section
    dist_keys = ['total', 'hub', 'nonhub', 'hub_percentage', 'nonhub_percentage']
    if not all(k in dist_comps for k in dist_keys):
        logging.warning(f"Missing required distance computation keys in {dataset_name}, skipping comprehensive summary plot")
        return
    
    hub_keys = ['hub_node_count', 'nonhub_node_count', 'avg_comps_per_hub_node', 'avg_comps_per_nonhub_node',
                'estimated_time_ms_hubs', 'estimated_time_ms_nonhubs', 'avg_time_per_hub_node_ms', 'avg_time_per_nonhub_node_ms']
    if not all(k in hub_nonhub for k in hub_keys):
        logging.warning(f"Missing required hub breakdown keys in {dataset_name}, skipping comprehensive summary plot")
        return
    
    timing_keys = ['total_search_time_ms']
    if not all(k in timing for k in timing_keys):
        logging.warning(f"Missing required timing keys in {dataset_name}, skipping comprehensive summary plot")
        return
    
    accuracy_keys = ['recall_at_k_percent']
    if not all(k in baseline['accuracy'] for k in accuracy_keys):
        logging.warning(f"Missing required accuracy keys in {dataset_name}, skipping comprehensive summary plot")
        return
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # 1. Distance computations pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    sizes = [dist_comps['hub'], dist_comps['nonhub']]
    colors = ['#FF6B6B', '#4ECDC4']
    ax1.pie(sizes, labels=['Hub', 'Non-Hub'], colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Distance Computations')
    
    # 2. Node counts pie chart
    ax2 = fig.add_subplot(gs[0, 1])
    node_sizes = [hub_nonhub['hub_node_count'], hub_nonhub['nonhub_node_count']]
    ax2.pie(node_sizes, labels=['Hub', 'Non-Hub'], colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Node Counts')
    
    # 3. Time split pie chart
    ax3 = fig.add_subplot(gs[0, 2])
    time_sizes = [hub_nonhub['estimated_time_ms_hubs'], hub_nonhub['estimated_time_ms_nonhubs']]
    ax3.pie(time_sizes, labels=['Hub', 'Non-Hub'], colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Estimated Time Split')
    
    # 4. Avg comps per node
    ax4 = fig.add_subplot(gs[1, 0])
    x = np.arange(2)
    avg_comps = [hub_nonhub['avg_comps_per_hub_node'], hub_nonhub['avg_comps_per_nonhub_node']]
    bars = ax4.bar(x, avg_comps, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Avg Computations')
    ax4.set_title('Avg Comps per Node')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Hub', 'Non-Hub'])
    ax4.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, avg_comps):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Avg time per node
    ax5 = fig.add_subplot(gs[1, 1])
    avg_times = [hub_nonhub['avg_time_per_hub_node_ms'], hub_nonhub['avg_time_per_nonhub_node_ms']]
    bars = ax5.bar(x, avg_times, color=colors, alpha=0.8, edgecolor='black')
    ax5.set_ylabel('Avg Time (ms)')
    ax5.set_title('Avg Time per Node')
    ax5.set_xticks(x)
    ax5.set_xticklabels(['Hub', 'Non-Hub'])
    ax5.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, avg_times):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.2e}', ha='center', va='bottom', fontsize=9)
    
    # 6. Total time and cycles split
    ax6 = fig.add_subplot(gs[1, 2])
    categories = ['Hub Time\n(ms)', 'NonHub Time\n(ms)']
    time_vals = [hub_nonhub['estimated_time_ms_hubs'], hub_nonhub['estimated_time_ms_nonhubs']]
    bars = ax6.bar(categories, time_vals, color=colors, alpha=0.8, edgecolor='black')
    ax6.set_ylabel('Time (ms)')
    ax6.set_title('Total Estimated Time')
    ax6.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, time_vals):
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 7. Summary text box
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    summary_text = f"""
    Dataset: {dataset_name}
    
    Distance Computations:
        Total: {dist_comps['total']:,}  |  Hub: {dist_comps['hub']:,} ({dist_comps['hub_percentage']:.1f}%)  |  Non-Hub: {dist_comps['nonhub']:,} ({dist_comps['nonhub_percentage']:.1f}%)
    
    Node Counts:
        Hub: {hub_nonhub['hub_node_count']:,}  |  Non-Hub: {hub_nonhub['nonhub_node_count']:,}
    
    Per-Node Averages:
        Hub: {hub_nonhub['avg_comps_per_hub_node']:.2f} comps/node, {hub_nonhub['avg_time_per_hub_node_ms']:.6f} ms/node
        Non-Hub: {hub_nonhub['avg_comps_per_nonhub_node']:.2f} comps/node, {hub_nonhub['avg_time_per_nonhub_node_ms']:.6f} ms/node
    
    Estimated Time Split:
        Hub: {hub_nonhub['estimated_time_ms_hubs']:.2f} ms  |  Non-Hub: {hub_nonhub['estimated_time_ms_nonhubs']:.2f} ms  |  Total: {timing['total_search_time_ms']:.2f} ms
    
    Recall@k: {baseline['accuracy']['recall_at_k_percent']:.2f}%
    """
    
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f"Hub Profile Summary: {dataset_name}", fontsize=14, fontweight='bold', y=0.995)
    
    filepath = os.path.join(output_path, f"{dataset_name}_comprehensive_summary.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    logging.info(f"Saved {filepath}")
    plt.close()


def plot_hub_profile(profile_path: str, output_path: str) -> None:
    """Load profile and generate all plots."""
    os.makedirs(output_path, exist_ok=True)
    
    result = load_profile(profile_path)
    dataset_name = result.get('dataset', 'unknown_dataset')
    
    logging.info(f"Plotting hub profile for {dataset_name}...")
    
    # Generate all plots, but continue if individual plots fail
    plot_functions = [
        ('distance computation breakdown', plot_distance_computation_breakdown),
        ('per-node averages', plot_per_node_averages),
        ('estimated time split', plot_estimated_time_split),
        ('estimated cycles split', plot_estimated_cycles_split),
        ('node counts comparison', plot_node_counts_comparison),
        ('comprehensive summary', plot_comprehensive_summary),
    ]
    
    successful_plots = 0
    for plot_name, plot_func in plot_functions:
        try:
            plot_func(result, dataset_name, output_path)
            successful_plots += 1
        except Exception as e:
            logging.warning(f"Failed to generate {plot_name} plot for {dataset_name}: {e}")
            continue
    
    if successful_plots == 0:
        logging.error(f"No plots could be generated for {dataset_name}")
    else:
        logging.info(f"Successfully generated {successful_plots}/{len(plot_functions)} plots for {dataset_name}")


def collect_per_node_averages(profile_dir: str) -> Dict[str, Dict[str, float]]:
    """Collect per-node average computations from all hub profile files."""
    datasets_data = {}
    
    if not os.path.exists(profile_dir):
        logging.error(f"Profile directory not found: {profile_dir}")
        return datasets_data
    
    profiles = sorted([f for f in os.listdir(profile_dir) if f.endswith('_hub_profile.json')])
    
    for profile_file in profiles:
        profile_path = os.path.join(profile_dir, profile_file)
        try:
            result = load_profile(profile_path)
            dataset_name = result.get('dataset', profile_file.replace('_hub_profile.json', ''))
            
            # Handle both old and new JSON formats
            if 'baseline' in result:
                # New format with baseline wrapper
                baseline = result['baseline']
                if 'hub_nonhub_breakdown' not in baseline:
                    logging.warning(f"Missing 'hub_nonhub_breakdown' key in {dataset_name} (new format), skipping")
                    continue
                hub_nonhub = baseline['hub_nonhub_breakdown']
            else:
                # Old format - check if hub_nonhub_breakdown exists at top level
                if 'hub_nonhub_breakdown' not in result:
                    logging.warning(f"Missing 'hub_nonhub_breakdown' key in {dataset_name} (old format), skipping")
                    continue
                hub_nonhub = result['hub_nonhub_breakdown']
            
            # Required computation keys
            required_keys = ['avg_comps_per_hub_node', 'avg_comps_per_nonhub_node']
            if not all(k in hub_nonhub for k in required_keys):
                logging.warning(f"Missing required per-node keys in {dataset_name}, skipping")
                continue

            # Optional time keys (may be missing)
            hub_time = hub_nonhub.get('avg_time_per_hub_node_ms')
            nonhub_time = hub_nonhub.get('avg_time_per_nonhub_node_ms')

            datasets_data[dataset_name] = {
                'avg_comps_per_hub_node': hub_nonhub['avg_comps_per_hub_node'],
                'avg_comps_per_nonhub_node': hub_nonhub['avg_comps_per_nonhub_node'],
                'avg_time_per_hub_node_ms': hub_time,
                'avg_time_per_nonhub_node_ms': nonhub_time,
            }
            
        except Exception as e:
            logging.warning(f"Error loading {profile_file}: {e}")
            continue
    
    return datasets_data


def plot_per_node_averages_across_datasets(profile_dir: str, output_path: str) -> None:
    """Create bar plots showing per-node averages across all datasets."""
    datasets_data = collect_per_node_averages(profile_dir)
    
    if not datasets_data:
        logging.error("No valid dataset data found for aggregate plots")
        return
    
    os.makedirs(output_path, exist_ok=True)
    
    # Prepare data for plotting
    dataset_names = list(datasets_data.keys())
    hub_averages = [datasets_data[name]['avg_comps_per_hub_node'] for name in dataset_names]
    nonhub_averages = [datasets_data[name]['avg_comps_per_nonhub_node'] for name in dataset_names]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Average computations per hub node
    bars1 = ax1.bar(dataset_names, hub_averages, color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Average Computations per Hub Node', fontsize=12)
    ax1.set_title('Average Distance Computations per Hub Node\nAcross Datasets', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dataset', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels for better readability
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars1, hub_averages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Average computations per non-hub node
    bars2 = ax2.bar(dataset_names, nonhub_averages, color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Average Computations per Non-Hub Node', fontsize=12)
    ax2.set_title('Average Distance Computations per Non-Hub Node\nAcross Datasets', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dataset', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels for better readability
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars2, nonhub_averages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the combined plot
    filepath = os.path.join(output_path, "per_node_averages_across_datasets.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    logging.info(f"Saved aggregate per-node averages plot: {filepath}")
    plt.close()
    
    # Also create separate plots for each metric
    # Hub nodes plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(dataset_names, hub_averages, color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Average Computations per Hub Node', fontsize=12)
    ax.set_title('Average Distance Computations per Hub Node Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, hub_averages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(output_path, "hub_node_averages_across_datasets.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    logging.info(f"Saved hub node averages plot: {filepath}")
    plt.close()
    
    # Non-hub nodes plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(dataset_names, nonhub_averages, color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Average Computations per Non-Hub Node', fontsize=12)
    ax.set_title('Average Distance Computations per Non-Hub Node Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, nonhub_averages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(output_path, "nonhub_node_averages_across_datasets.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    logging.info(f"Saved non-hub node averages plot: {filepath}")
    plt.close()


def plot_combined_per_node_metrics_across_datasets(profile_dir: str, output_path: str) -> None:
    """Create a combined bar plot showing hub (red) and non-hub (blue) averages
    for both distance computations per node and average time per node across datasets.
    """
    datasets_data = collect_per_node_averages(profile_dir)
    if not datasets_data:
        logging.error("No valid dataset data found for combined aggregate plot")
        return

    os.makedirs(output_path, exist_ok=True)

    dataset_names = list(datasets_data.keys())
    hub_comps = [datasets_data[name]['avg_comps_per_hub_node'] for name in dataset_names]
    nonhub_comps = [datasets_data[name]['avg_comps_per_nonhub_node'] for name in dataset_names]
    hub_times = [datasets_data[name].get('avg_time_per_hub_node_ms', None) for name in dataset_names]
    nonhub_times = [datasets_data[name].get('avg_time_per_nonhub_node_ms', None) for name in dataset_names]

    # Convert missing times to np.nan for plotting
    hub_times = [np.nan if t is None else t for t in hub_times]
    nonhub_times = [np.nan if t is None else t for t in nonhub_times]

    # Create combined figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    x = np.arange(len(dataset_names))
    width = 0.35

    # Colors: hubs red, non-hubs blue
    hub_color = '#FF6B6B'
    nonhub_color = '#4E79A7'

    # Left: average computations per node
    bars_hub = ax1.bar(x - width/2, hub_comps, width, label='Hubs', color=hub_color, edgecolor='black')
    bars_nonhub = ax1.bar(x + width/2, nonhub_comps, width, label='Non-Hubs', color=nonhub_color, edgecolor='black')
    ax1.set_ylabel('Average Distance Computations per Node', fontsize=12)
    ax1.set_title('Avg Computations per Node Across Datasets', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()

    # Annotate computation bars
    for bar, val in zip(bars_hub, hub_comps):
        ax1.text(bar.get_x() + bar.get_width()/2., val, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars_nonhub, nonhub_comps):
        ax1.text(bar.get_x() + bar.get_width()/2., val, f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # Right: average time per node (ms)
    bars_hub_t = ax2.bar(x - width/2, hub_times, width, label='Hubs', color=hub_color, edgecolor='black')
    bars_nonhub_t = ax2.bar(x + width/2, nonhub_times, width, label='Non-Hubs', color=nonhub_color, edgecolor='black')
    ax2.set_ylabel('Average Time per Node (ms)', fontsize=12)
    ax2.set_title('Avg Time per Node Across Datasets', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()

    # Annotate time bars only for non-NaN values
    for bar, val in zip(bars_hub_t, hub_times):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2., val, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars_nonhub_t, nonhub_times):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2., val, f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    filepath = os.path.join(output_path, "combined_per_node_metrics_across_datasets.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    logging.info(f"Saved combined per-node metrics plot: {filepath}")
    plt.close()


def plot_aggregates_file(aggregates_path: str, output_path: str) -> None:
    """Load aggregates.json and create two grouped-bar plots (comps and time)
    in the same figure (two subplots). Hubs are red, non-hubs blue.
    """
    if not os.path.exists(aggregates_path):
        logging.warning(f"Aggregates file not found: {aggregates_path}, skipping")
        return

    with open(aggregates_path, 'r') as f:
        data = json.load(f)

    if not data:
        logging.warning(f"Aggregates file is empty: {aggregates_path}, skipping")
        return

    # Sort datasets for consistent ordering
    dataset_names = sorted(data.keys())

    hub_comps = [data[name].get('avg_comps_per_hub_node', np.nan) for name in dataset_names]
    nonhub_comps = [data[name].get('avg_comps_per_nonhub_node', np.nan) for name in dataset_names]
    hub_times = [data[name].get('avg_time_per_hub_node_ms', np.nan) for name in dataset_names]
    nonhub_times = [data[name].get('avg_time_per_nonhub_node_ms', np.nan) for name in dataset_names]

    os.makedirs(output_path, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    x = np.arange(len(dataset_names))
    width = 0.35

    hub_color = '#d62728'  # red
    nonhub_color = '#1f77b4'  # blue

    # Convert to numpy arrays for NaN handling
    hub_comps_arr = np.array(hub_comps, dtype=float)
    nonhub_comps_arr = np.array(nonhub_comps, dtype=float)
    hub_times_arr = np.array(hub_times, dtype=float)
    nonhub_times_arr = np.array(nonhub_times, dtype=float)

    # Left subplot: comps
    ax = axes[0]
    bars_hub = ax.bar(x - width/2, np.nan_to_num(hub_comps_arr, nan=0.0), width, label='Hubs', color=hub_color, edgecolor='black')
    bars_nonhub = ax.bar(x + width/2, np.nan_to_num(nonhub_comps_arr, nan=0.0), width, label='Non-Hubs', color=nonhub_color, edgecolor='black')
    ax.set_ylabel('Average Distance Computations per Node')
    ax.set_title('Average Computations per Node Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # Mark missing comps with hatch and faded alpha, annotate 'n/a'
    for i, (bar_h, val_h, bar_nh, val_nh) in enumerate(zip(bars_hub, hub_comps_arr, bars_nonhub, nonhub_comps_arr)):
        if np.isnan(val_h):
            bar_h.set_alpha(0.3); bar_h.set_hatch('///')
            ax.text(bar_h.get_x() + bar_h.get_width()/2., 0.01, 'n/a', ha='center', va='bottom', fontsize=8, color='black')
        else:
            ax.text(bar_h.get_x() + bar_h.get_width()/2., val_h, f'{val_h:.1f}', ha='center', va='bottom', fontsize=9)
        if np.isnan(val_nh):
            bar_nh.set_alpha(0.3); bar_nh.set_hatch('\\\\')
            ax.text(bar_nh.get_x() + bar_nh.get_width()/2., 0.01, 'n/a', ha='center', va='bottom', fontsize=8, color='black')
        else:
            ax.text(bar_nh.get_x() + bar_nh.get_width()/2., val_nh, f'{val_nh:.1f}', ha='center', va='bottom', fontsize=9)

    # Right subplot: times
    ax = axes[1]
    bars_hub_t = ax.bar(x - width/2, np.nan_to_num(hub_times_arr, nan=0.0), width, label='Hubs', color=hub_color, edgecolor='black')
    bars_nonhub_t = ax.bar(x + width/2, np.nan_to_num(nonhub_times_arr, nan=0.0), width, label='Non-Hubs', color=nonhub_color, edgecolor='black')
    ax.set_ylabel('Average Time per Node (ms)')
    ax.set_title('Average Time per Node Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    for bar, val in zip(bars_hub_t, hub_times_arr):
        if np.isnan(val):
            bar.set_alpha(0.3); bar.set_hatch('///')
            ax.text(bar.get_x() + bar.get_width()/2., 0.0, 'n/a', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(bar.get_x() + bar.get_width()/2., val, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars_nonhub_t, nonhub_times_arr):
        if np.isnan(val):
            bar.set_alpha(0.3); bar.set_hatch('\\\\')
            ax.text(bar.get_x() + bar.get_width()/2., 0.0, 'n/a', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(bar.get_x() + bar.get_width()/2., val, f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    outpath = os.path.join(output_path, 'aggregates_comps_and_time.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    logging.info(f"Saved aggregates plot: {outpath}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot hub profile metrics")
    parser.add_argument(
        "--profile-path",
        type=str,
        help="Path to hub profile JSON file"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Output directory for plots"
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=DEFAULT_PROFILE_PATH,
        help="Directory containing hub profile JSON files"
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only generate aggregate plots across datasets, skip individual plots"
    )
    
    args = parser.parse_args()
    
    # If specific profile path provided, plot that one
    if args.profile_path:
        if not os.path.exists(args.profile_path):
            logging.error(f"Profile file not found: {args.profile_path}")
            sys.exit(1)
        plot_hub_profile(args.profile_path, args.output_path)
    else:
        # Otherwise, find all hub profile files in directory
        profile_dir = args.profile_dir
        if not os.path.exists(profile_dir):
            logging.error(f"Profile directory not found: {profile_dir}")
            sys.exit(1)
        
        profiles = sorted([f for f in os.listdir(profile_dir) if f.endswith('_hub_profile.json')])
        if not profiles:
            logging.error(f"No hub profile JSON files found in {profile_dir}")
            sys.exit(1)
        
        # Generate aggregate plots across datasets
        plot_per_node_averages_across_datasets(profile_dir, args.output_path)
        # Also produce the combined per-node metrics plot (computations + time)
        plot_combined_per_node_metrics_across_datasets(profile_dir, args.output_path)
        # If aggregates.json exists, create dedicated aggregates plots (comps + time in same figure)
        aggregates_file = os.path.join(profile_dir, 'aggregates.json')
        plot_aggregates_file(aggregates_file, args.output_path)
        
        if not args.aggregate_only:
            # Generate individual plots for each dataset
            for profile_file in profiles:
                profile_path = os.path.join(profile_dir, profile_file)
                plot_hub_profile(profile_path, args.output_path)


if __name__ == "__main__":
    main()
