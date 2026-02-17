#!/usr/bin/env python3
"""
Script to plot horizontal bar charts for dm_trivial_reuse summary data.
Generates plots comparing dm_trivial_offload vs dm_fetch for hubs and nonhubs separately.
Uses colorblind-safe colors and legible fonts/legends.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# Colorblind-safe palette (Okabe-Ito colors)
# These colors are distinguishable for all types of color blindness
COLORS = {
    'dm_trivial_offload': '#0173B2',  # Blue
    'dm_fetch': '#DE8F05',            # Orange
}

# Alternative diverse palette (more options if needed)
COLORS_ALT = {
    'dm_trivial_offload': '#1f77b4',  # Blue
    'dm_fetch': '#ff7f0e',            # Orange
}


def load_summary(json_path):
    """Load the dm_trivial_reuse summary JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def prepare_data(summary_data):
    """Prepare data for plotting."""
    datasets = [item['dataset'] for item in summary_data]
    
    # Extract hubs data
    hubs_trivial = np.array([item['dm_trivial_offload_hubs_gb'] for item in summary_data])
    hubs_reuse = np.array([item['dm_reuse_hubs_gb'] for item in summary_data])
    
    # Extract nonhubs data
    nonhubs_trivial = np.array([item['dm_trivial_offload_nonhubs_gb'] for item in summary_data])
    nonhubs_reuse = np.array([item['dm_reuse_nonhubs_gb'] for item in summary_data])
    
    return datasets, hubs_trivial, hubs_reuse, nonhubs_trivial, nonhubs_reuse


def create_relative_combined_plot(datasets, hubs_trivial, hubs_reuse, nonhubs_trivial, nonhubs_reuse, output_path):
    """Create a normalized relative plot with all data on one plot.
    
    Normalization:
    - hubs: normalized by dm_trivial_offload_hubs_gb
    - nonhubs: normalized by dm_trivial_offload_nonhubs_gb
    
    Y-axis labels indicate hubs vs nonhubs for each dataset.
    """
    
    n_datasets = len(datasets)
    # 2 entries per dataset (hubs, nonhubs)
    n_total_entries = n_datasets * 2
    
    fig, ax = plt.subplots(figsize=(14, max(8, n_total_entries * 0.4)))
    
    y_positions = np.arange(n_total_entries)
    bar_height = 0.35
    
    # Normalize by dm_trivial_offload baseline
    hubs_trivial_norm = np.ones_like(hubs_trivial)
    hubs_reuse_norm = np.divide(hubs_reuse, hubs_trivial, where=hubs_trivial != 0, 
                                 out=np.zeros_like(hubs_reuse, dtype=float))
    
    nonhubs_trivial_norm = np.ones_like(nonhubs_trivial)
    nonhubs_reuse_norm = np.divide(nonhubs_reuse, nonhubs_trivial, where=nonhubs_trivial != 0,
                                    out=np.zeros_like(nonhubs_reuse, dtype=float))
    
    # Interleave hubs and nonhubs data
    trivial_norm_all = []
    reuse_norm_all = []
    y_labels = []
    
    for i, dataset in enumerate(datasets):
        # Hubs entry
        trivial_norm_all.append(hubs_trivial_norm[i])
        reuse_norm_all.append(hubs_reuse_norm[i])
        y_labels.append(f'{dataset}\n(hubs)')
        
        # Nonhubs entry
        trivial_norm_all.append(nonhubs_trivial_norm[i])
        reuse_norm_all.append(nonhubs_reuse_norm[i])
        y_labels.append(f'{dataset}\n(nonhubs)')
    
    trivial_norm_all = np.array(trivial_norm_all)
    reuse_norm_all = np.array(reuse_norm_all)
    
    # Create horizontal bars (dm_fetch first for legend order)
    bars_b = ax.barh(y_positions + bar_height/2, reuse_norm_all, bar_height,
                     label='dm_fetch', color=COLORS['dm_fetch'],
                     edgecolor='black', linewidth=0.5)
    bars_a = ax.barh(y_positions - bar_height/2, trivial_norm_all, bar_height,
                     label='dm_trivial_offload', color=COLORS['dm_trivial_offload'],
                     edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10, fontweight='bold')
    ax.set_xlabel('Relative Memory (normalized to dm_trivial_offload)', fontsize=12, fontweight='bold')
    ax.set_title('Memory Usage Comparison: Hubs vs Non-Hubs (Relative)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=14, loc='lower right', framealpha=0.95, edgecolor='black')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.axvline(x=1.0, color='black', linestyle=':', linewidth=2, alpha=0.7, label='Baseline')
    
    # Add value labels
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            width = bar.get_width()
            if width > 0:
                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{value:.2f}x', va='center', fontsize=9, fontweight='bold')
    
    add_value_labels(bars_b, reuse_norm_all)
    add_value_labels(bars_a, trivial_norm_all)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved relative plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create horizontal bar plots for dm_trivial_reuse summary data'
    )
    parser.add_argument('--input-file', type=str, 
                       default='../metrics/search_step_trace/plots/dm_trivial_reuse_summary.json',
                       help='Path to the dm_trivial_reuse_summary.json file')
    parser.add_argument('--output-dir', type=str,
                       default='../metrics/search_step_trace/plots',
                       help='Directory to save output plots')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    print(f"Loading data from {args.input_file}")
    summary_data = load_summary(args.input_file)
    datasets, hubs_trivial, hubs_reuse, nonhubs_trivial, nonhubs_reuse = prepare_data(summary_data)
    
    # Create relative normalized plot
    print("Creating relative normalized plot...")
    create_relative_combined_plot(
        datasets, hubs_trivial, hubs_reuse, nonhubs_trivial, nonhubs_reuse,
        output_dir / 'dm_trivial_reuse_relative_horizontal.png'
    )
    
    print("Done!")


if __name__ == '__main__':
    main()
