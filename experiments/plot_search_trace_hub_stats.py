import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

def load_trace(trace_path):
    with open(trace_path, 'r') as f:
        trace = json.load(f)
    return trace

def get_hub_set(trace):
    # Find all nodes marked as hub in any level, any query
    hub_set = set()
    for q in trace['queries']:
        for level in q['levels']:
            for node in level['nodes']:
                if node['is_hub']:
                    hub_set.add(node['node_id'])
    return hub_set

def plot_per_level(trace, out_prefix):
    num_levels = max(q['num_levels'] for q in trace['queries'])
    hub_set = get_hub_set(trace)
    
    # For each level, accumulate stats
    unique_hubs = []
    unique_nonhubs = []
    total_hub_accesses = []
    total_nonhub_accesses = []
    # For 3rd plot: ratio of nodes accessed >1x per level
    hub_access_count = defaultdict(lambda: [0]*num_levels)  # node_id -> [count per level]
    nonhub_access_count = defaultdict(lambda: [0]*num_levels)
    for q in trace['queries']:
        for level_idx, level in enumerate(q['levels']):
            for node in level['nodes']:
                if node['is_hub']:
                    hub_access_count[node['node_id']][level_idx] += 1
                else:
                    nonhub_access_count[node['node_id']][level_idx] += 1
    
    for level_idx in range(num_levels):
        hubs = set()
        nonhubs = set()
        hub_accesses = 0
        nonhub_accesses = 0
        for q in trace['queries']:
            if level_idx < len(q['levels']):
                for node in q['levels'][level_idx]['nodes']:
                    if node['is_hub']:
                        hubs.add(node['node_id'])
                        hub_accesses += 1
                    else:
                        nonhubs.add(node['node_id'])
                        nonhub_accesses += 1
        unique_hubs.append(len(hubs))
        unique_nonhubs.append(len(nonhubs))
        total_hub_accesses.append(hub_accesses)
        total_nonhub_accesses.append(nonhub_accesses)
    
    x = np.arange(num_levels)
    width = 0.2
    
    # 1st plot
    plt.figure(figsize=(16,8))
    bars1 = plt.bar(x - 1.5*width, unique_hubs, width, label='Unique Hubs')
    bars2 = plt.bar(x - 0.5*width, unique_nonhubs, width, label='Unique Non-Hubs')
    bars3 = plt.bar(x + 0.5*width, total_hub_accesses, width, label='Total Hub Accesses')
    bars4 = plt.bar(x + 1.5*width, total_nonhub_accesses, width, label='Total Non-Hub Accesses')
    plt.xlabel('Level')
    plt.ylabel('Count')
    plt.title('Node Access Stats per Level (All Queries)')
    plt.legend()
    # Add value labels
    for bars, values in zip([bars1, bars2, bars3, bars4], [unique_hubs, unique_nonhubs, total_hub_accesses, total_nonhub_accesses]):
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=60)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_per_level.png', dpi=200)
    plt.close()

    # 2nd plot: ratio of nodes accessed >1x per level (hubs/non-hubs)
    hub_multi_access_ratio = []
    nonhub_multi_access_ratio = []
    for level_idx in range(num_levels):
        hub_multi = sum(1 for node_id in hub_access_count if hub_access_count[node_id][level_idx] > 1)
        hub_total = sum(1 for node_id in hub_access_count if hub_access_count[node_id][level_idx] > 0)
        nonhub_multi = sum(1 for node_id in nonhub_access_count if nonhub_access_count[node_id][level_idx] > 1)
        nonhub_total = sum(1 for node_id in nonhub_access_count if nonhub_access_count[node_id][level_idx] > 0)
        hub_ratio = (100.0 * hub_multi / hub_total) if hub_total > 0 else 0.0
        nonhub_ratio = (100.0 * nonhub_multi / nonhub_total) if nonhub_total > 0 else 0.0
        hub_multi_access_ratio.append(hub_ratio)
        nonhub_multi_access_ratio.append(nonhub_ratio)
    
    plt.figure(figsize=(16,8))
    bars1 = plt.bar(x - 0.15, hub_multi_access_ratio, width, label='Hubs >1x Access Ratio')
    bars2 = plt.bar(x + 0.15, nonhub_multi_access_ratio, width, label='Non-Hubs >1x Access Ratio')
    plt.xlabel('Level')
    plt.ylabel('Percentage (%)')
    plt.title('Ratio of Nodes Accessed >1x per Level')
    plt.legend()
    # Add value labels
    for bars, values in zip([bars1, bars2], [hub_multi_access_ratio, nonhub_multi_access_ratio]):
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=60)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_per_level_multi_access_ratio.png', dpi=200)
    plt.close()

    # 3rd plot: previously accessed nodes per level (counts)
    prev_hubs = set()
    prev_nonhubs = set()
    count_unique_hubs_prev = []
    count_unique_nonhubs_prev = []
    count_hub_accesses_prev = []
    count_nonhub_accesses_prev = []
    
    for level_idx in range(num_levels):
        hubs = set()
        nonhubs = set()
        hub_accesses = 0
        nonhub_accesses = 0
        hub_accesses_prev = 0
        nonhub_accesses_prev = 0
        for q in trace['queries']:
            if level_idx < len(q['levels']):
                for node in q['levels'][level_idx]['nodes']:
                    if node['is_hub']:
                        hubs.add(node['node_id'])
                        hub_accesses += 1
                        if level_idx > 0 and node['node_id'] in prev_hubs:
                            hub_accesses_prev += 1
                    else:
                        nonhubs.add(node['node_id'])
                        nonhub_accesses += 1
                        if level_idx > 0 and node['node_id'] in prev_nonhubs:
                            nonhub_accesses_prev += 1
        # Only compare to previous level, not all previous
        if level_idx > 0:
            count_unique_hubs_prev.append(len(hubs & prev_hubs))
            count_unique_nonhubs_prev.append(len(nonhubs & prev_nonhubs))
        else:
            count_unique_hubs_prev.append(0)
            count_unique_nonhubs_prev.append(0)
        count_hub_accesses_prev.append(hub_accesses_prev)
        count_nonhub_accesses_prev.append(nonhub_accesses_prev)
        prev_hubs = hubs
        prev_nonhubs = nonhubs
    
    # Plot counts - use matplotlib's default colors
    plt.figure(figsize=(16,8))
    bars1 = plt.bar(x - 1.5*width, count_unique_hubs_prev, width, label='Unique Hubs Previously Accessed')
    bars2 = plt.bar(x - 0.5*width, count_unique_nonhubs_prev, width, label='Unique Non-Hubs Previously Accessed')
    bars3 = plt.bar(x + 0.5*width, count_hub_accesses_prev, width, label='Hub Accesses Previously Accessed')
    bars4 = plt.bar(x + 1.5*width, count_nonhub_accesses_prev, width, label='Non-Hub Accesses Previously Accessed')
    plt.xlabel('Level')
    plt.ylabel('Count')
    plt.title('Previously Accessed Node Stats per Level')
    plt.legend()
    for bars, values in zip([bars1, bars2, bars3, bars4], [count_unique_hubs_prev, count_unique_nonhubs_prev, count_hub_accesses_prev, count_nonhub_accesses_prev]):
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=60)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_per_level_prev_access.png', dpi=200)
    plt.close()

    # 4th plot: previously accessed nodes per level (percentages)
    # We'll recalculate hubs/nonhubs and use count_*_prev from above
    pct_unique_hubs_prev = []
    pct_unique_nonhubs_prev = []
    pct_hub_accesses_prev = []
    pct_nonhub_accesses_prev = []
    
    for level_idx in range(num_levels):
        hubs = set()
        nonhubs = set()
        hub_accesses = 0
        nonhub_accesses = 0
        for q in trace['queries']:
            if level_idx < len(q['levels']):
                for node in q['levels'][level_idx]['nodes']:
                    if node['is_hub']:
                        hubs.add(node['node_id'])
                        hub_accesses += 1
                    else:
                        nonhubs.add(node['node_id'])
                        nonhub_accesses += 1
        if level_idx > 0:
            pct_unique_hubs_prev.append(100.0 * count_unique_hubs_prev[level_idx] / len(hubs) if len(hubs) > 0 else 0.0)
            pct_unique_nonhubs_prev.append(100.0 * count_unique_nonhubs_prev[level_idx] / len(nonhubs) if len(nonhubs) > 0 else 0.0)
        else:
            pct_unique_hubs_prev.append(0.0)
            pct_unique_nonhubs_prev.append(0.0)
        pct_hub_accesses_prev.append(100.0 * count_hub_accesses_prev[level_idx] / hub_accesses if hub_accesses > 0 else 0.0)
        pct_nonhub_accesses_prev.append(100.0 * count_nonhub_accesses_prev[level_idx] / nonhub_accesses if nonhub_accesses > 0 else 0.0)
    
    # Get colors from dummy plot
    plt.figure(figsize=(16,8))
    dummy_bars1 = plt.bar(x - 1.5*width, [0]*num_levels, width)
    dummy_bars2 = plt.bar(x - 0.5*width, [0]*num_levels, width)
    dummy_bars3 = plt.bar(x + 0.5*width, [0]*num_levels, width)
    dummy_bars4 = plt.bar(x + 1.5*width, [0]*num_levels, width)
    color1 = dummy_bars1.patches[0].get_facecolor()
    color2 = dummy_bars2.patches[0].get_facecolor()
    color3 = dummy_bars3.patches[0].get_facecolor()
    color4 = dummy_bars4.patches[0].get_facecolor()
    plt.clf()
    
    bars1 = plt.bar(x - 1.5*width, pct_unique_hubs_prev, width, label='Unique Hubs Previously Accessed (%)', color=color1)
    bars2 = plt.bar(x - 0.5*width, pct_unique_nonhubs_prev, width, label='Unique Non-Hubs Previously Accessed (%)', color=color2)
    bars3 = plt.bar(x + 0.5*width, pct_hub_accesses_prev, width, label='Hub Accesses Previously Accessed (%)', color=color3)
    bars4 = plt.bar(x + 1.5*width, pct_nonhub_accesses_prev, width, label='Non-Hub Accesses Previously Accessed (%)', color=color4)
    plt.xlabel('Level')
    plt.ylabel('Percentage (%)')
    plt.title('Previously Accessed Node Stats per Level (Percentage)')
    plt.legend()
    for bars, values in zip([bars1, bars2, bars3, bars4], [pct_unique_hubs_prev, pct_unique_nonhubs_prev, pct_hub_accesses_prev, pct_nonhub_accesses_prev]):
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=60)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_per_level_prev_access_pct.png', dpi=200)
    plt.close()

def plot_per_window(trace, window_size, out_prefix):
    # First, flatten all queries into windows
    query_windows = []  # List of list-of-windows, one per query
    max_windows = 0
    for q in trace['queries']:
        flat_nodes = []
        for level in q['levels']:
            for node in level['nodes']:
                flat_nodes.append((node['node_id'], node['is_hub']))
        num_windows = (len(flat_nodes) + window_size - 1) // window_size
        max_windows = max(max_windows, num_windows)
        windows = []
        for w in range(num_windows):
            start = w * window_size
            end = min((w+1)*window_size, len(flat_nodes))
            windows.append(flat_nodes[start:end])
        query_windows.append(windows)
    
    # Compute stats for each window
    unique_hubs = []
    unique_nonhubs = []
    total_hub_accesses = []
    total_nonhub_accesses = []
    hub_multi_access_ratio = []
    nonhub_multi_access_ratio = []
    
    for w in range(max_windows):
        merged = []
        for qw in query_windows:
            if w < len(qw):
                merged.extend(qw[w])
        hubs = set()
        nonhubs = set()
        hub_accesses = 0
        nonhub_accesses = 0
        hub_access_count = defaultdict(int)
        nonhub_access_count = defaultdict(int)
        for node_id, is_hub in merged:
            if is_hub:
                hubs.add(node_id)
                hub_accesses += 1
                hub_access_count[node_id] += 1
            else:
                nonhubs.add(node_id)
                nonhub_accesses += 1
                nonhub_access_count[node_id] += 1
        unique_hubs.append(len(hubs))
        unique_nonhubs.append(len(nonhubs))
        total_hub_accesses.append(hub_accesses)
        total_nonhub_accesses.append(nonhub_accesses)
        # Ratio of nodes accessed >1x in window
        hub_multi = sum(1 for node_id in hub_access_count if hub_access_count[node_id] > 1)
        hub_total = sum(1 for node_id in hub_access_count if hub_access_count[node_id] > 0)
        nonhub_multi = sum(1 for node_id in nonhub_access_count if nonhub_access_count[node_id] > 1)
        nonhub_total = sum(1 for node_id in nonhub_access_count if nonhub_access_count[node_id] > 0)
        hub_ratio = (100.0 * hub_multi / hub_total) if hub_total > 0 else 0.0
        nonhub_ratio = (100.0 * nonhub_multi / nonhub_total) if nonhub_total > 0 else 0.0
        hub_multi_access_ratio.append(hub_ratio)
        nonhub_multi_access_ratio.append(nonhub_ratio)
    
    x = np.arange(max_windows)
    width = 0.2
    
    # 1st plot per-window
    plt.figure(figsize=(16,8))
    bars1 = plt.bar(x - 1.5*width, unique_hubs, width, label='Unique Hubs')
    bars2 = plt.bar(x - 0.5*width, unique_nonhubs, width, label='Unique Non-Hubs')
    bars3 = plt.bar(x + 0.5*width, total_hub_accesses, width, label='Total Hub Accesses')
    bars4 = plt.bar(x + 1.5*width, total_nonhub_accesses, width, label='Total Non-Hub Accesses')
    plt.xlabel(f'Window Index (size={window_size} per query)')
    plt.ylabel('Count')
    plt.title('Node Access Stats per Window (Merged Across Queries)')
    plt.legend()
    # Add value labels
    for bars, values in zip([bars1, bars2, bars3, bars4], [unique_hubs, unique_nonhubs, total_hub_accesses, total_nonhub_accesses]):
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=60)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_per_window.png', dpi=200)
    plt.close()

    # 2nd plot: ratio of nodes accessed >1x per window (hubs/non-hubs)
    plt.figure(figsize=(16,8))
    bars1 = plt.bar(x - 0.15, hub_multi_access_ratio, width, label='Hubs >1x Access Ratio')
    bars2 = plt.bar(x + 0.15, nonhub_multi_access_ratio, width, label='Non-Hubs >1x Access Ratio')
    plt.xlabel(f'Window Index (size={window_size} per query)')
    plt.ylabel('Percentage (%)')
    plt.title('Ratio of Nodes Accessed >1x per Window')
    plt.legend()
    # Add value labels
    for bars, values in zip([bars1, bars2], [hub_multi_access_ratio, nonhub_multi_access_ratio]):
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=60)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_per_window_multi_access_ratio.png', dpi=200)
    plt.close()

    # 3rd plot: previously accessed nodes per window (counts)
    prev_hubs = set()
    prev_nonhubs = set()
    count_unique_hubs_prev = []
    count_unique_nonhubs_prev = []
    count_hub_accesses_prev = []
    count_nonhub_accesses_prev = []
    
    for w in range(max_windows):
        hubs = set()
        nonhubs = set()
        hub_accesses = 0
        nonhub_accesses = 0
        hub_accesses_prev = 0
        nonhub_accesses_prev = 0
        for qw in query_windows:
            if w < len(qw):
                for node_id, is_hub in qw[w]:
                    if is_hub:
                        hubs.add(node_id)
                        hub_accesses += 1
                        if w > 0 and node_id in prev_hubs:
                            hub_accesses_prev += 1
                    else:
                        nonhubs.add(node_id)
                        nonhub_accesses += 1
                        if w > 0 and node_id in prev_nonhubs:
                            nonhub_accesses_prev += 1
        if w > 0:
            count_unique_hubs_prev.append(len(hubs & prev_hubs))
            count_unique_nonhubs_prev.append(len(nonhubs & prev_nonhubs))
        else:
            count_unique_hubs_prev.append(0)
            count_unique_nonhubs_prev.append(0)
        count_hub_accesses_prev.append(hub_accesses_prev)
        count_nonhub_accesses_prev.append(nonhub_accesses_prev)
        prev_hubs = hubs
        prev_nonhubs = nonhubs
    
    plt.figure(figsize=(16,8))
    bars1 = plt.bar(x - 1.5*width, count_unique_hubs_prev, width, label='Unique Hubs Previously Accessed')
    bars2 = plt.bar(x - 0.5*width, count_unique_nonhubs_prev, width, label='Unique Non-Hubs Previously Accessed')
    bars3 = plt.bar(x + 0.5*width, count_hub_accesses_prev, width, label='Hub Accesses Previously Accessed')
    bars4 = plt.bar(x + 1.5*width, count_nonhub_accesses_prev, width, label='Non-Hub Accesses Previously Accessed')
    plt.xlabel(f'Window Index (size={window_size} per query)')
    plt.ylabel('Count')
    plt.title('Previously Accessed Node Stats per Window')
    plt.legend()
    for bars, values in zip([bars1, bars2, bars3, bars4], [count_unique_hubs_prev, count_unique_nonhubs_prev, count_hub_accesses_prev, count_nonhub_accesses_prev]):
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=60)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_per_window_prev_access.png', dpi=200)
    plt.close()

    # 4th plot: previously accessed nodes per window (percentages)
    pct_unique_hubs_prev = []
    pct_unique_nonhubs_prev = []
    pct_hub_accesses_prev = []
    pct_nonhub_accesses_prev = []
    
    for w in range(max_windows):
        hubs = set()
        nonhubs = set()
        hub_accesses = 0
        nonhub_accesses = 0
        for qw in query_windows:
            if w < len(qw):
                for node_id, is_hub in qw[w]:
                    if is_hub:
                        hubs.add(node_id)
                        hub_accesses += 1
                    else:
                        nonhubs.add(node_id)
                        nonhub_accesses += 1
        if w > 0:
            pct_unique_hubs_prev.append(100.0 * count_unique_hubs_prev[w] / len(hubs) if len(hubs) > 0 else 0.0)
            pct_unique_nonhubs_prev.append(100.0 * count_unique_nonhubs_prev[w] / len(nonhubs) if len(nonhubs) > 0 else 0.0)
        else:
            pct_unique_hubs_prev.append(0.0)
            pct_unique_nonhubs_prev.append(0.0)
        pct_hub_accesses_prev.append(100.0 * count_hub_accesses_prev[w] / hub_accesses if hub_accesses > 0 else 0.0)
        pct_nonhub_accesses_prev.append(100.0 * count_nonhub_accesses_prev[w] / nonhub_accesses if nonhub_accesses > 0 else 0.0)
    
    # Get colors from dummy plot
    plt.figure(figsize=(16,8))
    dummy_bars1 = plt.bar(x - 1.5*width, [0]*max_windows, width)
    dummy_bars2 = plt.bar(x - 0.5*width, [0]*max_windows, width)
    dummy_bars3 = plt.bar(x + 0.5*width, [0]*max_windows, width)
    dummy_bars4 = plt.bar(x + 1.5*width, [0]*max_windows, width)
    color1 = dummy_bars1.patches[0].get_facecolor()
    color2 = dummy_bars2.patches[0].get_facecolor()
    color3 = dummy_bars3.patches[0].get_facecolor()
    color4 = dummy_bars4.patches[0].get_facecolor()
    plt.clf()
    
    bars1 = plt.bar(x - 1.5*width, pct_unique_hubs_prev, width, label='Unique Hubs Previously Accessed (%)', color=color1)
    bars2 = plt.bar(x - 0.5*width, pct_unique_nonhubs_prev, width, label='Unique Non-Hubs Previously Accessed (%)', color=color2)
    bars3 = plt.bar(x + 0.5*width, pct_hub_accesses_prev, width, label='Hub Accesses Previously Accessed (%)', color=color3)
    bars4 = plt.bar(x + 1.5*width, pct_nonhub_accesses_prev, width, label='Non-Hub Accesses Previously Accessed (%)', color=color4)
    plt.xlabel(f'Window Index (size={window_size} per query)')
    plt.ylabel('Percentage (%)')
    plt.title('Previously Accessed Node Stats per Window (Percentage)')
    plt.legend()
    for bars, values in zip([bars1, bars2, bars3, bars4], [pct_unique_hubs_prev, pct_unique_nonhubs_prev, pct_hub_accesses_prev, pct_nonhub_accesses_prev]):
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=60)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_per_window_prev_access_pct.png', dpi=200)
    plt.close()
def main():
    parser = argparse.ArgumentParser(description='Plot hub/non-hub access stats from search trace JSON files')
    parser.add_argument('--trace', type=str, nargs='+', required=True, help='Paths to search trace JSON files (space-separated)')
    parser.add_argument('--window-size', type=int, default=1000, help='Window size for topological binning')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for results (subdirectories will be created for each trace file)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each trace file
    for trace_path in args.trace:
        # Extract prefix from filename (remove .json extension and directory path)
        filename = os.path.basename(trace_path)
        prefix = os.path.splitext(filename)[0]
        
        # Create subdirectory for this trace file
        trace_output_dir = os.path.join(args.output_dir, prefix)
        os.makedirs(trace_output_dir, exist_ok=True)
        
        # Load and process trace
        out_prefix = os.path.join(trace_output_dir, prefix)
        trace = load_trace(trace_path)
        plot_per_level(trace, out_prefix)
        plot_per_window(trace, args.window_size, out_prefix)

if __name__ == '__main__':
    main()
