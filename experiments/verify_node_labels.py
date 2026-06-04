"""
Verify that the label stored at a graph node matches what search returns.

Given that we observe serial and threaded searches returning DIFFERENT labels
for the SAME node_id and SAME distance, this script checks whether the label
discrepancy is in getNodeLabel() itself by querying a node's own vector.

If getNodeLabel(node_id) == node_id  (identity mapping, expected), then
searching with train[node_id] as the query should return label=node_id as top-1.

Usage:
    python verify_node_labels.py \\
        --dataset  ../data/sift-128-euclidean-20M/sift20M_base.fvecs \\
        --existing-mtx ../data/sift20m_m32_hnsw_base_layer.mtx \\
        --node-ids 18778717 18778716 18778718 \\
        --num-threads 1 2
"""

import argparse
import numpy as np
import flatnav
from flatnav.data_type import DataType
from pathlib import Path
from typing import List


def count_fvecs(path: str) -> int:
    import os
    node_bytes = 4 + 128 * 4  # 4-byte dim + 128 floats
    return os.path.getsize(path) // node_bytes


def read_fvecs(path: str, limit: int = None) -> np.ndarray:
    node_bytes = 4 + 128 * 4
    total = count_fvecs(path)
    n = total if limit is None else min(limit, total)
    result = np.empty((n, 128), dtype=np.float32)
    with open(path, 'rb') as f:
        for i in range(n):
            dim = int.from_bytes(f.read(4), 'little')
            result[i] = np.frombuffer(f.read(dim * 4), dtype=np.float32)
    return result


def build_index(train_data: np.ndarray, mtx_path: str, num_node_links: int = 32):
    n, dim = train_data.shape
    index = flatnav.index.create(
        distance_type='l2',
        index_data_type=DataType.float32,
        dim=dim,
        dataset_size=n,
        max_edges_per_node=num_node_links,
        verbose=False,
        collect_stats=False,
    )
    index.allocate_nodes(data=train_data).build_graph_links(mtx_path)
    return index


def verify_node(index, train_data: np.ndarray, node_id: int,
                num_threads_list: List[int], ef_search: int = 50, K: int = 5):
    query = train_data[node_id]
    print(f"\nNode {node_id}:")
    print(f"  Querying with train[{node_id}] (self-query, expected top-1 dist=0):")

    for nt in num_threads_list:
        index.set_num_threads(nt)
        if nt <= 1:
            distances, labels = index.search_single(
                query=query, ef_search=ef_search, K=K, num_initializations=100)
            labels_list = labels.tolist()
            distances_list = distances.tolist()
        else:
            query_batch = train_data[node_id:node_id + 1]
            distances_mat, labels_mat = index.search(
                queries=query_batch, K=K, ef_search=ef_search, num_initializations=100)
            labels_list = labels_mat[0].tolist()
            distances_list = distances_mat[0].tolist()

        top1_label = labels_list[0]
        top1_dist = distances_list[0]
        identity_ok = (top1_label == node_id and top1_dist == 0.0)
        print(f"  threads={nt}: top-{K} = {list(zip(labels_list, distances_list))}")
        print(f"    → getNodeLabel({node_id}) inferred = {top1_label}  "
              f"(dist={top1_dist})  identity_mapping={'OK' if identity_ok else 'MISMATCH!'}")


def main():
    parser = argparse.ArgumentParser(description="Verify node label → node_id identity mapping.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--existing-mtx", required=True)
    parser.add_argument("--node-ids", nargs="+", type=int, default=[18778717, 18778716, 18778718])
    parser.add_argument("--num-threads", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--num-node-links", type=int, default=32)
    parser.add_argument("--ef-search", type=int, default=50)
    args = parser.parse_args()

    mtx_path = str(Path(args.existing_mtx).expanduser().resolve())
    total = count_fvecs(args.dataset)
    print(f"Loading {total} training vectors from {args.dataset} ...")
    train_data = read_fvecs(args.dataset)
    print(f"Building index from {mtx_path} ...")
    index = build_index(train_data, mtx_path, args.num_node_links)
    print("Index ready.")

    for node_id in args.node_ids:
        if node_id >= len(train_data):
            print(f"\nNode {node_id}: out of range (max={len(train_data)-1}), skipping.")
            continue
        verify_node(index, train_data, node_id, args.num_threads,
                    ef_search=args.ef_search)


if __name__ == "__main__":
    main()
