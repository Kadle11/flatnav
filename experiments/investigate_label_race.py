"""
Root-cause investigation for the serial-vs-threaded label mismatch.

Background
----------
The trace analysis (analyse_trace.py) showed that the EXACT SAME node_id appears
in both serial and threaded heaps, but the label returned from that node differs
by 1 between runs. This script systematically checks:

  1. What label is stored at each diverging node BEFORE any search.
  2. Whether the stored label changes AFTER serial or threaded search.
  3. Whether the .mtx graph file uses 0- or 1-indexed node IDs
     (off-by-one in buildGraphLinks could corrupt the first-node label).
  4. Direct self-query: search for train[node_id] as the query and verify
     that the top-1 result has distance=0 and label=node_id.

Usage
-----
  cd /mydata/flatnav/experiments
  numactl --cpunodebind=0 --membind=0,2 poetry run python investigate_label_race.py \\
      --dataset   ../data/sift-128-euclidean-20M/sift20M_base.fvecs \\
      --existing-mtx ../data/sift20m_m32_hnsw_base_layer.mtx \\
      --queries   ../data/sift-128-euclidean-20M/sift20M_query.fvecs \\
      --gtruth    ../data/sift-128-euclidean-20M/bigann_gnd_20M.ivecs \\
      --diverging-nodes 18778717 19115669 18778641 \\
      --query-index 0 \\
      --ef-search 200 \\
      --num-threads 2
"""

import argparse
import numpy as np
import flatnav
from flatnav.data_type import DataType
from pathlib import Path
import struct


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def count_fvecs(path: str) -> int:
    import os
    return os.path.getsize(path) // (4 + 128 * 4)


def read_fvecs(path: str, limit: int = None) -> np.ndarray:
    total = count_fvecs(path)
    n = total if limit is None else min(limit, total)
    out = np.empty((n, 128), dtype=np.float32)
    with open(path, 'rb') as f:
        for i in range(n):
            d = int.from_bytes(f.read(4), 'little')
            out[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
    return out


def read_ivecs(path: str, limit: int = None) -> np.ndarray:
    row_size = 4 + 100 * 4
    import os
    total = os.path.getsize(path) // row_size
    n = total if limit is None else min(limit, total)
    out = np.empty((n, 100), dtype=np.int32)
    with open(path, 'rb') as f:
        for i in range(n):
            k = int.from_bytes(f.read(4), 'little')
            out[i] = np.frombuffer(f.read(k * 4), dtype=np.int32)
    return out


def build_index(train: np.ndarray, mtx: str, M: int = 32) -> object:
    n, dim = train.shape
    idx = flatnav.index.create(
        distance_type='l2', index_data_type=DataType.float32, dim=dim,
        dataset_size=n, max_edges_per_node=M, verbose=False, collect_stats=False)
    idx.allocate_nodes(data=train).build_graph_links(mtx)
    return idx


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_stored_labels(index, node_ids: list, tag: str):
    print(f"\n[{tag}] get_stored_node_label:")
    for nid in node_ids:
        lbl = index.get_stored_node_label(nid)
        match = "OK" if lbl == nid else f"MISMATCH! expected {nid}"
        print(f"  node {nid:12d} → stored_label={lbl:12d}  {match}")


def check_mtx_first_lines(mtx_path: str, n: int = 10):
    print(f"\n[MTX] First {n} data lines of {mtx_path}:")
    with open(mtx_path) as f:
        for line in f:
            if line.startswith('%'):
                continue
            # header line: rows cols nnz
            print(f"  HEADER: {line.rstrip()}")
            break
        count = 0
        for line in f:
            u, v = map(int, line.split())
            print(f"  edge: {u} {v}  (0-based: {u-1} {v-1})")
            count += 1
            if count >= n:
                break


def run_serial_search(index, query: np.ndarray, K: int, ef: int) -> list:
    index.set_num_threads(1)
    dists, labels = index.search_single(query=query, ef_search=ef, K=K, num_initializations=100)
    return list(zip(labels.tolist(), [float(d) for d in dists.tolist()]))


def run_threaded_search(index, queries: np.ndarray, K: int, ef: int, num_threads: int) -> list:
    index.set_num_threads(num_threads)
    dists_mat, labels_mat = index.search(
        queries=queries, K=K, ef_search=ef, num_initializations=100)
    return list(zip(labels_mat[0].tolist(), [float(d) for d in dists_mat[0].tolist()]))


def self_query_check(index, train: np.ndarray, node_ids: list, K: int = 5, ef: int = 50):
    """Search for each node's own training vector and check that top-1 = (dist=0, label=node_id)."""
    print(f"\n[SELF-QUERY] Searching train[node_id] for each diverging node:")
    print(f"  {'node_id':>12}  {'serial_top1_lbl':>16}  {'serial_top1_dist':>16}  "
          f"{'threaded_top1_lbl':>18}  {'threaded_top1_dist':>18}  match")
    for nid in node_ids:
        query = train[nid]
        serial = run_serial_search(index, query, K, ef)
        threaded = run_threaded_search(index, train[nid:nid+1], K, ef, 2)
        s_lbl, s_dist = serial[0]
        t_lbl, t_dist = threaded[0]
        same = s_lbl == t_lbl
        print(f"  {nid:>12}  {s_lbl:>16}  {s_dist:>16.1f}  {t_lbl:>18}  {t_dist:>18.1f}  "
              f"{'OK' if same else 'MISMATCH!'}")


def full_search_comparison(index, query: np.ndarray, queries1: np.ndarray,
                           node_ids: list, K: int, ef: int, num_threads: int, tag: str):
    """Run serial + threaded on the same query, compare labels at each rank and for diverging nodes."""
    serial_results = run_serial_search(index, query, K, ef)
    threaded_results = run_threaded_search(index, queries1, K, ef, num_threads)

    serial_labels = [l for l, _ in serial_results]
    threaded_labels = [l for l, _ in threaded_results]

    print(f"\n[{tag}] Serial vs threaded results for diverging nodes:")
    for nid in node_ids:
        in_serial = nid in serial_labels
        in_threaded = nid in threaded_labels
        s_rank = serial_labels.index(nid) if in_serial else None
        t_rank = threaded_labels.index(nid) if in_threaded else None
        print(f"  node {nid:12d}: serial_rank={str(s_rank):>5}  threaded_rank={str(t_rank):>5}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--existing-mtx", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--gtruth", default="")
    parser.add_argument("--diverging-nodes", nargs="+", type=int,
                        default=[18778717, 19115669, 18778641, 18778699, 18707267])
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--ef-search", type=int, default=200)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--num-node-links", type=int, default=32)
    args = parser.parse_args()

    mtx_path = str(Path(args.existing_mtx).expanduser().resolve())

    # --- 1. Inspect MTX file format -----------------------------------------------
    check_mtx_first_lines(mtx_path)

    # --- 2. Load data ---------------------------------------------------------------
    print(f"\nLoading training data from {args.dataset} ...")
    train = read_fvecs(args.dataset)
    print(f"  {len(train)} vectors loaded")

    queries = read_fvecs(args.queries, limit=args.query_index + 1)
    query_0 = queries[args.query_index]

    # --- 3. Build index -------------------------------------------------------------
    print(f"\nBuilding index from {mtx_path} ...")
    index = build_index(train, mtx_path, args.num_node_links)
    print("  Index ready")

    # --- 4. Check stored labels BEFORE any search -----------------------------------
    check_stored_labels(index, args.diverging_nodes, "BEFORE SEARCH")

    # --- 5. Run serial search, then re-check labels --------------------------------
    print(f"\nRunning serial search (1 thread, query_index={args.query_index}) ...")
    _ = run_serial_search(index, query_0, args.k, args.ef_search)
    check_stored_labels(index, args.diverging_nodes, "AFTER SERIAL SEARCH")

    # --- 6. Run threaded search, then re-check labels ------------------------------
    print(f"\nRunning threaded search ({args.num_threads} threads, query_index={args.query_index}) ...")
    _ = run_threaded_search(index, queries[args.query_index:args.query_index+1],
                            args.k, args.ef_search, args.num_threads)
    check_stored_labels(index, args.diverging_nodes, "AFTER THREADED SEARCH")

    # --- 7. Self-query check -------------------------------------------------------
    self_query_check(index, train, args.diverging_nodes[:3], K=5, ef=50)

    # --- 8. Full search comparison at query_index ----------------------------------
    q1 = queries[args.query_index:args.query_index+1]
    full_search_comparison(index, query_0, q1,
                           args.diverging_nodes, args.k, args.ef_search,
                           args.num_threads, "FINAL COMPARISON")


if __name__ == "__main__":
    main()
