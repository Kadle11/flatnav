#!/usr/bin/env python3
"""Parse and analyze a FlatNav access trace (cross-query reuse analysis).

Trace format (written by include/flatnav/util/AccessTracer.h, built with
-DFLATNAV_TRACE_ACCESS): a flat binary file of 12-byte records, each three
little-endian uint32:

    (query_id, node_id, hop)

one record per GraphRead in processOneLink. Because the per-query visited set
fetches each node at most once, a node's record count == the number of distinct
queries that fetched it == a(v); all reuse here is cross-query.

What it computes:
  - T, U, R_max = T/U          (persistent-cache reuse ceiling)
  - access-frequency concentration: top-k% share of accesses
  - downstream reuse by first-fetch hop
  - R_inflight(C): reuse an in-flight C-slot dedup window captures, via an
    offline round-robin scheduler simulation.

Usage:
    python parse_access_trace.py run.trace
    python parse_access_trace.py run.trace --inflight 1,2,4,8,16,32 --inflight-queries 2000
    python parse_access_trace.py run.trace --mmap          # for multi-GB traces
"""

import argparse
import sys
import numpy as np

RECORD_DTYPE = np.dtype([("query_id", "<u4"), ("node_id", "<u4"), ("hop", "<u4")])


# ----------------------------------------------------------------------------- load
def load(path, mmap=False):
    """Return a structured array with fields query_id, node_id, hop."""
    import os

    nbytes = os.path.getsize(path)
    if nbytes == 0:
        raise ValueError(f"{path} is empty")
    if nbytes % RECORD_DTYPE.itemsize != 0:
        raise ValueError(
            f"{path}: size {nbytes} is not a multiple of {RECORD_DTYPE.itemsize} "
            "-- truncated/corrupt trace?"
        )
    if mmap:
        return np.memmap(path, dtype=RECORD_DTYPE, mode="r")
    return np.fromfile(path, dtype=RECORD_DTYPE)


# --------------------------------------------------------------------- basic metrics
def summary(rec):
    """T, U, R_max, dedup savings, and per-query access stats."""
    qid, nid = rec["query_id"], rec["node_id"]
    T = int(rec.shape[0])
    U = int(np.unique(nid).size)
    n_queries = int(np.unique(qid).size)
    # accesses per query (visited set => |S_q|)
    _, per_q_counts = np.unique(qid, return_counts=True)
    return {
        "T_total_accesses": T,
        "U_unique_nodes": U,
        "n_queries": n_queries,
        "R_max": T / U if U else float("inf"),
        "dedup_savings": 1.0 - U / T if T else 0.0,
        "mean_accesses_per_query": float(per_q_counts.mean()),
        "max_hop": int(rec["hop"].max()),
    }


def access_frequency(rec):
    """a(v) for every touched node, sorted descending."""
    _, counts = np.unique(rec["node_id"], return_counts=True)
    return np.sort(counts)[::-1]


def concentration(a_sorted, fractions=(0.0001, 0.001, 0.01, 0.10, 0.50)):
    """Cumulative share of total accesses held by the top-f fraction of nodes."""
    T = int(a_sorted.sum())
    U = a_sorted.size
    cum = np.cumsum(a_sorted)
    out = []
    for f in fractions:
        k = max(1, int(round(f * U)))
        out.append((f, k, cum[min(k, U) - 1] / T))
    return out  # list of (fraction, k_nodes, accesses_share)


def downstream_by_hop(rec, edges=(0, 2, 5, 10, np.iinfo(np.int64).max)):
    """Downstream reuse (a(v)-1) bucketed by each node's first-fetch hop.

    First-fetch hop = min hop over all queries that fetch the node.
    """
    nid, hop = rec["node_id"], rec["hop"].astype(np.int64)
    order = np.argsort(nid, kind="stable")
    nid_s, hop_s = nid[order], hop[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(nid_s)) + 1]
    counts = np.diff(np.r_[bounds, nid_s.size])          # a(v) per node
    first_hop = np.minimum.reduceat(hop_s, bounds)        # min hop per node
    downstream = counts - 1                               # reuse per node
    total_reuse = int(downstream.sum())
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (first_hop >= lo) & (first_hop < hi)
        ds = int(downstream[m].sum())
        rows.append(
            {
                "hop_lo": lo,
                "hop_hi": (None if hi > 10 ** 9 else hi),
                "n_nodes": int(m.sum()),
                "downstream_reuse": ds,
                "share_of_total_reuse": (ds / total_reuse if total_reuse else 0.0),
            }
        )
    return rows, total_reuse


# ----------------------------------------------------------------- in-flight reuse
def per_query_sequences(rec, max_queries=None):
    """List of node-id arrays, one per query in query_id order (fetch order within)."""
    order = np.argsort(rec["query_id"], kind="stable")
    qid_s = rec["query_id"][order]
    nid_s = rec["node_id"][order]
    bounds = np.r_[0, np.flatnonzero(np.diff(qid_s)) + 1]
    seqs = np.split(nid_s, bounds[1:])
    if max_queries is not None:
        seqs = seqs[:max_queries]
    return seqs


def r_inflight(seqs, C):
    """Reuse captured by an in-flight dedup window over C co-resident queries.

    Round-robin C slots, one fetch per slot per turn; refill a finished slot
    with the next query. A fetch is a HIT if the node is currently held by any
    in-flight query (a co-resident query already fetched it). On finish, a
    slot's nodes are released (ref-counted across slots).

    Returns (R_inflight, hits, total). R = total / (total - hits).
    Note: a per-fetch round-robin approximation of the per-link scheduler; it
    ignores the Select/Traverse turns between fetches.
    """
    from collections import defaultdict

    nq = len(seqs)
    if nq == 0:
        return float("nan"), 0, 0
    C = min(C, nq)
    owners = defaultdict(int)          # node -> # in-flight queries holding it
    slot_q = list(range(C))            # query index in each slot (or None)
    pos = [0] * C
    held = [[] for _ in range(C)]      # nodes fetched by each slot's current query
    next_q = C
    active = C
    hits = 0
    total = 0
    while active > 0:
        for s in range(C):
            q = slot_q[s]
            if q is None:
                continue
            seq = seqs[q]
            if pos[s] < len(seq):
                v = int(seq[pos[s]])
                pos[s] += 1
                total += 1
                if owners[v] > 0:
                    hits += 1
                owners[v] += 1
                held[s].append(v)
            if pos[s] >= len(seq):       # query finished -> release + refill
                for v in held[s]:
                    owners[v] -= 1
                held[s] = []
                if next_q < nq:
                    slot_q[s] = next_q
                    pos[s] = 0
                    next_q += 1
                else:
                    slot_q[s] = None
                    active -= 1
    misses = total - hits
    return (total / misses if misses else float("inf")), hits, total


# ------------------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="path to the binary access trace")
    ap.add_argument("--mmap", action="store_true",
                    help="memory-map the file (for multi-GB traces)")
    ap.add_argument("--inflight", default="1,2,4,8,16,32",
                    help="comma-separated C values for R_inflight(C)")
    ap.add_argument("--inflight-queries", type=int, default=2000,
                    help="cap queries used for the (Python) scheduler sim; "
                         "0 = all (slow on big traces)")
    args = ap.parse_args()

    rec = load(args.trace, mmap=args.mmap)

    s = summary(rec)
    print(f"=== {args.trace} ===")
    print(f"records (T)            : {s['T_total_accesses']:,}")
    print(f"unique nodes (U)       : {s['U_unique_nodes']:,}")
    print(f"queries                : {s['n_queries']:,}")
    print(f"R_max = T/U (persist)  : {s['R_max']:.3f}")
    print(f"dedup savings          : {s['dedup_savings']*100:.1f}%")
    print(f"mean accesses/query    : {s['mean_accesses_per_query']:.1f}")
    print(f"max hop                : {s['max_hop']}")

    print("\n-- access concentration (top-k of nodes -> share of accesses) --")
    a = access_frequency(rec)
    for frac, k, share in concentration(a):
        print(f"  top {frac*100:7.3f}%  ({k:>10,} nodes)  -> {share*100:5.1f}%")

    print("\n-- downstream reuse by first-fetch hop --")
    rows, total_reuse = downstream_by_hop(rec)
    print(f"  total reuse (T-U) = {total_reuse:,}")
    for r in rows:
        hi = "+" if r["hop_hi"] is None else f"-{r['hop_hi']-1}"
        label = f"{r['hop_lo']}{hi}"
        print(f"  hop {label:<6}  nodes={r['n_nodes']:>10,}  "
              f"reuse={r['downstream_reuse']:>12,}  "
              f"({r['share_of_total_reuse']*100:5.1f}% of reuse)")

    print("\n-- in-flight reuse R_inflight(C) vs persistent R_max --")
    cap = None if args.inflight_queries == 0 else args.inflight_queries
    seqs = per_query_sequences(rec, max_queries=cap)
    note = "" if cap is None else f" (first {len(seqs):,} queries)"
    print(f"  R_max (persistent) = {s['R_max']:.3f}{note}")
    rmax = s["R_max"]
    for C in [int(x) for x in args.inflight.split(",") if x.strip()]:
        R, hits, total = r_inflight(seqs, C)
        pct = (R / rmax * 100) if rmax not in (0, float("inf")) else float("nan")
        print(f"  C={C:<4}  R_inflight={R:7.3f}  ({pct:4.0f}% of R_max)  "
              f"hits={hits:,}/{total:,}")


if __name__ == "__main__":
    sys.exit(main())
