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
  - T, U, R_max = T/U                  (persistent-cache reuse ceiling)
  - access-frequency concentration: top-k% share of accesses
  - phase split: reuse in the early (hub) phase vs the late (neighborhood) phase
  - downstream reuse by first-fetch hop, and (at a chosen C) its split into
    co-resident (in-flight-capturable) vs later-stream (persistent-only) parts
  - R_inflight(C): reuse an in-flight C-slot dedup window captures, via an
    offline round-robin scheduler simulation, compared against R_max on the
    SAME sampled queries (apples-to-apples).

Usage:
    python parse_access_trace.py run.trace
    python parse_access_trace.py run.trace --inflight 1,2,4,8,16,32 --inflight-queries 2000
    python parse_access_trace.py run.trace --early-hops 2 --split-c 32
    python parse_access_trace.py run.trace --mmap          # for multi-GB traces
"""

import argparse
import sys
import numpy as np

RECORD_DTYPE = np.dtype([("query_id", "<u4"), ("node_id", "<u4"), ("hop", "<u4")])
HOP_EDGES = (0, 2, 5, 10, np.iinfo(np.int64).max)


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


def take_first_queries(rec, max_queries):
    """Records belonging to the first `max_queries` distinct query_ids."""
    if max_queries is None:
        return rec
    uq = np.unique(rec["query_id"])
    if uq.size <= max_queries:
        return rec
    cutoff = uq[max_queries]  # smallest id to EXCLUDE
    return rec[rec["query_id"] < cutoff]


# --------------------------------------------------------------------- basic metrics
def summary(rec):
    """T, U, R_max, dedup savings, and per-query access stats."""
    qid, nid = rec["query_id"], rec["node_id"]
    T = int(rec.shape[0])
    U = int(np.unique(nid).size)
    n_queries = int(np.unique(qid).size)
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


def per_node_stats(rec):
    """Return (uniq_nodes_sorted, a(v), first_hop(v)) as parallel arrays."""
    nid = rec["node_id"]
    hop = rec["hop"].astype(np.int64)
    order = np.argsort(nid, kind="stable")
    nid_s, hop_s = nid[order], hop[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(nid_s)) + 1]
    uniq = nid_s[bounds]
    counts = np.diff(np.r_[bounds, nid_s.size])      # a(v)
    first_hop = np.minimum.reduceat(hop_s, bounds)   # min hop per node
    return uniq, counts, first_hop


def phase_split(rec, early_hops):
    """Reuse in the early (hop < early_hops) phase vs the late phase, over the
    full trace. Note: a node touched early by one query and late by another is
    counted in both phases' U, so the two reuse shares need not sum to 100%."""
    hop = rec["hop"]
    nid = rec["node_id"]
    n_queries = int(np.unique(rec["query_id"]).size)
    total_reuse = rec.shape[0] - int(np.unique(nid).size)
    rows = []
    for label, mask in (("early (hubs)", hop < early_hops), ("late (neighborhood)", hop >= early_hops)):
        sub = nid[mask]
        T = int(sub.size)
        U = int(np.unique(sub).size) if T else 0
        rows.append(
            {
                "phase": label,
                "mean_nodes_per_query": (T / n_queries) if n_queries else 0.0,
                "U": U,
                "R": (T / U) if U else float("nan"),
                "share_of_total_reuse": ((T - U) / total_reuse) if total_reuse else 0.0,
            }
        )
    return rows


def downstream_by_hop(rec, edges=HOP_EDGES):
    """Downstream reuse (a(v)-1) bucketed by each node's first-fetch hop."""
    uniq, counts, first_hop = per_node_stats(rec)
    downstream = counts - 1
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


def r_inflight(seqs, C, return_node_hits=False):
    """Reuse captured by an in-flight dedup window over C co-resident queries.

    Round-robin C slots, one fetch per slot per turn; refill a finished slot
    with the next query. A fetch is a HIT if the node is currently held by any
    in-flight query (a co-resident query already fetched it). On finish, a
    slot's nodes are released (ref-counted across slots).

    Returns (R_inflight, hits, total) -- and node_hits (node->#hits) if
    return_node_hits. R = total / (total - hits). Per-fetch round-robin
    approximation of the per-link scheduler (ignores Select/Traverse turns).
    """
    from collections import defaultdict

    nq = len(seqs)
    if nq == 0:
        return (float("nan"), 0, 0, {}) if return_node_hits else (float("nan"), 0, 0)
    C = min(C, nq)
    owners = defaultdict(int)
    node_hits = defaultdict(int) if return_node_hits else None
    slot_q = list(range(C))
    pos = [0] * C
    held = [[] for _ in range(C)]
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
                    if node_hits is not None:
                        node_hits[v] += 1
                owners[v] += 1
                held[s].append(v)
            if pos[s] >= len(seq):
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
    R = (total / misses) if misses else float("inf")
    if return_node_hits:
        return R, hits, total, node_hits
    return R, hits, total


def downstream_coresident_split(sample_rec, C, edges=HOP_EDGES):
    """Split downstream reuse into co-resident (captured by a C-slot in-flight
    window) vs later-stream (persistent-only), bucketed by first-fetch hop.
    Computed on `sample_rec` at concurrency C."""
    seqs = per_query_sequences(sample_rec)
    R, hits, total, node_hits = r_inflight(seqs, C, return_node_hits=True)
    uniq, counts, first_hop = per_node_stats(sample_rec)
    downstream = counts - 1
    co_res = np.zeros(uniq.size, dtype=np.int64)
    if node_hits:
        hk = np.fromiter(node_hits.keys(), dtype=np.uint32, count=len(node_hits))
        hv = np.fromiter(node_hits.values(), dtype=np.int64, count=len(node_hits))
        idx = np.searchsorted(uniq, hk)        # keys are a subset of uniq
        co_res[idx] = hv
    later = downstream - co_res
    total_reuse = int(downstream.sum())
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (first_hop >= lo) & (first_hop < hi)
        rows.append(
            {
                "hop_lo": lo,
                "hop_hi": (None if hi > 10 ** 9 else hi),
                "n_nodes": int(m.sum()),
                "co_resident": int(co_res[m].sum()),
                "later_stream": int(later[m].sum()),
                "share_of_total_reuse": (int(downstream[m].sum()) / total_reuse
                                         if total_reuse else 0.0),
            }
        )
    return rows, total_reuse, R


# ------------------------------------------------------------------------- report
def _hop_label(lo, hi):
    return f"{lo}{'+' if hi is None else '-' + str(hi - 1)}"


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
    ap.add_argument("--early-hops", type=int, default=2,
                    help="hops < this are the 'early/hub' phase (default 2 => hops 0,1)")
    ap.add_argument("--split-c", type=int, default=32,
                    help="C for the co-resident/later-stream downstream split; 0 to skip")
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
    for frac, k, share in concentration(access_frequency(rec)):
        print(f"  top {frac*100:7.3f}%  ({k:>10,} nodes)  -> {share*100:5.1f}%")

    print(f"\n-- phase split (early = hops < {args.early_hops}; shares may not sum to 100%) --")
    for r in phase_split(rec, args.early_hops):
        print(f"  {r['phase']:<20}  nodes/query={r['mean_nodes_per_query']:8.1f}  "
              f"U={r['U']:>12,}  R={r['R']:6.3f}  "
              f"({r['share_of_total_reuse']*100:5.1f}% of reuse)")

    print("\n-- downstream reuse by first-fetch hop --")
    rows, total_reuse = downstream_by_hop(rec)
    print(f"  total reuse (T-U) = {total_reuse:,}")
    for r in rows:
        print(f"  hop {_hop_label(r['hop_lo'], r['hop_hi']):<6}  nodes={r['n_nodes']:>12,}  "
              f"reuse={r['downstream_reuse']:>13,}  "
              f"({r['share_of_total_reuse']*100:5.1f}% of reuse)")

    cap = None if args.inflight_queries == 0 else args.inflight_queries
    sample = take_first_queries(rec, cap)
    seqs = per_query_sequences(sample)
    Ts = int(sample.shape[0])
    Us = int(np.unique(sample["node_id"]).size)
    rmax_s = Ts / Us if Us else float("inf")
    note = "" if cap is None else f" (first {len(seqs):,} queries)"

    print(f"\n-- in-flight reuse R_inflight(C){note} --")
    print(f"  R_max(sample) = {rmax_s:.3f}   [R_max(full) = {s['R_max']:.3f}]")
    for C in [int(x) for x in args.inflight.split(",") if x.strip()]:
        R, hits, total = r_inflight(seqs, C)
        pct = (R / rmax_s * 100) if rmax_s not in (0, float("inf")) else float("nan")
        print(f"  C={C:<4}  R_inflight={R:7.3f}  ({pct:4.0f}% of R_max(sample))  "
              f"hits={hits:,}/{total:,}")

    if args.split_c > 0:
        print(f"\n-- downstream split: co-resident (C={args.split_c}) vs later-stream{note} --")
        rows, tr, _ = downstream_coresident_split(sample, args.split_c)
        print(f"  total reuse (sample) = {tr:,}")
        for r in rows:
            print(f"  hop {_hop_label(r['hop_lo'], r['hop_hi']):<6}  "
                  f"nodes={r['n_nodes']:>12,}  "
                  f"co-resident={r['co_resident']:>12,}  "
                  f"later-stream={r['later_stream']:>13,}  "
                  f"({r['share_of_total_reuse']*100:5.1f}% of reuse)")


if __name__ == "__main__":
    sys.exit(main())
