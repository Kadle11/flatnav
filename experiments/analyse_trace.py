"""
Analyse search-path trace JSON produced by sift_big_flatnav_recall_thread_compare.py.

Usage:
    python analyse_trace.py <path-to-json> [--ef-search 200] [--query-index 0]

Flags
-----
--ef-search   Which ef_search bucket to analyse (default: first key found).
--query-index  Index of the single query to dump step-level detail for (default: 0).
--all-queries  Print per-query summary for every query (no step-level detail).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _first_step_diff(serial: List[Dict], threaded: List[Dict]) -> Optional[int]:
    """Return the index of the first step that differs (any field), or None."""
    for i, (s, t) in enumerate(zip(serial, threaded)):
        if (s["node_id"] != t["node_id"]
                or s["accepted"] != t["accepted"]
                or s["dist"] != t["dist"]
                or s["worst_dist"] != t["worst_dist"]
                or s["is_tie"] != t["is_tie"]
                or s["tie_break_win"] != t["tie_break_win"]):
            return i
    if len(serial) != len(threaded):
        return min(len(serial), len(threaded))
    return None


def _trace_stats(trace: List[Dict]) -> Dict[str, Any]:
    non_entry = [s for s in trace if not s.get("is_entry")]
    return {
        "total_steps": len(trace),
        "accepted": sum(1 for s in trace if s.get("accepted")),
        "rejected": sum(1 for s in non_entry if not s.get("accepted")),
        "ties": sum(1 for s in non_entry if s.get("is_tie")),
        "tie_break_wins": sum(1 for s in trace if s.get("tie_break_win")),
        "unique_nodes": len({s["node_id"] for s in trace}),
        "entry_node": next((s["node_id"] for s in trace if s.get("is_entry")), None),
    }


def _compare_full(serial: List[Dict], threaded: List[Dict]) -> Dict[str, Any]:
    first_diff = _first_step_diff(serial, threaded)
    serial_stats = _trace_stats(serial)
    threaded_stats = _trace_stats(threaded)

    UINT32_MAX = 2**32 - 1

    serial_visited = {s["node_id"] for s in serial}
    threaded_visited = {s["node_id"] for s in threaded}
    serial_accepted = {s["node_id"] for s in serial if s.get("accepted")}
    threaded_accepted = {s["node_id"] for s in threaded if s.get("accepted")}

    # Map popped_node_id → step_index where it was evicted
    serial_pop_at: Dict[int, int] = {}
    threaded_pop_at: Dict[int, int] = {}
    for i, s in enumerate(serial):
        pid = s.get("popped_node_id", UINT32_MAX)
        if pid != UINT32_MAX:
            serial_pop_at[pid] = i
    for i, t in enumerate(threaded):
        pid = t.get("popped_node_id", UINT32_MAX)
        if pid != UINT32_MAX:
            threaded_pop_at[pid] = i

    serial_popped = set(serial_pop_at.keys())
    threaded_popped = set(threaded_pop_at.keys())

    return {
        "path_fully_identical": first_diff is None,
        "first_divergence_step": first_diff,
        "entry_nodes_match": serial_stats["entry_node"] == threaded_stats["entry_node"],
        "serial_entry_node": serial_stats["entry_node"],
        "threaded_entry_node": threaded_stats["entry_node"],
        "serial": serial_stats,
        "threaded": threaded_stats,
        "visited_serial_only": len(serial_visited - threaded_visited),
        "visited_threaded_only": len(threaded_visited - serial_visited),
        "visited_overlap": len(serial_visited & threaded_visited),
        "accepted_serial_only": len(serial_accepted - threaded_accepted),
        "accepted_threaded_only": len(threaded_accepted - serial_accepted),
        "accepted_overlap": len(serial_accepted & threaded_accepted),
        "popped_serial_only": len(serial_popped - threaded_popped),
        "popped_threaded_only": len(threaded_popped - serial_popped),
        "pop_sets_identical": serial_popped == threaded_popped,
        "serial_pop_at": serial_pop_at,
        "threaded_pop_at": threaded_pop_at,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_step(step: Dict, idx: int, flag: str = "") -> str:
    return (
        f"  [{idx:5d}]{flag} node={step['node_id']:10d}  "
        f"dist={step['dist']:.6f}  worst={step['worst_dist']:.6f}  "
        f"acc={int(step['accepted'])}  tie={int(step.get('is_tie',False))}  "
        f"tbw={int(step.get('tie_break_win',False))}  "
        f"entry={int(step.get('is_entry',False))}"
    )


def print_per_query_summary(comp: List[Dict], ef_search: str) -> None:
    print(f"\n{'='*80}")
    print(f"PER-QUERY SUMMARY  ef_search={ef_search}")
    print(f"{'='*80}")
    header = (
        f"{'Q':>3}  {'recall_s':>8}  {'recall_t':>8}  "
        f"{'steps_s':>7}  {'steps_t':>7}  "
        f"{'acc_s':>5}  {'acc_t':>5}  "
        f"{'ties_s':>6}  {'ties_t':>6}  "
        f"{'tbw_s':>5}  {'tbw_t':>5}  "
        f"{'1st_div':>7}  {'identical':>9}  "
        f"{'acc_s_only':>10}  {'acc_t_only':>10}  "
        f"{'pop_identical':>13}"
    )
    print(header)
    print("-" * len(header))

    for q in comp:
        qi = q.get("query_index", "?")
        st = q.get("status", "?")
        if st != "ok":
            print(f"{qi:>3}  [{st}]")
            continue

        s_trace = q.get("serial_trace", [])
        t_trace = q.get("threaded_trace", [])
        if not s_trace or not t_trace:
            print(f"{qi:>3}  [no trace]")
            continue

        cmp = _compare_full(s_trace, t_trace)
        ss = cmp["serial"]
        ts = cmp["threaded"]

        # per-query recalls from parent query_results
        s_recall = q.get("serial_recall", "n/a")
        t_recall = q.get("threaded_recall", "n/a")

        print(
            f"{qi:>3}  {s_recall!s:>8}  {t_recall!s:>8}  "
            f"{ss['total_steps']:>7}  {ts['total_steps']:>7}  "
            f"{ss['accepted']:>5}  {ts['accepted']:>5}  "
            f"{ss['ties']:>6}  {ts['ties']:>6}  "
            f"{ss['tie_break_wins']:>5}  {ts['tie_break_wins']:>5}  "
            f"{str(cmp['first_divergence_step']):>7}  "
            f"{str(cmp['path_fully_identical']):>9}  "
            f"{cmp['accepted_serial_only']:>10}  {cmp['accepted_threaded_only']:>10}  "
            f"{str(cmp.get('pop_sets_identical', 'n/a')):>13}"
        )


def print_step_detail(q: Dict, max_steps: int = 60) -> None:
    qi = q.get("query_index", "?")
    print(f"\n{'='*80}")
    print(f"STEP-LEVEL DETAIL  query_index={qi}")
    print(f"{'='*80}")

    s_trace = q.get("serial_trace", [])
    t_trace = q.get("threaded_trace", [])
    if not s_trace or not t_trace:
        print("  [no trace data available]")
        return

    cmp = _compare_full(s_trace, t_trace)
    print(f"  path_fully_identical : {cmp['path_fully_identical']}")
    print(f"  first_divergence_step: {cmp['first_divergence_step']}")
    print(f"  entry nodes          : serial={cmp['serial_entry_node']}  threaded={cmp['threaded_entry_node']}  match={cmp['entry_nodes_match']}")
    print(f"  total steps          : serial={cmp['serial']['total_steps']}  threaded={cmp['threaded']['total_steps']}")
    print(f"  accepted nodes       : serial={cmp['serial']['accepted']}  threaded={cmp['threaded']['accepted']}")
    print(f"  accepted serial-only : {cmp['accepted_serial_only']}")
    print(f"  accepted threaded-only: {cmp['accepted_threaded_only']}")
    print(f"  accepted overlap     : {cmp['accepted_overlap']}")
    print(f"  tie events           : serial={cmp['serial']['ties']}  threaded={cmp['threaded']['ties']}")
    print(f"  tie-break wins       : serial={cmp['serial']['tie_break_wins']}  threaded={cmp['threaded']['tie_break_wins']}")

    first_diff = cmp["first_divergence_step"]

    if first_diff is None:
        print("\n  Paths are FULLY IDENTICAL — showing first and last steps:")
        show_indices = list(range(min(5, len(s_trace)))) + list(
            range(max(0, len(s_trace) - 5), len(s_trace))
        )
    else:
        # Show context around divergence
        ctx = 10
        start = max(0, first_diff - ctx)
        end = min(len(s_trace), len(t_trace), first_diff + ctx)
        show_indices = list(range(start, end))
        print(f"\n  Showing steps {start}–{end - 1} (divergence at step {first_diff}):")

    print(f"\n  {'':>8}  {'SERIAL':^70}")
    print(f"  {'':>8}  {'THREADED':^70}")
    prev_diff = False
    for i in show_indices:
        if first_diff is not None and i > first_diff + max_steps:
            print("  ... (truncated)")
            break
        s = s_trace[i] if i < len(s_trace) else None
        t = t_trace[i] if i < len(t_trace) else None

        same = (s is not None and t is not None and
                s["node_id"] == t["node_id"] and
                s["accepted"] == t["accepted"] and
                s["dist"] == t["dist"] and
                s["worst_dist"] == t["worst_dist"])

        flag = " [DIFF]" if not same else "       "
        if s:
            print(_fmt_step(s, i, flag))
        if t and not same:
            print(_fmt_step(t, i, "  >>>>"))
        if not same and not prev_diff:
            print()
        prev_diff = not same


def print_distance_tie_analysis(q: Dict) -> None:
    """
    Show where the K-boundary falls relative to equal-distance nodes.

    Same distances but different labels at a rank means both runs found
    nodes with identical distances to the query, but unstable sort placed
    different nodes inside/outside the top-K cutoff.
    """
    qi = q.get("query_index", "?")
    label_diffs = q.get("label_differences", [])
    dist_diffs = q.get("distance_differences", [])
    if not label_diffs or not dist_diffs:
        return

    # Find ranks where label differs but distance is (nearly) the same
    print(f"\n{'='*80}")
    print(f"DISTANCE-TIE ANALYSIS  query_index={qi}")
    print(f"{'='*80}")

    k = len(label_diffs)
    threshold = 1e-6

    # Collect all distances and find duplicates near the K boundary
    serial_dists = [d["serial"] for d in dist_diffs]
    threaded_dists = [d["threaded"] for d in dist_diffs]

    mismatch_ranks = [d["rank"] for d in label_diffs if not d["same"]]
    if not mismatch_ranks:
        print("  No label mismatches found.")
        return

    print(f"  K={k}, mismatch ranks: {mismatch_ranks[:20]}{'...' if len(mismatch_ranks) > 20 else ''}")
    print(f"  Serial   dist@K-1: {serial_dists[k-2] if k >= 2 else 'n/a'}  dist@K: {serial_dists[k-1]}")
    print(f"  Threaded dist@K-1: {threaded_dists[k-2] if k >= 2 else 'n/a'}  dist@K: {threaded_dists[k-1]}")

    # Find runs of equal distances in serial results (tie groups)
    print(f"\n  Tie groups in serial results (consecutive equal distances):")
    i = 0
    while i < len(serial_dists):
        j = i + 1
        while j < len(serial_dists) and abs(serial_dists[j] - serial_dists[i]) < threshold:
            j += 1
        if j - i > 1:
            spans_k_boundary = (i < k <= j)
            flag = " *** SPANS K-BOUNDARY ***" if spans_k_boundary else ""
            print(f"    ranks [{i}..{j-1}] dist={serial_dists[i]:.8f} ({j-i} nodes){flag}")
        i = j

    # Build pop-event maps from the traces for this query
    serial_trace = q.get("serial_trace", [])
    threaded_trace = q.get("threaded_trace", [])
    UINT32_MAX = 2**32 - 1
    s_pop_at: Dict[int, int] = {}
    t_pop_at: Dict[int, int] = {}
    for i, s in enumerate(serial_trace):
        pid = s.get("popped_node_id", UINT32_MAX)
        if pid != UINT32_MAX:
            s_pop_at[pid] = i
    for i, t in enumerate(threaded_trace):
        pid = t.get("popped_node_id", UINT32_MAX)
        if pid != UINT32_MAX:
            t_pop_at[pid] = i

    # Also build accepted-step map to know when each node was first accepted
    s_acc_at: Dict[int, int] = {}
    t_acc_at: Dict[int, int] = {}
    for i, s in enumerate(serial_trace):
        if s.get("accepted") and s["node_id"] not in s_acc_at:
            s_acc_at[s["node_id"]] = i
    for i, t in enumerate(threaded_trace):
        if t.get("accepted") and t["node_id"] not in t_acc_at:
            t_acc_at[t["node_id"]] = i

    # Detect the key pattern: same node in both traces but different result label.
    # This would mean getNodeLabel(node_id) returns different values per run.
    sl_at_dist: dict = {}  # dist → list of (node_id, run) for serial accepted nodes
    tl_at_dist: dict = {}
    for s in serial_trace:
        if s.get("accepted"):
            d = s["dist"]
            sl_at_dist.setdefault(d, []).append(s["node_id"])
    for t in threaded_trace:
        if t.get("accepted"):
            d = t["dist"]
            tl_at_dist.setdefault(d, []).append(t["node_id"])

    print(f"\n  Mismatching ranks — distance ties with pop/accept tracking:")
    print(f"  (sl/tl = serial/threaded label; _acc@/_pop@ = step index in respective trace)")
    print(f"  {'rank':>5}  {'sl':>12}  {'tl':>14}  {'dist':>12}  "
          f"{'dist_diff':>9}  {'sl_acc@':>7}  {'sl_pop@':>7}  {'tl_acc@':>7}  {'tl_pop@':>7}  note")
    for d in label_diffs:
        if not d["same"]:
            r = d["rank"]
            sl = d["serial"]
            tl = d["threaded"]
            sd = dist_diffs[r]["serial"] if r < len(dist_diffs) else float("nan")
            td = dist_diffs[r]["threaded"] if r < len(dist_diffs) else float("nan")
            diff = abs(td - sd)
            s_acc = s_acc_at.get(sl, "—")
            s_pop = s_pop_at.get(sl, "—")
            t_acc = t_acc_at.get(tl, "—")
            t_pop = t_pop_at.get(tl, "—")
            # Is the serial label node popped in threaded (but not serial)?
            s_node_in_t_pop = sl in t_pop_at and sl not in s_pop_at
            t_node_in_s_pop = tl in s_pop_at and tl not in t_pop_at
            note = ""
            if s_node_in_t_pop:
                note = f"← sl popped in threaded@{t_pop_at[sl]}"
            if t_node_in_s_pop:
                note += f" ← tl popped in serial@{s_pop_at[tl]}"
            # Check for the "same-node-different-label" pattern:
            # If only one node was accepted at this distance, and both serial
            # and threaded accepted it, but serial returns label=sl and threaded
            # returns label=tl, the same node_id maps to different labels.
            same_dist_s_nodes = sl_at_dist.get(sd, [])
            same_dist_t_nodes = tl_at_dist.get(td, [])
            if (len(same_dist_s_nodes) == 1 and len(same_dist_t_nodes) == 1
                    and same_dist_s_nodes[0] == same_dist_t_nodes[0]
                    and sl != tl):
                same_node = same_dist_s_nodes[0]
                note = (f"*** SAME NODE {same_node} → label {sl} (serial) "
                        f"vs {tl} (threaded): getNodeLabel mismatch!")

            if not note and diff < threshold:
                note = "dist-tie, same pops"
            print(f"  {r:>5}  {sl:>12}  {tl:>14}  {sd:>12.4f}  "
                  f"{diff:>9.1e}  {str(s_acc):>7}  {str(s_pop):>7}  "
                  f"{str(t_acc):>7}  {str(t_pop):>7}  {note}")


def print_global_summary(data: Dict, ef_search: str) -> None:
    results = data["results"][ef_search]
    print(f"\n{'='*80}")
    print(f"GLOBAL SUMMARY  ef_search={ef_search}")
    print(f"{'='*80}")
    print(f"  dataset       : {data.get('dataset','?')}")
    print(f"  num_queries   : {data.get('num_queries','?')}")
    print(f"  num_threads   : {data.get('num_search_threads','?')}")
    print(f"  serial recall : {results['serial']['recall']:.4f}")
    print(f"  threaded recall: {results['threaded']['recall']:.4f}")
    print(f"  delta recall  : {results['threaded']['recall'] - results['serial']['recall']:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse search-path trace JSON.")
    parser.add_argument("json_path", help="Path to trace JSON file.")
    parser.add_argument("--ef-search", default=None, help="ef_search bucket (default: first found).")
    parser.add_argument("--query-index", type=int, default=0, help="Query index for step-level detail.")
    parser.add_argument("--all-queries", action="store_true", help="Show per-query summary for all queries.")
    args = parser.parse_args()

    path = Path(args.json_path)
    if not path.is_file():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(path.read_text())
    available_ef = list(data.get("results", {}).keys())
    if not available_ef:
        print("ERROR: no results found in JSON.", file=sys.stderr)
        sys.exit(1)

    ef_key = args.ef_search if args.ef_search is not None else available_ef[0]
    if ef_key not in data["results"]:
        print(f"ERROR: ef_search={ef_key} not found. Available: {available_ef}", file=sys.stderr)
        sys.exit(1)

    print_global_summary(data, ef_key)

    results = data["results"][ef_key]
    comp = results["comparison"]["query_differences"]

    # Attach per-query recall from query_results for display purposes
    s_qr = {q["query_index"]: q for q in results["serial"].get("query_results", [])}
    t_qr = {q["query_index"]: q for q in results["threaded"].get("query_results", [])}
    for q in comp:
        qi = q.get("query_index")
        q["serial_recall"] = round(s_qr[qi]["recall"], 4) if qi in s_qr and s_qr[qi].get("recall") is not None else "n/a"
        q["threaded_recall"] = round(t_qr[qi]["recall"], 4) if qi in t_qr and t_qr[qi].get("recall") is not None else "n/a"

    if args.all_queries:
        print_per_query_summary(comp, ef_key)
        for q in comp:
            if q.get("status") == "ok":
                print_distance_tie_analysis(q)
    else:
        print_per_query_summary(comp, ef_key)
        target = next((q for q in comp if q.get("query_index") == args.query_index), None)
        if target is None:
            print(f"\nWARNING: query_index={args.query_index} not found in comparisons.")
        else:
            print_distance_tie_analysis(target)
            print_step_detail(target)


if __name__ == "__main__":
    main()
