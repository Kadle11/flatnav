# Root Cause Analysis: Serial vs. Threaded Search Recall Divergence

## Summary

FlatNav's batched `search()` returned lower recall under multiple threads
(0.835) than the serial `search_single()` path (0.899) on SIFT-20M. The cause
turned out to be **not** a threading, tie-breaking, or search-order problem at
all, but a **silent int→float type cast** in the Python bindings that corrupted
result labels. Node IDs above 2²⁴ that are odd were rounded to the nearest even
integer, shifting them by 1 and dropping true neighbors from the result set.

One-line fix in the bindings; the search algorithm was never wrong.

---

## Symptom

Running `sift_big_flatnav_recall_thread_compare.py` on SIFT-20M
(ef_search=200, K=100, 10 queries):

| Mode | Threads | Recall |
|------|---------|--------|
| serial   | 1 | 0.8990 |
| threaded | 2 | 0.8350 |

Every query showed label mismatches, and serial was consistently better — it
found true-positive neighbors the threaded run missed.

---

## Exploration

### 1. Lock-free search-path logging

To find where the two runs diverged, we instrumented `beamSearch` /
`processCandidateNode` with a lock-free, per-query trace log
(`SearchStepLog`): each query atomically claims one slot in a pre-allocated
array of arrays and writes only to its own slot, so no locks are needed and the
index search logic is untouched. The log records, per visited node:
`node_id, dist, worst_dist, accepted, is_entry, is_tie, tie_break_win`, and
later `popped_node_id` (which node was evicted from the heap on overflow).

The threaded path was wired so each query's batch index maps to its log slot
deterministically (`query_batch_index` threaded through `Index::search()`),
making serial and threaded traces directly comparable.

Analysis tooling: `analyse_trace.py`.

### 2. The search paths were identical

Comparing traces for the same query across serial and threaded runs:

```
Q  steps_s steps_t acc_s acc_t ties_s ties_t  1st_div  identical  pop_identical
0   4103    4103    734   734    0      0      None     True       True
1   3129    3129    577   577    0      0      None     True       True
...
```

- **Identical entry node**, identical candidate-expansion order.
- **Identical acceptance decisions** at every step (`accepted_serial_only = 0`,
  `accepted_threaded_only = 0`).
- **Zero tie events** (`is_tie = 0`) and zero tie-break wins inside `beamSearch`.
- **Identical pop sets** — the same nodes were evicted to maintain the beam.

So the heaps contained exactly the same multiset of nodes. The divergence was
entirely *after* `beamSearch`.

### 3. Same distances, different labels

The per-rank comparison showed every mismatch had `dist_diff = 0.0` and the two
labels differed by exactly **1**:

```
rank  serial_label  threaded_label   s_dist      t_dist     dist_diff  label_diff
   6    18778717      18778716       72170.00    72170.00    0.0e+00      1
  11    19115669      19115668       73842.00    73842.00    0.0e+00      1
  12    18778641      18778640       74456.00    74456.00    0.0e+00      1
  ...
```

Crucially, the trace showed the **same node_id 18778717** in both heaps at the
same distance, yet serial returned label `18778717` and threaded returned
`18778716`. That means `getNodeLabel(18778717)` was effectively returning
different values depending on which code path read it.

### 4. The decisive self-query test

We added a direct label accessor (`get_stored_node_label`) and a self-query
probe (`investigate_label_race.py`): search for a node's own training vector and
confirm the top-1 result is `(dist=0, label=node_id)`.

```
[BEFORE SEARCH] get_stored_node_label:
  node 18778717 → stored_label=18778717  OK     # memory is correct

[SELF-QUERY] Searching train[node_id]:
  node_id    serial_top1_lbl  serial_top1_dist   threaded_top1_lbl  threaded_top1_dist  match
  18778717      18778717            0.0           18778716.0              0.0           MISMATCH!
  19115669      19115669            0.0           19115668.0              0.0           MISMATCH!
  18778641      18778641            0.0           18778640.0              0.0           MISMATCH!
```

Two things stood out:
- The label stored in index memory was always correct (`18778717`).
- The threaded result came back as **`18778716.0` — a float** with a fractional
  representation, off by 1 from an odd value.

---

## Root Cause

The batched search in the Python bindings returned its result pair in the wrong
order relative to the pair's declared type:

```cpp
// DistancesLabelsPair = std::pair<py::array_t<float>, py::array_t<label_t>>;
//                                  ^^^^^ distances first   ^^^^^^^ labels second

// searchImpl (batched) — BUG:
return {labels, dists};   // labels (int32) forced into the float slot
```

pybind11's pair conversion constructed a `py::array_t<float>` from the int32
`labels` array — **casting every label to float32**.

`float32` has a 24-bit mantissa, so it represents every integer exactly only up
to 2²⁴ = 16,777,216. Above that, only even integers are exactly representable.
SIFT-20M has node IDs up to ~20M, so any **odd** label > 2²⁴ rounds to its even
neighbor:

```
18778717 (odd)  --cast to float32-->  18778716.0  --back to int-->  18778716  ✗
18778699 (odd)  --cast to float32-->  18778700.0  --back to int-->  18778700  ✗
```

Every per-query mismatch was exactly this: an odd node ID above 16.7M rounded by
1. The serial path used `searchSingleImpl`, which returned the pair in the
correct `{distances, labels}` order, so its labels stayed int32 and exact —
hence its higher recall. The "threading" framing was a red herring; the bug
fired on **any** call to batched `search()`, single- or multi-threaded.

---

## The Fix

### 1. Bindings — return the pair in the declared order

```cpp
// python-bindings/src/flatnav/bindings.cpp  (searchImpl)
return {dists, labels};   // distances in the float slot, labels in the label_t slot
```

This makes batched `search()` consistent with `search_single()` and with the
documented contract `Returns: Tuple[distances, labels]`.

### 2. Index — deterministic tie-break in the final sort

With labels now correct, a secondary sort key guarantees serial and threaded
produce identical ordering when distances are exactly equal at the K-boundary:

```cpp
// include/flatnav/index/Index.h  (Index::search)
std::sort(results.begin(), results.end(),
          [](const dist_label_t& left, const dist_label_t& right) {
            if (left.first != right.first) return left.first < right.first;
            return left.second < right.second;   // tie-break by label
          });
```

### 3. Caller updates

Because the old (buggy) batched `search()` returned labels-first, callers that
unpacked labels-first had to be flipped to match the corrected
`(distances, labels)` order:

- `sift_big_flatnav_recall_thread_compare.py` — `distances, labels = index.search(...)`
- `sift_big_flatnav_recall_batched.py` — use the second element for labels

Callers that already used `_, top_k_indices = index.search(...)` (e.g.
`utils.py`, `python-bindings/unit_tests/test_utils.py`) were **latently broken**
before — they were silently consuming the int-cast distances as indices — and
become correct with no change. FAISS-based callers were untouched.

---

## Why float32 and not the labels were the giveaway

The `.0` suffix in the threaded output (`18778716.0`) was the smoking gun: an
integer label has no business being a float. That single observation collapsed
the entire "threads take different paths" hypothesis — the paths were identical;
only the *encoding* of the answer was lossy.

---

## Reproduction / verification scripts

| Script | Purpose |
|--------|---------|
| `sift_big_flatnav_recall_thread_compare.py` | Serial vs threaded recall + per-query trace capture |
| `analyse_trace.py` | Compare serial/threaded traces: path identity, ties, pops, distance-tie / same-node-different-label detection |
| `investigate_label_race.py` | Stored-label checks before/after search, MTX index-base check, self-query probe |
| `verify_node_labels.py` | Self-query identity check: `getNodeLabel(n) == n` |

Quick confirmation after the fix (self-query should show no `MISMATCH!` and no
float labels); the full recall run should bring threaded recall up to the serial
0.899.
