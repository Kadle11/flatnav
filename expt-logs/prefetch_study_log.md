# Candidate-driven tier prefetch — study log

Next study in the high-bandwidth FlatNav work: **can the candidate priority queue predict, with
enough lead time, which nodes the search will touch next — so we can prefetch them from remote → local
DRAM before they're accessed?** Newest entries at the bottom.

> **Scope note — "PF" here is a *tier* prefetch, NOT the CPU `_mm_prefetch` hints in `Index.h`.**
> It means: bring a node (its links / data) from the **remote DRAM tier into the local tier ahead of
> the access**, using the candidate array as the *predictor*. The whole value is **lead time** — the
> fetch must be issued early enough to hide remote-DRAM latency before the search reaches the node.

Predecessors: [hbw_progress_log.md](hbw_progress_log.md) (search is memory-latency bound: 77–83% stall,
MLP ~3.4 ≈ 35% of the LFB ceiling) and [caching_benchmark_log.md](caching_benchmark_log.md) (NUMA tier:
hot graph links/vectors local, cold remote; Hub-placement at 60% local ≈ all-local QPS).

## Beam-search access model (what we prefetch, and the lead time we have)
Each node has two access events; the gap between them is the prefetch lead time.
- **Discovery** — node X is first reached as a neighbor → its **vector** is fetched (distance computed)
  → X is pushed into the candidate array (PQ ordered by distance-to-query).
- **Expansion** — X is popped (it's the closest pending) → its **links** are fetched and its neighbors
  discovered.

X sits in the PQ between discovery and expansion — **that residency is the only lead time available.**
So the prefetchable work is the **links (and next-level neighbor data) of nodes already in the PQ**, and
the predictor is the candidate array itself: every node in the PQ *will* be expanded later (or pruned).

## The claim, reframed (lead-time coverage — not "nodes visited")
Claim (refreshed): **~75%** of a step's expansions come from the prior candidate array. The
prefetch-meaningful version is about **future expansions**, not just-discovered neighbors (which have
zero lead time and are skipped on re-encounter via `visited_set`):

> **coverage(k) = fraction of nodes EXPANDED (popped) in steps N+1 … N+k that were already in the
> candidate array at step N.**

coverage(1) ≈ "is the next expanded node already in the frontier" (claim ≈ **75%**). `candidates[N−1] ∩
candidates[N+1]` is a fine 2-step proxy (the PQ slice that persists ≥2 steps = had ≥2 steps of lead).

## Where the missing ~25% comes from (the irreducible miss)
The ~25% of upcoming expansions **not** in `candidates[N]` are nodes **discovered inside the window**
(neighbors of an expansion at N+1…N+k−1) that are **strictly closer to the query**, so they **leapfrog**
to the top of the PQ and get expanded before the window ends. These are the **greedy-descent "progress"
moves** — you couldn't know their IDs at step N because they weren't discovered yet.
- **Lead-time dependent:** the miss grows with k (more intervening expansions ⇒ more leapfroggers) and
  shrinks toward convergence (filling the ef-neighborhood discovers little that's closer ⇒ coverage→~100%).
  So ~75% is an average — lower during early descent, higher near the end.
- **Mostly recoverable:** a leapfrogger is, by definition, an **out-neighbor of a candidate you already
  had.** ⇒ prefetching one hop deeper (candidates **+ their out-neighbors**) catches most of the 18%,
  trading extra prefetch bandwidth/footprint and less lead time for those neighbors.

## The design knob: prefetch depth (how many frontier hops)
- **depth-0** — prefetch the candidate set's own links/data → covers ~the coverage(k) fraction (~82%).
- **depth-1** — also prefetch candidates' out-neighbors → closes most of the leapfrog miss toward ~100%,
  at higher fetch volume and shorter lead time for the neighbor hop.

## Study plan
1. **Measure coverage(k) and the miss breakdown.** Instrument `beamSearch` to record, per step, the PQ
   snapshot vs the nodes popped over the next k steps. Output (a) **coverage(k) curve** k∈{1,2,4,8,16},
   (b) miss split: leapfrogger-is-neighbor-of-an-in-window-candidate (recoverable by depth-1) vs truly
   novel (expected ~0 in a NSW graph), (c) dependence on **ef** and **SSSP common-source vs per-query**.
2. **Lead-time budget.** Translate k (steps) into time at our per-step cost (~tens–hundreds of ns) vs
   remote-DRAM latency, to size the achievable prefetch window.
3. **Prototype + measure** the candidate-driven remote→local prefetch (depth-0 vs depth-1); report QPS,
   local-hit, per-node DRAM/UPI BW, recall (must stay flat). Build on `tools/tier_run.cpp` placement +
   `tools/hbw_search`.

## Open questions
1. Prefetch **links** (expansion, the random/remote-latency path — 78% of DRAM misses) vs **vectors**
   (discovery, the bandwidth) vs both?
2. Does coverage(k) hold for **per-query** entry, or is it specifically high under the SSSP common source?
3. Mechanism for remote→local "fetch": page migration (`move_pages`) vs a software local staging buffer —
   and is the lead time (k steps) actually enough to hide UPI latency?

## 2026-06-24 — Milestone 1: PQ-residency instrumentation + coverage(L) (`tools/pq_overlap.cpp`)

Instrumented `beamSearch` (macro `FLATNAV_PROFILE_PQ`): a node's prefetch lead time = its **PQ
residency = pop_step − discovery_step** (per-thread maps record discovery step + parent residency;
global atomic histograms aggregate). Corner case checked: the `visited_set` gate ⇒ each node is pushed
to the candidate PQ **at most once** (no add→remove→re-add), so residency is unambiguous. Ran 200k
queries, ef=200, **SSSP common source** (medoid id 65610822), 213.8 expansions/query.

**coverage(L) = P(residency ≥ L+1) = expansions predictable L steps ahead from the candidate array:**

| L steps ahead | 1 | 2 | 3 | 4 | 8 |
|---------------|------|------|------|------|------|
| coverage | **70.1%** | 65.3% | 63.1% | 61.6% | 58.0% |

- **Leapfrogger miss P(res==1) = 29.9%** — discovered during the immediately-prior expansion (greedy
  descent finding a strictly-closer node), zero start-of-step lead.
- **Fat residency tail** (r=16 still 0.52%): near convergence many expansions are long-resident PQ
  nodes, so a *deep* prefetch still covers ~58% at 8-step lead.
- **Depth-1 recovers nearly all leapfroggers:** of res==1 nodes, **P(parent res ≥1)=98.4%**,
  **P(parent res ≥2)=50.5%** → prefetching candidates' out-neighbors (when the parent entered the PQ)
  gives ≥1-step lead for ~all leapfroggers, ≥2-step for half.

**On the claim (~75%):** the rigorous, actionable number is **70.1% at 1-step lead** — **consistent with
the ~75% claim** (within ~5pp; the small remainder is snapshot timing: measured against the snapshot you
can actually act on, the *start* of the prior step). So the claim is essentially **validated**: ~70–75%
of the next expansion is already in the candidate frontier one step ahead.

**Takeaways for the prefetcher:** depth-0 (prefetch the candidate set) ≈ 70% at 1-step / ~58% at 8-step
lead; the ~30% leapfrogger gap is real but **almost fully recoverable with depth-1** (candidates +
out-neighbors), at the cost of more fetch volume. Next: (a) rerun **per-query** entry to see if coverage
differs from the common source; (b) ef dependence; (c) translate L (steps) → time vs UPI latency to size
the achievable prefetch window; then prototype depth-0 vs depth-1 remote→local prefetch.

## 2026-06-24 — Milestone 2: step-wise predictability (early descent vs late convergence)

Bucketed every expansion by its **search-step index** (pop/hop number within the query) and split
EARLY [0,5) vs LATE [5,N). Common source, ef=200, ~214 steps/query.

**Full per-step breakdown** (every search step; step = pop/hop index). Steps 0–212 are
reached by all 200k queries, counts taper after as queries finish (tail to step 411 = single
queries). Full-resolution copy: `.claude/assets/prefetch_stepwise_sssp.csv`.

EARLY [0,5): #exp=1,000,000   mean_res=0.87  leapfrog=74.29%  coverage(1)=7.14%
LATE  [5,N): #exp=41,950,985  mean_res=47.64 leapfrog=28.67%  coverage(1)=71.33%

**Finding — predictability rises sharply with search depth:**
- **Early descent (steps 1–4):** ~93% leapfrogger (excl. entry), **~7% coverage**, mean residency <1 ⇒
  **~zero prefetch lead.** Each step finds a strictly-closer node and expands it immediately (greedy
  descent from the medoid); nothing dwells in the PQ.
- **Late convergence (bulk, steps ~15–214):** coverage ~**75%**, **mean residency ~48 steps** ⇒ huge
  prefetch lead — deep-phase expansions pull from long-resident candidates.
- Transition is **gradual, later than step 5** (steps 5–12 still ~90% leapfrogger; ramps from ~step 13).
  The ~200-step deep tail dominates, so the overall coverage(1)=70% is essentially the late phase.

**Implication — two phases want two mechanisms (ties to the caching log):**
- The **early shared prefix** (first ~5–12 hops from the fixed medoid) is identical across queries and
  unpredictable from the PQ ⇒ **statically pin/cache it local** (it is the near-source super-hot core:
  caching log's ≤hop-5 ≈ 202K nodes, reused thousands of times). Don't prefetch it.
- The **deep convergence phase** (per-query bulk) is **highly predictable with ~48-step lead** ⇒ the
  sweet spot for **candidate-array tier-prefetch.**

## 2026-06-24 — Milestone 3: DEFAULT FlatNav (per-query entry) — same study

Reran the identical instrumentation with the **default per-query entry** (`initializeSearch`: closest of
100 sampled nodes), ef=200, 200k queries. 205.6 expansions/query (vs 213.8 for the common source). Full
per-step breakdown: `.claude/assets/prefetch_stepwise_default.csv` (cf. `..._sssp.csv`).

**Overall coverage(L) — default vs common-source:**

| L steps ahead | 1 | 2 | 4 | 8 |
|---------------|------|------|------|------|
| **default (per-query)** | **72.3%** | 67.7% | 64.0% | 60.3% |
| common-source (SSSP)    | 70.1% | 65.3% | 61.6% | 58.0% |

Leapfrogger P(res==1): default **27.7%** vs cs 29.9%. Depth-1 parent lead P(parent≥2): default 55.1% vs
cs 50.5%. EARLY[0,5)/LATE[5,N) coverage: default **1.9% / 73.7%**; cs 7.1% / 71.3%.

**Finding — same ~70% endpoint, different *shape* (crossover at ~step 5–6):**

| step | 2 | 4 | 6 | 8 | 10 | 15 | 19 |
|------|----|----|----|----|----|----|----|
| default leapfrog% | 99.2 | 95.4 | 87.0 | 74.9 | 63.9 | 47.8 | 41.8 |
| common  leapfrog% | 95.7 | 87.9 | 90.0 | 90.2 | 89.2 | 74.2 | 59.4 |

- **Default starts *near* the target**, so steps 1–4 are immediate fine descent (≈95–99% leapfrog, *worse*
  than cs) — but then **ramps fast and monotonically** (52% coverage by step 15 vs cs 26%). No plateau.
- **Common-source plateaus** at ~90% leapfrog for ~12 steps — the **long shared transit from the distant
  medoid** to each query's region — then ramps late.
- Net: default's short per-query descent → faster ramp ⇒ marginally **higher** overall coverage (72 vs 70%).

**Design implication — the two designs want different prefetch strategies:**
- **Default (per-query):** the early descent is **short and per-query (different entries) ⇒ NO shared
  cacheable prefix.** But predictability ramps by ~step 6, so **candidate-array prefetch covers nearly the
  whole search** on its own; little to statically pin.
- **Common-source (SSSP):** the early descent is a **long ~12-step shared prefix** that IS statically
  cacheable, but unpredictable from the PQ ⇒ **cache the prefix + prefetch the deep phase.**

## 2026-06-24 — Milestone 4: top-K prefetch policy — precision / waste / lead sweep

Added the **policy** instrumentation (same `FLATNAV_PROFILE_PQ`): each step, snapshot the **K closest
pending candidates** (the nodes we'd prefetch); a node is prefetched once at its **first top-K entry**;
**success** = eventually popped, **waste** = never popped; lead = pop_step − first-prefetch-step. Common
source, ef=200, 20k-query subset (the per-step PQ copy is heavy). Per-step CSV: `prefetch_policy_sssp.csv`.

Every popped node is rank-1 the step before its pop ⇒ it's always captured ⇒ **recall ≈ 100% for any K**;
the real tradeoff is **waste vs lead**:

| K | precision (success/prefetched) | waste rate | mean lead (steps) |
|---|--------------------------------|-----------|-------------------|
| 1  | **99.5%** |  0.5% | 1.0  |
| 5  | 83.1% | 16.9% | 6.9  |
| 10 | 70.3% | 29.7% | 13.0 |
| 50 | 39.9% | 60.1% | 39.7 |

**Deeper K buys lead time but wastes bandwidth** — K=1 is accurate but only 1-step lead (useless for
hiding remote latency); K=50 gives ~40-step lead but 60% of prefetches are wasted.

**The waste is concentrated in early descent** (precision by phase):

| K | EARLY (steps<5) precision | LATE (≥5) precision |
|---|---------------------------|---------------------|
| 5  | 25.9% | 87.9% |
| 10 | 14.7% | 77.6% |
| 50 |  6.8% | 45.3% |

Per-step (K=10) precision *rises sharply with depth*: step 1 = 15.6%, step 10 = 28%, step 20 = 77%,
step 50 = 98.7%, step 100 = ~100%. Same shape as the predictability curve — the churny early frontier
makes top-K prefetch mostly wasted; the stable convergence frontier makes it almost always right.

**Design implication (ties Milestones 2–4 together):**
- **Don't prefetch the early descent** — the frontier churns, top-K is ~85% wasted. Use the **static
  cache** for the shared prefix (SSSP) instead.
- **Prefetch the convergence phase with a moderate K (~10)** — gated to steps ≳15, K=10 delivers ~13-step
  lead at ~78–99% precision. K=50 only if you need ~40-step lead and can afford 60% waste.

## 2026-06-24 — Milestone 5: top-K policy on DEFAULT (per-query) entry

Same policy sweep, default per-query entry, ef=200, 20k queries. Per-step CSV: `prefetch_policy_default.csv`.

| K | precision (default / sssp) | mean lead (default / sssp) |
|---|----------------------------|----------------------------|
| 1  | 99.5% / 99.5% | 1.0 / 1.0 |
| 5  | **89.1%** / 83.1% | 7.1 / 6.9 |
| 10 | **78.9%** / 70.3% | 13.4 / 13.0 |
| 50 | **46.5%** / 39.9% | 41.2 / 39.7 |

**Default per-query wastes *less* prefetch bandwidth at every K** (higher precision, same lead) — because
it starts near the target, the frontier stabilises sooner, so top-K nodes get popped more reliably.

Phase precision (default vs sssp):

| K | EARLY default / sssp | LATE default / sssp |
|---|----------------------|---------------------|
| 5  | 28.2% / 25.9% | **95.6%** / 87.9% |
| 10 | 16.1% / 14.7% | **90.5%** / 77.6% |
| 50 |  5.8% / 6.8%  | **60.6%** / 45.3% |

Early descent is ~equally wasteful in both (≈15–28% precision), but the **convergence phase is markedly
cleaner under default** — K=10 hits **90.5% precision** late (vs 77.6% for sssp). So for default FlatNav,
**K=10 gated to the convergence phase ≈ 13-step lead at ~90% precision** — a strong operating point.

## 2026-06-24 — Milestone 6: PRE-expansion snapshot (leapfroggers excluded from the policy)

Changed the top-K policy itself (not an added slot): the snapshot now happens **just after the pop,
before `processCandidateNode` discovers the node's neighbors**, so this step's leapfroggers — not yet
born — are excluded from **every** K. This is the "prefetch the next candidate at the start of the
current expansion" policy. Both entry modes, **full 200k**, ef=200. Artifacts:
`prefetch_pre_{default,sssp}.out`.

| K | default precision / mean_lead | sssp precision / mean_lead |
|---|-------------------------------|----------------------------|
| 1  | **96.2%** / **2.33** | **93.0%** / **2.32** |
| 5  | 83.0% / 10.28 | 73.7% / 10.18 |
| 10 | 71.0% / 18.78 | 60.0% / 18.61 |
| 50 | 38.5% / 56.70 | 31.3% / 56.22 |

**The big reframe — recall is now capped at coverage(1), NOT ~100%.** `success(popped)` is **constant
across K** (default 29.74M = 72.3% of expansions; sssp 29.98M = 70.1%) = exactly coverage(1). The
leapfroggers (28–30%) are **structurally un-prefetchable**: discovered and popped between two
snapshots, so no K ever sees them. Milestone 4's "recall≈100% for any K" was an artifact of the
**post**-expansion snapshot capturing the leapfrogger 1 step before its immediate pop (zero useful lead).

**K=1 already captures the entire prefetchable set.** Every resident (non-leapfrog) node is rank-1
pre-expansion at some step ⇒ K=1 success == K=50 success. Larger K adds **zero recall** — it only buys
lead (nodes enter the top-K earlier) at the cost of waste. So **K is purely a lead-vs-waste dial;
recall is fixed at coverage(1).**

**Answer to "K=1 without the leapfrogger":** lead **2.33 steps** (default) / **2.32** (sssp) at **93–96%**
precision — vs the old post-expansion K=1's 1.0 lead / 99.5%. Excluding the leapfrogger **more than
doubles** the lead (the K=1 pick is the true 2nd-min, displaced ~2.3 steps by chains of closer
leapfroggers) and exposes the **honest** precision: the 4–7% waste = 2nd-min picks pruned out of the
ef-window before they pop (the old K=1 could never waste, since its pick was literally the next pop).

**All waste is early-descent; convergence is clean** (K=1 precision by phase):

| K=1 | EARLY (<5) | LATE (≥5) |
|-----|-----------|-----------|
| default | 9.3% | **98.4%** |
| sssp    | 11.6% | **94.8%** |

Default still cleaner than sssp at every K (Milestone 5 holds).

**Design takeaway:** **K=1 pre-expansion, gated to the convergence phase**, is a strong minimal policy —
~2.3-step lead, ~95–98% precision, captures 100% of the prefetchable (non-leapfrog) set with minimal
waste. To go past ~2 steps of lead, raise K (buys lead only, no recall, growing waste). To cover the
~30% leapfroggers **at all** needs depth-1 (candidate + out-neighbor) prefetch — K alone can't.

## Status: Milestones 1–6 done. coverage(1)≈70–72% = the hard recall ceiling for candidate-array prefetch.
**Default per-query is the cleaner prefetch target.** Pre-expansion policy: K=1 ⇒ ~2.3-step lead / ~95%
precision (convergence ~98%), captures the full non-leapfrog set; K only trades lead for waste. Leapfroggers
(~30%) need depth-1. Artifacts: `prefetch_pre_{default,sssp}.out`, `prefetch_policy_{sssp,default}.csv`,
`prefetch_stepwise_{sssp,default}.csv`. Next: ef dependence; depth-1 (out-neighbor) prefetch to attack the
leapfrog ceiling; prototype remote→local prefetch gated to convergence.

## 2026-07-17 — Milestone 7: PROTOTYPE — candidate-driven cache prefetch (links & vectors), QPS/miss/perf

Moved from predictability study to an actual **executable prefetch** and measured end-to-end. Mechanism is a
**cache-line prefetch** (`_mm_prefetch`, remote line → local cache hierarchy, hides UPI latency), NOT page
migration — the ~2.3-step lead (M6) is nanoseconds-to-µs, which fits a cache prefetch but is far too short for
a `move_pages` syscall (µs + TLB shootdown across 32 threads); DRAM-tier migration only amortises over reuse
(= the caching phase), so it's not a per-access prefetcher.

**Setup:** SIFT100M, M=32, 60 GB flatnav index, 200k queries, K=100, ef=200, 32 workers pinned to node 0
(Skylake-SP 2-socket). Driver `tools/hbw_search.cpp` via `scripts/bench.sh`. New `CACHE=1` bench mode = a
dedicated perf run with only 2 GP events (`mem_load_retired.l3_miss`/`.l2_miss`) + fixed + IMC → **no counter
multiplexing** (verified 100% running). Three opt-in macros in `Index.h`, each **independent of
`FLATNAV_DISABLE_PREFETCH`** (so the existing SSE prefetches can be off, ours on, no confound):
`FLATNAV_PF_LINKS` (Exp A), `FLATNAV_PF_VEC_BURST` / `FLATNAV_PF_VEC_AHEAD` (Exp B). All runs `PREFETCH=0`
baseline = **vs no prefetch at all**. Recall stayed **0.8921 across every run** (pure hints). 3-rep variance,
run-to-run QPS spread ~±0.04% (effects below are cleanly separated → real).

### Exp A — prefetch the next candidate's LINKS at pop (depth-0, T1, full ~3-line block)
The pop-site block already prefetched the next candidate's *vector* — but that vector is resident from
discovery and is NOT read at expansion; the expansion reads its **links** (the remote pointer-chase, 78% of
DRAM misses), which were un-prefetched. Fix = prefetch `getNodeLinks(next)` (every cache line) at the same
post-pop / pre-expansion instant (M6's K=1 snapshot). T1 (L2, skip L1) since links are used a full expansion
later.

| placement | QPS Δ | L3 miss Δ | L2 miss Δ | p50 Δ |
|-----------|-------|-----------|-----------|-------|
| all-local           | **+0.78%** | −3.9% | −3.5% | −1.0% |
| links remote (node1)| **+1.0%**  | −3.1% | −2.8% | −1.1% |

Modest but real, **concentrated in the median** (p50 −1%), tail (p99/p99.9) flat-to-slightly-worse. **Local ≈
remote** benefit ⇒ not primarily hiding UPI; just starting the pointer-chase a bit earlier. Links are only 3
lines, so cheap.

### Exp B — prefetch the VECTORS, stressed with ALL VECTORS REMOTE (node 1)
Vectors are the bulk of the bytes (M×512 B/expansion) — so all-remote is where a vector prefetch has real UPI
latency to hide. `ceiling` = vectors local. Two mechanisms: **B-burst** (prefetch this node's M neighbor
vectors upfront at expansion start) and **B-ahead** (depth-1: prefetch the *next* candidate's neighbor
vectors — requires a real read of next's links to get the ids).

**First cut (heavy: full ~8-line vector, all M neighbors):** *counterintuitive* — massive miss reduction, QPS
**worse**.

| config | QPS | vs ceiling | L3 miss |
|--------|-----|-----------|---------|
| ceiling (vec local) | 17,184 | — | 5,913 M |
| remote_off          | 14,004 | −18.5% | 6,211 M |
| remote_burst heavy  | 13,320 | −22.5% | **852 M (−86%)** |
| remote_ahead heavy  | 12,830 | −25.3% | 2,400 M (−61%) |

**PERF=1 diagnosis (the key result):** the regime is **memory-LATENCY-bound, not bandwidth-bound** —
`remote_off` util only **38%** (40/103 GB/s), mem_stall **83%**. The prefetch **did hide the latency**
(remote_burst: stall 83%→**49.5%**, IPC 0.203→0.242, demand misses −86%) — but its **instruction overhead
exceeded the savings**: +25% instructions (256 `_mm_prefetch`/expansion: 8 lines × 32 neighbors incl.
already-visited) ÷ +19% IPC ≈ **+5% cycles → −5% QPS**. Arithmetic closes exactly (observed −4.3%). So earlier
"bandwidth wall / latency already hidden" framing was WRONG: it's latency-bound, latency IS hidden by the
prefetch, but the software-prefetch instruction cost dominates.

**Lightweight fix (1 cache line/vector + skip already-visited neighbors) → FLIPS POSITIVE:**

| B-burst | QPS vs remote_off | L3 miss Δ | instr Δ |
|---------|-------------------|-----------|---------|
| heavy | **−4.9%** | −86% | +25% |
| **light** | **+4.1%** | −21% | **+3.4%** |

Lightweight `remote_burst` = **14,586 QPS, +4.1% over remote_off** (recovers ~3.3 of the ~19 pp remote
penalty ≈ 1/6 of the gap), flat recall, latency mean −4.0% / p50 −4.6% / p90–p99 better (p99.9 ~flat).
**B-ahead stays neutral** (+0.15%) even lightweight — one-step-ahead mispredicts + the next-link read cancel
the gain. **B-burst is the mechanism that works.**

### Takeaways / next
- **Cache prefetch is the right mechanism** at this lead (not migration). **Links (Exp A): +~1%**, cheap,
  median-only. **Vectors (Exp B): +4%** once made lightweight, but only recovers ~1/6 of the remote penalty.
- The remote-vector penalty is **latency** (util 39%, headroom exists), and lightweight prefetch chips at it;
  the **bulk still needs tiering** (hot vectors local → less UPI traffic). Prefetch + tiering are
  **complementary**.
- **Overhead, not bandwidth, is the binding constraint** for software vector prefetch — keep it minimal
  (1 line, skip-visited).
- Open knobs: 1-vs-2 lines, T0 vs T1, convergence-phase gating, Exp-A links prefetch stacked on B-burst,
  and B-burst over a **fractional-local** vector placement (prefetch × tiering).
- Artifacts: `run_tableB2.log` (lightweight), `run_perfB.log` (PERF), `run_variance.log` (Exp A). Macros in
  `include/flatnav/index/Index.h`; `CACHE=1` in `scripts/bench.sh`.

## 2026-07-21 — Milestone 8: fan-out per step + lead-time-vs-progress (is there PF value?)

Gate check before the hub→neighborhood mapping study: does the prefetch *opportunity* (lead
time) actually exist where the fetches are? Added `g_step_fanout[step]` to `FLATNAV_PROFILE_PQ`
(counts NEW/unvisited-neighbor vector fetches per expansion, bucketed by pop step) + a
`mean_fanout` column in `pq_overlap.cpp`'s stepwise CSV. Reran the M1 setup on clnode222:
SIFT100M, ef=200, **200k queries**, 32T node0 (search 22.5s). Fan-out is the per-step fetch
rate; overlay it on `mean_residency` (lead time) and `leapfrog_pct`.

| step | mean_fanout (new fetches/exp) | mean_res (lead, steps) | leapfrog% | coverage(1)% |
|------|-------|-------|-------|-------|
| 1    | 25.2  | 1.0   | 100%  | 0%    |
| 5    | 24.4  | 1.1   | 92%   | 8%    |
| 10   | 23.2  | 1.7   | 64%   | 36%   |
| 20   | 21.9  | 4.4   | 41%   | 59%   |
| 40   | 20.5  | 12.4  | 31%   | 69%   |
| 80   | 19.2  | 33.3  | 24%   | 76%   |
| 160  | 17.8  | 81.3  | 19%   | 81%   |

Fetch-weighted aggregates over all ~796M fetches:
- **~3,983 fetches/query** (~207 expansions/query) — high for a k=100 search.
- **Fetch-weighted mean lead = 45.7 steps.**
- **91.6%** of fetches occur *past* the greedy zero-lead regime (leapfrog < 50%).
- **70.6%** fetch-weighted coverage(1) (fetches at steps whose expansions had ≥1-step lead).

Findings:
1. **Fan-out is essentially FLAT** — ~25 early decaying only to ~18 late; it does NOT collapse.
   Every expansion keeps fetching ~18–25 genuinely new vectors regardless of phase ⇒ there is
   **no fetch-cheap phase**; the remote-fetch load is spread across the whole search.
2. **Lead time grows monotonically** 1→~80 steps as leapfrog% falls 100%→19% (greedy descent →
   convergence). The zero-lead greedy phase is SHORT (~first 10 steps) and holds only ~8% of
   fetches; ~92% of fetches carry real lead.
3. **Value-in-PF verdict: lead time is NOT the constraint.** The average fetch sits behind ~46
   steps of lead. So the earlier +4.1% cache-prefetch ceiling (M7) was a **cost / BW-wall**
   limit, not a lack of opportunity — a cache hint just reorders bytes against the fixed ~24
   GB/s remote-read wall (sec 4 of `prefetch-mechanism.log`). The useful flip side: **46 steps
   of lead is far more than a cache hint needs — it's enough to STAGE or page-MIGRATE a
   predicted region to local DRAM** (µs-scale). That is exactly the lead budget the hub→
   neighborhood speculation needs: predict the late region during the short early phase, move
   it local with dozens of steps to spare.
4. ~4,000 fetches/query for k=100 ⇒ most fetched vectors never enter the result — reinforces
   that **reducing fetch COUNT** (speculation / predict-and-pre-place) is the high-value lever,
   not hiding latency that is already ~fully leadable.

⇒ **Green-lights the hub→neighborhood mapping study**: the lead-time budget exists and grows
where the fetches are; the only open question is **predictability** (does the early-hub
signature predict the late neighborhood?). A positive result would justify staging/migration
(a byte-reducing lever), not just the +4% cache hint.

Artifacts: `.claude/assets/prefetch_stepwise_fanout_default.csv` (per-step, +`mean_fanout`).
Instrumentation: `g_step_fanout` in `include/flatnav/index/Index.h` (FLATNAV_PROFILE_PQ),
`mean_fanout` column in `tools/pq_overlap.cpp`. Binary `~/pq_overlap_bench` on clnode222.
Next: hub→neighborhood predictability (early top-X% in-degree hubs vs late-phase activated set).

## 2026-07-22 — Milestone 9: hub → terminal-basin predictability (speculation gate) — CLOSED

Follow-on to M8's green light. Hypothesis: greedy search rides a hub highway then
descends into a terminal "basin"; if the early-hub signature PREDICTS the basin, you
could speculatively stage/migrate it with M8's ~46-step (~510 µs) lead. Built a C++
port of a basin-extraction script over the real 100M index (`tools/basin_map.cpp`,
new `FLATNAV_PROFILE_TRAJ` hook: per-query expanded-node trajectory + closest/terminal
node). Basin = co-occurrence cluster of terminal-core nodes (union-find). Signature =
LAST hub before the terminal core. SIFT100M, ef=200, 200k probes, clnode222.

The in-sample number looked perfect and was a TRAP: U(basin|hub)=0.98, top1=87%. But
basins don't populate (queries/basin≈1.3, H(basin)≈log2(#queries)) → basin ≈ a per-
query id, so predicting it from a near-unique last-hub, in-sample, is memorization.
The **held-out** test (train hub→basin table on half, predict the other half) is the
decisive control and destroys it: **+0.1 pp lift**, only 13% of test hubs seen in train.
A non-circular geometric funnel (group terminals by the PREDICTOR, between/within
variance) shows a REAL but MODEST signal: recurring last-hubs concentrate terminal
LOCATION ~2.2x (var_ratio 4.9) — enough to matter, nowhere near enough to pre-place.

Re-ran with ACCESS-frequency hubs (the true empirical highway; in-degree hubs overlap
it only ~6% at top-1%, see caching log 2026-07-22) via HUB_METRIC=link:

| Metric | in-degree hubs | access(link) hubs |
|--------|---|---|
| queries w/ a hub | 79.8% | 100.0% (access = real highway: every query) |
| mean lead | 46.2 steps | 16.8 steps (~510 -> ~190 us) |
| queries/hub(funnel) | 8.30 | 1.19 |
| held-out lift | +0.1 pp | +0.0 pp |
| geometric var_ratio | 4.90 | 5.05 |

Using the faithful highway does NOT help and slightly hurts: it's crossed by 100% of
queries (confirming it's the real highway) but its exit sits much later (hot nodes are
traversed near convergence too), so **lead collapses 46->17 steps** while held-out lift
stays ZERO. The more faithfully you define the highway, the later its exit, the less
runway. Funnel is genuinely absent under both hub definitions.

Verdict: **hub → neighborhood speculation CLOSED under both in-degree and access hubs.**
Lead time exists (M8) but predictive RESOLUTION does not: the last hub (however defined)
barely recurs across queries and narrows terminal location only ~2x. Consistent with the
whole prefetch/MLP arc — the exploitable lever is REDUCING FETCH COUNT (compressed-code
speculation), not predicting location from the traversal.

Methodology note (why the first read was wrong): a label that is ~unique per sample,
predicted from a feature ~unique per sample, evaluated IN-SAMPLE, gives ~100% by
construction. Always held-out-evaluate a predictability claim. Tools: `basin_map.cpp`
(FLATNAV_PROFILE_TRAJ + HUB_METRIC=indeg|link|data), binary `~/basin_map_bench`.

## 2026-07-22 — Milestone 10: when are the top-K found? (early-termination headroom)

Q: for each query, at which expansion STEP is each final top-K result first discovered
(distance-computed)? New instrumentation `FLATNAV_PROFILE_TOPK` (Index.h, reads tl_disc):
at beamSearch end, sort the final beam, take the K smallest, record each one's discovery step
(g_topk_found), the LAST one's step (g_topk_complete = "answer complete"), and the 1-NN's
step. Tool `tools/topk_found.cpp`. SIFT100M, ef=200, 200k queries, K=100. NOTE: this is a
property of search DYNAMICS (which node found at which step) -> placement/remote%-INVARIANT.

  mean steps/query = 206.6   1-NN found @ step 26.5   answer-complete @ step 182.8
  -> only 11.5% of the search runs AFTER the top-100 is assembled.

| bin[steps] | %top100 found | cumFound | %queries complete | cumComplete |
|---|---|---|---|---|
| 0 [  0, 30) | 49.1% | 49.1% | 0.0% | 0.0% |
| 1 [ 30, 60) | 22.7% | 71.7% | 0.2% | 0.2% |
| 2 [ 60, 90) | 11.2% | 82.9% | 1.7% | 1.9% |
| 3 [ 90,120) | 6.8% | 89.7% | 3.5% | 5.4% |
| 4 [120,150) | 4.6% | 94.3% | 6.6% | 11.9% |
| 5 [150,180) | 3.3% | 97.6% | 16.8% | 28.8% |
| 6 [180,210) | 2.3% | 99.9% | 68.4% | 97.1% |

Finding: **results are found EARLY, but the search runs LONG to nail the last few.**
Half the top-100 is found in the first 30 steps, 90% by step 90 (~44% of the search); the
1-NN by step 26. BUT the LAST member of the top-100 arrives at step ~183 (89% through), and
68% of queries only complete in bin 6 (180-210). The search grinds nearly to the end hunting
the last 1-3 stragglers (hard, far-flung true neighbors needing deep exploration).

Two headrooms:
 1. Full high-recall top-100 -> ~NO headroom: search already stops just 11.5% after completion
    (beam terminates tightly once no better candidate remains).
 2. Relaxed recall -> HUGE: the last ~10% of the result set costs ~56% of the search.
    1-NN: stop at ~13% of steps. 90 of 100 true neighbors: stop at step ~90 = ~44% of the
    search (~56% saved). The 99->100% convergence tail is the expensive part.

Ties the arc together: most of the ~4000 fetches/query (M8) and most of the steps go to the
last few result members, not the first many. The exploitable lever is cutting the STRAGGLER
HUNT (early-termination at a recall target, or a shortcut predictor), NOT hiding latency.
Instrumentation: `FLATNAV_PROFILE_TOPK` in Index.h + `tools/topk_found.cpp`; binary
`~/topk_found_bench` on clnode222. Next candidate: recall@100 vs step-budget curve.

### M8 x M10 joint cumulative table — cost (fetches) vs benefit (results), same step axis

Both milestones are the same config (SIFT100M, ef=200, 200k queries, DEFAULT entry, k=100) and both
are placement-invariant search dynamics, so their step axes join directly. Cost side recomputed
per-bin from `.claude/assets/prefetch_stepwise_fanout_default.csv` (sum of `n_expansions x
mean_fanout`, totals reconcile: 206.6 exp/q, 3982.8 fetches/q). Benefit side from M10's bins.

| bin[steps] | fetch/q in bin | cumFetch/q | cumFetch% | cumExp% | mean_res fw | mean_res uw | cumFound% (top-100) | cumComplete% (queries) | fetches per result found in bin |
|---|---|---|---|---|---|---|---|---|---|
| 0 [  0, 30) | 683.2 | 683.2  | 17.2%  | 14.5% | 3.2   | 3.3   | 49.1% | 0.0%  | 13.9  |
| 1 [ 30, 60) | 611.1 | 1294.3 | 32.5%  | 29.0% | 14.6  | 14.7  | 71.7% | 0.2%  | 26.9  |
| 2 [ 60, 90) | 580.8 | 1875.1 | 47.1%  | 43.6% | 30.2  | 30.3  | 82.9% | 1.9%  | 51.9  |
| 3 [ 90,120) | 560.8 | 2436.0 | 61.2%  | 58.1% | 47.4  | 47.5  | 89.7% | 5.4%  | 82.5  |
| 4 [120,150) | 545.9 | 2981.8 | 74.9%  | 72.6% | 65.4  | 65.4  | 94.3% | 11.9% | 118.7 |
| 5 [150,180) | 533.7 | 3515.5 | 88.3%  | 87.1% | 83.9  | 84.0  | 97.6% | 28.8% | 161.7 |
| 6 [180,210) | 455.1 | 3970.6 | 99.7%  | 99.7% | 99.9  | 97.4  | 99.9% | 97.1% | 197.9 |
| 7 [210,  -) | 12.1  | 3982.8 | 100.0% | 100%  | 64.2  | 67.1  | 100%  | 100%  | -     |

`mean_res fw` = **fetch-weighted** lead per bin (each step's mean_residency weighted by that step's
fetches) — the staging budget for a typical fetched byte. `mean_res uw` = **unweighted** per-step mean
(each of the bin's steps counted equally). They track within ~0.1 step through bin 5 (fan-out is flat,
M8) and diverge only in the tail bins where fan-out finally decays. Coverage(1) per bin: 39.6 / 70.0 /
74.9 / 77.7 / 79.5 / 80.9 / 81.2 %.

Reading:
- **Cost is FLAT, benefit DECAYS.** Fetches accrue almost linearly (17%/15%/15%/14%/14%/13%/11% per
  bin — M8's flat fan-out) while results accrue 49%/23%/11%/7%/5%/3%/2%. Marginal price of a result
  member escalates **14x**: ~14 fetches for an early member, ~198 for a late straggler.
- **Halfway through the fetch budget (47%) you already hold 83% of the top-100.** The last 10 members
  cost ~2,100 fetches/query (~53% of all remote traffic).
- The lead column shows the two levers are on **opposite ends**: the cheap-result phase (bin 0) has
  ~3 steps of lead (nothing to prefetch/stage against), while the deep lead (48-100 steps, enough to
  page-migrate) sits exactly in the expensive straggler bins. So staging can only ever subsidize the
  phase that early termination would rather DELETE.
- Consolidated verdict for the arc: **cut the tail, don't hide it.** Every latency-hiding lever (M7
  +4.1% cache PF, M9 closed speculation) operates on bytes that a recall-target early stop would not
  fetch at all.

## 2026-07-24 — Milestone 11: step-gated PQ traversal — recall@100 vs gate length N — REFUTED

Q: the PQ-tiering design (`.claude/assets/pq_prefetch_engine_design.md`) assumes exactness is
needed LATE ("early steps are greedy descent, the approximate distance ranks direction well
enough"; last top-100 member lands ~step 183, M10). Its stated verification: run traversal on
PQ for the first N expansions and exact after, and check recall vs N — recall should stay flat
until N is fairly large. New hook `FLATNAV_PQ_GATE` (Index.h) + `tools/pq_stepgate.cpp`
(self-contained PQ: per-subspace k-means, node-id-indexed codes, per-query LUT).
SIFT100M, ef=200, K=100, 200k queries, 32T node0, PQ m=16/8bit (16B/vec, 32x), mean relative
distance error 5.3%. Gate N counts EXPANSIONS (pops); at the switch the beam AND the pending
candidates are rescored exact (deduped), so everything still alive is corrected; if the search
ends inside the PQ phase the final beam is rescored — every recall below is top-K by EXACT
distance.

Validation: gate=0 reproduces the baseline recall 0.8921 AND `exact/q` = 3982.8, matching M8's
measured 3,982.8 fetches/query to the decimal.

| N (exp) | recall@100 | Δ recall | pq-scored/q | %of dists | dropped uncorrected/q | vec_reads/q | traffic saved |
|---|---|---|---|---|---|---|---|
| 0 (exact) | 0.8921 | —       | 0      | 0%    | 0     | 3982.8 | —     |
| 5         | 0.8922 | +0.01pp | 123    | 3.1%  | 0     | 3991.9 | -0.2% |
| 10        | 0.8924 | +0.03pp | 242    | 6.1%  | 1     | 4003.8 | -0.5% |
| 20        | 0.8928 | +0.07pp | 465    | 11.7% | 50    | 3977.7 | 0.1%  |
| 40        | 0.8834 | -0.87pp | 886    | 22.3% | 324   | 3840.5 | 3.6%  |
| 60        | 0.8635 | -2.86pp | 1288   | 32.3% | 662   | 3727.3 | 6.4%  |
| 80        | 0.8437 | -4.84pp | 1677   | 42.1% | 1015  | 3620.6 | 9.1%  |
| 120       | 0.8119 | -8.02pp | 2427   | 60.9% | 1725  | 3402.3 | 14.6% |
| 160       | 0.7889 | -10.3pp | 3151   | 79.1% | 2426  | 3175.3 | 20.3% |
| 200       | 0.7720 | -12.0pp | 3856   | 96.8% | 3116  | 2942.2 | 26.1% |
| all-PQ    | 0.6917 | -20.0pp | 3992   | 100%  | 3792  | 200.0  | 95.0% |

`dropped uncorrected/q` = pq_scored - rescored = nodes scored approximately and evicted from
beam ∪ candidates before the switch could correct them — structurally unrecoverable.

Findings:
- **The "exactness is only needed late" hypothesis is REFUTED.** Recall is flat only through
  N=20 (~10% of the ~207-step search, 11.7% of distances) and is already breaking at N=40.
  Exactness starts mattering in the DESCENT, not in the late fine-ranking.
- **The loss is a recall LEAK, not ranking noise** (design open question #4, now answered and
  it is the dominant term). Survivors of the PQ phase are all corrected at the switch, so the
  only loss channel is beam eviction during the PQ phase — and `dropped uncorrected/q` tracks
  the recall curve tightly (50 -> no loss; 324 -> -0.9pp; 1015 -> -4.8pp).
- **The step gate is dominated at both ends.** The recall-neutral region (N<=20) saves 0.1% of
  fetches = nothing; every N that saves real traffic costs recall. Per 1% of traffic saved the
  EXTREMES beat the middle: all-PQ 0.21pp, N=200 0.46pp, N=120 0.55pp. Partial gating is the
  worst of both worlds — if using PQ, go all the way and rerank.
- **Early termination strictly dominates the step gate.** M10: ~10pp recall buys ~56% of the
  work. Here N=160 gives the same ~10pp for only 20.3%. Consistent with the arc's verdict,
  cut the tail, don't hide it.
- **The all-PQ row is the live option:** 0.6917 at 5.0% of baseline traffic (the only 200
  reads/query are the final beam rerank). That is the capacity play, and it IS what the design
  actually describes (walk on codes, rerank exact) — the step gate was only a probe of the
  "late" intuition. Caveat: vec_reads/q excludes the per-query entry-point initialization, a
  fixed cost negligible against 3,983 but not against 200.
- **The switch REOPENS the search** — the gate makes queries longer, not shorter (below), so
  traffic saved lags the gate length and the mid-gates are worse than the recall column implies.

### The switch REOPENS the search (why N=200 still reads 74% of baseline)

Mean search is ~207 steps (M10), so a gate at N=200 should leave nothing to do exactly — yet
vec_reads/q = 2942.2. Summing both distance counters shows why: the gated search does far MORE
total work than the baseline.

| N | 0 | 20 | 40 | 60 | 80 | 120 | 160 | 200 | all-PQ |
|---|---|---|---|---|---|---|---|---|---|
| total dists/q | 3982.8 | 4028.0 | 4164.4 | 4389.6 | 4635.5 | 5126.9 | 5601.4 | 6058.2 | 3991.5 |
| vs baseline | 100% | 101.1% | 104.6% | 110.2% | 116.4% | 128.7% | 140.6% | **152.1%** | 100.2% |

At N=200: 2202.7 exact distances at late fan-out ~18-19 (M8) ≈ **120 extra expansions** (~320
steps, not ~207). Mechanism: the PQ phase prunes on noisy distances so `max_dist` is wrong; the
rescore installs true distances, `max_dist` moves, the candidate queue is full of nodes that now
beat it, termination stops holding, and the search re-expands to convergence — re-fetching
vectors it already touched. Cost = 739.5 rescore reads PLUS a 2202.7-read re-convergence phase
the baseline never had. Control: **all-PQ never switches and sits at 100.2%** — all inflation is
caused by the switch. So `traffic saved` understates the damage (saved reads come with 16-52%
more steps); on QPS the mid-gates are likely a net loss even ignoring recall.

Consequence for the design: the exact-fetch trigger CANNOT be "how deep are we" (a phase/step
rule). It must be per-node — the quality gate of Part 2 (proximity to the K-th best + untrusted
code). The work inflation generalizes this: correcting a PQ-ranked beam necessarily reopens
exploration, so ANY phase-switch pays a re-convergence phase — the argument is against
phase-switching itself, not just against switching late. Next candidate: all-PQ traversal with
the rerank budget swept (raise ef, cheap on codes, and/or rerank deeper than the final beam)
-> recall vs exact-reads/query frontier.
Instrumentation: `FLATNAV_PQ_GATE` in Index.h + `tools/pq_stepgate.cpp` + `scripts/pq_stepgate.sh`;
binary `~/pq_stepgate_bench`, raw `~/pq_stepgate_0724_1340/sweep.log` on clnode222.

## 2026-07-24 — Milestone 12: PQ WINDOW sweep — which phase needs exactness (dose held equal)

M11's prefix gates [0,N) conflate WHEN PQ is used with HOW MUCH — deeper N also means more
nodes PQ-scored. M12 slides a FIXED-WIDTH window [lo,hi) so each run has ~equal PQ dose
(~530-680 pq-scored/q); the recall differences are then PHASE, not dose. Extended `FLATNAV_PQ_GATE`
to a window (setPQGate(codes, m, lo, hi); expansions in [lo,hi) scored on PQ, exact outside;
beam+candidates rescored exact at hi). Width 30 aligns with M10's discovery bins. Same config
(SIFT100M, ef=200, K=100, 200k q, m=16/8bit, err 5.3%). Validation: window 0:0 reproduces
recall 0.8921 and exact/q 3982.8; prefix windows reproduce M11 bit-for-bit.

| window [lo,hi) | recall@100 | Δ vs exact | pq-scored/q | total_dists vs base |
|---|---|---|---|---|
| 0:0 (exact) | 0.8921 | —       | 0    | 100.0% |
| 0:30        | 0.8904 | -0.17pp | 679  | 102.4% |
| 30:60       | 0.8732 | -1.89pp | 608  | 102.6% |
| 60:90       | 0.8781 | -1.40pp | 578  | 102.1% |
| 90:120      | 0.8826 | -0.95pp | 559  | 102.3% |
| 120:150     | 0.8860 | -0.61pp | 544  | 103.1% |
| 150:180     | 0.8897 | -0.24pp | 533  | 105.1% |
| 180:210     | 0.8888 | -0.33pp | 457  | 102.1% |
| all-PQ      | 0.6917 | -20.0pp | 3992 | 100.2% |

Findings:
- **Refutes the design's timing claim from the opposite side.** The design says exactness only
  matters LATE (early = greedy descent, approx ranks direction well enough). Measured: with dose
  held equal the MOST damaging phase is **[30,60) — early descent** (-1.89pp), and the late
  convergence windows [150,210) are nearly free (-0.24 / -0.33pp). Approximate ranking hurts most
  exactly where the design assumed it was safe.
- **M11's monotone-looking curve was a dose artifact.** Deeper prefix gates looked progressively
  worse only because they PQ-scored more nodes; per unit of PQ the damage is front-loaded, not
  back-loaded.
- **0:30 is anomalously gentle (-0.17pp)** — the switch at step 30 rescores a still-small,
  highly-overlapping beam, correcting most of what PQ touched before it can propagate. Damage
  needs the search to have MOVED ON before the correction lands, which is why 30:60 is the worst.
- **Shape = discovery, not confirmation.** The hurt phase (early) is where M10 shows results are
  FOUND (49% by step 30, +23% in [30,60)); PQ misranking there evicts true neighbors during their
  one chance to enter the beam. Recovery is monotone 60->180 as the search shifts from finding
  members to confirming an already-latched beam that the final rerank fixes.
- **Cross-checks the arc:** exactness matters where neighbors are discovered (early), the same
  early phase M10 flagged; the free-to-approximate phase is the expensive straggler-confirmation
  tail. Also: total_dists stays 102-105% for every real window (vs M11 prefix's up to 152%) — a
  narrow window reopens little because little ran on PQ before the switch.

Consequence: reinforces M11 — no step/phase rule works, because the phase that needs exactness is
the SAME phase (early discovery) where the whole beam is still in flux. The trigger must be
per-node (Part 2 quality gate). Instrumentation: `FLATNAV_PQ_GATE` window in Index.h +
`tools/pq_stepgate.cpp` (WINDOWS=lo:hi,...) + `scripts/pq_stepgate.sh`; raw
`~/pq_window_w30/sweep.log` on clnode222.
