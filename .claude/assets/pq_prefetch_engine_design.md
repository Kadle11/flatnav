# Design: PQ-tiered graph vector search

**Status:** draft v0.2. Based on `expt-logs/prefetch_study_log.md` (M1–M10).

## The idea in one line

Explore the graph on **compressed vectors kept in local memory**, and fetch a
**full-precision vector from remote memory only for the few nodes where the compression
error might change the answer** — hiding those fetches behind the time the node already
sits in the search queue.

## Memory layout

```
LOCAL (fast)                         REMOTE / CXL (slow, big)
  - the whole graph (links)            - full-precision vectors
  - PQ codes for every vector          - cold graph links (only if local runs out)
  - PQ codebooks
```

Local holds everything the search *walks over*. Remote holds the big exact vectors we
touch rarely. PQ codes are ~8× smaller than raw vectors, so the local tier fits a much
bigger index.

## Three parts

### 1. Explore on compressed vectors (cuts remote traffic to ~zero)

Beam search ranks nodes by their **PQ (approximate) distance**, computed from the local
codes. No remote reads during exploration. This is the main win: today's search makes
~3,983 remote vector fetches per query (M8); walking on codes makes that number ~0.

### 2. Fetch exact vectors only when it matters (the quality gate)

A node earns a full-precision fetch only when **both** are true:
- **It could change the top-K** — its approximate distance is close to the current
  K-th best. (Far-off nodes never get fetched.)
- **Its compression is untrustworthy** — the PQ error for this node is large enough that
  the exact distance could flip the ranking.

Everything else stays on the approximate distance. This keeps remote fetches down to the
handful the answer actually depends on.

**When is exactness needed? Late, not early (needs verification).** The early steps are
greedy descent — the search just needs to move toward the query's region, and the
approximate distance ranks *direction* well enough. Exact distances only matter for the fine
ranking of the final top-K, which is settled **late**: M10 shows the last top-100 member
lands around step ~183 and the convergence tail is the expensive part. So the exact fetches
are naturally back-loaded — the same place the lead time is largest (Part 3).
*To verify:* run traversal on PQ only for the first N steps and exact after, and check recall
vs N — recall should stay flat until N is fairly large.

> **REFUTED — M11/M12 (2026-07-24, see `expt-logs/prefetch_study_log.md`).** The opposite is
> true. With the PQ dose held equal across a sliding window (M12, SIFT100M ef=200), the phase
> that most needs exactness is **early descent [30,60)** (−1.89pp recall), while the late
> convergence windows [150,210) are nearly free (−0.24pp). Approximate ranking hurts most
> exactly where this paragraph assumed it was safe: early is where true neighbors are
> *discovered* (M10: 49% of the top-100 by step 30), and a mis-rank there evicts a neighbor
> during its one chance to enter the beam — an unrecoverable leak, not a fixable fine-ranking
> error. Prefix gating (M11) looked monotonically bad with depth only because deeper N also
> means more nodes PQ-scored (a dose artifact). **Consequence:** the exact-fetch trigger cannot
> be a step/phase rule of any shape; it must be **per-node** — the quality gate of Part 2.

### 3. Prefetch by residency, split by phase

When a node is marked for an exact fetch, we start the remote read **immediately** and use
the result later — by the time we need it, it's already local.

**Is Residency the PF signal in late (and early) phases?** A node's *residency*
is how long it has already sat in the queue. A node that has survived many steps
without being pruned is close to the query and likely to be explored — so it's 
worth fetching now. Short-lived nodes churn back out and aren't worth the fetch.
*To verify:* measure P(node is eventually explored | it has survived R steps)
and see if it rises with R. If so, the residency threshold R\* that triggers a 
prefetch is the prefetch policy.

**When and how much to prefetch — two forces pulling opposite ways.** Going deeper trades one
against the other:
- **Volume we can prefetch falls** — fan-out drops from ~25 to ~18 new nodes per step (fewer
  fresh vectors to fetch each step).
- **Lead time rises** — the wait before a queued node is explored grows from ~1 to ~81 steps.

| search depth (step) | explorations per step (volume) | lead time (queue wait) |
|---|---|---|
| ~5   | 24 | ~1 step   |
| 20   | 22 | ~4 steps  |
| 40   | 21 | ~12 steps |
| 80   | 19 | ~33 steps |
| 160  | 18 | ~81 steps |

(M8, `prefetch_stepwise_fanout_default.csv`. ~3,983 fetches/query total; query-average lead
~46 steps.)

So there is no free "prefetch everything late": the deep steps that give the lead to stage a
fetch also offer *less* to stage, while the early steps with the most to fetch have no lead to
hide it behind. **Navigating it:** since volume only *decreases* with depth, the best operating
point is the **shallowest depth where lead first covers the remote-fetch latency** — start
earlier and you can't hide the fetch; wait later and you forfeit the higher volume of the steps
in between. That crossover depth is set by the **mechanism**: a cache-line hint needs only ~1–2
steps of lead (M7's +4% at 2.3-step lead), so it can start almost immediately and ride the
high-volume early stream; a page-migration or staging copy needs tens of steps (M8), so it only
pays in the deep phase — which is also where Part 2 says the exact vectors are actually needed.

**The threshold is phase-dependent (open).** Early descent churns (short residencies, low
predictability) while late convergence is stable (long residencies, ~75% of expansions come
from long-resident candidates — M2). So the residency threshold R\* that says "this node is
likely enough to explore — prefetch it" should differ by phase. The concrete study: within
each phase, measure **P(node is eventually explored | it has survived R steps)** — is it
monotonic in R, and is there a threshold R\* where it's high enough to prefetch with little
waste? That threshold *is* the prefetch policy.

## Per-query flow

```
1. Beam search on PQ codes (local, no remote reads).
2. For each strong candidate, if the quality gate fires:
      start an async remote fetch of its exact vector.
3. When we need the exact distance, the vector is already local — use it, reorder.
4. Return top-K by exact distance.
```

## Why this fits what we already measured

| Design choice | Evidence |
|---|---|
| Stop touching remote during traversal | ~4,000 fetches/query today, spread across the whole search (M8) |
| Fewer fetches beats faster fetches | latency is already easy to hide; the real cost is fetch *count* (M8, M10) |
| Exact fetches are few and late | 90% of the top-100 is found by step ~90 (M10) |
| Depth trades volume for lead | prefetchable volume falls (~25→18/step) while lead grows (~1→81 steps) — navigate the crossover (M8) |
| Don't try to *predict* where the search goes | that was tested and failed (M9) |

## Open questions

1. **The quality gate** — is it a fixed margin, a fixed rerank budget (top-R), or tuned to
   a recall target? This sets both recall and fetch count. (Biggest open item.)
2. **PQ settings** — more subquantizers = better accuracy, fewer exact fetches, but bigger
   codes. What's the sweet spot?
3. **Is the win speed or capacity?** Simply stopping the search early already saves ~56% of
   the work (M10). We should measure this design *on top of* early-stop, and be honest if the
   real benefit is fitting a bigger index (capacity) rather than raw speed.
4. **Recall leak** — a true neighbor can get dropped on approximate distance before it's
   ever fetched and corrected. Needs measuring.
5. **Residency threshold per phase (the prefetch policy).** Measure P(node is eventually
   explored | it has survived R steps), split by phase, and find the threshold R\* that
   triggers a prefetch with little waste. Existing data brackets it (coverage rises 8%→81%
   with depth, M8) but doesn't yet give P(explored | age) directly.

## Design directions (after M11/M12)

M11/M12 killed the phase-switch traversal (exact-fetch cannot be a step rule), but they also
reframe how PQ and the prefetch (PF) mechanism should combine. Three ideas:

1. **Speculate with PQ, verify with PF.** Keep the two mechanisms in distinct roles: PQ ranks
   *direction* from local codes (no remote read) and decides *what* is worth an exact vector;
   PF then hides the latency of fetching those exact vectors. PQ = the speculation/selection
   layer, PF = the verification/latency-hiding layer. This is the quality gate (open Q1)
   expressed as a division of labor rather than a step boundary.

2. **Dynamic and micro-PQ windows.** The fixed [lo,hi) window of M12 is a blunt instrument;
   make the PQ/exact boundary *adaptive* instead — driven by a per-query convergence signal
   (beam churn, K-th-best stability) rather than a fixed step count. "Micro-windows" = short
   PQ bursts that scout a few hops ahead and then hand back to exact verification, repeatedly,
   instead of one long PQ phase. The window becomes a control loop, not a constant.

3. **PQ buys the lead time PF lacks early.** M8 is the crux: early steps have ~1–8 steps of
   lead, too little for PF to hide a remote fetch, while late steps have 46–81 (plenty). So PF
   is trivially useful late and near-useless early — the opposite of where exactness matters
   (M12: early). Resolve it by letting a PQ traversal run **ahead** of the exact search as a
   scout: PQ needs no remote read, so it can leap forward in the zero-lead early phase and emit
   exact-fetch targets *with* lead time that the early phase otherwise doesn't have. PQ
   manufactures the lead; PF spends it. Late phase falls back to plain PF (lead is already
   there). Note the tension this resolves: M12 says early nodes must be *verified* exactly —
   PQ here is not the early answer, it is the scout that makes early exact prefetch possible.

## How we'll know it works

- **QPS at equal recall** vs. two baselines: (a) all vectors local + early-stop,
  (b) all vectors remote, no compression.
- **Remote fetches per query** — target far below M8's ~3,983.
- **Index size that fits locally** — the headline if the speed win is small.
- **Recall** held at target; measure how much is lost to dropped-before-fetch nodes.
