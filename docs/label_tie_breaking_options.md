# Label-Based Tie Breaking for Equal Distances

In this codebase, `node_id` and `label_id` are the same, so using the lower label as a tie-break is effectively the same as using the lower node id.

## What changes if you do this internally

If you update the comparator and candidate handling so that equal distances prefer the lower label, the search becomes more deterministic on ties. That is useful when two runs produce the same distance but different labels, because the lower label would win instead of whichever item happened to arrive first.

The consequence is that this is not just a reporting change. It can alter the ANN search path itself. In a graph search, the next node expanded can change the future frontier, so a tie-break rule can slightly change recall, latency, and the exact top-k set even when the tied distances are identical.

## Options

### 1. Leave the search internals unchanged

This preserves the current ANN behavior. Equal-distance cases remain dependent on heap order and insertion order. It is the least risky choice for search quality, but it is the least deterministic.

### 2. Stabilize only the returned ordering

This keeps the search path unchanged and sorts equal-distance outputs by lower label at the end. This improves reproducibility of the output without changing how the graph is explored.

### 3. Apply label-based tie breaking inside search

This makes the exploration itself prefer lower labels on exact ties. It gives the most deterministic internal behavior, but it can change the ANN path and therefore slightly affect recall or latency.

## Impact on ANN search in general

In approximate nearest neighbor search, exact ties are usually a small corner case, but they can still matter because graph search is path-dependent. A tie-break change can cascade: one different candidate can lead to a different expansion order, which can produce a different final top-k result even when all compared distances are the same.

That means label-based tie-breaking is safe if your goal is consistency, but it is not behavior-neutral.

## Recommendation

If your main goal is to compare runs and understand whether serial and threaded search are functionally equivalent, I recommend stabilizing only the final output ordering first. That gives you deterministic reporting while keeping the ANN search behavior unchanged.

If you later want full internal determinism for exact ties, then use label-based tie-breaking in the comparator and candidate handling as a second step, with the expectation that search results may shift slightly on tie-heavy queries.