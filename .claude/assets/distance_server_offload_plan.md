# Plan: Multithreaded distance-server offload

Goal: Two-region NUMA split (graph on the local node, vectors on the
remote node) and move **query-time distance computation across a process boundary**
into a distance server pinned to the remote node where a process mimics CXL
near-memory-compute. 

We want to enable **high-bandwidth, multithreaded** offload, benchmarked
against our local-distance baseline with [hbw_search](../../tools/hbw_search.cpp).

**Branches to Combine**

1. **`hiBW-flatnav`** branch: the two-region NUMA split
([Index.h](../../include/flatnav/index/Index.h)). 
2. **`cboumalh/cboumalh-cxl-sim`** fork: files `include/flatnav/util/CxlSimulation.h`,
`tools/distance_server.cpp`, and the `#ifdef FLATNAV_CXL_OFFLOAD` hooks in its `Index.h`.

## Primer — run the current `hiBW-flatnav` search (baseline)

Today's local-distance search on clnode227; the offload is measured against this.

**hiBW-branch specifics** (why the commands differ from stock FlatNav):
- Build with **`-DFLATNAV_USE_NUMA -lnuma`** — enables this branch's two-region NUMA
  allocator ([NumaAllocation.h](../../include/flatnav/util/NumaAllocation.h)); without it,
  allocation falls back to a single `new char[]`.
- The `.bin` is this branch's **two-region format** (separate vectors + graph blobs) — not
  loadable by stock FlatNav or the fork.
- Runtime NUMA placement is per-region: env `FLATNAV_GRAPH_NUMA` / `FLATNAV_VECTORS_NUMA`
  feed `loadIndex(path, vec_node, graph_node)`. Optional CPU pinning: `FLATNAV_PIN_CPUS`.

```bash
cd ~/flatnav && git checkout hiBW-flatnav      # SIFT100M data is on /data

# Generate the index (once, ~20 min). Inputs already on /data:
# base vectors (.fvecs) + prebuilt M=32 HNSW base-layer graph (.mtx).
g++ -std=c++17 -O3 -march=native -fopenmp -DFLATNAV_USE_NUMA \
  -I include -I external/cereal/include tools/build_sift100m.cpp -o ~/build_sift100m -lnuma
~/build_sift100m /data/index/sift100m_base.fvecs \
  /data/index/sift100m_m32_hnsw_base_layer.mtx \
  /data/index/sift100m_flatnav.bin 32

# Build the search driver (NUMA-enabled)
g++ -std=c++17 -O3 -march=native -fopenmp -DFLATNAV_USE_NUMA \
  -I include -I external/cereal/include tools/hbw_search.cpp -o ~/hbw_search -lnuma

# Run: index, queries, gtruth, then K=100 threads=32 ef=200; pin workers to node 0
numactl --cpunodebind=0 ~/hbw_search \
  /data/index/sift100m_flatnav.bin \
  /data/queries/sift100m_200k_extra_query.fvecs \
  /data/queries/sift100m_200k_extra_query.gtruth.ivecs \
  100 32 200
```

Prints one row per ef: `ef  time_s  QPS  recall@K`. To place the two regions on
different nodes, prefix env: `FLATNAV_GRAPH_NUMA=0 FLATNAV_VECTORS_NUMA=1`.

## Integrating Offload Impl w/ `hiBW-flatnav` branch

### Multi-threaded CXL-client Slots
- The server is multi-slot, can serve concurrent requests: `distance_server.cpp` 
  creates `thread_count` slots and runs one worker thread per slot, all pinned
  to the remote node. 
- The client/index side cannot drive them. The fork's `Index` holds one
  `_cxl_client` bound to one slot; `computeDistanceCxl(node_id)` keys off a
  `thread_local` query id and writes to a single slot. 
- The 32-thread hiBW throughput run would have every thread collide on slot 0.

### High-bandwidth Search Harness
`hbw_search` is the multithreaded driver, and it stays **almost unchanged** — the offload
lives behind `Index::search()`, gated by `FLATNAV_CXL_OFFLOAD`. The same driver and search
loop measure both: built with the flag off = local-distance baseline, on = offload.

- **Threads → slots.** `hbw_search`'s `NumaThreadPool` enables concurrency. Each pool worker 
  carries a `thread_local` worker id; the Index's offload path maps worker id →
  `CxlClient` on `slot = worker id` (connected lazily on first search). 
- **Host pinning.** `hbw_search` already pins the pool to a node. In offload mode pin it
  to the local node's cores. The server pins itself to the remote node (separate
  process).
- **Server lifecycle is external.** Launch `distance_server` (loads vectors, remote node)
  first; the Index connects to its shm region by name when `hbw_search` loads the index.
  So the run is: start server → run `hbw_search`.
- Index loaded via `loadIndex(path, vec_node, graph_node)`; in offload mode the
  host's vectors sit idle at query time (server owns them).

## Target architecture

```
Host process (local node)                 Distance server (remote node)
  our two-region Index                       owns vectors (remote node)
  graph in _graph_memory (local)             S slots, 1 worker thread / slot
  NumaThreadPool: N search threads   <shm>   each worker computes L2 for its slot
    thread t  -> CxlClient -> slot t
    beam step: send neighbor ids,
    busy-poll for distances
```

- One `CxlClient` **per search thread**, each bound to a distinct slot; `S >= N`.
- Construction stays local (distances via `getNodeData`, as the fork already does in
  `selectNeighbors`/`connectNeighbors`). Only the query path offloads. Build the index
  normally; the server loads the vectors separately.

## Performance levers for high-bandwidth offload

1. **Per-thread slots** — map `NumaThreadPool` worker id → slot; zero slot contention.
2. **Batch distances per beam step.** Granularity is per expanded
   (popped) node: each beam step reads one node's `≤ M` links and
   computes a distance per neighbor
   ([Index.h:1085-1108](../../include/flatnav/index/Index.h#L1085-L1108)). The fork sends
   one id per request (`num_node_ids = 1`); instead send that node's unvisited
   neighbors as one request.

   Restructure the inner loop into three passes (the batch at most the `M` links —
   visited neighbors are skipped today at
   [1098](../../include/flatnav/index/Index.h#L1098)):
   1. **gather** — walk the `M` links, drop `visited_set->isVisited(...)`, mark unvisited,
      collect ids into a `≤ M` array;
   2. **one `computeDistances`** — send the array, get the distances back (`CxlRequest`
      already carries `num_node_ids`);
   3. **scatter** — push each `(id, dist)` into the `candidates`/`neighbors` PQs.

   Batch size `≤ M` (32), shrinking as the search progresses and more neighbors are
   already visited. Entry-node and `initializeSearch` distances stay single (`n = 1`).
3. **Busy-poll, one thread per core.** Both sides spin. Host threads spin
   for the response, server workers spin on their slot (no yield/sleep). Pin **one
   thread per core**: N host search threads on the local-node cores, S server workers on
   the remote-node cores.
4. **Query caching** — `cacheQuery` once per query so only node ids cross the boundary,
   not the query vector each call. Fork already does this; keep it.

## Steps

Build correctness single-threaded first, then add concurrency, then performance. Each
step has a check.

**Phase 1 — Import + build (no behavior change)**

1. Copy `CxlSimulation.h`, `distance_server.cpp`, `OffloadMetrics.h` from the fork into
   the `hiBW-flatnav` branch. → *check:* files compile standalone.
2. Build wiring: cmake `FLATNAV_CXL_OFFLOAD` option + a `distance_server` target linking
   `-lrt -lnuma`. → *check:* configures; builds both with the flag off (baseline) and on.

**Phase 2 — Single-thread correctness (one client, slot 0)**

3. Add offload state to the `hiBW-flatnav` [Index.h](../../include/flatnav/index/Index.h)
   behind the ifdef: one `CxlClient`, `connect`/disconnect in ctor/dtor, `computeDistanceCxl`,
   `cacheQuery`/`evictQuery` around `search`. → *check:* builds with flag on.
4. Swap only the query-path distance calls to `computeDistanceCxl`
   ([936](../../include/flatnav/index/Index.h#L936),
   [1108](../../include/flatnav/index/Index.h#L1108), entry-point init); construction
   stays local. Start the server (vectors loaded from `sift100m_base.fvecs`, pinned to
   the remote node). → **check (correctness gate):** 1-thread search recall@10 == baseline
   on the same index.

**Phase 3 — Multithread**

5. Per-thread `CxlClient`, slot = `NumaThreadPool` worker id; run server with `S =
   remote cores`. → *check:* N-thread search, recall unchanged, no races.
6. Pin: host search threads on local-node cores, server workers on remote-node cores,
   both busy-polling. → *check:* pinning confirmed; QPS recorded.

**Phase 4 — Performance**

7. Batch the up-to-`M` neighbor distances per beam step into one request (lever #2). →
   *check:* recall unchanged, QPS up vs. step 6.

**Phase 5 — Benchmark**

8. Run baseline vs. offload with `hbw_search` — no search-loop changes (offload is behind
   `Index::search()`); just launch the server, then point `hbw_search` at it (shm name).
   → *check:* QPS @ iso-recall@100 on SIFT100M, thread sweep, batch on/off.

## Benchmark plan

- Data: SIFT100M on `/data` (`sift100m_base.fvecs`, `sift100m_query.fvecs`, gtruth).
- Baseline: `hbw_search` local distance (current). Compare offload QPS at matched
  recall, sweeping thread count and the per-step batch on/off.
