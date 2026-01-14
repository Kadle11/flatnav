# Plan: Profile Hub vs Non-Hub Vector Search Performance

Instrument FlatNav to measure compute and memory bandwidth utilization separately for hub and non-hub node distance computations during vector search.

## Core Strategy: Separate Hub vs Non-Hub Profiling

The key challenge is isolating hub and non-hub computations for hardware-level profiling. We have three approaches:

### Approach A: Software Counters + Aggregate Hardware Metrics
**Mechanism:**
- Add C++ counters to track hub vs non-hub distance computation counts
- Run full search with Linux `perf` to get aggregate hardware metrics
- Use software counters to infer per-computation cost: `hub_cost = total_metric / hub_count`

**Pros:** Simple, no kernel access needed, works on any Linux system
**Cons:** Cannot directly measure hub-specific cache misses or memory bandwidth

### Approach B: Separate Search Passes (Recommended)
**Mechanism:**
1. **Pass 1: Hub-only search** - Modify graph to only traverse hub nodes, run queries, measure with `perf`
2. **Pass 2: Non-hub-only search** - Modify graph to exclude hubs, run queries, measure with `perf`
3. Compare hardware metrics directly between passes

**Implementation:**
- Add `Index::setSearchMode(HUB_ONLY | NONHUB_ONLY | NORMAL)` method
- In `processCandidateNode`, skip neighbors based on `_hub_nodes[neighbor_node_id]` and mode
- Run identical query workload in both modes with `perf stat -e` events

**Pros:** Direct measurement of hub vs non-hub hardware behavior, clean separation
**Cons:** Requires two search passes, may have cold cache effects

### Approach C: Per-Node Instrumentation with PAPI
**Mechanism:**
- Wrap each `_distance->distance()` call with PAPI counter start/stop
- Accumulate hardware counters separately for hub vs non-hub nodes
- Requires PAPI library and root/kernel access for hardware counters

**Pros:** Most accurate, measures exact hardware events per computation
**Cons:** High overhead (~10-100x slowdown), needs PAPI library and permissions

## Detailed Search Flow and Profiling Points

### Understanding the Search Algorithm

The FlatNav search uses a **beam search** algorithm (similar to HNSW) with the following steps per query:

```
Query → Entry Node Selection → Beam Search Loop → Return K-NN
         |                      |
         |                      └─→ While(candidates not empty):
         |                           1. Pop best candidate
         |                           2. Get neighbors of candidate
         |                           3. For each neighbor:
         |                              - Check if visited
         |                              - Compute distance (PROFILING POINT)
         |                              - Update candidates/neighbors queues
         |
         └─→ Distance computation (PROFILING POINT)
```

### Search Steps Breakdown with Profiling Instrumentation

#### **Step 1: Entry Node Initialization** (`beamSearch`, lines 720-750)
**What happens:**
1. Select entry node (either greedy or random initialization)
2. Prefetch entry node data (`_mm_prefetch` line 735)
3. **Compute distance** to entry node (line 738)
4. Initialize candidates and neighbors queues

**Profiling instrumentation:**
```cpp
// Line 738: Entry node distance computation
float dist = _distance->distance(query, getNodeData(entry_node), true);

// ADD AFTER LINE 750:
if (_collect_stats && is_search_stage) {
  _distance_computations.fetch_add(1);
  if (_hub_nodes[entry_node]) {
    _hub_distance_computations.fetch_add(1);
  } else {
    _nonhub_distance_computations.fetch_add(1);
  }
}
```

**Metrics captured:**
- 1 distance computation (hub or non-hub depending on entry node)
- Memory access: 1 vector load (entry node data)
- Cache: Cold cache for first query, potentially warm for subsequent

---

#### **Step 2: Beam Search Loop** (`beamSearch`, lines 752-777)
**What happens per iteration:**
1. Pop best candidate from queue (line 753)
2. Check termination condition (lines 755-758)
3. Prefetch next candidate data (lines 765-768)
4. **Process all neighbors** of current candidate → calls `processCandidateNode`

**Loop characteristics:**
- Iterates ~ef_search times (controlled by buffer_size parameter)
- Each iteration processes M neighbors (max_edges_per_node)
- Total distance computations per query: ~ef_search × M

**Profiling instrumentation:**
- No direct instrumentation here (loop control only)
- Actual profiling happens in `processCandidateNode` call

---

#### **Step 3: Process Candidate Node** (`processCandidateNode`, lines 795-847)
**What happens:**
1. Lock the candidate node (line 798)
2. Iterate through M neighbors (line 802)
3. **For each neighbor** (lines 802-846):

   **a. Access tracking** (lines 803-807):
   ```cpp
   if (is_search_stage) {
     _node_access_counts[neighbor_node_id]++;
   }
   ```
   - Tracks how many times each node is visited across all queries
   - Hub nodes will have higher counts

   **b. Prefetch next neighbor** (lines 811-815):
   ```cpp
   _mm_prefetch(getNodeData(neighbor_node_links[i + 1]), _MM_HINT_T0);
   ```
   - Brings next neighbor's vector data into L1 cache
   - Reduces latency of subsequent distance computation

   **c. Check visited set** (lines 817-821):
   ```cpp
   bool neighbor_is_visited = visited_set->isVisited(neighbor_node_id);
   if (neighbor_is_visited) continue;
   ```
   - Bitmap check to avoid recomputing distances
   - O(1) operation

   **d. **DISTANCE COMPUTATION** (lines 823-825):**
   ```cpp
   float dist = _distance->distance(query, getNodeData(neighbor_node_id), true);
   ```
   **THIS IS THE CRITICAL PROFILING POINT**
   - Calls SIMD-optimized distance function (AVX-512/AVX2/SSE)
   - Memory access: Loads neighbor vector data (e.g., 512 bytes for 128-dim float32)
   - Compute: Dot product or L2 distance with SIMD instructions
   - Most expensive operation in search

   **e. Stats collection** (lines 827-829):
   ```cpp
   if (_collect_stats) {
     _distance_computations.fetch_add(1);
   }
   ```
   **NEED TO ADD HUB/NON-HUB TRACKING HERE:**
   ```cpp
   if (_collect_stats) {
     _distance_computations.fetch_add(1);
     if (_hub_nodes[neighbor_node_id]) {
       _hub_distance_computations.fetch_add(1);
     } else {
       _nonhub_distance_computations.fetch_add(1);
     }
   }
   ```

   **f. Update queues** (lines 831-845):
   ```cpp
   if (neighbors.size() < buffer_size || dist < max_dist) {
     candidates.emplace(-dist, neighbor_node_id);
     neighbors.emplace(dist, neighbor_node_id);
     // ...
   }
   ```
   - Priority queue operations (log N complexity)
   - Minimal compared to distance computation cost

---

### Per-Query Resource Breakdown

For a typical query with `ef_search=200`, `M=32`:

| Search Phase | Operations | Hub Nodes Visited | Non-Hub Nodes Visited |
|--------------|------------|-------------------|----------------------|
| Entry node init | 1 distance comp | 0-1 | 0-1 |
| Beam search loop (~200 iters) | | | |
| └─ Process candidates | ~200 × 32 = 6400 neighbor checks | ~N_hub | ~N_nonhub |
| └─ Distance computations | ~2000-4000 (after visited filter) | Variable | Variable |

**Key insight:** The number of hub vs non-hub distance computations depends on graph topology and hub placement.

---

### How Each Search Pass is Profiled

#### **Pass 1: Normal Search (Baseline)**
```python
index.set_search_mode(NORMAL)
index.reset_stats()

# Run queries
for query in queries:
    results = index.search(query, k=100, ef_search=200)

# Get aggregate stats
total_dist_comps = index.get_distance_computations()
hub_dist_comps = index.get_hub_distance_computations()
nonhub_dist_comps = index.get_nonhub_distance_computations()
```

**perf measurement (subprocess):**
```bash
perf stat -e cycles,instructions,cache-misses,L1-dcache-load-misses \
  python search_script.py
```

**What's measured:**
- Total hardware events across all queries
- Software counters track hub vs non-hub breakdown
- **Cannot directly attribute hardware events to hub vs non-hub**

---

#### **Pass 2: Hub-Only Search**
```python
index.set_search_mode(HUB_ONLY)
index.reset_stats()

# Run identical queries - but only traverse hub nodes
for query in queries:
    results = index.search(query, k=100, ef_search=200)
```

**Modified behavior in `processCandidateNode`:**
```cpp
for (uint32_t i = 0; i < _M; i++) {
  node_id_t neighbor_node_id = neighbor_node_links[i];
  
  // SKIP NON-HUB NEIGHBORS
  if (_search_mode == HUB_ONLY && !_hub_nodes[neighbor_node_id]) {
    continue;  // Don't visit, don't compute distance
  }
  
  // Rest of processing (only for hub neighbors)
}
```

**perf measurement:**
```bash
perf stat -e cycles,instructions,cache-misses,L1-dcache-load-misses \
  python search_script_hub_only.py
```

**What's measured:**
- Hardware events ONLY for operations touching hub nodes
- All distance computations are to hub nodes
- Memory accesses are only for hub node data
- **Direct measurement of hub node compute/memory cost**

---

#### **Pass 3: Non-Hub-Only Search**
```python
index.set_search_mode(NONHUB_ONLY)
index.reset_stats()

# Run identical queries - but only traverse non-hub nodes
for query in queries:
    results = index.search(query, k=100, ef_search=200)
```

**Modified behavior:**
```cpp
// SKIP HUB NEIGHBORS
if (_search_mode == NONHUB_ONLY && _hub_nodes[neighbor_node_id]) {
  continue;
}
```

**perf measurement:**
```bash
perf stat -e cycles,instructions,cache-misses,L1-dcache-load-misses \
  python search_script_nonhub_only.py
```

**What's measured:**
- Hardware events ONLY for non-hub operations
- **Direct measurement of non-hub node compute/memory cost**

---

### Metrics Interpretation

After running all 3 passes:

```python
# Hub-only metrics
hub_cycles = perf_results['hub_only']['cycles']
hub_distance_comps = results['hub_only']['distance_computations']
hub_cache_misses = perf_results['hub_only']['cache-misses']

# Non-hub metrics
nonhub_cycles = perf_results['nonhub_only']['cycles']
nonhub_distance_comps = results['nonhub_only']['distance_computations']
nonhub_cache_misses = perf_results['nonhub_only']['cache-misses']

# Compute per-distance-computation cost
hub_cycles_per_comp = hub_cycles / hub_distance_comps
nonhub_cycles_per_comp = nonhub_cycles / nonhub_distance_comps

hub_cache_miss_rate = hub_cache_misses / hub_distance_comps
nonhub_cache_miss_rate = nonhub_cache_misses / nonhub_distance_comps

print(f"Hub nodes: {hub_cycles_per_comp} cycles/comp, {hub_cache_miss_rate} misses/comp")
print(f"Non-hub nodes: {nonhub_cycles_per_comp} cycles/comp, {nonhub_cache_miss_rate} misses/comp")
```

**Interpretation:**
- **Higher cycles/comp for hubs** → Hubs are more expensive to compute (memory bound?)
- **Higher cache miss rate for hubs** → Hub data not in cache (memory bandwidth bottleneck)
- **Lower IPC for hubs** → Stalls waiting for memory

---

### Critical Profiling Requirement

**Problem:** The current plan has a gap - distance computations to the **entry node** and inside the **beam search loop** happen at different places.

**Solution:** Instrument ALL distance computation sites:

1. **Entry node** (`beamSearch` line 738)
2. **Neighbor nodes** (`processCandidateNode` line 824)
3. **Initialization** (`initializeSearch` - multiple distance comps)

All three must check `_hub_nodes[node_id]` and increment appropriate counters.

## Recommended Implementation: Hybrid Approach (A + B)

Combine software counters with separate search passes for best accuracy/practicality balance.

## Steps

### 1. Add hub-specific software counters to `Index.h`
**Location:** Lines 88-108 region

**Code changes:**
```cpp
// Add member variables
std::atomic<uint64_t> _hub_distance_computations = 0;
std::atomic<uint64_t> _nonhub_distance_computations = 0;

// Add getters
inline uint64_t hubDistanceComputations() const { 
  return _hub_distance_computations.load(); 
}
inline uint64_t nonhubDistanceComputations() const { 
  return _nonhub_distance_computations.load(); 
}
```

**Modify `resetStats()` (line ~642):**
```cpp
void resetStats() {
  _distance_computations = 0;
  _hub_distance_computations = 0;
  _nonhub_distance_computations = 0;
  _metric_hops = 0;
}
```

**Expose in Python bindings** (`python-bindings/src/flatnav/bindings.cpp`, ~line 481):
```cpp
.def("get_hub_distance_computations", &IndexType::hubDistanceComputations)
.def("get_nonhub_distance_computations", &IndexType::nonhubDistanceComputations)
```

### 2. Instrument `processCandidateNode` to track hub vs non-hub
**Location:** `Index.h` lines ~824-831

**Modify existing stats collection:**
```cpp
if (_collect_stats) {
  _distance_computations.fetch_add(1);
  if (_hub_nodes[neighbor_node_id]) {
    _hub_distance_computations.fetch_add(1);
  } else {
    _nonhub_distance_computations.fetch_add(1);
  }
}
```

**Also instrument entry node** in `beamSearch` (line ~738):
```cpp
if (_collect_stats) {
  _distance_computations.fetch_add(1);
  if (_hub_nodes[entry_node]) {
    _hub_distance_computations.fetch_add(1);
  } else {
    _nonhub_distance_computations.fetch_add(1);
  }
}
```

### 3. Add search mode filtering to `Index.h`
**Location:** Add enum and member variable after line 108

**Code:**
```cpp
enum SearchMode { NORMAL, HUB_ONLY, NONHUB_ONLY };
SearchMode _search_mode = NORMAL;

void setSearchMode(SearchMode mode) { _search_mode = mode; }
SearchMode getSearchMode() const { return _search_mode; }
```

**Modify `processCandidateNode`** (line ~811) to filter by mode:
```cpp
for (uint32_t i = 0; i < _M; i++) {
  node_id_t neighbor_node_id = neighbor_node_links[i];
  
  // Filter based on search mode
  if (_search_mode == HUB_ONLY && !_hub_nodes[neighbor_node_id]) continue;
  if (_search_mode == NONHUB_ONLY && _hub_nodes[neighbor_node_id]) continue;
  
  // ... rest of processing
}
```

**Expose in Python bindings:**
```cpp
.def("set_search_mode", &IndexType::setSearchMode)
```

### 4. Create hardware profiling script `experiments/profile_hub_compute.py`
**Purpose:** Run searches with Linux `perf` to measure hardware events

**Key functionality:**
```python
import subprocess
import json
import flatnav.index

def run_with_perf(index, queries, mode_name, search_mode):
    """Run search with perf stat and capture hardware metrics"""
    index.set_search_mode(search_mode)
    index.reset_stats()
    
    # perf events to measure
    events = [
        'cycles',
        'instructions',
        'cache-references',
        'cache-misses',
        'L1-dcache-loads',
        'L1-dcache-load-misses',
        'LLC-loads',
        'LLC-load-misses',
        'mem_load_retired.l3_miss',  # Memory bandwidth proxy
        'mem_inst_retired.all_loads',
    ]
    
    perf_cmd = [
        'perf', 'stat', '-e', ','.join(events),
        '-x', ',',  # CSV output
        '--', 'python3', '-c', f'''
import flatnav.index
import numpy as np
# Load index and run search in subprocess
# ... search code ...
        '''
    ]
    
    result = subprocess.run(perf_cmd, capture_output=True, text=True)
    
    # Parse perf output
    metrics = parse_perf_output(result.stderr)
    
    # Get software counters
    metrics['hub_distance_computations'] = index.get_hub_distance_computations()
    metrics['nonhub_distance_computations'] = index.get_nonhub_distance_computations()
    
    return metrics

# Run three modes
results = {}
results['normal'] = run_with_perf(index, queries, 'normal', NORMAL)
results['hub_only'] = run_with_perf(index, queries, 'hub_only', HUB_ONLY)
results['nonhub_only'] = run_with_perf(index, queries, 'nonhub_only', NONHUB_ONLY)

# Compare
print(f"Hub IPC: {results['hub_only']['instructions'] / results['hub_only']['cycles']}")
print(f"Non-hub IPC: {results['nonhub_only']['instructions'] / results['nonhub_only']['cycles']}")
print(f"Hub cache miss rate: {results['hub_only']['cache-misses'] / results['hub_only']['cache-references']}")
```

### 5. PAPI Integration Decision

**Question: Is PAPI needed?**

**Assessment:**
- **Not required for initial profiling** - Linux `perf` provides sufficient hardware counters
- **Consider PAPI if:**
  - Need per-distance-computation granularity (not per-query)
  - Need exotic counters not in perf (e.g., specific uncore events)
  - Want programmatic counter control within C++ (not external wrapper)
  - Already have PAPI installed and configured

**Recommendation:** Start with Approach B (separate passes + perf). Add PAPI later only if:
1. Overhead from separate passes is too high (cold cache effects)
2. Need to measure individual distance computations, not aggregate query metrics

**If PAPI is needed later:**
- Add PAPI wrapper around `_distance->distance()` call in `processCandidateNode`
- Accumulate counters into `_hub_papi_events[]` and `_nonhub_papi_events[]` arrays
- Accept 10-100x slowdown for detailed profiling mode

## Expected Metrics Output

### Hardware Metrics (from perf)
| Metric | Hub-Only | Non-Hub-Only | Normal |
|--------|----------|--------------|--------|
| Cycles | X | Y | Z |
| Instructions | X | Y | Z |
| IPC | X/cycles | Y/cycles | Z/cycles |
| L1 cache misses | X | Y | Z |
| LLC cache misses | X | Y | Z |
| Cache miss rate | X/refs | Y/refs | Z/refs |
| Memory loads | X | Y | Z |

### Software Metrics (from counters)
- Hub distance computations: N_hub
- Non-hub distance computations: N_nonhub  
- Avg cycles per hub computation: hub_cycles / N_hub
- Avg cycles per non-hub computation: nonhub_cycles / N_nonhub
- Avg cache misses per hub computation: hub_misses / N_hub

### Analysis Questions Answered
1. Do hub nodes have higher cache miss rates? (memory bandwidth bound)
2. Do hub nodes have lower IPC? (compute bound)
3. What is the per-computation cost difference: hub vs non-hub?
4. Are hubs memory-bound or compute-bound?

## Further Considerations

1. **Cold cache effects in separate passes:** First pass will have cold caches, second will have warm caches. Solution: Run multiple iterations and average, or clear caches between passes (`echo 3 > /proc/sys/vm/drop_caches` with sudo)

2. **Multi-threading:** Perf measurements work with multi-threaded search, but attribution becomes complex. Recommend single-threaded mode (`index.set_num_threads(1)`) for initial profiling

3. **Entry node bias:** Entry node selection may favor certain nodes. Track entry node stats separately to understand if hubs are disproportionately used as entry points

## Key Code Locations Reference

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Hub marking | `Index.h` | 268-273 | Set hub flags |
| Distance computation | `Index.h` | 824-831 | Main profiling point |
| Node access tracking | `Index.h` | 803-806 | Access counting |
| SIMD dispatch | `L2DistanceDispatcher.h` | 42-84 | Compute selection |
| Memory layout | `Index.h` | 670-676 | Data access |
| Search loop | `Index.h` | 720-780 | Traversal pattern |
| Python bindings | `bindings.cpp` | 278-281 | Stats retrieval |
| Benchmarking | `run_benchmark.py` | 66-82 | Timing framework |

## Existing Infrastructure

### Built-in Statistics
- `_collect_stats` flag enables metric collection
- `_distance_computations` atomic counter tracks total distance operations
- `_node_access_counts` map tracks per-node visit frequency
- `_hub_nodes` boolean array marks hub nodes

### Python Benchmarking
- `run_benchmark.py` measures latency percentiles (p50, p90, p95, p99)
- Computes QPS (queries per second)
- Uses `time.perf_counter()` for timing

### Distance Computation
SIMD implementations in `SquaredL2SimdExtensions.h`:
- AVX-512: 16-float operations
- AVX2: 8-float operations
- SSE: 4-float operations
- Scalar fallback

### Node Memory Layout
```cpp
// Node: [data] [M links] [data label]
// Chosen for cache efficiency
```
