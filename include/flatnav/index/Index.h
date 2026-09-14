#pragma once

#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/util/Macros.h>
#include <flatnav/util/Multithreading.h>
#include <flatnav/util/Reordering.h>
#include <flatnav/util/VisitedSetPool.h>
#include <flatnav/util/Datatype.h>
#include <flatnav/util/NumaAllocation.h>
#include <flatnav/util/PrefetchStaging.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <optional>
#include <x86intrin.h>  // __rdtsc for FLATNAV_PROFILE_PHASE timing

using flatnav::distances::DistanceInterface;
using flatnav::util::VisitedSet;
using flatnav::util::VisitedSetPool;
using flatnav::util::DataType;
using flatnav::util::kNoNumaNode;

namespace flatnav {

#ifdef FLATNAV_PROFILE_PQ
// Candidate-PQ residency instrumentation (tier-prefetch study). A node's prefetch
// lead time = its PQ residency = (pop_step - discovery_step). Per-thread maps track
// discovery step + parent residency; global atomic histograms aggregate across queries.
inline constexpr int kPQHistCap = 4096;
inline std::atomic<uint64_t> g_pq_residency_hist[kPQHistCap];     // residency of every expanded node
inline std::atomic<uint64_t> g_leapfrog_parent_hist[kPQHistCap];  // parent residency of residency==1 nodes
// Per-search-step (pop/hop index) buckets, for the early-vs-late predictability split.
inline constexpr int kPQStepCap = 1024;
inline std::atomic<uint64_t> g_step_count[kPQStepCap];   // # expansions at this step
inline std::atomic<uint64_t> g_step_leap[kPQStepCap];    // of those, residency==1 (leapfroggers)
inline std::atomic<uint64_t> g_step_sumres[kPQStepCap];  // sum of residency (for mean)
inline std::atomic<uint64_t> g_step_fanout[kPQStepCap];  // # NEW (unvisited) neighbors fetched/dist-computed at this step
inline thread_local std::unordered_map<uint32_t, uint32_t> tl_disc;        // node -> discovery step
inline thread_local std::unordered_map<uint32_t, uint32_t> tl_parent_res;  // node -> parent's residency
inline thread_local uint32_t tl_step;          // pops done so far this query
inline thread_local uint32_t tl_cur_residency; // residency of the node currently being expanded
// "top-K candidate prefetch" policy: each step, just after the current node is popped (and BEFORE
// its neighbors are discovered), we'd prefetch the K closest pending candidates. Because the snapshot
// is taken pre-expansion, this step's leapfroggers (not yet born) are excluded from every K. A node
// is "prefetched" once, at the first step it enters the top-K. success = eventually popped; waste =
// never popped. Buckets keyed by first-prefetch step; lead = pop_step - first_prefetch_step.
inline constexpr int kPFnumK = 4;
inline constexpr int kPFK[kPFnumK] = {1, 5, 10, 50};
inline std::atomic<uint64_t> g_pf_prefetched[kPFnumK][kPQStepCap];
inline std::atomic<uint64_t> g_pf_success[kPFnumK][kPQStepCap];
inline std::atomic<uint64_t> g_pf_leadsum[kPFnumK];
struct PFEntry {
  uint16_t fe[kPFnumK] = {0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF}; // first step entered top-K (0xFFFF=never)
  uint16_t pop_step = 0;
  bool popped = false;
};
inline thread_local std::unordered_map<uint32_t, PFEntry> tl_pf;
// Pre-expansion K=1 NEXT-STEP precision: at each step we prefetch the 2nd-min (next candidate);
// did exactly that node get popped at the immediately next step? (= used with 1-step lead, the
// strictest "is the next vector known" test). Bucketed by the prefetch step.
inline std::atomic<uint64_t> g_pf1_next_total[kPQStepCap];
inline std::atomic<uint64_t> g_pf1_next_hit[kPQStepCap];
inline thread_local uint32_t tl_pf1_node;   // 2nd-min prefetched last step
inline thread_local uint32_t tl_pf1_step;   // the step it was prefetched at
inline thread_local bool tl_pf1_valid;

#ifdef FLATNAV_PROFILE_TOPK
// tools/topk_found.cpp: at which step was each final top-K result first DISCOVERED
// (distance-computed)? Reads tl_disc. g_topk_found = histogram of discovery steps over all
// (query x top-K) results; g_topk_complete = per-query step of the LAST top-K discovery
// (= "answer complete"); g_topk_rank1 = discovery step of the 1-NN. Needs FLATNAV_PROFILE_PQ.
inline int g_topk_K = 100;
inline std::atomic<uint64_t> g_topk_found[kPQStepCap];
inline std::atomic<uint64_t> g_topk_complete[kPQStepCap];
inline std::atomic<uint64_t> g_topk_rank1[kPQStepCap];
inline std::atomic<uint64_t> g_topk_complete_sum;  // sum over queries of the answer-complete step
inline std::atomic<uint64_t> g_topk_total_sum;     // sum over queries of total steps
inline std::atomic<uint64_t> g_topk_nq;            // #queries counted
#endif
#endif

#ifdef FLATNAV_PQ_GATE
// Step-windowed PQ traversal (tools/pq_stepgate.cpp). Expansions inside the window
// [_pq_gate_lo, _pq_gate_hi) score newly-discovered nodes with the ASYMMETRIC PQ distance (a
// lookup over the node's code -- no vector read); expansions outside it use exact distances.
// At the window's end the beam and the pending candidates are rescored exact, so the search
// resumes off a correct beam. Sliding the window isolates WHICH PHASE of the search actually
// needs exactness; the window [0, N) is the prefix gate that M11 measured. The per-query LUT
// (_pq_m x 256 floats: query subvector to each subquantizer centroid) is built by the caller
// and installed here before search(); null LUT = exact distances everywhere.
inline thread_local const float* tl_pq_lut = nullptr;
inline thread_local uint32_t tl_gate_step = 0;         // expansions done so far this query
// Per-query tallies, flushed into the globals once at the end of beamSearch. These count one
// per DISTANCE (~4k/query, the innermost loop), so they must not be atomics: 32 threads
// incrementing two adjacent globals would false-share the line through the whole run.
inline thread_local uint32_t tl_gate_pq = 0;
inline thread_local uint32_t tl_gate_exact = 0;
inline std::atomic<uint64_t> g_gate_pq_dists{0};       // distances served from PQ codes
inline std::atomic<uint64_t> g_gate_exact_dists{0};    // traversal distances that read a full vector
inline std::atomic<uint64_t> g_gate_rescore_dists{0};  // vector reads spent rescoring PQ-ranked queues

// specBeamSearch tallies, same per-query-then-flush discipline as the gate counters above.
//   checks     -- predictions tested. The first pick off an empty ring is correct by construction
//                 and is not one.
//   floor      -- checks whose true winner no width could have found, because the speculative
//                 lane's score kept it off the overlay of the step that discovered it. The
//                 ceiling on what widening the search can buy.
//   miss_rej / miss_tie / miss_order -- a miss broken down by where the predicted node ended up,
//                 filled in only under setSpecDiag(). Rejected: never admitted to the beam, so no
//                 ranking could have found it. Tie: admitted at the same distance as the node
//                 actually expanded, and lost on heap order alone. Order: admitted, at a
//                 different distance, and genuinely ranked wrong.
//   committed  -- exact vector reads performed, which equals the exact search's own count
//   wasted     -- vectors belonging to discarded steps: the bandwidth a speculative step costs
//                 when its prediction does not hold.
inline thread_local uint32_t tl_spec_checks = 0;
inline thread_local uint32_t tl_spec_hits = 0;
inline thread_local uint32_t tl_spec_misses = 0;
inline thread_local uint32_t tl_spec_miss_rejected = 0;
inline thread_local uint32_t tl_spec_miss_tie = 0;
inline thread_local uint32_t tl_spec_miss_order = 0;
inline thread_local uint32_t tl_spec_floor = 0;

// Which speculative step discovered a node, and whether its score got it onto that step's
// overlay. Together with the in-flight range recorded on each step, this answers whether a node
// was reachable by the speculative lane at the moment it made a given pick. Kept per query, under
// setSpecDiag() only.
struct SpecOrigin { uint32_t step; bool admitted; };
inline thread_local std::unordered_map<uint32_t, SpecOrigin> tl_spec_origin;

// Checks and misses bucketed by how many steps were actually in flight when the prediction was
// made, which a miss resets to zero. The configured depth is only an upper bound on this.
inline constexpr int kSpecDepthCap = 17;
inline thread_local uint32_t tl_spec_depth_checks[kSpecDepthCap] = {};
inline thread_local uint32_t tl_spec_depth_misses[kSpecDepthCap] = {};
inline std::atomic<uint64_t> g_spec_depth_checks[kSpecDepthCap];
inline std::atomic<uint64_t> g_spec_depth_misses[kSpecDepthCap];
inline thread_local uint32_t tl_spec_discarded = 0;
inline thread_local uint32_t tl_spec_stalls = 0;
inline thread_local uint32_t tl_spec_committed = 0;
inline thread_local uint32_t tl_spec_wasted = 0;
inline std::atomic<uint64_t> g_spec_checks{0};
inline std::atomic<uint64_t> g_spec_hits{0};
inline std::atomic<uint64_t> g_spec_misses{0};
inline std::atomic<uint64_t> g_spec_miss_rejected{0};
inline std::atomic<uint64_t> g_spec_miss_tie{0};
inline std::atomic<uint64_t> g_spec_miss_order{0};
inline std::atomic<uint64_t> g_spec_floor{0};
inline std::atomic<uint64_t> g_spec_discarded{0};
inline std::atomic<uint64_t> g_spec_stalls{0};
inline std::atomic<uint64_t> g_spec_committed{0};
inline std::atomic<uint64_t> g_spec_wasted{0};
#endif

#ifdef FLATNAV_SPEC_TRACE
// Speculation-divergence trace (tools/pq_specdiverge.cpp): starting from the exact search's own
// beam at the window's lower edge, does a PQ scout make the same decisions the exact search
// does INSIDE the window [lo,hi)? The PQ approximation begins at step lo -- the search is exact
// for steps [0,lo), so the exact baseline and the gated run share an identical prefix and
// diverge only within the window; the divergence measured over [lo,hi) is purely the effect of
// PQ speculation in that phase, from a correct starting beam. Two captures, both keyed by
// expansion step (== tl_gate_step, the beam-loop expansion index):
//   tl_spec_expand -- the ordered list of expanded node ids. Filled on the exact baseline AND
//     each gated run; the tool Jaccard-compares the two trajectories' node SETS over [lo,hi)
//     (scout drift; decision unit = expanded-node set).
//   tl_spec_disc   -- {step, node, exact, pq, thresh} per newly-discovered neighbor. Whichever
//     of exact/pq did NOT drive this expansion is computed alongside as a measurement (a vector
//     read inside the PQ window, a code lookup outside it), so the tool can score whether PQ
//     ranked the discovered set the way exact would (fetch-target agreement; counterfactual per
//     expansion). Filled inside the PQ window on a gated run AND on every expansion of an exact
//     run that has a LUT installed -- the latter is M14's per-vector error along the EXACT
//     trajectory, whose step axis is comparable to M8/M10/M12 and whose touched population is
//     not self-selected by PQ's own mistakes.
//     `node` attributes the error to a vector; `thresh` is the beam's K-th best at the moment of
//     the decision (+inf while the beam is still filling), so the tool can separate a harmless
//     error from one that flips this node's admission: (exact < thresh) != (pq < thresh).
// The caller installs the target vectors on its thread before each search and reads them after;
// a null pointer disables that capture.
struct SpecDisc { uint32_t step; uint32_t node; float exact; float pq; float thresh; };
inline thread_local std::vector<uint32_t>* tl_spec_expand = nullptr;
inline thread_local std::vector<SpecDisc>* tl_spec_disc = nullptr;
#endif

// Terminal-basin trajectory capture (tools/basin_map.cpp), gated by FLATNAV_PROFILE_TRAJ
// at the use sites. Declared unconditionally (thread_locals, zero cost when unused) so
// the tool can read them without pulling in the FLATNAV_PROFILE_PQ residency machinery.
inline thread_local std::vector<uint32_t> tl_traj;   // expanded node ids, in pop order
inline thread_local uint32_t tl_term_node;           // closest node found so far (internal id)
inline thread_local float tl_term_dist;              // its distance

// Per-search-depth timing (tools/phase_time.cpp): FIXED-WIDTH bins of kBinW=30 expansions
// (bin b = expansion steps [30b, 30b+30)). Accumulate rdtsc cycles per ABSOLUTE bin, so the
// number of populated bins is set by each query's search length (longer searches reach more
// bins). One rdtsc + a couple of atomics per expansion -> negligible vs an ~11 us step.
inline constexpr int kBinW = 30;
inline constexpr int kBins = 64;                     // covers searches up to 1920 expansions
inline std::atomic<uint64_t> g_bin_cycles[kBins];    // total rdtsc cycles attributed to bin b
inline std::atomic<uint64_t> g_bin_steps[kBins];     // # expansions in bin b
inline std::atomic<uint64_t> g_bin_queries[kBins];   // # queries that reached bin b
inline thread_local uint32_t tl_pstep;               // expansion index within the current query
inline thread_local uint64_t tl_bin_ts;              // rdtsc taken at the current bin's start boundary

// dist_t: A distance function implementing DistanceInterface.
// label_t: A fixed-width data type for the label (meta-data) of each point.
template <typename dist_t, typename label_t>
class Index {
  typedef std::pair<float, label_t> dist_label_t;
  // internal node numbering scheme. We might need to change this to uint64_t
  typedef uint32_t node_id_t;
  typedef std::pair<float, node_id_t> dist_node_t;

  // NOTE: by default this is a max-heap. We could make this a min-heap
  // by using std::greater, but we want to use the queue as both a max-heap and
  // min-heap depending on the context.

  struct CompareByFirst {
    constexpr bool operator()(dist_node_t const& a, dist_node_t const& b) const noexcept{
      return a.first < b.first;
    }
  };

  typedef std::priority_queue<dist_node_t, std::vector<dist_node_t>, CompareByFirst> PriorityQueue;

  // Read-only access to a PriorityQueue's underlying heap array (to cheaply read the top-K
  // pending candidates without copying/popping). Index 0 = root = closest; the rest is heap
  // order, so the first K entries approximate the K closest -- good enough for a prefetch
  // hint. Uses the standard protected-member-access idiom.
  static const std::vector<dist_node_t>& pqHeap(const PriorityQueue& pq) {
    struct Hack : private PriorityQueue {
      static const std::vector<dist_node_t>& get(const PriorityQueue& p) {
        return p.*&Hack::c;
      }
    };
    return Hack::get(pq);
  }

#ifdef FLATNAV_PQ_GATE
  // The r best entries of a binary max-heap's array, best first, WITHOUT popping -- so the
  // speculative lane can read V's pending candidates without disturbing them. Bounded best-first
  // over the implicit tree: the array root is the max, and a node's children (2i+1, 2i+2) are the
  // only entries that can succeed it, so a frontier heap of at most 2r indices yields them in
  // order. O(r log r) against O(n) for a full sort. `skip` filters ids already claimed by an
  // in-flight step; its subtree is still expanded, since a skipped parent can hide good children.
  template <typename Skip>
  static void heapTopR(const std::vector<dist_node_t>& heap, int r, const Skip& skip,
                       std::vector<dist_node_t>& out) {
    out.clear();
    if (heap.empty() || r <= 0) return;
    std::priority_queue<std::pair<float, uint32_t>> frontier;
    frontier.emplace(heap[0].first, 0u);
    while (!frontier.empty() && (int)out.size() < r) {
      const uint32_t i = frontier.top().second;
      frontier.pop();
      if (!skip(heap[i].second)) out.push_back(heap[i]);
      const uint32_t l = 2 * i + 1, rt = 2 * i + 2;
      if (l < heap.size()) frontier.emplace(heap[l].first, l);
      if (rt < heap.size()) frontier.emplace(heap[rt].first, rt);
    }
  }

  // One speculative step. `fresh` is the main pick's unvisited neighbours, ALREADY marked visited
  // (so a discard has to un-mark them); `overlay` is their PQ scores, negated to match the
  // candidate queue's convention, standing in for V.candidates entries until validation replaces
  // them with exact ones.
  struct SpecStep {
    node_id_t node = 0;
    std::vector<node_id_t> fresh;
    std::vector<dist_node_t> overlay;
    // Step ids are monotonic within a query, never reused after a discard. `inflight_lo` is the
    // oldest id in flight when this step was chosen, which bounds the steps whose neighbours had
    // reached the overlay but not yet the candidate queue at that moment.
    uint32_t id = 0;
    uint32_t inflight_lo = 0;
    // Steps in flight when this one was chosen. This, not _spec_depth, is how far ahead the
    // speculation actually ran for this step: a miss empties the ring, so the steps chosen while
    // it refills see far fewer unvalidated neighbours than the configured depth allows.
    uint32_t depth_at_push = 0;
  };
#endif

  // NUMA-aware storage. The data and the graph are kept in two separate,
  // contiguous (structure-of-arrays) allocations so each can be bound to its
  // own NUMA node.
  //   Vectors block: one [data] entry per node,         stride _data_size_bytes.
  //   Graph block:   one [M links][label] entry per node, stride
  //                  _graph_node_size_bytes.
  char* _vectors_memory = nullptr;
  char* _graph_memory = nullptr;
  int _vectors_numa_node = kNoNumaNode;
  int _graph_numa_node = kNoNumaNode;

  size_t _M;
  // size of one data point (does not support variable-size data, strings)
  size_t _data_size_bytes;
  // Logical size of a node: [data] + [M links] + [label]. The data lives in
  // _vectors_memory and the links+label live in _graph_memory, so this is kept
  // only for reporting/serialization metadata. It satisfies
  // _node_size_bytes == _data_size_bytes + _graph_node_size_bytes.
  size_t _node_size_bytes;
  // Size of one entry in _graph_memory: [M links][label].
  size_t _graph_node_size_bytes;
  size_t _max_node_count;  // Determines size of internal pre-allocated memory
  size_t _cur_num_nodes;
  std::unique_ptr<DistanceInterface<dist_t>> _distance;
  std::mutex _index_data_guard;

  uint32_t _num_threads;

  // Remembers which nodes we've visited, to avoid re-computing distances.
  VisitedSetPool* _visited_set_pool;
  std::vector<std::mutex> _node_links_mutexes;

  bool _collect_stats = false;
  DataType _data_type;

  // NOTE: These metrics are meaningful the most with single-threaded search.
  // With multi-threaded search, for instance, the number of distance computations will 
  // accumulate across queries, which means at the end of the batched search, the number 
  // you get is the cumulative sum of all distance computations across all queries.
  // Maybe that's what you want, but it's worth noting
  mutable std::atomic<uint64_t> _distance_computations = 0;
  mutable std::atomic<uint64_t> _metric_hops = 0;

  // Keep track of the sequence of nodes visited during search.
  // Each internal list consists of a sequence of boolean flags indicating
  // whether a visited node is a hub node or not.
  std::vector<std::vector<bool>> _visited_nodes_sequence;

  bool* _hub_nodes = nullptr; // A boolean array to keep track of hub nodes.
  // If a node is a hub, then _hub_nodes[node] = true, else false.

  // Tracking metrics for node access patterns. This unordered map is used to
  // record how many times each node is visited during search. The key is the
  // node id and the value is the number of times the node is visited.
  std::unordered_map<uint32_t, uint32_t> _node_access_counts;

  // Optional flat per-node graph-link visit counter for the caching study.
  // Allocated only when enableVisitProfiling() is called (400 MB at 100M nodes);
  // incremented in processCandidateNode (hot path) only under FLATNAV_PROFILE_VISITS,
  // so non-profiling builds are byte-identical on the search path.
  std::atomic<uint32_t>* _node_visit_counts = nullptr;
  // Companion counter: per-node DATA (vector) accesses = full activated footprint
  // (superset of link-expanded nodes). Also allocated by enableVisitProfiling().
  std::atomic<uint32_t>* _node_data_counts = nullptr;

  // Design-1 (SSSP) common source: when >= 0, every search starts from this fixed
  // entry node (skipping the per-query initialization scan), so all traversals are
  // rooted at one vertex. Set via setFixedEntryNode(); -1 = normal per-query entry.
  int64_t _fixed_entry_node = -1;

  // Helper-thread vector staging (PrefetchStaging.h): shared local-DRAM buffer + request
  // ring + top-K knob. Null/0 = off. Set via setPrefetchStaging().
  StagingBuffer* _pf_buf = nullptr;
  MPMCRing* _pf_ring = nullptr;
  int _pf_k = 0;

#ifdef FLATNAV_PQ_GATE
  // Step-windowed PQ traversal. _pq_codes is _pq_m bytes per node, indexed by internal node id;
  // expansions in [_pq_gate_lo, _pq_gate_hi) are scored on PQ. hi <= lo = window empty = gate
  // off (exact search). Set via setPQGate().
  const uint8_t* _pq_codes = nullptr;
  uint32_t _pq_m = 0;
  int _pq_gate_lo = 0;
  int _pq_gate_hi = 0;
  // Optional per-vector residual norm ||x - c(x)||^2, node-id indexed: the static part of each
  // vector's PQ error, added back by pqDistance(). Null = uncorrected PQ. Set via setPQResidual().
  const float* _pq_resid = nullptr;
  // M15 quality gate: fetch the exact vector for a PQ-scored neighbor whose score lands within
  // _pq_margin (relative) of the beam's K-th best. 0 = off (pure PQ). _pq_margin_band selects the
  // shape: one-sided (verify everything PQ would admit, plus the band above the threshold) or a
  // two-sided band (skip nodes PQ places safely inside). Set via setPQMargin().
  float _pq_margin = 0.0f;
  bool _pq_margin_band = false;
  // Speculate-then-validate traversal (specBeamSearch). _spec_depth (k) is how many steps
  // validation trails speculation, _spec_width (w) how many candidates are speculated per step.
  // k = 0 leaves search() on beamSearch. _spec_oracle scores the speculative lane with exact
  // distances, which removes approximation error from the prediction. Set via setSpecPipeline().
  int _spec_depth = 0;
  int _spec_width = 1;
  bool _spec_oracle = false;
  // Classify every miss by scanning the candidate queue for the node that was predicted. Linear
  // in the queue and only worth paying for when the breakdown is the measurement. Set via
  // setSpecDiag().
  bool _spec_diag = false;
#endif

  // Randomization parameters
  bool _use_random_initialization = false;
  std::mt19937 _generator;
  std::uniform_int_distribution<> _distribution;


  Index(const Index &) = delete;
  Index &operator=(const Index &) = delete;

  // A custom move constructor is needed because the class manages dynamic
  // resources (_vectors_memory, _graph_memory, _visited_set_pool),
  // which require explicit ownership transfer and cleanup to avoid resource
  // leaks or double frees. The default move constructor cannot ensure these
  // resources are safely transferred and the source object is left in a valid
  // state.
  Index(Index&& other) noexcept
      : _vectors_memory(other._vectors_memory),
        _graph_memory(other._graph_memory),
        _vectors_numa_node(other._vectors_numa_node),
        _graph_numa_node(other._graph_numa_node),
        _M(other._M),
        _data_size_bytes(other._data_size_bytes),
        _node_size_bytes(other._node_size_bytes),
        _graph_node_size_bytes(other._graph_node_size_bytes),
        _max_node_count(other._max_node_count),
        _cur_num_nodes(other._cur_num_nodes),
        _distance(std::move(other._distance)),
        _num_threads(other._num_threads),
        _visited_set_pool(std::move(other._visited_set_pool)),
        _node_links_mutexes(std::move(other._node_links_mutexes)),
        _hub_nodes(other._hub_nodes) {
    other._vectors_memory = nullptr;
    other._graph_memory = nullptr;
    other._visited_set_pool = nullptr;
    other._hub_nodes = nullptr;
  }

  Index& operator=(Index&& other) noexcept {
    if (this != &other) {
      util::freeBytes(_vectors_memory, vectorsMemoryBytes(), _vectors_numa_node);
      util::freeBytes(_graph_memory, graphMemoryBytes(), _graph_numa_node);
      delete _visited_set_pool;
      delete[] _hub_nodes;

      _vectors_memory = other._vectors_memory;
      _graph_memory = other._graph_memory;
      _vectors_numa_node = other._vectors_numa_node;
      _graph_numa_node = other._graph_numa_node;
      _M = other._M;
      _data_size_bytes = other._data_size_bytes;
      _node_size_bytes = other._node_size_bytes;
      _graph_node_size_bytes = other._graph_node_size_bytes;
      _max_node_count = other._max_node_count;
      _cur_num_nodes = other._cur_num_nodes;
      _distance = std::move(other._distance);
      _num_threads = other._num_threads;
      _visited_set_pool = std::move(other._visited_set_pool);
      _node_links_mutexes = std::move(other._node_links_mutexes);
      _hub_nodes = other._hub_nodes;

      other._vectors_memory = nullptr;
      other._graph_memory = nullptr;
      other._visited_set_pool = nullptr;
      other._hub_nodes = nullptr;
    }
    return *this;
  }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_data_type, _M, _data_size_bytes, _node_size_bytes,
            _graph_node_size_bytes, _max_node_count, _cur_num_nodes, *_distance);

    // Serialize the two storage regions separately. NUMA placement is a runtime
    // concern and is intentionally not persisted.
    archive(cereal::binary_data(_vectors_memory, vectorsMemoryBytes()));
    archive(cereal::binary_data(_graph_memory, graphMemoryBytes()));
  }

 public:
  /**
   * @brief Construct a new Index object for approximate near neighbor search.
   *
   * This constructor initializes an Index object with the specified distance
   * metric, dataset size, and maximum number of links per node. It also allows
   * for collecting statistics during the search process.
   *
   * @param dist The distance metric for the index. Options include l2
   * (euclidean) and inner product.
   * @param dataset_size The maximum number of vectors that can be inserted in
   * the index.
   * @param max_edges_per_node The maximum number of links per node.
   * @param collect_stats Flag indicating whether to collect statistics during
   * the search process.
   */
  Index(std::unique_ptr<DistanceInterface<dist_t>> dist, int dataset_size,
        int max_edges_per_node, bool collect_stats = false,
        bool use_random_initialization = false,
        std::optional<size_t> random_seed = std::nullopt,
        DataType data_type = DataType::float32,
        int vectors_numa_node = kNoNumaNode, int graph_numa_node = kNoNumaNode)
      : _vectors_numa_node(vectors_numa_node), _graph_numa_node(graph_numa_node),
        _M(max_edges_per_node), _max_node_count(dataset_size),
        _cur_num_nodes(0), _distance(std::move(dist)), _num_threads(1),
        _visited_set_pool(new VisitedSetPool(
            /* initial_pool_size = */ 1,
            /* num_elements = */ dataset_size)),
        _node_links_mutexes(dataset_size), _collect_stats(collect_stats),
        _use_random_initialization(use_random_initialization),
        _data_type(data_type) {

    if (random_seed.has_value()) {
      _generator = std::mt19937(random_seed.value());
      _distribution = std::uniform_int_distribution<>(0, _max_node_count - 1);
    }

    initNodeAccessCounts();

    _data_size_bytes = _distance->dataSize();
    _graph_node_size_bytes = (sizeof(node_id_t) * _M) + sizeof(label_t);
    _node_size_bytes = _data_size_bytes + _graph_node_size_bytes;

    _vectors_memory =
        util::allocateBytes(vectorsMemoryBytes(), _vectors_numa_node);
    _graph_memory = util::allocateBytes(graphMemoryBytes(), _graph_numa_node);

    _hub_nodes = new bool[_max_node_count];
    std::fill_n(_hub_nodes, _max_node_count, false);
  }

  void initNodeAccessCounts() {
    // Initialize the node access counts to 0 for all nodes.
    for (uint32_t i = 0; i < _max_node_count; i++) {
      _node_access_counts[i] = 0;
    }
  }

  ~Index() {
    util::freeBytes(_vectors_memory, vectorsMemoryBytes(), _vectors_numa_node);
    util::freeBytes(_graph_memory, graphMemoryBytes(), _graph_numa_node);
    delete _visited_set_pool;
    delete[] _hub_nodes;
    delete[] _node_visit_counts;
    delete[] _node_data_counts;
  }

  /**
   * @brief re-prune the graph by removing edges to hub nodes.
   * @param hub_nodes The hub nodes to prune edges from.
   * @param alpha The pruning threshold. \alpha ranges from 0 to 1.
   * Ex. if alpha = 0.5, then we remove 50% of the edges from the hub nodes.
   * Edge removal is done by setting the edge to the node itself.
   * The edge selection process is done using random selection.
   */
  void rePruneGraph(const std::vector<uint32_t> &hub_nodes, float alpha) {

    if (alpha < 0 || alpha > 1) {
      throw std::invalid_argument("Alpha must be in the range [0, 1].");
    }

    std::vector<std::pair<uint32_t, uint32_t>> edges_between_hub_nodes;
    for (const auto &hub_node : hub_nodes) {
      node_id_t *links = getNodeLinks(hub_node);
      for (size_t i = 0; i < _M; i++) {
        if (links[i] != hub_node) {
          edges_between_hub_nodes.emplace_back(hub_node, links[i]);
        }
      }
    }

    // Now randomly pick alpha * |edges_between_hub_nodes| edges to remove.
    std::shuffle(edges_between_hub_nodes.begin(), edges_between_hub_nodes.end(),
                 _generator);

    size_t num_edges_to_remove =
        static_cast<size_t>(alpha * edges_between_hub_nodes.size());
    for (size_t i = 0; i < num_edges_to_remove; i++) {
      auto [a, b] = edges_between_hub_nodes[i];
      node_id_t *links = getNodeLinks(a);
      for (size_t j = 0; j < _M; j++) {
        if (links[j] == b) {
          links[j] = a;
          break;
        }
      }
    }
  }

  void resetNodeAccessDistribution() { _node_access_counts.clear(); }

  // Caching study: allocate the flat per-node link-expansion + data-access counters (zeroed).
  void enableVisitProfiling() {
    delete[] _node_visit_counts;
    delete[] _node_data_counts;
    _node_visit_counts = new std::atomic<uint32_t>[_max_node_count]();
    _node_data_counts = new std::atomic<uint32_t>[_max_node_count]();
  }
  const std::atomic<uint32_t>* nodeVisitCounts() const { return _node_visit_counts; }
  const std::atomic<uint32_t>* nodeDataCounts() const { return _node_data_counts; }

  // Design-1 SSSP: fix the common source vertex for all subsequent searches.
  void setFixedEntryNode(int64_t s) { _fixed_entry_node = s; }

#ifdef FLATNAV_PQ_GATE
  // Step-windowed PQ traversal: score expansions in [lo, hi) on the PQ codes (`m` bytes per
  // node, node-id indexed), exact outside. The caller must also install a per-query LUT in
  // flatnav::tl_pq_lut on the searching thread. hi <= lo -> exact search everywhere.
  void setPQGate(const uint8_t* codes, uint32_t m, int lo, int hi) {
    _pq_codes = codes;
    _pq_m = m;
    _pq_gate_lo = lo;
    _pq_gate_hi = hi;
  }

  // Install (or clear, with nullptr) the per-vector residual-norm correction used by
  // pqDistance(). `resid[n] = ||x_n - c(x_n)||^2`, node-id indexed.
  void setPQResidual(const float* resid) { _pq_resid = resid; }

  // M15 quality gate. `margin` is relative to the beam's K-th best (0.1 = within 10%); 0 disables
  // it. `band` = two-sided (only near-ties are verified) vs one-sided (everything at or below the
  // threshold is verified too).
  void setPQMargin(float margin, bool band) { _pq_margin = margin; _pq_margin_band = band; }

  // k = validation depth (0 = off), w = speculation width. Requires a per-query LUT in
  // flatnav::tl_pq_lut, as the gate does. `oracle` scores the speculative lane with exact
  // distances instead of PQ, leaving the stale admission threshold as the only source of misses.
  void setSpecPipeline(int k, int w, bool oracle = false) {
    _spec_depth = k;
    _spec_width = w < 1 ? 1 : w;
    _spec_oracle = oracle;
  }

  void setSpecDiag(bool on) { _spec_diag = on; }
#endif

  // Helper-thread vector staging (PrefetchStaging.h). When configured, each pop enqueues
  // the top-K pending candidate ids into `ring`; helper threads dequeue them and call
  // stageNeighbors() to copy the candidate's neighbors' vectors remote->`buf`; the search
  // reads staged copies (local) when available. All hints -- results unchanged.
  void setPrefetchStaging(StagingBuffer* buf, MPMCRing* ring, int k) {
    _pf_buf = buf;
    _pf_ring = ring;
    _pf_k = k;
  }

  // Helper thread: copy the vectors of `candidate`'s neighbors into the staging buffer.
  void stageNeighbors(uint32_t candidate) {
    if (!_pf_buf) return;
    node_id_t* links = getNodeLinks(candidate);
    for (uint32_t j = 0; j < _M; j++) {
      node_id_t nbr = links[j];
      _pf_buf->put(nbr, getNodeData(nbr));
    }
  }

  // Ablation: read `candidate`'s neighbors' vectors but discard them (no buffer write).
  // Injects the helper remote-read traffic without the buffer/coherence, isolating
  // shared remote-path contention from the buffer-coherence term.
  void touchNeighbors(uint32_t candidate) {
    node_id_t* links = getNodeLinks(candidate);
    volatile char sink = 0;
    for (uint32_t j = 0; j < _M; j++) {
      const char* v = getNodeData(links[j]);
      for (size_t off = 0; off < _data_size_bytes; off += 64) sink ^= v[off];
    }
    (void)sink;
  }

  // Medoid = node nearest to the dataset mean vector (a principled central source).
  // Two streaming passes over the vectors.
  node_id_t computeMedoid() {
    const size_t dim = _data_size_bytes / sizeof(float);
    std::vector<double> mean(dim, 0.0);
    for (node_id_t n = 0; n < _cur_num_nodes; n++) {
      const float* v = reinterpret_cast<const float*>(getNodeData(n));
      for (size_t d = 0; d < dim; d++) mean[d] += v[d];
    }
    std::vector<float> m(dim);
    for (size_t d = 0; d < dim; d++) m[d] = static_cast<float>(mean[d] / (double)_cur_num_nodes);
    node_id_t best = 0; double best_dist = std::numeric_limits<double>::max();
    for (node_id_t n = 0; n < _cur_num_nodes; n++) {
      const float* v = reinterpret_cast<const float*>(getNodeData(n));
      double s = 0;
      for (size_t d = 0; d < dim; d++) { double diff = (double)v[d] - m[d]; s += diff * diff; }
      if (s < best_dist) { best_dist = s; best = n; }
    }
    return best;
  }

  // Public access to relabel for external placement policies (P[old]=new id).
  void reorderByPermutation(const std::vector<node_id_t>& P) { relabel(P); }

  // In-degree of every node = number of incoming out-edges from other nodes
  // (self-loops excluded). A "hub" is then defined as a top-percentile in-degree node.
  std::vector<uint32_t> computeInDegrees() {
    std::vector<uint32_t> indeg(_cur_num_nodes, 0);
    for (node_id_t u = 0; u < _cur_num_nodes; u++) {
      const node_id_t* links = getNodeLinks(u);
      for (uint32_t i = 0; i < _M; i++) {
        node_id_t w = links[i];
        if (w < _cur_num_nodes && w != u) indeg[w]++;
      }
    }
    return indeg;
  }

  // Hop (graph-BFS) distance from `source` to every node over out-edges; 255 = unreachable.
  std::vector<uint8_t> bfsHopDistances(node_id_t source) {
    std::vector<uint8_t> dist(_cur_num_nodes, 255);
    std::vector<node_id_t> frontier, next;
    dist[source] = 0; frontier.push_back(source);
    uint8_t h = 0;
    while (!frontier.empty() && h < 254) {
      for (node_id_t u : frontier) {
        const node_id_t* links = getNodeLinks(u);
        for (uint32_t i = 0; i < _M; i++) {
          node_id_t w = links[i];
          if (w < _cur_num_nodes && dist[w] == 255) { dist[w] = h + 1; next.push_back(w); }
        }
      }
      frontier.swap(next); next.clear(); h++;
    }
    return dist;
  }

  void setHubNodeFlags(const std::vector<uint32_t>& hub_nodes) {
      for (const auto& hub_node: hub_nodes) {
        _hub_nodes[hub_node] = true;  
      }
  }

  std::vector<std::vector<bool>> getVisitedNodesSequence() {
    return _visited_nodes_sequence;
  }


  void buildGraphLinks(const std::string& mtx_filename) {
    std::ifstream input_file(mtx_filename);
    if (!input_file.is_open()) {
      throw std::runtime_error("Unable to open file for reading: " + mtx_filename);
    }

    std::string line;
    // Skip the header
    while (std::getline(input_file, line)) {
      if (line[0] != '%')
        break;
    }

    std::istringstream iss(line);
    int num_vertices, num_edges;
    iss >> num_vertices >> num_vertices >> num_edges;

    // check that the number of vertices in the mtx file matches the number of
    // nodes in the index and that the number of edges is equal to the number of
    // links per node.
    if (num_vertices != _max_node_count) {
      throw std::runtime_error(
          "Number of vertices in the mtx file does not "
          "match the size allocated for the index.");
    }

    if (num_edges != _M) {
      throw std::runtime_error(
          "Number of edges in the mtx file does not match "
          "the number of links per node.");
    }

    int u, v;
    while (input_file >> u >> v) {
      // Adjust for 1-based indexing in Matrix Market format
      u--;
      v--;
      node_id_t* links = getNodeLinks(u);
      // Now add a directed edge from u to v. We need to check for the first
      // available slot in the links array since there might be other edges
      // added before this one. By definition, a slot is available if and only
      // if it points to the node itself.
      for (size_t i = 0; i < _M; i++) {
        if (links[i] == u) {
          links[i] = v;
          break;
        }
      }
    }

    input_file.close();
  }

  std::vector<std::vector<uint32_t>> getGraphOutdegreeTable() {
    std::vector<std::vector<uint32_t>> outdegree_table(_cur_num_nodes);
    for (node_id_t node = 0; node < _cur_num_nodes; node++) {
      // allocate a vector of size 0 so that each node has an entry in the
      // outdegree table.
      outdegree_table[node] = std::vector<uint32_t>();
      node_id_t *links = getNodeLinks(node);
      for (int i = 0; i < _M; i++) {
        if (links[i] != node) {
          outdegree_table[node].push_back(links[i]);
        }
      }
    }
    return outdegree_table;
  }

  size_t cantorPairing(node_id_t a, node_id_t b) {
    // if (a > b) {
    //   std::swap(a, b);
    // }
    return (a + b) * (a + b + 1) / 2 + b;
  }

  /**
   * @brief Store the new node in the global data structure. In a
   * multi-threaded setting, the index data guard should be held by the caller
   * with an exclusive lock.
   *
   * @param data The vector to add.
   * @param label The label (meta-data) of the vector.
   * @param new_node_id The id of the new node.
   */
  void allocateNode(void* data, label_t& label, node_id_t& new_node_id) {
    new_node_id = _cur_num_nodes;
    _distance->transformData(
        /* destination = */ getNodeData(new_node_id),
        /* src = */ data);
    *(getNodeLabel(new_node_id)) = label;
    node_id_t* links = getNodeLinks(new_node_id);
    // Initialize all edges to self
    std::fill_n(links, _M, new_node_id);
    _cur_num_nodes++;
  }

  /**
   * @brief Adds vectors to the index in batches.
   *
   * This method is responsible for adding vectors in batches, represented by
   * `data`, to the underlying graph. Each vector is associated with a label
   * provided in the `labels` vector. The method efficiently handles concurrent
   * additions by dividing the workload among multiple threads, defined by
   * `_num_threads`.
   *
   * The method ensures thread safety by employing locking mechanisms at the
   * node level in the underlying `connectNeighbors` and `beamSearch` methods.
   * This allows multiple threads to safely add vectors to the index without
   * causing data races or inconsistencies in the graph structure.
   *
   * @param data Pointer to the array of vectors to be added.
   * @param labels A vector of labels corresponding to each vector in `data`.
   * @param ef_construction Parameter for controlling the size of the dynamic
   * candidate list during the construction of the graph.
   * @param num_initializations Number of initializations for the search
   * algorithm. Must be greater than 0.
   *
   * @exception std::invalid_argument Thrown if `num_initializations` is less
   * than or equal to 0.
   * @exception std::runtime_error Thrown if the maximum number of nodes in the
   * index is reached.
   */
  template <typename data_type>
  void addBatch(void* data, std::vector<label_t>& labels, int ef_construction,
                int num_initializations = 100) {
      if (num_initializations <= 0) {
          throw std::invalid_argument("num_initializations must be greater than 0.");
      }
      uint32_t total_num_nodes = labels.size();
      uint32_t data_dimension = _distance->dimension();

      // Don't spawn any threads if we are only using one.
      if (_num_threads == 1) {
          for (uint32_t row_index = 0; row_index < total_num_nodes; row_index++) {
              uint64_t offset = static_cast<uint64_t>(row_index) * static_cast<uint64_t>(data_dimension);
              void* vector = (data_type*)data + offset;
              label_t label = labels[row_index];
              this->add(vector, label, ef_construction, num_initializations);
          }
          return;
      }

      flatnav::executeInParallel(
          /* start_index = */ 0, /* end_index = */ total_num_nodes,
          /* num_threads = */ _num_threads, /* function = */
          [&](uint32_t row_index) {
              uint64_t offset = static_cast<uint64_t>(row_index) * static_cast<uint64_t>(data_dimension);
              void* vector = (data_type*)data + offset;
              label_t label = labels[row_index];
              this->add(vector, label, ef_construction, num_initializations);
          });
  }

  /**
   * @brief Adds a single vector to the index.
   *
   * This method is called internally by `addBatch` for each vector in the
   * batch. The method ensures thread safety by using locking primitives,
   * allowing it to be safely used in a multi-threaded environment.
   *
   * The method first checks if the current number of nodes has reached the
   * maximum capacity. If so, it throws a runtime error. It then locks the index
   * structure to prevent concurrent modifications while allocating a new node.
   * After unlocking, it connects the new node to its neighbors in the graph.
   *
   * @param data Pointer to the vector data being added.
   * @param label Label associated with the vector.
   * @param ef_construction Parameter controlling the size of the dynamic
   * candidate list during the construction of the graph.
   * @param num_initializations Number of initializations for the search
   * algorithm.
   *
   * @exception std::runtime_error Thrown if the maximum number of nodes is
   * reached.
   */
  void add(void* data, label_t& label, int ef_construction, int num_initializations) {

    if (_cur_num_nodes >= _max_node_count) {
      throw std::runtime_error(
          "Maximum number of nodes reached. Consider "
          "increasing the `max_node_count` parameter to "
          "create a larger index.");
    }
    std::unique_lock<std::mutex> global_lock(_index_data_guard);
    auto entry_node = initializeSearch(data, num_initializations);
    node_id_t new_node_id;
    allocateNode(data, label, new_node_id);
    global_lock.unlock();

    if (new_node_id == 0) {
      return;
    }

    auto neighbors = beamSearch<false>(
        /* query = */ data, /* entry_node = */ entry_node,
        /* buffer_size = */ ef_construction);

    int selection_M = std::max(static_cast<int>(_M / 2), 1);
    selectNeighbors(/* neighbors = */ neighbors, /* M = */ selection_M);
    connectNeighbors(neighbors, new_node_id);
  }

  /***
   * @brief Search the index for the k nearest neighbors of the query.
   * @param query The query vector.
   * @param K The number of nearest neighbors to return.
   * @param ef_search The search beam width.
   * @param num_initializations The number of random initializations to use.
   */
  std::vector<dist_label_t> search(const void* query, const int K, int ef_search,
                                   int num_initializations = 100) {
    node_id_t entry_node;
    if (_fixed_entry_node >= 0) {
      entry_node = static_cast<node_id_t>(_fixed_entry_node);
    } else if (_use_random_initialization) {
      entry_node = randomlyInitializeSearch(query, num_initializations);
    } else {
      entry_node = initializeSearch(query, num_initializations);
    }
    const int buffer_size = std::max(K, ef_search);
#ifdef FLATNAV_PQ_GATE
    PriorityQueue neighbors =
        (_spec_depth > 0 && tl_pq_lut)
            ? specBeamSearch(/* query = */ query, /* entry_node = */ entry_node,
                             /* buffer_size = */ buffer_size)
            : beamSearch<true>(/* query = */ query, /* entry_node = */ entry_node,
                               /* buffer_size = */ buffer_size);
#else
    PriorityQueue neighbors =
        beamSearch<true>(/* query = */ query,
                         /* entry_node = */ entry_node,
                         /* buffer_size = */ buffer_size);
#endif
    auto size = neighbors.size();
    std::vector<dist_label_t> results;
    results.reserve(size);
    while (!neighbors.empty()) {
      auto [distance, node_id] = neighbors.top();
      auto label = *getNodeLabel(node_id);
      results.emplace_back(distance, label);
      neighbors.pop();
    }
    std::sort(results.begin(), results.end(),
              [](const dist_label_t& left, const dist_label_t& right) { return left.first < right.first; });
    if (results.size() > static_cast<size_t>(K)) {
      results.resize(K);
    }

    return results;
  }


  void doGraphReordering(const std::vector<std::string>& reordering_methods) {

    for (const auto& method : reordering_methods) {
      auto outdegree_table = getGraphOutdegreeTable();
      std::vector<node_id_t> P;
      if (method == "gorder") {
        P = std::move(util::gOrder<node_id_t>(outdegree_table, 5));
      } else if (method == "rcm") {
        P = std::move(util::rcmOrder<node_id_t>(outdegree_table));
      } else {
        throw std::invalid_argument("Invalid reordering method: " + method);
      }

      relabel(P);
    }
  }

  void reorderGOrder(const int window_size = 5) {
    auto outdegree_table = getGraphOutdegreeTable();
    std::vector<node_id_t> P = util::gOrder<node_id_t>(outdegree_table, window_size);

    relabel(P);
  }

  void reorderRCM() {
    auto outdegree_table = getGraphOutdegreeTable();
    std::vector<node_id_t> P = util::rcmOrder<node_id_t>(outdegree_table);
    relabel(P);
  }

  // NUMA placement (vectors_numa_node, graph_numa_node) lets the caller bind each
  // storage region to a specific NUMA node at load time (caching study: hot graph
  // links on the local node, vectors on the remote node). kNoNumaNode = default
  // allocator. Requires building with FLATNAV_USE_NUMA.
  static std::unique_ptr<Index<dist_t, label_t>> loadIndex(
      const std::string& filename,
      int vectors_numa_node = util::kNoNumaNode,
      int graph_numa_node = util::kNoNumaNode) {
    std::ifstream stream(filename, std::ios::binary);

    if (!stream.is_open()) {
      throw std::runtime_error("Unable to open file for reading: " + filename);
    }

    cereal::BinaryInputArchive archive(stream);
    std::unique_ptr<Index<dist_t, label_t>> index(new Index<dist_t, label_t>());

    std::unique_ptr<DistanceInterface<dist_t>> dist = std::make_unique<dist_t>();

    // 1. Deserialize metadata
    archive(index->_data_type,
            index->_M,
            index->_data_size_bytes,
            index->_node_size_bytes,
            index->_graph_node_size_bytes,
            index->_max_node_count,
            index->_cur_num_nodes,
            *dist
    );
    index->_visited_set_pool = new VisitedSetPool(
        /* initial_pool_size = */ 1,
        /* num_elements = */ index->_max_node_count);
    index->_distance = std::move(dist);
    index->_num_threads = std::max((uint32_t)1, (uint32_t)std::thread::hardware_concurrency() / 2);
    index->_node_links_mutexes = std::vector<std::mutex>(index->_max_node_count);

    // Hub-node flags are not serialized; a loaded index starts with no hubs
    // marked (matching a freshly constructed index before setHubNodeFlags).
    index->_hub_nodes = new bool[index->_max_node_count];
    std::fill_n(index->_hub_nodes, index->_max_node_count, false);

    // 2. Allocate the two storage regions using deserialized metadata. NUMA
    // placement is not persisted; the caller may bind each region via the
    // loadIndex(filename, vectors_node, graph_node) overload.
    index->_vectors_numa_node = vectors_numa_node;
    index->_graph_numa_node = graph_numa_node;
    index->_vectors_memory =
        util::allocateBytes(index->vectorsMemoryBytes(), index->_vectors_numa_node);
    index->_graph_memory =
        util::allocateBytes(index->graphMemoryBytes(), index->_graph_numa_node);

    // 3. Deserialize content into the allocated regions.
    archive(cereal::binary_data(index->_vectors_memory, index->vectorsMemoryBytes()));
    archive(cereal::binary_data(index->_graph_memory, index->graphMemoryBytes()));

    return index;
  }

  void saveIndex(const std::string& filename) {
    std::ofstream stream(filename, std::ios::binary);

    if (!stream.is_open()) {
      throw std::runtime_error("Unable to open file for writing: " + filename);
    }

    cereal::BinaryOutputArchive archive(stream);
    archive(*this);
  }

  inline void setNumThreads(uint32_t num_threads) {
    if (num_threads == 0 || num_threads > std::thread::hardware_concurrency()) {
      throw std::invalid_argument(
          "Number of threads must be greater than 0 and less than or equal to "
          "the number of hardware threads.");
    }
    _num_threads = num_threads;
    if (_num_threads == 1) {
      _visited_set_pool->setPoolSize(1);
    }
  }


  inline uint64_t getTotalIndexMemory() const {
    return static_cast<uint64_t>(_node_size_bytes) * static_cast<uint64_t>(_max_node_count);
  }

  // Byte size of the vectors region ([data] per node).
  inline uint64_t vectorsMemoryBytes() const {
    return static_cast<uint64_t>(_data_size_bytes) * static_cast<uint64_t>(_max_node_count);
  }

  // Byte size of the graph region ([M links][label] per node).
  inline uint64_t graphMemoryBytes() const {
    return static_cast<uint64_t>(_graph_node_size_bytes) * static_cast<uint64_t>(_max_node_count);
  }
  inline uint64_t mutexesAllocatedMemory() const {
    return static_cast<uint64_t>(_node_links_mutexes.size() * sizeof(std::mutex));
  }

  inline uint64_t visitedSetPoolAllocatedMemory() const {
    size_t pool_size = _visited_set_pool->poolSize();
    return static_cast<uint64_t>(pool_size * sizeof(VisitedSet));
  }

  inline uint32_t getNumThreads() const { return _num_threads; }

  inline size_t maxEdgesPerNode() const { return _M; }
  inline size_t dataSizeBytes() const { return _data_size_bytes; }

  inline size_t nodeSizeBytes() const { return _node_size_bytes; }

  inline size_t maxNodeCount() const { return _max_node_count; }

  inline size_t currentNumNodes() const { return _cur_num_nodes; }
  inline size_t dataDimension() const { return _distance->dimension(); }

  // Region pointers + graph stride — for external NUMA tiering (mbind) after relabel.
  inline char* vectorsMemory() const { return _vectors_memory; }
  inline char* graphMemory() const { return _graph_memory; }
  inline size_t graphNodeSizeBytes() const { return _graph_node_size_bytes; }

  // Internal node id -> dataset label. Tools that trace internal ids (FLATNAV_SPEC_TRACE) need
  // this to line a traced node up with a ground-truth file, which is keyed by label.
  inline label_t nodeLabel(const node_id_t& n) const { return *getNodeLabel(n); }

  inline uint64_t distanceComputations() const { return _distance_computations.load(); }

  inline DataType getDataType() const { return _data_type; }

  void resetStats() {
    _distance_computations = 0;
    _metric_hops = 0;
  }

  // Return a reference to the node access counts
  inline const std::unordered_map<uint32_t, uint32_t> &
  getNodeAccessCounts() const {
    return _node_access_counts;
  }

  void getIndexSummary() const {
    std::cout << "\nIndex Parameters\n" << std::flush;
    std::cout << "-----------------------------\n" << std::flush;
    std::cout << "max_edges_per_node (M): " << _M << "\n" << std::flush;
    std::cout << "data_size_bytes: " << _data_size_bytes << "\n" << std::flush;
    std::cout << "node_size_bytes: " << _node_size_bytes << "\n" << std::flush;
    std::cout << "max_node_count: " << _max_node_count << "\n" << std::flush;
    std::cout << "cur_num_nodes: " << _cur_num_nodes << "\n" << std::flush;

    _distance->getSummary();
  }

 private:
  friend class cereal::access;
  // Default constructor for cereal
  Index() = default;

  char* getNodeData(const node_id_t& n) const {
    uint64_t byte_offset = static_cast<uint64_t>(n) * static_cast<uint64_t>(_data_size_bytes);
    return _vectors_memory + byte_offset;
  }

  node_id_t* getNodeLinks(const node_id_t& n) const {
    uint64_t byte_offset = static_cast<uint64_t>(n) * static_cast<uint64_t>(_graph_node_size_bytes);
    char* location = _graph_memory + byte_offset;
    return reinterpret_cast<node_id_t*>(location);
  }

#ifdef FLATNAV_PQ_GATE
  // Asymmetric PQ distance: sum over subquantizers of the query-to-centroid distance picked
  // out by this node's code byte. Touches the m-byte code only -- never the full vector.
  //
  // With x = c + r (reconstruction + residual), the exact distance expands as
  //     ||q-x||^2 = ||q-c||^2 - 2(q-c).r + ||r||^2,
  // i.e. this sum is short of the truth by ||r||^2 (a per-vector constant -- PQ systematically
  // UNDERestimates) plus a zero-mean query-dependent cross term. When _pq_resid is installed
  // (M14) the per-vector bias is added back, leaving only the cross term; the correction is one
  // local float per vector and does not read the vector either.
  float pqDistance(const node_id_t& n) const {
    const uint8_t* code = _pq_codes + static_cast<uint64_t>(n) * static_cast<uint64_t>(_pq_m);
    float d = 0.0f;
    for (uint32_t m = 0; m < _pq_m; m++) {
      d += tl_pq_lut[(m << 8) + code[m]];
    }
    return _pq_resid ? d + _pq_resid[n] : d;
  }

  // End of the PQ phase: replace every PQ score in the beam with the exact distance (one
  // vector read each). A priority_queue has no in-place key update, so it is drained and
  // rebuilt. The beam stores +distance, so its max-heap top is the furthest member. When a
  // candidate pass follows (the gate-step switch) pass a `cache` to record the distances --
  // a node usually sits in both queues, and g_gate_rescore_dists must count DISTINCT vector
  // reads. Pass nullptr when nothing follows, so no map is built.
  void rescoreBeamExact(const void* query, PriorityQueue& beam,
                        std::unordered_map<node_id_t, float>* cache) {
    PriorityQueue rescored;
    uint64_t reads = 0;
    while (!beam.empty()) {
      node_id_t n = beam.top().second;
      beam.pop();
      float d = _distance->distance(/* x = */ query, /* y = */ getNodeData(n),
                                    /* asymmetric = */ true);
      if (cache) (*cache)[n] = d;
      rescored.emplace(d, n);
      reads++;
    }
    beam = std::move(rescored);
    g_gate_rescore_dists.fetch_add(reads, std::memory_order_relaxed);
  }

  // Same, for the candidate queue -- which stores NEGATED distances so its top is the nearest
  // pending node. Nodes already rescored with the beam are reused from `cache` (no vector read).
  void rescoreCandidatesExact(const void* query, PriorityQueue& candidates,
                              const std::unordered_map<node_id_t, float>& cache) {
    PriorityQueue rescored;
    uint64_t reads = 0;
    while (!candidates.empty()) {
      node_id_t n = candidates.top().second;
      candidates.pop();
      auto it = cache.find(n);
      float d;
      if (it != cache.end()) {
        d = it->second;
      } else {
        d = _distance->distance(/* x = */ query, /* y = */ getNodeData(n),
                                /* asymmetric = */ true);
        reads++;
      }
      rescored.emplace(-d, n);
    }
    candidates = std::move(rescored);
    g_gate_rescore_dists.fetch_add(reads, std::memory_order_relaxed);
  }
#endif

  label_t* getNodeLabel(const node_id_t& n) const {
    uint64_t byte_offset = static_cast<uint64_t>(n) * static_cast<uint64_t>(_graph_node_size_bytes);
    byte_offset += (_M * sizeof(node_id_t));
    char* location = _graph_memory + byte_offset;
    return reinterpret_cast<label_t*>(location);
  }

  inline void swapNodes(node_id_t a, node_id_t b, void* temp_data, node_id_t* temp_links,
                        label_t* temp_label) {

    // stash b in temp
    std::memcpy(temp_data, getNodeData(b), _data_size_bytes);
    std::memcpy(temp_links, getNodeLinks(b), _M * sizeof(node_id_t));
    std::memcpy(temp_label, getNodeLabel(b), sizeof(label_t));

    // place node at a in b
    std::memcpy(getNodeData(b), getNodeData(a), _data_size_bytes);
    std::memcpy(getNodeLinks(b), getNodeLinks(a), _M * sizeof(node_id_t));
    std::memcpy(getNodeLabel(b), getNodeLabel(a), sizeof(label_t));

    // put node b in a
    std::memcpy(getNodeData(a), temp_data, _data_size_bytes);
    std::memcpy(getNodeLinks(a), temp_links, _M * sizeof(node_id_t));
    std::memcpy(getNodeLabel(a), temp_label, sizeof(label_t));
  }

  /**
   * @brief Performs beam search for the nearest neighbors of the query.
   * @TODO: Add `entry_node_dist` argument to this function since we expect to
   * have computed that a priori.
   *
   * @param query               The query vector.
   * @param entry_node          The node to start the search from.
   * @param buffer_size         This is equivalent to `ef_search` in the HNSW
   *
   * @return PriorityQueue
   */
  template <bool is_search_stage = false>
  PriorityQueue beamSearch(const void *query, const node_id_t entry_node,
                           const int buffer_size) {
    PriorityQueue neighbors;
    PriorityQueue candidates;

    // Keep track of the nodes visited during the search.
    // Add True if the node is a hub node, else False.
    std::vector<bool> query_visited_nodes_flags;

    auto *visited_set = _visited_set_pool->pollAvailableSet();
    visited_set->clear();

    // Prefetch the data for entry node before computing its distance.
#if defined(USE_SSE) && !defined(FLATNAV_DISABLE_PREFETCH)
    _mm_prefetch(getNodeData(entry_node), _MM_HINT_T0);
#endif

    float dist;
#ifdef FLATNAV_PQ_GATE
    tl_gate_step = 0;
    tl_gate_pq = 0;
    tl_gate_exact = 0;
    if (tl_pq_lut && _pq_gate_hi > 0 && _pq_gate_lo == 0) {
      dist = pqDistance(entry_node);  // step 0 is inside the window
      tl_gate_pq++;
    } else
#endif
      dist = _distance->distance(/* x = */ query, /* y = */ getNodeData(entry_node),
                                 /* asymmetric = */ true);
#ifdef FLATNAV_PROFILE_VISITS
    if (_node_data_counts) _node_data_counts[entry_node].fetch_add(1, std::memory_order_relaxed);
#endif

    float max_dist = dist;
    candidates.emplace(-dist, entry_node);
    neighbors.emplace(dist, entry_node);
    query_visited_nodes_flags.push_back(_hub_nodes[entry_node]);
    visited_set->insert(entry_node);
#ifdef FLATNAV_PROFILE_PQ
    tl_disc.clear(); tl_parent_res.clear(); tl_step = 0; tl_cur_residency = 0;
    tl_disc[entry_node] = 0; tl_parent_res[entry_node] = 0;
    tl_pf.clear();
    tl_pf1_valid = false;
#endif
#ifdef FLATNAV_PROFILE_TRAJ
    tl_traj.clear();
    tl_term_node = entry_node; tl_term_dist = dist;  // entry is the initial closest
#endif
#ifdef FLATNAV_PROFILE_PHASE
    tl_pstep = 0; tl_bin_ts = 0;
#endif

    while (!candidates.empty()) {
#ifdef FLATNAV_PQ_GATE
      // The window ends at this expansion: rescore the beam + pending candidates with exact
      // distances, so the search resumes off a correct beam. Nodes scored exactly before the
      // window are re-read too (idempotent, deliberately not tracked). Fires at most once per
      // query, before this step reads candidates.top().
      if (tl_pq_lut && _pq_gate_hi > _pq_gate_lo && tl_gate_step == (uint32_t)_pq_gate_hi) {
        std::unordered_map<node_id_t, float> exact_cache;
        rescoreBeamExact(query, neighbors, &exact_cache);
        rescoreCandidatesExact(query, candidates, exact_cache);
        max_dist = neighbors.top().first;
      }
#endif
      auto [distance, node] = candidates.top();

      if (-distance > max_dist && neighbors.size() >= buffer_size) {
        break;
      }
      candidates.pop();
#ifdef FLATNAV_PROFILE_PHASE
      // Timestamp ONLY at 30-step bin boundaries: one rdtsc per bin, not per step. The delta
      // between consecutive boundaries is the wall time of that 30-expansion block.
      if (tl_pstep % kBinW == 0) {
        uint64_t now = __rdtsc();
        if (tl_bin_ts) {  // close out the bin that just finished (its full 30 steps)
          uint32_t pb = (tl_pstep / kBinW) - 1; if (pb >= (uint32_t)kBins) pb = kBins - 1;
          g_bin_cycles[pb].fetch_add(now - tl_bin_ts, std::memory_order_relaxed);
          g_bin_steps[pb].fetch_add(kBinW, std::memory_order_relaxed);
        }
        tl_bin_ts = now;
        uint32_t cb = tl_pstep / kBinW; if (cb >= (uint32_t)kBins) cb = kBins - 1;
        g_bin_queries[cb].fetch_add(1, std::memory_order_relaxed);  // this query reached bin cb
      }
      tl_pstep++;
#endif
#ifdef FLATNAV_PROFILE_PQ
      // Residency of the node being expanded = tl_step (its pop step) - its discovery step.
      // tl_step is NOT incremented until after the expansion, so neighbors discovered below
      // are tagged with this node's pop step.
      {
        auto it = tl_disc.find(node);
        uint32_t res = tl_step - (it != tl_disc.end() ? it->second : tl_step);
        g_pq_residency_hist[res < kPQHistCap ? res : kPQHistCap - 1].fetch_add(1, std::memory_order_relaxed);
        // per-step (pop index) buckets for the early/late split
        uint32_t sidx = tl_step < kPQStepCap ? tl_step : kPQStepCap - 1;
        g_step_count[sidx].fetch_add(1, std::memory_order_relaxed);
        g_step_sumres[sidx].fetch_add(res, std::memory_order_relaxed);
        if (res == 1) g_step_leap[sidx].fetch_add(1, std::memory_order_relaxed);
        tl_cur_residency = res;            // neighbors discovered now inherit this as parent residency
        if (res == 1) {                    // leapfrogger (res 0 = entry node): how much lead did its parent give?
          auto pit = tl_parent_res.find(node);
          uint32_t pr = (pit != tl_parent_res.end()) ? pit->second : 0;
          g_leapfrog_parent_hist[pr < kPQHistCap ? pr : kPQHistCap - 1].fetch_add(1, std::memory_order_relaxed);
        }
        // top-K prefetch policy: mark this node as popped (a prefetch "success").
        { auto& e = tl_pf[node]; e.popped = true; e.pop_step = (uint16_t)tl_step; }
        // top-K prefetch policy: snapshot the K closest pending candidates NOW -- just after the
        // pop, before processCandidateNode discovers this node's neighbors -- so this step's
        // leapfroggers are excluded from every K. Record each node's first-prefetch step. tl_step
        // is still this step.
        {
          PriorityQueue tmp = candidates;  // copy; pop to read closest-first
          int maxK = kPFK[kPFnumK - 1];
          for (int r = 1; r <= maxK && !tmp.empty(); ++r) {
            node_id_t x = tmp.top().second; tmp.pop();
            auto& e = tl_pf[x];
            for (int ki = 0; ki < kPFnumK; ++ki)
              if (r <= kPFK[ki] && e.fe[ki] == 0xFFFF) e.fe[ki] = (uint16_t)tl_step;
          }
        }
        // pre-expansion K=1 NEXT-STEP precision: did last step's prefetched 2nd-min == the node
        // popped now? (i.e. used at the immediately next step). Then record this step's 2nd-min.
        if (tl_pf1_valid) {
          uint32_t b = tl_pf1_step < kPQStepCap ? tl_pf1_step : kPQStepCap - 1;
          g_pf1_next_total[b].fetch_add(1, std::memory_order_relaxed);
          if (node == tl_pf1_node) g_pf1_next_hit[b].fetch_add(1, std::memory_order_relaxed);
        }
        if (!candidates.empty()) {
          tl_pf1_node = candidates.top().second; tl_pf1_step = tl_step; tl_pf1_valid = true;
        } else {
          tl_pf1_valid = false;  // nothing left to prefetch this step
        }
      }
#endif

      // Prefetching the next candidate node data and visited set marker
      // before processing it. Note that this might not be useful if the current
      // iteration finds a neighbor that is closer than the current max
      // distance. In that case we would have prefetched data that is not used
      // immediately, but I think the cost of prefetching is low enough that
      // it's probably worth it.
#if defined(USE_SSE) && !defined(FLATNAV_DISABLE_PREFETCH)
      if (!candidates.empty()) {
        _mm_prefetch(getNodeData(candidates.top().second), _MM_HINT_T0);
        visited_set->prefetch(candidates.top().second);
      }
#endif

      // Experiment A (depth-0 tier prefetch): the next-best candidate is expanded
      // next iteration; its expansion reads its LINKS (the remote pointer-chase),
      // which is NOT covered by the getNodeData/vector prefetch above. Start that
      // fetch now so it overlaps the current node's expansion. The link block spans
      // _graph_node_size_bytes (~3 lines), so prefetch every cache line of it.
      // T1 (L2, skip L1): the links are used a full expansion later, so filling
      // L1 risks eviction by this expansion's ~16KB of neighbor vectors before use;
      // L2 residency still hides the remote/UPI latency with less L1 pollution.
      //
      // Guarded ONLY by FLATNAV_PF_LINKS -- deliberately independent of
      // FLATNAV_DISABLE_PREFETCH -- so this prefetch can be measured in isolation
      // (existing prefetches off, ours on) without the two mechanisms confounding.
#if defined(USE_SSE) && defined(FLATNAV_PF_LINKS)
      if (!candidates.empty()) {
        const char* links_base =
            reinterpret_cast<const char*>(getNodeLinks(candidates.top().second));
        for (size_t off = 0; off < _graph_node_size_bytes; off += 64)
          _mm_prefetch(links_base + off, _MM_HINT_T1);
      }
#endif

      // Helper-thread staging: enqueue the top-K pending candidates so the helper pool
      // stages their neighbors' vectors into the local buffer ahead of expansion. Lossy
      // (a full ring just drops the request -> remote read). Runtime-gated; no macro.
      if (_pf_ring && _pf_k > 0) {
        const std::vector<dist_node_t>& heap = pqHeap(candidates);
        int kk = std::min((int)heap.size(), _pf_k);
        for (int r = 0; r < kk; r++) _pf_ring->enqueue(heap[r].second);
      }

      // Exp B-ahead (vector prefetch, depth-1): the next-best candidate is expanded next
      // iteration; prefetch ITS neighbors' vectors now -> one full expansion of lead, and
      // covers leapfroggers (a leapfrogger is an out-neighbor of a candidate). Requires
      // READING next's links (a real load) to get the neighbor ids, then prefetching each
      // neighbor's full vector. Independent of FLATNAV_DISABLE_PREFETCH for isolation.
#if defined(USE_SSE) && defined(FLATNAV_PF_VEC_AHEAD)
      if (!candidates.empty()) {
        node_id_t* nxt_links = getNodeLinks(candidates.top().second);
        for (uint32_t j = 0; j < _M; j++) {
          node_id_t nid = nxt_links[j];
          if (!visited_set->isVisited(nid))
            _mm_prefetch(getNodeData(nid), _MM_HINT_T0);  // 1 line, skip visited (lightweight)
        }
      }
#endif

#ifdef FLATNAV_SPEC_TRACE
      // This node is being expanded now, at step tl_gate_step. Record the trajectory (this fires
      // after the early-break check, so only truly-expanded nodes are logged).
      if (tl_spec_expand) tl_spec_expand->push_back(node);
#endif
      processCandidateNode<is_search_stage>(
          /* query = */ query, /* node = */ node,
          /* max_dist = */ max_dist, /* buffer_size = */ buffer_size,
          /* visited_set = */ visited_set,
          /* neighbors = */ neighbors, /* candidates = */ candidates,
          /* query_visited_nodes = */ query_visited_nodes_flags);
#ifdef FLATNAV_PROFILE_PQ
      tl_step++;  // advance the step counter after the expansion completes
#endif
#ifdef FLATNAV_PQ_GATE
      tl_gate_step++;
#endif
    }
#ifdef FLATNAV_PROFILE_PHASE
    // Close the final (possibly partial) bin: from its start boundary to now.
    if (tl_bin_ts) {
      uint32_t lb = (tl_pstep - 1) / kBinW; if (lb >= (uint32_t)kBins) lb = kBins - 1;
      g_bin_cycles[lb].fetch_add(__rdtsc() - tl_bin_ts, std::memory_order_relaxed);
      g_bin_steps[lb].fetch_add(tl_pstep - lb * kBinW, std::memory_order_relaxed);  // steps in the last bin
    }
#endif
#ifdef FLATNAV_PROFILE_PQ
    // Terminal: the 2nd-min prefetched at the final step has no "next step" (search ended) -> miss.
    if (tl_pf1_valid) {
      uint32_t b = tl_pf1_step < kPQStepCap ? tl_pf1_step : kPQStepCap - 1;
      g_pf1_next_total[b].fetch_add(1, std::memory_order_relaxed);
    }
    // End of query: tally each prefetched node as success (popped) or waste (never popped),
    // bucketed by its first-prefetch step, for every K.
    for (auto& kv : tl_pf) {
      PFEntry& e = kv.second;
      for (int ki = 0; ki < kPFnumK; ++ki) {
        if (e.fe[ki] == 0xFFFF) continue;
        uint32_t s = e.fe[ki] < kPQStepCap ? e.fe[ki] : kPQStepCap - 1;
        g_pf_prefetched[ki][s].fetch_add(1, std::memory_order_relaxed);
        if (e.popped) {
          g_pf_success[ki][s].fetch_add(1, std::memory_order_relaxed);
          g_pf_leadsum[ki].fetch_add((uint32_t)(e.pop_step - e.fe[ki]), std::memory_order_relaxed);
        }
      }
    }
#endif
#ifdef FLATNAV_PROFILE_TOPK
    // `neighbors` holds the final beam (buffer_size elements). The top-K = the K smallest by
    // distance; record the discovery step (tl_disc) of each, and the last one (answer-complete).
    {
      std::vector<std::pair<float, node_id_t>> beam;
      beam.reserve(neighbors.size());
      { PriorityQueue nb = neighbors; while (!nb.empty()) { beam.push_back(nb.top()); nb.pop(); } }
      std::sort(beam.begin(), beam.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
      size_t topk = std::min((size_t)g_topk_K, beam.size());
      uint32_t maxdisc = 0;
      for (size_t r = 0; r < topk; r++) {
        auto it = tl_disc.find(beam[r].second);
        uint32_t ds = (it != tl_disc.end()) ? it->second : 0;
        uint32_t sb = ds < (uint32_t)kPQStepCap ? ds : kPQStepCap - 1;
        g_topk_found[sb].fetch_add(1, std::memory_order_relaxed);
        if (r == 0) g_topk_rank1[sb].fetch_add(1, std::memory_order_relaxed);
        if (ds > maxdisc) maxdisc = ds;
      }
      g_topk_complete[maxdisc < (uint32_t)kPQStepCap ? maxdisc : kPQStepCap - 1].fetch_add(1, std::memory_order_relaxed);
      g_topk_complete_sum.fetch_add(maxdisc, std::memory_order_relaxed);
      g_topk_total_sum.fetch_add(tl_step, std::memory_order_relaxed);
      g_topk_nq.fetch_add(1, std::memory_order_relaxed);
    }
#endif

#ifdef FLATNAV_PQ_GATE
    // The search ended inside the window (it never closed), so the beam may still hold PQ
    // scores -- rescore it so the returned top-K is ordered by exact distance. Skipped when
    // the search ended before ever reaching `lo`, since then nothing was scored on PQ.
    if (tl_pq_lut && _pq_gate_hi > _pq_gate_lo && tl_gate_step > (uint32_t)_pq_gate_lo &&
        tl_gate_step <= (uint32_t)_pq_gate_hi) {
      rescoreBeamExact(query, neighbors, /* cache = */ nullptr);  // nothing follows
    }
    // One pair of atomics per query instead of one per distance.
    g_gate_pq_dists.fetch_add(tl_gate_pq, std::memory_order_relaxed);
    g_gate_exact_dists.fetch_add(tl_gate_exact, std::memory_order_relaxed);
#endif

    _visited_set_pool->pushVisitedSet(
        /* visited_set = */ visited_set);

    return neighbors;
  }

  template <bool is_search_stage>
#ifdef FLATNAV_PROFILE_NOINLINE
  __attribute__((noinline))
#endif
  void processCandidateNode(const void *query, node_id_t &node, float &max_dist,
                            const int buffer_size, VisitedSet *visited_set,
                            PriorityQueue &neighbors,
                            PriorityQueue &candidates, std::vector<bool>& query_visited_nodes_flags) {
    // Lock all operations on this specific node. During search the graph is
    // frozen (read-only), so the per-node lock is pure overhead and is skipped;
    // it is only needed during construction (is_search_stage == false) to guard
    // concurrent writers.
    std::unique_lock<std::mutex> lock(_node_links_mutexes[node], std::defer_lock);
    if constexpr (!is_search_stage) {
      lock.lock();
    }

    node_id_t *neighbor_node_links = getNodeLinks(node);
#ifdef FLATNAV_PROFILE_VISITS
    // Count this node's graph-link access (the tiered/cached quantity).
    if (_node_visit_counts) _node_visit_counts[node].fetch_add(1, std::memory_order_relaxed);
#endif
#if defined(USE_SSE) && defined(FLATNAV_PF_VEC_BURST)
    // Exp B-burst (vector prefetch, depth-0, lightweight): prefetch the FIRST cache line of
    // each not-yet-visited neighbor's vector upfront, so the (remote/UPI) fetches overlap
    // the distance loop with MLP -- at a fraction of the instruction cost of the full-vector
    // version (1 line vs ~8; the HW streamer picks up the rest on demand). Skipping visited
    // neighbors avoids the prefetches/UPI traffic the distance loop below would just prune.
    for (uint32_t j = 0; j < _M; j++) {
      node_id_t nid = neighbor_node_links[j];
      if (!visited_set->isVisited(nid))
        _mm_prefetch(getNodeData(nid), _MM_HINT_T0);
    }
#endif
    query_visited_nodes_flags.push_back(_hub_nodes[node]);
#ifdef FLATNAV_PROFILE_TRAJ
    tl_traj.push_back(node);  // this node is being expanded now
#endif
#ifdef FLATNAV_PROFILE_PQ
    uint32_t pq_fanout = 0;  // new (unvisited) neighbors fetched this expansion
#endif
#ifdef FLATNAV_PQ_GATE
    // Invariant for this whole expansion -- tl_gate_step only advances in beamSearch's loop --
    // so it is computed once here rather than per neighbor. An empty window (hi <= lo) can
    // never satisfy both bounds, so it needs no separate off switch.
    const bool use_pq = tl_pq_lut && tl_gate_step >= (uint32_t)_pq_gate_lo &&
                        tl_gate_step < (uint32_t)_pq_gate_hi;
#endif
    for (uint32_t i = 0; i < _M; i++) {
      node_id_t neighbor_node_id = neighbor_node_links[i];

      // If using SSE, prefetch the next neighbor node data and the visited
      // marker
#if defined(USE_SSE) && !defined(FLATNAV_DISABLE_PREFETCH)
      if (i != _M - 1) {
        _mm_prefetch(getNodeData(neighbor_node_links[i + 1]), _MM_HINT_T0);
        visited_set->prefetch(neighbor_node_links[i + 1]);
      }
#endif

      bool neighbor_is_visited = visited_set->isVisited(/* num = */ neighbor_node_id);

      if (neighbor_is_visited) {
        continue;
      }
      visited_set->insert(/* num = */ neighbor_node_id);
#ifdef FLATNAV_PROFILE_PQ
      pq_fanout++;  // this neighbor is unvisited -> its vector is fetched/dist-computed below
#endif
#ifdef FLATNAV_PROFILE_VISITS
      // Count this neighbor's DATA (vector) access — the full activated footprint,
      // a superset of link-expanded nodes (these neighbors may never be expanded).
      if (_node_data_counts) _node_data_counts[neighbor_node_id].fetch_add(1, std::memory_order_relaxed);
#endif
      // Helper-thread staging: use the local staged copy of this neighbor's vector if it's
      // available (seqlock-validated), else the remote original. Pure latency hint.
      auto distFrom = [&](const char* v) {
        return _distance->distance(/* x = */ query, /* y = */ v, /* asymmetric = */ true);
      };
      float dist;
#ifdef FLATNAV_PQ_GATE
      // Inside the PQ phase this neighbor is scored from its code -- no vector read at all.
      (use_pq ? tl_gate_pq : tl_gate_exact)++;
      float pq_raw = 0.0f;  // the code-only score, kept for the trace when the gate overrides it
      if (use_pq) {
        dist = pq_raw = pqDistance(neighbor_node_id);
        // M15 quality gate: spend an exact vector read only where the PQ score is close enough to
        // the beam's K-th best that the true distance could change this node's admission. M14
        // killed the per-vector half of the design's gate (compressibility does not predict harm),
        // leaving this margin as the only surviving signal -- and it costs nothing to evaluate.
        // While the beam is still filling, max_dist is the entry node's distance, not a threshold:
        // every node is admitted regardless, so the "could change the top-K" test is trivially
        // true and the gate fetches (bounded by ~buffer_size reads per query).
        if (_pq_margin > 0.0f &&
            (neighbors.size() < (size_t)buffer_size ||
             (dist < max_dist * (1.0f + _pq_margin) &&
              (!_pq_margin_band || dist > max_dist * (1.0f - _pq_margin))))) {
          dist = distFrom(getNodeData(neighbor_node_id));
          tl_gate_pq--;
          tl_gate_exact++;  // this neighbor did read a vector after all
        }
      } else
#endif
      if (!(_pf_buf && _pf_buf->computeIfStaged(neighbor_node_id, distFrom, dist)))
        dist = distFrom(getNodeData(neighbor_node_id));
#ifdef FLATNAV_PROFILE_TRAJ
      if (dist < tl_term_dist) { tl_term_dist = dist; tl_term_node = neighbor_node_id; }
#endif
#ifdef FLATNAV_SPEC_TRACE
      // `dist` is whichever score drove this expansion; the other one is computed here as a
      // measurement only (neither read exists in a real run). Inside the PQ window that means an
      // extra vector read; outside it, an extra code lookup -- which is what lets the trace cover
      // the whole EXACT trajectory, not just a PQ-gated window.
      // Gate active -> in-window expansions only (M13's scout, unchanged). Gate off -> every
      // expansion of the exact run (M14's per-vector error along the exact trajectory).
      if (tl_spec_disc && tl_pq_lut && (use_pq || _pq_gate_hi <= _pq_gate_lo))
        tl_spec_disc->push_back({tl_gate_step, (uint32_t)neighbor_node_id,
                                 use_pq ? _distance->distance(/* x = */ query,
                                                              /* y = */ getNodeData(neighbor_node_id),
                                                              /* asymmetric = */ true)
                                        : dist,
                                 use_pq ? pq_raw : pqDistance(neighbor_node_id),
                                 neighbors.size() >= (size_t)buffer_size
                                     ? max_dist
                                     : std::numeric_limits<float>::max()});
#endif

      if (_collect_stats) {
        _distance_computations.fetch_add(1);
      }

      if (neighbors.size() < buffer_size || dist < max_dist) {
        candidates.emplace(-dist, neighbor_node_id);
        neighbors.emplace(dist, neighbor_node_id);
#ifdef FLATNAV_PROFILE_PQ
        // Node enters the PQ now (discovered during the current expansion).
        tl_disc[neighbor_node_id] = tl_step;
        tl_parent_res[neighbor_node_id] = tl_cur_residency;
#endif
        // query_visited_nodes_flags.push_back(_hub_nodes[neighbor_node_id]);
#if defined(USE_SSE) && !defined(FLATNAV_DISABLE_PREFETCH)
        _mm_prefetch(getNodeData(candidates.top().second), _MM_HINT_T0);
#endif
        if (neighbors.size() > buffer_size) {
          neighbors.pop();
        }
        if (!neighbors.empty()) {
          max_dist = neighbors.top().first;
        }
      }
    }
#ifdef FLATNAV_PROFILE_PQ
    // Bucket this expansion's fan-out by its step (tl_step, not yet incremented here).
    { uint32_t s = tl_step < kPQStepCap ? tl_step : kPQStepCap - 1;
      g_step_fanout[s].fetch_add(pq_fanout, std::memory_order_relaxed); }
#endif
  }

#ifdef FLATNAV_PQ_GATE
  /**
   * @brief Beam search that advances on PQ scores and validates with exact distances k steps
   * behind, so a node's vector is known to be needed k steps before it is read.
   *
   * The validated lane keeps the `neighbors`/`candidates` heaps and reproduces beamSearch's
   * pop/emplace sequence exactly -- same pop, same link order, same in-loop max_dist update --
   * so the returned beam is bit-identical to beamSearch's. The speculative lane reads only links
   * and PQ codes, choosing which node the validated lane will expand next; when that choice turns
   * out wrong, the in-flight steps are discarded and the validated lane's own top is expanded
   * instead.
   *
   * Only _spec_width == 1 is implemented. Larger widths currently widen the candidate scan but
   * still speculate on a single node per step.
   */
  PriorityQueue specBeamSearch(const void* query, const node_id_t entry_node, const int buffer_size) {
    const uint32_t ring_size = static_cast<uint32_t>(_spec_depth) + 1;

    PriorityQueue neighbors;   // +dist, top = furthest member of the beam
    PriorityQueue candidates;  // -dist, top = nearest pending node
    auto* visited_set = _visited_set_pool->pollAvailableSet();
    visited_set->clear();

    tl_spec_checks = tl_spec_hits = tl_spec_misses = tl_spec_miss_rejected = 0;
    tl_spec_miss_tie = tl_spec_miss_order = tl_spec_floor = 0;
    tl_spec_discarded = tl_spec_stalls = tl_spec_committed = tl_spec_wasted = 0;
    if (_spec_diag) {
      tl_spec_origin.clear();
      std::memset(tl_spec_depth_checks, 0, sizeof(tl_spec_depth_checks));
      std::memset(tl_spec_depth_misses, 0, sizeof(tl_spec_depth_misses));
    }

    float dist = _distance->distance(/* x = */ query, /* y = */ getNodeData(entry_node),
                                     /* asymmetric = */ true);
    tl_spec_committed++;
    float max_dist = dist;
    candidates.emplace(-dist, entry_node);
    neighbors.emplace(dist, entry_node);
    visited_set->insert(entry_node);

    std::vector<SpecStep> ring(ring_size);
    uint32_t head = 0, depth = 0, next_step_id = 0;
    std::vector<dist_node_t> top;  // heapTopR scratch

    auto slot = [&](uint32_t i) -> SpecStep& { return ring[(head + i) % ring_size]; };
    auto outstanding = [&](node_id_t n) {
      for (uint32_t i = 0; i < depth; i++)
        if (slot(i).node == n) return true;
      return false;
    };

    // Read `node`'s links, take and mark its unvisited neighbours, and admit them onto the
    // overlay. max_dist here is k steps stale and therefore looser than the threshold validation
    // will apply, so the overlay can hold nodes the beam later rejects.
    auto expand = [&](SpecStep& st, node_id_t node) {
      st.node = node;
      st.fresh.clear();
      st.overlay.clear();
      const node_id_t* links = getNodeLinks(node);
      for (uint32_t i = 0; i < _M; i++) {
        const node_id_t nbr = links[i];
        if (visited_set->isVisited(nbr)) continue;
        visited_set->insert(nbr);
        st.fresh.push_back(nbr);
      }
      const bool filling = neighbors.size() < static_cast<size_t>(buffer_size);
      for (const node_id_t nbr : st.fresh) {
        const float d = _spec_oracle ? _distance->distance(/* x = */ query,
                                                           /* y = */ getNodeData(nbr),
                                                           /* asymmetric = */ true)
                                     : pqDistance(nbr);
        const bool admit = filling || d < max_dist;
        if (admit) st.overlay.emplace_back(-d, nbr);
        if (_spec_diag) tl_spec_origin[nbr] = {st.id, admit};
      }
    };

    auto push = [&](node_id_t node) {
      SpecStep& st = slot(depth);
      st.id = next_step_id++;
      st.inflight_lo = depth > 0 ? slot(0).id : st.id;
      st.depth_at_push = depth;
      expand(st, node);
      depth++;
    };

    // Un-mark the nodes each in-flight step was the first to visit. Per-step fresh lists are
    // disjoint -- a node marked by one step is never unvisited for a later one -- so erasing them
    // in any order restores exactly the state before these steps ran.
    auto discardAll = [&]() {
      for (uint32_t i = 0; i < depth; i++) {
        SpecStep& st = slot(i);
        for (const node_id_t n : st.fresh) {
          visited_set->erase(n);
          if (_spec_diag) tl_spec_origin.erase(n);
        }
        tl_spec_wasted += static_cast<uint32_t>(st.fresh.size());
      }
      tl_spec_discarded += depth;
      depth = 0;
    };

    // The nearest pending candidate that no in-flight step has claimed, across both the beam's
    // queue and the overlay. Scores are negated throughout, so larger is nearer.
    auto pick = [&](node_id_t& out) {
      heapTopR(pqHeap(candidates), _spec_width + static_cast<int>(ring_size), outstanding, top);
      float best = 0.0f;
      bool found = false;
      if (!top.empty()) {
        best = top[0].first;
        out = top[0].second;
        found = true;
      }
      for (uint32_t i = 0; i < depth; i++)
        for (const dist_node_t& e : slot(i).overlay)
          if ((!found || e.first > best) && !outstanding(e.second)) {
            best = e.first;
            out = e.second;
            found = true;
          }
      if (found && neighbors.size() >= static_cast<size_t>(buffer_size) && -best > max_dist)
        found = false;
      return found;
    };

    bool finished = false;
    while (!finished) {
      bool produced = false;
      if (depth < ring_size) {
        node_id_t n;
        if (pick(n)) {
          push(n);
          produced = true;
        }
      }
      if (!produced) {
        // With nothing in flight, the pick fails on exactly beamSearch's loop guard: an empty
        // queue, or a full beam whose nearest pending candidate is already beyond max_dist.
        if (depth == 0) break;
        tl_spec_stalls++;
      }

      // Having added a step, validation only needs to catch up to depth k. Having added none it
      // drains the ring completely, or the loop would spin with work still in flight.
      const uint32_t drain_to = produced ? static_cast<uint32_t>(_spec_depth) : 0;
      while (depth > drain_to) {
        if (candidates.empty() ||
            (neighbors.size() >= static_cast<size_t>(buffer_size) &&
             -candidates.top().first > max_dist)) {
          discardAll();
          finished = true;
          break;
        }

        // The oldest in-flight step's node is candidates.top(), either by construction for the
        // first pick off an empty ring, or because the check at the bottom re-established it.
        SpecStep& st = slot(0);
        candidates.pop();
#ifdef FLATNAV_SPEC_TRACE
        if (tl_spec_expand) tl_spec_expand->push_back(st.node);
#endif
        for (const node_id_t nbr : st.fresh) {
          const float d = _distance->distance(/* x = */ query, /* y = */ getNodeData(nbr),
                                              /* asymmetric = */ true);
          tl_spec_committed++;
          if (neighbors.size() < static_cast<size_t>(buffer_size) || d < max_dist) {
            candidates.emplace(-d, nbr);
            neighbors.emplace(d, nbr);
            if (neighbors.size() > static_cast<size_t>(buffer_size)) neighbors.pop();
            if (!neighbors.empty()) max_dist = neighbors.top().first;
          }
        }

        head = (head + 1) % ring_size;
        depth--;

        if (depth == 0) continue;
        if (candidates.empty()) {
          discardAll();
          finished = true;
          break;
        }
        tl_spec_checks++;
        const node_id_t predicted = slot(0).node;
        const node_id_t truth = candidates.top().second;
        // Could this pick have found the true winner at any width? Only if the winner was visible
        // when the pick was made: already in the candidate queue, or on the overlay of a step that
        // was then in flight. A winner discovered by an in-flight step but scored off its overlay
        // was invisible, and no width recovers it.
        if (_spec_diag) {
          const auto it = tl_spec_origin.find(truth);
          if (it != tl_spec_origin.end() && it->second.step >= slot(0).inflight_lo &&
              !it->second.admitted)
            tl_spec_floor++;
        }
        const bool hit = (predicted == truth);
        if (_spec_diag) {
          const uint32_t d = std::min(slot(0).depth_at_push,
                                      static_cast<uint32_t>(kSpecDepthCap - 1));
          tl_spec_depth_checks[d]++;
          if (!hit) tl_spec_depth_misses[d]++;
        }
        if (hit) {
          tl_spec_hits++;
          continue;
        }
        tl_spec_misses++;
        // Where the predicted node actually ended up. A node absent from the queue was never
        // admitted, by this validation or an earlier one, so no ranking could have picked it; one
        // present at the same distance as the node expanded lost on heap order alone.
        if (_spec_diag) {
          const std::vector<dist_node_t>& heap = pqHeap(candidates);
          const float truth_dist = -candidates.top().first;
          bool present = false;
          float predicted_dist = 0.0f;
          for (const dist_node_t& e : heap)
            if (e.second == predicted) {
              present = true;
              predicted_dist = -e.first;
              break;
            }
          if (!present) tl_spec_miss_rejected++;
          else if (predicted_dist == truth_dist) tl_spec_miss_tie++;
          else tl_spec_miss_order++;
        }
        discardAll();
        push(truth);
      }
    }

    _visited_set_pool->pushVisitedSet(/* visited_set = */ visited_set);
    g_spec_checks.fetch_add(tl_spec_checks, std::memory_order_relaxed);
    g_spec_hits.fetch_add(tl_spec_hits, std::memory_order_relaxed);
    g_spec_misses.fetch_add(tl_spec_misses, std::memory_order_relaxed);
    g_spec_miss_rejected.fetch_add(tl_spec_miss_rejected, std::memory_order_relaxed);
    g_spec_miss_tie.fetch_add(tl_spec_miss_tie, std::memory_order_relaxed);
    g_spec_miss_order.fetch_add(tl_spec_miss_order, std::memory_order_relaxed);
    g_spec_floor.fetch_add(tl_spec_floor, std::memory_order_relaxed);
    if (_spec_diag)
      for (int i = 0; i < kSpecDepthCap; i++) {
        g_spec_depth_checks[i].fetch_add(tl_spec_depth_checks[i], std::memory_order_relaxed);
        g_spec_depth_misses[i].fetch_add(tl_spec_depth_misses[i], std::memory_order_relaxed);
      }
    g_spec_discarded.fetch_add(tl_spec_discarded, std::memory_order_relaxed);
    g_spec_stalls.fetch_add(tl_spec_stalls, std::memory_order_relaxed);
    g_spec_committed.fetch_add(tl_spec_committed, std::memory_order_relaxed);
    g_spec_wasted.fetch_add(tl_spec_wasted, std::memory_order_relaxed);
    return neighbors;
  }
#endif

  /**
   * @brief Selects neighbors from the PriorityQueue, according to the HNSW
   * heuristic. The neighbors priority queue contains elements sorted by
   * distance where the top element is the furthest neighbor from the query.
   */
  void selectNeighbors(PriorityQueue& neighbors, int M) {
    if (neighbors.size() < M) {
      return;
    }

    std::priority_queue<std::pair<float, node_id_t>> candidates;
    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(M);

    while (neighbors.size() > 0) {
      auto [distance, id] = neighbors.top();

      candidates.emplace(-distance, id);
      neighbors.pop();
    }

    while (candidates.size() > 0) {
      if (saved_candidates.size() >= M) {
        break;
      }
      // Extract the closest element from candidates.
      auto [distance_to_query, current_node_id] = candidates.top();
      distance_to_query = -distance_to_query;
      candidates.pop();

      bool should_keep_candidate = true;
      for (const auto& [_, second_pair_node_id] : saved_candidates) {
        float cur_dist = _distance->distance(/* x = */ getNodeData(second_pair_node_id),
                                       /* y = */ getNodeData(current_node_id));

        if (cur_dist < distance_to_query) {
          should_keep_candidate = false;
          break;
        }
      }
      if (should_keep_candidate) {
        // We could do neighbors.emplace except we have to iterate
        // through saved_candidates, and std::priority_queue doesn't
        // support iteration (there is no technical reason why not).
        auto current_pair = std::make_pair(-distance_to_query, current_node_id);
        saved_candidates.push_back(current_pair);
      }
    }
    // TODO: implement my own priority queue, get rid of vector
    // saved_candidates, add directly to neighborqueue earlier.
    for (const dist_node_t& current_pair : saved_candidates) {
      neighbors.emplace(-current_pair.first, current_pair.second);
    }

  }

  void connectNeighbors(PriorityQueue& neighbors, node_id_t new_node_id) {
    // connects neighbors according to the HSNW heuristic

    // Lock all operations on this node
    std::unique_lock<std::mutex> lock(_node_links_mutexes[new_node_id]);

    node_id_t* new_node_links = getNodeLinks(new_node_id);
    int i = 0;  // iterates through links for "new_node_id"

    while (neighbors.size() > 0) {
      node_id_t neighbor_node_id = neighbors.top().second;
      // add link to the current new node
      new_node_links[i] = neighbor_node_id;
      // now do the back-connections (a little tricky)

      std::unique_lock<std::mutex> neighbor_lock(_node_links_mutexes[neighbor_node_id]);
      node_id_t* neighbor_node_links = getNodeLinks(neighbor_node_id);
      bool is_inserted = false;
      for (size_t j = 0; j < _M; j++) {
        if (neighbor_node_links[j] == neighbor_node_id) {
          // If there is a self-loop, replace the self-loop with
          // the desired link.
          neighbor_node_links[j] = new_node_id;
          is_inserted = true;
          break;
        }
      }
      if (!is_inserted) {
        // now, we may to replace one of the links. This will disconnect
        // the old neighbor and create a directed edge, so we have to be
        // very careful. To ensure we respect the pruning heuristic, we
        // construct a candidate set including the old links AND our new
        // one, then prune this candidate set to get the new neighbors.

        float max_dist = _distance->distance(/* x = */ getNodeData(neighbor_node_id),
                                             /* y = */ getNodeData(new_node_id));

        PriorityQueue candidates;
        candidates.emplace(max_dist, new_node_id);
        for (size_t j = 0; j < _M; j++) {
          if (neighbor_node_links[j] != neighbor_node_id) {
            auto label = neighbor_node_links[j];
            auto distance = _distance->distance(/* x = */ getNodeData(neighbor_node_id),
                                                /* y = */ getNodeData(label));
            candidates.emplace(distance, label);
          }
        }
        // 2X larger than the previous call to selectNeighbors.
        selectNeighbors(candidates, _M);
        // connect the pruned set of candidates, including self-loops:
        size_t j = 0;
        while (candidates.size() > 0) {  // candidates
          neighbor_node_links[j] = candidates.top().second;
          candidates.pop();
          j++;
        }
        while (j < _M) {  // self-loops (unused links)
          neighbor_node_links[j] = neighbor_node_id;
          j++;
        }
      }

      // Unlock the current node we are iterating over
      neighbor_lock.unlock();

      // loop increments:
      i++;
      neighbors.pop();
    }
  }

  /**
   * @brief Selects a node to use as the entry point for a new node.
   * This proceeds in a greedy fashion, by selecting the node with
   * the smallest distance to the query.
   *
   * @param query
   * @param num_initializations
   * @return node_id_t
   */
  inline node_id_t initializeSearch(const void* query, int num_initializations) {
    // select entry_node from a set of random entry point options
    if (num_initializations <= 0) {
      throw std::invalid_argument("num_initializations must be greater than 0.");
    }

    int step_size = _cur_num_nodes / num_initializations;
    step_size = step_size ? step_size : 1;

    float min_dist = std::numeric_limits<float>::max();
    node_id_t entry_node = 0;

    if (_collect_stats) {
      _distance_computations.fetch_add(num_initializations);
    }

    for (node_id_t node = 0; node < _cur_num_nodes; node += step_size) {
      float dist = _distance->distance(/* x = */ query, /* y = */ getNodeData(node),
                                       /* asymmetric = */ true);
      if (dist < min_dist) {
        min_dist = dist;
        entry_node = node;
      }
    }
    return entry_node;
  }

  // Use this during search to select a random entry point
  node_id_t randomlyInitializeSearch(const void *query,
                                     int num_initializations) {
    // select entry_node from a set of random entry point options
    if (num_initializations <= 0) {
      throw std::invalid_argument(
          "num_initializations must be greater than 0.");
    }

    float min_dist = std::numeric_limits<float>::max();
    node_id_t entry_node = 0;

    if (_collect_stats) {
      _distance_computations.fetch_add(num_initializations);
    }

    for (int i = 0; i < num_initializations; i++) {
      node_id_t node = _distribution(_generator);
      float dist =
          _distance->distance(/* x = */ query, /* y = */ getNodeData(node),
                              /* asymmetric = */ true);
      if (dist < min_dist) {
        min_dist = dist;
        entry_node = node;
      }
    }
    return entry_node;
  }

  void relabel(const std::vector<node_id_t> &P) {
    // 1. Rewire all of the node connections
    for (node_id_t n = 0; n < _cur_num_nodes; n++) {
      node_id_t* links = getNodeLinks(n);
      for (int m = 0; m < _M; m++) {
        links[m] = P[links[m]];
      }
    }

    // 2. Physically re-layout the nodes (in place)
    char* temp_data = new char[_data_size_bytes];
    node_id_t* temp_links = new node_id_t[_M];
    label_t* temp_label = new label_t;

    auto* visited_set = _visited_set_pool->pollAvailableSet();

    // In this context, is_visited stores which nodes have been relocated
    // (it would be equivalent to name this variable "is_relocated").
    visited_set->clear();

    for (node_id_t n = 0; n < _cur_num_nodes; n++) {
      if (visited_set->isVisited(/* num = */ n)) {
        continue;
      }

      node_id_t src = n;
      node_id_t dest = P[src];

      // swap node at src with node at dest
      swapNodes(src, dest, temp_data, temp_links, temp_label);

      // mark src as having been relocated
      visited_set->insert(src);

      // recursively relocate the node from "dest"
      while (!visited_set->isVisited(/* num = */ dest)) {
        // mark node as having been relocated
        visited_set->insert(dest);
        // the value of src remains the same. However, dest needs
        // to change because the node located at src was previously
        // located at dest, and must be relocated to P[dest].
        dest = P[dest];

        // swap node at src with node at dest
        swapNodes(src, dest, temp_data, temp_links, temp_label);
      }
    }

    _visited_set_pool->pushVisitedSet(
        /* visited_set = */ visited_set);

    delete[] temp_data;
    delete[] temp_links;
    delete temp_label;
  }
}; // namespace flatnav

}  // namespace flatnav
