#pragma once

// Validation offload: run the exact-distance half of specBeamSearch on cores that sit next to
// the vectors rather than next to the search. The speculative lane (links, PQ codes, the beam
// heaps) stays on the near node at full clock; only "read these vectors, return these distances"
// crosses, so a step moves ~20 ids out and ~20 floats back instead of ~20 512-byte vectors.
//
// Flow:  search thread finishes expand() and knows the step's fresh neighbour ids k steps before
//        validation needs them -> post() hands the batch to its lane -> a far worker computes the
//        distances into the step's own array -> the search thread wait()s for that batch when the
//        drain loop reaches it, then does the heap merge itself.
//
// The merge stays near on purpose: it is sequential and order-sensitive (max_dist is updated
// inside the loop), and keeping it there is what leaves the returned beam bit-identical to
// beamSearch. Distances come back in `fresh` order, so the offload is invisible to the result.
//
// Each search thread owns one lane and each far worker drains exactly one lane, so the rings are
// strictly single-producer / single-consumer and carry NO atomic read-modify-write on the hot
// path -- only a release store to post and an acquire load to observe. This is deliberate: the
// staging experiment that preceded this funnelled every neighbour through one shared MPMC head
// and paid a contended CAS 723M times. Here the handoff is per STEP (~207/query, not ~4134) and
// never contended. Lanes must therefore outnumber search threads; a thread that cannot claim one
// falls back to computing distances inline.

#include <immintrin.h>

#include <atomic>
#include <cstdint>
#include <vector>

namespace flatnav {

// One batch: a speculative step's fresh neighbours, and where their distances go.
struct OffloadJob {
  const void* query = nullptr;
  const uint32_t* ids = nullptr;
  float* out = nullptr;
  uint32_t n = 0;
};

class OffloadLane {
 public:
  static constexpr uint32_t kNoJob = 0xFFFFFFFFu;

  // capacity_pow2 must exceed the deepest ring the search will run (k + 1 outstanding batches).
  explicit OffloadLane(uint32_t capacity_pow2)
      : _mask(capacity_pow2 - 1), _job(capacity_pow2), _state(capacity_pow2) {
    for (uint32_t i = 0; i < capacity_pow2; i++) _state[i].v.store(kFree, std::memory_order_relaxed);
  }

  // --- search thread (producer) ---

  // Hand off one batch. `ids` and `out` must stay alive and unmoved until wait() returns.
  uint32_t post(const void* query, const uint32_t* ids, float* out, uint32_t n) {
    const uint32_t s = _head++ & _mask;
    _job[s] = {query, ids, out, n};
    _state[s].v.store(kPosted, std::memory_order_release);
    return s;
  }

  // Block until slot `s` holds its distances, then release it. Batches are waited for in the
  // order they were posted, which is also the order the far worker executes them.
  void wait(uint32_t s) {
    while (_state[s].v.load(std::memory_order_acquire) != kDone) _mm_pause();
    _state[s].v.store(kFree, std::memory_order_relaxed);
  }

  // --- far worker (consumer) ---

  // Execute the oldest posted batch, if there is one. Returns false when the lane is idle.
  template <typename ComputeFn>
  bool runOnce(ComputeFn&& compute) {
    const uint32_t s = _tail & _mask;
    if (_state[s].v.load(std::memory_order_acquire) != kPosted) return false;
    compute(_job[s]);
    _state[s].v.store(kDone, std::memory_order_release);
    _tail++;
    return true;
  }

 private:
  static constexpr uint32_t kFree = 0, kPosted = 1, kDone = 2;
  struct alignas(64) Slot { std::atomic<uint32_t> v; };

  uint32_t _mask;
  std::vector<OffloadJob> _job;
  std::vector<Slot> _state;
  alignas(64) uint32_t _head = 0;  // producer-local
  alignas(64) uint32_t _tail = 0;  // consumer-local
};

// One lane per far worker. A search thread claims a lane on first use and keeps it.
class OffloadPool {
 public:
  OffloadPool(uint32_t lanes, uint32_t capacity_pow2) {
    _lanes.reserve(lanes);
    for (uint32_t i = 0; i < lanes; i++) _lanes.emplace_back(new OffloadLane(capacity_pow2));
  }
  ~OffloadPool() { for (OffloadLane* l : _lanes) delete l; }

  uint32_t numLanes() const { return static_cast<uint32_t>(_lanes.size()); }
  OffloadLane& lane(uint32_t i) { return *_lanes[i]; }

  // The calling search thread's lane, or nullptr when every lane is already taken.
  OffloadLane* claim() {
    thread_local int id = _next.fetch_add(1, std::memory_order_relaxed);
    return id < static_cast<int>(_lanes.size()) ? _lanes[id] : nullptr;
  }

 private:
  std::vector<OffloadLane*> _lanes;
  std::atomic<int> _next{0};
};

}  // namespace flatnav
