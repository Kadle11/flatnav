#pragma once

// Helper-thread staging: a shared, bounded, local-DRAM buffer of node vectors, populated
// by a small pool of helper threads and read by the search workers. It converts remote
// (cross-NUMA) vector reads into local reads for the fraction of accesses the helpers
// manage to stage in time. It is a pure LATENCY HINT -- every staged value is validated by
// a per-slot seqlock and the worker falls back to the (remote) original on any miss/race,
// so search results are unaffected (recall must stay flat).
//
// Flow:  worker (at each pop) enqueues the top-K candidate ids into an MPMC ring ->
//        a helper dequeues an id, reads that candidate's links and copies its neighbors'
//        vectors remote->buffer ->  worker (at the distance comp) reads the local copy if
//        the neighbor is staged, else remote.
//
// The buffer is DIRECT-MAPPED (slot = id % num_slots). Eviction is implicit: staging a node
// that maps to an occupied slot overwrites (evicts) the previous occupant. Expected
// residency ~ num_slots / staging_rate, so num_slots is the "limited buffer" window knob
// (bigger = longer residency = can stage earlier). Ring and buffer are both lossy -- a
// dropped request or a lost slot race just costs a remote read.

#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

#include <flatnav/util/NumaAllocation.h>

namespace flatnav {

// ---- Vyukov bounded MPMC queue of node ids (many producers / few consumers, lock-free) ----
class MPMCRing {
 public:
  explicit MPMCRing(size_t capacity_pow2) : _mask(capacity_pow2 - 1), _buf(capacity_pow2) {
    for (size_t i = 0; i < capacity_pow2; i++)
      _buf[i].seq.store(i, std::memory_order_relaxed);
    _enq.store(0, std::memory_order_relaxed);
    _deq.store(0, std::memory_order_relaxed);
  }

  bool enqueue(uint32_t v) {
    Cell* cell;
    size_t pos = _enq.load(std::memory_order_relaxed);
    for (;;) {
      cell = &_buf[pos & _mask];
      size_t seq = cell->seq.load(std::memory_order_acquire);
      intptr_t diff = (intptr_t)seq - (intptr_t)pos;
      if (diff == 0) {
        if (_enq.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) break;
      } else if (diff < 0) {
        return false;  // full
      } else {
        pos = _enq.load(std::memory_order_relaxed);
      }
    }
    cell->val = v;
    cell->seq.store(pos + 1, std::memory_order_release);
    return true;
  }

  bool dequeue(uint32_t& v) {
    Cell* cell;
    size_t pos = _deq.load(std::memory_order_relaxed);
    for (;;) {
      cell = &_buf[pos & _mask];
      size_t seq = cell->seq.load(std::memory_order_acquire);
      intptr_t diff = (intptr_t)seq - (intptr_t)(pos + 1);
      if (diff == 0) {
        if (_deq.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) break;
      } else if (diff < 0) {
        return false;  // empty
      } else {
        pos = _deq.load(std::memory_order_relaxed);
      }
    }
    v = cell->val;
    cell->seq.store(pos + _mask + 1, std::memory_order_release);
    return true;
  }

 private:
  struct Cell {
    std::atomic<size_t> seq;
    uint32_t val;
  };
  size_t _mask;
  std::vector<Cell> _buf;
  alignas(64) std::atomic<size_t> _enq;
  alignas(64) std::atomic<size_t> _deq;
};

// ---- Direct-mapped local-DRAM staging buffer with per-slot seqlock ----
class StagingBuffer {
 public:
  static constexpr uint32_t kEmpty = 0xFFFFFFFFu;

  // num_slots: buffer capacity. vec_bytes: one vector. local_node: NUMA node to place the
  // buffer on (workers read it locally).
  StagingBuffer(size_t num_slots, size_t vec_bytes, int local_node)
      : _n(num_slots), _vb(vec_bytes), _node(local_node) {
    _ver = new std::atomic<uint32_t>[_n];
    _tag = new std::atomic<uint32_t>[_n];
    for (size_t i = 0; i < _n; i++) {
      _ver[i].store(0, std::memory_order_relaxed);
      _tag[i].store(kEmpty, std::memory_order_relaxed);
    }
    _data = util::allocateBytes(_n * _vb, local_node);  // local DRAM
    std::memset(_data, 0, _n * _vb);                    // first-touch local
    _hits = new Ctr[kShards];
    _uses = new Ctr[kShards];
  }
  ~StagingBuffer() {
    delete[] _ver;
    delete[] _tag;
    delete[] _hits;
    delete[] _uses;
    util::freeBytes(_data, _n * _vb, _node);
  }

  // Helper: copy `src` (the remote vector of node `id`) into id's slot, under the seqlock.
  // Serialized per slot by the version CAS; lossy (skips if already staged or contended).
  // Overwriting a different occupant is the (implicit) eviction.
  void put(uint32_t id, const char* src) {
    size_t s = id % _n;
    if (_tag[s].load(std::memory_order_relaxed) == id) return;  // already staged
    uint32_t v = _ver[s].load(std::memory_order_relaxed);
    if (v & 1u) return;  // another helper is writing this slot
    if (!_ver[s].compare_exchange_strong(v, v + 1, std::memory_order_acquire)) return;
    _tag[s].store(kEmpty, std::memory_order_relaxed);   // invalid while writing
    std::memcpy(_data + s * _vb, src, _vb);
    _tag[s].store(id, std::memory_order_relaxed);
    _ver[s].store(v + 2, std::memory_order_release);    // even -> stable
  }

  // Worker: if node `id` is staged, compute the distance from the LOCAL copy and return
  // true; else false (caller uses the remote original). Seqlock-validated so a concurrent
  // helper write is never observed as a valid result.
  template <typename DistFn>
  bool computeIfStaged(uint32_t id, DistFn&& fn, float& out) {
    int sh = shardId();
    _uses[sh].v.fetch_add(1, std::memory_order_relaxed);
    size_t s = id % _n;
    uint32_t v1 = _ver[s].load(std::memory_order_acquire);
    if (v1 & 1u) return false;                                        // mid-write
    if (_tag[s].load(std::memory_order_acquire) != id) return false;  // miss
    float d = fn(_data + s * _vb);
    uint32_t v2 = _ver[s].load(std::memory_order_acquire);
    if (v1 != v2) return false;                                       // raced -> remote
    out = d;
    _hits[sh].v.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  uint64_t hits() const { return sum(_hits); }
  uint64_t uses() const { return sum(_uses); }
  void resetStats() {
    for (int i = 0; i < kShards; i++) {
      _hits[i].v.store(0, std::memory_order_relaxed);
      _uses[i].v.store(0, std::memory_order_relaxed);
    }
  }
  size_t numSlots() const { return _n; }

 private:
  // Per-thread sharded counters: each worker hits its own cacheline, so the hot-path
  // hit/use counting adds no cross-thread contention that would distort the QPS measurement.
  static constexpr int kShards = 64;
  struct alignas(64) Ctr { std::atomic<uint64_t> v{0}; };
  static int shardId() {
    static std::atomic<int> next{0};
    thread_local int id = next.fetch_add(1, std::memory_order_relaxed) & (kShards - 1);
    return id;
  }
  static uint64_t sum(const Ctr* c) {
    uint64_t t = 0;
    for (int i = 0; i < kShards; i++) t += c[i].v.load(std::memory_order_relaxed);
    return t;
  }

  size_t _n, _vb;
  int _node;
  std::atomic<uint32_t>* _ver;
  std::atomic<uint32_t>* _tag;
  char* _data;
  Ctr* _hits;
  Ctr* _uses;
};

}  // namespace flatnav
