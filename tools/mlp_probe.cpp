// MLP probe -- how far does per-core memory-level parallelism carry a remote,
// latency-bound gather? Strips away the graph and the beam search: T threads on
// the compute node read random fixed-stride vectors from a (remote) NUMA node and
// run an L2-distance reduction, with a knob K = how many independent vectors are
// issued in flight per group. Sweep K and watch throughput while perf watches
// util / mem_stall. If the cores are MLP-starved, throughput rises and mem_stall
// falls as K grows, saturating near the line-fill-buffer limit (~12 on Skylake).
//
// Design invariants (see scripts/mlp_probe.sh for the sweep):
//   1. working set >> L3  -> every access is a real DRAM/UPI miss, not an L3 hit.
//   2. data on --data-node, threads on --cpu-node -> reproduce the remote path.
//   3. the K ids come from a PRNG, never from loaded data -> the K loads are
//      mutually independent (a dependent chain would collapse MLP to 1).
//   4. issue (prefetch) is decoupled from consume (reduce) so K first-lines are
//      actually outstanding together; pure demand loads are ROB-limited.
//
//   args: N_vectors dim K prefetch_lines T data_node cpu_node dur_sec
//
// build: see scripts/mlp_probe.sh  (g++ -O3 -march=native -DFLATNAV_USE_NUMA ... -lnuma)

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <x86intrin.h>

#include <flatnav/util/NumaAllocation.h>

// splitmix64: fast PRNG, ids independent of any loaded value (invariant 3).
static inline uint64_t sm64(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// One vector = dim floats (128 -> 512 B = 8 lines). Reading all of it keeps the
// per-access byte traffic representative of the real distance computation.
static inline float sqdist(const float* q, const float* v, int dim) {
  float acc = 0.f;
  for (int d = 0; d < dim; d++) {
    float t = q[d] - v[d];
    acc += t * t;
  }
  return acc;  // -O3 -march=native autovectorizes; touches every line of v
}

static inline double now_sec() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
  const size_t N    = argc > 1 ? strtoull(argv[1], nullptr, 10) : (4ULL << 30) / 512;  // ~4 GB
  const int    dim  = argc > 2 ? atoi(argv[2]) : 128;
  const int    K    = argc > 3 ? atoi(argv[3]) : 8;    // <-- the swept knob (in-flight vectors)
  const int    PFL  = argc > 4 ? atoi(argv[4]) : 1;    // first-lines prefetched/vec (0 = pure demand)
  const int    T    = argc > 5 ? atoi(argv[5]) : 16;   // start at 1/phys-core to isolate from SMT
  const int    dnode= argc > 6 ? atoi(argv[6]) : 1;    // data node (remote)
  const int    cnode= argc > 7 ? atoi(argv[7]) : 0;    // cpu node (compute)
  const double dur  = argc > 8 ? atof(argv[8]) : 5.0;

  if (dim > 512 || K > 64) { fprintf(stderr, "dim<=512, K<=64\n"); return 2; }
  const size_t stride = (size_t)dim * sizeof(float);

  // Invariant 2: bind the array to the remote node. numa_alloc_onnode sets an
  // MPOL_BIND policy, so the pages land on dnode no matter which CPU faults them.
  char* data = flatnav::util::allocateBytes(N * stride, dnode);
  if (!data) { fprintf(stderr, "alloc failed\n"); return 1; }
  memset(data, 1, N * stride);  // fault every page in now (nonzero, no NaNs)

  std::atomic<uint64_t> total{0};
  std::atomic<uint64_t> sink{0};  // defeat dead-code elimination of the reductions (bit-XOR)

  auto worker = [&](int tid) {
    // node0 = even CPUs, node1 = odd CPUs; cnode+2*tid walks that node's CPUs,
    // filling physical primaries first (tid 0..15) then SMT siblings (16..31).
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cnode + 2 * tid, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);

    uint64_t rng = 0xD1CE5EEDULL ^ (0x1000ULL * (uint64_t)tid);
    float q[512];
    for (int d = 0; d < dim; d++) q[d] = (float)d;

    uint32_t id[64];
    double local = 0.0;
    uint64_t cnt = 0;
    const double t0 = now_sec();
    while (now_sec() - t0 < dur) {
      for (int k = 0; k < K; k++) id[k] = (uint32_t)(sm64(rng) % N);

      // ISSUE: put K vectors' first line(s) in flight together  -> MLP = K.
      for (int k = 0; k < K; k++) {
        const char* base = data + (size_t)id[k] * stride;
        for (int l = 0; l < PFL; l++) _mm_prefetch(base + l * 64, _MM_HINT_T0);
      }
      // CONSUME: K independent reductions (each reads the full 512 B vector).
      for (int k = 0; k < K; k++)
        local += sqdist(q, (const float*)(data + (size_t)id[k] * stride), dim);

      cnt += K;
    }
    total.fetch_add(cnt);
    uint64_t bits;
    memcpy(&bits, &local, sizeof(bits));
    sink.fetch_xor(bits, std::memory_order_relaxed);
  };

  const double t0 = now_sec();
  std::vector<std::thread> ths;
  for (int i = 0; i < T; i++) ths.emplace_back(worker, i);
  for (auto& t : ths) t.join();
  const double elapsed = now_sec() - t0;

  const double macc = total.load() / elapsed / 1e6;
  // CSV-friendly: K,PFL,T,Macc/s,Macc/s-per-thread,GB/s(512B/acc),sink
  printf("K=%d PFL=%d T=%d  %.1f Macc/s  %.2f /thread  %.1f GB/s  sink=%llx\n",
         K, PFL, T, macc, macc / T, macc * stride / 1e3,
         (unsigned long long)sink.load());

  flatnav::util::freeBytes(data, N * stride, dnode);
  return 0;
}
