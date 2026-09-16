// roofline_probe -- is the distance kernel compute-bound or memory-bound?
//
// Sweeps the working-set size N (number of DB vectors) for flatnav's real
// SquaredL2Distance kernel, single-threaded, no graph. For each N, repeatedly
// computes distance(query, db[random_id % N]) for a fixed duration and reports
// throughput (vectors/sec, GB/s of db reads, GFLOP/s).
//
// Design invariants:
//   1. ids come from a PRNG, not a scan -> defeats hardware stride prefetchers
//      once N is large, so large-N throughput reflects real random-access DRAM
//      bandwidth, not sequential streaming bandwidth.
//   2. query q is reused every iteration (stays hot in L1) -> the only cold
//      read per iteration is the db vector itself (dim*4 bytes for float32).
//   3. uses flatnav::distances::SquaredL2Distance::distanceImpl directly (the
//      same SIMD dispatch the real index uses), not a hand-rolled loop.
//
// Reading the output: throughput should be flat (compute-bound, in-cache) for
// N small enough that N*dim*4 bytes fits in L2/L3, then fall and re-flatten at
// a lower plateau once N*dim*4 >> LLC (memory-bound, DRAM-latency/BW limited).
// The N where it breaks is the L1/L2/L3 crossover on this CPU.
//
//   args: dim N_min N_max dur_sec_per_point
//
// build: g++ -O3 -march=native -I include -I external/cereal/include
//          tools/roofline_probe.cpp -o roofline_probe

#include <flatnav/distances/SquaredL2Distance.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

using flatnav::distances::SquaredL2Distance;
using flatnav::util::DataType;
using dist_t = SquaredL2Distance<DataType::float32>;

static inline uint64_t sm64(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static inline double now_sec() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
  const int dim      = argc > 1 ? atoi(argv[1]) : 128;
  const size_t n_min = argc > 2 ? strtoull(argv[2], nullptr, 10) : 1ULL << 10;
  const size_t n_max = argc > 3 ? strtoull(argv[3], nullptr, 10) : 1ULL << 24;
  const double dur   = argc > 4 ? atof(argv[4]) : 2.0;

  dist_t dist(dim);
  const size_t stride = (size_t)dim * sizeof(float);

  float q[512];
  for (int d = 0; d < dim; d++) q[d] = (float)d;

  printf("# N vectors_per_sec GBps GFLOPps bytes\n");
  for (size_t N = n_min; N <= n_max; N *= 2) {
    std::vector<float> db(N * dim);
    memset(db.data(), 1, db.size() * sizeof(float));  // fault every page in now

    uint64_t rng = 0xD1CE5EEDULL;
    volatile float sink = 0.f;
    uint64_t cnt = 0;
    const double t0 = now_sec();
    while (now_sec() - t0 < dur) {
      // batch of 256 iterations between clock checks to keep now_sec() off the
      // hot path
      for (int b = 0; b < 256; b++) {
        size_t id = sm64(rng) % N;
        sink = dist.distanceImpl(q, db.data() + id * dim);
        cnt++;
      }
    }
    const double elapsed = now_sec() - t0;
    const double vecs_per_sec = cnt / elapsed;
    const double gbps = vecs_per_sec * stride / 1e9;
    const double gflopps = vecs_per_sec * (2.0 * dim) / 1e9;  // sub + fma per dim
    printf("%zu %.0f %.3f %.3f %zu\n", N, vecs_per_sec, gbps, gflopps, N * stride);
    fflush(stdout);
    (void)sink;
  }
  return 0;
}
