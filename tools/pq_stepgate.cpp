// Step-windowed PQ traversal: recall@K vs the window [lo,hi) of expansions ranked on PQ
// (compressed, no vector read). Expansions outside the window use exact distances.
//
//   pq_stepgate <index.bin> <query.fvecs> <gt.ivecs> [K=100] [threads=32] [ef=200]
//
// env:
//   PQ_M=16        subquantizers = code bytes per vector (dim must be divisible by it)
//   TRAIN=200000   nodes sampled to train the codebooks
//   ITERS=25       k-means iterations per subspace
//   WINDOWS=...    comma-separated lo:hi windows to sweep. hi<=lo (e.g. 0:0) is the exact
//                  baseline; 0:N is the M11 prefix gate; 0:1000000 is the all-PQ floor.
//   NQ=...         cap on the number of queries
//
// M11 measured prefix windows [0,N) and refuted the design's "exactness is only needed LATE"
// claim. Sliding a FIXED-WIDTH window across the search instead isolates which phase actually
// needs exactness: equal PQ volume at every offset, so the recall differences are phase, not
// dose. At the window's end the beam and candidate queue are rescored with exact distances
// (Index.h, FLATNAV_PQ_GATE) -- including nodes scored exactly before the window, which is
// idempotent but costs a re-read; if the search ends inside the window the beam is rescored
// before returning, so every reported recall is a top-K by EXACT distance.
#define FLATNAV_PQ_GATE
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/index/Index.h>
#include <flatnav/util/Multithreading.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

using flatnav::Index;
using flatnav::distances::SquaredL2Distance;
using flatnav::util::DataType;
using dist_t = SquaredL2Distance<DataType::float32>;
using clk = std::chrono::steady_clock;

static const int kCentroids = 256;  // one code byte per subquantizer

static std::vector<float> readFvecs(const char* path, int& dim, size_t& n) {
  int fd = open(path, O_RDONLY); if (fd < 0) { perror("open fvecs"); exit(1); }
  struct stat st; fstat(fd, &st); size_t fsize = st.st_size;
  void* map = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) { perror("mmap fvecs"); exit(1); }
  const char* base = static_cast<const char*>(map);
  dim = *reinterpret_cast<const int32_t*>(base);
  size_t rec = 4 + (size_t)dim * 4; n = fsize / rec;
  std::vector<float> out(n * dim);
  for (size_t i = 0; i < n; i++) memcpy(&out[i*dim], base + i*rec + 4, dim*4);
  munmap(map, fsize); close(fd); return out;
}
static std::vector<int> readIvecs(const char* path, int& w, size_t& n) {
  int fd = open(path, O_RDONLY); if (fd < 0) { perror("open ivecs"); exit(1); }
  struct stat st; fstat(fd, &st); size_t fsize = st.st_size;
  void* map = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) { perror("mmap ivecs"); exit(1); }
  const char* base = static_cast<const char*>(map);
  w = *reinterpret_cast<const int32_t*>(base);
  size_t rec = 4 + (size_t)w * 4; n = fsize / rec;
  std::vector<int> out(n * w);
  for (size_t i = 0; i < n; i++) memcpy(&out[i*w], base + i*rec + 4, w*4);
  munmap(map, fsize); close(fd); return out;
}

static inline float l2(const float* a, const float* b, int d) {
  float s = 0.0f;
  for (int i = 0; i < d; i++) { float t = a[i] - b[i]; s += t * t; }
  return s;
}

// Nearest centroid of one subspace codebook (kCentroids x sub_dim, contiguous).
static inline int nearestCentroid(const float* sub, const float* codebook, int sub_dim) {
  float best = FLT_MAX; int bc = 0;
  for (int c = 0; c < kCentroids; c++) {
    float d = l2(sub, codebook + (size_t)c * sub_dim, sub_dim);
    if (d < best) { best = d; bc = c; }
  }
  return bc;
}

// Lloyd's k-means on one subspace. `train` is n x sub_dim contiguous; `codebook` is the
// output (kCentroids x sub_dim). Empty clusters are reseeded on a random training point.
static void trainSubspace(const float* train, size_t n, int sub_dim, int iters,
                          float* codebook, unsigned seed) {
  std::mt19937 rng(seed);
  std::vector<size_t> perm(n);
  std::iota(perm.begin(), perm.end(), (size_t)0);
  std::shuffle(perm.begin(), perm.end(), rng);
  for (int c = 0; c < kCentroids; c++)
    memcpy(codebook + (size_t)c * sub_dim, train + perm[c] * sub_dim, sub_dim * sizeof(float));

  std::vector<uint8_t> assign(n);
  std::vector<double> sums((size_t)kCentroids * sub_dim);
  std::vector<uint32_t> counts(kCentroids);
  for (int it = 0; it < iters; it++) {
    for (size_t i = 0; i < n; i++)
      assign[i] = (uint8_t)nearestCentroid(train + i * sub_dim, codebook, sub_dim);

    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(counts.begin(), counts.end(), 0u);
    for (size_t i = 0; i < n; i++) {
      int c = assign[i];
      counts[c]++;
      for (int k = 0; k < sub_dim; k++) sums[(size_t)c * sub_dim + k] += train[i * sub_dim + k];
    }
    for (int c = 0; c < kCentroids; c++) {
      if (counts[c] == 0) {
        memcpy(codebook + (size_t)c * sub_dim, train + (rng() % n) * sub_dim, sub_dim * sizeof(float));
        continue;
      }
      for (int k = 0; k < sub_dim; k++)
        codebook[(size_t)c * sub_dim + k] = (float)(sums[(size_t)c * sub_dim + k] / counts[c]);
    }
  }
}

// Query LUT consumed by Index::pqDistance: lut[j*256 + c] = |q_j - centroid_{j,c}|^2.
static void buildLUT(const float* q, const float* codebooks, int m, int sub_dim, float* lut) {
  for (int j = 0; j < m; j++) {
    const float* sub = q + (size_t)j * sub_dim;
    const float* cb = codebooks + (size_t)j * kCentroids * sub_dim;
    for (int c = 0; c < kCentroids; c++)
      lut[(size_t)j * kCentroids + c] = l2(sub, cb + (size_t)c * sub_dim, sub_dim);
  }
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <index.bin> <query.fvecs> <gt.ivecs> [K=100] [threads=32] [ef=200]\n", argv[0]);
    return 1;
  }
  const int K = argc > 4 ? atoi(argv[4]) : 100;
  const int threads = argc > 5 ? atoi(argv[5]) : 32;
  const int ef = argc > 6 ? atoi(argv[6]) : 200;
  const int m = getenv("PQ_M") ? atoi(getenv("PQ_M")) : 16;
  const size_t n_train = getenv("TRAIN") ? (size_t)atoll(getenv("TRAIN")) : 200000;
  const int iters = getenv("ITERS") ? atoi(getenv("ITERS")) : 25;
  std::string win_str = getenv("WINDOWS")
                            ? getenv("WINDOWS")
                            : "0:0,0:40,40:80,80:120,120:160,160:200,0:1000000";
  std::vector<std::pair<int,int>> windows;
  for (size_t p = 0; p < win_str.size(); ) {
    size_t c = win_str.find(',', p);
    std::string tok = win_str.substr(p, c == std::string::npos ? c : c - p);
    if (!tok.empty()) {
      size_t colon = tok.find(':');
      if (colon == std::string::npos) {
        fprintf(stderr, "bad window '%s' (expected lo:hi)\n", tok.c_str());
        return 1;
      }
      windows.emplace_back(atoi(tok.substr(0, colon).c_str()), atoi(tok.substr(colon + 1).c_str()));
    }
    if (c == std::string::npos) break;
    p = c + 1;
  }

  int qdim; size_t nq; std::vector<float> queries = readFvecs(argv[2], qdim, nq);
  int gw; size_t ngt; std::vector<int> gt = readIvecs(argv[3], gw, ngt);

  auto t0 = clk::now();
  auto index = Index<dist_t, int>::loadIndex(argv[1]);
  const size_t N = index->currentNumNodes();
  const int dim = (int)(index->dataSizeBytes() / sizeof(float));
  if (dim % m) { fprintf(stderr, "dim %d not divisible by PQ_M %d\n", dim, m); return 1; }
  const int sub_dim = dim / m;
  if (getenv("NQ")) nq = std::min(nq, (size_t)atoll(getenv("NQ")));
  printf("[load] %.1fs nodes=%zu dim=%d | PQ m=%d sub_dim=%d code=%dB (%.0fx) | queries=%zu ef=%d K=%d T=%d\n",
         std::chrono::duration<double>(clk::now()-t0).count(), N, dim, m, sub_dim, m,
         (double)(dim * sizeof(float)) / m, nq, ef, K, threads);
  fflush(stdout);

  const char* vectors = index->vectorsMemory();
  const size_t stride = index->dataSizeBytes();
  auto nodeVec = [&](size_t n) { return reinterpret_cast<const float*>(vectors + n * stride); };

  // --- train the codebooks: one k-means per subspace, on a uniform stride sample ---
  const size_t n_tr = std::min(n_train, N);
  const size_t step = std::max<size_t>(1, N / n_tr);
  std::vector<float> codebooks((size_t)m * kCentroids * sub_dim);
  auto t1 = clk::now();
  flatnav::executeInParallel(0, (uint32_t)m, (uint32_t)std::min(threads, m), [&](uint32_t j) {
    std::vector<float> slice(n_tr * sub_dim);
    for (size_t i = 0; i < n_tr; i++)
      memcpy(&slice[i * sub_dim], nodeVec(i * step) + (size_t)j * sub_dim, sub_dim * sizeof(float));
    trainSubspace(slice.data(), n_tr, sub_dim, iters,
                  &codebooks[(size_t)j * kCentroids * sub_dim], 1234u + j);
  });
  printf("[train] %zu vectors x %d subspaces, %d iters in %.1fs\n", n_tr, m, iters,
         std::chrono::duration<double>(clk::now()-t1).count());
  fflush(stdout);

  // --- encode every node (codes are indexed by internal node id, as Index expects) ---
  std::vector<uint8_t> codes(N * m);
  const size_t CHUNK = 65536;
  const uint32_t nchunks = (uint32_t)((N + CHUNK - 1) / CHUNK);
  auto t2 = clk::now();
  flatnav::executeInParallel(0, nchunks, (uint32_t)threads, [&](uint32_t ch) {
    size_t lo = (size_t)ch * CHUNK, hi = std::min(lo + CHUNK, N);
    for (size_t n = lo; n < hi; n++) {
      const float* v = nodeVec(n);
      uint8_t* code = &codes[n * m];
      for (int j = 0; j < m; j++)
        code[j] = (uint8_t)nearestCentroid(v + (size_t)j * sub_dim,
                                           &codebooks[(size_t)j * kCentroids * sub_dim], sub_dim);
    }
  });
  printf("[encode] %zu nodes in %.1fs (%.2f GB of codes)\n", N,
         std::chrono::duration<double>(clk::now()-t2).count(), (double)codes.size() / 1e9);

  // Sanity check on the codebooks: mean relative error of the PQ distance against the exact
  // distance, over random (query, node) pairs. A sane 8-bit PQ lands in the low percents;
  // a large number here means the codebooks are bad and the sweep below is meaningless.
  {
    std::mt19937 rng(7);
    double sum_rel = 0; int pairs = 20000;
    for (int t = 0; t < pairs; t++) {
      size_t qi = rng() % nq, ni = rng() % N;
      const float* q = &queries[qi * qdim];
      float approx = 0;
      for (int j = 0; j < m; j++) {
        const float* cb = &codebooks[(size_t)j * kCentroids * sub_dim];
        approx += l2(q + (size_t)j * sub_dim,
                     cb + (size_t)codes[ni * m + j] * sub_dim, sub_dim);
      }
      float exact = l2(q, nodeVec(ni), dim);
      if (exact > 0) sum_rel += std::abs(approx - exact) / exact;
    }
    printf("[pq-err] mean relative distance error = %.3f%% over %d random (query,node) pairs\n",
           100.0 * sum_rel / pairs, pairs);
  }
  fflush(stdout);

  std::vector<std::unordered_set<int>> gtTopK(nq);
  for (size_t i = 0; i < nq; i++)
    for (int j = 0; j < K && j < gw; j++) gtTopK[i].insert(gt[i*gw+j]);

  // --- the sweep: recall@K and vector reads per query, vs the PQ window [lo,hi) ---
  // vec_reads/q = traversal exact distances + rescore reads = the remote fetches the design
  // is trying to cut. total_dists/q = pq + exact: a window makes the search LONGER (M11), so
  // this must be watched alongside the reads. Percentages are relative to the FIRST row.
  std::vector<std::vector<std::pair<float,int>>> results(nq);
  double baseline_reads = 0, baseline_dists = 0;
  for (auto [lo, hi] : windows) {
    index->setPQGate(codes.data(), (uint32_t)m, lo, hi);
    flatnav::g_gate_pq_dists.store(0);
    flatnav::g_gate_exact_dists.store(0);
    flatnav::g_gate_rescore_dists.store(0);

    auto ts = clk::now();
    flatnav::executeInParallel(0, (uint32_t)nq, (uint32_t)threads, [&](uint32_t i) {
      std::vector<float> lut((size_t)m * kCentroids);
      buildLUT(&queries[(size_t)i * qdim], codebooks.data(), m, sub_dim, lut.data());
      flatnav::tl_pq_lut = lut.data();
      results[i] = index->search((const void*)&queries[(size_t)i * qdim], K, ef);
      flatnav::tl_pq_lut = nullptr;
    });
    double dt = std::chrono::duration<double>(clk::now()-ts).count();

    size_t hits = 0, total = 0;
    for (size_t i = 0; i < nq; i++) {
      for (auto& pr : results[i]) if (gtTopK[i].count(pr.second)) hits++;
      total += std::min((size_t)K, gtTopK[i].size());
    }
    double pqd = (double)flatnav::g_gate_pq_dists.load() / nq;
    double exd = (double)flatnav::g_gate_exact_dists.load() / nq;
    double rsd = (double)flatnav::g_gate_rescore_dists.load() / nq;
    double reads = exd + rsd, dists = pqd + exd;
    if (baseline_reads == 0) { baseline_reads = reads; baseline_dists = dists; }
    char win[32];
    snprintf(win, sizeof(win), "%d:%d", lo, hi);
    printf("RESULT window=%-12s recall@%d=%.4f  pq/q=%.1f  exact/q=%.1f  rescore/q=%.1f  "
           "vec_reads/q=%.1f (%.1f%% of row1)  total_dists/q=%.1f (%.1f%% of row1)  %.1fs\n",
           win, K, (double)hits/total, pqd, exd, rsd,
           reads, baseline_reads > 0 ? 100.0 * reads / baseline_reads : 100.0,
           dists, baseline_dists > 0 ? 100.0 * dists / baseline_dists : 100.0, dt);
    fflush(stdout);
  }
  printf("\nnote: vec_reads/q excludes the ~num_initializations exact distances the per-query\n"
         "      entry-point search does before beamSearch (not gated).\n");
  return 0;
}
