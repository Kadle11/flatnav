// PQ speculation divergence: starting from the exact search's own beam at a window's lower edge,
// how far does a PQ scout diverge from the exact search INSIDE the window [lo,hi)? The search is
// exact for steps [0,lo) and PQ inside [lo,hi) (FLATNAV_PQ_GATE), so the exact baseline and the
// gated run share an identical prefix and diverge only within the window -- the divergence is
// purely the effect of PQ speculation in that phase, from a correct starting beam.
//
//   pq_specdiverge <index.bin> <query.fvecs> [threads=32] [ef=200] [K=100]
//
// Two decision units, reported per sliding window:
//   drift    (expanded-node set)  -- Jaccard of the gated vs exact trajectories over [lo,hi):
//                                     "does the scout walk the same nodes the exact search does?"
//   fetch    (candidate ranking)  -- at each in-window expansion, rank that expansion's newly
//                                     discovered neighbors by the PQ score that drove the scout
//                                     vs their exact distance:
//              kendall  = mean Kendall tau-a over expansions (local ordering agreement)
//              top1     = P(scout's nearest discovered == exact's nearest discovered)
//              pool_ov  = over the whole window's discovered set, overlap of the r nearest by PQ
//                         vs the r nearest by exact ("would the scout fetch the same targets?")
//
// env:
//   PQ_M=16       subquantizers = code bytes per vector (dim must be divisible by it)
//   TRAIN=200000  nodes sampled to train the codebooks
//   ITERS=25      k-means iterations per subspace
//   WIDTHS=30,60  window widths to sweep (the sizes asked for)
//   MAXOFF=240    highest window lower-edge (offsets 0,w,2w,... up to MAXOFF)
//   POOLR=32      r for the pooled top-r fetch-target overlap
//   NQ=...        cap on the number of queries
#define FLATNAV_PQ_GATE
#define FLATNAV_SPEC_TRACE
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
#include <climits>
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
using flatnav::SpecDisc;
using flatnav::distances::SquaredL2Distance;
using flatnav::util::DataType;
using dist_t = SquaredL2Distance<DataType::float32>;
using clk = std::chrono::steady_clock;

static const int kCentroids = 256;  // one code byte per subquantizer

// --- fvecs/ivecs readers + PQ training/encoding: identical to tools/pq_stepgate.cpp ---
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

static inline float l2(const float* a, const float* b, int d) {
  float s = 0.0f;
  for (int i = 0; i < d; i++) { float t = a[i] - b[i]; s += t * t; }
  return s;
}

static inline int nearestCentroid(const float* sub, const float* codebook, int sub_dim) {
  float best = FLT_MAX; int bc = 0;
  for (int c = 0; c < kCentroids; c++) {
    float d = l2(sub, codebook + (size_t)c * sub_dim, sub_dim);
    if (d < best) { best = d; bc = c; }
  }
  return bc;
}

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

static void buildLUT(const float* q, const float* codebooks, int m, int sub_dim, float* lut) {
  for (int j = 0; j < m; j++) {
    const float* sub = q + (size_t)j * sub_dim;
    const float* cb = codebooks + (size_t)j * kCentroids * sub_dim;
    for (int c = 0; c < kCentroids; c++)
      lut[(size_t)j * kCentroids + c] = l2(sub, cb + (size_t)c * sub_dim, sub_dim);
  }
}

// Kendall tau-a between exact and pq over one expansion's discovered set (small n, O(n^2)).
// +1 per pair the two rankings agree on, -1 per disagreement, ties contribute 0.
static double kendallTau(const std::vector<SpecDisc>& g, size_t lo, size_t hi) {
  long conc = 0, disc = 0;
  for (size_t a = lo; a < hi; a++)
    for (size_t b = a + 1; b < hi; b++) {
      float de = g[a].exact - g[b].exact, dp = g[a].pq - g[b].pq;
      if (de == 0 || dp == 0) continue;
      if ((de < 0) == (dp < 0)) conc++; else disc++;
    }
  long tot = conc + disc;
  return tot ? (double)(conc - disc) / tot : 0.0;
}

// Percentile of an already-sorted vector (nearest-rank).
static double pct(const std::vector<double>& s, double p) {
  if (s.empty()) return 0.0;
  size_t k = (size_t)(p / 100.0 * (s.size() - 1) + 0.5);
  return s[std::min(k, s.size() - 1)];
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <index.bin> <query.fvecs> [threads=32] [ef=200] [K=100]\n", argv[0]);
    return 1;
  }
  const int threads = argc > 3 ? atoi(argv[3]) : 32;
  const int ef = argc > 4 ? atoi(argv[4]) : 200;
  const int K = argc > 5 ? atoi(argv[5]) : 100;
  const int m = getenv("PQ_M") ? atoi(getenv("PQ_M")) : 16;
  const size_t n_train = getenv("TRAIN") ? (size_t)atoll(getenv("TRAIN")) : 200000;
  const int iters = getenv("ITERS") ? atoi(getenv("ITERS")) : 25;
  const int maxoff = getenv("MAXOFF") ? atoi(getenv("MAXOFF")) : 240;
  const int poolr = getenv("POOLR") ? atoi(getenv("POOLR")) : 32;
  const char* drift_csv = getenv("DRIFT_CSV");  // if set, per-window drift histogram (20 bins) is written here
  std::string width_str = getenv("WIDTHS") ? getenv("WIDTHS") : "30,60";
  std::vector<int> widths;
  for (size_t p = 0; p < width_str.size(); ) {
    size_t c = width_str.find(',', p);
    std::string tok = width_str.substr(p, c == std::string::npos ? c : c - p);
    if (!tok.empty()) widths.push_back(atoi(tok.c_str()));
    if (c == std::string::npos) break;
    p = c + 1;
  }

  int qdim; size_t nq; std::vector<float> queries = readFvecs(argv[2], qdim, nq);

  auto t0 = clk::now();
  auto index = Index<dist_t, int>::loadIndex(argv[1]);
  const size_t N = index->currentNumNodes();
  const int dim = (int)(index->dataSizeBytes() / sizeof(float));
  if (dim % m) { fprintf(stderr, "dim %d not divisible by PQ_M %d\n", dim, m); return 1; }
  const int sub_dim = dim / m;
  if (getenv("NQ")) nq = std::min(nq, (size_t)atoll(getenv("NQ")));
  printf("[load] %.1fs nodes=%zu dim=%d | PQ m=%d sub_dim=%d code=%dB | queries=%zu ef=%d K=%d T=%d\n",
         std::chrono::duration<double>(clk::now()-t0).count(), N, dim, m, sub_dim, m, nq, ef, K, threads);
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
    size_t clo = (size_t)ch * CHUNK, chi = std::min(clo + CHUNK, N);
    for (size_t n = clo; n < chi; n++) {
      const float* v = nodeVec(n);
      uint8_t* code = &codes[n * m];
      for (int j = 0; j < m; j++)
        code[j] = (uint8_t)nearestCentroid(v + (size_t)j * sub_dim,
                                           &codebooks[(size_t)j * kCentroids * sub_dim], sub_dim);
    }
  });
  printf("[encode] %zu nodes in %.1fs\n", N, std::chrono::duration<double>(clk::now()-t2).count());
  fflush(stdout);

  // --- Pass 0: exact baseline, once. Capture each query's exact trajectory (the ground truth
  // the gated windows are compared against). Gate empty -> pure exact search. ---
  std::vector<std::vector<uint32_t>> exp_exact(nq);
  index->setPQGate(codes.data(), (uint32_t)m, 0, 0);
  auto tb = clk::now();
  flatnav::executeInParallel(0, (uint32_t)nq, (uint32_t)threads, [&](uint32_t i) {
    flatnav::tl_spec_expand = &exp_exact[i];
    index->search((const void*)&queries[(size_t)i * qdim], K, ef);
    flatnav::tl_spec_expand = nullptr;
  });
  printf("[exact] baseline trajectories captured in %.1fs (mean %.0f expansions/q)\n",
         std::chrono::duration<double>(clk::now()-tb).count(),
         [&]{ double s = 0; for (auto& t : exp_exact) s += t.size(); return s / nq; }());
  fflush(stdout);

  printf("\n# drift = Jaccard(gated, exact) expanded-node sets over [lo,hi)  (1.0 = identical walk)\n");
  printf("# fetch = per-expansion PQ-vs-exact ranking of discovered neighbors\n");
  printf("#   kendall (tau-a, local order)  top1 (nearest agrees)  pool_ov@%d (window fetch-target overlap)\n",
         poolr);
  printf("# support n = queries whose search reached the window\n\n");
  fflush(stdout);

  // per-query scratch, filled in parallel (each query writes only its own slot), reduced serially
  std::vector<char> sup2(nq), sup1(nq);
  std::vector<double> jac(nq), ken(nq), top1(nq), pool(nq);

  const int kHistBins = 20;  // drift histogram: 20 equal bins over [0,1]
  FILE* hist = drift_csv ? fopen(drift_csv, "w") : nullptr;
  if (hist) fprintf(hist, "width,lo,hi,bin_lo,bin_hi,count,frac\n");

  for (int w : widths) {
    for (int lo = 0; lo + 1 <= maxoff; lo += w) {
      const int hi = lo + w;
      index->setPQGate(codes.data(), (uint32_t)m, lo, hi);
      auto ts = clk::now();
      flatnav::executeInParallel(0, (uint32_t)nq, (uint32_t)threads, [&](uint32_t i) {
        std::vector<float> lut((size_t)m * kCentroids);
        buildLUT(&queries[(size_t)i * qdim], codebooks.data(), m, sub_dim, lut.data());
        flatnav::tl_pq_lut = lut.data();
        std::vector<uint32_t> exp_g;
        std::vector<SpecDisc> disc;
        flatnav::tl_spec_expand = &exp_g;
        flatnav::tl_spec_disc = &disc;
        index->search((const void*)&queries[(size_t)i * qdim], K, ef);
        flatnav::tl_spec_expand = nullptr;
        flatnav::tl_spec_disc = nullptr;
        flatnav::tl_pq_lut = nullptr;

        // Trace 2: expanded-node-set drift over [lo,hi). Prefix [0,lo) is identical by
        // construction, so only the window slice carries divergence.
        const auto& ex = exp_exact[i];
        sup2[i] = (ex.size() > (size_t)lo && exp_g.size() > (size_t)lo) ? 1 : 0;
        if (sup2[i]) {
          std::unordered_set<uint32_t> se(ex.begin() + lo, ex.begin() + std::min((size_t)hi, ex.size()));
          std::unordered_set<uint32_t> sg(exp_g.begin() + lo, exp_g.begin() + std::min((size_t)hi, exp_g.size()));
          size_t inter = 0;
          for (uint32_t x : sg) if (se.count(x)) inter++;
          size_t uni = se.size() + sg.size() - inter;
          jac[i] = uni ? (double)inter / uni : 1.0;
        } else jac[i] = 0.0;

        // Trace 1: fetch-target ranking from the in-window discovered set (disc is in step order).
        sup1[i] = disc.empty() ? 0 : 1;
        if (sup1[i]) {
          double ksum = 0, t1sum = 0; int kgroups = 0, egroups = 0;
          size_t s = 0;
          while (s < disc.size()) {
            size_t e = s;
            while (e < disc.size() && disc[e].step == disc[s].step) e++;  // one expansion's group
            if (e - s >= 1) {
              size_t am = s;
              for (size_t j = s + 1; j < e; j++) if (disc[j].exact < disc[am].exact) am = j;
              size_t pm = s;
              for (size_t j = s + 1; j < e; j++) if (disc[j].pq < disc[pm].pq) pm = j;
              t1sum += (am == pm) ? 1.0 : 0.0; egroups++;
            }
            if (e - s >= 2) { ksum += kendallTau(disc, s, e); kgroups++; }
            s = e;
          }
          ken[i] = kgroups ? ksum / kgroups : 0.0;
          top1[i] = egroups ? t1sum / egroups : 0.0;
          // pooled top-r overlap over the whole window's discovered set
          int r = std::min((int)disc.size(), poolr);
          std::vector<int> idx(disc.size());
          std::iota(idx.begin(), idx.end(), 0);
          std::partial_sort(idx.begin(), idx.begin() + r, idx.end(),
                            [&](int a, int b) { return disc[a].exact < disc[b].exact; });
          std::unordered_set<int> topExact(idx.begin(), idx.begin() + r);
          std::partial_sort(idx.begin(), idx.begin() + r, idx.end(),
                            [&](int a, int b) { return disc[a].pq < disc[b].pq; });
          int ov = 0;
          for (int j = 0; j < r; j++) if (topExact.count(idx[j])) ov++;
          pool[i] = r ? (double)ov / r : 0.0;
        } else { ken[i] = top1[i] = pool[i] = 0.0; }
      });
      double dt = std::chrono::duration<double>(clk::now()-ts).count();

      size_t n1 = 0; double ksum = 0, tsum = 0, psum = 0;
      std::vector<double> ds;       // supported per-query drift, for the distribution
      ds.reserve(nq);
      std::vector<size_t> hbin(kHistBins, 0);
      for (size_t i = 0; i < nq; i++) {
        if (sup2[i]) {
          ds.push_back(jac[i]);
          int b = (int)(jac[i] * kHistBins); if (b >= kHistBins) b = kHistBins - 1;
          hbin[b]++;
        }
        if (sup1[i]) { ksum += ken[i]; tsum += top1[i]; psum += pool[i]; n1++; }
      }
      const size_t n2 = ds.size();
      std::sort(ds.begin(), ds.end());
      double jmean = 0; for (double v : ds) jmean += v; jmean = n2 ? jmean / n2 : 0.0;
      printf("SPEC w=%-3d win=%-8s drift_jaccard=%.3f  fetch: kendall=%.3f top1=%.3f pool_ov@%d=%.3f  "
             "| support n=%zu (%.0f%%)  %.1fs\n",
             w, (std::to_string(lo) + ":" + std::to_string(hi)).c_str(),
             jmean, n1 ? ksum / n1 : 0.0, n1 ? tsum / n1 : 0.0,
             poolr, n1 ? psum / n1 : 0.0, n2, 100.0 * n2 / nq, dt);
      printf("     drift dist: p5=%.3f p10=%.3f p25=%.3f p50=%.3f p75=%.3f p90=%.3f p95=%.3f  "
             "| frac<0.05: %.3f  frac>=0.95: %.3f\n",
             pct(ds,5), pct(ds,10), pct(ds,25), pct(ds,50), pct(ds,75), pct(ds,90), pct(ds,95),
             n2 ? (double)hbin[0] / n2 : 0.0, n2 ? (double)hbin[kHistBins-1] / n2 : 0.0);
      fflush(stdout);
      if (hist) {
        for (int b = 0; b < kHistBins; b++)
          fprintf(hist, "%d,%d,%d,%.2f,%.2f,%zu,%.5f\n", w, lo, hi,
                  (double)b / kHistBins, (double)(b + 1) / kHistBins, hbin[b],
                  n2 ? (double)hbin[b] / n2 : 0.0);
        fflush(hist);
      }
      if (n2 < nq / 100) break;  // window beyond nearly every trajectory: stop this width
    }
    printf("\n");
  }
  return 0;
}
