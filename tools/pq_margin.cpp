// M15: the margin quality gate -- traverse on PQ, spend an exact vector read only where the PQ
// score lands close enough to the beam's K-th best to change this node's admission.
//
//   pq_margin <index.bin> <query.fvecs> <gt.ivecs> [threads=32] [ef=200] [K=100]
//
// M14 closed the per-vector half of the design's Part 2 gate: compressibility is a real per-vector
// property but predicts a beam-admission flip at only 1.05-1.28x (bar was 3x), and the realized
// error |d_exact - d_pq| needs the very fetch it would gate. The margin to the beam's K-th best is
// what survives -- it is query-dependent, free at runtime, and it is the last unmeasured row of
// M13's predicate table. Two passes:
//
//   [offline] From the M14 trace on the EXACT trajectory, treat the margin as a DETECTOR: for each
//             delta, what fraction of decisions does it fire on (its cost) and what fraction of the
//             admission flips -- and of the false rejects of TRUE top-K neighbors, the ones that
//             cost recall -- does it catch (its benefit)? This is the predicate's precision/recall
//             with no search machinery, and it bounds what any online gate can achieve.
//   [online]  The gate actually running: all-PQ traversal + a margin-triggered exact fetch, swept
//             over delta, reporting recall@K against vector reads per query. Two shapes:
//               one-sided  fetch if d_pq <  thresh*(1+delta)   -- verify everything PQ would admit
//               band       fetch if |d_pq - thresh| < delta*thresh -- verify only near-ties
//
// Success criterion, pre-registered in the design doc: beat all-PQ+rerank on the recall/remote-
// fetch frontier -- recall above 0.6917 at reads/q far below the exact baseline's 3982.8.
//
// env:
//   PQ_M=16       subquantizers = code bytes per vector (dim must be divisible by it)
//   TRAIN=200000  nodes sampled to train the codebooks
//   ITERS=25      k-means iterations per subspace
//   DELTAS=...    comma-separated margins to sweep (default 0.02,0.05,0.1,0.15,0.2,0.3,0.5)
//   NQ=...        cap on the number of queries
//   TRACE_NQ=20000  queries used for the offline detector pass (the trace is memory-heavy)
//   DET_CSV=...   offline detector curve
//   FRONT_CSV=... online recall/reads frontier
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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
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

// --- fvecs/ivecs readers + PQ training: identical to tools/pq_vecerr.cpp ---
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

// Offline detector tallies, one set per delta.
struct Det {
  double decisions = 0, fires = 0;         // cost: how often the predicate demands a fetch
  double flips = 0, flips_caught = 0;      // benefit: admission flips it would correct
  double gtfrej = 0, gtfrej_caught = 0;    // the recall-costing subset (true top-K, falsely rejected)
  void merge(const Det& o) {
    decisions += o.decisions; fires += o.fires;
    flips += o.flips; flips_caught += o.flips_caught;
    gtfrej += o.gtfrej; gtfrej_caught += o.gtfrej_caught;
  }
};

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <index.bin> <query.fvecs> <gt.ivecs> [threads=32] [ef=200] [K=100]\n",
            argv[0]);
    return 1;
  }
  const int threads = argc > 4 ? atoi(argv[4]) : 32;
  const int ef = argc > 5 ? atoi(argv[5]) : 200;
  const int K = argc > 6 ? atoi(argv[6]) : 100;
  const int m = getenv("PQ_M") ? atoi(getenv("PQ_M")) : 16;
  const size_t n_train = getenv("TRAIN") ? (size_t)atoll(getenv("TRAIN")) : 200000;
  const int iters = getenv("ITERS") ? atoi(getenv("ITERS")) : 25;
  const char* det_csv = getenv("DET_CSV");
  const char* front_csv = getenv("FRONT_CSV");
  std::string dstr = getenv("DELTAS") ? getenv("DELTAS") : "0.02,0.05,0.1,0.15,0.2,0.3,0.5";
  std::vector<float> deltas;
  for (size_t p = 0; p < dstr.size(); ) {
    size_t c = dstr.find(',', p);
    std::string tok = dstr.substr(p, c == std::string::npos ? c : c - p);
    if (!tok.empty()) deltas.push_back((float)atof(tok.c_str()));
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
  if (nq > ngt) { fprintf(stderr, "gt has %zu rows < %zu queries\n", ngt, nq); return 1; }
  const size_t trace_nq =
      std::min(nq, getenv("TRACE_NQ") ? (size_t)atoll(getenv("TRACE_NQ")) : (size_t)20000);
  printf("[load] %.1fs nodes=%zu dim=%d | PQ m=%d sub_dim=%d code=%dB | queries=%zu ef=%d K=%d T=%d\n",
         std::chrono::duration<double>(clk::now()-t0).count(), N, dim, m, sub_dim, m, nq, ef, K, threads);
  fflush(stdout);

  const char* vectors = index->vectorsMemory();
  const size_t stride = index->dataSizeBytes();
  auto nodeVec = [&](size_t n) { return reinterpret_cast<const float*>(vectors + n * stride); };

  // --- train + encode (same recipe as M11-M14, so every number is comparable) ---
  const size_t n_tr = std::min(n_train, N);
  const size_t tstep = std::max<size_t>(1, N / n_tr);
  std::vector<float> codebooks((size_t)m * kCentroids * sub_dim);
  auto t1 = clk::now();
  flatnav::executeInParallel(0, (uint32_t)m, (uint32_t)std::min(threads, m), [&](uint32_t j) {
    std::vector<float> slice(n_tr * sub_dim);
    for (size_t i = 0; i < n_tr; i++)
      memcpy(&slice[i * sub_dim], nodeVec(i * tstep) + (size_t)j * sub_dim, sub_dim * sizeof(float));
    trainSubspace(slice.data(), n_tr, sub_dim, iters,
                  &codebooks[(size_t)j * kCentroids * sub_dim], 1234u + j);
  });
  printf("[train] %zu vectors x %d subspaces, %d iters in %.1fs\n", n_tr, m, iters,
         std::chrono::duration<double>(clk::now()-t1).count());
  fflush(stdout);

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

  // ================= OFFLINE: the margin as a detector, on the exact trajectory =================
  // Only decisions with a live threshold count (the beam full), because that is where the gate has
  // anything to decide -- while the beam fills it fetches unconditionally.
  index->setPQGate(codes.data(), (uint32_t)m, 0, 0);
  index->setPQMargin(0.0f, false);
  std::vector<Det> det_one(deltas.size()), det_band(deltas.size());
  std::mutex mu;
  const uint32_t BLOCK = 256;
  const uint32_t nblocks = (uint32_t)((trace_nq + BLOCK - 1) / BLOCK);
  auto t3 = clk::now();
  flatnav::executeInParallel(0, nblocks, (uint32_t)threads, [&](uint32_t blk) {
    std::vector<Det> one(deltas.size()), band(deltas.size());
    std::vector<float> lut((size_t)m * kCentroids);
    std::vector<SpecDisc> disc;
    std::vector<int> gtrow;
    size_t qlo = (size_t)blk * BLOCK, qhi = std::min(qlo + BLOCK, trace_nq);
    for (size_t i = qlo; i < qhi; i++) {
      gtrow.assign(gt.begin() + i * gw, gt.begin() + i * gw + std::min(K, gw));
      std::sort(gtrow.begin(), gtrow.end());
      buildLUT(&queries[i * qdim], codebooks.data(), m, sub_dim, lut.data());
      disc.clear();
      flatnav::tl_pq_lut = lut.data();
      flatnav::tl_spec_disc = &disc;
      index->search((const void*)&queries[i * qdim], K, ef);
      flatnav::tl_spec_disc = nullptr;
      flatnav::tl_pq_lut = nullptr;

      for (const SpecDisc& r : disc) {
        if (r.thresh == std::numeric_limits<float>::max()) continue;  // beam still filling
        const bool in_e = r.exact < r.thresh, in_p = r.pq < r.thresh;
        const bool flip = in_e != in_p;
        const bool is_gt = std::binary_search(gtrow.begin(), gtrow.end(),
                                              (int)index->nodeLabel(r.node));
        const bool gtfr = is_gt && in_e && !in_p;
        for (size_t d = 0; d < deltas.size(); d++) {
          const float dl = deltas[d];
          const bool f_one = r.pq < r.thresh * (1.0f + dl);
          const bool f_band = f_one && r.pq > r.thresh * (1.0f - dl);
          one[d].decisions++;  band[d].decisions++;
          one[d].fires += f_one; band[d].fires += f_band;
          one[d].flips += flip; band[d].flips += flip;
          one[d].flips_caught += flip && f_one;  band[d].flips_caught += flip && f_band;
          one[d].gtfrej += gtfr; band[d].gtfrej += gtfr;
          one[d].gtfrej_caught += gtfr && f_one; band[d].gtfrej_caught += gtfr && f_band;
        }
      }
    }
    std::lock_guard<std::mutex> lk(mu);
    for (size_t d = 0; d < deltas.size(); d++) { det_one[d].merge(one[d]); det_band[d].merge(band[d]); }
  });
  printf("\n[offline] %zu queries traced in %.1fs\n", trace_nq,
         std::chrono::duration<double>(clk::now()-t3).count());
  printf("\n=== OFFLINE: margin as a detector (exact trajectory, live-threshold decisions only) ===\n");
  printf("# fire   = share of decisions demanding an exact fetch (the gate's cost)\n");
  printf("# flip   = share of admission flips the predicate would catch and correct\n");
  printf("# gt_frej= share of FALSELY REJECTED TRUE top-%d neighbors caught (the recall leak)\n\n", K);
  printf("%-8s | %-28s | %-28s\n", "", "one-sided  d_pq<t(1+d)", "band  |d_pq-t|<d*t");
  printf("%-8s %9s %9s %9s | %9s %9s %9s\n", "delta", "fire%", "flip%", "gt_frej%",
         "fire%", "flip%", "gt_frej%");
  for (size_t d = 0; d < deltas.size(); d++) {
    const Det& o = det_one[d];
    const Det& b = det_band[d];
    printf("%-8.2f %8.2f%% %8.2f%% %8.2f%% | %8.2f%% %8.2f%% %8.2f%%\n", deltas[d],
           100.0 * o.fires / o.decisions, 100.0 * o.flips_caught / o.flips,
           o.gtfrej > 0 ? 100.0 * o.gtfrej_caught / o.gtfrej : 0.0,
           100.0 * b.fires / b.decisions, 100.0 * b.flips_caught / b.flips,
           b.gtfrej > 0 ? 100.0 * b.gtfrej_caught / b.gtfrej : 0.0);
  }
  fflush(stdout);
  if (det_csv) {
    FILE* f = fopen(det_csv, "w");
    fprintf(f, "shape,delta,decisions,fire_rate,flip_capture,gt_frej_capture\n");
    for (size_t d = 0; d < deltas.size(); d++)
      for (int s = 0; s < 2; s++) {
        const Det& x = s ? det_band[d] : det_one[d];
        fprintf(f, "%s,%.4f,%.0f,%.6f,%.6f,%.6f\n", s ? "band" : "one-sided", deltas[d],
                x.decisions, x.fires / x.decisions, x.flips_caught / x.flips,
                x.gtfrej > 0 ? x.gtfrej_caught / x.gtfrej : 0.0);
      }
    fclose(f);
  }

  // ================= ONLINE: the gate running, recall vs remote fetches =================
  std::vector<std::unordered_set<int>> gtTopK(nq);
  for (size_t i = 0; i < nq; i++)
    for (int j = 0; j < K && j < gw; j++) gtTopK[i].insert(gt[i*gw+j]);

  printf("\n=== ONLINE: all-PQ traversal + margin-gated exact fetch (%zu queries) ===\n", nq);
  printf("# vec_reads/q = traversal vector reads + final rerank = the remote fetches to beat\n");
  printf("# baseline to beat: all-PQ+rerank (M11) -- anything at higher recall AND lower reads wins\n\n");
  std::vector<std::vector<std::pair<float,int>>> results(nq);
  double base_reads = 0;

  // The competing way to buy fewer vector reads is simply a smaller beam, so the exact search is
  // swept over ef as the reference frontier. A margin point only counts as a win if it sits above
  // and to the left of THIS curve -- beating all-PQ alone proves nothing, since all-PQ is a single
  // cheap/low-recall point rather than a frontier.
  struct Row { const char* name; int lo, hi; float margin; bool band; int ef; };
  std::vector<Row> rows;
  for (int e : {ef, 150, 125, 100}) {
    if (e > ef || e < K) continue;
    rows.push_back({"exact", 0, 0, 0.0f, false, e});
  }
  rows.push_back({"all-PQ", 0, 1000000, 0.0f, false, ef});
  for (float d : deltas) rows.push_back({"margin one-sided", 0, 1000000, d, false, ef});
  for (float d : deltas) rows.push_back({"margin band", 0, 1000000, d, true, ef});

  FILE* ff = front_csv ? fopen(front_csv, "w") : nullptr;
  if (ff) fprintf(ff, "shape,delta,ef,recall,vec_reads_per_q,pq_per_q,exact_per_q,rescore_per_q,total_dists_per_q,seconds\n");
  for (const Row& row : rows) {
    index->setPQGate(codes.data(), (uint32_t)m, row.lo, row.hi);
    index->setPQMargin(row.margin, row.band);
    flatnav::g_gate_pq_dists.store(0);
    flatnav::g_gate_exact_dists.store(0);
    flatnav::g_gate_rescore_dists.store(0);
    auto ts = clk::now();
    flatnav::executeInParallel(0, (uint32_t)nq, (uint32_t)threads, [&](uint32_t i) {
      std::vector<float> lut((size_t)m * kCentroids);
      buildLUT(&queries[(size_t)i * qdim], codebooks.data(), m, sub_dim, lut.data());
      flatnav::tl_pq_lut = lut.data();
      results[i] = index->search((const void*)&queries[(size_t)i * qdim], K, row.ef);
      flatnav::tl_pq_lut = nullptr;
    });
    double dt = std::chrono::duration<double>(clk::now()-ts).count();
    size_t hits = 0, tot_gt = 0;
    for (size_t i = 0; i < nq; i++) {
      for (auto& pr : results[i]) if (gtTopK[i].count(pr.second)) hits++;
      tot_gt += std::min((size_t)K, gtTopK[i].size());
    }
    const double pqd = (double)flatnav::g_gate_pq_dists.load() / nq;
    const double exd = (double)flatnav::g_gate_exact_dists.load() / nq;
    const double rsd = (double)flatnav::g_gate_rescore_dists.load() / nq;
    const double reads = exd + rsd;
    if (base_reads == 0) base_reads = reads;
    char lbl[48];
    if (row.margin > 0) snprintf(lbl, sizeof(lbl), "%s d=%.2f", row.name, row.margin);
    else snprintf(lbl, sizeof(lbl), "%s ef=%d", row.name, row.ef);
    printf("RESULT %-26s recall@%d=%.4f  vec_reads/q=%8.1f (%5.1f%% of exact)  "
           "pq/q=%.0f total_dists/q=%.0f  %.1fs\n",
           lbl, K, (double)hits / tot_gt, reads, 100.0 * reads / base_reads, pqd, pqd + exd, dt);
    fflush(stdout);
    if (ff)
      fprintf(ff, "%s,%.4f,%d,%.6f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
              row.margin > 0 ? (row.band ? "band" : "one-sided") : row.name, row.margin, row.ef,
              (double)hits / tot_gt, reads, pqd, exd, rsd, pqd + exd, dt);
  }
  if (ff) fclose(ff);
  index->setPQMargin(0.0f, false);
  return 0;
}
