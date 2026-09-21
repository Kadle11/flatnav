// Live speculate-then-validate search (Index::specBeamSearch), against the exact search.
//
//   spec_search <index.bin> <query.fvecs> [threads=16] [ef=200] [K=100]
//
// The pipeline advances on PQ scores and validates with exact distances k steps behind. Its
// validated lane reproduces beamSearch's pop/emplace sequence, so for every (k, w) this checks:
//
//   results    -- the returned top-K is bit-identical to the exact search's
//   expansions -- the ordered list of expanded nodes is identical, not merely the same length
//   reads      -- exact vector reads equal the exact search's own count
//
// The first two are hard gates: the validated lane is correct however badly speculation predicts,
// so a mismatch is a visited-set bookkeeping bug. What speculation itself gets right is the miss
// rate, and ORACLE=1 scores the speculative lane with exact distances to separate the prediction
// from PQ's error -- every remaining miss is then a node the stale admission threshold let onto
// the overlay and the beam later rejected, reported as `rej`.
//
// env:
//   PQ_M=16  TRAIN=200000  ITERS=25   PQ codebook config (as pq_top1)
//   DEPTHS=1,2,3,4,8                  validation depths k to sweep
//   WIDTHS=1                          speculation widths w (only w=1 is implemented)
//   ORACLE=0                          1 = score the speculative lane exact instead of PQ
//   DIAG=0                            1 = classify every miss (costs a queue scan per miss)
//   NQ=...                            cap on the number of queries
//   GT=<gt.ivecs>                     ground truth; adds recall@K to the exact line and each row
//
// Tiered placement + helper staging (the point of running ahead). Vectors sit on the far NUMA
// node, graph and PQ codes on the local one; the speculative lane hands each freshly discovered
// id to helper threads, which copy that vector far->local while speculation carries on, and
// validation reads the local copy k slots later.
//   VEC_NODE=-1  GRAPH_NODE=-1        NUMA nodes for vectors / graph (-1 = default allocator)
//   PF_HELPER_CPUS=<list>             cpus to pin staging helpers to (empty = no staging)
//   PF_BUF_SLOTS=1048576              staging buffer capacity, in vectors
//   PF_LOCAL_NODE=0                   NUMA node holding the staging buffer
//
// Validation offload is the other way round: rather than copying vectors toward the search, it
// runs the exact-distance half on cores pinned next to the vectors, so a step sends ~20 ids and
// receives ~20 floats instead of ~20 vectors. Speculation, the graph and the PQ codes stay near.
// Needs one lane per search thread; short-handed threads validate inline and are warned about.
//   VO_CPUS=<list>                    cpus to pin validation workers to (empty = no offload)
#define FLATNAV_PQ_GATE
#define FLATNAV_SPEC_TRACE
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/index/Index.h>
#include <flatnav/util/Multithreading.h>
#include <flatnav/util/NumaThreadPool.h>
#include <flatnav/util/PrefetchStaging.h>

#include <pthread.h>
#include <thread>
#include <unordered_set>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using flatnav::Index;
using flatnav::distances::SquaredL2Distance;
using flatnav::util::DataType;
using dist_t = SquaredL2Distance<DataType::float32>;
using clk = std::chrono::steady_clock;

static const int kCentroids = 256;

// --- fvecs reader + PQ training: identical to tools/pq_top1.cpp ---
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

static std::vector<int> parseInts(const char* env, const char* fallback) {
  std::string s = env ? env : fallback;
  std::vector<int> out;
  for (size_t p = 0; p < s.size();) {
    size_t e = s.find(',', p);
    if (e == std::string::npos) e = s.size();
    out.push_back(atoi(s.substr(p, e - p).c_str()));
    p = e + 1;
  }
  return out;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <index.bin> <query.fvecs> [threads=16] [ef=200] [K=100]\n", argv[0]);
    return 1;
  }
  const int threads = argc > 3 ? atoi(argv[3]) : 16;
  const int ef = argc > 4 ? atoi(argv[4]) : 200;
  const int K = argc > 5 ? atoi(argv[5]) : 100;
  const int m = getenv("PQ_M") ? atoi(getenv("PQ_M")) : 16;
  const size_t n_train = getenv("TRAIN") ? (size_t)atoll(getenv("TRAIN")) : 200000;
  const int iters = getenv("ITERS") ? atoi(getenv("ITERS")) : 25;
  const bool oracle = getenv("ORACLE") && atoi(getenv("ORACLE")) != 0;
  const bool diag = getenv("DIAG") && atoi(getenv("DIAG")) != 0;
  const std::vector<int> depths = parseInts(getenv("DEPTHS"), "1,2,3,4,8");
  const std::vector<int> widths = parseInts(getenv("WIDTHS"), "1");

  int qdim; size_t nq; std::vector<float> queries = readFvecs(argv[2], qdim, nq);
  if (getenv("NQ")) nq = std::min(nq, (size_t)atoll(getenv("NQ")));

  // Ground truth is optional: the pipeline's own gate is bit-exactness against the exact search,
  // and recall is reported so a placement or offload change can be shown not to have moved it.
  std::vector<std::unordered_set<int>> gt_topk;
  if (const char* gt_path = getenv("GT")) {
    int gw; size_t gn; std::vector<int> gt = readIvecs(gt_path, gw, gn);
    if (gn < nq) { fprintf(stderr, "GT has %zu rows, need %zu\n", gn, nq); return 1; }
    gt_topk.resize(nq);
    for (size_t i = 0; i < nq; i++)
      for (int j = 0; j < K && j < gw; j++) gt_topk[i].insert(gt[i * gw + j]);
  }
  // Fraction of each query's true top-K that `res` returned, summed over queries.
  auto recallOf = [&](const std::vector<std::vector<std::pair<float, int>>>& res) {
    size_t hits = 0, tot = 0;
    for (size_t i = 0; i < nq; i++) {
      for (const auto& pr : res[i]) if (gt_topk[i].count(pr.second)) hits++;
      tot += std::min((size_t)K, gt_topk[i].size());
    }
    return tot ? (double)hits / tot : 0.0;
  };

  const int vec_node = getenv("VEC_NODE") ? atoi(getenv("VEC_NODE")) : flatnav::util::kNoNumaNode;
  const int graph_node = getenv("GRAPH_NODE") ? atoi(getenv("GRAPH_NODE")) : flatnav::util::kNoNumaNode;

  auto t0 = clk::now();
  auto index = Index<dist_t, int>::loadIndex(argv[1], vec_node, graph_node);
  const size_t N = index->currentNumNodes();
  const int dim = (int)(index->dataSizeBytes() / sizeof(float));
  if (dim % m) { fprintf(stderr, "dim %d not divisible by PQ_M %d\n", dim, m); return 1; }
  const int sub_dim = dim / m;
  printf("[load] %.1fs nodes=%zu dim=%d | PQ m=%d sub_dim=%d | queries=%zu ef=%d K=%d T=%d%s\n",
         std::chrono::duration<double>(clk::now()-t0).count(), N, dim, m, sub_dim, nq, ef, K,
         threads, oracle ? " | ORACLE" : "");
  fflush(stdout);

  const char* vectors = index->vectorsMemory();
  const size_t stride = index->dataSizeBytes();
  auto nodeVec = [&](size_t n) { return reinterpret_cast<const float*>(vectors + n * stride); };

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

  std::vector<uint8_t> codes(N * m);
  const size_t CHUNK = 65536;
  const uint32_t nchunks = (uint32_t)((N + CHUNK - 1) / CHUNK);
  auto t2 = clk::now();
  flatnav::executeInParallel(0, nchunks, (uint32_t)threads, [&](uint32_t ch) {
    size_t clo = (size_t)ch * CHUNK, chi = std::min(clo + CHUNK, N);
    for (size_t n = clo; n < chi; n++)
      for (int j = 0; j < m; j++) {
        const float* sub = nodeVec(n) + (size_t)j * sub_dim;
        const float* cb = &codebooks[(size_t)j * kCentroids * sub_dim];
        codes[n * m + j] = (uint8_t)nearestCentroid(sub, cb, sub_dim);
      }
  });
  printf("[encode] %zu nodes in %.1fs\n\n", N, std::chrono::duration<double>(clk::now()-t2).count());
  fflush(stdout);

  // Codes installed, gate window empty: beamSearch stays fully exact and is the baseline.
  index->setPQGate(codes.data(), (uint32_t)m, 0, 0);

  const uint32_t BLOCK = 64;
  const uint32_t nblocks = (uint32_t)((nq + BLOCK - 1) / BLOCK);

  // --- exact baseline: results, expansion order, and vector reads, per query ---
  std::vector<std::vector<std::pair<float, int>>> base_res(nq);
  std::vector<std::vector<uint32_t>> base_exp(nq);
  std::vector<uint32_t> base_reads(nq);
  // One exact pass. Run first as the reference, and again after the last row.
  auto exactPass = [&](std::vector<std::vector<std::pair<float, int>>>& res,
                       std::vector<std::vector<uint32_t>>& exps, std::vector<uint32_t>& reads) {
    index->setSpecPipeline(0, 1, false);
    auto t = clk::now();
    flatnav::executeInParallel(0, nblocks, (uint32_t)threads, [&](uint32_t blk) {
      std::vector<float> lut((size_t)m * kCentroids);
      size_t qlo = (size_t)blk * BLOCK, qhi = std::min(qlo + BLOCK, nq);
      for (size_t i = qlo; i < qhi; i++) {
        buildLUT(&queries[i * qdim], codebooks.data(), m, sub_dim, lut.data());
        exps[i].clear();
        flatnav::tl_pq_lut = lut.data();
        flatnav::tl_spec_expand = &exps[i];
        res[i] = index->search((const void*)&queries[i * qdim], K, ef);
        flatnav::tl_spec_expand = nullptr;
        flatnav::tl_pq_lut = nullptr;
        reads[i] = flatnav::tl_gate_exact + 1;  // + the entry node, which the gate does not count
      }
    });
    return std::chrono::duration<double>(clk::now() - t).count();
  };
  const double base_secs = exactPass(base_res, base_exp, base_reads);
  double base_steps = 0, base_rd = 0;
  for (size_t i = 0; i < nq; i++) { base_steps += base_exp[i].size(); base_rd += base_reads[i]; }
  printf("[exact] %zu queries in %.2fs (%.0f qps), mean %.1f expansions, %.1f reads/query",
         nq, base_secs, nq / base_secs, base_steps / nq, base_rd / nq);
  if (!gt_topk.empty()) printf(", recall@%d=%.4f", K, recallOf(base_res));
  printf("\n\n");
  fflush(stdout);

  // Worker threads start only now, so none of them spin while the baseline above is timed.
  // Staging helpers. The ring is fed by the speculative lane with the exact ids validation will
  // read, so `_pf_k` stays 0 -- the top-K-candidate feeder that beamSearch uses is not wanted here.
  std::vector<int> helper_cpus = flatnav::parseCpuList(getenv("PF_HELPER_CPUS"));
  const size_t pf_slots = getenv("PF_BUF_SLOTS") ? (size_t)atoll(getenv("PF_BUF_SLOTS")) : (1u << 20);
  const int pf_node = getenv("PF_LOCAL_NODE") ? atoi(getenv("PF_LOCAL_NODE")) : 0;
  flatnav::StagingBuffer* pf_buf = nullptr;
  flatnav::MPMCRing* pf_ring = nullptr;
  std::vector<std::thread> helpers;
  std::atomic<bool> pf_stop{false};
  if (!helper_cpus.empty()) {
    pf_buf = new flatnav::StagingBuffer(pf_slots, index->dataSizeBytes(), pf_node);
    pf_ring = new flatnav::MPMCRing(1u << 16);
    for (int cpu : helper_cpus) {
      helpers.emplace_back([&index, pf_ring, &pf_stop, cpu]() {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
        uint32_t id;
        while (!pf_stop.load(std::memory_order_relaxed)) {
          if (pf_ring->dequeue(id)) index->stageNode(id);
          else _mm_pause();
        }
      });
    }
    printf("[stage] helpers=%zu slots=%zu buf_node=%d buf_MB=%.0f | vec_node=%d graph_node=%d\n",
           helper_cpus.size(), pf_slots, pf_node,
           (double)pf_slots * index->dataSizeBytes() / 1e6, vec_node, graph_node);
    fflush(stdout);
  }

  // Validation offload. Workers pinned to the node holding the vectors run the exact-distance
  // half; the search keeps links, PQ codes and the beam heaps on the near node at full clock.
  // Needs one lane per search thread -- fewer, and the unlucky threads validate inline.
  std::vector<int> vo_cpus = flatnav::parseCpuList(getenv("VO_CPUS"));
  flatnav::OffloadPool* vo_pool = nullptr;
  std::vector<std::thread> vo_workers;
  std::atomic<bool> vo_stop{false};
  if (!vo_cpus.empty()) {
    if (vo_cpus.size() < (size_t)threads)
      printf("[offload] WARNING: %zu lanes for %d search threads -- %zu will validate inline\n",
             vo_cpus.size(), threads, threads - vo_cpus.size());
    vo_pool = new flatnav::OffloadPool((uint32_t)vo_cpus.size(), 16);
    for (uint32_t l = 0; l < vo_cpus.size(); l++) {
      vo_workers.emplace_back([&index, vo_pool, &vo_stop, cpu = vo_cpus[l], l]() {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
        while (!vo_stop.load(std::memory_order_relaxed))
          if (!index->runValidationOffload(l)) _mm_pause();
      });
    }
    printf("[offload] lanes=%zu cpus=%s | vec_node=%d graph_node=%d\n",
           vo_cpus.size(), getenv("VO_CPUS"), vec_node, graph_node);
    fflush(stdout);
  }

  // Armed only now: the exact baseline above must not pay for staged lookups that can never hit.
  if (pf_buf) index->setPrefetchStaging(pf_buf, pf_ring, /* k = */ 0);
  if (vo_pool) index->setValidationOffload(vo_pool);
  index->setSpecDiag(diag);
  if (diag)
    printf("%-5s %-5s %9s %9s %9s %9s %9s %9s %9s\n", "k", "w", "checks/q", "miss%", "floor%",
           "rej%", "tie%", "order%", "wasted/q");
  else
    printf("%-5s %-5s %9s %9s %9s %9s %9s %9s %8s", "k", "w", "checks/q", "miss%", "disc/q",
           "reads/q", "wasted/q", "stalls/q", "qps");
  if (!gt_topk.empty()) printf(" %8s", "recall");
  printf("\n");

  bool all_ok = true;
  for (int k : depths) {
    for (int w : widths) {
      flatnav::g_spec_checks = 0; flatnav::g_spec_hits = 0; flatnav::g_spec_misses = 0;
      flatnav::g_spec_miss_rejected = 0; flatnav::g_spec_discarded = 0;
      flatnav::g_spec_stalls = 0; flatnav::g_spec_committed = 0; flatnav::g_spec_wasted = 0;
      flatnav::g_spec_miss_tie = 0; flatnav::g_spec_miss_order = 0; flatnav::g_spec_floor = 0;
      if (pf_buf) pf_buf->resetStats();
      if (vo_pool) vo_pool->resetClaims();  // each row runs on fresh threads
      for (int i = 0; i < flatnav::kSpecDepthCap; i++) {
        flatnav::g_spec_depth_checks[i] = 0;
        flatnav::g_spec_depth_misses[i] = 0;
      }
      index->setSpecPipeline(k, w, oracle);

      std::atomic<uint64_t> bad_res{0}, bad_exp{0}, bad_reads{0}, hits{0};
      auto ts = clk::now();
      flatnav::executeInParallel(0, nblocks, (uint32_t)threads, [&](uint32_t blk) {
        std::vector<float> lut((size_t)m * kCentroids);
        std::vector<uint32_t> exp;
        size_t qlo = (size_t)blk * BLOCK, qhi = std::min(qlo + BLOCK, nq);
        for (size_t i = qlo; i < qhi; i++) {
          buildLUT(&queries[i * qdim], codebooks.data(), m, sub_dim, lut.data());
          exp.clear();
          flatnav::tl_pq_lut = lut.data();
          flatnav::tl_spec_expand = &exp;
          auto res = index->search((const void*)&queries[i * qdim], K, ef);
          flatnav::tl_spec_expand = nullptr;
          flatnav::tl_pq_lut = nullptr;

          if (!gt_topk.empty()) {
            size_t h = 0;
            for (const auto& pr : res) if (gt_topk[i].count(pr.second)) h++;
            hits.fetch_add(h, std::memory_order_relaxed);
          }
          if (res != base_res[i]) bad_res.fetch_add(1, std::memory_order_relaxed);
          if (exp != base_exp[i]) bad_exp.fetch_add(1, std::memory_order_relaxed);
          if (flatnav::tl_spec_committed != base_reads[i])
            bad_reads.fetch_add(1, std::memory_order_relaxed);
        }
      });
      const double secs = std::chrono::duration<double>(clk::now()-ts).count();

      const double checks = (double)flatnav::g_spec_checks.load();
      const double misses = (double)flatnav::g_spec_misses.load();
      auto pct = [&](uint64_t v) { return misses > 0 ? 100.0 * v / misses : 0.0; };
      if (diag)
        printf("%-5d %-5d %9.1f %8.2f%% %8.2f%% %8.2f%% %8.2f%% %8.2f%% %9.1f", k, w, checks / nq,
               checks > 0 ? 100.0 * misses / checks : 0.0,
               checks > 0 ? 100.0 * flatnav::g_spec_floor.load() / checks : 0.0,
               pct(flatnav::g_spec_miss_rejected.load()), pct(flatnav::g_spec_miss_tie.load()),
               pct(flatnav::g_spec_miss_order.load()),
               (double)flatnav::g_spec_wasted.load() / nq);
      else
        printf("%-5d %-5d %9.1f %8.2f%% %9.2f %9.1f %9.1f %9.2f %8.0f", k, w, checks / nq,
               checks > 0 ? 100.0 * misses / checks : 0.0,
               (double)flatnav::g_spec_discarded.load() / nq,
               (double)flatnav::g_spec_committed.load() / nq,
               (double)flatnav::g_spec_wasted.load() / nq,
               (double)flatnav::g_spec_stalls.load() / nq, nq / secs);
      if (!gt_topk.empty()) {
        size_t tot = 0;
        for (size_t i = 0; i < nq; i++) tot += std::min((size_t)K, gt_topk[i].size());
        printf(" %8.4f", tot ? (double)hits.load() / tot : 0.0);
      }
      printf("\n");

      if (diag) {
        // How deep the pipeline was actually running when each prediction was made. A miss empties
        // the ring, so the configured k is only a ceiling on this.
        double tot = 0, wsum = 0;
        for (int i = 0; i < flatnav::kSpecDepthCap; i++) {
          const double c = (double)flatnav::g_spec_depth_checks[i].load();
          tot += c; wsum += c * i;
        }
        printf("      effective depth: mean %.2f of k=%d |", tot > 0 ? wsum / tot : 0.0, k);
        for (int i = 1; i <= k && i < flatnav::kSpecDepthCap; i++) {
          const double c = (double)flatnav::g_spec_depth_checks[i].load();
          if (c == 0) continue;
          printf("  d=%d %.1f%% miss %.1f%%", i, 100.0 * c / tot,
                 100.0 * flatnav::g_spec_depth_misses[i].load() / c);
        }
        printf("\n");
      }

      if (pf_buf) {
        const double uses = (double)pf_buf->uses();
        printf("      staged %.2f%% of validated reads (%.0f hits / %.0f)\n",
               uses > 0 ? 100.0 * pf_buf->hits() / uses : 0.0, (double)pf_buf->hits(), uses);
      }
      // Results stay bit-exact when a thread validates inline, but the row no longer measures
      // offload, so it fails rather than being filed as an offload number.
      if (vo_pool && vo_pool->inlineSearches() > 0) {
        all_ok = false;
        printf("      FAIL k=%d w=%d: %llu of %zu searches found no offload lane and validated inline\n",
               k, w, (unsigned long long)vo_pool->inlineSearches(), nq);
      }
      if (bad_res || bad_exp || bad_reads) {
        all_ok = false;
        printf("      FAIL k=%d w=%d: %llu result, %llu expansion-order, %llu read-count mismatches\n",
               k, w, (unsigned long long)bad_res.load(), (unsigned long long)bad_exp.load(),
               (unsigned long long)bad_reads.load());
      }
      fflush(stdout);
    }
  }

  // Exact again, last, with every worker stopped and the pipeline disarmed -- the conditions of
  // the first pass. That pass ran before any row and paid warm-up the rows did not, so it can
  // understate the baseline; report both and let the reader take the later one.
  pf_stop.store(true, std::memory_order_relaxed);
  for (auto& h : helpers) h.join();
  vo_stop.store(true, std::memory_order_relaxed);
  for (auto& w : vo_workers) w.join();
  index->setPrefetchStaging(nullptr, nullptr, 0);
  index->setValidationOffload(nullptr);
  {
    std::vector<std::vector<std::pair<float, int>>> end_res(nq);
    std::vector<std::vector<uint32_t>> end_exp(nq);
    std::vector<uint32_t> end_reads(nq);
    const double end_secs = exactPass(end_res, end_exp, end_reads);
    printf("\n[exact-end] %zu queries in %.2fs (%.0f qps), qps %+.1f%% vs the first exact pass\n",
           nq, end_secs, nq / end_secs, 100.0 * (base_secs / end_secs - 1.0));
    if (end_res != base_res || end_exp != base_exp || end_reads != base_reads) {
      all_ok = false;
      printf("      FAIL exact-end: results differ from the first exact pass\n");
    }
    fflush(stdout);
  }

  printf("\n# checks/q  predictions tested per query; the first pick off an empty ring is not one\n");
  printf("# miss%%     share of predictions the validated lane did not confirm\n");
  printf("# disc/q    speculative steps thrown away | wasted/q  their vectors\n");
  if (diag) {
    printf("# floor%%    share of checks whose true winner was invisible to speculation, so no\n");
    printf("#           width recovers it -- comparable to pq_top1's `rejected` coverage column\n");
    printf("# rej%%      predicted node never admitted to the beam: no ranking could have found it\n");
    printf("# tie%%      admitted at the SAME distance as the node expanded; lost on heap order\n");
    printf("# order%%    admitted at a different distance and genuinely ranked wrong\n");
    if (oracle)
      printf("# ORACLE + DIAG: order%% must be 0 -- the speculative lane has the exact scores\n");
  } else {
    printf("# qps is below exact by construction: same reads, plus PQ and discarded work\n");
  }
  if (depths.empty() || widths.empty())
    printf("\nexact baseline only: no (k, w) configured, nothing compared\n");
  else
    printf("\n%s\n", all_ok ? "PASS: results, expansion order and read counts identical to exact"
                            : "FAIL: see mismatches above");

  delete pf_buf;
  delete pf_ring;
  delete vo_pool;
  return all_ok ? 0 : 1;
}
