// Top-1 frontier divergence of one-step PQ speculation (.claude/assets/speculate_validate_pipeline.md).
//
//   pq_top1 <index.bin> <query.fvecs> [threads=16] [ef=200] [K=100]
//
// The pipeline expands step s on PQ scores while step s-1 is validated exact; a validation failure
// is a TOP-1 FRONTIER MISMATCH -- the node the speculative heap pops next differs from the node
// the exact search pops next -- and forces a redo from the last validated step. This measures
// that per-step miss rate from a VALIDATED beam at every step (unlike M13, whose scout drifts).
//
// Method: run the EXACT search with a LUT installed (gate off), capturing the expansion order
// (tl_spec_expand) and every discovered neighbor with both scores (tl_spec_disc). beamSearch's
// candidate heap is never pruned -- nodes leave it only by being popped -- and a neighbor enters
// it iff exact < thresh, so the heap after each expansion is replayed exactly from the trace.
// Sanity: the replayed heap's exact argmin must be the node the search really expands next
// (compared by exact distance, so duplicate vectors -- which tie -- count as the same pick).
// After expanding step s, the speculative heap differs from the exact one only in step s's
// neighbors (admitted iff pq < thresh) and in how it ranks:
//   mixed  -- candidates discovered >= DELAY steps ago keep their validated exact score; newer
//             ones use PQ (DELAY = how many steps validation trails the speculative search;
//             a candidate is still PQ-only iff its residency in the heap is <= DELAY)
//   all-PQ -- every candidate is ranked by its PQ score (membership still validated)
//   low    -- mixed, but a newest neighbor uses PQ only if it is among the PQ_FRAC of nodes with
//             the smallest compression error ||x - c(x)||^2; the rest are scored exact
//   rnd    -- same, with a pseudo-random PQ_FRAC of nodes (control: is it the error, or just fewer
//             PQ-scored nodes?)
// A miss = the speculative argmin's exact distance != that of the node expanded at step s+1.
//
// env:
//   PQ_M=16  TRAIN=200000  ITERS=25   PQ codebook config (as M11-M14)
//   BINW=30  MAXSTEP=300               step bins (last bin is the overflow)
//   PQ_FRAC=0.25                       share of nodes the low/rnd gates score with PQ
//   DELAY=1                            steps validation trails the speculative search
//   SAFE_MAX=16                        cap on the per-step safe delay k* (largest k whose delays
//                                      1..k all pick the true winner; oracle bound, mixed scoring)
//   SAFE_CSV=...                       per-bin k* histogram
//   WMAX=8                             top-w coverage: P(true winner within the mixed heap's top w),
//                                      i.e. the miss rate if the top w are speculated in parallel
//   COV_CSV=...                        per-bin histogram of the true winner's mixed rank
//   LAZY_DELTA=0.10                    PQ admission margin for the validation schedules
//   LAZY_ALPHAS=0,0.05,...,0.3         lower-bound margins for lazy (decision-relevance) validation
//   NQ=...                             cap on the number of queries
//   STEP_CSV=...                       per-bin table
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
#include <cmath>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using flatnav::Index;
using flatnav::SpecDisc;
using flatnav::distances::SquaredL2Distance;
using flatnav::util::DataType;
using dist_t = SquaredL2Distance<DataType::float32>;
using clk = std::chrono::steady_clock;

static const int kCentroids = 256;  // one code byte per subquantizer

// --- fvecs reader + PQ training: identical to tools/pq_vecerr.cpp ---
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

// Per-bin tallies, keyed by the step whose pop is being predicted (s+1).
struct Acc {
  std::vector<double> n, sane, win_pq, miss_mixed, miss_allpq, miss_low, miss_rnd;
  std::vector<double> safe;  // [bin * nk + k*] histogram of the per-step safe delay
  std::vector<double> cov;   // [bin * nw + r] true winner's mixed rank r-1 (r = WMAX: >WMAX, +1: rejected)
  double disc = 0, disc_low = 0, disc_rnd = 0;  // PQ share of discovered neighbors per gate
  Acc(int nb, int nk, int nw)
      : n(nb), sane(nb), win_pq(nb), miss_mixed(nb), miss_allpq(nb), miss_low(nb), miss_rnd(nb),
        safe((size_t)nb * nk), cov((size_t)nb * nw) {}
  void merge(const Acc& o) {
    for (size_t b = 0; b < n.size(); b++) {
      n[b] += o.n[b]; sane[b] += o.sane[b]; win_pq[b] += o.win_pq[b];
      miss_mixed[b] += o.miss_mixed[b]; miss_allpq[b] += o.miss_allpq[b];
      miss_low[b] += o.miss_low[b]; miss_rnd[b] += o.miss_rnd[b];
    }
    for (size_t i = 0; i < safe.size(); i++) safe[i] += o.safe[i];
    for (size_t i = 0; i < cov.size(); i++) cov[i] += o.cov[i];
    disc += o.disc; disc_low += o.disc_low; disc_rnd += o.disc_rnd;
  }
};

struct Cand { uint32_t node, step; float exact, pq, thresh; bool in_e, in_p, low, rnd, popped; };

// Control gate: a fixed pseudo-random `frac` of nodes (Knuth multiplicative hash of the id).
static inline bool rndPick(uint32_t n, double frac) {
  return (uint32_t)(n * 2654435761u) < (uint32_t)(frac * 4294967295.0);
}

// Per-schedule tallies for replaySchedule.
struct Sched {
  double q = 0, steps = 0, miss = 0, read_steps = 0, reads = 0, reads_final = 0, overlap = 0;
  void add(const Sched& o) {
    q += o.q; steps += o.steps; miss += o.miss; read_steps += o.read_steps;
    reads += o.reads; reads_final += o.reads_final; overlap += o.overlap;
  }
};

// Share of `ref` (sorted node ids) present in `got`.
static double overlapK(const std::vector<uint32_t>& got, const std::vector<uint32_t>& ref) {
  if (ref.empty()) return 1.0;
  size_t hit = 0;
  for (uint32_t x : got) hit += std::binary_search(ref.begin(), ref.end(), x);
  return (double)hit / ref.size();
}

// Validation schedules replayed on the exact trajectory (the path a correct pipeline follows).
// Admission is PQ-only with the M15 margin: a neighbor enters the heap iff pq < thresh*(1+delta).
//   eager -- every admitted neighbor is read on discovery (M15's one-sided gate)
//   lazy  -- a neighbor is read only when it could be the next pick (decision relevance): the heap
//            is keyed by exact if read, else by the PQ lower bound pq*(1-alpha), and the top is read
//            until a read candidate is on top -- that is the pick. The top-K at the end is certified
//            the same way: unread candidates are read in lower-bound order while they could still
//            enter it.
// Either way the node the exact search expands is read if it was not (it is a beam member), and a
// miss is a pick whose exact distance differs from that node's.
static void replaySchedule(const std::vector<Cand>& cands, const std::unordered_map<uint32_t, uint32_t>& where,
                           const std::vector<uint32_t>& exp, float delta, float alpha, bool eager, int K,
                           const std::vector<uint32_t>& ref, Sched& st) {
  const size_t n = cands.size();
  std::vector<uint8_t> rd(n, 0), popped(n, 0), adm(n, 0);
  auto key = [&](uint32_t i) { return rd[i] ? cands[i].exact : cands[i].pq * (1.0f - alpha); };
  using E = std::pair<float, uint32_t>;
  std::priority_queue<E, std::vector<E>, std::greater<E>> heap;
  bool read = false;
  auto readNode = [&](uint32_t i) { if (!rd[i]) { rd[i] = 1; st.reads++; read = true; } };

  size_t r = 0;
  for (size_t s = 0; s < exp.size(); s++) {
    read = false;
    auto it = where.find(exp[s]);
    if (it != where.end()) { readNode(it->second); popped[it->second] = 1; }
    if (s + 1 == exp.size()) break;
    for (; r < n && cands[r].step == s; r++) {
      if (!(cands[r].pq < cands[r].thresh * (1.0f + delta))) continue;
      adm[r] = 1;
      if (eager) readNode((uint32_t)r);
      heap.push({key((uint32_t)r), (uint32_t)r});
    }
    uint32_t pick = UINT32_MAX;
    while (!heap.empty()) {
      const E top = heap.top();
      const uint32_t i = top.second;
      if (popped[i] || top.first != key(i)) { heap.pop(); continue; }  // expanded, or a stale key
      if (!rd[i]) { heap.pop(); readNode(i); heap.push({key(i), i}); continue; }
      pick = i;
      break;
    }
    st.steps++;
    st.miss += (pick == UINT32_MAX || cands[pick].exact != cands[where.at(exp[s + 1])].exact);
    st.read_steps += read;
  }

  const double before = st.reads;
  std::priority_queue<E> best;  // max-heap: the K smallest read exact distances
  auto offer = [&](uint32_t i) { best.push({cands[i].exact, i}); if ((int)best.size() > K) best.pop(); };
  std::vector<E> unread;
  for (uint32_t i = 0; i < n; i++) {
    if (rd[i]) offer(i);
    else if (adm[i]) unread.push_back({cands[i].pq * (1.0f - alpha), i});
  }
  std::sort(unread.begin(), unread.end());
  for (const E& u : unread) {
    if ((int)best.size() >= K && u.first >= best.top().first) break;
    readNode(u.second);
    offer(u.second);
  }
  st.reads_final += st.reads - before;
  std::vector<uint32_t> got;
  for (; !best.empty(); best.pop()) got.push_back(cands[best.top().second].node);
  st.overlap += overlapK(got, ref);
  st.q++;
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
  const int binw = getenv("BINW") ? atoi(getenv("BINW")) : 30;
  const int maxstep = getenv("MAXSTEP") ? atoi(getenv("MAXSTEP")) : 300;
  const double frac = getenv("PQ_FRAC") ? atof(getenv("PQ_FRAC")) : 0.25;
  const int delay = getenv("DELAY") ? atoi(getenv("DELAY")) : 1;
  const int safe_max = getenv("SAFE_MAX") ? atoi(getenv("SAFE_MAX")) : 16;
  const int nk = safe_max + 1;
  const char* safe_csv = getenv("SAFE_CSV");
  const int wmax = getenv("WMAX") ? atoi(getenv("WMAX")) : 8;
  const int nw = wmax + 2;  // ranks 1..WMAX, >WMAX, rejected
  const char* cov_csv = getenv("COV_CSV");
  const float lazy_delta = getenv("LAZY_DELTA") ? (float)atof(getenv("LAZY_DELTA")) : 0.10f;
  struct SchedCfg { std::string name; float delta, alpha; bool eager; };
  std::vector<SchedCfg> scheds = {{"exact", INFINITY, 0.0f, true}};
  {
    char nm[64];
    snprintf(nm, sizeof nm, "eager (M15) d=%.2f", lazy_delta);
    scheds.push_back({nm, lazy_delta, 0.0f, true});
    std::string alphas = getenv("LAZY_ALPHAS") ? getenv("LAZY_ALPHAS") : "0,0.05,0.1,0.15,0.2,0.3";
    for (size_t p = 0; p < alphas.size();) {
      size_t e = alphas.find(',', p);
      if (e == std::string::npos) e = alphas.size();
      const float a = (float)atof(alphas.substr(p, e - p).c_str());
      snprintf(nm, sizeof nm, "lazy d=%.2f a=%.2f", lazy_delta, a);
      scheds.push_back({nm, lazy_delta, a, false});
      p = e + 1;
    }
  }
  std::vector<Sched> total_sched(scheds.size());
  const char* step_csv = getenv("STEP_CSV");
  const int nb = maxstep / binw + 1;

  int qdim; size_t nq; std::vector<float> queries = readFvecs(argv[2], qdim, nq);
  if (getenv("NQ")) nq = std::min(nq, (size_t)atoll(getenv("NQ")));

  auto t0 = clk::now();
  auto index = Index<dist_t, int>::loadIndex(argv[1]);
  const size_t N = index->currentNumNodes();
  const int dim = (int)(index->dataSizeBytes() / sizeof(float));
  if (dim % m) { fprintf(stderr, "dim %d not divisible by PQ_M %d\n", dim, m); return 1; }
  const int sub_dim = dim / m;
  printf("[load] %.1fs nodes=%zu dim=%d | PQ m=%d sub_dim=%d | queries=%zu ef=%d K=%d T=%d\n",
         std::chrono::duration<double>(clk::now()-t0).count(), N, dim, m, sub_dim, nq, ef, K, threads);
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
  std::vector<float> resid(N);
  const size_t CHUNK = 65536;
  const uint32_t nchunks = (uint32_t)((N + CHUNK - 1) / CHUNK);
  auto t2 = clk::now();
  flatnav::executeInParallel(0, nchunks, (uint32_t)threads, [&](uint32_t ch) {
    size_t clo = (size_t)ch * CHUNK, chi = std::min(clo + CHUNK, N);
    for (size_t n = clo; n < chi; n++) {
      float r = 0.0f;
      for (int j = 0; j < m; j++) {
        const float* sub = nodeVec(n) + (size_t)j * sub_dim;
        const float* cb = &codebooks[(size_t)j * kCentroids * sub_dim];
        const int c = nearestCentroid(sub, cb, sub_dim);
        codes[n * m + j] = (uint8_t)c;
        r += l2(sub, cb + (size_t)c * sub_dim, sub_dim);  // ||x - c(x)||^2, separable over subspaces
      }
      resid[n] = r;
    }
  });
  printf("[encode] %zu nodes in %.1fs\n", N, std::chrono::duration<double>(clk::now()-t2).count());

  // Low-error gate: the `frac` of nodes with the smallest compression error use PQ.
  std::vector<float> sorted(resid);
  const size_t kth = std::min(N - 1, (size_t)(frac * N));
  std::nth_element(sorted.begin(), sorted.begin() + kth, sorted.end());
  const float low_thr = sorted[kth];
  std::vector<float>().swap(sorted);
  printf("[gate] PQ_FRAC=%.2f -> low-error threshold ||r||^2 < %.1f\n", frac, low_thr);
  fflush(stdout);

  // Exact traversal (gate off) with a LUT installed: every discovered neighbor carries both scores.
  index->setPQGate(codes.data(), (uint32_t)m, 0, 0);

  Acc total(nb, nk, nw);
  double total_steps = 0;
  std::mutex mu;
  const uint32_t BLOCK = 64;
  const uint32_t nblocks = (uint32_t)((nq + BLOCK - 1) / BLOCK);
  auto t3 = clk::now();
  flatnav::executeInParallel(0, nblocks, (uint32_t)threads, [&](uint32_t blk) {
    Acc acc(nb, nk, nw);
    double steps = 0;
    std::vector<float> lut((size_t)m * kCentroids);
    std::vector<SpecDisc> disc;
    std::vector<uint32_t> exp;
    std::vector<Cand> cands;
    std::unordered_map<uint32_t, uint32_t> where;  // node -> index in cands
    std::vector<float> minE, minP, exP, sufE;      // per-age bests for the safe-delay scan
    std::vector<Sched> sch(scheds.size());
    std::vector<uint32_t> ref;                     // the exact search's top-K node ids, sorted
    std::vector<std::pair<float, uint32_t>> ev;
    size_t qlo = (size_t)blk * BLOCK, qhi = std::min(qlo + BLOCK, nq);
    for (size_t i = qlo; i < qhi; i++) {
      buildLUT(&queries[i * qdim], codebooks.data(), m, sub_dim, lut.data());
      disc.clear();
      exp.clear();
      flatnav::tl_pq_lut = lut.data();
      flatnav::tl_spec_disc = &disc;
      flatnav::tl_spec_expand = &exp;
      index->search((const void*)&queries[i * qdim], K, ef);
      flatnav::tl_spec_expand = nullptr;
      flatnav::tl_spec_disc = nullptr;
      flatnav::tl_pq_lut = nullptr;
      steps += exp.size();

      // Replay the heap. disc is in step order; the entry node (exp[0]) is never in it.
      cands.clear();
      where.clear();
      size_t r = 0;
      for (size_t s = 0; s + 1 < exp.size(); s++) {
        auto it = where.find(exp[s]);
        if (it != where.end()) cands[it->second].popped = true;
        for (; r < disc.size() && disc[r].step == s; r++) {
          const SpecDisc& d = disc[r];
          where[d.node] = (uint32_t)cands.size();
          const bool low = resid[d.node] < low_thr, rnd = rndPick(d.node, frac);
          cands.push_back({d.node, d.step, d.exact, d.pq, d.thresh, d.exact < d.thresh, d.pq < d.thresh, low, rnd,
                           false});
          acc.disc++; acc.disc_low += low; acc.disc_rnd += rnd;
        }

        // Winners are compared by EXACT distance: duplicate vectors tie on it and the heap breaks
        // ties by node id, so any pick with the expanded node's exact distance is the same pick.
        float be = FLT_MAX, bm = FLT_MAX, bp = FLT_MAX, bl = FLT_MAX, br = FLT_MAX;
        float em = 0, ep = 0, el = 0, er = 0;
        // Top-w coverage: the true winner's rank in the mixed heap. It is out of reach at any width
        // if it is PQ-only and PQ rejected it; otherwise count the candidates ranked ahead of it
        // (duplicates of it -- same exact distance -- are not ahead).
        const Cand& tc = cands[where.at(exp[s + 1])];
        const float te = tc.exact;
        const bool t_old = (size_t)tc.step + delay <= s;
        const bool t_elig = t_old ? tc.in_e : tc.in_p;
        const float t_sc = t_old ? tc.exact : tc.pq;
        size_t ahead = 0;
        minE.assign(s + 1, FLT_MAX);
        minP.assign(s + 1, FLT_MAX);
        exP.assign(s + 1, 0.0f);
        for (const Cand& c : cands) {
          if (c.popped) continue;
          if (c.in_e && c.exact < be) be = c.exact;
          const size_t age = s - c.step;             // 0 = discovered by this expansion
          if (c.in_e && c.exact < minE[age]) minE[age] = c.exact;
          if (c.in_p && c.pq < minP[age]) { minP[age] = c.pq; exP[age] = c.exact; }
          const bool old = (size_t)c.step + delay <= s;  // validated: discovered >= DELAY steps ago
          if (old ? c.in_e : c.in_p) {
            const float sm = old ? c.exact : c.pq;
            if (sm < bm) { bm = sm; em = c.exact; }
            if (c.pq < bp) { bp = c.pq; ep = c.exact; }
            if (sm < t_sc && c.exact != te) ahead++;
          }
          // Gated mixed: a newest neighbor is scored PQ only if its gate selects it; otherwise it
          // was read exact inside the speculative step.
          const bool pl = !old && c.low, pr = !old && c.rnd;
          if (pl ? c.in_p : c.in_e) { const float sc = pl ? c.pq : c.exact; if (sc < bl) { bl = sc; el = c.exact; } }
          if (pr ? c.in_p : c.in_e) { const float sc = pr ? c.pq : c.exact; if (sc < br) { br = sc; er = c.exact; } }
        }
        const int b = std::min((int)((s + 1) / binw), nb - 1);
        acc.cov[(size_t)b * nw + (t_elig ? (int)std::min<size_t>(ahead, wmax) : wmax + 1)]++;
        acc.n[b]++;
        acc.sane[b] += (be == te);
        acc.win_pq[b] += ((size_t)tc.step + delay > s);  // true winner's residency <= DELAY
        acc.miss_mixed[b] += (em != te);
        acc.miss_allpq[b] += (ep != te);
        acc.miss_low[b] += (el != te);
        acc.miss_rnd[b] += (er != te);

        // Safe delay k*: the largest k <= SAFE_MAX such that delays 1..k ALL pick the true winner.
        // Under delay k a candidate is PQ-only iff its age < k, so the delay-k winner is the better
        // of the best validated candidate (age >= k, exact score) and the best PQ-only one (age < k,
        // PQ score) -- a suffix min and a running prefix min over ages give every k in one pass.
        sufE.assign(s + 2, FLT_MAX);
        for (size_t a = s + 1; a-- > 0;) sufE[a] = std::min(sufE[a + 1], minE[a]);
        float pbest = FLT_MAX, pex = 0.0f;
        int ksafe = 0;
        for (int k = 1; k <= safe_max; k++) {
          const size_t a = (size_t)k - 1;            // the age that turns PQ-only at delay k
          if (a <= s && minP[a] < pbest) { pbest = minP[a]; pex = exP[a]; }
          const float ve = (size_t)k <= s ? sufE[k] : FLT_MAX;
          if (ve == FLT_MAX && pbest == FLT_MAX) break;
          if ((ve < pbest ? ve : pex) != te) break;
          ksafe = k;
        }
        acc.safe[(size_t)b * nk + ksafe]++;
      }

      // Validation schedules (replaySchedule), scored against the exact search's own top-K: the K
      // smallest exact distances ever admitted to its beam.
      ev.clear();
      for (const Cand& c : cands) if (c.in_e) ev.push_back({c.exact, c.node});
      const size_t kk = std::min((size_t)K, ev.size());
      std::partial_sort(ev.begin(), ev.begin() + kk, ev.end());
      ref.clear();
      for (size_t j = 0; j < kk; j++) ref.push_back(ev[j].second);
      std::sort(ref.begin(), ref.end());
      for (size_t j = 0; j < scheds.size(); j++)
        replaySchedule(cands, where, exp, scheds[j].delta, scheds[j].alpha, scheds[j].eager, K, ref, sch[j]);
    }
    std::lock_guard<std::mutex> lk(mu);
    for (size_t j = 0; j < scheds.size(); j++) total_sched[j].add(sch[j]);
    total.merge(acc);
    total_steps += steps;
  });
  printf("[trace] %zu queries in %.1fs, mean %.1f expansions/query\n", nq,
         std::chrono::duration<double>(clk::now()-t3).count(), total_steps / nq);

  printf("\n=== Top-1 frontier miss rate of one-step PQ speculation (from a validated beam) ===\n");
  printf("# sanity = replayed exact argmin == node really expanded (must be ~100%%)\n");
  printf("# DELAY=%d: candidates discovered in the last %d step(s) are still PQ-only\n", delay, delay);
  printf("# win_pq = P(true winner is still PQ-only) = P(winner's residency <= DELAY)\n");
  printf("# mixed  = validated candidates exact, PQ-only ones PQ | all-PQ = every candidate PQ\n");
  printf("# low%%   = mixed, but only newest neighbors in the lowest-error %.0f%% use PQ (rest exact)\n",
         100.0 * frac);
  printf("# rnd%%   = same, with a random %.0f%% of nodes (control)\n", 100.0 * frac);
  printf("# PQ share of discovered neighbors: mixed 100%% | low %.1f%% | rnd %.1f%%\n\n",
         100.0 * total.disc_low / total.disc, 100.0 * total.disc_rnd / total.disc);
  printf("%-10s %12s %9s %9s %9s %9s %9s %9s\n", "steps", "predictions", "sanity", "win_pq", "mixed",
         "all-PQ", "low", "rnd");
  double n = 0, sane = 0, wp = 0, mm = 0, mp = 0, ml = 0, mr = 0;
  for (int b = 0; b < nb; b++) {
    if (total.n[b] == 0) continue;
    char lbl[32];
    if (b == nb - 1) snprintf(lbl, sizeof lbl, "%d+", b * binw);
    else snprintf(lbl, sizeof lbl, "%d:%d", b * binw, (b + 1) * binw);
    printf("%-10s %12.0f %8.3f%% %8.3f%% %8.3f%% %8.3f%% %8.3f%% %8.3f%%\n", lbl, total.n[b],
           100.0 * total.sane[b] / total.n[b], 100.0 * total.win_pq[b] / total.n[b],
           100.0 * total.miss_mixed[b] / total.n[b], 100.0 * total.miss_allpq[b] / total.n[b],
           100.0 * total.miss_low[b] / total.n[b], 100.0 * total.miss_rnd[b] / total.n[b]);
    n += total.n[b]; sane += total.sane[b]; wp += total.win_pq[b]; mm += total.miss_mixed[b];
    mp += total.miss_allpq[b]; ml += total.miss_low[b]; mr += total.miss_rnd[b];
  }
  printf("%-10s %12.0f %8.3f%% %8.3f%% %8.3f%% %8.3f%% %8.3f%% %8.3f%%\n", "all", n, 100.0 * sane / n,
         100.0 * wp / n, 100.0 * mm / n, 100.0 * mp / n, 100.0 * ml / n, 100.0 * mr / n);

  printf("\n=== Safe validation delay k* per step: largest k with delays 1..k all correct (cap %d) ===\n",
         safe_max);
  printf("# k*=0: even a 1-step delay misses here | >=k: share of steps where delay k needs no redo\n\n");
  printf("%-10s %9s %9s %9s %9s %9s %9s\n", "steps", "k*=0", ">=1", ">=2", ">=4", ">=8", "mean k*");
  std::vector<double> all_safe(nk, 0.0);
  auto safeRow = [&](const char* lbl, const double* h) {
    double tot = 0, mean = 0;
    for (int k = 0; k < nk; k++) { tot += h[k]; mean += k * h[k]; }
    auto atLeast = [&](int k) { double c = 0; for (int j = std::min(k, nk); j < nk; j++) c += h[j]; return c; };
    printf("%-10s %8.2f%% %8.2f%% %8.2f%% %8.2f%% %8.2f%% %9.2f\n", lbl, 100.0 * h[0] / tot,
           100.0 * atLeast(1) / tot, 100.0 * atLeast(2) / tot, 100.0 * atLeast(4) / tot,
           100.0 * atLeast(8) / tot, mean / tot);
  };
  for (int b = 0; b < nb; b++) {
    if (total.n[b] == 0) continue;
    char lbl[32];
    if (b == nb - 1) snprintf(lbl, sizeof lbl, "%d+", b * binw);
    else snprintf(lbl, sizeof lbl, "%d:%d", b * binw, (b + 1) * binw);
    safeRow(lbl, &total.safe[(size_t)b * nk]);
    for (int k = 0; k < nk; k++) all_safe[k] += total.safe[(size_t)b * nk + k];
  }
  safeRow("all", all_safe.data());

  printf("\n=== Top-w coverage: P(true winner among the mixed heap's top w) (DELAY=%d) ===\n", delay);
  printf("# speculating the top w candidates per step misses only if the winner ranks below w\n");
  printf("# rejected = winner is PQ-only and PQ kept it out of the heap: no width recovers it\n\n");
  const int ws[] = {1, 2, 3, 4, 6, 8, 16};
  printf("%-10s", "steps");
  for (int w : ws) if (w <= wmax) printf("  %6s%-2d", "w=", w);
  printf(" %9s\n", "rejected");
  std::vector<double> all_cov(nw, 0.0);
  auto covRow = [&](const char* lbl, const double* h) {
    double tot = 0;
    for (int r = 0; r < nw; r++) tot += h[r];
    printf("%-10s", lbl);
    for (int w : ws) {
      if (w > wmax) continue;
      double c = 0;
      for (int r = 0; r < w; r++) c += h[r];
      printf("  %7.2f%%", 100.0 * c / tot);
    }
    printf(" %8.2f%%\n", 100.0 * h[wmax + 1] / tot);
  };
  for (int b = 0; b < nb; b++) {
    if (total.n[b] == 0) continue;
    char lbl[32];
    if (b == nb - 1) snprintf(lbl, sizeof lbl, "%d+", b * binw);
    else snprintf(lbl, sizeof lbl, "%d:%d", b * binw, (b + 1) * binw);
    covRow(lbl, &total.cov[(size_t)b * nw]);
    for (int r = 0; r < nw; r++) all_cov[r] += total.cov[(size_t)b * nw + r];
  }
  covRow("all", all_cov.data());

  if (cov_csv) {
    FILE* f = fopen(cov_csv, "w");
    fprintf(f, "step_lo,step_hi,rank,count\n");  // rank WMAX+1 = beyond WMAX, WMAX+2 = rejected
    for (int b = 0; b < nb; b++)
      for (int r = 0; r < nw; r++)
        if (total.cov[(size_t)b * nw + r] > 0)
          fprintf(f, "%d,%d,%d,%.0f\n", b * binw, b == nb - 1 ? -1 : (b + 1) * binw, r + 1,
                  total.cov[(size_t)b * nw + r]);
    fclose(f);
  }

  if (safe_csv) {
    FILE* f = fopen(safe_csv, "w");
    fprintf(f, "step_lo,step_hi,k_safe,count\n");
    for (int b = 0; b < nb; b++)
      for (int k = 0; k < nk; k++)
        if (total.safe[(size_t)b * nk + k] > 0)
          fprintf(f, "%d,%d,%d,%.0f\n", b * binw, b == nb - 1 ? -1 : (b + 1) * binw, k,
                  total.safe[(size_t)b * nk + k]);
    fclose(f);
  }

  if (step_csv) {
    FILE* f = fopen(step_csv, "w");
    fprintf(f, "step_lo,step_hi,predictions,sanity,win_pq,miss_mixed,miss_allpq,miss_low,miss_rnd\n");
    for (int b = 0; b < nb; b++)
      if (total.n[b] > 0)
        fprintf(f, "%d,%d,%.0f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", b * binw, b == nb - 1 ? -1 : (b + 1) * binw,
                total.n[b], total.sane[b] / total.n[b], total.win_pq[b] / total.n[b],
                total.miss_mixed[b] / total.n[b], total.miss_allpq[b] / total.n[b],
                total.miss_low[b] / total.n[b], total.miss_rnd[b] / total.n[b]);
    fclose(f);
  }

  printf("\n=== Validation schedules on the exact trajectory (K=%d) ===\n", K);
  printf("# reads/q    = exact vector reads per query (traversal + final top-K certification)\n");
  printf("# read_steps = share of steps issuing >= 1 read; a step with none has no validation latency\n");
  printf("# step_miss  = share of steps whose pick differs from the exact search's next node\n");
  printf("# top-K      = share of the exact search's top-K the schedule returns\n\n");
  printf("%-24s %9s %9s %8s %11s %10s %8s\n", "schedule", "reads/q", "(final)", "%exact", "read_steps",
         "step_miss", "top-K");
  const double exact_reads = total_sched[0].reads / total_sched[0].q;
  for (size_t j = 0; j < scheds.size(); j++) {
    const Sched& t = total_sched[j];
    printf("%-24s %9.1f %9.1f %7.1f%% %10.2f%% %9.2f%% %7.2f%%\n", scheds[j].name.c_str(), t.reads / t.q,
           t.reads_final / t.q, 100.0 * (t.reads / t.q) / exact_reads, 100.0 * t.read_steps / t.steps,
           100.0 * t.miss / t.steps, 100.0 * t.overlap / t.q);
  }
  return 0;
}
