// Tiered-placement QPS benchmark: cache the best LOCAL_FRAC of nodes (whole node =
// vector+links) on the local NUMA node, the rest on remote, per a placement POLICY.
//
//   tier_run <index.bin> <query.fvecs> <gt.ivecs> [K=100] [threads=32] [ef=200]
//
// env:
//   POLICY=sssp|hub      sssp = hop-proximity to SOURCE (closest LOCAL_FRAC local;
//                        search starts from SOURCE); hub = top in-degree local (per-query)
//   LOCAL_FRAC=0.6       fraction of nodes (and thus index bytes) kept local
//   SOURCE=medoid|<id>   common source for POLICY=sssp (default medoid)
//   LOCAL_NODE=0 REMOTE_NODE=1
//
// Run pinned to the local node:  numactl --cpunodebind=0 ./tier_run ...
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/index/Index.h>
#include <flatnav/util/Multithreading.h>
#include <flatnav/util/NumaThreadPool.h>

#include <fcntl.h>
#include <numaif.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

using flatnav::Index;
using flatnav::distances::SquaredL2Distance;
using flatnav::util::DataType;
using dist_t = SquaredL2Distance<DataType::float32>;
using clk = std::chrono::steady_clock;

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

// Bind [off, off+len) of `base` to `node`, page-aligning the start.
static void bindRange(char* base, size_t off, size_t len, int node) {
  if (len == 0) return;
  uintptr_t start = (uintptr_t)(base + off);
  uintptr_t astart = start & ~(uintptr_t)4095;
  size_t alen = len + (start - astart);
  unsigned long mask = 1UL << node;
  if (mbind((void*)astart, alen, MPOL_BIND, &mask, sizeof(mask)*8, MPOL_MF_MOVE) != 0)
    perror("mbind");
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <index.bin> <query.fvecs> <gt.ivecs> [K=100] [threads=32] [ef=200]\n", argv[0]);
    return 1;
  }
  const char* index_path = argv[1];
  int K = argc > 4 ? atoi(argv[4]) : 100;
  int threads = argc > 5 ? atoi(argv[5]) : 32;
  int ef = argc > 6 ? atoi(argv[6]) : 200;
  std::string policy = getenv("POLICY") ? getenv("POLICY") : "hub";
  std::string lf_str = getenv("LOCAL_FRAC") ? getenv("LOCAL_FRAC") : "0.6";
  std::vector<double> fracs;
  for (size_t p = 0; p < lf_str.size(); ) {
    size_t c = lf_str.find(',', p);
    std::string tok = lf_str.substr(p, c == std::string::npos ? c : c - p);
    if (!tok.empty()) fracs.push_back(atof(tok.c_str()));
    if (c == std::string::npos) break; p = c + 1;
  }
  int local_node = getenv("LOCAL_NODE") ? atoi(getenv("LOCAL_NODE")) : 0;
  int remote_node = getenv("REMOTE_NODE") ? atoi(getenv("REMOTE_NODE")) : 1;
  const char* src_env = getenv("SOURCE");

  int qdim; size_t nq; std::vector<float> queries = readFvecs(argv[2], qdim, nq);
  int gw; size_t ngt; std::vector<int> gt = readIvecs(argv[3], gw, ngt);

  auto t0 = clk::now();
  auto index = Index<dist_t, int>::loadIndex(index_path);
  size_t N = index->currentNumNodes();
  printf("[load] %.1fs nodes=%zu | policy=%s local_frac=%s local=node%d remote=node%d ef=%d K=%d T=%d\n",
         std::chrono::duration<double>(clk::now()-t0).count(), N, policy.c_str(),
         lf_str.c_str(), local_node, remote_node, ef, K, threads);
  fflush(stdout);

  // --- 1. ranking -> permutation P (P[old]=new; best nodes get low ids) ---
  std::vector<uint32_t> order(N);
  std::iota(order.begin(), order.end(), 0u);
  long S = -1;
  if (policy == "sssp") {
    S = (src_env && strcmp(src_env, "medoid") != 0) ? atol(src_env) : (long)index->computeMedoid();
    auto t = clk::now();
    std::vector<uint8_t> hop = index->bfsHopDistances((uint32_t)S);
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return hop[a] < hop[b]; });
    printf("[rank] sssp source S=%ld, hop-proximity order in %.1fs (S->new id 0)\n",
           S, std::chrono::duration<double>(clk::now()-t).count());
  } else { // hub
    auto t = clk::now();
    std::vector<uint32_t> indeg = index->computeInDegrees();
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return indeg[a] > indeg[b]; });
    printf("[rank] hub in-degree order in %.1fs\n",
           std::chrono::duration<double>(clk::now()-t).count());
  }
  std::vector<uint32_t> P(N);
  for (size_t r = 0; r < N; r++) P[order[r]] = (uint32_t)r;

  // --- 2. relabel so the local set is the contiguous low-id prefix ---
  auto t1 = clk::now();
  index->reorderByPermutation(P);
  printf("[relabel] %.1fs\n", std::chrono::duration<double>(clk::now()-t1).count());
  fflush(stdout);

  // search mode (fraction-independent): sssp -> common source (relabeled S = id 0).
  if (policy == "sssp") index->setFixedEntryNode(0);
  std::vector<std::unordered_set<int>> gtTopK(nq);
  for (size_t i = 0; i < nq; i++)
    for (int j = 0; j < K && j < gw; j++) gtTopK[i].insert(gt[i*gw+j]);
  size_t ds = index->dataSizeBytes(), gs = index->graphNodeSizeBytes();
  std::vector<std::vector<std::pair<float,int>>> results(nq);
  auto runBatch = [&](int e) {
    flatnav::executeInParallel(0, nq, threads, [&](uint32_t i){
      results[i] = index->search((const void*)&queries[(size_t)i*qdim], K, e);
    });
  };

  // Sweep local fractions: re-mbind the split, then warmup + timed search.
  for (double local_frac : fracs) {
    size_t split = (size_t)(local_frac * N);
    bindRange(index->vectorsMemory(), 0,        split*ds,     local_node);
    bindRange(index->vectorsMemory(), split*ds, (N-split)*ds, remote_node);
    bindRange(index->graphMemory(),   0,        split*gs,     local_node);
    bindRange(index->graphMemory(),   split*gs, (N-split)*gs, remote_node);
    printf("[place] local_frac=%.2f split=%zu local=%.1fGB remote=%.1fGB\n", local_frac, split,
           (double)split*(ds+gs)/1e9, (double)(N-split)*(ds+gs)/1e9);
    fflush(stdout);

    runBatch(ef); // warmup (faults migrated pages onto their new node)
    auto ts = clk::now();
    runBatch(ef);
    double dt = std::chrono::duration<double>(clk::now()-ts).count();
    size_t hits = 0, total = 0;
    for (size_t i = 0; i < nq; i++) {
      for (auto& pr : results[i]) if (gtTopK[i].count(pr.second)) hits++;
      total += std::min((size_t)K, gtTopK[i].size());
    }
    printf("RESULT policy=%s local_frac=%.2f ef=%d  QPS=%.0f  recall@%d=%.4f  time=%.2fs\n",
           policy.c_str(), local_frac, ef, nq/dt, K, (double)hits/total, dt);
    fflush(stdout);
  }
  return 0;
}
