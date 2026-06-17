// Builds a FlatNav index from a SIFT .fvecs base file plus a prebuilt HNSW
// base-layer graph in Matrix Market (.mtx) format, then serializes it.
//
//   build_sift100m <base.fvecs> <graph.mtx> <out.bin> [M=32]
//
// Vector i is given label i (file order), which matches the node ids used by
// the .mtx graph (1-indexed there) and the bigann ground-truth indices.
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/index/Index.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

using flatnav::Index;
using flatnav::distances::SquaredL2Distance;
using flatnav::util::DataType;
using dist_t = SquaredL2Distance<DataType::float32>;
using clk = std::chrono::steady_clock;

static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <base.fvecs> <graph.mtx> <out.bin> [M=32] [--probe]\n", argv[0]);
    return 1;
  }
  const char* fvecs_path = argv[1];
  const char* mtx_path = argv[2];
  const char* out_path = argv[3];
  int M = (argc > 4 && argv[4][0] != '-') ? atoi(argv[4]) : 32;
  bool probe = (argc > 4 && std::string(argv[argc - 1]) == "--probe");

  // mmap the .fvecs base file. Each record: [int32 dim][dim float32].
  int fd = open(fvecs_path, O_RDONLY);
  if (fd < 0) { perror("open fvecs"); return 1; }
  struct stat st;
  if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
  size_t fsize = st.st_size;
  void* map = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) { perror("mmap"); return 1; }
  madvise(map, fsize, MADV_SEQUENTIAL);
  const char* base = static_cast<const char*>(map);

  int dim = *reinterpret_cast<const int32_t*>(base);
  size_t rec = 4 + static_cast<size_t>(dim) * 4;
  size_t n = fsize / rec;
  printf("[fvecs] dim=%d n=%zu rec_bytes=%zu M=%d\n", dim, n, rec, M);
  fflush(stdout);
  if (probe) { printf("[probe] exiting before build\n"); return 0; }

  auto dist = dist_t::create(dim);
  Index<dist_t, int> index(std::move(dist), static_cast<int>(n), M);

  // 1. Place all vectors (data + label, links init to self). No graph build.
  auto t0 = clk::now();
  for (size_t i = 0; i < n; i++) {
    const float* v = reinterpret_cast<const float*>(base + i * rec + 4);
    int label = static_cast<int>(i);
    uint32_t id;
    index.allocateNode(reinterpret_cast<void*>(const_cast<float*>(v)), label, id);
    if ((i & 0x3FFFFF) == 0) { printf("\r[alloc] %zu/%zu", i, n); fflush(stdout); }
  }
  printf("\r[alloc] %zu/%zu done in %.1fs\n", n, n, secs(t0, clk::now()));
  fflush(stdout);

  // 2. Load the prebuilt graph links from the .mtx (this is the slow part:
  //    ~3.2B edge pairs parsed via iostream).
  printf("[graph] parsing %s ...\n", mtx_path);
  fflush(stdout);
  auto g0 = clk::now();
  index.buildGraphLinks(mtx_path);
  printf("[graph] loaded in %.1fs\n", secs(g0, clk::now()));
  fflush(stdout);

  // 3. Serialize.
  printf("[save] writing %s ...\n", out_path);
  fflush(stdout);
  auto s0 = clk::now();
  index.saveIndex(out_path);
  printf("[save] done in %.1fs\n", secs(s0, clk::now()));
  fflush(stdout);

  munmap(map, fsize);
  close(fd);
  return 0;
}
