#pragma once

// Access-trace capture for the batched search path (concurrentBatchSearch).
//
// Logs one record per GraphRead (neighbor data fetch in processOneLink):
//     (query_id, node_id, hop)
// Grouping by query_id reconstructs each query's visited sequence seq_q; the
// (query_id, hop) lets the offline analysis bucket by first-fetch hop and
// simulate the C-slot scheduler for any concurrency. Per-query sequences are
// concurrency-independent, so one single-threaded run suffices.
//
// Zero cost unless built with -DFLATNAV_TRACE_ACCESS. The buffer is
// thread_local; flushDefault() appends it (binary, 12 bytes/record) to the path
// in $FLATNAV_TRACE_PATH (default "flatnav_access.trace"). Capture
// single-threaded for a clean global trace.

#include <cstdint>

#if defined(FLATNAV_TRACE_ACCESS)
#include <cstdio>
#include <cstdlib>
#include <vector>
#endif

namespace flatnav {
namespace tracing {

#if defined(FLATNAV_TRACE_ACCESS)

struct AccessRecord {
  uint32_t query_id;
  uint32_t node_id;
  uint32_t hop;
};

inline thread_local std::vector<AccessRecord> g_trace;

inline void record(uint32_t query_id, uint32_t node_id, uint32_t hop) {
  g_trace.push_back({query_id, node_id, hop});
}

inline void reserve(size_t n) { g_trace.reserve(n); }

// Append this thread's buffer to `path` and clear it.
inline void flush(const char* path) {
  if (g_trace.empty()) {
    return;
  }
  std::FILE* f = std::fopen(path, "ab");
  if (f) {
    std::fwrite(g_trace.data(), sizeof(AccessRecord), g_trace.size(), f);
    std::fclose(f);
  }
  g_trace.clear();
}

inline void flushDefault() {
  const char* p = std::getenv("FLATNAV_TRACE_PATH");
  flush(p ? p : "flatnav_access.trace");
}

#define FN_TRACE_ACCESS(qid, nid, hop) \
  ::flatnav::tracing::record(static_cast<uint32_t>(qid), \
                             static_cast<uint32_t>(nid), \
                             static_cast<uint32_t>(hop))

#else  // tracing disabled -> no-ops

inline void reserve(size_t) {}
inline void flushDefault() {}
#define FN_TRACE_ACCESS(qid, nid, hop) ((void)0)

#endif

}  // namespace tracing
}  // namespace flatnav
