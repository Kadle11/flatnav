#pragma once

// Access-trace capture for the batched search path (concurrentBatchSearch).
//
// Logs one record per GraphRead in processOneLink: (query_id, node_id, hop).
// Grouping by query_id reconstructs each query's visited sequence; (query_id,
// hop) lets the offline analysis bucket by first-fetch hop and simulate the
// C-slot scheduler for any concurrency.
//
// STREAMS to disk with a bounded per-thread buffer, so resident memory is
// O(buffer) (~3 MB/thread), NOT O(total accesses) -- safe at billion scale.
//
// Zero cost unless built with -DFLATNAV_TRACE_ACCESS. Output path:
// $FLATNAV_TRACE_PATH (default "flatnav_access.trace"), opened once per process
// (truncating). Format: 12 bytes/record = three little-endian uint32. Records
// from concurrent threads are interleaved at flush granularity but never torn;
// the offline analysis groups by query_id so order does not matter.

#include <cstddef>
#include <cstdint>

#if defined(FLATNAV_TRACE_ACCESS)
#include <cstdio>
#include <cstdlib>
#include <mutex>
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

// Records buffered per thread before a single batched write (~3 MB/thread).
constexpr size_t kFlushRecords = 1u << 18;

namespace detail {

inline std::mutex& file_mutex() {
  static std::mutex m;
  return m;
}

inline std::FILE*& file_handle() {
  static std::FILE* f = nullptr;
  return f;
}

inline thread_local std::vector<AccessRecord> tls_buf;

// Append this thread's buffer to the shared file and clear it. The vector keeps
// its capacity, so steady-state memory is bounded by kFlushRecords.
inline void drain() {
  if (tls_buf.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(file_mutex());
  std::FILE*& f = file_handle();
  if (!f) {
    const char* p = std::getenv("FLATNAV_TRACE_PATH");
    f = std::fopen(p ? p : "flatnav_access.trace", "wb");
  }
  if (f) {
    std::fwrite(tls_buf.data(), sizeof(AccessRecord), tls_buf.size(), f);
  }
  tls_buf.clear();
}

}  // namespace detail

inline void record(uint32_t query_id, uint32_t node_id, uint32_t hop) {
  auto& buf = detail::tls_buf;
  buf.push_back({query_id, node_id, hop});
  if (buf.size() >= kFlushRecords) {
    detail::drain();
  }
}

inline void reserve(size_t) {}  // kept for API compat; streaming makes it moot

// Flush remaining buffered records and sync to disk. Called at the end of a
// search; each recording thread drains its own buffer here.
inline void flushDefault() {
  detail::drain();
  std::lock_guard<std::mutex> lock(detail::file_mutex());
  if (detail::file_handle()) {
    std::fflush(detail::file_handle());
  }
}

#define FN_TRACE_ACCESS(qid, nid, hop)                  \
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
