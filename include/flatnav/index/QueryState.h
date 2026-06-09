#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_set>
#include <vector>

namespace flatnav {

enum class QueryExecutionState : uint8_t {
  Unscheduled,
  ProcessingLinks,
  Done
};


template <typename DistNode, typename node_id_t>
struct QueryState {
    const void* query;
    int K;
    size_t buffer_size;
    std::vector<DistNode> neighbors;
    std::vector<DistNode> candidates;
    std::unordered_set<node_id_t> visited;
    QueryExecutionState execution_state;
    float max_dist;
    node_id_t* current_links;
    uint32_t link_idx;
    uint32_t query_id = 0;  // index of the query this slot serves (access tracing)
    uint32_t hop = 0;       // expansion count within this query (access tracing)


    void resetForQuery(const void* new_query,
                       std::optional<int> new_K = std::nullopt,
                       std::optional<size_t> new_buffer_size = std::nullopt) {
      query = new_query;
      if (new_K.has_value()) {
        K = *new_K;
      }
      if (new_buffer_size.has_value()) {
        buffer_size = *new_buffer_size;
      }
      neighbors.clear();
      candidates.clear();
      visited.clear();
      execution_state = QueryExecutionState::Unscheduled;
      max_dist = std::numeric_limits<float>::max();
      current_links = nullptr;
      link_idx = 0;
      hop = 0;
    }
};
}