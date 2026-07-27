#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace engram {

struct StreamingAttentionConfig {
  std::size_t query_heads{};
  std::size_t key_value_heads{};
  std::size_t head_dimension{};
  std::size_t local_window{16};
  std::size_t older_candidates{8};
  std::size_t older_top_k{4};
  std::size_t sink_tokens{2};
  float scale{};
};

struct StreamingAttentionMetrics {
  std::uint64_t tokens_seen{};
  std::uint64_t local_entries{};
  std::uint64_t active_older_entries{};
  std::uint64_t candidate_key_bytes{};
  std::uint64_t selected_value_bytes{};
  std::uint64_t local_kv_bytes{};
  std::uint64_t eviction_events{};
  std::uint64_t older_candidate_entries_scored{};
  std::uint64_t older_selected_entries{};
  std::uint64_t sink_insertions{};
  std::uint64_t heavy_hitter_updates{};
  std::uint64_t state_bytes{};
  std::uint64_t scratch_bytes{};
};

// Stateful causal attention with an exact local window, fixed attention sinks,
// and a bounded per-query-head cumulative-attention heavy-hitter cache.
class StreamingAttention {
 public:
  explicit StreamingAttention(StreamingAttentionConfig config);

  void reset() noexcept;
  StreamingAttentionMetrics step(std::span<const float> query,
                                 std::span<const float> key,
                                 std::span<const float> value,
                                 std::span<float> output);

  [[nodiscard]] const StreamingAttentionConfig& config() const noexcept;
  [[nodiscard]] std::size_t tokens_seen() const noexcept;
  [[nodiscard]] std::size_t active_older_entries() const noexcept;
  [[nodiscard]] std::size_t allocated_state_bytes() const noexcept;
  [[nodiscard]] std::size_t scratch_bytes() const noexcept;

 private:
  [[nodiscard]] std::size_t recent_offset(std::size_t slot,
                                          std::size_t kv_head) const noexcept;
  [[nodiscard]] std::size_t older_offset(std::size_t head,
                                         std::size_t slot) const noexcept;
  void evict_recent(std::size_t slot, std::uint64_t& sink_insertions,
                    std::uint64_t& heavy_hitter_updates);
  void validate_inputs(std::span<const float> query,
                       std::span<const float> key,
                       std::span<const float> value,
                       std::span<float> output) const;

  StreamingAttentionConfig config_;
  std::size_t groups_{};
  std::size_t tokens_seen_{};
  std::size_t recent_start_{};
  std::size_t recent_size_{};

  std::vector<float> recent_keys_;
  std::vector<float> recent_values_;
  std::vector<float> recent_mass_;
  std::vector<std::uint64_t> recent_positions_;

  std::vector<float> older_keys_;
  std::vector<float> older_values_;
  std::vector<float> older_scores_;
  std::vector<std::uint64_t> older_positions_;
  std::vector<std::uint8_t> older_active_;

  std::vector<float> score_scratch_;
  std::vector<float> candidate_score_scratch_;
  std::vector<float> weight_scratch_;
  std::vector<std::size_t> selected_scratch_;
};

}  // namespace engram
