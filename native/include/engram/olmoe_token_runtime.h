#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "engram/olmoe_q7.h"
#include "engram/olmoe_weights.h"
#include "engram/streaming_attention.h"
#include "engram/thread_pool.h"

namespace engram {

struct OLMoETokenConfig {
  std::filesystem::path non_mlp_safetensors;
  std::filesystem::path q7_artifact;
  std::size_t layers{};
  std::size_t hidden_size{};
  std::size_t query_heads{};
  std::size_t key_value_heads{};
  std::size_t head_dimension{};
  std::size_t threads{1};
  std::size_t local_window{16};
  std::size_t older_candidates{8};
  std::size_t older_top_k{4};
  std::size_t sink_tokens{2};
  float rms_norm_epsilon{1.0e-5F};
  float rope_theta{10000.0F};
  std::vector<std::int64_t> eos_token_ids;
};

struct OLMoETokenMetrics {
  std::uint64_t positions_processed{};
  std::uint64_t attention_weight_bytes{};
  std::uint64_t attention_logical_read_bytes{};
  std::uint64_t attention_scratch_bytes{};
  std::uint64_t attention_eviction_events{};
  std::uint64_t attention_older_candidate_entries_scored{};
  std::uint64_t attention_older_selected_entries{};
  std::uint64_t attention_sink_insertions{};
  std::uint64_t attention_heavy_hitter_updates{};
  std::uint64_t q7_scheduled_bytes{};
  std::uint64_t q7_elapsed_ns{};
  std::uint64_t attention_state_bytes{};
  std::uint64_t elapsed_ns{};
};

// Transformer-shell-free CPU token runtime for mapped OLMoE non-MLP weights
// and the native packed-Q7 expert artifact.
class OLMoETokenRuntime {
 public:
  explicit OLMoETokenRuntime(OLMoETokenConfig config);

  [[nodiscard]] std::int64_t forward(
      std::span<const std::int64_t> token_ids);
  [[nodiscard]] std::vector<std::int64_t> generate(
      std::span<const std::int64_t> prompt, std::size_t max_new_tokens);
  void reset();

  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  [[nodiscard]] std::size_t vocabulary_size() const noexcept {
    return weights_.vocabulary_size();
  }
  [[nodiscard]] const OLMoETokenMetrics& metrics() const noexcept {
    return metrics_;
  }
  [[nodiscard]] bool has_diagnostics() const noexcept {
    return !last_final_state_.empty() && !last_vocabulary_scores_.empty();
  }
  [[nodiscard]] std::span<const float> last_final_state() const noexcept {
    return last_final_state_;
  }
  [[nodiscard]] std::span<const float>
  last_vocabulary_scores() const noexcept {
    return last_vocabulary_scores_;
  }

 private:
  void project(std::span<const float> input,
               std::span<const std::uint16_t> weight, std::size_t rows,
               std::size_t input_width, std::size_t output_width,
               std::span<float> output);
  void normalize(std::span<const float> input,
                 std::span<const std::uint16_t> weight, std::size_t rows,
                 std::size_t width, std::span<float> output) const;
  void apply_rope(std::span<float> values, std::size_t heads,
                  std::size_t position) const;
  [[nodiscard]] bool is_eos(std::int64_t token) const;

  OLMoETokenConfig config_;
  OLMoEWeights weights_;
  OLMoEQ7Kernel q7_;
  ThreadPool pool_;
  std::vector<std::unique_ptr<StreamingAttention>> attention_;
  std::vector<float> last_final_state_;
  std::vector<float> last_vocabulary_scores_;
  std::size_t position_{};
  OLMoETokenMetrics metrics_{};
};

}  // namespace engram
