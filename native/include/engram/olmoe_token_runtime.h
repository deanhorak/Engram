#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "engram/olmoe_q7.h"
#include "engram/olmoe_weights.h"
#include "engram/streaming_attention.h"
#include "engram/thread_pool.h"

namespace engram {

struct OLMoEAttentionPolicy {
  std::size_t local_window{16};
  std::size_t older_candidates{8};
  std::size_t older_top_k{4};
  std::size_t sink_tokens{2};
};

struct OLMoEEpisodicPolicy {
  std::size_t slots{};
  std::size_t span_size{};
};

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
  // Empty selects the legacy scalar capacities above. Otherwise this must
  // contain exactly one policy for every transformer layer.
  std::vector<OLMoEAttentionPolicy> attention_policies;
  // Additive head-wise mode. When non-empty, this must contain exactly
  // layers * query_heads policies in layer-major/head-minor order. Head-wise
  // mode uses one independent streaming-attention cache per head and requires
  // query_heads == key_value_heads.
  std::vector<OLMoEAttentionPolicy> head_attention_policies;
  // Optional causal episodic K/V bank. This capacity experiment is supported
  // only with the scalar grouped attention policy above.
  OLMoEEpisodicPolicy episodic_policy;
  // Optional layer-major/query-head-minor 0/1 mask for episodic reads. Empty
  // preserves the original all-layer/all-head episodic behavior.
  std::vector<std::uint8_t> episodic_head_mask;
  // Additive attention-logit bias for selected episodic entries. The public
  // C ABI exposes this only through the additive head-wise V2 open.
  float episodic_logit_bias{};
  // Optional evaluator-only same-state attention teacher. One independent
  // non-episodic cache is allocated per layer. Its outputs never enter the
  // base hidden state and its state/traffic are excluded from base metrics.
  std::optional<OLMoEAttentionPolicy> shadow_attention_policy;
  // Additive evaluator-only capture of the production C28 RoPE-closed Q/K
  // band trace. This is enabled only by the versioned shadow-trace V2 C ABI.
  bool c28_qk_partial_trace{};
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
  // Writes, active slots, and read events count active layer caches. Entry
  // and read-byte counters count only selected query heads. Write bytes and
  // state capacity include full grouped K/V rows for every active layer.
  std::uint64_t episodic_slots_written{};
  std::uint64_t episodic_read_events{};
  std::uint64_t episodic_active_slots{};
  std::uint64_t episodic_entries_read{};
  std::uint64_t episodic_write_bytes{};
  std::uint64_t episodic_key_read_bytes{};
  std::uint64_t episodic_value_read_bytes{};
  std::uint64_t episodic_duplicate_older_entries_suppressed{};
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
  [[nodiscard]] std::int64_t forward_episodic(
      std::span<const std::int64_t> token_ids,
      std::span<const std::int32_t> write_slots,
      std::span<const std::int32_t> read_spans);
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
  [[nodiscard]] bool has_shadow_trace() const noexcept {
    return shadow_trace_valid_;
  }
  [[nodiscard]] std::span<const float>
  last_shadow_input_norm() const noexcept {
    return last_shadow_input_norm_;
  }
  [[nodiscard]] std::span<const float>
  last_shadow_base_projected() const noexcept {
    return last_shadow_base_projected_;
  }
  [[nodiscard]] std::span<const float>
  last_shadow_target_residual() const noexcept {
    return last_shadow_target_residual_;
  }
  [[nodiscard]] bool has_episodic_mass_trace() const noexcept {
    return episodic_mass_trace_valid_;
  }
  [[nodiscard]] std::span<const float>
  last_episodic_mass_base_pre_wo() const noexcept {
    return last_episodic_mass_base_pre_wo_;
  }
  [[nodiscard]] std::span<const float>
  last_episodic_mass_regular_component() const noexcept {
    return last_episodic_mass_regular_component_;
  }
  [[nodiscard]] std::span<const float>
  last_episodic_mass_episodic_component() const noexcept {
    return last_episodic_mass_episodic_component_;
  }
  [[nodiscard]] std::span<const float>
  last_episodic_mass_regular_mass() const noexcept {
    return last_episodic_mass_regular_mass_;
  }
  [[nodiscard]] std::span<const float>
  last_episodic_mass_episodic_mass() const noexcept {
    return last_episodic_mass_episodic_mass_;
  }
  [[nodiscard]] std::span<const float>
  last_episodic_mass_shadow_source_mass() const noexcept {
    return last_episodic_mass_shadow_source_mass_;
  }
  [[nodiscard]] bool has_episodic_slot_trace() const noexcept {
    return episodic_slot_trace_valid_;
  }
  [[nodiscard]] std::span<const float>
  last_episodic_slot_mass() const noexcept {
    return last_episodic_slot_mass_;
  }
  [[nodiscard]] std::span<const float>
  last_episodic_slot_values() const noexcept {
    return last_episodic_slot_values_;
  }
  [[nodiscard]] bool has_regular_entry_trace() const noexcept {
    return regular_entry_trace_valid_;
  }
  [[nodiscard]] std::span<const float>
  last_regular_entry_mass() const noexcept {
    return last_regular_entry_mass_;
  }
  [[nodiscard]] std::span<const float>
  last_regular_entry_values() const noexcept {
    return last_regular_entry_values_;
  }
  [[nodiscard]] std::span<const std::uint8_t>
  last_regular_entry_valid_kind() const noexcept {
    return last_regular_entry_valid_kind_;
  }
  [[nodiscard]] std::span<const std::uint64_t>
  last_regular_entry_positions() const noexcept {
    return last_regular_entry_positions_;
  }
  [[nodiscard]] bool has_c28_qk_partial_trace() const noexcept {
    return c28_qk_partial_trace_valid_;
  }
  [[nodiscard]] std::span<const float>
  last_c28_qk_partials() const noexcept {
    return last_c28_qk_partials_;
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
  [[nodiscard]] std::int64_t forward_impl(
      std::span<const std::int64_t> token_ids,
      std::span<const std::int32_t> write_slots,
      std::span<const std::int32_t> read_spans,
      std::span<const std::uint64_t> read_source_positions);

  OLMoETokenConfig config_;
  OLMoEWeights weights_;
  OLMoEQ7Kernel q7_;
  ThreadPool pool_;
  std::vector<std::unique_ptr<StreamingAttention>> attention_;
  // Evaluator-only state. These caches and trace buffers are deliberately
  // omitted from production/base capacity and traffic counters.
  std::vector<std::unique_ptr<StreamingAttention>> shadow_attention_;
  std::vector<std::uint8_t> episodic_layer_active_;
  std::vector<std::uint64_t> episodic_active_slots_by_cache_;
  std::vector<std::uint64_t> episodic_slot_positions_;
  std::vector<float> last_final_state_;
  std::vector<float> last_vocabulary_scores_;
  std::vector<float> last_shadow_input_norm_;
  std::vector<float> last_shadow_base_projected_;
  std::vector<float> last_shadow_target_residual_;
  std::vector<float> last_episodic_mass_base_pre_wo_;
  std::vector<float> last_episodic_mass_regular_component_;
  std::vector<float> last_episodic_mass_episodic_component_;
  std::vector<float> last_episodic_mass_regular_mass_;
  std::vector<float> last_episodic_mass_episodic_mass_;
  std::vector<float> last_episodic_mass_shadow_source_mass_;
  std::vector<float> last_episodic_slot_mass_;
  std::vector<float> last_episodic_slot_values_;
  std::vector<float> last_regular_entry_mass_;
  std::vector<float> last_regular_entry_values_;
  std::vector<std::uint8_t> last_regular_entry_valid_kind_;
  std::vector<std::uint64_t> last_regular_entry_positions_;
  std::vector<float> last_c28_qk_partials_;
  bool shadow_trace_valid_{};
  bool episodic_mass_trace_valid_{};
  bool episodic_slot_trace_valid_{};
  bool regular_entry_trace_valid_{};
  bool c28_qk_partial_trace_valid_{};
  std::size_t position_{};
  std::uint64_t attention_state_capacity_bytes_{};
  std::uint64_t attention_scratch_capacity_bytes_{};
  OLMoETokenMetrics metrics_{};
};

}  // namespace engram
