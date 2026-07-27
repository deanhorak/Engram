#include "engram/native_bitnet_token_runtime.h"

#include "engram/native_attention_stage_c.h"
#include "engram/native_shell_c.h"
#include "engram/native_stage_c.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace engram {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(const Clock::time_point started) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now() - started)
          .count());
}

class StageHandle {
 public:
  StageHandle(const std::size_t vectors, const std::size_t width) {
    char error[512] = {};
    handle_ = engram_native_stage_create(vectors, width, error, sizeof(error));
    if (handle_ == nullptr) throw std::runtime_error(error);
  }
  ~StageHandle() { engram_native_stage_destroy(handle_); }
  StageHandle(const StageHandle&) = delete;
  StageHandle& operator=(const StageHandle&) = delete;
  [[nodiscard]] void* get() const noexcept { return handle_; }

 private:
  void* handle_ = nullptr;
};

void require_shape(const NpyArray& array,
                   const std::vector<std::size_t>& expected,
                   const char* name) {
  if (array.dtype() != NpyDType::Float32 || array.shape() != expected) {
    throw std::invalid_argument(std::string(name) + " has an invalid shape");
  }
}

}  // namespace

NativeBitNetTokenRuntime::NativeBitNetTokenRuntime(
    NativeBitNetTokenConfig config)
    : config_(std::move(config)),
      weights_(config_.non_mlp_safetensors, config_.layers,
               config_.hidden_size, config_.query_heads,
               config_.key_value_heads, config_.head_dimension,
               config_.threads),
      semantic_(config_.mlp_artifact, config_.dip_coordinate_index,
                config_.threads),
      operator_scales_(load_npy(config_.controller_directory /
                                "operator_residual_scale.npy")),
      correction_scales_(
          load_npy(config_.controller_directory / "step_scale.npy")) {
  if (semantic_.layer_count() != config_.layers ||
      semantic_.hidden_size() != config_.hidden_size ||
      config_.query_heads * config_.head_dimension != config_.hidden_size ||
      config_.older_top_k > config_.older_candidates ||
      config_.sink_tokens > config_.older_top_k) {
    throw std::invalid_argument("native token runtime dimensions are invalid");
  }
  require_shape(operator_scales_, {config_.layers, 2},
                "operator residual scale");
  require_shape(correction_scales_, {config_.layers}, "controller step scale");
  if (!std::all_of(correction_scales_.float32().begin(),
                   correction_scales_.float32().end(),
                   [](const float value) { return value == 0.0F; })) {
    throw std::invalid_argument(
        "native token runtime requires zero controller correction");
  }
  attention_.reserve(config_.layers);
  for (std::size_t layer = 0; layer < config_.layers; ++layer) {
    attention_.push_back(std::make_unique<StreamingAttention>(
        StreamingAttentionConfig{
            .query_heads = config_.query_heads,
            .key_value_heads = config_.key_value_heads,
            .head_dimension = config_.head_dimension,
            .local_window = config_.local_window,
            .older_candidates = config_.older_candidates,
            .older_top_k = config_.older_top_k,
            .sink_tokens = config_.sink_tokens,
            .scale = 1.0F / std::sqrt(static_cast<float>(
                                config_.head_dimension)),
        }));
  }
}

std::int64_t NativeBitNetTokenRuntime::forward(
    const std::span<const std::int64_t> token_ids) {
  if (token_ids.empty()) {
    throw std::invalid_argument("native token input must not be empty");
  }
  if (token_ids.size() >
      std::numeric_limits<std::size_t>::max() / config_.hidden_size) {
    throw std::overflow_error("native token input dimensions overflow");
  }
  const std::size_t length = token_ids.size();
  std::vector<std::uint16_t> embedding(length * config_.hidden_size);
  const auto table = weights_.embedding();
  const std::size_t vocabulary = weights_.vocabulary_size();
  for (std::size_t token = 0; token < length; ++token) {
    if (token_ids[token] < 0 ||
        static_cast<std::size_t>(token_ids[token]) >= vocabulary) {
      throw std::out_of_range("native token identifier is outside vocabulary");
    }
    std::copy_n(table.data() +
                    static_cast<std::size_t>(token_ids[token]) *
                        config_.hidden_size,
                config_.hidden_size,
                embedding.data() + token * config_.hidden_size);
  }
  StageHandle stage(length, config_.hidden_size);
  char error[512] = {};
  if (engram_native_stage_begin_bf16(stage.get(), embedding.data(), error,
                                     sizeof(error)) != 0) {
    throw std::runtime_error(error);
  }

  std::vector<std::int64_t> positions(length);
  for (std::size_t token = 0; token < length; ++token) {
    positions[token] = static_cast<std::int64_t>(position_ + token);
  }
  const auto scales = operator_scales_.float32();
  const auto& layers = weights_.layers();
  std::vector<engram_native_attention_stage_metrics> attention_metrics(
      config_.layers);
  std::vector<NativeBitNetDIPMetrics> semantic_metrics(config_.layers);
  std::vector<std::uint16_t> semantic_input(length * config_.hidden_size);
  std::vector<std::uint16_t> semantic_output(length * config_.hidden_size);
  std::vector<std::uint32_t> selected_counts(length);
  for (std::size_t layer = 0; layer < config_.layers; ++layer) {
    const auto& weights = layers[layer];
    void* cache_handle = attention_[layer].get();
    if (engram_native_stage_attention_bf16(
            stage.get(), &weights_.projections(), weights.query_projection,
            weights.key_projection, weights.value_projection,
            weights.output_projection, &cache_handle, 1, length,
            config_.hidden_size, config_.query_heads,
            config_.key_value_heads, config_.head_dimension, positions.data(),
            1, config_.rope_theta, weights.input_norm.data(),
            config_.rms_norm_epsilon, weights.attention_sub_norm.data(),
            config_.rms_norm_epsilon, &attention_metrics[layer], error,
            sizeof(error)) != 0) {
      throw std::runtime_error(error);
    }
    if (engram_native_stage_semantic_input_bf16(
            stage.get(), weights.post_attention_norm.data(),
            config_.rms_norm_epsilon, semantic_input.data(), error,
            sizeof(error)) != 0) {
      throw std::runtime_error(error);
    }
    semantic_.forward_bf16(
        layer, semantic_input, length, semantic_output, selected_counts,
        &semantic_metrics[layer]);
    if (engram_native_stage_accept_semantic_bf16(
            stage.get(), semantic_output.data(), scales[layer * 2],
            scales[layer * 2 + 1], error, sizeof(error)) != 0) {
      throw std::runtime_error(error);
    }
  }
  std::vector<std::uint16_t> hidden(length * config_.hidden_size);
  if (engram_native_stage_final_norm_bf16(
          stage.get(), weights_.final_norm().data(), config_.rms_norm_epsilon,
          hidden.data(), error, sizeof(error)) != 0) {
    throw std::runtime_error(error);
  }
  std::int64_t next_token = -1;
  float score = 0.0F;
  if (engram_vocab_argmax_bf16(
          hidden.data() + (length - 1) * config_.hidden_size,
          weights_.embedding().data(), weights_.vocabulary_size(),
          config_.hidden_size, config_.threads, &next_token, &score) != 0) {
    throw std::runtime_error("native vocabulary argmax failed");
  }
  std::uint64_t attention_state_bytes = 0;
  std::uint64_t attention_scratch_bytes = 0;
  for (std::size_t layer = 0; layer < config_.layers; ++layer) {
    metrics_.semantic_elapsed_ns += semantic_metrics[layer].elapsed_ns;
    metrics_.semantic_kernel_cache_line_bytes +=
        semantic_metrics[layer].scheduled_cache_line_bytes;
    metrics_.semantic_selected_records +=
        semantic_metrics[layer].selected_count_total;
    metrics_.semantic_rows += semantic_metrics[layer].rows;
    metrics_.semantic_maximum_scratch_bytes =
        std::max(metrics_.semantic_maximum_scratch_bytes,
                 semantic_metrics[layer].scratch_bytes);
    const auto& current_attention = attention_metrics[layer];
    metrics_.qkv_projection_ns += current_attention.qkv_projection_ns;
    metrics_.rope_ns += current_attention.rope_ns;
    metrics_.native_attention_ns += current_attention.native_attention_ns;
    metrics_.output_projection_ns += current_attention.output_projection_ns;
    metrics_.attention_logical_read_bytes +=
        current_attention.attention.candidate_key_bytes +
        current_attention.attention.selected_value_bytes +
        current_attention.attention.local_kv_bytes;
    metrics_.attention_eviction_events +=
        current_attention.attention.eviction_events;
    metrics_.attention_older_candidate_entries_scored +=
        current_attention.attention.older_candidate_entries_scored;
    metrics_.attention_older_selected_entries +=
        current_attention.attention.older_selected_entries;
    metrics_.attention_sink_insertions +=
        current_attention.attention.sink_insertions;
    metrics_.attention_heavy_hitter_updates +=
        current_attention.attention.heavy_hitter_updates;
    attention_state_bytes += current_attention.attention.state_bytes;
    attention_scratch_bytes +=
        current_attention.attention.scratch_bytes +
        current_attention.projection_scratch_bytes;
  }
  metrics_.attention_elapsed_ns =
      metrics_.qkv_projection_ns + metrics_.rope_ns +
      metrics_.native_attention_ns + metrics_.output_projection_ns;
  metrics_.attention_state_bytes =
      std::max(metrics_.attention_state_bytes, attention_state_bytes);
  metrics_.attention_scratch_bytes =
      std::max(metrics_.attention_scratch_bytes, attention_scratch_bytes);
  const std::uint64_t global_metadata_bytes =
      static_cast<std::uint64_t>(
          semantic_.global_metadata_cache_line_bytes()) *
      static_cast<std::uint64_t>(length);
  metrics_.semantic_global_metadata_bytes += global_metadata_bytes;
  metrics_.semantic_scheduled_cache_line_bytes =
      metrics_.semantic_kernel_cache_line_bytes +
      metrics_.semantic_global_metadata_bytes;
  position_ += length;
  metrics_.positions_processed += length;
  metrics_.stage_calls += config_.layers;
  metrics_.semantic_calls += config_.layers;
  return next_token;
}

std::vector<std::int64_t> NativeBitNetTokenRuntime::generate(
    const std::span<const std::int64_t> prompt,
    const std::size_t max_new_tokens) {
  if (prompt.empty() || max_new_tokens == 0) {
    throw std::invalid_argument(
        "native generation needs a prompt and positive token budget");
  }
  std::vector<std::int64_t> generated;
  generated.reserve(max_new_tokens);
  const auto prefill_started = Clock::now();
  std::int64_t token = forward(prompt);
  metrics_.prefill_elapsed_ns += elapsed_ns(prefill_started);
  generated.push_back(token);
  if (generated.size() < max_new_tokens && !is_eos(token)) {
    const auto decode_started = Clock::now();
    while (generated.size() < max_new_tokens && !is_eos(token)) {
      token = forward(std::span<const std::int64_t>(&token, 1));
      generated.push_back(token);
    }
    metrics_.decode_elapsed_ns += elapsed_ns(decode_started);
  }
  return generated;
}

void NativeBitNetTokenRuntime::reset() {
  for (auto& cache : attention_) cache->reset();
  position_ = 0;
  metrics_ = {};
}

bool NativeBitNetTokenRuntime::is_eos(const std::int64_t token) const {
  return std::find(config_.eos_token_ids.begin(),
                   config_.eos_token_ids.end(),
                   token) != config_.eos_token_ids.end();
}

}  // namespace engram
