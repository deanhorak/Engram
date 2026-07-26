#include "engram/native_bitnet_token_runtime.h"

#include "engram/native_attention_stage_c.h"
#include "engram/native_bitnet_c.h"
#include "engram/native_shell_c.h"
#include "engram/native_stage_c.h"
#include "engram/native_stage_runner_c.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace engram {
namespace {

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
      semantic_(config_.mlp_artifact, config_.threads),
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
  std::vector<void*> cache_handles(config_.layers);
  std::vector<engram_native_stage_descriptor> descriptors(config_.layers);
  const auto scales = operator_scales_.float32();
  const auto& layers = weights_.layers();
  for (std::size_t layer = 0; layer < config_.layers; ++layer) {
    cache_handles[layer] = attention_[layer].get();
    const auto& weights = layers[layer];
    descriptors[layer] = engram_native_stage_descriptor{
        .projection_handle = &weights_.projections(),
        .query_projection = weights.query_projection,
        .key_projection = weights.key_projection,
        .value_projection = weights.value_projection,
        .output_projection = weights.output_projection,
        .attention_handles = &cache_handles[layer],
        .input_norm_weight = weights.input_norm.data(),
        .input_norm_epsilon = config_.rms_norm_epsilon,
        .attention_norm_weight = weights.attention_sub_norm.data(),
        .attention_norm_epsilon = config_.rms_norm_epsilon,
        .semantic_norm_weight = weights.post_attention_norm.data(),
        .semantic_norm_epsilon = config_.rms_norm_epsilon,
        .semantic_scale = scales[layer * 2],
        .episodic_scale = scales[layer * 2 + 1],
        .semantic_layer = layer,
    };
  }
  std::vector<engram_native_attention_stage_metrics> attention_metrics(
      config_.layers);
  std::vector<engram_bitnet_metrics> semantic_metrics(config_.layers);
  if (engram_native_run_stages_bf16(
          stage.get(), &semantic_, descriptors.data(), descriptors.size(), 1,
          length, config_.hidden_size, config_.query_heads,
          config_.key_value_heads, config_.head_dimension, positions.data(), 1,
          config_.rope_theta, attention_metrics.data(),
          semantic_metrics.data(), error, sizeof(error)) != 0) {
    throw std::runtime_error(error);
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
  for (std::size_t layer = 0; layer < config_.layers; ++layer) {
    metrics_.semantic_elapsed_ns += semantic_metrics[layer].elapsed_ns;
    metrics_.attention_elapsed_ns +=
        attention_metrics[layer].qkv_projection_ns +
        attention_metrics[layer].rope_ns +
        attention_metrics[layer].native_attention_ns +
        attention_metrics[layer].output_projection_ns;
  }
  position_ += length;
  metrics_.positions_processed += length;
  metrics_.stage_calls += config_.layers;
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
  std::int64_t token = forward(prompt);
  generated.push_back(token);
  while (generated.size() < max_new_tokens && !is_eos(token)) {
    token = forward(std::span<const std::int64_t>(&token, 1));
    generated.push_back(token);
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
