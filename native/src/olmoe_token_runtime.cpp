#include "engram/olmoe_token_runtime.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace engram {
namespace {

using Clock = std::chrono::steady_clock;

float bf16_to_float(const std::uint16_t value) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::uint64_t elapsed_ns(const Clock::time_point started) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now() - started)
          .count());
}

bool valid_attention_policy(const OLMoEAttentionPolicy& policy) noexcept {
  return policy.local_window > 0 && policy.older_candidates > 0 &&
         policy.older_top_k > 0 &&
         policy.older_top_k <= policy.older_candidates &&
         policy.sink_tokens <= policy.older_candidates;
}

std::uint64_t checked_metric_sum(const std::uint64_t accumulated,
                                 const std::size_t value) {
  if (value > std::numeric_limits<std::uint64_t>::max() - accumulated) {
    throw std::invalid_argument(
        "native OLMoE attention metric capacity overflows");
  }
  return accumulated + static_cast<std::uint64_t>(value);
}

}  // namespace

OLMoETokenRuntime::OLMoETokenRuntime(OLMoETokenConfig config)
    : config_(std::move(config)),
      weights_(config_.non_mlp_safetensors, config_.layers,
               config_.hidden_size,
               config_.key_value_heads * config_.head_dimension),
      q7_(config_.q7_artifact, config_.threads),
      pool_(config_.threads) {
  if (config_.layers == 0 || config_.hidden_size == 0 ||
      config_.query_heads == 0 || config_.key_value_heads == 0 ||
      config_.head_dimension == 0 ||
      config_.query_heads * config_.head_dimension != config_.hidden_size ||
      config_.query_heads % config_.key_value_heads != 0 ||
      q7_.layer_count() != config_.layers ||
      q7_.hidden_size() != config_.hidden_size ||
      (!config_.attention_policies.empty() &&
       config_.attention_policies.size() != config_.layers) ||
      !std::isfinite(config_.rms_norm_epsilon) ||
      config_.rms_norm_epsilon <= 0.0F || !std::isfinite(config_.rope_theta) ||
      config_.rope_theta <= 0.0F) {
    throw std::invalid_argument("native OLMoE token configuration is invalid");
  }
  const OLMoEAttentionPolicy scalar_policy{
      .local_window = config_.local_window,
      .older_candidates = config_.older_candidates,
      .older_top_k = config_.older_top_k,
      .sink_tokens = config_.sink_tokens,
  };
  if ((config_.attention_policies.empty() &&
       !valid_attention_policy(scalar_policy)) ||
      (!config_.attention_policies.empty() &&
       !std::all_of(config_.attention_policies.begin(),
                    config_.attention_policies.end(),
                    valid_attention_policy))) {
    throw std::invalid_argument("native OLMoE attention policy is invalid");
  }
  attention_.reserve(config_.layers);
  for (std::size_t layer = 0; layer < config_.layers; ++layer) {
    const OLMoEAttentionPolicy& policy =
        config_.attention_policies.empty()
            ? scalar_policy
            : config_.attention_policies[layer];
    attention_.push_back(std::make_unique<StreamingAttention>(
        StreamingAttentionConfig{
            .query_heads = config_.query_heads,
            .key_value_heads = config_.key_value_heads,
            .head_dimension = config_.head_dimension,
            .local_window = policy.local_window,
            .older_candidates = policy.older_candidates,
            .older_top_k = policy.older_top_k,
            .sink_tokens = policy.sink_tokens,
            .scale = 1.0F /
                     std::sqrt(static_cast<float>(config_.head_dimension)),
        }));
    attention_state_capacity_bytes_ = checked_metric_sum(
        attention_state_capacity_bytes_,
        attention_.back()->allocated_state_bytes());
    attention_scratch_capacity_bytes_ = checked_metric_sum(
        attention_scratch_capacity_bytes_,
        attention_.back()->scratch_bytes());
  }
}

void OLMoETokenRuntime::project(
    const std::span<const float> input,
    const std::span<const std::uint16_t> weight, const std::size_t rows,
    const std::size_t input_width, const std::size_t output_width,
    const std::span<float> output) {
  if (input.size() != rows * input_width ||
      weight.size() != output_width * input_width ||
      output.size() != rows * output_width) {
    throw std::invalid_argument("native OLMoE projection shape is invalid");
  }
  pool_.parallel_for(0, rows * output_width, 8,
                     [&](const std::size_t flat) {
    const std::size_t row = flat / output_width;
    const std::size_t destination = flat % output_width;
    const float* state = input.data() + row * input_width;
    const std::uint16_t* matrix =
        weight.data() + destination * input_width;
    float sum = 0.0F;
    for (std::size_t column = 0; column < input_width; ++column) {
      sum += state[column] * bf16_to_float(matrix[column]);
    }
    output[flat] = sum;
  });
}

void OLMoETokenRuntime::normalize(
    const std::span<const float> input,
    const std::span<const std::uint16_t> weight, const std::size_t rows,
    const std::size_t width, const std::span<float> output) const {
  if (input.size() != rows * width || weight.size() != width ||
      output.size() != input.size()) {
    throw std::invalid_argument("native OLMoE normalization shape is invalid");
  }
  for (std::size_t row = 0; row < rows; ++row) {
    const float* source = input.data() + row * width;
    float sum = 0.0F;
    for (std::size_t column = 0; column < width; ++column) {
      sum += source[column] * source[column];
    }
    const float inverse =
        1.0F / std::sqrt(sum / static_cast<float>(width) +
                         config_.rms_norm_epsilon);
    for (std::size_t column = 0; column < width; ++column) {
      output[row * width + column] =
          source[column] * inverse * bf16_to_float(weight[column]);
    }
  }
}

void OLMoETokenRuntime::apply_rope(const std::span<float> values,
                                   const std::size_t heads,
                                   const std::size_t position) const {
  if (config_.head_dimension % 2 != 0 ||
      values.size() != heads * config_.head_dimension) {
    throw std::invalid_argument("native OLMoE RoPE shape is invalid");
  }
  const std::size_t half = config_.head_dimension / 2;
  for (std::size_t head = 0; head < heads; ++head) {
    float* vector = values.data() + head * config_.head_dimension;
    for (std::size_t index = 0; index < half; ++index) {
      const float frequency = std::pow(
          config_.rope_theta,
          -2.0F * static_cast<float>(index) /
              static_cast<float>(config_.head_dimension));
      const float angle = static_cast<float>(position) * frequency;
      const float cosine = std::cos(angle);
      const float sine = std::sin(angle);
      const float first = vector[index];
      const float second = vector[index + half];
      vector[index] = first * cosine - second * sine;
      vector[index + half] = second * cosine + first * sine;
    }
  }
}

std::int64_t OLMoETokenRuntime::forward(
    const std::span<const std::int64_t> token_ids) {
  if (token_ids.empty()) {
    throw std::invalid_argument("native OLMoE token input must not be empty");
  }
  const auto started = Clock::now();
  const std::size_t rows = token_ids.size();
  const std::size_t hidden_width = config_.hidden_size;
  const std::size_t kv_width =
      config_.key_value_heads * config_.head_dimension;
  std::vector<float> hidden(rows * hidden_width);
  const auto embedding = weights_.embedding();
  for (std::size_t row = 0; row < rows; ++row) {
    if (token_ids[row] < 0 ||
        static_cast<std::size_t>(token_ids[row]) >=
            weights_.vocabulary_size()) {
      throw std::out_of_range("native OLMoE token is outside vocabulary");
    }
    const std::uint16_t* source =
        embedding.data() + static_cast<std::size_t>(token_ids[row]) *
                               hidden_width;
    for (std::size_t column = 0; column < hidden_width; ++column) {
      hidden[row * hidden_width + column] = bf16_to_float(source[column]);
    }
  }
  std::vector<float> normalized(rows * hidden_width);
  std::vector<float> query(rows * hidden_width);
  std::vector<float> key(rows * kv_width);
  std::vector<float> value(rows * kv_width);
  std::vector<float> key_normalized(rows * kv_width);
  std::vector<float> attention_output(rows * hidden_width);
  std::vector<float> projected(rows * hidden_width);
  std::vector<float> semantic_input(rows * hidden_width);
  std::vector<float> semantic_output(rows * hidden_width);
  std::vector<std::uint32_t> selected(rows * q7_.top_k());
  const auto& layers = weights_.layers();
  for (std::size_t layer = 0; layer < config_.layers; ++layer) {
    const OLMoELayerWeights& weights = layers[layer];
    normalize(hidden, weights.input_norm, rows, hidden_width, normalized);
    project(normalized, weights.query_projection, rows, hidden_width,
            hidden_width, query);
    project(normalized, weights.key_projection, rows, hidden_width, kv_width,
            key);
    project(normalized, weights.value_projection, rows, hidden_width, kv_width,
            value);
    normalize(query, weights.query_norm, rows, hidden_width, query);
    normalize(key, weights.key_norm, rows, kv_width, key_normalized);
    for (std::size_t row = 0; row < rows; ++row) {
      apply_rope(
          std::span<float>(query.data() + row * hidden_width, hidden_width),
          config_.query_heads, position_ + row);
      apply_rope(
          std::span<float>(key_normalized.data() + row * kv_width, kv_width),
          config_.key_value_heads, position_ + row);
      const StreamingAttentionMetrics attention_metrics =
          attention_[layer]->step(
              std::span<const float>(query.data() + row * hidden_width,
                                     hidden_width),
              std::span<const float>(
                  key_normalized.data() + row * kv_width, kv_width),
              std::span<const float>(value.data() + row * kv_width, kv_width),
              std::span<float>(
                  attention_output.data() + row * hidden_width, hidden_width));
      metrics_.attention_state_bytes =
          std::max(metrics_.attention_state_bytes,
                   attention_state_capacity_bytes_);
      metrics_.attention_scratch_bytes =
          std::max(metrics_.attention_scratch_bytes,
                   attention_scratch_capacity_bytes_);
      metrics_.attention_logical_read_bytes +=
          attention_metrics.candidate_key_bytes +
          attention_metrics.selected_value_bytes +
          attention_metrics.local_kv_bytes;
      metrics_.attention_eviction_events +=
          attention_metrics.eviction_events;
      metrics_.attention_older_candidate_entries_scored +=
          attention_metrics.older_candidate_entries_scored;
      metrics_.attention_older_selected_entries +=
          attention_metrics.older_selected_entries;
      metrics_.attention_sink_insertions +=
          attention_metrics.sink_insertions;
      metrics_.attention_heavy_hitter_updates +=
          attention_metrics.heavy_hitter_updates;
    }
    project(attention_output, weights.output_projection, rows, hidden_width,
            hidden_width, projected);
    for (std::size_t index = 0; index < hidden.size(); ++index) {
      hidden[index] += projected[index];
    }
    normalize(hidden, weights.post_attention_norm, rows, hidden_width,
              semantic_input);
    OLMoEQ7Metrics q7_metrics{};
    q7_.forward(layer, semantic_input, rows, semantic_output, selected,
                &q7_metrics);
    metrics_.q7_scheduled_bytes += q7_metrics.scheduled_stream_bytes;
    metrics_.q7_elapsed_ns += q7_metrics.elapsed_ns;
    for (std::size_t index = 0; index < hidden.size(); ++index) {
      hidden[index] += semantic_output[index];
    }
    metrics_.attention_weight_bytes +=
        static_cast<std::uint64_t>(rows) *
        static_cast<std::uint64_t>(
            (2 * hidden_width * hidden_width + 2 * kv_width * hidden_width) *
            sizeof(std::uint16_t));
  }
  normalize(hidden, weights_.final_norm(), rows, hidden_width, normalized);
  const float* final_state =
      normalized.data() + (rows - 1) * hidden_width;
  const auto head = weights_.language_head();
  std::vector<float> vocabulary_scores(weights_.vocabulary_size());
  project(std::span<const float>(final_state, hidden_width), head, 1,
          hidden_width, weights_.vocabulary_size(), vocabulary_scores);
  const auto best =
      std::max_element(vocabulary_scores.begin(), vocabulary_scores.end());
  const std::int64_t next_token = static_cast<std::int64_t>(
      std::distance(vocabulary_scores.begin(), best));
  last_final_state_.assign(final_state, final_state + hidden_width);
  last_vocabulary_scores_ = std::move(vocabulary_scores);
  position_ += rows;
  metrics_.positions_processed += rows;
  metrics_.elapsed_ns += elapsed_ns(started);
  return next_token;
}

std::vector<std::int64_t> OLMoETokenRuntime::generate(
    const std::span<const std::int64_t> prompt,
    const std::size_t max_new_tokens) {
  if (prompt.empty() || max_new_tokens == 0) {
    throw std::invalid_argument(
        "native OLMoE generation needs prompt and token budget");
  }
  std::vector<std::int64_t> result;
  result.reserve(max_new_tokens);
  std::int64_t token = forward(prompt);
  result.push_back(token);
  while (result.size() < max_new_tokens && !is_eos(token)) {
    token = forward(std::span<const std::int64_t>(&token, 1));
    result.push_back(token);
  }
  return result;
}

void OLMoETokenRuntime::reset() {
  for (auto& cache : attention_) cache->reset();
  position_ = 0;
  metrics_ = {};
  last_final_state_.clear();
  last_vocabulary_scores_.clear();
}

bool OLMoETokenRuntime::is_eos(const std::int64_t token) const {
  return std::find(config_.eos_token_ids.begin(),
                   config_.eos_token_ids.end(),
                   token) != config_.eos_token_ids.end();
}

}  // namespace engram
