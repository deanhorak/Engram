#include "engram/olmoe_token_runtime.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

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

std::size_t checked_policy_count(const std::size_t layers,
                                 const std::size_t heads) {
  if (layers != 0 &&
      heads > std::numeric_limits<std::size_t>::max() / layers) {
    throw std::invalid_argument(
        "native OLMoE head-wise attention policy count overflows");
  }
  return layers * heads;
}

std::size_t checked_trace_count(const std::size_t layers,
                                const std::size_t hidden_size) {
  if (layers != 0 &&
      hidden_size > std::numeric_limits<std::size_t>::max() / layers) {
    throw std::invalid_argument(
        "native OLMoE shadow trace storage overflows");
  }
  return layers * hidden_size;
}

std::size_t checked_source_position_count(const std::size_t rows,
                                          const std::size_t span_size) {
  if (rows != 0 &&
      span_size > std::numeric_limits<std::size_t>::max() / rows) {
    throw std::invalid_argument(
        "native OLMoE episodic source-position ledger overflows");
  }
  return rows * span_size;
}

std::size_t checked_slot_trace_count(const std::size_t layers,
                                     const std::size_t heads,
                                     const std::size_t span_size,
                                     const std::size_t width = 1) {
  const std::size_t head_count = checked_policy_count(layers, heads);
  if ((head_count != 0 &&
       span_size > std::numeric_limits<std::size_t>::max() / head_count) ||
      (head_count * span_size != 0 &&
       width >
           std::numeric_limits<std::size_t>::max() /
               (head_count * span_size))) {
    throw std::invalid_argument(
        "native OLMoE episodic slot trace storage overflows");
  }
  return head_count * span_size * width;
}

std::size_t checked_regular_entry_trace_count(
    const std::size_t layers, const std::size_t heads,
    const std::size_t width = 1) {
  const std::size_t head_count = checked_policy_count(layers, heads);
  constexpr std::size_t entries =
      StreamingAttention::kRegularTraceEntries;
  if ((head_count != 0 &&
       entries > std::numeric_limits<std::size_t>::max() / head_count) ||
      (head_count * entries != 0 &&
       width > std::numeric_limits<std::size_t>::max() /
                   (head_count * entries))) {
    throw std::invalid_argument(
        "native OLMoE regular-entry trace storage overflows");
  }
  return head_count * entries * width;
}

std::size_t checked_c28_qk_partial_trace_count(
    const std::size_t layers, const std::size_t heads) {
  const std::size_t head_count = checked_policy_count(layers, heads);
  constexpr std::size_t entries =
      StreamingAttention::kC28TraceEntries;
  constexpr std::size_t bands =
      StreamingAttention::kQKPartialBands;
  if ((head_count != 0 &&
       entries > std::numeric_limits<std::size_t>::max() / head_count) ||
      (head_count * entries != 0 &&
       bands > std::numeric_limits<std::size_t>::max() /
                   (head_count * entries))) {
    throw std::invalid_argument(
        "native OLMoE C28 QK partial trace storage overflows");
  }
  return head_count * entries * bands;
}

std::size_t checked_c28_qk_candidate_trace_count(
    const std::size_t layers, const std::size_t heads,
    const std::size_t older_candidates) {
  const std::size_t head_count = checked_policy_count(layers, heads);
  constexpr std::size_t bands = StreamingAttention::kQKPartialBands;
  if ((head_count != 0 &&
       older_candidates > std::numeric_limits<std::size_t>::max() /
                              head_count) ||
      (head_count * older_candidates != 0 &&
       bands > std::numeric_limits<std::size_t>::max() /
                   (head_count * older_candidates))) {
    throw std::invalid_argument(
        "native OLMoE C28 QK candidate trace storage overflows");
  }
  return head_count * older_candidates * bands;
}

std::size_t checked_c28_qk_candidate_key_trace_count(
    const std::size_t layers, const std::size_t heads,
    const std::size_t older_candidates, const std::size_t head_dimension) {
  const std::size_t head_count = checked_policy_count(layers, heads);
  if (head_count != 0 &&
      older_candidates > std::numeric_limits<std::size_t>::max() / head_count) {
    throw std::invalid_argument(
        "native OLMoE C28 candidate-key trace storage overflows");
  }
  const std::size_t candidate_count = head_count * older_candidates;
  if (candidate_count != 0 &&
      head_dimension >
          std::numeric_limits<std::size_t>::max() / candidate_count) {
    throw std::invalid_argument(
        "native OLMoE C28 candidate-key trace storage overflows");
  }
  return candidate_count * head_dimension;
}

std::size_t checked_c28_qk_candidate_value_trace_count(
    const std::size_t layers, const std::size_t heads,
    const std::size_t older_candidates, const std::size_t head_dimension) {
  return checked_c28_qk_candidate_key_trace_count(
      layers, heads, older_candidates, head_dimension);
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
      (!config_.attention_policies.empty() &&
       !config_.head_attention_policies.empty()) ||
      !std::isfinite(config_.rms_norm_epsilon) ||
      config_.rms_norm_epsilon <= 0.0F || !std::isfinite(config_.rope_theta) ||
      config_.rope_theta <= 0.0F) {
    throw std::invalid_argument("native OLMoE token configuration is invalid");
  }
  const bool episodic_enabled =
      config_.episodic_policy.slots != 0 ||
      config_.episodic_policy.span_size != 0;
  const bool masked_episodic = !config_.episodic_head_mask.empty();
  const std::size_t episodic_head_count =
      checked_policy_count(config_.layers, config_.query_heads);
  if (!std::isfinite(config_.episodic_logit_bias) ||
      (!episodic_enabled &&
       (masked_episodic || config_.episodic_logit_bias != 0.0F)) ||
      (episodic_enabled &&
       (config_.episodic_policy.slots == 0 ||
        config_.episodic_policy.span_size == 0 ||
        config_.episodic_policy.slots %
                config_.episodic_policy.span_size !=
            0 ||
        !config_.attention_policies.empty() ||
        !config_.head_attention_policies.empty() ||
        (masked_episodic &&
         (config_.episodic_head_mask.size() != episodic_head_count ||
          !std::all_of(config_.episodic_head_mask.begin(),
                       config_.episodic_head_mask.end(),
                       [](const std::uint8_t selected) {
                         return selected <= 1;
                       }) ||
          std::none_of(config_.episodic_head_mask.begin(),
                       config_.episodic_head_mask.end(),
                       [](const std::uint8_t selected) {
                         return selected != 0;
                       })))))) {
    throw std::invalid_argument(
        "native OLMoE episodic configuration is invalid");
  }
  if (!config_.head_attention_policies.empty() &&
      (config_.query_heads != config_.key_value_heads ||
       config_.head_attention_policies.size() !=
           checked_policy_count(config_.layers, config_.query_heads))) {
    throw std::invalid_argument(
        "native OLMoE head-wise attention configuration is invalid");
  }
  if (config_.shadow_attention_policy.has_value() &&
      (!config_.head_attention_policies.empty() ||
       !valid_attention_policy(*config_.shadow_attention_policy))) {
    throw std::invalid_argument(
        "native OLMoE shadow attention configuration is invalid");
  }
  const OLMoEAttentionPolicy scalar_policy{
      .local_window = config_.local_window,
      .older_candidates = config_.older_candidates,
      .older_top_k = config_.older_top_k,
      .sink_tokens = config_.sink_tokens,
  };
  if (config_.c28_qk_partial_trace &&
      (!config_.shadow_attention_policy.has_value() ||
       config_.head_dimension !=
           StreamingAttention::kQKTraceHeadDimension ||
       scalar_policy.local_window !=
           StreamingAttention::kRegularTraceLocalEntries ||
       scalar_policy.older_top_k !=
           StreamingAttention::kRegularTraceOlderEntries ||
       config_.episodic_policy.span_size !=
           StreamingAttention::kC28TraceEpisodicEntries)) {
    throw std::invalid_argument(
        "native OLMoE C28 QK partial trace configuration is invalid");
  }
  if (config_.c28_qk_candidate_trace &&
      (!config_.shadow_attention_policy.has_value() ||
       config_.head_dimension !=
           StreamingAttention::kQKTraceHeadDimension ||
       scalar_policy.local_window !=
           StreamingAttention::kRegularTraceLocalEntries ||
       scalar_policy.older_top_k !=
           StreamingAttention::kRegularTraceOlderEntries ||
       config_.episodic_policy.span_size !=
           StreamingAttention::kC28TraceEpisodicEntries)) {
    throw std::invalid_argument(
        "native OLMoE C28 QK candidate trace configuration is invalid");
  }
  if (config_.c28_qk_candidate_key_trace &&
      (!config_.shadow_attention_policy.has_value() ||
       config_.head_dimension !=
           StreamingAttention::kQKTraceHeadDimension ||
       scalar_policy.local_window !=
           StreamingAttention::kRegularTraceLocalEntries ||
       scalar_policy.older_top_k !=
           StreamingAttention::kRegularTraceOlderEntries ||
       config_.episodic_policy.span_size !=
           StreamingAttention::kC28TraceEpisodicEntries)) {
    throw std::invalid_argument(
        "native OLMoE C28 candidate-key trace configuration is invalid");
  }
  if (config_.c28_qk_candidate_value_trace &&
      (!config_.shadow_attention_policy.has_value() ||
       config_.head_dimension !=
           StreamingAttention::kQKTraceHeadDimension ||
       scalar_policy.local_window !=
           StreamingAttention::kRegularTraceLocalEntries ||
       scalar_policy.older_top_k !=
           StreamingAttention::kRegularTraceOlderEntries ||
       config_.episodic_policy.span_size !=
           StreamingAttention::kC28TraceEpisodicEntries)) {
    throw std::invalid_argument(
        "native OLMoE C28 candidate-value trace configuration is invalid");
  }
  if ((config_.attention_policies.empty() &&
       config_.head_attention_policies.empty() &&
       !valid_attention_policy(scalar_policy)) ||
      (!config_.attention_policies.empty() &&
       !std::all_of(config_.attention_policies.begin(),
                    config_.attention_policies.end(),
                    valid_attention_policy)) ||
      (!config_.head_attention_policies.empty() &&
       !std::all_of(config_.head_attention_policies.begin(),
                    config_.head_attention_policies.end(),
                    valid_attention_policy))) {
    throw std::invalid_argument("native OLMoE attention policy is invalid");
  }
  attention_.reserve(config_.head_attention_policies.empty()
                         ? config_.layers
                         : config_.head_attention_policies.size());
  episodic_layer_active_.reserve(config_.layers);
  for (std::size_t layer = 0; layer < config_.layers; ++layer) {
    if (!config_.head_attention_policies.empty()) {
      for (std::size_t head = 0; head < config_.query_heads; ++head) {
        const OLMoEAttentionPolicy& policy =
            config_.head_attention_policies[
                layer * config_.query_heads + head];
        attention_.push_back(std::make_unique<StreamingAttention>(
            StreamingAttentionConfig{
                .query_heads = 1,
                .key_value_heads = 1,
                .head_dimension = config_.head_dimension,
                .local_window = policy.local_window,
                .older_candidates = policy.older_candidates,
                .older_top_k = policy.older_top_k,
                .sink_tokens = policy.sink_tokens,
                .episodic_slots = 0,
                .episodic_span_size = 0,
                .episodic_head_mask = {},
                .scale =
                    1.0F /
                    std::sqrt(static_cast<float>(config_.head_dimension)),
                .episodic_logit_bias = 0.0F,
                .local_bf16 = config_.local_bf16,
                .local_values_bf16 = config_.local_values_bf16,
                .local_values_fp16 = config_.local_values_fp16,
                .local_fp16 = config_.local_fp16,
                .local_int8 = config_.local_int8,
            }));
        attention_state_capacity_bytes_ = checked_metric_sum(
            attention_state_capacity_bytes_,
            attention_.back()->allocated_state_bytes());
        attention_scratch_capacity_bytes_ = checked_metric_sum(
            attention_scratch_capacity_bytes_,
            attention_.back()->scratch_bytes());
      }
      episodic_layer_active_.push_back(0);
      continue;
    }
    const OLMoEAttentionPolicy& policy =
        config_.attention_policies.empty()
            ? scalar_policy
            : config_.attention_policies[layer];
    std::vector<std::uint8_t> layer_episodic_mask;
    if (masked_episodic) {
      const auto begin =
          config_.episodic_head_mask.begin() +
          static_cast<std::ptrdiff_t>(layer * config_.query_heads);
      layer_episodic_mask.assign(
          begin,
          begin + static_cast<std::ptrdiff_t>(config_.query_heads));
    }
    const bool layer_episodic_active =
        episodic_enabled &&
        (!masked_episodic ||
         std::any_of(layer_episodic_mask.begin(),
                     layer_episodic_mask.end(),
                     [](const std::uint8_t selected) {
                       return selected != 0;
                     }));
    episodic_layer_active_.push_back(
        layer_episodic_active ? std::uint8_t{1} : std::uint8_t{0});
    attention_.push_back(std::make_unique<StreamingAttention>(
        StreamingAttentionConfig{
            .query_heads = config_.query_heads,
            .key_value_heads = config_.key_value_heads,
            .head_dimension = config_.head_dimension,
            .local_window = policy.local_window,
            .older_candidates = policy.older_candidates,
            .older_top_k = policy.older_top_k,
            .sink_tokens = policy.sink_tokens,
            .episodic_slots =
                layer_episodic_active ? config_.episodic_policy.slots : 0,
            .episodic_span_size =
                layer_episodic_active
                    ? config_.episodic_policy.span_size
                    : 0,
            .episodic_head_mask =
                layer_episodic_active
                    ? std::move(layer_episodic_mask)
                    : std::vector<std::uint8_t>{},
            .scale = 1.0F /
                     std::sqrt(static_cast<float>(config_.head_dimension)),
            .episodic_logit_bias =
                layer_episodic_active
                    ? config_.episodic_logit_bias
                    : 0.0F,
            .local_bf16 = config_.local_bf16,
            .local_values_bf16 = config_.local_values_bf16,
            .local_values_fp16 = config_.local_values_fp16,
            .local_fp16 = config_.local_fp16,
            .local_int8 = config_.local_int8,
        }));
    attention_state_capacity_bytes_ = checked_metric_sum(
        attention_state_capacity_bytes_,
        attention_.back()->allocated_state_bytes());
    attention_scratch_capacity_bytes_ = checked_metric_sum(
        attention_scratch_capacity_bytes_,
        attention_.back()->scratch_bytes());
  }
  episodic_active_slots_by_cache_.resize(attention_.size());
  episodic_slot_positions_.resize(
      config_.episodic_policy.slots,
      std::numeric_limits<std::uint64_t>::max());
  if (config_.shadow_attention_policy.has_value()) {
    const OLMoEAttentionPolicy& policy =
        *config_.shadow_attention_policy;
    shadow_attention_.reserve(config_.layers);
    for (std::size_t layer = 0; layer < config_.layers; ++layer) {
      shadow_attention_.push_back(std::make_unique<StreamingAttention>(
          StreamingAttentionConfig{
              .query_heads = config_.query_heads,
              .key_value_heads = config_.key_value_heads,
              .head_dimension = config_.head_dimension,
              .local_window = policy.local_window,
              .older_candidates = policy.older_candidates,
              .older_top_k = policy.older_top_k,
              .sink_tokens = policy.sink_tokens,
              .episodic_slots = 0,
              .episodic_span_size = 0,
              .episodic_head_mask = {},
              .scale =
                  1.0F /
                  std::sqrt(static_cast<float>(config_.head_dimension)),
              .episodic_logit_bias = 0.0F,
          }));
    }
    const std::size_t trace_count =
        checked_trace_count(config_.layers, config_.hidden_size);
    last_shadow_input_norm_.resize(trace_count);
    last_shadow_base_projected_.resize(trace_count);
    last_shadow_target_residual_.resize(trace_count);
    last_episodic_mass_base_pre_wo_.resize(trace_count);
    last_episodic_mass_regular_component_.resize(trace_count);
    last_episodic_mass_episodic_component_.resize(trace_count);
    const std::size_t mass_count =
        checked_policy_count(config_.layers, config_.query_heads);
    last_episodic_mass_regular_mass_.resize(mass_count);
    last_episodic_mass_episodic_mass_.resize(mass_count);
    last_episodic_mass_shadow_source_mass_.resize(mass_count);
    last_episodic_slot_mass_.resize(checked_slot_trace_count(
        config_.layers, config_.query_heads,
        config_.episodic_policy.span_size));
    last_episodic_slot_values_.resize(checked_slot_trace_count(
        config_.layers, config_.query_heads,
        config_.episodic_policy.span_size, config_.head_dimension));
    if (config_.local_window ==
            StreamingAttention::kRegularTraceLocalEntries &&
        config_.older_top_k ==
            StreamingAttention::kRegularTraceOlderEntries) {
      last_regular_entry_mass_.resize(
          checked_regular_entry_trace_count(
              config_.layers, config_.query_heads));
      last_regular_entry_values_.resize(
          checked_regular_entry_trace_count(
              config_.layers, config_.query_heads,
              config_.head_dimension));
      last_regular_entry_valid_kind_.resize(
          checked_regular_entry_trace_count(
              config_.layers, config_.query_heads));
      last_regular_entry_positions_.resize(
          checked_regular_entry_trace_count(
              config_.layers, config_.query_heads));
    }
    if (config_.c28_qk_partial_trace) {
      last_c28_qk_partials_.resize(
          checked_c28_qk_partial_trace_count(
              config_.layers, config_.query_heads));
    }
    if (config_.c28_qk_candidate_trace) {
      last_c28_qk_candidates_.resize(
          checked_c28_qk_candidate_trace_count(
              config_.layers, config_.query_heads,
              scalar_policy.older_candidates));
    }
    if (config_.c28_qk_candidate_key_trace) {
      last_c28_qk_candidate_keys_.resize(
          checked_c28_qk_candidate_key_trace_count(
              config_.layers, config_.query_heads,
              scalar_policy.older_candidates, config_.head_dimension));
    }
    if (config_.c28_qk_candidate_value_trace) {
      last_c28_qk_candidate_values_.resize(
          checked_c28_qk_candidate_value_trace_count(
              config_.layers, config_.query_heads,
              scalar_policy.older_candidates, config_.head_dimension));
    }
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
  return forward_impl(token_ids, {}, {}, {});
}

std::int64_t OLMoETokenRuntime::forward_episodic(
    const std::span<const std::int64_t> token_ids,
    const std::span<const std::int32_t> write_slots,
    const std::span<const std::int32_t> read_spans) {
  return forward_episodic_masked(token_ids, write_slots, read_spans, {});
}

std::int64_t OLMoETokenRuntime::forward_episodic_masked(
    const std::span<const std::int64_t> token_ids,
    const std::span<const std::int32_t> write_slots,
    const std::span<const std::int32_t> read_spans,
    const std::span<const std::uint8_t> candidate_masks) {
  if (config_.episodic_policy.slots == 0 ||
      write_slots.size() != token_ids.size() ||
      read_spans.size() != token_ids.size()) {
    throw std::invalid_argument(
        "native OLMoE episodic directives are invalid");
  }
  const std::size_t span_count =
      config_.episodic_policy.slots / config_.episodic_policy.span_size;
  const std::size_t masks_per_row = config_.layers * config_.query_heads *
                                    config_.older_candidates;
  if (!candidate_masks.empty() &&
      (candidate_masks.size() != token_ids.size() * masks_per_row ||
       !std::all_of(candidate_masks.begin(), candidate_masks.end(),
                    [](const std::uint8_t value) { return value <= 1; }) ||
       !config_.head_attention_policies.empty())) {
    throw std::invalid_argument(
        "native OLMoE candidate masks are invalid for this route");
  }
  if (token_ids.size() >
      std::numeric_limits<std::uint64_t>::max() - position_) {
    throw std::invalid_argument(
        "native OLMoE episodic position overflows");
  }
  std::vector<std::uint64_t> next_slot_positions =
      episodic_slot_positions_;
  std::vector<std::uint64_t> read_source_positions(
      checked_source_position_count(
          token_ids.size(), config_.episodic_policy.span_size),
      std::numeric_limits<std::uint64_t>::max());
  for (std::size_t row = 0; row < token_ids.size(); ++row) {
    if (write_slots[row] < -1 ||
        (write_slots[row] >= 0 &&
         static_cast<std::size_t>(write_slots[row]) >=
             config_.episodic_policy.slots) ||
        read_spans[row] < -1 ||
        (read_spans[row] >= 0 &&
         static_cast<std::size_t>(read_spans[row]) >= span_count)) {
      throw std::invalid_argument(
          "native OLMoE episodic directive is outside capacity");
    }
    const std::uint64_t absolute_position =
        static_cast<std::uint64_t>(position_ + row);
    if (read_spans[row] >= 0) {
      const std::size_t begin =
          static_cast<std::size_t>(read_spans[row]) *
          config_.episodic_policy.span_size;
      for (std::size_t offset = 0;
           offset < config_.episodic_policy.span_size; ++offset) {
        const std::uint64_t source_position =
            next_slot_positions[begin + offset];
        if (source_position ==
                std::numeric_limits<std::uint64_t>::max() ||
            source_position >= absolute_position) {
          throw std::invalid_argument(
              "native OLMoE episodic read is not strictly causal");
        }
        read_source_positions[
            row * config_.episodic_policy.span_size + offset] =
            source_position;
      }
    }
    if (write_slots[row] >= 0) {
      const std::size_t slot =
          static_cast<std::size_t>(write_slots[row]);
      if (next_slot_positions[slot] !=
          std::numeric_limits<std::uint64_t>::max()) {
        throw std::invalid_argument(
            "native OLMoE episodic slot is already active");
      }
      next_slot_positions[slot] = absolute_position;
    }
  }
  const std::int64_t next_token =
      forward_impl(token_ids, write_slots, read_spans,
                   read_source_positions, candidate_masks);
  episodic_slot_positions_ = std::move(next_slot_positions);
  return next_token;
}

std::int64_t OLMoETokenRuntime::forward_impl(
    const std::span<const std::int64_t> token_ids,
    const std::span<const std::int32_t> write_slots,
    const std::span<const std::int32_t> read_spans,
    const std::span<const std::uint64_t> read_source_positions,
    const std::span<const std::uint8_t> candidate_masks) {
  if (token_ids.empty()) {
    throw std::invalid_argument("native OLMoE token input must not be empty");
  }
  if ((!write_slots.empty() &&
       read_source_positions.size() !=
           checked_source_position_count(
               token_ids.size(), config_.episodic_policy.span_size)) ||
      (write_slots.empty() && !read_source_positions.empty())) {
    throw std::invalid_argument(
        "native OLMoE episodic source-position ledger is invalid");
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
  std::vector<float> shadow_attention_output;
  std::vector<float> shadow_projected;
  std::vector<float> next_shadow_input_norm;
  std::vector<float> next_shadow_base_projected;
  std::vector<float> next_shadow_target_residual;
  const bool capture_episodic_mass_trace =
      !shadow_attention_.empty() && !read_spans.empty() &&
      read_spans.back() >= 0;
  const bool capture_regular_entry_trace =
      capture_episodic_mass_trace &&
      config_.local_window ==
          StreamingAttention::kRegularTraceLocalEntries &&
      config_.older_top_k ==
          StreamingAttention::kRegularTraceOlderEntries;
  const bool capture_c28_qk_partial_trace =
      config_.c28_qk_partial_trace || config_.c28_qk_candidate_trace;
  const bool capture_c28_qk_candidate_trace =
      config_.c28_qk_candidate_trace;
  const bool capture_c28_qk_candidate_key_trace =
      config_.c28_qk_candidate_key_trace;
  const bool capture_c28_qk_candidate_value_trace =
      config_.c28_qk_candidate_value_trace;
  if (capture_episodic_mass_trace) {
    const std::uint64_t absolute_position =
        static_cast<std::uint64_t>(position_ + rows - 1);
    const std::size_t source_offset =
        (rows - 1) * config_.episodic_policy.span_size;
    for (std::size_t offset = 0;
         offset < config_.episodic_policy.span_size; ++offset) {
      const std::uint64_t source_position =
          read_source_positions[source_offset + offset];
      if (source_position >= absolute_position ||
          absolute_position - source_position >=
              config_.shadow_attention_policy->local_window) {
        throw std::invalid_argument(
            "native OLMoE shadow source position is outside the exact "
            "local window");
      }
    }
  }
  std::vector<float> next_episodic_mass_base_pre_wo;
  std::vector<float> next_episodic_mass_regular_component;
  std::vector<float> next_episodic_mass_episodic_component;
  std::vector<float> next_episodic_mass_regular_mass;
  std::vector<float> next_episodic_mass_episodic_mass;
  std::vector<float> next_episodic_mass_shadow_source_mass;
  std::vector<float> next_episodic_slot_mass;
  std::vector<float> next_episodic_slot_values;
  std::vector<float> next_regular_entry_mass;
  std::vector<float> next_regular_entry_values;
  std::vector<std::uint8_t> next_regular_entry_valid_kind;
  std::vector<std::uint64_t> next_regular_entry_positions;
  std::vector<float> next_c28_qk_partials;
  std::vector<float> next_c28_qk_candidate_keys;
  std::vector<float> next_c28_qk_candidate_values;
  const std::size_t qk_partial_count =
      checked_c28_qk_partial_trace_count(
          config_.layers, config_.query_heads);
  const std::size_t qk_candidate_count =
      checked_c28_qk_candidate_trace_count(
          config_.layers, config_.query_heads,
          config_.older_candidates);
  const std::size_t qk_candidate_key_count =
      checked_c28_qk_candidate_key_trace_count(
          config_.layers, config_.query_heads,
          config_.older_candidates, config_.head_dimension);
  const std::size_t qk_candidate_value_count =
      checked_c28_qk_candidate_value_trace_count(
          config_.layers, config_.query_heads,
          config_.older_candidates, config_.head_dimension);
  if (!shadow_attention_.empty()) {
    shadow_attention_output.resize(rows * hidden_width);
    shadow_projected.resize(rows * hidden_width);
    const std::size_t trace_count =
        checked_trace_count(config_.layers, hidden_width);
    next_shadow_input_norm.resize(trace_count);
    next_shadow_base_projected.resize(trace_count);
    next_shadow_target_residual.resize(trace_count);
    if (capture_episodic_mass_trace) {
      next_episodic_mass_base_pre_wo.resize(trace_count);
      next_episodic_mass_regular_component.resize(trace_count);
      next_episodic_mass_episodic_component.resize(trace_count);
      const std::size_t mass_count =
          checked_policy_count(config_.layers, config_.query_heads);
      next_episodic_mass_regular_mass.resize(mass_count);
      next_episodic_mass_episodic_mass.resize(mass_count);
      next_episodic_mass_shadow_source_mass.resize(mass_count);
      next_episodic_slot_mass.resize(checked_slot_trace_count(
          config_.layers, config_.query_heads,
          config_.episodic_policy.span_size));
      next_episodic_slot_values.resize(checked_slot_trace_count(
          config_.layers, config_.query_heads,
          config_.episodic_policy.span_size, config_.head_dimension));
      if (capture_regular_entry_trace) {
        next_regular_entry_mass.resize(
            checked_regular_entry_trace_count(
                config_.layers, config_.query_heads));
        next_regular_entry_values.resize(
            checked_regular_entry_trace_count(
                config_.layers, config_.query_heads,
                config_.head_dimension));
        next_regular_entry_valid_kind.resize(
            checked_regular_entry_trace_count(
                config_.layers, config_.query_heads));
        next_regular_entry_positions.resize(
            checked_regular_entry_trace_count(
                config_.layers, config_.query_heads));
      }
    }
  }
  if (capture_c28_qk_partial_trace) {
    next_c28_qk_partials.resize(
        qk_partial_count +
        (capture_c28_qk_candidate_trace ? qk_candidate_count : 0));
  }
  if (capture_c28_qk_candidate_key_trace) {
    next_c28_qk_candidate_keys.resize(qk_candidate_key_count);
  }
  if (capture_c28_qk_candidate_value_trace) {
    next_c28_qk_candidate_values.resize(qk_candidate_value_count);
  }
  std::vector<float> semantic_input(rows * hidden_width);
  std::vector<float> semantic_output(rows * hidden_width);
  std::vector<std::uint32_t> selected(rows * q7_.top_k());
  const auto& layers = weights_.layers();
  for (std::size_t layer = 0; layer < config_.layers; ++layer) {
    const OLMoELayerWeights& weights = layers[layer];
    normalize(hidden, weights.input_norm, rows, hidden_width, normalized);
    if (!shadow_attention_.empty()) {
      const float* last_normalized =
          normalized.data() + (rows - 1) * hidden_width;
      std::copy(
          last_normalized, last_normalized + hidden_width,
          next_shadow_input_norm.begin() +
              static_cast<std::ptrdiff_t>(layer * hidden_width));
    }
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
      metrics_.attention_state_bytes =
          std::max(metrics_.attention_state_bytes,
                   attention_state_capacity_bytes_);
      metrics_.attention_scratch_bytes =
          std::max(metrics_.attention_scratch_bytes,
                   attention_scratch_capacity_bytes_);
      const auto accumulate_attention_metrics =
          [this](const StreamingAttentionMetrics& attention_metrics,
                 const std::size_t cache_index) {
            metrics_.attention_logical_read_bytes +=
                attention_metrics.candidate_key_bytes +
                attention_metrics.selected_value_bytes +
                attention_metrics.local_kv_bytes +
                attention_metrics.episodic_key_read_bytes +
                attention_metrics.episodic_value_read_bytes;
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
            metrics_.episodic_slots_written +=
                attention_metrics.episodic_slots_written;
            metrics_.episodic_read_events +=
                attention_metrics.episodic_read_events;
            metrics_.episodic_entries_read +=
                attention_metrics.episodic_entries_read;
            metrics_.episodic_write_bytes +=
                attention_metrics.episodic_write_bytes;
            metrics_.episodic_key_read_bytes +=
                attention_metrics.episodic_key_read_bytes;
            metrics_.episodic_value_read_bytes +=
                attention_metrics.episodic_value_read_bytes;
            metrics_.episodic_duplicate_older_entries_suppressed +=
                attention_metrics
                    .episodic_duplicate_older_entries_suppressed;
            episodic_active_slots_by_cache_[cache_index] =
                attention_metrics.episodic_active_slots;
          };
      if (config_.head_attention_policies.empty()) {
        const auto query_row = std::span<const float>(
            query.data() + row * hidden_width, hidden_width);
        const auto key_row = std::span<const float>(
            key_normalized.data() + row * kv_width, kv_width);
        const auto value_row = std::span<const float>(
            value.data() + row * kv_width, kv_width);
        const auto output_row = std::span<float>(
            attention_output.data() + row * hidden_width, hidden_width);
        const bool capture_row =
            capture_episodic_mass_trace && row + 1 == rows;
        const bool capture_qk_row =
            capture_c28_qk_partial_trace && row + 1 == rows;
        const bool capture_qk_candidate_key_row =
            capture_c28_qk_candidate_key_trace && row + 1 == rows;
        const bool capture_qk_candidate_value_row =
            capture_c28_qk_candidate_value_trace && row + 1 == rows;
        const std::size_t qk_layer_partial_count =
            config_.query_heads * StreamingAttention::kC28TraceEntries *
            StreamingAttention::kQKPartialBands;
        const std::size_t qk_layer_candidate_count =
            config_.query_heads * config_.older_candidates *
            StreamingAttention::kQKPartialBands;
        const std::size_t qk_offset =
            layer * (qk_layer_partial_count +
                     (capture_c28_qk_candidate_trace
                          ? qk_layer_candidate_count
                          : 0));
        const auto qk_trace = [&]() {
          return capture_qk_row
                     ? std::span<float>(
                           next_c28_qk_partials.data() + qk_offset,
                           qk_layer_partial_count +
                               (capture_c28_qk_candidate_trace
                                    ? qk_layer_candidate_count
                                    : 0))
                     : std::span<float>{};
        };
        const std::size_t qk_candidate_key_layer_count =
            config_.query_heads * config_.older_candidates *
            config_.head_dimension;
        const auto qk_candidate_key_trace = [&]() {
          return capture_qk_candidate_key_row
                     ? std::span<float>(
                           next_c28_qk_candidate_keys.data() +
                               layer * qk_candidate_key_layer_count,
                           qk_candidate_key_layer_count)
                     : std::span<float>{};
        };
        const std::size_t qk_candidate_value_layer_count =
            config_.query_heads * config_.older_candidates *
            config_.head_dimension;
        const auto qk_candidate_value_trace = [&]() {
          return capture_qk_candidate_value_row
                     ? std::span<float>(
                           next_c28_qk_candidate_values.data() +
                               layer * qk_candidate_value_layer_count,
                           qk_candidate_value_layer_count)
                     : std::span<float>{};
        };
        const auto candidate_mask =
            candidate_masks.empty()
                ? std::span<const std::uint8_t>{}
                : std::span<const std::uint8_t>(
                      candidate_masks.data() +
                          (row * config_.layers + layer) *
                              config_.query_heads * config_.older_candidates,
                      config_.query_heads * config_.older_candidates);
        const bool use_episodic =
            !write_slots.empty() && episodic_layer_active_[layer] != 0;
        StreamingAttentionMetrics attention_metrics{};
        if (!use_episodic) {
          if (capture_row) {
            const std::size_t trace_offset = layer * hidden_width;
            const std::size_t mass_offset =
                layer * config_.query_heads;
            if (capture_regular_entry_trace) {
              const std::size_t entry_offset =
                  layer * config_.query_heads *
                  StreamingAttention::kRegularTraceEntries;
              const std::size_t entry_value_offset =
                  entry_offset * config_.head_dimension;
              attention_metrics = capture_qk_row
                  ? attention_[layer]->step_regular_entries_qk_traced(
                      query_row, key_row, value_row, output_row,
                      std::span<float>(
                          next_regular_entry_mass.data() + entry_offset,
                          config_.query_heads *
                              StreamingAttention::kRegularTraceEntries),
                      std::span<float>(
                          next_regular_entry_values.data() +
                              entry_value_offset,
                          config_.query_heads *
                              StreamingAttention::kRegularTraceEntries *
                              config_.head_dimension),
                      std::span<std::uint8_t>(
                          next_regular_entry_valid_kind.data() +
                              entry_offset,
                          config_.query_heads *
                              StreamingAttention::kRegularTraceEntries),
                      std::span<std::uint64_t>(
                          next_regular_entry_positions.data() +
                              entry_offset,
                          config_.query_heads *
                              StreamingAttention::kRegularTraceEntries),
                      qk_trace())
                  : attention_[layer]->step_regular_entries_traced(
                      query_row, key_row, value_row, output_row,
                      std::span<float>(
                          next_regular_entry_mass.data() + entry_offset,
                          config_.query_heads *
                              StreamingAttention::kRegularTraceEntries),
                      std::span<float>(
                          next_regular_entry_values.data() +
                              entry_value_offset,
                          config_.query_heads *
                              StreamingAttention::kRegularTraceEntries *
                              config_.head_dimension),
                      std::span<std::uint8_t>(
                          next_regular_entry_valid_kind.data() +
                              entry_offset,
                          config_.query_heads *
                              StreamingAttention::kRegularTraceEntries),
                      std::span<std::uint64_t>(
                          next_regular_entry_positions.data() +
                              entry_offset,
                          config_.query_heads *
                              StreamingAttention::kRegularTraceEntries));
            } else {
              attention_metrics = attention_[layer]->step(
                  query_row, key_row, value_row, output_row);
            }
            std::copy(
                output_row.begin(), output_row.end(),
                next_episodic_mass_regular_component.begin() +
                    static_cast<std::ptrdiff_t>(trace_offset));
            std::fill_n(
                next_episodic_mass_episodic_component.begin() +
                    static_cast<std::ptrdiff_t>(trace_offset),
                hidden_width, 0.0F);
            std::fill_n(
                next_episodic_mass_regular_mass.begin() +
                    static_cast<std::ptrdiff_t>(mass_offset),
                config_.query_heads, 1.0F);
            std::fill_n(
                next_episodic_mass_episodic_mass.begin() +
                    static_cast<std::ptrdiff_t>(mass_offset),
                config_.query_heads, 0.0F);
            const std::size_t slot_mass_offset =
                layer * config_.query_heads *
                config_.episodic_policy.span_size;
            const std::size_t slot_value_offset =
                slot_mass_offset * config_.head_dimension;
            std::fill_n(
                next_episodic_slot_mass.begin() +
                    static_cast<std::ptrdiff_t>(slot_mass_offset),
                config_.query_heads * config_.episodic_policy.span_size,
                0.0F);
            std::fill_n(
                next_episodic_slot_values.begin() +
                    static_cast<std::ptrdiff_t>(slot_value_offset),
                config_.query_heads * config_.episodic_policy.span_size *
                    config_.head_dimension,
                0.0F);
          } else if (capture_qk_row) {
            attention_metrics = attention_[layer]->step_c28_qk_traced(
                query_row, key_row, value_row,
                StreamingAttention::kNoEpisodicDirective,
                StreamingAttention::kNoEpisodicDirective, output_row,
                qk_trace());
          } else if (!candidate_mask.empty()) {
            attention_metrics = attention_[layer]->step_episodic_masked(
                query_row, key_row, value_row,
                StreamingAttention::kNoEpisodicDirective,
                StreamingAttention::kNoEpisodicDirective, output_row,
                candidate_mask);
          } else {
            attention_metrics = attention_[layer]->step(
                query_row, key_row, value_row, output_row);
          }
        } else {
          const std::size_t write_slot =
              write_slots[row] < 0
                  ? StreamingAttention::kNoEpisodicDirective
                  : static_cast<std::size_t>(write_slots[row]);
          const std::size_t read_span =
              read_spans[row] < 0
                  ? StreamingAttention::kNoEpisodicDirective
                  : static_cast<std::size_t>(read_spans[row]);
          if (capture_row) {
            const std::size_t trace_offset = layer * hidden_width;
            const std::size_t mass_offset =
                layer * config_.query_heads;
            const std::size_t slot_mass_offset =
                layer * config_.query_heads *
                config_.episodic_policy.span_size;
            const std::size_t slot_value_offset =
                slot_mass_offset * config_.head_dimension;
            if (capture_regular_entry_trace) {
              const std::size_t entry_offset =
                  layer * config_.query_heads *
                  StreamingAttention::kRegularTraceEntries;
              const std::size_t entry_value_offset =
                  entry_offset * config_.head_dimension;
              const auto regular_component_trace = std::span<float>(
                  next_episodic_mass_regular_component.data() + trace_offset,
                  hidden_width);
              const auto episodic_component_trace = std::span<float>(
                  next_episodic_mass_episodic_component.data() + trace_offset,
                  hidden_width);
              const auto regular_mass_trace = std::span<float>(
                  next_episodic_mass_regular_mass.data() + mass_offset,
                  config_.query_heads);
              const auto episodic_mass_trace = std::span<float>(
                  next_episodic_mass_episodic_mass.data() + mass_offset,
                  config_.query_heads);
              const auto slot_mass_trace = std::span<float>(
                  next_episodic_slot_mass.data() + slot_mass_offset,
                  config_.query_heads * config_.episodic_policy.span_size);
              const auto slot_value_trace = std::span<float>(
                  next_episodic_slot_values.data() + slot_value_offset,
                  config_.query_heads * config_.episodic_policy.span_size *
                      config_.head_dimension);
              const auto entry_mass_trace = std::span<float>(
                  next_regular_entry_mass.data() + entry_offset,
                  config_.query_heads *
                      StreamingAttention::kRegularTraceEntries);
              const auto entry_value_trace = std::span<float>(
                  next_regular_entry_values.data() + entry_value_offset,
                  config_.query_heads *
                      StreamingAttention::kRegularTraceEntries *
                      config_.head_dimension);
              const auto entry_kind_trace = std::span<std::uint8_t>(
                  next_regular_entry_valid_kind.data() + entry_offset,
                  config_.query_heads *
                      StreamingAttention::kRegularTraceEntries);
              const auto entry_position_trace = std::span<std::uint64_t>(
                  next_regular_entry_positions.data() + entry_offset,
                  config_.query_heads *
                      StreamingAttention::kRegularTraceEntries);
              attention_metrics =
                  (capture_qk_row || capture_qk_candidate_key_row)
                  ? attention_[layer]->step_episodic_full_key_candidates_traced(
                        query_row, key_row, value_row, write_slot, read_span,
                        output_row, regular_component_trace,
                        episodic_component_trace, regular_mass_trace,
                        episodic_mass_trace, slot_mass_trace,
                        slot_value_trace, entry_mass_trace, entry_value_trace,
                        entry_kind_trace, entry_position_trace, qk_trace(),
                        qk_candidate_key_trace())
                  : capture_qk_candidate_value_row
                  ? attention_[layer]->step_episodic_full_candidate_values_traced(
                        query_row, key_row, value_row, write_slot, read_span,
                        output_row, regular_component_trace,
                        episodic_component_trace, regular_mass_trace,
                        episodic_mass_trace, slot_mass_trace,
                        slot_value_trace, entry_mass_trace, entry_value_trace,
                        entry_kind_trace, entry_position_trace,
                        qk_candidate_value_trace())
                  : attention_[layer]->step_episodic_full_traced(
                        query_row, key_row, value_row, write_slot, read_span,
                        output_row, regular_component_trace,
                        episodic_component_trace, regular_mass_trace,
                        episodic_mass_trace, slot_mass_trace,
                        slot_value_trace, entry_mass_trace, entry_value_trace,
                        entry_kind_trace, entry_position_trace);
            } else {
              attention_metrics =
                  attention_[layer]->step_episodic_slots_traced(
                    query_row, key_row, value_row, write_slot,
                    read_span, output_row,
                    std::span<float>(
                        next_episodic_mass_regular_component.data() +
                            trace_offset,
                        hidden_width),
                    std::span<float>(
                        next_episodic_mass_episodic_component.data() +
                            trace_offset,
                        hidden_width),
                    std::span<float>(
                        next_episodic_mass_regular_mass.data() +
                            mass_offset,
                        config_.query_heads),
                    std::span<float>(
                        next_episodic_mass_episodic_mass.data() +
                            mass_offset,
                        config_.query_heads),
                    std::span<float>(
                        next_episodic_slot_mass.data() +
                            slot_mass_offset,
                        config_.query_heads *
                            config_.episodic_policy.span_size),
                    std::span<float>(
                        next_episodic_slot_values.data() +
                            slot_value_offset,
                        config_.query_heads *
                            config_.episodic_policy.span_size *
                            config_.head_dimension));
            }
          } else if (capture_qk_row) {
            attention_metrics = attention_[layer]->step_c28_qk_traced(
                query_row, key_row, value_row, write_slot, read_span,
                output_row, qk_trace());
          } else if (!candidate_mask.empty()) {
            attention_metrics = attention_[layer]->step_episodic_masked(
                query_row, key_row, value_row, write_slot, read_span,
                output_row, candidate_mask);
          } else {
            attention_metrics = attention_[layer]->step_episodic(
                query_row, key_row, value_row, write_slot, read_span,
                output_row);
          }
        }
        if (capture_row) {
          std::copy(
              output_row.begin(), output_row.end(),
              next_episodic_mass_base_pre_wo.begin() +
                  static_cast<std::ptrdiff_t>(layer * hidden_width));
        }
        accumulate_attention_metrics(attention_metrics, layer);
        if (!shadow_attention_.empty()) {
          // The shadow consumes the exact same current post-RoPE rows. Since
          // its output never enters hidden state, all cached K/V history is
          // derived exclusively from past base states.
          const auto shadow_output_row = std::span<float>(
              shadow_attention_output.data() + row * hidden_width,
              hidden_width);
          if (capture_row) {
            const std::size_t source_offset =
                row * config_.episodic_policy.span_size;
            shadow_attention_[layer]->step_tracked_positions(
                query_row, key_row, value_row,
                std::span<const std::uint64_t>(
                    read_source_positions.data() + source_offset,
                    config_.episodic_policy.span_size),
                std::span<float>(
                    next_episodic_mass_shadow_source_mass.data() +
                        layer * config_.query_heads,
                    config_.query_heads),
                shadow_output_row);
          } else {
            shadow_attention_[layer]->step(
                query_row, key_row, value_row, shadow_output_row);
          }
        }
      } else {
        for (std::size_t head = 0; head < config_.query_heads; ++head) {
          const std::size_t offset = head * config_.head_dimension;
          accumulate_attention_metrics(
              attention_[layer * config_.query_heads + head]->step(
                  std::span<const float>(
                      query.data() + row * hidden_width + offset,
                      config_.head_dimension),
                  std::span<const float>(
                      key_normalized.data() + row * kv_width + offset,
                      config_.head_dimension),
                  std::span<const float>(
                      value.data() + row * kv_width + offset,
                      config_.head_dimension),
                  std::span<float>(
                      attention_output.data() + row * hidden_width + offset,
                      config_.head_dimension)),
              layer * config_.query_heads + head);
        }
      }
    }
    project(attention_output, weights.output_projection, rows, hidden_width,
            hidden_width, projected);
    if (!shadow_attention_.empty()) {
      project(shadow_attention_output, weights.output_projection, rows,
              hidden_width, hidden_width, shadow_projected);
      const std::size_t row_offset = (rows - 1) * hidden_width;
      const std::size_t trace_offset = layer * hidden_width;
      for (std::size_t column = 0; column < hidden_width; ++column) {
        const float base = projected[row_offset + column];
        next_shadow_base_projected[trace_offset + column] = base;
        next_shadow_target_residual[trace_offset + column] =
            shadow_projected[row_offset + column] - base;
      }
    }
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
  if (!shadow_attention_.empty()) {
    last_shadow_input_norm_ = std::move(next_shadow_input_norm);
    last_shadow_base_projected_ =
        std::move(next_shadow_base_projected);
    last_shadow_target_residual_ =
        std::move(next_shadow_target_residual);
    shadow_trace_valid_ = true;
  }
  if (capture_episodic_mass_trace) {
    last_episodic_mass_base_pre_wo_ =
        std::move(next_episodic_mass_base_pre_wo);
    last_episodic_mass_regular_component_ =
        std::move(next_episodic_mass_regular_component);
    last_episodic_mass_episodic_component_ =
        std::move(next_episodic_mass_episodic_component);
    last_episodic_mass_regular_mass_ =
        std::move(next_episodic_mass_regular_mass);
    last_episodic_mass_episodic_mass_ =
        std::move(next_episodic_mass_episodic_mass);
    last_episodic_mass_shadow_source_mass_ =
        std::move(next_episodic_mass_shadow_source_mass);
    last_episodic_slot_mass_ =
        std::move(next_episodic_slot_mass);
    last_episodic_slot_values_ =
        std::move(next_episodic_slot_values);
    episodic_mass_trace_valid_ = true;
    episodic_slot_trace_valid_ = true;
    if (capture_regular_entry_trace) {
      last_regular_entry_mass_ =
          std::move(next_regular_entry_mass);
      last_regular_entry_values_ =
          std::move(next_regular_entry_values);
      last_regular_entry_valid_kind_ =
          std::move(next_regular_entry_valid_kind);
      last_regular_entry_positions_ =
          std::move(next_regular_entry_positions);
      regular_entry_trace_valid_ = true;
    } else {
      regular_entry_trace_valid_ = false;
    }
  } else {
    episodic_mass_trace_valid_ = false;
    episodic_slot_trace_valid_ = false;
    regular_entry_trace_valid_ = false;
  }
  if (capture_c28_qk_partial_trace) {
    if (capture_c28_qk_candidate_trace) {
      const std::size_t layer_partial_count =
          config_.query_heads * StreamingAttention::kC28TraceEntries *
          StreamingAttention::kQKPartialBands;
      const std::size_t layer_candidate_count =
          config_.query_heads * config_.older_candidates *
          StreamingAttention::kQKPartialBands;
      last_c28_qk_partials_.resize(qk_partial_count);
      last_c28_qk_candidates_.resize(qk_candidate_count);
      for (std::size_t layer = 0; layer < config_.layers; ++layer) {
        const std::size_t source =
            layer * (layer_partial_count + layer_candidate_count);
        std::copy_n(next_c28_qk_partials.data() + source,
                    layer_partial_count,
                    last_c28_qk_partials_.data() +
                        layer * layer_partial_count);
        std::copy_n(next_c28_qk_partials.data() + source +
                        layer_partial_count,
                    layer_candidate_count,
                    last_c28_qk_candidates_.data() +
                        layer * layer_candidate_count);
      }
      c28_qk_candidate_trace_valid_ = true;
    } else {
      last_c28_qk_partials_ = std::move(next_c28_qk_partials);
      c28_qk_candidate_trace_valid_ = false;
    }
    c28_qk_partial_trace_valid_ = true;
  } else {
    c28_qk_partial_trace_valid_ = false;
    c28_qk_candidate_trace_valid_ = false;
  }
  if (capture_c28_qk_candidate_key_trace) {
    last_c28_qk_candidate_keys_ = std::move(next_c28_qk_candidate_keys);
    c28_qk_candidate_key_trace_valid_ = true;
  } else {
    c28_qk_candidate_key_trace_valid_ = false;
  }
  if (capture_c28_qk_candidate_value_trace) {
    last_c28_qk_candidate_values_ = std::move(next_c28_qk_candidate_values);
    c28_qk_candidate_value_trace_valid_ = true;
  } else {
    c28_qk_candidate_value_trace_valid_ = false;
  }
  position_ += rows;
  metrics_.positions_processed += rows;
  metrics_.episodic_active_slots = std::accumulate(
      episodic_active_slots_by_cache_.begin(),
      episodic_active_slots_by_cache_.end(), std::uint64_t{0});
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
  for (auto& cache : shadow_attention_) cache->reset();
  position_ = 0;
  metrics_ = {};
  std::fill(episodic_active_slots_by_cache_.begin(),
            episodic_active_slots_by_cache_.end(), std::uint64_t{0});
  std::fill(episodic_slot_positions_.begin(),
            episodic_slot_positions_.end(),
            std::numeric_limits<std::uint64_t>::max());
  last_final_state_.clear();
  last_vocabulary_scores_.clear();
  shadow_trace_valid_ = false;
  episodic_mass_trace_valid_ = false;
  episodic_slot_trace_valid_ = false;
  regular_entry_trace_valid_ = false;
  c28_qk_partial_trace_valid_ = false;
  c28_qk_candidate_trace_valid_ = false;
  c28_qk_candidate_key_trace_valid_ = false;
  c28_qk_candidate_value_trace_valid_ = false;
}

bool OLMoETokenRuntime::is_eos(const std::int64_t token) const {
  return std::find(config_.eos_token_ids.begin(),
                   config_.eos_token_ids.end(),
                   token) != config_.eos_token_ids.end();
}

}  // namespace engram
