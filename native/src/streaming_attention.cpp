#include "engram/streaming_attention.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace engram {
namespace {

std::size_t checked_product(const std::size_t left, const std::size_t right,
                            const char* name) {
  if (left != 0 &&
      right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::invalid_argument(name);
  }
  return left * right;
}

std::size_t checked_sum(const std::size_t left, const std::size_t right,
                        const char* name) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::invalid_argument(name);
  }
  return left + right;
}

float dot(const float* left, const float* right,
          const std::size_t width) noexcept {
  float result = 0.0F;
  for (std::size_t index = 0; index < width; ++index) {
    result += left[index] * right[index];
  }
  return result;
}

float dot_rope_bands(const float* left, const float* right,
                     const std::size_t width, const float scale,
                     const std::span<float> partials) noexcept {
  std::fill(partials.begin(), partials.end(), 0.0F);
  float result = 0.0F;
  const std::size_t half = width / 2;
  for (std::size_t index = 0; index < width; ++index) {
    const float product = left[index] * right[index];
    // Keep the legacy scalar accumulation in its original dimension order.
    result += product;
    const std::size_t frequency = index < half ? index : index - half;
    partials[frequency / StreamingAttention::kQKPartialBandHalfWidth] +=
        product;
  }
  for (float& partial : partials) {
    partial *= scale;
  }
  return result;
}

float bf16_to_float(const std::uint16_t value) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::uint16_t float_to_bf16(const float value) noexcept {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t rounding_bias =
      0x7FFFU + ((bits >> 16U) & 1U);
  bits += rounding_bias;
  return static_cast<std::uint16_t>(bits >> 16U);
}

std::uint16_t float_to_fp16(const float value) noexcept {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::uint32_t exponent = (bits >> 23U) & 0xFFU;
  const std::uint32_t mantissa = bits & 0x7FFFFFU;
  if (exponent == 0xFFU) {
    return static_cast<std::uint16_t>(sign | 0x7C00U |
                                       (mantissa != 0U ? 0x0200U : 0U));
  }
  const int unbiased = static_cast<int>(exponent) - 127;
  if (unbiased > 15) return static_cast<std::uint16_t>(sign | 0x7C00U);
  if (unbiased < -14) {
    if (unbiased < -24) return static_cast<std::uint16_t>(sign);
    const std::uint32_t shifted = mantissa | 0x800000U;
    const int shift = -unbiased - 14;
    const std::uint32_t rounded =
        (shifted + (1U << (shift + 12))) >> (shift + 13);
    return static_cast<std::uint16_t>(sign | rounded);
  }
  std::uint32_t half_exponent = static_cast<std::uint32_t>(unbiased + 15);
  std::uint32_t half_mantissa = (mantissa + 0x1000U) >> 13U;
  if (half_mantissa == 0x400U) {
    half_mantissa = 0;
    ++half_exponent;
  }
  if (half_exponent >= 0x1FU) return static_cast<std::uint16_t>(sign | 0x7C00U);
  return static_cast<std::uint16_t>(sign | (half_exponent << 10U) |
                                    half_mantissa);
}

float fp16_to_float(const std::uint16_t value) noexcept {
  const std::uint32_t sign = (static_cast<std::uint32_t>(value) & 0x8000U)
                             << 16U;
  const std::uint32_t exponent = (value >> 10U) & 0x1FU;
  const std::uint32_t mantissa = value & 0x03FFU;
  std::uint32_t bits{};
  if (exponent == 0U) {
    if (mantissa == 0U) {
      bits = sign;
    } else {
      std::uint32_t normalized = mantissa;
      int exponent_value = -14;
      while ((normalized & 0x0400U) == 0U) {
        normalized <<= 1U;
        --exponent_value;
      }
      normalized &= 0x03FFU;
      bits = sign |
             (static_cast<std::uint32_t>(exponent_value + 127) << 23U) |
             (normalized << 13U);
    }
  } else if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (mantissa << 13U);
  } else {
    bits = sign |
           (static_cast<std::uint32_t>(exponent - 15U + 127U) << 23U) |
           (mantissa << 13U);
  }
  return std::bit_cast<float>(bits);
}

float dot_bf16(const float* left, const std::uint16_t* right,
               const std::size_t width) noexcept {
  float result = 0.0F;
  for (std::size_t index = 0; index < width; ++index) {
    result += left[index] * bf16_to_float(right[index]);
  }
  return result;
}

float dot_bf16_rope_bands(const float* left, const std::uint16_t* right,
                          const std::size_t width, const float scale,
                          const std::span<float> partials) noexcept {
  std::fill(partials.begin(), partials.end(), 0.0F);
  float result = 0.0F;
  const std::size_t half = width / 2;
  for (std::size_t index = 0; index < width; ++index) {
    const float product = left[index] * bf16_to_float(right[index]);
    // Keep the legacy scalar accumulation in its original dimension order.
    result += product;
    const std::size_t frequency = index < half ? index : index - half;
    partials[frequency / StreamingAttention::kQKPartialBandHalfWidth] +=
        product;
  }
  for (float& partial : partials) {
    partial *= scale;
  }
  return result;
}

float dot_fp16(const float* left, const std::uint16_t* right,
               const std::size_t width) noexcept {
  float result = 0.0F;
  for (std::size_t index = 0; index < width; ++index) {
    result += left[index] * fp16_to_float(right[index]);
  }
  return result;
}

float dot_fp16_rope_bands(const float* left, const std::uint16_t* right,
                          const std::size_t width, const float scale,
                          const std::span<float> partials) noexcept {
  std::fill(partials.begin(), partials.end(), 0.0F);
  float result = 0.0F;
  const std::size_t half = width / 2;
  for (std::size_t index = 0; index < width; ++index) {
    const float product = left[index] * fp16_to_float(right[index]);
    result += product;
    const std::size_t frequency = index < half ? index : index - half;
    partials[frequency / StreamingAttention::kQKPartialBandHalfWidth] +=
        product;
  }
  for (float& partial : partials) partial *= scale;
  return result;
}

float dot_int8(const float* left, const std::int8_t* right,
               const float scale, const std::size_t width) noexcept {
  float result = 0.0F;
  for (std::size_t index = 0; index < width; ++index) {
    result += left[index] * (static_cast<float>(right[index]) * scale);
  }
  return result;
}

float dot_int8_rope_bands(const float* left, const std::int8_t* right,
                          const float value_scale, const std::size_t width,
                          const float scale,
                          const std::span<float> partials) noexcept {
  std::fill(partials.begin(), partials.end(), 0.0F);
  float result = 0.0F;
  const std::size_t half = width / 2;
  for (std::size_t index = 0; index < width; ++index) {
    const float product =
        left[index] * (static_cast<float>(right[index]) * value_scale);
    result += product;
    const std::size_t frequency = index < half ? index : index - half;
    partials[frequency / StreamingAttention::kQKPartialBandHalfWidth] +=
        product;
  }
  for (float& partial : partials) partial *= scale;
  return result;
}

float int8_scale(const std::span<const float> values) noexcept {
  float maximum = 0.0F;
  for (const float value : values) maximum = std::max(maximum, std::abs(value));
  return maximum > 0.0F ? maximum / 127.0F : 1.0F;
}

std::int8_t float_to_int8(const float value, const float scale) noexcept {
  const float quantized = std::round(value / scale);
  return static_cast<std::int8_t>(std::clamp(quantized, -127.0F, 127.0F));
}

}  // namespace

StreamingAttention::StreamingAttention(StreamingAttentionConfig config)
    : config_(config) {
  if (config_.query_heads == 0 || config_.key_value_heads == 0 ||
      config_.head_dimension == 0 || config_.local_window == 0 ||
      config_.older_candidates == 0 || config_.older_top_k == 0) {
    throw std::invalid_argument(
        "streaming attention dimensions and capacities must be positive");
  }
  if (config_.query_heads % config_.key_value_heads != 0) {
    throw std::invalid_argument(
        "query heads must be divisible by key/value heads");
  }
  if (config_.older_top_k > config_.older_candidates ||
      config_.sink_tokens > config_.older_candidates) {
    throw std::invalid_argument(
        "streaming attention capacities are inconsistent");
  }
  if (!std::isfinite(config_.scale) || config_.scale <= 0.0F) {
    throw std::invalid_argument("streaming attention scale must be positive");
  }
  if (!std::isfinite(config_.episodic_logit_bias)) {
    throw std::invalid_argument(
        "streaming episodic logit bias must be finite");
  }
  if ((config_.local_bf16 ? 1 : 0) +
          (config_.local_values_bf16 ? 1 : 0) +
          (config_.local_values_fp16 ? 1 : 0) +
          (config_.local_fp16 ? 1 : 0) +
          (config_.local_int8 ? 1 : 0) >
      1) {
    throw std::invalid_argument("streaming local compression modes conflict");
  }
  if ((config_.episodic_slots == 0) !=
          (config_.episodic_span_size == 0) ||
      (config_.episodic_slots != 0 &&
       (config_.episodic_span_size > config_.episodic_slots ||
        config_.episodic_slots % config_.episodic_span_size != 0)) ||
      (config_.episodic_slots == 0 &&
       (!config_.episodic_head_mask.empty() ||
        config_.episodic_logit_bias != 0.0F)) ||
      (!config_.episodic_head_mask.empty() &&
       (config_.episodic_head_mask.size() != config_.query_heads ||
        !std::all_of(config_.episodic_head_mask.begin(),
                     config_.episodic_head_mask.end(),
                     [](const std::uint8_t selected) {
                       return selected <= 1;
                     }) ||
        std::none_of(config_.episodic_head_mask.begin(),
                     config_.episodic_head_mask.end(),
                     [](const std::uint8_t selected) {
                       return selected != 0;
                     })))) {
    throw std::invalid_argument(
        "streaming episodic capacities are inconsistent");
  }
  groups_ = config_.query_heads / config_.key_value_heads;
  const std::size_t recent_vectors =
      checked_product(config_.local_window, config_.key_value_heads,
                      "streaming recent vector count overflows");
  const std::size_t recent_elements =
      checked_product(recent_vectors, config_.head_dimension,
                      "streaming recent storage overflows");
  const std::size_t older_vectors =
      checked_product(config_.query_heads, config_.older_candidates,
                      "streaming older vector count overflows");
  const std::size_t older_elements =
      checked_product(older_vectors, config_.head_dimension,
                      "streaming older storage overflows");
  const std::size_t episodic_vectors =
      checked_product(config_.episodic_slots, config_.key_value_heads,
                      "streaming episodic vector count overflows");
  const std::size_t episodic_elements =
      checked_product(episodic_vectors, config_.head_dimension,
                      "streaming episodic storage overflows");
  if (config_.local_bf16) {
    recent_keys_bf16_.resize(recent_elements);
    recent_values_bf16_.resize(recent_elements);
  } else if (config_.local_values_bf16) {
    recent_keys_.resize(recent_elements);
    recent_values_bf16_.resize(recent_elements);
  } else if (config_.local_values_fp16) {
    recent_keys_.resize(recent_elements);
    recent_values_fp16_.resize(recent_elements);
  } else if (config_.local_fp16) {
    recent_keys_fp16_.resize(recent_elements);
    recent_values_fp16_.resize(recent_elements);
  } else if (config_.local_int8) {
    recent_keys_int8_.resize(recent_elements);
    recent_values_int8_.resize(recent_elements);
    recent_key_scales_int8_.resize(recent_vectors);
    recent_value_scales_int8_.resize(recent_vectors);
  } else {
    recent_keys_.resize(recent_elements);
    recent_values_.resize(recent_elements);
  }
  recent_mass_.resize(config_.query_heads * config_.local_window);
  recent_positions_.resize(config_.local_window);
  older_keys_.resize(older_elements);
  older_values_.resize(older_elements);
  older_scores_.resize(older_vectors);
  older_positions_.resize(older_vectors);
  older_active_.resize(older_vectors);
  episodic_keys_.resize(episodic_elements);
  episodic_values_.resize(episodic_elements);
  episodic_positions_.resize(config_.episodic_slots);
  const std::size_t score_capacity = checked_sum(
      checked_sum(config_.local_window, config_.older_candidates,
                  "streaming score scratch overflows"),
      config_.episodic_span_size, "streaming score scratch overflows");
  const std::size_t weight_capacity = checked_sum(
      checked_sum(config_.local_window, config_.older_top_k,
                  "streaming weight scratch overflows"),
      config_.episodic_span_size, "streaming weight scratch overflows");
  score_scratch_.resize(score_capacity);
  candidate_score_scratch_.resize(config_.older_candidates);
  weight_scratch_.resize(weight_capacity);
  selected_scratch_.resize(config_.older_top_k);
  reset();
}

void StreamingAttention::reset() noexcept {
  tokens_seen_ = 0;
  recent_start_ = 0;
  recent_size_ = 0;
  std::fill(recent_mass_.begin(), recent_mass_.end(), 0.0F);
  std::fill(older_scores_.begin(), older_scores_.end(), 0.0F);
  std::fill(older_positions_.begin(), older_positions_.end(), 0);
  std::fill(older_active_.begin(), older_active_.end(), std::uint8_t{0});
  std::fill(episodic_positions_.begin(), episodic_positions_.end(),
            kNoEpisodicDirective);
  episodic_active_slots_ = 0;
}

const StreamingAttentionConfig& StreamingAttention::config() const noexcept {
  return config_;
}

std::size_t StreamingAttention::tokens_seen() const noexcept {
  return tokens_seen_;
}

std::size_t StreamingAttention::recent_offset(
    const std::size_t slot, const std::size_t kv_head) const noexcept {
  return (slot * config_.key_value_heads + kv_head) * config_.head_dimension;
}

std::size_t StreamingAttention::older_offset(
    const std::size_t head, const std::size_t slot) const noexcept {
  return (head * config_.older_candidates + slot) * config_.head_dimension;
}

std::size_t StreamingAttention::episodic_offset(
    const std::size_t slot, const std::size_t kv_head) const noexcept {
  return (slot * config_.key_value_heads + kv_head) *
         config_.head_dimension;
}

float StreamingAttention::recent_key_at(const std::size_t slot,
                                         const std::size_t kv_head,
                                         const std::size_t dimension) const
    noexcept {
  const std::size_t offset = recent_offset(slot, kv_head) + dimension;
  if (config_.local_bf16) return bf16_to_float(recent_keys_bf16_[offset]);
  if (config_.local_fp16) return fp16_to_float(recent_keys_fp16_[offset]);
  if (config_.local_int8) {
    const std::size_t vector = slot * config_.key_value_heads + kv_head;
    return static_cast<float>(recent_keys_int8_[offset]) *
           recent_key_scales_int8_[vector];
  }
  return recent_keys_[offset];
}

float StreamingAttention::recent_value_at(const std::size_t slot,
                                           const std::size_t kv_head,
                                           const std::size_t dimension) const
    noexcept {
  const std::size_t offset = recent_offset(slot, kv_head) + dimension;
  if (config_.local_bf16 || config_.local_values_bf16) {
    return bf16_to_float(recent_values_bf16_[offset]);
  }
  if (config_.local_values_fp16 || config_.local_fp16) {
    return fp16_to_float(recent_values_fp16_[offset]);
  }
  if (config_.local_int8) {
    const std::size_t vector = slot * config_.key_value_heads + kv_head;
    return static_cast<float>(recent_values_int8_[offset]) *
           recent_value_scales_int8_[vector];
  }
  return recent_values_[offset];
}

float StreamingAttention::recent_key_dot(
    const float* query, const std::size_t slot, const std::size_t kv_head,
    const std::span<float> partials) const noexcept {
  const std::size_t offset = recent_offset(slot, kv_head);
  if (config_.local_bf16) {
    if (partials.empty()) {
      return dot_bf16(query, recent_keys_bf16_.data() + offset,
                      config_.head_dimension);
    }
    return dot_bf16_rope_bands(
        query, recent_keys_bf16_.data() + offset, config_.head_dimension,
        config_.scale, partials);
  }
  if (config_.local_fp16) {
    if (partials.empty()) {
      return dot_fp16(query, recent_keys_fp16_.data() + offset,
                      config_.head_dimension);
    }
    return dot_fp16_rope_bands(
        query, recent_keys_fp16_.data() + offset, config_.head_dimension,
        config_.scale, partials);
  }
  if (config_.local_int8) {
    const std::size_t vector = slot * config_.key_value_heads + kv_head;
    if (partials.empty()) {
      return dot_int8(query, recent_keys_int8_.data() + offset,
                      recent_key_scales_int8_[vector], config_.head_dimension);
    }
    return dot_int8_rope_bands(
        query, recent_keys_int8_.data() + offset,
        recent_key_scales_int8_[vector], config_.head_dimension, config_.scale,
        partials);
  }
  if (partials.empty()) {
    return dot(query, recent_keys_.data() + offset, config_.head_dimension);
  }
  return dot_rope_bands(query, recent_keys_.data() + offset,
                        config_.head_dimension, config_.scale, partials);
}

void StreamingAttention::store_recent(
    const std::size_t slot, const std::size_t kv_head,
    const std::span<const float> key,
    const std::span<const float> value) noexcept {
  const std::size_t target = recent_offset(slot, kv_head);
  if (config_.local_bf16) {
    for (std::size_t dimension = 0; dimension < config_.head_dimension;
         ++dimension) {
      recent_keys_bf16_[target + dimension] = float_to_bf16(key[dimension]);
      recent_values_bf16_[target + dimension] =
          float_to_bf16(value[dimension]);
    }
    return;
  }
  if (config_.local_fp16) {
    for (std::size_t dimension = 0; dimension < config_.head_dimension;
         ++dimension) {
      recent_keys_fp16_[target + dimension] = float_to_fp16(key[dimension]);
      recent_values_fp16_[target + dimension] =
          float_to_fp16(value[dimension]);
    }
    return;
  }
  if (config_.local_int8) {
    const std::size_t vector = slot * config_.key_value_heads + kv_head;
    const float key_scale = int8_scale(key);
    const float value_scale = int8_scale(value);
    recent_key_scales_int8_[vector] = key_scale;
    recent_value_scales_int8_[vector] = value_scale;
    for (std::size_t dimension = 0; dimension < config_.head_dimension;
         ++dimension) {
      recent_keys_int8_[target + dimension] =
          float_to_int8(key[dimension], key_scale);
      recent_values_int8_[target + dimension] =
          float_to_int8(value[dimension], value_scale);
    }
    return;
  }
  std::copy_n(key.data(), config_.head_dimension,
              recent_keys_.data() + target);
  if (config_.local_values_bf16) {
    for (std::size_t dimension = 0; dimension < config_.head_dimension;
         ++dimension) {
      recent_values_bf16_[target + dimension] =
          float_to_bf16(value[dimension]);
    }
  } else if (config_.local_values_fp16) {
    for (std::size_t dimension = 0; dimension < config_.head_dimension;
         ++dimension) {
      recent_values_fp16_[target + dimension] =
          float_to_fp16(value[dimension]);
    }
  } else {
    std::copy_n(value.data(), config_.head_dimension,
                recent_values_.data() + target);
  }
}

void StreamingAttention::evict_recent(
    const std::size_t slot, std::uint64_t& sink_insertions,
    std::uint64_t& heavy_hitter_updates) {
  const std::uint64_t position = recent_positions_[slot];
  for (std::size_t head = 0; head < config_.query_heads; ++head) {
    const std::size_t base = head * config_.older_candidates;
    const float incoming_score =
        recent_mass_[head * config_.local_window + slot];
    std::size_t destination = config_.older_candidates;
    if (position < config_.sink_tokens) {
      destination = static_cast<std::size_t>(position);
    } else {
      for (std::size_t candidate = config_.sink_tokens;
           candidate < config_.older_candidates; ++candidate) {
        if (older_active_[base + candidate] == 0) {
          destination = candidate;
          break;
        }
      }
      if (destination == config_.older_candidates) {
        destination = config_.sink_tokens;
        for (std::size_t candidate = config_.sink_tokens + 1;
             candidate < config_.older_candidates; ++candidate) {
          const std::size_t current = base + candidate;
          const std::size_t selected = base + destination;
          if (older_scores_[current] < older_scores_[selected] ||
              (older_scores_[current] == older_scores_[selected] &&
               older_positions_[current] < older_positions_[selected])) {
            destination = candidate;
          }
        }
        if (incoming_score < older_scores_[base + destination]) {
          continue;
        }
      }
    }
    const std::size_t index = base + destination;
    const std::size_t kv_head = head / groups_;
    const std::size_t target = older_offset(head, destination);
    for (std::size_t dimension = 0; dimension < config_.head_dimension;
         ++dimension) {
      older_keys_[target + dimension] =
          recent_key_at(slot, kv_head, dimension);
      older_values_[target + dimension] =
          recent_value_at(slot, kv_head, dimension);
    }
    older_scores_[index] = incoming_score;
    older_positions_[index] = position;
    older_active_[index] = 1;
    if (position < config_.sink_tokens) {
      ++sink_insertions;
    } else {
      ++heavy_hitter_updates;
    }
  }
}

void StreamingAttention::validate_inputs(
    const std::span<const float> query, const std::span<const float> key,
    const std::span<const float> value, const std::span<float> output) const {
  const std::size_t query_elements =
      config_.query_heads * config_.head_dimension;
  const std::size_t kv_elements =
      config_.key_value_heads * config_.head_dimension;
  if (query.size() != query_elements || output.size() != query_elements ||
      key.size() != kv_elements || value.size() != kv_elements) {
    throw std::invalid_argument("streaming attention input shape mismatch");
  }
  const auto finite = [](const float element) { return std::isfinite(element); };
  if (!std::all_of(query.begin(), query.end(), finite) ||
      !std::all_of(key.begin(), key.end(), finite) ||
      !std::all_of(value.begin(), value.end(), finite)) {
    throw std::invalid_argument("streaming attention input is non-finite");
  }
}

StreamingAttentionMetrics StreamingAttention::step(
    const std::span<const float> query, const std::span<const float> key,
    const std::span<const float> value, const std::span<float> output) {
  return step_episodic(query, key, value, kNoEpisodicDirective,
                       kNoEpisodicDirective, output);
}

StreamingAttentionMetrics StreamingAttention::step_episodic(
    const std::span<const float> query, const std::span<const float> key,
    const std::span<const float> value, const std::size_t episodic_write_slot,
    const std::size_t episodic_read_span,
    const std::span<float> output) {
  return step_episodic_impl(
      query, key, value, episodic_write_slot, episodic_read_span, output,
      {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {});
}

StreamingAttentionMetrics StreamingAttention::step_episodic_masked(
    const std::span<const float> query, const std::span<const float> key,
    const std::span<const float> value, const std::size_t write_slot,
    const std::size_t read_span, const std::span<float> output,
    const std::span<const std::uint8_t> candidate_mask) {
  const std::size_t mask_count =
      config_.query_heads * config_.older_candidates;
  if (candidate_mask.size() != mask_count ||
      !std::all_of(candidate_mask.begin(), candidate_mask.end(),
                   [](const std::uint8_t value) { return value <= 1; }) ||
      std::none_of(candidate_mask.begin(), candidate_mask.end(),
                   [](const std::uint8_t value) { return value != 0; })) {
    throw std::invalid_argument("streaming candidate mask shape is invalid");
  }
  return step_episodic_impl(
      query, key, value, write_slot, read_span, output, {}, {}, {}, {}, {},
      {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, candidate_mask);
}

StreamingAttentionMetrics StreamingAttention::step_episodic_traced(
    const std::span<const float> query, const std::span<const float> key,
    const std::span<const float> value,
    const std::size_t episodic_write_slot,
    const std::size_t episodic_read_span,
    const std::span<float> output,
    const std::span<float> regular_component,
    const std::span<float> episodic_component,
    const std::span<float> regular_mass,
    const std::span<float> episodic_mass) {
  const std::size_t query_elements =
      config_.query_heads * config_.head_dimension;
  if (regular_component.size() != query_elements ||
      episodic_component.size() != query_elements ||
      regular_mass.size() != config_.query_heads ||
      episodic_mass.size() != config_.query_heads) {
    throw std::invalid_argument(
        "streaming episodic trace shape mismatch");
  }
  return step_episodic_impl(
      query, key, value, episodic_write_slot, episodic_read_span, output,
      regular_component, episodic_component, regular_mass, episodic_mass,
      {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {});
}

StreamingAttentionMetrics StreamingAttention::step_episodic_slots_traced(
    const std::span<const float> query, const std::span<const float> key,
    const std::span<const float> value,
    const std::size_t episodic_write_slot,
    const std::size_t episodic_read_span,
    const std::span<float> output,
    const std::span<float> regular_component,
    const std::span<float> episodic_component,
    const std::span<float> regular_mass,
    const std::span<float> episodic_mass,
    const std::span<float> slot_mass,
    const std::span<float> slot_values) {
  const std::size_t query_elements =
      config_.query_heads * config_.head_dimension;
  const std::size_t slot_count =
      config_.query_heads * config_.episodic_span_size;
  const std::size_t slot_value_count =
      slot_count * config_.head_dimension;
  if (episodic_read_span == kNoEpisodicDirective ||
      regular_component.size() != query_elements ||
      episodic_component.size() != query_elements ||
      regular_mass.size() != config_.query_heads ||
      episodic_mass.size() != config_.query_heads ||
      slot_mass.size() != slot_count ||
      slot_values.size() != slot_value_count) {
    throw std::invalid_argument(
        "streaming episodic slot trace shape or read mismatch");
  }
  return step_episodic_impl(
      query, key, value, episodic_write_slot, episodic_read_span, output,
      regular_component, episodic_component, regular_mass, episodic_mass,
      {}, {}, slot_mass, slot_values, {}, {}, {}, {}, {}, {}, {});
}

StreamingAttentionMetrics StreamingAttention::step_regular_entries_traced(
    const std::span<const float> query, const std::span<const float> key,
    const std::span<const float> value, const std::span<float> output,
    const std::span<float> entry_mass,
    const std::span<float> entry_values,
    const std::span<std::uint8_t> valid_kind,
    const std::span<std::uint64_t> positions) {
  return step_episodic_impl(
      query, key, value, kNoEpisodicDirective, kNoEpisodicDirective,
      output, {}, {}, {}, {}, {}, {}, {}, {}, entry_mass, entry_values,
      valid_kind, positions, {}, {}, {});
}

StreamingAttentionMetrics
StreamingAttention::step_regular_entries_qk_traced(
    const std::span<const float> query, const std::span<const float> key,
    const std::span<const float> value, const std::span<float> output,
    const std::span<float> entry_mass,
    const std::span<float> entry_values,
    const std::span<std::uint8_t> valid_kind,
    const std::span<std::uint64_t> positions,
    const std::span<float> qk_partials) {
  return step_episodic_impl(
      query, key, value, kNoEpisodicDirective, kNoEpisodicDirective,
      output, {}, {}, {}, {}, {}, {}, {}, {}, entry_mass, entry_values,
      valid_kind, positions, qk_partials, {}, {});
}

StreamingAttentionMetrics StreamingAttention::step_c28_qk_traced(
    const std::span<const float> query, const std::span<const float> key,
    const std::span<const float> value,
    const std::size_t episodic_write_slot,
    const std::size_t episodic_read_span,
    const std::span<float> output,
    const std::span<float> qk_partials) {
  return step_episodic_impl(
      query, key, value, episodic_write_slot, episodic_read_span, output,
      {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, qk_partials, {}, {});
}

StreamingAttentionMetrics StreamingAttention::step_episodic_full_traced(
    const std::span<const float> query, const std::span<const float> key,
    const std::span<const float> value,
    const std::size_t episodic_write_slot,
    const std::size_t episodic_read_span,
    const std::span<float> output,
    const std::span<float> regular_component,
    const std::span<float> episodic_component,
    const std::span<float> regular_mass,
    const std::span<float> episodic_mass,
    const std::span<float> slot_mass,
    const std::span<float> slot_values,
    const std::span<float> regular_entry_mass,
    const std::span<float> regular_entry_values,
    const std::span<std::uint8_t> regular_entry_valid_kind,
    const std::span<std::uint64_t> regular_entry_positions) {
  const std::size_t query_elements =
      config_.query_heads * config_.head_dimension;
  const std::size_t slot_count =
      config_.query_heads * config_.episodic_span_size;
  const std::size_t slot_value_count =
      slot_count * config_.head_dimension;
  if (episodic_read_span == kNoEpisodicDirective ||
      regular_component.size() != query_elements ||
      episodic_component.size() != query_elements ||
      regular_mass.size() != config_.query_heads ||
      episodic_mass.size() != config_.query_heads ||
      slot_mass.size() != slot_count ||
      slot_values.size() != slot_value_count) {
    throw std::invalid_argument(
        "streaming full episodic trace shape or read mismatch");
  }
  return step_episodic_impl(
      query, key, value, episodic_write_slot, episodic_read_span, output,
      regular_component, episodic_component, regular_mass, episodic_mass,
      {}, {}, slot_mass, slot_values, regular_entry_mass,
      regular_entry_values, regular_entry_valid_kind,
      regular_entry_positions, {}, {}, {});
}

StreamingAttentionMetrics
StreamingAttention::step_episodic_full_qk_traced(
    const std::span<const float> query, const std::span<const float> key,
    const std::span<const float> value,
    const std::size_t episodic_write_slot,
    const std::size_t episodic_read_span,
    const std::span<float> output,
    const std::span<float> regular_component,
    const std::span<float> episodic_component,
    const std::span<float> regular_mass,
    const std::span<float> episodic_mass,
    const std::span<float> slot_mass,
    const std::span<float> slot_values,
    const std::span<float> regular_entry_mass,
    const std::span<float> regular_entry_values,
    const std::span<std::uint8_t> regular_entry_valid_kind,
    const std::span<std::uint64_t> regular_entry_positions,
    const std::span<float> qk_partials) {
  return step_episodic_impl(
      query, key, value, episodic_write_slot, episodic_read_span, output,
      regular_component, episodic_component, regular_mass, episodic_mass,
      {}, {}, slot_mass, slot_values, regular_entry_mass,
      regular_entry_values, regular_entry_valid_kind,
      regular_entry_positions, qk_partials, {}, {});
}

StreamingAttentionMetrics
StreamingAttention::step_episodic_full_qk_candidates_traced(
    const std::span<const float> query, const std::span<const float> key,
    const std::span<const float> value,
    const std::size_t episodic_write_slot,
    const std::size_t episodic_read_span,
    const std::span<float> output,
    const std::span<float> regular_component,
    const std::span<float> episodic_component,
    const std::span<float> regular_mass,
    const std::span<float> episodic_mass,
    const std::span<float> slot_mass,
    const std::span<float> slot_values,
    const std::span<float> regular_entry_mass,
    const std::span<float> regular_entry_values,
    const std::span<std::uint8_t> regular_entry_valid_kind,
    const std::span<std::uint64_t> regular_entry_positions,
    const std::span<float> qk_partials) {
  return step_episodic_impl(
      query, key, value, episodic_write_slot, episodic_read_span, output,
      regular_component, episodic_component, regular_mass, episodic_mass,
      {}, {}, slot_mass, slot_values, regular_entry_mass,
      regular_entry_values, regular_entry_valid_kind,
      regular_entry_positions, qk_partials, {}, {});
}

StreamingAttentionMetrics
StreamingAttention::step_episodic_full_key_candidates_traced(
    const std::span<const float> query, const std::span<const float> key,
    const std::span<const float> value,
    const std::size_t episodic_write_slot,
    const std::size_t episodic_read_span,
    const std::span<float> output,
    const std::span<float> regular_component,
    const std::span<float> episodic_component,
    const std::span<float> regular_mass,
    const std::span<float> episodic_mass,
    const std::span<float> slot_mass,
    const std::span<float> slot_values,
    const std::span<float> regular_entry_mass,
    const std::span<float> regular_entry_values,
    const std::span<std::uint8_t> regular_entry_valid_kind,
    const std::span<std::uint64_t> regular_entry_positions,
    const std::span<float> qk_partials,
    const std::span<float> candidate_keys) {
  return step_episodic_impl(
      query, key, value, episodic_write_slot, episodic_read_span, output,
      regular_component, episodic_component, regular_mass, episodic_mass,
      {}, {}, slot_mass, slot_values, regular_entry_mass,
      regular_entry_values, regular_entry_valid_kind,
      regular_entry_positions, qk_partials, candidate_keys, {});
}

StreamingAttentionMetrics
StreamingAttention::step_episodic_full_candidate_values_traced(
    const std::span<const float> query, const std::span<const float> key,
    const std::span<const float> value,
    const std::size_t episodic_write_slot,
    const std::size_t episodic_read_span,
    const std::span<float> output,
    const std::span<float> regular_component,
    const std::span<float> episodic_component,
    const std::span<float> regular_mass,
    const std::span<float> episodic_mass,
    const std::span<float> slot_mass,
    const std::span<float> slot_values,
    const std::span<float> regular_entry_mass,
    const std::span<float> regular_entry_values,
    const std::span<std::uint8_t> regular_entry_valid_kind,
    const std::span<std::uint64_t> regular_entry_positions,
    const std::span<float> candidate_values) {
  return step_episodic_impl(
      query, key, value, episodic_write_slot, episodic_read_span, output,
      regular_component, episodic_component, regular_mass, episodic_mass,
      {}, {}, slot_mass, slot_values, regular_entry_mass,
      regular_entry_values, regular_entry_valid_kind,
      regular_entry_positions, {}, {}, candidate_values);
}

StreamingAttentionMetrics StreamingAttention::step_tracked_positions(
    const std::span<const float> query, const std::span<const float> key,
    const std::span<const float> value,
    const std::span<const std::uint64_t> tracked_positions,
    const std::span<float> tracked_mass,
    const std::span<float> output) {
  if (tracked_positions.empty() ||
      tracked_mass.size() != config_.query_heads) {
    throw std::invalid_argument(
        "streaming tracked-position trace shape mismatch");
  }
  return step_episodic_impl(
      query, key, value, kNoEpisodicDirective, kNoEpisodicDirective,
      output, {}, {}, {}, {}, tracked_positions, tracked_mass, {}, {}, {},
      {}, {}, {}, {}, {}, {});
}

StreamingAttentionMetrics StreamingAttention::step_episodic_impl(
    const std::span<const float> query, const std::span<const float> key,
    const std::span<const float> value,
    const std::size_t episodic_write_slot,
    const std::size_t episodic_read_span,
    const std::span<float> output,
    const std::span<float> regular_component,
    const std::span<float> episodic_component,
    const std::span<float> regular_mass,
    const std::span<float> episodic_mass,
    const std::span<const std::uint64_t> tracked_positions,
    const std::span<float> tracked_mass,
    const std::span<float> episodic_slot_mass,
    const std::span<float> episodic_slot_values,
    const std::span<float> regular_entry_mass,
    const std::span<float> regular_entry_values,
    const std::span<std::uint8_t> regular_entry_valid_kind,
    const std::span<std::uint64_t> regular_entry_positions,
    const std::span<float> qk_partials,
    const std::span<float> candidate_keys,
    const std::span<float> candidate_values,
    const std::span<const std::uint8_t> candidate_mask) {
  validate_inputs(query, key, value, output);
  const bool trace_partitions = !regular_component.empty();
  const bool trace_positions = !tracked_positions.empty();
  const bool trace_episodic_slots = !episodic_slot_mass.empty();
  const bool trace_regular_entries = !regular_entry_mass.empty();
  const bool trace_qk_partials = !qk_partials.empty();
  const std::size_t slot_count =
      config_.query_heads * config_.episodic_span_size;
  const std::size_t slot_value_count =
      slot_count * config_.head_dimension;
  const std::size_t regular_entry_count = checked_product(
      config_.query_heads, kRegularTraceEntries,
      "streaming regular-entry trace shape overflows");
  const std::size_t regular_entry_value_count = checked_product(
      regular_entry_count, config_.head_dimension,
      "streaming regular-entry value trace shape overflows");
  const std::size_t qk_partial_count = checked_product(
      checked_product(config_.query_heads, kC28TraceEntries,
                      "streaming C28 QK trace shape overflows"),
      kQKPartialBands, "streaming C28 QK trace shape overflows");
  const std::size_t qk_candidate_count = checked_product(
      checked_product(config_.query_heads, config_.older_candidates,
                      "streaming QK candidate trace shape overflows"),
      kQKPartialBands, "streaming QK candidate trace shape overflows");
  const std::size_t candidate_key_count = checked_product(
      checked_product(config_.query_heads, config_.older_candidates,
                      "streaming candidate-key trace shape overflows"),
      config_.head_dimension,
      "streaming candidate-key trace shape overflows");
  const std::size_t candidate_value_count = candidate_key_count;
  const bool trace_qk_candidates =
      !qk_partials.empty() && qk_partials.size() ==
                                   qk_partial_count + qk_candidate_count;
  const bool trace_candidate_keys = !candidate_keys.empty();
  const bool trace_candidate_values = !candidate_values.empty();
  const std::size_t candidate_mask_count = checked_product(
      config_.query_heads, config_.older_candidates,
      "streaming candidate mask shape overflows");
  const bool trace_candidate_mask = !candidate_mask.empty();
  if ((!trace_partitions &&
       (!episodic_component.empty() || !regular_mass.empty() ||
        !episodic_mass.empty())) ||
      (trace_partitions &&
       (regular_component.size() != output.size() ||
        episodic_component.size() != output.size() ||
        regular_mass.size() != config_.query_heads ||
        episodic_mass.size() != config_.query_heads)) ||
      (!trace_positions && !tracked_mass.empty()) ||
      (trace_positions &&
       tracked_mass.size() != config_.query_heads) ||
      (!trace_episodic_slots && !episodic_slot_values.empty()) ||
      (trace_episodic_slots &&
       (!trace_partitions ||
        episodic_slot_mass.size() != slot_count ||
        episodic_slot_values.size() != slot_value_count ||
        episodic_read_span == kNoEpisodicDirective)) ||
      (!trace_regular_entries &&
       (!regular_entry_values.empty() ||
        !regular_entry_valid_kind.empty() ||
        !regular_entry_positions.empty())) ||
      (trace_regular_entries &&
       (config_.local_window != kRegularTraceLocalEntries ||
        config_.older_top_k != kRegularTraceOlderEntries ||
        regular_entry_mass.size() != regular_entry_count ||
        regular_entry_values.size() != regular_entry_value_count ||
        regular_entry_valid_kind.size() != regular_entry_count ||
        regular_entry_positions.size() != regular_entry_count)) ||
      (trace_qk_partials &&
       (config_.head_dimension != kQKTraceHeadDimension ||
        config_.local_window != kRegularTraceLocalEntries ||
        config_.older_top_k != kRegularTraceOlderEntries ||
        (config_.episodic_span_size != 0 &&
         config_.episodic_span_size != kC28TraceEpisodicEntries) ||
        (qk_partials.size() != qk_partial_count &&
         qk_partials.size() != qk_partial_count + qk_candidate_count))) ||
      (trace_candidate_keys &&
       (config_.head_dimension != kQKTraceHeadDimension ||
        config_.local_window != kRegularTraceLocalEntries ||
        config_.older_top_k != kRegularTraceOlderEntries ||
        candidate_keys.size() != candidate_key_count)) ||
      (trace_candidate_values &&
       (config_.head_dimension != kQKTraceHeadDimension ||
        config_.local_window != kRegularTraceLocalEntries ||
        config_.older_top_k != kRegularTraceOlderEntries ||
        candidate_values.size() != candidate_value_count)) ||
      (trace_candidate_mask &&
       (candidate_mask.size() != candidate_mask_count ||
        std::none_of(candidate_mask.begin(), candidate_mask.end(),
                     [](const std::uint8_t value) { return value != 0; }) ||
        !std::all_of(candidate_mask.begin(), candidate_mask.end(),
                     [](const std::uint8_t value) { return value <= 1; }))) ||
      (trace_partitions && trace_positions) ||
      (trace_episodic_slots && trace_positions)) {
    throw std::invalid_argument(
        "streaming attention trace shape mismatch");
  }
  std::vector<float> candidate_qk_scratch;
  if (trace_qk_partials) {
    candidate_qk_scratch.resize(checked_product(
        config_.older_candidates, kQKPartialBands,
        "streaming C28 QK candidate scratch overflows"));
    std::fill(qk_partials.begin(), qk_partials.end(), 0.0F);
  }
  if (trace_candidate_keys) {
    std::fill(candidate_keys.begin(), candidate_keys.end(), 0.0F);
  }
  if (trace_candidate_values) {
    std::fill(candidate_values.begin(), candidate_values.end(), 0.0F);
  }
  if (trace_positions) {
    for (std::size_t index = 0; index < tracked_positions.size();
         ++index) {
      const std::uint64_t position = tracked_positions[index];
      if (position >= tokens_seen_ ||
          std::find(tracked_positions.begin(),
                    tracked_positions.begin() +
                        static_cast<std::ptrdiff_t>(index),
                    position) !=
              tracked_positions.begin() +
                  static_cast<std::ptrdiff_t>(index)) {
        throw std::invalid_argument(
            "streaming tracked position is invalid");
      }
      bool retained = false;
      for (std::size_t local = 0; local < recent_size_; ++local) {
        if (recent_size_ == config_.local_window && local == 0) {
          continue;
        }
        const std::size_t slot =
            (recent_start_ + local) % config_.local_window;
        retained = retained || recent_positions_[slot] == position;
      }
      if (!retained) {
        throw std::invalid_argument(
            "streaming tracked position is outside the retained local "
            "window");
      }
    }
  }
  const bool write_episodic =
      episodic_write_slot != kNoEpisodicDirective;
  const bool read_episodic =
      episodic_read_span != kNoEpisodicDirective;
  if (write_episodic &&
      (config_.episodic_slots == 0 ||
       episodic_write_slot >= config_.episodic_slots)) {
    throw std::invalid_argument(
        "streaming episodic write directive is invalid");
  }
  if (read_episodic &&
      (config_.episodic_span_size == 0 ||
       episodic_read_span >=
           config_.episodic_slots / config_.episodic_span_size)) {
    throw std::invalid_argument(
        "streaming episodic read directive is invalid");
  }
  if (write_episodic &&
      episodic_positions_[episodic_write_slot] !=
          kNoEpisodicDirective) {
    throw std::invalid_argument(
        "streaming episodic slot is already active");
  }
  const std::size_t episodic_begin =
      read_episodic
          ? episodic_read_span * config_.episodic_span_size
          : 0;
  if (read_episodic) {
    for (std::size_t offset = 0;
         offset < config_.episodic_span_size; ++offset) {
      const std::uint64_t position =
          episodic_positions_[episodic_begin + offset];
      if (position == kNoEpisodicDirective ||
          position >= tokens_seen_) {
        throw std::invalid_argument(
            "streaming episodic read is not strictly causal");
      }
    }
  }
  if (write_episodic) {
    for (std::size_t kv_head = 0; kv_head < config_.key_value_heads;
         ++kv_head) {
      const std::size_t target =
          episodic_offset(episodic_write_slot, kv_head);
      const std::size_t source = kv_head * config_.head_dimension;
      for (std::size_t dimension = 0;
           dimension < config_.head_dimension; ++dimension) {
        episodic_keys_[target + dimension] =
            float_to_bf16(key[source + dimension]);
        episodic_values_[target + dimension] =
            float_to_bf16(value[source + dimension]);
      }
    }
    episodic_positions_[episodic_write_slot] = tokens_seen_;
    ++episodic_active_slots_;
  }
  std::uint64_t eviction_events{};
  std::uint64_t sink_insertions{};
  std::uint64_t heavy_hitter_updates{};
  std::size_t write_slot{};
  if (recent_size_ == config_.local_window) {
    write_slot = recent_start_;
    evict_recent(write_slot, sink_insertions, heavy_hitter_updates);
    eviction_events = 1;
    recent_start_ = (recent_start_ + 1) % config_.local_window;
  } else {
    write_slot = (recent_start_ + recent_size_) % config_.local_window;
    ++recent_size_;
  }
  for (std::size_t kv_head = 0; kv_head < config_.key_value_heads; ++kv_head) {
    const std::size_t source = kv_head * config_.head_dimension;
    store_recent(
        write_slot, kv_head,
        std::span<const float>(key.data() + source, config_.head_dimension),
        std::span<const float>(value.data() + source, config_.head_dimension));
  }
  recent_positions_[write_slot] = tokens_seen_;
  for (std::size_t head = 0; head < config_.query_heads; ++head) {
    recent_mass_[head * config_.local_window + write_slot] = 0.0F;
  }
  ++tokens_seen_;

  std::size_t active_older{};
  std::uint64_t candidate_bytes{};
  std::uint64_t selected_value_bytes{};
  std::uint64_t local_bytes{};
  std::uint64_t older_candidate_entries_scored{};
  std::uint64_t older_selected_entries{};
  std::uint64_t episodic_entries_read{};
  std::uint64_t episodic_key_read_bytes{};
  std::uint64_t episodic_value_read_bytes{};
  std::uint64_t episodic_duplicate_older_entries_suppressed{};
  for (std::size_t head = 0; head < config_.query_heads; ++head) {
    const bool head_reads_episodic =
        read_episodic &&
        (config_.episodic_head_mask.empty() ||
         config_.episodic_head_mask[head] != 0);
    const float* query_row =
        query.data() + head * config_.head_dimension;
    const std::size_t kv_head = head / groups_;
    const auto qk_entry = [&](const std::size_t entry) {
      return std::span<float>(
          qk_partials.data() +
              (head * kC28TraceEntries + entry) * kQKPartialBands,
          kQKPartialBands);
    };
    const auto qk_candidate_entry = [&](const std::size_t slot) {
      return std::span<float>(
          qk_partials.data() + qk_partial_count +
              (head * config_.older_candidates + slot) * kQKPartialBands,
          kQKPartialBands);
    };
    std::vector<std::size_t> candidates;
    candidates.reserve(config_.older_candidates);
    for (std::size_t slot = 0; slot < config_.older_candidates; ++slot) {
      const std::size_t index = head * config_.older_candidates + slot;
      if (older_active_[index] == 0) continue;
      if (trace_candidate_mask && candidate_mask[index] == 0) continue;
      candidates.push_back(slot);
      const float raw_score =
          trace_qk_partials
              ? dot_rope_bands(
                    query_row,
                    older_keys_.data() + older_offset(head, slot),
                    config_.head_dimension, config_.scale,
                    std::span<float>(
                        candidate_qk_scratch.data() +
                            slot * kQKPartialBands,
                        kQKPartialBands))
              : dot(query_row,
                    older_keys_.data() + older_offset(head, slot),
                    config_.head_dimension);
      candidate_score_scratch_[slot] = raw_score * config_.scale;
      if (trace_qk_candidates) {
        std::copy_n(
            candidate_qk_scratch.data() + slot * kQKPartialBands,
            kQKPartialBands, qk_candidate_entry(slot).data());
      }
      if (trace_candidate_keys) {
        std::copy_n(
            older_keys_.data() + older_offset(head, slot),
            config_.head_dimension,
            candidate_keys.data() +
                (head * config_.older_candidates + slot) *
                    config_.head_dimension);
      }
      if (trace_candidate_values) {
        std::copy_n(
            older_values_.data() + older_offset(head, slot),
            config_.head_dimension,
            candidate_values.data() +
                (head * config_.older_candidates + slot) *
                    config_.head_dimension);
      }
    }
    active_older += candidates.size();
    older_candidate_entries_scored += candidates.size();
    candidate_bytes += candidates.size() * config_.head_dimension * sizeof(float);
    if (head_reads_episodic) {
      const auto duplicate = [&](const std::size_t candidate) {
        const std::uint64_t position =
            older_positions_[head * config_.older_candidates + candidate];
        for (std::size_t offset = 0;
             offset < config_.episodic_span_size; ++offset) {
          if (position == episodic_positions_[episodic_begin + offset]) {
            return true;
          }
        }
        return false;
      };
      const std::size_t before = candidates.size();
      candidates.erase(
          std::remove_if(candidates.begin(), candidates.end(), duplicate),
          candidates.end());
      episodic_duplicate_older_entries_suppressed +=
          before - candidates.size();
    }
    const std::size_t selected_count =
        std::min(config_.older_top_k, candidates.size());
    older_selected_entries += selected_count;
    std::partial_sort(
        candidates.begin(), candidates.begin() + selected_count,
        candidates.end(), [&](const std::size_t left, const std::size_t right) {
          if (candidate_score_scratch_[left] !=
              candidate_score_scratch_[right]) {
            return candidate_score_scratch_[left] >
                   candidate_score_scratch_[right];
          }
          return older_positions_[head * config_.older_candidates + left] <
                 older_positions_[head * config_.older_candidates + right];
        });
    for (std::size_t index = 0; index < selected_count; ++index) {
      selected_scratch_[index] = candidates[index];
      if (trace_qk_partials) {
        const std::size_t slot = candidates[index];
        std::copy_n(
            candidate_qk_scratch.data() + slot * kQKPartialBands,
            kQKPartialBands,
            qk_entry(kRegularTraceLocalEntries + index).data());
      }
    }

    std::size_t visible = 0;
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t local = 0; local < recent_size_; ++local) {
      const std::size_t slot = (recent_start_ + local) % config_.local_window;
      const float raw_score = recent_key_dot(
          query_row, slot, kv_head,
          trace_qk_partials ? qk_entry(local) : std::span<float>{});
      const float score = raw_score * config_.scale;
      score_scratch_[visible++] = score;
      maximum = std::max(maximum, score);
    }
    for (std::size_t index = 0; index < selected_count; ++index) {
      const float score = candidate_score_scratch_[selected_scratch_[index]];
      score_scratch_[visible++] = score;
      maximum = std::max(maximum, score);
    }
    if (head_reads_episodic) {
      for (std::size_t offset = 0;
           offset < config_.episodic_span_size; ++offset) {
        const std::size_t slot = episodic_begin + offset;
        const float raw_score =
            trace_qk_partials
                ? dot_bf16_rope_bands(
                      query_row,
                      episodic_keys_.data() +
                          episodic_offset(slot, kv_head),
                      config_.head_dimension, config_.scale,
                      qk_entry(kRegularTraceEntries + offset))
                : dot_bf16(
                      query_row,
                      episodic_keys_.data() +
                          episodic_offset(slot, kv_head),
                      config_.head_dimension);
        float score = raw_score * config_.scale;
        // Preserve the exact legacy arithmetic route for zero bias.
        if (config_.episodic_logit_bias != 0.0F) {
          score += config_.episodic_logit_bias;
        }
        score_scratch_[visible++] = score;
        maximum = std::max(maximum, score);
      }
    }
    float denominator = 0.0F;
    for (std::size_t index = 0; index < visible; ++index) {
      weight_scratch_[index] = std::exp(score_scratch_[index] - maximum);
      denominator += weight_scratch_[index];
    }
    std::fill_n(output.data() + head * config_.head_dimension,
                config_.head_dimension, 0.0F);
    std::size_t weight_index = 0;
    for (std::size_t local = 0; local < recent_size_; ++local) {
      const std::size_t slot = (recent_start_ + local) % config_.local_window;
      const float weight = weight_scratch_[weight_index++] / denominator;
      recent_mass_[head * config_.local_window + slot] += weight;
      float* target = output.data() + head * config_.head_dimension;
      for (std::size_t dimension = 0; dimension < config_.head_dimension;
           ++dimension) {
        target[dimension] +=
            weight * recent_value_at(slot, kv_head, dimension);
      }
    }
    for (std::size_t index = 0; index < selected_count; ++index) {
      const std::size_t slot = selected_scratch_[index];
      const float weight = weight_scratch_[weight_index++] / denominator;
      older_scores_[head * config_.older_candidates + slot] += weight;
      const float* source = older_values_.data() + older_offset(head, slot);
      float* target = output.data() + head * config_.head_dimension;
      for (std::size_t dimension = 0; dimension < config_.head_dimension;
           ++dimension) {
        target[dimension] += weight * source[dimension];
      }
    }
    if (head_reads_episodic) {
      float* target = output.data() + head * config_.head_dimension;
      for (std::size_t offset = 0;
           offset < config_.episodic_span_size; ++offset) {
        const std::size_t slot = episodic_begin + offset;
        const float weight = weight_scratch_[weight_index++] / denominator;
        const std::uint16_t* source =
            episodic_values_.data() + episodic_offset(slot, kv_head);
        for (std::size_t dimension = 0;
             dimension < config_.head_dimension; ++dimension) {
          target[dimension] +=
              weight * bf16_to_float(source[dimension]);
        }
      }
      episodic_entries_read += config_.episodic_span_size;
      episodic_key_read_bytes += config_.episodic_span_size *
                                 config_.head_dimension *
                                 sizeof(std::uint16_t);
      episodic_value_read_bytes += config_.episodic_span_size *
                                   config_.head_dimension *
                                   sizeof(std::uint16_t);
    }
    if (trace_partitions) {
      float* regular_target =
          regular_component.data() + head * config_.head_dimension;
      float* episodic_target =
          episodic_component.data() + head * config_.head_dimension;
      std::fill_n(regular_target, config_.head_dimension, 0.0F);
      std::fill_n(episodic_target, config_.head_dimension, 0.0F);
      float* slot_mass_target = nullptr;
      float* slot_value_target = nullptr;
      if (trace_episodic_slots) {
        slot_mass_target =
            episodic_slot_mass.data() +
            head * config_.episodic_span_size;
        slot_value_target =
            episodic_slot_values.data() +
            head * config_.episodic_span_size *
                config_.head_dimension;
        std::fill_n(
            slot_mass_target, config_.episodic_span_size, 0.0F);
        std::fill_n(
            slot_value_target,
            config_.episodic_span_size * config_.head_dimension,
            0.0F);
      }
      if (!head_reads_episodic) {
        std::copy_n(
            output.data() + head * config_.head_dimension,
            config_.head_dimension, regular_target);
        regular_mass[head] = 1.0F;
        episodic_mass[head] = 0.0F;
      } else {
        float current_regular_mass = 0.0F;
        float current_episodic_mass = 0.0F;
        std::size_t trace_weight_index = 0;
        for (std::size_t local = 0; local < recent_size_; ++local) {
          const std::size_t slot =
              (recent_start_ + local) % config_.local_window;
          const float weight =
              weight_scratch_[trace_weight_index++] / denominator;
          current_regular_mass += weight;
          for (std::size_t dimension = 0;
               dimension < config_.head_dimension; ++dimension) {
            regular_target[dimension] +=
                weight * recent_value_at(slot, kv_head, dimension);
          }
        }
        for (std::size_t index = 0; index < selected_count; ++index) {
          const std::size_t slot = selected_scratch_[index];
          const float weight =
              weight_scratch_[trace_weight_index++] / denominator;
          current_regular_mass += weight;
          const float* source =
              older_values_.data() + older_offset(head, slot);
          for (std::size_t dimension = 0;
               dimension < config_.head_dimension; ++dimension) {
            regular_target[dimension] += weight * source[dimension];
          }
        }
        for (std::size_t offset = 0;
             offset < config_.episodic_span_size; ++offset) {
          const std::size_t slot = episodic_begin + offset;
          const float weight =
              weight_scratch_[trace_weight_index++] / denominator;
          current_episodic_mass += weight;
          const std::uint16_t* source =
              episodic_values_.data() + episodic_offset(slot, kv_head);
          if (trace_episodic_slots) {
            slot_mass_target[offset] = weight;
          }
          for (std::size_t dimension = 0;
               dimension < config_.head_dimension; ++dimension) {
            const float decoded = bf16_to_float(source[dimension]);
            episodic_target[dimension] += weight * decoded;
            if (trace_episodic_slots) {
              slot_value_target[
                  offset * config_.head_dimension + dimension] =
                  decoded;
            }
          }
        }
        regular_mass[head] = current_regular_mass;
        episodic_mass[head] = current_episodic_mass;
      }
    }
    if (trace_positions) {
      float current_tracked_mass = 0.0F;
      std::size_t trace_weight_index = 0;
      for (std::size_t local = 0; local < recent_size_; ++local) {
        const std::size_t slot =
            (recent_start_ + local) % config_.local_window;
        const std::uint64_t position = recent_positions_[slot];
        const float weight =
            weight_scratch_[trace_weight_index++] / denominator;
        if (std::find(tracked_positions.begin(),
                      tracked_positions.end(), position) !=
            tracked_positions.end()) {
          current_tracked_mass += weight;
        }
      }
      tracked_mass[head] = current_tracked_mass;
    }
    if (trace_regular_entries) {
      const std::size_t entry_offset = head * kRegularTraceEntries;
      const std::size_t value_offset =
          entry_offset * config_.head_dimension;
      std::fill_n(regular_entry_mass.data() + entry_offset,
                  kRegularTraceEntries, 0.0F);
      std::fill_n(regular_entry_values.data() + value_offset,
                  kRegularTraceEntries * config_.head_dimension, 0.0F);
      std::fill_n(regular_entry_valid_kind.data() + entry_offset,
                  kRegularTraceEntries, kRegularTraceInvalid);
      std::fill_n(regular_entry_positions.data() + entry_offset,
                  kRegularTraceEntries, kNoEpisodicDirective);
      std::size_t trace_weight_index = 0;
      for (std::size_t local = 0; local < recent_size_; ++local) {
        const std::size_t slot =
            (recent_start_ + local) % config_.local_window;
        const std::size_t entry = entry_offset + local;
        const float weight =
            weight_scratch_[trace_weight_index++] / denominator;
        regular_entry_mass[entry] = weight;
        regular_entry_valid_kind[entry] = kRegularTraceLocal;
        regular_entry_positions[entry] = recent_positions_[slot];
        for (std::size_t dimension = 0;
             dimension < config_.head_dimension; ++dimension) {
          regular_entry_values[entry * config_.head_dimension + dimension] =
              recent_value_at(slot, kv_head, dimension);
        }
      }
      for (std::size_t index = 0; index < selected_count; ++index) {
        const std::size_t slot = selected_scratch_[index];
        const std::size_t entry =
            entry_offset + kRegularTraceLocalEntries + index;
        const float weight =
            weight_scratch_[trace_weight_index++] / denominator;
        regular_entry_mass[entry] = weight;
        regular_entry_valid_kind[entry] = kRegularTraceOlder;
        regular_entry_positions[entry] =
            older_positions_[head * config_.older_candidates + slot];
        std::copy_n(
            older_values_.data() + older_offset(head, slot),
            config_.head_dimension,
            regular_entry_values.data() +
                entry * config_.head_dimension);
      }
    }
    const std::size_t key_bytes =
        (config_.local_bf16 || config_.local_fp16)
            ? sizeof(std::uint16_t)
            : (config_.local_int8 ? sizeof(std::int8_t) : sizeof(float));
    const std::size_t value_bytes =
      (config_.local_bf16 || config_.local_values_bf16 ||
       config_.local_values_fp16 || config_.local_fp16 || config_.local_int8)
            ? (config_.local_int8 ? sizeof(std::int8_t)
                                   : sizeof(std::uint16_t))
            : sizeof(float);
    local_bytes += recent_size_ * config_.head_dimension *
                   (key_bytes + value_bytes);
    selected_value_bytes +=
        selected_count * config_.head_dimension * sizeof(float);
  }

  return StreamingAttentionMetrics{
      .tokens_seen = tokens_seen_,
      .local_entries = recent_size_,
      .active_older_entries = active_older,
      .candidate_key_bytes = candidate_bytes,
      .selected_value_bytes = selected_value_bytes,
      .local_kv_bytes = local_bytes,
      .eviction_events = eviction_events,
      .older_candidate_entries_scored = older_candidate_entries_scored,
      .older_selected_entries = older_selected_entries,
      .sink_insertions = sink_insertions,
      .heavy_hitter_updates = heavy_hitter_updates,
      .state_bytes = allocated_state_bytes(),
      .scratch_bytes = scratch_bytes(),
      .episodic_slots_written = write_episodic ? 1U : 0U,
      .episodic_active_slots = episodic_active_slots_,
      .episodic_read_events = read_episodic ? 1U : 0U,
      .episodic_entries_read = episodic_entries_read,
      .episodic_write_bytes =
          write_episodic
              ? 2U * config_.key_value_heads *
                    config_.head_dimension * sizeof(std::uint16_t)
              : 0U,
      .episodic_key_read_bytes = episodic_key_read_bytes,
      .episodic_value_read_bytes = episodic_value_read_bytes,
      .episodic_duplicate_older_entries_suppressed =
          episodic_duplicate_older_entries_suppressed,
  };
}

std::size_t StreamingAttention::active_older_entries() const noexcept {
  return static_cast<std::size_t>(
      std::count(older_active_.begin(), older_active_.end(), std::uint8_t{1}));
}

std::size_t StreamingAttention::active_episodic_slots() const noexcept {
  return episodic_active_slots_;
}

std::size_t StreamingAttention::allocated_state_bytes() const noexcept {
  const std::size_t recent_key_bytes =
      (config_.local_bf16 || config_.local_fp16)
          ? (config_.local_bf16 ? recent_keys_bf16_.size()
                                : recent_keys_fp16_.size()) *
                sizeof(std::uint16_t)
          : (config_.local_int8
                 ? recent_keys_int8_.size() * sizeof(std::int8_t)
                 : recent_keys_.size() * sizeof(float));
  const std::size_t recent_value_bytes =
      (config_.local_bf16 || config_.local_values_bf16)
          ? recent_values_bf16_.size() * sizeof(std::uint16_t)
          : ((config_.local_values_fp16 || config_.local_fp16)
                 ? recent_values_fp16_.size() * sizeof(std::uint16_t)
                 : (config_.local_int8
                        ? recent_values_int8_.size() * sizeof(std::int8_t)
                        : recent_values_.size() * sizeof(float)));
  const std::size_t int8_scale_bytes =
      config_.local_int8
          ? (recent_key_scales_int8_.size() +
             recent_value_scales_int8_.size()) * sizeof(float)
          : 0;
  return recent_key_bytes + recent_value_bytes +
         int8_scale_bytes +
         recent_mass_.size() * sizeof(float) +
         recent_positions_.size() * sizeof(std::uint64_t) +
         older_keys_.size() * sizeof(float) +
         older_values_.size() * sizeof(float) +
         older_scores_.size() * sizeof(float) +
         older_positions_.size() * sizeof(std::uint64_t) +
         older_active_.size() * sizeof(std::uint8_t) +
         episodic_keys_.size() * sizeof(std::uint16_t) +
         episodic_values_.size() * sizeof(std::uint16_t) +
         episodic_positions_.size() * sizeof(std::uint64_t);
}

std::size_t StreamingAttention::scratch_bytes() const noexcept {
  return score_scratch_.size() * sizeof(float) +
         candidate_score_scratch_.size() * sizeof(float) +
         weight_scratch_.size() * sizeof(float) +
         selected_scratch_.size() * sizeof(std::size_t);
}

}  // namespace engram
