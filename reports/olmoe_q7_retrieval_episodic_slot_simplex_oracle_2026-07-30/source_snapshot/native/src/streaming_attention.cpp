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

float dot_bf16(const float* left, const std::uint16_t* right,
               const std::size_t width) noexcept {
  float result = 0.0F;
  for (std::size_t index = 0; index < width; ++index) {
    result += left[index] * bf16_to_float(right[index]);
  }
  return result;
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
  recent_keys_.resize(recent_elements);
  recent_values_.resize(recent_elements);
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
    const std::size_t source = recent_offset(slot, kv_head);
    const std::size_t target = older_offset(head, destination);
    std::copy_n(recent_keys_.data() + source, config_.head_dimension,
                older_keys_.data() + target);
    std::copy_n(recent_values_.data() + source, config_.head_dimension,
                older_values_.data() + target);
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
      {}, {}, {}, {}, {}, {}, {}, {});
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
      {}, {}, {}, {});
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
      {}, {}, slot_mass, slot_values);
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
      output, {}, {}, {}, {}, tracked_positions, tracked_mass, {}, {});
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
    const std::span<float> episodic_slot_values) {
  validate_inputs(query, key, value, output);
  const bool trace_partitions = !regular_component.empty();
  const bool trace_positions = !tracked_positions.empty();
  const bool trace_episodic_slots = !episodic_slot_mass.empty();
  const std::size_t slot_count =
      config_.query_heads * config_.episodic_span_size;
  const std::size_t slot_value_count =
      slot_count * config_.head_dimension;
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
      (trace_partitions && trace_positions) ||
      (trace_episodic_slots && trace_positions)) {
    throw std::invalid_argument(
        "streaming attention trace shape mismatch");
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
    const std::size_t target = recent_offset(write_slot, kv_head);
    const std::size_t source = kv_head * config_.head_dimension;
    std::copy_n(key.data() + source, config_.head_dimension,
                recent_keys_.data() + target);
    std::copy_n(value.data() + source, config_.head_dimension,
                recent_values_.data() + target);
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
    std::vector<std::size_t> candidates;
    candidates.reserve(config_.older_candidates);
    for (std::size_t slot = 0; slot < config_.older_candidates; ++slot) {
      const std::size_t index = head * config_.older_candidates + slot;
      if (older_active_[index] == 0) continue;
      candidates.push_back(slot);
      candidate_score_scratch_[slot] =
          dot(query_row, older_keys_.data() + older_offset(head, slot),
              config_.head_dimension) *
          config_.scale;
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
    }

    std::size_t visible = 0;
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t local = 0; local < recent_size_; ++local) {
      const std::size_t slot = (recent_start_ + local) % config_.local_window;
      const float score =
          dot(query_row, recent_keys_.data() + recent_offset(slot, kv_head),
              config_.head_dimension) *
          config_.scale;
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
        float score =
            dot_bf16(
                query_row,
                episodic_keys_.data() + episodic_offset(slot, kv_head),
                config_.head_dimension) *
            config_.scale;
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
      const float* source =
          recent_values_.data() + recent_offset(slot, kv_head);
      float* target = output.data() + head * config_.head_dimension;
      for (std::size_t dimension = 0; dimension < config_.head_dimension;
           ++dimension) {
        target[dimension] += weight * source[dimension];
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
          const float* source =
              recent_values_.data() + recent_offset(slot, kv_head);
          for (std::size_t dimension = 0;
               dimension < config_.head_dimension; ++dimension) {
            regular_target[dimension] += weight * source[dimension];
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
    local_bytes += recent_size_ * config_.head_dimension * sizeof(float) * 2;
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
  return recent_keys_.size() * sizeof(float) +
         recent_values_.size() * sizeof(float) +
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
