#include "engram/streaming_attention.h"

#include <algorithm>
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

float dot(const float* left, const float* right,
          const std::size_t width) noexcept {
  float result = 0.0F;
  for (std::size_t index = 0; index < width; ++index) {
    result += left[index] * right[index];
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
  recent_keys_.resize(recent_elements);
  recent_values_.resize(recent_elements);
  recent_mass_.resize(config_.query_heads * config_.local_window);
  recent_positions_.resize(config_.local_window);
  older_keys_.resize(older_elements);
  older_values_.resize(older_elements);
  older_scores_.resize(older_vectors);
  older_positions_.resize(older_vectors);
  older_active_.resize(older_vectors);
  score_scratch_.resize(config_.local_window + config_.older_candidates);
  candidate_score_scratch_.resize(config_.older_candidates);
  weight_scratch_.resize(config_.local_window + config_.older_top_k);
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

void StreamingAttention::evict_recent(const std::size_t slot) {
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
  validate_inputs(query, key, value, output);
  std::size_t write_slot{};
  if (recent_size_ == config_.local_window) {
    write_slot = recent_start_;
    evict_recent(write_slot);
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
  for (std::size_t head = 0; head < config_.query_heads; ++head) {
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
    candidate_bytes += candidates.size() * config_.head_dimension * sizeof(float);
    const std::size_t selected_count =
        std::min(config_.older_top_k, candidates.size());
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
      .state_bytes = allocated_state_bytes(),
      .scratch_bytes = scratch_bytes(),
  };
}

std::size_t StreamingAttention::active_older_entries() const noexcept {
  return static_cast<std::size_t>(
      std::count(older_active_.begin(), older_active_.end(), std::uint8_t{1}));
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
         older_active_.size() * sizeof(std::uint8_t);
}

std::size_t StreamingAttention::scratch_bytes() const noexcept {
  return score_scratch_.size() * sizeof(float) +
         candidate_score_scratch_.size() * sizeof(float) +
         weight_scratch_.size() * sizeof(float) +
         selected_scratch_.size() * sizeof(std::size_t);
}

}  // namespace engram
