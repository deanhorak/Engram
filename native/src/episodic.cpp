#include "engram/episodic.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace engram {
namespace {

double positive_feature(const float value) {
  if (value >= 0.0F) {
    return static_cast<double>(value) + 1.0;
  }
  return std::max(std::exp(static_cast<double>(value)),
                  std::numeric_limits<double>::min());
}

float quantization_scale(const float* vector, const std::size_t width) {
  float maximum = 0.0F;
  for (std::size_t index = 0; index < width; ++index) {
    maximum = std::max(maximum, std::abs(vector[index]));
  }
  return maximum > 0.0F ? maximum / 127.0F : 1.0F;
}

void quantize(const float* vector, const std::size_t width, const float scale,
              std::int8_t* codes) {
  for (std::size_t index = 0; index < width; ++index) {
    const double rounded =
        std::nearbyint(static_cast<double>(vector[index] / scale));
    const double clipped = std::clamp(rounded, -127.0, 127.0);
    codes[index] = static_cast<std::int8_t>(clipped);
  }
}

}  // namespace

HybridEpisodicMemory::HybridEpisodicMemory(EpisodicConfig config)
    : config_(config) {
  if (config_.key_width == 0 || config_.value_width == 0 ||
      config_.local_window == 0) {
    throw std::invalid_argument(
        "key width, value width, and local window must be positive");
  }
  if (config_.retrieval_candidates == 0 || config_.retrieval_top_k == 0 ||
      config_.retrieval_top_k > config_.retrieval_candidates) {
    throw std::invalid_argument(
        "retrieval counts must be positive and top-k must not exceed candidates");
  }
  if (!std::isfinite(config_.decay) || config_.decay < 0.0 ||
      config_.decay > 1.0) {
    throw std::invalid_argument("decay must lie in [0, 1]");
  }
  if (!std::isfinite(config_.older_weight) || config_.older_weight < 0.0 ||
      config_.older_weight > 1.0) {
    throw std::invalid_argument("older weight must lie in [0, 1]");
  }
  if (!std::isfinite(config_.epsilon) || config_.epsilon <= 0.0) {
    throw std::invalid_argument("epsilon must be finite and positive");
  }

  recent_keys_.resize(config_.local_window * config_.key_width);
  recent_values_.resize(config_.local_window * config_.value_width);
  recent_positions_.resize(config_.local_window);
  older_key_codes_.resize(config_.retrieval_capacity * config_.key_width);
  older_key_scales_.resize(config_.retrieval_capacity, 1.0F);
  older_value_codes_.resize(config_.retrieval_capacity * config_.value_width);
  older_value_scales_.resize(config_.retrieval_capacity, 1.0F);
  older_positions_.resize(config_.retrieval_capacity);
  recurrent_numerator_.resize(config_.key_width * config_.value_width);
  recurrent_normalizer_.resize(config_.key_width);

  query_features_.resize(config_.key_width);
  key_features_.resize(config_.key_width);
  local_scores_.resize(config_.local_window);
  candidate_slots_.resize(config_.retrieval_candidates);
  candidate_scores_.resize(config_.retrieval_candidates);
  selected_slots_.resize(config_.retrieval_top_k);
  selected_scores_.resize(config_.retrieval_top_k);
  selected_positions_.resize(config_.retrieval_top_k);
  local_output_.resize(config_.value_width);
  recurrent_output_.resize(config_.value_width);
  retrieval_output_.resize(config_.value_width);
  reset();
}

void HybridEpisodicMemory::reset() noexcept {
  tokens_seen_ = 0;
  recent_start_ = 0;
  recent_size_ = 0;
  older_start_ = 0;
  older_size_ = 0;
  recurrent_steps_ = 0;
  last_retrieval_count_ = 0;
  std::fill(recurrent_numerator_.begin(), recurrent_numerator_.end(), 0.0);
  std::fill(recurrent_normalizer_.begin(), recurrent_normalizer_.end(), 0.0);
  std::fill(local_output_.begin(), local_output_.end(), 0.0F);
  std::fill(recurrent_output_.begin(), recurrent_output_.end(), 0.0F);
  std::fill(retrieval_output_.begin(), retrieval_output_.end(), 0.0F);
}

const EpisodicConfig& HybridEpisodicMemory::config() const noexcept {
  return config_;
}

std::size_t HybridEpisodicMemory::local_count() const noexcept {
  return recent_size_;
}

std::size_t HybridEpisodicMemory::older_count() const noexcept {
  return older_size_;
}

std::size_t HybridEpisodicMemory::tokens_seen() const noexcept {
  return tokens_seen_;
}

std::size_t HybridEpisodicMemory::allocated_state_bytes() const noexcept {
  return recent_keys_.size() * sizeof(float) +
         recent_values_.size() * sizeof(float) +
         recent_positions_.size() * sizeof(std::uint64_t) +
         older_key_codes_.size() * sizeof(std::int8_t) +
         older_key_scales_.size() * sizeof(float) +
         older_value_codes_.size() * sizeof(std::int8_t) +
         older_value_scales_.size() * sizeof(float) +
         older_positions_.size() * sizeof(std::uint64_t) +
         recurrent_numerator_.size() * sizeof(double) +
         recurrent_normalizer_.size() * sizeof(double);
}

std::size_t HybridEpisodicMemory::scratch_bytes() const noexcept {
  return (query_features_.size() + key_features_.size() +
          local_scores_.size() + candidate_scores_.size() +
          selected_scores_.size()) *
             sizeof(double) +
         (candidate_slots_.size() + selected_slots_.size()) *
             sizeof(std::size_t) +
         selected_positions_.size() * sizeof(std::uint64_t) +
         (local_output_.size() + recurrent_output_.size() +
          retrieval_output_.size()) *
             sizeof(float);
}

std::span<const float> HybridEpisodicMemory::last_local_output() const noexcept {
  return local_output_;
}

std::span<const float> HybridEpisodicMemory::last_recurrent_output() const
    noexcept {
  return recurrent_output_;
}

std::span<const float> HybridEpisodicMemory::last_retrieval_output() const
    noexcept {
  return retrieval_output_;
}

std::span<const std::uint64_t>
HybridEpisodicMemory::last_retrieved_positions() const noexcept {
  return {selected_positions_.data(), last_retrieval_count_};
}

void HybridEpisodicMemory::validate_vector(const float* vector,
                                           const std::size_t width,
                                           const char* name) const {
  if (vector == nullptr) {
    throw std::invalid_argument(std::string(name) + " must not be null");
  }
  for (std::size_t index = 0; index < width; ++index) {
    if (!std::isfinite(vector[index])) {
      throw std::invalid_argument(std::string(name) + " must be finite");
    }
  }
}

void HybridEpisodicMemory::update_recurrent(const float* query,
                                            const float* key,
                                            const float* value) {
  for (std::size_t row = 0; row < config_.key_width; ++row) {
    query_features_[row] = positive_feature(query[row]);
    key_features_[row] = positive_feature(key[row]);
    recurrent_normalizer_[row] =
        config_.decay * recurrent_normalizer_[row] + key_features_[row];
    for (std::size_t column = 0; column < config_.value_width; ++column) {
      const std::size_t offset = row * config_.value_width + column;
      recurrent_numerator_[offset] =
          config_.decay * recurrent_numerator_[offset] +
          key_features_[row] * static_cast<double>(value[column]);
    }
  }
  ++recurrent_steps_;
  read_recurrent(query);
}

void HybridEpisodicMemory::read_recurrent(const float* query) {
  std::fill(recurrent_output_.begin(), recurrent_output_.end(), 0.0F);
  if (recurrent_steps_ == 0) {
    return;
  }
  double denominator = 0.0;
  for (std::size_t row = 0; row < config_.key_width; ++row) {
    query_features_[row] = positive_feature(query[row]);
    denominator += query_features_[row] * recurrent_normalizer_[row];
  }
  denominator = std::max(denominator, config_.epsilon);
  for (std::size_t column = 0; column < config_.value_width; ++column) {
    double total = 0.0;
    for (std::size_t row = 0; row < config_.key_width; ++row) {
      total += query_features_[row] *
               recurrent_numerator_[row * config_.value_width + column];
    }
    recurrent_output_[column] = static_cast<float>(total / denominator);
  }
}

void HybridEpisodicMemory::append_older(const std::uint64_t position,
                                        const float* key,
                                        const float* value) {
  if (config_.retrieval_capacity == 0) {
    return;
  }
  std::size_t slot{};
  if (older_size_ < config_.retrieval_capacity) {
    slot = (older_start_ + older_size_) % config_.retrieval_capacity;
    ++older_size_;
  } else {
    slot = older_start_;
    older_start_ = (older_start_ + 1) % config_.retrieval_capacity;
  }
  const float key_scale = quantization_scale(key, config_.key_width);
  const float value_scale = quantization_scale(value, config_.value_width);
  quantize(key, config_.key_width, key_scale,
           older_key_codes_.data() + slot * config_.key_width);
  quantize(value, config_.value_width, value_scale,
           older_value_codes_.data() + slot * config_.value_width);
  older_key_scales_[slot] = key_scale;
  older_value_scales_[slot] = value_scale;
  older_positions_[slot] = position;
}

void HybridEpisodicMemory::compute_local(const float* query) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(config_.key_width));
  double maximum = -std::numeric_limits<double>::infinity();
  for (std::size_t logical = 0; logical < recent_size_; ++logical) {
    const std::size_t slot =
        (recent_start_ + logical) % config_.local_window;
    const float* key = recent_keys_.data() + slot * config_.key_width;
    double score = 0.0;
    for (std::size_t column = 0; column < config_.key_width; ++column) {
      score += static_cast<double>(key[column]) * query[column];
    }
    local_scores_[logical] = score * scale;
    maximum = std::max(maximum, local_scores_[logical]);
  }
  double denominator = 0.0;
  for (std::size_t logical = 0; logical < recent_size_; ++logical) {
    local_scores_[logical] = std::exp(local_scores_[logical] - maximum);
    denominator += local_scores_[logical];
  }
  std::fill(local_output_.begin(), local_output_.end(), 0.0F);
  for (std::size_t logical = 0; logical < recent_size_; ++logical) {
    const std::size_t slot =
        (recent_start_ + logical) % config_.local_window;
    const float* value =
        recent_values_.data() + slot * config_.value_width;
    const double weight = local_scores_[logical] / denominator;
    for (std::size_t column = 0; column < config_.value_width; ++column) {
      local_output_[column] += static_cast<float>(weight * value[column]);
    }
  }
}

std::size_t HybridEpisodicMemory::retrieve(const float* query,
                                           std::size_t& bytes_read) {
  std::fill(retrieval_output_.begin(), retrieval_output_.end(), 0.0F);
  last_retrieval_count_ = 0;
  bytes_read = 0;
  if (older_size_ == 0) {
    return 0;
  }
  double query_squared_norm = 0.0;
  for (std::size_t column = 0; column < config_.key_width; ++column) {
    query_squared_norm += static_cast<double>(query[column]) * query[column];
  }
  const double query_norm = std::sqrt(query_squared_norm);
  const std::size_t candidate_limit =
      std::min(config_.retrieval_candidates, older_size_);
  std::size_t candidate_count = 0;
  for (std::size_t logical = 0; logical < older_size_; ++logical) {
    const std::size_t slot =
        (older_start_ + logical) % config_.retrieval_capacity;
    const std::int8_t* codes =
        older_key_codes_.data() + slot * config_.key_width;
    double raw = 0.0;
    double code_squared_norm = 0.0;
    for (std::size_t column = 0; column < config_.key_width; ++column) {
      const double code = static_cast<double>(codes[column]);
      raw += code * query[column];
      code_squared_norm += code * code;
    }
    const double denominator =
        std::sqrt(code_squared_norm) * older_key_scales_[slot] * query_norm;
    const double cosine = denominator > 0.0
                              ? raw * older_key_scales_[slot] / denominator
                              : 0.0;
    std::size_t insertion = 0;
    while (insertion < candidate_count &&
           candidate_scores_[insertion] >= cosine) {
      ++insertion;
    }
    if (insertion >= candidate_limit) {
      continue;
    }
    const std::size_t next_count =
        std::min(candidate_count + 1, candidate_limit);
    for (std::size_t index = next_count - 1; index > insertion; --index) {
      candidate_scores_[index] = candidate_scores_[index - 1];
      candidate_slots_[index] = candidate_slots_[index - 1];
    }
    candidate_scores_[insertion] = cosine;
    candidate_slots_[insertion] = slot;
    candidate_count = next_count;
  }

  const std::size_t selected_limit =
      std::min(config_.retrieval_top_k, candidate_count);
  std::size_t selected_count = 0;
  for (std::size_t candidate = 0; candidate < candidate_count; ++candidate) {
    const std::size_t slot = candidate_slots_[candidate];
    const std::int8_t* codes =
        older_key_codes_.data() + slot * config_.key_width;
    double exact_score = 0.0;
    for (std::size_t column = 0; column < config_.key_width; ++column) {
      exact_score += static_cast<double>(codes[column]) *
                     older_key_scales_[slot] * query[column];
    }
    std::size_t insertion = 0;
    while (insertion < selected_count &&
           selected_scores_[insertion] >= exact_score) {
      ++insertion;
    }
    if (insertion >= selected_limit) {
      continue;
    }
    const std::size_t next_count =
        std::min(selected_count + 1, selected_limit);
    for (std::size_t index = next_count - 1; index > insertion; --index) {
      selected_scores_[index] = selected_scores_[index - 1];
      selected_slots_[index] = selected_slots_[index - 1];
    }
    selected_scores_[insertion] = exact_score;
    selected_slots_[insertion] = slot;
    selected_count = next_count;
  }

  for (std::size_t selected = 0; selected < selected_count; ++selected) {
    const std::size_t slot = selected_slots_[selected];
    selected_positions_[selected] = older_positions_[slot];
    const std::int8_t* value_codes =
        older_value_codes_.data() + slot * config_.value_width;
    for (std::size_t column = 0; column < config_.value_width; ++column) {
      retrieval_output_[column] +=
          static_cast<float>(value_codes[column]) * older_value_scales_[slot];
    }
  }
  if (selected_count != 0) {
    const float inverse = 1.0F / static_cast<float>(selected_count);
    for (float& value : retrieval_output_) {
      value *= inverse;
    }
  }
  last_retrieval_count_ = selected_count;
  const std::size_t key_record =
      config_.key_width * sizeof(std::int8_t) + sizeof(float);
  const std::size_t value_record =
      config_.value_width * sizeof(std::int8_t) + sizeof(float);
  bytes_read = older_size_ * key_record + candidate_count * key_record +
               selected_count * value_record;
  return candidate_count;
}

EpisodicStepMetrics HybridEpisodicMemory::step(const float* query,
                                               const float* key,
                                               const float* value,
                                               float* output) {
  validate_vector(query, config_.key_width, "query");
  validate_vector(key, config_.key_width, "key");
  validate_vector(value, config_.value_width, "value");
  if (output == nullptr) {
    throw std::invalid_argument("output must not be null");
  }

  if (recent_size_ == config_.local_window) {
    const std::size_t evicted_slot = recent_start_;
    const float* evicted_key =
        recent_keys_.data() + evicted_slot * config_.key_width;
    const float* evicted_value =
        recent_values_.data() + evicted_slot * config_.value_width;
    update_recurrent(query, evicted_key, evicted_value);
    append_older(recent_positions_[evicted_slot], evicted_key, evicted_value);
    recent_start_ = (recent_start_ + 1) % config_.local_window;
    const std::size_t insertion =
        (recent_start_ + recent_size_ - 1) % config_.local_window;
    std::copy(key, key + config_.key_width,
              recent_keys_.begin() + insertion * config_.key_width);
    std::copy(value, value + config_.value_width,
              recent_values_.begin() + insertion * config_.value_width);
    recent_positions_[insertion] = tokens_seen_;
  } else {
    const std::size_t insertion =
        (recent_start_ + recent_size_) % config_.local_window;
    std::copy(key, key + config_.key_width,
              recent_keys_.begin() + insertion * config_.key_width);
    std::copy(value, value + config_.value_width,
              recent_values_.begin() + insertion * config_.value_width);
    recent_positions_[insertion] = tokens_seen_;
    ++recent_size_;
    read_recurrent(query);
  }
  ++tokens_seen_;

  compute_local(query);
  std::size_t bytes_read = 0;
  const std::size_t candidates = retrieve(query, bytes_read);
  const std::size_t older_parts =
      (recurrent_steps_ != 0 ? 1U : 0U) +
      (last_retrieval_count_ != 0 ? 1U : 0U);
  for (std::size_t column = 0; column < config_.value_width; ++column) {
    if (older_parts == 0) {
      output[column] = local_output_[column];
      continue;
    }
    const double older =
        (static_cast<double>(recurrent_output_[column]) +
         static_cast<double>(retrieval_output_[column])) /
        static_cast<double>(older_parts);
    output[column] = static_cast<float>(
        (1.0 - config_.older_weight) * local_output_[column] +
        config_.older_weight * older);
  }
  return {
      .tokens_seen = tokens_seen_,
      .local_tokens = recent_size_,
      .older_tokens = older_size_,
      .recurrent_steps = recurrent_steps_,
      .retrieval_candidates = candidates,
      .retrievals = last_retrieval_count_,
      .bytes_read = bytes_read,
      .state_bytes = allocated_state_bytes(),
      .scratch_bytes = scratch_bytes(),
  };
}

}  // namespace engram
