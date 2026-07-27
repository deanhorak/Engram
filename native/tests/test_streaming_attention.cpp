#include "engram/streaming_attention.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

float dot(const float* left, const float* right, const std::size_t width) {
  float result = 0.0F;
  for (std::size_t index = 0; index < width; ++index) {
    result += left[index] * right[index];
  }
  return result;
}

std::vector<float> dense_row(const std::vector<float>& queries,
                             const std::vector<float>& keys,
                             const std::vector<float>& values,
                             const std::size_t position,
                             const std::size_t heads,
                             const std::size_t kv_heads,
                             const std::size_t width, const float scale) {
  const std::size_t groups = heads / kv_heads;
  std::vector<float> output(heads * width);
  std::vector<float> scores(position + 1);
  for (std::size_t head = 0; head < heads; ++head) {
    const std::size_t kv_head = head / groups;
    const float* query =
        queries.data() + (position * heads + head) * width;
    float maximum = -INFINITY;
    for (std::size_t token = 0; token <= position; ++token) {
      const float* key =
          keys.data() + (token * kv_heads + kv_head) * width;
      scores[token] = dot(query, key, width) * scale;
      maximum = std::max(maximum, scores[token]);
    }
    float denominator = 0.0F;
    for (std::size_t token = 0; token <= position; ++token) {
      scores[token] = std::exp(scores[token] - maximum);
      denominator += scores[token];
    }
    for (std::size_t token = 0; token <= position; ++token) {
      const float weight = scores[token] / denominator;
      const float* value =
          values.data() + (token * kv_heads + kv_head) * width;
      for (std::size_t dimension = 0; dimension < width; ++dimension) {
        output[head * width + dimension] += weight * value[dimension];
      }
    }
  }
  return output;
}

int fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}

}  // namespace

int main() {
  constexpr std::size_t kHeads = 4;
  constexpr std::size_t kKvHeads = 2;
  constexpr std::size_t kWidth = 3;
  constexpr std::size_t kLength = 6;
  constexpr float kScale = 0.577350269F;
  const engram::StreamingAttentionConfig config{
      .query_heads = kHeads,
      .key_value_heads = kKvHeads,
      .head_dimension = kWidth,
      .local_window = 2,
      .older_candidates = 8,
      .older_top_k = 8,
      .sink_tokens = 2,
      .scale = kScale,
  };
  engram::StreamingAttention attention(config);
  const std::size_t state_bytes = attention.allocated_state_bytes();
  const std::size_t scratch_bytes = attention.scratch_bytes();
  std::vector<float> queries(kLength * kHeads * kWidth);
  std::vector<float> keys(kLength * kKvHeads * kWidth);
  std::vector<float> values(kLength * kKvHeads * kWidth);
  for (std::size_t index = 0; index < queries.size(); ++index) {
    queries[index] = std::sin(static_cast<float>(index + 1) * 0.31F);
  }
  for (std::size_t index = 0; index < keys.size(); ++index) {
    keys[index] = std::cos(static_cast<float>(index + 2) * 0.23F);
    values[index] = std::sin(static_cast<float>(index + 3) * 0.17F);
  }
  for (std::size_t position = 0; position < kLength; ++position) {
    std::vector<float> output(kHeads * kWidth);
    const auto metrics = attention.step(
        std::span<const float>(
            queries.data() + position * kHeads * kWidth, kHeads * kWidth),
        std::span<const float>(
            keys.data() + position * kKvHeads * kWidth, kKvHeads * kWidth),
        std::span<const float>(
            values.data() + position * kKvHeads * kWidth, kKvHeads * kWidth),
        output);
    const auto expected = dense_row(queries, keys, values, position, kHeads,
                                    kKvHeads, kWidth, kScale);
    for (std::size_t index = 0; index < output.size(); ++index) {
      if (std::abs(output[index] - expected[index]) > 2e-5F) {
        return fail("streaming attention differs from dense covered history");
      }
    }
    if (metrics.tokens_seen != position + 1 ||
        metrics.local_entries != std::min(position + 1, std::size_t{2}) ||
        metrics.state_bytes != state_bytes ||
        metrics.scratch_bytes != scratch_bytes) {
      return fail("streaming attention metrics are inconsistent");
    }
    if (position < config.local_window) {
      if (metrics.eviction_events != 0 ||
          metrics.older_candidate_entries_scored != 0 ||
          metrics.older_selected_entries != 0 ||
          metrics.sink_insertions != 0 ||
          metrics.heavy_hitter_updates != 0) {
        return fail("older-memory counters advanced before eviction");
      }
    } else {
      const std::uint64_t active_per_head = position - 1;
      if (metrics.eviction_events != 1 ||
          metrics.older_candidate_entries_scored !=
              kHeads * active_per_head ||
          metrics.older_selected_entries != kHeads * active_per_head ||
          metrics.sink_insertions !=
              (position < config.local_window + config.sink_tokens
                   ? kHeads
                   : 0) ||
          metrics.heavy_hitter_updates !=
              (position >= config.local_window + config.sink_tokens
                   ? kHeads
                   : 0)) {
        return fail("older-memory counters are inconsistent");
      }
    }
  }

  for (std::size_t position = 0; position < 32; ++position) {
    std::vector<float> output(kHeads * kWidth);
    attention.step(
        std::span<const float>(queries.data(), kHeads * kWidth),
        std::span<const float>(keys.data(), kKvHeads * kWidth),
        std::span<const float>(values.data(), kKvHeads * kWidth), output);
  }
  if (attention.active_older_entries() > kHeads * config.older_candidates ||
      attention.allocated_state_bytes() != state_bytes) {
    return fail("streaming attention state is not bounded");
  }
  attention.reset();
  if (attention.tokens_seen() != 0 || attention.active_older_entries() != 0 ||
      attention.allocated_state_bytes() != state_bytes) {
    return fail("streaming attention reset failed");
  }
  return 0;
}
