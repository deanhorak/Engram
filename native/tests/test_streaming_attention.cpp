#include "engram/streaming_attention.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

float dot(const float* left, const float* right, const std::size_t width) {
  float result = 0.0F;
  for (std::size_t index = 0; index < width; ++index) {
    result += left[index] * right[index];
  }
  return result;
}

float rounded_bf16(const float value) {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return std::bit_cast<float>((bits >> 16U) << 16U);
}

bool same_metrics(const engram::StreamingAttentionMetrics& left,
                  const engram::StreamingAttentionMetrics& right) {
  return left.tokens_seen == right.tokens_seen &&
         left.local_entries == right.local_entries &&
         left.active_older_entries == right.active_older_entries &&
         left.candidate_key_bytes == right.candidate_key_bytes &&
         left.selected_value_bytes == right.selected_value_bytes &&
         left.local_kv_bytes == right.local_kv_bytes &&
         left.eviction_events == right.eviction_events &&
         left.older_candidate_entries_scored ==
             right.older_candidate_entries_scored &&
         left.older_selected_entries == right.older_selected_entries &&
         left.sink_insertions == right.sink_insertions &&
         left.heavy_hitter_updates == right.heavy_hitter_updates &&
         left.state_bytes == right.state_bytes &&
         left.scratch_bytes == right.scratch_bytes &&
         left.episodic_slots_written == right.episodic_slots_written &&
         left.episodic_active_slots == right.episodic_active_slots &&
         left.episodic_read_events == right.episodic_read_events &&
         left.episodic_entries_read == right.episodic_entries_read &&
         left.episodic_write_bytes == right.episodic_write_bytes &&
         left.episodic_key_read_bytes == right.episodic_key_read_bytes &&
         left.episodic_value_read_bytes == right.episodic_value_read_bytes &&
         left.episodic_duplicate_older_entries_suppressed ==
             right.episodic_duplicate_older_entries_suppressed;
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

std::vector<float> joint_row(const float* query,
                             const std::vector<float>& keys,
                             const std::vector<float>& values,
                             const std::size_t width, const float scale,
                             const std::size_t biased_tail = 0,
                             const float tail_bias = 0.0F) {
  const std::size_t entries = keys.size() / width;
  std::vector<float> scores(entries);
  float maximum = -INFINITY;
  for (std::size_t entry = 0; entry < entries; ++entry) {
    scores[entry] = dot(query, keys.data() + entry * width, width) * scale;
    if (entry >= entries - biased_tail && tail_bias != 0.0F) {
      scores[entry] += tail_bias;
    }
    maximum = std::max(maximum, scores[entry]);
  }
  float denominator = 0.0F;
  for (float& score : scores) {
    score = std::exp(score - maximum);
    denominator += score;
  }
  std::vector<float> output(width);
  for (std::size_t entry = 0; entry < entries; ++entry) {
    const float weight = scores[entry] / denominator;
    for (std::size_t dimension = 0; dimension < width; ++dimension) {
      output[dimension] +=
          weight * values[entry * width + dimension];
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
  engram::StreamingAttention explicit_legacy(config);
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
    std::vector<float> explicit_output(kHeads * kWidth);
    const auto metrics = attention.step(
        std::span<const float>(
            queries.data() + position * kHeads * kWidth, kHeads * kWidth),
        std::span<const float>(
            keys.data() + position * kKvHeads * kWidth, kKvHeads * kWidth),
        std::span<const float>(
            values.data() + position * kKvHeads * kWidth, kKvHeads * kWidth),
        output);
    const auto explicit_metrics = explicit_legacy.step_episodic(
        std::span<const float>(
            queries.data() + position * kHeads * kWidth, kHeads * kWidth),
        std::span<const float>(
            keys.data() + position * kKvHeads * kWidth, kKvHeads * kWidth),
        std::span<const float>(
            values.data() + position * kKvHeads * kWidth, kKvHeads * kWidth),
        engram::StreamingAttention::kNoEpisodicDirective,
        engram::StreamingAttention::kNoEpisodicDirective, explicit_output);
    if (output != explicit_output || !same_metrics(metrics, explicit_metrics)) {
      return fail("legacy step differs from no-directive delegation");
    }
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
        metrics.scratch_bytes != scratch_bytes ||
        metrics.episodic_slots_written != 0 ||
        metrics.episodic_active_slots != 0 ||
        metrics.episodic_read_events != 0 ||
        metrics.episodic_entries_read != 0 ||
        metrics.episodic_write_bytes != 0 ||
        metrics.episodic_key_read_bytes != 0 ||
        metrics.episodic_value_read_bytes != 0 ||
        metrics.episodic_duplicate_older_entries_suppressed != 0) {
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
      attention.active_episodic_slots() != 0 ||
      attention.allocated_state_bytes() != state_bytes) {
    return fail("streaming attention reset failed");
  }

  // The blockwise-QK trace must be observationally inert and its eight
  // RoPE-closed bands must reconstruct every visible native score.
  constexpr std::size_t kQKWidth = 128;
  constexpr std::size_t kQKHeads = 1;
  constexpr std::size_t kQKLength = 28;
  const engram::StreamingAttentionConfig qk_config{
      .query_heads = kQKHeads,
      .key_value_heads = kQKHeads,
      .head_dimension = kQKWidth,
      .local_window = 16,
      .older_candidates = 8,
      .older_top_k = 4,
      .sink_tokens = 2,
      .scale = 1.0F / std::sqrt(static_cast<float>(kQKWidth)),
  };
  engram::StreamingAttention qk_plain(qk_config);
  engram::StreamingAttention qk_traced(qk_config);
  std::vector<float> qk_query(kQKWidth);
  std::vector<float> qk_key(kQKWidth);
  std::vector<float> qk_value(kQKWidth);
  std::vector<std::vector<float>> qk_keys;
  qk_keys.reserve(kQKLength);
  for (std::size_t position = 0; position < kQKLength; ++position) {
    for (std::size_t dimension = 0; dimension < kQKWidth; ++dimension) {
      qk_query[dimension] =
          std::sin(static_cast<float>((position + 1) * (dimension + 3)) *
                   0.017F);
      qk_key[dimension] =
          std::cos(static_cast<float>((position + 2) * (dimension + 5)) *
                   0.013F);
      qk_value[dimension] =
          std::sin(static_cast<float>((position + 4) * (dimension + 7)) *
                   0.011F);
    }
    qk_keys.push_back(qk_key);
    std::vector<float> plain_output(kQKWidth);
    std::vector<float> traced_output(kQKWidth);
    const auto plain_metrics = qk_plain.step(qk_query, qk_key, qk_value,
                                              plain_output);
    std::vector<float> partials(
        kQKHeads * engram::StreamingAttention::kC28TraceEntries *
        engram::StreamingAttention::kQKPartialBands);
    const auto traced_metrics = qk_traced.step_c28_qk_traced(
        qk_query, qk_key, qk_value,
        engram::StreamingAttention::kNoEpisodicDirective,
        engram::StreamingAttention::kNoEpisodicDirective, traced_output,
        partials);
    if (plain_output != traced_output ||
        !same_metrics(plain_metrics, traced_metrics)) {
      return fail("blockwise QK tracing changed attention output or metrics");
    }
    for (std::size_t entry = 0;
         entry < engram::StreamingAttention::kC28TraceEntries; ++entry) {
      float reconstructed = 0.0F;
      for (std::size_t band = 0;
           band < engram::StreamingAttention::kQKPartialBands; ++band) {
        reconstructed += partials[
            entry * engram::StreamingAttention::kQKPartialBands + band];
      }
      const std::size_t local_count = std::min(
          position + 1, engram::StreamingAttention::kRegularTraceLocalEntries);
      if (entry < local_count) {
        const std::size_t first_local = position + 1 - local_count;
        const float expected =
            dot(qk_query.data(), qk_keys[first_local + entry].data(),
                kQKWidth) * qk_config.scale;
        if (!std::isfinite(reconstructed) ||
            std::abs(reconstructed - expected) > 2.0e-5F) {
          return fail("blockwise QK trace failed local score reconstruction");
        }
      } else if (entry <
                 engram::StreamingAttention::kRegularTraceEntries) {
        if (!std::isfinite(reconstructed)) {
          return fail("blockwise QK trace produced a non-finite older score");
        }
      } else {
        for (std::size_t band = 0;
             band < engram::StreamingAttention::kQKPartialBands; ++band) {
          if (partials[entry * engram::StreamingAttention::kQKPartialBands +
                       band] != 0.0F) {
            return fail("blockwise QK trace populated an invalid entry");
          }
        }
      }
    }
  }

  bool rejected = false;
  try {
    const engram::StreamingAttention invalid({
        .query_heads = 1,
        .key_value_heads = 1,
        .head_dimension = 2,
        .local_window = 2,
        .older_candidates = 2,
        .older_top_k = 1,
        .sink_tokens = 0,
        .episodic_slots = 2,
        .episodic_span_size = 0,
        .scale = 0.707106781F,
    });
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) {
    return fail("streaming attention accepted a partial episodic config");
  }
  rejected = false;
  try {
    const engram::StreamingAttention invalid({
        .query_heads = 1,
        .key_value_heads = 1,
        .head_dimension = 2,
        .local_window = 2,
        .older_candidates = 2,
        .older_top_k = 1,
        .sink_tokens = 0,
        .episodic_slots = 3,
        .episodic_span_size = 2,
        .scale = 0.707106781F,
    });
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) {
    return fail("streaming attention accepted a partial episodic span");
  }
  rejected = false;
  try {
    const engram::StreamingAttention invalid({
        .query_heads = 2,
        .key_value_heads = 2,
        .head_dimension = 2,
        .local_window = 2,
        .older_candidates = 2,
        .older_top_k = 1,
        .sink_tokens = 0,
        .episodic_slots = 2,
        .episodic_span_size = 2,
        .episodic_head_mask = {0, 0},
        .scale = 0.707106781F,
    });
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) {
    return fail("streaming attention accepted an all-zero episodic mask");
  }
  rejected = false;
  try {
    const engram::StreamingAttention invalid({
        .query_heads = 2,
        .key_value_heads = 2,
        .head_dimension = 2,
        .local_window = 2,
        .older_candidates = 2,
        .older_top_k = 1,
        .sink_tokens = 0,
        .episodic_slots = 2,
        .episodic_span_size = 2,
        .episodic_head_mask = {1},
        .scale = 0.707106781F,
    });
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) {
    return fail("streaming attention accepted a short episodic mask");
  }
  for (const float invalid_bias :
       {std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()}) {
    rejected = false;
    try {
      const engram::StreamingAttention invalid({
          .query_heads = 1,
          .key_value_heads = 1,
          .head_dimension = 2,
          .local_window = 2,
          .older_candidates = 2,
          .older_top_k = 1,
          .sink_tokens = 0,
          .episodic_slots = 2,
          .episodic_span_size = 2,
          .scale = 0.707106781F,
          .episodic_logit_bias = invalid_bias,
      });
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected) {
      return fail("streaming attention accepted a non-finite episodic bias");
    }
  }
  rejected = false;
  try {
    const engram::StreamingAttention invalid({
        .query_heads = 1,
        .key_value_heads = 1,
        .head_dimension = 2,
        .local_window = 2,
        .older_candidates = 2,
        .older_top_k = 1,
        .sink_tokens = 0,
        .scale = 0.707106781F,
        .episodic_logit_bias = 1.0F,
    });
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) {
    return fail("streaming attention accepted bias without episodic storage");
  }

  const engram::StreamingAttentionConfig episodic_config{
      .query_heads = 1,
      .key_value_heads = 1,
      .head_dimension = 2,
      .local_window = 2,
      .older_candidates = 3,
      .older_top_k = 1,
      .sink_tokens = 0,
      .episodic_slots = 2,
      .episodic_span_size = 2,
      .scale = 0.707106781F,
  };
  const std::vector<float> episodic_queries = {
      0.2F, -0.1F, 0.5F, 0.4F, -0.3F,
      0.8F, 0.1F,  -0.5F, 0.7F, -0.3F,
  };
  const std::vector<float> episodic_keys = {
      0.3333F, -0.625F, -0.2F, 0.91F,
      0.4F,    0.3F,    0.6F,  -0.1F,
      0.1F,    -0.8F,
  };
  const std::vector<float> episodic_values = {
      1.2345F, -0.75F, 0.875F, 2.003F,
      -0.4F,   0.6F,   0.2F,   0.9F,
      1.1F,    -0.2F,
  };

  engram::StreamingAttention directives(episodic_config);
  std::vector<float> directive_output(2);
  rejected = false;
  try {
    directives.step_episodic(
        std::span<const float>(episodic_queries.data(), 2),
        std::span<const float>(episodic_keys.data(), 2),
        std::span<const float>(episodic_values.data(), 2),
        engram::StreamingAttention::kNoEpisodicDirective, 0,
        directive_output);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected || directives.tokens_seen() != 0 ||
      directives.active_episodic_slots() != 0) {
    return fail("inactive episodic read mutated attention state");
  }
  rejected = false;
  try {
    directives.step_episodic(
        std::span<const float>(episodic_queries.data(), 2),
        std::span<const float>(episodic_keys.data(), 2),
        std::span<const float>(episodic_values.data(), 2), 2,
        engram::StreamingAttention::kNoEpisodicDirective,
        directive_output);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected || directives.tokens_seen() != 0) {
    return fail("out-of-range episodic write mutated attention state");
  }
  const auto first_write = directives.step_episodic(
      std::span<const float>(episodic_queries.data(), 2),
      std::span<const float>(episodic_keys.data(), 2),
      std::span<const float>(episodic_values.data(), 2), 0,
      engram::StreamingAttention::kNoEpisodicDirective, directive_output);
  if (first_write.episodic_slots_written != 1 ||
      first_write.episodic_active_slots != 1 ||
      first_write.episodic_write_bytes != 8 ||
      directives.active_episodic_slots() != 1) {
    return fail("episodic write metrics are inexact");
  }
  rejected = false;
  try {
    directives.step_episodic(
        std::span<const float>(episodic_queries.data() + 2, 2),
        std::span<const float>(episodic_keys.data() + 2, 2),
        std::span<const float>(episodic_values.data() + 2, 2), 0,
        engram::StreamingAttention::kNoEpisodicDirective,
        directive_output);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected || directives.tokens_seen() != 1 ||
      directives.active_episodic_slots() != 1) {
    return fail("episodic overwrite mutated attention state");
  }
  rejected = false;
  try {
    directives.step_episodic(
        std::span<const float>(episodic_queries.data() + 2, 2),
        std::span<const float>(episodic_keys.data() + 2, 2),
        std::span<const float>(episodic_values.data() + 2, 2),
        engram::StreamingAttention::kNoEpisodicDirective, 0,
        directive_output);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected || directives.tokens_seen() != 1) {
    return fail("incomplete episodic span read mutated attention state");
  }

  engram::StreamingAttention episodic(episodic_config);
  engram::StreamingAttention traced_episodic(episodic_config);
  engram::StreamingAttention slot_traced_episodic(episodic_config);
  auto explicit_zero_config = episodic_config;
  explicit_zero_config.episodic_logit_bias = 0.0F;
  engram::StreamingAttention explicit_zero(explicit_zero_config);
  auto biased_config = episodic_config;
  biased_config.episodic_logit_bias = 1.25F;
  engram::StreamingAttention biased(biased_config);
  if (episodic.allocated_state_bytes() != 175 ||
      episodic.scratch_bytes() != 68 ||
      explicit_zero.allocated_state_bytes() != 175 ||
      explicit_zero.scratch_bytes() != 68 ||
      biased.allocated_state_bytes() != 175 ||
      biased.scratch_bytes() != 68) {
    return fail("episodic state or scratch accounting is inexact");
  }
  engram::StreamingAttentionMetrics episodic_metrics{};
  engram::StreamingAttentionMetrics biased_metrics{};
  std::vector<float> episodic_output(2);
  std::vector<float> traced_episodic_output(2);
  std::vector<float> slot_traced_episodic_output(2);
  std::vector<float> regular_component(2);
  std::vector<float> episodic_component(2);
  std::vector<float> regular_mass(1);
  std::vector<float> episodic_mass(1);
  std::vector<float> slot_regular_component(2);
  std::vector<float> slot_episodic_component(2);
  std::vector<float> slot_regular_mass(1);
  std::vector<float> slot_episodic_mass(1);
  std::vector<float> episodic_slot_mass(2);
  std::vector<float> episodic_slot_values(4);
  std::vector<float> explicit_zero_output(2);
  std::vector<float> biased_output(2);
  for (std::size_t position = 0; position < 5; ++position) {
    const std::size_t write_slot =
        position < 2
            ? position
            : engram::StreamingAttention::kNoEpisodicDirective;
    const std::size_t read_span =
        position == 4
            ? 0
            : engram::StreamingAttention::kNoEpisodicDirective;
    episodic_metrics = episodic.step_episodic(
        std::span<const float>(
            episodic_queries.data() + position * 2, 2),
        std::span<const float>(
            episodic_keys.data() + position * 2, 2),
        std::span<const float>(
            episodic_values.data() + position * 2, 2),
        write_slot, read_span, episodic_output);
    const auto traced_metrics =
        traced_episodic.step_episodic_traced(
            std::span<const float>(
                episodic_queries.data() + position * 2, 2),
            std::span<const float>(
                episodic_keys.data() + position * 2, 2),
            std::span<const float>(
                episodic_values.data() + position * 2, 2),
            write_slot, read_span, traced_episodic_output,
            regular_component, episodic_component, regular_mass,
            episodic_mass);
    const auto slot_traced_metrics =
        read_span == engram::StreamingAttention::kNoEpisodicDirective
            ? slot_traced_episodic.step_episodic(
                  std::span<const float>(
                      episodic_queries.data() + position * 2, 2),
                  std::span<const float>(
                      episodic_keys.data() + position * 2, 2),
                  std::span<const float>(
                      episodic_values.data() + position * 2, 2),
                  write_slot, read_span, slot_traced_episodic_output)
            : slot_traced_episodic.step_episodic_slots_traced(
                  std::span<const float>(
                      episodic_queries.data() + position * 2, 2),
                  std::span<const float>(
                      episodic_keys.data() + position * 2, 2),
                  std::span<const float>(
                      episodic_values.data() + position * 2, 2),
                  write_slot, read_span, slot_traced_episodic_output,
                  slot_regular_component, slot_episodic_component,
                  slot_regular_mass, slot_episodic_mass,
                  episodic_slot_mass, episodic_slot_values);
    const auto explicit_zero_metrics = explicit_zero.step_episodic(
        std::span<const float>(
            episodic_queries.data() + position * 2, 2),
        std::span<const float>(
            episodic_keys.data() + position * 2, 2),
        std::span<const float>(
            episodic_values.data() + position * 2, 2),
        write_slot, read_span, explicit_zero_output);
    biased_metrics = biased.step_episodic(
        std::span<const float>(
            episodic_queries.data() + position * 2, 2),
        std::span<const float>(
            episodic_keys.data() + position * 2, 2),
        std::span<const float>(
            episodic_values.data() + position * 2, 2),
        write_slot, read_span, biased_output);
    if (episodic_output != explicit_zero_output ||
        !same_metrics(episodic_metrics, explicit_zero_metrics) ||
        traced_episodic_output != episodic_output ||
        !same_metrics(traced_metrics, episodic_metrics) ||
        slot_traced_episodic_output != episodic_output ||
        !same_metrics(slot_traced_metrics, episodic_metrics)) {
      return fail("explicit zero episodic bias changed the legacy route");
    }
  }
  const std::vector<float> visible_keys = {
      episodic_keys[6],
      episodic_keys[7],
      episodic_keys[8],
      episodic_keys[9],
      episodic_keys[4],
      episodic_keys[5],
      rounded_bf16(episodic_keys[0]),
      rounded_bf16(episodic_keys[1]),
      rounded_bf16(episodic_keys[2]),
      rounded_bf16(episodic_keys[3]),
  };
  const std::vector<float> visible_values = {
      episodic_values[6],
      episodic_values[7],
      episodic_values[8],
      episodic_values[9],
      episodic_values[4],
      episodic_values[5],
      rounded_bf16(episodic_values[0]),
      rounded_bf16(episodic_values[1]),
      rounded_bf16(episodic_values[2]),
      rounded_bf16(episodic_values[3]),
  };
  const auto joint = joint_row(episodic_queries.data() + 8, visible_keys,
                               visible_values, 2, episodic_config.scale);
  const auto biased_joint =
      joint_row(episodic_queries.data() + 8, visible_keys,
                visible_values, 2, episodic_config.scale, 2,
                biased_config.episodic_logit_bias);
  for (std::size_t dimension = 0; dimension < joint.size(); ++dimension) {
    if (std::abs(episodic_output[dimension] - joint[dimension]) > 2e-6F) {
      return fail("episodic joint softmax differs from explicit union");
    }
    if (std::abs(biased_output[dimension] - biased_joint[dimension]) >
        2e-6F) {
      return fail("biased episodic softmax differs from explicit union");
    }
  }
  if (biased_output == episodic_output ||
      !same_metrics(biased_metrics, episodic_metrics)) {
    return fail("episodic bias did not isolate attention weighting");
  }
  const float gamma = std::exp(biased_config.episodic_logit_bias);
  const float adjusted_denominator =
      regular_mass[0] + gamma * episodic_mass[0];
  if (!(regular_mass[0] > 0.0F) || !(episodic_mass[0] > 0.0F) ||
      std::abs(regular_mass[0] + episodic_mass[0] - 1.0F) >
          2.0e-6F) {
    return fail("episodic partition masses are invalid");
  }
  for (std::size_t dimension = 0; dimension < episodic_output.size();
       ++dimension) {
    if (std::abs(regular_component[dimension] +
                     episodic_component[dimension] -
                 episodic_output[dimension]) >
            2.0e-6F ||
        std::abs((regular_component[dimension] +
                  gamma * episodic_component[dimension]) /
                     adjusted_denominator -
                 biased_output[dimension]) >
            2.0e-6F) {
      return fail(
          "episodic partition trace did not reconstruct the joint softmax");
    }
  }
  if (slot_regular_component != regular_component ||
      slot_episodic_component != episodic_component ||
      slot_regular_mass != regular_mass ||
      slot_episodic_mass != episodic_mass ||
      !(episodic_slot_mass[0] > 0.0F) ||
      !(episodic_slot_mass[1] > 0.0F) ||
      std::abs(
          episodic_slot_mass[0] + episodic_slot_mass[1] -
          episodic_mass[0]) >
          2.0e-6F) {
    return fail("episodic slot trace did not preserve partition masses");
  }
  for (std::size_t slot = 0; slot < 2; ++slot) {
    for (std::size_t dimension = 0; dimension < 2; ++dimension) {
      const std::size_t flat = slot * 2 + dimension;
      if (episodic_slot_values[flat] !=
          rounded_bf16(episodic_values[flat])) {
        return fail("episodic slot trace changed a BF16-decoded value");
      }
    }
  }
  for (std::size_t dimension = 0; dimension < 2; ++dimension) {
    const float reconstructed =
        episodic_slot_mass[0] * episodic_slot_values[dimension] +
        episodic_slot_mass[1] *
            episodic_slot_values[2 + dimension];
    if (std::abs(
            reconstructed - episodic_component[dimension]) >
        2.0e-6F) {
      return fail("episodic slot trace did not reconstruct its component");
    }
  }
  const std::array<std::pair<std::uint32_t, float>, 7> gamma_grid = {{
      {0xC0051592U, 0.125F},
      {0xBFB17218U, 0.25F},
      {0xBF317218U, 0.5F},
      {0x00000000U, 1.0F},
      {0x3F317218U, 2.0F},
      {0x3FB17218U, 4.0F},
      {0x40051592U, 8.0F},
  }};
  for (const auto& [bias_bits, grid_gamma] : gamma_grid) {
    auto grid_config = episodic_config;
    grid_config.episodic_logit_bias =
        std::bit_cast<float>(bias_bits);
    engram::StreamingAttention grid_attention(grid_config);
    std::vector<float> grid_output(2);
    for (std::size_t position = 0; position < 5; ++position) {
      const std::size_t write_slot =
          position < 2
              ? position
              : engram::StreamingAttention::kNoEpisodicDirective;
      const std::size_t read_span =
          position == 4
              ? 0
              : engram::StreamingAttention::kNoEpisodicDirective;
      grid_attention.step_episodic(
          std::span<const float>(
              episodic_queries.data() + position * 2, 2),
          std::span<const float>(
              episodic_keys.data() + position * 2, 2),
          std::span<const float>(
              episodic_values.data() + position * 2, 2),
          write_slot, read_span, grid_output);
    }
    const float grid_denominator =
        regular_mass[0] + grid_gamma * episodic_mass[0];
    for (std::size_t dimension = 0; dimension < grid_output.size();
         ++dimension) {
      const float reconstructed =
          (regular_component[dimension] +
           grid_gamma * episodic_component[dimension]) /
          grid_denominator;
      if (std::abs(reconstructed - grid_output[dimension]) >
          2.0e-6F) {
        return fail(
            "episodic partition trace missed a frozen gamma grid arm");
      }
    }
  }
  const auto first_slot_output = slot_traced_episodic_output;
  const auto first_slot_regular_component = slot_regular_component;
  const auto first_slot_episodic_component = slot_episodic_component;
  const auto first_slot_regular_mass = slot_regular_mass;
  const auto first_slot_episodic_mass = slot_episodic_mass;
  const auto first_slot_mass = episodic_slot_mass;
  const auto first_slot_values = episodic_slot_values;
  slot_traced_episodic.reset();
  engram::StreamingAttentionMetrics slot_replay_metrics{};
  for (std::size_t position = 0; position < 5; ++position) {
    const std::size_t write_slot =
        position < 2
            ? position
            : engram::StreamingAttention::kNoEpisodicDirective;
    const std::size_t read_span =
        position == 4
            ? 0
            : engram::StreamingAttention::kNoEpisodicDirective;
    if (read_span == engram::StreamingAttention::kNoEpisodicDirective) {
      slot_replay_metrics = slot_traced_episodic.step_episodic(
          std::span<const float>(
              episodic_queries.data() + position * 2, 2),
          std::span<const float>(
              episodic_keys.data() + position * 2, 2),
          std::span<const float>(
              episodic_values.data() + position * 2, 2),
          write_slot, read_span, slot_traced_episodic_output);
    } else {
      slot_replay_metrics =
          slot_traced_episodic.step_episodic_slots_traced(
              std::span<const float>(
                  episodic_queries.data() + position * 2, 2),
              std::span<const float>(
                  episodic_keys.data() + position * 2, 2),
              std::span<const float>(
                  episodic_values.data() + position * 2, 2),
              write_slot, read_span, slot_traced_episodic_output,
              slot_regular_component, slot_episodic_component,
              slot_regular_mass, slot_episodic_mass,
              episodic_slot_mass, episodic_slot_values);
    }
  }
  if (slot_traced_episodic_output != first_slot_output ||
      slot_regular_component != first_slot_regular_component ||
      slot_episodic_component != first_slot_episodic_component ||
      slot_regular_mass != first_slot_regular_mass ||
      slot_episodic_mass != first_slot_episodic_mass ||
      episodic_slot_mass != first_slot_mass ||
      episodic_slot_values != first_slot_values ||
      !same_metrics(slot_replay_metrics, episodic_metrics) ||
      slot_traced_episodic.tokens_seen() != 5 ||
      slot_traced_episodic.active_episodic_slots() != 2) {
    return fail("episodic slot trace reset replay was not deterministic");
  }
  if (episodic_metrics.tokens_seen != 5 ||
      episodic_metrics.active_older_entries != 3 ||
      episodic_metrics.candidate_key_bytes != 24 ||
      episodic_metrics.selected_value_bytes != 8 ||
      episodic_metrics.local_kv_bytes != 32 ||
      episodic_metrics.older_candidate_entries_scored != 3 ||
      episodic_metrics.older_selected_entries != 1 ||
      episodic_metrics.episodic_slots_written != 0 ||
      episodic_metrics.episodic_active_slots != 2 ||
      episodic_metrics.episodic_read_events != 1 ||
      episodic_metrics.episodic_entries_read != 2 ||
      episodic_metrics.episodic_write_bytes != 0 ||
      episodic_metrics.episodic_key_read_bytes != 8 ||
      episodic_metrics.episodic_value_read_bytes != 8 ||
      episodic_metrics.episodic_duplicate_older_entries_suppressed != 2 ||
      episodic_metrics.state_bytes != 175 ||
      episodic_metrics.scratch_bytes != 68) {
    return fail("episodic read or deduplication metrics are inexact");
  }
  episodic.reset();
  if (episodic.tokens_seen() != 0 ||
      episodic.active_older_entries() != 0 ||
      episodic.active_episodic_slots() != 0 ||
      episodic.allocated_state_bytes() != 175 ||
      episodic.scratch_bytes() != 68) {
    return fail("episodic reset retained active state or changed capacity");
  }
  const auto reset_write = episodic.step_episodic(
      std::span<const float>(episodic_queries.data(), 2),
      std::span<const float>(episodic_keys.data(), 2),
      std::span<const float>(episodic_values.data(), 2), 0,
      engram::StreamingAttention::kNoEpisodicDirective, episodic_output);
  if (reset_write.episodic_slots_written != 1 ||
      reset_write.episodic_active_slots != 1) {
    return fail("episodic slot was not reusable after reset");
  }

  engram::StreamingAttention invalid_trace(episodic_config);
  rejected = false;
  try {
    invalid_trace.step_episodic_traced(
        std::span<const float>(episodic_queries.data(), 2),
        std::span<const float>(episodic_keys.data(), 2),
        std::span<const float>(episodic_values.data(), 2),
        engram::StreamingAttention::kNoEpisodicDirective,
        engram::StreamingAttention::kNoEpisodicDirective,
        directive_output, std::span<float>(regular_component.data(), 1),
        episodic_component, regular_mass, episodic_mass);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected || invalid_trace.tokens_seen() != 0) {
    return fail("invalid episodic trace storage mutated attention state");
  }
  engram::StreamingAttention invalid_slot_trace(episodic_config);
  std::vector<float> valid_slot_mass(2);
  std::vector<float> valid_slot_values(4);
  rejected = false;
  try {
    invalid_slot_trace.step_episodic_slots_traced(
        std::span<const float>(episodic_queries.data(), 2),
        std::span<const float>(episodic_keys.data(), 2),
        std::span<const float>(episodic_values.data(), 2),
        engram::StreamingAttention::kNoEpisodicDirective,
        engram::StreamingAttention::kNoEpisodicDirective,
        directive_output, regular_component, episodic_component,
        regular_mass, episodic_mass, valid_slot_mass,
        valid_slot_values);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected || invalid_slot_trace.tokens_seen() != 0 ||
      invalid_slot_trace.active_episodic_slots() != 0) {
    return fail("slot trace accepted a missing episodic read");
  }
  rejected = false;
  try {
    invalid_slot_trace.step_episodic_slots_traced(
        std::span<const float>(episodic_queries.data(), 2),
        std::span<const float>(episodic_keys.data(), 2),
        std::span<const float>(episodic_values.data(), 2),
        engram::StreamingAttention::kNoEpisodicDirective, 0,
        directive_output, regular_component, episodic_component,
        regular_mass, episodic_mass,
        std::span<float>(valid_slot_mass.data(), 1),
        valid_slot_values);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected || invalid_slot_trace.tokens_seen() != 0 ||
      invalid_slot_trace.active_episodic_slots() != 0) {
    return fail("slot trace accepted a short mass array");
  }
  rejected = false;
  try {
    invalid_slot_trace.step_episodic_slots_traced(
        std::span<const float>(episodic_queries.data(), 2),
        std::span<const float>(episodic_keys.data(), 2),
        std::span<const float>(episodic_values.data(), 2),
        engram::StreamingAttention::kNoEpisodicDirective, 0,
        directive_output, regular_component, episodic_component,
        regular_mass, episodic_mass, valid_slot_mass,
        std::span<float>(valid_slot_values.data(), 3));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected || invalid_slot_trace.tokens_seen() != 0 ||
      invalid_slot_trace.active_episodic_slots() != 0) {
    return fail("slot trace accepted a short value array");
  }
  rejected = false;
  try {
    invalid_slot_trace.step_episodic_slots_traced(
        std::span<const float>(episodic_queries.data(), 2),
        std::span<const float>(episodic_keys.data(), 2),
        std::span<const float>(episodic_values.data(), 2),
        engram::StreamingAttention::kNoEpisodicDirective, 0,
        directive_output, regular_component, episodic_component,
        regular_mass, episodic_mass, valid_slot_mass,
        valid_slot_values);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected || invalid_slot_trace.tokens_seen() != 0 ||
      invalid_slot_trace.active_episodic_slots() != 0) {
    return fail("incomplete slot trace read mutated attention state");
  }

  const engram::StreamingAttentionConfig regular_trace_config{
      .query_heads = 1,
      .key_value_heads = 1,
      .head_dimension = 2,
      .local_window =
          engram::StreamingAttention::kRegularTraceLocalEntries,
      .older_candidates = 12,
      .older_top_k =
          engram::StreamingAttention::kRegularTraceOlderEntries,
      .sink_tokens = 0,
      .episodic_slots = 8,
      .episodic_span_size = 8,
      .scale = 0.707106781F,
  };
  constexpr std::size_t kRegularTraceLength = 25;
  std::vector<float> regular_trace_queries(kRegularTraceLength * 2);
  std::vector<float> regular_trace_keys(kRegularTraceLength * 2);
  std::vector<float> regular_trace_values(kRegularTraceLength * 2);
  for (std::size_t position = 0; position < kRegularTraceLength;
       ++position) {
    regular_trace_queries[position * 2] =
        0.11F * static_cast<float>(position + 1);
    regular_trace_queries[position * 2 + 1] =
        -0.07F * static_cast<float>((position % 5) + 1);
    regular_trace_keys[position * 2] =
        std::sin(static_cast<float>(position + 1) * 0.19F);
    regular_trace_keys[position * 2 + 1] =
        std::cos(static_cast<float>(position + 2) * 0.17F);
    regular_trace_values[position * 2] =
        static_cast<float>(position) + 0.25F;
    regular_trace_values[position * 2 + 1] =
        -static_cast<float>(position) - 0.75F;
  }

  engram::StreamingAttention early_regular_trace(regular_trace_config);
  engram::StreamingAttention early_regular_reference(regular_trace_config);
  std::array<float, 2> early_regular_output{};
  std::array<float, 2> early_regular_reference_output{};
  std::array<float, engram::StreamingAttention::kRegularTraceEntries>
      early_entry_mass{};
  std::array<
      float, engram::StreamingAttention::kRegularTraceEntries * 2>
      early_entry_values{};
  std::array<std::uint8_t,
             engram::StreamingAttention::kRegularTraceEntries>
      early_valid_kind{};
  std::array<std::uint64_t,
             engram::StreamingAttention::kRegularTraceEntries>
      early_positions{};
  const auto early_regular_metrics =
      early_regular_trace.step_regular_entries_traced(
          std::span<const float>(regular_trace_queries.data(), 2),
          std::span<const float>(regular_trace_keys.data(), 2),
          std::span<const float>(regular_trace_values.data(), 2),
          early_regular_output, early_entry_mass, early_entry_values,
          early_valid_kind, early_positions);
  const auto early_reference_metrics = early_regular_reference.step(
      std::span<const float>(regular_trace_queries.data(), 2),
      std::span<const float>(regular_trace_keys.data(), 2),
      std::span<const float>(regular_trace_values.data(), 2),
      early_regular_reference_output);
  if (early_regular_output != early_regular_reference_output ||
      !same_metrics(early_regular_metrics, early_reference_metrics) ||
      early_entry_mass[0] != 1.0F ||
      early_valid_kind[0] !=
          engram::StreamingAttention::kRegularTraceLocal ||
      early_positions[0] != 0 ||
      early_entry_values[0] != regular_trace_values[0] ||
      early_entry_values[1] != regular_trace_values[1]) {
    return fail("early regular-entry trace changed the attention route");
  }
  for (std::size_t entry = 1;
       entry < engram::StreamingAttention::kRegularTraceEntries;
       ++entry) {
    if (early_entry_mass[entry] != 0.0F ||
        early_valid_kind[entry] !=
            engram::StreamingAttention::kRegularTraceInvalid ||
        early_positions[entry] !=
            engram::StreamingAttention::kNoEpisodicDirective ||
        early_entry_values[entry * 2] != 0.0F ||
        early_entry_values[entry * 2 + 1] != 0.0F) {
      return fail("regular-entry trace padding is not canonical");
    }
  }

  engram::StreamingAttention regular_trace(regular_trace_config);
  engram::StreamingAttention regular_trace_reference(
      regular_trace_config);
  std::array<float, 2> regular_trace_output{};
  std::array<float, 2> regular_trace_reference_output{};
  std::array<float, 2> trace_regular_component{};
  std::array<float, 2> trace_episodic_component{};
  std::array<float, 1> trace_regular_mass{};
  std::array<float, 1> trace_episodic_mass{};
  std::array<float, 8> trace_slot_mass{};
  std::array<float, 16> trace_slot_values{};
  std::array<float, engram::StreamingAttention::kRegularTraceEntries>
      regular_entry_mass{};
  std::array<
      float, engram::StreamingAttention::kRegularTraceEntries * 2>
      regular_entry_values{};
  std::array<std::uint8_t,
             engram::StreamingAttention::kRegularTraceEntries>
      regular_entry_valid_kind{};
  std::array<std::uint64_t,
             engram::StreamingAttention::kRegularTraceEntries>
      regular_entry_positions{};
  engram::StreamingAttentionMetrics regular_trace_metrics{};
  engram::StreamingAttentionMetrics regular_trace_reference_metrics{};
  const auto run_regular_trace = [&](engram::StreamingAttention& traced,
                                     const bool capture) {
    for (std::size_t position = 0; position < kRegularTraceLength;
         ++position) {
      const std::size_t write_slot =
          position >= 4 && position < 12
              ? position - 4
              : engram::StreamingAttention::kNoEpisodicDirective;
      const std::size_t read_span =
          position + 1 == kRegularTraceLength
              ? 0
              : engram::StreamingAttention::kNoEpisodicDirective;
      if (capture && position + 1 == kRegularTraceLength) {
        regular_trace_metrics = traced.step_episodic_full_traced(
            std::span<const float>(
                regular_trace_queries.data() + position * 2, 2),
            std::span<const float>(
                regular_trace_keys.data() + position * 2, 2),
            std::span<const float>(
                regular_trace_values.data() + position * 2, 2),
            write_slot, read_span, regular_trace_output,
            trace_regular_component, trace_episodic_component,
            trace_regular_mass, trace_episodic_mass, trace_slot_mass,
            trace_slot_values, regular_entry_mass,
            regular_entry_values, regular_entry_valid_kind,
            regular_entry_positions);
      } else if (!capture && position + 1 == kRegularTraceLength) {
        regular_trace_reference_metrics =
            traced.step_episodic_slots_traced(
                std::span<const float>(
                    regular_trace_queries.data() + position * 2, 2),
                std::span<const float>(
                    regular_trace_keys.data() + position * 2, 2),
                std::span<const float>(
                    regular_trace_values.data() + position * 2, 2),
                write_slot, read_span, regular_trace_reference_output,
                trace_regular_component, trace_episodic_component,
                trace_regular_mass, trace_episodic_mass,
                trace_slot_mass, trace_slot_values);
      } else {
        auto& output = capture ? regular_trace_output
                               : regular_trace_reference_output;
        traced.step_episodic(
            std::span<const float>(
                regular_trace_queries.data() + position * 2, 2),
            std::span<const float>(
                regular_trace_keys.data() + position * 2, 2),
            std::span<const float>(
                regular_trace_values.data() + position * 2, 2),
            write_slot, read_span, output);
      }
    }
  };
  run_regular_trace(regular_trace_reference, false);
  run_regular_trace(regular_trace, true);
  if (regular_trace_output != regular_trace_reference_output ||
      !same_metrics(regular_trace_metrics,
                    regular_trace_reference_metrics)) {
    return fail("regular-entry tracing changed output or counters");
  }
  for (std::size_t local = 0;
       local < engram::StreamingAttention::kRegularTraceLocalEntries;
       ++local) {
    const std::uint64_t position = local + 9;
    if (regular_entry_valid_kind[local] !=
            engram::StreamingAttention::kRegularTraceLocal ||
        regular_entry_positions[local] != position ||
        regular_entry_mass[local] <= 0.0F ||
        regular_entry_values[local * 2] !=
            regular_trace_values[position * 2] ||
        regular_entry_values[local * 2 + 1] !=
            regular_trace_values[position * 2 + 1]) {
      return fail(
          "regular-entry local layout is not chronological and exact");
    }
  }
  std::array<std::size_t,
             engram::StreamingAttention::kRegularTraceOlderEntries>
      expected_older = {0, 1, 2, 3};
  const float* final_query =
      regular_trace_queries.data() + (kRegularTraceLength - 1) * 2;
  std::sort(
      expected_older.begin(), expected_older.end(),
      [&](const std::size_t left, const std::size_t right) {
        const float left_score =
            dot(final_query, regular_trace_keys.data() + left * 2, 2) *
            regular_trace_config.scale;
        const float right_score =
            dot(final_query, regular_trace_keys.data() + right * 2, 2) *
            regular_trace_config.scale;
        return left_score != right_score ? left_score > right_score
                                         : left < right;
      });
  for (std::size_t older = 0;
       older < engram::StreamingAttention::kRegularTraceOlderEntries;
       ++older) {
    const std::size_t entry =
        engram::StreamingAttention::kRegularTraceLocalEntries + older;
    const std::size_t position = expected_older[older];
    if (regular_entry_valid_kind[entry] !=
            engram::StreamingAttention::kRegularTraceOlder ||
        regular_entry_positions[entry] != position ||
        regular_entry_mass[entry] <= 0.0F ||
        regular_entry_values[entry * 2] !=
            regular_trace_values[position * 2] ||
        regular_entry_values[entry * 2 + 1] !=
            regular_trace_values[position * 2 + 1]) {
      return fail(
          "regular-entry older layout changed native selection order");
    }
  }
  float reconstructed_regular_mass = 0.0F;
  std::array<float, 2> reconstructed_regular_component{};
  for (std::size_t entry = 0;
       entry < engram::StreamingAttention::kRegularTraceEntries;
       ++entry) {
    reconstructed_regular_mass += regular_entry_mass[entry];
    for (std::size_t dimension = 0; dimension < 2; ++dimension) {
      reconstructed_regular_component[dimension] +=
          regular_entry_mass[entry] *
          regular_entry_values[entry * 2 + dimension];
    }
  }
  if (std::abs(reconstructed_regular_mass - trace_regular_mass[0]) >
      2.0e-6F) {
    return fail("regular-entry masses do not reconstruct regular mass");
  }
  for (std::size_t dimension = 0; dimension < 2; ++dimension) {
    if (std::abs(reconstructed_regular_component[dimension] -
                 trace_regular_component[dimension]) > 2.0e-6F) {
      return fail(
          "regular-entry values do not reconstruct regular component");
    }
  }
  const auto first_regular_trace_output = regular_trace_output;
  const auto first_regular_entry_mass = regular_entry_mass;
  const auto first_regular_entry_values = regular_entry_values;
  const auto first_regular_entry_valid_kind =
      regular_entry_valid_kind;
  const auto first_regular_entry_positions = regular_entry_positions;
  const auto first_regular_trace_metrics = regular_trace_metrics;
  regular_trace.reset();
  run_regular_trace(regular_trace, true);
  if (regular_trace_output != first_regular_trace_output ||
      regular_entry_mass != first_regular_entry_mass ||
      regular_entry_values != first_regular_entry_values ||
      regular_entry_valid_kind != first_regular_entry_valid_kind ||
      regular_entry_positions != first_regular_entry_positions ||
      !same_metrics(regular_trace_metrics,
                    first_regular_trace_metrics)) {
    return fail("regular-entry trace reset replay is not exact");
  }

  engram::StreamingAttention invalid_regular_trace(
      regular_trace_config);
  rejected = false;
  try {
    invalid_regular_trace.step_regular_entries_traced(
        std::span<const float>(regular_trace_queries.data(), 2),
        std::span<const float>(regular_trace_keys.data(), 2),
        std::span<const float>(regular_trace_values.data(), 2),
        early_regular_output,
        std::span<float>(early_entry_mass.data(),
                         early_entry_mass.size() - 1),
        early_entry_values, early_valid_kind, early_positions);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected || invalid_regular_trace.tokens_seen() != 0) {
    return fail("invalid regular-entry trace storage mutated state");
  }
  auto invalid_regular_policy = regular_trace_config;
  invalid_regular_policy.local_window = 15;
  engram::StreamingAttention invalid_regular_policy_trace(
      invalid_regular_policy);
  rejected = false;
  try {
    invalid_regular_policy_trace.step_regular_entries_traced(
        std::span<const float>(regular_trace_queries.data(), 2),
        std::span<const float>(regular_trace_keys.data(), 2),
        std::span<const float>(regular_trace_values.data(), 2),
        early_regular_output, early_entry_mass, early_entry_values,
        early_valid_kind, early_positions);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected || invalid_regular_policy_trace.tokens_seen() != 0) {
    return fail("regular-entry trace accepted a non-W16/K4 policy");
  }

  auto tracked_config = episodic_config;
  tracked_config.local_window = 5;
  tracked_config.episodic_slots = 0;
  tracked_config.episodic_span_size = 0;
  engram::StreamingAttention tracked(tracked_config);
  engram::StreamingAttention tracked_reference(tracked_config);
  std::vector<float> tracked_output(2);
  std::vector<float> tracked_reference_output(2);
  for (std::size_t position = 0; position < 4; ++position) {
    tracked.step(
        std::span<const float>(
            episodic_queries.data() + position * 2, 2),
        std::span<const float>(
            episodic_keys.data() + position * 2, 2),
        std::span<const float>(
            episodic_values.data() + position * 2, 2),
        tracked_output);
    tracked_reference.step(
        std::span<const float>(
            episodic_queries.data() + position * 2, 2),
        std::span<const float>(
            episodic_keys.data() + position * 2, 2),
        std::span<const float>(
            episodic_values.data() + position * 2, 2),
        tracked_reference_output);
  }
  const std::array<std::uint64_t, 2> tracked_positions = {0, 1};
  std::array<float, 1> tracked_mass{};
  const auto tracked_metrics = tracked.step_tracked_positions(
      std::span<const float>(episodic_queries.data() + 8, 2),
      std::span<const float>(episodic_keys.data() + 8, 2),
      std::span<const float>(episodic_values.data() + 8, 2),
      tracked_positions, tracked_mass, tracked_output);
  const auto tracked_reference_metrics = tracked_reference.step(
      std::span<const float>(episodic_queries.data() + 8, 2),
      std::span<const float>(episodic_keys.data() + 8, 2),
      std::span<const float>(episodic_values.data() + 8, 2),
      tracked_reference_output);
  std::array<float, 5> tracked_scores{};
  float tracked_maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t position = 0; position < tracked_scores.size();
       ++position) {
    tracked_scores[position] =
        dot(episodic_queries.data() + 8,
            episodic_keys.data() + position * 2, 2) *
        tracked_config.scale;
    tracked_maximum =
        std::max(tracked_maximum, tracked_scores[position]);
  }
  float tracked_denominator = 0.0F;
  for (float& score : tracked_scores) {
    score = std::exp(score - tracked_maximum);
    tracked_denominator += score;
  }
  const float expected_tracked_mass =
      (tracked_scores[0] + tracked_scores[1]) /
      tracked_denominator;
  if (tracked_output != tracked_reference_output ||
      !same_metrics(tracked_metrics, tracked_reference_metrics) ||
      std::abs(tracked_mass[0] - expected_tracked_mass) > 2.0e-6F) {
    return fail("tracked-position mass changed or mismeasured attention");
  }
  engram::StreamingAttention invalid_tracked(tracked_config);
  const std::array<std::uint64_t, 2> duplicate_positions = {0, 0};
  rejected = false;
  try {
    invalid_tracked.step_tracked_positions(
        std::span<const float>(episodic_queries.data(), 2),
        std::span<const float>(episodic_keys.data(), 2),
        std::span<const float>(episodic_values.data(), 2),
        duplicate_positions, tracked_mass, tracked_output);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected || invalid_tracked.tokens_seen() != 0) {
    return fail("invalid tracked positions mutated attention state");
  }

  const engram::StreamingAttentionConfig ungated_config{
      .query_heads = 2,
      .key_value_heads = 2,
      .head_dimension = 2,
      .local_window = 2,
      .older_candidates = 3,
      .older_top_k = 1,
      .sink_tokens = 0,
      .episodic_slots = 2,
      .episodic_span_size = 2,
      .scale = 0.707106781F,
  };
  auto all_ones_config = ungated_config;
  all_ones_config.episodic_head_mask = {1, 1};
  auto mixed_mask_config = ungated_config;
  mixed_mask_config.episodic_head_mask = {1, 0};
  auto biased_mixed_config = mixed_mask_config;
  biased_mixed_config.episodic_logit_bias = 1.25F;
  auto legacy_config = ungated_config;
  legacy_config.episodic_slots = 0;
  legacy_config.episodic_span_size = 0;
  engram::StreamingAttention ungated(ungated_config);
  engram::StreamingAttention all_ones(all_ones_config);
  engram::StreamingAttention mixed_mask(mixed_mask_config);
  engram::StreamingAttention slot_traced_mixed_mask(mixed_mask_config);
  engram::StreamingAttention biased_mixed(biased_mixed_config);
  engram::StreamingAttention masked_legacy(legacy_config);
  std::vector<float> masked_queries(5 * 4);
  std::vector<float> masked_keys(5 * 4);
  std::vector<float> masked_values(5 * 4);
  for (std::size_t index = 0; index < masked_queries.size(); ++index) {
    masked_queries[index] =
        std::sin(static_cast<float>(index + 1) * 0.29F);
    masked_keys[index] =
        std::cos(static_cast<float>(index + 2) * 0.19F);
    masked_values[index] =
        std::sin(static_cast<float>(index + 3) * 0.13F);
  }
  engram::StreamingAttentionMetrics mixed_metrics{};
  std::vector<float> mixed_slot_output(4);
  std::vector<float> mixed_regular_component(4);
  std::vector<float> mixed_episodic_component(4);
  std::vector<float> mixed_regular_mass(2);
  std::vector<float> mixed_episodic_mass(2);
  std::vector<float> mixed_slot_mass(4);
  std::vector<float> mixed_slot_values(8);
  for (std::size_t position = 0; position < 5; ++position) {
    const std::size_t write_slot =
        position < 2
            ? position
            : engram::StreamingAttention::kNoEpisodicDirective;
    const std::size_t read_span =
        position == 4
            ? 0
            : engram::StreamingAttention::kNoEpisodicDirective;
    const auto query = std::span<const float>(
        masked_queries.data() + position * 4, 4);
    const auto key = std::span<const float>(
        masked_keys.data() + position * 4, 4);
    const auto value = std::span<const float>(
        masked_values.data() + position * 4, 4);
    std::vector<float> ungated_output(4);
    std::vector<float> all_ones_output(4);
    std::vector<float> mixed_output(4);
    std::vector<float> biased_mixed_output(4);
    std::vector<float> legacy_output(4);
    const auto ungated_metrics = ungated.step_episodic(
        query, key, value, write_slot, read_span, ungated_output);
    const auto all_ones_metrics = all_ones.step_episodic(
        query, key, value, write_slot, read_span, all_ones_output);
    mixed_metrics = mixed_mask.step_episodic(
        query, key, value, write_slot, read_span, mixed_output);
    const auto mixed_slot_metrics =
        read_span == engram::StreamingAttention::kNoEpisodicDirective
            ? slot_traced_mixed_mask.step_episodic(
                  query, key, value, write_slot, read_span,
                  mixed_slot_output)
            : slot_traced_mixed_mask.step_episodic_slots_traced(
                  query, key, value, write_slot, read_span,
                  mixed_slot_output, mixed_regular_component,
                  mixed_episodic_component, mixed_regular_mass,
                  mixed_episodic_mass, mixed_slot_mass,
                  mixed_slot_values);
    const auto biased_mixed_metrics = biased_mixed.step_episodic(
        query, key, value, write_slot, read_span, biased_mixed_output);
    masked_legacy.step(query, key, value, legacy_output);
    if (ungated_output != all_ones_output ||
        !same_metrics(ungated_metrics, all_ones_metrics)) {
      return fail("all-ones episodic mask changed legacy episodic behavior");
    }
    if (mixed_output[2] != legacy_output[2] ||
        mixed_output[3] != legacy_output[3]) {
      return fail("unselected episodic head changed legacy softmax");
    }
    if (mixed_slot_output != mixed_output ||
        !same_metrics(mixed_slot_metrics, mixed_metrics)) {
      return fail("masked episodic slot trace changed the attention route");
    }
    if (biased_mixed_output[2] != mixed_output[2] ||
        biased_mixed_output[3] != mixed_output[3] ||
        !same_metrics(biased_mixed_metrics, mixed_metrics)) {
      return fail("episodic bias changed an unselected head or capacity");
    }
    if (read_span != engram::StreamingAttention::kNoEpisodicDirective &&
        biased_mixed_output[0] == mixed_output[0] &&
        biased_mixed_output[1] == mixed_output[1]) {
      return fail("episodic bias did not change the selected head");
    }
  }
  if (!(mixed_slot_mass[0] > 0.0F) ||
      !(mixed_slot_mass[1] > 0.0F) ||
      mixed_slot_mass[2] != 0.0F ||
      mixed_slot_mass[3] != 0.0F ||
      std::any_of(
          mixed_slot_values.begin() + 4, mixed_slot_values.end(),
          [](const float value) { return value != 0.0F; }) ||
      std::abs(
          mixed_slot_mass[0] + mixed_slot_mass[1] -
          mixed_episodic_mass[0]) >
          2.0e-6F ||
      mixed_regular_mass[1] != 1.0F ||
      mixed_episodic_mass[1] != 0.0F) {
    return fail("masked head episodic slot trace was not zero");
  }
  for (std::size_t slot = 0; slot < 2; ++slot) {
    for (std::size_t dimension = 0; dimension < 2; ++dimension) {
      const std::size_t slot_flat = slot * 2 + dimension;
      const std::size_t source_flat = slot * 4 + dimension;
      if (mixed_slot_values[slot_flat] !=
          rounded_bf16(masked_values[source_flat])) {
        return fail("selected head episodic slot value changed");
      }
    }
  }
  for (std::size_t dimension = 0; dimension < 2; ++dimension) {
    const float reconstructed =
        mixed_slot_mass[0] * mixed_slot_values[dimension] +
        mixed_slot_mass[1] * mixed_slot_values[2 + dimension];
    if (std::abs(
            reconstructed - mixed_episodic_component[dimension]) >
        2.0e-6F) {
      return fail("selected head slot trace did not reconstruct");
    }
  }
  if (mixed_metrics.episodic_read_events != 1 ||
      mixed_metrics.episodic_active_slots != 2 ||
      mixed_metrics.episodic_entries_read != 2 ||
      mixed_metrics.episodic_key_read_bytes != 8 ||
      mixed_metrics.episodic_value_read_bytes != 8 ||
      mixed_metrics.episodic_duplicate_older_entries_suppressed != 2) {
    return fail("mixed episodic head mask counters are inexact");
  }
  mixed_mask.reset();
  std::vector<float> mixed_reset_output(4);
  const auto mixed_reset = mixed_mask.step_episodic(
      std::span<const float>(masked_queries.data(), 4),
      std::span<const float>(masked_keys.data(), 4),
      std::span<const float>(masked_values.data(), 4), 0,
      engram::StreamingAttention::kNoEpisodicDirective,
      mixed_reset_output);
  if (mixed_reset.tokens_seen != 1 ||
      mixed_reset.episodic_slots_written != 1 ||
      mixed_reset.episodic_active_slots != 1 ||
      mixed_reset.episodic_read_events != 0) {
    return fail("mixed episodic head mask reset retained counters");
  }
  return 0;
}
