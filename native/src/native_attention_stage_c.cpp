#include "engram/native_attention_stage_c.h"

#include "engram/native_shell_c.h"
#include "engram/native_stage_c.h"
#include "engram/ternary_projection_c.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

float bf16_to_float(const std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

void error_text(char* output, const std::size_t capacity,
                const char* text) noexcept {
  if (output == nullptr || capacity == 0) return;
  const std::size_t length = std::min(capacity - 1, std::strlen(text));
  std::memcpy(output, text, length);
  output[length] = '\0';
}

std::uint64_t elapsed_ns(const Clock::time_point started) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now() - started)
          .count());
}

}  // namespace

extern "C" int engram_native_stage_attention_bf16(
    void* stage_handle, void* projection_handle,
    const std::size_t query_projection, const std::size_t key_projection,
    const std::size_t value_projection, const std::size_t output_projection,
    void* const* attention_handles, const std::size_t batch,
    const std::size_t length, const std::size_t width,
    const std::size_t query_heads, const std::size_t key_value_heads,
    const std::size_t head_dimension, const std::int64_t* positions,
    const std::size_t position_rows, const float rope_theta,
    const std::uint16_t* input_norm_weight, const float input_norm_epsilon,
    const std::uint16_t* attention_norm_weight,
    const float attention_norm_epsilon,
    engram_native_attention_stage_metrics* metrics, char* error,
    const std::size_t error_capacity) {
  try {
    if (stage_handle == nullptr || projection_handle == nullptr ||
        attention_handles == nullptr || positions == nullptr ||
        input_norm_weight == nullptr || attention_norm_weight == nullptr ||
        batch == 0 || length == 0 || width == 0 || query_heads == 0 ||
        key_value_heads == 0 || head_dimension == 0 ||
        query_heads * head_dimension != width ||
        (position_rows != 1 && position_rows != batch) ||
        batch > std::numeric_limits<std::size_t>::max() / length ||
        batch * length > std::numeric_limits<std::size_t>::max() / width) {
      throw std::invalid_argument(
          "native attention stage received invalid dimensions or storage");
    }
    const std::size_t rows = batch * length;
    const std::size_t hidden_elements = rows * width;
    const std::size_t kv_width = key_value_heads * head_dimension;
    const std::size_t kv_elements = rows * kv_width;
    std::vector<std::uint16_t> normalized(hidden_elements);
    std::vector<std::uint16_t> query(hidden_elements);
    std::vector<std::uint16_t> key(kv_elements);
    std::vector<std::uint16_t> value(kv_elements);
    std::vector<std::uint16_t> query_rope(hidden_elements);
    std::vector<std::uint16_t> key_rope(kv_elements);
    std::vector<float> attention_output(hidden_elements);
    std::vector<std::uint16_t> projected_input(hidden_elements);
    std::vector<std::uint16_t> projected_output(hidden_elements);
    engram_native_attention_stage_metrics aggregate{};

    if (engram_native_stage_attention_input_bf16(
            stage_handle, input_norm_weight, input_norm_epsilon,
            normalized.data(), error, error_capacity) != 0) {
      return 1;
    }
    const auto project = [&](const std::size_t projection,
                             const std::uint16_t* input,
                             std::uint16_t* output,
                             engram_ternary_projection_metrics* result) {
      return engram_ternary_projection_forward_bf16(
          projection_handle, projection, input, rows, output, result, error,
          error_capacity);
    };
    engram_ternary_projection_metrics q_metrics{};
    engram_ternary_projection_metrics k_metrics{};
    engram_ternary_projection_metrics v_metrics{};
    if (project(query_projection, normalized.data(), query.data(), &q_metrics) ||
        project(key_projection, normalized.data(), key.data(), &k_metrics) ||
        project(value_projection, normalized.data(), value.data(), &v_metrics)) {
      return 1;
    }
    aggregate.qkv_projection_ns =
        q_metrics.elapsed_ns + k_metrics.elapsed_ns + v_metrics.elapsed_ns;
    aggregate.packed_weight_bytes =
        q_metrics.packed_weight_bytes + k_metrics.packed_weight_bytes +
        v_metrics.packed_weight_bytes;
    aggregate.projection_scratch_bytes =
        std::max({q_metrics.scratch_bytes, k_metrics.scratch_bytes,
                  v_metrics.scratch_bytes});

    // Projection output is [batch, length, heads, dimension]; RoPE consumes
    // [batch, heads, length, dimension].
    for (std::size_t row = 0; row < batch; ++row) {
      for (std::size_t token = 0; token < length; ++token) {
        for (std::size_t head = 0; head < query_heads; ++head) {
          for (std::size_t column = 0; column < head_dimension; ++column) {
            query_rope[((row * query_heads + head) * length + token) *
                           head_dimension +
                       column] =
                query[((row * length + token) * query_heads + head) *
                          head_dimension +
                      column];
          }
        }
        for (std::size_t head = 0; head < key_value_heads; ++head) {
          for (std::size_t column = 0; column < head_dimension; ++column) {
            key_rope[((row * key_value_heads + head) * length + token) *
                         head_dimension +
                     column] =
                key[((row * length + token) * key_value_heads + head) *
                        head_dimension +
                    column];
          }
        }
      }
    }
    const auto rope_started = Clock::now();
    if (engram_rope_bf16(
            query_rope.data(), query_heads, key_rope.data(), key_value_heads,
            batch, length, head_dimension, positions, position_rows,
            rope_theta) != 0) {
      throw std::runtime_error("native attention stage RoPE failed");
    }
    aggregate.rope_ns = elapsed_ns(rope_started);

    const auto attention_started = Clock::now();
    for (std::size_t row = 0; row < batch; ++row) {
      if (attention_handles[row] == nullptr) {
        throw std::invalid_argument("native attention stage cache is null");
      }
      std::vector<float> query_stream(length * width);
      std::vector<float> key_stream(length * kv_width);
      std::vector<float> value_stream(length * kv_width);
      for (std::size_t token = 0; token < length; ++token) {
        for (std::size_t head = 0; head < query_heads; ++head) {
          for (std::size_t column = 0; column < head_dimension; ++column) {
            query_stream[(token * query_heads + head) * head_dimension +
                         column] = bf16_to_float(
                query_rope[((row * query_heads + head) * length + token) *
                               head_dimension +
                           column]);
          }
        }
        for (std::size_t head = 0; head < key_value_heads; ++head) {
          for (std::size_t column = 0; column < head_dimension; ++column) {
            const std::size_t stream_index =
                (token * key_value_heads + head) * head_dimension + column;
            key_stream[stream_index] = bf16_to_float(
                key_rope[((row * key_value_heads + head) * length + token) *
                             head_dimension +
                         column]);
            value_stream[stream_index] = bf16_to_float(
                value[((row * length + token) * key_value_heads + head) *
                          head_dimension +
                      column]);
          }
        }
      }
      engram_streaming_attention_metrics current{};
      if (engram_streaming_attention_stream_f32(
              attention_handles[row], query_stream.data(), key_stream.data(),
              value_stream.data(), length,
              attention_output.data() + row * length * width, &current, error,
              error_capacity) != 0) {
        return 1;
      }
      aggregate.attention.tokens_seen = current.tokens_seen;
      aggregate.attention.local_entries += current.local_entries;
      aggregate.attention.active_older_entries +=
          current.active_older_entries;
      aggregate.attention.candidate_key_bytes += current.candidate_key_bytes;
      aggregate.attention.selected_value_bytes +=
          current.selected_value_bytes;
      aggregate.attention.local_kv_bytes += current.local_kv_bytes;
      aggregate.attention.eviction_events += current.eviction_events;
      aggregate.attention.older_candidate_entries_scored +=
          current.older_candidate_entries_scored;
      aggregate.attention.older_selected_entries +=
          current.older_selected_entries;
      aggregate.attention.sink_insertions += current.sink_insertions;
      aggregate.attention.heavy_hitter_updates +=
          current.heavy_hitter_updates;
      aggregate.attention.state_bytes += current.state_bytes;
      aggregate.attention.scratch_bytes += current.scratch_bytes;
    }
    aggregate.native_attention_ns = elapsed_ns(attention_started);

    if (engram_rms_norm_f32_to_bf16(
            attention_output.data(), attention_norm_weight, rows, width,
            attention_norm_epsilon, projected_input.data()) != 0) {
      throw std::runtime_error(
          "native attention stage sub-normalization failed");
    }
    engram_ternary_projection_metrics o_metrics{};
    if (project(output_projection, projected_input.data(),
                projected_output.data(), &o_metrics)) {
      return 1;
    }
    aggregate.output_projection_ns = o_metrics.elapsed_ns;
    aggregate.packed_weight_bytes += o_metrics.packed_weight_bytes;
    aggregate.projection_scratch_bytes =
        std::max(aggregate.projection_scratch_bytes, o_metrics.scratch_bytes);
    if (engram_native_stage_accept_attention_bf16(
            stage_handle, projected_output.data(), error, error_capacity) !=
        0) {
      return 1;
    }
    if (metrics != nullptr) *metrics = aggregate;
    return 0;
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return 1;
  }
}
