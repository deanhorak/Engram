#include "engram/streaming_attention_c.h"

#include "engram/streaming_attention.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <span>
#include <stdexcept>

namespace {

void write_error(char* error, const std::size_t capacity,
                 const char* message) noexcept {
  if (error == nullptr || capacity == 0) return;
  const std::size_t length = std::min(capacity - 1, std::strlen(message));
  std::memcpy(error, message, length);
  error[length] = '\0';
}

void clear_error(char* error, const std::size_t capacity) noexcept {
  if (error != nullptr && capacity != 0) error[0] = '\0';
}

void copy_metrics(const engram::StreamingAttentionMetrics& source,
                  engram_streaming_attention_metrics* target) noexcept {
  if (target == nullptr) return;
  target->tokens_seen = source.tokens_seen;
  target->local_entries = source.local_entries;
  target->active_older_entries = source.active_older_entries;
  target->candidate_key_bytes = source.candidate_key_bytes;
  target->selected_value_bytes = source.selected_value_bytes;
  target->local_kv_bytes = source.local_kv_bytes;
  target->state_bytes = source.state_bytes;
  target->scratch_bytes = source.scratch_bytes;
}

}  // namespace

extern "C" {

void* engram_streaming_attention_create(
    const engram_streaming_attention_config* config, char* error,
    const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  try {
    if (config == nullptr) {
      throw std::invalid_argument("streaming attention config is null");
    }
    return new engram::StreamingAttention(engram::StreamingAttentionConfig{
        .query_heads = config->query_heads,
        .key_value_heads = config->key_value_heads,
        .head_dimension = config->head_dimension,
        .local_window = config->local_window,
        .older_candidates = config->older_candidates,
        .older_top_k = config->older_top_k,
        .sink_tokens = config->sink_tokens,
        .scale = config->scale,
    });
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return nullptr;
  }
}

void engram_streaming_attention_destroy(void* handle) {
  delete static_cast<engram::StreamingAttention*>(handle);
}

void engram_streaming_attention_reset(void* handle) {
  if (handle != nullptr) {
    static_cast<engram::StreamingAttention*>(handle)->reset();
  }
}

int engram_streaming_attention_step_f32(
    void* handle, const float* query, const float* key, const float* value,
    float* output, engram_streaming_attention_metrics* metrics, char* error,
    const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  try {
    auto* attention = static_cast<engram::StreamingAttention*>(handle);
    if (attention == nullptr || query == nullptr || key == nullptr ||
        value == nullptr || output == nullptr) {
      throw std::invalid_argument("streaming attention step received null storage");
    }
    const auto& config = attention->config();
    const std::size_t query_elements =
        config.query_heads * config.head_dimension;
    const std::size_t kv_elements =
        config.key_value_heads * config.head_dimension;
    const auto result = attention->step(
        std::span<const float>(query, query_elements),
        std::span<const float>(key, kv_elements),
        std::span<const float>(value, kv_elements),
        std::span<float>(output, query_elements));
    copy_metrics(result, metrics);
    return 0;
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return 1;
  }
}

int engram_streaming_attention_stream_f32(
    void* handle, const float* queries, const float* keys, const float* values,
    const std::size_t length, float* outputs,
    engram_streaming_attention_metrics* metrics, char* error,
    const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  try {
    auto* attention = static_cast<engram::StreamingAttention*>(handle);
    if (attention == nullptr || queries == nullptr || keys == nullptr ||
        values == nullptr || outputs == nullptr) {
      throw std::invalid_argument(
          "streaming attention stream received null storage");
    }
    if (length == 0) {
      throw std::invalid_argument(
          "streaming attention stream length must be positive");
    }
    const auto& config = attention->config();
    const std::size_t query_elements =
        config.query_heads * config.head_dimension;
    const std::size_t kv_elements =
        config.key_value_heads * config.head_dimension;
    if (length > std::numeric_limits<std::size_t>::max() / query_elements ||
        length > std::numeric_limits<std::size_t>::max() / kv_elements) {
      throw std::overflow_error("streaming attention stream size overflow");
    }
    engram::StreamingAttentionMetrics aggregate{};
    for (std::size_t position = 0; position < length; ++position) {
      const auto result = attention->step(
          std::span<const float>(
              queries + position * query_elements, query_elements),
          std::span<const float>(keys + position * kv_elements, kv_elements),
          std::span<const float>(values + position * kv_elements, kv_elements),
          std::span<float>(
              outputs + position * query_elements, query_elements));
      aggregate.tokens_seen = result.tokens_seen;
      aggregate.local_entries = result.local_entries;
      aggregate.active_older_entries = result.active_older_entries;
      aggregate.candidate_key_bytes += result.candidate_key_bytes;
      aggregate.selected_value_bytes += result.selected_value_bytes;
      aggregate.local_kv_bytes += result.local_kv_bytes;
      aggregate.state_bytes = result.state_bytes;
      aggregate.scratch_bytes = result.scratch_bytes;
    }
    copy_metrics(aggregate, metrics);
    return 0;
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return 1;
  }
}

}  // extern "C"
