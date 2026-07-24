#include "engram/streaming_attention_c.h"

#include "engram/streaming_attention.h"

#include <algorithm>
#include <cstring>
#include <exception>
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
    if (metrics != nullptr) {
      metrics->tokens_seen = result.tokens_seen;
      metrics->local_entries = result.local_entries;
      metrics->active_older_entries = result.active_older_entries;
      metrics->candidate_key_bytes = result.candidate_key_bytes;
      metrics->selected_value_bytes = result.selected_value_bytes;
      metrics->local_kv_bytes = result.local_kv_bytes;
      metrics->state_bytes = result.state_bytes;
      metrics->scratch_bytes = result.scratch_bytes;
    }
    return 0;
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return 1;
  }
}

}  // extern "C"
