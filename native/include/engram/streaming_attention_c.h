#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct engram_streaming_attention_config {
  size_t query_heads;
  size_t key_value_heads;
  size_t head_dimension;
  size_t local_window;
  size_t older_candidates;
  size_t older_top_k;
  size_t sink_tokens;
  float scale;
} engram_streaming_attention_config;

typedef struct engram_streaming_attention_metrics {
  uint64_t tokens_seen;
  uint64_t local_entries;
  uint64_t active_older_entries;
  uint64_t candidate_key_bytes;
  uint64_t selected_value_bytes;
  uint64_t local_kv_bytes;
  uint64_t state_bytes;
  uint64_t scratch_bytes;
} engram_streaming_attention_metrics;

void* engram_streaming_attention_create(
    const engram_streaming_attention_config* config, char* error,
    size_t error_capacity);
void engram_streaming_attention_destroy(void* handle);
void engram_streaming_attention_reset(void* handle);

int engram_streaming_attention_step_f32(
    void* handle, const float* query, const float* key, const float* value,
    float* output, engram_streaming_attention_metrics* metrics, char* error,
    size_t error_capacity);

// Process position-major [length, heads, head_dimension] streams in one call.
// Byte counters in metrics are accumulated across the complete stream; state
// counters describe the final state.
int engram_streaming_attention_stream_f32(
    void* handle, const float* queries, const float* keys, const float* values,
    size_t length, float* outputs,
    engram_streaming_attention_metrics* metrics, char* error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif
