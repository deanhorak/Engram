#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct engram_olmoe_token_config {
  const char* non_mlp_safetensors;
  const char* q7_artifact;
  size_t layers;
  size_t hidden_size;
  size_t query_heads;
  size_t key_value_heads;
  size_t head_dimension;
  size_t threads;
  size_t local_window;
  size_t older_candidates;
  size_t older_top_k;
  size_t sink_tokens;
  float rms_norm_epsilon;
  float rope_theta;
} engram_olmoe_token_config;

typedef struct engram_olmoe_token_metrics {
  uint64_t positions_processed;
  uint64_t attention_weight_bytes;
  uint64_t q7_scheduled_bytes;
  uint64_t q7_elapsed_ns;
  uint64_t attention_state_bytes;
  uint64_t elapsed_ns;
} engram_olmoe_token_metrics;

void* engram_olmoe_token_open(const engram_olmoe_token_config* config,
                              char* error, size_t error_capacity);
void engram_olmoe_token_close(void* handle);
void engram_olmoe_token_reset(void* handle);
size_t engram_olmoe_token_vocabulary_size(const void* handle);
size_t engram_olmoe_token_position(const void* handle);
int engram_olmoe_token_forward(void* handle, const int64_t* token_ids,
                               size_t length, int64_t* next_token,
                               engram_olmoe_token_metrics* metrics,
                               char* error, size_t error_capacity);
int engram_olmoe_token_copy_last_diagnostics(
    const void* handle, float* final_state, size_t final_state_count,
    float* vocabulary_scores, size_t vocabulary_score_count, char* error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif
