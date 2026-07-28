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

// Per-layer attention capacity policy for the additive layered-open ABI.
typedef struct engram_olmoe_attention_policy_v1 {
  size_t local_window;
  size_t older_candidates;
  size_t older_top_k;
  size_t sink_tokens;
} engram_olmoe_attention_policy_v1;

typedef struct engram_olmoe_token_metrics {
  uint64_t positions_processed;
  uint64_t attention_weight_bytes;
  uint64_t q7_scheduled_bytes;
  uint64_t q7_elapsed_ns;
  uint64_t attention_state_bytes;
  uint64_t elapsed_ns;
} engram_olmoe_token_metrics;

// Additive, versioned attention diagnostics. Keeping this separate preserves
// the original forward-metrics ABI for already-built runtime libraries.
typedef struct engram_olmoe_attention_metrics_v1 {
  uint64_t logical_read_bytes;
  uint64_t state_bytes;
  uint64_t scratch_bytes;
  uint64_t eviction_events;
  uint64_t older_candidate_entries_scored;
  uint64_t older_selected_entries;
  uint64_t sink_insertions;
  uint64_t heavy_hitter_updates;
} engram_olmoe_attention_metrics_v1;

void* engram_olmoe_token_open(const engram_olmoe_token_config* config,
                              char* error, size_t error_capacity);
// Opens a runtime with exactly one policy entry per configured layer. This is
// additive: engram_olmoe_token_open retains its original scalar behavior and
// ABI.
void* engram_olmoe_token_open_layered_v1(
    const engram_olmoe_token_config* config,
    const engram_olmoe_attention_policy_v1* policies, size_t policy_count,
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
int engram_olmoe_token_copy_attention_metrics_v1(
    const void* handle, engram_olmoe_attention_metrics_v1* metrics,
    char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
