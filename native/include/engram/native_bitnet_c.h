#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct engram_bitnet_metrics {
  uint64_t elapsed_ns;
  uint64_t gate_up_stream_bytes;
  uint64_t norm_stream_bytes;
  uint64_t down_stream_bytes;
  uint64_t layer_metadata_bytes;
  uint64_t scheduled_cache_line_bytes;
  uint64_t scratch_bytes;
  uint64_t rows;
  uint64_t threads;
} engram_bitnet_metrics;

void* engram_bitnet_open(const char* artifact_path, size_t thread_count,
                         char* error, size_t error_capacity);
void engram_bitnet_close(void* handle);
size_t engram_bitnet_layer_count(const void* handle);
size_t engram_bitnet_hidden_size(const void* handle);
size_t engram_bitnet_intermediate_size(const void* handle);
size_t engram_bitnet_thread_count(const void* handle);
size_t engram_bitnet_artifact_bytes(const void* handle);

int engram_bitnet_forward_bf16(void* handle, size_t layer,
                               const uint16_t* input, size_t rows,
                               uint16_t* output,
                               engram_bitnet_metrics* metrics, char* error,
                               size_t error_capacity);

int engram_bitnet_forward_oracle_bf16(
    void* handle, size_t layer, const uint16_t* input, size_t rows,
    size_t top_k, uint16_t* output, engram_bitnet_metrics* metrics,
    char* error, size_t error_capacity);

// Normalize the stage's post-attention state, execute one packed MLP layer,
// and feed its output back into the persistent native stage state.
int engram_bitnet_stage_semantic_bf16(
    void* handle, void* stage_handle, size_t layer,
    const uint16_t* norm_weight, float norm_epsilon, size_t rows,
    float semantic_scale, float episodic_scale,
    engram_bitnet_metrics* metrics, char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
