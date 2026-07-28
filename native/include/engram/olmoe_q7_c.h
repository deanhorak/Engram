#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct engram_olmoe_q7_metrics {
  uint64_t elapsed_ns;
  uint64_t router_stream_bytes;
  uint64_t selected_expert_stream_bytes;
  uint64_t scheduled_stream_bytes;
  uint64_t scratch_bytes;
  uint64_t rows;
  uint64_t threads;
  uint64_t selected_experts;
} engram_olmoe_q7_metrics;

void* engram_olmoe_q7_open(const char* artifact_path, size_t thread_count,
                           char* error, size_t error_capacity);
void engram_olmoe_q7_close(void* handle);
size_t engram_olmoe_q7_layer_count(const void* handle);
size_t engram_olmoe_q7_hidden_size(const void* handle);
size_t engram_olmoe_q7_intermediate_size(const void* handle);
size_t engram_olmoe_q7_expert_count(const void* handle);
size_t engram_olmoe_q7_top_k(const void* handle);
size_t engram_olmoe_q7_group_size(const void* handle);
size_t engram_olmoe_q7_artifact_bytes(const void* handle);

int engram_olmoe_q7_forward(
    void* handle, size_t layer, const float* input, size_t rows, float* output,
    uint32_t* selected_experts, engram_olmoe_q7_metrics* metrics, char* error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif
