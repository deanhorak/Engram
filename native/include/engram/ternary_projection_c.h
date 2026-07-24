#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct engram_ternary_projection_metrics {
  uint64_t elapsed_ns;
  uint64_t packed_weight_bytes;
  uint64_t scratch_bytes;
  uint64_t rows;
} engram_ternary_projection_metrics;

void* engram_ternary_projection_create(size_t threads, char* error,
                                       size_t error_capacity);
void engram_ternary_projection_destroy(void* handle);
int engram_ternary_projection_add(void* handle, const uint8_t* packed,
                                  size_t packed_bytes, size_t input_features,
                                  size_t output_features, float weight_scale,
                                  size_t* projection, char* error,
                                  size_t error_capacity);
int engram_ternary_projection_forward_bf16(
    void* handle, size_t projection, const uint16_t* input, size_t rows,
    uint16_t* output, engram_ternary_projection_metrics* metrics, char* error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif
