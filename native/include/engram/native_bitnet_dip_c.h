#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct engram_bitnet_dip_metrics {
  uint64_t elapsed_ns;
  uint64_t coordinate_stream_bytes;
  uint64_t candidate_completion_bytes;
  uint64_t gain_stream_bytes;
  uint64_t down_norm_stream_bytes;
  uint64_t selected_down_stream_bytes;
  uint64_t layer_metadata_bytes;
  uint64_t scheduled_cache_line_bytes;
  uint64_t scratch_bytes;
  uint64_t rows;
  uint64_t threads;
  uint64_t input_coordinates;
  uint64_t candidate_count;
  uint64_t selected_count_total;
  uint64_t selected_count_min;
  uint64_t selected_count_max;
} engram_bitnet_dip_metrics;

void* engram_bitnet_dip_open(const char* record_artifact_path,
                             const char* coordinate_index_path,
                             size_t thread_count, char* error,
                             size_t error_capacity);
void engram_bitnet_dip_close(void* handle);
size_t engram_bitnet_dip_layer_count(const void* handle);
size_t engram_bitnet_dip_hidden_size(const void* handle);
size_t engram_bitnet_dip_intermediate_size(const void* handle);
size_t engram_bitnet_dip_thread_count(const void* handle);
size_t engram_bitnet_dip_record_artifact_bytes(const void* handle);
size_t engram_bitnet_dip_coordinate_index_bytes(const void* handle);

int engram_bitnet_dip_layer_policy(
    const void* handle, size_t layer, size_t* input_coordinates,
    size_t* candidate_count, size_t* minimum_top_k, size_t* maximum_top_k,
    float* energy_target, size_t* rms_audit_count,
    uint32_t* rms_estimator, uint32_t* rms_audit_strategy, char* error,
    size_t error_capacity);

int engram_bitnet_dip_forward_bf16(
    void* handle, size_t layer, const uint16_t* input, size_t rows,
    uint16_t* output, uint32_t* selected_counts,
    engram_bitnet_dip_metrics* metrics, char* error, size_t error_capacity);

int engram_bitnet_dip_forward_debug_bf16(
    void* handle, size_t layer, const uint16_t* input, size_t rows,
    uint16_t* output, uint32_t* selected_counts,
    uint32_t* input_coordinate_ids, uint32_t* candidate_ids,
    uint32_t* selected_record_ids, engram_bitnet_dip_metrics* metrics,
    char* error, size_t error_capacity);

int engram_bitnet_dip_teacher_top_k_bf16(
    void* handle, size_t layer, const uint16_t* input, size_t rows,
    size_t top_k, uint32_t* teacher_record_ids, char* error,
    size_t error_capacity);

int engram_bitnet_dip_teacher_top_k_positive_bf16(
    void* handle, size_t layer, const uint16_t* input, size_t rows,
    size_t top_k, uint32_t* teacher_record_ids,
    uint32_t* positive_utility_counts, char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
