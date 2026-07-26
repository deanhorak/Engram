#pragma once

#include <cstddef>
#include <cstdint>

#include "engram/streaming_attention_c.h"

extern "C" {

typedef struct engram_native_attention_stage_metrics {
  std::uint64_t qkv_projection_ns;
  std::uint64_t rope_ns;
  std::uint64_t native_attention_ns;
  std::uint64_t output_projection_ns;
  std::uint64_t packed_weight_bytes;
  std::uint64_t projection_scratch_bytes;
  engram_streaming_attention_metrics attention;
} engram_native_attention_stage_metrics;

// Execute normalized Q/K/V projections, RoPE, persistent bounded attention,
// attention sub-normalization, output projection, and residual insertion.
int engram_native_stage_attention_bf16(
    void* stage_handle, void* projection_handle, std::size_t query_projection,
    std::size_t key_projection, std::size_t value_projection,
    std::size_t output_projection, void* const* attention_handles,
    std::size_t batch, std::size_t length, std::size_t width,
    std::size_t query_heads, std::size_t key_value_heads,
    std::size_t head_dimension, const std::int64_t* positions,
    std::size_t position_rows, float rope_theta,
    const std::uint16_t* input_norm_weight, float input_norm_epsilon,
    const std::uint16_t* attention_norm_weight, float attention_norm_epsilon,
    engram_native_attention_stage_metrics* metrics, char* error,
    std::size_t error_capacity);

}
