#pragma once

#include <cstddef>
#include <cstdint>

#include "engram/native_attention_stage_c.h"
#include "engram/native_bitnet_c.h"

extern "C" {

typedef struct engram_native_stage_descriptor {
  void* projection_handle;
  std::size_t query_projection;
  std::size_t key_projection;
  std::size_t value_projection;
  std::size_t output_projection;
  void* const* attention_handles;
  const std::uint16_t* input_norm_weight;
  float input_norm_epsilon;
  const std::uint16_t* attention_norm_weight;
  float attention_norm_epsilon;
  const std::uint16_t* semantic_norm_weight;
  float semantic_norm_epsilon;
  float semantic_scale;
  float episodic_scale;
  std::size_t semantic_layer;
} engram_native_stage_descriptor;

int engram_native_run_stages_bf16(
    void* stage_handle, void* semantic_handle,
    const engram_native_stage_descriptor* stages, std::size_t stage_count,
    std::size_t batch, std::size_t length, std::size_t width,
    std::size_t query_heads, std::size_t key_value_heads,
    std::size_t head_dimension, const std::int64_t* positions,
    std::size_t position_rows, float rope_theta,
    engram_native_attention_stage_metrics* attention_metrics,
    engram_bitnet_metrics* semantic_metrics, char* error,
    std::size_t error_capacity);

}
