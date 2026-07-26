#include "engram/native_stage_runner_c.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <stdexcept>

namespace {

void error_text(char* output, const std::size_t capacity,
                const char* text) noexcept {
  if (output == nullptr || capacity == 0) return;
  const std::size_t length = std::min(capacity - 1, std::strlen(text));
  std::memcpy(output, text, length);
  output[length] = '\0';
}

}  // namespace

extern "C" int engram_native_run_stages_bf16(
    void* stage_handle, void* semantic_handle,
    const engram_native_stage_descriptor* stages,
    const std::size_t stage_count, const std::size_t batch,
    const std::size_t length, const std::size_t width,
    const std::size_t query_heads, const std::size_t key_value_heads,
    const std::size_t head_dimension, const std::int64_t* positions,
    const std::size_t position_rows, const float rope_theta,
    engram_native_attention_stage_metrics* attention_metrics,
    engram_bitnet_metrics* semantic_metrics, char* error,
    const std::size_t error_capacity) {
  try {
    if (stage_handle == nullptr || semantic_handle == nullptr ||
        stages == nullptr || stage_count == 0) {
      throw std::invalid_argument("native stage runner received null storage");
    }
    for (std::size_t stage = 0; stage < stage_count; ++stage) {
      const auto& descriptor = stages[stage];
      if (engram_native_stage_attention_bf16(
              stage_handle, descriptor.projection_handle,
              descriptor.query_projection, descriptor.key_projection,
              descriptor.value_projection, descriptor.output_projection,
              descriptor.attention_handles, batch, length, width, query_heads,
              key_value_heads, head_dimension, positions, position_rows,
              rope_theta, descriptor.input_norm_weight,
              descriptor.input_norm_epsilon,
              descriptor.attention_norm_weight,
              descriptor.attention_norm_epsilon,
              attention_metrics == nullptr ? nullptr
                                           : attention_metrics + stage,
              error, error_capacity) != 0) {
        return 1;
      }
      if (engram_bitnet_stage_semantic_bf16(
              semantic_handle, stage_handle, descriptor.semantic_layer,
              descriptor.semantic_norm_weight,
              descriptor.semantic_norm_epsilon, batch * length,
              descriptor.semantic_scale, descriptor.episodic_scale,
              semantic_metrics == nullptr ? nullptr : semantic_metrics + stage,
              error, error_capacity) != 0) {
        return 1;
      }
    }
    return 0;
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return 1;
  }
}
