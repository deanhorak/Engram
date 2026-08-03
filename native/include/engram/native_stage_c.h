#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

void* engram_native_stage_create(std::size_t vectors, std::size_t width,
                                 char* error, std::size_t error_capacity);
void engram_native_stage_destroy(void* handle);

int engram_native_stage_begin_bf16(void* handle,
                                   const std::uint16_t* embedding, char* error,
                                   std::size_t error_capacity);
int engram_native_stage_attention_input_bf16(
    void* handle, const std::uint16_t* weight, float epsilon,
    std::uint16_t* output, char* error, std::size_t error_capacity);
int engram_native_stage_accept_attention_bf16(
    void* handle, const std::uint16_t* attention_output, char* error,
    std::size_t error_capacity);
int engram_native_stage_semantic_input_bf16(
    void* handle, const std::uint16_t* weight, float epsilon,
    std::uint16_t* output, char* error, std::size_t error_capacity);
int engram_native_stage_accept_semantic_bf16(
    void* handle, const std::uint16_t* semantic_output, float semantic_scale,
    float episodic_scale, char* error, std::size_t error_capacity);

// Apply the factorized recurrent-controller transition to the normalized
// operator residual. All tensors are row-major float32 views. The stage
// embedding and adapters point at the current stage slice; optional input
// adapters may be null when their rank is zero. This evaluator-facing entry
// point is deliberately separate from the exact operator-residual path.
struct engram_native_controller_weights_f32 {
  std::size_t input_dim;
  std::size_t state_dim;
  std::size_t rank;
  std::size_t adapter_rank;
  std::size_t input_adapter_rank;
  const float* input_down;
  const float* recurrent_down;
  const float* gate_up;
  const float* bias;
  const float* stage_embedding;
  const float* adapter_down;
  const float* adapter_up;
  const float* input_adapter_down;
  const float* input_adapter_up;
  float step_scale;
};

int engram_native_stage_accept_controller_f32(
    void* handle, const std::uint16_t* semantic_output, float semantic_scale,
    float episodic_scale,
    const engram_native_controller_weights_f32* controller, char* error,
    std::size_t error_capacity);
int engram_native_stage_final_norm_bf16(
    void* handle, const std::uint16_t* weight, float epsilon,
    std::uint16_t* output, char* error, std::size_t error_capacity);
int engram_native_stage_copy_state_f32(void* handle, float* normalized_state,
                                       float* residual_rms, char* error,
                                       std::size_t error_capacity);

}
