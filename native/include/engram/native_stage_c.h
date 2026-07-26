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
int engram_native_stage_final_norm_bf16(
    void* handle, const std::uint16_t* weight, float epsilon,
    std::uint16_t* output, char* error, std::size_t error_capacity);
int engram_native_stage_copy_state_f32(void* handle, float* normalized_state,
                                       float* residual_rms, char* error,
                                       std::size_t error_capacity);

}
