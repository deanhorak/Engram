#include "engram/native_stage_c.h"

#include <cmath>
#include <cstdint>
#include <iostream>

namespace {

int fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}

}  // namespace

int main() {
  char error[256] = {};
  void* handle = engram_native_stage_create(1, 2, error, sizeof(error));
  if (handle == nullptr) return fail(error);

  const std::uint16_t embedding[] = {0x4040, 0x4080};  // 3, 4
  const std::uint16_t unit_weight[] = {0x3F80, 0x3F80};
  std::uint16_t normalized[2] = {};
  if (engram_native_stage_begin_bf16(handle, embedding, error,
                                     sizeof(error)) != 0 ||
      engram_native_stage_attention_input_bf16(
          handle, unit_weight, 1.0e-6F, normalized, error, sizeof(error)) !=
          0) {
    engram_native_stage_destroy(handle);
    return fail(error);
  }

  const std::uint16_t attention[] = {0x3F80, 0x0000};  // 1, 0
  const std::uint16_t semantic[] = {0x0000, 0x3F80};   // 0, 1
  if (engram_native_stage_accept_attention_bf16(
          handle, attention, error, sizeof(error)) != 0 ||
      engram_native_stage_semantic_input_bf16(
          handle, unit_weight, 1.0e-6F, normalized, error, sizeof(error)) !=
          0 ||
      engram_native_stage_accept_semantic_bf16(
          handle, semantic, 1.0F, 1.0F, error, sizeof(error)) != 0) {
    engram_native_stage_destroy(handle);
    return fail(error);
  }

  float state[2] = {};
  float rms[1] = {};
  if (engram_native_stage_copy_state_f32(handle, state, rms, error,
                                         sizeof(error)) != 0) {
    engram_native_stage_destroy(handle);
    return fail(error);
  }
  const float initial_rms = std::sqrt(12.5F);
  const float first = 3.0F / initial_rms + 1.0F / initial_rms;
  const float second = 4.0F / initial_rms + 1.0F / initial_rms;
  const float relative = std::sqrt((first * first + second * second) / 2.0F);
  const float normalization = std::sqrt(relative * relative + 1.0e-6F);
  if (std::abs(state[0] - first / normalization) > 2.0e-6F ||
      std::abs(state[1] - second / normalization) > 2.0e-6F ||
      std::abs(rms[0] - initial_rms * relative) > 2.0e-6F) {
    engram_native_stage_destroy(handle);
    return fail("native stage residual state mismatch");
  }

  // The controller entry point must be bit-for-bit equivalent to the exact
  // operator-residual path when its correction scale is zero.
  const engram_native_controller_weights_f32 incomplete_controller{
      .input_dim = 6,
      .state_dim = 2,
      .rank = 1,
      .adapter_rank = 1,
      .input_adapter_rank = 0,
      .input_down = nullptr,
      .recurrent_down = nullptr,
      .gate_up = nullptr,
      .bias = nullptr,
      .stage_embedding = nullptr,
      .adapter_down = nullptr,
      .adapter_up = nullptr,
      .input_adapter_down = nullptr,
      .input_adapter_up = nullptr,
      .step_scale = 0.0F,
  };
  void* controller_handle =
      engram_native_stage_create(1, 2, error, sizeof(error));
  if (controller_handle == nullptr) return fail(error);
  if (engram_native_stage_begin_bf16(controller_handle, embedding, error,
                                     sizeof(error)) != 0 ||
      engram_native_stage_accept_attention_bf16(
          controller_handle, attention, error, sizeof(error)) != 0 ||
      engram_native_stage_accept_controller_f32(
          controller_handle, semantic, 1.0F, 1.0F, &incomplete_controller,
          error, sizeof(error)) == 0) {
    // Null tensors are intentionally rejected even for a zero correction;
    // callers must authenticate and bind the complete controller artifact.
    engram_native_stage_destroy(controller_handle);
    return fail("native controller accepted incomplete tensors");
  }
  engram_native_stage_destroy(controller_handle);

  float input_down[6] = {};
  float recurrent_down[2] = {};
  float gate_up[4] = {};
  float bias[4] = {};
  float stage_embedding[2] = {1.0F, -1.0F};
  float adapter_down[2] = {};
  float adapter_up[2] = {};
  const engram_native_controller_weights_f32 controller{
      .input_dim = 6,
      .state_dim = 2,
      .rank = 1,
      .adapter_rank = 1,
      .input_adapter_rank = 0,
      .input_down = input_down,
      .recurrent_down = recurrent_down,
      .gate_up = gate_up,
      .bias = bias,
      .stage_embedding = stage_embedding,
      .adapter_down = adapter_down,
      .adapter_up = adapter_up,
      .input_adapter_down = nullptr,
      .input_adapter_up = nullptr,
      .step_scale = 0.25F,
  };
  auto zero_controller = controller;
  zero_controller.step_scale = 0.0F;
  void* zero_handle = engram_native_stage_create(1, 2, error, sizeof(error));
  if (zero_handle == nullptr) return fail(error);
  if (engram_native_stage_begin_bf16(zero_handle, embedding, error,
                                     sizeof(error)) != 0 ||
      engram_native_stage_accept_attention_bf16(
          zero_handle, attention, error, sizeof(error)) != 0 ||
      engram_native_stage_accept_controller_f32(
          zero_handle, semantic, 1.0F, 1.0F, &zero_controller, error,
          sizeof(error)) != 0 ||
      engram_native_stage_copy_state_f32(zero_handle, state, rms, error,
                                         sizeof(error)) != 0 ||
      std::abs(state[0] - first / normalization) > 2.0e-6F ||
      std::abs(state[1] - second / normalization) > 2.0e-6F ||
      std::abs(rms[0] - initial_rms * relative) > 2.0e-6F) {
    engram_native_stage_destroy(zero_handle);
    return fail("zero native controller did not preserve exact residual parity");
  }
  engram_native_stage_destroy(zero_handle);
  void* nonzero_handle =
      engram_native_stage_create(1, 2, error, sizeof(error));
  if (nonzero_handle == nullptr) return fail(error);
  if (engram_native_stage_begin_bf16(nonzero_handle, embedding, error,
                                     sizeof(error)) != 0 ||
      engram_native_stage_accept_attention_bf16(
          nonzero_handle, attention, error, sizeof(error)) != 0 ||
      engram_native_stage_accept_controller_f32(
          nonzero_handle, semantic, 1.0F, 1.0F, &controller, error,
          sizeof(error)) != 0 ||
      engram_native_stage_copy_state_f32(nonzero_handle, state, rms, error,
                                         sizeof(error)) != 0) {
    engram_native_stage_destroy(nonzero_handle);
    return fail(error);
  }
  const float delta = 0.125F * std::tanh(1.0F);
  const float corrected0 = first + delta;
  const float corrected1 = second - delta;
  const float corrected_rms =
      std::sqrt((corrected0 * corrected0 + corrected1 * corrected1) / 2.0F);
  const float corrected_normalization =
      std::sqrt(corrected_rms * corrected_rms + 1.0e-6F);
  if (std::abs(state[0] - corrected0 / corrected_normalization) > 2.0e-6F ||
      std::abs(state[1] - corrected1 / corrected_normalization) > 2.0e-6F ||
      std::abs(rms[0] - initial_rms * corrected_rms) > 2.0e-6F) {
    std::cerr << "got=" << state[0] << "," << state[1] << " rms=" << rms[0]
              << " expected=" << corrected0 / corrected_normalization << ","
              << corrected1 / corrected_normalization << " rms="
              << initial_rms * corrected_rms << " raw=" << first << "," << second
              << "\n";
    engram_native_stage_destroy(nonzero_handle);
    return fail("native recurrent controller transition mismatch");
  }
  engram_native_stage_destroy(nonzero_handle);

  // Reject semantic output before an attention output opens that phase.
  if (engram_native_stage_accept_semantic_bf16(
          handle, semantic, 1.0F, 1.0F, error, sizeof(error)) == 0) {
    engram_native_stage_destroy(handle);
    return fail("native stage accepted an out-of-order semantic result");
  }
  engram_native_stage_destroy(handle);
  return 0;
}
