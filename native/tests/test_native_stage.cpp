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

  // Reject semantic output before an attention output opens that phase.
  if (engram_native_stage_accept_semantic_bf16(
          handle, semantic, 1.0F, 1.0F, error, sizeof(error)) == 0) {
    engram_native_stage_destroy(handle);
    return fail("native stage accepted an out-of-order semantic result");
  }
  engram_native_stage_destroy(handle);
  return 0;
}
