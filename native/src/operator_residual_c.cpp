#include "engram/operator_residual_c.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

extern "C" int engram_operator_residual_step_f32(
    const float* state, const float* semantic, const float* episodic,
    const std::size_t vectors, const std::size_t width,
    const float semantic_scale, const float episodic_scale, float* output,
    float* relative_rms) {
  if (state == nullptr || semantic == nullptr || episodic == nullptr ||
      output == nullptr || relative_rms == nullptr || vectors == 0 ||
      width == 0 || !std::isfinite(semantic_scale) ||
      !std::isfinite(episodic_scale)) {
    return 1;
  }
  for (std::size_t row = 0; row < vectors; ++row) {
    const std::size_t offset = row * width;
    double squared = 0.0;
    for (std::size_t column = 0; column < width; ++column) {
      const std::size_t index = offset + column;
      const float value =
          state[index] + semantic_scale * semantic[index] +
          episodic_scale * episodic[index];
      if (!std::isfinite(value)) {
        return 2;
      }
      output[index] = value;
      squared += static_cast<double>(value) * static_cast<double>(value);
    }
    const float raw_rms = static_cast<float>(
        std::sqrt(squared / static_cast<double>(width)));
    const float normalization_rms = static_cast<float>(
        std::sqrt(squared / static_cast<double>(width) + 1.0e-6));
    relative_rms[row] = std::max(raw_rms, 1.0e-6F);
    for (std::size_t column = 0; column < width; ++column) {
      output[offset + column] /= normalization_rms;
    }
  }
  return 0;
}
