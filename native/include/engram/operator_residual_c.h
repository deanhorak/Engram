#pragma once

#include <cstddef>

extern "C" {

int engram_operator_residual_step_f32(
    const float* state, const float* semantic, const float* episodic,
    std::size_t vectors, std::size_t width, float semantic_scale,
    float episodic_scale, float* output, float* relative_rms);

}
