#pragma once

#include <cstddef>

namespace engram {

// Scalar correctness kernel. SIMD work deliberately follows feasibility gates.
void swiglu_scalar(const float* hidden, const float* gate, const float* up,
                   const float* down, std::size_t hidden_size,
                   std::size_t intermediate_size, float* output);

}  // namespace engram
