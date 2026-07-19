#pragma once

#include <cstddef>

namespace engram {

enum class VectorKernelKind { scalar, avx2 };

[[nodiscard]] float dot_product_scalar(const float* left, const float* right,
                                       std::size_t size) noexcept;

// Safe public entry point: executes the isolated AVX2 implementation only when
// both compiled and usable on this process, otherwise executes the scalar path.
[[nodiscard]] float dot_product_avx2(const float* left, const float* right,
                                    std::size_t size) noexcept;

// Cached runtime-dispatched hot-path entry point.
[[nodiscard]] float dot_product(const float* left, const float* right,
                               std::size_t size) noexcept;
[[nodiscard]] bool avx2_dot_kernel_compiled() noexcept;
[[nodiscard]] VectorKernelKind selected_dot_kernel() noexcept;
[[nodiscard]] const char* selected_dot_kernel_name() noexcept;

}  // namespace engram
