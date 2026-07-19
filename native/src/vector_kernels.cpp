#include "engram/vector_kernels.h"

#include "engram/cpu_features.h"

#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define ENGRAM_COMPILED_AVX2_DOT 1
#define ENGRAM_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define ENGRAM_COMPILED_AVX2_DOT 0
#define ENGRAM_TARGET_AVX2
#endif

namespace engram {
namespace {

using DotKernel = float (*)(const float*, const float*, std::size_t) noexcept;

#if ENGRAM_COMPILED_AVX2_DOT
ENGRAM_TARGET_AVX2 float dot_product_avx2_implementation(
    const float* left, const float* right, const std::size_t size) noexcept {
  __m256 accumulator = _mm256_setzero_ps();
  std::size_t index = 0;
  for (; index + 8 <= size; index += 8) {
    const __m256 left_values = _mm256_loadu_ps(left + index);
    const __m256 right_values = _mm256_loadu_ps(right + index);
    accumulator =
        _mm256_add_ps(accumulator, _mm256_mul_ps(left_values, right_values));
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, accumulator);
  float result = 0.0F;
  for (const float lane : lanes) {
    result += lane;
  }
  for (; index < size; ++index) {
    result += left[index] * right[index];
  }
  return result;
}
#endif

DotKernel dispatched_kernel() noexcept {
#if ENGRAM_COMPILED_AVX2_DOT
  if (cpu_features().avx2) {
    return &dot_product_avx2_implementation;
  }
#endif
  return &dot_product_scalar;
}

}  // namespace

float dot_product_scalar(const float* left, const float* right,
                         const std::size_t size) noexcept {
  float result = 0.0F;
  for (std::size_t index = 0; index < size; ++index) {
    result += left[index] * right[index];
  }
  return result;
}

float dot_product_avx2(const float* left, const float* right,
                       const std::size_t size) noexcept {
#if ENGRAM_COMPILED_AVX2_DOT
  if (cpu_features().avx2) {
    return dot_product_avx2_implementation(left, right, size);
  }
#endif
  return dot_product_scalar(left, right, size);
}

float dot_product(const float* left, const float* right,
                  const std::size_t size) noexcept {
  static const DotKernel kernel = dispatched_kernel();
  return kernel(left, right, size);
}

bool avx2_dot_kernel_compiled() noexcept {
  return ENGRAM_COMPILED_AVX2_DOT != 0;
}

VectorKernelKind selected_dot_kernel() noexcept {
#if ENGRAM_COMPILED_AVX2_DOT
  if (cpu_features().avx2) {
    return VectorKernelKind::avx2;
  }
#endif
  return VectorKernelKind::scalar;
}

const char* selected_dot_kernel_name() noexcept {
  return selected_dot_kernel() == VectorKernelKind::avx2 ? "avx2" : "scalar";
}

}  // namespace engram
