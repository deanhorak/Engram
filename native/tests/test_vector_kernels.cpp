#include "engram/cpu_features.h"
#include "engram/vector_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool close(const float actual, const float expected, const float tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

int fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}

}  // namespace

int main() {
  constexpr float left[] = {1.0F, -2.0F, 3.5F, 0.25F, -4.0F};
  constexpr float right[] = {2.0F, 0.5F, -1.0F, 8.0F, -0.25F};
  constexpr float expected = 0.5F;
  if (engram::dot_product_scalar(left, right, 5) != expected ||
      engram::dot_product(left, right, 5) != expected ||
      engram::dot_product_avx2(left, right, 5) != expected) {
    return fail("fixed scalar/dispatched dot product mismatch");
  }
  if (engram::dot_product(nullptr, nullptr, 0) != 0.0F) {
    return fail("empty dot product must be zero");
  }

  constexpr std::int8_t quantized[] = {2, -1, 3, 4, -2};
  constexpr float quantized_scale = 0.5F;
  constexpr float quantized_expected = 11.75F;
  if (!close(engram::dot_product_int8_scalar(
                 left, quantized, quantized_scale, 5),
             quantized_expected, 1e-6F) ||
      !close(engram::dot_product_int8(
                 left, quantized, quantized_scale, 5),
             quantized_expected, 1e-6F) ||
      !close(engram::dot_product_int8_avx2(
                 left, quantized, quantized_scale, 5),
             quantized_expected, 1e-6F)) {
    return fail("INT8 scalar/dispatched dot product mismatch");
  }

  // Exercise vector blocks plus a scalar tail using deterministic finite data.
  constexpr std::size_t size = 1031;
  std::vector<float> long_left(size);
  std::vector<float> long_right(size);
  for (std::size_t index = 0; index < size; ++index) {
    const float position = static_cast<float>(index);
    long_left[index] = std::sin(position * 0.031F) + position * 1e-4F;
    long_right[index] = std::cos(position * 0.017F) - position * 2e-4F;
  }
  const float scalar =
      engram::dot_product_scalar(long_left.data(), long_right.data(), size);
  const float dispatched =
      engram::dot_product(long_left.data(), long_right.data(), size);
  const float safe_avx2 =
      engram::dot_product_avx2(long_left.data(), long_right.data(), size);
  const float tolerance = 3e-5F * std::max(1.0F, std::abs(scalar));
  if (!close(dispatched, scalar, tolerance) ||
      !close(safe_avx2, scalar, tolerance)) {
    return fail("AVX2/scalar dot product parity mismatch");
  }

  const engram::CpuFeatures detected = engram::detect_cpu_features();
  const engram::CpuFeatures& cached = engram::cpu_features();
  if (detected.x86 != cached.x86 || detected.sse2 != cached.sse2 ||
      detected.avx != cached.avx || detected.avx2 != cached.avx2) {
    return fail("cached CPU features differ from direct detection");
  }
  const bool should_dispatch_avx2 =
      engram::avx2_dot_kernel_compiled() && cached.avx2;
  if ((engram::selected_dot_kernel() == engram::VectorKernelKind::avx2) !=
          should_dispatch_avx2 ||
      (std::string(engram::selected_dot_kernel_name()) == "avx2") !=
          should_dispatch_avx2) {
    return fail("dot kernel dispatch disagrees with usable CPU features");
  }
  return 0;
}
