#pragma once

namespace engram {

struct CpuFeatures {
  bool x86{};
  bool sse2{};
  bool avx{};
  bool avx2{};
};

// Detect features usable by the current process. On supported x86 GCC/Clang
// builds, the compiler builtin accounts for CPU and OS extended-state support.
[[nodiscard]] CpuFeatures detect_cpu_features() noexcept;
[[nodiscard]] const CpuFeatures& cpu_features() noexcept;

}  // namespace engram
