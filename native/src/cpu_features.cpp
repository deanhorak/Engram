#include "engram/cpu_features.h"

namespace engram {

CpuFeatures detect_cpu_features() noexcept {
  CpuFeatures result{};
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || \
    defined(_M_X64)
  result.x86 = true;
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  result.sse2 = __builtin_cpu_supports("sse2") != 0;
  result.avx = __builtin_cpu_supports("avx") != 0;
  result.avx2 = __builtin_cpu_supports("avx2") != 0;
#else
  // This prototype currently provides guarded runtime probing through the
  // GCC-compatible builtin. Other toolchains retain the portable scalar path.
#if defined(__SSE2__)
  result.sse2 = true;
#endif
#if defined(__AVX__)
  result.avx = true;
#endif
#if defined(__AVX2__)
  result.avx2 = true;
#endif
#endif
#endif
  return result;
}

const CpuFeatures& cpu_features() noexcept {
  static const CpuFeatures features = detect_cpu_features();
  return features;
}

}  // namespace engram
