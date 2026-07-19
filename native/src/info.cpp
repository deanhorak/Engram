#include "engram/cpu_features.h"
#include "engram/vector_kernels.h"

#include <iostream>

int main() {
  const auto features = engram::detect_cpu_features();
  std::cout << "engram native runtime 0.1.0\n"
            << "avx2_available=" << (features.avx2 ? "true" : "false") << '\n'
            << "vector_kernel=" << engram::selected_dot_kernel_name() << '\n';
  return 0;
}
