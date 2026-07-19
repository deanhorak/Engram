#include "engram/swiglu.h"

#include <cmath>
#include <iostream>

int main() {
  constexpr float hidden[] = {1.0F, -2.0F};
  constexpr float gate[] = {0.5F, 0.25F, -1.0F, 0.5F};
  constexpr float up[] = {2.0F, 1.0F, 0.5F, -0.25F};
  constexpr float down[] = {1.0F, 2.0F, -0.5F, 0.25F};
  float output[2] = {};
  engram::swiglu_scalar(hidden, gate, up, down, 2, 2, output);

  const float gate0 = 0.5F - 0.5F;
  const float gate1 = -1.0F - 1.0F;
  const float a0 = (gate0 / (1.0F + std::exp(-gate0))) * 0.0F;
  const float a1 = (gate1 / (1.0F + std::exp(-gate1))) * 1.0F;
  const float expected0 = a0 + 2.0F * a1;
  const float expected1 = -0.5F * a0 + 0.25F * a1;
  if (std::abs(output[0] - expected0) > 1e-6F ||
      std::abs(output[1] - expected1) > 1e-6F) {
    std::cerr << "scalar SwiGLU mismatch\n";
    return 1;
  }
  return 0;
}
