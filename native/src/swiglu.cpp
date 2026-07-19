#include "engram/swiglu.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace engram {

void swiglu_scalar(const float* hidden, const float* gate, const float* up,
                   const float* down, const std::size_t hidden_size,
                   const std::size_t intermediate_size, float* output) {
  std::fill(output, output + hidden_size, 0.0F);
  std::vector<float> activation(intermediate_size);
  for (std::size_t neuron = 0; neuron < intermediate_size; ++neuron) {
    float gate_value = 0.0F;
    float up_value = 0.0F;
    for (std::size_t column = 0; column < hidden_size; ++column) {
      gate_value += gate[neuron * hidden_size + column] * hidden[column];
      up_value += up[neuron * hidden_size + column] * hidden[column];
    }
    activation[neuron] = (gate_value / (1.0F + std::exp(-gate_value))) * up_value;
  }
  for (std::size_t row = 0; row < hidden_size; ++row) {
    for (std::size_t neuron = 0; neuron < intermediate_size; ++neuron) {
      output[row] += down[row * intermediate_size + neuron] * activation[neuron];
    }
  }
}

}  // namespace engram
