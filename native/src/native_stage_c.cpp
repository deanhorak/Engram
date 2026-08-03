#include "engram/native_stage_c.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

float bf16_to_float(const std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::uint16_t float_to_bf16(const float value) {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

void error_text(char* output, const std::size_t capacity,
                const char* text) noexcept {
  if (output == nullptr || capacity == 0) return;
  const std::size_t length = std::min(capacity - 1, std::strlen(text));
  std::memcpy(output, text, length);
  output[length] = '\0';
}

class StageState {
 public:
  StageState(const std::size_t vectors, const std::size_t width)
      : vectors_(vectors),
        width_(width),
        state_(vectors * width),
        embedding_state_(vectors * width),
        attention_(vectors * width),
        post_attention_(vectors * width),
        residual_rms_(vectors) {
    if (vectors == 0 || width == 0) {
      throw std::invalid_argument("native stage dimensions must be positive");
    }
  }

  void begin(const std::uint16_t* embedding) {
    require(embedding, "embedding");
    for (std::size_t row = 0; row < vectors_; ++row) {
      float sum = 0.0F;
      for (std::size_t column = 0; column < width_; ++column) {
        const float value = bf16_to_float(embedding[row * width_ + column]);
        state_[row * width_ + column] = value;
        sum += value * value;
      }
      const float rms =
          std::max(std::sqrt(sum / static_cast<float>(width_)), 1.0e-6F);
      residual_rms_[row] = rms;
      for (std::size_t column = 0; column < width_; ++column) {
        state_[row * width_ + column] /= rms;
        embedding_state_[row * width_ + column] =
            state_[row * width_ + column];
      }
    }
    phase_ = Phase::attention;
  }

  void attention_input(const std::uint16_t* weight, const float epsilon,
                       std::uint16_t* output) const {
    require_phase(Phase::attention);
    norm(state_, weight, epsilon, output);
  }

  void accept_attention(const std::uint16_t* output) {
    require_phase(Phase::attention);
    require(output, "attention output");
    for (std::size_t row = 0; row < vectors_; ++row) {
      for (std::size_t column = 0; column < width_; ++column) {
        const std::size_t index = row * width_ + column;
        attention_[index] =
            bf16_to_float(output[index]) / residual_rms_[row];
        post_attention_[index] = state_[index] + attention_[index];
      }
    }
    phase_ = Phase::semantic;
  }

  void semantic_input(const std::uint16_t* weight, const float epsilon,
                      std::uint16_t* output) const {
    require_phase(Phase::semantic);
    norm(post_attention_, weight, epsilon, output);
  }

  void accept_semantic(const std::uint16_t* output,
                       const float semantic_scale,
                       const float episodic_scale) {
    require_phase(Phase::semantic);
    require(output, "semantic output");
    for (std::size_t row = 0; row < vectors_; ++row) {
      double sum = 0.0;
      for (std::size_t column = 0; column < width_; ++column) {
        const std::size_t index = row * width_ + column;
        const float semantic =
            bf16_to_float(output[index]) / residual_rms_[row];
        state_[index] += semantic_scale * semantic +
                         episodic_scale * attention_[index];
        sum += static_cast<double>(state_[index]) *
               static_cast<double>(state_[index]);
      }
      const float raw_rms = static_cast<float>(
          std::sqrt(sum / static_cast<double>(width_)));
      const float normalization_rms = static_cast<float>(
          std::sqrt(sum / static_cast<double>(width_) + 1.0e-6));
      residual_rms_[row] *= std::max(raw_rms, 1.0e-6F);
      for (std::size_t column = 0; column < width_; ++column) {
        state_[row * width_ + column] /= normalization_rms;
      }
    }
    phase_ = Phase::attention;
  }

  void accept_controller(
      const std::uint16_t* output, const float semantic_scale,
      const float episodic_scale,
      const engram_native_controller_weights_f32& controller) {
    require_phase(Phase::semantic);
    require(output, "semantic output");
    validate_controller(controller);
    std::vector<float> feature(controller.rank);
    std::vector<float> projected(2 * width_);
    std::vector<float> supplied(controller.input_dim);
    std::vector<float> residual(width_);
    for (std::size_t row = 0; row < vectors_; ++row) {
      const std::size_t base = row * width_;
      for (std::size_t column = 0; column < width_; ++column) {
        supplied[column] = embedding_state_[base + column];
        supplied[width_ + column] =
            bf16_to_float(output[base + column]) / residual_rms_[row];
        supplied[2 * width_ + column] = attention_[base + column];
        residual[column] =
            state_[base + column] + semantic_scale * supplied[width_ + column] +
            episodic_scale * supplied[2 * width_ + column];
      }
      if (controller.step_scale != 0.0F) {
        for (std::size_t bottleneck = 0; bottleneck < controller.rank;
             ++bottleneck) {
          float value = 0.0F;
          for (std::size_t input = 0; input < controller.input_dim; ++input) {
            value += supplied[input] *
                     controller.input_down[input * controller.rank + bottleneck];
          }
          for (std::size_t input_rank = 0;
               input_rank < controller.input_adapter_rank; ++input_rank) {
            float adapted = 0.0F;
            for (std::size_t input = 0; input < controller.input_dim; ++input) {
              adapted += supplied[input] * controller.input_adapter_down[
                  input * controller.input_adapter_rank + input_rank];
            }
            value += adapted * controller.input_adapter_up[
                input_rank * controller.rank + bottleneck];
          }
          float recurrent = 0.0F;
          for (std::size_t column = 0; column < width_; ++column) {
            recurrent += state_[base + column] *
                         controller.recurrent_down[column * controller.rank +
                                                   bottleneck];
          }
          feature[bottleneck] = silu(value + recurrent);
        }
        for (std::size_t output_column = 0; output_column < 2 * width_;
             ++output_column) {
          float value = controller.bias[output_column];
          for (std::size_t bottleneck = 0; bottleneck < controller.rank;
               ++bottleneck) {
            value += feature[bottleneck] *
                     controller.gate_up[bottleneck * (2 * width_) +
                                        output_column];
          }
          projected[output_column] = value;
        }
        for (std::size_t column = 0; column < width_; ++column) {
          float candidate = projected[width_ + column] +
                            controller.stage_embedding[column];
          for (std::size_t adapter = 0; adapter < controller.adapter_rank;
               ++adapter) {
            float down = 0.0F;
            for (std::size_t state_column = 0; state_column < width_;
                 ++state_column) {
              down += state_[base + state_column] *
                      controller.adapter_down[state_column *
                                                  controller.adapter_rank +
                                              adapter];
            }
            candidate += down *
                         controller.adapter_up[adapter * width_ + column];
          }
          const float gate = sigmoid(projected[column]);
          residual[column] += controller.step_scale * gate * std::tanh(candidate);
        }
      }
      double sum = 0.0;
      for (const float value : residual) sum += value * value;
      const float raw_rms = static_cast<float>(
          std::sqrt(sum / static_cast<double>(width_)));
      const float normalization_rms = static_cast<float>(
          std::sqrt(sum / static_cast<double>(width_) + 1.0e-6));
      residual_rms_[row] *= std::max(raw_rms, 1.0e-6F);
      for (std::size_t column = 0; column < width_; ++column) {
        state_[base + column] = residual[column] / normalization_rms;
      }
    }
    phase_ = Phase::attention;
  }

  void final_norm(const std::uint16_t* weight, const float epsilon,
                  std::uint16_t* output) const {
    require_phase(Phase::attention);
    norm(state_, weight, epsilon, output);
  }

  void copy(float* state, float* rms) const {
    require(state, "state output");
    require(rms, "RMS output");
    std::copy(state_.begin(), state_.end(), state);
    std::copy(residual_rms_.begin(), residual_rms_.end(), rms);
  }

 private:
  enum class Phase { uninitialized, attention, semantic };

  static void require(const void* pointer, const char* name) {
    if (pointer == nullptr) {
      throw std::invalid_argument(std::string("native stage missing ") + name);
    }
  }

  void validate_controller(
      const engram_native_controller_weights_f32& controller) const {
    if (controller.input_dim != 3 * width_ || controller.state_dim != width_ ||
        controller.rank == 0 ||
        (controller.input_adapter_rank != 0 &&
         (controller.input_adapter_down == nullptr ||
          controller.input_adapter_up == nullptr)) ||
        controller.input_down == nullptr || controller.recurrent_down == nullptr ||
        controller.gate_up == nullptr || controller.bias == nullptr ||
        controller.stage_embedding == nullptr || controller.adapter_down == nullptr ||
        controller.adapter_up == nullptr) {
      throw std::invalid_argument("native controller dimensions or tensors are invalid");
    }
  }

  static float sigmoid(const float value) {
    if (value >= 0.0F) {
      const float e = std::exp(-value);
      return 1.0F / (1.0F + e);
    }
    const float e = std::exp(value);
    return e / (1.0F + e);
  }

  static float silu(const float value) { return value * sigmoid(value); }

  void require_phase(const Phase expected) const {
    if (phase_ != expected) {
      throw std::logic_error("native stage operation is out of order");
    }
  }

  void norm(const std::vector<float>& input, const std::uint16_t* weight,
            const float epsilon, std::uint16_t* output) const {
    require(weight, "norm weight");
    require(output, "norm output");
    for (std::size_t row = 0; row < vectors_; ++row) {
      float sum = 0.0F;
      for (std::size_t column = 0; column < width_; ++column) {
        const float rounded = bf16_to_float(
            float_to_bf16(input[row * width_ + column]));
        sum += rounded * rounded;
      }
      const float inverse =
          1.0F /
          std::sqrt(sum / static_cast<float>(width_) + epsilon);
      for (std::size_t column = 0; column < width_; ++column) {
        const float rounded = bf16_to_float(
            float_to_bf16(input[row * width_ + column]));
        const float normalized =
            bf16_to_float(float_to_bf16(rounded * inverse));
        output[row * width_ + column] = float_to_bf16(
            normalized * bf16_to_float(weight[column]));
      }
    }
  }

  std::size_t vectors_;
  std::size_t width_;
  std::vector<float> state_;
  std::vector<float> embedding_state_;
  std::vector<float> attention_;
  std::vector<float> post_attention_;
  std::vector<float> residual_rms_;
  Phase phase_ = Phase::uninitialized;
};

template <typename Function>
int protect(Function&& function, char* error,
            const std::size_t error_capacity) {
  try {
    function();
    return 0;
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return 1;
  }
}

StageState& state(void* handle) {
  if (handle == nullptr) throw std::invalid_argument("native stage handle is null");
  return *static_cast<StageState*>(handle);
}

}  // namespace

extern "C" void* engram_native_stage_create(
    const std::size_t vectors, const std::size_t width, char* error,
    const std::size_t error_capacity) {
  try {
    return new StageState(vectors, width);
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return nullptr;
  }
}

extern "C" void engram_native_stage_destroy(void* handle) {
  delete static_cast<StageState*>(handle);
}

extern "C" int engram_native_stage_begin_bf16(
    void* handle, const std::uint16_t* embedding, char* error,
    const std::size_t capacity) {
  return protect([&] { state(handle).begin(embedding); }, error, capacity);
}

extern "C" int engram_native_stage_attention_input_bf16(
    void* handle, const std::uint16_t* weight, const float epsilon,
    std::uint16_t* output, char* error, const std::size_t capacity) {
  return protect(
      [&] { state(handle).attention_input(weight, epsilon, output); }, error,
      capacity);
}

extern "C" int engram_native_stage_accept_attention_bf16(
    void* handle, const std::uint16_t* output, char* error,
    const std::size_t capacity) {
  return protect([&] { state(handle).accept_attention(output); }, error,
                 capacity);
}

extern "C" int engram_native_stage_semantic_input_bf16(
    void* handle, const std::uint16_t* weight, const float epsilon,
    std::uint16_t* output, char* error, const std::size_t capacity) {
  return protect(
      [&] { state(handle).semantic_input(weight, epsilon, output); }, error,
      capacity);
}

extern "C" int engram_native_stage_accept_semantic_bf16(
    void* handle, const std::uint16_t* output, const float semantic_scale,
    const float episodic_scale, char* error, const std::size_t capacity) {
  return protect(
      [&] {
        state(handle).accept_semantic(output, semantic_scale, episodic_scale);
      },
      error, capacity);
}

extern "C" int engram_native_stage_accept_controller_f32(
    void* handle, const std::uint16_t* output, const float semantic_scale,
    const float episodic_scale,
    const engram_native_controller_weights_f32* controller, char* error,
    const std::size_t capacity) {
  return protect(
      [&] {
        if (controller == nullptr) {
          throw std::invalid_argument("native controller weights are null");
        }
        state(handle).accept_controller(output, semantic_scale, episodic_scale,
                                        *controller);
      },
      error, capacity);
}

extern "C" int engram_native_stage_final_norm_bf16(
    void* handle, const std::uint16_t* weight, const float epsilon,
    std::uint16_t* output, char* error, const std::size_t capacity) {
  return protect([&] { state(handle).final_norm(weight, epsilon, output); },
                 error, capacity);
}

extern "C" int engram_native_stage_copy_state_f32(
    void* handle, float* normalized_state, float* residual_rms, char* error,
    const std::size_t capacity) {
  return protect(
      [&] { state(handle).copy(normalized_state, residual_rms); }, error,
      capacity);
}
