#include "engram/ternary_projection.h"

#include "engram/thread_pool.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace engram {
namespace {
float bf16_to_float(std::uint16_t bits) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}
std::uint16_t float_to_bf16(float value) noexcept {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}
float bf16_round(float value) noexcept {
  return bf16_to_float(float_to_bf16(value));
}
struct Matrix {
  std::vector<std::uint8_t> owned;
  std::span<const std::uint8_t> mapped;
  std::size_t input{};
  std::size_t output{};
  float scale{};

  [[nodiscard]] std::span<const std::uint8_t> packed() const {
    return owned.empty() ? mapped
                         : std::span<const std::uint8_t>(owned);
  }
};
}  // namespace

class TernaryProjectionKernel::Impl {
 public:
  explicit Impl(std::size_t threads) : pool_(threads) {
    if (threads == 0 || threads > 256)
      throw std::invalid_argument("ternary projection thread count is invalid");
  }

  void validate(std::span<const std::uint8_t> packed, std::size_t input,
                std::size_t output, float scale) const {
    if (input == 0 || output == 0 || output % 4 != 0 ||
        packed.size() != (output / 4) * input || !std::isfinite(scale))
      throw std::invalid_argument("ternary projection matrix is invalid");
    for (std::uint8_t byte : packed) {
      for (unsigned shift = 0; shift < 8; shift += 2)
        if (((byte >> shift) & 3U) == 3U)
          throw std::invalid_argument("ternary projection code 3 is invalid");
    }
  }
  std::size_t add(std::span<const std::uint8_t> packed, std::size_t input,
                  std::size_t output, float scale) {
    validate(packed, input, output, scale);
    matrices_.push_back(Matrix{
        .owned = std::vector<std::uint8_t>(packed.begin(), packed.end()),
        .mapped = {},
        .input = input,
        .output = output,
        .scale = scale,
    });
    return matrices_.size() - 1;
  }
  std::size_t add_mapped(std::span<const std::uint8_t> packed,
                         std::size_t input, std::size_t output, float scale) {
    validate(packed, input, output, scale);
    matrices_.push_back(Matrix{
        .owned = {},
        .mapped = packed,
        .input = input,
        .output = output,
        .scale = scale,
    });
    return matrices_.size() - 1;
  }

  void forward(std::size_t index, std::span<const std::uint16_t> input,
               std::size_t rows, std::span<std::uint16_t> output,
               TernaryProjectionMetrics* metrics) {
    if (index >= matrices_.size() || rows == 0)
      throw std::invalid_argument("ternary projection dimensions are invalid");
    const Matrix& matrix = matrices_[index];
    if (input.size() != rows * matrix.input ||
        output.size() != rows * matrix.output)
      throw std::invalid_argument("ternary projection tensor size mismatch");
    const auto started = std::chrono::steady_clock::now();
    quantized_.resize(matrix.input * rows);
    accumulator_.resize(matrix.output * rows);
    pool_.parallel_for(0, rows, 1, [&](std::size_t row) {
      float maximum = 0.0F;
      for (std::size_t coordinate = 0; coordinate < matrix.input; ++coordinate)
        maximum = std::max(
            maximum,
            std::abs(bf16_to_float(input[row * matrix.input + coordinate])));
      const float activation_scale = 127.0F / std::max(maximum, 1.0e-5F);
      for (std::size_t coordinate = 0; coordinate < matrix.input; ++coordinate) {
        const float value =
            bf16_to_float(input[row * matrix.input + coordinate]);
        quantized_[coordinate * rows + row] = bf16_round(
            std::clamp(std::nearbyint(value * activation_scale), -128.0F,
                       127.0F) /
            activation_scale);
      }
    });
    const std::size_t packed_rows = matrix.output / 4;
    pool_.parallel_for(0, packed_rows, 4, [&](std::size_t packed_row) {
      for (std::size_t digit = 0; digit < 4; ++digit)
        std::fill_n(accumulator_.data() +
                        (packed_row + digit * packed_rows) * rows,
                    rows, 0.0F);
      const std::uint8_t* weights =
          matrix.packed().data() + packed_row * matrix.input;
      for (std::size_t coordinate = 0; coordinate < matrix.input; ++coordinate) {
        const float* values = quantized_.data() + coordinate * rows;
        const std::uint8_t byte = weights[coordinate];
        for (std::size_t digit = 0; digit < 4; ++digit) {
          const int trit = static_cast<int>((byte >> (2 * digit)) & 3U) - 1;
          float* target = accumulator_.data() +
                          (packed_row + digit * packed_rows) * rows;
          if (trit > 0)
            for (std::size_t row = 0; row < rows; ++row) target[row] += values[row];
          else if (trit < 0)
            for (std::size_t row = 0; row < rows; ++row) target[row] -= values[row];
        }
      }
      for (std::size_t digit = 0; digit < 4; ++digit) {
        const std::size_t feature = packed_row + digit * packed_rows;
        const float* source = accumulator_.data() + feature * rows;
        for (std::size_t row = 0; row < rows; ++row)
          output[row * matrix.output + feature] =
              float_to_bf16(bf16_round(source[row]) * matrix.scale);
      }
    });
    if (metrics != nullptr) {
      metrics->elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - started)
                                .count();
      metrics->packed_weight_bytes = matrix.packed().size();
      metrics->scratch_bytes =
          (quantized_.size() + accumulator_.size()) * sizeof(float);
      metrics->rows = rows;
    }
  }
  std::size_t input_features(std::size_t index) const {
    if (index >= matrices_.size())
      throw std::invalid_argument("ternary projection index is invalid");
    return matrices_[index].input;
  }
  std::size_t output_features(std::size_t index) const {
    if (index >= matrices_.size())
      throw std::invalid_argument("ternary projection index is invalid");
    return matrices_[index].output;
  }

 private:
  ThreadPool pool_;
  std::vector<Matrix> matrices_;
  std::vector<float> quantized_;
  std::vector<float> accumulator_;
};

TernaryProjectionKernel::TernaryProjectionKernel(std::size_t threads)
    : impl_(std::make_unique<Impl>(threads)) {}
TernaryProjectionKernel::~TernaryProjectionKernel() = default;
std::size_t TernaryProjectionKernel::add(std::span<const std::uint8_t> packed,
                                         std::size_t input,
                                         std::size_t output, float scale) {
  return impl_->add(packed, input, output, scale);
}
std::size_t TernaryProjectionKernel::add_mapped(
    std::span<const std::uint8_t> packed, std::size_t input,
    std::size_t output, float scale) {
  return impl_->add_mapped(packed, input, output, scale);
}
std::size_t TernaryProjectionKernel::input_features(
    std::size_t projection) const {
  return impl_->input_features(projection);
}
std::size_t TernaryProjectionKernel::output_features(
    std::size_t projection) const {
  return impl_->output_features(projection);
}
void TernaryProjectionKernel::forward_bf16(
    std::size_t projection, std::span<const std::uint16_t> input,
    std::size_t rows, std::span<std::uint16_t> output,
    TernaryProjectionMetrics* metrics) {
  impl_->forward(projection, input, rows, output, metrics);
}
}  // namespace engram
