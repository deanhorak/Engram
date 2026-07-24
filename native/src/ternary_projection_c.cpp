#include "engram/ternary_projection_c.h"

#include "engram/ternary_projection.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <span>
#include <stdexcept>
#include <limits>

namespace {
void error_text(char* output, std::size_t capacity, const char* text) noexcept {
  if (output == nullptr || capacity == 0) return;
  const std::size_t length = std::min(capacity - 1, std::strlen(text));
  std::memcpy(output, text, length);
  output[length] = '\0';
}
}

extern "C" {
void* engram_ternary_projection_create(size_t threads, char* error,
                                       size_t capacity) {
  try {
    return new engram::TernaryProjectionKernel(threads);
  } catch (const std::exception& exception) {
    error_text(error, capacity, exception.what());
    return nullptr;
  }
}
void engram_ternary_projection_destroy(void* handle) {
  delete static_cast<engram::TernaryProjectionKernel*>(handle);
}
int engram_ternary_projection_add(
    void* handle, const uint8_t* packed, size_t packed_bytes,
    size_t input_features, size_t output_features, float scale,
    size_t* projection, char* error, size_t capacity) {
  try {
    auto* kernel = static_cast<engram::TernaryProjectionKernel*>(handle);
    if (kernel == nullptr || packed == nullptr || projection == nullptr)
      throw std::invalid_argument("ternary projection add received null storage");
    *projection = kernel->add(
        std::span<const std::uint8_t>(packed, packed_bytes), input_features,
        output_features, scale);
    return 0;
  } catch (const std::exception& exception) {
    error_text(error, capacity, exception.what());
    return 1;
  }
}
int engram_ternary_projection_forward_bf16(
    void* handle, size_t projection, const uint16_t* input, size_t rows,
    uint16_t* output, engram_ternary_projection_metrics* metrics, char* error,
    size_t capacity) {
  try {
    auto* kernel = static_cast<engram::TernaryProjectionKernel*>(handle);
    if (kernel == nullptr || input == nullptr || output == nullptr)
      throw std::invalid_argument(
          "ternary projection forward received null storage");
    const std::size_t input_features = kernel->input_features(projection);
    const std::size_t output_features = kernel->output_features(projection);
    if (rows == 0 ||
        rows > std::numeric_limits<std::size_t>::max() / input_features ||
        rows > std::numeric_limits<std::size_t>::max() / output_features)
      throw std::invalid_argument("ternary projection dimensions overflow");
    engram::TernaryProjectionMetrics native;
    kernel->forward_bf16(
        projection,
        std::span<const std::uint16_t>(input, rows * input_features), rows,
        std::span<std::uint16_t>(output, rows * output_features),
        metrics == nullptr ? nullptr : &native);
    if (metrics != nullptr) {
      metrics->elapsed_ns = native.elapsed_ns;
      metrics->packed_weight_bytes = native.packed_weight_bytes;
      metrics->scratch_bytes = native.scratch_bytes;
      metrics->rows = native.rows;
    }
    return 0;
  } catch (const std::exception& exception) {
    error_text(error, capacity, exception.what());
    return 1;
  }
}
}
