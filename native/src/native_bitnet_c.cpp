#include "engram/native_bitnet_c.h"

#include "engram/native_bitnet.h"
#include "engram/native_stage_c.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <span>
#include <vector>

namespace {

void write_error(char* error, const std::size_t capacity,
                 const char* message) noexcept {
  if (error == nullptr || capacity == 0) return;
  const std::size_t length = std::min(capacity - 1, std::strlen(message));
  std::memcpy(error, message, length);
  error[length] = '\0';
}

void clear_error(char* error, const std::size_t capacity) noexcept {
  if (error != nullptr && capacity != 0) error[0] = '\0';
}

}  // namespace

extern "C" {

void* engram_bitnet_open(const char* artifact_path,
                         const std::size_t thread_count, char* error,
                         const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  try {
    if (artifact_path == nullptr) {
      throw std::invalid_argument("native BitNet artifact path is null");
    }
    return new engram::NativeBitNetKernel(std::filesystem::path(artifact_path),
                                          thread_count);
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return nullptr;
  } catch (...) {
    write_error(error, error_capacity, "unknown native BitNet open failure");
    return nullptr;
  }
}

void engram_bitnet_close(void* handle) {
  delete static_cast<engram::NativeBitNetKernel*>(handle);
}

std::size_t engram_bitnet_layer_count(const void* handle) {
  const auto* kernel = static_cast<const engram::NativeBitNetKernel*>(handle);
  return kernel == nullptr ? 0 : kernel->layer_count();
}

std::size_t engram_bitnet_hidden_size(const void* handle) {
  const auto* kernel = static_cast<const engram::NativeBitNetKernel*>(handle);
  return kernel == nullptr ? 0 : kernel->hidden_size();
}

std::size_t engram_bitnet_intermediate_size(const void* handle) {
  const auto* kernel = static_cast<const engram::NativeBitNetKernel*>(handle);
  return kernel == nullptr ? 0 : kernel->intermediate_size();
}

std::size_t engram_bitnet_thread_count(const void* handle) {
  const auto* kernel = static_cast<const engram::NativeBitNetKernel*>(handle);
  return kernel == nullptr ? 0 : kernel->thread_count();
}

std::size_t engram_bitnet_artifact_bytes(const void* handle) {
  const auto* kernel = static_cast<const engram::NativeBitNetKernel*>(handle);
  return kernel == nullptr ? 0 : kernel->serialized_artifact_bytes();
}

int engram_bitnet_forward_bf16(
    void* handle, const std::size_t layer, const std::uint16_t* input,
    const std::size_t rows, std::uint16_t* output,
    engram_bitnet_metrics* const metrics, char* error,
    const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  try {
    auto* kernel = static_cast<engram::NativeBitNetKernel*>(handle);
    if (kernel == nullptr || input == nullptr || output == nullptr) {
      throw std::invalid_argument("native BitNet forward received null storage");
    }
    if (rows == 0 ||
        rows > std::numeric_limits<std::size_t>::max() /
                   kernel->hidden_size()) {
      throw std::invalid_argument(
          "native BitNet forward dimensions overflow");
    }
    const std::size_t elements = rows * kernel->hidden_size();
    engram::NativeBitNetMetrics native_metrics;
    kernel->forward_bf16(layer, std::span<const std::uint16_t>(input, elements),
                         rows, std::span<std::uint16_t>(output, elements),
                         metrics == nullptr ? nullptr : &native_metrics);
    if (metrics != nullptr) {
      metrics->elapsed_ns = native_metrics.elapsed_ns;
      metrics->gate_up_stream_bytes = native_metrics.gate_up_stream_bytes;
      metrics->norm_stream_bytes = native_metrics.norm_stream_bytes;
      metrics->down_stream_bytes = native_metrics.down_stream_bytes;
      metrics->layer_metadata_bytes = native_metrics.layer_metadata_bytes;
      metrics->scheduled_cache_line_bytes =
          native_metrics.scheduled_cache_line_bytes;
      metrics->scratch_bytes = native_metrics.scratch_bytes;
      metrics->rows = native_metrics.rows;
      metrics->threads = native_metrics.threads;
    }
    return 0;
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    write_error(error, error_capacity, "unknown native BitNet forward failure");
    return 1;
  }
}

int engram_bitnet_forward_oracle_bf16(
    void* handle, const std::size_t layer, const std::uint16_t* input,
    const std::size_t rows, const std::size_t top_k, std::uint16_t* output,
    engram_bitnet_metrics* const metrics, char* error,
    const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  try {
    auto* kernel = static_cast<engram::NativeBitNetKernel*>(handle);
    if (kernel == nullptr || input == nullptr || output == nullptr ||
        top_k == 0 || top_k > kernel->intermediate_size()) {
      throw std::invalid_argument(
          "native BitNet oracle forward received invalid storage or top-K");
    }
    if (rows == 0 ||
        rows > std::numeric_limits<std::size_t>::max() /
                   kernel->hidden_size()) {
      throw std::invalid_argument(
          "native BitNet oracle forward dimensions overflow");
    }
    const std::size_t elements = rows * kernel->hidden_size();
    engram::NativeBitNetMetrics native_metrics;
    kernel->forward_oracle_bf16(
        layer, std::span<const std::uint16_t>(input, elements), rows, top_k,
        std::span<std::uint16_t>(output, elements),
        metrics == nullptr ? nullptr : &native_metrics);
    if (metrics != nullptr) {
      metrics->elapsed_ns = native_metrics.elapsed_ns;
      metrics->gate_up_stream_bytes = native_metrics.gate_up_stream_bytes;
      metrics->norm_stream_bytes = native_metrics.norm_stream_bytes;
      metrics->down_stream_bytes = native_metrics.down_stream_bytes;
      metrics->layer_metadata_bytes = native_metrics.layer_metadata_bytes;
      metrics->scheduled_cache_line_bytes =
          native_metrics.scheduled_cache_line_bytes;
      metrics->scratch_bytes = native_metrics.scratch_bytes;
      metrics->rows = native_metrics.rows;
      metrics->threads = native_metrics.threads;
    }
    return 0;
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    write_error(error, error_capacity,
                "unknown native BitNet oracle forward failure");
    return 1;
  }
}

int engram_bitnet_stage_semantic_bf16(
    void* handle, void* stage_handle, const std::size_t layer,
    const std::uint16_t* norm_weight, const float norm_epsilon,
    const std::size_t rows, const float semantic_scale,
    const float episodic_scale, engram_bitnet_metrics* const metrics,
    char* error, const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  try {
    auto* kernel = static_cast<engram::NativeBitNetKernel*>(handle);
    if (kernel == nullptr || stage_handle == nullptr || norm_weight == nullptr ||
        rows == 0 ||
        rows > std::numeric_limits<std::size_t>::max() /
                   kernel->hidden_size()) {
      throw std::invalid_argument(
          "native BitNet stage semantic received invalid storage");
    }
    const std::size_t elements = rows * kernel->hidden_size();
    std::vector<std::uint16_t> input(elements);
    std::vector<std::uint16_t> output(elements);
    if (engram_native_stage_semantic_input_bf16(
            stage_handle, norm_weight, norm_epsilon, input.data(), error,
            error_capacity) != 0) {
      return 1;
    }
    if (engram_bitnet_forward_bf16(handle, layer, input.data(), rows,
                                   output.data(), metrics, error,
                                   error_capacity) != 0) {
      return 1;
    }
    if (engram_native_stage_accept_semantic_bf16(
            stage_handle, output.data(), semantic_scale, episodic_scale, error,
            error_capacity) != 0) {
      return 1;
    }
    return 0;
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    write_error(error, error_capacity,
                "unknown native BitNet stage semantic failure");
    return 1;
  }
}

}  // extern "C"
