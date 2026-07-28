#include "engram/olmoe_q7_c.h"

#include "engram/olmoe_q7.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>

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

void* engram_olmoe_q7_open(const char* artifact_path,
                           const std::size_t thread_count, char* error,
                           const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  try {
    if (artifact_path == nullptr) {
      throw std::invalid_argument("OLMoE Q7 artifact path is null");
    }
    return new engram::OLMoEQ7Kernel(std::filesystem::path(artifact_path),
                                     thread_count);
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return nullptr;
  } catch (...) {
    write_error(error, error_capacity, "unknown OLMoE Q7 open failure");
    return nullptr;
  }
}

void engram_olmoe_q7_close(void* handle) {
  delete static_cast<engram::OLMoEQ7Kernel*>(handle);
}

size_t engram_olmoe_q7_layer_count(const void* handle) {
  const auto* kernel = static_cast<const engram::OLMoEQ7Kernel*>(handle);
  return kernel == nullptr ? 0 : kernel->layer_count();
}
size_t engram_olmoe_q7_hidden_size(const void* handle) {
  const auto* kernel = static_cast<const engram::OLMoEQ7Kernel*>(handle);
  return kernel == nullptr ? 0 : kernel->hidden_size();
}
size_t engram_olmoe_q7_intermediate_size(const void* handle) {
  const auto* kernel = static_cast<const engram::OLMoEQ7Kernel*>(handle);
  return kernel == nullptr ? 0 : kernel->intermediate_size();
}
size_t engram_olmoe_q7_expert_count(const void* handle) {
  const auto* kernel = static_cast<const engram::OLMoEQ7Kernel*>(handle);
  return kernel == nullptr ? 0 : kernel->expert_count();
}
size_t engram_olmoe_q7_top_k(const void* handle) {
  const auto* kernel = static_cast<const engram::OLMoEQ7Kernel*>(handle);
  return kernel == nullptr ? 0 : kernel->top_k();
}
size_t engram_olmoe_q7_group_size(const void* handle) {
  const auto* kernel = static_cast<const engram::OLMoEQ7Kernel*>(handle);
  return kernel == nullptr ? 0 : kernel->group_size();
}
size_t engram_olmoe_q7_artifact_bytes(const void* handle) {
  const auto* kernel = static_cast<const engram::OLMoEQ7Kernel*>(handle);
  return kernel == nullptr ? 0 : kernel->serialized_artifact_bytes();
}

int engram_olmoe_q7_forward(
    void* handle, const std::size_t layer, const float* input,
    const std::size_t rows, float* output, std::uint32_t* selected_experts,
    engram_olmoe_q7_metrics* metrics, char* error,
    const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  try {
    auto* kernel = static_cast<engram::OLMoEQ7Kernel*>(handle);
    if (kernel == nullptr || input == nullptr || output == nullptr ||
        selected_experts == nullptr || rows == 0 ||
        rows > std::numeric_limits<std::size_t>::max() /
                   kernel->hidden_size() ||
        rows > std::numeric_limits<std::size_t>::max() / kernel->top_k()) {
      throw std::invalid_argument("OLMoE Q7 forward received invalid storage");
    }
    engram::OLMoEQ7Metrics native_metrics{};
    kernel->forward(
        layer,
        std::span<const float>(input, rows * kernel->hidden_size()), rows,
        std::span<float>(output, rows * kernel->hidden_size()),
        std::span<std::uint32_t>(selected_experts, rows * kernel->top_k()),
        metrics == nullptr ? nullptr : &native_metrics);
    if (metrics != nullptr) {
      metrics->elapsed_ns = native_metrics.elapsed_ns;
      metrics->router_stream_bytes = native_metrics.router_stream_bytes;
      metrics->selected_expert_stream_bytes =
          native_metrics.selected_expert_stream_bytes;
      metrics->scheduled_stream_bytes = native_metrics.scheduled_stream_bytes;
      metrics->scratch_bytes = native_metrics.scratch_bytes;
      metrics->rows = native_metrics.rows;
      metrics->threads = native_metrics.threads;
      metrics->selected_experts = native_metrics.selected_experts;
    }
    return 0;
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    write_error(error, error_capacity, "unknown OLMoE Q7 forward failure");
    return 1;
  }
}

}  // extern "C"
