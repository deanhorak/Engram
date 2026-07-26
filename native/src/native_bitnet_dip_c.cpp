#include "engram/native_bitnet_dip_c.h"

#include "engram/native_bitnet_dip.h"

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

void copy_metrics(const engram::NativeBitNetDIPMetrics& source,
                  engram_bitnet_dip_metrics* destination) {
  destination->elapsed_ns = source.elapsed_ns;
  destination->coordinate_stream_bytes = source.coordinate_stream_bytes;
  destination->candidate_completion_bytes =
      source.candidate_completion_bytes;
  destination->gain_stream_bytes = source.gain_stream_bytes;
  destination->down_norm_stream_bytes = source.down_norm_stream_bytes;
  destination->selected_down_stream_bytes =
      source.selected_down_stream_bytes;
  destination->layer_metadata_bytes = source.layer_metadata_bytes;
  destination->scheduled_cache_line_bytes =
      source.scheduled_cache_line_bytes;
  destination->scratch_bytes = source.scratch_bytes;
  destination->rows = source.rows;
  destination->threads = source.threads;
  destination->input_coordinates = source.input_coordinates;
  destination->candidate_count = source.candidate_count;
  destination->selected_count_total = source.selected_count_total;
  destination->selected_count_min = source.selected_count_min;
  destination->selected_count_max = source.selected_count_max;
}

}  // namespace

extern "C" {

void* engram_bitnet_dip_open(const char* record_artifact_path,
                             const char* coordinate_index_path,
                             const std::size_t thread_count, char* error,
                             const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  try {
    if (record_artifact_path == nullptr || coordinate_index_path == nullptr) {
      throw std::invalid_argument(
          "native BitNet DIP artifact path is null");
    }
    return new engram::NativeBitNetDIPKernel(
        std::filesystem::path(record_artifact_path),
        std::filesystem::path(coordinate_index_path), thread_count);
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return nullptr;
  } catch (...) {
    write_error(error, error_capacity,
                "unknown native BitNet DIP open failure");
    return nullptr;
  }
}

void engram_bitnet_dip_close(void* handle) {
  delete static_cast<engram::NativeBitNetDIPKernel*>(handle);
}

std::size_t engram_bitnet_dip_layer_count(const void* handle) {
  const auto* kernel =
      static_cast<const engram::NativeBitNetDIPKernel*>(handle);
  return kernel == nullptr ? 0 : kernel->layer_count();
}

std::size_t engram_bitnet_dip_hidden_size(const void* handle) {
  const auto* kernel =
      static_cast<const engram::NativeBitNetDIPKernel*>(handle);
  return kernel == nullptr ? 0 : kernel->hidden_size();
}

std::size_t engram_bitnet_dip_intermediate_size(const void* handle) {
  const auto* kernel =
      static_cast<const engram::NativeBitNetDIPKernel*>(handle);
  return kernel == nullptr ? 0 : kernel->intermediate_size();
}

std::size_t engram_bitnet_dip_thread_count(const void* handle) {
  const auto* kernel =
      static_cast<const engram::NativeBitNetDIPKernel*>(handle);
  return kernel == nullptr ? 0 : kernel->thread_count();
}

std::size_t engram_bitnet_dip_record_artifact_bytes(const void* handle) {
  const auto* kernel =
      static_cast<const engram::NativeBitNetDIPKernel*>(handle);
  return kernel == nullptr ? 0 : kernel->record_artifact_bytes();
}

std::size_t engram_bitnet_dip_coordinate_index_bytes(const void* handle) {
  const auto* kernel =
      static_cast<const engram::NativeBitNetDIPKernel*>(handle);
  return kernel == nullptr ? 0 : kernel->coordinate_index_bytes();
}

int engram_bitnet_dip_layer_policy(
    const void* handle, const std::size_t layer,
    std::size_t* input_coordinates, std::size_t* candidate_count,
    std::size_t* minimum_top_k, std::size_t* maximum_top_k,
    float* energy_target, std::size_t* rms_audit_count,
    std::uint32_t* rms_estimator, std::uint32_t* rms_audit_strategy,
    char* error, const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  try {
    const auto* kernel =
        static_cast<const engram::NativeBitNetDIPKernel*>(handle);
    if (kernel == nullptr || input_coordinates == nullptr ||
        candidate_count == nullptr || minimum_top_k == nullptr ||
        maximum_top_k == nullptr || energy_target == nullptr ||
        rms_audit_count == nullptr || rms_estimator == nullptr ||
        rms_audit_strategy == nullptr) {
      throw std::invalid_argument(
          "native BitNet DIP policy received null storage");
    }
    const engram::NativeBitNetDIPPolicy& policy = kernel->policy(layer);
    *input_coordinates = policy.input_coordinates;
    *candidate_count = policy.candidate_count;
    *minimum_top_k = policy.minimum_top_k;
    *maximum_top_k = policy.maximum_top_k;
    *energy_target = policy.energy_target;
    *rms_audit_count = policy.rms_audit_count;
    *rms_estimator = static_cast<std::uint32_t>(policy.rms_estimator);
    *rms_audit_strategy =
        static_cast<std::uint32_t>(policy.rms_audit_strategy);
    return 0;
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    write_error(error, error_capacity,
                "unknown native BitNet DIP policy failure");
    return 1;
  }
}

int engram_bitnet_dip_forward_bf16(
    void* handle, const std::size_t layer, const std::uint16_t* input,
    const std::size_t rows, std::uint16_t* output,
    std::uint32_t* selected_counts, engram_bitnet_dip_metrics* metrics,
    char* error, const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  try {
    auto* kernel = static_cast<engram::NativeBitNetDIPKernel*>(handle);
    if (kernel == nullptr || input == nullptr || output == nullptr ||
        selected_counts == nullptr || rows == 0 ||
        rows > std::numeric_limits<std::size_t>::max() /
                   kernel->hidden_size()) {
      throw std::invalid_argument(
          "native BitNet DIP forward received invalid storage");
    }
    const std::size_t elements = rows * kernel->hidden_size();
    engram::NativeBitNetDIPMetrics native_metrics;
    kernel->forward_bf16(
        layer, std::span<const std::uint16_t>(input, elements), rows,
        std::span<std::uint16_t>(output, elements),
        std::span<std::uint32_t>(selected_counts, rows),
        metrics == nullptr ? nullptr : &native_metrics);
    if (metrics != nullptr) {
      copy_metrics(native_metrics, metrics);
    }
    return 0;
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    write_error(error, error_capacity,
                "unknown native BitNet DIP forward failure");
    return 1;
  }
}

int engram_bitnet_dip_forward_debug_bf16(
    void* handle, const std::size_t layer, const std::uint16_t* input,
    const std::size_t rows, std::uint16_t* output,
    std::uint32_t* selected_counts,
    std::uint32_t* input_coordinate_ids, std::uint32_t* candidate_ids,
    std::uint32_t* selected_record_ids,
    engram_bitnet_dip_metrics* metrics, char* error,
    const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  try {
    auto* kernel = static_cast<engram::NativeBitNetDIPKernel*>(handle);
    if (kernel == nullptr || input == nullptr || output == nullptr ||
        selected_counts == nullptr || input_coordinate_ids == nullptr ||
        candidate_ids == nullptr || selected_record_ids == nullptr ||
        rows == 0 ||
        rows > std::numeric_limits<std::size_t>::max() /
                   kernel->hidden_size()) {
      throw std::invalid_argument(
          "native BitNet DIP debug forward received invalid storage");
    }
    const engram::NativeBitNetDIPPolicy& policy =
        kernel->policy(layer);
    if (rows > std::numeric_limits<std::size_t>::max() /
                   policy.input_coordinates ||
        rows > std::numeric_limits<std::size_t>::max() /
                   policy.candidate_count ||
        rows > std::numeric_limits<std::size_t>::max() /
                   policy.maximum_top_k) {
      throw std::invalid_argument(
          "native BitNet DIP debug dimensions overflow");
    }
    const std::size_t elements = rows * kernel->hidden_size();
    engram::NativeBitNetDIPMetrics native_metrics;
    kernel->forward_debug_bf16(
        layer, std::span<const std::uint16_t>(input, elements), rows,
        std::span<std::uint16_t>(output, elements),
        std::span<std::uint32_t>(selected_counts, rows),
        std::span<std::uint32_t>(
            input_coordinate_ids, rows * policy.input_coordinates),
        std::span<std::uint32_t>(
            candidate_ids, rows * policy.candidate_count),
        std::span<std::uint32_t>(
            selected_record_ids, rows * policy.maximum_top_k),
        metrics == nullptr ? nullptr : &native_metrics);
    if (metrics != nullptr) {
      copy_metrics(native_metrics, metrics);
    }
    return 0;
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    write_error(error, error_capacity,
                "unknown native BitNet DIP debug forward failure");
    return 1;
  }
}

int engram_bitnet_dip_teacher_top_k_bf16(
    void* handle, const std::size_t layer, const std::uint16_t* input,
    const std::size_t rows, const std::size_t top_k,
    std::uint32_t* teacher_record_ids, char* error,
    const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  try {
    auto* kernel = static_cast<engram::NativeBitNetDIPKernel*>(handle);
    if (kernel == nullptr || input == nullptr ||
        teacher_record_ids == nullptr || rows == 0 || top_k == 0 ||
        top_k > kernel->intermediate_size() ||
        rows > std::numeric_limits<std::size_t>::max() /
                   kernel->hidden_size() ||
        rows > std::numeric_limits<std::size_t>::max() / top_k) {
      throw std::invalid_argument(
          "native BitNet DIP teacher top-K received invalid storage");
    }
    kernel->teacher_top_k_bf16(
        layer,
        std::span<const std::uint16_t>(
            input, rows * kernel->hidden_size()),
        rows, top_k,
        std::span<std::uint32_t>(teacher_record_ids, rows * top_k));
    return 0;
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    write_error(error, error_capacity,
                "unknown native BitNet DIP teacher top-K failure");
    return 1;
  }
}

int engram_bitnet_dip_teacher_top_k_positive_bf16(
    void* handle, const std::size_t layer, const std::uint16_t* input,
    const std::size_t rows, const std::size_t top_k,
    std::uint32_t* teacher_record_ids,
    std::uint32_t* positive_utility_counts, char* error,
    const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  try {
    auto* kernel = static_cast<engram::NativeBitNetDIPKernel*>(handle);
    if (kernel == nullptr || input == nullptr ||
        teacher_record_ids == nullptr || positive_utility_counts == nullptr ||
        rows == 0 || top_k == 0 ||
        top_k > kernel->intermediate_size() ||
        rows > std::numeric_limits<std::size_t>::max() /
                   kernel->hidden_size() ||
        rows > std::numeric_limits<std::size_t>::max() / top_k) {
      throw std::invalid_argument(
          "native BitNet DIP positive teacher top-K received invalid storage");
    }
    kernel->teacher_top_k_bf16(
        layer,
        std::span<const std::uint16_t>(
            input, rows * kernel->hidden_size()),
        rows, top_k,
        std::span<std::uint32_t>(teacher_record_ids, rows * top_k),
        std::span<std::uint32_t>(positive_utility_counts, rows));
    return 0;
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    write_error(
        error, error_capacity,
        "unknown native BitNet DIP positive teacher top-K failure");
    return 1;
  }
}

}  // extern "C"
