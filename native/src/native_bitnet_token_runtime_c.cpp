#include "engram/native_bitnet_token_runtime_c.h"

#include "engram/native_bitnet_token_runtime.h"
#include "engram/package.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

enum class RuntimeState {
  fresh,
  completed,
  poisoned,
};

void clear_error(char* error, const std::size_t capacity) noexcept {
  if (error != nullptr && capacity != 0) error[0] = '\0';
}

void write_error(char* error, const std::size_t capacity,
                 const std::string_view message) noexcept {
  if (error == nullptr || capacity == 0) return;
  const std::size_t length = std::min(capacity - 1, message.size());
  std::memcpy(error, message.data(), length);
  error[length] = '\0';
}

engram_native_bitnet_token_status fail(
    const engram_native_bitnet_token_status status,
    const std::string_view message, char* error,
    const std::size_t error_capacity) noexcept {
  write_error(error, error_capacity, message);
  return status;
}

std::uint64_t elapsed_ns(const Clock::time_point started) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now() - started)
          .count());
}

bool config_reserved_zero(
    const engram_native_bitnet_token_config_v1& config) noexcept {
  return std::all_of(std::begin(config.reserved), std::end(config.reserved),
                     [](const std::uint64_t value) { return value == 0; });
}

engram_native_bitnet_token_status prepare_info(
    engram_native_bitnet_token_info_v1* info, char* error,
    const std::size_t error_capacity) noexcept {
  if (info == nullptr) {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT,
                "native token runtime info output is null", error,
                error_capacity);
  }
  if (info->abi_version != ENGRAM_NATIVE_BITNET_TOKEN_ABI_V1 ||
      info->struct_size != sizeof(*info)) {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_ABI_MISMATCH,
                "native token runtime info ABI or size mismatch", error,
                error_capacity);
  }
  std::memset(info, 0, sizeof(*info));
  info->abi_version = ENGRAM_NATIVE_BITNET_TOKEN_ABI_V1;
  info->struct_size = sizeof(*info);
  return ENGRAM_NATIVE_BITNET_TOKEN_OK;
}

engram_native_bitnet_token_status prepare_metrics(
    engram_native_bitnet_token_metrics_v1* metrics, char* error,
    const std::size_t error_capacity) noexcept {
  if (metrics == nullptr) return ENGRAM_NATIVE_BITNET_TOKEN_OK;
  if (metrics->abi_version != ENGRAM_NATIVE_BITNET_TOKEN_ABI_V1 ||
      metrics->struct_size != sizeof(*metrics)) {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_ABI_MISMATCH,
                "native token runtime metrics ABI or size mismatch", error,
                error_capacity);
  }
  std::memset(metrics, 0, sizeof(*metrics));
  metrics->abi_version = ENGRAM_NATIVE_BITNET_TOKEN_ABI_V1;
  metrics->struct_size = sizeof(*metrics);
  return ENGRAM_NATIVE_BITNET_TOKEN_OK;
}

template <std::size_t Capacity>
void copy_text(char (&output)[Capacity], const std::string_view input) {
  static_assert(Capacity > 0);
  const std::size_t length = std::min(Capacity - 1, input.size());
  std::memcpy(output, input.data(), length);
  output[length] = '\0';
}

engram::NativeBitNetTokenConfig runtime_config(
    const engram::NativeBitNetDIPPackageMetadata& metadata,
    const std::size_t threads) {
  return engram::NativeBitNetTokenConfig{
      .non_mlp_safetensors = metadata.non_mlp_safetensors,
      .mlp_artifact = metadata.mlp_artifact,
      .dip_coordinate_index = metadata.dip_coordinate_index,
      .controller_directory = metadata.controller_directory,
      .layers = metadata.layers,
      .hidden_size = metadata.hidden_size,
      .query_heads = metadata.query_heads,
      .key_value_heads = metadata.key_value_heads,
      .head_dimension = metadata.head_dimension,
      .threads = threads,
      .local_window = metadata.local_window,
      .older_candidates = metadata.older_candidates,
      .older_top_k = metadata.older_top_k,
      .sink_tokens = metadata.sink_tokens,
      .rms_norm_epsilon = metadata.rms_norm_epsilon,
      .rope_theta = metadata.rope_theta,
      .eos_token_ids = metadata.eos_token_ids,
  };
}

}  // namespace

struct engram_native_bitnet_token_handle {
  engram_native_bitnet_token_handle(
      engram::NativeBitNetDIPPackageMetadata package_metadata,
      const std::size_t configured_threads,
      std::unique_ptr<engram::NativeBitNetTokenRuntime> token_runtime)
      : metadata(std::move(package_metadata)),
        threads(configured_threads),
        runtime(std::move(token_runtime)) {}

  std::mutex mutex;
  engram::NativeBitNetDIPPackageMetadata metadata;
  std::size_t threads = 0;
  std::unique_ptr<engram::NativeBitNetTokenRuntime> runtime;
  RuntimeState state = RuntimeState::fresh;
};

extern "C" {

std::uint32_t engram_native_bitnet_token_abi_version_v1(void) {
  return ENGRAM_NATIVE_BITNET_TOKEN_ABI_V1;
}

engram_native_bitnet_token_status engram_native_bitnet_token_create_v1(
    const engram_native_bitnet_token_config_v1* config,
    engram_native_bitnet_token_handle** output, char* error,
    const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  if (output == nullptr) {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT,
                "native token runtime output handle is null", error,
                error_capacity);
  }
  *output = nullptr;
  if (config == nullptr || config->package_path == nullptr ||
      config->package_path[0] == '\0') {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT,
                "native token runtime config or package path is null", error,
                error_capacity);
  }
  if (config->abi_version != ENGRAM_NATIVE_BITNET_TOKEN_ABI_V1 ||
      config->struct_size != sizeof(*config)) {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_ABI_MISMATCH,
                "native token runtime config ABI or size mismatch", error,
                error_capacity);
  }
  if (config->flags != 0 || !config_reserved_zero(*config)) {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_ABI_MISMATCH,
                "native token runtime flags and reserved fields must be zero",
                error, error_capacity);
  }
  if (config->threads > 256) {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT,
                "native token runtime threads must be zero or in [1, 256]",
                error, error_capacity);
  }

  try {
    engram::NativeBitNetDIPPackageMetadata metadata =
        engram::load_native_bitnet_dip_package(
            std::filesystem::path(config->package_path));
    const std::size_t threads =
        config->threads == 0 ? metadata.kernel_threads : config->threads;
    if (threads == 0 || threads > 256) {
      return fail(ENGRAM_NATIVE_BITNET_TOKEN_AUTHENTICATION_FAILED,
                  "authenticated native token thread policy is unsupported",
                  error, error_capacity);
    }
    if (metadata.eos_token_ids.empty() ||
        metadata.eos_token_ids.size() >
            ENGRAM_NATIVE_BITNET_TOKEN_MAX_EOS_IDS_V1 ||
        metadata.vocabulary_size >
            static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max())) {
      return fail(ENGRAM_NATIVE_BITNET_TOKEN_AUTHENTICATION_FAILED,
                  "authenticated native token EOS set is unsupported", error,
                  error_capacity);
    }
    auto runtime = std::make_unique<engram::NativeBitNetTokenRuntime>(
        runtime_config(metadata, threads));
    auto handle = std::make_unique<engram_native_bitnet_token_handle>(
        std::move(metadata), threads, std::move(runtime));
    *output = handle.release();
    return ENGRAM_NATIVE_BITNET_TOKEN_OK;
  } catch (const engram::PackageError& exception) {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_AUTHENTICATION_FAILED,
                exception.what(), error, error_capacity);
  } catch (const std::exception& exception) {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_RUNTIME_FAILED, exception.what(),
                error, error_capacity);
  } catch (...) {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_INTERNAL_FAILED,
                "unknown native token runtime creation failure", error,
                error_capacity);
  }
}

void engram_native_bitnet_token_destroy_v1(
    engram_native_bitnet_token_handle* handle) {
  delete handle;
}

engram_native_bitnet_token_status engram_native_bitnet_token_reset_v1(
    engram_native_bitnet_token_handle* handle, char* error,
    const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  if (handle == nullptr) {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT,
                "native token runtime handle is null", error,
                error_capacity);
  }
  std::unique_lock<std::mutex> lock;
  try {
    lock = std::unique_lock<std::mutex>(handle->mutex);
    handle->runtime->reset();
    handle->state = RuntimeState::fresh;
    return ENGRAM_NATIVE_BITNET_TOKEN_OK;
  } catch (const std::exception& exception) {
    if (lock.owns_lock()) handle->state = RuntimeState::poisoned;
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_RUNTIME_FAILED, exception.what(),
                error, error_capacity);
  } catch (...) {
    if (lock.owns_lock()) handle->state = RuntimeState::poisoned;
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_INTERNAL_FAILED,
                "unknown native token runtime reset failure", error,
                error_capacity);
  }
}

engram_native_bitnet_token_status engram_native_bitnet_token_get_info_v1(
    engram_native_bitnet_token_handle* handle,
    engram_native_bitnet_token_info_v1* info, char* error,
    const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  const engram_native_bitnet_token_status prepared =
      prepare_info(info, error, error_capacity);
  if (prepared != ENGRAM_NATIVE_BITNET_TOKEN_OK) return prepared;
  if (handle == nullptr) {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT,
                "native token runtime handle is null", error,
                error_capacity);
  }
  try {
    std::lock_guard<std::mutex> lock(handle->mutex);
    const auto& metadata = handle->metadata;
    info->max_position_embeddings = metadata.max_position_embeddings;
    info->vocabulary_size = metadata.vocabulary_size;
    info->layers = metadata.layers;
    info->hidden_size = metadata.hidden_size;
    info->intermediate_size = metadata.intermediate_size;
    info->query_heads = metadata.query_heads;
    info->key_value_heads = metadata.key_value_heads;
    info->head_dimension = metadata.head_dimension;
    info->local_window = metadata.local_window;
    info->older_candidates = metadata.older_candidates;
    info->older_top_k = metadata.older_top_k;
    info->sink_tokens = metadata.sink_tokens;
    info->thread_count = static_cast<std::uint32_t>(handle->threads);
    info->eos_token_count =
        static_cast<std::uint32_t>(metadata.eos_token_ids.size());
    info->rms_norm_epsilon = metadata.rms_norm_epsilon;
    info->rope_theta = metadata.rope_theta;
    std::copy(metadata.eos_token_ids.begin(), metadata.eos_token_ids.end(),
              info->eos_token_ids);
    copy_text(info->semantic_backend, handle->runtime->semantic_backend());
    copy_text(info->package_manifest_sha256, metadata.manifest_sha256);
    return ENGRAM_NATIVE_BITNET_TOKEN_OK;
  } catch (const std::exception& exception) {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_RUNTIME_FAILED, exception.what(),
                error, error_capacity);
  } catch (...) {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_INTERNAL_FAILED,
                "unknown native token runtime info failure", error,
                error_capacity);
  }
}

engram_native_bitnet_token_status engram_native_bitnet_token_generate_v1(
    engram_native_bitnet_token_handle* handle,
    const std::int64_t* prompt_tokens, const std::uint64_t prompt_count,
    const std::uint64_t max_new_tokens, std::int64_t* output_tokens,
    const std::uint64_t output_capacity, std::uint64_t* output_count,
    engram_native_bitnet_token_metrics_v1* metrics, char* error,
    const std::size_t error_capacity) {
  clear_error(error, error_capacity);
  if (output_count == nullptr) {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT,
                "native token runtime output count is null", error,
                error_capacity);
  }
  *output_count = 0;
  const engram_native_bitnet_token_status prepared =
      prepare_metrics(metrics, error, error_capacity);
  if (prepared != ENGRAM_NATIVE_BITNET_TOKEN_OK) return prepared;
  if (handle == nullptr || prompt_tokens == nullptr ||
      output_tokens == nullptr || prompt_count == 0 || max_new_tokens == 0) {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT,
                "native token runtime generation received null storage or "
                "zero dimensions",
                error, error_capacity);
  }
  if (prompt_count > std::numeric_limits<std::size_t>::max() ||
      max_new_tokens > std::numeric_limits<std::size_t>::max()) {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT,
                "native token runtime generation dimensions overflow", error,
                error_capacity);
  }
  if (output_capacity < max_new_tokens) {
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_BUFFER_TOO_SMALL,
                "native token runtime output capacity is below token budget",
                error, error_capacity);
  }

  std::unique_lock<std::mutex> lock;
  try {
    lock = std::unique_lock<std::mutex>(handle->mutex);
    if (handle->state != RuntimeState::fresh) {
      return fail(ENGRAM_NATIVE_BITNET_TOKEN_INVALID_STATE,
                  handle->state == RuntimeState::poisoned
                      ? "native token runtime is poisoned; reset is required"
                      : "native token runtime generation requires reset",
                  error, error_capacity);
    }
    const std::uint64_t context = handle->metadata.max_position_embeddings;
    if (prompt_count > context ||
        max_new_tokens - 1 > context - prompt_count) {
      return fail(ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT,
                  "native token prompt and budget exceed authenticated "
                  "context",
                  error, error_capacity);
    }
    const std::int64_t vocabulary =
        static_cast<std::int64_t>(handle->metadata.vocabulary_size);
    for (std::uint64_t index = 0; index < prompt_count; ++index) {
      if (prompt_tokens[index] < 0 || prompt_tokens[index] >= vocabulary) {
        return fail(
            ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT,
            "native token prompt id is outside authenticated vocabulary",
            error, error_capacity);
      }
    }
    const auto started = Clock::now();
    const std::vector<std::int64_t> generated = handle->runtime->generate(
        std::span<const std::int64_t>(
            prompt_tokens, static_cast<std::size_t>(prompt_count)),
        static_cast<std::size_t>(max_new_tokens));
    const std::uint64_t call_ns = elapsed_ns(started);
    if (generated.empty() || generated.size() > max_new_tokens ||
        generated.size() > output_capacity) {
      throw std::runtime_error(
          "native token runtime returned an invalid token count");
    }
    std::copy(generated.begin(), generated.end(), output_tokens);
    *output_count = generated.size();
    handle->state = RuntimeState::completed;

    if (metrics != nullptr) {
      const auto& native = handle->runtime->metrics();
      metrics->prompt_tokens = prompt_count;
      metrics->generated_tokens = generated.size();
      metrics->positions_processed = native.positions_processed;
      metrics->stage_calls = native.stage_calls;
      metrics->semantic_calls = native.semantic_calls;
      metrics->semantic_rows = native.semantic_rows;
      metrics->semantic_selected_records =
          native.semantic_selected_records;
      metrics->semantic_kernel_cache_line_bytes =
          native.semantic_kernel_cache_line_bytes;
      metrics->semantic_global_metadata_bytes =
          native.semantic_global_metadata_bytes;
      metrics->semantic_cache_line_bytes =
          native.semantic_scheduled_cache_line_bytes;
      metrics->semantic_maximum_scratch_bytes =
          native.semantic_maximum_scratch_bytes;
      metrics->attention_logical_read_bytes =
          native.attention_logical_read_bytes;
      metrics->attention_state_bytes = native.attention_state_bytes;
      metrics->attention_scratch_bytes = native.attention_scratch_bytes;
      metrics->qkv_projection_ns = native.qkv_projection_ns;
      metrics->rope_ns = native.rope_ns;
      metrics->native_attention_ns = native.native_attention_ns;
      metrics->output_projection_ns = native.output_projection_ns;
      metrics->semantic_elapsed_ns = native.semantic_elapsed_ns;
      metrics->attention_elapsed_ns = native.attention_elapsed_ns;
      metrics->call_elapsed_ns = call_ns;
      metrics->prefill_elapsed_ns = native.prefill_elapsed_ns;
      metrics->decode_elapsed_ns = native.decode_elapsed_ns;
      metrics->stopped_on_eos = static_cast<std::uint32_t>(
          std::find(handle->metadata.eos_token_ids.begin(),
                    handle->metadata.eos_token_ids.end(),
                    generated.back()) !=
          handle->metadata.eos_token_ids.end());
    }
    return ENGRAM_NATIVE_BITNET_TOKEN_OK;
  } catch (const std::exception& exception) {
    if (lock.owns_lock()) handle->state = RuntimeState::poisoned;
    *output_count = 0;
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_RUNTIME_FAILED, exception.what(),
                error, error_capacity);
  } catch (...) {
    if (lock.owns_lock()) handle->state = RuntimeState::poisoned;
    *output_count = 0;
    return fail(ENGRAM_NATIVE_BITNET_TOKEN_INTERNAL_FAILED,
                "unknown native token runtime generation failure", error,
                error_capacity);
  }
}

}  // extern "C"
