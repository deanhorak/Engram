#include "engram/olmoe_token_runtime_c.h"

#include "engram/olmoe_token_runtime.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <span>
#include <stdexcept>

namespace {

void error_text(char* output, const std::size_t capacity,
                const char* text) noexcept {
  if (output == nullptr || capacity == 0) return;
  const std::size_t length = std::min(capacity - 1, std::strlen(text));
  std::memcpy(output, text, length);
  output[length] = '\0';
}

}  // namespace

extern "C" {

void* engram_olmoe_token_open(const engram_olmoe_token_config* config,
                              char* error,
                              const std::size_t error_capacity) {
  try {
    if (config == nullptr || config->non_mlp_safetensors == nullptr ||
        config->q7_artifact == nullptr) {
      throw std::invalid_argument("native OLMoE token config is null");
    }
    return new engram::OLMoETokenRuntime(engram::OLMoETokenConfig{
        .non_mlp_safetensors =
            std::filesystem::path(config->non_mlp_safetensors),
        .q7_artifact = std::filesystem::path(config->q7_artifact),
        .layers = config->layers,
        .hidden_size = config->hidden_size,
        .query_heads = config->query_heads,
        .key_value_heads = config->key_value_heads,
        .head_dimension = config->head_dimension,
        .threads = config->threads,
        .local_window = config->local_window,
        .older_candidates = config->older_candidates,
        .older_top_k = config->older_top_k,
        .sink_tokens = config->sink_tokens,
        .rms_norm_epsilon = config->rms_norm_epsilon,
        .rope_theta = config->rope_theta,
        .eos_token_ids = {},
    });
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return nullptr;
  } catch (...) {
    error_text(error, error_capacity, "unknown native OLMoE token open failure");
    return nullptr;
  }
}

void engram_olmoe_token_close(void* handle) {
  delete static_cast<engram::OLMoETokenRuntime*>(handle);
}

void engram_olmoe_token_reset(void* handle) {
  if (handle != nullptr) {
    static_cast<engram::OLMoETokenRuntime*>(handle)->reset();
  }
}

size_t engram_olmoe_token_vocabulary_size(const void* handle) {
  const auto* runtime =
      static_cast<const engram::OLMoETokenRuntime*>(handle);
  return runtime == nullptr ? 0 : runtime->vocabulary_size();
}

size_t engram_olmoe_token_position(const void* handle) {
  const auto* runtime =
      static_cast<const engram::OLMoETokenRuntime*>(handle);
  return runtime == nullptr ? 0 : runtime->position();
}

int engram_olmoe_token_forward(
    void* handle, const int64_t* token_ids, const std::size_t length,
    int64_t* next_token, engram_olmoe_token_metrics* metrics, char* error,
    const std::size_t error_capacity) {
  try {
    auto* runtime = static_cast<engram::OLMoETokenRuntime*>(handle);
    if (runtime == nullptr || token_ids == nullptr || next_token == nullptr ||
        length == 0) {
      throw std::invalid_argument("native OLMoE token forward storage is invalid");
    }
    *next_token =
        runtime->forward(std::span<const std::int64_t>(token_ids, length));
    if (metrics != nullptr) {
      const auto& source = runtime->metrics();
      metrics->positions_processed = source.positions_processed;
      metrics->attention_weight_bytes = source.attention_weight_bytes;
      metrics->q7_scheduled_bytes = source.q7_scheduled_bytes;
      metrics->q7_elapsed_ns = source.q7_elapsed_ns;
      metrics->attention_state_bytes = source.attention_state_bytes;
      metrics->elapsed_ns = source.elapsed_ns;
    }
    return 0;
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    error_text(error, error_capacity,
               "unknown native OLMoE token forward failure");
    return 1;
  }
}

int engram_olmoe_token_copy_last_diagnostics(
    const void* handle, float* final_state,
    const std::size_t final_state_count, float* vocabulary_scores,
    const std::size_t vocabulary_score_count, char* error,
    const std::size_t error_capacity) {
  try {
    const auto* runtime =
        static_cast<const engram::OLMoETokenRuntime*>(handle);
    if (runtime == nullptr || !runtime->has_diagnostics() ||
        final_state == nullptr || vocabulary_scores == nullptr ||
        final_state_count != runtime->last_final_state().size() ||
        vocabulary_score_count !=
            runtime->last_vocabulary_scores().size()) {
      throw std::invalid_argument(
          "native OLMoE diagnostic storage is invalid");
    }
    std::copy(runtime->last_final_state().begin(),
              runtime->last_final_state().end(), final_state);
    std::copy(runtime->last_vocabulary_scores().begin(),
              runtime->last_vocabulary_scores().end(),
              vocabulary_scores);
    return 0;
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    error_text(error, error_capacity,
               "unknown native OLMoE diagnostic copy failure");
    return 1;
  }
}

}  // extern "C"
