#include "engram/olmoe_token_runtime_c.h"

#include "engram/olmoe_token_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

void error_text(char* output, const std::size_t capacity,
                const char* text) noexcept {
  if (output == nullptr || capacity == 0) return;
  const std::size_t length = std::min(capacity - 1, std::strlen(text));
  std::memcpy(output, text, length);
  output[length] = '\0';
}

engram::OLMoETokenConfig native_config(
    const engram_olmoe_token_config* config,
    const std::span<const engram_olmoe_attention_policy_v1> layer_policies,
    const std::span<const engram_olmoe_attention_policy_v1> head_policies,
    const engram_olmoe_episodic_policy_v1* episodic_policy,
    const std::span<const std::uint8_t> episodic_head_mask,
    const float episodic_logit_bias = 0.0F,
    const engram_olmoe_attention_policy_v1* shadow_policy = nullptr) {
  if (config == nullptr || config->non_mlp_safetensors == nullptr ||
      config->q7_artifact == nullptr) {
    throw std::invalid_argument("native OLMoE token config is null");
  }
  const auto copy_policies =
      [](const std::span<const engram_olmoe_attention_policy_v1> policies) {
        std::vector<engram::OLMoEAttentionPolicy> result;
        result.reserve(policies.size());
        for (const auto& policy : policies) {
          result.push_back(engram::OLMoEAttentionPolicy{
              .local_window = policy.local_window,
              .older_candidates = policy.older_candidates,
              .older_top_k = policy.older_top_k,
              .sink_tokens = policy.sink_tokens,
          });
        }
        return result;
      };
  return engram::OLMoETokenConfig{
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
      .attention_policies = copy_policies(layer_policies),
      .head_attention_policies = copy_policies(head_policies),
      .episodic_policy =
          episodic_policy == nullptr
              ? engram::OLMoEEpisodicPolicy{}
              : engram::OLMoEEpisodicPolicy{
                    .slots = episodic_policy->slots,
                    .span_size = episodic_policy->span_size,
                },
      .episodic_head_mask = std::vector<std::uint8_t>(
          episodic_head_mask.begin(), episodic_head_mask.end()),
      .episodic_logit_bias = episodic_logit_bias,
      .shadow_attention_policy =
          shadow_policy == nullptr
              ? std::nullopt
              : std::optional<engram::OLMoEAttentionPolicy>(
                    engram::OLMoEAttentionPolicy{
                        .local_window = shadow_policy->local_window,
                        .older_candidates =
                            shadow_policy->older_candidates,
                        .older_top_k = shadow_policy->older_top_k,
                        .sink_tokens = shadow_policy->sink_tokens,
                    }),
  };
}

bool valid_policy(
    const engram_olmoe_attention_policy_v1& policy) noexcept {
  return policy.local_window > 0 && policy.older_candidates > 0 &&
         policy.older_top_k > 0 &&
         policy.older_top_k <= policy.older_candidates &&
         policy.sink_tokens <= policy.older_candidates;
}

std::size_t checked_head_policy_count(
    const engram_olmoe_token_config& config) {
  if (config.layers != 0 &&
      config.query_heads >
          std::numeric_limits<std::size_t>::max() / config.layers) {
    throw std::invalid_argument(
        "native OLMoE head-wise attention policy count overflows");
  }
  return config.layers * config.query_heads;
}

void* open_episodic_headwise(
    const engram_olmoe_token_config* config,
    const engram_olmoe_episodic_policy_v1* policy,
    const std::uint8_t* head_mask, const std::size_t mask_count,
    const float episodic_logit_bias,
    const engram_olmoe_attention_policy_v1* shadow_policy = nullptr) {
  if (!std::isfinite(episodic_logit_bias)) {
    throw std::invalid_argument(
        "native OLMoE episodic logit bias is invalid");
  }
  if (shadow_policy != nullptr && !valid_policy(*shadow_policy)) {
    throw std::invalid_argument(
        "native OLMoE shadow attention policy is invalid");
  }
  if (config == nullptr || policy == nullptr || policy->slots == 0 ||
      policy->span_size == 0 ||
      policy->slots % policy->span_size != 0 ||
      head_mask == nullptr ||
      mask_count != checked_head_policy_count(*config)) {
    throw std::invalid_argument(
        "native OLMoE head-gated episodic policy is invalid");
  }
  const std::span<const std::uint8_t> mask(head_mask, mask_count);
  if (!std::all_of(mask.begin(), mask.end(),
                   [](const std::uint8_t selected) {
                     return selected <= 1;
                   }) ||
      std::none_of(mask.begin(), mask.end(),
                   [](const std::uint8_t selected) {
                     return selected != 0;
                   })) {
    throw std::invalid_argument(
        "native OLMoE head-gated episodic mask is invalid");
  }
  return new engram::OLMoETokenRuntime(native_config(
      config, {}, {}, policy, mask, episodic_logit_bias,
      shadow_policy));
}

void copy_forward_metrics(const engram::OLMoETokenMetrics& source,
                          engram_olmoe_token_metrics* metrics) {
  if (metrics == nullptr) return;
  metrics->positions_processed = source.positions_processed;
  metrics->attention_weight_bytes = source.attention_weight_bytes;
  metrics->q7_scheduled_bytes = source.q7_scheduled_bytes;
  metrics->q7_elapsed_ns = source.q7_elapsed_ns;
  metrics->attention_state_bytes = source.attention_state_bytes;
  metrics->elapsed_ns = source.elapsed_ns;
}

}  // namespace

extern "C" {

void* engram_olmoe_token_open(const engram_olmoe_token_config* config,
                              char* error,
                              const std::size_t error_capacity) {
  try {
    return new engram::OLMoETokenRuntime(
        native_config(config, {}, {}, nullptr, {}));
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return nullptr;
  } catch (...) {
    error_text(error, error_capacity, "unknown native OLMoE token open failure");
    return nullptr;
  }
}

void* engram_olmoe_token_open_layered_v1(
    const engram_olmoe_token_config* config,
    const engram_olmoe_attention_policy_v1* policies,
    const std::size_t policy_count, char* error,
    const std::size_t error_capacity) {
  try {
    if (config == nullptr || policies == nullptr ||
        policy_count != config->layers) {
      throw std::invalid_argument(
          "native OLMoE layered attention policy count is invalid");
    }
    const std::span<const engram_olmoe_attention_policy_v1> policy_span(
        policies, policy_count);
    if (!std::all_of(policy_span.begin(), policy_span.end(), valid_policy)) {
      throw std::invalid_argument(
          "native OLMoE layered attention policy is invalid");
    }
    return new engram::OLMoETokenRuntime(
        native_config(config, policy_span, {}, nullptr, {}));
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return nullptr;
  } catch (...) {
    error_text(
        error, error_capacity,
        "unknown native OLMoE layered token open failure");
    return nullptr;
  }
}

void* engram_olmoe_token_open_headwise_v1(
    const engram_olmoe_token_config* config,
    const engram_olmoe_attention_policy_v1* policies,
    const std::size_t policy_count, char* error,
    const std::size_t error_capacity) {
  try {
    if (config == nullptr || policies == nullptr ||
        policy_count != checked_head_policy_count(*config)) {
      throw std::invalid_argument(
          "native OLMoE head-wise attention policy count is invalid");
    }
    if (config->query_heads != config->key_value_heads) {
      throw std::invalid_argument(
          "native OLMoE head-wise attention requires equal query and "
          "key/value head counts");
    }
    const std::span<const engram_olmoe_attention_policy_v1> policy_span(
        policies, policy_count);
    if (!std::all_of(policy_span.begin(), policy_span.end(), valid_policy)) {
      throw std::invalid_argument(
          "native OLMoE head-wise attention policy is invalid");
    }
    return new engram::OLMoETokenRuntime(
        native_config(config, {}, policy_span, nullptr, {}));
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return nullptr;
  } catch (...) {
    error_text(
        error, error_capacity,
        "unknown native OLMoE head-wise token open failure");
    return nullptr;
  }
}

void* engram_olmoe_token_open_episodic_v1(
    const engram_olmoe_token_config* config,
    const engram_olmoe_episodic_policy_v1* policy, char* error,
    const std::size_t error_capacity) {
  try {
    if (policy == nullptr || policy->slots == 0 ||
        policy->span_size == 0 ||
        policy->slots % policy->span_size != 0) {
      throw std::invalid_argument(
          "native OLMoE episodic policy is invalid");
    }
    return new engram::OLMoETokenRuntime(
        native_config(config, {}, {}, policy, {}));
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return nullptr;
  } catch (...) {
    error_text(error, error_capacity,
               "unknown native OLMoE episodic token open failure");
    return nullptr;
  }
}

void* engram_olmoe_token_open_episodic_headwise_v1(
    const engram_olmoe_token_config* config,
    const engram_olmoe_episodic_policy_v1* policy,
    const std::uint8_t* head_mask, const std::size_t mask_count,
    char* error, const std::size_t error_capacity) {
  try {
    return open_episodic_headwise(
        config, policy, head_mask, mask_count, 0.0F);
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return nullptr;
  } catch (...) {
    error_text(
        error, error_capacity,
        "unknown native OLMoE head-gated episodic token open failure");
    return nullptr;
  }
}

void* engram_olmoe_token_open_episodic_headwise_v2(
    const engram_olmoe_token_config* config,
    const engram_olmoe_episodic_policy_v1* policy,
    const std::uint8_t* head_mask, const std::size_t mask_count,
    const float episodic_logit_bias, char* error,
    const std::size_t error_capacity) {
  try {
    return open_episodic_headwise(
        config, policy, head_mask, mask_count, episodic_logit_bias);
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return nullptr;
  } catch (...) {
    error_text(
        error, error_capacity,
        "unknown native OLMoE additive head-gated episodic token open failure");
    return nullptr;
  }
}

void* engram_olmoe_token_open_episodic_shadow_trace_v1(
    const engram_olmoe_token_config* config,
    const engram_olmoe_episodic_policy_v1* policy,
    const std::uint8_t* head_mask, const std::size_t mask_count,
    const float episodic_logit_bias,
    const engram_olmoe_attention_policy_v1* shadow_policy,
    char* error, const std::size_t error_capacity) {
  try {
    if (shadow_policy == nullptr) {
      throw std::invalid_argument(
          "native OLMoE shadow attention policy is null");
    }
    return open_episodic_headwise(
        config, policy, head_mask, mask_count, episodic_logit_bias,
        shadow_policy);
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return nullptr;
  } catch (...) {
    error_text(
        error, error_capacity,
        "unknown native OLMoE episodic shadow trace open failure");
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
    copy_forward_metrics(runtime->metrics(), metrics);
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

int engram_olmoe_token_forward_episodic_v1(
    void* handle, const int64_t* token_ids, const std::size_t length,
    const int32_t* write_slots, const int32_t* read_spans,
    int64_t* next_token, engram_olmoe_token_metrics* metrics, char* error,
    const std::size_t error_capacity) {
  try {
    auto* runtime = static_cast<engram::OLMoETokenRuntime*>(handle);
    if (runtime == nullptr || token_ids == nullptr ||
        write_slots == nullptr || read_spans == nullptr ||
        next_token == nullptr || length == 0) {
      throw std::invalid_argument(
          "native OLMoE episodic forward storage is invalid");
    }
    *next_token = runtime->forward_episodic(
        std::span<const std::int64_t>(token_ids, length),
        std::span<const std::int32_t>(write_slots, length),
        std::span<const std::int32_t>(read_spans, length));
    copy_forward_metrics(runtime->metrics(), metrics);
    return 0;
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    error_text(error, error_capacity,
               "unknown native OLMoE episodic forward failure");
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

int engram_olmoe_token_copy_attention_metrics_v1(
    const void* handle, engram_olmoe_attention_metrics_v1* metrics,
    char* error, const std::size_t error_capacity) {
  try {
    const auto* runtime =
        static_cast<const engram::OLMoETokenRuntime*>(handle);
    if (runtime == nullptr || metrics == nullptr) {
      throw std::invalid_argument(
          "native OLMoE attention metric storage is invalid");
    }
    const auto& source = runtime->metrics();
    metrics->logical_read_bytes = source.attention_logical_read_bytes;
    metrics->state_bytes = source.attention_state_bytes;
    metrics->scratch_bytes = source.attention_scratch_bytes;
    metrics->eviction_events = source.attention_eviction_events;
    metrics->older_candidate_entries_scored =
        source.attention_older_candidate_entries_scored;
    metrics->older_selected_entries =
        source.attention_older_selected_entries;
    metrics->sink_insertions = source.attention_sink_insertions;
    metrics->heavy_hitter_updates =
        source.attention_heavy_hitter_updates;
    return 0;
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    error_text(error, error_capacity,
               "unknown native OLMoE attention metric failure");
    return 1;
  }
}

int engram_olmoe_token_copy_episodic_metrics_v1(
    const void* handle, engram_olmoe_episodic_metrics_v1* metrics,
    char* error, const std::size_t error_capacity) {
  try {
    const auto* runtime =
        static_cast<const engram::OLMoETokenRuntime*>(handle);
    if (runtime == nullptr || metrics == nullptr) {
      throw std::invalid_argument(
          "native OLMoE episodic metric storage is invalid");
    }
    const auto& source = runtime->metrics();
    metrics->slots_written = source.episodic_slots_written;
    metrics->read_events = source.episodic_read_events;
    metrics->active_slots = source.episodic_active_slots;
    metrics->entries_read = source.episodic_entries_read;
    metrics->write_bytes = source.episodic_write_bytes;
    metrics->key_read_bytes = source.episodic_key_read_bytes;
    metrics->value_read_bytes = source.episodic_value_read_bytes;
    metrics->duplicate_older_entries_suppressed =
        source.episodic_duplicate_older_entries_suppressed;
    metrics->state_bytes = source.attention_state_bytes;
    metrics->scratch_bytes = source.attention_scratch_bytes;
    return 0;
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    error_text(error, error_capacity,
               "unknown native OLMoE episodic metric failure");
    return 1;
  }
}

int engram_olmoe_token_copy_last_shadow_trace_v1(
    const void* handle, float* input_norm,
    const std::size_t input_count, float* base_projected,
    const std::size_t base_count, float* target_residual,
    const std::size_t target_count, char* error,
    const std::size_t error_capacity) {
  try {
    const auto* runtime =
        static_cast<const engram::OLMoETokenRuntime*>(handle);
    if (runtime == nullptr || !runtime->has_shadow_trace() ||
        input_norm == nullptr || base_projected == nullptr ||
        target_residual == nullptr ||
        input_count != runtime->last_shadow_input_norm().size() ||
        base_count != runtime->last_shadow_base_projected().size() ||
        target_count !=
            runtime->last_shadow_target_residual().size()) {
      throw std::invalid_argument(
          "native OLMoE shadow trace storage is invalid");
    }
    std::copy(runtime->last_shadow_input_norm().begin(),
              runtime->last_shadow_input_norm().end(), input_norm);
    std::copy(runtime->last_shadow_base_projected().begin(),
              runtime->last_shadow_base_projected().end(),
              base_projected);
    std::copy(runtime->last_shadow_target_residual().begin(),
              runtime->last_shadow_target_residual().end(),
              target_residual);
    return 0;
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    error_text(error, error_capacity,
               "unknown native OLMoE shadow trace copy failure");
    return 1;
  }
}

int engram_olmoe_token_copy_last_episodic_mass_trace_v1(
    const void* handle, float* base_pre_wo,
    const std::size_t base_pre_wo_count, float* regular_component,
    const std::size_t regular_component_count,
    float* episodic_component,
    const std::size_t episodic_component_count, float* regular_mass,
    const std::size_t regular_mass_count, float* episodic_mass,
    const std::size_t episodic_mass_count, float* shadow_source_mass,
    const std::size_t shadow_source_mass_count, char* error,
    const std::size_t error_capacity) {
  try {
    const auto* runtime =
        static_cast<const engram::OLMoETokenRuntime*>(handle);
    if (runtime == nullptr || !runtime->has_episodic_mass_trace() ||
        base_pre_wo == nullptr || regular_component == nullptr ||
        episodic_component == nullptr || regular_mass == nullptr ||
        episodic_mass == nullptr || shadow_source_mass == nullptr ||
        base_pre_wo_count !=
            runtime->last_episodic_mass_base_pre_wo().size() ||
        regular_component_count !=
            runtime->last_episodic_mass_regular_component().size() ||
        episodic_component_count !=
            runtime->last_episodic_mass_episodic_component().size() ||
        regular_mass_count !=
            runtime->last_episodic_mass_regular_mass().size() ||
        episodic_mass_count !=
            runtime->last_episodic_mass_episodic_mass().size() ||
        shadow_source_mass_count !=
            runtime->last_episodic_mass_shadow_source_mass().size()) {
      throw std::invalid_argument(
          "native OLMoE episodic mass trace storage is invalid");
    }
    std::copy(runtime->last_episodic_mass_base_pre_wo().begin(),
              runtime->last_episodic_mass_base_pre_wo().end(),
              base_pre_wo);
    std::copy(
        runtime->last_episodic_mass_regular_component().begin(),
        runtime->last_episodic_mass_regular_component().end(),
        regular_component);
    std::copy(
        runtime->last_episodic_mass_episodic_component().begin(),
        runtime->last_episodic_mass_episodic_component().end(),
        episodic_component);
    std::copy(runtime->last_episodic_mass_regular_mass().begin(),
              runtime->last_episodic_mass_regular_mass().end(),
              regular_mass);
    std::copy(runtime->last_episodic_mass_episodic_mass().begin(),
              runtime->last_episodic_mass_episodic_mass().end(),
              episodic_mass);
    std::copy(
        runtime->last_episodic_mass_shadow_source_mass().begin(),
        runtime->last_episodic_mass_shadow_source_mass().end(),
        shadow_source_mass);
    return 0;
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    error_text(error, error_capacity,
               "unknown native OLMoE episodic mass trace copy failure");
    return 1;
  }
}

int engram_olmoe_token_copy_last_episodic_slot_trace_v1(
    const void* handle, float* slot_mass,
    const std::size_t slot_mass_count, float* slot_values,
    const std::size_t slot_value_count, char* error,
    const std::size_t error_capacity) {
  try {
    const auto* runtime =
        static_cast<const engram::OLMoETokenRuntime*>(handle);
    if (runtime == nullptr || !runtime->has_episodic_slot_trace() ||
        slot_mass == nullptr || slot_values == nullptr ||
        slot_mass_count != runtime->last_episodic_slot_mass().size() ||
        slot_value_count !=
            runtime->last_episodic_slot_values().size()) {
      throw std::invalid_argument(
          "native OLMoE episodic slot trace storage is invalid");
    }
    std::copy(runtime->last_episodic_slot_mass().begin(),
              runtime->last_episodic_slot_mass().end(), slot_mass);
    std::copy(runtime->last_episodic_slot_values().begin(),
              runtime->last_episodic_slot_values().end(), slot_values);
    return 0;
  } catch (const std::exception& exception) {
    error_text(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    error_text(error, error_capacity,
               "unknown native OLMoE episodic slot trace copy failure");
    return 1;
  }
}

}  // extern "C"
