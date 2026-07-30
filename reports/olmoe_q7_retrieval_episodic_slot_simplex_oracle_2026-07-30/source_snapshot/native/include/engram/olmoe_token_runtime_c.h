#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct engram_olmoe_token_config {
  const char* non_mlp_safetensors;
  const char* q7_artifact;
  size_t layers;
  size_t hidden_size;
  size_t query_heads;
  size_t key_value_heads;
  size_t head_dimension;
  size_t threads;
  size_t local_window;
  size_t older_candidates;
  size_t older_top_k;
  size_t sink_tokens;
  float rms_norm_epsilon;
  float rope_theta;
} engram_olmoe_token_config;

// Per-layer attention capacity policy for the additive layered-open ABI.
typedef struct engram_olmoe_attention_policy_v1 {
  size_t local_window;
  size_t older_candidates;
  size_t older_top_k;
  size_t sink_tokens;
} engram_olmoe_attention_policy_v1;

// Fixed-capacity causal K/V bank used by the additive episodic-open ABI.
typedef struct engram_olmoe_episodic_policy_v1 {
  size_t slots;
  size_t span_size;
} engram_olmoe_episodic_policy_v1;

typedef struct engram_olmoe_token_metrics {
  uint64_t positions_processed;
  uint64_t attention_weight_bytes;
  uint64_t q7_scheduled_bytes;
  uint64_t q7_elapsed_ns;
  uint64_t attention_state_bytes;
  uint64_t elapsed_ns;
} engram_olmoe_token_metrics;

// Additive, versioned attention diagnostics. Keeping this separate preserves
// the original forward-metrics ABI for already-built runtime libraries.
typedef struct engram_olmoe_attention_metrics_v1 {
  uint64_t logical_read_bytes;
  uint64_t state_bytes;
  uint64_t scratch_bytes;
  uint64_t eviction_events;
  uint64_t older_candidate_entries_scored;
  uint64_t older_selected_entries;
  uint64_t sink_insertions;
  uint64_t heavy_hitter_updates;
} engram_olmoe_attention_metrics_v1;

typedef struct engram_olmoe_episodic_metrics_v1 {
  // Layer-cache counts: only layers with at least one selected episodic head
  // allocate, write, and report read events.
  uint64_t slots_written;
  uint64_t read_events;
  uint64_t active_slots;
  // Selected-query-head entry count.
  uint64_t entries_read;
  // Full grouped K/V rows written by active layers.
  uint64_t write_bytes;
  // Selected-query-head read traffic. Duplicate suppression is likewise
  // restricted to selected heads.
  uint64_t key_read_bytes;
  uint64_t value_read_bytes;
  uint64_t duplicate_older_entries_suppressed;
  uint64_t state_bytes;
  uint64_t scratch_bytes;
} engram_olmoe_episodic_metrics_v1;

void* engram_olmoe_token_open(const engram_olmoe_token_config* config,
                              char* error, size_t error_capacity);
// Opens a runtime with exactly one policy entry per configured layer. This is
// additive: engram_olmoe_token_open retains its original scalar behavior and
// ABI.
void* engram_olmoe_token_open_layered_v1(
    const engram_olmoe_token_config* config,
    const engram_olmoe_attention_policy_v1* policies, size_t policy_count,
    char* error, size_t error_capacity);
// Opens a runtime with exactly layers * query_heads policies, flattened in
// layer-major/head-minor order. This version requires query_heads to equal
// key_value_heads so every policy owns one independent K/V cache.
void* engram_olmoe_token_open_headwise_v1(
    const engram_olmoe_token_config* config,
    const engram_olmoe_attention_policy_v1* policies, size_t policy_count,
    char* error, size_t error_capacity);
// Opens the scalar grouped-attention runtime with a causal BF16 episodic K/V
// bank. Existing open functions and their ABIs are unchanged.
void* engram_olmoe_token_open_episodic_v1(
    const engram_olmoe_token_config* config,
    const engram_olmoe_episodic_policy_v1* policy, char* error,
    size_t error_capacity);
// Opens the scalar grouped-attention runtime with a layer-major/query-head-
// minor 0/1 mask controlling which query heads may read episodic spans. At
// least one mask entry must be selected. Layers with no selected heads do not
// allocate or write an episodic bank.
void* engram_olmoe_token_open_episodic_headwise_v1(
    const engram_olmoe_token_config* config,
    const engram_olmoe_episodic_policy_v1* policy,
    const uint8_t* head_mask, size_t mask_count, char* error,
    size_t error_capacity);
// Additive extension of the head-gated episodic open. The bias is applied to
// selected episodic logits before the joint attention softmax. V1 remains the
// exact zero-bias route and all existing structs retain their layouts.
void* engram_olmoe_token_open_episodic_headwise_v2(
    const engram_olmoe_token_config* config,
    const engram_olmoe_episodic_policy_v1* policy,
    const uint8_t* head_mask, size_t mask_count,
    float episodic_logit_bias, char* error, size_t error_capacity);
// Evaluator-only extension of the scalar grouped episodic V2 base. The shadow
// policy owns one independent non-episodic cache per layer, consumes the exact
// same post-RoPE Q/K/V rows as the base, and never changes base outputs.
// Shadow state, scratch, and traffic are deliberately excluded from all base
// inference metrics returned by this ABI. Wall-clock elapsed_ns still measures
// the complete instrumented evaluator call, including shadow execution.
void* engram_olmoe_token_open_episodic_shadow_trace_v1(
    const engram_olmoe_token_config* config,
    const engram_olmoe_episodic_policy_v1* policy,
    const uint8_t* head_mask, size_t mask_count,
    float episodic_logit_bias,
    const engram_olmoe_attention_policy_v1* shadow_policy,
    char* error, size_t error_capacity);
void engram_olmoe_token_close(void* handle);
void engram_olmoe_token_reset(void* handle);
size_t engram_olmoe_token_vocabulary_size(const void* handle);
size_t engram_olmoe_token_position(const void* handle);
int engram_olmoe_token_forward(void* handle, const int64_t* token_ids,
                               size_t length, int64_t* next_token,
                               engram_olmoe_token_metrics* metrics,
                               char* error, size_t error_capacity);
int engram_olmoe_token_forward_episodic_v1(
    void* handle, const int64_t* token_ids, size_t length,
    const int32_t* write_slots, const int32_t* read_spans,
    int64_t* next_token, engram_olmoe_token_metrics* metrics, char* error,
    size_t error_capacity);
int engram_olmoe_token_copy_last_diagnostics(
    const void* handle, float* final_state, size_t final_state_count,
    float* vocabulary_scores, size_t vocabulary_score_count, char* error,
    size_t error_capacity);
int engram_olmoe_token_copy_attention_metrics_v1(
    const void* handle, engram_olmoe_attention_metrics_v1* metrics,
    char* error, size_t error_capacity);
int engram_olmoe_token_copy_episodic_metrics_v1(
    const void* handle, engram_olmoe_episodic_metrics_v1* metrics,
    char* error, size_t error_capacity);
// Copies the last row at every layer in layer-major [layers, hidden] order.
// Every count must equal layers * hidden exactly. A reset invalidates the
// trace until the next successful forward.
int engram_olmoe_token_copy_last_shadow_trace_v1(
    const void* handle, float* input_norm, size_t input_count,
    float* base_projected, size_t base_count,
    float* target_residual, size_t target_count,
    char* error, size_t error_capacity);
// Additive evaluator-only trace available on handles opened through
// engram_olmoe_token_open_episodic_shadow_trace_v1. Pre-Wo value tensors are
// layer-major [layers, hidden]. Mass tensors are layer-major
// [layers, query_heads]. The regular/episodic components use the exact joint
// beta-configured softmax denominator. The final mass is the W128 shadow mass
// on the exact source positions named by the last row's episodic read.
int engram_olmoe_token_copy_last_episodic_mass_trace_v1(
    const void* handle,
    float* base_pre_wo, size_t base_pre_wo_count,
    float* regular_component, size_t regular_component_count,
    float* episodic_component, size_t episodic_component_count,
    float* regular_mass, size_t regular_mass_count,
    float* episodic_mass, size_t episodic_mass_count,
    float* shadow_source_mass, size_t shadow_source_mass_count,
    char* error, size_t error_capacity);
// Additive evaluator-only per-slot trace for the same last-row capture as the
// mass trace above. Slot masses use layer-major
// [layers, query_heads, span_size] order. Stored BF16 values are expanded to
// float and use [layers, query_heads, span_size, head_dimension] order.
// Inactive layers and masked heads are exactly zero.
int engram_olmoe_token_copy_last_episodic_slot_trace_v1(
    const void* handle, float* slot_mass, size_t slot_mass_count,
    float* slot_values, size_t slot_value_count, char* error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif
