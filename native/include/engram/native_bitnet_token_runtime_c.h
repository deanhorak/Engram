#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(ENGRAM_NATIVE_BITNET_TOKEN_BUILDING)
#define ENGRAM_NATIVE_BITNET_TOKEN_API __declspec(dllexport)
#else
#define ENGRAM_NATIVE_BITNET_TOKEN_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define ENGRAM_NATIVE_BITNET_TOKEN_API \
  __attribute__((visibility("default")))
#else
#define ENGRAM_NATIVE_BITNET_TOKEN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ENGRAM_NATIVE_BITNET_TOKEN_ABI_V1 UINT32_C(1)
#define ENGRAM_NATIVE_BITNET_TOKEN_MAX_EOS_IDS_V1 UINT32_C(8)
#define ENGRAM_NATIVE_BITNET_TOKEN_BACKEND_CAPACITY_V1 UINT32_C(64)
#define ENGRAM_NATIVE_BITNET_TOKEN_SHA256_CAPACITY_V1 UINT32_C(65)

typedef int32_t engram_native_bitnet_token_status;

enum {
  ENGRAM_NATIVE_BITNET_TOKEN_OK = 0,
  ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT = 1,
  ENGRAM_NATIVE_BITNET_TOKEN_ABI_MISMATCH = 2,
  ENGRAM_NATIVE_BITNET_TOKEN_AUTHENTICATION_FAILED = 3,
  ENGRAM_NATIVE_BITNET_TOKEN_BUFFER_TOO_SMALL = 4,
  ENGRAM_NATIVE_BITNET_TOKEN_INVALID_STATE = 5,
  ENGRAM_NATIVE_BITNET_TOKEN_RUNTIME_FAILED = 6,
  ENGRAM_NATIVE_BITNET_TOKEN_INTERNAL_FAILED = 7,
};

typedef struct engram_native_bitnet_token_handle
    engram_native_bitnet_token_handle;

// package_path is borrowed only for create_v1. The runtime owns all mapped
// model state after successful creation. flags and reserved fields must be
// zero. threads=0 selects the authenticated package default; otherwise the
// supported range is 1..256.
typedef struct engram_native_bitnet_token_config_v1 {
  uint32_t abi_version;
  uint32_t struct_size;
  const char* package_path;
  uint32_t threads;
  uint32_t flags;
  uint64_t reserved[4];
} engram_native_bitnet_token_config_v1;

// Static authenticated package/runtime information. Callers initialize
// abi_version and struct_size before get_info_v1. All other fields are output.
typedef struct engram_native_bitnet_token_info_v1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint64_t max_position_embeddings;
  uint64_t vocabulary_size;
  uint64_t layers;
  uint64_t hidden_size;
  uint64_t intermediate_size;
  uint64_t query_heads;
  uint64_t key_value_heads;
  uint64_t head_dimension;
  uint64_t local_window;
  uint64_t older_candidates;
  uint64_t older_top_k;
  uint64_t sink_tokens;
  uint32_t thread_count;
  uint32_t eos_token_count;
  float rms_norm_epsilon;
  float rope_theta;
  int64_t eos_token_ids[ENGRAM_NATIVE_BITNET_TOKEN_MAX_EOS_IDS_V1];
  char semantic_backend[ENGRAM_NATIVE_BITNET_TOKEN_BACKEND_CAPACITY_V1];
  char package_manifest_sha256[ENGRAM_NATIVE_BITNET_TOKEN_SHA256_CAPACITY_V1];
  uint64_t reserved[4];
} engram_native_bitnet_token_info_v1;

// Per-generation metrics. Byte/time counters cover the complete call after a
// fresh create/reset. Callers initialize abi_version and struct_size before
// generate_v1. Timing counters are observational and need not replay exactly.
typedef struct engram_native_bitnet_token_metrics_v1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint64_t prompt_tokens;
  uint64_t generated_tokens;
  uint64_t positions_processed;
  uint64_t stage_calls;
  uint64_t semantic_calls;
  uint64_t semantic_rows;
  uint64_t semantic_selected_records;
  uint64_t semantic_kernel_cache_line_bytes;
  uint64_t semantic_global_metadata_bytes;
  uint64_t semantic_cache_line_bytes;
  uint64_t semantic_maximum_scratch_bytes;
  uint64_t attention_logical_read_bytes;
  uint64_t attention_state_bytes;
  uint64_t attention_scratch_bytes;
  uint64_t qkv_projection_ns;
  uint64_t rope_ns;
  uint64_t native_attention_ns;
  uint64_t output_projection_ns;
  uint64_t semantic_elapsed_ns;
  uint64_t attention_elapsed_ns;
  uint64_t call_elapsed_ns;
  uint64_t prefill_elapsed_ns;
  uint64_t decode_elapsed_ns;
  uint32_t stopped_on_eos;
  uint32_t reserved32;
  uint64_t reserved[4];
} engram_native_bitnet_token_metrics_v1;

ENGRAM_NATIVE_BITNET_TOKEN_API uint32_t
engram_native_bitnet_token_abi_version_v1(void);

// Production creation accepts only an authenticated package root and always
// uses the compiled deployment trust root. No artifact path, digest, semantic
// policy, attention policy, or EOS override exists in this ABI.
ENGRAM_NATIVE_BITNET_TOKEN_API engram_native_bitnet_token_status
engram_native_bitnet_token_create_v1(
    const engram_native_bitnet_token_config_v1* config,
    engram_native_bitnet_token_handle** output, char* error,
    size_t error_capacity);

// A null handle is accepted. The caller must not race destroy with any other
// operation on the same handle.
ENGRAM_NATIVE_BITNET_TOKEN_API void
engram_native_bitnet_token_destroy_v1(
    engram_native_bitnet_token_handle* handle);

ENGRAM_NATIVE_BITNET_TOKEN_API engram_native_bitnet_token_status
engram_native_bitnet_token_reset_v1(
    engram_native_bitnet_token_handle* handle, char* error,
    size_t error_capacity);

ENGRAM_NATIVE_BITNET_TOKEN_API engram_native_bitnet_token_status
engram_native_bitnet_token_get_info_v1(
    engram_native_bitnet_token_handle* handle,
    engram_native_bitnet_token_info_v1* info, char* error,
    size_t error_capacity);

// V1 deliberately permits one generation after create/reset. A second call
// fails with INVALID_STATE until reset, preventing accidental duplicate
// re-prefill. output_capacity must be at least max_new_tokens; all arrays are
// caller-owned and output_count is set to zero before validation/execution.
ENGRAM_NATIVE_BITNET_TOKEN_API engram_native_bitnet_token_status
engram_native_bitnet_token_generate_v1(
    engram_native_bitnet_token_handle* handle, const int64_t* prompt_tokens,
    uint64_t prompt_count, uint64_t max_new_tokens, int64_t* output_tokens,
    uint64_t output_capacity, uint64_t* output_count,
    engram_native_bitnet_token_metrics_v1* metrics, char* error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif

