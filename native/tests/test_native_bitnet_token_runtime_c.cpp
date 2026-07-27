#include "engram/native_bitnet_token_runtime_c.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

int fail(const std::string& message) {
  std::cerr << message << '\n';
  return 1;
}

engram_native_bitnet_token_info_v1 empty_info() {
  engram_native_bitnet_token_info_v1 result{};
  result.abi_version = ENGRAM_NATIVE_BITNET_TOKEN_ABI_V1;
  result.struct_size = sizeof(result);
  return result;
}

engram_native_bitnet_token_metrics_v1 empty_metrics() {
  engram_native_bitnet_token_metrics_v1 result{};
  result.abi_version = ENGRAM_NATIVE_BITNET_TOKEN_ABI_V1;
  result.struct_size = sizeof(result);
  return result;
}

bool includes_eos(const engram_native_bitnet_token_info_v1& info,
                  const std::int64_t token) {
  return std::find(info.eos_token_ids,
                   info.eos_token_ids + info.eos_token_count,
                   token) != info.eos_token_ids + info.eos_token_count;
}

bool replay_metrics_match(
    const engram_native_bitnet_token_metrics_v1& left,
    const engram_native_bitnet_token_metrics_v1& right) {
  return left.prompt_tokens == right.prompt_tokens &&
         left.generated_tokens == right.generated_tokens &&
         left.positions_processed == right.positions_processed &&
         left.stage_calls == right.stage_calls &&
         left.semantic_calls == right.semantic_calls &&
         left.semantic_rows == right.semantic_rows &&
         left.semantic_selected_records ==
             right.semantic_selected_records &&
         left.semantic_kernel_cache_line_bytes ==
             right.semantic_kernel_cache_line_bytes &&
         left.semantic_global_metadata_bytes ==
             right.semantic_global_metadata_bytes &&
         left.semantic_cache_line_bytes == right.semantic_cache_line_bytes &&
         left.semantic_maximum_scratch_bytes ==
             right.semantic_maximum_scratch_bytes &&
         left.attention_logical_read_bytes ==
             right.attention_logical_read_bytes &&
         left.attention_state_bytes == right.attention_state_bytes &&
         left.attention_scratch_bytes == right.attention_scratch_bytes &&
         left.attention_eviction_events ==
             right.attention_eviction_events &&
         left.attention_older_candidate_entries_scored ==
             right.attention_older_candidate_entries_scored &&
         left.attention_older_selected_entries ==
             right.attention_older_selected_entries &&
         left.attention_sink_insertions ==
             right.attention_sink_insertions &&
         left.attention_heavy_hitter_updates ==
             right.attention_heavy_hitter_updates &&
         left.stopped_on_eos == right.stopped_on_eos;
}

}  // namespace

int main() {
  if (engram_native_bitnet_token_abi_version_v1() !=
      ENGRAM_NATIVE_BITNET_TOKEN_ABI_V1) {
    return fail("native token ABI version mismatch");
  }
  engram_native_bitnet_token_destroy_v1(nullptr);

  char error[512] = {};
  engram_native_bitnet_token_handle* handle =
      reinterpret_cast<engram_native_bitnet_token_handle*>(1);
  engram_native_bitnet_token_config_v1 config{
      .abi_version = ENGRAM_NATIVE_BITNET_TOKEN_ABI_V1,
      .struct_size = sizeof(config),
      .package_path = "/does/not/exist",
      .threads = 1,
      .flags = 0,
      .reserved = {},
  };
  if (engram_native_bitnet_token_create_v1(
          &config, nullptr, error, sizeof(error)) !=
          ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT ||
      error[0] == '\0') {
    return fail("null create output was not rejected");
  }
  if (engram_native_bitnet_token_create_v1(
          &config, &handle, error, sizeof(error)) !=
          ENGRAM_NATIVE_BITNET_TOKEN_AUTHENTICATION_FAILED ||
      handle != nullptr || error[0] == '\0') {
    return fail("unauthenticated package root was accepted");
  }

  config.abi_version = 0;
  if (engram_native_bitnet_token_create_v1(
          &config, &handle, error, sizeof(error)) !=
      ENGRAM_NATIVE_BITNET_TOKEN_ABI_MISMATCH) {
    return fail("wrong config ABI was accepted");
  }
  config.abi_version = ENGRAM_NATIVE_BITNET_TOKEN_ABI_V1;
  config.struct_size -= 1;
  if (engram_native_bitnet_token_create_v1(
          &config, &handle, error, sizeof(error)) !=
      ENGRAM_NATIVE_BITNET_TOKEN_ABI_MISMATCH) {
    return fail("wrong config size was accepted");
  }
  config.struct_size = sizeof(config);
  config.flags = 1;
  char tiny_error[4] = {'x', 'x', 'x', 'x'};
  if (engram_native_bitnet_token_create_v1(
          &config, &handle, tiny_error, sizeof(tiny_error)) !=
          ENGRAM_NATIVE_BITNET_TOKEN_ABI_MISMATCH ||
      tiny_error[3] != '\0') {
    return fail("flags or bounded error termination was not enforced");
  }
  config.flags = 0;
  config.reserved[2] = 1;
  if (engram_native_bitnet_token_create_v1(
          &config, &handle, error, sizeof(error)) !=
      ENGRAM_NATIVE_BITNET_TOKEN_ABI_MISMATCH) {
    return fail("nonzero config reserved field was accepted");
  }
  config.reserved[2] = 0;
  config.threads = 257;
  if (engram_native_bitnet_token_create_v1(
          &config, &handle, error, sizeof(error)) !=
      ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT) {
    return fail("oversized thread count was accepted");
  }

  engram_native_bitnet_token_info_v1 info = empty_info();
  if (engram_native_bitnet_token_get_info_v1(
          nullptr, &info, error, sizeof(error)) !=
      ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT) {
    return fail("null info handle was accepted");
  }
  info = empty_info();
  info.struct_size -= 1;
  if (engram_native_bitnet_token_get_info_v1(
          nullptr, &info, error, sizeof(error)) !=
      ENGRAM_NATIVE_BITNET_TOKEN_ABI_MISMATCH) {
    return fail("wrong info size was accepted");
  }
  if (engram_native_bitnet_token_reset_v1(nullptr, error, sizeof(error)) !=
      ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT) {
    return fail("null reset handle was accepted");
  }
  std::int64_t prompt = 128000;
  std::int64_t output = -7;
  std::uint64_t output_count = 99;
  engram_native_bitnet_token_metrics_v1 metrics = empty_metrics();
  if (engram_native_bitnet_token_generate_v1(
          nullptr, &prompt, 1, 1, &output, 1, &output_count, &metrics, error,
          sizeof(error)) != ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT ||
      output_count != 0 || output != -7 ||
      metrics.abi_version != ENGRAM_NATIVE_BITNET_TOKEN_ABI_V1 ||
      metrics.prompt_tokens != 0) {
    return fail("null generation validation mutated caller output");
  }

  const std::filesystem::path package(
      ENGRAM_NATIVE_BITNET_TOKEN_TEST_PACKAGE);
  if (!std::filesystem::is_directory(package)) {
    std::cout << "native token production package absent; lifecycle test "
                 "skipped\n";
    return 0;
  }

  const std::string package_text = package.string();
  config.package_path = package_text.c_str();
  config.threads = 0;
  error[0] = 'x';
  if (engram_native_bitnet_token_create_v1(
          &config, &handle, error, sizeof(error)) !=
          ENGRAM_NATIVE_BITNET_TOKEN_OK ||
      handle == nullptr || error[0] != '\0') {
    return fail(std::string("authenticated create failed: ") + error);
  }

  info = empty_info();
  if (engram_native_bitnet_token_get_info_v1(
          handle, &info, error, sizeof(error)) !=
          ENGRAM_NATIVE_BITNET_TOKEN_OK ||
      info.max_position_embeddings != 4096 ||
      info.vocabulary_size != 128256 || info.layers != 30 ||
      info.hidden_size != 2560 || info.intermediate_size != 6912 ||
      info.query_heads != 20 || info.key_value_heads != 5 ||
      info.head_dimension != 128 || info.local_window != 16 ||
      info.older_candidates != 8 || info.older_top_k != 4 ||
      info.sink_tokens != 2 || info.thread_count != 12 ||
      info.eos_token_count < 2 || !includes_eos(info, 128001) ||
      !includes_eos(info, 128009) ||
      std::string(info.semantic_backend) !=
          "native_bitnet_dynamic_input_pruning_v2" ||
      std::string(info.package_manifest_sha256) !=
          "707bbe069ef6892ce9bfe98258f3289e28af15a400922e950c4386f56dd26926") {
    engram_native_bitnet_token_destroy_v1(handle);
    return fail("authenticated runtime info mismatch");
  }

  output = -7;
  output_count = 99;
  metrics = empty_metrics();
  if (engram_native_bitnet_token_generate_v1(
          handle, &prompt, 1, 2, &output, 1, &output_count, &metrics, error,
          sizeof(error)) != ENGRAM_NATIVE_BITNET_TOKEN_BUFFER_TOO_SMALL ||
      output_count != 0 || output != -7) {
    engram_native_bitnet_token_destroy_v1(handle);
    return fail("short caller output buffer was not fail-closed");
  }
  const std::int64_t invalid_prompt = -1;
  if (engram_native_bitnet_token_generate_v1(
          handle, &invalid_prompt, 1, 1, &output, 1, &output_count, nullptr,
          error, sizeof(error)) !=
      ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT) {
    engram_native_bitnet_token_destroy_v1(handle);
    return fail("invalid prompt token was accepted");
  }
  std::vector<std::int64_t> oversized_output(4097);
  if (engram_native_bitnet_token_generate_v1(
          handle, &prompt, 1, 4097, oversized_output.data(),
          oversized_output.size(), &output_count, nullptr, error,
          sizeof(error)) != ENGRAM_NATIVE_BITNET_TOKEN_INVALID_ARGUMENT) {
    engram_native_bitnet_token_destroy_v1(handle);
    return fail("context-overflowing generation was accepted");
  }

  const std::vector<std::int64_t> boundary_prompt(17, prompt);
  output = -7;
  output_count = 0;
  metrics = empty_metrics();
  if (engram_native_bitnet_token_generate_v1(
          handle, boundary_prompt.data(), boundary_prompt.size(), 1, &output,
          1, &output_count, &metrics, error, sizeof(error)) !=
          ENGRAM_NATIVE_BITNET_TOKEN_OK ||
      output_count != 1 || output < 0 ||
      static_cast<std::uint64_t>(output) >= info.vocabulary_size ||
      metrics.prompt_tokens != boundary_prompt.size() ||
      metrics.generated_tokens != 1 ||
      metrics.positions_processed != boundary_prompt.size() ||
      metrics.stage_calls != info.layers ||
      metrics.semantic_calls != info.layers ||
      metrics.semantic_rows != info.layers * boundary_prompt.size() ||
      metrics.semantic_selected_records == 0 ||
      metrics.semantic_kernel_cache_line_bytes == 0 ||
      metrics.semantic_global_metadata_bytes == 0 ||
      metrics.semantic_cache_line_bytes !=
          metrics.semantic_kernel_cache_line_bytes +
              metrics.semantic_global_metadata_bytes ||
      metrics.semantic_maximum_scratch_bytes == 0 ||
      metrics.attention_logical_read_bytes == 0 ||
      metrics.attention_state_bytes == 0 ||
      metrics.attention_scratch_bytes == 0 ||
      metrics.qkv_projection_ns == 0 || metrics.rope_ns == 0 ||
      metrics.native_attention_ns == 0 ||
      metrics.output_projection_ns == 0 ||
      metrics.attention_elapsed_ns !=
          metrics.qkv_projection_ns + metrics.rope_ns +
              metrics.native_attention_ns + metrics.output_projection_ns ||
      metrics.call_elapsed_ns < metrics.prefill_elapsed_ns ||
      metrics.prefill_elapsed_ns == 0 || metrics.decode_elapsed_ns != 0 ||
      metrics.attention_eviction_events != info.layers ||
      metrics.attention_older_candidate_entries_scored !=
          info.layers * info.query_heads ||
      metrics.attention_older_selected_entries !=
          info.layers * info.query_heads ||
      metrics.attention_sink_insertions != info.layers * info.query_heads ||
      metrics.attention_heavy_hitter_updates != 0 ||
      metrics.stopped_on_eos !=
          static_cast<std::uint32_t>(includes_eos(info, output))) {
    engram_native_bitnet_token_destroy_v1(handle);
    return fail(std::string("native one-token generation mismatch: ") +
                error);
  }
  const std::int64_t first_output = output;
  const engram_native_bitnet_token_metrics_v1 first_metrics = metrics;

  output = -7;
  output_count = 99;
  metrics = empty_metrics();
  if (engram_native_bitnet_token_generate_v1(
          handle, &prompt, 1, 1, &output, 1, &output_count, &metrics, error,
          sizeof(error)) != ENGRAM_NATIVE_BITNET_TOKEN_INVALID_STATE ||
      output_count != 0 || output != -7) {
    engram_native_bitnet_token_destroy_v1(handle);
    return fail("completed runtime accepted generation without reset");
  }
  if (engram_native_bitnet_token_reset_v1(
          handle, error, sizeof(error)) != ENGRAM_NATIVE_BITNET_TOKEN_OK ||
      error[0] != '\0') {
    engram_native_bitnet_token_destroy_v1(handle);
    return fail("native runtime reset failed");
  }

  output = -7;
  output_count = 0;
  metrics = empty_metrics();
  if (engram_native_bitnet_token_generate_v1(
          handle, boundary_prompt.data(), boundary_prompt.size(), 1, &output,
          1, &output_count, &metrics, error, sizeof(error)) !=
          ENGRAM_NATIVE_BITNET_TOKEN_OK ||
      output_count != 1 || output != first_output ||
      !replay_metrics_match(first_metrics, metrics)) {
    engram_native_bitnet_token_destroy_v1(handle);
    return fail("native runtime reset replay mismatch");
  }

  engram_native_bitnet_token_destroy_v1(handle);
  return 0;
}
