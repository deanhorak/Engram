#include "engram/native_bitnet_token_runtime.h"
#include "engram/package.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::size_t positive_size(const std::string_view text,
                          const std::string_view name,
                          const std::size_t maximum) {
  std::uint64_t parsed = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (text.empty() || result.ec != std::errc{} ||
      result.ptr != text.data() + text.size() || parsed == 0 ||
      parsed > maximum ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(std::string(name) +
                                " must be a positive bounded integer");
  }
  return static_cast<std::size_t>(parsed);
}

std::int64_t token_id(const std::string_view text) {
  std::uint64_t parsed = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (text.empty() || result.ec != std::errc{} ||
      result.ptr != text.data() + text.size() ||
      parsed > static_cast<std::uint64_t>(
                   std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument(
        "prompt token ids must be non-negative 64-bit integers");
  }
  return static_cast<std::int64_t>(parsed);
}

bool zero_metrics(const engram::NativeBitNetTokenMetrics& metrics) {
  return metrics.positions_processed == 0 && metrics.stage_calls == 0 &&
         metrics.semantic_calls == 0 && metrics.semantic_rows == 0 &&
         metrics.semantic_elapsed_ns == 0 &&
         metrics.semantic_kernel_cache_line_bytes == 0 &&
         metrics.semantic_global_metadata_bytes == 0 &&
         metrics.semantic_scheduled_cache_line_bytes == 0 &&
         metrics.semantic_selected_records == 0 &&
         metrics.attention_eviction_events == 0 &&
         metrics.attention_older_candidate_entries_scored == 0 &&
         metrics.attention_older_selected_entries == 0 &&
         metrics.attention_sink_insertions == 0 &&
         metrics.attention_heavy_hitter_updates == 0 &&
         metrics.attention_elapsed_ns == 0;
}

bool matching_structural_metrics(
    const engram::NativeBitNetTokenMetrics& left,
    const engram::NativeBitNetTokenMetrics& right) {
  return left.positions_processed == right.positions_processed &&
         left.stage_calls == right.stage_calls &&
         left.semantic_calls == right.semantic_calls &&
         left.semantic_rows == right.semantic_rows &&
         left.semantic_kernel_cache_line_bytes ==
             right.semantic_kernel_cache_line_bytes &&
         left.semantic_global_metadata_bytes ==
             right.semantic_global_metadata_bytes &&
         left.semantic_scheduled_cache_line_bytes ==
             right.semantic_scheduled_cache_line_bytes &&
         left.semantic_selected_records == right.semantic_selected_records &&
         left.attention_eviction_events ==
             right.attention_eviction_events &&
         left.attention_older_candidate_entries_scored ==
             right.attention_older_candidate_entries_scored &&
         left.attention_older_selected_entries ==
             right.attention_older_selected_entries &&
         left.attention_sink_insertions ==
             right.attention_sink_insertions &&
         left.attention_heavy_hitter_updates ==
             right.attention_heavy_hitter_updates;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: engram-bitnet-token-generate PACKAGE MAX_NEW THREADS "
                 "[--verify-reset] [--enable-recurrent-correction] "
                 "[--controller-directory PATH] TOKEN [TOKEN ...]\n";
    return 2;
  }
  try {
    const std::filesystem::path package(argv[1]);
    constexpr std::size_t kMaximumTokenBudget = 1024 * 1024;
    constexpr std::size_t kMaximumThreads = 256;
    const std::size_t max_new =
        positive_size(argv[2], "MAX_NEW", kMaximumTokenBudget);
    const std::size_t threads =
        positive_size(argv[3], "THREADS", kMaximumThreads);
    int token_start = 4;
    bool verify_reset = false;
    bool enable_recurrent_correction = false;
    std::filesystem::path controller_override;
    while (token_start < argc) {
      const std::string argument(argv[token_start]);
      if (argument == "--verify-reset") {
        verify_reset = true;
        ++token_start;
      } else if (argument == "--enable-recurrent-correction") {
        enable_recurrent_correction = true;
        ++token_start;
      } else if (argument == "--controller-directory") {
        if (++token_start >= argc) {
          throw std::invalid_argument(
              "--controller-directory requires a path");
        }
        controller_override = argv[token_start++];
      } else {
        break;
      }
    }
    if (!controller_override.empty() && !enable_recurrent_correction) {
      throw std::invalid_argument(
          "controller overrides require --enable-recurrent-correction");
    }
    if (token_start == argc) {
      throw std::invalid_argument("native generation prompt is empty");
    }
    std::vector<std::int64_t> prompt;
    for (int index = token_start; index < argc; ++index) {
      prompt.push_back(token_id(argv[index]));
    }

    // This preflight hashes an exact, symlink-free inventory and anchors the
    // semantic promotion in immutable reviewed digests before any large model
    // mapping or thread-pool construction occurs.
    const engram::NativeBitNetDIPPackageMetadata metadata =
        engram::load_native_bitnet_dip_package(package);
    if (std::any_of(prompt.begin(), prompt.end(),
                    [&metadata](const std::int64_t token) {
                      return static_cast<std::uint64_t>(token) >=
                             metadata.vocabulary_size;
                    })) {
      throw std::invalid_argument(
          "prompt token id is outside the authenticated vocabulary");
    }
    if (prompt.size() > metadata.max_position_embeddings ||
        max_new - 1 >
            metadata.max_position_embeddings - prompt.size()) {
      throw std::invalid_argument(
          "prompt and generation budget exceed authenticated context length");
    }

    const auto controller_directory = controller_override.empty()
                                          ? metadata.controller_directory
                                          : controller_override;
    if (!controller_override.empty()) {
      std::cerr
          << "warning: using an unauthenticated evaluator controller artifact"
          << '\n';
    }
    engram::NativeBitNetTokenRuntime runtime(
        engram::NativeBitNetTokenConfig{
            .non_mlp_safetensors = metadata.non_mlp_safetensors,
            .mlp_artifact = metadata.mlp_artifact,
            .dip_coordinate_index = metadata.dip_coordinate_index,
            .controller_directory = controller_directory,
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
            .enable_recurrent_correction = enable_recurrent_correction,
            .eos_token_ids = metadata.eos_token_ids,
        });
    const auto generated = runtime.generate(prompt, max_new);
    const auto first_metrics = runtime.metrics();
    bool reset_counters_zeroed = false;
    bool replay_tokens_match = false;
    bool replay_metrics_match = false;
    if (verify_reset) {
      runtime.reset();
      reset_counters_zeroed =
          runtime.position() == 0 && zero_metrics(runtime.metrics());
      const auto repeated = runtime.generate(prompt, max_new);
      replay_tokens_match = repeated == generated;
      replay_metrics_match =
          matching_structural_metrics(first_metrics, runtime.metrics());
    }
    const bool reset_verified =
        verify_reset && reset_counters_zeroed && replay_tokens_match &&
        replay_metrics_match;
    for (std::size_t index = 0; index < generated.size(); ++index) {
      if (index != 0) std::cout << ' ';
      std::cout << generated[index];
    }
    std::cout << '\n';
    const auto& metrics = first_metrics;
    std::cerr << "semantic_backend=" << runtime.semantic_backend()
              << " controller_mode="
              << (enable_recurrent_correction
                      ? "factorized_recurrent_correction_evaluator"
                      : "exact_operator_residual")
              << " positions=" << metrics.positions_processed
              << " stage_calls=" << metrics.stage_calls
              << " semantic_calls=" << metrics.semantic_calls
              << " semantic_rows=" << metrics.semantic_rows
              << " selected_records=" << metrics.semantic_selected_records
              << " semantic_kernel_cache_line_bytes="
              << metrics.semantic_kernel_cache_line_bytes
              << " semantic_global_metadata_bytes="
              << metrics.semantic_global_metadata_bytes
              << " semantic_cache_line_bytes="
              << metrics.semantic_scheduled_cache_line_bytes
              << " semantic_seconds="
              << static_cast<double>(metrics.semantic_elapsed_ns) / 1.0e9
              << " attention_seconds="
              << static_cast<double>(metrics.attention_elapsed_ns) / 1.0e9
              << " attention_evictions="
              << metrics.attention_eviction_events
              << " attention_older_candidates_scored="
              << metrics.attention_older_candidate_entries_scored
              << " attention_older_entries_selected="
              << metrics.attention_older_selected_entries
              << " attention_sink_insertions="
              << metrics.attention_sink_insertions
              << " attention_heavy_hitter_updates="
              << metrics.attention_heavy_hitter_updates
              << " reset_verified=" << static_cast<int>(reset_verified)
              << " reset_counters_zeroed="
              << static_cast<int>(reset_counters_zeroed)
              << " replay_metrics_match="
              << static_cast<int>(replay_metrics_match)
              << '\n';
    if (verify_reset && !reset_verified) {
      std::cerr << "native reset replay failed: tokens_match="
                << static_cast<int>(replay_tokens_match) << '\n';
      return 1;
    }
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
