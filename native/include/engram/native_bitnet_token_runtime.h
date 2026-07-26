#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "engram/native_bitnet.h"
#include "engram/native_bitnet_weights.h"
#include "engram/npy.h"
#include "engram/streaming_attention.h"

namespace engram {

struct NativeBitNetTokenConfig {
  std::filesystem::path non_mlp_safetensors;
  std::filesystem::path mlp_artifact;
  std::filesystem::path controller_directory;
  std::size_t layers = 0;
  std::size_t hidden_size = 0;
  std::size_t query_heads = 0;
  std::size_t key_value_heads = 0;
  std::size_t head_dimension = 0;
  std::size_t threads = 1;
  std::size_t local_window = 16;
  std::size_t older_candidates = 8;
  std::size_t older_top_k = 4;
  std::size_t sink_tokens = 2;
  float rms_norm_epsilon = 1.0e-5F;
  float rope_theta = 500000.0F;
  std::vector<std::int64_t> eos_token_ids;
};

struct NativeBitNetTokenMetrics {
  std::size_t positions_processed = 0;
  std::size_t stage_calls = 0;
  std::uint64_t semantic_elapsed_ns = 0;
  std::uint64_t attention_elapsed_ns = 0;
};

class NativeBitNetTokenRuntime {
 public:
  explicit NativeBitNetTokenRuntime(NativeBitNetTokenConfig config);

  NativeBitNetTokenRuntime(const NativeBitNetTokenRuntime&) = delete;
  NativeBitNetTokenRuntime& operator=(const NativeBitNetTokenRuntime&) = delete;

  [[nodiscard]] std::int64_t forward(std::span<const std::int64_t> token_ids);
  [[nodiscard]] std::vector<std::int64_t> generate(
      std::span<const std::int64_t> prompt, std::size_t max_new_tokens);
  void reset();

  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  [[nodiscard]] const NativeBitNetTokenMetrics& metrics() const noexcept {
    return metrics_;
  }

 private:
  [[nodiscard]] bool is_eos(std::int64_t token) const;

  NativeBitNetTokenConfig config_;
  NativeBitNetWeights weights_;
  NativeBitNetKernel semantic_;
  NpyArray operator_scales_;
  NpyArray correction_scales_;
  std::vector<std::unique_ptr<StreamingAttention>> attention_;
  std::size_t position_ = 0;
  NativeBitNetTokenMetrics metrics_;
};

}  // namespace engram
