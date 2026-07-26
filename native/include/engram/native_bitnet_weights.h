#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "engram/safetensors.h"
#include "engram/ternary_projection.h"

namespace engram {

struct NativeBitNetLayerWeights {
  std::span<const std::uint16_t> input_norm;
  std::span<const std::uint16_t> post_attention_norm;
  std::span<const std::uint16_t> attention_sub_norm;
  std::size_t query_projection;
  std::size_t key_projection;
  std::size_t value_projection;
  std::size_t output_projection;
};

class NativeBitNetWeights {
 public:
  NativeBitNetWeights(const std::filesystem::path& non_mlp_safetensors,
                      std::size_t layers, std::size_t hidden_size,
                      std::size_t query_heads, std::size_t key_value_heads,
                      std::size_t head_dimension, std::size_t threads);

  [[nodiscard]] std::span<const std::uint16_t> embedding() const {
    return embedding_;
  }
  [[nodiscard]] std::span<const std::uint16_t> final_norm() const {
    return final_norm_;
  }
  [[nodiscard]] const std::vector<NativeBitNetLayerWeights>& layers() const {
    return layers_;
  }
  [[nodiscard]] TernaryProjectionKernel& projections() {
    return projections_;
  }
  [[nodiscard]] std::size_t vocabulary_size() const noexcept {
    return vocabulary_size_;
  }
  [[nodiscard]] std::size_t hidden_size() const noexcept {
    return hidden_size_;
  }
  [[nodiscard]] std::size_t mapped_bytes() const noexcept {
    return tensors_.mapped_bytes();
  }

 private:
  SafetensorFile tensors_;
  TernaryProjectionKernel projections_;
  std::span<const std::uint16_t> embedding_;
  std::span<const std::uint16_t> final_norm_;
  std::vector<NativeBitNetLayerWeights> layers_;
  std::size_t vocabulary_size_ = 0;
  std::size_t hidden_size_ = 0;
};

}  // namespace engram
