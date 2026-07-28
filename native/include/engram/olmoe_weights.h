#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "engram/safetensors.h"

namespace engram {

struct OLMoELayerWeights {
  std::span<const std::uint16_t> input_norm;
  std::span<const std::uint16_t> post_attention_norm;
  std::span<const std::uint16_t> query_norm;
  std::span<const std::uint16_t> key_norm;
  std::span<const std::uint16_t> query_projection;
  std::span<const std::uint16_t> key_projection;
  std::span<const std::uint16_t> value_projection;
  std::span<const std::uint16_t> output_projection;
};

class OLMoEWeights {
 public:
  OLMoEWeights(const std::filesystem::path& non_mlp_safetensors,
               std::size_t layers, std::size_t hidden_size,
               std::size_t key_value_width);

  [[nodiscard]] std::span<const std::uint16_t> embedding() const {
    return embedding_;
  }
  [[nodiscard]] std::span<const std::uint16_t> final_norm() const {
    return final_norm_;
  }
  [[nodiscard]] std::span<const std::uint16_t> language_head() const {
    return language_head_;
  }
  [[nodiscard]] const std::vector<OLMoELayerWeights>& layers() const {
    return layers_;
  }
  [[nodiscard]] std::size_t vocabulary_size() const noexcept {
    return vocabulary_size_;
  }
  [[nodiscard]] std::size_t mapped_bytes() const noexcept {
    return tensors_.mapped_bytes();
  }

 private:
  SafetensorFile tensors_;
  std::span<const std::uint16_t> embedding_;
  std::span<const std::uint16_t> final_norm_;
  std::span<const std::uint16_t> language_head_;
  std::vector<OLMoELayerWeights> layers_;
  std::size_t vocabulary_size_{};
};

}  // namespace engram
