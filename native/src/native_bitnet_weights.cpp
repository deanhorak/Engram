#include "engram/native_bitnet_weights.h"

#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace engram {
namespace {

float bf16_to_float(const std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::span<const std::uint16_t> bf16_vector(const SafetensorFile& file,
                                           const std::string& name,
                                           const std::size_t width) {
  const SafetensorView tensor = file.tensor(name);
  if (tensor.shape != std::vector<std::size_t>{width}) {
    throw SafetensorError(name + " has an invalid norm shape");
  }
  return tensor.bf16();
}

std::size_t add_projection(TernaryProjectionKernel& kernel,
                           const SafetensorFile& file,
                           const std::string& prefix,
                           const std::size_t input,
                           const std::size_t output) {
  const SafetensorView weight = file.tensor(prefix + ".weight");
  const SafetensorView scale = file.tensor(prefix + ".weight_scale");
  if (weight.shape != std::vector<std::size_t>{output / 4, input} ||
      scale.shape != std::vector<std::size_t>{1}) {
    throw SafetensorError(prefix + " has an invalid packed projection shape");
  }
  return kernel.add_mapped(weight.uint8(), input, output,
                           bf16_to_float(scale.bf16()[0]));
}

}  // namespace

NativeBitNetWeights::NativeBitNetWeights(
    const std::filesystem::path& non_mlp_safetensors,
    const std::size_t layers, const std::size_t hidden_size,
    const std::size_t query_heads, const std::size_t key_value_heads,
    const std::size_t head_dimension, const std::size_t threads)
    : tensors_(load_safetensors(non_mlp_safetensors)),
      projections_(threads),
      hidden_size_(hidden_size) {
  if (layers == 0 || hidden_size == 0 || query_heads == 0 ||
      key_value_heads == 0 || head_dimension == 0 ||
      query_heads * head_dimension != hidden_size) {
    throw std::invalid_argument("native BitNet weight dimensions are invalid");
  }
  const SafetensorView embedding =
      tensors_.tensor("model.embed_tokens.weight");
  if (embedding.shape.size() != 2 || embedding.shape[1] != hidden_size) {
    throw SafetensorError("model embedding has an invalid shape");
  }
  embedding_ = embedding.bf16();
  vocabulary_size_ = embedding.shape[0];
  final_norm_ = bf16_vector(tensors_, "model.norm.weight", hidden_size);
  const std::size_t kv_width = key_value_heads * head_dimension;
  layers_.reserve(layers);
  for (std::size_t layer = 0; layer < layers; ++layer) {
    const std::string base = "model.layers." + std::to_string(layer);
    const std::string attention = base + ".self_attn";
    layers_.push_back(NativeBitNetLayerWeights{
        .input_norm =
            bf16_vector(tensors_, base + ".input_layernorm.weight",
                        hidden_size),
        .post_attention_norm =
            bf16_vector(tensors_, base + ".post_attention_layernorm.weight",
                        hidden_size),
        .attention_sub_norm =
            bf16_vector(tensors_, attention + ".attn_sub_norm.weight",
                        hidden_size),
        .query_projection = add_projection(
            projections_, tensors_, attention + ".q_proj", hidden_size,
            hidden_size),
        .key_projection = add_projection(
            projections_, tensors_, attention + ".k_proj", hidden_size,
            kv_width),
        .value_projection = add_projection(
            projections_, tensors_, attention + ".v_proj", hidden_size,
            kv_width),
        .output_projection = add_projection(
            projections_, tensors_, attention + ".o_proj", hidden_size,
            hidden_size),
    });
  }
}

}  // namespace engram
