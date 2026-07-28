#include "engram/olmoe_weights.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace engram {
namespace {

std::span<const std::uint16_t> bf16_tensor(
    const SafetensorFile& file, const std::string& name,
    const std::vector<std::size_t>& shape) {
  const SafetensorView tensor = file.tensor(name);
  if (tensor.shape != shape) {
    throw SafetensorError(name + " has an invalid shape");
  }
  return tensor.bf16();
}

}  // namespace

OLMoEWeights::OLMoEWeights(
    const std::filesystem::path& non_mlp_safetensors,
    const std::size_t layers, const std::size_t hidden_size,
    const std::size_t key_value_width)
    : tensors_(load_safetensors(non_mlp_safetensors)) {
  if (layers == 0 || hidden_size == 0 || key_value_width == 0) {
    throw std::invalid_argument("native OLMoE weight dimensions are invalid");
  }
  const SafetensorView embedding =
      tensors_.tensor("model.embed_tokens.weight");
  if (embedding.shape.size() != 2 || embedding.shape[1] != hidden_size) {
    throw SafetensorError("OLMoE embedding has an invalid shape");
  }
  vocabulary_size_ = embedding.shape[0];
  embedding_ = embedding.bf16();
  final_norm_ =
      bf16_tensor(tensors_, "model.norm.weight", {hidden_size});
  language_head_ = bf16_tensor(
      tensors_, "lm_head.weight", {vocabulary_size_, hidden_size});
  layers_.reserve(layers);
  for (std::size_t layer = 0; layer < layers; ++layer) {
    const std::string base = "model.layers." + std::to_string(layer);
    const std::string attention = base + ".self_attn";
    layers_.push_back(OLMoELayerWeights{
        .input_norm = bf16_tensor(
            tensors_, base + ".input_layernorm.weight", {hidden_size}),
        .post_attention_norm = bf16_tensor(
            tensors_, base + ".post_attention_layernorm.weight",
            {hidden_size}),
        .query_norm = bf16_tensor(
            tensors_, attention + ".q_norm.weight", {hidden_size}),
        .key_norm = bf16_tensor(
            tensors_, attention + ".k_norm.weight", {key_value_width}),
        .query_projection = bf16_tensor(
            tensors_, attention + ".q_proj.weight",
            {hidden_size, hidden_size}),
        .key_projection = bf16_tensor(
            tensors_, attention + ".k_proj.weight",
            {key_value_width, hidden_size}),
        .value_projection = bf16_tensor(
            tensors_, attention + ".v_proj.weight",
            {key_value_width, hidden_size}),
        .output_projection = bf16_tensor(
            tensors_, attention + ".o_proj.weight",
            {hidden_size, hidden_size}),
    });
  }
  if (tensors_.tensor_count() != 3 + layers * 8) {
    throw SafetensorError("OLMoE non-MLP tensor inventory is not exact");
  }
}

}  // namespace engram
