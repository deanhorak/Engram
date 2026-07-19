#include "engram/semantic_quantized.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

bool close(const float left, const float right, const float tolerance = 1e-6F) {
  return std::abs(left - right) <= tolerance;
}

int fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}

struct Fixture {
  std::vector<std::uint8_t> gate_codes = {
      4, 2, 3, 3, 2, 4, 0, 2};
  std::vector<std::uint8_t> up_codes = {
      2, 4, 4, 2, 4, 2, 4, 2};
  std::array<float, 2> offsets = {-1.0F, -1.0F};
  std::array<float, 2> scales = {0.5F, 0.5F};
  std::vector<std::uint8_t> value_codes = {
      0, 3, 1, 0, 2, 1, 3, 2};
  std::vector<float> value_codebooks = {
      1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
      0.0F, 0.0F, 0.0F,
      0.5F, 0.0F, 0.0F,
      0.0F, 0.5F, 0.0F,
      0.0F, 0.0F, 0.5F,
      0.0F, 0.0F, 0.0F};

  engram::QuantizedSemanticLayerView view() const {
    return {4,
            2,
            3,
            2,
            4,
            gate_codes.data(),
            offsets.data(),
            scales.data(),
            up_codes.data(),
            offsets.data(),
            scales.data(),
            value_codes.data(),
            value_codebooks.data()};
  }
};

std::vector<float> decode_keys(const std::vector<std::uint8_t>& codes,
                               const std::array<float, 2>& offsets,
                               const std::array<float, 2>& scales) {
  std::vector<float> decoded(codes.size());
  for (std::size_t index = 0; index < codes.size(); ++index) {
    const std::size_t column = index % offsets.size();
    decoded[index] =
        offsets[column] + static_cast<float>(codes[index]) * scales[column];
  }
  return decoded;
}

std::vector<float> decode_values(const Fixture& fixture) {
  constexpr std::size_t records = 4;
  constexpr std::size_t output_size = 3;
  constexpr std::size_t codebooks = 2;
  constexpr std::size_t codebook_size = 4;
  std::vector<float> decoded(records * output_size, 0.0F);
  for (std::size_t record = 0; record < records; ++record) {
    for (std::size_t column = 0; column < output_size; ++column) {
      for (std::size_t stage = 0; stage < codebooks; ++stage) {
        const std::size_t code =
            fixture.value_codes[record * codebooks + stage];
        decoded[record * output_size + column] +=
            fixture.value_codebooks[
                (stage * codebook_size + code) * output_size + column];
      }
    }
  }
  return decoded;
}

}  // namespace

int main() {
  Fixture fixture;
  const engram::QuantizedSemanticLayerView layer = fixture.view();
  const std::array<float, 2> hidden = {1.0F, 0.0F};

  // Decode independently and require the quantized path to match the float
  // reference's ranking, scores, and accumulation.
  const std::vector<float> gate =
      decode_keys(fixture.gate_codes, fixture.offsets, fixture.scales);
  const std::vector<float> up =
      decode_keys(fixture.up_codes, fixture.offsets, fixture.scales);
  const std::vector<float> values = decode_values(fixture);
  const engram::SemanticLayerView float_layer{
      4, 2, 3, gate.data(), up.data(), values.data()};

  engram::QuantizedSemanticScratch quantized_scratch(4);
  engram::SemanticScratch float_scratch(4);
  std::array<float, 3> quantized_output{};
  std::array<float, 3> float_output{};
  std::array<engram::SemanticRecordResult, 2> quantized_selected{};
  std::array<engram::SemanticRecordResult, 2> float_selected{};
  engram::SemanticReadMetrics metrics;
  engram::semantic_read_quantized_scalar(
      layer, hidden, 4, 2, quantized_output, quantized_selected,
      quantized_scratch, &metrics);
  engram::semantic_read_scalar(float_layer, hidden, 4, 2, float_output,
                               float_selected, float_scratch);

  for (std::size_t rank = 0; rank < quantized_selected.size(); ++rank) {
    if (quantized_selected[rank].index != float_selected[rank].index ||
        !close(quantized_selected[rank].proxy_score,
               float_selected[rank].proxy_score) ||
        !close(quantized_selected[rank].activation,
               float_selected[rank].activation) ||
        !close(quantized_selected[rank].exact_score,
               float_selected[rank].exact_score)) {
      return fail("quantized semantic ranking differs from decoded float reference");
    }
  }
  for (std::size_t column = 0; column < quantized_output.size(); ++column) {
    if (!close(quantized_output[column], float_output[column])) {
      return fail("quantized semantic output differs from decoded float reference");
    }
  }
  if (quantized_selected[0].index != 1 || quantized_selected[1].index != 3) {
    return fail("decoded value-norm exact rerank mismatch");
  }
  if (metrics.proxy_records != 4 || metrics.candidate_records != 4 ||
      metrics.active_records != 2 || metrics.proxy_key_bytes != 48 ||
      metrics.exact_key_bytes != 48 || metrics.exact_value_bytes != 104 ||
      metrics.active_value_bytes != 52 || metrics.total_bytes_read != 252 ||
      metrics.zero_norm_query) {
    return fail("quantized semantic byte metrics mismatch");
  }

  // Zero queries and exact-score ties retain deterministic routing order.
  const std::array<float, 2> zero = {0.0F, 0.0F};
  std::array<engram::SemanticRecordResult, 2> tied{};
  const std::size_t scratch_bytes = quantized_scratch.persistent_bytes();
  engram::semantic_read_quantized_scalar(
      layer, zero, 2, 2, quantized_output, tied, quantized_scratch, &metrics);
  if (tied[0].index != 0 || tied[1].index != 1 ||
      !metrics.zero_norm_query || metrics.proxy_key_bytes != 0 ||
      metrics.exact_key_bytes != 40 || metrics.exact_value_bytes != 52 ||
      metrics.active_value_bytes != 52 || metrics.total_bytes_read != 144 ||
      scratch_bytes != quantized_scratch.persistent_bytes()) {
    return fail("zero-query tie, metrics, or scratch reuse mismatch");
  }

  // With all decoded values tied at zero, exact ranking must preserve proxy
  // order, including the stronger record 1 preceding lower record indices.
  std::vector<float> zero_codebooks(fixture.value_codebooks.size(), 0.0F);
  engram::QuantizedSemanticLayerView tie_layer = layer;
  tie_layer.value_codebooks = zero_codebooks.data();
  engram::semantic_read_quantized_scalar(
      tie_layer, hidden, 4, 2, quantized_output, tied, quantized_scratch);
  if (tied[0].index != 1 || tied[1].index != 0) {
    return fail("stable exact tie did not preserve quantized proxy order");
  }

  // Views are non-owning: changing a referenced codeword is visible without
  // rebuilding either the view or its scratch storage.
  fixture.value_codebooks[3] = 2.0F;
  const engram::QuantizedSemanticLayerView mutable_layer = fixture.view();
  engram::semantic_read_quantized_scalar(
      mutable_layer, hidden, 4, 1, quantized_output, quantized_selected,
      quantized_scratch);
  if (!close(quantized_output[0],
             2.5F * quantized_selected[0].activation)) {
    return fail("quantized semantic view unexpectedly owns codebook storage");
  }

  try {
    engram::QuantizedSemanticScratch too_small(3);
    engram::semantic_read_quantized_scalar(
        layer, hidden, 4, 2, quantized_output, quantized_selected, too_small);
    return fail("undersized quantized semantic scratch was accepted");
  } catch (const std::invalid_argument&) {
  }
  try {
    engram::QuantizedSemanticLayerView invalid = layer;
    const std::array<float, 2> invalid_scales = {0.5F, 0.0F};
    invalid.gate_scales = invalid_scales.data();
    engram::semantic_read_quantized_scalar(
        invalid, hidden, 4, 2, quantized_output, quantized_selected,
        quantized_scratch);
    return fail("non-positive affine scale was accepted");
  } catch (const std::invalid_argument&) {
  }
  try {
    engram::QuantizedSemanticLayerView invalid = layer;
    std::vector<std::uint8_t> invalid_codes = fixture.value_codes;
    invalid_codes[0] = 4;
    invalid.value_codes = invalid_codes.data();
    engram::semantic_read_quantized_scalar(
        invalid, hidden, 4, 2, quantized_output, quantized_selected,
        quantized_scratch);
    return fail("out-of-range additive value code was accepted");
  } catch (const std::invalid_argument&) {
  }
  try {
    engram::QuantizedSemanticLayerView invalid = layer;
    std::vector<float> invalid_codebooks = fixture.value_codebooks;
    invalid_codebooks[0] = std::numeric_limits<float>::quiet_NaN();
    invalid.value_codebooks = invalid_codebooks.data();
    engram::semantic_read_quantized_scalar(
        invalid, hidden, 4, 2, quantized_output, quantized_selected,
        quantized_scratch);
    return fail("non-finite additive codeword was accepted");
  } catch (const std::invalid_argument&) {
  }

  return 0;
}
