#include "engram/semantic_quantized.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace engram {
namespace {

using ScratchRecord = QuantizedSemanticScratch::ScratchRecord;

bool better_proxy(const ScratchRecord& left,
                  const ScratchRecord& right) noexcept {
  if (left.proxy_score > right.proxy_score) {
    return true;
  }
  if (left.proxy_score < right.proxy_score) {
    return false;
  }
  return left.index < right.index;
}

bool better_exact(const ScratchRecord& left,
                  const ScratchRecord& right) noexcept {
  if (left.exact_score > right.exact_score) {
    return true;
  }
  if (left.exact_score < right.exact_score) {
    return false;
  }
  return left.candidate_order < right.candidate_order;
}

double stable_silu(const double value) noexcept {
  const double exponential = std::exp(-std::abs(value));
  const double sigmoid =
      value >= 0.0 ? 1.0 / (1.0 + exponential)
                   : exponential / (1.0 + exponential);
  return value * sigmoid;
}

std::size_t checked_product(const std::size_t left, const std::size_t right,
                            const char* message) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::invalid_argument(message);
  }
  return left * right;
}

std::size_t checked_sum(const std::size_t left, const std::size_t right,
                        const char* message) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::invalid_argument(message);
  }
  return left + right;
}

void validate_layer(const QuantizedSemanticLayerView& layer) {
  if (layer.records == 0 || layer.hidden_size == 0 ||
      layer.output_size == 0 || layer.value_codebook_count == 0 ||
      layer.value_codebook_size == 0) {
    throw std::invalid_argument(
        "quantized semantic layer dimensions must be positive");
  }
  if (layer.records > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("semantic record count exceeds uint32 range");
  }
  if (layer.value_codebook_size > 256) {
    throw std::invalid_argument(
        "uint8 value codes support at most 256 codewords");
  }
  if (layer.gate_codes == nullptr || layer.gate_offsets == nullptr ||
      layer.gate_scales == nullptr || layer.up_codes == nullptr ||
      layer.up_offsets == nullptr || layer.up_scales == nullptr ||
      layer.value_codes == nullptr || layer.value_codebooks == nullptr) {
    throw std::invalid_argument(
        "quantized semantic layer pointers must not be null");
  }
  static_cast<void>(checked_product(layer.records, layer.hidden_size,
                                    "semantic key dimensions overflow"));
  static_cast<void>(checked_product(
      layer.records, layer.value_codebook_count,
      "semantic value-code dimensions overflow"));
  const std::size_t codewords = checked_product(
      layer.value_codebook_count, layer.value_codebook_size,
      "semantic value-codebook dimensions overflow");
  static_cast<void>(checked_product(
      codewords, layer.output_size,
      "semantic value-codebook dimensions overflow"));

  for (std::size_t column = 0; column < layer.hidden_size; ++column) {
    if (!std::isfinite(layer.gate_offsets[column]) ||
        !std::isfinite(layer.up_offsets[column]) ||
        !std::isfinite(layer.gate_scales[column]) ||
        !std::isfinite(layer.up_scales[column]) ||
        layer.gate_scales[column] <= 0.0F ||
        layer.up_scales[column] <= 0.0F) {
      throw std::invalid_argument(
          "semantic affine offsets must be finite and scales positive");
    }
  }
}

float decode_key(const std::uint8_t code, const float offset,
                 const float scale) noexcept {
  // Keeping this expression in float mirrors materializing a decoded float32
  // tensor before performing the semantic read.
  return offset + static_cast<float>(code) * scale;
}

float decode_value_element(const QuantizedSemanticLayerView& layer,
                           const std::size_t record,
                           const std::size_t column) {
  const std::size_t code_offset = record * layer.value_codebook_count;
  float decoded = 0.0F;
  for (std::size_t stage = 0; stage < layer.value_codebook_count; ++stage) {
    const std::size_t code = layer.value_codes[code_offset + stage];
    if (code >= layer.value_codebook_size) {
      throw std::invalid_argument(
          "semantic value code is outside its codebook");
    }
    const std::size_t codeword = stage * layer.value_codebook_size + code;
    const float component =
        layer.value_codebooks[codeword * layer.output_size + column];
    if (!std::isfinite(component)) {
      throw std::invalid_argument(
          "semantic value codebooks must contain finite values");
    }
    decoded += component;
  }
  if (!std::isfinite(decoded)) {
    throw std::invalid_argument(
        "decoded semantic values must contain finite values");
  }
  return decoded;
}

void reset_metrics(SemanticReadMetrics* metrics) noexcept {
  if (metrics != nullptr) {
    *metrics = SemanticReadMetrics{};
  }
}

std::size_t key_bytes(const std::size_t records,
                      const std::size_t hidden_size) {
  const std::size_t code_elements = checked_product(
      checked_product(records, hidden_size, "semantic byte metric overflows"),
      2, "semantic byte metric overflows");
  const std::size_t codec_elements = checked_product(
      hidden_size, 4, "semantic byte metric overflows");
  const std::size_t codec_bytes = checked_product(
      codec_elements, sizeof(float), "semantic byte metric overflows");
  return checked_sum(code_elements, codec_bytes,
                     "semantic byte metric overflows");
}

std::size_t value_bytes(const QuantizedSemanticLayerView& layer,
                        const std::size_t records) {
  const std::size_t codes = checked_product(
      records, layer.value_codebook_count,
      "semantic value byte metric overflows");
  const std::size_t components_per_record = checked_product(
      layer.value_codebook_count, layer.output_size,
      "semantic value byte metric overflows");
  const std::size_t components = checked_product(
      records, components_per_record,
      "semantic value byte metric overflows");
  const std::size_t codebook_bytes = checked_product(
      components, sizeof(float), "semantic value byte metric overflows");
  return checked_sum(codes, codebook_bytes,
                     "semantic value byte metric overflows");
}

}  // namespace

QuantizedSemanticScratch::QuantizedSemanticScratch(
    const std::size_t record_capacity)
    : proxy_ranking_(record_capacity), exact_ranking_(record_capacity) {
  if (record_capacity == 0 ||
      record_capacity > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(
        "quantized semantic scratch capacity must be positive and fit uint32");
  }
}

std::size_t QuantizedSemanticScratch::capacity() const noexcept {
  return proxy_ranking_.size();
}

std::size_t QuantizedSemanticScratch::persistent_bytes() const noexcept {
  return (proxy_ranking_.capacity() + exact_ranking_.capacity()) *
         sizeof(ScratchRecord);
}

void semantic_read_quantized_scalar(
    const QuantizedSemanticLayerView& layer,
    const std::span<const float> hidden, const std::size_t candidate_count,
    const std::size_t top_k, const std::span<float> output,
    const std::span<SemanticRecordResult> selected,
    QuantizedSemanticScratch& scratch, SemanticReadMetrics* const metrics) {
  validate_layer(layer);
  reset_metrics(metrics);
  if (hidden.size() != layer.hidden_size) {
    throw std::invalid_argument("hidden width does not match semantic layer");
  }
  if (!std::all_of(hidden.begin(), hidden.end(),
                   [](const float value) { return std::isfinite(value); })) {
    throw std::invalid_argument("hidden must contain finite values");
  }
  if (candidate_count == 0 || candidate_count > layer.records) {
    throw std::invalid_argument(
        "candidate_count must be positive and within semantic records");
  }
  if (top_k == 0 || top_k > candidate_count || selected.size() < top_k) {
    throw std::invalid_argument(
        "top_k must be positive, no larger than candidates, and fit selected output");
  }
  if (output.size() < layer.output_size) {
    throw std::invalid_argument("semantic output span is too small");
  }
  if (scratch.capacity() < layer.records ||
      scratch.exact_ranking_.size() < layer.records) {
    throw std::invalid_argument("quantized semantic scratch capacity is too small");
  }

  double hidden_squared_norm = 0.0;
  for (const float value : hidden) {
    hidden_squared_norm += static_cast<double>(value) * value;
  }
  const double hidden_norm = std::sqrt(hidden_squared_norm);
  for (std::size_t record = 0; record < layer.records; ++record) {
    double proxy_score = 0.0;
    if (hidden_norm > 0.0) {
      const std::size_t key_offset = record * layer.hidden_size;
      double gate_dot = 0.0;
      double up_dot = 0.0;
      double gate_squared_norm = 0.0;
      double up_squared_norm = 0.0;
      for (std::size_t column = 0; column < layer.hidden_size; ++column) {
        const float gate = decode_key(layer.gate_codes[key_offset + column],
                                      layer.gate_offsets[column],
                                      layer.gate_scales[column]);
        const float up = decode_key(layer.up_codes[key_offset + column],
                                    layer.up_offsets[column],
                                    layer.up_scales[column]);
        if (!std::isfinite(gate) || !std::isfinite(up)) {
          throw std::invalid_argument(
              "decoded semantic keys must contain finite values");
        }
        gate_dot += static_cast<double>(gate) * hidden[column];
        up_dot += static_cast<double>(up) * hidden[column];
        gate_squared_norm += static_cast<double>(gate) * gate;
        up_squared_norm += static_cast<double>(up) * up;
      }
      const double gate_norm = std::sqrt(gate_squared_norm);
      const double up_norm = std::sqrt(up_squared_norm);
      const double gate_alignment =
          gate_norm > 0.0 ? gate_dot / (gate_norm * hidden_norm) : 0.0;
      const double up_alignment =
          up_norm > 0.0 ? up_dot / (up_norm * hidden_norm) : 0.0;
      proxy_score = std::max(gate_alignment, 0.0) * std::abs(up_alignment);
    }
    scratch.proxy_ranking_[record] = ScratchRecord{
        static_cast<std::uint32_t>(record), 0, proxy_score, 0.0, 0.0};
  }
  std::partial_sort(scratch.proxy_ranking_.begin(),
                    scratch.proxy_ranking_.begin() + candidate_count,
                    scratch.proxy_ranking_.begin() + layer.records,
                    better_proxy);

  for (std::size_t order = 0; order < candidate_count; ++order) {
    const ScratchRecord proxy = scratch.proxy_ranking_[order];
    const std::size_t record = proxy.index;
    const std::size_t key_offset = record * layer.hidden_size;
    double gate_dot = 0.0;
    double up_dot = 0.0;
    for (std::size_t column = 0; column < layer.hidden_size; ++column) {
      const float gate = decode_key(layer.gate_codes[key_offset + column],
                                    layer.gate_offsets[column],
                                    layer.gate_scales[column]);
      const float up = decode_key(layer.up_codes[key_offset + column],
                                  layer.up_offsets[column],
                                  layer.up_scales[column]);
      if (!std::isfinite(gate) || !std::isfinite(up)) {
        throw std::invalid_argument(
            "decoded semantic keys must contain finite values");
      }
      gate_dot += static_cast<double>(gate) * hidden[column];
      up_dot += static_cast<double>(up) * hidden[column];
    }
    const double activation = stable_silu(gate_dot) * up_dot;
    double value_squared_norm = 0.0;
    for (std::size_t column = 0; column < layer.output_size; ++column) {
      const float value = decode_value_element(layer, record, column);
      value_squared_norm += static_cast<double>(value) * value;
    }
    scratch.exact_ranking_[order] = ScratchRecord{
        proxy.index, static_cast<std::uint32_t>(order), proxy.proxy_score,
        activation, std::abs(activation) * std::sqrt(value_squared_norm)};
  }
  std::partial_sort(scratch.exact_ranking_.begin(),
                    scratch.exact_ranking_.begin() + top_k,
                    scratch.exact_ranking_.begin() + candidate_count,
                    better_exact);

  std::fill_n(output.begin(), layer.output_size, 0.0F);
  for (std::size_t rank = 0; rank < top_k; ++rank) {
    const ScratchRecord record = scratch.exact_ranking_[rank];
    selected[rank] = SemanticRecordResult{
        record.index, static_cast<float>(record.proxy_score),
        static_cast<float>(record.activation),
        static_cast<float>(record.exact_score)};
    for (std::size_t column = 0; column < layer.output_size; ++column) {
      const float value = decode_value_element(layer, record.index, column);
      output[column] +=
          static_cast<float>(record.activation * static_cast<double>(value));
    }
  }

  if (metrics != nullptr) {
    metrics->proxy_records = layer.records;
    metrics->candidate_records = candidate_count;
    metrics->active_records = top_k;
    metrics->zero_norm_query = hidden_norm == 0.0;
    metrics->proxy_key_bytes =
        hidden_norm == 0.0 ? 0 : key_bytes(layer.records, layer.hidden_size);
    metrics->exact_key_bytes = key_bytes(candidate_count, layer.hidden_size);
    metrics->exact_value_bytes = value_bytes(layer, candidate_count);
    metrics->active_value_bytes = value_bytes(layer, top_k);
    const std::size_t key_total = checked_sum(
        metrics->proxy_key_bytes, metrics->exact_key_bytes,
        "total byte metric overflows");
    const std::size_t rerank_total = checked_sum(
        key_total, metrics->exact_value_bytes,
        "total byte metric overflows");
    metrics->total_bytes_read = checked_sum(
        rerank_total, metrics->active_value_bytes,
        "total byte metric overflows");
  }
}

void semantic_read_quantized_candidates_scalar(
    const QuantizedSemanticLayerView& layer,
    const std::span<const float> hidden,
    const std::span<const std::uint32_t> candidate_indices,
    const std::span<const float> candidate_proxy_scores,
    const std::size_t top_k, const std::span<float> output,
    const std::span<SemanticRecordResult> selected,
    QuantizedSemanticScratch& scratch, SemanticReadMetrics* const metrics) {
  validate_layer(layer);
  reset_metrics(metrics);
  if (hidden.size() != layer.hidden_size ||
      !std::all_of(hidden.begin(), hidden.end(),
                   [](const float value) { return std::isfinite(value); })) {
    throw std::invalid_argument(
        "hidden must match semantic width and contain finite values");
  }
  const std::size_t candidate_count = candidate_indices.size();
  if (candidate_count == 0 || candidate_count > layer.records ||
      candidate_proxy_scores.size() != candidate_count) {
    throw std::invalid_argument(
        "semantic candidate indices and scores must be aligned and non-empty");
  }
  if (top_k == 0 || top_k > candidate_count || selected.size() < top_k ||
      output.size() < layer.output_size || scratch.capacity() < candidate_count) {
    throw std::invalid_argument(
        "semantic candidate read output or scratch capacity is invalid");
  }

  for (std::size_t order = 0; order < candidate_count; ++order) {
    const std::size_t record = candidate_indices[order];
    if (record >= layer.records ||
        !std::isfinite(candidate_proxy_scores[order])) {
      throw std::invalid_argument(
          "semantic candidate index or proxy score is invalid");
    }
    for (std::size_t earlier = 0; earlier < order; ++earlier) {
      if (candidate_indices[earlier] == record) {
        throw std::invalid_argument("semantic candidate indices must be unique");
      }
    }
    const std::size_t key_offset = record * layer.hidden_size;
    double gate_dot = 0.0;
    double up_dot = 0.0;
    for (std::size_t column = 0; column < layer.hidden_size; ++column) {
      const float gate = decode_key(layer.gate_codes[key_offset + column],
                                    layer.gate_offsets[column],
                                    layer.gate_scales[column]);
      const float up = decode_key(layer.up_codes[key_offset + column],
                                  layer.up_offsets[column],
                                  layer.up_scales[column]);
      if (!std::isfinite(gate) || !std::isfinite(up)) {
        throw std::invalid_argument(
            "decoded semantic keys must contain finite values");
      }
      gate_dot += static_cast<double>(gate) * hidden[column];
      up_dot += static_cast<double>(up) * hidden[column];
    }
    const double activation = stable_silu(gate_dot) * up_dot;
    double value_squared_norm = 0.0;
    for (std::size_t column = 0; column < layer.output_size; ++column) {
      const float value = decode_value_element(layer, record, column);
      value_squared_norm += static_cast<double>(value) * value;
    }
    scratch.exact_ranking_[order] = QuantizedSemanticScratch::ScratchRecord{
        static_cast<std::uint32_t>(record), static_cast<std::uint32_t>(order),
        candidate_proxy_scores[order], activation,
        std::abs(activation) * std::sqrt(value_squared_norm)};
  }
  std::partial_sort(scratch.exact_ranking_.begin(),
                    scratch.exact_ranking_.begin() + top_k,
                    scratch.exact_ranking_.begin() + candidate_count,
                    better_exact);
  std::fill_n(output.begin(), layer.output_size, 0.0F);
  for (std::size_t rank = 0; rank < top_k; ++rank) {
    const auto record = scratch.exact_ranking_[rank];
    selected[rank] = SemanticRecordResult{
        record.index, static_cast<float>(record.proxy_score),
        static_cast<float>(record.activation),
        static_cast<float>(record.exact_score)};
    for (std::size_t column = 0; column < layer.output_size; ++column) {
      output[column] += static_cast<float>(
          record.activation * decode_value_element(layer, record.index, column));
    }
  }
  if (metrics != nullptr) {
    metrics->candidate_records = candidate_count;
    metrics->active_records = top_k;
    metrics->exact_key_bytes = key_bytes(candidate_count, layer.hidden_size);
    metrics->exact_value_bytes = value_bytes(layer, candidate_count);
    metrics->active_value_bytes = value_bytes(layer, top_k);
    metrics->total_bytes_read = checked_sum(
        checked_sum(metrics->exact_key_bytes, metrics->exact_value_bytes,
                    "total byte metric overflows"),
        metrics->active_value_bytes, "total byte metric overflows");
  }
}

}  // namespace engram
