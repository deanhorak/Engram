#include "engram/semantic.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace engram {
namespace {

using ScratchRecord = SemanticScratch::ScratchRecord;

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

void validate_layer(const SemanticLayerView& layer) {
  if (layer.records == 0 || layer.hidden_size == 0 || layer.output_size == 0) {
    throw std::invalid_argument("semantic layer dimensions must be positive");
  }
  if (layer.records > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("semantic record count exceeds uint32 range");
  }
  if (layer.gate_keys == nullptr || layer.up_keys == nullptr ||
      layer.values == nullptr) {
    throw std::invalid_argument("semantic layer pointers must not be null");
  }
  static_cast<void>(checked_product(layer.records, layer.hidden_size,
                                    "semantic key dimensions overflow"));
  static_cast<void>(checked_product(layer.records, layer.output_size,
                                    "semantic value dimensions overflow"));
}

void reset_metrics(SemanticReadMetrics* metrics) noexcept {
  if (metrics != nullptr) {
    *metrics = SemanticReadMetrics{};
  }
}

}  // namespace

SemanticScratch::SemanticScratch(const std::size_t record_capacity)
    : proxy_ranking_(record_capacity), exact_ranking_(record_capacity) {
  if (record_capacity == 0 ||
      record_capacity > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(
        "semantic scratch capacity must be positive and fit uint32");
  }
}

std::size_t SemanticScratch::capacity() const noexcept {
  return proxy_ranking_.size();
}

std::size_t SemanticScratch::persistent_bytes() const noexcept {
  return (proxy_ranking_.capacity() + exact_ranking_.capacity()) *
         sizeof(ScratchRecord);
}

void semantic_read_scalar(
    const SemanticLayerView& layer, const std::span<const float> hidden,
    const std::size_t candidate_count, const std::size_t top_k,
    const std::span<float> output,
    const std::span<SemanticRecordResult> selected, SemanticScratch& scratch,
    SemanticReadMetrics* const metrics) {
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
    throw std::invalid_argument("semantic scratch capacity is too small");
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
        const float gate = layer.gate_keys[key_offset + column];
        const float up = layer.up_keys[key_offset + column];
        if (!std::isfinite(gate) || !std::isfinite(up)) {
          throw std::invalid_argument("semantic keys must contain finite values");
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
      const float gate = layer.gate_keys[key_offset + column];
      const float up = layer.up_keys[key_offset + column];
      if (!std::isfinite(gate) || !std::isfinite(up)) {
        throw std::invalid_argument("semantic keys must contain finite values");
      }
      gate_dot += static_cast<double>(gate) * hidden[column];
      up_dot += static_cast<double>(up) * hidden[column];
    }
    const double activation = stable_silu(gate_dot) * up_dot;
    const std::size_t value_offset = record * layer.output_size;
    double value_squared_norm = 0.0;
    for (std::size_t column = 0; column < layer.output_size; ++column) {
      const float value = layer.values[value_offset + column];
      if (!std::isfinite(value)) {
        throw std::invalid_argument("semantic values must contain finite values");
      }
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
    const std::size_t value_offset =
        static_cast<std::size_t>(record.index) * layer.output_size;
    for (std::size_t column = 0; column < layer.output_size; ++column) {
      output[column] += static_cast<float>(
          record.activation * layer.values[value_offset + column]);
    }
  }

  if (metrics != nullptr) {
    metrics->proxy_records = layer.records;
    metrics->candidate_records = candidate_count;
    metrics->active_records = top_k;
    metrics->zero_norm_query = hidden_norm == 0.0;
    const std::size_t two_hidden = checked_product(
        layer.hidden_size, 2, "semantic byte metric overflows");
    metrics->proxy_key_bytes =
        hidden_norm == 0.0
            ? 0
            : checked_product(
                  checked_product(layer.records, two_hidden,
                                  "proxy byte metric overflows"),
                  sizeof(float), "proxy byte metric overflows");
    metrics->exact_key_bytes = checked_product(
        checked_product(candidate_count, two_hidden,
                        "exact-key byte metric overflows"),
        sizeof(float), "exact-key byte metric overflows");
    metrics->exact_value_bytes = checked_product(
        checked_product(candidate_count, layer.output_size,
                        "exact-value byte metric overflows"),
        sizeof(float), "exact-value byte metric overflows");
    metrics->active_value_bytes = checked_product(
        checked_product(top_k, layer.output_size,
                        "active-value byte metric overflows"),
        sizeof(float), "active-value byte metric overflows");
    const std::size_t key_bytes = checked_sum(
        metrics->proxy_key_bytes, metrics->exact_key_bytes,
        "total byte metric overflows");
    const std::size_t rerank_bytes = checked_sum(
        key_bytes, metrics->exact_value_bytes,
        "total byte metric overflows");
    metrics->total_bytes_read = checked_sum(
        rerank_bytes, metrics->active_value_bytes,
        "total byte metric overflows");
  }
}

}  // namespace engram
