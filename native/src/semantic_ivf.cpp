#include "engram/semantic_ivf.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace engram {
namespace {

bool better_cluster(const SemanticIvfScratch::ClusterScore& left,
                    const SemanticIvfScratch::ClusterScore& right) noexcept {
  if (left.proxy_score > right.proxy_score) {
    return true;
  }
  if (left.proxy_score < right.proxy_score) {
    return false;
  }
  return left.index < right.index;
}

bool better_record(const SemanticIvfScratch::RecordScore& left,
                   const SemanticIvfScratch::RecordScore& right) noexcept {
  if (left.proxy_score > right.proxy_score) {
    return true;
  }
  if (left.proxy_score < right.proxy_score) {
    return false;
  }
  return left.index < right.index;
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

float decode_key(const std::uint8_t code, const float offset,
                 const float scale) noexcept {
  return offset + static_cast<float>(code) * scale;
}

double joint_proxy(const float* gate, const float* up,
                   const std::span<const float> hidden,
                   const double hidden_norm) {
  double gate_dot = 0.0;
  double up_dot = 0.0;
  double gate_squared_norm = 0.0;
  double up_squared_norm = 0.0;
  for (std::size_t column = 0; column < hidden.size(); ++column) {
    if (!std::isfinite(gate[column]) || !std::isfinite(up[column])) {
      throw std::invalid_argument(
          "semantic IVF centroids must contain finite values");
    }
    gate_dot += static_cast<double>(gate[column]) * hidden[column];
    up_dot += static_cast<double>(up[column]) * hidden[column];
    gate_squared_norm += static_cast<double>(gate[column]) * gate[column];
    up_squared_norm += static_cast<double>(up[column]) * up[column];
  }
  const double gate_norm = std::sqrt(gate_squared_norm);
  const double up_norm = std::sqrt(up_squared_norm);
  const double gate_alignment =
      gate_norm > 0.0 ? gate_dot / (gate_norm * hidden_norm) : 0.0;
  const double up_alignment =
      up_norm > 0.0 ? up_dot / (up_norm * hidden_norm) : 0.0;
  return std::max(gate_alignment, 0.0) * std::abs(up_alignment);
}

double decoded_joint_proxy(const QuantizedSemanticLayerView& layer,
                           const std::size_t record,
                           const std::span<const float> hidden,
                           const double hidden_norm) {
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
  return std::max(gate_alignment, 0.0) * std::abs(up_alignment);
}

void validate_key_layer(const QuantizedSemanticLayerView& layer) {
  if (layer.records == 0 || layer.hidden_size == 0) {
    throw std::invalid_argument(
        "quantized semantic key dimensions must be positive");
  }
  if (layer.records > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("semantic record count exceeds uint32 range");
  }
  if (layer.gate_codes == nullptr || layer.gate_offsets == nullptr ||
      layer.gate_scales == nullptr || layer.up_codes == nullptr ||
      layer.up_offsets == nullptr || layer.up_scales == nullptr) {
    throw std::invalid_argument(
        "quantized semantic key pointers must not be null");
  }
  static_cast<void>(checked_product(layer.records, layer.hidden_size,
                                    "semantic key dimensions overflow"));
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

void validate_index(const SemanticIvfIndexView& index,
                    const QuantizedSemanticLayerView& layer,
                    const std::size_t cluster_capacity,
                    const std::span<std::uint8_t> seen_records) {
  if (index.clusters == 0 || index.hidden_size == 0) {
    throw std::invalid_argument("semantic IVF dimensions must be positive");
  }
  if (index.clusters > std::numeric_limits<std::uint32_t>::max() ||
      index.posting_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("semantic IVF dimensions exceed uint32 range");
  }
  if (index.hidden_size != layer.hidden_size) {
    throw std::invalid_argument(
        "semantic IVF hidden width does not match quantized keys");
  }
  if (index.joint_centroids == nullptr || index.posting_offsets == nullptr ||
      (index.posting_count != 0 && index.postings == nullptr)) {
    throw std::invalid_argument("semantic IVF pointers must not be null");
  }
  if (cluster_capacity < index.clusters ||
      seen_records.size() < layer.records) {
    throw std::invalid_argument("semantic IVF scratch capacity is too small");
  }
  const std::size_t joint_width = checked_product(
      index.hidden_size, 2, "semantic IVF centroid dimensions overflow");
  const std::size_t centroid_elements = checked_product(
      index.clusters, joint_width,
      "semantic IVF centroid dimensions overflow");
  for (std::size_t element = 0; element < centroid_elements; ++element) {
    if (!std::isfinite(index.joint_centroids[element])) {
      throw std::invalid_argument(
          "semantic IVF centroids must contain finite values");
    }
  }
  if (index.posting_offsets[0] != 0 ||
      index.posting_offsets[index.clusters] != index.posting_count) {
    throw std::invalid_argument(
        "semantic IVF posting offsets must span all postings");
  }
  for (std::size_t cluster = 0; cluster < index.clusters; ++cluster) {
    const std::size_t begin = index.posting_offsets[cluster];
    const std::size_t end = index.posting_offsets[cluster + 1];
    if (begin > end || end > index.posting_count) {
      throw std::invalid_argument(
          "semantic IVF posting offsets must be monotonic and in bounds");
    }
  }

  std::fill(seen_records.begin(), seen_records.end(), 0);
  for (std::size_t position = 0; position < index.posting_count; ++position) {
    const std::size_t record = index.postings[position];
    if (record >= layer.records) {
      throw std::invalid_argument(
          "semantic IVF posting record is out of bounds");
    }
    if (seen_records[record] != 0) {
      throw std::invalid_argument(
          "semantic IVF postings must not contain duplicate records");
    }
    seen_records[record] = 1;
  }
}

void reset_metrics(SemanticIvfSearchMetrics* metrics) noexcept {
  if (metrics != nullptr) {
    *metrics = SemanticIvfSearchMetrics{};
  }
}

}  // namespace

SemanticIvfScratch::SemanticIvfScratch(const std::size_t cluster_capacity,
                                       const std::size_t record_capacity)
    : cluster_ranking_(cluster_capacity),
      record_ranking_(record_capacity),
      seen_records_(record_capacity) {
  if (cluster_capacity == 0 || record_capacity == 0 ||
      cluster_capacity > std::numeric_limits<std::uint32_t>::max() ||
      record_capacity > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(
        "semantic IVF scratch capacities must be positive and fit uint32");
  }
}

std::size_t SemanticIvfScratch::cluster_capacity() const noexcept {
  return cluster_ranking_.size();
}

std::size_t SemanticIvfScratch::record_capacity() const noexcept {
  return record_ranking_.size();
}

std::size_t SemanticIvfScratch::persistent_bytes() const noexcept {
  return cluster_ranking_.capacity() * sizeof(ClusterScore) +
         record_ranking_.capacity() * sizeof(RecordScore) +
         seen_records_.capacity() * sizeof(std::uint8_t);
}

void validate_semantic_ivf_index(const SemanticIvfIndexView& index,
                                 const QuantizedSemanticLayerView& layer,
                                 SemanticIvfScratch& scratch) {
  validate_key_layer(layer);
  validate_index(index, layer, scratch.cluster_capacity(),
                 scratch.seen_records_);
}

void semantic_ivf_search_scalar(
    const SemanticIvfIndexView& index,
    const QuantizedSemanticLayerView& layer,
    const std::span<const float> hidden, const std::size_t probe_count,
    const std::size_t candidate_count,
    const std::span<std::uint32_t> probed_clusters,
    const std::span<SemanticIvfCandidate> candidates,
    SemanticIvfScratch& scratch, SemanticIvfSearchMetrics* const metrics) {
  validate_key_layer(layer);
  reset_metrics(metrics);
  if (index.clusters == 0 || index.hidden_size != layer.hidden_size ||
      index.joint_centroids == nullptr || index.posting_offsets == nullptr ||
      index.postings == nullptr || scratch.cluster_capacity() < index.clusters ||
      scratch.record_capacity() < layer.records ||
      index.posting_count > layer.records || index.posting_offsets[0] != 0 ||
      index.posting_offsets[index.clusters] != index.posting_count) {
    throw std::invalid_argument(
        "semantic IVF search received invalid or unvalidated dimensions");
  }
  if (hidden.size() != layer.hidden_size) {
    throw std::invalid_argument("hidden width does not match semantic IVF");
  }
  if (!std::all_of(hidden.begin(), hidden.end(),
                   [](const float value) { return std::isfinite(value); })) {
    throw std::invalid_argument("hidden must contain finite values");
  }
  if (probe_count == 0 || probe_count > index.clusters ||
      probed_clusters.size() < probe_count) {
    throw std::invalid_argument(
        "probe_count must be positive, within clusters, and fit probe output");
  }
  if (candidate_count == 0 || candidate_count > layer.records ||
      candidates.size() < candidate_count) {
    throw std::invalid_argument(
        "candidate_count must be positive, within records, and fit candidate output");
  }

  double hidden_squared_norm = 0.0;
  for (const float value : hidden) {
    hidden_squared_norm += static_cast<double>(value) * value;
  }
  const double hidden_norm = std::sqrt(hidden_squared_norm);
  const std::size_t joint_width = index.hidden_size * 2;
  for (std::size_t cluster = 0; cluster < index.clusters; ++cluster) {
    double score = 0.0;
    if (hidden_norm > 0.0) {
      const float* gate = index.joint_centroids + cluster * joint_width;
      score = joint_proxy(gate, gate + index.hidden_size, hidden, hidden_norm);
    }
    scratch.cluster_ranking_[cluster] = {
        static_cast<std::uint32_t>(cluster), score};
  }
  std::sort(scratch.cluster_ranking_.begin(),
            scratch.cluster_ranking_.begin() + index.clusters,
            better_cluster);

  std::size_t actual_probe_count = probe_count;
  std::size_t scored_records = 0;
  for (std::size_t probe = 0; probe < actual_probe_count; ++probe) {
    const std::size_t cluster = scratch.cluster_ranking_[probe].index;
    const std::size_t begin = index.posting_offsets[cluster];
    const std::size_t end = index.posting_offsets[cluster + 1];
    if (begin > end || end > index.posting_count) {
      throw std::invalid_argument("semantic IVF posting range is invalid");
    }
    scored_records += end - begin;
    if (probe + 1 == actual_probe_count && scored_records < candidate_count &&
        actual_probe_count < index.clusters) {
      ++actual_probe_count;
    }
  }
  if (candidate_count > scored_records ||
      probed_clusters.size() < actual_probe_count) {
    throw std::invalid_argument(
        "candidate count exceeds indexed records or probe output capacity");
  }

  scored_records = 0;
  for (std::size_t probe = 0; probe < actual_probe_count; ++probe) {
    const std::size_t cluster = scratch.cluster_ranking_[probe].index;
    probed_clusters[probe] = static_cast<std::uint32_t>(cluster);
    const std::size_t begin = index.posting_offsets[cluster];
    const std::size_t end = index.posting_offsets[cluster + 1];
    for (std::size_t position = begin; position < end; ++position) {
      const std::uint32_t record = index.postings[position];
      const double score =
          hidden_norm > 0.0
              ? decoded_joint_proxy(layer, record, hidden, hidden_norm)
              : 0.0;
      scratch.record_ranking_[scored_records++] = {record, score};
    }
  }
  std::partial_sort(scratch.record_ranking_.begin(),
                    scratch.record_ranking_.begin() + candidate_count,
                    scratch.record_ranking_.begin() + scored_records,
                    better_record);
  for (std::size_t rank = 0; rank < candidate_count; ++rank) {
    const SemanticIvfScratch::RecordScore& record =
        scratch.record_ranking_[rank];
    candidates[rank] = {record.index, static_cast<float>(record.proxy_score)};
  }

  if (metrics != nullptr) {
    metrics->centroid_records = index.clusters;
    metrics->probed_clusters = actual_probe_count;
    metrics->scored_records = scored_records;
    metrics->zero_norm_query = hidden_norm == 0.0;
    const std::size_t centroid_elements = checked_product(
        checked_product(index.clusters, joint_width,
                        "semantic IVF byte metric overflows"),
        sizeof(float), "semantic IVF byte metric overflows");
    metrics->centroid_bytes = hidden_norm > 0.0 ? centroid_elements : 0;
    const std::size_t offset_bytes = checked_product(
        checked_product(actual_probe_count, 2,
                        "semantic IVF byte metric overflows"),
        sizeof(std::uint32_t), "semantic IVF byte metric overflows");
    const std::size_t posting_bytes = checked_product(
        scored_records, sizeof(std::uint32_t),
        "semantic IVF byte metric overflows");
    metrics->posting_bytes = checked_sum(
        offset_bytes, posting_bytes, "semantic IVF byte metric overflows");
    if (hidden_norm > 0.0) {
      const std::size_t code_bytes = checked_product(
          checked_product(scored_records, joint_width,
                          "semantic IVF byte metric overflows"),
          sizeof(std::uint8_t), "semantic IVF byte metric overflows");
      const std::size_t affine_bytes = checked_product(
          checked_product(layer.hidden_size, 4,
                          "semantic IVF byte metric overflows"),
          sizeof(float), "semantic IVF byte metric overflows");
      metrics->key_bytes = checked_sum(
          code_bytes, affine_bytes, "semantic IVF byte metric overflows");
    }
    metrics->index_bytes_read = checked_sum(
        checked_sum(metrics->centroid_bytes, metrics->posting_bytes,
                    "semantic IVF byte metric overflows"),
        metrics->key_bytes, "semantic IVF byte metric overflows");
  }
}

}  // namespace engram
