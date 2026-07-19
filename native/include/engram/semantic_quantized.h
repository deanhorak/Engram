#pragma once

#include "engram/semantic.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace engram {

// Non-owning row-major view of scalar-affine quantized keys and additive
// codebook quantized values. A key element decodes as
//   offset[column] + code[record, column] * scale[column].
// A value vector is the sum of one codeword from each value codebook.
struct QuantizedSemanticLayerView {
  std::size_t records{};
  std::size_t hidden_size{};
  std::size_t output_size{};
  std::size_t value_codebook_count{};
  std::size_t value_codebook_size{};

  const std::uint8_t* gate_codes{};  // [records, hidden_size]
  const float* gate_offsets{};       // [hidden_size]
  const float* gate_scales{};        // [hidden_size]
  const std::uint8_t* up_codes{};    // [records, hidden_size]
  const float* up_offsets{};         // [hidden_size]
  const float* up_scales{};          // [hidden_size]

  const std::uint8_t* value_codes{};  // [records, value_codebook_count]
  const float* value_codebooks{};
  // [value_codebook_count, value_codebook_size, output_size]
};

// Caller-owned workspace. Once constructed at the layer's record capacity,
// semantic_read_quantized_scalar performs no dynamic allocation.
class QuantizedSemanticScratch {
 public:
  // Public only so scalar ordering helpers can use the type; callers should
  // treat entries as an implementation detail and use the result spans.
  struct ScratchRecord {
    std::uint32_t index{};
    std::uint32_t candidate_order{};
    double proxy_score{};
    double activation{};
    double exact_score{};
  };

  explicit QuantizedSemanticScratch(std::size_t record_capacity);

  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t persistent_bytes() const noexcept;

 private:
  std::vector<ScratchRecord> proxy_ranking_;
  std::vector<ScratchRecord> exact_ranking_;

  friend void semantic_read_quantized_scalar(
      const QuantizedSemanticLayerView&, std::span<const float>, std::size_t,
      std::size_t, std::span<float>, std::span<SemanticRecordResult>,
      QuantizedSemanticScratch&, SemanticReadMetrics*);
  friend void semantic_read_quantized_candidates_scalar(
      const QuantizedSemanticLayerView&, std::span<const float>,
      std::span<const std::uint32_t>, std::span<const float>, std::size_t,
      std::span<float>, std::span<SemanticRecordResult>,
      QuantizedSemanticScratch&, SemanticReadMetrics*);
};

// Quantized equivalent of semantic_read_scalar. Ranking is performed on keys
// and values after decoding to float32, with stable source-index proxy ties and
// stable candidate-order exact ties. SemanticReadMetrics byte fields contain
// code bytes plus affine metadata or referenced codebook bytes for each pass.
void semantic_read_quantized_scalar(
    const QuantizedSemanticLayerView& layer,
    std::span<const float> hidden, std::size_t candidate_count,
    std::size_t top_k, std::span<float> output,
    std::span<SemanticRecordResult> selected,
    QuantizedSemanticScratch& scratch,
    SemanticReadMetrics* metrics = nullptr);

// Exact SwiGLU reranking/read for a candidate set supplied by an external
// index. Candidate indices and proxy scores must be aligned and unique.
void semantic_read_quantized_candidates_scalar(
    const QuantizedSemanticLayerView& layer,
    std::span<const float> hidden,
    std::span<const std::uint32_t> candidate_indices,
    std::span<const float> candidate_proxy_scores, std::size_t top_k,
    std::span<float> output, std::span<SemanticRecordResult> selected,
    QuantizedSemanticScratch& scratch,
    SemanticReadMetrics* metrics = nullptr);

}  // namespace engram
