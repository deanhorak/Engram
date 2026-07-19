#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace engram {

// Non-owning float32 row-major views. Values are stored one output vector per
// semantic record, matching Python's [records, output_width] layout.
struct SemanticLayerView {
  std::size_t records{};
  std::size_t hidden_size{};
  std::size_t output_size{};
  const float* gate_keys{};  // [records, hidden_size]
  const float* up_keys{};    // [records, hidden_size]
  const float* values{};     // [records, output_size]
};

struct SemanticRecordResult {
  std::uint32_t index{};
  float proxy_score{};
  float activation{};
  float exact_score{};
};

struct SemanticReadMetrics {
  std::size_t proxy_records{};
  std::size_t candidate_records{};
  std::size_t active_records{};
  std::size_t proxy_key_bytes{};
  std::size_t exact_key_bytes{};
  std::size_t exact_value_bytes{};
  std::size_t active_value_bytes{};
  std::size_t total_bytes_read{};
  bool zero_norm_query{};
};

// Caller-owned workspace. Once constructed at the layer's record capacity,
// semantic_read_scalar performs no dynamic allocation.
class SemanticScratch {
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

  explicit SemanticScratch(std::size_t record_capacity);

  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t persistent_bytes() const noexcept;

 private:
  std::vector<ScratchRecord> proxy_ranking_;
  std::vector<ScratchRecord> exact_ranking_;

  friend void semantic_read_scalar(const SemanticLayerView&,
                                   std::span<const float>, std::size_t,
                                   std::size_t, std::span<float>,
                                   std::span<SemanticRecordResult>,
                                   SemanticScratch&, SemanticReadMetrics*);
};

// Brute-force normalized joint-key proxy:
//   max(cos(hidden, gate), 0) * abs(cos(hidden, up))
// followed by exact candidate-only SwiGLU reranking using
//   abs(SiLU(gate dot hidden) * (up dot hidden)) * ||value||.
// Stable ties preserve source-index order for proxy candidates and proxy order
// during exact reranking. The first top_k entries of selected are populated in
// decreasing exact-score order; output receives their weighted value sum.
void semantic_read_scalar(
    const SemanticLayerView& layer, std::span<const float> hidden,
    std::size_t candidate_count, std::size_t top_k,
    std::span<float> output, std::span<SemanticRecordResult> selected,
    SemanticScratch& scratch, SemanticReadMetrics* metrics = nullptr);

}  // namespace engram
