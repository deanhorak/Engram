#pragma once

#include "engram/vocabulary.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace engram {

// Non-owning IVF vocabulary data. normalized_embeddings contains the unit-row
// proxy matrix (zero rows are allowed). Centroids need not be pre-normalized.
// The uint32 posting arrays use conventional CSR layout.
struct VocabularyIvfIndexView {
  std::size_t vocabulary_size{};
  std::size_t hidden_size{};
  std::size_t clusters{};
  std::size_t posting_count{};
  const float* normalized_embeddings{};   // [vocabulary_size, hidden_size]
  const float* centroids{};               // [clusters, hidden_size]
  const std::uint32_t* posting_offsets{};  // [clusters + 1]
  const std::uint32_t* postings{};         // [posting_count]
};

struct VocabularyIvfSearchMetrics {
  std::size_t centroid_proxy_scores{};
  std::size_t embedding_proxy_scores{};
  std::size_t minimum_probes{};
  std::size_t probed_clusters{};
  std::size_t expanded_probes{};
  std::size_t scored_records{};
  std::size_t returned_candidates{};
  std::size_t centroid_bytes_read{};
  std::size_t posting_bytes_read{};
  std::size_t embedding_bytes_read{};
  std::size_t index_bytes_read{};
  bool zero_norm_query{};
};

class PreparedVocabularyIvfIndex;

// Caller-owned validation and search workspace. No allocation occurs in index
// preparation or search after this object has been constructed.
class VocabularyIvfScratch {
 public:
  struct ClusterScore {
    std::uint32_t index{};
    std::uint32_t posting_begin{};
    std::uint32_t posting_end{};
    double proxy_score{};
  };

  VocabularyIvfScratch(std::size_t cluster_capacity,
                       std::size_t vocabulary_capacity);

  [[nodiscard]] std::size_t cluster_capacity() const noexcept;
  [[nodiscard]] std::size_t vocabulary_capacity() const noexcept;
  [[nodiscard]] std::size_t persistent_bytes() const noexcept;

 private:
  std::vector<ClusterScore> cluster_ranking_;
  std::vector<TokenScore> token_ranking_;
  std::vector<std::uint8_t> seen_tokens_;

  friend class PreparedVocabularyIvfIndex;
  friend PreparedVocabularyIvfIndex prepare_vocabulary_ivf_index(
      const VocabularyIvfIndexView&, VocabularyIvfScratch&);
  friend void vocabulary_ivf_search_scalar(
      const PreparedVocabularyIvfIndex&, std::span<const float>, std::size_t,
      std::size_t, std::span<TokenScore>, VocabularyIvfScratch&,
      VocabularyIvfSearchMetrics*);
};

// Lightweight validation certificate. It owns no index data. The source view's
// pointers must remain valid, and its CSR arrays must not change, while this
// handle is used. Float centroid and embedding contents remain non-owning.
class PreparedVocabularyIvfIndex {
 public:
  [[nodiscard]] std::size_t vocabulary_size() const noexcept;
  [[nodiscard]] std::size_t hidden_size() const noexcept;
  [[nodiscard]] std::size_t clusters() const noexcept;

 private:
  explicit PreparedVocabularyIvfIndex(VocabularyIvfIndexView view) noexcept;

  VocabularyIvfIndexView view_;

  friend PreparedVocabularyIvfIndex prepare_vocabulary_ivf_index(
      const VocabularyIvfIndexView&, VocabularyIvfScratch&);
  friend void vocabulary_ivf_search_scalar(
      const PreparedVocabularyIvfIndex&, std::span<const float>, std::size_t,
      std::size_t, std::span<TokenScore>, VocabularyIvfScratch&,
      VocabularyIvfSearchMetrics*);
};

// Performs the O(V) CSR coverage, token-bound, and duplicate checks once.
// Every token must occur exactly once across the posting lists.
[[nodiscard]] PreparedVocabularyIvfIndex prepare_vocabulary_ivf_index(
    const VocabularyIvfIndexView& view, VocabularyIvfScratch& scratch);

// Scores centroids, probes at least minimum_probe_count lists, and expands the
// deterministic centroid prefix until it contains candidate_count tokens. Only
// normalized embedding rows in that prefix are scored. Candidate ties use the
// lower source token ID. The first candidate_count output entries are written.
void vocabulary_ivf_search_scalar(
    const PreparedVocabularyIvfIndex& index,
    std::span<const float> query, std::size_t minimum_probe_count,
    std::size_t candidate_count, std::span<TokenScore> candidates,
    VocabularyIvfScratch& scratch,
    VocabularyIvfSearchMetrics* metrics = nullptr);

}  // namespace engram
