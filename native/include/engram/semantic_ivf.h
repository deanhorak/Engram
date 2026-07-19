#pragma once

#include "engram/semantic_quantized.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace engram {

// Non-owning IVF index. Each row of joint_centroids stores a gate centroid
// followed by an up centroid. posting_offsets and postings use conventional
// uint32 CSR layout; a semantic record may occur in at most one posting list.
struct SemanticIvfIndexView {
  std::size_t clusters{};
  std::size_t hidden_size{};
  std::size_t posting_count{};
  const float* joint_centroids{};          // [clusters, 2 * hidden_size]
  const std::uint32_t* posting_offsets{};  // [clusters + 1]
  const std::uint32_t* postings{};         // [posting_count]
};

struct SemanticIvfCandidate {
  std::uint32_t index{};
  float proxy_score{};
};

// Logical bytes read by ranking, excluding structural validation.
// key_bytes includes uint8 gate/up codes and their float32 affine metadata.
struct SemanticIvfSearchMetrics {
  std::size_t centroid_records{};
  std::size_t probed_clusters{};
  std::size_t scored_records{};
  std::size_t centroid_bytes{};
  std::size_t posting_bytes{};
  std::size_t key_bytes{};
  std::size_t index_bytes_read{};
  bool zero_norm_query{};
};

// Caller-owned workspace. Construction is the only allocation performed by
// semantic_ivf_search_scalar. record_capacity also bounds duplicate checking.
class SemanticIvfScratch {
 public:
  struct ClusterScore {
    std::uint32_t index{};
    double proxy_score{};
  };

  struct RecordScore {
    std::uint32_t index{};
    double proxy_score{};
  };

  SemanticIvfScratch(std::size_t cluster_capacity,
                     std::size_t record_capacity);

  [[nodiscard]] std::size_t cluster_capacity() const noexcept;
  [[nodiscard]] std::size_t record_capacity() const noexcept;
  [[nodiscard]] std::size_t persistent_bytes() const noexcept;

 private:
  std::vector<ClusterScore> cluster_ranking_;
  std::vector<RecordScore> record_ranking_;
  std::vector<std::uint8_t> seen_records_;

  friend void semantic_ivf_search_scalar(
      const SemanticIvfIndexView&, const QuantizedSemanticLayerView&,
      std::span<const float>, std::size_t, std::size_t,
      std::span<std::uint32_t>, std::span<SemanticIvfCandidate>,
      SemanticIvfScratch&, SemanticIvfSearchMetrics*);
  friend void validate_semantic_ivf_index(
      const SemanticIvfIndexView&, const QuantizedSemanticLayerView&,
      SemanticIvfScratch&);
};

// Performs the O(records) CSR/permutation validation intended for package load.
// The hot search path assumes this one-time validation has succeeded.
void validate_semantic_ivf_index(const SemanticIvfIndexView& index,
                                 const QuantizedSemanticLayerView& layer,
                                 SemanticIvfScratch& scratch);

// Scores all joint centroids, probes the best probe_count clusters, then scores
// only their quantized key rows. Ties are deterministic: lower cluster indices
// win centroid ties and lower record indices win candidate ties. The first
// probe_count/candidate_count elements of the output spans are populated.
void semantic_ivf_search_scalar(
    const SemanticIvfIndexView& index,
    const QuantizedSemanticLayerView& layer,
    std::span<const float> hidden, std::size_t probe_count,
    std::size_t candidate_count, std::span<std::uint32_t> probed_clusters,
    std::span<SemanticIvfCandidate> candidates, SemanticIvfScratch& scratch,
    SemanticIvfSearchMetrics* metrics = nullptr);

}  // namespace engram
