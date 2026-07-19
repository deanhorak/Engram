#include "engram/vocabulary_ivf.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace engram {
namespace {

bool better_cluster(const VocabularyIvfScratch::ClusterScore& left,
                    const VocabularyIvfScratch::ClusterScore& right) noexcept {
  if (left.proxy_score > right.proxy_score) {
    return true;
  }
  if (left.proxy_score < right.proxy_score) {
    return false;
  }
  return left.index < right.index;
}

bool better_token(const TokenScore& left, const TokenScore& right) noexcept {
  if (left.score > right.score) {
    return true;
  }
  if (left.score < right.score) {
    return false;
  }
  return left.token < right.token;
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

double centroid_score(const float* centroid,
                      const std::span<const float> query,
                      const double query_norm) {
  double dot = 0.0;
  double squared_norm = 0.0;
  for (std::size_t column = 0; column < query.size(); ++column) {
    const float value = centroid[column];
    if (!std::isfinite(value)) {
      throw std::invalid_argument(
          "vocabulary IVF centroids must contain finite values");
    }
    dot += static_cast<double>(value) * query[column];
    squared_norm += static_cast<double>(value) * value;
  }
  const double norm = std::sqrt(squared_norm);
  return norm > 0.0 ? dot / (norm * query_norm) : 0.0;
}

float embedding_score(const float* normalized_embedding,
                      const std::span<const float> query,
                      const double query_norm) {
  double dot = 0.0;
  for (std::size_t column = 0; column < query.size(); ++column) {
    const float value = normalized_embedding[column];
    if (!std::isfinite(value)) {
      throw std::invalid_argument(
          "normalized vocabulary embeddings must contain finite values");
    }
    dot += static_cast<double>(value) * query[column];
  }
  return static_cast<float>(dot / query_norm);
}

void validate_scratch(const VocabularyIvfIndexView& view,
                      const VocabularyIvfScratch& scratch) {
  if (scratch.cluster_capacity() < view.clusters ||
      scratch.vocabulary_capacity() < view.vocabulary_size) {
    throw std::invalid_argument("vocabulary IVF scratch capacity is too small");
  }
}

void reset_metrics(VocabularyIvfSearchMetrics* metrics) noexcept {
  if (metrics != nullptr) {
    *metrics = VocabularyIvfSearchMetrics{};
  }
}

}  // namespace

VocabularyIvfScratch::VocabularyIvfScratch(
    const std::size_t cluster_capacity,
    const std::size_t vocabulary_capacity)
    : cluster_ranking_(cluster_capacity),
      token_ranking_(vocabulary_capacity),
      seen_tokens_(vocabulary_capacity) {
  if (cluster_capacity == 0 || vocabulary_capacity == 0 ||
      cluster_capacity > std::numeric_limits<std::uint32_t>::max() ||
      vocabulary_capacity > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(
        "vocabulary IVF scratch capacities must be positive and fit uint32");
  }
}

std::size_t VocabularyIvfScratch::cluster_capacity() const noexcept {
  return cluster_ranking_.size();
}

std::size_t VocabularyIvfScratch::vocabulary_capacity() const noexcept {
  return token_ranking_.size();
}

std::size_t VocabularyIvfScratch::persistent_bytes() const noexcept {
  return cluster_ranking_.capacity() * sizeof(ClusterScore) +
         token_ranking_.capacity() * sizeof(TokenScore) +
         seen_tokens_.capacity() * sizeof(std::uint8_t);
}

PreparedVocabularyIvfIndex::PreparedVocabularyIvfIndex(
    const VocabularyIvfIndexView view) noexcept
    : view_(view) {}

std::size_t PreparedVocabularyIvfIndex::vocabulary_size() const noexcept {
  return view_.vocabulary_size;
}

std::size_t PreparedVocabularyIvfIndex::hidden_size() const noexcept {
  return view_.hidden_size;
}

std::size_t PreparedVocabularyIvfIndex::clusters() const noexcept {
  return view_.clusters;
}

PreparedVocabularyIvfIndex prepare_vocabulary_ivf_index(
    const VocabularyIvfIndexView& view, VocabularyIvfScratch& scratch) {
  if (view.vocabulary_size == 0 || view.hidden_size == 0 ||
      view.clusters == 0) {
    throw std::invalid_argument("vocabulary IVF dimensions must be positive");
  }
  if (view.vocabulary_size > std::numeric_limits<std::uint32_t>::max() ||
      view.clusters > std::numeric_limits<std::uint32_t>::max() ||
      view.posting_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("vocabulary IVF dimensions exceed uint32 range");
  }
  if (view.posting_count != view.vocabulary_size) {
    throw std::invalid_argument(
        "vocabulary IVF postings must cover every token exactly once");
  }
  if (view.normalized_embeddings == nullptr || view.centroids == nullptr ||
      view.posting_offsets == nullptr || view.postings == nullptr) {
    throw std::invalid_argument("vocabulary IVF pointers must not be null");
  }
  static_cast<void>(checked_product(
      view.vocabulary_size, view.hidden_size,
      "vocabulary IVF embedding dimensions overflow"));
  static_cast<void>(checked_product(
      view.clusters, view.hidden_size,
      "vocabulary IVF centroid dimensions overflow"));
  validate_scratch(view, scratch);
  if (view.posting_offsets[0] != 0 ||
      view.posting_offsets[view.clusters] != view.posting_count) {
    throw std::invalid_argument(
        "vocabulary IVF posting offsets must span all postings");
  }
  for (std::size_t cluster = 0; cluster < view.clusters; ++cluster) {
    const std::size_t begin = view.posting_offsets[cluster];
    const std::size_t end = view.posting_offsets[cluster + 1];
    if (begin > end || end > view.posting_count) {
      throw std::invalid_argument(
          "vocabulary IVF posting offsets must be monotonic and in bounds");
    }
  }

  std::fill_n(scratch.seen_tokens_.begin(), view.vocabulary_size, 0);
  for (std::size_t position = 0; position < view.posting_count; ++position) {
    const std::size_t token = view.postings[position];
    if (token >= view.vocabulary_size) {
      throw std::invalid_argument(
          "vocabulary IVF posting token is out of bounds");
    }
    if (scratch.seen_tokens_[token] != 0) {
      throw std::invalid_argument(
          "vocabulary IVF postings contain a duplicate token");
    }
    scratch.seen_tokens_[token] = 1;
  }
  return PreparedVocabularyIvfIndex(view);
}

void vocabulary_ivf_search_scalar(
    const PreparedVocabularyIvfIndex& index,
    const std::span<const float> query,
    const std::size_t minimum_probe_count,
    const std::size_t candidate_count,
    const std::span<TokenScore> candidates, VocabularyIvfScratch& scratch,
    VocabularyIvfSearchMetrics* const metrics) {
  const VocabularyIvfIndexView& view = index.view_;
  reset_metrics(metrics);
  validate_scratch(view, scratch);
  if (query.size() != view.hidden_size) {
    throw std::invalid_argument("query width does not match vocabulary IVF");
  }
  if (!std::all_of(query.begin(), query.end(),
                   [](const float value) { return std::isfinite(value); })) {
    throw std::invalid_argument("query must contain finite values");
  }
  if (minimum_probe_count == 0 || minimum_probe_count > view.clusters) {
    throw std::invalid_argument(
        "minimum_probe_count must be positive and within IVF clusters");
  }
  if (candidate_count == 0 || candidate_count > view.vocabulary_size ||
      candidates.size() < candidate_count) {
    throw std::invalid_argument(
        "candidate_count must be positive, within vocabulary, and fit output");
  }

  double query_squared_norm = 0.0;
  for (const float value : query) {
    query_squared_norm += static_cast<double>(value) * value;
  }
  const double query_norm = std::sqrt(query_squared_norm);
  if (query_norm == 0.0) {
    for (std::size_t token = 0; token < candidate_count; ++token) {
      candidates[token] = TokenScore{static_cast<std::uint32_t>(token), 0.0F};
    }
    if (metrics != nullptr) {
      metrics->minimum_probes = minimum_probe_count;
      metrics->scored_records = candidate_count;
      metrics->returned_candidates = candidate_count;
      metrics->zero_norm_query = true;
    }
    return;
  }
  for (std::size_t cluster = 0; cluster < view.clusters; ++cluster) {
    double score = 0.0;
    score = centroid_score(
        view.centroids + cluster * view.hidden_size, query, query_norm);
    scratch.cluster_ranking_[cluster] = {
        static_cast<std::uint32_t>(cluster), 0, 0, score};
  }
  std::sort(scratch.cluster_ranking_.begin(),
            scratch.cluster_ranking_.begin() + view.clusters,
            better_cluster);

  std::size_t probe_count = 0;
  std::size_t scored_records = 0;
  while (probe_count < minimum_probe_count ||
         scored_records < candidate_count) {
    if (probe_count == view.clusters) {
      throw std::invalid_argument(
          "vocabulary IVF postings cannot satisfy candidate_count");
    }
    VocabularyIvfScratch::ClusterScore& cluster =
        scratch.cluster_ranking_[probe_count];
    cluster.posting_begin = view.posting_offsets[cluster.index];
    cluster.posting_end = view.posting_offsets[cluster.index + 1];
    scored_records = checked_sum(
        scored_records,
        static_cast<std::size_t>(cluster.posting_end - cluster.posting_begin),
        "vocabulary IVF candidate count overflows");
    ++probe_count;
  }

  std::size_t ranking_position = 0;
  for (std::size_t probe = 0; probe < probe_count; ++probe) {
    const VocabularyIvfScratch::ClusterScore& cluster =
        scratch.cluster_ranking_[probe];
    for (std::size_t position = cluster.posting_begin;
         position < cluster.posting_end; ++position) {
      const std::uint32_t token = view.postings[position];
      float score = 0.0F;
      score = embedding_score(
          view.normalized_embeddings +
              static_cast<std::size_t>(token) * view.hidden_size,
          query, query_norm);
      scratch.token_ranking_[ranking_position++] = {token, score};
    }
  }
  std::partial_sort(scratch.token_ranking_.begin(),
                    scratch.token_ranking_.begin() + candidate_count,
                    scratch.token_ranking_.begin() + scored_records,
                    better_token);
  std::copy_n(scratch.token_ranking_.begin(), candidate_count,
              candidates.begin());

  if (metrics != nullptr) {
    metrics->centroid_proxy_scores = view.clusters;
    metrics->embedding_proxy_scores = scored_records;
    metrics->minimum_probes = minimum_probe_count;
    metrics->probed_clusters = probe_count;
    metrics->expanded_probes = probe_count - minimum_probe_count;
    metrics->scored_records = scored_records;
    metrics->returned_candidates = candidate_count;
    metrics->zero_norm_query = false;
    metrics->centroid_bytes_read =
        checked_product(
            checked_product(view.clusters, view.hidden_size,
                            "vocabulary IVF byte metric overflows"),
            sizeof(float), "vocabulary IVF byte metric overflows");
    const std::size_t offset_bytes = checked_product(
        checked_product(probe_count, 2,
                        "vocabulary IVF byte metric overflows"),
        sizeof(std::uint32_t), "vocabulary IVF byte metric overflows");
    const std::size_t token_bytes = checked_product(
        scored_records, sizeof(std::uint32_t),
        "vocabulary IVF byte metric overflows");
    metrics->posting_bytes_read = checked_sum(
        offset_bytes, token_bytes, "vocabulary IVF byte metric overflows");
    metrics->embedding_bytes_read =
        checked_product(
            checked_product(scored_records, view.hidden_size,
                            "vocabulary IVF byte metric overflows"),
            sizeof(float), "vocabulary IVF byte metric overflows");
    metrics->index_bytes_read = checked_sum(
        checked_sum(metrics->centroid_bytes_read,
                    metrics->posting_bytes_read,
                    "vocabulary IVF byte metric overflows"),
        metrics->embedding_bytes_read,
        "vocabulary IVF byte metric overflows");
  }
}

}  // namespace engram
