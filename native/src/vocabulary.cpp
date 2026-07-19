#include "engram/vocabulary.h"
#include "engram/vector_kernels.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace engram {
namespace {

bool better(const TokenScore& left, const TokenScore& right) noexcept {
  if (left.score > right.score) {
    return true;
  }
  if (left.score < right.score) {
    return false;
  }
  return left.token < right.token;
}

void reset_metrics(VocabularySearchMetrics* metrics) noexcept {
  if (metrics != nullptr) {
    *metrics = VocabularySearchMetrics{};
  }
}

}  // namespace

VocabularyScratch::VocabularyScratch(const std::size_t vocabulary_capacity)
    : ranking_(vocabulary_capacity), rescored_(vocabulary_capacity) {
  if (vocabulary_capacity == 0) {
    throw std::invalid_argument("vocabulary scratch capacity must be positive");
  }
}

std::size_t VocabularyScratch::capacity() const noexcept {
  return ranking_.size();
}

VocabularyIndex::VocabularyIndex(std::vector<float> embeddings,
                                 const std::size_t vocabulary_size,
                                 const std::size_t hidden_size)
    : vocabulary_size_(vocabulary_size),
      hidden_size_(hidden_size),
      embeddings_(std::move(embeddings)),
      normalized_embeddings_(embeddings_.size(), 0.0F) {
  if (vocabulary_size_ == 0 || hidden_size_ == 0) {
    throw std::invalid_argument("vocabulary and hidden sizes must be positive");
  }
  if (vocabulary_size_ >
      std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("vocabulary exceeds uint32 token range");
  }
  if (hidden_size_ > std::numeric_limits<std::size_t>::max() /
                         vocabulary_size_ ||
      embeddings_.size() != vocabulary_size_ * hidden_size_) {
    throw std::invalid_argument(
        "embedding storage must have vocabulary_size * hidden_size values");
  }
  for (const float value : embeddings_) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("embeddings must contain finite values");
    }
  }

  for (std::size_t token = 0; token < vocabulary_size_; ++token) {
    double squared_norm = 0.0;
    const std::size_t offset = token * hidden_size_;
    for (std::size_t column = 0; column < hidden_size_; ++column) {
      const double value = embeddings_[offset + column];
      squared_norm += value * value;
    }
    const double norm = std::sqrt(squared_norm);
    if (norm > 0.0) {
      for (std::size_t column = 0; column < hidden_size_; ++column) {
        normalized_embeddings_[offset + column] =
            static_cast<float>(embeddings_[offset + column] / norm);
      }
    }
  }
}

std::size_t VocabularyIndex::vocabulary_size() const noexcept {
  return vocabulary_size_;
}

std::size_t VocabularyIndex::hidden_size() const noexcept {
  return hidden_size_;
}

void VocabularyIndex::validate_query(
    const std::span<const float> query) const {
  if (query.size() != hidden_size_) {
    throw std::invalid_argument("query width does not match vocabulary index");
  }
  if (!std::all_of(query.begin(), query.end(),
                   [](const float value) { return std::isfinite(value); })) {
    throw std::invalid_argument("query must contain finite values");
  }
}

void VocabularyIndex::validate_scratch(
    const VocabularyScratch& scratch) const {
  if (scratch.capacity() < vocabulary_size_ ||
      scratch.rescored_.size() < vocabulary_size_) {
    throw std::invalid_argument("vocabulary scratch capacity is too small");
  }
}

float VocabularyIndex::exact_score(const std::span<const float> query,
                                   const std::size_t token) const noexcept {
  const std::size_t offset = token * hidden_size_;
  return dot_product(embeddings_.data() + offset, query.data(), hidden_size_);
}

TokenScore VocabularyIndex::exact_greedy(
    const std::span<const float> query,
    VocabularySearchMetrics* const metrics) const {
  validate_query(query);
  reset_metrics(metrics);
  TokenScore best{0, exact_score(query, 0)};
  for (std::size_t token = 1; token < vocabulary_size_; ++token) {
    const TokenScore current{static_cast<std::uint32_t>(token),
                             exact_score(query, token)};
    if (better(current, best)) {
      best = current;
    }
  }
  if (metrics != nullptr) {
    metrics->exact_scores = vocabulary_size_;
    metrics->returned_tokens = 1;
    metrics->embedding_bytes_read =
        vocabulary_size_ * hidden_size_ * sizeof(float);
  }
  return best;
}

void VocabularyIndex::exact_top_k(
    const std::span<const float> query, const std::size_t top_k,
    const std::span<TokenScore> output, VocabularyScratch& scratch,
    VocabularySearchMetrics* const metrics) const {
  validate_query(query);
  validate_scratch(scratch);
  if (top_k == 0 || top_k > vocabulary_size_ || output.size() < top_k) {
    throw std::invalid_argument(
        "top_k must be positive, within the vocabulary, and fit output");
  }
  reset_metrics(metrics);
  for (std::size_t token = 0; token < vocabulary_size_; ++token) {
    scratch.ranking_[token] = TokenScore{static_cast<std::uint32_t>(token),
                                         exact_score(query, token)};
  }
  std::partial_sort(scratch.ranking_.begin(),
                    scratch.ranking_.begin() + top_k,
                    scratch.ranking_.begin() + vocabulary_size_, better);
  std::copy_n(scratch.ranking_.begin(), top_k, output.begin());
  if (metrics != nullptr) {
    metrics->exact_scores = vocabulary_size_;
    metrics->returned_tokens = top_k;
    metrics->embedding_bytes_read =
        vocabulary_size_ * hidden_size_ * sizeof(float);
  }
}

void VocabularyIndex::approximate_top_k(
    const std::span<const float> query, const std::size_t candidate_count,
    const std::size_t top_k, const std::span<TokenScore> output,
    VocabularyScratch& scratch,
    VocabularySearchMetrics* const metrics) const {
  validate_query(query);
  validate_scratch(scratch);
  if (candidate_count == 0 || candidate_count > vocabulary_size_) {
    throw std::invalid_argument(
        "candidate_count must be positive and within the vocabulary");
  }
  if (top_k == 0 || top_k > candidate_count || output.size() < top_k) {
    throw std::invalid_argument(
        "top_k must be positive, no larger than candidates, and fit output");
  }
  reset_metrics(metrics);

  double query_squared_norm = 0.0;
  for (const float value : query) {
    query_squared_norm += static_cast<double>(value) * value;
  }
  const double query_norm = std::sqrt(query_squared_norm);
  for (std::size_t token = 0; token < vocabulary_size_; ++token) {
    const std::size_t offset = token * hidden_size_;
    double score = 0.0;
    if (query_norm > 0.0) {
      for (std::size_t column = 0; column < hidden_size_; ++column) {
        score += static_cast<double>(normalized_embeddings_[offset + column]) *
                 query[column];
      }
      score /= query_norm;
    }
    scratch.ranking_[token] = TokenScore{static_cast<std::uint32_t>(token),
                                         static_cast<float>(score)};
  }
  std::partial_sort(scratch.ranking_.begin(),
                    scratch.ranking_.begin() + candidate_count,
                    scratch.ranking_.begin() + vocabulary_size_, better);

  for (std::size_t index = 0; index < candidate_count; ++index) {
    const std::uint32_t token = scratch.ranking_[index].token;
    scratch.rescored_[index] =
        TokenScore{token, exact_score(query, static_cast<std::size_t>(token))};
  }
  std::partial_sort(scratch.rescored_.begin(),
                    scratch.rescored_.begin() + top_k,
                    scratch.rescored_.begin() + candidate_count, better);
  std::copy_n(scratch.rescored_.begin(), top_k, output.begin());
  if (metrics != nullptr) {
    metrics->proxy_scores = vocabulary_size_;
    metrics->exact_scores = candidate_count;
    metrics->returned_tokens = top_k;
    metrics->embedding_bytes_read =
        (vocabulary_size_ + candidate_count) * hidden_size_ * sizeof(float);
    metrics->approximate = candidate_count < vocabulary_size_;
    metrics->zero_norm_query = query_norm == 0.0;
  }
}

TokenScore VocabularyIndex::approximate_greedy(
    const std::span<const float> query, const std::size_t candidate_count,
    VocabularyScratch& scratch,
    VocabularySearchMetrics* const metrics) const {
  TokenScore result{};
  approximate_top_k(query, candidate_count, 1,
                    std::span<TokenScore>(&result, 1), scratch, metrics);
  return result;
}

void VocabularyIndex::rescore_candidates(
    const std::span<const float> query,
    const std::span<const TokenScore> candidates, const std::size_t top_k,
    const std::span<TokenScore> output, VocabularyScratch& scratch,
    VocabularySearchMetrics* const metrics) const {
  validate_query(query);
  validate_scratch(scratch);
  if (candidates.empty() || candidates.size() > vocabulary_size_ ||
      top_k == 0 || top_k > candidates.size() || output.size() < top_k) {
    throw std::invalid_argument("invalid vocabulary candidate rescore sizes");
  }
  reset_metrics(metrics);
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    const std::size_t token = candidates[index].token;
    if (token >= vocabulary_size_) {
      throw std::invalid_argument("vocabulary candidate token is out of range");
    }
    for (std::size_t earlier = 0; earlier < index; ++earlier) {
      if (candidates[earlier].token == token) {
        throw std::invalid_argument("vocabulary candidate tokens must be unique");
      }
    }
    scratch.rescored_[index] = TokenScore{
        static_cast<std::uint32_t>(token), exact_score(query, token)};
  }
  std::partial_sort(scratch.rescored_.begin(),
                    scratch.rescored_.begin() + top_k,
                    scratch.rescored_.begin() + candidates.size(), better);
  std::copy_n(scratch.rescored_.begin(), top_k, output.begin());
  if (metrics != nullptr) {
    metrics->exact_scores = candidates.size();
    metrics->returned_tokens = top_k;
    metrics->embedding_bytes_read =
        candidates.size() * hidden_size_ * sizeof(float);
    metrics->approximate = candidates.size() < vocabulary_size_;
  }
}

TokenScore VocabularyIndex::rescore_greedy(
    const std::span<const float> query,
    const std::span<const TokenScore> candidates, VocabularyScratch& scratch,
    VocabularySearchMetrics* const metrics) const {
  TokenScore result{};
  rescore_candidates(query, candidates, 1, std::span<TokenScore>(&result, 1),
                     scratch, metrics);
  return result;
}

float vocabulary_token_recall(const std::span<const TokenScore> retrieved,
                              const std::span<const TokenScore> reference) {
  std::unordered_set<std::uint32_t> relevant;
  relevant.reserve(reference.size());
  for (const TokenScore item : reference) {
    relevant.insert(item.token);
  }
  if (relevant.empty()) {
    return 1.0F;
  }
  std::unordered_set<std::uint32_t> observed;
  observed.reserve(retrieved.size());
  std::size_t hits = 0;
  for (const TokenScore item : retrieved) {
    if (observed.insert(item.token).second && relevant.contains(item.token)) {
      ++hits;
    }
  }
  return static_cast<float>(hits) / static_cast<float>(relevant.size());
}

}  // namespace engram
