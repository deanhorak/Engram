#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace engram {

struct TokenScore {
  std::uint32_t token = 0;
  float score = 0.0F;
};

struct VocabularySearchMetrics {
  std::size_t proxy_scores = 0;
  std::size_t exact_scores = 0;
  std::size_t returned_tokens = 0;
  std::size_t embedding_bytes_read = 0;
  bool approximate = false;
  bool zero_norm_query = false;
};

// Caller-owned reusable workspace. Search calls perform no dynamic allocation.
class VocabularyScratch {
 public:
  explicit VocabularyScratch(std::size_t vocabulary_capacity);

  [[nodiscard]] std::size_t capacity() const noexcept;

 private:
  friend class VocabularyIndex;
  std::vector<TokenScore> ranking_;
  std::vector<TokenScore> rescored_;
};

// Row-major float32 vocabulary embeddings with scalar reference searches.
class VocabularyIndex {
 public:
  VocabularyIndex(std::vector<float> embeddings, std::size_t vocabulary_size,
                  std::size_t hidden_size);

  [[nodiscard]] std::size_t vocabulary_size() const noexcept;
  [[nodiscard]] std::size_t hidden_size() const noexcept;

  // Exact maximum-inner-product greedy search over every vocabulary row.
  [[nodiscard]] TokenScore exact_greedy(
      std::span<const float> query,
      VocabularySearchMetrics* metrics = nullptr) const;

  // Exact top-K MIPS. output.size() must be at least top_k.
  void exact_top_k(std::span<const float> query, std::size_t top_k,
                   std::span<TokenScore> output, VocabularyScratch& scratch,
                   VocabularySearchMetrics* metrics = nullptr) const;

  // Normalized candidate search followed by exact original-embedding rescoring.
  void approximate_top_k(
      std::span<const float> query, std::size_t candidate_count,
      std::size_t top_k, std::span<TokenScore> output,
      VocabularyScratch& scratch,
      VocabularySearchMetrics* metrics = nullptr) const;

  [[nodiscard]] TokenScore approximate_greedy(
      std::span<const float> query, std::size_t candidate_count,
      VocabularyScratch& scratch,
      VocabularySearchMetrics* metrics = nullptr) const;

  // Exact-rescore candidate IDs supplied by an external coarse index.
  void rescore_candidates(
      std::span<const float> query, std::span<const TokenScore> candidates,
      std::size_t top_k, std::span<TokenScore> output,
      VocabularyScratch& scratch,
      VocabularySearchMetrics* metrics = nullptr) const;

  [[nodiscard]] TokenScore rescore_greedy(
      std::span<const float> query, std::span<const TokenScore> candidates,
      VocabularyScratch& scratch,
      VocabularySearchMetrics* metrics = nullptr) const;

 private:
  void validate_query(std::span<const float> query) const;
  void validate_scratch(const VocabularyScratch& scratch) const;
  [[nodiscard]] float exact_score(std::span<const float> query,
                                  std::size_t token) const noexcept;

  std::size_t vocabulary_size_;
  std::size_t hidden_size_;
  std::vector<float> embeddings_;
  std::vector<float> normalized_embeddings_;
};

// Set recall over token IDs; duplicate IDs count once.
[[nodiscard]] float vocabulary_token_recall(
    std::span<const TokenScore> retrieved,
    std::span<const TokenScore> reference);

}  // namespace engram
