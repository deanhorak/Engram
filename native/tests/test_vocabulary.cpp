#include "engram/vocabulary.h"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

bool close(const float left, const float right) {
  return std::abs(left - right) < 1e-6F;
}

int fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}

}  // namespace

int main() {
  // Row 0 is the best normalized match; row 1 has the larger exact inner
  // product. Rows 2 and 3 deliberately tie for deterministic token ordering.
  const std::vector<float> embeddings = {
      1.0F, 0.0F,   // token 0
      2.0F, 0.2F,   // token 1
      0.5F, 0.5F,   // token 2
      0.5F, 0.5F,   // token 3
      -1.0F, 0.0F,  // token 4
  };
  engram::VocabularyIndex index(embeddings, 5, 2);
  engram::VocabularyScratch scratch(5);
  const std::array<float, 2> query = {1.0F, 0.0F};

  engram::VocabularySearchMetrics exact_metrics;
  const engram::TokenScore exact = index.exact_greedy(query, &exact_metrics);
  if (exact.token != 1 || !close(exact.score, 2.0F)) {
    return fail("exact greedy MIPS mismatch");
  }
  if (exact_metrics.exact_scores != 5 ||
      exact_metrics.embedding_bytes_read != 5 * 2 * sizeof(float) ||
      exact_metrics.approximate) {
    return fail("exact metrics mismatch");
  }

  std::array<engram::TokenScore, 4> exact_top{};
  index.exact_top_k(query, 4, exact_top, scratch);
  if (exact_top[0].token != 1 || exact_top[1].token != 0 ||
      exact_top[2].token != 2 || exact_top[3].token != 3) {
    return fail("exact top-k or stable tie ordering mismatch");
  }

  engram::VocabularySearchMetrics approximate_metrics;
  const engram::TokenScore narrow =
      index.approximate_greedy(query, 1, scratch, &approximate_metrics);
  if (narrow.token != 0 || approximate_metrics.proxy_scores != 5 ||
      approximate_metrics.exact_scores != 1 ||
      !approximate_metrics.approximate) {
    return fail("normalized narrow candidate search mismatch");
  }
  const engram::TokenScore expanded =
      index.approximate_greedy(query, 2, scratch, &approximate_metrics);
  if (expanded.token != 1 || !close(expanded.score, 2.0F)) {
    return fail("candidate exact rescoring mismatch");
  }
  const std::array<engram::TokenScore, 2> external_candidates = {
      engram::TokenScore{0, 1.0F}, engram::TokenScore{1, 0.9F}};
  const engram::TokenScore external = index.rescore_greedy(
      query, external_candidates, scratch, &approximate_metrics);
  if (external.token != 1 || !close(external.score, 2.0F) ||
      approximate_metrics.exact_scores != 2 ||
      approximate_metrics.embedding_bytes_read != 2 * 2 * sizeof(float)) {
    return fail("external candidate exact rescoring mismatch");
  }

  const std::array<engram::TokenScore, 1> narrow_set = {narrow};
  const std::array<engram::TokenScore, 1> exact_set = {exact};
  if (!close(engram::vocabulary_token_recall(narrow_set, exact_set), 0.0F) ||
      !close(engram::vocabulary_token_recall(exact_set, exact_set), 1.0F)) {
    return fail("vocabulary recall mismatch");
  }

  const std::array<float, 2> zero_query = {0.0F, 0.0F};
  const engram::TokenScore zero =
      index.approximate_greedy(zero_query, 3, scratch, &approximate_metrics);
  if (zero.token != 0 || !approximate_metrics.zero_norm_query) {
    return fail("zero-query stable tie mismatch");
  }

  try {
    engram::VocabularyScratch too_small(4);
    static_cast<void>(index.approximate_greedy(query, 2, too_small));
    return fail("undersized scratch was accepted");
  } catch (const std::invalid_argument&) {
  }

  return 0;
}
