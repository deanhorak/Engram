#include "engram/vocabulary_ivf.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

bool close(const float left, const float right, const float tolerance = 1e-6F) {
  return std::abs(left - right) <= tolerance;
}

int fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}

struct Fixture {
  std::vector<float> embeddings = {
      1.0F, 0.0F,   // token 0, perfect but in an unprobed cluster
      0.6F, 0.8F,   // token 1
      0.6F, 0.8F,   // token 2, ties token 1
      0.0F, 1.0F,   // token 3
      0.0F, 1.0F,   // token 4
      -1.0F, 0.0F,  // token 5
      -1.0F, 0.0F,  // token 6
  };
  std::vector<float> centroids = {
      1.0F, 0.0F,
      0.8F, 0.6F,
      0.0F, 1.0F,
      -1.0F, 0.0F};
  std::array<std::uint32_t, 5> offsets = {0, 1, 3, 5, 7};
  std::array<std::uint32_t, 7> postings = {4, 2, 1, 0, 3, 6, 5};

  engram::VocabularyIvfIndexView view() const {
    return {7, 2, 4, postings.size(), embeddings.data(), centroids.data(),
            offsets.data(), postings.data()};
  }
};

}  // namespace

int main() {
  Fixture fixture;
  engram::VocabularyIvfScratch scratch(4, 7);
  const engram::PreparedVocabularyIvfIndex index =
      engram::prepare_vocabulary_ivf_index(fixture.view(), scratch);
  if (index.vocabulary_size() != 7 || index.hidden_size() != 2 ||
      index.clusters() != 4) {
    return fail("prepared vocabulary IVF dimensions mismatch");
  }

  const std::array<float, 2> query = {1.0F, 0.0F};
  std::array<engram::TokenScore, 3> candidates{};
  engram::VocabularyIvfSearchMetrics metrics;
  const std::size_t scratch_bytes = scratch.persistent_bytes();
  engram::vocabulary_ivf_search_scalar(index, query, 1, 3, candidates,
                                       scratch, &metrics);
  if (candidates[0].token != 1 || !close(candidates[0].score, 0.6F) ||
      candidates[1].token != 2 || !close(candidates[1].score, 0.6F) ||
      candidates[2].token != 4 || !close(candidates[2].score, 0.0F)) {
    return fail("vocabulary IVF posted proxy ranking mismatch");
  }
  // Token 0 is a perfect normalized match but is outside the probe prefix.
  for (const engram::TokenScore candidate : candidates) {
    if (candidate.token == 0) {
      return fail("vocabulary IVF scored an unprobed embedding row");
    }
  }
  if (metrics.centroid_proxy_scores != 4 ||
      metrics.embedding_proxy_scores != 3 || metrics.minimum_probes != 1 ||
      metrics.probed_clusters != 2 || metrics.expanded_probes != 1 ||
      metrics.scored_records != 3 || metrics.returned_candidates != 3 ||
      metrics.centroid_bytes_read != 32 ||
      metrics.posting_bytes_read != 28 ||
      metrics.embedding_bytes_read != 24 ||
      metrics.index_bytes_read != 84 || metrics.zero_norm_query ||
      scratch.persistent_bytes() != scratch_bytes) {
    return fail("vocabulary IVF expansion, metrics, or scratch reuse mismatch");
  }

  // A minimum of two probes does not expand when two candidates are requested.
  engram::vocabulary_ivf_search_scalar(index, query, 2, 2, candidates,
                                       scratch, &metrics);
  if (metrics.probed_clusters != 2 || metrics.expanded_probes != 0 ||
      candidates[0].token != 1 || candidates[1].token != 2) {
    return fail("vocabulary IVF minimum probe handling mismatch");
  }

  // A zero query preserves global source-token order without index reads.
  const std::array<float, 2> zero = {0.0F, 0.0F};
  engram::vocabulary_ivf_search_scalar(index, zero, 1, 3, candidates, scratch,
                                       &metrics);
  if (candidates[0].token != 0 || candidates[1].token != 1 ||
      candidates[2].token != 2 || !metrics.zero_norm_query ||
      metrics.probed_clusters != 0 || metrics.centroid_bytes_read != 0 ||
      metrics.embedding_bytes_read != 0 ||
      metrics.posting_bytes_read != 0 || metrics.index_bytes_read != 0) {
    return fail("zero-query vocabulary IVF ordering or metrics mismatch");
  }

  // Float payloads remain non-owning after structural preparation.
  fixture.embeddings[2] = 1.0F;
  fixture.embeddings[3] = 0.0F;
  engram::vocabulary_ivf_search_scalar(index, query, 1, 2, candidates,
                                       scratch);
  if (candidates[0].token != 1 || !close(candidates[0].score, 1.0F)) {
    return fail("prepared vocabulary IVF unexpectedly owns embedding storage");
  }

  try {
    std::array<std::uint32_t, 7> duplicate = fixture.postings;
    duplicate[1] = duplicate[0];
    engram::VocabularyIvfIndexView invalid = fixture.view();
    invalid.postings = duplicate.data();
    static_cast<void>(
        engram::prepare_vocabulary_ivf_index(invalid, scratch));
    return fail("duplicate vocabulary IVF posting was accepted");
  } catch (const std::invalid_argument&) {
  }
  try {
    std::array<std::uint32_t, 7> out_of_bounds = fixture.postings;
    out_of_bounds[0] = 7;
    engram::VocabularyIvfIndexView invalid = fixture.view();
    invalid.postings = out_of_bounds.data();
    static_cast<void>(
        engram::prepare_vocabulary_ivf_index(invalid, scratch));
    return fail("out-of-bounds vocabulary IVF posting was accepted");
  } catch (const std::invalid_argument&) {
  }
  try {
    const std::array<std::uint32_t, 5> invalid_offsets = {0, 3, 2, 5, 7};
    engram::VocabularyIvfIndexView invalid = fixture.view();
    invalid.posting_offsets = invalid_offsets.data();
    static_cast<void>(
        engram::prepare_vocabulary_ivf_index(invalid, scratch));
    return fail("non-monotonic vocabulary IVF offsets were accepted");
  } catch (const std::invalid_argument&) {
  }
  try {
    engram::VocabularyIvfIndexView invalid = fixture.view();
    invalid.posting_count = 6;
    static_cast<void>(
        engram::prepare_vocabulary_ivf_index(invalid, scratch));
    return fail("incomplete vocabulary IVF coverage was accepted");
  } catch (const std::invalid_argument&) {
  }
  try {
    engram::VocabularyIvfScratch too_small(3, 7);
    static_cast<void>(
        engram::prepare_vocabulary_ivf_index(fixture.view(), too_small));
    return fail("undersized vocabulary IVF scratch was accepted");
  } catch (const std::invalid_argument&) {
  }
  try {
    const std::array<float, 2> invalid_query = {
        std::numeric_limits<float>::quiet_NaN(), 0.0F};
    engram::vocabulary_ivf_search_scalar(index, invalid_query, 1, 2,
                                         candidates, scratch);
    return fail("non-finite vocabulary IVF query was accepted");
  } catch (const std::invalid_argument&) {
  }
  try {
    std::array<float, 8> invalid_centroids{};
    invalid_centroids[0] = std::numeric_limits<float>::quiet_NaN();
    Fixture invalid_fixture;
    invalid_fixture.centroids.assign(invalid_centroids.begin(),
                                     invalid_centroids.end());
    const engram::PreparedVocabularyIvfIndex invalid =
        engram::prepare_vocabulary_ivf_index(invalid_fixture.view(), scratch);
    engram::vocabulary_ivf_search_scalar(invalid, query, 1, 2, candidates,
                                         scratch);
    return fail("non-finite vocabulary IVF centroid was accepted in search");
  } catch (const std::invalid_argument&) {
  }

  return 0;
}
