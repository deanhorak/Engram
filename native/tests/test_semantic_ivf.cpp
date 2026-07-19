#include "engram/semantic_ivf.h"

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
  // Centroid rows are [gate_x, gate_y, up_x, up_y].
  std::vector<float> centroids = {
      1.0F, 0.0F, 0.0F, 1.0F,
      0.5F, 0.5F, 1.0F, 0.0F,
      1.0F, 0.0F, 1.0F, 0.0F};
  std::array<std::uint32_t, 4> offsets = {0, 2, 4, 6};
  // Posting order is deliberately not record order in cluster 2.
  std::array<std::uint32_t, 6> postings = {0, 1, 2, 3, 5, 4};
  std::vector<std::uint8_t> gate_codes = {
      4, 2, 4, 2, 3, 3, 4, 2, 4, 2, 2, 4};
  std::vector<std::uint8_t> up_codes = {
      4, 2, 4, 2, 4, 2, 2, 4, 4, 2, 4, 2};
  std::array<float, 2> affine_offsets = {-1.0F, -1.0F};
  std::array<float, 2> affine_scales = {0.5F, 0.5F};
  std::array<std::uint8_t, 6> unused_value_codes{};
  std::array<float, 1> unused_codebook{};

  engram::SemanticIvfIndexView index() const {
    return {3, 2, postings.size(), centroids.data(), offsets.data(),
            postings.data()};
  }

  engram::QuantizedSemanticLayerView layer() const {
    return {6,
            2,
            1,
            1,
            1,
            gate_codes.data(),
            affine_offsets.data(),
            affine_scales.data(),
            up_codes.data(),
            affine_offsets.data(),
            affine_scales.data(),
            unused_value_codes.data(),
            unused_codebook.data()};
  }
};

}  // namespace

int main() {
  Fixture fixture;
  const engram::SemanticIvfIndexView index = fixture.index();
  const engram::QuantizedSemanticLayerView layer = fixture.layer();
  const std::array<float, 2> hidden = {1.0F, 0.0F};
  engram::SemanticIvfScratch scratch(3, 6);
  std::array<std::uint32_t, 3> probes{};
  std::array<engram::SemanticIvfCandidate, 3> candidates{};
  engram::SemanticIvfSearchMetrics metrics;
  engram::validate_semantic_ivf_index(index, layer, scratch);

  engram::semantic_ivf_search_scalar(index, layer, hidden, 2, 3, probes,
                                     candidates, scratch, &metrics);
  if (probes[0] != 2 || probes[1] != 1) {
    return fail("joint centroid probing order mismatch");
  }
  if (candidates[0].index != 4 || !close(candidates[0].proxy_score, 1.0F) ||
      candidates[1].index != 2 ||
      !close(candidates[1].proxy_score,
             static_cast<float>(1.0 / std::sqrt(2.0))) ||
      candidates[2].index != 3 || !close(candidates[2].proxy_score, 0.0F)) {
    return fail("quantized posted-record ranking mismatch");
  }
  // Record 0 has a perfect score but belongs to an unprobed list.
  for (const engram::SemanticIvfCandidate& candidate : candidates) {
    if (candidate.index == 0) {
      return fail("semantic IVF scored a record outside probed postings");
    }
  }
  if (metrics.centroid_records != 3 || metrics.probed_clusters != 2 ||
      metrics.scored_records != 4 || metrics.centroid_bytes != 48 ||
      metrics.posting_bytes != 32 || metrics.key_bytes != 48 ||
      metrics.index_bytes_read != 128 || metrics.zero_norm_query) {
    return fail("semantic IVF metrics mismatch");
  }

  // Reversed posting order cannot disturb record-index tie ordering.
  engram::semantic_ivf_search_scalar(index, layer, hidden, 1, 2, probes,
                                     candidates, scratch);
  if (probes[0] != 2 || candidates[0].index != 4 ||
      candidates[1].index != 5) {
    return fail("semantic IVF candidate ties are not deterministic");
  }

  // A zero query gives source-index cluster and record order and avoids all
  // centroid/key payload reads while still reading the selected CSR lists.
  const std::array<float, 2> zero = {0.0F, 0.0F};
  const std::size_t scratch_bytes = scratch.persistent_bytes();
  engram::semantic_ivf_search_scalar(index, layer, zero, 2, 3, probes,
                                     candidates, scratch, &metrics);
  if (probes[0] != 0 || probes[1] != 1 || candidates[0].index != 0 ||
      candidates[1].index != 1 || candidates[2].index != 2 ||
      !metrics.zero_norm_query || metrics.centroid_bytes != 0 ||
      metrics.posting_bytes != 32 || metrics.key_bytes != 0 ||
      metrics.index_bytes_read != 32 ||
      scratch_bytes != scratch.persistent_bytes()) {
    return fail("zero-query ordering, metrics, or scratch reuse mismatch");
  }

  // Views are non-owning. Raising cluster 1 to a perfect joint centroid makes
  // its lower index win the tie with cluster 2 on the next search.
  fixture.centroids[4] = 1.0F;
  fixture.centroids[5] = 0.0F;
  engram::semantic_ivf_search_scalar(fixture.index(), layer, hidden, 1, 1,
                                     probes, candidates, scratch);
  if (probes[0] != 1 || candidates[0].index != 2) {
    return fail("semantic IVF view unexpectedly owns centroid storage");
  }

  try {
    std::array<std::uint32_t, 6> duplicate = fixture.postings;
    duplicate[1] = duplicate[0];
    engram::SemanticIvfIndexView invalid = fixture.index();
    invalid.postings = duplicate.data();
    engram::validate_semantic_ivf_index(invalid, layer, scratch);
    return fail("duplicate semantic IVF posting was accepted");
  } catch (const std::invalid_argument&) {
  }
  try {
    std::array<std::uint32_t, 6> out_of_bounds = fixture.postings;
    out_of_bounds[0] = 6;
    engram::SemanticIvfIndexView invalid = fixture.index();
    invalid.postings = out_of_bounds.data();
    engram::validate_semantic_ivf_index(invalid, layer, scratch);
    return fail("out-of-bounds semantic IVF posting was accepted");
  } catch (const std::invalid_argument&) {
  }
  try {
    const std::array<std::uint32_t, 4> bad_offsets = {0, 3, 2, 6};
    engram::SemanticIvfIndexView invalid = fixture.index();
    invalid.posting_offsets = bad_offsets.data();
    engram::validate_semantic_ivf_index(invalid, layer, scratch);
    return fail("non-monotonic semantic IVF offsets were accepted");
  } catch (const std::invalid_argument&) {
  }
  try {
    engram::SemanticIvfIndexView invalid = fixture.index();
    invalid.hidden_size = 3;
    engram::semantic_ivf_search_scalar(invalid, layer, hidden, 1, 1, probes,
                                       candidates, scratch);
    return fail("mismatched semantic IVF hidden width was accepted");
  } catch (const std::invalid_argument&) {
  }
  try {
    engram::SemanticIvfScratch too_small(2, 6);
    engram::semantic_ivf_search_scalar(index, layer, hidden, 1, 1, probes,
                                       candidates, too_small);
    return fail("undersized semantic IVF scratch was accepted");
  } catch (const std::invalid_argument&) {
  }
  try {
    std::array<float, 12> invalid_centroids{};
    invalid_centroids[0] = std::numeric_limits<float>::quiet_NaN();
    engram::SemanticIvfIndexView invalid = fixture.index();
    invalid.joint_centroids = invalid_centroids.data();
    engram::semantic_ivf_search_scalar(invalid, layer, hidden, 1, 1, probes,
                                       candidates, scratch);
    return fail("non-finite semantic IVF centroid was accepted");
  } catch (const std::invalid_argument&) {
  }
  try {
    // Probe expansion supplies enough records for the request.
    engram::semantic_ivf_search_scalar(index, layer, hidden, 1, 3, probes,
                                       candidates, scratch, &metrics);
    if (metrics.probed_clusters != 2) {
      return fail("semantic IVF probe expansion count mismatch");
    }
  } catch (const std::invalid_argument&) {
    return fail("semantic IVF failed to expand probes for candidates");
  }

  return 0;
}
