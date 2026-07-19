#include "engram/semantic.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

bool close(const float left, const float right, const float tolerance = 1e-5F) {
  return std::abs(left - right) <= tolerance;
}

int fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}

}  // namespace

int main() {
  // The proxy must use both normalized keys: record 0 wins gate-only, while
  // record 1 is the joint-key winner, matching Python JointKeyRouter.
  const std::vector<float> gate = {
      1.0F, 0.0F, 0.8F, 0.6F, 0.2F, 0.98F};
  const std::vector<float> up = {
      0.0F, 1.0F, 1.0F, 0.0F, 0.5F, 0.866F};
  const std::vector<float> values = {
      1.0F, 0.0F, 0.5F, -1.0F, -0.25F, 2.0F};
  const engram::SemanticLayerView proxy_layer{3, 2, 2, gate.data(), up.data(),
                                               values.data()};
  engram::SemanticScratch proxy_scratch(3);
  const std::array<float, 2> hidden = {1.0F, 0.0F};
  std::array<float, 2> output{};
  std::array<engram::SemanticRecordResult, 1> selected{};
  engram::SemanticReadMetrics metrics;
  engram::semantic_read_scalar(proxy_layer, hidden, 1, 1, output, selected,
                               proxy_scratch, &metrics);
  if (selected[0].index != 1 || selected[0].activation <= 0.0F ||
      metrics.proxy_records != 3 || metrics.candidate_records != 1 ||
      metrics.active_records != 1 || metrics.total_bytes_read != 80) {
    return fail("joint-key proxy selection or metrics mismatch");
  }
  if (!close(output[0], selected[0].activation * 0.5F) ||
      !close(output[1], -selected[0].activation)) {
    return fail("selected value accumulation mismatch");
  }

  // All normalized proxy scores tie. Exact SwiGLU activation and value norms
  // rerank the three candidates, again matching the Python fixture.
  const std::vector<float> rerank_gate = {
      0.2F, 0.0F, 3.0F, 0.0F, 1.0F, 0.0F};
  const std::vector<float> rerank_up = {
      10.0F, 0.0F, 0.5F, 0.0F, -2.0F, 0.0F};
  const std::vector<float> rerank_values = {
      1.0F, 0.0F, 1.0F, 0.0F, 0.1F, 0.0F};
  const engram::SemanticLayerView rerank_layer{
      3, 2, 2, rerank_gate.data(), rerank_up.data(), rerank_values.data()};
  engram::SemanticScratch rerank_scratch(3);
  std::array<engram::SemanticRecordResult, 2> reranked{};
  engram::semantic_read_scalar(rerank_layer, hidden, 3, 2, output, reranked,
                               rerank_scratch, &metrics);
  const double activation0 =
      (0.2 / (1.0 + std::exp(-0.2))) * 10.0;
  const double activation1 =
      (3.0 / (1.0 + std::exp(-3.0))) * 0.5;
  if (reranked[0].index != 1 || reranked[1].index != 0 ||
      !close(reranked[0].activation, static_cast<float>(activation1)) ||
      !close(reranked[1].activation, static_cast<float>(activation0)) ||
      !close(output[0], static_cast<float>(activation0 + activation1)) ||
      !close(output[1], 0.0F)) {
    return fail("exact SwiGLU candidate rerank mismatch");
  }
  if (metrics.proxy_key_bytes != 48 || metrics.exact_key_bytes != 48 ||
      metrics.exact_value_bytes != 24 || metrics.active_value_bytes != 16 ||
      metrics.total_bytes_read != 136) {
    return fail("semantic byte metrics mismatch");
  }

  // Exact ties retain proxy order, not record-index order. Record 1 has the
  // stronger proxy, while zero value vectors make both exact scores zero.
  const std::vector<float> tie_gate = {0.5F, 0.866F, 1.0F, 0.0F};
  const std::vector<float> tie_up = {1.0F, 0.0F, 1.0F, 0.0F};
  const std::vector<float> tie_values(4, 0.0F);
  const engram::SemanticLayerView tie_layer{
      2, 2, 2, tie_gate.data(), tie_up.data(), tie_values.data()};
  engram::SemanticScratch tie_scratch(2);
  std::array<engram::SemanticRecordResult, 2> tied{};
  const std::size_t scratch_bytes = tie_scratch.persistent_bytes();
  engram::semantic_read_scalar(tie_layer, hidden, 2, 2, output, tied,
                               tie_scratch, &metrics);
  if (tied[0].index != 1 || tied[1].index != 0 ||
      scratch_bytes != tie_scratch.persistent_bytes()) {
    return fail("stable exact tie or scratch reuse mismatch");
  }

  // Zero-query proxy ties are stable by source index. Mutating caller-owned
  // values changes the next read, proving the view does not own/copy tensors.
  std::vector<float> mutable_values = {1.0F, 0.0F, 0.0F, 1.0F};
  const std::vector<float> identity = {1.0F, 0.0F, 0.0F, 1.0F};
  const engram::SemanticLayerView non_owning{
      2, 2, 2, identity.data(), identity.data(), mutable_values.data()};
  engram::SemanticScratch reusable(2);
  const std::array<float, 2> zero = {0.0F, 0.0F};
  engram::semantic_read_scalar(non_owning, zero, 2, 1, output, selected,
                               reusable, &metrics);
  if (selected[0].index != 0 || !metrics.zero_norm_query ||
      metrics.proxy_key_bytes != 0) {
    return fail("zero-query stable proxy tie mismatch");
  }
  mutable_values[0] = 3.0F;
  engram::semantic_read_scalar(non_owning, hidden, 1, 1, output, selected,
                               reusable);
  const float expected_activation = 1.0F / (1.0F + std::exp(-1.0F));
  if (!close(output[0], 3.0F * expected_activation)) {
    return fail("semantic layer view unexpectedly owns value storage");
  }

  try {
    engram::SemanticScratch too_small(1);
    engram::semantic_read_scalar(proxy_layer, hidden, 1, 1, output, selected,
                                 too_small);
    return fail("undersized semantic scratch was accepted");
  } catch (const std::invalid_argument&) {
  }
  try {
    const std::array<float, 2> invalid = {
        std::numeric_limits<float>::quiet_NaN(), 0.0F};
    engram::semantic_read_scalar(proxy_layer, invalid, 1, 1, output, selected,
                                 proxy_scratch);
    return fail("non-finite hidden state was accepted");
  } catch (const std::invalid_argument&) {
  }
  return 0;
}
