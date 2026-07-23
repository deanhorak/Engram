#include "engram/dip.h"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool close(const float left, const float right, const float tolerance = 1e-4F) {
  return std::abs(left - right) <= tolerance;
}

int fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}

}  // namespace

int main() {
  // Four coordinates and three records in [coordinate, record] order.
  const std::vector<float> gate = {
      1.0F, 0.9F, 0.2F, 0.0F, 0.0F, 0.0F,
      20.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  const std::vector<float> up = {
      1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F,
      1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  const std::vector<float> values = {
      1.0F, 0.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F, 0.0F,
      0.0F, 0.0F, 1.0F, 0.0F};
  const std::vector<float> gate_rows = {
      1.0F, 0.0F, 20.0F, 0.0F, 0.9F, 0.0F,
      0.0F, 0.0F, 0.2F, 0.0F, 0.0F, 0.0F};
  const std::vector<float> up_rows = {
      1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F};
  const std::array<float, 3> norms = {1.0F, 1.0F, 1.0F};
  const engram::DIPLayerView layer{3, 4, gate.data(), up.data(),
                                   values.data(), norms.data()};
  const engram::DIPLayerView dual_layer{
      3,          4,              gate.data(),      up.data(),
      values.data(), norms.data(), gate_rows.data(), up_rows.data()};
  const std::array<float, 4> hidden = {2.0F, 0.0F, 0.1F, 0.0F};
  std::array<float, 4> output{};
  std::array<float, 4> dense{};
  std::array<engram::DIPRecordResult, 3> selected{};
  engram::DIPScratch scratch(4, 3);
  engram::DIPReadMetrics metrics;

  engram::dip_read_scalar(layer, hidden, 1, 2, 1, output, selected, scratch,
                          &metrics);
  if (selected[0].index != 0 || selected[0].activation <= 0.0F ||
      !close(output[0], selected[0].activation) || !close(output[1], 0.0F)) {
    return fail("candidate completion or selected-row accumulation mismatch");
  }
  if (metrics.partial_projection_bytes != 24 ||
      metrics.candidate_completion_bytes != 48 ||
      metrics.selected_down_bytes != 16 || metrics.logical_weight_bytes != 88 ||
      metrics.executed_weight_bytes != 88 ||
      metrics.dense_weight_bytes != 144) {
    return fail("DIP logical byte accounting mismatch");
  }

  std::array<float, 4> streamed_output{};
  std::array<engram::DIPRecordResult, 1> streamed_selected{};
  engram::dip_read_scalar(
      layer, hidden, 1, 2, 1, streamed_output, streamed_selected, scratch,
      &metrics, engram::DIPCompletionMode::FullCoordinateStream);
  if (streamed_selected[0].index != selected[0].index ||
      !close(streamed_output[0], output[0]) ||
      metrics.logical_weight_bytes != 88 || metrics.executed_weight_bytes != 112) {
    return fail("streamed completion parity or executed-byte accounting mismatch");
  }
  std::array<float, 4> record_output{};
  std::array<engram::DIPRecordResult, 1> record_selected{};
  engram::dip_read_scalar(
      dual_layer, hidden, 1, 2, 1, record_output, record_selected, scratch,
      &metrics, engram::DIPCompletionMode::RecordMajorGather);
  if (record_selected[0].index != selected[0].index ||
      !close(record_output[0], output[0]) || metrics.logical_weight_bytes != 88 ||
      metrics.executed_weight_bytes != 88) {
    return fail("record-major completion parity or byte accounting mismatch");
  }
  engram::dip_read_scalar(
      dual_layer, hidden, 1, 2, 1, record_output, record_selected, scratch,
      &metrics, engram::DIPCompletionMode::RecordMajorStream);
  if (record_selected[0].index != selected[0].index ||
      !close(record_output[0], output[0]) || metrics.logical_weight_bytes != 88 ||
      metrics.executed_weight_bytes != 104 ||
      metrics.cache_line_weight_bytes != 104) {
    return fail("record-major streamed completion parity or byte accounting mismatch");
  }

  // With all input blocks, candidates, and records selected, the routed path
  // must be numerically identical to the dense reference.
  const std::size_t scratch_bytes = scratch.persistent_bytes();
  engram::dip_read_scalar(layer, hidden, 4, 3, 3, output, selected, scratch);
  engram::dip_dense_scalar(layer, hidden, dense, scratch);
  for (std::size_t index = 0; index < output.size(); ++index) {
    if (!close(output[index], dense[index])) {
      return fail("all-record DIP output does not match dense reference");
    }
  }
  if (scratch_bytes != scratch.persistent_bytes()) {
    return fail("DIP scratch allocated during reuse");
  }

  try {
    engram::dip_read_scalar(layer, hidden, 0, 2, 1, output, selected, scratch);
    return fail("zero selected input blocks were accepted");
  } catch (const std::invalid_argument&) {
  }
  return 0;
}
