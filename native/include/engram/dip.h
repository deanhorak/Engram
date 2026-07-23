#pragma once

#include "engram/npy.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace engram {

struct DIPLayerView {
  std::size_t records{};
  std::size_t hidden_size{};
  const float* gate_coordinates{};  // [hidden_size, records]
  const float* up_coordinates{};    // [hidden_size, records]
  const float* down_rows{};    // [records, hidden_size]
  const float* value_norms{};  // [records]
  const float* gate_rows{};    // optional [records, hidden_size]
  const float* up_rows{};      // optional [records, hidden_size]
};

struct DIPRecordResult {
  std::uint32_t index{};
  float proxy_score{};
  float activation{};
  float exact_score{};
};

struct DIPReadMetrics {
  std::size_t selected_input_coordinates{};
  std::size_t candidate_records{};
  std::size_t active_records{};
  std::size_t partial_projection_bytes{};
  std::size_t candidate_completion_bytes{};
  std::size_t selected_down_bytes{};
  std::size_t logical_weight_bytes{};
  std::size_t executed_weight_bytes{};
  std::size_t cache_line_weight_bytes{};
  std::size_t dense_weight_bytes{};
  std::size_t coordinate_selection_ns{};
  std::size_t partial_projection_ns{};
  std::size_t proxy_scoring_ns{};
  std::size_t candidate_selection_ns{};
  std::size_t candidate_completion_ns{};
  std::size_t exact_scoring_ns{};
  std::size_t exact_selection_ns{};
  std::size_t down_accumulation_ns{};
};

enum class DIPCompletionMode {
  CandidateGather,
  FullCoordinateStream,
  RecordMajorGather,
  RecordMajorStream
};

class DIPScratch {
 public:
  struct CoordinateRank {
    std::uint32_t index{};
    float score{};
  };
  struct RecordRank {
    std::uint32_t index{};
    float proxy_score{};
    float gate{};
    float up{};
    float activation{};
    float exact_score{};
  };

  DIPScratch(std::size_t hidden_capacity, std::size_t record_capacity);
  [[nodiscard]] std::size_t hidden_capacity() const noexcept;
  [[nodiscard]] std::size_t record_capacity() const noexcept;
  [[nodiscard]] std::size_t persistent_bytes() const noexcept;

 private:
  std::vector<CoordinateRank> coordinate_ranking_;
  std::vector<std::uint8_t> selected_coordinates_;
  std::vector<std::uint32_t> omitted_coordinates_;
  std::vector<std::uint8_t> candidate_cache_lines_;
  std::vector<std::uint32_t> candidate_indices_;
  std::vector<float> proxy_scores_;
  std::vector<RecordRank> proxy_ranking_;
  std::vector<RecordRank> exact_ranking_;
  std::vector<float> dense_gate_;
  std::vector<float> dense_up_;
  std::vector<float> candidate_gate_;
  std::vector<float> candidate_up_;

  friend void dip_read_scalar(const DIPLayerView&, std::span<const float>,
                              std::size_t, std::size_t, std::size_t,
                              std::span<float>, std::span<DIPRecordResult>,
                              DIPScratch&, DIPReadMetrics*, DIPCompletionMode);
  friend void dip_dense_scalar(const DIPLayerView&, std::span<const float>,
                               std::span<float>, DIPScratch&);
};

// Predictor-free coordinate-major selection. The partial pass streams all
// records only for selected input coordinates. Exact completion gathers
// omitted coordinates only for candidates, then reads selected down rows.
void dip_read_scalar(
    const DIPLayerView& layer, std::span<const float> hidden,
    std::size_t selected_input_coordinates, std::size_t candidate_count,
    std::size_t top_k, std::span<float> output,
    std::span<DIPRecordResult> selected, DIPScratch& scratch,
    DIPReadMetrics* metrics = nullptr,
    DIPCompletionMode completion_mode = DIPCompletionMode::CandidateGather);

// Dense reference over the same coordinate-major storage, used for parity and timing.
void dip_dense_scalar(const DIPLayerView& layer, std::span<const float> hidden,
                      std::span<float> output, DIPScratch& scratch);

class DIPLayerStorage {
 public:
  explicit DIPLayerStorage(
      const std::filesystem::path& directory,
      NpyLoadMode mode = NpyLoadMode::MemoryMap);

  [[nodiscard]] DIPLayerView view() const noexcept;

 private:
  NpyArray config_;
  NpyArray gate_coordinates_;
  NpyArray up_coordinates_;
  NpyArray down_rows_;
  NpyArray value_norms_;
  std::unique_ptr<NpyArray> gate_rows_;
  std::unique_ptr<NpyArray> up_rows_;
  std::size_t records_{};
  std::size_t hidden_size_{};
};

}  // namespace engram
