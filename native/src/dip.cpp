#include "engram/dip.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace engram {
namespace {

using Clock = std::chrono::steady_clock;

std::size_t elapsed_ns(const Clock::time_point start) {
  return static_cast<std::size_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
          .count());
}

float silu(const float value) noexcept {
  const float exponential = std::exp(-std::abs(value));
  const float sigmoid = value >= 0.0F ? 1.0F / (1.0F + exponential)
                                      : exponential / (1.0F + exponential);
  return value * sigmoid;
}

std::size_t product(const std::size_t left, const std::size_t right,
                    const char* message) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::invalid_argument(message);
  }
  return left * right;
}

std::size_t sum(const std::size_t left, const std::size_t right,
                const char* message) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::invalid_argument(message);
  }
  return left + right;
}

void validate(const DIPLayerView& layer) {
  if (layer.records == 0 || layer.hidden_size == 0 ||
      layer.records > std::numeric_limits<std::uint32_t>::max() ||
      layer.hidden_size > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("invalid DIP layer dimensions");
  }
  if (layer.gate_coordinates == nullptr || layer.up_coordinates == nullptr ||
      layer.down_rows == nullptr || layer.value_norms == nullptr) {
    throw std::invalid_argument("DIP layer pointers must not be null");
  }
  static_cast<void>(product(layer.records, layer.hidden_size,
                            "DIP dimensions overflow"));
}

bool better_coordinate(const DIPScratch::CoordinateRank& left,
                       const DIPScratch::CoordinateRank& right) noexcept {
  return left.score != right.score ? left.score > right.score
                                   : left.index < right.index;
}

bool better_proxy(const DIPScratch::RecordRank& left,
                  const DIPScratch::RecordRank& right) noexcept {
  return left.proxy_score != right.proxy_score
             ? left.proxy_score > right.proxy_score
             : left.index < right.index;
}

bool better_exact(const DIPScratch::RecordRank& left,
                  const DIPScratch::RecordRank& right) noexcept {
  return left.exact_score != right.exact_score
             ? left.exact_score > right.exact_score
             : left.index < right.index;
}

void finite(const float value, const char* message) {
  if (!std::isfinite(value)) throw std::invalid_argument(message);
}

}  // namespace

DIPScratch::DIPScratch(const std::size_t hidden_capacity,
                       const std::size_t record_capacity)
    : coordinate_ranking_(hidden_capacity),
      selected_coordinates_(hidden_capacity),
      omitted_coordinates_(hidden_capacity),
      candidate_cache_lines_(std::max(
          record_capacity,
          (hidden_capacity + (64 / sizeof(float)) - 1) /
              (64 / sizeof(float)))),
      candidate_indices_(record_capacity),
      proxy_scores_(record_capacity),
      proxy_ranking_(record_capacity),
      exact_ranking_(record_capacity),
      dense_gate_(record_capacity),
      dense_up_(record_capacity),
      candidate_gate_(record_capacity),
      candidate_up_(record_capacity) {
  if (hidden_capacity == 0 || record_capacity == 0 ||
      hidden_capacity > std::numeric_limits<std::uint32_t>::max() ||
      record_capacity > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("DIP scratch capacities must be positive uint32 values");
  }
}

std::size_t DIPScratch::hidden_capacity() const noexcept {
  return coordinate_ranking_.size();
}

std::size_t DIPScratch::record_capacity() const noexcept {
  return proxy_ranking_.size();
}

std::size_t DIPScratch::persistent_bytes() const noexcept {
  return coordinate_ranking_.capacity() * sizeof(CoordinateRank) +
         selected_coordinates_.capacity() * sizeof(std::uint8_t) +
         omitted_coordinates_.capacity() * sizeof(std::uint32_t) +
         candidate_cache_lines_.capacity() * sizeof(std::uint8_t) +
         candidate_indices_.capacity() * sizeof(std::uint32_t) +
         proxy_scores_.capacity() * sizeof(float) +
         (proxy_ranking_.capacity() + exact_ranking_.capacity()) *
             sizeof(RecordRank) +
         (dense_gate_.capacity() + dense_up_.capacity()) * sizeof(float) +
         (candidate_gate_.capacity() + candidate_up_.capacity()) *
             sizeof(float);
}

void dip_read_scalar(
    const DIPLayerView& layer, const std::span<const float> hidden,
    const std::size_t selected_input_coordinates,
    const std::size_t candidate_count, const std::size_t top_k,
    const std::span<float> output,
    const std::span<DIPRecordResult> selected, DIPScratch& scratch,
    DIPReadMetrics* const metrics, const DIPCompletionMode completion_mode) {
  validate(layer);
  if (metrics != nullptr) *metrics = DIPReadMetrics{};
  auto phase_start = Clock::now();
  if (hidden.size() != layer.hidden_size || output.size() < layer.hidden_size) {
    throw std::invalid_argument("DIP hidden or output width mismatch");
  }
  if (selected_input_coordinates == 0 ||
      selected_input_coordinates > layer.hidden_size) {
    throw std::invalid_argument("selected input coordinates must be within the layer");
  }
  if (candidate_count == 0 || candidate_count > layer.records || top_k == 0 ||
      top_k > candidate_count || selected.size() < top_k) {
    throw std::invalid_argument("invalid DIP candidate or top-k count");
  }
  if (scratch.hidden_capacity() < layer.hidden_size ||
      scratch.record_capacity() < layer.records) {
    throw std::invalid_argument("DIP scratch capacity is too small");
  }
  if ((completion_mode == DIPCompletionMode::RecordMajorGather ||
       completion_mode == DIPCompletionMode::RecordMajorStream) &&
      (layer.gate_rows == nullptr || layer.up_rows == nullptr)) {
    throw std::invalid_argument(
        "record-major DIP completion requires dual-layout weights");
  }
  std::fill(scratch.selected_coordinates_.begin(),
            scratch.selected_coordinates_.begin() + layer.hidden_size, 0U);
  for (std::size_t coordinate = 0; coordinate < layer.hidden_size; ++coordinate) {
    finite(hidden[coordinate], "DIP hidden must be finite");
    scratch.coordinate_ranking_[coordinate] = {
        static_cast<std::uint32_t>(coordinate), std::abs(hidden[coordinate])};
  }
  std::partial_sort(
      scratch.coordinate_ranking_.begin(),
      scratch.coordinate_ranking_.begin() + selected_input_coordinates,
      scratch.coordinate_ranking_.begin() + layer.hidden_size,
      better_coordinate);
  for (std::size_t rank = 0; rank < selected_input_coordinates; ++rank) {
    scratch.selected_coordinates_[scratch.coordinate_ranking_[rank].index] = 1U;
  }
  std::size_t omitted_count = 0;
  for (std::size_t coordinate = 0; coordinate < layer.hidden_size;
       ++coordinate) {
    if (scratch.selected_coordinates_[coordinate] == 0U) {
      scratch.omitted_coordinates_[omitted_count++] =
          static_cast<std::uint32_t>(coordinate);
    }
  }
  if (metrics != nullptr) {
    metrics->coordinate_selection_ns = elapsed_ns(phase_start);
    phase_start = Clock::now();
  }
  for (std::size_t record = 0; record < layer.records; ++record) {
    scratch.proxy_ranking_[record] = {
        static_cast<std::uint32_t>(record), 0.0, 0.0, 0.0, 0.0, 0.0};
  }
  std::fill_n(scratch.dense_gate_.begin(), layer.records, 0.0);
  std::fill_n(scratch.dense_up_.begin(), layer.records, 0.0);
  for (std::size_t rank = 0; rank < selected_input_coordinates; ++rank) {
    const std::size_t coordinate = scratch.coordinate_ranking_[rank].index;
    const std::size_t offset = coordinate * layer.records;
    const float input = hidden[coordinate];
    for (std::size_t record = 0; record < layer.records; ++record) {
      const float gate = layer.gate_coordinates[offset + record];
      const float up = layer.up_coordinates[offset + record];
      scratch.dense_gate_[record] += input * gate;
      scratch.dense_up_[record] += input * up;
    }
  }
  if (metrics != nullptr) {
    metrics->partial_projection_ns = elapsed_ns(phase_start);
    phase_start = Clock::now();
  }
  for (std::size_t record = 0; record < layer.records; ++record) {
    const float norm = layer.value_norms[record];
    finite(norm, "DIP value norms must be finite");
    if (norm < 0.0F) throw std::invalid_argument("DIP value norms must be non-negative");
    auto& item = scratch.proxy_ranking_[record];
    item.proxy_score =
        std::abs(silu(scratch.dense_gate_[record]) * scratch.dense_up_[record]) *
        norm;
    scratch.proxy_scores_[record] = item.proxy_score;
  }
  if (metrics != nullptr) {
    metrics->proxy_scoring_ns = elapsed_ns(phase_start);
    phase_start = Clock::now();
  }
  if (candidate_count < layer.records) {
    std::nth_element(scratch.proxy_ranking_.begin(),
                     scratch.proxy_ranking_.begin() + candidate_count,
                     scratch.proxy_ranking_.begin() + layer.records,
                     better_proxy);
  }
  for (std::size_t rank = 0; rank < candidate_count; ++rank) {
    scratch.candidate_indices_[rank] = scratch.proxy_ranking_[rank].index;
  }
  std::sort(scratch.candidate_indices_.begin(),
            scratch.candidate_indices_.begin() + candidate_count);
  if (completion_mode != DIPCompletionMode::FullCoordinateStream &&
      completion_mode != DIPCompletionMode::RecordMajorStream) {
    for (std::size_t rank = 0; rank < candidate_count; ++rank) {
      const auto record = scratch.candidate_indices_[rank];
      scratch.candidate_gate_[rank] = scratch.dense_gate_[record];
      scratch.candidate_up_[rank] = scratch.dense_up_[record];
    }
  }
  if (metrics != nullptr) {
    metrics->candidate_selection_ns = elapsed_ns(phase_start);
    phase_start = Clock::now();
  }
  if (completion_mode == DIPCompletionMode::RecordMajorStream) {
    for (std::size_t rank = 0; rank < candidate_count; ++rank) {
      const auto record = scratch.candidate_indices_[rank];
      const std::size_t offset =
          static_cast<std::size_t>(record) * layer.hidden_size;
      float gate = 0.0F;
      float up = 0.0F;
      for (std::size_t coordinate = 0; coordinate < layer.hidden_size;
           ++coordinate) {
        gate += hidden[coordinate] * layer.gate_rows[offset + coordinate];
        up += hidden[coordinate] * layer.up_rows[offset + coordinate];
      }
      scratch.candidate_gate_[rank] = gate;
      scratch.candidate_up_[rank] = up;
    }
  } else if (completion_mode == DIPCompletionMode::RecordMajorGather) {
    for (std::size_t rank = 0; rank < candidate_count; ++rank) {
      const auto record = scratch.candidate_indices_[rank];
      const std::size_t offset =
          static_cast<std::size_t>(record) * layer.hidden_size;
      for (std::size_t omitted_rank = 0; omitted_rank < omitted_count;
           ++omitted_rank) {
        const std::size_t coordinate =
            scratch.omitted_coordinates_[omitted_rank];
        const float input = hidden[coordinate];
        scratch.candidate_gate_[rank] +=
            input * layer.gate_rows[offset + coordinate];
        scratch.candidate_up_[rank] +=
            input * layer.up_rows[offset + coordinate];
      }
    }
  } else {
    for (std::size_t omitted_rank = 0; omitted_rank < omitted_count;
         ++omitted_rank) {
      const std::size_t coordinate = scratch.omitted_coordinates_[omitted_rank];
      const std::size_t offset = coordinate * layer.records;
      const float input = hidden[coordinate];
      if (completion_mode == DIPCompletionMode::CandidateGather) {
        for (std::size_t rank = 0; rank < candidate_count; ++rank) {
          const auto record = scratch.candidate_indices_[rank];
          scratch.candidate_gate_[rank] +=
              input * layer.gate_coordinates[offset + record];
          scratch.candidate_up_[rank] +=
              input * layer.up_coordinates[offset + record];
        }
      } else {
        for (std::size_t record = 0; record < layer.records; ++record) {
          scratch.dense_gate_[record] +=
              input * layer.gate_coordinates[offset + record];
          scratch.dense_up_[record] +=
              input * layer.up_coordinates[offset + record];
        }
      }
    }
  }
  if (completion_mode == DIPCompletionMode::FullCoordinateStream) {
    for (std::size_t rank = 0; rank < candidate_count; ++rank) {
      const auto record = scratch.candidate_indices_[rank];
      scratch.candidate_gate_[rank] = scratch.dense_gate_[record];
      scratch.candidate_up_[rank] = scratch.dense_up_[record];
    }
  }
  if (metrics != nullptr) {
    metrics->candidate_completion_ns = elapsed_ns(phase_start);
    phase_start = Clock::now();
  }
  for (std::size_t rank = 0; rank < candidate_count; ++rank) {
    const auto record = scratch.candidate_indices_[rank];
    const float activation =
        silu(scratch.candidate_gate_[rank]) * scratch.candidate_up_[rank];
    scratch.exact_ranking_[rank] = {
        record,
        scratch.proxy_scores_[record],
        scratch.candidate_gate_[rank],
        scratch.candidate_up_[rank],
        activation,
        std::abs(activation) * layer.value_norms[record]};
  }
  if (metrics != nullptr) {
    metrics->exact_scoring_ns = elapsed_ns(phase_start);
    phase_start = Clock::now();
  }
  if (top_k < candidate_count) {
    std::nth_element(scratch.exact_ranking_.begin(),
                     scratch.exact_ranking_.begin() + top_k,
                     scratch.exact_ranking_.begin() + candidate_count,
                     better_exact);
  }
  std::sort(scratch.exact_ranking_.begin(),
            scratch.exact_ranking_.begin() + top_k, better_exact);
  if (metrics != nullptr) {
    metrics->exact_selection_ns = elapsed_ns(phase_start);
    phase_start = Clock::now();
  }
  std::fill_n(output.begin(), layer.hidden_size, 0.0F);
  for (std::size_t rank = 0; rank < top_k; ++rank) {
    const auto& item = scratch.exact_ranking_[rank];
    selected[rank] = {item.index, static_cast<float>(item.proxy_score),
                      static_cast<float>(item.activation),
                      static_cast<float>(item.exact_score)};
    const std::size_t offset = static_cast<std::size_t>(item.index) * layer.hidden_size;
    for (std::size_t coordinate = 0; coordinate < layer.hidden_size; ++coordinate) {
      const float value = layer.down_rows[offset + coordinate];
      output[coordinate] += item.activation * value;
    }
  }
  if (metrics != nullptr) {
    metrics->down_accumulation_ns = elapsed_ns(phase_start);
  }
  if (metrics != nullptr) {
    constexpr std::size_t cache_line = 64;
    constexpr std::size_t records_per_line = cache_line / sizeof(float);
    const std::size_t line_count =
        (layer.records + records_per_line - 1) / records_per_line;
    std::fill(scratch.candidate_cache_lines_.begin(),
              scratch.candidate_cache_lines_.begin() + line_count, 0U);
    std::size_t candidate_lines = 0;
    for (std::size_t rank = 0; rank < candidate_count; ++rank) {
      const std::size_t line = scratch.candidate_indices_[rank] / records_per_line;
      if (scratch.candidate_cache_lines_[line] == 0U) {
        scratch.candidate_cache_lines_[line] = 1U;
        ++candidate_lines;
      }
    }
    metrics->selected_input_coordinates = selected_input_coordinates;
    metrics->candidate_records = candidate_count;
    metrics->active_records = top_k;
    metrics->partial_projection_bytes = product(
        product(product(2, layer.records, "DIP metrics overflow"),
                selected_input_coordinates, "DIP metrics overflow"),
        sizeof(float), "DIP metrics overflow");
    metrics->candidate_completion_bytes = product(
        product(product(2, candidate_count, "DIP metrics overflow"),
                layer.hidden_size - selected_input_coordinates,
                "DIP metrics overflow"),
        sizeof(float), "DIP metrics overflow");
    metrics->selected_down_bytes = product(
        product(top_k, layer.hidden_size, "DIP metrics overflow"),
        sizeof(float), "DIP metrics overflow");
    metrics->logical_weight_bytes = sum(
        sum(metrics->partial_projection_bytes,
            metrics->candidate_completion_bytes, "DIP metrics overflow"),
        metrics->selected_down_bytes, "DIP metrics overflow");
    metrics->executed_weight_bytes = metrics->logical_weight_bytes;
    if (completion_mode == DIPCompletionMode::FullCoordinateStream) {
      const std::size_t streamed_completion_bytes = product(
          product(product(2, layer.records, "DIP metrics overflow"),
                  layer.hidden_size - selected_input_coordinates,
                  "DIP metrics overflow"),
          sizeof(float), "DIP metrics overflow");
      metrics->executed_weight_bytes = sum(
          sum(metrics->partial_projection_bytes, streamed_completion_bytes,
              "DIP metrics overflow"),
          metrics->selected_down_bytes, "DIP metrics overflow");
    } else if (completion_mode == DIPCompletionMode::RecordMajorStream) {
      const std::size_t streamed_completion_bytes = product(
          product(product(2, candidate_count, "DIP metrics overflow"),
                  layer.hidden_size, "DIP metrics overflow"),
          sizeof(float), "DIP metrics overflow");
      metrics->executed_weight_bytes = sum(
          sum(metrics->partial_projection_bytes, streamed_completion_bytes,
              "DIP metrics overflow"),
          metrics->selected_down_bytes, "DIP metrics overflow");
    }
    std::size_t completion_cache_bytes = 0;
    if (completion_mode == DIPCompletionMode::FullCoordinateStream) {
      completion_cache_bytes = product(
          product(product(2, layer.records, "DIP metrics overflow"),
                  layer.hidden_size - selected_input_coordinates,
                  "DIP metrics overflow"),
          sizeof(float), "DIP metrics overflow");
    } else if (completion_mode == DIPCompletionMode::RecordMajorStream) {
      completion_cache_bytes = product(
          product(product(2, candidate_count, "DIP metrics overflow"),
                  layer.hidden_size, "DIP metrics overflow"),
          sizeof(float), "DIP metrics overflow");
    } else if (completion_mode == DIPCompletionMode::RecordMajorGather) {
      const std::size_t coordinate_line_count =
          (layer.hidden_size + records_per_line - 1) / records_per_line;
      std::fill(scratch.candidate_cache_lines_.begin(),
                scratch.candidate_cache_lines_.begin() + coordinate_line_count,
                0U);
      std::size_t omitted_lines = 0;
      for (std::size_t omitted_rank = 0; omitted_rank < omitted_count;
           ++omitted_rank) {
        const std::size_t line =
            scratch.omitted_coordinates_[omitted_rank] / records_per_line;
        if (scratch.candidate_cache_lines_[line] == 0U) {
          scratch.candidate_cache_lines_[line] = 1U;
          ++omitted_lines;
        }
      }
      completion_cache_bytes = product(
          product(product(2, candidate_count, "DIP metrics overflow"),
                  omitted_lines, "DIP metrics overflow"),
          cache_line, "DIP metrics overflow");
    } else {
      completion_cache_bytes = product(
          product(product(2, candidate_lines, "DIP metrics overflow"),
                  layer.hidden_size - selected_input_coordinates,
                  "DIP metrics overflow"),
          cache_line, "DIP metrics overflow");
    }
    metrics->cache_line_weight_bytes = sum(
        sum(metrics->partial_projection_bytes, completion_cache_bytes,
            "DIP metrics overflow"),
        metrics->selected_down_bytes, "DIP metrics overflow");
    metrics->dense_weight_bytes = product(
        product(product(3, layer.records, "DIP metrics overflow"),
                layer.hidden_size, "DIP metrics overflow"),
        sizeof(float), "DIP metrics overflow");
  }
}

void dip_dense_scalar(const DIPLayerView& layer,
                      const std::span<const float> hidden,
                      const std::span<float> output, DIPScratch& scratch) {
  validate(layer);
  if (hidden.size() != layer.hidden_size || output.size() < layer.hidden_size) {
    throw std::invalid_argument("DIP dense hidden or output width mismatch");
  }
  if (scratch.record_capacity() < layer.records) {
    throw std::invalid_argument("DIP dense scratch capacity is too small");
  }
  std::fill_n(scratch.dense_gate_.begin(), layer.records, 0.0F);
  std::fill_n(scratch.dense_up_.begin(), layer.records, 0.0F);
  for (std::size_t coordinate = 0; coordinate < layer.hidden_size; ++coordinate) {
    const std::size_t offset = coordinate * layer.records;
    for (std::size_t record = 0; record < layer.records; ++record) {
      scratch.dense_gate_[record] +=
          hidden[coordinate] * layer.gate_coordinates[offset + record];
      scratch.dense_up_[record] +=
          hidden[coordinate] * layer.up_coordinates[offset + record];
    }
  }
  std::fill_n(output.begin(), layer.hidden_size, 0.0F);
  for (std::size_t record = 0; record < layer.records; ++record) {
    const float activation =
        silu(scratch.dense_gate_[record]) * scratch.dense_up_[record];
    const std::size_t offset = record * layer.hidden_size;
    for (std::size_t coordinate = 0; coordinate < layer.hidden_size; ++coordinate) {
      output[coordinate] += activation * layer.down_rows[offset + coordinate];
    }
  }
}

DIPLayerStorage::DIPLayerStorage(const std::filesystem::path& directory,
                                 const NpyLoadMode mode)
    : config_(load_npy(directory / "config.npy", mode)),
      gate_coordinates_(load_npy(directory / "gate_coordinates.npy", mode)),
      up_coordinates_(load_npy(directory / "up_coordinates.npy", mode)),
      down_rows_(load_npy(directory / "down_rows.npy", mode)),
      value_norms_(load_npy(directory / "value_norms.npy", mode)) {
  const auto config = config_.uint32();
  if (config_.shape() != std::vector<std::size_t>{4} ||
      (config[0] != 2U && config[0] != 3U) ||
      config[3] != 64U) {
    throw NpyError("unsupported DIP binary configuration");
  }
  records_ = config[1];
  hidden_size_ = config[2];
  if (records_ == 0 || hidden_size_ == 0 ||
      gate_coordinates_.shape() !=
          std::vector<std::size_t>{hidden_size_, records_} ||
      up_coordinates_.shape() != gate_coordinates_.shape() ||
      down_rows_.shape() != std::vector<std::size_t>{records_, hidden_size_} ||
      value_norms_.shape() != std::vector<std::size_t>{records_}) {
    throw NpyError("DIP binary array shape mismatch");
  }
  static_cast<void>(gate_coordinates_.float32());
  static_cast<void>(up_coordinates_.float32());
  static_cast<void>(down_rows_.float32());
  static_cast<void>(value_norms_.float32());
  if (config[0] >= 3U) {
    gate_rows_ =
        std::make_unique<NpyArray>(load_npy(directory / "gate_rows.npy", mode));
    up_rows_ =
        std::make_unique<NpyArray>(load_npy(directory / "up_rows.npy", mode));
    if (gate_rows_->shape() !=
            std::vector<std::size_t>{records_, hidden_size_} ||
        up_rows_->shape() != gate_rows_->shape()) {
      throw NpyError("DIP record-major array shape mismatch");
    }
    static_cast<void>(gate_rows_->float32());
    static_cast<void>(up_rows_->float32());
  }
}

DIPLayerView DIPLayerStorage::view() const noexcept {
  return {records_, hidden_size_, gate_coordinates_.float32().data(),
          up_coordinates_.float32().data(), down_rows_.float32().data(),
          value_norms_.float32().data(),
          gate_rows_ ? gate_rows_->float32().data() : nullptr,
          up_rows_ ? up_rows_->float32().data() : nullptr};
}

}  // namespace engram
