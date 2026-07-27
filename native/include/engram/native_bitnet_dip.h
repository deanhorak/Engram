#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace engram {

enum class NativeBitNetDIPRMSEstimator : std::uint32_t {
  corrected_proxy = 1,
  candidate_ratio = 2,
};

enum class NativeBitNetDIPAuditStrategy : std::uint32_t {
  none = 0,
  top_proxy_raw_square = 2,
};

struct NativeBitNetDIPPolicy {
  std::size_t input_coordinates{};
  std::size_t candidate_count{};
  std::size_t minimum_top_k{};
  std::size_t maximum_top_k{};
  float energy_target{};
  std::size_t rms_audit_count{};
  NativeBitNetDIPRMSEstimator rms_estimator{};
  NativeBitNetDIPAuditStrategy rms_audit_strategy{};
};

struct NativeBitNetDIPMetrics {
  std::uint64_t elapsed_ns{};
  std::uint64_t coordinate_stream_bytes{};
  std::uint64_t candidate_completion_bytes{};
  std::uint64_t gain_stream_bytes{};
  std::uint64_t down_norm_stream_bytes{};
  std::uint64_t selected_down_stream_bytes{};
  std::uint64_t layer_metadata_bytes{};
  std::uint64_t scheduled_cache_line_bytes{};
  std::uint64_t scratch_bytes{};
  std::uint64_t rows{};
  std::uint64_t threads{};
  std::uint64_t input_coordinates{};
  std::uint64_t candidate_count{};
  std::uint64_t selected_count_total{};
  std::uint64_t selected_count_min{};
  std::uint64_t selected_count_max{};
};

// CPU-only, memory-mapped implementation of the authenticated BitNet DIP
// index v2 plus the existing record-major native BitNet MLP artifact.
class NativeBitNetDIPKernel {
 public:
  NativeBitNetDIPKernel(const std::filesystem::path& record_artifact,
                        const std::filesystem::path& coordinate_index,
                        std::size_t thread_count);
  ~NativeBitNetDIPKernel();

  NativeBitNetDIPKernel(const NativeBitNetDIPKernel&) = delete;
  NativeBitNetDIPKernel& operator=(const NativeBitNetDIPKernel&) = delete;
  NativeBitNetDIPKernel(NativeBitNetDIPKernel&&) noexcept;
  NativeBitNetDIPKernel& operator=(NativeBitNetDIPKernel&&) noexcept;

  [[nodiscard]] std::size_t layer_count() const noexcept;
  [[nodiscard]] std::size_t hidden_size() const noexcept;
  [[nodiscard]] std::size_t intermediate_size() const noexcept;
  [[nodiscard]] std::size_t thread_count() const noexcept;
  [[nodiscard]] std::size_t record_artifact_bytes() const noexcept;
  [[nodiscard]] std::size_t coordinate_index_bytes() const noexcept;
  [[nodiscard]] std::size_t global_metadata_cache_line_bytes() const noexcept;
  [[nodiscard]] const NativeBitNetDIPPolicy& policy(
      std::size_t layer) const;

  // Input/output are row-major BF16 bit patterns [rows, hidden].  The optional
  // selected_counts output has one uint32 value per row.
  void forward_bf16(std::size_t layer,
                    std::span<const std::uint16_t> input, std::size_t rows,
                    std::span<std::uint16_t> output,
                    std::span<std::uint32_t> selected_counts = {},
                    NativeBitNetDIPMetrics* metrics = nullptr);

  // Evaluation-only route identity. Diagnostic copies occur after the timed
  // sparse kernel and are excluded from traffic/latency metrics.
  void forward_debug_bf16(
      std::size_t layer, std::span<const std::uint16_t> input,
      std::size_t rows, std::span<std::uint16_t> output,
      std::span<std::uint32_t> selected_counts,
      std::span<std::uint32_t> input_coordinate_ids,
      std::span<std::uint32_t> candidate_ids,
      std::span<std::uint32_t> selected_record_ids,
      NativeBitNetDIPMetrics* metrics = nullptr);

  // Evaluation-only exact dense native-BF16 teacher utility ordering. This is
  // intentionally separate from sparse metrics and is never called by
  // inference.
  void teacher_top_k_bf16(std::size_t layer,
                          std::span<const std::uint16_t> input,
                          std::size_t rows, std::size_t top_k,
                          std::span<std::uint32_t> teacher_record_ids,
                          std::span<std::uint32_t> positive_utility_counts = {});

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace engram
