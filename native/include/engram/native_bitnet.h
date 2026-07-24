#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace engram {

struct NativeBitNetMetrics {
  std::uint64_t elapsed_ns{};
  std::uint64_t gate_up_stream_bytes{};
  std::uint64_t norm_stream_bytes{};
  std::uint64_t down_stream_bytes{};
  std::uint64_t layer_metadata_bytes{};
  std::uint64_t scheduled_cache_line_bytes{};
  std::uint64_t scratch_bytes{};
  std::uint64_t rows{};
  std::uint64_t threads{};
};

// Memory-mapped, fail-closed reader for native_bitnet_phase_base3_v1.
// The mapped phase streams remain packed; this class never materializes a
// dense ternary projection.
class NativeBitNetKernel {
 public:
  NativeBitNetKernel(const std::filesystem::path& artifact,
                     std::size_t thread_count);
  ~NativeBitNetKernel();

  NativeBitNetKernel(const NativeBitNetKernel&) = delete;
  NativeBitNetKernel& operator=(const NativeBitNetKernel&) = delete;
  NativeBitNetKernel(NativeBitNetKernel&&) noexcept;
  NativeBitNetKernel& operator=(NativeBitNetKernel&&) noexcept;

  [[nodiscard]] std::size_t layer_count() const noexcept;
  [[nodiscard]] std::size_t hidden_size() const noexcept;
  [[nodiscard]] std::size_t intermediate_size() const noexcept;
  [[nodiscard]] std::size_t thread_count() const noexcept;
  [[nodiscard]] std::size_t serialized_artifact_bytes() const noexcept;

  // Input/output are row-major BF16 bit patterns with shape [rows, hidden].
  // Arithmetic follows the native BitNet BF16 MLP sequence: Q8 activation,
  // ternary gate/up, ReLU squared, intermediate RMS normalization and gain,
  // a second Q8 activation, then the ternary down projection.
  void forward_bf16(std::size_t layer, std::span<const std::uint16_t> input,
                    std::size_t rows, std::span<std::uint16_t> output,
                    NativeBitNetMetrics* metrics = nullptr);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace engram
