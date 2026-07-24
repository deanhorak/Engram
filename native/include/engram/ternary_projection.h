#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace engram {

struct TernaryProjectionMetrics {
  std::uint64_t elapsed_ns{};
  std::uint64_t packed_weight_bytes{};
  std::uint64_t scratch_bytes{};
  std::uint64_t rows{};
};

// Threaded Q8-activation/ternary-weight projection using the official
// four-output-codes-per-byte BitNet layout.
class TernaryProjectionKernel {
 public:
  explicit TernaryProjectionKernel(std::size_t threads);
  ~TernaryProjectionKernel();
  TernaryProjectionKernel(const TernaryProjectionKernel&) = delete;
  TernaryProjectionKernel& operator=(const TernaryProjectionKernel&) = delete;

  std::size_t add(std::span<const std::uint8_t> packed,
                  std::size_t input_features, std::size_t output_features,
                  float weight_scale);
  [[nodiscard]] std::size_t input_features(std::size_t projection) const;
  [[nodiscard]] std::size_t output_features(std::size_t projection) const;
  void forward_bf16(std::size_t projection,
                    std::span<const std::uint16_t> input, std::size_t rows,
                    std::span<std::uint16_t> output,
                    TernaryProjectionMetrics* metrics = nullptr);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace engram
