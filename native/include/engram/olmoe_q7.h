#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace engram {

struct OLMoEQ7Metrics {
  std::uint64_t elapsed_ns{};
  std::uint64_t router_stream_bytes{};
  std::uint64_t selected_expert_stream_bytes{};
  std::uint64_t scheduled_stream_bytes{};
  std::uint64_t scratch_bytes{};
  std::uint64_t rows{};
  std::uint64_t threads{};
  std::uint64_t selected_experts{};
};

// CPU-only mmap reader and sparse expert kernel for
// olmoe_native_groupwise_q7_v1. Packed experts are decoded coefficient by
// coefficient during dot products; dense expert matrices are never built.
class OLMoEQ7Kernel {
 public:
  OLMoEQ7Kernel(const std::filesystem::path& artifact,
                std::size_t thread_count);
  ~OLMoEQ7Kernel();

  OLMoEQ7Kernel(const OLMoEQ7Kernel&) = delete;
  OLMoEQ7Kernel& operator=(const OLMoEQ7Kernel&) = delete;
  OLMoEQ7Kernel(OLMoEQ7Kernel&&) noexcept;
  OLMoEQ7Kernel& operator=(OLMoEQ7Kernel&&) noexcept;

  [[nodiscard]] std::size_t layer_count() const noexcept;
  [[nodiscard]] std::size_t hidden_size() const noexcept;
  [[nodiscard]] std::size_t intermediate_size() const noexcept;
  [[nodiscard]] std::size_t expert_count() const noexcept;
  [[nodiscard]] std::size_t top_k() const noexcept;
  [[nodiscard]] std::size_t group_size() const noexcept;
  [[nodiscard]] std::size_t thread_count() const noexcept;
  [[nodiscard]] std::size_t serialized_artifact_bytes() const noexcept;

  // Input/output are row-major float32 [rows, hidden]. selected_experts, when
  // supplied, is row-major uint32 [rows, top_k] in descending router order.
  void forward(std::size_t layer, std::span<const float> input,
               std::size_t rows, std::span<float> output,
               std::span<std::uint32_t> selected_experts = {},
               OLMoEQ7Metrics* metrics = nullptr);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace engram
