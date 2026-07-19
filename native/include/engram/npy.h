#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <vector>

namespace engram {

enum class NpyDType { UInt8, UInt32, Float32, Float64 };
enum class NpyLoadMode { MemoryMap, ReadOnly };

class NpyError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class NpyArray {
 public:
  NpyArray(const NpyArray&) = delete;
  NpyArray& operator=(const NpyArray&) = delete;
  NpyArray(NpyArray&& other) noexcept;
  NpyArray& operator=(NpyArray&& other) noexcept;
  ~NpyArray();

  [[nodiscard]] NpyDType dtype() const noexcept { return dtype_; }
  [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept {
    return shape_;
  }
  [[nodiscard]] std::size_t element_count() const noexcept {
    return element_count_;
  }
  [[nodiscard]] std::size_t item_size() const noexcept;
  [[nodiscard]] std::size_t byte_size() const noexcept;
  [[nodiscard]] bool memory_mapped() const noexcept { return mapping_ != nullptr; }
  [[nodiscard]] const std::byte* bytes() const noexcept;

  [[nodiscard]] std::span<const float> float32() const;
  [[nodiscard]] std::span<const double> float64() const;
  [[nodiscard]] std::span<const std::uint8_t> uint8() const;
  [[nodiscard]] std::span<const std::uint32_t> uint32() const;

 private:
  NpyArray() = default;
  void release() noexcept;

  NpyDType dtype_ = NpyDType::Float32;
  std::vector<std::size_t> shape_;
  std::size_t element_count_ = 0;
  std::size_t data_offset_ = 0;
  void* mapping_ = nullptr;
  std::size_t mapping_size_ = 0;
  std::vector<std::byte> storage_;

  friend NpyArray load_npy(const std::filesystem::path&, NpyLoadMode);
};

// Loads NumPy format 1.x/2.x arrays. Only C-contiguous uint8 and little-endian
// uint32, float32, and float64 payloads are accepted. Returned storage is
// exposed read-only.
[[nodiscard]] NpyArray load_npy(
    const std::filesystem::path& path,
    NpyLoadMode mode = NpyLoadMode::MemoryMap);

}  // namespace engram
