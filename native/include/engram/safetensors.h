#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace engram {

enum class SafetensorDType { BF16, UInt8, Float32 };

class SafetensorError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct SafetensorView {
  SafetensorDType dtype;
  std::vector<std::size_t> shape;
  std::span<const std::byte> bytes;

  [[nodiscard]] std::size_t element_count() const noexcept;
  [[nodiscard]] std::span<const std::uint16_t> bf16() const;
  [[nodiscard]] std::span<const std::uint8_t> uint8() const;
  [[nodiscard]] std::span<const float> float32() const;
};

class SafetensorFile {
 public:
  SafetensorFile(const SafetensorFile&) = delete;
  SafetensorFile& operator=(const SafetensorFile&) = delete;
  SafetensorFile(SafetensorFile&& other) noexcept;
  SafetensorFile& operator=(SafetensorFile&& other) noexcept;
  ~SafetensorFile();

  [[nodiscard]] bool contains(const std::string& name) const;
  [[nodiscard]] SafetensorView tensor(const std::string& name) const;
  [[nodiscard]] std::size_t tensor_count() const noexcept {
    return tensors_.size();
  }
  [[nodiscard]] std::size_t mapped_bytes() const noexcept {
    return mapping_size_;
  }

 private:
  struct Descriptor {
    SafetensorDType dtype;
    std::vector<std::size_t> shape;
    std::size_t begin;
    std::size_t end;
  };

  SafetensorFile() = default;
  void release() noexcept;

  void* mapping_ = nullptr;
  std::size_t mapping_size_ = 0;
  std::size_t data_offset_ = 0;
  std::map<std::string, Descriptor> tensors_;

  friend SafetensorFile load_safetensors(const std::filesystem::path&);
};

[[nodiscard]] SafetensorFile load_safetensors(
    const std::filesystem::path& path);

}  // namespace engram
