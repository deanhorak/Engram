#include "engram/npy.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("engram-npy-test-" + std::to_string(nonce));
    std::filesystem::create_directory(path_);
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

template <typename Value>
std::vector<std::byte> payload(const std::vector<Value>& values) {
  std::vector<std::byte> result(values.size() * sizeof(Value));
  std::memcpy(result.data(), values.data(), result.size());
  return result;
}

void write_npy(const std::filesystem::path& path, int major,
               const std::string& descriptor, const std::string& shape,
               const std::vector<std::byte>& data,
               bool fortran = false) {
  const std::size_t preamble = major == 1 ? 10 : 12;
  std::string header = "{'descr': '" + descriptor +
                       "', 'fortran_order': " +
                       (fortran ? "True" : "False") + ", 'shape': " + shape +
                       ", }";
  const std::size_t padding = (16 - ((preamble + header.size() + 1) % 16)) % 16;
  header.append(padding, ' ');
  header.push_back('\n');

  std::ofstream output(path, std::ios::binary);
  const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
  output.write(reinterpret_cast<const char*>(magic), sizeof(magic));
  const unsigned char version[] = {static_cast<unsigned char>(major), 0};
  output.write(reinterpret_cast<const char*>(version), sizeof(version));
  const std::uint32_t length = static_cast<std::uint32_t>(header.size());
  const unsigned char encoded[] = {
      static_cast<unsigned char>(length & 0xffU),
      static_cast<unsigned char>((length >> 8U) & 0xffU),
      static_cast<unsigned char>((length >> 16U) & 0xffU),
      static_cast<unsigned char>((length >> 24U) & 0xffU)};
  output.write(reinterpret_cast<const char*>(encoded), major == 1 ? 2 : 4);
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  output.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
}

bool expect_rejected(const std::filesystem::path& path,
                     engram::NpyLoadMode mode = engram::NpyLoadMode::ReadOnly) {
  try {
    static_cast<void>(engram::load_npy(path, mode));
  } catch (const engram::NpyError&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  TemporaryDirectory temporary;

  const auto float_path = temporary.path() / "float32-v1.npy";
  const std::vector<float> floats = {1.25F, -2.5F, 3.75F, 4.0F, 5.5F, -6.0F};
  write_npy(float_path, 1, "<f4", "(2, 3)", payload(floats));
  auto mapped = engram::load_npy(float_path, engram::NpyLoadMode::MemoryMap);
  if (!mapped.memory_mapped() || mapped.dtype() != engram::NpyDType::Float32 ||
      mapped.shape() != std::vector<std::size_t>({2, 3}) ||
      mapped.element_count() != floats.size() ||
      !std::equal(mapped.float32().begin(), mapped.float32().end(), floats.begin())) {
    std::cerr << "NPY v1 mmap float32 mismatch\n";
    return 1;
  }

  const auto double_path = temporary.path() / "float64-v2.npy";
  const std::vector<double> doubles = {0.125, -9.5, 22.0, 7.25};
  write_npy(double_path, 2, "<f8", "(2, 1, 2)", payload(doubles));
  auto loaded = engram::load_npy(double_path, engram::NpyLoadMode::ReadOnly);
  if (loaded.memory_mapped() || loaded.dtype() != engram::NpyDType::Float64 ||
      loaded.shape() != std::vector<std::size_t>({2, 1, 2}) ||
      !std::equal(loaded.float64().begin(), loaded.float64().end(), doubles.begin())) {
    std::cerr << "NPY v2 read-only float64 mismatch\n";
    return 1;
  }

  const auto uint32_path = temporary.path() / "uint32-v1.npy";
  const std::vector<std::uint32_t> uint32s = {0U, 7U, 65539U, 4000000000U};
  write_npy(uint32_path, 1, "<u4", "(4,)", payload(uint32s));
  auto uint32_array = engram::load_npy(uint32_path);
  if (uint32_array.dtype() != engram::NpyDType::UInt32 ||
      !std::equal(uint32_array.uint32().begin(), uint32_array.uint32().end(),
                  uint32s.begin())) {
    std::cerr << "NPY uint32 mismatch\n";
    return 1;
  }

  const auto vector_path = temporary.path() / "vector.npy";
  write_npy(vector_path, 1, "<f4", "(2,)",
            payload(std::vector<float>{6.0F, 7.0F}));
  const auto vector = engram::load_npy(vector_path);
  if (vector.shape() != std::vector<std::size_t>({2}) ||
      vector.float32()[1] != 7.0F) {
    std::cerr << "one-dimensional tuple shape mismatch\n";
    return 1;
  }
  bool wrong_typed_access_rejected = false;
  try {
    static_cast<void>(loaded.float32());
  } catch (const engram::NpyError&) {
    wrong_typed_access_rejected = true;
  }
  if (!wrong_typed_access_rejected) {
    std::cerr << "wrong typed access was not rejected\n";
    return 1;
  }

  const auto endian_path = temporary.path() / "big-endian.npy";
  write_npy(endian_path, 1, ">f4", "(1,)", payload(std::vector<float>{1.0F}));
  const auto fortran_path = temporary.path() / "fortran.npy";
  write_npy(fortran_path, 1, "<f4", "(1,)",
            payload(std::vector<float>{1.0F}), true);
  const auto truncated_path = temporary.path() / "truncated.npy";
  write_npy(truncated_path, 2, "<f8", "(2,)", payload(std::vector<double>{1.0}));
  const auto magic_path = temporary.path() / "bad-magic.npy";
  {
    std::ofstream output(magic_path, std::ios::binary);
    output << "not a numpy file";
  }
  if (!expect_rejected(endian_path) || !expect_rejected(fortran_path) ||
      !expect_rejected(truncated_path, engram::NpyLoadMode::MemoryMap) ||
      !expect_rejected(magic_path)) {
    std::cerr << "corrupt or unsupported NPY input was accepted\n";
    return 1;
  }
  return 0;
}
