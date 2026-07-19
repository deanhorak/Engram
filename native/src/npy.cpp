#include "engram/npy.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace engram {
namespace {

constexpr unsigned char kMagic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};

[[noreturn]] void fail(const std::filesystem::path& path,
                       const std::string& message) {
  throw NpyError(path.string() + ": " + message);
}

std::uint16_t little_u16(const std::byte* input) {
  return static_cast<std::uint16_t>(std::to_integer<unsigned char>(input[0])) |
         static_cast<std::uint16_t>(
             std::to_integer<unsigned char>(input[1]) << 8U);
}

std::uint32_t little_u32(const std::byte* input) {
  return static_cast<std::uint32_t>(
             std::to_integer<unsigned char>(input[0])) |
         (static_cast<std::uint32_t>(
              std::to_integer<unsigned char>(input[1]))
          << 8U) |
         (static_cast<std::uint32_t>(
              std::to_integer<unsigned char>(input[2]))
          << 16U) |
         (static_cast<std::uint32_t>(
              std::to_integer<unsigned char>(input[3]))
          << 24U);
}

std::string one_match(const std::filesystem::path& path,
                      const std::string& header, const std::regex& pattern,
                      const char* field) {
  std::sregex_iterator current(header.begin(), header.end(), pattern);
  const std::sregex_iterator end;
  if (current == end) {
    fail(path, std::string("missing or malformed ") + field);
  }
  const std::smatch match = *current;
  ++current;
  if (current != end) {
    fail(path, std::string("duplicate ") + field);
  }
  return match[1].str();
}

std::string trim(std::string_view value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return std::string(value.substr(first, last - first + 1));
}

std::vector<std::size_t> parse_shape(const std::filesystem::path& path,
                                     const std::string& body,
                                     std::size_t* element_count) {
  std::vector<std::size_t> shape;
  const std::string stripped = trim(body);
  if (stripped.empty()) {
    *element_count = 1;  // NumPy scalar, shape ().
    return shape;
  }

  std::size_t product = 1;
  std::size_t start = 0;
  bool saw_dimension = false;
  while (start <= stripped.size()) {
    const std::size_t comma = stripped.find(',', start);
    const std::size_t end = comma == std::string::npos ? stripped.size() : comma;
    const std::string token = trim(std::string_view(stripped).substr(start, end - start));
    if (token.empty()) {
      if (comma == std::string::npos && start == stripped.size() &&
          start > 0 && stripped[start - 1] == ',') {
        break;  // Legal trailing comma in a tuple.
      }
      fail(path, "malformed shape tuple");
    }
    if (!std::all_of(token.begin(), token.end(),
                     [](unsigned char character) { return character >= '0' && character <= '9'; })) {
      fail(path, "shape dimensions must be non-negative decimal integers");
    }
    std::size_t dimension = 0;
    try {
      std::size_t consumed = 0;
      dimension = std::stoull(token, &consumed, 10);
      if (consumed != token.size()) {
        fail(path, "malformed shape dimension");
      }
    } catch (const std::exception&) {
      fail(path, "shape dimension is out of range");
    }
    if (dimension != 0 && product > std::numeric_limits<std::size_t>::max() / dimension) {
      fail(path, "shape element count overflows size_t");
    }
    product *= dimension;
    shape.push_back(dimension);
    saw_dimension = true;
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  if (!saw_dimension) {
    fail(path, "malformed shape tuple");
  }
  *element_count = product;
  return shape;
}

struct ParsedArray {
  NpyDType dtype;
  std::vector<std::size_t> shape;
  std::size_t element_count;
  std::size_t data_offset;
};

ParsedArray parse_array(const std::filesystem::path& path,
                        const std::byte* bytes,
                        const std::size_t file_size) {
  if (file_size < 10) {
    fail(path, "file is too short for an NPY preamble");
  }
  if (std::memcmp(bytes, kMagic, sizeof(kMagic)) != 0) {
    fail(path, "invalid NPY magic");
  }
  const auto major = std::to_integer<unsigned char>(bytes[6]);
  const auto minor = std::to_integer<unsigned char>(bytes[7]);
  if ((major != 1 && major != 2) || minor != 0) {
    fail(path, "unsupported NPY version (expected 1.0 or 2.0)");
  }
  const std::size_t preamble_size = major == 1 ? 10 : 12;
  if (file_size < preamble_size) {
    fail(path, "truncated NPY preamble");
  }
  const std::size_t header_size =
      major == 1 ? little_u16(bytes + 8) : little_u32(bytes + 8);
  if (header_size == 0 || header_size > file_size - preamble_size) {
    fail(path, "invalid or truncated NPY header length");
  }
  const std::size_t data_offset = preamble_size + header_size;
  const char* header_begin = reinterpret_cast<const char*>(bytes + preamble_size);
  std::string header(header_begin, header_size);
  if (header.back() != '\n' || header.find('\0') != std::string::npos) {
    fail(path, "NPY header must be newline-terminated ASCII text");
  }
  const std::string dictionary = trim(header);
  if (dictionary.size() < 2 || dictionary.front() != '{' ||
      dictionary.back() != '}') {
    fail(path, "NPY header must contain a Python dictionary literal");
  }

  static const std::regex descr_pattern(
      R"re(['"]descr['"]\s*:\s*['"]([^'"]+)['"])re");
  static const std::regex fortran_pattern(
      R"re(['"]fortran_order['"]\s*:\s*(True|False))re");
  static const std::regex shape_pattern(
      R"re(['"]shape['"]\s*:\s*\(([^)]*)\))re");
  const std::string descr = one_match(path, header, descr_pattern, "descr");
  const std::string fortran =
      one_match(path, header, fortran_pattern, "fortran_order");
  const std::string shape_body = one_match(path, header, shape_pattern, "shape");
  if (fortran != "False") {
    fail(path, "Fortran-order arrays are not supported");
  }
  NpyDType dtype;
  if (descr == "|u1" || descr == "<u1") {
    dtype = NpyDType::UInt8;
  } else if (descr == "<u4") {
    dtype = NpyDType::UInt32;
  } else if (descr == "<f4") {
    dtype = NpyDType::Float32;
  } else if (descr == "<f8") {
    dtype = NpyDType::Float64;
  } else {
    fail(path, "only uint8 and little-endian <u4, <f4, and <f8 arrays are supported");
  }
  std::size_t element_count = 0;
  std::vector<std::size_t> shape =
      parse_shape(path, shape_body, &element_count);
  const std::size_t item_size = dtype == NpyDType::UInt8
                                    ? 1
                                    : (dtype == NpyDType::Float64 ? 8 : 4);
  if (element_count >
      std::numeric_limits<std::size_t>::max() / item_size) {
    fail(path, "array byte size overflows size_t");
  }
  const std::size_t payload_size = element_count * item_size;
  if (payload_size != file_size - data_offset) {
    fail(path, payload_size > file_size - data_offset ? "truncated NPY payload"
                                                      : "unexpected bytes after NPY payload");
  }
  if (reinterpret_cast<std::uintptr_t>(bytes + data_offset) % item_size != 0) {
    fail(path, "NPY payload is not naturally aligned");
  }
  return {dtype, std::move(shape), element_count, data_offset};
}

}  // namespace

NpyArray::NpyArray(NpyArray&& other) noexcept { *this = std::move(other); }

NpyArray& NpyArray::operator=(NpyArray&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  dtype_ = other.dtype_;
  shape_ = std::move(other.shape_);
  element_count_ = other.element_count_;
  data_offset_ = other.data_offset_;
  mapping_ = other.mapping_;
  mapping_size_ = other.mapping_size_;
  storage_ = std::move(other.storage_);
  other.mapping_ = nullptr;
  other.mapping_size_ = 0;
  other.element_count_ = 0;
  other.data_offset_ = 0;
  return *this;
}

NpyArray::~NpyArray() { release(); }

void NpyArray::release() noexcept {
  if (mapping_ != nullptr) {
    ::munmap(mapping_, mapping_size_);
    mapping_ = nullptr;
    mapping_size_ = 0;
  }
}

std::size_t NpyArray::item_size() const noexcept {
  if (dtype_ == NpyDType::UInt8) return sizeof(std::uint8_t);
  if (dtype_ == NpyDType::UInt32) return sizeof(std::uint32_t);
  return dtype_ == NpyDType::Float32 ? sizeof(float) : sizeof(double);
}

std::size_t NpyArray::byte_size() const noexcept {
  return element_count_ * item_size();
}

const std::byte* NpyArray::bytes() const noexcept {
  const auto* base = mapping_ != nullptr
                         ? static_cast<const std::byte*>(mapping_)
                         : storage_.data();
  return base + data_offset_;
}

std::span<const float> NpyArray::float32() const {
  if (dtype_ != NpyDType::Float32) {
    throw NpyError("NPY dtype is not float32");
  }
  return {reinterpret_cast<const float*>(bytes()), element_count_};
}

std::span<const double> NpyArray::float64() const {
  if (dtype_ != NpyDType::Float64) {
    throw NpyError("NPY dtype is not float64");
  }
  return {reinterpret_cast<const double*>(bytes()), element_count_};
}

std::span<const std::uint8_t> NpyArray::uint8() const {
  if (dtype_ != NpyDType::UInt8) {
    throw NpyError("NPY dtype is not uint8");
  }
  return {reinterpret_cast<const std::uint8_t*>(bytes()), element_count_};
}

std::span<const std::uint32_t> NpyArray::uint32() const {
  if (dtype_ != NpyDType::UInt32) {
    throw NpyError("NPY dtype is not uint32");
  }
  return {reinterpret_cast<const std::uint32_t*>(bytes()), element_count_};
}

NpyArray load_npy(const std::filesystem::path& path, const NpyLoadMode mode) {
  NpyArray result;
  if (mode == NpyLoadMode::MemoryMap) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
      fail(path, std::string("cannot open file: ") + std::strerror(errno));
    }
    struct stat status {};
    if (::fstat(descriptor, &status) != 0 || status.st_size <= 0) {
      const std::string message = status.st_size <= 0 ? "file is empty"
                                                       : std::strerror(errno);
      ::close(descriptor);
      fail(path, std::string("cannot stat file: ") + message);
    }
    if (static_cast<std::uintmax_t>(status.st_size) >
        std::numeric_limits<std::size_t>::max()) {
      ::close(descriptor);
      fail(path, "file is too large for this process");
    }
    result.mapping_size_ = static_cast<std::size_t>(status.st_size);
    result.mapping_ = ::mmap(nullptr, result.mapping_size_, PROT_READ,
                             MAP_PRIVATE, descriptor, 0);
    const int saved_errno = errno;
    ::close(descriptor);
    if (result.mapping_ == MAP_FAILED) {
      result.mapping_ = nullptr;
      result.mapping_size_ = 0;
      fail(path, std::string("mmap failed: ") + std::strerror(saved_errno));
    }
    const ParsedArray parsed =
        parse_array(path, static_cast<const std::byte*>(result.mapping_),
                    result.mapping_size_);
    result.dtype_ = parsed.dtype;
    result.shape_ = parsed.shape;
    result.element_count_ = parsed.element_count;
    result.data_offset_ = parsed.data_offset;
  } else {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
      fail(path, "cannot open file");
    }
    const std::streamoff length = input.tellg();
    if (length <= 0 || static_cast<std::uintmax_t>(length) >
                           std::numeric_limits<std::size_t>::max()) {
      fail(path, "invalid file size");
    }
    result.storage_.resize(static_cast<std::size_t>(length));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(result.storage_.data()), length);
    if (!input) {
      fail(path, "failed to read complete file");
    }
    const ParsedArray parsed =
        parse_array(path, result.storage_.data(), result.storage_.size());
    result.dtype_ = parsed.dtype;
    result.shape_ = parsed.shape;
    result.element_count_ = parsed.element_count;
    result.data_offset_ = parsed.data_offset;
  }
  return result;
}

}  // namespace engram
