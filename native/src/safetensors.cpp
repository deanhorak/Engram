#include "engram/safetensors.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <regex>
#include <string_view>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace engram {
namespace {

[[noreturn]] void fail(const std::filesystem::path& path,
                       const std::string& message) {
  throw SafetensorError(path.string() + ": " + message);
}

std::uint64_t little_u64(const std::byte* input) {
  std::uint64_t value = 0;
  for (std::size_t byte = 0; byte < 8; ++byte) {
    value |= static_cast<std::uint64_t>(
                 std::to_integer<unsigned char>(input[byte]))
             << (8U * byte);
  }
  return value;
}

std::size_t item_size(const SafetensorDType dtype) {
  if (dtype == SafetensorDType::UInt8) return 1;
  return dtype == SafetensorDType::BF16 ? 2 : 4;
}

std::vector<std::size_t> parse_shape(const std::filesystem::path& path,
                                     const std::string& text,
                                     std::size_t* elements) {
  std::vector<std::size_t> shape;
  std::size_t product = 1;
  if (text.empty()) {
    *elements = 1;
    return shape;
  }
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const std::size_t comma = text.find(',', begin);
    const std::string token =
        text.substr(begin, comma == std::string::npos ? std::string::npos
                                                      : comma - begin);
    if (token.empty() ||
        !std::all_of(token.begin(), token.end(), [](const char value) {
          return value >= '0' && value <= '9';
        })) {
      fail(path, "invalid tensor shape");
    }
    std::size_t dimension = 0;
    try {
      dimension = std::stoull(token);
    } catch (const std::exception&) {
      fail(path, "tensor shape dimension is out of range");
    }
    if (dimension != 0 &&
        product > std::numeric_limits<std::size_t>::max() / dimension) {
      fail(path, "tensor element count overflows");
    }
    product *= dimension;
    shape.push_back(dimension);
    if (comma == std::string::npos) break;
    begin = comma + 1;
  }
  *elements = product;
  return shape;
}

}  // namespace

std::size_t SafetensorView::element_count() const noexcept {
  std::size_t result = 1;
  for (const std::size_t dimension : shape) result *= dimension;
  return result;
}

std::span<const std::uint16_t> SafetensorView::bf16() const {
  if (dtype != SafetensorDType::BF16) {
    throw SafetensorError("requested BF16 view of another dtype");
  }
  return {reinterpret_cast<const std::uint16_t*>(bytes.data()),
          element_count()};
}

std::span<const std::uint8_t> SafetensorView::uint8() const {
  if (dtype != SafetensorDType::UInt8) {
    throw SafetensorError("requested uint8 view of another dtype");
  }
  return {reinterpret_cast<const std::uint8_t*>(bytes.data()),
          element_count()};
}

std::span<const float> SafetensorView::float32() const {
  if (dtype != SafetensorDType::Float32) {
    throw SafetensorError("requested float32 view of another dtype");
  }
  return {reinterpret_cast<const float*>(bytes.data()), element_count()};
}

SafetensorFile::SafetensorFile(SafetensorFile&& other) noexcept {
  *this = std::move(other);
}

SafetensorFile& SafetensorFile::operator=(SafetensorFile&& other) noexcept {
  if (this == &other) return *this;
  release();
  mapping_ = other.mapping_;
  mapping_size_ = other.mapping_size_;
  data_offset_ = other.data_offset_;
  tensors_ = std::move(other.tensors_);
  other.mapping_ = nullptr;
  other.mapping_size_ = 0;
  other.data_offset_ = 0;
  return *this;
}

SafetensorFile::~SafetensorFile() { release(); }

void SafetensorFile::release() noexcept {
  if (mapping_ != nullptr) {
    ::munmap(mapping_, mapping_size_);
    mapping_ = nullptr;
  }
  mapping_size_ = 0;
  data_offset_ = 0;
}

bool SafetensorFile::contains(const std::string& name) const {
  return tensors_.contains(name);
}

SafetensorView SafetensorFile::tensor(const std::string& name) const {
  const auto found = tensors_.find(name);
  if (found == tensors_.end()) {
    throw SafetensorError("safetensors file has no tensor named " + name);
  }
  const Descriptor& descriptor = found->second;
  const auto* bytes = static_cast<const std::byte*>(mapping_);
  return {
      descriptor.dtype,
      descriptor.shape,
      std::span<const std::byte>(
          bytes + data_offset_ + descriptor.begin,
          descriptor.end - descriptor.begin),
  };
}

SafetensorFile load_safetensors(const std::filesystem::path& path) {
  const int descriptor = ::open(path.c_str(), O_RDONLY);
  if (descriptor < 0) fail(path, "cannot open file");
  struct stat status {};
  if (::fstat(descriptor, &status) != 0 || status.st_size < 10) {
    ::close(descriptor);
    fail(path, "file is too short");
  }
  const std::size_t size = static_cast<std::size_t>(status.st_size);
  void* mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, descriptor, 0);
  ::close(descriptor);
  if (mapping == MAP_FAILED) fail(path, "memory mapping failed");

  SafetensorFile result;
  result.mapping_ = mapping;
  result.mapping_size_ = size;
  const auto* bytes = static_cast<const std::byte*>(mapping);
  const std::uint64_t header_size_u64 = little_u64(bytes);
  if (header_size_u64 == 0 ||
      header_size_u64 > static_cast<std::uint64_t>(size - 8)) {
    fail(path, "invalid header length");
  }
  const std::size_t header_size = static_cast<std::size_t>(header_size_u64);
  result.data_offset_ = 8 + header_size;
  std::string header(reinterpret_cast<const char*>(bytes + 8), header_size);
  while (!header.empty() && header.back() == ' ') header.pop_back();
  if (header.size() < 2 || header.front() != '{' || header.back() != '}') {
    fail(path, "header is not a JSON object");
  }

  static const std::regex tensor_pattern(
      R"re("([^"\\]+)":\{"dtype":"(BF16|U8|F32)","shape":\[([0-9,]*)\],"data_offsets":\[([0-9]+),([0-9]+)\]\})re");
  std::size_t cursor = 1;
  std::size_t payload_end = 0;
  for (std::sregex_iterator current(header.begin(), header.end(),
                                    tensor_pattern),
       end;
       current != end; ++current) {
    const std::smatch& match = *current;
    const std::size_t match_begin = static_cast<std::size_t>(match.position());
    const std::string separator =
        header.substr(cursor, match_begin - cursor);
    if (!separator.empty() && separator != ",") {
      fail(path, "unsupported or malformed header entry");
    }
    const std::string dtype_text = match[2].str();
    const SafetensorDType dtype =
        dtype_text == "BF16"
            ? SafetensorDType::BF16
            : (dtype_text == "U8" ? SafetensorDType::UInt8
                                   : SafetensorDType::Float32);
    std::size_t elements = 0;
    std::vector<std::size_t> shape =
        parse_shape(path, match[3].str(), &elements);
    const std::size_t begin = std::stoull(match[4].str());
    const std::size_t finish = std::stoull(match[5].str());
    const std::size_t bytes_expected = elements * item_size(dtype);
    if (finish < begin || finish - begin != bytes_expected ||
        begin != payload_end || finish > size - result.data_offset_) {
      fail(path, "tensor offsets are invalid or non-contiguous");
    }
    if (!result.tensors_
             .emplace(match[1].str(),
                      SafetensorFile::Descriptor{
                          dtype, std::move(shape), begin, finish})
             .second) {
      fail(path, "duplicate tensor name");
    }
    payload_end = finish;
    cursor = match_begin + static_cast<std::size_t>(match.length());
  }
  if (result.tensors_.empty() || header.substr(cursor) != "}" ||
      payload_end != size - result.data_offset_) {
    fail(path, "header or payload is incomplete");
  }
  return result;
}

}  // namespace engram
