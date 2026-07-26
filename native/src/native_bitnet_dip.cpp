#include "engram/native_bitnet_dip.h"

#include "engram/thread_pool.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace engram {
namespace {

constexpr std::array<char, 8> kBaseMagic = {'E', 'N', 'G', 'B',
                                             'N', 'P', '1', '1'};
constexpr std::array<char, 8> kBaseLayerMagic = {'E', 'N', 'G', 'B',
                                                  'N', 'P', 'L', '1'};
constexpr std::array<char, 8> kIndexMagic = {'E', 'N', 'G', 'B',
                                              'D', 'I', '1', '2'};
constexpr std::array<char, 8> kIndexLayerMagic = {'E', 'N', 'G', 'B',
                                                   'D', 'I', 'L', '2'};
constexpr std::uint32_t kVersion = 2;
constexpr std::uint32_t kBaseVersion = 1;
constexpr std::uint32_t kEndianMarker = 0x01020304U;
constexpr std::size_t kBaseHeaderBytes = 64;
constexpr std::size_t kIndexHeaderBytes = 128;
constexpr std::size_t kDirectoryEntryBytes = 32;
constexpr std::size_t kBaseLayerHeaderBytes = 64;
constexpr std::size_t kIndexLayerHeaderBytes = 128;
constexpr std::size_t kIndexLayerHeaderCoreBytes = 76;
constexpr std::size_t kProjectionScaleBytes = 6;
constexpr std::size_t kCacheLineBytes = 64;
constexpr std::size_t kTritsPerByte = 5;
constexpr std::size_t kMaximumDimension = 1U << 20U;
constexpr std::size_t kMaximumLayers = 1U << 12U;
constexpr std::uint32_t kCoordinateEncoding = 1;
constexpr std::uint32_t kNormUint16 = 1;
constexpr std::uint32_t kNormUint32 = 2;
constexpr std::uint32_t kChecksumSha256 = 1;

std::uint16_t little_u16(const std::byte* input) noexcept {
  return static_cast<std::uint16_t>(std::to_integer<unsigned char>(input[0])) |
         static_cast<std::uint16_t>(
             std::to_integer<unsigned char>(input[1]) << 8U);
}

std::uint32_t little_u32(const std::byte* input) noexcept {
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

std::uint64_t little_u64(const std::byte* input) noexcept {
  return static_cast<std::uint64_t>(little_u32(input)) |
         (static_cast<std::uint64_t>(little_u32(input + 4)) << 32U);
}

float little_f32(const std::byte* input) noexcept {
  return std::bit_cast<float>(little_u32(input));
}

void store_little_u32(std::uint8_t* output,
                      const std::uint32_t value) noexcept {
  output[0] = static_cast<std::uint8_t>(value);
  output[1] = static_cast<std::uint8_t>(value >> 8U);
  output[2] = static_cast<std::uint8_t>(value >> 16U);
  output[3] = static_cast<std::uint8_t>(value >> 24U);
}

std::size_t align_up(const std::size_t value, const std::size_t alignment) {
  if (alignment == 0 ||
      value > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
    throw std::invalid_argument("native BitNet DIP alignment overflow");
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

std::size_t checked_product(const std::size_t left, const std::size_t right,
                            const char* message) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::invalid_argument(message);
  }
  return left * right;
}

void require_zero(const std::byte* bytes, const std::size_t begin,
                  const std::size_t end, const char* message) {
  for (std::size_t index = begin; index < end; ++index) {
    if (bytes[index] != std::byte{0}) {
      throw std::invalid_argument(message);
    }
  }
}

float bf16_to_float(const std::uint16_t bits) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

std::uint16_t float_to_bf16_bits(const float value) noexcept {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t rounded =
      bits + 0x7FFFU + ((bits >> 16U) & static_cast<std::uint32_t>(1));
  return static_cast<std::uint16_t>(rounded >> 16U);
}

float bf16_round(const float value) noexcept {
  return bf16_to_float(float_to_bf16_bits(value));
}

float round_ties_to_even(const float value) noexcept {
  const float lower = std::floor(value);
  const float fraction = value - lower;
  if (fraction < 0.5F) return lower;
  if (fraction > 0.5F) return lower + 1.0F;
  return std::fmod(std::abs(lower), 2.0F) == 0.0F ? lower
                                                  : lower + 1.0F;
}

float quantized_bf16(const float value, const float scale) noexcept {
  const float rounded = round_ties_to_even(value * scale);
  return bf16_round(std::clamp(rounded, -128.0F, 127.0F) / scale);
}

constexpr auto make_trit_table() {
  std::array<std::array<std::int8_t, kTritsPerByte>, 243> result{};
  for (std::size_t packed = 0; packed < result.size(); ++packed) {
    std::size_t value = packed;
    for (std::size_t digit = 0; digit < kTritsPerByte; ++digit) {
      result[packed][digit] =
          static_cast<std::int8_t>(value % 3U) - 1;
      value /= 3U;
    }
  }
  return result;
}

constexpr auto kTritTable = make_trit_table();

void validate_packed_rows(const std::uint8_t* stream,
                          const std::size_t records,
                          const std::size_t packed_width,
                          const std::size_t logical_width) {
  const std::size_t used_tail = logical_width % kTritsPerByte;
  const std::uint8_t tail_limit =
      used_tail == 0
          ? 243U
          : static_cast<std::uint8_t>(
                std::array<std::uint16_t, 5>{1U, 3U, 9U, 27U, 81U}[used_tail]);
  for (std::size_t record = 0; record < records; ++record) {
    const std::uint8_t* row = stream + record * packed_width;
    for (std::size_t packed = 0; packed < packed_width; ++packed) {
      if (row[packed] > 242U) {
        throw std::invalid_argument(
            "native BitNet DIP base-3 byte is not canonical");
      }
    }
    if (used_tail != 0 && row[packed_width - 1] >= tail_limit) {
      throw std::invalid_argument(
          "native BitNet DIP base-3 tail is not canonical");
    }
  }
}

class MappedFile {
 public:
  explicit MappedFile(const std::filesystem::path& path) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
      throw std::invalid_argument("cannot open native BitNet DIP artifact");
    }
    struct stat status {};
    if (::fstat(descriptor, &status) != 0 || status.st_size <= 0 ||
        static_cast<std::uintmax_t>(status.st_size) >
            std::numeric_limits<std::size_t>::max()) {
      ::close(descriptor);
      throw std::invalid_argument("cannot stat native BitNet DIP artifact");
    }
    size_ = static_cast<std::size_t>(status.st_size);
    mapping_ =
        ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor, 0);
    const int saved_errno = errno;
    ::close(descriptor);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      size_ = 0;
      throw std::invalid_argument("cannot mmap native BitNet DIP artifact: " +
                                  std::to_string(saved_errno));
    }
  }

  ~MappedFile() {
    if (mapping_ != nullptr) {
      ::munmap(mapping_, size_);
    }
  }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  [[nodiscard]] const std::byte* bytes() const noexcept {
    return static_cast<const std::byte*>(mapping_);
  }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  void* mapping_{};
  std::size_t size_{};
};

constexpr std::array<std::uint32_t, 64> kShaConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

std::uint32_t rotate_right(const std::uint32_t value,
                           const unsigned count) noexcept {
  return (value >> count) | (value << (32U - count));
}

class Sha256 {
 public:
  void update(const std::uint8_t* data, std::size_t size) {
    total_bytes_ += size;
    while (size != 0) {
      const std::size_t count = std::min(size, block_.size() - buffered_);
      std::copy_n(data, count, block_.begin() + buffered_);
      buffered_ += count;
      data += count;
      size -= count;
      if (buffered_ == block_.size()) {
        transform(block_.data());
        buffered_ = 0;
      }
    }
  }

  std::array<std::uint8_t, 32> finish() {
    const std::uint64_t bit_count = total_bytes_ * 8U;
    block_[buffered_++] = 0x80U;
    if (buffered_ > 56) {
      std::fill(block_.begin() + buffered_, block_.end(), 0U);
      transform(block_.data());
      buffered_ = 0;
    }
    std::fill(block_.begin() + buffered_, block_.begin() + 56, 0U);
    for (std::size_t index = 0; index < 8; ++index) {
      block_[63 - index] =
          static_cast<std::uint8_t>(bit_count >> (index * 8U));
    }
    transform(block_.data());
    std::array<std::uint8_t, 32> result{};
    for (std::size_t word = 0; word < state_.size(); ++word) {
      for (std::size_t byte = 0; byte < 4; ++byte) {
        result[word * 4 + byte] = static_cast<std::uint8_t>(
            state_[word] >> ((3U - byte) * 8U));
      }
    }
    return result;
  }

 private:
  void transform(const std::uint8_t* block) {
    std::array<std::uint32_t, 64> schedule{};
    for (std::size_t index = 0; index < 16; ++index) {
      schedule[index] =
          (static_cast<std::uint32_t>(block[index * 4]) << 24U) |
          (static_cast<std::uint32_t>(block[index * 4 + 1]) << 16U) |
          (static_cast<std::uint32_t>(block[index * 4 + 2]) << 8U) |
          static_cast<std::uint32_t>(block[index * 4 + 3]);
    }
    for (std::size_t index = 16; index < schedule.size(); ++index) {
      const std::uint32_t s0 =
          rotate_right(schedule[index - 15], 7) ^
          rotate_right(schedule[index - 15], 18) ^
          (schedule[index - 15] >> 3U);
      const std::uint32_t s1 =
          rotate_right(schedule[index - 2], 17) ^
          rotate_right(schedule[index - 2], 19) ^
          (schedule[index - 2] >> 10U);
      schedule[index] = schedule[index - 16] + s0 +
                        schedule[index - 7] + s1;
    }
    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t index = 0; index < schedule.size(); ++index) {
      const std::uint32_t sigma1 =
          rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
      const std::uint32_t choice = (e & f) ^ (~e & g);
      const std::uint32_t temporary1 =
          h + sigma1 + choice + kShaConstants[index] + schedule[index];
      const std::uint32_t sigma0 =
          rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = sigma0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_ = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<std::uint8_t, 64> block_{};
  std::size_t buffered_{};
  std::uint64_t total_bytes_{};
};

std::array<std::uint8_t, 32> sha256(const std::byte* bytes,
                                    const std::size_t size) {
  Sha256 hash;
  hash.update(reinterpret_cast<const std::uint8_t*>(bytes), size);
  return hash.finish();
}

struct BaseLayerView {
  const std::uint8_t* gate{};
  const std::uint8_t* up{};
  const std::byte* gain{};
  const std::uint8_t* down{};
  float gate_scale{};
  float up_scale{};
  float down_scale{};
  std::size_t block_bytes{};
};

struct IndexLayerView {
  const std::uint8_t* gate_coordinates{};
  const std::uint8_t* up_coordinates{};
  const std::byte* down_norm{};
  NativeBitNetDIPPolicy policy;
  std::size_t block_bytes{};
};

}  // namespace

class NativeBitNetDIPKernel::Impl {
 public:
  Impl(const std::filesystem::path& record_artifact,
       const std::filesystem::path& coordinate_index,
       const std::size_t thread_count)
      : base_mapping_(record_artifact),
        index_mapping_(coordinate_index),
        pool_(thread_count) {
    if (thread_count == 0 || thread_count > 256) {
      throw std::invalid_argument(
          "native BitNet DIP thread count is invalid");
    }
    parse_base();
    parse_index();
  }

  void forward(const std::size_t layer_index,
               const std::span<const std::uint16_t> input,
               const std::size_t rows,
               const std::span<std::uint16_t> output,
               const std::span<std::uint32_t> selected_counts,
               NativeBitNetDIPMetrics* const metrics,
               const std::span<std::uint32_t> input_coordinate_ids = {},
               const std::span<std::uint32_t> candidate_ids = {},
               const std::span<std::uint32_t> selected_record_ids = {}) {
    if (layer_index >= base_layers_.size() || rows == 0) {
      throw std::invalid_argument(
          "native BitNet DIP forward dimensions are invalid");
    }
    const std::size_t elements = checked_product(
        rows, hidden_size_, "native BitNet DIP dimensions overflow");
    const NativeBitNetDIPPolicy& debug_policy =
        index_layers_[layer_index].policy;
    if (input.size() != elements || output.size() != elements ||
        (!selected_counts.empty() && selected_counts.size() != rows) ||
        (!input_coordinate_ids.empty() &&
         input_coordinate_ids.size() !=
             rows * debug_policy.input_coordinates) ||
        (!candidate_ids.empty() &&
         candidate_ids.size() != rows * debug_policy.candidate_count) ||
        (!selected_record_ids.empty() &&
         selected_record_ids.size() !=
             rows * debug_policy.maximum_top_k)) {
      throw std::invalid_argument(
          "native BitNet DIP input/output/diagnostic size mismatch");
    }
    ensure_scratch(rows);
    const auto started = std::chrono::steady_clock::now();
    const BaseLayerView& base = base_layers_[layer_index];
    const IndexLayerView& index = index_layers_[layer_index];
    std::fill_n(row_selected_counts_.begin(), rows, std::uint32_t{0});

    // Rows are independent and share immutable mappings.  This keeps the token
    // path scalar and deterministic while using the persistent pool for real
    // batches without allocating worker-local task objects.
    pool_.parallel_for(0, rows, 1, [&](const std::size_t row) {
      forward_row(base, index, input.data() + row * hidden_size_,
                  output.data() + row * hidden_size_, row);
    });

    const auto finished = std::chrono::steady_clock::now();
    if (metrics != nullptr) {
      populate_metrics(index, rows, started, finished, metrics);
    }
    // These copies are diagnostic and deliberately outside the timed region.
    if (!selected_counts.empty()) {
      std::copy_n(row_selected_counts_.begin(), rows,
                  selected_counts.begin());
    }
    for (std::size_t row = 0; row < rows; ++row) {
      if (!input_coordinate_ids.empty()) {
        const std::size_t* coordinates =
            coordinate_order_.data() + row * hidden_size_;
        std::transform(
            coordinates,
            coordinates + debug_policy.input_coordinates,
            input_coordinate_ids.begin() +
                row * debug_policy.input_coordinates,
            [](const std::size_t value) {
              return static_cast<std::uint32_t>(value);
            });
      }
      const std::size_t* candidates =
          candidate_indices_.data() + row * intermediate_size_;
      if (!candidate_ids.empty()) {
        std::transform(
            candidates,
            candidates + debug_policy.candidate_count,
            candidate_ids.begin() + row * debug_policy.candidate_count,
            [](const std::size_t value) {
              return static_cast<std::uint32_t>(value);
            });
      }
      if (!selected_record_ids.empty()) {
        std::uint32_t* destination =
            selected_record_ids.data() +
            row * debug_policy.maximum_top_k;
        std::fill_n(destination, debug_policy.maximum_top_k,
                    std::numeric_limits<std::uint32_t>::max());
        const std::size_t* exact_order =
            record_order_.data() + row * intermediate_size_;
        for (std::size_t selected = 0;
             selected < row_selected_counts_[row]; ++selected) {
          destination[selected] =
              static_cast<std::uint32_t>(
                  candidates[exact_order[selected]]);
        }
      }
    }
  }

  void teacher_top_k(
      const std::size_t layer_index,
      const std::span<const std::uint16_t> input,
      const std::size_t rows, const std::size_t top_k,
      const std::span<std::uint32_t> teacher_record_ids,
      const std::span<std::uint32_t> positive_utility_counts = {}) {
    if (layer_index >= base_layers_.size() || rows == 0 || top_k == 0 ||
        top_k > intermediate_size_) {
      throw std::invalid_argument(
          "native BitNet DIP teacher top-K dimensions are invalid");
    }
    const std::size_t elements = checked_product(
        rows, hidden_size_,
        "native BitNet DIP teacher dimensions overflow");
    if (input.size() != elements ||
        teacher_record_ids.size() != rows * top_k ||
        (!positive_utility_counts.empty() &&
         positive_utility_counts.size() != rows)) {
      throw std::invalid_argument(
          "native BitNet DIP teacher storage size mismatch");
    }
    ensure_scratch(rows);
    const BaseLayerView& base = base_layers_[layer_index];
    const IndexLayerView& index = index_layers_[layer_index];
    pool_.parallel_for(0, rows, 1, [&](const std::size_t row) {
      teacher_top_k_row(
          base, index, input.data() + row * hidden_size_, top_k,
          teacher_record_ids.data() + row * top_k,
          positive_utility_counts.empty()
              ? nullptr
              : positive_utility_counts.data() + row,
          row);
    });
  }

  [[nodiscard]] std::size_t layer_count() const noexcept {
    return base_layers_.size();
  }
  [[nodiscard]] std::size_t hidden_size() const noexcept {
    return hidden_size_;
  }
  [[nodiscard]] std::size_t intermediate_size() const noexcept {
    return intermediate_size_;
  }
  [[nodiscard]] std::size_t thread_count() const noexcept {
    return pool_.thread_count();
  }
  [[nodiscard]] std::size_t base_bytes() const noexcept {
    return base_mapping_.size();
  }
  [[nodiscard]] std::size_t index_bytes() const noexcept {
    return index_mapping_.size();
  }
  [[nodiscard]] const NativeBitNetDIPPolicy& policy(
      const std::size_t layer) const {
    if (layer >= index_layers_.size()) {
      throw std::out_of_range("native BitNet DIP layer is out of range");
    }
    return index_layers_[layer].policy;
  }

 private:
  void parse_base() {
    if (base_mapping_.size() < kBaseHeaderBytes) {
      throw std::invalid_argument("native BitNet DIP base is truncated");
    }
    const std::byte* bytes = base_mapping_.bytes();
    if (std::memcmp(bytes, kBaseMagic.data(), kBaseMagic.size()) != 0 ||
        little_u32(bytes + 8) != kBaseVersion) {
      throw std::invalid_argument(
          "native BitNet DIP base magic/version mismatch");
    }
    const std::size_t layer_count = little_u32(bytes + 12);
    hidden_size_ = little_u32(bytes + 16);
    intermediate_size_ = little_u32(bytes + 20);
    const std::size_t cache_line = little_u32(bytes + 24);
    const std::size_t directory_entry = little_u32(bytes + 28);
    const std::size_t directory_block = little_u32(bytes + 32);
    const std::size_t logical_record = little_u32(bytes + 36);
    rms_norm_eps_ = little_f32(bytes + 40);
    if (layer_count == 0 || layer_count > kMaximumLayers ||
        hidden_size_ == 0 || hidden_size_ > kMaximumDimension ||
        intermediate_size_ == 0 ||
        intermediate_size_ > kMaximumDimension ||
        cache_line != kCacheLineBytes ||
        directory_entry != kDirectoryEntryBytes ||
        !std::isfinite(rms_norm_eps_) || rms_norm_eps_ <= 0.0F) {
      throw std::invalid_argument("native BitNet DIP base header is invalid");
    }
    packed_width_ = (hidden_size_ + kTritsPerByte - 1) / kTritsPerByte;
    if (packed_width_ % kCacheLineBytes != 0 ||
        logical_record !=
            3U * packed_width_ + sizeof(std::uint16_t)) {
      throw std::invalid_argument(
          "native BitNet DIP base record is not cache-line schedulable");
    }
    const std::size_t directory_payload =
        checked_product(layer_count, kDirectoryEntryBytes,
                        "native BitNet DIP base directory overflows");
    if (directory_block != align_up(directory_payload, kCacheLineBytes)) {
      throw std::invalid_argument(
          "native BitNet DIP base directory is invalid");
    }
    base_metadata_bytes_ =
        align_up(kBaseLayerHeaderBytes + kProjectionScaleBytes,
                 kCacheLineBytes);
    base_stream_bytes_ = checked_product(
        intermediate_size_, packed_width_,
        "native BitNet DIP base stream overflows");
    const std::size_t gate_offset = base_metadata_bytes_;
    const std::size_t up_offset =
        align_up(gate_offset + base_stream_bytes_, kCacheLineBytes);
    const std::size_t gain_offset =
        align_up(up_offset + base_stream_bytes_, kCacheLineBytes);
    const std::size_t gain_bytes =
        intermediate_size_ * sizeof(std::uint16_t);
    const std::size_t down_offset =
        align_up(gain_offset + gain_bytes, kCacheLineBytes);
    const std::size_t payload = down_offset + base_stream_bytes_;
    const std::size_t block = align_up(payload, kCacheLineBytes);
    const std::size_t expected_size =
        kBaseHeaderBytes + directory_block + layer_count * block;
    if (base_mapping_.size() != expected_size) {
      throw std::invalid_argument(
          "native BitNet DIP base length mismatch");
    }
    require_zero(bytes, 44, kBaseHeaderBytes,
                 "native BitNet DIP base header padding is nonzero");
    require_zero(bytes, kBaseHeaderBytes + directory_payload,
                 kBaseHeaderBytes + directory_block,
                 "native BitNet DIP base directory padding is nonzero");
    base_layers_.reserve(layer_count);
    std::size_t expected_offset = kBaseHeaderBytes + directory_block;
    for (std::size_t layer = 0; layer < layer_count; ++layer) {
      const std::byte* entry =
          bytes + kBaseHeaderBytes + layer * kDirectoryEntryBytes;
      const std::size_t offset = little_u64(entry + 8);
      if (little_u32(entry) != layer || little_u32(entry + 4) != 0 ||
          offset != expected_offset || little_u64(entry + 16) != block ||
          little_u32(entry + 24) != payload ||
          little_u32(entry + 28) != 0) {
        throw std::invalid_argument(
            "native BitNet DIP base directory entry is invalid");
      }
      const std::byte* header = bytes + offset;
      if (std::memcmp(header, kBaseLayerMagic.data(),
                      kBaseLayerMagic.size()) != 0 ||
          little_u32(header + 8) != kBaseVersion ||
          little_u32(header + 12) != layer ||
          little_u32(header + 16) != hidden_size_ ||
          little_u32(header + 20) != intermediate_size_ ||
          little_u32(header + 24) != intermediate_size_ ||
          little_u32(header + 28) != packed_width_ ||
          little_u32(header + 32) != logical_record ||
          little_u32(header + 36) != payload ||
          little_u32(header + 40) != 3 ||
          little_u32(header + 44) != 0) {
        throw std::invalid_argument(
            "native BitNet DIP base layer header is invalid");
      }
      require_zero(bytes, offset + 48, offset + kBaseLayerHeaderBytes,
                   "native BitNet DIP base layer-header padding is nonzero");
      const float gate_scale =
          bf16_to_float(little_u16(bytes + offset + 64));
      const float up_scale =
          bf16_to_float(little_u16(bytes + offset + 66));
      const float down_scale =
          bf16_to_float(little_u16(bytes + offset + 68));
      if (!std::isfinite(gate_scale) || gate_scale <= 0.0F ||
          !std::isfinite(up_scale) || up_scale <= 0.0F ||
          !std::isfinite(down_scale) || down_scale <= 0.0F) {
        throw std::invalid_argument(
            "native BitNet DIP projection scale is invalid");
      }
      require_zero(bytes, offset + 70, offset + gate_offset,
                   "native BitNet DIP base metadata padding is nonzero");
      const auto* gate = reinterpret_cast<const std::uint8_t*>(
          bytes + offset + gate_offset);
      const auto* up = reinterpret_cast<const std::uint8_t*>(
          bytes + offset + up_offset);
      const std::byte* gain = bytes + offset + gain_offset;
      const auto* down = reinterpret_cast<const std::uint8_t*>(
          bytes + offset + down_offset);
      require_zero(bytes, offset + gate_offset + base_stream_bytes_,
                   offset + up_offset,
                   "native BitNet DIP base gate padding is nonzero");
      require_zero(bytes, offset + up_offset + base_stream_bytes_,
                   offset + gain_offset,
                   "native BitNet DIP base up padding is nonzero");
      require_zero(bytes, offset + gain_offset + gain_bytes,
                   offset + down_offset,
                   "native BitNet DIP base gain padding is nonzero");
      require_zero(bytes, offset + payload, offset + block,
                   "native BitNet DIP base layer padding is nonzero");
      validate_packed_rows(gate, intermediate_size_, packed_width_,
                           hidden_size_);
      validate_packed_rows(up, intermediate_size_, packed_width_,
                           hidden_size_);
      validate_packed_rows(down, intermediate_size_, packed_width_,
                           hidden_size_);
      for (std::size_t record = 0; record < intermediate_size_; ++record) {
        if (!std::isfinite(
                bf16_to_float(little_u16(gain + 2U * record)))) {
          throw std::invalid_argument(
              "native BitNet DIP gain is not finite");
        }
      }
      base_layers_.push_back({gate, up, gain, down, gate_scale, up_scale,
                              down_scale, block});
      expected_offset += block;
    }
  }

  void parse_index() {
    if (index_mapping_.size() < kIndexHeaderBytes) {
      throw std::invalid_argument("native BitNet DIP index is truncated");
    }
    const std::byte* bytes = index_mapping_.bytes();
    if (std::memcmp(bytes, kIndexMagic.data(), kIndexMagic.size()) != 0 ||
        little_u32(bytes + 8) != kVersion ||
        little_u32(bytes + 12) != kEndianMarker ||
        little_u32(bytes + 16) != kIndexHeaderBytes ||
        little_u32(bytes + 20) != kDirectoryEntryBytes ||
        little_u32(bytes + 28) != kIndexLayerHeaderBytes ||
        little_u32(bytes + 32) != kCacheLineBytes ||
        little_u32(bytes + 36) != hidden_size_ ||
        little_u32(bytes + 40) != intermediate_size_ ||
        little_u32(bytes + 44) != base_layers_.size() ||
        little_u32(bytes + 48) != kTritsPerByte ||
        little_u32(bytes + 52) != kCoordinateEncoding ||
        little_u32(bytes + 60) != kChecksumSha256) {
      throw std::invalid_argument(
          "native BitNet DIP index metadata is invalid");
    }
    const std::size_t layer_count = base_layers_.size();
    const std::size_t directory_block = little_u32(bytes + 24);
    const std::size_t directory_payload =
        layer_count * kDirectoryEntryBytes;
    if (directory_block != align_up(directory_payload, kCacheLineBytes)) {
      throw std::invalid_argument(
          "native BitNet DIP index directory is invalid");
    }
    const std::size_t norm_value_bytes =
        hidden_size_ <= std::numeric_limits<std::uint16_t>::max() ? 2 : 4;
    const std::uint32_t expected_norm_code =
        norm_value_bytes == 2 ? kNormUint16 : kNormUint32;
    if (little_u32(bytes + 56) != expected_norm_code) {
      throw std::invalid_argument(
          "native BitNet DIP index norm dtype is invalid");
    }
    const auto base_digest =
        sha256(base_mapping_.bytes(), base_mapping_.size());
    if (!std::equal(base_digest.begin(), base_digest.end(),
                    reinterpret_cast<const std::uint8_t*>(bytes + 64))) {
      throw std::invalid_argument(
          "native BitNet DIP index source-artifact SHA-256 mismatch");
    }
    require_zero(bytes, 96, kIndexHeaderBytes,
                 "native BitNet DIP index header padding is nonzero");
    require_zero(bytes, kIndexHeaderBytes + directory_payload,
                 kIndexHeaderBytes + directory_block,
                 "native BitNet DIP index directory padding is nonzero");

    coordinate_payload_ =
        (intermediate_size_ + kTritsPerByte - 1) / kTritsPerByte;
    coordinate_stride_ =
        align_up(coordinate_payload_, kCacheLineBytes);
    coordinate_stream_bytes_ =
        hidden_size_ * coordinate_stride_;
    down_norm_stream_bytes_ = align_up(
        intermediate_size_ * norm_value_bytes, kCacheLineBytes);
    const std::size_t gate_offset = kIndexLayerHeaderBytes;
    const std::size_t up_offset =
        gate_offset + coordinate_stream_bytes_;
    const std::size_t norm_offset =
        up_offset + coordinate_stream_bytes_;
    const std::size_t payload =
        norm_offset + down_norm_stream_bytes_;
    const std::size_t block = align_up(payload, kCacheLineBytes);
    const std::size_t expected_size =
        kIndexHeaderBytes + directory_block + layer_count * block;
    if (index_mapping_.size() != expected_size) {
      throw std::invalid_argument(
          "native BitNet DIP index length mismatch");
    }

    index_layers_.reserve(layer_count);
    std::size_t expected_offset = kIndexHeaderBytes + directory_block;
    for (std::size_t layer = 0; layer < layer_count; ++layer) {
      const std::byte* entry =
          bytes + kIndexHeaderBytes + layer * kDirectoryEntryBytes;
      const std::size_t input_count = little_u32(entry + 4);
      const std::size_t candidate_count = little_u32(entry + 8);
      const std::size_t maximum_top_k = little_u32(entry + 12);
      const std::size_t offset = little_u64(entry + 16);
      if (little_u32(entry) != layer || input_count == 0 ||
          input_count > hidden_size_ || candidate_count == 0 ||
          candidate_count > intermediate_size_ || maximum_top_k == 0 ||
          maximum_top_k > candidate_count || offset != expected_offset ||
          little_u64(entry + 24) != block) {
        throw std::invalid_argument(
            "native BitNet DIP index directory entry is invalid");
      }
      const std::byte* header = bytes + offset;
      const std::size_t minimum_top_k = little_u32(header + 8);
      const float energy_target = little_f32(header + 12);
      const std::size_t rms_audit_count = little_u32(header + 64);
      const auto rms_estimator = static_cast<NativeBitNetDIPRMSEstimator>(
          little_u32(header + 68));
      const auto rms_audit_strategy =
          static_cast<NativeBitNetDIPAuditStrategy>(
              little_u32(header + 72));
      if (std::memcmp(header, kIndexLayerMagic.data(),
                      kIndexLayerMagic.size()) != 0 ||
          minimum_top_k == 0 || minimum_top_k > maximum_top_k ||
          !std::isfinite(energy_target) || energy_target <= 0.0F ||
          energy_target > 1.0F ||
          little_u32(header + 48) != gate_offset ||
          little_u32(header + 52) != up_offset ||
          little_u32(header + 56) != norm_offset ||
          little_u32(header + 60) != payload ||
          rms_audit_count > candidate_count ||
          maximum_top_k > candidate_count - rms_audit_count ||
          (rms_estimator !=
               NativeBitNetDIPRMSEstimator::candidate_ratio &&
           rms_estimator !=
               NativeBitNetDIPRMSEstimator::corrected_proxy) ||
          (rms_estimator ==
               NativeBitNetDIPRMSEstimator::candidate_ratio &&
           rms_audit_count != 0) ||
          (rms_estimator ==
               NativeBitNetDIPRMSEstimator::corrected_proxy &&
           (rms_audit_count == 0 ||
            rms_audit_strategy !=
                NativeBitNetDIPAuditStrategy::top_proxy_raw_square)) ||
          (rms_audit_count == 0 &&
           rms_audit_strategy != NativeBitNetDIPAuditStrategy::none)) {
        throw std::invalid_argument(
            "native BitNet DIP authenticated layer policy is invalid");
      }
      require_zero(bytes, offset + kIndexLayerHeaderCoreBytes,
                   offset + kIndexLayerHeaderBytes,
                   "native BitNet DIP index layer-header padding is nonzero");
      const NativeBitNetDIPPolicy policy{
          .input_coordinates = input_count,
          .candidate_count = candidate_count,
          .minimum_top_k = minimum_top_k,
          .maximum_top_k = maximum_top_k,
          .energy_target = energy_target,
          .rms_audit_count = rms_audit_count,
          .rms_estimator = rms_estimator,
          .rms_audit_strategy = rms_audit_strategy,
      };
      std::array<std::uint8_t, 36> policy_bytes{};
      store_little_u32(policy_bytes.data(), static_cast<std::uint32_t>(layer));
      store_little_u32(policy_bytes.data() + 4,
                       static_cast<std::uint32_t>(input_count));
      store_little_u32(policy_bytes.data() + 8,
                       static_cast<std::uint32_t>(candidate_count));
      store_little_u32(policy_bytes.data() + 12,
                       static_cast<std::uint32_t>(minimum_top_k));
      store_little_u32(policy_bytes.data() + 16,
                       static_cast<std::uint32_t>(maximum_top_k));
      store_little_u32(policy_bytes.data() + 20,
                       static_cast<std::uint32_t>(rms_audit_count));
      store_little_u32(
          policy_bytes.data() + 24,
          static_cast<std::uint32_t>(rms_estimator));
      store_little_u32(
          policy_bytes.data() + 28,
          static_cast<std::uint32_t>(rms_audit_strategy));
      store_little_u32(policy_bytes.data() + 32,
                       std::bit_cast<std::uint32_t>(energy_target));
      Sha256 layer_hash;
      layer_hash.update(policy_bytes.data(), policy_bytes.size());
      layer_hash.update(
          reinterpret_cast<const std::uint8_t*>(
              bytes + offset + kIndexLayerHeaderBytes),
          block - kIndexLayerHeaderBytes);
      const auto digest = layer_hash.finish();
      if (!std::equal(digest.begin(), digest.end(),
                      reinterpret_cast<const std::uint8_t*>(header + 16))) {
        throw std::invalid_argument(
            "native BitNet DIP index layer checksum mismatch");
      }
      const auto* gate_coordinates =
          reinterpret_cast<const std::uint8_t*>(
              bytes + offset + gate_offset);
      const auto* up_coordinates =
          reinterpret_cast<const std::uint8_t*>(
              bytes + offset + up_offset);
      const std::size_t used_tail =
          intermediate_size_ % kTritsPerByte;
      const std::uint8_t tail_limit =
          used_tail == 0
              ? 243U
              : static_cast<std::uint8_t>(
                    std::array<std::uint16_t, 5>{1U, 3U, 9U, 27U, 81U}
                        [used_tail]);
      for (const std::uint8_t* stream :
           {gate_coordinates, up_coordinates}) {
        for (std::size_t coordinate = 0; coordinate < hidden_size_;
             ++coordinate) {
          const std::uint8_t* row =
              stream + coordinate * coordinate_stride_;
          for (std::size_t packed = 0; packed < coordinate_payload_;
               ++packed) {
            if (row[packed] > 242U) {
              throw std::invalid_argument(
                  "native BitNet DIP coordinate byte is not canonical");
            }
          }
          if (used_tail != 0 &&
              row[coordinate_payload_ - 1] >= tail_limit) {
            throw std::invalid_argument(
                "native BitNet DIP coordinate tail is not canonical");
          }
          for (std::size_t padding = coordinate_payload_;
               padding < coordinate_stride_; ++padding) {
            if (row[padding] != 0U) {
              throw std::invalid_argument(
                  "native BitNet DIP coordinate padding is nonzero");
            }
          }
        }
      }
      const std::byte* down_norm = bytes + offset + norm_offset;
      for (std::size_t record = 0; record < intermediate_size_; ++record) {
        const std::size_t norm =
            norm_value_bytes == 2
                ? little_u16(down_norm + 2U * record)
                : little_u32(down_norm + 4U * record);
        if (norm > hidden_size_) {
          throw std::invalid_argument(
              "native BitNet DIP down norm exceeds hidden size");
        }
      }
      require_zero(
          bytes, offset + norm_offset + intermediate_size_ * norm_value_bytes,
          offset + norm_offset + down_norm_stream_bytes_,
          "native BitNet DIP down-norm padding is nonzero");
      require_zero(bytes, offset + payload, offset + block,
                   "native BitNet DIP index layer padding is nonzero");
      index_layers_.push_back(
          {gate_coordinates, up_coordinates, down_norm, policy, block});
      expected_offset += block;
    }
  }

  [[nodiscard]] float gain(const BaseLayerView& layer,
                           const std::size_t record) const noexcept {
    return bf16_to_float(little_u16(layer.gain + 2U * record));
  }

  [[nodiscard]] float down_norm(const IndexLayerView& layer,
                                const std::size_t record) const noexcept {
    if (hidden_size_ <= std::numeric_limits<std::uint16_t>::max()) {
      return static_cast<float>(
          little_u16(layer.down_norm + 2U * record));
    }
    return static_cast<float>(
        little_u32(layer.down_norm + 4U * record));
  }

  void forward_row(const BaseLayerView& base, const IndexLayerView& index,
                   const std::uint16_t* input, std::uint16_t* output,
                   const std::size_t row) {
    const NativeBitNetDIPPolicy& policy = index.policy;
    float* state = quantized_input_.data() + row * hidden_size_;
    std::size_t* coordinates =
        coordinate_order_.data() + row * hidden_size_;
    float maximum = 0.0F;
    for (std::size_t coordinate = 0; coordinate < hidden_size_;
         ++coordinate) {
      const float value = bf16_to_float(input[coordinate]);
      if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "native BitNet DIP input must be finite");
      }
      maximum = std::max(maximum, std::abs(value));
      coordinates[coordinate] = coordinate;
    }
    const float input_scale = 127.0F / std::max(maximum, 1.0e-5F);
    for (std::size_t coordinate = 0; coordinate < hidden_size_;
         ++coordinate) {
      state[coordinate] =
          quantized_bf16(bf16_to_float(input[coordinate]), input_scale);
    }
    const auto stronger_input = [&](const std::size_t left,
                                    const std::size_t right) {
      const float left_value = std::abs(state[left]);
      const float right_value = std::abs(state[right]);
      return left_value == right_value ? left < right
                                       : left_value > right_value;
    };
    std::partial_sort(coordinates,
                      coordinates + policy.input_coordinates,
                      coordinates + hidden_size_, stronger_input);

    float* proxy_gate =
        proxy_gate_.data() + row * intermediate_size_;
    float* proxy_up = proxy_up_.data() + row * intermediate_size_;
    float* proxy_raw = proxy_raw_.data() + row * intermediate_size_;
    std::fill_n(proxy_gate, intermediate_size_, 0.0F);
    std::fill_n(proxy_up, intermediate_size_, 0.0F);
    for (std::size_t selected = 0;
         selected < policy.input_coordinates; ++selected) {
      const std::size_t coordinate = coordinates[selected];
      const float value = state[coordinate];
      const std::uint8_t* gate_row =
          index.gate_coordinates + coordinate * coordinate_stride_;
      const std::uint8_t* up_row =
          index.up_coordinates + coordinate * coordinate_stride_;
      for (std::size_t packed = 0; packed < coordinate_payload_; ++packed) {
        const auto& gate_trits = kTritTable[gate_row[packed]];
        const auto& up_trits = kTritTable[up_row[packed]];
        const std::size_t record_begin = packed * kTritsPerByte;
        for (std::size_t digit = 0; digit < kTritsPerByte; ++digit) {
          const std::size_t record = record_begin + digit;
          if (record >= intermediate_size_) break;
          proxy_gate[record] +=
              static_cast<float>(gate_trits[digit]) * value;
          proxy_up[record] +=
              static_cast<float>(up_trits[digit]) * value;
        }
      }
    }
    for (std::size_t record = 0; record < intermediate_size_; ++record) {
      const float gate =
          bf16_round(bf16_round(proxy_gate[record]) * base.gate_scale);
      const float up =
          bf16_round(bf16_round(proxy_up[record]) * base.up_scale);
      const float positive = std::max(gate, 0.0F);
      proxy_raw[record] =
          bf16_round(bf16_round(positive * positive) * up);
    }

    std::size_t* record_order =
        record_order_.data() + row * intermediate_size_;
    std::iota(record_order, record_order + intermediate_size_, 0U);
    const auto stronger_proxy = [&](const std::size_t left,
                                    const std::size_t right) {
      const float left_raw = proxy_raw[left];
      const float right_raw = proxy_raw[right];
      const float left_gain = gain(base, left);
      const float right_gain = gain(base, right);
      const float left_utility = left_raw * left_raw * left_gain *
                                 left_gain * down_norm(index, left);
      const float right_utility = right_raw * right_raw * right_gain *
                                  right_gain * down_norm(index, right);
      return left_utility == right_utility ? left < right
                                           : left_utility > right_utility;
    };
    const std::size_t routed_count =
        policy.candidate_count - policy.rms_audit_count;
    std::partial_sort(record_order, record_order + routed_count,
                      record_order + intermediate_size_, stronger_proxy);
    std::size_t* candidates =
        candidate_indices_.data() + row * intermediate_size_;
    std::copy_n(record_order, routed_count, candidates);
    if (policy.rms_audit_count != 0) {
      std::vector<std::uint8_t>& routed_flags = routed_flags_[row];
      std::fill(routed_flags.begin(), routed_flags.end(), std::uint8_t{0});
      for (std::size_t position = 0; position < routed_count; ++position) {
        routed_flags[candidates[position]] = 1;
      }
      std::size_t tail_count = 0;
      for (std::size_t record = 0; record < intermediate_size_; ++record) {
        if (routed_flags[record] == 0) {
          record_order[tail_count++] = record;
        }
      }
      const auto stronger_audit = [&](const std::size_t left,
                                      const std::size_t right) {
        const float left_value = proxy_raw[left] * proxy_raw[left];
        const float right_value = proxy_raw[right] * proxy_raw[right];
        return left_value == right_value ? left < right
                                         : left_value > right_value;
      };
      std::partial_sort(record_order,
                        record_order + policy.rms_audit_count,
                        record_order + tail_count, stronger_audit);
      std::copy_n(record_order, policy.rms_audit_count,
                  candidates + routed_count);
    }

    float* corrected_raw =
        corrected_raw_.data() + row * intermediate_size_;
    std::copy_n(proxy_raw, intermediate_size_, corrected_raw);
    for (std::size_t position = 0; position < policy.candidate_count;
         ++position) {
      const std::size_t record = candidates[position];
      float gate_accumulator = 0.0F;
      float up_accumulator = 0.0F;
      const std::uint8_t* gate_row =
          base.gate + record * packed_width_;
      const std::uint8_t* up_row =
          base.up + record * packed_width_;
      for (std::size_t packed = 0; packed < packed_width_; ++packed) {
        const auto& gate_trits = kTritTable[gate_row[packed]];
        const auto& up_trits = kTritTable[up_row[packed]];
        const std::size_t coordinate_begin = packed * kTritsPerByte;
        for (std::size_t digit = 0; digit < kTritsPerByte; ++digit) {
          const std::size_t coordinate = coordinate_begin + digit;
          if (coordinate >= hidden_size_) break;
          gate_accumulator +=
              static_cast<float>(gate_trits[digit]) * state[coordinate];
          up_accumulator +=
              static_cast<float>(up_trits[digit]) * state[coordinate];
        }
      }
      const float gate =
          bf16_round(bf16_round(gate_accumulator) * base.gate_scale);
      const float up =
          bf16_round(bf16_round(up_accumulator) * base.up_scale);
      const float positive = std::max(gate, 0.0F);
      corrected_raw[record] =
          bf16_round(bf16_round(positive * positive) * up);
    }

    double proxy_square_sum = 0.0;
    for (std::size_t record = 0; record < intermediate_size_; ++record) {
      proxy_square_sum +=
          static_cast<double>(proxy_raw[record] * proxy_raw[record]);
    }
    double exact_candidate_square_sum = 0.0;
    double proxy_candidate_square_sum = 0.0;
    for (std::size_t position = 0; position < policy.candidate_count;
         ++position) {
      const std::size_t record = candidates[position];
      exact_candidate_square_sum +=
          static_cast<double>(corrected_raw[record]) *
          static_cast<double>(corrected_raw[record]);
      proxy_candidate_square_sum +=
          static_cast<double>(proxy_raw[record]) *
          static_cast<double>(proxy_raw[record]);
    }
    double corrected_square_sum = proxy_square_sum;
    if (policy.rms_estimator ==
        NativeBitNetDIPRMSEstimator::candidate_ratio) {
      const double tail_scale =
          proxy_candidate_square_sum <= 1.0e-30
              ? 1.0
              : exact_candidate_square_sum /
                    proxy_candidate_square_sum;
      corrected_square_sum =
          exact_candidate_square_sum +
          tail_scale *
              std::max(proxy_square_sum - proxy_candidate_square_sum, 0.0);
    } else {
      for (std::size_t position = 0;
           position < policy.candidate_count; ++position) {
        const std::size_t record = candidates[position];
        const double exact = static_cast<double>(corrected_raw[record]);
        const double proxy = static_cast<double>(proxy_raw[record]);
        corrected_square_sum += exact * exact - proxy * proxy;
      }
    }
    const float variance = static_cast<float>(
        std::max(corrected_square_sum /
                     static_cast<double>(intermediate_size_),
                 0.0));
    const float inverse_rms =
        1.0F / std::sqrt(variance + rms_norm_eps_);
    float normalized_absmax = 0.0F;
    for (std::size_t record = 0; record < intermediate_size_; ++record) {
      const float normalized =
          bf16_round(bf16_round(corrected_raw[record] * inverse_rms) *
                     gain(base, record));
      normalized_absmax =
          std::max(normalized_absmax, std::abs(normalized));
    }
    const float coefficient_scale =
        127.0F / std::max(normalized_absmax, 1.0e-5F);
    float* coefficients =
        candidate_coefficients_.data() + row * intermediate_size_;
    float* utilities =
        candidate_utilities_.data() + row * intermediate_size_;
    for (std::size_t position = 0; position < policy.candidate_count;
         ++position) {
      const std::size_t record = candidates[position];
      const float normalized =
          bf16_round(bf16_round(corrected_raw[record] * inverse_rms) *
                     gain(base, record));
      const float coefficient =
          quantized_bf16(normalized, coefficient_scale);
      coefficients[position] = coefficient;
      utilities[position] =
          coefficient * coefficient * down_norm(index, record);
      record_order[position] = position;
    }
    const auto stronger_exact = [&](const std::size_t left,
                                    const std::size_t right) {
      const float left_value = utilities[left];
      const float right_value = utilities[right];
      const std::size_t left_record = candidates[left];
      const std::size_t right_record = candidates[right];
      return left_value == right_value ? left_record < right_record
                                       : left_value > right_value;
    };
    std::sort(record_order, record_order + policy.candidate_count,
              stronger_exact);
    std::size_t target_count = 0;
    if (policy.energy_target >= 1.0F) {
      for (std::size_t position = 0; position < policy.candidate_count;
           ++position) {
        target_count += utilities[record_order[position]] > 0.0F ? 1U : 0U;
      }
    } else {
      double total = 0.0;
      for (std::size_t position = 0; position < policy.candidate_count;
           ++position) {
        total += utilities[record_order[position]];
      }
      if (total > 0.0) {
        double cumulative = 0.0;
        for (std::size_t position = 0; position < policy.candidate_count;
             ++position) {
          cumulative += utilities[record_order[position]];
          ++target_count;
          if (cumulative >=
              static_cast<double>(policy.energy_target) * total) {
            break;
          }
        }
      }
    }
    const std::size_t selected_count =
        std::min(policy.maximum_top_k,
                 std::max(policy.minimum_top_k, target_count));
    row_selected_counts_[row] =
        static_cast<std::uint32_t>(selected_count);
    float* down_accumulator =
        down_accumulator_.data() + row * hidden_size_;
    std::fill_n(down_accumulator, hidden_size_, 0.0F);
    for (std::size_t selected = 0; selected < selected_count; ++selected) {
      const std::size_t candidate_position = record_order[selected];
      const std::size_t record = candidates[candidate_position];
      const float coefficient = coefficients[candidate_position];
      if (coefficient == 0.0F) {
        continue;
      }
      const std::uint8_t* down_row =
          base.down + record * packed_width_;
      for (std::size_t packed = 0; packed < packed_width_; ++packed) {
        const auto& down_trits = kTritTable[down_row[packed]];
        const std::size_t coordinate_begin = packed * kTritsPerByte;
        for (std::size_t digit = 0; digit < kTritsPerByte; ++digit) {
          const std::size_t coordinate = coordinate_begin + digit;
          if (coordinate >= hidden_size_) break;
          down_accumulator[coordinate] +=
              static_cast<float>(down_trits[digit]) * coefficient;
        }
      }
    }
    for (std::size_t coordinate = 0; coordinate < hidden_size_;
         ++coordinate) {
      const float value =
          bf16_round(bf16_round(down_accumulator[coordinate]) *
                     base.down_scale);
      output[coordinate] = float_to_bf16_bits(value);
    }
  }

  void teacher_top_k_row(const BaseLayerView& base,
                         const IndexLayerView& index,
                         const std::uint16_t* input,
                         const std::size_t top_k,
                         std::uint32_t* teacher_ids,
                         std::uint32_t* positive_utility_count,
                         const std::size_t row) {
    float* state = quantized_input_.data() + row * hidden_size_;
    float maximum = 0.0F;
    for (std::size_t coordinate = 0; coordinate < hidden_size_;
         ++coordinate) {
      const float value = bf16_to_float(input[coordinate]);
      if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "native BitNet DIP teacher input must be finite");
      }
      maximum = std::max(maximum, std::abs(value));
    }
    const float input_scale = 127.0F / std::max(maximum, 1.0e-5F);
    for (std::size_t coordinate = 0; coordinate < hidden_size_;
         ++coordinate) {
      state[coordinate] =
          quantized_bf16(bf16_to_float(input[coordinate]), input_scale);
    }
    float* raw = corrected_raw_.data() + row * intermediate_size_;
    for (std::size_t record = 0; record < intermediate_size_; ++record) {
      float gate_accumulator = 0.0F;
      float up_accumulator = 0.0F;
      const std::uint8_t* gate_row =
          base.gate + record * packed_width_;
      const std::uint8_t* up_row =
          base.up + record * packed_width_;
      for (std::size_t packed = 0; packed < packed_width_; ++packed) {
        const auto& gate_trits = kTritTable[gate_row[packed]];
        const auto& up_trits = kTritTable[up_row[packed]];
        const std::size_t coordinate_begin = packed * kTritsPerByte;
        for (std::size_t digit = 0; digit < kTritsPerByte; ++digit) {
          const std::size_t coordinate = coordinate_begin + digit;
          if (coordinate >= hidden_size_) break;
          gate_accumulator +=
              static_cast<float>(gate_trits[digit]) * state[coordinate];
          up_accumulator +=
              static_cast<float>(up_trits[digit]) * state[coordinate];
        }
      }
      const float gate =
          bf16_round(bf16_round(gate_accumulator) * base.gate_scale);
      const float up =
          bf16_round(bf16_round(up_accumulator) * base.up_scale);
      const float positive = std::max(gate, 0.0F);
      raw[record] =
          bf16_round(bf16_round(positive * positive) * up);
    }
    float square_sum = 0.0F;
    for (std::size_t record = 0; record < intermediate_size_; ++record) {
      square_sum += raw[record] * raw[record];
    }
    const float variance =
        square_sum / static_cast<float>(intermediate_size_);
    const float inverse_rms =
        1.0F / std::sqrt(variance + rms_norm_eps_);
    float normalized_absmax = 0.0F;
    float* coefficients =
        candidate_coefficients_.data() + row * intermediate_size_;
    for (std::size_t record = 0; record < intermediate_size_; ++record) {
      coefficients[record] =
          bf16_round(bf16_round(raw[record] * inverse_rms) *
                     gain(base, record));
      normalized_absmax =
          std::max(normalized_absmax, std::abs(coefficients[record]));
    }
    const float coefficient_scale =
        127.0F / std::max(normalized_absmax, 1.0e-5F);
    float* utilities =
        candidate_utilities_.data() + row * intermediate_size_;
    std::size_t* order =
        record_order_.data() + row * intermediate_size_;
    for (std::size_t record = 0; record < intermediate_size_; ++record) {
      const float coefficient =
          quantized_bf16(coefficients[record], coefficient_scale);
      coefficients[record] = coefficient;
      utilities[record] =
          coefficient * coefficient * down_norm(index, record);
      order[record] = record;
    }
    if (positive_utility_count != nullptr) {
      *positive_utility_count = static_cast<std::uint32_t>(
          std::count_if(
              utilities, utilities + intermediate_size_,
              [](const float utility) { return utility > 0.0F; }));
    }
    const auto stronger = [&](const std::size_t left,
                              const std::size_t right) {
      return utilities[left] == utilities[right]
                 ? left < right
                 : utilities[left] > utilities[right];
    };
    std::partial_sort(order, order + top_k,
                      order + intermediate_size_, stronger);
    std::transform(order, order + top_k, teacher_ids,
                   [](const std::size_t value) {
                     return static_cast<std::uint32_t>(value);
                   });
  }

  void ensure_scratch(const std::size_t rows) {
    if (rows <= scratch_rows_) {
      return;
    }
    scratch_rows_ = rows;
    const std::size_t hidden_rows = checked_product(
        rows, hidden_size_, "native BitNet DIP scratch overflows");
    const std::size_t intermediate_rows = checked_product(
        rows, intermediate_size_, "native BitNet DIP scratch overflows");
    quantized_input_.resize(hidden_rows);
    coordinate_order_.resize(hidden_rows);
    proxy_gate_.resize(intermediate_rows);
    proxy_up_.resize(intermediate_rows);
    proxy_raw_.resize(intermediate_rows);
    corrected_raw_.resize(intermediate_rows);
    record_order_.resize(intermediate_rows);
    candidate_indices_.resize(intermediate_rows);
    candidate_coefficients_.resize(intermediate_rows);
    candidate_utilities_.resize(intermediate_rows);
    down_accumulator_.resize(hidden_rows);
    row_selected_counts_.resize(rows);
    routed_flags_.resize(rows);
    for (auto& flags : routed_flags_) {
      flags.resize(intermediate_size_);
    }
  }

  [[nodiscard]] std::uint64_t scratch_bytes() const noexcept {
    std::uint64_t total =
        sizeof(float) *
            (quantized_input_.capacity() + proxy_gate_.capacity() +
             proxy_up_.capacity() + proxy_raw_.capacity() +
             corrected_raw_.capacity() +
             candidate_coefficients_.capacity() +
             candidate_utilities_.capacity() +
             down_accumulator_.capacity()) +
        sizeof(std::size_t) *
            (coordinate_order_.capacity() + record_order_.capacity() +
             candidate_indices_.capacity()) +
        sizeof(std::uint32_t) * row_selected_counts_.capacity();
    for (const auto& flags : routed_flags_) {
      total += flags.capacity();
    }
    return total;
  }

  void populate_metrics(
      const IndexLayerView& index, const std::size_t rows,
      const std::chrono::steady_clock::time_point started,
      const std::chrono::steady_clock::time_point finished,
      NativeBitNetDIPMetrics* metrics) const {
    *metrics = NativeBitNetDIPMetrics{};
    metrics->elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            finished - started)
            .count());
    const std::uint64_t selected_total = std::accumulate(
        row_selected_counts_.begin(),
        row_selected_counts_.begin() + rows, std::uint64_t{0});
    const auto [minimum, maximum] = std::minmax_element(
        row_selected_counts_.begin(), row_selected_counts_.begin() + rows);
    const std::uint64_t coordinate =
        2U * index.policy.input_coordinates * coordinate_stride_ * rows;
    const std::uint64_t completion =
        2U * index.policy.candidate_count * packed_width_ * rows;
    const std::uint64_t gains =
        align_up(intermediate_size_ * sizeof(std::uint16_t),
                 kCacheLineBytes) *
        rows;
    const std::uint64_t norms = down_norm_stream_bytes_ * rows;
    const std::uint64_t down = selected_total * packed_width_;
    const std::uint64_t metadata =
        (base_metadata_bytes_ + kIndexLayerHeaderBytes) * rows;
    metrics->coordinate_stream_bytes = coordinate;
    metrics->candidate_completion_bytes = completion;
    metrics->gain_stream_bytes = gains;
    metrics->down_norm_stream_bytes = norms;
    metrics->selected_down_stream_bytes = down;
    metrics->layer_metadata_bytes = metadata;
    metrics->scheduled_cache_line_bytes =
        coordinate + completion + gains + norms + down + metadata;
    metrics->scratch_bytes = scratch_bytes();
    metrics->rows = rows;
    metrics->threads = pool_.thread_count();
    metrics->input_coordinates = index.policy.input_coordinates;
    metrics->candidate_count = index.policy.candidate_count;
    metrics->selected_count_total = selected_total;
    metrics->selected_count_min = *minimum;
    metrics->selected_count_max = *maximum;
  }

  MappedFile base_mapping_;
  MappedFile index_mapping_;
  ThreadPool pool_;
  std::size_t hidden_size_{};
  std::size_t intermediate_size_{};
  std::size_t packed_width_{};
  std::size_t base_metadata_bytes_{};
  std::size_t base_stream_bytes_{};
  std::size_t coordinate_payload_{};
  std::size_t coordinate_stride_{};
  std::size_t coordinate_stream_bytes_{};
  std::size_t down_norm_stream_bytes_{};
  float rms_norm_eps_{};
  std::vector<BaseLayerView> base_layers_;
  std::vector<IndexLayerView> index_layers_;
  std::size_t scratch_rows_{};
  std::vector<float> quantized_input_;
  std::vector<std::size_t> coordinate_order_;
  std::vector<float> proxy_gate_;
  std::vector<float> proxy_up_;
  std::vector<float> proxy_raw_;
  std::vector<float> corrected_raw_;
  std::vector<std::size_t> record_order_;
  std::vector<std::size_t> candidate_indices_;
  std::vector<float> candidate_coefficients_;
  std::vector<float> candidate_utilities_;
  std::vector<float> down_accumulator_;
  std::vector<std::uint32_t> row_selected_counts_;
  std::vector<std::vector<std::uint8_t>> routed_flags_;
};

NativeBitNetDIPKernel::NativeBitNetDIPKernel(
    const std::filesystem::path& record_artifact,
    const std::filesystem::path& coordinate_index,
    const std::size_t thread_count)
    : impl_(std::make_unique<Impl>(record_artifact, coordinate_index,
                                   thread_count)) {}

NativeBitNetDIPKernel::~NativeBitNetDIPKernel() = default;
NativeBitNetDIPKernel::NativeBitNetDIPKernel(
    NativeBitNetDIPKernel&&) noexcept = default;
NativeBitNetDIPKernel& NativeBitNetDIPKernel::operator=(
    NativeBitNetDIPKernel&&) noexcept = default;

std::size_t NativeBitNetDIPKernel::layer_count() const noexcept {
  return impl_->layer_count();
}
std::size_t NativeBitNetDIPKernel::hidden_size() const noexcept {
  return impl_->hidden_size();
}
std::size_t NativeBitNetDIPKernel::intermediate_size() const noexcept {
  return impl_->intermediate_size();
}
std::size_t NativeBitNetDIPKernel::thread_count() const noexcept {
  return impl_->thread_count();
}
std::size_t NativeBitNetDIPKernel::record_artifact_bytes() const noexcept {
  return impl_->base_bytes();
}
std::size_t NativeBitNetDIPKernel::coordinate_index_bytes() const noexcept {
  return impl_->index_bytes();
}
const NativeBitNetDIPPolicy& NativeBitNetDIPKernel::policy(
    const std::size_t layer) const {
  return impl_->policy(layer);
}
void NativeBitNetDIPKernel::forward_bf16(
    const std::size_t layer, const std::span<const std::uint16_t> input,
    const std::size_t rows, const std::span<std::uint16_t> output,
    const std::span<std::uint32_t> selected_counts,
    NativeBitNetDIPMetrics* const metrics) {
  impl_->forward(layer, input, rows, output, selected_counts, metrics);
}

void NativeBitNetDIPKernel::forward_debug_bf16(
    const std::size_t layer,
    const std::span<const std::uint16_t> input,
    const std::size_t rows, const std::span<std::uint16_t> output,
    const std::span<std::uint32_t> selected_counts,
    const std::span<std::uint32_t> input_coordinate_ids,
    const std::span<std::uint32_t> candidate_ids,
    const std::span<std::uint32_t> selected_record_ids,
    NativeBitNetDIPMetrics* const metrics) {
  impl_->forward(layer, input, rows, output, selected_counts, metrics,
                 input_coordinate_ids, candidate_ids,
                 selected_record_ids);
}

void NativeBitNetDIPKernel::teacher_top_k_bf16(
    const std::size_t layer,
    const std::span<const std::uint16_t> input,
    const std::size_t rows, const std::size_t top_k,
    const std::span<std::uint32_t> teacher_record_ids,
    const std::span<std::uint32_t> positive_utility_counts) {
  impl_->teacher_top_k(layer, input, rows, top_k, teacher_record_ids,
                       positive_utility_counts);
}

}  // namespace engram
