#include "engram/native_bitnet.h"

#include "engram/thread_pool.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace engram {
namespace {

constexpr char kMagic[8] = {'E', 'N', 'G', 'B', 'N', 'P', '1', '1'};
constexpr char kLayerMagic[8] = {'E', 'N', 'G', 'B', 'N', 'P', 'L', '1'};
constexpr std::size_t kHeaderBytes = 64;
constexpr std::size_t kLayerHeaderBytes = 64;
constexpr std::size_t kDirectoryEntryBytes = 32;
constexpr std::size_t kProjectionScaleBytes = 6;
constexpr std::size_t kCacheLineBytes = 64;
constexpr std::size_t kTritsPerByte = 5;
constexpr std::size_t kDownPackedBlock = 32;
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kMaximumDimension = 1U << 20U;
constexpr std::size_t kMaximumLayers = 1U << 12U;

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

std::size_t align_up(const std::size_t value, const std::size_t alignment) {
  if (alignment == 0 ||
      value > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
    throw std::invalid_argument("native BitNet alignment overflow");
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

std::size_t checked_sum(const std::size_t left, const std::size_t right,
                        const char* message) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::invalid_argument(message);
  }
  return left + right;
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

float quantized_bf16(const float value, const float scale) noexcept {
  const float rounded = std::nearbyint(value * scale);
  const float clamped = std::clamp(rounded, -128.0F, 127.0F);
  return bf16_round(clamped / scale);
}

void require_zero(const std::byte* bytes, const std::size_t begin,
                  const std::size_t end, const char* message) {
  for (std::size_t index = begin; index < end; ++index) {
    if (bytes[index] != std::byte{0}) {
      throw std::invalid_argument(message);
    }
  }
}

struct LayerView {
  const std::uint8_t* gate{};
  const std::uint8_t* up{};
  const std::uint16_t* norm{};
  const std::uint8_t* down{};
  float gate_scale{};
  float up_scale{};
  float down_scale{};
  std::size_t block_bytes{};
};

void validate_stream(const std::uint8_t* stream, const std::size_t records,
                     const std::size_t packed_width,
                     const std::size_t logical_width) {
  const std::size_t tail_digits = packed_width * kTritsPerByte - logical_width;
  for (std::size_t record = 0; record < records; ++record) {
    const auto* row = stream + record * packed_width;
    for (std::size_t index = 0; index < packed_width; ++index) {
      if (row[index] > 242U) {
        throw std::invalid_argument(
            "native BitNet base-3 byte is outside the canonical range");
      }
    }
    if (tail_digits != 0) {
      std::uint8_t value = row[packed_width - 1];
      const std::size_t used_digits = kTritsPerByte - tail_digits;
      for (std::size_t digit = 0; digit < used_digits; ++digit) value /= 3U;
      if (value != 0U) {
        throw std::invalid_argument(
            "native BitNet base-3 tail is not canonical");
      }
    }
  }
}

}  // namespace

class NativeBitNetKernel::Impl {
 public:
  Impl(const std::filesystem::path& artifact, const std::size_t thread_count)
      : pool_(thread_count) {
    if (thread_count == 0 || thread_count > 256) {
      throw std::invalid_argument("native BitNet thread count is invalid");
    }
    const int descriptor = ::open(artifact.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
      throw std::invalid_argument("cannot open native BitNet artifact");
    }
    struct stat status {};
    if (::fstat(descriptor, &status) != 0 || status.st_size <= 0 ||
        static_cast<std::uintmax_t>(status.st_size) >
            std::numeric_limits<std::size_t>::max()) {
      ::close(descriptor);
      throw std::invalid_argument("cannot stat native BitNet artifact");
    }
    mapping_size_ = static_cast<std::size_t>(status.st_size);
    mapping_ = ::mmap(nullptr, mapping_size_, PROT_READ, MAP_PRIVATE, descriptor,
                      0);
    const int saved_errno = errno;
    ::close(descriptor);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      mapping_size_ = 0;
      throw std::invalid_argument("cannot mmap native BitNet artifact: " +
                                  std::to_string(saved_errno));
    }
    try {
      parse();
    } catch (...) {
      release();
      throw;
    }
  }

  ~Impl() { release(); }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  void forward(const std::size_t layer_index,
               const std::span<const std::uint16_t> input,
               const std::size_t rows,
               const std::span<std::uint16_t> output,
               NativeBitNetMetrics* const metrics) {
    if (layer_index >= layers_.size() || rows == 0) {
      throw std::invalid_argument("native BitNet forward dimensions are invalid");
    }
    const std::size_t elements = checked_product(
        rows, hidden_size_, "native BitNet input dimensions overflow");
    if (input.size() != elements || output.size() != elements) {
      throw std::invalid_argument("native BitNet input/output size mismatch");
    }
    ensure_scratch(rows);
    const auto started = std::chrono::steady_clock::now();
    const LayerView& layer = layers_[layer_index];

    // Q8 input activation, retained as BF16-valued float32 and transposed so
    // the row dimension is contiguous for the gate/up stream pass.
    pool_.parallel_for(0, rows, 1, [&](const std::size_t row) {
      const std::size_t input_offset = row * hidden_size_;
      float maximum = 0.0F;
      for (std::size_t coordinate = 0; coordinate < hidden_size_; ++coordinate) {
        const float value = bf16_to_float(input[input_offset + coordinate]);
        if (!std::isfinite(value)) {
          throw std::invalid_argument("native BitNet input must be finite");
        }
        maximum = std::max(maximum, std::abs(value));
      }
      const float scale = 127.0F / std::max(maximum, 1.0e-5F);
      for (std::size_t coordinate = 0; coordinate < hidden_size_; ++coordinate) {
        const float value = bf16_to_float(input[input_offset + coordinate]);
        quantized_input_[coordinate * rows + row] =
            quantized_bf16(value, scale);
      }
    });

    // Gate and up are separate physical streams but share one execution pass.
    pool_.parallel_for(0, intermediate_size_, 8,
                       [&](const std::size_t record) {
      float* gate_accumulator = gate_accumulator_.data() + record * rows;
      float* up_accumulator = up_accumulator_.data() + record * rows;
      std::fill_n(gate_accumulator, rows, 0.0F);
      std::fill_n(up_accumulator, rows, 0.0F);
      const auto* gate_row = layer.gate + record * packed_width_;
      const auto* up_row = layer.up + record * packed_width_;
      for (std::size_t packed = 0; packed < packed_width_; ++packed) {
        std::uint8_t gate_byte = gate_row[packed];
        std::uint8_t up_byte = up_row[packed];
        for (std::size_t digit = 0; digit < kTritsPerByte; ++digit) {
          const std::size_t coordinate = packed * kTritsPerByte + digit;
          if (coordinate >= hidden_size_) break;
          const int gate_trit = static_cast<int>(gate_byte % 3U) - 1;
          const int up_trit = static_cast<int>(up_byte % 3U) - 1;
          gate_byte /= 3U;
          up_byte /= 3U;
          const float* values = quantized_input_.data() + coordinate * rows;
          if (gate_trit > 0) {
            for (std::size_t row = 0; row < rows; ++row)
              gate_accumulator[row] += values[row];
          } else if (gate_trit < 0) {
            for (std::size_t row = 0; row < rows; ++row)
              gate_accumulator[row] -= values[row];
          }
          if (up_trit > 0) {
            for (std::size_t row = 0; row < rows; ++row)
              up_accumulator[row] += values[row];
          } else if (up_trit < 0) {
            for (std::size_t row = 0; row < rows; ++row)
              up_accumulator[row] -= values[row];
          }
        }
      }
      for (std::size_t row = 0; row < rows; ++row) {
        const float gate = bf16_round(
            bf16_round(gate_accumulator[row]) * layer.gate_scale);
        const float up =
            bf16_round(bf16_round(up_accumulator[row]) * layer.up_scale);
        const float positive_gate = std::max(gate, 0.0F);
        const float squared_gate =
            bf16_round(positive_gate * positive_gate);
        activation_[row * intermediate_size_ + record] =
            bf16_round(squared_gate * up);
      }
    });

    // Shared RMS normalization forces this separate gain/Q8 phase.
    pool_.parallel_for(0, rows, 1, [&](const std::size_t row) {
      float* activation = activation_.data() + row * intermediate_size_;
      float sum = 0.0F;
      for (std::size_t record = 0; record < intermediate_size_; ++record) {
        sum += activation[record] * activation[record];
      }
      const float variance = sum / static_cast<float>(intermediate_size_);
      const float inverse_rms = 1.0F / std::sqrt(variance + rms_norm_eps_);
      float maximum = 0.0F;
      for (std::size_t record = 0; record < intermediate_size_; ++record) {
        const float gain = bf16_to_float(layer.norm[record]);
        const float normalized =
            bf16_round(bf16_round(activation[record] * inverse_rms) * gain);
        activation[record] = normalized;
        maximum = std::max(maximum, std::abs(normalized));
      }
      const float scale = 127.0F / std::max(maximum, 1.0e-5F);
      for (std::size_t record = 0; record < intermediate_size_; ++record) {
        quantized_normalized_[record * rows + row] =
            quantized_bf16(activation[record], scale);
      }
    });

    std::fill(down_accumulator_.begin(),
              down_accumulator_.begin() + elements, 0.0F);
    const std::size_t down_blocks =
        (packed_width_ + kDownPackedBlock - 1) / kDownPackedBlock;
    pool_.parallel_for(0, down_blocks, 1, [&](const std::size_t block) {
      const std::size_t packed_begin = block * kDownPackedBlock;
      const std::size_t packed_end =
          std::min(packed_width_, packed_begin + kDownPackedBlock);
      for (std::size_t record = 0; record < intermediate_size_; ++record) {
        const float* values = quantized_normalized_.data() + record * rows;
        const auto* down_row = layer.down + record * packed_width_;
        for (std::size_t packed = packed_begin; packed < packed_end; ++packed) {
          std::uint8_t down_byte = down_row[packed];
          for (std::size_t digit = 0; digit < kTritsPerByte; ++digit) {
            const std::size_t coordinate = packed * kTritsPerByte + digit;
            if (coordinate >= hidden_size_) break;
            const int trit = static_cast<int>(down_byte % 3U) - 1;
            down_byte /= 3U;
            float* accumulator = down_accumulator_.data() + coordinate * rows;
            if (trit > 0) {
              for (std::size_t row = 0; row < rows; ++row)
                accumulator[row] += values[row];
            } else if (trit < 0) {
              for (std::size_t row = 0; row < rows; ++row)
                accumulator[row] -= values[row];
            }
          }
        }
      }
      for (std::size_t packed = packed_begin; packed < packed_end; ++packed) {
        for (std::size_t digit = 0; digit < kTritsPerByte; ++digit) {
          const std::size_t coordinate = packed * kTritsPerByte + digit;
          if (coordinate >= hidden_size_) break;
          const float* accumulator =
              down_accumulator_.data() + coordinate * rows;
          for (std::size_t row = 0; row < rows; ++row) {
            const float value = bf16_round(
                bf16_round(accumulator[row]) * layer.down_scale);
            output[row * hidden_size_ + coordinate] =
                float_to_bf16_bits(value);
          }
        }
      }
    });

    if (metrics != nullptr) {
      *metrics = NativeBitNetMetrics{};
      metrics->elapsed_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started)
              .count());
      metrics->gate_up_stream_bytes = 2U * stream_bytes_;
      metrics->norm_stream_bytes = intermediate_size_ * sizeof(std::uint16_t);
      metrics->down_stream_bytes = stream_bytes_;
      metrics->layer_metadata_bytes = metadata_bytes_;
      metrics->scheduled_cache_line_bytes = layer.block_bytes;
      metrics->scratch_bytes = scratch_bytes();
      metrics->rows = rows;
      metrics->threads = pool_.thread_count();
    }
  }

  [[nodiscard]] std::size_t layer_count() const noexcept {
    return layers_.size();
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
  [[nodiscard]] std::size_t artifact_bytes() const noexcept {
    return mapping_size_;
  }

 private:
  void release() noexcept {
    if (mapping_ != nullptr) {
      ::munmap(mapping_, mapping_size_);
      mapping_ = nullptr;
      mapping_size_ = 0;
    }
  }

  void parse() {
    if (mapping_size_ < kHeaderBytes) {
      throw std::invalid_argument("native BitNet artifact is truncated");
    }
    const auto* bytes = static_cast<const std::byte*>(mapping_);
    if (std::memcmp(bytes, kMagic, sizeof(kMagic)) != 0 ||
        little_u32(bytes + 8) != kVersion) {
      throw std::invalid_argument("native BitNet artifact magic/version mismatch");
    }
    const std::size_t layer_count = little_u32(bytes + 12);
    hidden_size_ = little_u32(bytes + 16);
    intermediate_size_ = little_u32(bytes + 20);
    const std::size_t cache_line = little_u32(bytes + 24);
    const std::size_t directory_entry = little_u32(bytes + 28);
    const std::size_t directory_block = little_u32(bytes + 32);
    const std::size_t record_payload = little_u32(bytes + 36);
    rms_norm_eps_ = little_f32(bytes + 40);
    if (layer_count == 0 || layer_count > kMaximumLayers || hidden_size_ == 0 ||
        hidden_size_ > kMaximumDimension || intermediate_size_ == 0 ||
        intermediate_size_ > kMaximumDimension || cache_line != kCacheLineBytes ||
        directory_entry != kDirectoryEntryBytes ||
        !std::isfinite(rms_norm_eps_) || rms_norm_eps_ <= 0.0F) {
      throw std::invalid_argument("native BitNet artifact header is invalid");
    }
    packed_width_ = (hidden_size_ + kTritsPerByte - 1) / kTritsPerByte;
    if (record_payload != 3U * packed_width_ + sizeof(std::uint16_t)) {
      throw std::invalid_argument("native BitNet logical record size is invalid");
    }
    const std::size_t directory_payload =
        checked_product(layer_count, kDirectoryEntryBytes,
                        "native BitNet directory overflows");
    if (directory_block != align_up(directory_payload, kCacheLineBytes)) {
      throw std::invalid_argument("native BitNet directory size is invalid");
    }
    metadata_bytes_ = align_up(kLayerHeaderBytes + kProjectionScaleBytes,
                               kCacheLineBytes);
    stream_bytes_ = checked_product(intermediate_size_, packed_width_,
                                    "native BitNet stream size overflows");
    gate_offset_ = metadata_bytes_;
    up_offset_ = align_up(checked_sum(gate_offset_, stream_bytes_,
                                      "native BitNet gate end overflows"),
                          kCacheLineBytes);
    norm_offset_ = align_up(checked_sum(up_offset_, stream_bytes_,
                                        "native BitNet up end overflows"),
                            kCacheLineBytes);
    const std::size_t norm_bytes = checked_product(
        intermediate_size_, sizeof(std::uint16_t),
        "native BitNet normalization stream overflows");
    down_offset_ = align_up(checked_sum(norm_offset_, norm_bytes,
                                        "native BitNet norm end overflows"),
                            kCacheLineBytes);
    const std::size_t layer_payload = checked_sum(
        down_offset_, stream_bytes_, "native BitNet layer payload overflows");
    const std::size_t layer_block = align_up(layer_payload, kCacheLineBytes);
    const std::size_t expected_size = checked_sum(
        checked_sum(kHeaderBytes, directory_block,
                    "native BitNet preamble overflows"),
        checked_product(layer_count, layer_block,
                        "native BitNet artifact size overflows"),
        "native BitNet artifact size overflows");
    if (mapping_size_ != expected_size) {
      throw std::invalid_argument("native BitNet artifact length mismatch");
    }
    require_zero(bytes, 44, kHeaderBytes,
                 "native BitNet header padding is nonzero");
    require_zero(bytes, kHeaderBytes + directory_payload,
                 kHeaderBytes + directory_block,
                 "native BitNet directory padding is nonzero");

    layers_.reserve(layer_count);
    std::size_t expected_offset = kHeaderBytes + directory_block;
    for (std::size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
      const std::byte* entry =
          bytes + kHeaderBytes + layer_index * kDirectoryEntryBytes;
      const std::size_t entry_index = little_u32(entry);
      const std::size_t reserved = little_u32(entry + 4);
      const std::size_t offset = little_u64(entry + 8);
      const std::size_t block_bytes = little_u64(entry + 16);
      const std::size_t payload_bytes = little_u32(entry + 24);
      const std::size_t reserved_2 = little_u32(entry + 28);
      if (entry_index != layer_index || reserved != 0 || reserved_2 != 0 ||
          offset != expected_offset || offset % kCacheLineBytes != 0 ||
          block_bytes != layer_block || payload_bytes != layer_payload) {
        throw std::invalid_argument("native BitNet directory entry is invalid");
      }
      const std::byte* header = bytes + offset;
      if (std::memcmp(header, kLayerMagic, sizeof(kLayerMagic)) != 0 ||
          little_u32(header + 8) != kVersion ||
          little_u32(header + 12) != layer_index ||
          little_u32(header + 16) != hidden_size_ ||
          little_u32(header + 20) != intermediate_size_ ||
          little_u32(header + 24) != intermediate_size_ ||
          little_u32(header + 28) != packed_width_ ||
          little_u32(header + 32) != record_payload ||
          little_u32(header + 36) != layer_payload ||
          little_u32(header + 40) != 3U || little_u32(header + 44) != 0U) {
        throw std::invalid_argument("native BitNet layer header is invalid");
      }
      require_zero(bytes, offset + 48, offset + kLayerHeaderBytes,
                   "native BitNet layer-header padding is nonzero");
      const std::uint16_t gate_scale_bits = little_u16(bytes + offset + 64);
      const std::uint16_t up_scale_bits = little_u16(bytes + offset + 66);
      const std::uint16_t down_scale_bits = little_u16(bytes + offset + 68);
      const float gate_scale = bf16_to_float(gate_scale_bits);
      const float up_scale = bf16_to_float(up_scale_bits);
      const float down_scale = bf16_to_float(down_scale_bits);
      if (!std::isfinite(gate_scale) || gate_scale <= 0.0F ||
          !std::isfinite(up_scale) || up_scale <= 0.0F ||
          !std::isfinite(down_scale) || down_scale <= 0.0F) {
        throw std::invalid_argument("native BitNet projection scale is invalid");
      }
      require_zero(bytes, offset + 70, offset + gate_offset_,
                   "native BitNet metadata padding is nonzero");
      const auto* gate = reinterpret_cast<const std::uint8_t*>(
          bytes + offset + gate_offset_);
      const auto* up = reinterpret_cast<const std::uint8_t*>(
          bytes + offset + up_offset_);
      const auto* norm = reinterpret_cast<const std::uint16_t*>(
          bytes + offset + norm_offset_);
      const auto* down = reinterpret_cast<const std::uint8_t*>(
          bytes + offset + down_offset_);
      require_zero(bytes, offset + gate_offset_ + stream_bytes_,
                   offset + up_offset_,
                   "native BitNet gate-stream padding is nonzero");
      require_zero(bytes, offset + up_offset_ + stream_bytes_,
                   offset + norm_offset_,
                   "native BitNet up-stream padding is nonzero");
      require_zero(bytes, offset + norm_offset_ + norm_bytes,
                   offset + down_offset_,
                   "native BitNet norm-stream padding is nonzero");
      require_zero(bytes, offset + layer_payload, offset + layer_block,
                   "native BitNet layer padding is nonzero");
      validate_stream(gate, intermediate_size_, packed_width_, hidden_size_);
      validate_stream(up, intermediate_size_, packed_width_, hidden_size_);
      validate_stream(down, intermediate_size_, packed_width_, hidden_size_);
      for (std::size_t record = 0; record < intermediate_size_; ++record) {
        if (!std::isfinite(bf16_to_float(norm[record]))) {
          throw std::invalid_argument(
              "native BitNet normalization gain is invalid");
        }
      }
      layers_.push_back({gate, up, norm, down, gate_scale, up_scale,
                         down_scale, block_bytes});
      expected_offset += block_bytes;
    }
  }

  void ensure_scratch(const std::size_t rows) {
    const std::size_t hidden_rows = checked_product(
        hidden_size_, rows, "native BitNet scratch size overflows");
    const std::size_t intermediate_rows = checked_product(
        intermediate_size_, rows, "native BitNet scratch size overflows");
    quantized_input_.resize(hidden_rows);
    gate_accumulator_.resize(intermediate_rows);
    up_accumulator_.resize(intermediate_rows);
    activation_.resize(intermediate_rows);
    quantized_normalized_.resize(intermediate_rows);
    down_accumulator_.resize(hidden_rows);
  }

  [[nodiscard]] std::size_t scratch_bytes() const noexcept {
    return sizeof(float) *
           (quantized_input_.capacity() + gate_accumulator_.capacity() +
            up_accumulator_.capacity() + activation_.capacity() +
            quantized_normalized_.capacity() + down_accumulator_.capacity());
  }

  void* mapping_{};
  std::size_t mapping_size_{};
  std::size_t hidden_size_{};
  std::size_t intermediate_size_{};
  std::size_t packed_width_{};
  std::size_t metadata_bytes_{};
  std::size_t stream_bytes_{};
  std::size_t gate_offset_{};
  std::size_t up_offset_{};
  std::size_t norm_offset_{};
  std::size_t down_offset_{};
  float rms_norm_eps_{};
  std::vector<LayerView> layers_;
  ThreadPool pool_;
  std::vector<float> quantized_input_;
  std::vector<float> gate_accumulator_;
  std::vector<float> up_accumulator_;
  std::vector<float> activation_;
  std::vector<float> quantized_normalized_;
  std::vector<float> down_accumulator_;
};

NativeBitNetKernel::NativeBitNetKernel(const std::filesystem::path& artifact,
                                       const std::size_t thread_count)
    : impl_(std::make_unique<Impl>(artifact, thread_count)) {}

NativeBitNetKernel::~NativeBitNetKernel() = default;
NativeBitNetKernel::NativeBitNetKernel(NativeBitNetKernel&&) noexcept = default;
NativeBitNetKernel& NativeBitNetKernel::operator=(
    NativeBitNetKernel&&) noexcept = default;

std::size_t NativeBitNetKernel::layer_count() const noexcept {
  return impl_->layer_count();
}
std::size_t NativeBitNetKernel::hidden_size() const noexcept {
  return impl_->hidden_size();
}
std::size_t NativeBitNetKernel::intermediate_size() const noexcept {
  return impl_->intermediate_size();
}
std::size_t NativeBitNetKernel::thread_count() const noexcept {
  return impl_->thread_count();
}
std::size_t NativeBitNetKernel::serialized_artifact_bytes() const noexcept {
  return impl_->artifact_bytes();
}

void NativeBitNetKernel::forward_bf16(
    const std::size_t layer, const std::span<const std::uint16_t> input,
    const std::size_t rows, const std::span<std::uint16_t> output,
    NativeBitNetMetrics* const metrics) {
  impl_->forward(layer, input, rows, output, metrics);
}

}  // namespace engram
