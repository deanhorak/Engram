#include "engram/olmoe_q7.h"

#include "engram/thread_pool.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

constexpr char kMagic[8] = {'E', 'N', 'G', 'O', 'Q', '7', '1', '1'};
constexpr char kLayerMagic[8] = {'E', 'N', 'G', 'O', 'Q', '7', 'L', '1'};
constexpr char kExpertMagic[8] = {'E', 'N', 'G', 'O', 'Q', '7', 'E', '1'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kEndianMarker = 0x01020304U;
constexpr std::size_t kHeaderBytes = 128;
constexpr std::size_t kDirectoryEntryBytes = 64;
constexpr std::size_t kLayerHeaderBytes = 64;
constexpr std::size_t kExpertHeaderBytes = 64;
constexpr std::size_t kCacheLineBytes = 64;
constexpr std::size_t kBits = 7;
constexpr std::size_t kCodeBias = 63;
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

float bf16_to_float(const std::uint16_t bits) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

std::size_t align_up(const std::size_t value,
                     const std::size_t alignment = kCacheLineBytes) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0 ||
      value > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
    throw std::invalid_argument("OLMoE Q7 alignment overflow");
  }
  return (value + alignment - 1) & ~(alignment - 1);
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

bool equal_magic(const std::byte* bytes, const char (&magic)[8]) noexcept {
  for (std::size_t index = 0; index < 8; ++index) {
    if (bytes[index] != static_cast<std::byte>(magic[index])) return false;
  }
  return true;
}

void require_zero(const std::byte* bytes, const std::size_t begin,
                  const std::size_t end, const char* message) {
  for (std::size_t index = begin; index < end; ++index) {
    if (bytes[index] != std::byte{0}) throw std::invalid_argument(message);
  }
}

std::size_t validated_thread_count(const std::size_t value) {
  if (value == 0 || value > 256) {
    throw std::invalid_argument("OLMoE Q7 thread count is invalid");
  }
  return value;
}

struct MatrixLayout {
  std::size_t rows{};
  std::size_t columns{};
  std::size_t code_bytes{};
  std::size_t scale_bytes{};
  std::size_t block_bytes{};
};

struct Layout {
  std::size_t layers{};
  std::size_t hidden{};
  std::size_t intermediate{};
  std::size_t experts{};
  std::size_t top_k{};
  std::size_t group{};
  std::size_t router_bytes{};
  MatrixLayout gate{};
  MatrixLayout down{};
  std::size_t expert_payload_bytes{};
  std::size_t expert_stride{};
  std::size_t experts_offset{};
  std::size_t layer_payload_bytes{};
  std::size_t layer_block_bytes{};
  std::size_t directory_bytes{};
  std::size_t file_bytes{};
};

MatrixLayout matrix_layout(const std::size_t rows, const std::size_t columns,
                           const std::size_t group) {
  const std::size_t elements =
      checked_product(rows, columns, "OLMoE Q7 matrix dimensions overflow");
  const std::size_t code_bytes =
      checked_sum(checked_product(elements, kBits,
                                  "OLMoE Q7 packed codes overflow"),
                  7, "OLMoE Q7 packed codes overflow") /
      8;
  const std::size_t groups = checked_sum(columns, group - 1,
                                         "OLMoE Q7 scale groups overflow") /
                             group;
  const std::size_t scale_bytes = checked_product(
      checked_product(rows, groups, "OLMoE Q7 scales overflow"), 2,
      "OLMoE Q7 scales overflow");
  return MatrixLayout{rows, columns, code_bytes, scale_bytes,
                      checked_sum(align_up(code_bytes), align_up(scale_bytes),
                                  "OLMoE Q7 matrix block overflow")};
}

Layout make_layout(const std::size_t layers, const std::size_t hidden,
                   const std::size_t intermediate,
                   const std::size_t experts, const std::size_t top_k,
                   const std::size_t group) {
  if (layers == 0 || layers > kMaximumLayers || hidden == 0 ||
      hidden > kMaximumDimension || intermediate == 0 ||
      intermediate > kMaximumDimension || experts == 0 ||
      experts > kMaximumDimension || top_k == 0 || top_k > experts ||
      group == 0 || group > kMaximumDimension) {
    throw std::invalid_argument("OLMoE Q7 dimensions are invalid");
  }
  Layout result{};
  result.layers = layers;
  result.hidden = hidden;
  result.intermediate = intermediate;
  result.experts = experts;
  result.top_k = top_k;
  result.group = group;
  result.router_bytes = checked_product(
      checked_product(experts, hidden, "OLMoE Q7 router overflow"), 2,
      "OLMoE Q7 router overflow");
  result.gate = matrix_layout(intermediate, hidden, group);
  result.down = matrix_layout(hidden, intermediate, group);
  result.expert_payload_bytes = checked_sum(
      kExpertHeaderBytes,
      checked_sum(checked_product(2, result.gate.block_bytes,
                                  "OLMoE Q7 expert block overflow"),
                  result.down.block_bytes, "OLMoE Q7 expert block overflow"),
      "OLMoE Q7 expert block overflow");
  result.expert_stride = align_up(result.expert_payload_bytes);
  result.experts_offset =
      align_up(checked_sum(kLayerHeaderBytes, result.router_bytes,
                           "OLMoE Q7 layer block overflow"));
  result.layer_payload_bytes = checked_sum(
      result.experts_offset,
      checked_product(experts, result.expert_stride,
                      "OLMoE Q7 layer block overflow"),
      "OLMoE Q7 layer block overflow");
  result.layer_block_bytes = align_up(result.layer_payload_bytes);
  result.directory_bytes =
      align_up(checked_product(layers, kDirectoryEntryBytes,
                               "OLMoE Q7 directory overflow"));
  result.file_bytes = checked_sum(
      checked_sum(kHeaderBytes, result.directory_bytes,
                  "OLMoE Q7 file size overflow"),
      checked_product(layers, result.layer_block_bytes,
                      "OLMoE Q7 file size overflow"),
      "OLMoE Q7 file size overflow");
  return result;
}

struct MatrixView {
  const std::byte* codes{};
  const std::byte* scales{};
  MatrixLayout layout{};
};

struct ExpertView {
  MatrixView gate{};
  MatrixView up{};
  MatrixView down{};
};

struct LayerView {
  const std::byte* router{};
  std::vector<ExpertView> experts{};
};

std::uint8_t q7_unsigned_at(const std::byte* codes,
                            const std::size_t index) noexcept {
  const std::size_t bit = index * kBits;
  const std::size_t byte = bit / 8;
  const unsigned shift = static_cast<unsigned>(bit % 8);
  std::uint16_t value =
      static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(codes[byte])) >>
      shift;
  if (shift > 1) {
    value |= static_cast<std::uint16_t>(
                 std::to_integer<std::uint8_t>(codes[byte + 1]))
             << (8U - shift);
  }
  return static_cast<std::uint8_t>(value & 0x7FU);
}

}  // namespace

class OLMoEQ7Kernel::Impl {
 public:
  Impl(const std::filesystem::path& path, const std::size_t thread_count)
      : pool_(validated_thread_count(thread_count)) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
      throw std::invalid_argument("cannot open OLMoE Q7 artifact");
    }
    struct stat status {};
    if (::fstat(descriptor, &status) != 0 || status.st_size <= 0 ||
        static_cast<std::uintmax_t>(status.st_size) >
            std::numeric_limits<std::size_t>::max()) {
      ::close(descriptor);
      throw std::invalid_argument("cannot stat OLMoE Q7 artifact");
    }
    mapping_size_ = static_cast<std::size_t>(status.st_size);
    mapping_ =
        ::mmap(nullptr, mapping_size_, PROT_READ, MAP_PRIVATE, descriptor, 0);
    const int saved_errno = errno;
    ::close(descriptor);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      mapping_size_ = 0;
      throw std::invalid_argument("cannot mmap OLMoE Q7 artifact: " +
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

  void forward(const std::size_t layer_index,
               const std::span<const float> input, const std::size_t rows,
               const std::span<float> output,
               const std::span<std::uint32_t> selected,
               OLMoEQ7Metrics* const metrics) {
    if (layer_index >= layer_views_.size() || rows == 0) {
      throw std::invalid_argument("OLMoE Q7 forward dimensions are invalid");
    }
    const std::size_t elements =
        checked_product(rows, layout_.hidden, "OLMoE Q7 rows overflow");
    if (input.size() != elements || output.size() != elements ||
        (!selected.empty() &&
         selected.size() !=
             checked_product(rows, layout_.top_k,
                             "OLMoE Q7 selected output overflow"))) {
      throw std::invalid_argument("OLMoE Q7 input/output size mismatch");
    }
    for (const float value : input) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("OLMoE Q7 input must be finite");
      }
    }
    const auto started = std::chrono::steady_clock::now();
    const LayerView& layer = layer_views_[layer_index];
    if (rows == 1 && pool_.thread_count() > 1 && layout_.top_k > 1) {
      forward_single_parallel(layer, input.data(), output.data(),
                              selected.empty() ? nullptr : selected.data());
    } else {
      pool_.parallel_for(0, rows, 1, [&](const std::size_t row) {
        forward_row_serial(
            layer, input.data() + row * layout_.hidden,
            output.data() + row * layout_.hidden,
            selected.empty() ? nullptr
                             : selected.data() + row * layout_.top_k);
      });
    }
    if (metrics != nullptr) {
      const std::size_t selected_payload =
          2 * (layout_.gate.code_bytes + layout_.gate.scale_bytes) +
          layout_.down.code_bytes + layout_.down.scale_bytes;
      *metrics = OLMoEQ7Metrics{};
      metrics->elapsed_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started)
              .count());
      metrics->router_stream_bytes = rows * layout_.router_bytes;
      metrics->selected_expert_stream_bytes =
          rows * layout_.top_k * selected_payload;
      metrics->scheduled_stream_bytes =
          metrics->router_stream_bytes + metrics->selected_expert_stream_bytes;
      metrics->scratch_bytes =
          rows == 1 && pool_.thread_count() > 1 && layout_.top_k > 1
              ? (layout_.experts * (sizeof(float) + sizeof(std::size_t)) +
                 layout_.top_k *
                     (3 * layout_.intermediate + layout_.hidden) *
                     sizeof(float))
              : pool_.thread_count() *
                    (layout_.experts * (sizeof(float) + sizeof(std::size_t)) +
                     (3 * layout_.intermediate + layout_.hidden) *
                         sizeof(float));
      metrics->rows = rows;
      metrics->threads = pool_.thread_count();
      metrics->selected_experts = rows * layout_.top_k;
    }
  }

  [[nodiscard]] const Layout& layout() const noexcept { return layout_; }
  [[nodiscard]] std::size_t threads() const noexcept {
    return pool_.thread_count();
  }
  [[nodiscard]] std::size_t artifact_bytes() const noexcept {
    return mapping_size_;
  }

 private:
  void route(const LayerView& layer, const float* state,
             std::vector<float>& logits,
             std::vector<std::size_t>& order) const {
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t expert = 0; expert < layout_.experts; ++expert) {
      float value = 0.0F;
      const std::byte* router =
          layer.router + expert * layout_.hidden * sizeof(std::uint16_t);
      for (std::size_t coordinate = 0; coordinate < layout_.hidden;
           ++coordinate) {
        value += state[coordinate] *
                 bf16_to_float(little_u16(router + coordinate * 2));
      }
      logits[expert] = value;
      maximum = std::max(maximum, value);
    }
    float denominator = 0.0F;
    for (float& value : logits) {
      value = std::exp(value - maximum);
      denominator += value;
    }
    for (float& value : logits) value /= denominator;
    std::iota(order.begin(), order.end(), 0);
    const auto stronger = [&](const std::size_t left,
                              const std::size_t right) {
      return logits[left] == logits[right] ? left < right
                                           : logits[left] > logits[right];
    };
    std::partial_sort(order.begin(), order.begin() + layout_.top_k,
                      order.end(), stronger);
  }

  void evaluate_expert(const ExpertView& expert, const float* state,
                       float* gate, float* up, float* activation,
                       float* expert_output) const {
    project(expert.gate, state, gate);
    project(expert.up, state, up);
    for (std::size_t coordinate = 0; coordinate < layout_.intermediate;
         ++coordinate) {
      const float gate_value = gate[coordinate];
      activation[coordinate] =
          (gate_value / (1.0F + std::exp(-gate_value))) * up[coordinate];
    }
    project(expert.down, activation, expert_output);
  }

  void forward_row_serial(const LayerView& layer, const float* state,
                          float* destination,
                          std::uint32_t* selected) const {
    std::fill_n(destination, layout_.hidden, 0.0F);
    std::vector<float> logits(layout_.experts);
    std::vector<std::size_t> order(layout_.experts);
    route(layer, state, logits, order);
    std::vector<float> gate(layout_.intermediate);
    std::vector<float> up(layout_.intermediate);
    std::vector<float> activation(layout_.intermediate);
    std::vector<float> expert_output(layout_.hidden);
    for (std::size_t rank = 0; rank < layout_.top_k; ++rank) {
      const std::size_t expert_index = order[rank];
      if (selected != nullptr) {
        selected[rank] = static_cast<std::uint32_t>(expert_index);
      }
      evaluate_expert(layer.experts[expert_index], state, gate.data(),
                      up.data(), activation.data(), expert_output.data());
      const float weight = logits[expert_index];
      for (std::size_t coordinate = 0; coordinate < layout_.hidden;
           ++coordinate) {
        destination[coordinate] += weight * expert_output[coordinate];
      }
    }
  }

  void forward_single_parallel(const LayerView& layer, const float* state,
                               float* destination,
                               std::uint32_t* selected) {
    std::vector<float> logits(layout_.experts);
    std::vector<std::size_t> order(layout_.experts);
    route(layer, state, logits, order);
    std::vector<float> gate(layout_.top_k * layout_.intermediate);
    std::vector<float> up(layout_.top_k * layout_.intermediate);
    std::vector<float> activation(layout_.top_k * layout_.intermediate);
    std::vector<float> expert_output(layout_.top_k * layout_.hidden);
    pool_.parallel_for(0, layout_.top_k, 1, [&](const std::size_t rank) {
      const std::size_t expert_index = order[rank];
      evaluate_expert(
          layer.experts[expert_index], state,
          gate.data() + rank * layout_.intermediate,
          up.data() + rank * layout_.intermediate,
          activation.data() + rank * layout_.intermediate,
          expert_output.data() + rank * layout_.hidden);
    });
    std::fill_n(destination, layout_.hidden, 0.0F);
    for (std::size_t rank = 0; rank < layout_.top_k; ++rank) {
      const std::size_t expert_index = order[rank];
      if (selected != nullptr) {
        selected[rank] = static_cast<std::uint32_t>(expert_index);
      }
      const float weight = logits[expert_index];
      const float* source = expert_output.data() + rank * layout_.hidden;
      for (std::size_t coordinate = 0; coordinate < layout_.hidden;
           ++coordinate) {
        destination[coordinate] += weight * source[coordinate];
      }
    }
  }

  void project(const MatrixView& matrix, const float* input,
               float* output) const {
    const std::size_t groups =
        (matrix.layout.columns + layout_.group - 1) / layout_.group;
    if (matrix.layout.columns % layout_.group == 0 &&
        layout_.group % 8 == 0) {
      const std::size_t row_code_bytes =
          matrix.layout.columns * kBits / 8;
      const std::size_t group_code_bytes = layout_.group * kBits / 8;
      for (std::size_t row = 0; row < matrix.layout.rows; ++row) {
        float sum = 0.0F;
        const std::byte* row_codes =
            matrix.codes + row * row_code_bytes;
        for (std::size_t group = 0; group < groups; ++group) {
          const float scale = bf16_to_float(
              little_u16(matrix.scales + (row * groups + group) * 2));
          const std::byte* group_codes =
              row_codes + group * group_code_bytes;
          const std::size_t group_column = group * layout_.group;
          for (std::size_t block = 0; block < layout_.group / 8;
               ++block) {
            const std::byte* packed = group_codes + block * 7;
            std::uint64_t word = 0;
            for (std::size_t byte = 0; byte < 7; ++byte) {
              word |= static_cast<std::uint64_t>(
                          std::to_integer<unsigned char>(packed[byte]))
                      << (byte * 8);
            }
            const std::size_t block_column =
                group_column + block * 8;
            for (std::size_t lane = 0; lane < 8; ++lane) {
              const int code =
                  static_cast<int>((word >> (lane * 7)) & 0x7FU) -
                  static_cast<int>(kCodeBias);
              const std::size_t column = block_column + lane;
              sum += input[column] * static_cast<float>(code) * scale;
            }
          }
        }
        output[row] = sum;
      }
      return;
    }
    for (std::size_t row = 0; row < matrix.layout.rows; ++row) {
      float sum = 0.0F;
      for (std::size_t column = 0; column < matrix.layout.columns; ++column) {
        const std::size_t flat = row * matrix.layout.columns + column;
        const std::uint8_t encoded = q7_unsigned_at(matrix.codes, flat);
        const int code = static_cast<int>(encoded) -
                         static_cast<int>(kCodeBias);
        const std::size_t scale_index =
            row * groups + column / layout_.group;
        const float scale =
            bf16_to_float(little_u16(matrix.scales + scale_index * 2));
        sum += input[column] * static_cast<float>(code) * scale;
      }
      output[row] = sum;
    }
  }

  void parse() {
    const auto* bytes = static_cast<const std::byte*>(mapping_);
    if (mapping_size_ < kHeaderBytes || !equal_magic(bytes, kMagic) ||
        little_u32(bytes + 8) != kVersion ||
        little_u32(bytes + 12) != kEndianMarker ||
        little_u32(bytes + 16) != kHeaderBytes ||
        little_u32(bytes + 20) != kDirectoryEntryBytes ||
        little_u32(bytes + 24) != kLayerHeaderBytes ||
        little_u32(bytes + 28) != kExpertHeaderBytes ||
        little_u32(bytes + 32) != kCacheLineBytes ||
        little_u32(bytes + 36) != kBits ||
        little_u32(bytes + 40) != kCodeBias ||
        little_u32(bytes + 68) != 0 || little_u64(bytes + 96) != 0) {
      throw std::invalid_argument("OLMoE Q7 header contract is unsupported");
    }
    layout_ = make_layout(
        little_u32(bytes + 48), little_u32(bytes + 52),
        little_u32(bytes + 56), little_u32(bytes + 60),
        little_u32(bytes + 64), little_u32(bytes + 44));
    if (little_u64(bytes + 72) != kHeaderBytes ||
        little_u64(bytes + 80) != layout_.directory_bytes ||
        little_u64(bytes + 88) != layout_.file_bytes ||
        mapping_size_ != layout_.file_bytes) {
      throw std::invalid_argument("OLMoE Q7 header sizes are invalid");
    }
    require_zero(bytes, 104, kHeaderBytes,
                 "OLMoE Q7 header padding is non-zero");
    require_zero(bytes, kHeaderBytes + layout_.layers * kDirectoryEntryBytes,
                 kHeaderBytes + layout_.directory_bytes,
                 "OLMoE Q7 directory padding is non-zero");
    layer_views_.resize(layout_.layers);
    pool_.parallel_for(0, layout_.layers, 1, [&](const std::size_t layer) {
      const std::size_t expected_offset =
          kHeaderBytes + layout_.directory_bytes +
          layer * layout_.layer_block_bytes;
      const std::byte* entry =
          bytes + kHeaderBytes + layer * kDirectoryEntryBytes;
      if (little_u32(entry) != layer || little_u32(entry + 4) != 0 ||
          little_u64(entry + 8) != expected_offset ||
          little_u64(entry + 16) != layout_.layer_block_bytes ||
          little_u64(entry + 24) != kLayerHeaderBytes ||
          little_u64(entry + 32) != layout_.router_bytes ||
          little_u64(entry + 40) != layout_.experts_offset ||
          little_u64(entry + 48) != layout_.expert_stride ||
          little_u64(entry + 56) != layout_.layer_payload_bytes) {
        throw std::invalid_argument("OLMoE Q7 directory entry is invalid");
      }
      layer_views_[layer] = parse_layer(bytes + expected_offset, layer);
    });
  }

  LayerView parse_layer(const std::byte* layer, const std::size_t index) {
    if (!equal_magic(layer, kLayerMagic) ||
        little_u32(layer + 8) != kVersion ||
        little_u32(layer + 12) != index ||
        little_u32(layer + 16) != layout_.hidden ||
        little_u32(layer + 20) != layout_.intermediate ||
        little_u32(layer + 24) != layout_.experts ||
        little_u32(layer + 28) != layout_.top_k ||
        little_u32(layer + 32) != layout_.group ||
        little_u32(layer + 36) != kBits ||
        little_u64(layer + 40) != kLayerHeaderBytes ||
        little_u64(layer + 48) != layout_.experts_offset ||
        little_u64(layer + 56) != layout_.layer_block_bytes) {
      throw std::invalid_argument("OLMoE Q7 layer header is invalid");
    }
    const std::byte* router = layer + kLayerHeaderBytes;
    for (std::size_t value = 0; value < layout_.router_bytes / 2; ++value) {
      if (!std::isfinite(bf16_to_float(little_u16(router + value * 2)))) {
        throw std::invalid_argument("OLMoE Q7 router contains non-finite values");
      }
    }
    require_zero(layer, kLayerHeaderBytes + layout_.router_bytes,
                 layout_.experts_offset,
                 "OLMoE Q7 router padding is non-zero");
    LayerView result{};
    result.router = router;
    result.experts.reserve(layout_.experts);
    for (std::size_t expert = 0; expert < layout_.experts; ++expert) {
      result.experts.push_back(parse_expert(
          layer + layout_.experts_offset + expert * layout_.expert_stride,
          expert));
    }
    require_zero(layer, layout_.layer_payload_bytes, layout_.layer_block_bytes,
                 "OLMoE Q7 layer padding is non-zero");
    return result;
  }

  ExpertView parse_expert(const std::byte* expert,
                          const std::size_t index) const {
    const std::size_t gate_offset = kExpertHeaderBytes;
    const std::size_t up_offset = gate_offset + layout_.gate.block_bytes;
    const std::size_t down_offset = up_offset + layout_.gate.block_bytes;
    if (!equal_magic(expert, kExpertMagic) ||
        little_u32(expert + 8) != kVersion ||
        little_u32(expert + 12) != index ||
        little_u32(expert + 16) != layout_.intermediate ||
        little_u32(expert + 20) != layout_.hidden ||
        little_u32(expert + 24) != layout_.hidden ||
        little_u32(expert + 28) != layout_.intermediate ||
        little_u64(expert + 32) != gate_offset ||
        little_u64(expert + 40) != up_offset ||
        little_u64(expert + 48) != down_offset ||
        little_u64(expert + 56) != layout_.expert_stride) {
      throw std::invalid_argument("OLMoE Q7 expert header is invalid");
    }
    ExpertView result{};
    result.gate = parse_matrix(expert + gate_offset, layout_.gate);
    result.up = parse_matrix(expert + up_offset, layout_.gate);
    result.down = parse_matrix(expert + down_offset, layout_.down);
    require_zero(expert, layout_.expert_payload_bytes, layout_.expert_stride,
                 "OLMoE Q7 expert padding is non-zero");
    return result;
  }

  MatrixView parse_matrix(const std::byte* matrix,
                          const MatrixLayout& layout) const {
    const std::size_t elements = layout.rows * layout.columns;
    for (std::size_t index = 0; index < elements; ++index) {
      if (q7_unsigned_at(matrix, index) > 126U) {
        throw std::invalid_argument("OLMoE Q7 stream contains reserved code");
      }
    }
    const std::size_t used_tail = elements * kBits % 8;
    if (used_tail != 0 &&
        (std::to_integer<std::uint8_t>(matrix[layout.code_bytes - 1]) >>
         used_tail) != 0) {
      throw std::invalid_argument("OLMoE Q7 packed tail is not canonical");
    }
    const std::byte* scales = matrix + align_up(layout.code_bytes);
    const std::size_t scale_count = layout.scale_bytes / 2;
    for (std::size_t index = 0; index < scale_count; ++index) {
      const float scale = bf16_to_float(little_u16(scales + index * 2));
      if (!std::isfinite(scale) || scale <= 0.0F) {
        throw std::invalid_argument("OLMoE Q7 scale is invalid");
      }
    }
    require_zero(matrix, layout.code_bytes, align_up(layout.code_bytes),
                 "OLMoE Q7 code padding is non-zero");
    require_zero(scales, layout.scale_bytes, align_up(layout.scale_bytes),
                 "OLMoE Q7 scale padding is non-zero");
    return MatrixView{matrix, scales, layout};
  }

  void release() noexcept {
    if (mapping_ != nullptr) {
      ::munmap(mapping_, mapping_size_);
      mapping_ = nullptr;
      mapping_size_ = 0;
    }
  }

  void* mapping_{};
  std::size_t mapping_size_{};
  Layout layout_{};
  std::vector<LayerView> layer_views_{};
  ThreadPool pool_;
};

OLMoEQ7Kernel::OLMoEQ7Kernel(const std::filesystem::path& artifact,
                             const std::size_t thread_count)
    : impl_(std::make_unique<Impl>(artifact, thread_count)) {}

OLMoEQ7Kernel::~OLMoEQ7Kernel() = default;
OLMoEQ7Kernel::OLMoEQ7Kernel(OLMoEQ7Kernel&&) noexcept = default;
OLMoEQ7Kernel& OLMoEQ7Kernel::operator=(OLMoEQ7Kernel&&) noexcept = default;

std::size_t OLMoEQ7Kernel::layer_count() const noexcept {
  return impl_->layout().layers;
}
std::size_t OLMoEQ7Kernel::hidden_size() const noexcept {
  return impl_->layout().hidden;
}
std::size_t OLMoEQ7Kernel::intermediate_size() const noexcept {
  return impl_->layout().intermediate;
}
std::size_t OLMoEQ7Kernel::expert_count() const noexcept {
  return impl_->layout().experts;
}
std::size_t OLMoEQ7Kernel::top_k() const noexcept {
  return impl_->layout().top_k;
}
std::size_t OLMoEQ7Kernel::group_size() const noexcept {
  return impl_->layout().group;
}
std::size_t OLMoEQ7Kernel::thread_count() const noexcept {
  return impl_->threads();
}
std::size_t OLMoEQ7Kernel::serialized_artifact_bytes() const noexcept {
  return impl_->artifact_bytes();
}

void OLMoEQ7Kernel::forward(
    const std::size_t layer, const std::span<const float> input,
    const std::size_t rows, const std::span<float> output,
    const std::span<std::uint32_t> selected_experts,
    OLMoEQ7Metrics* const metrics) {
  impl_->forward(layer, input, rows, output, selected_experts, metrics);
}

}  // namespace engram
