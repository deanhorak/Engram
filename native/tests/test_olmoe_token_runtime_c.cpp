#include "engram/olmoe_token_runtime_c.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint16_t kBFloatZero = 0x0000;
constexpr std::uint16_t kBFloatOne = 0x3F80;

int fail(const std::string& message) {
  std::cerr << message << '\n';
  return 1;
}

void store_u32(std::vector<std::uint8_t>& output,
               const std::size_t offset, const std::uint32_t value) {
  for (std::size_t byte = 0; byte < 4; ++byte) {
    output[offset + byte] =
        static_cast<std::uint8_t>((value >> (8 * byte)) & 0xFFU);
  }
}

void store_u64(std::vector<std::uint8_t>& output,
               const std::size_t offset, const std::uint64_t value) {
  for (std::size_t byte = 0; byte < 8; ++byte) {
    output[offset + byte] =
        static_cast<std::uint8_t>((value >> (8 * byte)) & 0xFFU);
  }
}

void write_u64(std::ofstream& output, const std::uint64_t value) {
  for (std::size_t byte = 0; byte < 8; ++byte) {
    output.put(static_cast<char>((value >> (8 * byte)) & 0xFFU));
  }
}

void write_u16(std::ofstream& output, const std::uint16_t value) {
  output.put(static_cast<char>(value & 0xFFU));
  output.put(static_cast<char>((value >> 8U) & 0xFFU));
}

struct Tensor {
  std::string name;
  std::vector<std::size_t> shape;
  std::vector<std::uint16_t> values;
};

void add_tensor(std::vector<Tensor>& tensors, std::string name,
                std::vector<std::size_t> shape,
                std::vector<std::uint16_t> values) {
  tensors.push_back(Tensor{
      .name = std::move(name),
      .shape = std::move(shape),
      .values = std::move(values),
  });
}

void write_safetensors(const std::filesystem::path& path) {
  const std::vector<std::uint16_t> vector = {
      kBFloatOne, kBFloatOne, kBFloatOne, kBFloatOne};
  const std::vector<std::uint16_t> identity = {
      kBFloatOne,  kBFloatZero, kBFloatZero, kBFloatZero,
      kBFloatZero, kBFloatOne,  kBFloatZero, kBFloatZero,
      kBFloatZero, kBFloatZero, kBFloatOne,  kBFloatZero,
      kBFloatZero, kBFloatZero, kBFloatZero, kBFloatOne};
  std::vector<Tensor> tensors;
  add_tensor(
      tensors, "model.embed_tokens.weight", {3, 4},
      {kBFloatOne, kBFloatZero, kBFloatOne,  kBFloatZero,
       kBFloatZero, kBFloatOne, kBFloatZero, kBFloatOne,
       kBFloatOne, kBFloatOne, kBFloatOne, kBFloatOne});
  add_tensor(tensors, "model.norm.weight", {4}, vector);
  add_tensor(
      tensors, "lm_head.weight", {3, 4},
      {kBFloatOne, kBFloatZero, kBFloatOne,  kBFloatZero,
       kBFloatZero, kBFloatOne, kBFloatZero, kBFloatOne,
       kBFloatOne, kBFloatOne, kBFloatOne, kBFloatOne});
  for (std::size_t layer = 0; layer < 2; ++layer) {
    const std::string base = "model.layers." + std::to_string(layer);
    const std::string attention = base + ".self_attn";
    add_tensor(tensors, base + ".input_layernorm.weight", {4}, vector);
    add_tensor(tensors, base + ".post_attention_layernorm.weight", {4},
               vector);
    add_tensor(tensors, attention + ".q_norm.weight", {4}, vector);
    add_tensor(tensors, attention + ".k_norm.weight", {4}, vector);
    add_tensor(tensors, attention + ".q_proj.weight", {4, 4}, identity);
    add_tensor(tensors, attention + ".k_proj.weight", {4, 4}, identity);
    add_tensor(tensors, attention + ".v_proj.weight", {4, 4}, identity);
    add_tensor(tensors, attention + ".o_proj.weight", {4, 4}, identity);
  }

  std::ostringstream header;
  header << '{';
  std::size_t offset = 0;
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    const Tensor& tensor = tensors[index];
    const std::size_t begin = offset;
    offset += tensor.values.size() * sizeof(std::uint16_t);
    if (index != 0) header << ',';
    header << '"' << tensor.name
           << R"(":{"dtype":"BF16","shape":[)";
    for (std::size_t dimension = 0; dimension < tensor.shape.size();
         ++dimension) {
      if (dimension != 0) header << ',';
      header << tensor.shape[dimension];
    }
    header << R"(],"data_offsets":[)" << begin << ',' << offset << "]}";
  }
  header << '}';
  const std::string header_text = header.str();
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  write_u64(output, header_text.size());
  output.write(header_text.data(),
               static_cast<std::streamsize>(header_text.size()));
  for (const Tensor& tensor : tensors) {
    for (const std::uint16_t value : tensor.values) {
      write_u16(output, value);
    }
  }
}

void write_q7_matrix(std::vector<std::uint8_t>& output,
                     const std::size_t offset,
                     const std::size_t coefficient_count,
                     const std::size_t scale_count) {
  // Canonical seven-bit zero coefficients (code bias 63).
  for (std::size_t coefficient = 0; coefficient < coefficient_count;
       ++coefficient) {
    const std::size_t bit = coefficient * 7;
    const std::size_t byte = bit / 8;
    const unsigned shift = static_cast<unsigned>(bit % 8);
    const std::uint16_t encoded =
        static_cast<std::uint16_t>(63U) << shift;
    output[offset + byte] |= static_cast<std::uint8_t>(encoded & 0xFFU);
    if (shift > 1) {
      output[offset + byte + 1] |=
          static_cast<std::uint8_t>((encoded >> 8U) & 0xFFU);
    }
  }
  for (std::size_t scale = 0; scale < scale_count; ++scale) {
    output[offset + 64 + scale * 2] = 0x80;
    output[offset + 64 + scale * 2 + 1] = 0x3F;
  }
}

void write_q7(const std::filesystem::path& path) {
  constexpr std::size_t header_bytes = 128;
  constexpr std::size_t directory_bytes = 128;
  constexpr std::size_t layer_bytes = 576;
  constexpr std::size_t file_bytes =
      header_bytes + directory_bytes + 2 * layer_bytes;
  std::vector<std::uint8_t> output(file_bytes);
  std::memcpy(output.data(), "ENGOQ711", 8);
  store_u32(output, 8, 1);
  store_u32(output, 12, 0x01020304U);
  store_u32(output, 16, 128);
  store_u32(output, 20, 64);
  store_u32(output, 24, 64);
  store_u32(output, 28, 64);
  store_u32(output, 32, 64);
  store_u32(output, 36, 7);
  store_u32(output, 40, 63);
  store_u32(output, 44, 2);
  store_u32(output, 48, 2);
  store_u32(output, 52, 4);
  store_u32(output, 56, 1);
  store_u32(output, 60, 1);
  store_u32(output, 64, 1);
  store_u64(output, 72, header_bytes);
  store_u64(output, 80, directory_bytes);
  store_u64(output, 88, file_bytes);

  for (std::size_t layer = 0; layer < 2; ++layer) {
    const std::size_t entry = header_bytes + layer * 64;
    const std::size_t layer_offset =
        header_bytes + directory_bytes + layer * layer_bytes;
    store_u32(output, entry, static_cast<std::uint32_t>(layer));
    store_u64(output, entry + 8, layer_offset);
    store_u64(output, entry + 16, layer_bytes);
    store_u64(output, entry + 24, 64);
    store_u64(output, entry + 32, 8);
    store_u64(output, entry + 40, 128);
    store_u64(output, entry + 48, 448);
    store_u64(output, entry + 56, layer_bytes);

    std::memcpy(output.data() + layer_offset, "ENGOQ7L1", 8);
    store_u32(output, layer_offset + 8, 1);
    store_u32(output, layer_offset + 12,
              static_cast<std::uint32_t>(layer));
    store_u32(output, layer_offset + 16, 4);
    store_u32(output, layer_offset + 20, 1);
    store_u32(output, layer_offset + 24, 1);
    store_u32(output, layer_offset + 28, 1);
    store_u32(output, layer_offset + 32, 2);
    store_u32(output, layer_offset + 36, 7);
    store_u64(output, layer_offset + 40, 64);
    store_u64(output, layer_offset + 48, 128);
    store_u64(output, layer_offset + 56, layer_bytes);

    const std::size_t expert = layer_offset + 128;
    std::memcpy(output.data() + expert, "ENGOQ7E1", 8);
    store_u32(output, expert + 8, 1);
    store_u32(output, expert + 12, 0);
    store_u32(output, expert + 16, 1);
    store_u32(output, expert + 20, 4);
    store_u32(output, expert + 24, 4);
    store_u32(output, expert + 28, 1);
    store_u64(output, expert + 32, 64);
    store_u64(output, expert + 40, 192);
    store_u64(output, expert + 48, 320);
    store_u64(output, expert + 56, 448);
    write_q7_matrix(output, expert + 64, 4, 2);
    write_q7_matrix(output, expert + 192, 4, 2);
    write_q7_matrix(output, expert + 320, 4, 4);
  }

  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(output.data()),
             static_cast<std::streamsize>(output.size()));
}

struct Fixture {
  explicit Fixture(const std::filesystem::path& value) : directory(value) {
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    weights = directory / "non_mlp.safetensors";
    q7 = directory / "experts.q7";
    write_safetensors(weights);
    write_q7(q7);
  }
  ~Fixture() { std::filesystem::remove_all(directory); }

  std::filesystem::path directory;
  std::filesystem::path weights;
  std::filesystem::path q7;
};

engram_olmoe_token_config config_for(const Fixture& fixture) {
  static std::string weights;
  static std::string q7;
  weights = fixture.weights.string();
  q7 = fixture.q7.string();
  return engram_olmoe_token_config{
      .non_mlp_safetensors = weights.c_str(),
      .q7_artifact = q7.c_str(),
      .layers = 2,
      .hidden_size = 4,
      .query_heads = 2,
      .key_value_heads = 2,
      .head_dimension = 2,
      .threads = 1,
      .local_window = 2,
      .older_candidates = 2,
      .older_top_k = 1,
      .sink_tokens = 1,
      .rms_norm_epsilon = 1.0e-5F,
      .rope_theta = 10000.0F,
  };
}

bool same_attention_metrics(
    const engram_olmoe_attention_metrics_v1& left,
    const engram_olmoe_attention_metrics_v1& right) {
  return left.logical_read_bytes == right.logical_read_bytes &&
         left.state_bytes == right.state_bytes &&
         left.scratch_bytes == right.scratch_bytes &&
         left.eviction_events == right.eviction_events &&
         left.older_candidate_entries_scored ==
             right.older_candidate_entries_scored &&
         left.older_selected_entries == right.older_selected_entries &&
         left.sink_insertions == right.sink_insertions &&
         left.heavy_hitter_updates == right.heavy_hitter_updates;
}

int forward_four(void* handle, std::int64_t& next,
                 engram_olmoe_token_metrics& metrics,
                 engram_olmoe_attention_metrics_v1& attention,
                 char* error, const std::size_t error_capacity) {
  const std::array<std::int64_t, 4> tokens = {1, 2, 1, 2};
  if (engram_olmoe_token_forward(
          handle, tokens.data(), tokens.size(), &next, &metrics, error,
          error_capacity) != 0) {
    return 1;
  }
  return engram_olmoe_token_copy_attention_metrics_v1(
      handle, &attention, error, error_capacity);
}

int copy_diagnostics(void* handle, std::array<float, 4>& final_state,
                     std::array<float, 3>& vocabulary_scores, char* error,
                     const std::size_t error_capacity) {
  return engram_olmoe_token_copy_last_diagnostics(
      handle, final_state.data(), final_state.size(),
      vocabulary_scores.data(), vocabulary_scores.size(), error,
      error_capacity);
}

}  // namespace

int main() {
  const Fixture fixture(
      std::filesystem::temp_directory_path() /
      "engram-olmoe-layered-token-runtime-test");
  engram_olmoe_token_config config = config_for(fixture);
  char error[512] = {};
  const std::array<engram_olmoe_attention_policy_v1, 2> homogeneous = {{
      {2, 2, 1, 1},
      {2, 2, 1, 1},
  }};
  const std::array<engram_olmoe_attention_policy_v1, 4>
      homogeneous_headwise = {{
          {2, 2, 1, 1},
          {2, 2, 1, 1},
          {2, 2, 1, 1},
          {2, 2, 1, 1},
      }};

  if (engram_olmoe_token_open_layered_v1(
          &config, homogeneous.data(), homogeneous.size() - 1, error,
          sizeof(error)) != nullptr ||
      std::string(error).find("count") == std::string::npos) {
    return fail("layered open accepted a short policy array");
  }
  if (engram_olmoe_token_open_layered_v1(
          &config, nullptr, homogeneous.size(), error, sizeof(error)) !=
      nullptr) {
    return fail("layered open accepted null policies");
  }
  auto invalid = homogeneous;
  invalid[1].older_top_k = invalid[1].older_candidates + 1;
  if (engram_olmoe_token_open_layered_v1(
          &config, invalid.data(), invalid.size(), error, sizeof(error)) !=
      nullptr ||
      std::string(error).find("policy") == std::string::npos) {
    return fail("layered open accepted inconsistent capacities");
  }
  invalid = homogeneous;
  invalid[0].local_window = 0;
  if (engram_olmoe_token_open_layered_v1(
          &config, invalid.data(), invalid.size(), error, sizeof(error)) !=
      nullptr) {
    return fail("layered open accepted a zero local window");
  }
  if (engram_olmoe_token_open_headwise_v1(
          &config, homogeneous_headwise.data(),
          homogeneous_headwise.size() - 1, error, sizeof(error)) != nullptr ||
      std::string(error).find("count") == std::string::npos) {
    return fail("head-wise open accepted a short policy array");
  }
  if (engram_olmoe_token_open_headwise_v1(
          &config, nullptr, homogeneous_headwise.size(), error,
          sizeof(error)) != nullptr) {
    return fail("head-wise open accepted null policies");
  }
  if (engram_olmoe_token_open_headwise_v1(
          nullptr, homogeneous_headwise.data(),
          homogeneous_headwise.size(), error, sizeof(error)) != nullptr) {
    return fail("head-wise open accepted a null config");
  }
  auto invalid_headwise = homogeneous_headwise;
  invalid_headwise[2].sink_tokens =
      invalid_headwise[2].older_candidates + 1;
  if (engram_olmoe_token_open_headwise_v1(
          &config, invalid_headwise.data(), invalid_headwise.size(), error,
          sizeof(error)) != nullptr ||
      std::string(error).find("policy") == std::string::npos) {
    return fail("head-wise open accepted inconsistent capacities");
  }
  engram_olmoe_token_config gqa_config = config;
  gqa_config.key_value_heads = 1;
  if (engram_olmoe_token_open_headwise_v1(
          &gqa_config, homogeneous_headwise.data(),
          homogeneous_headwise.size(), error, sizeof(error)) != nullptr ||
      std::string(error).find("equal") == std::string::npos) {
    return fail("head-wise open accepted grouped-query attention");
  }
  engram_olmoe_token_config unused_scalar_config = config;
  unused_scalar_config.local_window = 0;
  void* unused_scalar_headwise = engram_olmoe_token_open_headwise_v1(
      &unused_scalar_config, homogeneous_headwise.data(),
      homogeneous_headwise.size(), error, sizeof(error));
  if (unused_scalar_headwise == nullptr) {
    return fail(
        "head-wise open incorrectly validated unused scalar capacities");
  }
  engram_olmoe_token_close(unused_scalar_headwise);

  void* scalar =
      engram_olmoe_token_open(&config, error, sizeof(error));
  void* layered = engram_olmoe_token_open_layered_v1(
      &config, homogeneous.data(), homogeneous.size(), error, sizeof(error));
  void* headwise = engram_olmoe_token_open_headwise_v1(
      &config, homogeneous_headwise.data(), homogeneous_headwise.size(),
      error, sizeof(error));
  if (scalar == nullptr || layered == nullptr || headwise == nullptr) {
    engram_olmoe_token_close(scalar);
    engram_olmoe_token_close(layered);
    engram_olmoe_token_close(headwise);
    return fail(std::string("homogeneous runtime open failed: ") + error);
  }
  std::int64_t scalar_next = -1;
  std::int64_t layered_next = -1;
  std::int64_t headwise_next = -1;
  engram_olmoe_token_metrics scalar_metrics{};
  engram_olmoe_token_metrics layered_metrics{};
  engram_olmoe_token_metrics headwise_metrics{};
  engram_olmoe_attention_metrics_v1 scalar_attention{};
  engram_olmoe_attention_metrics_v1 layered_attention{};
  engram_olmoe_attention_metrics_v1 headwise_attention{};
  if (forward_four(scalar, scalar_next, scalar_metrics, scalar_attention,
                   error, sizeof(error)) != 0 ||
      forward_four(layered, layered_next, layered_metrics,
                   layered_attention, error, sizeof(error)) != 0 ||
      forward_four(headwise, headwise_next, headwise_metrics,
                   headwise_attention, error, sizeof(error)) != 0) {
    engram_olmoe_token_close(scalar);
    engram_olmoe_token_close(layered);
    engram_olmoe_token_close(headwise);
    return fail(std::string("homogeneous forward failed: ") + error);
  }
  if (scalar_next != layered_next ||
      scalar_metrics.positions_processed !=
          layered_metrics.positions_processed ||
      scalar_metrics.attention_weight_bytes !=
          layered_metrics.attention_weight_bytes ||
      scalar_metrics.q7_scheduled_bytes !=
          layered_metrics.q7_scheduled_bytes ||
      scalar_metrics.attention_state_bytes !=
          layered_metrics.attention_state_bytes ||
      !same_attention_metrics(scalar_attention, layered_attention) ||
      scalar_attention.state_bytes != 424 ||
      scalar_attention.scratch_bytes != 88) {
    engram_olmoe_token_close(scalar);
    engram_olmoe_token_close(layered);
    engram_olmoe_token_close(headwise);
    return fail("legacy scalar and homogeneous layered runtimes diverged");
  }
  std::array<float, 4> layered_final_state{};
  std::array<float, 4> headwise_final_state{};
  std::array<float, 3> layered_vocabulary_scores{};
  std::array<float, 3> headwise_vocabulary_scores{};
  if (copy_diagnostics(layered, layered_final_state,
                       layered_vocabulary_scores, error,
                       sizeof(error)) != 0 ||
      copy_diagnostics(headwise, headwise_final_state,
                       headwise_vocabulary_scores, error,
                       sizeof(error)) != 0 ||
      headwise_next != layered_next ||
      headwise_metrics.positions_processed !=
          layered_metrics.positions_processed ||
      headwise_metrics.attention_weight_bytes !=
          layered_metrics.attention_weight_bytes ||
      headwise_metrics.q7_scheduled_bytes !=
          layered_metrics.q7_scheduled_bytes ||
      engram_olmoe_token_position(headwise) !=
          engram_olmoe_token_position(layered) ||
      headwise_final_state != layered_final_state ||
      headwise_vocabulary_scores != layered_vocabulary_scores ||
      headwise_attention.logical_read_bytes !=
          layered_attention.logical_read_bytes ||
      headwise_attention.older_candidate_entries_scored !=
          layered_attention.older_candidate_entries_scored ||
      headwise_attention.older_selected_entries !=
          layered_attention.older_selected_entries ||
      headwise_attention.sink_insertions !=
          layered_attention.sink_insertions ||
      headwise_attention.heavy_hitter_updates !=
          layered_attention.heavy_hitter_updates ||
      headwise_attention.eviction_events !=
          layered_attention.eviction_events * config.query_heads ||
      headwise_attention.state_bytes != 456 ||
      headwise_attention.scratch_bytes != 176) {
    engram_olmoe_token_close(scalar);
    engram_olmoe_token_close(layered);
    engram_olmoe_token_close(headwise);
    return fail(
        "homogeneous head-wise runtime diverged from layered semantics");
  }
  engram_olmoe_token_close(scalar);
  engram_olmoe_token_close(layered);
  engram_olmoe_token_close(headwise);

  const std::array<engram_olmoe_attention_policy_v1, 2> heterogeneous = {{
      {1, 1, 1, 0},
      {3, 2, 1, 1},
  }};
  void* mixed = engram_olmoe_token_open_layered_v1(
      &config, heterogeneous.data(), heterogeneous.size(), error,
      sizeof(error));
  if (mixed == nullptr) {
    return fail(std::string("heterogeneous runtime open failed: ") + error);
  }
  std::int64_t mixed_next = -1;
  engram_olmoe_token_metrics mixed_metrics{};
  engram_olmoe_attention_metrics_v1 mixed_attention{};
  if (forward_four(mixed, mixed_next, mixed_metrics, mixed_attention, error,
                   sizeof(error)) != 0 ||
      engram_olmoe_token_position(mixed) != 4 ||
      mixed_metrics.positions_processed != 4 ||
      mixed_metrics.attention_state_bytes != 366 ||
      mixed_attention.state_bytes != 366 ||
      mixed_attention.scratch_bytes != 80 ||
      mixed_attention.eviction_events != 4 ||
      mixed_attention.older_candidate_entries_scored != 8 ||
      mixed_attention.older_selected_entries != 8 ||
      mixed_attention.sink_insertions != 2 ||
      mixed_attention.heavy_hitter_updates != 2) {
    std::ostringstream detail;
    detail << "heterogeneous capacity or counter sum is inexact: state="
           << mixed_attention.state_bytes
           << " scratch=" << mixed_attention.scratch_bytes
           << " evictions=" << mixed_attention.eviction_events
           << " scored="
           << mixed_attention.older_candidate_entries_scored
           << " selected=" << mixed_attention.older_selected_entries
           << " sinks=" << mixed_attention.sink_insertions
           << " heavy=" << mixed_attention.heavy_hitter_updates;
    engram_olmoe_token_close(mixed);
    return fail(detail.str());
  }
  engram_olmoe_token_reset(mixed);
  std::int64_t replay_next = -1;
  const std::int64_t token = 1;
  engram_olmoe_token_metrics replay_metrics{};
  if (engram_olmoe_token_position(mixed) != 0 ||
      engram_olmoe_token_forward(mixed, &token, 1, &replay_next,
                                 &replay_metrics, error,
                                 sizeof(error)) != 0 ||
      engram_olmoe_token_copy_attention_metrics_v1(
          mixed, &mixed_attention, error, sizeof(error)) != 0 ||
      engram_olmoe_token_position(mixed) != 1 ||
      replay_metrics.positions_processed != 1 ||
      mixed_attention.state_bytes != 366 ||
      mixed_attention.scratch_bytes != 80 ||
      mixed_attention.eviction_events != 0 ||
      mixed_attention.older_candidate_entries_scored != 0 ||
      mixed_attention.older_selected_entries != 0 ||
      mixed_attention.sink_insertions != 0 ||
      mixed_attention.heavy_hitter_updates != 0) {
    engram_olmoe_token_close(mixed);
    return fail("heterogeneous reset did not clear cumulative counters");
  }
  engram_olmoe_token_close(mixed);

  const std::array<engram_olmoe_attention_policy_v1, 4>
      heterogeneous_headwise = {{
          {1, 1, 1, 0},
          {3, 2, 1, 1},
          {1, 1, 1, 0},
          {3, 2, 1, 1},
      }};
  void* mixed_headwise = engram_olmoe_token_open_headwise_v1(
      &config, heterogeneous_headwise.data(),
      heterogeneous_headwise.size(), error, sizeof(error));
  if (mixed_headwise == nullptr) {
    return fail(std::string("mixed head-wise runtime open failed: ") +
                error);
  }
  std::int64_t mixed_headwise_next = -1;
  engram_olmoe_token_metrics mixed_headwise_metrics{};
  engram_olmoe_attention_metrics_v1 mixed_headwise_attention{};
  if (forward_four(mixed_headwise, mixed_headwise_next,
                   mixed_headwise_metrics, mixed_headwise_attention, error,
                   sizeof(error)) != 0 ||
      engram_olmoe_token_position(mixed_headwise) != 4 ||
      mixed_headwise_metrics.positions_processed != 4 ||
      mixed_headwise_metrics.attention_state_bytes != 398 ||
      mixed_headwise_attention.logical_read_bytes != 544 ||
      mixed_headwise_attention.state_bytes != 398 ||
      mixed_headwise_attention.scratch_bytes != 160 ||
      mixed_headwise_attention.eviction_events != 8 ||
      mixed_headwise_attention.older_candidate_entries_scored != 8 ||
      mixed_headwise_attention.older_selected_entries != 8 ||
      mixed_headwise_attention.sink_insertions != 2 ||
      mixed_headwise_attention.heavy_hitter_updates != 2) {
    std::ostringstream detail;
    detail << "mixed head-wise capacity or counter sum is inexact: logical="
           << mixed_headwise_attention.logical_read_bytes
           << " state=" << mixed_headwise_attention.state_bytes
           << " scratch=" << mixed_headwise_attention.scratch_bytes
           << " evictions=" << mixed_headwise_attention.eviction_events
           << " scored="
           << mixed_headwise_attention.older_candidate_entries_scored
           << " selected="
           << mixed_headwise_attention.older_selected_entries
           << " sinks=" << mixed_headwise_attention.sink_insertions
           << " heavy="
           << mixed_headwise_attention.heavy_hitter_updates;
    engram_olmoe_token_close(mixed_headwise);
    return fail(detail.str());
  }
  engram_olmoe_token_reset(mixed_headwise);
  replay_next = -1;
  replay_metrics = {};
  if (engram_olmoe_token_position(mixed_headwise) != 0 ||
      engram_olmoe_token_forward(
          mixed_headwise, &token, 1, &replay_next, &replay_metrics, error,
          sizeof(error)) != 0 ||
      engram_olmoe_token_copy_attention_metrics_v1(
          mixed_headwise, &mixed_headwise_attention, error,
          sizeof(error)) != 0 ||
      engram_olmoe_token_position(mixed_headwise) != 1 ||
      replay_metrics.positions_processed != 1 ||
      mixed_headwise_attention.logical_read_bytes != 64 ||
      mixed_headwise_attention.state_bytes != 398 ||
      mixed_headwise_attention.scratch_bytes != 160 ||
      mixed_headwise_attention.eviction_events != 0 ||
      mixed_headwise_attention.older_candidate_entries_scored != 0 ||
      mixed_headwise_attention.older_selected_entries != 0 ||
      mixed_headwise_attention.sink_insertions != 0 ||
      mixed_headwise_attention.heavy_hitter_updates != 0) {
    engram_olmoe_token_close(mixed_headwise);
    return fail("mixed head-wise reset did not clear cumulative counters");
  }
  engram_olmoe_token_close(mixed_headwise);
  return 0;
}
