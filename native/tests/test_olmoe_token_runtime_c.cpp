#include "engram/olmoe_token_runtime_c.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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

bool same_base_counters(const engram_olmoe_token_metrics& left,
                        const engram_olmoe_token_metrics& right) {
  // Wall-clock fields are observations, not deterministic work counters.
  return left.positions_processed == right.positions_processed &&
         left.attention_weight_bytes == right.attention_weight_bytes &&
         left.q7_scheduled_bytes == right.q7_scheduled_bytes &&
         left.attention_state_bytes == right.attention_state_bytes;
}

template <std::size_t Size>
bool all_finite(const std::array<float, Size>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](const float value) {
                       return std::isfinite(value);
                     });
}

float bf16_round_trip(const float value) {
  std::uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  bits &= 0xFFFF0000U;
  float result{};
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::array<float, 4> normalize_four(
    const std::array<float, 4>& input, const float epsilon) {
  float sum = 0.0F;
  for (const float value : input) sum += value * value;
  const float inverse = 1.0F / std::sqrt(sum / 4.0F + epsilon);
  std::array<float, 4> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = input[index] * inverse;
  }
  return result;
}

std::array<float, 2> rotate_pair(
    const std::array<float, 2>& input, const std::size_t position) {
  const float angle = static_cast<float>(position);
  const float cosine = std::cos(angle);
  const float sine = std::sin(angle);
  return {
      input[0] * cosine - input[1] * sine,
      input[1] * cosine + input[0] * sine,
  };
}

float dot_pair(const std::array<float, 2>& left,
               const std::array<float, 2>& right) {
  float result = 0.0F;
  result += left[0] * right[0];
  result += left[1] * right[1];
  return result;
}

template <std::size_t Size>
std::array<float, 2> softmax_value(
    const std::array<float, Size>& scores,
    const std::array<std::array<float, 2>, Size>& values) {
  float maximum = -std::numeric_limits<float>::infinity();
  for (const float score : scores) maximum = std::max(maximum, score);
  std::array<float, Size> weights{};
  float denominator = 0.0F;
  for (std::size_t index = 0; index < Size; ++index) {
    weights[index] = std::exp(scores[index] - maximum);
    denominator += weights[index];
  }
  std::array<float, 2> result{};
  for (std::size_t index = 0; index < Size; ++index) {
    const float weight = weights[index] / denominator;
    result[0] += weight * values[index][0];
    result[1] += weight * values[index][1];
  }
  return result;
}

std::array<float, 4> expected_first_read_layer_zero_target(
    const float epsilon, const float episodic_logit_bias) {
  const std::array<float, 4> token_one =
      normalize_four({0.0F, 1.0F, 0.0F, 1.0F}, epsilon);
  const std::array<float, 4> token_two =
      normalize_four({1.0F, 1.0F, 1.0F, 1.0F}, epsilon);
  const std::array<float, 4> query_one =
      normalize_four(token_one, epsilon);
  const std::array<float, 4> query_two =
      normalize_four(token_two, epsilon);
  const std::array<float, 2> query =
      rotate_pair({query_one[0], query_one[1]}, 2);
  const std::array<float, 2> key_zero =
      rotate_pair({query_one[0], query_one[1]}, 0);
  const std::array<float, 2> key_one =
      rotate_pair({query_two[0], query_two[1]}, 1);
  const std::array<float, 2> key_two = query;
  const std::array<float, 2> value_zero = {
      token_one[0], token_one[1]};
  const std::array<float, 2> value_one = {
      token_two[0], token_two[1]};
  const std::array<float, 2> value_two = value_zero;
  const float scale = 1.0F / std::sqrt(2.0F);

  const std::array<float, 3> shadow_scores = {
      dot_pair(query, key_zero) * scale,
      dot_pair(query, key_one) * scale,
      dot_pair(query, key_two) * scale,
  };
  const std::array<std::array<float, 2>, 3> shadow_values = {
      value_zero, value_one, value_two};
  const std::array<float, 2> shadow =
      softmax_value(shadow_scores, shadow_values);

  const std::array<float, 2> episodic_key_zero = {
      bf16_round_trip(key_zero[0]), bf16_round_trip(key_zero[1])};
  const std::array<float, 2> episodic_key_one = {
      bf16_round_trip(key_one[0]), bf16_round_trip(key_one[1])};
  const std::array<float, 2> episodic_value_zero = {
      bf16_round_trip(value_zero[0]), bf16_round_trip(value_zero[1])};
  const std::array<float, 2> episodic_value_one = {
      bf16_round_trip(value_one[0]), bf16_round_trip(value_one[1])};
  const std::array<float, 4> base_scores = {
      dot_pair(query, key_one) * scale,
      dot_pair(query, key_two) * scale,
      dot_pair(query, episodic_key_zero) * scale +
          episodic_logit_bias,
      dot_pair(query, episodic_key_one) * scale +
          episodic_logit_bias,
  };
  const std::array<std::array<float, 2>, 4> base_values = {
      value_one, value_two, episodic_value_zero, episodic_value_one};
  const std::array<float, 2> base =
      softmax_value(base_scores, base_values);
  return {
      shadow[0] - base[0], shadow[1] - base[1],
      shadow[0] - base[0], shadow[1] - base[1],
  };
}

float expected_first_read_layer_zero_shadow_source_mass(
    const float epsilon) {
  const std::array<float, 4> token_one =
      normalize_four({0.0F, 1.0F, 0.0F, 1.0F}, epsilon);
  const std::array<float, 4> token_two =
      normalize_four({1.0F, 1.0F, 1.0F, 1.0F}, epsilon);
  const std::array<float, 4> query_one =
      normalize_four(token_one, epsilon);
  const std::array<float, 4> query_two =
      normalize_four(token_two, epsilon);
  const std::array<float, 2> query =
      rotate_pair({query_one[0], query_one[1]}, 2);
  const std::array<float, 2> key_zero =
      rotate_pair({query_one[0], query_one[1]}, 0);
  const std::array<float, 2> key_one =
      rotate_pair({query_two[0], query_two[1]}, 1);
  const std::array<float, 2> key_two = query;
  const float scale = 1.0F / std::sqrt(2.0F);
  std::array<float, 3> scores = {
      dot_pair(query, key_zero) * scale,
      dot_pair(query, key_one) * scale,
      dot_pair(query, key_two) * scale,
  };
  const float maximum =
      *std::max_element(scores.begin(), scores.end());
  float denominator = 0.0F;
  for (float& score : scores) {
    score = std::exp(score - maximum);
    denominator += score;
  }
  return (scores[0] + scores[1]) / denominator;
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

  const engram_olmoe_episodic_policy_v1 episodic_policy{
      .slots = 4,
      .span_size = 2,
  };
  const engram_olmoe_episodic_policy_v1 invalid_episodic_policy{
      .slots = 3,
      .span_size = 2,
  };
  if (engram_olmoe_token_open_episodic_v1(
          &config, nullptr, error, sizeof(error)) != nullptr ||
      engram_olmoe_token_open_episodic_v1(
          &config, &invalid_episodic_policy, error, sizeof(error)) !=
          nullptr ||
      std::string(error).find("policy") == std::string::npos) {
    return fail("episodic open accepted an invalid policy");
  }
  void* read_before_write = engram_olmoe_token_open_episodic_v1(
      &config, &episodic_policy, error, sizeof(error));
  if (read_before_write == nullptr) {
    return fail(std::string("episodic runtime open failed: ") + error);
  }
  const std::int32_t no_write = -1;
  const std::int32_t read_first_span = 0;
  std::int64_t invalid_next = -1;
  if (engram_olmoe_token_forward_episodic_v1(
          read_before_write, &token, 1, &no_write, &read_first_span,
          &invalid_next, nullptr, error, sizeof(error)) == 0 ||
      std::string(error).find("causal") == std::string::npos) {
    engram_olmoe_token_close(read_before_write);
    return fail("episodic runtime accepted a read before capture");
  }
  engram_olmoe_token_close(read_before_write);

  void* rejected_batch = engram_olmoe_token_open_episodic_v1(
      &config, &episodic_policy, error, sizeof(error));
  void* rejection_reference = engram_olmoe_token_open_episodic_v1(
      &config, &episodic_policy, error, sizeof(error));
  if (rejected_batch == nullptr || rejection_reference == nullptr) {
    engram_olmoe_token_close(rejected_batch);
    engram_olmoe_token_close(rejection_reference);
    return fail(std::string("episodic rejection runtime open failed: ") +
                error);
  }
  const std::array<std::int64_t, 2> rejection_tokens = {1, 2};
  const std::array<std::int32_t, 2> duplicate_writes = {0, 0};
  const std::array<std::int32_t, 2> no_reads = {-1, -1};
  engram_olmoe_token_metrics rejected_base{};
  if (engram_olmoe_token_forward_episodic_v1(
          rejected_batch, rejection_tokens.data(), rejection_tokens.size(),
          duplicate_writes.data(), no_reads.data(), &invalid_next,
          &rejected_base, error, sizeof(error)) == 0 ||
      std::string(error).find("already active") == std::string::npos ||
      engram_olmoe_token_position(rejected_batch) != 0) {
    engram_olmoe_token_close(rejected_batch);
    engram_olmoe_token_close(rejection_reference);
    return fail("later-row episodic rejection mutated the public position");
  }
  engram_olmoe_episodic_metrics_v1 rejected_metrics{};
  if (engram_olmoe_token_copy_episodic_metrics_v1(
          rejected_batch, &rejected_metrics, error, sizeof(error)) != 0 ||
      rejected_metrics.slots_written != 0 ||
      rejected_metrics.read_events != 0 ||
      rejected_metrics.active_slots != 0 ||
      rejected_metrics.entries_read != 0 ||
      rejected_metrics.write_bytes != 0 ||
      rejected_metrics.key_read_bytes != 0 ||
      rejected_metrics.value_read_bytes != 0 ||
      rejected_metrics.duplicate_older_entries_suppressed != 0 ||
      rejected_metrics.state_bytes != 0 ||
      rejected_metrics.scratch_bytes != 0) {
    engram_olmoe_token_close(rejected_batch);
    engram_olmoe_token_close(rejection_reference);
    return fail("later-row episodic rejection mutated counters");
  }
  const std::array<std::int32_t, 2> valid_writes = {0, 1};
  std::int64_t rejected_recovery_next = -1;
  std::int64_t rejection_reference_next = -1;
  engram_olmoe_token_metrics rejection_reference_base{};
  if (engram_olmoe_token_forward_episodic_v1(
          rejected_batch, rejection_tokens.data(), rejection_tokens.size(),
          valid_writes.data(), no_reads.data(), &rejected_recovery_next,
          &rejected_base, error, sizeof(error)) != 0 ||
      engram_olmoe_token_forward_episodic_v1(
          rejection_reference, rejection_tokens.data(),
          rejection_tokens.size(), valid_writes.data(), no_reads.data(),
          &rejection_reference_next, &rejection_reference_base, error,
          sizeof(error)) != 0) {
    engram_olmoe_token_close(rejected_batch);
    engram_olmoe_token_close(rejection_reference);
    return fail(std::string("episodic rejection recovery failed: ") +
                error);
  }
  std::array<float, 4> rejected_recovery_state{};
  std::array<float, 4> rejection_reference_state{};
  std::array<float, 3> rejected_recovery_logits{};
  std::array<float, 3> rejection_reference_logits{};
  engram_olmoe_episodic_metrics_v1 rejection_reference_metrics{};
  if (copy_diagnostics(rejected_batch, rejected_recovery_state,
                       rejected_recovery_logits, error, sizeof(error)) != 0 ||
      copy_diagnostics(rejection_reference, rejection_reference_state,
                       rejection_reference_logits, error,
                       sizeof(error)) != 0 ||
      engram_olmoe_token_copy_episodic_metrics_v1(
          rejected_batch, &rejected_metrics, error, sizeof(error)) != 0 ||
      engram_olmoe_token_copy_episodic_metrics_v1(
          rejection_reference, &rejection_reference_metrics, error,
          sizeof(error)) != 0 ||
      rejected_recovery_next != rejection_reference_next ||
      rejected_recovery_state != rejection_reference_state ||
      rejected_recovery_logits != rejection_reference_logits ||
      std::memcmp(&rejected_metrics, &rejection_reference_metrics,
                  sizeof(rejected_metrics)) != 0) {
    engram_olmoe_token_close(rejected_batch);
    engram_olmoe_token_close(rejection_reference);
    return fail("later-row rejection did not preserve recovery parity");
  }
  engram_olmoe_token_close(rejected_batch);
  engram_olmoe_token_close(rejection_reference);

  void* episodic_batch = engram_olmoe_token_open_episodic_v1(
      &config, &episodic_policy, error, sizeof(error));
  void* episodic_steps = engram_olmoe_token_open_episodic_v1(
      &config, &episodic_policy, error, sizeof(error));
  if (episodic_batch == nullptr || episodic_steps == nullptr) {
    engram_olmoe_token_close(episodic_batch);
    engram_olmoe_token_close(episodic_steps);
    return fail(std::string("episodic runtime open failed: ") + error);
  }
  const std::array<std::int64_t, 4> episodic_tokens = {1, 2, 1, 2};
  const std::array<std::int32_t, 4> episodic_writes = {0, 1, -1, -1};
  const std::array<std::int32_t, 4> episodic_reads = {-1, -1, 0, 0};
  std::int64_t episodic_batch_next = -1;
  std::int64_t episodic_step_next = -1;
  engram_olmoe_token_metrics episodic_batch_base{};
  engram_olmoe_token_metrics episodic_step_base{};
  if (engram_olmoe_token_forward_episodic_v1(
          episodic_batch, episodic_tokens.data(), episodic_tokens.size(),
          episodic_writes.data(), episodic_reads.data(),
          &episodic_batch_next, &episodic_batch_base, error,
          sizeof(error)) != 0) {
    engram_olmoe_token_close(episodic_batch);
    engram_olmoe_token_close(episodic_steps);
    return fail(std::string("batched episodic forward failed: ") + error);
  }
  for (std::size_t row = 0; row < episodic_tokens.size(); ++row) {
    if (engram_olmoe_token_forward_episodic_v1(
            episodic_steps, &episodic_tokens[row], 1,
            &episodic_writes[row], &episodic_reads[row],
            &episodic_step_next, &episodic_step_base, error,
            sizeof(error)) != 0) {
      engram_olmoe_token_close(episodic_batch);
      engram_olmoe_token_close(episodic_steps);
      return fail(std::string("singleton episodic forward failed: ") +
                  error);
    }
  }
  engram_olmoe_episodic_metrics_v1 episodic_batch_metrics{};
  engram_olmoe_episodic_metrics_v1 episodic_step_metrics{};
  std::array<float, 4> episodic_batch_state{};
  std::array<float, 4> episodic_step_state{};
  std::array<float, 3> episodic_batch_logits{};
  std::array<float, 3> episodic_step_logits{};
  if (engram_olmoe_token_copy_episodic_metrics_v1(
          episodic_batch, &episodic_batch_metrics, error,
          sizeof(error)) != 0 ||
      engram_olmoe_token_copy_episodic_metrics_v1(
          episodic_steps, &episodic_step_metrics, error,
          sizeof(error)) != 0 ||
      copy_diagnostics(episodic_batch, episodic_batch_state,
                       episodic_batch_logits, error, sizeof(error)) != 0 ||
      copy_diagnostics(episodic_steps, episodic_step_state,
                       episodic_step_logits, error, sizeof(error)) != 0 ||
      episodic_batch_next != episodic_step_next ||
      episodic_batch_state != episodic_step_state ||
      episodic_batch_logits != episodic_step_logits ||
      std::memcmp(&episodic_batch_metrics, &episodic_step_metrics,
                  sizeof(episodic_batch_metrics)) != 0 ||
      episodic_batch_metrics.slots_written != 4 ||
      episodic_batch_metrics.read_events != 4 ||
      episodic_batch_metrics.active_slots != 4 ||
      episodic_batch_metrics.entries_read != 16 ||
      episodic_batch_metrics.write_bytes != 64 ||
      episodic_batch_metrics.key_read_bytes != 64 ||
      episodic_batch_metrics.value_read_bytes != 64 ||
      episodic_batch_metrics.duplicate_older_entries_suppressed != 12 ||
      episodic_batch_metrics.state_bytes != 616 ||
      episodic_batch_metrics.scratch_bytes != 120) {
    std::ostringstream detail;
    detail << "episodic batch/singleton parity or counters failed: writes="
           << episodic_batch_metrics.slots_written
           << " reads=" << episodic_batch_metrics.read_events
           << " active=" << episodic_batch_metrics.active_slots
           << " entries=" << episodic_batch_metrics.entries_read
           << " write_bytes=" << episodic_batch_metrics.write_bytes
           << " key_bytes=" << episodic_batch_metrics.key_read_bytes
           << " value_bytes=" << episodic_batch_metrics.value_read_bytes
           << " dedup="
           << episodic_batch_metrics
                  .duplicate_older_entries_suppressed
           << " state=" << episodic_batch_metrics.state_bytes
           << " scratch=" << episodic_batch_metrics.scratch_bytes;
    engram_olmoe_token_close(episodic_batch);
    engram_olmoe_token_close(episodic_steps);
    return fail(detail.str());
  }
  engram_olmoe_token_reset(episodic_batch);
  const std::int32_t write_slot_zero = 0;
  if (engram_olmoe_token_forward_episodic_v1(
          episodic_batch, &token, 1, &write_slot_zero, &no_write,
          &episodic_batch_next, &episodic_batch_base, error,
          sizeof(error)) != 0 ||
      engram_olmoe_token_copy_episodic_metrics_v1(
          episodic_batch, &episodic_batch_metrics, error,
          sizeof(error)) != 0 ||
      episodic_batch_metrics.slots_written != 2 ||
      episodic_batch_metrics.read_events != 0 ||
      episodic_batch_metrics.active_slots != 2 ||
      episodic_batch_metrics.entries_read != 0 ||
      episodic_batch_metrics.write_bytes != 32 ||
      episodic_batch_metrics.key_read_bytes != 0 ||
      episodic_batch_metrics.value_read_bytes != 0 ||
      episodic_batch_metrics.duplicate_older_entries_suppressed != 0 ||
      episodic_batch_metrics.state_bytes != 616 ||
      episodic_batch_metrics.scratch_bytes != 120) {
    engram_olmoe_token_close(episodic_batch);
    engram_olmoe_token_close(episodic_steps);
    return fail("episodic reset did not clear state and counters");
  }
  engram_olmoe_token_close(episodic_batch);
  engram_olmoe_token_close(episodic_steps);

  const std::array<std::uint8_t, 4> all_episodic_heads = {1, 1, 1, 1};
  const std::array<std::uint8_t, 4> mixed_episodic_heads = {1, 0, 0, 0};
  const std::array<std::uint8_t, 4> zero_episodic_heads = {0, 0, 0, 0};
  const std::array<std::uint8_t, 4> invalid_episodic_heads = {1, 0, 2, 0};
  if (engram_olmoe_token_open_episodic_headwise_v1(
          &config, &episodic_policy, nullptr, all_episodic_heads.size(),
          error, sizeof(error)) != nullptr ||
      engram_olmoe_token_open_episodic_headwise_v1(
          &config, &episodic_policy, all_episodic_heads.data(),
          all_episodic_heads.size() - 1, error, sizeof(error)) != nullptr ||
      engram_olmoe_token_open_episodic_headwise_v1(
          &config, &episodic_policy, zero_episodic_heads.data(),
          zero_episodic_heads.size(), error, sizeof(error)) != nullptr ||
      engram_olmoe_token_open_episodic_headwise_v1(
          &config, &episodic_policy, invalid_episodic_heads.data(),
          invalid_episodic_heads.size(), error, sizeof(error)) != nullptr ||
      std::string(error).find("mask") == std::string::npos) {
    return fail("head-gated episodic open accepted an invalid mask");
  }
  for (const float invalid_bias :
       {std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()}) {
    if (engram_olmoe_token_open_episodic_headwise_v2(
            &config, &episodic_policy, all_episodic_heads.data(),
            all_episodic_heads.size(), invalid_bias, error,
            sizeof(error)) != nullptr ||
        std::string(error).find("bias") == std::string::npos) {
      return fail("additive episodic open accepted a non-finite bias");
    }
  }
  if (engram_olmoe_token_open_episodic_headwise_v2(
          &config, &invalid_episodic_policy, all_episodic_heads.data(),
          all_episodic_heads.size(), 0.0F, error, sizeof(error)) != nullptr ||
      engram_olmoe_token_open_episodic_headwise_v2(
          &config, &episodic_policy, invalid_episodic_heads.data(),
          invalid_episodic_heads.size(), 0.0F, error,
          sizeof(error)) != nullptr) {
    return fail("additive episodic open accepted an invalid policy or mask");
  }

  void* legacy_all_heads = engram_olmoe_token_open_episodic_v1(
      &config, &episodic_policy, error, sizeof(error));
  void* explicit_all_heads =
      engram_olmoe_token_open_episodic_headwise_v1(
          &config, &episodic_policy, all_episodic_heads.data(),
          all_episodic_heads.size(), error, sizeof(error));
  void* mixed_heads = engram_olmoe_token_open_episodic_headwise_v1(
      &config, &episodic_policy, mixed_episodic_heads.data(),
      mixed_episodic_heads.size(), error, sizeof(error));
  void* explicit_all_heads_v2_zero =
      engram_olmoe_token_open_episodic_headwise_v2(
          &config, &episodic_policy, all_episodic_heads.data(),
          all_episodic_heads.size(), 0.0F, error, sizeof(error));
  void* explicit_all_heads_v2_biased =
      engram_olmoe_token_open_episodic_headwise_v2(
          &config, &episodic_policy, all_episodic_heads.data(),
          all_episodic_heads.size(), 1.25F, error, sizeof(error));
  if (legacy_all_heads == nullptr || explicit_all_heads == nullptr ||
      mixed_heads == nullptr || explicit_all_heads_v2_zero == nullptr ||
      explicit_all_heads_v2_biased == nullptr) {
    engram_olmoe_token_close(legacy_all_heads);
    engram_olmoe_token_close(explicit_all_heads);
    engram_olmoe_token_close(mixed_heads);
    engram_olmoe_token_close(explicit_all_heads_v2_zero);
    engram_olmoe_token_close(explicit_all_heads_v2_biased);
    return fail(std::string("head-gated episodic open failed: ") + error);
  }
  std::int64_t legacy_all_next = -1;
  std::int64_t explicit_all_next = -1;
  std::int64_t gated_next = -1;
  std::int64_t v2_zero_next = -1;
  std::int64_t v2_biased_next = -1;
  engram_olmoe_token_metrics legacy_all_base{};
  engram_olmoe_token_metrics explicit_all_base{};
  engram_olmoe_token_metrics mixed_base{};
  engram_olmoe_token_metrics v2_zero_base{};
  engram_olmoe_token_metrics v2_biased_base{};
  if (engram_olmoe_token_forward_episodic_v1(
          legacy_all_heads, episodic_tokens.data(), episodic_tokens.size(),
          episodic_writes.data(), episodic_reads.data(), &legacy_all_next,
          &legacy_all_base, error, sizeof(error)) != 0 ||
      engram_olmoe_token_forward_episodic_v1(
          explicit_all_heads, episodic_tokens.data(),
          episodic_tokens.size(), episodic_writes.data(),
          episodic_reads.data(), &explicit_all_next, &explicit_all_base,
          error, sizeof(error)) != 0 ||
      engram_olmoe_token_forward_episodic_v1(
          mixed_heads, episodic_tokens.data(), episodic_tokens.size(),
          episodic_writes.data(), episodic_reads.data(), &gated_next,
          &mixed_base, error, sizeof(error)) != 0 ||
      engram_olmoe_token_forward_episodic_v1(
          explicit_all_heads_v2_zero, episodic_tokens.data(),
          episodic_tokens.size(), episodic_writes.data(),
          episodic_reads.data(), &v2_zero_next, &v2_zero_base, error,
          sizeof(error)) != 0 ||
      engram_olmoe_token_forward_episodic_v1(
          explicit_all_heads_v2_biased, episodic_tokens.data(),
          episodic_tokens.size(), episodic_writes.data(),
          episodic_reads.data(), &v2_biased_next, &v2_biased_base, error,
          sizeof(error)) != 0) {
    engram_olmoe_token_close(legacy_all_heads);
    engram_olmoe_token_close(explicit_all_heads);
    engram_olmoe_token_close(mixed_heads);
    engram_olmoe_token_close(explicit_all_heads_v2_zero);
    engram_olmoe_token_close(explicit_all_heads_v2_biased);
    return fail(std::string("head-gated episodic forward failed: ") + error);
  }
  engram_olmoe_episodic_metrics_v1 legacy_all_metrics{};
  engram_olmoe_episodic_metrics_v1 explicit_all_metrics{};
  engram_olmoe_episodic_metrics_v1 gated_metrics{};
  engram_olmoe_episodic_metrics_v1 v2_zero_metrics{};
  engram_olmoe_episodic_metrics_v1 v2_biased_metrics{};
  engram_olmoe_attention_metrics_v1 explicit_all_attention{};
  engram_olmoe_attention_metrics_v1 v2_zero_attention{};
  std::array<float, 4> legacy_all_state{};
  std::array<float, 4> explicit_all_state{};
  std::array<float, 3> legacy_all_logits{};
  std::array<float, 3> explicit_all_logits{};
  std::array<float, 4> v2_zero_state{};
  std::array<float, 4> v2_biased_state{};
  std::array<float, 3> v2_zero_logits{};
  std::array<float, 3> v2_biased_logits{};
  if (engram_olmoe_token_copy_episodic_metrics_v1(
          legacy_all_heads, &legacy_all_metrics, error, sizeof(error)) != 0 ||
      engram_olmoe_token_copy_episodic_metrics_v1(
          explicit_all_heads, &explicit_all_metrics, error,
          sizeof(error)) != 0 ||
      engram_olmoe_token_copy_episodic_metrics_v1(
          mixed_heads, &gated_metrics, error, sizeof(error)) != 0 ||
      engram_olmoe_token_copy_episodic_metrics_v1(
          explicit_all_heads_v2_zero, &v2_zero_metrics, error,
          sizeof(error)) != 0 ||
      engram_olmoe_token_copy_episodic_metrics_v1(
          explicit_all_heads_v2_biased, &v2_biased_metrics, error,
          sizeof(error)) != 0 ||
      engram_olmoe_token_copy_attention_metrics_v1(
          explicit_all_heads, &explicit_all_attention, error,
          sizeof(error)) != 0 ||
      engram_olmoe_token_copy_attention_metrics_v1(
          explicit_all_heads_v2_zero, &v2_zero_attention, error,
          sizeof(error)) != 0 ||
      copy_diagnostics(legacy_all_heads, legacy_all_state,
                       legacy_all_logits, error, sizeof(error)) != 0 ||
      copy_diagnostics(explicit_all_heads, explicit_all_state,
                       explicit_all_logits, error, sizeof(error)) != 0 ||
      copy_diagnostics(explicit_all_heads_v2_zero, v2_zero_state,
                       v2_zero_logits, error, sizeof(error)) != 0 ||
      copy_diagnostics(explicit_all_heads_v2_biased, v2_biased_state,
                       v2_biased_logits, error, sizeof(error)) != 0 ||
      legacy_all_next != explicit_all_next ||
      legacy_all_state != explicit_all_state ||
      legacy_all_logits != explicit_all_logits ||
      explicit_all_next != v2_zero_next ||
      explicit_all_state != v2_zero_state ||
      explicit_all_logits != v2_zero_logits ||
      std::memcmp(&legacy_all_metrics, &explicit_all_metrics,
                  sizeof(legacy_all_metrics)) != 0 ||
      std::memcmp(&explicit_all_metrics, &v2_zero_metrics,
                  sizeof(explicit_all_metrics)) != 0 ||
      !same_attention_metrics(explicit_all_attention,
                              v2_zero_attention) ||
      legacy_all_base.positions_processed !=
          explicit_all_base.positions_processed ||
      legacy_all_base.attention_weight_bytes !=
          explicit_all_base.attention_weight_bytes ||
      legacy_all_base.q7_scheduled_bytes !=
          explicit_all_base.q7_scheduled_bytes ||
      legacy_all_base.attention_state_bytes !=
          explicit_all_base.attention_state_bytes ||
      explicit_all_base.positions_processed !=
          v2_zero_base.positions_processed ||
      explicit_all_base.attention_weight_bytes !=
          v2_zero_base.attention_weight_bytes ||
      explicit_all_base.q7_scheduled_bytes !=
          v2_zero_base.q7_scheduled_bytes ||
      explicit_all_base.attention_state_bytes !=
          v2_zero_base.attention_state_bytes) {
    engram_olmoe_token_close(legacy_all_heads);
    engram_olmoe_token_close(explicit_all_heads);
    engram_olmoe_token_close(mixed_heads);
    engram_olmoe_token_close(explicit_all_heads_v2_zero);
    engram_olmoe_token_close(explicit_all_heads_v2_biased);
    return fail("V1/V2 zero-bias episodic parity failed");
  }
  if ((v2_biased_state == v2_zero_state &&
       v2_biased_logits == v2_zero_logits) ||
      v2_biased_base.positions_processed !=
          v2_zero_base.positions_processed ||
      v2_biased_base.attention_weight_bytes !=
          v2_zero_base.attention_weight_bytes ||
      v2_biased_base.q7_scheduled_bytes !=
          v2_zero_base.q7_scheduled_bytes ||
      v2_biased_base.attention_state_bytes !=
          v2_zero_base.attention_state_bytes ||
      v2_biased_metrics.slots_written != v2_zero_metrics.slots_written ||
      v2_biased_metrics.read_events != v2_zero_metrics.read_events ||
      v2_biased_metrics.active_slots != v2_zero_metrics.active_slots ||
      v2_biased_metrics.entries_read != v2_zero_metrics.entries_read ||
      v2_biased_metrics.write_bytes != v2_zero_metrics.write_bytes ||
      v2_biased_metrics.key_read_bytes != v2_zero_metrics.key_read_bytes ||
      v2_biased_metrics.value_read_bytes !=
          v2_zero_metrics.value_read_bytes ||
      v2_biased_metrics.state_bytes != v2_zero_metrics.state_bytes ||
      v2_biased_metrics.scratch_bytes != v2_zero_metrics.scratch_bytes) {
    engram_olmoe_token_close(legacy_all_heads);
    engram_olmoe_token_close(explicit_all_heads);
    engram_olmoe_token_close(mixed_heads);
    engram_olmoe_token_close(explicit_all_heads_v2_zero);
    engram_olmoe_token_close(explicit_all_heads_v2_biased);
    return fail("nonzero episodic bias did not preserve fixed resources");
  }
  engram_olmoe_token_reset(explicit_all_heads_v2_biased);
  std::int64_t v2_biased_replay_next = -1;
  engram_olmoe_token_metrics v2_biased_replay_base{};
  engram_olmoe_episodic_metrics_v1 v2_biased_replay_metrics{};
  std::array<float, 4> v2_biased_replay_state{};
  std::array<float, 3> v2_biased_replay_logits{};
  if (engram_olmoe_token_forward_episodic_v1(
          explicit_all_heads_v2_biased, episodic_tokens.data(),
          episodic_tokens.size(), episodic_writes.data(),
          episodic_reads.data(), &v2_biased_replay_next,
          &v2_biased_replay_base, error, sizeof(error)) != 0 ||
      engram_olmoe_token_copy_episodic_metrics_v1(
          explicit_all_heads_v2_biased, &v2_biased_replay_metrics, error,
          sizeof(error)) != 0 ||
      copy_diagnostics(
          explicit_all_heads_v2_biased, v2_biased_replay_state,
          v2_biased_replay_logits, error, sizeof(error)) != 0 ||
      v2_biased_replay_next != v2_biased_next ||
      v2_biased_replay_state != v2_biased_state ||
      v2_biased_replay_logits != v2_biased_logits ||
      std::memcmp(&v2_biased_replay_metrics, &v2_biased_metrics,
                  sizeof(v2_biased_metrics)) != 0 ||
      v2_biased_replay_base.positions_processed !=
          v2_biased_base.positions_processed ||
      v2_biased_replay_base.attention_weight_bytes !=
          v2_biased_base.attention_weight_bytes ||
      v2_biased_replay_base.q7_scheduled_bytes !=
          v2_biased_base.q7_scheduled_bytes ||
      v2_biased_replay_base.attention_state_bytes !=
          v2_biased_base.attention_state_bytes) {
    engram_olmoe_token_close(legacy_all_heads);
    engram_olmoe_token_close(explicit_all_heads);
    engram_olmoe_token_close(mixed_heads);
    engram_olmoe_token_close(explicit_all_heads_v2_zero);
    engram_olmoe_token_close(explicit_all_heads_v2_biased);
    return fail("additive episodic reset was not deterministic");
  }
  if (gated_metrics.slots_written != 2 ||
      gated_metrics.read_events != 2 ||
      gated_metrics.active_slots != 2 ||
      gated_metrics.entries_read != 4 ||
      gated_metrics.write_bytes != 32 ||
      gated_metrics.key_read_bytes != 16 ||
      gated_metrics.value_read_bytes != 16 ||
      gated_metrics.duplicate_older_entries_suppressed != 3 ||
      gated_metrics.state_bytes != 520 ||
      gated_metrics.scratch_bytes != 104) {
    engram_olmoe_token_close(legacy_all_heads);
    engram_olmoe_token_close(explicit_all_heads);
    engram_olmoe_token_close(mixed_heads);
    engram_olmoe_token_close(explicit_all_heads_v2_zero);
    engram_olmoe_token_close(explicit_all_heads_v2_biased);
    return fail("mixed head-gated episodic counters are inexact");
  }
  engram_olmoe_token_reset(mixed_heads);
  if (engram_olmoe_token_forward_episodic_v1(
          mixed_heads, &token, 1, &write_slot_zero, &no_write,
          &gated_next, &mixed_base, error, sizeof(error)) != 0 ||
      engram_olmoe_token_copy_episodic_metrics_v1(
          mixed_heads, &gated_metrics, error, sizeof(error)) != 0 ||
      gated_metrics.slots_written != 1 ||
      gated_metrics.read_events != 0 ||
      gated_metrics.active_slots != 1 ||
      gated_metrics.entries_read != 0 ||
      gated_metrics.write_bytes != 16 ||
      gated_metrics.key_read_bytes != 0 ||
      gated_metrics.value_read_bytes != 0 ||
      gated_metrics.duplicate_older_entries_suppressed != 0 ||
      gated_metrics.state_bytes != 520 ||
      gated_metrics.scratch_bytes != 104) {
    engram_olmoe_token_close(legacy_all_heads);
    engram_olmoe_token_close(explicit_all_heads);
    engram_olmoe_token_close(mixed_heads);
    engram_olmoe_token_close(explicit_all_heads_v2_zero);
    engram_olmoe_token_close(explicit_all_heads_v2_biased);
    return fail("mixed head-gated episodic reset retained state");
  }
  engram_olmoe_token_close(legacy_all_heads);
  engram_olmoe_token_close(explicit_all_heads);
  engram_olmoe_token_close(mixed_heads);
  engram_olmoe_token_close(explicit_all_heads_v2_zero);
  engram_olmoe_token_close(explicit_all_heads_v2_biased);

  const engram_olmoe_attention_policy_v1 shadow_policy{
      .local_window = 4,
      .older_candidates = 2,
      .older_top_k = 1,
      .sink_tokens = 1,
  };
  engram_olmoe_attention_policy_v1 invalid_shadow_policy =
      shadow_policy;
  invalid_shadow_policy.local_window = 0;
  if (engram_olmoe_token_open_episodic_shadow_trace_v1(
          &config, &episodic_policy, all_episodic_heads.data(),
          all_episodic_heads.size(), 0.5F, nullptr, error,
          sizeof(error)) != nullptr ||
      std::string(error).find("shadow") == std::string::npos ||
      engram_olmoe_token_open_episodic_shadow_trace_v1(
          &config, &episodic_policy, all_episodic_heads.data(),
          all_episodic_heads.size(), 0.5F, &invalid_shadow_policy,
          error, sizeof(error)) != nullptr ||
      std::string(error).find("shadow") == std::string::npos ||
      engram_olmoe_token_open_episodic_shadow_trace_v1(
          &config, &episodic_policy, nullptr,
          all_episodic_heads.size(), 0.5F, &shadow_policy, error,
          sizeof(error)) != nullptr ||
      engram_olmoe_token_open_episodic_shadow_trace_v1(
          &config, &episodic_policy, all_episodic_heads.data(),
          all_episodic_heads.size(),
          std::numeric_limits<float>::quiet_NaN(), &shadow_policy,
          error, sizeof(error)) != nullptr) {
    return fail("shadow trace open accepted invalid storage or policies");
  }
  auto short_shadow_policy = shadow_policy;
  short_shadow_policy.local_window = 2;
  void* short_shadow =
      engram_olmoe_token_open_episodic_shadow_trace_v1(
          &config, &episodic_policy, all_episodic_heads.data(),
          all_episodic_heads.size(), 0.5F, &short_shadow_policy, error,
          sizeof(error));
  std::int64_t short_shadow_next = -1;
  engram_olmoe_token_metrics short_shadow_metrics{};
  if (short_shadow == nullptr ||
      engram_olmoe_token_forward_episodic_v1(
          short_shadow, episodic_tokens.data(), episodic_tokens.size(),
          episodic_writes.data(), episodic_reads.data(),
          &short_shadow_next, &short_shadow_metrics, error,
          sizeof(error)) == 0 ||
      std::string(error).find("local window") == std::string::npos ||
      engram_olmoe_token_position(short_shadow) != 0) {
    engram_olmoe_token_close(short_shadow);
    return fail(
        "short shadow local window did not reject the batched source ledger");
  }
  engram_olmoe_token_close(short_shadow);

  void* shadow_reference =
      engram_olmoe_token_open_episodic_headwise_v2(
          &config, &episodic_policy, all_episodic_heads.data(),
          all_episodic_heads.size(), 0.5F, error, sizeof(error));
  void* shadow_trace =
      engram_olmoe_token_open_episodic_shadow_trace_v1(
          &config, &episodic_policy, all_episodic_heads.data(),
          all_episodic_heads.size(), 0.5F, &shadow_policy, error,
          sizeof(error));
  if (shadow_reference == nullptr || shadow_trace == nullptr) {
    engram_olmoe_token_close(shadow_reference);
    engram_olmoe_token_close(shadow_trace);
    return fail(std::string("shadow trace runtime open failed: ") + error);
  }

  constexpr std::size_t shadow_trace_count = 2 * 4;
  std::array<float, shadow_trace_count> shadow_input{};
  std::array<float, shadow_trace_count> shadow_base{};
  std::array<float, shadow_trace_count> shadow_target{};
  constexpr std::size_t shadow_mass_count = 2 * 2;
  std::array<float, shadow_trace_count> mass_base_pre_wo{};
  std::array<float, shadow_trace_count> mass_regular_component{};
  std::array<float, shadow_trace_count> mass_episodic_component{};
  std::array<float, shadow_mass_count> mass_regular{};
  std::array<float, shadow_mass_count> mass_episodic{};
  std::array<float, shadow_mass_count> mass_shadow_source{};
  constexpr std::size_t episodic_slot_mass_count = 2 * 2 * 2;
  constexpr std::size_t episodic_slot_value_count =
      episodic_slot_mass_count * 2;
  std::array<float, episodic_slot_mass_count> episodic_slot_mass{};
  std::array<float, episodic_slot_value_count> episodic_slot_values{};
  if (engram_olmoe_token_copy_last_shadow_trace_v1(
          shadow_trace, shadow_input.data(), shadow_input.size(),
          shadow_base.data(), shadow_base.size(), shadow_target.data(),
          shadow_target.size(), error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_shadow_trace_v1(
          shadow_reference, shadow_input.data(), shadow_input.size(),
          shadow_base.data(), shadow_base.size(), shadow_target.data(),
          shadow_target.size(), error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_episodic_mass_trace_v1(
          shadow_trace, mass_base_pre_wo.data(),
          mass_base_pre_wo.size(), mass_regular_component.data(),
          mass_regular_component.size(),
          mass_episodic_component.data(),
          mass_episodic_component.size(), mass_regular.data(),
          mass_regular.size(), mass_episodic.data(),
          mass_episodic.size(), mass_shadow_source.data(),
          mass_shadow_source.size(), error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_episodic_slot_trace_v1(
          shadow_trace, episodic_slot_mass.data(),
          episodic_slot_mass.size(), episodic_slot_values.data(),
          episodic_slot_values.size(), error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_episodic_slot_trace_v1(
          shadow_reference, episodic_slot_mass.data(),
          episodic_slot_mass.size(), episodic_slot_values.data(),
          episodic_slot_values.size(), error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_episodic_mass_trace_v1(
          shadow_reference, mass_base_pre_wo.data(),
          mass_base_pre_wo.size(), mass_regular_component.data(),
          mass_regular_component.size(),
          mass_episodic_component.data(),
          mass_episodic_component.size(), mass_regular.data(),
          mass_regular.size(), mass_episodic.data(),
          mass_episodic.size(), mass_shadow_source.data(),
          mass_shadow_source.size(), error, sizeof(error)) == 0) {
    engram_olmoe_token_close(shadow_reference);
    engram_olmoe_token_close(shadow_trace);
    return fail("shadow trace copy succeeded before a trace was valid");
  }

  std::int64_t shadow_reference_next = -1;
  std::int64_t shadow_trace_next = -1;
  engram_olmoe_token_metrics shadow_reference_base{};
  engram_olmoe_token_metrics shadow_trace_base{};
  engram_olmoe_attention_metrics_v1 shadow_reference_attention{};
  engram_olmoe_attention_metrics_v1 shadow_trace_attention{};
  engram_olmoe_episodic_metrics_v1 shadow_reference_episodic{};
  engram_olmoe_episodic_metrics_v1 shadow_trace_episodic{};
  std::array<float, 4> shadow_reference_state{};
  std::array<float, 4> shadow_trace_state{};
  std::array<float, 3> shadow_reference_logits{};
  std::array<float, 3> shadow_trace_logits{};
  std::array<float, shadow_trace_count> before_read_target{};
  std::array<float, shadow_trace_count> first_read_target{};
  for (std::size_t row = 0; row < episodic_tokens.size(); ++row) {
    int mass_trace_status = 1;
    int slot_trace_status = 1;
    if (engram_olmoe_token_forward_episodic_v1(
            shadow_reference, &episodic_tokens[row], 1,
            &episodic_writes[row], &episodic_reads[row],
            &shadow_reference_next, &shadow_reference_base, error,
            sizeof(error)) != 0 ||
        engram_olmoe_token_forward_episodic_v1(
            shadow_trace, &episodic_tokens[row], 1,
            &episodic_writes[row], &episodic_reads[row],
            &shadow_trace_next, &shadow_trace_base, error,
            sizeof(error)) != 0 ||
        engram_olmoe_token_copy_attention_metrics_v1(
            shadow_reference, &shadow_reference_attention, error,
            sizeof(error)) != 0 ||
        engram_olmoe_token_copy_attention_metrics_v1(
            shadow_trace, &shadow_trace_attention, error,
            sizeof(error)) != 0 ||
        engram_olmoe_token_copy_episodic_metrics_v1(
            shadow_reference, &shadow_reference_episodic, error,
            sizeof(error)) != 0 ||
        engram_olmoe_token_copy_episodic_metrics_v1(
            shadow_trace, &shadow_trace_episodic, error,
            sizeof(error)) != 0 ||
        copy_diagnostics(shadow_reference, shadow_reference_state,
                         shadow_reference_logits, error,
                         sizeof(error)) != 0 ||
        copy_diagnostics(shadow_trace, shadow_trace_state,
                         shadow_trace_logits, error, sizeof(error)) != 0 ||
        engram_olmoe_token_copy_last_shadow_trace_v1(
            shadow_trace, shadow_input.data(), shadow_input.size(),
            shadow_base.data(), shadow_base.size(), shadow_target.data(),
            shadow_target.size(), error, sizeof(error)) != 0 ||
        ((mass_trace_status =
              engram_olmoe_token_copy_last_episodic_mass_trace_v1(
                  shadow_trace, mass_base_pre_wo.data(),
                  mass_base_pre_wo.size(),
                  mass_regular_component.data(),
                  mass_regular_component.size(),
                  mass_episodic_component.data(),
                  mass_episodic_component.size(),
                  mass_regular.data(), mass_regular.size(),
                  mass_episodic.data(), mass_episodic.size(),
                  mass_shadow_source.data(),
                  mass_shadow_source.size(), error,
                  sizeof(error))) == 0) !=
            (episodic_reads[row] >= 0) ||
        ((slot_trace_status =
              engram_olmoe_token_copy_last_episodic_slot_trace_v1(
                  shadow_trace, episodic_slot_mass.data(),
                  episodic_slot_mass.size(),
                  episodic_slot_values.data(),
                  episodic_slot_values.size(), error,
                  sizeof(error))) == 0) !=
            (episodic_reads[row] >= 0) ||
        shadow_reference_next != shadow_trace_next ||
        shadow_reference_state != shadow_trace_state ||
        shadow_reference_logits != shadow_trace_logits ||
        !same_base_counters(shadow_reference_base, shadow_trace_base) ||
        !same_attention_metrics(shadow_reference_attention,
                                shadow_trace_attention) ||
        std::memcmp(&shadow_reference_episodic, &shadow_trace_episodic,
                    sizeof(shadow_reference_episodic)) != 0 ||
        !all_finite(shadow_input) || !all_finite(shadow_base) ||
        !all_finite(shadow_target) ||
        (mass_trace_status == 0 &&
         (!all_finite(mass_base_pre_wo) ||
          !all_finite(mass_regular_component) ||
          !all_finite(mass_episodic_component) ||
          !all_finite(mass_regular) ||
          !all_finite(mass_episodic) ||
          !all_finite(mass_shadow_source))) ||
        (slot_trace_status == 0 &&
         (!all_finite(episodic_slot_mass) ||
          !all_finite(episodic_slot_values)))) {
      engram_olmoe_token_close(shadow_reference);
      engram_olmoe_token_close(shadow_trace);
      return fail("shadow trace changed base behavior or emitted bad data");
    }
    if (mass_trace_status == 0) {
      for (std::size_t index = 0; index < shadow_trace_count;
           ++index) {
        if (std::abs(mass_regular_component[index] +
                         mass_episodic_component[index] -
                     mass_base_pre_wo[index]) >
            2.0e-6F) {
          engram_olmoe_token_close(shadow_reference);
          engram_olmoe_token_close(shadow_trace);
          return fail(
              "episodic mass components do not reconstruct base pre-Wo");
        }
      }
      for (std::size_t index = 0; index < shadow_mass_count;
           ++index) {
        if (std::abs(
                mass_regular[index] + mass_episodic[index] - 1.0F) >
                2.0e-6F ||
            mass_regular[index] <= 0.0F ||
            mass_episodic[index] <= 0.0F ||
            mass_shadow_source[index] <= 0.0F ||
            mass_shadow_source[index] >= 1.0F) {
          engram_olmoe_token_close(shadow_reference);
          engram_olmoe_token_close(shadow_trace);
          return fail("episodic or shadow source masses are invalid");
        }
      }
      for (std::size_t layer = 0; layer < 2; ++layer) {
        for (std::size_t head = 0; head < 2; ++head) {
          const std::size_t mass_index = layer * 2 + head;
          const std::size_t slot_begin = mass_index * 2;
          const float slot_mass_sum =
              episodic_slot_mass[slot_begin] +
              episodic_slot_mass[slot_begin + 1];
          if (std::abs(slot_mass_sum -
                       mass_episodic[mass_index]) > 2.0e-6F) {
            engram_olmoe_token_close(shadow_reference);
            engram_olmoe_token_close(shadow_trace);
            return fail(
                "episodic slot masses do not reconstruct episodic mass");
          }
          for (std::size_t dimension = 0; dimension < 2; ++dimension) {
            float component = 0.0F;
            for (std::size_t slot = 0; slot < 2; ++slot) {
              const std::size_t slot_index = slot_begin + slot;
              component +=
                  episodic_slot_mass[slot_index] *
                  episodic_slot_values[slot_index * 2 + dimension];
            }
            const std::size_t component_index =
                layer * 4 + head * 2 + dimension;
            if (std::abs(component -
                         mass_episodic_component[component_index]) >
                2.0e-6F) {
              engram_olmoe_token_close(shadow_reference);
              engram_olmoe_token_close(shadow_trace);
              return fail(
                  "episodic slot values do not reconstruct the component");
            }
          }
        }
      }
    }
    if (row == 1) before_read_target = shadow_target;
    if (row == 2) {
      first_read_target = shadow_target;
      const float expected_source_mass =
          expected_first_read_layer_zero_shadow_source_mass(
              config.rms_norm_epsilon);
      if (std::abs(mass_shadow_source[0] -
                   expected_source_mass) > 2.0e-6F ||
          shadow_target == before_read_target ||
          std::none_of(
              shadow_target.begin(), shadow_target.end(),
              [](const float value) { return value != 0.0F; })) {
        engram_olmoe_token_close(shadow_reference);
        engram_olmoe_token_close(shadow_trace);
        return fail(
            "shadow target or scheduled-source mass is wrong on a read row");
      }
    }
  }
  const std::array<float, 4> expected_first_read =
      expected_first_read_layer_zero_target(
          config.rms_norm_epsilon, 0.5F);
  for (std::size_t index = 0; index < expected_first_read.size();
       ++index) {
    if (std::abs(first_read_target[index] -
                 expected_first_read[index]) > 1.0e-5F) {
      engram_olmoe_token_close(shadow_reference);
      engram_olmoe_token_close(shadow_trace);
      return fail(
          "shadow residual sign/order did not match the independent "
          "layer-zero oracle");
    }
  }
  const float expected_last_row_input =
      1.0F / std::sqrt(1.0F + config.rms_norm_epsilon);
  if (!std::all_of(
          shadow_input.begin(), shadow_input.begin() + 4,
          [expected_last_row_input](const float value) {
            return std::abs(value - expected_last_row_input) <= 1.0e-6F;
          })) {
    engram_olmoe_token_close(shadow_reference);
    engram_olmoe_token_close(shadow_trace);
    return fail(
        "shadow trace did not start with the last row's layer-zero input");
  }

  void* shadow_batch =
      engram_olmoe_token_open_episodic_shadow_trace_v1(
          &config, &episodic_policy, all_episodic_heads.data(),
          all_episodic_heads.size(), 0.5F, &shadow_policy, error,
          sizeof(error));
  std::int64_t shadow_batch_next = -1;
  engram_olmoe_token_metrics shadow_batch_base{};
  engram_olmoe_attention_metrics_v1 shadow_batch_attention{};
  engram_olmoe_episodic_metrics_v1 shadow_batch_episodic{};
  std::array<float, 4> shadow_batch_state{};
  std::array<float, 3> shadow_batch_logits{};
  std::array<float, shadow_trace_count> shadow_batch_input{};
  std::array<float, shadow_trace_count> shadow_batch_projected{};
  std::array<float, shadow_trace_count> shadow_batch_target{};
  std::array<float, shadow_trace_count> shadow_batch_mass_base{};
  std::array<float, shadow_trace_count> shadow_batch_mass_regular_component{};
  std::array<float, shadow_trace_count> shadow_batch_mass_episodic_component{};
  std::array<float, shadow_mass_count> shadow_batch_mass_regular{};
  std::array<float, shadow_mass_count> shadow_batch_mass_episodic{};
  std::array<float, shadow_mass_count> shadow_batch_mass_source{};
  std::array<float, episodic_slot_mass_count> shadow_batch_slot_mass{};
  std::array<float, episodic_slot_value_count> shadow_batch_slot_values{};
  if (shadow_batch == nullptr ||
      engram_olmoe_token_forward_episodic_v1(
          shadow_batch, episodic_tokens.data(), episodic_tokens.size(),
          episodic_writes.data(), episodic_reads.data(),
          &shadow_batch_next, &shadow_batch_base, error,
          sizeof(error)) != 0 ||
      engram_olmoe_token_copy_attention_metrics_v1(
          shadow_batch, &shadow_batch_attention, error,
          sizeof(error)) != 0 ||
      engram_olmoe_token_copy_episodic_metrics_v1(
          shadow_batch, &shadow_batch_episodic, error,
          sizeof(error)) != 0 ||
      copy_diagnostics(shadow_batch, shadow_batch_state,
                       shadow_batch_logits, error, sizeof(error)) != 0 ||
      engram_olmoe_token_copy_last_shadow_trace_v1(
          shadow_batch, shadow_batch_input.data(),
          shadow_batch_input.size(), shadow_batch_projected.data(),
          shadow_batch_projected.size(), shadow_batch_target.data(),
          shadow_batch_target.size(), error, sizeof(error)) != 0 ||
      engram_olmoe_token_copy_last_episodic_mass_trace_v1(
          shadow_batch, shadow_batch_mass_base.data(),
          shadow_batch_mass_base.size(),
          shadow_batch_mass_regular_component.data(),
          shadow_batch_mass_regular_component.size(),
          shadow_batch_mass_episodic_component.data(),
          shadow_batch_mass_episodic_component.size(),
          shadow_batch_mass_regular.data(),
          shadow_batch_mass_regular.size(),
          shadow_batch_mass_episodic.data(),
          shadow_batch_mass_episodic.size(),
          shadow_batch_mass_source.data(),
          shadow_batch_mass_source.size(), error, sizeof(error)) != 0 ||
      engram_olmoe_token_copy_last_episodic_slot_trace_v1(
          shadow_batch, shadow_batch_slot_mass.data(),
          shadow_batch_slot_mass.size(),
          shadow_batch_slot_values.data(),
          shadow_batch_slot_values.size(), error, sizeof(error)) != 0 ||
      shadow_batch_next != shadow_trace_next ||
      shadow_batch_state != shadow_trace_state ||
      shadow_batch_logits != shadow_trace_logits ||
      !same_base_counters(shadow_batch_base, shadow_trace_base) ||
      !same_attention_metrics(shadow_batch_attention,
                              shadow_trace_attention) ||
      std::memcmp(&shadow_batch_episodic, &shadow_trace_episodic,
                  sizeof(shadow_batch_episodic)) != 0 ||
      shadow_batch_input != shadow_input ||
      shadow_batch_projected != shadow_base ||
      shadow_batch_target != shadow_target ||
      shadow_batch_mass_base != mass_base_pre_wo ||
      shadow_batch_mass_regular_component !=
          mass_regular_component ||
      shadow_batch_mass_episodic_component !=
          mass_episodic_component ||
      shadow_batch_mass_regular != mass_regular ||
      shadow_batch_mass_episodic != mass_episodic ||
      shadow_batch_mass_source != mass_shadow_source ||
      shadow_batch_slot_mass != episodic_slot_mass ||
      shadow_batch_slot_values != episodic_slot_values) {
    engram_olmoe_token_close(shadow_batch);
    engram_olmoe_token_close(shadow_reference);
    engram_olmoe_token_close(shadow_trace);
    return fail(
        "batched shadow trace did not match singleton last-row trace");
  }
  engram_olmoe_token_close(shadow_batch);

  if (engram_olmoe_token_copy_last_shadow_trace_v1(
          shadow_trace, nullptr, shadow_input.size(),
          shadow_base.data(), shadow_base.size(), shadow_target.data(),
          shadow_target.size(), error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_shadow_trace_v1(
          shadow_trace, shadow_input.data(), shadow_input.size() - 1,
          shadow_base.data(), shadow_base.size(), shadow_target.data(),
          shadow_target.size(), error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_shadow_trace_v1(
          shadow_trace, shadow_input.data(), shadow_input.size(), nullptr,
          shadow_base.size(), shadow_target.data(), shadow_target.size(),
          error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_shadow_trace_v1(
          shadow_trace, shadow_input.data(), shadow_input.size(),
          shadow_base.data(), shadow_base.size() - 1,
          shadow_target.data(), shadow_target.size(), error,
          sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_shadow_trace_v1(
          shadow_trace, shadow_input.data(), shadow_input.size(),
          shadow_base.data(), shadow_base.size(), nullptr,
          shadow_target.size(), error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_shadow_trace_v1(
          shadow_trace, shadow_input.data(), shadow_input.size(),
          shadow_base.data(), shadow_base.size(), shadow_target.data(),
          shadow_target.size() - 1, error, sizeof(error)) == 0) {
    engram_olmoe_token_close(shadow_reference);
    engram_olmoe_token_close(shadow_trace);
    return fail("shadow trace copy accepted inexact output storage");
  }
  if (engram_olmoe_token_copy_last_episodic_mass_trace_v1(
          shadow_trace, nullptr, mass_base_pre_wo.size(),
          mass_regular_component.data(),
          mass_regular_component.size(),
          mass_episodic_component.data(),
          mass_episodic_component.size(), mass_regular.data(),
          mass_regular.size(), mass_episodic.data(),
          mass_episodic.size(), mass_shadow_source.data(),
          mass_shadow_source.size(), error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_episodic_mass_trace_v1(
          shadow_trace, mass_base_pre_wo.data(),
          mass_base_pre_wo.size() - 1,
          mass_regular_component.data(),
          mass_regular_component.size(),
          mass_episodic_component.data(),
          mass_episodic_component.size(), mass_regular.data(),
          mass_regular.size(), mass_episodic.data(),
          mass_episodic.size(), mass_shadow_source.data(),
          mass_shadow_source.size(), error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_episodic_mass_trace_v1(
          shadow_trace, mass_base_pre_wo.data(),
          mass_base_pre_wo.size(), mass_regular_component.data(),
          mass_regular_component.size() - 1,
          mass_episodic_component.data(),
          mass_episodic_component.size(), mass_regular.data(),
          mass_regular.size(), mass_episodic.data(),
          mass_episodic.size(), mass_shadow_source.data(),
          mass_shadow_source.size(), error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_episodic_mass_trace_v1(
          shadow_trace, mass_base_pre_wo.data(),
          mass_base_pre_wo.size(), mass_regular_component.data(),
          mass_regular_component.size(),
          mass_episodic_component.data(),
          mass_episodic_component.size() - 1,
          mass_regular.data(), mass_regular.size(),
          mass_episodic.data(), mass_episodic.size(),
          mass_shadow_source.data(), mass_shadow_source.size(),
          error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_episodic_mass_trace_v1(
          shadow_trace, mass_base_pre_wo.data(),
          mass_base_pre_wo.size(), mass_regular_component.data(),
          mass_regular_component.size(),
          mass_episodic_component.data(),
          mass_episodic_component.size(), mass_regular.data(),
          mass_regular.size() - 1, mass_episodic.data(),
          mass_episodic.size(), mass_shadow_source.data(),
          mass_shadow_source.size(), error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_episodic_mass_trace_v1(
          shadow_trace, mass_base_pre_wo.data(),
          mass_base_pre_wo.size(), mass_regular_component.data(),
          mass_regular_component.size(),
          mass_episodic_component.data(),
          mass_episodic_component.size(), mass_regular.data(),
          mass_regular.size(), mass_episodic.data(),
          mass_episodic.size() - 1, mass_shadow_source.data(),
          mass_shadow_source.size(), error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_episodic_mass_trace_v1(
          shadow_trace, mass_base_pre_wo.data(),
          mass_base_pre_wo.size(), mass_regular_component.data(),
          mass_regular_component.size(),
          mass_episodic_component.data(),
          mass_episodic_component.size(), mass_regular.data(),
          mass_regular.size(), mass_episodic.data(),
          mass_episodic.size(), nullptr, mass_shadow_source.size(),
          error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_episodic_mass_trace_v1(
          shadow_trace, mass_base_pre_wo.data(),
          mass_base_pre_wo.size(), mass_regular_component.data(),
          mass_regular_component.size(),
          mass_episodic_component.data(),
          mass_episodic_component.size(), mass_regular.data(),
          mass_regular.size(), mass_episodic.data(),
          mass_episodic.size(), mass_shadow_source.data(),
          mass_shadow_source.size() - 1, error, sizeof(error)) == 0) {
    engram_olmoe_token_close(shadow_reference);
    engram_olmoe_token_close(shadow_trace);
    return fail(
        "episodic mass trace copy accepted inexact output storage");
  }
  if (engram_olmoe_token_copy_last_episodic_slot_trace_v1(
          shadow_trace, nullptr, episodic_slot_mass.size(),
          episodic_slot_values.data(), episodic_slot_values.size(),
          error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_episodic_slot_trace_v1(
          shadow_trace, episodic_slot_mass.data(),
          episodic_slot_mass.size() - 1, episodic_slot_values.data(),
          episodic_slot_values.size(), error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_episodic_slot_trace_v1(
          shadow_trace, episodic_slot_mass.data(),
          episodic_slot_mass.size(), nullptr,
          episodic_slot_values.size(), error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_episodic_slot_trace_v1(
          shadow_trace, episodic_slot_mass.data(),
          episodic_slot_mass.size(), episodic_slot_values.data(),
          episodic_slot_values.size() - 1, error, sizeof(error)) == 0) {
    engram_olmoe_token_close(shadow_reference);
    engram_olmoe_token_close(shadow_trace);
    return fail(
        "episodic slot trace copy accepted inexact output storage");
  }

  const auto first_shadow_input = shadow_input;
  const auto first_shadow_base = shadow_base;
  const auto first_shadow_target = shadow_target;
  const auto first_shadow_state = shadow_trace_state;
  const auto first_shadow_logits = shadow_trace_logits;
  const std::int64_t first_shadow_next = shadow_trace_next;
  const auto first_shadow_base_counters = shadow_trace_base;
  const auto first_shadow_attention = shadow_trace_attention;
  const auto first_shadow_episodic = shadow_trace_episodic;
  const auto first_mass_base_pre_wo = mass_base_pre_wo;
  const auto first_mass_regular_component =
      mass_regular_component;
  const auto first_mass_episodic_component =
      mass_episodic_component;
  const auto first_mass_regular = mass_regular;
  const auto first_mass_episodic = mass_episodic;
  const auto first_mass_shadow_source = mass_shadow_source;
  const auto first_slot_mass = episodic_slot_mass;
  const auto first_slot_values = episodic_slot_values;

  engram_olmoe_token_reset(shadow_reference);
  engram_olmoe_token_reset(shadow_trace);
  if (engram_olmoe_token_position(shadow_reference) != 0 ||
      engram_olmoe_token_position(shadow_trace) != 0 ||
      engram_olmoe_token_copy_last_shadow_trace_v1(
          shadow_trace, shadow_input.data(), shadow_input.size(),
          shadow_base.data(), shadow_base.size(), shadow_target.data(),
          shadow_target.size(), error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_episodic_mass_trace_v1(
          shadow_trace, mass_base_pre_wo.data(),
          mass_base_pre_wo.size(), mass_regular_component.data(),
          mass_regular_component.size(),
          mass_episodic_component.data(),
          mass_episodic_component.size(), mass_regular.data(),
          mass_regular.size(), mass_episodic.data(),
          mass_episodic.size(), mass_shadow_source.data(),
          mass_shadow_source.size(), error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_episodic_slot_trace_v1(
          shadow_trace, episodic_slot_mass.data(),
          episodic_slot_mass.size(), episodic_slot_values.data(),
          episodic_slot_values.size(), error, sizeof(error)) == 0) {
    engram_olmoe_token_close(shadow_reference);
    engram_olmoe_token_close(shadow_trace);
    return fail("shadow reset did not invalidate the trace");
  }

  for (std::size_t row = 0; row < episodic_tokens.size(); ++row) {
    if (engram_olmoe_token_forward_episodic_v1(
            shadow_reference, &episodic_tokens[row], 1,
            &episodic_writes[row], &episodic_reads[row],
            &shadow_reference_next, &shadow_reference_base, error,
            sizeof(error)) != 0 ||
        engram_olmoe_token_forward_episodic_v1(
            shadow_trace, &episodic_tokens[row], 1,
            &episodic_writes[row], &episodic_reads[row],
            &shadow_trace_next, &shadow_trace_base, error,
            sizeof(error)) != 0) {
      engram_olmoe_token_close(shadow_reference);
      engram_olmoe_token_close(shadow_trace);
      return fail(std::string("shadow reset replay failed: ") + error);
    }
  }
  if (engram_olmoe_token_copy_attention_metrics_v1(
          shadow_reference, &shadow_reference_attention, error,
          sizeof(error)) != 0 ||
      engram_olmoe_token_copy_attention_metrics_v1(
          shadow_trace, &shadow_trace_attention, error,
          sizeof(error)) != 0 ||
      engram_olmoe_token_copy_episodic_metrics_v1(
          shadow_reference, &shadow_reference_episodic, error,
          sizeof(error)) != 0 ||
      engram_olmoe_token_copy_episodic_metrics_v1(
          shadow_trace, &shadow_trace_episodic, error,
          sizeof(error)) != 0 ||
      copy_diagnostics(shadow_reference, shadow_reference_state,
                       shadow_reference_logits, error,
                       sizeof(error)) != 0 ||
      copy_diagnostics(shadow_trace, shadow_trace_state,
                       shadow_trace_logits, error, sizeof(error)) != 0 ||
      engram_olmoe_token_copy_last_shadow_trace_v1(
          shadow_trace, shadow_input.data(), shadow_input.size(),
          shadow_base.data(), shadow_base.size(), shadow_target.data(),
          shadow_target.size(), error, sizeof(error)) != 0 ||
      engram_olmoe_token_copy_last_episodic_mass_trace_v1(
          shadow_trace, mass_base_pre_wo.data(),
          mass_base_pre_wo.size(), mass_regular_component.data(),
          mass_regular_component.size(),
          mass_episodic_component.data(),
          mass_episodic_component.size(), mass_regular.data(),
          mass_regular.size(), mass_episodic.data(),
          mass_episodic.size(), mass_shadow_source.data(),
          mass_shadow_source.size(), error, sizeof(error)) != 0 ||
      engram_olmoe_token_copy_last_episodic_slot_trace_v1(
          shadow_trace, episodic_slot_mass.data(),
          episodic_slot_mass.size(), episodic_slot_values.data(),
          episodic_slot_values.size(), error, sizeof(error)) != 0 ||
      shadow_reference_next != shadow_trace_next ||
      shadow_trace_next != first_shadow_next ||
      shadow_reference_state != shadow_trace_state ||
      shadow_trace_state != first_shadow_state ||
      shadow_reference_logits != shadow_trace_logits ||
      shadow_trace_logits != first_shadow_logits ||
      !same_base_counters(shadow_reference_base, shadow_trace_base) ||
      !same_base_counters(shadow_trace_base,
                          first_shadow_base_counters) ||
      !same_attention_metrics(shadow_reference_attention,
                              shadow_trace_attention) ||
      !same_attention_metrics(shadow_trace_attention,
                              first_shadow_attention) ||
      std::memcmp(&shadow_reference_episodic, &shadow_trace_episodic,
                  sizeof(shadow_reference_episodic)) != 0 ||
      std::memcmp(&shadow_trace_episodic, &first_shadow_episodic,
                  sizeof(shadow_trace_episodic)) != 0 ||
      shadow_input != first_shadow_input ||
      shadow_base != first_shadow_base ||
      shadow_target != first_shadow_target ||
      mass_base_pre_wo != first_mass_base_pre_wo ||
      mass_regular_component != first_mass_regular_component ||
      mass_episodic_component !=
          first_mass_episodic_component ||
      mass_regular != first_mass_regular ||
      mass_episodic != first_mass_episodic ||
      mass_shadow_source != first_mass_shadow_source ||
      episodic_slot_mass != first_slot_mass ||
      episodic_slot_values != first_slot_values) {
    engram_olmoe_token_close(shadow_reference);
    engram_olmoe_token_close(shadow_trace);
    return fail("shadow trace reset replay was not exact");
  }
  engram_olmoe_token_close(shadow_reference);
  engram_olmoe_token_close(shadow_trace);

  auto regular_trace_config = config;
  regular_trace_config.local_window =
      ENGRAM_OLMOE_REGULAR_TRACE_LOCAL_ENTRIES_V1;
  regular_trace_config.older_candidates = 8;
  regular_trace_config.older_top_k =
      ENGRAM_OLMOE_REGULAR_TRACE_OLDER_ENTRIES_V1;
  regular_trace_config.sink_tokens = 2;
  void* regular_trace_reference =
      engram_olmoe_token_open_episodic_headwise_v2(
          &regular_trace_config, &episodic_policy,
          all_episodic_heads.data(), all_episodic_heads.size(), 0.5F,
          error, sizeof(error));
  void* regular_trace_runtime =
      engram_olmoe_token_open_episodic_shadow_trace_v1(
          &regular_trace_config, &episodic_policy,
          all_episodic_heads.data(), all_episodic_heads.size(), 0.5F,
          &shadow_policy, error, sizeof(error));
  constexpr std::size_t regular_entry_count =
      2 * 2 * ENGRAM_OLMOE_REGULAR_TRACE_ENTRIES_V1;
  constexpr std::size_t regular_entry_value_count =
      regular_entry_count * 2;
  std::array<float, regular_entry_count> regular_entry_mass{};
  std::array<float, regular_entry_value_count> regular_entry_values{};
  std::array<std::uint8_t, regular_entry_count>
      regular_entry_valid_kind{};
  std::array<std::uint64_t, regular_entry_count>
      regular_entry_positions{};
  if (regular_trace_reference == nullptr ||
      regular_trace_runtime == nullptr ||
      engram_olmoe_token_copy_last_regular_entry_trace_v1(
          regular_trace_runtime, regular_entry_mass.data(),
          regular_entry_mass.size(), regular_entry_values.data(),
          regular_entry_values.size(),
          regular_entry_valid_kind.data(),
          regular_entry_valid_kind.size(),
          regular_entry_positions.data(), regular_entry_positions.size(),
          error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_regular_entry_trace_v1(
          regular_trace_reference, regular_entry_mass.data(),
          regular_entry_mass.size(), regular_entry_values.data(),
          regular_entry_values.size(),
          regular_entry_valid_kind.data(),
          regular_entry_valid_kind.size(),
          regular_entry_positions.data(), regular_entry_positions.size(),
          error, sizeof(error)) == 0) {
    engram_olmoe_token_close(regular_trace_reference);
    engram_olmoe_token_close(regular_trace_runtime);
    return fail(
        "regular-entry trace open or pre-forward validity was not fail-closed");
  }
  std::int64_t regular_trace_reference_next = -1;
  std::int64_t regular_trace_next = -1;
  engram_olmoe_token_metrics regular_trace_reference_base{};
  engram_olmoe_token_metrics regular_trace_base{};
  engram_olmoe_attention_metrics_v1 regular_trace_reference_attention{};
  engram_olmoe_attention_metrics_v1 regular_trace_attention{};
  engram_olmoe_episodic_metrics_v1 regular_trace_reference_episodic{};
  engram_olmoe_episodic_metrics_v1 regular_trace_episodic{};
  std::array<float, 4> regular_trace_reference_state{};
  std::array<float, 4> regular_trace_state{};
  std::array<float, 3> regular_trace_reference_logits{};
  std::array<float, 3> regular_trace_logits{};
  if (engram_olmoe_token_forward_episodic_v1(
          regular_trace_reference, episodic_tokens.data(),
          episodic_tokens.size(), episodic_writes.data(),
          episodic_reads.data(), &regular_trace_reference_next,
          &regular_trace_reference_base, error, sizeof(error)) != 0 ||
      engram_olmoe_token_forward_episodic_v1(
          regular_trace_runtime, episodic_tokens.data(),
          episodic_tokens.size(), episodic_writes.data(),
          episodic_reads.data(), &regular_trace_next,
          &regular_trace_base, error, sizeof(error)) != 0 ||
      engram_olmoe_token_copy_attention_metrics_v1(
          regular_trace_reference, &regular_trace_reference_attention,
          error, sizeof(error)) != 0 ||
      engram_olmoe_token_copy_attention_metrics_v1(
          regular_trace_runtime, &regular_trace_attention, error,
          sizeof(error)) != 0 ||
      engram_olmoe_token_copy_episodic_metrics_v1(
          regular_trace_reference, &regular_trace_reference_episodic,
          error, sizeof(error)) != 0 ||
      engram_olmoe_token_copy_episodic_metrics_v1(
          regular_trace_runtime, &regular_trace_episodic, error,
          sizeof(error)) != 0 ||
      copy_diagnostics(
          regular_trace_reference, regular_trace_reference_state,
          regular_trace_reference_logits, error, sizeof(error)) != 0 ||
      copy_diagnostics(
          regular_trace_runtime, regular_trace_state,
          regular_trace_logits, error, sizeof(error)) != 0 ||
      engram_olmoe_token_copy_last_episodic_mass_trace_v1(
          regular_trace_runtime, mass_base_pre_wo.data(),
          mass_base_pre_wo.size(), mass_regular_component.data(),
          mass_regular_component.size(),
          mass_episodic_component.data(),
          mass_episodic_component.size(), mass_regular.data(),
          mass_regular.size(), mass_episodic.data(),
          mass_episodic.size(), mass_shadow_source.data(),
          mass_shadow_source.size(), error, sizeof(error)) != 0 ||
      engram_olmoe_token_copy_last_regular_entry_trace_v1(
          regular_trace_runtime, regular_entry_mass.data(),
          regular_entry_mass.size(), regular_entry_values.data(),
          regular_entry_values.size(),
          regular_entry_valid_kind.data(),
          regular_entry_valid_kind.size(),
          regular_entry_positions.data(), regular_entry_positions.size(),
          error, sizeof(error)) != 0 ||
      regular_trace_reference_next != regular_trace_next ||
      regular_trace_reference_state != regular_trace_state ||
      regular_trace_reference_logits != regular_trace_logits ||
      !same_base_counters(regular_trace_reference_base,
                          regular_trace_base) ||
      !same_attention_metrics(regular_trace_reference_attention,
                              regular_trace_attention) ||
      std::memcmp(&regular_trace_reference_episodic,
                  &regular_trace_episodic,
                  sizeof(regular_trace_episodic)) != 0) {
    engram_olmoe_token_close(regular_trace_reference);
    engram_olmoe_token_close(regular_trace_runtime);
    return fail(
        "regular-entry runtime trace changed output, counters, or ABI");
  }
  for (std::size_t layer = 0; layer < 2; ++layer) {
    for (std::size_t head = 0; head < 2; ++head) {
      const std::size_t head_offset =
          (layer * 2 + head) *
          ENGRAM_OLMOE_REGULAR_TRACE_ENTRIES_V1;
      float reconstructed_mass = 0.0F;
      std::array<float, 2> reconstructed_component{};
      for (std::size_t entry = 0;
           entry < ENGRAM_OLMOE_REGULAR_TRACE_ENTRIES_V1; ++entry) {
        const std::size_t flat = head_offset + entry;
        if (entry < episodic_tokens.size()) {
          if (regular_entry_valid_kind[flat] !=
                  ENGRAM_OLMOE_REGULAR_TRACE_LOCAL_V1 ||
              regular_entry_positions[flat] != entry ||
              regular_entry_mass[flat] <= 0.0F) {
            engram_olmoe_token_close(regular_trace_reference);
            engram_olmoe_token_close(regular_trace_runtime);
            return fail(
                "regular-entry C trace local order is not chronological");
          }
        } else if (regular_entry_valid_kind[flat] !=
                       ENGRAM_OLMOE_REGULAR_TRACE_INVALID_V1 ||
                   regular_entry_positions[flat] !=
                       std::numeric_limits<std::uint64_t>::max() ||
                   regular_entry_mass[flat] != 0.0F ||
                   regular_entry_values[flat * 2] != 0.0F ||
                   regular_entry_values[flat * 2 + 1] != 0.0F) {
          engram_olmoe_token_close(regular_trace_reference);
          engram_olmoe_token_close(regular_trace_runtime);
          return fail(
              "regular-entry C trace padding is not canonical");
        }
        reconstructed_mass += regular_entry_mass[flat];
        reconstructed_component[0] +=
            regular_entry_mass[flat] *
            regular_entry_values[flat * 2];
        reconstructed_component[1] +=
            regular_entry_mass[flat] *
            regular_entry_values[flat * 2 + 1];
      }
      const std::size_t mass_index = layer * 2 + head;
      const std::size_t component_index = layer * 4 + head * 2;
      if (std::abs(reconstructed_mass - mass_regular[mass_index]) >
              2.0e-6F ||
          std::abs(reconstructed_component[0] -
                   mass_regular_component[component_index]) > 2.0e-6F ||
          std::abs(reconstructed_component[1] -
                   mass_regular_component[component_index + 1]) >
              2.0e-6F) {
        engram_olmoe_token_close(regular_trace_reference);
        engram_olmoe_token_close(regular_trace_runtime);
        return fail(
            "regular-entry C trace did not reconstruct regular attention");
      }
    }
  }
  if (engram_olmoe_token_copy_last_regular_entry_trace_v1(
          regular_trace_runtime, regular_entry_mass.data(),
          regular_entry_mass.size() - 1, regular_entry_values.data(),
          regular_entry_values.size(),
          regular_entry_valid_kind.data(),
          regular_entry_valid_kind.size(),
          regular_entry_positions.data(), regular_entry_positions.size(),
          error, sizeof(error)) == 0 ||
      engram_olmoe_token_copy_last_regular_entry_trace_v1(
          regular_trace_runtime, nullptr, regular_entry_mass.size(),
          regular_entry_values.data(), regular_entry_values.size(),
          regular_entry_valid_kind.data(),
          regular_entry_valid_kind.size(),
          regular_entry_positions.data(), regular_entry_positions.size(),
          error, sizeof(error)) == 0) {
    engram_olmoe_token_close(regular_trace_reference);
    engram_olmoe_token_close(regular_trace_runtime);
    return fail(
        "regular-entry C trace accepted inexact or null storage");
  }
  const auto first_regular_entry_mass = regular_entry_mass;
  const auto first_regular_entry_values = regular_entry_values;
  const auto first_regular_entry_valid_kind =
      regular_entry_valid_kind;
  const auto first_regular_entry_positions = regular_entry_positions;
  engram_olmoe_token_reset(regular_trace_runtime);
  if (engram_olmoe_token_copy_last_regular_entry_trace_v1(
          regular_trace_runtime, regular_entry_mass.data(),
          regular_entry_mass.size(), regular_entry_values.data(),
          regular_entry_values.size(),
          regular_entry_valid_kind.data(),
          regular_entry_valid_kind.size(),
          regular_entry_positions.data(), regular_entry_positions.size(),
          error, sizeof(error)) == 0 ||
      engram_olmoe_token_forward_episodic_v1(
          regular_trace_runtime, episodic_tokens.data(),
          episodic_tokens.size(), episodic_writes.data(),
          episodic_reads.data(), &regular_trace_next,
          &regular_trace_base, error, sizeof(error)) != 0 ||
      engram_olmoe_token_copy_last_regular_entry_trace_v1(
          regular_trace_runtime, regular_entry_mass.data(),
          regular_entry_mass.size(), regular_entry_values.data(),
          regular_entry_values.size(),
          regular_entry_valid_kind.data(),
          regular_entry_valid_kind.size(),
          regular_entry_positions.data(), regular_entry_positions.size(),
          error, sizeof(error)) != 0 ||
      regular_entry_mass != first_regular_entry_mass ||
      regular_entry_values != first_regular_entry_values ||
      regular_entry_valid_kind != first_regular_entry_valid_kind ||
      regular_entry_positions != first_regular_entry_positions) {
    engram_olmoe_token_close(regular_trace_reference);
    engram_olmoe_token_close(regular_trace_runtime);
    return fail("regular-entry C trace reset replay was not exact");
  }
  engram_olmoe_token_close(regular_trace_reference);
  engram_olmoe_token_close(regular_trace_runtime);
  return 0;
}
