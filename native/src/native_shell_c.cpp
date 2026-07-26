#include "engram/native_shell_c.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace {

float bf16_to_float(const std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::uint16_t float_to_bf16(const float value) {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t rounding_bias = 0x7FFFU + ((bits >> 16U) & 1U);
  bits += rounding_bias;
  return static_cast<std::uint16_t>(bits >> 16U);
}

}  // namespace

extern "C" int engram_embedding_lookup_bf16(
    const std::uint16_t* table, const std::size_t rows,
    const std::size_t width, const std::int64_t* token_ids,
    const std::size_t token_count, std::uint16_t* output) {
  if (table == nullptr || token_ids == nullptr || output == nullptr ||
      rows == 0 || width == 0) {
    return 1;
  }
  for (std::size_t token = 0; token < token_count; ++token) {
    const std::int64_t index = token_ids[token];
    if (index < 0 || static_cast<std::size_t>(index) >= rows) {
      return 2;
    }
    std::memcpy(output + token * width,
                table + static_cast<std::size_t>(index) * width,
                width * sizeof(std::uint16_t));
  }
  return 0;
}

extern "C" int engram_rms_norm_f32_to_bf16(
    const float* input, const std::uint16_t* weight,
    const std::size_t vectors, const std::size_t width, const float epsilon,
    std::uint16_t* output) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      vectors == 0 || width == 0 || !(epsilon >= 0.0F)) {
    return 1;
  }
  for (std::size_t vector = 0; vector < vectors; ++vector) {
    const float* input_row = input + vector * width;
    std::uint16_t* output_row = output + vector * width;
    float sum_squares = 0.0F;
    for (std::size_t column = 0; column < width; ++column) {
      const float rounded = bf16_to_float(float_to_bf16(input_row[column]));
      sum_squares += rounded * rounded;
    }
    const float inverse_rms =
        1.0F / std::sqrt(sum_squares / static_cast<float>(width) + epsilon);
    for (std::size_t column = 0; column < width; ++column) {
      const float rounded = bf16_to_float(float_to_bf16(input_row[column]));
      const float normalized =
          bf16_to_float(float_to_bf16(rounded * inverse_rms));
      output_row[column] =
          float_to_bf16(bf16_to_float(weight[column]) * normalized);
    }
  }
  return 0;
}

extern "C" int engram_vocab_argmax_bf16(
    const std::uint16_t* input, const std::uint16_t* weight,
    const std::size_t vocabulary, const std::size_t width,
    const std::size_t threads, std::int64_t* output_token,
    float* output_score) {
  if (input == nullptr || weight == nullptr || output_token == nullptr ||
      vocabulary == 0 || width == 0 || threads == 0) {
    return 1;
  }
  const std::size_t workers = std::min(threads, vocabulary);
  struct Best {
    float score = -std::numeric_limits<float>::infinity();
    std::size_t token = 0;
  };
  std::vector<Best> best(workers);
  std::vector<std::thread> pool;
  pool.reserve(workers);
  for (std::size_t worker = 0; worker < workers; ++worker) {
    pool.emplace_back([&, worker]() {
      const std::size_t begin = vocabulary * worker / workers;
      const std::size_t end = vocabulary * (worker + 1) / workers;
      Best local;
      local.token = begin;
      for (std::size_t token = begin; token < end; ++token) {
        const std::uint16_t* row = weight + token * width;
        float score = 0.0F;
        for (std::size_t column = 0; column < width; ++column) {
          score += bf16_to_float(input[column]) *
                   bf16_to_float(row[column]);
        }
        // CPU BitNet's BF16 linear returns BF16 logits. Compare scores only
        // after the same output rounding so near-ties choose the same token.
        score = bf16_to_float(float_to_bf16(score));
        if (score > local.score) {
          local.score = score;
          local.token = token;
        }
      }
      best[worker] = local;
    });
  }
  for (std::thread& worker : pool) worker.join();
  Best result;
  for (const Best candidate : best) {
    if (candidate.score > result.score ||
        (candidate.score == result.score && candidate.token < result.token)) {
      result = candidate;
    }
  }
  *output_token = static_cast<std::int64_t>(result.token);
  if (output_score != nullptr) *output_score = result.score;
  return 0;
}

extern "C" int engram_rope_bf16(
    std::uint16_t* query, const std::size_t query_heads, std::uint16_t* key,
    const std::size_t key_heads, const std::size_t batch,
    const std::size_t length, const std::size_t width,
    const std::int64_t* positions, const std::size_t position_rows,
    const float theta) {
  if (query == nullptr || key == nullptr || positions == nullptr ||
      query_heads == 0 || key_heads == 0 || batch == 0 || length == 0 ||
      width == 0 || width % 2 != 0 ||
      (position_rows != 1 && position_rows != batch) || !(theta > 0.0F)) {
    return 1;
  }
  const std::size_t half = width / 2;
  const auto rotate = [&](std::uint16_t* values, const std::size_t heads) {
    for (std::size_t row = 0; row < batch; ++row) {
      const std::size_t position_row = position_rows == 1 ? 0 : row;
      for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t token = 0; token < length; ++token) {
          std::uint16_t* vector =
              values + ((row * heads + head) * length + token) * width;
          const float position =
              static_cast<float>(positions[position_row * length + token]);
          for (std::size_t column = 0; column < half; ++column) {
            const float exponent =
                (2.0F * static_cast<float>(column)) /
                static_cast<float>(width);
            const float frequency = position / std::pow(theta, exponent);
            const float cosine =
                bf16_to_float(float_to_bf16(std::cos(frequency)));
            const float sine =
                bf16_to_float(float_to_bf16(std::sin(frequency)));
            const float first = bf16_to_float(vector[column]);
            const float second = bf16_to_float(vector[column + half]);
            const float rotated_first = bf16_to_float(float_to_bf16(
                bf16_to_float(float_to_bf16(first * cosine)) +
                bf16_to_float(float_to_bf16(-second * sine))));
            const float rotated_second = bf16_to_float(float_to_bf16(
                bf16_to_float(float_to_bf16(second * cosine)) +
                bf16_to_float(float_to_bf16(first * sine))));
            vector[column] = float_to_bf16(rotated_first);
            vector[column + half] = float_to_bf16(rotated_second);
          }
        }
      }
    }
  };
  rotate(query, query_heads);
  rotate(key, key_heads);
  return 0;
}
