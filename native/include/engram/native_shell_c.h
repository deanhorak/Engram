#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

// Copy BF16 embedding rows selected by signed 64-bit token identifiers.
int engram_embedding_lookup_bf16(
    const std::uint16_t* table, std::size_t rows, std::size_t width,
    const std::int64_t* token_ids, std::size_t token_count,
    std::uint16_t* output);

// Match BitNetRMSNorm's operation order for a float32 controller state:
// round the input to BF16, compute the variance in float32, round the
// normalized activation to BF16, then multiply by the BF16 scale.
int engram_rms_norm_f32_to_bf16(
    const float* input, const std::uint16_t* weight, std::size_t vectors,
    std::size_t width, float epsilon, std::uint16_t* output);

// Compute only the greedy token from a tied BF16 vocabulary matrix. This
// avoids allocating and returning a full vocabulary-sized logits tensor.
int engram_vocab_argmax_bf16(
    const std::uint16_t* input, const std::uint16_t* weight,
    std::size_t vocabulary, std::size_t width, std::size_t threads,
    std::int64_t* output_token, float* output_score);

// Apply default RoPE directly to contiguous [batch, heads, length, width]
// BF16 query/key tensors. Query and key head counts may differ.
int engram_rope_bf16(
    std::uint16_t* query, std::size_t query_heads, std::uint16_t* key,
    std::size_t key_heads, std::size_t batch, std::size_t length,
    std::size_t width, const std::int64_t* positions,
    std::size_t position_rows, float theta);

}
