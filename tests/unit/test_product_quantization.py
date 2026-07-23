from dataclasses import replace

import numpy as np
import pytest

from engram.semantic.product_quantization import (
    ProductAdditiveEncoding,
    ProductAdditiveQuantizationError,
    decode_product_additive,
    fit_product_additive,
)


def test_product_additive_is_deterministic_packed_and_exactly_accounted():
    rng = np.random.default_rng(2026)
    matrix = rng.normal(size=(32, 10)).astype(np.float32)
    first = fit_product_additive(
        matrix,
        group_size=4,
        num_codebooks=2,
        codebook_size=8,
        iterations=4,
        sample_limit=24,
        seed=17,
        per_record_scale=True,
    )
    second = fit_product_additive(
        matrix,
        group_size=4,
        num_codebooks=2,
        codebook_size=8,
        iterations=4,
        sample_limit=24,
        seed=17,
        per_record_scale=True,
    )

    np.testing.assert_array_equal(first.packed_codes, second.packed_codes)
    np.testing.assert_array_equal(first.codebooks, second.codebooks)
    np.testing.assert_array_equal(first.record_scales, second.record_scales)
    assert first.metadata == second.metadata
    assert first.metadata.groups == 3
    assert first.metadata.padded_width == 12
    assert first.metadata.code_bits == 3
    assert first.code_count == 32 * 2 * 3
    assert first.packed_code_bits == first.code_count * 3
    assert first.packed_code_bytes == (first.packed_code_bits + 7) // 8
    expected_codebook_bytes = 2 * 8 * 4 * np.dtype(np.float16).itemsize
    expected_scale_bytes = 32 * np.dtype(np.float16).itemsize
    assert first.storage_components() == {
        "packed_codes": first.packed_code_bytes,
        "codebooks": expected_codebook_bytes,
        "record_scales": expected_scale_bytes,
        "total": first.packed_code_bytes
        + expected_codebook_bytes
        + expected_scale_bytes,
    }
    assert first.storage_bits == first.storage_bytes * 8
    assert first.bits_per_weight == first.storage_bits / matrix.size
    assert first.unpack_codes().shape == (32, 2, 3)
    assert first.codebooks.shape == (2, 8, 4)
    assert first.codebooks.dtype == np.float16
    assert decode_product_additive(first).shape == matrix.shape


def test_fixed_width_codes_are_really_bit_packed():
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(16, 6)).astype(np.float32)
    encoding = fit_product_additive(
        matrix,
        group_size=3,
        num_codebooks=1,
        codebook_size=4,
        iterations=2,
        seed=3,
        per_record_scale=False,
    )

    # 16 records * 2 groups * 2 bits = 64 bits, not 32 uint8 codes.
    assert encoding.code_count == 32
    assert encoding.packed_code_bits == 64
    assert encoding.packed_codes.nbytes == 8
    assert encoding.unpack_codes().dtype == np.uint16
    assert int(np.max(encoding.unpack_codes())) < 4


def test_additive_residual_stage_reduces_training_reconstruction_error():
    rng = np.random.default_rng(91)
    left = rng.normal(size=(48, 3))
    right = rng.normal(size=(3, 12))
    matrix = (left @ right).astype(np.float32)
    vector = fit_product_additive(
        matrix,
        group_size=4,
        num_codebooks=1,
        codebook_size=8,
        iterations=6,
        seed=11,
        per_record_scale=True,
    )
    additive = fit_product_additive(
        matrix,
        group_size=4,
        num_codebooks=2,
        codebook_size=8,
        iterations=6,
        seed=11,
        per_record_scale=True,
    )

    vector_error = np.linalg.norm(matrix - decode_product_additive(vector))
    additive_error = np.linalg.norm(matrix - decode_product_additive(additive))
    assert additive_error < vector_error


def test_full_codebook_can_reconstruct_records_and_zero_tail():
    matrix = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, -1.0, 0.5, 0.25, -0.25],
            [2.0, 1.0, -2.0, 0.0, 1.0],
            [-1.0, 0.5, 0.25, -0.5, 0.0],
        ],
        dtype=np.float32,
    )
    encoding = fit_product_additive(
        matrix,
        group_size=3,
        num_codebooks=1,
        codebook_size=8,
        iterations=2,
        seed=5,
        per_record_scale=True,
    )

    np.testing.assert_allclose(
        decode_product_additive(encoding), matrix, rtol=0.0, atol=1e-7
    )
    assert encoding.metadata.padded_width == 6
    assert encoding.record_scales is not None
    assert encoding.record_scales[0] == 0.0


def test_product_additive_rejects_invalid_configuration_and_payload():
    matrix = np.arange(24, dtype=np.float32).reshape(6, 4)
    with pytest.raises(ProductAdditiveQuantizationError, match="num_codebooks"):
        fit_product_additive(matrix, num_codebooks=3, codebook_size=4)
    with pytest.raises(ProductAdditiveQuantizationError, match="fitted subvectors"):
        fit_product_additive(matrix, codebook_size=8)
    with pytest.raises(ProductAdditiveQuantizationError, match="sample_limit"):
        fit_product_additive(matrix, codebook_size=4, sample_limit=3)
    with pytest.raises(ProductAdditiveQuantizationError, match="finite"):
        fit_product_additive([[0.0, np.nan]], codebook_size=2)

    encoding = fit_product_additive(
        matrix,
        group_size=4,
        num_codebooks=1,
        codebook_size=4,
        iterations=2,
        per_record_scale=False,
    )
    bad_metadata = replace(encoding.metadata, code_bits=3)
    with pytest.raises(ProductAdditiveQuantizationError, match="code_bits"):
        ProductAdditiveEncoding(
            encoding.packed_codes,
            encoding.codebooks,
            encoding.record_scales,
            bad_metadata,
        ).validate()

    corrupted = encoding.packed_codes.copy()
    corrupted[-1] |= np.uint8(0x80)
    with pytest.raises(ProductAdditiveQuantizationError, match="padding bits"):
        ProductAdditiveEncoding(
            corrupted,
            encoding.codebooks,
            encoding.record_scales,
            encoding.metadata,
        ).validate()
