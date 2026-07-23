from dataclasses import replace

import numpy as np
import pytest

from engram.semantic.multiset_additive_quantization import (
    POSITION_3X4_FP16_SCALE,
    PROJECTION_2X7_SCALE6,
    MultiSetAdditiveEncoding,
    MultiSetAdditiveQuantizationError,
    decode_multiset_additive,
    deserialize_multiset_additive,
    fit_multiset_additive,
    load_multiset_additive,
    multiset_storage_plan,
    save_multiset_additive,
    serialize_multiset_additive,
)


def test_primary_projection_layout_charges_every_byte_and_fits_cold_gate():
    plan = multiset_storage_plan()

    assert plan == {
        "profile": PROJECTION_2X7_SCALE6,
        "shape": (4608, 576),
        "group_size": 8,
        "num_stages": 2,
        "code_bits": 7,
        "projection_count": 3,
        "position_buckets": 1,
        "num_codebook_sets": 3,
        "header_and_checksum": 256,
        "packed_codes": 580_608,
        "codebooks": 12_288,
        "packed_scale_indices": 3_456,
        "scale_codebooks": 384,
        "set_mapping": 0,
        "total": 596_992,
        "dense_q4_bytes": 1_327_104,
        "fraction_of_dense_q4": 596_992 / 1_327_104,
        "effective_bits_per_weight": 596_992 * 8 / (4608 * 576),
    }
    assert plan["fraction_of_dense_q4"] < 0.45
    assert 0.45 * plan["dense_q4_bytes"] - plan["total"] == pytest.approx(
        204.8
    )


def test_primary_fit_is_deterministic_and_reload_decodes_identically(tmp_path):
    rng = np.random.default_rng(20260722)
    matrix = rng.normal(size=(24, 10)).astype(np.float32)
    matrix[0] = 0.0
    kwargs = {
        "iterations": 3,
        "sample_limit": 24,
        "seed": 19,
    }
    first = fit_multiset_additive(matrix, **kwargs)
    second = fit_multiset_additive(matrix, **kwargs)

    assert first.metadata.profile == PROJECTION_2X7_SCALE6
    assert first.codebooks.shape == (3, 2, 128, 8)
    assert first.scale_codebooks is not None
    assert first.scale_codebooks.shape == (3, 64)
    assert first.row_scales is None
    assert first.set_mapping is None
    assert first.packed_scale_indices is not None
    assert first.packed_scale_indices.nbytes == (24 * 6 + 7) // 8
    assert first.packed_codes.nbytes == (24 * 2 * 2 * 7 + 7) // 8
    np.testing.assert_array_equal(first.packed_codes, second.packed_codes)
    np.testing.assert_array_equal(first.codebooks, second.codebooks)
    np.testing.assert_array_equal(
        first.packed_scale_indices, second.packed_scale_indices
    )
    np.testing.assert_array_equal(first.scale_codebooks, second.scale_codebooks)

    artifact = serialize_multiset_additive(first)
    assert artifact == serialize_multiset_additive(second)
    assert len(artifact) == first.storage_bytes
    restored = deserialize_multiset_additive(artifact)
    np.testing.assert_array_equal(restored.packed_codes, first.packed_codes)
    np.testing.assert_array_equal(restored.codebooks, first.codebooks)
    np.testing.assert_array_equal(
        restored.packed_scale_indices, first.packed_scale_indices
    )
    np.testing.assert_array_equal(restored.scale_codebooks, first.scale_codebooks)
    np.testing.assert_array_equal(restored.decoded_scales(), first.decoded_scales())
    assert restored.decoded_scales()[0] == 0.0
    np.testing.assert_array_equal(
        decode_multiset_additive(restored), decode_multiset_additive(first)
    )
    np.testing.assert_array_equal(decode_multiset_additive(restored)[0], 0.0)
    assert decode_multiset_additive(restored).shape == matrix.shape

    path = tmp_path / "joint-weights.msaq"
    checksum = save_multiset_additive(path, first)
    assert len(checksum) == 64
    assert path.stat().st_size == first.storage_bytes
    np.testing.assert_array_equal(
        decode_multiset_additive(load_multiset_additive(path)),
        decode_multiset_additive(first),
    )


def test_position_fallback_really_nibble_packs_and_accounts_for_mapping():
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(12, 16)).astype(np.float32)
    encoding = fit_multiset_additive(
        matrix,
        profile=POSITION_3X4_FP16_SCALE,
        position_buckets=1,
        iterations=2,
        sample_limit=12,
        seed=5,
    )

    assert encoding.metadata.profile == POSITION_3X4_FP16_SCALE
    assert encoding.metadata.code_count == 12 * 3 * 2
    # Exactly two four-bit codes per byte, not one uint8 per code.
    assert encoding.packed_codes.nbytes == encoding.metadata.code_count // 2
    assert encoding.codebooks.shape == (3, 3, 16, 8)
    assert encoding.row_scales is not None
    assert encoding.packed_scale_indices is None
    assert encoding.scale_codebooks is None
    assert encoding.set_mapping is not None
    assert encoding.set_mapping.nbytes == 3 * 2
    assert encoding.storage_components()["set_mapping"] == 6
    artifact = serialize_multiset_additive(encoding)
    assert len(artifact) == encoding.storage_bytes
    restored = deserialize_multiset_additive(artifact)
    np.testing.assert_array_equal(
        decode_multiset_additive(restored), decode_multiset_additive(encoding)
    )

    target_plan = multiset_storage_plan(profile=POSITION_3X4_FP16_SCALE)
    assert target_plan["total"] == 590_296
    assert target_plan["set_mapping"] == 216
    assert target_plan["fraction_of_dense_q4"] < 0.45


def test_rejects_checksum_padding_mapping_and_metadata_corruption():
    rng = np.random.default_rng(88)
    small = fit_multiset_additive(
        rng.normal(size=(3, 8)).astype(np.float32),
        iterations=1,
        sample_limit=None,
        seed=2,
    )
    artifact = bytearray(serialize_multiset_additive(small))
    artifact[-1] ^= 0x01
    with pytest.raises(MultiSetAdditiveQuantizationError, match="checksum mismatch"):
        deserialize_multiset_additive(artifact)
    with pytest.raises(MultiSetAdditiveQuantizationError, match="length"):
        deserialize_multiset_additive(serialize_multiset_additive(small)[:-1])

    # 3 records * 1 group * 2 stages * 7 bits leaves six padding bits.
    corrupted_codes = small.packed_codes.copy()
    corrupted_codes[-1] |= np.uint8(0xFC)
    with pytest.raises(MultiSetAdditiveQuantizationError, match="padding bits"):
        replace(small, packed_codes=corrupted_codes).validate()
    with pytest.raises(MultiSetAdditiveQuantizationError, match="profile fields"):
        replace(small, metadata=replace(small.metadata, scale_bits=5)).validate()

    fallback = fit_multiset_additive(
        rng.normal(size=(12, 16)).astype(np.float32),
        profile=POSITION_3X4_FP16_SCALE,
        position_buckets=1,
        iterations=1,
        sample_limit=None,
        seed=3,
    )
    assert fallback.set_mapping is not None
    bad_mapping = fallback.set_mapping.copy()
    bad_mapping[0, 0] = 2
    with pytest.raises(MultiSetAdditiveQuantizationError, match="set_mapping disagrees"):
        replace(fallback, set_mapping=bad_mapping).validate()

    bad_codebooks = fallback.codebooks[:, :, :, :-1].copy()
    with pytest.raises(MultiSetAdditiveQuantizationError, match="codebooks"):
        MultiSetAdditiveEncoding(
            packed_codes=fallback.packed_codes,
            codebooks=bad_codebooks,
            metadata=fallback.metadata,
            row_scales=fallback.row_scales,
            set_mapping=fallback.set_mapping,
        ).validate()
