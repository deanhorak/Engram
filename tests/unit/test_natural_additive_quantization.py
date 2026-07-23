from dataclasses import replace

import numpy as np
import pytest

from engram.semantic.natural_additive_quantization import (
    NaturalAdditiveEncoding,
    NaturalAdditiveQuantizationError,
    build_natural_additive_encoding,
    decode_natural_additive,
    deserialize_natural_additive,
    fit_natural_additive,
    load_natural_additive,
    natural_additive_storage_plan,
    save_natural_additive,
    serialize_natural_additive,
)


def test_smollm_natural_layout_charges_every_byte_and_passes_cold_gate():
    plan = natural_additive_storage_plan()

    assert plan == {
        "gate_shape": (1536, 576),
        "up_shape": (1536, 576),
        "down_shape": (576, 1536),
        "groups": (72, 72, 192),
        "group_size": 8,
        "num_stages": 2,
        "code_bits": 7,
        "header_and_checksum": 256,
        "packed_codes": 580_608,
        "codebooks": 12_288,
        "packed_scale_indices": 2_736,
        "scale_codebooks": 384,
        "total": 596_272,
        "dense_q4_bytes": 1_327_104,
        "fraction_of_dense_q4": 596_272 / 1_327_104,
        "effective_bits_per_weight": 596_272 * 8 / (3 * 1536 * 576),
    }
    assert plan["fraction_of_dense_q4"] == pytest.approx(0.44930314429012347)
    assert plan["fraction_of_dense_q4"] < 0.45
    assert 0.45 * plan["dense_q4_bytes"] - plan["total"] == pytest.approx(
        924.8
    )


def test_fit_is_deterministic_rebuildable_and_reload_decodes_identically(tmp_path):
    rng = np.random.default_rng(20260722)
    gate = rng.normal(size=(11, 7)).astype(np.float32)
    up = rng.normal(size=(11, 7)).astype(np.float32)
    down = rng.normal(size=(7, 11)).astype(np.float32)
    gate[0] = 0.0
    kwargs = {"iterations": 3, "sample_limit": None, "seed": 19}

    first = fit_natural_additive(gate, up, down, **kwargs)
    second = fit_natural_additive(gate, up, down, **kwargs)

    assert first.metadata.groups == (1, 1, 2)
    assert first.metadata.code_shapes == ((11, 2, 1), (11, 2, 1), (7, 2, 2))
    assert first.codebooks.shape == (3, 2, 128, 8)
    assert first.scale_codebooks.shape == (3, 64)
    assert first.packed_codes.nbytes == ((22 + 22 + 28) * 7 + 7) // 8
    assert first.packed_scale_indices.nbytes == ((11 + 11 + 7) * 6 + 7) // 8
    np.testing.assert_array_equal(first.packed_codes, second.packed_codes)
    np.testing.assert_array_equal(first.codebooks, second.codebooks)
    np.testing.assert_array_equal(
        first.packed_scale_indices, second.packed_scale_indices
    )
    np.testing.assert_array_equal(first.scale_codebooks, second.scale_codebooks)

    codes = first.unpack_codes()
    scales = first.unpack_scale_indices()
    rebuilt = build_natural_additive_encoding(
        first.metadata,
        codes=codes,
        codebooks=first.codebooks,
        scale_indices=scales,
        scale_codebooks=first.scale_codebooks,
    )
    assert serialize_natural_additive(rebuilt) == serialize_natural_additive(first)

    artifact = serialize_natural_additive(first)
    assert artifact == serialize_natural_additive(second)
    assert len(artifact) == first.storage_bytes
    restored = deserialize_natural_additive(artifact)
    for restored_matrix, first_matrix in zip(
        decode_natural_additive(restored), decode_natural_additive(first), strict=True
    ):
        np.testing.assert_array_equal(restored_matrix, first_matrix)
    decoded_gate, decoded_up, decoded_down = decode_natural_additive(restored)
    assert decoded_gate.shape == gate.shape
    assert decoded_up.shape == up.shape
    assert decoded_down.shape == down.shape
    np.testing.assert_array_equal(decoded_gate[0], 0.0)

    path = tmp_path / "natural-weights.naaq"
    checksum = save_natural_additive(path, first)
    assert len(checksum) == 64
    assert path.stat().st_size == first.storage_bytes
    for loaded_matrix, first_matrix in zip(
        decode_natural_additive(load_natural_additive(path)),
        decode_natural_additive(first),
        strict=True,
    ):
        np.testing.assert_array_equal(loaded_matrix, first_matrix)


def test_grouping_uses_each_linear_operators_actual_input_dimension():
    rng = np.random.default_rng(8)
    hidden, intermediate = 10, 18
    encoding = fit_natural_additive(
        rng.normal(size=(intermediate, hidden)).astype(np.float32),
        rng.normal(size=(intermediate, hidden)).astype(np.float32),
        rng.normal(size=(hidden, intermediate)).astype(np.float32),
        iterations=1,
        sample_limit=32,
        seed=4,
    )

    # Gate/up group their H-wide input axis; down groups its I-wide input axis.
    assert encoding.metadata.groups == (2, 2, 3)
    gate_codes, up_codes, down_codes = encoding.unpack_codes()
    assert gate_codes.shape == (intermediate, 2, 2)
    assert up_codes.shape == (intermediate, 2, 2)
    assert down_codes.shape == (hidden, 2, 3)
    assert encoding.metadata.code_count == (18 * 2 + 18 * 2 + 10 * 3) * 2


def test_rejects_checksum_padding_metadata_and_array_corruption():
    rng = np.random.default_rng(88)
    encoding = fit_natural_additive(
        rng.normal(size=(2, 1)).astype(np.float32),
        rng.normal(size=(2, 1)).astype(np.float32),
        rng.normal(size=(1, 2)).astype(np.float32),
        iterations=1,
        sample_limit=None,
        seed=2,
    )
    artifact = bytearray(serialize_natural_additive(encoding))
    artifact[-1] ^= 0x01
    with pytest.raises(NaturalAdditiveQuantizationError, match="checksum mismatch"):
        deserialize_natural_additive(artifact)
    with pytest.raises(NaturalAdditiveQuantizationError, match="length"):
        deserialize_natural_additive(serialize_natural_additive(encoding)[:-1])

    # Ten 7-bit codes leave two padding bits in the final byte.
    assert encoding.metadata.code_count == 10
    corrupted_codes = encoding.packed_codes.copy()
    corrupted_codes[-1] |= np.uint8(0xC0)
    with pytest.raises(NaturalAdditiveQuantizationError, match="padding bits"):
        replace(encoding, packed_codes=corrupted_codes).validate()

    bad_metadata = replace(encoding.metadata, down_shape=(1, 3))
    with pytest.raises(NaturalAdditiveQuantizationError, match="down_shape"):
        replace(encoding, metadata=bad_metadata).validate()

    bad_codebooks = encoding.codebooks.copy()
    bad_codebooks[0, 0, 0, 0] = np.nan
    with pytest.raises(NaturalAdditiveQuantizationError, match="finite"):
        replace(encoding, codebooks=bad_codebooks).validate()

    bad_scales = encoding.scale_codebooks.copy()
    bad_scales[0, 0] = -1.0
    with pytest.raises(NaturalAdditiveQuantizationError, match="non-negative"):
        replace(encoding, scale_codebooks=bad_scales).validate()

    codes = encoding.unpack_codes()
    with pytest.raises(NaturalAdditiveQuantizationError, match="up_codes"):
        build_natural_additive_encoding(
            encoding.metadata,
            codes=(codes[0], codes[1][..., :-1], codes[2]),
            codebooks=encoding.codebooks,
            scale_indices=encoding.unpack_scale_indices(),
            scale_codebooks=encoding.scale_codebooks,
        )


def test_rejects_nonfinite_or_incompatible_source_matrices():
    gate = np.ones((4, 3), dtype=np.float32)
    up = np.ones((4, 3), dtype=np.float32)
    down = np.ones((3, 4), dtype=np.float32)
    with pytest.raises(NaturalAdditiveQuantizationError, match="finite"):
        fit_natural_additive(gate, up * np.nan, down)
    with pytest.raises(NaturalAdditiveQuantizationError, match="identical"):
        fit_natural_additive(gate, np.ones((5, 3), dtype=np.float32), down)
    with pytest.raises(NaturalAdditiveQuantizationError, match="down_shape"):
        fit_natural_additive(gate, up, np.ones((3, 5), dtype=np.float32))


def test_encoding_type_rejects_wrong_packed_storage_dtype():
    rng = np.random.default_rng(9)
    encoding = fit_natural_additive(
        rng.normal(size=(3, 2)).astype(np.float32),
        rng.normal(size=(3, 2)).astype(np.float32),
        rng.normal(size=(2, 3)).astype(np.float32),
        iterations=1,
        sample_limit=None,
    )
    malformed = NaturalAdditiveEncoding(
        packed_codes=encoding.packed_codes.astype(np.int16),
        codebooks=encoding.codebooks,
        packed_scale_indices=encoding.packed_scale_indices,
        scale_codebooks=encoding.scale_codebooks,
        metadata=encoding.metadata,
    )
    with pytest.raises(NaturalAdditiveQuantizationError, match="uint8"):
        malformed.validate()
