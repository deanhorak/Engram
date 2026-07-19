import json
from dataclasses import replace

import numpy as np
import pytest

from engram.semantic.quantization import (
    AdditiveVectorEncoding,
    AdditiveVectorMetadata,
    QuantizationError,
    ScalarAffineEncoding,
    ScalarAffineMetadata,
    decode_additive_vectors,
    decode_scalar_affine,
    encode_additive_vectors,
    encode_scalar_affine,
)


def test_scalar_affine_is_deterministic_bounded_and_metadata_round_trips():
    rng = np.random.default_rng(41)
    keys = rng.normal(size=(37, 9)).astype(np.float32)
    keys[:, 3] = 2.75
    first = encode_scalar_affine(keys, bits=8)
    second = encode_scalar_affine(keys, bits=8)

    np.testing.assert_array_equal(first.codes, second.codes)
    assert first.metadata == second.metadata
    assert first.codes.dtype == np.uint8
    restored_metadata = ScalarAffineMetadata.from_dict(
        json.loads(json.dumps(first.metadata.to_dict()))
    )
    decoded = decode_scalar_affine(ScalarAffineEncoding(first.codes, restored_metadata))

    assert restored_metadata.shape == keys.shape
    assert restored_metadata.codec == "scalar_affine_per_dimension"
    assert restored_metadata.schema_version == 1
    assert restored_metadata.byte_order == "little"
    np.testing.assert_array_equal(decoded[:, 3], keys[:, 3])
    error = np.abs(decoded.astype(np.float64) - keys)
    bounds = np.asarray(restored_metadata.scales) / 2.0 + 2e-7
    assert np.all(error <= bounds)


def test_scalar_affine_uses_wider_codes_and_rejects_corruption():
    keys = np.linspace(-3.0, 5.0, 48, dtype=np.float32).reshape(8, 6)
    encoded = encode_scalar_affine(keys, bits=9)
    assert encoded.codes.dtype == np.uint16

    with pytest.raises(QuantizationError, match="finite"):
        encode_scalar_affine([[0.0, np.nan]])
    with pytest.raises(QuantizationError, match="bits"):
        encode_scalar_affine(keys, bits=17)
    with pytest.raises(QuantizationError, match="code shape"):
        decode_scalar_affine(ScalarAffineEncoding(encoded.codes[:-1], encoded.metadata))

    corrupted = encoded.codes.copy()
    corrupted[0, 0] = 512
    with pytest.raises(QuantizationError, match="bit range"):
        decode_scalar_affine(ScalarAffineEncoding(corrupted, encoded.metadata))
    with pytest.raises(QuantizationError, match="storage_dtype"):
        replace(encoded.metadata, storage_dtype="uint8").validate()


def test_vector_codebook_can_represent_training_vectors_exactly():
    values = np.array(
        [[-2.0, 1.0, 0.5], [0.0, 0.0, 0.0], [3.0, -1.0, 4.0], [1.5, 2.5, -3.0]],
        dtype=np.float32,
    )
    encoded = encode_additive_vectors(
        values, num_codebooks=1, codebook_size=len(values), iterations=5
    )
    decoded = decode_additive_vectors(encoded)
    np.testing.assert_allclose(decoded, values, rtol=0.0, atol=0.0)
    assert encoded.metadata.codec == "additive_residual_vector_codebook"
    assert encoded.metadata.num_codebooks == 1
    assert encoded.metadata.byte_order == "little"


def test_additive_codebooks_are_deterministic_and_reduce_residual_error():
    rng = np.random.default_rng(72)
    left = rng.normal(size=(64, 6)) @ rng.normal(size=(6, 12))
    values = left.astype(np.float32)
    vector = encode_additive_vectors(values, num_codebooks=1, codebook_size=8)
    additive = encode_additive_vectors(values, num_codebooks=3, codebook_size=8)
    repeated = encode_additive_vectors(values, num_codebooks=3, codebook_size=8)

    np.testing.assert_array_equal(additive.codes, repeated.codes)
    np.testing.assert_array_equal(additive.codebooks, repeated.codebooks)
    vector_error = np.linalg.norm(values - decode_additive_vectors(vector))
    additive_error = np.linalg.norm(values - decode_additive_vectors(additive))
    assert additive_error < vector_error * 0.75
    assert additive.codes.shape == (64, 3)
    assert additive.codebooks.shape == (3, 8, 12)
    assert int(np.max(additive.codes)) < 8

    metadata = AdditiveVectorMetadata.from_dict(
        json.loads(json.dumps(additive.metadata.to_dict()))
    )
    np.testing.assert_array_equal(
        decode_additive_vectors(
            AdditiveVectorEncoding(additive.codes, additive.codebooks, metadata)
        ),
        decode_additive_vectors(additive),
    )


def test_additive_codec_rejects_invalid_inputs_and_encoded_arrays():
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    with pytest.raises(QuantizationError, match="codebook_size"):
        encode_additive_vectors(values, codebook_size=7)
    with pytest.raises(QuantizationError, match="positive integer"):
        encode_additive_vectors(values, num_codebooks=0)
    with pytest.raises(QuantizationError, match="finite"):
        encode_additive_vectors([[1.0, np.inf]])

    encoded = encode_additive_vectors(values, num_codebooks=2, codebook_size=4)
    wrong_codes = encoded.codes.copy()
    wrong_codes[0, 0] = 4
    with pytest.raises(QuantizationError, match="outside its codebook"):
        decode_additive_vectors(
            AdditiveVectorEncoding(wrong_codes, encoded.codebooks, encoded.metadata)
        )
    wrong_codebooks = encoded.codebooks[:, :, :-1]
    with pytest.raises(QuantizationError, match="codebook shape"):
        decode_additive_vectors(
            AdditiveVectorEncoding(encoded.codes, wrong_codebooks, encoded.metadata)
        )
    with pytest.raises(QuantizationError, match="code_dtype"):
        replace(encoded.metadata, code_dtype="uint16").validate()
