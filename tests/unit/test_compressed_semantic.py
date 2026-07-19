import numpy as np
import pytest

from engram.semantic.compressed import CompressedSemanticLayer
from engram.semantic.quantization import (
    decode_additive_vectors,
    decode_scalar_affine,
    encode_additive_vectors,
    encode_scalar_affine,
)
from engram.semantic.router import JointKeyRouter


def test_quantized_runtime_matches_decoded_reference_routing_and_read():
    rng = np.random.default_rng(915)
    gate = rng.normal(size=(24, 6)).astype(np.float32)
    up = rng.normal(size=(24, 6)).astype(np.float32)
    values = rng.normal(size=(24, 6)).astype(np.float32)
    hidden = rng.normal(size=6).astype(np.float32)
    gate_encoding = encode_scalar_affine(gate, bits=8)
    up_encoding = encode_scalar_affine(up, bits=8)
    value_encoding = encode_additive_vectors(
        values, num_codebooks=2, codebook_size=8, iterations=8
    )
    layer = CompressedSemanticLayer.from_encodings(
        gate_encoding, up_encoding, value_encoding
    )

    decoded_gate = decode_scalar_affine(gate_encoding)
    decoded_up = decode_scalar_affine(up_encoding)
    decoded_values = decode_additive_vectors(value_encoding)
    router = JointKeyRouter(
        decoded_gate,
        decoded_up,
        values=decoded_values,
        candidate_count=10,
        top_k=4,
    )
    reference = router.route(hidden)
    result = layer.read(hidden, candidate_count=10, top_k=4)

    np.testing.assert_allclose(layer.decode_gate(), decoded_gate, atol=2e-6, rtol=0.0)
    np.testing.assert_allclose(layer.decode_up(), decoded_up, atol=2e-6, rtol=0.0)
    np.testing.assert_array_equal(layer.decode_values(), decoded_values)
    np.testing.assert_array_equal(result.candidate_indices, reference.candidate_indices)
    np.testing.assert_array_equal(result.indices, reference.selected_indices)
    np.testing.assert_allclose(
        result.candidate_activations, reference.candidate_activations, rtol=2e-6, atol=2e-7
    )
    expected_output = (
        reference.selected_activations
        @ decoded_values[reference.selected_indices].astype(np.float64)
    )
    np.testing.assert_allclose(result.output, expected_output, rtol=2e-6, atol=2e-7)
    assert result.active_records == 4
    assert result.candidate_records == 10
    assert result.estimated_bytes_read > 0
    assert not hasattr(layer, "gate_keys")
    assert not hasattr(layer, "up_keys")
    assert not hasattr(layer, "values")


def test_quantized_directory_roundtrip_uses_explicit_codec_arrays(tmp_path):
    rng = np.random.default_rng(81)
    layer = CompressedSemanticLayer.compress(
        rng.normal(size=(16, 5)),
        rng.normal(size=(16, 5)),
        rng.normal(size=(16, 5)),
        key_bits=9,
        value_codebooks=2,
        value_codebook_size=6,
        iterations=5,
    )
    hidden = rng.normal(size=5)
    expected = layer.read(hidden, candidate_count=9, top_k=3)
    directory = layer.save(tmp_path / "layer-0000" / "quantized")
    loaded = CompressedSemanticLayer.load(directory)
    actual = loaded.read(hidden, candidate_count=9, top_k=3)

    for name in (
        "gate_codes",
        "gate_offsets",
        "gate_scales",
        "up_codes",
        "up_offsets",
        "up_scales",
        "value_codes",
        "value_codebooks",
    ):
        assert (directory / f"{name}.npy").is_file()
        np.testing.assert_array_equal(getattr(loaded, name), getattr(layer, name))
    assert (directory / "codecs.json").is_file()
    assert isinstance(loaded.gate_codes, np.memmap)
    assert loaded.gate_metadata == layer.gate_metadata
    assert loaded.up_metadata == layer.up_metadata
    assert loaded.value_metadata == layer.value_metadata
    np.testing.assert_array_equal(actual.candidate_indices, expected.candidate_indices)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_allclose(actual.output, expected.output, rtol=0.0, atol=0.0)
    assert loaded.compressed_bytes == layer.compressed_bytes


def test_joint_proxy_and_validation_without_exact_arrays():
    gate = np.array([[1.0, 0.0], [0.8, 0.6], [0.2, 0.98]], dtype=np.float32)
    up = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.866]], dtype=np.float32)
    values = np.eye(3, 2, dtype=np.float32)
    layer = CompressedSemanticLayer.compress(
        gate,
        up,
        values,
        key_bits=8,
        value_codebooks=1,
        value_codebook_size=3,
    )
    result = layer.read([1.0, 0.0], candidate_count=1, top_k=1)
    assert result.candidate_indices.tolist() == [1]

    with pytest.raises(ValueError, match="top_k must not exceed"):
        layer.read([1.0, 0.0], candidate_count=1, top_k=2)
    with pytest.raises(ValueError, match="shape"):
        layer.read([1.0], candidate_count=1, top_k=1)
