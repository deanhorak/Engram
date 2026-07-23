from pathlib import Path

import numpy as np
import pytest

from engram.training.interleaved_entropy_q4_codec import (
    EntropyQ4LayerWeights,
    decode_interleaved_entropy_q4_artifact,
    interleaved_entropy_q4_dynamic_traffic,
    interleaved_entropy_q4_forward,
    load_interleaved_entropy_q4_artifact,
    save_interleaved_entropy_q4_artifact,
)


def _fixture(layer_count: int = 3):
    rng = np.random.default_rng(6061)
    hidden = 7
    source_intermediate = 13
    layers = []
    for index in range(layer_count):
        width = 4 + index % 3
        layers.append(
            EntropyQ4LayerWeights(
                rng.normal(scale=0.4, size=(width, hidden)).astype(np.float32),
                rng.normal(scale=0.4, size=(width, hidden)).astype(np.float32),
                rng.normal(scale=0.4, size=(hidden, width)).astype(np.float32),
                np.arange(index, index + width, dtype=np.int64),
            )
        )
    return layers, hidden, source_intermediate


def _save_fixture(path: Path, layer_count: int = 3):
    layers, hidden, source_intermediate = _fixture(layer_count)
    save_interleaved_entropy_q4_artifact(
        path, layers, source_intermediate_size=source_intermediate
    )
    return load_interleaved_entropy_q4_artifact(path), hidden, source_intermediate


def test_interleaved_entropy_q4_is_deterministic_and_all_record_traffic_is_file(
    tmp_path,
):
    first = tmp_path / "first.interleaved-q4.bin"
    second = tmp_path / "second.interleaved-q4.bin"
    layers, hidden, source_intermediate = _fixture(4)
    save_interleaved_entropy_q4_artifact(
        first, layers, source_intermediate_size=source_intermediate
    )
    save_interleaved_entropy_q4_artifact(
        second, layers, source_intermediate_size=source_intermediate
    )
    artifact = load_interleaved_entropy_q4_artifact(first)
    all_ids = [list(range(layer.width)) for layer in artifact.layers]
    traffic = interleaved_entropy_q4_dynamic_traffic(artifact, all_ids)

    assert first.read_bytes() == second.read_bytes()
    assert traffic["total_cold_bytes"] == first.stat().st_size
    assert traffic["serialized_artifact_bytes"] == first.stat().st_size
    assert (
        traffic["dense_q4_source_mlp_bytes"]
        == (4 * 3 * hidden * source_intermediate + 1) // 2
    )
    assert all(
        record.offset % 64 == 0 for layer in artifact.layers for record in layer.records
    )
    assert all(
        record.block_bytes % 64 == 0
        for layer in artifact.layers
        for record in layer.records
    )


def test_interleaved_entropy_q4_selected_forward_matches_full_decode(tmp_path):
    artifact, _, _ = _save_fixture(tmp_path / "interleaved-q4.bin")
    decoded = decode_interleaved_entropy_q4_artifact(artifact)
    states = np.asarray(
        [[0.2, -0.4, 0.8, 0.1, -0.3, 0.7, 0.5], [-0.5, 0.3, 0.1, 0.6, -0.2, 0.4, -0.7]],
        dtype=np.float32,
    )
    selected = np.asarray([[0, 2, 3], [1, 2, 3]], dtype=np.int64)

    actual = interleaved_entropy_q4_forward(artifact, 0, states, selected)
    gate = states @ decoded[0]["gate"].T
    activations = (gate / (1.0 + np.exp(-gate))) * (states @ decoded[0]["up"].T)
    expected = np.stack(
        [
            activations[row, ids] @ decoded[0]["down"][:, ids].T
            for row, ids in enumerate(selected)
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_interleaved_entropy_q4_rejects_encoded_and_padding_corruption(tmp_path):
    path = tmp_path / "interleaved-q4.bin"
    artifact, _, _ = _save_fixture(path)
    record = artifact.layers[0].records[0]
    payload = bytearray(path.read_bytes())
    payload[record.offset + 64] ^= 1
    corrupt_encoded = tmp_path / "corrupt-encoded.bin"
    corrupt_encoded.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum"):
        load_interleaved_entropy_q4_artifact(corrupt_encoded)

    payload = bytearray(path.read_bytes())
    payload[record.offset + record.block_bytes - 1] = 1
    corrupt_padding = tmp_path / "corrupt-padding.bin"
    corrupt_padding.write_bytes(payload)
    with pytest.raises(ValueError, match="padding"):
        load_interleaved_entropy_q4_artifact(corrupt_padding)


def test_interleaved_entropy_q4_router_bytes_are_cache_line_aligned(tmp_path):
    artifact, _, _ = _save_fixture(tmp_path / "interleaved-q4.bin")
    baseline = interleaved_entropy_q4_dynamic_traffic(artifact, 3)
    routed = interleaved_entropy_q4_dynamic_traffic(
        artifact, 3, router_bytes_per_token=65
    )

    assert routed["router_cache_aligned_bytes"] == 128
    assert routed["total_cold_bytes"] == baseline["total_cold_bytes"] + 128
