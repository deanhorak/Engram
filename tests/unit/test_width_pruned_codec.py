from pathlib import Path

import numpy as np
import pytest

from engram.training.width_pruned_codec import (
    WidthPrunedQ4LayerWeights,
    decode_width_pruned_q4_artifact,
    load_width_pruned_q4_artifact,
    save_width_pruned_q4_artifact,
    width_pruned_q4_forward,
    width_pruned_q4_traffic,
)


def _fixture(layer_count: int = 3):
    rng = np.random.default_rng(2027)
    hidden = 6
    source_intermediate = 12
    widths = [3 + index % 3 for index in range(layer_count)]
    layers = []
    for index, width in enumerate(widths):
        layers.append(
            WidthPrunedQ4LayerWeights(
                rng.normal(scale=0.3, size=(width, hidden)).astype(np.float32),
                rng.normal(scale=0.3, size=(width, hidden)).astype(np.float32),
                rng.normal(scale=0.3, size=(hidden, width)).astype(np.float32),
                np.arange(index, index + width, dtype=np.int64) % source_intermediate,
            )
        )
    return layers, hidden, source_intermediate, widths


def _save_fixture(path: Path, layer_count: int = 3):
    layers, hidden, source_intermediate, widths = _fixture(layer_count)
    save_width_pruned_q4_artifact(
        path, layers, source_intermediate_size=source_intermediate
    )
    return (
        load_width_pruned_q4_artifact(path),
        hidden,
        source_intermediate,
        widths,
    )


def test_width_pruned_q4_round_trip_uses_exact_file_traffic(tmp_path):
    path = tmp_path / "model.width-q4.bin"
    loaded, hidden, source_intermediate, widths = _save_fixture(path, layer_count=30)
    traffic = width_pruned_q4_traffic(hidden, source_intermediate, widths)
    decoded = decode_width_pruned_q4_artifact(loaded)

    assert path.stat().st_size == traffic["serialized_artifact_bytes"]
    assert traffic["traffic_numerator_bytes"] == path.stat().st_size
    assert traffic["total_cold_bytes"] == path.stat().st_size
    assert (
        traffic["dense_q4_source_mlp_bytes"]
        == (len(widths) * 3 * hidden * source_intermediate + 1) // 2
    )
    assert loaded.serialized_artifact_bytes == path.stat().st_size
    assert len(loaded.layers) == 30
    assert len(decoded) == 30
    assert [len(layer.source_ids) for layer in loaded.layers] == widths
    assert all(offset % 64 == 0 for offset in loaded.layer_offsets)
    assert all(block % 64 == 0 for block in loaded.layer_block_bytes)
    assert sum(loaded.layer_block_bytes) == traffic["total_layer_block_bytes"]
    for layer, width in zip(decoded, widths, strict=True):
        assert layer["gate"].shape == (width, hidden)
        assert layer["up"].shape == (width, hidden)
        assert layer["down"].shape == (hidden, width)


def test_reloaded_width_pruned_forward_matches_reference_decode(tmp_path):
    loaded, _, _, _ = _save_fixture(tmp_path / "model.width-q4.bin")
    decoded = decode_width_pruned_q4_artifact(loaded)
    states = np.asarray(
        [[0.25, -0.5, 0.75, 0.1, -0.2, 0.4], [-0.3, 0.2, 0.4, -0.8, 0.6, 0.1]],
        dtype=np.float32,
    )

    actual = width_pruned_q4_forward(loaded, 1, states)
    gate = states @ decoded[1]["gate"].T
    expected = (gate / (1.0 + np.exp(-gate))) * (states @ decoded[1]["up"].T)
    expected = expected @ decoded[1]["down"].T

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_width_pruned_loader_rejects_nonzero_layer_padding(tmp_path):
    path = tmp_path / "model.width-q4.bin"
    loaded, _, _, _ = _save_fixture(path)
    payload = bytearray(path.read_bytes())
    payload[loaded.layer_offsets[0] + loaded.layer_block_bytes[0] - 1] = 1
    corrupt = tmp_path / "corrupt.width-q4.bin"
    corrupt.write_bytes(payload)

    with pytest.raises(ValueError, match="layer padding is non-zero"):
        load_width_pruned_q4_artifact(corrupt)


def test_width_pruned_save_rejects_duplicate_source_ids(tmp_path):
    layers, _, source_intermediate, _ = _fixture()
    first = layers[0]
    layers[0] = WidthPrunedQ4LayerWeights(
        first.gate,
        first.up,
        first.down,
        np.zeros(3, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="unique source-channel indices"):
        save_width_pruned_q4_artifact(
            tmp_path / "invalid.bin",
            layers,
            source_intermediate_size=source_intermediate,
        )


def test_width_pruned_loader_rejects_noncanonical_directory_offset(tmp_path):
    path = tmp_path / "model.width-q4.bin"
    loaded, _, _, _ = _save_fixture(path)
    payload = bytearray(path.read_bytes())
    # First directory entry starts at the end of the 64-byte global header;
    # its uint64 offset begins eight bytes into the entry.
    payload[loaded.header_block_bytes + 8] ^= 64
    corrupt = tmp_path / "corrupt-offset.width-q4.bin"
    corrupt.write_bytes(payload)

    with pytest.raises(ValueError, match="directory entry is invalid"):
        load_width_pruned_q4_artifact(corrupt)
