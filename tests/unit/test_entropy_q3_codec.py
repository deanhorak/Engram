from pathlib import Path

import numpy as np
import pytest

from engram.training.entropy_q3_codec import (
    EntropyQ3LayerWeights,
    decode_entropy_q3_artifact,
    entropy_q3_dynamic_traffic,
    entropy_q3_forward,
    load_entropy_q3_artifact,
    save_entropy_q3_artifact,
)


def _fixture(layer_count: int = 3):
    rng = np.random.default_rng(5051)
    hidden = 7
    source_intermediate = 13
    layers = []
    for index in range(layer_count):
        width = 4 + index % 3
        layers.append(
            EntropyQ3LayerWeights(
                rng.normal(scale=0.4, size=(width, hidden)).astype(np.float32),
                rng.normal(scale=0.4, size=(width, hidden)).astype(np.float32),
                rng.normal(scale=0.4, size=(hidden, width)).astype(np.float32),
                np.arange(index, index + width, dtype=np.int64),
            )
        )
    return layers, hidden, source_intermediate


def _save_fixture(path: Path, layer_count: int = 3):
    layers, hidden, source_intermediate = _fixture(layer_count)
    save_entropy_q3_artifact(path, layers, source_intermediate_size=source_intermediate)
    return load_entropy_q3_artifact(path), hidden, source_intermediate


def test_sequential_entropy_q3_round_trip_and_exact_all_record_traffic(tmp_path):
    path = tmp_path / "sequential.entropy-q3.bin"
    artifact, hidden, source_intermediate = _save_fixture(path, layer_count=4)
    decoded = decode_entropy_q3_artifact(artifact)
    all_ids = [list(range(layer.width)) for layer in artifact.layers]
    traffic = entropy_q3_dynamic_traffic(artifact, all_ids)

    assert path.stat().st_size == artifact.serialized_artifact_bytes
    assert traffic["total_cold_bytes"] == path.stat().st_size
    assert traffic["serialized_artifact_bytes"] == path.stat().st_size
    assert (
        traffic["dense_q4_source_mlp_bytes"]
        == (4 * 3 * hidden * source_intermediate + 1) // 2
    )
    assert len(decoded) == 4
    assert all(
        record.offset % 64 == 0
        for layer in artifact.layers
        for record in layer.down_records
    )


def test_sequential_entropy_q3_forward_matches_validated_reference(tmp_path):
    artifact, _, _ = _save_fixture(tmp_path / "sequential.entropy-q3.bin")
    decoded = decode_entropy_q3_artifact(artifact)
    states = np.asarray(
        [[0.2, -0.4, 0.8, 0.1, -0.3, 0.7, 0.5], [-0.5, 0.3, 0.1, 0.6, -0.2, 0.4, -0.7]],
        dtype=np.float32,
    )
    selected = np.asarray([[0, 2, 3], [1, 2, 3]], dtype=np.int64)

    actual, actual_ids = entropy_q3_forward(artifact, 0, states, selected_ids=selected)
    gate = states @ decoded[0]["gate"].T
    activations = (gate / (1.0 + np.exp(-gate))) * (states @ decoded[0]["up"].T)
    expected = np.stack(
        [
            activations[row, ids] @ decoded[0]["down"][:, ids].T
            for row, ids in enumerate(selected)
        ]
    )

    np.testing.assert_array_equal(actual_ids, selected)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_sequential_entropy_q3_strict_integrity_covers_header_and_record_metadata(
    tmp_path,
):
    path = tmp_path / "sequential.entropy-q3.bin"
    artifact, _, _ = _save_fixture(path)
    payload = bytearray(path.read_bytes())
    # Global source_intermediate_size is the fourth uint32 after the magic.
    payload[20:24] = (800).to_bytes(4, "little")
    corrupt_header = tmp_path / "corrupt-header.bin"
    corrupt_header.write_bytes(payload)
    with pytest.raises(ValueError, match="integrity|checksum"):
        load_entropy_q3_artifact(corrupt_header)

    payload = bytearray(path.read_bytes())
    # source_id starts 12 bytes into a down-record header.
    offset = artifact.layers[0].down_records[0].offset + 12
    payload[offset : offset + 4] = (7).to_bytes(4, "little")
    corrupt_record = tmp_path / "corrupt-record.bin"
    corrupt_record.write_bytes(payload)
    with pytest.raises(ValueError, match="record.*checksum|integrity"):
        load_entropy_q3_artifact(corrupt_record)


def test_entropy_q3_requires_the_declared_64_byte_physical_alignment(tmp_path):
    layers, _, source_intermediate = _fixture(1)
    with pytest.raises(ValueError, match="exactly 64"):
        save_entropy_q3_artifact(
            tmp_path / "bad.bin",
            layers,
            source_intermediate_size=source_intermediate,
            cache_line_bytes=128,
        )


def test_sequential_entropy_q3_rejects_reserved_stream_padding_and_empty_selection(
    tmp_path,
):
    path = tmp_path / "sequential.entropy-q3.bin"
    artifact, _, _ = _save_fixture(path)
    payload = bytearray(path.read_bytes())
    # layer header (64) + model (64) + 20-byte stream struct reaches the
    # reserved portion of the fixed 32-byte gate-stream header.
    reserved = artifact.layers[0].layer_offset + 64 + 64 + 20
    payload[reserved] = 1
    corrupt = tmp_path / "corrupt-reserved.bin"
    corrupt.write_bytes(payload)
    with pytest.raises(ValueError, match="reserved padding"):
        load_entropy_q3_artifact(corrupt)

    with pytest.raises(ValueError, match="selected_ids"):
        entropy_q3_forward(
            artifact,
            0,
            np.zeros((1, artifact.hidden_size), dtype=np.float32),
            selected_ids=np.empty((1, 0), dtype=np.int64),
        )
