import json

import numpy as np
import pytest

from engram.semantic.dip import stable_top_k
from engram.semantic.dip_package import SerializedDIPLayer, write_serialized_dip_layer
from engram.semantic.swiglu import silu


def _weights():
    rng = np.random.default_rng(123)
    gate = rng.normal(size=(7, 32)).astype(np.float32)
    up = rng.normal(size=(7, 32)).astype(np.float32)
    down = rng.normal(size=(32, 7)).astype(np.float32)
    return gate, up, down


def test_serialized_dip_all_records_matches_dense(tmp_path):
    gate, up, down = _weights()
    write_serialized_dip_layer(tmp_path, gate, up, down)
    layer = SerializedDIPLayer(tmp_path)
    hidden = np.linspace(-0.8, 0.7, 32, dtype=np.float32)

    result = layer.read(hidden, input_fraction=1.0, candidate_count=7, top_k=7)
    activations = silu(gate.astype(np.float64) @ hidden) * (
        up.astype(np.float64) @ hidden
    )
    expected = down.astype(np.float64) @ activations

    np.testing.assert_allclose(result.output, expected, rtol=2e-6, atol=2e-6)
    assert result.metrics.logical_weight_bytes == result.metrics.dense_weight_bytes
    assert result.metrics.candidate_completion_bytes == 0
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["version"] == 2
    assert metadata["gate_up_layout"] == "coordinate_record"
    assert not (tmp_path / "gate_rows.npy").exists()
    assert not (tmp_path / "up_rows.npy").exists()
    assert metadata["cache_line_bytes"] == 64


def test_candidate_completion_is_exact_and_reports_cache_line_amplification(tmp_path):
    gate, up, down = _weights()
    write_serialized_dip_layer(tmp_path, gate, up, down)
    layer = SerializedDIPLayer(tmp_path)
    hidden = np.linspace(-1.0, 1.0, 32, dtype=np.float32)

    result = layer.read(hidden, input_fraction=0.5, candidate_count=5, top_k=3)
    exact_activations = silu(gate.astype(np.float64) @ hidden) * (
        up.astype(np.float64) @ hidden
    )
    exact_scores = np.abs(exact_activations) * np.linalg.norm(down, axis=0)
    candidate_by_index = np.argsort(result.candidate_indices, kind="stable")
    expected_local = candidate_by_index[
        stable_top_k(
            exact_scores[result.candidate_indices[candidate_by_index]], 3
        )
    ]

    np.testing.assert_array_equal(
        result.selected_indices, result.candidate_indices[expected_local]
    )
    np.testing.assert_allclose(
        result.selected_activations,
        exact_activations[result.selected_indices],
        rtol=2e-6,
        atol=2e-6,
    )
    assert result.metrics.logical_weight_bytes < result.metrics.dense_weight_bytes
    assert result.metrics.cache_line_weight_bytes >= result.metrics.logical_weight_bytes


def test_serialized_dip_detects_corruption(tmp_path):
    gate, up, down = _weights()
    write_serialized_dip_layer(tmp_path, gate, up, down)
    path = tmp_path / "gate_coordinates.npy"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0x01
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="checksum mismatch"):
        SerializedDIPLayer(tmp_path)


def test_dual_layout_is_explicit_and_versions_must_agree(tmp_path):
    gate, up, down = _weights()
    write_serialized_dip_layer(tmp_path, gate, up, down, dual_layout=True)
    layer = SerializedDIPLayer(tmp_path)
    assert layer.gate_rows is not None
    assert layer.up_rows is not None
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["version"] == 3
    assert metadata["dual_layout_diagnostic"] is True

    config = np.load(tmp_path / "config.npy")
    config[0] = 2
    np.save(tmp_path / "config.npy", config, allow_pickle=False)
    with pytest.raises(ValueError, match="versions differ"):
        SerializedDIPLayer(tmp_path, verify=False)


def test_serialized_dip_rejects_wrong_down_shape(tmp_path):
    gate, up, down = _weights()
    with pytest.raises(ValueError, match="down must have shape"):
        write_serialized_dip_layer(tmp_path, gate, up, down[:-1])
