from pathlib import Path

import numpy as np
import pytest

from engram.training.grouped_sparse_codec import (
    decode_grouped_sparse_artifact,
    grouped_sparse_forward,
    grouped_sparse_traffic,
    load_grouped_sparse_artifact,
    save_grouped_sparse_artifact,
)


def _fixture():
    rng = np.random.default_rng(1701)
    hidden = 6
    intermediate = 12
    group_size = 3
    rank = 2
    gate = rng.normal(scale=0.3, size=(intermediate, hidden)).astype(np.float32)
    up = rng.normal(scale=0.3, size=(intermediate, hidden)).astype(np.float32)
    down = rng.normal(scale=0.3, size=(hidden, intermediate)).astype(np.float32)
    router_input = rng.normal(scale=0.2, size=(hidden, rank)).astype(np.float32)
    router_output = rng.normal(scale=0.2, size=(rank, intermediate // group_size)).astype(
        np.float32
    )
    bias = rng.normal(scale=0.1, size=(intermediate // group_size,)).astype(np.float32)
    nonlinear = np.asarray([0.25, -0.5], dtype=np.float32)
    permutation = np.asarray([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int64)
    return (
        gate,
        up,
        down,
        router_input,
        router_output,
        bias,
        nonlinear,
        permutation,
        group_size,
    )


def _save_fixture(path: Path):
    (
        gate,
        up,
        down,
        router_input,
        router_output,
        bias,
        nonlinear,
        permutation,
        group_size,
    ) = _fixture()
    save_grouped_sparse_artifact(
        path,
        gate,
        up,
        down,
        router_input,
        router_output,
        bias,
        group_size=group_size,
        selected_records=6,
        router_nonlinear_scale=nonlinear,
        permutation=permutation,
    )
    return load_grouped_sparse_artifact(path)


def test_grouped_q4_artifact_round_trip_matches_exact_accounting(tmp_path):
    path = tmp_path / "layer.grouped-q4.bin"
    loaded = _save_fixture(path)
    traffic = grouped_sparse_traffic(
        6, 12, selected_records=6, router_rank=2, group_size=3
    )
    decoded = decode_grouped_sparse_artifact(loaded)

    assert path.stat().st_size == traffic["serialized_artifact_bytes"]
    assert loaded.hidden_size == 6
    assert loaded.intermediate_size == 12
    assert loaded.group_size == 3
    assert loaded.selected_groups == 2
    assert loaded.router_rank == 2
    assert len(loaded.groups) == 4
    assert all(offset % 64 == 0 for offset in loaded.group_offsets)
    assert decoded["gate"].shape == (12, 6)
    assert decoded["up"].shape == (12, 6)
    assert decoded["down"].shape == (6, 12)
    assert decoded["router_input"].shape == (6, 2)
    assert decoded["router_output"].shape == (2, 4)
    np.testing.assert_array_equal(decoded["permutation"], _fixture()[7])


def test_reloaded_grouped_forward_uses_only_selected_packed_groups(tmp_path):
    loaded = _save_fixture(tmp_path / "layer.grouped-q4.bin")
    decoded = decode_grouped_sparse_artifact(loaded)
    states = np.asarray(
        [[0.25, -0.5, 0.75, 0.1, -0.2, 0.4], [-0.3, 0.2, 0.4, -0.8, 0.6, 0.1]],
        dtype=np.float32,
    )

    actual, selected = grouped_sparse_forward(loaded, states)
    latent = states @ decoded["router_input"]
    sigmoid = 1.0 / (1.0 + np.exp(-latent))
    features = latent + decoded["router_nonlinear_scale"] * latent * sigmoid
    scores = features @ decoded["router_output"] + decoded["router_bias"]
    expected_selected = np.argsort(-scores, axis=1, kind="stable")[:, :2]
    expected = np.zeros_like(actual)
    for row, state in enumerate(states):
        for group_id in expected_selected[row]:
            start = int(group_id) * loaded.group_size
            stop = start + loaded.group_size
            projected_gate = state @ decoded["gate"][start:stop].T
            activation = (
                projected_gate / (1.0 + np.exp(-projected_gate))
            ) * (state @ decoded["up"][start:stop].T)
            expected[row] += activation @ decoded["down"][:, start:stop].T

    np.testing.assert_array_equal(selected, expected_selected)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_loader_rejects_nonzero_group_padding(tmp_path):
    path = tmp_path / "layer.grouped-q4.bin"
    loaded = _save_fixture(path)
    payload = bytearray(path.read_bytes())
    # The tiny fixture leaves padding at the end of every group block.
    payload[loaded.group_offsets[0] + loaded.group_block_bytes - 1] = 1
    corrupt = tmp_path / "corrupt.grouped-q4.bin"
    corrupt.write_bytes(payload)

    with pytest.raises(ValueError, match="padding is non-zero"):
        load_grouped_sparse_artifact(corrupt)


def test_save_rejects_non_permutation_metadata(tmp_path):
    values = list(_fixture())
    values[7] = np.zeros(12, dtype=np.int64)
    with pytest.raises(ValueError, match="every record exactly once"):
        save_grouped_sparse_artifact(
            tmp_path / "invalid.bin",
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            group_size=values[8],
            selected_records=6,
            router_nonlinear_scale=values[6],
            permutation=values[7],
        )
