import numpy as np
import pytest

from engram.training.budget_native_ternary_codec import (
    BudgetNativeTernaryLayerWeights,
    budget_native_ternary_forward,
    budget_native_ternary_traffic,
    decode_budget_native_ternary_artifact,
    grouped_ternary_decode,
    grouped_ternary_quantize,
    load_budget_native_ternary_artifact,
    pack_ternary_codes,
    save_budget_native_ternary_artifact,
    unpack_ternary_codes,
)


def _fixture(layer_count=3):
    rng = np.random.default_rng(20260723)
    hidden = 6
    intermediate = 10
    layers = [
        BudgetNativeTernaryLayerWeights(
            rng.normal(scale=0.25, size=(intermediate, hidden)),
            rng.normal(scale=0.25, size=(intermediate, hidden)),
            rng.normal(scale=0.25, size=(hidden, intermediate)),
        )
        for _ in range(layer_count)
    ]
    return layers, hidden, intermediate


def test_base3_pack_round_trip_and_canonical_tail():
    codes = np.asarray(
        [-1, 0, 1, 1, -1, 0, -1, 1, 0, 1, -1, 0],
        dtype=np.int8,
    )
    packed = pack_ternary_codes(codes)

    np.testing.assert_array_equal(
        unpack_ternary_codes(packed, len(codes)),
        codes,
    )
    corrupt = packed.copy()
    corrupt[-1] += 27
    with pytest.raises(ValueError, match="tail is not canonical"):
        unpack_ternary_codes(corrupt, len(codes))


def test_grouped_quantizer_decodes_fp16_scales_exactly():
    values = np.asarray(
        [[-1.2, -0.2, 0.0, 0.3, 1.6], [0.2, -0.8, 0.9, 0.0, -0.1]],
        dtype=np.float32,
    )
    codes, scales = grouped_ternary_quantize(values, group_size=4)
    decoded = grouped_ternary_decode(
        codes,
        scales,
        shape=values.shape,
        group_size=4,
    )
    expected_scales = np.repeat(scales.astype(np.float32), 4)[: values.size]

    np.testing.assert_array_equal(
        decoded.reshape(-1),
        codes.astype(np.float32) * expected_scales,
    )


def test_grouped_ternary_round_trip_matches_exact_file_traffic(tmp_path):
    layers, hidden, intermediate = _fixture(layer_count=30)
    path = tmp_path / "model.ternary.bin"
    save_budget_native_ternary_artifact(path, layers, group_size=16)
    loaded = load_budget_native_ternary_artifact(path)
    decoded = decode_budget_native_ternary_artifact(loaded)
    traffic = budget_native_ternary_traffic(
        hidden,
        intermediate,
        layer_count=30,
        group_size=16,
    )

    assert path.stat().st_size == traffic["serialized_artifact_bytes"]
    assert loaded.serialized_artifact_bytes == path.stat().st_size
    assert len(decoded) == 30
    assert decoded[0]["gate"].shape == (intermediate, hidden)
    assert decoded[0]["up"].shape == (intermediate, hidden)
    assert decoded[0]["down"].shape == (hidden, intermediate)
    assert all(offset % 64 == 0 for offset in loaded.layer_offsets)


def test_reloaded_ternary_forward_matches_decoded_reference(tmp_path):
    layers, _, _ = _fixture()
    path = tmp_path / "model.ternary.bin"
    save_budget_native_ternary_artifact(path, layers, group_size=16)
    loaded = load_budget_native_ternary_artifact(path)
    decoded = decode_budget_native_ternary_artifact(loaded)
    states = np.asarray(
        [[0.25, -0.5, 0.75, 0.1, -0.2, 0.4]],
        dtype=np.float32,
    )

    actual = budget_native_ternary_forward(loaded, 1, states)
    gate = states @ decoded[1]["gate"].T
    expected = (gate / (1.0 + np.exp(-gate))) * (
        states @ decoded[1]["up"].T
    )
    expected = expected @ decoded[1]["down"].T

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_loader_rejects_nonzero_layer_padding(tmp_path):
    layers, _, _ = _fixture()
    path = tmp_path / "model.ternary.bin"
    save_budget_native_ternary_artifact(path, layers, group_size=16)
    loaded = load_budget_native_ternary_artifact(path)
    payload = bytearray(path.read_bytes())
    payload[loaded.layer_offsets[0] + loaded.layer_block_bytes[0] - 1] = 1
    corrupt = tmp_path / "corrupt.ternary.bin"
    corrupt.write_bytes(payload)

    with pytest.raises(ValueError, match="layer padding is non-zero"):
        load_budget_native_ternary_artifact(corrupt)


def test_smollm2_group128_layout_passes_complete_traffic_gate():
    traffic = budget_native_ternary_traffic(
        576,
        1536,
        layer_count=30,
        group_size=128,
    )

    assert traffic["passes_45_percent_traffic_gate"]
    assert traffic["fraction_of_dense_q4"] < 0.45
    assert traffic["headroom_bytes_to_45_percent"] > 0
