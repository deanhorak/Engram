import numpy as np
import pytest

from engram.training.switch_expert_boundaries import (
    _balanced_kmeans_regions,
    _decode_symmetric_q4_rows,
    _nearest_centroid_routes,
    _pack_symmetric_q4_rows,
    load_switch_expert_artifact,
    run_switch_expert_boundary_screen,
    save_switch_expert_artifact,
    switch_expert_traffic,
)


def test_symmetric_q4_pack_round_trip_is_deterministic_and_bounded():
    rng = np.random.default_rng(17)
    values = rng.normal(size=(5, 7)).astype(np.float32)
    values[2] = 0.0

    packed = _pack_symmetric_q4_rows(values)
    decoded = _decode_symmetric_q4_rows(packed)
    repacked = _pack_symmetric_q4_rows(decoded)

    assert packed.packed.dtype == np.uint8
    assert packed.scales.dtype == np.float16
    assert packed.packed.size == 18
    np.testing.assert_array_equal(repacked.packed, packed.packed)
    np.testing.assert_array_equal(repacked.scales, packed.scales)
    maximum_error = np.max(np.abs(decoded - values), axis=1)
    assert np.all(maximum_error <= packed.scales.astype(np.float32) * 0.51 + 1e-6)


def test_balanced_kmeans_and_nearest_centroid_routing_are_deterministic():
    rng = np.random.default_rng(23)
    left = rng.normal(loc=(-3.0, 0.0), scale=0.08, size=(20, 2))
    right = rng.normal(loc=(3.0, 0.0), scale=0.08, size=(20, 2))
    values = np.concatenate((left, right)).astype(np.float32)

    assignment_a, centroids_a = _balanced_kmeans_regions(values, 2, seed=9)
    assignment_b, centroids_b = _balanced_kmeans_regions(values, 2, seed=9)

    np.testing.assert_array_equal(assignment_a, assignment_b)
    np.testing.assert_array_equal(centroids_a, centroids_b)
    np.testing.assert_array_equal(
        _nearest_centroid_routes(values, centroids_a), assignment_a
    )
    np.testing.assert_array_equal(np.bincount(assignment_a), np.asarray([20, 20]))


def test_default_and_low_rank_switch_layouts_are_strictly_under_gate():
    default = switch_expert_traffic(576, 1536)
    low_rank = switch_expert_traffic(
        576, 1536, width=640, centroid_bits=8, residual_rank=15
    )

    assert default["selected_expert_q4_code_bytes"] == 580_608
    assert default["selected_expert_fp16_scale_bytes"] == 3_840
    assert default["router_centroid_bytes"] == 9_216
    assert default["total_cold_bytes"] == 593_728
    assert default["fraction_of_dense_q4"] == pytest.approx(
        593_728 / 1_327_104
    )
    assert default["passes_45_percent_traffic_gate"] is True

    assert low_rank["selected_expert_q4_code_bytes"] == 552_960
    assert low_rank["selected_expert_fp16_scale_bytes"] == 3_712
    assert low_rank["selected_expert_fp16_residual_bytes"] == 34_560
    assert low_rank["router_centroid_bytes"] == 4_608
    assert low_rank["router_centroid_fp16_scale_bytes"] == 16
    assert low_rank["total_cold_bytes"] == 595_968
    assert low_rank["fraction_of_dense_q4"] == pytest.approx(
        595_968 / 1_327_104
    )
    assert low_rank["passes_45_percent_traffic_gate"] is True


def test_artifact_reload_uses_contiguous_cache_aligned_expert_blocks(tmp_path):
    rng = np.random.default_rng(31)
    centroids = rng.normal(size=(3, 6)).astype(np.float32)
    packed_experts = []
    for _ in range(3):
        gate = _pack_symmetric_q4_rows(rng.normal(size=(4, 6)).astype(np.float32))
        up = _pack_symmetric_q4_rows(rng.normal(size=(4, 6)).astype(np.float32))
        down = _pack_symmetric_q4_rows(rng.normal(size=(6, 4)).astype(np.float32))
        from engram.training.switch_expert_boundaries import PackedSwitchExpert

        packed_experts.append(PackedSwitchExpert(gate, up, down))
    path = tmp_path / "switch.q4.bin"

    save_switch_expert_artifact(path, centroids, packed_experts, centroid_bits=8)
    loaded = load_switch_expert_artifact(path)

    assert loaded.centroid_bits == 8
    assert loaded.hidden_size == 6
    assert loaded.width == 4
    assert len(loaded.experts) == 3
    assert all(offset % 64 == 0 for offset in loaded.expert_offsets)
    for expected, actual in zip(packed_experts, loaded.experts):
        np.testing.assert_array_equal(actual.gate.packed, expected.gate.packed)
        np.testing.assert_array_equal(actual.up.packed, expected.up.packed)
        np.testing.assert_array_equal(actual.down.packed, expected.down.packed)
        np.testing.assert_array_equal(actual.gate.scales, expected.gate.scales)


def test_fixed_step_training_improves_strict_reloaded_boundary(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(47)
    hidden = 4
    intermediate = 10
    gate = (0.30 * rng.normal(size=(intermediate, hidden))).astype(np.float32)
    up = (0.26 * rng.normal(size=(intermediate, hidden))).astype(np.float32)
    down = (0.24 * rng.normal(size=(hidden, intermediate))).astype(np.float32)

    def dense(inputs):
        projected_gate = inputs @ gate.T
        activation = projected_gate / (1.0 + np.exp(-projected_gate))
        return ((activation * (inputs @ up.T)) @ down.T).astype(np.float32)

    train_x = np.concatenate(
        (
            rng.normal(loc=(-2.0, 0.0, 0.0, 0.0), scale=0.55, size=(64, hidden)),
            rng.normal(loc=(2.0, 0.0, 0.0, 0.0), scale=0.55, size=(64, hidden)),
        )
    ).astype(np.float32)
    validation_x = np.concatenate(
        (
            rng.normal(loc=(-2.0, 0.0, 0.0, 0.0), scale=0.55, size=(32, hidden)),
            rng.normal(loc=(2.0, 0.0, 0.0, 0.0), scale=0.55, size=(32, hidden)),
        )
    ).astype(np.float32)
    artifact = tmp_path / "trained.q4.bin"
    report_file = tmp_path / "trained.json"

    report = run_switch_expert_boundary_screen(
        gate,
        up,
        down,
        train_x,
        dense(train_x),
        validation_x,
        dense(validation_x),
        artifact_path=artifact,
        report_path=report_file,
        experts=2,
        width=5,
        steps=48,
        batch_size=32,
        learning_rate=3e-3,
        seed=5,
        maximum_mean_relative_l2=2.0,
        enforce_traffic_gate=False,
    )

    initial = report["validation"]["initial_packed_initialized_experts"]
    strict = report["validation"]["strict_reloaded_packed_experts"]
    assert strict["relative_l2"]["mean"] < initial["relative_l2"]["mean"]
    assert report["configuration"]["validation_checkpoint_selection"] is False
    assert report["validation"]["source"].startswith("serialized_reloaded_q4")
    assert report["artifact"]["bytes"] == artifact.stat().st_size
    assert report["artifact"]["all_expert_offsets_cache_aligned"] is True
    assert report_file.is_file()
    assert sum(report["router"]["training_cluster_counts"]) == len(train_x)
    assert report["dense_reference_validation_parity"]["relative_l2"]["mean"] < 1e-5

