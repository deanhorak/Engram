import numpy as np
import pytest

from engram.semantic.multiset_additive_quantization import (
    decode_multiset_additive,
    load_multiset_additive,
    make_multiset_additive_metadata,
)
from engram.training.projection_aq_boundaries import (
    train_projection_aq_boundaries,
)


def _fixture(seed=44):
    rng = np.random.default_rng(seed)
    hidden_size = 16
    intermediate_size = 136
    gate = (rng.normal(size=(intermediate_size, hidden_size)) * 0.12).astype(
        np.float32
    )
    up = (rng.normal(size=(intermediate_size, hidden_size)) * 0.12).astype(
        np.float32
    )
    down = (rng.normal(size=(hidden_size, intermediate_size)) * 0.10).astype(
        np.float32
    )

    def boundary(inputs):
        gate_projection = inputs @ gate.T
        silu = gate_projection / (1.0 + np.exp(-gate_projection))
        return ((silu * (inputs @ up.T)) @ down.T).astype(np.float32)

    training_inputs = rng.normal(size=(40, hidden_size)).astype(np.float32)
    validation_inputs = rng.normal(size=(20, hidden_size)).astype(np.float32)
    return (
        gate,
        up,
        down,
        training_inputs,
        boundary(training_inputs),
        validation_inputs,
        boundary(validation_inputs),
    )


def _run(values, artifact_dir):
    return train_projection_aq_boundaries(
        *values,
        artifact_dir=artifact_dir,
        p_steps_per_cycle=6,
        v_cycles=1,
        batch_size=20,
        learning_rate=6e-3,
        checkpoint_interval=2,
        fit_iterations=1,
        fit_sample_limit=96,
        v_max_records=20,
        selection_records=20,
        seed=5,
        device="cpu",
        # Fixed codec overhead dominates this deliberately tiny unit fixture.
        # The canonical shape is tested against the real gate separately.
        enforce_traffic_gate=False,
    )


def test_canonical_projection_codec_is_strictly_below_q4_traffic_gate():
    metadata = make_multiset_additive_metadata((4608, 576))

    assert metadata.storage_bytes == 596_992
    assert metadata.dense_q4_bytes == 1_327_104
    assert metadata.fraction_of_dense_q4 == pytest.approx(0.4498456790123457)
    assert metadata.fraction_of_dense_q4 < 0.45
    assert metadata.storage_components() == {
        "header_and_checksum": 256,
        "packed_codes": 580_608,
        "codebooks": 12_288,
        "packed_scale_indices": 3_456,
        "scale_codebooks": 384,
        "set_mapping": 0,
        "total": 596_992,
    }


def test_projection_pv_training_improves_and_reports_only_packed_reload(tmp_path):
    pytest.importorskip("torch")
    values = _fixture()
    result = _run(values, tmp_path / "first")
    report = result.report
    initial = report["initial"]["strict_reloaded_validation"]
    final = report["final"]["strict_reloaded_validation"]

    assert report["cycles"][0]["p"]["best_selection_training_objective"] < report[
        "cycles"
    ][0]["p"]["initial_selection_training_objective"]
    assert report["cycles"][0]["v"]["total_code_changes"] > 0
    assert report["cycles"][0]["v"]["nonlinear_candidate_forwards"] == 0
    assert final["mean_relative_l2"] < initial["mean_relative_l2"]
    assert report["final"][
        "validation_mean_relative_l2_fractional_improvement"
    ] > 0
    assert report["validity"] == {
        "validation_was_not_used_for_checkpoint_selection": True,
        "all_reported_metrics_use_serialized_reloaded_decode": True,
        "oracle_is_not_deployable": True,
    }

    assert result.artifact_path is not None
    assert result.artifact_path.is_file()
    loaded = load_multiset_additive(result.artifact_path)
    np.testing.assert_array_equal(
        decode_multiset_additive(loaded), decode_multiset_additive(result.encoding)
    )
    assert result.artifact_path.stat().st_size == result.encoding.storage_bytes
    assert report["final"]["artifact"]["exact_round_trip"] is True
    assert report["final"]["artifact"]["metrics_source"] == (
        "checksum_validated_serialized_reload_decode"
    )


def test_projection_pv_training_is_deterministic_for_a_fixed_seed(tmp_path):
    pytest.importorskip("torch")
    values = _fixture()
    first = _run(values, tmp_path / "first")
    second = _run(values, tmp_path / "second")

    np.testing.assert_array_equal(first.encoding.packed_codes, second.encoding.packed_codes)
    np.testing.assert_array_equal(first.encoding.codebooks, second.encoding.codebooks)
    np.testing.assert_array_equal(
        first.encoding.packed_scale_indices, second.encoding.packed_scale_indices
    )
    np.testing.assert_array_equal(
        first.encoding.scale_codebooks, second.encoding.scale_codebooks
    )
    assert first.report["final"]["strict_reloaded_validation"] == second.report[
        "final"
    ]["strict_reloaded_validation"]


def test_projection_screen_rejects_impossible_cosine_threshold():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="must not exceed 1"):
        train_projection_aq_boundaries(
            *_fixture(),
            p_steps_per_cycle=1,
            v_cycles=1,
            minimum_mean_cosine=1.01,
            enforce_traffic_gate=False,
        )
