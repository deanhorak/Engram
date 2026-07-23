import numpy as np
import pytest

from engram.semantic.multiset_additive_quantization import (
    POSITION_3X4_FP16_SCALE,
    decode_multiset_additive,
    load_multiset_additive,
    make_multiset_additive_metadata,
)
from engram.training.position_aq_boundaries import train_position_aq_boundaries


def _fixture(seed=44):
    rng = np.random.default_rng(seed)
    hidden_size = 16
    intermediate_size = 40
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


def _run(values, destination):
    return train_position_aq_boundaries(
        *values,
        artifact_dir=destination,
        position_buckets=2,
        p_steps_per_cycle=6,
        v_cycles=1,
        batch_size=20,
        learning_rate=6e-3,
        checkpoint_interval=2,
        fit_iterations=1,
        fit_sample_limit=64,
        v_max_records=20,
        v_trust_fraction=0.01,
        selection_records=20,
        seed=5,
        device="cpu",
        # The fixed header/codebooks dominate this deliberately tiny fixture.
        enforce_traffic_gate=False,
    )


def test_canonical_position_codec_has_predeclared_sub_q4_layout():
    metadata = make_multiset_additive_metadata(
        (4608, 576), profile=POSITION_3X4_FP16_SCALE
    )

    assert metadata.num_codebook_sets == 108
    assert metadata.storage_bytes == 590_296
    assert metadata.dense_q4_bytes == 1_327_104
    assert metadata.fraction_of_dense_q4 == pytest.approx(0.444800106095679)
    assert metadata.fraction_of_dense_q4 < 0.45
    assert metadata.storage_components() == {
        "header_and_checksum": 256,
        "packed_codes": 497_664,
        "codebooks": 82_944,
        "row_scales": 9_216,
        "scale_codebooks": 0,
        "set_mapping": 216,
        "total": 590_296,
    }


def test_position_pv_improves_and_limits_discrete_update(tmp_path):
    pytest.importorskip("torch")
    result = _run(_fixture(), tmp_path / "run")
    report = result.report
    cycle = report["cycles"][0]
    initial = report["initial"]["strict_reloaded_validation"]
    final = report["final"]["strict_reloaded_validation"]

    assert cycle["p"]["best_selection_training_objective"] < cycle["p"][
        "initial_selection_training_objective"
    ]
    assert cycle["v"]["total_proposed_code_changes"] > 0
    assert cycle["v"]["total_applied_code_changes"] <= cycle["v"][
        "trust_limit_assignments"
    ]
    assert cycle["v"]["total_applied_code_changes"] <= int(
        cycle["v"]["total_code_assignments"] * 0.01
    )
    assert cycle["v"]["candidate_count_per_assignment"] == 16
    assert cycle["v"]["nonlinear_candidate_forwards"] == 0
    assert final["mean_relative_l2"] < initial["mean_relative_l2"]
    assert report["validity"]["validation_was_not_used_for_checkpoint_selection"]
    assert report["validity"]["set_selection_is_static"]

    assert result.artifact_path is not None
    assert result.artifact_path.is_file()
    loaded = load_multiset_additive(result.artifact_path)
    np.testing.assert_array_equal(
        decode_multiset_additive(loaded), decode_multiset_additive(result.encoding)
    )
    assert result.artifact_path.stat().st_size == result.encoding.storage_bytes
    assert report["final"]["artifact"]["exact_round_trip"] is True


def test_position_pv_is_deterministic_for_fixed_seed(tmp_path):
    pytest.importorskip("torch")
    values = _fixture()
    first = _run(values, tmp_path / "first")
    second = _run(values, tmp_path / "second")

    np.testing.assert_array_equal(first.encoding.packed_codes, second.encoding.packed_codes)
    np.testing.assert_array_equal(first.encoding.codebooks, second.encoding.codebooks)
    np.testing.assert_array_equal(first.encoding.row_scales, second.encoding.row_scales)
    np.testing.assert_array_equal(first.encoding.set_mapping, second.encoding.set_mapping)
    assert first.report["final"]["strict_reloaded_validation"] == second.report[
        "final"
    ]["strict_reloaded_validation"]


def test_position_pv_rejects_invalid_trust_fraction():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="within"):
        train_position_aq_boundaries(
            *_fixture(),
            position_buckets=2,
            p_steps_per_cycle=1,
            v_cycles=1,
            v_trust_fraction=0,
            enforce_traffic_gate=False,
        )
