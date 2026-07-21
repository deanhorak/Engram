import numpy as np
import pytest

from engram.evaluation.dip_sweep import evaluate_dip_exact_completion_sweep
from engram.models.fixture import create_tiny_fixture
from engram.models.inspection import inspect_model
from engram.tracing.format import TraceWriter


def _write_trace(path, model_hash, *, split="validation"):
    rng = np.random.default_rng(311)
    with TraceWriter(
        path,
        model_hash=model_hash,
        dataset_hash="held-out-data",
        split=split,
        seed=311,
        metadata={"fixture_only": True},
    ) as writer:
        for sample in range(3):
            arrays = {
                "sample_id": np.full(2, sample, dtype=np.int64),
                "token_id": np.asarray(
                    [100 + sample * 2, 101 + sample * 2], dtype=np.int64
                ),
            }
            for layer in range(2):
                arrays[f"layer_{layer}_mlp_input"] = rng.normal(size=(2, 8)).astype(
                    np.float32
                )
            writer.append(arrays)


def test_dip_sweep_measures_full_candidates_and_exact_traffic(tmp_path):
    model = create_tiny_fixture(
        tmp_path / "model",
        hidden_size=8,
        intermediate_size=12,
        num_layers=2,
        num_heads=2,
    )
    trace = tmp_path / "validation"
    _write_trace(trace, inspect_model(model).source_hash)

    report = evaluate_dip_exact_completion_sweep(
        model,
        trace,
        input_fractions=(0.5, 1.0),
        top_k=4,
        candidate_counts=(6, 12),
        validation_records=3,
    )

    assert report["validation"]["records_per_layer"] == 3
    assert report["validation"]["unique_sequence_count"] == 2
    assert len(report["arms"]) == 4
    full = next(
        arm
        for arm in report["arms"]
        if arm["input_fraction"] == 1.0 and arm["candidate_count"] == 12
    )
    assert full["candidate_recall"]["mean"] == 1.0
    assert full["oracle_score_mass_recall"]["mean"] == pytest.approx(1.0)
    assert full["mlp_output_relative_l2"]["mean"] == pytest.approx(
        report["oracle"]["mlp_output_relative_l2"]["mean"]
    )

    half = next(
        arm
        for arm in report["arms"]
        if arm["input_fraction"] == 0.5 and arm["candidate_count"] == 6
    )
    traffic = half["projected_traffic"]
    assert traffic["projected_weight_elements_per_token_layer"] == 176
    assert traffic["dense_weight_elements_per_token_layer"] == 288
    assert traffic["projected_fraction_of_dense"] == pytest.approx(176 / 288)
    assert report["screening_decision"] in {
        "eligible_for_causal_intervention",
        "reject_before_causal_intervention",
    }


def test_dip_sweep_requires_validation_trace_and_valid_budgets(tmp_path):
    model = create_tiny_fixture(
        tmp_path / "model",
        hidden_size=8,
        intermediate_size=12,
        num_layers=2,
        num_heads=2,
    )
    trace = tmp_path / "calibration"
    _write_trace(trace, inspect_model(model).source_hash, split="calibration")

    with pytest.raises(ValueError, match="validation"):
        evaluate_dip_exact_completion_sweep(
            model,
            trace,
            input_fractions=(0.5,),
            top_k=4,
            candidate_counts=(6,),
        )

    validation = tmp_path / "validation"
    _write_trace(validation, inspect_model(model).source_hash)
    with pytest.raises(ValueError, match="candidate counts"):
        evaluate_dip_exact_completion_sweep(
            model,
            validation,
            input_fractions=(0.5,),
            top_k=4,
            candidate_counts=(3,),
        )
