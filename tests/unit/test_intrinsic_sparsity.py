import numpy as np
import pytest

from engram.evaluation.intrinsic_sparsity import (
    evaluate_intrinsic_sparse_gate_sweep,
    exact_gate_sparse_traffic,
    write_intrinsic_sparse_gate_report,
)
from engram.models.fixture import create_tiny_fixture
from engram.models.inspection import inspect_model, load_layer_mlp
from engram.semantic.swiglu import swiglu
from engram.tracing.format import TraceWriter
from engram.training.intrinsic_sparsity import train_intrinsic_sparse_boundaries
from engram.training.fully_sparse import (
    fully_sparse_mlp_traffic,
    train_fully_sparse_boundaries,
)


def _write_trace(path, model_hash, *, split, seed):
    rng = np.random.default_rng(seed)
    with TraceWriter(
        path,
        model_hash=model_hash,
        dataset_hash=f"{split}-intrinsic-sparse",
        split=split,
        seed=seed,
        metadata={"fixture_only": True},
    ) as writer:
        for sample in range(3):
            arrays = {
                "sample_id": np.full(3, sample, dtype=np.int64),
                "token_id": np.arange(3, dtype=np.int64) + 10 * sample,
            }
            for layer in range(2):
                arrays[f"layer_{layer}_mlp_input"] = rng.normal(size=(3, 8)).astype(
                    np.float32
                )
            writer.append(arrays)


def _write_boundary_trace(path, model, *, split, seed):
    rng = np.random.default_rng(seed)
    model_hash = inspect_model(model).source_hash
    with TraceWriter(
        path,
        model_hash=model_hash,
        dataset_hash=f"{split}-boundary-{seed}",
        split=split,
        seed=seed,
    ) as writer:
        arrays = {
            "sample_id": np.repeat(np.arange(3, dtype=np.int64), 3),
            "token_id": np.arange(9, dtype=np.int64),
        }
        for layer in range(2):
            hidden = rng.normal(size=(9, 8)).astype(np.float32)
            gate, up, down = load_layer_mlp(model, layer)
            arrays[f"layer_{layer}_mlp_input"] = hidden
            arrays[f"layer_{layer}_mlp_output"] = swiglu(hidden, gate, up, down).astype(
                np.float32
            )
        writer.append(arrays)


def test_exact_gate_sparse_traffic_has_45_percent_boundary():
    traffic = exact_gate_sparse_traffic(576, 1536, 0.175)

    assert traffic["fraction_of_dense"] == pytest.approx(0.45)
    assert traffic["gate_weights_per_token_layer"] == 576 * 1536
    assert traffic["dense_weights_per_token_layer"] == 3 * 576 * 1536
    assert traffic["metadata_included"] is False

    with pytest.raises(ValueError, match="active_fraction"):
        exact_gate_sparse_traffic(8, 12, 1.1)


def test_fully_sparse_traffic_charges_both_input_projections():
    traffic = fully_sparse_mlp_traffic(576, 1536, 288, 538)

    expected = (2 * 1536 * 288 + 576 * 538) / (3 * 576 * 1536)
    assert traffic["fraction_of_dense"] == pytest.approx(expected)
    assert traffic["candidate_recall_applicable"] is False
    assert traffic["metadata_included"] is False


def test_intrinsic_sparse_sweep_fits_calibration_and_measures_validation(tmp_path):
    model = create_tiny_fixture(
        tmp_path / "model",
        hidden_size=8,
        intermediate_size=12,
        num_layers=2,
        num_heads=2,
    )
    model_hash = inspect_model(model).source_hash
    calibration = tmp_path / "calibration"
    validation = tmp_path / "validation"
    _write_trace(calibration, model_hash, split="calibration", seed=120)
    _write_trace(validation, model_hash, split="validation", seed=121)

    report = evaluate_intrinsic_sparse_gate_sweep(
        model,
        calibration,
        validation,
        sparsities=(0.5, 0.85),
        activations=("cats_silu", "fatrelu"),
        calibration_records=6,
        validation_records=6,
        maximum_mean_relative_l2=2.0,
    )

    assert report["experiment"] == "intrinsic_sparse_gate_sweep"
    assert report["calibration"]["records_per_layer"] == 6
    assert report["validation"]["records_per_layer"] == 6
    assert len(report["validation"]["sequence_hashes"]) == 2
    assert len(report["arms"]) == 4
    for arm in report["arms"]:
        assert len(arm["layers"]) == 2
        assert 0 <= arm["activation_sparsity"]["mean"] <= 1
        assert 1 / 3 <= arm["projected_traffic"]["fraction_of_dense"] <= 1
        assert arm["projected_traffic"]["metadata_included"] is False

    json_path, markdown_path = write_intrinsic_sparse_gate_report(
        report, tmp_path / "report"
    )
    assert json_path.is_file()
    assert "Exact gate-selected" in markdown_path.read_text(encoding="utf-8")


def test_intrinsic_sparse_sweep_rejects_role_leakage(tmp_path):
    model = create_tiny_fixture(
        tmp_path / "model",
        hidden_size=8,
        intermediate_size=12,
        num_layers=2,
        num_heads=2,
    )
    model_hash = inspect_model(model).source_hash
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_trace(first, model_hash, split="validation", seed=122)
    _write_trace(second, model_hash, split="validation", seed=123)

    with pytest.raises(ValueError, match="calibration"):
        evaluate_intrinsic_sparse_gate_sweep(
            model,
            first,
            second,
            sparsities=(0.85,),
        )


def test_intrinsic_sparse_boundary_training_uses_hard_budget_and_writes_artifact(
    tmp_path,
):
    model = create_tiny_fixture(
        tmp_path / "model",
        hidden_size=8,
        intermediate_size=12,
        num_layers=2,
        num_heads=2,
    )
    training = tmp_path / "training"
    validation = tmp_path / "validation"
    _write_boundary_trace(training, model, split="calibration", seed=124)
    _write_boundary_trace(validation, model, split="validation", seed=125)

    report = train_intrinsic_sparse_boundaries(
        model,
        training,
        validation,
        tmp_path / "trained",
        layers=(1,),
        target_sparsity=0.85,
        steps=2,
        warmup_steps=0,
        batch_size=4,
        evaluation_interval=1,
        maximum_mean_relative_l2=10.0,
    )

    assert report["experiment"] == "intrinsic_sparse_boundary_training"
    assert report["artifact"]["formal_deployment_artifact"] is False
    assert (tmp_path / "trained" / "intrinsic_sparse_boundaries.safetensors").is_file()
    layer = report["layers"][0]
    assert layer["layer"] == 1
    assert len(layer["history"]) == 3
    assert 1 / 3 <= layer["final"]["traffic_fraction"] <= 1


def test_fully_sparse_boundary_training_uses_exact_counts(tmp_path):
    model = create_tiny_fixture(
        tmp_path / "model",
        hidden_size=8,
        intermediate_size=12,
        num_layers=2,
        num_heads=2,
    )
    training = tmp_path / "training"
    validation = tmp_path / "validation"
    _write_boundary_trace(training, model, split="calibration", seed=126)
    _write_boundary_trace(validation, model, split="validation", seed=127)

    report = train_fully_sparse_boundaries(
        model,
        training,
        validation,
        tmp_path / "fully-sparse",
        layers=(0,),
        input_fraction=0.5,
        intermediate_fraction=1 / 3,
        steps=2,
        warmup_steps=0,
        batch_size=4,
        evaluation_interval=1,
        maximum_mean_relative_l2=10.0,
        maximum_traffic_fraction=1.0,
    )

    assert report["traffic"]["input_count"] == 4
    assert report["traffic"]["intermediate_count"] == 4
    assert report["traffic"]["fraction_of_dense"] == pytest.approx(4 / 9)
    assert report["artifact"]["formal_deployment_artifact"] is False
    assert (tmp_path / "fully-sparse" / "fully_sparse_boundaries.safetensors").is_file()
    assert len(report["layers"][0]["history"]) == 3
