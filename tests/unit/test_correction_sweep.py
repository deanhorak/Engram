import numpy as np

from engram.evaluation.correction_sweep import evaluate_correction_capsule_sweep
from engram.models.fixture import create_tiny_fixture
from engram.models.inspection import inspect_model
from engram.tracing.format import TraceWriter


def _trace(path, model_hash, dataset_hash, split, seed):
    rng = np.random.default_rng(seed)
    with TraceWriter(
        path,
        model_hash=model_hash,
        dataset_hash=dataset_hash,
        split=split,
        seed=seed,
    ) as writer:
        for sample in range(4):
            arrays = {
                "sample_id": np.full(3, sample, dtype=np.int64),
                "token_id": np.arange(seed + sample * 3, seed + sample * 3 + 3),
            }
            for layer in range(2):
                arrays[f"layer_{layer}_mlp_input"] = rng.normal(size=(3, 8)).astype(
                    np.float32
                )
            writer.append(arrays)


def test_correction_sweep_reports_local_improvement_and_accounting(tmp_path):
    model = create_tiny_fixture(
        tmp_path / "model",
        hidden_size=8,
        intermediate_size=12,
        num_layers=2,
        num_heads=2,
    )
    source_hash = inspect_model(model).source_hash
    calibration = tmp_path / "calibration"
    validation = tmp_path / "validation"
    _trace(calibration, source_hash, "calibration-data", "calibration", 10)
    _trace(validation, source_hash, "validation-data", "validation", 100)

    report = evaluate_correction_capsule_sweep(
        model,
        calibration,
        validation,
        membership_cache=tmp_path / "cache",
        router_rank=2,
        router_regularization=3.0,
        top_k=4,
        candidate_count=8,
        capsule_counts=[1, 2],
        capsule_ranks=[1],
        capsule_ridge=2.0,
        calibration_records=12,
        validation_records=12,
    )

    assert len(report["arms"]) == 2
    assert len(report["per_layer"]) == 2
    assert report["membership_cache"]["misses"] == 2
    assert report["data_separation"]["held_out"] is True
    for arm in report["arms"]:
        assert arm["parameter_bytes_float32_per_layer"] > 0
        assert arm["logical_correction_bytes_per_token_all_layers"] > 0
        assert 0.0 <= arm["match_fraction"] <= 1.0
