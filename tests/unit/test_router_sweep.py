import numpy as np
import pytest

from engram.evaluation.router_sweep import evaluate_rank_router_regularization_sweep
from engram.models.fixture import create_tiny_fixture
from engram.models.inspection import inspect_model
from engram.tracing.format import TraceWriter


def _write_trace(path, model_hash, dataset_hash, split, token_offset):
    rng = np.random.default_rng(token_offset)
    with TraceWriter(
        path,
        model_hash=model_hash,
        dataset_hash=dataset_hash,
        split=split,
        seed=token_offset,
        metadata={"fixture_only": True},
    ) as writer:
        for sample in range(3):
            arrays = {
                "sample_id": np.full(2, sample, dtype=np.int64),
                "token_id": np.asarray(
                    [token_offset + sample * 2, token_offset + sample * 2 + 1],
                    dtype=np.int64,
                ),
            }
            for layer in range(2):
                arrays[f"layer_{layer}_mlp_input"] = rng.normal(size=(2, 8)).astype(
                    np.float32
                )
            writer.append(arrays)


def test_rank_router_sweep_reuses_membership_cache(tmp_path):
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
    _write_trace(calibration, model_hash, "calibration-data", "calibration", 10)
    _write_trace(validation, model_hash, "validation-data", "validation", 100)
    arguments = dict(
        regularizations=[1.0, 3.0],
        top_k=4,
        candidate_counts=[6, 8],
        rank=2,
        cache_dir=tmp_path / "cache",
    )

    first = evaluate_rank_router_regularization_sweep(
        model, calibration, validation, **arguments
    )
    second = evaluate_rank_router_regularization_sweep(
        model, calibration, validation, **arguments
    )

    assert first["membership_cache"]["misses"] == 2
    assert second["membership_cache"]["hits"] == 2
    assert len(second["arms"]) == 4
    assert 0.0 <= second["best_arm"]["candidate_recall"]["mean"] <= 1.0
    assert second["data_separation"]["held_out"] is True


def test_rank_router_sweep_rejects_sequence_overlap(tmp_path):
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
    _write_trace(calibration, model_hash, "calibration-data", "calibration", 10)
    _write_trace(validation, model_hash, "validation-data", "validation", 10)

    with pytest.raises(ValueError, match="matching token sequences"):
        evaluate_rank_router_regularization_sweep(
            model,
            calibration,
            validation,
            regularizations=[1.0],
            top_k=4,
            candidate_counts=[6],
            rank=2,
            cache_dir=tmp_path / "cache",
        )
