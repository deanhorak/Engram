from engram.models.fixture import create_tiny_fixture
from engram.semantic.evaluate import evaluate_practical_routing
from engram.tracing.teacher import capture_teacher_traces


def test_practical_router_uses_separate_fit_and_validation_splits(tmp_path):
    model = create_tiny_fixture(tmp_path / "model")
    calibration = tmp_path / "calibration"
    validation = tmp_path / "validation"
    capture_teacher_traces(model, calibration, split="calibration", seed=31, samples=16)
    capture_teacher_traces(model, validation, split="validation", seed=32, samples=8)
    report = evaluate_practical_routing(
        model,
        calibration,
        validation,
        top_k=8,
        candidate_count=16,
        background_rank=4,
    )
    assert report["fixture_only"] is True
    overall = next(group for group in report["groups"] if group["scope"] == "all")
    assert overall["samples"] == 16
    assert 0.0 <= overall["metrics"]["candidate_recall"]["mean"] <= 1.0
    assert report["end_to_end_logit_effect"]["status"] == "not_run"
