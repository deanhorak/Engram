import json

from engram.evaluation.report import write_oracle_report
from engram.models.fixture import create_tiny_fixture
from engram.models.inspection import inspect_model
from engram.semantic.oracle import analyze_magnitude_oracle
from engram.tracing.teacher import capture_teacher_traces


def test_fixture_to_oracle_report(tmp_path):
    model = create_tiny_fixture(tmp_path / "model", seed=21)
    traces = tmp_path / "traces"
    capture_teacher_traces(model, traces, samples=8, seed=22)
    report = analyze_magnitude_oracle(model, traces)
    json_path, markdown_path = write_oracle_report(report, tmp_path / "report")

    assert report["fixture_only"] is True
    assert report["status"] == "pipeline_validation"
    assert report["background_operator"]["status"] == "not_run"
    assert report["record_count"] == 16
    assert any(group["scope"] == "layer_input_type" for group in report["groups"])
    overall = next(group for group in report["groups"] if group["scope"] == "all")
    assert set(overall["targets"]) == {"90pct", "95pct", "99pct"}
    assert overall["teacher_reconstruction_relative_l2"]["p95"] < 1e-5
    assert json.loads(json_path.read_text())["source_model_hash"] == inspect_model(model).source_hash
    assert "make no claim" in markdown_path.read_text()
