from engram.evaluation.controller_gate import evaluate_controller_gate


def test_controller_gate_is_measured_and_synthetic_labeled():
    report = evaluate_controller_gate(samples=8, width=8)
    assert report["status"] == "synthetic_pipeline_validation"
    assert report["source_transformer_hidden_states"]["status"] == "not_run"
    assert 2 <= report["adaptive_cycle"]["mean_cycles"] <= 8
