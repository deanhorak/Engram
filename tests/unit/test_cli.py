import pytest

from engram.cli import _parser, main


def test_composite_gate_cannot_overwrite_an_input_report(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "mlp_intervention.json"
    second = second_dir / "mlp_intervention.json"
    first.write_text("{}\n", encoding="utf-8")
    second.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not overwrite"):
        main(
            [
                "gate-mlp-intervention",
                "--report",
                str(first),
                str(second),
                "--out",
                str(first_dir),
            ]
        )


def test_width_pruned_cli_exposes_strict_q4_deployment_flag(tmp_path):
    arguments = _parser().parse_args(
        [
            "train-width-pruned-student",
            "--model",
            "model",
            "--training-dataset",
            str(tmp_path / "train.jsonl"),
            "--validation-dataset",
            str(tmp_path / "validation.jsonl"),
            "--out",
            str(tmp_path / "out"),
            "--strict-q4-deployment",
            "--fake-q4-training",
            "--target-widths",
            "384",
            "672",
        ]
    )

    assert arguments.strict_q4_deployment
    assert arguments.fake_q4_training
    assert arguments.target_widths == [384, 672]


def test_width_pruned_cli_forwards_layer_schedule(tmp_path, monkeypatch):
    captured = {}

    def fake_train(*_args, **kwargs):
        captured.update(kwargs)
        return {"gate": {"passed": True}}

    monkeypatch.setattr("engram.cli.train_width_pruned_student", fake_train)
    result = main(
        [
            "train-width-pruned-student",
            "--model",
            "model",
            "--training-dataset",
            str(tmp_path / "train.jsonl"),
            "--validation-dataset",
            str(tmp_path / "validation.jsonl"),
            "--out",
            str(tmp_path / "out"),
            "--target-widths",
            "384",
            "672",
            "--strict-q4-deployment",
            "--fake-q4-training",
        ]
    )

    assert result == 0
    assert captured["target_width"] == 672
    assert captured["target_widths"] == [384, 672]
    assert captured["fake_q4_training"]


def test_sparse_teacher_cli_exposes_progressive_layer_pilot_controls(tmp_path):
    arguments = _parser().parse_args(
        [
            "train-sparse-student",
            "--model",
            "model",
            "--calibration-dataset",
            str(tmp_path / "calibration.jsonl"),
            "--validation-dataset",
            str(tmp_path / "validation.jsonl"),
            "--calibration-traces",
            str(tmp_path / "traces"),
            "--out",
            str(tmp_path / "out"),
            "--layers",
            "25",
            "26",
            "27",
            "28",
            "29",
            "--exact-dense-start",
            "--dense-warmup-steps",
            "32",
            "--anneal-steps",
            "256",
            "--checkpoint-selection-records",
            "128",
            "--checkpoint-selection-every",
            "64",
        ]
    )

    assert arguments.layers == [25, 26, 27, 28, 29]
    assert arguments.exact_dense_start
    assert arguments.dense_warmup_steps == 32
    assert arguments.anneal_steps == 256
    assert arguments.checkpoint_selection_records == 128
    assert arguments.checkpoint_selection_every == 64


def test_trace_cli_exposes_selected_layers_and_dry_run_plan(tmp_path):
    arguments = _parser().parse_args(
        [
            "trace",
            "--model",
            "model",
            "--dataset",
            str(tmp_path / "corpus.jsonl"),
            "--out",
            str(tmp_path / "traces"),
            "--layers",
            "14",
            "--samples",
            "2048",
            "--mlp-only",
            "--dry-run",
            "--plan-out",
            str(tmp_path / "plan.json"),
        ]
    )

    assert arguments.layers == [14]
    assert arguments.samples == 2048
    assert arguments.mlp_only
    assert arguments.dry_run
    assert arguments.plan_out == tmp_path / "plan.json"
