import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from engram.models.fixture import create_tiny_fixture  # noqa: E402
from engram.models.inspection import inspect_model  # noqa: E402
from engram.tracing.format import TraceWriter  # noqa: E402
from engram.training.width_ceiling import (  # noqa: E402
    evaluate_width_pruned_local_ceiling,
)


def _compact_output(states, gate, up, down):
    gate_values = states @ gate.T
    up_values = states @ up.T
    activated = gate_values / (1.0 + np.exp(-gate_values))
    return (activated * up_values) @ down.T


def test_width_local_ceiling_improves_teacher_boundary_fit(tmp_path):
    model = create_tiny_fixture(tmp_path / "model", seed=71)
    inspection = inspect_model(model)
    rng = np.random.default_rng(73)
    target_width = 4
    hidden = inspection.hidden_size
    truth_gate = rng.normal(scale=0.2, size=(target_width, hidden)).astype(np.float32)
    truth_up = rng.normal(scale=0.2, size=(target_width, hidden)).astype(np.float32)
    truth_down = rng.normal(scale=0.2, size=(hidden, target_width)).astype(np.float32)
    initial_gate = truth_gate + rng.normal(scale=0.08, size=truth_gate.shape)
    initial_up = truth_up + rng.normal(scale=0.08, size=truth_up.shape)
    initial_down = truth_down + rng.normal(scale=0.08, size=truth_down.shape)

    checkpoint = tmp_path / "width_pruned_checkpoint.pt"
    torch.save(
        {
            "parameters": {
                "layers.0.gate_weight": torch.tensor(initial_gate, dtype=torch.float32),
                "layers.0.up_weight": torch.tensor(initial_up, dtype=torch.float32),
                "layers.0.down_weight": torch.tensor(initial_down, dtype=torch.float32),
            }
        },
        checkpoint,
    )
    checkpoint.with_suffix(".json").write_text(
        json.dumps(
            {
                "configuration": {
                    "source_model_hash": inspection.source_hash,
                    "target_width": target_width,
                }
            }
        ),
        encoding="utf-8",
    )

    for name, split, count in (("train", "calibration", 96), ("valid", "validation", 48)):
        states = rng.normal(size=(count, hidden)).astype(np.float32)
        targets = _compact_output(states, truth_gate, truth_up, truth_down).astype(
            np.float32
        )
        with TraceWriter(
            tmp_path / name,
            model_hash=inspection.source_hash,
            dataset_hash=f"{name}-dataset",
            split=split,
            seed=79,
        ) as writer:
            writer.append(
                {
                    "layer_0_mlp_input": states,
                    "layer_0_mlp_output": targets,
                }
            )

    report = evaluate_width_pruned_local_ceiling(
        model,
        tmp_path / "train",
        tmp_path / "valid",
        checkpoint,
        tmp_path / "report",
        layers=[0],
        target_width=target_width,
        steps=80,
        batch_size=24,
        learning_rate=3e-3,
        maximum_mean_relative_l2=2.0,
        minimum_improvement_fraction=0.01,
    )

    assert report["summary"]["mean_relative_l2_after"] < report["summary"][
        "mean_relative_l2_before"
    ]
    assert report["screen"]["passed"]
    assert report["screen"]["decision"] == "eligible_for_causal_confirmation"
    assert (tmp_path / "report" / "width_local_ceiling.safetensors").is_file()
