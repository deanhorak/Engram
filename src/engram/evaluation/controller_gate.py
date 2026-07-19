from __future__ import annotations

from typing import Any

import numpy as np

from engram.controller import SharedRecurrentController


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / np.sum(values)


def evaluate_controller_gate(*, seed: int = 73, samples: int = 64, width: int = 16) -> dict[str, Any]:
    """Synthetic Gate 4 instrumentation; not source-trajectory distillation evidence."""
    rng = np.random.default_rng(seed)
    controller = SharedRecurrentController.initialize(
        input_dim=3 * width, state_dim=width, num_stages=4, adapter_rank=2, seed=seed
    )
    vocabulary = rng.normal(size=(64, width))
    fixed_cosines, adaptive_cosines, fixed_kls, adaptive_kls, adaptive_cycles = [], [], [], [], []
    for _ in range(samples):
        initial = rng.normal(size=width)
        supplied = rng.normal(size=3 * width)
        teacher = controller.run(initial, supplied, fixed_cycles=8).state
        fixed = controller.run(initial, supplied, fixed_cycles=4)
        adaptive = controller.run(
            initial, supplied, mode="adaptive", min_cycles=2, max_cycles=8, residual_tolerance=0.04
        )
        teacher_norm = max(float(np.linalg.norm(teacher)), 1e-12)
        for state, cosines, divergences in (
            (fixed.state, fixed_cosines, fixed_kls),
            (adaptive.state, adaptive_cosines, adaptive_kls),
        ):
            cosines.append(float(np.dot(state, teacher) / max(np.linalg.norm(state) * teacher_norm, 1e-12)))
            teacher_probability = _softmax(vocabulary @ teacher)
            student_probability = _softmax(vocabulary @ state)
            divergences.append(
                float(np.sum(teacher_probability * np.log(np.maximum(teacher_probability, 1e-30) / np.maximum(student_probability, 1e-30))))
            )
        adaptive_cycles.append(adaptive.cycles)
    return {
        "schema_version": 1,
        "experiment": "gate_4_shared_controller",
        "status": "synthetic_pipeline_validation",
        "teacher_definition": "same initialized shared controller run for eight cycles",
        "source_transformer_hidden_states": {"status": "not_run"},
        "fixed_cycle": {
            "cycles": 4,
            "mean_hidden_cosine": float(np.mean(fixed_cosines)),
            "mean_final_logit_kl": float(np.mean(fixed_kls)),
        },
        "adaptive_cycle": {
            "mean_cycles": float(np.mean(adaptive_cycles)),
            "min_cycles": int(np.min(adaptive_cycles)),
            "max_cycles": int(np.max(adaptive_cycles)),
            "mean_hidden_cosine": float(np.mean(adaptive_cosines)),
            "mean_final_logit_kl": float(np.mean(adaptive_kls)),
        },
        "quality_claim": None,
    }
