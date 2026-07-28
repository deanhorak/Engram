"""Reproducible systems confirmation for the native packed OLMoE Q7 path."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from engram.evaluation.olmoe_q7_native import OLMoEQ7NativeKernel
from engram.models.olmoe_q7 import LoadedOLMoEQ7Artifact
from engram.utils import atomic_json


def evaluate_olmoe_q7_native_systems(
    artifact: str | Path,
    library: str | Path,
    out: str | Path,
    *,
    layer: int = 0,
    states: int = 1,
    threads: int = 1,
    seed: int = 7,
    maximum_relative_l2: float = 1e-5,
    maximum_traffic_fraction: float = 0.45,
) -> dict[str, Any]:
    """Compare direct packed execution to an independent decoded reference."""

    if states <= 0:
        raise ValueError("states must be positive")
    started = time.perf_counter()
    with LoadedOLMoEQ7Artifact(artifact) as decoded_artifact:
        validation_seconds = time.perf_counter() - started
        layout = decoded_artifact.layout
        if not 0 <= layer < layout.layer_count:
            raise ValueError("layer is outside the Q7 artifact")
        rng = np.random.default_rng(seed)
        hidden = rng.normal(size=(states, layout.hidden_size)).astype(np.float32)
        router = decoded_artifact.router(layer)
        logits = hidden @ router.T
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        selected = np.argsort(-probabilities, axis=1, kind="stable")[
            :, : layout.top_k
        ]
        reference = np.zeros_like(hidden)
        for expert in np.unique(selected):
            weights = decoded_artifact.expert(layer, int(expert))
            rows = np.nonzero(np.any(selected == expert, axis=1))[0]
            for row in rows:
                state = hidden[row]
                gate = state @ weights["gate"].T
                activation = (gate / (1.0 + np.exp(-gate))) * (
                    state @ weights["up"].T
                )
                reference[row] += probabilities[row, expert] * (
                    activation @ weights["down"].T
                )
        artifact_bytes = layout.file_bytes
        ideal_all_expert_q4 = (
            layout.num_experts
            * 3
            * layout.hidden_size
            * layout.intermediate_size
            // 2
        )

    native_open_started = time.perf_counter()
    with OLMoEQ7NativeKernel(artifact, library, threads=threads) as kernel:
        native_validation_seconds = time.perf_counter() - native_open_started
        native = kernel.forward(layer, hidden)
    difference = native.output - reference
    reference_norm = max(float(np.linalg.norm(reference)), 1e-12)
    relative_l2 = float(np.linalg.norm(difference) / reference_norm)
    maximum_absolute_error = float(np.max(np.abs(difference)))
    selection_exact = bool(np.array_equal(native.selected_experts, selected))
    outputs_finite = bool(np.isfinite(native.output).all())
    traffic_fraction = (
        native.metrics["scheduled_stream_bytes"] / states / ideal_all_expert_q4
    )
    checks = {
        "artifact_strictly_validated": True,
        "native_route_exact": selection_exact,
        "native_output_finite": outputs_finite,
        "native_decoded_relative_l2_within_limit": (
            relative_l2 <= maximum_relative_l2
        ),
        "scheduled_packed_traffic_within_limit": (
            traffic_fraction <= maximum_traffic_fraction
        ),
    }
    result = {
        "schema_version": 1,
        "experiment": "olmoe_native_q7_systems_confirmation",
        "artifact": str(Path(artifact).resolve()),
        "library": str(Path(library).resolve()),
        "artifact_bytes": artifact_bytes,
        "layer": layer,
        "states": states,
        "threads": threads,
        "seed": seed,
        "validation_seconds": {
            "python_independent_reader": validation_seconds,
            "native_reader": native_validation_seconds,
        },
        "parity": {
            "route_exact": selection_exact,
            "maximum_absolute_error": maximum_absolute_error,
            "relative_l2": relative_l2,
            "maximum_relative_l2": maximum_relative_l2,
            "outputs_finite": outputs_finite,
        },
        "traffic": {
            **native.metrics,
            "all_expert_ideal_q4_bytes_per_layer": ideal_all_expert_q4,
            "scheduled_bytes_per_state": (
                native.metrics["scheduled_stream_bytes"] / states
            ),
            "fraction_of_all_expert_ideal_q4": traffic_fraction,
            "maximum_fraction": maximum_traffic_fraction,
        },
        "checks": checks,
        "gate_passed": all(checks.values()),
    }
    atomic_json(out, result)
    return result


__all__ = ["evaluate_olmoe_q7_native_systems"]
