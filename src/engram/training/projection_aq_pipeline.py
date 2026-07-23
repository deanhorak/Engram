"""Trace/model orchestration for projection-local additive-quantization screens."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.evaluation.router_sweep import _sequence_hashes
from engram.models.inspection import inspect_model, load_layer_mlp, resolve_model_path
from engram.tracing.format import TraceReader
from engram.training.projection_aq_boundaries import train_projection_aq_boundaries
from engram.training.structured_experts import _load_trace_field
from engram.utils import atomic_json


def train_projection_aq_layers(
    model: str | Path,
    training_traces: str | Path,
    validation_traces: str | Path,
    out: str | Path,
    *,
    layers: Sequence[int],
    p_steps_per_cycle: int = 64,
    v_cycles: int = 2,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    checkpoint_interval: int = 16,
    fit_iterations: int = 12,
    fit_sample_limit: int | None = 65_536,
    v_max_records: int | None = 1024,
    v_change_fraction: float = 0.01,
    selection_records: int | None = 512,
    maximum_mean_relative_l2: float = 0.08,
    maximum_p95_relative_l2: float = 0.18,
    minimum_mean_cosine: float = 0.99,
    max_train_records: int | None = 4096,
    max_validation_records: int | None = 2048,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train selected layers while preserving a disjoint held-out trace split."""

    if not layers:
        raise ValueError("at least one layer is required")
    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    selected_layers = sorted(set(int(layer) for layer in layers))
    if selected_layers[0] < 0 or selected_layers[-1] >= inspection.num_hidden_layers:
        raise ValueError("layer index is outside the source model")

    training_reader = TraceReader(training_traces)
    validation_reader = TraceReader(validation_traces)
    for name, reader, split in (
        ("training", training_reader, "calibration"),
        ("validation", validation_reader, "validation"),
    ):
        if reader.manifest["model_hash"] != inspection.source_hash:
            raise ValueError(f"{name} trace/model hash mismatch")
        if reader.manifest["split"] != split:
            raise ValueError(f"expected {split!r} {name} traces")
    if training_reader.manifest["dataset_hash"] == validation_reader.manifest["dataset_hash"]:
        raise ValueError("training and validation boundary datasets must differ")
    if set(_sequence_hashes(training_reader)).intersection(
        _sequence_hashes(validation_reader)
    ):
        raise ValueError("training and validation boundary sequences overlap")

    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    layer_reports: dict[str, Any] = {}
    for layer in selected_layers:
        gate, up, down = load_layer_mlp(model_path, layer)
        fields = {}
        for split, reader, limit in (
            ("training", training_reader, max_train_records),
            ("validation", validation_reader, max_validation_records),
        ):
            fields[f"{split}_inputs"] = _load_trace_field(
                reader, f"layer_{layer}_mlp_input", limit
            ).astype(np.float32)
            fields[f"{split}_outputs"] = _load_trace_field(
                reader, f"layer_{layer}_mlp_output", limit
            ).astype(np.float32)
        result = train_projection_aq_boundaries(
            gate,
            up,
            down,
            fields["training_inputs"],
            fields["training_outputs"],
            fields["validation_inputs"],
            fields["validation_outputs"],
            artifact_dir=destination / f"layer_{layer}",
            p_steps_per_cycle=p_steps_per_cycle,
            v_cycles=v_cycles,
            batch_size=batch_size,
            learning_rate=learning_rate,
            checkpoint_interval=checkpoint_interval,
            fit_iterations=fit_iterations,
            fit_sample_limit=fit_sample_limit,
            v_max_records=v_max_records,
            v_change_fraction=v_change_fraction,
            selection_records=selection_records,
            maximum_mean_relative_l2=maximum_mean_relative_l2,
            maximum_p95_relative_l2=maximum_p95_relative_l2,
            minimum_mean_cosine=minimum_mean_cosine,
            seed=seed + layer,
            device=device,
        )
        atomic_json(destination / f"layer_{layer}" / "report.json", result.report)
        layer_reports[str(layer)] = result.report

    passed = all(report["screen"]["passed"] for report in layer_reports.values())
    report = {
        "schema_version": 1,
        "experiment": "projection_local_activation_aware_aq_layer_screen",
        "model": str(model_path),
        "model_hash": inspection.source_hash,
        "training_dataset_hash": training_reader.manifest["dataset_hash"],
        "validation_dataset_hash": validation_reader.manifest["dataset_hash"],
        "layers": layer_reports,
        "screen": {
            "passed": passed,
            "scope": "held_out_cached_mlp_boundaries_not_causal_gate",
            "all_layers_strictly_below_45_percent_dense_q4": all(
                report["traffic"]["strictly_below_45_percent"]
                for report in layer_reports.values()
            ),
        },
    }
    atomic_json(destination / "projection_aq_layers.json", report)
    return report


__all__ = ["train_projection_aq_layers"]
