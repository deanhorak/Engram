"""Low-rank affine residual screening for a compact SwiGLU core."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.evaluation.mlp_intervention import _relative_and_cosine_rows
from engram.models.inspection import inspect_model, resolve_model_path
from engram.semantic.background import LowRankLinearBackground
from engram.semantic.swiglu import silu
from engram.tracing.format import TraceReader
from engram.training.structured_experts import _load_trace_field, _stats
from engram.utils import atomic_json, sha256_file


def width_residual_traffic_fraction(
    intermediate_size: int,
    compact_width: int,
    residual_rank: int,
    hidden_size: int,
) -> float:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (intermediate_size, compact_width, residual_rank, hidden_size)
    ):
        raise ValueError("traffic dimensions must be positive integers")
    if compact_width > intermediate_size or residual_rank > hidden_size:
        raise ValueError("compact width/rank exceeds source dimensions")
    compact = 3 * hidden_size * compact_width
    residual = 2 * hidden_size * residual_rank + hidden_size
    dense = 3 * hidden_size * intermediate_size
    return (compact + residual) / dense


def evaluate_width_residual_sweep(
    model: str | Path,
    training_traces: str | Path,
    validation_traces: str | Path,
    initial_checkpoint: str | Path,
    out: str | Path,
    *,
    layers: Sequence[int],
    compact_width: int = 672,
    ranks: Sequence[int] = (8, 16, 24, 28),
    ridge_factor: float = 0.5,
    max_train_records: int | None = 4096,
    max_validation_records: int | None = 2048,
    maximum_mean_relative_l2: float = 0.10,
) -> dict[str, Any]:
    """Fit one ridge/SVD map to the compact-core residual at each selected layer."""

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("install engram-lm[conversion] for residual fitting") from exc
    if not layers or not ranks:
        raise ValueError("layers and ranks must be nonempty")
    selected_layers = sorted(set(int(layer) for layer in layers))
    selected_ranks = sorted(set(int(rank) for rank in ranks))
    if ridge_factor < 0 or not np.isfinite(ridge_factor):
        raise ValueError("ridge_factor must be finite and nonnegative")
    if maximum_mean_relative_l2 <= 0:
        raise ValueError("maximum_mean_relative_l2 must be positive")

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    if selected_layers[0] < 0 or selected_layers[-1] >= inspection.num_hidden_layers:
        raise ValueError("layer index is outside the source model")
    if selected_ranks[0] <= 0 or selected_ranks[-1] > inspection.hidden_size:
        raise ValueError("residual rank is outside the hidden size")
    traffic = {
        rank: width_residual_traffic_fraction(
            inspection.intermediate_size,
            compact_width,
            rank,
            inspection.hidden_size,
        )
        for rank in selected_ranks
    }

    training_reader = TraceReader(training_traces)
    validation_reader = TraceReader(validation_traces)
    for name, reader in (("training", training_reader), ("validation", validation_reader)):
        if reader.manifest["model_hash"] != inspection.source_hash:
            raise ValueError(f"{name} trace/model hash mismatch")
    if training_reader.manifest["dataset_hash"] == validation_reader.manifest["dataset_hash"]:
        raise ValueError("training and validation traces must use different datasets")
    if validation_reader.manifest["split"] != "validation":
        raise ValueError("validation traces must declare the validation split")

    checkpoint_path = Path(initial_checkpoint)
    checkpoint_manifest_path = checkpoint_path.with_suffix(".json")
    if not checkpoint_path.is_file() or not checkpoint_manifest_path.is_file():
        raise ValueError("width-pruned checkpoint and manifest are required")
    checkpoint_manifest = json.loads(
        checkpoint_manifest_path.read_text(encoding="utf-8")
    )
    configuration = checkpoint_manifest.get("configuration", {})
    if (
        configuration.get("source_model_hash") != inspection.source_hash
        or configuration.get("target_width") != compact_width
    ):
        raise ValueError("width-pruned checkpoint model/width mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parameters = checkpoint.get("parameters", {})
    del checkpoint

    def compact_output(states: np.ndarray, layer: int) -> np.ndarray:
        gate = parameters[f"layers.{layer}.gate_weight"].float().numpy()
        up = parameters[f"layers.{layer}.up_weight"].float().numpy()
        down = parameters[f"layers.{layer}.down_weight"].float().numpy()
        gate_values = states @ gate.T
        up_values = states @ up.T
        return ((silu(gate_values) * up_values) @ down.T).astype(np.float32)

    accumulated: dict[int, list[dict[str, Any]]] = {
        rank: [] for rank in selected_ranks
    }
    artifacts: dict[int, dict[str, Any]] = {}
    for layer in selected_layers:
        train_input = _load_trace_field(
            training_reader, f"layer_{layer}_mlp_input", max_train_records
        ).astype(np.float32)
        train_teacher = _load_trace_field(
            training_reader, f"layer_{layer}_mlp_output", max_train_records
        ).astype(np.float32)
        validation_input = _load_trace_field(
            validation_reader, f"layer_{layer}_mlp_input", max_validation_records
        ).astype(np.float32)
        validation_teacher = _load_trace_field(
            validation_reader, f"layer_{layer}_mlp_output", max_validation_records
        ).astype(np.float32)
        train_core = compact_output(train_input, layer)
        validation_core = compact_output(validation_input, layer)
        residual = train_teacher - train_core
        centered = train_input.astype(np.float64) - np.mean(train_input, axis=0)
        ridge = ridge_factor * float(np.mean(np.sum(centered**2, axis=0)))
        fitted = LowRankLinearBackground.fit(
            train_input,
            residual,
            rank=selected_ranks[-1],
            ridge=ridge,
        )
        for rank in selected_ranks:
            prediction = (
                (validation_input - fitted.input_mean)
                @ fitted.input_factor[:, :rank]
                @ fitted.output_factor[:rank]
                + fitted.output_mean
            )
            corrected = validation_core + prediction
            relative, cosine = _relative_and_cosine_rows(
                corrected, validation_teacher
            )
            core_relative, _ = _relative_and_cosine_rows(
                validation_core, validation_teacher
            )
            accumulated[rank].append(
                {
                    "layer": layer,
                    "ridge": ridge,
                    "core_relative_l2": _stats(core_relative.tolist()),
                    "corrected_relative_l2": _stats(relative.tolist()),
                    "corrected_cosine": _stats(cosine.tolist()),
                }
            )
        artifacts[layer] = fitted

    arms = []
    for rank in selected_ranks:
        rows = accumulated[rank]
        before = float(np.mean([row["core_relative_l2"]["mean"] for row in rows]))
        after = float(
            np.mean([row["corrected_relative_l2"]["mean"] for row in rows])
        )
        checks = {
            "mean_relative_l2": after <= maximum_mean_relative_l2,
            "every_layer_improved": all(
                row["corrected_relative_l2"]["mean"]
                < row["core_relative_l2"]["mean"]
                for row in rows
            ),
            "projected_traffic": traffic[rank] <= 0.45,
        }
        arms.append(
            {
                "rank": rank,
                "projected_traffic_fraction": traffic[rank],
                "mean_relative_l2_before": before,
                "mean_relative_l2_after": after,
                "improvement_fraction": (before - after) / max(before, 1e-12),
                "layers": rows,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    eligible = [arm for arm in arms if arm["passed"]]
    best = min(arms, key=lambda arm: arm["mean_relative_l2_after"])
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    artifact_path = target / "width_linear_residual.safetensors"
    tensors = {}
    for layer, fitted in artifacts.items():
        rank = int(best["rank"])
        bias = fitted.output_mean - (
            fitted.input_mean @ fitted.input_factor[:, :rank]
        ) @ fitted.output_factor[:rank]
        tensors[f"layers.{layer}.input_factor"] = torch.from_numpy(
            fitted.input_factor[:, :rank].astype(np.float32)
        ).contiguous()
        tensors[f"layers.{layer}.output_factor"] = torch.from_numpy(
            fitted.output_factor[:rank].astype(np.float32)
        ).contiguous()
        tensors[f"layers.{layer}.bias"] = torch.from_numpy(
            bias.astype(np.float32)
        ).contiguous()
    save_file(
        tensors,
        artifact_path,
        metadata={
            "format": "engram_width_linear_residual_v1",
            "source_model_hash": inspection.source_hash,
            "compact_width": str(compact_width),
            "rank": str(best["rank"]),
        },
    )
    report = {
        "schema_version": 1,
        "experiment": "width_pruned_linear_residual_sweep",
        "source_model_hash": inspection.source_hash,
        "configuration": {
            "layers": selected_layers,
            "compact_width": compact_width,
            "ranks": selected_ranks,
            "ridge_factor": ridge_factor,
        },
        "provenance": {
            "training_trace_dataset_hash": training_reader.manifest["dataset_hash"],
            "validation_trace_dataset_hash": validation_reader.manifest["dataset_hash"],
            "initial_checkpoint": str(checkpoint_path.resolve()),
            "initial_checkpoint_sha256": sha256_file(checkpoint_path),
        },
        "arms": arms,
        "selected": best,
        "screen": {
            "passed": bool(eligible),
            "decision": (
                "eligible_for_causal_test" if eligible else "reject_linear_residual"
            ),
            "maximum_mean_relative_l2": maximum_mean_relative_l2,
        },
        "artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": sha256_file(artifact_path),
        },
    }
    atomic_json(target / "width_residual_sweep.json", report)
    return report


__all__ = ["evaluate_width_residual_sweep", "width_residual_traffic_fraction"]
