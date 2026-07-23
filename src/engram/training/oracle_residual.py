"""Affine residual ceiling around exact sparse semantic-record reads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.evaluation.mlp_intervention import _relative_and_cosine_rows
from engram.models.inspection import inspect_model, load_layer_mlp, resolve_model_path
from engram.semantic.background import LowRankLinearBackground
from engram.tracing.format import TraceReader
from engram.training.gated_background import _oracle_selected_output
from engram.training.structured_experts import _load_trace_field, _stats
from engram.utils import atomic_json


def oracle_residual_traffic_fraction(
    intermediate_size: int,
    top_k: int,
    residual_rank: int,
    hidden_size: int,
) -> float:
    selected = 3 * hidden_size * top_k
    residual = 2 * hidden_size * residual_rank + hidden_size
    dense = 3 * hidden_size * intermediate_size
    return (selected + residual) / dense


def evaluate_oracle_residual_ceiling(
    model: str | Path,
    training_traces: str | Path,
    validation_traces: str | Path,
    out: str | Path,
    *,
    layers: Sequence[int],
    top_k: int = 640,
    ranks: Sequence[int] = (16, 32, 48, 64, 75),
    ridge_factor: float = 0.5,
    max_train_records: int | None = 4096,
    max_validation_records: int | None = 2048,
    maximum_mean_relative_l2: float = 0.10,
) -> dict[str, Any]:
    """Fit ridge/SVD residual maps around a full-information top-K reference."""

    if not layers or not ranks:
        raise ValueError("layers and ranks must be nonempty")
    selected_layers = sorted(set(int(layer) for layer in layers))
    selected_ranks = sorted(set(int(rank) for rank in ranks))
    if ridge_factor < 0 or not np.isfinite(ridge_factor):
        raise ValueError("ridge_factor must be finite and nonnegative")
    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    if selected_layers[0] < 0 or selected_layers[-1] >= inspection.num_hidden_layers:
        raise ValueError("layer index is outside the source model")
    if not 0 < top_k <= inspection.intermediate_size:
        raise ValueError("top_k is outside the intermediate size")
    if selected_ranks[0] <= 0 or selected_ranks[-1] > inspection.hidden_size:
        raise ValueError("rank is outside the hidden size")
    traffic = {
        rank: oracle_residual_traffic_fraction(
            inspection.intermediate_size, top_k, rank, inspection.hidden_size
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

    by_rank: dict[int, list[dict[str, Any]]] = {rank: [] for rank in selected_ranks}
    for layer in selected_layers:
        gate, up, down = load_layer_mlp(model_path, layer)
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
        train_sparse = _oracle_selected_output(train_input, gate, up, down, top_k)
        validation_sparse = _oracle_selected_output(
            validation_input, gate, up, down, top_k
        )
        residual = train_teacher - train_sparse
        centered = train_input.astype(np.float64) - np.mean(train_input, axis=0)
        ridge = ridge_factor * float(np.mean(np.sum(centered**2, axis=0)))
        fitted = LowRankLinearBackground.fit(
            train_input, residual, rank=selected_ranks[-1], ridge=ridge
        )
        baseline, _ = _relative_and_cosine_rows(
            validation_sparse, validation_teacher
        )
        for rank in selected_ranks:
            correction = (
                (validation_input - fitted.input_mean)
                @ fitted.input_factor[:, :rank]
                @ fitted.output_factor[:rank]
                + fitted.output_mean
            )
            relative, cosine = _relative_and_cosine_rows(
                validation_sparse + correction, validation_teacher
            )
            by_rank[rank].append(
                {
                    "layer": layer,
                    "ridge": ridge,
                    "sparse_relative_l2": _stats(baseline.tolist()),
                    "corrected_relative_l2": _stats(relative.tolist()),
                    "corrected_cosine": _stats(cosine.tolist()),
                }
            )
    arms = []
    for rank in selected_ranks:
        rows = by_rank[rank]
        before = float(
            np.mean([row["sparse_relative_l2"]["mean"] for row in rows])
        )
        after = float(
            np.mean([row["corrected_relative_l2"]["mean"] for row in rows])
        )
        checks = {
            "mean_relative_l2": after <= maximum_mean_relative_l2,
            "every_layer_improved": all(
                row["corrected_relative_l2"]["mean"]
                < row["sparse_relative_l2"]["mean"]
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
    best = min(arms, key=lambda arm: arm["mean_relative_l2_after"])
    report = {
        "schema_version": 1,
        "experiment": "oracle_topk_affine_residual_ceiling",
        "source_model_hash": inspection.source_hash,
        "configuration": {
            "layers": selected_layers,
            "top_k": top_k,
            "ranks": selected_ranks,
            "ridge_factor": ridge_factor,
        },
        "provenance": {
            "training_trace_dataset_hash": training_reader.manifest["dataset_hash"],
            "validation_trace_dataset_hash": validation_reader.manifest["dataset_hash"],
        },
        "arms": arms,
        "selected": best,
        "screen": {
            "passed": any(arm["passed"] for arm in arms),
            "decision": (
                "eligible_for_causal_oracle_test"
                if any(arm["passed"] for arm in arms)
                else "reject_oracle_affine_residual"
            ),
            "maximum_mean_relative_l2": maximum_mean_relative_l2,
            "caveat": "full-information top-K selection is not deployable",
        },
    }
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    atomic_json(target / "oracle_residual_ceiling.json", report)
    return report


__all__ = ["evaluate_oracle_residual_ceiling", "oracle_residual_traffic_fraction"]
