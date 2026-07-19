from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from engram.models.inspection import inspect_model, load_layer_mlp
from engram.semantic.swiglu import neuron_activations
from engram.tracing.format import TraceReader
from engram.utils import percentile


ENERGY_TARGETS = (0.90, 0.95, 0.99)


@dataclass(frozen=True)
class ThresholdResult:
    target: float
    k: int
    fraction: float
    relative_l2: float
    cosine: float


def _similarity(approximation: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    reference_norm = float(np.linalg.norm(reference))
    error_norm = float(np.linalg.norm(reference - approximation))
    if reference_norm <= 1e-12:
        relative_l2 = 0.0 if error_norm <= 1e-12 else float("inf")
    else:
        relative_l2 = error_norm / reference_norm
    approximation_norm = float(np.linalg.norm(approximation))
    if reference_norm <= 1e-12 and approximation_norm <= 1e-12:
        cosine = 1.0
    elif reference_norm <= 1e-12 or approximation_norm <= 1e-12:
        cosine = 0.0
    else:
        cosine = float(np.dot(reference, approximation) / (reference_norm * approximation_norm))
    return relative_l2, max(-1.0, min(1.0, cosine))


def magnitude_oracle_sample(
    activations: np.ndarray,
    values: np.ndarray,
    *,
    targets: Iterable[float] = ENERGY_TARGETS,
) -> tuple[list[ThresholdResult], np.ndarray]:
    """Rank with full activations and value norms, then scan every prefix.

    This is a contribution-magnitude oracle, not the combinatorial best K-subset.
    Scanning all prefixes matters because vector cancellation makes residual energy
    non-monotonic as K increases.
    """
    activations = np.asarray(activations, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if activations.ndim != 1 or values.ndim != 2 or values.shape[0] != activations.size:
        raise ValueError("expected activations [I] and values [I, H]")
    full = activations @ values
    full_energy = float(np.dot(full, full))
    order = np.argsort(-(np.abs(activations) * np.linalg.norm(values, axis=1)), kind="stable")
    target_list = sorted(set(float(target) for target in targets))
    if any(target <= 0.0 or target > 1.0 for target in target_list):
        raise ValueError("energy targets must lie in (0, 1]")
    found: dict[float, ThresholdResult] = {}
    approximation = np.zeros(values.shape[1], dtype=np.float64)

    if full_energy <= 1e-24:
        return [ThresholdResult(target, 0, 0.0, 0.0, 1.0) for target in target_list], order
    for k, index in enumerate(order, start=1):
        approximation += activations[index] * values[index]
        residual = full - approximation
        residual_ratio = float(np.dot(residual, residual) / full_energy)
        for target in target_list:
            if target not in found and residual_ratio <= 1.0 - target + 1e-12:
                relative_l2, cosine = _similarity(approximation, full)
                found[target] = ThresholdResult(
                    target=target,
                    k=k,
                    fraction=k / activations.size,
                    relative_l2=relative_l2,
                    cosine=cosine,
                )
    # K=I is algebraically exact; numerical roundoff can narrowly miss target=1.
    relative_l2, cosine = _similarity(approximation, full)
    for target in target_list:
        found.setdefault(
            target,
            ThresholdResult(target, activations.size, 1.0, relative_l2, cosine),
        )
    return [found[target] for target in target_list], order


def _stats(values: list[float]) -> dict[str, float | int | None]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"mean": None, "median": None, "p95": None, "finite_count": 0}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": percentile(finite, 95),
        "finite_count": len(finite),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"samples": len(rows), "targets": {}}
    for target in ENERGY_TARGETS:
        key = f"{int(target * 100)}pct"
        selected = [row["targets"][key] for row in rows]
        result["targets"][key] = {
            "required_neuron_fraction": _stats([item["fraction"] for item in selected]),
            "relative_l2": _stats([item["relative_l2"] for item in selected]),
            "cosine_similarity": _stats([item["cosine"] for item in selected]),
        }
    result["teacher_reconstruction_relative_l2"] = _stats(
        [row["teacher_reconstruction_relative_l2"] for row in rows]
    )
    return result


def analyze_magnitude_oracle(
    model: str | Path,
    traces: str | Path,
    *,
    max_records: int | None = None,
) -> dict[str, Any]:
    inspection = inspect_model(model)
    reader = TraceReader(traces)
    if reader.manifest["model_hash"] != inspection.source_hash:
        raise ValueError("trace model hash does not match the supplied source model")
    rows: list[dict[str, Any]] = []
    for layer in range(inspection.num_hidden_layers):
        gate, up, down = load_layer_mlp(model, layer)
        values = down.T.astype(np.float64)
        processed = 0
        fields = ["input_type", f"layer_{layer}_mlp_input", f"layer_{layer}_mlp_output"]
        for shard in reader.iter_shards(fields):
            hidden_batch = np.asarray(shard[fields[1]], dtype=np.float64)
            teacher_batch = np.asarray(shard[fields[2]], dtype=np.float64)
            types = np.asarray(shard["input_type"])
            activations_batch = neuron_activations(hidden_batch, gate.astype(np.float64), up.astype(np.float64))
            for hidden_index, (activations, teacher) in enumerate(zip(activations_batch, teacher_batch)):
                if max_records is not None and processed >= max_records:
                    break
                full = activations @ values
                teacher_error, _ = _similarity(full, teacher)
                thresholds, _ = magnitude_oracle_sample(activations, values)
                target_values = {
                    f"{int(item.target * 100)}pct": {
                        "k": item.k,
                        "fraction": item.fraction,
                        "relative_l2": item.relative_l2,
                        "cosine": item.cosine,
                    }
                    for item in thresholds
                }
                rows.append(
                    {
                        "layer": layer,
                        "input_type": str(types[hidden_index]),
                        "teacher_reconstruction_relative_l2": teacher_error,
                        "targets": target_values,
                    }
                )
                processed += 1
            if max_records is not None and processed >= max_records:
                break

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[("all",)].append(row)
        groups[("layer", row["layer"])].append(row)
        groups[("layer_input_type", row["layer"], row["input_type"])].append(row)
    aggregated = []
    for key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        descriptor: dict[str, Any] = {"scope": key[0]}
        if key[0] in {"layer", "layer_input_type"}:
            descriptor["layer"] = key[1]
        if key[0] == "layer_input_type":
            descriptor["input_type"] = key[2]
        aggregated.append({**descriptor, **_aggregate(group_rows)})
    return {
        "schema_version": 1,
        "experiment": "gate_1_mlp_magnitude_oracle",
        "status": "pipeline_validation" if reader.manifest["metadata"].get("fixture_only") else "measured_local_model",
        "fixture_only": bool(reader.manifest["metadata"].get("fixture_only")),
        "source_model_hash": inspection.source_hash,
        "trace_dataset_hash": reader.manifest["dataset_hash"],
        "energy_definition": "retained when ||full - prefix_k||^2 / ||full||^2 <= 1 - target",
        "oracle_definition": (
            "all neuron activations are computed; neurons are ranked by "
            "abs(activation_j) * L2(value_j), and every ranked prefix is scanned"
        ),
        "oracle_limit": "magnitude oracle; not a combinatorial optimum under vector cancellation",
        "background_operator": {"status": "not_run", "planned_milestone": 2},
        "record_count": len(rows),
        "groups": aggregated,
    }
