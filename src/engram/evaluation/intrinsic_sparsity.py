"""Trace-only screening for exact gate-selected semantic memory.

The deployment mechanism evaluates the complete gate projection, applies an
activation threshold, and reads up/down records only where the gate is exactly
active. Unlike approximate top-K routing, every active record is known exactly
after the gate scan and candidate recall is therefore not applicable.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from engram.models.inspection import inspect_model, load_layer_mlp, resolve_model_path
from engram.semantic.swiglu import neuron_activations, silu
from engram.tracing.format import TraceReader
from engram.utils import atomic_json, percentile, sha256_json

INTRINSIC_SPARSE_REFERENCES = (
    {
        "title": "ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models",
        "url": "https://arxiv.org/abs/2402.13516",
    },
    {
        "title": "CATS: Contextually-Aware Thresholding for Sparsity in Large Language Models",
        "url": "https://arxiv.org/abs/2404.08763",
    },
)


def exact_gate_sparse_traffic(
    hidden_size: int,
    intermediate_size: int,
    active_fraction: float,
    *,
    bytes_per_weight: float = 0.5,
) -> dict[str, int | float]:
    """Return ideal cold-weight traffic for a full gate plus active up/down.

    ``bytes_per_weight=0.5`` is the frozen dense-Q4 reference. The calculation
    intentionally excludes headers, scales, indices, and cache-line padding;
    an artifact must leave headroom for those costs before it can pass.
    """

    if (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or hidden_size <= 0
        or isinstance(intermediate_size, bool)
        or not isinstance(intermediate_size, int)
        or intermediate_size <= 0
    ):
        raise ValueError("hidden and intermediate sizes must be positive integers")
    if not np.isfinite(active_fraction) or not 0 <= active_fraction <= 1:
        raise ValueError("active_fraction must lie in [0, 1]")
    if not np.isfinite(bytes_per_weight) or bytes_per_weight <= 0:
        raise ValueError("bytes_per_weight must be finite and positive")

    projection_weights = hidden_size * intermediate_size
    dense_weights = 3 * projection_weights
    selected_weights = projection_weights * (1.0 + 2.0 * active_fraction)
    return {
        "bytes_per_weight": float(bytes_per_weight),
        "gate_weights_per_token_layer": projection_weights,
        "active_up_down_weights_per_token_layer": float(
            2 * projection_weights * active_fraction
        ),
        "projected_weights_per_token_layer": float(selected_weights),
        "dense_weights_per_token_layer": dense_weights,
        "projected_bytes_per_token_layer": float(selected_weights * bytes_per_weight),
        "dense_bytes_per_token_layer": float(dense_weights * bytes_per_weight),
        "fraction_of_dense": float(selected_weights / dense_weights),
        "maximum_active_fraction_at_45_percent": 0.175,
        "metadata_included": False,
    }


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError("cannot summarize empty or non-finite values")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": percentile(array, 95),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _load_states(reader: TraceReader, layer: int, limit: int | None) -> np.ndarray:
    field = f"layer_{layer}_mlp_input"
    batches: list[np.ndarray] = []
    count = 0
    for shard in reader.iter_shards([field]):
        values = np.asarray(shard[field], dtype=np.float32)
        if limit is not None:
            remaining = limit - count
            if remaining <= 0:
                break
            values = values[:remaining]
        batches.append(values)
        count += len(values)
    if not batches:
        raise ValueError(f"trace contains no MLP inputs for layer {layer}")
    return np.concatenate(batches)


def _sequence_hashes(reader: TraceReader, limit: int | None) -> list[str]:
    sequences: dict[int, list[int]] = defaultdict(list)
    count = 0
    for shard in reader.iter_shards(["sample_id", "token_id"]):
        sample_ids = np.asarray(shard["sample_id"])
        token_ids = np.asarray(shard["token_id"])
        if (
            sample_ids.ndim != 1
            or token_ids.ndim != 1
            or sample_ids.shape != token_ids.shape
        ):
            raise ValueError("trace sample_id/token_id fields must be matching vectors")
        if limit is not None:
            remaining = limit - count
            if remaining <= 0:
                break
            sample_ids = sample_ids[:remaining]
            token_ids = token_ids[:remaining]
        for sample_id, token_id in zip(sample_ids, token_ids, strict=True):
            sequences[int(sample_id)].append(int(token_id))
        count += len(sample_ids)
    if not sequences:
        raise ValueError("trace contains no sequence provenance")
    return [sha256_json({"input_ids": sequences[index]}) for index in sorted(sequences)]


def _relative_and_cosine_rows(
    approximation: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    approximate = np.asarray(approximation, dtype=np.float64)
    exact = np.asarray(reference, dtype=np.float64)
    exact_norm = np.linalg.norm(exact, axis=1)
    approximate_norm = np.linalg.norm(approximate, axis=1)
    relative = np.linalg.norm(approximate - exact, axis=1) / np.maximum(
        exact_norm, 1e-12
    )
    cosine = np.sum(approximate * exact, axis=1) / np.maximum(
        exact_norm * approximate_norm, 1e-12
    )
    cosine[(exact_norm <= 1e-12) & (approximate_norm <= 1e-12)] = 1.0
    return relative, np.clip(cosine, -1.0, 1.0)


def _gate_values(logits: np.ndarray, activation: str) -> np.ndarray:
    if activation == "cats_silu":
        return silu(logits)
    if activation == "fatrelu":
        return np.maximum(logits, 0.0)
    raise ValueError("activation must be 'cats_silu' or 'fatrelu'")


def _threshold(values: np.ndarray, sparsity: float, activation: str) -> float:
    magnitudes = np.abs(values) if activation == "cats_silu" else values
    # "higher" guarantees at least the requested fraction is at or below the
    # cutoff before the strict comparison below.
    try:
        return float(np.quantile(magnitudes, sparsity, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        return float(np.quantile(magnitudes, sparsity, interpolation="higher"))


def evaluate_intrinsic_sparse_gate_sweep(
    model: str | Path,
    calibration_traces: str | Path,
    validation_traces: str | Path,
    *,
    sparsities: Sequence[float] = (0.5, 0.7, 0.8, 0.825, 0.85, 0.9),
    activations: Sequence[str] = ("cats_silu", "fatrelu"),
    calibration_records: int | None = None,
    validation_records: int | None = None,
    maximum_mean_relative_l2: float = 0.18,
    maximum_traffic_fraction: float = 0.45,
) -> dict[str, Any]:
    """Fit per-layer cutoffs on calibration traces and screen held-out error."""

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    calibration = TraceReader(calibration_traces)
    validation = TraceReader(validation_traces)
    for reader in (calibration, validation):
        if reader.manifest["model_hash"] != inspection.source_hash:
            raise ValueError("trace/model hash mismatch")
    if calibration.manifest["split"] != "calibration":
        raise ValueError("expected calibration traces")
    if validation.manifest["split"] != "validation":
        raise ValueError("expected validation traces")
    if calibration_records is not None and calibration_records <= 0:
        raise ValueError("calibration_records must be positive")
    if validation_records is not None and validation_records <= 0:
        raise ValueError("validation_records must be positive")
    sparsity_values = tuple(dict.fromkeys(float(value) for value in sparsities))
    if not sparsity_values or any(
        not np.isfinite(value) or not 0 < value < 1 for value in sparsity_values
    ):
        raise ValueError("sparsities must lie in (0, 1)")
    activation_values = tuple(dict.fromkeys(str(value) for value in activations))
    if not activation_values or any(
        value not in {"cats_silu", "fatrelu"} for value in activation_values
    ):
        raise ValueError("unsupported intrinsic sparse activation")
    if not np.isfinite(maximum_mean_relative_l2) or maximum_mean_relative_l2 <= 0:
        raise ValueError("maximum_mean_relative_l2 must be positive")
    if (
        not np.isfinite(maximum_traffic_fraction)
        or not 0 < maximum_traffic_fraction <= 1
    ):
        raise ValueError("maximum_traffic_fraction must lie in (0, 1]")

    accumulators: dict[
        tuple[str, float], dict[str, list[float] | list[dict[str, Any]]]
    ] = {
        (activation, sparsity): {
            "relative": [],
            "cosine": [],
            "active": [],
            "layers": [],
        }
        for activation in activation_values
        for sparsity in sparsity_values
    }
    for layer in range(inspection.num_hidden_layers):
        calibration_states = _load_states(calibration, layer, calibration_records)
        validation_states = _load_states(validation, layer, validation_records)
        gate, up, down = load_layer_mlp(model_path, layer)
        reference_activation = neuron_activations(validation_states, gate, up)
        reference_output = reference_activation @ down.T
        calibration_logits = calibration_states @ gate.T
        validation_logits = validation_states @ gate.T
        validation_up = validation_states @ up.T

        for activation in activation_values:
            calibration_gate = _gate_values(calibration_logits, activation)
            validation_gate = _gate_values(validation_logits, activation)
            for sparsity in sparsity_values:
                cutoff = _threshold(calibration_gate, sparsity, activation)
                magnitudes = (
                    np.abs(validation_gate)
                    if activation == "cats_silu"
                    else validation_gate
                )
                active = magnitudes > cutoff
                approximation = (validation_gate * active * validation_up) @ down.T
                relative, cosine = _relative_and_cosine_rows(
                    approximation, reference_output
                )
                active_fraction = float(np.mean(active))
                accumulator = accumulators[(activation, sparsity)]
                accumulator["relative"].extend(relative.tolist())
                accumulator["cosine"].extend(cosine.tolist())
                accumulator["active"].append(active_fraction)
                accumulator["layers"].append(
                    {
                        "layer": layer,
                        "threshold": cutoff,
                        "active_fraction": active_fraction,
                        "mlp_output_relative_l2": float(np.mean(relative)),
                        "mlp_output_cosine": float(np.mean(cosine)),
                    }
                )

    arms: list[dict[str, Any]] = []
    for (activation, sparsity), accumulator in accumulators.items():
        active = _stats(accumulator["active"])
        traffic = exact_gate_sparse_traffic(
            inspection.hidden_size,
            inspection.intermediate_size,
            float(active["mean"]),
        )
        relative = _stats(accumulator["relative"])
        cosine = _stats(accumulator["cosine"])
        local_pass = float(relative["mean"]) <= maximum_mean_relative_l2
        traffic_pass = float(traffic["fraction_of_dense"]) <= maximum_traffic_fraction
        arms.append(
            {
                "name": f"{activation}_sparsity_{sparsity:.6f}",
                "activation": activation,
                "target_sparsity": sparsity,
                "activation_sparsity": {
                    **_stats(1.0 - np.asarray(accumulator["active"])),
                    "validation_active_fraction": active,
                },
                "mlp_output_relative_l2": relative,
                "mlp_output_cosine": cosine,
                "projected_traffic": traffic,
                "layers": accumulator["layers"],
                "screen": {
                    "local_quality_pass": local_pass,
                    "traffic_pass_before_metadata": traffic_pass,
                    "eligible_for_causal_intervention": local_pass and traffic_pass,
                },
            }
        )

    eligible = [
        arm["name"] for arm in arms if arm["screen"]["eligible_for_causal_intervention"]
    ]
    return {
        "experiment": "intrinsic_sparse_gate_sweep",
        "status": (
            "eligible_for_causal_intervention"
            if eligible
            else "requires_progressive_sparse_training"
        ),
        "references": list(INTRINSIC_SPARSE_REFERENCES),
        "source": {
            "model_path": str(model_path),
            "model_hash": inspection.source_hash,
            "hidden_size": inspection.hidden_size,
            "intermediate_size": inspection.intermediate_size,
            "layers": inspection.num_hidden_layers,
        },
        "calibration": {
            "trace_path": str(Path(calibration_traces).resolve()),
            "dataset_hash": calibration.manifest["dataset_hash"],
            "records_per_layer": len(_load_states(calibration, 0, calibration_records)),
        },
        "validation": {
            "trace_path": str(Path(validation_traces).resolve()),
            "dataset_hash": validation.manifest["dataset_hash"],
            "records_per_layer": len(_load_states(validation, 0, validation_records)),
            "sequence_hashes": _sequence_hashes(validation, validation_records),
        },
        "gate": {
            "maximum_mean_relative_l2": maximum_mean_relative_l2,
            "maximum_traffic_fraction": maximum_traffic_fraction,
            "candidate_recall_applicable": False,
            "artifact_required_for_formal_pass": True,
        },
        "arms": arms,
        "eligible_arms": eligible,
        "decision": (
            "run_causal_development_intervention"
            if eligible
            else "train_gate_distribution_progressively_before_causal_evaluation"
        ),
        "scope_caveat": (
            "Thresholds are fitted only on calibration traces and evaluated on "
            "disjoint validation traces. Traffic excludes metadata and cache-line "
            "padding; this trace screen is not an all-layer causal or artifact pass."
        ),
    }


def write_intrinsic_sparse_gate_report(
    report: dict[str, Any], out: str | Path
) -> tuple[Path, Path]:
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "intrinsic_sparse_gate_sweep.json"
    markdown_path = target / "intrinsic_sparse_gate_sweep.md"
    atomic_json(json_path, report)
    lines = [
        "# Exact gate-selected semantic-memory screen",
        "",
        f"Status: **{report['status']}**",
        "",
        "| Activation | Target sparsity | Actual active | Mean relative L2 | Ideal Q4 traffic | Progression |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for arm in report["arms"]:
        active = arm["activation_sparsity"]["validation_active_fraction"]["mean"]
        relative = arm["mlp_output_relative_l2"]["mean"]
        traffic = arm["projected_traffic"]["fraction_of_dense"]
        progression = (
            "eligible" if arm["screen"]["eligible_for_causal_intervention"] else "stop"
        )
        lines.append(
            f"| {arm['activation']} | {arm['target_sparsity']:.3f} | "
            f"{active:.4f} | {relative:.6f} | {traffic:.4%} | {progression} |"
        )
    lines.extend(
        [
            "",
            "The mechanism scans the complete gate projection and reads up/down",
            "records only for exact nonzero gate outputs. It has no approximate",
            "candidate stage. Reported traffic excludes metadata and cache-line",
            "padding, and the screen is not a causal or serialized-artifact pass.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


__all__ = [
    "evaluate_intrinsic_sparse_gate_sweep",
    "exact_gate_sparse_traffic",
    "write_intrinsic_sparse_gate_report",
]
