"""Held-out screening for state-selected MLP residual correction capsules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.corrections import fit_correction_capsules
from engram.evaluation.router_sweep import (
    _cache_metadata,
    _load_or_build_membership,
    _load_states,
    _sequence_hashes,
    _stats,
)
from engram.models.inspection import inspect_model, load_layer_mlp, resolve_model_path
from engram.semantic.multilabel_router import LowRankMultiLabelRouter
from engram.semantic.swiglu import neuron_activations
from engram.tracing.format import TraceReader


def _routed_outputs(
    states: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    router: LowRankMultiLabelRouter,
    *,
    top_k: int,
    candidate_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    activations = neuron_activations(states, gate, up)
    full = activations @ down.T
    router_scores = (states @ router.input_factors) @ router.output_factors + router.bias
    candidates = np.argsort(-router_scores, axis=1, kind="stable")[:, :candidate_count]
    exact_scores = np.abs(activations) * np.linalg.norm(down, axis=0)[None, :]
    candidate_scores = np.take_along_axis(exact_scores, candidates, axis=1)
    local_order = np.argsort(-candidate_scores, axis=1, kind="stable")[:, :top_k]
    active_ids = np.take_along_axis(candidates, local_order, axis=1)
    active_values = np.take_along_axis(activations, active_ids, axis=1)
    masked = np.zeros_like(activations)
    np.put_along_axis(masked, active_ids, active_values, axis=1)
    sparse = masked @ down.T
    return full, sparse, candidates


def _relative_rows(approximation: np.ndarray, reference: np.ndarray) -> np.ndarray:
    error = np.linalg.norm(approximation - reference, axis=1)
    norm = np.linalg.norm(reference, axis=1)
    return error / np.maximum(norm, 1e-12)


def evaluate_correction_capsule_sweep(
    model: str | Path,
    calibration_traces: str | Path,
    validation_traces: str | Path,
    *,
    membership_cache: str | Path,
    router_rank: int = 16,
    router_regularization: float = 8000.0,
    top_k: int = 768,
    candidate_count: int = 1280,
    capsule_counts: Sequence[int] = (1, 4, 8),
    capsule_ranks: Sequence[int] = (8, 16),
    capsule_ridge: float = 1000.0,
    capsule_iterations: int = 8,
    radius_scale: float = 1.25,
    priority_fractions: Sequence[float] = (1.0,),
    radius_quantile: float = 1.0,
    calibration_records: int | None = None,
    validation_records: int | None = None,
) -> dict[str, Any]:
    """Fit local low-rank predictors of the residual left by a routed MLP read."""

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    calibration = TraceReader(calibration_traces)
    validation = TraceReader(validation_traces)
    for trace, split in ((calibration, "calibration"), (validation, "validation")):
        if trace.manifest["model_hash"] != inspection.source_hash:
            raise ValueError("trace/model hash mismatch")
        if trace.manifest["split"] != split:
            raise ValueError(f"expected {split!r} trace split")
    calibration_sequences = _sequence_hashes(calibration)
    validation_sequences = _sequence_hashes(validation)
    if set(calibration_sequences).intersection(validation_sequences):
        raise ValueError("calibration and validation traces contain matching token sequences")
    counts = tuple(dict.fromkeys(int(value) for value in capsule_counts))
    ranks = tuple(dict.fromkeys(int(value) for value in capsule_ranks))
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("capsule_counts must be positive")
    if not ranks or any(value <= 0 or value > inspection.hidden_size for value in ranks):
        raise ValueError("capsule_ranks must lie within the hidden size")
    fractions = tuple(dict.fromkeys(float(value) for value in priority_fractions))
    if not fractions or any(not np.isfinite(value) or not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("priority_fractions must lie in (0, 1]")
    if not 0 < top_k <= candidate_count <= inspection.intermediate_size:
        raise ValueError("require 0 < top_k <= candidate_count <= intermediate size")

    configurations = [
        (count, rank, fraction)
        for fraction in fractions
        for count in counts
        for rank in ranks
    ]
    accumulated: dict[tuple[int, int, float], dict[str, list[float]]] = {
        configuration: {
            "corrected_relative_l2": [],
            "residual_prediction_relative_l2": [],
            "hard_subset_corrected_relative_l2": [],
            "matched": [],
        }
        for configuration in configurations
    }
    baseline_values: list[float] = []
    per_layer: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    calibration_count = 0
    validation_count = 0
    for layer in range(inspection.num_hidden_layers):
        gate, up, down = (
            np.asarray(value, dtype=np.float64) for value in load_layer_mlp(model_path, layer)
        )
        fit_states = _load_states(calibration, layer, calibration_records)
        held_out_states = _load_states(validation, layer, validation_records)
        calibration_count = len(fit_states)
        validation_count = len(held_out_states)
        metadata = _cache_metadata(
            source_hash=inspection.source_hash,
            calibration_hash=calibration.manifest["dataset_hash"],
            validation_hash=validation.manifest["dataset_hash"],
            layer=layer,
            top_k=top_k,
            intermediate_size=inspection.intermediate_size,
            calibration_records=calibration_records,
            validation_records=validation_records,
        )
        fit_membership, held_out_membership, cache_hit = _load_or_build_membership(
            Path(membership_cache), metadata, fit_states, held_out_states, gate, up, down
        )
        cache_hits += int(cache_hit)
        cache_misses += int(not cache_hit)
        router = LowRankMultiLabelRouter.fit(
            fit_states,
            fit_membership.astype(np.float64),
            rank=router_rank,
            regularization=router_regularization,
        )
        fit_full, fit_sparse, _ = _routed_outputs(
            fit_states,
            gate,
            up,
            down,
            router,
            top_k=top_k,
            candidate_count=candidate_count,
        )
        held_out_full, held_out_sparse, held_out_candidates = _routed_outputs(
            held_out_states,
            gate,
            up,
            down,
            router,
            top_k=top_k,
            candidate_count=candidate_count,
        )
        baseline = _relative_rows(held_out_sparse, held_out_full)
        baseline_values.extend(baseline.tolist())
        hard_threshold = float(np.quantile(baseline, 0.8))
        hard = baseline >= hard_threshold
        candidate_hits = np.sum(
            np.take_along_axis(held_out_membership, held_out_candidates, axis=1), axis=1
        ) / float(top_k)
        layer_result: dict[str, Any] = {
            "layer": layer,
            "baseline_relative_l2": _stats(baseline),
            "candidate_recall": _stats(candidate_hits),
            "hard_threshold": hard_threshold,
            "configurations": [],
        }
        fit_residual = fit_full - fit_sparse
        held_out_residual = held_out_full - held_out_sparse
        for capsule_count, capsule_rank, priority_fraction in configurations:
            fitted = fit_correction_capsules(
                fit_states,
                fit_residual,
                capsules=capsule_count,
                rank=capsule_rank,
                ridge=capsule_ridge,
                iterations=capsule_iterations,
                radius_scale=radius_scale,
                priority_fraction=priority_fraction,
                radius_quantile=radius_quantile,
            )
            prediction, matched = fitted.predict(held_out_states)
            corrected = held_out_sparse + prediction
            corrected_error = _relative_rows(corrected, held_out_full)
            prediction_error = _relative_rows(prediction, held_out_residual)
            hard_error = corrected_error[hard]
            metrics = accumulated[(capsule_count, capsule_rank, priority_fraction)]
            metrics["corrected_relative_l2"].extend(corrected_error.tolist())
            metrics["residual_prediction_relative_l2"].extend(prediction_error.tolist())
            metrics["hard_subset_corrected_relative_l2"].extend(hard_error.tolist())
            metrics["matched"].extend(matched.astype(np.float64).tolist())
            layer_result["configurations"].append(
                {
                    "capsules": capsule_count,
                    "rank": capsule_rank,
                    "priority_fraction": priority_fraction,
                    "corrected_relative_l2": _stats(corrected_error),
                    "residual_prediction_relative_l2": _stats(prediction_error),
                    "hard_subset_corrected_relative_l2": _stats(hard_error),
                    "match_fraction": float(np.mean(matched)),
                    "parameter_bytes_float32": fitted.parameter_bytes(),
                }
            )
        per_layer.append(layer_result)

    baseline_summary = _stats(baseline_values)
    arms: list[dict[str, Any]] = []
    for capsule_count, capsule_rank, priority_fraction in configurations:
        metrics = accumulated[(capsule_count, capsule_rank, priority_fraction)]
        corrected = _stats(metrics["corrected_relative_l2"])
        parameter_bytes_per_layer = (
            capsule_count
            * (2 * inspection.hidden_size * capsule_rank + 2 * inspection.hidden_size)
            * 4
        )
        match_fraction = float(np.mean(metrics["matched"]))
        selection_bytes = capsule_count * inspection.hidden_size * 4
        selected_factor_bytes = (
            2 * inspection.hidden_size * capsule_rank + inspection.hidden_size
        ) * 4
        improvement = 1.0 - corrected["mean"] / baseline_summary["mean"]
        arms.append(
            {
                "capsules": capsule_count,
                "rank": capsule_rank,
                "priority_fraction": priority_fraction,
                "corrected_relative_l2": corrected,
                "residual_prediction_relative_l2": _stats(
                    metrics["residual_prediction_relative_l2"]
                ),
                "hard_subset_corrected_relative_l2": _stats(
                    metrics["hard_subset_corrected_relative_l2"]
                ),
                "match_fraction": match_fraction,
                "relative_l2_improvement": float(improvement),
                "parameter_bytes_float32_per_layer": parameter_bytes_per_layer,
                "logical_correction_bytes_per_token_all_layers": (
                    int(selection_bytes + match_fraction * selected_factor_bytes)
                    * inspection.num_hidden_layers
                ),
                "correction_macs_per_token_all_layers": (
                    (2 * inspection.hidden_size * capsule_rank
                    + capsule_count * inspection.hidden_size)
                    * inspection.num_hidden_layers
                ),
            }
        )
    best = min(arms, key=lambda arm: arm["corrected_relative_l2"]["mean"])
    viable = (
        best["relative_l2_improvement"] >= 0.25
        and best["corrected_relative_l2"]["mean"] <= 0.15
        and best["match_fraction"] >= 0.05
    )
    return {
        "schema_version": 1,
        "experiment": "correction_capsule_residual_sweep",
        "status": "measured_local_model",
        "source_model_hash": inspection.source_hash,
        "calibration": {
            "trace_path": str(Path(calibration_traces).resolve()),
            "dataset_hash": calibration.manifest["dataset_hash"],
            "records_per_layer": calibration_count,
            "sequence_count": len(calibration_sequences),
        },
        "validation": {
            "trace_path": str(Path(validation_traces).resolve()),
            "dataset_hash": validation.manifest["dataset_hash"],
            "records_per_layer": validation_count,
            "sequence_count": len(validation_sequences),
        },
        "data_separation": {
            "method": "exact_token_sequence_hashes",
            "overlapping_sequence_count": 0,
            "held_out": True,
        },
        "router": {
            "rank": router_rank,
            "regularization": router_regularization,
            "top_k": top_k,
            "candidate_count": candidate_count,
        },
        "capsule_fit": {
            "ridge": capsule_ridge,
            "iterations": capsule_iterations,
            "radius_scale": radius_scale,
            "radius_quantile": radius_quantile,
            "priority_fractions": list(fractions),
            "priority": "largest_residual_seed_then_residual_weighted_farthest_point",
        },
        "membership_cache": {
            "path": str(Path(membership_cache).resolve()),
            "hits": cache_hits,
            "misses": cache_misses,
        },
        "baseline_relative_l2": baseline_summary,
        "arms": arms,
        "best_arm": best,
        "per_layer": per_layer,
        "screening_thresholds": {
            "minimum_relative_l2_improvement": 0.25,
            "maximum_corrected_relative_l2": 0.15,
            "minimum_match_fraction": 0.05,
        },
        "screening_decision": (
            "eligible_for_causal_intervention" if viable else "reject_before_causal_intervention"
        ),
        "scope_caveat": (
            "This trace-only local MLP screen does not measure accumulated hidden-state drift, "
            "logit KL, NLL, or realized hardware traffic."
        ),
    }


__all__ = ["evaluate_correction_capsule_sweep"]
