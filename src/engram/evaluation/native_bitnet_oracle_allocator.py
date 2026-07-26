"""Robust trace-only allocation of native BitNet oracle record budgets.

The existing causal oracle establishes that additive BitNet MLP records can
preserve model quality at roughly one quarter of the full width.  This module
uses a separate controller-validation trajectory to choose *where* that fixed
record budget should be spent.  It deliberately does not run, inspect, or fit
against a causal confirmation corpus.

The allocation target is stronger than local MLP relative error alone.  For
each candidate width, the sparse MLP output is added back to the captured
post-attention residual and normalized.  The resulting one-step boundary
perturbation is a teacher-forced causal-sensitivity proxy available from the
trajectory contract.  The allocator minimizes a blend of mean, token-p95,
sequence-p95, and worst-sequence boundary error, taking the worse value across
an even/odd sequence split.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.evaluation.native_bitnet_router import _trace_mlp_states
from engram.models.native_bitnet import (
    _activation_quant,
    decode_native_bitnet_layer,
    load_native_bitnet_artifact,
)
from engram.training.controller_distillation import _load_trajectories
from engram.utils import atomic_json, sha256_file


DEFAULT_ORACLE_ALLOCATOR_FRACTIONS = (
    0.15,
    0.175,
    0.20,
    0.225,
    0.25,
    0.275,
    0.30,
    0.325,
    0.35,
    0.375,
    0.40,
)

ROBUST_RISK_WEIGHTS = {
    "token_mean": 0.45,
    "token_p95": 0.30,
    "sequence_p95": 0.15,
    "worst_sequence": 0.10,
}


def _summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary values must be a non-empty finite vector")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _row_rms_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    inverse = np.reciprocal(
        np.maximum(
            np.sqrt(np.mean(array * array, axis=1, keepdims=True)),
            np.float32(1e-12),
        )
    )
    return array * inverse


def _row_relative_l2(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    actual = np.asarray(candidate, dtype=np.float64)
    exact = np.asarray(reference, dtype=np.float64)
    return np.linalg.norm(actual - exact, axis=1) / np.maximum(
        np.linalg.norm(exact, axis=1),
        1e-12,
    )


def _row_cosine(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    actual = np.asarray(candidate, dtype=np.float64)
    exact = np.asarray(reference, dtype=np.float64)
    denominator = np.linalg.norm(actual, axis=1) * np.linalg.norm(exact, axis=1)
    cosine = np.sum(actual * exact, axis=1) / np.maximum(denominator, 1e-12)
    cosine[(denominator <= 1e-12)] = 1.0
    return np.clip(cosine, -1.0, 1.0)


def _sequence_means(values: np.ndarray, sample_ids: np.ndarray) -> np.ndarray:
    errors = np.asarray(values, dtype=np.float64)
    samples = np.asarray(sample_ids)
    if errors.ndim != 1 or samples.shape != errors.shape:
        raise ValueError("values and sample_ids must be equal-length vectors")
    unique = np.unique(samples)
    if unique.size == 0:
        raise ValueError("sample_ids must not be empty")
    return np.asarray(
        [np.mean(errors[samples == sample]) for sample in unique],
        dtype=np.float64,
    )


def robust_boundary_risk(
    errors: np.ndarray,
    sample_ids: np.ndarray,
) -> dict[str, Any]:
    """Score one arm using both token and sequence-level tail behavior."""

    values = np.asarray(errors, dtype=np.float64)
    sequence_values = _sequence_means(values, sample_ids)
    components = {
        "token_mean": float(np.mean(values)),
        "token_p95": float(np.percentile(values, 95)),
        "sequence_p95": float(np.percentile(sequence_values, 95)),
        "worst_sequence": float(np.max(sequence_values)),
    }
    score = sum(
        ROBUST_RISK_WEIGHTS[name] * components[name]
        for name in ROBUST_RISK_WEIGHTS
    )
    return {
        "score": float(score),
        "weights": dict(ROBUST_RISK_WEIGHTS),
        "components": components,
        "sequence_count": int(sequence_values.size),
    }


def allocate_layer_schedule(
    risk_matrix: np.ndarray,
    record_counts: Sequence[int],
    *,
    intermediate_size: int,
    mean_budget: float,
) -> dict[str, Any]:
    """Solve the fixed-budget multiple-choice layer allocation exactly."""

    risks = np.asarray(risk_matrix, dtype=np.float64)
    counts = tuple(int(value) for value in record_counts)
    if (
        risks.ndim != 2
        or risks.shape[1] != len(counts)
        or risks.shape[0] == 0
        or not np.all(np.isfinite(risks))
        or np.any(risks < 0)
    ):
        raise ValueError("risk_matrix must be a finite non-negative layer/arm matrix")
    if (
        not counts
        or tuple(sorted(set(counts))) != counts
        or counts[0] <= 0
        or counts[-1] > intermediate_size
    ):
        raise ValueError("record_counts must be unique, increasing, and in range")
    if not np.isfinite(mean_budget) or not 0 < mean_budget <= 1:
        raise ValueError("mean_budget must lie in (0, 1]")

    layer_count = risks.shape[0]
    available = int(math.floor(mean_budget * layer_count * intermediate_size))
    if layer_count * counts[0] > available:
        raise ValueError("minimum arm exceeds the requested mean budget")

    # State maps used records to the best objective and arm choices reaching
    # that exact count.  Dominated states with the same count are discarded.
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for layer in range(layer_count):
        updated: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (objective, choices) in states.items():
            for arm, count in enumerate(counts):
                next_used = used + count
                if next_used > available:
                    continue
                next_objective = objective + float(risks[layer, arm])
                incumbent = updated.get(next_used)
                if (
                    incumbent is None
                    or next_objective < incumbent[0] - 1e-15
                    or (
                        abs(next_objective - incumbent[0]) <= 1e-15
                        and choices + (arm,) < incumbent[1]
                    )
                ):
                    updated[next_used] = (
                        next_objective,
                        choices + (arm,),
                    )
        states = updated
    if not states:
        raise RuntimeError("no complete layer allocation satisfies the budget")

    used, (objective, selected) = min(
        states.items(),
        key=lambda item: (item[1][0], -item[0], item[1][1]),
    )
    top_ks = [counts[arm] for arm in selected]
    return {
        "arm_indices": list(selected),
        "layer_top_ks": top_ks,
        "objective": float(objective),
        "record_budget_used": int(used),
        "record_budget_available": int(available),
        "mean_active_fraction": float(np.mean(top_ks) / intermediate_size),
        "minimum_active_fraction": float(np.min(top_ks) / intermediate_size),
        "maximum_active_fraction": float(np.max(top_ks) / intermediate_size),
    }


def _analyze_layer(
    artifact,
    layer: int,
    states: np.ndarray,
    trace,
    *,
    fractions: tuple[float, ...],
) -> tuple[dict[str, Any], np.ndarray]:
    """Return JSON metrics and per-arm one-step boundary errors."""

    decoded = decode_native_bitnet_layer(artifact, layer)
    quantized_state = _activation_quant(np.asarray(states, dtype=np.float32))
    gate_codes = np.asarray(decoded["gate_codes"], dtype=np.float32)
    up_codes = np.asarray(decoded["up_codes"], dtype=np.float32)
    down_codes = np.asarray(decoded["down_codes"], dtype=np.float32)
    gate = quantized_state @ gate_codes.T * np.float32(decoded["gate_scale"])
    up = quantized_state @ up_codes.T * np.float32(decoded["up_scale"])
    raw = np.maximum(gate, np.float32(0.0)) ** 2 * up
    normalized = raw * np.reciprocal(
        np.sqrt(
            np.mean(raw * raw, axis=1, keepdims=True)
            + np.float32(artifact.rms_norm_eps)
        )
    )
    normalized *= np.asarray(decoded["ffn_sub_norm"], dtype=np.float32)[None, :]
    coefficients = _activation_quant(normalized)
    down_scale = np.float32(decoded["down_scale"])
    reference = coefficients @ down_codes.T * down_scale

    down_norm_squared = np.sum(down_codes * down_codes, axis=0, dtype=np.float32)
    utility = coefficients * coefficients * down_norm_squared[None, :]
    order = np.argsort(-utility, axis=1, kind="stable")
    ordered_utility = np.take_along_axis(utility, order, axis=1)
    cumulative_utility = np.cumsum(ordered_utility, axis=1)
    total_utility = np.sum(ordered_utility, axis=1)

    # Recover the otherwise discarded incoming-residual scale by aligning the
    # reconstructed full MLP output with the normalized semantic trace.  The
    # alignment should be almost perfectly collinear; its diagnostics are
    # retained so a bad trajectory reconstruction cannot silently fit a
    # misleading schedule.
    semantic = trace.semantic_outputs[:, layer].astype(np.float32)
    denominator = np.sum(reference * reference, axis=1)
    scale = np.sum(reference * semantic, axis=1) / np.maximum(
        denominator,
        np.float32(1e-20),
    )
    scale = np.maximum(scale, np.float32(0.0))
    aligned_reference = reference * scale[:, None]
    semantic_alignment_relative = _row_relative_l2(aligned_reference, semantic)
    semantic_alignment_cosine = _row_cosine(reference, semantic)

    post_attention = (
        trace.teacher_states[:, layer].astype(np.float32)
        + trace.episodic_outputs[:, layer].astype(np.float32)
    )
    baseline_boundary = _row_rms_normalize(post_attention + aligned_reference)
    captured_next = trace.teacher_states[:, layer + 1].astype(np.float32)
    trace_boundary_relative = _row_relative_l2(
        baseline_boundary,
        captured_next,
    )
    trace_boundary_cosine = _row_cosine(baseline_boundary, captured_next)

    width = artifact.intermediate_size
    arms: list[dict[str, Any]] = []
    boundary_errors: list[np.ndarray] = []
    for fraction in fractions:
        top_k = min(width, max(1, int(math.ceil(fraction * width))))
        selected = order[:, :top_k]
        sparse_coefficients = np.zeros_like(coefficients)
        np.put_along_axis(
            sparse_coefficients,
            selected,
            np.take_along_axis(coefficients, selected, axis=1),
            axis=1,
        )
        approximation = sparse_coefficients @ down_codes.T * down_scale
        local_relative = _row_relative_l2(approximation, reference)
        local_cosine = _row_cosine(approximation, reference)
        candidate_boundary = _row_rms_normalize(
            post_attention + approximation * scale[:, None]
        )
        boundary_relative = _row_relative_l2(
            candidate_boundary,
            baseline_boundary,
        )
        boundary_cosine = _row_cosine(candidate_boundary, baseline_boundary)
        sequence_boundary = _sequence_means(
            boundary_relative,
            trace.sample_id,
        )
        captured_energy = cumulative_utility[:, top_k - 1] / np.maximum(
            total_utility,
            np.float32(1e-20),
        )
        arms.append(
            {
                "requested_fraction": float(fraction),
                "top_k": int(top_k),
                "actual_fraction": float(top_k / width),
                "local_mlp_relative_l2": _summary(local_relative),
                "local_mlp_cosine": _summary(local_cosine),
                "independent_contribution_energy": _summary(captured_energy),
                "one_step_boundary_relative_l2": _summary(boundary_relative),
                "one_step_boundary_cosine": _summary(boundary_cosine),
                "sequence_mean_boundary_relative_l2": _summary(
                    sequence_boundary
                ),
                "all_trace_robust_risk": robust_boundary_risk(
                    boundary_relative,
                    trace.sample_id,
                ),
            }
        )
        boundary_errors.append(boundary_relative)

    report = {
        "layer": int(layer),
        "states": int(states.shape[0]),
        "sequence_count": int(np.unique(trace.sample_id).size),
        "trace_reconstruction": {
            "semantic_alignment_relative_l2": _summary(
                semantic_alignment_relative
            ),
            "semantic_alignment_cosine": _summary(semantic_alignment_cosine),
            "full_mlp_boundary_vs_captured_next_relative_l2": _summary(
                trace_boundary_relative
            ),
            "full_mlp_boundary_vs_captured_next_cosine": _summary(
                trace_boundary_cosine
            ),
            "recovered_inverse_incoming_rms": _summary(scale),
        },
        "arms": arms,
    }
    return report, np.stack(boundary_errors, axis=0)


def _split_masks(sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(sample_ids)
    if unique.size < 2:
        all_records = np.ones(sample_ids.shape, dtype=bool)
        return all_records, all_records
    first = unique[::2]
    second = unique[1::2]
    return np.isin(sample_ids, first), np.isin(sample_ids, second)


def _risk_matrix(
    error_grid: np.ndarray,
    sample_ids: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    layers, arms, records = error_grid.shape
    if mask.shape != (records,) or not np.any(mask):
        raise ValueError("risk mask must select at least one trace record")
    selected_ids = sample_ids[mask]
    result = np.empty((layers, arms), dtype=np.float64)
    for layer in range(layers):
        for arm in range(arms):
            result[layer, arm] = robust_boundary_risk(
                error_grid[layer, arm, mask],
                selected_ids,
            )["score"]
    return result


def _evaluate_allocation_risk(
    risk_matrix: np.ndarray,
    allocation: dict[str, Any],
) -> float:
    return float(
        sum(
            risk_matrix[layer, arm]
            for layer, arm in enumerate(allocation["arm_indices"])
        )
    )


def evaluate_native_bitnet_trace_oracle_schedule(
    package: str | Path,
    validation_trace: str | Path,
    *,
    out: str | Path,
    fractions: Sequence[float] = DEFAULT_ORACLE_ALLOCATOR_FRACTIONS,
    mean_budget: float = 0.25,
) -> dict[str, Any]:
    """Propose a robust 30-layer oracle schedule from validation traces only."""

    package_path = Path(package).resolve()
    trace_path = Path(validation_trace).resolve()
    requested = tuple(dict.fromkeys(float(value) for value in fractions))
    if not requested or any(
        not np.isfinite(value) or not 0 < value <= 1 for value in requested
    ):
        raise ValueError("fractions must be unique finite values in (0, 1]")
    if tuple(sorted(requested)) != requested:
        raise ValueError("fractions must be increasing")
    if (
        not np.isfinite(mean_budget)
        or mean_budget < requested[0]
        or mean_budget > requested[-1]
    ):
        raise ValueError("mean_budget must lie inside the fraction sweep")

    manifest_path = package_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = package_path / manifest["mlp"]["path"]
    artifact = load_native_bitnet_artifact(artifact_path)
    trace = _load_trajectories(trace_path)
    layer_count = len(artifact.layers)
    if (
        trace.records != 256
        or np.unique(trace.sample_id).size != 16
        or trace.hidden_size != artifact.hidden_size
        or trace.num_stages != layer_count
    ):
        raise ValueError(
            "allocator requires the separate 16-sequence/256-position "
            "controller validation trace with matching model dimensions"
        )
    expected_model_hash = manifest.get("source", {}).get("weight_sha256")
    if expected_model_hash and trace.manifest.get("model_hash") != expected_model_hash:
        raise ValueError("validation trace was captured from a different teacher")

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("trace oracle allocation requires safetensors") from exc
    weights_path = package_path / manifest["transformer"]["non_mlp_path"]
    norm_weights = []
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for layer in range(layer_count):
            norm_weights.append(
                handle.get_tensor(
                    f"model.layers.{layer}.post_attention_layernorm.weight"
                )
                .float()
                .numpy()
            )

    started = time.perf_counter()
    layer_reports = []
    error_layers = []
    epsilon = float(manifest["model"]["rms_norm_eps"])
    for layer in range(layer_count):
        states = _trace_mlp_states(
            trace,
            layer,
            norm_weights[layer],
            epsilon,
        )
        report, errors = _analyze_layer(
            artifact,
            layer,
            states,
            trace,
            fractions=requested,
        )
        layer_reports.append(report)
        error_layers.append(errors)

    error_grid = np.stack(error_layers, axis=0)
    record_counts = tuple(
        min(
            artifact.intermediate_size,
            max(1, int(math.ceil(fraction * artifact.intermediate_size))),
        )
        for fraction in requested
    )
    even_mask, odd_mask = _split_masks(trace.sample_id)
    all_mask = np.ones(trace.records, dtype=bool)
    all_risks = _risk_matrix(error_grid, trace.sample_id, all_mask)
    even_risks = _risk_matrix(error_grid, trace.sample_id, even_mask)
    odd_risks = _risk_matrix(error_grid, trace.sample_id, odd_mask)
    worst_split_risks = np.maximum(even_risks, odd_risks)

    robust = allocate_layer_schedule(
        worst_split_risks,
        record_counts,
        intermediate_size=artifact.intermediate_size,
        mean_budget=mean_budget,
    )
    all_data = allocate_layer_schedule(
        all_risks,
        record_counts,
        intermediate_size=artifact.intermediate_size,
        mean_budget=mean_budget,
    )
    fit_even = allocate_layer_schedule(
        even_risks,
        record_counts,
        intermediate_size=artifact.intermediate_size,
        mean_budget=mean_budget,
    )
    fit_odd = allocate_layer_schedule(
        odd_risks,
        record_counts,
        intermediate_size=artifact.intermediate_size,
        mean_budget=mean_budget,
    )

    for allocation in (robust, all_data, fit_even, fit_odd):
        allocation["layer_requested_fractions"] = [
            requested[arm] for arm in allocation["arm_indices"]
        ]
    robust["all_trace_objective"] = _evaluate_allocation_risk(all_risks, robust)
    robust["even_split_objective"] = _evaluate_allocation_risk(even_risks, robust)
    robust["odd_split_objective"] = _evaluate_allocation_risk(odd_risks, robust)
    fit_even["held_out_odd_objective"] = _evaluate_allocation_risk(
        odd_risks,
        fit_even,
    )
    fit_odd["held_out_even_objective"] = _evaluate_allocation_risk(
        even_risks,
        fit_odd,
    )

    robust_selected = [
        layer_reports[layer]["arms"][arm]
        for layer, arm in enumerate(robust["arm_indices"])
    ]
    uniform_arm = int(np.argmin(np.abs(np.asarray(requested) - mean_budget)))
    uniform_top_k = record_counts[uniform_arm]
    uniform_risk = float(np.sum(worst_split_risks[:, uniform_arm]))
    robust_risk = float(
        sum(
            worst_split_risks[layer, arm]
            for layer, arm in enumerate(robust["arm_indices"])
        )
    )
    robust["selected_metrics"] = {
        "macro_mean_one_step_boundary_relative_l2": float(
            np.mean(
                [
                    arm["one_step_boundary_relative_l2"]["mean"]
                    for arm in robust_selected
                ]
            )
        ),
        "worst_layer_p95_one_step_boundary_relative_l2": float(
            np.max(
                [
                    arm["one_step_boundary_relative_l2"]["p95"]
                    for arm in robust_selected
                ]
            )
        ),
        "worst_layer_maximum_one_step_boundary_relative_l2": float(
            np.max(
                [
                    arm["one_step_boundary_relative_l2"]["maximum"]
                    for arm in robust_selected
                ]
            )
        ),
        "macro_mean_local_mlp_relative_l2": float(
            np.mean(
                [
                    arm["local_mlp_relative_l2"]["mean"]
                    for arm in robust_selected
                ]
            )
        ),
    }

    result = {
        "experiment": "native_bitnet_validation_trace_robust_oracle_allocator",
        "status": "development_schedule_proposal",
        "scope": (
            "all 30 MLP layers; exact-membership oracle; separate controller "
            "validation trajectory only"
        ),
        "package": {
            "path": str(package_path),
            "manifest_sha256": sha256_file(manifest_path),
            "artifact_sha256": artifact.payload_sha256,
            "model_hash": expected_model_hash,
        },
        "validation_trace": {
            "path": str(trace_path),
            "manifest_sha256": sha256_file(trace_path / "manifest.json"),
            "model_hash": trace.manifest.get("model_hash"),
            "dataset_hash": trace.manifest.get("dataset_hash"),
            "records": trace.records,
            "sequences": int(np.unique(trace.sample_id).size),
            "explicitly_excluded": (
                "causal confirmation datasets and offset-8 confirmation result"
            ),
        },
        "configuration": {
            "fractions": list(requested),
            "record_counts": list(record_counts),
            "mean_budget": float(mean_budget),
            "intermediate_size": artifact.intermediate_size,
            "stable_oracle_tie_break": (
                "descending_contribution_utility_then_ascending_source_index"
            ),
            "risk": {
                "metric": (
                    "teacher-forced one-step normalized residual-boundary "
                    "relative L2"
                ),
                "weights": dict(ROBUST_RISK_WEIGHTS),
                "robust_fit": (
                    "minimize the worse per-arm risk across disjoint even/odd "
                    "sequence splits"
                ),
            },
        },
        "layers": layer_reports,
        "allocation": robust,
        "comparisons": {
            "uniform_nearest_budget": {
                "requested_fraction": requested[uniform_arm],
                "top_k": uniform_top_k,
                "worst_split_objective": uniform_risk,
            },
            "robust_objective_improvement_over_uniform": (
                (uniform_risk - robust_risk) / max(uniform_risk, 1e-20)
            ),
            "all_trace_fit": all_data,
            "cross_fit": {
                "fit_even_evaluate_odd": fit_even,
                "fit_odd_evaluate_even": fit_odd,
                "identical_layer_choices": int(
                    np.sum(
                        np.asarray(fit_even["arm_indices"])
                        == np.asarray(fit_odd["arm_indices"])
                    )
                ),
                "mean_absolute_fraction_difference": float(
                    np.mean(
                        np.abs(
                            np.asarray(
                                fit_even["layer_requested_fractions"],
                                dtype=np.float64,
                            )
                            - np.asarray(
                                fit_odd["layer_requested_fractions"],
                                dtype=np.float64,
                            )
                        )
                    )
                ),
            },
        },
        "causal_sensitivity": {
            "available": True,
            "kind": "teacher_forced_one_step_boundary_perturbation",
            "construction": (
                "reconstruct post-attention RMSNorm input from normalized "
                "teacher state plus attention output, align the exact MLP "
                "output to the normalized semantic trace, replace it with the "
                "oracle-sparse output, add the residual, and compare normalized "
                "next boundaries"
            ),
            "limitation": (
                "the trace does not contain perturbed downstream rollouts or "
                "layer Jacobians, so this is not a full causal confirmation"
            ),
        },
        "decision": (
            "run one untouched causal confirmation of this frozen schedule; "
            "do not treat this fitted validation result as a gate pass"
        ),
        "milestone_2_status": "blocked",
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(Path(out), result)
    return result


__all__ = [
    "DEFAULT_ORACLE_ALLOCATOR_FRACTIONS",
    "ROBUST_RISK_WEIGHTS",
    "allocate_layer_schedule",
    "evaluate_native_bitnet_trace_oracle_schedule",
    "robust_boundary_risk",
]
