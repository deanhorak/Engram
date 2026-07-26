"""Validation-only screen for an adaptive-DIP conditional background capsule.

The experiment is intentionally narrow.  It examines one layer whose
target-1 adaptive K policy has a small, explicit clamp-failure population.  A
capsule is fit only on those failures, evaluated with sequence-disjoint parity
folds and leave-one-trigger-sequence-out cross-validation, and charged against
the same cache-line traffic model as the DIP index.

No causal confirmation or final corpus is read by this module.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from engram.evaluation.native_bitnet_adaptive_k import (
    adaptive_k_from_candidate_utility,
)
from engram.evaluation.native_bitnet_dip_traffic import (
    native_bitnet_dip_physical_accounting,
)
from engram.evaluation.native_bitnet_router import _trace_mlp_states
from engram.models.native_bitnet import (
    _activation_quant,
    decode_native_bitnet_layer,
    load_native_bitnet_artifact,
)
from engram.semantic.native_bitnet_dip_background import (
    fit_native_bitnet_conditional_background,
)
from engram.training.controller_distillation import _load_trajectories
from engram.utils import atomic_json, sha256_file


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


def _row_relative_l2(
    approximation: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    actual = np.asarray(approximation, dtype=np.float64)
    exact = np.asarray(reference, dtype=np.float64)
    return np.linalg.norm(actual - exact, axis=1) / np.maximum(
        np.linalg.norm(exact, axis=1),
        1e-12,
    )


def _layer_traffic(
    hidden_size: int,
    intermediate_size: int,
    *,
    input_count: int,
    candidate_count: int,
    top_k: int,
) -> dict[str, Any]:
    accounting = native_bitnet_dip_physical_accounting(
        hidden_size,
        intermediate_size,
        input_counts=[input_count],
        candidate_counts=[candidate_count],
        top_ks=[top_k],
    )
    return accounting["traffic"]["layers"][0]


def maximum_candidates_with_background(
    hidden_size: int,
    intermediate_size: int,
    *,
    input_count: int,
    maximum_k: int,
    background_bytes: int,
    maximum_traffic_fraction: float = 0.45,
) -> int:
    """Largest C whose worst-case K plus capsule stays inside the byte cap."""

    if (
        min(
            hidden_size,
            intermediate_size,
            input_count,
            maximum_k,
            background_bytes,
        )
        <= 0
    ):
        raise ValueError("traffic dimensions and background_bytes must be positive")
    low = maximum_k
    high = intermediate_size
    best = 0
    while low <= high:
        candidate = (low + high) // 2
        layer = _layer_traffic(
            hidden_size,
            intermediate_size,
            input_count=input_count,
            candidate_count=candidate,
            top_k=maximum_k,
        )
        fraction = (
            layer["complete_modelled_cold_bytes"] + background_bytes
        ) / layer["dense_q4_bytes"]
        if fraction <= maximum_traffic_fraction:
            best = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    return best


def maximum_k_for_candidates(
    hidden_size: int,
    intermediate_size: int,
    *,
    input_count: int,
    candidate_count: int,
    maximum_traffic_fraction: float = 0.45,
) -> int:
    """Largest worst-case K for a fixed practical candidate count."""

    low = 1
    high = candidate_count
    best = 0
    while low <= high:
        top_k = (low + high) // 2
        layer = _layer_traffic(
            hidden_size,
            intermediate_size,
            input_count=input_count,
            candidate_count=candidate_count,
            top_k=top_k,
        )
        if layer["fraction_of_dense_q4"] <= maximum_traffic_fraction:
            best = top_k
            low = top_k + 1
        else:
            high = top_k - 1
    return best


def _evaluate_cross_fit(
    dense: np.ndarray,
    selected: np.ndarray,
    target_attained: np.ndarray,
    sample_ids: np.ndarray,
    *,
    layer: int,
) -> dict[str, Any]:
    baseline = _row_relative_l2(selected, dense)
    parity_reports = []
    parity_corrected = np.empty_like(selected)
    for training_parity in (0, 1):
        training = sample_ids % 2 == training_parity
        evaluation = ~training
        capsule = fit_native_bitnet_conditional_background(
            dense[training],
            selected[training],
            target_attained=target_attained[training],
            layer=layer,
        )
        corrected = capsule.apply(
            selected[evaluation],
            target_attained=target_attained[evaluation],
        )
        parity_corrected[evaluation] = corrected
        errors = _row_relative_l2(corrected, dense[evaluation])
        reference = baseline[evaluation]
        parity_reports.append(
            {
                "training_sequence_parity": training_parity,
                "training_sequences": int(
                    np.unique(sample_ids[training]).size
                ),
                "evaluation_sequences": int(
                    np.unique(sample_ids[evaluation]).size
                ),
                "training_trigger_rows": int(
                    np.count_nonzero(~target_attained[training])
                ),
                "evaluation_trigger_rows": int(
                    np.count_nonzero(~target_attained[evaluation])
                ),
                "baseline_relative_l2": _summary(reference),
                "corrected_relative_l2": _summary(errors),
                "mean_relative_improvement": float(
                    1.0 - np.mean(errors) / max(np.mean(reference), 1e-20)
                ),
            }
        )
    parity_errors = _row_relative_l2(parity_corrected, dense)

    triggered_sequences = np.unique(sample_ids[~target_attained])
    loso_corrected = selected.copy()
    for held_out in triggered_sequences:
        training = (sample_ids != held_out)
        evaluation = sample_ids == held_out
        capsule = fit_native_bitnet_conditional_background(
            dense[training],
            selected[training],
            target_attained=target_attained[training],
            layer=layer,
        )
        loso_corrected[evaluation] = capsule.apply(
            selected[evaluation],
            target_attained=target_attained[evaluation],
        )
    loso_errors = _row_relative_l2(loso_corrected, dense)
    triggered = ~target_attained
    return {
        "baseline_relative_l2": _summary(baseline),
        "parity_folds": parity_reports,
        "parity_cross_fit_relative_l2": _summary(parity_errors),
        "parity_mean_relative_improvement": float(
            1.0 - np.mean(parity_errors) / max(np.mean(baseline), 1e-20)
        ),
        "leave_one_trigger_sequence_out": {
            "trigger_sequences": int(triggered_sequences.size),
            "relative_l2": _summary(loso_errors),
            "triggered_row_relative_l2": _summary(loso_errors[triggered]),
            "mean_relative_improvement": float(
                1.0 - np.mean(loso_errors) / max(np.mean(baseline), 1e-20)
            ),
        },
    }


def evaluate_native_bitnet_conditional_background(
    package: str | Path,
    validation_trace: str | Path,
    router_policy: str | Path,
    *,
    out: str | Path,
    joint_policy: str | Path | None = None,
    layer: int = 7,
    minimum_k: int = 346,
    maximum_k: int = 2420,
    maximum_traffic_fraction: float = 0.45,
) -> dict[str, Any]:
    """Fit and cross-validate the target-unattained residual capsule."""

    started = time.perf_counter()
    package_path = Path(package).resolve()
    trace_path = Path(validation_trace).resolve()
    policy_path = Path(router_policy).resolve()
    manifest = json.loads(
        (package_path / "manifest.json").read_text(encoding="utf-8")
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    trace = _load_trajectories(trace_path)
    artifact = load_native_bitnet_artifact(
        package_path / manifest["mlp"]["path"]
    )
    if (
        not 0 <= layer < len(artifact.layers)
        or trace.hidden_size != artifact.hidden_size
        or trace.num_stages != len(artifact.layers)
    ):
        raise ValueError("layer or validation-trace dimensions are incompatible")
    if (
        trace.manifest.get("model_hash")
        != manifest.get("source", {}).get("weight_sha256")
    ):
        raise ValueError("validation trace was captured from another teacher")
    if (
        policy.get("validation_trace", {}).get("manifest_sha256")
        != sha256_file(trace_path / "manifest.json")
    ):
        raise ValueError("router policy was fitted on another validation trace")
    candidate_counts = policy.get("aggregate", {}).get(
        "selected_candidate_counts"
    )
    input_fraction = policy.get("configuration", {}).get("input_fraction")
    if (
        not isinstance(candidate_counts, list)
        or len(candidate_counts) != len(artifact.layers)
        or not isinstance(input_fraction, (int, float))
        or not 0 < float(input_fraction) <= 1
    ):
        raise ValueError("router policy has no valid q/C configuration")

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("background evaluation requires safetensors") from exc
    with safe_open(
        package_path / manifest["transformer"]["non_mlp_path"],
        framework="pt",
        device="cpu",
    ) as handle:
        norm_weight = (
            handle.get_tensor(
                f"model.layers.{layer}.post_attention_layernorm.weight"
            )
            .float()
            .numpy()
        )
    states = _trace_mlp_states(
        trace,
        layer,
        norm_weight,
        float(manifest["model"]["rms_norm_eps"]),
    )
    decoded = decode_native_bitnet_layer(artifact, layer)
    gate_codes = np.asarray(decoded["gate_codes"], dtype=np.float32)
    up_codes = np.asarray(decoded["up_codes"], dtype=np.float32)
    down_codes = np.asarray(decoded["down_codes"], dtype=np.float32)
    gate_scale = np.float32(decoded["gate_scale"])
    up_scale = np.float32(decoded["up_scale"])
    down_scale = np.float32(decoded["down_scale"])
    gain = np.asarray(decoded["ffn_sub_norm"], dtype=np.float32)
    down_norm_squared = np.sum(
        down_codes * down_codes,
        axis=0,
        dtype=np.float32,
    )
    quantized = _activation_quant(states)
    gate = quantized @ gate_codes.T * gate_scale
    up = quantized @ up_codes.T * up_scale
    raw = np.maximum(gate, np.float32(0.0)) ** 2 * up
    inverse = np.reciprocal(
        np.sqrt(
            np.mean(raw * raw, axis=1, keepdims=True)
            + artifact.rms_norm_eps
        )
    )
    coefficients = _activation_quant(raw * inverse * gain[None, :])
    utility = coefficients * coefficients * down_norm_squared[None, :]
    dense_output = coefficients @ down_codes.T * down_scale

    input_count = min(
        artifact.hidden_size,
        max(1, int(math.ceil(float(input_fraction) * artifact.hidden_size))),
    )
    coordinate_order = np.argsort(
        -np.abs(quantized),
        axis=1,
        kind="stable",
    )
    coordinates = coordinate_order[:, :input_count]
    masked = np.zeros_like(quantized)
    np.put_along_axis(
        masked,
        coordinates,
        np.take_along_axis(quantized, coordinates, axis=1),
        axis=1,
    )
    partial_gate = masked @ gate_codes.T * gate_scale
    partial_up = masked @ up_codes.T * up_scale
    partial_raw = np.maximum(partial_gate, np.float32(0.0)) ** 2 * partial_up
    proxy_utility = (
        partial_raw
        * partial_raw
        * gain[None, :]
        * gain[None, :]
        * down_norm_squared[None, :]
    )
    candidate_order = np.argsort(
        -proxy_utility,
        axis=1,
        kind="stable",
    )

    provisional_capsule = fit_native_bitnet_conditional_background
    # A hidden-width BF16 capsule always has the same byte count; use a dummy
    # instance to reserve those bytes before choosing C.
    dummy_dense = np.zeros((1, artifact.hidden_size), dtype=np.float32)
    dummy_selected = np.ones_like(dummy_dense)
    dummy = provisional_capsule(
        dummy_dense,
        dummy_selected,
        target_attained=np.asarray([False]),
        layer=layer,
    )
    capsule_bytes = int(dummy.traffic()["worst_case_triggered_cold_bytes"])
    fitted_candidate_count = maximum_candidates_with_background(
        artifact.hidden_size,
        artifact.intermediate_size,
        input_count=input_count,
        maximum_k=maximum_k,
        background_bytes=capsule_bytes,
        maximum_traffic_fraction=maximum_traffic_fraction,
    )

    def evaluate_arm(candidate_count: int, maximum: int) -> dict[str, Any]:
        candidates = candidate_order[:, :candidate_count]
        candidate_utility = np.take_along_axis(
            utility,
            candidates,
            axis=1,
        )
        adaptive = adaptive_k_from_candidate_utility(
            candidates,
            candidate_utility,
            energy_targets=[1.0],
            minimum_k=minimum_k,
            maximum_k=min(maximum, candidate_count),
        )[1.0]
        sparse = np.zeros_like(coefficients)
        for row, selected_k in enumerate(adaptive["selected_k"]):
            selected = adaptive["sorted_indices"][row, :selected_k]
            sparse[row, selected] = coefficients[row, selected]
        selected_output = sparse @ down_codes.T * down_scale
        errors = _row_relative_l2(selected_output, dense_output)
        traffic = _layer_traffic(
            artifact.hidden_size,
            artifact.intermediate_size,
            input_count=input_count,
            candidate_count=candidate_count,
            top_k=min(maximum, candidate_count),
        )
        return {
            "candidate_count": candidate_count,
            "maximum_k": min(maximum, candidate_count),
            "mean_k": float(np.mean(adaptive["selected_k"])),
            "p95_k": float(np.percentile(adaptive["selected_k"], 95)),
            "target_attainment_rate": float(
                np.mean(adaptive["target_attained"])
            ),
            "relative_l2": _summary(errors),
            "traffic": traffic,
            "_selected_output": selected_output,
            "_target_attained": adaptive["target_attained"],
        }

    original_candidates = int(candidate_counts[layer])
    original_maximum_k = maximum_k_for_candidates(
        artifact.hidden_size,
        artifact.intermediate_size,
        input_count=input_count,
        candidate_count=original_candidates,
        maximum_traffic_fraction=maximum_traffic_fraction,
    )
    no_capsule_candidates = maximum_candidates_with_background(
        artifact.hidden_size,
        artifact.intermediate_size,
        input_count=input_count,
        maximum_k=maximum_k,
        background_bytes=1,
        maximum_traffic_fraction=maximum_traffic_fraction,
    )
    trade_arms = [
        evaluate_arm(original_candidates, original_maximum_k),
        evaluate_arm(no_capsule_candidates, maximum_k),
        evaluate_arm(fitted_candidate_count, maximum_k),
    ]
    selected_arm = trade_arms[-1]
    selected_output = selected_arm.pop("_selected_output")
    target_attained = selected_arm.pop("_target_attained")
    for arm in trade_arms[:-1]:
        arm.pop("_selected_output")
        arm.pop("_target_attained")

    cross_fit = _evaluate_cross_fit(
        dense_output,
        selected_output,
        target_attained,
        np.asarray(trace.sample_id),
        layer=layer,
    )
    capsule = fit_native_bitnet_conditional_background(
        dense_output,
        selected_output,
        target_attained=target_attained,
        layer=layer,
    )
    selected_traffic = selected_arm["traffic"]
    combined_worst_bytes = (
        selected_traffic["complete_modelled_cold_bytes"] + capsule_bytes
    )
    combined_fraction = combined_worst_bytes / selected_traffic["dense_q4_bytes"]
    parity_improvements = [
        fold["mean_relative_improvement"]
        for fold in cross_fit["parity_folds"]
    ]
    ablation_passed = (
        min(parity_improvements) > 0
        and cross_fit["leave_one_trigger_sequence_out"][
            "mean_relative_improvement"
        ]
        > 0
        and combined_fraction <= maximum_traffic_fraction
        and cross_fit["parity_cross_fit_relative_l2"]["p95"]
        <= cross_fit["baseline_relative_l2"]["p95"] + 1e-12
    )
    superseding: dict[str, Any] | None = None
    joint_dominates = False
    if joint_policy is not None:
        joint_path = Path(joint_policy).resolve()
        joint = json.loads(joint_path.read_text(encoding="utf-8"))
        layers = joint.get("layers")
        if (
            joint.get("experiment")
            != "native_bitnet_dip_joint_candidate_adaptive_k_policy"
            or joint.get("artifact_sha256") != artifact.payload_sha256
            or joint.get("validation_trace", {}).get("manifest_sha256")
            != sha256_file(trace_path / "manifest.json")
            or not isinstance(layers, list)
            or len(layers) != len(artifact.layers)
            or layers[layer].get("layer") != layer
        ):
            raise ValueError(
                "joint policy is not aligned to this package/validation trace"
            )
        joint_selected = layers[layer].get("selected_policy")
        if not isinstance(joint_selected, dict):
            raise ValueError("joint policy has no selected layer policy")
        joint_error = joint_selected["local_output_relative_l2"]
        joint_traffic = joint_selected["physical_maximum"][
            "fraction_of_dense_q4"
        ]
        joint_dominates = bool(
            joint_selected["candidate_energy_target_attainment_rate"] == 1
            and joint_error["mean"]
            <= cross_fit["parity_cross_fit_relative_l2"]["mean"]
            and joint_error["maximum"]
            <= cross_fit["parity_cross_fit_relative_l2"]["maximum"]
            and joint_traffic <= combined_fraction
        )
        superseding = {
            "path": str(joint_path),
            "sha256": sha256_file(joint_path),
            "layer_selected_policy": joint_selected,
            "capsule_trigger_rows_under_joint_policy": (
                0
                if joint_selected[
                    "candidate_energy_target_attainment_rate"
                ]
                == 1
                else None
            ),
            "dominates_capsule_ablation": joint_dominates,
            "comparison": {
                "joint_mean_relative_l2": joint_error["mean"],
                "capsule_parity_cross_fit_mean_relative_l2": (
                    cross_fit["parity_cross_fit_relative_l2"]["mean"]
                ),
                "joint_maximum_relative_l2": joint_error["maximum"],
                "capsule_parity_cross_fit_maximum_relative_l2": (
                    cross_fit["parity_cross_fit_relative_l2"]["maximum"]
                ),
                "joint_worst_case_traffic_fraction": joint_traffic,
                "capsule_worst_case_traffic_fraction": combined_fraction,
            },
        }
    selected_for_policy = ablation_passed and not joint_dominates
    result = {
        "experiment": "native_bitnet_adaptive_dip_conditional_background",
        "scope": "validation_only_layer_local_clamp_failure_correction",
        "status": (
            "positive_ablation_superseded_by_joint_policy"
            if ablation_passed and joint_dominates
            else (
                "positive_development_evidence"
                if ablation_passed
                else "rejected_development_probe"
            )
        ),
        "package": {
            "path": str(package_path),
            "artifact_sha256": artifact.payload_sha256,
        },
        "validation_trace": {
            "path": str(trace_path),
            "manifest_sha256": sha256_file(trace_path / "manifest.json"),
            "dataset_hash": trace.manifest.get("dataset_hash"),
            "records": trace.records,
            "sequences": int(np.unique(trace.sample_id).size),
            "causal_or_final_confirmation_corpus_used": False,
        },
        "router_policy": {
            "path": str(policy_path),
            "sha256": sha256_file(policy_path),
            "input_fraction": float(input_fraction),
            "original_layer_candidate_count": original_candidates,
        },
        "configuration": {
            "layer": layer,
            "energy_target": 1.0,
            "minimum_k": minimum_k,
            "requested_maximum_k": maximum_k,
            "maximum_traffic_fraction": maximum_traffic_fraction,
            "trigger": capsule.trigger,
            "fitter": (
                "BF16-rounded mean dense-minus-selected output over only "
                "target-unattained rows"
            ),
        },
        "candidate_k_trade": {
            "preserve_original_c_reduce_k": trade_arms[0],
            "no_capsule_maximum_c_at_requested_k": trade_arms[1],
            "reserve_capsule_bytes_at_requested_k": selected_arm,
        },
        "capsule": {
            "fitting_trigger_rows": capsule.fitting_trigger_count,
            "trigger_rate": float(np.mean(~target_attained)),
            "residual_l2": float(np.linalg.norm(capsule.residual)),
            "traffic": capsule.traffic(),
            "combined_worst_case_cold_bytes": combined_worst_bytes,
            "combined_worst_case_fraction_of_dense_q4": combined_fraction,
            "reads_omitted_down_records": False,
        },
        "cross_validation": cross_fit,
        "superseding_joint_policy": superseding,
        "progression_screen": {
            "both_sequence_parity_folds_improve_mean": bool(
                min(parity_improvements) > 0
            ),
            "leave_one_trigger_sequence_out_improves_mean": bool(
                cross_fit["leave_one_trigger_sequence_out"][
                    "mean_relative_improvement"
                ]
                > 0
            ),
            "parity_p95_not_worse": bool(
                cross_fit["parity_cross_fit_relative_l2"]["p95"]
                <= cross_fit["baseline_relative_l2"]["p95"] + 1e-12
            ),
            "worst_case_traffic_within_45_percent": bool(
                combined_fraction <= maximum_traffic_fraction
            ),
            "background_ablation_passed": ablation_passed,
            "joint_policy_dominates": joint_dominates,
            "selected_for_causal_or_runtime_policy": selected_for_policy,
            "passed": selected_for_policy,
        },
        "decision": (
            "test_capsule_inside_candidate_only_causal_development_path"
            if selected_for_policy
            else (
                "retain_as_ablation_do_not_wire_use_joint_policy"
                if joint_dominates
                else "do_not_implement_background_operator"
            )
        ),
        "caveat": (
            "This uses exact full-width coefficients and normalization to "
            "isolate adaptive selection and the omitted residual. The capsule "
            "has not passed causal substitution, native serialization/reload, "
            "or measured CPU latency. When a supplied joint q/C/K policy "
            "attains target 1 with lower error and traffic, this primitive is "
            "retained only as a documented ablation and must not be wired "
            "into causal or runtime code."
        ),
        "milestone_2_status": "blocked",
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(Path(out), result)
    return result


__all__ = [
    "evaluate_native_bitnet_conditional_background",
    "maximum_candidates_with_background",
    "maximum_k_for_candidates",
]
