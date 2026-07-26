"""Validation-only adaptive-K experiment for practical native BitNet DIP.

Candidate membership is fixed by the practical q=0.75 DIP selector and the
accepted robust per-layer candidate schedule. Within those candidates this
experiment uses exact coefficient-times-down-column utility, selects the
smallest K reaching a requested cumulative-energy target, clamps K to explicit
minimum/maximum bounds, and measures local MLP-output reconstruction.

Exact candidate coefficients make this a semantic-selection ceiling. A
deployable kernel must reproduce their normalization scale from candidate-only
statistics and prove the same result causally.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.evaluation.native_bitnet_dip_traffic import (
    native_bitnet_dip_physical_accounting,
)
from engram.evaluation.native_bitnet_router import _trace_mlp_states
from engram.models.native_bitnet import (
    decode_native_bitnet_layer,
    load_native_bitnet_artifact,
)
from engram.semantic.native_bitnet_dip import (
    _activation_quant_bf16,
    _bf16_round,
    _native_raw_activation,
)
from engram.training.controller_distillation import _load_trajectories
from engram.utils import atomic_json, sha256_file


DEFAULT_ADAPTIVE_K_ENERGY_TARGETS = (
    0.90,
    0.95,
    0.975,
    0.99,
    0.995,
    0.999,
)

DEFAULT_JOINT_CANDIDATE_COUNTS = (
    3200,
    3456,
    3712,
    3968,
    4224,
    4480,
    4736,
    4992,
    5248,
    5504,
)


def _summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("cannot summarize empty or non-finite values")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def adaptive_k_from_candidate_utility(
    candidate_indices: np.ndarray,
    candidate_utility: np.ndarray,
    *,
    energy_targets: Sequence[float],
    minimum_k: int,
    maximum_k: int,
) -> dict[float, dict[str, np.ndarray]]:
    """Return deterministic exact-energy selections for multiple targets."""

    indices = np.asarray(candidate_indices)
    utility = np.asarray(candidate_utility, dtype=np.float64)
    if (
        indices.ndim != 2
        or utility.shape != indices.shape
        or indices.size == 0
        or not np.issubdtype(indices.dtype, np.integer)
        or not np.all(np.isfinite(utility))
        or np.any(utility < 0)
    ):
        raise ValueError("candidate indices/utility must be aligned finite matrices")
    candidate_count = indices.shape[1]
    if (
        isinstance(minimum_k, bool)
        or isinstance(maximum_k, bool)
        or not isinstance(minimum_k, int)
        or not isinstance(maximum_k, int)
        or not 0 < minimum_k <= maximum_k <= candidate_count
    ):
        raise ValueError("K clamps must satisfy 0 < minimum_k <= maximum_k <= C")
    targets = tuple(dict.fromkeys(float(value) for value in energy_targets))
    if not targets or any(
        not np.isfinite(value) or not 0 < value <= 1 for value in targets
    ):
        raise ValueError("energy targets must lie in (0, 1]")

    # lexsort's last key is primary: descending utility, then ascending
    # source-record id provides a stable contract for exact-zero ties.
    order = np.lexsort((indices, -utility), axis=1)
    sorted_indices = np.take_along_axis(indices, order, axis=1)
    sorted_utility = np.take_along_axis(utility, order, axis=1)
    cumulative = np.cumsum(sorted_utility, axis=1)
    total = cumulative[:, -1]
    result: dict[float, dict[str, np.ndarray]] = {}
    for target in targets:
        threshold = target * total
        required = np.sum(cumulative < threshold[:, None], axis=1) + 1
        required[total <= 1e-30] = minimum_k
        selected_k = np.clip(required, minimum_k, maximum_k).astype(
            np.int64,
            copy=False,
        )
        captured = cumulative[
            np.arange(len(cumulative)),
            selected_k - 1,
        ] / np.maximum(total, 1e-30)
        captured[total <= 1e-30] = 1.0
        result[target] = {
            "selected_k": selected_k,
            "sorted_indices": sorted_indices,
            "captured_candidate_energy": captured,
            "target_attained": captured + 1e-12 >= target,
        }
    return result


def maximum_k_under_physical_limit(
    hidden_size: int,
    intermediate_size: int,
    *,
    input_count: int,
    candidate_count: int,
    maximum_traffic_fraction: float = 0.45,
) -> dict[str, int | float]:
    """Resolve the largest K for fixed q/C under the physical byte limit."""

    if not np.isfinite(maximum_traffic_fraction) or not (
        0 < maximum_traffic_fraction <= 1
    ):
        raise ValueError("maximum_traffic_fraction must lie in (0, 1]")
    base = native_bitnet_dip_physical_accounting(
        hidden_size,
        intermediate_size,
        input_counts=(input_count,),
        candidate_counts=(candidate_count,),
        top_ks=(1,),
    )["traffic"]["layers"][0]
    record_bytes = math.ceil(hidden_size / 5)
    fixed_bytes = int(base["complete_modelled_cold_bytes"]) - record_bytes
    budget_bytes = math.floor(
        maximum_traffic_fraction * int(base["dense_q4_bytes"])
    )
    maximum_k = min(
        candidate_count,
        max(1, (budget_bytes - fixed_bytes) // record_bytes),
    )
    complete = fixed_bytes + maximum_k * record_bytes
    return {
        "maximum_k": maximum_k,
        "complete_modelled_cold_bytes": complete,
        "dense_q4_bytes": int(base["dense_q4_bytes"]),
        "fraction_of_dense_q4": complete / int(base["dense_q4_bytes"]),
        "audit_reserve_bytes": budget_bytes - complete,
    }


def _load_router_policy(
    path: str | Path,
    *,
    layer_count: int,
    intermediate_size: int,
    validation_manifest_sha256: str,
) -> dict[str, Any]:
    policy_path = Path(path).resolve()
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("router policy is not readable JSON") from exc
    if (
        payload.get("experiment")
        != "native_bitnet_dip_all_layer_recall_sweep"
        or payload.get("progression_screen", {}).get("passed") is not True
        or payload.get("validation_trace", {}).get("manifest_sha256")
        != validation_manifest_sha256
    ):
        raise ValueError("router policy is not a passing policy for this trace")
    aggregate = payload.get("aggregate", {})
    candidates = aggregate.get("selected_candidate_counts")
    top_ks = payload.get("oracle_schedule", {}).get("layer_top_ks")
    if (
        not isinstance(candidates, list)
        or not isinstance(top_ks, list)
        or len(candidates) != layer_count
        or len(top_ks) != layer_count
        or any(
            not isinstance(value, int) or not 0 < value <= intermediate_size
            for value in candidates + top_ks
        )
    ):
        raise ValueError("router policy schedules differ from the package")
    return {
        "path": str(policy_path),
        "sha256": sha256_file(policy_path),
        "input_fraction": float(payload["configuration"]["input_fraction"]),
        "candidate_counts": candidates,
        "fixed_oracle_top_ks": top_ks,
    }


def _row_output_metrics(
    approximation: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(approximation, dtype=np.float64)
    exact = np.asarray(reference, dtype=np.float64)
    exact_norm = np.linalg.norm(exact, axis=1)
    actual_norm = np.linalg.norm(actual, axis=1)
    relative = np.linalg.norm(actual - exact, axis=1) / np.maximum(
        exact_norm,
        1e-12,
    )
    cosine = np.sum(actual * exact, axis=1) / np.maximum(
        actual_norm * exact_norm,
        1e-12,
    )
    cosine[(actual_norm <= 1e-12) & (exact_norm <= 1e-12)] = 1.0
    return relative, np.clip(cosine, -1.0, 1.0)


def evaluate_native_bitnet_dip_adaptive_k(
    package: str | Path,
    validation_trace: str | Path,
    router_policy: str | Path,
    *,
    out: str | Path,
    energy_targets: Sequence[float] = DEFAULT_ADAPTIVE_K_ENERGY_TARGETS,
    minimum_fraction: float = 0.05,
    maximum_fraction: float = 0.425,
    mean_budget_fraction: float = 0.25,
    device: str = "cuda",
) -> dict[str, Any]:
    """Sweep token-adaptive exact candidate-energy K on validation states."""

    fractions = (minimum_fraction, maximum_fraction, mean_budget_fraction)
    if any(
        not np.isfinite(value) or not 0 < value <= 1 for value in fractions
    ) or minimum_fraction > maximum_fraction:
        raise ValueError("adaptive-K fractions must be ordered within (0, 1]")
    package_path = Path(package).resolve()
    trace_path = Path(validation_trace).resolve()
    trace = _load_trajectories(trace_path)
    trace_manifest_sha = sha256_file(trace_path / "manifest.json")
    manifest = json.loads(
        (package_path / "manifest.json").read_text(encoding="utf-8")
    )
    artifact = load_native_bitnet_artifact(
        package_path / manifest["mlp"]["path"]
    )
    layer_count = len(artifact.layers)
    width = artifact.intermediate_size
    hidden = artifact.hidden_size
    if trace.hidden_size != hidden or trace.num_stages != layer_count:
        raise ValueError("validation trace dimensions differ from the package")
    if (
        trace.manifest.get("model_hash")
        != manifest.get("source", {}).get("weight_sha256")
    ):
        raise ValueError("validation trace was captured from another teacher")
    policy = _load_router_policy(
        router_policy,
        layer_count=layer_count,
        intermediate_size=width,
        validation_manifest_sha256=trace_manifest_sha,
    )
    targets = tuple(dict.fromkeys(float(value) for value in energy_targets))
    minimum_k = min(width, max(1, int(math.ceil(minimum_fraction * width))))
    maximum_k = min(width, max(minimum_k, int(math.ceil(maximum_fraction * width))))

    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("adaptive-K evaluation requires torch and safetensors") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA adaptive-K evaluation requested but unavailable")

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
    epsilon = float(manifest["model"]["rms_norm_eps"])
    per_target: dict[float, dict[str, list[np.ndarray] | list[dict[str, Any]]]] = {
        target: {
            "selected_k": [],
            "captured_energy": [],
            "target_attained": [],
            "relative_l2": [],
            "cosine": [],
            "layers": [],
        }
        for target in targets
    }
    started = time.perf_counter()

    for layer, candidate_count in enumerate(policy["candidate_counts"]):
        states = _trace_mlp_states(
            trace,
            layer,
            norm_weights[layer],
            epsilon,
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
        # The policy is consumed by the direct BF16 CPU kernel.  Fit its
        # membership and zero/nonzero counts at the same stored-activation
        # boundaries instead of relying on the looser float32 research path.
        quantized = _activation_quant_bf16(_bf16_round(states))
        exact_gate = _bf16_round(
            _bf16_round(quantized @ gate_codes.T) * gate_scale
        )
        exact_up = _bf16_round(
            _bf16_round(quantized @ up_codes.T) * up_scale
        )
        raw = _native_raw_activation(exact_gate, exact_up)
        inverse = np.reciprocal(
            np.sqrt(
                np.mean(raw * raw, axis=1, keepdims=True)
                + artifact.rms_norm_eps
            )
        )
        normalized = _bf16_round(
            _bf16_round(raw * inverse) * gain[None, :]
        )
        coefficients = _activation_quant_bf16(normalized)
        exact_utility = (
            coefficients * coefficients * down_norm_squared[None, :]
        )

        input_count = min(
            hidden,
            max(1, int(math.ceil(policy["input_fraction"] * hidden))),
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
        partial_gate = _bf16_round(
            _bf16_round(masked @ gate_codes.T) * gate_scale
        )
        partial_up = _bf16_round(
            _bf16_round(masked @ up_codes.T) * up_scale
        )
        partial_raw = _native_raw_activation(partial_gate, partial_up)
        proxy = (
            partial_raw
            * partial_raw
            * gain[None, :]
            * gain[None, :]
            * down_norm_squared[None, :]
        )
        candidates = np.argsort(-proxy, axis=1, kind="stable")[
            :, :candidate_count
        ]
        candidate_utility = np.take_along_axis(
            exact_utility,
            candidates,
            axis=1,
        )
        adaptive = adaptive_k_from_candidate_utility(
            candidates,
            candidate_utility,
            energy_targets=targets,
            minimum_k=minimum_k,
            maximum_k=min(maximum_k, candidate_count),
        )

        coefficient_batches = []
        for target in targets:
            arm = adaptive[target]
            sparse = np.zeros_like(coefficients)
            for row, selected_k in enumerate(arm["selected_k"]):
                selected = arm["sorted_indices"][row, :selected_k]
                sparse[row, selected] = coefficients[row, selected]
            coefficient_batches.append(sparse)
        stacked = np.concatenate(coefficient_batches, axis=0)
        coefficients_tensor = torch.from_numpy(stacked).to(device)
        down_tensor = torch.from_numpy(down_codes.T.copy()).to(device)
        reference_tensor = torch.from_numpy(coefficients).to(device)
        with torch.inference_mode():
            outputs = (
                coefficients_tensor @ down_tensor * down_scale
            ).float().cpu().numpy()
            reference = (
                reference_tensor @ down_tensor * down_scale
            ).float().cpu().numpy()
        del coefficients_tensor, down_tensor, reference_tensor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        row_count = len(states)
        for target_index, target in enumerate(targets):
            arm = adaptive[target]
            approximation = outputs[
                target_index * row_count : (target_index + 1) * row_count
            ]
            relative, cosine = _row_output_metrics(approximation, reference)
            selected_k = arm["selected_k"]
            target_report = per_target[target]
            target_report["selected_k"].append(selected_k)
            target_report["captured_energy"].append(
                arm["captured_candidate_energy"]
            )
            target_report["target_attained"].append(arm["target_attained"])
            target_report["relative_l2"].append(relative)
            target_report["cosine"].append(cosine)
            target_report["layers"].append(
                {
                    "layer": layer,
                    "candidate_count": candidate_count,
                    "mean_k": float(np.mean(selected_k)),
                    "p95_k": float(np.percentile(selected_k, 95)),
                    "maximum_k": int(np.max(selected_k)),
                    "maximum_k_clamp": min(maximum_k, candidate_count),
                    "mean_fraction": float(np.mean(selected_k) / width),
                    "target_attainment_rate": float(
                        np.mean(arm["target_attained"])
                    ),
                    "local_output_relative_l2": _summary(relative),
                }
            )

    input_count = min(
        hidden,
        max(1, int(math.ceil(policy["input_fraction"] * hidden))),
    )
    physical_base = native_bitnet_dip_physical_accounting(
        hidden,
        width,
        input_counts=[input_count] * layer_count,
        candidate_counts=policy["candidate_counts"],
        top_ks=[1] * layer_count,
    )
    record_bytes = int(physical_base["layout"]["base_record_payload_bytes"])
    fixed_layer_bytes = np.asarray(
        [
            layer["complete_modelled_cold_bytes"] - record_bytes
            for layer in physical_base["traffic"]["layers"]
        ],
        dtype=np.int64,
    )
    global_bytes = int(
        physical_base["traffic"]["global_header_directory_bytes"]
    )
    dense_q4_bytes = int(physical_base["traffic"]["dense_q4_bytes"])

    arms = []
    for target in targets:
        values = per_target[target]
        selected_k_by_layer = np.stack(values["selected_k"], axis=0)
        selected_k = selected_k_by_layer.reshape(-1)
        captured = np.concatenate(values["captured_energy"])
        attained = np.concatenate(values["target_attained"])
        relative = np.concatenate(values["relative_l2"])
        cosine = np.concatenate(values["cosine"])
        mean_fraction = float(np.mean(selected_k) / width)
        token_cold_bytes = (
            global_bytes
            + np.sum(
                fixed_layer_bytes[:, None]
                + selected_k_by_layer * record_bytes,
                axis=0,
            )
        )
        token_traffic_fraction = token_cold_bytes / dense_q4_bytes
        arms.append(
            {
                "energy_target": target,
                "selected_k": _summary(selected_k),
                "mean_active_fraction": mean_fraction,
                "p95_active_fraction": float(
                    np.percentile(selected_k, 95) / width
                ),
                "maximum_active_fraction": float(np.max(selected_k) / width),
                "captured_candidate_energy": _summary(captured),
                "target_attainment_rate": float(np.mean(attained)),
                "local_output_relative_l2": _summary(relative),
                "local_output_cosine_similarity": _summary(cosine),
                "physical_cold_traffic": {
                    "bytes_per_token": _summary(token_cold_bytes),
                    "fraction_of_dense_q4": _summary(
                        token_traffic_fraction
                    ),
                    "passes_45_percent_mean": float(
                        np.mean(token_traffic_fraction)
                    )
                    <= 0.45,
                    "accounting": (
                        "fixed coordinate scans and candidate completion from "
                        "the robust C policy plus token-adaptive selected-down "
                        "cache lines, gain/down-norm scans, and metadata"
                    ),
                },
                "layers": values["layers"],
                "meets_mean_k_budget": (
                    mean_fraction <= mean_budget_fraction + 1e-12
                ),
                "meets_physical_traffic_budget": float(
                    np.mean(token_traffic_fraction)
                )
                <= 0.45,
            }
        )
    eligible = [
        arm
        for arm in arms
        if arm["meets_mean_k_budget"]
        and arm["meets_physical_traffic_budget"]
    ]
    selected_arm = max(eligible, key=lambda arm: arm["energy_target"]) if eligible else None
    result = {
        "experiment": "native_bitnet_dip_token_adaptive_k",
        "scope": "validation_only_local_mlp_output_ceiling",
        "package": str(package_path),
        "artifact_sha256": artifact.payload_sha256,
        "validation_trace": {
            "path": str(trace_path),
            "manifest_sha256": trace_manifest_sha,
            "dataset_hash": trace.manifest.get("dataset_hash"),
            "records": trace.records,
            "causal_or_final_confirmation_corpus_used": False,
        },
        "router_policy": policy,
        "configuration": {
            "energy_targets": list(targets),
            "minimum_k": minimum_k,
            "maximum_k": maximum_k,
            "minimum_fraction": minimum_k / width,
            "maximum_fraction": maximum_k / width,
            "mean_budget_fraction": mean_budget_fraction,
            "device": device,
            "utility": "exact_candidate_coefficient_squared_times_down_column_l2",
            "energy_denominator": "sum_of_exact_utility_inside_practical_candidates",
            "stable_tie_break": "descending_utility_then_ascending_source_index",
        },
        "arms": arms,
        "physical_layout": {
            "serialization": physical_base["serialization"],
            "layout": physical_base["layout"],
            "fixed_candidate_policy": policy["candidate_counts"],
            "selected_down_record_bytes": record_bytes,
        },
        "selected_highest_energy_target_within_mean_budget": selected_arm,
        "decision": (
            "evaluate_adaptive_k_in_candidate_only_causal_path"
            if selected_arm is not None
            else "adaptive_k_targets_exceed_mean_budget"
        ),
        "milestone_2_status": "blocked",
        "caveat": (
            "Candidate membership is practical DIP, but exact teacher "
            "coefficient values and their full-width normalization scale are "
            "used to isolate adaptive-K selection and local output error. "
            "Traffic is cache-line modeled rather than hardware measured. "
            "This does not validate candidate-only scale estimation, causal "
            "quality, or CPU latency."
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(Path(out), result)
    return result


def evaluate_native_bitnet_dip_joint_policy(
    package: str | Path,
    validation_trace: str | Path,
    *,
    out: str | Path,
    candidate_counts: Sequence[int] = DEFAULT_JOINT_CANDIDATE_COUNTS,
    input_fraction: float = 0.75,
    minimum_fraction: float = 0.05,
    mean_budget_fraction: float = 0.25,
    maximum_traffic_fraction: float = 0.45,
    device: str = "cuda",
) -> dict[str, Any]:
    """Jointly select C and target=1 adaptive Kmax under physical traffic."""

    if any(
        not np.isfinite(value) or not 0 < value <= 1
        for value in (
            input_fraction,
            minimum_fraction,
            mean_budget_fraction,
            maximum_traffic_fraction,
        )
    ):
        raise ValueError("joint-policy fractions must lie in (0, 1]")
    package_path = Path(package).resolve()
    trace_path = Path(validation_trace).resolve()
    trace = _load_trajectories(trace_path)
    trace_manifest_sha = sha256_file(trace_path / "manifest.json")
    manifest = json.loads(
        (package_path / "manifest.json").read_text(encoding="utf-8")
    )
    artifact = load_native_bitnet_artifact(
        package_path / manifest["mlp"]["path"]
    )
    hidden = artifact.hidden_size
    width = artifact.intermediate_size
    layer_count = len(artifact.layers)
    if trace.hidden_size != hidden or trace.num_stages != layer_count:
        raise ValueError("validation trace dimensions differ from the package")
    if (
        trace.manifest.get("model_hash")
        != manifest.get("source", {}).get("weight_sha256")
    ):
        raise ValueError("validation trace was captured from another teacher")
    requested_candidates = tuple(
        sorted(dict.fromkeys(int(value) for value in candidate_counts))
    )
    if not requested_candidates or any(
        not 0 < value <= width for value in requested_candidates
    ):
        raise ValueError("candidate counts must be within the intermediate width")
    input_count = min(
        hidden,
        max(1, int(math.ceil(input_fraction * hidden))),
    )
    minimum_k = min(
        width,
        max(1, int(math.ceil(minimum_fraction * width))),
    )
    candidate_grid = []
    for candidate_count in requested_candidates:
        physical = maximum_k_under_physical_limit(
            hidden,
            width,
            input_count=input_count,
            candidate_count=candidate_count,
            maximum_traffic_fraction=maximum_traffic_fraction,
        )
        if int(physical["maximum_k"]) >= minimum_k:
            candidate_grid.append((candidate_count, physical))
    if not candidate_grid:
        raise ValueError("no candidate arm leaves room for the minimum K")

    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("joint C/K evaluation requires torch and safetensors") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA joint C/K evaluation requested but unavailable")
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
    epsilon = float(manifest["model"]["rms_norm_eps"])
    started = time.perf_counter()
    layer_reports = []
    selected_k_by_layer = []
    selected_relative_by_layer = []
    selected_cosine_by_layer = []

    for layer in range(layer_count):
        states = _trace_mlp_states(
            trace,
            layer,
            norm_weights[layer],
            epsilon,
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
        # Fit the policy at the same stored-activation and operator boundaries
        # used by the direct CPU kernel.  The older float32 research path can
        # change exact-zero membership after ReLU-squared gating, which in turn
        # changes target=1 adaptive K.
        quantized = _activation_quant_bf16(_bf16_round(states))
        exact_gate = _bf16_round(
            _bf16_round(quantized @ gate_codes.T) * gate_scale
        )
        exact_up = _bf16_round(
            _bf16_round(quantized @ up_codes.T) * up_scale
        )
        raw = _native_raw_activation(exact_gate, exact_up)
        inverse = np.reciprocal(
            np.sqrt(
                np.mean(raw * raw, axis=1, keepdims=True)
                + artifact.rms_norm_eps
            )
        )
        normalized = _bf16_round(
            _bf16_round(raw * inverse) * gain[None, :]
        )
        coefficients = _activation_quant_bf16(normalized)
        exact_utility = (
            coefficients * coefficients * down_norm_squared[None, :]
        )
        full_utility = np.sum(exact_utility, axis=1)

        coordinates = np.argsort(
            -np.abs(quantized),
            axis=1,
            kind="stable",
        )[:, :input_count]
        masked = np.zeros_like(quantized)
        np.put_along_axis(
            masked,
            coordinates,
            np.take_along_axis(quantized, coordinates, axis=1),
            axis=1,
        )
        partial_gate = _bf16_round(
            _bf16_round(masked @ gate_codes.T) * gate_scale
        )
        partial_up = _bf16_round(
            _bf16_round(masked @ up_codes.T) * up_scale
        )
        partial_raw = _native_raw_activation(partial_gate, partial_up)
        proxy = (
            partial_raw
            * partial_raw
            * gain[None, :]
            * gain[None, :]
            * down_norm_squared[None, :]
        )
        proxy_order = np.argsort(-proxy, axis=1, kind="stable")

        sparse_batches = []
        arm_state = []
        for candidate_count, physical in candidate_grid:
            candidates = proxy_order[:, :candidate_count]
            adaptive = adaptive_k_from_candidate_utility(
                candidates,
                np.take_along_axis(exact_utility, candidates, axis=1),
                energy_targets=(1.0,),
                minimum_k=minimum_k,
                maximum_k=int(physical["maximum_k"]),
            )[1.0]
            sparse = np.zeros_like(coefficients)
            selected_utility = np.empty(len(states), dtype=np.float64)
            for row, selected_k in enumerate(adaptive["selected_k"]):
                selected = adaptive["sorted_indices"][row, :selected_k]
                sparse[row, selected] = coefficients[row, selected]
                selected_utility[row] = np.sum(
                    exact_utility[row, selected],
                    dtype=np.float64,
                )
            sparse_batches.append(sparse)
            captured_full = selected_utility / np.maximum(full_utility, 1e-30)
            captured_full[full_utility <= 1e-30] = 1.0
            arm_state.append(
                {
                    "candidate_count": candidate_count,
                    "physical": physical,
                    "selected_k": adaptive["selected_k"],
                    "candidate_energy_target_attainment_rate": float(
                        np.mean(adaptive["target_attained"])
                    ),
                    "captured_full_oracle_energy": captured_full,
                }
            )

        stacked = np.concatenate(sparse_batches, axis=0)
        coefficients_tensor = torch.from_numpy(stacked).to(device)
        reference_tensor = torch.from_numpy(coefficients).to(device)
        down_tensor = torch.from_numpy(down_codes.T.copy()).to(device)
        with torch.inference_mode():
            outputs = _bf16_round(
                _bf16_round(
                    (coefficients_tensor @ down_tensor).float().cpu().numpy()
                )
                * down_scale
            )
            reference = _bf16_round(
                _bf16_round(
                    (reference_tensor @ down_tensor).float().cpu().numpy()
                )
                * down_scale
            )
        del coefficients_tensor, reference_tensor, down_tensor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        row_count = len(states)
        arms = []
        for arm_index, state in enumerate(arm_state):
            approximation = outputs[
                arm_index * row_count : (arm_index + 1) * row_count
            ]
            relative, cosine = _row_output_metrics(approximation, reference)
            selected_k = state["selected_k"]
            arms.append(
                {
                    "candidate_count": state["candidate_count"],
                    "maximum_k": int(state["physical"]["maximum_k"]),
                    "selected_k": _summary(selected_k),
                    "mean_active_fraction": float(np.mean(selected_k) / width),
                    "candidate_energy_target_attainment_rate": state[
                        "candidate_energy_target_attainment_rate"
                    ],
                    "captured_full_oracle_energy": _summary(
                        state["captured_full_oracle_energy"]
                    ),
                    "local_output_relative_l2": _summary(relative),
                    "local_output_cosine_similarity": _summary(cosine),
                    "physical_maximum": state["physical"],
                    "_selected_k": selected_k,
                    "_relative": relative,
                    "_cosine": cosine,
                }
            )
        selected_arm = min(
            arms,
            key=lambda arm: (
                arm["local_output_relative_l2"]["mean"],
                arm["local_output_relative_l2"]["p95"],
                arm["local_output_relative_l2"]["maximum"],
                arm["candidate_count"],
            ),
        )
        selected_k_by_layer.append(selected_arm.pop("_selected_k"))
        selected_relative_by_layer.append(selected_arm.pop("_relative"))
        selected_cosine_by_layer.append(selected_arm.pop("_cosine"))
        for arm in arms:
            arm.pop("_selected_k", None)
            arm.pop("_relative", None)
            arm.pop("_cosine", None)
        layer_reports.append(
            {
                "layer": layer,
                "selected_policy": selected_arm,
                "arms": arms,
            }
        )

    selected_candidates = [
        report["selected_policy"]["candidate_count"]
        for report in layer_reports
    ]
    selected_maximum_ks = [
        report["selected_policy"]["maximum_k"]
        for report in layer_reports
    ]
    selected_k_matrix = np.stack(selected_k_by_layer, axis=0)
    relative = np.concatenate(selected_relative_by_layer)
    cosine = np.concatenate(selected_cosine_by_layer)
    physical_base = native_bitnet_dip_physical_accounting(
        hidden,
        width,
        input_counts=[input_count] * layer_count,
        candidate_counts=selected_candidates,
        top_ks=[1] * layer_count,
    )
    record_bytes = int(physical_base["layout"]["base_record_payload_bytes"])
    fixed_layer_bytes = np.asarray(
        [
            layer["complete_modelled_cold_bytes"] - record_bytes
            for layer in physical_base["traffic"]["layers"]
        ],
        dtype=np.int64,
    )
    token_bytes = (
        int(physical_base["traffic"]["global_header_directory_bytes"])
        + np.sum(
            fixed_layer_bytes[:, None]
            + selected_k_matrix * record_bytes,
            axis=0,
        )
    )
    dense_bytes = int(physical_base["traffic"]["dense_q4_bytes"])
    token_fractions = token_bytes / dense_bytes
    mean_k_fraction = float(np.mean(selected_k_matrix) / width)
    worst_physical = max(
        report["selected_policy"]["physical_maximum"]["fraction_of_dense_q4"]
        for report in layer_reports
    )
    passes = (
        mean_k_fraction <= mean_budget_fraction + 1e-12
        and worst_physical <= maximum_traffic_fraction + 1e-12
        and float(np.mean(token_fractions)) <= maximum_traffic_fraction
    )
    result = {
        "experiment": "native_bitnet_dip_joint_candidate_adaptive_k_policy",
        "scope": "validation_only_target1_local_mlp_optimization",
        "package": str(package_path),
        "artifact_sha256": artifact.payload_sha256,
        "validation_trace": {
            "path": str(trace_path),
            "manifest_sha256": trace_manifest_sha,
            "dataset_hash": trace.manifest.get("dataset_hash"),
            "records": trace.records,
            "causal_or_final_confirmation_corpus_used": False,
        },
        "configuration": {
            "input_fraction": input_fraction,
            "input_coordinates": input_count,
            "candidate_grid": [
                candidate for candidate, _physical in candidate_grid
            ],
            "minimum_k": minimum_k,
            "energy_target": 1.0,
            "mean_k_budget_fraction": mean_budget_fraction,
            "maximum_physical_traffic_fraction_per_layer": (
                maximum_traffic_fraction
            ),
            "device": device,
        },
        "selected_policy": {
            "candidate_counts": selected_candidates,
            "maximum_ks": selected_maximum_ks,
            "selected_k": _summary(selected_k_matrix),
            "mean_active_fraction": mean_k_fraction,
            "local_output_relative_l2": _summary(relative),
            "local_output_cosine_similarity": _summary(cosine),
            "physical_cold_traffic": {
                "bytes_per_token": _summary(token_bytes),
                "fraction_of_dense_q4": _summary(token_fractions),
                "worst_layer_maximum_fraction_of_dense_q4": worst_physical,
                "passes_45_percent": passes,
            },
        },
        "physical_layout": {
            "serialization": physical_base["serialization"],
            "layout": physical_base["layout"],
        },
        "layers": layer_reports,
        "progression_screen": {
            "mean_k_at_most_25_percent": (
                mean_k_fraction <= mean_budget_fraction + 1e-12
            ),
            "every_layer_maximum_at_most_45_percent": (
                worst_physical <= maximum_traffic_fraction + 1e-12
            ),
            "mean_physical_traffic_at_most_45_percent": (
                float(np.mean(token_fractions))
                <= maximum_traffic_fraction
            ),
            "passed": passes,
        },
        "decision": (
            "use_joint_policy_for_candidate_only_causal_development"
            if passes
            else "joint_policy_requires_global_budget_optimization"
        ),
        "milestone_2_status": "blocked",
        "caveat": (
            "Candidate membership is practical q=0.75 DIP, while exact "
            "teacher coefficients and full-width normalization scale are "
            "used for target=1 utility and local output selection. Traffic "
            "is cache-line modeled, not hardware measured. No causal or "
            "final-confirmation corpus is used."
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(Path(out), result)
    return result


__all__ = [
    "DEFAULT_ADAPTIVE_K_ENERGY_TARGETS",
    "DEFAULT_JOINT_CANDIDATE_COUNTS",
    "adaptive_k_from_candidate_utility",
    "evaluate_native_bitnet_dip_adaptive_k",
    "evaluate_native_bitnet_dip_joint_policy",
    "maximum_k_under_physical_limit",
]
