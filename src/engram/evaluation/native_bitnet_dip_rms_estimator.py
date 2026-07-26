"""Validation-only omitted-energy estimators for practical native-BitNet DIP.

The deployed DIP normalization problem is one scalar per state and layer:
after exact completion of the practical candidates, estimate the sum of
``raw_activation**2`` over records that were not completed.  This module
compares estimators without reading a causal or final-confirmation corpus.

The most important arm reserves a few slots from the existing exact candidate
completion budget.  It routes the top ``C-S`` records by semantic proxy
utility, then uses the remaining ``S`` exact reads for the largest proxy raw
energies outside that routed set.  The union still contains exactly ``C``
records, so record traffic is unchanged and audited records remain eligible
for semantic reranking.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.evaluation.native_bitnet_adaptive_k import (
    adaptive_k_from_candidate_utility,
)
from engram.evaluation.native_bitnet_router import _trace_mlp_states
from engram.models.native_bitnet import (
    decode_native_bitnet_layer,
    load_native_bitnet_artifact,
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


def top_proxy_raw_audit_union(
    candidate_order: np.ndarray,
    proxy_square: np.ndarray,
    *,
    candidate_count: int,
    audit_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return routed, top-raw audit, and exact-completion union indices."""

    order = np.asarray(candidate_order)
    proxy = np.asarray(proxy_square, dtype=np.float64)
    if (
        order.ndim != 2
        or proxy.shape != order.shape
        or not np.issubdtype(order.dtype, np.integer)
        or not np.all(np.isfinite(proxy))
        or np.any(proxy < 0)
        or not 0 < audit_count < candidate_count <= order.shape[1]
    ):
        raise ValueError("invalid candidate order/proxy/audit configuration")
    routed_count = candidate_count - audit_count
    routed = order[:, :routed_count].copy()
    audits = np.empty((len(order), audit_count), dtype=np.int64)
    for row in range(len(order)):
        tail = order[row, routed_count:]
        raw_order = np.argsort(
            -proxy[row, tail],
            kind="stable",
        )
        audits[row] = tail[raw_order[:audit_count]]
    union = np.concatenate([routed, audits], axis=1)
    return routed, audits, union


def stratified_proxy_audit(
    candidate_order: np.ndarray,
    proxy_square: np.ndarray,
    *,
    candidate_count: int,
    audit_count: int,
) -> tuple[np.ndarray, list[list[np.ndarray]], np.ndarray]:
    """Choose one deterministic median-rank audit per proxy-energy stratum."""

    order = np.asarray(candidate_order)
    proxy = np.asarray(proxy_square, dtype=np.float64)
    if (
        order.ndim != 2
        or proxy.shape != order.shape
        or not 0 < audit_count < candidate_count <= order.shape[1]
    ):
        raise ValueError("invalid stratified audit configuration")
    routed_count = candidate_count - audit_count
    routed = order[:, :routed_count].copy()
    all_strata: list[list[np.ndarray]] = []
    audits = np.empty((len(order), audit_count), dtype=np.int64)
    for row in range(len(order)):
        tail = order[row, routed_count:]
        tail = tail[
            np.argsort(-proxy[row, tail], kind="stable")
        ]
        strata = [
            np.asarray(values, dtype=np.int64)
            for values in np.array_split(tail, audit_count)
        ]
        if any(values.size == 0 for values in strata):
            raise ValueError("audit_count produces an empty stratum")
        all_strata.append(strata)
        audits[row] = [
            values[len(values) // 2]
            for values in strata
        ]
    return routed, all_strata, audits


def _fit_log_ratio(
    features: np.ndarray,
    target_ratio: np.ndarray,
    training: np.ndarray,
    evaluation: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    train_x = np.asarray(features[training], dtype=np.float64)
    mean = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (train_x - mean) / scale
    design = np.column_stack(
        [np.ones(len(standardized)), standardized]
    )
    target = np.log(np.maximum(target_ratio[training], 1e-12))
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ target,
    )
    validation = np.column_stack(
        [
            np.ones(int(np.count_nonzero(evaluation))),
            (features[evaluation] - mean) / scale,
        ]
    )
    return np.exp(validation @ coefficients)


def _fit_nonnegative_ratio(
    features: np.ndarray,
    target_ratio: np.ndarray,
    training: np.ndarray,
    evaluation: np.ndarray,
    *,
    iterations: int = 256,
) -> np.ndarray:
    """Fit a tiny NNLS ratio model with deterministic coordinate descent."""

    train_x = np.asarray(features[training], dtype=np.float64)
    target = np.asarray(target_ratio[training], dtype=np.float64)
    scale = np.sqrt(np.mean(train_x * train_x, axis=0))
    scale[scale < 1e-12] = 1.0
    train_x = train_x / scale
    coefficients = np.zeros(train_x.shape[1], dtype=np.float64)
    prediction = np.zeros_like(target)
    for _ in range(iterations):
        for feature in range(train_x.shape[1]):
            column = train_x[:, feature]
            residual = target - prediction + column * coefficients[feature]
            denominator = float(column @ column)
            updated = (
                max(0.0, float(column @ residual) / denominator)
                if denominator > 1e-20
                else 0.0
            )
            prediction += column * (updated - coefficients[feature])
            coefficients[feature] = updated
    return np.maximum(
        np.asarray(features[evaluation], dtype=np.float64)
        / scale
        @ coefficients,
        0.0,
    )


def _estimator_metrics(
    estimate: np.ndarray,
    exact_total: np.ndarray,
    exact_candidate: np.ndarray,
    exact_omitted: np.ndarray,
    sample_ids: np.ndarray,
    *,
    energy_floor: float,
) -> dict[str, Any]:
    predicted = np.maximum(np.asarray(estimate, dtype=np.float64), energy_floor)
    exact = np.maximum(np.asarray(exact_total, dtype=np.float64), energy_floor)
    full_relative = np.abs(predicted - exact) / exact
    inverse_rms_relative = np.abs(np.sqrt(exact / predicted) - 1.0)
    omitted_prediction = np.maximum(predicted - exact_candidate, 0.0)
    nonzero_omitted = exact_omitted > energy_floor
    omitted_relative = np.abs(
        omitted_prediction[nonzero_omitted]
        - exact_omitted[nonzero_omitted]
    ) / exact_omitted[nonzero_omitted]
    sequence_means = []
    sequence_maxima = []
    for sample in np.unique(sample_ids):
        values = full_relative[sample_ids == sample]
        sequence_means.append(float(np.mean(values)))
        sequence_maxima.append(float(np.max(values)))
    return {
        "full_raw_square_sum_relative_error": _summary(full_relative),
        "inverse_rms_relative_error": _summary(inverse_rms_relative),
        "omitted_raw_square_sum_relative_error_nonzero_only": (
            _summary(omitted_relative)
        ),
        "estimated_to_exact_full_energy_ratio": _summary(predicted / exact),
        "even_sequence_relative_error": _summary(
            full_relative[sample_ids % 2 == 0]
        ),
        "odd_sequence_relative_error": _summary(
            full_relative[sample_ids % 2 == 1]
        ),
        "per_sequence_mean_relative_error": _summary(
            np.asarray(sequence_means)
        ),
        "per_sequence_maximum_relative_error": _summary(
            np.asarray(sequence_maxima)
        ),
    }


def _row_output_metrics(
    approximation: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    actual = np.asarray(approximation, dtype=np.float64)
    exact = np.asarray(reference, dtype=np.float64)
    relative = np.linalg.norm(actual - exact, axis=1) / np.maximum(
        np.linalg.norm(exact, axis=1),
        1e-12,
    )
    return _summary(relative)


def evaluate_native_bitnet_dip_rms_estimators(
    package: str | Path,
    validation_trace: str | Path,
    joint_policy: str | Path,
    *,
    out: str | Path,
    layer: int = 9,
    audit_counts: Sequence[int] = (8, 32, 64),
    log_ridge: float = 1.0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Compare state-dependent omitted-energy estimators on one fitted trace."""

    started = time.perf_counter()
    package_path = Path(package).resolve()
    trace_path = Path(validation_trace).resolve()
    policy_path = Path(joint_policy).resolve()
    manifest = json.loads(
        (package_path / "manifest.json").read_text(encoding="utf-8")
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    trace = _load_trajectories(trace_path)
    artifact = load_native_bitnet_artifact(
        package_path / manifest["mlp"]["path"]
    )
    if (
        policy.get("experiment")
        != "native_bitnet_dip_joint_candidate_adaptive_k_policy"
        or policy.get("artifact_sha256") != artifact.payload_sha256
        or policy.get("validation_trace", {}).get("manifest_sha256")
        != sha256_file(trace_path / "manifest.json")
        or trace.manifest.get("model_hash")
        != manifest.get("source", {}).get("weight_sha256")
    ):
        raise ValueError("joint policy, package, and validation trace differ")
    layers = policy.get("layers")
    if (
        not isinstance(layers, list)
        or not 0 <= layer < len(layers)
        or layers[layer].get("layer") != layer
    ):
        raise ValueError("joint policy has no compatible layer")
    selected_policy = layers[layer].get("selected_policy")
    if not isinstance(selected_policy, dict):
        raise ValueError("joint policy has no selected layer policy")
    candidate_count = int(selected_policy["candidate_count"])
    maximum_k = int(selected_policy["maximum_k"])
    counts = tuple(dict.fromkeys(int(value) for value in audit_counts))
    if (
        not counts
        or any(value <= 0 or value >= candidate_count for value in counts)
        or log_ridge < 0
    ):
        raise ValueError("invalid audit counts or log ridge")

    try:
        import torch
        import torch.nn.functional as functional
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "native BitNet RMS evaluation requires torch and safetensors"
        ) from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA RMS evaluation requested but unavailable")
    torch_device = torch.device(device)
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

    def activation_quant(values):
        values32 = values.float()
        scale = 127.0 / values32.abs().amax(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-5)
        return (
            (values32 * scale).round().clamp(-128, 127) / scale
        ).to(values.dtype)

    with torch.inference_mode():
        state = torch.from_numpy(states).to(
            device=torch_device,
            dtype=torch.bfloat16,
        )
        gate_codes = torch.from_numpy(
            np.asarray(decoded["gate_codes"], dtype=np.int8)
        ).to(device=torch_device, dtype=torch.bfloat16)
        up_codes = torch.from_numpy(
            np.asarray(decoded["up_codes"], dtype=np.int8)
        ).to(device=torch_device, dtype=torch.bfloat16)
        down_codes = torch.from_numpy(
            np.asarray(decoded["down_codes"], dtype=np.int8)
        ).to(device=torch_device, dtype=torch.bfloat16)
        gain = torch.from_numpy(
            np.asarray(decoded["ffn_sub_norm"], dtype=np.float32)
        ).to(device=torch_device, dtype=torch.bfloat16)
        down_norm = torch.count_nonzero(down_codes, dim=0).float()
        gate_scale = torch.as_tensor(
            decoded["gate_scale"],
            device=torch_device,
            dtype=torch.bfloat16,
        )
        up_scale = torch.as_tensor(
            decoded["up_scale"],
            device=torch_device,
            dtype=torch.bfloat16,
        )
        down_scale = torch.as_tensor(
            decoded["down_scale"],
            device=torch_device,
            dtype=torch.bfloat16,
        )
        quantized = activation_quant(state)
        input_count = int(policy["configuration"]["input_coordinates"])
        coordinates = torch.argsort(
            quantized.abs(),
            dim=1,
            descending=True,
            stable=True,
        )[:, :input_count]
        masked = torch.zeros_like(quantized)
        masked.scatter_(
            1,
            coordinates,
            quantized.gather(1, coordinates),
        )
        proxy_gate = functional.linear(masked, gate_codes) * gate_scale
        proxy_up = functional.linear(masked, up_codes) * up_scale
        proxy_raw = torch.relu(proxy_gate).square() * proxy_up
        exact_gate = functional.linear(quantized, gate_codes) * gate_scale
        exact_up = functional.linear(quantized, up_codes) * up_scale
        exact_raw = torch.relu(exact_gate).square() * exact_up
        proxy_utility = (
            proxy_raw.float().square()
            * gain.float().square()[None, :]
            * down_norm[None, :]
        )
        candidate_order = torch.argsort(
            proxy_utility,
            dim=1,
            descending=True,
            stable=True,
        ).cpu().numpy()
        exact_variance = exact_raw.float().square().mean(dim=1)
        exact_normalized = (
            exact_raw.float()
            * torch.rsqrt(
                exact_variance[:, None] + artifact.rms_norm_eps
            )
        ).to(torch.bfloat16) * gain[None, :]
        exact_coefficients = activation_quant(exact_normalized)
        dense_output = (
            functional.linear(exact_coefficients, down_codes) * down_scale
        ).float().cpu().numpy()

    proxy_square = (
        proxy_raw.float().cpu().numpy().astype(np.float64) ** 2
    )
    exact_square = (
        exact_raw.float().cpu().numpy().astype(np.float64) ** 2
    )
    exact_coefficients_numpy = exact_coefficients.float().cpu().numpy()
    down_norm_numpy = down_norm.cpu().numpy()
    sample_ids = np.asarray(trace.sample_id)
    rows = np.arange(trace.records)[:, None]
    base_candidates = candidate_order[:, :candidate_count]
    proxy_candidate = np.sum(
        proxy_square[rows, base_candidates],
        axis=1,
    )
    exact_candidate = np.sum(
        exact_square[rows, base_candidates],
        axis=1,
    )
    proxy_total = np.sum(proxy_square, axis=1)
    exact_total = np.sum(exact_square, axis=1)
    proxy_omitted = np.maximum(proxy_total - proxy_candidate, 0.0)
    exact_omitted = np.maximum(exact_total - exact_candidate, 0.0)
    energy_floor = max(float(np.median(exact_total)) * 1e-12, 1e-20)

    arms: list[dict[str, Any]] = []

    def add_arm(
        name: str,
        estimate: np.ndarray,
        *,
        kind: str,
        audit_count: int = 0,
        routed_candidates: int = candidate_count,
        fitted_parameters: int = 0,
        local_output: dict[str, Any] | None = None,
    ) -> None:
        arms.append(
            {
                "name": name,
                "kind": kind,
                "audit_count": audit_count,
                "routed_semantic_candidates": routed_candidates,
                "exact_completion_records": candidate_count,
                "fitted_parameters": fitted_parameters,
                "parameter_cold_bytes": (
                    0 if fitted_parameters == 0 else 64
                ),
                "metrics": _estimator_metrics(
                    estimate,
                    exact_total,
                    exact_candidate,
                    exact_omitted,
                    sample_ids,
                    energy_floor=energy_floor,
                ),
                "exact_candidate_semantic_output_relative_l2": local_output,
            }
        )

    identity = exact_candidate + proxy_omitted
    add_arm("corrected_proxy_identity", identity, kind="no_fit_no_audit")
    candidate_ratio = exact_candidate / np.maximum(
        proxy_candidate,
        energy_floor,
    )
    ratio_estimate = exact_candidate + candidate_ratio * proxy_omitted
    add_arm("candidate_ratio", ratio_estimate, kind="no_fit_no_audit")

    delta = exact_square - proxy_square
    top_union_by_count: dict[int, np.ndarray] = {}
    for audit_count in counts:
        routed, _, union = top_proxy_raw_audit_union(
            candidate_order,
            proxy_square,
            candidate_count=candidate_count,
            audit_count=audit_count,
        )
        top_union_by_count[audit_count] = union
        estimate = proxy_total + np.sum(delta[rows, union], axis=1)
        add_arm(
            f"top_proxy_raw_square_reserve_{audit_count}",
            estimate,
            kind="no_fit_in_budget_audit",
            audit_count=audit_count,
            routed_candidates=candidate_count - audit_count,
        )

        routed_stratified, strata, audits = stratified_proxy_audit(
            candidate_order,
            proxy_square,
            candidate_count=candidate_count,
            audit_count=audit_count,
        )
        stratified_estimate = (
            proxy_total
            + np.sum(delta[rows, routed_stratified], axis=1)
        )
        for row, row_strata in enumerate(strata):
            stratified_estimate[row] += sum(
                len(stratum) * delta[row, audit]
                for stratum, audit in zip(
                    row_strata,
                    audits[row],
                    strict=True,
                )
            )
        add_arm(
            f"stratified_proxy_reserve_{audit_count}",
            np.maximum(stratified_estimate, energy_floor),
            kind="no_fit_in_budget_audit",
            audit_count=audit_count,
            routed_candidates=candidate_count - audit_count,
        )

    tail_top32 = np.empty(trace.records, dtype=np.float64)
    tail_nonzero_fraction = np.empty(trace.records, dtype=np.float64)
    for row in range(trace.records):
        tail = candidate_order[row, candidate_count:]
        ordered = np.sort(proxy_square[row, tail])
        tail_top32[row] = np.sum(ordered[-min(32, len(ordered)):])
        tail_nonzero_fraction[row] = np.count_nonzero(ordered) / len(ordered)
    safe_candidate_ratio = exact_candidate / np.maximum(
        proxy_candidate,
        energy_floor,
    )
    target_ratio = (
        exact_omitted + energy_floor
    ) / (proxy_omitted + energy_floor)
    log_features = np.column_stack(
        [
            np.log(np.maximum(proxy_candidate, energy_floor)),
            np.log(np.maximum(exact_candidate, energy_floor)),
            np.log(np.maximum(proxy_omitted, energy_floor)),
            np.log(np.maximum(proxy_total, energy_floor)),
            np.log(np.maximum(safe_candidate_ratio, 1e-12)),
            tail_top32 / np.maximum(proxy_omitted, energy_floor),
            tail_nonzero_fraction,
        ]
    )
    nonnegative_features = np.column_stack(
        [
            np.ones(trace.records),
            safe_candidate_ratio,
            tail_top32 / np.maximum(proxy_omitted, energy_floor),
            tail_nonzero_fraction,
            proxy_omitted / np.maximum(proxy_total, energy_floor),
        ]
    )
    log_prediction = np.empty(trace.records, dtype=np.float64)
    nonnegative_prediction = np.empty(trace.records, dtype=np.float64)
    for training_parity in (0, 1):
        training = sample_ids % 2 == training_parity
        evaluation = ~training
        log_prediction[evaluation] = _fit_log_ratio(
            log_features,
            target_ratio,
            training,
            evaluation,
            ridge=log_ridge,
        )
        nonnegative_prediction[evaluation] = _fit_nonnegative_ratio(
            nonnegative_features,
            target_ratio,
            training,
            evaluation,
        )
    add_arm(
        "sequence_crossfit_log_ratio_regression",
        exact_candidate + log_prediction * proxy_omitted,
        kind="sequence_disjoint_fitted",
        fitted_parameters=log_features.shape[1] + 1,
    )
    add_arm(
        "sequence_crossfit_nonnegative_ratio_regression",
        exact_candidate + nonnegative_prediction * proxy_omitted,
        kind="sequence_disjoint_fitted",
        fitted_parameters=nonnegative_features.shape[1],
    )

    # Confirm that the best audit union remains a semantic candidate set:
    # audited records are exact and are eligible for the same rerank.
    primary_audit = min(counts)
    primary_union = top_union_by_count[primary_audit]
    exact_utility = (
        exact_coefficients_numpy
        * exact_coefficients_numpy
        * down_norm_numpy[None, :]
    )

    def candidate_output(candidate_indices: np.ndarray) -> np.ndarray:
        candidate_utility = np.take_along_axis(
            exact_utility,
            candidate_indices,
            axis=1,
        )
        adaptive = adaptive_k_from_candidate_utility(
            candidate_indices,
            candidate_utility,
            energy_targets=[1.0],
            minimum_k=int(policy["configuration"]["minimum_k"]),
            maximum_k=maximum_k,
        )[1.0]
        sparse = np.zeros_like(exact_coefficients_numpy)
        for row, selected_k in enumerate(adaptive["selected_k"]):
            selected = adaptive["sorted_indices"][row, :selected_k]
            sparse[row, selected] = exact_coefficients_numpy[row, selected]
        with torch.inference_mode():
            return (
                functional.linear(
                    torch.from_numpy(sparse).to(
                        device=torch_device,
                        dtype=torch.bfloat16,
                    ),
                    down_codes,
                )
                * down_scale
            ).float().cpu().numpy()

    base_output = candidate_output(base_candidates)
    audit_output = candidate_output(primary_union)
    semantic_comparison = {
        "base_joint_candidate_output_relative_l2": _row_output_metrics(
            base_output,
            dense_output,
        ),
        "top_audit_union_output_relative_l2": _row_output_metrics(
            audit_output,
            dense_output,
        ),
        "audit_records_remain_semantic_rerank_eligible": True,
    }
    for arm in arms:
        if arm["name"] == f"top_proxy_raw_square_reserve_{primary_audit}":
            arm["exact_candidate_semantic_output_relative_l2"] = (
                semantic_comparison[
                    "top_audit_union_output_relative_l2"
                ]
            )
        elif arm["name"] == "corrected_proxy_identity":
            arm["exact_candidate_semantic_output_relative_l2"] = (
                semantic_comparison[
                    "base_joint_candidate_output_relative_l2"
                ]
            )

    identity_arm = next(
        arm for arm in arms if arm["name"] == "corrected_proxy_identity"
    )
    primary_arm = next(
        arm
        for arm in arms
        if arm["name"] == f"top_proxy_raw_square_reserve_{primary_audit}"
    )
    identity_error = identity_arm["metrics"][
        "full_raw_square_sum_relative_error"
    ]
    primary_error = primary_arm["metrics"][
        "full_raw_square_sum_relative_error"
    ]
    selected = bool(
        primary_error["mean"] < identity_error["mean"]
        and primary_error["p95"] < identity_error["p95"]
        and primary_error["maximum"] < identity_error["maximum"]
        and primary_arm["exact_completion_records"] == candidate_count
    )
    physical = selected_policy["physical_maximum"]
    result = {
        "experiment": "native_bitnet_dip_state_dependent_rms_estimator",
        "scope": "validation_only_joint_policy_single_layer",
        "status": (
            "positive_development_evidence"
            if selected
            else "rejected_development_probe"
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
            "sequences": int(np.unique(sample_ids).size),
            "causal_or_final_confirmation_corpus_used": False,
        },
        "joint_policy": {
            "path": str(policy_path),
            "sha256": sha256_file(policy_path),
            "layer": layer,
            "selected_layer_policy": selected_policy,
        },
        "configuration": {
            "device": device,
            "input_coordinates": input_count,
            "candidate_count": candidate_count,
            "maximum_k": maximum_k,
            "audit_counts": list(counts),
            "primary_audit_count": primary_audit,
            "log_ridge": log_ridge,
            "sequence_split": "sample_id_even_odd_crossfit",
            "energy_target": 1.0,
            "bf16_operator_boundaries": True,
        },
        "estimator_arms": arms,
        "primary_semantic_comparison": semantic_comparison,
        "traffic": {
            "record_completion_invariant": True,
            "base_exact_completion_records": candidate_count,
            "primary_routed_semantic_records": (
                candidate_count - primary_audit
            ),
            "primary_audit_records": primary_audit,
            "primary_exact_completion_union_records": candidate_count,
            "candidate_completion_record_bytes_unchanged": True,
            "additional_parameter_bytes": 0,
            "joint_policy_complete_modelled_cold_bytes": physical[
                "complete_modelled_cold_bytes"
            ],
            "joint_policy_fraction_of_dense_q4": physical[
                "fraction_of_dense_q4"
            ],
            "passes_45_percent": physical["fraction_of_dense_q4"] <= 0.45,
            "note": (
                "audit slots replace the lowest-priority routed slots; the "
                "exact-completion union stays C and all union records remain "
                "eligible for exact semantic reranking"
            ),
        },
        "progression_screen": {
            "improves_mean_full_energy_relative_error": (
                primary_error["mean"] < identity_error["mean"]
            ),
            "improves_p95_full_energy_relative_error": (
                primary_error["p95"] < identity_error["p95"]
            ),
            "improves_maximum_full_energy_relative_error": (
                primary_error["maximum"] < identity_error["maximum"]
            ),
            "exact_completion_traffic_unchanged": True,
            "passed": selected,
        },
        "decision": (
            "test_top_proxy_raw_audit_in_candidate_only_causal_development"
            if selected
            else "retain_corrected_proxy_estimator"
        ),
        "caveat": (
            "This is validation-only local normalization evidence. Exact "
            "dense raw energy is used only as an evaluation label. No causal "
            "or final corpus, native serialization/reload, measured DRAM "
            "traffic, or sustained CPU latency is claimed."
        ),
        "milestone_2_status": "blocked",
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(Path(out), result)
    return result


__all__ = [
    "evaluate_native_bitnet_dip_rms_estimators",
    "stratified_proxy_audit",
    "top_proxy_raw_audit_union",
]
