"""Learned practical-router probes for the native BitNet semantic oracle."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.models.native_bitnet import (
    _activation_quant,
    decode_native_bitnet_layer,
    load_native_bitnet_artifact,
)
from engram.evaluation.native_bitnet_dip_traffic import (
    native_bitnet_dip_physical_accounting,
)
from engram.training.controller_distillation import _load_trajectories
from engram.utils import atomic_json, sha256_file


DEFAULT_DIP_CANDIDATE_MULTIPLIERS = (
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
)


def native_bitnet_router_traffic(
    hidden_size: int,
    intermediate_size: int,
    *,
    rank: int,
    candidate_count: int,
    bytes_per_router_parameter: int = 2,
) -> dict[str, Any]:
    """Model cold bytes for a nonlinear low-rank router plus candidate records."""

    if min(hidden_size, intermediate_size, rank, candidate_count) <= 0:
        raise ValueError("router traffic dimensions must be positive")
    if candidate_count > intermediate_size:
        raise ValueError("candidate_count exceeds intermediate size")
    router_parameters = (
        hidden_size * rank + rank * intermediate_size + intermediate_size
    )
    router_bytes = router_parameters * bytes_per_router_parameter
    packed_width = (hidden_size + 4) // 5
    candidate_record_bytes = candidate_count * (3 * packed_width + 2)
    dense_q4_bytes = (3 * hidden_size * intermediate_size + 1) // 2
    total = router_bytes + candidate_record_bytes
    return {
        "router_parameters": router_parameters,
        "router_bytes": router_bytes,
        "candidate_record_bytes": candidate_record_bytes,
        "complete_modelled_bytes": total,
        "dense_q4_bytes": dense_q4_bytes,
        "fraction_of_dense_q4": total / dense_q4_bytes,
        "passes_45_percent": total / dense_q4_bytes <= 0.45,
        "includes": (
            "FP16 input/output router factors and bias plus complete packed "
            "gate/up/gain/down payload for every candidate"
        ),
        "excludes": "headers, alignment, and runtime scratch",
    }


def native_bitnet_dip_traffic(
    hidden_size: int,
    intermediate_size: int,
    *,
    input_fraction: float,
    candidate_count: int,
    top_k: int | None = None,
    maximum_traffic_fraction: float = 0.45,
) -> dict[str, Any]:
    """Return cache-line-honest physical traffic for one DIP layer."""

    if min(hidden_size, intermediate_size, candidate_count) <= 0:
        raise ValueError("DIP traffic dimensions must be positive")
    if candidate_count > intermediate_size:
        raise ValueError("candidate_count exceeds intermediate size")
    if (
        not np.isfinite(input_fraction)
        or not 0 < input_fraction <= 1
        or not np.isfinite(maximum_traffic_fraction)
        or not 0 < maximum_traffic_fraction <= 1
    ):
        raise ValueError("DIP traffic fractions must lie in (0, 1]")
    input_count = min(
        hidden_size,
        max(1, int(math.ceil(input_fraction * hidden_size))),
    )
    selected = candidate_count if top_k is None else int(top_k)
    if selected <= 0 or selected > candidate_count:
        raise ValueError("top_k must lie in [1, candidate_count]")
    physical = native_bitnet_dip_physical_accounting(
        hidden_size,
        intermediate_size,
        input_counts=(input_count,),
        candidate_counts=(candidate_count,),
        top_ks=(selected,),
    )
    layer = physical["traffic"]["layers"][0]
    traffic_fraction = float(layer["fraction_of_dense_q4"])
    return {
        "input_coordinates": input_count,
        "partial_coordinate_scan_bytes": layer[
            "partial_coordinate_scan_bytes"
        ],
        "candidate_completion_record_bytes": layer[
            "candidate_completion_record_bytes"
        ],
        "selected_down_record_bytes": layer["selected_down_record_bytes"],
        "gain_scan_bytes": layer["gain_scan_bytes"],
        "down_norm_scan_bytes": layer["down_norm_scan_bytes"],
        "layer_header_and_scale_bytes": layer[
            "layer_header_and_scale_bytes"
        ],
        "complete_modelled_cold_bytes": layer[
            "complete_modelled_cold_bytes"
        ],
        "complete_modelled_bytes": layer["complete_modelled_cold_bytes"],
        "dense_q4_bytes": layer["dense_q4_bytes"],
        "fraction_of_dense_q4": traffic_fraction,
        "maximum_traffic_fraction": maximum_traffic_fraction,
        "passes_traffic_limit": traffic_fraction <= maximum_traffic_fraction,
        "passes_45_percent": traffic_fraction <= 0.45,
        "metadata_included": True,
        "cache_line_bytes": physical["layout"]["cache_line_bytes"],
        "accounting": (
            "cache-line-aligned coordinate gate/up rows, complete record-major "
            "gate/up candidate completion, selected top-K down rows, full "
            "gain/down-norm scans, and layer headers/scales"
        ),
    }


def maximum_native_bitnet_dip_candidates(
    hidden_size: int,
    intermediate_size: int,
    *,
    input_fraction: float,
    top_k: int | None = None,
    maximum_traffic_fraction: float = 0.45,
) -> int:
    """Return the largest complete-record candidate count within a byte cap."""

    selected = 1 if top_k is None else int(top_k)
    if selected <= 0 or selected > intermediate_size:
        raise ValueError("top_k is outside the intermediate width")
    low = selected
    high = intermediate_size
    best = 0
    while low <= high:
        candidate = (low + high) // 2
        physical = native_bitnet_dip_physical_accounting(
            hidden_size,
            intermediate_size,
            input_counts=(
                min(
                    hidden_size,
                    max(1, int(math.ceil(input_fraction * hidden_size))),
                ),
            ),
            candidate_counts=(candidate,),
            top_ks=(selected,),
        )
        fraction = physical["traffic"]["layers"][0][
            "fraction_of_dense_q4"
        ]
        if fraction <= maximum_traffic_fraction:
            best = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    return best


def _rms_norm(states: np.ndarray, weight: np.ndarray, epsilon: float) -> np.ndarray:
    values = np.asarray(states, dtype=np.float32)
    inverse = np.reciprocal(
        np.sqrt(np.mean(values * values, axis=1, keepdims=True) + epsilon)
    )
    return values * inverse * np.asarray(weight, dtype=np.float32)[None, :]


def _oracle_membership(
    artifact,
    layer: int,
    states: np.ndarray,
    top_k: int,
) -> np.ndarray:
    decoded = decode_native_bitnet_layer(artifact, layer)
    quantized = _activation_quant(np.asarray(states, dtype=np.float32))
    gate = (
        quantized @ np.asarray(decoded["gate_codes"], dtype=np.float32).T
        * np.asarray(decoded["gate_scale"], dtype=np.float32)
    )
    up = (
        quantized @ np.asarray(decoded["up_codes"], dtype=np.float32).T
        * np.asarray(decoded["up_scale"], dtype=np.float32)
    )
    activation = np.maximum(gate, 0.0) ** 2 * up
    inverse = np.reciprocal(
        np.sqrt(
            np.mean(activation * activation, axis=1, keepdims=True)
            + artifact.rms_norm_eps
        )
    )
    normalized = (
        activation
        * inverse
        * np.asarray(decoded["ffn_sub_norm"], dtype=np.float32)[None, :]
    )
    coefficients = _activation_quant(normalized)
    down = np.asarray(decoded["down_codes"], dtype=np.float32)
    utility = coefficients * coefficients * np.sum(down * down, axis=0)[None, :]
    # Stable ordering is part of the oracle contract. It is especially
    # important for BitNet because ReLU-squared gating can leave a large
    # exactly-zero tail; argpartition would choose that tail arbitrarily and
    # turn recall into a test of nondeterministic zero ties.
    selected = np.argsort(-utility, axis=1, kind="stable")[:, :top_k]
    membership = np.zeros(utility.shape, dtype=bool)
    np.put_along_axis(membership, selected, True, axis=1)
    return membership


def _trace_mlp_states(trace, layer: int, norm_weight, epsilon: float) -> np.ndarray:
    # Both terms were divided by the same incoming-state RMS at capture time.
    # RMSNorm removes that common scale; only the negligible epsilon scaling
    # differs from the uncaptured raw boundary.
    post_attention = (
        trace.teacher_states[:, layer].astype(np.float32)
        + trace.episodic_outputs[:, layer].astype(np.float32)
    )
    return _rms_norm(post_attention, norm_weight, epsilon)


def _validated_candidate_counts(
    top_k: int,
    maximum_candidates: int,
    intermediate_size: int,
    multipliers: Sequence[float],
) -> tuple[int, ...]:
    requested = tuple(dict.fromkeys(float(value) for value in multipliers))
    if not requested or any(
        not np.isfinite(value) or value < 1 for value in requested
    ):
        raise ValueError("candidate_multipliers must be finite and at least one")
    if maximum_candidates < top_k:
        return ()
    counts = {
        min(intermediate_size, int(math.ceil(multiplier * top_k)))
        for multiplier in requested
        if int(math.ceil(multiplier * top_k)) <= maximum_candidates
    }
    counts.add(top_k)
    counts.add(maximum_candidates)
    return tuple(sorted(counts))


def analyze_native_bitnet_dip_layer(
    artifact,
    layer: int,
    states: np.ndarray,
    *,
    top_k: int,
    input_fraction: float = 0.75,
    candidate_multipliers: Sequence[float] = DEFAULT_DIP_CANDIDATE_MULTIPLIERS,
    maximum_traffic_fraction: float = 0.45,
    recall_gate: float = 0.95,
    tail_recall_preference: float = 0.99,
    worst_row_recall_preference: float = 0.95,
) -> dict[str, Any]:
    """Measure stable DIP-to-oracle recall for one native BitNet layer.

    Exact gate/up coefficients are computed once for the evaluation-only
    oracle. The practical selector then uses only a stable top-magnitude
    subset of input coordinates to produce one reusable candidate ordering.
    """

    values = np.asarray(states, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[1] != artifact.hidden_size
        or not values.size
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("states must be a non-empty finite [records, hidden] matrix")
    if not 0 <= layer < len(artifact.layers):
        raise ValueError("layer is outside the artifact")
    width = artifact.intermediate_size
    hidden = artifact.hidden_size
    if not 0 < top_k <= width:
        raise ValueError("top_k is outside the intermediate width")
    if not np.isfinite(recall_gate) or not 0 < recall_gate <= 1:
        raise ValueError("recall_gate must lie in (0, 1]")
    if (
        not np.isfinite(tail_recall_preference)
        or not 0 < tail_recall_preference <= 1
    ):
        raise ValueError("tail_recall_preference must lie in (0, 1]")
    if (
        not np.isfinite(worst_row_recall_preference)
        or not 0 < worst_row_recall_preference <= 1
    ):
        raise ValueError("worst_row_recall_preference must lie in (0, 1]")

    maximum_candidates = maximum_native_bitnet_dip_candidates(
        hidden,
        width,
        input_fraction=input_fraction,
        top_k=top_k,
        maximum_traffic_fraction=maximum_traffic_fraction,
    )
    candidate_counts = _validated_candidate_counts(
        top_k,
        maximum_candidates,
        width,
        candidate_multipliers,
    )
    if not candidate_counts:
        raise ValueError("the oracle top_k cannot fit within the DIP traffic limit")

    decoded = decode_native_bitnet_layer(artifact, layer)
    gate_codes = np.asarray(decoded["gate_codes"], dtype=np.float32)
    up_codes = np.asarray(decoded["up_codes"], dtype=np.float32)
    down = np.asarray(decoded["down_codes"], dtype=np.float32)
    gate_scale = np.asarray(decoded["gate_scale"], dtype=np.float32)
    up_scale = np.asarray(decoded["up_scale"], dtype=np.float32)
    gain_squared = np.asarray(decoded["ffn_sub_norm"], dtype=np.float32) ** 2
    down_norm_squared = np.sum(down * down, axis=0, dtype=np.float32)
    quantized = _activation_quant(values)

    # Compute the exact oracle once. Stable sorting gives source-index
    # precedence to exact-zero ties created by ReLU-squared gating.
    exact_gate = quantized @ gate_codes.T * gate_scale
    exact_up = quantized @ up_codes.T * up_scale
    raw = np.maximum(exact_gate, 0.0) ** 2 * exact_up
    inverse = np.reciprocal(
        np.sqrt(
            np.mean(raw * raw, axis=1, keepdims=True)
            + artifact.rms_norm_eps
        )
    )
    normalized = (
        raw
        * inverse
        * np.asarray(decoded["ffn_sub_norm"], dtype=np.float32)[None, :]
    )
    coefficients = _activation_quant(normalized)
    oracle_utility = coefficients * coefficients * down_norm_squared[None, :]
    oracle_order = np.argsort(-oracle_utility, axis=1, kind="stable")
    oracle = np.zeros(oracle_utility.shape, dtype=bool)
    np.put_along_axis(oracle, oracle_order[:, :top_k], True, axis=1)

    input_count = min(
        hidden,
        max(1, int(math.ceil(input_fraction * hidden))),
    )
    coordinate_order = np.argsort(-np.abs(quantized), axis=1, kind="stable")
    selected_coordinates = coordinate_order[:, :input_count]
    masked = np.zeros_like(quantized)
    np.put_along_axis(
        masked,
        selected_coordinates,
        np.take_along_axis(quantized, selected_coordinates, axis=1),
        axis=1,
    )
    partial_gate = masked @ gate_codes.T * gate_scale
    partial_up = masked @ up_codes.T * up_scale
    partial_raw = np.maximum(partial_gate, 0.0) ** 2 * partial_up
    proxy = (
        partial_raw
        * partial_raw
        * gain_squared[None, :]
        * down_norm_squared[None, :]
    )
    candidate_order = np.argsort(-proxy, axis=1, kind="stable")

    ordered_membership = np.take_along_axis(oracle, candidate_order, axis=1)
    cumulative_hits = np.cumsum(
        ordered_membership,
        axis=1,
        dtype=np.int32,
    )
    usable_hits = cumulative_hits[:, top_k - 1 : maximum_candidates]
    mean_curve = np.mean(usable_hits, axis=0) / top_k
    p05_curve = np.percentile(usable_hits, 5, axis=0) / top_k
    minimum_curve = np.min(usable_hits, axis=0) / top_k
    mean_passing = np.flatnonzero(mean_curve >= recall_gate)
    tail_passing = np.flatnonzero(
        (mean_curve >= recall_gate)
        & (p05_curve >= tail_recall_preference)
        & (minimum_curve >= worst_row_recall_preference)
    )
    minimum_mean_count = (
        top_k + int(mean_passing[0]) if mean_passing.size else None
    )
    minimum_tail_count = (
        top_k + int(tail_passing[0]) if tail_passing.size else None
    )
    candidate_counts = tuple(
        sorted(
            set(candidate_counts)
            | {
                value
                for value in (minimum_mean_count, minimum_tail_count)
                if value is not None
            }
        )
    )

    def build_arm(candidates: int) -> dict[str, Any]:
        row_hits = cumulative_hits[:, candidates - 1]
        row_recall = row_hits / top_k
        traffic = native_bitnet_dip_traffic(
            hidden,
            width,
            input_fraction=input_fraction,
            candidate_count=candidates,
            top_k=top_k,
            maximum_traffic_fraction=maximum_traffic_fraction,
        )
        mean_recall = float(np.mean(row_recall))
        p05_recall = float(np.percentile(row_recall, 5))
        return {
            "candidate_count": candidates,
            "candidate_multiplier": candidates / top_k,
            "candidate_recall": {
                "mean": mean_recall,
                "minimum": float(np.min(row_recall)),
                "p05": p05_recall,
                "membership_hits": int(np.sum(row_hits)),
                "membership_total": int(values.shape[0] * top_k),
            },
            "traffic": traffic,
            "meets_joint_screen": (
                mean_recall >= recall_gate
                and traffic["passes_traffic_limit"]
            ),
            "meets_tail_preference": (
                mean_recall >= recall_gate
                and p05_recall >= tail_recall_preference
                and float(np.min(row_recall)) >= worst_row_recall_preference
                and traffic["passes_traffic_limit"]
            ),
        }

    arms = []
    for candidates in candidate_counts:
        arms.append(build_arm(candidates))
    passing = [arm for arm in arms if arm["meets_joint_screen"]]
    tail_preferred = [arm for arm in arms if arm["meets_tail_preference"]]
    mean_only_arm = (
        min(
            passing,
            key=lambda arm: (
                arm["traffic"]["complete_modelled_bytes"],
                -arm["candidate_recall"]["mean"],
            ),
        )
        if passing
        else max(
            arms,
            key=lambda arm: (
                arm["candidate_recall"]["mean"],
                -arm["traffic"]["complete_modelled_bytes"],
            ),
        )
    )
    selected_arm = (
        min(
            tail_preferred,
            key=lambda arm: (
                arm["traffic"]["complete_modelled_bytes"],
                -arm["candidate_recall"]["minimum"],
            ),
        )
        if tail_preferred
        else mean_only_arm
    )
    boundary_arm = next(
        arm
        for arm in arms
        if arm["candidate_count"] == maximum_candidates
    )
    return {
        "layer": int(layer),
        "records": int(values.shape[0]),
        "top_k": int(top_k),
        "input_fraction": float(input_fraction),
        "input_coordinates": input_count,
        "maximum_candidate_count_under_traffic_limit": maximum_candidates,
        "stable_tie_break": "descending_score_then_ascending_source_index",
        "arms": arms,
        "minimum_mean_recall_arm": mean_only_arm,
        "minimum_tail_preferred_arm": (
            min(
                tail_preferred,
                key=lambda arm: arm["traffic"]["complete_modelled_bytes"],
            )
            if tail_preferred
            else None
        ),
        "physical_boundary_arm": boundary_arm,
        "selected_arm": selected_arm,
        "passes_recall_and_traffic": bool(passing),
        "meets_tail_preference": bool(tail_preferred),
        "tail_recall_preference": float(tail_recall_preference),
        "worst_row_recall_preference": float(
            worst_row_recall_preference
        ),
    }


def load_native_bitnet_adaptive_schedule(
    path: str | Path,
    *,
    layer_count: int,
    intermediate_size: int,
) -> dict[str, Any]:
    """Validate and return the frozen, quality-passing oracle schedule."""

    schedule_path = Path(path).resolve()
    try:
        payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("oracle schedule is not readable JSON") from exc
    if (
        payload.get("experiment")
        != "native_bitnet_oracle_causal_substitution"
        or payload.get("scope") != "all_mlp_layers_exact_membership_ceiling"
        or payload.get("quality_passed") is not True
    ):
        raise ValueError("oracle schedule must be a quality-passing causal oracle")
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("oracle schedule has no configuration")
    top_ks = configuration.get("layer_top_ks")
    if (
        configuration.get("requested_fraction") is not None
        or not isinstance(top_ks, list)
        or len(top_ks) != layer_count
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 < value <= intermediate_size
            for value in top_ks
        )
    ):
        raise ValueError("oracle schedule is not a valid layer-adaptive allocation")
    sequence_count = configuration.get("sequence_count")
    prediction_positions = configuration.get("prediction_positions")
    if (
        not isinstance(sequence_count, int)
        or sequence_count < 8
        or not isinstance(prediction_positions, int)
        or prediction_positions < 256
    ):
        raise ValueError(
            "oracle schedule does not meet the frozen 8-sequence/256-position protocol"
        )
    if configuration.get("intermediate_size") != intermediate_size:
        raise ValueError("oracle schedule intermediate size differs from the package")
    mean_fraction = float(np.mean(top_ks) / intermediate_size)
    if mean_fraction > 0.25 + 1e-12:
        raise ValueError("oracle schedule exceeds the 25 percent mean record budget")
    return {
        "path": str(schedule_path),
        "sha256": sha256_file(schedule_path),
        "experiment": payload["experiment"],
        "sequence_count": sequence_count,
        "prediction_positions": prediction_positions,
        "quality_passed": True,
        "layer_top_ks": [int(value) for value in top_ks],
        "mean_active_fraction": mean_fraction,
    }


def evaluate_native_bitnet_dip_all_layers(
    package: str | Path,
    validation_trace: str | Path,
    oracle_schedule: str | Path,
    *,
    out: str | Path,
    input_fraction: float = 0.75,
    candidate_multipliers: Sequence[float] = DEFAULT_DIP_CANDIDATE_MULTIPLIERS,
    maximum_traffic_fraction: float = 0.45,
    recall_gate: float = 0.95,
    tail_recall_preference: float = 0.99,
    worst_row_recall_preference: float = 0.95,
) -> dict[str, Any]:
    """Run a deterministic DIP recall/traffic sweep over every BitNet MLP."""

    package_path = Path(package).resolve()
    trace_path = Path(validation_trace).resolve()
    trace = _load_trajectories(trace_path)
    manifest = json.loads(
        (package_path / "manifest.json").read_text(encoding="utf-8")
    )
    artifact = load_native_bitnet_artifact(
        package_path / manifest["mlp"]["path"]
    )
    layer_count = len(artifact.layers)
    if (
        trace.hidden_size != artifact.hidden_size
        or trace.num_stages != layer_count
    ):
        raise ValueError("validation trace dimensions differ from the BitNet artifact")
    expected_model_hash = manifest.get("source", {}).get("weight_sha256")
    if expected_model_hash and trace.manifest.get("model_hash") != expected_model_hash:
        raise ValueError("validation trace was captured from a different teacher")
    schedule = load_native_bitnet_adaptive_schedule(
        oracle_schedule,
        layer_count=layer_count,
        intermediate_size=artifact.intermediate_size,
    )

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("all-layer DIP evaluation requires safetensors") from exc
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
    for layer, top_k in enumerate(schedule["layer_top_ks"]):
        states = _trace_mlp_states(
            trace,
            layer,
            norm_weights[layer],
            epsilon,
        )
        layer_reports.append(
            analyze_native_bitnet_dip_layer(
                artifact,
                layer,
                states,
                top_k=top_k,
                input_fraction=input_fraction,
                candidate_multipliers=candidate_multipliers,
                maximum_traffic_fraction=maximum_traffic_fraction,
                recall_gate=recall_gate,
                tail_recall_preference=tail_recall_preference,
                worst_row_recall_preference=worst_row_recall_preference,
            )
        )

    selected = [report["selected_arm"] for report in layer_reports]
    mean_only = [
        report["minimum_mean_recall_arm"] for report in layer_reports
    ]
    boundaries = [report["physical_boundary_arm"] for report in layer_reports]
    hits = sum(
        arm["candidate_recall"]["membership_hits"] for arm in selected
    )
    memberships = sum(
        arm["candidate_recall"]["membership_total"] for arm in selected
    )
    layers_passing = sum(
        report["passes_recall_and_traffic"] for report in layer_reports
    )
    layers_meeting_tail = sum(
        report["meets_tail_preference"] for report in layer_reports
    )
    all_layers_passed = layers_passing == layer_count
    input_counts = [report["input_coordinates"] for report in layer_reports]
    selected_physical = native_bitnet_dip_physical_accounting(
        artifact.hidden_size,
        artifact.intermediate_size,
        input_counts=input_counts,
        candidate_counts=[arm["candidate_count"] for arm in selected],
        top_ks=schedule["layer_top_ks"],
    )
    mean_only_physical = native_bitnet_dip_physical_accounting(
        artifact.hidden_size,
        artifact.intermediate_size,
        input_counts=input_counts,
        candidate_counts=[arm["candidate_count"] for arm in mean_only],
        top_ks=schedule["layer_top_ks"],
    )
    boundary_physical = native_bitnet_dip_physical_accounting(
        artifact.hidden_size,
        artifact.intermediate_size,
        input_counts=input_counts,
        candidate_counts=[arm["candidate_count"] for arm in boundaries],
        top_ks=schedule["layer_top_ks"],
    )
    mean_recalls = [arm["candidate_recall"]["mean"] for arm in selected]
    p05_recalls = [arm["candidate_recall"]["p05"] for arm in selected]
    minimum_recalls = [
        arm["candidate_recall"]["minimum"] for arm in selected
    ]
    boundary_p05 = [
        arm["candidate_recall"]["p05"] for arm in boundaries
    ]
    boundary_minimum = [
        arm["candidate_recall"]["minimum"] for arm in boundaries
    ]
    result = {
        "experiment": "native_bitnet_dip_all_layer_recall_sweep",
        "scope": "all_mlp_layers_practical_router_recall_screen",
        "package": str(package_path),
        "artifact_sha256": artifact.payload_sha256,
        "validation_trace": {
            "path": str(trace_path),
            "manifest_sha256": sha256_file(trace_path / "manifest.json"),
            "dataset_hash": trace.manifest.get("dataset_hash"),
            "model_hash": trace.manifest.get("model_hash"),
            "records": trace.records,
        },
        "oracle_schedule": schedule,
        "configuration": {
            "input_fraction": float(input_fraction),
            "candidate_multipliers": [
                float(value) for value in candidate_multipliers
            ],
            "maximum_traffic_fraction": float(maximum_traffic_fraction),
            "recall_gate": float(recall_gate),
            "tail_recall_preference": float(tail_recall_preference),
            "worst_row_recall_preference": float(
                worst_row_recall_preference
            ),
            "selection_policy": (
                "choose the minimum physical-traffic arm meeting mean recall "
                "plus p05 and worst-row preferences; fall back to the "
                "minimum-traffic mean-recall arm when robust preferences "
                "are unreachable"
            ),
            "stable_tie_break": (
                "descending_score_then_ascending_source_index"
            ),
        },
        "layers": layer_reports,
        "aggregate": {
            "layer_count": layer_count,
            "layers_passing_recall_and_traffic": layers_passing,
            "layers_meeting_tail_recall_preference": layers_meeting_tail,
            "all_layers_passed": all_layers_passed,
            "selected_macro_mean_recall": float(np.mean(mean_recalls)),
            "selected_micro_membership_recall": hits / memberships,
            "selected_worst_layer_mean_recall": float(np.min(mean_recalls)),
            "selected_worst_layer_p05_recall": float(np.min(p05_recalls)),
            "selected_worst_row_recall": float(np.min(minimum_recalls)),
            "selected_complete_physical_traffic_fraction_of_dense_q4": (
                selected_physical["traffic"]["fraction_of_dense_q4"]
            ),
            "selected_maximum_traffic_fraction_of_dense_q4": float(
                np.max(
                    [
                        layer["fraction_of_dense_q4"]
                        for layer in selected_physical["traffic"]["layers"]
                    ]
                )
            ),
            "selected_candidate_counts": [
                arm["candidate_count"] for arm in selected
            ],
            "mean_only_candidate_counts": [
                arm["candidate_count"] for arm in mean_only
            ],
            "physical_boundary_candidate_counts": [
                arm["candidate_count"] for arm in boundaries
            ],
            "physical_boundary_worst_layer_p05_recall": float(
                np.min(boundary_p05)
            ),
            "physical_boundary_worst_row_recall": float(
                np.min(boundary_minimum)
            ),
        },
        "physical_accounting": {
            "selected_tail_preferred_schedule": selected_physical,
            "minimum_mean_recall_schedule": mean_only_physical,
            "maximum_candidate_boundary_schedule": boundary_physical,
        },
        "progression_screen": {
            "minimum_mean_candidate_recall_per_layer": float(recall_gate),
            "preferred_minimum_p05_candidate_recall_per_layer": float(
                tail_recall_preference
            ),
            "preferred_minimum_worst_row_candidate_recall": float(
                worst_row_recall_preference
            ),
            "maximum_complete_modelled_traffic_per_layer": float(
                maximum_traffic_fraction
            ),
            "passed": all_layers_passed,
        },
        "decision": (
            "implement_selected_record_coefficient_and_causal_kernel"
            if all_layers_passed
            else "revise_dip_before_selected_record_kernel"
        ),
        "milestone_2_status": "blocked",
        "caveat": (
            "This is a held-out recall and cache-line-honest modeled physical "
            "cold-traffic screen. It does not yet compute exact candidate "
            "coefficients, run causal substitution, measure hardware DRAM "
            "bytes, or benchmark the selected-record CPU kernel."
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(Path(out), result)
    return result


def evaluate_native_bitnet_low_rank_router(
    package: str | Path,
    training_trace: str | Path,
    validation_trace: str | Path,
    *,
    out: str | Path,
    layers: Sequence[int] = (0, 14, 29),
    top_ks: Sequence[int] = (1728, 1728, 2074),
    rank: int = 128,
    steps: int = 500,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    device: str = "cuda",
    seed: int = 20260726,
) -> dict[str, Any]:
    """Train nonlinear low-rank membership routers and measure held-out recall."""

    if len(layers) != len(top_ks) or not layers:
        raise ValueError("layers and top_ks must be non-empty and aligned")
    if min(rank, steps, batch_size) <= 0 or learning_rate <= 0:
        raise ValueError("router training hyperparameters must be positive")
    package_path = Path(package).resolve()
    train = _load_trajectories(training_trace)
    validation = _load_trajectories(validation_trace)
    if train.manifest["model_hash"] != validation.manifest["model_hash"]:
        raise ValueError("router traces use different teachers")
    if train.manifest["dataset_hash"] == validation.manifest["dataset_hash"]:
        raise ValueError("router training and validation datasets must differ")

    manifest = __import__("json").loads(
        (package_path / "manifest.json").read_text(encoding="utf-8")
    )
    artifact = load_native_bitnet_artifact(
        package_path / manifest["mlp"]["path"]
    )
    hidden = artifact.hidden_size
    width = artifact.intermediate_size
    selected_layers = tuple(int(value) for value in layers)
    selected_top_ks = tuple(int(value) for value in top_ks)
    if any(not 0 <= layer < len(artifact.layers) for layer in selected_layers):
        raise ValueError("router layer is outside the artifact")
    if any(not 0 < value <= width for value in selected_top_ks):
        raise ValueError("router top-K is outside the intermediate width")

    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("router probe requires torch and safetensors") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    weights_path = package_path / manifest["transformer"]["non_mlp_path"]
    norm_weights = {}
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for layer in selected_layers:
            name = f"model.layers.{layer}.post_attention_layernorm.weight"
            norm_weights[layer] = handle.get_tensor(name).float().numpy()
    epsilon = float(manifest["model"]["rms_norm_eps"])
    started = time.perf_counter()
    layer_reports = []

    for layer, top_k in zip(selected_layers, selected_top_ks, strict=True):
        train_states = _trace_mlp_states(train, layer, norm_weights[layer], epsilon)
        validation_states = _trace_mlp_states(
            validation, layer, norm_weights[layer], epsilon
        )
        train_membership = _oracle_membership(
            artifact, layer, train_states, top_k
        )
        validation_membership = _oracle_membership(
            artifact, layer, validation_states, top_k
        )
        mean = train_states.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = train_states.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-5] = 1.0
        train_x = torch.from_numpy((train_states - mean) / scale).to(device)
        train_y = torch.from_numpy(train_membership.astype(np.float32)).to(device)
        validation_x = torch.from_numpy(
            (validation_states - mean) / scale
        ).to(device)

        model = torch.nn.Sequential(
            torch.nn.Linear(hidden, rank, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(rank, width, bias=True),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        positive_weight = torch.tensor(
            (width - top_k) / top_k, dtype=torch.float32, device=device
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + layer)
        final_loss = None
        model.train()
        for _ in range(steps):
            indices = torch.randint(
                len(train_x),
                (min(batch_size, len(train_x)),),
                generator=generator,
            ).to(device)
            logits = model(train_x[indices])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                train_y[indices],
                pos_weight=positive_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())

        model.eval()
        with torch.inference_mode():
            scores = model(validation_x).float().cpu().numpy()
        order = np.argsort(-scores, axis=1, kind="stable")
        candidate_counts = tuple(
            sorted(
                {
                    top_k,
                    min(width, int(math.ceil(1.25 * top_k))),
                    min(width, int(math.ceil(1.5 * top_k))),
                }
            )
        )
        recalls = {}
        for candidates in candidate_counts:
            selected = order[:, :candidates]
            row_recall = (
                np.take_along_axis(validation_membership, selected, axis=1).sum(axis=1)
                / top_k
            )
            traffic = native_bitnet_router_traffic(
                hidden,
                width,
                rank=rank,
                candidate_count=candidates,
            )
            recalls[str(candidates)] = {
                "mean": float(np.mean(row_recall)),
                "minimum": float(np.min(row_recall)),
                "p05": float(np.percentile(row_recall, 5)),
                "meets_95_percent": float(np.mean(row_recall)) >= 0.95,
                "traffic": traffic,
            }
        layer_reports.append(
            {
                "layer": layer,
                "top_k": top_k,
                "rank": rank,
                "training_records": len(train_states),
                "validation_records": len(validation_states),
                "final_training_loss": final_loss,
                "candidate_recall": recalls,
            }
        )

    eligible = all(
        any(
            arm["meets_95_percent"] and arm["traffic"]["passes_45_percent"]
            for arm in report["candidate_recall"].values()
        )
        for report in layer_reports
    )
    result = {
        "experiment": "native_bitnet_nonlinear_low_rank_membership_router",
        "package": str(package_path),
        "artifact_sha256": artifact.payload_sha256,
        "training_trace": {
            "path": str(Path(training_trace).resolve()),
            "dataset_hash": train.manifest["dataset_hash"],
            "records": train.records,
        },
        "validation_trace": {
            "path": str(Path(validation_trace).resolve()),
            "dataset_hash": validation.manifest["dataset_hash"],
            "records": validation.records,
        },
        "configuration": {
            "layers": list(selected_layers),
            "top_ks": list(selected_top_ks),
            "rank": rank,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device,
            "seed": seed,
        },
        "layers": layer_reports,
        "recall_gate": 0.95,
        "eligible_for_all_layer_training": eligible,
        "decision": (
            "expand_router_training_to_all_layers"
            if eligible
            else "reject_or_revise_router_before_causal_work"
        ),
        "milestone_2_status": "blocked",
        "elapsed_seconds": time.perf_counter() - started,
        "trace_reconstruction_caveat": (
            "MLP inputs are reconstructed from normalized stage and attention "
            "traces; RMSNorm removes their shared scale up to epsilon."
        ),
    }
    atomic_json(Path(out), result)
    return result


def evaluate_native_bitnet_dip_router(
    package: str | Path,
    validation_trace: str | Path,
    *,
    out: str | Path,
    layer: int = 14,
    top_k: int = 1728,
    input_fractions: Sequence[float] = (0.25, 0.5, 0.75),
    candidate_multipliers: Sequence[float] = (1.0, 1.25, 1.5),
) -> dict[str, Any]:
    """Screen top-magnitude input-coordinate routing on held-out BitNet states."""

    fractions = tuple(dict.fromkeys(float(value) for value in input_fractions))
    multipliers = tuple(dict.fromkeys(float(value) for value in candidate_multipliers))
    if not fractions or any(not 0 < value <= 1 for value in fractions):
        raise ValueError("input_fractions must lie in (0, 1]")
    if not multipliers or any(value < 1 for value in multipliers):
        raise ValueError("candidate_multipliers must be at least one")
    package_path = Path(package).resolve()
    trace = _load_trajectories(validation_trace)
    import json
    from safetensors import safe_open

    manifest = json.loads((package_path / "manifest.json").read_text(encoding="utf-8"))
    artifact = load_native_bitnet_artifact(package_path / manifest["mlp"]["path"])
    if not 0 <= layer < len(artifact.layers) or not 0 < top_k <= artifact.intermediate_size:
        raise ValueError("layer or top_k is outside the artifact")
    weights_path = package_path / manifest["transformer"]["non_mlp_path"]
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
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
    oracle = _oracle_membership(artifact, layer, states, top_k)
    decoded = decode_native_bitnet_layer(artifact, layer)
    gate_codes = np.asarray(decoded["gate_codes"], dtype=np.float32)
    up_codes = np.asarray(decoded["up_codes"], dtype=np.float32)
    gain_squared = np.asarray(decoded["ffn_sub_norm"], dtype=np.float32) ** 2
    down = np.asarray(decoded["down_codes"], dtype=np.float32)
    down_norm_squared = np.sum(down * down, axis=0)
    quantized = _activation_quant(states)
    hidden = artifact.hidden_size
    width = artifact.intermediate_size
    dense_q4 = (3 * hidden * width + 1) // 2
    started = time.perf_counter()
    arms = []

    for fraction in fractions:
        count = min(hidden, max(1, int(math.ceil(fraction * hidden))))
        selected_coordinates = np.argpartition(
            -np.abs(quantized), count - 1, axis=1
        )[:, :count]
        masked = np.zeros_like(quantized)
        np.put_along_axis(
            masked,
            selected_coordinates,
            np.take_along_axis(quantized, selected_coordinates, axis=1),
            axis=1,
        )
        gate = (
            masked @ gate_codes.T
            * np.asarray(decoded["gate_scale"], dtype=np.float32)
        )
        up = (
            masked @ up_codes.T
            * np.asarray(decoded["up_scale"], dtype=np.float32)
        )
        raw = np.maximum(gate, 0.0) ** 2 * up
        proxy = raw * raw * gain_squared[None, :] * down_norm_squared[None, :]
        order = np.argsort(-proxy, axis=1, kind="stable")
        coordinate_index_bytes = math.ceil(2 * count * width / 5)
        for multiplier in multipliers:
            candidates = min(width, int(math.ceil(multiplier * top_k)))
            selected = order[:, :candidates]
            row_recall = (
                np.take_along_axis(oracle, selected, axis=1).sum(axis=1) / top_k
            )
            candidate_bytes = candidates * (3 * ((hidden + 4) // 5) + 2)
            complete = coordinate_index_bytes + candidate_bytes
            arms.append(
                {
                    "input_fraction": fraction,
                    "input_coordinates": count,
                    "candidate_multiplier": multiplier,
                    "candidate_count": candidates,
                    "candidate_recall": {
                        "mean": float(np.mean(row_recall)),
                        "minimum": float(np.min(row_recall)),
                        "p05": float(np.percentile(row_recall, 5)),
                    },
                    "traffic": {
                        "coordinate_major_gate_up_bytes": coordinate_index_bytes,
                        "complete_candidate_record_bytes": candidate_bytes,
                        "complete_modelled_bytes": complete,
                        "dense_q4_bytes": dense_q4,
                        "fraction_of_dense_q4": complete / dense_q4,
                        "passes_45_percent": complete / dense_q4 <= 0.45,
                        "metadata_included": False,
                    },
                    "meets_joint_screen": (
                        float(np.mean(row_recall)) >= 0.95
                        and complete / dense_q4 <= 0.45
                    ),
                }
            )
    best = max(
        arms,
        key=lambda arm: (
            arm["meets_joint_screen"],
            arm["candidate_recall"]["mean"],
            -arm["traffic"]["fraction_of_dense_q4"],
        ),
    )
    result = {
        "experiment": "native_bitnet_dynamic_input_pruning_router",
        "package": str(package_path),
        "validation_trace": str(Path(validation_trace).resolve()),
        "layer": layer,
        "top_k": top_k,
        "records": len(states),
        "arms": arms,
        "best_arm": best,
        "recall_gate": 0.95,
        "decision": (
            "implement_selected_record_causal_dip"
            if best["meets_joint_screen"]
            else "dip_recall_or_traffic_insufficient"
        ),
        "milestone_2_status": "blocked",
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(Path(out), result)
    return result


__all__ = [
    "DEFAULT_DIP_CANDIDATE_MULTIPLIERS",
    "analyze_native_bitnet_dip_layer",
    "evaluate_native_bitnet_dip_all_layers",
    "evaluate_native_bitnet_low_rank_router",
    "evaluate_native_bitnet_dip_router",
    "load_native_bitnet_adaptive_schedule",
    "maximum_native_bitnet_dip_candidates",
    "native_bitnet_dip_traffic",
    "native_bitnet_router_traffic",
]
