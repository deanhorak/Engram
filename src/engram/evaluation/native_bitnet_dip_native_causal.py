"""Fail-closed causal evaluation for the native selected-record DIP kernel.

The evaluator deliberately keeps three kinds of evidence separate:

* a dense native-BitNet teacher pass;
* one timed, non-debug native DIP pass; and
* an optional untimed debug pass used for route identities, canonical
  teacher-membership recall, and Python/native parity.

Every causal sequence supplies ``N + 1`` tokens and only the first ``N``
positions are scored.  The final input row is therefore excluded from active
record and traffic evidence as well as from quality metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from engram.evaluation.native_bitnet_dip_causal import (
    _causal_evidence_passed,
    _json_native,
    _load_causal_records,
    _prediction_views,
)
from engram.evaluation.native_bitnet_dip_kernel import (
    NativeBitNetDIPCPUKernel,
    NativeBitNetDIPKernelPolicy,
    NativeBitNetDIPTorchDiagnostics,
    substitute_native_bitnet_dip_kernel_mlps,
)
from engram.evaluation.native_bitnet_dip_traffic import (
    native_bitnet_dip_physical_accounting,
)
from engram.evaluation.native_bitnet_parity import (
    _logit_metrics,
    _tensor_metrics,
)
from engram.models.native_bitnet import load_native_bitnet_artifact
from engram.runtime.native_bitnet import NativeBitNetRuntime
from engram.semantic.native_bitnet_dip import NativeBitNetDIPLayer
from engram.utils import atomic_json, sha256_file, sha256_json


DatasetRole = Literal["development", "final"]

_QUALITY_THRESHOLDS = {
    "maximum_mean_kl_divergence": 0.05,
    "minimum_top1_agreement": 0.90,
    "maximum_nll_delta": 0.05,
    "maximum_final_hidden_relative_l2": 0.10,
}
_MAXIMUM_ACTIVE_FRACTION = 0.25
_MAXIMUM_PHYSICAL_TRAFFIC_FRACTION = 0.45
_MINIMUM_CANDIDATE_RECALL = 0.95
_UINT32_SENTINEL = np.iinfo(np.uint32).max


def _file_descriptor(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"required evidence artifact is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _array_sha256(values: NDArray[Any]) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _policy_report(
    policies: Sequence[NativeBitNetDIPKernelPolicy],
) -> dict[str, dict[str, Any]]:
    return {
        str(layer): {
            "input_coordinates": int(policy.input_coordinates),
            "candidate_count": int(policy.candidate_count),
            "minimum_top_k": int(policy.minimum_top_k),
            "maximum_top_k": int(policy.maximum_top_k),
            "energy_target": float(policy.energy_target),
            "rms_audit_count": int(policy.rms_audit_count),
            "rms_estimator": policy.rms_estimator,
            "rms_audit_strategy": policy.rms_audit_strategy,
            "rms_variance_scale": 1.0,
            "rms_variance_bias": 0.0,
            "output_scale": 1.0,
        }
        for layer, policy in enumerate(policies)
    }


def _validate_native_calls(
    calls: Sequence[Mapping[str, int]],
    *,
    policies: Sequence[NativeBitNetDIPKernelPolicy],
    rows: int,
) -> None:
    if len(calls) != len(policies):
        raise RuntimeError(
            "native DIP kernel did not execute exactly once per MLP layer"
        )
    for expected_layer, (call, policy) in enumerate(
        zip(calls, policies, strict=True)
    ):
        expected = {
            "layer": expected_layer,
            "rows": rows,
            "input_coordinates": policy.input_coordinates,
            "candidate_count": policy.candidate_count,
        }
        for name, value in expected.items():
            if int(call.get(name, -1)) != int(value):
                raise RuntimeError(
                    f"native DIP call contract mismatch at layer "
                    f"{expected_layer}: {name}"
                )
        selected_total = int(call.get("selected_count_total", -1))
        minimum = int(call.get("selected_count_min", -1))
        maximum = int(call.get("selected_count_max", -1))
        if (
            selected_total < rows * policy.minimum_top_k
            or selected_total > rows * policy.maximum_top_k
            or minimum < policy.minimum_top_k
            or maximum > policy.maximum_top_k
        ):
            raise RuntimeError(
                f"native DIP selected-count metrics are invalid at layer "
                f"{expected_layer}"
            )


def _selection_report(
    layer_counts: Mapping[int, NDArray[np.uint32]],
    *,
    sequence_count: int,
    predictions_per_sequence: int,
    intermediate_size: int,
) -> dict[str, Any]:
    layers = tuple(sorted(layer_counts))
    if layers != tuple(range(len(layers))):
        raise RuntimeError("selected-count evidence must cover consecutive layers")
    expected_shape = (sequence_count, predictions_per_sequence)
    normalized: dict[int, NDArray[np.uint32]] = {}
    for layer in layers:
        counts = np.asarray(layer_counts[layer], dtype=np.uint32)
        if counts.shape != expected_shape:
            raise RuntimeError(
                f"selected-count shape is invalid at layer {layer}"
            )
        normalized[layer] = counts

    schedules = [
        [
            int(normalized[layer][sequence, token])
            for layer in layers
        ]
        for sequence in range(sequence_count)
        for token in range(predictions_per_sequence)
    ]
    flat = np.asarray(schedules, dtype=np.uint32)
    total = int(np.sum(flat, dtype=np.uint64))
    count = int(flat.size)
    active_fraction = total / (count * intermediate_size)
    return {
        "per_token_layer_k": schedules,
        "global": {
            "sum": total,
            "count": count,
            "minimum": int(np.min(flat)),
            "maximum": int(np.max(flat)),
            "mean": total / count,
            "active_fraction": active_fraction,
        },
        "layers": {
            str(layer): {
                "sum": int(np.sum(normalized[layer], dtype=np.uint64)),
                "count": int(normalized[layer].size),
                "minimum": int(np.min(normalized[layer])),
                "maximum": int(np.max(normalized[layer])),
                "mean": float(np.mean(normalized[layer])),
                "active_fraction": float(
                    np.mean(normalized[layer]) / intermediate_size
                ),
            }
            for layer in layers
        },
    }


def _physical_traffic_report(
    *,
    hidden_size: int,
    intermediate_size: int,
    policies: Sequence[NativeBitNetDIPKernelPolicy],
    schedules: Sequence[Sequence[int]],
    predictions_per_sequence: int,
) -> dict[str, Any]:
    if not schedules:
        raise RuntimeError("physical traffic requires scored token schedules")
    input_counts = [int(policy.input_coordinates) for policy in policies]
    candidate_counts = [int(policy.candidate_count) for policy in policies]
    per_token: list[dict[str, Any]] = []
    scheduled_total = 0
    dense_total = 0
    layer_scheduled = [0] * len(policies)
    layer_dense = [0] * len(policies)
    worst_layer: dict[str, Any] | None = None
    for token, raw_schedule in enumerate(schedules):
        schedule = [int(value) for value in raw_schedule]
        accounting = native_bitnet_dip_physical_accounting(
            hidden_size,
            intermediate_size,
            input_counts=input_counts,
            candidate_counts=candidate_counts,
            top_ks=schedule,
        )
        traffic = accounting["traffic"]
        scheduled = int(traffic["complete_modelled_cold_bytes"])
        dense = int(traffic["dense_q4_bytes"])
        fraction = scheduled / dense
        scheduled_total += scheduled
        dense_total += dense
        per_token.append(
            {
                "token": token,
                "sequence": token // predictions_per_sequence,
                "prediction_position": token % predictions_per_sequence,
                "scheduled_cache_line_bytes": scheduled,
                "dense_q4_bytes": dense,
                "fraction_of_dense_q4": fraction,
            }
        )
        for layer_report in traffic["layers"]:
            layer_index = int(layer_report["layer"])
            layer_scheduled[layer_index] += int(
                layer_report["complete_modelled_cold_bytes"]
            )
            layer_dense[layer_index] += int(layer_report["dense_q4_bytes"])
            candidate = {
                "token": token,
                "sequence": token // predictions_per_sequence,
                "prediction_position": token % predictions_per_sequence,
                "layer": int(layer_report["layer"]),
                "top_k": int(layer_report["top_k"]),
                "scheduled_cache_line_bytes": int(
                    layer_report["complete_modelled_cold_bytes"]
                ),
                "dense_q4_bytes": int(layer_report["dense_q4_bytes"]),
                "fraction_of_dense_q4": float(
                    layer_report["fraction_of_dense_q4"]
                ),
            }
            if (
                worst_layer is None
                or candidate["fraction_of_dense_q4"]
                > worst_layer["fraction_of_dense_q4"]
            ):
                worst_layer = candidate
    assert worst_layer is not None
    worst_token = max(
        per_token,
        key=lambda item: float(item["fraction_of_dense_q4"]),
    )
    global_fraction = scheduled_total / dense_total
    passes = bool(
        global_fraction <= _MAXIMUM_PHYSICAL_TRAFFIC_FRACTION
        and float(worst_token["fraction_of_dense_q4"])
        <= _MAXIMUM_PHYSICAL_TRAFFIC_FRACTION
        and float(worst_layer["fraction_of_dense_q4"])
        <= _MAXIMUM_PHYSICAL_TRAFFIC_FRACTION
    )
    return {
        "accounting_version": "native_bitnet_dip_dual_layout_v2",
        "global": {
            "scheduled_cache_line_bytes": scheduled_total,
            "dense_q4_bytes": dense_total,
            "fraction_of_dense_q4": global_fraction,
        },
        "per_token": per_token,
        "layers": [
            {
                "layer": layer,
                "scheduled_cache_line_bytes": layer_scheduled[layer],
                "dense_q4_bytes": layer_dense[layer],
                "fraction_of_dense_q4": (
                    layer_scheduled[layer] / layer_dense[layer]
                ),
            }
            for layer in range(len(policies))
        ],
        "worst_token": {
            "token": int(worst_token["token"]),
            "sequence": int(worst_token["sequence"]),
            "prediction_position": int(
                worst_token["prediction_position"]
            ),
            "fraction_of_dense_q4": float(
                worst_token["fraction_of_dense_q4"]
            ),
        },
        "worst_layer": worst_layer,
        "maximum_fraction_of_dense_q4": (
            _MAXIMUM_PHYSICAL_TRAFFIC_FRACTION
        ),
        "passes_45_percent": passes,
        "measured_hardware_dram_bytes": False,
    }


def _candidate_recall(
    kernel: NativeBitNetDIPCPUKernel,
    *,
    layer: int,
    reference_top_k: int,
    hidden_bf16_bits: NDArray[np.uint16],
    diagnostics: NativeBitNetDIPTorchDiagnostics,
    predictions_per_sequence: int,
) -> dict[str, Any]:
    candidates = diagnostics.candidate_ids
    selected = diagnostics.selected_record_ids
    coordinates = diagnostics.input_coordinate_ids
    if candidates is None or selected is None or coordinates is None:
        raise RuntimeError("debug DIP diagnostics omitted route identities")
    counts = np.asarray(diagnostics.selected_counts, dtype=np.uint32)
    if counts.ndim != 2 or counts.shape[1] != predictions_per_sequence + 1:
        raise RuntimeError("debug selected-count shape is invalid")
    scored_counts = np.ascontiguousarray(
        counts[:, :predictions_per_sequence]
    ).reshape(-1)
    scored_hidden = np.ascontiguousarray(
        hidden_bf16_bits[:, :predictions_per_sequence, :]
    ).reshape(-1, kernel.hidden_size)
    scored_candidates = np.ascontiguousarray(
        candidates[:, :predictions_per_sequence, :]
    ).reshape(-1, kernel.policies[layer].candidate_count)
    scored_selected = np.ascontiguousarray(
        selected[:, :predictions_per_sequence, :]
    ).reshape(-1, kernel.policies[layer].maximum_top_k)
    scored_coordinates = np.ascontiguousarray(
        coordinates[:, :predictions_per_sequence, :]
    ).reshape(-1, kernel.policies[layer].input_coordinates)
    policy = kernel.policies[layer]
    if (
        np.any(scored_counts < policy.minimum_top_k)
        or np.any(scored_counts > policy.maximum_top_k)
        or np.any(scored_candidates >= kernel.intermediate_size)
        or np.any(scored_coordinates >= kernel.hidden_size)
    ):
        raise RuntimeError(f"debug route identities are invalid at layer {layer}")
    for values, width, name in (
        (scored_candidates, policy.candidate_count, "candidate"),
        (scored_coordinates, policy.input_coordinates, "coordinate"),
    ):
        if any(np.unique(row).size != width for row in values):
            raise RuntimeError(
                f"debug {name} identities contain duplicates at layer {layer}"
            )

    if (
        isinstance(reference_top_k, bool)
        or not isinstance(reference_top_k, int)
        or not 0 < reference_top_k <= kernel.intermediate_size
    ):
        raise ValueError("reference_top_k must be within intermediate width")
    teacher_width = max(reference_top_k, policy.maximum_top_k)
    teacher, positive_counts = (
        kernel.teacher_top_k_with_positive_counts_bf16_bits(
            layer,
            scored_hidden,
            top_k=teacher_width,
        )
    )
    positive_counts = np.asarray(positive_counts, dtype=np.uint32).reshape(-1)
    if (
        positive_counts.shape != scored_counts.shape
        or np.any(positive_counts > kernel.intermediate_size)
    ):
        raise RuntimeError(
            f"teacher positive-utility counts are invalid at layer {layer}"
        )
    primary_teacher = np.ascontiguousarray(
        teacher[:, :reference_top_k]
    )
    positive_target_counts = np.clip(
        positive_counts,
        policy.minimum_top_k,
        policy.maximum_top_k,
    ).astype(np.uint32)
    primary_candidate_hits = 0
    primary_target_total = 0
    primary_candidate_row_recall: list[float] = []
    selected_fixed_reference_hits = 0
    selected_fixed_reference_row_recall: list[float] = []
    secondary_candidate_hits = 0
    secondary_selected_hits = 0
    secondary_target_total = 0
    secondary_candidate_row_recall: list[float] = []
    secondary_selected_row_recall: list[float] = []
    selected_count_total = 0
    for row, raw_count in enumerate(scored_counts):
        count = int(raw_count)
        primary_targets = primary_teacher[row]
        positive_target_count = int(positive_target_counts[row])
        positive_targets = teacher[row, :positive_target_count]
        candidate_row = scored_candidates[row]
        selected_row = scored_selected[row, :count]
        if (
            np.any(selected_row >= kernel.intermediate_size)
            or np.unique(selected_row).size != count
            or not np.all(np.isin(selected_row, candidate_row))
        ):
            raise RuntimeError(
                f"debug selected identities are invalid at layer {layer}"
            )
        if np.any(scored_selected[row, count:] != _UINT32_SENTINEL):
            raise RuntimeError(
                f"debug selected padding is invalid at layer {layer}"
            )
        primary_candidate_count = int(
            np.count_nonzero(np.isin(primary_targets, candidate_row))
        )
        selected_fixed_count = int(
            np.count_nonzero(np.isin(primary_targets, selected_row))
        )
        secondary_candidate_count = int(
            np.count_nonzero(np.isin(positive_targets, candidate_row))
        )
        secondary_selected_count = int(
            np.count_nonzero(np.isin(positive_targets, selected_row))
        )
        primary_candidate_hits += primary_candidate_count
        primary_target_total += reference_top_k
        primary_candidate_row_recall.append(
            primary_candidate_count / reference_top_k
        )
        selected_fixed_reference_hits += selected_fixed_count
        selected_fixed_reference_row_recall.append(
            selected_fixed_count / reference_top_k
        )
        secondary_candidate_hits += secondary_candidate_count
        secondary_selected_hits += secondary_selected_count
        secondary_target_total += positive_target_count
        secondary_candidate_row_recall.append(
            secondary_candidate_count / positive_target_count
        )
        secondary_selected_row_recall.append(
            secondary_selected_count / positive_target_count
        )
        selected_count_total += count
    secondary = {
        "definition": (
            "canonical dense positive-utility count clipped to the frozen "
            "per-layer minimum_top_k and maximum_top_k"
        ),
        "minimum_top_k": policy.minimum_top_k,
        "maximum_top_k": policy.maximum_top_k,
        "raw_positive_count_sum": int(
            np.sum(positive_counts, dtype=np.uint64)
        ),
        "clipped_target_count_sum": int(
            np.sum(positive_target_counts, dtype=np.uint64)
        ),
        "target_records": secondary_target_total,
        "candidate_hits": secondary_candidate_hits,
        "candidate_micro_recall": (
            secondary_candidate_hits / secondary_target_total
        ),
        "candidate_mean_row_recall": float(
            np.mean(secondary_candidate_row_recall)
        ),
        "candidate_p05_row_recall": float(
            np.percentile(secondary_candidate_row_recall, 5)
        ),
        "candidate_minimum_row_recall": min(
            secondary_candidate_row_recall
        ),
        "selected_hits": secondary_selected_hits,
        "selected_micro_recall": (
            secondary_selected_hits / secondary_target_total
        ),
        "selected_mean_row_recall": float(
            np.mean(secondary_selected_row_recall)
        ),
        "selected_p05_row_recall": float(
            np.percentile(secondary_selected_row_recall, 5)
        ),
        "selected_minimum_row_recall": min(
            secondary_selected_row_recall
        ),
        "raw_positive_counts_sha256": _array_sha256(positive_counts),
        "clipped_target_counts_sha256": _array_sha256(
            positive_target_counts
        ),
    }
    return {
        "layer": layer,
        "rows": int(scored_counts.size),
        "reference_top_k": reference_top_k,
        "target_records": primary_target_total,
        "candidate_hits": primary_candidate_hits,
        "candidate_micro_recall": (
            primary_candidate_hits / primary_target_total
        ),
        "candidate_mean_row_recall": float(
            np.mean(primary_candidate_row_recall)
        ),
        "candidate_p05_row_recall": float(
            np.percentile(primary_candidate_row_recall, 5)
        ),
        "candidate_minimum_row_recall": min(
            primary_candidate_row_recall
        ),
        "adaptive_selected_count_sum": selected_count_total,
        "selected_fixed_reference_hits": selected_fixed_reference_hits,
        "selected_fixed_reference_clipped_micro_recall": (
            selected_fixed_reference_hits / primary_target_total
        ),
        "selected_fixed_reference_clipped_mean_row_recall": float(
            np.mean(selected_fixed_reference_row_recall)
        ),
        "selected_fixed_reference_clipped_minimum_row_recall": min(
            selected_fixed_reference_row_recall
        ),
        "secondary_teacher_positive_utility_recall_clipped_to_"
        "frozen_minimum_and_maximum_k": secondary,
        "live_input_bf16_sha256": _array_sha256(scored_hidden),
        "input_coordinate_ids_sha256": _array_sha256(scored_coordinates),
        "candidate_ids_sha256": _array_sha256(scored_candidates),
        "selected_counts_sha256": _array_sha256(scored_counts),
        "selected_record_ids_sha256": _array_sha256(scored_selected),
        "canonical_teacher_fixed_top_k_sha256": _array_sha256(
            primary_teacher
        ),
        "canonical_teacher_maximum_needed_top_k_sha256": _array_sha256(
            teacher
        ),
        "teacher_definition": (
            "exact native-BF16 dense utility ordering at the frozen fixed "
            "per-layer reference_top_k; adaptive selected K is not the "
            "candidate-recall denominator"
        ),
    }


def _aggregate_recall(
    layers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    targets = sum(int(layer["target_records"]) for layer in layers)
    candidate_hits = sum(int(layer["candidate_hits"]) for layer in layers)
    if targets <= 0:
        raise RuntimeError("canonical recall evidence contains no targets")
    candidate = candidate_hits / targets
    macro_mean = float(
        np.mean(
            [
                float(layer["candidate_mean_row_recall"])
                for layer in layers
            ]
        )
    )
    minimum_layer_mean = min(
        float(layer["candidate_mean_row_recall"]) for layer in layers
    )
    secondary_layers = [
        layer[
            "secondary_teacher_positive_utility_recall_clipped_to_"
            "frozen_minimum_and_maximum_k"
        ]
        for layer in layers
    ]
    secondary_targets = sum(
        int(layer["target_records"]) for layer in secondary_layers
    )
    secondary_candidate_hits = sum(
        int(layer["candidate_hits"]) for layer in secondary_layers
    )
    secondary_selected_hits = sum(
        int(layer["selected_hits"]) for layer in secondary_layers
    )
    secondary = {
        "target_records": secondary_targets,
        "candidate_hits": secondary_candidate_hits,
        "candidate_micro_recall": (
            secondary_candidate_hits / secondary_targets
        ),
        "candidate_macro_mean_layer_recall": float(
            np.mean(
                [
                    float(layer["candidate_mean_row_recall"])
                    for layer in secondary_layers
                ]
            )
        ),
        "selected_hits": secondary_selected_hits,
        "selected_micro_recall": (
            secondary_selected_hits / secondary_targets
        ),
        "selected_macro_mean_layer_recall": float(
            np.mean(
                [
                    float(layer["selected_mean_row_recall"])
                    for layer in secondary_layers
                ]
            )
        ),
    }
    return {
        "rows": sum(int(layer["rows"]) for layer in layers),
        "target_records": targets,
        "candidate_hits": candidate_hits,
        "candidate_micro_recall": candidate,
        "macro_mean_layer_recall": macro_mean,
        "candidate_minimum_layer_mean_recall": minimum_layer_mean,
        "secondary_teacher_positive_utility_recall_clipped_to_"
        "frozen_minimum_and_maximum_k": secondary,
        "minimum_candidate_recall": _MINIMUM_CANDIDATE_RECALL,
        "global_micro_passes_95_percent": (
            candidate >= _MINIMUM_CANDIDATE_RECALL
        ),
        "every_layer_mean_passes_95_percent": (
            minimum_layer_mean >= _MINIMUM_CANDIDATE_RECALL
        ),
        "passes_95_percent": (
            candidate >= _MINIMUM_CANDIDATE_RECALL
            and minimum_layer_mean >= _MINIMUM_CANDIDATE_RECALL
        ),
    }


def _python_native_layer_parity(
    *,
    artifact: Any,
    kernel: NativeBitNetDIPCPUKernel,
    layer: int,
    hidden_bf16_bits: NDArray[np.uint16],
    diagnostics: NativeBitNetDIPTorchDiagnostics,
    native_output_bf16_bits: NDArray[np.uint16],
) -> dict[str, Any]:
    """Compare one live row with the frozen NumPy reference implementation."""

    policy = kernel.policies[layer]
    source_bits = np.ascontiguousarray(
        hidden_bf16_bits.reshape(-1, kernel.hidden_size)[:1]
    )
    source = (
        np.asarray(source_bits, dtype=np.uint32) << np.uint32(16)
    ).view(np.float32)
    reference = NativeBitNetDIPLayer(
        artifact,
        layer,
        input_fraction=policy.input_coordinates / kernel.hidden_size,
        candidate_count=policy.candidate_count,
        top_k=policy.maximum_top_k,
        rms_audit_count=policy.rms_audit_count,
        energy_target=policy.energy_target,
        minimum_top_k=policy.minimum_top_k,
        maximum_top_k=policy.maximum_top_k,
        rms_estimator=policy.rms_estimator,
        rms_audit_strategy=(
            policy.rms_audit_strategy
            if policy.rms_audit_strategy != "none"
            else "hashed_tail"
        ),
    )(source)
    expected_output_bits = (
        np.ascontiguousarray(reference.output, dtype=np.float32).view(np.uint32)
        >> np.uint32(16)
    ).astype(np.uint16)
    actual_output_bits = np.ascontiguousarray(
        native_output_bf16_bits.reshape(-1, kernel.hidden_size)[:1]
    )
    candidate_ids = diagnostics.candidate_ids
    coordinate_ids = diagnostics.input_coordinate_ids
    selected_ids = diagnostics.selected_record_ids
    if candidate_ids is None or coordinate_ids is None or selected_ids is None:
        raise RuntimeError("native parity requires debug route identities")
    expected_selected = np.asarray(reference.selected_indices[:1]).copy()
    expected_selected[expected_selected < 0] = _UINT32_SENTINEL
    checks = {
        "output_bf16": np.array_equal(expected_output_bits, actual_output_bits),
        "input_coordinate_ids": np.array_equal(
            np.asarray(reference.input_indices[:1], dtype=np.uint32),
            np.asarray(coordinate_ids).reshape(
                -1, policy.input_coordinates
            )[:1],
        ),
        "candidate_ids": np.array_equal(
            np.asarray(reference.candidate_indices[:1], dtype=np.uint32),
            np.asarray(candidate_ids).reshape(
                -1, policy.candidate_count
            )[:1],
        ),
        "selected_counts": np.array_equal(
            np.asarray(reference.selected_counts[:1], dtype=np.uint32),
            np.asarray(diagnostics.selected_counts).reshape(-1)[:1],
        ),
        "selected_record_ids": np.array_equal(
            expected_selected.astype(np.uint32),
            np.asarray(selected_ids).reshape(
                -1, policy.maximum_top_k
            )[:1],
        ),
    }
    return {
        "layer": layer,
        "rows": 1,
        "checks": checks,
        "passed": all(checks.values()),
        "live_input_bf16_sha256": _array_sha256(source_bits),
        "native_output_bf16_sha256": _array_sha256(actual_output_bits),
        "python_output_bf16_sha256": _array_sha256(expected_output_bits),
    }


def _native_scored_schedule_bytes(
    calls: Sequence[Mapping[str, int]],
    *,
    layer_counts: Mapping[int, NDArray[np.uint32]],
    hidden_size: int,
    all_rows: int,
) -> dict[str, Any]:
    """Remove each sequence's context-only row from native byte counters."""

    packed_record_bytes = math.ceil(hidden_size / 5)
    scored_rows = sum(values.size for values in layer_counts.values()) // len(
        layer_counts
    )
    per_layer = []
    total = 0
    for layer, call in enumerate(calls):
        scheduled = int(call["scheduled_cache_line_bytes"])
        selected_total = int(call["selected_count_total"])
        fixed_total = scheduled - selected_total * packed_record_bytes
        if fixed_total < 0 or fixed_total % all_rows:
            raise RuntimeError("native DIP byte counters are internally invalid")
        fixed_per_row = fixed_total // all_rows
        scored_selected = int(
            np.sum(layer_counts[layer], dtype=np.uint64)
        )
        scored_bytes = (
            fixed_per_row * scored_rows
            + scored_selected * packed_record_bytes
        )
        total += scored_bytes
        per_layer.append(
            {
                "layer": layer,
                "scored_rows": scored_rows,
                "selected_count_total": scored_selected,
                "scheduled_cache_line_bytes": scored_bytes,
            }
        )
    return {
        "scored_rows_per_layer": scored_rows,
        "scheduled_cache_line_bytes": total,
        "layers": per_layer,
    }


def evaluate_native_bitnet_dip_native_causal(
    package: str | Path,
    coordinate_index: str | Path,
    dataset: str | Path,
    *,
    out: str | Path,
    reference_top_ks: Sequence[int],
    sequence_count: int = 8,
    predictions_per_sequence: int = 32,
    record_offset: int = 0,
    dataset_role: DatasetRole = "development",
    dense_library: str | Path | None = None,
    dip_library: str | Path | None = None,
    threads: int | None = None,
    expected_layer_count: int = 30,
    debug_recall: bool = True,
    verify_python_native_parity: bool = True,
) -> dict[str, Any]:
    """Evaluate a serialized native DIP policy at the live BF16 MLP boundary.

    ``dataset_role`` is recorded but does not alter the metric thresholds.
    Development and final confirmation callers therefore share one evaluator
    while retaining an explicit, fail-closed provenance boundary.
    """

    if (
        isinstance(sequence_count, bool)
        or not isinstance(sequence_count, int)
        or sequence_count <= 0
        or isinstance(predictions_per_sequence, bool)
        or not isinstance(predictions_per_sequence, int)
        or predictions_per_sequence <= 0
    ):
        raise ValueError("causal evidence counts must be positive integers")
    if (
        isinstance(record_offset, bool)
        or not isinstance(record_offset, int)
        or record_offset < 0
    ):
        raise ValueError("record_offset must be a non-negative integer")
    if dataset_role not in {"development", "final"}:
        raise ValueError("dataset_role must be development or final")
    if (
        isinstance(expected_layer_count, bool)
        or not isinstance(expected_layer_count, int)
        or expected_layer_count <= 0
    ):
        raise ValueError("expected_layer_count must be a positive integer")
    if verify_python_native_parity and not debug_recall:
        raise ValueError(
            "Python/native parity requires the untimed debug recall pass"
        )
    requested_reference_top_ks = tuple(reference_top_ks)
    if not requested_reference_top_ks or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        for value in requested_reference_top_ks
    ):
        raise ValueError(
            "reference_top_ks must contain positive integer values"
        )

    package_path = Path(package).resolve()
    index_path = Path(coordinate_index).resolve()
    dataset_path = Path(dataset).resolve()
    output_path = Path(out)
    records = _load_causal_records(
        dataset_path,
        offset=record_offset,
        count=sequence_count,
    )
    started = time.perf_counter()

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("native causal DIP evaluation requires torch") from exc

    required_tokens = predictions_per_sequence + 1
    encoded: list[list[int]] = []
    sequence_hashes: list[str] = []
    dense_seconds = 0.0
    sparse_seconds = 0.0
    debug_seconds = 0.0
    debug_report: dict[str, Any] = {
        "enabled": False,
        "timed": False,
        "reason": "debug_recall_disabled",
    }
    python_parity_report: dict[str, Any] = {
        "evaluated": False,
        "passed": False,
        "reason": "verification_not_requested",
    }

    with NativeBitNetRuntime(
        package_path,
        library=dense_library,
        threads=threads,
    ) as runtime:
        for sequence, record in enumerate(records):
            token_ids = (
                [int(value) for value in record["input_ids"]]
                if "input_ids" in record
                else runtime.encode(str(record.get("text", "")))
            )
            if len(token_ids) < required_tokens:
                raise ValueError(
                    f"selected record {sequence} has {len(token_ids)} tokens; "
                    f"required {required_tokens}"
                )
            selected = token_ids[:required_tokens]
            encoded.append(selected)
            sequence_hashes.append(sha256_json(selected))
        if len(set(map(tuple, encoded))) != sequence_count:
            unique_sequences = len(set(map(tuple, encoded)))
        else:
            unique_sequences = sequence_count
        input_ids = torch.tensor(encoded, dtype=torch.long)

        with NativeBitNetDIPCPUKernel(
            runtime.artifact_path,
            index_path,
            threads=(
                runtime.kernel.thread_count if threads is None else threads
            ),
            library=dip_library,
            expected_record_sha256=runtime.manifest["mlp"]["sha256"],
        ) as kernel:
            layer_count = len(runtime.model.model.layers)
            dimensions = {
                "layers": (
                    layer_count,
                    runtime.kernel.layer_count,
                    kernel.layer_count,
                    int(runtime.manifest["model"]["num_hidden_layers"]),
                ),
                "hidden": (
                    kernel.hidden_size,
                    runtime.kernel.hidden_size,
                    int(runtime.manifest["model"]["hidden_size"]),
                ),
                "intermediate": (
                    kernel.intermediate_size,
                    runtime.kernel.intermediate_size,
                    int(runtime.manifest["model"]["intermediate_size"]),
                ),
            }
            if any(len(set(values)) != 1 for values in dimensions.values()):
                raise RuntimeError(
                    "package, dense kernel, and DIP kernel dimensions differ"
                )
            if layer_count != expected_layer_count:
                raise RuntimeError(
                    f"native causal DIP evidence requires "
                    f"{expected_layer_count} layers; package has {layer_count}"
                )
            if (
                len(requested_reference_top_ks) != layer_count
                or any(
                    value > kernel.intermediate_size
                    for value in requested_reference_top_ks
                )
            ):
                raise ValueError(
                    "reference_top_ks must contain one in-width value per "
                    "MLP layer"
                )
            all_rows = sequence_count * required_tokens

            runtime.kernel.clear_metrics()
            dense_started = time.perf_counter()
            with torch.inference_mode():
                dense_result = runtime.model(
                    input_ids=input_ids,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            dense_seconds = time.perf_counter() - dense_started
            dense_calls = list(runtime.kernel.calls)
            if len(dense_calls) != layer_count:
                raise RuntimeError(
                    "dense native kernel did not execute once per MLP layer"
                )

            kernel.calls.clear()
            with substitute_native_bitnet_dip_kernel_mlps(
                runtime.model,
                kernel,
                debug_routes=False,
            ) as replacements:
                sparse_started = time.perf_counter()
                with torch.inference_mode():
                    sparse_result = runtime.model(
                        input_ids=input_ids,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                sparse_seconds = time.perf_counter() - sparse_started
                timed_diagnostics = {
                    layer: replacement.last_result
                    for layer, replacement in replacements.items()
                }
            timed_calls = list(kernel.calls)
            _validate_native_calls(
                timed_calls,
                policies=kernel.policies,
                rows=all_rows,
            )
            layer_counts: dict[int, NDArray[np.uint32]] = {}
            for layer in range(layer_count):
                diagnostics = timed_diagnostics[layer]
                if diagnostics is None:
                    raise RuntimeError(
                        f"native DIP layer {layer} emitted no diagnostics"
                    )
                counts = np.asarray(
                    diagnostics.selected_counts,
                    dtype=np.uint32,
                )
                if counts.shape != (
                    sequence_count,
                    required_tokens,
                ):
                    raise RuntimeError(
                        f"native DIP layer {layer} row shape is invalid"
                    )
                layer_counts[layer] = np.ascontiguousarray(
                    counts[:, :predictions_per_sequence]
                )

            (
                dense_logits,
                sparse_logits,
                dense_hidden,
                sparse_hidden,
                labels,
            ) = _prediction_views(dense_result, sparse_result, input_ids)
            logits = _logit_metrics(dense_logits, sparse_logits)
            hidden = _tensor_metrics(dense_hidden, sparse_hidden)
            dense_nll = functional.cross_entropy(
                dense_logits.reshape(-1, dense_logits.shape[-1]),
                labels.reshape(-1),
            )
            sparse_nll = functional.cross_entropy(
                sparse_logits.reshape(-1, sparse_logits.shape[-1]),
                labels.reshape(-1),
            )
            nll_delta = float((sparse_nll - dense_nll).item())
            quality_passed = bool(
                logits["mean_kl_divergence"]
                <= _QUALITY_THRESHOLDS["maximum_mean_kl_divergence"]
                and logits["top1_agreement"]
                >= _QUALITY_THRESHOLDS["minimum_top1_agreement"]
                and nll_delta
                <= _QUALITY_THRESHOLDS["maximum_nll_delta"]
                and hidden["relative_l2"]
                <= _QUALITY_THRESHOLDS[
                    "maximum_final_hidden_relative_l2"
                ]
            )
            quality = {
                "mean_kl_divergence": logits["mean_kl_divergence"],
                "top1_agreement": logits["top1_agreement"],
                "reference_nll": float(dense_nll.item()),
                "candidate_nll": float(sparse_nll.item()),
                "nll_delta": nll_delta,
                "final_hidden_relative_l2": hidden["relative_l2"],
                "logit_maximum_absolute_error": (
                    logits["maximum_absolute_error"]
                ),
                "final_hidden_maximum_absolute_error": (
                    hidden["maximum_absolute_error"]
                ),
                "passed": quality_passed,
            }

            selection = _selection_report(
                layer_counts,
                sequence_count=sequence_count,
                predictions_per_sequence=predictions_per_sequence,
                intermediate_size=kernel.intermediate_size,
            )
            active_passed = bool(
                selection["global"]["active_fraction"]
                <= _MAXIMUM_ACTIVE_FRACTION
            )
            traffic = _physical_traffic_report(
                hidden_size=kernel.hidden_size,
                intermediate_size=kernel.intermediate_size,
                policies=kernel.policies,
                schedules=selection["per_token_layer_k"],
                predictions_per_sequence=predictions_per_sequence,
            )
            native_scored = _native_scored_schedule_bytes(
                timed_calls,
                layer_counts=layer_counts,
                hidden_size=kernel.hidden_size,
                all_rows=all_rows,
            )
            for modelled, executed in zip(
                traffic["layers"],
                native_scored["layers"],
                strict=True,
            ):
                if (
                    int(modelled["layer"]) != int(executed["layer"])
                    or int(modelled["scheduled_cache_line_bytes"])
                    != int(executed["scheduled_cache_line_bytes"])
                ):
                    raise RuntimeError(
                        "native DIP scored byte counters differ from the "
                        "cache-line accounting model"
                    )

            if debug_recall:
                captured_inputs: dict[int, Any] = {}
                captured_outputs: dict[int, Any] = {}
                hooks = []
                debug_call_begin = len(kernel.calls)
                with substitute_native_bitnet_dip_kernel_mlps(
                    runtime.model,
                    kernel,
                    debug_routes=True,
                ) as debug_replacements:
                    for layer, replacement in debug_replacements.items():
                        def capture_input(
                            _module,
                            args,
                            *,
                            layer_index=layer,
                        ):
                            captured_inputs[layer_index] = (
                                args[0].detach().contiguous().clone()
                            )

                        def capture_output(
                            _module,
                            _args,
                            value,
                            *,
                            layer_index=layer,
                        ):
                            captured_outputs[layer_index] = (
                                value.detach().contiguous()[
                                    :1, :1, :
                                ].clone()
                            )

                        hooks.append(
                            replacement.register_forward_pre_hook(capture_input)
                        )
                        hooks.append(
                            replacement.register_forward_hook(capture_output)
                        )
                    try:
                        debug_started = time.perf_counter()
                        with torch.inference_mode():
                            debug_result = runtime.model(
                                input_ids=input_ids,
                                use_cache=False,
                                output_hidden_states=True,
                                return_dict=True,
                            )
                        debug_seconds = time.perf_counter() - debug_started
                    finally:
                        for hook in hooks:
                            hook.remove()
                    debug_diagnostics = {
                        layer: replacement.last_result
                        for layer, replacement in debug_replacements.items()
                    }
                debug_calls = list(kernel.calls[debug_call_begin:])
                _validate_native_calls(
                    debug_calls,
                    policies=kernel.policies,
                    rows=all_rows,
                )
                logits_equal = bool(
                    torch.equal(sparse_result.logits, debug_result.logits)
                )
                hidden_equal = bool(
                    torch.equal(
                        sparse_result.hidden_states[-1],
                        debug_result.hidden_states[-1],
                    )
                )
                if not logits_equal or not hidden_equal:
                    raise RuntimeError(
                        "untimed debug pass differs from timed sparse pass"
                    )
                if set(captured_inputs) != set(range(layer_count)):
                    raise RuntimeError(
                        "debug pass did not capture every live MLP input"
                    )
                layer_recall = []
                for layer in range(layer_count):
                    diagnostics = debug_diagnostics[layer]
                    if diagnostics is None:
                        raise RuntimeError(
                            f"debug DIP layer {layer} emitted no diagnostics"
                        )
                    hidden_bits = (
                        captured_inputs[layer]
                        .view(torch.uint16)
                        .cpu()
                        .numpy()
                    )
                    layer_recall.append(
                        _candidate_recall(
                            kernel,
                            layer=layer,
                            reference_top_k=(
                                requested_reference_top_ks[layer]
                            ),
                            hidden_bf16_bits=hidden_bits,
                            diagnostics=diagnostics,
                            predictions_per_sequence=(
                                predictions_per_sequence
                            ),
                        )
                    )
                recall_global = _aggregate_recall(layer_recall)
                debug_report = {
                    "enabled": True,
                    "timed": False,
                    "seconds": debug_seconds,
                    "timed_sparse_parity": {
                        "logits_exact": logits_equal,
                        "final_hidden_exact": hidden_equal,
                        "passed": logits_equal and hidden_equal,
                    },
                    "global": recall_global,
                    "layers": {
                        str(item["layer"]): item
                        for item in layer_recall
                    },
                }

                if verify_python_native_parity:
                    artifact = load_native_bitnet_artifact(
                        runtime.artifact_path
                    )
                    parity_layers = []
                    for layer in range(layer_count):
                        diagnostics = debug_diagnostics[layer]
                        assert diagnostics is not None
                        hidden_bits = (
                            captured_inputs[layer]
                            .view(torch.uint16)
                            .cpu()
                            .numpy()
                        )
                        output_bits = (
                            captured_outputs[layer]
                            .view(torch.uint16)
                            .cpu()
                            .numpy()
                        )
                        parity_layers.append(
                            _python_native_layer_parity(
                                artifact=artifact,
                                kernel=kernel,
                                layer=layer,
                                hidden_bf16_bits=hidden_bits,
                                diagnostics=diagnostics,
                                native_output_bf16_bits=output_bits,
                            )
                        )
                    parity_passed = all(
                        bool(item["passed"]) for item in parity_layers
                    )
                    if not parity_passed:
                        raise RuntimeError(
                            "live Python/native DIP parity failed"
                        )
                    python_parity_report = {
                        "evaluated": True,
                        "rows_per_layer": 1,
                        "all_layers": True,
                        "passed": parity_passed,
                        "layers": {
                            str(item["layer"]): item
                            for item in parity_layers
                        },
                    }

            if len(runtime.kernel.calls) != len(dense_calls):
                raise RuntimeError(
                    "dense native MLP kernel executed during DIP substitution; "
                    "dense fallback cannot be excluded"
                )
            artifacts = {
                "package_manifest": _file_descriptor(
                    package_path / "manifest.json"
                ),
                "base_record_artifact": _file_descriptor(
                    runtime.artifact_path
                ),
                "coordinate_index": _file_descriptor(index_path),
                "dense_kernel_library": _file_descriptor(
                    runtime.kernel.library_path
                ),
                "dip_kernel_library": _file_descriptor(
                    kernel.library_path
                ),
            }
            if (
                artifacts["base_record_artifact"]["sha256"]
                != kernel.record_sha256
                or artifacts["coordinate_index"]["sha256"]
                != kernel.index_sha256
            ):
                raise RuntimeError(
                    "native kernel artifact hashes changed during evaluation"
                )

            prediction_positions = (
                sequence_count * predictions_per_sequence
            )
            evidence_passed = _causal_evidence_passed(
                sequences=sequence_count,
                unique_sequences=unique_sequences,
                predictions_per_sequence=predictions_per_sequence,
                prediction_positions=prediction_positions,
                all_mlp_layers=True,
            )
            candidate_recall_passed = bool(
                debug_report.get("global", {}).get(
                    "passes_95_percent",
                    False,
                )
            )
            debug_parity_passed = bool(
                debug_report.get("timed_sparse_parity", {}).get(
                    "passed",
                    False,
                )
            )
            python_native_parity_passed = bool(
                python_parity_report.get("passed", False)
            )
            systems_evidence_passed = bool(
                debug_parity_passed and python_native_parity_passed
            )
            overall_gate_passed = bool(
                evidence_passed
                and quality_passed
                and active_passed
                and traffic["passes_45_percent"]
                and candidate_recall_passed
                and systems_evidence_passed
            )
            result = {
                "experiment": "native_bitnet_dip_native_causal",
                "dataset_role": dataset_role,
                "artifacts": artifacts,
                "dataset": {
                    "path": str(dataset_path),
                    "sha256": sha256_file(dataset_path),
                    "record_offset": record_offset,
                    "sequence_count": sequence_count,
                    "predictions_per_sequence": (
                        predictions_per_sequence
                    ),
                    "required_input_tokens_per_sequence": required_tokens,
                    "prediction_positions": prediction_positions,
                    "input_token_ids_sha256": sha256_json(encoded),
                    "sequence_token_ids_sha256": sequence_hashes,
                },
                "configuration": _policy_report(kernel.policies),
                "reference_top_ks": {
                    "values": list(requested_reference_top_ks),
                    "sha256": sha256_json(
                        list(requested_reference_top_ks)
                    ),
                    "role": (
                        "frozen_fixed_per_layer_candidate_recall_denominator"
                    ),
                },
                "execution": {
                    "input_boundary": "live_native_bf16",
                    "kernel": "native_cpu",
                    "device": "cpu",
                    "dense_fallback": False,
                    "all_mlp_layers_substituted": True,
                    "serialized_index_reloaded": True,
                    "python_native_parity_passed": (
                        python_native_parity_passed
                    ),
                    "timed_sparse_debug_routes": False,
                    "debug_pass_outside_timing": bool(debug_recall),
                    "dense_threads": runtime.kernel.thread_count,
                    "dip_threads": kernel.thread_count,
                    "dense_kernel_calls": len(dense_calls),
                    "timed_dip_kernel_calls": len(timed_calls),
                    "input_rows_per_layer": all_rows,
                    "scored_rows_per_layer": prediction_positions,
                },
                "quality": quality,
                "quality_passed": quality_passed,
                "thresholds": {
                    **_QUALITY_THRESHOLDS,
                    "maximum_mean_active_fraction": (
                        _MAXIMUM_ACTIVE_FRACTION
                    ),
                    "maximum_physical_cold_traffic_fraction": (
                        _MAXIMUM_PHYSICAL_TRAFFIC_FRACTION
                    ),
                    "minimum_candidate_recall": (
                        _MINIMUM_CANDIDATE_RECALL
                    ),
                },
                "selected_records": selection,
                "active_record_budget": {
                    **selection["global"],
                    "maximum_mean_active_fraction": (
                        _MAXIMUM_ACTIVE_FRACTION
                    ),
                    "passes_25_percent": active_passed,
                },
                "physical_cold_traffic": traffic,
                "native_kernel_traffic": {
                    "timed_input_includes_context_only_last_row": True,
                    "timed_all_input_rows": {
                        "rows_per_layer": all_rows,
                        "scheduled_cache_line_bytes": sum(
                            int(call["scheduled_cache_line_bytes"])
                            for call in timed_calls
                        ),
                    },
                    "scored_prediction_rows": native_scored,
                    "global_header_directory_bytes_are_modelled_in_"
                    "physical_cold_traffic": True,
                },
                "debug_recall": debug_report,
                "candidate_recall_passed": candidate_recall_passed,
                "python_native_parity": python_parity_report,
                "systems_evidence_passed": systems_evidence_passed,
                "scoring_protocol_valid": True,
                "prediction_protocol": (
                    "load N+1 tokens; score logits and final hidden on the "
                    "first N positions against next-token labels [1:]"
                ),
                "evidence_requirements": {
                    "minimum_unique_sequences": 8,
                    "minimum_predictions_per_sequence": 32,
                    "minimum_prediction_positions": 256,
                    "requires_all_mlp_layers": True,
                },
                "evidence_observed": {
                    "sequences": sequence_count,
                    "unique_sequences": unique_sequences,
                    "predictions_per_sequence": (
                        predictions_per_sequence
                    ),
                    "prediction_positions": prediction_positions,
                    "all_mlp_layers": True,
                    "layer_count": layer_count,
                    "expected_layer_count": expected_layer_count,
                    "layers_executed": list(range(layer_count)),
                },
                "evidence_passed": evidence_passed,
                "protocol_qualifying": (
                    evidence_passed and systems_evidence_passed
                ),
                "overall_gate_passed": overall_gate_passed,
                "timing": {
                    "dense_seconds": dense_seconds,
                    "timed_sparse_seconds": sparse_seconds,
                    "sparse_over_dense": (
                        sparse_seconds / dense_seconds
                        if dense_seconds > 0
                        else None
                    ),
                    "debug_seconds_outside_timing": debug_seconds,
                    "timed_native_dip_kernel_seconds": sum(
                        int(call["elapsed_ns"]) for call in timed_calls
                    )
                    / 1e9,
                    "elapsed_seconds": time.perf_counter() - started,
                    "latency_is_a_gate": False,
                },
                "decision": (
                    "freeze_policy_and_run_protected_final_confirmation"
                    if overall_gate_passed
                    and dataset_role == "development"
                    else (
                        "milestone_2_semantic_gate_passed"
                        if overall_gate_passed
                        else "milestone_2_semantic_gate_remains_blocked"
                    )
                ),
                "milestone_2_status": (
                    (
                        "passed"
                        if dataset_role == "final"
                        else "development_gate_passed_pending_final"
                    )
                    if overall_gate_passed
                    else "blocked"
                ),
            }

    result = _json_native(result)
    json.dumps(result, allow_nan=False)
    atomic_json(output_path, result)
    return result


__all__ = ["evaluate_native_bitnet_dip_native_causal"]
