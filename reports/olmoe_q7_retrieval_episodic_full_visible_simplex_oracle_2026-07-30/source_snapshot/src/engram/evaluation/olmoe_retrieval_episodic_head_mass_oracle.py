"""Train-only dynamic per-head episodic logit-mass capacity oracle.

This evaluator is intentionally confirmation-blind.  It reconstructs a frozen
finite set of per-head episodic-mass candidates from the authenticated
same-state shadow trace and measures their post-output-projection recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import engram.evaluation.olmoe_retrieval_episodic_residual_capacity as capacity
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.utils import atomic_json, sha256_file, sha256_json


_SCHEMA_VERSION = 1
_EXPECTED_CAPACITY_PROTOCOL_SHA256 = (
    "584302d17a3224cda1b61dfe1f62685497fa5a0dc335cfc0a074439456ee1606"
)
_EXPECTED_CAPACITY_RESULT_SHA256 = (
    "c636ad124d570f3675a36f0a23b276ba2e4cd4f5efc23dbf98cc10cd2cfd8e33"
)
_EXPECTED_CAPACITY_PARITY_SHA256 = (
    "56e4b730dc7580895e952a5746d105f5ca01ec36d83f6b37044c5f331061f8dd"
)
_EXPECTED_CAPACITY_MANIFEST_SHA256 = (
    "1f255a59a20089abe4d6805c625a119c167b71153bc21f5edbfcf0fd8050f461"
)
_PARITY_EXPERIMENT = "olmoe_q7_retrieval_episodic_head_mass_oracle_parity"
_PROTOCOL_EXPERIMENT = "olmoe_q7_retrieval_episodic_head_mass_oracle_protocol"
_RESULT_EXPERIMENT = "olmoe_q7_retrieval_episodic_head_mass_oracle_train_screen"
_PARITY_STATUS = "same_state_episodic_head_mass_trace_parity_passed"
_PROTOCOL_STATUS = "frozen_before_dynamic_head_mass_trace_execution"
_MASS_COPY_SYMBOL = "engram_olmoe_token_copy_last_episodic_mass_trace_v1"
_REQUIRED_SYMBOLS = (
    capacity._TRACE_OPEN_SYMBOL,
    capacity._TRACE_COPY_SYMBOL,
    _MASS_COPY_SYMBOL,
)
_RECORDS = capacity._RECORDS
_POSITIONS = capacity._POSITIONS
_READ_POSITIONS = capacity._READ_POSITIONS
_BLOCK_ENTRY_POSITIONS = capacity._BLOCK_ENTRY_POSITIONS
_LAYERS = capacity._LAYERS
_HIDDEN_SIZE = capacity._HIDDEN_SIZE
_QUERY_HEADS = 16
_HEAD_DIMENSION = 128
_BASE_POLICY = dict(capacity._BASE_POLICY)
_SHADOW_POLICY = dict(capacity._SHADOW_POLICY)
_EPISODIC_POLICY = dict(capacity._EPISODIC_POLICY)
_TRACE_KEYS = (
    "base_attention_output",
    "regular_component",
    "episodic_component",
    "regular_mass",
    "episodic_mass",
    "shadow_scheduled_mass",
    "base_projected",
    "target_residual",
)
_VALUE_TRACE_KEYS = (
    "base_attention_output",
    "regular_component",
    "episodic_component",
    "base_projected",
    "target_residual",
)
_MASS_TRACE_KEYS = (
    "regular_mass",
    "episodic_mass",
    "shadow_scheduled_mass",
)
_TIE_PRIORITY = (4, 3, 5, 2, 6, 1, 7, 0)
_GAMMA_ROWS = (
    {"code": 0, "gamma": 0.0, "beta_bits": None},
    {"code": 1, "gamma": 0.125, "beta_bits": "0xc0051592"},
    {"code": 2, "gamma": 0.25, "beta_bits": "0xbfb17218"},
    {"code": 3, "gamma": 0.5, "beta_bits": "0xbf317218"},
    {"code": 4, "gamma": 1.0, "beta_bits": "0x00000000"},
    {"code": 5, "gamma": 2.0, "beta_bits": "0x3f317218"},
    {"code": 6, "gamma": 4.0, "beta_bits": "0x3fb17218"},
    {"code": 7, "gamma": 8.0, "beta_bits": "0x40051592"},
)
_FIXED_COMBINED_TRAFFIC_BYTES = 714_866_688
_FIXED_DENSE_TRAFFIC_FRACTION = 0.33030523255813954
_FIXED_ATTENTION_STATE_BYTES = 10_534_912
_SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *capacity._SOURCE_FILES,
            "src/engram/evaluation/olmoe_retrieval_episodic_head_mass_oracle.py",
        )
    )
)


def _progress(message: str) -> None:
    print(
        f"[retrieval-episodic-head-mass-oracle] {message}",
        file=sys.stderr,
        flush=True,
    )


def _source_inventory() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    return {name: sha256_file(repository / name) for name in _SOURCE_FILES}


def _float32_from_bits(bits: str) -> np.float32:
    if not isinstance(bits, str) or len(bits) != 10 or not bits.startswith("0x"):
        raise ValueError("head-mass beta bits are invalid")
    return np.frombuffer(struct.pack(">I", int(bits[2:], 16)), dtype=">f4")[0].astype(
        np.float32
    )


def _float32_bits(value: float | np.floating[Any]) -> str:
    packed = struct.unpack(">I", struct.pack(">f", float(np.float32(value))))[0]
    return f"0x{packed:08x}"


def _gamma_table() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for expected_code, row in enumerate(_GAMMA_ROWS):
        if row["code"] != expected_code:
            raise AssertionError("head-mass gamma codes are not dense")
        if expected_code == 0:
            rows.append(
                {
                    **row,
                    "beta_float32": None,
                    "multiplier_float32": 0.0,
                    "endpoint": "exact_zero_episodic_mass",
                }
            )
            continue
        beta = _float32_from_bits(str(row["beta_bits"]))
        if _float32_bits(beta) != row["beta_bits"]:
            raise AssertionError("head-mass beta bit round trip changed")
        multiplier = np.float32(np.exp(beta))
        rows.append(
            {
                **row,
                "beta_float32": float(beta),
                "multiplier_float32": float(multiplier),
                "endpoint": None,
            }
        )
    return tuple(rows)


def _gamma_multipliers() -> np.ndarray:
    return np.asarray(
        [row["multiplier_float32"] for row in _gamma_table()],
        dtype=np.float32,
    )


def _validate_mass_inputs(
    regular_mass: np.ndarray,
    episodic_mass: np.ndarray,
    target_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    regular = np.ascontiguousarray(regular_mass, dtype=np.float32)
    episodic = np.ascontiguousarray(episodic_mass, dtype=np.float32)
    target = np.ascontiguousarray(target_mass, dtype=np.float32)
    if (
        regular.shape != episodic.shape
        or regular.shape != target.shape
        or not regular.size
        or not np.isfinite(regular).all()
        or not np.isfinite(episodic).all()
        or not np.isfinite(target).all()
        or np.any(regular <= np.float32(0.0))
        or np.any(episodic <= np.float32(0.0))
        or np.any(target < np.float32(0.0))
        or np.any(target > np.float32(1.0))
    ):
        raise ValueError("head-mass trace masses are invalid")
    total = regular + episodic
    if np.max(np.abs(total - np.float32(1.0))) > np.float32(2.0e-5):
        raise ValueError("head-mass regular and episodic masses do not partition one")
    return regular, episodic, target


def _candidate_masses(
    regular_mass: np.ndarray,
    episodic_mass: np.ndarray,
) -> np.ndarray:
    regular = np.ascontiguousarray(regular_mass, dtype=np.float32)
    episodic = np.ascontiguousarray(episodic_mass, dtype=np.float32)
    if (
        regular.shape != episodic.shape
        or not regular.size
        or not np.isfinite(regular).all()
        or not np.isfinite(episodic).all()
        or np.any(regular <= np.float32(0.0))
        or np.any(episodic <= np.float32(0.0))
    ):
        raise ValueError("head-mass partition masses are invalid")
    multipliers = _gamma_multipliers()
    denominator = regular[..., None] + episodic[..., None] * multipliers.reshape(
        (1,) * regular.ndim + (-1,)
    )
    if np.any(denominator <= np.float32(0.0)):
        raise ValueError("head-mass counterfactual denominator is invalid")
    masses = (
        episodic[..., None] * multipliers.reshape((1,) * regular.ndim + (-1,))
    ) / denominator
    masses[..., 0] = np.float32(0.0)
    masses[..., 4] = episodic
    if not np.isfinite(masses).all():
        raise ValueError("head-mass counterfactual masses are non-finite")
    return np.ascontiguousarray(masses, dtype=np.float32)


def _select_gamma_codes(
    regular_mass: np.ndarray,
    episodic_mass: np.ndarray,
    target_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    regular, episodic, target = _validate_mass_inputs(
        regular_mass,
        episodic_mass,
        target_mass,
    )
    candidates = _candidate_masses(regular, episodic)
    distances = np.abs(candidates - target[..., None])
    selected = np.full(target.shape, _TIE_PRIORITY[0], dtype=np.uint8)
    best = distances[..., _TIE_PRIORITY[0]].copy()
    for code in _TIE_PRIORITY[1:]:
        distance = distances[..., code]
        improve = distance < best
        selected[improve] = np.uint8(code)
        best[improve] = distance[improve]
    baseline = distances[..., 4]
    if np.any(best > baseline):
        raise AssertionError("head-mass selection regressed against gamma one")
    chosen = np.take_along_axis(
        candidates,
        selected[..., None].astype(np.int64),
        axis=-1,
    )[..., 0]
    return (
        np.ascontiguousarray(selected),
        np.ascontiguousarray(chosen, dtype=np.float32),
        np.ascontiguousarray(distances, dtype=np.float32),
    )


def _counterfactual_pre_wo(
    base_attention_output: np.ndarray,
    regular_component: np.ndarray,
    episodic_component: np.ndarray,
    regular_mass: np.ndarray,
    episodic_mass: np.ndarray,
    selected_codes: np.ndarray,
    *,
    query_heads: int | None = None,
) -> np.ndarray:
    if query_heads is None:
        query_heads = _QUERY_HEADS
    base = np.ascontiguousarray(base_attention_output, dtype=np.float32)
    regular = np.ascontiguousarray(regular_component, dtype=np.float32)
    episodic = np.ascontiguousarray(episodic_component, dtype=np.float32)
    codes = np.ascontiguousarray(selected_codes, dtype=np.uint8)
    regular_probability = np.ascontiguousarray(regular_mass, dtype=np.float32)
    episodic_probability = np.ascontiguousarray(episodic_mass, dtype=np.float32)
    if (
        base.shape != regular.shape
        or base.shape != episodic.shape
        or base.ndim < 1
        or base.shape[-1] % query_heads
        or codes.shape != base.shape[:-1] + (query_heads,)
        or regular_probability.shape != codes.shape
        or episodic_probability.shape != codes.shape
        or np.any(codes > 7)
        or not all(
            np.isfinite(array).all()
            for array in (
                base,
                regular,
                episodic,
                regular_probability,
                episodic_probability,
            )
        )
        or np.any(regular_probability <= np.float32(0.0))
        or np.any(episodic_probability <= np.float32(0.0))
    ):
        raise ValueError("head-mass counterfactual trace shapes are invalid")
    head_dimension = base.shape[-1] // query_heads
    head_shape = base.shape[:-1] + (query_heads, head_dimension)
    base_heads = base.reshape(head_shape)
    regular_heads = regular.reshape(head_shape)
    episodic_heads = episodic.reshape(head_shape)
    multiplier = _gamma_multipliers()[codes.astype(np.int64)]
    denominator = regular_probability + multiplier * episodic_probability
    if np.any(denominator <= np.float32(0.0)):
        raise ValueError("head-mass counterfactual output denominator is invalid")
    candidate = (regular_heads + multiplier[..., None] * episodic_heads) / denominator[
        ..., None
    ]
    gamma_one = codes == 4
    candidate[gamma_one] = base_heads[gamma_one]
    if not np.isfinite(candidate).all():
        raise ValueError("head-mass counterfactual output is non-finite")
    return np.ascontiguousarray(candidate.reshape(base.shape), dtype=np.float32)


def _project_counterfactual_delta(
    base_attention_output: np.ndarray,
    candidate_attention_output: np.ndarray,
    output_projection: np.ndarray,
) -> np.ndarray:
    base = np.ascontiguousarray(base_attention_output, dtype=np.float32)
    candidate = np.ascontiguousarray(candidate_attention_output, dtype=np.float32)
    weights = np.ascontiguousarray(output_projection, dtype=np.float32)
    if (
        base.shape != candidate.shape
        or base.ndim != 4
        or weights.shape != (base.shape[2], base.shape[3], base.shape[3])
        or not np.isfinite(weights).all()
    ):
        raise ValueError("head-mass output projection shapes are invalid")
    correction = np.empty_like(base)
    for layer in range(base.shape[2]):
        delta = candidate[:, :, layer, :] - base[:, :, layer, :]
        correction[:, :, layer, :] = (
            delta.reshape(-1, delta.shape[-1]) @ weights[layer].T
        ).reshape(delta.shape)
    if not np.isfinite(correction).all():
        raise ValueError("head-mass projected correction is non-finite")
    return np.ascontiguousarray(correction, dtype=np.float32)


def _energy_metric(target_squared: float, error_squared: float) -> dict[str, float]:
    if (
        not math.isfinite(target_squared)
        or not math.isfinite(error_squared)
        or target_squared <= 0.0
        or error_squared < 0.0
    ):
        raise ValueError("head-mass recovery energy is invalid")
    return {
        "target_squared_frobenius": target_squared,
        "error_squared_frobenius": error_squared,
        "error_ratio": math.sqrt(error_squared / target_squared),
        "recovery": 1.0 - error_squared / target_squared,
    }


def _oracle_metrics(
    target_residual: np.ndarray,
    correction: np.ndarray,
    selected_distances: np.ndarray,
    baseline_distances: np.ndarray,
) -> dict[str, Any]:
    target = np.ascontiguousarray(target_residual, dtype=np.float64)
    predicted = np.ascontiguousarray(correction, dtype=np.float64)
    selected = np.ascontiguousarray(selected_distances, dtype=np.float64)
    baseline = np.ascontiguousarray(baseline_distances, dtype=np.float64)
    expected = (_RECORDS, len(_READ_POSITIONS), _LAYERS, _HIDDEN_SIZE)
    mass_shape = (_RECORDS, len(_READ_POSITIONS), _LAYERS, _QUERY_HEADS)
    if (
        target.shape != expected
        or predicted.shape != expected
        or selected.shape != mass_shape
        or baseline.shape != mass_shape
        or not np.isfinite(target).all()
        or not np.isfinite(predicted).all()
        or not np.isfinite(selected).all()
        or not np.isfinite(baseline).all()
        or np.any(selected > baseline)
    ):
        raise ValueError("head-mass oracle metric inputs are invalid")
    error = target - predicted

    def metric(target_view: np.ndarray, error_view: np.ndarray) -> dict[str, float]:
        return _energy_metric(
            float(np.sum(target_view * target_view, dtype=np.float64)),
            float(np.sum(error_view * error_view, dtype=np.float64)),
        )

    global_metric = metric(target, error)
    sequences = [
        {"record_index": index, **metric(target[index], error[index])}
        for index in range(_RECORDS)
    ]
    layers = [
        {"layer": layer, **metric(target[:, :, layer], error[:, :, layer])}
        for layer in range(_LAYERS)
    ]
    blocks = []
    for position in _BLOCK_ENTRY_POSITIONS:
        offset = _READ_POSITIONS.index(position)
        blocks.append(
            {
                "position": position,
                **metric(target[:, offset], error[:, offset]),
            }
        )
    positive_layers = sum(row["recovery"] > 0.0 for row in layers)
    gate = {
        "finite": True,
        "global_recovery_at_least_0_50": global_metric["recovery"] >= 0.50,
        "every_sequence_recovery_at_least_0_25": all(
            row["recovery"] >= 0.25 for row in sequences
        ),
        "every_block_entry_recovery_at_least_0_25": all(
            row["recovery"] >= 0.25 for row in blocks
        ),
        "at_least_12_of_16_layers_positive_recovery": positive_layers >= 12,
        "selected_mass_never_worse_than_gamma_one": bool(np.all(selected <= baseline)),
    }
    gate["passed"] = all(gate.values())
    return {
        "global": global_metric,
        "heldout_sequences": sequences,
        "layers": layers,
        "block_entry_positions": blocks,
        "positive_recovery_layer_count": positive_layers,
        "mass_selection": {
            "coordinates": int(selected.size),
            "selected_distance_mean": float(selected.mean()),
            "selected_distance_max": float(selected.max()),
            "gamma_one_distance_mean": float(baseline.mean()),
            "gamma_one_distance_max": float(baseline.max()),
        },
        "gate": gate,
        "passed": bool(gate["passed"]),
    }


def _require_mass_trace_symbols(path: Path) -> None:
    import ctypes

    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise ValueError("head-mass trace library could not be loaded") from exc
    missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(library, name)]
    if missing:
        raise ValueError(
            "head-mass trace library lacks required symbols: " + ", ".join(missing)
        )


def _validate_capacity_failure(value: Any, protocol_path: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("head-mass residual-capacity result is invalid")
    decision = value.get("decision")
    capacity_result = value.get("capacity")
    post = value.get("post_run_authentication")
    trace_manifest = value.get("trace_manifest")
    if (
        value.get("schema_version") != capacity._SCHEMA_VERSION
        or value.get("experiment") != capacity._RESULT_EXPERIMENT
        or value.get("status") != "train_residual_capacity_gate_failed"
        or value.get("protocol")
        != {
            "path": str(protocol_path),
            "sha256": _EXPECTED_CAPACITY_PROTOCOL_SHA256,
        }
        or value.get("confirmation_split_opened") is not False
        or not isinstance(decision, Mapping)
        or decision.get("capacity_gate_passed") is not False
        or decision.get("train_only_predictor_fit_authorized") is not False
        or decision.get("native_integration_authorized") is not False
        or decision.get("development_authorized") is not False
        or decision.get("confirmation_authorized") is not False
        or not isinstance(capacity_result, Mapping)
        or capacity_result.get("passed") is not False
        or capacity_result.get("selected_rank") != 8
        or capacity_result.get("selection_role")
        != "best_failed_rank_for_diagnostic_replay"
        or capacity_result.get("selected_metric_replay", {}).get("passed") is not True
        or not isinstance(post, Mapping)
        or not post
        or not all(check is True for check in post.values())
        or not isinstance(trace_manifest, Mapping)
        or trace_manifest.get("sha256") != _EXPECTED_CAPACITY_MANIFEST_SHA256
        or trace_manifest.get("shard_count") != _RECORDS
    ):
        raise ValueError("head-mass residual-capacity failure contract changed")
    ranks = capacity_result.get("rank_outcomes")
    if (
        not isinstance(ranks, Mapping)
        or set(ranks) != {"2", "4", "8"}
        or capacity_result.get("rank_order") != [2, 4, 8]
    ):
        raise ValueError("head-mass residual-capacity rank evidence changed")
    expected_global = {
        2: 0.40046952208141817,
        4: 0.4286862133341903,
        8: 0.469252618228868,
    }
    for rank in (2, 4, 8):
        row = ranks[str(rank)]
        gate = row.get("gate")
        if (
            row.get("rank") != rank
            or row.get("global", {}).get("recovery") != expected_global[rank]
            or not isinstance(gate, Mapping)
            or gate.get("global_recovery_at_least_0_50") is not False
            or gate.get("every_sequence_recovery_at_least_0_25") is not True
            or gate.get("every_block_entry_recovery_at_least_0_25") is not True
            or gate.get("at_least_12_of_16_layers_positive_recovery") is not True
            or gate.get("passed") is not False
        ):
            raise ValueError("head-mass residual-capacity gate evidence changed")
    return {
        "status": value["status"],
        "selected_rank": 8,
        "selected_global_recovery": expected_global[8],
        "failure_condition": "global_recovery_below_0_50_only",
        "confirmation_split_opened": False,
    }


def _authenticate_capacity_inputs(
    *,
    capacity_protocol: str | Path,
    capacity_protocol_sha256: str,
    capacity_result: str | Path,
    capacity_result_sha256: str,
    trace_library: str | Path,
    trace_library_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if capacity_protocol_sha256.lower() != _EXPECTED_CAPACITY_PROTOCOL_SHA256:
        raise ValueError("head-mass capacity protocol root changed")
    if capacity_result_sha256.lower() != _EXPECTED_CAPACITY_RESULT_SHA256:
        raise ValueError("head-mass capacity result root changed")
    protocol_path = capacity.bias.episodic._checked_file(
        capacity_protocol,
        capacity_protocol_sha256,
        "head-mass capacity protocol",
    )
    result_path = capacity.bias.episodic._checked_file(
        capacity_result,
        capacity_result_sha256,
        "head-mass capacity result",
    )
    library_path = capacity.bias.episodic._checked_file(
        trace_library,
        trace_library_sha256,
        "head-mass trace library",
    )
    _require_mass_trace_symbols(library_path)
    raw_protocol = capacity.bias.rank.retrieval._read_json(
        protocol_path,
        "head-mass capacity protocol",
    )
    frozen_inventory = raw_protocol.get("source_sha256")
    if not isinstance(frozen_inventory, Mapping) or not frozen_inventory:
        raise ValueError("head-mass frozen capacity source inventory is invalid")
    original_source_inventory = capacity._source_inventory
    capacity._source_inventory = lambda: dict(frozen_inventory)
    try:
        context, training, frozen = capacity._authenticate_protocol(
            protocol_path,
            capacity_protocol_sha256,
        )
    finally:
        capacity._source_inventory = original_source_inventory
    result_value = capacity.bias.rank.retrieval._read_json(
        result_path,
        "head-mass capacity result",
    )
    failure = _validate_capacity_failure(result_value, protocol_path)
    parity_binding = frozen.get("trace_parity")
    manifest_binding = result_value.get("trace_manifest")
    if (
        not isinstance(parity_binding, Mapping)
        or parity_binding.get("sha256") != _EXPECTED_CAPACITY_PARITY_SHA256
        or not isinstance(manifest_binding, Mapping)
    ):
        raise ValueError("head-mass historical trace bindings changed")
    manifest_path = capacity.bias.episodic._checked_file(
        manifest_binding.get("path"),
        manifest_binding.get("sha256"),
        "head-mass historical trace manifest",
    )
    context = dict(context)
    context.update(
        {
            "historical_capacity_protocol_path": protocol_path,
            "historical_capacity_protocol_sha256": capacity_protocol_sha256.lower(),
            "historical_capacity_result_path": result_path,
            "historical_capacity_result_sha256": capacity_result_sha256.lower(),
            "historical_capacity_result": result_value,
            "historical_capacity_manifest_path": manifest_path,
            "historical_capacity_manifest_sha256": manifest_binding["sha256"],
            "mass_trace_library_path": library_path,
            "mass_trace_library_sha256": trace_library_sha256.lower(),
            # Existing execution helpers consume these generic trace bindings.
            "trace_library_path": library_path,
            "trace_library_sha256": trace_library_sha256.lower(),
        }
    )
    return context, training, frozen, failure


def _open_mass_trace_runtime(
    context: Mapping[str, Any],
    beta: float = 0.0,
) -> Any:
    return OLMoENativeTokenRuntime(
        context["config_path"],
        context["non_mlp_path"],
        context["q7_path"],
        context["mass_trace_library_path"],
        threads=capacity.bias._THREADS,
        **_BASE_POLICY,
        episodic_policy=_EPISODIC_POLICY,
        episodic_head_mask=capacity.bias._all_ones_mask(),
        episodic_logit_bias=beta,
        shadow_attention_policy=_SHADOW_POLICY,
    )


def _validate_runtime_route(runtime: Any, *, beta: float = 0.0) -> None:
    if (
        runtime.position != 0
        or not runtime.attention_metrics_available
        or not runtime.episodic_metrics_available
        or runtime.episodic_policy != _EPISODIC_POLICY
        or runtime.episodic_open_abi != "shadow_trace_v1"
        or runtime.shadow_trace_available is not True
        or runtime.episodic_mass_trace_available is not True
        or runtime.shadow_attention_policy != _SHADOW_POLICY
        or not capacity.bias.rank.fixed._runtime_mask_matches(
            runtime,
            capacity.bias._all_ones_mask(),
        )
        or capacity.bias._float32_bits(runtime.episodic_logit_bias)
        != capacity.bias._float32_bits(beta)
    ):
        raise ValueError("head-mass native runtime route changed")


def _trace_array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in _TRACE_KEYS:
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _trace_summary(
    arrays: Mapping[str, np.ndarray],
    positions: Sequence[int],
) -> dict[str, Any]:
    if list(positions) != list(_READ_POSITIONS) or set(arrays) != set(_TRACE_KEYS):
        raise ValueError("head-mass trace support changed")
    tensor_sha256: dict[str, str] = {}
    for name in _VALUE_TRACE_KEYS:
        array = arrays[name]
        expected = (len(_READ_POSITIONS), _LAYERS, _HIDDEN_SIZE)
        if (
            not isinstance(array, np.ndarray)
            or array.shape != expected
            or array.dtype != np.float32
            or not array.flags.c_contiguous
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"head-mass {name} trace is invalid")
        tensor_sha256[name] = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    for name in _MASS_TRACE_KEYS:
        array = arrays[name]
        expected = (len(_READ_POSITIONS), _LAYERS, _QUERY_HEADS)
        if (
            not isinstance(array, np.ndarray)
            or array.shape != expected
            or array.dtype != np.float32
            or not array.flags.c_contiguous
            or not np.isfinite(array).all()
            or np.any(array < np.float32(0.0))
            or np.any(array > np.float32(1.0))
        ):
            raise ValueError(f"head-mass {name} trace is invalid")
        tensor_sha256[name] = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    regular_mass = arrays["regular_mass"]
    episodic_mass = arrays["episodic_mass"]
    shadow_mass = arrays["shadow_scheduled_mass"]
    if (
        np.max(np.abs(regular_mass + episodic_mass - np.float32(1.0)))
        > np.float32(2.0e-5)
        or np.any(regular_mass <= np.float32(0.0))
        or np.any(episodic_mass <= np.float32(0.0))
        or not np.any(shadow_mass > np.float32(0.0))
    ):
        raise ValueError("head-mass probability traces are invalid")
    reconstructed = arrays["regular_component"] + arrays["episodic_component"]
    component_max_abs = float(
        np.max(np.abs(reconstructed - arrays["base_attention_output"]))
    )
    if component_max_abs > 5.0e-5:
        raise ValueError("head-mass components do not reconstruct beta zero")
    selected, chosen_mass, distances = _select_gamma_codes(
        regular_mass,
        episodic_mass,
        shadow_mass,
    )
    gamma_one_codes = np.full(selected.shape, 4, dtype=np.uint8)
    gamma_one = _counterfactual_pre_wo(
        arrays["base_attention_output"],
        arrays["regular_component"],
        arrays["episodic_component"],
        regular_mass,
        episodic_mass,
        gamma_one_codes,
    )
    if not np.array_equal(gamma_one, arrays["base_attention_output"]):
        raise ValueError("head-mass gamma-one anchor changed")
    return {
        "positions": list(positions),
        "value_shape": [len(_READ_POSITIONS), _LAYERS, _HIDDEN_SIZE],
        "mass_shape": [len(_READ_POSITIONS), _LAYERS, _QUERY_HEADS],
        "dtype": "float32",
        "layout": "position_layer_coordinate",
        "tensor_sha256": tensor_sha256,
        "trace_sha256": _trace_array_digest(arrays),
        "component_reconstruction_max_abs": component_max_abs,
        "mass_partition_max_abs": float(
            np.max(np.abs(regular_mass + episodic_mass - np.float32(1.0)))
        ),
        "shadow_scheduled_mass_nonzero_coordinates": int(np.count_nonzero(shadow_mass)),
        "selected_code_stream_sha256": hashlib.sha256(
            selected.tobytes(order="C")
        ).hexdigest(),
        "selected_mass_sha256": hashlib.sha256(
            chosen_mass.tobytes(order="C")
        ).hexdigest(),
        "selected_mass_distance_mean": float(
            np.take_along_axis(
                distances,
                selected[..., None].astype(np.int64),
                axis=-1,
            ).mean()
        ),
        "gamma_one_exact_anchor": True,
    }


class _MassTraceCaptureRuntime:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._positions: list[int] = []
        self._rows: dict[str, list[np.ndarray]] = {name: [] for name in _TRACE_KEYS}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    @property
    def position(self) -> int:
        return int(self._runtime.position)

    def forward_episodic(
        self,
        token_ids: Sequence[int],
        write_slots: Sequence[int],
        read_spans: Sequence[int],
    ) -> Any:
        result = self._runtime.forward_episodic(token_ids, write_slots, read_spans)
        if len(read_spans) != 1:
            raise ValueError("head-mass trace requires one-token calls")
        if int(read_spans[0]) >= 0:
            old_input, base_projected, target_residual = (
                self._runtime.last_shadow_trace()
            )
            del old_input
            mass = self._runtime.last_episodic_mass_trace()
            row = {
                "base_attention_output": mass.base_attention_output,
                "regular_component": mass.regular_component,
                "episodic_component": mass.episodic_component,
                "regular_mass": mass.regular_mass,
                "episodic_mass": mass.episodic_mass,
                "shadow_scheduled_mass": mass.shadow_scheduled_mass,
                "base_projected": base_projected,
                "target_residual": target_residual,
            }
            self._positions.append(self.position - 1)
            for name in _TRACE_KEYS:
                self._rows[name].append(
                    np.ascontiguousarray(row[name], dtype=np.float32)
                )
        return result

    def captured(self) -> tuple[dict[str, np.ndarray], list[int]]:
        if len(self._positions) != len(_READ_POSITIONS):
            raise ValueError("head-mass trace capture is incomplete")
        arrays = {
            name: np.ascontiguousarray(np.stack(rows), dtype=np.float32)
            for name, rows in self._rows.items()
        }
        _trace_summary(arrays, self._positions)
        return arrays, list(self._positions)

    def reset(self) -> None:
        self._runtime.reset()
        self._positions.clear()
        for rows in self._rows.values():
            rows.clear()

    def close(self) -> None:
        self._runtime.close()


def _execute_record(
    runtime: _MassTraceCaptureRuntime,
    *,
    record: Mapping[str, Any],
    context: Mapping[str, Any],
    schedule: Mapping[str, Any],
    resource: Mapping[str, Any],
    progress_label: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[int]]:
    _logits, _hidden, evidence = capacity.bias.episodic._execute_episodic_record(
        runtime,
        record=record,
        context=context,
        schedule=schedule,
        resource=resource,
        progress_label=progress_label,
    )
    arrays, positions = runtime.captured()
    return evidence, arrays, positions


def _base_post_authentication(
    context: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
) -> dict[str, bool]:
    checks = capacity._base_post_authentication(context, checkpoint=checkpoint)
    checks.update(
        {
            "historical_capacity_protocol": (
                sha256_file(context["historical_capacity_protocol_path"])
                == context["historical_capacity_protocol_sha256"]
            ),
            "historical_capacity_result": (
                sha256_file(context["historical_capacity_result_path"])
                == context["historical_capacity_result_sha256"]
            ),
            "historical_capacity_manifest": (
                sha256_file(context["historical_capacity_manifest_path"])
                == context["historical_capacity_manifest_sha256"]
            ),
            "mass_trace_library": (
                sha256_file(context["mass_trace_library_path"])
                == context["mass_trace_library_sha256"]
            ),
        }
    )
    return checks


def _historical_trace_arrays(
    context: Mapping[str, Any],
    *,
    record_index: int,
) -> dict[str, np.ndarray]:
    result = context["historical_capacity_result"]
    descriptors = result["trace_manifest"]["shards"]
    descriptor = descriptors[record_index]
    path = (
        Path(context["historical_capacity_manifest_path"]).parent / descriptor["file"]
    )
    return capacity._validate_trace_shard(path, descriptor)


def _projected_trace_hashes(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in ("base_projected", "target_residual"):
        array = np.ascontiguousarray(arrays[name])
        expected = (len(_READ_POSITIONS), _LAYERS, _HIDDEN_SIZE)
        if (
            array.shape != expected
            or array.dtype != np.float32
            or not np.isfinite(array).all()
        ):
            raise ValueError("head-mass projected trace is invalid")
        hashes[name] = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    return hashes


def _direct_first_read_qualification(
    *,
    context: Mapping[str, Any],
    record: Mapping[str, Any],
    schedule: Mapping[str, Any],
    beta_zero_arrays: Mapping[str, np.ndarray],
    output_projection: np.ndarray,
    runtime_factory: Callable[[Mapping[str, Any], float], Any] = (
        _open_mass_trace_runtime
    ),
) -> dict[str, Any]:
    first_read_position = _READ_POSITIONS[0]
    base = {
        name: beta_zero_arrays[name][0:1]
        for name in (
            "base_attention_output",
            "regular_component",
            "episodic_component",
            "regular_mass",
            "episodic_mass",
            "base_projected",
        )
    }
    rows: list[dict[str, Any]] = []
    for gamma in _gamma_table():
        code = int(gamma["code"])
        if code in (0, 4):
            continue
        beta = float(gamma["beta_float32"])
        _progress(f"qualifying direct first-read gamma code {code}")
        runtime = runtime_factory(context, beta)
        try:
            _validate_runtime_route(runtime, beta=beta)
            for position in range(first_read_position + 1):
                schedule_row = schedule["rows"][position]
                runtime.forward_episodic(
                    [int(record["input_ids"][position])],
                    [int(schedule_row["write_slot"])],
                    [int(schedule_row["read_span"])],
                )
            direct = runtime.last_episodic_mass_trace()
            _old_input, direct_projected, _target_residual = runtime.last_shadow_trace()
        finally:
            runtime.close()
        codes = np.full(
            (1, _LAYERS, _QUERY_HEADS),
            code,
            dtype=np.uint8,
        )
        analytic_output = _counterfactual_pre_wo(
            base["base_attention_output"],
            base["regular_component"],
            base["episodic_component"],
            base["regular_mass"],
            base["episodic_mass"],
            codes,
        )[0]
        analytic_mass = _candidate_masses(
            base["regular_mass"],
            base["episodic_mass"],
        )[0, ..., code]
        analytic_correction = _project_counterfactual_delta(
            base["base_attention_output"][None, ...],
            analytic_output[None, None, ...],
            output_projection,
        )[0, 0]
        analytic_projected = base["base_projected"][0] + analytic_correction
        output_error = np.abs(analytic_output - direct.base_attention_output)
        mass_error = np.abs(analytic_mass - direct.episodic_mass)
        projected_error = np.abs(analytic_projected - direct_projected)
        layer_zero_output_max_abs = float(np.max(output_error[0]))
        layer_zero_mass_max_abs = float(np.max(mass_error[0]))
        layer_zero_projected_max_abs = float(np.max(projected_error[0]))
        downstream_output_max_abs = float(
            np.max(output_error[1:]) if _LAYERS > 1 else 0.0
        )
        downstream_mass_max_abs = float(np.max(mass_error[1:]) if _LAYERS > 1 else 0.0)
        downstream_projected_max_abs = float(
            np.max(projected_error[1:]) if _LAYERS > 1 else 0.0
        )
        rows.append(
            {
                "code": code,
                "gamma": gamma["gamma"],
                "beta_bits": gamma["beta_bits"],
                "layer_zero_output_max_abs": layer_zero_output_max_abs,
                "layer_zero_episodic_mass_max_abs": layer_zero_mass_max_abs,
                "layer_zero_projected_output_max_abs": (layer_zero_projected_max_abs),
                "downstream_causal_output_max_abs": downstream_output_max_abs,
                "downstream_causal_episodic_mass_max_abs": (downstream_mass_max_abs),
                "downstream_causal_projected_output_max_abs": (
                    downstream_projected_max_abs
                ),
                "passed": (
                    layer_zero_output_max_abs <= 5.0e-5
                    and layer_zero_mass_max_abs <= 5.0e-6
                    and layer_zero_projected_max_abs <= 5.0e-4
                ),
            }
        )
        _progress(
            "gamma code "
            f"{code} layer zero: pre-Wo {layer_zero_output_max_abs:.9g}, "
            f"mass {layer_zero_mass_max_abs:.9g}, "
            f"projected {layer_zero_projected_max_abs:.9g}"
        )
    if len(rows) != 6 or not all(row["passed"] for row in rows):
        raise ValueError(
            "head-mass direct first-read qualification failed: "
            + json.dumps(rows, sort_keys=True, separators=(",", ":"))
        )
    return {
        "position": first_read_position,
        "qualified_layer": 0,
        "qualification_scope": (
            "layer zero only: it has the same input state in beta-zero and "
            "direct biased execution; later layers are causal diagnostics"
        ),
        "shared_streaming_attention_kernel_grid_unit_tested": True,
        "downstream_layers_are_nonqualifying_causal_diagnostics": True,
        "rows": rows,
        "layer_zero_output_max_abs_tolerance": 5.0e-5,
        "layer_zero_episodic_mass_max_abs_tolerance": 5.0e-6,
        "layer_zero_projected_output_max_abs_tolerance": 5.0e-4,
        "passed": True,
    }


def _run_trace_parity(
    *,
    context: Mapping[str, Any],
    frozen: Mapping[str, Any],
    runtime_factory: Callable[[Mapping[str, Any], float], Any] = (
        _open_mass_trace_runtime
    ),
) -> dict[str, Any]:
    record = context["train_records"][0]
    schedule = capacity.bias.rank.fixed._derive_schedule(
        record["input_ids"],
        frozen["schedule_contract"]["tokenizer_fact_anchor_ids"],
    )
    raw = runtime_factory(context, 0.0)
    trace = _MassTraceCaptureRuntime(raw)
    try:
        _validate_runtime_route(raw, beta=0.0)
        first, arrays, positions = _execute_record(
            trace,
            record=record,
            context=context,
            schedule=schedule,
            resource=frozen["fixed_K256_arm"]["resource_contract"],
            progress_label="head-mass parity first",
        )
        first_summary = _trace_summary(arrays, positions)
        trace.reset()
        replay, replay_arrays, replay_positions = _execute_record(
            trace,
            record=record,
            context=context,
            schedule=schedule,
            resource=frozen["fixed_K256_arm"]["resource_contract"],
            progress_label="head-mass parity reset",
        )
        replay_summary = _trace_summary(replay_arrays, replay_positions)
    finally:
        trace.close()
    historical = context["historical_k256_evidence"][0]
    historical_arrays = _historical_trace_arrays(context, record_index=0)
    historical_tensor_hashes = _projected_trace_hashes(historical_arrays)
    new_tensor_hashes = _projected_trace_hashes(arrays)
    output_projection, projection_hashes = _load_output_projections(context)
    qualification = _direct_first_read_qualification(
        context=context,
        record=record,
        schedule=schedule,
        beta_zero_arrays=arrays,
        output_projection=output_projection,
        runtime_factory=runtime_factory,
    )
    checks = {
        "historical_base_outputs_counters_and_loss_exact": (
            capacity._evidence_exact(first, historical)
        ),
        "reset_outputs_counters_and_loss_exact": capacity._evidence_exact(
            first,
            replay,
        ),
        "reset_trace_exact": first_summary == replay_summary,
        "historical_projected_trace_exact": (
            historical_tensor_hashes == new_tensor_hashes
        ),
        "direct_first_read_layer_zero_gamma_grid_qualified": qualification["passed"],
        "gamma_one_exact_anchor": first_summary["gamma_one_exact_anchor"],
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise ValueError("head-mass real-model trace parity failed")
    return {
        "record_index": 0,
        "schedule_rows_sha256": schedule["rows_sha256"],
        "first_output_evidence_sha256": sha256_json(capacity._without_elapsed(first)),
        "reset_output_evidence_sha256": sha256_json(capacity._without_elapsed(replay)),
        "first_trace": first_summary,
        "reset_trace": replay_summary,
        "historical_tensor_sha256": historical_tensor_hashes,
        "output_projection_tensor_sha256": projection_hashes,
        "direct_first_read_qualification": qualification,
        "checks": checks,
        "native_sequence_forwards": 8,
        "native_token_steps": 2 * _POSITIONS + 6 * (_READ_POSITIONS[0] + 1),
        "passed": True,
    }


def generate_trace_parity_report(
    *,
    capacity_protocol: str | Path,
    capacity_protocol_sha256: str,
    capacity_result: str | Path,
    capacity_result_sha256: str,
    trace_library: str | Path,
    trace_library_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    output = capacity.bias.rank.retrieval._new_output(
        out,
        "head-mass trace parity report",
    )
    context, _training, frozen, failure = _authenticate_capacity_inputs(
        capacity_protocol=capacity_protocol,
        capacity_protocol_sha256=capacity_protocol_sha256,
        capacity_result=capacity_result,
        capacity_result_sha256=capacity_result_sha256,
        trace_library=trace_library,
        trace_library_sha256=trace_library_sha256,
    )
    _progress("running same-state mass trace/reset and frozen-grid qualification")
    parity = _run_trace_parity(context=context, frozen=frozen)
    post = _base_post_authentication(
        context,
        checkpoint=frozen["training_checkpoint"],
    )
    if not post or not all(post.values()):
        raise ValueError("head-mass parity post-run authentication failed")
    report = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _PARITY_EXPERIMENT,
        "status": _PARITY_STATUS,
        "capacity_protocol": {
            "path": str(context["historical_capacity_protocol_path"]),
            "sha256": context["historical_capacity_protocol_sha256"],
        },
        "capacity_result": {
            "path": str(context["historical_capacity_result_path"]),
            "sha256": context["historical_capacity_result_sha256"],
            "authenticated_failure": failure,
        },
        "capacity_trace_manifest": {
            "path": str(context["historical_capacity_manifest_path"]),
            "sha256": context["historical_capacity_manifest_sha256"],
        },
        "trace_library": {
            "path": str(context["mass_trace_library_path"]),
            "sha256": context["mass_trace_library_sha256"],
            "required_symbols": list(_REQUIRED_SYMBOLS),
        },
        "base_policy": dict(_BASE_POLICY),
        "shadow_policy": dict(_SHADOW_POLICY),
        "episodic_policy": dict(_EPISODIC_POLICY),
        "gamma_table": [dict(row) for row in _gamma_table()],
        "tie_priority": list(_TIE_PRIORITY),
        "scope": {
            "split": "train",
            "record_index": 0,
            "positions": _POSITIONS,
            "trace_positions": list(_READ_POSITIONS),
            "same_state_capacity_only": True,
            "development_outcomes_used": False,
            "confirmation_split_opened": False,
        },
        "parity": parity,
        "source_sha256": _source_inventory(),
        "post_run_authentication": post,
        "confirmation_split_opened": False,
    }
    atomic_json(output, report)
    return report


def _validate_parity_report(
    *,
    path: str | Path,
    expected_sha256: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    source = capacity.bias.episodic._checked_file(
        path,
        expected_sha256,
        "head-mass parity report",
    )
    value = capacity.bias.rank.retrieval._read_json(
        source,
        "head-mass parity report",
    )
    parity = value.get("parity")
    checks = parity.get("checks") if isinstance(parity, Mapping) else None
    first_trace = parity.get("first_trace") if isinstance(parity, Mapping) else None
    first_tensor_hashes = (
        first_trace.get("tensor_sha256") if isinstance(first_trace, Mapping) else None
    )
    projection_hashes = (
        parity.get("output_projection_tensor_sha256")
        if isinstance(parity, Mapping)
        else None
    )
    qualification = (
        parity.get("direct_first_read_qualification")
        if isinstance(parity, Mapping)
        else None
    )
    qualification_rows = (
        qualification.get("rows") if isinstance(qualification, Mapping) else None
    )
    expected_qualification = [
        row for row in _gamma_table() if int(row["code"]) not in (0, 4)
    ]
    qualification_valid = (
        isinstance(qualification, Mapping)
        and qualification.get("position") == _READ_POSITIONS[0]
        and qualification.get("qualified_layer") == 0
        and qualification.get("qualification_scope")
        == (
            "layer zero only: it has the same input state in beta-zero and "
            "direct biased execution; later layers are causal diagnostics"
        )
        and qualification.get("shared_streaming_attention_kernel_grid_unit_tested")
        is True
        and qualification.get("downstream_layers_are_nonqualifying_causal_diagnostics")
        is True
        and qualification.get("layer_zero_output_max_abs_tolerance") == 5.0e-5
        and qualification.get("layer_zero_episodic_mass_max_abs_tolerance") == 5.0e-6
        and qualification.get("layer_zero_projected_output_max_abs_tolerance") == 5.0e-4
        and qualification.get("passed") is True
        and isinstance(qualification_rows, list)
        and len(qualification_rows) == len(expected_qualification)
        and all(isinstance(row, Mapping) for row in qualification_rows)
        and all(
            observed.get("code") == expected["code"]
            and observed.get("gamma") == expected["gamma"]
            and observed.get("beta_bits") == expected["beta_bits"]
            and observed.get("passed") is True
            and isinstance(
                observed.get("layer_zero_output_max_abs"),
                (int, float),
            )
            and 0.0
            <= observed["layer_zero_output_max_abs"]
            <= qualification["layer_zero_output_max_abs_tolerance"]
            and isinstance(
                observed.get("layer_zero_episodic_mass_max_abs"),
                (int, float),
            )
            and 0.0
            <= observed["layer_zero_episodic_mass_max_abs"]
            <= qualification["layer_zero_episodic_mass_max_abs_tolerance"]
            and isinstance(
                observed.get("layer_zero_projected_output_max_abs"),
                (int, float),
            )
            and 0.0
            <= observed["layer_zero_projected_output_max_abs"]
            <= qualification["layer_zero_projected_output_max_abs_tolerance"]
            and all(
                isinstance(observed.get(name), (int, float))
                and math.isfinite(float(observed[name]))
                and observed[name] >= 0.0
                for name in (
                    "downstream_causal_output_max_abs",
                    "downstream_causal_episodic_mass_max_abs",
                    "downstream_causal_projected_output_max_abs",
                )
            )
            for observed, expected in zip(
                qualification_rows,
                expected_qualification,
                strict=True,
            )
        )
    )
    expected_checks = {
        "historical_base_outputs_counters_and_loss_exact",
        "reset_outputs_counters_and_loss_exact",
        "reset_trace_exact",
        "historical_projected_trace_exact",
        "direct_first_read_layer_zero_gamma_grid_qualified",
        "gamma_one_exact_anchor",
        "passed",
    }
    parity_valid = (
        isinstance(parity, Mapping)
        and parity.get("passed") is True
        and isinstance(checks, Mapping)
        and set(checks) == expected_checks
        and all(check is True for check in checks.values())
        and parity.get("native_sequence_forwards") == 8
        and parity.get("native_token_steps")
        == 2 * _POSITIONS + 6 * (_READ_POSITIONS[0] + 1)
        and parity.get("first_trace") == parity.get("reset_trace")
        and isinstance(first_trace, Mapping)
        and first_trace.get("gamma_one_exact_anchor") is True
        and isinstance(first_tensor_hashes, Mapping)
        and parity.get("historical_tensor_sha256")
        == {
            name: first_tensor_hashes.get(name)
            for name in ("base_projected", "target_residual")
        }
        and isinstance(projection_hashes, Mapping)
        and set(projection_hashes)
        == {f"model.layers.{layer}.self_attn.o_proj.weight" for layer in range(_LAYERS)}
        and all(
            capacity.bias.rank.retrieval._is_sha256(digest)
            for digest in projection_hashes.values()
        )
        and qualification_valid
    )
    if (
        value.get("schema_version") != _SCHEMA_VERSION
        or value.get("experiment") != _PARITY_EXPERIMENT
        or value.get("status") != _PARITY_STATUS
        or value.get("capacity_protocol", {}).get("sha256")
        != _EXPECTED_CAPACITY_PROTOCOL_SHA256
        or value.get("capacity_result", {}).get("sha256")
        != _EXPECTED_CAPACITY_RESULT_SHA256
        or value.get("capacity_trace_manifest", {}).get("sha256")
        != _EXPECTED_CAPACITY_MANIFEST_SHA256
        or value.get("trace_library")
        != {
            "path": str(context["mass_trace_library_path"]),
            "sha256": context["mass_trace_library_sha256"],
            "required_symbols": list(_REQUIRED_SYMBOLS),
        }
        or value.get("gamma_table") != [dict(row) for row in _gamma_table()]
        or value.get("tie_priority") != list(_TIE_PRIORITY)
        or value.get("base_policy") != _BASE_POLICY
        or value.get("shadow_policy") != _SHADOW_POLICY
        or value.get("episodic_policy") != _EPISODIC_POLICY
        or value.get("scope")
        != {
            "split": "train",
            "record_index": 0,
            "positions": _POSITIONS,
            "trace_positions": list(_READ_POSITIONS),
            "same_state_capacity_only": True,
            "development_outcomes_used": False,
            "confirmation_split_opened": False,
        }
        or not parity_valid
        or value.get("source_sha256") != _source_inventory()
        or value.get("confirmation_split_opened") is not False
        or not isinstance(value.get("post_run_authentication"), Mapping)
        or not all(check is True for check in value["post_run_authentication"].values())
    ):
        raise ValueError("head-mass parity report contract changed")
    return {"path": str(source), "sha256": expected_sha256.lower(), "report": value}


def _build_protocol(
    *,
    context: Mapping[str, Any],
    training: Mapping[str, Any],
    frozen_capacity: Mapping[str, Any],
    failure: Mapping[str, Any],
    parity: Mapping[str, Any],
) -> dict[str, Any]:
    schedules = [
        capacity.bias.rank.fixed._derive_schedule(
            record["input_ids"],
            frozen_capacity["schedule_contract"]["tokenizer_fact_anchor_ids"],
        )
        for record in context["train_records"]
    ]
    base_protocol = frozen_capacity
    return {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _PROTOCOL_EXPERIMENT,
        "status": _PROTOCOL_STATUS,
        "capacity_protocol": {
            "path": str(context["historical_capacity_protocol_path"]),
            "sha256": context["historical_capacity_protocol_sha256"],
        },
        "capacity_result": {
            "path": str(context["historical_capacity_result_path"]),
            "sha256": context["historical_capacity_result_sha256"],
            "authenticated_failure": dict(failure),
        },
        "capacity_trace_manifest": {
            "path": str(context["historical_capacity_manifest_path"]),
            "sha256": context["historical_capacity_manifest_sha256"],
        },
        "trace_library": {
            "path": str(context["mass_trace_library_path"]),
            "sha256": context["mass_trace_library_sha256"],
            "required_symbols": list(_REQUIRED_SYMBOLS),
        },
        "trace_parity": dict(parity),
        "training_checkpoint": dict(base_protocol["training_checkpoint"]),
        "training_checkpoint_payload_sha256": sha256_json(training),
        "package": dict(base_protocol["package"]),
        "corpus": dict(base_protocol["corpus"]),
        "source_model": dict(base_protocol["source_model"]),
        "libraries": dict(base_protocol["libraries"]),
        "base_choice": dict(base_protocol["base_choice"]),
        "fixed_K256_arm": dict(base_protocol["fixed_K256_arm"]),
        "policies": {
            "base": dict(_BASE_POLICY),
            "shadow": dict(_SHADOW_POLICY),
            "episodic": dict(_EPISODIC_POLICY),
            "mask": "all_ones_K256",
            "trajectory_beta_float32": 0.0,
            "trajectory_beta_bits": "0x00000000",
        },
        "trace_schema": {
            "value_native_step_shape": [_LAYERS, _HIDDEN_SIZE],
            "mass_native_step_shape": [_LAYERS, _QUERY_HEADS],
            "value_stored_shape": [
                len(_READ_POSITIONS),
                _LAYERS,
                _HIDDEN_SIZE,
            ],
            "mass_stored_shape": [
                len(_READ_POSITIONS),
                _LAYERS,
                _QUERY_HEADS,
            ],
            "dtype": "float32",
            "layout": "position_layer_coordinate",
            "keys": list(_TRACE_KEYS),
            "support_rule": "capture iff schedule read_span >= 0",
            "positions": list(_READ_POSITIONS),
            "block_entry_positions": list(_BLOCK_ENTRY_POSITIONS),
            "W128_exact_full_context_only_for_position_horizon": _POSITIONS,
            "unbounded_context_claim": False,
            "storage": "safetensors",
            "pickle_permitted": False,
        },
        "schedule_contract": {
            "records": _RECORDS,
            "positions_per_record": _POSITIONS,
            "tokenizer_fact_anchor_ids": {
                label: list(values)
                for label, values in frozen_capacity["schedule_contract"][
                    "tokenizer_fact_anchor_ids"
                ].items()
            },
            "per_record_rows_sha256": [
                schedule["rows_sha256"] for schedule in schedules
            ],
            "read_positions": list(_READ_POSITIONS),
            "read_rows_per_record": len(_READ_POSITIONS),
            "coordinates": (_RECORDS * len(_READ_POSITIONS) * _LAYERS * _QUERY_HEADS),
        },
        "oracle_method": {
            "single_arm": True,
            "gamma_table": [dict(row) for row in _gamma_table()],
            "tie_priority": list(_TIE_PRIORITY),
            "target": (
                "same-state W128 attention probability mass on the exact "
                "eight scheduled source positions"
            ),
            "selection": (
                "independently minimize absolute target-mass error at each "
                "record/read-row/layer/head coordinate"
            ),
            "counterfactual": (
                "(R+gamma*E)/(m_regular+gamma*m_episodic), with gamma-one "
                "anchored to exact beta-zero pre-Wo output"
            ),
            "counterfactual_qualification": (
                "the shared StreamingAttention kernel is unit-tested over the "
                "full gamma grid; direct real-model qualification uses layer "
                "zero at the first read because only that layer has an "
                "identical input in beta-zero and biased executions"
            ),
            "direct_downstream_layers": (
                "diagnostic only because earlier biased layers causally "
                "change their input state"
            ),
            "gamma_zero": (
                "exact zero episodic mass while retaining the K256 "
                "duplicate-suppressed regular candidate set; not ordinary W16"
            ),
            "output_metric": (
                "project selected pre-Wo delta through authenticated BF16 "
                "o_proj and compare against exact native W128-minus-K256 "
                "projected residual"
            ),
            "energy_aggregation": "squared Frobenius energy before ratios",
            "answer_cross_entropy_gate": False,
            "counterfactual_updates_hidden_or_cache": False,
        },
        "progression_gate": {
            "finite": True,
            "minimum_global_recovery": 0.50,
            "minimum_every_sequence_recovery": 0.25,
            "minimum_every_block_entry_position_recovery": 0.25,
            "minimum_positive_recovery_layers": 12,
            "selected_mass_never_worse_than_gamma_one": True,
            "deterministic_metric_and_code_replay_required": True,
        },
        "resource_contract": {
            "fixed_combined_attention_and_episodic_traffic_bytes": (
                _FIXED_COMBINED_TRAFFIC_BYTES
            ),
            "fixed_fraction_of_dense_full_context_KV": (_FIXED_DENSE_TRAFFIC_FRACTION),
            "fixed_attention_state_bytes": _FIXED_ATTENTION_STATE_BYTES,
            "gamma_zero_earns_read_savings": False,
            "oracle_shadow_trace_and_projection_evaluator_only": True,
            "predictor_weights_features_and_execution_not_counted": True,
        },
        "scope": {
            "split": "train",
            "same_state_capacity_evidence_only": True,
            "learned_predictor": False,
            "causal_rollout": False,
            "semantic_or_M3_pass": False,
            "development_outcomes_used": False,
            "confirmation_file_access_permitted": False,
        },
        "authorized_next_step_on_pass": (
            "freeze a train-only causal predictor for the selected 3-bit "
            "gamma codes, including predictor traffic and rolled-out native "
            "state updates"
        ),
        "failure_scope": (
            "this exact fixed-K256 independent-head scheduled-source-mass "
            "matching grid only"
        ),
        "authenticated_confirmation_descriptor": dict(
            base_protocol["authenticated_confirmation_descriptor"]
        ),
        "source_sha256": _source_inventory(),
        "confirmation_split_opened": False,
    }


def freeze_head_mass_protocol(
    *,
    capacity_protocol: str | Path,
    capacity_protocol_sha256: str,
    capacity_result: str | Path,
    capacity_result_sha256: str,
    trace_library: str | Path,
    trace_library_sha256: str,
    parity_report: str | Path,
    parity_report_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    output = capacity.bias.rank.retrieval._new_output(
        out,
        "head-mass protocol",
    )
    context, training, frozen, failure = _authenticate_capacity_inputs(
        capacity_protocol=capacity_protocol,
        capacity_protocol_sha256=capacity_protocol_sha256,
        capacity_result=capacity_result,
        capacity_result_sha256=capacity_result_sha256,
        trace_library=trace_library,
        trace_library_sha256=trace_library_sha256,
    )
    parity = _validate_parity_report(
        path=parity_report,
        expected_sha256=parity_report_sha256,
        context=context,
    )
    protocol = _build_protocol(
        context=context,
        training=training,
        frozen_capacity=frozen,
        failure=failure,
        parity=parity,
    )
    atomic_json(output, protocol)
    return {"path": str(output), "sha256": sha256_file(output), "protocol": protocol}


def _authenticate_protocol(
    protocol: str | Path,
    protocol_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = capacity.bias.episodic._checked_file(
        protocol,
        protocol_sha256,
        "head-mass protocol",
    )
    value = capacity.bias.rank.retrieval._read_json(
        source,
        "head-mass protocol",
    )
    bindings = (
        value.get("capacity_protocol"),
        value.get("capacity_result"),
        value.get("trace_library"),
        value.get("trace_parity"),
    )
    if not all(isinstance(item, Mapping) for item in bindings):
        raise ValueError("head-mass protocol bindings are invalid")
    capacity_binding, result_binding, library_binding, parity_binding = bindings
    context, training, frozen, failure = _authenticate_capacity_inputs(
        capacity_protocol=capacity_binding.get("path"),
        capacity_protocol_sha256=capacity_binding.get("sha256"),
        capacity_result=result_binding.get("path"),
        capacity_result_sha256=result_binding.get("sha256"),
        trace_library=library_binding.get("path"),
        trace_library_sha256=library_binding.get("sha256"),
    )
    parity = _validate_parity_report(
        path=parity_binding.get("path"),
        expected_sha256=parity_binding.get("sha256"),
        context=context,
    )
    expected = _build_protocol(
        context=context,
        training=training,
        frozen_capacity=frozen,
        failure=failure,
        parity=parity,
    )
    if value != expected:
        raise ValueError("head-mass frozen protocol changed")
    context = dict(context)
    context.update(
        {
            "head_mass_protocol_path": source,
            "head_mass_protocol_sha256": protocol_sha256.lower(),
            "head_mass_protocol": expected,
            "head_mass_parity_path": Path(parity["path"]).resolve(),
            "head_mass_parity_sha256": parity["sha256"],
        }
    )
    return context, training, expected


def _prepare_shard_directory(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    if requested.is_symlink():
        raise ValueError("head-mass shard directory is invalid")
    path = requested.resolve()
    if path.exists() and not path.is_dir():
        raise ValueError("head-mass shard directory is invalid")
    if path.exists() and any(path.iterdir()):
        raise ValueError("head-mass shard directory is not empty")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_trace_shard(
    directory: Path,
    *,
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    positions: Sequence[int],
    source_record_sha256: str,
    output_sha256: str,
    reset_output_sha256: str,
    reset_trace_sha256: str,
) -> dict[str, Any]:
    summary = _trace_summary(arrays, positions)
    filename = f"train-{int(record['record_index']):02d}.safetensors"
    path = directory / filename
    if path.exists() or path.is_symlink():
        raise ValueError("head-mass trace shard already exists")
    try:
        from safetensors.numpy import save_file
    except ImportError as exc:  # pragma: no cover - required dependency
        raise RuntimeError("head-mass shards require safetensors") from exc
    payload = {
        name: np.ascontiguousarray(arrays[name], dtype=np.float32)
        for name in _TRACE_KEYS
    }
    payload["positions"] = np.asarray(positions, dtype=np.int64)
    temporary = directory / f".{filename}.tmp-{os.getpid()}"
    save_file(payload, str(temporary))
    temporary.replace(path)
    descriptor = {
        "record_index": int(record["record_index"]),
        "record_id": record["record_id"],
        "file": filename,
        "file_sha256": sha256_file(path),
        "format": "safetensors",
        "keys": [*_TRACE_KEYS, "positions"],
        "value_shape": [len(_READ_POSITIONS), _LAYERS, _HIDDEN_SIZE],
        "mass_shape": [len(_READ_POSITIONS), _LAYERS, _QUERY_HEADS],
        "dtype": "float32",
        "positions": list(_READ_POSITIONS),
        "tensor_sha256": summary["tensor_sha256"],
        "trace_sha256": summary["trace_sha256"],
        "reset_trace_sha256": reset_trace_sha256,
        "source_record_sha256": source_record_sha256,
        "output_evidence_sha256": output_sha256,
        "reset_output_evidence_sha256": reset_output_sha256,
        "selected_code_stream_sha256": summary["selected_code_stream_sha256"],
    }
    _validate_trace_shard(path, descriptor)
    return descriptor


def _validate_trace_shard(
    path: str | Path,
    descriptor: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError("head-mass trace shard descriptor is invalid")
    source = requested.resolve()
    if (
        not source.is_file()
        or source.name != descriptor.get("file")
        or sha256_file(source) != descriptor.get("file_sha256")
        or descriptor.get("format") != "safetensors"
        or descriptor.get("keys") != [*_TRACE_KEYS, "positions"]
        or descriptor.get("value_shape")
        != [len(_READ_POSITIONS), _LAYERS, _HIDDEN_SIZE]
        or descriptor.get("mass_shape") != [len(_READ_POSITIONS), _LAYERS, _QUERY_HEADS]
        or descriptor.get("dtype") != "float32"
        or descriptor.get("positions") != list(_READ_POSITIONS)
        or set(descriptor.get("tensor_sha256", {})) != set(_TRACE_KEYS)
        or not all(
            capacity.bias.rank.retrieval._is_sha256(digest)
            for digest in descriptor.get("tensor_sha256", {}).values()
        )
        or not all(
            capacity.bias.rank.retrieval._is_sha256(descriptor.get(name))
            for name in (
                "file_sha256",
                "trace_sha256",
                "reset_trace_sha256",
                "source_record_sha256",
                "output_evidence_sha256",
                "reset_output_evidence_sha256",
                "selected_code_stream_sha256",
            )
        )
    ):
        raise ValueError("head-mass trace shard descriptor is invalid")
    try:
        from safetensors import safe_open
        from safetensors.numpy import load_file
    except ImportError as exc:  # pragma: no cover - required dependency
        raise RuntimeError("head-mass shards require safetensors") from exc
    with safe_open(source, framework="numpy") as handle:
        if sorted(handle.keys()) != sorted([*_TRACE_KEYS, "positions"]):
            raise ValueError("head-mass trace shard keys changed")
    loaded = load_file(source)
    positions = loaded["positions"]
    arrays = {name: np.ascontiguousarray(loaded[name]) for name in _TRACE_KEYS}
    if (
        positions.dtype != np.int64
        or positions.shape != (len(_READ_POSITIONS),)
        or positions.tolist() != list(_READ_POSITIONS)
    ):
        raise ValueError("head-mass trace shard positions changed")
    summary = _trace_summary(arrays, positions.tolist())
    if (
        summary["trace_sha256"] != descriptor.get("trace_sha256")
        or summary["tensor_sha256"] != descriptor.get("tensor_sha256")
        or summary["selected_code_stream_sha256"]
        != descriptor.get("selected_code_stream_sha256")
    ):
        raise ValueError("head-mass trace shard tensor hash changed")
    return arrays


def _load_output_projections(
    context: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, str]]:
    source = Path(context["non_mlp_path"]).expanduser()
    if source.is_symlink():
        raise ValueError("head-mass output projection source is invalid")
    source = source.resolve()
    if not source.is_file() or source.suffix != ".safetensors":
        raise ValueError("head-mass output projection source is invalid")
    names = [
        f"model.layers.{layer}.self_attn.o_proj.weight" for layer in range(_LAYERS)
    ]
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - required dependency
        raise RuntimeError(
            "head-mass output projections require torch and safetensors"
        ) from exc
    rows = []
    hashes: dict[str, str] = {}
    with safe_open(source, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        if not set(names).issubset(available):
            raise ValueError("head-mass output projection tensor is missing")
        for name in names:
            tensor = handle.get_tensor(name)
            if (
                tuple(tensor.shape) != (_HIDDEN_SIZE, _HIDDEN_SIZE)
                or tensor.dtype != torch.bfloat16
            ):
                raise ValueError("head-mass output projection tensor is invalid")
            array = np.ascontiguousarray(
                tensor.detach().to(dtype=torch.float32).numpy(),
                dtype=np.float32,
            )
            if not np.isfinite(array).all():
                raise ValueError("head-mass output projection tensor is invalid")
            rows.append(array)
            hashes[name] = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    return np.ascontiguousarray(np.stack(rows), dtype=np.float32), hashes


def _run_oracle_from_arrays(
    arrays: Mapping[str, np.ndarray],
    output_projection: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    expected_keys = set(_TRACE_KEYS)
    if set(arrays) != expected_keys:
        raise ValueError("head-mass stacked trace keys changed")
    selected_codes, selected_mass, distances = _select_gamma_codes(
        arrays["regular_mass"],
        arrays["episodic_mass"],
        arrays["shadow_scheduled_mass"],
    )
    candidate = _counterfactual_pre_wo(
        arrays["base_attention_output"],
        arrays["regular_component"],
        arrays["episodic_component"],
        arrays["regular_mass"],
        arrays["episodic_mass"],
        selected_codes,
    )
    correction = _project_counterfactual_delta(
        arrays["base_attention_output"],
        candidate,
        output_projection,
    )
    selected_distance = np.take_along_axis(
        distances,
        selected_codes[..., None].astype(np.int64),
        axis=-1,
    )[..., 0]
    baseline_distance = distances[..., 4]
    metrics = _oracle_metrics(
        arrays["target_residual"],
        correction,
        selected_distance,
        baseline_distance,
    )
    histogram = np.bincount(selected_codes.reshape(-1), minlength=8)
    if int(histogram.sum()) != (
        _RECORDS * len(_READ_POSITIONS) * _LAYERS * _QUERY_HEADS
    ):
        raise AssertionError("head-mass selected-code population changed")
    result = {
        "single_oracle_arm": True,
        "gamma_table": [dict(row) for row in _gamma_table()],
        "tie_priority": list(_TIE_PRIORITY),
        "selected_code_histogram": {
            str(code): int(count) for code, count in enumerate(histogram)
        },
        "selected_code_stream_sha256": hashlib.sha256(
            selected_codes.tobytes(order="C")
        ).hexdigest(),
        "selected_mass_sha256": hashlib.sha256(
            selected_mass.tobytes(order="C")
        ).hexdigest(),
        "projected_correction_sha256": hashlib.sha256(
            correction.tobytes(order="C")
        ).hexdigest(),
        "metrics": metrics,
        "passed": bool(metrics["passed"]),
    }
    return result, selected_codes


def _screen_post_authentication(
    context: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
) -> dict[str, bool]:
    checks = _base_post_authentication(context, checkpoint=checkpoint)
    checks.update(
        {
            "head_mass_protocol": (
                sha256_file(context["head_mass_protocol_path"])
                == context["head_mass_protocol_sha256"]
            ),
            "head_mass_parity": (
                sha256_file(context["head_mass_parity_path"])
                == context["head_mass_parity_sha256"]
            ),
            "head_mass_source_inventory": (
                context["head_mass_protocol"]["source_sha256"] == _source_inventory()
            ),
        }
    )
    return checks


def screen_head_mass_oracle(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    shard_dir: str | Path,
    out: str | Path,
    runtime_factory: Callable[[Mapping[str, Any], float], Any] = (
        _open_mass_trace_runtime
    ),
) -> dict[str, Any]:
    output = capacity.bias.rank.retrieval._new_output(
        out,
        "head-mass result",
    )
    directory = _prepare_shard_directory(shard_dir)
    started = time.perf_counter()
    context, _training, frozen = _authenticate_protocol(protocol, protocol_sha256)
    records = context["train_records"]
    schedules = [
        capacity.bias.rank.fixed._derive_schedule(
            record["input_ids"],
            frozen["schedule_contract"]["tokenizer_fact_anchor_ids"],
        )
        for record in records
    ]
    if [row["rows_sha256"] for row in schedules] != frozen["schedule_contract"][
        "per_record_rows_sha256"
    ]:
        raise ValueError("head-mass execution schedule changed")
    raw = runtime_factory(context, 0.0)
    trace = _MassTraceCaptureRuntime(raw)
    manifest_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    try:
        _validate_runtime_route(raw, beta=0.0)
        for index, (record, schedule, historical) in enumerate(
            zip(
                records,
                schedules,
                context["historical_k256_evidence"],
                strict=True,
            )
        ):
            _progress(f"capturing train record {index + 1}/{_RECORDS}")
            first, arrays, positions = _execute_record(
                trace,
                record=record,
                context=context,
                schedule=schedule,
                resource=frozen["fixed_K256_arm"]["resource_contract"],
                progress_label=f"head-mass train record {index + 1}/{_RECORDS}",
            )
            if not capacity._evidence_exact(first, historical):
                raise ValueError(f"head-mass record {index} base evidence changed")
            first_summary = _trace_summary(arrays, positions)
            historical_arrays = _historical_trace_arrays(
                context,
                record_index=index,
            )
            historical_projected_hashes = _projected_trace_hashes(historical_arrays)
            observed_projected_hashes = _projected_trace_hashes(arrays)
            if historical_projected_hashes != observed_projected_hashes:
                raise ValueError(f"head-mass record {index} projected trace changed")
            trace.reset()
            replay, replay_arrays, replay_positions = _execute_record(
                trace,
                record=record,
                context=context,
                schedule=schedule,
                resource=frozen["fixed_K256_arm"]["resource_contract"],
            )
            replay_summary = _trace_summary(replay_arrays, replay_positions)
            if (
                not capacity._evidence_exact(first, replay)
                or first_summary != replay_summary
            ):
                raise ValueError(f"head-mass record {index} reset replay changed")
            source_record_sha256 = sha256_json(record)
            output_sha256 = sha256_json(capacity._without_elapsed(first))
            reset_output_sha256 = sha256_json(capacity._without_elapsed(replay))
            shard = _write_trace_shard(
                directory,
                record=record,
                arrays=arrays,
                positions=positions,
                source_record_sha256=source_record_sha256,
                output_sha256=output_sha256,
                reset_output_sha256=reset_output_sha256,
                reset_trace_sha256=replay_summary["trace_sha256"],
            )
            manifest_rows.append(shard)
            output_rows.append(
                {
                    "record_index": index,
                    "record_id": record["record_id"],
                    "historical_output_evidence_sha256": sha256_json(
                        capacity._without_elapsed(historical)
                    ),
                    "observed_output_evidence_sha256": output_sha256,
                    "reset_output_evidence_sha256": reset_output_sha256,
                    "base_outputs_counters_and_loss_exact": True,
                    "historical_projected_trace_exact": True,
                    "historical_projected_tensor_sha256": (historical_projected_hashes),
                    "observed_projected_tensor_sha256": observed_projected_hashes,
                    "reset_outputs_counters_loss_and_trace_exact": True,
                    "answer_cross_entropy": first["answer_cross_entropy"],
                    "hidden_sha256": first["hidden_sha256"],
                    "logits_sha256": first["logits_sha256"],
                    "counter_stream_sha256": first["counter_stream_sha256"],
                    "episodic_call_stream_sha256": first["episodic_call_stream_sha256"],
                    "source_record_sha256": source_record_sha256,
                    "trace_sha256": first_summary["trace_sha256"],
                    "reset_trace_sha256": replay_summary["trace_sha256"],
                    "shard_file_sha256": shard["file_sha256"],
                    "selected_code_stream_sha256": first_summary[
                        "selected_code_stream_sha256"
                    ],
                }
            )
            trace.reset()
    finally:
        trace.close()
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _RESULT_EXPERIMENT,
        "protocol": {
            "path": str(context["head_mass_protocol_path"]),
            "sha256": context["head_mass_protocol_sha256"],
        },
        "format": "safetensors",
        "record_order": list(range(_RECORDS)),
        "shards": manifest_rows,
        "confirmation_split_opened": False,
    }
    manifest_path = directory / "manifest.json"
    atomic_json(manifest_path, manifest)
    loaded: dict[str, list[np.ndarray]] = {name: [] for name in _TRACE_KEYS}
    for descriptor in manifest_rows:
        arrays = _validate_trace_shard(directory / descriptor["file"], descriptor)
        for name in _TRACE_KEYS:
            loaded[name].append(arrays[name])
    stacked = {
        name: np.ascontiguousarray(np.stack(rows), dtype=np.float32)
        for name, rows in loaded.items()
    }
    _progress("loading authenticated BF16 output projections")
    output_projection, projection_hashes = _load_output_projections(context)
    parity_projection_hashes = frozen["trace_parity"]["report"]["parity"][
        "output_projection_tensor_sha256"
    ]
    if projection_hashes != parity_projection_hashes:
        raise ValueError("head-mass output projection hashes changed after parity")
    _progress("running frozen dynamic per-head mass oracle")
    oracle, selected_codes = _run_oracle_from_arrays(stacked, output_projection)
    replay_oracle, replay_codes = _run_oracle_from_arrays(
        stacked,
        output_projection,
    )
    replay_checks = {
        "selected_codes_exact": np.array_equal(selected_codes, replay_codes),
        "oracle_metrics_exact": oracle == replay_oracle,
        "oracle_sha256_exact": sha256_json(oracle) == sha256_json(replay_oracle),
    }
    replay_checks["passed"] = all(replay_checks.values())
    if not replay_checks["passed"]:
        raise ValueError("head-mass deterministic oracle replay failed")
    post = _screen_post_authentication(
        context,
        checkpoint=frozen["training_checkpoint"],
    )
    if not post or not all(post.values()):
        raise ValueError("head-mass post-run authentication failed")
    passed = bool(oracle["passed"])
    report = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _RESULT_EXPERIMENT,
        "status": (
            "train_episodic_head_mass_oracle_gate_passed"
            if passed
            else "train_episodic_head_mass_oracle_gate_failed"
        ),
        "protocol": {
            "path": str(context["head_mass_protocol_path"]),
            "sha256": context["head_mass_protocol_sha256"],
        },
        "scope": {
            "split": "train",
            "records": _RECORDS,
            "positions_per_record": _POSITIONS,
            "trace_positions_per_record": len(_READ_POSITIONS),
            "same_state_capacity_evidence_only": True,
            "causal_rollout": False,
            "answer_cross_entropy_gate": False,
            "semantic_or_M3_gate_passed": False,
            "development_outcomes_used": False,
            "confirmation_split_opened": False,
        },
        "base_output_authentication": output_rows,
        "trace_manifest": {
            "directory": str(directory),
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "shard_count": len(manifest_rows),
            "shards": manifest_rows,
        },
        "output_projection": {
            "source": str(context["non_mlp_path"]),
            "dtype": "authenticated_BF16_loaded_as_float32",
            "orientation": "candidate_delta @ o_proj.weight.T",
            "tensor_sha256": projection_hashes,
            "parity_tensor_sha256_exact": True,
        },
        "oracle": oracle,
        "deterministic_replay": {
            "reference_sha256": sha256_json(oracle),
            "recomputed_sha256": sha256_json(replay_oracle),
            "checks": replay_checks,
            "passed": True,
        },
        "resource_contract": dict(frozen["resource_contract"]),
        "decision": {
            "train_head_mass_capacity_gate_passed": passed,
            "semantic_or_M3_gate_passed": False,
            "native_causal_integration_authorized": False,
            "development_authorized": False,
            "confirmation_authorized": False,
            "train_only_causal_predictor_experiment_authorized": passed,
            "next_step": (
                "freeze a train-only causal 3-bit gamma predictor with "
                "rolled-out native state updates and full traffic accounting"
                if passed
                else (
                    "close the fixed-K256 independent-head scheduled-source-"
                    "mass matching gamma grid"
                )
            ),
            "failure_scope": None if passed else frozen["failure_scope"],
        },
        "post_run_authentication": post,
        "confirmation_split_opened": False,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output, report)
    _progress(f"head-mass result written to {output}")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train-only OLMoE dynamic per-head episodic-mass oracle",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    parity = commands.add_parser("parity")
    parity.add_argument("--capacity-protocol", required=True)
    parity.add_argument("--capacity-protocol-sha256", required=True)
    parity.add_argument("--capacity-result", required=True)
    parity.add_argument("--capacity-result-sha256", required=True)
    parity.add_argument("--trace-library", required=True)
    parity.add_argument("--trace-library-sha256", required=True)
    parity.add_argument("--out", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--capacity-protocol", required=True)
    freeze.add_argument("--capacity-protocol-sha256", required=True)
    freeze.add_argument("--capacity-result", required=True)
    freeze.add_argument("--capacity-result-sha256", required=True)
    freeze.add_argument("--trace-library", required=True)
    freeze.add_argument("--trace-library-sha256", required=True)
    freeze.add_argument("--parity-report", required=True)
    freeze.add_argument("--parity-report-sha256", required=True)
    freeze.add_argument("--out", required=True)
    screen = commands.add_parser("screen")
    screen.add_argument("--protocol", required=True)
    screen.add_argument("--protocol-sha256", required=True)
    screen.add_argument("--shard-dir", required=True)
    screen.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "parity":
        value = generate_trace_parity_report(
            capacity_protocol=args.capacity_protocol,
            capacity_protocol_sha256=args.capacity_protocol_sha256,
            capacity_result=args.capacity_result,
            capacity_result_sha256=args.capacity_result_sha256,
            trace_library=args.trace_library,
            trace_library_sha256=args.trace_library_sha256,
            out=args.out,
        )
    elif args.command == "freeze":
        value = freeze_head_mass_protocol(
            capacity_protocol=args.capacity_protocol,
            capacity_protocol_sha256=args.capacity_protocol_sha256,
            capacity_result=args.capacity_result,
            capacity_result_sha256=args.capacity_result_sha256,
            trace_library=args.trace_library,
            trace_library_sha256=args.trace_library_sha256,
            parity_report=args.parity_report,
            parity_report_sha256=args.parity_report_sha256,
            out=args.out,
        )
    elif args.command == "screen":
        value = screen_head_mass_oracle(
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            shard_dir=args.shard_dir,
            out=args.out,
        )
    else:  # pragma: no cover - argparse enforces the set
        raise AssertionError(f"unknown command: {args.command}")
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
