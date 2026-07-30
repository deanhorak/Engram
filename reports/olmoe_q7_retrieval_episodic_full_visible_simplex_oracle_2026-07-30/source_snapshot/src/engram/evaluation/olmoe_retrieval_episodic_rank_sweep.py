"""Prospectively frozen train-only ranked-prefix episodic sweep.

This experiment follows a systems-clean fixed-K51 failure.  It derives four
larger layer/head masks from the already frozen M2 projected-score ordering,
using stable descending score and layer-major tie-breaking.  In order, it
tests K={64,96,128,165} with the original causal 32-slot payload schedule and
the generic head-gated native episodic ABI.

Each candidate executes all eight train records before its strict
cross-entropy gate is evaluated against the authenticated M2 checkpoint.  The
smallest passing K stops the sweep.  If none passes, all four execute and the
lexicographically best (worst loss, mean loss, K) failed candidate is retained
for deterministic reset replay.  M0 and the prerequisite K51 result are
attribution only.  No teacher, development outcome, or confirmation file is
accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import engram.evaluation.olmoe_retrieval_episodic_head_mask_oracle as fixed
import engram.evaluation.olmoe_retrieval_episodic_oracle as episodic
import engram.evaluation.olmoe_retrieval_head_selector as retrieval
from engram.compiler.olmoe_native import validate_olmoe_native_package
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.utils import atomic_json, sha256_file, sha256_json


_SCHEMA_VERSION = 1
_PROTOCOL_EXPERIMENT = "olmoe_q7_retrieval_episodic_rank_sweep_protocol"
_RESULT_EXPERIMENT = "olmoe_q7_retrieval_episodic_rank_sweep_train_screen"
_PROTOCOL_STATUS = "frozen_before_ranked_prefix_train_candidate_execution"
_RECORDS = 8
_POSITIONS = 128
_ANSWER_START = 96
_ANSWER_POSITIONS = 32
_SPAN_TOKENS = 8
_SLOTS = 32
_CACHE_DTYPE_BYTES = 2
_POSITION_DTYPE_BYTES = 8
_SCRATCH_DTYPE_BYTES = 4
_THREADS = 12
_MAXIMUM_TRAFFIC_FRACTION = 0.45
_CANDIDATE_K = (64, 96, 128, 165)
_EXPECTED_MASK_SHA256 = {
    64: "954517badefeb1fb9cdf9acf06c3861dff0955fcd827dd600fb84d2570e42098",
    96: "1691d6086a546152cad0a095e0ad2db2839cf57a8b4363217d79efaa396cac6a",
    128: "e2a663d8d4f5ae8e3d7d8b7ac71ba68cdfffa27bf663a310c9aea4f953977110",
    165: "758f0938c4a775ba11f9e124ccaf07d9097a482e2d5be1b3d9aef369aed2ede8",
}
_EXPECTED_RESOURCES = {
    64: {
        "combined_state_bytes": 10_010_112,
        "combined_scratch_bytes": 4_736,
        "combined_attention_and_episodic_read_bytes": 685_506_560,
        "combined_attention_and_episodic_traffic_bytes": 689_176_576,
    },
    96: {
        "combined_state_bytes": 10_272_512,
        "combined_scratch_bytes": 4_800,
        "combined_attention_and_episodic_read_bytes": 689_700_864,
        "combined_attention_and_episodic_traffic_bytes": 693_633_024,
    },
    128: {
        "combined_state_bytes": 10_534_912,
        "combined_scratch_bytes": 4_864,
        "combined_attention_and_episodic_read_bytes": 693_895_168,
        "combined_attention_and_episodic_traffic_bytes": 698_089_472,
    },
    165: {
        "combined_state_bytes": 10_534_912,
        "combined_scratch_bytes": 4_864,
        "combined_attention_and_episodic_read_bytes": 698_744_832,
        "combined_attention_and_episodic_traffic_bytes": 702_939_136,
    },
}
_EPISODIC_COUNTER_NAMES = episodic._EPISODIC_COUNTER_NAMES
_SOURCE_FILES = (
    *fixed._SOURCE_FILES,
    "src/engram/evaluation/olmoe_retrieval_episodic_rank_sweep.py",
)


def _progress(message: str) -> None:
    print(
        f"[retrieval-episodic-rank-sweep] {message}",
        file=sys.stderr,
        flush=True,
    )


def _source_inventory() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    return {name: sha256_file(repository / name) for name in _SOURCE_FILES}


def _validate_frozen_ordering(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("retrieval episodic rank ordering is invalid")
    rows = value.get("ordering")
    if (
        not isinstance(rows, list)
        or len(rows) != retrieval._LAYERS * retrieval._HEADS
        or value.get("ordering_sha256") != sha256_json(rows)
        or value.get("positive_score_count")
        != fixed._EXPECTED_POSITIVE_PROJECTED_SCORES
    ):
        raise ValueError("retrieval episodic rank ordering contract changed")
    normalized: list[dict[str, Any]] = []
    coordinates: list[tuple[int, int]] = []
    previous_key: tuple[float, int] | None = None
    for rank, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError("retrieval episodic rank row is invalid")
        layer = row.get("layer")
        head = row.get("head")
        index = row.get("layer_major_index")
        score = row.get("projected_score")
        if (
            row.get("rank") != rank
            or isinstance(layer, bool)
            or not isinstance(layer, int)
            or isinstance(head, bool)
            or not isinstance(head, int)
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index != layer * retrieval._HEADS + head
            or not 0 <= layer < retrieval._LAYERS
            or not 0 <= head < retrieval._HEADS
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not np.isfinite(float(score))
        ):
            raise ValueError("retrieval episodic rank coordinate is invalid")
        key = (-float(score), index)
        if previous_key is not None and key < previous_key:
            raise ValueError("retrieval episodic rank stable ordering changed")
        previous_key = key
        coordinates.append((layer, head))
        normalized.append(dict(row))
    if len(set(coordinates)) != retrieval._LAYERS * retrieval._HEADS:
        raise ValueError("retrieval episodic rank coordinates are not unique")
    positive_count = sum(float(row["projected_score"]) > 0.0 for row in normalized)
    if positive_count != fixed._EXPECTED_POSITIVE_PROJECTED_SCORES:
        raise ValueError("retrieval episodic rank positive boundary changed")
    return normalized


def _rank_prefix_mask(
    ordering: Sequence[Mapping[str, Any]],
    k: int,
) -> np.ndarray:
    if k not in _CANDIDATE_K:
        raise ValueError("retrieval episodic rank candidate K is invalid")
    if len(ordering) != retrieval._LAYERS * retrieval._HEADS:
        raise ValueError("retrieval episodic rank population is invalid")
    mask = np.zeros(
        (retrieval._LAYERS, retrieval._HEADS),
        dtype=np.bool_,
    )
    for row in ordering[:k]:
        mask[int(row["layer"]), int(row["head"])] = True
    observed_sha256 = sha256_json(mask.tolist())
    if int(mask.sum()) != k or observed_sha256 != _EXPECTED_MASK_SHA256[k]:
        raise ValueError(f"retrieval episodic rank K{k} mask contract changed")
    return mask


def _validate_candidate_mask(value: Any, k: int) -> np.ndarray:
    if (
        isinstance(value, np.ndarray)
        and value.shape == (retrieval._LAYERS, retrieval._HEADS)
        and value.dtype == np.bool_
    ):
        mask = value.copy()
    else:
        mask = retrieval._boolean_mask(
            value,
            f"retrieval episodic rank K{k}",
        )
    if int(mask.sum()) != k or sha256_json(mask.tolist()) != _EXPECTED_MASK_SHA256[k]:
        raise ValueError(f"retrieval episodic rank K{k} mask contract changed")
    return mask


def _mask_descriptor(mask: np.ndarray, k: int) -> dict[str, Any]:
    frozen = _validate_candidate_mask(mask, k)
    counts = [int(row.sum()) for row in frozen]
    active_layers = [layer for layer, count in enumerate(counts) if count > 0]
    return {
        "K": k,
        "mask": frozen.tolist(),
        "mask_sha256": sha256_json(frozen.tolist()),
        "selected_layer_head_pairs": int(frozen.sum()),
        "selected_heads_per_layer": counts,
        "active_layers": active_layers,
        "active_layer_count": len(active_layers),
        "source": "prefix of frozen M2 projected-score ordering",
        "fitted_by_this_experiment": False,
    }


def _resource_contract(
    model: Mapping[str, int],
    q7_expectations: Mapping[str, int],
    mask: np.ndarray,
    k: int,
) -> dict[str, Any]:
    frozen = _validate_candidate_mask(mask, k)
    base = episodic.sustained._attention_expectations(
        dict(model),
        retrieval._BASE_POLICY,
        positions=_POSITIONS,
    )
    layers = int(model["layers"])
    query_heads = int(model["query_heads"])
    key_value_heads = int(model["key_value_heads"])
    head_dimension = int(model["head_dimension"])
    hidden_size = int(model["hidden_size"])
    if (
        layers != retrieval._LAYERS
        or query_heads != retrieval._HEADS
        or hidden_size != query_heads * head_dimension
    ):
        raise ValueError("retrieval episodic rank model geometry changed")
    active_layers = int(np.count_nonzero(np.any(frozen, axis=1)))
    selected_pairs = int(frozen.sum())
    key_value_width = key_value_heads * head_dimension
    slot_payload_bytes_per_active_layer = 2 * key_value_width * _CACHE_DTYPE_BYTES
    payload_state_bytes = _SLOTS * active_layers * slot_payload_bytes_per_active_layer
    position_state_bytes = _SLOTS * active_layers * _POSITION_DTYPE_BYTES
    write_bytes = payload_state_bytes
    read_events = _ANSWER_POSITIONS * active_layers
    entries_read = _ANSWER_POSITIONS * _SPAN_TOKENS * selected_pairs
    key_read_bytes = entries_read * head_dimension * _CACHE_DTYPE_BYTES
    value_read_bytes = key_read_bytes
    read_bytes = key_read_bytes + value_read_bytes
    joint_softmax_scratch_bytes = (
        active_layers * 2 * _SPAN_TOKENS * _SCRATCH_DTYPE_BYTES
    )
    combined_state_bytes = (
        int(base["attention_state_bytes"]) + payload_state_bytes + position_state_bytes
    )
    combined_scratch_bytes = (
        int(base["attention_scratch_bytes"]) + joint_softmax_scratch_bytes
    )
    dense_bytes = int(base["dense_full_context_logical_kv_bytes"])
    combined_read_bytes = int(base["attention_logical_read_bytes"]) + read_bytes
    combined_traffic_bytes = combined_read_bytes + write_bytes
    read_fraction = combined_read_bytes / dense_bytes
    traffic_fraction = combined_traffic_bytes / dense_bytes
    contract = {
        "candidate_head_mask": _mask_descriptor(frozen, k),
        "cache_payload": {
            "dtype": "bfloat16",
            "dtype_bytes": _CACHE_DTYPE_BYTES,
            "capacity_slots_per_active_layer": _SLOTS,
            "span_size": _SPAN_TOKENS,
            "active_layers": active_layers,
            "slot_payload_bytes_per_active_layer": (
                slot_payload_bytes_per_active_layer
            ),
            "payload_state_bytes": payload_state_bytes,
            "position_dtype": "uint64",
            "position_state_bytes": position_state_bytes,
            "state_bytes": payload_state_bytes + position_state_bytes,
        },
        "schedule": {
            "unique_write_slots": _SLOTS,
            "source_write_rows": _SLOTS,
            "layer_write_events": _SLOTS * active_layers,
            "answer_read_rows": _ANSWER_POSITIONS,
            "layer_read_events": read_events,
            "read_span_tokens": _SPAN_TOKENS,
            "selected_layer_head_pairs": selected_pairs,
            "entries_read": entries_read,
        },
        "base_attention_expectations_per_sequence": base,
        "episodic_write_bytes_per_sequence": write_bytes,
        "episodic_key_read_bytes_per_sequence": key_read_bytes,
        "episodic_value_read_bytes_per_sequence": value_read_bytes,
        "episodic_read_bytes_per_sequence": read_bytes,
        "combined_attention_and_episodic_read_bytes": combined_read_bytes,
        "combined_attention_and_episodic_traffic_bytes": combined_traffic_bytes,
        "dense_full_context_logical_kv_bytes": dense_bytes,
        "combined_read_fraction_of_dense": read_fraction,
        "combined_traffic_fraction_of_dense": traffic_fraction,
        "maximum_traffic_fraction_of_dense": _MAXIMUM_TRAFFIC_FRACTION,
        "within_read_budget": read_fraction <= _MAXIMUM_TRAFFIC_FRACTION,
        "within_total_traffic_budget": (traffic_fraction <= _MAXIMUM_TRAFFIC_FRACTION),
        "combined_state_bytes": combined_state_bytes,
        "episodic_joint_softmax_scratch_bytes": joint_softmax_scratch_bytes,
        "combined_scratch_bytes": combined_scratch_bytes,
        "q7_expectations_per_sequence": dict(q7_expectations),
    }
    observed = {name: int(contract[name]) for name in _EXPECTED_RESOURCES[k]}
    if (
        observed != _EXPECTED_RESOURCES[k]
        or not contract["within_read_budget"]
        or not contract["within_total_traffic_budget"]
    ):
        raise ValueError(f"retrieval episodic rank K{k} resource contract changed")
    return contract


def _validate_k51_result(
    *,
    path: str | Path,
    expected_sha256: str,
    protocol_path: Path,
    protocol_sha256: str,
) -> dict[str, Any]:
    source = episodic._checked_file(
        path,
        expected_sha256,
        "retrieval episodic rank K51 result",
    )
    result = retrieval._read_json(
        source,
        "retrieval episodic rank K51 result",
    )
    candidate = result.get("episodic_head_mask_candidate")
    decision = result.get("decision")
    scope = result.get("scope")
    authentication = result.get("post_run_authentication")
    required_authentication = {
        "base_protocol",
        "package",
        "corpus_manifest",
        "train_split",
        "confirmation_not_opened",
        "source_config",
        "source_index",
        "source_shards",
        "expert_proxy",
        "headwise_episodic_library",
        "head_mask_protocol",
        "head_mask_source_inventory",
        "training_checkpoint",
        "layered_library",
        "headwise_library",
        "attention_library",
    }
    if (
        result.get("schema_version") != fixed._SCHEMA_VERSION
        or result.get("experiment") != fixed._RESULT_EXPERIMENT
        or result.get("status") != "train_episodic_head_mask_gate_failed"
        or result.get("protocol")
        != {
            "path": str(protocol_path),
            "sha256": protocol_sha256,
        }
        or not isinstance(candidate, Mapping)
        or candidate.get("resource_passed") is not True
        or candidate.get("loss_gate", {}).get("passed") is not False
        or candidate.get("passed") is not False
        or candidate.get("fixed_M2_head_mask", {}).get("mask_sha256")
        != fixed._EXPECTED_M2_MASK_SHA256
        or not isinstance(decision, Mapping)
        or decision.get("passed") is not False
        or decision.get("semantic_gate_passed") is not False
        or decision.get("confirmation_authorized") is not False
        or not isinstance(scope, Mapping)
        or scope.get("development_outcomes_used") is not False
        or scope.get("confirmation_split_opened") is not False
        or not isinstance(authentication, Mapping)
        or not authentication
        or not all(value is True for value in authentication.values())
        or not required_authentication.issubset(authentication)
        or result.get("confirmation_split_opened") is not False
    ):
        raise ValueError("retrieval episodic rank K51 prerequisite is invalid")
    loss_gate = candidate["loss_gate"]
    matrix = loss_gate.get("matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != _RECORDS
        or [row.get("record_index") for row in matrix] != list(range(_RECORDS))
        or any(
            not isinstance(row.get("record_id"), str)
            or isinstance(row.get("candidate_answer_cross_entropy"), bool)
            or not isinstance(
                row.get("candidate_answer_cross_entropy"),
                (int, float),
            )
            or not np.isfinite(float(row["candidate_answer_cross_entropy"]))
            or float(row["candidate_answer_cross_entropy"]) < 0.0
            for row in matrix
        )
    ):
        raise ValueError("retrieval episodic rank K51 loss evidence is invalid")
    return {
        "path": str(source),
        "sha256": expected_sha256.lower(),
        "status": result["status"],
        "systems_clean": True,
        "loss_gate_passed": False,
        "loss_summaries": loss_gate.get("summaries"),
        "record_ids": [row["record_id"] for row in matrix],
        "record_answer_cross_entropy": [
            float(row["candidate_answer_cross_entropy"]) for row in matrix
        ],
        "loss_matrix_sha256": sha256_json(matrix),
        "attribution_only": True,
    }


def _validate_all_head_attribution(
    *,
    protocol_path: str | Path,
    protocol_sha256: str,
    result_path: str | Path,
    result_sha256: str,
    context: Mapping[str, Any],
    checkpoint: Mapping[str, str],
) -> dict[str, Any]:
    frozen_path = episodic._checked_file(
        protocol_path,
        protocol_sha256,
        "retrieval episodic rank all-head protocol",
    )
    frozen = retrieval._read_json(
        frozen_path,
        "retrieval episodic rank all-head protocol",
    )
    library = frozen.get("episodic_library")
    train_scope = frozen.get("train_scope")
    if (
        frozen.get("schema_version") != episodic._SCHEMA_VERSION
        or frozen.get("experiment") != episodic._PROTOCOL_EXPERIMENT
        or frozen.get("status") != episodic._PROTOCOL_STATUS
        or frozen.get("base_retrieval_protocol")
        != {
            "path": str(context["protocol_path"]),
            "sha256": context["protocol_sha256"],
        }
        or frozen.get("training_checkpoint", {}).get("path") != checkpoint["path"]
        or frozen.get("training_checkpoint", {}).get("sha256") != checkpoint["sha256"]
        or not isinstance(library, Mapping)
        or not isinstance(train_scope, Mapping)
        or train_scope.get("development_outcomes_used") is not False
        or train_scope.get("confirmation_file_access_permitted") is not False
        or frozen.get("confirmation_split_opened") is not False
    ):
        raise ValueError("retrieval episodic rank all-head protocol is invalid")
    historical_library = episodic._checked_file(
        library.get("path"),
        library.get("sha256"),
        "retrieval episodic rank historical all-head library",
    )
    source = episodic._checked_file(
        result_path,
        result_sha256,
        "retrieval episodic rank all-head result",
    )
    result = retrieval._read_json(
        source,
        "retrieval episodic rank all-head result",
    )
    candidate = result.get("episodic_candidate")
    control = result.get("fresh_full_W128_Q7_control")
    teacher = result.get("fresh_teacher_retrieval_evidence")
    scope = result.get("scope")
    decision = result.get("decision")
    authentication = result.get("post_run_authentication")
    required_authentication = {
        "base_protocol",
        "package",
        "corpus_manifest",
        "train_split",
        "confirmation_not_opened",
        "source_config",
        "source_index",
        "source_shards",
        "expert_proxy",
        "episodic_library",
        "episodic_protocol",
        "episodic_source_inventory",
        "training_checkpoint",
        "layered_library",
        "headwise_library",
        "attention_library",
    }
    if (
        result.get("schema_version") != episodic._SCHEMA_VERSION
        or result.get("experiment") != episodic._RESULT_EXPERIMENT
        or result.get("status") != "train_episodic_oracle_gate_failed"
        or result.get("protocol")
        != {
            "path": str(frozen_path),
            "sha256": protocol_sha256.lower(),
        }
        or not isinstance(candidate, Mapping)
        or candidate.get("resource_passed") is not True
        or candidate.get("quality_passed") is not False
        or candidate.get("passed") is not False
        or not isinstance(control, Mapping)
        or control.get("quality_passed") is not True
        or not isinstance(teacher, Mapping)
        or teacher.get("passed") is not True
        or not isinstance(scope, Mapping)
        or scope.get("development_outcomes_used") is not False
        or scope.get("confirmation_split_opened") is not False
        or not isinstance(decision, Mapping)
        or decision.get("passed") is not False
        or decision.get("confirmation_authorized") is not False
        or not isinstance(authentication, Mapping)
        or not authentication
        or not all(value is True for value in authentication.values())
        or not required_authentication.issubset(authentication)
        or result.get("confirmation_split_opened") is not False
    ):
        raise ValueError("retrieval episodic rank all-head result is invalid")
    evidence = candidate.get("sequence_evidence")
    if (
        not isinstance(evidence, list)
        or len(evidence) != _RECORDS
        or [row.get("record_index") for row in evidence] != list(range(_RECORDS))
        or any(
            not isinstance(row.get("record_id"), str)
            or row.get("counter_stream_passed") is not True
            or isinstance(row.get("answer_cross_entropy"), bool)
            or not isinstance(
                row.get("answer_cross_entropy"),
                (int, float),
            )
            or not np.isfinite(float(row["answer_cross_entropy"]))
            or float(row["answer_cross_entropy"]) < 0.0
            for row in evidence
        )
    ):
        raise ValueError(
            "retrieval episodic rank all-head sequence evidence is invalid"
        )
    return {
        "protocol": {
            "path": str(frozen_path),
            "sha256": protocol_sha256.lower(),
        },
        "result": {
            "path": str(source),
            "sha256": result_sha256.lower(),
        },
        "historical_episodic_library": {
            "path": str(historical_library),
            "sha256": library["sha256"],
        },
        "status": result["status"],
        "systems_clean": True,
        "record_ids": [row["record_id"] for row in evidence],
        "record_answer_cross_entropy": [
            float(row["answer_cross_entropy"]) for row in evidence
        ],
        "sequence_evidence_sha256": sha256_json(evidence),
        "attribution_only": True,
    }


def _authenticate_sweep_inputs(
    *,
    base_protocol: str | Path,
    base_protocol_sha256: str,
    training_checkpoint: str | Path,
    training_checkpoint_sha256: str,
    headwise_episodic_library: str | Path,
    headwise_episodic_library_sha256: str,
    k51_protocol: str | Path,
    k51_protocol_sha256: str,
    k51_result: str | Path,
    k51_result_sha256: str,
    all_head_protocol: str | Path,
    all_head_protocol_sha256: str,
    all_head_result: str | Path,
    all_head_result_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    context, training, selection, checkpoint = episodic._authenticate_base_inputs(
        base_protocol,
        base_protocol_sha256,
        training_checkpoint,
        training_checkpoint_sha256,
        headwise_episodic_library,
        headwise_episodic_library_sha256,
    )
    if (
        selection.get("screen_eligible") is not True
        or selection.get("selected_mask_name") != "M2"
    ):
        raise ValueError("retrieval episodic rank sweep requires eligible M2 evidence")
    k51_context, k51_training, frozen_k51 = fixed._authenticate_protocol(
        k51_protocol,
        k51_protocol_sha256,
    )
    k51_path = Path(k51_protocol).expanduser().resolve()
    if (
        frozen_k51["base_retrieval_protocol"]
        != {
            "path": str(context["protocol_path"]),
            "sha256": context["protocol_sha256"],
        }
        or frozen_k51["training_checkpoint"]["path"] != checkpoint["path"]
        or frozen_k51["training_checkpoint"]["sha256"] != checkpoint["sha256"]
        or frozen_k51["headwise_episodic_library"]
        != {
            "path": str(context["episodic_library_path"]),
            "sha256": context["episodic_library_sha256"],
            "required_open_symbol": ("engram_olmoe_token_open_episodic_headwise_v1"),
        }
        or sha256_json(k51_training) != sha256_json(training)
        or k51_context["protocol_path"] != context["protocol_path"]
        or k51_context["episodic_library_path"] != context["episodic_library_path"]
    ):
        raise ValueError("retrieval episodic rank K51 bindings changed")
    prerequisite = _validate_k51_result(
        path=k51_result,
        expected_sha256=k51_result_sha256,
        protocol_path=k51_path,
        protocol_sha256=k51_protocol_sha256.lower(),
    )
    all_head = _validate_all_head_attribution(
        protocol_path=all_head_protocol,
        protocol_sha256=all_head_protocol_sha256,
        result_path=all_head_result,
        result_sha256=all_head_result_sha256,
        context=context,
        checkpoint=checkpoint,
    )
    if all_head["record_ids"] != prerequisite["record_ids"] or not all(
        all_head_loss < k51_loss
        for all_head_loss, k51_loss in zip(
            all_head["record_answer_cross_entropy"],
            prerequisite["record_answer_cross_entropy"],
            strict=True,
        )
    ):
        raise ValueError("retrieval episodic rank all-head/K51 attribution changed")
    all_head = dict(all_head)
    all_head["strictly_better_than_K51_on_all_train_records"] = True
    return (
        context,
        training,
        checkpoint,
        frozen_k51,
        prerequisite,
        all_head,
    )


def _build_protocol(
    *,
    context: Mapping[str, Any],
    training: Mapping[str, Any],
    checkpoint: Mapping[str, str],
    frozen_k51: Mapping[str, Any],
    k51_protocol_path: str | Path,
    k51_protocol_sha256: str,
    prerequisite: Mapping[str, Any],
    all_head_attribution: Mapping[str, Any],
) -> dict[str, Any]:
    state = fixed._checkpoint_references(training)
    ordering_payload = frozen_k51["authenticated_M2_projected_score_ordering"]
    if ordering_payload != state["projected_score_ordering"]:
        raise ValueError("retrieval episodic rank ordering/checkpoint binding changed")
    ordering = _validate_frozen_ordering(ordering_payload)
    candidates: list[dict[str, Any]] = []
    for k in _CANDIDATE_K:
        mask = _rank_prefix_mask(ordering, k)
        candidates.append(
            {
                "K": k,
                "head_mask": _mask_descriptor(mask, k),
                "resource_contract": _resource_contract(
                    context["model"],
                    context["q7_expectations"],
                    mask,
                    k,
                ),
            }
        )
    anchors = fixed._fact_anchor_ids(context["tokenizer_path"])
    schedules = [
        fixed._derive_schedule(record["input_ids"], anchors)
        for record in context["train_records"]
    ]
    return {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _PROTOCOL_EXPERIMENT,
        "status": _PROTOCOL_STATUS,
        "base_retrieval_protocol": {
            "path": str(context["protocol_path"]),
            "sha256": context["protocol_sha256"],
        },
        "training_checkpoint": {
            "path": checkpoint["path"],
            "sha256": checkpoint["sha256"],
            "training_sha256": sha256_json(training),
        },
        "headwise_episodic_library": {
            "path": str(context["episodic_library_path"]),
            "sha256": context["episodic_library_sha256"],
            "required_open_symbol": ("engram_olmoe_token_open_episodic_headwise_v1"),
        },
        "K51_prerequisite": {
            "protocol": {
                "path": str(Path(k51_protocol_path).expanduser().resolve()),
                "sha256": k51_protocol_sha256.lower(),
            },
            "result": dict(prerequisite),
            "required_outcome": ("systems-clean strict train loss-gate failure"),
        },
        "all_head_K256_attribution": dict(all_head_attribution),
        "authenticated_confirmation_descriptor": dict(
            context["confirmation_descriptor"]
        ),
        "source_sha256": _source_inventory(),
        "train_scope": {
            "records": _RECORDS,
            "positions_per_record": _POSITIONS,
            "answer_positions_per_record": _ANSWER_POSITIONS,
            "record_identity_sha256": sha256_json(
                [record["identity_sha256"] for record in context["train_records"]]
            ),
            "dense_teacher_forwards": 0,
            "candidate_masks_fitted_by_this_experiment": False,
            "candidate_selection_uses_train_outcomes": True,
            "development_outcomes_used": False,
            "confirmation_file_access_permitted": False,
            "cpu_only_candidate_execution": True,
        },
        "frozen_projected_score_ordering": ordering_payload,
        "candidate_order": list(_CANDIDATE_K),
        "candidates": candidates,
        "tokenizer_fact_anchor_ids": {
            label: list(values) for label, values in anchors.items()
        },
        "schedule_contract": {
            "derivation_input": ("input_ids[0:97] and authenticated tokenizer anchors"),
            "last_input_index_observed": _ANSWER_START,
            "write_source_starts": list(retrieval._PASSKEY_SOURCE_STARTS),
            "payload_spans": 4,
            "payload_tokens_per_span": _SPAN_TOKENS,
            "unique_canonical_slots": _SLOTS,
            "answer_prediction_rows": list(range(_ANSWER_START, _POSITIONS)),
            "per_record_rows_sha256": [
                schedule["rows_sha256"] for schedule in schedules
            ],
        },
        "runtime_abi": {
            "class": "OLMoENativeTokenRuntime",
            "method": "forward_episodic",
            "episodic_policy": {
                "slots": _SLOTS,
                "span_size": _SPAN_TOKENS,
            },
            "mask_argument": "episodic_head_mask",
            "required_open_symbol": ("engram_olmoe_token_open_episodic_headwise_v1"),
            "cumulative_metric_names": list(_EPISODIC_COUNTER_NAMES),
        },
        "execution_contract": {
            "ordered_smallest_passing_early_stop": True,
            "all_eight_records_before_each_gate": True,
            "strict_gate_reference": "authenticated M2 train losses",
            "strict_mean_improvement": True,
            "strict_worst_improvement": True,
            "per_record_regression_permitted": False,
            "total_failure_selection_key": [
                "maximum_answer_cross_entropy",
                "mean_answer_cross_entropy",
                "K",
            ],
            "reset_replay": (
                "retained selected/best runtime after its eight-record "
                "population; one reset replay of record 0"
            ),
            "executed_and_skipped_candidates_must_be_explicit": True,
        },
        "reused_checkpoint_evidence": state["baselines"],
        "attribution_contract": {
            "M0": "checkpoint attribution only",
            "K51": "authenticated prerequisite attribution only",
            "K256": "authenticated historical attribution only",
            "K32": {
                "ruled_out": False,
                "executed_by_this_sweep": False,
                "reason": (
                    "outside the precommitted candidate order; K51/K256 "
                    "evidence does not establish monotonic benefit"
                ),
                "candidate_set_revision": {
                    "K51_prospective_boundaries_included_K32": True,
                    "current_precommit_omits_K32": True,
                    "basis": (
                        "authenticated K51/K256 train evidence pointed "
                        "toward testing larger K next"
                    ),
                    "interpretation_limit": (
                        "K32 is not ruled out and the endpoint evidence "
                        "does not establish monotonicity"
                    ),
                },
            },
        },
        "confirmation_split_opened": False,
    }


def freeze_episodic_rank_sweep_protocol(
    *,
    base_protocol: str | Path,
    base_protocol_sha256: str,
    training_checkpoint: str | Path,
    training_checkpoint_sha256: str,
    headwise_episodic_library: str | Path,
    headwise_episodic_library_sha256: str,
    k51_protocol: str | Path,
    k51_protocol_sha256: str,
    k51_result: str | Path,
    k51_result_sha256: str,
    all_head_protocol: str | Path,
    all_head_protocol_sha256: str,
    all_head_result: str | Path,
    all_head_result_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    output = retrieval._new_output(
        out,
        "retrieval episodic rank-sweep protocol",
    )
    context, training, checkpoint, frozen_k51, prerequisite, all_head = (
        _authenticate_sweep_inputs(
            base_protocol=base_protocol,
            base_protocol_sha256=base_protocol_sha256,
            training_checkpoint=training_checkpoint,
            training_checkpoint_sha256=training_checkpoint_sha256,
            headwise_episodic_library=headwise_episodic_library,
            headwise_episodic_library_sha256=(headwise_episodic_library_sha256),
            k51_protocol=k51_protocol,
            k51_protocol_sha256=k51_protocol_sha256,
            k51_result=k51_result,
            k51_result_sha256=k51_result_sha256,
            all_head_protocol=all_head_protocol,
            all_head_protocol_sha256=all_head_protocol_sha256,
            all_head_result=all_head_result,
            all_head_result_sha256=all_head_result_sha256,
        )
    )
    protocol = _build_protocol(
        context=context,
        training=training,
        checkpoint=checkpoint,
        frozen_k51=frozen_k51,
        k51_protocol_path=k51_protocol,
        k51_protocol_sha256=k51_protocol_sha256,
        prerequisite=prerequisite,
        all_head_attribution=all_head,
    )
    atomic_json(output, protocol)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "protocol": protocol,
    }


def _authenticate_protocol(
    protocol: str | Path,
    protocol_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = episodic._checked_file(
        protocol,
        protocol_sha256,
        "retrieval episodic rank-sweep protocol",
    )
    value = retrieval._read_json(
        source,
        "retrieval episodic rank-sweep protocol",
    )
    base = value.get("base_retrieval_protocol")
    checkpoint = value.get("training_checkpoint")
    library = value.get("headwise_episodic_library")
    k51 = value.get("K51_prerequisite")
    k51_protocol = k51.get("protocol") if isinstance(k51, Mapping) else None
    k51_result = k51.get("result") if isinstance(k51, Mapping) else None
    all_head = value.get("all_head_K256_attribution")
    all_head_protocol = (
        all_head.get("protocol") if isinstance(all_head, Mapping) else None
    )
    all_head_result = all_head.get("result") if isinstance(all_head, Mapping) else None
    if not all(
        isinstance(binding, Mapping)
        for binding in (
            base,
            checkpoint,
            library,
            k51_protocol,
            k51_result,
            all_head_protocol,
            all_head_result,
        )
    ):
        raise ValueError("retrieval episodic rank-sweep bindings are invalid")
    (
        context,
        training,
        loaded_checkpoint,
        frozen_k51,
        prerequisite,
        loaded_all_head,
    ) = _authenticate_sweep_inputs(
        base_protocol=base.get("path"),
        base_protocol_sha256=base.get("sha256"),
        training_checkpoint=checkpoint.get("path"),
        training_checkpoint_sha256=checkpoint.get("sha256"),
        headwise_episodic_library=library.get("path"),
        headwise_episodic_library_sha256=library.get("sha256"),
        k51_protocol=k51_protocol.get("path"),
        k51_protocol_sha256=k51_protocol.get("sha256"),
        k51_result=k51_result.get("path"),
        k51_result_sha256=k51_result.get("sha256"),
        all_head_protocol=all_head_protocol.get("path"),
        all_head_protocol_sha256=all_head_protocol.get("sha256"),
        all_head_result=all_head_result.get("path"),
        all_head_result_sha256=all_head_result.get("sha256"),
    )
    expected = _build_protocol(
        context=context,
        training=training,
        checkpoint=loaded_checkpoint,
        frozen_k51=frozen_k51,
        k51_protocol_path=k51_protocol["path"],
        k51_protocol_sha256=k51_protocol["sha256"],
        prerequisite=prerequisite,
        all_head_attribution=loaded_all_head,
    )
    if value != expected:
        raise ValueError("retrieval episodic rank-sweep protocol contract changed")
    context = dict(context)
    context["rank_sweep_protocol_path"] = source
    context["rank_sweep_protocol_sha256"] = protocol_sha256.lower()
    context["rank_sweep_protocol"] = expected
    context["k51_protocol_path"] = Path(k51_protocol["path"]).resolve()
    context["k51_protocol_sha256"] = k51_protocol["sha256"]
    context["k51_result_path"] = Path(k51_result["path"]).resolve()
    context["k51_result_sha256"] = k51_result["sha256"]
    context["all_head_protocol_path"] = Path(all_head_protocol["path"]).resolve()
    context["all_head_protocol_sha256"] = all_head_protocol["sha256"]
    context["all_head_result_path"] = Path(all_head_result["path"]).resolve()
    context["all_head_result_sha256"] = all_head_result["sha256"]
    return context, training, expected


def _schedule_counters(
    schedule: Mapping[str, Any],
    *,
    positions: int,
    model: Mapping[str, int],
    mask: np.ndarray,
    k: int,
    resource: Mapping[str, Any],
) -> dict[str, int]:
    rows = schedule.get("rows")
    episodic._validate_schedule_rows(rows)
    frozen = _validate_candidate_mask(mask, k)
    if (
        isinstance(positions, bool)
        or not isinstance(positions, int)
        or not 0 <= positions <= _POSITIONS
    ):
        raise ValueError("retrieval episodic rank counter position is invalid")
    writes = [
        int(row["write_slot"]) for row in rows[:positions] if row["write_slot"] >= 0
    ]
    reads = [int(row["read_span"]) for row in rows[:positions] if row["read_span"] >= 0]
    active_layers = int(np.count_nonzero(np.any(frozen, axis=1)))
    key_value_width = int(model["key_value_heads"]) * int(model["head_dimension"])
    return {
        "episodic_slots_written": len(writes) * active_layers,
        "episodic_read_events": len(reads) * active_layers,
        "episodic_active_slots": len(set(writes)) * active_layers,
        "episodic_entries_read": len(reads) * _SPAN_TOKENS * k,
        "episodic_write_bytes": (
            len(writes) * active_layers * 2 * key_value_width * _CACHE_DTYPE_BYTES
        ),
        "episodic_key_read_bytes": (
            len(reads)
            * _SPAN_TOKENS
            * k
            * int(model["head_dimension"])
            * _CACHE_DTYPE_BYTES
        ),
        "episodic_value_read_bytes": (
            len(reads)
            * _SPAN_TOKENS
            * k
            * int(model["head_dimension"])
            * _CACHE_DTYPE_BYTES
        ),
        "episodic_state_bytes": int(resource["combined_state_bytes"]),
        "episodic_scratch_bytes": int(resource["combined_scratch_bytes"]),
    }


def _counter_checks(
    metrics: Mapping[str, int],
    *,
    context: Mapping[str, Any],
    schedule: Mapping[str, Any],
    positions: int,
    mask: np.ndarray,
    k: int,
    resource: Mapping[str, Any],
) -> dict[str, bool]:
    base = episodic.sustained._attention_expectations(
        dict(context["model"]),
        retrieval._BASE_POLICY,
        positions=positions,
    )
    expected = _schedule_counters(
        schedule,
        positions=positions,
        model=context["model"],
        mask=mask,
        k=k,
        resource=resource,
    )
    checks = {
        name: int(metrics.get(name, -1)) == value for name, value in expected.items()
    }
    duplicate = int(metrics.get("episodic_duplicate_older_entries_suppressed", -1))
    expected_selected = int(base["attention_older_selected_entries"])
    actual_selected = int(metrics.get("attention_older_selected_entries", -1))
    selected_loss = expected_selected - actual_selected
    logical_expected = (
        int(base["attention_logical_read_bytes"])
        + expected["episodic_key_read_bytes"]
        + expected["episodic_value_read_bytes"]
        - selected_loss * int(context["model"]["head_dimension"]) * 4
    )
    exact_base_names = (
        "positions_processed",
        "attention_eviction_events",
        "attention_older_candidate_entries_scored",
        "attention_sink_insertions",
    )
    checks.update(
        {
            name: int(metrics.get(name, -1)) == int(base[name])
            for name in exact_base_names
        }
    )
    checks.update(
        {
            "cache_position": positions == int(base["positions_processed"]),
            "attention_state_bytes": (
                int(metrics.get("attention_state_bytes", -1))
                == int(resource["combined_state_bytes"])
            ),
            "attention_scratch_bytes": (
                int(metrics.get("attention_scratch_bytes", -1))
                == int(resource["combined_scratch_bytes"])
            ),
            "attention_older_selected_entries": (0 <= selected_loss <= duplicate),
            "attention_logical_read_bytes": (
                int(metrics.get("attention_logical_read_bytes", -1)) == logical_expected
            ),
            "q7_scheduled_bytes": (
                int(metrics.get("q7_scheduled_bytes", -1))
                == positions
                * int(context["q7_expectations"]["scheduled_bytes_per_position"])
            ),
            "episodic_duplicate_older_entries_suppressed": (
                0 <= duplicate <= expected["episodic_entries_read"]
            ),
        }
    )
    heavy = int(metrics.get("attention_heavy_hitter_updates", -1))
    checks["attention_heavy_hitter_updates"] = (
        int(base["attention_heavy_hitter_updates_minimum"])
        <= heavy
        <= int(base["attention_heavy_hitter_updates_maximum"])
    )
    return checks


def _open_runtime(
    context: Mapping[str, Any],
    mask: np.ndarray,
) -> Any:
    policy = retrieval._BASE_POLICY
    return OLMoENativeTokenRuntime(
        context["config_path"],
        context["non_mlp_path"],
        context["q7_path"],
        context["episodic_library_path"],
        threads=_THREADS,
        local_window=policy["local_window"],
        older_candidates=policy["older_candidates"],
        older_top_k=policy["older_top_k"],
        sink_tokens=policy["sink_tokens"],
        episodic_policy={"slots": _SLOTS, "span_size": _SPAN_TOKENS},
        episodic_head_mask=mask,
    )


def _execute_record(
    runtime: Any,
    *,
    record: Mapping[str, Any],
    context: Mapping[str, Any],
    schedule: Mapping[str, Any],
    mask: np.ndarray,
    k: int,
    resource: Mapping[str, Any],
    progress_label: str | None = None,
) -> dict[str, Any]:
    if runtime.position != 0:
        runtime.reset()
    if (
        runtime.position != 0
        or not runtime.attention_metrics_available
        or not callable(getattr(runtime, "forward_episodic", None))
        or getattr(runtime, "episodic_policy", None)
        != {"slots": _SLOTS, "span_size": _SPAN_TOKENS}
        or not fixed._runtime_mask_matches(runtime, mask)
    ):
        raise ValueError("retrieval episodic rank runtime capability is unavailable")
    logits_rows: list[np.ndarray] = []
    top1_tokens: list[int] = []
    logit_digest = hashlib.sha256()
    counter_digest = hashlib.sha256()
    call_digest = hashlib.sha256()
    counter_stream: list[dict[str, Any]] = []
    final_metrics: dict[str, int] | None = None
    previous_duplicate_suppressions = 0
    started = time.perf_counter()
    for position, token_id in enumerate(record["input_ids"][:-1]):
        row = schedule["rows"][position]
        write_slots = [int(row["write_slot"])]
        read_spans = [int(row["read_span"])]
        result = runtime.forward_episodic(
            [int(token_id)],
            write_slots,
            read_spans,
        )
        if runtime.position != position + 1:
            raise ValueError("retrieval episodic rank position did not advance")
        final_metrics = dict(result.metrics)
        checks = _counter_checks(
            final_metrics,
            context=context,
            schedule=schedule,
            positions=position + 1,
            mask=mask,
            k=k,
            resource=resource,
        )
        duplicate_suppressions = int(
            final_metrics["episodic_duplicate_older_entries_suppressed"]
        )
        checks["episodic_duplicate_suppressions_monotonic"] = (
            duplicate_suppressions >= previous_duplicate_suppressions
        )
        previous_duplicate_suppressions = duplicate_suppressions
        counter_stream.append(
            {
                "position": position,
                "checks": checks,
                "passed": all(checks.values()),
                "episodic_metrics": {
                    name: final_metrics.get(name) for name in _EPISODIC_COUNTER_NAMES
                },
            }
        )
        episodic._counter_digest_update(counter_digest, final_metrics)
        call_digest.update(
            json.dumps(
                {
                    "position": position,
                    "write_slots": write_slots,
                    "read_spans": read_spans,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        call_digest.update(b"\n")
        top1_tokens.append(int(result.next_token))
        if position in record["answer_prediction_positions"]:
            hidden, logits = runtime.last_diagnostics()
            if (
                hidden.shape != (int(context["model"]["hidden_size"]),)
                or hidden.dtype != np.float32
                or not np.isfinite(hidden).all()
                or logits.shape != (int(context["model"]["vocab_size"]),)
                or logits.dtype != np.float32
                or not np.isfinite(logits).all()
                or int(np.argmax(logits)) != int(result.next_token)
            ):
                raise ValueError("retrieval episodic rank diagnostics are invalid")
            logits_rows.append(logits)
            logit_digest.update(logits.tobytes())
        if progress_label is not None and (position + 1) % 32 == 0:
            _progress(f"{progress_label}: position {position + 1}/{_POSITIONS}")
    if (
        final_metrics is None
        or runtime.position != _POSITIONS
        or len(logits_rows) != _ANSWER_POSITIONS
    ):
        raise ValueError("retrieval episodic rank record execution is incomplete")
    native_logits = np.stack(logits_rows).astype(np.float32, copy=False)
    targets = np.asarray(
        record["input_ids"][_ANSWER_START + 1 :],
        dtype=np.int64,
    )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for episodic rank evaluation"
        ) from exc
    with torch.inference_mode():
        answer_cross_entropy = float(
            retrieval._answer_cross_entropy(
                torch.from_numpy(native_logits).unsqueeze(0),
                torch.from_numpy(targets).unsqueeze(0),
            ).item()
        )
    return {
        "record_index": int(record["record_index"]),
        "record_id": record["record_id"],
        "K": k,
        "candidate_mask_sha256": _EXPECTED_MASK_SHA256[k],
        "answer_cross_entropy": answer_cross_entropy,
        "top1_tokens": top1_tokens,
        "logits_sha256": logit_digest.hexdigest(),
        "counter_stream_sha256": counter_digest.hexdigest(),
        "episodic_call_stream_sha256": call_digest.hexdigest(),
        "schedule_rows_sha256": schedule["rows_sha256"],
        "counter_stream": counter_stream,
        "counter_stream_passed": all(row["passed"] for row in counter_stream),
        "final_metrics": final_metrics,
        "final_position": runtime.position,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _replay_checks(
    replay: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, bool]:
    replay_metrics = replay.get("final_metrics")
    reference_metrics = reference.get("final_metrics")
    checks = {
        "K": replay.get("K") == reference.get("K"),
        "candidate_mask_sha256": (
            replay.get("candidate_mask_sha256")
            == reference.get("candidate_mask_sha256")
        ),
        "top1_tokens": replay.get("top1_tokens") == reference.get("top1_tokens"),
        "logits_sha256": (
            replay.get("logits_sha256") == reference.get("logits_sha256")
        ),
        "counter_stream_sha256": (
            replay.get("counter_stream_sha256")
            == reference.get("counter_stream_sha256")
        ),
        "episodic_call_stream_sha256": (
            replay.get("episodic_call_stream_sha256")
            == reference.get("episodic_call_stream_sha256")
        ),
        "schedule_rows_sha256": (
            replay.get("schedule_rows_sha256") == reference.get("schedule_rows_sha256")
        ),
        "answer_cross_entropy": (
            replay.get("answer_cross_entropy") == reference.get("answer_cross_entropy")
        ),
        "counter_stream_passed": replay.get("counter_stream_passed") is True,
        "deterministic_final_metrics": (
            isinstance(replay_metrics, dict)
            and isinstance(reference_metrics, dict)
            and episodic.sustained._deterministic_metrics(replay_metrics)
            == episodic.sustained._deterministic_metrics(reference_metrics)
        ),
    }
    checks["passed"] = all(checks.values())
    return checks


def _loss_gate(
    *,
    records: Sequence[Mapping[str, Any]],
    baselines: Mapping[str, Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    k: int,
) -> dict[str, Any]:
    m2 = baselines["M2"]["record_answer_cross_entropy"]
    m0 = baselines["M0"]["record_answer_cross_entropy"]
    if (
        len(records) != _RECORDS
        or len(m2) != _RECORDS
        or len(m0) != _RECORDS
        or len(evidence) != _RECORDS
        or [row.get("record_index") for row in evidence] != list(range(_RECORDS))
        or [row.get("record_id") for row in evidence]
        != [row.get("record_id") for row in records]
        or any(row.get("K") != k for row in evidence)
    ):
        raise ValueError(f"retrieval episodic rank K{k} loss population is invalid")
    candidate = [float(row["answer_cross_entropy"]) for row in evidence]
    if any(not np.isfinite(value) or value < 0.0 for value in candidate):
        raise ValueError(f"retrieval episodic rank K{k} loss is invalid")
    matrix = [
        {
            "record_index": index,
            "record_id": record["record_id"],
            "same_policy_M0_answer_cross_entropy": float(m0[index]),
            "fixed_M2_answer_cross_entropy": float(m2[index]),
            "candidate_answer_cross_entropy": candidate[index],
            "candidate_minus_same_policy_M0": (candidate[index] - float(m0[index])),
            "candidate_minus_fixed_M2": (candidate[index] - float(m2[index])),
            "no_fixed_M2_record_regression": (candidate[index] <= float(m2[index])),
        }
        for index, record in enumerate(records)
    ]
    m2_maximum = float(max(m2))
    m2_mean = float(np.mean(m2, dtype=np.float64))
    m0_maximum = float(max(m0))
    m0_mean = float(np.mean(m0, dtype=np.float64))
    candidate_maximum = float(max(candidate))
    candidate_mean = float(np.mean(candidate, dtype=np.float64))
    checks = {
        "complete_record_population": len(matrix) == _RECORDS,
        "maximum_answer_cross_entropy_strictly_improved": (
            candidate_maximum < m2_maximum
        ),
        "mean_answer_cross_entropy_strictly_improved": (candidate_mean < m2_mean),
        "no_record_regression": all(
            row["no_fixed_M2_record_regression"] for row in matrix
        ),
    }
    return {
        "K": k,
        "matrix": matrix,
        "summaries": {
            "same_policy_M0_attribution": {
                "maximum_answer_cross_entropy": m0_maximum,
                "mean_answer_cross_entropy": m0_mean,
            },
            "fixed_M2_reference": {
                "maximum_answer_cross_entropy": m2_maximum,
                "mean_answer_cross_entropy": m2_mean,
            },
            "candidate": {
                "maximum_answer_cross_entropy": candidate_maximum,
                "mean_answer_cross_entropy": candidate_mean,
                "maximum_minus_fixed_M2": candidate_maximum - m2_maximum,
                "mean_minus_fixed_M2": candidate_mean - m2_mean,
            },
        },
        "gate_checks": checks,
        "passed": all(checks.values()),
    }


def _population_resource_checks(
    evidence: Sequence[Mapping[str, Any]],
    *,
    schedule: Mapping[str, Any],
    context: Mapping[str, Any],
    mask: np.ndarray,
    k: int,
    resource: Mapping[str, Any],
) -> dict[str, bool]:
    final_expected = _schedule_counters(
        schedule,
        positions=_POSITIONS,
        model=context["model"],
        mask=mask,
        k=k,
        resource=resource,
    )
    return {
        "all_sequence_counter_streams": all(
            row["counter_stream_passed"] for row in evidence
        ),
        "exact_final_episodic_counters": all(
            all(
                int(row["final_metrics"].get(name, -1)) == expected
                for name, expected in final_expected.items()
            )
            for row in evidence
        ),
        "duplicate_suppression_bounded": all(
            0
            <= int(row["final_metrics"]["episodic_duplicate_older_entries_suppressed"])
            <= final_expected["episodic_entries_read"]
            for row in evidence
        ),
        "measured_read_at_or_below_analytic_upper_bound": all(
            int(row["final_metrics"]["attention_logical_read_bytes"])
            <= int(resource["combined_attention_and_episodic_read_bytes"])
            for row in evidence
        ),
        "measured_total_traffic_at_or_below_budget": all(
            (
                int(row["final_metrics"]["attention_logical_read_bytes"])
                + int(row["final_metrics"]["episodic_write_bytes"])
            )
            <= int(resource["combined_attention_and_episodic_traffic_bytes"])
            for row in evidence
        ),
        "combined_read_budget": resource["within_read_budget"] is True,
        "combined_total_traffic_budget": (
            resource["within_total_traffic_budget"] is True
        ),
    }


def _candidate_population(
    runtime: Any,
    *,
    context: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    schedules: Sequence[Mapping[str, Any]],
    baselines: Mapping[str, Mapping[str, Any]],
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    k = int(descriptor["K"])
    mask = _validate_candidate_mask(descriptor["head_mask"]["mask"], k)
    resource = descriptor["resource_contract"]
    evidence = [
        _execute_record(
            runtime,
            record=record,
            context=context,
            schedule=schedule,
            mask=mask,
            k=k,
            resource=resource,
            progress_label=(
                f"rank K{k} train record {int(record['record_index']) + 1}/{_RECORDS}"
            ),
        )
        for record, schedule in zip(records, schedules, strict=True)
    ]
    loss_gate = _loss_gate(
        records=records,
        baselines=baselines,
        evidence=evidence,
        k=k,
    )
    resource_checks = _population_resource_checks(
        evidence,
        schedule=schedules[0],
        context=context,
        mask=mask,
        k=k,
        resource=resource,
    )
    resource_passed = all(resource_checks.values())
    return {
        "K": k,
        "role": "ranked_prefix_head_gated_payload_candidate",
        "head_mask": descriptor["head_mask"],
        "resource_contract": resource,
        "population_native_sequence_forwards": _RECORDS,
        "population_native_token_steps": _RECORDS * _POSITIONS,
        "sequence_evidence": evidence,
        "loss_gate": loss_gate,
        "population_resource_checks": resource_checks,
        "population_resource_passed": resource_passed,
        "pre_replay_passed": loss_gate["passed"] and resource_passed,
        "reset_replay": {
            "executed": False,
            "native_sequence_forwards": 0,
        },
        "passed": False,
    }


def _attach_reset_replay(
    candidate: dict[str, Any],
    runtime: Any,
    *,
    context: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    schedules: Sequence[Mapping[str, Any]],
) -> None:
    k = int(candidate["K"])
    mask = _validate_candidate_mask(candidate["head_mask"]["mask"], k)
    replay = _execute_record(
        runtime,
        record=records[0],
        context=context,
        schedule=schedules[0],
        mask=mask,
        k=k,
        resource=candidate["resource_contract"],
        progress_label=f"rank K{k} retained-runtime reset replay",
    )
    checks = _replay_checks(replay, candidate["sequence_evidence"][0])
    candidate["reset_replay"] = {
        "executed": True,
        "native_sequence_forwards": 1,
        "native_token_steps": _POSITIONS,
        "checks": checks,
        "passed": checks["passed"],
    }
    candidate["passed"] = candidate["pre_replay_passed"] and checks["passed"]


def _run_sweep(
    *,
    context: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    baselines: Mapping[str, Mapping[str, Any]],
    runtime_factory: Callable[[Mapping[str, Any], np.ndarray], Any] = (_open_runtime),
) -> dict[str, Any]:
    if len(records) != _RECORDS or [row.get("record_index") for row in records] != list(
        range(_RECORDS)
    ):
        raise ValueError("retrieval episodic rank train population is invalid")
    if protocol.get("candidate_order") != list(_CANDIDATE_K):
        raise ValueError("retrieval episodic rank candidate order changed")
    descriptors = protocol.get("candidates")
    if not isinstance(descriptors, list) or [
        row.get("K") for row in descriptors
    ] != list(_CANDIDATE_K):
        raise ValueError("retrieval episodic rank candidate descriptors changed")
    anchors = protocol["tokenizer_fact_anchor_ids"]
    schedules = [
        fixed._derive_schedule(record["input_ids"], anchors) for record in records
    ]
    if [schedule["rows_sha256"] for schedule in schedules] != protocol[
        "schedule_contract"
    ]["per_record_rows_sha256"]:
        raise ValueError("retrieval episodic rank schedule authentication failed")
    outcomes: dict[int, dict[str, Any]] = {}
    manifest: list[dict[str, Any]] = []
    retained_runtime: Any | None = None
    retained_k: int | None = None
    retained_key: tuple[float, float, int] | None = None
    active_runtime: Any | None = None
    selected_k: int | None = None
    selection_role: str | None = None
    try:
        for candidate_index, descriptor in enumerate(descriptors):
            k = int(descriptor["K"])
            mask = _validate_candidate_mask(
                descriptor["head_mask"]["mask"],
                k,
            )
            active_runtime = runtime_factory(context, mask)
            outcome = _candidate_population(
                active_runtime,
                context=context,
                records=records,
                schedules=schedules,
                baselines=baselines,
                descriptor=descriptor,
            )
            outcomes[k] = outcome
            manifest.append(
                {
                    "K": k,
                    "executed": True,
                    "status": (
                        "strict_gate_passed"
                        if outcome["pre_replay_passed"]
                        else "strict_gate_failed"
                    ),
                    "population_native_sequence_forwards": _RECORDS,
                    "reset_replay_executed": False,
                }
            )
            if not outcome["population_resource_passed"]:
                raise ValueError(
                    f"retrieval episodic rank K{k} systems contract failed"
                )
            summary = outcome["loss_gate"]["summaries"]["candidate"]
            key = (
                float(summary["maximum_answer_cross_entropy"]),
                float(summary["mean_answer_cross_entropy"]),
                k,
            )
            if outcome["pre_replay_passed"]:
                if retained_runtime is not None:
                    retained_runtime.close()
                    retained_runtime = None
                    retained_k = None
                    retained_key = None
                _attach_reset_replay(
                    outcome,
                    active_runtime,
                    context=context,
                    records=records,
                    schedules=schedules,
                )
                if not outcome["reset_replay"]["passed"]:
                    raise ValueError(
                        f"retrieval episodic rank K{k} reset replay failed"
                    )
                active_runtime.close()
                active_runtime = None
                selected_k = k
                selection_role = "smallest_passing_candidate"
                manifest[-1]["status"] = (
                    "selected_smallest_passing"
                    if outcome["passed"]
                    else "selected_gate_pass_but_reset_replay_failed"
                )
                manifest[-1]["reset_replay_executed"] = True
                for skipped in descriptors[candidate_index + 1 :]:
                    manifest.append(
                        {
                            "K": int(skipped["K"]),
                            "executed": False,
                            "status": "skipped_after_smallest_passing_gate",
                            "population_native_sequence_forwards": 0,
                            "reset_replay_executed": False,
                        }
                    )
                break
            if retained_key is None or key < retained_key:
                if retained_runtime is not None:
                    retained_runtime.close()
                retained_runtime = active_runtime
                retained_k = k
                retained_key = key
                active_runtime = None
            else:
                active_runtime.close()
                active_runtime = None
        else:
            if retained_runtime is None or retained_k is None:
                raise ValueError(
                    "retrieval episodic rank failed to retain best candidate"
                )
            selected_k = retained_k
            selection_role = "best_failed_candidate_for_diagnostic_replay"
            selected = outcomes[selected_k]
            _attach_reset_replay(
                selected,
                retained_runtime,
                context=context,
                records=records,
                schedules=schedules,
            )
            if not selected["reset_replay"]["passed"]:
                raise ValueError(
                    f"retrieval episodic rank K{selected_k} reset replay failed"
                )
            retained_runtime.close()
            retained_runtime = None
            for row in manifest:
                if row["K"] == selected_k:
                    row["status"] = "best_failed_candidate_replayed"
                    row["reset_replay_executed"] = True
                    break
    finally:
        if active_runtime is not None:
            active_runtime.close()
        if retained_runtime is not None:
            retained_runtime.close()
    if (
        selected_k is None
        or selection_role is None
        or len(manifest) != len(_CANDIDATE_K)
    ):
        raise ValueError("retrieval episodic rank sweep execution is incomplete")
    selected = outcomes[selected_k]
    executed = [row["K"] for row in manifest if row["executed"]]
    skipped = [row["K"] for row in manifest if not row["executed"]]
    replay_count = sum(
        int(outcome["reset_replay"].get("native_sequence_forwards", 0))
        for outcome in outcomes.values()
    )
    passed = selection_role == "smallest_passing_candidate" and selected["passed"]
    return {
        "candidate_order": list(_CANDIDATE_K),
        "execution_manifest": manifest,
        "executed_candidates": executed,
        "skipped_candidates": skipped,
        "candidate_outcomes": {f"K{k}": outcomes[k] for k in executed},
        "selected_K": selected_k,
        "selection_role": selection_role,
        "selection_key": [
            float(
                selected["loss_gate"]["summaries"]["candidate"][
                    "maximum_answer_cross_entropy"
                ]
            ),
            float(
                selected["loss_gate"]["summaries"]["candidate"][
                    "mean_answer_cross_entropy"
                ]
            ),
            selected_k,
        ],
        "population_native_sequence_forwards": len(executed) * _RECORDS,
        "reset_replay_native_sequence_forwards": replay_count,
        "total_native_sequence_forwards": (len(executed) * _RECORDS + replay_count),
        "passed": passed,
    }


def _post_authentication(
    context: Mapping[str, Any],
    training_checkpoint: Mapping[str, Any],
) -> dict[str, bool]:
    protocol = context["protocol"]
    source = protocol["source_model"]
    checks = {
        "base_protocol": (
            sha256_file(context["protocol_path"]) == context["protocol_sha256"]
        ),
        "package": (
            validate_olmoe_native_package(
                context["package_path"],
                expected_manifest_sha256=protocol["package"]["manifest_sha256"],
            )
            == context["manifest"]
        ),
        "corpus_manifest": (
            sha256_file(context["manifest_path"])
            == protocol["corpus"]["manifest_sha256"]
        ),
        "train_split": (
            sha256_file(context["split_paths"]["train"])
            == protocol["corpus"]["splits"]["train"]["sha256"]
        ),
        "confirmation_not_opened": context[
            "confirmation_descriptor_authenticated_without_file_access"
        ],
        "source_config": (
            sha256_file(context["model_path"] / "config.json")
            == source["config_sha256"]
        ),
        "source_index": (
            sha256_file(context["model_path"] / "model.safetensors.index.json")
            == source["index_sha256"]
        ),
        "source_shards": (
            retrieval._source_shard_inventory(context["model_path"])
            == source["shard_sha256"]
        ),
        "expert_proxy": (
            sha256_file(context["proxy_path"])
            == protocol["expert_proxy_qualifier"]["sha256"]
        ),
        "headwise_episodic_library": (
            sha256_file(context["episodic_library_path"])
            == context["episodic_library_sha256"]
        ),
        "K51_protocol": (
            sha256_file(context["k51_protocol_path"]) == context["k51_protocol_sha256"]
        ),
        "K51_result": (
            sha256_file(context["k51_result_path"]) == context["k51_result_sha256"]
        ),
        "all_head_protocol": (
            sha256_file(context["all_head_protocol_path"])
            == context["all_head_protocol_sha256"]
        ),
        "all_head_result": (
            sha256_file(context["all_head_result_path"])
            == context["all_head_result_sha256"]
        ),
        "historical_all_head_library": (
            sha256_file(
                Path(
                    context["rank_sweep_protocol"]["all_head_K256_attribution"][
                        "historical_episodic_library"
                    ]["path"]
                )
            )
            == context["rank_sweep_protocol"]["all_head_K256_attribution"][
                "historical_episodic_library"
            ]["sha256"]
        ),
        "rank_sweep_protocol": (
            sha256_file(context["rank_sweep_protocol_path"])
            == context["rank_sweep_protocol_sha256"]
        ),
        "rank_sweep_source_inventory": (
            context["rank_sweep_protocol"]["source_sha256"] == _source_inventory()
        ),
        "training_checkpoint": (
            sha256_file(Path(training_checkpoint["path"]))
            == training_checkpoint["sha256"]
        ),
    }
    for name, path in context["library_paths"].items():
        checks[f"{name}_library"] = (
            sha256_file(path) == protocol["libraries"][name]["sha256"]
        )
    return checks


def screen_episodic_rank_sweep(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    output = retrieval._new_output(
        out,
        "retrieval episodic rank-sweep result",
    )
    started = time.perf_counter()
    _progress("authenticating frozen rank-sweep and K51 prerequisite")
    context, training, frozen = _authenticate_protocol(
        protocol,
        protocol_sha256,
    )
    state = fixed._checkpoint_references(training)
    _progress("starting ordered CPU-native ranked-prefix sweep")
    sweep = _run_sweep(
        context=context,
        records=context["train_records"],
        protocol=frozen,
        baselines=state["baselines"],
    )
    post_authentication = _post_authentication(
        context,
        frozen["training_checkpoint"],
    )
    if not post_authentication or not all(post_authentication.values()):
        raise ValueError("retrieval episodic rank-sweep post-run authentication failed")
    report = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _RESULT_EXPERIMENT,
        "status": (
            "train_episodic_rank_sweep_gate_passed"
            if sweep["passed"]
            else "train_episodic_rank_sweep_gate_failed"
        ),
        "protocol": {
            "path": str(context["rank_sweep_protocol_path"]),
            "sha256": context["rank_sweep_protocol_sha256"],
        },
        "scope": {
            "split": "train",
            "threads": _THREADS,
            "device": "cpu",
            "dense_teacher_forwards": 0,
            "candidate_masks_fitted_by_this_experiment": False,
            "candidate_selection_uses_train_outcomes": True,
            "development_outcomes_used": False,
            "confirmation_split_opened": False,
        },
        "reused_checkpoint_evidence": state["baselines"],
        "K51_attribution": frozen["K51_prerequisite"],
        "K256_attribution": frozen["all_head_K256_attribution"],
        "K32_status": frozen["attribution_contract"]["K32"],
        "rank_sweep": sweep,
        "decision": {
            "passed": sweep["passed"],
            "semantic_gate_passed": False,
            "confirmation_authorized": False,
            "next_step": (
                "freeze a distinct dense-teacher semantic development screen"
                if sweep["passed"]
                else "reject this ranked-prefix payload-cache family"
            ),
        },
        "post_run_authentication": post_authentication,
        "confirmation_split_opened": False,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output, report)
    _progress(f"result written to {output}")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Train-only OLMoE head-gated episodic ranked-prefix sweep"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--base-protocol", required=True)
    freeze.add_argument("--base-protocol-sha256", required=True)
    freeze.add_argument("--training-checkpoint", required=True)
    freeze.add_argument("--training-checkpoint-sha256", required=True)
    freeze.add_argument("--headwise-episodic-library", required=True)
    freeze.add_argument("--headwise-episodic-library-sha256", required=True)
    freeze.add_argument("--k51-protocol", required=True)
    freeze.add_argument("--k51-protocol-sha256", required=True)
    freeze.add_argument("--k51-result", required=True)
    freeze.add_argument("--k51-result-sha256", required=True)
    freeze.add_argument("--all-head-protocol", required=True)
    freeze.add_argument("--all-head-protocol-sha256", required=True)
    freeze.add_argument("--all-head-result", required=True)
    freeze.add_argument("--all-head-result-sha256", required=True)
    freeze.add_argument("--out", required=True)
    screen = commands.add_parser("screen")
    screen.add_argument("--protocol", required=True)
    screen.add_argument("--protocol-sha256", required=True)
    screen.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "freeze":
        value = freeze_episodic_rank_sweep_protocol(
            base_protocol=args.base_protocol,
            base_protocol_sha256=args.base_protocol_sha256,
            training_checkpoint=args.training_checkpoint,
            training_checkpoint_sha256=args.training_checkpoint_sha256,
            headwise_episodic_library=args.headwise_episodic_library,
            headwise_episodic_library_sha256=(args.headwise_episodic_library_sha256),
            k51_protocol=args.k51_protocol,
            k51_protocol_sha256=args.k51_protocol_sha256,
            k51_result=args.k51_result,
            k51_result_sha256=args.k51_result_sha256,
            all_head_protocol=args.all_head_protocol,
            all_head_protocol_sha256=args.all_head_protocol_sha256,
            all_head_result=args.all_head_result,
            all_head_result_sha256=args.all_head_result_sha256,
            out=args.out,
        )
    elif args.command == "screen":
        value = screen_episodic_rank_sweep(
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            out=args.out,
        )
    else:  # pragma: no cover - argparse owns this boundary
        raise AssertionError("unknown retrieval episodic rank-sweep command")
    print(json.dumps(value, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())
