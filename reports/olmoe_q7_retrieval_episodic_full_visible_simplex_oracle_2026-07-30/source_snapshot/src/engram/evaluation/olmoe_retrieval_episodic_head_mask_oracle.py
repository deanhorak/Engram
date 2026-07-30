"""Train-only fixed-M2 head-gated episodic payload experiment.

This evaluator reuses the original causal payload-only schedule: four
eight-token source payloads are written into 32 canonical slots and the
matching span is read at each eight-token answer block.  Unlike the all-head
episodic diagnostic, reads are enabled only for the authenticated M2
51-of-256 layer/head mask.  No selector is fit by this experiment.

The frozen candidate-only screen compares native answer cross-entropy against
the authenticated M2 train checkpoint.  It requires strict mean and worst
record improvements with no per-record regression, exact analytic resource
and counter agreement, and deterministic reset replay.  M0 is reported only
as same-policy attribution.  No dense teacher is loaded, no development
outcome is consumed, and the sealed confirmation file is never opened or
hashed.
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

import engram.evaluation.olmoe_retrieval_episodic_oracle as episodic
import engram.evaluation.olmoe_retrieval_head_selector as retrieval
from engram.compiler.olmoe_native import validate_olmoe_native_package
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.utils import atomic_json, sha256_file, sha256_json


_SCHEMA_VERSION = 1
_PROTOCOL_EXPERIMENT = "olmoe_q7_retrieval_episodic_head_mask_oracle_protocol"
_RESULT_EXPERIMENT = "olmoe_q7_retrieval_episodic_head_mask_oracle_train_screen"
_PROTOCOL_STATUS = "frozen_before_fixed_M2_train_candidate_execution"
_RECORDS = 8
_POSITIONS = 128
_ANSWER_START = 96
_ANSWER_POSITIONS = 32
_FACT_SPANS = 4
_SPAN_TOKENS = 8
_SLOTS = 32
_CACHE_DTYPE_BYTES = 2
_POSITION_DTYPE_BYTES = 8
_SCRATCH_DTYPE_BYTES = 4
_THREADS = 12
_MAXIMUM_TRAFFIC_FRACTION = 0.45
_EXPECTED_M2_MASK_SHA256 = (
    "49802a2d37abd44e4015e87633c9a321e333315b9400f6a69d4713ec2270b446"
)
_EXPECTED_LAYER_HEAD_COUNTS = (3, 3, 1, 4, 0, 7, 7, 4, 1, 6, 4, 1, 5, 3, 2, 0)
_EXPECTED_ACTIVE_LAYERS = (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
_EXPECTED_SELECTED_PAIRS = 51
_EXPECTED_POSITIVE_PROJECTED_SCORES = 165
_FUTURE_RANK_SWEEP_BOUNDARIES = (32, 64, 96, 128, 165)
_EPISODIC_COUNTER_NAMES = episodic._EPISODIC_COUNTER_NAMES
_SOURCE_FILES = (
    *episodic._SOURCE_FILES,
    "src/engram/evaluation/olmoe_retrieval_episodic_head_mask_oracle.py",
)


def _progress(message: str) -> None:
    print(
        f"[retrieval-episodic-head-mask-oracle] {message}",
        file=sys.stderr,
        flush=True,
    )


def _source_inventory() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    return {name: sha256_file(repository / name) for name in _SOURCE_FILES}


def _derive_schedule(
    input_ids: Sequence[int],
    anchors: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    return episodic._derive_schedule(input_ids, anchors)


def _fact_anchor_ids(tokenizer_path: str | Path) -> dict[str, tuple[int, ...]]:
    return episodic._fact_anchor_ids(tokenizer_path)


def _validate_fixed_m2_mask(value: Any) -> np.ndarray:
    if (
        isinstance(value, np.ndarray)
        and value.shape == (retrieval._LAYERS, retrieval._HEADS)
        and value.dtype == np.bool_
    ):
        mask = value.copy()
    else:
        mask = retrieval._boolean_mask(
            value,
            "retrieval episodic fixed M2 head mask",
        )
    counts = tuple(int(row.sum()) for row in mask)
    active_layers = tuple(layer for layer, count in enumerate(counts) if count > 0)
    if (
        int(mask.sum()) != _EXPECTED_SELECTED_PAIRS
        or counts != _EXPECTED_LAYER_HEAD_COUNTS
        or active_layers != _EXPECTED_ACTIVE_LAYERS
        or sha256_json(mask.tolist()) != _EXPECTED_M2_MASK_SHA256
    ):
        raise ValueError("retrieval episodic fixed M2 head mask contract changed")
    return mask


def _projected_score_ordering(
    scores: Any,
    mask: np.ndarray,
) -> dict[str, Any]:
    values = retrieval._finite_matrix(
        scores,
        "retrieval episodic M2 projected scores",
    )
    flat = values.reshape(-1)
    order = sorted(
        range(flat.size),
        key=lambda index: (-float(flat[index]), index),
    )
    selected = set(int(index) for index in np.flatnonzero(mask.reshape(-1)))
    if set(order[:_EXPECTED_SELECTED_PAIRS]) != selected:
        raise ValueError("retrieval episodic M2 projected-score ordering changed")
    positive_count = int(np.count_nonzero(flat > 0.0))
    if positive_count != _EXPECTED_POSITIVE_PROJECTED_SCORES:
        raise ValueError("retrieval episodic M2 positive-score boundary changed")
    rows = [
        {
            "rank": rank,
            "layer": index // retrieval._HEADS,
            "head": index % retrieval._HEADS,
            "layer_major_index": index,
            "projected_score": float(flat[index]),
            "selected_in_fixed_M2": index in selected,
        }
        for rank, index in enumerate(order, start=1)
    ]
    return {
        "ordering": rows,
        "ordering_sha256": sha256_json(rows),
        "positive_score_count": positive_count,
        "future_rank_sweep_boundaries_not_executed": list(
            _FUTURE_RANK_SWEEP_BOUNDARIES
        ),
        "positive_score_boundary_is_not_a_benefit_claim": True,
    }


def _checkpoint_references(training: Mapping[str, Any]) -> dict[str, Any]:
    entries = training.get("masks")
    if not isinstance(entries, Mapping):
        raise ValueError("retrieval episodic head-mask checkpoint masks are missing")
    result: dict[str, Any] = {}
    masks: dict[str, np.ndarray] = {}
    for name, expected_count in (("M0", 0), ("M2", _EXPECTED_SELECTED_PAIRS)):
        entry = entries.get(name)
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"retrieval episodic head-mask checkpoint {name} is missing"
            )
        mask = retrieval._boolean_mask(
            entry.get("mask"),
            f"retrieval episodic head-mask {name}",
        )
        if name == "M2":
            mask = _validate_fixed_m2_mask(mask)
        rows = entry.get("records")
        if (
            int(mask.sum()) != expected_count
            or not isinstance(rows, list)
            or len(rows) != _RECORDS
            or [row.get("record_index") for row in rows] != list(range(_RECORDS))
        ):
            raise ValueError(
                f"retrieval episodic head-mask checkpoint {name} is invalid"
            )
        losses: list[float] = []
        record_ids: list[str] = []
        for row in rows:
            loss = row.get("loss")
            value = (
                loss.get("answer_cross_entropy") if isinstance(loss, Mapping) else None
            )
            record_id = row.get("record_id")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                or float(value) < 0.0
                or not isinstance(record_id, str)
                or not record_id
            ):
                raise ValueError(
                    f"retrieval episodic head-mask checkpoint {name} record is invalid"
                )
            losses.append(float(value))
            record_ids.append(record_id)
        masks[name] = mask
        result[name] = {
            "role": (
                "same_policy_W16_C8_K4_S2_no_episodic_cache_attribution"
                if name == "M0"
                else "fixed_exact_51_head_train_gate_reference"
            ),
            "mask_sha256": sha256_json(mask.tolist()),
            "selected_head_count": int(mask.sum()),
            "record_ids": record_ids,
            "record_answer_cross_entropy": losses,
            "maximum_answer_cross_entropy": float(max(losses)),
            "mean_answer_cross_entropy": float(np.mean(losses, dtype=np.float64)),
        }
    ordering = _projected_score_ordering(
        entries["M2"].get("projected_scores"),
        masks["M2"],
    )
    return {
        "mask": masks["M2"],
        "baselines": result,
        "projected_score_ordering": ordering,
    }


def _mask_descriptor(mask: np.ndarray) -> dict[str, Any]:
    frozen = _validate_fixed_m2_mask(mask)
    counts = [int(row.sum()) for row in frozen]
    return {
        "name": "M2",
        "mask": frozen.tolist(),
        "mask_sha256": sha256_json(frozen.tolist()),
        "selected_layer_head_pairs": int(frozen.sum()),
        "selected_heads_per_layer": counts,
        "active_layers": [layer for layer, count in enumerate(counts) if count > 0],
        "active_layer_count": len(_EXPECTED_ACTIVE_LAYERS),
        "fitted_by_this_experiment": False,
    }


def _resource_contract(
    model: Mapping[str, int],
    q7_expectations: Mapping[str, int],
    mask: np.ndarray,
) -> dict[str, Any]:
    frozen = _validate_fixed_m2_mask(mask)
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
        raise ValueError("retrieval episodic head-mask model geometry changed")
    active_layers = int(np.count_nonzero(np.any(frozen, axis=1)))
    selected_pairs = int(frozen.sum())
    kv_width = key_value_heads * head_dimension
    slot_payload_bytes_per_active_layer = 2 * kv_width * _CACHE_DTYPE_BYTES
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
        "fixed_head_mask": _mask_descriptor(frozen),
        "cache_payload": {
            "dtype": "bfloat16",
            "dtype_bytes": _CACHE_DTYPE_BYTES,
            "contents": ("full K/V layer rows allocated only for M2-active layers"),
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
            "plan_dtype": "int32",
            "plan_arrays_are_caller_memory": True,
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
    if (
        not contract["within_read_budget"]
        or not contract["within_total_traffic_budget"]
        or active_layers != len(_EXPECTED_ACTIVE_LAYERS)
        or selected_pairs != _EXPECTED_SELECTED_PAIRS
    ):
        raise ValueError(
            "retrieval episodic head-mask resource contract exceeds budget"
        )
    return contract


def _build_protocol(
    *,
    context: Mapping[str, Any],
    checkpoint_descriptor: Mapping[str, str],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    state = _checkpoint_references(training)
    mask = state["mask"]
    anchors = _fact_anchor_ids(context["tokenizer_path"])
    schedules = [
        _derive_schedule(record["input_ids"], anchors)
        for record in context["train_records"]
    ]
    expected_record_ids = [record["record_id"] for record in context["train_records"]]
    if any(
        baseline["record_ids"] != expected_record_ids
        for baseline in state["baselines"].values()
    ):
        raise ValueError("retrieval episodic head-mask checkpoint record IDs changed")
    return {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _PROTOCOL_EXPERIMENT,
        "status": _PROTOCOL_STATUS,
        "base_retrieval_protocol": {
            "path": str(context["protocol_path"]),
            "sha256": context["protocol_sha256"],
        },
        "training_checkpoint": {
            "path": checkpoint_descriptor["path"],
            "sha256": checkpoint_descriptor["sha256"],
            "training_sha256": sha256_json(training),
        },
        "headwise_episodic_library": {
            "path": str(context["episodic_library_path"]),
            "sha256": context["episodic_library_sha256"],
            "required_open_symbol": ("engram_olmoe_token_open_episodic_headwise_v1"),
        },
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
            "fixed_checkpoint_mask_only": True,
            "mask_fitting_or_selection": False,
            "dense_teacher_forwards": 0,
            "development_outcomes_used": False,
            "confirmation_file_access_permitted": False,
            "cpu_only_candidate_execution": True,
        },
        "fixed_M2_head_mask": _mask_descriptor(mask),
        "authenticated_M2_projected_score_ordering": state["projected_score_ordering"],
        "tokenizer_fact_anchor_ids": {
            label: list(values) for label, values in anchors.items()
        },
        "schedule_contract": {
            "derivation_input": ("input_ids[0:97] and authenticated tokenizer anchors"),
            "last_input_index_observed": _ANSWER_START,
            "write_source_starts": list(retrieval._PASSKEY_SOURCE_STARTS),
            "payload_spans": _FACT_SPANS,
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
            "constructor_arguments": [
                "episodic_policy",
                "episodic_head_mask",
            ],
            "episodic_policy": {
                "slots": _SLOTS,
                "span_size": _SPAN_TOKENS,
            },
            "episodic_head_mask_sha256": _EXPECTED_M2_MASK_SHA256,
            "required_open_symbol": ("engram_olmoe_token_open_episodic_headwise_v1"),
            "cumulative_metric_names": list(_EPISODIC_COUNTER_NAMES),
        },
        "resource_contract": _resource_contract(
            context["model"],
            context["q7_expectations"],
            mask,
        ),
        "train_loss_gate": {
            "primary_reference": ("authenticated fixed-M2 train answer cross-entropy"),
            "maximum_answer_cross_entropy_strictly_improves": True,
            "mean_answer_cross_entropy_strictly_improves": True,
            "per_record_regression_permitted": False,
            "M0_is_attribution_only": True,
        },
        "evaluation_contract": {
            "candidate_native_sequence_forwards": _RECORDS,
            "reset_replay_record_index": 0,
            "exact_per_position_analytic_counters_required": True,
            "duplicate_suppression_is_replay_exact_and_bounded": True,
            "semantic_gate_claimed_by_this_screen": False,
        },
        "reused_checkpoint_evidence": state["baselines"],
        "all_head_payload_attribution": {
            "consumed": False,
            "reason": (
                "the prior all-head result is not an input to this frozen "
                "candidate-only M2 comparison"
            ),
        },
        "confirmation_split_opened": False,
    }


def freeze_episodic_head_mask_oracle_protocol(
    *,
    base_protocol: str | Path,
    base_protocol_sha256: str,
    training_checkpoint: str | Path,
    training_checkpoint_sha256: str,
    headwise_episodic_library: str | Path,
    headwise_episodic_library_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    """Freeze the fixed-M2 experiment before native candidate execution."""

    output = retrieval._new_output(
        out,
        "retrieval episodic head-mask protocol",
    )
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
        raise ValueError(
            "retrieval episodic head-mask protocol requires eligible M2 evidence"
        )
    protocol = _build_protocol(
        context=context,
        checkpoint_descriptor=checkpoint,
        training=training,
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
        "retrieval episodic head-mask protocol",
    )
    value = retrieval._read_json(
        source,
        "retrieval episodic head-mask protocol",
    )
    base = value.get("base_retrieval_protocol")
    checkpoint_binding = value.get("training_checkpoint")
    library_binding = value.get("headwise_episodic_library")
    if not all(
        isinstance(binding, Mapping)
        for binding in (base, checkpoint_binding, library_binding)
    ):
        raise ValueError("retrieval episodic head-mask protocol bindings are invalid")
    context, training, selection, checkpoint = episodic._authenticate_base_inputs(
        base.get("path"),
        base.get("sha256"),
        checkpoint_binding.get("path"),
        checkpoint_binding.get("sha256"),
        library_binding.get("path"),
        library_binding.get("sha256"),
    )
    if (
        selection.get("screen_eligible") is not True
        or selection.get("selected_mask_name") != "M2"
    ):
        raise ValueError("retrieval episodic head-mask checkpoint selection changed")
    expected = _build_protocol(
        context=context,
        checkpoint_descriptor=checkpoint,
        training=training,
    )
    if value != expected:
        raise ValueError("retrieval episodic head-mask protocol contract changed")
    context = dict(context)
    context["head_mask_protocol_path"] = source
    context["head_mask_protocol_sha256"] = protocol_sha256.lower()
    context["head_mask_protocol"] = expected
    return context, training, expected


def _schedule_counters(
    schedule: Mapping[str, Any],
    *,
    positions: int,
    model: Mapping[str, int],
    mask: np.ndarray,
    resource: Mapping[str, Any],
) -> dict[str, int]:
    rows = schedule.get("rows")
    episodic._validate_schedule_rows(rows)
    frozen = _validate_fixed_m2_mask(mask)
    if (
        isinstance(positions, bool)
        or not isinstance(positions, int)
        or not 0 <= positions <= _POSITIONS
    ):
        raise ValueError("retrieval episodic head-mask counter position is invalid")
    writes = [
        int(row["write_slot"]) for row in rows[:positions] if row["write_slot"] >= 0
    ]
    reads = [int(row["read_span"]) for row in rows[:positions] if row["read_span"] >= 0]
    active_layers = int(np.count_nonzero(np.any(frozen, axis=1)))
    selected_pairs = int(frozen.sum())
    key_value_width = int(model["key_value_heads"]) * int(model["head_dimension"])
    return {
        "episodic_slots_written": len(writes) * active_layers,
        "episodic_read_events": len(reads) * active_layers,
        "episodic_active_slots": len(set(writes)) * active_layers,
        "episodic_entries_read": (len(reads) * _SPAN_TOKENS * selected_pairs),
        "episodic_write_bytes": (
            len(writes) * active_layers * 2 * key_value_width * _CACHE_DTYPE_BYTES
        ),
        "episodic_key_read_bytes": (
            len(reads)
            * _SPAN_TOKENS
            * selected_pairs
            * int(model["head_dimension"])
            * _CACHE_DTYPE_BYTES
        ),
        "episodic_value_read_bytes": (
            len(reads)
            * _SPAN_TOKENS
            * selected_pairs
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


def _open_runtime(context: Mapping[str, Any], mask: np.ndarray) -> Any:
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


def _runtime_mask_matches(runtime: Any, mask: np.ndarray) -> bool:
    observed = getattr(runtime, "episodic_head_mask", None)
    if observed is None:
        return False
    try:
        normalized = np.asarray(observed)
    except (TypeError, ValueError):
        return False
    return (
        normalized.shape == mask.shape
        and normalized.dtype.kind == "b"
        and np.array_equal(normalized, mask)
    )


def _execute_record(
    runtime: Any,
    *,
    record: Mapping[str, Any],
    context: Mapping[str, Any],
    schedule: Mapping[str, Any],
    mask: np.ndarray,
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
        or not _runtime_mask_matches(runtime, mask)
    ):
        raise ValueError(
            "retrieval episodic head-mask runtime capability is unavailable"
        )
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
            raise ValueError("retrieval episodic head-mask position did not advance")
        final_metrics = dict(result.metrics)
        checks = _counter_checks(
            final_metrics,
            context=context,
            schedule=schedule,
            positions=position + 1,
            mask=mask,
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
                raise ValueError("retrieval episodic head-mask diagnostics are invalid")
            logits_rows.append(logits)
            logit_digest.update(logits.tobytes())
        if progress_label is not None and (position + 1) % 32 == 0:
            _progress(f"{progress_label}: position {position + 1}/{_POSITIONS}")
    if (
        final_metrics is None
        or runtime.position != _POSITIONS
        or len(logits_rows) != _ANSWER_POSITIONS
    ):
        raise ValueError("retrieval episodic head-mask record execution is incomplete")
    native_logits = np.stack(logits_rows).astype(np.float32, copy=False)
    targets = np.asarray(
        record["input_ids"][_ANSWER_START + 1 :],
        dtype=np.int64,
    )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for episodic head-mask evaluation"
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
        "answer_cross_entropy": answer_cross_entropy,
        "top1_tokens": top1_tokens,
        "logits_sha256": logit_digest.hexdigest(),
        "counter_stream_sha256": counter_digest.hexdigest(),
        "episodic_call_stream_sha256": call_digest.hexdigest(),
        "schedule_rows_sha256": schedule["rows_sha256"],
        "fixed_M2_mask_sha256": _EXPECTED_M2_MASK_SHA256,
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
        "fixed_M2_mask_sha256": (
            replay.get("fixed_M2_mask_sha256") == reference.get("fixed_M2_mask_sha256")
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
) -> dict[str, Any]:
    m2_baseline = baselines["M2"]["record_answer_cross_entropy"]
    m0_attribution = baselines["M0"]["record_answer_cross_entropy"]
    if (
        len(records) != _RECORDS
        or len(m2_baseline) != _RECORDS
        or len(m0_attribution) != _RECORDS
        or len(evidence) != _RECORDS
        or [row.get("record_index") for row in evidence] != list(range(_RECORDS))
        or [row.get("record_id") for row in evidence]
        != [row.get("record_id") for row in records]
    ):
        raise ValueError("retrieval episodic head-mask loss population is invalid")
    candidate: list[float] = []
    matrix: list[dict[str, Any]] = []
    for index, (record, m2_value, m0_value, row) in enumerate(
        zip(
            records,
            m2_baseline,
            m0_attribution,
            evidence,
            strict=True,
        )
    ):
        value = row.get("answer_cross_entropy")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError("retrieval episodic head-mask cross-entropy is invalid")
        candidate_value = float(value)
        m2_reference = float(m2_value)
        m0_reference = float(m0_value)
        candidate.append(candidate_value)
        matrix.append(
            {
                "record_index": index,
                "record_id": record["record_id"],
                "same_policy_M0_answer_cross_entropy": m0_reference,
                "fixed_M2_answer_cross_entropy": m2_reference,
                "candidate_answer_cross_entropy": candidate_value,
                "candidate_minus_same_policy_M0": (candidate_value - m0_reference),
                "candidate_minus_fixed_M2": (candidate_value - m2_reference),
                "no_fixed_M2_record_regression": (candidate_value <= m2_reference),
            }
        )
    m2_maximum = float(max(m2_baseline))
    m2_mean = float(np.mean(m2_baseline, dtype=np.float64))
    m0_maximum = float(max(m0_attribution))
    m0_mean = float(np.mean(m0_attribution, dtype=np.float64))
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
            "fixed_M2_head_gated_payload_candidate": {
                "maximum_answer_cross_entropy": candidate_maximum,
                "mean_answer_cross_entropy": candidate_mean,
                "maximum_minus_same_policy_M0": (candidate_maximum - m0_maximum),
                "mean_minus_same_policy_M0": candidate_mean - m0_mean,
                "maximum_minus_fixed_M2": candidate_maximum - m2_maximum,
                "mean_minus_fixed_M2": candidate_mean - m2_mean,
            },
        },
        "gate_checks": checks,
        "passed": all(checks.values()),
    }


def _evaluate_candidate(
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
        raise ValueError("retrieval episodic head-mask train population is invalid")
    mask = _validate_fixed_m2_mask(protocol["fixed_M2_head_mask"]["mask"])
    anchors = protocol["tokenizer_fact_anchor_ids"]
    resource = protocol["resource_contract"]
    schedules = [_derive_schedule(record["input_ids"], anchors) for record in records]
    if [schedule["rows_sha256"] for schedule in schedules] != protocol[
        "schedule_contract"
    ]["per_record_rows_sha256"]:
        raise ValueError("retrieval episodic head-mask schedule authentication failed")
    sequence_evidence: list[dict[str, Any]] = []
    runtime = runtime_factory(context, mask)
    try:
        for record, schedule in zip(records, schedules, strict=True):
            index = int(record["record_index"])
            evidence = _execute_record(
                runtime,
                record=record,
                context=context,
                schedule=schedule,
                mask=mask,
                resource=resource,
                progress_label=(
                    f"fixed-M2 payload train record {index + 1}/{_RECORDS}"
                ),
            )
            sequence_evidence.append(evidence)
        replay = _execute_record(
            runtime,
            record=records[0],
            context=context,
            schedule=schedules[0],
            mask=mask,
            resource=resource,
            progress_label="fixed-M2 payload reset replay",
        )
    finally:
        runtime.close()
    replay_checks = _replay_checks(replay, sequence_evidence[0])
    final_expected = _schedule_counters(
        schedules[0],
        positions=_POSITIONS,
        model=context["model"],
        mask=mask,
        resource=resource,
    )
    resource_checks = {
        "all_sequence_counter_streams": all(
            row["counter_stream_passed"] for row in sequence_evidence
        ),
        "exact_final_episodic_counters": all(
            all(
                int(row["final_metrics"].get(name, -1)) == expected
                for name, expected in final_expected.items()
            )
            for row in sequence_evidence
        ),
        "duplicate_suppression_bounded": all(
            0
            <= int(row["final_metrics"]["episodic_duplicate_older_entries_suppressed"])
            <= final_expected["episodic_entries_read"]
            for row in sequence_evidence
        ),
        "measured_read_at_or_below_analytic_upper_bound": all(
            int(row["final_metrics"]["attention_logical_read_bytes"])
            <= int(resource["combined_attention_and_episodic_read_bytes"])
            for row in sequence_evidence
        ),
        "measured_total_traffic_at_or_below_budget": all(
            (
                int(row["final_metrics"]["attention_logical_read_bytes"])
                + int(row["final_metrics"]["episodic_write_bytes"])
            )
            <= int(resource["combined_attention_and_episodic_traffic_bytes"])
            for row in sequence_evidence
        ),
        "combined_read_budget": resource["within_read_budget"] is True,
        "combined_total_traffic_budget": (
            resource["within_total_traffic_budget"] is True
        ),
        "reset_replay": replay_checks["passed"],
    }
    loss_gate = _loss_gate(
        records=records,
        baselines=baselines,
        evidence=sequence_evidence,
    )
    resource_passed = all(resource_checks.values())
    return {
        "role": "fixed_M2_head_gated_oracle_payload_episodic_cache",
        "native_sequence_forwards": _RECORDS + 1,
        "native_token_steps": (_RECORDS + 1) * _POSITIONS,
        "fixed_M2_head_mask": _mask_descriptor(mask),
        "loss_gate": loss_gate,
        "resource_contract": resource,
        "resource_checks": resource_checks,
        "resource_passed": resource_passed,
        "sequence_evidence": sequence_evidence,
        "reset_replay": replay_checks,
        "passed": loss_gate["passed"] and resource_passed,
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
        "head_mask_protocol": (
            sha256_file(context["head_mask_protocol_path"])
            == context["head_mask_protocol_sha256"]
        ),
        "head_mask_source_inventory": (
            context["head_mask_protocol"]["source_sha256"] == _source_inventory()
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


def screen_episodic_head_mask_oracle(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    """Run the frozen fixed-M2 candidate-only train CE screen."""

    output = retrieval._new_output(
        out,
        "retrieval episodic head-mask result",
    )
    started = time.perf_counter()
    _progress("authenticating frozen protocol and fixed M2 checkpoint")
    context, training, frozen = _authenticate_protocol(
        protocol,
        protocol_sha256,
    )
    state = _checkpoint_references(training)
    _progress("starting eight CPU-native train sequences plus reset replay")
    candidate = _evaluate_candidate(
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
        raise ValueError("retrieval episodic head-mask post-run authentication failed")
    passed = candidate["passed"]
    report = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _RESULT_EXPERIMENT,
        "status": (
            "train_episodic_head_mask_gate_passed"
            if passed
            else "train_episodic_head_mask_gate_failed"
        ),
        "protocol": {
            "path": str(context["head_mask_protocol_path"]),
            "sha256": context["head_mask_protocol_sha256"],
        },
        "scope": {
            "split": "train",
            "fixed_M2_selected_layer_head_pairs": _EXPECTED_SELECTED_PAIRS,
            "threads": _THREADS,
            "device": "cpu",
            "dense_teacher_forwards": 0,
            "mask_fitting_or_selection": False,
            "development_outcomes_used": False,
            "confirmation_split_opened": False,
        },
        "reused_checkpoint_evidence": state["baselines"],
        "episodic_head_mask_candidate": candidate,
        "decision": {
            "passed": passed,
            "semantic_gate_passed": False,
            "confirmation_authorized": False,
            "next_step": (
                "freeze a distinct dense-teacher semantic development screen"
                if passed
                else "reject fixed-K51 or freeze a distinct ranked-prefix train sweep"
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
        description=(
            "Train-only fixed-M2 head-gated OLMoE episodic payload experiment"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--base-protocol", required=True)
    freeze.add_argument("--base-protocol-sha256", required=True)
    freeze.add_argument("--training-checkpoint", required=True)
    freeze.add_argument("--training-checkpoint-sha256", required=True)
    freeze.add_argument("--headwise-episodic-library", required=True)
    freeze.add_argument("--headwise-episodic-library-sha256", required=True)
    freeze.add_argument("--out", required=True)
    screen = commands.add_parser("screen")
    screen.add_argument("--protocol", required=True)
    screen.add_argument("--protocol-sha256", required=True)
    screen.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "freeze":
        value = freeze_episodic_head_mask_oracle_protocol(
            base_protocol=args.base_protocol,
            base_protocol_sha256=args.base_protocol_sha256,
            training_checkpoint=args.training_checkpoint,
            training_checkpoint_sha256=args.training_checkpoint_sha256,
            headwise_episodic_library=args.headwise_episodic_library,
            headwise_episodic_library_sha256=(args.headwise_episodic_library_sha256),
            out=args.out,
        )
    elif args.command == "screen":
        value = screen_episodic_head_mask_oracle(
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            out=args.out,
        )
    else:  # pragma: no cover - argparse owns this boundary
        raise AssertionError("unknown retrieval episodic head-mask command")
    print(json.dumps(value, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())
