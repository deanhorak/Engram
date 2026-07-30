"""Prospective train-only oracle episodic span-cache experiment.

The experiment asks a narrow systems question: if the native OLMoE runtime is
given a causal, exact span schedule for the synthetic passkey task, can a
bounded episodic K/V cache recover dense-teacher behavior without full-context
attention?  The schedule writes the 32 source payload tokens into canonical
slots and reads one corresponding eight-slot span at each answer prediction
row.

The native ABI is implemented by ``OLMoENativeTokenRuntime.forward_episodic``;
this evaluator validates its counters fail-closed. Unit tests use a mock
runtime, and the real screen binds an immutable native library. The sealed
confirmation file is never opened or hashed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import engram.evaluation.olmoe_native_causal as native_causal
import engram.evaluation.olmoe_native_sustained as sustained
import engram.evaluation.olmoe_retrieval_head_selector as retrieval
import engram.evaluation.olmoe_retrieval_prefix_selector as prefix_selector
from engram.compiler.olmoe_native import validate_olmoe_native_package
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.utils import atomic_json, sha256_file, sha256_json


_SCHEMA_VERSION = 1
_PROTOCOL_EXPERIMENT = "olmoe_q7_retrieval_episodic_oracle_protocol"
_RESULT_EXPERIMENT = "olmoe_q7_retrieval_episodic_oracle_train_screen"
_PROTOCOL_STATUS = "frozen_before_train_teacher_control_or_candidate_execution"
_RECORDS = 8
_POSITIONS = 128
_ANSWER_START = 96
_ANSWER_POSITIONS = 32
_SLOTS = 32
_PAYLOAD_SPANS = 4
_SPAN_TOKENS = 8
_CACHE_DTYPE_BYTES = 2
_POSITION_DTYPE_BYTES = 8
_SCRATCH_DTYPE_BYTES = 4
_THREADS = 12
_MAXIMUM_TRAFFIC_FRACTION = 0.45
_THRESHOLDS = {
    "maximum_mean_kl": 0.05,
    "minimum_top1_agreement": 0.90,
    "maximum_target_nll_delta": 0.05,
    "maximum_hidden_relative_l2": 0.10,
}
_EPISODIC_COUNTER_NAMES = (
    "episodic_slots_written",
    "episodic_read_events",
    "episodic_active_slots",
    "episodic_entries_read",
    "episodic_write_bytes",
    "episodic_key_read_bytes",
    "episodic_value_read_bytes",
    "episodic_duplicate_older_entries_suppressed",
    "episodic_state_bytes",
    "episodic_scratch_bytes",
)
_SOURCE_FILES = (
    "native/include/engram/streaming_attention.h",
    "native/include/engram/streaming_attention_c.h",
    "native/include/engram/olmoe_token_runtime.h",
    "native/include/engram/olmoe_token_runtime_c.h",
    "native/src/streaming_attention.cpp",
    "native/src/streaming_attention_c.cpp",
    "native/src/olmoe_token_runtime.cpp",
    "native/src/olmoe_token_runtime_c.cpp",
    "src/engram/compiler/olmoe_native.py",
    "src/engram/evaluation/olmoe_native_causal.py",
    "src/engram/evaluation/olmoe_native_headwise.py",
    "src/engram/evaluation/olmoe_native_sustained.py",
    "src/engram/evaluation/olmoe_retrieval_head_selector.py",
    "src/engram/evaluation/olmoe_retrieval_prefix_selector.py",
    "src/engram/evaluation/olmoe_retrieval_episodic_oracle.py",
    "src/engram/runtime/olmoe_native.py",
)


def _progress(message: str) -> None:
    print(f"[retrieval-episodic-oracle] {message}", file=sys.stderr, flush=True)


def _source_inventory() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    return {name: sha256_file(repository / name) for name in _SOURCE_FILES}


def _validate_fact_anchor_ids(
    anchors: Mapping[str, Sequence[int]],
) -> dict[str, tuple[int, ...]]:
    return prefix_selector._validate_fact_anchor_ids(anchors)


def _fact_anchor_ids(tokenizer_path: str | Path) -> dict[str, tuple[int, ...]]:
    return prefix_selector._fact_anchor_ids(tokenizer_path)


def _fact_order_from_causal_prefix(
    input_ids: Sequence[int],
    anchors: Mapping[str, Sequence[int]],
) -> tuple[str, ...]:
    return prefix_selector._fact_order_from_causal_prefix(input_ids, anchors)


def _derive_schedule(
    input_ids: Sequence[int],
    anchors: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Derive all operations without observing an input token after row 96."""

    order = _fact_order_from_causal_prefix(input_ids, anchors)
    rows = [
        {
            "position": position,
            "write_slot": -1,
            "read_span": -1,
        }
        for position in range(_POSITIONS)
    ]
    for depth, label in enumerate(order):
        label_index = retrieval._LABELS.index(label)
        source_start = retrieval._PASSKEY_SOURCE_STARTS[depth]
        for offset in range(_SPAN_TOKENS):
            rows[source_start + offset]["write_slot"] = (
                label_index * _SPAN_TOKENS + offset
            )
    for position in range(_ANSWER_START, _POSITIONS):
        answer_offset = position - _ANSWER_START
        rows[position]["read_span"] = answer_offset // _SPAN_TOKENS
    _validate_schedule_rows(rows)
    prefix_ids = [int(value) for value in input_ids[: _ANSWER_START + 1]]
    return {
        "last_input_index_observed_during_derivation": _ANSWER_START,
        "future_answer_tokens_observed_during_derivation": False,
        "prefix_fact_order": list(order),
        "causal_prefix_sha256": sha256_json(prefix_ids),
        "rows": rows,
        "rows_sha256": sha256_json(rows),
    }


def _validate_schedule_rows(rows: Any) -> None:
    if (
        not isinstance(rows, list)
        or len(rows) != _POSITIONS
        or any(
            not isinstance(row, Mapping)
            or row.get("position") != position
            or set(row) != {"position", "write_slot", "read_span"}
            for position, row in enumerate(rows)
        )
    ):
        raise ValueError("retrieval episodic schedule rows are invalid")
    writes: list[tuple[int, int]] = []
    reads: list[tuple[int, int]] = []
    for position, row in enumerate(rows):
        slot = row["write_slot"]
        span = row["read_span"]
        if (
            isinstance(slot, bool)
            or not isinstance(slot, int)
            or slot < -1
            or slot >= _SLOTS
        ):
            raise ValueError("retrieval episodic write slot is invalid")
        if (
            isinstance(span, bool)
            or not isinstance(span, int)
            or span < -1
            or span >= _PAYLOAD_SPANS
        ):
            raise ValueError("retrieval episodic read span is invalid")
        if slot >= 0:
            writes.append((position, slot))
        if span >= 0:
            reads.append((position, span))
    expected_write_positions = [
        start + offset
        for start in retrieval._PASSKEY_SOURCE_STARTS
        for offset in range(_SPAN_TOKENS)
    ]
    expected_reads = [
        (_ANSWER_START + offset, offset // _SPAN_TOKENS)
        for offset in range(_ANSWER_POSITIONS)
    ]
    if (
        [position for position, _slot in writes] != expected_write_positions
        or len(writes) != _SLOTS
        or sorted(slot for _position, slot in writes) != list(range(_SLOTS))
        or reads != expected_reads
    ):
        raise ValueError("retrieval episodic schedule contract changed")


def _resource_contract(
    model: Mapping[str, int],
    q7_expectations: Mapping[str, int],
) -> dict[str, Any]:
    base = sustained._attention_expectations(
        dict(model),
        retrieval._BASE_POLICY,
        positions=_POSITIONS,
    )
    layers = int(model["layers"])
    hidden_size = int(model["hidden_size"])
    query_heads = int(model["query_heads"])
    slot_payload_bytes_per_layer = 2 * hidden_size * _CACHE_DTYPE_BYTES
    payload_state_bytes = _SLOTS * layers * slot_payload_bytes_per_layer
    position_state_bytes = _SLOTS * layers * _POSITION_DTYPE_BYTES
    write_bytes = payload_state_bytes
    read_events = _ANSWER_POSITIONS
    entries_read = read_events * _SPAN_TOKENS * query_heads * layers
    key_read_bytes = (
        read_events * _SPAN_TOKENS * layers * hidden_size * _CACHE_DTYPE_BYTES
    )
    value_read_bytes = key_read_bytes
    read_bytes = key_read_bytes + value_read_bytes
    joint_softmax_scratch_bytes = (
        layers * 2 * _SPAN_TOKENS * _SCRATCH_DTYPE_BYTES
    )
    combined_state_bytes = (
        int(base["attention_state_bytes"])
        + payload_state_bytes
        + position_state_bytes
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
        "cache_payload": {
            "dtype": "bfloat16",
            "dtype_bytes": _CACHE_DTYPE_BYTES,
            "contents": "per-layer key and value rows across the hidden width",
            "capacity_slots": _SLOTS,
            "slot_payload_bytes_per_layer": slot_payload_bytes_per_layer,
            "payload_state_bytes": payload_state_bytes,
            "position_dtype": "uint64",
            "position_state_bytes": position_state_bytes,
            "state_bytes": payload_state_bytes + position_state_bytes,
        },
        "schedule": {
            "unique_write_slots": _SLOTS,
            "write_rows": _SLOTS,
            "read_rows": read_events,
            "read_span_tokens": _SPAN_TOKENS,
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
        "within_total_traffic_budget": (
            traffic_fraction <= _MAXIMUM_TRAFFIC_FRACTION
        ),
        "combined_state_bytes": combined_state_bytes,
        "episodic_joint_softmax_scratch_bytes": (
            joint_softmax_scratch_bytes
        ),
        "combined_scratch_bytes": combined_scratch_bytes,
        "q7_expectations_per_sequence": dict(q7_expectations),
    }
    if (
        not contract["within_read_budget"]
        or not contract["within_total_traffic_budget"]
        or slot_payload_bytes_per_layer <= 0
    ):
        raise ValueError("retrieval episodic resource contract exceeds budget")
    return contract


def _checkpoint_baseline(training: Mapping[str, Any]) -> dict[str, Any]:
    entries = training.get("masks")
    m2 = entries.get("M2") if isinstance(entries, Mapping) else None
    if not isinstance(m2, Mapping):
        raise ValueError("retrieval episodic checkpoint M2 evidence is missing")
    mask = retrieval._boolean_mask(m2.get("mask"), "retrieval episodic M2")
    rows = m2.get("records")
    if (
        int(mask.sum()) != retrieval._RESCUED_HEADS
        or not isinstance(rows, list)
        or len(rows) != _RECORDS
    ):
        raise ValueError("retrieval episodic checkpoint M2 evidence is invalid")
    losses: list[float] = []
    for row in rows:
        loss = row.get("loss")
        value = (
            loss.get("answer_cross_entropy")
            if isinstance(loss, Mapping)
            else None
        )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError("retrieval episodic checkpoint M2 loss is invalid")
        losses.append(float(value))
    return {
        "mask_sha256": sha256_json(mask.tolist()),
        "selected_head_count": int(mask.sum()),
        "record_answer_cross_entropy": losses,
        "maximum_answer_cross_entropy": float(max(losses)),
        "mean_answer_cross_entropy": float(np.mean(losses, dtype=np.float64)),
    }


def _build_protocol(
    *,
    context: Mapping[str, Any],
    checkpoint_descriptor: Mapping[str, str],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    anchors = _fact_anchor_ids(context["tokenizer_path"])
    schedules = [
        _derive_schedule(record["input_ids"], anchors)
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
            "path": checkpoint_descriptor["path"],
            "sha256": checkpoint_descriptor["sha256"],
            "training_sha256": sha256_json(training),
        },
        "episodic_library": {
            "path": str(context["episodic_library_path"]),
            "sha256": context["episodic_library_sha256"],
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
            "development_outcomes_used": False,
            "confirmation_file_access_permitted": False,
        },
        "tokenizer_fact_anchor_ids": {
            label: list(values) for label, values in anchors.items()
        },
        "schedule_contract": {
            "derivation_input": "input_ids[0:97] and authenticated tokenizer anchors",
            "last_input_index_observed": _ANSWER_START,
            "write_source_starts": list(retrieval._PASSKEY_SOURCE_STARTS),
            "payload_spans": _PAYLOAD_SPANS,
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
            "arguments": [
                "token_ids",
                "write_slots",
                "read_spans",
            ],
            "write_slots": "one int32 per token: canonical slot 0..31 or -1",
            "read_spans": "one int32 per token: span index 0..3 or -1",
            "episodic_policy": {
                "slots": _SLOTS,
                "span_size": _SPAN_TOKENS,
            },
            "cumulative_metric_names": list(_EPISODIC_COUNTER_NAMES),
        },
        "resource_contract": _resource_contract(
            context["model"],
            context["q7_expectations"],
        ),
        "semantic_thresholds": dict(_THRESHOLDS),
        "evaluation_contract": {
            "fresh_dense_teacher_on_train": True,
            "fresh_full_W128_Q7_control_on_train": True,
            "base_W16_C8_K4_S2_attention_plus_oracle_episodic_candidate": True,
            "reset_replay_record_index": 0,
            "overall_and_each_source_depth_must_pass": True,
            "exact_per_position_analytic_counters_required": True,
            "duplicate_suppression_is_replay_exact_and_bounded": True,
        },
        "reused_checkpoint_evidence": _checkpoint_baseline(training),
        "confirmation_split_opened": False,
    }


def freeze_episodic_oracle_protocol(
    *,
    base_protocol: str | Path,
    base_protocol_sha256: str,
    training_checkpoint: str | Path,
    training_checkpoint_sha256: str,
    episodic_library: str | Path,
    episodic_library_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    """Freeze the experiment before any new teacher or candidate execution."""

    output = retrieval._new_output(out, "retrieval episodic protocol")
    context, training, selection, checkpoint = _authenticate_base_inputs(
        base_protocol,
        base_protocol_sha256,
        training_checkpoint,
        training_checkpoint_sha256,
        episodic_library,
        episodic_library_sha256,
    )
    if (
        selection.get("screen_eligible") is not True
        or selection.get("selected_mask_name") != "M2"
    ):
        raise ValueError("retrieval episodic protocol requires eligible M2 evidence")
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


def _checked_file(
    value: str | Path,
    expected_sha256: str,
    label: str,
) -> Path:
    requested = Path(value).expanduser()
    if requested.is_symlink():
        raise ValueError(f"{label} authentication failed")
    path = requested.resolve()
    if (
        not path.is_file()
        or not retrieval._is_sha256(expected_sha256)
        or sha256_file(path) != expected_sha256.lower()
    ):
        raise ValueError(f"{label} authentication failed")
    return path


def _validate_split_descriptor(descriptor: Any, label: str) -> dict[str, Any]:
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("records") != _RECORDS
        or descriptor.get("tokens_per_record") != _POSITIONS + 1
        or descriptor.get("prediction_positions_per_record") != _POSITIONS
        or descriptor.get("answer_prediction_positions_per_record")
        != _ANSWER_POSITIONS
        or not isinstance(descriptor.get("file"), str)
        or not retrieval._is_sha256(descriptor.get("sha256"))
        or not retrieval._is_sha256(descriptor.get("record_identity_sha256"))
    ):
        raise ValueError(f"retrieval episodic {label} descriptor is invalid")
    return dict(descriptor)


def _load_checkpoint_train_only(
    path: str | Path,
    expected_sha256: str,
    *,
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    source = _checked_file(
        path,
        expected_sha256,
        "retrieval episodic training checkpoint",
    )
    checkpoint = retrieval._read_json(
        source,
        "retrieval episodic training checkpoint",
    )
    expected_keys = {
        "schema_version",
        "experiment",
        "status",
        "protocol",
        "source_sha256",
        "train_record_identity_sha256",
        "training",
        "training_sha256",
        "post_training_authentication",
        "confirmation_split_opened",
    }
    expected_identity = sha256_json(
        [row["identity_sha256"] for row in context["train_records"]]
    )
    historical_authentication = checkpoint.get("post_training_authentication")
    required_historical_checks = {
        "protocol",
        "package",
        "corpus_manifest",
        "train_split",
        "confirmation_not_opened",
        "source_config",
        "source_index",
        "source_shards",
        "layered_library",
        "headwise_library",
        "attention_library",
    }
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != expected_keys
        or checkpoint.get("schema_version") != retrieval._SCHEMA_VERSION
        or checkpoint.get("experiment")
        != retrieval._TRAINING_CHECKPOINT_EXPERIMENT
        or checkpoint.get("status") != retrieval._TRAINING_CHECKPOINT_STATUS
        or checkpoint.get("protocol")
        != {
            "path": str(context["protocol_path"]),
            "sha256": context["protocol_sha256"],
        }
        or checkpoint.get("source_sha256")
        != context["protocol"]["source_sha256"]
        or checkpoint.get("train_record_identity_sha256") != expected_identity
        or checkpoint.get("confirmation_split_opened") is not False
        or checkpoint.get("training_sha256")
        != sha256_json(checkpoint.get("training"))
        or not isinstance(historical_authentication, dict)
        or not historical_authentication
        or not all(value is True for value in historical_authentication.values())
        or not required_historical_checks.issubset(historical_authentication)
    ):
        raise ValueError("retrieval episodic training checkpoint contract changed")
    training = checkpoint["training"]
    selection, _selected_heads = retrieval._validate_training_payload(
        training,
        context=context,
    )
    return (
        training,
        selection,
        {
            "path": str(source),
            "sha256": expected_sha256.lower(),
            "mode": "resumed",
        },
    )


def _authenticate_base_inputs(
    base_protocol: str | Path,
    base_protocol_sha256: str,
    training_checkpoint: str | Path,
    training_checkpoint_sha256: str,
    episodic_library: str | Path,
    episodic_library_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    """Authenticate immutable base evidence while reading only the train split."""

    path = _checked_file(
        base_protocol,
        base_protocol_sha256,
        "retrieval episodic base protocol",
    )
    protocol = retrieval._read_json(path, "retrieval episodic base protocol")
    if (
        protocol.get("schema_version") != retrieval._SCHEMA_VERSION
        or protocol.get("experiment") != retrieval._PROTOCOL_EXPERIMENT
        or protocol.get("status") != retrieval._PROTOCOL_STATUS
        or protocol.get("seed") != retrieval._SEED
        or protocol.get("training", {}).get("workers") != retrieval._WORKERS
        or protocol.get("training", {}).get("iht_steps") != retrieval._IHT_STEPS
        or protocol.get("training", {}).get("masks")
        != list(retrieval._MASK_NAMES)
        or protocol.get("training", {}).get("answer_prediction_positions")
        != list(range(_ANSWER_START, _POSITIONS))
        or protocol.get("attention_policies", {}).get("base")
        != retrieval._BASE_POLICY
    ):
        raise ValueError("retrieval episodic base protocol contract changed")
    package_contract = protocol.get("package")
    source_contract = protocol.get("source_model")
    libraries = protocol.get("libraries")
    corpus_contract = protocol.get("corpus")
    proxy_contract = protocol.get("expert_proxy_qualifier")
    if not all(
        isinstance(value, dict)
        for value in (
            package_contract,
            source_contract,
            libraries,
            corpus_contract,
            proxy_contract,
        )
    ):
        raise ValueError("retrieval episodic base bindings are invalid")
    package_path = Path(package_contract["path"]).expanduser().resolve()
    manifest = validate_olmoe_native_package(
        package_path,
        expected_manifest_sha256=package_contract["manifest_sha256"],
    )
    config_path = package_path / manifest["model"]["config_path"]
    non_mlp_path = package_path / manifest["transformer"]["path"]
    q7_path = package_path / manifest["mlp"]["path"]
    tokenizer_path = package_path / manifest["tokenizer"]["path"] / "tokenizer.json"
    config = retrieval._read_json(config_path, "packaged OLMoE config")
    model = sustained._model_descriptor(manifest, config)
    q7_expectations = sustained._q7_expectations(model)
    if (
        model != package_contract.get("model")
        or q7_expectations != package_contract.get("q7_expectations_per_sequence")
        or sha256_file(tokenizer_path) != package_contract.get("tokenizer_sha256")
    ):
        raise ValueError("retrieval episodic package contract changed")
    library_paths: dict[str, Path] = {}
    for name in ("layered", "headwise", "attention"):
        descriptor = libraries.get(name)
        if not isinstance(descriptor, dict):
            raise ValueError("retrieval episodic base library descriptor is invalid")
        library_paths[name] = _checked_file(
            descriptor.get("path", ""),
            descriptor.get("sha256", ""),
            f"retrieval episodic {name} library",
        )
    proxy_path = _checked_file(
        proxy_contract.get("path", ""),
        proxy_contract.get("sha256", ""),
        "retrieval episodic expert proxy",
    )
    model_path = Path(source_contract.get("path", "")).expanduser().resolve()
    if (
        model_path != Path(manifest["source"]["path"]).resolve()
        or sha256_file(model_path / "config.json")
        != source_contract.get("config_sha256")
        or sha256_file(model_path / "model.safetensors.index.json")
        != source_contract.get("index_sha256")
        or retrieval._source_shard_inventory(model_path)
        != source_contract.get("shard_sha256")
        or manifest.get("source", {}).get("revision")
        != source_contract.get("revision")
    ):
        raise ValueError("retrieval episodic source model changed")
    manifest_path = retrieval._safe_relative_path(
        path.parent,
        corpus_contract.get("manifest_file"),
        "retrieval episodic corpus manifest",
    )
    corpus_manifest = retrieval._read_json(
        manifest_path,
        "retrieval episodic corpus manifest",
    )
    split_contracts = corpus_contract.get("splits")
    if (
        sha256_file(manifest_path) != corpus_contract.get("manifest_sha256")
        or corpus_manifest.get("schema_version") != retrieval._SCHEMA_VERSION
        or corpus_manifest.get("experiment")
        != "olmoe_q7_synthetic_passkey_corpus"
        or corpus_manifest.get("generator_seed") != retrieval._SEED
        or corpus_manifest.get("tokenizer_sha256") != sha256_file(tokenizer_path)
        or corpus_manifest.get("splits") != split_contracts
        or not isinstance(split_contracts, dict)
        or set(split_contracts) != set(retrieval._SPLITS)
    ):
        raise ValueError("retrieval episodic corpus manifest changed")
    train_descriptor = _validate_split_descriptor(
        split_contracts["train"],
        "train",
    )
    confirmation_descriptor = _validate_split_descriptor(
        split_contracts["confirmation"],
        "confirmation",
    )
    train_path = retrieval._safe_relative_path(
        path.parent,
        train_descriptor["file"],
        "retrieval episodic train split",
    )
    if sha256_file(train_path) != train_descriptor["sha256"]:
        raise ValueError("retrieval episodic train split changed")
    train_records = retrieval._read_split(train_path, split="train")
    if sha256_json(
        [row["identity_sha256"] for row in train_records]
    ) != train_descriptor.get("record_identity_sha256"):
        raise ValueError("retrieval episodic train identity changed")
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for episodic evaluation"
        ) from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pool = retrieval._token_pool(tokenizer)
    partitions = retrieval._partition_code_tokens(pool, seed=retrieval._SEED)
    for index, row in enumerate(train_records):
        expected = retrieval._generate_record(
            tokenizer,
            token_pool=pool,
            split="train",
            index=index,
            seed=retrieval._SEED,
            code_token_ids=partitions[("train", index)],
        )
        if row != expected:
            raise ValueError(
                f"retrieval episodic train record {index} failed reconstruction"
            )
    episodic_path = _checked_file(
        episodic_library,
        episodic_library_sha256,
        "retrieval episodic native library",
    )
    context: dict[str, Any] = {
        "protocol": protocol,
        "protocol_path": path,
        "protocol_sha256": base_protocol_sha256.lower(),
        "manifest": manifest,
        "package_path": package_path,
        "config_path": config_path,
        "non_mlp_path": non_mlp_path,
        "q7_path": q7_path,
        "tokenizer_path": tokenizer_path,
        "model_path": model_path,
        "model": model,
        "q7_expectations": q7_expectations,
        "library_paths": library_paths,
        "proxy_path": proxy_path,
        "manifest_path": manifest_path,
        "split_paths": {"train": train_path},
        "train_records": train_records,
        "confirmation_descriptor": confirmation_descriptor,
        "confirmation_descriptor_authenticated_without_file_access": True,
        "episodic_library_path": episodic_path,
        "episodic_library_sha256": episodic_library_sha256.lower(),
    }
    training, selection, checkpoint = _load_checkpoint_train_only(
        training_checkpoint,
        training_checkpoint_sha256,
        context=context,
    )
    return context, training, selection, checkpoint


def _authenticate_protocol(
    protocol: str | Path,
    protocol_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = Path(protocol).expanduser()
    if path.is_symlink():
        raise ValueError("retrieval episodic protocol authentication failed")
    source = path.resolve()
    if (
        not source.is_file()
        or not retrieval._is_sha256(protocol_sha256)
        or sha256_file(source) != protocol_sha256.lower()
    ):
        raise ValueError("retrieval episodic protocol authentication failed")
    value = retrieval._read_json(source, "retrieval episodic protocol")
    base = value.get("base_retrieval_protocol")
    checkpoint_binding = value.get("training_checkpoint")
    library_binding = value.get("episodic_library")
    if not all(
        isinstance(binding, Mapping)
        for binding in (base, checkpoint_binding, library_binding)
    ):
        raise ValueError("retrieval episodic protocol bindings are invalid")
    context, training, selection, checkpoint = _authenticate_base_inputs(
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
        raise ValueError("retrieval episodic checkpoint selection changed")
    expected = _build_protocol(
        context=context,
        checkpoint_descriptor=checkpoint,
        training=training,
    )
    if value != expected:
        raise ValueError("retrieval episodic protocol contract changed")
    context = dict(context)
    context["episodic_protocol_path"] = source
    context["episodic_protocol_sha256"] = protocol_sha256.lower()
    return context, training, expected


def _schedule_counters(
    schedule: Mapping[str, Any],
    *,
    positions: int,
    model: Mapping[str, int],
    resource: Mapping[str, Any],
) -> dict[str, int]:
    rows = schedule.get("rows")
    _validate_schedule_rows(rows)
    if (
        isinstance(positions, bool)
        or not isinstance(positions, int)
        or not 0 <= positions <= _POSITIONS
    ):
        raise ValueError("retrieval episodic counter position is invalid")
    writes: list[int] = []
    read_spans: list[int] = []
    for row in rows[:positions]:
        if row["write_slot"] >= 0:
            writes.append(int(row["write_slot"]))
        if row["read_span"] >= 0:
            read_spans.append(int(row["read_span"]))
    layers = int(model["layers"])
    query_heads = int(model["query_heads"])
    hidden_size = int(model["hidden_size"])
    write_rows = len(writes)
    read_rows = len(read_spans)
    return {
        "episodic_slots_written": write_rows * layers,
        "episodic_read_events": read_rows * layers,
        "episodic_active_slots": len(set(writes)) * layers,
        "episodic_entries_read": (
            read_rows * _SPAN_TOKENS * query_heads * layers
        ),
        "episodic_write_bytes": (
            write_rows * layers * 2 * hidden_size * _CACHE_DTYPE_BYTES
        ),
        "episodic_key_read_bytes": (
            read_rows
            * _SPAN_TOKENS
            * layers
            * hidden_size
            * _CACHE_DTYPE_BYTES
        ),
        "episodic_value_read_bytes": (
            read_rows
            * _SPAN_TOKENS
            * layers
            * hidden_size
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
    resource: Mapping[str, Any],
) -> dict[str, bool]:
    base = sustained._attention_expectations(
        dict(context["model"]),
        retrieval._BASE_POLICY,
        positions=positions,
    )
    expected = _schedule_counters(
        schedule,
        positions=positions,
        model=context["model"],
        resource=resource,
    )
    checks = {
        name: int(metrics.get(name, -1)) == value
        for name, value in expected.items()
    }
    duplicate = int(
        metrics.get("episodic_duplicate_older_entries_suppressed", -1)
    )
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
            "attention_older_selected_entries": (
                0 <= selected_loss <= duplicate
            ),
            "attention_logical_read_bytes": (
                int(metrics.get("attention_logical_read_bytes", -1))
                == logical_expected
            ),
            "q7_scheduled_bytes": (
                int(metrics.get("q7_scheduled_bytes", -1))
                == positions
                * int(
                    context["q7_expectations"][
                        "scheduled_bytes_per_position"
                    ]
                )
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


def _counter_digest_update(digest: Any, metrics: Mapping[str, int]) -> None:
    digest.update(
        json.dumps(
            sustained._deterministic_metrics(dict(metrics)),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")


def _open_episodic_runtime(context: Mapping[str, Any]) -> Any:
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
    )


def _execute_episodic_record(
    runtime: Any,
    *,
    record: Mapping[str, Any],
    context: Mapping[str, Any],
    schedule: Mapping[str, Any],
    resource: Mapping[str, Any],
    progress_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if runtime.position != 0:
        runtime.reset()
    if (
        runtime.position != 0
        or not runtime.attention_metrics_available
        or not callable(getattr(runtime, "forward_episodic", None))
        or getattr(runtime, "episodic_policy", None)
        != {"slots": _SLOTS, "span_size": _SPAN_TOKENS}
    ):
        raise ValueError("retrieval episodic runtime capability is unavailable")
    rows = schedule["rows"]
    logits_rows: list[np.ndarray] = []
    hidden_rows: list[np.ndarray] = []
    top1_tokens: list[int] = []
    hidden_digest = hashlib.sha256()
    logit_digest = hashlib.sha256()
    counter_digest = hashlib.sha256()
    call_digest = hashlib.sha256()
    counter_stream: list[dict[str, Any]] = []
    final_metrics: dict[str, int] | None = None
    previous_duplicate_suppressions = 0
    started = time.perf_counter()
    for position, token_id in enumerate(record["input_ids"][:-1]):
        row = rows[position]
        write_slots = [int(row["write_slot"])]
        read_spans = [int(row["read_span"])]
        result = runtime.forward_episodic(
            [int(token_id)],
            write_slots,
            read_spans,
        )
        if runtime.position != position + 1:
            raise ValueError("retrieval episodic cache position did not advance")
        final_metrics = dict(result.metrics)
        checks = _counter_checks(
            final_metrics,
            context=context,
            schedule=schedule,
            positions=position + 1,
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
                    name: final_metrics.get(name)
                    for name in _EPISODIC_COUNTER_NAMES
                },
            }
        )
        _counter_digest_update(counter_digest, final_metrics)
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
                or logits.shape != (int(context["model"]["vocab_size"]),)
                or hidden.dtype != np.float32
                or logits.dtype != np.float32
                or not np.isfinite(hidden).all()
                or not np.isfinite(logits).all()
                or int(np.argmax(logits)) != int(result.next_token)
            ):
                raise ValueError("retrieval episodic diagnostics are invalid")
            hidden_rows.append(hidden)
            logits_rows.append(logits)
            hidden_digest.update(hidden.tobytes())
            logit_digest.update(logits.tobytes())
        if progress_label is not None and (position + 1) % 32 == 0:
            _progress(f"{progress_label}: position {position + 1}/{_POSITIONS}")
    if (
        final_metrics is None
        or runtime.position != _POSITIONS
        or len(hidden_rows) != _ANSWER_POSITIONS
        or len(logits_rows) != _ANSWER_POSITIONS
    ):
        raise ValueError("retrieval episodic record execution is incomplete")
    native_hidden = np.stack(hidden_rows).astype(np.float32, copy=False)
    native_logits = np.stack(logits_rows).astype(np.float32, copy=False)
    targets = np.asarray(record["input_ids"][_ANSWER_START + 1 :], dtype=np.int64)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for episodic evaluation"
        ) from exc
    with torch.inference_mode():
        answer_cross_entropy = float(
            retrieval._answer_cross_entropy(
                torch.from_numpy(native_logits).unsqueeze(0),
                torch.from_numpy(targets).unsqueeze(0),
            ).item()
        )
    evidence = {
        "record_index": int(record["record_index"]),
        "record_id": record["record_id"],
        "answer_cross_entropy": answer_cross_entropy,
        "top1_tokens": top1_tokens,
        "hidden_sha256": hidden_digest.hexdigest(),
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
    return native_logits, native_hidden, evidence


def _quality_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "mean_kl": (
            float(metrics["teacher_to_native_kl"])
            <= _THRESHOLDS["maximum_mean_kl"]
        ),
        "top1_agreement": (
            float(metrics["teacher_top1_agreement"])
            >= _THRESHOLDS["minimum_top1_agreement"]
        ),
        "target_nll_delta": (
            float(metrics["target_nll_delta"])
            <= _THRESHOLDS["maximum_target_nll_delta"]
        ),
        "hidden_relative_l2": (
            float(metrics["final_hidden_relative_l2"])
            <= _THRESHOLDS["maximum_hidden_relative_l2"]
        ),
    }


def _replay_checks(
    replay: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, bool]:
    replay_metrics = replay.get("final_metrics")
    reference_metrics = reference.get("final_metrics")
    checks = {
        "top1_tokens": replay.get("top1_tokens") == reference.get("top1_tokens"),
        "hidden_sha256": replay.get("hidden_sha256")
        == reference.get("hidden_sha256"),
        "logits_sha256": replay.get("logits_sha256")
        == reference.get("logits_sha256"),
        "counter_stream_sha256": replay.get("counter_stream_sha256")
        == reference.get("counter_stream_sha256"),
        "episodic_call_stream_sha256": replay.get(
            "episodic_call_stream_sha256"
        )
        == reference.get("episodic_call_stream_sha256"),
        "schedule_rows_sha256": replay.get("schedule_rows_sha256")
        == reference.get("schedule_rows_sha256"),
        "answer_cross_entropy": replay.get("answer_cross_entropy")
        == reference.get("answer_cross_entropy"),
        "counter_stream_passed": replay.get("counter_stream_passed") is True,
        "deterministic_final_metrics": (
            isinstance(replay_metrics, dict)
            and isinstance(reference_metrics, dict)
            and sustained._deterministic_metrics(replay_metrics)
            == sustained._deterministic_metrics(reference_metrics)
        ),
    }
    checks["passed"] = all(checks.values())
    return checks


def _evaluate_candidate(
    *,
    context: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    teacher: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    runtime_factory: Callable[[Mapping[str, Any]], Any] = _open_episodic_runtime,
) -> dict[str, Any]:
    if (
        len(records) != _RECORDS
        or len(teacher) != _RECORDS
        or [row.get("record_index") for row in teacher] != list(range(_RECORDS))
    ):
        raise ValueError("retrieval episodic train population is invalid")
    anchors = protocol["tokenizer_fact_anchor_ids"]
    resource = protocol["resource_contract"]
    schedules = [
        _derive_schedule(record["input_ids"], anchors) for record in records
    ]
    if [schedule["rows_sha256"] for schedule in schedules] != protocol[
        "schedule_contract"
    ]["per_record_rows_sha256"]:
        raise ValueError("retrieval episodic schedule authentication failed")
    position_rows: list[dict[str, Any]] = []
    sequence_evidence: list[dict[str, Any]] = []
    replay_reference: dict[str, Any] | None = None
    runtime = runtime_factory(context)
    try:
        for record, reference, schedule in zip(
            records,
            teacher,
            schedules,
            strict=True,
        ):
            index = int(record["record_index"])
            label = f"episodic oracle train record {index + 1}/{_RECORDS}"
            native_logits, native_hidden, evidence = _execute_episodic_record(
                runtime,
                record=record,
                context=context,
                schedule=schedule,
                resource=resource,
                progress_label=label,
            )
            for answer_offset, position in enumerate(
                record["answer_prediction_positions"]
            ):
                metrics = native_causal._position_metrics(
                    reference["logits"][answer_offset],
                    native_logits[answer_offset],
                    reference["hidden"][answer_offset],
                    native_hidden[answer_offset],
                    int(reference["targets"][answer_offset]),
                )
                position_rows.append(
                    {
                        "record_index": index,
                        "position": int(position),
                        "answer_offset": answer_offset,
                        "source_depth": record["answer_source_depths"][
                            answer_offset
                        ],
                        **metrics,
                    }
                )
            sequence_evidence.append(evidence)
            if replay_reference is None:
                replay_reference = evidence
        if replay_reference is None:
            raise ValueError("retrieval episodic replay reference is missing")
        _logits, _hidden, replay = _execute_episodic_record(
            runtime,
            record=records[0],
            context=context,
            schedule=schedules[0],
            resource=resource,
            progress_label="episodic oracle reset replay",
        )
    finally:
        runtime.close()
    replay = _replay_checks(replay, replay_reference)
    overall = native_causal._aggregate(position_rows)
    depths = {
        depth: native_causal._aggregate(
            [row for row in position_rows if row["source_depth"] == depth]
        )
        for depth in retrieval._SOURCE_DEPTH_NAMES
    }
    quality_checks = {
        "overall": _quality_checks(overall),
        "source_depths": {
            depth: _quality_checks(metrics) for depth, metrics in depths.items()
        },
    }
    quality_passed = all(quality_checks["overall"].values()) and all(
        all(checks.values())
        for checks in quality_checks["source_depths"].values()
    )
    final_expected = _schedule_counters(
        schedules[0],
        positions=_POSITIONS,
        model=context["model"],
        resource=resource,
    )
    resource_checks = {
        "all_sequence_counter_streams": all(
            evidence["counter_stream_passed"] for evidence in sequence_evidence
        ),
        "exact_final_episodic_counters": all(
            all(
                int(evidence["final_metrics"].get(name, -1)) == expected
                for name, expected in final_expected.items()
            )
            for evidence in sequence_evidence
        ),
        "duplicate_suppression_bounded": all(
            0
            <= int(
                evidence["final_metrics"][
                    "episodic_duplicate_older_entries_suppressed"
                ]
            )
            <= final_expected["episodic_entries_read"]
            for evidence in sequence_evidence
        ),
        "measured_read_at_or_below_analytic_upper_bound": all(
            int(evidence["final_metrics"]["attention_logical_read_bytes"])
            <= int(resource["combined_attention_and_episodic_read_bytes"])
            for evidence in sequence_evidence
        ),
        "measured_total_traffic_at_or_below_budget": all(
            (
                int(evidence["final_metrics"]["attention_logical_read_bytes"])
                + int(evidence["final_metrics"]["episodic_write_bytes"])
            )
            <= int(resource["combined_attention_and_episodic_traffic_bytes"])
            for evidence in sequence_evidence
        ),
        "combined_read_budget": resource["within_read_budget"] is True,
        "combined_total_traffic_budget": (
            resource["within_total_traffic_budget"] is True
        ),
        "reset_replay": replay["passed"],
    }
    return {
        "role": "base_attention_plus_oracle_episodic_span_cache",
        "overall_answer_positions": overall,
        "source_depths": depths,
        "quality_checks": quality_checks,
        "quality_passed": quality_passed,
        "resource_contract": resource,
        "resource_checks": resource_checks,
        "resource_passed": all(resource_checks.values()),
        "position_rows": position_rows,
        "sequence_evidence": sequence_evidence,
        "reset_replay": replay,
        "passed": quality_passed and all(resource_checks.values()),
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
        "episodic_library": (
            sha256_file(context["episodic_library_path"])
            == context["episodic_library_sha256"]
        ),
        "episodic_protocol": (
            sha256_file(context["episodic_protocol_path"])
            == context["episodic_protocol_sha256"]
        ),
        "episodic_source_inventory": (
            context["episodic_protocol"]["source_sha256"]
            == _source_inventory()
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


def screen_episodic_oracle(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    """Run fresh train teacher/control and the prospective episodic candidate."""

    output = retrieval._new_output(out, "retrieval episodic result")
    context, training, frozen = _authenticate_protocol(
        protocol,
        protocol_sha256,
    )
    context["episodic_protocol"] = frozen
    started = time.perf_counter()
    loaded = retrieval._load_frozen_surrogate(context)
    try:
        teacher, teacher_evidence = retrieval._capture_dense_teacher(
            loaded,
            context["train_records"],
        )
    finally:
        del loaded
        gc.collect()
    full_heads = [
        (layer, head)
        for layer in range(retrieval._LAYERS)
        for head in range(retrieval._HEADS)
    ]
    control = retrieval._evaluate_native_development(
        context=context,
        records=context["train_records"],
        teacher=teacher,
        selected_heads=full_heads,
        role="train_full_W128_Q7_control",
    )
    candidate = _evaluate_candidate(
        context=context,
        records=context["train_records"],
        teacher=teacher,
        protocol=frozen,
    )
    passed = (
        teacher_evidence["passed"]
        and control["quality_passed"]
        and candidate["passed"]
    )
    post_authentication = _post_authentication(
        context,
        frozen["training_checkpoint"],
    )
    if not post_authentication or not all(post_authentication.values()):
        raise ValueError("retrieval episodic post-run authentication failed")
    report = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _RESULT_EXPERIMENT,
        "status": (
            "train_episodic_oracle_gate_passed"
            if passed
            else "train_episodic_oracle_gate_failed"
        ),
        "protocol": {
            "path": str(context["episodic_protocol_path"]),
            "sha256": context["episodic_protocol_sha256"],
        },
        "scope": {
            "split": "train",
            "development_outcomes_used": False,
            "confirmation_split_opened": False,
        },
        "reused_checkpoint_evidence": _checkpoint_baseline(training),
        "fresh_teacher_retrieval_evidence": teacher_evidence,
        "fresh_full_W128_Q7_control": control,
        "episodic_candidate": candidate,
        "decision": {
            "passed": passed,
            "confirmation_authorized": False,
            "next_step": (
                "freeze a distinct development episodic screen"
                if passed
                else "reject or revise the oracle episodic cache contract"
            ),
        },
        "post_run_authentication": post_authentication,
        "confirmation_split_opened": False,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prospective train-only OLMoE episodic span-cache oracle",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--base-protocol", required=True)
    freeze.add_argument("--base-protocol-sha256", required=True)
    freeze.add_argument("--training-checkpoint", required=True)
    freeze.add_argument("--training-checkpoint-sha256", required=True)
    freeze.add_argument("--episodic-library", required=True)
    freeze.add_argument("--episodic-library-sha256", required=True)
    freeze.add_argument("--out", required=True)
    screen = commands.add_parser("screen")
    screen.add_argument("--protocol", required=True)
    screen.add_argument("--protocol-sha256", required=True)
    screen.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "freeze":
        value = freeze_episodic_oracle_protocol(
            base_protocol=args.base_protocol,
            base_protocol_sha256=args.base_protocol_sha256,
            training_checkpoint=args.training_checkpoint,
            training_checkpoint_sha256=args.training_checkpoint_sha256,
            episodic_library=args.episodic_library,
            episodic_library_sha256=args.episodic_library_sha256,
            out=args.out,
        )
    elif args.command == "screen":
        value = screen_episodic_oracle(
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            out=args.out,
        )
    else:  # pragma: no cover - argparse owns this boundary
        raise AssertionError("unknown retrieval episodic command")
    print(json.dumps(value, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
