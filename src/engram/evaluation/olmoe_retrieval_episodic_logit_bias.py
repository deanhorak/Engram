"""Train-only calibration of the fixed K256 episodic softmax partition.

The ranked K64/K96/K128/K165 payload-cache family failed its strict train
gate.  Its least-bad member, K165, was still materially worse than the
previously authenticated all-head K256 payload oracle.  This evaluator
therefore fixes the head mask prospectively to K256 and changes exactly one
quantity: a shared additive float32 bias on episodic logits.

The ordered candidates are gamma={1/2,1/4,3/16,1/8}, represented as
beta=ln(gamma) with exact frozen float32 bit patterns.  Every candidate runs
all eight training records.  The first strict mean/worst/no-regression pass is
reset-replayed before selection.  If none passes, the lexicographically best
(worst loss, mean loss, candidate order) failure alone is reset-replayed.

The new ABI is not trusted merely because beta=0 is intended to be a no-op.
The ``parity`` command runs the historical all-head V1 DSO and the proposed V2
DSO at explicit beta=0 on the same authenticated real-package record, then
requires exact outputs, deterministic counters, and reset replay.  That
generated report is a required input to ``freeze``.

No dense teacher, development outcome, or confirmation file is accessed by
any command in this module.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import engram.evaluation.olmoe_retrieval_episodic_oracle as episodic
import engram.evaluation.olmoe_retrieval_episodic_rank_sweep as rank
from engram.compiler.olmoe_native import validate_olmoe_native_package
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.utils import atomic_json, sha256_file, sha256_json


_SCHEMA_VERSION = 1
_PARITY_EXPERIMENT = "olmoe_q7_retrieval_episodic_logit_bias_v2_parity"
_PROTOCOL_EXPERIMENT = "olmoe_q7_retrieval_episodic_logit_bias_protocol"
_RESULT_EXPERIMENT = "olmoe_q7_retrieval_episodic_logit_bias_train_screen"
_PARITY_STATUS = "beta_zero_v1_v2_parity_passed"
_PROTOCOL_STATUS = "frozen_before_episodic_logit_bias_train_candidate_execution"
_EXPECTED_RANK_PROTOCOL_SHA256 = (
    "e238b5cfc4359422abc96c227f4010ade747ff161c946b2edae14c171a7bf04c"
)
_EXPECTED_RANK_RESULT_SHA256 = (
    "a6fb045bbad526411b8318e7de2412ea56d69b3f2be0d977ca6819635c0718da"
)
_FIXED_K = 256
_FIXED_MASK_SHA256 = "924932a9df2ba2670f1d44a0f55a5e7aae42a12582ccca2440b5b320ce9b9922"
_EXPECTED_K165_MASK_SHA256 = rank._EXPECTED_MASK_SHA256[165]
_EXPECTED_K165_RESOURCE = rank._EXPECTED_RESOURCES[165]
_EXPECTED_K256_RESOURCE = {
    "combined_state_bytes": 10_534_912,
    "combined_scratch_bytes": 4_864,
    "combined_attention_and_episodic_read_bytes": 710_672_384,
    "combined_attention_and_episodic_traffic_bytes": 714_866_688,
}
_RECORDS = rank._RECORDS
_POSITIONS = rank._POSITIONS
_ANSWER_POSITIONS = rank._ANSWER_POSITIONS
_THREADS = rank._THREADS
_SLOTS = rank._SLOTS
_SPAN_TOKENS = rank._SPAN_TOKENS
_REQUIRED_V1_SYMBOL = "engram_olmoe_token_open_episodic_v1"
_REQUIRED_V2_SYMBOL = "engram_olmoe_token_open_episodic_headwise_v2"
_BIAS_CANDIDATES = (
    {
        "candidate_id": "gamma_1_2",
        "order": 0,
        "gamma_numerator": 1,
        "gamma_denominator": 2,
        "gamma_float32": 0.5,
        "gamma_float32_bits": "0x3f000000",
        "beta_float32": -0.6931471824645996,
        "beta_float32_bits": "0xbf317218",
    },
    {
        "candidate_id": "gamma_1_4",
        "order": 1,
        "gamma_numerator": 1,
        "gamma_denominator": 4,
        "gamma_float32": 0.25,
        "gamma_float32_bits": "0x3e800000",
        "beta_float32": -1.3862943649291992,
        "beta_float32_bits": "0xbfb17218",
    },
    {
        "candidate_id": "gamma_3_16",
        "order": 2,
        "gamma_numerator": 3,
        "gamma_denominator": 16,
        "gamma_float32": 0.1875,
        "gamma_float32_bits": "0x3e400000",
        "beta_float32": -1.6739764213562012,
        "beta_float32_bits": "0xbfd644dc",
    },
    {
        "candidate_id": "gamma_1_8",
        "order": 3,
        "gamma_numerator": 1,
        "gamma_denominator": 8,
        "gamma_float32": 0.125,
        "gamma_float32_bits": "0x3e000000",
        "beta_float32": -2.079441547393799,
        "beta_float32_bits": "0xc0051592",
    },
)
_SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *rank._SOURCE_FILES,
            "native/src/native_bitnet_token_runtime.cpp",
            "src/engram/evaluation/olmoe_retrieval_episodic_logit_bias.py",
            "src/engram/utils.py",
        )
    )
)
_EXPECTED_POST_AUTHENTICATION_KEYS = frozenset(
    {
        "attention_library",
        "base_protocol",
        "confirmation_not_opened",
        "corpus_manifest",
        "headwise_library",
        "historical_episodic_library",
        "layered_library",
        "logit_bias_library",
        "package",
        "rank_protocol",
        "rank_result",
        "source_config",
        "source_index",
        "source_shards",
        "train_split",
        "training_checkpoint",
    }
)


def _progress(message: str) -> None:
    print(f"[retrieval-episodic-logit-bias] {message}", file=sys.stderr, flush=True)


def _source_inventory() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    return {name: sha256_file(repository / name) for name in _SOURCE_FILES}


def _float32_bits(value: float) -> str:
    return f"0x{struct.unpack('>I', struct.pack('>f', np.float32(value)))[0]:08x}"


def _validated_bias_candidates(value: Any = _BIAS_CANDIDATES) -> list[dict[str, Any]]:
    if not isinstance(value, (tuple, list)) or len(value) != len(_BIAS_CANDIDATES):
        raise ValueError("retrieval episodic logit-bias candidate grid changed")
    normalized: list[dict[str, Any]] = []
    for expected, observed in zip(_BIAS_CANDIDATES, value, strict=True):
        if not isinstance(observed, Mapping) or dict(observed) != expected:
            raise ValueError("retrieval episodic logit-bias candidate changed")
        gamma = np.float32(
            int(observed["gamma_numerator"]) / int(observed["gamma_denominator"])
        )
        beta = np.float32(observed["beta_float32"])
        if (
            _float32_bits(float(gamma)) != observed["gamma_float32_bits"]
            or _float32_bits(float(beta)) != observed["beta_float32_bits"]
            or float(gamma) != float(observed["gamma_float32"])
            or _float32_bits(float(np.float32(math.log(float(gamma)))))
            != observed["beta_float32_bits"]
        ):
            raise ValueError("retrieval episodic logit-bias float32 bits changed")
        normalized.append(dict(observed))
    if [row["order"] for row in normalized] != list(range(len(normalized))):
        raise ValueError("retrieval episodic logit-bias order changed")
    return normalized


def _all_ones_mask() -> np.ndarray:
    mask = np.ones(
        (rank.retrieval._LAYERS, rank.retrieval._HEADS),
        dtype=np.bool_,
    )
    if int(mask.sum()) != _FIXED_K or sha256_json(mask.tolist()) != _FIXED_MASK_SHA256:
        raise ValueError("retrieval episodic K256 mask contract changed")
    return mask


def _fixed_mask_descriptor() -> dict[str, Any]:
    mask = _all_ones_mask()
    return {
        "K": _FIXED_K,
        "mask": mask.tolist(),
        "mask_sha256": _FIXED_MASK_SHA256,
        "selected_layer_head_pairs": _FIXED_K,
        "selected_heads_per_layer": [rank.retrieval._HEADS] * rank.retrieval._LAYERS,
        "active_layers": list(range(rank.retrieval._LAYERS)),
        "active_layer_count": rank.retrieval._LAYERS,
        "source": "fixed all-head K256 historical payload attribution",
        "fitted_by_this_experiment": False,
    }


def _fixed_resource_contract(
    model: Mapping[str, int],
    q7_expectations: Mapping[str, int],
) -> dict[str, Any]:
    resource = episodic._resource_contract(model, q7_expectations)
    observed = {name: int(resource[name]) for name in _EXPECTED_K256_RESOURCE}
    if observed != _EXPECTED_K256_RESOURCE:
        raise ValueError("retrieval episodic K256 resource contract changed")
    return resource


def _require_native_symbol(path: Path, symbol: str) -> None:
    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise ValueError(
            "retrieval episodic native library could not be loaded"
        ) from exc
    if not hasattr(library, symbol):
        raise ValueError(f"retrieval episodic native library lacks {symbol}")


def _validate_rank_failure_value(
    value: Any,
    *,
    protocol_path: Path,
    protocol_sha256: str,
    frozen_rank: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("retrieval episodic ranked failure is invalid")
    scope = value.get("scope")
    sweep = value.get("rank_sweep")
    decision = value.get("decision")
    authentication = value.get("post_run_authentication")
    if (
        value.get("schema_version") != rank._SCHEMA_VERSION
        or value.get("experiment") != rank._RESULT_EXPERIMENT
        or value.get("status") != "train_episodic_rank_sweep_gate_failed"
        or value.get("protocol")
        != {"path": str(protocol_path), "sha256": protocol_sha256}
        or not isinstance(scope, Mapping)
        or scope.get("dense_teacher_forwards") != 0
        or scope.get("development_outcomes_used") is not False
        or scope.get("confirmation_split_opened") is not False
        or not isinstance(sweep, Mapping)
        or sweep.get("candidate_order") != list(rank._CANDIDATE_K)
        or sweep.get("executed_candidates") != list(rank._CANDIDATE_K)
        or sweep.get("skipped_candidates") != []
        or sweep.get("passed") is not False
        or sweep.get("selected_K") != 165
        or sweep.get("selection_role") != "best_failed_candidate_for_diagnostic_replay"
        or not isinstance(decision, Mapping)
        or decision.get("passed") is not False
        or decision.get("semantic_gate_passed") is not False
        or decision.get("confirmation_authorized") is not False
        or not isinstance(authentication, Mapping)
        or not authentication
        or not all(check is True for check in authentication.values())
        or value.get("confirmation_split_opened") is not False
    ):
        raise ValueError("retrieval episodic ranked failure is invalid")
    outcomes = sweep.get("candidate_outcomes")
    descriptors = frozen_rank.get("candidates")
    if (
        not isinstance(outcomes, Mapping)
        or not isinstance(descriptors, list)
        or [row.get("K") for row in descriptors] != list(rank._CANDIDATE_K)
        or set(outcomes) != {f"K{k}" for k in rank._CANDIDATE_K}
    ):
        raise ValueError("retrieval episodic ranked population is invalid")
    for descriptor in descriptors:
        k = int(descriptor["K"])
        outcome = outcomes[f"K{k}"]
        if (
            not isinstance(outcome, Mapping)
            or outcome.get("K") != k
            or outcome.get("head_mask") != descriptor.get("head_mask")
            or outcome.get("resource_contract") != descriptor.get("resource_contract")
            or outcome.get("population_resource_passed") is not True
            or outcome.get("loss_gate", {}).get("passed") is not False
            or outcome.get("pre_replay_passed") is not False
            or outcome.get("passed") is not False
        ):
            raise ValueError(f"retrieval episodic ranked K{k} failure changed")
    selected = outcomes["K165"]
    selected_summary = selected["loss_gate"]["summaries"]["candidate"]
    expected_key = [
        float(selected_summary["maximum_answer_cross_entropy"]),
        float(selected_summary["mean_answer_cross_entropy"]),
        165,
    ]
    selected_resource = {
        name: int(selected["resource_contract"][name])
        for name in _EXPECTED_K165_RESOURCE
    }
    if (
        selected["head_mask"].get("mask_sha256") != _EXPECTED_K165_MASK_SHA256
        or selected_resource != _EXPECTED_K165_RESOURCE
        or selected.get("reset_replay", {}).get("passed") is not True
        or sweep.get("selection_key") != expected_key
        or value.get("K256_attribution") != frozen_rank.get("all_head_K256_attribution")
    ):
        raise ValueError("retrieval episodic ranked K165 attribution changed")
    return {
        "path": "",
        "sha256": "",
        "status": value["status"],
        "candidate_order": list(rank._CANDIDATE_K),
        "all_candidates_failed": True,
        "selected_K": 165,
        "selection_role": sweep["selection_role"],
        "selection_key": expected_key,
        "selected_head_mask": selected["head_mask"],
        "selected_resource_contract": selected["resource_contract"],
        "selected_loss_summary": selected["loss_gate"]["summaries"],
        "selected_reset_replay_passed": True,
        "systems_clean": True,
        "attribution_only": True,
    }


def _validate_historical_k256(
    *,
    frozen_rank: Mapping[str, Any],
    context: Mapping[str, Any],
    checkpoint: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    historical = frozen_rank.get("all_head_K256_attribution")
    if not isinstance(historical, Mapping):
        raise ValueError("retrieval episodic K256 attribution is missing")
    protocol_binding = historical.get("protocol")
    result_binding = historical.get("result")
    if not isinstance(protocol_binding, Mapping) or not isinstance(
        result_binding, Mapping
    ):
        raise ValueError("retrieval episodic K256 attribution binding is invalid")
    loaded = rank._validate_all_head_attribution(
        protocol_path=protocol_binding.get("path"),
        protocol_sha256=protocol_binding.get("sha256"),
        result_path=result_binding.get("path"),
        result_sha256=result_binding.get("sha256"),
        context=context,
        checkpoint=checkpoint,
    )
    loaded["strictly_better_than_K51_on_all_train_records"] = True
    if loaded != historical:
        raise ValueError("retrieval episodic K256 attribution changed")
    result_path = episodic._checked_file(
        result_binding["path"],
        result_binding["sha256"],
        "retrieval episodic K256 result",
    )
    result = rank.retrieval._read_json(result_path, "retrieval episodic K256 result")
    candidate = result.get("episodic_candidate")
    resource = _fixed_resource_contract(
        context["model"],
        context["q7_expectations"],
    )
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("resource_contract") != resource
        or [row.get("record_id") for row in candidate.get("sequence_evidence", [])]
        != loaded["record_ids"]
        or [
            float(row.get("answer_cross_entropy"))
            for row in candidate.get("sequence_evidence", [])
        ]
        != loaded["record_answer_cross_entropy"]
    ):
        raise ValueError("retrieval episodic K256 result evidence changed")
    return dict(loaded), resource


def _authenticate_fixed_inputs(
    *,
    rank_protocol: str | Path,
    rank_protocol_sha256: str,
    rank_result: str | Path,
    rank_result_sha256: str,
    logit_bias_library: str | Path,
    logit_bias_library_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if rank_protocol_sha256.lower() != _EXPECTED_RANK_PROTOCOL_SHA256:
        raise ValueError("retrieval episodic rank protocol root changed")
    if rank_result_sha256.lower() != _EXPECTED_RANK_RESULT_SHA256:
        raise ValueError("retrieval episodic rank result root changed")
    rank_path = episodic._checked_file(
        rank_protocol,
        rank_protocol_sha256,
        "retrieval episodic rank protocol",
    )
    result_path = episodic._checked_file(
        rank_result,
        rank_result_sha256,
        "retrieval episodic rank result",
    )
    frozen_rank = rank.retrieval._read_json(
        rank_path,
        "retrieval episodic rank protocol",
    )
    base = frozen_rank.get("base_retrieval_protocol")
    checkpoint_binding = frozen_rank.get("training_checkpoint")
    if (
        frozen_rank.get("schema_version") != rank._SCHEMA_VERSION
        or frozen_rank.get("experiment") != rank._PROTOCOL_EXPERIMENT
        or frozen_rank.get("status") != rank._PROTOCOL_STATUS
        or frozen_rank.get("candidate_order") != list(rank._CANDIDATE_K)
        or frozen_rank.get("confirmation_split_opened") is not False
        or not isinstance(base, Mapping)
        or not isinstance(checkpoint_binding, Mapping)
    ):
        raise ValueError("retrieval episodic rank protocol is invalid")
    train_scope = frozen_rank.get("train_scope")
    if (
        not isinstance(train_scope, Mapping)
        or train_scope.get("dense_teacher_forwards") != 0
        or train_scope.get("development_outcomes_used") is not False
        or train_scope.get("confirmation_file_access_permitted") is not False
    ):
        raise ValueError("retrieval episodic rank scope is invalid")
    new_library = episodic._checked_file(
        logit_bias_library,
        logit_bias_library_sha256,
        "retrieval episodic logit-bias library",
    )
    _require_native_symbol(new_library, _REQUIRED_V2_SYMBOL)
    context, training, selection, checkpoint = episodic._authenticate_base_inputs(
        base.get("path"),
        base.get("sha256"),
        checkpoint_binding.get("path"),
        checkpoint_binding.get("sha256"),
        new_library,
        logit_bias_library_sha256,
    )
    if (
        selection.get("screen_eligible") is not True
        or selection.get("selected_mask_name") != "M2"
        or checkpoint["path"] != checkpoint_binding.get("path")
        or checkpoint["sha256"] != checkpoint_binding.get("sha256")
    ):
        raise ValueError("retrieval episodic logit-bias checkpoint changed")
    historical, resource = _validate_historical_k256(
        frozen_rank=frozen_rank,
        context=context,
        checkpoint=checkpoint,
    )
    historical_library = episodic._checked_file(
        historical["historical_episodic_library"]["path"],
        historical["historical_episodic_library"]["sha256"],
        "retrieval episodic historical K256 library",
    )
    _require_native_symbol(historical_library, _REQUIRED_V1_SYMBOL)
    rank_result_value = rank.retrieval._read_json(
        result_path,
        "retrieval episodic rank result",
    )
    rank_failure = _validate_rank_failure_value(
        rank_result_value,
        protocol_path=rank_path,
        protocol_sha256=rank_protocol_sha256.lower(),
        frozen_rank=frozen_rank,
    )
    k256_losses = historical["record_answer_cross_entropy"]
    k165_summary = rank_failure["selected_loss_summary"]["candidate"]
    if (
        float(np.mean(k256_losses, dtype=np.float64))
        >= float(k165_summary["mean_answer_cross_entropy"])
        or float(max(k256_losses))
        >= float(k165_summary["maximum_answer_cross_entropy"])
        or int(resource["combined_attention_and_episodic_traffic_bytes"])
        <= int(
            rank_failure["selected_resource_contract"][
                "combined_attention_and_episodic_traffic_bytes"
            ]
        )
        or resource["within_total_traffic_budget"] is not True
    ):
        raise ValueError("retrieval episodic fixed-K256 calibration basis changed")
    rank_failure["path"] = str(result_path)
    rank_failure["sha256"] = rank_result_sha256.lower()
    context = dict(context)
    context.update(
        {
            "rank_protocol_path": rank_path,
            "rank_protocol_sha256": rank_protocol_sha256.lower(),
            "rank_result_path": result_path,
            "rank_result_sha256": rank_result_sha256.lower(),
            "logit_bias_library_path": new_library,
            "logit_bias_library_sha256": logit_bias_library_sha256.lower(),
            "historical_episodic_library_path": historical_library,
            "historical_episodic_library_sha256": historical[
                "historical_episodic_library"
            ]["sha256"],
            "rank_protocol": frozen_rank,
            "rank_failure": rank_failure,
        }
    )
    return (
        context,
        training,
        checkpoint,
        frozen_rank,
        rank_failure,
        {
            "historical": historical,
            "resource_contract": resource,
        },
    )


def _open_legacy_runtime(context: Mapping[str, Any]) -> Any:
    policy = rank.retrieval._BASE_POLICY
    return OLMoENativeTokenRuntime(
        context["config_path"],
        context["non_mlp_path"],
        context["q7_path"],
        context["historical_episodic_library_path"],
        threads=_THREADS,
        local_window=policy["local_window"],
        older_candidates=policy["older_candidates"],
        older_top_k=policy["older_top_k"],
        sink_tokens=policy["sink_tokens"],
        episodic_policy={"slots": _SLOTS, "span_size": _SPAN_TOKENS},
        episodic_logit_bias=None,
    )


def _open_bias_runtime(context: Mapping[str, Any], beta: float) -> Any:
    policy = rank.retrieval._BASE_POLICY
    return OLMoENativeTokenRuntime(
        context["config_path"],
        context["non_mlp_path"],
        context["q7_path"],
        context["logit_bias_library_path"],
        threads=_THREADS,
        local_window=policy["local_window"],
        older_candidates=policy["older_candidates"],
        older_top_k=policy["older_top_k"],
        sink_tokens=policy["sink_tokens"],
        episodic_policy={"slots": _SLOTS, "span_size": _SPAN_TOKENS},
        episodic_head_mask=_all_ones_mask(),
        episodic_logit_bias=float(np.float32(beta)),
    )


def _validate_runtime_route(
    runtime: Any,
    *,
    beta: float | None,
    expect_mask: bool,
) -> None:
    mask = getattr(runtime, "episodic_head_mask", None)
    observed_bias = getattr(runtime, "episodic_logit_bias", object())
    observed_abi = getattr(runtime, "episodic_open_abi", None)
    if (
        runtime.position != 0
        or not runtime.attention_metrics_available
        or not getattr(runtime, "episodic_metrics_available", False)
        or getattr(runtime, "episodic_policy", None)
        != {"slots": _SLOTS, "span_size": _SPAN_TOKENS}
    ):
        raise ValueError("retrieval episodic logit-bias runtime is unavailable")
    if expect_mask:
        if observed_abi != "v2" or not rank.fixed._runtime_mask_matches(
            runtime, _all_ones_mask()
        ):
            raise ValueError("retrieval episodic logit-bias runtime mask changed")
        if (
            beta is None
            or observed_bias is None
            or _float32_bits(float(observed_bias)) != _float32_bits(beta)
        ):
            raise ValueError("retrieval episodic logit-bias runtime beta changed")
    elif (
        observed_abi != "v1"
        or mask is not None
        or _float32_bits(float(observed_bias)) != "0x00000000"
    ):
        raise ValueError("retrieval episodic legacy runtime did not use V1")


def _execute_parity_record(
    runtime: Any,
    *,
    record: Mapping[str, Any],
    context: Mapping[str, Any],
    schedule: Mapping[str, Any],
    resource: Mapping[str, Any],
) -> dict[str, Any]:
    if runtime.position != 0:
        raise ValueError("retrieval episodic parity runtime was not reset")
    output_digest = hashlib.sha256()
    hidden_digest = hashlib.sha256()
    logits_digest = hashlib.sha256()
    counter_digest = hashlib.sha256()
    call_digest = hashlib.sha256()
    top1_tokens: list[int] = []
    all_counter_checks = True
    final_metrics: dict[str, int] | None = None
    previous_duplicate = 0
    for position, token_id in enumerate(record["input_ids"][:-1]):
        row = schedule["rows"][position]
        writes = [int(row["write_slot"])]
        reads = [int(row["read_span"])]
        result = runtime.forward_episodic([int(token_id)], writes, reads)
        if runtime.position != position + 1:
            raise ValueError("retrieval episodic parity position did not advance")
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
            raise ValueError("retrieval episodic parity diagnostics are invalid")
        final_metrics = dict(result.metrics)
        checks = episodic._counter_checks(
            final_metrics,
            context=context,
            schedule=schedule,
            positions=position + 1,
            resource=resource,
        )
        duplicate = int(final_metrics["episodic_duplicate_older_entries_suppressed"])
        checks["episodic_duplicate_suppressions_monotonic"] = (
            duplicate >= previous_duplicate
        )
        previous_duplicate = duplicate
        all_counter_checks = all_counter_checks and all(checks.values())
        top1_tokens.append(int(result.next_token))
        next_bytes = struct.pack(">q", int(result.next_token))
        output_digest.update(next_bytes)
        output_digest.update(hidden.tobytes())
        output_digest.update(logits.tobytes())
        hidden_digest.update(hidden.tobytes())
        logits_digest.update(logits.tobytes())
        episodic._counter_digest_update(counter_digest, final_metrics)
        call_digest.update(
            json.dumps(
                {
                    "position": position,
                    "write_slots": writes,
                    "read_spans": reads,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        call_digest.update(b"\n")
    if final_metrics is None or runtime.position != _POSITIONS:
        raise ValueError("retrieval episodic parity execution is incomplete")
    return {
        "record_index": int(record["record_index"]),
        "record_id": record["record_id"],
        "top1_tokens": top1_tokens,
        "output_sha256": output_digest.hexdigest(),
        "hidden_sha256": hidden_digest.hexdigest(),
        "logits_sha256": logits_digest.hexdigest(),
        "counter_stream_sha256": counter_digest.hexdigest(),
        "episodic_call_stream_sha256": call_digest.hexdigest(),
        "schedule_rows_sha256": schedule["rows_sha256"],
        "counter_stream_passed": all_counter_checks,
        "final_metrics": final_metrics,
        "final_position": runtime.position,
    }


def _parity_evidence_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left.get("record_index") == right.get("record_index")
        and left.get("record_id") == right.get("record_id")
        and left.get("top1_tokens") == right.get("top1_tokens")
        and left.get("output_sha256") == right.get("output_sha256")
        and left.get("hidden_sha256") == right.get("hidden_sha256")
        and left.get("logits_sha256") == right.get("logits_sha256")
        and left.get("counter_stream_sha256") == right.get("counter_stream_sha256")
        and left.get("episodic_call_stream_sha256")
        == right.get("episodic_call_stream_sha256")
        and left.get("schedule_rows_sha256") == right.get("schedule_rows_sha256")
        and left.get("counter_stream_passed") is True
        and right.get("counter_stream_passed") is True
        and episodic.sustained._deterministic_metrics(
            dict(left.get("final_metrics", {}))
        )
        == episodic.sustained._deterministic_metrics(
            dict(right.get("final_metrics", {}))
        )
    )


def _run_zero_bias_parity(
    *,
    context: Mapping[str, Any],
    record: Mapping[str, Any],
    schedule: Mapping[str, Any],
    resource: Mapping[str, Any],
    legacy_factory: Callable[[Mapping[str, Any]], Any] = _open_legacy_runtime,
    v2_factory: Callable[[Mapping[str, Any], float], Any] = _open_bias_runtime,
) -> dict[str, Any]:
    legacy = legacy_factory(context)
    try:
        _validate_runtime_route(legacy, beta=None, expect_mask=False)
        legacy_first = _execute_parity_record(
            legacy,
            record=record,
            context=context,
            schedule=schedule,
            resource=resource,
        )
        legacy.reset()
        if legacy.position != 0:
            raise ValueError("retrieval episodic legacy reset failed")
        legacy_replay = _execute_parity_record(
            legacy,
            record=record,
            context=context,
            schedule=schedule,
            resource=resource,
        )
    finally:
        legacy.close()
    v2 = v2_factory(context, 0.0)
    try:
        _validate_runtime_route(v2, beta=0.0, expect_mask=True)
        v2_first = _execute_parity_record(
            v2,
            record=record,
            context=context,
            schedule=schedule,
            resource=resource,
        )
        v2.reset()
        if v2.position != 0:
            raise ValueError("retrieval episodic V2 reset failed")
        v2_replay = _execute_parity_record(
            v2,
            record=record,
            context=context,
            schedule=schedule,
            resource=resource,
        )
    finally:
        v2.close()
    checks = {
        "legacy_reset_replay_exact": _parity_evidence_equal(
            legacy_first, legacy_replay
        ),
        "v2_reset_replay_exact": _parity_evidence_equal(v2_first, v2_replay),
        "v1_v2_first_outputs_and_counters_exact": _parity_evidence_equal(
            legacy_first, v2_first
        ),
        "v1_v2_replay_outputs_and_counters_exact": _parity_evidence_equal(
            legacy_replay, v2_replay
        ),
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise ValueError("retrieval episodic beta-zero V1/V2 parity failed")
    return {
        "legacy_v1_first": legacy_first,
        "legacy_v1_reset_replay": legacy_replay,
        "explicit_beta_zero_v2_first": v2_first,
        "explicit_beta_zero_v2_reset_replay": v2_replay,
        "checks": checks,
        "native_sequence_forwards": 4,
        "native_token_steps": 4 * _POSITIONS,
        "passed": True,
    }


def _post_input_authentication(
    context: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, str],
) -> dict[str, bool]:
    protocol = context["protocol"]
    source = protocol["source_model"]
    checks = {
        "base_protocol": (
            sha256_file(context["protocol_path"]) == context["protocol_sha256"]
        ),
        "rank_protocol": (
            sha256_file(context["rank_protocol_path"])
            == context["rank_protocol_sha256"]
        ),
        "rank_result": (
            sha256_file(context["rank_result_path"]) == context["rank_result_sha256"]
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
            rank.retrieval._source_shard_inventory(context["model_path"])
            == source["shard_sha256"]
        ),
        "training_checkpoint": (
            sha256_file(Path(checkpoint["path"])) == checkpoint["sha256"]
        ),
        "historical_episodic_library": (
            sha256_file(context["historical_episodic_library_path"])
            == context["historical_episodic_library_sha256"]
        ),
        "logit_bias_library": (
            sha256_file(context["logit_bias_library_path"])
            == context["logit_bias_library_sha256"]
        ),
    }
    for name, path in context["library_paths"].items():
        checks[f"{name}_library"] = (
            sha256_file(path) == protocol["libraries"][name]["sha256"]
        )
    return checks


def generate_beta_zero_parity_report(
    *,
    rank_protocol: str | Path,
    rank_protocol_sha256: str,
    rank_result: str | Path,
    rank_result_sha256: str,
    logit_bias_library: str | Path,
    logit_bias_library_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    output = rank.retrieval._new_output(
        out,
        "retrieval episodic beta-zero parity report",
    )
    _progress("authenticating rank failure, historical V1 DSO, and proposed V2 DSO")
    (
        context,
        _training,
        checkpoint,
        frozen_rank,
        rank_failure,
        fixed,
    ) = _authenticate_fixed_inputs(
        rank_protocol=rank_protocol,
        rank_protocol_sha256=rank_protocol_sha256,
        rank_result=rank_result,
        rank_result_sha256=rank_result_sha256,
        logit_bias_library=logit_bias_library,
        logit_bias_library_sha256=logit_bias_library_sha256,
    )
    anchors = frozen_rank["tokenizer_fact_anchor_ids"]
    record = context["train_records"][0]
    schedule = rank.fixed._derive_schedule(record["input_ids"], anchors)
    _progress("running full-record V1/V2 beta-zero first and reset-replay parity")
    parity = _run_zero_bias_parity(
        context=context,
        record=record,
        schedule=schedule,
        resource=fixed["resource_contract"],
    )
    post = _post_input_authentication(context, checkpoint=checkpoint)
    if not post or not all(post.values()):
        raise ValueError("retrieval episodic parity post-run authentication failed")
    report = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _PARITY_EXPERIMENT,
        "status": _PARITY_STATUS,
        "rank_protocol": {
            "path": str(context["rank_protocol_path"]),
            "sha256": context["rank_protocol_sha256"],
        },
        "rank_result": {
            "path": str(context["rank_result_path"]),
            "sha256": context["rank_result_sha256"],
            "authenticated_failure": rank_failure,
        },
        "legacy_v1_library": {
            "path": str(context["historical_episodic_library_path"]),
            "sha256": context["historical_episodic_library_sha256"],
            "required_symbol": _REQUIRED_V1_SYMBOL,
        },
        "proposed_v2_library": {
            "path": str(context["logit_bias_library_path"]),
            "sha256": context["logit_bias_library_sha256"],
            "required_symbol": _REQUIRED_V2_SYMBOL,
        },
        "package": {
            "path": str(context["package_path"]),
            "manifest_sha256": context["protocol"]["package"]["manifest_sha256"],
        },
        "fixed_arm": {
            "head_mask": _fixed_mask_descriptor(),
            "resource_contract": fixed["resource_contract"],
            "historical_K256_attribution": fixed["historical"],
        },
        "beta_zero": {
            "value": 0.0,
            "float32_bits": "0x00000000",
            "legacy_route": "episodic V1 all-head open",
            "new_route": "episodic headwise V2 all-ones open",
        },
        "scope": {
            "split": "train",
            "record_index": 0,
            "positions": _POSITIONS,
            "dense_teacher_forwards": 0,
            "development_outcomes_used": False,
            "confirmation_split_opened": False,
        },
        "schedule_rows_sha256": schedule["rows_sha256"],
        "parity": parity,
        "source_sha256": _source_inventory(),
        "post_run_authentication": post,
        "confirmation_split_opened": False,
    }
    atomic_json(output, report)
    _progress(f"beta-zero parity report written to {output}")
    return report


def _validate_parity_report(
    *,
    path: str | Path,
    expected_sha256: str,
    context: Mapping[str, Any],
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    source = episodic._checked_file(
        path,
        expected_sha256,
        "retrieval episodic beta-zero parity report",
    )
    value = rank.retrieval._read_json(
        source,
        "retrieval episodic beta-zero parity report",
    )
    parity = value.get("parity")
    scope = value.get("scope")
    post = value.get("post_run_authentication")
    schedule = rank.fixed._derive_schedule(
        context["train_records"][0]["input_ids"],
        context["rank_protocol"]["tokenizer_fact_anchor_ids"],
    )
    evidence_names = (
        "legacy_v1_first",
        "legacy_v1_reset_replay",
        "explicit_beta_zero_v2_first",
        "explicit_beta_zero_v2_reset_replay",
    )
    evidence = (
        [parity.get(name) for name in evidence_names]
        if isinstance(parity, Mapping)
        else []
    )
    evidence_valid = len(evidence) == len(evidence_names) and all(
        isinstance(row, Mapping)
        and row.get("record_index") == 0
        and row.get("record_id") == context["train_records"][0]["record_id"]
        and row.get("final_position") == _POSITIONS
        and row.get("schedule_rows_sha256") == schedule["rows_sha256"]
        and isinstance(row.get("top1_tokens"), list)
        and len(row["top1_tokens"]) == _POSITIONS
        and all(
            rank.retrieval._is_sha256(row.get(name))
            for name in (
                "output_sha256",
                "hidden_sha256",
                "logits_sha256",
                "counter_stream_sha256",
                "episodic_call_stream_sha256",
            )
        )
        and all(
            episodic._counter_checks(
                row.get("final_metrics", {}),
                context=context,
                schedule=schedule,
                positions=_POSITIONS,
                resource=fixed["resource_contract"],
            ).values()
        )
        for row in evidence
    )
    parity_cross_checks = (
        evidence_valid
        and _parity_evidence_equal(evidence[0], evidence[1])
        and _parity_evidence_equal(evidence[0], evidence[2])
        and _parity_evidence_equal(evidence[2], evidence[3])
    )
    if (
        value.get("schema_version") != _SCHEMA_VERSION
        or value.get("experiment") != _PARITY_EXPERIMENT
        or value.get("status") != _PARITY_STATUS
        or value.get("rank_protocol")
        != {
            "path": str(context["rank_protocol_path"]),
            "sha256": context["rank_protocol_sha256"],
        }
        or value.get("rank_result", {}).get("path") != str(context["rank_result_path"])
        or value.get("rank_result", {}).get("sha256") != context["rank_result_sha256"]
        or value.get("rank_result", {}).get("authenticated_failure")
        != context["rank_failure"]
        or value.get("legacy_v1_library")
        != {
            "path": str(context["historical_episodic_library_path"]),
            "sha256": context["historical_episodic_library_sha256"],
            "required_symbol": _REQUIRED_V1_SYMBOL,
        }
        or value.get("proposed_v2_library")
        != {
            "path": str(context["logit_bias_library_path"]),
            "sha256": context["logit_bias_library_sha256"],
            "required_symbol": _REQUIRED_V2_SYMBOL,
        }
        or value.get("fixed_arm", {}).get("head_mask") != _fixed_mask_descriptor()
        or value.get("fixed_arm", {}).get("resource_contract")
        != fixed["resource_contract"]
        or value.get("fixed_arm", {}).get("historical_K256_attribution")
        != fixed["historical"]
        or value.get("package")
        != {
            "path": str(context["package_path"]),
            "manifest_sha256": context["protocol"]["package"]["manifest_sha256"],
        }
        or value.get("beta_zero")
        != {
            "value": 0.0,
            "float32_bits": "0x00000000",
            "legacy_route": "episodic V1 all-head open",
            "new_route": "episodic headwise V2 all-ones open",
        }
        or not isinstance(scope, Mapping)
        or scope.get("split") != "train"
        or scope.get("record_index") != 0
        or scope.get("positions") != _POSITIONS
        or scope.get("dense_teacher_forwards") != 0
        or scope.get("development_outcomes_used") is not False
        or scope.get("confirmation_split_opened") is not False
        or not isinstance(parity, Mapping)
        or parity.get("passed") is not True
        or parity.get("native_sequence_forwards") != 4
        or parity.get("native_token_steps") != 4 * _POSITIONS
        or parity.get("checks", {}).get("passed") is not True
        or not all(check is True for check in parity["checks"].values())
        or parity.get("checks")
        != {
            "legacy_reset_replay_exact": True,
            "v2_reset_replay_exact": True,
            "v1_v2_first_outputs_and_counters_exact": True,
            "v1_v2_replay_outputs_and_counters_exact": True,
            "passed": True,
        }
        or not parity_cross_checks
        or value.get("schedule_rows_sha256") != schedule["rows_sha256"]
        or value.get("source_sha256") != _source_inventory()
        or not isinstance(post, Mapping)
        or set(post) != _EXPECTED_POST_AUTHENTICATION_KEYS
        or not all(check is True for check in post.values())
        or value.get("confirmation_split_opened") is not False
    ):
        raise ValueError("retrieval episodic beta-zero parity report is invalid")
    return {
        "path": str(source),
        "sha256": expected_sha256.lower(),
        "status": value["status"],
        "fixed_K": _FIXED_K,
        "beta_float32_bits": "0x00000000",
        "outputs_counters_and_reset_exact": True,
        "native_sequence_forwards": 4,
        "native_token_steps": 4 * _POSITIONS,
    }


def _build_protocol(
    *,
    context: Mapping[str, Any],
    training: Mapping[str, Any],
    checkpoint: Mapping[str, str],
    frozen_rank: Mapping[str, Any],
    rank_failure: Mapping[str, Any],
    fixed: Mapping[str, Any],
    parity: Mapping[str, Any],
) -> dict[str, Any]:
    state = rank.fixed._checkpoint_references(training)
    anchors = frozen_rank["tokenizer_fact_anchor_ids"]
    schedules = [
        rank.fixed._derive_schedule(record["input_ids"], anchors)
        for record in context["train_records"]
    ]
    candidates = _validated_bias_candidates()
    return {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _PROTOCOL_EXPERIMENT,
        "status": _PROTOCOL_STATUS,
        "rank_protocol": {
            "path": str(context["rank_protocol_path"]),
            "sha256": context["rank_protocol_sha256"],
        },
        "rank_result": {
            "path": str(context["rank_result_path"]),
            "sha256": context["rank_result_sha256"],
            "authenticated_failure": dict(rank_failure),
        },
        "training_checkpoint": {
            "path": checkpoint["path"],
            "sha256": checkpoint["sha256"],
            "training_sha256": sha256_json(training),
        },
        "logit_bias_library": {
            "path": str(context["logit_bias_library_path"]),
            "sha256": context["logit_bias_library_sha256"],
            "required_open_symbol": _REQUIRED_V2_SYMBOL,
        },
        "beta_zero_parity": dict(parity),
        "fixed_arm": {
            "K": _FIXED_K,
            "head_mask": _fixed_mask_descriptor(),
            "resource_contract": fixed["resource_contract"],
            "historical_K256_attribution": fixed["historical"],
            "selection_basis": (
                "K256 strictly dominated rank-selected K165 on authenticated "
                "raw payload mean/worst train loss for 0.5511 percentage "
                "points additional dense-relative traffic"
            ),
            "fitted_or_selected_by_bias_outcomes": False,
        },
        "rank_family_attribution": {
            "all_failed": True,
            "K165_is_diagnostic_only": True,
            "K165": {
                "head_mask": rank_failure["selected_head_mask"],
                "resource_contract": rank_failure["selected_resource_contract"],
                "loss_summary": rank_failure["selected_loss_summary"],
                "reset_replay_passed": True,
            },
        },
        "candidate_order": [row["candidate_id"] for row in candidates],
        "candidates": candidates,
        "bias_semantics": {
            "operation": "episodic_score_plus_beta",
            "beta_definition": "float32(log(gamma))",
            "effect": "scale only the episodic softmax partition by gamma",
            "base_attention_logits_changed": False,
            "episodic_values_changed": False,
            "resource_schedule_changed": False,
            "analytic_per_sequence": {
                "episodic_logit_bias_additions": (
                    _ANSWER_POSITIONS * _SPAN_TOKENS * _FIXED_K
                ),
                "added_state_bytes": 0,
                "added_scratch_bytes": 0,
                "added_read_bytes": 0,
                "added_write_bytes": 0,
                "added_total_traffic_bytes": 0,
                "new_metric_abi_fields": 0,
            },
        },
        "tokenizer_fact_anchor_ids": {
            label: list(values) for label, values in anchors.items()
        },
        "schedule_contract": {
            "derivation_input": "input_ids[0:97] and authenticated tokenizer anchors",
            "last_input_index_observed": rank._ANSWER_START,
            "write_source_starts": list(rank.retrieval._PASSKEY_SOURCE_STARTS),
            "payload_spans": 4,
            "payload_tokens_per_span": _SPAN_TOKENS,
            "unique_canonical_slots": _SLOTS,
            "answer_prediction_rows": list(range(rank._ANSWER_START, _POSITIONS)),
            "per_record_rows_sha256": [
                schedule["rows_sha256"] for schedule in schedules
            ],
        },
        "runtime_abi": {
            "class": "OLMoENativeTokenRuntime",
            "method": "forward_episodic",
            "required_open_symbol": _REQUIRED_V2_SYMBOL,
            "episodic_policy": {"slots": _SLOTS, "span_size": _SPAN_TOKENS},
            "mask_argument": "episodic_head_mask=all_ones_K256",
            "bias_argument": "episodic_logit_bias=float32_beta",
        },
        "execution_contract": {
            "candidate_order": "least intervention first",
            "all_eight_records_before_each_gate": True,
            "strict_gate_reference": "authenticated M2 train losses",
            "strict_mean_improvement": True,
            "strict_worst_improvement": True,
            "per_record_regression_permitted": False,
            "first_strict_pass_requires_reset_replay_before_selection": True,
            "stop_after_first_replay_qualified_pass": True,
            "total_failure_selection_key": [
                "maximum_answer_cross_entropy",
                "mean_answer_cross_entropy",
                "candidate_order",
            ],
            "total_failure_reset_replay": "lexicographically best failure only",
            "systems_or_replay_failure": "abort without selection",
        },
        "train_scope": {
            "records": _RECORDS,
            "positions_per_record": _POSITIONS,
            "answer_positions_per_record": _ANSWER_POSITIONS,
            "dense_teacher_forwards": 0,
            "candidate_mask_fixed_before_execution": True,
            "candidate_selection_uses_train_outcomes": True,
            "development_outcomes_used": False,
            "confirmation_file_access_permitted": False,
            "cpu_only_candidate_execution": True,
        },
        "reused_checkpoint_evidence": state["baselines"],
        "source_sha256": _source_inventory(),
        "authenticated_confirmation_descriptor": dict(
            context["confirmation_descriptor"]
        ),
        "confirmation_split_opened": False,
    }


def freeze_episodic_logit_bias_protocol(
    *,
    rank_protocol: str | Path,
    rank_protocol_sha256: str,
    rank_result: str | Path,
    rank_result_sha256: str,
    logit_bias_library: str | Path,
    logit_bias_library_sha256: str,
    parity_report: str | Path,
    parity_report_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    output = rank.retrieval._new_output(
        out,
        "retrieval episodic logit-bias protocol",
    )
    (
        context,
        training,
        checkpoint,
        frozen_rank,
        rank_failure,
        fixed,
    ) = _authenticate_fixed_inputs(
        rank_protocol=rank_protocol,
        rank_protocol_sha256=rank_protocol_sha256,
        rank_result=rank_result,
        rank_result_sha256=rank_result_sha256,
        logit_bias_library=logit_bias_library,
        logit_bias_library_sha256=logit_bias_library_sha256,
    )
    parity = _validate_parity_report(
        path=parity_report,
        expected_sha256=parity_report_sha256,
        context=context,
        fixed=fixed,
    )
    protocol = _build_protocol(
        context=context,
        training=training,
        checkpoint=checkpoint,
        frozen_rank=frozen_rank,
        rank_failure=rank_failure,
        fixed=fixed,
        parity=parity,
    )
    atomic_json(output, protocol)
    return {"path": str(output), "sha256": sha256_file(output), "protocol": protocol}


def _authenticate_protocol(
    protocol: str | Path,
    protocol_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = episodic._checked_file(
        protocol,
        protocol_sha256,
        "retrieval episodic logit-bias protocol",
    )
    value = rank.retrieval._read_json(
        source,
        "retrieval episodic logit-bias protocol",
    )
    rank_binding = value.get("rank_protocol")
    result_binding = value.get("rank_result")
    library = value.get("logit_bias_library")
    parity_binding = value.get("beta_zero_parity")
    if not all(
        isinstance(binding, Mapping)
        for binding in (rank_binding, result_binding, library, parity_binding)
    ):
        raise ValueError("retrieval episodic logit-bias bindings are invalid")
    (
        context,
        training,
        checkpoint,
        frozen_rank,
        rank_failure,
        fixed,
    ) = _authenticate_fixed_inputs(
        rank_protocol=rank_binding.get("path"),
        rank_protocol_sha256=rank_binding.get("sha256"),
        rank_result=result_binding.get("path"),
        rank_result_sha256=result_binding.get("sha256"),
        logit_bias_library=library.get("path"),
        logit_bias_library_sha256=library.get("sha256"),
    )
    parity = _validate_parity_report(
        path=parity_binding.get("path"),
        expected_sha256=parity_binding.get("sha256"),
        context=context,
        fixed=fixed,
    )
    expected = _build_protocol(
        context=context,
        training=training,
        checkpoint=checkpoint,
        frozen_rank=frozen_rank,
        rank_failure=rank_failure,
        fixed=fixed,
        parity=parity,
    )
    if value != expected:
        raise ValueError("retrieval episodic logit-bias protocol contract changed")
    context = dict(context)
    context["logit_bias_protocol_path"] = source
    context["logit_bias_protocol_sha256"] = protocol_sha256.lower()
    context["logit_bias_protocol"] = expected
    context["parity_report_path"] = Path(parity["path"]).resolve()
    context["parity_report_sha256"] = parity["sha256"]
    return context, training, expected


def _candidate_resource_checks(
    evidence: Sequence[Mapping[str, Any]],
    *,
    context: Mapping[str, Any],
    schedule: Mapping[str, Any],
    resource: Mapping[str, Any],
) -> dict[str, bool]:
    final_expected = episodic._schedule_counters(
        schedule,
        positions=_POSITIONS,
        model=context["model"],
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
            int(row["final_metrics"]["attention_logical_read_bytes"])
            + int(row["final_metrics"]["episodic_write_bytes"])
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
    candidate: Mapping[str, Any],
    resource: Mapping[str, Any],
) -> dict[str, Any]:
    beta = float(candidate["beta_float32"])
    _validate_runtime_route(runtime, beta=beta, expect_mask=True)
    evidence: list[dict[str, Any]] = []
    for record, schedule in zip(records, schedules, strict=True):
        _logits, _hidden, row = episodic._execute_episodic_record(
            runtime,
            record=record,
            context=context,
            schedule=schedule,
            resource=resource,
            progress_label=(
                f"{candidate['candidate_id']} train record "
                f"{int(record['record_index']) + 1}/{_RECORDS}"
            ),
        )
        row = dict(row)
        row.update(
            {
                "K": _FIXED_K,
                "candidate_id": candidate["candidate_id"],
                "beta_float32": beta,
                "beta_float32_bits": candidate["beta_float32_bits"],
            }
        )
        evidence.append(row)
    loss_gate = rank._loss_gate(
        records=records,
        baselines=baselines,
        evidence=evidence,
        k=_FIXED_K,
    )
    resource_checks = _candidate_resource_checks(
        evidence,
        context=context,
        schedule=schedules[0],
        resource=resource,
    )
    resource_passed = all(resource_checks.values())
    return {
        "candidate": dict(candidate),
        "role": "fixed_K256_all_head_episodic_logit_bias_candidate",
        "head_mask": _fixed_mask_descriptor(),
        "resource_contract": resource,
        "population_native_sequence_forwards": _RECORDS,
        "population_native_token_steps": _RECORDS * _POSITIONS,
        "sequence_evidence": evidence,
        "loss_gate": loss_gate,
        "population_resource_checks": resource_checks,
        "population_resource_passed": resource_passed,
        "pre_replay_passed": loss_gate["passed"] and resource_passed,
        "reset_replay": {"executed": False, "native_sequence_forwards": 0},
        "passed": False,
    }


def _attach_reset_replay(
    outcome: dict[str, Any],
    runtime: Any,
    *,
    context: Mapping[str, Any],
    record: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> None:
    candidate = outcome["candidate"]
    _logits, _hidden, replay = episodic._execute_episodic_record(
        runtime,
        record=record,
        context=context,
        schedule=schedule,
        resource=outcome["resource_contract"],
        progress_label=f"{candidate['candidate_id']} retained-runtime reset replay",
    )
    replay = dict(replay)
    replay.update(
        {
            "K": _FIXED_K,
            "candidate_id": candidate["candidate_id"],
            "beta_float32": candidate["beta_float32"],
            "beta_float32_bits": candidate["beta_float32_bits"],
        }
    )
    checks = episodic._replay_checks(replay, outcome["sequence_evidence"][0])
    checks["candidate_id"] = (
        replay["candidate_id"] == outcome["sequence_evidence"][0]["candidate_id"]
    )
    checks["beta_float32_bits"] = (
        replay["beta_float32_bits"]
        == outcome["sequence_evidence"][0]["beta_float32_bits"]
    )
    checks["passed"] = all(value for name, value in checks.items() if name != "passed")
    outcome["reset_replay"] = {
        "executed": True,
        "native_sequence_forwards": 1,
        "native_token_steps": _POSITIONS,
        "checks": checks,
        "passed": checks["passed"],
    }
    outcome["passed"] = outcome["pre_replay_passed"] and checks["passed"]


def _run_bias_sweep(
    *,
    context: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    baselines: Mapping[str, Mapping[str, Any]],
    runtime_factory: Callable[[Mapping[str, Any], float], Any] = _open_bias_runtime,
) -> dict[str, Any]:
    if (
        len(records) != _RECORDS
        or [row.get("record_index") for row in records] != list(range(_RECORDS))
        or protocol.get("candidate_order")
        != [row["candidate_id"] for row in _BIAS_CANDIDATES]
    ):
        raise ValueError("retrieval episodic logit-bias population changed")
    candidates = _validated_bias_candidates(protocol.get("candidates"))
    resource = protocol["fixed_arm"]["resource_contract"]
    anchors = protocol["tokenizer_fact_anchor_ids"]
    schedules = [
        rank.fixed._derive_schedule(record["input_ids"], anchors) for record in records
    ]
    if [row["rows_sha256"] for row in schedules] != protocol["schedule_contract"][
        "per_record_rows_sha256"
    ]:
        raise ValueError("retrieval episodic logit-bias schedule changed")
    outcomes: dict[str, dict[str, Any]] = {}
    manifest: list[dict[str, Any]] = []
    retained_runtime: Any | None = None
    retained_id: str | None = None
    retained_key: tuple[float, float, int] | None = None
    active_runtime: Any | None = None
    selected_id: str | None = None
    selection_role: str | None = None
    try:
        for index, candidate in enumerate(candidates):
            candidate_id = str(candidate["candidate_id"])
            active_runtime = runtime_factory(
                context,
                float(candidate["beta_float32"]),
            )
            outcome = _candidate_population(
                active_runtime,
                context=context,
                records=records,
                schedules=schedules,
                baselines=baselines,
                candidate=candidate,
                resource=resource,
            )
            outcomes[candidate_id] = outcome
            manifest.append(
                {
                    "candidate_id": candidate_id,
                    "order": index,
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
                    f"retrieval episodic logit-bias {candidate_id} "
                    "systems contract failed"
                )
            summary = outcome["loss_gate"]["summaries"]["candidate"]
            key = (
                float(summary["maximum_answer_cross_entropy"]),
                float(summary["mean_answer_cross_entropy"]),
                index,
            )
            if outcome["pre_replay_passed"]:
                if retained_runtime is not None:
                    retained_runtime.close()
                    retained_runtime = None
                    retained_id = None
                    retained_key = None
                _attach_reset_replay(
                    outcome,
                    active_runtime,
                    context=context,
                    record=records[0],
                    schedule=schedules[0],
                )
                if not outcome["reset_replay"]["passed"]:
                    raise ValueError(
                        f"retrieval episodic logit-bias {candidate_id} "
                        "reset replay failed"
                    )
                active_runtime.close()
                active_runtime = None
                selected_id = candidate_id
                selection_role = "first_replay_qualified_strict_pass"
                manifest[-1]["status"] = "selected_first_replay_qualified_pass"
                manifest[-1]["reset_replay_executed"] = True
                for skipped in candidates[index + 1 :]:
                    manifest.append(
                        {
                            "candidate_id": skipped["candidate_id"],
                            "order": skipped["order"],
                            "executed": False,
                            "status": "skipped_after_first_qualified_pass",
                            "population_native_sequence_forwards": 0,
                            "reset_replay_executed": False,
                        }
                    )
                break
            if retained_key is None or key < retained_key:
                if retained_runtime is not None:
                    retained_runtime.close()
                retained_runtime = active_runtime
                retained_id = candidate_id
                retained_key = key
                active_runtime = None
            else:
                active_runtime.close()
                active_runtime = None
        else:
            if retained_runtime is None or retained_id is None:
                raise ValueError(
                    "retrieval episodic logit-bias failed to retain a candidate"
                )
            selected_id = retained_id
            selection_role = "best_failed_candidate_for_diagnostic_replay"
            selected = outcomes[selected_id]
            _attach_reset_replay(
                selected,
                retained_runtime,
                context=context,
                record=records[0],
                schedule=schedules[0],
            )
            if not selected["reset_replay"]["passed"]:
                raise ValueError(
                    f"retrieval episodic logit-bias {selected_id} reset replay failed"
                )
            retained_runtime.close()
            retained_runtime = None
            for row in manifest:
                if row["candidate_id"] == selected_id:
                    row["status"] = "best_failed_candidate_replayed"
                    row["reset_replay_executed"] = True
                    break
    finally:
        if active_runtime is not None:
            active_runtime.close()
        if retained_runtime is not None:
            retained_runtime.close()
    if (
        selected_id is None
        or selection_role is None
        or len(manifest) != len(candidates)
    ):
        raise ValueError("retrieval episodic logit-bias sweep is incomplete")
    selected = outcomes[selected_id]
    executed = [row["candidate_id"] for row in manifest if row["executed"]]
    skipped = [row["candidate_id"] for row in manifest if not row["executed"]]
    replay_count = sum(
        int(row["reset_replay"].get("native_sequence_forwards", 0))
        for row in outcomes.values()
    )
    passed = (
        selection_role == "first_replay_qualified_strict_pass" and selected["passed"]
    )
    summary = selected["loss_gate"]["summaries"]["candidate"]
    return {
        "candidate_order": [row["candidate_id"] for row in candidates],
        "execution_manifest": manifest,
        "executed_candidates": executed,
        "skipped_candidates": skipped,
        "candidate_outcomes": {
            candidate_id: outcomes[candidate_id] for candidate_id in executed
        },
        "selected_candidate_id": selected_id,
        "selected_candidate": selected["candidate"],
        "selection_role": selection_role,
        "selection_key": [
            float(summary["maximum_answer_cross_entropy"]),
            float(summary["mean_answer_cross_entropy"]),
            int(selected["candidate"]["order"]),
        ],
        "population_native_sequence_forwards": len(executed) * _RECORDS,
        "reset_replay_native_sequence_forwards": replay_count,
        "total_native_sequence_forwards": len(executed) * _RECORDS + replay_count,
        "passed": passed,
    }


def _screen_post_authentication(
    context: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
) -> dict[str, bool]:
    checks = _post_input_authentication(context, checkpoint=checkpoint)
    checks.update(
        {
            "logit_bias_protocol": (
                sha256_file(context["logit_bias_protocol_path"])
                == context["logit_bias_protocol_sha256"]
            ),
            "beta_zero_parity_report": (
                sha256_file(context["parity_report_path"])
                == context["parity_report_sha256"]
            ),
            "logit_bias_source_inventory": (
                context["logit_bias_protocol"]["source_sha256"] == _source_inventory()
            ),
        }
    )
    return checks


def screen_episodic_logit_bias(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    output = rank.retrieval._new_output(
        out,
        "retrieval episodic logit-bias result",
    )
    started = time.perf_counter()
    _progress("authenticating frozen K256 logit-bias protocol and beta-zero parity")
    context, training, frozen = _authenticate_protocol(protocol, protocol_sha256)
    state = rank.fixed._checkpoint_references(training)
    _progress("starting ordered CPU-native fixed-K256 logit-bias sweep")
    sweep = _run_bias_sweep(
        context=context,
        records=context["train_records"],
        protocol=frozen,
        baselines=state["baselines"],
    )
    post = _screen_post_authentication(
        context,
        checkpoint=frozen["training_checkpoint"],
    )
    if not post or not all(post.values()):
        raise ValueError("retrieval episodic logit-bias post-run authentication failed")
    report = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _RESULT_EXPERIMENT,
        "status": (
            "train_episodic_logit_bias_gate_passed"
            if sweep["passed"]
            else "train_episodic_logit_bias_gate_failed"
        ),
        "protocol": {
            "path": str(context["logit_bias_protocol_path"]),
            "sha256": context["logit_bias_protocol_sha256"],
        },
        "scope": {
            "split": "train",
            "threads": _THREADS,
            "device": "cpu",
            "dense_teacher_forwards": 0,
            "fixed_K": _FIXED_K,
            "mask_fitted_by_this_experiment": False,
            "candidate_selection_uses_train_outcomes": True,
            "development_outcomes_used": False,
            "confirmation_split_opened": False,
        },
        "reused_checkpoint_evidence": state["baselines"],
        "rank_family_attribution": frozen["rank_family_attribution"],
        "historical_K256_attribution": frozen["fixed_arm"][
            "historical_K256_attribution"
        ],
        "beta_zero_parity": frozen["beta_zero_parity"],
        "logit_bias_sweep": sweep,
        "decision": {
            "train_progression_gate_passed": sweep["passed"],
            "semantic_gate_passed": False,
            "development_authorized": sweep["passed"],
            "confirmation_authorized": False,
            "next_step": (
                "freeze a distinct dense-teacher semantic development screen"
                if sweep["passed"]
                else "reject shared fixed-K256 episodic logit-bias calibration"
            ),
        },
        "post_run_authentication": post,
        "confirmation_split_opened": False,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output, report)
    _progress(f"logit-bias result written to {output}")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train-only fixed-K256 OLMoE episodic logit-bias sweep",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    parity = commands.add_parser("parity")
    parity.add_argument("--rank-protocol", required=True)
    parity.add_argument("--rank-protocol-sha256", required=True)
    parity.add_argument("--rank-result", required=True)
    parity.add_argument("--rank-result-sha256", required=True)
    parity.add_argument("--logit-bias-library", required=True)
    parity.add_argument("--logit-bias-library-sha256", required=True)
    parity.add_argument("--out", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--rank-protocol", required=True)
    freeze.add_argument("--rank-protocol-sha256", required=True)
    freeze.add_argument("--rank-result", required=True)
    freeze.add_argument("--rank-result-sha256", required=True)
    freeze.add_argument("--logit-bias-library", required=True)
    freeze.add_argument("--logit-bias-library-sha256", required=True)
    freeze.add_argument("--parity-report", required=True)
    freeze.add_argument("--parity-report-sha256", required=True)
    freeze.add_argument("--out", required=True)
    screen = commands.add_parser("screen")
    screen.add_argument("--protocol", required=True)
    screen.add_argument("--protocol-sha256", required=True)
    screen.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "parity":
        value = generate_beta_zero_parity_report(
            rank_protocol=args.rank_protocol,
            rank_protocol_sha256=args.rank_protocol_sha256,
            rank_result=args.rank_result,
            rank_result_sha256=args.rank_result_sha256,
            logit_bias_library=args.logit_bias_library,
            logit_bias_library_sha256=args.logit_bias_library_sha256,
            out=args.out,
        )
    elif args.command == "freeze":
        value = freeze_episodic_logit_bias_protocol(
            rank_protocol=args.rank_protocol,
            rank_protocol_sha256=args.rank_protocol_sha256,
            rank_result=args.rank_result,
            rank_result_sha256=args.rank_result_sha256,
            logit_bias_library=args.logit_bias_library,
            logit_bias_library_sha256=args.logit_bias_library_sha256,
            parity_report=args.parity_report,
            parity_report_sha256=args.parity_report_sha256,
            out=args.out,
        )
    elif args.command == "screen":
        value = screen_episodic_logit_bias(
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            out=args.out,
        )
    else:  # pragma: no cover - argparse owns this boundary
        raise AssertionError("unknown retrieval episodic logit-bias command")
    print(json.dumps(value, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())
