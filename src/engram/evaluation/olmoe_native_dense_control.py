"""Post-failure matched Q7 plus full-local-attention attribution control."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from engram.compiler.olmoe_native import (
    OLMOE_CONFIG_PATH,
    OLMOE_NON_MLP_PATH,
    OLMOE_Q7_PATH,
    validate_olmoe_native_package,
)
from engram.evaluation.olmoe_native_causal import (
    _aggregate,
    _authenticate_evaluator_sources,
    _position_metrics,
)
from engram.evaluation.olmoe_native_generation import (
    _read_object,
    _validate_teacher_source,
)
from engram.evaluation.olmoe_native_sustained import (
    _POSITIONS_PER_SEQUENCE,
    _QUALITY_BANDS,
    _SEQUENCES,
    _TEACHER_CONFIGURATION,
    _THRESHOLDS,
    _TOKENS_PER_SEQUENCE,
    _attention_expectations,
    _deterministic_metrics,
    _post_source_shards,
    _q7_expectations,
    _quality_checks,
    _teacher_configuration,
    _update_diagnostic_hashes,
    _validate_corpus_manifest,
)
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.utils import atomic_json, sha256_file, sha256_json


_CONTROL_EXPERIMENT = "olmoe_native_q7_dense_attention_attribution_control"
_CONTROL_STATUS = "frozen_after_sustained_failure_before_control_execution"
_CONTROL_THREADS = 12


def _control_policy() -> dict[str, int]:
    return {
        "local_window": _POSITIONS_PER_SEQUENCE,
        "older_candidates": 8,
        "older_top_k": 4,
        "sink_tokens": 2,
    }


def _expected_bands() -> list[dict[str, int | str]]:
    return [
        {"name": name, "start": start, "stop": stop}
        for name, start, stop in _QUALITY_BANDS
    ]


def _validate_failed_sustained_result(
    protocol: dict[str, Any],
    failed_result: dict[str, Any],
    *,
    protocol_hash: str,
) -> None:
    artifacts = failed_result.get("artifacts")
    post_authentication = failed_result.get("post_run_authentication")
    position_results = failed_result.get("position_results")
    if (
        protocol.get("schema_version") != 1
        or protocol.get("experiment") != "olmoe_native_sustained_context_confirmation"
        or protocol.get("status") != "frozen_before_candidate_execution"
        or failed_result.get("schema_version") != 1
        or failed_result.get("experiment")
        != "olmoe_native_sustained_context_confirmation"
        or failed_result.get("status") != "frozen_confirmation_failed"
        or failed_result.get("gate_passed") is not False
        or failed_result.get("evidence_passed") is not True
        or failed_result.get("quality_passed") is not False
        or failed_result.get("decision") != "run_matched_q7_dense_attention_control"
        or not isinstance(artifacts, dict)
        or artifacts.get("protocol_sha256") != protocol_hash
        or artifacts.get("package_manifest_sha256")
        != protocol.get("package_manifest_sha256")
        or artifacts.get("native_library_sha256")
        != protocol.get("native_library_sha256")
        or artifacts.get("dataset_sha256") != protocol.get("dataset_sha256")
        or artifacts.get("corpus_manifest_sha256")
        != protocol.get("corpus_manifest_sha256")
        or artifacts.get("teacher_reference_sha256")
        != protocol.get("teacher_reference_sha256")
        or artifacts.get("teacher_arrays_sha256")
        != protocol.get("teacher_arrays_sha256")
        or not isinstance(post_authentication, dict)
        or not post_authentication
        or not all(value is True for value in post_authentication.values())
        or not isinstance(failed_result.get("metrics"), dict)
        or not isinstance(failed_result.get("position_bands"), dict)
        or not isinstance(position_results, list)
        or len(position_results) != _SEQUENCES * _POSITIONS_PER_SEQUENCE
    ):
        raise ValueError("dense-attention control prerequisite is invalid")


def _compare_metrics(
    bounded: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, dict[str, float]]:
    names = (
        "teacher_to_native_kl",
        "teacher_top1_agreement",
        "target_nll_delta",
        "final_hidden_relative_l2",
    )
    return {
        name: {
            "bounded": float(bounded[name]),
            "control": float(control[name]),
            "control_minus_bounded": float(control[name]) - float(bounded[name]),
        }
        for name in names
    }


def _pre_intervention_identity(
    bounded_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    bounded = [row for row in bounded_rows if int(row.get("position", -1)) < 16]
    control = [row for row in control_rows if int(row.get("position", -1)) < 16]
    expected = _SEQUENCES * 16
    return {
        "expected_positions": expected,
        "bounded_positions": len(bounded),
        "control_positions": len(control),
        "exact_position_metrics_match": (
            len(bounded) == expected and len(control) == expected and bounded == control
        ),
    }


def _structural_checks(
    metrics: dict[str, int],
    expectations: dict[str, int | float],
    q7_expectations: dict[str, int],
    *,
    position: int,
) -> dict[str, bool]:
    exact = (
        "positions_processed",
        "attention_state_bytes",
        "attention_scratch_bytes",
        "attention_eviction_events",
        "attention_older_candidate_entries_scored",
        "attention_older_selected_entries",
        "attention_sink_insertions",
        "attention_logical_read_bytes",
    )
    checks = {
        name: int(metrics.get(name, -1)) == int(expectations[name]) for name in exact
    }
    checks.update(
        {
            "cache_position": position == _POSITIONS_PER_SEQUENCE,
            "attention_heavy_hitter_updates": (
                metrics.get("attention_heavy_hitter_updates")
                == expectations["attention_heavy_hitter_updates_minimum"]
                == expectations["attention_heavy_hitter_updates_maximum"]
            ),
            "q7_scheduled_bytes": (
                metrics.get("q7_scheduled_bytes")
                == q7_expectations["scheduled_bytes_per_sequence"]
            ),
        }
    )
    return checks


def _paths_from_manifest(
    package: Path,
    manifest: dict[str, Any],
) -> tuple[Path, Path, Path, Path]:
    tokenizer = package / manifest["tokenizer"]["path"] / "tokenizer.json"
    return (
        package / OLMOE_CONFIG_PATH,
        package / OLMOE_NON_MLP_PATH,
        package / OLMOE_Q7_PATH,
        tokenizer,
    )


def freeze_native_olmoe_dense_attention_control_protocol(
    *,
    package: str | Path,
    manifest_sha256: str,
    library: str | Path,
    dataset: str | Path,
    corpus_manifest: str | Path,
    teacher_reference: str | Path,
    teacher_arrays: str | Path,
    sustained_protocol: str | Path,
    sustained_protocol_sha256: str,
    sustained_result: str | Path,
    sustained_result_sha256: str,
    out: str | Path,
    threads: int = _CONTROL_THREADS,
) -> dict[str, Any]:
    """Freeze the post-failure attribution control before observing its output."""

    output_path = Path(out).expanduser().resolve()
    if output_path.exists():
        raise ValueError("dense-attention control protocol target already exists")
    if threads != _CONTROL_THREADS:
        raise ValueError("dense-attention control requires the matched 12 threads")
    package_path = Path(package).expanduser().resolve()
    library_path = Path(library).expanduser().resolve()
    dataset_path = Path(dataset).expanduser().resolve()
    corpus_manifest_path = Path(corpus_manifest).expanduser().resolve()
    reference_path = Path(teacher_reference).expanduser().resolve()
    arrays_path = Path(teacher_arrays).expanduser().resolve()
    sustained_protocol_path = Path(sustained_protocol).expanduser().resolve()
    sustained_result_path = Path(sustained_result).expanduser().resolve()
    protocol = _read_object(sustained_protocol_path, "sustained protocol")
    failed_result = _read_object(sustained_result_path, "sustained result")
    reference = _read_object(reference_path, "sustained teacher reference")
    protocol_hash = sha256_file(sustained_protocol_path)
    failed_result_hash = sha256_file(sustained_result_path)
    if (
        protocol_hash != sustained_protocol_sha256.lower()
        or failed_result_hash != sustained_result_sha256.lower()
    ):
        raise ValueError("dense-attention control prerequisite hash is invalid")
    _validate_failed_sustained_result(
        protocol,
        failed_result,
        protocol_hash=protocol_hash,
    )
    evaluator_sources = _authenticate_evaluator_sources(protocol)
    _validate_teacher_source(reference, protocol)
    if (
        protocol.get("attention_policy")
        != {
            "local_window": 16,
            "older_candidates": 8,
            "older_top_k": 4,
            "sink_tokens": 2,
        }
        or protocol.get("thresholds") != _THRESHOLDS
        or protocol.get("quality_bands") != _expected_bands()
        or protocol.get("sequences") != _SEQUENCES
        or protocol.get("tokens_per_sequence") != _TOKENS_PER_SEQUENCE
        or protocol.get("scope", {}).get("candidate_threads") != _CONTROL_THREADS
        or protocol.get("scope", {}).get("teacher_configuration")
        != _TEACHER_CONFIGURATION
        or _teacher_configuration(reference) != _TEACHER_CONFIGURATION
        or reference.get("dataset", {}).get("input_ids") != protocol.get("input_ids")
        or reference.get("dataset", {}).get("input_identity")
        != protocol.get("input_identity")
    ):
        raise ValueError("dense-attention control frozen contract is invalid")
    manifest = validate_olmoe_native_package(
        package_path,
        expected_manifest_sha256=manifest_sha256,
    )
    identities = {
        "package_manifest_sha256": manifest_sha256.lower(),
        "native_library_sha256": sha256_file(library_path),
        "dataset_sha256": sha256_file(dataset_path),
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "teacher_reference_sha256": sha256_file(reference_path),
        "teacher_arrays_sha256": sha256_file(arrays_path),
    }
    if any(
        protocol.get(name) != value for name, value in identities.items()
    ) or not _post_source_shards(reference, protocol["source_shard_sha256"]):
        raise ValueError("dense-attention control artifact identity is invalid")
    input_ids = protocol["input_ids"]
    if sha256_json(input_ids) != protocol["input_identity"]:
        raise ValueError("dense-attention control input identity is invalid")
    _config_path, _non_mlp_path, _q7_path, tokenizer_path = _paths_from_manifest(
        package_path,
        manifest,
    )
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] to freeze the dense-attention control"
        ) from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    retokenized, _corpus = _validate_corpus_manifest(
        dataset_path,
        corpus_manifest_path,
        tokenizer,
        tokenizer_sha256=manifest["files"]["tokenizer/tokenizer.json"]["sha256"],
    )
    if retokenized != input_ids:
        raise ValueError("dense-attention control retokenization differs")
    model = protocol["model"]
    q7_expectations = _q7_expectations(model)
    attention_expectations = _attention_expectations(model, _control_policy())
    if (
        q7_expectations != protocol["q7_expectations_per_sequence"]
        or attention_expectations["attention_eviction_events"] != 0
        or attention_expectations["attention_older_candidate_entries_scored"] != 0
        or attention_expectations["attention_older_selected_entries"] != 0
        or attention_expectations["attention_sink_insertions"] != 0
        or attention_expectations["attention_heavy_hitter_updates_minimum"] != 0
        or attention_expectations["attention_heavy_hitter_updates_maximum"] != 0
        or attention_expectations["attention_logical_read_fraction"] != 1.0
    ):
        raise ValueError("dense-attention control expectations are invalid")
    control_source = Path(__file__).resolve()
    control_protocol = {
        "schema_version": 1,
        "experiment": _CONTROL_EXPERIMENT,
        "status": _CONTROL_STATUS,
        "source_revision": protocol["source_revision"],
        **identities,
        "sustained_protocol_sha256": protocol_hash,
        "sustained_result_sha256": failed_result_hash,
        "control_source_sha256": sha256_file(control_source),
        "frozen_evaluator_source_sha256": evaluator_sources,
        "source_config_sha256": protocol["source_config_sha256"],
        "source_index_sha256": protocol["source_index_sha256"],
        "source_shard_sha256": protocol["source_shard_sha256"],
        "input_identity": protocol["input_identity"],
        "input_ids": input_ids,
        "model": model,
        "bounded_attention_policy": protocol["attention_policy"],
        "control_attention_policy": _control_policy(),
        "attention_expectations_per_sequence": attention_expectations,
        "q7_expectations_per_sequence": q7_expectations,
        "quality_bands": _expected_bands(),
        "thresholds": _THRESHOLDS,
        "scope": {
            "candidate_device": "cpu",
            "candidate_threads": threads,
            "candidate_transformers_model_shell": False,
            "only_intervention": "local_window_16_to_128",
            "full_causal_attention_for_positions": _POSITIONS_PER_SEQUENCE,
            "q7_artifact_or_policy_changed": False,
            "deployable_attention_traffic_gate_applies": False,
            "reset_replay_sequence": 0,
            "control_frozen_before_execution": True,
            "sustained_failure_known_before_control_freeze": True,
            "teacher_configuration": _TEACHER_CONFIGURATION,
        },
        "decision_rule": {
            "evidence_failure": (
                "stop without semantic attribution if any artifact, source, "
                "counter, replay, or diagnostic check fails"
            ),
            "quality_pass": (
                "attribute the sustained semantic failure primarily to the "
                "bounded W16/C8/K4/S2 attention policy"
            ),
            "quality_failure": (
                "conclude that bounded attention is not sufficient as the sole "
                "attribution; quantify the residual before the next control"
            ),
        },
        "limitations": [
            "This protocol is frozen after observing the sustained failure.",
            "It is a matched attribution diagnostic, not a new gate.",
            "W=128 is full causal attention only for the frozen 128 positions.",
            "Its 100-percent logical attention-read fraction is intentionally non-deployable.",
        ],
    }
    atomic_json(output_path, control_protocol)
    return control_protocol


def evaluate_native_olmoe_dense_attention_control(
    *,
    package: str | Path,
    manifest_sha256: str,
    library: str | Path,
    dataset: str | Path,
    corpus_manifest: str | Path,
    teacher_reference: str | Path,
    teacher_arrays: str | Path,
    sustained_protocol: str | Path,
    sustained_protocol_sha256: str,
    sustained_result: str | Path,
    sustained_result_sha256: str,
    control_protocol: str | Path,
    control_protocol_sha256: str,
    out: str | Path,
    threads: int = _CONTROL_THREADS,
) -> dict[str, Any]:
    """Attribute the authenticated sustained failure with exact W=128 attention."""

    output_path = Path(out).expanduser().resolve()
    if output_path.exists():
        raise ValueError("dense-attention control result target already exists")
    if threads != _CONTROL_THREADS:
        raise ValueError("dense-attention control requires the matched 12 threads")
    package_path = Path(package).expanduser().resolve()
    library_path = Path(library).expanduser().resolve()
    dataset_path = Path(dataset).expanduser().resolve()
    corpus_manifest_path = Path(corpus_manifest).expanduser().resolve()
    reference_path = Path(teacher_reference).expanduser().resolve()
    arrays_path = Path(teacher_arrays).expanduser().resolve()
    protocol_path = Path(sustained_protocol).expanduser().resolve()
    sustained_result_path = Path(sustained_result).expanduser().resolve()
    control_protocol_path = Path(control_protocol).expanduser().resolve()
    protocol = _read_object(protocol_path, "sustained protocol")
    failed_result = _read_object(sustained_result_path, "sustained result")
    control_protocol_value = _read_object(
        control_protocol_path,
        "dense-attention control protocol",
    )
    reference = _read_object(reference_path, "sustained teacher reference")
    protocol_hash = sha256_file(protocol_path)
    failed_result_hash = sha256_file(sustained_result_path)
    library_hash = sha256_file(library_path)
    dataset_hash = sha256_file(dataset_path)
    corpus_manifest_hash = sha256_file(corpus_manifest_path)
    reference_hash = sha256_file(reference_path)
    arrays_hash = sha256_file(arrays_path)
    control_protocol_hash = sha256_file(control_protocol_path)
    control_source = Path(__file__).resolve()
    control_source_hash = sha256_file(control_source)
    if (
        protocol_hash != sustained_protocol_sha256.lower()
        or failed_result_hash != sustained_result_sha256.lower()
        or control_protocol_hash != control_protocol_sha256.lower()
        or protocol.get("experiment") != "olmoe_native_sustained_context_confirmation"
        or protocol.get("status") != "frozen_before_candidate_execution"
        or protocol.get("package_manifest_sha256") != manifest_sha256.lower()
        or protocol.get("native_library_sha256") != library_hash
        or protocol.get("dataset_sha256") != dataset_hash
        or protocol.get("corpus_manifest_sha256") != corpus_manifest_hash
        or protocol.get("teacher_reference_sha256") != reference_hash
        or protocol.get("teacher_arrays_sha256") != arrays_hash
    ):
        raise ValueError("dense-attention control prerequisite is invalid")
    _validate_failed_sustained_result(
        protocol,
        failed_result,
        protocol_hash=protocol_hash,
    )
    control_scope = control_protocol_value.get("scope")
    if (
        control_protocol_value.get("schema_version") != 1
        or control_protocol_value.get("experiment") != _CONTROL_EXPERIMENT
        or control_protocol_value.get("status") != _CONTROL_STATUS
        or control_protocol_value.get("source_revision")
        != protocol.get("source_revision")
        or control_protocol_value.get("sustained_protocol_sha256") != protocol_hash
        or control_protocol_value.get("sustained_result_sha256") != failed_result_hash
        or control_protocol_value.get("package_manifest_sha256")
        != manifest_sha256.lower()
        or control_protocol_value.get("native_library_sha256") != library_hash
        or control_protocol_value.get("dataset_sha256") != dataset_hash
        or control_protocol_value.get("corpus_manifest_sha256") != corpus_manifest_hash
        or control_protocol_value.get("teacher_reference_sha256") != reference_hash
        or control_protocol_value.get("teacher_arrays_sha256") != arrays_hash
        or control_protocol_value.get("control_source_sha256") != control_source_hash
        or control_protocol_value.get("input_ids") != protocol.get("input_ids")
        or control_protocol_value.get("input_identity")
        != protocol.get("input_identity")
        or control_protocol_value.get("model") != protocol.get("model")
        or control_protocol_value.get("bounded_attention_policy")
        != protocol.get("attention_policy")
        or control_protocol_value.get("control_attention_policy") != _control_policy()
        or control_protocol_value.get("quality_bands") != _expected_bands()
        or control_protocol_value.get("thresholds") != _THRESHOLDS
        or not isinstance(control_scope, dict)
        or control_scope.get("candidate_threads") != _CONTROL_THREADS
        or control_scope.get("candidate_device") != "cpu"
        or control_scope.get("candidate_transformers_model_shell") is not False
        or control_scope.get("only_intervention") != "local_window_16_to_128"
        or control_scope.get("full_causal_attention_for_positions")
        != _POSITIONS_PER_SEQUENCE
        or control_scope.get("q7_artifact_or_policy_changed") is not False
        or control_scope.get("deployable_attention_traffic_gate_applies") is not False
        or control_scope.get("control_frozen_before_execution") is not True
        or control_scope.get("reset_replay_sequence") != 0
        or control_scope.get("sustained_failure_known_before_control_freeze")
        is not True
    ):
        raise ValueError("dense-attention control protocol contract is invalid")
    evaluator_sources = _authenticate_evaluator_sources(protocol)
    _validate_teacher_source(reference, protocol)
    if (
        _teacher_configuration(reference) != _TEACHER_CONFIGURATION
        or protocol.get("scope", {}).get("teacher_configuration")
        != _TEACHER_CONFIGURATION
        or protocol.get("attention_policy")
        != {
            "local_window": 16,
            "older_candidates": 8,
            "older_top_k": 4,
            "sink_tokens": 2,
        }
        or protocol.get("quality_bands") != _expected_bands()
        or protocol.get("thresholds") != _THRESHOLDS
        or control_protocol_value.get("frozen_evaluator_source_sha256")
        != evaluator_sources
        or control_protocol_value.get("source_config_sha256")
        != protocol.get("source_config_sha256")
        or control_protocol_value.get("source_index_sha256")
        != protocol.get("source_index_sha256")
        or control_protocol_value.get("source_shard_sha256")
        != protocol.get("source_shard_sha256")
        or control_scope.get("teacher_configuration") != _TEACHER_CONFIGURATION
        or reference.get("dataset", {}).get("input_ids") != protocol.get("input_ids")
        or reference.get("dataset", {}).get("input_identity")
        != protocol.get("input_identity")
        or reference.get("dataset", {}).get("prediction_positions")
        != _SEQUENCES * _POSITIONS_PER_SEQUENCE
    ):
        raise ValueError("dense-attention control teacher contract is invalid")
    input_ids = protocol["input_ids"]
    if sha256_json(input_ids) != protocol["input_identity"]:
        raise ValueError("dense-attention control input identity is invalid")
    manifest = validate_olmoe_native_package(
        package_path,
        expected_manifest_sha256=manifest_sha256,
    )
    config_path, non_mlp_path, q7_path, tokenizer_path = _paths_from_manifest(
        package_path,
        manifest,
    )
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for the dense-attention control"
        ) from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    retokenized, _corpus = _validate_corpus_manifest(
        dataset_path,
        corpus_manifest_path,
        tokenizer,
        tokenizer_sha256=manifest["files"]["tokenizer/tokenizer.json"]["sha256"],
    )
    if retokenized != input_ids:
        raise ValueError("dense-attention control retokenization differs")
    model = protocol["model"]
    q7_expectations = _q7_expectations(model)
    control_policy = _control_policy()
    attention_expectations = _attention_expectations(
        model,
        control_policy,
    )
    if (
        attention_expectations["attention_eviction_events"] != 0
        or attention_expectations["attention_logical_read_fraction"] != 1.0
        or q7_expectations != protocol["q7_expectations_per_sequence"]
        or control_protocol_value.get("attention_expectations_per_sequence")
        != attention_expectations
        or control_protocol_value.get("q7_expectations_per_sequence") != q7_expectations
    ):
        raise ValueError("dense-attention control expectations are invalid")
    prediction_positions = _SEQUENCES * _POSITIONS_PER_SEQUENCE
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"logits", "hidden", "targets"}:
            raise ValueError("dense-attention control arrays have unexpected keys")
        teacher_logits = np.asarray(arrays["logits"], dtype=np.float32)
        teacher_hidden = np.asarray(arrays["hidden"], dtype=np.float32)
        targets = np.asarray(arrays["targets"], dtype=np.int64)
    expected_targets = np.asarray(
        [token for sequence in input_ids for token in sequence[1:]],
        dtype=np.int64,
    )
    if (
        teacher_logits.shape != (prediction_positions, int(model["vocab_size"]))
        or teacher_hidden.shape != (prediction_positions, int(model["hidden_size"]))
        or targets.shape != (prediction_positions,)
        or not np.array_equal(targets, expected_targets)
    ):
        raise ValueError("dense-attention control array shapes are invalid")

    all_rows: list[dict[str, float | bool | int]] = []
    band_rows: dict[str, list[dict[str, float | bool | int]]] = {
        name: [] for name, _start, _stop in _QUALITY_BANDS
    }
    sequence_results: list[dict[str, Any]] = []
    total_q7_bytes = 0
    replay_reference: dict[str, Any] | None = None
    load_started = time.perf_counter()
    runtime = OLMoENativeTokenRuntime(
        config_path,
        non_mlp_path,
        q7_path,
        library_path,
        threads=threads,
        local_window=control_policy["local_window"],
        older_candidates=control_policy["older_candidates"],
        older_top_k=control_policy["older_top_k"],
        sink_tokens=control_policy["sink_tokens"],
    )
    cold_load_seconds = time.perf_counter() - load_started
    try:
        if not runtime.attention_metrics_available:
            raise ValueError("dense-attention metric ABI is unavailable")
        offset = 0
        for sequence_index, sequence in enumerate(input_ids):
            runtime.reset()
            rows: list[dict[str, float | bool | int]] = []
            top1_tokens: list[int] = []
            hidden_digest = hashlib.sha256()
            logit_digest = hashlib.sha256()
            started = time.perf_counter()
            for position, token_id in enumerate(sequence[:-1]):
                native_result = runtime.forward([token_id])
                native_hidden, native_logits = runtime.last_diagnostics()
                if int(np.argmax(native_logits)) != native_result.next_token:
                    raise ValueError("dense-attention diagnostic argmax differs")
                row = _position_metrics(
                    teacher_logits[offset],
                    native_logits,
                    teacher_hidden[offset],
                    native_hidden,
                    int(targets[offset]),
                )
                row.update(
                    {
                        "sequence": sequence_index,
                        "position": position,
                        "target": int(targets[offset]),
                    }
                )
                all_rows.append(row)
                rows.append(row)
                for name, start, stop in _QUALITY_BANDS:
                    if start <= position < stop:
                        band_rows[name].append(row)
                        break
                top1_tokens.append(native_result.next_token)
                _update_diagnostic_hashes(
                    hidden_digest,
                    logit_digest,
                    native_hidden,
                    native_logits,
                )
                offset += 1
            elapsed = time.perf_counter() - started
            metrics = dict(runtime.last_result.metrics)
            structural = _structural_checks(
                metrics,
                attention_expectations,
                q7_expectations,
                position=runtime.position,
            )
            total_q7_bytes += metrics["q7_scheduled_bytes"]
            diagnostic_hashes = {
                "hidden_sha256": hidden_digest.hexdigest(),
                "logits_sha256": logit_digest.hexdigest(),
            }
            if sequence_index == 0:
                replay_reference = {
                    "top1_tokens": top1_tokens,
                    "diagnostic_hashes": diagnostic_hashes,
                    "deterministic_metrics": _deterministic_metrics(metrics),
                }
            sequence_results.append(
                {
                    "sequence": sequence_index,
                    "elapsed_seconds": elapsed,
                    "metrics": _aggregate(rows),
                    "native_metrics": metrics,
                    "diagnostic_hashes": diagnostic_hashes,
                    "structural_checks": structural,
                    "structural_passed": all(structural.values()),
                }
            )
        if replay_reference is None:
            raise RuntimeError("dense-attention replay reference is missing")
        runtime.reset()
        replay_tokens: list[int] = []
        hidden_digest = hashlib.sha256()
        logit_digest = hashlib.sha256()
        replay_started = time.perf_counter()
        for token_id in input_ids[0][:-1]:
            replay_result = runtime.forward([token_id])
            replay_hidden, replay_logits = runtime.last_diagnostics()
            replay_tokens.append(replay_result.next_token)
            _update_diagnostic_hashes(
                hidden_digest,
                logit_digest,
                replay_hidden,
                replay_logits,
            )
        replay_seconds = time.perf_counter() - replay_started
        replay_metrics = dict(runtime.last_result.metrics)
        replay_hashes = {
            "hidden_sha256": hidden_digest.hexdigest(),
            "logits_sha256": logit_digest.hexdigest(),
        }
        replay_structural = _structural_checks(
            replay_metrics,
            attention_expectations,
            q7_expectations,
            position=runtime.position,
        )
        reset_replay = {
            "sequence": 0,
            "elapsed_seconds": replay_seconds,
            "top1_tokens_match": replay_tokens == replay_reference["top1_tokens"],
            "diagnostic_hashes_match": replay_hashes
            == replay_reference["diagnostic_hashes"],
            "deterministic_metrics_match": (
                _deterministic_metrics(replay_metrics)
                == replay_reference["deterministic_metrics"]
            ),
            "structural_checks": replay_structural,
        }
        reset_replay["passed"] = (
            reset_replay["top1_tokens_match"]
            and reset_replay["diagnostic_hashes_match"]
            and reset_replay["deterministic_metrics_match"]
            and all(replay_structural.values())
        )
    finally:
        runtime.close()

    source_model = Path(reference["source"]["model"]).expanduser().resolve()
    source_shards = protocol["source_shard_sha256"]
    repository = Path(__file__).resolve().parents[3]
    post_authentication = {
        "package": validate_olmoe_native_package(
            package_path,
            expected_manifest_sha256=manifest_sha256,
        )
        == manifest,
        "library": sha256_file(library_path) == library_hash,
        "dataset": sha256_file(dataset_path) == dataset_hash,
        "corpus_manifest": (sha256_file(corpus_manifest_path) == corpus_manifest_hash),
        "teacher_reference": sha256_file(reference_path) == reference_hash,
        "teacher_arrays": sha256_file(arrays_path) == arrays_hash,
        "sustained_protocol": sha256_file(protocol_path) == protocol_hash,
        "sustained_result": (sha256_file(sustained_result_path) == failed_result_hash),
        "control_protocol": (
            sha256_file(control_protocol_path) == control_protocol_hash
        ),
        "teacher_source_config": (
            sha256_file(source_model / "config.json")
            == protocol["source_config_sha256"]
        ),
        "teacher_source_index": (
            sha256_file(source_model / "model.safetensors.index.json")
            == protocol["source_index_sha256"]
        ),
        "teacher_source_shards": _post_source_shards(reference, source_shards),
        "frozen_evaluator_sources": all(
            sha256_file(repository / relative) == expected
            for relative, expected in evaluator_sources.items()
        ),
        "control_source": (
            sha256_file(control_source)
            == control_source_hash
            == control_protocol_value["control_source_sha256"]
        ),
    }
    aggregate = _aggregate(all_rows)
    bands = {name: _aggregate(rows) for name, rows in band_rows.items()}
    pre_intervention_identity = _pre_intervention_identity(
        failed_result["position_results"],
        all_rows,
    )
    per_position_offset = {
        str(position): _aggregate(
            [row for row in all_rows if row["position"] == position]
        )
        for position in range(_POSITIONS_PER_SEQUENCE)
    }
    bounded_per_position_offset = {
        str(position): _aggregate(
            [
                row
                for row in failed_result["position_results"]
                if row["position"] == position
            ]
        )
        for position in range(_POSITIONS_PER_SEQUENCE)
    }
    comparison = {
        "overall": _compare_metrics(failed_result["metrics"], aggregate),
        "position_bands": {
            name: _compare_metrics(failed_result["position_bands"][name], metrics)
            for name, metrics in bands.items()
        },
        "position_offsets": {
            str(position): _compare_metrics(
                bounded_per_position_offset[str(position)],
                per_position_offset[str(position)],
            )
            for position in range(_POSITIONS_PER_SEQUENCE)
        },
    }
    quality_checks = _quality_checks("overall", aggregate)
    for name, metrics in bands.items():
        quality_checks.update(_quality_checks(name, metrics))
    evidence_checks = {
        "prediction_positions": len(all_rows) == prediction_positions,
        "sequence_structural_checks": all(
            result["structural_passed"] for result in sequence_results
        ),
        "q7_scheduled_bytes": (
            total_q7_bytes
            == _SEQUENCES * q7_expectations["scheduled_bytes_per_sequence"]
        ),
        "full_attention_logical_reads": (
            attention_expectations["attention_logical_read_fraction"] == 1.0
        ),
        "pre_intervention_identity": pre_intervention_identity[
            "exact_position_metrics_match"
        ],
        "reset_replay": reset_replay["passed"],
        "post_run_authentication": all(post_authentication.values()),
    }
    evidence_passed = all(evidence_checks.values())
    quality_passed = all(quality_checks.values())
    if not evidence_passed:
        diagnosis = "control_invalid_stop_and_diagnose_evidence"
    elif quality_passed:
        diagnosis = "bounded_attention_is_dominant_sustained_drift_source"
    else:
        diagnosis = "bounded_attention_not_sufficient_as_sole_attribution"
    report = {
        "schema_version": 1,
        "experiment": _CONTROL_EXPERIMENT,
        "status": "post_failure_diagnostic_complete",
        "provenance": {
            "control_protocol_frozen_before_execution": True,
            "sustained_failure_was_known_before_control_freeze": True,
            "scientific_role": "matched attribution diagnostic, not a new gate",
            "sustained_protocol_sha256": protocol_hash,
            "sustained_result_sha256": failed_result_hash,
            "control_protocol_sha256": control_protocol_hash,
            "control_source_sha256": control_source_hash,
        },
        "artifacts": {
            "package_manifest_sha256": manifest_sha256.lower(),
            "native_library_sha256": library_hash,
            "dataset_sha256": dataset_hash,
            "corpus_manifest_sha256": corpus_manifest_hash,
            "teacher_reference_sha256": reference_hash,
            "teacher_arrays_sha256": arrays_hash,
            "sustained_protocol_sha256": protocol_hash,
            "sustained_result_sha256": failed_result_hash,
            "control_protocol_sha256": control_protocol_hash,
            "control_source_sha256": control_source_hash,
        },
        "configuration": {
            "candidate_device": "cpu",
            "candidate_threads": threads,
            "transformers_model_shell_used": False,
            "q7_artifact_or_policy_changed": False,
            "only_intervention": "local_window_16_to_128",
            "attention_policy": control_policy,
            "positions_per_sequence": _POSITIONS_PER_SEQUENCE,
        },
        "metrics": aggregate,
        "position_bands": bands,
        "bounded_control_comparison": comparison,
        "pre_intervention_identity": pre_intervention_identity,
        "per_position_offset": per_position_offset,
        "first_top1_divergence": next(
            (
                {
                    "sequence": int(row["sequence"]),
                    "position": int(row["position"]),
                    "teacher_top1": int(row["teacher_top1"]),
                    "native_top1": int(row["native_top1"]),
                    "kl": float(row["kl"]),
                }
                for row in all_rows
                if not row["top1_match"]
            ),
            None,
        ),
        "attention_expectations_per_sequence": attention_expectations,
        "q7_expectations_per_sequence": q7_expectations,
        "traffic": {
            "q7_scheduled_bytes": total_q7_bytes,
            "attention_logical_read_bytes_per_sequence": attention_expectations[
                "attention_logical_read_bytes"
            ],
            "dense_full_context_logical_kv_bytes_per_sequence": (
                attention_expectations["dense_full_context_logical_kv_bytes"]
            ),
            "attention_logical_read_fraction": attention_expectations[
                "attention_logical_read_fraction"
            ],
            "measured_hardware_traffic": False,
        },
        "sequence_results": sequence_results,
        "position_results": all_rows,
        "reset_replay": reset_replay,
        "performance": {
            "cold_load_seconds": cold_load_seconds,
            "sequence_seconds": [
                result["elapsed_seconds"] for result in sequence_results
            ],
            "total_sequence_seconds": sum(
                result["elapsed_seconds"] for result in sequence_results
            ),
            "reset_replay_seconds": replay_seconds,
        },
        "quality_checks": quality_checks,
        "evidence_checks": evidence_checks,
        "quality_passed": quality_passed,
        "evidence_passed": evidence_passed,
        "diagnosis": diagnosis,
        "post_run_authentication": post_authentication,
        "limitations": [
            "This control was specified after observing the sustained gate failure.",
            "W=128 is exact full causal attention only for this 128-position protocol.",
            "Passing attributes this corpus-level drift to bounded attention; it does not prove task-sensitive retrieval quality.",
        ],
    }
    atomic_json(output_path, report)
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze or run the native OLMoE dense-attention control"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--package", required=True, type=Path)
        command.add_argument("--manifest-sha256", required=True)
        command.add_argument("--library", required=True, type=Path)
        command.add_argument("--dataset", required=True, type=Path)
        command.add_argument("--corpus-manifest", required=True, type=Path)
        command.add_argument("--teacher-reference", required=True, type=Path)
        command.add_argument("--teacher-arrays", required=True, type=Path)
        command.add_argument("--sustained-protocol", required=True, type=Path)
        command.add_argument("--sustained-protocol-sha256", required=True)
        command.add_argument("--sustained-result", required=True, type=Path)
        command.add_argument("--sustained-result-sha256", required=True)
        command.add_argument("--out", required=True, type=Path)
        command.add_argument("--threads", type=int, default=_CONTROL_THREADS)

    freeze_parser = commands.add_parser(
        "freeze",
        help="freeze the matched control protocol before execution",
    )
    add_common_arguments(freeze_parser)
    evaluate_parser = commands.add_parser(
        "evaluate",
        help="execute a previously frozen matched control",
    )
    add_common_arguments(evaluate_parser)
    evaluate_parser.add_argument("--control-protocol", required=True, type=Path)
    evaluate_parser.add_argument("--control-protocol-sha256", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        protocol = freeze_native_olmoe_dense_attention_control_protocol(
            package=args.package,
            manifest_sha256=args.manifest_sha256,
            library=args.library,
            dataset=args.dataset,
            corpus_manifest=args.corpus_manifest,
            teacher_reference=args.teacher_reference,
            teacher_arrays=args.teacher_arrays,
            sustained_protocol=args.sustained_protocol,
            sustained_protocol_sha256=args.sustained_protocol_sha256,
            sustained_result=args.sustained_result,
            sustained_result_sha256=args.sustained_result_sha256,
            out=args.out,
            threads=args.threads,
        )
        print(
            json.dumps(
                {
                    "status": protocol["status"],
                    "control_attention_policy": protocol["control_attention_policy"],
                    "protocol_sha256": sha256_file(args.out),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = evaluate_native_olmoe_dense_attention_control(
        package=args.package,
        manifest_sha256=args.manifest_sha256,
        library=args.library,
        dataset=args.dataset,
        corpus_manifest=args.corpus_manifest,
        teacher_reference=args.teacher_reference,
        teacher_arrays=args.teacher_arrays,
        sustained_protocol=args.sustained_protocol,
        sustained_protocol_sha256=args.sustained_protocol_sha256,
        sustained_result=args.sustained_result,
        sustained_result_sha256=args.sustained_result_sha256,
        control_protocol=args.control_protocol,
        control_protocol_sha256=args.control_protocol_sha256,
        out=args.out,
        threads=args.threads,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "evidence_passed": result["evidence_passed"],
                "quality_passed": result["quality_passed"],
                "diagnosis": result["diagnosis"],
                "metrics": result["metrics"],
                "position_bands": result["position_bands"],
                "result_sha256": sha256_file(args.out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["evidence_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(_main())
