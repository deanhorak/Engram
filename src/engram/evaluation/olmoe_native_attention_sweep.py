"""Frozen post-attribution development sweep for bounded native OLMoE attention."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import engram.evaluation.olmoe_native_dense_control as dense_control_source
from engram.compiler.olmoe_native import validate_olmoe_native_package
from engram.evaluation.olmoe_native_causal import (
    _aggregate,
    _authenticate_evaluator_sources,
    _position_metrics,
)
from engram.evaluation.olmoe_native_dense_control import (
    _compare_metrics,
    _control_policy,
    _expected_bands,
    _paths_from_manifest,
    _validate_failed_sustained_result,
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
    _structural_checks as _sustained_structural_checks,
    _teacher_configuration,
    _update_diagnostic_hashes,
    _validate_corpus_manifest,
)
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.utils import atomic_json, sha256_file, sha256_json


_SWEEP_EXPERIMENT = "olmoe_native_q7_bounded_attention_development_sweep"
_SWEEP_STATUS = "frozen_after_dense_attribution_before_sweep_execution"
_SWEEP_THREADS = 12
_EXPECTED_LOGICAL_READ_BYTES = 968_753_152
_EXPECTED_LOGICAL_READ_FRACTION = 0.44761385658914726
_MATURE_VISIBLE_VALUES = 32
_MATURE_VISIBLE_KEY_ROWS = 34
_CONTROL_EXPERIMENT = "olmoe_native_q7_dense_attention_attribution_control"
_CONTROL_PROTOCOL_STATUS = (
    "frozen_after_sustained_failure_before_control_execution"
)
_SOURCE_PACKAGE_ATTENTION_POLICY = {
    "local_window": 16,
    "older_candidates": 8,
    "older_top_k": 4,
    "sink_tokens": 2,
}
_EXECUTION_INTERFACE = "raw_native_token_runtime"


def _arms() -> list[dict[str, Any]]:
    return [
        {
            "name": "w16_c18_k16_s2",
            "attention_policy": {
                "local_window": 16,
                "older_candidates": 18,
                "older_top_k": 16,
                "sink_tokens": 2,
            },
        },
        {
            "name": "w24_c10_k8_s2",
            "attention_policy": {
                "local_window": 24,
                "older_candidates": 10,
                "older_top_k": 8,
                "sink_tokens": 2,
            },
        },
        {
            "name": "w30_c4_k2_s2",
            "attention_policy": {
                "local_window": 30,
                "older_candidates": 4,
                "older_top_k": 2,
                "sink_tokens": 2,
            },
        },
    ]


def _ranking_rule() -> dict[str, Any]:
    return {
        "eligibility": (
            "all overall and frozen-band quality checks pass; any evidence "
            "failure invalidates the complete sweep"
        ),
        "normalized_margins": {
            "teacher_to_native_kl": (
                "(maximum_mean_kl - value) / maximum_mean_kl"
            ),
            "teacher_top1_agreement": (
                "(value - minimum_top1_agreement) / "
                "(1 - minimum_top1_agreement)"
            ),
            "target_nll_delta": (
                "(maximum_mean_target_nll_delta - value) / "
                "maximum_mean_target_nll_delta"
            ),
            "final_hidden_relative_l2": (
                "(maximum_mean_final_hidden_relative_l2 - value) / "
                "maximum_mean_final_hidden_relative_l2"
            ),
        },
        "arm_score": (
            "minimum normalized margin over all four metrics in overall and "
            "the four post-common-prefix frozen position bands; positions "
            "0-15 remain a required quality gate but are excluded from ranking"
        ),
        "ordering": [
            "descending_worst_normalized_quality_margin",
            "ascending_attention_state_bytes",
            "frozen_arm_order",
        ],
        "selection_role": (
            "development_only_package_integration_then_fresh_confirmation"
        ),
        "none_pass_decision": "investigate_layer_adaptive_or_learned_selector",
    }


def _per_position_read_contract() -> dict[str, Any]:
    return {
        "key_rows": "min(position + 1, 34)",
        "value_rows": "min(position + 1, 32)",
        "first_full_causal_value_omission_offset": 32,
        "row_bytes": 131_072,
    }


def _arm_descriptors(model: dict[str, int]) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    logical_row_bytes = (
        int(model["layers"])
        * int(model["query_heads"])
        * int(model["head_dimension"])
        * 4
    )
    if logical_row_bytes != _per_position_read_contract()["row_bytes"]:
        raise ValueError("attention sweep logical row size is invalid")
    for ordinal, arm in enumerate(_arms()):
        policy = arm["attention_policy"]
        expectations = _attention_expectations(model, policy)
        mature_visible_values = int(policy["local_window"]) + int(
            policy["older_top_k"]
        )
        mature_visible_key_rows = int(policy["local_window"]) + int(
            policy["older_candidates"]
        )
        if (
            mature_visible_values != _MATURE_VISIBLE_VALUES
            or mature_visible_key_rows != _MATURE_VISIBLE_KEY_ROWS
            or expectations["attention_logical_read_bytes"]
            != _EXPECTED_LOGICAL_READ_BYTES
            or expectations["attention_logical_read_fraction"]
            != _EXPECTED_LOGICAL_READ_FRACTION
        ):
            raise ValueError("attention sweep analytical arm contract is invalid")
        previous_logical_read_bytes = 0
        for position in range(_POSITIONS_PER_SEQUENCE):
            position_expectations = _attention_expectations(
                model,
                policy,
                positions=position + 1,
            )
            logical_read_bytes = int(
                position_expectations["attention_logical_read_bytes"]
            )
            expected_delta = logical_row_bytes * (
                min(position + 1, _MATURE_VISIBLE_KEY_ROWS)
                + min(position + 1, _MATURE_VISIBLE_VALUES)
            )
            if logical_read_bytes - previous_logical_read_bytes != expected_delta:
                raise ValueError(
                    "attention sweep per-position read contract is invalid"
                )
            previous_logical_read_bytes = logical_read_bytes
        descriptors.append(
            {
                "ordinal": ordinal,
                "name": arm["name"],
                "attention_policy": policy,
                "mature_visible_values": mature_visible_values,
                "mature_visible_key_rows": mature_visible_key_rows,
                "attention_expectations_per_sequence": expectations,
            }
        )
    return descriptors


def _position_grid_is_exact(rows: Any) -> bool:
    if (
        not isinstance(rows, list)
        or len(rows) != _SEQUENCES * _POSITIONS_PER_SEQUENCE
    ):
        return False
    coordinates: list[tuple[int, int]] = []
    for row in rows:
        if not isinstance(row, dict):
            return False
        try:
            coordinates.append((int(row["sequence"]), int(row["position"])))
        except (KeyError, TypeError, ValueError):
            return False
    expected = [
        (sequence, position)
        for sequence in range(_SEQUENCES)
        for position in range(_POSITIONS_PER_SEQUENCE)
    ]
    return coordinates == expected


def _bands_from_rows(
    rows: list[dict[str, float | bool | int]],
) -> dict[str, dict[str, Any]]:
    return {
        name: _aggregate(
            [row for row in rows if start <= int(row["position"]) < stop]
        )
        for name, start, stop in _QUALITY_BANDS
    }


def _validate_control_prerequisite(
    sustained_protocol: dict[str, Any],
    sustained_result: dict[str, Any],
    control_protocol: dict[str, Any],
    control_result: dict[str, Any],
    *,
    sustained_protocol_hash: str,
    sustained_result_hash: str,
    control_protocol_hash: str,
    control_result_hash: str,
    identities: dict[str, str],
    evaluator_sources: dict[str, str],
    control_source_hash: str,
) -> None:
    del control_result_hash
    _validate_failed_sustained_result(
        sustained_protocol,
        sustained_result,
        protocol_hash=sustained_protocol_hash,
    )
    control_scope = control_protocol.get("scope")
    if (
        control_protocol.get("schema_version") != 1
        or control_protocol.get("experiment") != _CONTROL_EXPERIMENT
        or control_protocol.get("status") != _CONTROL_PROTOCOL_STATUS
        or control_protocol.get("source_revision")
        != sustained_protocol.get("source_revision")
        or control_protocol.get("sustained_protocol_sha256")
        != sustained_protocol_hash
        or control_protocol.get("sustained_result_sha256")
        != sustained_result_hash
        or control_protocol.get("package_manifest_sha256")
        != identities["package_manifest_sha256"]
        or control_protocol.get("native_library_sha256")
        != identities["native_library_sha256"]
        or control_protocol.get("dataset_sha256") != identities["dataset_sha256"]
        or control_protocol.get("corpus_manifest_sha256")
        != identities["corpus_manifest_sha256"]
        or control_protocol.get("teacher_reference_sha256")
        != identities["teacher_reference_sha256"]
        or control_protocol.get("teacher_arrays_sha256")
        != identities["teacher_arrays_sha256"]
        or control_protocol.get("control_source_sha256") != control_source_hash
        or control_protocol.get("frozen_evaluator_source_sha256")
        != evaluator_sources
        or control_protocol.get("source_config_sha256")
        != sustained_protocol.get("source_config_sha256")
        or control_protocol.get("source_index_sha256")
        != sustained_protocol.get("source_index_sha256")
        or control_protocol.get("source_shard_sha256")
        != sustained_protocol.get("source_shard_sha256")
        or control_protocol.get("input_ids") != sustained_protocol.get("input_ids")
        or control_protocol.get("input_identity")
        != sustained_protocol.get("input_identity")
        or control_protocol.get("model") != sustained_protocol.get("model")
        or control_protocol.get("bounded_attention_policy")
        != sustained_protocol.get("attention_policy")
        or control_protocol.get("control_attention_policy") != _control_policy()
        or control_protocol.get("quality_bands") != _expected_bands()
        or control_protocol.get("thresholds") != _THRESHOLDS
        or not isinstance(control_scope, dict)
        or control_scope.get("candidate_threads") != _SWEEP_THREADS
        or control_scope.get("candidate_device") != "cpu"
        or control_scope.get("candidate_transformers_model_shell") is not False
        or control_scope.get("only_intervention") != "local_window_16_to_128"
        or control_scope.get("full_causal_attention_for_positions")
        != _POSITIONS_PER_SEQUENCE
        or control_scope.get("q7_artifact_or_policy_changed") is not False
        or control_scope.get("deployable_attention_traffic_gate_applies") is not False
        or control_scope.get("control_frozen_before_execution") is not True
    ):
        raise ValueError("attention sweep control protocol prerequisite is invalid")

    model = sustained_protocol["model"]
    control_expectations = _attention_expectations(model, _control_policy())
    q7_expectations = _q7_expectations(model)
    rows = control_result.get("position_results")
    if not _position_grid_is_exact(rows):
        raise ValueError("attention sweep control position grid is invalid")
    aggregate = _aggregate(rows)
    bands = _bands_from_rows(rows)
    expected_quality_checks = _quality_checks("overall", aggregate)
    for name, metrics in bands.items():
        expected_quality_checks.update(_quality_checks(name, metrics))
    artifacts = control_result.get("artifacts")
    evidence_checks = control_result.get("evidence_checks")
    post_authentication = control_result.get("post_run_authentication")
    if (
        control_protocol.get("attention_expectations_per_sequence")
        != control_expectations
        or control_protocol.get("q7_expectations_per_sequence") != q7_expectations
        or control_result.get("schema_version") != 1
        or control_result.get("experiment") != _CONTROL_EXPERIMENT
        or control_result.get("status") != "post_failure_diagnostic_complete"
        or control_result.get("evidence_passed") is not True
        or control_result.get("quality_passed") is not True
        or control_result.get("diagnosis")
        != "bounded_attention_is_dominant_sustained_drift_source"
        or control_result.get("metrics") != aggregate
        or control_result.get("position_bands") != bands
        or control_result.get("quality_checks") != expected_quality_checks
        or not all(expected_quality_checks.values())
        or not isinstance(evidence_checks, dict)
        or not evidence_checks
        or not all(value is True for value in evidence_checks.values())
        or not isinstance(post_authentication, dict)
        or not post_authentication
        or not all(value is True for value in post_authentication.values())
        or control_result.get("attention_expectations_per_sequence")
        != control_expectations
        or control_result.get("q7_expectations_per_sequence") != q7_expectations
        or control_result.get("configuration", {}).get("attention_policy")
        != _control_policy()
        or control_result.get("configuration", {}).get("candidate_threads")
        != _SWEEP_THREADS
        or control_result.get("configuration", {}).get(
            "q7_artifact_or_policy_changed"
        )
        is not False
        or control_result.get("pre_intervention_identity")
        != {
            "expected_positions": _SEQUENCES * 16,
            "bounded_positions": _SEQUENCES * 16,
            "control_positions": _SEQUENCES * 16,
            "exact_position_metrics_match": True,
        }
        or not isinstance(artifacts, dict)
        or artifacts.get("sustained_protocol_sha256")
        != sustained_protocol_hash
        or artifacts.get("sustained_result_sha256") != sustained_result_hash
        or artifacts.get("control_protocol_sha256") != control_protocol_hash
        or artifacts.get("control_source_sha256") != control_source_hash
        or any(
            artifacts.get(name) != value
            for name, value in identities.items()
            if name
            in {
                "package_manifest_sha256",
                "native_library_sha256",
                "dataset_sha256",
                "corpus_manifest_sha256",
                "teacher_reference_sha256",
                "teacher_arrays_sha256",
            }
        )
    ):
        raise ValueError("attention sweep control result prerequisite is invalid")


def _authenticate_prerequisites(
    *,
    package_path: Path,
    manifest_sha256: str,
    library_path: Path,
    dataset_path: Path,
    corpus_manifest_path: Path,
    reference_path: Path,
    arrays_path: Path,
    sustained_protocol_path: Path,
    sustained_protocol_sha256: str,
    sustained_result_path: Path,
    sustained_result_sha256: str,
    control_protocol_path: Path,
    control_protocol_sha256: str,
    control_result_path: Path,
    control_result_sha256: str,
) -> dict[str, Any]:
    sustained_protocol = _read_object(
        sustained_protocol_path,
        "sustained protocol",
    )
    sustained_result = _read_object(sustained_result_path, "sustained result")
    control_protocol = _read_object(
        control_protocol_path,
        "dense-attention control protocol",
    )
    control_result = _read_object(
        control_result_path,
        "dense-attention control result",
    )
    reference = _read_object(reference_path, "sustained teacher reference")
    hashes = {
        "sustained_protocol_sha256": sha256_file(sustained_protocol_path),
        "sustained_result_sha256": sha256_file(sustained_result_path),
        "control_protocol_sha256": sha256_file(control_protocol_path),
        "control_result_sha256": sha256_file(control_result_path),
    }
    if (
        hashes["sustained_protocol_sha256"]
        != sustained_protocol_sha256.lower()
        or hashes["sustained_result_sha256"] != sustained_result_sha256.lower()
        or hashes["control_protocol_sha256"] != control_protocol_sha256.lower()
        or hashes["control_result_sha256"] != control_result_sha256.lower()
    ):
        raise ValueError("attention sweep prerequisite hash is invalid")
    identities = {
        "package_manifest_sha256": manifest_sha256.lower(),
        "native_library_sha256": sha256_file(library_path),
        "dataset_sha256": sha256_file(dataset_path),
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "teacher_reference_sha256": sha256_file(reference_path),
        "teacher_arrays_sha256": sha256_file(arrays_path),
    }
    if (
        sustained_protocol.get("schema_version") != 1
        or sustained_protocol.get("experiment")
        != "olmoe_native_sustained_context_confirmation"
        or sustained_protocol.get("status") != "frozen_before_candidate_execution"
        or sustained_protocol.get("package_manifest_sha256")
        != identities["package_manifest_sha256"]
        or any(
            sustained_protocol.get(name) != value
            for name, value in identities.items()
            if name != "package_manifest_sha256"
        )
        or sustained_protocol.get("sequences") != _SEQUENCES
        or sustained_protocol.get("tokens_per_sequence") != _TOKENS_PER_SEQUENCE
        or sustained_protocol.get("quality_bands") != _expected_bands()
        or sustained_protocol.get("thresholds") != _THRESHOLDS
        or sustained_protocol.get("attention_policy")
        != {
            "local_window": 16,
            "older_candidates": 8,
            "older_top_k": 4,
            "sink_tokens": 2,
        }
        or sustained_protocol.get("scope", {}).get("candidate_threads")
        != _SWEEP_THREADS
        or sustained_protocol.get("scope", {}).get("teacher_configuration")
        != _TEACHER_CONFIGURATION
    ):
        raise ValueError("attention sweep sustained prerequisite is invalid")
    evaluator_sources = _authenticate_evaluator_sources(sustained_protocol)
    _validate_teacher_source(reference, sustained_protocol)
    if (
        _teacher_configuration(reference) != _TEACHER_CONFIGURATION
        or reference.get("dataset", {}).get("input_ids")
        != sustained_protocol.get("input_ids")
        or reference.get("dataset", {}).get("input_identity")
        != sustained_protocol.get("input_identity")
        or reference.get("dataset", {}).get("prediction_positions")
        != _SEQUENCES * _POSITIONS_PER_SEQUENCE
    ):
        raise ValueError("attention sweep teacher prerequisite is invalid")
    control_source_path = Path(dense_control_source.__file__).resolve()
    control_source_hash = sha256_file(control_source_path)
    _validate_control_prerequisite(
        sustained_protocol,
        sustained_result,
        control_protocol,
        control_result,
        sustained_protocol_hash=hashes["sustained_protocol_sha256"],
        sustained_result_hash=hashes["sustained_result_sha256"],
        control_protocol_hash=hashes["control_protocol_sha256"],
        control_result_hash=hashes["control_result_sha256"],
        identities=identities,
        evaluator_sources=evaluator_sources,
        control_source_hash=control_source_hash,
    )
    input_ids = sustained_protocol["input_ids"]
    if sha256_json(input_ids) != sustained_protocol["input_identity"]:
        raise ValueError("attention sweep input identity is invalid")
    manifest = validate_olmoe_native_package(
        package_path,
        expected_manifest_sha256=manifest_sha256,
    )
    config_path, non_mlp_path, q7_path, tokenizer_path = _paths_from_manifest(
        package_path,
        manifest,
    )
    if (
        manifest.get("runtime", {}).get("kernel_threads") != _SWEEP_THREADS
        or manifest.get("runtime", {}).get("attention_policy")
        != sustained_protocol["attention_policy"]
        or manifest.get("source", {}).get("revision")
        != sustained_protocol["source_revision"]
        or manifest.get("files", {}).get("model/config.json", {}).get("sha256")
        != sustained_protocol["source_config_sha256"]
    ):
        raise ValueError("attention sweep package prerequisite is invalid")
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for the attention sweep"
        ) from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    retokenized, _corpus = _validate_corpus_manifest(
        dataset_path,
        corpus_manifest_path,
        tokenizer,
        tokenizer_sha256=manifest["files"]["tokenizer/tokenizer.json"]["sha256"],
    )
    if retokenized != input_ids:
        raise ValueError("attention sweep retokenization differs")
    model = sustained_protocol["model"]
    q7_expectations = _q7_expectations(model)
    if (
        q7_expectations != sustained_protocol["q7_expectations_per_sequence"]
        or q7_expectations != control_protocol["q7_expectations_per_sequence"]
        or q7_expectations != control_result["q7_expectations_per_sequence"]
    ):
        raise ValueError("attention sweep Q7 prerequisite is invalid")
    return {
        "sustained_protocol": sustained_protocol,
        "sustained_result": sustained_result,
        "control_protocol": control_protocol,
        "control_result": control_result,
        "reference": reference,
        "hashes": hashes,
        "identities": identities,
        "evaluator_sources": evaluator_sources,
        "control_source_path": control_source_path,
        "control_source_hash": control_source_hash,
        "manifest": manifest,
        "config_path": config_path,
        "non_mlp_path": non_mlp_path,
        "q7_path": q7_path,
        "model": model,
        "q7_expectations": q7_expectations,
        "input_ids": input_ids,
    }


def freeze_native_olmoe_attention_sweep_protocol(
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
    control_result: str | Path,
    control_result_sha256: str,
    out: str | Path,
    threads: int = _SWEEP_THREADS,
) -> dict[str, Any]:
    """Freeze all post-attribution sweep arms before executing any arm."""

    output_path = Path(out).expanduser().resolve()
    if output_path.exists():
        raise ValueError("attention sweep protocol target already exists")
    if threads != _SWEEP_THREADS:
        raise ValueError("attention sweep requires the matched 12 threads")
    paths = {
        "package_path": Path(package).expanduser().resolve(),
        "library_path": Path(library).expanduser().resolve(),
        "dataset_path": Path(dataset).expanduser().resolve(),
        "corpus_manifest_path": Path(corpus_manifest).expanduser().resolve(),
        "reference_path": Path(teacher_reference).expanduser().resolve(),
        "arrays_path": Path(teacher_arrays).expanduser().resolve(),
        "sustained_protocol_path": Path(sustained_protocol)
        .expanduser()
        .resolve(),
        "sustained_result_path": Path(sustained_result).expanduser().resolve(),
        "control_protocol_path": Path(control_protocol).expanduser().resolve(),
        "control_result_path": Path(control_result).expanduser().resolve(),
    }
    context = _authenticate_prerequisites(
        **paths,
        manifest_sha256=manifest_sha256,
        sustained_protocol_sha256=sustained_protocol_sha256,
        sustained_result_sha256=sustained_result_sha256,
        control_protocol_sha256=control_protocol_sha256,
        control_result_sha256=control_result_sha256,
    )
    protocol = context["sustained_protocol"]
    hashes = context["hashes"]
    sweep_source = Path(__file__).resolve()
    sweep_protocol = {
        "schema_version": 1,
        "experiment": _SWEEP_EXPERIMENT,
        "status": _SWEEP_STATUS,
        "source_revision": protocol["source_revision"],
        **context["identities"],
        **hashes,
        "control_source_sha256": context["control_source_hash"],
        "sweep_source_sha256": sha256_file(sweep_source),
        "frozen_evaluator_source_sha256": context["evaluator_sources"],
        "source_config_sha256": protocol["source_config_sha256"],
        "source_index_sha256": protocol["source_index_sha256"],
        "source_shard_sha256": protocol["source_shard_sha256"],
        "input_identity": protocol["input_identity"],
        "input_ids": context["input_ids"],
        "model": context["model"],
        "q7_expectations_per_sequence": context["q7_expectations"],
        "quality_bands": _expected_bands(),
        "thresholds": _THRESHOLDS,
        "arms": _arm_descriptors(context["model"]),
        "per_position_read_contract": _per_position_read_contract(),
        "ranking_rule": _ranking_rule(),
        "scope": {
            "candidate_device": "cpu",
            "candidate_threads": threads,
            "candidate_transformers_model_shell": False,
            "execution_interface": _EXECUTION_INTERFACE,
            "source_package_attention_policy":
            _SOURCE_PACKAGE_ATTENTION_POLICY,
            "attention_policy_overridden_for_development": True,
            "package_manifest_mutated": False,
            "arms_execute_sequentially_in_frozen_order": True,
            "intermediate_outputs_inspected_or_used_to_adapt_later_arms": False,
            "q7_artifact_or_policy_changed": False,
            "corpus_or_teacher_changed": False,
            "mature_visible_values_per_arm": _MATURE_VISIBLE_VALUES,
            "mature_visible_key_rows_per_arm": _MATURE_VISIBLE_KEY_ROWS,
            "attention_logical_read_bytes_per_sequence":
            _EXPECTED_LOGICAL_READ_BYTES,
            "maximum_attention_logical_read_fraction": _THRESHOLDS[
                "maximum_attention_logical_read_fraction"
            ],
            "reset_replay_sequence_per_arm": 0,
            "development_selection_only": True,
            "fresh_confirmation_required": True,
            "protocol_frozen_before_any_arm_execution": True,
            "dense_attribution_result_known_before_sweep_freeze": True,
            "teacher_configuration": _TEACHER_CONFIGURATION,
        },
        "decision_rule": {
            "evidence_failure": (
                "invalidate the complete sweep and stop without selecting an arm"
            ),
            "one_or_more_quality_passes": (
                "rank passing arms by the frozen worst normalized quality "
                "margin, then state bytes; integrate the development-only "
                "selection into package tooling before fresh confirmation"
            ),
            "no_quality_passes": (
                "investigate a layer-adaptive or learned attention selector"
            ),
        },
        "limitations": [
            "This is a post-attribution development sweep on a consumed corpus.",
            "A selected arm must pass a separately frozen fresh confirmation.",
            "The immutable source package declares W16/C8/K4/S2; this development evaluator constructs the raw native token runtime with each frozen arm override.",
            "A selected policy must be integrated into the package compiler and validator before package-native fresh confirmation.",
            "Logical attention bytes are not measured hardware DRAM traffic.",
            "Arms match logical-read bytes and mature visible-value count, not state bytes, DRAM traffic, or latency.",
        ],
    }
    atomic_json(output_path, sweep_protocol)
    return sweep_protocol


def _validate_sweep_protocol(
    sweep_protocol: dict[str, Any],
    context: dict[str, Any],
    *,
    sweep_protocol_hash: str,
    supplied_sweep_protocol_hash: str,
    sweep_source_hash: str,
) -> None:
    scope = sweep_protocol.get("scope")
    protocol = context["sustained_protocol"]
    hashes = context["hashes"]
    if (
        sweep_protocol_hash != supplied_sweep_protocol_hash.lower()
        or sweep_protocol.get("schema_version") != 1
        or sweep_protocol.get("experiment") != _SWEEP_EXPERIMENT
        or sweep_protocol.get("status") != _SWEEP_STATUS
        or sweep_protocol.get("source_revision") != protocol["source_revision"]
        or any(sweep_protocol.get(name) != value for name, value in hashes.items())
        or any(
            sweep_protocol.get(name) != value
            for name, value in context["identities"].items()
        )
        or sweep_protocol.get("control_source_sha256")
        != context["control_source_hash"]
        or sweep_protocol.get("sweep_source_sha256") != sweep_source_hash
        or sweep_protocol.get("frozen_evaluator_source_sha256")
        != context["evaluator_sources"]
        or sweep_protocol.get("source_config_sha256")
        != protocol["source_config_sha256"]
        or sweep_protocol.get("source_index_sha256")
        != protocol["source_index_sha256"]
        or sweep_protocol.get("source_shard_sha256")
        != protocol["source_shard_sha256"]
        or sweep_protocol.get("input_identity") != protocol["input_identity"]
        or sweep_protocol.get("input_ids") != context["input_ids"]
        or sweep_protocol.get("model") != context["model"]
        or sweep_protocol.get("q7_expectations_per_sequence")
        != context["q7_expectations"]
        or sweep_protocol.get("quality_bands") != _expected_bands()
        or sweep_protocol.get("thresholds") != _THRESHOLDS
        or sweep_protocol.get("arms") != _arm_descriptors(context["model"])
        or sweep_protocol.get("per_position_read_contract")
        != _per_position_read_contract()
        or sweep_protocol.get("ranking_rule") != _ranking_rule()
        or not isinstance(scope, dict)
        or scope.get("candidate_device") != "cpu"
        or scope.get("candidate_threads") != _SWEEP_THREADS
        or scope.get("candidate_transformers_model_shell") is not False
        or scope.get("execution_interface") != _EXECUTION_INTERFACE
        or scope.get("source_package_attention_policy")
        != _SOURCE_PACKAGE_ATTENTION_POLICY
        or scope.get("attention_policy_overridden_for_development") is not True
        or scope.get("package_manifest_mutated") is not False
        or scope.get("arms_execute_sequentially_in_frozen_order") is not True
        or scope.get("intermediate_outputs_inspected_or_used_to_adapt_later_arms")
        is not False
        or scope.get("q7_artifact_or_policy_changed") is not False
        or scope.get("corpus_or_teacher_changed") is not False
        or scope.get("mature_visible_values_per_arm") != _MATURE_VISIBLE_VALUES
        or scope.get("mature_visible_key_rows_per_arm")
        != _MATURE_VISIBLE_KEY_ROWS
        or scope.get("attention_logical_read_bytes_per_sequence")
        != _EXPECTED_LOGICAL_READ_BYTES
        or scope.get("maximum_attention_logical_read_fraction")
        != _THRESHOLDS["maximum_attention_logical_read_fraction"]
        or scope.get("reset_replay_sequence_per_arm") != 0
        or scope.get("development_selection_only") is not True
        or scope.get("fresh_confirmation_required") is not True
        or scope.get("protocol_frozen_before_any_arm_execution") is not True
        or scope.get("dense_attribution_result_known_before_sweep_freeze")
        is not True
        or scope.get("teacher_configuration") != _TEACHER_CONFIGURATION
    ):
        raise ValueError("attention sweep protocol contract is invalid")


def _structural_checks(
    metrics: dict[str, int],
    expectations: dict[str, int | float],
    q7_expectations: dict[str, int],
    *,
    position: int,
) -> dict[str, bool]:
    checks = _sustained_structural_checks(
        metrics,
        expectations,
        position=position,
    )
    checks.update(
        {
            "q7_scheduled_bytes": (
                metrics.get("q7_scheduled_bytes")
                == q7_expectations["scheduled_bytes_per_sequence"]
            ),
            "fixed_attention_logical_read_bytes": (
                metrics.get("attention_logical_read_bytes")
                == _EXPECTED_LOGICAL_READ_BYTES
            ),
        }
    )
    return checks


def _counter_snapshot_checks(
    metrics: dict[str, int],
    expectations: dict[str, int | float],
    q7_expectations: dict[str, int],
    *,
    position: int,
    previous_logical_read_bytes: int,
    expected_logical_read_delta_bytes: int,
) -> dict[str, bool]:
    checks = _sustained_structural_checks(
        metrics,
        expectations,
        position=position,
    )
    logical_read_bytes = int(metrics.get("attention_logical_read_bytes", -1))
    checks.update(
        {
            "q7_scheduled_bytes": (
                metrics.get("q7_scheduled_bytes")
                == position * q7_expectations["scheduled_bytes_per_position"]
            ),
            "logical_read_delta_bytes": (
                logical_read_bytes - previous_logical_read_bytes
                == expected_logical_read_delta_bytes
            ),
        }
    )
    return checks


def _update_counter_digest(
    digest: Any,
    deterministic_metrics: dict[str, int],
) -> None:
    digest.update(
        json.dumps(
            deterministic_metrics,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")


def _pre_eviction_identity(
    control_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    *,
    local_window: int,
) -> dict[str, Any]:
    control = [
        row for row in control_rows if int(row.get("position", -1)) < local_window
    ]
    arm = [
        row for row in arm_rows if int(row.get("position", -1)) < local_window
    ]
    expected = _SEQUENCES * local_window
    return {
        "local_window": local_window,
        "expected_positions": expected,
        "control_positions": len(control),
        "arm_positions": len(arm),
        "exact_position_metrics_match": (
            len(control) == expected and len(arm) == expected and control == arm
        ),
    }


def _quality_margin(
    overall: dict[str, Any],
    bands: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    populations = {
        "overall": overall,
        **{
            name: metrics
            for name, metrics in bands.items()
            if name != "positions_0_15"
        },
    }
    details: dict[str, dict[str, float]] = {}
    worst: dict[str, Any] | None = None
    for population, metrics in populations.items():
        values = {
            "teacher_to_native_kl": (
                (
                    _THRESHOLDS["maximum_mean_kl"]
                    - float(metrics["teacher_to_native_kl"])
                )
                / _THRESHOLDS["maximum_mean_kl"]
            ),
            "teacher_top1_agreement": (
                (
                    float(metrics["teacher_top1_agreement"])
                    - _THRESHOLDS["minimum_top1_agreement"]
                )
                / (1.0 - _THRESHOLDS["minimum_top1_agreement"])
            ),
            "target_nll_delta": (
                (
                    _THRESHOLDS["maximum_mean_target_nll_delta"]
                    - float(metrics["target_nll_delta"])
                )
                / _THRESHOLDS["maximum_mean_target_nll_delta"]
            ),
            "final_hidden_relative_l2": (
                (
                    _THRESHOLDS["maximum_mean_final_hidden_relative_l2"]
                    - float(metrics["final_hidden_relative_l2"])
                )
                / _THRESHOLDS["maximum_mean_final_hidden_relative_l2"]
            ),
        }
        details[population] = values
        for metric, margin in values.items():
            if worst is None or margin < float(worst["normalized_margin"]):
                worst = {
                    "population": population,
                    "metric": metric,
                    "normalized_margin": margin,
                }
    if worst is None:
        raise RuntimeError("attention sweep quality margin has no populations")
    return {
        "normalized_margins": details,
        "worst": worst,
        "worst_normalized_quality_margin": float(worst["normalized_margin"]),
    }


def _evaluate_arm(
    descriptor: dict[str, Any],
    *,
    context: dict[str, Any],
    library_path: Path,
    teacher_logits: np.ndarray,
    teacher_hidden: np.ndarray,
    targets: np.ndarray,
    threads: int,
) -> dict[str, Any]:
    policy = descriptor["attention_policy"]
    expectations = descriptor["attention_expectations_per_sequence"]
    q7_expectations = context["q7_expectations"]
    input_ids = context["input_ids"]
    model = context["model"]
    per_position_expectations = [
        _attention_expectations(model, policy, positions=position)
        for position in range(1, _POSITIONS_PER_SEQUENCE + 1)
    ]
    logical_row_bytes = (
        int(model["layers"])
        * int(model["query_heads"])
        * int(model["head_dimension"])
        * 4
    )
    all_rows: list[dict[str, float | bool | int]] = []
    band_rows: dict[str, list[dict[str, float | bool | int]]] = {
        name: [] for name, _start, _stop in _QUALITY_BANDS
    }
    sequence_results: list[dict[str, Any]] = []
    total_q7_bytes = 0
    replay_reference: dict[str, Any] | None = None
    load_started = time.perf_counter()
    runtime = OLMoENativeTokenRuntime(
        context["config_path"],
        context["non_mlp_path"],
        context["q7_path"],
        library_path,
        threads=threads,
        local_window=policy["local_window"],
        older_candidates=policy["older_candidates"],
        older_top_k=policy["older_top_k"],
        sink_tokens=policy["sink_tokens"],
    )
    cold_load_seconds = time.perf_counter() - load_started
    try:
        if not runtime.attention_metrics_available:
            raise ValueError("attention sweep metric ABI is unavailable")
        offset = 0
        for sequence_index, sequence in enumerate(input_ids):
            runtime.reset()
            rows: list[dict[str, float | bool | int]] = []
            top1_tokens: list[int] = []
            hidden_digest = hashlib.sha256()
            logit_digest = hashlib.sha256()
            counter_digest = hashlib.sha256()
            counter_stream: list[dict[str, Any]] = []
            previous_logical_read_bytes = 0
            started = time.perf_counter()
            for position, token_id in enumerate(sequence[:-1]):
                native_result = runtime.forward([token_id])
                native_hidden, native_logits = runtime.last_diagnostics()
                if int(np.argmax(native_logits)) != native_result.next_token:
                    raise ValueError("attention sweep diagnostic argmax differs")
                cumulative_metrics = dict(native_result.metrics)
                deterministic_metrics = _deterministic_metrics(cumulative_metrics)
                expected_logical_read_delta_bytes = logical_row_bytes * (
                    min(position + 1, _MATURE_VISIBLE_KEY_ROWS)
                    + min(position + 1, _MATURE_VISIBLE_VALUES)
                )
                counter_checks = _counter_snapshot_checks(
                    cumulative_metrics,
                    per_position_expectations[position],
                    q7_expectations,
                    position=runtime.position,
                    previous_logical_read_bytes=previous_logical_read_bytes,
                    expected_logical_read_delta_bytes=(
                        expected_logical_read_delta_bytes
                    ),
                )
                previous_logical_read_bytes = int(
                    cumulative_metrics["attention_logical_read_bytes"]
                )
                _update_counter_digest(counter_digest, deterministic_metrics)
                counter_stream.append(
                    {
                        "position": position,
                        "deterministic_metrics": deterministic_metrics,
                        "expected_logical_read_delta_bytes": (
                            expected_logical_read_delta_bytes
                        ),
                        "checks": counter_checks,
                        "passed": all(counter_checks.values()),
                    }
                )
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
            if runtime.last_result is None:
                raise RuntimeError("attention sweep runtime has no final result")
            metrics = dict(runtime.last_result.metrics)
            structural = _structural_checks(
                metrics,
                expectations,
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
                    "counter_stream_sha256": counter_digest.hexdigest(),
                }
            sequence_results.append(
                {
                    "sequence": sequence_index,
                    "elapsed_seconds": elapsed,
                    "metrics": _aggregate(rows),
                    "native_metrics": metrics,
                    "diagnostic_hashes": diagnostic_hashes,
                    "counter_stream": counter_stream,
                    "counter_stream_sha256": counter_digest.hexdigest(),
                    "counter_stream_passed": all(
                        entry["passed"] for entry in counter_stream
                    ),
                    "structural_checks": structural,
                    "structural_passed": all(structural.values()),
                }
            )
        if replay_reference is None:
            raise RuntimeError("attention sweep replay reference is missing")
        runtime.reset()
        replay_tokens: list[int] = []
        hidden_digest = hashlib.sha256()
        logit_digest = hashlib.sha256()
        counter_digest = hashlib.sha256()
        replay_counter_checks_passed = True
        previous_logical_read_bytes = 0
        replay_started = time.perf_counter()
        for position, token_id in enumerate(input_ids[0][:-1]):
            replay_result = runtime.forward([token_id])
            replay_hidden, replay_logits = runtime.last_diagnostics()
            replay_tokens.append(replay_result.next_token)
            cumulative_metrics = dict(replay_result.metrics)
            deterministic_metrics = _deterministic_metrics(cumulative_metrics)
            expected_logical_read_delta_bytes = logical_row_bytes * (
                min(position + 1, _MATURE_VISIBLE_KEY_ROWS)
                + min(position + 1, _MATURE_VISIBLE_VALUES)
            )
            counter_checks = _counter_snapshot_checks(
                cumulative_metrics,
                per_position_expectations[position],
                q7_expectations,
                position=runtime.position,
                previous_logical_read_bytes=previous_logical_read_bytes,
                expected_logical_read_delta_bytes=(
                    expected_logical_read_delta_bytes
                ),
            )
            previous_logical_read_bytes = int(
                cumulative_metrics["attention_logical_read_bytes"]
            )
            replay_counter_checks_passed = (
                replay_counter_checks_passed and all(counter_checks.values())
            )
            _update_counter_digest(counter_digest, deterministic_metrics)
            _update_diagnostic_hashes(
                hidden_digest,
                logit_digest,
                replay_hidden,
                replay_logits,
            )
        replay_seconds = time.perf_counter() - replay_started
        if runtime.last_result is None:
            raise RuntimeError("attention sweep replay has no final result")
        replay_metrics = dict(runtime.last_result.metrics)
        replay_hashes = {
            "hidden_sha256": hidden_digest.hexdigest(),
            "logits_sha256": logit_digest.hexdigest(),
        }
        replay_structural = _structural_checks(
            replay_metrics,
            expectations,
            q7_expectations,
            position=runtime.position,
        )
        reset_replay = {
            "sequence": 0,
            "elapsed_seconds": replay_seconds,
            "top1_tokens_match": replay_tokens == replay_reference["top1_tokens"],
            "diagnostic_hashes_match": (
                replay_hashes == replay_reference["diagnostic_hashes"]
            ),
            "deterministic_metrics_match": (
                _deterministic_metrics(replay_metrics)
                == replay_reference["deterministic_metrics"]
            ),
            "counter_stream_sha256": counter_digest.hexdigest(),
            "diagnostic_hashes": replay_hashes,
            "deterministic_metrics": _deterministic_metrics(replay_metrics),
            "counter_stream_hash_match": (
                counter_digest.hexdigest()
                == replay_reference["counter_stream_sha256"]
            ),
            "counter_stream_checks_passed": replay_counter_checks_passed,
            "structural_checks": replay_structural,
        }
        reset_replay["passed"] = (
            reset_replay["top1_tokens_match"]
            and reset_replay["diagnostic_hashes_match"]
            and reset_replay["deterministic_metrics_match"]
            and reset_replay["counter_stream_hash_match"]
            and reset_replay["counter_stream_checks_passed"]
            and all(replay_structural.values())
        )
    finally:
        runtime.close()

    aggregate = _aggregate(all_rows)
    bands = {name: _aggregate(rows) for name, rows in band_rows.items()}
    prediction_positions = len(all_rows)
    ideal_q4_bytes_per_position = (
        int(model["layers"])
        * int(model["experts"])
        * 3
        * int(model["hidden_size"])
        * int(model["intermediate_size"])
        // 2
    )
    all_expert_ideal_q4_bytes = (
        prediction_positions * ideal_q4_bytes_per_position
    )
    q7_traffic_fraction = total_q7_bytes / all_expert_ideal_q4_bytes
    quality_checks = _quality_checks("overall", aggregate)
    for name, metrics in bands.items():
        quality_checks.update(_quality_checks(name, metrics))
    control_result = context["control_result"]
    pre_eviction_identity = _pre_eviction_identity(
        control_result["position_results"],
        all_rows,
        local_window=int(policy["local_window"]),
    )
    comparison = {
        "overall": _compare_metrics(control_result["metrics"], aggregate),
        "position_bands": {
            name: _compare_metrics(control_result["position_bands"][name], metrics)
            for name, metrics in bands.items()
        },
    }
    per_position_offset = {
        str(position): _aggregate(
            [row for row in all_rows if row["position"] == position]
        )
        for position in range(_POSITIONS_PER_SEQUENCE)
    }
    evidence_checks = {
        "prediction_positions": (
            len(all_rows) == _SEQUENCES * _POSITIONS_PER_SEQUENCE
        ),
        "position_grid": _position_grid_is_exact(all_rows),
        "sequence_structural_checks": all(
            result["structural_passed"] for result in sequence_results
        ),
        "per_token_counter_streams": all(
            result["counter_stream_passed"] for result in sequence_results
        ),
        "q7_scheduled_bytes": (
            total_q7_bytes
            == _SEQUENCES * q7_expectations["scheduled_bytes_per_sequence"]
        ),
        "q7_traffic_fraction": (
            q7_traffic_fraction
            <= _THRESHOLDS["maximum_q7_traffic_fraction"]
        ),
        "fixed_attention_logical_reads": (
            expectations["attention_logical_read_bytes"]
            == _EXPECTED_LOGICAL_READ_BYTES
        ),
        "attention_logical_read_fraction": (
            expectations["attention_logical_read_fraction"]
            == _EXPECTED_LOGICAL_READ_FRACTION
            and expectations["attention_logical_read_fraction"]
            <= _THRESHOLDS["maximum_attention_logical_read_fraction"]
        ),
        "mature_visible_values": (
            descriptor["mature_visible_values"] == _MATURE_VISIBLE_VALUES
        ),
        "mature_visible_key_rows": (
            descriptor["mature_visible_key_rows"]
            == _MATURE_VISIBLE_KEY_ROWS
        ),
        "pre_eviction_identity": pre_eviction_identity[
            "exact_position_metrics_match"
        ],
        "reset_replay": reset_replay["passed"],
    }
    return {
        "ordinal": descriptor["ordinal"],
        "name": descriptor["name"],
        "attention_policy": policy,
        "mature_visible_values": descriptor["mature_visible_values"],
        "mature_visible_key_rows": descriptor["mature_visible_key_rows"],
        "attention_expectations_per_sequence": expectations,
        "q7_expectations_per_sequence": q7_expectations,
        "metrics": aggregate,
        "position_bands": bands,
        "per_position_offset": per_position_offset,
        "dense_control_comparison": comparison,
        "pre_eviction_identity": pre_eviction_identity,
        "quality_margin": _quality_margin(aggregate, bands),
        "quality_checks": quality_checks,
        "quality_passed": all(quality_checks.values()),
        "evidence_checks": evidence_checks,
        "sequence_results": sequence_results,
        "position_results": all_rows,
        "reset_replay": reset_replay,
        "traffic": {
            "q7_scheduled_bytes": total_q7_bytes,
            "all_expert_ideal_q4_bytes": all_expert_ideal_q4_bytes,
            "q7_fraction_of_all_expert_ideal_q4": q7_traffic_fraction,
            "attention_logical_read_bytes_per_sequence": expectations[
                "attention_logical_read_bytes"
            ],
            "dense_full_context_logical_kv_bytes_per_sequence": expectations[
                "dense_full_context_logical_kv_bytes"
            ],
            "attention_logical_read_fraction": expectations[
                "attention_logical_read_fraction"
            ],
            "measured_hardware_traffic": False,
        },
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
    }


def _post_authentication(
    *,
    context: dict[str, Any],
    package_path: Path,
    manifest_sha256: str,
    library_path: Path,
    dataset_path: Path,
    corpus_manifest_path: Path,
    reference_path: Path,
    arrays_path: Path,
    sustained_protocol_path: Path,
    sustained_result_path: Path,
    control_protocol_path: Path,
    control_result_path: Path,
    sweep_protocol_path: Path,
    sweep_protocol_hash: str,
    sweep_source_path: Path,
    sweep_source_hash: str,
) -> dict[str, bool]:
    protocol = context["sustained_protocol"]
    reference = context["reference"]
    identities = context["identities"]
    hashes = context["hashes"]
    source_model = Path(reference["source"]["model"]).expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    return {
        "package": (
            validate_olmoe_native_package(
                package_path,
                expected_manifest_sha256=manifest_sha256,
            )
            == context["manifest"]
        ),
        "library": (
            sha256_file(library_path) == identities["native_library_sha256"]
        ),
        "dataset": sha256_file(dataset_path) == identities["dataset_sha256"],
        "corpus_manifest": (
            sha256_file(corpus_manifest_path)
            == identities["corpus_manifest_sha256"]
        ),
        "teacher_reference": (
            sha256_file(reference_path)
            == identities["teacher_reference_sha256"]
        ),
        "teacher_arrays": (
            sha256_file(arrays_path) == identities["teacher_arrays_sha256"]
        ),
        "sustained_protocol": (
            sha256_file(sustained_protocol_path)
            == hashes["sustained_protocol_sha256"]
        ),
        "sustained_result": (
            sha256_file(sustained_result_path)
            == hashes["sustained_result_sha256"]
        ),
        "control_protocol": (
            sha256_file(control_protocol_path)
            == hashes["control_protocol_sha256"]
        ),
        "control_result": (
            sha256_file(control_result_path) == hashes["control_result_sha256"]
        ),
        "sweep_protocol": (
            sha256_file(sweep_protocol_path) == sweep_protocol_hash
        ),
        "teacher_source_config": (
            sha256_file(source_model / "config.json")
            == protocol["source_config_sha256"]
        ),
        "teacher_source_index": (
            sha256_file(source_model / "model.safetensors.index.json")
            == protocol["source_index_sha256"]
        ),
        "teacher_source_shards": _post_source_shards(
            reference,
            protocol["source_shard_sha256"],
        ),
        "frozen_evaluator_sources": all(
            sha256_file(repository / relative) == expected
            for relative, expected in context["evaluator_sources"].items()
        ),
        "control_source": (
            sha256_file(context["control_source_path"])
            == context["control_source_hash"]
        ),
        "sweep_source": (
            sha256_file(sweep_source_path) == sweep_source_hash
        ),
    }


def evaluate_native_olmoe_attention_sweep(
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
    control_result: str | Path,
    control_result_sha256: str,
    sweep_protocol: str | Path,
    sweep_protocol_sha256: str,
    out: str | Path,
    threads: int = _SWEEP_THREADS,
) -> dict[str, Any]:
    """Execute every frozen bounded-attention arm in fixed sequential order."""

    output_path = Path(out).expanduser().resolve()
    if output_path.exists():
        raise ValueError("attention sweep result target already exists")
    if threads != _SWEEP_THREADS:
        raise ValueError("attention sweep requires the matched 12 threads")
    paths = {
        "package_path": Path(package).expanduser().resolve(),
        "library_path": Path(library).expanduser().resolve(),
        "dataset_path": Path(dataset).expanduser().resolve(),
        "corpus_manifest_path": Path(corpus_manifest).expanduser().resolve(),
        "reference_path": Path(teacher_reference).expanduser().resolve(),
        "arrays_path": Path(teacher_arrays).expanduser().resolve(),
        "sustained_protocol_path": Path(sustained_protocol)
        .expanduser()
        .resolve(),
        "sustained_result_path": Path(sustained_result).expanduser().resolve(),
        "control_protocol_path": Path(control_protocol).expanduser().resolve(),
        "control_result_path": Path(control_result).expanduser().resolve(),
    }
    context = _authenticate_prerequisites(
        **paths,
        manifest_sha256=manifest_sha256,
        sustained_protocol_sha256=sustained_protocol_sha256,
        sustained_result_sha256=sustained_result_sha256,
        control_protocol_sha256=control_protocol_sha256,
        control_result_sha256=control_result_sha256,
    )
    sweep_protocol_path = Path(sweep_protocol).expanduser().resolve()
    sweep_protocol_value = _read_object(
        sweep_protocol_path,
        "attention sweep protocol",
    )
    sweep_protocol_hash = sha256_file(sweep_protocol_path)
    sweep_source_path = Path(__file__).resolve()
    sweep_source_hash = sha256_file(sweep_source_path)
    _validate_sweep_protocol(
        sweep_protocol_value,
        context,
        sweep_protocol_hash=sweep_protocol_hash,
        supplied_sweep_protocol_hash=sweep_protocol_sha256,
        sweep_source_hash=sweep_source_hash,
    )
    prediction_positions = _SEQUENCES * _POSITIONS_PER_SEQUENCE
    with np.load(paths["arrays_path"], allow_pickle=False) as arrays:
        if set(arrays.files) != {"logits", "hidden", "targets"}:
            raise ValueError("attention sweep teacher arrays have unexpected keys")
        teacher_logits = np.asarray(arrays["logits"], dtype=np.float32)
        teacher_hidden = np.asarray(arrays["hidden"], dtype=np.float32)
        targets = np.asarray(arrays["targets"], dtype=np.int64)
    expected_targets = np.asarray(
        [token for sequence in context["input_ids"] for token in sequence[1:]],
        dtype=np.int64,
    )
    model = context["model"]
    if (
        teacher_logits.shape
        != (prediction_positions, int(model["vocab_size"]))
        or teacher_hidden.shape
        != (prediction_positions, int(model["hidden_size"]))
        or targets.shape != (prediction_positions,)
        or not np.array_equal(targets, expected_targets)
    ):
        raise ValueError("attention sweep teacher array shapes are invalid")

    sweep_started = time.perf_counter()
    arm_results: list[dict[str, Any]] = []
    for descriptor in sweep_protocol_value["arms"]:
        arm_results.append(
            _evaluate_arm(
                descriptor,
                context=context,
                library_path=paths["library_path"],
                teacher_logits=teacher_logits,
                teacher_hidden=teacher_hidden,
                targets=targets,
                threads=threads,
            )
        )
    execution_seconds = time.perf_counter() - sweep_started
    post_authentication = _post_authentication(
        context=context,
        **paths,
        manifest_sha256=manifest_sha256,
        sweep_protocol_path=sweep_protocol_path,
        sweep_protocol_hash=sweep_protocol_hash,
        sweep_source_path=sweep_source_path,
        sweep_source_hash=sweep_source_hash,
    )
    post_authentication_passed = all(post_authentication.values())
    for arm in arm_results:
        arm["evidence_checks"]["post_run_authentication"] = (
            post_authentication_passed
        )
        arm["evidence_passed"] = all(arm["evidence_checks"].values())

    evidence_passed = post_authentication_passed and all(
        arm["evidence_passed"] for arm in arm_results
    )
    passing_arms = [arm for arm in arm_results if arm["quality_passed"]]
    ranked: list[dict[str, Any]] = []
    if evidence_passed:
        passing_arms.sort(
            key=lambda arm: (
                -float(
                    arm["quality_margin"][
                        "worst_normalized_quality_margin"
                    ]
                ),
                int(
                    arm["attention_expectations_per_sequence"][
                        "attention_state_bytes"
                    ]
                ),
                int(arm["ordinal"]),
            )
        )
        ranked = [
            {
                "rank": rank,
                "name": arm["name"],
                "worst_normalized_quality_margin": arm["quality_margin"][
                    "worst_normalized_quality_margin"
                ],
                "attention_state_bytes": arm[
                    "attention_expectations_per_sequence"
                ]["attention_state_bytes"],
                "frozen_ordinal": arm["ordinal"],
            }
            for rank, arm in enumerate(passing_arms, start=1)
        ]
    if not evidence_passed:
        status = "development_sweep_invalid"
        decision = "stop_and_diagnose_evidence"
        selected_arm = None
    elif ranked:
        status = "development_sweep_complete"
        decision = (
            "integrate_selected_attention_policy_then_freeze_fresh_"
            "package_native_confirmation"
        )
        selected_arm = ranked[0]["name"]
    else:
        status = "development_sweep_complete"
        decision = "investigate_layer_adaptive_or_learned_selector"
        selected_arm = None
    report = {
        "schema_version": 1,
        "experiment": _SWEEP_EXPERIMENT,
        "status": status,
        "provenance": {
            "protocol_frozen_before_any_arm_execution": True,
            "dense_attribution_result_known_before_sweep_freeze": True,
            "arms_executed_sequentially_in_frozen_order": True,
            "intermediate_outputs_used_to_adapt_later_arms": False,
            "scientific_role": (
                "post-attribution development selection; not confirmation"
            ),
            "execution_interface": _EXECUTION_INTERFACE,
            "source_package_attention_policy":
            _SOURCE_PACKAGE_ATTENTION_POLICY,
            "attention_policy_overridden_for_development": True,
            "package_manifest_mutated": False,
            "sweep_protocol_sha256": sweep_protocol_hash,
            "sweep_source_sha256": sweep_source_hash,
            **context["hashes"],
        },
        "artifacts": {
            **context["identities"],
            **context["hashes"],
            "control_source_sha256": context["control_source_hash"],
            "sweep_protocol_sha256": sweep_protocol_hash,
            "sweep_source_sha256": sweep_source_hash,
        },
        "configuration": {
            "candidate_device": "cpu",
            "candidate_threads": threads,
            "transformers_model_shell_used": False,
            "q7_artifact_or_policy_changed": False,
            "execution_interface": _EXECUTION_INTERFACE,
            "source_package_attention_policy":
            _SOURCE_PACKAGE_ATTENTION_POLICY,
            "attention_policy_overridden_for_development": True,
            "package_manifest_mutated": False,
            "execution_order": [arm["name"] for arm in arm_results],
            "mature_visible_values_per_arm": _MATURE_VISIBLE_VALUES,
            "mature_visible_key_rows_per_arm": _MATURE_VISIBLE_KEY_ROWS,
            "attention_logical_read_bytes_per_sequence":
            _EXPECTED_LOGICAL_READ_BYTES,
            "attention_logical_read_fraction":
            _EXPECTED_LOGICAL_READ_FRACTION,
            "per_position_read_contract": _per_position_read_contract(),
            "measured_hardware_traffic": False,
        },
        "quality_bands": _expected_bands(),
        "thresholds": _THRESHOLDS,
        "ranking_rule": _ranking_rule(),
        "arm_results": arm_results,
        "ranking": ranked,
        "selected_arm": selected_arm,
        "selection_is_development_only": selected_arm is not None,
        "fresh_confirmation_required": selected_arm is not None,
        "evidence_passed": evidence_passed,
        "diagnostic_quality_passing_arm_count": len(passing_arms),
        "decision": decision,
        "post_run_authentication": post_authentication,
        "performance": {
            "arm_execution_seconds": execution_seconds,
            "arm_total_sequence_seconds": {
                arm["name"]: arm["performance"]["total_sequence_seconds"]
                for arm in arm_results
            },
            "arm_reset_replay_seconds": {
                arm["name"]: arm["performance"]["reset_replay_seconds"]
                for arm in arm_results
            },
        },
        "limitations": [
            "The corpus and teacher were already consumed by prior diagnostics.",
            "Selection is development-only and requires fresh confirmation.",
            "The source package manifest remains immutable at W16/C8/K4/S2 while the raw native token runtime receives development-only policy overrides.",
            "A selected policy requires compiler and validator integration before a fresh package-native confirmation can begin.",
            "Pre-eviction identity proves exact metric-row equality, not hidden-state or logit tensor identity.",
            "Logical attention bytes are not measured hardware DRAM traffic.",
            "Equal arm budgets mean matched logical reads and visible-value count, not matched state bytes, DRAM traffic, or latency.",
        ],
    }
    atomic_json(output_path, report)
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze or run the native OLMoE bounded-attention sweep"
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
        command.add_argument("--control-protocol", required=True, type=Path)
        command.add_argument("--control-protocol-sha256", required=True)
        command.add_argument("--control-result", required=True, type=Path)
        command.add_argument("--control-result-sha256", required=True)
        command.add_argument("--out", required=True, type=Path)
        command.add_argument("--threads", type=int, default=_SWEEP_THREADS)

    freeze_parser = commands.add_parser(
        "freeze",
        help="freeze every matched attention arm before execution",
    )
    add_common_arguments(freeze_parser)
    evaluate_parser = commands.add_parser(
        "evaluate",
        help="execute a previously frozen attention sweep",
    )
    add_common_arguments(evaluate_parser)
    evaluate_parser.add_argument("--sweep-protocol", required=True, type=Path)
    evaluate_parser.add_argument("--sweep-protocol-sha256", required=True)
    args = parser.parse_args()
    common = {
        "package": args.package,
        "manifest_sha256": args.manifest_sha256,
        "library": args.library,
        "dataset": args.dataset,
        "corpus_manifest": args.corpus_manifest,
        "teacher_reference": args.teacher_reference,
        "teacher_arrays": args.teacher_arrays,
        "sustained_protocol": args.sustained_protocol,
        "sustained_protocol_sha256": args.sustained_protocol_sha256,
        "sustained_result": args.sustained_result,
        "sustained_result_sha256": args.sustained_result_sha256,
        "control_protocol": args.control_protocol,
        "control_protocol_sha256": args.control_protocol_sha256,
        "control_result": args.control_result,
        "control_result_sha256": args.control_result_sha256,
        "out": args.out,
        "threads": args.threads,
    }
    if args.command == "freeze":
        protocol = freeze_native_olmoe_attention_sweep_protocol(**common)
        print(
            json.dumps(
                {
                    "status": protocol["status"],
                    "arms": [
                        {
                            "name": arm["name"],
                            "attention_policy": arm["attention_policy"],
                        }
                        for arm in protocol["arms"]
                    ],
                    "protocol_sha256": sha256_file(args.out),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = evaluate_native_olmoe_attention_sweep(
        **common,
        sweep_protocol=args.sweep_protocol,
        sweep_protocol_sha256=args.sweep_protocol_sha256,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "evidence_passed": result["evidence_passed"],
                "diagnostic_quality_passing_arm_count": result[
                    "diagnostic_quality_passing_arm_count"
                ],
                "selected_arm": result["selected_arm"],
                "decision": result["decision"],
                "ranking": result["ranking"],
                "result_sha256": sha256_file(args.out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["evidence_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(_main())
