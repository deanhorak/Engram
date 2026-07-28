"""Frozen greedy layer-rescue experiment for native OLMoE attention.

This is deliberately separate from the already-frozen sustained, dense-control,
and matched-budget sweep evaluators.  It consumes those immutable artifacts as
authenticated prerequisites and changes only the per-layer attention schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

import engram.evaluation.olmoe_native_attention_sweep as sweep_source
from engram.compiler.olmoe_native import validate_olmoe_native_package
from engram.evaluation.olmoe_native_causal import _aggregate, _position_metrics
from engram.evaluation.olmoe_native_sustained import (
    _POSITIONS_PER_SEQUENCE,
    _QUALITY_BANDS,
    _SEQUENCES,
    _TEACHER_CONFIGURATION,
    _THRESHOLDS,
    _attention_expectations,
    _deterministic_metrics,
    _post_source_shards,
    _quality_checks,
    _structural_checks as _sustained_structural_checks,
    _update_diagnostic_hashes,
)
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.utils import atomic_json, sha256_file, sha256_json


_EXPERIMENT = "olmoe_native_q7_greedy_three_layer_dense_attention_rescue"
_PROTOCOL_STATUS = "frozen_before_any_layer_rescue_candidate_execution"
_THREADS = 12
_LAYERS = 16
_ROUNDS = 3
_SELECTION_SEQUENCES = 2
_HOLDOUT_SEQUENCES = 6
_CANDIDATE_COUNTS = [16, 15, 14]
_EXPECTED_CANDIDATE_EVALUATIONS = sum(_CANDIDATE_COUNTS)
_EXPECTED_FINAL_LOGICAL_READ_BYTES = 955_957_248
_EXPECTED_FINAL_LOGICAL_READ_FRACTION = 0.4417014898255814
_FOUR_RESCUE_LOGICAL_READ_BYTES = 1_048_903_680
_FOUR_RESCUE_LOGICAL_READ_FRACTION = 0.48464752906976744
_EXECUTION_INTERFACE = "raw_native_token_runtime_per_layer_attention_policies"

_BASE_POLICY = {
    "local_window": 16,
    "older_candidates": 8,
    "older_top_k": 4,
    "sink_tokens": 2,
}
_RESCUE_POLICY = {
    "local_window": 128,
    "older_candidates": 8,
    "older_top_k": 4,
    "sink_tokens": 2,
}
_SCORE_POPULATIONS = (
    "overall",
    "positions_16_31",
    "positions_32_63",
    "positions_64_95",
    "positions_96_127",
)
_SCORE_METRICS = (
    "teacher_to_native_kl",
    "teacher_top1_agreement",
    "target_nll_delta",
    "final_hidden_relative_l2",
)
_ROUND_RESOURCE_EXPECTATIONS = [
    {
        "rescued_layer_count": 1,
        "attention_logical_read_bytes": 770_064_384,
        "attention_state_bytes": 8_179_584,
        "attention_scratch_bytes": 4_736,
        "attention_eviction_events": 1_680,
        "attention_older_candidate_entries_scored": 208_320,
        "attention_older_selected_entries": 106_080,
        "attention_sink_insertions": 480,
        "attention_heavy_hitter_updates_minimum": 1_440,
        "attention_heavy_hitter_updates_maximum": 26_400,
    },
    {
        "rescued_layer_count": 2,
        "attention_logical_read_bytes": 863_010_816,
        "attention_state_bytes": 10_022_656,
        "attention_scratch_bytes": 5_632,
        "attention_eviction_events": 1_568,
        "attention_older_candidate_entries_scored": 194_432,
        "attention_older_selected_entries": 99_008,
        "attention_sink_insertions": 448,
        "attention_heavy_hitter_updates_minimum": 1_344,
        "attention_heavy_hitter_updates_maximum": 24_640,
    },
    {
        "rescued_layer_count": 3,
        "attention_logical_read_bytes": 955_957_248,
        "attention_state_bytes": 11_865_728,
        "attention_scratch_bytes": 6_528,
        "attention_eviction_events": 1_456,
        "attention_older_candidate_entries_scored": 180_544,
        "attention_older_selected_entries": 91_936,
        "attention_sink_insertions": 416,
        "attention_heavy_hitter_updates_minimum": 1_248,
        "attention_heavy_hitter_updates_maximum": 22_880,
    },
]


def _candidate_layer_order() -> list[int]:
    return list(range(_LAYERS))


def _record_split(record_ids: list[str]) -> dict[str, Any]:
    """Return the frozen SHA-256 ordering and the 2/6 sequence split."""

    if (
        len(record_ids) != _SEQUENCES
        or len(set(record_ids)) != _SEQUENCES
        or any(not isinstance(value, str) or not value for value in record_ids)
    ):
        raise ValueError("layer-rescue split requires eight distinct record IDs")
    ranked = sorted(
        (
            {
                "record_id": record_id,
                "sequence_index": sequence_index,
                "record_id_sha256": hashlib.sha256(
                    record_id.encode("utf-8")
                ).hexdigest(),
            }
            for sequence_index, record_id in enumerate(record_ids)
        ),
        key=lambda row: (
            row["record_id_sha256"],
            row["record_id"],
            row["sequence_index"],
        ),
    )
    selection = ranked[:_SELECTION_SEQUENCES]
    holdout = ranked[_SELECTION_SEQUENCES:]
    split = {
        "algorithm": (
            "ascending sha256(utf8(record_id)), then record_id, then original "
            "sequence index; first 2 selection, remaining 6 internal holdout"
        ),
        "ranked_records": ranked,
        "selection": selection,
        "internal_holdout": holdout,
    }
    split["split_identity"] = sha256_json(split)
    return split


def _dataset_record_ids(dataset: Path) -> list[str]:
    record_ids: list[str] = []
    with dataset.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"layer-rescue dataset line {line_number} is invalid JSON"
                ) from exc
            record_id = record.get("record_id") if isinstance(record, dict) else None
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(
                    f"layer-rescue dataset line {line_number} has no record ID"
                )
            record_ids.append(record_id)
    if len(record_ids) != _SEQUENCES or len(set(record_ids)) != _SEQUENCES:
        raise ValueError("layer-rescue dataset record inventory is invalid")
    return record_ids


def _schedule_policies(
    rescued_layers: list[int] | tuple[int, ...],
    *,
    layers: int = _LAYERS,
) -> list[dict[str, int]]:
    rescued = list(rescued_layers)
    if (
        layers != _LAYERS
        or len(set(rescued)) != len(rescued)
        or any(
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer < 0
            or layer >= layers
            for layer in rescued
        )
        or len(rescued) > _ROUNDS
    ):
        raise ValueError("layer-rescue schedule is invalid")
    rescued_set = set(rescued)
    return [
        dict(_RESCUE_POLICY if layer in rescued_set else _BASE_POLICY)
        for layer in range(layers)
    ]


def _schedule_expectations(
    model: dict[str, int],
    rescued_layers: list[int] | tuple[int, ...],
    *,
    positions: int = _POSITIONS_PER_SEQUENCE,
) -> dict[str, int | float]:
    """Analytically sum the exact native counters for a mixed layer schedule."""

    layers = int(model["layers"])
    if layers != _LAYERS:
        raise ValueError("layer rescue is frozen for the 16-layer OLMoE source")
    policies = _schedule_policies(rescued_layers, layers=layers)
    one_layer_model = dict(model)
    one_layer_model["layers"] = 1
    per_layer = [
        _attention_expectations(one_layer_model, policy, positions=positions)
        for policy in policies
    ]
    integer_names = (
        "attention_state_bytes",
        "attention_scratch_bytes",
        "attention_eviction_events",
        "attention_older_candidate_entries_scored",
        "attention_older_selected_entries",
        "attention_sink_insertions",
        "attention_heavy_hitter_updates_minimum",
        "attention_heavy_hitter_updates_maximum",
        "attention_local_kv_bytes",
        "attention_candidate_key_bytes",
        "attention_selected_value_bytes",
        "attention_logical_read_bytes",
        "dense_full_context_logical_kv_bytes",
    )
    result: dict[str, int | float] = {"positions_processed": positions}
    for name in integer_names:
        result[name] = sum(int(row[name]) for row in per_layer)
    dense = int(result["dense_full_context_logical_kv_bytes"])
    logical = int(result["attention_logical_read_bytes"])
    result["attention_logical_read_fraction"] = logical / dense
    return result


def _final_schedule_contract(model: dict[str, int]) -> dict[str, Any]:
    # Layer identity cannot affect analytical traffic, so use the first three.
    expectations = _schedule_expectations(model, [0, 1, 2])
    if (
        expectations["attention_logical_read_bytes"]
        != _EXPECTED_FINAL_LOGICAL_READ_BYTES
        or expectations["attention_logical_read_fraction"]
        != _EXPECTED_FINAL_LOGICAL_READ_FRACTION
    ):
        raise ValueError("layer-rescue analytical traffic contract is invalid")
    return {
        "base_layer_count": _LAYERS - _ROUNDS,
        "rescued_layer_count": _ROUNDS,
        "base_attention_policy": dict(_BASE_POLICY),
        "rescued_attention_policy": dict(_RESCUE_POLICY),
        "attention_expectations_per_sequence": expectations,
        "attention_logical_read_bytes_per_sequence": (
            _EXPECTED_FINAL_LOGICAL_READ_BYTES
        ),
        "attention_logical_read_fraction": (
            _EXPECTED_FINAL_LOGICAL_READ_FRACTION
        ),
    }


def _round_resource_contracts(model: dict[str, int]) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for round_index in range(_ROUNDS):
        rescued_layer_count = round_index + 1
        expectations = _schedule_expectations(
            model,
            list(range(rescued_layer_count)),
        )
        expected = _ROUND_RESOURCE_EXPECTATIONS[round_index]
        if any(
            expectations[name] != value
            for name, value in expected.items()
            if name != "rescued_layer_count"
        ):
            raise ValueError(
                f"layer-rescue round {rescued_layer_count} resource "
                "contract is invalid"
            )
        contracts.append(
            {
                "round": rescued_layer_count,
                "candidate_count": _CANDIDATE_COUNTS[round_index],
                "rescued_layer_count": rescued_layer_count,
                "base_layer_count": _LAYERS - rescued_layer_count,
                "attention_expectations_per_sequence": expectations,
            }
        )
    return contracts


def _population_contract(sequence_count: int) -> dict[str, int]:
    if sequence_count <= 0:
        raise ValueError("layer-rescue population must contain sequences")
    return {
        "overall": sequence_count * _POSITIONS_PER_SEQUENCE,
        **{
            name: sequence_count * (stop - start)
            for name, start, stop in _QUALITY_BANDS
        },
    }


def _q7_traffic_contract(
    model: dict[str, int],
    q7_expectations: dict[str, int],
) -> dict[str, int | float]:
    all_expert_ideal_q4_bytes = _POSITIONS_PER_SEQUENCE * (
        int(model["layers"])
        * int(model["experts"])
        * 3
        * int(model["hidden_size"])
        * int(model["intermediate_size"])
        // 2
    )
    scheduled = int(q7_expectations["scheduled_bytes_per_sequence"])
    return {
        "q7_scheduled_bytes_per_sequence": scheduled,
        "all_expert_ideal_q4_bytes_per_sequence": all_expert_ideal_q4_bytes,
        "q7_fraction_of_all_expert_ideal_q4": (
            scheduled / all_expert_ideal_q4_bytes
        ),
    }


def _bands_from_rows(
    rows: list[dict[str, float | bool | int]],
) -> dict[str, dict[str, Any]]:
    return {
        name: _aggregate(
            [row for row in rows if start <= int(row["position"]) < stop]
        )
        for name, start, stop in _QUALITY_BANDS
    }


def _position_grid_is_exact(
    rows: Any,
    sequence_indices: list[int],
) -> bool:
    if (
        not isinstance(rows, list)
        or len(rows) != len(sequence_indices) * _POSITIONS_PER_SEQUENCE
    ):
        return False
    actual: list[tuple[int, int]] = []
    for row in rows:
        if not isinstance(row, dict):
            return False
        try:
            actual.append((int(row["sequence_index"]), int(row["position"])))
        except (KeyError, TypeError, ValueError):
            return False
    expected = [
        (sequence_index, position)
        for sequence_index in sequence_indices
        for position in range(_POSITIONS_PER_SEQUENCE)
    ]
    return actual == expected


def _normalized_quality_score(
    overall: dict[str, Any],
    bands: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Score a greedy candidate exactly as frozen in the protocol."""

    populations = {"overall": overall, **bands}
    missing = [name for name in _SCORE_POPULATIONS if name not in populations]
    if missing:
        raise ValueError(f"layer-rescue score populations are missing: {missing}")
    details: dict[str, dict[str, float]] = {}
    flat: list[tuple[str, str, float]] = []
    for population_name in _SCORE_POPULATIONS:
        metrics = populations[population_name]
        margins = {
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
        details[population_name] = margins
        if not all(math.isfinite(value) for value in margins.values()):
            raise ValueError("layer-rescue score contains a non-finite margin")
        flat.extend(
            (population_name, metric_name, margins[metric_name])
            for metric_name in _SCORE_METRICS
        )
    worst = min(flat, key=lambda row: (row[2], row[0], row[1]))
    mean_margin = float(np.mean([row[2] for row in flat]))
    return {
        "normalized_margins": details,
        "worst": {
            "population": worst[0],
            "metric": worst[1],
            "normalized_margin": worst[2],
        },
        "worst_normalized_margin": float(worst[2]),
        "mean_normalized_margin": mean_margin,
    }


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, int]:
    score = candidate["selection_score"]
    return (
        -float(score["worst_normalized_margin"]),
        -float(score["mean_normalized_margin"]),
        int(candidate["candidate_layer"]),
    )


def _scoring_rule() -> dict[str, Any]:
    return {
        "source": (
            "causal native outputs versus authenticated dense-teacher final "
            "logits, final normalized hidden states, and targets"
        ),
        "populations": list(_SCORE_POPULATIONS),
        "metrics": list(_SCORE_METRICS),
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
        "candidate_ordering": [
            "descending_worst_normalized_margin",
            "descending_mean_normalized_margin",
            "ascending_layer_id",
        ],
        "round_rule": (
            "evaluate every unselected layer in frozen ascending candidate "
            "order, then append the best layer under candidate_ordering"
        ),
        "band_population_rationale": (
            "positions 0-15 remain in the overall population but are not "
            "scored as a separate band because every W>=16 schedule is "
            "causally invariant there; positions 16-31 and every later band "
            "are scored separately"
        ),
        "rounds": _ROUNDS,
        "candidate_counts": list(_CANDIDATE_COUNTS),
        "early_stop": False,
    }


def _validate_static_sweep_result(
    sweep_protocol: dict[str, Any],
    sweep_result: dict[str, Any],
    *,
    sweep_protocol_hash: str,
    sweep_result_hash: str,
    context: dict[str, Any],
    sweep_source_hash: str,
) -> None:
    """Authenticate the exact, evidence-valid, zero-pass sweep conclusion."""

    del sweep_result_hash
    descriptors = sweep_source._arm_descriptors(context["model"])
    arm_results = sweep_result.get("arm_results")
    artifacts = sweep_result.get("artifacts")
    post_authentication = sweep_result.get("post_run_authentication")
    if (
        sweep_protocol.get("sweep_source_sha256") != sweep_source_hash
        or not isinstance(artifacts, dict)
        or artifacts.get("sweep_protocol_sha256") != sweep_protocol_hash
        or artifacts.get("sweep_source_sha256") != sweep_source_hash
        or any(
            artifacts.get(name) != value
            for name, value in {
                **context["identities"],
                **context["hashes"],
                "control_source_sha256": context["control_source_hash"],
            }.items()
        )
        or sweep_result.get("schema_version") != 1
        or sweep_result.get("experiment") != sweep_source._SWEEP_EXPERIMENT
        or sweep_result.get("status") != "development_sweep_complete"
        or sweep_result.get("evidence_passed") is not True
        or sweep_result.get("diagnostic_quality_passing_arm_count") != 0
        or sweep_result.get("selected_arm") is not None
        or sweep_result.get("ranking") != []
        or sweep_result.get("selection_is_development_only") is not False
        or sweep_result.get("fresh_confirmation_required") is not False
        or sweep_result.get("decision")
        != "investigate_layer_adaptive_or_learned_selector"
        or sweep_result.get("thresholds") != _THRESHOLDS
        or sweep_result.get("quality_bands") != sweep_source._expected_bands()
        or sweep_result.get("ranking_rule") != sweep_source._ranking_rule()
        or not isinstance(post_authentication, dict)
        or not post_authentication
        or not all(value is True for value in post_authentication.values())
        or not isinstance(arm_results, list)
        or len(arm_results) != len(descriptors)
    ):
        raise ValueError("layer-rescue static sweep result prerequisite is invalid")

    for descriptor, result in zip(descriptors, arm_results, strict=True):
        rows = result.get("position_results")
        if not sweep_source._position_grid_is_exact(rows):
            raise ValueError("layer-rescue static sweep position grid is invalid")
        overall = _aggregate(rows)
        bands = _bands_from_rows(rows)
        quality_checks = _quality_checks("overall", overall)
        for name, metrics in bands.items():
            quality_checks.update(_quality_checks(name, metrics))
        evidence_checks = result.get("evidence_checks")
        if (
            result.get("ordinal") != descriptor["ordinal"]
            or result.get("name") != descriptor["name"]
            or result.get("attention_policy") != descriptor["attention_policy"]
            or result.get("attention_expectations_per_sequence")
            != descriptor["attention_expectations_per_sequence"]
            or result.get("q7_expectations_per_sequence")
            != context["q7_expectations"]
            or result.get("metrics") != overall
            or result.get("position_bands") != bands
            or result.get("quality_checks") != quality_checks
            or result.get("quality_margin")
            != sweep_source._quality_margin(overall, bands)
            or result.get("quality_passed") is not False
            or all(quality_checks.values())
            or not isinstance(evidence_checks, dict)
            or not evidence_checks
            or not all(value is True for value in evidence_checks.values())
            or result.get("evidence_passed") is not True
            or result.get("reset_replay", {}).get("passed") is not True
        ):
            raise ValueError(
                "layer-rescue static sweep arm prerequisite is invalid"
            )


def _historical_source_inventory(
    sustained_protocol: dict[str, Any],
) -> dict[str, str]:
    """Validate descriptors without pretending the historical tree is current."""

    inventory = sustained_protocol.get("evaluator_source_sha256")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("layer-rescue historical source inventory is invalid")
    for relative, digest in inventory.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "layer-rescue historical source descriptor is invalid"
            )
    return dict(inventory)


def _rescue_source_inventory(
    historical_inventory: dict[str, str],
) -> dict[str, str]:
    """Hash the complete current runtime/evaluator source boundary."""

    repository = Path(__file__).resolve().parents[3]
    relative_paths = set(historical_inventory)
    relative_paths.update(
        {
            "native/include/engram/olmoe_token_runtime.h",
            "native/include/engram/olmoe_token_runtime_c.h",
            "native/src/olmoe_token_runtime.cpp",
            "native/src/olmoe_token_runtime_c.cpp",
            "src/engram/runtime/olmoe_native.py",
            "src/engram/evaluation/olmoe_native_dense_control.py",
            "src/engram/evaluation/olmoe_native_attention_sweep.py",
            "src/engram/evaluation/olmoe_native_layer_rescue.py",
        }
    )
    inventory: dict[str, str] = {}
    for relative in sorted(relative_paths):
        source = repository / relative
        if not source.is_file():
            raise ValueError(
                f"layer-rescue current source is missing: {relative}"
            )
        inventory[relative] = sha256_file(source)
    return inventory


def _authenticate_prerequisites(
    *,
    package_path: Path,
    manifest_sha256: str,
    reference_library_path: Path,
    candidate_library_path: Path,
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
    sweep_protocol_path: Path,
    sweep_protocol_sha256: str,
    sweep_result_path: Path,
    sweep_result_sha256: str,
) -> dict[str, Any]:
    sustained_protocol = sweep_source._read_object(
        sustained_protocol_path,
        "sustained protocol",
    )
    sustained_result = sweep_source._read_object(
        sustained_result_path,
        "sustained result",
    )
    control_protocol = sweep_source._read_object(
        control_protocol_path,
        "dense-attention control protocol",
    )
    control_result = sweep_source._read_object(
        control_result_path,
        "dense-attention control result",
    )
    reference = sweep_source._read_object(
        reference_path,
        "sustained teacher reference",
    )
    hashes = {
        "sustained_protocol_sha256": sha256_file(sustained_protocol_path),
        "sustained_result_sha256": sha256_file(sustained_result_path),
        "control_protocol_sha256": sha256_file(control_protocol_path),
        "control_result_sha256": sha256_file(control_result_path),
    }
    if (
        hashes["sustained_protocol_sha256"]
        != sustained_protocol_sha256.lower()
        or hashes["sustained_result_sha256"]
        != sustained_result_sha256.lower()
        or hashes["control_protocol_sha256"]
        != control_protocol_sha256.lower()
        or hashes["control_result_sha256"]
        != control_result_sha256.lower()
    ):
        raise ValueError("layer-rescue prerequisite hash is invalid")
    identities = {
        "package_manifest_sha256": manifest_sha256.lower(),
        "native_library_sha256": sha256_file(reference_library_path),
        "dataset_sha256": sha256_file(dataset_path),
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "teacher_reference_sha256": sha256_file(reference_path),
        "teacher_arrays_sha256": sha256_file(arrays_path),
    }
    if (
        sustained_protocol.get("schema_version") != 1
        or sustained_protocol.get("experiment")
        != "olmoe_native_sustained_context_confirmation"
        or sustained_protocol.get("status")
        != "frozen_before_candidate_execution"
        or sustained_protocol.get("package_manifest_sha256")
        != identities["package_manifest_sha256"]
        or any(
            sustained_protocol.get(name) != value
            for name, value in identities.items()
            if name != "package_manifest_sha256"
        )
        or sustained_protocol.get("sequences") != _SEQUENCES
        or sustained_protocol.get("tokens_per_sequence")
        != _POSITIONS_PER_SEQUENCE + 1
        or sustained_protocol.get("quality_bands")
        != sweep_source._expected_bands()
        or sustained_protocol.get("thresholds") != _THRESHOLDS
        or sustained_protocol.get("attention_policy") != _BASE_POLICY
        or sustained_protocol.get("scope", {}).get("candidate_threads")
        != _THREADS
        or sustained_protocol.get("scope", {}).get("teacher_configuration")
        != _TEACHER_CONFIGURATION
    ):
        raise ValueError("layer-rescue sustained prerequisite is invalid")
    historical_sources = _historical_source_inventory(sustained_protocol)
    sweep_source._validate_teacher_source(reference, sustained_protocol)
    if (
        sweep_source._teacher_configuration(reference)
        != _TEACHER_CONFIGURATION
        or reference.get("dataset", {}).get("input_ids")
        != sustained_protocol.get("input_ids")
        or reference.get("dataset", {}).get("input_identity")
        != sustained_protocol.get("input_identity")
        or reference.get("dataset", {}).get("prediction_positions")
        != _SEQUENCES * _POSITIONS_PER_SEQUENCE
    ):
        raise ValueError("layer-rescue teacher prerequisite is invalid")
    control_source_path = Path(
        sweep_source.dense_control_source.__file__
    ).resolve()
    control_source_hash = sha256_file(control_source_path)
    sweep_source._validate_control_prerequisite(
        sustained_protocol,
        sustained_result,
        control_protocol,
        control_result,
        sustained_protocol_hash=hashes["sustained_protocol_sha256"],
        sustained_result_hash=hashes["sustained_result_sha256"],
        control_protocol_hash=hashes["control_protocol_sha256"],
        control_result_hash=hashes["control_result_sha256"],
        identities=identities,
        evaluator_sources=historical_sources,
        control_source_hash=control_source_hash,
    )
    input_ids = sustained_protocol["input_ids"]
    if sha256_json(input_ids) != sustained_protocol["input_identity"]:
        raise ValueError("layer-rescue input identity is invalid")
    manifest = validate_olmoe_native_package(
        package_path,
        expected_manifest_sha256=manifest_sha256,
    )
    config_path, non_mlp_path, q7_path, tokenizer_path = (
        sweep_source._paths_from_manifest(package_path, manifest)
    )
    if (
        manifest.get("runtime", {}).get("kernel_threads") != _THREADS
        or manifest.get("runtime", {}).get("attention_policy")
        != sustained_protocol["attention_policy"]
        or manifest.get("source", {}).get("revision")
        != sustained_protocol["source_revision"]
        or manifest.get("files", {})
        .get("model/config.json", {})
        .get("sha256")
        != sustained_protocol["source_config_sha256"]
    ):
        raise ValueError("layer-rescue package prerequisite is invalid")
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for the layer rescue"
        ) from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    retokenized, _corpus = sweep_source._validate_corpus_manifest(
        dataset_path,
        corpus_manifest_path,
        tokenizer,
        tokenizer_sha256=manifest["files"]["tokenizer/tokenizer.json"]["sha256"],
    )
    if retokenized != input_ids:
        raise ValueError("layer-rescue retokenization differs")
    model = sustained_protocol["model"]
    q7_expectations = sweep_source._q7_expectations(model)
    if (
        q7_expectations != sustained_protocol["q7_expectations_per_sequence"]
        or q7_expectations != control_protocol["q7_expectations_per_sequence"]
        or q7_expectations != control_result["q7_expectations_per_sequence"]
    ):
        raise ValueError("layer-rescue Q7 prerequisite is invalid")
    context = {
        "sustained_protocol": sustained_protocol,
        "sustained_result": sustained_result,
        "control_protocol": control_protocol,
        "control_result": control_result,
        "reference": reference,
        "hashes": hashes,
        "identities": identities,
        "historical_evaluator_sources": historical_sources,
        # Compatibility input for validation of the immutable sweep protocol.
        # It is intentionally not authenticated against the now-modified tree.
        "evaluator_sources": historical_sources,
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
    sweep_protocol = sweep_source._read_object(
        sweep_protocol_path,
        "static attention sweep protocol",
    )
    sweep_result = sweep_source._read_object(
        sweep_result_path,
        "static attention sweep result",
    )
    sweep_hashes = {
        "sweep_protocol_sha256": sha256_file(sweep_protocol_path),
        "sweep_result_sha256": sha256_file(sweep_result_path),
    }
    if (
        sweep_hashes["sweep_protocol_sha256"]
        != sweep_protocol_sha256.lower()
        or sweep_hashes["sweep_result_sha256"] != sweep_result_sha256.lower()
    ):
        raise ValueError("layer-rescue static sweep hash is invalid")
    sweep_source_path = Path(sweep_source.__file__).resolve()
    sweep_source_hash = sha256_file(sweep_source_path)
    sweep_source._validate_sweep_protocol(
        sweep_protocol,
        context,
        sweep_protocol_hash=sweep_hashes["sweep_protocol_sha256"],
        supplied_sweep_protocol_hash=sweep_protocol_sha256,
        sweep_source_hash=sweep_source_hash,
    )
    _validate_static_sweep_result(
        sweep_protocol,
        sweep_result,
        sweep_protocol_hash=sweep_hashes["sweep_protocol_sha256"],
        sweep_result_hash=sweep_hashes["sweep_result_sha256"],
        context=context,
        sweep_source_hash=sweep_source_hash,
    )
    if not _post_source_shards(
        context["reference"],
        context["sustained_protocol"]["source_shard_sha256"],
    ):
        raise ValueError("layer-rescue teacher source shards changed")
    record_ids = _dataset_record_ids(dataset_path)
    split = _record_split(record_ids)
    if int(context["model"]["layers"]) != _LAYERS:
        raise ValueError("layer rescue requires the 16-layer OLMoE source")
    final_contract = _final_schedule_contract(context["model"])
    rescue_sources = _rescue_source_inventory(historical_sources)
    context.update(
        {
            "sweep_protocol": sweep_protocol,
            "sweep_result": sweep_result,
            "sweep_hashes": sweep_hashes,
            "sweep_source_path": sweep_source_path,
            "sweep_source_hash": sweep_source_hash,
            "record_ids": record_ids,
            "split": split,
            "final_schedule_contract": final_contract,
            "candidate_library_sha256": sha256_file(candidate_library_path),
            "rescue_source_inventory": rescue_sources,
        }
    )
    return context


def _build_protocol(
    context: dict[str, Any],
    *,
    rescue_source_hash: str,
) -> dict[str, Any]:
    sustained = context["sustained_protocol"]
    return {
        "schema_version": 1,
        "experiment": _EXPERIMENT,
        "status": _PROTOCOL_STATUS,
        "source_revision": sustained["source_revision"],
        **context["identities"],
        **context["hashes"],
        **context["sweep_hashes"],
        "control_source_sha256": context["control_source_hash"],
        "sweep_source_sha256": context["sweep_source_hash"],
        "rescue_source_sha256": rescue_source_hash,
        "historical_frozen_evaluator_source_sha256": context[
            "historical_evaluator_sources"
        ],
        "rescue_source_inventory_sha256": context[
            "rescue_source_inventory"
        ],
        "candidate_native_library_sha256": context[
            "candidate_library_sha256"
        ],
        "source_config_sha256": sustained["source_config_sha256"],
        "source_index_sha256": sustained["source_index_sha256"],
        "source_shard_sha256": sustained["source_shard_sha256"],
        "input_identity": sustained["input_identity"],
        "input_ids": context["input_ids"],
        "dataset_record_ids": context["record_ids"],
        "record_split": context["split"],
        "model": context["model"],
        "q7_expectations_per_sequence": context["q7_expectations"],
        "base_attention_policy": dict(_BASE_POLICY),
        "rescue_attention_policy": dict(_RESCUE_POLICY),
        "final_schedule_contract": context["final_schedule_contract"],
        "next_rescue_budget_boundary": {
            "rescued_layer_count": 4,
            "attention_logical_read_bytes_per_sequence": (
                _FOUR_RESCUE_LOGICAL_READ_BYTES
            ),
            "attention_logical_read_fraction": (
                _FOUR_RESCUE_LOGICAL_READ_FRACTION
            ),
            "maximum_attention_logical_read_fraction": _THRESHOLDS[
                "maximum_attention_logical_read_fraction"
            ],
            "within_budget": False,
            "reason_exactly_three_layers": (
                "three W128 layers are the largest integer rescue count under "
                "the inherited 45% logical attention-read cap"
            ),
        },
        "candidate_layer_order": _candidate_layer_order(),
        "greedy_rounds": _ROUNDS,
        "candidate_counts": list(_CANDIDATE_COUNTS),
        "expected_candidate_evaluations": _EXPECTED_CANDIDATE_EVALUATIONS,
        "round_resource_contracts": _round_resource_contracts(
            context["model"]
        ),
        "scoring_rule": _scoring_rule(),
        "quality_bands": sweep_source._expected_bands(),
        "thresholds": _THRESHOLDS,
        "population_contracts": {
            "selection": _population_contract(_SELECTION_SEQUENCES),
            "internal_holdout": _population_contract(_HOLDOUT_SEQUENCES),
        },
        "scope": {
            "candidate_device": "cpu",
            "candidate_threads": _THREADS,
            "candidate_transformers_model_shell": False,
            "execution_interface": _EXECUTION_INTERFACE,
            "source_package_attention_policy": dict(_BASE_POLICY),
            "per_layer_attention_policy_overridden_for_development": True,
            "package_manifest_mutated": False,
            "q7_artifact_or_policy_changed": False,
            "corpus_or_teacher_changed": False,
            "selection_sequences": _SELECTION_SEQUENCES,
            "internal_holdout_sequences": _HOLDOUT_SEQUENCES,
            "internal_holdout_outputs_unseen_during_greedy_selection": True,
            "candidate_results_inspected_only_after_each_full_round": True,
            "candidate_order_or_scoring_adapted_after_freeze": False,
            "all_candidates_execute_in_each_round": True,
            "early_stop": False,
            "final_base_layers": _LAYERS - _ROUNDS,
            "final_rescued_layers": _ROUNDS,
            "layered_abi_all_base_parity_sequence": context["split"][
                "selection"
            ][0]["sequence_index"],
            "layered_abi_all_base_exact_parity_required": True,
            "holdout_primary_schedule_executions": 1,
            "holdout_reset_replay_excluded_from_semantic_metrics": True,
            "development_selection_only": True,
            "fresh_confirmation_required_after_holdout_pass": True,
            "internal_holdout_is_not_sustained_gate_confirmation": True,
            "protocol_frozen_before_any_candidate_execution": True,
            "static_sweep_zero_pass_result_known_before_protocol_freeze": True,
            "teacher_configuration": _TEACHER_CONFIGURATION,
        },
        "decision_rule": {
            "layered_abi_parity_failure": (
                "write an invalid no-candidate result and stop before search"
            ),
            "evidence_failure": "stop_and_diagnose_layer_rescue_evidence",
            "holdout_quality_pass": (
                "integrate selected per-layer schedule into package tooling, "
                "then freeze a fresh package-native confirmation"
            ),
            "authenticated_holdout_quality_failure": (
                "investigate head-wise teacher-guided attention allocation"
            ),
        },
        "limitations": [
            "The two-sequence selection split and six-sequence internal holdout are drawn from a corpus already consumed by earlier diagnostics.",
            "The greedy three-layer schedule is development-only even if the internal holdout passes.",
            "A passing schedule requires package integration and a separately frozen fresh confirmation.",
            "The six-record internal screen cannot meet the inherited eight-sequence sustained gate population requirement.",
            "Logical attention bytes are analytical native reads, not measured hardware DRAM traffic.",
            "The greedy search can miss interacting layer combinations.",
            "W128 is full-context only over this 128-position horizon; beyond it, the rescued layer remains bounded W128/C8/K4/S2 rather than dense.",
            "Forty-five adaptive comparisons on two independent sequences are high-variance and overfit-prone because positions within a sequence are correlated; the six-record screen and a fresh sealed eight-record confirmation are mandatory.",
        ],
    }


def freeze_native_olmoe_layer_rescue_protocol(
    *,
    package: str | Path,
    manifest_sha256: str,
    reference_library: str | Path,
    candidate_library: str | Path,
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
    sweep_result: str | Path,
    sweep_result_sha256: str,
    out: str | Path,
    threads: int = _THREADS,
) -> dict[str, Any]:
    """Freeze the split, search algorithm, traffic, and decision rule."""

    output_path = Path(out).expanduser().resolve()
    if output_path.exists():
        raise ValueError("layer-rescue protocol target already exists")
    if threads != _THREADS:
        raise ValueError("layer rescue requires the matched 12 threads")
    paths = _resolve_paths(
        package=package,
        reference_library=reference_library,
        candidate_library=candidate_library,
        dataset=dataset,
        corpus_manifest=corpus_manifest,
        teacher_reference=teacher_reference,
        teacher_arrays=teacher_arrays,
        sustained_protocol=sustained_protocol,
        sustained_result=sustained_result,
        control_protocol=control_protocol,
        control_result=control_result,
        sweep_protocol=sweep_protocol,
        sweep_result=sweep_result,
    )
    context = _authenticate_prerequisites(
        **paths,
        manifest_sha256=manifest_sha256,
        sustained_protocol_sha256=sustained_protocol_sha256,
        sustained_result_sha256=sustained_result_sha256,
        control_protocol_sha256=control_protocol_sha256,
        control_result_sha256=control_result_sha256,
        sweep_protocol_sha256=sweep_protocol_sha256,
        sweep_result_sha256=sweep_result_sha256,
    )
    rescue_source_hash = sha256_file(Path(__file__).resolve())
    protocol = _build_protocol(
        context,
        rescue_source_hash=rescue_source_hash,
    )
    atomic_json(output_path, protocol)
    return protocol


def _validate_protocol(
    protocol: dict[str, Any],
    context: dict[str, Any],
    *,
    protocol_hash: str,
    supplied_protocol_hash: str,
    rescue_source_hash: str,
) -> None:
    expected = _build_protocol(
        context,
        rescue_source_hash=rescue_source_hash,
    )
    if (
        protocol_hash != supplied_protocol_hash.lower()
        or protocol != expected
    ):
        raise ValueError("layer-rescue protocol contract is invalid")


def _resolve_paths(
    *,
    package: str | Path,
    reference_library: str | Path,
    candidate_library: str | Path,
    dataset: str | Path,
    corpus_manifest: str | Path,
    teacher_reference: str | Path,
    teacher_arrays: str | Path,
    sustained_protocol: str | Path,
    sustained_result: str | Path,
    control_protocol: str | Path,
    control_result: str | Path,
    sweep_protocol: str | Path,
    sweep_result: str | Path,
) -> dict[str, Path]:
    return {
        "package_path": Path(package).expanduser().resolve(),
        "reference_library_path": Path(reference_library)
        .expanduser()
        .resolve(),
        "candidate_library_path": Path(candidate_library)
        .expanduser()
        .resolve(),
        "dataset_path": Path(dataset).expanduser().resolve(),
        "corpus_manifest_path": Path(corpus_manifest).expanduser().resolve(),
        "reference_path": Path(teacher_reference).expanduser().resolve(),
        "arrays_path": Path(teacher_arrays).expanduser().resolve(),
        "sustained_protocol_path": Path(sustained_protocol).expanduser().resolve(),
        "sustained_result_path": Path(sustained_result).expanduser().resolve(),
        "control_protocol_path": Path(control_protocol).expanduser().resolve(),
        "control_result_path": Path(control_result).expanduser().resolve(),
        "sweep_protocol_path": Path(sweep_protocol).expanduser().resolve(),
        "sweep_result_path": Path(sweep_result).expanduser().resolve(),
    }


def _counter_checks(
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
    checks["q7_scheduled_bytes"] = (
        metrics.get("q7_scheduled_bytes")
        == position * q7_expectations["scheduled_bytes_per_position"]
    )
    return checks


def _update_counter_digest(
    digest: Any,
    metrics: dict[str, int],
) -> None:
    digest.update(
        json.dumps(
            _deterministic_metrics(metrics),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")


def _evaluate_schedule(
    rescued_layers: list[int],
    *,
    sequence_indices: list[int],
    split_name: str,
    context: dict[str, Any],
    library_path: Path,
    teacher_logits: np.ndarray,
    teacher_hidden: np.ndarray,
    targets: np.ndarray,
    threads: int,
    replay_sequence_index: int | None = None,
) -> dict[str, Any]:
    """Causally evaluate one immutable per-layer schedule."""

    policies = _schedule_policies(rescued_layers)
    expectations = _schedule_expectations(context["model"], rescued_layers)
    q7_expectations = context["q7_expectations"]
    per_position_expectations = [
        _schedule_expectations(
            context["model"],
            rescued_layers,
            positions=position,
        )
        for position in range(1, _POSITIONS_PER_SEQUENCE + 1)
    ]
    input_ids = context["input_ids"]
    record_ids = context["record_ids"]
    all_rows: list[dict[str, float | bool | int]] = []
    sequence_results: list[dict[str, Any]] = []
    replay_reference: dict[str, Any] | None = None
    load_started = time.perf_counter()
    runtime = OLMoENativeTokenRuntime(
        context["config_path"],
        context["non_mlp_path"],
        context["q7_path"],
        library_path,
        threads=threads,
        attention_policies=policies,
    )
    cold_load_seconds = time.perf_counter() - load_started
    try:
        if not runtime.attention_metrics_available:
            raise ValueError("layer-rescue attention metric ABI is unavailable")
        for sequence_index in sequence_indices:
            runtime.reset()
            rows: list[dict[str, float | bool | int]] = []
            top1_tokens: list[int] = []
            hidden_digest = hashlib.sha256()
            logit_digest = hashlib.sha256()
            counter_digest = hashlib.sha256()
            final_metrics: dict[str, int] | None = None
            counter_checks_passed = True
            started = time.perf_counter()
            for position, token_id in enumerate(input_ids[sequence_index][:-1]):
                native_result = runtime.forward([token_id])
                native_hidden, native_logits = runtime.last_diagnostics()
                if int(np.argmax(native_logits)) != native_result.next_token:
                    raise ValueError("layer-rescue diagnostic argmax differs")
                final_metrics = dict(native_result.metrics)
                checks = _counter_checks(
                    final_metrics,
                    per_position_expectations[position],
                    q7_expectations,
                    position=runtime.position,
                )
                counter_checks_passed = (
                    counter_checks_passed and all(checks.values())
                )
                _update_counter_digest(counter_digest, final_metrics)
                _update_diagnostic_hashes(
                    hidden_digest,
                    logit_digest,
                    native_hidden,
                    native_logits,
                )
                teacher_offset = (
                    sequence_index * _POSITIONS_PER_SEQUENCE + position
                )
                row = _position_metrics(
                    teacher_logits[teacher_offset],
                    native_logits,
                    teacher_hidden[teacher_offset],
                    native_hidden,
                    int(targets[teacher_offset]),
                )
                row.update(
                    {
                        "sequence_index": sequence_index,
                        "position": position,
                    }
                )
                rows.append(row)
                top1_tokens.append(native_result.next_token)
            elapsed_seconds = time.perf_counter() - started
            if final_metrics is None:
                raise RuntimeError("layer-rescue schedule processed no positions")
            final_checks = _counter_checks(
                final_metrics,
                expectations,
                q7_expectations,
                position=runtime.position,
            )
            sequence_result = {
                "sequence_index": sequence_index,
                "record_id": record_ids[sequence_index],
                "prediction_positions": len(rows),
                "metrics": _aggregate(rows),
                "final_native_metrics": _deterministic_metrics(final_metrics),
                "final_counter_checks": final_checks,
                "per_token_counter_checks_passed": counter_checks_passed,
                "counter_stream_sha256": counter_digest.hexdigest(),
                "top1_sha256": sha256_json(top1_tokens),
                "hidden_sha256": hidden_digest.hexdigest(),
                "logits_sha256": logit_digest.hexdigest(),
                "elapsed_seconds": elapsed_seconds,
            }
            sequence_results.append(sequence_result)
            all_rows.extend(rows)
            if sequence_index == replay_sequence_index:
                replay_reference = {
                    "sequence_index": sequence_index,
                    "top1_tokens": top1_tokens,
                    "top1_sha256": sequence_result["top1_sha256"],
                    "hidden_sha256": sequence_result["hidden_sha256"],
                    "logits_sha256": sequence_result["logits_sha256"],
                    "counter_stream_sha256": sequence_result[
                        "counter_stream_sha256"
                    ],
                    "final_native_metrics": sequence_result[
                        "final_native_metrics"
                    ],
                }

        reset_replay: dict[str, Any] | None = None
        if replay_sequence_index is not None:
            if (
                replay_sequence_index not in sequence_indices
                or replay_reference is None
            ):
                raise ValueError("layer-rescue replay sequence is not in population")
            runtime.reset()
            replay_top1: list[int] = []
            replay_hidden_digest = hashlib.sha256()
            replay_logit_digest = hashlib.sha256()
            replay_counter_digest = hashlib.sha256()
            replay_metrics: dict[str, int] | None = None
            replay_counter_checks_passed = True
            replay_started = time.perf_counter()
            for position, token_id in enumerate(
                input_ids[replay_sequence_index][:-1]
            ):
                native_result = runtime.forward([token_id])
                native_hidden, native_logits = runtime.last_diagnostics()
                replay_metrics = dict(native_result.metrics)
                replay_counter_checks_passed = (
                    replay_counter_checks_passed
                    and all(
                        _counter_checks(
                            replay_metrics,
                            per_position_expectations[position],
                            q7_expectations,
                            position=runtime.position,
                        ).values()
                    )
                )
                replay_top1.append(native_result.next_token)
                _update_diagnostic_hashes(
                    replay_hidden_digest,
                    replay_logit_digest,
                    native_hidden,
                    native_logits,
                )
                _update_counter_digest(replay_counter_digest, replay_metrics)
            replay_seconds = time.perf_counter() - replay_started
            if replay_metrics is None:
                raise RuntimeError("layer-rescue replay processed no positions")
            replay_value = {
                "sequence_index": replay_sequence_index,
                "top1_tokens": replay_top1,
                "top1_sha256": sha256_json(replay_top1),
                "hidden_sha256": replay_hidden_digest.hexdigest(),
                "logits_sha256": replay_logit_digest.hexdigest(),
                "counter_stream_sha256": replay_counter_digest.hexdigest(),
                "final_native_metrics": _deterministic_metrics(replay_metrics),
            }
            reset_replay = {
                "sequence_index": replay_sequence_index,
                "reference": replay_reference,
                "replay": replay_value,
                "per_token_counter_checks_passed": (
                    replay_counter_checks_passed
                ),
                "passed": (
                    replay_value == replay_reference
                    and replay_counter_checks_passed
                ),
                "elapsed_seconds": replay_seconds,
            }
    finally:
        runtime.close()

    aggregate = _aggregate(all_rows)
    bands = _bands_from_rows(all_rows)
    quality_checks = _quality_checks("overall", aggregate)
    for name, metrics in bands.items():
        quality_checks.update(_quality_checks(name, metrics))
    population_contract = _population_contract(len(sequence_indices))
    actual_populations = {
        "overall": int(aggregate["prediction_positions"]),
        **{
            name: int(metrics["prediction_positions"])
            for name, metrics in bands.items()
        },
    }
    evidence_checks = {
        "sequence_count": len(sequence_results) == len(sequence_indices),
        "prediction_grid": _position_grid_is_exact(
            all_rows,
            sequence_indices,
        ),
        "population_sizes": actual_populations == population_contract,
        "final_counter_checks": all(
            all(row["final_counter_checks"].values())
            for row in sequence_results
        ),
        "per_token_counter_checks": all(
            row["per_token_counter_checks_passed"]
            for row in sequence_results
        ),
        "q7_policy_unchanged": all(
            row["final_native_metrics"].get("q7_scheduled_bytes")
            == q7_expectations["scheduled_bytes_per_sequence"]
            for row in sequence_results
        ),
        "q7_traffic_fraction": (
            _q7_traffic_contract(
                context["model"],
                q7_expectations,
            )["q7_fraction_of_all_expert_ideal_q4"]
            <= _THRESHOLDS["maximum_q7_traffic_fraction"]
        ),
        "attention_logical_read_fraction": (
            expectations["attention_logical_read_fraction"]
            <= _THRESHOLDS["maximum_attention_logical_read_fraction"]
        ),
        "reset_replay": (
            reset_replay is None or reset_replay["passed"] is True
        ),
    }
    return {
        "split": split_name,
        "sequence_indices": list(sequence_indices),
        "record_ids": [record_ids[index] for index in sequence_indices],
        "rescued_layers": list(rescued_layers),
        "attention_policies": policies,
        "attention_expectations_per_sequence": expectations,
        "q7_expectations_per_sequence": q7_expectations,
        "q7_traffic_contract_per_sequence": _q7_traffic_contract(
            context["model"],
            q7_expectations,
        ),
        "metrics": aggregate,
        "position_bands": bands,
        "selection_score": _normalized_quality_score(aggregate, bands),
        "quality_checks": quality_checks,
        "quality_passed": all(quality_checks.values()),
        "population_contract": population_contract,
        "evidence_checks": evidence_checks,
        "evidence_passed": all(evidence_checks.values()),
        "sequence_results": sequence_results,
        "position_results": all_rows,
        "reset_replay": reset_replay,
        "performance": {
            "cold_load_seconds": cold_load_seconds,
            "primary_sequence_seconds": sum(
                row["elapsed_seconds"] for row in sequence_results
            ),
            "reset_replay_seconds": (
                0.0
                if reset_replay is None
                else reset_replay["elapsed_seconds"]
            ),
        },
    }


def _evaluate_layered_abi_parity(
    *,
    context: dict[str, Any],
    reference_library_path: Path,
    candidate_library_path: Path,
    sequence_index: int,
    threads: int,
) -> dict[str, Any]:
    """Prove the additive layered ABI is inert under 16 identical base policies."""

    expected_by_position = [
        _schedule_expectations(context["model"], [], positions=position)
        for position in range(1, _POSITIONS_PER_SEQUENCE + 1)
    ]
    reference = OLMoENativeTokenRuntime(
        context["config_path"],
        context["non_mlp_path"],
        context["q7_path"],
        reference_library_path,
        threads=threads,
        **_BASE_POLICY,
    )
    candidate = OLMoENativeTokenRuntime(
        context["config_path"],
        context["non_mlp_path"],
        context["q7_path"],
        candidate_library_path,
        threads=threads,
        attention_policies=_schedule_policies([]),
    )
    reference_hidden_digest = hashlib.sha256()
    reference_logit_digest = hashlib.sha256()
    candidate_hidden_digest = hashlib.sha256()
    candidate_logit_digest = hashlib.sha256()
    reference_counter_digest = hashlib.sha256()
    candidate_counter_digest = hashlib.sha256()
    token_matches: list[bool] = []
    hidden_matches: list[bool] = []
    logit_matches: list[bool] = []
    counter_matches: list[bool] = []
    reference_counter_checks: list[bool] = []
    candidate_counter_checks: list[bool] = []
    reference_final: dict[str, int] | None = None
    candidate_final: dict[str, int] | None = None
    reference_final_position = 0
    candidate_final_position = 0
    started = time.perf_counter()
    try:
        if (
            not reference.attention_metrics_available
            or not candidate.attention_metrics_available
        ):
            raise ValueError("layer-rescue parity metric ABI is unavailable")
        for position, token_id in enumerate(
            context["input_ids"][sequence_index][:-1]
        ):
            reference_result = reference.forward([token_id])
            candidate_result = candidate.forward([token_id])
            reference_hidden, reference_logits = reference.last_diagnostics()
            candidate_hidden, candidate_logits = candidate.last_diagnostics()
            reference_final = dict(reference_result.metrics)
            candidate_final = dict(candidate_result.metrics)
            expected = expected_by_position[position]
            token_matches.append(
                reference_result.next_token == candidate_result.next_token
            )
            hidden_matches.append(
                np.array_equal(reference_hidden, candidate_hidden)
            )
            logit_matches.append(
                np.array_equal(reference_logits, candidate_logits)
            )
            counter_matches.append(
                _deterministic_metrics(reference_final)
                == _deterministic_metrics(candidate_final)
            )
            reference_counter_checks.append(
                all(
                    _counter_checks(
                        reference_final,
                        expected,
                        context["q7_expectations"],
                        position=reference.position,
                    ).values()
                )
            )
            candidate_counter_checks.append(
                all(
                    _counter_checks(
                        candidate_final,
                        expected,
                        context["q7_expectations"],
                        position=candidate.position,
                    ).values()
                )
            )
            _update_diagnostic_hashes(
                reference_hidden_digest,
                reference_logit_digest,
                reference_hidden,
                reference_logits,
            )
            _update_diagnostic_hashes(
                candidate_hidden_digest,
                candidate_logit_digest,
                candidate_hidden,
                candidate_logits,
            )
            _update_counter_digest(reference_counter_digest, reference_final)
            _update_counter_digest(candidate_counter_digest, candidate_final)
        reference_final_position = reference.position
        candidate_final_position = candidate.position
    finally:
        reference.close()
        candidate.close()
    elapsed_seconds = time.perf_counter() - started
    if reference_final is None or candidate_final is None:
        raise RuntimeError("layer-rescue parity processed no positions")

    historical_sequences = context["sustained_result"].get("sequence_results")
    historical = (
        next(
            (
                row
                for row in historical_sequences
                if int(row.get("sequence", -1)) == sequence_index
            ),
            None,
        )
        if isinstance(historical_sequences, list)
        else None
    )
    reference_hashes = {
        "hidden_sha256": reference_hidden_digest.hexdigest(),
        "logits_sha256": reference_logit_digest.hexdigest(),
        "counter_stream_sha256": reference_counter_digest.hexdigest(),
    }
    candidate_hashes = {
        "hidden_sha256": candidate_hidden_digest.hexdigest(),
        "logits_sha256": candidate_logit_digest.hexdigest(),
        "counter_stream_sha256": candidate_counter_digest.hexdigest(),
    }
    historical_metrics = (
        historical.get("native_metrics") if isinstance(historical, dict) else None
    )
    checks = {
        "prediction_positions": len(token_matches) == _POSITIONS_PER_SEQUENCE,
        "cache_positions": (
            reference_final_position == _POSITIONS_PER_SEQUENCE
            and candidate_final_position == _POSITIONS_PER_SEQUENCE
        ),
        "tokens_exact": all(token_matches),
        "hidden_exact": all(hidden_matches),
        "logits_exact": all(logit_matches),
        "deterministic_counters_exact": all(counter_matches),
        "reference_counter_contract": all(reference_counter_checks),
        "candidate_counter_contract": all(candidate_counter_checks),
        "diagnostic_hashes_exact": reference_hashes == candidate_hashes,
        "historical_reference_present": isinstance(historical, dict),
        "reference_matches_historical_diagnostics": (
            isinstance(historical, dict)
            and historical.get("diagnostic_hashes")
            == {
                "hidden_sha256": reference_hashes["hidden_sha256"],
                "logits_sha256": reference_hashes["logits_sha256"],
            }
        ),
        "reference_matches_historical_counters": (
            isinstance(historical_metrics, dict)
            and _deterministic_metrics(historical_metrics)
            == _deterministic_metrics(reference_final)
        ),
    }
    return {
        "sequence_index": sequence_index,
        "record_id": context["record_ids"][sequence_index],
        "reference_library_sha256": context["identities"][
            "native_library_sha256"
        ],
        "candidate_library_sha256": context["candidate_library_sha256"],
        "scalar_reference_policy": dict(_BASE_POLICY),
        "layered_candidate_policies": _schedule_policies([]),
        "reference_hashes": reference_hashes,
        "candidate_hashes": candidate_hashes,
        "reference_final_metrics": _deterministic_metrics(reference_final),
        "candidate_final_metrics": _deterministic_metrics(candidate_final),
        "checks": checks,
        "passed": all(checks.values()),
        "elapsed_seconds": elapsed_seconds,
    }


def _post_authentication(
    *,
    context: dict[str, Any],
    package_path: Path,
    manifest_sha256: str,
    reference_library_path: Path,
    candidate_library_path: Path,
    dataset_path: Path,
    corpus_manifest_path: Path,
    reference_path: Path,
    arrays_path: Path,
    sustained_protocol_path: Path,
    sustained_result_path: Path,
    control_protocol_path: Path,
    control_result_path: Path,
    sweep_protocol_path: Path,
    sweep_result_path: Path,
    rescue_protocol_path: Path,
    rescue_protocol_hash: str,
    rescue_source_path: Path,
    rescue_source_hash: str,
) -> dict[str, bool]:
    identities = context["identities"]
    hashes = context["hashes"]
    sweep_hashes = context["sweep_hashes"]
    sustained = context["sustained_protocol"]
    reference = context["reference"]
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
        "reference_library": sha256_file(reference_library_path)
        == identities["native_library_sha256"],
        "candidate_library": (
            sha256_file(candidate_library_path)
            == context["candidate_library_sha256"]
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
            sha256_file(control_result_path)
            == hashes["control_result_sha256"]
        ),
        "sweep_protocol": (
            sha256_file(sweep_protocol_path)
            == sweep_hashes["sweep_protocol_sha256"]
        ),
        "sweep_result": (
            sha256_file(sweep_result_path)
            == sweep_hashes["sweep_result_sha256"]
        ),
        "rescue_protocol": (
            sha256_file(rescue_protocol_path) == rescue_protocol_hash
        ),
        "teacher_source_config": (
            sha256_file(source_model / "config.json")
            == sustained["source_config_sha256"]
        ),
        "teacher_source_index": (
            sha256_file(source_model / "model.safetensors.index.json")
            == sustained["source_index_sha256"]
        ),
        "teacher_source_shards": _post_source_shards(
            reference,
            sustained["source_shard_sha256"],
        ),
        "rescue_source_inventory": all(
            sha256_file(repository / relative) == expected
            for relative, expected in context[
                "rescue_source_inventory"
            ].items()
        ),
        "control_source": (
            sha256_file(context["control_source_path"])
            == context["control_source_hash"]
        ),
        "sweep_source": (
            sha256_file(context["sweep_source_path"])
            == context["sweep_source_hash"]
        ),
        "rescue_source": (
            sha256_file(rescue_source_path) == rescue_source_hash
        ),
    }


def evaluate_native_olmoe_layer_rescue(
    *,
    package: str | Path,
    manifest_sha256: str,
    reference_library: str | Path,
    candidate_library: str | Path,
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
    sweep_result: str | Path,
    sweep_result_sha256: str,
    rescue_protocol: str | Path,
    rescue_protocol_sha256: str,
    out: str | Path,
    threads: int = _THREADS,
) -> dict[str, Any]:
    """Run the frozen 45-candidate greedy search and one internal holdout."""

    output_path = Path(out).expanduser().resolve()
    if output_path.exists():
        raise ValueError("layer-rescue result target already exists")
    if threads != _THREADS:
        raise ValueError("layer rescue requires the matched 12 threads")
    paths = _resolve_paths(
        package=package,
        reference_library=reference_library,
        candidate_library=candidate_library,
        dataset=dataset,
        corpus_manifest=corpus_manifest,
        teacher_reference=teacher_reference,
        teacher_arrays=teacher_arrays,
        sustained_protocol=sustained_protocol,
        sustained_result=sustained_result,
        control_protocol=control_protocol,
        control_result=control_result,
        sweep_protocol=sweep_protocol,
        sweep_result=sweep_result,
    )
    context = _authenticate_prerequisites(
        **paths,
        manifest_sha256=manifest_sha256,
        sustained_protocol_sha256=sustained_protocol_sha256,
        sustained_result_sha256=sustained_result_sha256,
        control_protocol_sha256=control_protocol_sha256,
        control_result_sha256=control_result_sha256,
        sweep_protocol_sha256=sweep_protocol_sha256,
        sweep_result_sha256=sweep_result_sha256,
    )
    rescue_protocol_path = Path(rescue_protocol).expanduser().resolve()
    rescue_protocol_value = sweep_source._read_object(
        rescue_protocol_path,
        "layer-rescue protocol",
    )
    rescue_protocol_hash = sha256_file(rescue_protocol_path)
    rescue_source_path = Path(__file__).resolve()
    rescue_source_hash = sha256_file(rescue_source_path)
    _validate_protocol(
        rescue_protocol_value,
        context,
        protocol_hash=rescue_protocol_hash,
        supplied_protocol_hash=rescue_protocol_sha256,
        rescue_source_hash=rescue_source_hash,
    )

    prediction_positions = _SEQUENCES * _POSITIONS_PER_SEQUENCE
    with np.load(paths["arrays_path"], allow_pickle=False) as arrays:
        if set(arrays.files) != {"logits", "hidden", "targets"}:
            raise ValueError("layer-rescue teacher arrays have unexpected keys")
        teacher_logits = np.asarray(arrays["logits"], dtype=np.float32)
        teacher_hidden = np.asarray(arrays["hidden"], dtype=np.float32)
        targets = np.asarray(arrays["targets"], dtype=np.int64)
    model = context["model"]
    expected_targets = np.asarray(
        [token for sequence in context["input_ids"] for token in sequence[1:]],
        dtype=np.int64,
    )
    if (
        teacher_logits.shape
        != (prediction_positions, int(model["vocab_size"]))
        or teacher_hidden.shape
        != (prediction_positions, int(model["hidden_size"]))
        or targets.shape != (prediction_positions,)
        or not np.array_equal(targets, expected_targets)
    ):
        raise ValueError("layer-rescue teacher array shapes are invalid")

    split = rescue_protocol_value["record_split"]
    selection_indices = [
        int(row["sequence_index"]) for row in split["selection"]
    ]
    holdout_indices = [
        int(row["sequence_index"]) for row in split["internal_holdout"]
    ]
    started = time.perf_counter()
    layered_abi_parity = _evaluate_layered_abi_parity(
        context=context,
        reference_library_path=paths["reference_library_path"],
        candidate_library_path=paths["candidate_library_path"],
        sequence_index=selection_indices[0],
        threads=threads,
    )
    if not layered_abi_parity["passed"]:
        post_authentication = _post_authentication(
            context=context,
            **paths,
            manifest_sha256=manifest_sha256,
            rescue_protocol_path=rescue_protocol_path,
            rescue_protocol_hash=rescue_protocol_hash,
            rescue_source_path=rescue_source_path,
            rescue_source_hash=rescue_source_hash,
        )
        report = {
            "schema_version": 1,
            "experiment": _EXPERIMENT,
            "status": "layer_rescue_invalid",
            "provenance": {
                "protocol_frozen_before_any_candidate_execution": True,
                "static_sweep_zero_pass_result_known_before_protocol_freeze": True,
                "record_split_fixed_before_candidate_execution": True,
                "candidate_order_and_scoring_fixed_before_candidate_execution": True,
                "layered_abi_parity_failed_before_candidate_execution": True,
                "scientific_role": "invalid execution; no selection",
                "execution_interface": _EXECUTION_INTERFACE,
                "rescue_protocol_sha256": rescue_protocol_hash,
                "rescue_source_sha256": rescue_source_hash,
                **context["hashes"],
                **context["sweep_hashes"],
            },
            "artifacts": {
                **context["identities"],
                **context["hashes"],
                **context["sweep_hashes"],
                "control_source_sha256": context["control_source_hash"],
                "sweep_source_sha256": context["sweep_source_hash"],
                "rescue_protocol_sha256": rescue_protocol_hash,
                "rescue_source_sha256": rescue_source_hash,
                "candidate_native_library_sha256": context[
                    "candidate_library_sha256"
                ],
                "rescue_source_inventory_sha256": context[
                    "rescue_source_inventory"
                ],
            },
            "record_split": split,
            "layered_abi_all_base_parity": layered_abi_parity,
            "greedy_round_results": [],
            "candidate_evaluation_count": 0,
            "selected_rescued_layers": None,
            "selected_schedule_is_development_only": False,
            "internal_holdout_result": None,
            "evidence_checks": {
                "layered_abi_all_base_parity": False,
                "no_candidate_outputs_inspected": True,
                "post_run_authentication": all(
                    post_authentication.values()
                ),
            },
            "evidence_passed": False,
            "internal_holdout_quality_passed": False,
            "fresh_confirmation_required": False,
            "decision": "stop_and_diagnose_layered_abi_all_base_parity",
            "post_run_authentication": post_authentication,
            "performance": {
                "execution_seconds": layered_abi_parity["elapsed_seconds"],
                "layered_abi_parity_seconds": layered_abi_parity[
                    "elapsed_seconds"
                ],
                "candidate_primary_sequence_seconds": 0.0,
                "holdout_primary_sequence_seconds": 0.0,
                "holdout_reset_replay_seconds": 0.0,
            },
            "limitations": rescue_protocol_value["limitations"],
        }
        atomic_json(output_path, report)
        return report
    selected_layers: list[int] = []
    round_results: list[dict[str, Any]] = []
    for round_index in range(_ROUNDS):
        candidate_layers = [
            layer
            for layer in rescue_protocol_value["candidate_layer_order"]
            if layer not in selected_layers
        ]
        if len(candidate_layers) != _CANDIDATE_COUNTS[round_index]:
            raise RuntimeError("layer-rescue candidate count drifted")
        candidate_results: list[dict[str, Any]] = []
        for candidate_layer in candidate_layers:
            schedule = [*selected_layers, int(candidate_layer)]
            candidate_result = _evaluate_schedule(
                schedule,
                sequence_indices=selection_indices,
                split_name="selection",
                context=context,
                library_path=paths["candidate_library_path"],
                teacher_logits=teacher_logits,
                teacher_hidden=teacher_hidden,
                targets=targets,
                threads=threads,
            )
            candidate_result["candidate_layer"] = int(candidate_layer)
            candidate_results.append(candidate_result)
        ranked_candidates = sorted(candidate_results, key=_candidate_sort_key)
        selected_layer = int(ranked_candidates[0]["candidate_layer"])
        round_results.append(
            {
                "round": round_index + 1,
                "starting_rescued_layers": list(selected_layers),
                "candidate_order": candidate_layers,
                "candidate_count": len(candidate_results),
                "candidate_results": candidate_results,
                "selected_layer": selected_layer,
                "selected_schedule": [*selected_layers, selected_layer],
                "selected_score": ranked_candidates[0]["selection_score"],
                "ranking": [
                    {
                        "rank": rank,
                        "candidate_layer": result["candidate_layer"],
                        "worst_normalized_margin": result[
                            "selection_score"
                        ]["worst_normalized_margin"],
                        "mean_normalized_margin": result[
                            "selection_score"
                        ]["mean_normalized_margin"],
                    }
                    for rank, result in enumerate(ranked_candidates, start=1)
                ],
            }
        )
        selected_layers.append(selected_layer)

    if len(selected_layers) != _ROUNDS or len(set(selected_layers)) != _ROUNDS:
        raise RuntimeError("layer-rescue greedy schedule is invalid")
    holdout_result = _evaluate_schedule(
        selected_layers,
        sequence_indices=holdout_indices,
        split_name="internal_holdout",
        context=context,
        library_path=paths["candidate_library_path"],
        teacher_logits=teacher_logits,
        teacher_hidden=teacher_hidden,
        targets=targets,
        threads=threads,
        replay_sequence_index=holdout_indices[0],
    )
    execution_seconds = time.perf_counter() - started
    post_authentication = _post_authentication(
        context=context,
        **paths,
        manifest_sha256=manifest_sha256,
        rescue_protocol_path=rescue_protocol_path,
        rescue_protocol_hash=rescue_protocol_hash,
        rescue_source_path=rescue_source_path,
        rescue_source_hash=rescue_source_hash,
    )
    candidate_results = [
        candidate
        for round_result in round_results
        for candidate in round_result["candidate_results"]
    ]
    candidate_evidence_passed = (
        len(candidate_results) == _EXPECTED_CANDIDATE_EVALUATIONS
        and all(candidate["evidence_passed"] for candidate in candidate_results)
    )
    round_resource_checks = {
        f"round_{round_index + 1}": all(
            candidate["attention_expectations_per_sequence"]
            == rescue_protocol_value["round_resource_contracts"][round_index][
                "attention_expectations_per_sequence"
            ]
            for candidate in round_result["candidate_results"]
        )
        for round_index, round_result in enumerate(round_results)
    }
    final_expectations = holdout_result["attention_expectations_per_sequence"]
    final_schedule_checks = {
        "exactly_three_distinct_rescued_layers": (
            len(selected_layers) == _ROUNDS
            and len(set(selected_layers)) == _ROUNDS
        ),
        "thirteen_base_layers": (
            len(_schedule_policies(selected_layers)) - len(selected_layers)
            == _LAYERS - _ROUNDS
        ),
        "attention_logical_read_bytes": (
            final_expectations["attention_logical_read_bytes"]
            == _EXPECTED_FINAL_LOGICAL_READ_BYTES
        ),
        "attention_logical_read_fraction": (
            final_expectations["attention_logical_read_fraction"]
            == _EXPECTED_FINAL_LOGICAL_READ_FRACTION
            and final_expectations["attention_logical_read_fraction"]
            <= _THRESHOLDS["maximum_attention_logical_read_fraction"]
        ),
        "q7_expectations_unchanged": (
            holdout_result["q7_expectations_per_sequence"]
            == context["q7_expectations"]
        ),
        "q7_traffic_fraction": (
            holdout_result["q7_traffic_contract_per_sequence"][
                "q7_fraction_of_all_expert_ideal_q4"
            ]
            <= _THRESHOLDS["maximum_q7_traffic_fraction"]
        ),
    }
    evidence_checks = {
        "layered_abi_all_base_parity": layered_abi_parity["passed"],
        "candidate_evaluation_count": (
            len(candidate_results) == _EXPECTED_CANDIDATE_EVALUATIONS
        ),
        "candidate_counts_by_round": (
            [row["candidate_count"] for row in round_results]
            == _CANDIDATE_COUNTS
        ),
        "candidate_evidence": candidate_evidence_passed,
        "round_resource_contracts": all(round_resource_checks.values()),
        "holdout_evidence": holdout_result["evidence_passed"],
        "final_schedule": all(final_schedule_checks.values()),
        "post_run_authentication": all(post_authentication.values()),
    }
    evidence_passed = all(evidence_checks.values())
    if not evidence_passed:
        status = "layer_rescue_invalid"
        decision = "stop_and_diagnose_layer_rescue_evidence"
    elif holdout_result["quality_passed"]:
        status = "layer_rescue_development_complete"
        decision = (
            "integrate_selected_schedule_then_freeze_fresh_package_native_"
            "confirmation"
        )
    else:
        status = "layer_rescue_development_complete"
        decision = "investigate_head_wise_teacher_guided_attention_allocation"
    report = {
        "schema_version": 1,
        "experiment": _EXPERIMENT,
        "status": status,
        "provenance": {
            "protocol_frozen_before_any_candidate_execution": True,
            "static_sweep_zero_pass_result_known_before_protocol_freeze": True,
            "record_split_fixed_before_candidate_execution": True,
            "candidate_order_and_scoring_fixed_before_candidate_execution": True,
            "all_candidates_executed_before_each_round_selection": True,
            "internal_holdout_outputs_unseen_during_greedy_selection": True,
            "scientific_role": "development selection; not confirmation",
            "execution_interface": _EXECUTION_INTERFACE,
            "rescue_protocol_sha256": rescue_protocol_hash,
            "rescue_source_sha256": rescue_source_hash,
            "candidate_native_library_sha256": context[
                "candidate_library_sha256"
            ],
            "rescue_source_inventory_sha256": context[
                "rescue_source_inventory"
            ],
            **context["hashes"],
            **context["sweep_hashes"],
        },
        "artifacts": {
            **context["identities"],
            **context["hashes"],
            **context["sweep_hashes"],
            "control_source_sha256": context["control_source_hash"],
            "sweep_source_sha256": context["sweep_source_hash"],
            "rescue_protocol_sha256": rescue_protocol_hash,
            "rescue_source_sha256": rescue_source_hash,
            "candidate_native_library_sha256": context[
                "candidate_library_sha256"
            ],
            "rescue_source_inventory_sha256": context[
                "rescue_source_inventory"
            ],
        },
        "configuration": {
            "candidate_device": "cpu",
            "candidate_threads": threads,
            "transformers_model_shell_used": False,
            "execution_interface": _EXECUTION_INTERFACE,
            "q7_artifact_or_policy_changed": False,
            "package_manifest_mutated": False,
            "base_attention_policy": dict(_BASE_POLICY),
            "rescue_attention_policy": dict(_RESCUE_POLICY),
            "selected_rescued_layers": selected_layers,
            "final_attention_policies": _schedule_policies(selected_layers),
            "final_schedule_contract": context["final_schedule_contract"],
            "q7_traffic_contract_per_sequence": (
                holdout_result["q7_traffic_contract_per_sequence"]
            ),
            "measured_hardware_traffic": False,
        },
        "record_split": split,
        "layered_abi_all_base_parity": layered_abi_parity,
        "quality_bands": sweep_source._expected_bands(),
        "thresholds": _THRESHOLDS,
        "population_contracts": rescue_protocol_value["population_contracts"],
        "scoring_rule": _scoring_rule(),
        "greedy_round_results": round_results,
        "candidate_evaluation_count": len(candidate_results),
        "selected_rescued_layers": selected_layers,
        "selected_schedule_is_development_only": True,
        "internal_holdout_result": holdout_result,
        "final_schedule_checks": final_schedule_checks,
        "round_resource_checks": round_resource_checks,
        "evidence_checks": evidence_checks,
        "evidence_passed": evidence_passed,
        "internal_holdout_quality_passed": holdout_result["quality_passed"],
        "fresh_confirmation_required": (
            evidence_passed and holdout_result["quality_passed"]
        ),
        "decision": decision,
        "post_run_authentication": post_authentication,
        "performance": {
            "execution_seconds": execution_seconds,
            "layered_abi_parity_seconds": layered_abi_parity[
                "elapsed_seconds"
            ],
            "candidate_primary_sequence_seconds": sum(
                candidate["performance"]["primary_sequence_seconds"]
                for candidate in candidate_results
            ),
            "holdout_primary_sequence_seconds": holdout_result["performance"][
                "primary_sequence_seconds"
            ],
            "holdout_reset_replay_seconds": holdout_result["performance"][
                "reset_replay_seconds"
            ],
        },
        "limitations": rescue_protocol_value["limitations"],
    }
    atomic_json(output_path, report)
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze or run the native OLMoE three-layer rescue"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--package", required=True, type=Path)
        command.add_argument("--manifest-sha256", required=True)
        command.add_argument("--reference-library", required=True, type=Path)
        command.add_argument("--candidate-library", required=True, type=Path)
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
        command.add_argument("--sweep-protocol", required=True, type=Path)
        command.add_argument("--sweep-protocol-sha256", required=True)
        command.add_argument("--sweep-result", required=True, type=Path)
        command.add_argument("--sweep-result-sha256", required=True)
        command.add_argument("--out", required=True, type=Path)
        command.add_argument("--threads", type=int, default=_THREADS)

    freeze_parser = commands.add_parser(
        "freeze",
        help="freeze the split and complete greedy search before execution",
    )
    add_common(freeze_parser)
    evaluate_parser = commands.add_parser(
        "evaluate",
        help="execute the frozen greedy search and internal holdout",
    )
    add_common(evaluate_parser)
    evaluate_parser.add_argument("--rescue-protocol", required=True, type=Path)
    evaluate_parser.add_argument("--rescue-protocol-sha256", required=True)
    args = parser.parse_args()
    common = {
        "package": args.package,
        "manifest_sha256": args.manifest_sha256,
        "reference_library": args.reference_library,
        "candidate_library": args.candidate_library,
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
        "sweep_protocol": args.sweep_protocol,
        "sweep_protocol_sha256": args.sweep_protocol_sha256,
        "sweep_result": args.sweep_result,
        "sweep_result_sha256": args.sweep_result_sha256,
        "out": args.out,
        "threads": args.threads,
    }
    if args.command == "freeze":
        result = freeze_native_olmoe_layer_rescue_protocol(**common)
    else:
        result = evaluate_native_olmoe_layer_rescue(
            **common,
            rescue_protocol=args.rescue_protocol,
            rescue_protocol_sha256=args.rescue_protocol_sha256,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
