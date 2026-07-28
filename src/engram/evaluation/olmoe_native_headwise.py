"""Prospective teacher-guided head-wise OLMoE attention experiment.

The experiment is intentionally split into two independently frozen phases.
First, a dense untouched teacher exposes attention maps for only the two
development-selection records and those maps are reduced to one deterministic
51-of-256 head mask.  Second, after that mask exists, a separate protocol binds
an immutable native head-wise library and evaluates exactly one causal
candidate on the six records that were not used to choose the mask.

Nothing in this module promotes the six-record screen to confirmation.  A
passing screen still requires a fresh, sealed eight-sequence confirmation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import engram.evaluation.olmoe_native_attention_sweep as sweep_source
import engram.evaluation.olmoe_native_layer_rescue as layer_rescue
from engram.evaluation.olmoe_native_causal import _position_metrics
from engram.evaluation.olmoe_native_sustained import (
    _POSITIONS_PER_SEQUENCE,
    _QUALITY_BANDS,
    _THRESHOLDS,
    _attention_expectations,
    _deterministic_metrics,
    _post_source_shards,
    _quality_checks,
    _structural_checks as _sustained_structural_checks,
    _update_diagnostic_hashes,
)
from engram.models.olmoe import audit_olmoe_source
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.tracing.olmoe import _prepare_transformers_imports
from engram.utils import atomic_json, sha256_file, sha256_json


_TRACE_PROTOCOL_EXPERIMENT = (
    "olmoe_native_q7_headwise_dense_attention_trace_protocol"
)
_TRACE_CAPTURE_EXPERIMENT = "olmoe_native_q7_headwise_dense_attention_trace"
_MASK_EXPERIMENT = "olmoe_native_q7_teacher_guided_head_mask"
_SCREEN_EXPERIMENT = "olmoe_native_q7_headwise_causal_internal_screen"
_TRACE_PROTOCOL_STATUS = "frozen_before_dense_attention_map_capture"
_SCREEN_PROTOCOL_STATUS = "frozen_before_headwise_parity_or_candidate_execution"
_THREADS = 12
_LAYERS = 16
_HEADS = 16
_TOTAL_HEADS = _LAYERS * _HEADS
_RESCUED_HEADS = 51
_SELECTION_SEQUENCES = 2
_INTERNAL_SEQUENCES = 6
_LOCAL_WINDOW = 16
_TOP_OLDER = 4
_EXECUTION_INTERFACE = "raw_native_token_runtime_per_head_attention_policies"

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
_EXPECTED_HEADWISE = {
    "positions_processed": 128,
    "attention_state_bytes": 12_284_864,
    "attention_scratch_bytes": 107_136,
    "attention_eviction_events": 22_960,
    "attention_older_candidate_entries_scored": 177_940,
    "attention_older_selected_entries": 90_610,
    "attention_sink_insertions": 410,
    "attention_heavy_hitter_updates_minimum": 1_230,
    "attention_heavy_hitter_updates_maximum": 22_550,
    "attention_local_kv_bytes": 835_887_104,
    "attention_candidate_key_bytes": 91_105_280,
    "attention_selected_value_bytes": 46_392_320,
    "attention_logical_read_bytes": 973_384_704,
    "dense_full_context_logical_kv_bytes": 2_164_260_864,
}
_EXPECTED_FRACTION = 0.44975387218386625
_FIFTY_TWO_FRACTION = 0.4524379996366279


def _read_object(path: Path, label: str) -> dict[str, Any]:
    return sweep_source._read_object(path, label)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write a compressed-independent NPZ through an adjacent atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp.npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_common_paths(
    *,
    package: str | Path,
    reference_library: str | Path,
    layered_library: str | Path,
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
    layer_rescue_protocol: str | Path,
    layer_rescue_result: str | Path,
) -> dict[str, Path]:
    paths = layer_rescue._resolve_paths(
        package=package,
        reference_library=reference_library,
        candidate_library=layered_library,
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
    paths.update(
        {
            "layer_rescue_protocol_path": Path(layer_rescue_protocol)
            .expanduser()
            .resolve(),
            "layer_rescue_result_path": Path(layer_rescue_result)
            .expanduser()
            .resolve(),
        }
    )
    return paths


def _validate_historical_source_inventory(inventory: Any) -> dict[str, str]:
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("head-wise historical source inventory is invalid")
    normalized: dict[str, str] = {}
    for relative, digest in inventory.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not _is_sha256(digest)
        ):
            raise ValueError("head-wise historical source descriptor is invalid")
        normalized[relative] = digest
    return normalized


def _validate_layer_rescue_archive(
    protocol: dict[str, Any],
    result: dict[str, Any],
    *,
    protocol_sha256: str,
    result_sha256: str,
    context: dict[str, Any],
) -> None:
    """Authenticate the immutable failed layer-rescue boundary by content.

    Historical source hashes are descriptors, not assertions that the current
    tree still equals the old tree.  The immutable protocol/result digests bind
    those descriptors and the result's post-run authentication records that
    they were checked when the experiment executed.
    """

    del result_sha256
    inventory = _validate_historical_source_inventory(
        protocol.get("rescue_source_inventory_sha256")
    )
    rescue_source_hash = protocol.get("rescue_source_sha256")
    artifacts = result.get("artifacts")
    post = result.get("post_run_authentication")
    archived_parity = result.get("layered_abi_all_base_parity")
    archived_candidate_hashes = (
        archived_parity.get("candidate_hashes")
        if isinstance(archived_parity, dict)
        else None
    )
    split = context["split"]
    holdout_indices = [
        int(row["sequence_index"]) for row in split["internal_holdout"]
    ]
    holdout = result.get("internal_holdout_result")
    if (
        protocol.get("schema_version") != 1
        or protocol.get("experiment") != layer_rescue._EXPERIMENT
        or protocol.get("status") != layer_rescue._PROTOCOL_STATUS
        or protocol.get("record_split") != split
        or protocol.get("candidate_native_library_sha256")
        != context["candidate_library_sha256"]
        or protocol.get("base_attention_policy") != _BASE_POLICY
        or protocol.get("rescue_attention_policy") != _RESCUE_POLICY
        or protocol.get("expected_candidate_evaluations")
        != layer_rescue._EXPECTED_CANDIDATE_EVALUATIONS
        or protocol.get("thresholds") != _THRESHOLDS
        or protocol.get("quality_bands") != sweep_source._expected_bands()
        or not _is_sha256(rescue_source_hash)
        or inventory.get(
            "src/engram/evaluation/olmoe_native_layer_rescue.py"
        )
        != rescue_source_hash
        or not isinstance(artifacts, dict)
        or artifacts.get("rescue_protocol_sha256") != protocol_sha256
        or artifacts.get("rescue_source_sha256") != rescue_source_hash
        or artifacts.get("rescue_source_inventory_sha256") != inventory
        or artifacts.get("candidate_native_library_sha256")
        != context["candidate_library_sha256"]
        or result.get("schema_version") != 1
        or result.get("experiment") != layer_rescue._EXPERIMENT
        or result.get("status") != "layer_rescue_development_complete"
        or result.get("record_split") != split
        or result.get("candidate_evaluation_count")
        != layer_rescue._EXPECTED_CANDIDATE_EVALUATIONS
        or result.get("evidence_passed") is not True
        or result.get("internal_holdout_quality_passed") is not False
        or result.get("fresh_confirmation_required") is not False
        or result.get("decision")
        != "investigate_head_wise_teacher_guided_attention_allocation"
        or result.get("thresholds") != _THRESHOLDS
        or result.get("quality_bands") != sweep_source._expected_bands()
        or not isinstance(archived_parity, dict)
        or archived_parity.get("passed") is not True
        or not isinstance(archived_candidate_hashes, dict)
        or any(
            not _is_sha256(archived_candidate_hashes.get(name))
            for name in (
                "hidden_sha256",
                "logits_sha256",
                "counter_stream_sha256",
            )
        )
        or not isinstance(post, dict)
        or not post
        or not all(value is True for value in post.values())
        or not isinstance(holdout, dict)
        or holdout.get("evidence_passed") is not True
        or holdout.get("quality_passed") is not False
    ):
        raise ValueError("head-wise layer-rescue archive is invalid")
    rows = holdout.get("position_results")
    if not layer_rescue._position_grid_is_exact(rows, holdout_indices):
        raise ValueError("head-wise layer-rescue holdout grid is invalid")
    overall = layer_rescue._aggregate(rows)
    bands = layer_rescue._bands_from_rows(rows)
    checks = _quality_checks("overall", overall)
    for name, metrics in bands.items():
        checks.update(_quality_checks(name, metrics))
    if (
        holdout.get("metrics") != overall
        or holdout.get("position_bands") != bands
        or holdout.get("quality_checks") != checks
        or all(checks.values())
        or result.get("selected_rescued_layers") is None
        or len(result["selected_rescued_layers"]) != 3
        or len(set(result["selected_rescued_layers"])) != 3
    ):
        raise ValueError("head-wise layer-rescue conclusion is invalid")


def _authenticate_common(
    *,
    paths: dict[str, Path],
    manifest_sha256: str,
    sustained_protocol_sha256: str,
    sustained_result_sha256: str,
    control_protocol_sha256: str,
    control_result_sha256: str,
    sweep_protocol_sha256: str,
    sweep_result_sha256: str,
    layer_rescue_protocol_sha256: str,
    layer_rescue_result_sha256: str,
) -> dict[str, Any]:
    context = layer_rescue._authenticate_prerequisites(
        package_path=paths["package_path"],
        manifest_sha256=manifest_sha256,
        reference_library_path=paths["reference_library_path"],
        candidate_library_path=paths["candidate_library_path"],
        dataset_path=paths["dataset_path"],
        corpus_manifest_path=paths["corpus_manifest_path"],
        reference_path=paths["reference_path"],
        arrays_path=paths["arrays_path"],
        sustained_protocol_path=paths["sustained_protocol_path"],
        sustained_protocol_sha256=sustained_protocol_sha256,
        sustained_result_path=paths["sustained_result_path"],
        sustained_result_sha256=sustained_result_sha256,
        control_protocol_path=paths["control_protocol_path"],
        control_protocol_sha256=control_protocol_sha256,
        control_result_path=paths["control_result_path"],
        control_result_sha256=control_result_sha256,
        sweep_protocol_path=paths["sweep_protocol_path"],
        sweep_protocol_sha256=sweep_protocol_sha256,
        sweep_result_path=paths["sweep_result_path"],
        sweep_result_sha256=sweep_result_sha256,
    )
    layer_protocol_hash = sha256_file(paths["layer_rescue_protocol_path"])
    layer_result_hash = sha256_file(paths["layer_rescue_result_path"])
    if (
        layer_protocol_hash != layer_rescue_protocol_sha256.lower()
        or layer_result_hash != layer_rescue_result_sha256.lower()
    ):
        raise ValueError("head-wise layer-rescue archive hash is invalid")
    layer_protocol = _read_object(
        paths["layer_rescue_protocol_path"],
        "layer-rescue protocol",
    )
    layer_result = _read_object(
        paths["layer_rescue_result_path"],
        "layer-rescue result",
    )
    _validate_layer_rescue_archive(
        layer_protocol,
        layer_result,
        protocol_sha256=layer_protocol_hash,
        result_sha256=layer_result_hash,
        context=context,
    )
    model = context["model"]
    if (
        int(model["layers"]) != _LAYERS
        or int(model["query_heads"]) != _HEADS
        or int(model["key_value_heads"]) != _HEADS
    ):
        raise ValueError("head-wise v1 requires 16 equal query/KV heads")
    context.update(
        {
            "layer_rescue_protocol": layer_protocol,
            "layer_rescue_result": layer_result,
            "layer_rescue_protocol_sha256": layer_protocol_hash,
            "layer_rescue_result_sha256": layer_result_hash,
            "layer_rescue_historical_source_inventory": (
                _validate_historical_source_inventory(
                    layer_protocol["rescue_source_inventory_sha256"]
                )
            ),
        }
    )
    return context


def _current_source_inventory(
    historical_inventory: Mapping[str, str],
) -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    relative_paths = set(historical_inventory)
    relative_paths.update(
        {
            "src/engram/evaluation/olmoe_native_headwise.py",
            "src/engram/models/olmoe.py",
            "src/engram/tracing/olmoe.py",
            "src/engram/runtime/olmoe_native.py",
            "native/include/engram/olmoe_token_runtime.h",
            "native/include/engram/olmoe_token_runtime_c.h",
            "native/src/olmoe_token_runtime.cpp",
            "native/src/olmoe_token_runtime_c.cpp",
        }
    )
    inventory: dict[str, str] = {}
    for relative in sorted(relative_paths):
        source = repository / relative
        if not source.is_file():
            raise ValueError(f"head-wise current source is missing: {relative}")
        inventory[relative] = sha256_file(source)
    return inventory


def _headwise_expectations(
    model: Mapping[str, int],
    rescued_heads: Sequence[tuple[int, int]],
    *,
    positions: int = _POSITIONS_PER_SEQUENCE,
) -> dict[str, int | float]:
    rescued = list(rescued_heads)
    if (
        int(model["layers"]) != _LAYERS
        or int(model["query_heads"]) != _HEADS
        or int(model["key_value_heads"]) != _HEADS
        or len(set(rescued)) != len(rescued)
        or any(
            isinstance(layer, bool)
            or isinstance(head, bool)
            or not isinstance(layer, int)
            or not isinstance(head, int)
            or layer < 0
            or layer >= _LAYERS
            or head < 0
            or head >= _HEADS
            for layer, head in rescued
        )
        or len(rescued) > _TOTAL_HEADS
    ):
        raise ValueError("head-wise rescue mask is invalid")
    one_head_model = {
        "layers": 1,
        "query_heads": 1,
        "key_value_heads": 1,
        "head_dimension": int(model["head_dimension"]),
    }
    base = _attention_expectations(
        one_head_model,
        _BASE_POLICY,
        positions=positions,
    )
    dense = _attention_expectations(
        one_head_model,
        _RESCUE_POLICY,
        positions=positions,
    )
    base_count = _TOTAL_HEADS - len(rescued)
    dense_count = len(rescued)
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
        result[name] = (
            base_count * int(base[name]) + dense_count * int(dense[name])
        )
    result["attention_logical_read_fraction"] = (
        int(result["attention_logical_read_bytes"])
        / int(result["dense_full_context_logical_kv_bytes"])
    )
    return result


def _headwise_budget_contract(model: Mapping[str, int]) -> dict[str, Any]:
    selected = [(index // _HEADS, index % _HEADS) for index in range(51)]
    selected_expectations = _headwise_expectations(model, selected)
    if any(
        selected_expectations[name] != value
        for name, value in _EXPECTED_HEADWISE.items()
    ) or selected_expectations["attention_logical_read_fraction"] != (
        _EXPECTED_FRACTION
    ):
        raise ValueError("head-wise 51-head resource contract is invalid")
    fifty_two = _headwise_expectations(
        model,
        [(index // _HEADS, index % _HEADS) for index in range(52)],
    )
    if fifty_two["attention_logical_read_fraction"] != _FIFTY_TWO_FRACTION:
        raise ValueError("head-wise 52-head boundary is invalid")
    return {
        "total_heads": _TOTAL_HEADS,
        "rescued_heads": _RESCUED_HEADS,
        "base_heads": _TOTAL_HEADS - _RESCUED_HEADS,
        "base_policy": dict(_BASE_POLICY),
        "rescue_policy": dict(_RESCUE_POLICY),
        "attention_expectations_per_sequence": selected_expectations,
        "attention_logical_read_bytes_per_sequence": (
            _EXPECTED_HEADWISE["attention_logical_read_bytes"]
        ),
        "dense_full_context_logical_kv_bytes_per_sequence": (
            _EXPECTED_HEADWISE["dense_full_context_logical_kv_bytes"]
        ),
        "attention_logical_read_fraction": _EXPECTED_FRACTION,
        "maximum_attention_logical_read_fraction": _THRESHOLDS[
            "maximum_attention_logical_read_fraction"
        ],
        "next_head_boundary": {
            "rescued_heads": 52,
            "attention_logical_read_fraction": _FIFTY_TWO_FRACTION,
            "within_budget": False,
        },
    }


def _head_policies(
    selected_heads: Sequence[tuple[int, int]],
) -> list[list[dict[str, int]]]:
    selected = set(selected_heads)
    if len(selected) != len(selected_heads):
        raise ValueError("head-wise rescue mask contains duplicates")
    # Reuse the analytical validator for the shape and coordinates.
    _headwise_expectations(
        {
            "layers": _LAYERS,
            "query_heads": _HEADS,
            "key_value_heads": _HEADS,
            "head_dimension": 128,
        },
        list(selected_heads),
    )
    return [
        [
            dict(_RESCUE_POLICY if (layer, head) in selected else _BASE_POLICY)
            for head in range(_HEADS)
        ]
        for layer in range(_LAYERS)
    ]


def _selection_rows(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(context["split"]["selection"])


def _base_bindings(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **context["identities"],
        **context["hashes"],
        **context["sweep_hashes"],
        "control_source_sha256": context["control_source_hash"],
        "sweep_source_sha256": context["sweep_source_hash"],
        "layer_rescue_protocol_sha256": context[
            "layer_rescue_protocol_sha256"
        ],
        "layer_rescue_result_sha256": context["layer_rescue_result_sha256"],
        "layered_native_library_sha256": context["candidate_library_sha256"],
        "layer_rescue_source_sha256": context["layer_rescue_protocol"][
            "rescue_source_sha256"
        ],
        "layer_rescue_source_inventory_sha256": context[
            "layer_rescue_historical_source_inventory"
        ],
    }


def _build_trace_protocol(
    context: Mapping[str, Any],
    *,
    source_sha256: str,
    source_inventory: Mapping[str, str],
    device: str,
    threads: int,
) -> dict[str, Any]:
    selection = _selection_rows(context)
    selection_indices = [int(row["sequence_index"]) for row in selection]
    selected_inputs = [
        context["input_ids"][index][:-1] for index in selection_indices
    ]
    return {
        "schema_version": 1,
        "experiment": _TRACE_PROTOCOL_EXPERIMENT,
        "status": _TRACE_PROTOCOL_STATUS,
        **_base_bindings(context),
        "headwise_source_sha256": source_sha256,
        "headwise_source_inventory_sha256": dict(source_inventory),
        "source_revision": context["sustained_protocol"]["source_revision"],
        "source_config_sha256": context["sustained_protocol"][
            "source_config_sha256"
        ],
        "source_index_sha256": context["sustained_protocol"][
            "source_index_sha256"
        ],
        "source_shard_sha256": context["sustained_protocol"][
            "source_shard_sha256"
        ],
        "record_split": context["split"],
        "record_split_identity": context["split"]["split_identity"],
        "selection_records": selection,
        "selection_sequence_indices": selection_indices,
        "selection_input_identity": sha256_json(selected_inputs),
        "selection_sequences": _SELECTION_SEQUENCES,
        "internal_screen_sequences": _INTERNAL_SEQUENCES,
        "tokens_per_selection_sequence": _POSITIONS_PER_SEQUENCE,
        "attention_array_contract": {
            "key": "attentions",
            "shape": [
                _SELECTION_SEQUENCES,
                _LAYERS,
                _HEADS,
                _POSITIONS_PER_SEQUENCE,
                _POSITIONS_PER_SEQUENCE,
            ],
            "dtype": "float32",
            "layout": "selection_sequence_layer_head_query_key",
            "causal_upper_triangle": "zero_within_1e-7",
            "expected_uncompressed_bytes": (
                _SELECTION_SEQUENCES
                * _LAYERS
                * _HEADS
                * _POSITIONS_PER_SEQUENCE
                * _POSITIONS_PER_SEQUENCE
                * 4
            ),
        },
        "teacher_capture": {
            "model_role": "untouched_dense_teacher",
            "dtype": "bfloat16",
            "device": device,
            "threads": threads,
            "attention_implementation": "eager",
            "eval": True,
            "inference_mode": True,
            "use_cache": False,
            "output_attentions": True,
            "output_hidden_states": False,
            "return_dict": True,
            "batch_size": 1,
        },
        "derivation": {
            "input_cast": "attention maps persisted as float32",
            "queries": "each selection sequence query p in [16,127]",
            "older_keys": "all causal key indices k <= p-16",
            "per_query_deficit": (
                "float64_sum(older_attention_mass) minus "
                "float64_sum(largest_four_older_attention_weights)"
            ),
            "sequence_bands": [
                {
                    "name": name,
                    "start": max(start, _LOCAL_WINDOW),
                    "stop": stop,
                }
                for name, start, stop in _QUALITY_BANDS
                if stop > _LOCAL_WINDOW
            ],
            "head_primary_score": (
                "float64 sum of per-query deficits over both selection records"
            ),
            "head_secondary_score": (
                "minimum of the per-record, per-band mean deficits"
            ),
            "stable_ranking": [
                "descending_total_deficit",
                "descending_minimum_sequence_band_mean_deficit",
                "ascending_layer",
                "ascending_head",
            ],
            "selected_prefix": _RESCUED_HEADS,
            "maps_from_internal_screen_records": False,
        },
        "budget_contract": _headwise_budget_contract(context["model"]),
        "provenance": {
            "protocol_frozen_before_attention_map_capture": True,
            "layer_rescue_failure_known_before_protocol_freeze": True,
            "selection_split_reused_exactly": True,
            "internal_record_attention_maps_prohibited": True,
            "mask_is_development_only": True,
            "fresh_eight_sequence_confirmation_required_after_screen_pass": True,
        },
        "limitations": [
            "The dense attention maps come from two development-selection records and can overfit those records.",
            "Teacher attention weight is an attribution heuristic, not a causal guarantee that rescuing a head improves logits.",
            "No attention map from any of the six internal-screen records may be captured or used to choose the mask.",
            "The fixed head mask is development-only even if the later six-record causal screen passes.",
            "W128 is full-context only over this 128-position protocol; beyond it, rescued heads remain bounded W128/C8/K4/S2 rather than dense.",
        ],
    }


def _validate_trace_protocol(
    protocol: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    protocol_sha256: str,
    supplied_sha256: str,
    source_sha256: str,
    source_inventory: Mapping[str, str],
) -> None:
    expected = _build_trace_protocol(
        context,
        source_sha256=source_sha256,
        source_inventory=source_inventory,
        device=str(protocol.get("teacher_capture", {}).get("device")),
        threads=int(protocol.get("teacher_capture", {}).get("threads", 0)),
    )
    if (
        protocol_sha256 != supplied_sha256.lower()
        or protocol != expected
        or protocol["teacher_capture"]["device"] not in {"cpu", "cuda"}
        or protocol["teacher_capture"]["threads"] != _THREADS
    ):
        raise ValueError("head-wise trace protocol contract is invalid")


def _common_context(
    *,
    package: str | Path,
    manifest_sha256: str,
    reference_library: str | Path,
    layered_library: str | Path,
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
    layer_rescue_protocol: str | Path,
    layer_rescue_protocol_sha256: str,
    layer_rescue_result: str | Path,
    layer_rescue_result_sha256: str,
) -> tuple[dict[str, Path], dict[str, Any]]:
    paths = _resolve_common_paths(
        package=package,
        reference_library=reference_library,
        layered_library=layered_library,
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
        layer_rescue_protocol=layer_rescue_protocol,
        layer_rescue_result=layer_rescue_result,
    )
    context = _authenticate_common(
        paths=paths,
        manifest_sha256=manifest_sha256,
        sustained_protocol_sha256=sustained_protocol_sha256,
        sustained_result_sha256=sustained_result_sha256,
        control_protocol_sha256=control_protocol_sha256,
        control_result_sha256=control_result_sha256,
        sweep_protocol_sha256=sweep_protocol_sha256,
        sweep_result_sha256=sweep_result_sha256,
        layer_rescue_protocol_sha256=layer_rescue_protocol_sha256,
        layer_rescue_result_sha256=layer_rescue_result_sha256,
    )
    return paths, context


def _current_inventory_matches(inventory: Mapping[str, str]) -> bool:
    repository = Path(__file__).resolve().parents[3]
    return all(
        (repository / relative).is_file()
        and sha256_file(repository / relative) == expected
        for relative, expected in inventory.items()
    )


def _common_post_authentication(
    context: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    manifest_sha256: str,
    source_inventory: Mapping[str, str],
) -> dict[str, bool]:
    reference = context["reference"]
    sustained = context["sustained_protocol"]
    source_model = Path(reference["source"]["model"]).expanduser().resolve()
    return {
        "package": (
            layer_rescue.validate_olmoe_native_package(
                paths["package_path"],
                expected_manifest_sha256=manifest_sha256,
            )
            == context["manifest"]
        ),
        "reference_library": (
            sha256_file(paths["reference_library_path"])
            == context["identities"]["native_library_sha256"]
        ),
        "layered_library": (
            sha256_file(paths["candidate_library_path"])
            == context["candidate_library_sha256"]
        ),
        "dataset": (
            sha256_file(paths["dataset_path"])
            == context["identities"]["dataset_sha256"]
        ),
        "corpus_manifest": (
            sha256_file(paths["corpus_manifest_path"])
            == context["identities"]["corpus_manifest_sha256"]
        ),
        "teacher_reference": (
            sha256_file(paths["reference_path"])
            == context["identities"]["teacher_reference_sha256"]
        ),
        "teacher_arrays": (
            sha256_file(paths["arrays_path"])
            == context["identities"]["teacher_arrays_sha256"]
        ),
        "sustained_protocol": (
            sha256_file(paths["sustained_protocol_path"])
            == context["hashes"]["sustained_protocol_sha256"]
        ),
        "sustained_result": (
            sha256_file(paths["sustained_result_path"])
            == context["hashes"]["sustained_result_sha256"]
        ),
        "control_protocol": (
            sha256_file(paths["control_protocol_path"])
            == context["hashes"]["control_protocol_sha256"]
        ),
        "control_result": (
            sha256_file(paths["control_result_path"])
            == context["hashes"]["control_result_sha256"]
        ),
        "sweep_protocol": (
            sha256_file(paths["sweep_protocol_path"])
            == context["sweep_hashes"]["sweep_protocol_sha256"]
        ),
        "sweep_result": (
            sha256_file(paths["sweep_result_path"])
            == context["sweep_hashes"]["sweep_result_sha256"]
        ),
        "layer_rescue_protocol": (
            sha256_file(paths["layer_rescue_protocol_path"])
            == context["layer_rescue_protocol_sha256"]
        ),
        "layer_rescue_result": (
            sha256_file(paths["layer_rescue_result_path"])
            == context["layer_rescue_result_sha256"]
        ),
        "layer_rescue_historical_source_descriptors": (
            _validate_historical_source_inventory(
                context["layer_rescue_protocol"][
                    "rescue_source_inventory_sha256"
                ]
            )
            == context["layer_rescue_historical_source_inventory"]
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
        "headwise_source_inventory": _current_inventory_matches(
            source_inventory
        ),
    }


def freeze_native_olmoe_headwise_trace_protocol(
    *,
    out: str | Path,
    device: str = "cpu",
    threads: int = _THREADS,
    **common: Any,
) -> dict[str, Any]:
    """Freeze teacher capture and mask derivation before exposing maps."""

    output_path = Path(out).expanduser().resolve()
    if output_path.exists():
        raise ValueError("head-wise trace protocol target already exists")
    if device not in {"cpu", "cuda"} or threads != _THREADS:
        raise ValueError("head-wise trace capture configuration is invalid")
    _paths, context = _common_context(**common)
    source_path = Path(__file__).resolve()
    source_inventory = _current_source_inventory(
        context["layer_rescue_historical_source_inventory"]
    )
    protocol = _build_trace_protocol(
        context,
        source_sha256=sha256_file(source_path),
        source_inventory=source_inventory,
        device=device,
        threads=threads,
    )
    atomic_json(output_path, protocol)
    return protocol


def _attention_array_evidence(
    attentions: np.ndarray,
    *,
    expected_shape: Sequence[int],
) -> tuple[dict[str, bool], dict[str, float]]:
    shape = tuple(int(value) for value in expected_shape)
    finite = bool(np.isfinite(attentions).all())
    minimum = float(np.min(attentions)) if attentions.size else math.nan
    maximum = float(np.max(attentions)) if attentions.size else math.nan
    if attentions.ndim == 5 and attentions.shape[-1] == attentions.shape[-2]:
        causal_violation = 0.0
        for query in range(attentions.shape[-2]):
            future = attentions[..., query, query + 1 :]
            if future.size:
                causal_violation = max(
                    causal_violation,
                    float(np.max(np.abs(future))),
                )
        row_error = float(
            np.max(
                np.abs(
                    np.sum(attentions, axis=-1, dtype=np.float64) - 1.0
                )
            )
        )
    else:
        causal_violation = math.inf
        row_error = math.inf
    checks = {
        "shape": attentions.shape == shape,
        "dtype_float32": attentions.dtype == np.float32,
        "finite": finite,
        "nonnegative_within_tolerance": minimum >= -1.0e-7,
        "causal_upper_triangle_zero_within_tolerance": (
            causal_violation <= 1.0e-7
        ),
        "attention_rows_normalized_within_tolerance": row_error <= 1.0e-2,
    }
    observations = {
        "minimum_attention_weight": minimum,
        "maximum_attention_weight": maximum,
        "maximum_causal_upper_triangle_absolute_weight": causal_violation,
        "maximum_attention_row_sum_absolute_error": row_error,
    }
    return checks, observations


def capture_native_olmoe_headwise_dense_attention(
    *,
    trace_protocol: str | Path,
    trace_protocol_sha256: str,
    arrays_out: str | Path,
    trace_out: str | Path,
    manifest_sha256: str,
    **common: Any,
) -> dict[str, Any]:
    """Capture only the two predeclared dense-teacher attention traces."""

    arrays_path = Path(arrays_out).expanduser().resolve()
    output_path = Path(trace_out).expanduser().resolve()
    if (
        arrays_path == output_path
        or arrays_path.exists()
        or output_path.exists()
    ):
        raise ValueError("head-wise trace output target already exists")
    paths, context = _common_context(
        manifest_sha256=manifest_sha256,
        **common,
    )
    protocol_path = Path(trace_protocol).expanduser().resolve()
    protocol = _read_object(protocol_path, "head-wise trace protocol")
    protocol_hash = sha256_file(protocol_path)
    source_path = Path(__file__).resolve()
    source_hash = sha256_file(source_path)
    source_inventory = _current_source_inventory(
        context["layer_rescue_historical_source_inventory"]
    )
    _validate_trace_protocol(
        protocol,
        context,
        protocol_sha256=protocol_hash,
        supplied_sha256=trace_protocol_sha256,
        source_sha256=source_hash,
        source_inventory=source_inventory,
    )
    capture = protocol["teacher_capture"]
    device = str(capture["device"])
    try:
        import torch

        _prepare_transformers_imports()
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for head-wise trace capture"
        ) from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for head-wise trace capture")
    model_path = Path(context["reference"]["source"]["model"]).resolve()
    audit = audit_olmoe_source(model_path)
    sustained = context["sustained_protocol"]
    if (
        audit.decision != "proceed_to_router_trace"
        or audit.resolved_revision != sustained["source_revision"]
        or audit.config_sha256 != sustained["source_config_sha256"]
        or audit.index_sha256 != sustained["source_index_sha256"]
    ):
        raise ValueError("head-wise dense teacher source changed")
    torch.set_num_threads(_THREADS)
    loaded = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    loaded.to(device)
    maps: list[np.ndarray] = []
    started = time.perf_counter()
    try:
        for sequence_index in protocol["selection_sequence_indices"]:
            token_ids = context["input_ids"][int(sequence_index)][:-1]
            with torch.inference_mode():
                output = loaded(
                    input_ids=torch.tensor(
                        [token_ids],
                        dtype=torch.long,
                        device=device,
                    ),
                    use_cache=False,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict=True,
                )
            if output.attentions is None or len(output.attentions) != _LAYERS:
                raise ValueError("dense teacher returned no complete attentions")
            layers: list[np.ndarray] = []
            for attention in output.attentions:
                value = (
                    attention.detach().float().cpu().numpy().astype(
                        np.float32,
                        copy=False,
                    )
                )
                if value.shape != (
                    1,
                    _HEADS,
                    _POSITIONS_PER_SEQUENCE,
                    _POSITIONS_PER_SEQUENCE,
                ):
                    raise ValueError(
                        "dense teacher attention tensor shape is invalid"
                    )
                layers.append(value[0])
            maps.append(np.stack(layers, axis=0))
            del output
    finally:
        del loaded
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    elapsed_seconds = time.perf_counter() - started
    attentions = np.ascontiguousarray(np.stack(maps, axis=0), dtype=np.float32)
    checks, observations = _attention_array_evidence(
        attentions,
        expected_shape=protocol["attention_array_contract"]["shape"],
    )
    _atomic_npz(arrays_path, attentions=attentions)
    arrays_hash = sha256_file(arrays_path)
    post = _common_post_authentication(
        context,
        paths,
        manifest_sha256=manifest_sha256,
        source_inventory=source_inventory,
    )
    post.update(
        {
            "trace_protocol": sha256_file(protocol_path) == protocol_hash,
            "headwise_source": sha256_file(source_path) == source_hash,
            "trace_arrays": sha256_file(arrays_path) == arrays_hash,
        }
    )
    evidence = {
        **checks,
        "exactly_two_selection_records": (
            len(maps) == _SELECTION_SEQUENCES
        ),
        "selection_indices_exact": (
            protocol["selection_sequence_indices"]
            == [
                int(row["sequence_index"])
                for row in context["split"]["selection"]
            ]
        ),
        "no_internal_record_maps": (
            len(maps) == _SELECTION_SEQUENCES
            and not (
                set(protocol["selection_sequence_indices"])
                & {
                    int(row["sequence_index"])
                    for row in context["split"]["internal_holdout"]
                }
            )
        ),
        "post_capture_authentication": all(post.values()),
    }
    report = {
        "schema_version": 1,
        "experiment": _TRACE_CAPTURE_EXPERIMENT,
        "status": (
            "dense_attention_trace_complete"
            if all(evidence.values())
            else "dense_attention_trace_invalid"
        ),
        "artifacts": {
            **_base_bindings(context),
            "trace_protocol_sha256": protocol_hash,
            "trace_arrays_sha256": arrays_hash,
            "headwise_source_sha256": source_hash,
            "headwise_source_inventory_sha256": source_inventory,
        },
        "selection_records": protocol["selection_records"],
        "selection_sequence_indices": protocol["selection_sequence_indices"],
        "internal_record_attention_maps_captured": False,
        "configuration": dict(capture),
        "attention_array": {
            **protocol["attention_array_contract"],
            "sha256": arrays_hash,
        },
        "attention_observations": observations,
        "evidence_checks": evidence,
        "evidence_passed": all(evidence.values()),
        "post_capture_authentication": post,
        "elapsed_seconds": elapsed_seconds,
    }
    atomic_json(output_path, report)
    return report


def _trace_artifacts(
    *,
    trace_protocol: Path,
    trace_protocol_sha256: str,
    trace_metadata: Path,
    trace_metadata_sha256: str,
    trace_arrays: Path,
    trace_arrays_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, dict[str, str]]:
    protocol = _read_object(trace_protocol, "head-wise trace protocol")
    metadata = _read_object(trace_metadata, "head-wise trace metadata")
    hashes = {
        "trace_protocol_sha256": sha256_file(trace_protocol),
        "trace_metadata_sha256": sha256_file(trace_metadata),
        "trace_arrays_sha256": sha256_file(trace_arrays),
    }
    split = protocol.get("record_split")
    selection = protocol.get("selection_records")
    expected_shape = [
        _SELECTION_SEQUENCES,
        _LAYERS,
        _HEADS,
        _POSITIONS_PER_SEQUENCE,
        _POSITIONS_PER_SEQUENCE,
    ]
    array_contract = protocol.get("attention_array_contract")
    derivation = protocol.get("derivation")
    metadata_artifacts = metadata.get("artifacts")
    base_binding_names = (
        "package_manifest_sha256",
        "native_library_sha256",
        "dataset_sha256",
        "corpus_manifest_sha256",
        "teacher_reference_sha256",
        "teacher_arrays_sha256",
        "sustained_protocol_sha256",
        "sustained_result_sha256",
        "control_protocol_sha256",
        "control_result_sha256",
        "sweep_protocol_sha256",
        "sweep_result_sha256",
        "control_source_sha256",
        "sweep_source_sha256",
        "layer_rescue_protocol_sha256",
        "layer_rescue_result_sha256",
        "layered_native_library_sha256",
        "layer_rescue_source_sha256",
        "layer_rescue_source_inventory_sha256",
        "headwise_source_sha256",
        "headwise_source_inventory_sha256",
    )
    if (
        hashes["trace_protocol_sha256"] != trace_protocol_sha256.lower()
        or hashes["trace_metadata_sha256"] != trace_metadata_sha256.lower()
        or hashes["trace_arrays_sha256"] != trace_arrays_sha256.lower()
        or protocol.get("schema_version") != 1
        or protocol.get("experiment") != _TRACE_PROTOCOL_EXPERIMENT
        or protocol.get("status") != _TRACE_PROTOCOL_STATUS
        or metadata.get("schema_version") != 1
        or metadata.get("experiment") != _TRACE_CAPTURE_EXPERIMENT
        or metadata.get("status") != "dense_attention_trace_complete"
        or metadata.get("evidence_passed") is not True
        or not isinstance(metadata.get("evidence_checks"), dict)
        or not metadata["evidence_checks"]
        or not all(
            value is True for value in metadata["evidence_checks"].values()
        )
        or metadata.get("internal_record_attention_maps_captured") is not False
        or metadata.get("selection_records") != protocol.get("selection_records")
        or metadata.get("selection_sequence_indices")
        != protocol.get("selection_sequence_indices")
        or metadata.get("artifacts", {}).get("trace_protocol_sha256")
        != hashes["trace_protocol_sha256"]
        or metadata.get("artifacts", {}).get("trace_arrays_sha256")
        != hashes["trace_arrays_sha256"]
        or not isinstance(metadata_artifacts, dict)
        or any(
            metadata_artifacts.get(name) != protocol.get(name)
            for name in base_binding_names
        )
        or metadata.get("attention_array", {}).get("sha256")
        != hashes["trace_arrays_sha256"]
        or not isinstance(split, dict)
        or protocol.get("record_split_identity") != split.get("split_identity")
        or selection != split.get("selection")
        or not isinstance(selection, list)
        or len(selection) != _SELECTION_SEQUENCES
        or not isinstance(split.get("internal_holdout"), list)
        or len(split["internal_holdout"]) != _INTERNAL_SEQUENCES
        or protocol.get("selection_sequence_indices")
        != [int(row["sequence_index"]) for row in selection]
        or not isinstance(array_contract, dict)
        or array_contract.get("key") != "attentions"
        or array_contract.get("shape") != expected_shape
        or array_contract.get("dtype") != "float32"
        or array_contract.get("layout")
        != "selection_sequence_layer_head_query_key"
        or metadata.get("attention_array", {}).get("shape") != expected_shape
        or metadata.get("attention_array", {}).get("dtype") != "float32"
        or not isinstance(derivation, dict)
        or derivation.get("stable_ranking")
        != [
            "descending_total_deficit",
            "descending_minimum_sequence_band_mean_deficit",
            "ascending_layer",
            "ascending_head",
        ]
        or derivation.get("selected_prefix") != _RESCUED_HEADS
        or protocol.get("derivation", {}).get("maps_from_internal_screen_records")
        is not False
        or not all(
            value is True
            for value in metadata.get("post_capture_authentication", {}).values()
        )
    ):
        raise ValueError("head-wise trace artifact chain is invalid")
    source_inventory = _validate_historical_source_inventory(
        protocol.get("headwise_source_inventory_sha256")
    )
    if (
        protocol.get("headwise_source_sha256")
        != source_inventory.get(
            "src/engram/evaluation/olmoe_native_headwise.py"
        )
        or not _current_inventory_matches(source_inventory)
    ):
        raise ValueError("head-wise trace source authentication failed")
    with np.load(trace_arrays, allow_pickle=False) as arrays:
        if set(arrays.files) != {"attentions"}:
            raise ValueError("head-wise trace arrays have unexpected keys")
        attentions = np.ascontiguousarray(arrays["attentions"])
    checks, _observations = _attention_array_evidence(
        attentions,
        expected_shape=protocol["attention_array_contract"]["shape"],
    )
    if not all(checks.values()):
        raise ValueError("head-wise trace attention arrays are invalid")
    return protocol, metadata, attentions, hashes


def _derive_head_scores(
    attentions: np.ndarray,
    *,
    local_window: int = _LOCAL_WINDOW,
    top_k: int = _TOP_OLDER,
    bands: Sequence[tuple[str, int, int]] = _QUALITY_BANDS,
) -> list[dict[str, Any]]:
    """Compute the frozen dense-attention deficit and stable head ranking."""

    if (
        attentions.ndim != 5
        or attentions.shape[0] <= 0
        or attentions.shape[1] <= 0
        or attentions.shape[2] <= 0
        or attentions.shape[3] != attentions.shape[4]
        or attentions.dtype != np.float32
        or local_window <= 0
        or top_k <= 0
        or attentions.shape[-1] <= local_window
    ):
        raise ValueError("head-wise score attention tensor is invalid")
    sequence_count, layers, heads, positions, _keys = attentions.shape
    active_bands = [
        (name, max(start, local_window), min(stop, positions))
        for name, start, stop in bands
        if min(stop, positions) > max(start, local_window)
    ]
    if not active_bands:
        raise ValueError("head-wise score has no non-prefix bands")
    scores: list[dict[str, Any]] = []
    for layer in range(layers):
        for head in range(heads):
            deficits = np.empty(
                (sequence_count, positions - local_window),
                dtype=np.float64,
            )
            for sequence in range(sequence_count):
                for query in range(local_window, positions):
                    # k <= p-W, so the exclusive stop is p-W+1.
                    older = attentions[
                        sequence,
                        layer,
                        head,
                        query,
                        : query - local_window + 1,
                    ]
                    ordered = np.sort(older, kind="stable")
                    largest = ordered[-min(top_k, ordered.size) :]
                    deficits[sequence, query - local_window] = (
                        np.sum(older, dtype=np.float64)
                        - np.sum(largest, dtype=np.float64)
                    )
            sequence_band_means: list[dict[str, Any]] = []
            for sequence in range(sequence_count):
                for name, start, stop in active_bands:
                    values = deficits[
                        sequence,
                        start - local_window : stop - local_window,
                    ]
                    sequence_band_means.append(
                        {
                            "selection_sequence_ordinal": sequence,
                            "band": name,
                            "mean_deficit": float(
                                np.mean(values, dtype=np.float64)
                            ),
                        }
                    )
            total = float(np.sum(deficits, dtype=np.float64))
            minimum = float(
                min(row["mean_deficit"] for row in sequence_band_means)
            )
            if not math.isfinite(total) or not math.isfinite(minimum):
                raise ValueError("head-wise score is non-finite")
            scores.append(
                {
                    "layer": layer,
                    "head": head,
                    "total_deficit": total,
                    "minimum_sequence_band_mean_deficit": minimum,
                    "sequence_band_mean_deficits": sequence_band_means,
                }
            )
    ranked = sorted(
        scores,
        key=lambda row: (
            -float(row["total_deficit"]),
            -float(row["minimum_sequence_band_mean_deficit"]),
            int(row["layer"]),
            int(row["head"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _mask_from_ranking(
    ranking: Sequence[Mapping[str, Any]],
    *,
    layers: int = _LAYERS,
    heads: int = _HEADS,
    selected_count: int = _RESCUED_HEADS,
) -> tuple[list[tuple[int, int]], list[list[bool]]]:
    if (
        len(ranking) != layers * heads
        or selected_count <= 0
        or selected_count > len(ranking)
    ):
        raise ValueError("head-wise ranking size is invalid")
    selected = [
        (int(row["layer"]), int(row["head"]))
        for row in ranking[:selected_count]
    ]
    if (
        len(set(selected)) != selected_count
        or any(
            layer < 0
            or layer >= layers
            or head < 0
            or head >= heads
            for layer, head in selected
        )
    ):
        raise ValueError("head-wise ranking prefix is invalid")
    selected_set = set(selected)
    mask = [
        [(layer, head) in selected_set for head in range(heads)]
        for layer in range(layers)
    ]
    return selected, mask


def _build_mask_result(
    protocol: Mapping[str, Any],
    metadata: Mapping[str, Any],
    attentions: np.ndarray,
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    ranking = _derive_head_scores(attentions)
    selected, mask = _mask_from_ranking(ranking)
    selected_rows = [
        {
            "rank": rank,
            "layer": layer,
            "head": head,
            "layer_major_index": layer * _HEADS + head,
        }
        for rank, (layer, head) in enumerate(selected, start=1)
    ]
    mask_identity = sha256_json(mask)
    evidence_checks = {
        "trace_evidence": metadata["evidence_passed"] is True,
        "exactly_two_selection_maps": (
            attentions.shape[0] == _SELECTION_SEQUENCES
        ),
        "no_internal_record_maps": (
            metadata["internal_record_attention_maps_captured"] is False
        ),
        "all_heads_ranked_once": (
            len(ranking) == _TOTAL_HEADS
            and len(
                {
                    (int(row["layer"]), int(row["head"]))
                    for row in ranking
                }
            )
            == _TOTAL_HEADS
        ),
        "exactly_51_heads_selected": (
            sum(sum(row) for row in mask) == _RESCUED_HEADS
        ),
        "within_45_percent_attention_cap": (
            protocol["budget_contract"]["attention_logical_read_fraction"]
            <= _THRESHOLDS["maximum_attention_logical_read_fraction"]
        ),
        "52_heads_exceeds_attention_cap": (
            protocol["budget_contract"]["next_head_boundary"][
                "attention_logical_read_fraction"
            ]
            > _THRESHOLDS["maximum_attention_logical_read_fraction"]
        ),
    }
    post = {
        "trace_protocol_identity": _is_sha256(
            hashes["trace_protocol_sha256"]
        ),
        "trace_metadata_identity": _is_sha256(
            hashes["trace_metadata_sha256"]
        ),
        "trace_arrays_identity": _is_sha256(hashes["trace_arrays_sha256"]),
        "trace_source_inventory": _current_inventory_matches(
            protocol["headwise_source_inventory_sha256"]
        ),
        "trace_capture_post_authentication": all(
            metadata["post_capture_authentication"].values()
        ),
    }
    return {
        "schema_version": 1,
        "experiment": _MASK_EXPERIMENT,
        "status": "teacher_guided_head_mask_derived",
        "artifacts": {
            **dict(hashes),
            "headwise_source_sha256": protocol["headwise_source_sha256"],
            "headwise_source_inventory_sha256": protocol[
                "headwise_source_inventory_sha256"
            ],
            "layer_rescue_protocol_sha256": protocol[
                "layer_rescue_protocol_sha256"
            ],
            "layer_rescue_result_sha256": protocol[
                "layer_rescue_result_sha256"
            ],
        },
        "selection_records": metadata["selection_records"],
        "selection_sequence_indices": metadata["selection_sequence_indices"],
        "internal_record_attention_maps_used": False,
        "derivation": protocol["derivation"],
        "ranking": ranking,
        "ranking_sha256": sha256_json(ranking),
        "selected_head_count": _RESCUED_HEADS,
        "selected_heads": selected_rows,
        "attention_head_mask": mask,
        "attention_head_mask_sha256": mask_identity,
        "budget_contract": protocol["budget_contract"],
        "evidence_checks": evidence_checks,
        "evidence_passed": all(evidence_checks.values()),
        "post_derivation_authentication": post,
        "development_selection_only": True,
        "fresh_eight_sequence_confirmation_required_after_screen_pass": True,
    }


def derive_native_olmoe_headwise_mask(
    *,
    trace_protocol: str | Path,
    trace_protocol_sha256: str,
    trace_metadata: str | Path,
    trace_metadata_sha256: str,
    trace_arrays: str | Path,
    trace_arrays_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    """Derive and persist the only mask allowed by the frozen trace protocol."""

    output_path = Path(out).expanduser().resolve()
    if output_path.exists():
        raise ValueError("head-wise mask target already exists")
    protocol_path = Path(trace_protocol).expanduser().resolve()
    metadata_path = Path(trace_metadata).expanduser().resolve()
    arrays_path = Path(trace_arrays).expanduser().resolve()
    protocol, metadata, attentions, hashes = _trace_artifacts(
        trace_protocol=protocol_path,
        trace_protocol_sha256=trace_protocol_sha256,
        trace_metadata=metadata_path,
        trace_metadata_sha256=trace_metadata_sha256,
        trace_arrays=arrays_path,
        trace_arrays_sha256=trace_arrays_sha256,
    )
    result = _build_mask_result(protocol, metadata, attentions, hashes)
    if not all(result["evidence_checks"].values()):
        raise ValueError("head-wise mask derivation evidence is invalid")
    atomic_json(output_path, result)
    return result


def _validated_mask_artifacts(
    *,
    context: Mapping[str, Any],
    source_sha256: str,
    source_inventory: Mapping[str, str],
    trace_protocol: Path,
    trace_protocol_sha256: str,
    trace_metadata: Path,
    trace_metadata_sha256: str,
    trace_arrays: Path,
    trace_arrays_sha256: str,
    head_mask: Path,
    head_mask_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    np.ndarray,
    dict[str, str],
]:
    protocol, metadata, attentions, hashes = _trace_artifacts(
        trace_protocol=trace_protocol,
        trace_protocol_sha256=trace_protocol_sha256,
        trace_metadata=trace_metadata,
        trace_metadata_sha256=trace_metadata_sha256,
        trace_arrays=trace_arrays,
        trace_arrays_sha256=trace_arrays_sha256,
    )
    _validate_trace_protocol(
        protocol,
        context,
        protocol_sha256=hashes["trace_protocol_sha256"],
        supplied_sha256=trace_protocol_sha256,
        source_sha256=source_sha256,
        source_inventory=source_inventory,
    )
    actual_mask_hash = sha256_file(head_mask)
    if actual_mask_hash != head_mask_sha256.lower():
        raise ValueError("head-wise mask hash is invalid")
    mask = _read_object(head_mask, "head-wise mask")
    expected = _build_mask_result(protocol, metadata, attentions, hashes)
    if (
        mask != expected
        or mask.get("status") != "teacher_guided_head_mask_derived"
        or mask.get("evidence_passed") is not True
        or not all(mask.get("evidence_checks", {}).values())
        or not all(
            mask.get("post_derivation_authentication", {}).values()
        )
    ):
        raise ValueError("head-wise mask artifact is invalid")
    hashes = {**hashes, "head_mask_sha256": actual_mask_hash}
    return protocol, metadata, mask, attentions, hashes


def _selected_pairs(mask: Mapping[str, Any]) -> list[tuple[int, int]]:
    rows = mask.get("selected_heads")
    nested = mask.get("attention_head_mask")
    if (
        not isinstance(rows, list)
        or len(rows) != _RESCUED_HEADS
        or not isinstance(nested, list)
        or len(nested) != _LAYERS
        or any(not isinstance(row, list) or len(row) != _HEADS for row in nested)
        or any(
            not isinstance(value, bool)
            for layer in nested
            for value in layer
        )
    ):
        raise ValueError("head-wise selected mask shape is invalid")
    pairs: list[tuple[int, int]] = []
    for rank, row in enumerate(rows, start=1):
        if (
            not isinstance(row, dict)
            or isinstance(row.get("rank"), bool)
            or not isinstance(row.get("rank"), int)
            or row.get("rank") != rank
            or isinstance(row.get("layer"), bool)
            or not isinstance(row.get("layer"), int)
            or isinstance(row.get("head"), bool)
            or not isinstance(row.get("head"), int)
        ):
            raise ValueError("head-wise selected mask row is invalid")
        layer = int(row["layer"])
        head = int(row["head"])
        if (
            layer < 0
            or layer >= _LAYERS
            or head < 0
            or head >= _HEADS
            or row.get("layer_major_index") != layer * _HEADS + head
        ):
            raise ValueError("head-wise selected mask coordinate is invalid")
        pairs.append((layer, head))
    expected_nested = [
        [(layer, head) in set(pairs) for head in range(_HEADS)]
        for layer in range(_LAYERS)
    ]
    if (
        len(set(pairs)) != _RESCUED_HEADS
        or nested != expected_nested
        or sha256_json(nested) != mask.get("attention_head_mask_sha256")
    ):
        raise ValueError("head-wise selected mask identity is invalid")
    return pairs


def _population_contract(sequence_count: int) -> dict[str, int]:
    return layer_rescue._population_contract(sequence_count)


def _build_screen_protocol(
    context: Mapping[str, Any],
    *,
    trace_protocol: Mapping[str, Any],
    mask: Mapping[str, Any],
    trace_hashes: Mapping[str, str],
    candidate_library_sha256: str,
    source_sha256: str,
    source_inventory: Mapping[str, str],
) -> dict[str, Any]:
    selected = _selected_pairs(mask)
    holdout = list(context["split"]["internal_holdout"])
    holdout_indices = [int(row["sequence_index"]) for row in holdout]
    parity_sequence = int(context["split"]["selection"][0]["sequence_index"])
    return {
        "schema_version": 1,
        "experiment": _SCREEN_EXPERIMENT,
        "status": _SCREEN_PROTOCOL_STATUS,
        **_base_bindings(context),
        **dict(trace_hashes),
        "candidate_native_library_sha256": candidate_library_sha256,
        "headwise_source_sha256": source_sha256,
        "headwise_source_inventory_sha256": dict(source_inventory),
        "head_mask_identity_sha256": mask["attention_head_mask_sha256"],
        "selected_heads": mask["selected_heads"],
        "attention_head_mask": mask["attention_head_mask"],
        "selected_head_count": len(selected),
        "budget_contract": _headwise_budget_contract(context["model"]),
        "analytical_byte_components": {
            "attention_local_kv_bytes": _EXPECTED_HEADWISE[
                "attention_local_kv_bytes"
            ],
            "attention_candidate_key_bytes": _EXPECTED_HEADWISE[
                "attention_candidate_key_bytes"
            ],
            "attention_selected_value_bytes": _EXPECTED_HEADWISE[
                "attention_selected_value_bytes"
            ],
            "runtime_observes_only_total_logical_read_bytes": True,
        },
        "model": context["model"],
        "q7_expectations_per_sequence": context["q7_expectations"],
        "quality_bands": sweep_source._expected_bands(),
        "thresholds": _THRESHOLDS,
        "record_split": context["split"],
        "record_split_identity": context["split"]["split_identity"],
        "internal_screen_records": holdout,
        "internal_screen_sequence_indices": holdout_indices,
        "population_contract": _population_contract(_INTERNAL_SEQUENCES),
        "all_base_parity": {
            "required_before_candidate_execution": True,
            "sequence_index": parity_sequence,
            "record_id": context["record_ids"][parity_sequence],
            "layered_reference_library_sha256": context[
                "candidate_library_sha256"
            ],
            "headwise_candidate_library_sha256": candidate_library_sha256,
            "layered_policy": [
                dict(_BASE_POLICY) for _layer in range(_LAYERS)
            ],
            "headwise_policy": _head_policies([]),
            "exact_outputs": [
                "next_tokens",
                "final_hidden_states",
                "vocabulary_logits",
                "cache_positions",
            ],
            "exact_counter_parity": [
                "positions_processed",
                "attention_weight_bytes",
                "attention_logical_read_bytes",
                "attention_older_candidate_entries_scored",
                "attention_older_selected_entries",
                "attention_sink_insertions",
                "attention_heavy_hitter_updates",
                "q7_scheduled_bytes",
            ],
            "headwise_eviction_events_equal_layered_times_query_heads": True,
            "state_and_scratch_require_separate_analytical_contracts": True,
            "layered_expectations_per_sequence": (
                layer_rescue._schedule_expectations(
                    context["model"],
                    [],
                )
            ),
            "headwise_expectations_per_sequence": _headwise_expectations(
                context["model"],
                [],
            ),
        },
        "scope": {
            "candidate_device": "cpu",
            "candidate_threads": _THREADS,
            "candidate_transformers_model_shell": False,
            "execution_interface": _EXECUTION_INTERFACE,
            "candidate_count": 1,
            "candidate_mask_fixed_before_screen_protocol": True,
            "candidate_mask_adaptation_after_freeze": False,
            "internal_screen_sequences": _INTERNAL_SEQUENCES,
            "attention_maps_from_internal_screen_records": False,
            "internal_outputs_unseen_during_mask_selection": True,
            "primary_schedule_executions": 1,
            "reset_replay_sequence_index": holdout_indices[0],
            "reset_replay_excluded_from_semantic_metrics": True,
            "q7_artifact_or_policy_changed": False,
            "package_manifest_mutated": False,
            "development_screen_only": True,
            "fresh_eight_sequence_confirmation_required_after_pass": True,
        },
        "decision_rule": {
            "all_base_parity_failure": (
                "persist an invalid zero-candidate result and stop"
            ),
            "evidence_failure": "stop_and_diagnose_headwise_evidence",
            "authenticated_quality_pass": (
                "freeze a fresh sealed eight-sequence package-native confirmation"
            ),
            "authenticated_quality_failure": (
                "investigate value/sensitivity-guided or dynamic head allocation"
            ),
        },
        "provenance": {
            "dense_trace_protocol_frozen_before_maps": (
                trace_protocol["provenance"][
                    "protocol_frozen_before_attention_map_capture"
                ]
            ),
            "mask_derived_only_from_two_selection_records": True,
            "screen_protocol_frozen_after_mask_and_native_library": True,
            "screen_protocol_frozen_before_parity_or_candidate_execution": True,
            "six_internal_records_cannot_change_mask": True,
        },
        "limitations": [
            "The fixed mask was selected from two records and this six-record evaluation is only an internal development screen.",
            "The screen reuses records previously consumed by other diagnostics; it is not a fresh confirmation.",
            "A pass requires a fresh sealed eight-sequence confirmation under an integrated package policy.",
            "Logical attention bytes are analytical native reads, not measured hardware DRAM traffic.",
            "The v1 head-wise runtime is restricted to equal query-head and KV-head counts.",
            "W128 is full-context only over this 128-position protocol; beyond it, rescued heads remain bounded W128/C8/K4/S2 rather than dense.",
        ],
    }


def freeze_native_olmoe_headwise_screen_protocol(
    *,
    candidate_library: str | Path,
    trace_protocol: str | Path,
    trace_protocol_sha256: str,
    trace_metadata: str | Path,
    trace_metadata_sha256: str,
    trace_arrays: str | Path,
    trace_arrays_sha256: str,
    head_mask: str | Path,
    head_mask_sha256: str,
    out: str | Path,
    manifest_sha256: str,
    threads: int = _THREADS,
    **common: Any,
) -> dict[str, Any]:
    """Freeze the one-candidate causal screen after mask and DSO exist."""

    output_path = Path(out).expanduser().resolve()
    candidate_path = Path(candidate_library).expanduser().resolve()
    if output_path.exists():
        raise ValueError("head-wise screen protocol target already exists")
    if threads != _THREADS:
        raise ValueError("head-wise screen requires 12 CPU threads")
    if not candidate_path.is_file():
        raise ValueError("head-wise candidate library is missing")
    paths, context = _common_context(
        manifest_sha256=manifest_sha256,
        **common,
    )
    candidate_hash = sha256_file(candidate_path)
    if (
        candidate_path == paths["candidate_library_path"]
        or candidate_hash == context["candidate_library_sha256"]
    ):
        raise ValueError("head-wise candidate library must be a new DSO")
    source_path = Path(__file__).resolve()
    source_inventory = _current_source_inventory(
        context["layer_rescue_historical_source_inventory"]
    )
    source_hash = sha256_file(source_path)
    trace_protocol_path = Path(trace_protocol).expanduser().resolve()
    trace_metadata_path = Path(trace_metadata).expanduser().resolve()
    trace_arrays_path = Path(trace_arrays).expanduser().resolve()
    head_mask_path = Path(head_mask).expanduser().resolve()
    (
        trace_protocol_value,
        _metadata,
        mask,
        _attentions,
        trace_hashes,
    ) = _validated_mask_artifacts(
        context=context,
        source_sha256=source_hash,
        source_inventory=source_inventory,
        trace_protocol=trace_protocol_path,
        trace_protocol_sha256=trace_protocol_sha256,
        trace_metadata=trace_metadata_path,
        trace_metadata_sha256=trace_metadata_sha256,
        trace_arrays=trace_arrays_path,
        trace_arrays_sha256=trace_arrays_sha256,
        head_mask=head_mask_path,
        head_mask_sha256=head_mask_sha256,
    )
    protocol = _build_screen_protocol(
        context,
        trace_protocol=trace_protocol_value,
        mask=mask,
        trace_hashes=trace_hashes,
        candidate_library_sha256=candidate_hash,
        source_sha256=source_hash,
        source_inventory=source_inventory,
    )
    atomic_json(output_path, protocol)
    return protocol


def _validate_screen_protocol(
    protocol: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    trace_protocol: Mapping[str, Any],
    mask: Mapping[str, Any],
    trace_hashes: Mapping[str, str],
    candidate_library_sha256: str,
    protocol_sha256: str,
    supplied_protocol_sha256: str,
    source_sha256: str,
    source_inventory: Mapping[str, str],
) -> None:
    expected = _build_screen_protocol(
        context,
        trace_protocol=trace_protocol,
        mask=mask,
        trace_hashes=trace_hashes,
        candidate_library_sha256=candidate_library_sha256,
        source_sha256=source_sha256,
        source_inventory=source_inventory,
    )
    if (
        protocol_sha256 != supplied_protocol_sha256.lower()
        or protocol != expected
    ):
        raise ValueError("head-wise screen protocol contract is invalid")


def _counter_checks(
    metrics: Mapping[str, int],
    expectations: Mapping[str, int | float],
    q7_expectations: Mapping[str, int],
    *,
    position: int,
) -> dict[str, bool]:
    checks = _sustained_structural_checks(
        dict(metrics),
        dict(expectations),
        position=position,
    )
    checks["q7_scheduled_bytes"] = (
        int(metrics.get("q7_scheduled_bytes", -1))
        == position * int(q7_expectations["scheduled_bytes_per_position"])
    )
    return checks


def _counter_digest_update(
    digest: Any,
    metrics: Mapping[str, int],
) -> None:
    digest.update(
        json.dumps(
            _deterministic_metrics(dict(metrics)),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")


def _parity_counter_checks(
    layered: Mapping[str, int],
    headwise: Mapping[str, int],
    *,
    layered_expectations: Mapping[str, int | float],
    headwise_expectations: Mapping[str, int | float],
    q7_expectations: Mapping[str, int],
    position: int,
) -> dict[str, bool]:
    exact_parity_names = (
        "positions_processed",
        "attention_weight_bytes",
        "attention_logical_read_bytes",
        "attention_older_candidate_entries_scored",
        "attention_older_selected_entries",
        "attention_sink_insertions",
        "attention_heavy_hitter_updates",
        "q7_scheduled_bytes",
    )
    checks = {
        f"exact_{name}": int(layered.get(name, -1))
        == int(headwise.get(name, -2))
        for name in exact_parity_names
    }
    checks.update(
        {
            "layered_contract": all(
                _counter_checks(
                    layered,
                    layered_expectations,
                    q7_expectations,
                    position=position,
                ).values()
            ),
            "headwise_contract": all(
                _counter_checks(
                    headwise,
                    headwise_expectations,
                    q7_expectations,
                    position=position,
                ).values()
            ),
            "headwise_evictions_equal_layered_times_query_heads": (
                int(headwise.get("attention_eviction_events", -1))
                == int(layered.get("attention_eviction_events", -2)) * _HEADS
            ),
            "layered_state_analytical": (
                int(layered.get("attention_state_bytes", -1))
                == int(layered_expectations["attention_state_bytes"])
            ),
            "headwise_state_analytical": (
                int(headwise.get("attention_state_bytes", -1))
                == int(headwise_expectations["attention_state_bytes"])
            ),
            "layered_scratch_analytical": (
                int(layered.get("attention_scratch_bytes", -1))
                == int(layered_expectations["attention_scratch_bytes"])
            ),
            "headwise_scratch_analytical": (
                int(headwise.get("attention_scratch_bytes", -1))
                == int(headwise_expectations["attention_scratch_bytes"])
            ),
        }
    )
    return checks


def _evaluate_headwise_all_base_parity(
    *,
    context: Mapping[str, Any],
    layered_library: Path,
    candidate_library: Path,
    sequence_index: int,
    threads: int,
) -> dict[str, Any]:
    """Prove that splitting each layer into 16 base engines is semantic-inert."""

    layered: OLMoENativeTokenRuntime | None = None
    headwise: OLMoENativeTokenRuntime | None = None
    token_matches: list[bool] = []
    hidden_matches: list[bool] = []
    logit_matches: list[bool] = []
    counter_checks: list[bool] = []
    cache_position_matches: list[bool] = []
    layered_hidden_hash = hashlib.sha256()
    layered_logit_hash = hashlib.sha256()
    headwise_hidden_hash = hashlib.sha256()
    headwise_logit_hash = hashlib.sha256()
    layered_counter_hash = hashlib.sha256()
    headwise_counter_hash = hashlib.sha256()
    layered_final: dict[str, int] | None = None
    headwise_final: dict[str, int] | None = None
    layered_position = 0
    headwise_position = 0
    started = time.perf_counter()
    try:
        layered = OLMoENativeTokenRuntime(
            context["config_path"],
            context["non_mlp_path"],
            context["q7_path"],
            layered_library,
            threads=threads,
            attention_policies=[
                dict(_BASE_POLICY) for _layer in range(_LAYERS)
            ],
        )
        headwise = OLMoENativeTokenRuntime(
            context["config_path"],
            context["non_mlp_path"],
            context["q7_path"],
            candidate_library,
            threads=threads,
            attention_head_policies=_head_policies([]),
        )
        if (
            not layered.attention_metrics_available
            or not headwise.attention_metrics_available
        ):
            raise ValueError("head-wise parity metric ABI is unavailable")
        for ordinal, token_id in enumerate(
            context["input_ids"][sequence_index][:-1],
            start=1,
        ):
            layered_result = layered.forward([token_id])
            headwise_result = headwise.forward([token_id])
            layered_hidden, layered_logits = layered.last_diagnostics()
            headwise_hidden, headwise_logits = headwise.last_diagnostics()
            layered_final = dict(layered_result.metrics)
            headwise_final = dict(headwise_result.metrics)
            layered_expected = layer_rescue._schedule_expectations(
                context["model"],
                [],
                positions=ordinal,
            )
            headwise_expected = _headwise_expectations(
                context["model"],
                [],
                positions=ordinal,
            )
            token_matches.append(
                layered_result.next_token == headwise_result.next_token
            )
            hidden_matches.append(
                np.array_equal(layered_hidden, headwise_hidden)
            )
            logit_matches.append(
                np.array_equal(layered_logits, headwise_logits)
            )
            counter_checks.append(
                all(
                    _parity_counter_checks(
                        layered_final,
                        headwise_final,
                        layered_expectations=layered_expected,
                        headwise_expectations=headwise_expected,
                        q7_expectations=context["q7_expectations"],
                        position=ordinal,
                    ).values()
                )
            )
            cache_position_matches.append(
                layered.position == ordinal and headwise.position == ordinal
            )
            _update_diagnostic_hashes(
                layered_hidden_hash,
                layered_logit_hash,
                layered_hidden,
                layered_logits,
            )
            _update_diagnostic_hashes(
                headwise_hidden_hash,
                headwise_logit_hash,
                headwise_hidden,
                headwise_logits,
            )
            _counter_digest_update(layered_counter_hash, layered_final)
            _counter_digest_update(headwise_counter_hash, headwise_final)
        layered_position = layered.position
        headwise_position = headwise.position
    finally:
        if layered is not None:
            layered.close()
        if headwise is not None:
            headwise.close()
    elapsed = time.perf_counter() - started
    if layered_final is None or headwise_final is None:
        raise RuntimeError("head-wise parity processed no positions")
    layered_hashes = {
        "hidden_sha256": layered_hidden_hash.hexdigest(),
        "logits_sha256": layered_logit_hash.hexdigest(),
        "counter_stream_sha256": layered_counter_hash.hexdigest(),
    }
    headwise_hashes = {
        "hidden_sha256": headwise_hidden_hash.hexdigest(),
        "logits_sha256": headwise_logit_hash.hexdigest(),
        "counter_stream_sha256": headwise_counter_hash.hexdigest(),
    }
    archived_parity = context["layer_rescue_result"][
        "layered_abi_all_base_parity"
    ]
    archived_candidate_hashes = archived_parity["candidate_hashes"]
    checks = {
        "prediction_positions": len(token_matches) == _POSITIONS_PER_SEQUENCE,
        "cache_positions": (
            layered_position == _POSITIONS_PER_SEQUENCE
            and headwise_position == _POSITIONS_PER_SEQUENCE
        ),
        "tokens_exact": all(token_matches),
        "hidden_exact": all(hidden_matches),
        "logits_exact": all(logit_matches),
        "per_position_counter_contracts": all(counter_checks),
        "per_position_cache_positions": all(cache_position_matches),
        "diagnostic_hashes_exact": (
            layered_hashes["hidden_sha256"]
            == headwise_hashes["hidden_sha256"]
            and layered_hashes["logits_sha256"]
            == headwise_hashes["logits_sha256"]
        ),
        "layered_reference_matches_archived_diagnostics": (
            layered_hashes["hidden_sha256"]
            == archived_candidate_hashes["hidden_sha256"]
            and layered_hashes["logits_sha256"]
            == archived_candidate_hashes["logits_sha256"]
        ),
        "layered_reference_matches_archived_counter_stream": (
            layered_hashes["counter_stream_sha256"]
            == archived_candidate_hashes["counter_stream_sha256"]
        ),
        "separate_counter_streams_expected": (
            layered_hashes["counter_stream_sha256"]
            != headwise_hashes["counter_stream_sha256"]
        ),
    }
    return {
        "sequence_index": sequence_index,
        "record_id": context["record_ids"][sequence_index],
        "layered_library_sha256": context["candidate_library_sha256"],
        "headwise_library_sha256": sha256_file(candidate_library),
        "layered_hashes": layered_hashes,
        "headwise_hashes": headwise_hashes,
        "layered_final_metrics": _deterministic_metrics(layered_final),
        "headwise_final_metrics": _deterministic_metrics(headwise_final),
        "checks": checks,
        "passed": all(checks.values()),
        "elapsed_seconds": elapsed,
    }


def _evaluate_headwise_candidate(
    selected_heads: Sequence[tuple[int, int]],
    *,
    sequence_indices: Sequence[int],
    context: Mapping[str, Any],
    library: Path,
    teacher_logits: np.ndarray,
    teacher_hidden: np.ndarray,
    targets: np.ndarray,
    threads: int,
    replay_sequence_index: int,
) -> dict[str, Any]:
    policies = _head_policies(selected_heads)
    expectations = _headwise_expectations(context["model"], selected_heads)
    expected_by_position = [
        _headwise_expectations(
            context["model"],
            selected_heads,
            positions=position,
        )
        for position in range(1, _POSITIONS_PER_SEQUENCE + 1)
    ]
    q7 = context["q7_expectations"]
    load_started = time.perf_counter()
    runtime = OLMoENativeTokenRuntime(
        context["config_path"],
        context["non_mlp_path"],
        context["q7_path"],
        library,
        threads=threads,
        attention_head_policies=policies,
    )
    all_rows: list[dict[str, float | bool | int]] = []
    sequence_results: list[dict[str, Any]] = []
    replay_reference: dict[str, Any] | None = None
    cold_load_seconds = time.perf_counter() - load_started
    try:
        if not runtime.attention_metrics_available:
            raise ValueError("head-wise candidate metric ABI is unavailable")
        for sequence_index in sequence_indices:
            runtime.reset()
            rows: list[dict[str, float | bool | int]] = []
            tokens: list[int] = []
            hidden_hash = hashlib.sha256()
            logit_hash = hashlib.sha256()
            counter_hash = hashlib.sha256()
            final_metrics: dict[str, int] | None = None
            counters_passed = True
            cache_positions_passed = True
            started = time.perf_counter()
            for position, token_id in enumerate(
                context["input_ids"][sequence_index][:-1]
            ):
                result = runtime.forward([token_id])
                native_hidden, native_logits = runtime.last_diagnostics()
                if int(np.argmax(native_logits)) != result.next_token:
                    raise ValueError("head-wise candidate diagnostic argmax differs")
                final_metrics = dict(result.metrics)
                ordinal = position + 1
                cache_positions_passed = (
                    cache_positions_passed and runtime.position == ordinal
                )
                counters_passed = counters_passed and all(
                    _counter_checks(
                        final_metrics,
                        expected_by_position[position],
                        q7,
                        position=ordinal,
                    ).values()
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
                    {"sequence_index": sequence_index, "position": position}
                )
                rows.append(row)
                tokens.append(result.next_token)
                _update_diagnostic_hashes(
                    hidden_hash,
                    logit_hash,
                    native_hidden,
                    native_logits,
                )
                _counter_digest_update(counter_hash, final_metrics)
            elapsed = time.perf_counter() - started
            if final_metrics is None:
                raise RuntimeError("head-wise candidate processed no positions")
            sequence_result = {
                "sequence_index": sequence_index,
                "record_id": context["record_ids"][sequence_index],
                "prediction_positions": len(rows),
                "metrics": layer_rescue._aggregate(rows),
                "final_native_metrics": _deterministic_metrics(final_metrics),
                "final_counter_checks": _counter_checks(
                    final_metrics,
                    expectations,
                    q7,
                    position=_POSITIONS_PER_SEQUENCE,
                ),
                "per_token_counter_checks_passed": counters_passed,
                "cache_positions_passed": (
                    cache_positions_passed
                    and runtime.position == _POSITIONS_PER_SEQUENCE
                ),
                "top1_sha256": sha256_json(tokens),
                "hidden_sha256": hidden_hash.hexdigest(),
                "logits_sha256": logit_hash.hexdigest(),
                "counter_stream_sha256": counter_hash.hexdigest(),
                "elapsed_seconds": elapsed,
            }
            sequence_results.append(sequence_result)
            all_rows.extend(rows)
            if sequence_index == replay_sequence_index:
                replay_reference = {
                    key: sequence_result[key]
                    for key in (
                        "sequence_index",
                        "top1_sha256",
                        "hidden_sha256",
                        "logits_sha256",
                        "counter_stream_sha256",
                        "final_native_metrics",
                    )
                }
        if replay_reference is None:
            raise ValueError("head-wise replay sequence is not in screen")
        runtime.reset()
        replay_tokens: list[int] = []
        replay_hidden_hash = hashlib.sha256()
        replay_logit_hash = hashlib.sha256()
        replay_counter_hash = hashlib.sha256()
        replay_metrics: dict[str, int] | None = None
        replay_counters_passed = True
        replay_cache_positions_passed = True
        replay_started = time.perf_counter()
        for position, token_id in enumerate(
            context["input_ids"][replay_sequence_index][:-1]
        ):
            result = runtime.forward([token_id])
            native_hidden, native_logits = runtime.last_diagnostics()
            replay_metrics = dict(result.metrics)
            ordinal = position + 1
            replay_cache_positions_passed = (
                replay_cache_positions_passed and runtime.position == ordinal
            )
            replay_counters_passed = replay_counters_passed and all(
                _counter_checks(
                    replay_metrics,
                    expected_by_position[position],
                    q7,
                    position=ordinal,
                ).values()
            )
            replay_tokens.append(result.next_token)
            _update_diagnostic_hashes(
                replay_hidden_hash,
                replay_logit_hash,
                native_hidden,
                native_logits,
            )
            _counter_digest_update(replay_counter_hash, replay_metrics)
        replay_seconds = time.perf_counter() - replay_started
        if replay_metrics is None:
            raise RuntimeError("head-wise reset replay processed no positions")
        replay_value = {
            "sequence_index": replay_sequence_index,
            "top1_sha256": sha256_json(replay_tokens),
            "hidden_sha256": replay_hidden_hash.hexdigest(),
            "logits_sha256": replay_logit_hash.hexdigest(),
            "counter_stream_sha256": replay_counter_hash.hexdigest(),
            "final_native_metrics": _deterministic_metrics(replay_metrics),
        }
        reset_replay = {
            "sequence_index": replay_sequence_index,
            "reference": replay_reference,
            "replay": replay_value,
            "per_token_counter_checks_passed": replay_counters_passed,
            "cache_positions_passed": (
                replay_cache_positions_passed
                and runtime.position == _POSITIONS_PER_SEQUENCE
            ),
            "passed": (
                replay_value == replay_reference
                and replay_counters_passed
                and replay_cache_positions_passed
                and runtime.position == _POSITIONS_PER_SEQUENCE
            ),
            "elapsed_seconds": replay_seconds,
        }
    finally:
        runtime.close()
    overall = layer_rescue._aggregate(all_rows)
    bands = layer_rescue._bands_from_rows(all_rows)
    quality_checks = _quality_checks("overall", overall)
    for name, metrics in bands.items():
        quality_checks.update(_quality_checks(name, metrics))
    population_contract = _population_contract(len(sequence_indices))
    actual_populations = {
        "overall": int(overall["prediction_positions"]),
        **{
            name: int(metrics["prediction_positions"])
            for name, metrics in bands.items()
        },
    }
    evidence_checks = {
        "sequence_count": len(sequence_results) == len(sequence_indices),
        "prediction_grid": layer_rescue._position_grid_is_exact(
            all_rows,
            list(sequence_indices),
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
        "cache_positions": (
            all(row["cache_positions_passed"] for row in sequence_results)
            and reset_replay["cache_positions_passed"]
        ),
        "q7_policy_unchanged": all(
            row["final_native_metrics"].get("q7_scheduled_bytes")
            == q7["scheduled_bytes_per_sequence"]
            for row in sequence_results
        ),
        "attention_logical_read_fraction": (
            expectations["attention_logical_read_fraction"]
            == _EXPECTED_FRACTION
            and expectations["attention_logical_read_fraction"]
            <= _THRESHOLDS["maximum_attention_logical_read_fraction"]
        ),
        "reset_replay": reset_replay["passed"],
    }
    return {
        "split": "internal_screen",
        "sequence_indices": list(sequence_indices),
        "record_ids": [
            context["record_ids"][index] for index in sequence_indices
        ],
        "selected_heads": [
            {"layer": layer, "head": head} for layer, head in selected_heads
        ],
        "attention_head_policies": policies,
        "attention_expectations_per_sequence": expectations,
        "q7_expectations_per_sequence": q7,
        "metrics": overall,
        "position_bands": bands,
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
            "reset_replay_seconds": reset_replay["elapsed_seconds"],
        },
    }


def _screen_post_authentication(
    context: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    manifest_sha256: str,
    source_inventory: Mapping[str, str],
    candidate_library: Path,
    candidate_library_sha256: str,
    trace_protocol: Path,
    trace_metadata: Path,
    trace_arrays: Path,
    head_mask: Path,
    trace_hashes: Mapping[str, str],
    screen_protocol: Path,
    screen_protocol_sha256: str,
) -> dict[str, bool]:
    checks = _common_post_authentication(
        context,
        paths,
        manifest_sha256=manifest_sha256,
        source_inventory=source_inventory,
    )
    checks.update(
        {
            "candidate_library": (
                sha256_file(candidate_library) == candidate_library_sha256
            ),
            "trace_protocol": (
                sha256_file(trace_protocol)
                == trace_hashes["trace_protocol_sha256"]
            ),
            "trace_metadata": (
                sha256_file(trace_metadata)
                == trace_hashes["trace_metadata_sha256"]
            ),
            "trace_arrays": (
                sha256_file(trace_arrays)
                == trace_hashes["trace_arrays_sha256"]
            ),
            "head_mask": (
                sha256_file(head_mask) == trace_hashes["head_mask_sha256"]
            ),
            "screen_protocol": (
                sha256_file(screen_protocol) == screen_protocol_sha256
            ),
            "headwise_source": (
                sha256_file(Path(__file__).resolve())
                == source_inventory[
                    "src/engram/evaluation/olmoe_native_headwise.py"
                ]
            ),
        }
    )
    return checks


def _load_teacher_arrays(
    context: Mapping[str, Any],
    arrays_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"logits", "hidden", "targets"}:
            raise ValueError("head-wise teacher arrays have unexpected keys")
        logits = np.asarray(arrays["logits"], dtype=np.float32)
        hidden = np.asarray(arrays["hidden"], dtype=np.float32)
        targets = np.asarray(arrays["targets"], dtype=np.int64)
    prediction_positions = 8 * _POSITIONS_PER_SEQUENCE
    model = context["model"]
    expected_targets = np.asarray(
        [
            token
            for sequence in context["input_ids"]
            for token in sequence[1:]
        ],
        dtype=np.int64,
    )
    if (
        logits.shape != (prediction_positions, int(model["vocab_size"]))
        or hidden.shape != (
            prediction_positions,
            int(model["hidden_size"]),
        )
        or targets.shape != (prediction_positions,)
        or not np.array_equal(targets, expected_targets)
    ):
        raise ValueError("head-wise teacher array shape or targets are invalid")
    return logits, hidden, targets


def _invalid_parity_report(
    *,
    context: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    candidate_library_sha256: str,
    trace_hashes: Mapping[str, str],
    parity: Mapping[str, Any],
    post: Mapping[str, bool],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": _SCREEN_EXPERIMENT,
        "status": "headwise_screen_invalid",
        "provenance": {
            "screen_protocol_frozen_before_execution": True,
            "mask_fixed_before_screen_protocol": True,
            "parity_failed_before_candidate_execution": True,
            "six_screen_records_previously_consumed_by_diagnostics": True,
            "scientific_role": "invalid execution; no candidate screen",
        },
        "artifacts": {
            **_base_bindings(context),
            **dict(trace_hashes),
            "screen_protocol_sha256": protocol_sha256,
            "candidate_native_library_sha256": candidate_library_sha256,
            "headwise_source_sha256": protocol["headwise_source_sha256"],
            "headwise_source_inventory_sha256": protocol[
                "headwise_source_inventory_sha256"
            ],
        },
        "record_split": protocol["record_split"],
        "head_mask_identity_sha256": protocol["head_mask_identity_sha256"],
        "all_base_headwise_parity": dict(parity),
        "candidate_evaluation_count": 0,
        "internal_screen_result": None,
        "evidence_checks": {
            "all_base_headwise_parity": False,
            "no_candidate_outputs_inspected": True,
            "post_run_authentication": all(post.values()),
        },
        "evidence_passed": False,
        "internal_screen_quality_passed": False,
        "fresh_eight_sequence_confirmation_required": False,
        "decision": "stop_and_diagnose_headwise_all_base_parity",
        "post_run_authentication": dict(post),
        "performance": {
            "execution_seconds": float(parity["elapsed_seconds"]),
            "all_base_parity_seconds": float(parity["elapsed_seconds"]),
            "candidate_primary_sequence_seconds": 0.0,
            "candidate_reset_replay_seconds": 0.0,
        },
        "limitations": protocol["limitations"],
    }


def evaluate_native_olmoe_headwise_screen(
    *,
    candidate_library: str | Path,
    trace_protocol: str | Path,
    trace_protocol_sha256: str,
    trace_metadata: str | Path,
    trace_metadata_sha256: str,
    trace_arrays: str | Path,
    trace_arrays_sha256: str,
    head_mask: str | Path,
    head_mask_sha256: str,
    screen_protocol: str | Path,
    screen_protocol_sha256: str,
    out: str | Path,
    manifest_sha256: str,
    threads: int = _THREADS,
    **common: Any,
) -> dict[str, Any]:
    """Execute parity, then exactly one fixed-mask six-record causal screen."""

    output_path = Path(out).expanduser().resolve()
    if output_path.exists():
        raise ValueError("head-wise screen result target already exists")
    if threads != _THREADS:
        raise ValueError("head-wise screen requires 12 CPU threads")
    candidate_path = Path(candidate_library).expanduser().resolve()
    paths, context = _common_context(
        manifest_sha256=manifest_sha256,
        **common,
    )
    candidate_hash = sha256_file(candidate_path)
    source_path = Path(__file__).resolve()
    source_hash = sha256_file(source_path)
    source_inventory = _current_source_inventory(
        context["layer_rescue_historical_source_inventory"]
    )
    trace_protocol_path = Path(trace_protocol).expanduser().resolve()
    trace_metadata_path = Path(trace_metadata).expanduser().resolve()
    trace_arrays_path = Path(trace_arrays).expanduser().resolve()
    head_mask_path = Path(head_mask).expanduser().resolve()
    (
        trace_protocol_value,
        _metadata,
        mask,
        _attentions,
        trace_hashes,
    ) = _validated_mask_artifacts(
        context=context,
        source_sha256=source_hash,
        source_inventory=source_inventory,
        trace_protocol=trace_protocol_path,
        trace_protocol_sha256=trace_protocol_sha256,
        trace_metadata=trace_metadata_path,
        trace_metadata_sha256=trace_metadata_sha256,
        trace_arrays=trace_arrays_path,
        trace_arrays_sha256=trace_arrays_sha256,
        head_mask=head_mask_path,
        head_mask_sha256=head_mask_sha256,
    )
    protocol_path = Path(screen_protocol).expanduser().resolve()
    protocol = _read_object(protocol_path, "head-wise screen protocol")
    protocol_hash = sha256_file(protocol_path)
    _validate_screen_protocol(
        protocol,
        context,
        trace_protocol=trace_protocol_value,
        mask=mask,
        trace_hashes=trace_hashes,
        candidate_library_sha256=candidate_hash,
        protocol_sha256=protocol_hash,
        supplied_protocol_sha256=screen_protocol_sha256,
        source_sha256=source_hash,
        source_inventory=source_inventory,
    )
    started = time.perf_counter()
    parity_started = time.perf_counter()
    try:
        parity = _evaluate_headwise_all_base_parity(
            context=context,
            layered_library=paths["candidate_library_path"],
            candidate_library=candidate_path,
            sequence_index=protocol["all_base_parity"]["sequence_index"],
            threads=threads,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parity = {
            "sequence_index": protocol["all_base_parity"]["sequence_index"],
            "record_id": protocol["all_base_parity"]["record_id"],
            "layered_library_sha256": context["candidate_library_sha256"],
            "headwise_library_sha256": candidate_hash,
            "checks": {
                "runtime_open_and_execution": False,
            },
            "passed": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "elapsed_seconds": time.perf_counter() - parity_started,
        }
    if not parity["passed"]:
        post = _screen_post_authentication(
            context,
            paths,
            manifest_sha256=manifest_sha256,
            source_inventory=source_inventory,
            candidate_library=candidate_path,
            candidate_library_sha256=candidate_hash,
            trace_protocol=trace_protocol_path,
            trace_metadata=trace_metadata_path,
            trace_arrays=trace_arrays_path,
            head_mask=head_mask_path,
            trace_hashes=trace_hashes,
            screen_protocol=protocol_path,
            screen_protocol_sha256=protocol_hash,
        )
        report = _invalid_parity_report(
            context=context,
            protocol=protocol,
            protocol_sha256=protocol_hash,
            candidate_library_sha256=candidate_hash,
            trace_hashes=trace_hashes,
            parity=parity,
            post=post,
        )
        atomic_json(output_path, report)
        return report
    teacher_logits, teacher_hidden, targets = _load_teacher_arrays(
        context,
        paths["arrays_path"],
    )
    selected = _selected_pairs(mask)
    sequence_indices = [
        int(value) for value in protocol["internal_screen_sequence_indices"]
    ]
    candidate = _evaluate_headwise_candidate(
        selected,
        sequence_indices=sequence_indices,
        context=context,
        library=candidate_path,
        teacher_logits=teacher_logits,
        teacher_hidden=teacher_hidden,
        targets=targets,
        threads=threads,
        replay_sequence_index=protocol["scope"][
            "reset_replay_sequence_index"
        ],
    )
    execution_seconds = time.perf_counter() - started
    post = _screen_post_authentication(
        context,
        paths,
        manifest_sha256=manifest_sha256,
        source_inventory=source_inventory,
        candidate_library=candidate_path,
        candidate_library_sha256=candidate_hash,
        trace_protocol=trace_protocol_path,
        trace_metadata=trace_metadata_path,
        trace_arrays=trace_arrays_path,
        head_mask=head_mask_path,
        trace_hashes=trace_hashes,
        screen_protocol=protocol_path,
        screen_protocol_sha256=protocol_hash,
    )
    expected_resources = protocol["budget_contract"][
        "attention_expectations_per_sequence"
    ]
    actual_resources = candidate["attention_expectations_per_sequence"]
    resource_checks = {
        "exact_51_of_256_heads": (
            len(selected) == _RESCUED_HEADS
            and len(set(selected)) == _RESCUED_HEADS
        ),
        "head_mask_identity": (
            sha256_json(mask["attention_head_mask"])
            == protocol["head_mask_identity_sha256"]
        ),
        "exact_resource_contract": actual_resources == expected_resources,
        "logical_read_bytes": (
            actual_resources["attention_logical_read_bytes"]
            == _EXPECTED_HEADWISE["attention_logical_read_bytes"]
        ),
        "logical_read_fraction": (
            actual_resources["attention_logical_read_fraction"]
            == _EXPECTED_FRACTION
            and actual_resources["attention_logical_read_fraction"]
            <= _THRESHOLDS["maximum_attention_logical_read_fraction"]
        ),
        "52_head_boundary_inadmissible": (
            protocol["budget_contract"]["next_head_boundary"][
                "attention_logical_read_fraction"
            ]
            == _FIFTY_TWO_FRACTION
            and _FIFTY_TWO_FRACTION
            > _THRESHOLDS["maximum_attention_logical_read_fraction"]
        ),
        "q7_expectations_unchanged": (
            candidate["q7_expectations_per_sequence"]
            == context["q7_expectations"]
        ),
    }
    q7_traffic = layer_rescue._q7_traffic_contract(
        context["model"],
        context["q7_expectations"],
    )
    evidence_checks = {
        "all_base_headwise_parity": parity["passed"],
        "candidate_evaluation_count": True,
        "six_internal_sequences_only": (
            sequence_indices
            == [
                int(row["sequence_index"])
                for row in context["split"]["internal_holdout"]
            ]
            and len(sequence_indices) == _INTERNAL_SEQUENCES
        ),
        "mask_unchanged_by_internal_screen": (
            protocol["head_mask_identity_sha256"]
            == mask["attention_head_mask_sha256"]
        ),
        "candidate_evidence": candidate["evidence_passed"],
        "resource_contract": all(resource_checks.values()),
        "q7_traffic_fraction": (
            q7_traffic["q7_fraction_of_all_expert_ideal_q4"]
            <= _THRESHOLDS["maximum_q7_traffic_fraction"]
        ),
        "post_run_authentication": all(post.values()),
    }
    evidence_passed = all(evidence_checks.values())
    quality_passed = candidate["quality_passed"]
    if not evidence_passed:
        status = "headwise_screen_invalid"
        decision = "stop_and_diagnose_headwise_evidence"
    elif quality_passed:
        status = "headwise_screen_development_complete"
        decision = "freeze_fresh_sealed_eight_sequence_confirmation"
    else:
        status = "headwise_screen_development_complete"
        decision = "investigate_value_sensitivity_or_dynamic_head_allocation"
    report = {
        "schema_version": 1,
        "experiment": _SCREEN_EXPERIMENT,
        "status": status,
        "provenance": {
            "trace_protocol_frozen_before_dense_attention_maps": True,
            "mask_derived_only_from_two_selection_records": True,
            "screen_protocol_frozen_after_mask_and_candidate_library": True,
            "screen_protocol_frozen_before_execution": True,
            "all_base_parity_passed_before_candidate_execution": True,
            "six_screen_records_previously_consumed_by_diagnostics": True,
            "six_screen_records_are_internal_development_not_untouched_holdout": True,
            "internal_screen_outputs_could_not_influence_mask": True,
            "scientific_role": "development screen; not confirmation",
            "execution_interface": _EXECUTION_INTERFACE,
        },
        "artifacts": {
            **_base_bindings(context),
            **dict(trace_hashes),
            "screen_protocol_sha256": protocol_hash,
            "candidate_native_library_sha256": candidate_hash,
            "headwise_source_sha256": source_hash,
            "headwise_source_inventory_sha256": source_inventory,
        },
        "configuration": {
            "candidate_device": "cpu",
            "candidate_threads": threads,
            "transformers_model_shell_used": False,
            "candidate_count": 1,
            "selected_head_count": len(selected),
            "attention_head_mask": mask["attention_head_mask"],
            "attention_head_policies": _head_policies(selected),
            "budget_contract": protocol["budget_contract"],
            "analytical_byte_components": protocol[
                "analytical_byte_components"
            ],
            "q7_traffic_contract_per_sequence": q7_traffic,
            "measured_hardware_traffic": False,
        },
        "record_split": protocol["record_split"],
        "head_mask_identity_sha256": protocol["head_mask_identity_sha256"],
        "quality_bands": protocol["quality_bands"],
        "thresholds": protocol["thresholds"],
        "population_contract": protocol["population_contract"],
        "all_base_headwise_parity": parity,
        "candidate_evaluation_count": 1,
        "internal_screen_result": candidate,
        "resource_checks": resource_checks,
        "evidence_checks": evidence_checks,
        "evidence_passed": evidence_passed,
        "internal_screen_quality_passed": quality_passed,
        "fresh_eight_sequence_confirmation_required": (
            evidence_passed and quality_passed
        ),
        "decision": decision,
        "post_run_authentication": post,
        "performance": {
            "execution_seconds": execution_seconds,
            "all_base_parity_seconds": parity["elapsed_seconds"],
            "candidate_primary_sequence_seconds": candidate["performance"][
                "primary_sequence_seconds"
            ],
            "candidate_reset_replay_seconds": candidate["performance"][
                "reset_replay_seconds"
            ],
        },
        "limitations": protocol["limitations"],
    }
    atomic_json(output_path, report)
    return report


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--reference-library", required=True, type=Path)
    parser.add_argument("--layered-library", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--teacher-reference", required=True, type=Path)
    parser.add_argument("--teacher-arrays", required=True, type=Path)
    parser.add_argument("--sustained-protocol", required=True, type=Path)
    parser.add_argument("--sustained-protocol-sha256", required=True)
    parser.add_argument("--sustained-result", required=True, type=Path)
    parser.add_argument("--sustained-result-sha256", required=True)
    parser.add_argument("--control-protocol", required=True, type=Path)
    parser.add_argument("--control-protocol-sha256", required=True)
    parser.add_argument("--control-result", required=True, type=Path)
    parser.add_argument("--control-result-sha256", required=True)
    parser.add_argument("--sweep-protocol", required=True, type=Path)
    parser.add_argument("--sweep-protocol-sha256", required=True)
    parser.add_argument("--sweep-result", required=True, type=Path)
    parser.add_argument("--sweep-result-sha256", required=True)
    parser.add_argument("--layer-rescue-protocol", required=True, type=Path)
    parser.add_argument("--layer-rescue-protocol-sha256", required=True)
    parser.add_argument("--layer-rescue-result", required=True, type=Path)
    parser.add_argument("--layer-rescue-result-sha256", required=True)


def _add_trace_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trace-protocol", required=True, type=Path)
    parser.add_argument("--trace-protocol-sha256", required=True)
    parser.add_argument("--trace-metadata", required=True, type=Path)
    parser.add_argument("--trace-metadata-sha256", required=True)
    parser.add_argument("--trace-arrays", required=True, type=Path)
    parser.add_argument("--trace-arrays-sha256", required=True)


def _common_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        name: getattr(args, name)
        for name in (
            "package",
            "manifest_sha256",
            "reference_library",
            "layered_library",
            "dataset",
            "corpus_manifest",
            "teacher_reference",
            "teacher_arrays",
            "sustained_protocol",
            "sustained_protocol_sha256",
            "sustained_result",
            "sustained_result_sha256",
            "control_protocol",
            "control_protocol_sha256",
            "control_result",
            "control_result_sha256",
            "sweep_protocol",
            "sweep_protocol_sha256",
            "sweep_result",
            "sweep_result_sha256",
            "layer_rescue_protocol",
            "layer_rescue_protocol_sha256",
            "layer_rescue_result",
            "layer_rescue_result_sha256",
        )
    }


def _trace_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        name: getattr(args, name)
        for name in (
            "trace_protocol",
            "trace_protocol_sha256",
            "trace_metadata",
            "trace_metadata_sha256",
            "trace_arrays",
            "trace_arrays_sha256",
        )
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze and run the teacher-guided OLMoE head-wise screen"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_trace = commands.add_parser(
        "freeze-trace",
        help="freeze dense-teacher attention capture before maps exist",
    )
    _add_common_arguments(freeze_trace)
    freeze_trace.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    freeze_trace.add_argument("--threads", type=int, default=_THREADS)
    freeze_trace.add_argument("--out", required=True, type=Path)

    capture = commands.add_parser(
        "capture",
        help="capture only the frozen two-record dense attention trace",
    )
    _add_common_arguments(capture)
    capture.add_argument("--trace-protocol", required=True, type=Path)
    capture.add_argument("--trace-protocol-sha256", required=True)
    capture.add_argument("--arrays-out", required=True, type=Path)
    capture.add_argument("--trace-out", required=True, type=Path)

    derive = commands.add_parser(
        "derive",
        help="derive the deterministic 51-of-256 head mask",
    )
    _add_trace_artifact_arguments(derive)
    derive.add_argument("--out", required=True, type=Path)

    freeze_screen = commands.add_parser(
        "freeze-screen",
        help="freeze the fixed-mask causal screen and native DSO",
    )
    _add_common_arguments(freeze_screen)
    _add_trace_artifact_arguments(freeze_screen)
    freeze_screen.add_argument("--head-mask", required=True, type=Path)
    freeze_screen.add_argument("--head-mask-sha256", required=True)
    freeze_screen.add_argument("--candidate-library", required=True, type=Path)
    freeze_screen.add_argument("--threads", type=int, default=_THREADS)
    freeze_screen.add_argument("--out", required=True, type=Path)

    evaluate = commands.add_parser(
        "evaluate",
        help="prove all-base parity and run the one-candidate internal screen",
    )
    _add_common_arguments(evaluate)
    _add_trace_artifact_arguments(evaluate)
    evaluate.add_argument("--head-mask", required=True, type=Path)
    evaluate.add_argument("--head-mask-sha256", required=True)
    evaluate.add_argument("--screen-protocol", required=True, type=Path)
    evaluate.add_argument("--screen-protocol-sha256", required=True)
    evaluate.add_argument("--candidate-library", required=True, type=Path)
    evaluate.add_argument("--threads", type=int, default=_THREADS)
    evaluate.add_argument("--out", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "freeze-trace":
        result = freeze_native_olmoe_headwise_trace_protocol(
            **_common_from_args(args),
            device=args.device,
            threads=args.threads,
            out=args.out,
        )
    elif args.command == "capture":
        common = _common_from_args(args)
        manifest_sha256 = common.pop("manifest_sha256")
        result = capture_native_olmoe_headwise_dense_attention(
            **common,
            manifest_sha256=manifest_sha256,
            trace_protocol=args.trace_protocol,
            trace_protocol_sha256=args.trace_protocol_sha256,
            arrays_out=args.arrays_out,
            trace_out=args.trace_out,
        )
    elif args.command == "derive":
        result = derive_native_olmoe_headwise_mask(
            **_trace_from_args(args),
            out=args.out,
        )
    elif args.command == "freeze-screen":
        common = _common_from_args(args)
        manifest_sha256 = common.pop("manifest_sha256")
        result = freeze_native_olmoe_headwise_screen_protocol(
            **common,
            **_trace_from_args(args),
            manifest_sha256=manifest_sha256,
            head_mask=args.head_mask,
            head_mask_sha256=args.head_mask_sha256,
            candidate_library=args.candidate_library,
            threads=args.threads,
            out=args.out,
        )
    else:
        common = _common_from_args(args)
        manifest_sha256 = common.pop("manifest_sha256")
        result = evaluate_native_olmoe_headwise_screen(
            **common,
            **_trace_from_args(args),
            manifest_sha256=manifest_sha256,
            head_mask=args.head_mask,
            head_mask_sha256=args.head_mask_sha256,
            screen_protocol=args.screen_protocol,
            screen_protocol_sha256=args.screen_protocol_sha256,
            candidate_library=args.candidate_library,
            threads=args.threads,
            out=args.out,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
