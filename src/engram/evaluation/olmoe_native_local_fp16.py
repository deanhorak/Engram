"""Prospective causal gate for a compressed local-attention cache.

This evaluator is deliberately separate from the protected Milestone 2
protocol.  It freezes the already-authored sustained corpus and untouched
teacher arrays, then evaluates the CPU native runtime with a W56 local cache
whose keys and values are stored as IEEE FP16.  The ordinary package ABI and
the protected W16 ABI remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from engram.compiler.olmoe_native import validate_olmoe_native_package
from engram.evaluation.olmoe_native_causal import _aggregate, _position_metrics
from engram.evaluation.olmoe_native_sustained import (
    _POSITIONS_PER_SEQUENCE,
    _QUALITY_BANDS,
    _SEQUENCES,
    _TOKENS_PER_SEQUENCE,
    _attention_expectations,
    _model_descriptor,
    _q7_expectations,
)
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.utils import atomic_json, sha256_file, sha256_json


LOCAL_WINDOW = 56
OLDER_CANDIDATES = 8
OLDER_TOP_K = 4
SINK_TOKENS = 2
_THRESHOLDS = {
    "maximum_mean_kl": 0.05,
    "minimum_top1_agreement": 0.90,
    "maximum_mean_target_nll_delta": 0.05,
    "maximum_mean_final_hidden_relative_l2": 0.10,
    "maximum_attention_logical_read_fraction": 0.45,
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _compressed_attention_expectations(
    model: dict[str, int], storage: str = "fp16", local_window: int = LOCAL_WINDOW
) -> dict[str, int | float]:
    if storage not in {"fp16", "int8"}:
        raise ValueError("local storage must be fp16 or int8")
    policy = {
        "local_window": local_window,
        "older_candidates": OLDER_CANDIDATES,
        "older_top_k": OLDER_TOP_K,
        "sink_tokens": SINK_TOKENS,
    }
    base = _attention_expectations(model, policy)
    layers = int(model["layers"])
    kv_heads = int(model["key_value_heads"])
    dimension = int(model["head_dimension"])
    local_fp32 = int(base["attention_local_kv_bytes"])
    storage_bytes = 2 if storage == "fp16" else 1
    local_bytes = local_fp32 * storage_bytes // 4
    dense = int(base["dense_full_context_logical_kv_bytes"])
    state_local_fp32 = (
        2 * local_window * kv_heads * dimension * 4 * layers
    )
    state_local_compressed = (
        2 * local_window * kv_heads * dimension * storage_bytes * layers
    )
    result = dict(base)
    result["attention_local_kv_bytes"] = local_bytes
    result["attention_logical_read_bytes"] = (
        local_bytes
        + int(base["attention_candidate_key_bytes"])
        + int(base["attention_selected_value_bytes"])
    )
    result["attention_logical_read_fraction"] = (
        int(result["attention_logical_read_bytes"]) / dense
    )
    result["attention_state_bytes"] = (
        int(base["attention_state_bytes"])
        - state_local_fp32
        + state_local_compressed
        + (2 * local_window * kv_heads * 4 * layers if storage == "int8" else 0)
    )
    result["compression"] = (
        "local_kv_ieee_fp16" if storage == "fp16" else "local_kv_symmetric_int8"
    )
    return result


def freeze_local_fp16_protocol(
    *,
    package: str | Path,
    manifest_sha256: str,
    library: str | Path,
    dataset: str | Path,
    teacher_reference: str | Path,
    teacher_arrays: str | Path,
    out: str | Path,
    threads: int = 12,
    storage: str = "fp16",
    local_window: int = LOCAL_WINDOW,
) -> dict[str, Any]:
    """Freeze all inputs and the W56/FP16 decision before candidate reads."""

    if storage not in {"fp16", "int8"}:
        raise ValueError("local storage must be fp16 or int8")
    output = Path(out).expanduser().resolve()
    if output.exists():
        raise ValueError("local FP16 protocol target already exists")
    package_path = Path(package).expanduser().resolve()
    library_path = Path(library).expanduser().resolve()
    dataset_path = Path(dataset).expanduser().resolve()
    reference_path = Path(teacher_reference).expanduser().resolve()
    arrays_path = Path(teacher_arrays).expanduser().resolve()
    manifest = validate_olmoe_native_package(
        package_path, expected_manifest_sha256=manifest_sha256
    )
    reference = _read_json(reference_path)
    if (
        reference.get("experiment") != "olmoe_untouched_teacher_causal_reference"
        or reference.get("configuration", {}).get("weights_modified") is not False
        or reference.get("dataset", {}).get("sequences") != _SEQUENCES
        or reference.get("dataset", {}).get("tokens_per_sequence")
        != _TOKENS_PER_SEQUENCE
        or reference.get("arrays", {}).get("sha256") != sha256_file(arrays_path)
        or reference.get("dataset", {}).get("sha256") != sha256_file(dataset_path)
    ):
        raise ValueError("teacher or dataset identity is invalid")
    config = _read_json(package_path / manifest["model"]["config_path"])
    model = _model_descriptor(manifest, config)
    protocol = {
        "schema_version": 1,
        "experiment": "olmoe_native_local_fp16_sustained",
        "status": "frozen_before_candidate_execution",
        "package_manifest_sha256": manifest_sha256.lower(),
        "native_library_sha256": sha256_file(library_path),
        "dataset_sha256": sha256_file(dataset_path),
        "teacher_reference_sha256": sha256_file(reference_path),
        "teacher_arrays_sha256": sha256_file(arrays_path),
        "input_identity": reference["dataset"]["input_identity"],
        "input_ids": reference["dataset"]["input_ids"],
        "sequences": _SEQUENCES,
        "tokens_per_sequence": _TOKENS_PER_SEQUENCE,
        "model": model,
        "attention_policy": {
            "local_window": local_window,
            "older_candidates": OLDER_CANDIDATES,
            "older_top_k": OLDER_TOP_K,
            "sink_tokens": SINK_TOKENS,
            "local_storage": (
                "ieee_fp16_keys_and_values"
                if storage == "fp16"
                else "symmetric_int8_keys_and_values"
            ),
        },
        "attention_expectations_per_sequence": _compressed_attention_expectations(
            model, storage, local_window
        ),
        "q7_expectations_per_sequence": _q7_expectations(model),
        "quality_bands": [
            {"name": name, "start": start, "stop": stop}
            for name, start, stop in _QUALITY_BANDS
        ],
        "thresholds": _THRESHOLDS,
        "scope": {
            "candidate_device": "cpu",
            "candidate_threads": int(threads),
            "transformers_model_shell_used_by_candidate": False,
            "teacher_weights_modified": False,
            "protected_milestone_2_gate": False,
        },
    }
    atomic_json(output, protocol)
    return protocol


def _quality(metrics: dict[str, Any], prefix: str) -> dict[str, bool]:
    return {
        f"{prefix}_mean_kl": metrics["teacher_to_native_kl"] <= _THRESHOLDS["maximum_mean_kl"],
        f"{prefix}_top1_agreement": metrics["teacher_top1_agreement"] >= _THRESHOLDS["minimum_top1_agreement"],
        f"{prefix}_target_nll_delta": metrics["target_nll_delta"] <= _THRESHOLDS["maximum_mean_target_nll_delta"],
        f"{prefix}_hidden_relative_l2": metrics["final_hidden_relative_l2"] <= _THRESHOLDS["maximum_mean_final_hidden_relative_l2"],
    }


def evaluate_local_fp16(
    *,
    package: str | Path,
    manifest_sha256: str,
    library: str | Path,
    dataset: str | Path,
    teacher_reference: str | Path,
    teacher_arrays: str | Path,
    protocol: str | Path,
    protocol_sha256: str,
    out: str | Path,
    threads: int = 12,
    storage: str = "fp16",
    local_window: int = LOCAL_WINDOW,
) -> dict[str, Any]:
    """Run the frozen W56/FP16 candidate and write authenticated evidence."""

    if storage not in {"fp16", "int8"}:
        raise ValueError("local storage must be fp16 or int8")
    package_path = Path(package).expanduser().resolve()
    library_path = Path(library).expanduser().resolve()
    dataset_path = Path(dataset).expanduser().resolve()
    reference_path = Path(teacher_reference).expanduser().resolve()
    arrays_path = Path(teacher_arrays).expanduser().resolve()
    protocol_path = Path(protocol).expanduser().resolve()
    protocol_value = _read_json(protocol_path)
    if sha256_file(protocol_path) != protocol_sha256.lower():
        raise ValueError("local FP16 protocol hash mismatch")
    if protocol_value.get("status") != "frozen_before_candidate_execution":
        raise ValueError("local FP16 protocol is not frozen")
    if protocol_value.get("package_manifest_sha256") != manifest_sha256.lower():
        raise ValueError("local FP16 package identity changed")
    reference = _read_json(reference_path)
    arrays = np.load(arrays_path)
    input_ids = reference["dataset"]["input_ids"]
    if reference["dataset"]["input_identity"] != sha256_json(input_ids):
        raise ValueError("teacher input identity changed")
    if reference["dataset"]["sha256"] != sha256_file(dataset_path):
        raise ValueError("dataset identity changed")
    manifest = validate_olmoe_native_package(
        package_path, expected_manifest_sha256=manifest_sha256
    )
    config_path = package_path / manifest["model"]["config_path"]
    non_mlp = package_path / manifest["transformer"]["path"]
    q7 = package_path / manifest["mlp"]["path"]
    model = _model_descriptor(manifest, _read_json(config_path))
    expected = _compressed_attention_expectations(model, storage, local_window)
    q7_expected = _q7_expectations(model)
    all_rows: list[dict[str, Any]] = []
    band_rows: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in _QUALITY_BANDS}
    sequence_results: list[dict[str, Any]] = []
    offset = 0
    total_q7 = 0
    total_attention = 0
    replay_hashes: tuple[str, str] | None = None
    for sequence_index, sequence in enumerate(input_ids):
        with OLMoENativeTokenRuntime(
            config_path,
            non_mlp,
            q7,
            library_path,
            threads=threads,
            local_window=local_window,
            older_candidates=OLDER_CANDIDATES,
            older_top_k=OLDER_TOP_K,
            sink_tokens=SINK_TOKENS,
            local_fp16=storage == "fp16",
            local_int8=storage == "int8",
        ) as runtime:
            rows: list[dict[str, Any]] = []
            hidden_digest = hashlib.sha256()
            logits_digest = hashlib.sha256()
            started = time.perf_counter()
            native_metrics: dict[str, Any] | None = None
            for position, token_id in enumerate(sequence[:-1]):
                result = runtime.forward([token_id])
                native_hidden, native_logits = runtime.last_diagnostics()
                native_metrics = result.metrics
                row = _position_metrics(
                    arrays["logits"][offset + position],
                    native_logits,
                    arrays["hidden"][offset + position],
                    native_hidden,
                    int(sequence[position + 1]),
                )
                row.update({"sequence": sequence_index, "position": position})
                rows.append(row)
                all_rows.append(row)
                for name, start, stop in _QUALITY_BANDS:
                    if start <= position < stop:
                        band_rows[name].append(row)
                        break
                hidden_digest.update(native_hidden.tobytes())
                logits_digest.update(native_logits.tobytes())
            assert native_metrics is not None
            offset += len(sequence) - 1
            total_q7 += int(native_metrics["q7_scheduled_bytes"])
            total_attention += int(native_metrics["attention_logical_read_bytes"])
            structural = {
                "positions_processed": native_metrics["positions_processed"] == _POSITIONS_PER_SEQUENCE,
                "attention_state_bytes": native_metrics["attention_state_bytes"] == expected["attention_state_bytes"],
                "attention_scratch_bytes": native_metrics["attention_scratch_bytes"] == expected["attention_scratch_bytes"],
                "attention_eviction_events": native_metrics["attention_eviction_events"] == expected["attention_eviction_events"],
                "attention_candidate_entries": native_metrics["attention_older_candidate_entries_scored"] == expected["attention_older_candidate_entries_scored"],
                "attention_selected_entries": native_metrics["attention_older_selected_entries"] == expected["attention_older_selected_entries"],
                "q7_scheduled_bytes": native_metrics["q7_scheduled_bytes"] == q7_expected["scheduled_bytes_per_sequence"],
            }
            hashes = (hidden_digest.hexdigest(), logits_digest.hexdigest())
            if replay_hashes is None:
                replay_hashes = hashes
            sequence_results.append({
                "sequence": sequence_index,
                "elapsed_seconds": time.perf_counter() - started,
                "metrics": _aggregate(rows),
                "native_metrics": native_metrics,
                "diagnostic_hashes": {"hidden_sha256": hashes[0], "logits_sha256": hashes[1]},
                "structural_checks": structural,
                "structural_passed": all(structural.values()),
            })
    aggregate = _aggregate(all_rows)
    bands = {name: _aggregate(rows) for name, rows in band_rows.items()}
    dense = int(expected["dense_full_context_logical_kv_bytes"]) * _SEQUENCES
    attention_fraction = total_attention / dense
    q7_fraction = total_q7 / (q7_expected["scheduled_bytes_per_sequence"] * _SEQUENCES)
    checks: dict[str, bool] = {
        "sequence_count": len(sequence_results) == _SEQUENCES,
        "prediction_positions": len(all_rows) == _SEQUENCES * _POSITIONS_PER_SEQUENCE,
        "attention_traffic": attention_fraction <= _THRESHOLDS["maximum_attention_logical_read_fraction"],
        "sequence_structural_checks": all(r["structural_passed"] for r in sequence_results),
        "q7_identity": q7_fraction == 1.0,
    }
    checks.update(_quality(aggregate, "overall"))
    for name, metrics in bands.items():
        checks.update(_quality(metrics, name))
    report = {
        "schema_version": 1,
        "experiment": "olmoe_native_local_fp16_sustained",
        "status": "passed" if all(checks.values()) else "failed",
        "gate_passed": all(checks.values()),
        "protocol_sha256": protocol_sha256.lower(),
        "package_manifest_sha256": manifest_sha256.lower(),
        "configuration": {
            "device": "cpu",
            "threads": threads,
            "local_window": local_window,
            "older_candidates": OLDER_CANDIDATES,
            "older_top_k": OLDER_TOP_K,
            "sink_tokens": SINK_TOKENS,
            "local_storage": (
                "ieee_fp16_keys_and_values"
                if storage == "fp16"
                else "symmetric_int8_keys_and_values"
            ),
            "transformers_model_shell": False,
        },
        "metrics": aggregate,
        "position_bands": bands,
        "traffic": {
            "attention_logical_read_bytes": total_attention,
            "dense_full_context_logical_kv_bytes": dense,
            "attention_logical_read_fraction": attention_fraction,
            "q7_scheduled_bytes": total_q7,
            "q7_traffic_fraction": q7_fraction,
        },
        "checks": checks,
        "sequence_results": sequence_results,
    }
    atomic_json(Path(out).expanduser().resolve(), report)
    return report
