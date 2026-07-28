"""Prospective sustained-context gate for the native OLMoE package."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from engram.compiler.olmoe_native import validate_olmoe_native_package
from engram.evaluation.olmoe_native_causal import (
    _aggregate,
    _authenticate_evaluator_sources,
    _load_inputs,
    _position_metrics,
)
from engram.evaluation.olmoe_native_generation import (
    _read_object,
    _validate_teacher_source,
)
from engram.models.olmoe_q7 import olmoe_q7_layout
from engram.runtime.olmoe_native import OLMoENativePackageRuntime
from engram.utils import atomic_json, sha256_file, sha256_json


_SEQUENCES = 8
_TOKENS_PER_SEQUENCE = 129
_POSITIONS_PER_SEQUENCE = _TOKENS_PER_SEQUENCE - 1
_QUALITY_BANDS = (
    ("positions_0_15", 0, 16),
    ("positions_16_31", 16, 32),
    ("positions_32_63", 32, 64),
    ("positions_64_95", 64, 96),
    ("positions_96_127", 96, 128),
)
_THRESHOLDS = {
    "maximum_mean_kl": 0.05,
    "minimum_top1_agreement": 0.90,
    "maximum_mean_target_nll_delta": 0.05,
    "maximum_mean_final_hidden_relative_l2": 0.10,
    "maximum_q7_traffic_fraction": 0.45,
    "maximum_attention_logical_read_fraction": 0.45,
    "minimum_sequences": _SEQUENCES,
    "minimum_prediction_positions": _SEQUENCES * _POSITIONS_PER_SEQUENCE,
    "minimum_band_prediction_positions": _SEQUENCES * 16,
}
_EVALUATOR_SOURCES = (
    "CMakeLists.txt",
    "native/include/engram/olmoe_token_runtime.h",
    "native/include/engram/olmoe_token_runtime_c.h",
    "native/include/engram/olmoe_q7.h",
    "native/include/engram/olmoe_weights.h",
    "native/include/engram/safetensors.h",
    "native/include/engram/streaming_attention.h",
    "native/include/engram/thread_pool.h",
    "native/src/olmoe_token_runtime.cpp",
    "native/src/olmoe_token_runtime_c.cpp",
    "native/src/olmoe_q7.cpp",
    "native/src/olmoe_weights.cpp",
    "native/src/safetensors.cpp",
    "native/src/streaming_attention.cpp",
    "native/src/thread_pool.cpp",
    "src/engram/cli/__init__.py",
    "src/engram/compiler/olmoe_native.py",
    "src/engram/evaluation/olmoe_native_causal.py",
    "src/engram/evaluation/olmoe_native_generation.py",
    "src/engram/evaluation/olmoe_native_sustained.py",
    "src/engram/models/olmoe_q7.py",
    "src/engram/runtime/olmoe_native.py",
    "src/engram/utils.py",
)
_TEACHER_CONFIGURATION = {
    "dtype": "bfloat16",
    "device": "cpu",
    "threads": 12,
    "batch_size": 1,
    "expert_workers": 1,
    "sequence_workers": 4,
    "threaded_expert_layers": 0,
    "expert_backend": "transformers_reference",
    "sequence_backend": "thread_pool_shared_model_v1",
    "attention_implementation": "eager",
    "use_cache": False,
    "output_hidden_states": True,
    "weights_modified": False,
}


def _load_natural_inputs(
    dataset: Path,
    tokenizer: object,
) -> list[list[int]]:
    """Retokenize eight authored text records and reject precomputed IDs."""

    record_ids: list[str] = []
    texts: list[str] = []
    with dataset.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"sustained dataset line {line_number} is invalid JSON"
                ) from exc
            if (
                not isinstance(record, dict)
                or "input_ids" in record
                or record.get("source_kind") != "engram_authored_holdout"
                or not isinstance(record.get("record_id"), str)
                or not record["record_id"]
                or not isinstance(record.get("text"), str)
                or not record["text"].strip()
            ):
                raise ValueError(
                    f"sustained dataset line {line_number} is not a natural-text "
                    "holdout record"
                )
            record_ids.append(record["record_id"])
            texts.append(record["text"])
    if (
        len(record_ids) != _SEQUENCES
        or len(set(record_ids)) != _SEQUENCES
        or len(set(texts)) != _SEQUENCES
    ):
        raise ValueError("sustained dataset must contain eight distinct text records")
    inputs = _load_inputs(
        dataset,
        tokenizer,
        sequences=_SEQUENCES,
        tokens_per_sequence=_TOKENS_PER_SEQUENCE,
    )
    return inputs


def _validate_corpus_manifest(
    dataset: Path,
    corpus_manifest: Path,
    tokenizer: object,
    *,
    tokenizer_sha256: str,
) -> tuple[list[list[int]], dict[str, Any]]:
    inputs = _load_natural_inputs(dataset, tokenizer)
    manifest = _read_object(corpus_manifest, "sustained corpus manifest")
    records: list[dict[str, Any]] = []
    with dataset.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            full_ids = tokenizer.encode(record["text"]).ids
            records.append(
                {
                    "record_id": record["record_id"],
                    "domain": record.get("domain"),
                    "full_token_count": len(full_ids),
                    "window_identity": sha256_json(
                        {"input_ids": full_ids[:_TOKENS_PER_SEQUENCE]}
                    ),
                }
            )
    selection = manifest.get("selection")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("experiment") != "olmoe_sustained_context_authored_holdout"
        or manifest.get("status") != "authored_and_fixed_before_candidate_execution"
        or manifest.get("dataset_sha256") != sha256_file(dataset)
        or manifest.get("tokenizer_sha256") != tokenizer_sha256
        or manifest.get("input_identity") != sha256_json(inputs)
        or manifest.get("sequences") != _SEQUENCES
        or manifest.get("tokens_per_sequence") != _TOKENS_PER_SEQUENCE
        or manifest.get("records") != records
        or manifest.get("created_utc") != "2026-07-28"
        or not isinstance(selection, dict)
        or selection.get("source_kind") != "engram_authored_holdout"
        or not isinstance(selection.get("rule"), str)
        or not selection["rule"]
        or selection.get("candidate_or_teacher_outputs_inspected_during_selection")
        is not False
        or selection.get("previous_engram_calibration_or_confirmation_text_reused")
        is not False
        or len({record["domain"] for record in records}) != _SEQUENCES
    ):
        raise ValueError("sustained corpus manifest contract is invalid")
    return inputs, manifest


def _attention_expectations(
    model: dict[str, int],
    policy: dict[str, int],
    *,
    positions: int = _POSITIONS_PER_SEQUENCE,
) -> dict[str, int | float]:
    layers = int(model["layers"])
    query_heads = int(model["query_heads"])
    key_value_heads = int(model["key_value_heads"])
    head_dimension = int(model["head_dimension"])
    local_window = int(policy["local_window"])
    older_candidates = int(policy["older_candidates"])
    older_top_k = int(policy["older_top_k"])
    sink_tokens = int(policy["sink_tokens"])
    if (
        min(
            layers,
            query_heads,
            key_value_heads,
            head_dimension,
            local_window,
            older_candidates,
            older_top_k,
            positions,
        )
        <= 0
        or query_heads % key_value_heads
        or older_top_k > older_candidates
        or sink_tokens > older_candidates
    ):
        raise ValueError("sustained attention dimensions are invalid")

    state_per_layer = (
        2 * local_window * key_value_heads * head_dimension * 4
        + query_heads * local_window * 4
        + local_window * 8
        + 2 * query_heads * older_candidates * head_dimension * 4
        + query_heads * older_candidates * 4
        + query_heads * older_candidates * 8
        + query_heads * older_candidates
    )
    scratch_per_layer = (
        (local_window + older_candidates) * 4
        + older_candidates * 4
        + (local_window + older_top_k) * 4
        + older_top_k * 8
    )
    active_older_per_head = [
        min(older_candidates, max(0, position - local_window + 1))
        for position in range(positions)
    ]
    selected_older_per_head = [
        min(older_top_k, active) for active in active_older_per_head
    ]
    scored = sum(active_older_per_head) * query_heads * layers
    selected = sum(selected_older_per_head) * query_heads * layers
    local_bytes = (
        sum(
            min(position + 1, local_window) * query_heads * head_dimension * 4 * 2
            for position in range(positions)
        )
        * layers
    )
    candidate_bytes = scored * head_dimension * 4
    selected_bytes = selected * head_dimension * 4
    dense_bytes = (
        sum(
            (position + 1) * query_heads * head_dimension * 4 * 2
            for position in range(positions)
        )
        * layers
    )
    evicted_positions = max(0, positions - local_window)
    sink_evictions = min(sink_tokens, evicted_positions)
    guaranteed_heavy_evictions = min(
        max(0, older_candidates - sink_tokens),
        max(0, evicted_positions - sink_evictions),
    )
    maximum_heavy_evictions = max(0, evicted_positions - sink_evictions)
    return {
        "positions_processed": positions,
        "attention_state_bytes": state_per_layer * layers,
        "attention_scratch_bytes": scratch_per_layer * layers,
        "attention_eviction_events": evicted_positions * layers,
        "attention_older_candidate_entries_scored": scored,
        "attention_older_selected_entries": selected,
        "attention_sink_insertions": (sink_evictions * query_heads * layers),
        "attention_heavy_hitter_updates_minimum": (
            guaranteed_heavy_evictions * query_heads * layers
        ),
        "attention_heavy_hitter_updates_maximum": (
            maximum_heavy_evictions * query_heads * layers
        ),
        "attention_local_kv_bytes": local_bytes,
        "attention_candidate_key_bytes": candidate_bytes,
        "attention_selected_value_bytes": selected_bytes,
        "attention_logical_read_bytes": (
            local_bytes + candidate_bytes + selected_bytes
        ),
        "dense_full_context_logical_kv_bytes": dense_bytes,
        "attention_logical_read_fraction": (
            (local_bytes + candidate_bytes + selected_bytes) / dense_bytes
        ),
    }


def _source_shard_hashes(reference: dict[str, Any]) -> dict[str, str]:
    source = reference.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("model"), str):
        raise ValueError("sustained teacher source path is invalid")
    model_path = Path(source["model"]).expanduser().resolve()
    index = _read_object(
        model_path / "model.safetensors.index.json",
        "teacher weight index",
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("teacher weight index has no weight map")
    names = sorted(set(weight_map.values()))
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("teacher shard inventory is invalid")

    def digest(name: str) -> tuple[str, str]:
        return name, sha256_file(model_path / name)

    with ThreadPoolExecutor(max_workers=min(6, len(names))) as executor:
        return dict(executor.map(digest, names))


def _post_source_shards(
    reference: dict[str, Any],
    expected: dict[str, str],
) -> bool:
    source = reference["source"]
    model_path = Path(source["model"]).expanduser().resolve()

    def validate(item: tuple[str, str]) -> bool:
        name, expected_hash = item
        return sha256_file(model_path / name) == expected_hash

    with ThreadPoolExecutor(max_workers=min(6, len(expected))) as executor:
        return all(executor.map(validate, expected.items()))


def _model_descriptor(
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, int]:
    hidden = int(manifest["model"]["hidden_size"])
    query_heads = int(manifest["model"]["num_attention_heads"])
    if hidden % query_heads:
        raise ValueError("sustained model head dimensions are invalid")
    return {
        "layers": int(manifest["model"]["num_hidden_layers"]),
        "hidden_size": hidden,
        "intermediate_size": int(manifest["model"]["intermediate_size"]),
        "experts": int(config["num_experts"]),
        "vocab_size": int(manifest["model"]["vocab_size"]),
        "query_heads": query_heads,
        "key_value_heads": int(manifest["model"]["num_key_value_heads"]),
        "head_dimension": hidden // query_heads,
        "top_k": int(config["num_experts_per_tok"]),
        "q7_group_size": 64,
    }


def _q7_expectations(
    model: dict[str, int],
    *,
    positions: int = _POSITIONS_PER_SEQUENCE,
) -> dict[str, int]:
    layout = olmoe_q7_layout(
        layer_count=int(model["layers"]),
        hidden_size=int(model["hidden_size"]),
        intermediate_size=int(model["intermediate_size"]),
        num_experts=int(model["experts"]),
        top_k=int(model["top_k"]),
        group_size=int(model["q7_group_size"]),
    )
    selected_payload = (
        2 * (layout.gate_code_bytes + layout.gate_scale_bytes)
        + layout.down_code_bytes
        + layout.down_scale_bytes
    )
    per_layer_position = layout.router_bytes + layout.top_k * selected_payload
    return {
        "artifact_bytes": layout.file_bytes,
        "router_bytes_per_layer_position": layout.router_bytes,
        "selected_expert_bytes_per_layer_position": (layout.top_k * selected_payload),
        "scheduled_bytes_per_layer_position": per_layer_position,
        "scheduled_bytes_per_position": per_layer_position * layout.layer_count,
        "scheduled_bytes_per_sequence": (
            per_layer_position * layout.layer_count * positions
        ),
    }


def _teacher_configuration(reference: dict[str, Any]) -> dict[str, Any]:
    configuration = reference.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("sustained teacher configuration is missing")
    return {name: configuration.get(name) for name in _TEACHER_CONFIGURATION}


def freeze_olmoe_sustained_context_protocol(
    *,
    package: str | Path,
    manifest_sha256: str,
    library: str | Path,
    dataset: str | Path,
    corpus_manifest: str | Path,
    teacher_reference: str | Path,
    teacher_arrays: str | Path,
    out: str | Path,
    threads: int = 12,
) -> dict[str, Any]:
    """Freeze all identities and decisions before sustained candidate execution."""

    if threads <= 0:
        raise ValueError("sustained candidate threads must be positive")
    output = Path(out).expanduser().resolve()
    if output.exists():
        raise ValueError("sustained protocol target already exists")
    package_path = Path(package).expanduser().resolve()
    library_path = Path(library).expanduser().resolve()
    dataset_path = Path(dataset).expanduser().resolve()
    corpus_manifest_path = Path(corpus_manifest).expanduser().resolve()
    reference_path = Path(teacher_reference).expanduser().resolve()
    arrays_path = Path(teacher_arrays).expanduser().resolve()
    manifest = validate_olmoe_native_package(
        package_path,
        expected_manifest_sha256=manifest_sha256,
    )
    reference = _read_object(reference_path, "sustained teacher reference")
    if (
        reference.get("schema_version") != 1
        or reference.get("experiment") != "olmoe_untouched_teacher_causal_reference"
        or reference.get("configuration", {}).get("weights_modified") is not False
        or reference.get("dataset", {}).get("sequences") != _SEQUENCES
        or reference.get("dataset", {}).get("tokens_per_sequence")
        != _TOKENS_PER_SEQUENCE
        or reference.get("dataset", {}).get("prediction_positions")
        != _SEQUENCES * _POSITIONS_PER_SEQUENCE
        or reference.get("dataset", {}).get("sha256") != sha256_file(dataset_path)
        or reference.get("arrays", {}).get("sha256") != sha256_file(arrays_path)
        or _teacher_configuration(reference) != _TEACHER_CONFIGURATION
    ):
        raise ValueError("sustained teacher reference contract is invalid")
    runtime_policy = manifest["runtime"]["attention_policy"]
    policy = {
        "local_window": int(runtime_policy["local_window"]),
        "older_candidates": int(runtime_policy["older_candidates"]),
        "older_top_k": int(runtime_policy["older_top_k"]),
        "sink_tokens": int(runtime_policy["sink_tokens"]),
    }
    if policy != {
        "local_window": 16,
        "older_candidates": 8,
        "older_top_k": 4,
        "sink_tokens": 2,
    }:
        raise ValueError("sustained gate requires the frozen W16/C8/K4/S2 policy")
    config_path = package_path / manifest["model"]["config_path"]
    config = _read_object(config_path, "packaged OLMoE config")
    model = _model_descriptor(manifest, config)
    q7_expectations = _q7_expectations(model)
    if (
        manifest["files"]["mlp/experts.q7"]["bytes"]
        != q7_expectations["artifact_bytes"]
    ):
        raise ValueError("packaged Q7 artifact differs from the frozen layout")
    with OLMoENativePackageRuntime(
        package_path,
        manifest_sha256=manifest_sha256,
        library=library_path,
        threads=threads,
    ) as runtime:
        input_ids, _corpus = _validate_corpus_manifest(
            dataset_path,
            corpus_manifest_path,
            runtime.tokenizer,
            tokenizer_sha256=manifest["files"]["tokenizer/tokenizer.json"]["sha256"],
        )
        if not runtime.runtime.attention_metrics_available:
            raise ValueError("native sustained attention metric ABI is unavailable")
    if input_ids != reference.get("dataset", {}).get("input_ids") or sha256_json(
        input_ids
    ) != reference.get("dataset", {}).get("input_identity"):
        raise ValueError("retokenized sustained inputs differ from teacher capture")
    repository = Path(__file__).resolve().parents[3]
    source_inventory = {
        relative: sha256_file(repository / relative) for relative in _EVALUATOR_SOURCES
    }
    source = reference["source"]
    protocol = {
        "schema_version": 1,
        "experiment": "olmoe_native_sustained_context_confirmation",
        "status": "frozen_before_candidate_execution",
        "source_revision": source["revision"],
        "source_config_sha256": source["config_sha256"],
        "source_index_sha256": source["index_sha256"],
        "source_shard_sha256": _source_shard_hashes(reference),
        "package_manifest_sha256": manifest_sha256.lower(),
        "native_library_sha256": sha256_file(library_path),
        "dataset_sha256": sha256_file(dataset_path),
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "input_identity": sha256_json(input_ids),
        "input_ids": input_ids,
        "teacher_reference_sha256": sha256_file(reference_path),
        "teacher_arrays_sha256": sha256_file(arrays_path),
        "sequences": _SEQUENCES,
        "tokens_per_sequence": _TOKENS_PER_SEQUENCE,
        "model": model,
        "attention_policy": policy,
        "attention_expectations_per_sequence": _attention_expectations(
            model,
            policy,
        ),
        "q7_expectations_per_sequence": q7_expectations,
        "quality_bands": [
            {"name": name, "start": start, "stop": stop}
            for name, start, stop in _QUALITY_BANDS
        ],
        "thresholds": _THRESHOLDS,
        "evaluator_source_sha256": source_inventory,
        "scope": {
            "candidate_device": "cpu",
            "candidate_threads": threads,
            "candidate_transformers_model_shell": False,
            "teacher_weights_modified": False,
            "teacher_dtype": "bfloat16",
            "teacher_configuration": _TEACHER_CONFIGURATION,
            "natural_text_records": True,
            "retokenize_dataset_during_evaluation": True,
            "reset_replay_sequence": 0,
            "protocol_frozen_before_candidate_execution": True,
        },
        "decision_rule": {
            "pass": (
                "overall and every frozen position band pass the semantic "
                "thresholds; all structural counters, reset replay, traffic, "
                "and authentication checks pass"
            ),
            "authenticated_semantic_failure": (
                "run a matched Q7 plus dense-attention control before changing "
                "the bounded retrieval policy"
            ),
            "systems_or_evidence_failure": (
                "stop and diagnose the runtime, counters, traffic, reset, or "
                "authentication failure before semantic attribution"
            ),
        },
    }
    atomic_json(output, protocol)
    return protocol


def _quality_checks(prefix: str, metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        f"{prefix}_mean_kl": (
            metrics["teacher_to_native_kl"] <= _THRESHOLDS["maximum_mean_kl"]
        ),
        f"{prefix}_top1_agreement": (
            metrics["teacher_top1_agreement"] >= _THRESHOLDS["minimum_top1_agreement"]
        ),
        f"{prefix}_target_nll_delta": (
            metrics["target_nll_delta"] <= _THRESHOLDS["maximum_mean_target_nll_delta"]
        ),
        f"{prefix}_hidden_relative_l2": (
            metrics["final_hidden_relative_l2"]
            <= _THRESHOLDS["maximum_mean_final_hidden_relative_l2"]
        ),
    }


def _update_diagnostic_hashes(
    hidden_digest: Any,
    logit_digest: Any,
    hidden: np.ndarray,
    logits: np.ndarray,
) -> None:
    hidden_digest.update(np.ascontiguousarray(hidden, dtype=np.float32).tobytes())
    logit_digest.update(np.ascontiguousarray(logits, dtype=np.float32).tobytes())


def _structural_checks(
    metrics: dict[str, int],
    expectations: dict[str, int | float],
    *,
    position: int,
) -> dict[str, bool]:
    exact_names = (
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
        name: int(metrics.get(name, -1)) == int(expectations[name])
        for name in exact_names
    }
    checks["cache_position"] = position == int(expectations["positions_processed"])
    heavy = int(metrics.get("attention_heavy_hitter_updates", -1))
    checks["attention_heavy_hitter_updates"] = (
        int(expectations["attention_heavy_hitter_updates_minimum"])
        <= heavy
        <= int(expectations["attention_heavy_hitter_updates_maximum"])
    )
    return checks


def _deterministic_metrics(metrics: dict[str, int]) -> dict[str, int]:
    return {
        name: value
        for name, value in metrics.items()
        if name not in {"elapsed_ns", "q7_elapsed_ns"}
    }


def evaluate_native_olmoe_sustained_context(
    *,
    package: str | Path,
    manifest_sha256: str,
    library: str | Path,
    dataset: str | Path,
    corpus_manifest: str | Path,
    teacher_reference: str | Path,
    teacher_arrays: str | Path,
    protocol: str | Path,
    protocol_sha256: str,
    out: str | Path,
    threads: int | None = None,
) -> dict[str, Any]:
    """Run the prospectively frozen 8x128 sustained native attention gate."""

    output_path = Path(out).expanduser().resolve()
    if output_path.exists():
        raise ValueError("sustained result target already exists")
    package_path = Path(package).expanduser().resolve()
    library_path = Path(library).expanduser().resolve()
    dataset_path = Path(dataset).expanduser().resolve()
    corpus_manifest_path = Path(corpus_manifest).expanduser().resolve()
    reference_path = Path(teacher_reference).expanduser().resolve()
    arrays_path = Path(teacher_arrays).expanduser().resolve()
    protocol_path = Path(protocol).expanduser().resolve()
    reference = _read_object(reference_path, "sustained teacher reference")
    protocol_value = _read_object(protocol_path, "sustained protocol")
    identities = {
        "native_library_sha256": sha256_file(library_path),
        "dataset_sha256": sha256_file(dataset_path),
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "teacher_reference_sha256": sha256_file(reference_path),
        "teacher_arrays_sha256": sha256_file(arrays_path),
        "protocol_sha256": sha256_file(protocol_path),
    }
    scope = protocol_value.get("scope")
    frozen_threads = scope.get("candidate_threads") if isinstance(scope, dict) else None
    expected_bands = [
        {"name": name, "start": start, "stop": stop}
        for name, start, stop in _QUALITY_BANDS
    ]
    if (
        identities["protocol_sha256"] != protocol_sha256.lower()
        or protocol_value.get("schema_version") != 1
        or protocol_value.get("experiment")
        != "olmoe_native_sustained_context_confirmation"
        or protocol_value.get("status") != "frozen_before_candidate_execution"
        or protocol_value.get("package_manifest_sha256") != manifest_sha256.lower()
        or any(
            protocol_value.get(name) != value
            for name, value in identities.items()
            if name != "protocol_sha256"
        )
        or protocol_value.get("thresholds") != _THRESHOLDS
        or protocol_value.get("sequences") != _SEQUENCES
        or protocol_value.get("tokens_per_sequence") != _TOKENS_PER_SEQUENCE
        or protocol_value.get("quality_bands") != expected_bands
        or not isinstance(frozen_threads, int)
        or isinstance(frozen_threads, bool)
        or frozen_threads <= 0
        or (threads is not None and threads != frozen_threads)
        or scope.get("candidate_device") != "cpu"
        or scope.get("candidate_transformers_model_shell") is not False
        or scope.get("reset_replay_sequence") != 0
    ):
        raise ValueError("native OLMoE sustained protocol contract is invalid")
    evaluator_sources = _authenticate_evaluator_sources(protocol_value)
    _validate_teacher_source(reference, protocol_value)
    if (
        reference.get("schema_version") != 1
        or reference.get("experiment") != "olmoe_untouched_teacher_causal_reference"
        or reference.get("configuration", {}).get("weights_modified") is not False
        or reference.get("dataset", {}).get("sha256") != identities["dataset_sha256"]
        or reference.get("dataset", {}).get("sequences") != _SEQUENCES
        or reference.get("dataset", {}).get("tokens_per_sequence")
        != _TOKENS_PER_SEQUENCE
        or reference.get("dataset", {}).get("prediction_positions")
        != _SEQUENCES * _POSITIONS_PER_SEQUENCE
        or reference.get("dataset", {}).get("input_identity")
        != protocol_value.get("input_identity")
        or reference.get("dataset", {}).get("input_ids")
        != protocol_value.get("input_ids")
        or reference.get("arrays", {}).get("sha256")
        != identities["teacher_arrays_sha256"]
        or _teacher_configuration(reference) != _TEACHER_CONFIGURATION
        or scope.get("teacher_configuration") != _TEACHER_CONFIGURATION
    ):
        raise ValueError("native OLMoE sustained teacher contract is invalid")
    input_ids = protocol_value["input_ids"]
    if sha256_json(input_ids) != protocol_value["input_identity"]:
        raise ValueError("sustained input identity authentication failed")
    model = protocol_value["model"]
    policy = protocol_value["attention_policy"]
    expectations = _attention_expectations(model, policy)
    q7_expectations = _q7_expectations(model)
    if protocol_value.get("attention_expectations_per_sequence") != expectations:
        raise ValueError("sustained attention expectations are invalid")
    if protocol_value.get("q7_expectations_per_sequence") != q7_expectations:
        raise ValueError("sustained Q7 expectations are invalid")
    prediction_positions = _SEQUENCES * _POSITIONS_PER_SEQUENCE
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"logits", "hidden", "targets"}:
            raise ValueError("sustained teacher arrays have unexpected keys")
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
        raise ValueError("sustained teacher array shapes are invalid")

    all_rows: list[dict[str, float | bool | int]] = []
    band_rows: dict[str, list[dict[str, float | bool | int]]] = {
        name: [] for name, _start, _stop in _QUALITY_BANDS
    }
    sequence_results: list[dict[str, Any]] = []
    total_q7_bytes = 0
    replay_reference: dict[str, Any] | None = None
    load_started = time.perf_counter()
    runtime = OLMoENativePackageRuntime(
        package_path,
        manifest_sha256=manifest_sha256,
        library=library_path,
        threads=threads,
    )
    cold_load_seconds = time.perf_counter() - load_started
    try:
        effective_threads = (
            int(runtime.manifest["runtime"]["kernel_threads"])
            if threads is None
            else int(threads)
        )
        retokenized, _corpus = _validate_corpus_manifest(
            dataset_path,
            corpus_manifest_path,
            runtime.tokenizer,
            tokenizer_sha256=runtime.manifest["files"]["tokenizer/tokenizer.json"][
                "sha256"
            ],
        )
        if (
            retokenized != input_ids
            or sha256_json(retokenized) != protocol_value["input_identity"]
            or runtime.manifest["runtime"]["attention_policy"] != policy
            or runtime.manifest.get("source", {}).get("revision")
            != protocol_value["source_revision"]
            or runtime.manifest["files"]["model/config.json"]["sha256"]
            != protocol_value["source_config_sha256"]
            or runtime.manifest["files"]["mlp/experts.q7"]["bytes"]
            != q7_expectations["artifact_bytes"]
            or effective_threads != frozen_threads
            or not runtime.runtime.attention_metrics_available
        ):
            raise ValueError(
                "sustained package, tokenizer, policy, or thread identity is invalid"
            )
        offset = 0
        for sequence_index, sequence in enumerate(input_ids):
            runtime.reset()
            sequence_rows: list[dict[str, float | bool | int]] = []
            top1_tokens: list[int] = []
            hidden_digest = hashlib.sha256()
            logit_digest = hashlib.sha256()
            started = time.perf_counter()
            for position, token_id in enumerate(sequence[:-1]):
                native_result = runtime.runtime.forward([token_id])
                native_hidden, native_logits = runtime.runtime.last_diagnostics()
                diagnostic_top1 = int(np.argmax(native_logits))
                if diagnostic_top1 != native_result.next_token:
                    raise ValueError(
                        "native diagnostic argmax differs from returned token"
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
                sequence_rows.append(row)
                for name, start, stop in _QUALITY_BANDS:
                    if start <= position < stop:
                        band_rows[name].append(row)
                        break
                top1_tokens.append(diagnostic_top1)
                _update_diagnostic_hashes(
                    hidden_digest,
                    logit_digest,
                    native_hidden,
                    native_logits,
                )
                offset += 1
            elapsed = time.perf_counter() - started
            metrics = dict(runtime.runtime.last_result.metrics)
            structural = _structural_checks(
                metrics,
                expectations,
                position=runtime.runtime.position,
            )
            structural["q7_scheduled_bytes"] = (
                metrics["q7_scheduled_bytes"]
                == q7_expectations["scheduled_bytes_per_sequence"]
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
                    "structural_checks": structural,
                }
            sequence_results.append(
                {
                    "sequence": sequence_index,
                    "input_ids": sequence,
                    "elapsed_seconds": elapsed,
                    "metrics": _aggregate(sequence_rows),
                    "native_metrics": metrics,
                    "diagnostic_hashes": diagnostic_hashes,
                    "structural_checks": structural,
                    "structural_passed": all(structural.values()),
                }
            )
        if replay_reference is None:
            raise RuntimeError("sustained reset replay reference is missing")
        runtime.reset()
        replay_tokens: list[int] = []
        replay_hidden_digest = hashlib.sha256()
        replay_logit_digest = hashlib.sha256()
        replay_started = time.perf_counter()
        for token_id in input_ids[0][:-1]:
            replay_result = runtime.runtime.forward([token_id])
            replay_hidden, replay_logits = runtime.runtime.last_diagnostics()
            replay_tokens.append(replay_result.next_token)
            _update_diagnostic_hashes(
                replay_hidden_digest,
                replay_logit_digest,
                replay_hidden,
                replay_logits,
            )
        replay_seconds = time.perf_counter() - replay_started
        replay_metrics = dict(runtime.runtime.last_result.metrics)
        replay_hashes = {
            "hidden_sha256": replay_hidden_digest.hexdigest(),
            "logits_sha256": replay_logit_digest.hexdigest(),
        }
        replay_structural = _structural_checks(
            replay_metrics,
            expectations,
            position=runtime.runtime.position,
        )
        replay_structural["q7_scheduled_bytes"] = (
            replay_metrics["q7_scheduled_bytes"]
            == q7_expectations["scheduled_bytes_per_sequence"]
        )
        reset_replay = {
            "sequence": 0,
            "elapsed_seconds": replay_seconds,
            "top1_tokens_match": replay_tokens == replay_reference["top1_tokens"],
            "diagnostic_hashes": replay_hashes,
            "diagnostic_hashes_match": replay_hashes
            == replay_reference["diagnostic_hashes"],
            "native_metrics": replay_metrics,
            "deterministic_metrics_match": (
                _deterministic_metrics(replay_metrics)
                == replay_reference["deterministic_metrics"]
            ),
            "structural_checks": replay_structural,
            "passed": (
                replay_tokens == replay_reference["top1_tokens"]
                and replay_hashes == replay_reference["diagnostic_hashes"]
                and _deterministic_metrics(replay_metrics)
                == replay_reference["deterministic_metrics"]
                and all(replay_structural.values())
            ),
        }
        manifest_after_runtime = runtime.manifest
    finally:
        runtime.close()

    post_manifest = validate_olmoe_native_package(
        package_path,
        expected_manifest_sha256=manifest_sha256,
    )
    source_shards = protocol_value["source_shard_sha256"]
    source_model_path = Path(reference["source"]["model"]).expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    post_run_authentication = {
        "package": post_manifest == manifest_after_runtime,
        "library": sha256_file(library_path) == identities["native_library_sha256"],
        "protocol": sha256_file(protocol_path) == identities["protocol_sha256"],
        "dataset": sha256_file(dataset_path) == identities["dataset_sha256"],
        "corpus_manifest": (
            sha256_file(corpus_manifest_path) == identities["corpus_manifest_sha256"]
        ),
        "teacher_reference": sha256_file(reference_path)
        == identities["teacher_reference_sha256"],
        "teacher_arrays": sha256_file(arrays_path)
        == identities["teacher_arrays_sha256"],
        "teacher_source_config": (
            sha256_file(source_model_path / "config.json")
            == protocol_value["source_config_sha256"]
        ),
        "teacher_source_index": (
            sha256_file(source_model_path / "model.safetensors.index.json")
            == protocol_value["source_index_sha256"]
        ),
        "teacher_source_shards": _post_source_shards(reference, source_shards),
        "evaluator_sources": all(
            sha256_file(repository / relative) == expected_hash
            for relative, expected_hash in evaluator_sources.items()
        ),
    }
    aggregate = _aggregate(all_rows)
    bands = {name: _aggregate(rows) for name, rows in band_rows.items()}
    layers = int(model["layers"])
    hidden = int(model["hidden_size"])
    intermediate = int(model["intermediate_size"])
    experts = int(model["experts"])
    ideal_q4_bytes_per_position = layers * experts * 3 * hidden * intermediate // 2
    q7_traffic_fraction = total_q7_bytes / (
        prediction_positions * ideal_q4_bytes_per_position
    )
    attention_read_fraction = float(expectations["attention_logical_read_fraction"])
    evidence_checks = {
        "sequence_count": _SEQUENCES >= _THRESHOLDS["minimum_sequences"],
        "prediction_positions": (
            prediction_positions >= _THRESHOLDS["minimum_prediction_positions"]
        ),
        "band_positions": all(
            metrics["prediction_positions"]
            >= _THRESHOLDS["minimum_band_prediction_positions"]
            for metrics in bands.values()
        ),
        "q7_traffic": (
            q7_traffic_fraction <= _THRESHOLDS["maximum_q7_traffic_fraction"]
        ),
        "attention_logical_read_fraction": (
            attention_read_fraction
            <= _THRESHOLDS["maximum_attention_logical_read_fraction"]
        ),
        "sequence_structural_checks": all(
            result["structural_passed"] for result in sequence_results
        ),
        "reset_replay": reset_replay["passed"],
        "post_run_authentication": all(post_run_authentication.values()),
    }
    quality_checks = _quality_checks("overall", aggregate)
    for name, metrics in bands.items():
        quality_checks.update(_quality_checks(name, metrics))
    checks = {**evidence_checks, **quality_checks}
    gate_passed = all(checks.values())
    evidence_passed = all(evidence_checks.values())
    quality_passed = all(quality_checks.values())
    if gate_passed:
        decision = "promote_olmoe_sustained_causal_stability_boundary"
    elif evidence_passed and not quality_passed:
        decision = "run_matched_q7_dense_attention_control"
    else:
        decision = "stop_and_diagnose_runtime_or_evidence_failure"
    report = {
        "schema_version": 1,
        "experiment": "olmoe_native_sustained_context_confirmation",
        "status": (
            "frozen_confirmation_passed"
            if gate_passed
            else "frozen_confirmation_failed"
        ),
        "artifacts": {
            "package_manifest_sha256": manifest_sha256.lower(),
            **identities,
            "evaluator_source_sha256": evaluator_sources,
        },
        "configuration": {
            "sequences": _SEQUENCES,
            "tokens_per_sequence": _TOKENS_PER_SEQUENCE,
            "prediction_positions": prediction_positions,
            "cpu_only_candidate": True,
            "candidate_threads": effective_threads,
            "transformers_model_shell_used_by_candidate": False,
            "attention_policy": policy,
        },
        "metrics": aggregate,
        "position_bands": bands,
        "per_position_offset": {
            str(position): _aggregate(
                [row for row in all_rows if row["position"] == position]
            )
            for position in range(_POSITIONS_PER_SEQUENCE)
        },
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
        "attention": {
            "expectations_per_sequence": expectations,
            "q7_expectations_per_sequence": q7_expectations,
            "reset_replay": reset_replay,
        },
        "traffic": {
            "q7_scheduled_bytes": total_q7_bytes,
            "all_expert_ideal_q4_bytes": (
                prediction_positions * ideal_q4_bytes_per_position
            ),
            "q7_fraction_of_all_expert_ideal_q4": q7_traffic_fraction,
            "attention_logical_read_bytes_per_sequence": expectations[
                "attention_logical_read_bytes"
            ],
            "dense_full_context_logical_kv_bytes_per_sequence": expectations[
                "dense_full_context_logical_kv_bytes"
            ],
            "attention_logical_read_fraction": attention_read_fraction,
            "measured_hardware_traffic": False,
        },
        "performance": {
            "cold_authentication_and_load_seconds": cold_load_seconds,
            "sequence_seconds": [
                result["elapsed_seconds"] for result in sequence_results
            ],
            "total_sequence_seconds": sum(
                result["elapsed_seconds"] for result in sequence_results
            ),
            "total_native_elapsed_seconds": sum(
                result["native_metrics"]["elapsed_ns"] / 1.0e9
                for result in sequence_results
            ),
            "total_q7_elapsed_seconds": sum(
                result["native_metrics"]["q7_elapsed_ns"] / 1.0e9
                for result in sequence_results
            ),
            "reset_replay_seconds": reset_replay["elapsed_seconds"],
        },
        "sequence_results": sequence_results,
        "position_results": all_rows,
        "post_run_authentication": post_run_authentication,
        "thresholds": _THRESHOLDS,
        "checks": checks,
        "evidence_checks": evidence_checks,
        "quality_checks": quality_checks,
        "evidence_passed": evidence_passed,
        "quality_passed": quality_passed,
        "gate_passed": gate_passed,
        "decision": decision,
        "limitations": [
            "Logical attention and Q7 bytes are algorithmic counters, not hardware-counter DRAM traffic.",
            "This fixed eight-text corpus is a sustained-context gate, not a broad language benchmark.",
            "An authenticated quality failure combines Q7 and bounded-attention drift and requires a dense-attention control for attribution.",
        ],
    }
    atomic_json(output_path, report)
    return report


__all__ = [
    "evaluate_native_olmoe_sustained_context",
    "freeze_olmoe_sustained_context_protocol",
]
