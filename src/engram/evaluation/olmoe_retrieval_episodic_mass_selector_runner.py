"""Freeze the authenticated train-only C28 mass-selector protocol.

This module implements the immutable protocol handoff and its authenticated
fit-screen orchestration.  Numerical fitting lives in
:mod:`olmoe_retrieval_episodic_mass_selector` and is never started by
``freeze``.  The handoff authenticates the exact full-visible capacity pass,
its cached trace manifest, and the complete transitive source closure before
writing a new immutable protocol.

Every caller-supplied path is checked lexically as one batch before any
filesystem operation.  The authenticated confirmation descriptor is copied
from the capacity artifacts by value; its filename is never resolved, opened,
stat'ed, or hashed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file as load_safetensors
from safetensors.numpy import save_file as save_safetensors

import engram.evaluation.olmoe_retrieval_episodic_full_visible_simplex_oracle as full
import engram.evaluation.olmoe_retrieval_episodic_mass_selector as selector
from engram.utils import sha256_file

_SCHEMA_VERSION = 1
_PROTOCOL_EXPERIMENT = "olmoe_q7_retrieval_episodic_mass_selector_protocol"
_PROTOCOL_STATUS = "frozen_before_train_mass_selector_fit"
_RESULT_EXPERIMENT = "olmoe_q7_retrieval_episodic_mass_selector_oof_train_screen"
_PASSED_STATUS = "train_mass_selector_oof_gate_passed"
_FAILED_STATUS = "train_mass_selector_oof_gate_failed"
_FINAL_ARTIFACT_FAILED_STATUS = (
    "train_mass_selector_oof_gate_passed_but_final_artifact_failed"
)

_EXPECTED_CAPACITY_PROTOCOL_SHA256 = (
    "61e4b6da682bb501b21a0fa41d961f1f092e92e750ff913057c2f20ff4f34734"
)
_EXPECTED_CAPACITY_RESULT_SHA256 = (
    "a8711f07fdcbe48b16ebe962aae1962e4613873640eba20bc7fa88238216aee1"
)
_EXPECTED_TRACE_MANIFEST_SHA256 = (
    "ec80fb57e1c9c684b9d1e95ad672b69afa30a7915ea427cec9c76ca15219fef4"
)

_CAPACITY_PROTOCOL_EXPERIMENT = (
    "olmoe_q7_retrieval_episodic_full_visible_simplex_protocol"
)
_CAPACITY_PROTOCOL_STATUS = "frozen_before_full_visible_train_capture"
_CAPACITY_RESULT_EXPERIMENT = (
    "olmoe_q7_retrieval_episodic_full_visible_simplex_train_screen"
)
_CAPACITY_RESULT_STATUS = "train_full_visible_simplex_gate_passed"
_TRACE_MANIFEST_EXPERIMENT = "olmoe_q7_retrieval_episodic_full_visible_simplex_capture"

_CONFIRMATION_FILENAME = "confirmation.jsonl"
_CORE_SOURCE = "src/engram/evaluation/olmoe_retrieval_episodic_mass_selector.py"
_RUNNER_SOURCE = (
    "src/engram/evaluation/olmoe_retrieval_episodic_mass_selector_runner.py"
)
_SOURCE_FILES = tuple(
    dict.fromkeys((*full._SOURCE_FILES, _CORE_SOURCE, _RUNNER_SOURCE))
)

_RECORDS = 8
_POSITIONS_PER_RECORD = 128
_READ_POSITIONS_PER_RECORD = 32
_LAYERS = 16
_QUERY_HEADS = 16
_COMPONENTS = 28
_RANK = 16
_PARAMETER_COUNT = 25_600
_BF16_PARAMETER_BYTES = 51_200
_SELECTOR_SCRATCH_BYTES = 6_400
_MACS_PER_TOKEN = 229_376
_MACS_PER_128_TOKEN_SEQUENCE = 29_360_128

_TRAINING_STEPS = 1_536
_FINAL_TRAINING_STEPS = 2_048
_WARMUP_STEPS = 96
_FINAL_WARMUP_STEPS = 128
_PEAK_LEARNING_RATE = 0.005
_FINAL_LEARNING_RATE = 0.0005
_BETA1 = 0.9
_BETA2 = 0.999
_EPSILON = 1.0e-8
_UV_WEIGHT_DECAY = 1.0e-4
_GRADIENT_CLIP_NORM = 1.0
_ROWS_PER_LAYER_PER_STEP = 2
_TRAIN_ROWS_PER_LAYER_PER_FOLD = 192
_STEPS_PER_EPOCH = 96
_EPOCHS = 16
_INITIAL_U_STANDARD_DEVIATION = 0.02
_DELTA_CLAMP = 16.0
_INIT_SEED_BASE = 2_026_073_001
_SHUFFLE_SEED_BASE = 2_026_073_002

_DENSE_FULL_CONTEXT_LOGICAL_READ_BYTES = 2_164_260_864
_MAXIMUM_DEPLOYABLE_BYTES_AT_45_PERCENT = 973_917_388
_FIXED_COMBINED_ATTENTION_AND_EPISODIC_TRAFFIC_BYTES = 714_866_688
_SELECTOR_ALL_TOKEN_WEIGHT_TRAFFIC_BYTES = 6_553_600
_TOTAL_LOGICAL_TRAFFIC_BYTES = 721_420_288
_REMAINING_HEADROOM_BYTES = 252_497_100

_FOLDS = (
    (0, (0, 4)),
    (1, (1, 5)),
    (2, (2, 6)),
    (3, (3, 7)),
)


def _reject_confirmation_paths(
    values: Sequence[tuple[str, str | Path]],
) -> None:
    """Reject literal confirmation leaves for all inputs without filesystem I/O."""

    for label, value in values:
        requested = Path(value)
        if any(part.casefold() == _CONFIRMATION_FILENAME for part in requested.parts):
            raise ValueError(f"{label} cannot name the confirmation split")


def _guard_paths(values: Sequence[tuple[str, str | Path]]) -> None:
    """Guard all lexical paths first, then resolve parents and reject leaf links."""

    # Keep this as a separate complete pass.  A forbidden later argument must
    # prevent even a stat of an otherwise valid earlier argument.
    _reject_confirmation_paths(values)
    for label, value in values:
        requested = Path(value).expanduser()
        resolved = requested.parent.resolve(strict=False) / requested.name
        if any(part.casefold() == _CONFIRMATION_FILENAME for part in resolved.parts):
            raise ValueError(f"{label} resolves inside the confirmation split")
        # Do not resolve the leaf.  A leaf symlink may target confirmation data.
        if requested.is_symlink():
            raise ValueError(f"{label} must not be a symlink")


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _binding(path: Path, digest: str) -> dict[str, str]:
    return {"path": str(path), "sha256": digest.lower()}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _checked_file(
    value: str | Path,
    digest: str,
    *,
    expected_digest: str,
    label: str,
) -> Path:
    """Authenticate one already-guarded regular file without resolving its leaf."""

    if (
        not _is_sha256(digest)
        or digest != expected_digest
        or not _is_sha256(expected_digest)
    ):
        raise ValueError(f"{label} is not the frozen artifact")
    requested = Path(value).expanduser()
    source = requested.parent.resolve(strict=False) / requested.name
    if requested.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    if not source.is_file() or sha256_file(source) != digest:
        raise ValueError(f"{label} authentication failed")
    return source


def _new_output(value: str | Path, label: str) -> Path:
    """Return a new output path after guarding the parent and lexical leaf."""

    _guard_paths(((label, value),))
    requested = Path(value).expanduser()
    output = requested.parent.resolve(strict=False) / requested.name
    if requested.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    if output.exists():
        raise ValueError(f"{label} already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _publish_new_file(temporary: Path, destination: Path, label: str) -> None:
    """Atomically publish a new file without ever replacing an existing leaf."""

    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise ValueError(f"{label} already exists") from error
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_descriptor = os.open(destination.parent, directory_flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _atomic_json_new(path: Path, value: Mapping[str, Any]) -> None:
    """Write one durable JSON object with no-overwrite publication."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _publish_new_file(temporary, path, "mass-selector JSON output")
    finally:
        temporary.unlink(missing_ok=True)


def _source_inventory() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    return {name: sha256_file(repository / name) for name in _SOURCE_FILES}


def _resource_contract() -> dict[str, Any]:
    """Build and internally prove the frozen selector accounting."""

    calculated_parameters = (
        _LAYERS * _COMPONENTS * _RANK  # U
        + _LAYERS * _RANK * _COMPONENTS  # V
        + _LAYERS * _QUERY_HEADS * _RANK  # E
        + _LAYERS * _QUERY_HEADS * _COMPONENTS  # B
    )
    if calculated_parameters != _PARAMETER_COUNT:
        raise AssertionError("mass-selector parameter arithmetic changed")
    if _PARAMETER_COUNT * 2 != _BF16_PARAMETER_BYTES:
        raise AssertionError("mass-selector BF16 state arithmetic changed")
    if 10_534_912 + _BF16_PARAMETER_BYTES != 10_586_112:
        raise AssertionError("mass-selector combined state arithmetic changed")
    calculated_scratch = (
        _QUERY_HEADS * _COMPONENTS * 4
        + _QUERY_HEADS * _RANK * 4
        + _QUERY_HEADS * _COMPONENTS * 4
        + _QUERY_HEADS * _COMPONENTS * 4
    )
    if calculated_scratch != _SELECTOR_SCRATCH_BYTES:
        raise AssertionError("mass-selector scratch arithmetic changed")
    calculated_macs = (
        _LAYERS * _QUERY_HEADS * (_COMPONENTS * _RANK + _RANK * _COMPONENTS)
    )
    if calculated_macs != _MACS_PER_TOKEN:
        raise AssertionError("mass-selector MAC arithmetic changed")
    if _MACS_PER_TOKEN * _POSITIONS_PER_RECORD != _MACS_PER_128_TOKEN_SEQUENCE:
        raise AssertionError("mass-selector sequence MAC arithmetic changed")
    if (
        _BF16_PARAMETER_BYTES * _POSITIONS_PER_RECORD
        != _SELECTOR_ALL_TOKEN_WEIGHT_TRAFFIC_BYTES
    ):
        raise AssertionError("mass-selector weight-traffic arithmetic changed")
    if (
        _FIXED_COMBINED_ATTENTION_AND_EPISODIC_TRAFFIC_BYTES
        + _SELECTOR_ALL_TOKEN_WEIGHT_TRAFFIC_BYTES
        != _TOTAL_LOGICAL_TRAFFIC_BYTES
    ):
        raise AssertionError("mass-selector total-traffic arithmetic changed")
    if (
        _MAXIMUM_DEPLOYABLE_BYTES_AT_45_PERCENT - _TOTAL_LOGICAL_TRAFFIC_BYTES
        != _REMAINING_HEADROOM_BYTES
    ):
        raise AssertionError("mass-selector headroom arithmetic changed")
    if _TOTAL_LOGICAL_TRAFFIC_BYTES * 3 != _DENSE_FULL_CONTEXT_LOGICAL_READ_BYTES:
        raise AssertionError("mass-selector dense-fraction arithmetic changed")
    return {
        "parameter_count": _PARAMETER_COUNT,
        "serialized_parameter_dtype": "BF16",
        "serialized_parameter_bytes": _BF16_PARAMETER_BYTES,
        "deployment_artifact_contains_BF16_only": True,
        "deployment_artifact_BF16_tensor_bytes": _BF16_PARAMETER_BYTES,
        "FP32_training_audit_copy_loaded_by_runtime": False,
        "fixed_attention_state_bytes": 10_534_912,
        "combined_attention_and_selector_state_bytes": 10_586_112,
        "conservative_selector_scratch_bytes": _SELECTOR_SCRATCH_BYTES,
        "selector_multiply_accumulates_per_token": _MACS_PER_TOKEN,
        "selector_multiply_accumulates_per_128_token_sequence": (
            _MACS_PER_128_TOKEN_SEQUENCE
        ),
        "conservative_selector_weight_traffic_bytes_per_128_token_sequence": (
            _SELECTOR_ALL_TOKEN_WEIGHT_TRAFFIC_BYTES
        ),
        "fixed_combined_attention_and_episodic_traffic_bytes": (
            _FIXED_COMBINED_ATTENTION_AND_EPISODIC_TRAFFIC_BYTES
        ),
        "total_logical_traffic_bytes_per_128_token_sequence": (
            _TOTAL_LOGICAL_TRAFFIC_BYTES
        ),
        "dense_full_context_logical_read_bytes": (
            _DENSE_FULL_CONTEXT_LOGICAL_READ_BYTES
        ),
        "fraction_of_dense_full_context_logical_reads": (1.0 / 3.0),
        "maximum_deployable_bytes_at_45_percent": (
            _MAXIMUM_DEPLOYABLE_BYTES_AT_45_PERCENT
        ),
        "remaining_headroom_bytes": _REMAINING_HEADROOM_BYTES,
        "new_KV_state_bytes": 0,
        "new_KV_read_traffic_bytes": 0,
        "selector_weight_traffic_assumes_reload_for_every_token": True,
        "future_native_fused_single_value_pass_required": True,
        "unfused_second_value_read_pass_not_authorized": True,
    }


def _fold_contract() -> list[dict[str, Any]]:
    all_records = tuple(range(_RECORDS))
    rows: list[dict[str, Any]] = []
    for fold_index, heldout in _FOLDS:
        heldout_set = set(heldout)
        rows.append(
            {
                "fold_index": fold_index,
                "training_record_indices": [
                    value for value in all_records if value not in heldout_set
                ],
                "heldout_record_indices": list(heldout),
                "initialization_seed": _INIT_SEED_BASE + fold_index,
                "shuffle_seed": _SHUFFLE_SEED_BASE + fold_index,
            }
        )
    return rows


def _selector_model_contract() -> dict[str, Any]:
    return {
        "components_per_query_head": _COMPONENTS,
        "query_heads": _QUERY_HEADS,
        "layers": _LAYERS,
        "rank": _RANK,
        "parameter_shapes": {
            "U": [_LAYERS, _COMPONENTS, _RANK],
            "V": [_LAYERS, _RANK, _COMPONENTS],
            "E": [_LAYERS, _QUERY_HEADS, _RANK],
            "B": [_LAYERS, _QUERY_HEADS, _COMPONENTS],
        },
        "input_features": (
            "training and cached evaluation use clipped centered log of the "
            "authenticated native mass over the 28 constructible components"
        ),
        "candidate_native_input_features": (
            "clipped centered raw pre-softmax scores with the same valid mask"
        ),
        "raw_native_score_capture_available": False,
        "raw_native_score_equivalence_claimed_by_this_protocol": False,
        "raw_native_score_feature_and_output_parity_required_before_rollout": True,
        "cached_training_feature_reconstruction": (
            "centered clipped log-mass is the authenticated causal feature; "
            "log-mass-to-reconstructed-score parity is an implementation "
            "reference, not evidence about the unavailable original q.k scores"
        ),
        "forward": (
            "delta=clamp(gauge((relu(features@U+E))@V+B),-16,+16); "
            "cached coefficients=normalize(native_mass*exp(delta)); candidate "
            "native coefficients=masked_softmax(raw_scores+delta)"
        ),
        "execution_arithmetic": (
            "BF16 parameters decoded to FP32 with FP32 accumulation, "
            "gauge, clamp, exponential, and normalization"
        ),
        "hidden_activation": "ReLU",
        "feature_clip": [-_DELTA_CLAMP, _DELTA_CLAMP],
        "gauge": "subtract mean delta over valid components per query head",
        "delta_clamp": [-_DELTA_CLAMP, _DELTA_CLAMP],
        "invalid_components": (
            "remain exactly masked with zero coefficient before and after delta"
        ),
        "loss": "direct squared post-Wo error against authenticated target residual",
        "loss_aggregation": (
            "same sequence, block-entry-position, layer, and global recovery "
            "aggregation as the full-visible C28 gate"
        ),
        "counterfactual_updates_hidden_or_cache": False,
        "production_integration": (
            "implement the fused raw-score path, capture authenticated raw "
            "scores, and pass feature/coefficient/post-Wo parity against the "
            "log-mass reference before any causal rollout"
        ),
    }


def _training_contract() -> dict[str, Any]:
    return {
        "cross_validation": {
            "folds": _fold_contract(),
            "out_of_fold_records": list(range(_RECORDS)),
            "one_final_checkpoint_per_fold": True,
            "heldout_checkpointing_or_early_stopping": False,
        },
        "steps": _TRAINING_STEPS,
        "warmup_steps": _WARMUP_STEPS,
        "optimizer": "AdamW",
        "learning_rate_schedule": "linear_warmup_then_cosine_decay",
        "peak_learning_rate": _PEAK_LEARNING_RATE,
        "final_learning_rate": _FINAL_LEARNING_RATE,
        "betas": [_BETA1, _BETA2],
        "epsilon": _EPSILON,
        "weight_decay": {
            "U": _UV_WEIGHT_DECAY,
            "V": _UV_WEIGHT_DECAY,
            "E": 0.0,
            "B": 0.0,
        },
        "global_gradient_clip_norm": _GRADIENT_CLIP_NORM,
        "deterministic_rows_per_layer_per_step": _ROWS_PER_LAYER_PER_STEP,
        "training_rows_per_layer_per_fold": _TRAIN_ROWS_PER_LAYER_PER_FOLD,
        "steps_per_epoch": _STEPS_PER_EPOCH,
        "epochs": _EPOCHS,
        "initialization": {
            "U": {
                "distribution": "normal",
                "mean": 0.0,
                "standard_deviation": _INITIAL_U_STANDARD_DEVIATION,
            },
            "V": "zero",
            "E": "zero",
            "B": "zero",
        },
        "initialization_seed_base": _INIT_SEED_BASE,
        "shuffle_seed_base": _SHUFFLE_SEED_BASE,
        "seed_derivation": "fold_seed=base+fold_index",
        "shuffle_generator": "numpy.random.Generator(numpy.random.PCG64)",
        "shuffle_source_order": (
            "training record index ascending, then authenticated read "
            "position ascending; one independent 192-row permutation per "
            "layer and epoch, consumed two rows at a time"
        ),
        "shuffle_implementation_source": _CORE_SOURCE,
        "training_precision": "FP32",
        "training_device": "cuda",
        "CUDA_deterministic_algorithms_required": True,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "checkpoint_selection": "fixed final step only",
        "final_all_train_fit_on_out_of_fold_pass": {
            "training_record_indices": list(range(_RECORDS)),
            "steps": _FINAL_TRAINING_STEPS,
            "warmup_steps": _FINAL_WARMUP_STEPS,
            "steps_per_epoch": 128,
            "epochs": _EPOCHS,
            "initialization_seed": _INIT_SEED_BASE + len(_FOLDS),
            "shuffle_seed": _SHUFFLE_SEED_BASE + len(_FOLDS),
            "not_used_for_out_of_fold_gate": True,
            "purpose": "single BF16 artifact for train-only native integration",
        },
    }


def _serialization_contract() -> dict[str, Any]:
    return {
        "production_parameter_dtype": "BF16",
        "BF16_rounding": "round_to_nearest_even",
        "FP32_and_BF16_out_of_fold_evaluation_required": True,
        "mass_vs_reconstructed_score_coefficient_max_abs_tolerance": 1.0e-6,
        "mass_vs_reconstructed_score_delta_max_abs_tolerance": 1.0e-6,
        "native_raw_score_parity_deferred_to_native_integration": True,
        "zero_model_native_coefficient_max_abs_tolerance": 1.0e-6,
        "zero_model_native_post_Wo_output_max_abs_tolerance": 1.0e-6,
        "deterministic_serialized_replay_exact": True,
        "final_native_artifact_contains_BF16_only": True,
        "FP32_training_audit_artifact_not_runtime_loadable": True,
    }


def _progression_gate_contract() -> dict[str, Any]:
    return {
        "arms_required": ["FP32", "BF16_RNE"],
        "minimum_global_recovery": 0.50,
        "minimum_every_out_of_fold_sequence_recovery": 0.25,
        "minimum_every_block_entry_position_recovery": 0.25,
        "minimum_positive_recovery_layers": 12,
        "maximum_BF16_global_recovery_drop_from_FP32": 0.005,
        "finite_coefficients_outputs_and_metrics": True,
        "simplex_and_invalid_mask_invariants": True,
        "mass_vs_reconstructed_score_reference_parity_required": True,
        "native_raw_score_parity_not_claimed": True,
        "native_raw_score_feature_and_output_parity_required_before_rollout": True,
        "zero_model_native_coefficient_and_output_parity_required": True,
        "deterministic_serialized_replay_exact": True,
        "resource_contract_pass_required": True,
        "final_all_train_artifact_sanity_required_for_native_integration": True,
    }


def _scope_contract() -> dict[str, Any]:
    return {
        "split": "train",
        "records": _RECORDS,
        "read_positions_per_record": _READ_POSITIONS_PER_RECORD,
        "out_of_fold_selector_fit": True,
        "causal_features_only": True,
        "cached_trace_training_and_evaluation": True,
        "native_or_packaged_causal_rollout": False,
        "development_outcomes_used": False,
        "confirmation_file_access_permitted": False,
        "semantic_or_M3_pass": False,
    }


def _progression_authority_contract() -> dict[str, Any]:
    return {
        "on_pass": (
            "authorize implementing the fused CPU selector and authenticating "
            "raw-score feature/coefficient/post-Wo parity; do not roll out yet"
        ),
        "on_fail": (
            "do not tune on heldout fold outcomes; revise the causal "
            "feature/model class under a newly frozen train-only protocol"
        ),
        "does_not_authorize_confirmation": True,
        "does_not_pass_semantic_or_M3_gate": True,
        "does_not_authorize_native_runtime_or_development_rollout": True,
    }


def _capacity_source_inventory_is_current(
    capacity_protocol: Mapping[str, Any],
) -> bool:
    return capacity_protocol.get("source_sha256") == full._source_inventory()


def _validate_capacity_artifacts(
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    protocol_binding: Mapping[str, str],
    manifest_binding: Mapping[str, str],
) -> None:
    """Fail closed unless the exact passed C28 capacity evidence is intact."""

    protocol_scope = protocol.get("scope")
    protocol_oracle = protocol.get("oracle_method")
    protocol_resource = protocol.get("resource_contract")
    result_decision = result.get("decision")
    result_post = result.get("post_solve_authentication")
    result_records = result.get("record_authentication")
    result_resource = result.get("resource_contract")
    arms = result.get("authoritative_arms")
    constructible = arms.get("constructible") if isinstance(arms, Mapping) else None
    feasible_metrics = (
        constructible.get("feasible_solution_metrics")
        if isinstance(constructible, Mapping)
        else None
    )
    qualification = (
        constructible.get("qualification")
        if isinstance(constructible, Mapping)
        else None
    )
    feasible_gate = (
        feasible_metrics.get("gate") if isinstance(feasible_metrics, Mapping) else None
    )
    trace_binding = result.get("trace_manifest")
    descriptor = protocol.get("authenticated_confirmation_descriptor")

    expected_protocol_resource = {
        "dense_full_context_logical_read_bytes": (
            _DENSE_FULL_CONTEXT_LOGICAL_READ_BYTES
        ),
        "fixed_attention_state_bytes": 10_534_912,
        "fixed_combined_attention_and_episodic_traffic_bytes": (
            _FIXED_COMBINED_ATTENTION_AND_EPISODIC_TRAFFIC_BYTES
        ),
        "fixed_fraction_of_dense_full_context_KV": 0.33030523255813954,
        "fresh_trace_and_oracle_solver_evaluator_only": True,
        "gamma_zero_earns_read_savings": False,
        "maximum_deployable_bytes_at_45_percent": (
            _MAXIMUM_DEPLOYABLE_BYTES_AT_45_PERCENT
        ),
        "oracle_shadow_trace_and_projection_evaluator_only": True,
        "predictor_weights_features_and_execution_not_counted": True,
        "remaining_selector_headroom_bytes": 259_050_700,
    }
    expected_result_resource = {
        "fixed_attention_state_bytes": 10_534_912,
        "fixed_combined_attention_and_episodic_traffic_bytes": (
            _FIXED_COMBINED_ATTENTION_AND_EPISODIC_TRAFFIC_BYTES
        ),
        "fixed_fraction_of_dense_full_context_KV": 0.33030523255813954,
        "future_selector_not_counted_by_this_capacity_screen": True,
        "new_KV_state_or_read_traffic_bytes": 0,
        "trace_and_solver_evaluator_only": True,
    }

    if (
        protocol.get("schema_version") != _SCHEMA_VERSION
        or protocol.get("experiment") != _CAPACITY_PROTOCOL_EXPERIMENT
        or protocol.get("status") != _CAPACITY_PROTOCOL_STATUS
        or protocol.get("confirmation_split_opened") is not False
        or not isinstance(protocol_scope, Mapping)
        or protocol_scope.get("split") != "train"
        or protocol_scope.get("records") != _RECORDS
        or protocol_scope.get("read_positions_per_record") != _READ_POSITIONS_PER_RECORD
        or protocol_scope.get("causal_selector") is not False
        or protocol_scope.get("semantic_or_M3_pass") is not False
        or protocol_scope.get("confirmation_file_access_permitted") is not False
        or not isinstance(protocol_oracle, Mapping)
        or protocol_oracle.get("authoritative_constructible_components") != _COMPONENTS
        or protocol_resource != expected_protocol_resource
        or not _capacity_source_inventory_is_current(protocol)
    ):
        raise ValueError("mass-selector capacity protocol changed")

    if (
        result.get("schema_version") != _SCHEMA_VERSION
        or result.get("experiment") != _CAPACITY_RESULT_EXPERIMENT
        or result.get("status") != _CAPACITY_RESULT_STATUS
        or result.get("protocol") != protocol_binding
        or result.get("confirmation_split_opened") is not False
        or not isinstance(result_decision, Mapping)
        or result_decision.get("train_full_visible_constructible_gate_passed")
        is not True
        or result_decision.get("train_only_causal_selector_authorized") is not True
        or result_decision.get("certified_optimistic_gate_passed") is not True
        or result_decision.get("failure_is_decisive") is not False
        or result_decision.get("confirmation_authorized") is not False
        or result_decision.get("development_authorized") is not False
        or result_decision.get("semantic_or_M3_gate_passed") is not False
        or not isinstance(result_post, Mapping)
        or not result_post
        or not all(value is True for value in result_post.values())
        or not isinstance(result_records, Mapping)
        or not result_records
        or not all(value is True for value in result_records.values())
        or result_resource != expected_result_resource
        or not isinstance(constructible, Mapping)
        or constructible.get("arm") != "constructible_28_way"
        or constructible.get("components_per_head") != _COMPONENTS
        or constructible.get("diagnostic_only") is not False
        or constructible.get("exact_native_anchor_included") is not False
        or constructible.get("progression_authority") is not True
        or constructible.get("feasible_gate_passed") is not True
        or constructible.get("optimistic_gate_passed") is not True
        or not isinstance(feasible_metrics, Mapping)
        or feasible_metrics.get("passed") is not True
        or not isinstance(feasible_gate, Mapping)
        or feasible_gate.get("passed") is not True
        or not isinstance(qualification, Mapping)
        or qualification.get("passed") is not True
        or not all(value is True for value in qualification.values())
        or not isinstance(trace_binding, Mapping)
        or trace_binding.get("path") != manifest_binding.get("path")
        or trace_binding.get("sha256") != manifest_binding.get("sha256")
        or trace_binding.get("record_count") != _RECORDS
    ):
        raise ValueError("mass-selector capacity result is not authoritative")

    if (
        manifest.get("schema_version") != _SCHEMA_VERSION
        or manifest.get("experiment") != _TRACE_MANIFEST_EXPERIMENT
        or manifest.get("format") != "safetensors"
        or manifest.get("protocol") != protocol_binding
        or manifest.get("record_order") != list(range(_RECORDS))
        or manifest.get("confirmation_split_opened") is not False
        or not isinstance(manifest.get("shards"), list)
        or len(manifest["shards"]) != _RECORDS
    ):
        raise ValueError("mass-selector capacity trace manifest changed")

    if (
        not isinstance(descriptor, Mapping)
        or descriptor.get("file") != _CONFIRMATION_FILENAME
        or descriptor.get("records") != _RECORDS
        or descriptor.get("tokens_per_record") != _POSITIONS_PER_RECORD + 1
        or descriptor.get("prediction_positions_per_record") != _POSITIONS_PER_RECORD
        or descriptor.get("answer_prediction_positions_per_record")
        != _READ_POSITIONS_PER_RECORD
        or not _is_sha256(descriptor.get("sha256"))
        or not _is_sha256(descriptor.get("record_identity_sha256"))
        or result.get("authenticated_confirmation_descriptor") != descriptor
    ):
        raise ValueError("mass-selector confirmation descriptor changed")

    historical = protocol.get("historical_bindings")
    if not isinstance(historical, Mapping):
        raise ValueError("mass-selector historical bindings changed")
    for binding in historical.values():
        if (
            isinstance(binding, Mapping)
            and isinstance(binding.get("path"), str)
            and Path(binding["path"]).name.casefold() == _CONFIRMATION_FILENAME
        ):
            raise ValueError("mass-selector history points at confirmation data")


def _authenticate_capacity_inputs(
    *,
    capacity_protocol: str | Path,
    capacity_protocol_sha256: str,
    capacity_result: str | Path,
    capacity_result_sha256: str,
    trace_manifest: str | Path,
    trace_manifest_sha256: str,
) -> dict[str, Any]:
    _guard_paths(
        (
            ("mass-selector capacity protocol", capacity_protocol),
            ("mass-selector capacity result", capacity_result),
            ("mass-selector trace manifest", trace_manifest),
        )
    )
    protocol_path = _checked_file(
        capacity_protocol,
        capacity_protocol_sha256,
        expected_digest=_EXPECTED_CAPACITY_PROTOCOL_SHA256,
        label="mass-selector capacity protocol",
    )
    result_path = _checked_file(
        capacity_result,
        capacity_result_sha256,
        expected_digest=_EXPECTED_CAPACITY_RESULT_SHA256,
        label="mass-selector capacity result",
    )
    manifest_path = _checked_file(
        trace_manifest,
        trace_manifest_sha256,
        expected_digest=_EXPECTED_TRACE_MANIFEST_SHA256,
        label="mass-selector trace manifest",
    )
    protocol = _read_json(protocol_path, "mass-selector capacity protocol")
    result = _read_json(result_path, "mass-selector capacity result")
    manifest = _read_json(manifest_path, "mass-selector trace manifest")
    protocol_binding = _binding(protocol_path, capacity_protocol_sha256)
    manifest_binding = _binding(manifest_path, trace_manifest_sha256)
    _validate_capacity_artifacts(
        protocol,
        result,
        manifest,
        protocol_binding=protocol_binding,
        manifest_binding=manifest_binding,
    )
    return {
        "capacity_protocol_path": protocol_path,
        "capacity_protocol_sha256": capacity_protocol_sha256,
        "capacity_protocol": protocol,
        "capacity_result_path": result_path,
        "capacity_result_sha256": capacity_result_sha256,
        "capacity_result": result,
        "trace_manifest_path": manifest_path,
        "trace_manifest_sha256": trace_manifest_sha256,
        "trace_manifest": manifest,
    }


def _post_freeze_authentication(context: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "capacity_protocol": (
            sha256_file(context["capacity_protocol_path"])
            == context["capacity_protocol_sha256"]
        ),
        "capacity_result": (
            sha256_file(context["capacity_result_path"])
            == context["capacity_result_sha256"]
        ),
        "trace_manifest": (
            sha256_file(context["trace_manifest_path"])
            == context["trace_manifest_sha256"]
        ),
        "capacity_source_inventory": (
            context["capacity_protocol"]["source_sha256"] == full._source_inventory()
        ),
        "mass_selector_source_inventory": True,
        "confirmation_not_opened": True,
    }


def freeze_mass_selector_protocol(
    *,
    capacity_protocol: str | Path,
    capacity_protocol_sha256: str,
    capacity_result: str | Path,
    capacity_result_sha256: str,
    trace_manifest: str | Path,
    trace_manifest_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    """Authenticate the C28 capacity pass and atomically freeze selector fit."""

    paths = (
        ("capacity protocol", capacity_protocol),
        ("capacity result", capacity_result),
        ("trace manifest", trace_manifest),
        ("protocol output", out),
    )
    _guard_paths(paths)
    context = _authenticate_capacity_inputs(
        capacity_protocol=capacity_protocol,
        capacity_protocol_sha256=capacity_protocol_sha256,
        capacity_result=capacity_result,
        capacity_result_sha256=capacity_result_sha256,
        trace_manifest=trace_manifest,
        trace_manifest_sha256=trace_manifest_sha256,
    )
    source_inventory = _source_inventory()
    post = _post_freeze_authentication(context)
    post["mass_selector_source_inventory"] = source_inventory == _source_inventory()
    if not all(post.values()):
        raise ValueError("mass-selector post-freeze authentication failed")

    capacity = context["capacity_protocol"]
    capacity_result_value = context["capacity_result"]
    protocol_binding = _binding(
        context["capacity_protocol_path"],
        context["capacity_protocol_sha256"],
    )
    result_binding = _binding(
        context["capacity_result_path"],
        context["capacity_result_sha256"],
    )
    manifest_binding = _binding(
        context["trace_manifest_path"],
        context["trace_manifest_sha256"],
    )
    folds = _fold_contract()
    protocol = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _PROTOCOL_EXPERIMENT,
        "status": _PROTOCOL_STATUS,
        "capacity_protocol": protocol_binding,
        "capacity_result": {
            **result_binding,
            "authenticated_constructible_C28_gate_passed": True,
        },
        "trace_manifest": {
            **manifest_binding,
            "record_count": _RECORDS,
        },
        "historical_bindings": dict(capacity["historical_bindings"]),
        "schedule_contract": dict(capacity["schedule_contract"]),
        "output_projection": dict(capacity["output_projection"]),
        "selector_model": {
            "components_per_query_head": _COMPONENTS,
            "query_heads": _QUERY_HEADS,
            "layers": _LAYERS,
            "rank": _RANK,
            "parameter_shapes": {
                "U": [_LAYERS, _COMPONENTS, _RANK],
                "V": [_LAYERS, _RANK, _COMPONENTS],
                "E": [_LAYERS, _QUERY_HEADS, _RANK],
                "B": [_LAYERS, _QUERY_HEADS, _COMPONENTS],
            },
            "input_features": (
                "training and cached evaluation use clipped centered log of the "
                "authenticated native mass over the 28 constructible components"
            ),
            "candidate_native_input_features": (
                "clipped centered raw pre-softmax scores with the same valid mask"
            ),
            "raw_native_score_capture_available": False,
            "raw_native_score_equivalence_claimed_by_this_protocol": False,
            "raw_native_score_feature_and_output_parity_required_before_rollout": True,
            "cached_training_feature_reconstruction": (
                "centered clipped log-mass is the authenticated causal feature; "
                "log-mass-to-reconstructed-score parity is an implementation "
                "reference, not evidence about the unavailable original q.k scores"
            ),
            "forward": (
                "delta=clamp(gauge((relu(features@U+E))@V+B),-16,+16); "
                "cached coefficients=normalize(native_mass*exp(delta)); candidate "
                "native coefficients=masked_softmax(raw_scores+delta)"
            ),
            "execution_arithmetic": (
                "BF16 parameters decoded to FP32 with FP32 accumulation, "
                "gauge, clamp, exponential, and normalization"
            ),
            "hidden_activation": "ReLU",
            "feature_clip": [-_DELTA_CLAMP, _DELTA_CLAMP],
            "gauge": "subtract mean delta over valid components per query head",
            "delta_clamp": [-_DELTA_CLAMP, _DELTA_CLAMP],
            "invalid_components": (
                "remain exactly masked with zero coefficient before and after delta"
            ),
            "loss": (
                "direct squared post-Wo error against authenticated target residual"
            ),
            "loss_aggregation": (
                "same sequence, block-entry-position, layer, and global recovery "
                "aggregation as the full-visible C28 gate"
            ),
            "counterfactual_updates_hidden_or_cache": False,
            "production_integration": (
                "implement the fused raw-score path, capture authenticated raw "
                "scores, and pass feature/coefficient/post-Wo parity against the "
                "log-mass reference before any causal rollout"
            ),
        },
        "training": {
            "cross_validation": {
                "folds": folds,
                "out_of_fold_records": list(range(_RECORDS)),
                "one_final_checkpoint_per_fold": True,
                "heldout_checkpointing_or_early_stopping": False,
            },
            "steps": _TRAINING_STEPS,
            "warmup_steps": _WARMUP_STEPS,
            "optimizer": "AdamW",
            "learning_rate_schedule": "linear_warmup_then_cosine_decay",
            "peak_learning_rate": _PEAK_LEARNING_RATE,
            "final_learning_rate": _FINAL_LEARNING_RATE,
            "betas": [_BETA1, _BETA2],
            "epsilon": _EPSILON,
            "weight_decay": {
                "U": _UV_WEIGHT_DECAY,
                "V": _UV_WEIGHT_DECAY,
                "E": 0.0,
                "B": 0.0,
            },
            "global_gradient_clip_norm": _GRADIENT_CLIP_NORM,
            "deterministic_rows_per_layer_per_step": (_ROWS_PER_LAYER_PER_STEP),
            "training_rows_per_layer_per_fold": (_TRAIN_ROWS_PER_LAYER_PER_FOLD),
            "steps_per_epoch": _STEPS_PER_EPOCH,
            "epochs": _EPOCHS,
            "initialization": {
                "U": {
                    "distribution": "normal",
                    "mean": 0.0,
                    "standard_deviation": _INITIAL_U_STANDARD_DEVIATION,
                },
                "V": "zero",
                "E": "zero",
                "B": "zero",
            },
            "initialization_seed_base": _INIT_SEED_BASE,
            "shuffle_seed_base": _SHUFFLE_SEED_BASE,
            "seed_derivation": "fold_seed=base+fold_index",
            "shuffle_generator": "numpy.random.Generator(numpy.random.PCG64)",
            "shuffle_source_order": (
                "training record index ascending, then authenticated read "
                "position ascending; one independent 192-row permutation per "
                "layer and epoch, consumed two rows at a time"
            ),
            "shuffle_implementation_source": _CORE_SOURCE,
            "training_precision": "FP32",
            "training_device": "cuda",
            "CUDA_deterministic_algorithms_required": True,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "checkpoint_selection": "fixed final step only",
            "final_all_train_fit_on_out_of_fold_pass": {
                "training_record_indices": list(range(_RECORDS)),
                "steps": _FINAL_TRAINING_STEPS,
                "warmup_steps": _FINAL_WARMUP_STEPS,
                "steps_per_epoch": 128,
                "epochs": _EPOCHS,
                "initialization_seed": _INIT_SEED_BASE + len(_FOLDS),
                "shuffle_seed": _SHUFFLE_SEED_BASE + len(_FOLDS),
                "not_used_for_out_of_fold_gate": True,
                "purpose": "single BF16 artifact for train-only native integration",
            },
        },
        "serialization_and_parity": {
            "production_parameter_dtype": "BF16",
            "BF16_rounding": "round_to_nearest_even",
            "FP32_and_BF16_out_of_fold_evaluation_required": True,
            "mass_vs_reconstructed_score_coefficient_max_abs_tolerance": 1.0e-6,
            "mass_vs_reconstructed_score_delta_max_abs_tolerance": 1.0e-6,
            "native_raw_score_parity_deferred_to_native_integration": True,
            "zero_model_native_coefficient_max_abs_tolerance": 1.0e-6,
            "zero_model_native_post_Wo_output_max_abs_tolerance": 1.0e-6,
            "deterministic_serialized_replay_exact": True,
            "final_native_artifact_contains_BF16_only": True,
            "FP32_training_audit_artifact_not_runtime_loadable": True,
        },
        "progression_gate": {
            "arms_required": ["FP32", "BF16_RNE"],
            "minimum_global_recovery": 0.50,
            "minimum_every_out_of_fold_sequence_recovery": 0.25,
            "minimum_every_block_entry_position_recovery": 0.25,
            "minimum_positive_recovery_layers": 12,
            "maximum_BF16_global_recovery_drop_from_FP32": 0.005,
            "finite_coefficients_outputs_and_metrics": True,
            "simplex_and_invalid_mask_invariants": True,
            "mass_vs_reconstructed_score_reference_parity_required": True,
            "native_raw_score_parity_not_claimed": True,
            "native_raw_score_feature_and_output_parity_required_before_rollout": True,
            "zero_model_native_coefficient_and_output_parity_required": True,
            "deterministic_serialized_replay_exact": True,
            "resource_contract_pass_required": True,
            "final_all_train_artifact_sanity_required_for_native_integration": True,
        },
        "resource_contract": _resource_contract(),
        "scope": {
            "split": "train",
            "records": _RECORDS,
            "read_positions_per_record": _READ_POSITIONS_PER_RECORD,
            "out_of_fold_selector_fit": True,
            "causal_features_only": True,
            "cached_trace_training_and_evaluation": True,
            "native_or_packaged_causal_rollout": False,
            "development_outcomes_used": False,
            "confirmation_file_access_permitted": False,
            "semantic_or_M3_pass": False,
        },
        "progression_authority": {
            "on_pass": (
                "authorize implementing the fused CPU selector and authenticating "
                "raw-score feature/coefficient/post-Wo parity; do not roll out yet"
            ),
            "on_fail": (
                "do not tune on heldout fold outcomes; revise the causal "
                "feature/model class under a newly frozen train-only protocol"
            ),
            "does_not_authorize_confirmation": True,
            "does_not_pass_semantic_or_M3_gate": True,
            "does_not_authorize_native_runtime_or_development_rollout": True,
        },
        "authenticated_confirmation_descriptor": dict(
            capacity_result_value["authenticated_confirmation_descriptor"]
        ),
        "source_sha256": source_inventory,
        "post_freeze_authentication": post,
        "confirmation_split_opened": False,
    }
    expected_behavior = {
        "selector_model": _selector_model_contract(),
        "training": _training_contract(),
        "serialization_and_parity": _serialization_contract(),
        "progression_gate": _progression_gate_contract(),
        "resource_contract": _resource_contract(),
        "scope": _scope_contract(),
        "progression_authority": _progression_authority_contract(),
    }
    if any(protocol[name] != value for name, value in expected_behavior.items()):
        raise AssertionError("mass-selector canonical protocol construction changed")
    output = _new_output(out, "mass-selector protocol output")
    _atomic_json_new(output, protocol)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "protocol": protocol,
    }


def _checked_bound_file(
    value: str | Path,
    digest: str,
    label: str,
) -> Path:
    if not _is_sha256(digest):
        raise ValueError(f"{label} SHA-256 is invalid")
    requested = Path(value).expanduser()
    source = requested.parent.resolve(strict=False) / requested.name
    if requested.is_symlink() or not source.is_file():
        raise ValueError(f"{label} is invalid or is a symlink")
    if sha256_file(source) != digest:
        raise ValueError(f"{label} authentication failed")
    return source


def _new_directory(value: str | Path, label: str) -> Path:
    _guard_paths(((label, value),))
    requested = Path(value).expanduser()
    directory = requested.parent.resolve(strict=False) / requested.name
    if requested.is_symlink() or directory.exists():
        raise ValueError(f"{label} already exists or is a symlink")
    directory.mkdir(parents=True)
    return directory


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_safetensors(path: Path, tensors: Mapping[str, np.ndarray]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("mass-selector safetensors output already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_safetensors(
            {name: np.ascontiguousarray(value) for name, value in tensors.items()},
            str(temporary),
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        _publish_new_file(temporary, path, "mass-selector safetensors output")
    finally:
        temporary.unlink(missing_ok=True)


def _write_model_artifact(
    path: Path,
    parameters: selector.SelectorParameters,
    shape: selector.SelectorShape,
) -> dict[str, Any]:
    parameters.validate(shape)
    decoded, bits = selector.quantize_parameters_bf16(parameters, shape)
    tensors: dict[str, np.ndarray] = {}
    for name, value in parameters.as_dict().items():
        tensors[f"{name}_fp32"] = value
        tensors[f"{name}_bf16_bits"] = bits[name]
    _atomic_safetensors(path, tensors)
    loaded = load_safetensors(str(path))
    if set(loaded) != set(tensors):
        raise ValueError("mass-selector model artifact keys changed")
    for name, expected in tensors.items():
        if not np.array_equal(loaded[name], expected):
            raise ValueError("mass-selector model artifact replay changed")
    replay = selector.SelectorParameters(
        U=selector.bf16_bits_to_float32(loaded["U_bf16_bits"]),
        V=selector.bf16_bits_to_float32(loaded["V_bf16_bits"]),
        E=selector.bf16_bits_to_float32(loaded["E_bf16_bits"]),
        B=selector.bf16_bits_to_float32(loaded["B_bf16_bits"]),
    )
    replay.validate(shape)
    if any(
        not np.array_equal(replay.as_dict()[name], decoded.as_dict()[name])
        for name in ("U", "V", "E", "B")
    ):
        raise ValueError("mass-selector BF16 decode replay changed")
    return {
        "file": path.name,
        "file_sha256": sha256_file(path),
        "fp32_parameter_sha256": selector.parameters_sha256(parameters, shape),
        "bf16_decoded_parameter_sha256": selector.parameters_sha256(decoded, shape),
        "tensor_sha256": {
            name: _array_sha256(value) for name, value in tensors.items()
        },
        "parameter_count": shape.parameter_count,
        "embedded_BF16_tensor_bytes": shape.parameter_count * 2,
        "embedded_FP32_training_tensor_bytes": shape.parameter_count * 4,
        "artifact_role": "training_audit",
        "authorized_for_runtime_loading": False,
        "serialized_contains_FP32_training_audit_copy": True,
        "serialized_BF16_rounding": "round_to_nearest_even",
        "deterministic_replay_exact": True,
    }


def _write_bf16_deployment_artifact(
    path: Path,
    parameters: selector.SelectorParameters,
    shape: selector.SelectorShape,
) -> dict[str, Any]:
    """Write the exact BF16-only file authorized for the native runtime."""

    parameters.validate(shape)
    decoded, bits = selector.quantize_parameters_bf16(parameters, shape)
    tensors = {f"{name}_bf16_bits": value for name, value in bits.items()}
    if sum(value.nbytes for value in tensors.values()) != shape.parameter_count * 2:
        raise AssertionError("mass-selector deployment tensor bytes changed")
    _atomic_safetensors(path, tensors)
    loaded = load_safetensors(str(path))
    if set(loaded) != set(tensors):
        raise ValueError("mass-selector deployment artifact keys changed")
    for name, expected in tensors.items():
        if not np.array_equal(loaded[name], expected):
            raise ValueError("mass-selector deployment artifact replay changed")
    replay = selector.SelectorParameters(
        U=selector.bf16_bits_to_float32(loaded["U_bf16_bits"]),
        V=selector.bf16_bits_to_float32(loaded["V_bf16_bits"]),
        E=selector.bf16_bits_to_float32(loaded["E_bf16_bits"]),
        B=selector.bf16_bits_to_float32(loaded["B_bf16_bits"]),
    )
    replay.validate(shape)
    if selector.parameters_sha256(replay, shape) != selector.parameters_sha256(
        decoded,
        shape,
    ):
        raise ValueError("mass-selector deployment decode replay changed")
    return {
        "file": path.name,
        "file_sha256": sha256_file(path),
        "artifact_role": "native_BF16_deployment",
        "authorized_for_runtime_loading": True,
        "contains_FP32_training_copy": False,
        "parameter_count": shape.parameter_count,
        "BF16_tensor_bytes": shape.parameter_count * 2,
        "bf16_decoded_parameter_sha256": selector.parameters_sha256(
            decoded,
            shape,
        ),
        "tensor_sha256": {
            name: _array_sha256(value) for name, value in tensors.items()
        },
        "serialized_BF16_rounding": "round_to_nearest_even",
        "deterministic_replay_exact": True,
    }


def _load_bf16_deployment_artifact(
    path: Path,
    descriptor: Mapping[str, Any],
    shape: selector.SelectorShape,
) -> selector.SelectorParameters:
    if (
        descriptor.get("artifact_role") != "native_BF16_deployment"
        or descriptor.get("authorized_for_runtime_loading") is not True
        or descriptor.get("contains_FP32_training_copy") is not False
        or descriptor.get("BF16_tensor_bytes") != shape.parameter_count * 2
        or path.name != descriptor.get("file")
        or sha256_file(path) != descriptor.get("file_sha256")
    ):
        raise ValueError("mass-selector deployment artifact authentication failed")
    tensors = load_safetensors(str(path))
    expected_keys = {f"{name}_bf16_bits" for name in ("U", "V", "E", "B")}
    if set(tensors) != expected_keys:
        raise ValueError("mass-selector deployment artifact layout changed")
    parameters = selector.SelectorParameters(
        U=selector.bf16_bits_to_float32(tensors["U_bf16_bits"]),
        V=selector.bf16_bits_to_float32(tensors["V_bf16_bits"]),
        E=selector.bf16_bits_to_float32(tensors["E_bf16_bits"]),
        B=selector.bf16_bits_to_float32(tensors["B_bf16_bits"]),
    )
    parameters.validate(shape)
    if (
        selector.parameters_sha256(parameters, shape)
        != descriptor["bf16_decoded_parameter_sha256"]
    ):
        raise ValueError("mass-selector deployment parameter root changed")
    return parameters


def _load_model_artifact(
    path: Path,
    descriptor: Mapping[str, Any],
    shape: selector.SelectorShape,
    *,
    bf16: bool,
) -> selector.SelectorParameters:
    if path.name != descriptor.get("file") or sha256_file(path) != descriptor.get(
        "file_sha256"
    ):
        raise ValueError("mass-selector model artifact authentication failed")
    tensors = load_safetensors(str(path))
    suffix = "bf16_bits" if bf16 else "fp32"
    values: dict[str, np.ndarray] = {}
    for name in ("U", "V", "E", "B"):
        tensor = tensors[f"{name}_{suffix}"]
        values[name] = (
            selector.bf16_bits_to_float32(tensor)
            if bf16
            else np.ascontiguousarray(tensor, dtype=np.float32)
        )
    parameters = selector.SelectorParameters(**values)
    parameters.validate(shape)
    expected = (
        descriptor["bf16_decoded_parameter_sha256"]
        if bf16
        else descriptor["fp32_parameter_sha256"]
    )
    if selector.parameters_sha256(parameters, shape) != expected:
        raise ValueError("mass-selector model parameter root changed")
    return parameters


def _validate_selector_protocol(
    protocol: Mapping[str, Any],
) -> None:
    capacity = protocol.get("capacity_protocol")
    capacity_result = protocol.get("capacity_result")
    trace = protocol.get("trace_manifest")
    model = protocol.get("selector_model")
    training = protocol.get("training")
    serialization = protocol.get("serialization_and_parity")
    gate = protocol.get("progression_gate")
    resource = protocol.get("resource_contract")
    scope = protocol.get("scope")
    authority = protocol.get("progression_authority")
    post = protocol.get("post_freeze_authentication")
    final_fit = (
        training.get("final_all_train_fit_on_out_of_fold_pass")
        if isinstance(
            training,
            Mapping,
        )
        else None
    )
    if (
        protocol.get("schema_version") != _SCHEMA_VERSION
        or protocol.get("experiment") != _PROTOCOL_EXPERIMENT
        or protocol.get("status") != _PROTOCOL_STATUS
        or protocol.get("confirmation_split_opened") is not False
        or protocol.get("source_sha256") != _source_inventory()
        or not isinstance(capacity, Mapping)
        or capacity.get("sha256") != _EXPECTED_CAPACITY_PROTOCOL_SHA256
        or not isinstance(capacity_result, Mapping)
        or capacity_result.get("sha256") != _EXPECTED_CAPACITY_RESULT_SHA256
        or capacity_result.get("authenticated_constructible_C28_gate_passed")
        is not True
        or not isinstance(trace, Mapping)
        or trace.get("sha256") != _EXPECTED_TRACE_MANIFEST_SHA256
        or trace.get("record_count") != _RECORDS
        or not isinstance(model, Mapping)
        or model != _selector_model_contract()
        or model.get("components_per_query_head") != _COMPONENTS
        or model.get("rank") != _RANK
        or model.get("hidden_activation") != "ReLU"
        or model.get("feature_clip") != [-_DELTA_CLAMP, _DELTA_CLAMP]
        or model.get("delta_clamp") != [-_DELTA_CLAMP, _DELTA_CLAMP]
        or not isinstance(training, Mapping)
        or training != _training_contract()
        or training.get("steps") != _TRAINING_STEPS
        or training.get("warmup_steps") != _WARMUP_STEPS
        or training.get("epochs") != _EPOCHS
        or training.get("steps_per_epoch") != _STEPS_PER_EPOCH
        or training.get("training_device") != "cuda"
        or training.get("CUDA_deterministic_algorithms_required") is not True
        or training.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or training.get("cross_validation", {}).get("folds") != _fold_contract()
        or not isinstance(final_fit, Mapping)
        or final_fit.get("training_record_indices") != list(range(_RECORDS))
        or final_fit.get("steps") != _FINAL_TRAINING_STEPS
        or final_fit.get("warmup_steps") != _FINAL_WARMUP_STEPS
        or final_fit.get("steps_per_epoch") != 128
        or final_fit.get("epochs") != _EPOCHS
        or final_fit.get("initialization_seed") != _INIT_SEED_BASE + len(_FOLDS)
        or final_fit.get("shuffle_seed") != _SHUFFLE_SEED_BASE + len(_FOLDS)
        or final_fit.get("not_used_for_out_of_fold_gate") is not True
        or not isinstance(serialization, Mapping)
        or serialization != _serialization_contract()
        or serialization.get("production_parameter_dtype") != "BF16"
        or not isinstance(gate, Mapping)
        or gate != _progression_gate_contract()
        or gate.get("minimum_global_recovery") != 0.50
        or gate.get("maximum_BF16_global_recovery_drop_from_FP32") != 0.005
        or resource != _resource_contract()
        or not isinstance(scope, Mapping)
        or scope != _scope_contract()
        or scope.get("split") != "train"
        or scope.get("development_outcomes_used") is not False
        or scope.get("confirmation_file_access_permitted") is not False
        or scope.get("semantic_or_M3_pass") is not False
        or authority != _progression_authority_contract()
        or not isinstance(post, Mapping)
        or not post
        or not all(value is True for value in post.values())
    ):
        raise ValueError("mass-selector frozen protocol changed")


def _authenticate_selector_protocol(
    protocol: str | Path,
    protocol_sha256: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    protocol_path = _checked_bound_file(
        protocol,
        protocol_sha256,
        "mass-selector protocol",
    )
    frozen = _read_json(protocol_path, "mass-selector protocol")
    _validate_selector_protocol(frozen)
    context = _authenticate_capacity_inputs(
        capacity_protocol=frozen["capacity_protocol"]["path"],
        capacity_protocol_sha256=frozen["capacity_protocol"]["sha256"],
        capacity_result=frozen["capacity_result"]["path"],
        capacity_result_sha256=frozen["capacity_result"]["sha256"],
        trace_manifest=frozen["trace_manifest"]["path"],
        trace_manifest_sha256=frozen["trace_manifest"]["sha256"],
    )
    if (
        frozen["historical_bindings"]
        != context["capacity_protocol"]["historical_bindings"]
        or frozen["schedule_contract"]
        != context["capacity_protocol"]["schedule_contract"]
        or frozen["output_projection"]
        != context["capacity_protocol"]["output_projection"]
        or frozen["authenticated_confirmation_descriptor"]
        != context["capacity_result"]["authenticated_confirmation_descriptor"]
    ):
        raise ValueError("mass-selector inherited bindings changed")
    return protocol_path, frozen, context


def _training_config(
    *,
    init_seed: int,
    shuffle_seed: int,
    final_fit: bool = False,
) -> selector.TrainingConfig:
    return selector.TrainingConfig(
        steps=_FINAL_TRAINING_STEPS if final_fit else _TRAINING_STEPS,
        warmup_steps=_FINAL_WARMUP_STEPS if final_fit else _WARMUP_STEPS,
        peak_learning_rate=_PEAK_LEARNING_RATE,
        final_learning_rate=_FINAL_LEARNING_RATE,
        beta1=_BETA1,
        beta2=_BETA2,
        epsilon=_EPSILON,
        uv_weight_decay=_UV_WEIGHT_DECAY,
        gradient_clip_norm=_GRADIENT_CLIP_NORM,
        rows_per_layer_per_step=_ROWS_PER_LAYER_PER_STEP,
        epochs=_EPOCHS,
        feature_clip=_DELTA_CLAMP,
        delta_clamp=_DELTA_CLAMP,
        initial_u_standard_deviation=_INITIAL_U_STANDARD_DEVIATION,
        init_seed=init_seed,
        shuffle_seed=shuffle_seed,
    )


def _arm_metrics(
    target_energy: np.ndarray,
    error_energy: np.ndarray,
) -> dict[str, Any]:
    return full.joint._recovery_metrics_from_energy(
        np.ascontiguousarray(target_energy.reshape(-1), dtype=np.float64),
        np.ascontiguousarray(error_energy.reshape(-1), dtype=np.float64),
        batch_shape=(_RECORDS, _READ_POSITIONS_PER_RECORD, _LAYERS),
    )


def _maximum_output_correction(
    coefficients: np.ndarray,
    values: np.ndarray,
    base_heads: np.ndarray,
    projection: np.ndarray,
    shape: selector.SelectorShape,
) -> float:
    maximum = 0.0
    for layer in range(shape.layers):
        candidate = np.einsum(
            "nrhc,nrhcd->nrhd",
            coefficients[:, :, layer],
            values[:, :, layer],
            optimize=True,
        )
        delta = (candidate - base_heads[:, :, layer]).reshape(
            -1,
            shape.hidden_size,
        )
        correction = delta.astype(np.float64) @ projection[layer].astype(np.float64).T
        maximum = max(maximum, float(np.max(np.abs(correction))))
    return maximum


def _selector_gate(
    fp32_metrics: Mapping[str, Any],
    bf16_metrics: Mapping[str, Any],
    *,
    fp32_coefficients: np.ndarray,
    bf16_coefficients: np.ndarray,
    valid: np.ndarray,
    mass_score_max_abs: float,
    mass_score_delta_max_abs: float,
    zero_coefficient_max_abs: float,
    zero_output_max_abs: float,
    deterministic_replay_exact: bool,
) -> dict[str, bool]:
    finite = all(
        np.isfinite(value).all() for value in (fp32_coefficients, bf16_coefficients)
    )
    simplex = all(
        (
            np.max(np.abs(np.sum(value, axis=-1, dtype=np.float32) - np.float32(1.0)))
            <= 2.0e-6
            and np.all(value >= 0.0)
            and np.all(value[~valid] == 0.0)
        )
        for value in (fp32_coefficients, bf16_coefficients)
    )
    fp32_global = float(fp32_metrics["global"]["recovery"])
    bf16_global = float(bf16_metrics["global"]["recovery"])
    checks = {
        "FP32_gate_passed": fp32_metrics.get("passed") is True,
        "BF16_gate_passed": bf16_metrics.get("passed") is True,
        "BF16_global_recovery_drop_within_0_005": (fp32_global - bf16_global <= 0.005),
        "finite": finite,
        "simplex_and_invalid_mask": simplex,
        "mass_vs_reconstructed_score_coefficient_parity": (
            mass_score_max_abs <= 1.0e-6
        ),
        "mass_vs_reconstructed_score_delta_parity": (
            mass_score_delta_max_abs <= 1.0e-6
        ),
        "native_raw_score_parity_not_evaluated": True,
        "zero_model_native_coefficient_parity": (zero_coefficient_max_abs <= 1.0e-6),
        "zero_model_native_output_parity": zero_output_max_abs <= 1.0e-6,
        "deterministic_serialized_replay_exact": deterministic_replay_exact,
        "resource_contract": (
            _TOTAL_LOGICAL_TRAFFIC_BYTES <= _MAXIMUM_DEPLOYABLE_BYTES_AT_45_PERCENT
        ),
    }
    checks["passed"] = all(checks.values())
    return checks


def fit_screen_mass_selector(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    artifact_directory: str | Path,
    out: str | Path,
    device: str,
) -> dict[str, Any]:
    """Fit fixed OOF selectors and evaluate serialized BF16 artifacts."""

    _guard_paths(
        (
            ("selector protocol", protocol),
            ("selector artifact directory", artifact_directory),
            ("selector result output", out),
        )
    )
    protocol_path, frozen, context = _authenticate_selector_protocol(
        protocol,
        protocol_sha256,
    )
    if device != frozen["training"]["training_device"]:
        raise ValueError("mass-selector training device changed from frozen protocol")
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != frozen["training"]["CUBLAS_WORKSPACE_CONFIG"]
    ):
        raise ValueError("mass-selector CUDA workspace configuration changed")
    output = _new_output(out, "mass-selector result output")
    artifacts = _new_directory(
        artifact_directory,
        "mass-selector artifact directory",
    )
    capacity_binding = _binding(
        context["capacity_protocol_path"],
        context["capacity_protocol_sha256"],
    )
    arrays, manifest = full.load_stacked_full_visible_trace(
        context["trace_manifest_path"],
        context["trace_manifest_sha256"],
        protocol=capacity_binding,
    )
    record_authentication = full._audit_manifest_record_bindings(
        context["capacity_protocol"],
        manifest,
    )
    projection = full._load_authenticated_output_projection(
        context["capacity_protocol"]
    )
    basis = full.build_full_visible_basis(
        arrays,
        query_heads=_QUERY_HEADS,
        regular_entries=full._REGULAR_ENTRIES,
        episodic_entries=full._EPISODIC_ENTRIES,
        include_exact_native_anchor=False,
    )
    shape = selector.SelectorShape()
    mass = np.ascontiguousarray(basis.visible_mass, dtype=np.float32)
    valid = np.ascontiguousarray(basis.visible_valid, dtype=bool)
    values = np.ascontiguousarray(basis.visible_values, dtype=np.float32)
    base_heads = np.ascontiguousarray(basis.base_heads, dtype=np.float32)
    target = np.ascontiguousarray(basis.target_residual, dtype=np.float32)
    expected_mass_shape = (
        _RECORDS,
        _READ_POSITIONS_PER_RECORD,
        _LAYERS,
        _QUERY_HEADS,
        _COMPONENTS,
    )
    if (
        mass.shape != expected_mass_shape
        or valid.shape != expected_mass_shape
        or values.shape != expected_mass_shape + (shape.head_dimension,)
        or base_heads.shape
        != (
            _RECORDS,
            _READ_POSITIONS_PER_RECORD,
            _LAYERS,
            _QUERY_HEADS,
            shape.head_dimension,
        )
        or target.shape
        != (
            _RECORDS,
            _READ_POSITIONS_PER_RECORD,
            _LAYERS,
            shape.hidden_size,
        )
    ):
        raise ValueError("mass-selector authenticated trace shape changed")

    fp32_oof = np.empty(expected_mass_shape, dtype=np.float32)
    bf16_oof = np.empty(expected_mass_shape, dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    deterministic_replay = True
    mass_score_max_abs = 0.0
    mass_score_delta_max_abs = 0.0
    for fold in frozen["training"]["cross_validation"]["folds"]:
        fold_index = int(fold["fold_index"])
        training_records = tuple(
            int(value) for value in fold["training_record_indices"]
        )
        heldout = np.asarray(fold["heldout_record_indices"], dtype=np.int64)
        config = _training_config(
            init_seed=int(fold["initialization_seed"]),
            shuffle_seed=int(fold["shuffle_seed"]),
        )
        trained = selector.fit_direct_post_wo(
            mass,
            valid,
            values,
            base_heads,
            target,
            projection,
            training_records=training_records,
            shape=shape,
            config=config,
            device=device,
        )
        artifact_path = artifacts / f"fold-{fold_index}.safetensors"
        descriptor = _write_model_artifact(
            artifact_path,
            trained.parameters,
            shape,
        )
        fp32_parameters = _load_model_artifact(
            artifact_path,
            descriptor,
            shape,
            bf16=False,
        )
        bf16_parameters = _load_model_artifact(
            artifact_path,
            descriptor,
            shape,
            bf16=True,
        )
        fp32_coefficients, fp32_delta = selector.selector_forward(
            mass[heldout],
            valid[heldout],
            fp32_parameters,
            shape,
        )
        bf16_coefficients, bf16_delta = selector.selector_forward(
            mass[heldout],
            valid[heldout],
            bf16_parameters,
            shape,
        )
        fp32_replay, _ = selector.selector_forward(
            mass[heldout],
            valid[heldout],
            fp32_parameters,
            shape,
        )
        bf16_replay, _ = selector.selector_forward(
            mass[heldout],
            valid[heldout],
            bf16_parameters,
            shape,
        )
        deterministic_replay = deterministic_replay and np.array_equal(
            fp32_coefficients,
            fp32_replay,
        )
        deterministic_replay = deterministic_replay and np.array_equal(
            bf16_coefficients,
            bf16_replay,
        )
        fp32_oof[heldout] = fp32_coefficients
        bf16_oof[heldout] = bf16_coefficients
        for parameters, delta, coefficients in (
            (fp32_parameters, fp32_delta, fp32_coefficients),
            (bf16_parameters, bf16_delta, bf16_coefficients),
        ):
            score = np.zeros(mass[heldout].shape, dtype=np.float32)
            score[valid[heldout]] = np.log(mass[heldout][valid[heldout]])
            score_route, score_delta = selector.selector_forward_from_scores(
                score,
                valid[heldout],
                parameters,
                shape,
            )
            mass_score_max_abs = max(
                mass_score_max_abs,
                float(np.max(np.abs(coefficients - score_route))),
            )
            mass_score_delta_max_abs = max(
                mass_score_delta_max_abs,
                float(np.max(np.abs(delta - score_delta))),
            )
        fold_rows.append(
            {
                "fold_index": fold_index,
                "training_record_indices": list(training_records),
                "heldout_record_indices": heldout.tolist(),
                "initial_loss": trained.initial_loss,
                "final_loss": trained.final_loss,
                "steps": trained.steps,
                "device": trained.device,
                "learning_rate_sha256": trained.learning_rate_sha256,
                "schedule_sha256": trained.schedule_sha256,
                "artifact": descriptor,
            }
        )

    zero = selector.initialize_parameters(
        shape,
        seed=_INIT_SEED_BASE,
        standard_deviation=_INITIAL_U_STANDARD_DEVIATION,
    )
    zero_coefficients, _ = selector.selector_forward(mass, valid, zero, shape)
    zero_coefficient_max_abs = float(np.max(np.abs(zero_coefficients - mass)))
    zero_output_max_abs = _maximum_output_correction(
        zero_coefficients,
        values,
        base_heads,
        projection,
        shape,
    )
    fp32_error, target_energy = selector.direct_post_wo_error_energy(
        fp32_oof,
        values,
        base_heads,
        target,
        projection,
        shape,
    )
    bf16_error, target_energy_replay = selector.direct_post_wo_error_energy(
        bf16_oof,
        values,
        base_heads,
        target,
        projection,
        shape,
    )
    if not np.array_equal(target_energy, target_energy_replay):
        raise ValueError("mass-selector target energy replay changed")
    fp32_metrics = _arm_metrics(target_energy, fp32_error)
    bf16_metrics = _arm_metrics(target_energy, bf16_error)
    gate = _selector_gate(
        fp32_metrics,
        bf16_metrics,
        fp32_coefficients=fp32_oof,
        bf16_coefficients=bf16_oof,
        valid=valid,
        mass_score_max_abs=mass_score_max_abs,
        mass_score_delta_max_abs=mass_score_delta_max_abs,
        zero_coefficient_max_abs=zero_coefficient_max_abs,
        zero_output_max_abs=zero_output_max_abs,
        deterministic_replay_exact=deterministic_replay,
    )
    oof_path = artifacts / "oof-coefficients.safetensors"
    oof_tensors = {
        "fp32_coefficients": np.ascontiguousarray(fp32_oof),
        "bf16_coefficients": np.ascontiguousarray(bf16_oof),
        "fp32_error_energy": np.ascontiguousarray(fp32_error, dtype=np.float64),
        "bf16_error_energy": np.ascontiguousarray(bf16_error, dtype=np.float64),
        "target_energy": np.ascontiguousarray(target_energy, dtype=np.float64),
    }
    _atomic_safetensors(oof_path, oof_tensors)
    oof_replay = load_safetensors(str(oof_path))
    if set(oof_replay) != set(oof_tensors) or any(
        not np.array_equal(oof_replay[name], expected)
        for name, expected in oof_tensors.items()
    ):
        raise ValueError("mass-selector OOF artifact replay changed")
    final_artifact: dict[str, Any] | None = None
    final_artifact_sanity_passed = False
    if gate["passed"]:
        final_contract = frozen["training"]["final_all_train_fit_on_out_of_fold_pass"]
        final_trained = selector.fit_direct_post_wo(
            mass,
            valid,
            values,
            base_heads,
            target,
            projection,
            training_records=tuple(range(_RECORDS)),
            shape=shape,
            config=_training_config(
                init_seed=int(final_contract["initialization_seed"]),
                shuffle_seed=int(final_contract["shuffle_seed"]),
                final_fit=True,
            ),
            device=device,
        )
        final_audit_path = artifacts / "all-train-selector-audit.safetensors"
        final_audit_descriptor = _write_model_artifact(
            final_audit_path,
            final_trained.parameters,
            shape,
        )
        final_fp32_parameters = _load_model_artifact(
            final_audit_path,
            final_audit_descriptor,
            shape,
            bf16=False,
        )
        final_deployment_path = artifacts / "all-train-selector-bf16.safetensors"
        final_deployment_descriptor = _write_bf16_deployment_artifact(
            final_deployment_path,
            final_trained.parameters,
            shape,
        )
        final_bf16_parameters = _load_bf16_deployment_artifact(
            final_deployment_path,
            final_deployment_descriptor,
            shape,
        )
        final_fp32_coefficients, _final_fp32_delta = selector.selector_forward(
            mass,
            valid,
            final_fp32_parameters,
            shape,
        )
        final_bf16_coefficients, final_bf16_delta = selector.selector_forward(
            mass,
            valid,
            final_bf16_parameters,
            shape,
        )
        final_bf16_replay, _ = selector.selector_forward(
            mass,
            valid,
            final_bf16_parameters,
            shape,
        )
        final_score = np.zeros(mass.shape, dtype=np.float32)
        final_score[valid] = np.log(mass[valid])
        final_score_coefficients, final_score_delta = (
            selector.selector_forward_from_scores(
                final_score,
                valid,
                final_bf16_parameters,
                shape,
            )
        )
        final_fp32_error, final_target_energy = selector.direct_post_wo_error_energy(
            final_fp32_coefficients,
            values,
            base_heads,
            target,
            projection,
            shape,
        )
        final_bf16_error, final_target_replay = selector.direct_post_wo_error_energy(
            final_bf16_coefficients,
            values,
            base_heads,
            target,
            projection,
            shape,
        )
        if not np.array_equal(final_target_energy, final_target_replay):
            raise ValueError("mass-selector final target energy replay changed")
        final_fp32_metrics = _arm_metrics(
            final_target_energy,
            final_fp32_error,
        )
        final_bf16_metrics = _arm_metrics(
            final_target_energy,
            final_bf16_error,
        )
        final_finite = all(
            np.isfinite(value).all()
            for value in (final_fp32_coefficients, final_bf16_coefficients)
        )
        final_simplex = all(
            (
                np.max(
                    np.abs(np.sum(value, axis=-1, dtype=np.float32) - np.float32(1.0))
                )
                <= 2.0e-6
                and np.all(value >= 0.0)
                and np.all(value[~valid] == 0.0)
            )
            for value in (final_fp32_coefficients, final_bf16_coefficients)
        )
        final_checks = {
            "FP32_native_nonregression": (
                float(final_fp32_metrics["global"]["recovery"]) >= 0.0
            ),
            "BF16_native_nonregression": (
                float(final_bf16_metrics["global"]["recovery"]) >= 0.0
            ),
            "BF16_global_recovery_drop_within_0_005": (
                float(final_fp32_metrics["global"]["recovery"])
                - float(final_bf16_metrics["global"]["recovery"])
                <= 0.005
            ),
            "finite": final_finite,
            "simplex_and_invalid_mask": final_simplex,
            "mass_vs_reconstructed_score_coefficient_parity": (
                float(
                    np.max(np.abs(final_bf16_coefficients - final_score_coefficients))
                )
                <= 1.0e-6
            ),
            "mass_vs_reconstructed_score_delta_parity": (
                float(np.max(np.abs(final_bf16_delta - final_score_delta))) <= 1.0e-6
            ),
            "deterministic_deployment_replay_exact": np.array_equal(
                final_bf16_coefficients,
                final_bf16_replay,
            ),
            "BF16_only_deployment_tensor_bytes": (
                final_deployment_descriptor["BF16_tensor_bytes"]
                == _BF16_PARAMETER_BYTES
            ),
        }
        final_checks["passed"] = all(final_checks.values())
        final_artifact_sanity_passed = bool(final_checks["passed"])
        final_artifact = {
            "training_audit_artifact": final_audit_descriptor,
            "BF16_deployment_artifact": final_deployment_descriptor,
            "training": {
                "initial_loss": final_trained.initial_loss,
                "final_loss": final_trained.final_loss,
                "steps": final_trained.steps,
                "device": final_trained.device,
                "learning_rate_sha256": final_trained.learning_rate_sha256,
                "schedule_sha256": final_trained.schedule_sha256,
            },
            "sanity": {
                "FP32_metrics": final_fp32_metrics,
                "BF16_metrics": final_bf16_metrics,
                "FP32_coefficient_sha256": _array_sha256(final_fp32_coefficients),
                "BF16_coefficient_sha256": _array_sha256(final_bf16_coefficients),
                "checks": final_checks,
            },
            "not_used_for_out_of_fold_gate": True,
        }

    native_integration_implementation_authorized = bool(
        gate["passed"] and final_artifact_sanity_passed
    )
    status = (
        _PASSED_STATUS
        if native_integration_implementation_authorized
        else (_FINAL_ARTIFACT_FAILED_STATUS if gate["passed"] else _FAILED_STATUS)
    )
    result = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _RESULT_EXPERIMENT,
        "status": status,
        "protocol": _binding(protocol_path, protocol_sha256),
        "trace_manifest": dict(frozen["trace_manifest"]),
        "record_authentication": record_authentication,
        "folds": fold_rows,
        "arms": {
            "FP32": {
                "metrics": fp32_metrics,
                "coefficient_sha256": _array_sha256(fp32_oof),
                "error_energy_sha256": _array_sha256(fp32_error),
            },
            "BF16_RNE": {
                "metrics": bf16_metrics,
                "coefficient_sha256": _array_sha256(bf16_oof),
                "error_energy_sha256": _array_sha256(bf16_error),
            },
        },
        "parity": {
            "mass_vs_reconstructed_score_coefficient_max_abs": (mass_score_max_abs),
            "mass_vs_reconstructed_score_delta_max_abs": (mass_score_delta_max_abs),
            "native_raw_score_capture_available": False,
            "native_raw_score_parity_evaluated": False,
            "native_raw_score_parity_passed": False,
            "zero_model_native_coefficient_max_abs": zero_coefficient_max_abs,
            "zero_model_native_post_Wo_output_max_abs": zero_output_max_abs,
            "deterministic_serialized_replay_exact": deterministic_replay,
        },
        "gate": gate,
        "artifacts": {
            "directory": str(artifacts),
            "oof_coefficients": {
                "file": oof_path.name,
                "file_sha256": sha256_file(oof_path),
                "tensor_sha256": {
                    name: _array_sha256(value) for name, value in oof_tensors.items()
                },
                "evaluated_tensors_persisted_exactly": True,
            },
            "all_train_selector": final_artifact,
        },
        "resource_contract": dict(frozen["resource_contract"]),
        "decision": {
            "train_mass_selector_oof_gate_passed": gate["passed"],
            "final_BF16_artifact_sanity_passed": (final_artifact_sanity_passed),
            "train_only_native_integration_implementation_authorized": (
                native_integration_implementation_authorized
            ),
            "native_raw_score_parity_evaluated": False,
            "native_raw_score_parity_passed": False,
            "train_only_native_runtime_execution_authorized": False,
            "development_authorized": False,
            "confirmation_authorized": False,
            "semantic_or_M3_gate_passed": False,
            "next_step": (
                "integrate the BF16 selector into the fused native score/value "
                "path, capture raw scores, and pass feature/coefficient/post-Wo "
                "parity before freezing any causal rollout"
                if native_integration_implementation_authorized
                else (
                    "diagnose the all-train BF16 fit or serialization without "
                    "changing the passed out-of-fold gate"
                    if gate["passed"]
                    else "freeze a new train-only selector feature/model class"
                )
            ),
        },
        "post_fit_authentication": {
            "protocol": sha256_file(protocol_path) == protocol_sha256,
            "capacity_protocol": (
                sha256_file(context["capacity_protocol_path"])
                == context["capacity_protocol_sha256"]
            ),
            "capacity_result": (
                sha256_file(context["capacity_result_path"])
                == context["capacity_result_sha256"]
            ),
            "trace_manifest": (
                sha256_file(context["trace_manifest_path"])
                == context["trace_manifest_sha256"]
            ),
            "source_inventory": frozen["source_sha256"] == _source_inventory(),
            "record_authentication": all(record_authentication.values()),
            "confirmation_not_opened": True,
        },
        "authenticated_confirmation_descriptor": dict(
            frozen["authenticated_confirmation_descriptor"]
        ),
        "confirmation_split_opened": False,
    }
    if not all(result["post_fit_authentication"].values()):
        raise ValueError("mass-selector post-fit authentication failed")
    _atomic_json_new(output, result)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "result": result,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the train-only full-visible C28 mass selector",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--capacity-protocol", required=True)
    freeze.add_argument("--capacity-protocol-sha256", required=True)
    freeze.add_argument("--capacity-result", required=True)
    freeze.add_argument("--capacity-result-sha256", required=True)
    freeze.add_argument("--trace-manifest", required=True)
    freeze.add_argument("--trace-manifest-sha256", required=True)
    freeze.add_argument("--out", required=True)
    fit = commands.add_parser("fit-screen")
    fit.add_argument("--protocol", required=True)
    fit.add_argument("--protocol-sha256", required=True)
    fit.add_argument("--artifact-directory", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "freeze":
        freeze_mass_selector_protocol(
            capacity_protocol=args.capacity_protocol,
            capacity_protocol_sha256=args.capacity_protocol_sha256,
            capacity_result=args.capacity_result,
            capacity_result_sha256=args.capacity_result_sha256,
            trace_manifest=args.trace_manifest,
            trace_manifest_sha256=args.trace_manifest_sha256,
            out=args.out,
        )
    elif args.command == "fit-screen":
        fit_screen_mass_selector(
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            artifact_directory=args.artifact_directory,
            out=args.out,
            device=args.device,
        )
    else:  # pragma: no cover - argparse enforces this
        raise AssertionError("unreachable mass-selector command")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
