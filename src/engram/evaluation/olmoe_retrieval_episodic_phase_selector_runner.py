"""Freeze and execute the train-only phase-conditioned C28 selector screen.

The experiment is a fail-closed successor to both negative selector screens.
It reuses the authenticated native-mass feature and adds one schedule-relative
table row while an eight-token episodic read is active.  The table is indexed
only by causal offset inside the active span, never by token identity or
absolute position.

The confirmation descriptor is copied by value from predecessor evidence.
No function in this module resolves, opens, stats, hashes, or globs the
confirmation file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file as load_safetensors

import engram.evaluation.olmoe_retrieval_episodic_content_selector_runner as content
import engram.evaluation.olmoe_retrieval_episodic_full_visible_simplex_oracle as full
import engram.evaluation.olmoe_retrieval_episodic_mass_selector as mass
import engram.evaluation.olmoe_retrieval_episodic_mass_selector_runner as mass_runner
import engram.evaluation.olmoe_retrieval_episodic_phase_selector as selector
from engram.utils import sha256_file


_SCHEMA_VERSION = 1
_PROTOCOL_EXPERIMENT = "olmoe_q7_retrieval_episodic_phase_selector_protocol"
_PROTOCOL_STATUS = "frozen_before_train_phase_selector_fit"
_RESULT_EXPERIMENT = "olmoe_q7_retrieval_episodic_phase_selector_oof_train_screen"
_PASSED_STATUS = "train_phase_selector_oof_gate_passed"
_FAILED_STATUS = "train_phase_selector_oof_gate_failed"
_FINAL_ARTIFACT_FAILED_STATUS = (
    "train_phase_selector_oof_gate_passed_but_final_artifact_failed"
)

_EXPECTED_CONTENT_PROTOCOL_SHA256 = (
    "0a58ba3a59d2f0f816046ca28aac304baf7663ef890a6b298f0cc7277613d051"
)
_EXPECTED_CONTENT_RESULT_SHA256 = (
    "9ea504f83a487584cb9ae2127565674a8e341ca58f6777a03514b0c9a281995c"
)
_EXPECTED_MASS_PROTOCOL_SHA256 = content._EXPECTED_MASS_PROTOCOL_SHA256
_EXPECTED_MASS_RESULT_SHA256 = content._EXPECTED_MASS_RESULT_SHA256
_EXPECTED_RESIDUAL_MANIFEST_SHA256 = content._EXPECTED_RESIDUAL_MANIFEST_SHA256
_EXPECTED_PACKAGE_MANIFEST_SHA256 = content._EXPECTED_PACKAGE_MANIFEST_SHA256
_EXPECTED_NON_MLP_SHA256 = content._EXPECTED_NON_MLP_SHA256

_CORE_SOURCE = "src/engram/evaluation/olmoe_retrieval_episodic_phase_selector.py"
_RUNNER_SOURCE = (
    "src/engram/evaluation/olmoe_retrieval_episodic_phase_selector_runner.py"
)
_SOURCE_FILES = tuple(
    dict.fromkeys((*content._SOURCE_FILES, _CORE_SOURCE, _RUNNER_SOURCE))
)

_RECORDS = 8
_READS = 32
_POSITIONS_PER_RECORD = 128
_LAYERS = 16
_HEADS = 16
_COMPONENTS = 28
_PHASES = 8
_HEAD_DIMENSION = 128
_HIDDEN_SIZE = 2048
_BLOCK_POSITIONS = (96, 104, 112, 120)

_TRAINING_STEPS = 1_536
_FINAL_TRAINING_STEPS = 2_048
_WARMUP_STEPS = 96
_FINAL_WARMUP_STEPS = 128
_EPOCHS = 16
_ROWS_PER_LAYER_PER_STEP = 2
_PEAK_LEARNING_RATE = 5.0e-3
_FINAL_LEARNING_RATE = 5.0e-4
_INIT_SEED_BASE = 2_026_073_001
_SHUFFLE_SEED_BASE = 2_026_073_002

_PHASE_PARAMETER_COUNT = 57_344
_MASS_PARAMETER_COUNT = 25_600
_PARAMETER_COUNT = 82_944
_BF16_PARAMETER_BYTES = 165_888
_FIXED_ATTENTION_STATE_BYTES = 10_534_912
_COMBINED_STATE_BYTES = 10_700_800
_FIXED_ATTENTION_AND_EPISODIC_TRAFFIC_BYTES = 714_866_688
_SELECTOR_WEIGHT_TRAFFIC_BYTES = 21_233_664
_TOTAL_LOGICAL_TRAFFIC_BYTES = 736_100_352
_DENSE_FULL_CONTEXT_BYTES = 2_164_260_864
_EXACT_51_HEAD_CEILING_BYTES = 973_384_704
_HEADROOM_BYTES = 237_284_352
_MASS_MACS_PER_TOKEN = 229_376
_MASS_MACS_PER_SEQUENCE = 29_360_128

_FOLDS = (
    (0, (0, 4)),
    (1, (1, 5)),
    (2, (2, 6)),
    (3, (3, 7)),
)


def _progress(message: str) -> None:
    print(f"[phase-selector] {message}", file=sys.stderr, flush=True)


def _binding(path: Path, digest: str) -> dict[str, str]:
    return {"path": str(path), "sha256": digest}


def _source_inventory() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    return {name: sha256_file(repository / name) for name in _SOURCE_FILES}


def _resource_contract() -> dict[str, Any]:
    """Return the conservative all-token deployment accounting."""

    if _PHASES * _LAYERS * _HEADS * _COMPONENTS != _PHASE_PARAMETER_COUNT:
        raise AssertionError("phase-selector table arithmetic changed")
    if _PHASE_PARAMETER_COUNT + _MASS_PARAMETER_COUNT != _PARAMETER_COUNT:
        raise AssertionError("phase-selector parameter arithmetic changed")
    if _PARAMETER_COUNT * 2 != _BF16_PARAMETER_BYTES:
        raise AssertionError("phase-selector BF16 arithmetic changed")
    if _FIXED_ATTENTION_STATE_BYTES + _BF16_PARAMETER_BYTES != _COMBINED_STATE_BYTES:
        raise AssertionError("phase-selector state arithmetic changed")
    if _BF16_PARAMETER_BYTES * _POSITIONS_PER_RECORD != _SELECTOR_WEIGHT_TRAFFIC_BYTES:
        raise AssertionError("phase-selector traffic arithmetic changed")
    if (
        _FIXED_ATTENTION_AND_EPISODIC_TRAFFIC_BYTES + _SELECTOR_WEIGHT_TRAFFIC_BYTES
        != _TOTAL_LOGICAL_TRAFFIC_BYTES
    ):
        raise AssertionError("phase-selector total traffic changed")
    if _EXACT_51_HEAD_CEILING_BYTES - _TOTAL_LOGICAL_TRAFFIC_BYTES != _HEADROOM_BYTES:
        raise AssertionError("phase-selector headroom changed")
    return {
        "parameter_count": _PARAMETER_COUNT,
        "mass_selector_parameter_count": _MASS_PARAMETER_COUNT,
        "phase_table_parameter_count": _PHASE_PARAMETER_COUNT,
        "serialized_parameter_dtype": "BF16",
        "serialized_parameter_bytes": _BF16_PARAMETER_BYTES,
        "deployment_artifact_contains_BF16_only": True,
        "deployment_artifact_BF16_tensor_bytes": _BF16_PARAMETER_BYTES,
        "FP32_training_audit_copy_loaded_by_runtime": False,
        "fixed_attention_state_bytes": _FIXED_ATTENTION_STATE_BYTES,
        "combined_attention_and_selector_state_bytes": _COMBINED_STATE_BYTES,
        "mass_selector_multiply_accumulates_per_token": _MASS_MACS_PER_TOKEN,
        "mass_selector_multiply_accumulates_per_128_token_sequence": (
            _MASS_MACS_PER_SEQUENCE
        ),
        "phase_table_operation": "one indexed BF16 row add per active token",
        "conservative_selector_weight_traffic_bytes_per_128_token_sequence": (
            _SELECTOR_WEIGHT_TRAFFIC_BYTES
        ),
        "selector_weight_traffic_assumes_reload_for_every_token": True,
        "fixed_combined_attention_and_episodic_traffic_bytes": (
            _FIXED_ATTENTION_AND_EPISODIC_TRAFFIC_BYTES
        ),
        "total_logical_traffic_bytes_per_128_token_sequence": (
            _TOTAL_LOGICAL_TRAFFIC_BYTES
        ),
        "dense_full_context_logical_read_bytes": _DENSE_FULL_CONTEXT_BYTES,
        "fraction_of_dense_full_context_logical_reads": (
            _TOTAL_LOGICAL_TRAFFIC_BYTES / _DENSE_FULL_CONTEXT_BYTES
        ),
        "exact_51_head_equivalent_ceiling_bytes": (_EXACT_51_HEAD_CEILING_BYTES),
        "remaining_headroom_bytes": _HEADROOM_BYTES,
        "new_KV_state_bytes": 0,
        "new_KV_read_traffic_bytes": 0,
        "persistent_value_sidecar_bytes": 0,
        "full_KV_sidecar_or_second_value_read_pass": False,
        "future_native_fused_single_value_pass_required": True,
    }


def _fold_contract() -> list[dict[str, Any]]:
    records = tuple(range(_RECORDS))
    rows: list[dict[str, Any]] = []
    for fold_index, heldout in _FOLDS:
        heldout_set = set(heldout)
        rows.append(
            {
                "fold_index": fold_index,
                "training_record_indices": [
                    value for value in records if value not in heldout_set
                ],
                "heldout_record_indices": list(heldout),
                "initialization_seed": _INIT_SEED_BASE + fold_index,
                "shuffle_seed": _SHUFFLE_SEED_BASE + fold_index,
            }
        )
    return rows


def _derive_phase_schedule(
    positions: np.ndarray,
    *,
    block_positions: Sequence[int] = _BLOCK_POSITIONS,
    span_length: int = _PHASES,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive causal span-relative phase and its active mask."""

    position_array = np.asarray(positions)
    if (
        position_array.ndim != 1
        or position_array.dtype.kind not in "iu"
        or span_length != _PHASES
        or len(block_positions) == 0
    ):
        raise ValueError("phase-selector schedule shape changed")
    phase = np.zeros(position_array.shape, dtype=np.int64)
    active = np.zeros(position_array.shape, dtype=bool)
    for start_value in block_positions:
        start = int(start_value)
        selected = (position_array >= start) & (position_array < start + span_length)
        if np.any(active & selected):
            raise ValueError("phase-selector active read spans overlap")
        phase[selected] = position_array[selected].astype(np.int64) - start
        active |= selected
    if np.any((phase < 0) | (phase >= span_length)):
        raise ValueError("phase-selector derived phase is out of range")
    return np.ascontiguousarray(phase), np.ascontiguousarray(active)


def _schedule_contract() -> dict[str, Any]:
    return {
        "read_positions": list(range(96, 128)),
        "active_span_starts": list(_BLOCK_POSITIONS),
        "active_span_length": _PHASES,
        "phase_definition": (
            "phase=absolute_position-active_span_start for the unique active "
            "eight-token span; phase table contribution is exactly zero "
            "outside an active span"
        ),
        "phase_values_on_authenticated_reads": list(range(_PHASES)) * 4,
        "active_values_on_authenticated_reads": [True] * _READS,
        "absolute_position_or_token_identity_feature": False,
        "schedule_shift_equivariance_required": True,
    }


def _model_contract() -> dict[str, Any]:
    shape = selector.PhaseSelectorShape()
    return {
        "layers": shape.layers,
        "query_heads": shape.heads,
        "components_per_head": shape.components,
        "phases": shape.phases,
        "rank": shape.rank,
        "parameter_shapes": {
            "U": [shape.layers, shape.components, shape.rank],
            "V": [shape.layers, shape.rank, shape.components],
            "E": [shape.layers, shape.heads, shape.rank],
            "B": [shape.layers, shape.heads, shape.components],
            "T": [
                shape.phases,
                shape.layers,
                shape.heads,
                shape.components,
            ],
        },
        "mass_feature": "clipped centered log of authenticated native C28 mass",
        "phase_feature": (
            "causal offset 0..7 inside the currently active episodic read span"
        ),
        "forward": (
            "delta=gauge_clamp((relu(mass_features@U+E))@V+B+"
            "active*T[phase]); coefficients=normalize(native_mass*exp(delta))"
        ),
        "candidate_native_forward": (
            "replace reconstructed log_mass by centered raw q.k scores and "
            "masked_softmax(raw_scores+delta)"
        ),
        "inactive_behavior": "T contribution is exactly zero",
        "loss": "direct squared post-Wo residual error",
        "target_residual_is_label_only": True,
        "counterfactual_updates_hidden_or_cache_during_screen": False,
        "single_full_value_accumulation_pass_required": True,
    }


def _training_contract() -> dict[str, Any]:
    return {
        "cross_validation": {
            "folds": _fold_contract(),
            "out_of_fold_records": list(range(_RECORDS)),
            "one_fixed_final_checkpoint_per_fold": True,
            "heldout_checkpointing_or_early_stopping": False,
        },
        "steps": _TRAINING_STEPS,
        "warmup_steps": _WARMUP_STEPS,
        "epochs": _EPOCHS,
        "rows_per_layer_per_step": _ROWS_PER_LAYER_PER_STEP,
        "optimizer": "AdamW",
        "peak_learning_rate": _PEAK_LEARNING_RATE,
        "final_learning_rate": _FINAL_LEARNING_RATE,
        "learning_rate_schedule": "linear_warmup_then_cosine_decay",
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "weight_decay": {"U": 1.0e-4, "V": 1.0e-4, "T": 1.0e-4},
        "zero_weight_decay": ["E", "B"],
        "global_gradient_clip_norm": 1.0,
        "initialization": {
            "U_V_E_B": (
                "authenticated exact FP32 parameters from the matching "
                "predecessor mass-selector fold artifact"
            ),
            "T": "zero",
            "mass_branch_frozen_while_T_is_fit": True,
        },
        "sequential_fit": {
            "out_of_fold_mass_stage": (
                "reuse and authenticate the already-fitted matching predecessor "
                "mass-selector fold; do not refit or update U,V,E,B"
            ),
            "out_of_fold_phase_stage": (
                "initialize T=0 and optimize T only on that fold's declared "
                "training records"
            ),
            "no_joint_mass_phase_updates": True,
            "reason": (
                "preserve the known mass OOF baseline after joint content "
                "training was observed to damage it"
            ),
        },
        "training_precision": "FP32",
        "training_device": "cuda",
        "CUDA_deterministic_algorithms_required": True,
        "TF32_allowed": False,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "checkpoint_selection": "fixed final step only",
        "model_selection_scope": (
            "train-only architecture selection after predecessor OOF outcomes "
            "were observed; not independent development or confirmation evidence"
        ),
        "final_all_train_fit_on_out_of_fold_pass": {
            "training_record_indices": list(range(_RECORDS)),
            "steps": _FINAL_TRAINING_STEPS,
            "warmup_steps": _FINAL_WARMUP_STEPS,
            "epochs": _EPOCHS,
            "initialization_seed": _INIT_SEED_BASE + len(_FOLDS),
            "shuffle_seed": _SHUFFLE_SEED_BASE + len(_FOLDS),
            "sequential_stages": [
                (
                    "fit U,V,E,B with the exact predecessor mass-selector "
                    "all-train configuration"
                ),
                "freeze U,V,E,B; initialize T=0; fit T only",
            ],
            "not_used_for_out_of_fold_gate": True,
        },
    }


def _gate_contract() -> dict[str, Any]:
    return {
        "arms_required": ["FP32", "BF16_RNE"],
        "minimum_global_recovery": 0.50,
        "minimum_every_out_of_fold_sequence_recovery": 0.25,
        "minimum_every_block_entry_position_recovery": 0.25,
        "block_entry_positions": list(_BLOCK_POSITIONS),
        "minimum_positive_recovery_layers": 12,
        "maximum_BF16_global_recovery_drop_from_FP32": 0.005,
        "finite_simplex_mask_and_zero_model_parity_required": True,
        "mass_vs_reconstructed_score_reference_tolerance": 1.0e-6,
        "deterministic_serialized_replay_exact": True,
        "schedule_shift_equivariance_exact": True,
        "inactive_phase_table_disable_exact": True,
        "resource_below_exact_51_head_ceiling_required": True,
        "final_all_train_BF16_artifact_sanity_required_for_native_integration": True,
    }


def _validate_content_failure(
    value: Mapping[str, Any],
    *,
    protocol_path: Path,
) -> dict[str, Any]:
    gate = value.get("gate")
    decision = value.get("decision")
    arms = value.get("arms")
    post = value.get("post_fit_authentication")
    if (
        value.get("schema_version") != _SCHEMA_VERSION
        or value.get("experiment") != content._RESULT_EXPERIMENT
        or value.get("status") != content._FAILED_STATUS
        or value.get("protocol")
        != _binding(protocol_path, _EXPECTED_CONTENT_PROTOCOL_SHA256)
        or value.get("confirmation_split_opened") is not False
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not False
        or gate.get("FP32_gate_passed") is not False
        or gate.get("BF16_gate_passed") is not False
        or not isinstance(decision, Mapping)
        or decision.get("train_content_selector_oof_gate_passed") is not False
        or decision.get("train_only_native_integration_implementation_authorized")
        is not False
        or decision.get("development_authorized") is not False
        or decision.get("confirmation_authorized") is not False
        or decision.get("semantic_or_M3_gate_passed") is not False
        or not isinstance(arms, Mapping)
        or not isinstance(post, Mapping)
        or not post
        or not all(item is True for item in post.values())
    ):
        raise ValueError("phase-selector content predecessor failure changed")
    fp32 = arms.get("FP32", {}).get("metrics", {})
    bf16 = arms.get("BF16_RNE_parameters_and_sidecars", {}).get("metrics", {})
    expected = {
        "FP32_global_recovery": 0.25426155258896843,
        "BF16_global_recovery": 0.2542207419770499,
        "FP32_minimum_sequence_recovery": 0.23154313101688695,
        "BF16_minimum_sequence_recovery": 0.2316160008520175,
        "FP32_minimum_block_recovery": 0.18379959630601683,
        "BF16_minimum_block_recovery": 0.1837115447332116,
        "FP32_positive_layers": 16,
        "BF16_positive_layers": 16,
    }
    observed = {
        "FP32_global_recovery": fp32.get("global", {}).get("recovery"),
        "BF16_global_recovery": bf16.get("global", {}).get("recovery"),
        "FP32_minimum_sequence_recovery": min(
            row["recovery"] for row in fp32.get("heldout_sequences", [])
        ),
        "BF16_minimum_sequence_recovery": min(
            row["recovery"] for row in bf16.get("heldout_sequences", [])
        ),
        "FP32_minimum_block_recovery": min(
            row["recovery"] for row in fp32.get("block_entry_positions", [])
        ),
        "BF16_minimum_block_recovery": min(
            row["recovery"] for row in bf16.get("block_entry_positions", [])
        ),
        "FP32_positive_layers": fp32.get("positive_recovery_layer_count"),
        "BF16_positive_layers": bf16.get("positive_recovery_layer_count"),
    }
    if observed != expected:
        raise ValueError("phase-selector content predecessor metrics changed")
    return expected


def _authenticate_mass_fold_artifacts(
    mass_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Authenticate the exact four predecessor mass fold checkpoints."""

    artifacts = mass_result.get("artifacts")
    folds = mass_result.get("folds")
    if (
        not isinstance(artifacts, Mapping)
        or not isinstance(folds, list)
        or len(folds) != len(_FOLDS)
    ):
        raise ValueError("phase-selector mass artifact inventory changed")
    directory_value = artifacts.get("directory")
    if not isinstance(directory_value, str):
        raise ValueError("phase-selector mass artifact directory changed")
    directory = Path(directory_value).expanduser()
    mass_runner._guard_paths((("phase-selector mass artifact directory", directory),))
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("phase-selector mass artifact directory changed")
    shape = mass.SelectorShape()
    rows: list[dict[str, Any]] = []
    for expected_fold, row in zip(_fold_contract(), folds, strict=True):
        descriptor = row.get("artifact") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or row.get("fold_index") != expected_fold["fold_index"]
            or row.get("training_record_indices")
            != expected_fold["training_record_indices"]
            or row.get("heldout_record_indices")
            != expected_fold["heldout_record_indices"]
            or not isinstance(descriptor, Mapping)
            or descriptor.get("artifact_role") != "training_audit"
            or descriptor.get("authorized_for_runtime_loading") is not False
            or descriptor.get("parameter_count") != _MASS_PARAMETER_COUNT
        ):
            raise ValueError("phase-selector mass fold binding changed")
        artifact_path = directory / str(descriptor.get("file", ""))
        mass_runner._guard_paths(
            (("phase-selector mass fold artifact", artifact_path),)
        )
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise ValueError("phase-selector mass fold artifact changed")
        fp32 = mass_runner._load_model_artifact(
            artifact_path,
            descriptor,
            shape,
            bf16=False,
        )
        bf16 = mass_runner._load_model_artifact(
            artifact_path,
            descriptor,
            shape,
            bf16=True,
        )
        rows.append(
            {
                "fold_index": int(row["fold_index"]),
                "path": artifact_path.resolve(),
                "descriptor": dict(descriptor),
                "FP32_parameters": fp32,
                "BF16_parameters": bf16,
            }
        )
    return rows


def _authenticate_inputs(
    *,
    content_protocol: str | Path,
    content_protocol_sha256: str,
    content_result: str | Path,
    content_result_sha256: str,
) -> dict[str, Any]:
    mass_runner._guard_paths(
        (
            ("phase-selector content protocol", content_protocol),
            ("phase-selector content result", content_result),
        )
    )
    if content_protocol_sha256 != _EXPECTED_CONTENT_PROTOCOL_SHA256:
        raise ValueError("phase-selector content protocol root changed")
    protocol_path, frozen, predecessor_context = content._authenticate_frozen_protocol(
        content_protocol,
        content_protocol_sha256,
    )
    result_path = content._checked_exact(
        content_result,
        content_result_sha256,
        _EXPECTED_CONTENT_RESULT_SHA256,
        "phase-selector content result",
    )
    result = mass_runner._read_json(result_path, "phase-selector content result")
    failure = _validate_content_failure(result, protocol_path=protocol_path)
    mass_fold_artifacts = _authenticate_mass_fold_artifacts(
        predecessor_context["mass_result"]
    )
    # The content protocol authentication above re-authenticates the exact mass
    # failure, full/residual tensor join, package, non-MLP weights, and query
    # weight roots.  Keep each binding explicit in the returned context.
    return {
        "content_protocol_path": protocol_path,
        "content_protocol": frozen,
        "content_result_path": result_path,
        "content_result": result,
        "content_failure": failure,
        "predecessor": predecessor_context,
        "mass_fold_artifacts": mass_fold_artifacts,
    }


def _build_protocol(context: Mapping[str, Any]) -> dict[str, Any]:
    predecessor_context = context["predecessor"]
    capacity = predecessor_context["capacity"]
    package = predecessor_context["package"]
    content_protocol = context["content_protocol"]
    positions = content._query_positions(predecessor_context)
    phase, active = _derive_phase_schedule(positions)
    if phase.tolist() != list(range(_PHASES)) * 4 or not np.all(active):
        raise ValueError("phase-selector authenticated schedule changed")
    return {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _PROTOCOL_EXPERIMENT,
        "status": _PROTOCOL_STATUS,
        "predecessor_content_protocol": _binding(
            context["content_protocol_path"],
            _EXPECTED_CONTENT_PROTOCOL_SHA256,
        ),
        "predecessor_content_result": {
            **_binding(
                context["content_result_path"],
                _EXPECTED_CONTENT_RESULT_SHA256,
            ),
            "authenticated_failure_metrics": dict(context["content_failure"]),
        },
        "predecessor_mass_protocol": dict(
            content_protocol["predecessor_mass_protocol"]
        ),
        "predecessor_mass_result": dict(content_protocol["predecessor_mass_result"]),
        "predecessor_mass_fold_artifacts": [
            {
                "fold_index": row["fold_index"],
                **_binding(
                    row["path"],
                    str(row["descriptor"]["file_sha256"]),
                ),
                "FP32_parameter_sha256": row["descriptor"]["fp32_parameter_sha256"],
                "BF16_parameter_sha256": row["descriptor"][
                    "bf16_decoded_parameter_sha256"
                ],
            }
            for row in context["mass_fold_artifacts"]
        ],
        "full_visible_capacity_protocol": _binding(
            capacity["capacity_protocol_path"],
            capacity["capacity_protocol_sha256"],
        ),
        "full_visible_capacity_result": _binding(
            capacity["capacity_result_path"],
            capacity["capacity_result_sha256"],
        ),
        "full_visible_trace_manifest": _binding(
            capacity["trace_manifest_path"],
            capacity["trace_manifest_sha256"],
        ),
        "residual_input_trace_manifest": _binding(
            predecessor_context["residual_manifest_path"],
            _EXPECTED_RESIDUAL_MANIFEST_SHA256,
        ),
        "package_manifest": _binding(
            package["manifest_path"],
            _EXPECTED_PACKAGE_MANIFEST_SHA256,
        ),
        "non_mlp_weights": _binding(
            package["non_mlp_path"],
            _EXPECTED_NON_MLP_SHA256,
        ),
        "cross_manifest_record_join": dict(predecessor_context["record_join"]),
        "selector_model": _model_contract(),
        "phase_schedule": _schedule_contract(),
        "training": _training_contract(),
        "serialization_and_parity": {
            "production_parameter_dtype": "BF16",
            "BF16_rounding": "round_to_nearest_even",
            "FP32_and_BF16_out_of_fold_evaluation_required": True,
            "native_raw_score_parity_deferred_to_native_integration": True,
            "final_native_artifact_contains_BF16_only": True,
        },
        "progression_gate": _gate_contract(),
        "resource_contract": _resource_contract(),
        "scope": {
            "split": "train",
            "records": _RECORDS,
            "read_positions_per_record": _READS,
            "out_of_fold_fit": True,
            "causal_features_only": True,
            "cached_same_state_screen": True,
            "native_causal_rollout": False,
            "predecessor_OOF_outcomes_exposed_for_model_selection": True,
            "independent_generalization_claim": False,
            "development_outcomes_used": False,
            "confirmation_file_access_permitted": False,
            "semantic_or_M3_pass": False,
        },
        "progression_authority": {
            "on_pass": (
                "authorize native fused implementation and train-only parity; "
                "a separate frozen causal development rollout remains required"
            ),
            "on_fail": (
                "reject this phase-conditioned mass-selector class without "
                "tuning it on these OOF outcomes"
            ),
            "does_not_authorize_development": True,
            "does_not_authorize_confirmation": True,
            "does_not_pass_semantic_or_M3_gate": True,
        },
        "source_sha256": _source_inventory(),
        "authenticated_confirmation_descriptor": dict(
            content_protocol["authenticated_confirmation_descriptor"]
        ),
        "confirmation_split_opened": False,
    }


def freeze_phase_selector_protocol(
    *,
    content_protocol: str | Path,
    content_protocol_sha256: str,
    content_result: str | Path,
    content_result_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    mass_runner._guard_paths(
        (
            ("phase-selector content protocol", content_protocol),
            ("phase-selector content result", content_result),
            ("phase-selector protocol output", out),
        )
    )
    context = _authenticate_inputs(
        content_protocol=content_protocol,
        content_protocol_sha256=content_protocol_sha256,
        content_result=content_result,
        content_result_sha256=content_result_sha256,
    )
    protocol = _build_protocol(context)
    output = mass_runner._new_output(out, "phase-selector protocol output")
    mass_runner._atomic_json_new(output, protocol)
    if mass_runner._read_json(output, "phase-selector protocol replay") != protocol:
        raise AssertionError("phase-selector protocol replay changed")
    return {"path": str(output), "sha256": sha256_file(output), "protocol": protocol}


def _authenticate_frozen_protocol(
    protocol: str | Path,
    protocol_sha256: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    mass_runner._guard_paths((("phase-selector protocol", protocol),))
    requested = Path(protocol).expanduser()
    if (
        requested.is_symlink()
        or not requested.is_file()
        or not mass_runner._is_sha256(protocol_sha256)
        or sha256_file(requested) != protocol_sha256
    ):
        raise ValueError("phase-selector protocol authentication failed")
    path = requested.resolve()
    frozen = mass_runner._read_json(path, "phase-selector protocol")
    content_protocol = frozen.get("predecessor_content_protocol")
    content_result = frozen.get("predecessor_content_result")
    if not isinstance(content_protocol, Mapping) or not isinstance(
        content_result,
        Mapping,
    ):
        raise ValueError("phase-selector predecessor bindings changed")
    context = _authenticate_inputs(
        content_protocol=str(content_protocol.get("path")),
        content_protocol_sha256=str(content_protocol.get("sha256")),
        content_result=str(content_result.get("path")),
        content_result_sha256=str(content_result.get("sha256")),
    )
    expected = _build_protocol(context)
    if frozen != expected:
        raise ValueError("phase-selector frozen protocol changed")
    return path, frozen, context


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
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-8,
        uv_weight_decay=1.0e-4,
        t_weight_decay=1.0e-4,
        gradient_clip_norm=1.0,
        rows_per_layer_per_step=_ROWS_PER_LAYER_PER_STEP,
        epochs=_EPOCHS,
        feature_clip=16.0,
        delta_clamp=16.0,
        initial_u_standard_deviation=0.02,
        init_seed=init_seed,
        shuffle_seed=shuffle_seed,
    )


def _array_sha256(value: np.ndarray) -> str:
    return mass_runner._array_sha256(value)


def _phase_parameters_from_mass(
    parameters: mass.SelectorParameters,
    shape: selector.PhaseSelectorShape,
    *,
    table: np.ndarray | None = None,
) -> selector.PhaseSelectorParameters:
    parameters.validate(mass.SelectorShape())
    result = selector.PhaseSelectorParameters(
        U=np.ascontiguousarray(parameters.U, dtype=np.float32),
        V=np.ascontiguousarray(parameters.V, dtype=np.float32),
        E=np.ascontiguousarray(parameters.E, dtype=np.float32),
        B=np.ascontiguousarray(parameters.B, dtype=np.float32),
        T=(
            np.zeros(
                (
                    shape.phases,
                    shape.layers,
                    shape.heads,
                    shape.components,
                ),
                dtype=np.float32,
            )
            if table is None
            else np.ascontiguousarray(table, dtype=np.float32)
        ),
    )
    result.validate(shape)
    return result


def _parameters_from_tensors(
    tensors: Mapping[str, np.ndarray],
    *,
    suffix: str,
    shape: selector.PhaseSelectorShape,
) -> selector.PhaseSelectorParameters:
    values: dict[str, np.ndarray] = {}
    for name in ("U", "V", "E", "B", "T"):
        tensor = tensors[f"{name}_{suffix}"]
        values[name] = (
            selector.bf16_bits_to_float32(tensor)
            if suffix == "bf16_bits"
            else np.ascontiguousarray(tensor, dtype=np.float32)
        )
    parameters = selector.PhaseSelectorParameters(**values)
    parameters.validate(shape)
    return parameters


def _write_audit_artifact(
    path: Path,
    parameters: selector.PhaseSelectorParameters,
    shape: selector.PhaseSelectorShape,
) -> dict[str, Any]:
    parameters.validate(shape)
    decoded, bits = selector.quantize_parameters_bf16(parameters, shape)
    tensors: dict[str, np.ndarray] = {}
    for name, value in parameters.as_dict().items():
        tensors[f"{name}_fp32"] = value
        tensors[f"{name}_bf16_bits"] = bits[name]
    mass_runner._atomic_safetensors(path, tensors)
    loaded = load_safetensors(str(path))
    if set(loaded) != set(tensors) or any(
        not np.array_equal(loaded[name], expected) for name, expected in tensors.items()
    ):
        raise ValueError("phase-selector audit artifact replay changed")
    fp32 = _parameters_from_tensors(loaded, suffix="fp32", shape=shape)
    bf16 = _parameters_from_tensors(loaded, suffix="bf16_bits", shape=shape)
    if selector.parameters_sha256(fp32, shape) != selector.parameters_sha256(
        parameters, shape
    ) or selector.parameters_sha256(bf16, shape) != selector.parameters_sha256(
        decoded, shape
    ):
        raise ValueError("phase-selector audit parameter replay changed")
    return {
        "file": path.name,
        "file_sha256": sha256_file(path),
        "artifact_role": "training_audit",
        "authorized_for_runtime_loading": False,
        "parameter_count": shape.parameter_count,
        "FP32_tensor_bytes": shape.parameter_count * 4,
        "BF16_tensor_bytes": shape.parameter_count * 2,
        "FP32_parameter_sha256": selector.parameters_sha256(fp32, shape),
        "BF16_decoded_parameter_sha256": selector.parameters_sha256(bf16, shape),
        "tensor_sha256": {
            name: _array_sha256(value) for name, value in tensors.items()
        },
        "deterministic_replay_exact": True,
    }


def _load_audit_artifact(
    path: Path,
    descriptor: Mapping[str, Any],
    shape: selector.PhaseSelectorShape,
    *,
    bf16: bool,
) -> selector.PhaseSelectorParameters:
    if (
        path.name != descriptor.get("file")
        or sha256_file(path) != descriptor.get("file_sha256")
        or descriptor.get("artifact_role") != "training_audit"
        or descriptor.get("authorized_for_runtime_loading") is not False
    ):
        raise ValueError("phase-selector audit artifact authentication failed")
    parameters = _parameters_from_tensors(
        load_safetensors(str(path)),
        suffix="bf16_bits" if bf16 else "fp32",
        shape=shape,
    )
    expected = descriptor[
        "BF16_decoded_parameter_sha256" if bf16 else "FP32_parameter_sha256"
    ]
    if selector.parameters_sha256(parameters, shape) != expected:
        raise ValueError("phase-selector audit parameter root changed")
    return parameters


def _write_deployment_artifact(
    path: Path,
    parameters: selector.PhaseSelectorParameters,
    shape: selector.PhaseSelectorShape,
) -> dict[str, Any]:
    parameters.validate(shape)
    decoded, bits = selector.quantize_parameters_bf16(parameters, shape)
    tensors = {f"{name}_bf16_bits": value for name, value in bits.items()}
    if sum(value.nbytes for value in tensors.values()) != shape.parameter_count * 2:
        raise AssertionError("phase-selector deployment bytes changed")
    mass_runner._atomic_safetensors(path, tensors)
    loaded = load_safetensors(str(path))
    if set(loaded) != set(tensors) or any(
        not np.array_equal(loaded[name], expected) for name, expected in tensors.items()
    ):
        raise ValueError("phase-selector deployment replay changed")
    replay = _parameters_from_tensors(loaded, suffix="bf16_bits", shape=shape)
    if selector.parameters_sha256(replay, shape) != selector.parameters_sha256(
        decoded,
        shape,
    ):
        raise ValueError("phase-selector deployment parameter replay changed")
    return {
        "file": path.name,
        "file_sha256": sha256_file(path),
        "artifact_role": "native_BF16_deployment",
        "authorized_for_runtime_loading": True,
        "contains_FP32_training_copy": False,
        "parameter_count": shape.parameter_count,
        "BF16_tensor_bytes": shape.parameter_count * 2,
        "BF16_decoded_parameter_sha256": selector.parameters_sha256(replay, shape),
        "tensor_sha256": {
            name: _array_sha256(value) for name, value in tensors.items()
        },
        "deterministic_replay_exact": True,
    }


def _load_deployment_artifact(
    path: Path,
    descriptor: Mapping[str, Any],
    shape: selector.PhaseSelectorShape,
) -> selector.PhaseSelectorParameters:
    if (
        path.name != descriptor.get("file")
        or sha256_file(path) != descriptor.get("file_sha256")
        or descriptor.get("artifact_role") != "native_BF16_deployment"
        or descriptor.get("authorized_for_runtime_loading") is not True
        or descriptor.get("contains_FP32_training_copy") is not False
        or descriptor.get("BF16_tensor_bytes") != shape.parameter_count * 2
    ):
        raise ValueError("phase-selector deployment authentication failed")
    parameters = _parameters_from_tensors(
        load_safetensors(str(path)),
        suffix="bf16_bits",
        shape=shape,
    )
    if (
        selector.parameters_sha256(parameters, shape)
        != descriptor["BF16_decoded_parameter_sha256"]
    ):
        raise ValueError("phase-selector deployment parameter root changed")
    return parameters


def _arm_metrics(
    target_energy: np.ndarray,
    error_energy: np.ndarray,
) -> dict[str, Any]:
    return mass_runner._arm_metrics(target_energy, error_energy)


def _maximum_output_correction(
    coefficients: np.ndarray,
    values: np.ndarray,
    base_heads: np.ndarray,
    projection: np.ndarray,
    shape: selector.PhaseSelectorShape,
) -> float:
    return mass_runner._maximum_output_correction(
        coefficients,
        values,
        base_heads,
        projection,
        mass.SelectorShape(),
    )


def _without_phase_table(
    parameters: selector.PhaseSelectorParameters,
    shape: selector.PhaseSelectorShape,
) -> selector.PhaseSelectorParameters:
    return selector.PhaseSelectorParameters(
        U=parameters.U,
        V=parameters.V,
        E=parameters.E,
        B=parameters.B,
        T=np.zeros_like(parameters.T),
    )


def _selector_gate(
    fp32_metrics: Mapping[str, Any],
    bf16_metrics: Mapping[str, Any],
    *,
    fp32_coefficients: np.ndarray,
    bf16_coefficients: np.ndarray,
    valid: np.ndarray,
    score_coefficient_max_abs: float,
    score_delta_max_abs: float,
    zero_coefficient_max_abs: float,
    zero_output_max_abs: float,
    deterministic_replay_exact: bool,
    schedule_shift_coefficient_max_abs: float,
    schedule_shift_phase_exact: bool,
    inactive_disable_coefficient_max_abs: float,
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
    resource = _resource_contract()
    checks = {
        "FP32_gate_passed": fp32_metrics.get("passed") is True,
        "BF16_gate_passed": bf16_metrics.get("passed") is True,
        "BF16_global_recovery_drop_within_0_005": (fp32_global - bf16_global <= 0.005),
        "finite": finite,
        "simplex_and_invalid_mask": simplex,
        "mass_vs_reconstructed_score_coefficient_parity": (
            score_coefficient_max_abs <= 1.0e-6
        ),
        "mass_vs_reconstructed_score_delta_parity": (score_delta_max_abs <= 1.0e-6),
        "zero_model_native_coefficient_parity": (zero_coefficient_max_abs <= 1.0e-6),
        "zero_model_native_output_parity": zero_output_max_abs <= 1.0e-6,
        "deterministic_serialized_replay_exact": deterministic_replay_exact,
        "schedule_shift_phase_and_active_exact": schedule_shift_phase_exact,
        "schedule_shift_coefficient_equivariance_exact": (
            schedule_shift_coefficient_max_abs == 0.0
        ),
        "inactive_phase_table_disable_exact": (
            inactive_disable_coefficient_max_abs == 0.0
        ),
        "resource_below_exact_51_head_ceiling": (
            resource["total_logical_traffic_bytes_per_128_token_sequence"]
            < resource["exact_51_head_equivalent_ceiling_bytes"]
        ),
        "native_raw_score_parity_deferred": True,
    }
    checks["passed"] = all(checks.values())
    return checks


def _mass_branch_equal(
    phase_parameters: selector.PhaseSelectorParameters,
    mass_parameters: mass.SelectorParameters,
) -> bool:
    return all(
        np.array_equal(
            phase_parameters.as_dict()[name],
            mass_parameters.as_dict()[name],
        )
        for name in ("U", "V", "E", "B")
    )


def fit_screen_phase_selector(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    artifact_directory: str | Path,
    out: str | Path,
    device: str,
) -> dict[str, Any]:
    """Fit T only in fixed OOF folds and evaluate serialized BF16 artifacts."""

    mass_runner._guard_paths(
        (
            ("phase-selector protocol", protocol),
            ("phase-selector artifact directory", artifact_directory),
            ("phase-selector result output", out),
        )
    )
    protocol_path, frozen, context = _authenticate_frozen_protocol(
        protocol,
        protocol_sha256,
    )
    if device != frozen["training"]["training_device"]:
        raise ValueError("phase-selector training device changed")
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != frozen["training"]["CUBLAS_WORKSPACE_CONFIG"]
    ):
        raise ValueError("phase-selector CUDA workspace configuration changed")
    output = mass_runner._new_output(out, "phase-selector result output")
    artifacts = mass_runner._new_directory(
        artifact_directory,
        "phase-selector artifact directory",
    )

    _progress("loading authenticated full-visible train trace")
    predecessor_context = context["predecessor"]
    capacity = predecessor_context["capacity"]
    capacity_binding = _binding(
        capacity["capacity_protocol_path"],
        capacity["capacity_protocol_sha256"],
    )
    arrays, manifest = full.load_stacked_full_visible_trace(
        capacity["trace_manifest_path"],
        capacity["trace_manifest_sha256"],
        protocol=capacity_binding,
    )
    record_authentication = full._audit_manifest_record_bindings(
        capacity["capacity_protocol"],
        manifest,
    )
    projection = full._load_authenticated_output_projection(
        capacity["capacity_protocol"]
    )
    basis = full.build_full_visible_basis(
        arrays,
        query_heads=_HEADS,
        regular_entries=full._REGULAR_ENTRIES,
        episodic_entries=full._EPISODIC_ENTRIES,
        include_exact_native_anchor=False,
    )
    shape = selector.PhaseSelectorShape()
    native_mass = np.ascontiguousarray(basis.visible_mass, dtype=np.float32)
    valid = np.ascontiguousarray(basis.visible_valid, dtype=bool)
    values = np.ascontiguousarray(basis.visible_values, dtype=np.float32)
    base_heads = np.ascontiguousarray(basis.base_heads, dtype=np.float32)
    target = np.ascontiguousarray(basis.target_residual, dtype=np.float32)
    expected_mass_shape = (
        _RECORDS,
        _READS,
        _LAYERS,
        _HEADS,
        _COMPONENTS,
    )
    if (
        shape.parameter_count != _PARAMETER_COUNT
        or native_mass.shape != expected_mass_shape
        or valid.shape != expected_mass_shape
        or values.shape != expected_mass_shape + (_HEAD_DIMENSION,)
        or base_heads.shape != (_RECORDS, _READS, _LAYERS, _HEADS, _HEAD_DIMENSION)
        or target.shape != (_RECORDS, _READS, _LAYERS, _HIDDEN_SIZE)
        or projection.shape != (_LAYERS, _HIDDEN_SIZE, _HIDDEN_SIZE)
    ):
        raise ValueError("phase-selector authenticated tensor shape changed")
    positions = content._query_positions(predecessor_context)
    phase_one, active_one = _derive_phase_schedule(positions)
    shifted_phase_one, shifted_active_one = _derive_phase_schedule(
        positions + _PHASES,
        block_positions=tuple(value + _PHASES for value in _BLOCK_POSITIONS),
    )
    schedule_shift_phase_exact = bool(
        np.array_equal(phase_one, shifted_phase_one)
        and np.array_equal(active_one, shifted_active_one)
    )
    phase = np.ascontiguousarray(
        np.broadcast_to(phase_one[None, :], (_RECORDS, _READS))
    )
    active = np.ascontiguousarray(
        np.broadcast_to(active_one[None, :], (_RECORDS, _READS))
    )
    shifted_phase = np.ascontiguousarray(
        np.broadcast_to(shifted_phase_one[None, :], (_RECORDS, _READS))
    )
    shifted_active = np.ascontiguousarray(
        np.broadcast_to(shifted_active_one[None, :], (_RECORDS, _READS))
    )

    fp32_oof = np.empty(expected_mass_shape, dtype=np.float32)
    bf16_oof = np.empty(expected_mass_shape, dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    deterministic_replay = True
    score_coefficient_max_abs = 0.0
    score_delta_max_abs = 0.0
    schedule_shift_coefficient_max_abs = 0.0
    inactive_disable_coefficient_max_abs = 0.0
    for fold, inherited in zip(
        frozen["training"]["cross_validation"]["folds"],
        context["mass_fold_artifacts"],
        strict=True,
    ):
        fold_index = int(fold["fold_index"])
        if inherited["fold_index"] != fold_index:
            raise ValueError("phase-selector inherited fold order changed")
        training_records = tuple(
            int(value) for value in fold["training_record_indices"]
        )
        heldout = np.asarray(fold["heldout_record_indices"], dtype=np.int64)
        _progress(
            f"training phase table fold {fold_index + 1}/4 with frozen "
            f"mass branch; held out {heldout.tolist()}"
        )
        base_fp32 = inherited["FP32_parameters"]
        base_bf16 = inherited["BF16_parameters"]
        trained = selector.fit_phase_table_direct_post_wo(
            native_mass,
            valid,
            phase,
            active,
            values,
            base_heads,
            target,
            projection,
            base_parameters=base_fp32,
            training_records=training_records,
            shape=shape,
            config=_training_config(
                init_seed=int(fold["initialization_seed"]),
                shuffle_seed=int(fold["shuffle_seed"]),
            ),
            device=device,
        )
        if not _mass_branch_equal(trained.parameters, base_fp32):
            raise ValueError("phase-selector changed the frozen FP32 mass branch")
        artifact_path = artifacts / f"fold-{fold_index}.safetensors"
        descriptor = _write_audit_artifact(
            artifact_path,
            trained.parameters,
            shape,
        )
        fp32_parameters = _load_audit_artifact(
            artifact_path,
            descriptor,
            shape,
            bf16=False,
        )
        bf16_parameters = _load_audit_artifact(
            artifact_path,
            descriptor,
            shape,
            bf16=True,
        )
        if not _mass_branch_equal(fp32_parameters, base_fp32) or not _mass_branch_equal(
            bf16_parameters, base_bf16
        ):
            raise ValueError(
                "phase-selector serialized artifact changed inherited mass branch"
            )
        fp32_coefficients, fp32_delta = selector.selector_forward(
            native_mass[heldout],
            valid[heldout],
            phase[heldout],
            active[heldout],
            fp32_parameters,
            shape,
        )
        bf16_coefficients, bf16_delta = selector.selector_forward(
            native_mass[heldout],
            valid[heldout],
            phase[heldout],
            active[heldout],
            bf16_parameters,
            shape,
        )
        fp32_replay, _ = selector.selector_forward(
            native_mass[heldout],
            valid[heldout],
            phase[heldout],
            active[heldout],
            fp32_parameters,
            shape,
        )
        bf16_replay, _ = selector.selector_forward(
            native_mass[heldout],
            valid[heldout],
            phase[heldout],
            active[heldout],
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
        for parameters, coefficients, delta in (
            (fp32_parameters, fp32_coefficients, fp32_delta),
            (bf16_parameters, bf16_coefficients, bf16_delta),
        ):
            scores = np.zeros(native_mass[heldout].shape, dtype=np.float32)
            heldout_valid = valid[heldout]
            scores[heldout_valid] = np.log(native_mass[heldout][heldout_valid])
            score_coefficients, score_delta = selector.selector_forward_from_scores(
                scores,
                heldout_valid,
                phase[heldout],
                active[heldout],
                parameters,
                shape,
            )
            score_coefficient_max_abs = max(
                score_coefficient_max_abs,
                float(np.max(np.abs(coefficients - score_coefficients))),
            )
            score_delta_max_abs = max(
                score_delta_max_abs,
                float(np.max(np.abs(delta - score_delta))),
            )
            shifted_coefficients, _ = selector.selector_forward(
                native_mass[heldout],
                heldout_valid,
                shifted_phase[heldout],
                shifted_active[heldout],
                parameters,
                shape,
            )
            schedule_shift_coefficient_max_abs = max(
                schedule_shift_coefficient_max_abs,
                float(np.max(np.abs(coefficients - shifted_coefficients))),
            )
            inactive_coefficients, _ = selector.selector_forward(
                native_mass[heldout],
                heldout_valid,
                phase[heldout],
                np.zeros_like(active[heldout]),
                parameters,
                shape,
            )
            zero_table_coefficients, _ = selector.selector_forward(
                native_mass[heldout],
                heldout_valid,
                phase[heldout],
                active[heldout],
                _without_phase_table(parameters, shape),
                shape,
            )
            inactive_disable_coefficient_max_abs = max(
                inactive_disable_coefficient_max_abs,
                float(np.max(np.abs(inactive_coefficients - zero_table_coefficients))),
            )
        fold_rows.append(
            {
                "fold_index": fold_index,
                "training_record_indices": list(training_records),
                "heldout_record_indices": heldout.tolist(),
                "inherited_mass_artifact": {
                    **_binding(
                        inherited["path"],
                        inherited["descriptor"]["file_sha256"],
                    ),
                    "FP32_parameter_sha256": inherited["descriptor"][
                        "fp32_parameter_sha256"
                    ],
                    "BF16_parameter_sha256": inherited["descriptor"][
                        "bf16_decoded_parameter_sha256"
                    ],
                },
                "mass_branch_frozen_and_preserved_exact": True,
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
        standard_deviation=0.02,
    )
    zero_coefficients, _ = selector.selector_forward(
        native_mass,
        valid,
        phase,
        active,
        zero,
        shape,
    )
    zero_coefficient_max_abs = float(np.max(np.abs(zero_coefficients - native_mass)))
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
        raise ValueError("phase-selector target energy replay changed")
    fp32_metrics = _arm_metrics(target_energy, fp32_error)
    bf16_metrics = _arm_metrics(target_energy, bf16_error)
    gate = _selector_gate(
        fp32_metrics,
        bf16_metrics,
        fp32_coefficients=fp32_oof,
        bf16_coefficients=bf16_oof,
        valid=valid,
        score_coefficient_max_abs=score_coefficient_max_abs,
        score_delta_max_abs=score_delta_max_abs,
        zero_coefficient_max_abs=zero_coefficient_max_abs,
        zero_output_max_abs=zero_output_max_abs,
        deterministic_replay_exact=deterministic_replay,
        schedule_shift_coefficient_max_abs=(schedule_shift_coefficient_max_abs),
        schedule_shift_phase_exact=schedule_shift_phase_exact,
        inactive_disable_coefficient_max_abs=(inactive_disable_coefficient_max_abs),
    )
    oof_path = artifacts / "oof-coefficients.safetensors"
    oof_tensors = {
        "FP32_coefficients": np.ascontiguousarray(fp32_oof),
        "BF16_coefficients": np.ascontiguousarray(bf16_oof),
        "FP32_error_energy": np.ascontiguousarray(fp32_error, dtype=np.float64),
        "BF16_error_energy": np.ascontiguousarray(bf16_error, dtype=np.float64),
        "target_energy": np.ascontiguousarray(target_energy, dtype=np.float64),
        "phase": phase,
        "active": active,
    }
    mass_runner._atomic_safetensors(oof_path, oof_tensors)
    oof_replay = load_safetensors(str(oof_path))
    if set(oof_replay) != set(oof_tensors) or any(
        not np.array_equal(oof_replay[name], expected)
        for name, expected in oof_tensors.items()
    ):
        raise ValueError("phase-selector OOF artifact replay changed")

    final_artifact: dict[str, Any] | None = None
    final_artifact_sanity_passed = False
    if gate["passed"]:
        _progress("OOF gate passed; executing declared sequential all-train fit")
        final_contract = frozen["training"]["final_all_train_fit_on_out_of_fold_pass"]
        final_mass = mass.fit_direct_post_wo(
            native_mass,
            valid,
            values,
            base_heads,
            target,
            projection,
            training_records=tuple(range(_RECORDS)),
            shape=mass.SelectorShape(),
            config=mass_runner._training_config(
                init_seed=int(final_contract["initialization_seed"]),
                shuffle_seed=int(final_contract["shuffle_seed"]),
                final_fit=True,
            ),
            device=device,
        )
        final_trained = selector.fit_phase_table_direct_post_wo(
            native_mass,
            valid,
            phase,
            active,
            values,
            base_heads,
            target,
            projection,
            base_parameters=final_mass.parameters,
            training_records=tuple(range(_RECORDS)),
            shape=shape,
            config=_training_config(
                init_seed=int(final_contract["initialization_seed"]),
                shuffle_seed=int(final_contract["shuffle_seed"]),
                final_fit=True,
            ),
            device=device,
        )
        if not _mass_branch_equal(
            final_trained.parameters,
            final_mass.parameters,
        ):
            raise ValueError(
                "phase-selector final phase stage changed frozen mass branch"
            )
        final_audit_path = artifacts / "all-train-selector-audit.safetensors"
        final_audit_descriptor = _write_audit_artifact(
            final_audit_path,
            final_trained.parameters,
            shape,
        )
        final_fp32 = _load_audit_artifact(
            final_audit_path,
            final_audit_descriptor,
            shape,
            bf16=False,
        )
        final_deployment_path = artifacts / "all-train-selector-bf16.safetensors"
        final_deployment_descriptor = _write_deployment_artifact(
            final_deployment_path,
            final_trained.parameters,
            shape,
        )
        final_bf16 = _load_deployment_artifact(
            final_deployment_path,
            final_deployment_descriptor,
            shape,
        )
        final_fp32_coefficients, _ = selector.selector_forward(
            native_mass,
            valid,
            phase,
            active,
            final_fp32,
            shape,
        )
        final_bf16_coefficients, final_bf16_delta = selector.selector_forward(
            native_mass,
            valid,
            phase,
            active,
            final_bf16,
            shape,
        )
        final_bf16_replay, _ = selector.selector_forward(
            native_mass,
            valid,
            phase,
            active,
            final_bf16,
            shape,
        )
        final_scores = np.zeros(native_mass.shape, dtype=np.float32)
        final_scores[valid] = np.log(native_mass[valid])
        final_score_coefficients, final_score_delta = (
            selector.selector_forward_from_scores(
                final_scores,
                valid,
                phase,
                active,
                final_bf16,
                shape,
            )
        )
        final_fp32_error, final_target = selector.direct_post_wo_error_energy(
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
        if not np.array_equal(final_target, final_target_replay):
            raise ValueError("phase-selector final target replay changed")
        final_fp32_metrics = _arm_metrics(final_target, final_fp32_error)
        final_bf16_metrics = _arm_metrics(final_target, final_bf16_error)
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
            "mass_branch_frozen_during_phase_fit": _mass_branch_equal(
                final_fp32,
                final_mass.parameters,
            ),
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
            "finite": all(
                np.isfinite(value).all()
                for value in (
                    final_fp32_coefficients,
                    final_bf16_coefficients,
                )
            ),
            "simplex_and_invalid_mask": all(
                (
                    np.max(
                        np.abs(
                            np.sum(value, axis=-1, dtype=np.float32) - np.float32(1.0)
                        )
                    )
                    <= 2.0e-6
                    and np.all(value >= 0.0)
                    and np.all(value[~valid] == 0.0)
                )
                for value in (
                    final_fp32_coefficients,
                    final_bf16_coefficients,
                )
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
            "sequential_training": {
                "mass_stage": {
                    "initial_loss": final_mass.initial_loss,
                    "final_loss": final_mass.final_loss,
                    "steps": final_mass.steps,
                    "device": final_mass.device,
                    "learning_rate_sha256": final_mass.learning_rate_sha256,
                    "schedule_sha256": final_mass.schedule_sha256,
                },
                "phase_T_only_stage": {
                    "initial_loss": final_trained.initial_loss,
                    "final_loss": final_trained.final_loss,
                    "steps": final_trained.steps,
                    "device": final_trained.device,
                    "learning_rate_sha256": final_trained.learning_rate_sha256,
                    "schedule_sha256": final_trained.schedule_sha256,
                    "mass_branch_preserved_exact": True,
                },
            },
            "sanity": {
                "FP32_metrics": final_fp32_metrics,
                "BF16_metrics": final_bf16_metrics,
                "checks": final_checks,
            },
            "not_used_for_out_of_fold_gate": True,
        }

    native_implementation_authorized = bool(
        gate["passed"] and final_artifact_sanity_passed
    )
    status = (
        _PASSED_STATUS
        if native_implementation_authorized
        else (_FINAL_ARTIFACT_FAILED_STATUS if gate["passed"] else _FAILED_STATUS)
    )
    result = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _RESULT_EXPERIMENT,
        "status": status,
        "protocol": _binding(protocol_path, protocol_sha256),
        "record_authentication": record_authentication,
        "cross_manifest_record_join": dict(predecessor_context["record_join"]),
        "phase_schedule": {
            "positions_sha256": _array_sha256(positions),
            "phase_sha256": _array_sha256(phase),
            "active_sha256": _array_sha256(active),
            "schedule_shift_positions": _PHASES,
            "schedule_shift_phase_and_active_exact": (schedule_shift_phase_exact),
        },
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
            "mass_vs_reconstructed_score_coefficient_max_abs": (
                score_coefficient_max_abs
            ),
            "mass_vs_reconstructed_score_delta_max_abs": score_delta_max_abs,
            "zero_model_native_coefficient_max_abs": (zero_coefficient_max_abs),
            "zero_model_native_post_Wo_output_max_abs": zero_output_max_abs,
            "deterministic_serialized_replay_exact": deterministic_replay,
            "schedule_shift_coefficient_max_abs": (schedule_shift_coefficient_max_abs),
            "inactive_phase_table_disable_coefficient_max_abs": (
                inactive_disable_coefficient_max_abs
            ),
            "native_raw_score_parity_evaluated": False,
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
            },
            "all_train_selector": final_artifact,
        },
        "resource_contract": dict(frozen["resource_contract"]),
        "decision": {
            "train_phase_selector_oof_gate_passed": gate["passed"],
            "final_BF16_artifact_sanity_passed": final_artifact_sanity_passed,
            "train_only_native_integration_implementation_authorized": (
                native_implementation_authorized
            ),
            "native_runtime_execution_authorized": False,
            "development_authorized": False,
            "confirmation_authorized": False,
            "semantic_or_M3_gate_passed": False,
            "independent_generalization_claim": False,
            "next_step": (
                "implement the fused native phase selector and pass train-only "
                "score/coefficient/post-Wo parity before a separately frozen "
                "causal development rollout"
                if native_implementation_authorized
                else "reject this phase-conditioned mass-selector class"
            ),
        },
        "post_fit_authentication": {
            "protocol": sha256_file(protocol_path) == protocol_sha256,
            "content_protocol": (
                sha256_file(context["content_protocol_path"])
                == _EXPECTED_CONTENT_PROTOCOL_SHA256
            ),
            "content_result": (
                sha256_file(context["content_result_path"])
                == _EXPECTED_CONTENT_RESULT_SHA256
            ),
            "mass_protocol": (
                sha256_file(predecessor_context["mass_protocol_path"])
                == _EXPECTED_MASS_PROTOCOL_SHA256
            ),
            "mass_result": (
                sha256_file(predecessor_context["mass_result_path"])
                == _EXPECTED_MASS_RESULT_SHA256
            ),
            "mass_fold_artifacts": all(
                sha256_file(row["path"]) == row["descriptor"]["file_sha256"]
                for row in context["mass_fold_artifacts"]
            ),
            "full_visible_protocol": (
                sha256_file(capacity["capacity_protocol_path"])
                == capacity["capacity_protocol_sha256"]
            ),
            "full_visible_result": (
                sha256_file(capacity["capacity_result_path"])
                == capacity["capacity_result_sha256"]
            ),
            "full_visible_manifest": (
                sha256_file(capacity["trace_manifest_path"])
                == capacity["trace_manifest_sha256"]
            ),
            "residual_manifest": (
                sha256_file(predecessor_context["residual_manifest_path"])
                == _EXPECTED_RESIDUAL_MANIFEST_SHA256
            ),
            "package_manifest": (
                sha256_file(predecessor_context["package"]["manifest_path"])
                == _EXPECTED_PACKAGE_MANIFEST_SHA256
            ),
            "non_mlp_weights": (
                sha256_file(predecessor_context["package"]["non_mlp_path"])
                == _EXPECTED_NON_MLP_SHA256
            ),
            "source_inventory": frozen["source_sha256"] == _source_inventory(),
            "record_authentication": all(record_authentication.values()),
            "cross_manifest_record_join": all(
                predecessor_context["record_join"].values()
            ),
            "confirmation_not_opened": True,
        },
        "authenticated_confirmation_descriptor": dict(
            frozen["authenticated_confirmation_descriptor"]
        ),
        "confirmation_split_opened": False,
    }
    if not all(result["post_fit_authentication"].values()):
        raise ValueError("phase-selector post-fit authentication failed")
    mass_runner._atomic_json_new(output, result)
    _progress(
        f"completed with status {status}; "
        f"BF16 global recovery={bf16_metrics['global']['recovery']:.6f}"
    )
    return {"path": str(output), "sha256": sha256_file(output), "result": result}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze or execute the train-only C28 phase selector",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--content-protocol", required=True)
    freeze.add_argument("--content-protocol-sha256", required=True)
    freeze.add_argument("--content-result", required=True)
    freeze.add_argument("--content-result-sha256", required=True)
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
        outcome = freeze_phase_selector_protocol(
            content_protocol=args.content_protocol,
            content_protocol_sha256=args.content_protocol_sha256,
            content_result=args.content_result,
            content_result_sha256=args.content_result_sha256,
            out=args.out,
        )
    else:
        outcome = fit_screen_phase_selector(
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            artifact_directory=args.artifact_directory,
            out=args.out,
            device=args.device,
        )
    print(json.dumps({"path": outcome["path"], "sha256": outcome["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
