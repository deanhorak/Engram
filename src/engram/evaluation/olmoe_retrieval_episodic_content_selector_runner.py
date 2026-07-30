"""Freeze and execute the train-only C28 content-selector OOF screen.

This is a fail-closed handoff from the negative mass-only selector.  It joins
the authenticated full-visible value trace to the older authenticated
``input_norm`` trace, reconstructs only causal pre-RoPE query features from
the packaged BF16 Q projection, and fits fixed record-disjoint folds.

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

import engram.evaluation.olmoe_retrieval_episodic_content_selector as selector
import engram.evaluation.olmoe_retrieval_episodic_full_visible_simplex_oracle as full
import engram.evaluation.olmoe_retrieval_episodic_mass_selector as mass
import engram.evaluation.olmoe_retrieval_episodic_mass_selector_runner as predecessor
import engram.evaluation.olmoe_retrieval_episodic_query_features as query_features
import engram.evaluation.olmoe_retrieval_episodic_residual_capacity as residual
from engram.utils import sha256_file


_SCHEMA_VERSION = 1
_PROTOCOL_EXPERIMENT = "olmoe_q7_retrieval_episodic_content_selector_protocol"
_PROTOCOL_STATUS = "frozen_before_train_content_selector_fit"
_RESULT_EXPERIMENT = (
    "olmoe_q7_retrieval_episodic_content_selector_oof_train_screen"
)
_PASSED_STATUS = "train_content_selector_oof_gate_passed"
_FAILED_STATUS = "train_content_selector_oof_gate_failed"
_FINAL_ARTIFACT_FAILED_STATUS = (
    "train_content_selector_oof_gate_passed_but_final_artifact_failed"
)

_EXPECTED_MASS_PROTOCOL_SHA256 = (
    "fa8202fac033d15fa96949c80e73b01b55eff454128fdbeaef276fd83111abfa"
)
_EXPECTED_MASS_RESULT_SHA256 = (
    "05a6511e38b1a9154981e102a20ce6953e26bb541cf07320d173ab2190689e6c"
)
_EXPECTED_RESIDUAL_PROTOCOL_SHA256 = (
    "584302d17a3224cda1b61dfe1f62685497fa5a0dc335cfc0a074439456ee1606"
)
_EXPECTED_RESIDUAL_RESULT_SHA256 = (
    "c636ad124d570f3675a36f0a23b276ba2e4cd4f5efc23dbf98cc10cd2cfd8e33"
)
_EXPECTED_RESIDUAL_PARITY_SHA256 = (
    "56e4b730dc7580895e952a5746d105f5ca01ec36d83f6b37044c5f331061f8dd"
)
_EXPECTED_RESIDUAL_MANIFEST_SHA256 = (
    "1f255a59a20089abe4d6805c625a119c167b71153bc21f5edbfcf0fd8050f461"
)
_EXPECTED_HEAD_MASS_PROTOCOL_SHA256 = (
    "fe09689452e6ae4f1a1b15332c61c1cc990cfc29b6a8b0d5a1758d9490a93af5"
)
_EXPECTED_PACKAGE_MANIFEST_SHA256 = (
    "861e9cc472f9e1245db5d64e9253411d0b656a0f08df2f58264e9c708ed750db"
)
_EXPECTED_NON_MLP_SHA256 = (
    "93797e149ce7ecabc4fc8833f3ca11bb9e45839501b09b2ffed4940e798044d4"
)

_CORE_SOURCE = (
    "src/engram/evaluation/olmoe_retrieval_episodic_content_selector.py"
)
_QUERY_SOURCE = (
    "src/engram/evaluation/olmoe_retrieval_episodic_query_features.py"
)
_RUNNER_SOURCE = (
    "src/engram/evaluation/olmoe_retrieval_episodic_content_selector_runner.py"
)
_SOURCE_FILES = (
    _CORE_SOURCE,
    _QUERY_SOURCE,
    _RUNNER_SOURCE,
    predecessor._CORE_SOURCE,
    predecessor._RUNNER_SOURCE,
    "src/engram/evaluation/olmoe_retrieval_episodic_residual_capacity.py",
)

_RECORDS = 8
_READS = 32
_LAYERS = 16
_HEADS = 16
_COMPONENTS = 28
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

_FOLDS = (
    (0, (0, 4)),
    (1, (1, 5)),
    (2, (2, 6)),
    (3, (3, 7)),
)


def _progress(message: str) -> None:
    print(f"[content-selector] {message}", file=sys.stderr, flush=True)


def _binding(path: Path, digest: str) -> dict[str, str]:
    return {"path": str(path), "sha256": digest}


def _source_inventory() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    return {name: sha256_file(repository / name) for name in _SOURCE_FILES}


def _checked_exact(
    value: str | Path,
    digest: str,
    expected: str,
    label: str,
) -> Path:
    return predecessor._checked_file(
        value,
        digest,
        expected_digest=expected,
        label=label,
    )


def _validate_mass_failure(
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
        or value.get("experiment") != predecessor._RESULT_EXPERIMENT
        or value.get("status") != predecessor._FAILED_STATUS
        or value.get("protocol")
        != _binding(protocol_path, _EXPECTED_MASS_PROTOCOL_SHA256)
        or value.get("confirmation_split_opened") is not False
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not False
        or gate.get("FP32_gate_passed") is not False
        or gate.get("BF16_gate_passed") is not False
        or not isinstance(decision, Mapping)
        or decision.get("train_mass_selector_oof_gate_passed") is not False
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
        raise ValueError("content-selector predecessor failure changed")
    fp32 = arms.get("FP32", {}).get("metrics", {})
    bf16 = arms.get("BF16_RNE", {}).get("metrics", {})
    expected = {
        "FP32_global_recovery": 0.2578308473082854,
        "BF16_global_recovery": 0.2578029677758369,
        "FP32_minimum_sequence_recovery": 0.24096330836282664,
        "BF16_minimum_sequence_recovery": 0.2408850711935605,
        "FP32_minimum_block_recovery": 0.21989300555077917,
        "BF16_minimum_block_recovery": 0.21989179323756192,
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
        raise ValueError("content-selector predecessor metrics changed")
    return expected


def _authenticate_package(
    manifest: str | Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    path = _checked_exact(
        manifest,
        manifest_sha256,
        _EXPECTED_PACKAGE_MANIFEST_SHA256,
        "content-selector package manifest",
    )
    value = predecessor._read_json(path, "content-selector package manifest")
    model = value.get("model")
    files = value.get("files")
    transformer = value.get("transformer")
    if (
        value.get("format") != "engram-native-olmoe-q7"
        or not isinstance(model, Mapping)
        or model.get("hidden_size") != _HIDDEN_SIZE
        or model.get("num_hidden_layers") != _LAYERS
        or model.get("num_attention_heads") != _HEADS
        or model.get("num_key_value_heads") != _HEADS
        or model.get("rms_norm_eps") != 1.0e-5
        or model.get("rope_theta") != 10_000.0
        or not isinstance(files, Mapping)
        or not isinstance(transformer, Mapping)
        or transformer.get("path") != "transformer/non_mlp.safetensors"
        or transformer.get("sha256") != _EXPECTED_NON_MLP_SHA256
        or files.get("transformer/non_mlp.safetensors", {}).get("sha256")
        != _EXPECTED_NON_MLP_SHA256
    ):
        raise ValueError("content-selector package contract changed")
    non_mlp = path.parent / "transformer/non_mlp.safetensors"
    predecessor._guard_paths((("content-selector non-MLP weights", non_mlp),))
    if (
        non_mlp.is_symlink()
        or not non_mlp.is_file()
        or sha256_file(non_mlp) != _EXPECTED_NON_MLP_SHA256
    ):
        raise ValueError("content-selector non-MLP weights changed")
    weight_contract = query_features.query_weight_contract(
        non_mlp,
        non_mlp_sha256=_EXPECTED_NON_MLP_SHA256,
        layers=_LAYERS,
        hidden_size=_HIDDEN_SIZE,
    )
    return {
        "manifest_path": path,
        "manifest": value,
        "non_mlp_path": non_mlp,
        "query_weight_contract": weight_contract,
    }


def _load_residual_inputs(
    manifest_path: Path,
    manifest_value: Mapping[str, Any],
    full_manifest: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, bool]]:
    residual_shards = manifest_value.get("shards")
    full_shards = full_manifest.get("shards")
    if (
        manifest_value.get("schema_version") != _SCHEMA_VERSION
        or manifest_value.get("experiment") != residual._RESULT_EXPERIMENT
        or manifest_value.get("format") != "safetensors"
        or manifest_value.get("protocol", {}).get("sha256")
        != _EXPECTED_RESIDUAL_PROTOCOL_SHA256
        or manifest_value.get("record_order") != list(range(_RECORDS))
        or manifest_value.get("confirmation_split_opened") is not False
        or not isinstance(residual_shards, list)
        or not isinstance(full_shards, list)
        or len(residual_shards) != _RECORDS
        or len(full_shards) != _RECORDS
    ):
        raise ValueError("content-selector residual trace manifest changed")
    inputs: list[np.ndarray] = []
    checks: dict[str, bool] = {}
    for index, (residual_row, full_row) in enumerate(
        zip(residual_shards, full_shards, strict=True)
    ):
        if not isinstance(residual_row, Mapping) or not isinstance(
            full_row,
            Mapping,
        ):
            raise ValueError("content-selector trace descriptor changed")
        joined = (
            residual_row.get("record_index") == index
            and full_row.get("record_index") == index
            and residual_row.get("record_id") == full_row.get("record_id")
            and residual_row.get("source_record_sha256")
            == full_row.get("source_record_sha256")
            and residual_row.get("output_evidence_sha256")
            == full_row.get("output_evidence_sha256")
            and residual_row.get("reset_output_evidence_sha256")
            == full_row.get("reset_output_evidence_sha256")
            and residual_row.get("positions") == full_row.get("query_positions")
            and residual_row.get("tensor_sha256", {}).get("base_projected")
            == full_row.get("tensor_sha256", {}).get("base_projected")
            and residual_row.get("tensor_sha256", {}).get("target_residual")
            == full_row.get("tensor_sha256", {}).get("target_residual")
        )
        if not joined:
            raise ValueError("content-selector cross-manifest record join changed")
        loaded = residual._validate_trace_shard(
            manifest_path.parent / str(residual_row.get("file", "")),
            residual_row,
        )
        full_loaded = full.validate_full_visible_trace_shard(
            Path(full_manifest["_manifest_path"]).parent
            / str(full_row.get("file", "")),
            full_row,
        )
        actual_equal = np.array_equal(
            loaded["base_projected"],
            full_loaded["base_projected"],
        ) and np.array_equal(
            loaded["target_residual"],
            full_loaded["target_residual"],
        )
        if not actual_equal:
            raise ValueError("content-selector joined trace tensors differ")
        input_norm = np.ascontiguousarray(loaded["input_norm"], dtype=np.float32)
        if input_norm.shape != (_READS, _LAYERS, _HIDDEN_SIZE) or not np.isfinite(
            input_norm
        ).all():
            raise ValueError("content-selector input_norm trace changed")
        inputs.append(input_norm)
        checks[f"record_{index:02d}"] = joined and actual_equal
    return np.ascontiguousarray(np.stack(inputs)), checks


def _authenticate_residual_manifest(
    *,
    manifest: str | Path,
    manifest_sha256: str,
    capacity_protocol: Mapping[str, Any],
    full_manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], np.ndarray, dict[str, bool]]:
    historical = capacity_protocol.get("historical_bindings")
    if not isinstance(historical, Mapping):
        raise ValueError("content-selector capacity history changed")
    head_binding = historical.get("inherited_head_mass_protocol")
    if (
        not isinstance(head_binding, Mapping)
        or head_binding.get("sha256") != _EXPECTED_HEAD_MASS_PROTOCOL_SHA256
    ):
        raise ValueError("content-selector head-mass binding changed")
    head_path = _checked_exact(
        str(head_binding.get("path")),
        str(head_binding.get("sha256")),
        _EXPECTED_HEAD_MASS_PROTOCOL_SHA256,
        "content-selector inherited head-mass protocol",
    )
    head_protocol = predecessor._read_json(
        head_path,
        "content-selector inherited head-mass protocol",
    )
    expected_binding = {
        "path": str(Path(manifest).expanduser().resolve()),
        "sha256": manifest_sha256,
    }
    if (
        head_protocol.get("capacity_trace_manifest") != expected_binding
        or head_protocol.get("confirmation_split_opened") is not False
    ):
        raise ValueError("content-selector residual lineage changed")
    path = _checked_exact(
        manifest,
        manifest_sha256,
        _EXPECTED_RESIDUAL_MANIFEST_SHA256,
        "content-selector residual trace manifest",
    )
    value = predecessor._read_json(
        path,
        "content-selector residual trace manifest",
    )
    full_with_path = dict(full_manifest)
    input_norm, checks = _load_residual_inputs(
        path,
        value,
        full_with_path,
    )
    return path, value, input_norm, checks


def _authenticate_inputs(
    *,
    mass_protocol: str | Path,
    mass_protocol_sha256: str,
    mass_result: str | Path,
    mass_result_sha256: str,
    residual_manifest: str | Path,
    residual_manifest_sha256: str,
    package_manifest: str | Path,
    package_manifest_sha256: str,
) -> dict[str, Any]:
    predecessor._guard_paths(
        (
            ("content-selector mass protocol", mass_protocol),
            ("content-selector mass result", mass_result),
            ("content-selector residual manifest", residual_manifest),
            ("content-selector package manifest", package_manifest),
        )
    )
    if mass_protocol_sha256 != _EXPECTED_MASS_PROTOCOL_SHA256:
        raise ValueError("content-selector mass protocol root changed")
    protocol_path, frozen, capacity = predecessor._authenticate_selector_protocol(
        mass_protocol,
        mass_protocol_sha256,
    )
    result_path = _checked_exact(
        mass_result,
        mass_result_sha256,
        _EXPECTED_MASS_RESULT_SHA256,
        "content-selector mass result",
    )
    result = predecessor._read_json(result_path, "content-selector mass result")
    failure = _validate_mass_failure(result, protocol_path=protocol_path)
    package = _authenticate_package(package_manifest, package_manifest_sha256)
    full_manifest = dict(capacity["trace_manifest"])
    full_manifest["_manifest_path"] = str(capacity["trace_manifest_path"])
    residual_path, residual_value, input_norm, record_join = (
        _authenticate_residual_manifest(
            manifest=residual_manifest,
            manifest_sha256=residual_manifest_sha256,
            capacity_protocol=capacity["capacity_protocol"],
            full_manifest=full_manifest,
        )
    )
    return {
        "mass_protocol_path": protocol_path,
        "mass_protocol": frozen,
        "mass_result_path": result_path,
        "mass_result": result,
        "mass_failure": failure,
        "capacity": capacity,
        "residual_manifest_path": residual_path,
        "residual_manifest": residual_value,
        "input_norm": input_norm,
        "record_join": record_join,
        "package": package,
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
        "weight_decay": {"U": 1.0e-4, "V": 1.0e-4, "Q": 1.0e-4, "P": 1.0e-4},
        "zero_weight_decay": ["E", "B"],
        "global_gradient_clip_norm": 1.0,
        "initialization": {
            "U": "PCG64 normal(0,0.02)",
            "Q": "same PCG64 stream normal(0,0.02)",
            "V": "zero",
            "P": "zero",
            "E": "zero",
            "B": "zero",
            "exact_native_zero_model": True,
        },
        "training_precision": "FP32",
        "training_device": "cuda",
        "CUDA_deterministic_algorithms_required": True,
        "TF32_allowed": False,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "checkpoint_selection": "fixed final step only",
        "final_all_train_fit_on_out_of_fold_pass": {
            "training_record_indices": list(range(_RECORDS)),
            "steps": _FINAL_TRAINING_STEPS,
            "warmup_steps": _FINAL_WARMUP_STEPS,
            "epochs": _EPOCHS,
            "initialization_seed": _INIT_SEED_BASE + len(_FOLDS),
            "shuffle_seed": _SHUFFLE_SEED_BASE + len(_FOLDS),
            "not_used_for_out_of_fold_gate": True,
        },
    }


def _model_contract() -> dict[str, Any]:
    shape = selector.ContentSelectorShape()
    return {
        "layers": shape.layers,
        "query_heads": shape.heads,
        "components_per_head": shape.components,
        "head_dimension": shape.head_dimension,
        "mass_rank": shape.mass_rank,
        "content_rank": shape.content_rank,
        "parameter_shapes": {
            "U": [shape.layers, shape.components, shape.mass_rank],
            "V": [shape.layers, shape.mass_rank, shape.components],
            "E": [shape.layers, shape.heads, shape.mass_rank],
            "B": [shape.layers, shape.heads, shape.components],
            "Q": [
                shape.layers,
                shape.heads,
                shape.head_dimension,
                shape.content_rank,
            ],
            "P": [
                shape.layers,
                shape.heads,
                shape.head_dimension,
                shape.content_rank,
            ],
        },
        "causal_query_feature": (
            "packaged BF16 q_proj(input_norm), followed by packaged BF16 "
            "flattened q_norm, captured before RoPE"
        ),
        "value_feature": (
            "per-head rank-4 BF16-RNE sidecar computed once from the exact "
            "stored value representation"
        ),
        "mass_feature": "clipped centered log of authenticated native C28 mass",
        "cached_forward": (
            "delta=gauge_clamp(mass_MLP(log_mass)+"
            "(query@Q) dot BF16_RNE(value@P)/sqrt(4)); "
            "coefficients=normalize(native_mass*exp(delta))"
        ),
        "candidate_native_forward": (
            "replace reconstructed log_mass by centered raw q.k scores and "
            "masked_softmax(raw_scores+delta)"
        ),
        "loss": "direct squared post-Wo residual error",
        "target_residual_is_label_only": True,
        "counterfactual_updates_hidden_or_cache_during_screen": False,
        "single_full_value_accumulation_pass_required": True,
        "native_raw_score_query_and_sidecar_parity_required_before_rollout": True,
    }


def _gate_contract() -> dict[str, Any]:
    return {
        "arms_required": ["FP32", "BF16_RNE_parameters_and_sidecars"],
        "minimum_global_recovery": 0.50,
        "minimum_every_out_of_fold_sequence_recovery": 0.25,
        "minimum_every_block_entry_position_recovery": 0.25,
        "block_entry_positions": list(_BLOCK_POSITIONS),
        "minimum_positive_recovery_layers": 12,
        "maximum_BF16_global_recovery_drop_from_FP32": 0.005,
        "finite_simplex_mask_and_zero_model_parity_required": True,
        "mass_vs_reconstructed_score_reference_tolerance": 1.0e-6,
        "deterministic_serialized_replay_exact": True,
        "resource_below_exact_51_head_ceiling_required": True,
        "final_all_train_BF16_artifact_sanity_required_for_native_integration": True,
    }


def _build_protocol(context: Mapping[str, Any]) -> dict[str, Any]:
    capacity = context["capacity"]
    package = context["package"]
    return {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _PROTOCOL_EXPERIMENT,
        "status": _PROTOCOL_STATUS,
        "predecessor_mass_protocol": _binding(
            context["mass_protocol_path"],
            _EXPECTED_MASS_PROTOCOL_SHA256,
        ),
        "predecessor_mass_result": {
            **_binding(
                context["mass_result_path"],
                _EXPECTED_MASS_RESULT_SHA256,
            ),
            "authenticated_failure_metrics": dict(context["mass_failure"]),
        },
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
            context["residual_manifest_path"],
            _EXPECTED_RESIDUAL_MANIFEST_SHA256,
        ),
        "residual_lineage": {
            "protocol_sha256": _EXPECTED_RESIDUAL_PROTOCOL_SHA256,
            "result_sha256": _EXPECTED_RESIDUAL_RESULT_SHA256,
            "parity_sha256": _EXPECTED_RESIDUAL_PARITY_SHA256,
            "head_mass_protocol_sha256": _EXPECTED_HEAD_MASS_PROTOCOL_SHA256,
            "record_join": dict(context["record_join"]),
        },
        "package_manifest": _binding(
            package["manifest_path"],
            _EXPECTED_PACKAGE_MANIFEST_SHA256,
        ),
        "non_mlp_weights": {
            **_binding(package["non_mlp_path"], _EXPECTED_NON_MLP_SHA256),
            "query_weight_contract": package["query_weight_contract"],
        },
        "selector_model": _model_contract(),
        "training": _training_contract(),
        "serialization_and_parity": {
            "production_parameter_dtype": "BF16",
            "BF16_rounding": "round_to_nearest_even",
            "persistent_value_sidecar_dtype": "BF16",
            "FP32_and_BF16_out_of_fold_evaluation_required": True,
            "pre_RoPE_query_reconstruction_hash_required": True,
            "native_query_parity_required_before_causal_rollout": True,
            "native_raw_score_parity_required_before_causal_rollout": True,
            "final_native_artifact_contains_BF16_only": True,
        },
        "progression_gate": _gate_contract(),
        "resource_contract": selector.production_resource_contract(),
        "scope": {
            "split": "train",
            "records": _RECORDS,
            "read_positions_per_record": _READS,
            "out_of_fold_fit": True,
            "causal_features_only": True,
            "cached_same_state_screen": True,
            "native_causal_rollout": False,
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
                "reject this rank-4 content-selector class without tuning on "
                "the OOF records"
            ),
            "does_not_authorize_development": True,
            "does_not_authorize_confirmation": True,
            "does_not_pass_semantic_or_M3_gate": True,
        },
        "source_sha256": _source_inventory(),
        "authenticated_confirmation_descriptor": dict(
            capacity["capacity_result"]["authenticated_confirmation_descriptor"]
        ),
        "confirmation_split_opened": False,
    }


def freeze_content_selector_protocol(
    *,
    mass_protocol: str | Path,
    mass_protocol_sha256: str,
    mass_result: str | Path,
    mass_result_sha256: str,
    residual_manifest: str | Path,
    residual_manifest_sha256: str,
    package_manifest: str | Path,
    package_manifest_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    predecessor._guard_paths(
        (
            ("content-selector mass protocol", mass_protocol),
            ("content-selector mass result", mass_result),
            ("content-selector residual manifest", residual_manifest),
            ("content-selector package manifest", package_manifest),
            ("content-selector protocol output", out),
        )
    )
    context = _authenticate_inputs(
        mass_protocol=mass_protocol,
        mass_protocol_sha256=mass_protocol_sha256,
        mass_result=mass_result,
        mass_result_sha256=mass_result_sha256,
        residual_manifest=residual_manifest,
        residual_manifest_sha256=residual_manifest_sha256,
        package_manifest=package_manifest,
        package_manifest_sha256=package_manifest_sha256,
    )
    protocol = _build_protocol(context)
    output = predecessor._new_output(out, "content-selector protocol output")
    predecessor._atomic_json_new(output, protocol)
    if predecessor._read_json(output, "content-selector protocol replay") != protocol:
        raise AssertionError("content-selector protocol replay changed")
    return {"path": str(output), "sha256": sha256_file(output), "protocol": protocol}


def _authenticate_frozen_protocol(
    protocol: str | Path,
    protocol_sha256: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    predecessor._guard_paths((("content-selector protocol", protocol),))
    requested = Path(protocol).expanduser()
    if (
        requested.is_symlink()
        or not requested.is_file()
        or not predecessor._is_sha256(protocol_sha256)
        or sha256_file(requested) != protocol_sha256
    ):
        raise ValueError("content-selector protocol authentication failed")
    path = requested.resolve()
    frozen = predecessor._read_json(path, "content-selector protocol")
    bindings = (
        frozen.get("predecessor_mass_protocol"),
        frozen.get("predecessor_mass_result"),
        frozen.get("residual_input_trace_manifest"),
        frozen.get("package_manifest"),
    )
    if not all(isinstance(value, Mapping) for value in bindings):
        raise ValueError("content-selector protocol bindings changed")
    mass_protocol, mass_result, residual_manifest, package_manifest = bindings
    context = _authenticate_inputs(
        mass_protocol=str(mass_protocol.get("path")),
        mass_protocol_sha256=str(mass_protocol.get("sha256")),
        mass_result=str(mass_result.get("path")),
        mass_result_sha256=str(mass_result.get("sha256")),
        residual_manifest=str(residual_manifest.get("path")),
        residual_manifest_sha256=str(residual_manifest.get("sha256")),
        package_manifest=str(package_manifest.get("path")),
        package_manifest_sha256=str(package_manifest.get("sha256")),
    )
    expected = _build_protocol(context)
    if frozen != expected:
        raise ValueError("content-selector frozen protocol changed")
    return path, frozen, context


def _training_config(
    *,
    init_seed: int,
    shuffle_seed: int,
    final_fit: bool = False,
) -> mass.TrainingConfig:
    return mass.TrainingConfig(
        steps=_FINAL_TRAINING_STEPS if final_fit else _TRAINING_STEPS,
        warmup_steps=_FINAL_WARMUP_STEPS if final_fit else _WARMUP_STEPS,
        peak_learning_rate=_PEAK_LEARNING_RATE,
        final_learning_rate=_FINAL_LEARNING_RATE,
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-8,
        uv_weight_decay=1.0e-4,
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
    return predecessor._array_sha256(value)


def _parameters_from_tensors(
    tensors: Mapping[str, np.ndarray],
    *,
    suffix: str,
    shape: selector.ContentSelectorShape,
) -> selector.ContentSelectorParameters:
    values: dict[str, np.ndarray] = {}
    for name in ("U", "V", "E", "B", "Q", "P"):
        tensor = tensors[f"{name}_{suffix}"]
        values[name] = (
            mass.bf16_bits_to_float32(tensor)
            if suffix == "bf16_bits"
            else np.ascontiguousarray(tensor, dtype=np.float32)
        )
    parameters = selector.ContentSelectorParameters(**values)
    parameters.validate(shape)
    return parameters


def _write_audit_artifact(
    path: Path,
    parameters: selector.ContentSelectorParameters,
    shape: selector.ContentSelectorShape,
) -> dict[str, Any]:
    parameters.validate(shape)
    decoded, bits = selector.quantize_parameters_bf16(parameters, shape)
    tensors: dict[str, np.ndarray] = {}
    for name, value in parameters.as_dict().items():
        tensors[f"{name}_fp32"] = value
        tensors[f"{name}_bf16_bits"] = bits[name]
    predecessor._atomic_safetensors(path, tensors)
    loaded = load_safetensors(str(path))
    if set(loaded) != set(tensors) or any(
        not np.array_equal(loaded[name], expected)
        for name, expected in tensors.items()
    ):
        raise ValueError("content-selector audit artifact replay changed")
    fp32 = _parameters_from_tensors(loaded, suffix="fp32", shape=shape)
    bf16 = _parameters_from_tensors(loaded, suffix="bf16_bits", shape=shape)
    if (
        selector.parameters_sha256(fp32, shape)
        != selector.parameters_sha256(parameters, shape)
        or selector.parameters_sha256(bf16, shape)
        != selector.parameters_sha256(decoded, shape)
    ):
        raise ValueError("content-selector audit parameter replay changed")
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
    shape: selector.ContentSelectorShape,
    *,
    bf16: bool,
) -> selector.ContentSelectorParameters:
    if (
        path.name != descriptor.get("file")
        or sha256_file(path) != descriptor.get("file_sha256")
        or descriptor.get("artifact_role") != "training_audit"
        or descriptor.get("authorized_for_runtime_loading") is not False
    ):
        raise ValueError("content-selector audit artifact authentication failed")
    tensors = load_safetensors(str(path))
    parameters = _parameters_from_tensors(
        tensors,
        suffix="bf16_bits" if bf16 else "fp32",
        shape=shape,
    )
    expected = descriptor[
        "BF16_decoded_parameter_sha256" if bf16 else "FP32_parameter_sha256"
    ]
    if selector.parameters_sha256(parameters, shape) != expected:
        raise ValueError("content-selector audit parameter root changed")
    return parameters


def _write_deployment_artifact(
    path: Path,
    parameters: selector.ContentSelectorParameters,
    shape: selector.ContentSelectorShape,
) -> dict[str, Any]:
    parameters.validate(shape)
    decoded, bits = selector.quantize_parameters_bf16(parameters, shape)
    tensors = {f"{name}_bf16_bits": value for name, value in bits.items()}
    if sum(value.nbytes for value in tensors.values()) != shape.parameter_count * 2:
        raise AssertionError("content-selector deployment bytes changed")
    predecessor._atomic_safetensors(path, tensors)
    loaded = load_safetensors(str(path))
    if set(loaded) != set(tensors) or any(
        not np.array_equal(loaded[name], expected)
        for name, expected in tensors.items()
    ):
        raise ValueError("content-selector deployment replay changed")
    replay = _parameters_from_tensors(loaded, suffix="bf16_bits", shape=shape)
    if selector.parameters_sha256(replay, shape) != selector.parameters_sha256(
        decoded,
        shape,
    ):
        raise ValueError("content-selector deployment parameter replay changed")
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
    shape: selector.ContentSelectorShape,
) -> selector.ContentSelectorParameters:
    if (
        path.name != descriptor.get("file")
        or sha256_file(path) != descriptor.get("file_sha256")
        or descriptor.get("artifact_role") != "native_BF16_deployment"
        or descriptor.get("authorized_for_runtime_loading") is not True
        or descriptor.get("contains_FP32_training_copy") is not False
        or descriptor.get("BF16_tensor_bytes") != shape.parameter_count * 2
    ):
        raise ValueError("content-selector deployment authentication failed")
    parameters = _parameters_from_tensors(
        load_safetensors(str(path)),
        suffix="bf16_bits",
        shape=shape,
    )
    if (
        selector.parameters_sha256(parameters, shape)
        != descriptor["BF16_decoded_parameter_sha256"]
    ):
        raise ValueError("content-selector deployment parameter root changed")
    return parameters


def _arm_metrics(
    target_energy: np.ndarray,
    error_energy: np.ndarray,
) -> dict[str, Any]:
    return predecessor._arm_metrics(target_energy, error_energy)


def _maximum_output_correction(
    coefficients: np.ndarray,
    values: np.ndarray,
    base_heads: np.ndarray,
    projection: np.ndarray,
    shape: selector.ContentSelectorShape,
) -> float:
    return predecessor._maximum_output_correction(
        coefficients,
        values,
        base_heads,
        projection,
        shape,
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
) -> dict[str, bool]:
    finite = all(
        np.isfinite(value).all() for value in (fp32_coefficients, bf16_coefficients)
    )
    simplex = all(
        (
            np.max(np.abs(np.sum(value, axis=-1, dtype=np.float32) - 1.0))
            <= 2.0e-6
            and np.all(value >= 0.0)
            and np.all(value[~valid] == 0.0)
        )
        for value in (fp32_coefficients, bf16_coefficients)
    )
    fp32_global = float(fp32_metrics["global"]["recovery"])
    bf16_global = float(bf16_metrics["global"]["recovery"])
    resource = selector.production_resource_contract()
    checks = {
        "FP32_gate_passed": fp32_metrics.get("passed") is True,
        "BF16_gate_passed": bf16_metrics.get("passed") is True,
        "BF16_global_recovery_drop_within_0_005": (
            fp32_global - bf16_global <= 0.005
        ),
        "finite": finite,
        "simplex_and_invalid_mask": simplex,
        "mass_vs_reconstructed_score_coefficient_parity": (
            score_coefficient_max_abs <= 1.0e-6
        ),
        "mass_vs_reconstructed_score_delta_parity": score_delta_max_abs <= 1.0e-6,
        "zero_model_native_coefficient_parity": zero_coefficient_max_abs <= 1.0e-6,
        "zero_model_native_output_parity": zero_output_max_abs <= 1.0e-6,
        "deterministic_serialized_replay_exact": deterministic_replay_exact,
        "resource_below_exact_51_head_ceiling": (
            resource["total_logical_traffic_bytes_per_128_token_sequence"]
            < resource["exact_51_head_equivalent_ceiling_bytes"]
        ),
        "native_query_raw_score_and_sidecar_lifecycle_parity_deferred": True,
    }
    checks["passed"] = all(checks.values())
    return checks


def _query_positions(context: Mapping[str, Any]) -> np.ndarray:
    shards = context["residual_manifest"].get("shards")
    if not isinstance(shards, list) or len(shards) != _RECORDS:
        raise ValueError("content-selector residual position descriptors changed")
    positions = np.asarray(shards[0].get("positions"), dtype=np.int64)
    if (
        positions.shape != (_READS,)
        or positions.tolist() != list(range(96, 128))
        or any(row.get("positions") != positions.tolist() for row in shards)
    ):
        raise ValueError("content-selector read positions changed")
    return np.ascontiguousarray(positions)


def _derive_queries(
    context: Mapping[str, Any],
    *,
    device: str,
) -> query_features.QueryFeatureResult:
    input_norm = np.ascontiguousarray(context["input_norm"], dtype=np.float32)
    positions = _query_positions(context)
    weight_contract = context["package"]["query_weight_contract"]
    expected_hashes = weight_contract.get("tensor_sha256")
    if not isinstance(expected_hashes, Mapping):
        raise ValueError("content-selector query-weight contract changed")
    return query_features.reconstruct_authenticated_query_features(
        non_mlp_path=context["package"]["non_mlp_path"],
        non_mlp_sha256=_EXPECTED_NON_MLP_SHA256,
        input_norm=input_norm,
        input_norm_sha256=query_features.tensor_sha256(input_norm),
        positions=positions,
        positions_sha256=query_features.tensor_sha256(positions),
        device=device,
        expected_weight_tensor_sha256=expected_hashes,
    )


def fit_screen_content_selector(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    artifact_directory: str | Path,
    out: str | Path,
    device: str,
) -> dict[str, Any]:
    """Fit fixed OOF selectors and evaluate serialized BF16 sidecar artifacts."""

    predecessor._guard_paths(
        (
            ("content-selector protocol", protocol),
            ("content-selector artifact directory", artifact_directory),
            ("content-selector result output", out),
        )
    )
    protocol_path, frozen, context = _authenticate_frozen_protocol(
        protocol,
        protocol_sha256,
    )
    if device != frozen["training"]["training_device"]:
        raise ValueError("content-selector training device changed")
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != frozen["training"]["CUBLAS_WORKSPACE_CONFIG"]
    ):
        raise ValueError("content-selector CUDA workspace configuration changed")
    output = predecessor._new_output(out, "content-selector result output")
    artifacts = predecessor._new_directory(
        artifact_directory,
        "content-selector artifact directory",
    )

    _progress("loading and authenticating full-visible values")
    capacity = context["capacity"]
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
    shape = selector.ContentSelectorShape()
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
        native_mass.shape != expected_mass_shape
        or valid.shape != expected_mass_shape
        or values.shape != expected_mass_shape + (_HEAD_DIMENSION,)
        or base_heads.shape
        != (_RECORDS, _READS, _LAYERS, _HEADS, _HEAD_DIMENSION)
        or target.shape != (_RECORDS, _READS, _LAYERS, _HIDDEN_SIZE)
        or projection.shape != (_LAYERS, _HIDDEN_SIZE, _HIDDEN_SIZE)
    ):
        raise ValueError("content-selector authenticated tensor shape changed")

    _progress("reconstructing authenticated pre-RoPE query features on CUDA")
    query_result = _derive_queries(context, device=device)
    queries = np.array(
        query_result.queries,
        dtype=np.float32,
        order="C",
        copy=True,
    )
    if queries.shape != (
        _RECORDS,
        _READS,
        _LAYERS,
        _HEADS,
        _HEAD_DIMENSION,
    ):
        raise ValueError("content-selector reconstructed query shape changed")
    query_path = artifacts / "query-features.safetensors"
    query_tensors = {
        "post_qnorm_pre_rope_queries": queries,
        "positions": _query_positions(context),
    }
    predecessor._atomic_safetensors(query_path, query_tensors)
    query_replay = load_safetensors(str(query_path))
    if set(query_replay) != set(query_tensors) or any(
        not np.array_equal(query_replay[name], expected)
        for name, expected in query_tensors.items()
    ):
        raise ValueError("content-selector query artifact replay changed")

    fp32_oof = np.empty(expected_mass_shape, dtype=np.float32)
    bf16_oof = np.empty(expected_mass_shape, dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    deterministic_replay = True
    score_coefficient_max_abs = 0.0
    score_delta_max_abs = 0.0
    for fold in frozen["training"]["cross_validation"]["folds"]:
        fold_index = int(fold["fold_index"])
        training_records = tuple(
            int(value) for value in fold["training_record_indices"]
        )
        heldout = np.asarray(fold["heldout_record_indices"], dtype=np.int64)
        _progress(
            f"training fold {fold_index + 1}/4; "
            f"held out records {heldout.tolist()}"
        )
        trained = selector.fit_direct_post_wo(
            native_mass,
            valid,
            queries,
            values,
            base_heads,
            target,
            projection,
            training_records=training_records,
            shape=shape,
            config=_training_config(
                init_seed=int(fold["initialization_seed"]),
                shuffle_seed=int(fold["shuffle_seed"]),
            ),
            device=device,
        )
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
        fp32_coefficients, fp32_delta, _ = selector.selector_forward(
            native_mass[heldout],
            valid[heldout],
            queries[heldout],
            values[heldout],
            fp32_parameters,
            shape,
            quantize_sidecars=False,
        )
        bf16_coefficients, bf16_delta, bf16_sidecars = selector.selector_forward(
            native_mass[heldout],
            valid[heldout],
            queries[heldout],
            values[heldout],
            bf16_parameters,
            shape,
            quantize_sidecars=True,
        )
        fp32_replay, _, _ = selector.selector_forward(
            native_mass[heldout],
            valid[heldout],
            queries[heldout],
            values[heldout],
            fp32_parameters,
            shape,
            quantize_sidecars=False,
        )
        bf16_replay, _, bf16_sidecar_replay = selector.selector_forward(
            native_mass[heldout],
            valid[heldout],
            queries[heldout],
            values[heldout],
            bf16_parameters,
            shape,
            quantize_sidecars=True,
        )
        deterministic_replay = deterministic_replay and np.array_equal(
            fp32_coefficients,
            fp32_replay,
        )
        deterministic_replay = deterministic_replay and np.array_equal(
            bf16_coefficients,
            bf16_replay,
        )
        deterministic_replay = deterministic_replay and np.array_equal(
            bf16_sidecars,
            bf16_sidecar_replay,
        )
        fp32_oof[heldout] = fp32_coefficients
        bf16_oof[heldout] = bf16_coefficients
        for parameters, coefficients, delta, quantize_sidecars in (
            (fp32_parameters, fp32_coefficients, fp32_delta, False),
            (bf16_parameters, bf16_coefficients, bf16_delta, True),
        ):
            reconstructed_scores = np.zeros(
                native_mass[heldout].shape,
                dtype=np.float32,
            )
            heldout_valid = valid[heldout]
            reconstructed_scores[heldout_valid] = np.log(
                native_mass[heldout][heldout_valid]
            )
            score_coefficients, score_delta, _ = (
                selector.selector_forward_from_scores(
                    reconstructed_scores,
                    heldout_valid,
                    queries[heldout],
                    values[heldout],
                    parameters,
                    shape,
                    quantize_sidecars=quantize_sidecars,
                )
            )
            score_coefficient_max_abs = max(
                score_coefficient_max_abs,
                float(np.max(np.abs(coefficients - score_coefficients))),
            )
            score_delta_max_abs = max(
                score_delta_max_abs,
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
        standard_deviation=0.02,
    )
    zero_coefficients, _, _ = selector.selector_forward(
        native_mass,
        valid,
        queries,
        values,
        zero,
        shape,
        quantize_sidecars=True,
    )
    zero_coefficient_max_abs = float(
        np.max(np.abs(zero_coefficients - native_mass))
    )
    zero_output_max_abs = _maximum_output_correction(
        zero_coefficients,
        values,
        base_heads,
        projection,
        shape,
    )
    fp32_error, target_energy = mass.direct_post_wo_error_energy(
        fp32_oof,
        values,
        base_heads,
        target,
        projection,
        shape.mass_shape,
    )
    bf16_error, target_energy_replay = mass.direct_post_wo_error_energy(
        bf16_oof,
        values,
        base_heads,
        target,
        projection,
        shape.mass_shape,
    )
    if not np.array_equal(target_energy, target_energy_replay):
        raise ValueError("content-selector target energy replay changed")
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
    )
    oof_path = artifacts / "oof-coefficients.safetensors"
    oof_tensors = {
        "FP32_coefficients": np.ascontiguousarray(fp32_oof),
        "BF16_coefficients": np.ascontiguousarray(bf16_oof),
        "FP32_error_energy": np.ascontiguousarray(fp32_error, dtype=np.float64),
        "BF16_error_energy": np.ascontiguousarray(bf16_error, dtype=np.float64),
        "target_energy": np.ascontiguousarray(target_energy, dtype=np.float64),
    }
    predecessor._atomic_safetensors(oof_path, oof_tensors)
    oof_replay = load_safetensors(str(oof_path))
    if set(oof_replay) != set(oof_tensors) or any(
        not np.array_equal(oof_replay[name], expected)
        for name, expected in oof_tensors.items()
    ):
        raise ValueError("content-selector OOF artifact replay changed")

    final_artifact: dict[str, Any] | None = None
    final_artifact_sanity_passed = False
    if gate["passed"]:
        _progress("OOF gate passed; fitting the declared all-train artifact")
        final_contract = frozen["training"]["final_all_train_fit_on_out_of_fold_pass"]
        final_trained = selector.fit_direct_post_wo(
            native_mass,
            valid,
            queries,
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
        final_deployment_path = (
            artifacts / "all-train-selector-bf16.safetensors"
        )
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
        final_fp32_coefficients, _, _ = selector.selector_forward(
            native_mass,
            valid,
            queries,
            values,
            final_fp32,
            shape,
        )
        final_bf16_coefficients, final_bf16_delta, _ = selector.selector_forward(
            native_mass,
            valid,
            queries,
            values,
            final_bf16,
            shape,
            quantize_sidecars=True,
        )
        final_bf16_replay, _, _ = selector.selector_forward(
            native_mass,
            valid,
            queries,
            values,
            final_bf16,
            shape,
            quantize_sidecars=True,
        )
        final_scores = np.zeros(native_mass.shape, dtype=np.float32)
        final_scores[valid] = np.log(native_mass[valid])
        final_score_coefficients, final_score_delta, _ = (
            selector.selector_forward_from_scores(
                final_scores,
                valid,
                queries,
                values,
                final_bf16,
                shape,
                quantize_sidecars=True,
            )
        )
        final_fp32_error, final_target = mass.direct_post_wo_error_energy(
            final_fp32_coefficients,
            values,
            base_heads,
            target,
            projection,
            shape.mass_shape,
        )
        final_bf16_error, final_target_replay = mass.direct_post_wo_error_energy(
            final_bf16_coefficients,
            values,
            base_heads,
            target,
            projection,
            shape.mass_shape,
        )
        if not np.array_equal(final_target, final_target_replay):
            raise ValueError("content-selector final target replay changed")
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
            "mass_vs_reconstructed_score_coefficient_parity": (
                float(
                    np.max(
                        np.abs(
                            final_bf16_coefficients - final_score_coefficients
                        )
                    )
                )
                <= 1.0e-6
            ),
            "mass_vs_reconstructed_score_delta_parity": (
                float(np.max(np.abs(final_bf16_delta - final_score_delta)))
                <= 1.0e-6
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
                            np.sum(value, axis=-1, dtype=np.float32)
                            - np.float32(1.0)
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
                == shape.parameter_count * 2
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
        "query_features": {
            "artifact": {
                "file": query_path.name,
                "file_sha256": sha256_file(query_path),
                "tensor_sha256": {
                    name: _array_sha256(value)
                    for name, value in query_tensors.items()
                },
            },
            "derivation_contract": dict(query_result.contract),
            "derivation_contract_sha256": query_result.contract_sha256,
            "weight_tensor_sha256": dict(query_result.weight_tensor_sha256),
            "native_query_parity_evaluated": False,
            "native_query_parity_passed": False,
        },
        "record_authentication": record_authentication,
        "cross_manifest_record_join": dict(context["record_join"]),
        "folds": fold_rows,
        "arms": {
            "FP32": {
                "metrics": fp32_metrics,
                "coefficient_sha256": _array_sha256(fp32_oof),
                "error_energy_sha256": _array_sha256(fp32_error),
            },
            "BF16_RNE_parameters_and_sidecars": {
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
            "zero_model_native_coefficient_max_abs": zero_coefficient_max_abs,
            "zero_model_native_post_Wo_output_max_abs": zero_output_max_abs,
            "deterministic_serialized_replay_exact": deterministic_replay,
            "native_raw_score_parity_evaluated": False,
            "native_sidecar_lifecycle_parity_evaluated": False,
        },
        "gate": gate,
        "artifacts": {
            "directory": str(artifacts),
            "oof_coefficients": {
                "file": oof_path.name,
                "file_sha256": sha256_file(oof_path),
                "tensor_sha256": {
                    name: _array_sha256(value)
                    for name, value in oof_tensors.items()
                },
            },
            "all_train_selector": final_artifact,
        },
        "resource_contract": dict(frozen["resource_contract"]),
        "decision": {
            "train_content_selector_oof_gate_passed": gate["passed"],
            "final_BF16_artifact_sanity_passed": final_artifact_sanity_passed,
            "train_only_native_integration_implementation_authorized": (
                native_implementation_authorized
            ),
            "native_query_raw_score_and_sidecar_parity_evaluated": False,
            "native_runtime_execution_authorized": False,
            "development_authorized": False,
            "confirmation_authorized": False,
            "semantic_or_M3_gate_passed": False,
            "next_step": (
                "implement the fused native query/value-sidecar selector and "
                "pass query, score, coefficient, cache-lifecycle, and post-Wo "
                "parity before freezing a causal development rollout"
                if native_implementation_authorized
                else "reject this rank-4 content-selector feature/model class"
            ),
        },
        "post_fit_authentication": {
            "protocol": sha256_file(protocol_path) == protocol_sha256,
            "mass_protocol": (
                sha256_file(context["mass_protocol_path"])
                == _EXPECTED_MASS_PROTOCOL_SHA256
            ),
            "mass_result": (
                sha256_file(context["mass_result_path"])
                == _EXPECTED_MASS_RESULT_SHA256
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
                sha256_file(context["residual_manifest_path"])
                == _EXPECTED_RESIDUAL_MANIFEST_SHA256
            ),
            "package_manifest": (
                sha256_file(context["package"]["manifest_path"])
                == _EXPECTED_PACKAGE_MANIFEST_SHA256
            ),
            "non_mlp_weights": (
                sha256_file(context["package"]["non_mlp_path"])
                == _EXPECTED_NON_MLP_SHA256
            ),
            "source_inventory": frozen["source_sha256"] == _source_inventory(),
            "record_authentication": all(record_authentication.values()),
            "cross_manifest_record_join": all(context["record_join"].values()),
            "confirmation_not_opened": True,
        },
        "authenticated_confirmation_descriptor": dict(
            frozen["authenticated_confirmation_descriptor"]
        ),
        "confirmation_split_opened": False,
    }
    if not all(result["post_fit_authentication"].values()):
        raise ValueError("content-selector post-fit authentication failed")
    predecessor._atomic_json_new(output, result)
    _progress(
        f"completed with status {status}; "
        f"BF16 global recovery={bf16_metrics['global']['recovery']:.6f}"
    )
    return {"path": str(output), "sha256": sha256_file(output), "result": result}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze or execute the train-only C28 content selector",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--mass-protocol", required=True)
    freeze.add_argument("--mass-protocol-sha256", required=True)
    freeze.add_argument("--mass-result", required=True)
    freeze.add_argument("--mass-result-sha256", required=True)
    freeze.add_argument("--residual-manifest", required=True)
    freeze.add_argument("--residual-manifest-sha256", required=True)
    freeze.add_argument("--package-manifest", required=True)
    freeze.add_argument("--package-manifest-sha256", required=True)
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
        outcome = freeze_content_selector_protocol(
            mass_protocol=args.mass_protocol,
            mass_protocol_sha256=args.mass_protocol_sha256,
            mass_result=args.mass_result,
            mass_result_sha256=args.mass_result_sha256,
            residual_manifest=args.residual_manifest,
            residual_manifest_sha256=args.residual_manifest_sha256,
            package_manifest=args.package_manifest,
            package_manifest_sha256=args.package_manifest_sha256,
            out=args.out,
        )
    else:
        outcome = fit_screen_content_selector(
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
