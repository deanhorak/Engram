"""Fail-closed cached handoff for the train-only slot-simplex oracle.

The original slot-simplex screen completed its expensive native trace capture
before it was stopped during the numerical solve.  This module treats those
shards as an immutable input artifact.  It has three deliberately separate
operations:

``authenticate-cache``
    Re-authenticate the frozen V1 roots, every inherited and slot shard, the
    combined base/slot trace digest, historical output digests, and the BF16
    output projections.  It writes an immutable capture report only after all
    checks pass.

``freeze-cached``
    Freeze a distinct V2 numerical protocol around an authenticated capture
    report and the *current* solver sources.  V2 is not represented as an
    exact replay of the V1 solver source.

``solve-cached``
    Re-authenticate the V2 protocol and all cached inputs, then run the two
    frozen product-simplex arms without opening a native library, creating
    trace shards, or accessing the confirmation split.

Only the first-capture tensors were persisted.  The V1 descriptors recorded
equal reset digests, but the reset tensors and complete output-evidence
payloads were not stored.  The capture report therefore calls reset evidence
``descriptor-attested`` rather than independently re-derived.

The authenticated confirmation descriptor is copied by value.  This module
never opens, resolves, stats, or hashes the file named by that descriptor.
"""

from __future__ import annotations

import argparse
import os
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import engram.evaluation.olmoe_product_simplex_active_set_solver as active_solver
import engram.evaluation.olmoe_retrieval_episodic_head_mass_oracle as mass
import engram.evaluation.olmoe_retrieval_episodic_slot_simplex_oracle as slot
from engram.utils import atomic_json, sha256_file, sha256_json


_SCHEMA_VERSION = 1
_CAPTURE_EXPERIMENT = (
    "olmoe_q7_retrieval_episodic_slot_simplex_authenticated_capture"
)
_CAPTURE_STATUS = "authenticated_complete_v1_capture_for_cached_v2"
_PROTOCOL_EXPERIMENT = (
    "olmoe_q7_retrieval_episodic_slot_simplex_cached_v2_protocol"
)
_PROTOCOL_STATUS = "frozen_before_authenticated_cached_v2_solve"
_RESULT_EXPERIMENT = (
    "olmoe_q7_retrieval_episodic_slot_simplex_cached_v2_train_screen"
)

_EXPECTED_V1_PROTOCOL_SHA256 = (
    "56b33472ee23353e945abb9741f3b5b16e70965450d023a45cd6223a8d85c4cb"
)
_EXPECTED_V1_PARITY_SHA256 = (
    "4565a5fcaa2039f4229422243e0f121b3444c89495b4141b39f6358c19645a02"
)
_EXPECTED_SLOT_MANIFEST_SHA256 = (
    "0ac40bfa8f41d23627ce9e3ee89283f68828ae02d482c77a43be0d4d17129b04"
)
_EXPECTED_HEAD_MASS_MANIFEST_SHA256 = (
    "93df0a554744b97e7436b9a8b4bb71473bc21fa9f6c90985431274859164e0b6"
)
_EXPECTED_HEAD_MASS_PROTOCOL_SHA256 = (
    "fe09689452e6ae4f1a1b15332c61c1cc990cfc29b6a8b0d5a1758d9490a93af5"
)
_EXPECTED_HEAD_MASS_RESULT_SHA256 = (
    "f7060e7373c5faf8f154891e93efad35659723d8e3f04d83638b62fa9cf72596"
)
_EXPECTED_JOINT_PROTOCOL_SHA256 = (
    "aa03a71e3dd9e1fbb413a7773d57189c41029c26b2f4372b2fb7a26744305d24"
)
_EXPECTED_JOINT_RESULT_SHA256 = (
    "1329a51bac71cb81f44494c8ef70cb23a631eacebac5540b39e2e98ed5e30ea5"
)
_CONFIRMATION_FILENAME = "confirmation.jsonl"
_REFERENCE_SOLVER_SOURCE = (
    "src/engram/evaluation/olmoe_product_simplex_solver.py"
)
_ACTIVE_SET_SOLVER_SOURCE = (
    "src/engram/evaluation/olmoe_product_simplex_active_set_solver.py"
)
_CACHED_SOURCE = (
    "src/engram/evaluation/olmoe_retrieval_episodic_slot_simplex_cached.py"
)
_SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *mass._SOURCE_FILES,
            *slot._SOURCE_FILES,
            _REFERENCE_SOLVER_SOURCE,
            _ACTIVE_SET_SOLVER_SOURCE,
            _CACHED_SOURCE,
        )
    )
)
_DEFAULT_ROW_BATCH_SIZE = slot._RECORDS * len(slot._READ_POSITIONS)
_MAXIMUM_ACTIVE_SET_ITERATIONS = 128
_FALLBACK_MAXIMUM_ITERATIONS = 512
_RELATIVE_GAP_TOLERANCE = 1.0e-12
_ABSOLUTE_GAP_TOLERANCE = 1.0e-13
_WORKING_SET_TOLERANCE = 1.0e-12
_KKT_RESIDUAL_TOLERANCE = 1.0e-10
_REDUCED_COST_TOLERANCE = 1.0e-12
_FORK_ARRAYS: Mapping[str, np.ndarray] | None = None
_FORK_OUTPUT_PROJECTION: np.ndarray | None = None


def _progress(message: str) -> None:
    print(
        f"[retrieval-episodic-slot-simplex-cached] {message}",
        file=sys.stderr,
        flush=True,
    )


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _source_inventory() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    return {name: sha256_file(repository / name) for name in _SOURCE_FILES}


def _checked_file(
    value: str | Path,
    expected_sha256: str,
    label: str,
) -> Path:
    """Return a hash-authenticated regular file without following symlinks.

    The lexical confirmation-file rejection happens before any filesystem
    operation.  It is a defense-in-depth guarantee for malformed bindings,
    not an authorization to inspect that file.
    """

    requested = Path(value).expanduser()
    if requested.name == _CONFIRMATION_FILENAME:
        raise ValueError(f"{label} cannot name the confirmation split")
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{label} SHA-256 is invalid")
    if requested.is_symlink():
        raise ValueError(f"{label} is invalid")
    source = requested.resolve()
    if not source.is_file() or sha256_file(source) != expected_sha256:
        raise ValueError(f"{label} root changed")
    return source


def _new_output(value: str | Path, label: str) -> Path:
    requested = Path(value).expanduser()
    if requested.name == _CONFIRMATION_FILENAME:
        raise ValueError(f"{label} cannot name the confirmation split")
    if requested.exists() or requested.is_symlink():
        raise ValueError(f"{label} already exists")
    parent = requested.parent.resolve()
    if not parent.is_dir():
        raise ValueError(f"{label} parent does not exist")
    return parent / requested.name


def _checked_relative_file(
    parent: Path,
    value: Any,
    expected_sha256: str,
    label: str,
) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} path is invalid")
    relative = Path(value)
    if (
        relative.is_absolute()
        or relative.name == _CONFIRMATION_FILENAME
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{label} path is invalid")
    return _checked_file(parent / relative, expected_sha256, label)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is invalid")
    return value


def _require_binding(
    value: Any,
    label: str,
    *,
    expected_sha256: str | None = None,
) -> tuple[Path, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} binding is invalid")
    digest = value.get("sha256")
    if not _is_sha256(digest):
        raise ValueError(f"{label} binding is invalid")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"{label} root changed")
    return _checked_file(value.get("path"), digest, label), digest


def _require_false_confirmation(value: Mapping[str, Any], label: str) -> None:
    if value.get("confirmation_split_opened") is not False:
        raise ValueError(f"{label} confirmation contract changed")


def _authenticate_historical_roots(
    *,
    v1_protocol: str | Path,
    v1_protocol_sha256: str,
    slot_manifest: str | Path,
    slot_manifest_sha256: str,
    head_mass_result: str | Path,
    head_mass_result_sha256: str,
) -> dict[str, Any]:
    """Authenticate immutable historical JSON roots without rebuilding V1."""

    expected_arguments = (
        (v1_protocol_sha256, _EXPECTED_V1_PROTOCOL_SHA256, "V1 protocol"),
        (
            slot_manifest_sha256,
            _EXPECTED_SLOT_MANIFEST_SHA256,
            "slot manifest",
        ),
        (
            head_mass_result_sha256,
            _EXPECTED_HEAD_MASS_RESULT_SHA256,
            "head-mass result",
        ),
    )
    for observed, expected, label in expected_arguments:
        if not isinstance(observed, str) or observed.lower() != expected:
            raise ValueError(f"cached V2 {label} root changed")

    v1_path = _checked_file(
        v1_protocol,
        _EXPECTED_V1_PROTOCOL_SHA256,
        "cached V2 V1 protocol",
    )
    v1 = _read_json(v1_path, "cached V2 V1 protocol")
    _require_false_confirmation(v1, "cached V2 V1 protocol")
    scope = v1.get("scope")
    descriptor = v1.get("authenticated_confirmation_descriptor")
    if (
        v1.get("schema_version") != slot._SCHEMA_VERSION
        or v1.get("experiment") != slot._PROTOCOL_EXPERIMENT
        or v1.get("status") != slot._PROTOCOL_STATUS
        or not isinstance(scope, Mapping)
        or scope.get("confirmation_file_access_permitted") is not False
        or scope.get("split") != "train"
        or not isinstance(descriptor, Mapping)
        or descriptor.get("file") != _CONFIRMATION_FILENAME
        or not _is_sha256(descriptor.get("sha256"))
        or v1.get("oracle_method", {}).get("two_arms") is not True
    ):
        raise ValueError("cached V2 V1 protocol contract changed")

    parity_path, parity_sha256 = _require_binding(
        v1.get("trace_parity"),
        "cached V2 V1 parity",
        expected_sha256=_EXPECTED_V1_PARITY_SHA256,
    )
    parity = _read_json(parity_path, "cached V2 V1 parity")
    parity_body = parity.get("parity")
    if (
        parity != v1["trace_parity"].get("report")
        or parity.get("schema_version") != slot._SCHEMA_VERSION
        or parity.get("experiment") != slot._PARITY_EXPERIMENT
        or parity.get("status") != slot._PARITY_STATUS
        or not isinstance(parity_body, Mapping)
        or parity_body.get("passed") is not True
        or parity_body.get("first_trace") != parity_body.get("reset_trace")
        or parity_body.get("first_output_evidence_sha256")
        != parity_body.get("reset_output_evidence_sha256")
    ):
        raise ValueError("cached V2 V1 parity contract changed")
    _require_false_confirmation(parity, "cached V2 V1 parity")

    slot_path = _checked_file(
        slot_manifest,
        _EXPECTED_SLOT_MANIFEST_SHA256,
        "cached V2 slot manifest",
    )
    slot_value = _read_json(slot_path, "cached V2 slot manifest")
    _require_false_confirmation(slot_value, "cached V2 slot manifest")
    if (
        slot_value.get("schema_version") != slot._SCHEMA_VERSION
        or slot_value.get("experiment") != slot._RESULT_EXPERIMENT
        or slot_value.get("protocol", {}).get("sha256")
        != _EXPECTED_V1_PROTOCOL_SHA256
        or slot_value.get("format") != "safetensors"
        or slot_value.get("stored_tensors") != list(slot._SLOT_TRACE_KEYS)
        or slot_value.get("record_order") != list(range(slot._RECORDS))
        or not isinstance(slot_value.get("shards"), list)
        or len(slot_value["shards"]) != slot._RECORDS
    ):
        raise ValueError("cached V2 slot manifest contract changed")

    inherited_path, inherited_sha256 = _require_binding(
        slot_value.get("inherited_head_mass_trace_manifest"),
        "cached V2 inherited manifest",
        expected_sha256=_EXPECTED_HEAD_MASS_MANIFEST_SHA256,
    )
    if (
        v1.get("inherited_head_mass_trace_manifest", {}).get("sha256")
        != inherited_sha256
    ):
        raise ValueError("cached V2 inherited manifest binding changed")
    inherited = _read_json(inherited_path, "cached V2 inherited manifest")
    _require_false_confirmation(inherited, "cached V2 inherited manifest")

    joint_path, joint_sha256 = _require_binding(
        v1.get("joint_gamma_protocol"),
        "cached V2 joint-gamma protocol",
        expected_sha256=_EXPECTED_JOINT_PROTOCOL_SHA256,
    )
    joint_protocol = _read_json(joint_path, "cached V2 joint-gamma protocol")
    _require_false_confirmation(joint_protocol, "cached V2 joint-gamma protocol")
    if (
        joint_protocol.get("schema_version") != 1
        or joint_protocol.get("experiment")
        != "olmoe_q7_retrieval_episodic_joint_gamma_oracle_protocol"
        or joint_protocol.get("status")
        != "frozen_before_cached_joint_gamma_execution"
    ):
        raise ValueError("cached V2 joint-gamma protocol contract changed")

    joint_result_path, joint_result_sha256 = _require_binding(
        v1.get("joint_gamma_result"),
        "cached V2 joint-gamma result",
        expected_sha256=_EXPECTED_JOINT_RESULT_SHA256,
    )
    joint_result = _read_json(
        joint_result_path,
        "cached V2 joint-gamma result",
    )
    _require_false_confirmation(joint_result, "cached V2 joint-gamma result")
    if (
        joint_result.get("schema_version") != 1
        or joint_result.get("status")
        != "train_episodic_joint_gamma_oracle_gate_failed"
    ):
        raise ValueError("cached V2 joint-gamma failure changed")

    head_protocol_path, head_protocol_sha256 = _require_binding(
        joint_protocol.get("head_mass_protocol"),
        "cached V2 head-mass protocol",
        expected_sha256=_EXPECTED_HEAD_MASS_PROTOCOL_SHA256,
    )
    head_protocol = _read_json(
        head_protocol_path,
        "cached V2 head-mass protocol",
    )
    _require_false_confirmation(head_protocol, "cached V2 head-mass protocol")
    if (
        head_protocol.get("schema_version") != mass._SCHEMA_VERSION
        or head_protocol.get("experiment") != mass._PROTOCOL_EXPERIMENT
        or head_protocol.get("status") != mass._PROTOCOL_STATUS
    ):
        raise ValueError("cached V2 head-mass protocol contract changed")

    checkpoint_path, checkpoint_sha256 = _require_binding(
        head_protocol.get("training_checkpoint"),
        "cached V2 training checkpoint",
    )
    checkpoint = _read_json(
        checkpoint_path,
        "cached V2 training checkpoint",
    )
    _require_false_confirmation(checkpoint, "cached V2 training checkpoint")
    selector_path, selector_sha256 = _require_binding(
        checkpoint.get("protocol"),
        "cached V2 selector protocol",
    )
    selector = _read_json(selector_path, "cached V2 selector protocol")
    corpus = selector.get("corpus")
    if (
        not isinstance(corpus, Mapping)
        or corpus != head_protocol.get("corpus")
    ):
        raise ValueError("cached V2 corpus contract changed")
    corpus_manifest_path = _checked_relative_file(
        selector_path.parent,
        corpus.get("manifest_file"),
        corpus.get("manifest_sha256"),
        "cached V2 corpus manifest",
    )
    train_descriptor = corpus.get("splits", {}).get("train")
    if not isinstance(train_descriptor, Mapping):
        raise ValueError("cached V2 train descriptor changed")
    train_path = _checked_relative_file(
        selector_path.parent,
        train_descriptor.get("file"),
        train_descriptor.get("sha256"),
        "cached V2 train split",
    )
    train_records = mass.capacity.bias.rank.retrieval._read_split(
        train_path,
        split="train",
    )
    if (
        len(train_records) != slot._RECORDS
        or sha256_json([row["identity_sha256"] for row in train_records])
        != train_descriptor.get("record_identity_sha256")
    ):
        raise ValueError("cached V2 train record identity changed")

    result_path = _checked_file(
        head_mass_result,
        _EXPECTED_HEAD_MASS_RESULT_SHA256,
        "cached V2 head-mass result",
    )
    result = _read_json(result_path, "cached V2 head-mass result")
    _require_false_confirmation(result, "cached V2 head-mass result")
    output_rows = result.get("base_output_authentication")
    trace_binding = result.get("trace_manifest")
    post = result.get("post_run_authentication")
    if (
        result.get("schema_version") != mass._SCHEMA_VERSION
        or result.get("experiment") != mass._RESULT_EXPERIMENT
        or result.get("status")
        != "train_episodic_head_mass_oracle_gate_failed"
        or result.get("protocol", {}).get("sha256") != head_protocol_sha256
        or not isinstance(output_rows, list)
        or len(output_rows) != slot._RECORDS
        or not isinstance(trace_binding, Mapping)
        or trace_binding.get("sha256") != inherited_sha256
        or trace_binding.get("shard_count") != slot._RECORDS
        or not isinstance(post, Mapping)
        or not post
        or not all(check is True for check in post.values())
    ):
        raise ValueError("cached V2 head-mass result contract changed")
    if (
        joint_protocol.get("head_mass_result", {}).get("sha256")
        != _EXPECTED_HEAD_MASS_RESULT_SHA256
        or joint_protocol.get("cached_trace_manifest", {}).get("sha256")
        != inherited_sha256
    ):
        raise ValueError("cached V2 joint-gamma inherited roots changed")

    inherited_shards = inherited.get("shards")
    if (
        inherited.get("schema_version") != mass._SCHEMA_VERSION
        or inherited.get("experiment") != mass._RESULT_EXPERIMENT
        or inherited.get("format") != "safetensors"
        or inherited.get("record_order") != list(range(slot._RECORDS))
        or not isinstance(inherited_shards, list)
        or len(inherited_shards) != slot._RECORDS
        or trace_binding.get("shards") != inherited_shards
    ):
        raise ValueError("cached V2 inherited manifest contract changed")

    projection_contract = joint_protocol.get("output_projection")
    if (
        not isinstance(projection_contract, Mapping)
        or projection_contract.get("dtype")
        != "authenticated_BF16_loaded_as_float32"
        or not isinstance(projection_contract.get("tensor_sha256"), Mapping)
        or len(projection_contract["tensor_sha256"]) != slot._LAYERS
        or not all(
            _is_sha256(digest)
            for digest in projection_contract["tensor_sha256"].values()
        )
    ):
        raise ValueError("cached V2 output projection contract changed")
    projection_requested = Path(projection_contract.get("source")).expanduser()
    if projection_requested.name == _CONFIRMATION_FILENAME:
        raise ValueError("cached V2 projection cannot name confirmation data")
    if projection_requested.is_symlink():
        raise ValueError("cached V2 output projection source is invalid")
    projection_path = projection_requested.resolve()
    if (
        not projection_path.is_file()
        or projection_path.suffix != ".safetensors"
    ):
        raise ValueError("cached V2 output projection source is invalid")
    projection_file_sha256 = sha256_file(projection_path)
    output_projection, projection_hashes = mass._load_output_projections(
        {"non_mlp_path": projection_path}
    )
    if (
        projection_hashes != dict(projection_contract["tensor_sha256"])
        or sha256_file(projection_path) != projection_file_sha256
    ):
        raise ValueError("cached V2 output projection tensors changed")

    return {
        "v1_protocol_path": v1_path,
        "v1_protocol": v1,
        "parity_path": parity_path,
        "parity_sha256": parity_sha256,
        "parity": parity,
        "slot_manifest_path": slot_path,
        "slot_manifest": slot_value,
        "inherited_manifest_path": inherited_path,
        "inherited_manifest_sha256": inherited_sha256,
        "inherited_manifest": inherited,
        "head_mass_protocol_path": head_protocol_path,
        "head_mass_protocol_sha256": head_protocol_sha256,
        "head_mass_result_path": result_path,
        "head_mass_result": result,
        "training_checkpoint_path": checkpoint_path,
        "training_checkpoint_sha256": checkpoint_sha256,
        "selector_protocol_path": selector_path,
        "selector_protocol_sha256": selector_sha256,
        "corpus_manifest_path": corpus_manifest_path,
        "corpus_manifest_sha256": corpus["manifest_sha256"],
        "train_path": train_path,
        "train_sha256": train_descriptor["sha256"],
        "train_records": train_records,
        "joint_protocol_path": joint_path,
        "joint_protocol_sha256": joint_sha256,
        "joint_protocol": joint_protocol,
        "joint_result_path": joint_result_path,
        "joint_result_sha256": joint_result_sha256,
        "joint_result": joint_result,
        "projection_path": projection_path,
        "projection_file_sha256": projection_file_sha256,
        "projection_hashes": projection_hashes,
        "output_projection": output_projection,
    }


def _audit_cached_rows(
    context: Mapping[str, Any],
    *,
    stack_arrays: bool,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray] | None]:
    slot_descriptors = context["slot_manifest"]["shards"]
    inherited_descriptors = context["inherited_manifest"]["shards"]
    output_rows = context["head_mass_result"]["base_output_authentication"]
    train_records = context["train_records"]
    slot_directory = Path(context["slot_manifest_path"]).parent
    inherited_directory = Path(context["inherited_manifest_path"]).parent
    stacked_rows: dict[str, list[np.ndarray]] | None = (
        {name: [] for name in (*slot._BASE_TRACE_KEYS, *slot._SLOT_TRACE_KEYS)}
        if stack_arrays
        else None
    )
    report_rows: list[dict[str, Any]] = []

    for index, (
        slot_descriptor,
        inherited_descriptor,
        historical,
        train_record,
    ) in enumerate(
        zip(
            slot_descriptors,
            inherited_descriptors,
            output_rows,
            train_records,
            strict=True,
        )
    ):
        if (
            slot_descriptor.get("record_index") != index
            or inherited_descriptor.get("record_index") != index
            or historical.get("record_index") != index
            or slot_descriptor.get("record_id")
            != inherited_descriptor.get("record_id")
            or slot_descriptor.get("record_id") != historical.get("record_id")
            or slot_descriptor.get("record_id") != train_record.get("record_id")
        ):
            raise ValueError("cached V2 record binding changed")
        inherited_arrays = mass._validate_trace_shard(
            inherited_directory / inherited_descriptor["file"],
            inherited_descriptor,
        )
        slot_arrays = slot._validate_slot_shard(
            slot_directory / slot_descriptor["file"],
            slot_descriptor,
        )
        combined = {
            **{
                name: inherited_arrays[name]
                for name in slot._BASE_TRACE_KEYS
            },
            **slot_arrays,
        }
        summary = slot._trace_summary(combined, slot._READ_POSITIONS)
        source_sha256 = slot_descriptor.get("source_record_sha256")
        output_sha256 = slot_descriptor.get("output_evidence_sha256")
        reset_output_sha256 = slot_descriptor.get(
            "reset_output_evidence_sha256"
        )
        if (
            summary["trace_sha256"] != slot_descriptor.get("full_trace_sha256")
            or slot_descriptor.get("reset_full_trace_sha256")
            != summary["trace_sha256"]
            or source_sha256 != inherited_descriptor.get("source_record_sha256")
            or source_sha256 != historical.get("source_record_sha256")
            or source_sha256 != sha256_json(train_record)
            or output_sha256
            != inherited_descriptor.get("output_evidence_sha256")
            or output_sha256
            != historical.get("historical_output_evidence_sha256")
            or output_sha256
            != historical.get("observed_output_evidence_sha256")
            or reset_output_sha256 != output_sha256
            or historical.get("reset_output_evidence_sha256") != output_sha256
            or historical.get("trace_sha256")
            != inherited_descriptor.get("trace_sha256")
            or historical.get("reset_trace_sha256")
            != inherited_descriptor.get("trace_sha256")
            or historical.get("base_outputs_counters_and_loss_exact") is not True
            or historical.get("reset_outputs_counters_loss_and_trace_exact")
            is not True
        ):
            raise ValueError(f"cached V2 record {index} evidence changed")
        if index == 0:
            parity = context["parity"]["parity"]
            if (
                parity.get("record_index") != 0
                or parity.get("first_trace") != summary
                or parity.get("reset_trace") != summary
                or parity.get("first_output_evidence_sha256") != output_sha256
                or parity.get("reset_output_evidence_sha256") != output_sha256
                or parity.get("inherited_trace_sha256")
                != inherited_descriptor.get("trace_sha256")
            ):
                raise ValueError("cached V2 parity/capture binding changed")

        checks = {
            "slot_file_and_tensor_hashes_authenticated": True,
            "inherited_file_and_tensor_hashes_authenticated": True,
            "slot_values_exact_bf16_decodes": True,
            "combined_base_slot_trace_digest_recomputed": True,
            "source_record_rederived_from_authenticated_train_split": True,
            "output_digest_bound_to_historical_result": True,
            "reset_digest_matches_first_capture_descriptor": True,
        }
        report_rows.append(
            {
                "record_index": index,
                "record_id": slot_descriptor["record_id"],
                "source_record_sha256": source_sha256,
                "slot_shard": {
                    "file": slot_descriptor["file"],
                    "file_sha256": slot_descriptor["file_sha256"],
                    "tensor_sha256": dict(slot_descriptor["tensor_sha256"]),
                },
                "inherited_shard": {
                    "file": inherited_descriptor["file"],
                    "file_sha256": inherited_descriptor["file_sha256"],
                    "trace_sha256": inherited_descriptor["trace_sha256"],
                    "tensor_sha256": dict(
                        inherited_descriptor["tensor_sha256"]
                    ),
                },
                "combined_full_trace_sha256": summary["trace_sha256"],
                "descriptor_reset_full_trace_sha256": slot_descriptor[
                    "reset_full_trace_sha256"
                ],
                "historical_output_evidence_sha256": output_sha256,
                "descriptor_reset_output_evidence_sha256": reset_output_sha256,
                "reconstruction": {
                    "mass_partition_max_abs": summary[
                        "mass_partition_max_abs"
                    ],
                    "slot_mass_reconstruction_max_abs": summary[
                        "slot_mass_reconstruction_max_abs"
                    ],
                    "episodic_component_reconstruction_max_abs": summary[
                        "episodic_component_reconstruction_max_abs"
                    ],
                    "base_component_reconstruction_max_abs": summary[
                        "base_component_reconstruction_max_abs"
                    ],
                },
                "checks": checks,
            }
        )
        if stacked_rows is not None:
            for name in slot._BASE_TRACE_KEYS:
                stacked_rows[name].append(inherited_arrays[name])
            for name in slot._SLOT_TRACE_KEYS:
                stacked_rows[name].append(slot_arrays[name])

    stacked = (
        {
            name: np.ascontiguousarray(np.stack(values), dtype=np.float32)
            for name, values in stacked_rows.items()
        }
        if stacked_rows is not None
        else None
    )
    return report_rows, stacked


def _capture_bindings(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "v1_protocol": {
            "path": str(context["v1_protocol_path"]),
            "sha256": _EXPECTED_V1_PROTOCOL_SHA256,
        },
        "v1_trace_parity": {
            "path": str(context["parity_path"]),
            "sha256": context["parity_sha256"],
        },
        "slot_trace_manifest": {
            "path": str(context["slot_manifest_path"]),
            "sha256": _EXPECTED_SLOT_MANIFEST_SHA256,
        },
        "inherited_head_mass_manifest": {
            "path": str(context["inherited_manifest_path"]),
            "sha256": context["inherited_manifest_sha256"],
        },
        "inherited_head_mass_protocol": {
            "path": str(context["head_mass_protocol_path"]),
            "sha256": context["head_mass_protocol_sha256"],
        },
        "inherited_head_mass_result": {
            "path": str(context["head_mass_result_path"]),
            "sha256": _EXPECTED_HEAD_MASS_RESULT_SHA256,
        },
        "training_checkpoint": {
            "path": str(context["training_checkpoint_path"]),
            "sha256": context["training_checkpoint_sha256"],
        },
        "selector_protocol": {
            "path": str(context["selector_protocol_path"]),
            "sha256": context["selector_protocol_sha256"],
        },
        "corpus_manifest": {
            "path": str(context["corpus_manifest_path"]),
            "sha256": context["corpus_manifest_sha256"],
        },
        "train_split": {
            "path": str(context["train_path"]),
            "sha256": context["train_sha256"],
        },
        "joint_gamma_protocol": {
            "path": str(context["joint_protocol_path"]),
            "sha256": context["joint_protocol_sha256"],
        },
        "joint_gamma_result": {
            "path": str(context["joint_result_path"]),
            "sha256": context["joint_result_sha256"],
        },
    }


def _build_capture_report(
    *,
    v1_protocol: str | Path,
    v1_protocol_sha256: str,
    slot_manifest: str | Path,
    slot_manifest_sha256: str,
    head_mass_result: str | Path,
    head_mass_result_sha256: str,
    stack_arrays: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = _authenticate_historical_roots(
        v1_protocol=v1_protocol,
        v1_protocol_sha256=v1_protocol_sha256,
        slot_manifest=slot_manifest,
        slot_manifest_sha256=slot_manifest_sha256,
        head_mass_result=head_mass_result,
        head_mass_result_sha256=head_mass_result_sha256,
    )
    rows, stacked = _audit_cached_rows(context, stack_arrays=stack_arrays)
    report = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _CAPTURE_EXPERIMENT,
        "status": _CAPTURE_STATUS,
        "bindings": _capture_bindings(context),
        "capture": {
            "records": slot._RECORDS,
            "read_positions_per_record": len(slot._READ_POSITIONS),
            "record_order": list(range(slot._RECORDS)),
            "stored_slot_tensors": list(slot._SLOT_TRACE_KEYS),
            "native_rerun_performed": False,
            "native_runtime_opened_by_cached_handoff": False,
            "all_first_capture_tensors_reauthenticated": True,
        },
        "record_authentication": rows,
        "capture_rows_sha256": sha256_json(rows),
        "output_projection": {
            "source": str(context["projection_path"]),
            "file_sha256": context["projection_file_sha256"],
            "dtype": "authenticated_BF16_loaded_as_float32",
            "tensor_sha256": dict(context["projection_hashes"]),
        },
        "reset_replay_evidence": {
            "status": "descriptor_attested_not_independently_rederived",
            "first_capture_tensors_recomputed_from_persisted_shards": True,
            "reset_tensors_persisted": False,
            "complete_output_evidence_payloads_persisted": False,
            "native_rerun_performed": False,
            "claim": (
                "V1 recorded equal first/reset full-trace and output-evidence "
                "digests for every record; V2 verifies those descriptor "
                "equalities but cannot independently reconstruct reset "
                "execution from tensors that were not persisted"
            ),
        },
        "post_capture_authentication": {
            "historical_json_roots": True,
            "all_slot_shards": True,
            "all_inherited_shards": True,
            "all_combined_trace_digests": True,
            "all_historical_output_bindings": True,
            "output_projection_file_and_tensors": True,
            "confirmation_not_opened": True,
        },
        # Copied by value only.  Do not resolve or inspect its ``file`` field.
        "authenticated_confirmation_descriptor": dict(
            context["v1_protocol"]["authenticated_confirmation_descriptor"]
        ),
        "confirmation_split_opened": False,
    }
    context = dict(context)
    context["stacked_arrays"] = stacked
    return context, report


def authenticate_cached_capture(
    *,
    v1_protocol: str | Path,
    v1_protocol_sha256: str,
    slot_manifest: str | Path,
    slot_manifest_sha256: str,
    head_mass_result: str | Path,
    head_mass_result_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    """Write an immutable report for the completed V1 capture."""

    output = _new_output(out, "cached V2 capture report")
    _progress("authenticating historical roots, shards, outputs, and projection")
    _context, report = _build_capture_report(
        v1_protocol=v1_protocol,
        v1_protocol_sha256=v1_protocol_sha256,
        slot_manifest=slot_manifest,
        slot_manifest_sha256=slot_manifest_sha256,
        head_mass_result=head_mass_result,
        head_mass_result_sha256=head_mass_result_sha256,
    )
    atomic_json(output, report)
    _progress(f"authenticated capture report written to {output}")
    return {"path": str(output), "sha256": sha256_file(output), "report": report}


def _authenticate_capture_report(
    capture_report: str | Path,
    capture_report_sha256: str,
    *,
    stack_arrays: bool,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    source = _checked_file(
        capture_report,
        capture_report_sha256.lower(),
        "cached V2 capture report",
    )
    value = _read_json(source, "cached V2 capture report")
    _require_false_confirmation(value, "cached V2 capture report")
    bindings = value.get("bindings")
    if (
        value.get("schema_version") != _SCHEMA_VERSION
        or value.get("experiment") != _CAPTURE_EXPERIMENT
        or value.get("status") != _CAPTURE_STATUS
        or not isinstance(bindings, Mapping)
        or value.get("capture", {}).get("native_rerun_performed") is not False
        or value.get("reset_replay_evidence", {}).get("status")
        != "descriptor_attested_not_independently_rederived"
    ):
        raise ValueError("cached V2 capture report contract changed")
    v1 = bindings.get("v1_protocol")
    manifest = bindings.get("slot_trace_manifest")
    result = bindings.get("inherited_head_mass_result")
    if not all(isinstance(binding, Mapping) for binding in (v1, manifest, result)):
        raise ValueError("cached V2 capture report bindings changed")
    context, expected = _build_capture_report(
        v1_protocol=v1.get("path"),
        v1_protocol_sha256=v1.get("sha256"),
        slot_manifest=manifest.get("path"),
        slot_manifest_sha256=manifest.get("sha256"),
        head_mass_result=result.get("path"),
        head_mass_result_sha256=result.get("sha256"),
        stack_arrays=stack_arrays,
    )
    if value != expected:
        raise ValueError("cached V2 capture report changed")
    context = dict(context)
    context["capture_report_path"] = source
    context["capture_report_sha256"] = capture_report_sha256.lower()
    return source, value, context


def _default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1, slot._LAYERS))


def _build_cached_protocol(
    *,
    capture_path: Path,
    capture_sha256: str,
    capture: Mapping[str, Any],
    context: Mapping[str, Any],
    workers: int,
    row_batch_size: int,
) -> dict[str, Any]:
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers <= 0
        or workers > slot._LAYERS
        or isinstance(row_batch_size, bool)
        or not isinstance(row_batch_size, int)
        or row_batch_size <= 0
        or row_batch_size
        > slot._RECORDS * len(slot._READ_POSITIONS)
    ):
        raise ValueError("cached V2 parallel solve configuration is invalid")
    v1 = context["v1_protocol"]
    sources = _source_inventory()
    oracle_method = dict(v1["oracle_method"])
    v1_row_batch_size = oracle_method["row_batch_size"]
    v1_relative_gap_target = oracle_method["relative_objective_gap_target"]
    oracle_method["row_batch_size"] = row_batch_size
    oracle_method["v1_capture_protocol_row_batch_size"] = v1_row_batch_size
    oracle_method["row_batching_changes_feasible_set_or_objective"] = False
    oracle_method.update(
        {
            "solver": (
                "deterministic bulk product-simplex active-set KKT solves "
                "with scale-checked singular/cycle fail-closed handling, "
                "full Frank-Wolfe certification, and pairwise block "
                "Frank-Wolfe fallback"
            ),
            "maximum_active_set_iterations": _MAXIMUM_ACTIVE_SET_ITERATIONS,
            "fallback_maximum_iterations": _FALLBACK_MAXIMUM_ITERATIONS,
            "maximum_iterations": _FALLBACK_MAXIMUM_ITERATIONS,
            "relative_objective_gap_target": _RELATIVE_GAP_TOLERANCE,
            "absolute_objective_gap_target": _ABSOLUTE_GAP_TOLERANCE,
            "working_set_tolerance": _WORKING_SET_TOLERANCE,
            "kkt_residual_tolerance": _KKT_RESIDUAL_TOLERANCE,
            "reduced_cost_tolerance": _REDUCED_COST_TOLERANCE,
            "v1_relative_objective_gap_target": v1_relative_gap_target,
            "active_set_solution_authority": (
                "feasibility plus the recomputed full product-simplex "
                "Frank-Wolfe gap; KKT support termination alone has no "
                "progression authority"
            ),
        }
    )
    return {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _PROTOCOL_EXPERIMENT,
        "status": _PROTOCOL_STATUS,
        "capture_report": {
            "path": str(capture_path),
            "sha256": capture_sha256,
            "capture_rows_sha256": capture["capture_rows_sha256"],
        },
        "historical_bindings": dict(capture["bindings"]),
        "output_projection": dict(capture["output_projection"]),
        "oracle_method": oracle_method,
        "progression_gate": dict(v1["progression_gate"]),
        "resource_contract": dict(v1["resource_contract"]),
        "cache_contract": {
            "immutable_complete_v1_capture_required": True,
            "all_shards_reauthenticated_before_solve": True,
            "combined_base_slot_digests_recomputed_before_solve": True,
            "output_projection_reauthenticated_before_solve": True,
            "native_execution_permitted": False,
            "native_library_loading_permitted": False,
            "trace_shard_creation_permitted": False,
            "partial_v1_solver_output_permitted": False,
            "confirmation_file_access_permitted": False,
        },
        "solver_source_transition": {
            "v1_frozen_solver_sha256": v1["source_sha256"][
                _REFERENCE_SOLVER_SOURCE
            ],
            "cached_v2_reference_solver_sha256": sources[
                _REFERENCE_SOLVER_SOURCE
            ],
            "cached_v2_active_set_solver_sha256": sources[
                _ACTIVE_SET_SOLVER_SOURCE
            ],
            "exact_v1_solver_source_replay_claimed": False,
            "reason": (
                "the product-simplex solver source changed after V1 capture; "
                "V2 prospectively binds the current solver and reuses only "
                "the authenticated capture tensors"
            ),
        },
        "parallel_execution": {
            "backend": "forked_process_pool",
            "task_unit": "one_layer_one_arm",
            "workers": workers,
            "deterministic_destination_order": "record_position_layer",
            "arrays_and_output_projection_installed_before_fork": True,
            "worker_task_payload": [
                "arm",
                "layer",
                "row_batch_size",
                "maximum_active_set_iterations",
                "fallback_maximum_iterations",
                "relative_gap_tolerance",
                "absolute_gap_tolerance",
                "working_set_tolerance",
                "kkt_residual_tolerance",
                "reduced_cost_tolerance",
            ],
            "worker_blas_threads": 1,
            "full_deterministic_solver_replay_per_task": True,
        },
        "reset_replay_evidence": dict(capture["reset_replay_evidence"]),
        "scope": {
            "split": "train",
            "cached_same_state_capacity_evidence_only": True,
            "causal_rollout": False,
            "semantic_or_M3_pass": False,
            "development_outcomes_used": False,
            "confirmation_file_access_permitted": False,
        },
        "authorized_next_step_on_feasible_pass": v1[
            "authorized_next_step_on_feasible_pass"
        ],
        "failure_interpretation": v1["failure_interpretation"],
        # Copied by value only.
        "authenticated_confirmation_descriptor": dict(
            capture["authenticated_confirmation_descriptor"]
        ),
        "source_sha256": sources,
        "confirmation_split_opened": False,
    }


def freeze_cached_protocol(
    *,
    capture_report: str | Path,
    capture_report_sha256: str,
    out: str | Path,
    workers: int | None = None,
    row_batch_size: int = _DEFAULT_ROW_BATCH_SIZE,
) -> dict[str, Any]:
    """Freeze a distinct V2 solver protocol over an authenticated capture."""

    output = _new_output(out, "cached V2 protocol")
    _progress("re-authenticating capture before cached protocol freeze")
    capture_path, capture, context = _authenticate_capture_report(
        capture_report,
        capture_report_sha256,
        stack_arrays=False,
    )
    frozen_workers = _default_workers() if workers is None else workers
    protocol = _build_cached_protocol(
        capture_path=capture_path,
        capture_sha256=capture_report_sha256.lower(),
        capture=capture,
        context=context,
        workers=frozen_workers,
        row_batch_size=row_batch_size,
    )
    atomic_json(output, protocol)
    _progress(f"cached V2 protocol written to {output}")
    return {"path": str(output), "sha256": sha256_file(output), "protocol": protocol}


def _authenticate_cached_protocol(
    protocol: str | Path,
    protocol_sha256: str,
    *,
    stack_arrays: bool,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = _checked_file(
        protocol,
        protocol_sha256.lower(),
        "cached V2 protocol",
    )
    value = _read_json(source, "cached V2 protocol")
    _require_false_confirmation(value, "cached V2 protocol")
    binding = value.get("capture_report")
    if (
        value.get("schema_version") != _SCHEMA_VERSION
        or value.get("experiment") != _PROTOCOL_EXPERIMENT
        or value.get("status") != _PROTOCOL_STATUS
        or not isinstance(binding, Mapping)
        or value.get("cache_contract", {}).get("native_execution_permitted")
        is not False
        or value.get("cache_contract", {}).get(
            "confirmation_file_access_permitted"
        )
        is not False
    ):
        raise ValueError("cached V2 frozen protocol contract changed")
    capture_path, capture, context = _authenticate_capture_report(
        binding.get("path"),
        binding.get("sha256"),
        stack_arrays=stack_arrays,
    )
    expected = _build_cached_protocol(
        capture_path=capture_path,
        capture_sha256=binding["sha256"],
        capture=capture,
        context=context,
        workers=value.get("parallel_execution", {}).get("workers"),
        row_batch_size=value.get("oracle_method", {}).get("row_batch_size"),
    )
    if value != expected:
        raise ValueError("cached V2 frozen protocol changed")
    context = dict(context)
    context["cached_protocol_path"] = source
    context["cached_protocol_sha256"] = protocol_sha256.lower()
    context["cached_protocol"] = expected
    return source, expected, capture, context


def _post_solve_authentication(context: Mapping[str, Any]) -> dict[str, bool]:
    slot_manifest = context["slot_manifest"]
    inherited_manifest = context["inherited_manifest"]
    slot_directory = Path(context["slot_manifest_path"]).parent
    inherited_directory = Path(context["inherited_manifest_path"]).parent
    checks = {
        "cached_protocol": (
            sha256_file(context["cached_protocol_path"])
            == context["cached_protocol_sha256"]
        ),
        "capture_report": (
            sha256_file(context["capture_report_path"])
            == context["capture_report_sha256"]
        ),
        "v1_protocol": (
            sha256_file(context["v1_protocol_path"])
            == _EXPECTED_V1_PROTOCOL_SHA256
        ),
        "v1_trace_parity": (
            sha256_file(context["parity_path"]) == context["parity_sha256"]
        ),
        "slot_manifest": (
            sha256_file(context["slot_manifest_path"])
            == _EXPECTED_SLOT_MANIFEST_SHA256
        ),
        "inherited_manifest": (
            sha256_file(context["inherited_manifest_path"])
            == context["inherited_manifest_sha256"]
        ),
        "head_mass_result": (
            sha256_file(context["head_mass_result_path"])
            == _EXPECTED_HEAD_MASS_RESULT_SHA256
        ),
        "head_mass_protocol": (
            sha256_file(context["head_mass_protocol_path"])
            == _EXPECTED_HEAD_MASS_PROTOCOL_SHA256
        ),
        "joint_gamma_protocol": (
            sha256_file(context["joint_protocol_path"])
            == _EXPECTED_JOINT_PROTOCOL_SHA256
        ),
        "joint_gamma_result": (
            sha256_file(context["joint_result_path"])
            == _EXPECTED_JOINT_RESULT_SHA256
        ),
        "training_checkpoint": (
            sha256_file(context["training_checkpoint_path"])
            == context["training_checkpoint_sha256"]
        ),
        "selector_protocol": (
            sha256_file(context["selector_protocol_path"])
            == context["selector_protocol_sha256"]
        ),
        "corpus_manifest": (
            sha256_file(context["corpus_manifest_path"])
            == context["corpus_manifest_sha256"]
        ),
        "train_split": (
            sha256_file(context["train_path"]) == context["train_sha256"]
        ),
        "all_slot_shards": all(
            sha256_file(slot_directory / descriptor["file"])
            == descriptor["file_sha256"]
            for descriptor in slot_manifest["shards"]
        ),
        "all_inherited_shards": all(
            sha256_file(inherited_directory / descriptor["file"])
            == descriptor["file_sha256"]
            for descriptor in inherited_manifest["shards"]
        ),
        "output_projection_file": (
            sha256_file(context["projection_path"])
            == context["projection_file_sha256"]
        ),
        "source_inventory": (
            context["cached_protocol"]["source_sha256"] == _source_inventory()
        ),
        "confirmation_not_opened": True,
    }
    return checks


def _single_blas_thread_context() -> Any:
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - optional performance guard
        return nullcontext()
    return threadpool_limits(limits=1)


def _solve_cached_layer_task(
    task: tuple[str, int, int, int, int, float, float, float, float, float],
) -> tuple[str, int, slot.SlotSimplexOracleResult]:
    """Solve and replay one independent layer/arm inside a forked worker."""

    (
        arm,
        layer,
        row_batch_size,
        maximum_active_set_iterations,
        fallback_maximum_iterations,
        relative_gap_tolerance,
        absolute_gap_tolerance,
        working_set_tolerance,
        kkt_residual_tolerance,
        reduced_cost_tolerance,
    ) = task
    arrays = _FORK_ARRAYS
    output_projection = _FORK_OUTPUT_PROJECTION
    if arrays is None or output_projection is None:
        raise ValueError("cached V2 fork tensors are unavailable")
    if arm not in {"constructible", "optimistic"} or not 0 <= layer < slot._LAYERS:
        raise ValueError("cached V2 layer task is invalid")
    include_anchor = arm == "optimistic"
    rows_per_layer = slot._RECORDS * len(slot._READ_POSITIONS)
    component_count = (
        slot._OPTIMISTIC_COMPONENTS
        if include_anchor
        else slot._CONSTRUCTIBLE_COMPONENTS
    )
    coefficients = np.empty(
        (rows_per_layer, slot._QUERY_HEADS, component_count),
        dtype=np.float64,
    )
    target_energy = np.empty(rows_per_layer, dtype=np.float64)
    objective = np.empty(rows_per_layer, dtype=np.float64)
    objective_gap = np.empty(rows_per_layer, dtype=np.float64)
    direct_energy = np.empty(rows_per_layer, dtype=np.float64)
    iterations = np.empty(rows_per_layer, dtype=np.int32)
    converged = np.empty(rows_per_layer, dtype=bool)
    maxima = {
        "relative_gap": 0.0,
        "base": 0.0,
        "partition": 0.0,
        "episodic": 0.0,
        "mass": 0.0,
        "direct": 0.0,
    }
    replay_exact = True
    layer_weights = np.ascontiguousarray(
        output_projection[layer],
        dtype=np.float32,
    )

    with _single_blas_thread_context():
        for begin in range(0, rows_per_layer, row_batch_size):
            end = min(begin + row_batch_size, rows_per_layer)
            batch_arrays = slot._slice_layer_batch(
                arrays,
                layer=layer,
                begin=begin,
                end=end,
            )
            basis = slot.build_slot_basis(
                batch_arrays,
                query_heads=slot._QUERY_HEADS,
                slots=slot._SLOTS,
                include_exact_native_anchor=include_anchor,
            )
            projected = slot._project_head_basis(
                basis.correction_basis,
                layer_weights,
            )
            gram, linear, energy = slot._quadratic_from_projected_basis(
                projected,
                basis.target_residual,
            )
            solved = (
                active_solver.solve_product_simplex_least_squares_active_set(
                    gram,
                    linear,
                    energy,
                    basis.base_coefficients,
                    max_active_set_iterations=maximum_active_set_iterations,
                    fallback_max_iterations=fallback_maximum_iterations,
                    relative_tolerance=relative_gap_tolerance,
                    absolute_tolerance=absolute_gap_tolerance,
                    working_set_tolerance=working_set_tolerance,
                    kkt_residual_tolerance=kkt_residual_tolerance,
                    reduced_cost_tolerance=reduced_cost_tolerance,
                )
            )
            replay = (
                active_solver.solve_product_simplex_least_squares_active_set(
                    gram,
                    linear,
                    energy,
                    basis.base_coefficients,
                    max_active_set_iterations=maximum_active_set_iterations,
                    fallback_max_iterations=fallback_maximum_iterations,
                    relative_tolerance=relative_gap_tolerance,
                    absolute_tolerance=absolute_gap_tolerance,
                    working_set_tolerance=working_set_tolerance,
                    kkt_residual_tolerance=kkt_residual_tolerance,
                    reduced_cost_tolerance=reduced_cost_tolerance,
                )
            )
            replay_exact = replay_exact and all(
                (
                    np.array_equal(solved.coefficients, replay.coefficients),
                    np.array_equal(solved.objective, replay.objective),
                    np.array_equal(
                        solved.objective_gap_upper_bound,
                        replay.objective_gap_upper_bound,
                    ),
                    solved.iterations == replay.iterations,
                    solved.converged == replay.converged,
                    np.array_equal(solved.row_converged, replay.row_converged),
                )
            )
            direct = slot._direct_error_energy(
                basis,
                solved.coefficients,
                layer_weights,
            )
            destination = slice(begin, end)
            coefficients[destination] = solved.coefficients
            target_energy[destination] = energy
            objective[destination] = solved.objective
            objective_gap[destination] = solved.objective_gap_upper_bound
            direct_energy[destination] = direct
            iterations[destination] = solved.iterations
            converged[destination] = solved.row_converged
            maxima["relative_gap"] = max(
                maxima["relative_gap"],
                float(solved.max_relative_gap),
            )
            maxima["base"] = max(
                maxima["base"],
                basis.base_reconstruction_max_abs,
            )
            maxima["partition"] = max(
                maxima["partition"],
                basis.traced_partition_reconstruction_max_abs,
            )
            maxima["episodic"] = max(
                maxima["episodic"],
                basis.episodic_component_reconstruction_max_abs,
            )
            maxima["mass"] = max(
                maxima["mass"],
                basis.mass_partition_max_abs,
            )
            maxima["direct"] = max(
                maxima["direct"],
                float(np.max(np.abs(direct - solved.objective))),
            )
    if not replay_exact:
        raise ValueError("cached V2 deterministic layer replay changed")
    return (
        arm,
        layer,
        slot.SlotSimplexOracleResult(
            coefficients=coefficients,
            target_energy=target_energy,
            objective=objective,
            objective_gap_upper_bound=objective_gap,
            direct_error_energy=direct_energy,
            iterations=iterations,
            converged=converged,
            maximum_relative_objective_gap=maxima["relative_gap"],
            base_reconstruction_max_abs=maxima["base"],
            traced_partition_reconstruction_max_abs=maxima["partition"],
            episodic_component_reconstruction_max_abs=maxima["episodic"],
            mass_partition_max_abs=maxima["mass"],
            quadratic_direct_error_energy_max_abs=maxima["direct"],
            deterministic_replay_exact=True,
            batch_shape=(slot._RECORDS, len(slot._READ_POSITIONS), 1),
        ),
    )


def _aggregate_layer_results(
    results: Mapping[int, slot.SlotSimplexOracleResult],
    *,
    include_exact_native_anchor: bool,
) -> slot.SlotSimplexOracleResult:
    if set(results) != set(range(slot._LAYERS)):
        raise ValueError("cached V2 layer result set changed")
    rows_per_layer = slot._RECORDS * len(slot._READ_POSITIONS)
    total_rows = rows_per_layer * slot._LAYERS
    component_count = (
        slot._OPTIMISTIC_COMPONENTS
        if include_exact_native_anchor
        else slot._CONSTRUCTIBLE_COMPONENTS
    )
    coefficients = np.empty(
        (total_rows, slot._QUERY_HEADS, component_count),
        dtype=np.float64,
    )
    target_energy = np.empty(total_rows, dtype=np.float64)
    objective = np.empty(total_rows, dtype=np.float64)
    objective_gap = np.empty(total_rows, dtype=np.float64)
    direct_energy = np.empty(total_rows, dtype=np.float64)
    iterations = np.empty(total_rows, dtype=np.int32)
    converged = np.empty(total_rows, dtype=bool)
    for layer in range(slot._LAYERS):
        result = results[layer]
        if (
            result.coefficients.shape
            != (rows_per_layer, slot._QUERY_HEADS, component_count)
            or result.batch_shape
            != (slot._RECORDS, len(slot._READ_POSITIONS), 1)
        ):
            raise ValueError("cached V2 layer result shape changed")
        destination = (
            np.arange(rows_per_layer, dtype=np.int64) * slot._LAYERS + layer
        )
        coefficients[destination] = result.coefficients
        target_energy[destination] = result.target_energy
        objective[destination] = result.objective
        objective_gap[destination] = result.objective_gap_upper_bound
        direct_energy[destination] = result.direct_error_energy
        iterations[destination] = result.iterations
        converged[destination] = result.converged
    rows = list(results.values())
    return slot.SlotSimplexOracleResult(
        coefficients=coefficients,
        target_energy=target_energy,
        objective=objective,
        objective_gap_upper_bound=objective_gap,
        direct_error_energy=direct_energy,
        iterations=iterations,
        converged=converged,
        maximum_relative_objective_gap=max(
            row.maximum_relative_objective_gap for row in rows
        ),
        base_reconstruction_max_abs=max(
            row.base_reconstruction_max_abs for row in rows
        ),
        traced_partition_reconstruction_max_abs=max(
            row.traced_partition_reconstruction_max_abs for row in rows
        ),
        episodic_component_reconstruction_max_abs=max(
            row.episodic_component_reconstruction_max_abs for row in rows
        ),
        mass_partition_max_abs=max(
            row.mass_partition_max_abs for row in rows
        ),
        quadratic_direct_error_energy_max_abs=max(
            row.quadratic_direct_error_energy_max_abs for row in rows
        ),
        deterministic_replay_exact=all(
            row.deterministic_replay_exact for row in rows
        ),
        batch_shape=(
            slot._RECORDS,
            len(slot._READ_POSITIONS),
            slot._LAYERS,
        ),
    )


def _run_parallel_two_arms(
    arrays: Mapping[str, np.ndarray],
    output_projection: np.ndarray,
    *,
    workers: int,
    row_batch_size: int,
    maximum_active_set_iterations: int = _MAXIMUM_ACTIVE_SET_ITERATIONS,
    fallback_maximum_iterations: int = _FALLBACK_MAXIMUM_ITERATIONS,
    relative_gap_tolerance: float,
    absolute_gap_tolerance: float = _ABSOLUTE_GAP_TOLERANCE,
    working_set_tolerance: float = _WORKING_SET_TOLERANCE,
    kkt_residual_tolerance: float = _KKT_RESIDUAL_TOLERANCE,
    reduced_cost_tolerance: float = _REDUCED_COST_TOLERANCE,
) -> tuple[slot.SlotSimplexOracleResult, slot.SlotSimplexOracleResult]:
    """Run all independent layer/arm tasks in a forked process pool."""

    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError("cached V2 requires the frozen fork process backend")
    if not 0 < workers <= slot._LAYERS:
        raise ValueError("cached V2 worker count is invalid")
    installed = {
        name: np.ascontiguousarray(value)
        for name, value in arrays.items()
    }
    projection = np.ascontiguousarray(output_projection, dtype=np.float32)
    for value in (*installed.values(), projection):
        value.flags.writeable = False
    tasks = [
        (
            arm,
            layer,
            row_batch_size,
            maximum_active_set_iterations,
            fallback_maximum_iterations,
            relative_gap_tolerance,
            absolute_gap_tolerance,
            working_set_tolerance,
            kkt_residual_tolerance,
            reduced_cost_tolerance,
        )
        for arm in ("constructible", "optimistic")
        for layer in range(slot._LAYERS)
    ]

    global _FORK_ARRAYS, _FORK_OUTPUT_PROJECTION
    _FORK_ARRAYS = installed
    _FORK_OUTPUT_PROJECTION = projection
    try:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
        ) as executor:
            completed = list(
                executor.map(
                    _solve_cached_layer_task,
                    tasks,
                    chunksize=1,
                )
            )
    finally:
        _FORK_ARRAYS = None
        _FORK_OUTPUT_PROJECTION = None

    by_arm: dict[str, dict[int, slot.SlotSimplexOracleResult]] = {
        "constructible": {},
        "optimistic": {},
    }
    for arm, layer, result in completed:
        if arm not in by_arm or layer in by_arm[arm]:
            raise ValueError("cached V2 duplicate layer result")
        by_arm[arm][layer] = result
    return (
        _aggregate_layer_results(
            by_arm["constructible"],
            include_exact_native_anchor=False,
        ),
        _aggregate_layer_results(
            by_arm["optimistic"],
            include_exact_native_anchor=True,
        ),
    )


def _decision_next_step(
    frozen: Mapping[str, Any],
    *,
    feasible_passed: bool,
    decisive_failure: bool,
) -> str:
    if feasible_passed:
        return str(frozen["authorized_next_step_on_feasible_pass"])
    if decisive_failure:
        return str(frozen["failure_interpretation"])
    return (
        "improve or extend the certified cached solve until the optimistic "
        "hull is qualified; an inconclusive solve does not authorize the "
        "frozen failure interpretation"
    )


def solve_cached_slot_simplex(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    """Run both frozen oracle arms using authenticated cached tensors only."""

    output = _new_output(out, "cached V2 result")
    started = time.perf_counter()
    _progress("authenticating protocol, capture, shards, and BF16 projection")
    _source, frozen, capture, context = _authenticate_cached_protocol(
        protocol,
        protocol_sha256,
        stack_arrays=True,
    )
    arrays = context["stacked_arrays"]
    projection = context["output_projection"]
    if arrays is None:
        raise ValueError("cached V2 stacked tensors are unavailable")

    method = frozen["oracle_method"]
    parallel = frozen["parallel_execution"]
    _progress(
        "running both cached simplex arms as independent layer tasks "
        f"across {parallel['workers']} forked workers"
    )
    constructible, optimistic = _run_parallel_two_arms(
        arrays,
        projection,
        workers=parallel["workers"],
        row_batch_size=method["row_batch_size"],
        maximum_active_set_iterations=method[
            "maximum_active_set_iterations"
        ],
        fallback_maximum_iterations=method[
            "fallback_maximum_iterations"
        ],
        relative_gap_tolerance=method["relative_objective_gap_target"],
        absolute_gap_tolerance=method["absolute_objective_gap_target"],
        working_set_tolerance=method["working_set_tolerance"],
        kkt_residual_tolerance=method["kkt_residual_tolerance"],
        reduced_cost_tolerance=method["reduced_cost_tolerance"],
    )
    constructible_report = slot._summarize_oracle_arm(
        constructible,
        component_count=slot._CONSTRUCTIBLE_COMPONENTS,
        arm="constructible_regular_plus_eight_slots",
    )
    if not np.array_equal(constructible.target_energy, optimistic.target_energy):
        raise ValueError("cached V2 arm target energies changed")
    optimistic_report = slot._summarize_oracle_arm(
        optimistic,
        component_count=slot._OPTIMISTIC_COMPONENTS,
        arm="optimistic_exact_native_anchor_hull",
    )
    feasible_passed = bool(constructible_report["feasible_gate_passed"])
    optimistic_passed = bool(optimistic_report["optimistic_gate_passed"])
    if feasible_passed and not optimistic_passed:
        raise ValueError("cached V2 constructible pass exceeded optimistic hull")
    decisive_failure = bool(
        optimistic_report["qualification"]["passed"] and not optimistic_passed
    )
    status = (
        "train_episodic_slot_simplex_cached_v2_gate_passed"
        if feasible_passed
        else (
            "train_episodic_slot_simplex_cached_v2_gate_failed"
            if decisive_failure
            else "train_episodic_slot_simplex_cached_v2_gate_inconclusive"
        )
    )
    post = _post_solve_authentication(context)
    if not all(post.values()):
        raise ValueError("cached V2 post-solve authentication failed")

    oracle = {
        "two_continuous_capacity_arms": True,
        "shared_solver_contract": {
            "maximum_active_set_iterations": method[
                "maximum_active_set_iterations"
            ],
            "fallback_maximum_iterations": method[
                "fallback_maximum_iterations"
            ],
            "relative_objective_gap_target": method[
                "relative_objective_gap_target"
            ],
            "absolute_objective_gap_target": method[
                "absolute_objective_gap_target"
            ],
            "working_set_tolerance": method["working_set_tolerance"],
            "kkt_residual_tolerance": method["kkt_residual_tolerance"],
            "reduced_cost_tolerance": method["reduced_cost_tolerance"],
            "row_batch_size": method["row_batch_size"],
            "workers": parallel["workers"],
            "parallel_task_unit": parallel["task_unit"],
        },
        "constructible_arm": constructible_report,
        "optimistic_hull_arm": optimistic_report,
        "constructible_feasible_gate_passed": feasible_passed,
        "optimistic_hull_certified_gate_passed": optimistic_passed,
        "decisive_failure": decisive_failure,
    }
    report = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _RESULT_EXPERIMENT,
        "status": status,
        "protocol": {
            "path": str(context["cached_protocol_path"]),
            "sha256": context["cached_protocol_sha256"],
        },
        "capture_report": {
            "path": str(context["capture_report_path"]),
            "sha256": context["capture_report_sha256"],
            "capture_rows_sha256": capture["capture_rows_sha256"],
        },
        "scope": {
            "split": "train",
            "records": slot._RECORDS,
            "trace_positions_per_record": len(slot._READ_POSITIONS),
            "cached_same_state_capacity_evidence_only": True,
            "causal_rollout": False,
            "semantic_or_M3_gate_passed": False,
            "development_outcomes_used": False,
            "native_execution_performed": False,
            "confirmation_split_opened": False,
        },
        "cache_authentication": {
            "native_rerun_performed": False,
            "partial_v1_solver_output_used": False,
            "reset_replay_evidence": dict(frozen["reset_replay_evidence"]),
            "output_projection": dict(frozen["output_projection"]),
        },
        "oracle": oracle,
        "resource_contract": dict(frozen["resource_contract"]),
        "decision": {
            "train_slot_simplex_capacity_gate_passed": feasible_passed,
            "certified_optimistic_gate_passed": optimistic_passed,
            "failure_is_decisive": decisive_failure,
            "semantic_or_M3_gate_passed": False,
            "native_causal_integration_authorized": False,
            "development_authorized": False,
            "confirmation_authorized": False,
            "train_only_causal_slot_selector_authorized": feasible_passed,
            "next_step": _decision_next_step(
                frozen,
                feasible_passed=feasible_passed,
                decisive_failure=decisive_failure,
            ),
        },
        "post_run_authentication": post,
        "confirmation_split_opened": False,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output, report)
    _progress(f"cached V2 result written to {output}")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticated cached V2 slot-simplex handoff",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    authenticate = commands.add_parser("authenticate-cache")
    authenticate.add_argument("--v1-protocol", required=True)
    authenticate.add_argument("--v1-protocol-sha256", required=True)
    authenticate.add_argument("--slot-manifest", required=True)
    authenticate.add_argument("--slot-manifest-sha256", required=True)
    authenticate.add_argument("--head-mass-result", required=True)
    authenticate.add_argument("--head-mass-result-sha256", required=True)
    authenticate.add_argument("--out", required=True)
    freeze = commands.add_parser("freeze-cached")
    freeze.add_argument("--capture-report", required=True)
    freeze.add_argument("--capture-report-sha256", required=True)
    freeze.add_argument("--workers", type=int, default=_default_workers())
    freeze.add_argument(
        "--row-batch-size",
        type=int,
        default=_DEFAULT_ROW_BATCH_SIZE,
    )
    freeze.add_argument("--out", required=True)
    solve = commands.add_parser("solve-cached")
    solve.add_argument("--protocol", required=True)
    solve.add_argument("--protocol-sha256", required=True)
    solve.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "authenticate-cache":
        value = authenticate_cached_capture(
            v1_protocol=args.v1_protocol,
            v1_protocol_sha256=args.v1_protocol_sha256,
            slot_manifest=args.slot_manifest,
            slot_manifest_sha256=args.slot_manifest_sha256,
            head_mass_result=args.head_mass_result,
            head_mass_result_sha256=args.head_mass_result_sha256,
            out=args.out,
        )
    elif args.command == "freeze-cached":
        value = freeze_cached_protocol(
            capture_report=args.capture_report,
            capture_report_sha256=args.capture_report_sha256,
            out=args.out,
            workers=args.workers,
            row_batch_size=args.row_batch_size,
        )
    elif args.command == "solve-cached":
        value = solve_cached_slot_simplex(
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            out=args.out,
        )
    else:  # pragma: no cover - argparse enforces commands
        raise AssertionError(f"unknown command: {args.command}")
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
