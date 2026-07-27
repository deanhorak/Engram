"""Source-independent package compiler for the pinned native BitNet track."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from engram import __version__
from engram.models.native_bitnet import (
    OFFICIAL_NATIVE_BITNET_REPO,
    OFFICIAL_NATIVE_BITNET_REVISION,
    OFFICIAL_NATIVE_BITNET_WEIGHT_SHA256,
    NativeBitNetValidationError,
    _resolve_full_source,
    load_native_bitnet_artifact,
)
from engram.utils import atomic_json, sha256_file

NATIVE_BITNET_PACKAGE_FORMAT = "engram-native-bitnet"
NATIVE_BITNET_PACKAGE_VERSION = 1
NATIVE_BITNET_ARTIFACT_PATH = PurePosixPath("mlp/model.bitnet-records.bin")
NATIVE_BITNET_DIP_INDEX_PATH = PurePosixPath(
    "mlp/model.bitnet-dip-index.bin"
)
NATIVE_BITNET_NON_MLP_PATH = PurePosixPath("transformer/non_mlp.safetensors")
NATIVE_BITNET_CONTROLLER_PATH = PurePosixPath("controller")
NATIVE_BITNET_DIP_OPERATOR = "native_bitnet_dynamic_input_pruning_v2"
NATIVE_BITNET_ATTENTION_OPERATOR = "native_streaming_w16_c8_k4_sinks2"
NATIVE_BITNET_M2_PACKAGE_MANIFEST_SHA256 = (
    "cddd96a01ff03bd565c108ab58925e7463aad35ebd8b1cc315eb7b050030cd35"
)
NATIVE_BITNET_M2_BASE_ARTIFACT_SHA256 = (
    "4fcf598af4346d5391ba428e32ba1629daae2768b73ab6bf872d3f9fb300ab55"
)
NATIVE_BITNET_M2_COORDINATE_INDEX_SHA256 = (
    "b98ce4e46c8ae67d9d92d4d13f5de3d4fe45ef2c76400bd9d50be08b2bd60e15"
)
NATIVE_BITNET_M2_POLICY_MANIFEST_SHA256 = (
    "c572754e597a760bc5ea6ba337bdaaf092e4ae1d5b5e90b6a2a14cbfbea3768e"
)
NATIVE_BITNET_M2_ADJUDICATION_SHA256 = (
    "ebb5ca9568387ffd3c5b187f8e17f3ce706aaee86f4bbe9e314bf1760a7da5cc"
)
NATIVE_BITNET_DIP_DERIVED_MANIFEST_SHA256 = (
    "707bbe069ef6892ce9bfe98258f3289e28af15a400922e950c4386f56dd26926"
)
NATIVE_BITNET_DIP_DERIVED_MANIFEST_BYTES = 5787

# This is the checked-in trust root for promotion of a final semantic-memory
# result into an executable package.  A caller-provided document and its
# caller-provided hash are not, by themselves, an authorization.
_APPROVED_NATIVE_BITNET_M2_ADJUDICATIONS = {
    NATIVE_BITNET_M2_ADJUDICATION_SHA256: {
        "package_manifest": NATIVE_BITNET_M2_PACKAGE_MANIFEST_SHA256,
        "base_artifact": NATIVE_BITNET_M2_BASE_ARTIFACT_SHA256,
        "coordinate_index": NATIVE_BITNET_M2_COORDINATE_INDEX_SHA256,
        "policy_manifest": NATIVE_BITNET_M2_POLICY_MANIFEST_SHA256,
    },
}

_REQUIRED_M2_ADJUDICATION_CHECKS = (
    ("candidate_recall", "each_layer_mean_recall"),
    ("candidate_recall", "global_micro_membership_recall"),
    ("evaluator_attestations", "active_budget_passed"),
    ("evaluator_attestations", "candidate_recall_passed"),
    ("evaluator_attestations", "overall_gate_passed"),
    ("evaluator_attestations", "protocol_qualifying"),
    ("evaluator_attestations", "quality_passed"),
    ("evaluator_attestations", "systems_evidence_passed"),
    ("evaluator_attestations", "traffic_passed"),
    ("evidence_seal", "attempt_identity"),
    ("evidence_seal", "authorization"),
    ("evidence_seal", "opened_marker"),
    ("evidence_seal", "original_result"),
    ("evidence_seal", "protocol_policy_and_dataset"),
    ("evidence_seal", "raw_evaluator_report"),
    ("evidence_seal", "schema"),
    ("original_attempt", "attempt_identity_matches"),
    ("original_attempt", "authorization_matches"),
    ("original_attempt", "dataset_matches"),
    ("original_attempt", "implementation_matches"),
    ("original_attempt", "marker_is_terminal_nonreusable_error"),
    ("original_attempt", "protocol_matches"),
    ("original_attempt", "raw_report_is_separate_and_present"),
    ("original_attempt", "result_is_bound_by_marker"),
    ("original_attempt", "result_is_consumed_execution_error"),
    ("practical", "cpu_only"),
    ("practical", "global_complete_physical_cold_traffic"),
    ("practical", "maximum_token_complete_physical_cold_traffic"),
    ("practical", "mean_active_record_fraction"),
    ("practical", "no_dense_mlp_fallback"),
    ("practical", "worst_layer_complete_physical_cold_traffic"),
    ("protocol_token_identities", "canonical_hash_algorithm"),
    ("protocol_token_identities", "full_sequence_hashes"),
    ("protocol_token_identities", "full_token_lengths"),
    ("protocol_token_identities", "scored_prefix_length"),
    ("quality", "final_hidden_relative_l2"),
    ("quality", "nll_delta"),
    ("quality", "teacher_student_kl"),
    ("quality", "top1_agreement"),
    ("system_evidence", "cache_line_honest_traffic_accounting"),
    ("system_evidence", "exact_evidence_shape"),
    ("system_evidence", "measured_cpu_latency"),
    ("system_evidence", "python_native_numerical_parity"),
    ("system_evidence", "serialized_index_reload"),
    ("system_evidence", "timed_debug_parity"),
)


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_payload(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _write_non_mlp_weights(source: Path, destination: Path) -> dict[str, Any]:
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise NativeBitNetValidationError(
            "install engram-lm[conversion] to compile native BitNet packages"
        ) from exc

    tensors = {}
    source_tensors = 0
    excluded_mlp_tensors = 0
    with safe_open(source, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            source_tensors += 1
            if ".mlp." in name:
                excluded_mlp_tensors += 1
                continue
            tensor = handle.get_tensor(name)
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            tensors[name] = tensor.contiguous()
    if not tensors or not any(name == "model.embed_tokens.weight" for name in tensors):
        raise NativeBitNetValidationError(
            "native BitNet non-MLP tensor set is incomplete"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        save_file(tensors, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
        tensors.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "source_tensors": source_tensors,
        "packaged_tensors": source_tensors - excluded_mlp_tensors,
        "excluded_mlp_tensors": excluded_mlp_tensors,
        "packaged_bytes": destination.stat().st_size,
    }


def _inventory(root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != root / "manifest.json":
            relative = path.relative_to(root).as_posix()
            files[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return files


def _package_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str):
        raise NativeBitNetValidationError(
            f"unsafe native BitNet package path: {relative!r}"
        )
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise NativeBitNetValidationError(
            f"unsafe native BitNet package path: {relative!r}"
        )
    return root.joinpath(*pure.parts)


def _verified_package_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeBitNetValidationError(
            f"cannot read native BitNet package manifest: {exc}"
        ) from exc
    if (
        manifest.get("format") != NATIVE_BITNET_PACKAGE_FORMAT
        or manifest.get("version") != NATIVE_BITNET_PACKAGE_VERSION
    ):
        raise NativeBitNetValidationError("not a supported native BitNet package")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise NativeBitNetValidationError(
            "native BitNet package has no file inventory"
        )
    for relative, descriptor in files.items():
        if not isinstance(relative, str) or not isinstance(descriptor, dict):
            raise NativeBitNetValidationError(
                "native BitNet package inventory is malformed"
            )
        path = _package_path(root, relative)
        if path.is_symlink() or not path.is_file():
            raise NativeBitNetValidationError(
                f"native BitNet package is corrupt: {relative}"
            )
    actual_files = _inventory(root)
    if actual_files != files:
        raise NativeBitNetValidationError(
            "native BitNet package inventory is not exact"
        )
    return manifest


def _load_passing_m2_adjudication(
    path: Path,
    *,
    expected_sha256: str,
    package_manifest_sha256: str,
    base_artifact_sha256: str,
    coordinate_index_sha256: str,
    policy_manifest_sha256: str,
) -> dict[str, Any]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256.lower():
        raise NativeBitNetValidationError(
            "native BitNet M2 adjudication SHA-256 mismatch"
        )
    trusted_inputs = _APPROVED_NATIVE_BITNET_M2_ADJUDICATIONS.get(
        actual_sha256
    )
    expected_inputs = {
        "package_manifest": package_manifest_sha256,
        "base_artifact": base_artifact_sha256,
        "coordinate_index": coordinate_index_sha256,
        "policy_manifest": policy_manifest_sha256,
    }
    if trusted_inputs != expected_inputs:
        raise NativeBitNetValidationError(
            "native BitNet M2 adjudication is not in the checked-in "
            "promotion authorization registry"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeBitNetValidationError(
            f"cannot read native BitNet M2 adjudication: {exc}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("format")
        != "engram-native-bitnet-m2-final-adjudication"
        or payload.get("version") != 1
        or payload.get("status") != "pass"
        or payload.get("milestone_2_passed") is not True
        or payload.get("decision")
        != "milestone_2_semantic_gate_passed_by_postmortem_adjudication"
    ):
        raise NativeBitNetValidationError(
            "native BitNet M2 adjudication is not a passing decision"
        )
    adjudication = payload.get("adjudication")
    if (
        not isinstance(adjudication, dict)
        or adjudication.get("model_or_evaluator_executed") is not False
        or adjudication.get("original_result_rewritten") is not False
        or adjudication.get("holdout_reused_for_configuration") is not False
    ):
        raise NativeBitNetValidationError(
            "native BitNet M2 adjudication contract is invalid"
        )
    checks = payload.get("checks")
    if not isinstance(checks, dict) or any(
        not isinstance(checks.get(group), dict)
        or checks[group].get(name) is not True
        for group, name in _REQUIRED_M2_ADJUDICATION_CHECKS
    ):
        raise NativeBitNetValidationError(
            "native BitNet M2 adjudication is missing a required passing check"
        )
    inputs = payload.get("input_sha256")
    if not isinstance(inputs, dict) or any(
        inputs.get(name) != expected
        for name, expected in expected_inputs.items()
    ):
        raise NativeBitNetValidationError(
            "native BitNet M2 adjudication inputs do not match the package"
        )
    return payload


def _derived_semantic_memory_manifest(
    source_manifest: dict[str, Any],
    *,
    source_manifest_sha256: str,
    base_artifact_sha256: str,
    coordinate_index_sha256: str,
    coordinate_index_bytes: int,
    policy_manifest_sha256: str,
    adjudication_sha256: str,
    adjudication_decision: str,
) -> dict[str, Any]:
    """Return the one exact manifest permitted for a DIP derivation."""

    if (
        "semantic_memory" in source_manifest
        or NATIVE_BITNET_DIP_INDEX_PATH.as_posix()
        in source_manifest.get("files", {})
    ):
        raise NativeBitNetValidationError(
            "native BitNet semantic-memory source is already derived"
        )
    manifest = copy.deepcopy(source_manifest)
    manifest["semantic_memory"] = {
        "operator": NATIVE_BITNET_DIP_OPERATOR,
        "runtime_scope": "native_token_runtime",
        "path": NATIVE_BITNET_DIP_INDEX_PATH.as_posix(),
        "format": "engram-native-bitnet-dip-index",
        "version": 2,
        "sha256": coordinate_index_sha256,
        "serialized_bytes": coordinate_index_bytes,
        "source_artifact_sha256": base_artifact_sha256,
        "source_package_manifest_sha256": source_manifest_sha256,
        "runtime_policy": "embedded_authenticated_layer_headers",
        "policy_manifest_sha256": policy_manifest_sha256,
        "adjudication_sha256": adjudication_sha256,
        "adjudication_decision": adjudication_decision,
        "all_mlp_layers_substituted": True,
        "dense_fallback": False,
        "cpu_only": True,
        "traffic_accounting": "modelled_cache_line_v2",
    }
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise NativeBitNetValidationError(
            "native BitNet package runtime descriptor is malformed"
        )
    runtime["mlp_mode"] = NATIVE_BITNET_DIP_OPERATOR
    runtime["attention_mode"] = NATIVE_BITNET_ATTENTION_OPERATOR
    runtime["attention_policy"] = {
        "local_window": 16,
        "older_candidates": 8,
        "older_top_k": 4,
        "sink_tokens": 2,
    }
    files = copy.deepcopy(source_manifest["files"])
    files[NATIVE_BITNET_DIP_INDEX_PATH.as_posix()] = {
        "bytes": coordinate_index_bytes,
        "sha256": coordinate_index_sha256,
    }
    manifest["files"] = files
    return manifest


def _validate_semantic_memory_descriptor(
    root: Path,
    manifest: dict[str, Any],
) -> None:
    from engram.semantic.native_bitnet_dip_index import (
        NATIVE_BITNET_DIP_INDEX_FORMAT,
        NATIVE_BITNET_DIP_INDEX_VERSION,
        load_native_bitnet_dip_index,
    )

    descriptor = manifest.get("semantic_memory")
    runtime = manifest.get("runtime")
    if not isinstance(descriptor, dict) or not isinstance(runtime, dict):
        raise NativeBitNetValidationError(
            "native BitNet DIP package has no semantic-memory contract"
        )
    if (
        descriptor.get("operator") != NATIVE_BITNET_DIP_OPERATOR
        or descriptor.get("runtime_scope") != "native_token_runtime"
        or descriptor.get("format") != NATIVE_BITNET_DIP_INDEX_FORMAT
        or descriptor.get("version") != NATIVE_BITNET_DIP_INDEX_VERSION
        or descriptor.get("runtime_policy")
        != "embedded_authenticated_layer_headers"
        or descriptor.get("all_mlp_layers_substituted") is not True
        or descriptor.get("dense_fallback") is not False
        or descriptor.get("cpu_only") is not True
        or runtime.get("mlp_mode") != NATIVE_BITNET_DIP_OPERATOR
        or runtime.get("attention_mode") != NATIVE_BITNET_ATTENTION_OPERATOR
        or runtime.get("attention_policy")
        != {
            "local_window": 16,
            "older_candidates": 8,
            "older_top_k": 4,
            "sink_tokens": 2,
        }
    ):
        raise NativeBitNetValidationError(
            "native BitNet semantic-memory descriptor is unsupported"
        )
    index_path = _package_path(root, descriptor.get("path", ""))
    if (
        index_path.is_symlink()
        or not index_path.is_file()
        or index_path.stat().st_size != descriptor.get("serialized_bytes")
        or sha256_file(index_path) != descriptor.get("sha256")
    ):
        raise NativeBitNetValidationError(
            "native BitNet DIP package index is corrupt"
        )
    mlp = manifest.get("mlp")
    model = manifest.get("model")
    if not isinstance(mlp, dict) or not isinstance(model, dict):
        raise NativeBitNetValidationError(
            "native BitNet DIP package model metadata is missing"
        )
    if descriptor.get("source_artifact_sha256") != mlp.get("sha256"):
        raise NativeBitNetValidationError(
            "native BitNet DIP source-artifact descriptor mismatch"
        )
    with load_native_bitnet_dip_index(index_path) as index:
        if (
            index.payload_sha256 != descriptor.get("sha256")
            or index.source_artifact_sha256 != mlp.get("sha256")
            or index.hidden_size != model.get("hidden_size")
            or index.intermediate_size != model.get("intermediate_size")
            or len(index.layers) != model.get("num_hidden_layers")
        ):
            raise NativeBitNetValidationError(
                "native BitNet DIP index dimensions or bindings mismatch"
            )


def install_native_bitnet_controller(
    package: str | Path,
    controller: str | Path,
) -> Path:
    """Authenticate and install a zero-correction schema-v3 controller."""

    from engram.controller import FactorizedRecurrentController

    root = Path(package).resolve()
    manifest_path = root / "manifest.json"
    manifest = _verified_package_manifest(root)

    source = Path(controller).resolve()
    loaded = FactorizedRecurrentController.load(source)
    model = manifest.get("model", {})
    if (
        loaded.metadata().get("schema_version") != 3
        or not loaded.has_operator_residual
        or loaded.state_dim != int(model.get("hidden_size", -1))
        or loaded.num_stages != int(model.get("num_hidden_layers", -1))
        or loaded.input_dim != 3 * loaded.state_dim
        or bool((loaded.step_scale != 0.0).any())
    ):
        raise NativeBitNetValidationError(
            "controller is not a compatible zero-correction schema-v3 artifact"
        )
    destination = root.joinpath(*NATIVE_BITNET_CONTROLLER_PATH.parts)
    expected_files = {
        "metadata.json",
        *(f"{name}.npy" for name in loaded.tensors()),
    }
    if destination.exists():
        existing = {path.name for path in destination.iterdir() if path.is_file()}
        if existing != expected_files or any(
            sha256_file(destination / name) != sha256_file(source / name)
            for name in expected_files
        ):
            raise NativeBitNetValidationError(
                "native BitNet package already has a different controller"
            )
    else:
        destination.mkdir(parents=True)
        for name in sorted(expected_files):
            _copy_atomic(source / name, destination / name)

    metadata_sha256 = sha256_file(destination / "metadata.json")
    manifest["controller"] = {
        "path": NATIVE_BITNET_CONTROLLER_PATH.as_posix(),
        "format": loaded.metadata()["format"],
        "schema_version": loaded.metadata()["schema_version"],
        "operator": loaded.metadata()["operator"],
        "metadata_sha256": metadata_sha256,
        "serialized_bytes": loaded.serialized_bytes,
        "correction_enabled": False,
    }
    manifest["files"] = _inventory(root)
    atomic_json(manifest_path, manifest)
    return root


def install_native_bitnet_semantic_memory(
    package: str | Path,
    coordinate_index: str | Path,
    policy_manifest: str | Path,
    adjudication: str | Path,
    out: str | Path,
    *,
    coordinate_index_sha256: str,
    policy_manifest_sha256: str,
    adjudication_sha256: str,
) -> Path:
    """Create a derived package with the adjudicated DIP v2 index installed.

    The policy-bound source package is immutable evidence and is never
    modified.  The derived package embeds the authenticated runtime policy in
    the v2 coordinate index and records the exact policy/adjudication hashes
    that authorized promotion.
    """

    from engram.semantic.native_bitnet_dip_index import (
        load_native_bitnet_dip_index,
    )
    from engram.semantic.native_bitnet_dip_policy_manifest import (
        load_native_bitnet_dip_policy_manifest,
    )

    source = Path(package).resolve()
    target = Path(out).resolve()
    if target == source or target.is_relative_to(source):
        raise NativeBitNetValidationError(
            "native BitNet semantic memory must be installed outside the "
            "source package"
        )
    policy_path = Path(policy_manifest).resolve()
    index_path = Path(coordinate_index).resolve()
    adjudication_path = Path(adjudication).resolve()
    expected_index_sha256 = coordinate_index_sha256.lower()
    expected_policy_sha256 = policy_manifest_sha256.lower()
    expected_adjudication_sha256 = adjudication_sha256.lower()

    loaded_policy = load_native_bitnet_dip_policy_manifest(
        policy_path,
        expected_sha256=expected_policy_sha256,
    )
    bindings = loaded_policy.payload["bindings"]
    bound_manifest = bindings["package_manifest"]
    bound_base = bindings["base_record_artifact"]
    bound_index = bindings["coordinate_index"]
    frozen_package = Path(bound_manifest["path"]).resolve().parent
    if target == frozen_package or target.is_relative_to(frozen_package):
        raise NativeBitNetValidationError(
            "refusing to write inside the policy-bound native BitNet package"
        )
    source_manifest = _verified_package_manifest(source)
    source_manifest_sha256 = sha256_file(source / "manifest.json")
    if (
        source_manifest_sha256 != bound_manifest.get("sha256")
        or expected_index_sha256 != bound_index.get("sha256")
        or index_path.stat().st_size != bound_index.get("bytes")
        or sha256_file(index_path) != expected_index_sha256
        or source_manifest.get("mlp", {}).get("sha256")
        != bound_base.get("sha256")
    ):
        raise NativeBitNetValidationError(
            "native BitNet DIP inputs differ from the frozen policy bindings"
        )
    model = source_manifest.get("model", {})
    with load_native_bitnet_dip_index(index_path) as index:
        if (
            index.payload_sha256 != expected_index_sha256
            or index.source_artifact_sha256 != bound_base.get("sha256")
            or index.hidden_size != model.get("hidden_size")
            or index.intermediate_size != model.get("intermediate_size")
            or len(index.layers) != model.get("num_hidden_layers")
            or len(index.layers) != len(loaded_policy.layers)
        ):
            raise NativeBitNetValidationError(
                "native BitNet DIP index does not match the package or policy"
            )
    approval = _load_passing_m2_adjudication(
        adjudication_path,
        expected_sha256=expected_adjudication_sha256,
        package_manifest_sha256=source_manifest_sha256,
        base_artifact_sha256=str(bound_base["sha256"]),
        coordinate_index_sha256=expected_index_sha256,
        policy_manifest_sha256=expected_policy_sha256,
    )
    expected_manifest = _derived_semantic_memory_manifest(
        source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        base_artifact_sha256=str(bound_base["sha256"]),
        coordinate_index_sha256=expected_index_sha256,
        coordinate_index_bytes=index_path.stat().st_size,
        policy_manifest_sha256=expected_policy_sha256,
        adjudication_sha256=expected_adjudication_sha256,
        adjudication_decision=approval["decision"],
    )
    if expected_adjudication_sha256 == NATIVE_BITNET_M2_ADJUDICATION_SHA256:
        manifest_payload = _atomic_json_payload(expected_manifest)
        if (
            len(manifest_payload)
            != NATIVE_BITNET_DIP_DERIVED_MANIFEST_BYTES
            or hashlib.sha256(manifest_payload).hexdigest()
            != NATIVE_BITNET_DIP_DERIVED_MANIFEST_SHA256
        ):
            raise NativeBitNetValidationError(
                "derived native BitNet manifest does not match the compiled "
                "deployment trust root"
            )

    if target.exists():
        installed = _verified_package_manifest(target)
        if installed != expected_manifest:
            raise NativeBitNetValidationError(
                "derived native BitNet package is not the exact authenticated "
                "source derivation"
            )
        _validate_semantic_memory_descriptor(target, installed)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.semantic-memory-",
            dir=target.parent,
        )
    )
    staged = temporary_root / target.name
    try:
        shutil.copytree(source, staged)
        destination = staged.joinpath(*NATIVE_BITNET_DIP_INDEX_PATH.parts)
        _copy_atomic(index_path, destination)
        manifest_path = staged / "manifest.json"
        atomic_json(manifest_path, expected_manifest)
        verified = _verified_package_manifest(staged)
        if verified != expected_manifest:
            raise NativeBitNetValidationError(
                "source package changed while deriving semantic memory"
            )
        _validate_semantic_memory_descriptor(staged, verified)
        staged.replace(target)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return target


def compile_native_bitnet_package(
    model: str | Path,
    artifact: str | Path,
    out: str | Path,
    *,
    artifact_sha256: str,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    kernel_threads: int = 12,
) -> Path:
    """Compile the qualified source family without retaining source MLP weights."""

    if kernel_threads <= 0:
        raise ValueError("kernel_threads must be positive")
    artifact_path = Path(artifact).resolve()
    loaded_artifact = load_native_bitnet_artifact(artifact_path)
    expected_artifact_sha256 = artifact_sha256.lower()
    if (
        len(expected_artifact_sha256) != 64
        or loaded_artifact.payload_sha256 != expected_artifact_sha256
    ):
        raise NativeBitNetValidationError("native BitNet artifact SHA-256 mismatch")
    model_path, repo_id, resolved_revision = _resolve_full_source(
        model,
        revision=revision,
        cache_dir=cache_dir,
    )
    if (
        repo_id != OFFICIAL_NATIVE_BITNET_REPO
        or resolved_revision != OFFICIAL_NATIVE_BITNET_REVISION
    ):
        raise NativeBitNetValidationError(
            "formal native BitNet packaging requires the pinned official source"
        )
    source_weights = model_path / "model.safetensors"
    if sha256_file(source_weights) != OFFICIAL_NATIVE_BITNET_WEIGHT_SHA256:
        raise NativeBitNetValidationError("pinned reference weight SHA-256 mismatch")

    target = Path(out).resolve()
    manifest_path = target / "manifest.json"
    compile_options = {
        "kernel_threads": kernel_threads,
        "artifact_sha256": expected_artifact_sha256,
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("format") != NATIVE_BITNET_PACKAGE_FORMAT
            or existing.get("version") != NATIVE_BITNET_PACKAGE_VERSION
            or existing.get("compile_options") != compile_options
        ):
            raise NativeBitNetValidationError(
                "existing native BitNet package has different inputs or options"
            )
        for relative, descriptor in existing.get("files", {}).items():
            path = target.joinpath(*PurePosixPath(relative).parts)
            if (
                not path.is_file()
                or path.stat().st_size != descriptor.get("bytes")
                or sha256_file(path) != descriptor.get("sha256")
            ):
                raise NativeBitNetValidationError(
                    f"existing native BitNet package is corrupt: {relative}"
                )
        return target

    target.mkdir(parents=True, exist_ok=True)
    config_dir = target / "config"
    tokenizer_dir = target / "tokenizer"
    config_dir.mkdir(exist_ok=True)
    tokenizer_dir.mkdir(exist_ok=True)
    _copy_atomic(model_path / "config.json", config_dir / "config.json")
    copied_tokenizer = []
    for name in (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "generation_config.json",
    ):
        source = model_path / name
        if source.is_file():
            _copy_atomic(source, tokenizer_dir / name)
            copied_tokenizer.append(name)
    if "tokenizer.json" not in copied_tokenizer:
        raise NativeBitNetValidationError(
            "pinned native BitNet tokenizer is incomplete"
        )

    _copy_atomic(artifact_path, target.joinpath(*NATIVE_BITNET_ARTIFACT_PATH.parts))
    non_mlp = _write_non_mlp_weights(
        source_weights,
        target.joinpath(*NATIVE_BITNET_NON_MLP_PATH.parts),
    )
    files = _inventory(target)
    manifest = {
        "format": NATIVE_BITNET_PACKAGE_FORMAT,
        "version": NATIVE_BITNET_PACKAGE_VERSION,
        "engram_version": __version__,
        "source": {
            "repository": repo_id,
            "revision": resolved_revision,
            "weight_sha256": OFFICIAL_NATIVE_BITNET_WEIGHT_SHA256,
        },
        "model": {
            "hidden_size": loaded_artifact.hidden_size,
            "intermediate_size": loaded_artifact.intermediate_size,
            "num_hidden_layers": len(loaded_artifact.layers),
            "rms_norm_eps": loaded_artifact.rms_norm_eps,
        },
        "mlp": {
            "encoding": "native_bitnet_phase_base3_v1",
            "path": NATIVE_BITNET_ARTIFACT_PATH.as_posix(),
            "sha256": expected_artifact_sha256,
            "serialized_bytes": loaded_artifact.serialized_artifact_bytes,
            "dense_weight_materialization_bytes": 0,
        },
        "transformer": {
            "non_mlp_path": NATIVE_BITNET_NON_MLP_PATH.as_posix(),
            **non_mlp,
        },
        "tokenizer": {
            "path": "tokenizer",
            "files": copied_tokenizer,
            "fix_mistral_regex": True,
        },
        "runtime": {
            "kernel_threads": kernel_threads,
            "attention_mode": "dense_reference",
            "device": "cpu",
            "dtype": "bfloat16",
        },
        "compile_options": compile_options,
        "files": files,
        "does_not_require_source_transformer": True,
    }
    atomic_json(manifest_path, manifest)
    return target


__all__ = [
    "NATIVE_BITNET_ARTIFACT_PATH",
    "NATIVE_BITNET_CONTROLLER_PATH",
    "NATIVE_BITNET_DIP_INDEX_PATH",
    "NATIVE_BITNET_DIP_OPERATOR",
    "NATIVE_BITNET_NON_MLP_PATH",
    "NATIVE_BITNET_PACKAGE_FORMAT",
    "compile_native_bitnet_package",
    "install_native_bitnet_controller",
    "install_native_bitnet_semantic_memory",
]
