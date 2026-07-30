"""Authenticated package compiler for the native OLMoE Q7 token runtime."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

from engram import __version__
from engram.models.olmoe import audit_olmoe_source
from engram.models.olmoe_q7 import inspect_olmoe_q7_artifact
from engram.utils import atomic_json, sha256_file


OLMOE_NATIVE_PACKAGE_FORMAT = "engram-native-olmoe-q7"
OLMOE_NATIVE_PACKAGE_VERSION = 1
OLMOE_Q7_PATH = PurePosixPath("mlp/experts.q7")
OLMOE_NON_MLP_PATH = PurePosixPath("transformer/non_mlp.safetensors")
OLMOE_CONFIG_PATH = PurePosixPath("model/config.json")
OLMOE_TOKENIZER_PATH = PurePosixPath("tokenizer")


class OLMoENativePackageError(ValueError):
    """Raised when an OLMoE native package fails authentication."""


def _copy_atomic(source: Path, destination: Path) -> None:
    try:
        resolved_source = source.resolve(strict=True)
    except OSError as exc:
        raise OLMoENativePackageError(
            f"package input cannot be resolved: {source}"
        ) from exc
    if not resolved_source.is_file():
        raise OLMoENativePackageError(f"package input is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        shutil.copyfile(resolved_source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _inventory(root: Path) -> dict[str, dict[str, Any]]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise OLMoENativePackageError(
                f"native OLMoE package contains a symlink: {path}"
            )
        if path.is_file() and path != root / "manifest.json":
            files.append(path)

    def describe(path: Path) -> tuple[str, dict[str, Any]]:
        return (
            path.relative_to(root).as_posix(),
            {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            },
        )

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(files)))) as executor:
        return dict(executor.map(describe, files))


def _safe_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise OLMoENativePackageError(f"unsafe package path: {relative!r}")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise OLMoENativePackageError(f"unsafe package path: {relative!r}")
    return root.joinpath(*pure.parts)


def validate_olmoe_native_package(
    package: str | Path, *, expected_manifest_sha256: str
) -> dict[str, Any]:
    """Authenticate the manifest and exact symlink-free package inventory."""

    root = Path(package).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise OLMoENativePackageError(
            "native OLMoE package root must be a non-symlink directory"
        )
    expected = expected_manifest_sha256.lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise OLMoENativePackageError("expected manifest SHA-256 is invalid")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise OLMoENativePackageError("native OLMoE package manifest is missing")
    if sha256_file(manifest_path) != expected:
        raise OLMoENativePackageError(
            "native OLMoE manifest does not match the authentication root"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OLMoENativePackageError(f"invalid package manifest: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != OLMOE_NATIVE_PACKAGE_FORMAT
        or manifest.get("version") != OLMOE_NATIVE_PACKAGE_VERSION
        or manifest.get("does_not_require_transformers") is not True
        or manifest.get("runtime", {}).get("device") != "cpu"
        or manifest.get("runtime", {}).get("mlp_mode")
        != "native_olmoe_groupwise_q7_topk"
        or manifest.get("runtime", {}).get("attention_mode")
        != "native_streaming_w16_c8_k4_sinks2"
    ):
        raise OLMoENativePackageError("native OLMoE manifest contract is unsupported")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise OLMoENativePackageError("native OLMoE package inventory is missing")
    for relative, descriptor in files.items():
        path = _safe_path(root, relative)
        if (
            not isinstance(descriptor, dict)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != descriptor.get("bytes")
        ):
            raise OLMoENativePackageError(
                f"native OLMoE package file is invalid: {relative}"
            )
    if _inventory(root) != files:
        raise OLMoENativePackageError(
            "native OLMoE package inventory or file hash is not exact"
        )
    for descriptor_name, expected_format, expected_path in (
        ("mlp", "olmoe_native_groupwise_q7_v1", OLMOE_Q7_PATH),
        (
            "transformer",
            "olmoe_native_non_mlp_bf16_v1",
            OLMOE_NON_MLP_PATH,
        ),
    ):
        descriptor = manifest.get(descriptor_name)
        if (
            not isinstance(descriptor, dict)
            or descriptor.get("format") != expected_format
            or descriptor.get("path") != expected_path.as_posix()
            or descriptor.get("sha256") != files[expected_path.as_posix()]["sha256"]
            or descriptor.get("serialized_bytes")
            != files[expected_path.as_posix()]["bytes"]
        ):
            raise OLMoENativePackageError(
                f"native OLMoE {descriptor_name} descriptor is invalid"
            )
    if manifest.get("model", {}).get("config_path") != OLMOE_CONFIG_PATH.as_posix():
        raise OLMoENativePackageError("native OLMoE config descriptor is invalid")
    tokenizer = manifest.get("tokenizer")
    if (
        not isinstance(tokenizer, dict)
        or tokenizer.get("path") != OLMOE_TOKENIZER_PATH.as_posix()
        or "tokenizer.json" not in tokenizer.get("files", [])
    ):
        raise OLMoENativePackageError("native OLMoE tokenizer descriptor is invalid")
    return manifest


def compile_olmoe_native_package(
    model: str | Path,
    q7_artifact: str | Path,
    non_mlp_safetensors: str | Path,
    out: str | Path,
    *,
    kernel_threads: int = 12,
) -> dict[str, Any]:
    """Atomically assemble a complete authenticated native OLMoE package."""

    if kernel_threads <= 0 or kernel_threads > 256:
        raise OLMoENativePackageError("kernel_threads must be in [1, 256]")
    model_path = Path(model).expanduser().resolve()
    audit = audit_olmoe_source(model_path)
    if audit.decision != "proceed_to_router_trace":
        raise OLMoENativePackageError("model does not satisfy the OLMoE contract")
    q7_path = Path(q7_artifact).expanduser().resolve()
    non_mlp_path = Path(non_mlp_safetensors).expanduser().resolve()
    q7 = inspect_olmoe_q7_artifact(q7_path)
    dimensions = q7["dimensions"]
    if (
        dimensions["layers"] != audit.dimensions["num_hidden_layers"]
        or dimensions["hidden_size"] != audit.dimensions["hidden_size"]
        or dimensions["intermediate_size"] != audit.dimensions["intermediate_size"]
        or dimensions["experts"] != audit.dimensions["num_experts"]
        or dimensions["top_k"] != audit.dimensions["num_experts_per_tok"]
    ):
        raise OLMoENativePackageError("Q7 artifact dimensions do not match source")
    if non_mlp_path.is_symlink() or not non_mlp_path.is_file():
        raise OLMoENativePackageError("non-MLP artifact is not a regular file")

    target = Path(out).expanduser().absolute()
    if target.exists():
        raise OLMoENativePackageError("native OLMoE package target already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.compile-", dir=target.parent)
    )
    staged = temporary_root / target.name
    staged.mkdir()
    try:
        _copy_atomic(q7_path, staged.joinpath(*OLMOE_Q7_PATH.parts))
        _copy_atomic(non_mlp_path, staged.joinpath(*OLMOE_NON_MLP_PATH.parts))
        _copy_atomic(
            model_path / "config.json",
            staged.joinpath(*OLMOE_CONFIG_PATH.parts),
        )
        generation = model_path / "generation_config.json"
        if generation.is_file():
            _copy_atomic(generation, staged / "model" / generation.name)
        tokenizer_files = []
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ):
            source = model_path / name
            if source.is_file():
                _copy_atomic(source, staged / "tokenizer" / name)
                tokenizer_files.append(name)
        if "tokenizer.json" not in tokenizer_files:
            raise OLMoENativePackageError("OLMoE tokenizer.json is missing")
        files = _inventory(staged)
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        q7_descriptor = files[OLMOE_Q7_PATH.as_posix()]
        non_mlp_descriptor = files[OLMOE_NON_MLP_PATH.as_posix()]
        manifest = {
            "format": OLMOE_NATIVE_PACKAGE_FORMAT,
            "version": OLMOE_NATIVE_PACKAGE_VERSION,
            "engram_version": __version__,
            "source": {
                "repository": (
                    audit.source if audit.source_kind == "huggingface_hub" else None
                ),
                "revision": audit.resolved_revision,
                "path": str(model_path),
            },
            "model": {
                "config_path": OLMOE_CONFIG_PATH.as_posix(),
                "hidden_size": dimensions["hidden_size"],
                "intermediate_size": dimensions["intermediate_size"],
                "num_hidden_layers": dimensions["layers"],
                "num_attention_heads": int(config["num_attention_heads"]),
                "num_key_value_heads": int(config["num_key_value_heads"]),
                "vocab_size": int(config["vocab_size"]),
                "rms_norm_eps": float(config["rms_norm_eps"]),
                "rope_theta": float(config["rope_theta"]),
            },
            "mlp": {
                "path": OLMOE_Q7_PATH.as_posix(),
                "format": q7["format"],
                "sha256": q7_descriptor["sha256"],
                "serialized_bytes": q7_descriptor["bytes"],
                "dense_expert_materialization_bytes": 0,
            },
            "transformer": {
                "path": OLMOE_NON_MLP_PATH.as_posix(),
                "format": "olmoe_native_non_mlp_bf16_v1",
                "sha256": non_mlp_descriptor["sha256"],
                "serialized_bytes": non_mlp_descriptor["bytes"],
                "tensor_count": 3 + 8 * dimensions["layers"],
            },
            "tokenizer": {
                "path": OLMOE_TOKENIZER_PATH.as_posix(),
                "files": tokenizer_files,
            },
            "runtime": {
                "device": "cpu",
                "kernel_threads": kernel_threads,
                "mlp_mode": "native_olmoe_groupwise_q7_topk",
                "attention_mode": "native_streaming_w16_c8_k4_sinks2",
                "attention_policy": {
                    "local_window": 16,
                    "older_candidates": 8,
                    "older_top_k": 4,
                    "sink_tokens": 2,
                },
            },
            "files": files,
            "does_not_require_transformers": True,
        }
        atomic_json(staged / "manifest.json", manifest)
        manifest_sha256 = sha256_file(staged / "manifest.json")
        validate_olmoe_native_package(staged, expected_manifest_sha256=manifest_sha256)
        staged.replace(target)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return {
        "path": str(target),
        "manifest_sha256": sha256_file(target / "manifest.json"),
        "manifest": manifest,
    }


__all__ = [
    "OLMOE_CONFIG_PATH",
    "OLMOE_NATIVE_PACKAGE_FORMAT",
    "OLMOE_NATIVE_PACKAGE_VERSION",
    "OLMOE_NON_MLP_PATH",
    "OLMOE_Q7_PATH",
    "OLMoENativePackageError",
    "compile_olmoe_native_package",
    "validate_olmoe_native_package",
]
