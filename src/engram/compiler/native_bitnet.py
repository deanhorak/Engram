"""Source-independent package compiler for the pinned native BitNet track."""

from __future__ import annotations

import json
import os
import shutil
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
NATIVE_BITNET_NON_MLP_PATH = PurePosixPath("transformer/non_mlp.safetensors")


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
    "NATIVE_BITNET_NON_MLP_PATH",
    "NATIVE_BITNET_PACKAGE_FORMAT",
    "compile_native_bitnet_package",
]
