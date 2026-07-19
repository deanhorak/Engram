from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from engram.utils import sha256_file, sha256_json


SUPPORTED_MODEL_TYPES = {"llama", "mistral", "engram_tiny_llama"}


class ModelValidationError(ValueError):
    pass


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    shard: str


@dataclass(frozen=True)
class ModelInspection:
    model_path: str
    model_type: str
    architecture: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int
    tensor_count: int
    weight_bytes: int
    tensors: tuple[TensorInfo, ...]
    source_hash: str
    file_hashes: dict[str, str]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["tensors"] = [asdict(item) for item in self.tensors]
        return result


def _read_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise ModelValidationError(f"missing model config: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelValidationError(f"invalid model config: {exc}") from exc


def _npz_inventory(path: Path) -> list[TensorInfo]:
    with np.load(path, allow_pickle=False) as archive:
        return [
            TensorInfo(name, tuple(archive[name].shape), str(archive[name].dtype), path.name)
            for name in sorted(archive.files)
        ]


def _safetensor_inventory(path: Path) -> list[TensorInfo]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ModelValidationError("install the 'conversion' extra to inspect safetensors") from exc
    result: list[TensorInfo] = []
    with safe_open(path, framework="np", device="cpu") as handle:
        for name in sorted(handle.keys()):
            tensor = handle.get_slice(name)
            result.append(TensorInfo(name, tuple(tensor.get_shape()), str(tensor.get_dtype()), path.name))
    return result


def _weight_files(model_path: Path) -> list[Path]:
    npz = sorted(model_path.glob("*.npz"))
    safe = sorted(model_path.glob("*.safetensors"))
    binary = sorted(model_path.glob("pytorch_model*.bin"))
    return npz or safe or binary


def _binary_inventory(path: Path) -> list[TensorInfo]:
    try:
        import torch
    except ImportError as exc:
        raise ModelValidationError("install the 'conversion' extra to inspect PyTorch weights") from exc
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    try:
        return [
            TensorInfo(name, tuple(value.shape), str(value.dtype).replace("torch.", ""), path.name)
            for name, value in sorted(state.items())
            if hasattr(value, "shape")
        ]
    finally:
        del state


def _inventory(model_path: Path, files: list[Path]) -> list[TensorInfo]:
    tensors: list[TensorInfo] = []
    for path in files:
        if path.suffix == ".npz":
            tensors.extend(_npz_inventory(path))
        elif path.suffix == ".safetensors":
            tensors.extend(_safetensor_inventory(path))
        else:
            tensors.extend(_binary_inventory(path))
    unique: dict[str, TensorInfo] = {}
    for tensor in tensors:
        unique[tensor.name] = tensor
    return [unique[name] for name in sorted(unique)]


def _required_mlp_names(layer: int) -> tuple[str, str, str]:
    prefix = f"model.layers.{layer}.mlp"
    return (
        f"{prefix}.gate_proj.weight",
        f"{prefix}.up_proj.weight",
        f"{prefix}.down_proj.weight",
    )


def inspect_model(path: str | Path, *, hash_weights: bool = True) -> ModelInspection:
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_dir():
        raise ModelValidationError(f"model path is not a directory: {model_path}")
    config = _read_config(model_path)
    model_type = str(config.get("model_type", ""))
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ModelValidationError(
            f"unsupported model_type {model_type!r}; expected one of {sorted(SUPPORTED_MODEL_TYPES)}"
        )
    activation = config.get("hidden_act", "silu")
    if activation not in {"silu", "swish"}:
        raise ModelValidationError(f"unsupported MLP activation {activation!r}; expected SiLU")

    fields = {}
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "vocab_size"):
        try:
            fields[key] = int(config[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelValidationError(f"missing or invalid config field {key!r}") from exc
        if fields[key] <= 0:
            raise ModelValidationError(f"config field {key!r} must be positive")

    files = _weight_files(model_path)
    if not files:
        raise ModelValidationError("no .npz, .safetensors, or pytorch_model*.bin weights found")
    tensors = _inventory(model_path, files)
    by_name = {item.name: item for item in tensors}
    hidden = fields["hidden_size"]
    intermediate = fields["intermediate_size"]
    for layer in range(fields["num_hidden_layers"]):
        gate_name, up_name, down_name = _required_mlp_names(layer)
        for name, expected in (
            (gate_name, (intermediate, hidden)),
            (up_name, (intermediate, hidden)),
            (down_name, (hidden, intermediate)),
        ):
            if name not in by_name:
                raise ModelValidationError(f"missing required tensor {name!r}")
            if by_name[name].shape != expected:
                raise ModelValidationError(
                    f"tensor {name!r} has shape {by_name[name].shape}, expected {expected}"
                )

    files_to_hash = [model_path / "config.json", *files]
    file_hashes = {
        item.relative_to(model_path).as_posix(): sha256_file(item)
        for item in files_to_hash
        if hash_weights or item.name == "config.json"
    }
    source_hash = sha256_json({"config": config, "files": file_hashes})
    warnings = []
    if not hash_weights:
        warnings.append("weight hashing disabled; source hash covers configuration only")
    return ModelInspection(
        model_path=str(model_path),
        model_type=model_type,
        architecture=str((config.get("architectures") or ["LlamaForCausalLM"])[0]),
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=fields["num_hidden_layers"],
        num_attention_heads=fields["num_attention_heads"],
        vocab_size=fields["vocab_size"],
        tensor_count=len(tensors),
        weight_bytes=sum(item.stat().st_size for item in files),
        tensors=tuple(tensors),
        source_hash=source_hash,
        file_hashes=file_hashes,
        warnings=tuple(warnings),
    )


def _load_from_shard(model_path: Path, shard: str, names: list[str]) -> dict[str, np.ndarray]:
    path = model_path / shard
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in names}
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(path, framework="np", device="cpu") as handle:
            return {name: np.asarray(handle.get_tensor(name)) for name in names}
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    try:
        return {name: state[name].detach().float().numpy().copy() for name in names}
    finally:
        del state


def load_layer_mlp(path: str | Path, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inspection = inspect_model(path, hash_weights=False)
    if layer < 0 or layer >= inspection.num_hidden_layers:
        raise IndexError(f"layer {layer} outside [0, {inspection.num_hidden_layers})")
    names = _required_mlp_names(layer)
    by_name = {tensor.name: tensor for tensor in inspection.tensors}
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(by_name[name].shard, []).append(name)
    arrays: dict[str, np.ndarray] = {}
    for shard, shard_names in groups.items():
        arrays.update(_load_from_shard(Path(inspection.model_path), shard, shard_names))
    return tuple(np.asarray(arrays[name], dtype=np.float32) for name in names)  # type: ignore[return-value]


def load_named_tensors(path: str | Path, names: list[str]) -> dict[str, np.ndarray]:
    """Load selected tensors while opening each source shard at most once."""
    inspection = inspect_model(path, hash_weights=False)
    by_name = {tensor.name: tensor for tensor in inspection.tensors}
    missing = set(names) - set(by_name)
    if missing:
        raise KeyError(f"missing source tensors: {sorted(missing)}")
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(by_name[name].shard, []).append(name)
    arrays: dict[str, np.ndarray] = {}
    for shard, shard_names in groups.items():
        arrays.update(_load_from_shard(Path(inspection.model_path), shard, shard_names))
    return {name: np.asarray(arrays[name], dtype=np.float32) for name in names}
