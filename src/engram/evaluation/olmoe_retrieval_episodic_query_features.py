"""Authenticated offline reconstruction of OLMoE query features.

The residual-capacity trace stores ``input_norm`` after the transformer
layer's input RMS normalization.  This module applies only the packaged query
projection and flattened query RMS normalization.  It deliberately stops
before RoPE: positions are authenticated as trace provenance, but are not
consumed by the numerical derivation.

The production entry point is fixed to the packaged OLMoE Q7 geometry.  A
private shape-parametric implementation exists solely so the arithmetic and
fail-closed artifact checks can be unit tested without allocating the
production 128 MiB query-projection inventory.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from engram.utils import sha256_file, sha256_json


_SCHEMA_VERSION = 1
_OPERATION = "olmoe_q7_post_qnorm_pre_rope_query_reconstruction"
_RMS_EPSILON = 1.0e-5


class OLMoEQueryFeatureError(ValueError):
    """Raised when query reconstruction cannot honor its exact contract."""


@dataclass(frozen=True)
class _QueryShape:
    layers: int
    query_heads: int
    head_dimension: int

    @property
    def hidden_size(self) -> int:
        return self.query_heads * self.head_dimension

    def validate(self) -> None:
        values = (self.layers, self.query_heads, self.head_dimension)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ):
            raise OLMoEQueryFeatureError("query-feature dimensions must be integers")
        if min(values) <= 0 or self.hidden_size <= 0:
            raise OLMoEQueryFeatureError("query-feature dimensions must be positive")


_PRODUCTION_SHAPE = _QueryShape(
    layers=16,
    query_heads=16,
    head_dimension=128,
)


@dataclass(frozen=True)
class QueryFeatureResult:
    """One authenticated, immutable-by-convention query derivation."""

    queries: np.ndarray
    tensor_sha256: Mapping[str, str]
    weight_tensor_sha256: Mapping[str, str]
    contract: Mapping[str, Any]
    contract_sha256: str


def _require_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OLMoEQueryFeatureError(f"{label} must be a lowercase SHA-256 digest")
    return value


def tensor_sha256(array: np.ndarray) -> str:
    """Hash the exact C-order bytes of a contiguous NumPy tensor."""

    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise OLMoEQueryFeatureError(
            "query-feature tensor hashing requires a C-contiguous NumPy array"
        )
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _torch_tensor_sha256(torch: Any, tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _query_projection_name(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.q_proj.weight"


def _query_norm_name(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.q_norm.weight"


def _expected_non_mlp_names(shape: _QueryShape) -> set[str]:
    names = {
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
    }
    for layer in range(shape.layers):
        base = f"model.layers.{layer}"
        attention = f"{base}.self_attn"
        names.update(
            {
                f"{base}.input_layernorm.weight",
                f"{base}.post_attention_layernorm.weight",
                f"{attention}.k_norm.weight",
                f"{attention}.k_proj.weight",
                f"{attention}.o_proj.weight",
                f"{attention}.q_norm.weight",
                f"{attention}.q_proj.weight",
                f"{attention}.v_proj.weight",
            }
        )
    return names


def _validate_inputs(
    input_norm: np.ndarray,
    positions: np.ndarray,
    *,
    input_norm_sha256: str,
    positions_sha256: str,
    shape: _QueryShape,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    shape.validate()
    expected_input_hash = _require_sha256(
        input_norm_sha256,
        "input_norm_sha256",
    )
    expected_position_hash = _require_sha256(
        positions_sha256,
        "positions_sha256",
    )
    if (
        not isinstance(input_norm, np.ndarray)
        or input_norm.dtype != np.float32
        or not input_norm.flags.c_contiguous
        or input_norm.ndim != 4
        or input_norm.shape[0] <= 0
        or input_norm.shape[1] <= 0
        or input_norm.shape[2:] != (shape.layers, shape.hidden_size)
        or not np.isfinite(input_norm).all()
    ):
        raise OLMoEQueryFeatureError(
            "input_norm must be finite, C-contiguous float32 "
            "[records, reads, layers, hidden_size]"
        )
    records, reads = input_norm.shape[:2]
    if (
        not isinstance(positions, np.ndarray)
        or positions.dtype != np.int64
        or not positions.flags.c_contiguous
        or positions.shape not in ((reads,), (records, reads))
        or np.any(positions < 0)
    ):
        raise OLMoEQueryFeatureError(
            "positions must be nonnegative, C-contiguous int64 [reads] "
            "or [records, reads]"
        )
    position_rows = positions[None, :] if positions.ndim == 1 else positions
    if reads > 1 and np.any(np.diff(position_rows, axis=-1) <= 0):
        raise OLMoEQueryFeatureError(
            "query-feature positions must be strictly increasing per record"
        )
    actual_input_hash = tensor_sha256(input_norm)
    actual_position_hash = tensor_sha256(positions)
    if actual_input_hash != expected_input_hash:
        raise OLMoEQueryFeatureError("input_norm tensor SHA-256 changed")
    if actual_position_hash != expected_position_hash:
        raise OLMoEQueryFeatureError("positions tensor SHA-256 changed")
    position_grid = np.ascontiguousarray(
        np.broadcast_to(position_rows, (records, reads)),
        dtype=np.int64,
    )
    return input_norm, position_grid, {
        "input_norm": actual_input_hash,
        "positions": actual_position_hash,
        "position_grid": tensor_sha256(position_grid),
    }


def _bound_non_mlp_path(path: str | Path, expected_sha256: str) -> tuple[Path, str]:
    expected = _require_sha256(expected_sha256, "non_mlp_sha256")
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise OLMoEQueryFeatureError(
            "query-feature non-MLP safetensors path must not be a symlink"
        )
    try:
        source = requested.resolve(strict=True)
    except OSError as error:
        raise OLMoEQueryFeatureError(
            "query-feature non-MLP safetensors file does not exist"
        ) from error
    if not source.is_file() or source.suffix != ".safetensors":
        raise OLMoEQueryFeatureError(
            "query-feature non-MLP source must be a safetensors file"
        )
    actual = sha256_file(source)
    if actual != expected:
        raise OLMoEQueryFeatureError("query-feature non-MLP SHA-256 changed")
    return source, actual


def _resolve_device(torch: Any, device: str) -> Any:
    if not isinstance(device, str) or not device:
        raise OLMoEQueryFeatureError("query-feature device is invalid")
    try:
        requested = torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise OLMoEQueryFeatureError("query-feature device is invalid") from error
    if requested.type not in {"cpu", "cuda"}:
        raise OLMoEQueryFeatureError("query-feature device must be CPU or CUDA")
    if requested.type == "cpu":
        if requested.index is not None:
            raise OLMoEQueryFeatureError("indexed CPU devices are not supported")
        return requested
    if not torch.cuda.is_available():
        raise OLMoEQueryFeatureError("query-feature CUDA device is unavailable")
    index = torch.cuda.current_device() if requested.index is None else requested.index
    if index < 0 or index >= torch.cuda.device_count():
        raise OLMoEQueryFeatureError("query-feature CUDA device index is invalid")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise OLMoEQueryFeatureError(
            "deterministic query-feature CUDA requires CUBLAS_WORKSPACE_CONFIG"
        )
    return torch.device("cuda", index)


@contextmanager
def _deterministic_fp32(torch: Any) -> Iterator[None]:
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    matmul_precision = torch.get_float32_matmul_precision()
    cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = cuda_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.set_float32_matmul_precision(matmul_precision)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _validate_weight_inventory(handle: Any, shape: _QueryShape) -> None:
    names = set(handle.keys())
    if names != _expected_non_mlp_names(shape):
        raise OLMoEQueryFeatureError(
            "query-feature non-MLP tensor inventory is not exact"
        )
    embedding = handle.get_slice("model.embed_tokens.weight")
    language_head = handle.get_slice("lm_head.weight")
    if (
        embedding.get_dtype() != "BF16"
        or language_head.get_dtype() != "BF16"
        or embedding.get_shape() != language_head.get_shape()
        or len(embedding.get_shape()) != 2
        or embedding.get_shape()[0] <= 0
        or embedding.get_shape()[1] != shape.hidden_size
    ):
        raise OLMoEQueryFeatureError(
            "query-feature non-MLP vocabulary tensors are invalid"
        )
    expected_vector = [shape.hidden_size]
    expected_matrix = [shape.hidden_size, shape.hidden_size]
    if (
        handle.get_slice("model.norm.weight").get_dtype() != "BF16"
        or handle.get_slice("model.norm.weight").get_shape() != expected_vector
    ):
        raise OLMoEQueryFeatureError(
            "query-feature non-MLP final normalization tensor is invalid"
        )
    for layer in range(shape.layers):
        base = f"model.layers.{layer}"
        attention = f"{base}.self_attn"
        vectors = (
            f"{base}.input_layernorm.weight",
            f"{base}.post_attention_layernorm.weight",
            f"{attention}.k_norm.weight",
            f"{attention}.q_norm.weight",
        )
        matrices = (
            f"{attention}.k_proj.weight",
            f"{attention}.o_proj.weight",
            f"{attention}.q_proj.weight",
            f"{attention}.v_proj.weight",
        )
        for name in vectors:
            view = handle.get_slice(name)
            if view.get_dtype() != "BF16" or view.get_shape() != expected_vector:
                raise OLMoEQueryFeatureError(
                    f"query-feature non-MLP tensor {name} is invalid"
                )
        for name in matrices:
            view = handle.get_slice(name)
            if view.get_dtype() != "BF16" or view.get_shape() != expected_matrix:
                raise OLMoEQueryFeatureError(
                    f"query-feature non-MLP tensor {name} is invalid"
                )


def _weight_tensor_contract(
    torch: Any,
    *,
    name: str,
    value: Any,
) -> tuple[str, str, dict[str, Any]]:
    raw_sha256 = _torch_tensor_sha256(torch, value)
    decoded = value.detach().to(dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(decoded).all()):
        raise OLMoEQueryFeatureError(
            f"query-feature weight tensor {name} is non-finite"
        )
    decoded_sha256 = tensor_sha256(decoded.numpy())
    return raw_sha256, decoded_sha256, {
        "storage_dtype": "BF16",
        "storage_sha256": raw_sha256,
        "decoded_dtype": "float32",
        "decoded_sha256": decoded_sha256,
        "shape": list(value.shape),
    }


def _weight_contract_from_handle(
    torch: Any,
    handle: Any,
    shape: _QueryShape,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, Any]], str]:
    _validate_weight_inventory(handle, shape)
    raw_hashes: dict[str, str] = {}
    decoded_hashes: dict[str, str] = {}
    tensor_contracts: dict[str, dict[str, Any]] = {}
    ordered_names = [
        name
        for layer in range(shape.layers)
        for name in (_query_projection_name(layer), _query_norm_name(layer))
    ]
    for name in ordered_names:
        raw, decoded, contract = _weight_tensor_contract(
            torch,
            name=name,
            value=handle.get_tensor(name),
        )
        raw_hashes[name] = raw
        decoded_hashes[name] = decoded
        tensor_contracts[name] = contract
    root_sha256 = sha256_json(
        [{"name": name, **tensor_contracts[name]} for name in ordered_names]
    )
    return raw_hashes, decoded_hashes, tensor_contracts, root_sha256


def query_weight_contract(
    non_mlp_path: str | Path,
    *,
    non_mlp_sha256: str,
    layers: int = 16,
    hidden_size: int = 2048,
) -> dict[str, Any]:
    """Validate and describe the exact BF16 query-weight inventory.

    The complete safetensors file is bound by ``non_mlp_sha256``.  Returned
    per-tensor storage hashes cover the raw BF16 bytes; decoded hashes cover
    the exact float32 values consumed by reconstruction.
    """

    if (
        isinstance(layers, bool)
        or not isinstance(layers, int)
        or isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or layers <= 0
        or hidden_size <= 0
    ):
        raise OLMoEQueryFeatureError(
            "query-weight contract dimensions must be positive integers"
        )
    shape = _QueryShape(
        layers=layers,
        query_heads=1,
        head_dimension=hidden_size,
    )
    source, source_sha256 = _bound_non_mlp_path(
        non_mlp_path,
        non_mlp_sha256,
    )
    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - conversion dependency
        raise RuntimeError(
            "query-weight contracts require torch and safetensors"
        ) from error
    with safe_open(source, framework="pt", device="cpu") as handle:
        raw, decoded, tensors, root = _weight_contract_from_handle(
            torch,
            handle,
            shape,
        )
    if sha256_file(source) != source_sha256:
        raise OLMoEQueryFeatureError(
            "query-feature non-MLP file changed while it was being read"
        )
    return {
        "schema_version": _SCHEMA_VERSION,
        "format": "safetensors",
        "path": str(source),
        "sha256": source_sha256,
        "layers": layers,
        "hidden_size": hidden_size,
        "exact_non_mlp_tensor_count": 3 + 8 * layers,
        "query_tensor_count": 2 * layers,
        "tensor_sha256": raw,
        "decoded_float32_tensor_sha256": decoded,
        "tensors": tensors,
        "query_weight_root_sha256": root,
    }


def _reconstruct_authenticated_query_features(
    *,
    non_mlp_path: str | Path,
    non_mlp_sha256: str,
    input_norm: np.ndarray,
    input_norm_sha256: str,
    positions: np.ndarray,
    positions_sha256: str,
    device: str,
    expected_weight_tensor_sha256: Mapping[str, str] | None,
    shape: _QueryShape,
) -> QueryFeatureResult:
    input_array, position_grid, tensor_hashes = _validate_inputs(
        input_norm,
        positions,
        input_norm_sha256=input_norm_sha256,
        positions_sha256=positions_sha256,
        shape=shape,
    )
    source, source_sha256 = _bound_non_mlp_path(
        non_mlp_path,
        non_mlp_sha256,
    )
    if expected_weight_tensor_sha256 is not None:
        if not isinstance(expected_weight_tensor_sha256, Mapping):
            raise OLMoEQueryFeatureError(
                "expected query-weight tensor hashes are invalid"
            )
        expected_weight_hashes = {
            str(name): _require_sha256(str(digest), f"{name} SHA-256")
            for name, digest in expected_weight_tensor_sha256.items()
        }
    else:
        expected_weight_hashes = None

    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - conversion dependency
        raise RuntimeError(
            "query reconstruction requires torch and safetensors"
        ) from error

    torch_device = _resolve_device(torch, device)
    records, reads = input_array.shape[:2]
    output_shape = (
        records,
        reads,
        shape.layers,
        shape.query_heads,
        shape.head_dimension,
    )
    queries = np.empty(output_shape, dtype=np.float32)
    raw_weight_hashes: dict[str, str] = {}
    decoded_weight_hashes: dict[str, str] = {}
    weight_contracts: dict[str, dict[str, Any]] = {}

    with safe_open(source, framework="pt", device="cpu") as handle:
        _validate_weight_inventory(handle, shape)
        with _deterministic_fp32(torch), torch.inference_mode():
            source_tensor = torch.from_numpy(input_array).to(
                device=torch_device,
                dtype=torch.float32,
            )
            for layer in range(shape.layers):
                projection_name = _query_projection_name(layer)
                norm_name = _query_norm_name(layer)
                projection_bf16 = handle.get_tensor(projection_name)
                norm_bf16 = handle.get_tensor(norm_name)
                for name, value in (
                    (projection_name, projection_bf16),
                    (norm_name, norm_bf16),
                ):
                    raw, decoded, contract = _weight_tensor_contract(
                        torch,
                        name=name,
                        value=value,
                    )
                    raw_weight_hashes[name] = raw
                    decoded_weight_hashes[name] = decoded
                    weight_contracts[name] = contract

                projection = projection_bf16.to(
                    device=torch_device,
                    dtype=torch.float32,
                )
                norm = norm_bf16.to(
                    device=torch_device,
                    dtype=torch.float32,
                )
                layer_input = source_tensor[:, :, layer, :].contiguous()
                projected = torch.matmul(layer_input, projection.transpose(0, 1))
                squared_sum = torch.sum(
                    projected * projected,
                    dim=-1,
                    dtype=torch.float32,
                )
                inverse_rms = torch.reciprocal(
                    torch.sqrt(
                        squared_sum / float(shape.hidden_size) + _RMS_EPSILON
                    )
                )
                normalized = projected * inverse_rms.unsqueeze(-1)
                normalized = normalized * norm
                normalized = normalized.reshape(
                    records,
                    reads,
                    shape.query_heads,
                    shape.head_dimension,
                )
                layer_output = (
                    normalized.detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                    .numpy()
                )
                if not np.isfinite(layer_output).all():
                    raise OLMoEQueryFeatureError(
                        "query-feature reconstruction produced non-finite values"
                    )
                queries[:, :, layer, :, :] = layer_output
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)

    if (
        sha256_file(source) != source_sha256
        or tensor_sha256(input_array) != tensor_hashes["input_norm"]
        or tensor_sha256(positions) != tensor_hashes["positions"]
    ):
        raise OLMoEQueryFeatureError(
            "query-feature authenticated input changed during reconstruction"
        )
    if expected_weight_hashes is not None:
        if expected_weight_hashes != raw_weight_hashes:
            raise OLMoEQueryFeatureError("query-weight tensor SHA-256 changed")

    queries = np.ascontiguousarray(queries, dtype=np.float32)
    query_hash = tensor_sha256(queries)
    tensor_hashes = {
        **tensor_hashes,
        "post_qnorm_pre_rope_queries": query_hash,
    }
    ordered_weight_names = [
        name
        for layer in range(shape.layers)
        for name in (_query_projection_name(layer), _query_norm_name(layer))
    ]
    weight_root_payload = [
        {
            "name": name,
            **weight_contracts[name],
        }
        for name in ordered_weight_names
    ]
    weight_root_sha256 = sha256_json(weight_root_payload)
    contract: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "operation": _OPERATION,
        "source": {
            "path": str(source),
            "format": "safetensors",
            "sha256": source_sha256,
            "exact_tensor_count": 3 + 8 * shape.layers,
            "query_weight_tensors": weight_contracts,
            "query_weight_root_sha256": weight_root_sha256,
            "caller_supplied_query_weight_hashes": (
                expected_weight_hashes is not None
            ),
        },
        "input": {
            "input_norm": {
                "definition": "post-input-RMSNorm hidden state",
                "shape": list(input_array.shape),
                "dtype": "float32",
                "layout": "record_read_layer_hidden",
                "sha256": tensor_hashes["input_norm"],
            },
            "positions": {
                "definition": (
                    "authenticated absolute-position provenance; not consumed "
                    "before RoPE"
                ),
                "supplied_shape": list(positions.shape),
                "canonical_shape": list(position_grid.shape),
                "dtype": "int64",
                "layout": "record_read",
                "supplied_sha256": tensor_hashes["positions"],
                "canonical_sha256": tensor_hashes["position_grid"],
                "minimum": int(position_grid.min()),
                "maximum": int(position_grid.max()),
            },
        },
        "derivation": {
            "projection": "input_norm @ q_proj.weight.T",
            "projection_weight_layout": "output_input",
            "projection_accumulator_dtype": "float32",
            "query_normalization": "flattened_hidden_rms",
            "query_normalization_width": shape.hidden_size,
            "query_normalization_epsilon": _RMS_EPSILON,
            "query_normalization_weight_dtype": "BF16_decoded_to_float32",
            "rope_applied": False,
        },
        "execution": {
            "device": str(torch_device),
            "framework": "torch",
            "framework_version": str(torch.__version__),
            "deterministic_algorithms": True,
            "float32_matmul_precision": "highest",
            "cuda_matmul_tf32": False,
            "cudnn_tf32": False,
            "cublas_workspace_config": (
                os.environ.get("CUBLAS_WORKSPACE_CONFIG")
                if torch_device.type == "cuda"
                else None
            ),
        },
        "output": {
            "name": "post_qnorm_pre_rope_queries",
            "definition": "flattened-qnorm query before positional rotation",
            "shape": list(queries.shape),
            "dtype": "float32",
            "layout": "record_read_layer_query_head_head_dimension",
            "sha256": query_hash,
        },
    }
    contract_sha256 = sha256_json(contract)
    queries.setflags(write=False)
    return QueryFeatureResult(
        queries=queries,
        tensor_sha256=tensor_hashes,
        weight_tensor_sha256=raw_weight_hashes,
        contract=contract,
        contract_sha256=contract_sha256,
    )


def reconstruct_authenticated_query_features(
    *,
    non_mlp_path: str | Path,
    non_mlp_sha256: str,
    input_norm: np.ndarray,
    input_norm_sha256: str,
    positions: np.ndarray,
    positions_sha256: str,
    device: str = "cpu",
    expected_weight_tensor_sha256: Mapping[str, str] | None = None,
) -> QueryFeatureResult:
    """Reconstruct authenticated production OLMoE post-qnorm/pre-RoPE queries.

    ``input_norm_sha256`` and ``positions_sha256`` bind the exact supplied
    arrays.  ``non_mlp_sha256`` binds the complete exact non-MLP safetensors
    inventory.  Optional per-tensor BF16 hashes provide a second independent
    binding when a frozen protocol already contains them.
    """

    return _reconstruct_authenticated_query_features(
        non_mlp_path=non_mlp_path,
        non_mlp_sha256=non_mlp_sha256,
        input_norm=input_norm,
        input_norm_sha256=input_norm_sha256,
        positions=positions,
        positions_sha256=positions_sha256,
        device=device,
        expected_weight_tensor_sha256=expected_weight_tensor_sha256,
        shape=_PRODUCTION_SHAPE,
    )


__all__ = [
    "OLMoEQueryFeatureError",
    "QueryFeatureResult",
    "query_weight_contract",
    "reconstruct_authenticated_query_features",
    "tensor_sha256",
]
