"""Cache-aligned grouped-ternary artifact for budget-native SwiGLU MLPs.

The format stores every source MLP coefficient as a ternary code and one FP16
scale per contiguous group. Five ternary digits are packed into one byte. The
complete file size, including headers, directory entries, and cache padding,
is the cold-traffic numerator used by the Milestone 2 systems gate.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

_MAGIC = b"ENGBTN11"
_LAYER_MAGIC = b"ENGBTLY1"
_VERSION = 1
_HEADER_BYTES = 64
_LAYER_HEADER_BYTES = 64
_HEADER = struct.Struct("<8s8I")
_DIRECTORY_ENTRY = struct.Struct("<IIQQII")
_LAYER_HEADER = struct.Struct("<8s10I")
_SCALE_FIT_ITERATIONS = 2


def _align(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("alignment operands must be non-negative/positive")
    return ((value + alignment - 1) // alignment) * alignment


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _matrix(values: ArrayLike, name: str) -> NDArray[np.float32]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()  # type: ignore[union-attr]
    try:
        result = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if result.ndim != 2 or not result.shape[0] or not result.shape[1]:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def pack_ternary_codes(codes: ArrayLike) -> NDArray[np.uint8]:
    """Pack ``{-1, 0, +1}`` codes as five base-3 digits per byte."""

    values = np.asarray(codes)
    if values.ndim != 1:
        raise ValueError("ternary codes must be a vector")
    if not np.all((values == -1) | (values == 0) | (values == 1)):
        raise ValueError("ternary codes must contain only -1, 0, and +1")
    digits = values.astype(np.int16, copy=False) + 1
    padded_size = _align(len(digits), 5)
    padded = np.zeros(padded_size, dtype=np.uint8)
    padded[: len(digits)] = digits.astype(np.uint8)
    groups = padded.reshape(-1, 5).astype(np.uint16)
    packed = (
        groups[:, 0]
        + 3 * groups[:, 1]
        + 9 * groups[:, 2]
        + 27 * groups[:, 3]
        + 81 * groups[:, 4]
    )
    return np.ascontiguousarray(packed.astype(np.uint8))


def unpack_ternary_codes(
    packed: ArrayLike,
    logical_size: int,
) -> NDArray[np.int8]:
    """Decode a canonical packed ternary stream."""

    _positive_integer(logical_size, "logical_size")
    values = np.asarray(packed, dtype=np.uint8)
    if values.ndim != 1:
        raise ValueError("packed ternary codes must be a vector")
    expected = (logical_size + 4) // 5
    if len(values) != expected:
        raise ValueError("packed ternary stream length is inconsistent")
    if np.any(values > 242):
        raise ValueError("packed ternary byte is outside the base-3 range")
    working = values.astype(np.uint16)
    digits = np.empty((len(values), 5), dtype=np.uint8)
    for column in range(5):
        digits[:, column] = working % 3
        working //= 3
    flat = digits.reshape(-1)
    if np.any(flat[logical_size:] != 0):
        raise ValueError("packed ternary tail is not canonical")
    return np.ascontiguousarray(flat[:logical_size].astype(np.int8) - 1)


def grouped_ternary_quantize(
    values: ArrayLike,
    *,
    group_size: int,
) -> tuple[NDArray[np.int8], NDArray[np.float16]]:
    """Quantize with non-learned, MSE-refined FP16 group scales."""

    group_size = _positive_integer(group_size, "group_size")
    matrix = _matrix(values, "values")
    flat = matrix.reshape(-1)
    group_count = (len(flat) + group_size - 1) // group_size
    padded = np.zeros(group_count * group_size, dtype=np.float32)
    padded[: len(flat)] = flat
    groups = padded.reshape(group_count, group_size)
    counts = np.full(group_count, group_size, dtype=np.float32)
    tail = len(flat) % group_size
    if tail:
        counts[-1] = tail
    scale32 = np.sum(np.abs(groups), axis=1) / counts
    scale32 = np.where(scale32 > 0, scale32, 1.0)
    for _ in range(_SCALE_FIT_ITERATIONS):
        rounded_scale = scale32.astype(np.float16).astype(np.float32)
        codes = np.rint(groups / rounded_scale[:, None]).clip(-1, 1)
        denominator = np.sum(codes * codes, axis=1)
        fitted = np.sum(groups * codes, axis=1) / np.maximum(
            denominator, 1.0
        )
        scale32 = np.where(denominator > 0, fitted, 1.0)
    scales = scale32.astype(np.float16)
    scale32 = scales.astype(np.float32)
    codes = np.rint(groups / scale32[:, None]).clip(-1, 1).astype(np.int8)
    return np.ascontiguousarray(codes.reshape(-1)[: len(flat)]), scales


def grouped_ternary_decode(
    codes: ArrayLike,
    scales: ArrayLike,
    *,
    shape: tuple[int, int],
    group_size: int,
) -> NDArray[np.float32]:
    """Decode grouped ternary codes into their matrix orientation."""

    group_size = _positive_integer(group_size, "group_size")
    if (
        len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
    ):
        raise ValueError("shape must contain two positive integers")
    logical_size = shape[0] * shape[1]
    code_values = np.asarray(codes, dtype=np.int8)
    scale_values = np.asarray(scales, dtype=np.float16)
    group_count = (logical_size + group_size - 1) // group_size
    if code_values.shape != (logical_size,):
        raise ValueError("ternary code count does not match the matrix shape")
    if scale_values.shape != (group_count,):
        raise ValueError("ternary scale count does not match the matrix shape")
    if not np.all(
        (code_values == -1) | (code_values == 0) | (code_values == 1)
    ):
        raise ValueError("ternary codes must contain only -1, 0, and +1")
    expanded_scales = np.repeat(
        scale_values.astype(np.float32),
        group_size,
    )[:logical_size]
    decoded = code_values.astype(np.float32) * expanded_scales
    return np.ascontiguousarray(decoded.reshape(shape))


@dataclass(frozen=True)
class BudgetNativeTernaryLayerWeights:
    gate: ArrayLike
    up: ArrayLike
    down: ArrayLike


@dataclass(frozen=True)
class PackedBudgetNativeTernaryProjection:
    packed_codes: NDArray[np.uint8]
    scales: NDArray[np.float16]


@dataclass(frozen=True)
class PackedBudgetNativeTernaryLayer:
    gate: PackedBudgetNativeTernaryProjection
    up: PackedBudgetNativeTernaryProjection
    down: PackedBudgetNativeTernaryProjection


@dataclass(frozen=True)
class LoadedBudgetNativeTernaryArtifact:
    layers: tuple[PackedBudgetNativeTernaryLayer, ...]
    hidden_size: int
    intermediate_size: int
    group_size: int
    cache_line_bytes: int
    header_block_bytes: int
    directory_block_bytes: int
    layer_offsets: tuple[int, ...]
    layer_block_bytes: tuple[int, ...]
    serialized_artifact_bytes: int


def budget_native_ternary_traffic(
    hidden_size: int,
    intermediate_size: int,
    *,
    layer_count: int,
    group_size: int = 128,
    cache_line_bytes: int = 64,
) -> dict[str, Any]:
    """Account the exact serialized grouped-ternary MLP artifact."""

    hidden_size = _positive_integer(hidden_size, "hidden_size")
    intermediate_size = _positive_integer(
        intermediate_size, "intermediate_size"
    )
    layer_count = _positive_integer(layer_count, "layer_count")
    group_size = _positive_integer(group_size, "group_size")
    cache_line_bytes = _positive_integer(
        cache_line_bytes, "cache_line_bytes"
    )
    if cache_line_bytes < 64 or cache_line_bytes % 64:
        raise ValueError("cache_line_bytes must be a positive multiple of 64")
    elements = hidden_size * intermediate_size
    code_bytes_per_projection = (elements + 4) // 5
    scales_per_projection = (elements + group_size - 1) // group_size
    scale_bytes_per_projection = 2 * scales_per_projection
    tensor_payload_bytes = 3 * (
        code_bytes_per_projection + scale_bytes_per_projection
    )
    layer_payload_bytes = _LAYER_HEADER_BYTES + tensor_payload_bytes
    layer_block_bytes = _align(layer_payload_bytes, cache_line_bytes)
    header_block_bytes = _align(_HEADER_BYTES, cache_line_bytes)
    directory_payload_bytes = layer_count * _DIRECTORY_ENTRY.size
    directory_block_bytes = _align(
        directory_payload_bytes, cache_line_bytes
    )
    serialized_artifact_bytes = (
        header_block_bytes
        + directory_block_bytes
        + layer_count * layer_block_bytes
    )
    dense_q4_source_mlp_bytes = (
        layer_count * 3 * hidden_size * intermediate_size + 1
    ) // 2
    fraction = serialized_artifact_bytes / dense_q4_source_mlp_bytes
    return {
        "layout": "budget_native_grouped_ternary_v1",
        "packing": "five base-3 ternary digits per byte",
        "scale_policy": (
            "non-learned FP16 scale per group initialized by mean absolute "
            "weight and refined by two least-squares code/scale iterations"
        ),
        "scale_fit_iterations": _SCALE_FIT_ITERATIONS,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "layer_count": layer_count,
        "group_size": group_size,
        "cache_line_bytes": cache_line_bytes,
        "elements_per_projection": elements,
        "code_bytes_per_projection": code_bytes_per_projection,
        "scales_per_projection": scales_per_projection,
        "scale_bytes_per_projection": scale_bytes_per_projection,
        "ternary_code_bytes_per_layer": 3 * code_bytes_per_projection,
        "fp16_scale_bytes_per_layer": 3 * scale_bytes_per_projection,
        "tensor_payload_bytes_per_layer": tensor_payload_bytes,
        "layer_header_bytes": _LAYER_HEADER_BYTES,
        "layer_payload_bytes": layer_payload_bytes,
        "layer_cache_padding_bytes": layer_block_bytes - layer_payload_bytes,
        "layer_block_bytes": layer_block_bytes,
        "header_cache_aligned_bytes": header_block_bytes,
        "directory_entry_bytes": _DIRECTORY_ENTRY.size,
        "directory_payload_bytes": directory_payload_bytes,
        "directory_cache_aligned_bytes": directory_block_bytes,
        "serialized_artifact_bytes": serialized_artifact_bytes,
        "total_cold_bytes": serialized_artifact_bytes,
        "traffic_numerator_bytes": serialized_artifact_bytes,
        "dense_q4_source_mlp_bytes": dense_q4_source_mlp_bytes,
        "fraction_of_dense_q4": fraction,
        "passes_45_percent_traffic_gate": fraction <= 0.45,
        "headroom_bytes_to_45_percent": (
            int(0.45 * dense_q4_source_mlp_bytes)
            - serialized_artifact_bytes
        ),
        "accounting_policy": (
            "cold numerator is the exact serialized artifact including packed "
            "ternary codes, FP16 group scales, headers, directory, and "
            "cache-line padding; denominator is ideal code-only dense-Q4 MLP "
            "weights"
        ),
    }


def _pack_projection(
    matrix: NDArray[np.float32],
    *,
    group_size: int,
) -> PackedBudgetNativeTernaryProjection:
    codes, scales = grouped_ternary_quantize(matrix, group_size=group_size)
    return PackedBudgetNativeTernaryProjection(
        pack_ternary_codes(codes),
        scales,
    )


def _projection_payload(
    projection: PackedBudgetNativeTernaryProjection,
) -> bytes:
    return (
        projection.packed_codes.tobytes()
        + projection.scales.astype("<f2", copy=False).tobytes()
    )


def save_budget_native_ternary_artifact(
    path: str | Path,
    layers: list[BudgetNativeTernaryLayerWeights]
    | tuple[BudgetNativeTernaryLayerWeights, ...],
    *,
    group_size: int = 128,
    cache_line_bytes: int = 64,
) -> Path:
    """Quantize and atomically serialize full-width ternary SwiGLU layers."""

    group_size = _positive_integer(group_size, "group_size")
    logical_layers = tuple(layers)
    if not logical_layers:
        raise ValueError("layers must not be empty")
    packed_layers: list[PackedBudgetNativeTernaryLayer] = []
    hidden_size: int | None = None
    intermediate_size: int | None = None
    for index, layer in enumerate(logical_layers):
        gate = _matrix(layer.gate, f"layers[{index}].gate")
        up = _matrix(layer.up, f"layers[{index}].up")
        down = _matrix(layer.down, f"layers[{index}].down")
        if gate.shape != up.shape:
            raise ValueError(f"gate/up shape mismatch at layer {index}")
        intermediate, hidden = gate.shape
        if hidden_size is None:
            hidden_size = hidden
            intermediate_size = intermediate
        if (
            hidden != hidden_size
            or intermediate != intermediate_size
            or down.shape != (hidden, intermediate)
        ):
            raise ValueError(
                f"full-width ternary matrix shape mismatch at layer {index}"
            )
        packed_layers.append(
            PackedBudgetNativeTernaryLayer(
                _pack_projection(gate, group_size=group_size),
                _pack_projection(up, group_size=group_size),
                _pack_projection(down, group_size=group_size),
            )
        )
    assert hidden_size is not None and intermediate_size is not None
    traffic = budget_native_ternary_traffic(
        hidden_size,
        intermediate_size,
        layer_count=len(packed_layers),
        group_size=group_size,
        cache_line_bytes=cache_line_bytes,
    )
    header_block_bytes = int(traffic["header_cache_aligned_bytes"])
    directory_block_bytes = int(traffic["directory_cache_aligned_bytes"])
    layer_block_bytes = int(traffic["layer_block_bytes"])
    layer_payload_bytes = int(traffic["layer_payload_bytes"])
    offsets = [
        header_block_bytes + directory_block_bytes + index * layer_block_bytes
        for index in range(len(packed_layers))
    ]

    header = bytearray(header_block_bytes)
    _HEADER.pack_into(
        header,
        0,
        _MAGIC,
        _VERSION,
        len(packed_layers),
        hidden_size,
        intermediate_size,
        group_size,
        cache_line_bytes,
        _DIRECTORY_ENTRY.size,
        directory_block_bytes,
    )
    directory = bytearray(directory_block_bytes)
    for index, offset in enumerate(offsets):
        _DIRECTORY_ENTRY.pack_into(
            directory,
            index * _DIRECTORY_ENTRY.size,
            index,
            0,
            offset,
            layer_block_bytes,
            layer_payload_bytes,
            0,
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(header)
            handle.write(directory)
            for index, layer in enumerate(packed_layers):
                tensor_payload = b"".join(
                    (
                        _projection_payload(layer.gate),
                        _projection_payload(layer.up),
                        _projection_payload(layer.down),
                    )
                )
                if len(tensor_payload) != traffic["tensor_payload_bytes_per_layer"]:
                    raise AssertionError(
                        "packed ternary layer differs from traffic accounting"
                    )
                layer_header = bytearray(_LAYER_HEADER_BYTES)
                _LAYER_HEADER.pack_into(
                    layer_header,
                    0,
                    _LAYER_MAGIC,
                    _VERSION,
                    index,
                    hidden_size,
                    intermediate_size,
                    group_size,
                    int(traffic["code_bytes_per_projection"]),
                    int(traffic["scales_per_projection"]),
                    layer_payload_bytes,
                    0,
                    0,
                )
                handle.write(layer_header)
                handle.write(tensor_payload)
                handle.write(b"\0" * (layer_block_bytes - layer_payload_bytes))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    if destination.stat().st_size != traffic["serialized_artifact_bytes"]:
        raise AssertionError(
            "serialized grouped-ternary artifact has an unexpected size"
        )
    return destination


def load_budget_native_ternary_artifact(
    path: str | Path,
) -> LoadedBudgetNativeTernaryArtifact:
    """Strictly validate and load a grouped-ternary artifact."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"cannot read budget-native ternary artifact {source}"
        ) from exc
    if len(payload) < _HEADER_BYTES:
        raise ValueError("grouped-ternary artifact is shorter than its header")
    (
        magic,
        version,
        layer_count,
        hidden_size,
        intermediate_size,
        group_size,
        cache_line_bytes,
        directory_entry_bytes,
        directory_block_bytes,
    ) = _HEADER.unpack_from(payload)
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("grouped-ternary artifact magic/version mismatch")
    for value, name in (
        (layer_count, "layer_count"),
        (hidden_size, "hidden_size"),
        (intermediate_size, "intermediate_size"),
        (group_size, "group_size"),
        (cache_line_bytes, "cache_line_bytes"),
    ):
        _positive_integer(value, name)
    if directory_entry_bytes != _DIRECTORY_ENTRY.size:
        raise ValueError(
            "grouped-ternary directory entry size is unsupported"
        )
    header_block_bytes = _align(_HEADER_BYTES, cache_line_bytes)
    expected_directory_bytes = _align(
        layer_count * _DIRECTORY_ENTRY.size, cache_line_bytes
    )
    if directory_block_bytes != expected_directory_bytes:
        raise ValueError("grouped-ternary directory block size is invalid")
    if len(payload) < header_block_bytes + directory_block_bytes:
        raise ValueError("grouped-ternary artifact is shorter than its directory")
    if any(payload[_HEADER.size : header_block_bytes]):
        raise ValueError("grouped-ternary header padding is non-zero")
    directory_payload_end = (
        header_block_bytes + layer_count * _DIRECTORY_ENTRY.size
    )
    directory_end = header_block_bytes + directory_block_bytes
    if any(payload[directory_payload_end:directory_end]):
        raise ValueError("grouped-ternary directory padding is non-zero")

    traffic = budget_native_ternary_traffic(
        hidden_size,
        intermediate_size,
        layer_count=layer_count,
        group_size=group_size,
        cache_line_bytes=cache_line_bytes,
    )
    if len(payload) != traffic["serialized_artifact_bytes"]:
        raise ValueError(
            "grouped-ternary artifact length does not match its header"
        )
    code_bytes = int(traffic["code_bytes_per_projection"])
    scale_count = int(traffic["scales_per_projection"])
    scale_bytes = 2 * scale_count
    logical_elements = hidden_size * intermediate_size
    expected_offset = directory_end
    layers: list[PackedBudgetNativeTernaryLayer] = []
    offsets: list[int] = []
    block_sizes: list[int] = []
    for index in range(layer_count):
        entry = _DIRECTORY_ENTRY.unpack_from(
            payload, header_block_bytes + index * _DIRECTORY_ENTRY.size
        )
        (
            entry_index,
            entry_reserved,
            offset,
            block_bytes,
            layer_payload_bytes,
            entry_reserved_2,
        ) = entry
        if entry_index != index or entry_reserved or entry_reserved_2:
            raise ValueError(
                "grouped-ternary directory entry is not canonical"
            )
        if (
            offset != expected_offset
            or offset % cache_line_bytes
            or block_bytes != traffic["layer_block_bytes"]
            or layer_payload_bytes != traffic["layer_payload_bytes"]
        ):
            raise ValueError(
                "grouped-ternary layer directory entry is invalid"
            )
        header = _LAYER_HEADER.unpack_from(payload, offset)
        (
            layer_magic,
            layer_version,
            layer_index,
            layer_hidden,
            layer_intermediate,
            layer_group,
            layer_code_bytes,
            layer_scale_count,
            header_payload_bytes,
            reserved,
            reserved_2,
        ) = header
        if (
            layer_magic != _LAYER_MAGIC
            or layer_version != _VERSION
            or layer_index != index
            or layer_hidden != hidden_size
            or layer_intermediate != intermediate_size
            or layer_group != group_size
            or layer_code_bytes != code_bytes
            or layer_scale_count != scale_count
            or header_payload_bytes != layer_payload_bytes
            or reserved
            or reserved_2
        ):
            raise ValueError("grouped-ternary layer header is invalid")
        cursor = offset + _LAYER_HEADER_BYTES
        projections: list[PackedBudgetNativeTernaryProjection] = []
        for _ in range(3):
            packed_codes = np.frombuffer(
                payload,
                dtype=np.uint8,
                count=code_bytes,
                offset=cursor,
            ).copy()
            cursor += code_bytes
            scales = np.frombuffer(
                payload,
                dtype="<f2",
                count=scale_count,
                offset=cursor,
            ).copy()
            cursor += scale_bytes
            if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
                raise ValueError(
                    "grouped-ternary projection scales are invalid"
                )
            # Decode once during validation so malformed base-3 tails fail at
            # the package boundary rather than during inference.
            unpack_ternary_codes(packed_codes, logical_elements)
            projections.append(
                PackedBudgetNativeTernaryProjection(packed_codes, scales)
            )
        if cursor != offset + layer_payload_bytes:
            raise ValueError("grouped-ternary layer payload size is invalid")
        if any(payload[cursor : offset + block_bytes]):
            raise ValueError("grouped-ternary layer padding is non-zero")
        layers.append(PackedBudgetNativeTernaryLayer(*projections))
        offsets.append(offset)
        block_sizes.append(block_bytes)
        expected_offset += block_bytes
    if expected_offset != len(payload):
        raise ValueError("grouped-ternary artifact has trailing bytes")
    return LoadedBudgetNativeTernaryArtifact(
        layers=tuple(layers),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        group_size=group_size,
        cache_line_bytes=cache_line_bytes,
        header_block_bytes=header_block_bytes,
        directory_block_bytes=directory_block_bytes,
        layer_offsets=tuple(offsets),
        layer_block_bytes=tuple(block_sizes),
        serialized_artifact_bytes=len(payload),
    )


def _decode_projection(
    projection: PackedBudgetNativeTernaryProjection,
    *,
    shape: tuple[int, int],
    group_size: int,
) -> NDArray[np.float32]:
    logical_size = shape[0] * shape[1]
    codes = unpack_ternary_codes(projection.packed_codes, logical_size)
    return grouped_ternary_decode(
        codes,
        projection.scales,
        shape=shape,
        group_size=group_size,
    )


def decode_budget_native_ternary_artifact(
    artifact: LoadedBudgetNativeTernaryArtifact,
) -> list[dict[str, NDArray[np.float32]]]:
    """Decode all projections from a validated artifact."""

    gate_shape = (artifact.intermediate_size, artifact.hidden_size)
    down_shape = (artifact.hidden_size, artifact.intermediate_size)
    return [
        {
            "gate": _decode_projection(
                layer.gate,
                shape=gate_shape,
                group_size=artifact.group_size,
            ),
            "up": _decode_projection(
                layer.up,
                shape=gate_shape,
                group_size=artifact.group_size,
            ),
            "down": _decode_projection(
                layer.down,
                shape=down_shape,
                group_size=artifact.group_size,
            ),
        }
        for layer in artifact.layers
    ]


def budget_native_ternary_forward(
    artifact: LoadedBudgetNativeTernaryArtifact,
    layer: int,
    hidden: ArrayLike,
) -> NDArray[np.float32]:
    """Reference NumPy forward for one reloaded ternary SwiGLU layer."""

    if isinstance(layer, bool) or not isinstance(layer, int):
        raise ValueError("layer must be an integer")
    if not 0 <= layer < len(artifact.layers):
        raise ValueError("layer index is outside the artifact")
    states = np.asarray(hidden, dtype=np.float32)
    if states.ndim < 1 or states.shape[-1] != artifact.hidden_size:
        raise ValueError("hidden states have an incompatible shape")
    decoded = decode_budget_native_ternary_artifact(artifact)[layer]
    gate = states @ decoded["gate"].T
    activation = (gate / (1.0 + np.exp(-gate))) * (
        states @ decoded["up"].T
    )
    return np.asarray(activation @ decoded["down"].T, dtype=np.float32)


__all__ = [
    "BudgetNativeTernaryLayerWeights",
    "LoadedBudgetNativeTernaryArtifact",
    "budget_native_ternary_forward",
    "budget_native_ternary_traffic",
    "decode_budget_native_ternary_artifact",
    "grouped_ternary_decode",
    "grouped_ternary_quantize",
    "load_budget_native_ternary_artifact",
    "pack_ternary_codes",
    "save_budget_native_ternary_artifact",
    "unpack_ternary_codes",
]
