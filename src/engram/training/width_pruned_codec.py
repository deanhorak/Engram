"""Strict cache-aligned signed-Q4 artifact for compact SwiGLU layers.

Every layer is an independently validated cache-line-aligned block containing
signed-Q4 gate, up, and transposed-down rows, one FP16 scale per stored row,
and the uint16/uint32 source-channel IDs used to initialize the compact layer.
The traffic gate uses the complete serialized artifact as its cold numerator;
the denominator is the ideal code-only dense-Q4 source MLP.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.training.switch_expert_boundaries import (
    PackedQ4Rows,
    _decode_symmetric_q4_rows,
    _pack_symmetric_q4_rows,
)

_MAGIC = b"ENGFWQ41"
_LAYER_MAGIC = b"ENGFWLY1"
_VERSION = 1
_HEADER_BYTES = 64
_LAYER_HEADER_BYTES = 64
_HEADER = struct.Struct("<8s8I")
_DIRECTORY_ENTRY = struct.Struct("<IIQQII")
_LAYER_HEADER = struct.Struct("<8s10I")
_Q4_CODEC = 1
_SOURCE_IDS_UINT16 = 1
_SOURCE_IDS_UINT32 = 2


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


def _source_id_codec(source_intermediate_size: int) -> tuple[int, Any, int]:
    if source_intermediate_size <= np.iinfo(np.uint16).max:
        return _SOURCE_IDS_UINT16, np.dtype("<u2"), 2
    if source_intermediate_size <= np.iinfo(np.uint32).max:
        return _SOURCE_IDS_UINT32, np.dtype("<u4"), 4
    raise ValueError("source_intermediate_size cannot be represented by the artifact")


def _source_ids(
    values: ArrayLike, width: int, source_intermediate_size: int, dtype: Any
) -> NDArray[Any]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()  # type: ignore[union-attr]
    array = np.asarray(values)
    if array.shape != (width,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("source_ids must contain one integer per compact row")
    converted = array.astype(np.int64, copy=False)
    if (
        np.any(converted < 0)
        or np.any(converted >= source_intermediate_size)
        or len(np.unique(converted)) != width
    ):
        raise ValueError("source_ids must be unique source-channel indices")
    return np.ascontiguousarray(converted.astype(dtype))


@dataclass(frozen=True)
class WidthPrunedQ4LayerWeights:
    """Logical compact weights and their source-channel provenance."""

    gate: ArrayLike
    up: ArrayLike
    down: ArrayLike
    source_ids: ArrayLike


@dataclass(frozen=True)
class PackedWidthPrunedQ4Layer:
    gate: PackedQ4Rows
    up: PackedQ4Rows
    down_t: PackedQ4Rows
    source_ids: NDArray[Any]


@dataclass(frozen=True)
class LoadedWidthPrunedQ4Artifact:
    layers: tuple[PackedWidthPrunedQ4Layer, ...]
    hidden_size: int
    source_intermediate_size: int
    cache_line_bytes: int
    source_id_codec: int
    header_block_bytes: int
    directory_block_bytes: int
    layer_offsets: tuple[int, ...]
    layer_block_bytes: tuple[int, ...]
    serialized_artifact_bytes: int


def width_pruned_q4_traffic(
    hidden_size: int,
    source_intermediate_size: int,
    layer_widths: list[int] | tuple[int, ...],
    *,
    cache_line_bytes: int = 64,
) -> dict[str, Any]:
    """Account the exact serialized bytes of a compact signed-Q4 artifact.

    The numerator includes the global header, directory, every local layer
    header, all Q4 codes/scales/source IDs, and all cache-line padding.  It is
    therefore exactly equal to the file size produced by the serializer.
    """

    _positive_integer(hidden_size, "hidden_size")
    _positive_integer(source_intermediate_size, "source_intermediate_size")
    _positive_integer(cache_line_bytes, "cache_line_bytes")
    if cache_line_bytes < _HEADER_BYTES or cache_line_bytes % _HEADER_BYTES:
        raise ValueError("cache_line_bytes must be a positive multiple of 64")
    widths = tuple(layer_widths)
    if not widths:
        raise ValueError("layer_widths must not be empty")
    for width in widths:
        _positive_integer(width, "layer width")
        if width > source_intermediate_size:
            raise ValueError("layer width cannot exceed source_intermediate_size")
    source_id_codec, _, source_id_bytes = _source_id_codec(source_intermediate_size)
    header_block_bytes = _align(_HEADER_BYTES, cache_line_bytes)
    directory_payload_bytes = len(widths) * _DIRECTORY_ENTRY.size
    directory_block_bytes = _align(directory_payload_bytes, cache_line_bytes)
    layers = []
    total_layer_block_bytes = 0
    for index, width in enumerate(widths):
        one_matrix_q4_bytes = (width * hidden_size + 1) // 2
        q4_code_bytes = 3 * one_matrix_q4_bytes
        fp16_scale_bytes = 3 * width * 2
        ids_bytes = width * source_id_bytes
        tensor_payload_bytes = q4_code_bytes + fp16_scale_bytes + ids_bytes
        layer_payload_bytes = _LAYER_HEADER_BYTES + tensor_payload_bytes
        layer_block_bytes = _align(layer_payload_bytes, cache_line_bytes)
        total_layer_block_bytes += layer_block_bytes
        layers.append(
            {
                "layer": index,
                "width": width,
                "one_matrix_q4_bytes": one_matrix_q4_bytes,
                "q4_code_bytes": q4_code_bytes,
                "fp16_scale_bytes": fp16_scale_bytes,
                "source_id_bytes": ids_bytes,
                "tensor_payload_bytes": tensor_payload_bytes,
                "layer_header_bytes": _LAYER_HEADER_BYTES,
                "layer_payload_bytes": layer_payload_bytes,
                "cache_padding_bytes": layer_block_bytes - layer_payload_bytes,
                "layer_block_bytes": layer_block_bytes,
            }
        )
    serialized_artifact_bytes = (
        header_block_bytes + directory_block_bytes + total_layer_block_bytes
    )
    dense_q4_source_mlp_bytes = (
        len(widths) * 3 * hidden_size * source_intermediate_size + 1
    ) // 2
    fraction = serialized_artifact_bytes / dense_q4_source_mlp_bytes
    return {
        "layout": "width_pruned_cache_aligned_q4_v1",
        "layers": layers,
        "layer_count": len(widths),
        "layer_widths": list(widths),
        "hidden_size": hidden_size,
        "source_intermediate_size": source_intermediate_size,
        "source_id_codec": (
            "uint16" if source_id_codec == _SOURCE_IDS_UINT16 else "uint32"
        ),
        "source_id_bytes_per_record": source_id_bytes,
        "cache_line_bytes": cache_line_bytes,
        "header_cache_aligned_bytes": header_block_bytes,
        "directory_entry_bytes": _DIRECTORY_ENTRY.size,
        "directory_payload_bytes": directory_payload_bytes,
        "directory_cache_aligned_bytes": directory_block_bytes,
        "total_layer_block_bytes": total_layer_block_bytes,
        "global_metadata_bytes": header_block_bytes + directory_block_bytes,
        "serialized_artifact_bytes": serialized_artifact_bytes,
        "total_cold_bytes": serialized_artifact_bytes,
        "traffic_numerator_bytes": serialized_artifact_bytes,
        "dense_q4_source_mlp_bytes": dense_q4_source_mlp_bytes,
        "fraction_of_dense_q4": fraction,
        "passes_45_percent_traffic_gate": fraction <= 0.45,
        "accounting_policy": (
            "cold numerator is the complete serialized artifact, including Q4 codes, "
            "FP16 row scales, source IDs, headers, directory, and cache-line padding; "
            "denominator is ideal code-only dense-Q4 source MLP weights"
        ),
    }


def _layer_payload(layer: PackedWidthPrunedQ4Layer) -> bytes:
    return b"".join(
        (
            layer.gate.packed.tobytes(),
            layer.up.packed.tobytes(),
            layer.down_t.packed.tobytes(),
            layer.gate.scales.astype("<f2", copy=False).tobytes(),
            layer.up.scales.astype("<f2", copy=False).tobytes(),
            layer.down_t.scales.astype("<f2", copy=False).tobytes(),
            layer.source_ids.tobytes(),
        )
    )


def save_width_pruned_q4_artifact(
    path: str | Path,
    layers: list[WidthPrunedQ4LayerWeights] | tuple[WidthPrunedQ4LayerWeights, ...],
    *,
    source_intermediate_size: int,
    cache_line_bytes: int = 64,
) -> Path:
    """Quantize and atomically serialize all compact SwiGLU layers."""

    _positive_integer(source_intermediate_size, "source_intermediate_size")
    logical_layers = tuple(layers)
    if not logical_layers:
        raise ValueError("layers must not be empty")
    source_id_codec, source_id_dtype, _ = _source_id_codec(source_intermediate_size)
    packed_layers: list[PackedWidthPrunedQ4Layer] = []
    widths: list[int] = []
    hidden_size: int | None = None
    for index, layer in enumerate(logical_layers):
        gate = _matrix(layer.gate, f"layers[{index}].gate")
        up = _matrix(layer.up, f"layers[{index}].up")
        down = _matrix(layer.down, f"layers[{index}].down")
        if gate.shape != up.shape:
            raise ValueError(f"gate/up shape mismatch at layer {index}")
        width, hidden = gate.shape
        if hidden_size is None:
            hidden_size = hidden
        if hidden != hidden_size or down.shape != (hidden, width):
            raise ValueError(f"compact matrix shape mismatch at layer {index}")
        if width > source_intermediate_size:
            raise ValueError(f"compact width exceeds source width at layer {index}")
        ids = _source_ids(
            layer.source_ids, width, source_intermediate_size, source_id_dtype
        )
        widths.append(width)
        packed_layers.append(
            PackedWidthPrunedQ4Layer(
                _pack_symmetric_q4_rows(gate),
                _pack_symmetric_q4_rows(up),
                _pack_symmetric_q4_rows(down.T),
                ids,
            )
        )
    assert hidden_size is not None
    traffic = width_pruned_q4_traffic(
        hidden_size,
        source_intermediate_size,
        widths,
        cache_line_bytes=cache_line_bytes,
    )
    header_block_bytes = int(traffic["header_cache_aligned_bytes"])
    directory_block_bytes = int(traffic["directory_cache_aligned_bytes"])
    layer_metadata = traffic["layers"]
    offsets: list[int] = []
    cursor = header_block_bytes + directory_block_bytes
    for metadata in layer_metadata:
        offsets.append(cursor)
        cursor += int(metadata["layer_block_bytes"])
    if cursor != traffic["serialized_artifact_bytes"]:
        raise AssertionError("internal compact artifact accounting mismatch")

    header = bytearray(header_block_bytes)
    _HEADER.pack_into(
        header,
        0,
        _MAGIC,
        _VERSION,
        len(packed_layers),
        hidden_size,
        source_intermediate_size,
        cache_line_bytes,
        _DIRECTORY_ENTRY.size,
        directory_block_bytes,
        source_id_codec,
    )
    directory = bytearray(directory_block_bytes)
    for index, (offset, metadata) in enumerate(
        zip(offsets, layer_metadata, strict=True)
    ):
        _DIRECTORY_ENTRY.pack_into(
            directory,
            index * _DIRECTORY_ENTRY.size,
            index,
            int(metadata["width"]),
            offset,
            int(metadata["layer_block_bytes"]),
            int(metadata["layer_payload_bytes"]),
            source_id_codec,
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(header)
            handle.write(directory)
            for index, (layer, metadata) in enumerate(
                zip(packed_layers, layer_metadata, strict=True)
            ):
                tensor_payload = _layer_payload(layer)
                if len(tensor_payload) != metadata["tensor_payload_bytes"]:
                    raise AssertionError("packed layer differs from traffic accounting")
                layer_header = bytearray(_LAYER_HEADER_BYTES)
                _LAYER_HEADER.pack_into(
                    layer_header,
                    0,
                    _LAYER_MAGIC,
                    _VERSION,
                    index,
                    hidden_size,
                    int(metadata["width"]),
                    source_intermediate_size,
                    int(metadata["one_matrix_q4_bytes"]),
                    int(metadata["fp16_scale_bytes"]) // 3,
                    int(metadata["source_id_bytes"]),
                    source_id_codec,
                    int(metadata["layer_payload_bytes"]),
                )
                handle.write(layer_header)
                handle.write(tensor_payload)
                handle.write(
                    b"\0"
                    * (
                        int(metadata["layer_block_bytes"])
                        - int(metadata["layer_payload_bytes"])
                    )
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    if destination.stat().st_size != traffic["serialized_artifact_bytes"]:
        raise AssertionError("serialized compact artifact has an unexpected size")
    return destination


def load_width_pruned_q4_artifact(
    path: str | Path,
) -> LoadedWidthPrunedQ4Artifact:
    """Strictly validate and load a cache-aligned compact SwiGLU artifact."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read width-pruned Q4 artifact {source}") from exc
    if len(payload) < _HEADER_BYTES:
        raise ValueError("width-pruned Q4 artifact is shorter than its header")
    (
        magic,
        version,
        layer_count,
        hidden_size,
        source_intermediate_size,
        cache_line_bytes,
        directory_entry_bytes,
        directory_block_bytes,
        source_id_codec,
    ) = _HEADER.unpack_from(payload)
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("width-pruned Q4 artifact magic/version mismatch")
    if source_id_codec not in {_SOURCE_IDS_UINT16, _SOURCE_IDS_UINT32}:
        raise ValueError("width-pruned Q4 source-ID codec is unsupported")
    if directory_entry_bytes != _DIRECTORY_ENTRY.size:
        raise ValueError("width-pruned Q4 directory entry size is unsupported")
    _positive_integer(layer_count, "artifact layer_count")
    _positive_integer(hidden_size, "artifact hidden_size")
    _positive_integer(source_intermediate_size, "artifact source_intermediate_size")
    _positive_integer(cache_line_bytes, "artifact cache_line_bytes")
    expected_source_codec, source_id_dtype, source_id_bytes = _source_id_codec(
        source_intermediate_size
    )
    if source_id_codec != expected_source_codec:
        raise ValueError("width-pruned Q4 source-ID codec disagrees with dimensions")
    header_block_bytes = _align(_HEADER_BYTES, cache_line_bytes)
    expected_directory_bytes = _align(
        layer_count * _DIRECTORY_ENTRY.size, cache_line_bytes
    )
    if directory_block_bytes != expected_directory_bytes:
        raise ValueError("width-pruned Q4 directory block size is invalid")
    if len(payload) < header_block_bytes + directory_block_bytes:
        raise ValueError("width-pruned Q4 artifact is shorter than its directory")
    if any(payload[_HEADER.size : header_block_bytes]):
        raise ValueError("width-pruned Q4 header padding is non-zero")

    entries = []
    for index in range(layer_count):
        entry = _DIRECTORY_ENTRY.unpack_from(
            payload, header_block_bytes + index * _DIRECTORY_ENTRY.size
        )
        if entry[0] != index:
            raise ValueError("width-pruned Q4 directory indices are not canonical")
        entries.append(entry)
    directory_payload_end = header_block_bytes + layer_count * _DIRECTORY_ENTRY.size
    directory_end = header_block_bytes + directory_block_bytes
    if any(payload[directory_payload_end:directory_end]):
        raise ValueError("width-pruned Q4 directory padding is non-zero")
    widths = [int(entry[1]) for entry in entries]
    traffic = width_pruned_q4_traffic(
        hidden_size,
        source_intermediate_size,
        widths,
        cache_line_bytes=cache_line_bytes,
    )
    if len(payload) != traffic["serialized_artifact_bytes"]:
        raise ValueError("width-pruned Q4 artifact length does not match its header")

    packed_layers: list[PackedWidthPrunedQ4Layer] = []
    offsets: list[int] = []
    block_sizes: list[int] = []
    expected_offset = directory_end
    for index, (entry, metadata) in enumerate(
        zip(entries, traffic["layers"], strict=True)
    ):
        _, width, offset, block_bytes, layer_payload_bytes, entry_codec = entry
        if entry_codec != source_id_codec:
            raise ValueError("width-pruned Q4 layer source-ID codec is inconsistent")
        if (
            offset != expected_offset
            or offset % cache_line_bytes
            or block_bytes != metadata["layer_block_bytes"]
            or layer_payload_bytes != metadata["layer_payload_bytes"]
        ):
            raise ValueError("width-pruned Q4 layer directory entry is invalid")
        offsets.append(offset)
        block_sizes.append(block_bytes)
        (
            layer_magic,
            layer_version,
            layer_index,
            layer_hidden,
            layer_width,
            layer_source_intermediate,
            one_matrix_code_bytes,
            one_matrix_scale_bytes,
            layer_source_id_bytes,
            layer_source_codec,
            declared_payload_bytes,
        ) = _LAYER_HEADER.unpack_from(payload, offset)
        expected_header = (
            _LAYER_MAGIC,
            _VERSION,
            index,
            hidden_size,
            width,
            source_intermediate_size,
            metadata["one_matrix_q4_bytes"],
            metadata["fp16_scale_bytes"] // 3,
            metadata["source_id_bytes"],
            source_id_codec,
            metadata["layer_payload_bytes"],
        )
        actual_header = (
            layer_magic,
            layer_version,
            layer_index,
            layer_hidden,
            layer_width,
            layer_source_intermediate,
            one_matrix_code_bytes,
            one_matrix_scale_bytes,
            layer_source_id_bytes,
            layer_source_codec,
            declared_payload_bytes,
        )
        if actual_header != expected_header:
            raise ValueError("width-pruned Q4 layer header is invalid")
        if any(payload[offset + _LAYER_HEADER.size : offset + _LAYER_HEADER_BYTES]):
            raise ValueError("width-pruned Q4 layer header padding is non-zero")
        cursor = offset + _LAYER_HEADER_BYTES

        def codes() -> NDArray[np.uint8]:
            nonlocal cursor
            result = np.frombuffer(
                payload,
                dtype=np.uint8,
                count=one_matrix_code_bytes,
                offset=cursor,
            ).copy()
            cursor += one_matrix_code_bytes
            return result

        gate_codes, up_codes, down_codes = codes(), codes(), codes()

        def scales() -> NDArray[np.float16]:
            nonlocal cursor
            result = np.frombuffer(
                payload, dtype="<f2", count=width, offset=cursor
            ).copy()
            cursor += width * 2
            return result

        gate_scales, up_scales, down_scales = scales(), scales(), scales()
        ids = np.frombuffer(
            payload, dtype=source_id_dtype, count=width, offset=cursor
        ).copy()
        cursor += width * source_id_bytes
        if cursor != offset + layer_payload_bytes:
            raise AssertionError("internal width-pruned layer offset mismatch")
        if any(payload[cursor : offset + block_bytes]):
            raise ValueError("width-pruned Q4 layer padding is non-zero")
        layer = PackedWidthPrunedQ4Layer(
            PackedQ4Rows(gate_codes, gate_scales, width, hidden_size),
            PackedQ4Rows(up_codes, up_scales, width, hidden_size),
            PackedQ4Rows(down_codes, down_scales, width, hidden_size),
            ids,
        )
        _decode_symmetric_q4_rows(layer.gate)
        _decode_symmetric_q4_rows(layer.up)
        _decode_symmetric_q4_rows(layer.down_t)
        _source_ids(ids, width, source_intermediate_size, source_id_dtype)
        packed_layers.append(layer)
        expected_offset = offset + block_bytes
    if expected_offset != len(payload):
        raise AssertionError("internal width-pruned artifact length mismatch")
    return LoadedWidthPrunedQ4Artifact(
        tuple(packed_layers),
        hidden_size,
        source_intermediate_size,
        cache_line_bytes,
        source_id_codec,
        header_block_bytes,
        directory_block_bytes,
        tuple(offsets),
        tuple(block_sizes),
        len(payload),
    )


def decode_width_pruned_q4_artifact(
    artifact: LoadedWidthPrunedQ4Artifact,
) -> tuple[dict[str, NDArray[Any]], ...]:
    """Reference-decode every compact layer into logical float32 matrices."""

    return tuple(
        {
            "gate": _decode_symmetric_q4_rows(layer.gate),
            "up": _decode_symmetric_q4_rows(layer.up),
            "down": _decode_symmetric_q4_rows(layer.down_t).T,
            "source_ids": layer.source_ids.copy(),
        }
        for layer in artifact.layers
    )


def _silu(values: NDArray[np.float32]) -> NDArray[np.float32]:
    sigmoid = np.empty_like(values)
    positive = values >= 0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    sigmoid[~positive] = exponent / (1.0 + exponent)
    return values * sigmoid


def width_pruned_q4_forward(
    artifact: LoadedWidthPrunedQ4Artifact,
    layer_index: int,
    hidden: ArrayLike,
) -> NDArray[np.float32]:
    """Execute one compact SwiGLU layer from validated packed payloads."""

    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
        or layer_index >= len(artifact.layers)
    ):
        raise ValueError("layer_index is outside the artifact")
    states = _matrix(hidden, "hidden")
    if states.shape[1] != artifact.hidden_size:
        raise ValueError("hidden width does not match the width-pruned artifact")
    layer = artifact.layers[layer_index]
    gate = _decode_symmetric_q4_rows(layer.gate)
    up = _decode_symmetric_q4_rows(layer.up)
    down_t = _decode_symmetric_q4_rows(layer.down_t)
    activations = _silu(states @ gate.T) * (states @ up.T)
    return np.ascontiguousarray(activations @ down_t)


__all__ = [
    "LoadedWidthPrunedQ4Artifact",
    "PackedWidthPrunedQ4Layer",
    "WidthPrunedQ4LayerWeights",
    "decode_width_pruned_q4_artifact",
    "load_width_pruned_q4_artifact",
    "save_width_pruned_q4_artifact",
    "width_pruned_q4_forward",
    "width_pruned_q4_traffic",
]
