"""Strict cache-aligned signed-Q3 artifact for compact SwiGLU layers.

The codec stores each logical row as an independently byte-padded three-bit
stream with a positive FP16 row scale.  Codes use symmetric values [-3, 3];
the remaining three-bit pattern (-4) is forbidden and rejected on load.  The
complete serialized file is the cold-traffic numerator, while the comparison
denominator is the ideal code-only dense-Q4 source MLP.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

_MAGIC = b"ENGFWQ31"
_LAYER_MAGIC = b"ENGFWL31"
_VERSION = 1
_HEADER_BYTES = 64
_LAYER_HEADER_BYTES = 64
_HEADER = struct.Struct("<8s8I")
_DIRECTORY_ENTRY = struct.Struct("<IIQQII")
_LAYER_HEADER = struct.Struct("<8s10I")
_Q3_CODEC = 1
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
class PackedQ3Rows:
    packed: NDArray[np.uint8]
    scales: NDArray[np.float16]
    rows: int
    columns: int

    @property
    def bytes_per_row(self) -> int:
        return (self.columns * 3 + 7) // 8


def _pack_signed_q3_rows(values: ArrayLike) -> PackedQ3Rows:
    """Pack per-row symmetric Q3 using FP16 scales and codes [-3, 3]."""

    matrix = _matrix(values, "Q3 matrix")
    maximum = np.max(np.abs(matrix), axis=1)
    scale32 = maximum / np.float32(3.0)
    scale32[maximum == 0.0] = np.float32(1.0)
    scales = scale32.astype(np.float16)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Q3 row scale cannot be represented as positive FP16")
    signed = np.clip(
        np.rint(matrix / scales.astype(np.float32)[:, None]), -3, 3
    ).astype(np.int8)
    encoded = (signed.astype(np.int16) & 0x07).astype(np.uint16)
    rows, columns = matrix.shape
    row_bytes = (columns * 3 + 7) // 8
    packed = np.zeros((rows, row_bytes), dtype=np.uint8)
    bit_offsets = np.arange(columns, dtype=np.int64) * 3
    byte_indices = bit_offsets // 8
    shifts = bit_offsets % 8
    row_offsets = np.arange(rows, dtype=np.int64)[:, None] * row_bytes
    flat = packed.reshape(-1)
    low_indices = row_offsets + byte_indices[None, :]
    low_values = ((encoded << shifts[None, :]) & 0xFF).astype(np.uint8)
    np.bitwise_or.at(flat, low_indices.reshape(-1), low_values.reshape(-1))
    crossing = shifts > 5
    if np.any(crossing):
        high_indices = row_offsets + byte_indices[None, crossing] + 1
        high_values = (encoded[:, crossing] >> (8 - shifts[crossing])[None, :]).astype(
            np.uint8
        )
        np.bitwise_or.at(flat, high_indices.reshape(-1), high_values.reshape(-1))
    return PackedQ3Rows(
        np.ascontiguousarray(flat),
        np.ascontiguousarray(scales),
        rows,
        columns,
    )


def _decode_signed_q3_rows(packed: PackedQ3Rows) -> NDArray[np.float32]:
    row_bytes = (packed.columns * 3 + 7) // 8
    expected = packed.rows * row_bytes
    codes = np.asarray(packed.packed)
    scales = np.asarray(packed.scales)
    if codes.dtype != np.uint8 or codes.ndim != 1 or codes.size != expected:
        raise ValueError("packed Q3 code length does not match its row shape")
    if scales.dtype != np.float16 or scales.shape != (packed.rows,):
        raise ValueError("packed Q3 scales must be one FP16 value per row")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("packed Q3 scales must be finite and positive")
    rows = codes.reshape(packed.rows, row_bytes)
    used_tail_bits = (packed.columns * 3) % 8
    if used_tail_bits:
        padding_mask = np.uint8(0xFF ^ ((1 << used_tail_bits) - 1))
        if np.any(rows[:, -1] & padding_mask):
            raise ValueError("non-zero Q3 row padding bits")
    bit_offsets = np.arange(packed.columns, dtype=np.int64) * 3
    byte_indices = bit_offsets // 8
    shifts = bit_offsets % 8
    encoded = ((rows[:, byte_indices] >> shifts[None, :]) & 0x07).astype(np.uint8)
    crossing = shifts > 5
    if np.any(crossing):
        high = (
            (
                rows[:, byte_indices[crossing] + 1].astype(np.uint16)
                << (8 - shifts[crossing])[None, :]
            )
            & np.uint16(0x07)
        ).astype(np.uint8)
        encoded[:, crossing] |= high
    if np.any(encoded == 4):
        raise ValueError("packed Q3 artifact contains forbidden -4 codes")
    signed = encoded.astype(np.int8)
    signed[signed >= 4] -= np.int8(8)
    if np.any(signed < -3) or np.any(signed > 3):
        raise ValueError("packed Q3 artifact contains an out-of-range code")
    decoded = signed.astype(np.float32) * scales.astype(np.float32)[:, None]
    return np.ascontiguousarray(decoded)


@dataclass(frozen=True)
class WidthPrunedQ3LayerWeights:
    gate: ArrayLike
    up: ArrayLike
    down: ArrayLike
    source_ids: ArrayLike


@dataclass(frozen=True)
class PackedWidthPrunedQ3Layer:
    gate: PackedQ3Rows
    up: PackedQ3Rows
    down_t: PackedQ3Rows
    source_ids: NDArray[Any]


@dataclass(frozen=True)
class LoadedWidthPrunedQ3Artifact:
    layers: tuple[PackedWidthPrunedQ3Layer, ...]
    hidden_size: int
    source_intermediate_size: int
    cache_line_bytes: int
    source_id_codec: int
    header_block_bytes: int
    directory_block_bytes: int
    layer_offsets: tuple[int, ...]
    layer_block_bytes: tuple[int, ...]
    serialized_artifact_bytes: int


def width_pruned_q3_traffic(
    hidden_size: int,
    source_intermediate_size: int,
    layer_widths: list[int] | tuple[int, ...],
    *,
    cache_line_bytes: int = 64,
) -> dict[str, Any]:
    """Return exact complete-file cold bytes relative to ideal dense Q4."""

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
    q3_bytes_per_row = (hidden_size * 3 + 7) // 8
    layers = []
    total_layer_block_bytes = 0
    for index, width in enumerate(widths):
        one_matrix_q3_bytes = width * q3_bytes_per_row
        q3_code_bytes = 3 * one_matrix_q3_bytes
        fp16_scale_bytes = 3 * width * 2
        ids_bytes = width * source_id_bytes
        tensor_payload_bytes = q3_code_bytes + fp16_scale_bytes + ids_bytes
        layer_payload_bytes = _LAYER_HEADER_BYTES + tensor_payload_bytes
        layer_block_bytes = _align(layer_payload_bytes, cache_line_bytes)
        total_layer_block_bytes += layer_block_bytes
        layers.append(
            {
                "layer": index,
                "width": width,
                "q3_bytes_per_row": q3_bytes_per_row,
                "one_matrix_q3_bytes": one_matrix_q3_bytes,
                "q3_code_bytes": q3_code_bytes,
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
        "layout": "width_pruned_cache_aligned_signed_q3_v1",
        "codec": "symmetric_signed_q3_per_row_fp16_scale",
        "bits_per_weight": 3,
        "layers": layers,
        "layer_count": len(widths),
        "layer_widths": list(widths),
        "hidden_size": hidden_size,
        "source_intermediate_size": source_intermediate_size,
        "source_id_codec": (
            "uint16" if source_id_codec == _SOURCE_IDS_UINT16 else "uint32"
        ),
        "source_id_bytes_per_record": source_id_bytes,
        "row_scale_codec": "fp16",
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
            "cold numerator is the complete serialized artifact, including byte-padded "
            "signed-Q3 rows, FP16 row scales, source IDs, headers, directory, and "
            "cache-line padding; denominator is ideal code-only dense-Q4 source weights"
        ),
    }


def width_pruned_q3_dynamic_traffic(
    hidden_size: int,
    source_intermediate_size: int,
    selected_records: int,
    *,
    layer_count: int,
    router_bytes_per_token: int = 0,
    cache_line_bytes: int = 64,
) -> dict[str, int | float | bool | str]:
    """Return an ideal gathered-row budget with caller-supplied router bytes.

    This helper is a lower bound for a dynamic exact-record kernel: selected
    row payloads are assumed to be gathered/staged densely and only the whole
    per-layer gathered payload is cache-line rounded.  It deliberately does
    not claim the cache behavior of arbitrary row reads from the full artifact.
    """

    for value, name in (
        (hidden_size, "hidden_size"),
        (source_intermediate_size, "source_intermediate_size"),
        (selected_records, "selected_records"),
        (layer_count, "layer_count"),
        (cache_line_bytes, "cache_line_bytes"),
    ):
        _positive_integer(value, name)
    if selected_records > source_intermediate_size:
        raise ValueError("selected_records cannot exceed source_intermediate_size")
    if (
        isinstance(router_bytes_per_token, bool)
        or not isinstance(router_bytes_per_token, int)
        or router_bytes_per_token < 0
    ):
        raise ValueError("router_bytes_per_token must be a non-negative integer")
    if cache_line_bytes < _HEADER_BYTES or cache_line_bytes % _HEADER_BYTES:
        raise ValueError("cache_line_bytes must be a positive multiple of 64")
    _, _, source_id_bytes = _source_id_codec(source_intermediate_size)
    q3_bytes_per_row_per_matrix = (hidden_size * 3 + 7) // 8
    q3_bytes_per_record = 3 * q3_bytes_per_row_per_matrix
    fp16_scale_bytes_per_record = 3 * 2
    metadata_bytes_per_record = fp16_scale_bytes_per_record + source_id_bytes
    selected_q3_code_bytes = layer_count * selected_records * q3_bytes_per_record
    selected_metadata_bytes = layer_count * selected_records * metadata_bytes_per_record
    layer_payload_bytes = selected_records * (
        q3_bytes_per_record + metadata_bytes_per_record
    )
    layer_cache_aligned_bytes = _align(layer_payload_bytes, cache_line_bytes)
    selected_cache_aligned_bytes = layer_count * layer_cache_aligned_bytes
    router_cache_aligned_bytes = (
        _align(router_bytes_per_token, cache_line_bytes)
        if router_bytes_per_token
        else 0
    )
    total_cold_bytes = selected_cache_aligned_bytes + router_cache_aligned_bytes
    dense_q4_source_mlp_bytes = (
        layer_count * 3 * hidden_size * source_intermediate_size + 1
    ) // 2
    code_only_fraction = selected_q3_code_bytes / dense_q4_source_mlp_bytes
    total_fraction = total_cold_bytes / dense_q4_source_mlp_bytes
    return {
        "layout": "ideal_dense_gathered_signed_q3_rows_plus_router_v1",
        "selected_records": selected_records,
        "layer_count": layer_count,
        "q3_bytes_per_row_per_matrix": q3_bytes_per_row_per_matrix,
        "q3_bytes_per_record": q3_bytes_per_record,
        "fp16_scale_bytes_per_record": fp16_scale_bytes_per_record,
        "source_id_bytes_per_record": source_id_bytes,
        "metadata_bytes_per_record": metadata_bytes_per_record,
        "selected_q3_code_bytes": selected_q3_code_bytes,
        "selected_metadata_bytes": selected_metadata_bytes,
        "selected_layer_payload_bytes": layer_payload_bytes,
        "selected_layer_cache_aligned_bytes": layer_cache_aligned_bytes,
        "selected_cache_aligned_bytes": selected_cache_aligned_bytes,
        "router_bytes_per_token": router_bytes_per_token,
        "router_cache_aligned_bytes": router_cache_aligned_bytes,
        "total_cold_bytes": total_cold_bytes,
        "dense_q4_source_mlp_bytes": dense_q4_source_mlp_bytes,
        "code_only_fraction_of_dense_q4": code_only_fraction,
        "fraction_of_dense_q4": total_fraction,
        "remaining_fraction_before_45_percent": 0.45 - total_fraction,
        "passes_45_percent_traffic_gate": total_fraction <= 0.45,
        "accounting_policy": (
            "ideal gathered-row lower bound with one cache-line-rounded selected "
            "payload per layer and a cache-line-rounded caller-supplied router; "
            "arbitrary-source-row cache-line amplification is excluded"
        ),
    }


def _layer_payload(layer: PackedWidthPrunedQ3Layer) -> bytes:
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


def save_width_pruned_q3_artifact(
    path: str | Path,
    layers: list[WidthPrunedQ3LayerWeights] | tuple[WidthPrunedQ3LayerWeights, ...],
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
    packed_layers: list[PackedWidthPrunedQ3Layer] = []
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
            PackedWidthPrunedQ3Layer(
                _pack_signed_q3_rows(gate),
                _pack_signed_q3_rows(up),
                _pack_signed_q3_rows(down.T),
                ids,
            )
        )
    assert hidden_size is not None
    traffic = width_pruned_q3_traffic(
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
        raise AssertionError("internal compact Q3 artifact accounting mismatch")
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
                    raise AssertionError("packed Q3 layer differs from accounting")
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
                    int(metadata["one_matrix_q3_bytes"]),
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
        raise AssertionError("serialized compact Q3 artifact has an unexpected size")
    return destination


def load_width_pruned_q3_artifact(
    path: str | Path,
) -> LoadedWidthPrunedQ3Artifact:
    """Strictly validate and load a cache-aligned compact Q3 artifact."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read width-pruned Q3 artifact {source}") from exc
    if len(payload) < _HEADER_BYTES:
        raise ValueError("width-pruned Q3 artifact is shorter than its header")
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
        raise ValueError("width-pruned Q3 artifact magic/version mismatch")
    if source_id_codec not in {_SOURCE_IDS_UINT16, _SOURCE_IDS_UINT32}:
        raise ValueError("width-pruned Q3 source-ID codec is unsupported")
    if directory_entry_bytes != _DIRECTORY_ENTRY.size:
        raise ValueError("width-pruned Q3 directory entry size is unsupported")
    _positive_integer(layer_count, "artifact layer_count")
    _positive_integer(hidden_size, "artifact hidden_size")
    _positive_integer(source_intermediate_size, "artifact source_intermediate_size")
    _positive_integer(cache_line_bytes, "artifact cache_line_bytes")
    expected_source_codec, source_id_dtype, source_id_bytes = _source_id_codec(
        source_intermediate_size
    )
    if source_id_codec != expected_source_codec:
        raise ValueError("width-pruned Q3 source-ID codec disagrees with dimensions")
    header_block_bytes = _align(_HEADER_BYTES, cache_line_bytes)
    expected_directory_bytes = _align(
        layer_count * _DIRECTORY_ENTRY.size, cache_line_bytes
    )
    if directory_block_bytes != expected_directory_bytes:
        raise ValueError("width-pruned Q3 directory block size is invalid")
    if len(payload) < header_block_bytes + directory_block_bytes:
        raise ValueError("width-pruned Q3 artifact is shorter than its directory")
    if any(payload[_HEADER.size : header_block_bytes]):
        raise ValueError("width-pruned Q3 header padding is non-zero")
    entries = []
    for index in range(layer_count):
        entry = _DIRECTORY_ENTRY.unpack_from(
            payload, header_block_bytes + index * _DIRECTORY_ENTRY.size
        )
        if entry[0] != index:
            raise ValueError("width-pruned Q3 directory indices are not canonical")
        entries.append(entry)
    directory_payload_end = header_block_bytes + layer_count * _DIRECTORY_ENTRY.size
    directory_end = header_block_bytes + directory_block_bytes
    if any(payload[directory_payload_end:directory_end]):
        raise ValueError("width-pruned Q3 directory padding is non-zero")
    widths = [int(entry[1]) for entry in entries]
    traffic = width_pruned_q3_traffic(
        hidden_size,
        source_intermediate_size,
        widths,
        cache_line_bytes=cache_line_bytes,
    )
    if len(payload) != traffic["serialized_artifact_bytes"]:
        raise ValueError("width-pruned Q3 artifact length does not match its header")
    packed_layers: list[PackedWidthPrunedQ3Layer] = []
    offsets: list[int] = []
    block_sizes: list[int] = []
    expected_offset = directory_end
    for index, (entry, metadata) in enumerate(
        zip(entries, traffic["layers"], strict=True)
    ):
        _, width, offset, block_bytes, layer_payload_bytes, entry_codec = entry
        if entry_codec != source_id_codec:
            raise ValueError("width-pruned Q3 layer source-ID codec is inconsistent")
        if (
            offset != expected_offset
            or offset % cache_line_bytes
            or block_bytes != metadata["layer_block_bytes"]
            or layer_payload_bytes != metadata["layer_payload_bytes"]
        ):
            raise ValueError("width-pruned Q3 layer directory entry is invalid")
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
            metadata["one_matrix_q3_bytes"],
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
            raise ValueError("width-pruned Q3 layer header is invalid")
        if any(payload[offset + _LAYER_HEADER.size : offset + _LAYER_HEADER_BYTES]):
            raise ValueError("width-pruned Q3 layer header padding is non-zero")
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
            raise AssertionError("internal width-pruned Q3 layer offset mismatch")
        if any(payload[cursor : offset + block_bytes]):
            raise ValueError("width-pruned Q3 layer padding is non-zero")
        layer = PackedWidthPrunedQ3Layer(
            PackedQ3Rows(gate_codes, gate_scales, width, hidden_size),
            PackedQ3Rows(up_codes, up_scales, width, hidden_size),
            PackedQ3Rows(down_codes, down_scales, width, hidden_size),
            ids,
        )
        _decode_signed_q3_rows(layer.gate)
        _decode_signed_q3_rows(layer.up)
        _decode_signed_q3_rows(layer.down_t)
        _source_ids(ids, width, source_intermediate_size, source_id_dtype)
        packed_layers.append(layer)
        expected_offset = offset + block_bytes
    if expected_offset != len(payload):
        raise AssertionError("internal width-pruned Q3 artifact length mismatch")
    return LoadedWidthPrunedQ3Artifact(
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


def decode_width_pruned_q3_artifact(
    artifact: LoadedWidthPrunedQ3Artifact,
) -> tuple[dict[str, NDArray[Any]], ...]:
    """Reference-decode every compact layer into logical float32 matrices."""

    return tuple(
        {
            "gate": _decode_signed_q3_rows(layer.gate),
            "up": _decode_signed_q3_rows(layer.up),
            "down": _decode_signed_q3_rows(layer.down_t).T,
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


def width_pruned_q3_forward(
    artifact: LoadedWidthPrunedQ3Artifact,
    layer_index: int,
    hidden: ArrayLike,
) -> NDArray[np.float32]:
    """Execute one compact SwiGLU layer from validated packed Q3 payloads."""

    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
        or layer_index >= len(artifact.layers)
    ):
        raise ValueError("layer_index is outside the artifact")
    states = _matrix(hidden, "hidden")
    if states.shape[1] != artifact.hidden_size:
        raise ValueError("hidden width does not match the width-pruned Q3 artifact")
    layer = artifact.layers[layer_index]
    gate = _decode_signed_q3_rows(layer.gate)
    up = _decode_signed_q3_rows(layer.up)
    down_t = _decode_signed_q3_rows(layer.down_t)
    activations = _silu(states @ gate.T) * (states @ up.T)
    return np.ascontiguousarray(activations @ down_t)


__all__ = [
    "LoadedWidthPrunedQ3Artifact",
    "PackedQ3Rows",
    "PackedWidthPrunedQ3Layer",
    "WidthPrunedQ3LayerWeights",
    "decode_width_pruned_q3_artifact",
    "load_width_pruned_q3_artifact",
    "save_width_pruned_q3_artifact",
    "width_pruned_q3_dynamic_traffic",
    "width_pruned_q3_forward",
    "width_pruned_q3_traffic",
]
