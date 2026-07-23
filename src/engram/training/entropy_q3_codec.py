"""Strict sequential-key / random-down signed-Q3 entropy artifact.

Each layer stores gate and up as one deterministic static-model rANS stream,
with separate FP16 row-scale arrays.  Down rows are independent rANS records
with FP16 scales and 64-byte alignment so a runtime can evaluate every key,
select exact records, and read only the selected values.  Headers, model and
offset tables, checksums, padding, and canonical rANS termination are all
validated before an artifact is exposed.
"""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.training.canonical_rans import (
    normalized_frequencies,
    rans_decode,
    rans_encode,
)

_MAGIC = b"ENGEQ3S1"
_LAYER_MAGIC = b"ENGEQL31"
_MODEL_MAGIC = b"EQ3M"
_STREAM_MAGIC = b"EQ3S"
_VECTOR_MAGIC = b"EQ3V"
_RECORD_MAGIC = b"EQ3R"
_VERSION = 1
_CODEC = 1
_SCALE_BITS = 12
_ALPHABET = 7
_HEADER_BYTES = 64
_LAYER_HEADER_BYTES = 64
_MODEL_BLOCK_BYTES = 64
_STREAM_HEADER_BYTES = 32
_VECTOR_HEADER_BYTES = 16
_RECORD_HEADER_BYTES = 32
_HEADER = struct.Struct("<8s10I")
_DIRECTORY_ENTRY = struct.Struct("<IIQQII")
_LAYER_HEADER = struct.Struct("<8s14I")
_MODEL_HEADER = struct.Struct("<4sHBB21HI")
_STREAM_HEADER = struct.Struct("<4sHHIII")
_VECTOR_HEADER = struct.Struct("<4sHHII")
_RECORD_HEADER = struct.Struct("<4sHHIIHHIII")
_GATE = 1
_UP = 2
_DOWN = 3
_DOWN_NORM = 4


def _crc(payload: bytes | bytearray | memoryview) -> int:
    return zlib.crc32(payload) & 0xFFFFFFFF


def _global_integrity_crc(
    magic: bytes,
    version: int,
    layer_count: int,
    hidden_size: int,
    source_intermediate_size: int,
    cache_line_bytes: int,
    directory_entry_bytes: int,
    directory_block_bytes: int,
    codec: int,
    scale_bits: int,
    directory_payload: bytes | bytearray | memoryview,
) -> int:
    metadata = struct.pack(
        "<8s9I",
        magic,
        version,
        layer_count,
        hidden_size,
        source_intermediate_size,
        cache_line_bytes,
        directory_entry_bytes,
        directory_block_bytes,
        codec,
        scale_bits,
    )
    return _crc(metadata + bytes(directory_payload))


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
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix)


def _source_ids(
    values: ArrayLike, width: int, source_intermediate_size: int
) -> NDArray[np.uint32]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()  # type: ignore[union-attr]
    array = np.asarray(values)
    if array.shape != (width,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("source_ids must contain one integer per record")
    converted = array.astype(np.int64, copy=False)
    if (
        np.any(converted < 0)
        or np.any(converted >= source_intermediate_size)
        or len(np.unique(converted)) != width
    ):
        raise ValueError("source_ids must be unique source-record indices")
    return np.ascontiguousarray(converted.astype("<u4"))


@dataclass(frozen=True)
class _QuantizedRows:
    symbols: NDArray[np.uint8]
    scales: NDArray[np.float16]


def _quantize_rows(values: ArrayLike) -> _QuantizedRows:
    matrix = _matrix(values, "entropy-Q3 matrix")
    maximum = np.max(np.abs(matrix), axis=1)
    scale32 = maximum / np.float32(3.0)
    scale32[maximum == 0.0] = np.float32(1.0)
    scales = scale32.astype(np.float16)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("entropy-Q3 row scale cannot be represented as FP16")
    signed = np.clip(
        np.rint(matrix / scales.astype(np.float32)[:, None]), -3, 3
    ).astype(np.int8)
    symbols = (signed.astype(np.int16) + 3).astype(np.uint8)
    return _QuantizedRows(np.ascontiguousarray(symbols), np.ascontiguousarray(scales))


def _decode_symbols(
    symbols: NDArray[np.uint8], scales: NDArray[np.float16]
) -> NDArray[np.float32]:
    values = np.asarray(symbols)
    row_scales = np.asarray(scales)
    if values.ndim != 2 or values.dtype != np.uint8:
        raise ValueError("entropy-Q3 symbols must be a uint8 matrix")
    if np.any(values >= _ALPHABET):
        raise ValueError("entropy-Q3 stream contains a forbidden signed code")
    if row_scales.dtype != np.float16 or row_scales.shape != (len(values),):
        raise ValueError("entropy-Q3 scales must contain one FP16 value per row")
    if not np.all(np.isfinite(row_scales)) or np.any(row_scales <= 0):
        raise ValueError("entropy-Q3 scales must be finite and positive")
    signed = values.astype(np.int8) - np.int8(3)
    decoded = signed.astype(np.float32) * row_scales.astype(np.float32)[:, None]
    return np.ascontiguousarray(decoded)


def _row_norms(values: NDArray[np.float32]) -> NDArray[np.float16]:
    """Return platform-stable FP16 row norms using explicit FP64 accumulation."""

    matrix = np.asarray(values, dtype=np.float64)
    return np.sqrt(np.sum(matrix * matrix, axis=1, dtype=np.float64)).astype(np.float16)


@dataclass(frozen=True)
class EntropyQ3LayerWeights:
    gate: ArrayLike
    up: ArrayLike
    down: ArrayLike
    source_ids: ArrayLike


@dataclass(frozen=True)
class EncodedEntropyQ3Record:
    encoded: bytes
    scale: np.float16
    source_id: int
    offset: int
    block_bytes: int
    symbol_crc32: int


@dataclass(frozen=True)
class LoadedEntropyQ3Layer:
    frequencies: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    gate_stream: bytes
    up_stream: bytes
    gate_scales: NDArray[np.float16]
    up_scales: NDArray[np.float16]
    down_norms: NDArray[np.float16]
    down_records: tuple[EncodedEntropyQ3Record, ...]
    hidden_size: int
    width: int
    source_intermediate_size: int
    layer_offset: int
    layer_block_bytes: int
    gate_stream_block_bytes: int
    up_stream_block_bytes: int
    scale_block_bytes: int
    norm_block_bytes: int
    offset_block_bytes: int


@dataclass(frozen=True)
class LoadedEntropyQ3Artifact:
    layers: tuple[LoadedEntropyQ3Layer, ...]
    hidden_size: int
    source_intermediate_size: int
    cache_line_bytes: int
    header_block_bytes: int
    directory_block_bytes: int
    serialized_artifact_bytes: int


@dataclass(frozen=True)
class _PreparedLayer:
    frequencies: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    model_block: bytes
    gate_stream_block: bytes
    up_stream_block: bytes
    gate_scale_block: bytes
    up_scale_block: bytes
    norm_block: bytes
    record_blocks: tuple[bytes, ...]
    width: int
    hidden_size: int
    source_ids: NDArray[np.uint32]


def _model_block(
    frequencies: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> bytes:
    flattened = tuple(value for table in frequencies for value in table)
    if len(flattened) != 21 or any(value <= 0 for value in flattened):
        raise ValueError("entropy-Q3 model must contain three positive 7-symbol tables")
    raw = np.asarray(flattened, dtype="<u2").tobytes()
    block = bytearray(_MODEL_BLOCK_BYTES)
    _MODEL_HEADER.pack_into(
        block,
        0,
        _MODEL_MAGIC,
        _VERSION,
        _SCALE_BITS,
        _ALPHABET,
        *flattened,
        _crc(raw),
    )
    return bytes(block)


def _stream_block(
    encoded: bytes,
    symbols: NDArray[np.uint8],
    projection: int,
    cache_line_bytes: int,
) -> bytes:
    block_bytes = _align(_STREAM_HEADER_BYTES + len(encoded), cache_line_bytes)
    block = bytearray(block_bytes)
    _STREAM_HEADER.pack_into(
        block,
        0,
        _STREAM_MAGIC,
        _VERSION,
        projection,
        len(encoded),
        _crc(symbols.tobytes()),
        _crc(encoded),
    )
    block[_STREAM_HEADER_BYTES : _STREAM_HEADER_BYTES + len(encoded)] = encoded
    return bytes(block)


def _vector_block(
    values: NDArray[np.float16],
    kind: int,
    cache_line_bytes: int,
) -> bytes:
    raw = np.ascontiguousarray(values.astype("<f2", copy=False)).tobytes()
    block_bytes = _align(_VECTOR_HEADER_BYTES + len(raw), cache_line_bytes)
    block = bytearray(block_bytes)
    _VECTOR_HEADER.pack_into(
        block, 0, _VECTOR_MAGIC, _VERSION, kind, len(values), _crc(raw)
    )
    block[_VECTOR_HEADER_BYTES : _VECTOR_HEADER_BYTES + len(raw)] = raw
    return bytes(block)


def _record_block(
    encoded: bytes,
    symbols: NDArray[np.uint8],
    scale: np.float16,
    *,
    layer_index: int,
    record_index: int,
    source_id: int,
    columns: int,
    cache_line_bytes: int,
) -> bytes:
    scale_bytes = np.asarray([scale], dtype="<f2").tobytes()
    symbol_crc = _crc(symbols.tobytes())
    integrity_payload = (
        struct.pack(
            "<HHIIHHII",
            _VERSION,
            layer_index,
            record_index,
            source_id,
            columns,
            _CODEC,
            len(encoded),
            symbol_crc,
        )
        + scale_bytes
        + encoded
    )
    payload_bytes = _RECORD_HEADER_BYTES + len(scale_bytes) + len(encoded)
    block = bytearray(_align(payload_bytes, cache_line_bytes))
    _RECORD_HEADER.pack_into(
        block,
        0,
        _RECORD_MAGIC,
        _VERSION,
        layer_index,
        record_index,
        source_id,
        columns,
        _CODEC,
        len(encoded),
        symbol_crc,
        _crc(integrity_payload),
    )
    block[_RECORD_HEADER_BYTES : _RECORD_HEADER_BYTES + 2] = scale_bytes
    start = _RECORD_HEADER_BYTES + 2
    block[start : start + len(encoded)] = encoded
    return bytes(block)


def _prepare_layer(
    layer: EntropyQ3LayerWeights,
    layer_index: int,
    source_intermediate_size: int,
    cache_line_bytes: int,
) -> _PreparedLayer:
    gate = _matrix(layer.gate, f"layers[{layer_index}].gate")
    up = _matrix(layer.up, f"layers[{layer_index}].up")
    down = _matrix(layer.down, f"layers[{layer_index}].down")
    if gate.shape != up.shape:
        raise ValueError(f"gate/up shape mismatch at layer {layer_index}")
    width, hidden_size = gate.shape
    if down.shape != (hidden_size, width):
        raise ValueError(f"down shape mismatch at layer {layer_index}")
    if width > source_intermediate_size:
        raise ValueError(
            f"width exceeds source intermediate size at layer {layer_index}"
        )
    source_ids = _source_ids(layer.source_ids, width, source_intermediate_size)
    quantized = (
        _quantize_rows(gate),
        _quantize_rows(up),
        _quantize_rows(down.T),
    )
    frequencies = tuple(
        normalized_frequencies(rows.symbols.reshape(-1)) for rows in quantized
    )
    typed_frequencies = (
        frequencies[0],
        frequencies[1],
        frequencies[2],
    )
    gate_encoded = rans_encode(quantized[0].symbols.reshape(-1), typed_frequencies[0])
    up_encoded = rans_encode(quantized[1].symbols.reshape(-1), typed_frequencies[1])
    decoded_down = _decode_symbols(quantized[2].symbols, quantized[2].scales)
    down_norms = _row_norms(decoded_down)
    if np.any(~np.isfinite(down_norms)) or np.any(down_norms < 0):
        raise ValueError("entropy-Q3 down norms cannot be represented as FP16")
    records = []
    for record_index in range(width):
        symbols = quantized[2].symbols[record_index]
        encoded = rans_encode(symbols, typed_frequencies[2])
        records.append(
            _record_block(
                encoded,
                symbols,
                quantized[2].scales[record_index],
                layer_index=layer_index,
                record_index=record_index,
                source_id=int(source_ids[record_index]),
                columns=hidden_size,
                cache_line_bytes=cache_line_bytes,
            )
        )
    return _PreparedLayer(
        typed_frequencies,
        _model_block(typed_frequencies),
        _stream_block(gate_encoded, quantized[0].symbols, _GATE, cache_line_bytes),
        _stream_block(up_encoded, quantized[1].symbols, _UP, cache_line_bytes),
        _vector_block(quantized[0].scales, _GATE, cache_line_bytes),
        _vector_block(quantized[1].scales, _UP, cache_line_bytes),
        _vector_block(down_norms, _DOWN_NORM, cache_line_bytes),
        tuple(records),
        width,
        hidden_size,
        source_ids,
    )


def save_entropy_q3_artifact(
    path: str | Path,
    layers: list[EntropyQ3LayerWeights] | tuple[EntropyQ3LayerWeights, ...],
    *,
    source_intermediate_size: int,
    cache_line_bytes: int = 64,
) -> Path:
    """Atomically write the strict sequential-key/random-down entropy artifact."""

    _positive_integer(source_intermediate_size, "source_intermediate_size")
    _positive_integer(cache_line_bytes, "cache_line_bytes")
    if cache_line_bytes != 64:
        raise ValueError("cache_line_bytes must be exactly 64")
    logical_layers = tuple(layers)
    if not logical_layers:
        raise ValueError("layers must not be empty")
    prepared = tuple(
        _prepare_layer(layer, index, source_intermediate_size, cache_line_bytes)
        for index, layer in enumerate(logical_layers)
    )
    hidden_size = prepared[0].hidden_size
    if any(layer.hidden_size != hidden_size for layer in prepared):
        raise ValueError("all entropy-Q3 layers must share one hidden size")
    header_block_bytes = _align(_HEADER_BYTES, cache_line_bytes)
    directory_block_bytes = _align(
        len(prepared) * _DIRECTORY_ENTRY.size, cache_line_bytes
    )
    metadata = []
    cursor = header_block_bytes + directory_block_bytes
    for layer in prepared:
        offset_block_bytes = _align((layer.width + 1) * 8, cache_line_bytes)
        down_records_bytes = sum(len(record) for record in layer.record_blocks)
        layer_block_bytes = (
            _LAYER_HEADER_BYTES
            + len(layer.model_block)
            + len(layer.gate_stream_block)
            + len(layer.up_stream_block)
            + len(layer.gate_scale_block)
            + len(layer.up_scale_block)
            + len(layer.norm_block)
            + offset_block_bytes
            + down_records_bytes
        )
        metadata.append(
            {
                "offset": cursor,
                "offset_block_bytes": offset_block_bytes,
                "down_records_bytes": down_records_bytes,
                "layer_block_bytes": layer_block_bytes,
            }
        )
        cursor += layer_block_bytes
    directory = bytearray(directory_block_bytes)
    for index, (layer, values) in enumerate(zip(prepared, metadata, strict=True)):
        _DIRECTORY_ENTRY.pack_into(
            directory,
            index * _DIRECTORY_ENTRY.size,
            index,
            layer.width,
            values["offset"],
            values["layer_block_bytes"],
            values["down_records_bytes"],
            _CODEC,
        )
    header = bytearray(header_block_bytes)
    _HEADER.pack_into(
        header,
        0,
        _MAGIC,
        _VERSION,
        len(prepared),
        hidden_size,
        source_intermediate_size,
        cache_line_bytes,
        _DIRECTORY_ENTRY.size,
        directory_block_bytes,
        _global_integrity_crc(
            _MAGIC,
            _VERSION,
            len(prepared),
            hidden_size,
            source_intermediate_size,
            cache_line_bytes,
            _DIRECTORY_ENTRY.size,
            directory_block_bytes,
            _CODEC,
            _SCALE_BITS,
            directory[: len(prepared) * _DIRECTORY_ENTRY.size],
        ),
        _CODEC,
        _SCALE_BITS,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(header)
            handle.write(directory)
            for index, (layer, values) in enumerate(
                zip(prepared, metadata, strict=True)
            ):
                records_start = (
                    values["offset"]
                    + _LAYER_HEADER_BYTES
                    + len(layer.model_block)
                    + len(layer.gate_stream_block)
                    + len(layer.up_stream_block)
                    + len(layer.gate_scale_block)
                    + len(layer.up_scale_block)
                    + len(layer.norm_block)
                    + values["offset_block_bytes"]
                )
                record_offsets = [records_start]
                for record in layer.record_blocks:
                    record_offsets.append(record_offsets[-1] + len(record))
                offset_payload = np.asarray(record_offsets, dtype="<u8").tobytes()
                offset_block = bytearray(values["offset_block_bytes"])
                offset_block[: len(offset_payload)] = offset_payload
                layer_header = bytearray(_LAYER_HEADER_BYTES)
                gate_encoded = _STREAM_HEADER.unpack_from(layer.gate_stream_block)[3]
                up_encoded = _STREAM_HEADER.unpack_from(layer.up_stream_block)[3]
                _LAYER_HEADER.pack_into(
                    layer_header,
                    0,
                    _LAYER_MAGIC,
                    _VERSION,
                    index,
                    hidden_size,
                    layer.width,
                    gate_encoded,
                    len(layer.gate_stream_block),
                    up_encoded,
                    len(layer.up_stream_block),
                    len(layer.gate_scale_block),
                    len(layer.norm_block),
                    values["offset_block_bytes"],
                    values["down_records_bytes"],
                    _crc(offset_payload),
                    _CODEC,
                )
                handle.write(layer_header)
                handle.write(layer.model_block)
                handle.write(layer.gate_stream_block)
                handle.write(layer.up_stream_block)
                handle.write(layer.gate_scale_block)
                handle.write(layer.up_scale_block)
                handle.write(layer.norm_block)
                handle.write(offset_block)
                for record in layer.record_blocks:
                    handle.write(record)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    if destination.stat().st_size != cursor:
        raise AssertionError("serialized entropy-Q3 artifact has an unexpected size")
    return destination


def _parse_model(
    block: bytes,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    unpacked = _MODEL_HEADER.unpack_from(block)
    magic, version, scale_bits, alphabet = unpacked[:4]
    values = tuple(int(value) for value in unpacked[4:25])
    checksum = unpacked[25]
    if (
        magic != _MODEL_MAGIC
        or version != _VERSION
        or scale_bits != _SCALE_BITS
        or alphabet != _ALPHABET
    ):
        raise ValueError("entropy-Q3 model header is invalid")
    raw = np.asarray(values, dtype="<u2").tobytes()
    if _crc(raw) != checksum or any(value <= 0 for value in values):
        raise ValueError("entropy-Q3 model checksum/frequencies are invalid")
    result = (values[0:7], values[7:14], values[14:21])
    if any(sum(table) != 1 << _SCALE_BITS for table in result):
        raise ValueError("entropy-Q3 model frequencies have an invalid total")
    if any(block[_MODEL_HEADER.size :]):
        raise ValueError("entropy-Q3 model padding is non-zero")
    return result


def _parse_stream(
    block: bytes,
    projection: int,
    symbol_count: int,
    frequencies: tuple[int, ...],
) -> bytes:
    magic, version, kind, encoded_bytes, symbol_crc, encoded_crc = (
        _STREAM_HEADER.unpack_from(block)
    )
    if magic != _STREAM_MAGIC or version != _VERSION or kind != projection:
        raise ValueError("entropy-Q3 stream header is invalid")
    start = _STREAM_HEADER_BYTES
    stop = start + encoded_bytes
    if stop > len(block):
        raise ValueError("entropy-Q3 stream length exceeds its block")
    encoded = bytes(block[start:stop])
    if _crc(encoded) != encoded_crc:
        raise ValueError("entropy-Q3 encoded stream checksum is invalid")
    if any(block[_STREAM_HEADER.size : _STREAM_HEADER_BYTES]):
        raise ValueError("entropy-Q3 stream reserved padding is non-zero")
    symbols = rans_decode(encoded, symbol_count, frequencies)
    if np.any(symbols >= _ALPHABET) or _crc(symbols.tobytes()) != symbol_crc:
        raise ValueError("entropy-Q3 decoded stream checksum/code is invalid")
    if any(block[stop:]):
        raise ValueError("entropy-Q3 stream padding is non-zero")
    return encoded


def _parse_vector(block: bytes, kind: int, width: int) -> NDArray[np.float16]:
    magic, version, actual_kind, count, checksum = _VECTOR_HEADER.unpack_from(block)
    if (
        magic != _VECTOR_MAGIC
        or version != _VERSION
        or actual_kind != kind
        or count != width
    ):
        raise ValueError("entropy-Q3 vector header is invalid")
    start = _VECTOR_HEADER_BYTES
    stop = start + width * 2
    raw = block[start:stop]
    if _crc(raw) != checksum:
        raise ValueError("entropy-Q3 vector checksum is invalid")
    values = np.frombuffer(raw, dtype="<f2").copy()
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("entropy-Q3 vector contains invalid FP16 values")
    if kind in {_GATE, _UP} and np.any(values <= 0):
        raise ValueError("entropy-Q3 scale vector must be positive")
    if any(block[stop:]):
        raise ValueError("entropy-Q3 vector padding is non-zero")
    return values


def load_entropy_q3_artifact(path: str | Path) -> LoadedEntropyQ3Artifact:
    """Strictly reload, checksum, entropy-decode, and validate the artifact."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read entropy-Q3 artifact {source}") from exc
    if len(payload) < _HEADER_BYTES:
        raise ValueError("entropy-Q3 artifact is shorter than its header")
    unpacked = _HEADER.unpack_from(payload)
    (
        magic,
        version,
        layer_count,
        hidden_size,
        source_intermediate_size,
        cache_line_bytes,
        directory_entry_bytes,
        directory_block_bytes,
        directory_crc,
        codec,
        scale_bits,
    ) = unpacked
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("entropy-Q3 artifact magic/version mismatch")
    for value, name in (
        (layer_count, "layer_count"),
        (hidden_size, "hidden_size"),
        (source_intermediate_size, "source_intermediate_size"),
        (cache_line_bytes, "cache_line_bytes"),
    ):
        _positive_integer(value, name)
    if (
        cache_line_bytes != 64
        or directory_entry_bytes != _DIRECTORY_ENTRY.size
        or codec != _CODEC
        or scale_bits != _SCALE_BITS
    ):
        raise ValueError("entropy-Q3 artifact codec/alignment metadata is invalid")
    header_block_bytes = _align(_HEADER_BYTES, cache_line_bytes)
    expected_directory_block = _align(
        layer_count * _DIRECTORY_ENTRY.size, cache_line_bytes
    )
    if directory_block_bytes != expected_directory_block:
        raise ValueError("entropy-Q3 directory block size is invalid")
    if len(payload) < header_block_bytes + directory_block_bytes:
        raise ValueError("entropy-Q3 artifact is shorter than its directory")
    if any(payload[_HEADER.size : header_block_bytes]):
        raise ValueError("entropy-Q3 global header padding is non-zero")
    directory_start = header_block_bytes
    directory_payload_end = directory_start + layer_count * _DIRECTORY_ENTRY.size
    directory_end = directory_start + directory_block_bytes
    directory_payload = payload[directory_start:directory_payload_end]
    if (
        _global_integrity_crc(
            magic,
            version,
            layer_count,
            hidden_size,
            source_intermediate_size,
            cache_line_bytes,
            directory_entry_bytes,
            directory_block_bytes,
            codec,
            scale_bits,
            directory_payload,
        )
        != directory_crc
    ):
        raise ValueError("entropy-Q3 global/directory integrity checksum is invalid")
    if any(payload[directory_payload_end:directory_end]):
        raise ValueError("entropy-Q3 directory padding is non-zero")
    entries = [
        _DIRECTORY_ENTRY.unpack_from(
            payload, directory_start + index * _DIRECTORY_ENTRY.size
        )
        for index in range(layer_count)
    ]
    loaded_layers = []
    expected_layer_offset = directory_end
    for index, entry in enumerate(entries):
        layer_index, width, layer_offset, layer_block_bytes, down_bytes, entry_codec = (
            entry
        )
        if (
            layer_index != index
            or width <= 0
            or width > source_intermediate_size
            or layer_offset != expected_layer_offset
            or layer_offset % cache_line_bytes
            or layer_block_bytes <= 0
            or entry_codec != _CODEC
            or layer_offset + layer_block_bytes > len(payload)
        ):
            raise ValueError("entropy-Q3 directory entry is invalid")
        header = _LAYER_HEADER.unpack_from(payload, layer_offset)
        (
            layer_magic,
            layer_version,
            declared_index,
            declared_hidden,
            declared_width,
            gate_encoded_bytes,
            gate_stream_block_bytes,
            up_encoded_bytes,
            up_stream_block_bytes,
            scale_block_bytes,
            norm_block_bytes,
            offset_block_bytes,
            down_records_bytes,
            offset_crc,
            layer_codec,
        ) = header
        if (
            layer_magic != _LAYER_MAGIC
            or layer_version != _VERSION
            or declared_index != index
            or declared_hidden != hidden_size
            or declared_width != width
            or layer_codec != _CODEC
            or gate_stream_block_bytes % cache_line_bytes
            or up_stream_block_bytes % cache_line_bytes
            or scale_block_bytes % cache_line_bytes
            or norm_block_bytes % cache_line_bytes
            or offset_block_bytes != _align((width + 1) * 8, cache_line_bytes)
            or down_records_bytes != down_bytes
        ):
            raise ValueError("entropy-Q3 layer header is invalid")
        cursor = layer_offset + _LAYER_HEADER_BYTES
        model_block = payload[cursor : cursor + _MODEL_BLOCK_BYTES]
        frequencies = _parse_model(model_block)
        cursor += _MODEL_BLOCK_BYTES
        gate_block = payload[cursor : cursor + gate_stream_block_bytes]
        if _STREAM_HEADER.unpack_from(gate_block)[3] != gate_encoded_bytes:
            raise ValueError("entropy-Q3 gate encoded length is inconsistent")
        gate_stream = _parse_stream(
            gate_block, _GATE, width * hidden_size, frequencies[0]
        )
        cursor += gate_stream_block_bytes
        up_block = payload[cursor : cursor + up_stream_block_bytes]
        if _STREAM_HEADER.unpack_from(up_block)[3] != up_encoded_bytes:
            raise ValueError("entropy-Q3 up encoded length is inconsistent")
        up_stream = _parse_stream(up_block, _UP, width * hidden_size, frequencies[1])
        cursor += up_stream_block_bytes
        gate_scale_block = payload[cursor : cursor + scale_block_bytes]
        gate_scales = _parse_vector(gate_scale_block, _GATE, width)
        cursor += scale_block_bytes
        up_scale_block = payload[cursor : cursor + scale_block_bytes]
        up_scales = _parse_vector(up_scale_block, _UP, width)
        cursor += scale_block_bytes
        norm_payload = payload[cursor : cursor + norm_block_bytes]
        down_norms = _parse_vector(norm_payload, _DOWN_NORM, width)
        cursor += norm_block_bytes
        offset_start = cursor
        offset_payload_bytes = (width + 1) * 8
        raw_offsets = payload[offset_start : offset_start + offset_payload_bytes]
        if _crc(raw_offsets) != offset_crc:
            raise ValueError("entropy-Q3 down-offset checksum is invalid")
        offsets = np.frombuffer(raw_offsets, dtype="<u8").astype(np.int64)
        if any(
            payload[
                offset_start + offset_payload_bytes : offset_start + offset_block_bytes
            ]
        ):
            raise ValueError("entropy-Q3 down-offset padding is non-zero")
        cursor += offset_block_bytes
        if (
            offsets[0] != cursor
            or offsets[-1] != cursor + down_records_bytes
            or np.any(np.diff(offsets) <= 0)
            or np.any(offsets % cache_line_bytes)
        ):
            raise ValueError("entropy-Q3 down-record offsets are invalid")
        records = []
        source_ids = []
        for record_index in range(width):
            start = int(offsets[record_index])
            stop = int(offsets[record_index + 1])
            record_block = payload[start:stop]
            if len(record_block) % cache_line_bytes:
                raise ValueError("entropy-Q3 down record is not cache-line aligned")
            record_header = _RECORD_HEADER.unpack_from(record_block)
            (
                record_magic,
                record_version,
                record_layer,
                declared_record,
                source_id,
                columns,
                record_codec,
                encoded_bytes,
                symbol_crc,
                record_crc,
            ) = record_header
            if (
                record_magic != _RECORD_MAGIC
                or record_version != _VERSION
                or record_layer != index
                or declared_record != record_index
                or source_id >= source_intermediate_size
                or columns != hidden_size
                or record_codec != _CODEC
            ):
                raise ValueError("entropy-Q3 down-record header is invalid")
            scale = np.frombuffer(
                record_block, dtype="<f2", count=1, offset=_RECORD_HEADER_BYTES
            ).copy()[0]
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("entropy-Q3 down-record scale is invalid")
            encoded_start = _RECORD_HEADER_BYTES + 2
            encoded_stop = encoded_start + encoded_bytes
            if encoded_stop > len(record_block):
                raise ValueError("entropy-Q3 down-record encoded length is invalid")
            encoded = bytes(record_block[encoded_start:encoded_stop])
            integrity_payload = (
                struct.pack(
                    "<HHIIHHII",
                    record_version,
                    record_layer,
                    declared_record,
                    source_id,
                    columns,
                    record_codec,
                    encoded_bytes,
                    symbol_crc,
                )
                + record_block[_RECORD_HEADER_BYTES : _RECORD_HEADER_BYTES + 2]
                + encoded
            )
            if _crc(integrity_payload) != record_crc:
                raise ValueError("entropy-Q3 down-record integrity checksum is invalid")
            symbols = rans_decode(encoded, hidden_size, frequencies[2])
            if np.any(symbols >= _ALPHABET) or _crc(symbols.tobytes()) != symbol_crc:
                raise ValueError(
                    "entropy-Q3 down-record symbol checksum/code is invalid"
                )
            decoded = _decode_symbols(
                symbols.reshape(1, hidden_size),
                np.asarray([scale], dtype=np.float16),
            )[0]
            expected_norm = _row_norms(decoded.reshape(1, hidden_size))[0]
            if expected_norm.view(np.uint16) != down_norms[record_index].view(
                np.uint16
            ):
                raise ValueError("entropy-Q3 cached down norm is inconsistent")
            if any(record_block[encoded_stop:]):
                raise ValueError("entropy-Q3 down-record padding is non-zero")
            source_ids.append(source_id)
            records.append(
                EncodedEntropyQ3Record(
                    encoded,
                    scale,
                    source_id,
                    start,
                    len(record_block),
                    symbol_crc,
                )
            )
        if len(set(source_ids)) != width:
            raise ValueError("entropy-Q3 source record IDs are not unique")
        cursor = int(offsets[-1])
        if cursor != layer_offset + layer_block_bytes:
            raise ValueError("entropy-Q3 layer block length is inconsistent")
        loaded_layers.append(
            LoadedEntropyQ3Layer(
                frequencies,
                gate_stream,
                up_stream,
                gate_scales,
                up_scales,
                down_norms,
                tuple(records),
                hidden_size,
                width,
                source_intermediate_size,
                layer_offset,
                layer_block_bytes,
                gate_stream_block_bytes,
                up_stream_block_bytes,
                scale_block_bytes,
                norm_block_bytes,
                offset_block_bytes,
            )
        )
        expected_layer_offset = cursor
    if expected_layer_offset != len(payload):
        raise ValueError("entropy-Q3 artifact has trailing or missing bytes")
    return LoadedEntropyQ3Artifact(
        tuple(loaded_layers),
        hidden_size,
        source_intermediate_size,
        cache_line_bytes,
        header_block_bytes,
        directory_block_bytes,
        len(payload),
    )


def _decode_key_rows(
    layer: LoadedEntropyQ3Layer,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    count = layer.width * layer.hidden_size
    gate_symbols = rans_decode(layer.gate_stream, count, layer.frequencies[0]).reshape(
        layer.width, layer.hidden_size
    )
    up_symbols = rans_decode(layer.up_stream, count, layer.frequencies[1]).reshape(
        layer.width, layer.hidden_size
    )
    return (
        _decode_symbols(gate_symbols, layer.gate_scales),
        _decode_symbols(up_symbols, layer.up_scales),
    )


def _decode_down_record(
    layer: LoadedEntropyQ3Layer, record_index: int
) -> NDArray[np.float32]:
    record = layer.down_records[record_index]
    symbols = rans_decode(
        record.encoded, layer.hidden_size, layer.frequencies[2]
    ).reshape(1, layer.hidden_size)
    return _decode_symbols(symbols, np.asarray([record.scale], dtype=np.float16))[0]


def decode_entropy_q3_artifact(
    artifact: LoadedEntropyQ3Artifact,
) -> tuple[dict[str, NDArray[Any]], ...]:
    """Reference-decode all projections from the validated entropy payload."""

    result = []
    for layer in artifact.layers:
        gate, up = _decode_key_rows(layer)
        down_t = np.stack(
            [_decode_down_record(layer, index) for index in range(layer.width)]
        )
        result.append(
            {
                "gate": gate,
                "up": up,
                "down": np.ascontiguousarray(down_t.T),
                "source_ids": np.asarray(
                    [record.source_id for record in layer.down_records],
                    dtype=np.uint32,
                ),
                "down_norms": layer.down_norms.copy(),
            }
        )
    return tuple(result)


def _silu(values: NDArray[np.float32]) -> NDArray[np.float32]:
    sigmoid = np.empty_like(values)
    positive = values >= 0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    sigmoid[~positive] = exponent / (1.0 + exponent)
    return values * sigmoid


def entropy_q3_forward(
    artifact: LoadedEntropyQ3Artifact,
    layer_index: int,
    hidden: ArrayLike,
    *,
    top_k: int | None = None,
    selected_ids: ArrayLike | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Execute full sequential keys and only the selected down records."""

    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
        or layer_index >= len(artifact.layers)
    ):
        raise ValueError("layer_index is outside the entropy-Q3 artifact")
    layer = artifact.layers[layer_index]
    states = _matrix(hidden, "hidden")
    if states.shape[1] != layer.hidden_size:
        raise ValueError("hidden width does not match the entropy-Q3 artifact")
    gate, up = _decode_key_rows(layer)
    activations = _silu(states @ gate.T) * (states @ up.T)
    if selected_ids is None:
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
            or top_k > layer.width
        ):
            raise ValueError("top_k must lie within the entropy-Q3 layer width")
        scores = np.abs(activations) * layer.down_norms.astype(np.float32)[None, :]
        selected = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    else:
        selected = np.asarray(selected_ids)
        if (
            selected.ndim != 2
            or selected.shape[0] != len(states)
            or selected.shape[1] == 0
            or not np.issubdtype(selected.dtype, np.integer)
            or np.any(selected < 0)
            or np.any(selected >= layer.width)
        ):
            raise ValueError("selected_ids must contain valid IDs for every state")
        selected = selected.astype(np.int64, copy=False)
        if any(len(np.unique(row)) != len(row) for row in selected):
            raise ValueError("selected_ids must be unique within each state")
    output = np.zeros((len(states), layer.hidden_size), dtype=np.float32)
    cache: dict[int, NDArray[np.float32]] = {}
    for row in range(len(states)):
        for record_id in selected[row]:
            index = int(record_id)
            if index not in cache:
                cache[index] = _decode_down_record(layer, index)
            down = cache[index]
            output[row] += activations[row, index] * down
    return np.ascontiguousarray(output), np.ascontiguousarray(selected)


def entropy_q3_dynamic_traffic(
    artifact: LoadedEntropyQ3Artifact,
    selected_records: int | list[list[int]] | tuple[tuple[int, ...], ...],
    *,
    router_bytes_per_token: int = 0,
) -> dict[str, Any]:
    """Return exact serialized blocks read for K down rows plus router bytes.

    Integer ``selected_records`` uses the largest K serialized down blocks in
    each layer, giving a deterministic conservative bound.  Explicit per-layer
    ID lists return the exact blocks for that selection.
    """

    if (
        isinstance(router_bytes_per_token, bool)
        or not isinstance(router_bytes_per_token, int)
        or router_bytes_per_token < 0
    ):
        raise ValueError("router_bytes_per_token must be a non-negative integer")
    selected_by_layer: list[list[int]] = []
    policy: str
    if isinstance(selected_records, int) and not isinstance(selected_records, bool):
        if selected_records <= 0 or any(
            selected_records > layer.width for layer in artifact.layers
        ):
            raise ValueError("selected_records must lie within every layer width")
        policy = "largest_k_record_blocks_per_layer"
        for layer in artifact.layers:
            order = sorted(
                range(layer.width),
                key=lambda index: (-layer.down_records[index].block_bytes, index),
            )
            selected_by_layer.append(order[:selected_records])
    else:
        if len(selected_records) != len(artifact.layers):  # type: ignore[arg-type]
            raise ValueError("explicit selection must provide IDs for every layer")
        policy = "explicit_record_ids"
        for layer, ids in zip(artifact.layers, selected_records, strict=True):  # type: ignore[arg-type]
            values = list(ids)
            if (
                not values
                or len(set(values)) != len(values)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    or value >= layer.width
                    for value in values
                )
            ):
                raise ValueError("explicit selection contains invalid record IDs")
            selected_by_layer.append(values)
    global_metadata_bytes = artifact.header_block_bytes + artifact.directory_block_bytes
    sequential_key_bytes = 0
    layer_metadata_bytes = 0
    selected_down_bytes = 0
    selected_counts = []
    for layer, ids in zip(artifact.layers, selected_by_layer, strict=True):
        sequential_key_bytes += (
            layer.gate_stream_block_bytes
            + layer.up_stream_block_bytes
            + 2 * layer.scale_block_bytes
        )
        layer_metadata_bytes += (
            _LAYER_HEADER_BYTES
            + _MODEL_BLOCK_BYTES
            + layer.norm_block_bytes
            + layer.offset_block_bytes
        )
        selected_down_bytes += sum(
            layer.down_records[index].block_bytes for index in ids
        )
        selected_counts.append(len(ids))
    router_cache_aligned_bytes = (
        _align(router_bytes_per_token, artifact.cache_line_bytes)
        if router_bytes_per_token
        else 0
    )
    total_cold_bytes = (
        global_metadata_bytes
        + sequential_key_bytes
        + layer_metadata_bytes
        + selected_down_bytes
        + router_cache_aligned_bytes
    )
    dense_q4_source_mlp_bytes = (
        len(artifact.layers)
        * 3
        * artifact.hidden_size
        * artifact.source_intermediate_size
        + 1
    ) // 2
    fraction = total_cold_bytes / dense_q4_source_mlp_bytes
    return {
        "layout": "sequential_entropy_q3_gate_up_plus_aligned_down_records_v1",
        "selection_policy": policy,
        "selected_records_by_layer": selected_counts,
        "global_metadata_bytes": global_metadata_bytes,
        "sequential_gate_up_bytes": sequential_key_bytes,
        "layer_model_norm_offset_bytes": layer_metadata_bytes,
        "selected_down_record_bytes": selected_down_bytes,
        "router_bytes_per_token": router_bytes_per_token,
        "router_cache_aligned_bytes": router_cache_aligned_bytes,
        "total_cold_bytes": total_cold_bytes,
        "dense_q4_source_mlp_bytes": dense_q4_source_mlp_bytes,
        "fraction_of_dense_q4": fraction,
        "remaining_fraction_before_45_percent": 0.45 - fraction,
        "passes_45_percent_traffic_gate": fraction <= 0.45,
        "serialized_artifact_bytes": artifact.serialized_artifact_bytes,
    }


__all__ = [
    "EncodedEntropyQ3Record",
    "EntropyQ3LayerWeights",
    "LoadedEntropyQ3Artifact",
    "LoadedEntropyQ3Layer",
    "decode_entropy_q3_artifact",
    "entropy_q3_dynamic_traffic",
    "entropy_q3_forward",
    "load_entropy_q3_artifact",
    "save_entropy_q3_artifact",
]
