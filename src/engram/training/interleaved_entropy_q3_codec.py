"""Strict record-interleaved signed-Q3 entropy artifact.

Every intermediate channel is one independently framed and cache-line-aligned
record containing its gate, up, and down rows.  A nonlinear router can select
channel IDs first, after which the runtime reads only those complete records.
Each projection has a deterministic per-layer static rANS model; all tables,
offsets, record metadata, encoded payloads, decoded symbols, and padding are
validated on reload.
"""

from __future__ import annotations

import os
import struct
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
from engram.training.entropy_q3_codec import (
    EntropyQ3LayerWeights,
    _ALPHABET,
    _DIRECTORY_ENTRY,
    _HEADER,
    _HEADER_BYTES,
    _LAYER_HEADER,
    _LAYER_HEADER_BYTES,
    _MODEL_BLOCK_BYTES,
    _SCALE_BITS,
    _align,
    _crc,
    _decode_symbols,
    _global_integrity_crc,
    _matrix,
    _model_block,
    _parse_model,
    _positive_integer,
    _quantize_rows,
    _silu,
    _source_ids,
)

_MAGIC = b"ENGEQ3I1"
_LAYER_MAGIC = b"ENGEQ3IL"
_RECORD_MAGIC = b"EQ3I"
_VERSION = 1
_CODEC = 2
_RECORD_HEADER_BYTES = 64
_RECORD_HEADER = struct.Struct("<4sHHIIHH8I3H6x")


@dataclass(frozen=True)
class EncodedInterleavedEntropyQ3Record:
    gate_encoded: bytes
    up_encoded: bytes
    down_encoded: bytes
    gate_scale: np.float16
    up_scale: np.float16
    down_scale: np.float16
    source_id: int
    offset: int
    block_bytes: int
    symbol_crc32: tuple[int, int, int]


@dataclass(frozen=True)
class LoadedInterleavedEntropyQ3Layer:
    frequencies: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    records: tuple[EncodedInterleavedEntropyQ3Record, ...]
    hidden_size: int
    width: int
    source_intermediate_size: int
    layer_offset: int
    layer_block_bytes: int
    offset_block_bytes: int
    record_bytes: int


@dataclass(frozen=True)
class LoadedInterleavedEntropyQ3Artifact:
    layers: tuple[LoadedInterleavedEntropyQ3Layer, ...]
    hidden_size: int
    source_intermediate_size: int
    cache_line_bytes: int
    header_block_bytes: int
    directory_block_bytes: int
    serialized_artifact_bytes: int


@dataclass(frozen=True)
class _PreparedInterleavedLayer:
    frequencies: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    model_block: bytes
    record_blocks: tuple[bytes, ...]
    hidden_size: int
    width: int


def _scale_bits(value: np.float16) -> int:
    return int(np.asarray([value], dtype=np.float16).view(np.uint16)[0])


def _scale_from_bits(value: int) -> np.float16:
    return np.asarray([value], dtype=np.uint16).view(np.float16)[0]


def _record_integrity_payload(
    *,
    layer_index: int,
    record_index: int,
    source_id: int,
    columns: int,
    lengths: tuple[int, int, int],
    symbol_crcs: tuple[int, int, int],
    scale_bits: tuple[int, int, int],
    encoded: bytes,
) -> bytes:
    metadata = struct.pack(
        "<HHIIHH6I3H",
        _VERSION,
        layer_index,
        record_index,
        source_id,
        columns,
        _CODEC,
        *lengths,
        *symbol_crcs,
        *scale_bits,
    )
    return metadata + encoded


def _interleaved_record_block(
    gate_symbols: NDArray[np.uint8],
    up_symbols: NDArray[np.uint8],
    down_symbols: NDArray[np.uint8],
    scales: tuple[np.float16, np.float16, np.float16],
    frequencies: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    *,
    layer_index: int,
    record_index: int,
    source_id: int,
    cache_line_bytes: int,
) -> bytes:
    projections = (gate_symbols, up_symbols, down_symbols)
    columns = len(gate_symbols)
    if any(values.shape != (columns,) for values in projections):
        raise ValueError("interleaved entropy-Q3 record rows must share one width")
    encoded_parts = tuple(
        rans_encode(symbols, table)
        for symbols, table in zip(projections, frequencies, strict=True)
    )
    lengths = tuple(len(payload) for payload in encoded_parts)
    symbol_crcs = tuple(_crc(symbols.tobytes()) for symbols in projections)
    bits = tuple(_scale_bits(value) for value in scales)
    encoded = b"".join(encoded_parts)
    encoded_crc = _crc(encoded)
    record_crc = _crc(
        _record_integrity_payload(
            layer_index=layer_index,
            record_index=record_index,
            source_id=source_id,
            columns=columns,
            lengths=lengths,
            symbol_crcs=symbol_crcs,
            scale_bits=bits,
            encoded=encoded,
        )
    )
    block = bytearray(_align(_RECORD_HEADER_BYTES + len(encoded), cache_line_bytes))
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
        *lengths,
        *symbol_crcs,
        encoded_crc,
        record_crc,
        *bits,
    )
    block[_RECORD_HEADER_BYTES : _RECORD_HEADER_BYTES + len(encoded)] = encoded
    return bytes(block)


def _prepare_interleaved_layer(
    layer: EntropyQ3LayerWeights,
    layer_index: int,
    source_intermediate_size: int,
    cache_line_bytes: int,
) -> _PreparedInterleavedLayer:
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
        normalized_frequencies(values.symbols.reshape(-1)) for values in quantized
    )
    typed_frequencies = (frequencies[0], frequencies[1], frequencies[2])
    records = tuple(
        _interleaved_record_block(
            quantized[0].symbols[index],
            quantized[1].symbols[index],
            quantized[2].symbols[index],
            (
                quantized[0].scales[index],
                quantized[1].scales[index],
                quantized[2].scales[index],
            ),
            typed_frequencies,
            layer_index=layer_index,
            record_index=index,
            source_id=int(source_ids[index]),
            cache_line_bytes=cache_line_bytes,
        )
        for index in range(width)
    )
    return _PreparedInterleavedLayer(
        typed_frequencies,
        _model_block(typed_frequencies),
        records,
        hidden_size,
        width,
    )


def save_interleaved_entropy_q3_artifact(
    path: str | Path,
    layers: list[EntropyQ3LayerWeights] | tuple[EntropyQ3LayerWeights, ...],
    *,
    source_intermediate_size: int,
    cache_line_bytes: int = 64,
) -> Path:
    """Atomically write independently aligned gate/up/down records."""

    _positive_integer(source_intermediate_size, "source_intermediate_size")
    _positive_integer(cache_line_bytes, "cache_line_bytes")
    if cache_line_bytes != 64:
        raise ValueError("cache_line_bytes must be exactly 64")
    logical_layers = tuple(layers)
    if not logical_layers:
        raise ValueError("layers must not be empty")
    prepared = tuple(
        _prepare_interleaved_layer(
            layer, index, source_intermediate_size, cache_line_bytes
        )
        for index, layer in enumerate(logical_layers)
    )
    hidden_size = prepared[0].hidden_size
    if any(layer.hidden_size != hidden_size for layer in prepared):
        raise ValueError("all interleaved entropy-Q3 layers must share one hidden size")
    header_block_bytes = _align(_HEADER_BYTES, cache_line_bytes)
    directory_block_bytes = _align(
        len(prepared) * _DIRECTORY_ENTRY.size, cache_line_bytes
    )
    metadata: list[dict[str, int]] = []
    cursor = header_block_bytes + directory_block_bytes
    for layer in prepared:
        offset_block_bytes = _align((layer.width + 1) * 8, cache_line_bytes)
        record_bytes = sum(len(record) for record in layer.record_blocks)
        layer_block_bytes = (
            _LAYER_HEADER_BYTES + _MODEL_BLOCK_BYTES + offset_block_bytes + record_bytes
        )
        metadata.append(
            {
                "offset": cursor,
                "offset_block_bytes": offset_block_bytes,
                "record_bytes": record_bytes,
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
            values["record_bytes"],
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
                    + _MODEL_BLOCK_BYTES
                    + values["offset_block_bytes"]
                )
                offsets = [records_start]
                for record in layer.record_blocks:
                    offsets.append(offsets[-1] + len(record))
                offset_payload = np.asarray(offsets, dtype="<u8").tobytes()
                offset_block = bytearray(values["offset_block_bytes"])
                offset_block[: len(offset_payload)] = offset_payload
                layer_header = bytearray(_LAYER_HEADER_BYTES)
                _LAYER_HEADER.pack_into(
                    layer_header,
                    0,
                    _LAYER_MAGIC,
                    _VERSION,
                    index,
                    hidden_size,
                    layer.width,
                    _MODEL_BLOCK_BYTES,
                    values["offset_block_bytes"],
                    values["record_bytes"],
                    _crc(offset_payload),
                    _CODEC,
                    _SCALE_BITS,
                    _ALPHABET,
                    _RECORD_HEADER_BYTES,
                    0,
                    0,
                )
                handle.write(layer_header)
                handle.write(layer.model_block)
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
        raise AssertionError("serialized interleaved entropy-Q3 size is inconsistent")
    return destination


def _decode_record_payload(
    block: bytes,
    frequencies: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    *,
    expected_layer: int,
    expected_record: int,
    hidden_size: int,
    source_intermediate_size: int,
    offset: int,
) -> tuple[
    EncodedInterleavedEntropyQ3Record,
    tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]],
]:
    if len(block) < _RECORD_HEADER_BYTES:
        raise ValueError("interleaved entropy-Q3 record is shorter than its header")
    unpacked = _RECORD_HEADER.unpack_from(block)
    (
        magic,
        version,
        layer_index,
        record_index,
        source_id,
        columns,
        codec,
        gate_bytes,
        up_bytes,
        down_bytes,
        gate_symbol_crc,
        up_symbol_crc,
        down_symbol_crc,
        encoded_crc,
        record_crc,
        gate_scale_bits,
        up_scale_bits,
        down_scale_bits,
    ) = unpacked
    if (
        magic != _RECORD_MAGIC
        or version != _VERSION
        or layer_index != expected_layer
        or record_index != expected_record
        or source_id >= source_intermediate_size
        or columns != hidden_size
        or codec != _CODEC
        or min(gate_bytes, up_bytes, down_bytes) < 4
    ):
        raise ValueError("interleaved entropy-Q3 record header is invalid")
    lengths = (gate_bytes, up_bytes, down_bytes)
    symbol_crcs = (gate_symbol_crc, up_symbol_crc, down_symbol_crc)
    bits = (gate_scale_bits, up_scale_bits, down_scale_bits)
    scales = tuple(_scale_from_bits(value) for value in bits)
    if any(not np.isfinite(value) or value <= 0 for value in scales):
        raise ValueError("interleaved entropy-Q3 record scale is invalid")
    encoded_stop = _RECORD_HEADER_BYTES + sum(lengths)
    if encoded_stop > len(block):
        raise ValueError("interleaved entropy-Q3 encoded lengths exceed the record")
    encoded = bytes(block[_RECORD_HEADER_BYTES:encoded_stop])
    if _crc(encoded) != encoded_crc:
        raise ValueError("interleaved entropy-Q3 encoded checksum is invalid")
    if (
        _crc(
            _record_integrity_payload(
                layer_index=layer_index,
                record_index=record_index,
                source_id=source_id,
                columns=columns,
                lengths=lengths,
                symbol_crcs=symbol_crcs,
                scale_bits=bits,
                encoded=encoded,
            )
        )
        != record_crc
    ):
        raise ValueError("interleaved entropy-Q3 record checksum is invalid")
    if any(block[encoded_stop:]):
        raise ValueError("interleaved entropy-Q3 record padding is non-zero")
    gate_stop = gate_bytes
    up_stop = gate_stop + up_bytes
    encoded_parts = (
        encoded[:gate_stop],
        encoded[gate_stop:up_stop],
        encoded[up_stop:],
    )
    symbols = tuple(
        rans_decode(payload, hidden_size, table)
        for payload, table in zip(encoded_parts, frequencies, strict=True)
    )
    if any(
        np.any(values >= _ALPHABET) or _crc(values.tobytes()) != checksum
        for values, checksum in zip(symbols, symbol_crcs, strict=True)
    ):
        raise ValueError(
            "interleaved entropy-Q3 decoded symbol checksum/code is invalid"
        )
    record = EncodedInterleavedEntropyQ3Record(
        encoded_parts[0],
        encoded_parts[1],
        encoded_parts[2],
        scales[0],
        scales[1],
        scales[2],
        source_id,
        offset,
        len(block),
        symbol_crcs,
    )
    return record, (symbols[0], symbols[1], symbols[2])


def load_interleaved_entropy_q3_artifact(
    path: str | Path,
) -> LoadedInterleavedEntropyQ3Artifact:
    """Strictly reload and entropy-decode every independently framed record."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"cannot read interleaved entropy-Q3 artifact {source}"
        ) from exc
    if len(payload) < _HEADER_BYTES:
        raise ValueError("interleaved entropy-Q3 artifact is shorter than its header")
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
    ) = _HEADER.unpack_from(payload)
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("interleaved entropy-Q3 artifact magic/version mismatch")
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
        raise ValueError("interleaved entropy-Q3 codec/alignment metadata is invalid")
    header_block_bytes = _align(_HEADER_BYTES, cache_line_bytes)
    expected_directory_block = _align(
        layer_count * _DIRECTORY_ENTRY.size, cache_line_bytes
    )
    if directory_block_bytes != expected_directory_block:
        raise ValueError("interleaved entropy-Q3 directory block size is invalid")
    directory_start = header_block_bytes
    directory_payload_end = directory_start + layer_count * _DIRECTORY_ENTRY.size
    directory_end = directory_start + directory_block_bytes
    if len(payload) < directory_end:
        raise ValueError(
            "interleaved entropy-Q3 artifact is shorter than its directory"
        )
    if any(payload[_HEADER.size : header_block_bytes]):
        raise ValueError("interleaved entropy-Q3 global-header padding is non-zero")
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
        raise ValueError(
            "interleaved entropy-Q3 global/directory integrity checksum is invalid"
        )
    if any(payload[directory_payload_end:directory_end]):
        raise ValueError("interleaved entropy-Q3 directory padding is non-zero")
    entries = tuple(
        _DIRECTORY_ENTRY.unpack_from(
            payload, directory_start + index * _DIRECTORY_ENTRY.size
        )
        for index in range(layer_count)
    )
    loaded_layers = []
    expected_layer_offset = directory_end
    for index, entry in enumerate(entries):
        (
            layer_index,
            width,
            layer_offset,
            layer_block_bytes,
            entry_record_bytes,
            entry_codec,
        ) = entry
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
            raise ValueError("interleaved entropy-Q3 directory entry is invalid")
        if layer_offset + _LAYER_HEADER_BYTES > len(payload):
            raise ValueError("interleaved entropy-Q3 layer header is truncated")
        (
            layer_magic,
            layer_version,
            declared_index,
            declared_hidden,
            declared_width,
            model_block_bytes,
            offset_block_bytes,
            record_bytes,
            offset_crc,
            layer_codec,
            layer_scale_bits,
            alphabet,
            record_header_bytes,
            reserved_zero_a,
            reserved_zero_b,
        ) = _LAYER_HEADER.unpack_from(payload, layer_offset)
        if (
            layer_magic != _LAYER_MAGIC
            or layer_version != _VERSION
            or declared_index != index
            or declared_hidden != hidden_size
            or declared_width != width
            or model_block_bytes != _MODEL_BLOCK_BYTES
            or offset_block_bytes != _align((width + 1) * 8, cache_line_bytes)
            or record_bytes != entry_record_bytes
            or layer_codec != _CODEC
            or layer_scale_bits != _SCALE_BITS
            or alphabet != _ALPHABET
            or record_header_bytes != _RECORD_HEADER_BYTES
            or reserved_zero_a
            or reserved_zero_b
        ):
            raise ValueError("interleaved entropy-Q3 layer header is invalid")
        cursor = layer_offset + _LAYER_HEADER_BYTES
        model_stop = cursor + model_block_bytes
        if model_stop > len(payload):
            raise ValueError("interleaved entropy-Q3 model block is truncated")
        frequencies = _parse_model(payload[cursor:model_stop])
        cursor = model_stop
        offset_payload_bytes = (width + 1) * 8
        offset_stop = cursor + offset_block_bytes
        if offset_stop > len(payload):
            raise ValueError("interleaved entropy-Q3 offset block is truncated")
        raw_offsets = payload[cursor : cursor + offset_payload_bytes]
        if _crc(raw_offsets) != offset_crc:
            raise ValueError("interleaved entropy-Q3 offset checksum is invalid")
        if any(payload[cursor + offset_payload_bytes : offset_stop]):
            raise ValueError("interleaved entropy-Q3 offset padding is non-zero")
        offsets = np.frombuffer(raw_offsets, dtype="<u8").astype(np.int64)
        cursor = offset_stop
        if (
            offsets[0] != cursor
            or offsets[-1] != cursor + record_bytes
            or np.any(np.diff(offsets) <= 0)
            or np.any(offsets % cache_line_bytes)
        ):
            raise ValueError("interleaved entropy-Q3 record offsets are invalid")
        records = []
        source_ids = []
        for record_index in range(width):
            start = int(offsets[record_index])
            stop = int(offsets[record_index + 1])
            if stop > len(payload) or (stop - start) % cache_line_bytes:
                raise ValueError("interleaved entropy-Q3 record block is invalid")
            record, _ = _decode_record_payload(
                payload[start:stop],
                frequencies,
                expected_layer=index,
                expected_record=record_index,
                hidden_size=hidden_size,
                source_intermediate_size=source_intermediate_size,
                offset=start,
            )
            records.append(record)
            source_ids.append(record.source_id)
        if len(set(source_ids)) != width:
            raise ValueError("interleaved entropy-Q3 source record IDs are not unique")
        cursor = int(offsets[-1])
        if cursor != layer_offset + layer_block_bytes:
            raise ValueError("interleaved entropy-Q3 layer length is inconsistent")
        loaded_layers.append(
            LoadedInterleavedEntropyQ3Layer(
                frequencies,
                tuple(records),
                hidden_size,
                width,
                source_intermediate_size,
                layer_offset,
                layer_block_bytes,
                offset_block_bytes,
                record_bytes,
            )
        )
        expected_layer_offset = cursor
    if expected_layer_offset != len(payload):
        raise ValueError(
            "interleaved entropy-Q3 artifact has trailing or missing bytes"
        )
    return LoadedInterleavedEntropyQ3Artifact(
        tuple(loaded_layers),
        hidden_size,
        source_intermediate_size,
        cache_line_bytes,
        header_block_bytes,
        directory_block_bytes,
        len(payload),
    )


def _decode_interleaved_record(
    layer: LoadedInterleavedEntropyQ3Layer, record_index: int
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    if record_index < 0 or record_index >= layer.width:
        raise ValueError("record_index is outside the interleaved entropy-Q3 layer")
    record = layer.records[record_index]
    encoded = (record.gate_encoded, record.up_encoded, record.down_encoded)
    scales = (record.gate_scale, record.up_scale, record.down_scale)
    decoded = []
    for payload, frequency, scale in zip(
        encoded, layer.frequencies, scales, strict=True
    ):
        symbols = rans_decode(payload, layer.hidden_size, frequency).reshape(
            1, layer.hidden_size
        )
        decoded.append(
            _decode_symbols(symbols, np.asarray([scale], dtype=np.float16))[0]
        )
    return decoded[0], decoded[1], decoded[2]


def decode_interleaved_entropy_q3_artifact(
    artifact: LoadedInterleavedEntropyQ3Artifact,
) -> tuple[dict[str, NDArray[Any]], ...]:
    """Reference-decode every gate/up/down record in a validated artifact."""

    result = []
    for layer in artifact.layers:
        decoded = tuple(
            _decode_interleaved_record(layer, index) for index in range(layer.width)
        )
        result.append(
            {
                "gate": np.stack([record[0] for record in decoded]),
                "up": np.stack([record[1] for record in decoded]),
                "down": np.ascontiguousarray(
                    np.stack([record[2] for record in decoded]).T
                ),
                "source_ids": np.asarray(
                    [record.source_id for record in layer.records], dtype=np.uint32
                ),
            }
        )
    return tuple(result)


def interleaved_entropy_q3_forward(
    artifact: LoadedInterleavedEntropyQ3Artifact,
    layer_index: int,
    hidden: ArrayLike,
    selected_ids: ArrayLike,
) -> NDArray[np.float32]:
    """Execute exact reranking only inside externally selected record IDs."""

    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
        or layer_index >= len(artifact.layers)
    ):
        raise ValueError("layer_index is outside the interleaved entropy-Q3 artifact")
    layer = artifact.layers[layer_index]
    states = _matrix(hidden, "hidden")
    if states.shape[1] != layer.hidden_size:
        raise ValueError("hidden width does not match the interleaved artifact")
    selected = np.asarray(selected_ids)
    if (
        selected.ndim != 2
        or selected.shape[0] != len(states)
        or not np.issubdtype(selected.dtype, np.integer)
        or np.any(selected < 0)
        or np.any(selected >= layer.width)
    ):
        raise ValueError("selected_ids must contain valid IDs for every state")
    selected = selected.astype(np.int64, copy=False)
    if any(len(np.unique(row)) != len(row) for row in selected):
        raise ValueError("selected_ids must be unique within each state")
    output = np.zeros((len(states), layer.hidden_size), dtype=np.float32)
    cache: dict[
        int, tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]
    ] = {}
    for row_index, row_ids in enumerate(selected):
        state = states[row_index]
        for raw_record_id in row_ids:
            record_id = int(raw_record_id)
            if record_id not in cache:
                cache[record_id] = _decode_interleaved_record(layer, record_id)
            gate, up, down = cache[record_id]
            gate_value = np.asarray([state @ gate], dtype=np.float32)
            activation = _silu(gate_value)[0] * np.float32(state @ up)
            output[row_index] += activation * down
    return np.ascontiguousarray(output)


def interleaved_entropy_q3_dynamic_traffic(
    artifact: LoadedInterleavedEntropyQ3Artifact,
    selected_records: int | list[list[int]] | tuple[tuple[int, ...], ...],
    *,
    router_bytes_per_token: int = 0,
) -> dict[str, Any]:
    """Return exact cold bytes for selected aligned triples plus caller router."""

    if (
        isinstance(router_bytes_per_token, bool)
        or not isinstance(router_bytes_per_token, int)
        or router_bytes_per_token < 0
    ):
        raise ValueError("router_bytes_per_token must be a non-negative integer")
    selected_by_layer: list[list[int]] = []
    if isinstance(selected_records, int) and not isinstance(selected_records, bool):
        if selected_records <= 0 or any(
            selected_records > layer.width for layer in artifact.layers
        ):
            raise ValueError("selected_records must lie within every layer width")
        policy = "largest_k_record_blocks_per_layer"
        for layer in artifact.layers:
            order = sorted(
                range(layer.width),
                key=lambda index: (-layer.records[index].block_bytes, index),
            )
            selected_by_layer.append(order[:selected_records])
    else:
        try:
            selections = tuple(selected_records)
        except TypeError as exc:
            raise ValueError("selected_records must be K or per-layer IDs") from exc
        if len(selections) != len(artifact.layers):
            raise ValueError("explicit selection must provide IDs for every layer")
        policy = "explicit_record_ids"
        for layer, ids in zip(artifact.layers, selections, strict=True):
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
    layer_metadata_bytes = sum(
        _LAYER_HEADER_BYTES + _MODEL_BLOCK_BYTES + layer.offset_block_bytes
        for layer in artifact.layers
    )
    selected_record_bytes = sum(
        layer.records[index].block_bytes
        for layer, ids in zip(artifact.layers, selected_by_layer, strict=True)
        for index in ids
    )
    router_cache_aligned_bytes = (
        _align(router_bytes_per_token, artifact.cache_line_bytes)
        if router_bytes_per_token
        else 0
    )
    total_cold_bytes = (
        global_metadata_bytes
        + layer_metadata_bytes
        + selected_record_bytes
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
        "layout": "aligned_interleaved_gate_up_down_entropy_q3_records_v1",
        "selection_policy": policy,
        "selected_records_by_layer": [len(ids) for ids in selected_by_layer],
        "global_metadata_bytes": global_metadata_bytes,
        "layer_model_offset_bytes": layer_metadata_bytes,
        "selected_interleaved_record_bytes": selected_record_bytes,
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
    "EncodedInterleavedEntropyQ3Record",
    "LoadedInterleavedEntropyQ3Artifact",
    "LoadedInterleavedEntropyQ3Layer",
    "decode_interleaved_entropy_q3_artifact",
    "interleaved_entropy_q3_dynamic_traffic",
    "interleaved_entropy_q3_forward",
    "load_interleaved_entropy_q3_artifact",
    "save_interleaved_entropy_q3_artifact",
]
