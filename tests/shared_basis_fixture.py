"""Small deterministic ESIB v1 fixture writer used only by tests."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np

from engram.training.canonical_rans import (
    normalized_frequencies,
    rans_encode,
)

_MAGIC = b"ESIBQ344"
_HEADER_BYTES = 256
_LINE_BYTES = 64
_SCALE_BITS = 12
_PREFIX = struct.Struct("<8sHHIIIIIHHHHff")
_DESCRIPTOR = struct.Struct("<QQ")
_DOWN_HEADER = struct.Struct("<HH")
_SECTIONS = (
    "frequency_models",
    "basis_scales",
    "gate_scales",
    "up_scales",
    "down_norms",
    "down_index",
    "basis_payload",
    "gate_payload",
    "up_payload",
    "down_records",
)
_CHECKSUM_OFFSET = _PREFIX.size + len(_SECTIONS) * _DESCRIPTOR.size


def _align(value: int) -> int:
    return (value + _LINE_BYTES - 1) // _LINE_BYTES * _LINE_BYTES


def write_esib_fixture(
    path: Path,
    *,
    layer: int,
    hidden: int,
    width: int,
    rank: int,
    top_k: int,
) -> bytes:
    """Write a valid, non-trained ESIB artifact with deterministic codes."""

    basis_codes = (
        np.arange(rank * hidden, dtype=np.uint32).reshape(rank, hidden) % 7
    ).astype(np.uint8)
    gate_codes = (
        np.arange(width * rank, dtype=np.uint32).reshape(width, rank) % 4
    ).astype(np.uint8)
    up_codes = ((gate_codes.astype(np.uint16) * 3 + 1) % 4).astype(np.uint8)
    down_codes = (
        np.arange(width * hidden, dtype=np.uint32).reshape(width, hidden) % 15
    ).astype(np.uint8)
    basis_model = normalized_frequencies(
        basis_codes.reshape(-1), alphabet_size=7, scale_bits=_SCALE_BITS
    )
    gate_model = normalized_frequencies(
        gate_codes.reshape(-1), alphabet_size=4, scale_bits=_SCALE_BITS
    )
    up_model = normalized_frequencies(
        up_codes.reshape(-1), alphabet_size=4, scale_bits=_SCALE_BITS
    )
    down_model = normalized_frequencies(
        down_codes.reshape(-1), alphabet_size=15, scale_bits=_SCALE_BITS
    )
    basis_payload = rans_encode(
        basis_codes.reshape(-1), basis_model, scale_bits=_SCALE_BITS
    )
    gate_payload = rans_encode(
        gate_codes.reshape(-1), gate_model, scale_bits=_SCALE_BITS
    )
    up_payload = rans_encode(up_codes.reshape(-1), up_model, scale_bits=_SCALE_BITS)
    records = []
    record_bytes = []
    scale_raw = int(np.asarray([np.float16(0.05)], dtype=np.float16).view(np.uint16)[0])
    for row in range(width):
        encoded = rans_encode(down_codes[row], down_model, scale_bits=_SCALE_BITS)
        raw = _DOWN_HEADER.pack(scale_raw, len(encoded)) + encoded
        record = raw + bytes(_align(len(raw)) - len(raw))
        records.append(record)
        record_bytes.append(len(record))
    line_offsets = np.concatenate(
        (
            np.asarray([0], dtype=np.uint16),
            np.cumsum(record_bytes, dtype=np.uint32).astype(np.uint16) // _LINE_BYTES,
        )
    )
    raw_sections = {
        "frequency_models": np.asarray(
            basis_model + gate_model + up_model + down_model, dtype="<u2"
        ).tobytes(),
        "basis_scales": np.full(rank, 0.125, dtype="<f2").tobytes(),
        "gate_scales": np.full(width, 0.2, dtype="<f2").tobytes(),
        "up_scales": np.full(width, 0.15, dtype="<f2").tobytes(),
        "down_norms": np.full(width, 1.0, dtype="<f2").tobytes(),
        "down_index": np.asarray(line_offsets, dtype="<u2").tobytes(),
        "basis_payload": basis_payload,
        "gate_payload": gate_payload,
        "up_payload": up_payload,
        "down_records": b"".join(records),
    }
    descriptors: dict[str, tuple[int, int]] = {}
    cursor = _HEADER_BYTES
    for name in _SECTIONS:
        raw = raw_sections[name]
        descriptors[name] = cursor, len(raw)
        cursor += _align(len(raw))
    blob = bytearray(cursor)
    blob[: _PREFIX.size] = _PREFIX.pack(
        _MAGIC,
        1,
        _HEADER_BYTES,
        layer,
        rank,
        top_k,
        width,
        hidden,
        3,
        2,
        4,
        _SCALE_BITS,
        0.25,
        0.30,
    )
    descriptor_cursor = _PREFIX.size
    for name in _SECTIONS:
        offset, logical_bytes = descriptors[name]
        blob[descriptor_cursor : descriptor_cursor + _DESCRIPTOR.size] = (
            _DESCRIPTOR.pack(offset, logical_bytes)
        )
        descriptor_cursor += _DESCRIPTOR.size
        blob[offset : offset + logical_bytes] = raw_sections[name]
    checksum_blob = bytearray(blob)
    checksum_blob[_CHECKSUM_OFFSET : _CHECKSUM_OFFSET + 32] = bytes(32)
    blob[_CHECKSUM_OFFSET : _CHECKSUM_OFFSET + 32] = hashlib.sha256(
        checksum_blob
    ).digest()
    payload = bytes(blob)
    path.write_bytes(payload)
    return payload
