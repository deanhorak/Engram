"""Minimal deterministic static-model byte rANS coder.

The implementation is intentionally small and auditable.  Models use a fixed
power-of-two frequency total, streams carry a four-byte terminal state, and
decoding rejects truncation, trailing bytes, invalid states, and malformed
frequency tables.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

_RANS_BYTE_L = 1 << 23


def normalized_frequencies(
    symbols: ArrayLike,
    *,
    alphabet_size: int = 7,
    scale_bits: int = 12,
) -> tuple[int, ...]:
    """Fit deterministic positive integer frequencies summing to ``2**bits``."""

    if (
        isinstance(alphabet_size, bool)
        or not isinstance(alphabet_size, int)
        or alphabet_size <= 0
        or alphabet_size > 256
    ):
        raise ValueError("alphabet_size must be an integer in [1, 256]")
    if (
        isinstance(scale_bits, bool)
        or not isinstance(scale_bits, int)
        or scale_bits <= 0
        or scale_bits > 16
    ):
        raise ValueError("scale_bits must lie in [1, 16]")
    total_frequency = 1 << scale_bits
    if alphabet_size > total_frequency:
        raise ValueError("alphabet_size exceeds the frequency total")
    values = np.asarray(symbols)
    if (
        values.ndim != 1
        or not values.size
        or not np.issubdtype(values.dtype, np.integer)
    ):
        raise ValueError("symbols must be a non-empty integer vector")
    integers = values.astype(np.int64, copy=False)
    if np.any(integers < 0) or np.any(integers >= alphabet_size):
        raise ValueError("symbol is outside the model alphabet")
    counts = np.bincount(integers, minlength=alphabet_size).astype(np.int64)
    total = int(counts.sum())
    products = counts * total_frequency
    frequencies = np.maximum(1, products // total).astype(np.int64)
    remainders = products % total
    difference = total_frequency - int(frequencies.sum())
    if difference > 0:
        order = sorted(range(alphabet_size), key=lambda i: (-remainders[i], i))
        for index in range(difference):
            frequencies[order[index % alphabet_size]] += 1
    elif difference < 0:
        order = sorted(range(alphabet_size), key=lambda i: (remainders[i], -i))
        remaining = -difference
        while remaining:
            changed = False
            for index in order:
                if frequencies[index] > 1:
                    frequencies[index] -= 1
                    remaining -= 1
                    changed = True
                    if not remaining:
                        break
            if not changed:
                raise ValueError("cannot normalize the requested frequency model")
    result = tuple(int(value) for value in frequencies)
    _validate_frequencies(result, scale_bits)
    return result


def _validate_frequencies(
    frequencies: Sequence[int], scale_bits: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if (
        isinstance(scale_bits, bool)
        or not isinstance(scale_bits, int)
        or scale_bits <= 0
        or scale_bits > 16
    ):
        raise ValueError("scale_bits must lie in [1, 16]")
    if not frequencies:
        raise ValueError("frequency table must not be empty")
    normalized = tuple(frequencies)
    if len(normalized) > 256:
        raise ValueError("frequency table cannot exceed 256 symbols")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in normalized
    ):
        raise ValueError("frequency values must be positive integers")
    if sum(normalized) != 1 << scale_bits:
        raise ValueError("frequency values do not sum to the model total")
    starts = []
    cursor = 0
    for frequency in normalized:
        starts.append(cursor)
        cursor += frequency
    return normalized, tuple(starts)


def rans_encode(
    symbols: ArrayLike,
    frequencies: Sequence[int],
    *,
    scale_bits: int = 12,
) -> bytes:
    """Encode one independent symbol stream with a static canonical model."""

    model, starts = _validate_frequencies(frequencies, scale_bits)
    values = np.asarray(symbols)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("symbols must be a one-dimensional integer vector")
    integers = values.astype(np.int64, copy=False)
    if np.any(integers < 0) or np.any(integers >= len(model)):
        raise ValueError("symbol is outside the model alphabet")
    state = _RANS_BYTE_L
    emitted = bytearray()
    for raw_symbol in integers[::-1]:
        symbol = int(raw_symbol)
        frequency = model[symbol]
        maximum = ((_RANS_BYTE_L >> scale_bits) << 8) * frequency
        while state >= maximum:
            emitted.append(state & 0xFF)
            state >>= 8
        state = (
            ((state // frequency) << scale_bits) + (state % frequency) + starts[symbol]
        )
    if state >= _RANS_BYTE_L << 8:
        raise AssertionError("rANS terminal state is not canonically normalized")
    if state > np.iinfo(np.uint32).max:
        raise AssertionError("rANS terminal state exceeds uint32")
    return int(state).to_bytes(4, "little") + bytes(reversed(emitted))


def rans_decode(
    payload: bytes | bytearray | memoryview,
    symbol_count: int,
    frequencies: Sequence[int],
    *,
    scale_bits: int = 12,
) -> NDArray[np.uint8]:
    """Strictly decode one independent static-model rANS stream."""

    model, starts = _validate_frequencies(frequencies, scale_bits)
    if (
        isinstance(symbol_count, bool)
        or not isinstance(symbol_count, int)
        or symbol_count < 0
    ):
        raise ValueError("symbol_count must be a non-negative integer")
    data = memoryview(payload).cast("B")
    if len(data) < 4:
        raise ValueError("rANS payload is shorter than its terminal state")
    state = int.from_bytes(data[:4], "little")
    if state < _RANS_BYTE_L or state >= _RANS_BYTE_L << 8:
        raise ValueError("rANS terminal state is invalid")
    total = 1 << scale_bits
    mask = total - 1
    lookup = np.empty(total, dtype=np.uint8)
    for symbol, (start, frequency) in enumerate(zip(starts, model, strict=True)):
        lookup[start : start + frequency] = symbol
    decoded = np.empty(symbol_count, dtype=np.uint8)
    cursor = 4
    for index in range(symbol_count):
        slot = state & mask
        symbol = int(lookup[slot])
        decoded[index] = symbol
        state = model[symbol] * (state >> scale_bits) + slot - starts[symbol]
        while state < _RANS_BYTE_L:
            if cursor >= len(data):
                raise ValueError("truncated rANS payload")
            state = (state << 8) | int(data[cursor])
            cursor += 1
    if cursor != len(data):
        raise ValueError("rANS payload has trailing bytes")
    if state != _RANS_BYTE_L:
        raise ValueError("rANS payload did not terminate at the canonical state")
    return decoded


__all__ = ["normalized_frequencies", "rans_decode", "rans_encode"]
