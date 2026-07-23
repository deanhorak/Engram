import numpy as np
import pytest

from engram.training.canonical_rans import (
    normalized_frequencies,
    rans_decode,
    rans_encode,
)


def test_canonical_rans_round_trip_is_deterministic():
    rng = np.random.default_rng(4041)
    symbols = rng.choice(
        7, size=4097, p=[0.55, 0.16, 0.1, 0.07, 0.05, 0.04, 0.03]
    ).astype(np.uint8)
    frequencies = normalized_frequencies(symbols)

    first = rans_encode(symbols, frequencies)
    second = rans_encode(symbols, frequencies)

    assert first == second
    assert sum(frequencies) == 4096
    assert all(value > 0 for value in frequencies)
    np.testing.assert_array_equal(
        rans_decode(first, len(symbols), frequencies), symbols
    )


def test_canonical_rans_rejects_truncation_and_trailing_bytes():
    symbols = np.asarray([0, 1, 0, 6, 2, 0, 3] * 20, dtype=np.uint8)
    frequencies = normalized_frequencies(symbols)
    payload = rans_encode(symbols, frequencies)

    with pytest.raises(ValueError, match="truncated|terminate"):
        rans_decode(payload[:-1], len(symbols), frequencies)
    with pytest.raises(ValueError, match="trailing"):
        rans_decode(payload + b"\0", len(symbols), frequencies)


def test_canonical_rans_rejects_noncanonical_upper_terminal_state():
    # L<<8 would decode and eventually terminate under the recurrence, but a
    # canonical byte-rANS stream must renormalize that high byte into payload.
    noncanonical = (1 << 31).to_bytes(4, "little")
    with pytest.raises(ValueError, match="terminal state"):
        rans_decode(noncanonical, 8, (1, 1), scale_bits=1)


def test_canonical_rans_rejects_an_alphabet_that_cannot_fit_uint8():
    with pytest.raises(ValueError, match="256"):
        normalized_frequencies(np.asarray([0, 256], dtype=np.int64), alphabet_size=257)
