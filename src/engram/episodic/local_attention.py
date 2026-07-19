"""Exact causal attention over a bounded local window.

Arrays use ``[..., sequence, feature]`` layout.  The leading dimensions may be
empty for single-head attention or may contain batch/head dimensions.  The
``window`` includes the current token, so a window of one returns the current
value exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike


def _scale(head_dim: int, scale: float | None) -> float:
    result = 1.0 / math.sqrt(head_dim) if scale is None else float(scale)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("scale must be finite and positive")
    return result


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the final dimension."""

    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=-1, keepdims=True)


def _attention_dtype(*arrays: np.ndarray) -> np.dtype:
    dtype = np.result_type(*(array.dtype for array in arrays), np.float32)
    if not np.issubdtype(dtype, np.floating):
        return np.dtype(np.float64)
    return np.dtype(dtype)


def _validate_sequence_inputs(
    query: ArrayLike, key: ArrayLike, value: ArrayLike, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window <= 0:
        raise ValueError("window must be a positive integer")
    q = np.asarray(query)
    k = np.asarray(key)
    v = np.asarray(value)
    if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
        raise ValueError("query, key, and value must have shape [..., sequence, feature]")
    if q.shape[:-1] != k.shape[:-1] or q.shape[-1] != k.shape[-1]:
        raise ValueError(f"query/key shapes are incompatible: {q.shape} versus {k.shape}")
    if v.shape[:-2] != q.shape[:-2] or v.shape[-2] != q.shape[-2]:
        raise ValueError(f"value shape is incompatible with query: {v.shape} versus {q.shape}")
    if q.shape[-1] == 0 or v.shape[-1] == 0:
        raise ValueError("feature dimensions must be non-empty")
    if not all(np.all(np.isfinite(array)) for array in (q, k, v)):
        raise ValueError("query, key, and value must contain only finite values")
    dtype = _attention_dtype(q, k, v)
    return q.astype(dtype, copy=False), k.astype(dtype, copy=False), v.astype(dtype, copy=False)


def causal_local_attention(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    *,
    window: int,
    scale: float | None = None,
) -> np.ndarray:
    """Compute exact causal self-attention within ``window`` tokens.

    This reference implementation never forms a full sequence-by-sequence
    score matrix.  Each query is compared only with its visible local keys.
    """

    q, k, v = _validate_sequence_inputs(query, key, value, window)
    attention_scale = _scale(q.shape[-1], scale)
    output = np.empty((*q.shape[:-1], v.shape[-1]), dtype=_attention_dtype(q, k, v))
    for position in range(q.shape[-2]):
        start = max(0, position - window + 1)
        local_key = k[..., start : position + 1, :]
        local_value = v[..., start : position + 1, :]
        scores = np.einsum("...d,...td->...t", q[..., position, :], local_key)
        weights = _softmax(scores * attention_scale)
        output[..., position, :] = np.einsum("...t,...tv->...v", weights, local_value)
    return output


@dataclass
class LocalAttentionCache:
    """Bounded key/value cache for incremental causal local attention."""

    window: int
    scale: float | None = None
    _keys: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _values: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _key_shape: tuple[int, ...] | None = field(default=None, init=False, repr=False)
    _value_shape: tuple[int, ...] | None = field(default=None, init=False, repr=False)
    _tokens_seen: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.window, bool)
            or not isinstance(self.window, (int, np.integer))
            or self.window <= 0
        ):
            raise ValueError("window must be a positive integer")

    @property
    def cache_length(self) -> int:
        return len(self._keys)

    @property
    def tokens_seen(self) -> int:
        return self._tokens_seen

    def reset(self) -> None:
        self._keys.clear()
        self._values.clear()
        self._key_shape = None
        self._value_shape = None
        self._tokens_seen = 0

    def step(self, query: ArrayLike, key: ArrayLike, value: ArrayLike) -> np.ndarray:
        """Append one token and return its local-attention output.

        Token arrays use ``[..., feature]`` layout.  Leading dimensions may
        represent heads or batch and heads, but must remain fixed for the life
        of the cache.
        """

        q = np.asarray(query)
        k = np.asarray(key)
        v = np.asarray(value)
        if q.ndim < 1 or k.ndim < 1 or v.ndim < 1:
            raise ValueError("token query, key, and value must have shape [..., feature]")
        if q.shape != k.shape:
            raise ValueError(f"query/key token shapes differ: {q.shape} versus {k.shape}")
        if q.shape[:-1] != v.shape[:-1]:
            raise ValueError(f"value token shape is incompatible with query: {v.shape} versus {q.shape}")
        if q.shape[-1] == 0 or v.shape[-1] == 0:
            raise ValueError("feature dimensions must be non-empty")
        if not all(np.all(np.isfinite(array)) for array in (q, k, v)):
            raise ValueError("query, key, and value must contain only finite values")
        if self._key_shape is not None and k.shape != self._key_shape:
            raise ValueError(f"key shape changed from {self._key_shape} to {k.shape}")
        if self._value_shape is not None and v.shape != self._value_shape:
            raise ValueError(f"value shape changed from {self._value_shape} to {v.shape}")

        dtype = _attention_dtype(q, k, v)
        q = q.astype(dtype, copy=False)
        k = k.astype(dtype, copy=True)
        v = v.astype(dtype, copy=True)
        attention_scale = _scale(q.shape[-1], self.scale)
        self._key_shape = k.shape
        self._value_shape = v.shape
        self._keys.append(k)
        self._values.append(v)
        if len(self._keys) > self.window:
            del self._keys[0]
            del self._values[0]
        self._tokens_seen += 1

        local_key = np.stack(self._keys, axis=-2)
        local_value = np.stack(self._values, axis=-2)
        scores = np.einsum("...d,...td->...t", q, local_key)
        weights = _softmax(scores * attention_scale)
        return np.einsum("...t,...tv->...v", weights, local_value)
