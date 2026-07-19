"""Normalized recurrent linear attention reference implementation.

The persistent state is a feature/value outer-product accumulator and a
feature normalizer.  It grows with model dimensions, not sequence length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class RecurrentAttentionError(ValueError):
    """Raised for invalid recurrent-attention inputs or configuration."""


def positive_feature_map(values: ArrayLike) -> np.ndarray:
    """Apply the positive ELU+1 map without overflow on positive inputs."""

    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        raise RecurrentAttentionError("feature-map input must be numeric")
    dtype = array.dtype if np.issubdtype(array.dtype, np.floating) else np.dtype(np.float64)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        dtype = np.dtype(np.float64)
    array = np.asarray(array, dtype=dtype)
    if not np.all(np.isfinite(array)):
        raise RecurrentAttentionError("feature-map input must contain only finite values")
    result = np.empty_like(array)
    nonnegative = array >= 0.0
    result[nonnegative] = array[nonnegative] + 1.0
    result[~nonnegative] = np.exp(array[~nonnegative])
    # exp can underflow for a finite, very negative input.  Preserve the
    # feature map's strictly-positive contract in that case.
    np.maximum(result, np.finfo(dtype).tiny, out=result)
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RecurrentAttentionError(f"{name} must be a positive integer")
    return value


def _decay(value: Any) -> float:
    if isinstance(value, bool):
        raise RecurrentAttentionError("decay must be a finite scalar in [0, 1]")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RecurrentAttentionError("decay must be a finite scalar in [0, 1]") from error
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise RecurrentAttentionError("decay must be a finite scalar in [0, 1]")
    return result


def _floating_dtype(dtype: Any) -> np.dtype[Any]:
    try:
        result = np.dtype(dtype)
    except TypeError as error:
        raise RecurrentAttentionError("dtype must be float32 or float64") from error
    if result not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise RecurrentAttentionError("dtype must be float32 or float64")
    return result


@dataclass
class RecurrentAttentionState:
    """Persistent sequence state; no token history is retained."""

    numerator: np.ndarray
    normalizer: np.ndarray
    steps: int = 0

    @property
    def element_count(self) -> int:
        return int(self.numerator.size + self.normalizer.size)

    @property
    def nbytes(self) -> int:
        return int(self.numerator.nbytes + self.normalizer.nbytes)


class NormalizedRecurrentAttention:
    """Stateful normalized linear attention with exponential decay."""

    def __init__(
        self,
        key_dimension: int,
        value_dimension: int,
        *,
        decay: float = 1.0,
        epsilon: float = 1e-6,
        dtype: Any = np.float64,
    ) -> None:
        self.key_dimension = _positive_integer(key_dimension, name="key_dimension")
        self.value_dimension = _positive_integer(value_dimension, name="value_dimension")
        self.decay = _decay(decay)
        try:
            self.epsilon = float(epsilon)
        except (TypeError, ValueError) as error:
            raise RecurrentAttentionError("epsilon must be finite and positive") from error
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise RecurrentAttentionError("epsilon must be finite and positive")
        self.dtype = _floating_dtype(dtype)
        self.state = RecurrentAttentionState(
            numerator=np.zeros(
                (self.key_dimension, self.value_dimension), dtype=self.dtype
            ),
            normalizer=np.zeros(self.key_dimension, dtype=self.dtype),
        )

    def reset(self) -> None:
        """Clear the current sequence while retaining allocated state buffers."""

        self.state.numerator.fill(0.0)
        self.state.normalizer.fill(0.0)
        self.state.steps = 0

    @property
    def state_metrics(self) -> dict[str, int]:
        return {
            "steps": self.state.steps,
            "elements": self.state.element_count,
            "bytes": self.state.nbytes,
            "key_features": self.key_dimension,
            "value_width": self.value_dimension,
        }

    def _vector(self, values: ArrayLike, dimension: int, *, name: str) -> np.ndarray:
        try:
            vector = np.asarray(values, dtype=self.dtype)
        except (TypeError, ValueError) as error:
            raise RecurrentAttentionError(f"{name} must be numeric") from error
        if vector.shape != (dimension,):
            raise RecurrentAttentionError(
                f"{name} must have shape ({dimension},), got {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise RecurrentAttentionError(f"{name} must contain only finite values")
        return vector

    def step(
        self,
        query: ArrayLike,
        key: ArrayLike,
        value: ArrayLike,
        *,
        decay: float | None = None,
    ) -> np.ndarray:
        """Consume one token and return its normalized recurrent attention read."""

        query_vector = self._vector(query, self.key_dimension, name="query")
        key_vector = self._vector(key, self.key_dimension, name="key")
        value_vector = self._vector(value, self.value_dimension, name="value")
        step_decay = self.decay if decay is None else _decay(decay)
        query_features = positive_feature_map(query_vector)
        key_features = positive_feature_map(key_vector)

        self.state.numerator *= step_decay
        self.state.normalizer *= step_decay
        self.state.numerator += np.outer(key_features, value_vector)
        self.state.normalizer += key_features
        denominator = float(query_features @ self.state.normalizer)
        stabilized_denominator = max(denominator, self.epsilon)
        output = (query_features @ self.state.numerator) / stabilized_denominator
        self.state.steps += 1
        return np.asarray(output, dtype=self.dtype)

    def sequence(
        self,
        queries: ArrayLike,
        keys: ArrayLike,
        values: ArrayLike,
        *,
        decays: ArrayLike | None = None,
        reset: bool = True,
    ) -> np.ndarray:
        """Consume a sequence, optionally continuing the existing state."""

        try:
            query_matrix = np.asarray(queries, dtype=self.dtype)
            key_matrix = np.asarray(keys, dtype=self.dtype)
            value_matrix = np.asarray(values, dtype=self.dtype)
        except (TypeError, ValueError) as error:
            raise RecurrentAttentionError("queries, keys, and values must be numeric") from error
        expected_query_tail = (self.key_dimension,)
        expected_value_tail = (self.value_dimension,)
        if query_matrix.ndim != 2 or query_matrix.shape[1:] != expected_query_tail:
            raise RecurrentAttentionError(
                f"queries must have shape [T, {self.key_dimension}]"
            )
        if key_matrix.shape != query_matrix.shape:
            raise RecurrentAttentionError("keys must have the same shape as queries")
        if value_matrix.shape != (query_matrix.shape[0], *expected_value_tail):
            raise RecurrentAttentionError(
                f"values must have shape [T, {self.value_dimension}]"
            )
        if not (
            np.all(np.isfinite(query_matrix))
            and np.all(np.isfinite(key_matrix))
            and np.all(np.isfinite(value_matrix))
        ):
            raise RecurrentAttentionError(
                "queries, keys, and values must contain only finite values"
            )
        length = query_matrix.shape[0]
        if decays is None:
            decay_values = np.full(length, self.decay, dtype=np.float64)
        else:
            try:
                decay_values = np.asarray(decays, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise RecurrentAttentionError("decays must be numeric") from error
            if decay_values.shape != (length,):
                raise RecurrentAttentionError(f"decays must have shape ({length},)")
            # Validate all decay values before mutating state.
            try:
                decay_values = np.asarray([_decay(item) for item in decay_values])
            except RecurrentAttentionError as error:
                raise RecurrentAttentionError(
                    "decays must contain only finite values in [0, 1]"
                ) from error
        if reset:
            self.reset()
        outputs = np.empty((length, self.value_dimension), dtype=self.dtype)
        for index in range(length):
            outputs[index] = self.step(
                query_matrix[index],
                key_matrix[index],
                value_matrix[index],
                decay=float(decay_values[index]),
            )
        return outputs


def normalized_recurrent_attention(
    queries: ArrayLike,
    keys: ArrayLike,
    values: ArrayLike,
    *,
    decay: float = 1.0,
    epsilon: float = 1e-6,
    dtype: Any = np.float64,
) -> np.ndarray:
    """Stateless convenience API for a complete sequence."""

    query_matrix = np.asarray(queries)
    value_matrix = np.asarray(values)
    if query_matrix.ndim != 2 or value_matrix.ndim != 2:
        raise RecurrentAttentionError("queries and values must be rank-2 matrices")
    attention = NormalizedRecurrentAttention(
        query_matrix.shape[1],
        value_matrix.shape[1],
        decay=decay,
        epsilon=epsilon,
        dtype=dtype,
    )
    return attention.sequence(query_matrix, keys, value_matrix)


__all__ = [
    "NormalizedRecurrentAttention",
    "RecurrentAttentionError",
    "RecurrentAttentionState",
    "normalized_recurrent_attention",
    "positive_feature_map",
]
