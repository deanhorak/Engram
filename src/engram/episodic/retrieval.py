from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


ValueStorage = Literal["int8", "float32"]


@dataclass(frozen=True)
class RetrievalRecall:
    hits: int
    retrieved_count: int
    relevant_count: int
    recall: float
    precision: float


@dataclass(frozen=True)
class RetrievalMemoryMetrics:
    """Payload bytes owned by the store, excluding Python object overhead."""

    recent_active_bytes: int
    older_active_bytes: int
    older_allocated_bytes: int
    total_active_bytes: int
    total_allocated_bytes: int


@dataclass(frozen=True)
class RetrievalReadMetrics:
    """Logical bytes examined by one retrieval operation."""

    candidate_search_bytes: int
    exact_rerank_bytes: int
    selected_value_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class RetrievalResult:
    positions: NDArray[np.int64]
    scores: NDArray[np.float32]
    values: NDArray[np.float32]
    candidate_positions: NDArray[np.int64]
    candidate_scores: NDArray[np.float32]
    reads: RetrievalReadMetrics

    def recall_against(self, relevant_positions: Iterable[int]) -> RetrievalRecall:
        return retrieval_recall(self.positions, relevant_positions)


def _positions(values: Iterable[int], name: str) -> set[int]:
    array = np.asarray(list(values))
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    result = {int(value) for value in array}
    if any(value < 0 for value in result):
        raise ValueError(f"{name} must not contain negative positions")
    return result


def retrieval_recall(
    retrieved_positions: Iterable[int], relevant_positions: Iterable[int]
) -> RetrievalRecall:
    """Compute set recall and precision against relevant teacher positions."""

    retrieved = _positions(retrieved_positions, "retrieved_positions")
    relevant = _positions(relevant_positions, "relevant_positions")
    hits = len(retrieved & relevant)
    recall = hits / len(relevant) if relevant else 1.0
    if retrieved:
        precision = hits / len(retrieved)
    else:
        precision = 1.0 if not relevant else 0.0
    return RetrievalRecall(
        hits=hits,
        retrieved_count=len(retrieved),
        relevant_count=len(relevant),
        recall=recall,
        precision=precision,
    )


def _quantize_vector(vector: NDArray[np.float32]) -> tuple[NDArray[np.int8], np.float32]:
    """Symmetric per-vector int8 quantization with deterministic rounding."""

    maximum = float(np.max(np.abs(vector))) if vector.size else 0.0
    scale = np.float32(maximum / 127.0 if maximum > 0.0 else 1.0)
    codes = np.clip(np.rint(vector / scale), -127, 127).astype(np.int8)
    return codes, scale


class OlderContextRetrievalStore:
    """Bounded recent and quantized older-context storage.

    New entries remain as float32 in the recent window. Once evicted from that
    window, their keys are scalar-quantized into a fixed-capacity ring buffer.
    Older values can use the same 255-entry signed scalar codebook (``int8``)
    with a per-vector scale, or remain in a float32 fallback buffer.

    Retrieval searches only older context; exact local attention is expected to
    consume :attr:`recent_keys` and :attr:`recent_values` separately. Quantized
    cosine similarity supplies candidates. Candidates are then decoded and
    reranked by their exact dot product in the stored, quantized representation.
    """

    def __init__(
        self,
        key_width: int,
        value_width: int,
        *,
        recent_window: int,
        capacity: int,
        candidate_count: int = 16,
        value_storage: ValueStorage = "int8",
    ) -> None:
        for value, name in (
            (key_width, "key_width"),
            (value_width, "value_width"),
            (recent_window, "recent_window"),
            (capacity, "capacity"),
            (candidate_count, "candidate_count"),
        ):
            if not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if key_width <= 0 or value_width <= 0:
            raise ValueError("key_width and value_width must be positive")
        if recent_window < 0 or capacity < 0:
            raise ValueError("recent_window and capacity must be non-negative")
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if value_storage not in {"int8", "float32"}:
            raise ValueError("value_storage must be 'int8' or 'float32'")

        self.key_width = int(key_width)
        self.value_width = int(value_width)
        self.recent_window = int(recent_window)
        self.capacity = int(capacity)
        self.candidate_count = int(candidate_count)
        self.value_storage: ValueStorage = value_storage

        self._recent: list[tuple[int, NDArray[np.float32], NDArray[np.float32]]] = []
        self._key_codes = np.zeros((capacity, key_width), dtype=np.int8)
        self._key_scales = np.ones(capacity, dtype=np.float32)
        self._older_positions = np.zeros(capacity, dtype=np.int64)
        if value_storage == "int8":
            self._value_codes = np.zeros((capacity, value_width), dtype=np.int8)
            self._value_scales = np.ones(capacity, dtype=np.float32)
            self._float_values = None
        else:
            self._value_codes = None
            self._value_scales = None
            self._float_values = np.zeros((capacity, value_width), dtype=np.float32)
        self._start = 0
        self._size = 0
        self._last_position = -1

    @property
    def recent_count(self) -> int:
        return len(self._recent)

    @property
    def older_count(self) -> int:
        return self._size

    @property
    def recent_positions(self) -> NDArray[np.int64]:
        return np.asarray([entry[0] for entry in self._recent], dtype=np.int64)

    @property
    def recent_keys(self) -> NDArray[np.float32]:
        if not self._recent:
            return np.empty((0, self.key_width), dtype=np.float32)
        return np.stack([entry[1] for entry in self._recent])

    @property
    def recent_values(self) -> NDArray[np.float32]:
        if not self._recent:
            return np.empty((0, self.value_width), dtype=np.float32)
        return np.stack([entry[2] for entry in self._recent])

    def _logical_slots(self) -> NDArray[np.int64]:
        return (self._start + np.arange(self._size, dtype=np.int64)) % max(self.capacity, 1)

    @property
    def older_positions(self) -> NDArray[np.int64]:
        return self._older_positions[self._logical_slots()].copy()

    def _vector(self, value: ArrayLike, width: int, name: str) -> NDArray[np.float32]:
        result = np.asarray(value, dtype=np.float32)
        if result.ndim != 1 or result.shape[0] != width:
            raise ValueError(f"{name} must have shape [{width}], got {result.shape}")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain only finite values")
        return result.copy()

    def append(self, key: ArrayLike, value: ArrayLike, *, position: int | None = None) -> int:
        """Append a token state and return its assigned absolute position."""

        key_array = self._vector(key, self.key_width, "key")
        value_array = self._vector(value, self.value_width, "value")
        assigned = self._last_position + 1 if position is None else position
        if not isinstance(assigned, (int, np.integer)) or assigned < 0:
            raise ValueError("position must be a non-negative integer")
        assigned = int(assigned)
        if assigned <= self._last_position:
            raise ValueError("positions must be strictly increasing")
        self._last_position = assigned
        self._recent.append((assigned, key_array, value_array))
        if len(self._recent) > self.recent_window:
            old_position, old_key, old_value = self._recent.pop(0)
            self._append_older(old_position, old_key, old_value)
        return assigned

    def _append_older(
        self,
        position: int,
        key: NDArray[np.float32],
        value: NDArray[np.float32],
    ) -> None:
        if self.capacity == 0:
            return
        if self._size < self.capacity:
            slot = (self._start + self._size) % self.capacity
            self._size += 1
        else:
            slot = self._start
            self._start = (self._start + 1) % self.capacity

        key_codes, key_scale = _quantize_vector(key)
        self._key_codes[slot] = key_codes
        self._key_scales[slot] = key_scale
        self._older_positions[slot] = position
        if self.value_storage == "int8":
            value_codes, value_scale = _quantize_vector(value)
            assert self._value_codes is not None and self._value_scales is not None
            self._value_codes[slot] = value_codes
            self._value_scales[slot] = value_scale
        else:
            assert self._float_values is not None
            self._float_values[slot] = value

    def _decode_values(self, slots: NDArray[np.int64]) -> NDArray[np.float32]:
        if self.value_storage == "int8":
            assert self._value_codes is not None and self._value_scales is not None
            return self._value_codes[slots].astype(np.float32) * self._value_scales[slots, None]
        assert self._float_values is not None
        return self._float_values[slots].copy()

    def retrieve(
        self,
        query: ArrayLike,
        *,
        top_k: int,
        candidate_count: int | None = None,
    ) -> RetrievalResult:
        """Search quantized older keys and exactly rerank decoded candidates."""

        query_array = self._vector(query, self.key_width, "query")
        if not isinstance(top_k, (int, np.integer)) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        requested_candidates = self.candidate_count if candidate_count is None else candidate_count
        if not isinstance(requested_candidates, (int, np.integer)) or requested_candidates <= 0:
            raise ValueError("candidate_count must be a positive integer")
        if top_k > requested_candidates:
            raise ValueError("top_k must not exceed candidate_count")

        if self._size == 0:
            empty_positions = np.empty(0, dtype=np.int64)
            empty_scores = np.empty(0, dtype=np.float32)
            reads = RetrievalReadMetrics(0, 0, 0, 0)
            return RetrievalResult(
                positions=empty_positions,
                scores=empty_scores,
                values=np.empty((0, self.value_width), dtype=np.float32),
                candidate_positions=empty_positions.copy(),
                candidate_scores=empty_scores.copy(),
                reads=reads,
            )

        slots = self._logical_slots()
        codes = self._key_codes[slots]
        scales = self._key_scales[slots]
        raw_scores = (codes.astype(np.float32) @ query_array) * scales
        query_norm = float(np.linalg.norm(query_array))
        key_norms = np.linalg.norm(codes.astype(np.float32), axis=1) * scales
        denominator = key_norms * query_norm
        cosine_scores = np.divide(
            raw_scores,
            denominator,
            out=np.zeros_like(raw_scores),
            where=denominator > 0.0,
        )
        count = min(int(requested_candidates), self._size)
        candidate_order = np.argsort(-cosine_scores, kind="stable")[:count]
        candidate_slots = slots[candidate_order]

        decoded_keys = (
            self._key_codes[candidate_slots].astype(np.float32)
            * self._key_scales[candidate_slots, None]
        )
        exact_scores = decoded_keys @ query_array
        selected_count = min(int(top_k), count)
        reranked = np.argsort(-exact_scores, kind="stable")[:selected_count]
        selected_slots = candidate_slots[reranked]

        key_record_bytes = self.key_width * np.dtype(np.int8).itemsize + np.dtype(np.float32).itemsize
        value_record_bytes = self._value_record_bytes()
        search_bytes = self._size * key_record_bytes
        rerank_bytes = count * key_record_bytes
        selected_value_bytes = selected_count * value_record_bytes
        reads = RetrievalReadMetrics(
            candidate_search_bytes=search_bytes,
            exact_rerank_bytes=rerank_bytes,
            selected_value_bytes=selected_value_bytes,
            total_bytes=search_bytes + rerank_bytes + selected_value_bytes,
        )
        return RetrievalResult(
            positions=self._older_positions[selected_slots].copy(),
            scores=exact_scores[reranked].astype(np.float32, copy=False),
            values=self._decode_values(selected_slots),
            candidate_positions=self._older_positions[candidate_slots].copy(),
            candidate_scores=cosine_scores[candidate_order].astype(np.float32, copy=False),
            reads=reads,
        )

    def _value_record_bytes(self) -> int:
        if self.value_storage == "int8":
            return self.value_width * np.dtype(np.int8).itemsize + np.dtype(np.float32).itemsize
        return self.value_width * np.dtype(np.float32).itemsize

    def memory_metrics(self) -> RetrievalMemoryMetrics:
        """Return active and preallocated payload storage bytes."""

        position_bytes = np.dtype(np.int64).itemsize
        key_record_bytes = self.key_width * np.dtype(np.int8).itemsize + np.dtype(np.float32).itemsize
        older_record_bytes = key_record_bytes + self._value_record_bytes() + position_bytes
        recent_record_bytes = (
            (self.key_width + self.value_width) * np.dtype(np.float32).itemsize + position_bytes
        )
        recent_active = len(self._recent) * recent_record_bytes
        older_active = self._size * older_record_bytes
        older_allocated = self.capacity * older_record_bytes
        return RetrievalMemoryMetrics(
            recent_active_bytes=recent_active,
            older_active_bytes=older_active,
            older_allocated_bytes=older_allocated,
            total_active_bytes=recent_active + older_active,
            total_allocated_bytes=recent_active + older_allocated,
        )
