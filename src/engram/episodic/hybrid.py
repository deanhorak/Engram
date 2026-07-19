from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engram.episodic.recurrent import NormalizedRecurrentAttention, positive_feature_map
from engram.episodic.retrieval import OlderContextRetrievalStore, RetrievalResult


@dataclass(frozen=True)
class HybridEpisodicRead:
    output: np.ndarray
    local_output: np.ndarray
    recurrent_output: np.ndarray
    retrieval_output: np.ndarray
    local_tokens: int
    older_tokens: int
    retrievals: int
    bytes_read: int
    state_bytes: int


class HybridEpisodicMemory:
    """Exact local window plus bounded recurrent and quantized retrieval memory."""

    def __init__(
        self,
        key_width: int,
        value_width: int,
        *,
        local_window: int = 16,
        retrieval_capacity: int = 1024,
        retrieval_candidates: int = 16,
        retrieval_top_k: int = 4,
        decay: float = 0.99,
        older_weight: float = 0.5,
    ) -> None:
        if retrieval_top_k > retrieval_candidates:
            raise ValueError("retrieval_top_k must not exceed retrieval_candidates")
        if not 0.0 <= older_weight <= 1.0:
            raise ValueError("older_weight must be in [0, 1]")
        self.store = OlderContextRetrievalStore(
            key_width,
            value_width,
            recent_window=local_window,
            capacity=retrieval_capacity,
            candidate_count=retrieval_candidates,
        )
        self.recurrent = NormalizedRecurrentAttention(key_width, value_width, decay=decay)
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_candidates = retrieval_candidates
        self.older_weight = older_weight

    def _recurrent_read(self, query: np.ndarray) -> np.ndarray:
        if self.recurrent.state.steps == 0:
            return np.zeros(self.recurrent.value_dimension, dtype=self.recurrent.dtype)
        features = positive_feature_map(np.asarray(query, dtype=self.recurrent.dtype))
        denominator = max(float(features @ self.recurrent.state.normalizer), self.recurrent.epsilon)
        return (features @ self.recurrent.state.numerator) / denominator

    def step(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> HybridEpisodicRead:
        query_array = np.asarray(query, dtype=np.float32)
        evicted_key = evicted_value = None
        if self.store.recent_count == self.store.recent_window and self.store.recent_count:
            evicted_key = self.store.recent_keys[0].copy()
            evicted_value = self.store.recent_values[0].copy()
        self.store.append(key, value)
        if evicted_key is not None and evicted_value is not None:
            recurrent_output = self.recurrent.step(query_array, evicted_key, evicted_value)
        else:
            recurrent_output = self._recurrent_read(query_array)

        local_keys = self.store.recent_keys
        local_values = self.store.recent_values
        scores = local_keys @ query_array / np.sqrt(self.store.key_width)
        weights = np.exp(scores - np.max(scores))
        weights /= np.sum(weights)
        local_output = weights @ local_values
        retrieval: RetrievalResult = self.store.retrieve(
            query_array, top_k=self.retrieval_top_k, candidate_count=self.retrieval_candidates
        )
        retrieval_output = (
            np.mean(retrieval.values, axis=0)
            if retrieval.values.size
            else np.zeros(self.store.value_width, dtype=np.float32)
        )
        older_parts = int(self.recurrent.state.steps > 0) + int(retrieval.values.size > 0)
        older_output = (
            (recurrent_output + retrieval_output) / older_parts
            if older_parts
            else np.zeros_like(local_output)
        )
        output = (
            (1.0 - self.older_weight) * local_output + self.older_weight * older_output
            if older_parts
            else local_output
        )
        memory = self.store.memory_metrics()
        return HybridEpisodicRead(
            output=np.asarray(output, dtype=np.float32),
            local_output=np.asarray(local_output, dtype=np.float32),
            recurrent_output=np.asarray(recurrent_output, dtype=np.float32),
            retrieval_output=np.asarray(retrieval_output, dtype=np.float32),
            local_tokens=self.store.recent_count,
            older_tokens=self.store.older_count,
            retrievals=len(retrieval.positions),
            bytes_read=retrieval.reads.total_bytes,
            state_bytes=self.recurrent.state.nbytes + memory.total_allocated_bytes,
        )
