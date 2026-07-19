from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.semantic.swiglu import silu


@dataclass(frozen=True)
class CandidateRecall:
    """Set-recall measurements for a router candidate set."""

    hits: int
    candidate_count: int
    oracle_count: int
    recall: float
    precision: float


@dataclass(frozen=True)
class CandidateSelection:
    """Indices and proxy scores returned before exact SwiGLU reranking."""

    indices: NDArray[np.int64]
    scores: NDArray[np.float64]


@dataclass(frozen=True)
class RoutingResult:
    """Candidate and final selections for one hidden state.

    Candidate arrays are aligned with ``candidate_indices``. Selected arrays are
    ordered by decreasing exact contribution score.
    """

    candidate_indices: NDArray[np.int64]
    candidate_proxy_scores: NDArray[np.float64]
    candidate_activations: NDArray[np.float64]
    candidate_exact_scores: NDArray[np.float64]
    selected_indices: NDArray[np.int64]
    selected_activations: NDArray[np.float64]
    selected_exact_scores: NDArray[np.float64]

    def candidate_recall(self, oracle_indices: Iterable[int]) -> CandidateRecall:
        return candidate_recall(self.candidate_indices, oracle_indices)


def _index_set(indices: Iterable[int], name: str) -> set[int]:
    values = np.asarray(list(indices))
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional collection")
    if not np.issubdtype(values.dtype, np.integer) and values.size:
        raise ValueError(f"{name} must contain integer indices")
    result = {int(value) for value in values}
    if any(value < 0 for value in result):
        raise ValueError(f"{name} must not contain negative indices")
    return result


def candidate_recall(
    candidate_indices: Iterable[int], oracle_indices: Iterable[int]
) -> CandidateRecall:
    """Compare router candidates with an oracle top-K set.

    Duplicate indices are treated as one item. An empty oracle has recall 1.0,
    because there are no relevant records to miss.
    """

    candidates = _index_set(candidate_indices, "candidate_indices")
    oracle = _index_set(oracle_indices, "oracle_indices")
    hits = len(candidates & oracle)
    recall = hits / len(oracle) if oracle else 1.0
    if candidates:
        precision = hits / len(candidates)
    else:
        precision = 1.0 if not oracle else 0.0
    return CandidateRecall(
        hits=hits,
        candidate_count=len(candidates),
        oracle_count=len(oracle),
        recall=recall,
        precision=precision,
    )


class JointKeyRouter:
    """Deterministic brute-force fallback router for SwiGLU records.

    Candidate generation compares the hidden state with normalized gate and up
    keys, scoring their joint agreement as::

        max(cos(hidden, gate_key), 0) * abs(cos(hidden, up_key))

    This stage never evaluates SiLU or multiplies full gate/up projections into
    neuron activations. It is deliberately a simple, deterministic O(I*H)
    baseline for validating joint-key routing before an indexed implementation.

    Only candidate records are then evaluated with the exact two-key SwiGLU
    expression. If value vectors are supplied, reranking uses exact contribution
    magnitude ``abs(activation) * ||value||``; otherwise it uses activation
    magnitude.
    """

    def __init__(
        self,
        gate_keys: ArrayLike,
        up_keys: ArrayLike,
        *,
        candidate_count: int,
        top_k: int,
        values: ArrayLike | None = None,
    ) -> None:
        gate = np.asarray(gate_keys, dtype=np.float64)
        up = np.asarray(up_keys, dtype=np.float64)
        if gate.ndim != 2 or up.ndim != 2 or gate.shape != up.shape:
            raise ValueError("gate_keys and up_keys must have the same rank-2 shape [I, H]")
        if gate.shape[0] == 0 or gate.shape[1] == 0:
            raise ValueError("gate_keys and up_keys must not be empty")
        if not np.all(np.isfinite(gate)) or not np.all(np.isfinite(up)):
            raise ValueError("gate_keys and up_keys must contain only finite values")
        if not isinstance(candidate_count, (int, np.integer)) or candidate_count <= 0:
            raise ValueError("candidate_count must be a positive integer")
        if not isinstance(top_k, (int, np.integer)) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if top_k > candidate_count:
            raise ValueError("top_k must not exceed candidate_count")

        self.gate_keys = gate
        self.up_keys = up
        self.candidate_count = int(candidate_count)
        self.top_k = int(top_k)
        self._unit_gate = self._normalize_rows(gate)
        self._unit_up = self._normalize_rows(up)

        if values is None:
            self._value_norms = np.ones(gate.shape[0], dtype=np.float64)
        else:
            value_array = np.asarray(values, dtype=np.float64)
            if value_array.ndim != 2 or value_array.shape[0] != gate.shape[0]:
                raise ValueError("values must have shape [I, output_width]")
            if not np.all(np.isfinite(value_array)):
                raise ValueError("values must contain only finite values")
            self._value_norms = np.linalg.norm(value_array, axis=1)

    @staticmethod
    def _normalize_rows(keys: NDArray[np.float64]) -> NDArray[np.float64]:
        norms = np.linalg.norm(keys, axis=1, keepdims=True)
        return np.divide(keys, norms, out=np.zeros_like(keys), where=norms > 0.0)

    def _hidden(self, hidden: ArrayLike) -> NDArray[np.float64]:
        result = np.asarray(hidden, dtype=np.float64)
        if result.ndim != 1 or result.shape[0] != self.gate_keys.shape[1]:
            raise ValueError(
                f"hidden must have shape [{self.gate_keys.shape[1]}], got {result.shape}"
            )
        if not np.all(np.isfinite(result)):
            raise ValueError("hidden must contain only finite values")
        return result

    def select_candidates(self, hidden: ArrayLike) -> CandidateSelection:
        """Select candidates from key geometry without evaluating SwiGLU."""

        hidden_array = self._hidden(hidden)
        hidden_norm = float(np.linalg.norm(hidden_array))
        if hidden_norm > 0.0:
            query = hidden_array / hidden_norm
            gate_alignment = self._unit_gate @ query
            up_alignment = self._unit_up @ query
            scores = np.maximum(gate_alignment, 0.0) * np.abs(up_alignment)
        else:
            scores = np.zeros(self.gate_keys.shape[0], dtype=np.float64)

        count = min(self.candidate_count, scores.size)
        # A stable full sort defines deterministic index-order tie breaking.
        indices = np.argsort(-scores, kind="stable")[:count].astype(np.int64, copy=False)
        return CandidateSelection(indices=indices, scores=scores[indices])

    def route(self, hidden: ArrayLike) -> RoutingResult:
        """Generate candidates and exactly rerank only those candidates."""

        hidden_array = self._hidden(hidden)
        candidates = self.select_candidates(hidden_array)
        candidate_gate = self.gate_keys[candidates.indices] @ hidden_array
        candidate_up = self.up_keys[candidates.indices] @ hidden_array
        activations = np.asarray(silu(candidate_gate) * candidate_up, dtype=np.float64)
        exact_scores = np.abs(activations) * self._value_norms[candidates.indices]

        selected_count = min(self.top_k, candidates.indices.size)
        order = np.argsort(-exact_scores, kind="stable")[:selected_count]
        return RoutingResult(
            candidate_indices=candidates.indices,
            candidate_proxy_scores=candidates.scores,
            candidate_activations=activations,
            candidate_exact_scores=exact_scores,
            selected_indices=candidates.indices[order],
            selected_activations=activations[order],
            selected_exact_scores=exact_scores[order],
        )
