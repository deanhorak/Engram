"""Trace-calibrated semantic routing experiments.

This router learns coarse regions of hidden-state space from teacher traces and
stores the records with the largest mean contribution in each region. It is an
experimental quality baseline, not yet a serialized runtime index.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.semantic.ivf import IVFCandidateResult, IVFIndexError
from engram.semantic.swiglu import neuron_activations


def _matrix(value: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not result.shape[0] or not result.shape[1]:
        raise IVFIndexError(f"{name} must be a non-empty rank-2 matrix")
    if not np.all(np.isfinite(result)):
        raise IVFIndexError(f"{name} must contain only finite values")
    return result


def _normalize_rows(value: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    return np.divide(value, norms, out=np.zeros_like(value), where=norms > 0.0)


def _fit_state_centroids(
    states: NDArray[np.float64], clusters: int, iterations: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    if clusters > states.shape[0]:
        raise IVFIndexError("num_clusters cannot exceed calibration state count")
    selected = [0]
    nearest = np.sum((states - states[0]) ** 2, axis=1)
    while len(selected) < clusters:
        nearest[np.asarray(selected)] = -1.0
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(nearest, np.sum((states - states[index]) ** 2, axis=1))
    centroids = states[np.asarray(selected)].copy()
    assignments = np.zeros(states.shape[0], dtype=np.int64)
    previous = None
    for _ in range(iterations):
        scores = states @ centroids.T
        assignments = np.argmax(scores, axis=1).astype(np.int64, copy=False)
        populations = np.bincount(assignments, minlength=clusters)
        for empty in np.flatnonzero(populations == 0):
            confidence = scores[np.arange(states.shape[0]), assignments]
            eligible = populations[assignments] > 1
            donor = int(np.argmin(np.where(eligible, confidence, np.inf)))
            populations[assignments[donor]] -= 1
            assignments[donor] = int(empty)
            populations[empty] += 1
        if previous is not None and np.array_equal(assignments, previous):
            break
        for cluster in range(clusters):
            centroid = np.mean(states[assignments == cluster], axis=0)
            norm = float(np.linalg.norm(centroid))
            centroids[cluster] = centroid / norm if norm > 0.0 else centroid
        previous = assignments.copy()
    return centroids, assignments


class TraceCalibratedRouter:
    """Map hidden-state regions to records observed as useful in teacher traces."""

    def __init__(
        self,
        *,
        gate_keys: ArrayLike,
        up_keys: ArrayLike,
        value_norms: ArrayLike,
        state_centroids: ArrayLike,
        cluster_records: ArrayLike,
        cluster_scores: ArrayLike,
    ) -> None:
        self.gate_keys = _matrix(gate_keys, "gate_keys")
        self.up_keys = _matrix(up_keys, "up_keys")
        if self.gate_keys.shape != self.up_keys.shape:
            raise IVFIndexError("gate_keys and up_keys must have the same shape")
        self.value_norms = np.asarray(value_norms, dtype=np.float64)
        if self.value_norms.shape != (self.records,) or np.any(self.value_norms < 0.0):
            raise IVFIndexError("value_norms must contain one non-negative value per record")
        self.state_centroids = _matrix(state_centroids, "state_centroids")
        if self.state_centroids.shape[1] != self.hidden_size:
            raise IVFIndexError("state centroid width differs from key width")
        self.cluster_records = np.asarray(cluster_records, dtype=np.int64)
        self.cluster_scores = np.asarray(cluster_scores, dtype=np.float64)
        expected = (self.clusters, self.records_per_cluster)
        if self.cluster_records.shape != expected or self.cluster_scores.shape != expected:
            raise IVFIndexError("cluster posting arrays have incompatible shapes")
        if np.any(self.cluster_records < 0) or np.any(self.cluster_records >= self.records):
            raise IVFIndexError("cluster posting contains an invalid record ID")

    @classmethod
    def fit(
        cls,
        gate_keys: ArrayLike,
        up_keys: ArrayLike,
        values: ArrayLike,
        calibration_states: ArrayLike,
        *,
        num_clusters: int,
        records_per_cluster: int,
        iterations: int = 20,
    ) -> "TraceCalibratedRouter":
        gate = _matrix(gate_keys, "gate_keys")
        up = _matrix(up_keys, "up_keys")
        value_matrix = _matrix(values, "values")
        states = _matrix(calibration_states, "calibration_states")
        if gate.shape != up.shape or value_matrix.shape[0] != gate.shape[0]:
            raise IVFIndexError("gate, up, and value record counts must agree")
        if states.shape[1] != gate.shape[1]:
            raise IVFIndexError("calibration state width differs from key width")
        if not isinstance(num_clusters, int) or num_clusters <= 0:
            raise IVFIndexError("num_clusters must be a positive integer")
        if not isinstance(records_per_cluster, int) or not 0 < records_per_cluster <= gate.shape[0]:
            raise IVFIndexError("records_per_cluster must lie within the record count")
        if not isinstance(iterations, int) or iterations <= 0:
            raise IVFIndexError("iterations must be a positive integer")
        normalized_states = _normalize_rows(states)
        centroids, assignments = _fit_state_centroids(
            normalized_states, num_clusters, iterations
        )
        value_norms = np.linalg.norm(value_matrix, axis=1)
        contributions = np.abs(neuron_activations(states, gate, up)) * value_norms
        postings = np.empty((num_clusters, records_per_cluster), dtype=np.int64)
        posting_scores = np.empty((num_clusters, records_per_cluster), dtype=np.float64)
        global_scores = np.mean(contributions, axis=0)
        record_ids = np.arange(gate.shape[0], dtype=np.int64)
        for cluster in range(num_clusters):
            rows = contributions[assignments == cluster]
            scores = np.mean(rows, axis=0) if rows.size else global_scores
            order = np.lexsort((record_ids, -scores))[:records_per_cluster]
            postings[cluster] = order
            posting_scores[cluster] = scores[order]
        return cls(
            gate_keys=gate,
            up_keys=up,
            value_norms=value_norms,
            state_centroids=centroids,
            cluster_records=postings,
            cluster_scores=posting_scores,
        )

    @property
    def records(self) -> int:
        return int(self.gate_keys.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.gate_keys.shape[1])

    @property
    def clusters(self) -> int:
        return int(self.state_centroids.shape[0])

    @property
    def records_per_cluster(self) -> int:
        return int(self.cluster_records.shape[1])

    def search(
        self, hidden: ArrayLike, *, probes: int, candidate_count: int
    ) -> IVFCandidateResult:
        state = np.asarray(hidden, dtype=np.float64)
        if state.shape != (self.hidden_size,) or not np.all(np.isfinite(state)):
            raise IVFIndexError(f"hidden must be finite with shape [{self.hidden_size}]")
        if not isinstance(probes, int) or not 0 < probes <= self.clusters:
            raise IVFIndexError("probes must lie within the state-cluster count")
        if not isinstance(candidate_count, int) or candidate_count <= 0:
            raise IVFIndexError("candidate_count must be positive")
        norm = float(np.linalg.norm(state))
        query = state / norm if norm > 0.0 else np.zeros_like(state)
        cluster_ids = np.arange(self.clusters, dtype=np.int64)
        cluster_order = np.lexsort((cluster_ids, -(self.state_centroids @ query)))[:probes]
        candidates = np.unique(self.cluster_records[cluster_order].reshape(-1))
        activations = neuron_activations(
            state, self.gate_keys[candidates], self.up_keys[candidates]
        )
        scores = np.abs(activations) * self.value_norms[candidates]
        order = np.lexsort((candidates, -scores))
        chosen = order[: min(candidate_count, candidates.size)]
        return IVFCandidateResult(
            indices=candidates[chosen],
            proxy_scores=scores[chosen],
            probed_clusters=cluster_order,
            probed_record_count=int(candidates.size),
        )


__all__ = ["TraceCalibratedRouter"]
