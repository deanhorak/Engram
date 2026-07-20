"""Learned multi-label semantic routing experiments.

The router learns a linear score for every MLP record from oracle top-record
membership on teacher states. It is an offline quality baseline and is not yet
part of the compiled package format.
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


class MultiLabelLinearRouter:
    """Predict oracle record membership with a multi-output ridge model."""

    def __init__(self, weights: ArrayLike, bias: ArrayLike) -> None:
        self.weights = _matrix(weights, "weights")
        self.bias = np.asarray(bias, dtype=np.float64)
        if self.bias.shape != (self.records,) or not np.all(np.isfinite(self.bias)):
            raise IVFIndexError("bias must contain one finite value per record")

    @classmethod
    def fit(
        cls,
        calibration_states: ArrayLike,
        oracle_membership: ArrayLike,
        *,
        regularization: float = 1.0,
        positive_weight: float = 1.0,
    ) -> "MultiLabelLinearRouter":
        """Fit all record classifiers together using weighted ridge regression.

        ``oracle_membership`` is a state-by-record boolean or 0/1 matrix. The
        dual form keeps fitting practical when calibration examples are fewer
        than the hidden width.
        """

        states = _matrix(calibration_states, "calibration_states")
        membership = _matrix(oracle_membership, "oracle_membership")
        if membership.shape[0] != states.shape[0]:
            raise IVFIndexError("membership and state counts must agree")
        if np.any((membership != 0.0) & (membership != 1.0)):
            raise IVFIndexError("oracle_membership must contain only zero and one")
        if not np.isfinite(regularization) or regularization <= 0.0:
            raise IVFIndexError("regularization must be finite and positive")
        if not np.isfinite(positive_weight) or positive_weight <= 0.0:
            raise IVFIndexError("positive_weight must be finite and positive")

        state_mean = np.mean(states, axis=0)
        centered_states = states - state_mean
        scale = np.sqrt(np.mean(centered_states**2, axis=0))
        scale = np.where(scale > 1e-8, scale, 1.0)
        features = centered_states / scale

        prevalence = np.mean(membership, axis=0)
        targets = membership - prevalence
        if positive_weight != 1.0:
            targets = targets * np.where(membership > 0.0, positive_weight, 1.0)
        gram = features @ features.T
        gram.flat[:: gram.shape[0] + 1] += regularization
        dual = np.linalg.solve(gram, targets)
        standardized_weights = features.T @ dual
        weights = standardized_weights / scale[:, None]
        bias = prevalence - state_mean @ weights
        return cls(weights, bias)

    @property
    def hidden_size(self) -> int:
        return int(self.weights.shape[0])

    @property
    def records(self) -> int:
        return int(self.weights.shape[1])

    def search(self, hidden: ArrayLike, *, candidate_count: int) -> IVFCandidateResult:
        state = np.asarray(hidden, dtype=np.float64)
        if state.shape != (self.hidden_size,) or not np.all(np.isfinite(state)):
            raise IVFIndexError(f"hidden must be finite with shape [{self.hidden_size}]")
        if not isinstance(candidate_count, int) or not 0 < candidate_count <= self.records:
            raise IVFIndexError("candidate_count must lie within the record count")
        scores = state @ self.weights + self.bias
        record_ids = np.arange(self.records, dtype=np.int64)
        order = np.lexsort((record_ids, -scores))[:candidate_count]
        return IVFCandidateResult(
            indices=order,
            proxy_scores=scores[order],
            probed_clusters=np.empty(0, dtype=np.int64),
            probed_record_count=self.records,
        )


class LowRankMultiLabelRouter:
    """A factorized approximation to a fitted multi-label linear router."""

    def __init__(
        self, input_factors: ArrayLike, output_factors: ArrayLike, bias: ArrayLike
    ) -> None:
        self.input_factors = _matrix(input_factors, "input_factors")
        self.output_factors = _matrix(output_factors, "output_factors")
        if self.input_factors.shape[1] != self.output_factors.shape[0]:
            raise IVFIndexError("low-rank factor widths must agree")
        self.bias = np.asarray(bias, dtype=np.float64)
        if self.bias.shape != (self.records,) or not np.all(np.isfinite(self.bias)):
            raise IVFIndexError("bias must contain one finite value per record")

    @classmethod
    def compress(
        cls, router: MultiLabelLinearRouter, *, rank: int
    ) -> "LowRankMultiLabelRouter":
        """Return the optimal rank-constrained approximation in Frobenius norm."""

        maximum_rank = min(router.weights.shape)
        if not isinstance(rank, int) or not 0 < rank <= maximum_rank:
            raise IVFIndexError("rank must lie within the dense weight dimensions")
        left, singular_values, right = np.linalg.svd(router.weights, full_matrices=False)
        return cls(
            left[:, :rank] * singular_values[:rank],
            right[:rank],
            router.bias,
        )

    @property
    def hidden_size(self) -> int:
        return int(self.input_factors.shape[0])

    @property
    def rank(self) -> int:
        return int(self.input_factors.shape[1])

    @property
    def records(self) -> int:
        return int(self.output_factors.shape[1])

    def parameter_bytes(self, *, bytes_per_parameter: int = 4) -> int:
        if not isinstance(bytes_per_parameter, int) or bytes_per_parameter <= 0:
            raise IVFIndexError("bytes_per_parameter must be a positive integer")
        parameters = self.input_factors.size + self.output_factors.size + self.bias.size
        return int(parameters * bytes_per_parameter)

    def scores(self, hidden: ArrayLike) -> NDArray[np.float64]:
        state = np.asarray(hidden, dtype=np.float64)
        if state.shape != (self.hidden_size,) or not np.all(np.isfinite(state)):
            raise IVFIndexError(f"hidden must be finite with shape [{self.hidden_size}]")
        return (state @ self.input_factors) @ self.output_factors + self.bias

    def search(self, hidden: ArrayLike, *, candidate_count: int) -> IVFCandidateResult:
        if not isinstance(candidate_count, int) or not 0 < candidate_count <= self.records:
            raise IVFIndexError("candidate_count must lie within the record count")
        scores = self.scores(hidden)
        record_ids = np.arange(self.records, dtype=np.int64)
        order = np.lexsort((record_ids, -scores))[:candidate_count]
        return IVFCandidateResult(
            indices=order,
            proxy_scores=scores[order],
            probed_clusters=np.empty(0, dtype=np.int64),
            probed_record_count=self.records,
        )


def _balanced_groups(
    features: NDArray[np.float64], groups: int, iterations: int
) -> NDArray[np.int64]:
    """Cluster records with deterministic equal-capacity cosine k-means."""

    records = features.shape[0]
    if records % groups:
        raise IVFIndexError("record count must be divisible by group count")
    capacity = records // groups
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = np.divide(features, norms, out=np.zeros_like(features), where=norms > 0.0)
    selected = [0]
    nearest = np.sum((normalized - normalized[0]) ** 2, axis=1)
    while len(selected) < groups:
        nearest[np.asarray(selected)] = -1.0
        record = int(np.argmax(nearest))
        selected.append(record)
        nearest = np.minimum(
            nearest, np.sum((normalized - normalized[record]) ** 2, axis=1)
        )
    centroids = normalized[np.asarray(selected)].copy()
    assignments = np.zeros(records, dtype=np.int64)
    previous = None
    for _ in range(iterations):
        scores = normalized @ centroids.T
        preferences = np.argsort(-scores, axis=1, kind="stable")
        if groups == 1:
            priority = np.arange(records)
        else:
            confidence = scores[np.arange(records), preferences[:, 0]] - scores[
                np.arange(records), preferences[:, 1]
            ]
            priority = np.lexsort((np.arange(records), -confidence))
        populations = np.zeros(groups, dtype=np.int64)
        for record in priority:
            for group in preferences[record]:
                if populations[group] < capacity:
                    assignments[record] = group
                    populations[group] += 1
                    break
        if previous is not None and np.array_equal(assignments, previous):
            break
        for group in range(groups):
            centroid = np.mean(normalized[assignments == group], axis=0)
            norm = float(np.linalg.norm(centroid))
            centroids[group] = centroid / norm if norm > 0.0 else centroid
        previous = assignments.copy()
    return assignments


class HierarchicalLowRankRouter:
    """Select balanced posting groups with low-rank scores, then exact-rerank."""

    def __init__(
        self,
        *,
        input_factors: ArrayLike,
        group_factors: ArrayLike,
        group_bias: ArrayLike,
        postings: ArrayLike,
        gate_keys: ArrayLike,
        up_keys: ArrayLike,
        value_norms: ArrayLike,
    ) -> None:
        self.input_factors = _matrix(input_factors, "input_factors")
        self.group_factors = _matrix(group_factors, "group_factors")
        if self.input_factors.shape[1] != self.group_factors.shape[0]:
            raise IVFIndexError("low-rank factor widths must agree")
        self.group_bias = np.asarray(group_bias, dtype=np.float64)
        if self.group_bias.shape != (self.groups,) or not np.all(np.isfinite(self.group_bias)):
            raise IVFIndexError("group_bias must contain one finite value per group")
        self.postings = np.asarray(postings, dtype=np.int64)
        if self.postings.ndim != 2 or self.postings.shape[0] != self.groups:
            raise IVFIndexError("postings must contain one equal-sized row per group")
        self.gate_keys = _matrix(gate_keys, "gate_keys")
        self.up_keys = _matrix(up_keys, "up_keys")
        if self.gate_keys.shape != self.up_keys.shape:
            raise IVFIndexError("gate_keys and up_keys must have the same shape")
        self.value_norms = np.asarray(value_norms, dtype=np.float64)
        if self.value_norms.shape != (self.records,) or np.any(self.value_norms < 0.0):
            raise IVFIndexError("value_norms must contain one non-negative value per record")
        expected = np.arange(self.records)
        if not np.array_equal(np.sort(self.postings.reshape(-1)), expected):
            raise IVFIndexError("postings must partition every record exactly once")

    @classmethod
    def fit(
        cls,
        router: MultiLabelLinearRouter,
        gate_keys: ArrayLike,
        up_keys: ArrayLike,
        values: ArrayLike,
        *,
        rank: int,
        groups: int,
        iterations: int = 12,
    ) -> "HierarchicalLowRankRouter":
        if not isinstance(groups, int) or groups <= 0:
            raise IVFIndexError("groups must be a positive integer")
        if not isinstance(iterations, int) or iterations <= 0:
            raise IVFIndexError("iterations must be a positive integer")
        low_rank = LowRankMultiLabelRouter.compress(router, rank=rank)
        assignments = _balanced_groups(low_rank.output_factors.T, groups, iterations)
        posting_size = router.records // groups
        postings = np.empty((groups, posting_size), dtype=np.int64)
        group_factors = np.empty((rank, groups), dtype=np.float64)
        group_bias = np.empty(groups, dtype=np.float64)
        for group in range(groups):
            records = np.flatnonzero(assignments == group)
            postings[group] = records
            group_factors[:, group] = np.mean(low_rank.output_factors[:, records], axis=1)
            group_bias[group] = float(np.mean(router.bias[records]))
        value_matrix = _matrix(values, "values")
        return cls(
            input_factors=low_rank.input_factors,
            group_factors=group_factors,
            group_bias=group_bias,
            postings=postings,
            gate_keys=gate_keys,
            up_keys=up_keys,
            value_norms=np.linalg.norm(value_matrix, axis=1),
        )

    @classmethod
    def fit_coverage(
        cls,
        calibration_states: ArrayLike,
        oracle_membership: ArrayLike,
        gate_keys: ArrayLike,
        up_keys: ArrayLike,
        values: ArrayLike,
        *,
        rank: int,
        groups: int,
        regularization: float = 1000.0,
        iterations: int = 12,
    ) -> "HierarchicalLowRankRouter":
        """Train balanced groups and their scores directly for oracle coverage."""

        states = _matrix(calibration_states, "calibration_states")
        membership = _matrix(oracle_membership, "oracle_membership")
        gate = _matrix(gate_keys, "gate_keys")
        up = _matrix(up_keys, "up_keys")
        value_matrix = _matrix(values, "values")
        if membership.shape != (states.shape[0], gate.shape[0]):
            raise IVFIndexError("membership must contain one label per state and record")
        if np.any((membership != 0.0) & (membership != 1.0)):
            raise IVFIndexError("oracle_membership must contain only zero and one")
        if gate.shape != up.shape or states.shape[1] != gate.shape[1]:
            raise IVFIndexError("state, gate, and up dimensions must agree")
        if value_matrix.shape[0] != gate.shape[0]:
            raise IVFIndexError("values must contain one row per record")
        if not isinstance(groups, int) or groups <= 0 or gate.shape[0] % groups:
            raise IVFIndexError("groups must be positive and divide the record count")
        if not isinstance(iterations, int) or iterations <= 0:
            raise IVFIndexError("iterations must be a positive integer")
        if not np.isfinite(regularization) or regularization <= 0.0:
            raise IVFIndexError("regularization must be finite and positive")
        if not isinstance(rank, int) or not 0 < rank <= min(states.shape[1], groups):
            raise IVFIndexError("rank must lie within the state and group dimensions")

        assignments = _balanced_groups(membership.T, groups, iterations)
        posting_size = gate.shape[0] // groups
        postings = np.empty((groups, posting_size), dtype=np.int64)
        targets = np.empty((states.shape[0], groups), dtype=np.float64)
        for group in range(groups):
            records = np.flatnonzero(assignments == group)
            postings[group] = records
            targets[:, group] = np.sum(membership[:, records], axis=1)

        state_mean = np.mean(states, axis=0)
        centered_states = states - state_mean
        scale = np.sqrt(np.mean(centered_states**2, axis=0))
        scale = np.where(scale > 1e-8, scale, 1.0)
        features = centered_states / scale
        target_mean = np.mean(targets, axis=0)
        gram = features @ features.T
        gram.flat[:: gram.shape[0] + 1] += regularization
        dual = np.linalg.solve(gram, targets - target_mean)
        weights = (features.T @ dual) / scale[:, None]
        bias = target_mean - state_mean @ weights
        left, singular_values, right = np.linalg.svd(weights, full_matrices=False)
        return cls(
            input_factors=left[:, :rank] * singular_values[:rank],
            group_factors=right[:rank],
            group_bias=bias,
            postings=postings,
            gate_keys=gate,
            up_keys=up,
            value_norms=np.linalg.norm(value_matrix, axis=1),
        )

    @property
    def hidden_size(self) -> int:
        return int(self.input_factors.shape[0])

    @property
    def groups(self) -> int:
        return int(self.group_factors.shape[1])

    @property
    def records(self) -> int:
        return int(self.gate_keys.shape[0])

    @property
    def records_per_group(self) -> int:
        return int(self.postings.shape[1])

    def router_parameter_bytes(self, *, bytes_per_parameter: int = 4) -> int:
        if not isinstance(bytes_per_parameter, int) or bytes_per_parameter <= 0:
            raise IVFIndexError("bytes_per_parameter must be a positive integer")
        parameters = self.input_factors.size + self.group_factors.size + self.group_bias.size
        return int(parameters * bytes_per_parameter)

    def search(
        self, hidden: ArrayLike, *, groups_to_probe: int, candidate_count: int
    ) -> IVFCandidateResult:
        state = np.asarray(hidden, dtype=np.float64)
        if state.shape != (self.hidden_size,) or not np.all(np.isfinite(state)):
            raise IVFIndexError(f"hidden must be finite with shape [{self.hidden_size}]")
        if not isinstance(groups_to_probe, int) or not 0 < groups_to_probe <= self.groups:
            raise IVFIndexError("groups_to_probe must lie within the group count")
        if not isinstance(candidate_count, int) or candidate_count <= 0:
            raise IVFIndexError("candidate_count must be positive")
        group_scores = (state @ self.input_factors) @ self.group_factors + self.group_bias
        group_ids = np.arange(self.groups, dtype=np.int64)
        selected_groups = np.lexsort((group_ids, -group_scores))[:groups_to_probe]
        candidates = self.postings[selected_groups].reshape(-1)
        exact_scores = np.abs(
            neuron_activations(state, self.gate_keys[candidates], self.up_keys[candidates])
        ) * self.value_norms[candidates]
        order = np.lexsort((candidates, -exact_scores))[: min(candidate_count, candidates.size)]
        return IVFCandidateResult(
            indices=candidates[order],
            proxy_scores=exact_scores[order],
            probed_clusters=selected_groups,
            probed_record_count=int(candidates.size),
        )


__all__ = [
    "HierarchicalLowRankRouter",
    "LowRankMultiLabelRouter",
    "MultiLabelLinearRouter",
]
