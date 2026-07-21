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

    @classmethod
    def fit(
        cls,
        calibration_states: ArrayLike,
        oracle_membership: ArrayLike,
        *,
        rank: int,
        regularization: float = 1.0,
        positive_weight: float = 1.0,
    ) -> "LowRankMultiLabelRouter":
        """Fit the ridge router directly in low-rank factored form.

        A multi-output ridge solution trained on ``N`` examples has rank at
        most ``N``. For small calibration sets, this method solves the dual
        system and uses QR decompositions to reduce the SVD. Once examples
        outnumber hidden dimensions, it instead solves the smaller primal
        system. Both branches produce the same optimal truncated approximation
        as :meth:`compress` (up to floating-point roundoff).
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
        maximum_rank = min(states.shape[0], states.shape[1], membership.shape[1])
        if not isinstance(rank, int) or not 0 < rank <= maximum_rank:
            raise IVFIndexError("rank must lie within the factored ridge dimensions")

        state_mean = np.mean(states, axis=0)
        centered_states = states - state_mean
        scale = np.sqrt(np.mean(centered_states**2, axis=0))
        scale = np.where(scale > 1e-8, scale, 1.0)
        features = centered_states / scale
        prevalence = np.mean(membership, axis=0)
        targets = membership - prevalence
        if positive_weight != 1.0:
            targets = targets * np.where(membership > 0.0, positive_weight, 1.0)
        if states.shape[0] <= states.shape[1]:
            # The dual system is smaller for the usual small-calibration regime.
            gram = features @ features.T
            gram.flat[:: gram.shape[0] + 1] += regularization
            dual = np.linalg.solve(gram, targets)
            left_factor = features.T / scale[:, None]
            left_basis, left_reduced = np.linalg.qr(left_factor, mode="reduced")
            right_basis, right_reduced = np.linalg.qr(dual.T, mode="reduced")
            core = left_reduced @ right_reduced.T
            core_left, singular_values, core_right = np.linalg.svd(
                core, full_matrices=False
            )
            input_factors = (
                left_basis @ core_left[:, :rank]
            ) * singular_values[:rank]
            output_factors = core_right[:rank] @ right_basis.T
            bias = prevalence - (state_mean @ left_factor) @ dual
        else:
            # With more examples than hidden dimensions, the primal system avoids
            # an unnecessarily large examples-by-examples solve. Materializing the
            # hidden-by-record solution is modest and preserves the exact optimal
            # truncated SVD used by ``compress``.
            gram = features.T @ features
            gram.flat[:: gram.shape[0] + 1] += regularization
            standardized_weights = np.linalg.solve(gram, features.T @ targets)
            weights = standardized_weights / scale[:, None]
            if weights.shape[0] <= weights.shape[1]:
                eigenvalues, eigenvectors = np.linalg.eigh(weights @ weights.T)
                selected = np.argsort(eigenvalues, kind="stable")[-rank:][::-1]
                singular_values = np.sqrt(np.maximum(eigenvalues[selected], 0.0))
                left = eigenvectors[:, selected]
                input_factors = left * singular_values
                output_factors = np.zeros(
                    (rank, weights.shape[1]), dtype=np.float64
                )
                nonzero = singular_values > np.finfo(np.float64).eps
                output_factors[nonzero] = (
                    left[:, nonzero].T @ weights
                ) / singular_values[nonzero, None]
            else:
                eigenvalues, eigenvectors = np.linalg.eigh(weights.T @ weights)
                selected = np.argsort(eigenvalues, kind="stable")[-rank:][::-1]
                right = eigenvectors[:, selected]
                input_factors = weights @ right
                output_factors = right.T
            bias = prevalence - state_mean @ weights
        return cls(input_factors, output_factors, bias)

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
        if self.input_factors.shape[0] != self.gate_keys.shape[1]:
            raise IVFIndexError("router hidden width must match the record key width")
        self.value_norms = np.asarray(value_norms, dtype=np.float64)
        if (
            self.value_norms.shape != (self.records,)
            or not np.all(np.isfinite(self.value_norms))
            or np.any(self.value_norms < 0.0)
        ):
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


def _coverage_query_centers(
    membership: NDArray[np.float64], groups: int
) -> NDArray[np.int64]:
    """Choose deterministic, diverse calibration queries as posting seeds."""

    states = membership.shape[0]
    norms = np.linalg.norm(membership, axis=1, keepdims=True)
    normalized = np.divide(
        membership, norms, out=np.zeros_like(membership), where=norms > 0.0
    )
    selected = [0]
    nearest = 1.0 - normalized @ normalized[0]
    while len(selected) < min(groups, states):
        nearest[np.asarray(selected)] = -1.0
        selected.append(int(np.argmax(nearest)))
        distance = 1.0 - normalized @ normalized[selected[-1]]
        nearest = np.minimum(nearest, distance)
    return np.asarray([selected[index % len(selected)] for index in range(groups)])


def _assign_overlapping_postings(
    demand: NDArray[np.float64], posting_size: int, max_replication: int
) -> NDArray[np.int64]:
    """Fill fixed postings globally while enforcing record replication bounds."""

    groups, records = demand.shape
    slots = groups * posting_size
    if slots < records:
        raise IVFIndexError("overlapping postings need at least one slot per record")
    if slots > records * max_replication:
        raise IVFIndexError("max_replication is too small for the requested posting slots")
    postings: list[list[int]] = [[] for _ in range(groups)]
    populations = np.zeros(groups, dtype=np.int64)
    replication = np.zeros(records, dtype=np.int64)

    # Guarantee every record a home before allocating duplicate slots.  Harder
    # to place records (high peak demand) go first; IDs break all ties.
    record_ids = np.arange(records, dtype=np.int64)
    record_order = np.lexsort((record_ids, -np.max(demand, axis=0)))
    for record in record_order:
        group_ids = np.arange(groups, dtype=np.int64)
        available = populations < posting_size
        if not np.any(available):
            raise IVFIndexError("posting capacity exhausted before covering every record")
        group_order = np.lexsort(
            (group_ids, populations, -demand[:, record])
        )
        group = int(next(value for value in group_order if available[value]))
        postings[group].append(int(record))
        populations[group] += 1
        replication[record] += 1

    # Allocate remaining slots in descending global demand.  The alternating
    # path repair below handles the rare capacity dead ends without turning
    # every slot assignment into a large per-record sort.
    flat_group = np.repeat(np.arange(groups, dtype=np.int64), records)
    flat_record = np.tile(record_ids, groups)
    flat_order = np.lexsort((flat_record, flat_group, -demand.reshape(-1)))
    for flat_index in flat_order:
        if int(np.sum(populations)) == slots:
            break
        group = int(flat_group[flat_index])
        record = int(flat_record[flat_index])
        if populations[group] >= posting_size:
            continue
        if replication[record] >= max_replication or record in postings[group]:
            continue
        postings[group].append(record)
        populations[group] += 1
        replication[record] += 1
    # Repair any remaining dead end with an alternating path.  Moving a full
    # record from one group to another propagates the vacancy until it reaches
    # a record whose replication count is still below the cap.
    while np.any(populations < posting_size):
        root = int(np.flatnonzero(populations < posting_size)[0])
        queue = [root]
        parent: dict[int, tuple[int, int]] = {}
        visited = {root}
        terminal_group = -1
        terminal_record = -1
        for group in queue:
            present = set(postings[group])
            candidates = record_ids[
                np.lexsort((record_ids, -demand[group]))
            ]
            for record_value in candidates:
                record = int(record_value)
                if record in present:
                    continue
                if replication[record] < max_replication:
                    terminal_group = group
                    terminal_record = record
                    break
                for owner in range(groups):
                    if owner not in visited and record in postings[owner]:
                        visited.add(owner)
                        parent[owner] = (group, record)
                        queue.append(owner)
            if terminal_group >= 0:
                break
        if terminal_group < 0:
            break
        postings[terminal_group].append(terminal_record)
        populations[terminal_group] += 1
        replication[terminal_record] += 1
        current = terminal_group
        while current != root:
            parent_group, moved_record = parent[current]
            postings[current].remove(moved_record)
            populations[current] -= 1
            postings[parent_group].append(moved_record)
            populations[parent_group] += 1
            current = parent_group
    if np.any(populations != posting_size):
        raise IVFIndexError("could not fill postings under the replication constraints")

    result = np.empty((groups, posting_size), dtype=np.int64)
    for group, records_in_group in enumerate(postings):
        values = np.asarray(records_in_group, dtype=np.int64)
        order = np.lexsort((values, -demand[group, values]))
        result[group] = values[order]
    return result


def _greedy_coverage_labels(
    membership: NDArray[np.float64],
    postings: NDArray[np.int64],
    candidate_count: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Select posting combinations by true marginal coverage on calibration."""

    samples, records = membership.shape
    groups = postings.shape[0]
    labels = np.zeros((samples, groups), dtype=np.float64)
    marginal = np.zeros((samples, groups), dtype=np.float64)
    group_counts = np.zeros(samples, dtype=np.float64)
    recalls = np.zeros(samples, dtype=np.float64)
    group_ids = np.arange(groups, dtype=np.int64)
    for sample in range(samples):
        selected = np.zeros(groups, dtype=bool)
        covered = np.zeros(records, dtype=bool)
        covered_oracle = np.zeros(records, dtype=bool)
        while int(np.sum(covered)) < candidate_count and not np.all(selected):
            remaining = candidate_count - int(np.sum(covered))
            available = ~covered[postings]
            if remaining < postings.shape[1]:
                available &= np.cumsum(available, axis=1) <= remaining
            new_unique = np.sum(available, axis=1)
            new_hits = np.sum(
                available
                & (membership[sample, postings] > 0.0)
                & (~covered_oracle[postings]),
                axis=1,
            )
            new_hits[selected] = -1
            new_unique[selected] = -1
            best_group = int(
                np.lexsort((group_ids, -new_unique, -new_hits))[0]
            )
            if new_unique[best_group] <= 0:
                break
            posting = postings[best_group]
            new_records = posting[~covered[posting]][:remaining]
            selected[best_group] = True
            labels[sample, best_group] = 1.0
            marginal[sample, best_group] = float(new_hits[best_group])
            covered[new_records] = True
            covered_oracle[new_records] |= membership[sample, new_records] > 0.0
        group_counts[sample] = float(np.sum(selected))
        recalls[sample] = float(np.sum(covered_oracle)) / max(
            float(np.sum(membership[sample])), 1.0
        )
    return labels, marginal, group_counts, recalls


class OverlappingCoverageRouter:
    """Low-rank posting selector trained for multi-posting oracle coverage.

    Unlike :class:`HierarchicalLowRankRouter`, postings may overlap.  Training
    alternates between oracle-greedy posting selection and a globally bounded
    posting update, then learns rank-constrained group-selection labels.
    """

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
        max_replication: int,
        training_metadata: dict[str, float | int] | None = None,
    ) -> None:
        self.input_factors = _matrix(input_factors, "input_factors")
        self.group_factors = _matrix(group_factors, "group_factors")
        if self.input_factors.shape[1] != self.group_factors.shape[0]:
            raise IVFIndexError("low-rank factor widths must agree")
        self.group_bias = np.asarray(group_bias, dtype=np.float64)
        if self.group_bias.shape != (self.groups,) or not np.all(np.isfinite(self.group_bias)):
            raise IVFIndexError("group_bias must contain one finite value per group")
        self.postings = np.asarray(postings, dtype=np.int64)
        if (
            self.postings.ndim != 2
            or self.postings.shape[0] != self.groups
            or not self.postings.shape[1]
        ):
            raise IVFIndexError("postings must contain one fixed-size row per group")
        self.gate_keys = _matrix(gate_keys, "gate_keys")
        self.up_keys = _matrix(up_keys, "up_keys")
        if self.gate_keys.shape != self.up_keys.shape:
            raise IVFIndexError("gate_keys and up_keys must have the same shape")
        if self.input_factors.shape[0] != self.gate_keys.shape[1]:
            raise IVFIndexError("router hidden width must match the record key width")
        self.value_norms = np.asarray(value_norms, dtype=np.float64)
        if (
            self.value_norms.shape != (self.records,)
            or not np.all(np.isfinite(self.value_norms))
            or np.any(self.value_norms < 0.0)
        ):
            raise IVFIndexError("value_norms must contain one non-negative value per record")
        if np.any(self.postings < 0) or np.any(self.postings >= self.records):
            raise IVFIndexError("posting record IDs are out of range")
        if any(np.unique(row).size != row.size for row in self.postings):
            raise IVFIndexError("each posting must contain unique record IDs")
        if not isinstance(max_replication, int) or max_replication <= 0:
            raise IVFIndexError("max_replication must be a positive integer")
        replication = np.bincount(self.postings.reshape(-1), minlength=self.records)
        if np.any(replication < 1) or np.any(replication > max_replication):
            raise IVFIndexError("record replication lies outside [1, max_replication]")
        self.max_replication = max_replication
        self.training_metadata = dict(training_metadata or {})

    @classmethod
    def fit(
        cls,
        calibration_states: ArrayLike,
        oracle_membership: ArrayLike,
        gate_keys: ArrayLike,
        up_keys: ArrayLike,
        values: ArrayLike,
        *,
        rank: int,
        groups: int,
        posting_size: int,
        candidate_count: int,
        regularization: float = 1000.0,
        iterations: int = 8,
        max_replication: int = 4,
    ) -> "OverlappingCoverageRouter":
        states = _matrix(calibration_states, "calibration_states")
        membership = _matrix(oracle_membership, "oracle_membership")
        gate = _matrix(gate_keys, "gate_keys")
        up = _matrix(up_keys, "up_keys")
        value_matrix = _matrix(values, "values")
        records = gate.shape[0]
        if membership.shape != (states.shape[0], records):
            raise IVFIndexError("membership must contain one label per state and record")
        if np.any((membership != 0.0) & (membership != 1.0)):
            raise IVFIndexError("oracle_membership must contain only zero and one")
        if gate.shape != up.shape or states.shape[1] != gate.shape[1]:
            raise IVFIndexError("state, gate, and up dimensions must agree")
        if value_matrix.shape[0] != records:
            raise IVFIndexError("values must contain one row per record")
        if not isinstance(groups, int) or groups <= 0:
            raise IVFIndexError("groups must be a positive integer")
        if not isinstance(posting_size, int) or not 0 < posting_size <= records:
            raise IVFIndexError("posting_size must lie within the record count")
        if not isinstance(candidate_count, int) or not 0 < candidate_count <= records:
            raise IVFIndexError("candidate_count must lie within the record count")
        if not isinstance(iterations, int) or iterations <= 0:
            raise IVFIndexError("iterations must be a positive integer")
        if not isinstance(max_replication, int) or max_replication <= 0:
            raise IVFIndexError("max_replication must be a positive integer")

        centers = _coverage_query_centers(membership, groups)
        prevalence = np.mean(membership, axis=0)
        demand = np.empty((groups, records), dtype=np.float64)
        for group, center in enumerate(centers):
            demand[group] = 2.0 * membership[center] + prevalence
        postings = _assign_overlapping_postings(demand, posting_size, max_replication)
        group_counts = np.zeros(states.shape[0], dtype=np.float64)
        for _ in range(iterations):
            labels, marginal, group_counts, _ = _greedy_coverage_labels(
                membership, postings, candidate_count
            )
            demand = np.broadcast_to(0.01 * prevalence, (groups, records)).copy()
            for sample in range(states.shape[0]):
                selected = np.flatnonzero(labels[sample] > 0.0)
                if not selected.size:
                    continue
                coverage = np.bincount(
                    postings[selected].reshape(-1), minlength=records
                ).astype(np.float64)
                credit = np.divide(
                    membership[sample],
                    np.maximum(coverage, 1.0),
                )
                uncovered = (membership[sample] > 0.0) & (coverage == 0.0)
                credit[uncovered] = 1.0 / selected.size
                for order, group in enumerate(selected):
                    weight = 1.0 / (1.0 + 0.05 * order)
                    demand[group] += weight * credit
                    demand[group] += (
                        0.05
                        * marginal[sample, group]
                        / max(1, posting_size)
                        * membership[sample]
                    )
            demand[
                np.arange(groups)[:, None], postings
            ] += 0.02
            updated = _assign_overlapping_postings(
                demand, posting_size, max_replication
            )
            if np.array_equal(updated, postings):
                postings = updated
                break
            postings = updated

        labels, _, group_counts, oracle_recall = _greedy_coverage_labels(
            membership, postings, candidate_count
        )
        group_router = LowRankMultiLabelRouter.fit(
            states,
            labels,
            rank=rank,
            regularization=regularization,
        )
        replication = np.bincount(postings.reshape(-1), minlength=records)
        return cls(
            input_factors=group_router.input_factors,
            group_factors=group_router.output_factors,
            group_bias=group_router.bias,
            postings=postings,
            gate_keys=gate,
            up_keys=up,
            value_norms=np.linalg.norm(value_matrix, axis=1),
            max_replication=max_replication,
            training_metadata={
                "iterations": iterations,
                "candidate_count": candidate_count,
                "mean_oracle_greedy_recall": float(np.mean(oracle_recall)),
                "mean_oracle_groups_selected": float(np.mean(group_counts)),
                "minimum_replication": int(np.min(replication)),
                "maximum_replication": int(np.max(replication)),
            },
        )

    @property
    def hidden_size(self) -> int:
        return int(self.input_factors.shape[0])

    @property
    def rank(self) -> int:
        return int(self.input_factors.shape[1])

    @property
    def groups(self) -> int:
        return int(self.group_factors.shape[1])

    @property
    def records(self) -> int:
        return int(self.gate_keys.shape[0])

    @property
    def posting_size(self) -> int:
        return int(self.postings.shape[1])

    def router_parameter_bytes(self, *, bytes_per_parameter: int = 4) -> int:
        if not isinstance(bytes_per_parameter, int) or bytes_per_parameter <= 0:
            raise IVFIndexError("bytes_per_parameter must be a positive integer")
        parameters = self.input_factors.size + self.group_factors.size + self.group_bias.size
        return int(parameters * bytes_per_parameter)

    @property
    def posting_bytes(self) -> int:
        bytes_per_id = 2 if self.records <= np.iinfo(np.uint16).max else 4
        return int(self.postings.size * bytes_per_id)

    def candidates(
        self, hidden: ArrayLike, *, candidate_count: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
        """Return a deduplicated posting union without reading record keys."""

        state = np.asarray(hidden, dtype=np.float64)
        if state.shape != (self.hidden_size,) or not np.all(np.isfinite(state)):
            raise IVFIndexError(f"hidden must be finite with shape [{self.hidden_size}]")
        if not isinstance(candidate_count, int) or not 0 < candidate_count <= self.records:
            raise IVFIndexError("candidate_count must lie within the record count")
        group_scores = (state @ self.input_factors) @ self.group_factors + self.group_bias
        group_ids = np.arange(self.groups, dtype=np.int64)
        group_order = np.lexsort((group_ids, -group_scores))
        seen = np.zeros(self.records, dtype=bool)
        candidates: list[int] = []
        selected_groups: list[int] = []
        posting_entries_scanned = 0
        for group in group_order:
            selected_groups.append(int(group))
            posting_entries_scanned += self.posting_size
            for record in self.postings[group]:
                if not seen[record]:
                    seen[record] = True
                    candidates.append(int(record))
                    if len(candidates) == candidate_count:
                        break
            if len(candidates) == candidate_count:
                break
        if len(candidates) != candidate_count:
            raise IVFIndexError("postings could not fill the requested candidate budget")
        return (
            np.asarray(candidates, dtype=np.int64),
            np.asarray(selected_groups, dtype=np.int64),
            posting_entries_scanned,
        )

    def search(self, hidden: ArrayLike, *, candidate_count: int) -> IVFCandidateResult:
        state = np.asarray(hidden, dtype=np.float64)
        candidate_array, selected_groups, posting_entries_scanned = self.candidates(
            state, candidate_count=candidate_count
        )
        exact_scores = np.abs(
            neuron_activations(state, self.gate_keys[candidate_array], self.up_keys[candidate_array])
        ) * self.value_norms[candidate_array]
        order = np.lexsort((candidate_array, -exact_scores))
        return IVFCandidateResult(
            indices=candidate_array[order],
            proxy_scores=exact_scores[order],
            probed_clusters=selected_groups,
            probed_record_count=posting_entries_scanned,
        )


__all__ = [
    "HierarchicalLowRankRouter",
    "LowRankMultiLabelRouter",
    "MultiLabelLinearRouter",
    "OverlappingCoverageRouter",
]
