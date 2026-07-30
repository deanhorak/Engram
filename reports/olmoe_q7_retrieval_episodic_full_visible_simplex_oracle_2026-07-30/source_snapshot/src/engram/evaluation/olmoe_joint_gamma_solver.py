"""Deterministic batched solvers for the joint per-head gamma oracle."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np


_DEFAULT_TIE_PRIORITY = (4, 3, 5, 2, 6, 1, 7, 0)
_OBJECTIVE_EPSILON_MULTIPLIER = 64.0


@dataclass(frozen=True)
class ContinuousBoxResult:
    coefficients: np.ndarray
    objective: np.ndarray
    objective_gap_upper_bound: np.ndarray
    sweeps: int
    converged: bool
    maximum_normalized_kkt_violation: float
    maximum_relative_objective_gap: float


@dataclass(frozen=True)
class DiscreteGroupResult:
    codes: np.ndarray
    coefficients: np.ndarray
    objective: np.ndarray
    start_count: int
    maximum_coordinate_sweeps: int
    pair_sweeps: int
    coordinate_converged: bool
    pair_converged: bool
    one_flip_locally_optimal: bool
    two_flip_locally_optimal: bool


def _validated_quadratic(
    gram: np.ndarray,
    linear: np.ndarray,
    target_energy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    matrix = np.ascontiguousarray(gram, dtype=np.float64)
    vector = np.ascontiguousarray(linear, dtype=np.float64)
    energy = np.ascontiguousarray(target_energy, dtype=np.float64)
    if matrix.ndim != 5:
        raise ValueError("joint-gamma Gram tensor must have five dimensions")
    batch, groups, features, other_groups, other_features = matrix.shape
    if (
        batch <= 0
        or groups <= 0
        or features <= 0
        or other_groups != groups
        or other_features != features
        or vector.shape != (batch, groups, features)
        or energy.shape != (batch,)
        or not np.isfinite(matrix).all()
        or not np.isfinite(vector).all()
        or not np.isfinite(energy).all()
        or np.any(energy <= 0.0)
    ):
        raise ValueError("joint-gamma quadratic shapes or values are invalid")
    width = groups * features
    flat = matrix.reshape(batch, width, width)
    scale = np.maximum(
        1.0,
        np.max(np.abs(flat), axis=(1, 2)),
    )
    asymmetry = np.max(np.abs(flat - np.swapaxes(flat, 1, 2)), axis=(1, 2))
    if np.any(asymmetry > 1.0e-10 * scale):
        raise ValueError("joint-gamma Gram tensor is not symmetric")
    flat = np.ascontiguousarray(
        0.5 * (flat + np.swapaxes(flat, 1, 2)),
        dtype=np.float64,
    )
    diagonal = np.diagonal(flat, axis1=1, axis2=2)
    if np.any(diagonal < -1.0e-12 * scale[:, None]):
        raise ValueError("joint-gamma Gram tensor has a negative diagonal")
    minimum_eigenvalue = np.linalg.eigvalsh(flat)[:, 0]
    if np.any(minimum_eigenvalue < -1.0e-10 * scale):
        raise ValueError("joint-gamma Gram tensor is not positive semidefinite")
    return (
        flat,
        vector.reshape(batch, width),
        energy,
        batch,
        groups,
        features,
    )


def _objective(
    gram: np.ndarray,
    linear: np.ndarray,
    target_energy: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    quadratic = np.einsum(
        "ni,nij,nj->n",
        coefficients,
        gram,
        coefficients,
        optimize=True,
    )
    result = target_energy - 2.0 * np.einsum(
        "ni,ni->n",
        linear,
        coefficients,
        optimize=True,
    ) + quadratic
    tolerance = _OBJECTIVE_EPSILON_MULTIPLIER * np.finfo(np.float64).eps * np.maximum(
        1.0,
        target_energy,
    )
    if np.any(result < -tolerance):
        raise ValueError("joint-gamma quadratic produced negative error energy")
    return np.maximum(result, 0.0)


def _maximum_normalized_box_kkt_violation(
    gram: np.ndarray,
    linear: np.ndarray,
    coefficients: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    coefficient_tolerance: float,
) -> float:
    gradient = np.einsum(
        "nij,nj->ni",
        gram,
        coefficients,
        optimize=True,
    ) - linear
    bound_scale = np.maximum(
        1.0,
        np.maximum(np.abs(lower), np.abs(upper)),
    )
    at_lower = coefficients <= lower + coefficient_tolerance * bound_scale
    at_upper = coefficients >= upper - coefficient_tolerance * bound_scale
    violation = np.abs(gradient)
    violation[at_lower & ~at_upper] = np.maximum(
        -gradient[at_lower & ~at_upper],
        0.0,
    )
    violation[at_upper & ~at_lower] = np.maximum(
        gradient[at_upper & ~at_lower],
        0.0,
    )
    violation[at_lower & at_upper] = 0.0
    row_norm = np.max(np.sum(np.abs(gram), axis=2), axis=1)
    kkt_scale = np.maximum(
        1.0,
        np.maximum(
            np.max(np.abs(linear), axis=1),
            row_norm * np.maximum(1.0, np.max(np.abs(coefficients), axis=1)),
        ),
    )
    return float(np.max(violation / kkt_scale[:, None]))


def _box_objective_gap(
    gram: np.ndarray,
    linear: np.ndarray,
    target_energy: np.ndarray,
    coefficients: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    gradient_half = np.einsum(
        "nij,nj->ni",
        gram,
        coefficients,
        optimize=True,
    ) - linear
    minimizing_bound = np.where(gradient_half >= 0.0, lower, upper)
    gap_half = np.einsum(
        "ni,ni->n",
        gradient_half,
        coefficients - minimizing_bound,
        optimize=True,
    )
    linear_term = 2.0 * np.einsum(
        "ni,ni->n",
        linear,
        coefficients,
        optimize=True,
    )
    quadratic_term = np.einsum(
        "ni,nij,nj->n",
        coefficients,
        gram,
        coefficients,
        optimize=True,
    )
    numerical_scale = np.maximum(
        1.0,
        target_energy + np.abs(linear_term) + np.abs(quadratic_term),
    )
    numerical_tolerance = (
        256.0 * np.finfo(np.float64).eps * numerical_scale
    )
    if np.any(gap_half < -numerical_tolerance):
        raise ValueError("joint-gamma box objective gap became negative")
    objective_gap = 2.0 * np.maximum(gap_half, 0.0)
    allowed = np.maximum(
        relative_tolerance * target_energy,
        numerical_tolerance,
    )
    relative = objective_gap / np.maximum(1.0, target_energy)
    return objective_gap, allowed, float(np.max(relative))


def _two_feature_block_sweep(
    gram: np.ndarray,
    linear: np.ndarray,
    coefficients: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    groups: int,
    tolerance: np.ndarray,
) -> None:
    batch = coefficients.shape[0]
    for order in (range(groups), range(groups - 1, -1, -1)):
        for group in order:
            group_slice = slice(group * 2, (group + 1) * 2)
            current = coefficients[:, group_slice]
            block = gram[:, group_slice, group_slice]
            field = (
                np.einsum(
                    "nij,nj->ni",
                    gram[:, group_slice, :],
                    coefficients,
                    optimize=True,
                )
                - np.einsum(
                    "nij,nj->ni",
                    block,
                    current,
                    optimize=True,
                )
                - linear[:, group_slice]
            )
            block_lower = lower[:, group_slice]
            block_upper = upper[:, group_slice]
            candidates = np.empty((batch, 10, 2), dtype=np.float64)
            valid = np.ones((batch, 10), dtype=bool)
            candidates[:, 0] = current
            candidates[:, 1, 0] = block_lower[:, 0]
            candidates[:, 1, 1] = block_lower[:, 1]
            candidates[:, 2, 0] = block_lower[:, 0]
            candidates[:, 2, 1] = block_upper[:, 1]
            candidates[:, 3, 0] = block_upper[:, 0]
            candidates[:, 3, 1] = block_lower[:, 1]
            candidates[:, 4, 0] = block_upper[:, 0]
            candidates[:, 4, 1] = block_upper[:, 1]

            diagonal_scale = np.maximum(
                1.0,
                np.max(np.abs(block), axis=(1, 2)),
            )
            usable_one = block[:, 1, 1] > (
                np.finfo(np.float64).eps * diagonal_scale
            )
            for index, fixed_zero in (
                (5, block_lower[:, 0]),
                (6, block_upper[:, 0]),
            ):
                candidates[:, index, 0] = fixed_zero
                candidates[:, index, 1] = block_lower[:, 1]
                candidates[usable_one, index, 1] = np.clip(
                    -(
                        field[usable_one, 1]
                        + block[usable_one, 1, 0] * fixed_zero[usable_one]
                    )
                    / block[usable_one, 1, 1],
                    block_lower[usable_one, 1],
                    block_upper[usable_one, 1],
                )
                valid[:, index] = usable_one

            usable_zero = block[:, 0, 0] > (
                np.finfo(np.float64).eps * diagonal_scale
            )
            for index, fixed_one in (
                (7, block_lower[:, 1]),
                (8, block_upper[:, 1]),
            ):
                candidates[:, index, 0] = block_lower[:, 0]
                candidates[:, index, 1] = fixed_one
                candidates[usable_zero, index, 0] = np.clip(
                    -(
                        field[usable_zero, 0]
                        + block[usable_zero, 0, 1] * fixed_one[usable_zero]
                    )
                    / block[usable_zero, 0, 0],
                    block_lower[usable_zero, 0],
                    block_upper[usable_zero, 0],
                )
                valid[:, index] = usable_zero

            determinant = (
                block[:, 0, 0] * block[:, 1, 1]
                - block[:, 0, 1] * block[:, 1, 0]
            )
            usable_interior = determinant > (
                np.finfo(np.float64).eps * diagonal_scale * diagonal_scale
            )
            candidates[:, 9] = current
            candidates[usable_interior, 9, 0] = (
                block[usable_interior, 0, 1] * field[usable_interior, 1]
                - block[usable_interior, 1, 1] * field[usable_interior, 0]
            ) / determinant[usable_interior]
            candidates[usable_interior, 9, 1] = (
                block[usable_interior, 1, 0] * field[usable_interior, 0]
                - block[usable_interior, 0, 0] * field[usable_interior, 1]
            ) / determinant[usable_interior]
            valid[:, 9] = usable_interior & np.all(
                candidates[:, 9] >= block_lower,
                axis=1,
            ) & np.all(
                candidates[:, 9] <= block_upper,
                axis=1,
            )

            costs = np.einsum(
                "nci,nij,ncj->nc",
                candidates,
                block,
                candidates,
                optimize=True,
            ) + 2.0 * np.einsum(
                "nci,ni->nc",
                candidates,
                field,
                optimize=True,
            )
            costs[~valid] = np.inf
            selected = np.argmin(costs, axis=1)
            best = candidates[np.arange(batch), selected]
            current_cost = costs[:, 0]
            best_cost = costs[np.arange(batch), selected]
            improve = best_cost < current_cost - tolerance
            coefficients[improve, group_slice] = best[improve]


def _scalar_coordinate_sweep(
    gram: np.ndarray,
    linear: np.ndarray,
    coefficients: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    diagonal = np.diagonal(gram, axis1=1, axis2=2)
    width = coefficients.shape[1]
    for order in (range(width), range(width - 1, -1, -1)):
        for coordinate in order:
            row_dot = np.einsum(
                "ni,ni->n",
                gram[:, coordinate, :],
                coefficients,
                optimize=True,
            )
            numerator = (
                linear[:, coordinate]
                - row_dot
                + diagonal[:, coordinate] * coefficients[:, coordinate]
            )
            usable = diagonal[:, coordinate] > (
                np.finfo(np.float64).eps
                * np.maximum(
                    1.0,
                    np.max(np.abs(gram[:, coordinate, :]), axis=1),
                )
            )
            proposed = coefficients[:, coordinate].copy()
            proposed[usable] = numerator[usable] / diagonal[usable, coordinate]
            flat = ~usable
            proposed[flat & (numerator > 0.0)] = upper[
                flat & (numerator > 0.0),
                coordinate,
            ]
            proposed[flat & (numerator < 0.0)] = lower[
                flat & (numerator < 0.0),
                coordinate,
            ]
            proposed[flat & (numerator == 0.0)] = np.clip(
                0.0,
                lower[flat & (numerator == 0.0), coordinate],
                upper[flat & (numerator == 0.0), coordinate],
            )
            coefficients[:, coordinate] = np.clip(
                proposed,
                lower[:, coordinate],
                upper[:, coordinate],
            )


def solve_continuous_box(
    gram: np.ndarray,
    linear: np.ndarray,
    target_energy: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    maximum_sweeps: int = 10_000,
    coefficient_tolerance: float = 1.0e-12,
    kkt_relative_tolerance: float = 1.0e-10,
) -> ContinuousBoxResult:
    (
        flat_gram,
        flat_linear,
        energy,
        batch,
        groups,
        features,
    ) = _validated_quadratic(gram, linear, target_energy)
    lower_array = np.ascontiguousarray(lower, dtype=np.float64)
    upper_array = np.ascontiguousarray(upper, dtype=np.float64)
    expected = (batch, groups, features)
    if (
        lower_array.shape != expected
        or upper_array.shape != expected
        or not np.isfinite(lower_array).all()
        or not np.isfinite(upper_array).all()
        or np.any(lower_array > upper_array)
        or maximum_sweeps <= 0
        or not math_is_positive_finite(coefficient_tolerance)
        or not math_is_positive_finite(kkt_relative_tolerance)
    ):
        raise ValueError("joint-gamma continuous bounds or tolerances are invalid")
    flat_lower = lower_array.reshape(batch, -1)
    flat_upper = upper_array.reshape(batch, -1)
    coefficients = np.clip(
        np.zeros_like(flat_linear),
        flat_lower,
        flat_upper,
    )
    converged = False
    completed_sweeps = 0
    maximum_violation = float("inf")
    objective_gap = np.full(batch, np.inf, dtype=np.float64)
    maximum_relative_gap = float("inf")
    update_tolerance = np.zeros(batch, dtype=np.float64)
    for sweep in range(maximum_sweeps):
        if features == 2:
            _two_feature_block_sweep(
                flat_gram,
                flat_linear,
                coefficients,
                flat_lower,
                flat_upper,
                groups=groups,
                tolerance=update_tolerance,
            )
        else:
            _scalar_coordinate_sweep(
                flat_gram,
                flat_linear,
                coefficients,
                flat_lower,
                flat_upper,
            )
        completed_sweeps = sweep + 1
        objective_gap, allowed_gap, maximum_relative_gap = _box_objective_gap(
            flat_gram,
            flat_linear,
            energy,
            coefficients,
            flat_lower,
            flat_upper,
            relative_tolerance=kkt_relative_tolerance,
        )
        if np.all(objective_gap <= allowed_gap):
            converged = True
            break
    maximum_violation = _maximum_normalized_box_kkt_violation(
        flat_gram,
        flat_linear,
        coefficients,
        flat_lower,
        flat_upper,
        coefficient_tolerance=coefficient_tolerance,
    )
    objective_gap, allowed_gap, maximum_relative_gap = _box_objective_gap(
        flat_gram,
        flat_linear,
        energy,
        coefficients,
        flat_lower,
        flat_upper,
        relative_tolerance=kkt_relative_tolerance,
    )
    converged = bool(converged and np.all(objective_gap <= allowed_gap))
    objective = _objective(
        flat_gram,
        flat_linear,
        energy,
        coefficients,
    )
    return ContinuousBoxResult(
        coefficients=np.ascontiguousarray(
            coefficients.reshape(batch, groups, features),
            dtype=np.float64,
        ),
        objective=np.ascontiguousarray(objective, dtype=np.float64),
        objective_gap_upper_bound=np.ascontiguousarray(
            objective_gap,
            dtype=np.float64,
        ),
        sweeps=completed_sweeps,
        converged=converged,
        maximum_normalized_kkt_violation=maximum_violation,
        maximum_relative_objective_gap=maximum_relative_gap,
    )


def math_is_positive_finite(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0.0)


def _validate_candidates(
    candidates: np.ndarray,
    *,
    batch: int,
    groups: int,
    features: int,
    tie_priority: Sequence[int],
    base_code: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    values = np.ascontiguousarray(candidates, dtype=np.float64)
    if (
        values.ndim != 4
        or values.shape[:2] != (batch, groups)
        or values.shape[3] != features
        or values.shape[2] <= 1
        or not np.isfinite(values).all()
    ):
        raise ValueError("joint-gamma candidate coefficients are invalid")
    codes = values.shape[2]
    priority = tuple(int(code) for code in tie_priority)
    if (
        len(priority) != codes
        or set(priority) != set(range(codes))
        or base_code < 0
        or base_code >= codes
        or not np.array_equal(
            values[:, :, base_code, :],
            np.zeros((batch, groups, features), dtype=np.float64),
        )
    ):
        raise ValueError("joint-gamma candidate code contract is invalid")
    return values, priority


def _gather_candidates(
    candidates: np.ndarray,
    codes: np.ndarray,
) -> np.ndarray:
    return np.take_along_axis(
        candidates,
        codes[..., None, None],
        axis=2,
    )[:, :, 0, :]


def _coordinate_descent(
    gram: np.ndarray,
    linear: np.ndarray,
    candidates: np.ndarray,
    codes: np.ndarray,
    priority: tuple[int, ...],
    tolerance: np.ndarray,
    maximum_sweeps: int,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    batch, groups, _candidate_count, features = candidates.shape
    coefficients = _gather_candidates(candidates, codes)
    completed = 0
    converged = False
    priority_array = np.asarray(priority, dtype=np.int64)
    for sweep in range(maximum_sweeps):
        changed = np.zeros(batch, dtype=bool)
        for order in (range(groups), range(groups - 1, -1, -1)):
            for group in order:
                group_slice = slice(group * features, (group + 1) * features)
                flat_coefficients = coefficients.reshape(batch, -1)
                current = coefficients[:, group, :]
                row_dot = np.einsum(
                    "nij,nj->ni",
                    gram[:, group_slice, :],
                    flat_coefficients,
                    optimize=True,
                )
                self_gram = gram[:, group_slice, group_slice]
                field = (
                    row_dot
                    - np.einsum(
                        "nij,nj->ni",
                        self_gram,
                        current,
                        optimize=True,
                    )
                    - linear[:, group_slice]
                )
                ordered = candidates[:, group, priority_array, :]
                costs = np.einsum(
                    "nci,nij,ncj->nc",
                    ordered,
                    self_gram,
                    ordered,
                    optimize=True,
                ) + 2.0 * np.einsum(
                    "nci,ni->nc",
                    ordered,
                    field,
                    optimize=True,
                )
                best_priority = np.argmin(costs, axis=1)
                best_codes = priority_array[best_priority]
                best_cost = costs[np.arange(batch), best_priority]
                current_cost = np.einsum(
                    "ni,nij,nj->n",
                    current,
                    self_gram,
                    current,
                    optimize=True,
                ) + 2.0 * np.einsum(
                    "ni,ni->n",
                    current,
                    field,
                    optimize=True,
                )
                improve = best_cost < current_cost - tolerance
                if np.any(improve):
                    codes[improve, group] = best_codes[improve]
                    coefficients[improve, group, :] = candidates[
                        improve,
                        group,
                        best_codes[improve],
                        :,
                    ]
                    changed |= improve
        completed = sweep + 1
        if not np.any(changed):
            converged = True
            break
    return codes, coefficients, completed, converged


def _has_coordinate_improvement(
    gram: np.ndarray,
    linear: np.ndarray,
    candidates: np.ndarray,
    codes: np.ndarray,
    priority: tuple[int, ...],
    tolerance: np.ndarray,
) -> np.ndarray:
    _codes, _coefficients, _completed, converged = _coordinate_descent(
        gram,
        linear,
        candidates,
        codes.copy(),
        priority,
        tolerance,
        1,
    )
    return np.any(_codes != codes, axis=1) | (not converged)


def _pair_polish(
    gram: np.ndarray,
    linear: np.ndarray,
    candidates: np.ndarray,
    codes: np.ndarray,
    priority: tuple[int, ...],
    tolerance: np.ndarray,
    maximum_sweeps: int,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    batch, groups, candidate_count, features = candidates.shape
    priority_array = np.asarray(priority, dtype=np.int64)
    ordered_pairs = tuple(combinations(range(groups), 2))
    coefficients = _gather_candidates(candidates, codes)
    completed = 0
    converged = False
    for sweep in range(maximum_sweeps):
        changed = np.zeros(batch, dtype=bool)
        for pair_order in (ordered_pairs, tuple(reversed(ordered_pairs))):
            for left, right in pair_order:
                left_slice = slice(left * features, (left + 1) * features)
                right_slice = slice(right * features, (right + 1) * features)
                flat = coefficients.reshape(batch, -1)
                left_current = coefficients[:, left, :]
                right_current = coefficients[:, right, :]
                g_ll = gram[:, left_slice, left_slice]
                g_rr = gram[:, right_slice, right_slice]
                g_lr = gram[:, left_slice, right_slice]
                left_field = (
                    np.einsum(
                        "nij,nj->ni",
                        gram[:, left_slice, :],
                        flat,
                        optimize=True,
                    )
                    - np.einsum(
                        "nij,nj->ni",
                        g_ll,
                        left_current,
                        optimize=True,
                    )
                    - np.einsum(
                        "nij,nj->ni",
                        g_lr,
                        right_current,
                        optimize=True,
                    )
                    - linear[:, left_slice]
                )
                right_field = (
                    np.einsum(
                        "nij,nj->ni",
                        gram[:, right_slice, :],
                        flat,
                        optimize=True,
                    )
                    - np.einsum(
                        "nji,nj->ni",
                        g_lr,
                        left_current,
                        optimize=True,
                    )
                    - np.einsum(
                        "nij,nj->ni",
                        g_rr,
                        right_current,
                        optimize=True,
                    )
                    - linear[:, right_slice]
                )
                left_values = candidates[:, left, priority_array, :]
                right_values = candidates[:, right, priority_array, :]
                left_cost = np.einsum(
                    "nci,nij,ncj->nc",
                    left_values,
                    g_ll,
                    left_values,
                    optimize=True,
                ) + 2.0 * np.einsum(
                    "nci,ni->nc",
                    left_values,
                    left_field,
                    optimize=True,
                )
                right_cost = np.einsum(
                    "nci,nij,ncj->nc",
                    right_values,
                    g_rr,
                    right_values,
                    optimize=True,
                ) + 2.0 * np.einsum(
                    "nci,ni->nc",
                    right_values,
                    right_field,
                    optimize=True,
                )
                cross = 2.0 * np.einsum(
                    "nci,nij,ndj->ncd",
                    left_values,
                    g_lr,
                    right_values,
                    optimize=True,
                )
                costs = left_cost[:, :, None] + right_cost[:, None, :] + cross
                flat_costs = costs.reshape(batch, candidate_count * candidate_count)
                best_flat = np.argmin(flat_costs, axis=1)
                best_left_priority = best_flat // candidate_count
                best_right_priority = best_flat % candidate_count
                best_left = priority_array[best_left_priority]
                best_right = priority_array[best_right_priority]
                best_cost = flat_costs[np.arange(batch), best_flat]
                current_cost = (
                    np.einsum(
                        "ni,nij,nj->n",
                        left_current,
                        g_ll,
                        left_current,
                        optimize=True,
                    )
                    + np.einsum(
                        "ni,nij,nj->n",
                        right_current,
                        g_rr,
                        right_current,
                        optimize=True,
                    )
                    + 2.0
                    * np.einsum(
                        "ni,nij,nj->n",
                        left_current,
                        g_lr,
                        right_current,
                        optimize=True,
                    )
                    + 2.0
                    * np.einsum(
                        "ni,ni->n",
                        left_current,
                        left_field,
                        optimize=True,
                    )
                    + 2.0
                    * np.einsum(
                        "ni,ni->n",
                        right_current,
                        right_field,
                        optimize=True,
                    )
                )
                improve = best_cost < current_cost - tolerance
                if np.any(improve):
                    codes[improve, left] = best_left[improve]
                    codes[improve, right] = best_right[improve]
                    coefficients[improve, left, :] = candidates[
                        improve,
                        left,
                        best_left[improve],
                        :,
                    ]
                    coefficients[improve, right, :] = candidates[
                        improve,
                        right,
                        best_right[improve],
                        :,
                    ]
                    changed |= improve
        completed = sweep + 1
        if not np.any(changed):
            converged = True
            break
    return codes, coefficients, completed, converged


def solve_discrete_groups(
    gram: np.ndarray,
    linear: np.ndarray,
    target_energy: np.ndarray,
    candidates: np.ndarray,
    continuous_coefficients: np.ndarray,
    *,
    base_code: int = 4,
    tie_priority: Sequence[int] = _DEFAULT_TIE_PRIORITY,
    maximum_coordinate_sweeps: int = 128,
    maximum_pair_sweeps: int = 32,
) -> DiscreteGroupResult:
    (
        flat_gram,
        flat_linear,
        energy,
        batch,
        groups,
        features,
    ) = _validated_quadratic(gram, linear, target_energy)
    values, priority = _validate_candidates(
        candidates,
        batch=batch,
        groups=groups,
        features=features,
        tie_priority=tie_priority,
        base_code=base_code,
    )
    continuous = np.ascontiguousarray(continuous_coefficients, dtype=np.float64)
    if (
        continuous.shape != (batch, groups, features)
        or not np.isfinite(continuous).all()
        or maximum_coordinate_sweeps <= 0
        or maximum_pair_sweeps <= 0
    ):
        raise ValueError("joint-gamma discrete solver inputs are invalid")
    tolerance = (
        _OBJECTIVE_EPSILON_MULTIPLIER
        * np.finfo(np.float64).eps
        * np.maximum(1.0, energy)
    )
    priority_array = np.asarray(priority, dtype=np.int64)
    base = np.full((batch, groups), base_code, dtype=np.uint8)
    distances = np.sum(
        (values - continuous[:, :, None, :]) ** 2,
        axis=3,
    )
    rounded_priority = np.argmin(distances[:, :, priority_array], axis=2)
    rounded = priority_array[rounded_priority].astype(np.uint8)
    starts: list[np.ndarray] = [base, rounded]
    independent = np.empty((batch, groups), dtype=np.uint8)
    for group in range(groups):
        group_slice = slice(group * features, (group + 1) * features)
        self_gram = flat_gram[:, group_slice, group_slice]
        ordered = values[:, group, priority_array, :]
        costs = np.einsum(
            "nci,nij,ncj->nc",
            ordered,
            self_gram,
            ordered,
            optimize=True,
        ) - 2.0 * np.einsum(
            "nci,ni->nc",
            ordered,
            flat_linear[:, group_slice],
            optimize=True,
        )
        independent[:, group] = priority_array[np.argmin(costs, axis=1)]
    starts.append(independent)
    for code in range(values.shape[2]):
        if code != base_code:
            starts.append(np.full((batch, groups), code, dtype=np.uint8))
    best_codes: np.ndarray | None = None
    best_coefficients: np.ndarray | None = None
    best_objective: np.ndarray | None = None
    all_converged = True
    maximum_completed = 0
    rank = np.empty(values.shape[2], dtype=np.int64)
    rank[priority_array] = np.arange(values.shape[2])
    for start in starts:
        codes, coefficients, completed, converged = _coordinate_descent(
            flat_gram,
            flat_linear,
            values,
            start.copy(),
            priority,
            tolerance,
            maximum_coordinate_sweeps,
        )
        objective = _objective(
            flat_gram,
            flat_linear,
            energy,
            coefficients.reshape(batch, -1),
        )
        maximum_completed = max(maximum_completed, completed)
        all_converged = all_converged and converged
        if best_codes is None:
            best_codes = codes
            best_coefficients = coefficients
            best_objective = objective
            continue
        improve = objective < best_objective - tolerance
        tied = np.abs(objective - best_objective) <= tolerance
        lex_better = np.zeros(batch, dtype=bool)
        undecided = tied.copy()
        for group in range(groups):
            candidate_rank = rank[codes[:, group]]
            current_rank = rank[best_codes[:, group]]
            lex_better |= undecided & (candidate_rank < current_rank)
            undecided &= candidate_rank == current_rank
        replace = improve | lex_better
        if np.any(replace):
            best_codes[replace] = codes[replace]
            best_coefficients[replace] = coefficients[replace]
            best_objective[replace] = objective[replace]
    assert best_codes is not None
    assert best_coefficients is not None
    assert best_objective is not None
    (
        polished_codes,
        polished_coefficients,
        pair_sweeps,
        pair_converged,
    ) = _pair_polish(
        flat_gram,
        flat_linear,
        values,
        best_codes.copy(),
        priority,
        tolerance,
        maximum_pair_sweeps,
    )
    polished_objective = _objective(
        flat_gram,
        flat_linear,
        energy,
        polished_coefficients.reshape(batch, -1),
    )
    if np.any(polished_objective > best_objective + tolerance):
        raise AssertionError("joint-gamma pair polish regressed the objective")
    one_flip = not np.any(
        _has_coordinate_improvement(
            flat_gram,
            flat_linear,
            values,
            polished_codes,
            priority,
            tolerance,
        )
    )
    check_codes, _check_coefficients, _check_sweeps, pair_check = _pair_polish(
        flat_gram,
        flat_linear,
        values,
        polished_codes.copy(),
        priority,
        tolerance,
        1,
    )
    two_flip = pair_check and np.array_equal(check_codes, polished_codes)
    return DiscreteGroupResult(
        codes=np.ascontiguousarray(polished_codes, dtype=np.uint8),
        coefficients=np.ascontiguousarray(
            polished_coefficients,
            dtype=np.float64,
        ),
        objective=np.ascontiguousarray(polished_objective, dtype=np.float64),
        start_count=len(starts),
        maximum_coordinate_sweeps=maximum_completed,
        pair_sweeps=pair_sweeps,
        coordinate_converged=all_converged,
        pair_converged=pair_converged,
        one_flip_locally_optimal=one_flip,
        two_flip_locally_optimal=two_flip,
    )
