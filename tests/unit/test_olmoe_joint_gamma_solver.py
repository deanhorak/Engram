from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from engram.evaluation.olmoe_joint_gamma_solver import (
    solve_continuous_box,
    solve_discrete_groups,
)


def _structured_gram(flat: np.ndarray, groups: int, features: int) -> np.ndarray:
    return np.asarray(flat, dtype=np.float64).reshape(
        flat.shape[0],
        groups,
        features,
        groups,
        features,
    )


def _exact_discrete_minimum(
    flat_gram: np.ndarray,
    linear: np.ndarray,
    target_energy: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    batch, groups, candidate_count, _features = candidates.shape
    result = np.full(batch, np.inf, dtype=np.float64)
    for codes in product(range(candidate_count), repeat=groups):
        coefficients = np.concatenate(
            [
                candidates[:, group, code, :]
                for group, code in enumerate(codes)
            ],
            axis=1,
        )
        objective = (
            target_energy
            - 2.0 * np.einsum("ni,ni->n", linear.reshape(batch, -1), coefficients)
            + np.einsum(
                "ni,nij,nj->n",
                coefficients,
                flat_gram,
                coefficients,
            )
        )
        result = np.minimum(result, objective)
    return result


def test_continuous_box_recovers_interior_and_boundary_solution() -> None:
    flat_gram = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    gram = _structured_gram(flat_gram, groups=2, features=2)
    linear = np.asarray(
        [
            [[0.25, 0.75], [-0.50, 1.50]],
            [[0.50, -2.00], [2.00, 0.125]],
        ],
        dtype=np.float64,
    )
    lower = np.zeros_like(linear)
    upper = np.ones_like(linear)
    energy = np.asarray([10.0, 20.0], dtype=np.float64)

    result = solve_continuous_box(gram, linear, energy, lower, upper)

    expected = np.clip(linear, lower, upper)
    np.testing.assert_allclose(result.coefficients, expected, atol=0.0, rtol=0.0)
    expected_objective = (
        energy
        - 2.0 * np.sum(linear * expected, axis=(1, 2))
        + np.sum(expected * expected, axis=(1, 2))
    )
    np.testing.assert_allclose(
        result.objective,
        expected_objective,
        atol=1.0e-14,
        rtol=0.0,
    )
    assert result.converged
    assert result.sweeps <= 2
    assert result.maximum_normalized_kkt_violation == 0.0
    np.testing.assert_array_equal(result.objective_gap_upper_bound, 0.0)
    assert result.maximum_relative_objective_gap == 0.0


def test_continuous_box_is_deterministic_for_coupled_positive_definite_gram() -> None:
    flat = np.asarray(
        [
            [
                [2.0, 0.3, -0.2, 0.1],
                [0.3, 1.5, 0.2, -0.1],
                [-0.2, 0.2, 1.8, 0.4],
                [0.1, -0.1, 0.4, 1.2],
            ]
        ],
        dtype=np.float64,
    )
    gram = _structured_gram(flat, groups=2, features=2)
    linear = np.asarray([[[0.7, -0.2], [1.4, 0.3]]], dtype=np.float64)
    lower = np.asarray([[[-0.1, -0.1], [-0.1, -0.1]]], dtype=np.float64)
    upper = np.asarray([[[0.8, 0.8], [0.8, 0.8]]], dtype=np.float64)
    energy = np.asarray([5.0], dtype=np.float64)

    first = solve_continuous_box(gram, linear, energy, lower, upper)
    second = solve_continuous_box(gram, linear, energy, lower, upper)

    assert first.converged
    assert second.converged
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    np.testing.assert_array_equal(first.objective, second.objective)
    assert first.sweeps == second.sweeps
    assert (
        first.maximum_normalized_kkt_violation
        == second.maximum_normalized_kkt_violation
    )
    np.testing.assert_array_equal(
        first.objective_gap_upper_bound,
        second.objective_gap_upper_bound,
    )
    assert np.all(first.objective_gap_upper_bound <= 1.0e-10 * energy)


def test_two_feature_block_solver_handles_exact_near_null_direction() -> None:
    flat = np.asarray([[[0.04, -0.2], [-0.2, 1.0]]], dtype=np.float64)
    gram = _structured_gram(flat, groups=1, features=2)
    linear = np.asarray([[[-0.06, 0.30]]], dtype=np.float64)
    energy = np.asarray([2.0], dtype=np.float64)
    lower = np.zeros((1, 1, 2), dtype=np.float64)
    upper = np.ones((1, 1, 2), dtype=np.float64)

    result = solve_continuous_box(gram, linear, energy, lower, upper)

    assert result.converged
    assert result.sweeps <= 2
    effective = (
        result.coefficients[0, 0, 1]
        - 0.2 * result.coefficients[0, 0, 0]
    )
    assert effective == pytest.approx(0.3, abs=1.0e-12)
    assert result.objective_gap_upper_bound[0] <= 2.0e-10


def test_discrete_solver_matches_exhaustive_two_group_coupled_optimum() -> None:
    flat = np.asarray(
        [
            [[1.0, -0.9], [-0.9, 1.0]],
            [[1.4, 0.6], [0.6, 1.2]],
        ],
        dtype=np.float64,
    )
    gram = _structured_gram(flat, groups=2, features=1)
    linear = np.asarray([[[0.4], [0.4]], [[-0.3], [0.8]]], dtype=np.float64)
    energy = np.asarray([20.0, 20.0], dtype=np.float64)
    candidate_values = np.asarray(
        [
            [-1.5, -0.75, -0.25, 0.5, 0.0, 0.75, 1.25, 2.0],
            [-1.0, -0.5, -0.1, 0.4, 0.0, 0.9, 1.4, 1.8],
        ],
        dtype=np.float64,
    )
    candidates = np.broadcast_to(
        candidate_values[None, :, :, None],
        (2, 2, 8, 1),
    ).copy()
    continuous = solve_continuous_box(
        gram,
        linear,
        energy,
        np.full((2, 2, 1), -2.0),
        np.full((2, 2, 1), 2.0),
    )

    result = solve_discrete_groups(
        gram,
        linear,
        energy,
        candidates,
        continuous.coefficients,
    )
    exact = _exact_discrete_minimum(
        flat,
        linear,
        energy,
        candidates,
    )

    np.testing.assert_allclose(result.objective, exact, atol=1.0e-13, rtol=0.0)
    selected = np.take_along_axis(
        candidates,
        result.codes[..., None, None],
        axis=2,
    )[:, :, 0, :]
    np.testing.assert_array_equal(result.coefficients, selected)
    assert result.coordinate_converged
    assert result.pair_converged
    assert result.one_flip_locally_optimal
    assert result.two_flip_locally_optimal


def test_discrete_solver_uses_frozen_priority_for_exact_ties() -> None:
    flat = np.eye(3, dtype=np.float64)[None]
    gram = _structured_gram(flat, groups=3, features=1)
    linear = np.zeros((1, 3, 1), dtype=np.float64)
    energy = np.asarray([1.0], dtype=np.float64)
    candidates = np.zeros((1, 3, 8, 1), dtype=np.float64)

    result = solve_discrete_groups(
        gram,
        linear,
        energy,
        candidates,
        np.zeros((1, 3, 1), dtype=np.float64),
    )

    assert result.codes.tolist() == [[4, 4, 4]]
    np.testing.assert_array_equal(result.objective, np.asarray([1.0]))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("asymmetric", "not symmetric"),
        ("indefinite", "not positive semidefinite"),
        ("negative_energy", "shapes or values"),
        ("bad_priority", "code contract"),
        ("bad_base", "code contract"),
    ],
)
def test_solver_rejects_invalid_quadratic_or_code_contract(
    mutation: str,
    message: str,
) -> None:
    flat = np.eye(2, dtype=np.float64)[None]
    gram = _structured_gram(flat, groups=2, features=1)
    linear = np.zeros((1, 2, 1), dtype=np.float64)
    energy = np.asarray([1.0], dtype=np.float64)
    candidates = np.zeros((1, 2, 8, 1), dtype=np.float64)
    priority = (4, 3, 5, 2, 6, 1, 7, 0)
    if mutation == "asymmetric":
        gram[0, 0, 0, 1, 0] = 0.25
    elif mutation == "indefinite":
        gram[0, 0, 0, 1, 0] = 2.0
        gram[0, 1, 0, 0, 0] = 2.0
    elif mutation == "negative_energy":
        energy[0] = -1.0
    elif mutation == "bad_base":
        candidates[0, 0, 4, 0] = 1.0
    else:
        priority = (4, 3, 5, 2, 6, 1, 7, 7)

    with pytest.raises(ValueError, match=message):
        solve_discrete_groups(
            gram,
            linear,
            energy,
            candidates,
            np.zeros((1, 2, 1), dtype=np.float64),
            tie_priority=priority,
        )
