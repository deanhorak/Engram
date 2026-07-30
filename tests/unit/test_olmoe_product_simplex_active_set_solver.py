from __future__ import annotations

import numpy as np

from engram.evaluation import olmoe_product_simplex_active_set_solver as active
from engram.evaluation.olmoe_product_simplex_solver import (
    ProductSimplexSolution,
    solve_product_simplex_least_squares,
)


def _problem(
    *,
    seed: int,
    rows: int,
    heads: int,
    candidates: int,
    features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    design = rng.normal(size=(rows, features, heads * candidates))
    flat_gram = np.einsum("nki,nkj->nij", design, design, optimize=True)
    teacher = rng.normal(size=(rows, features))
    linear = np.einsum("nki,nk->ni", design, teacher, optimize=True).reshape(
        rows,
        heads,
        candidates,
    )
    energy = np.einsum("nk,nk->n", teacher, teacher, optimize=True)
    initial = rng.dirichlet(np.ones(candidates), size=(rows, heads))
    return (
        flat_gram.reshape(rows, heads, candidates, heads, candidates),
        linear,
        energy,
        initial,
    )


def test_bulk_active_set_matches_certified_reference_and_is_deterministic():
    gram, linear, energy, initial = _problem(
        seed=20260730,
        rows=8,
        heads=4,
        candidates=5,
        features=40,
    )
    first = active.solve_product_simplex_least_squares_active_set(
        gram,
        linear,
        energy,
        initial,
        relative_tolerance=1e-9,
    )
    replay = active.solve_product_simplex_least_squares_active_set(
        gram,
        linear,
        energy,
        initial,
        relative_tolerance=1e-9,
    )
    reference = solve_product_simplex_least_squares(
        gram,
        linear,
        energy,
        initial,
        max_iterations=20_000,
        relative_tolerance=1e-9,
    )
    assert isinstance(first, ProductSimplexSolution)
    assert first.converged
    np.testing.assert_array_equal(first.coefficients, replay.coefficients)
    np.testing.assert_array_equal(first.objective, replay.objective)
    np.testing.assert_array_equal(
        first.objective_gap_upper_bound,
        replay.objective_gap_upper_bound,
    )
    np.testing.assert_allclose(first.objective, reference.objective, atol=2e-8)
    assert np.max(first.objective_gap_upper_bound) <= 1e-9 * np.max(
        np.maximum(1.0, energy)
    )


def test_singular_linear_problem_falls_back_with_lowest_index_ties():
    rows, heads, candidates = 2, 2, 4
    gram = np.zeros((rows, heads, candidates, heads, candidates))
    linear = np.asarray(
        [
            [[3.0, 3.0, 1.0, 0.0], [0.0, 2.0, 2.0, 1.0]],
            [[-1.0, 0.0, 5.0, 4.0], [7.0, 1.0, 0.0, 7.0]],
        ]
    )
    energy = np.asarray([20.0, 30.0])
    solved = active.solve_product_simplex_least_squares_active_set(
        gram,
        linear,
        energy,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )
    expected = np.zeros_like(linear)
    expected[0, 0, 0] = 1.0
    expected[0, 1, 1] = 1.0
    expected[1, 0, 2] = 1.0
    expected[1, 1, 0] = 1.0
    np.testing.assert_array_equal(solved.coefficients, expected)
    np.testing.assert_array_equal(solved.objective_gap_upper_bound, 0.0)


def test_low_rank_cycle_or_singularity_returns_feasible_certified_fallback():
    gram, linear, energy, initial = _problem(
        seed=17,
        rows=4,
        heads=5,
        candidates=6,
        features=7,
    )
    solved = active.solve_product_simplex_least_squares_active_set(
        gram,
        linear,
        energy,
        initial,
        max_active_set_iterations=3,
        fallback_max_iterations=1000,
        relative_tolerance=1e-7,
    )
    np.testing.assert_allclose(
        np.sum(solved.coefficients, axis=2),
        1.0,
        rtol=0.0,
        atol=2e-10,
    )
    assert np.min(solved.coefficients) >= 0.0
    assert np.isfinite(solved.objective_gap_upper_bound).all()


def test_one_active_iteration_still_returns_a_valid_gap_certificate():
    gram, linear, energy, initial = _problem(
        seed=91,
        rows=3,
        heads=3,
        candidates=4,
        features=18,
    )
    solved = active.solve_product_simplex_least_squares_active_set(
        gram,
        linear,
        energy,
        initial,
        max_active_set_iterations=1,
        fallback_max_iterations=1,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )
    reference = solve_product_simplex_least_squares(
        gram,
        linear,
        energy,
        initial,
        max_iterations=20_000,
        relative_tolerance=1e-11,
    )
    suboptimality = solved.objective - reference.objective
    assert np.all(
        suboptimality
        <= solved.objective_gap_upper_bound
        + reference.objective_gap_upper_bound
        + 2e-8
    )


def test_bad_kkt_residual_tolerance_fails_closed_to_reference(monkeypatch):
    gram, linear, energy, initial = _problem(
        seed=5,
        rows=2,
        heads=2,
        candidates=3,
        features=12,
    )
    calls: list[int] = []
    original = active.reference.solve_product_simplex_least_squares

    def recorded(*args, **kwargs):
        calls.append(np.asarray(args[0]).shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        active.reference,
        "solve_product_simplex_least_squares",
        recorded,
    )
    solved = active.solve_product_simplex_least_squares_active_set(
        gram,
        linear,
        energy,
        initial,
        kkt_residual_tolerance=0.0,
        fallback_max_iterations=2000,
    )
    assert calls == [2]
    assert solved.converged
