from __future__ import annotations

import numpy as np
import pytest

import engram.evaluation.olmoe_product_simplex_solver as product_solver
from engram.evaluation.olmoe_product_simplex_solver import (
    ProductSimplexSolution,
    solve_product_simplex_least_squares,
)


def _gram_from_design(
    design: np.ndarray,
    *,
    heads: int,
    candidates: int,
) -> np.ndarray:
    gram = np.einsum("nkd,nke->nde", design, design, optimize=True)
    return gram.reshape(
        design.shape[0],
        heads,
        candidates,
        heads,
        candidates,
    )


def _scipy_solution(
    gram: np.ndarray,
    linear: np.ndarray,
    energy: float,
) -> tuple[np.ndarray, float]:
    scipy_optimize = pytest.importorskip("scipy.optimize")
    heads, candidates = linear.shape
    matrix = gram.reshape(heads * candidates, heads * candidates)
    vector = linear.reshape(-1)

    def objective(value):
        return float(energy - 2.0 * vector @ value + value @ matrix @ value)

    def gradient(value):
        return 2.0 * (matrix @ value - vector)

    constraints = [
        {
            "type": "eq",
            "fun": lambda value, head=head: (
                np.sum(value[head * candidates : (head + 1) * candidates]) - 1.0
            ),
            "jac": lambda _value, head=head: np.concatenate(
                (
                    np.zeros(head * candidates),
                    np.ones(candidates),
                    np.zeros((heads - head - 1) * candidates),
                )
            ),
        }
        for head in range(heads)
    ]
    result = scipy_optimize.minimize(
        objective,
        np.full(heads * candidates, 1.0 / candidates),
        jac=gradient,
        bounds=[(0.0, 1.0)] * (heads * candidates),
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 10_000},
    )
    assert result.success, result.message
    return result.x.reshape(heads, candidates), float(result.fun)


def test_random_small_problems_match_scipy_and_gap_certifies():
    rng = np.random.default_rng(20260730)
    rows, heads, candidates, features = 4, 3, 4, 18
    design = rng.normal(size=(rows, features, heads * candidates))
    gram = _gram_from_design(design, heads=heads, candidates=candidates)
    teacher = rng.normal(size=(rows, features))
    linear = np.einsum("nkd,nk->nd", design, teacher).reshape(rows, heads, candidates)
    energy = np.einsum("nk,nk->n", teacher, teacher)
    solution = solve_product_simplex_least_squares(
        gram,
        linear,
        energy,
        max_iterations=20_000,
        relative_tolerance=2e-9,
        absolute_tolerance=1e-11,
    )
    assert isinstance(solution, ProductSimplexSolution)
    assert solution.converged is True
    assert solution.row_converged.tolist() == [True] * rows
    np.testing.assert_allclose(
        solution.coefficients.sum(axis=2),
        1.0,
        rtol=0.0,
        atol=2e-12,
    )
    assert np.min(solution.coefficients) >= 0.0
    for row in range(rows):
        _scipy_x, scipy_objective = _scipy_solution(
            gram[row],
            linear[row],
            float(energy[row]),
        )
        observed_suboptimality = solution.objective[row] - scipy_objective
        assert observed_suboptimality >= -2e-8
        assert observed_suboptimality <= (
            solution.objective_gap_upper_bound[row] + 2e-8
        )
        assert observed_suboptimality <= 2e-7


def test_linear_degenerate_problem_uses_lowest_index_ties():
    rows, heads, candidates = 2, 2, 4
    gram = np.zeros((rows, heads, candidates, heads, candidates))
    linear = np.asarray(
        [
            [[3.0, 3.0, 1.0, 0.0], [0.0, 2.0, 2.0, 1.0]],
            [[-1.0, 0.0, 5.0, 4.0], [7.0, 1.0, 0.0, 7.0]],
        ]
    )
    energy = np.asarray([20.0, 30.0])
    solution = solve_product_simplex_least_squares(
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
    np.testing.assert_array_equal(solution.coefficients, expected)
    np.testing.assert_array_equal(
        solution.objective_gap_upper_bound,
        np.zeros(rows),
    )
    assert solution.converged is True


def test_zero_problem_preserves_feasible_uniform_initial_point():
    gram = np.zeros((1, 3, 5, 3, 5), dtype=np.float64)
    linear = np.zeros((1, 3, 5), dtype=np.float64)
    energy = np.zeros(1, dtype=np.float64)
    solution = solve_product_simplex_least_squares(gram, linear, energy)
    np.testing.assert_array_equal(
        solution.coefficients,
        np.full((1, 3, 5), 0.2),
    )
    np.testing.assert_array_equal(solution.objective, [0.0])
    np.testing.assert_array_equal(solution.objective_gap_upper_bound, [0.0])
    np.testing.assert_array_equal(solution.relative_gap, [0.0])
    assert solution.iterations == 0
    assert solution.max_relative_gap == 0.0


def test_warm_start_and_exact_replay_are_deterministic():
    rng = np.random.default_rng(41)
    rows, heads, candidates, features = 3, 2, 3, 8
    design = rng.normal(size=(rows, features, heads * candidates))
    gram = _gram_from_design(design, heads=heads, candidates=candidates)
    linear = rng.normal(size=(rows, heads, candidates))
    energy = np.full(rows, 50.0)
    initial = rng.dirichlet(np.ones(candidates), size=(rows, heads))
    first = solve_product_simplex_least_squares(
        gram,
        linear,
        energy,
        initial,
        max_iterations=5000,
    )
    replay = solve_product_simplex_least_squares(
        gram,
        linear,
        energy,
        initial,
        max_iterations=5000,
    )
    assert first.iterations == replay.iterations
    assert first.converged == replay.converged
    assert first.max_relative_gap == replay.max_relative_gap
    np.testing.assert_array_equal(first.coefficients, replay.coefficients)
    np.testing.assert_array_equal(first.objective, replay.objective)
    np.testing.assert_array_equal(
        first.objective_gap_upper_bound,
        replay.objective_gap_upper_bound,
    )


def test_batched_line_search_is_exactly_the_scalar_reference():
    gap = np.asarray([0.0, 2.0, 3.0, 4.0, 1e-20], dtype=np.float64)
    curvature = np.asarray([1.0, 0.0, 2.0, -0.5, 1.0], dtype=np.float64)
    maximum = np.asarray([1.0, 0.25, 1.0, 0.5, 1.0], dtype=np.float64)
    tolerance = 128.0 * np.finfo(np.float64).eps
    steps, changes = product_solver._batched_exact_steps(
        gap,
        curvature,
        maximum,
        numerical_tolerance=tolerance,
    )
    reference = [
        product_solver._exact_step(
            float(gap[index]),
            float(curvature[index]),
            float(maximum[index]),
            numerical_tolerance=tolerance,
        )
        for index in range(gap.size)
    ]
    np.testing.assert_array_equal(steps, [row[0] for row in reference])
    np.testing.assert_array_equal(changes, [row[1] for row in reference])


def test_vectorized_batch_rows_match_independent_row_solves_exactly():
    rng = np.random.default_rng(773)
    rows, heads, candidates, features = 6, 4, 5, 13
    design = rng.normal(size=(rows, features, heads * candidates))
    gram = _gram_from_design(design, heads=heads, candidates=candidates)
    linear = rng.normal(size=(rows, heads, candidates))
    energy = np.full(rows, 100.0)
    initial = rng.dirichlet(np.ones(candidates), size=(rows, heads))
    batched = solve_product_simplex_least_squares(
        gram,
        linear,
        energy,
        initial,
        max_iterations=75,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )
    independent = [
        solve_product_simplex_least_squares(
            gram[index : index + 1],
            linear[index : index + 1],
            energy[index : index + 1],
            initial[index : index + 1],
            max_iterations=75,
            relative_tolerance=0.0,
            absolute_tolerance=0.0,
        )
        for index in range(rows)
    ]
    np.testing.assert_array_equal(
        batched.coefficients,
        np.concatenate([row.coefficients for row in independent]),
    )
    np.testing.assert_array_equal(
        batched.objective,
        np.concatenate([row.objective for row in independent]),
    )
    np.testing.assert_array_equal(
        batched.objective_gap_upper_bound,
        np.concatenate([row.objective_gap_upper_bound for row in independent]),
    )
    np.testing.assert_array_equal(
        batched.row_converged,
        np.concatenate([row.row_converged for row in independent]),
    )


def test_gap_remains_valid_when_iteration_limit_is_hit():
    rng = np.random.default_rng(99)
    heads, candidates, features = 3, 3, 7
    design = rng.normal(size=(1, features, heads * candidates))
    gram = _gram_from_design(design, heads=heads, candidates=candidates)
    linear = rng.normal(size=(1, heads, candidates))
    energy = np.asarray([25.0])
    limited = solve_product_simplex_least_squares(
        gram,
        linear,
        energy,
        max_iterations=1,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )
    _scipy_x, optimum = _scipy_solution(gram[0], linear[0], energy[0])
    suboptimality = limited.objective[0] - optimum
    assert limited.iterations == 1
    assert limited.converged is False
    assert suboptimality <= limited.objective_gap_upper_bound[0] + 2e-8


def test_roundoff_scale_negative_eigenvalue_gets_conservative_certificate():
    epsilon = 1e-12
    gram = np.zeros((1, 1, 2, 1, 2), dtype=np.float64)
    gram[0, 0, 0, 0, 0] = -epsilon
    linear = np.zeros((1, 1, 2), dtype=np.float64)
    energy = np.zeros(1, dtype=np.float64)
    initial = np.full((1, 1, 2), 0.5)
    limited = solve_product_simplex_least_squares(
        gram,
        linear,
        energy,
        initial,
        max_iterations=1,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
        psd_tolerance=1e-10,
    )
    true_optimum = -epsilon
    assert limited.objective[0] - true_optimum <= (
        limited.objective_gap_upper_bound[0] + 1e-20
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("gram_rank", "shape"),
        ("shape", "incompatible"),
        ("nan", "finite"),
        ("asymmetric", "symmetric"),
        ("indefinite", "positive semidefinite"),
        ("energy", "nonnegative"),
        ("initial_negative", "simplex"),
        ("initial_sum", "simplex"),
    ],
)
def test_strict_problem_validation(mutation: str, match: str):
    gram = np.zeros((2, 2, 3, 2, 3), dtype=np.float64)
    linear = np.zeros((2, 2, 3), dtype=np.float64)
    energy = np.ones(2, dtype=np.float64)
    initial = np.full((2, 2, 3), 1.0 / 3.0)
    if mutation == "gram_rank":
        gram = gram.reshape(2, 6, 6)
    elif mutation == "shape":
        linear = linear[:, :, :2]
    elif mutation == "nan":
        linear[0, 0, 0] = np.nan
    elif mutation == "asymmetric":
        gram.reshape(2, 6, 6)[0, 0, 1] = 1.0
    elif mutation == "indefinite":
        gram.reshape(2, 6, 6)[0, 0, 0] = -1.0
    elif mutation == "energy":
        energy[0] = -1.0
    elif mutation == "initial_negative":
        initial[0, 0] = [-0.1, 0.5, 0.6]
    else:
        initial[0, 0] = [0.2, 0.2, 0.2]
    with pytest.raises(ValueError, match=match):
        solve_product_simplex_least_squares(
            gram,
            linear,
            energy,
            initial,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iterations": 0},
        {"max_iterations": True},
        {"relative_tolerance": -1.0},
        {"absolute_tolerance": np.nan},
        {"symmetry_tolerance": -1.0},
        {"psd_tolerance": np.inf},
        {"feasibility_tolerance": -1.0},
        {"feasibility_tolerance": 1.0},
    ],
)
def test_strict_option_validation(kwargs):
    gram = np.zeros((1, 1, 1, 1, 1), dtype=np.float64)
    linear = np.zeros((1, 1, 1), dtype=np.float64)
    energy = np.zeros(1, dtype=np.float64)
    with pytest.raises(ValueError):
        solve_product_simplex_least_squares(gram, linear, energy, **kwargs)
