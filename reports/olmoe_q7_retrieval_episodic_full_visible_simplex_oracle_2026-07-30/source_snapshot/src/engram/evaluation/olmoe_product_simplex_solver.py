"""Deterministic certified quadratic optimization on a product of simplices.

The solver minimizes, independently for every batch row,

    target_energy - 2 * linear.T @ coefficients
        + coefficients.T @ gram @ coefficients

subject to every head's coefficient vector being nonnegative and summing to
one.  It uses deterministic block-coordinate Frank-Wolfe steps, augmented by
pairwise away steps, with exact quadratic line searches.  The ordinary
product-simplex Frank-Wolfe dual gap is recomputed after every sweep; for a
positive-semidefinite Gram matrix it is a certificate on objective
suboptimality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ProductSimplexSolution:
    """Result of a batched product-simplex quadratic solve."""

    coefficients: NDArray[np.float64]
    objective: NDArray[np.float64]
    objective_gap_upper_bound: NDArray[np.float64]
    relative_gap: NDArray[np.float64]
    row_converged: NDArray[np.bool_]
    iterations: int
    converged: bool
    max_relative_gap: float


def _as_finite_float64(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(value)
    if (
        array.dtype.kind not in "fc"
        or array.dtype.kind == "c"
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite real floating-point array")
    return np.asarray(array, dtype=np.float64)


def _validate_problem(
    gram: ArrayLike,
    linear: ArrayLike,
    target_energy: ArrayLike,
    initial_coefficients: ArrayLike | None,
    *,
    symmetry_tolerance: float,
    psd_tolerance: float,
    feasibility_tolerance: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    if (
        not np.isfinite(symmetry_tolerance)
        or symmetry_tolerance < 0.0
        or not np.isfinite(psd_tolerance)
        or psd_tolerance < 0.0
        or not np.isfinite(feasibility_tolerance)
        or feasibility_tolerance < 0.0
        or feasibility_tolerance >= 1.0
    ):
        raise ValueError("validation tolerances must be finite and nonnegative")
    matrix = _as_finite_float64(gram, name="gram")
    vector = _as_finite_float64(linear, name="linear")
    energy = _as_finite_float64(target_energy, name="target_energy")
    if matrix.ndim != 5:
        raise ValueError("gram must have shape [N,H,C,H,C]")
    rows, heads, candidates, other_heads, other_candidates = matrix.shape
    if (
        rows <= 0
        or heads <= 0
        or candidates <= 0
        or other_heads != heads
        or other_candidates != candidates
        or vector.shape != (rows, heads, candidates)
        or energy.shape != (rows,)
    ):
        raise ValueError("gram, linear, and target_energy shapes are incompatible")
    if np.any(energy < -feasibility_tolerance):
        raise ValueError("target_energy must be nonnegative")
    energy = np.maximum(energy, 0.0)
    width = heads * candidates
    flat = matrix.reshape(rows, width, width)
    transpose = np.swapaxes(flat, 1, 2)
    scale = np.maximum(
        1.0,
        np.max(np.abs(flat), axis=(1, 2)),
    )
    asymmetry = np.max(np.abs(flat - transpose), axis=(1, 2))
    if np.any(asymmetry > symmetry_tolerance * scale):
        raise ValueError("gram must be symmetric within symmetry_tolerance")
    # Use the exactly symmetric representative for deterministic gradients and
    # line searches.  PSD validation is deliberately strict because the
    # Frank-Wolfe gap is a global certificate only for a convex quadratic.
    flat = (flat + transpose) * 0.5
    convexity_shift = np.empty(rows, dtype=np.float64)
    for row in range(rows):
        minimum_eigenvalue = float(np.linalg.eigvalsh(flat[row])[0])
        if minimum_eigenvalue < -psd_tolerance * float(scale[row]):
            raise ValueError("gram must be positive semidefinite")
        # A tiny negative eigenvalue accepted as numerical roundoff is not
        # silently ignored.  Its exact convexifying shift is retained so the
        # final certificate includes the corresponding worst-case correction.
        convexity_shift[row] = max(0.0, -minimum_eigenvalue)
    matrix = np.ascontiguousarray(
        flat.reshape(rows, heads, candidates, heads, candidates)
    )
    if initial_coefficients is None:
        coefficients = np.full(
            (rows, heads, candidates),
            1.0 / candidates,
            dtype=np.float64,
        )
    else:
        coefficients = _as_finite_float64(
            initial_coefficients,
            name="initial_coefficients",
        )
        if coefficients.shape != (rows, heads, candidates):
            raise ValueError("initial_coefficients must have shape [N,H,C]")
        minimum = np.min(coefficients, axis=2)
        sums = np.sum(coefficients, axis=2, dtype=np.float64)
        if np.any(minimum < -feasibility_tolerance) or np.any(
            np.abs(sums - 1.0) > feasibility_tolerance
        ):
            raise ValueError("each initial coefficient head must lie on the simplex")
        coefficients = np.maximum(coefficients, 0.0)
        sums = np.sum(coefficients, axis=2, keepdims=True, dtype=np.float64)
        if np.any(sums == 0.0):
            raise ValueError("each initial coefficient head must have positive mass")
        coefficients = coefficients / sums
    return (
        matrix,
        np.ascontiguousarray(vector),
        np.ascontiguousarray(energy),
        np.ascontiguousarray(coefficients),
        convexity_shift,
    )


def _objective_and_gradient(
    gram: NDArray[np.float64],
    linear: NDArray[np.float64],
    target_energy: NDArray[np.float64],
    coefficients: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rows = coefficients.shape[0]
    flat_gram = gram.reshape(rows, -1, coefficients.shape[1] * coefficients.shape[2])
    flat_x = coefficients.reshape(rows, -1)
    flat_b = linear.reshape(rows, -1)
    product = np.einsum(
        "nij,nj->ni",
        flat_gram,
        flat_x,
        optimize=True,
    )
    objective = (
        target_energy
        - 2.0 * np.einsum("ni,ni->n", flat_b, flat_x, optimize=True)
        + np.einsum("ni,ni->n", flat_x, product, optimize=True)
    )
    gradient = (2.0 * (product - flat_b)).reshape(coefficients.shape)
    return np.ascontiguousarray(objective), np.ascontiguousarray(gradient)


def _product_simplex_gap(
    gradient: NDArray[np.float64],
    coefficients: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the deterministic product-simplex Frank-Wolfe dual gap."""

    current = np.sum(gradient * coefficients, axis=2, dtype=np.float64)
    # np.argmin and np.min use the lowest index for exact ties.
    minimum = np.min(gradient, axis=2)
    gap = np.sum(current - minimum, axis=1, dtype=np.float64)
    numerical_scale = np.maximum(
        1.0,
        np.max(np.abs(gradient), axis=(1, 2)),
    )
    if np.any(gap < -64.0 * np.finfo(np.float64).eps * numerical_scale):
        raise RuntimeError("computed a negative product-simplex dual gap")
    return np.maximum(gap, 0.0)


def _relative_gap(
    gap: NDArray[np.float64],
    objective: NDArray[np.float64],
    target_energy: NDArray[np.float64],
) -> NDArray[np.float64]:
    scale = np.maximum.reduce(
        (
            np.ones_like(objective),
            np.abs(objective),
            np.abs(target_energy),
        )
    )
    return gap / scale


def _certified_product_simplex_gap(
    gradient: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    convexity_shift: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return a valid gap even for accepted roundoff-scale indefiniteness."""

    if not np.any(convexity_shift):
        return _product_simplex_gap(gradient, coefficients)
    shifted_gradient = gradient + (2.0 * convexity_shift[:, None, None] * coefficients)
    convex_gap = _product_simplex_gap(shifted_gradient, coefficients)
    maximum_squared_norm = float(coefficients.shape[1])
    squared_norm = np.sum(
        coefficients * coefficients,
        axis=(1, 2),
        dtype=np.float64,
    )
    correction = convexity_shift * np.maximum(
        0.0,
        maximum_squared_norm - squared_norm,
    )
    return convex_gap + correction


def _exact_step(
    gap: float,
    curvature: float,
    maximum_step: float,
    *,
    numerical_tolerance: float,
) -> tuple[float, float]:
    """Return exact clipped line-search step and predicted objective change."""

    if gap <= numerical_tolerance or maximum_step <= numerical_tolerance:
        return 0.0, 0.0
    if curvature <= numerical_tolerance:
        step = maximum_step
    else:
        step = min(maximum_step, gap / (2.0 * curvature))
    change = -step * gap + step * step * curvature
    return float(step), float(change)


def _batched_exact_steps(
    gap: NDArray[np.float64],
    curvature: NDArray[np.float64],
    maximum_step: NDArray[np.float64],
    *,
    numerical_tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Vectorized equivalent of :func:`_exact_step` for independent rows."""

    if gap.shape != curvature.shape or gap.shape != maximum_step.shape:
        raise ValueError("batched line-search arrays have incompatible shapes")
    step = np.zeros_like(gap)
    eligible = (gap > numerical_tolerance) & (maximum_step > numerical_tolerance)
    boundary = eligible & (curvature <= numerical_tolerance)
    step[boundary] = maximum_step[boundary]
    interior = eligible & ~boundary
    step[interior] = np.minimum(
        maximum_step[interior],
        gap[interior] / (2.0 * curvature[interior]),
    )
    change = -step * gap + step * step * curvature
    return step, change


def solve_product_simplex_least_squares(
    gram: ArrayLike,
    linear: ArrayLike,
    target_energy: ArrayLike,
    initial_coefficients: ArrayLike | None = None,
    *,
    max_iterations: int = 10_000,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-11,
    symmetry_tolerance: float = 1e-10,
    psd_tolerance: float = 1e-10,
    feasibility_tolerance: float = 1e-10,
) -> ProductSimplexSolution:
    """Solve a batch of convex product-simplex quadratic problems.

    ``iterations`` counts complete deterministic head sweeps.  Convergence is
    certified row-by-row when the Frank-Wolfe gap is no greater than
    ``absolute_tolerance + relative_tolerance * max(1, |E|, |objective|)``.
    The returned gap remains useful even when ``max_iterations`` is reached.
    """

    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise ValueError("max_iterations must be a positive integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if (
        not np.isfinite(relative_tolerance)
        or relative_tolerance < 0.0
        or not np.isfinite(absolute_tolerance)
        or absolute_tolerance < 0.0
    ):
        raise ValueError("solver tolerances must be finite and nonnegative")
    matrix, vector, energy, coefficients, convexity_shift = _validate_problem(
        gram,
        linear,
        target_energy,
        initial_coefficients,
        symmetry_tolerance=symmetry_tolerance,
        psd_tolerance=psd_tolerance,
        feasibility_tolerance=feasibility_tolerance,
    )
    rows, heads, candidates = coefficients.shape
    width = heads * candidates
    flat_gram = matrix.reshape(rows, width, width)
    objective, gradient = _objective_and_gradient(
        matrix,
        vector,
        energy,
        coefficients,
    )
    gap = _certified_product_simplex_gap(
        gradient,
        coefficients,
        convexity_shift,
    )

    def thresholds() -> NDArray[np.float64]:
        return absolute_tolerance + relative_tolerance * np.maximum.reduce(
            (np.ones(rows), np.abs(energy), np.abs(objective))
        )

    row_converged = gap <= thresholds()
    completed_sweeps = 0
    numerical_tolerance = 128.0 * np.finfo(np.float64).eps
    for sweep in range(1, max_iterations + 1):
        if bool(np.all(row_converged)):
            break
        active_rows = np.flatnonzero(~row_converged)
        batch_indices = np.arange(active_rows.size)
        flat_gradient = gradient.reshape(rows, width)
        for head in range(heads):
            block_start = head * candidates
            block_stop = block_start + candidates
            block_x = coefficients[active_rows, head].copy()
            block_gradient = gradient[active_rows, head]
            block_gram = flat_gram[
                active_rows,
                block_start:block_stop,
                block_start:block_stop,
            ]

            # NumPy's argmin/argmax choose the lowest candidate index on exact
            # ties, matching the scalar product-simplex LMO.
            toward = np.argmin(block_gradient, axis=1)
            toward_direction = -block_x.copy()
            toward_direction[batch_indices, toward] += 1.0
            toward_gap = -np.einsum(
                "ni,ni->n",
                block_gradient,
                toward_direction,
                optimize=False,
            )
            toward_product = np.matmul(
                toward_direction[:, None, :],
                block_gram,
            )[:, 0, :]
            toward_curvature = np.einsum(
                "ni,ni->n",
                toward_product,
                toward_direction,
                optimize=False,
            )
            toward_step, toward_change = _batched_exact_steps(
                toward_gap,
                toward_curvature,
                np.ones(active_rows.size, dtype=np.float64),
                numerical_tolerance=numerical_tolerance,
            )

            active = block_x > feasibility_tolerance
            away_scores = np.where(active, block_gradient, -np.inf)
            away = np.argmax(away_scores, axis=1)
            pair_direction = np.zeros_like(block_x)
            pair_direction[batch_indices, toward] += 1.0
            pair_direction[batch_indices, away] -= 1.0
            pair_gap = (
                block_gradient[batch_indices, away]
                - block_gradient[batch_indices, toward]
            )
            pair_product = np.matmul(
                pair_direction[:, None, :],
                block_gram,
            )[:, 0, :]
            pair_curvature = np.einsum(
                "ni,ni->n",
                pair_product,
                pair_direction,
                optimize=False,
            )
            pair_step, pair_change = _batched_exact_steps(
                pair_gap,
                pair_curvature,
                block_x[batch_indices, away],
                numerical_tolerance=numerical_tolerance,
            )

            # Prefer the ordinary toward step on an exact predicted tie.
            use_pair = pair_change < toward_change
            direction = np.where(
                use_pair[:, None],
                pair_direction,
                toward_direction,
            )
            step = np.where(use_pair, pair_step, toward_step)
            delta = step[:, None] * direction
            block_x += delta
            block_x[np.abs(block_x) <= feasibility_tolerance] = 0.0
            if np.any(block_x < -feasibility_tolerance):
                raise RuntimeError("line search left the product simplex")
            np.maximum(block_x, 0.0, out=block_x)
            correction = 1.0 - np.sum(
                block_x,
                axis=1,
                dtype=np.float64,
            )
            repair = np.argmax(block_x, axis=1)
            block_x[batch_indices, repair] += correction
            if np.any(block_x[batch_indices, repair] < -feasibility_tolerance):
                raise RuntimeError("simplex roundoff repair failed")
            coefficients[active_rows, head] = block_x

            # Only later heads consume the incrementally maintained gradient;
            # the complete gradient is recomputed after the sweep.  Avoiding
            # writes for already-processed blocks nearly halves this update.
            if block_stop < width:
                gradient_update = (
                    2.0
                    * np.matmul(
                        flat_gram[
                            active_rows,
                            block_stop:,
                            block_start:block_stop,
                        ],
                        delta[..., None],
                    )[..., 0]
                )
                flat_gradient[active_rows, block_stop:] = (
                    flat_gradient[active_rows, block_stop:] + gradient_update
                )
        objective, gradient = _objective_and_gradient(
            matrix,
            vector,
            energy,
            coefficients,
        )
        gap = _certified_product_simplex_gap(
            gradient,
            coefficients,
            convexity_shift,
        )
        row_converged = gap <= thresholds()
        completed_sweeps = sweep

    sums = np.sum(coefficients, axis=2, dtype=np.float64)
    if (
        not np.isfinite(coefficients).all()
        or np.any(coefficients < -feasibility_tolerance)
        or np.any(np.abs(sums - 1.0) > 8.0 * feasibility_tolerance)
    ):
        raise RuntimeError("solver returned infeasible coefficients")
    relative_gap = _relative_gap(gap, objective, energy)
    return ProductSimplexSolution(
        coefficients=np.ascontiguousarray(coefficients),
        objective=np.ascontiguousarray(objective),
        objective_gap_upper_bound=np.ascontiguousarray(gap),
        relative_gap=np.ascontiguousarray(relative_gap),
        row_converged=np.ascontiguousarray(row_converged),
        iterations=completed_sweeps,
        converged=bool(np.all(row_converged)),
        max_relative_gap=float(np.max(relative_gap)),
    )
