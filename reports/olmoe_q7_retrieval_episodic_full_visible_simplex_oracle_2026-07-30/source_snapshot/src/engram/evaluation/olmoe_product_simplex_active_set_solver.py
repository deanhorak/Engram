"""Certified bulk active-set solver for product-simplex quadratics.

The feasible set is a Cartesian product of probability simplices.  For a
fixed support, the minimizer is obtained from the equality-constrained KKT
system

    [G_SS  B_S.T] [x_S] = [b_S]
    [B_S       0] [ nu]   [  1],

where ``B`` sums the candidates in each head.  A deterministic bulk working
set update removes every negative free coefficient or admits every inactive
candidate with a negative reduced cost.  This is an acceleration heuristic,
not the source of the optimality claim: every accepted result is feasible and
is certified afterward with the ordinary full product-simplex Frank--Wolfe
gap.  Singular, inaccurate, cycling, or unfinished working-set solves fall
back to the reference pairwise block Frank--Wolfe solver.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.evaluation import olmoe_product_simplex_solver as reference
from engram.evaluation.olmoe_product_simplex_solver import ProductSimplexSolution


def _thresholds(
    objective: NDArray[np.float64],
    target_energy: NDArray[np.float64],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> NDArray[np.float64]:
    return absolute_tolerance + relative_tolerance * np.maximum.reduce(
        (
            np.ones_like(objective),
            np.abs(target_energy),
            np.abs(objective),
        )
    )


def _solve_support_kkt(
    gram: NDArray[np.float64],
    linear: NDArray[np.float64],
    support: NDArray[np.bool_],
    *,
    heads: int,
    candidates: int,
    residual_tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Solve one restricted KKT system, failing closed on singularity/error."""

    indices = np.flatnonzero(support)
    support_heads = indices // candidates
    equality = np.zeros((heads, indices.size), dtype=np.float64)
    equality[support_heads, np.arange(indices.size)] = 1.0
    restricted = gram[np.ix_(indices, indices)]
    kkt = np.empty(
        (indices.size + heads, indices.size + heads),
        dtype=np.float64,
    )
    kkt[: indices.size, : indices.size] = restricted
    kkt[: indices.size, indices.size :] = equality.T
    kkt[indices.size :, : indices.size] = equality
    kkt[indices.size :, indices.size :] = 0.0
    right_hand_side = np.concatenate(
        (linear[indices], np.ones(heads, dtype=np.float64))
    )
    try:
        solved = np.linalg.solve(kkt, right_hand_side)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(solved).all():
        return None
    residual = kkt @ solved - right_hand_side
    residual_scale = max(
        1.0,
        float(np.max(np.abs(right_hand_side))),
        float(
            np.max(np.sum(np.abs(kkt), axis=1))
            * np.max(np.abs(solved))
        ),
    )
    if float(np.max(np.abs(residual))) > residual_tolerance * residual_scale:
        return None
    coefficients = np.zeros(heads * candidates, dtype=np.float64)
    coefficients[indices] = solved[: indices.size]
    return coefficients, solved[indices.size :]


def _retain_one_candidate_per_head(
    support: NDArray[np.bool_],
    remove: NDArray[np.bool_],
    coefficients: NDArray[np.float64],
    *,
    heads: int,
    candidates: int,
) -> None:
    """Modify ``remove`` so a support never loses an entire simplex block."""

    for head in range(heads):
        begin = head * candidates
        end = begin + candidates
        retained = support[begin:end] & ~remove[begin:end]
        if np.any(retained):
            continue
        active = np.flatnonzero(support[begin:end])
        # np.argmax returns the lowest local index on an exact tie.
        keep = active[np.argmax(coefficients[begin:end][active])]
        remove[begin + keep] = False


def _repaired_feasible_candidate(
    coefficients: NDArray[np.float64],
    *,
    heads: int,
    candidates: int,
    feasibility_tolerance: float,
) -> NDArray[np.float64] | None:
    """Return a roundoff-repaired feasible point or ``None``."""

    shaped = coefficients.reshape(heads, candidates).copy()
    if np.any(shaped < -feasibility_tolerance):
        return None
    np.maximum(shaped, 0.0, out=shaped)
    sums = np.sum(shaped, axis=1, keepdims=True, dtype=np.float64)
    if np.any(sums <= 0.0):
        return None
    shaped /= sums
    if (
        not np.isfinite(shaped).all()
        or np.any(shaped < 0.0)
        or np.max(np.abs(np.sum(shaped, axis=1) - 1.0))
        > 8.0 * feasibility_tolerance
    ):
        return None
    return np.ascontiguousarray(shaped)


def _bulk_active_set_row(
    gram: NDArray[np.float64],
    linear: NDArray[np.float64],
    target_energy: float,
    initial: NDArray[np.float64],
    convexity_shift: float,
    *,
    maximum_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    feasibility_tolerance: float,
    working_set_tolerance: float,
    kkt_residual_tolerance: float,
    reduced_cost_tolerance: float,
) -> tuple[NDArray[np.float64], int, bool]:
    """Try the deterministic bulk support heuristic for one batch row."""

    heads, candidates = initial.shape
    width = heads * candidates
    support = np.ones(width, dtype=bool)
    support_history: set[bytes] = set()
    bulk_updates = True
    best = np.ascontiguousarray(initial.copy())
    best_objective, best_gradient = reference._objective_and_gradient(
        gram.reshape(1, heads, candidates, heads, candidates),
        linear.reshape(1, heads, candidates),
        np.asarray([target_energy], dtype=np.float64),
        best.reshape(1, heads, candidates),
    )
    best_gap = reference._certified_product_simplex_gap(
        best_gradient,
        best.reshape(1, heads, candidates),
        np.asarray([convexity_shift], dtype=np.float64),
    )
    threshold = _thresholds(
        best_objective,
        np.asarray([target_energy], dtype=np.float64),
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )[0]
    if best_gap[0] <= threshold:
        return best, 0, True

    flat_gram = gram.reshape(width, width)
    flat_linear = linear.reshape(width)
    for iteration in range(1, maximum_iterations + 1):
        support_key = support.tobytes()
        if support_key in support_history:
            # Bulk admission/removal can exceptionally revisit a support.
            # Switch permanently to deterministic one-coordinate refinements;
            # the hard iteration bound and certified reference fallback remain
            # the fail-safe if that refinement also cycles.
            bulk_updates = False
        else:
            support_history.add(support_key)
        solved = _solve_support_kkt(
            flat_gram,
            flat_linear,
            support,
            heads=heads,
            candidates=candidates,
            residual_tolerance=kkt_residual_tolerance,
        )
        if solved is None:
            return best, iteration - 1, False
        unrestricted, _multipliers = solved
        # Support membership needs a tighter threshold than final feasibility
        # repair.  Otherwise a small but genuinely negative free coefficient
        # can remain in the support, get clipped, and obscure its KKT signal.
        negative = support & (unrestricted < -working_set_tolerance)
        if np.any(negative):
            remove = negative.copy()
            _retain_one_candidate_per_head(
                support,
                remove,
                unrestricted,
                heads=heads,
                candidates=candidates,
            )
            if not bulk_updates:
                removable = np.flatnonzero(remove)
                keep_removing = removable[np.argmin(unrestricted[removable])]
                remove[:] = False
                remove[keep_removing] = True
            updated = support & ~remove
            if np.array_equal(updated, support):
                return best, iteration, False
            support = updated
            continue

        candidate = _repaired_feasible_candidate(
            unrestricted,
            heads=heads,
            candidates=candidates,
            feasibility_tolerance=feasibility_tolerance,
        )
        if candidate is None:
            return best, iteration, False
        objective, gradient = reference._objective_and_gradient(
            gram.reshape(1, heads, candidates, heads, candidates),
            linear.reshape(1, heads, candidates),
            np.asarray([target_energy], dtype=np.float64),
            candidate.reshape(1, heads, candidates),
        )
        gap = reference._certified_product_simplex_gap(
            gradient,
            candidate.reshape(1, heads, candidates),
            np.asarray([convexity_shift], dtype=np.float64),
        )
        if objective[0] < best_objective[0]:
            best = candidate
            best_objective = objective
            best_gap = gap
        threshold = _thresholds(
            objective,
            np.asarray([target_energy], dtype=np.float64),
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )[0]
        if gap[0] <= threshold:
            return candidate, iteration, True

        # The support solve makes q = Gx-b constant on every retained head
        # up to roundoff.  Compare inactive q directly with its head's
        # coefficient-weighted q; this is also the per-head FW gap baseline.
        half_gradient = gradient[0] * 0.5
        baseline = np.sum(
            half_gradient * candidate,
            axis=1,
            dtype=np.float64,
        )
        scale = np.maximum(
            1.0,
            np.max(np.abs(half_gradient), axis=1),
        )
        inactive = ~support.reshape(heads, candidates)
        violations = inactive & (
            half_gradient
            < baseline[:, None] - reduced_cost_tolerance * scale[:, None]
        )
        if not np.any(violations):
            return best, iteration, False
        if bulk_updates:
            support |= violations.reshape(width)
        else:
            scores = np.where(
                violations,
                half_gradient - baseline[:, None],
                np.inf,
            )
            support[np.argmin(scores)] = True
    return best, maximum_iterations, False


def solve_product_simplex_least_squares_active_set(
    gram: ArrayLike,
    linear: ArrayLike,
    target_energy: ArrayLike,
    initial_coefficients: ArrayLike | None = None,
    *,
    max_active_set_iterations: int = 128,
    fallback_max_iterations: int = 512,
    relative_tolerance: float = 1e-7,
    absolute_tolerance: float = 1e-11,
    symmetry_tolerance: float = 1e-10,
    psd_tolerance: float = 1e-10,
    feasibility_tolerance: float = 1e-10,
    working_set_tolerance: float = 1e-12,
    kkt_residual_tolerance: float = 1e-10,
    reduced_cost_tolerance: float = 1e-12,
) -> ProductSimplexSolution:
    """Solve a batch with a certified active-set fast path and FW fallback.

    The return type and certificate fields are identical to the reference
    solver.  ``iterations`` is the largest number of active-set refinements
    used by a row, plus the fallback solver's sweep count when fallback was
    required.
    """

    for value, name in (
        (max_active_set_iterations, "max_active_set_iterations"),
        (fallback_max_iterations, "fallback_max_iterations"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for value, name in (
        (relative_tolerance, "relative_tolerance"),
        (absolute_tolerance, "absolute_tolerance"),
        (working_set_tolerance, "working_set_tolerance"),
        (kkt_residual_tolerance, "kkt_residual_tolerance"),
        (reduced_cost_tolerance, "reduced_cost_tolerance"),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if working_set_tolerance > feasibility_tolerance:
        raise ValueError(
            "working_set_tolerance must not exceed feasibility_tolerance"
        )

    matrix, vector, energy, coefficients, convexity_shift = (
        reference._validate_problem(
            gram,
            linear,
            target_energy,
            initial_coefficients,
            symmetry_tolerance=symmetry_tolerance,
            psd_tolerance=psd_tolerance,
            feasibility_tolerance=feasibility_tolerance,
        )
    )
    rows = coefficients.shape[0]
    active_iterations = np.zeros(rows, dtype=np.int64)
    active_certified = np.zeros(rows, dtype=bool)
    for row in range(rows):
        row_coefficients, iterations, certified = _bulk_active_set_row(
            matrix[row],
            vector[row],
            float(energy[row]),
            coefficients[row],
            float(convexity_shift[row]),
            maximum_iterations=max_active_set_iterations,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            feasibility_tolerance=feasibility_tolerance,
            working_set_tolerance=working_set_tolerance,
            kkt_residual_tolerance=kkt_residual_tolerance,
            reduced_cost_tolerance=reduced_cost_tolerance,
        )
        coefficients[row] = row_coefficients
        active_iterations[row] = iterations
        active_certified[row] = certified

    fallback_iterations = 0
    unresolved = np.flatnonzero(~active_certified)
    if unresolved.size:
        fallback = reference.solve_product_simplex_least_squares(
            matrix[unresolved],
            vector[unresolved],
            energy[unresolved],
            coefficients[unresolved],
            max_iterations=fallback_max_iterations,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            symmetry_tolerance=symmetry_tolerance,
            psd_tolerance=psd_tolerance,
            feasibility_tolerance=feasibility_tolerance,
        )
        coefficients[unresolved] = fallback.coefficients
        fallback_iterations = fallback.iterations

    objective, gradient = reference._objective_and_gradient(
        matrix,
        vector,
        energy,
        coefficients,
    )
    gap = reference._certified_product_simplex_gap(
        gradient,
        coefficients,
        convexity_shift,
    )
    row_converged = gap <= _thresholds(
        objective,
        energy,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    sums = np.sum(coefficients, axis=2, dtype=np.float64)
    if (
        not np.isfinite(coefficients).all()
        or np.any(coefficients < -feasibility_tolerance)
        or np.any(np.abs(sums - 1.0) > 8.0 * feasibility_tolerance)
    ):
        raise RuntimeError("active-set solver returned infeasible coefficients")
    relative_gap = reference._relative_gap(gap, objective, energy)
    return ProductSimplexSolution(
        coefficients=np.ascontiguousarray(coefficients),
        objective=np.ascontiguousarray(objective),
        objective_gap_upper_bound=np.ascontiguousarray(gap),
        relative_gap=np.ascontiguousarray(relative_gap),
        row_converged=np.ascontiguousarray(row_converged),
        iterations=int(np.max(active_iterations)) + fallback_iterations,
        converged=bool(np.all(row_converged)),
        max_relative_gap=float(np.max(relative_gap)),
    )
