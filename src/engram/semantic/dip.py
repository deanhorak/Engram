from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.semantic.swiglu import silu


@dataclass(frozen=True)
class DIPTraffic:
    """Projected weight traffic for predictor-free dynamic input pruning.

    Counts are scalar weight elements. Byte counts use ``bytes_per_element``
    and exclude activations, indices, and kernel bookkeeping.
    """

    hidden_dim: int
    intermediate_dim: int
    input_count: int
    candidate_count: int
    top_k: int
    partial_projection_elements: int
    candidate_completion_elements: int
    selected_down_elements: int
    total_elements: int
    dense_elements: int
    total_bytes: int
    dense_bytes: int
    fraction_of_dense: float
    reduction_factor: float


@dataclass(frozen=True)
class DIPResult:
    """One predictor-free DIP selection with exact candidate completion.

    Proxy scores are evaluated for every intermediate record from only the
    selected input coordinates. Candidate activations and scores are then
    completed with the omitted coordinates before exact reranking. Full exact
    activations are retained only to expose oracle diagnostics; they are not
    part of the inference algorithm.
    """

    input_count: int
    input_indices: NDArray[np.int64]
    proxy_scores: NDArray[np.float64]
    candidate_indices: NDArray[np.int64]
    candidate_proxy_scores: NDArray[np.float64]
    candidate_activations: NDArray[np.float64]
    candidate_exact_scores: NDArray[np.float64]
    selected_indices: NDArray[np.int64]
    selected_activations: NDArray[np.float64]
    selected_exact_scores: NDArray[np.float64]
    exact_activations: NDArray[np.float64]
    exact_scores: NDArray[np.float64]
    oracle_indices: NDArray[np.int64]
    oracle_exact_scores: NDArray[np.float64]
    candidate_recall: float
    oracle_score_mass: float
    full_output: NDArray[np.float64] | None
    selected_output: NDArray[np.float64] | None
    output_relative_l2: float | None
    output_cosine: float | None


@dataclass(frozen=True)
class DIPProxyResult:
    """Partial SwiGLU proxy values and one reusable stable record ordering."""

    input_count: int
    input_indices: NDArray[np.int64]
    partial_gate: NDArray[np.float64]
    partial_up: NDArray[np.float64]
    proxy_activations: NDArray[np.float64]
    proxy_scores: NDArray[np.float64]
    order: NDArray[np.int64]


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _fraction(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.number)
    ):
        raise ValueError("input_fraction must be a finite number in (0, 1]")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0 or result > 1.0:
        raise ValueError("input_fraction must be a finite number in (0, 1]")
    return result


def input_coordinate_count(hidden_dim: int, input_fraction: float) -> int:
    """Return ``round(input_fraction * hidden_dim)`` after validation."""

    width = _positive_integer(hidden_dim, "hidden_dim")
    fraction = _fraction(input_fraction)
    return min(width, max(1, int(round(fraction * width))))


def projected_dip_traffic(
    hidden_dim: int,
    intermediate_dim: int,
    *,
    input_fraction: float,
    candidate_count: int,
    top_k: int,
    bytes_per_element: int = 4,
) -> DIPTraffic:
    """Project scalar weight reads for DIP with exact candidate completion.

    The accounting is
    ``2*I*q + 2*C*(H-q) + K*H`` versus dense ``3*I*H``, where ``q`` is
    ``round(input_fraction*H)``. It assumes a SwiGLU MLP whose output width is
    the hidden width.
    """

    hidden = _positive_integer(hidden_dim, "hidden_dim")
    intermediate = _positive_integer(intermediate_dim, "intermediate_dim")
    candidates = _positive_integer(candidate_count, "candidate_count")
    selected = _positive_integer(top_k, "top_k")
    element_bytes = _positive_integer(bytes_per_element, "bytes_per_element")
    if candidates > intermediate:
        raise ValueError("candidate_count must not exceed intermediate_dim")
    if selected > candidates:
        raise ValueError("top_k must not exceed candidate_count")

    input_count = input_coordinate_count(hidden, input_fraction)
    partial = 2 * intermediate * input_count
    completion = 2 * candidates * (hidden - input_count)
    down = selected * hidden
    total = partial + completion + down
    dense = 3 * intermediate * hidden
    fraction = total / dense
    return DIPTraffic(
        hidden_dim=hidden,
        intermediate_dim=intermediate,
        input_count=input_count,
        candidate_count=candidates,
        top_k=selected,
        partial_projection_elements=partial,
        candidate_completion_elements=completion,
        selected_down_elements=down,
        total_elements=total,
        dense_elements=dense,
        total_bytes=total * element_bytes,
        dense_bytes=dense * element_bytes,
        fraction_of_dense=fraction,
        reduction_factor=dense / total,
    )


def stable_top_k(scores: ArrayLike, count: int) -> NDArray[np.int64]:
    """Return a descending top-K with deterministic source-index tie breaking."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite one-dimensional array")
    if (
        isinstance(count, (bool, np.bool_))
        or not isinstance(count, (int, np.integer))
        or int(count) < 0
    ):
        raise ValueError("count must be a non-negative integer")
    selected = int(count)
    if selected > values.size:
        raise ValueError("count must not exceed the number of scores")
    return np.argsort(-values, kind="stable")[:selected].astype(np.int64, copy=False)


def partial_proxy_scores(
    hidden: ArrayLike,
    gate: ArrayLike,
    up: ArrayLike,
    value_norms: ArrayLike,
    *,
    input_fraction: float,
) -> DIPProxyResult:
    """Compute the predictor-free DIP proxy once for any number of C values."""

    hidden_array = np.asarray(hidden, dtype=np.float64)
    gate_array = np.asarray(gate, dtype=np.float64)
    up_array = np.asarray(up, dtype=np.float64)
    if hidden_array.ndim != 1:
        raise ValueError("hidden must have shape [H]")
    if gate_array.ndim != 2 or up_array.ndim != 2 or gate_array.shape != up_array.shape:
        raise ValueError("gate and up must have the same shape [I, H]")
    intermediate_dim, hidden_dim = gate_array.shape
    if intermediate_dim == 0 or hidden_dim == 0:
        raise ValueError("gate and up must not be empty")
    if hidden_array.shape != (hidden_dim,):
        raise ValueError(
            f"hidden must have shape [{hidden_dim}], got {hidden_array.shape}"
        )
    if not (
        np.all(np.isfinite(hidden_array))
        and np.all(np.isfinite(gate_array))
        and np.all(np.isfinite(up_array))
    ):
        raise ValueError("hidden, gate, and up must contain only finite values")
    norms = np.asarray(value_norms, dtype=np.float64)
    if norms.shape != (intermediate_dim,):
        raise ValueError(f"value_norms must have shape [{intermediate_dim}]")
    if not np.all(np.isfinite(norms)) or np.any(norms < 0.0):
        raise ValueError("value_norms must contain only finite non-negative values")

    input_count = input_coordinate_count(hidden_dim, input_fraction)
    input_indices = stable_top_k(np.abs(hidden_array), input_count)
    partial_gate = np.asarray(
        gate_array[:, input_indices] @ hidden_array[input_indices], dtype=np.float64
    )
    partial_up = np.asarray(
        up_array[:, input_indices] @ hidden_array[input_indices], dtype=np.float64
    )
    proxy_activations = np.asarray(silu(partial_gate) * partial_up, dtype=np.float64)
    proxy_scores = np.asarray(np.abs(proxy_activations) * norms, dtype=np.float64)
    return DIPProxyResult(
        input_count=input_count,
        input_indices=input_indices,
        partial_gate=partial_gate,
        partial_up=partial_up,
        proxy_activations=proxy_activations,
        proxy_scores=proxy_scores,
        order=stable_top_k(proxy_scores, intermediate_dim),
    )


def _similarity(
    approximation: np.ndarray, reference: np.ndarray
) -> tuple[float, float]:
    reference_norm = float(np.linalg.norm(reference))
    approximation_norm = float(np.linalg.norm(approximation))
    error_norm = float(np.linalg.norm(reference - approximation))
    if reference_norm <= 1e-12:
        relative_l2 = 0.0 if error_norm <= 1e-12 else float("inf")
    else:
        relative_l2 = error_norm / reference_norm
    if reference_norm <= 1e-12 and approximation_norm <= 1e-12:
        cosine = 1.0
    elif reference_norm <= 1e-12 or approximation_norm <= 1e-12:
        cosine = 0.0
    else:
        cosine = float(
            np.dot(approximation, reference) / (approximation_norm * reference_norm)
        )
    return relative_l2, max(-1.0, min(1.0, cosine))


def dynamic_input_pruning(
    hidden: ArrayLike,
    gate: ArrayLike,
    up: ArrayLike,
    *,
    input_fraction: float,
    candidate_count: int,
    top_k: int,
    down: ArrayLike | None = None,
    value_norms: ArrayLike | None = None,
) -> DIPResult:
    """Route one SwiGLU hidden state using predictor-free DIP.

    ``gate`` and ``up`` use shape ``[I, H]``. If supplied, ``down`` follows the
    project's linear-layer convention ``[H, I]``. Either ``down`` or explicit
    non-negative ``value_norms`` must be supplied so contribution magnitudes
    can be ranked. Supplying ``down`` additionally enables local MLP output
    relative-L2 and cosine diagnostics.
    """

    hidden_array = np.asarray(hidden, dtype=np.float64)
    gate_array = np.asarray(gate, dtype=np.float64)
    up_array = np.asarray(up, dtype=np.float64)
    if hidden_array.ndim != 1:
        raise ValueError("hidden must have shape [H]")
    if gate_array.ndim != 2 or up_array.ndim != 2 or gate_array.shape != up_array.shape:
        raise ValueError("gate and up must have the same shape [I, H]")
    intermediate_dim, hidden_dim = gate_array.shape
    if intermediate_dim == 0 or hidden_dim == 0:
        raise ValueError("gate and up must not be empty")
    if hidden_array.shape != (hidden_dim,):
        raise ValueError(
            f"hidden must have shape [{hidden_dim}], got {hidden_array.shape}"
        )
    if not (
        np.all(np.isfinite(hidden_array))
        and np.all(np.isfinite(gate_array))
        and np.all(np.isfinite(up_array))
    ):
        raise ValueError("hidden, gate, and up must contain only finite values")

    candidates = _positive_integer(candidate_count, "candidate_count")
    selected_count = _positive_integer(top_k, "top_k")
    if candidates > intermediate_dim:
        raise ValueError("candidate_count must not exceed the intermediate dimension")
    if selected_count > candidates:
        raise ValueError("top_k must not exceed candidate_count")
    down_array: NDArray[np.float64] | None
    if down is None:
        down_array = None
    else:
        down_array = np.asarray(down, dtype=np.float64)
        if down_array.shape != (hidden_dim, intermediate_dim):
            raise ValueError(
                f"down must have shape [{hidden_dim}, {intermediate_dim}], got {down_array.shape}"
            )
        if not np.all(np.isfinite(down_array)):
            raise ValueError("down must contain only finite values")

    if value_norms is None:
        if down_array is None:
            raise ValueError("either down or value_norms must be supplied")
        norms = np.linalg.norm(down_array, axis=0)
    else:
        norms = np.asarray(value_norms, dtype=np.float64)
        if norms.shape != (intermediate_dim,):
            raise ValueError(f"value_norms must have shape [{intermediate_dim}]")
        if not np.all(np.isfinite(norms)) or np.any(norms < 0.0):
            raise ValueError("value_norms must contain only finite non-negative values")

    proxy = partial_proxy_scores(
        hidden_array,
        gate_array,
        up_array,
        norms,
        input_fraction=input_fraction,
    )
    input_count = proxy.input_count
    input_indices = proxy.input_indices
    partial_gate = proxy.partial_gate
    partial_up = proxy.partial_up
    proxy_scores = proxy.proxy_scores
    candidate_indices = proxy.order[:candidates]

    omitted = np.ones(hidden_dim, dtype=bool)
    omitted[input_indices] = False
    omitted_indices = np.flatnonzero(omitted)
    candidate_gate = partial_gate[candidate_indices].copy()
    candidate_up = partial_up[candidate_indices].copy()
    if omitted_indices.size:
        candidate_gate += (
            gate_array[candidate_indices][:, omitted_indices]
            @ hidden_array[omitted_indices]
        )
        candidate_up += (
            up_array[candidate_indices][:, omitted_indices]
            @ hidden_array[omitted_indices]
        )
    candidate_activations = np.asarray(
        silu(candidate_gate) * candidate_up, dtype=np.float64
    )
    candidate_exact_scores = np.asarray(
        np.abs(candidate_activations) * norms[candidate_indices], dtype=np.float64
    )
    # Exact reranking must use the same record-index tie rule as a full-width
    # oracle, not the partial proxy order that happened to form the candidates.
    candidate_index_order = np.argsort(candidate_indices, kind="stable")
    local_by_index = stable_top_k(
        candidate_exact_scores[candidate_index_order], selected_count
    )
    local_order = candidate_index_order[local_by_index]
    selected_indices = candidate_indices[local_order]
    selected_activations = candidate_activations[local_order]
    selected_exact_scores = candidate_exact_scores[local_order]

    exact_gate = gate_array @ hidden_array
    exact_up = up_array @ hidden_array
    exact_activations = np.asarray(silu(exact_gate) * exact_up, dtype=np.float64)
    exact_scores = np.asarray(np.abs(exact_activations) * norms, dtype=np.float64)
    oracle_indices = stable_top_k(exact_scores, selected_count)
    oracle_exact_scores = exact_scores[oracle_indices]
    hits = np.intersect1d(candidate_indices, oracle_indices, assume_unique=True).size
    candidate_recall = float(hits / selected_count)
    oracle_total = float(np.sum(oracle_exact_scores))
    captured_mask = np.isin(oracle_indices, candidate_indices, assume_unique=True)
    captured = float(np.sum(oracle_exact_scores[captured_mask]))
    oracle_score_mass = 1.0 if oracle_total <= 1e-24 else captured / oracle_total

    full_output: NDArray[np.float64] | None = None
    selected_output: NDArray[np.float64] | None = None
    output_relative_l2: float | None = None
    output_cosine: float | None = None
    if down_array is not None:
        full_output = np.asarray(down_array @ exact_activations, dtype=np.float64)
        selected_output = np.asarray(
            down_array[:, selected_indices] @ selected_activations, dtype=np.float64
        )
        output_relative_l2, output_cosine = _similarity(selected_output, full_output)

    return DIPResult(
        input_count=input_count,
        input_indices=input_indices,
        proxy_scores=proxy_scores,
        candidate_indices=candidate_indices,
        candidate_proxy_scores=proxy_scores[candidate_indices],
        candidate_activations=candidate_activations,
        candidate_exact_scores=candidate_exact_scores,
        selected_indices=selected_indices,
        selected_activations=selected_activations,
        selected_exact_scores=selected_exact_scores,
        exact_activations=exact_activations,
        exact_scores=exact_scores,
        oracle_indices=oracle_indices,
        oracle_exact_scores=oracle_exact_scores,
        candidate_recall=candidate_recall,
        oracle_score_mass=oracle_score_mass,
        full_output=full_output,
        selected_output=selected_output,
        output_relative_l2=output_relative_l2,
        output_cosine=output_cosine,
    )
