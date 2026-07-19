from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def silu(x: NDArray[np.floating]) -> NDArray[np.floating]:
    # exp(-abs(x)) is bounded by one, so both np.where branches are safe to evaluate.
    exponential = np.exp(-np.abs(x))
    sigmoid = np.where(x >= 0, 1.0 / (1.0 + exponential), exponential / (1.0 + exponential))
    return x * sigmoid


def _validate(
    hidden: ArrayLike, gate: ArrayLike, up: ArrayLike, down: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = np.asarray(hidden)
    wg = np.asarray(gate)
    wu = np.asarray(up)
    wd = np.asarray(down)
    if h.ndim < 1 or wg.ndim != 2 or wu.ndim != 2 or wd.ndim != 2:
        raise ValueError("hidden must be [..., H] and weights must be rank-2")
    if wg.shape != wu.shape:
        raise ValueError(f"gate/up shapes differ: {wg.shape} versus {wu.shape}")
    intermediate, width = wg.shape
    if h.shape[-1] != width or wd.shape != (width, intermediate):
        raise ValueError(
            f"incompatible shapes hidden={h.shape}, gate={wg.shape}, up={wu.shape}, down={wd.shape}"
        )
    return h, wg, wu, wd


def neuron_activations(hidden: ArrayLike, gate: ArrayLike, up: ArrayLike) -> np.ndarray:
    h = np.asarray(hidden)
    wg = np.asarray(gate)
    wu = np.asarray(up)
    if wg.shape != wu.shape or wg.ndim != 2 or h.shape[-1] != wg.shape[1]:
        raise ValueError(f"incompatible shapes hidden={h.shape}, gate={wg.shape}, up={wu.shape}")
    gate_logits = h @ wg.T
    up_values = h @ wu.T
    return silu(gate_logits) * up_values


def swiglu(hidden: ArrayLike, gate: ArrayLike, up: ArrayLike, down: ArrayLike) -> np.ndarray:
    h, wg, wu, wd = _validate(hidden, gate, up, down)
    return neuron_activations(h, wg, wu) @ wd.T


def neuron_contributions(
    hidden: ArrayLike, gate: ArrayLike, up: ArrayLike, down: ArrayLike
) -> np.ndarray:
    h, wg, wu, wd = _validate(hidden, gate, up, down)
    activations = neuron_activations(h, wg, wu)
    return activations[..., :, None] * wd.T


def swiglu_decomposed(
    hidden: ArrayLike, gate: ArrayLike, up: ArrayLike, down: ArrayLike
) -> np.ndarray:
    return neuron_contributions(hidden, gate, up, down).sum(axis=-2)
