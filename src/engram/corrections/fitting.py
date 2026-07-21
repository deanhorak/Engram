"""Deterministic fitting for state-selected low-rank correction capsules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.semantic.background import LowRankLinearBackground

from .capsules import CorrectionCapsule, CorrectionDecision, CorrectionManager


def _matrix(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class FittedCorrectionCapsules:
    manager: CorrectionManager
    assignments: NDArray[np.int64]
    training_residual_norm: NDArray[np.float64]
    rank: int
    ridge: float

    def predict(self, inputs: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        states = _matrix(inputs, "inputs")
        corrections = np.zeros_like(states)
        matched = np.zeros(len(states), dtype=bool)
        for index, state in enumerate(states):
            decision = self.manager.decide(state, uncertainty=0.0)
            if decision.capsule_index is not None:
                matched[index] = True
                corrections[index] = self.manager.correction(state, decision)
        return corrections, matched

    def parameter_bytes(self, *, bytes_per_parameter: int = 4) -> int:
        return sum(
            capsule.parameter_bytes(bytes_per_parameter=bytes_per_parameter)
            for capsule in self.manager.capsules
        )


def fit_correction_capsules(
    inputs: ArrayLike,
    residuals: ArrayLike,
    *,
    capsules: int,
    rank: int,
    ridge: float = 1.0,
    iterations: int = 8,
    radius_scale: float = 1.25,
    priority_fraction: float = 1.0,
    radius_quantile: float = 1.0,
) -> FittedCorrectionCapsules:
    """Fit local affine low-rank predictors seeded by the largest residuals."""

    states = _matrix(inputs, "inputs")
    targets = _matrix(residuals, "residuals")
    if states.shape != targets.shape:
        raise ValueError("inputs and residuals must have the same shape")
    if not isinstance(capsules, int) or not 0 < capsules <= len(states):
        raise ValueError("capsules must lie within the sample count")
    if not isinstance(rank, int) or not 0 < rank <= states.shape[1]:
        raise ValueError("rank must lie within the state width")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be positive")
    if not np.isfinite(radius_scale) or radius_scale <= 0.0:
        raise ValueError("radius_scale must be finite and positive")
    if not np.isfinite(priority_fraction) or not 0.0 < priority_fraction <= 1.0:
        raise ValueError("priority_fraction must lie in (0, 1]")
    if not np.isfinite(radius_quantile) or not 0.0 < radius_quantile <= 1.0:
        raise ValueError("radius_quantile must lie in (0, 1]")

    residual_norm = np.linalg.norm(targets, axis=1)
    priority_count = max(capsules, int(np.ceil(len(states) * priority_fraction)))
    priority_ids = np.argsort(-residual_norm, kind="stable")[:priority_count]
    training_states = states[priority_ids]
    training_targets = targets[priority_ids]
    training_norm = residual_norm[priority_ids]
    scale = np.std(states, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (training_states - np.mean(states, axis=0)) / scale
    selected = [int(np.argmax(training_norm))]
    nearest = np.sum((standardized - standardized[selected[0]]) ** 2, axis=1)
    priority = training_norm / max(float(np.mean(training_norm)), 1e-12)
    while len(selected) < capsules:
        score = nearest * (1.0 + priority)
        score[np.asarray(selected)] = -1.0
        chosen = int(np.argmax(score))
        selected.append(chosen)
        nearest = np.minimum(
            nearest,
            np.sum((standardized - standardized[chosen]) ** 2, axis=1),
        )
    centers = standardized[np.asarray(selected)].copy()
    local_assignments = np.zeros(len(training_states), dtype=np.int64)
    for _ in range(iterations):
        distances = np.sum(
            (standardized[:, None, :] - centers[None, :, :]) ** 2, axis=2
        )
        updated = np.argmin(distances, axis=1).astype(np.int64)
        if np.array_equal(updated, local_assignments) and _ > 0:
            break
        local_assignments = updated
        for capsule_index in range(capsules):
            members = np.flatnonzero(local_assignments == capsule_index)
            if not members.size:
                candidate = int(np.argmax(np.min(distances, axis=1) * (1.0 + priority)))
                local_assignments[candidate] = capsule_index
                members = np.asarray([candidate])
            weights = 1.0 + priority[members]
            centers[capsule_index] = np.average(
                standardized[members], axis=0, weights=weights
            )

    fitted: list[CorrectionCapsule] = []
    for capsule_index in range(capsules):
        members = np.flatnonzero(local_assignments == capsule_index)
        local_rank = min(
            rank,
            training_states.shape[1],
            training_targets.shape[1],
            max(1, len(members) - 1),
        )
        background = LowRankLinearBackground.fit(
            training_states[members],
            training_targets[members],
            rank=local_rank,
            ridge=ridge,
        )
        center = background.input_mean
        distances = np.linalg.norm(training_states[members] - center, axis=1) / max(
            float(np.linalg.norm(center)), 1.0
        )
        radius = max(float(np.quantile(distances, radius_quantile)) * radius_scale, 1e-12)
        fitted.append(
            CorrectionCapsule(
                center=center,
                radius=radius,
                down=background.input_factor,
                up=background.output_factor,
                bias=background.output_mean,
                centered_input=True,
            )
        )
    assignments = np.full(len(states), -1, dtype=np.int64)
    assignments[priority_ids] = local_assignments
    return FittedCorrectionCapsules(
        manager=CorrectionManager(fitted),
        assignments=assignments,
        training_residual_norm=residual_norm,
        rank=rank,
        ridge=float(ridge),
    )


__all__ = ["FittedCorrectionCapsules", "fit_correction_capsules"]
