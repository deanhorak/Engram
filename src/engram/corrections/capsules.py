from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrectionCapsule:
    center: np.ndarray
    radius: float
    down: np.ndarray
    up: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        down = np.asarray(self.down, dtype=np.float64)
        up = np.asarray(self.up, dtype=np.float64)
        if center.ndim != 1 or down.ndim != 2 or up.ndim != 2:
            raise ValueError("invalid capsule tensor ranks")
        if down.shape[0] != center.size or up.shape != (down.shape[1], center.size):
            raise ValueError("incompatible capsule dimensions")
        if not np.isfinite(self.radius) or self.radius < 0:
            raise ValueError("radius must be finite and nonnegative")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "down", down)
        object.__setattr__(self, "up", up)

    def distance(self, state: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(state) - self.center) / max(np.linalg.norm(self.center), 1.0))

    def apply(self, state: np.ndarray) -> np.ndarray:
        value = np.asarray(state, dtype=np.float64)
        return value + (value @ self.down) @ self.up


@dataclass(frozen=True)
class CorrectionDecision:
    capsule_index: int | None
    extra_cycles: int
    expand_semantic: bool
    expand_episodic: bool
    expand_vocabulary: bool


class CorrectionManager:
    def __init__(self, capsules: list[CorrectionCapsule], *, uncertainty_threshold: float = 0.5) -> None:
        self.capsules = list(capsules)
        self.uncertainty_threshold = float(uncertainty_threshold)

    def decide(self, state: np.ndarray, uncertainty: float) -> CorrectionDecision:
        matches = [(capsule.distance(state), index) for index, capsule in enumerate(self.capsules)]
        matches = [(distance, index) for distance, index in matches if distance <= self.capsules[index].radius]
        capsule_index = min(matches)[1] if matches else None
        difficult = uncertainty >= self.uncertainty_threshold
        return CorrectionDecision(capsule_index, int(difficult), difficult, difficult, difficult)

    def apply(self, state: np.ndarray, decision: CorrectionDecision) -> np.ndarray:
        return np.asarray(state) if decision.capsule_index is None else self.capsules[decision.capsule_index].apply(state)
