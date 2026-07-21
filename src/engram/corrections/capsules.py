from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrectionCapsule:
    center: np.ndarray
    radius: float
    down: np.ndarray
    up: np.ndarray
    bias: np.ndarray | None = None
    centered_input: bool = False

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        down = np.asarray(self.down, dtype=np.float64)
        up = np.asarray(self.up, dtype=np.float64)
        bias = np.zeros(center.size, dtype=np.float64) if self.bias is None else np.asarray(
            self.bias, dtype=np.float64
        )
        if center.ndim != 1 or down.ndim != 2 or up.ndim != 2:
            raise ValueError("invalid capsule tensor ranks")
        if down.shape[0] != center.size or up.shape != (down.shape[1], center.size):
            raise ValueError("incompatible capsule dimensions")
        if bias.shape != center.shape:
            raise ValueError("capsule bias must match the center dimension")
        if not np.isfinite(self.radius) or self.radius < 0:
            raise ValueError("radius must be finite and nonnegative")
        if not all(np.all(np.isfinite(value)) for value in (center, down, up, bias)):
            raise ValueError("capsule tensors must be finite")
        if not isinstance(self.centered_input, (bool, np.bool_)):
            raise ValueError("centered_input must be boolean")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "down", down)
        object.__setattr__(self, "up", up)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "centered_input", bool(self.centered_input))

    def distance(self, state: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(state) - self.center) / max(np.linalg.norm(self.center), 1.0))

    def correction(self, state: np.ndarray) -> np.ndarray:
        value = np.asarray(state, dtype=np.float64)
        features = value - self.center if self.centered_input else value
        return (features @ self.down) @ self.up + self.bias

    def apply(self, state: np.ndarray) -> np.ndarray:
        value = np.asarray(state, dtype=np.float64)
        return value + self.correction(value)

    def parameter_bytes(self, *, bytes_per_parameter: int = 4) -> int:
        if not isinstance(bytes_per_parameter, int) or bytes_per_parameter <= 0:
            raise ValueError("bytes_per_parameter must be positive")
        return int(
            (self.center.size + self.down.size + self.up.size + self.bias.size)
            * bytes_per_parameter
        )


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

    def correction(self, state: np.ndarray, decision: CorrectionDecision) -> np.ndarray:
        value = np.asarray(state, dtype=np.float64)
        if decision.capsule_index is None:
            return np.zeros_like(value)
        return self.capsules[decision.capsule_index].correction(value)

    def apply_to(
        self,
        value: np.ndarray,
        state: np.ndarray,
        decision: CorrectionDecision,
    ) -> np.ndarray:
        return np.asarray(value) + self.correction(state, decision)
