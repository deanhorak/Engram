"""NumPy reference runtime for a shared recurrent Engram controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


CONTROLLER_FORMAT = "engram.controller.shared_gru"
CONTROLLER_SCHEMA_VERSION = 1


def _sigmoid(value: np.ndarray) -> np.ndarray:
    exponential = np.exp(-np.abs(value))
    return np.where(value >= 0.0, 1.0 / (1.0 + exponential), exponential / (1.0 + exponential))


def _finite_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class ControllerResult:
    state: NDArray[np.float64]
    cycles: int
    residual: float
    residual_history: tuple[float, ...]
    confidence_history: tuple[float, ...]
    converged: bool
    requested_extra_cycles: int
    cycle_limit: int


@dataclass(frozen=True)
class SharedRecurrentController:
    """GRU-like transition whose base weights are reused for every cycle.

    Stage identity is represented by a small candidate-state embedding and,
    optionally, a low-rank state adapter.  The large input and recurrent kernels
    have no cycle or stage axis and are therefore shared exactly.
    """

    input_kernel: NDArray[np.float64]
    recurrent_kernel: NDArray[np.float64]
    bias: NDArray[np.float64]
    stage_embeddings: NDArray[np.float64]
    adapter_down: NDArray[np.float64]
    adapter_up: NDArray[np.float64]

    def __post_init__(self) -> None:
        input_kernel = _finite_array(self.input_kernel, "input_kernel")
        recurrent_kernel = _finite_array(self.recurrent_kernel, "recurrent_kernel")
        bias = _finite_array(self.bias, "bias")
        stage_embeddings = _finite_array(self.stage_embeddings, "stage_embeddings")
        adapter_down = _finite_array(self.adapter_down, "adapter_down")
        adapter_up = _finite_array(self.adapter_up, "adapter_up")

        if input_kernel.ndim != 2 or recurrent_kernel.ndim != 2:
            raise ValueError("controller kernels must be rank-2")
        if recurrent_kernel.shape[1] % 3 != 0:
            raise ValueError("recurrent kernel width must be three times the state dimension")
        state_dim = recurrent_kernel.shape[0]
        if recurrent_kernel.shape[1] != 3 * state_dim:
            raise ValueError("recurrent kernel must have shape [state_dim, 3 * state_dim]")
        if input_kernel.shape[1] != 3 * state_dim:
            raise ValueError("input kernel must have shape [input_dim, 3 * state_dim]")
        if bias.shape != (3 * state_dim,):
            raise ValueError("bias must have shape [3 * state_dim]")
        if stage_embeddings.ndim != 2 or stage_embeddings.shape[1] != state_dim:
            raise ValueError("stage embeddings must have shape [num_stages, state_dim]")
        if stage_embeddings.shape[0] == 0:
            raise ValueError("at least one stage embedding is required")
        if adapter_down.ndim != 3 or adapter_up.ndim != 3:
            raise ValueError("stage adapters must be rank-3")
        rank = adapter_down.shape[2]
        if adapter_down.shape != (stage_embeddings.shape[0], state_dim, rank):
            raise ValueError("adapter_down has incompatible dimensions")
        if adapter_up.shape != (stage_embeddings.shape[0], rank, state_dim):
            raise ValueError("adapter_up has incompatible dimensions")

        for value in (
            input_kernel,
            recurrent_kernel,
            bias,
            stage_embeddings,
            adapter_down,
            adapter_up,
        ):
            value.setflags(write=False)
        object.__setattr__(self, "input_kernel", input_kernel)
        object.__setattr__(self, "recurrent_kernel", recurrent_kernel)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "stage_embeddings", stage_embeddings)
        object.__setattr__(self, "adapter_down", adapter_down)
        object.__setattr__(self, "adapter_up", adapter_up)

    @property
    def input_dim(self) -> int:
        return int(self.input_kernel.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.recurrent_kernel.shape[0])

    @property
    def num_stages(self) -> int:
        return int(self.stage_embeddings.shape[0])

    @property
    def adapter_rank(self) -> int:
        return int(self.adapter_down.shape[2])

    @classmethod
    def initialize(
        cls,
        *,
        input_dim: int,
        state_dim: int,
        num_stages: int,
        adapter_rank: int = 0,
        seed: int = 0,
        weight_scale: float | None = None,
    ) -> "SharedRecurrentController":
        """Create deterministic small random weights for fixtures and fitting."""

        dimensions = (input_dim, state_dim, num_stages)
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in dimensions):
            raise ValueError("controller dimensions must be integers")
        if input_dim <= 0 or state_dim <= 0 or num_stages <= 0:
            raise ValueError("controller dimensions must be positive")
        if (
            isinstance(adapter_rank, bool)
            or not isinstance(adapter_rank, (int, np.integer))
            or adapter_rank < 0
            or adapter_rank > state_dim
        ):
            raise ValueError("adapter_rank must lie in [0, state_dim]")
        scale = 1.0 / math.sqrt(max(input_dim, state_dim)) if weight_scale is None else float(weight_scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("weight_scale must be finite and non-negative")
        rng = np.random.default_rng(seed)
        return cls(
            input_kernel=rng.normal(scale=scale, size=(input_dim, 3 * state_dim)),
            recurrent_kernel=rng.normal(scale=scale, size=(state_dim, 3 * state_dim)),
            bias=np.zeros(3 * state_dim, dtype=np.float64),
            stage_embeddings=rng.normal(scale=scale, size=(num_stages, state_dim)),
            adapter_down=rng.normal(scale=scale, size=(num_stages, state_dim, adapter_rank)),
            adapter_up=rng.normal(scale=scale, size=(num_stages, adapter_rank, state_dim)),
        )

    def step(self, state: ArrayLike, controller_input: ArrayLike, *, stage: int) -> np.ndarray:
        current = _finite_array(state, "state")
        supplied = _finite_array(controller_input, "controller_input")
        if current.ndim < 1 or current.shape[-1] != self.state_dim:
            raise ValueError(f"state must have trailing dimension {self.state_dim}")
        if supplied.ndim < 1 or supplied.shape[-1] != self.input_dim:
            raise ValueError(f"controller_input must have trailing dimension {self.input_dim}")
        if supplied.shape[:-1] != current.shape[:-1]:
            raise ValueError("state and controller_input leading dimensions must match")
        if isinstance(stage, bool) or not isinstance(stage, (int, np.integer)):
            raise ValueError("stage must be an integer")
        if stage < 0 or stage >= self.num_stages:
            raise ValueError(f"stage must lie in [0, {self.num_stages - 1}]")

        state_dim = self.state_dim
        input_projection = supplied @ self.input_kernel
        update = _sigmoid(
            input_projection[..., :state_dim]
            + current @ self.recurrent_kernel[:, :state_dim]
            + self.bias[:state_dim]
        )
        reset = _sigmoid(
            input_projection[..., state_dim : 2 * state_dim]
            + current @ self.recurrent_kernel[:, state_dim : 2 * state_dim]
            + self.bias[state_dim : 2 * state_dim]
        )
        candidate_pre = (
            input_projection[..., 2 * state_dim :]
            + (reset * current) @ self.recurrent_kernel[:, 2 * state_dim :]
            + self.bias[2 * state_dim :]
            + self.stage_embeddings[stage]
        )
        if self.adapter_rank:
            candidate_pre += (current @ self.adapter_down[stage]) @ self.adapter_up[stage]
        candidate = np.tanh(candidate_pre)
        return (1.0 - update) * current + update * candidate

    def run(
        self,
        initial_state: ArrayLike,
        controller_input: ArrayLike,
        *,
        mode: str = "fixed",
        fixed_cycles: int = 1,
        min_cycles: int = 1,
        max_cycles: int = 8,
        residual_tolerance: float = 1e-3,
        confidence_threshold: float | None = None,
        confidence: float | None = None,
        requested_extra_cycles: int = 0,
        stage_offset: int = 0,
    ) -> ControllerResult:
        """Apply shared weights repeatedly under a fixed or adaptive policy.

        Adaptive mode exits after ``min_cycles`` when the state-update RMS is at
        most ``residual_tolerance`` and confidence is sufficient.  If no external
        confidence is supplied, ``1 / (1 + residual)`` is used as a deterministic
        proxy.  An extra-cycle request extends the hard cycle budget.
        """

        if mode not in {"fixed", "adaptive"}:
            raise ValueError("mode must be 'fixed' or 'adaptive'")
        integer_options = {
            "fixed_cycles": fixed_cycles,
            "min_cycles": min_cycles,
            "max_cycles": max_cycles,
            "requested_extra_cycles": requested_extra_cycles,
            "stage_offset": stage_offset,
        }
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in integer_options.values()):
            raise ValueError("cycle counts and stage_offset must be integers")
        if fixed_cycles <= 0 or min_cycles <= 0 or max_cycles < min_cycles:
            raise ValueError("cycle counts must be positive and max_cycles >= min_cycles")
        if requested_extra_cycles < 0:
            raise ValueError("requested_extra_cycles must be non-negative")
        if not math.isfinite(residual_tolerance) or residual_tolerance < 0.0:
            raise ValueError("residual_tolerance must be finite and non-negative")
        if confidence_threshold is not None and not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must lie in [0, 1]")
        if confidence is not None and (not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0):
            raise ValueError("confidence must lie in [0, 1]")

        state = _finite_array(initial_state, "initial_state").copy()
        supplied = _finite_array(controller_input, "controller_input")
        # Validate all shapes before starting the recurrent trajectory.
        if state.ndim < 1 or state.shape[-1] != self.state_dim:
            raise ValueError(f"initial_state must have trailing dimension {self.state_dim}")
        if supplied.ndim < 1 or supplied.shape[-1] != self.input_dim:
            raise ValueError(f"controller_input must have trailing dimension {self.input_dim}")
        if state.shape[:-1] != supplied.shape[:-1]:
            raise ValueError("initial_state and controller_input leading dimensions must match")

        cycle_limit = (
            fixed_cycles + requested_extra_cycles
            if mode == "fixed"
            else max_cycles + requested_extra_cycles
        )
        residuals: list[float] = []
        confidences: list[float] = []
        converged = False
        for cycle in range(cycle_limit):
            stage = (stage_offset + cycle) % self.num_stages
            next_state = self.step(state, supplied, stage=stage)
            residual = float(np.sqrt(np.mean(np.square(next_state - state))))
            measured_confidence = float(confidence if confidence is not None else 1.0 / (1.0 + residual))
            residuals.append(residual)
            confidences.append(measured_confidence)
            state = next_state
            confidence_ok = confidence_threshold is None or measured_confidence >= confidence_threshold
            converged = residual <= residual_tolerance and confidence_ok
            if mode == "adaptive" and cycle + 1 >= min_cycles and converged:
                break

        return ControllerResult(
            state=state,
            cycles=len(residuals),
            residual=residuals[-1],
            residual_history=tuple(residuals),
            confidence_history=tuple(confidences),
            converged=converged,
            requested_extra_cycles=requested_extra_cycles,
            cycle_limit=cycle_limit,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "format": CONTROLLER_FORMAT,
            "schema_version": CONTROLLER_SCHEMA_VERSION,
            "operator": "shared_gru_stage_adapter",
            "input_dim": self.input_dim,
            "state_dim": self.state_dim,
            "num_stages": self.num_stages,
            "adapter_rank": self.adapter_rank,
            "storage_dtype": "float64",
            "tensor_layout": {
                "input_kernel": [self.input_dim, 3 * self.state_dim],
                "recurrent_kernel": [self.state_dim, 3 * self.state_dim],
                "bias": [3 * self.state_dim],
                "stage_embeddings": [self.num_stages, self.state_dim],
                "adapter_down": [self.num_stages, self.state_dim, self.adapter_rank],
                "adapter_up": [self.num_stages, self.adapter_rank, self.state_dim],
            },
        }

    def tensors(self) -> dict[str, NDArray[np.float64]]:
        return {
            "input_kernel": self.input_kernel.copy(),
            "recurrent_kernel": self.recurrent_kernel.copy(),
            "bias": self.bias.copy(),
            "stage_embeddings": self.stage_embeddings.copy(),
            "adapter_down": self.adapter_down.copy(),
            "adapter_up": self.adapter_up.copy(),
        }

    @classmethod
    def from_state(
        cls, metadata: Mapping[str, Any], tensors: Mapping[str, ArrayLike]
    ) -> "SharedRecurrentController":
        if metadata.get("format") != CONTROLLER_FORMAT:
            raise ValueError("unsupported controller format")
        if metadata.get("schema_version") != CONTROLLER_SCHEMA_VERSION:
            raise ValueError("unsupported controller schema version")
        if metadata.get("operator") != "shared_gru_stage_adapter":
            raise ValueError("unsupported controller operator")
        expected = {
            "input_kernel",
            "recurrent_kernel",
            "bias",
            "stage_embeddings",
            "adapter_down",
            "adapter_up",
        }
        if set(tensors) != expected:
            raise ValueError(f"controller tensors must be exactly {sorted(expected)}")
        result = cls(**{name: np.asarray(tensors[name], dtype=np.float64) for name in expected})
        if result.metadata() != dict(metadata):
            raise ValueError("controller metadata does not match tensor dimensions")
        return result
