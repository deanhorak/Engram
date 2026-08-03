"""Transformer-free CPU replay for an authenticated Engram controller.

This runtime deliberately accepts operator streams as its input.  It does not
load a Transformers model, decoder layers, attention implementation, or MLP
weights.  A semantic-memory/episodic provider can therefore feed the same
controller state transition used by the packaged evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from engram.controller import FactorizedRecurrentController


@dataclass(frozen=True)
class ControllerOnlyResult:
    """State trajectory produced by the transformer-free controller."""

    final_state: np.ndarray
    stage_states: np.ndarray
    normalized_initial: np.ndarray


class ControllerOnlyRuntime:
    """Run a serialized factorized controller using only operator streams.

    ``semantic_outputs`` and ``episodic_outputs`` have shape
    ``[..., num_stages, state_dim]``.  The leading dimensions may represent
    batch and token positions.  ``initial_state`` has the matching leading
    dimensions and is normalized once before the first stage, matching the
    native BitNet state contract.

    The authenticated production artifact uses the exact operator-residual
    path with zero ``step_scale``.  Nonzero corrections are accepted only when
    explicitly requested, keeping evaluator-only experiments fail-closed.
    """

    def __init__(
        self,
        controller: FactorizedRecurrentController | str | Path,
        *,
        allow_correction: bool = False,
    ) -> None:
        if isinstance(controller, (str, Path)):
            controller = FactorizedRecurrentController.load(controller)
        if not isinstance(controller, FactorizedRecurrentController):
            raise TypeError("controller must be a FactorizedRecurrentController or path")
        if not controller.has_operator_residual:
            raise ValueError("controller-only runtime requires operator-residual schema")
        if not allow_correction and np.any(controller.step_scale != 0.0):
            raise ValueError(
                "nonzero controller correction is evaluator-only; pass allow_correction=True"
            )
        self.controller = controller
        self.allow_correction = bool(allow_correction)

    @staticmethod
    def _normalize_initial(initial_state: np.ndarray) -> np.ndarray:
        values = np.asarray(initial_state, dtype=np.float32)
        if values.ndim < 1 or values.shape[-1] <= 0:
            raise ValueError("initial_state must contain vectors")
        if not np.all(np.isfinite(values)):
            raise ValueError("initial_state must be finite")
        rms = np.sqrt(np.mean(np.square(values), axis=-1, keepdims=True) + 1e-6)
        return values / rms

    def run(
        self,
        initial_state: np.ndarray,
        semantic_outputs: np.ndarray,
        episodic_outputs: np.ndarray,
        *,
        initial_is_normalized: bool = False,
        stage_offset: int = 0,
    ) -> ControllerOnlyResult:
        initial = np.asarray(initial_state, dtype=np.float32)
        semantic = np.asarray(semantic_outputs, dtype=np.float32)
        episodic = np.asarray(episodic_outputs, dtype=np.float32)
        expected = (
            *initial.shape[:-1],
            self.controller.num_stages,
            self.controller.state_dim,
        )
        if initial.ndim < 1 or initial.shape[-1] != self.controller.state_dim:
            raise ValueError(
                f"initial_state must have trailing dimension {self.controller.state_dim}"
            )
        if semantic.shape != expected or episodic.shape != expected:
            raise ValueError(
                "operator streams must have shape [..., num_stages, state_dim] "
                f"with expected shape {expected}"
            )
        if not np.all(np.isfinite(semantic)) or not np.all(np.isfinite(episodic)):
            raise ValueError("operator streams must be finite")
        if isinstance(stage_offset, bool) or not isinstance(stage_offset, (int, np.integer)):
            raise ValueError("stage_offset must be an integer")
        if stage_offset < 0:
            raise ValueError("stage_offset must be non-negative")

        normalized = (
            np.ascontiguousarray(initial, dtype=np.float32)
            if initial_is_normalized
            else self._normalize_initial(initial)
        )
        state = normalized.copy()
        states = []
        token_embedding = normalized
        for cycle in range(self.controller.num_stages):
            stage = (stage_offset + cycle) % self.controller.num_stages
            supplied = np.concatenate(
                (token_embedding, semantic[..., cycle, :], episodic[..., cycle, :]),
                axis=-1,
            )
            state = self.controller.step(state, supplied, stage=stage)
            states.append(state)
        return ControllerOnlyResult(
            final_state=np.asarray(state, dtype=np.float32),
            stage_states=np.stack(states, axis=-2).astype(np.float32, copy=False),
            normalized_initial=normalized,
        )


__all__ = ["ControllerOnlyResult", "ControllerOnlyRuntime"]
