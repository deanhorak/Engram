"""Factorized shared recurrent controller for CPU deployment.

The original controller fixture stores two dense GRU projections.  At trained
model width those projections are larger than the CPU traffic budget.  This
module keeps the recurrent transition shared across depth while factorizing its
large projections through a small bottleneck.  Stage embeddings, low-rank state
adapters, an optional low-rank input adapter, and a scalar residual scale vary
by depth.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.utils import atomic_json

FACTORIZED_CONTROLLER_FORMAT = "engram.controller.factorized_residual"
FACTORIZED_CONTROLLER_SCHEMA_VERSION = 1
FACTORIZED_CONTROLLER_INPUT_ADAPTER_SCHEMA_VERSION = 2
FACTORIZED_CONTROLLER_OPERATOR_RESIDUAL_SCHEMA_VERSION = 3


def _array(value: ArrayLike, name: str) -> NDArray[np.float32]:
    result = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0.0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _silu(value: np.ndarray) -> np.ndarray:
    return value * _sigmoid(value)


@dataclass(frozen=True)
class FactorizedRecurrentController:
    """Shared, identity-biased depth recurrence with compact stage parameters.

    ``input_down`` and ``recurrent_down`` project the supplied operator inputs
    and recurrent state into one shared bottleneck. ``gate_up`` expands that
    bottleneck into a residual gate and candidate update.  The transition is::

        input_feature = input @ input_down
        input_feature += input @ input_adapter_down[stage] @ input_adapter_up[stage]
        feature = SiLU(input_feature + state @ recurrent_down)
        gate, candidate = split(feature @ gate_up + bias)
        candidate += stage_embedding + state @ adapter_down @ adapter_up
        residual = state + semantic_output + episodic_output
        residual += step_scale * sigmoid(gate) * tanh(candidate)
        next = residual / RMS(residual)

    The semantic and episodic terms are optional for compatibility with the
    original learned-transition artifacts.  New controller artifacts preserve
    that known residual algebra exactly and use the factorized recurrence only
    as a correction path.

    The explicit identity path is important for stable rollout over many shared
    depth cycles and makes a zero-initialized output projection an exact
    identity mapping.
    """

    input_down: NDArray[np.float32]
    recurrent_down: NDArray[np.float32]
    gate_up: NDArray[np.float32]
    bias: NDArray[np.float32]
    stage_embeddings: NDArray[np.float32]
    adapter_down: NDArray[np.float32]
    adapter_up: NDArray[np.float32]
    step_scale: NDArray[np.float32]
    input_adapter_down: NDArray[np.float32] | None = None
    input_adapter_up: NDArray[np.float32] | None = None
    operator_residual_scale: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        tensors = {
            name: _array(getattr(self, name), name)
            for name in (
                "input_down",
                "recurrent_down",
                "gate_up",
                "bias",
                "stage_embeddings",
                "adapter_down",
                "adapter_up",
                "step_scale",
            )
        }
        input_down = tensors["input_down"]
        recurrent_down = tensors["recurrent_down"]
        gate_up = tensors["gate_up"]
        bias = tensors["bias"]
        stages = tensors["stage_embeddings"]
        adapter_down = tensors["adapter_down"]
        adapter_up = tensors["adapter_up"]
        step_scale = tensors["step_scale"]
        if input_down.ndim != 2 or recurrent_down.ndim != 2 or gate_up.ndim != 2:
            raise ValueError("factorized controller kernels must be rank-2")
        rank = input_down.shape[1]
        state_dim = recurrent_down.shape[0]
        if rank <= 0 or recurrent_down.shape[1] != rank:
            raise ValueError("input and recurrent bottleneck ranks must match")
        if gate_up.shape != (rank, 2 * state_dim):
            raise ValueError("gate_up must have shape [rank, 2 * state_dim]")
        if bias.shape != (2 * state_dim,):
            raise ValueError("bias must have shape [2 * state_dim]")
        if stages.ndim != 2 or stages.shape[1] != state_dim or not stages.shape[0]:
            raise ValueError("stage_embeddings must have shape [num_stages, state_dim]")
        adapter_rank = adapter_down.shape[2] if adapter_down.ndim == 3 else -1
        if adapter_down.shape != (stages.shape[0], state_dim, adapter_rank):
            raise ValueError("adapter_down has incompatible dimensions")
        if adapter_up.shape != (stages.shape[0], adapter_rank, state_dim):
            raise ValueError("adapter_up has incompatible dimensions")
        if step_scale.shape != (stages.shape[0],):
            raise ValueError("step_scale must have shape [num_stages]")
        input_adapter_down = self.input_adapter_down
        input_adapter_up = self.input_adapter_up
        if input_adapter_down is None and input_adapter_up is None:
            input_adapter_down = np.zeros(
                (stages.shape[0], input_down.shape[0], 0), dtype=np.float32
            )
            input_adapter_up = np.zeros(
                (stages.shape[0], 0, rank), dtype=np.float32
            )
        elif input_adapter_down is None or input_adapter_up is None:
            raise ValueError("both input adapter tensors must be supplied together")
        else:
            input_adapter_down = _array(
                input_adapter_down, "input_adapter_down"
            )
            input_adapter_up = _array(input_adapter_up, "input_adapter_up")
        input_adapter_rank = (
            input_adapter_down.shape[2] if input_adapter_down.ndim == 3 else -1
        )
        if input_adapter_down.shape != (
            stages.shape[0],
            input_down.shape[0],
            input_adapter_rank,
        ):
            raise ValueError("input_adapter_down has incompatible dimensions")
        if input_adapter_up.shape != (
            stages.shape[0],
            input_adapter_rank,
            rank,
        ):
            raise ValueError("input_adapter_up has incompatible dimensions")
        tensors["input_adapter_down"] = input_adapter_down
        tensors["input_adapter_up"] = input_adapter_up
        operator_residual_scale = self.operator_residual_scale
        if operator_residual_scale is None:
            operator_residual_scale = np.zeros(
                (stages.shape[0], 0), dtype=np.float32
            )
        else:
            operator_residual_scale = _array(
                operator_residual_scale, "operator_residual_scale"
            )
        if operator_residual_scale.shape not in {
            (stages.shape[0], 0),
            (stages.shape[0], 2),
        }:
            raise ValueError(
                "operator_residual_scale must have shape [num_stages, 2]"
            )
        if (
            operator_residual_scale.shape[1] == 2
            and input_down.shape[0] != 3 * state_dim
        ):
            raise ValueError(
                "operator residual input must concatenate token, semantic, "
                "and episodic state-width vectors"
            )
        tensors["operator_residual_scale"] = operator_residual_scale
        for name, value in tensors.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def input_dim(self) -> int:
        return int(self.input_down.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.recurrent_down.shape[0])

    @property
    def rank(self) -> int:
        return int(self.input_down.shape[1])

    @property
    def num_stages(self) -> int:
        return int(self.stage_embeddings.shape[0])

    @property
    def adapter_rank(self) -> int:
        return int(self.adapter_down.shape[2])

    @property
    def input_adapter_rank(self) -> int:
        assert self.input_adapter_down is not None
        return int(self.input_adapter_down.shape[2])

    @property
    def has_operator_residual(self) -> bool:
        assert self.operator_residual_scale is not None
        return self.operator_residual_scale.shape[1] == 2

    @property
    def parameter_count(self) -> int:
        return sum(int(value.size) for value in self.tensors().values())

    @property
    def serialized_bytes(self) -> int:
        return sum(int(value.nbytes) for value in self.tensors().values())

    @classmethod
    def initialize(
        cls,
        *,
        input_dim: int,
        state_dim: int,
        num_stages: int,
        rank: int = 128,
        adapter_rank: int = 8,
        input_adapter_rank: int = 0,
        operator_residual: bool = False,
        seed: int = 0,
        residual_scale: float = 0.1,
    ) -> "FactorizedRecurrentController":
        dimensions = (
            input_dim,
            state_dim,
            num_stages,
            rank,
            adapter_rank,
            input_adapter_rank,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in dimensions
        ):
            raise ValueError("controller dimensions must be integers")
        if min(input_dim, state_dim, num_stages, rank) <= 0:
            raise ValueError("controller dimensions and rank must be positive")
        if adapter_rank < 0 or adapter_rank > state_dim:
            raise ValueError("adapter_rank must lie in [0, state_dim]")
        if input_adapter_rank < 0 or input_adapter_rank > rank:
            raise ValueError("input_adapter_rank must lie in [0, rank]")
        if not isinstance(operator_residual, (bool, np.bool_)):
            raise ValueError("operator_residual must be a boolean")
        if operator_residual and input_dim != 3 * state_dim:
            raise ValueError(
                "operator residual requires input_dim == 3 * state_dim"
            )
        if not math.isfinite(residual_scale) or residual_scale <= 0.0:
            raise ValueError("residual_scale must be finite and positive")
        rng = np.random.default_rng(seed)
        down_scale = 1.0 / math.sqrt(max(input_dim + state_dim, 1))
        adapter_scale = 1.0 / math.sqrt(max(state_dim, 1))
        # A small, rather than exactly zero, output projection preserves the
        # identity bias while allowing gradients into both down projections on
        # the first optimizer step.
        up_scale = 1e-3 / math.sqrt(rank)
        return cls(
            input_down=rng.normal(scale=down_scale, size=(input_dim, rank)).astype(
                np.float32
            ),
            recurrent_down=rng.normal(scale=down_scale, size=(state_dim, rank)).astype(
                np.float32
            ),
            gate_up=rng.normal(scale=up_scale, size=(rank, 2 * state_dim)).astype(
                np.float32
            ),
            bias=np.zeros(2 * state_dim, dtype=np.float32),
            stage_embeddings=np.zeros((num_stages, state_dim), dtype=np.float32),
            adapter_down=rng.normal(
                scale=adapter_scale,
                size=(num_stages, state_dim, adapter_rank),
            ).astype(np.float32),
            adapter_up=np.zeros(
                (num_stages, adapter_rank, state_dim), dtype=np.float32
            ),
            step_scale=np.full(num_stages, residual_scale, dtype=np.float32),
            input_adapter_down=rng.normal(
                scale=1.0 / math.sqrt(max(input_dim, 1)),
                size=(num_stages, input_dim, input_adapter_rank),
            ).astype(np.float32),
            input_adapter_up=np.zeros(
                (num_stages, input_adapter_rank, rank), dtype=np.float32
            ),
            operator_residual_scale=(
                np.ones((num_stages, 2), dtype=np.float32)
                if operator_residual
                else None
            ),
        )

    def step(
        self, state: ArrayLike, controller_input: ArrayLike, *, stage: int
    ) -> NDArray[np.float32]:
        current = _array(state, "state")
        supplied = _array(controller_input, "controller_input")
        if current.ndim < 1 or current.shape[-1] != self.state_dim:
            raise ValueError(f"state must have trailing dimension {self.state_dim}")
        if supplied.ndim < 1 or supplied.shape[-1] != self.input_dim:
            raise ValueError(
                f"controller_input must have trailing dimension {self.input_dim}"
            )
        if current.shape[:-1] != supplied.shape[:-1]:
            raise ValueError("state and controller_input leading dimensions must match")
        if isinstance(stage, bool) or not isinstance(stage, (int, np.integer)):
            raise ValueError("stage must be an integer")
        if stage < 0 or stage >= self.num_stages:
            raise ValueError(f"stage must lie in [0, {self.num_stages - 1}]")
        residual = current
        if self.has_operator_residual:
            assert self.operator_residual_scale is not None
            semantic = supplied[..., self.state_dim : 2 * self.state_dim]
            episodic = supplied[..., 2 * self.state_dim :]
            residual = (
                residual
                + self.operator_residual_scale[stage, 0] * semantic
                + self.operator_residual_scale[stage, 1] * episodic
            )
        # A compiled exact-residual artifact uses a zero correction scale.  Do
        # not touch the large factorized tensors in that CPU hot path.
        if self.step_scale[stage] != 0.0:
            input_feature = supplied @ self.input_down
            if self.input_adapter_rank:
                assert self.input_adapter_down is not None
                assert self.input_adapter_up is not None
                input_feature += (
                    supplied @ self.input_adapter_down[stage]
                ) @ self.input_adapter_up[stage]
            feature = _silu(input_feature + current @ self.recurrent_down)
            projected = feature @ self.gate_up + self.bias
            gate = _sigmoid(projected[..., : self.state_dim])
            candidate = (
                projected[..., self.state_dim :]
                + self.stage_embeddings[stage]
            )
            if self.adapter_rank:
                candidate += (
                    current @ self.adapter_down[stage]
                ) @ self.adapter_up[stage]
            delta = gate * np.tanh(candidate)
            residual = residual + self.step_scale[stage] * delta
        rms = np.sqrt(np.mean(np.square(residual), axis=-1, keepdims=True) + 1e-6)
        return residual / rms

    def run_staged(
        self,
        initial_state: ArrayLike,
        controller_inputs: ArrayLike,
        *,
        stage_offset: int = 0,
    ) -> NDArray[np.float32]:
        """Roll out a sequence of stage-specific, CPU-resident inputs."""

        state = _array(initial_state, "initial_state").copy()
        supplied = _array(controller_inputs, "controller_inputs")
        if supplied.ndim < 2 or supplied.shape[-1] != self.input_dim:
            raise ValueError(
                "controller_inputs must have shape [..., cycles, input_dim]"
            )
        if supplied.shape[:-2] != state.shape[:-1]:
            raise ValueError(
                "initial_state and controller_inputs leading dimensions must match"
            )
        if isinstance(stage_offset, bool) or not isinstance(
            stage_offset, (int, np.integer)
        ):
            raise ValueError("stage_offset must be an integer")
        for cycle in range(supplied.shape[-2]):
            stage = (stage_offset + cycle) % self.num_stages
            state = self.step(state, supplied[..., cycle, :], stage=stage)
        return state

    def metadata(self) -> dict[str, Any]:
        adapted = self.input_adapter_rank > 0
        operator_residual = self.has_operator_residual
        metadata = {
            "format": FACTORIZED_CONTROLLER_FORMAT,
            "schema_version": (
                FACTORIZED_CONTROLLER_OPERATOR_RESIDUAL_SCHEMA_VERSION
                if operator_residual
                else (
                    FACTORIZED_CONTROLLER_INPUT_ADAPTER_SCHEMA_VERSION
                    if adapted
                    else FACTORIZED_CONTROLLER_SCHEMA_VERSION
                )
            ),
            "operator": (
                "operator_residual_with_factorized_correction"
                if operator_residual
                else (
                    "factorized_residual_gate_stage_input_adapter"
                    if adapted
                    else "factorized_residual_gate_stage_adapter"
                )
            ),
            "state_normalization": "per_token_rms",
            "storage_dtype": "float32",
            "input_dim": self.input_dim,
            "state_dim": self.state_dim,
            "rank": self.rank,
            "num_stages": self.num_stages,
            "adapter_rank": self.adapter_rank,
            "parameter_count": self.parameter_count,
            "serialized_bytes": self.serialized_bytes,
            "tensor_layout": {
                name: list(value.shape) for name, value in self.tensors().items()
            },
        }
        if adapted:
            metadata["input_adapter_rank"] = self.input_adapter_rank
        if operator_residual:
            metadata["operator_residual_input_order"] = [
                "semantic_output",
                "episodic_output",
            ]
        return metadata

    def tensors(self) -> dict[str, NDArray[np.float32]]:
        tensors = {
            name: getattr(self, name).copy()
            for name in (
                "input_down",
                "recurrent_down",
                "gate_up",
                "bias",
                "stage_embeddings",
                "adapter_down",
                "adapter_up",
                "step_scale",
            )
        }
        if self.input_adapter_rank:
            assert self.input_adapter_down is not None
            assert self.input_adapter_up is not None
            tensors["input_adapter_down"] = self.input_adapter_down.copy()
            tensors["input_adapter_up"] = self.input_adapter_up.copy()
        if self.has_operator_residual:
            assert self.operator_residual_scale is not None
            tensors["operator_residual_scale"] = (
                self.operator_residual_scale.copy()
            )
        return tensors

    @classmethod
    def from_state(
        cls, metadata: Mapping[str, Any], tensors: Mapping[str, ArrayLike]
    ) -> "FactorizedRecurrentController":
        if metadata.get("format") != FACTORIZED_CONTROLLER_FORMAT:
            raise ValueError("unsupported factorized controller format")
        schema_version = metadata.get("schema_version")
        if schema_version not in {
            FACTORIZED_CONTROLLER_SCHEMA_VERSION,
            FACTORIZED_CONTROLLER_INPUT_ADAPTER_SCHEMA_VERSION,
            FACTORIZED_CONTROLLER_OPERATOR_RESIDUAL_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported factorized controller schema version")
        operator_residual = (
            schema_version
            == FACTORIZED_CONTROLLER_OPERATOR_RESIDUAL_SCHEMA_VERSION
        )
        adapted = (
            schema_version == FACTORIZED_CONTROLLER_INPUT_ADAPTER_SCHEMA_VERSION
            or (
                operator_residual
                and int(metadata.get("input_adapter_rank", 0)) > 0
            )
        )
        expected_operator = (
            "operator_residual_with_factorized_correction"
            if operator_residual
            else (
                "factorized_residual_gate_stage_input_adapter"
                if adapted
                else "factorized_residual_gate_stage_adapter"
            )
        )
        if metadata.get("operator") != expected_operator:
            raise ValueError("unsupported factorized controller operator")
        expected = {
            "input_down",
            "recurrent_down",
            "gate_up",
            "bias",
            "stage_embeddings",
            "adapter_down",
            "adapter_up",
            "step_scale",
        }
        if adapted:
            expected.update({"input_adapter_down", "input_adapter_up"})
        if operator_residual:
            expected.add("operator_residual_scale")
        if set(tensors) != expected:
            raise ValueError(f"controller tensors must be exactly {sorted(expected)}")
        controller = cls(
            **{name: np.asarray(tensors[name], dtype=np.float32) for name in expected}
        )
        if controller.metadata() != dict(metadata):
            raise ValueError(
                "factorized controller metadata does not match tensor dimensions"
            )
        return controller

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        for name, value in self.tensors().items():
            np.save(target / f"{name}.npy", value, allow_pickle=False)
        atomic_json(target / "metadata.json", self.metadata())

    @classmethod
    def load(cls, path: str | Path) -> "FactorizedRecurrentController":
        source = Path(path)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        layout = metadata.get("tensor_layout")
        if not isinstance(layout, dict):
            raise ValueError("factorized controller metadata has no tensor layout")
        tensors = {
            name: np.load(source / f"{name}.npy", mmap_mode="r") for name in layout
        }
        return cls.from_state(metadata, tensors)


__all__ = [
    "FACTORIZED_CONTROLLER_FORMAT",
    "FACTORIZED_CONTROLLER_INPUT_ADAPTER_SCHEMA_VERSION",
    "FACTORIZED_CONTROLLER_OPERATOR_RESIDUAL_SCHEMA_VERSION",
    "FACTORIZED_CONTROLLER_SCHEMA_VERSION",
    "FactorizedRecurrentController",
]
