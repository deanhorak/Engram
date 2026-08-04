"""Authenticated operator-stream providers for the layer-free controller.

The controller is intentionally separated from the component that supplies
semantic and episodic vectors.  This module defines that seam and provides two
implementations:

* :class:`TraceOperatorStreamProvider` replays a checksummed teacher trace. It
  is useful for parity and state-transition tests, but it is never presented
  as a learned provider.
* :class:`PCAOperatorStreamProvider` is a compact, CPU-only linear provider
  trained from a trace. It predicts each stage's semantic/episodic vectors
  from the current controller state and token embedding. It is deliberately a
  research artifact until its held-out causal result passes the controller
  gate.

Neither provider imports Transformers or decoder-layer weights.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from engram.utils import atomic_json, sha256_file


OPERATOR_PROVIDER_FORMAT = "engram.operator_stream_provider"
OPERATOR_PROVIDER_VERSION = 1


def _finite(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _silu(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30.0, 30.0)
    return value / (1.0 + np.exp(-clipped))


class OperatorStreamProvider(Protocol):
    """Protocol consumed by the transformer-free controller runtime."""

    state_dim: int
    num_stages: int
    provider_kind: str

    def step(
        self, state: np.ndarray, token_embedding: np.ndarray, stage: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return semantic and episodic vectors for one stage."""


class StatefulOperatorStreamProvider(OperatorStreamProvider, Protocol):
    """Provider seam for token-by-token persistent context."""

    def reset(self, batch_shape: tuple[int, ...]) -> None:
        """Discard token history and allocate state for a new sequence batch."""

    def begin_token(self, token_embedding: np.ndarray) -> None:
        """Advance the provider's token context exactly once."""


@dataclass(frozen=True)
class TraceOperatorStreamProvider:
    """Trace-backed provider for independent replay and parity checks."""

    semantic_outputs: np.ndarray
    episodic_outputs: np.ndarray
    trace_sha256: str = ""
    provider_kind: str = "trace_replay"

    def __post_init__(self) -> None:
        semantic = _finite(self.semantic_outputs, "semantic_outputs")
        episodic = _finite(self.episodic_outputs, "episodic_outputs")
        if semantic.ndim != 3 or semantic.shape != episodic.shape:
            raise ValueError(
                "operator streams must have shape [records, stages, state_dim]"
            )
        if semantic.shape[0] == 0 or semantic.shape[1] == 0:
            raise ValueError("operator stream provider cannot be empty")
        semantic.setflags(write=False)
        episodic.setflags(write=False)
        object.__setattr__(self, "semantic_outputs", semantic)
        object.__setattr__(self, "episodic_outputs", episodic)

    @property
    def state_dim(self) -> int:
        return int(self.semantic_outputs.shape[-1])

    @property
    def num_stages(self) -> int:
        return int(self.semantic_outputs.shape[1])

    @property
    def records(self) -> int:
        return int(self.semantic_outputs.shape[0])

    @classmethod
    def from_trace(
        cls, trace: str | Path, *, record_indices: np.ndarray | None = None
    ) -> "TraceOperatorStreamProvider":
        # Import lazily: controller_distillation imports the native runtime,
        # whose package exports this module during initialization.
        from engram.training.controller_distillation import _load_trajectories

        path = Path(trace).resolve()
        data = _load_trajectories(path)
        if record_indices is None:
            indices = np.arange(data.records, dtype=np.int64)
        else:
            indices = np.asarray(record_indices, dtype=np.int64)
            if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= data.records):
                raise ValueError("record_indices are outside the trace")
        return cls(
            data.semantic_outputs[indices],
            data.episodic_outputs[indices],
            trace_sha256=sha256_file(path / "manifest.json"),
        )

    def step(
        self, state: np.ndarray, token_embedding: np.ndarray, stage: int
    ) -> tuple[np.ndarray, np.ndarray]:
        del state, token_embedding
        if not isinstance(stage, (int, np.integer)) or isinstance(stage, bool):
            raise ValueError("stage must be an integer")
        if stage < 0 or stage >= self.num_stages:
            raise ValueError("stage is outside the provider")
        return self.semantic_outputs[:, stage], self.episodic_outputs[:, stage]

    def metadata(self) -> dict[str, Any]:
        return {
            "format": OPERATOR_PROVIDER_FORMAT,
            "version": OPERATOR_PROVIDER_VERSION,
            "provider_kind": self.provider_kind,
            "trace_sha256": self.trace_sha256,
            "records": self.records,
            "state_dim": self.state_dim,
            "num_stages": self.num_stages,
            "learned": False,
            "transformer_layers_loaded": False,
        }


@dataclass
class TraceSequenceOperatorStreamProvider:
    """Sequence-preserving replay provider for persistent-runtime parity."""

    semantic_outputs: np.ndarray
    episodic_outputs: np.ndarray
    trace_sha256: str = ""
    provider_kind: str = "trace_sequence_replay"
    _position: int = -1

    def __post_init__(self) -> None:
        semantic = _finite(self.semantic_outputs, "semantic_outputs")
        episodic = _finite(self.episodic_outputs, "episodic_outputs")
        if semantic.ndim != 4 or semantic.shape != episodic.shape:
            raise ValueError(
                "sequence streams must have shape [batch, sequence, stages, state_dim]"
            )
        if semantic.shape[0] == 0 or semantic.shape[1] == 0:
            raise ValueError("sequence provider cannot be empty")
        semantic.setflags(write=False)
        episodic.setflags(write=False)
        object.__setattr__(self, "semantic_outputs", semantic)
        object.__setattr__(self, "episodic_outputs", episodic)

    @property
    def state_dim(self) -> int:
        return int(self.semantic_outputs.shape[-1])

    @property
    def num_stages(self) -> int:
        return int(self.semantic_outputs.shape[2])

    @property
    def sequence_length(self) -> int:
        return int(self.semantic_outputs.shape[1])

    @classmethod
    def from_trace(cls, trace: str | Path) -> "TraceSequenceOperatorStreamProvider":
        from engram.training.controller_distillation import _load_trajectories

        path = Path(trace).resolve()
        data = _load_trajectories(path)
        sample_ids = np.asarray(data.sample_id, dtype=np.int64)
        unique = np.unique(sample_ids)
        counts = [int(np.sum(sample_ids == sample)) for sample in unique]
        if not counts or len(set(counts)) != 1:
            raise ValueError("sequence trace records must have equal sample lengths")
        order = np.concatenate(
            [np.flatnonzero(sample_ids == sample) for sample in unique]
        )
        sequence = counts[0]
        return cls(
            data.semantic_outputs[order].reshape(
                len(unique), sequence, data.num_stages, data.hidden_size
            ),
            data.episodic_outputs[order].reshape(
                len(unique), sequence, data.num_stages, data.hidden_size
            ),
            trace_sha256=sha256_file(path / "manifest.json"),
        )

    def reset(self, batch_shape: tuple[int, ...]) -> None:
        if batch_shape != (self.semantic_outputs.shape[0],):
            raise ValueError("trace sequence provider batch shape does not match trace")
        self._position = -1

    def begin_token(self, token_embedding: np.ndarray) -> None:
        token = _finite(token_embedding, "token_embedding")
        if token.shape != (self.semantic_outputs.shape[0], self.state_dim):
            raise ValueError("trace sequence token batch shape does not match trace")
        self._position += 1
        if self._position >= self.sequence_length:
            raise ValueError("trace sequence provider advanced past its sequence")

    def step(
        self, state: np.ndarray, token_embedding: np.ndarray, stage: int
    ) -> tuple[np.ndarray, np.ndarray]:
        del state, token_embedding
        if self._position < 0:
            raise ValueError("begin_token must precede provider.step")
        if not isinstance(stage, (int, np.integer)) or isinstance(stage, bool):
            raise ValueError("stage must be an integer")
        if stage < 0 or stage >= self.num_stages:
            raise ValueError("stage is outside the provider")
        return (
            self.semantic_outputs[:, self._position, stage],
            self.episodic_outputs[:, self._position, stage],
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "format": OPERATOR_PROVIDER_FORMAT,
            "version": OPERATOR_PROVIDER_VERSION,
            "provider_kind": self.provider_kind,
            "trace_sha256": self.trace_sha256,
            "records": int(np.prod(self.semantic_outputs.shape[:2])),
            "sequence_length": self.sequence_length,
            "state_dim": self.state_dim,
            "num_stages": self.num_stages,
            "learned": False,
            "transformer_layers_loaded": False,
        }

    def save(self, path: str | Path) -> Path:
        """Serialize the sequence replay provider with an authenticated manifest.

        This is deliberately a replay artifact, not a learned provider.  The
        explicit format is useful for validating the layer-free sequence
        runtime after the source model and its decoder layers are gone.
        """

        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        arrays = {
            "semantic_outputs": self.semantic_outputs,
            "episodic_outputs": self.episodic_outputs,
        }
        inventory: dict[str, dict[str, Any]] = {}
        for name, value in arrays.items():
            file_path = target / f"{name}.npy"
            np.save(file_path, value, allow_pickle=False)
            inventory[file_path.name] = {
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        manifest = self.metadata()
        manifest.update(
            {
                "source_trace": self.trace_sha256,
                "files": inventory,
            }
        )
        atomic_json(target / "manifest.json", manifest)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TraceSequenceOperatorStreamProvider":
        """Load and checksum a serialized sequence replay provider."""

        target = Path(path)
        manifest_path = target / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("sequence provider has no manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("format") != OPERATOR_PROVIDER_FORMAT
            or manifest.get("version") != OPERATOR_PROVIDER_VERSION
            or manifest.get("provider_kind") != "trace_sequence_replay"
        ):
            raise ValueError("unsupported sequence replay provider")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("sequence provider manifest has no file inventory")
        arrays: dict[str, np.ndarray] = {}
        for name, info in files.items():
            file_path = target / name
            if (
                not file_path.is_file()
                or file_path.stat().st_size != info.get("bytes")
                or sha256_file(file_path) != info.get("sha256")
            ):
                raise ValueError(f"sequence provider checksum mismatch: {name}")
            arrays[Path(name).stem] = np.load(file_path, allow_pickle=False)
        return cls(
            semantic_outputs=arrays["semantic_outputs"],
            episodic_outputs=arrays["episodic_outputs"],
            trace_sha256=str(manifest.get("source_trace", "")),
        )


@dataclass(frozen=True)
class PCAOperatorStreamProvider:
    """Low-rank CPU provider conditioned on state and token embedding.

    For each stage and stream kind, a PCA basis compresses the teacher vector
    and a ridge regression maps ``[state, token_embedding, 1]`` to its latent
    coordinates. The artifact has no source-model tensors and can be loaded
    with NumPy alone.
    """

    semantic_mean: np.ndarray
    semantic_basis: np.ndarray
    semantic_projection: np.ndarray
    episodic_mean: np.ndarray
    episodic_basis: np.ndarray
    episodic_projection: np.ndarray
    metadata_payload: dict[str, Any]
    provider_kind: str = "pca_state_token"

    def __post_init__(self) -> None:
        arrays = {
            "semantic_mean": _finite(self.semantic_mean, "semantic_mean"),
            "semantic_basis": _finite(self.semantic_basis, "semantic_basis"),
            "semantic_projection": _finite(
                self.semantic_projection, "semantic_projection"
            ),
            "episodic_mean": _finite(self.episodic_mean, "episodic_mean"),
            "episodic_basis": _finite(self.episodic_basis, "episodic_basis"),
            "episodic_projection": _finite(
                self.episodic_projection, "episodic_projection"
            ),
        }
        if arrays["semantic_mean"].ndim != 2:
            raise ValueError("provider means must have shape [stages, state_dim]")
        stages, width = arrays["semantic_mean"].shape
        if arrays["episodic_mean"].shape != (stages, width):
            raise ValueError("semantic and episodic means differ")
        for prefix in ("semantic", "episodic"):
            mean = arrays[f"{prefix}_mean"]
            basis = arrays[f"{prefix}_basis"]
            projection = arrays[f"{prefix}_projection"]
            if basis.ndim != 3 or basis.shape[0] != stages or basis.shape[2] != width:
                raise ValueError(f"{prefix} basis has incompatible dimensions")
            rank = basis.shape[1]
            if projection.shape != (stages, 2 * width + 1, rank):
                raise ValueError(f"{prefix} projection has incompatible dimensions")
            if mean.shape != (stages, width):
                raise ValueError(f"{prefix} mean has incompatible dimensions")
        for name, value in arrays.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not isinstance(self.metadata_payload, dict):
            raise ValueError("provider metadata must be an object")

    @property
    def state_dim(self) -> int:
        return int(self.semantic_mean.shape[1])

    @property
    def num_stages(self) -> int:
        return int(self.semantic_mean.shape[0])

    @property
    def output_rank(self) -> int:
        return int(self.semantic_basis.shape[1])

    def _predict(
        self,
        state: np.ndarray,
        token_embedding: np.ndarray,
        stage: int,
        *,
        mean: np.ndarray,
        basis: np.ndarray,
        projection: np.ndarray,
    ) -> np.ndarray:
        current = _finite(state, "state")
        token = _finite(token_embedding, "token_embedding")
        if current.shape != token.shape or current.ndim < 1 or current.shape[-1] != self.state_dim:
            raise ValueError("state and token_embedding must have matching state width")
        if not isinstance(stage, (int, np.integer)) or isinstance(stage, bool):
            raise ValueError("stage must be an integer")
        if stage < 0 or stage >= self.num_stages:
            raise ValueError("stage is outside the provider")
        features = np.concatenate((current, token, np.ones((*current.shape[:-1], 1), dtype=np.float32)), axis=-1)
        latent = features @ projection[stage]
        return np.asarray(mean[stage] + latent @ basis[stage], dtype=np.float32)

    def step(
        self, state: np.ndarray, token_embedding: np.ndarray, stage: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._predict(
                state,
                token_embedding,
                stage,
                mean=self.semantic_mean,
                basis=self.semantic_basis,
                projection=self.semantic_projection,
            ),
            self._predict(
                state,
                token_embedding,
                stage,
                mean=self.episodic_mean,
                basis=self.episodic_basis,
                projection=self.episodic_projection,
            ),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            **self.metadata_payload,
            "format": OPERATOR_PROVIDER_FORMAT,
            "version": OPERATOR_PROVIDER_VERSION,
            "provider_kind": self.provider_kind,
            "state_dim": self.state_dim,
            "num_stages": self.num_stages,
            "output_rank": self.output_rank,
            "learned": True,
            "transformer_layers_loaded": False,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        arrays = {
            "semantic_mean": self.semantic_mean,
            "semantic_basis": self.semantic_basis,
            "semantic_projection": self.semantic_projection,
            "episodic_mean": self.episodic_mean,
            "episodic_basis": self.episodic_basis,
            "episodic_projection": self.episodic_projection,
        }
        inventory: dict[str, dict[str, Any]] = {}
        for name, value in arrays.items():
            file_path = target / f"{name}.npy"
            np.save(file_path, value, allow_pickle=False)
            inventory[file_path.name] = {
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        manifest = self.metadata()
        manifest["files"] = inventory
        atomic_json(target / "manifest.json", manifest)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "PCAOperatorStreamProvider":
        target = Path(path)
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != OPERATOR_PROVIDER_FORMAT or manifest.get("version") != OPERATOR_PROVIDER_VERSION:
            raise ValueError("unsupported operator-stream provider")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("provider manifest has no file inventory")
        arrays: dict[str, np.ndarray] = {}
        for name, info in files.items():
            file_path = target / name
            if not file_path.is_file() or file_path.stat().st_size != info.get("bytes") or sha256_file(file_path) != info.get("sha256"):
                raise ValueError(f"operator provider checksum mismatch: {name}")
            arrays[Path(name).stem] = np.load(file_path, allow_pickle=False)
        return cls(
            semantic_mean=arrays["semantic_mean"],
            semantic_basis=arrays["semantic_basis"],
            semantic_projection=arrays["semantic_projection"],
            episodic_mean=arrays["episodic_mean"],
            episodic_basis=arrays["episodic_basis"],
            episodic_projection=arrays["episodic_projection"],
            metadata_payload={
                key: value
                for key, value in manifest.items()
                if key != "files"
            },
        )


@dataclass
class RecurrentContextProvider:
    """Small stateful provider for sequence-level layer-free execution.

    The provider keeps a compact token-history state and updates it once per
    token.  Depth stages then read that same context while the controller
    advances through semantic/episodic stages.  This separates sequence
    recurrence from depth recurrence, which a memoryless token provider cannot
    represent.

    This class is an executable architecture fixture until its weights are
    learned against the protected causal split; ``initialize`` intentionally
    produces deterministic random weights and is never a promotion artifact.
    """

    memory_input: np.ndarray
    output_down: np.ndarray
    semantic_up: np.ndarray
    episodic_up: np.ndarray
    metadata_payload: dict[str, Any]
    provider_kind: str = "recurrent_context_fixture"
    _memory: np.ndarray | None = None

    def __post_init__(self) -> None:
        arrays = {
            "memory_input": _finite(self.memory_input, "memory_input"),
            "output_down": _finite(self.output_down, "output_down"),
            "semantic_up": _finite(self.semantic_up, "semantic_up"),
            "episodic_up": _finite(self.episodic_up, "episodic_up"),
        }
        if arrays["memory_input"].ndim != 2 or arrays["output_down"].ndim != 2:
            raise ValueError("recurrent provider projections must be rank-2")
        memory_dim = arrays["memory_input"].shape[1]
        state_dim = arrays["memory_input"].shape[0] - memory_dim - 1
        if state_dim <= 0:
            raise ValueError("memory_input has no positive token width")
        output_width = 2 * state_dim + memory_dim + 1
        if arrays["output_down"].shape[0] != output_width:
            raise ValueError("output_down has incompatible input width")
        rank = arrays["output_down"].shape[1]
        for name in ("semantic_up", "episodic_up"):
            value = arrays[name]
            if value.ndim != 3 or value.shape[1:] != (rank, state_dim):
                raise ValueError(f"{name} has incompatible dimensions")
        if arrays["semantic_up"].shape != arrays["episodic_up"].shape:
            raise ValueError("semantic and episodic stage projections differ")
        for name, value in arrays.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not isinstance(self.metadata_payload, dict):
            raise ValueError("provider metadata must be an object")

    @property
    def state_dim(self) -> int:
        return int(self.memory_input.shape[0] - self.memory_dim - 1)

    @property
    def memory_dim(self) -> int:
        return int(self.memory_input.shape[1])

    @property
    def num_stages(self) -> int:
        return int(self.semantic_up.shape[0])

    @property
    def output_rank(self) -> int:
        return int(self.output_down.shape[1])

    @classmethod
    def initialize(
        cls,
        *,
        state_dim: int,
        num_stages: int,
        memory_dim: int = 32,
        output_rank: int = 8,
        seed: int = 0,
    ) -> "RecurrentContextProvider":
        if min(state_dim, num_stages, memory_dim, output_rank) <= 0:
            raise ValueError("recurrent provider dimensions must be positive")
        rng = np.random.default_rng(seed)
        memory_input = rng.normal(
            scale=1.0 / np.sqrt(state_dim + memory_dim),
            size=(state_dim + memory_dim + 1, memory_dim),
        ).astype(np.float32)
        output_down = rng.normal(
            scale=1.0 / np.sqrt(2 * state_dim + memory_dim),
            size=(2 * state_dim + memory_dim + 1, output_rank),
        ).astype(np.float32)
        semantic_up = rng.normal(
            scale=1e-3, size=(num_stages, output_rank, state_dim)
        ).astype(np.float32)
        episodic_up = rng.normal(
            scale=1e-3, size=(num_stages, output_rank, state_dim)
        ).astype(np.float32)
        return cls(
            memory_input=memory_input,
            output_down=output_down,
            semantic_up=semantic_up,
            episodic_up=episodic_up,
            metadata_payload={
                "format": OPERATOR_PROVIDER_FORMAT,
                "version": OPERATOR_PROVIDER_VERSION,
                "learned": False,
                "transformer_layers_loaded": False,
                "state_dim": state_dim,
                "num_stages": num_stages,
                "memory_dim": memory_dim,
                "output_rank": output_rank,
                "architecture": "shared_token_memory_plus_stage_low_rank_outputs",
            },
        )

    def reset(self, batch_shape: tuple[int, ...]) -> None:
        if not isinstance(batch_shape, tuple) or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0
            for value in batch_shape
        ):
            raise ValueError("batch_shape must contain positive integer dimensions")
        self._memory = np.zeros((*batch_shape, self.memory_dim), dtype=np.float32)

    def begin_token(self, token_embedding: np.ndarray) -> None:
        token = _finite(token_embedding, "token_embedding")
        if token.ndim < 1 or token.shape[-1] != self.state_dim:
            raise ValueError("token_embedding has incompatible width")
        if self._memory is None or self._memory.shape[:-1] != token.shape[:-1]:
            raise ValueError("reset must be called with the token batch shape first")
        memory_input = np.concatenate(
            (self._memory, token, np.ones((*token.shape[:-1], 1), dtype=np.float32)),
            axis=-1,
        )
        self._memory = np.tanh(memory_input @ self.memory_input).astype(
            np.float32, copy=False
        )

    def step(
        self, state: np.ndarray, token_embedding: np.ndarray, stage: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._memory is None:
            raise ValueError("reset and begin_token must precede provider.step")
        state = _finite(state, "state")
        token = _finite(token_embedding, "token_embedding")
        if state.shape != token.shape or state.shape[:-1] != self._memory.shape[:-1]:
            raise ValueError("provider state, token, and memory shapes differ")
        if not isinstance(stage, (int, np.integer)) or isinstance(stage, bool):
            raise ValueError("stage must be an integer")
        if stage < 0 or stage >= self.num_stages:
            raise ValueError("stage is outside the provider")
        features = np.concatenate(
            (state, token, self._memory, np.ones((*state.shape[:-1], 1), dtype=np.float32)),
            axis=-1,
        )
        hidden = _silu(features @ self.output_down)
        semantic = hidden @ self.semantic_up[stage]
        episodic = hidden @ self.episodic_up[stage]
        return semantic.astype(np.float32, copy=False), episodic.astype(
            np.float32, copy=False
        )

    def metadata(self) -> dict[str, Any]:
        return {
            **self.metadata_payload,
            "format": OPERATOR_PROVIDER_FORMAT,
            "version": OPERATOR_PROVIDER_VERSION,
            "provider_kind": self.provider_kind,
            "learned": bool(self.metadata_payload.get("learned", False)),
            "transformer_layers_loaded": False,
        }


@dataclass
class StateSpaceOperatorStreamProvider:
    """Learned diagonal state-space provider with low-rank stage outputs.

    The token memory is updated once per token with a diagonal recurrence,
    while each depth stage reads the same memory and predicts semantic and
    episodic PCA latents.  The artifact contains only NumPy tensors and can be
    executed by the sequence runtime without a Transformer model.  It is a
    research provider until its causal held-out result passes the promotion
    threshold.
    """

    memory_input: np.ndarray
    decay: np.ndarray
    state_projection: np.ndarray
    token_projection: np.ndarray
    stage_head: np.ndarray
    semantic_mean: np.ndarray
    semantic_basis: np.ndarray
    episodic_mean: np.ndarray
    episodic_basis: np.ndarray
    metadata_payload: dict[str, Any]
    provider_kind: str = "state_space_pca"
    _memory: np.ndarray | None = None

    def __post_init__(self) -> None:
        arrays = {
            "memory_input": _finite(self.memory_input, "memory_input"),
            "decay": _finite(self.decay, "decay"),
            "state_projection": _finite(self.state_projection, "state_projection"),
            "token_projection": _finite(self.token_projection, "token_projection"),
            "stage_head": _finite(self.stage_head, "stage_head"),
            "semantic_mean": _finite(self.semantic_mean, "semantic_mean"),
            "semantic_basis": _finite(self.semantic_basis, "semantic_basis"),
            "episodic_mean": _finite(self.episodic_mean, "episodic_mean"),
            "episodic_basis": _finite(self.episodic_basis, "episodic_basis"),
        }
        if arrays["memory_input"].ndim != 2:
            raise ValueError("memory_input must have shape [state_dim, memory_dim]")
        state_dim, memory_dim = arrays["memory_input"].shape
        if arrays["decay"].shape != (memory_dim,):
            raise ValueError("decay must match memory_dim")
        for name in ("state_projection", "token_projection"):
            if arrays[name].ndim != 2 or arrays[name].shape[0] != state_dim:
                raise ValueError(f"{name} must have shape [state_dim, projection_width]")
        projection_width = arrays["state_projection"].shape[1]
        if arrays["token_projection"].shape[1] != projection_width:
            raise ValueError("state and token projection widths differ")
        if arrays["semantic_mean"].ndim != 2:
            raise ValueError("semantic_mean must have shape [stages, state_dim]")
        stages = arrays["semantic_mean"].shape[0]
        if arrays["semantic_mean"].shape != (stages, state_dim):
            raise ValueError("semantic_mean has incompatible dimensions")
        if arrays["episodic_mean"].shape != (stages, state_dim):
            raise ValueError("episodic_mean has incompatible dimensions")
        for prefix in ("semantic", "episodic"):
            basis = arrays[f"{prefix}_basis"]
            if basis.ndim != 3 or basis.shape[0] != stages or basis.shape[2] != state_dim:
                raise ValueError(f"{prefix}_basis has incompatible dimensions")
        rank = arrays["semantic_basis"].shape[1]
        if arrays["episodic_basis"].shape[1] != rank:
            raise ValueError("semantic and episodic output ranks differ")
        expected_head = (stages, 2 * projection_width + memory_dim + 1, 2 * rank)
        if arrays["stage_head"].shape != expected_head:
            raise ValueError(f"stage_head must have shape {expected_head}")
        for name, value in arrays.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not isinstance(self.metadata_payload, dict):
            raise ValueError("provider metadata must be an object")

    @property
    def state_dim(self) -> int:
        return int(self.memory_input.shape[0])

    @property
    def memory_dim(self) -> int:
        return int(self.memory_input.shape[1])

    @property
    def projection_width(self) -> int:
        return int(self.state_projection.shape[1])

    @property
    def num_stages(self) -> int:
        return int(self.stage_head.shape[0])

    @property
    def output_rank(self) -> int:
        return int(self.semantic_basis.shape[1])

    def reset(self, batch_shape: tuple[int, ...]) -> None:
        if not isinstance(batch_shape, tuple) or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0
            for value in batch_shape
        ):
            raise ValueError("batch_shape must contain positive integer dimensions")
        self._memory = np.zeros((*batch_shape, self.memory_dim), dtype=np.float32)

    def begin_token(self, token_embedding: np.ndarray) -> None:
        token = _finite(token_embedding, "token_embedding")
        if token.ndim < 1 or token.shape[-1] != self.state_dim:
            raise ValueError("token_embedding has incompatible width")
        if self._memory is None or self._memory.shape[:-1] != token.shape[:-1]:
            raise ValueError("reset must be called with the token batch shape first")
        self._memory = np.tanh(
            self._memory * self.decay + token @ self.memory_input
        ).astype(np.float32, copy=False)

    def step(
        self, state: np.ndarray, token_embedding: np.ndarray, stage: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._memory is None:
            raise ValueError("reset and begin_token must precede provider.step")
        state = _finite(state, "state")
        token = _finite(token_embedding, "token_embedding")
        if state.shape != token.shape or state.shape[:-1] != self._memory.shape[:-1]:
            raise ValueError("provider state, token, and memory shapes differ")
        if not isinstance(stage, (int, np.integer)) or isinstance(stage, bool):
            raise ValueError("stage must be an integer")
        if stage < 0 or stage >= self.num_stages:
            raise ValueError("stage is outside the provider")
        features = np.concatenate(
            (
                state @ self.state_projection,
                token @ self.token_projection,
                self._memory,
                np.ones((*state.shape[:-1], 1), dtype=np.float32),
            ),
            axis=-1,
        )
        latent = features @ self.stage_head[stage]
        semantic = self.semantic_mean[stage] + latent[..., : self.output_rank] @ self.semantic_basis[stage]
        episodic = self.episodic_mean[stage] + latent[..., self.output_rank :] @ self.episodic_basis[stage]
        return np.asarray(semantic, dtype=np.float32), np.asarray(episodic, dtype=np.float32)

    def metadata(self) -> dict[str, Any]:
        return {
            **self.metadata_payload,
            "format": OPERATOR_PROVIDER_FORMAT,
            "version": OPERATOR_PROVIDER_VERSION,
            "provider_kind": self.provider_kind,
            "state_dim": self.state_dim,
            "num_stages": self.num_stages,
            "memory_dim": self.memory_dim,
            "projection_width": self.projection_width,
            "output_rank": self.output_rank,
            "learned": bool(self.metadata_payload.get("learned", True)),
            "transformer_layers_loaded": False,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        arrays = {
            "memory_input": self.memory_input,
            "decay": self.decay,
            "state_projection": self.state_projection,
            "token_projection": self.token_projection,
            "stage_head": self.stage_head,
            "semantic_mean": self.semantic_mean,
            "semantic_basis": self.semantic_basis,
            "episodic_mean": self.episodic_mean,
            "episodic_basis": self.episodic_basis,
        }
        inventory: dict[str, dict[str, Any]] = {}
        for name, value in arrays.items():
            file_path = target / f"{name}.npy"
            np.save(file_path, value, allow_pickle=False)
            inventory[file_path.name] = {
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        manifest = self.metadata()
        manifest["files"] = inventory
        atomic_json(target / "manifest.json", manifest)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "StateSpaceOperatorStreamProvider":
        target = Path(path)
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != OPERATOR_PROVIDER_FORMAT or manifest.get("version") != OPERATOR_PROVIDER_VERSION or manifest.get("provider_kind") != "state_space_pca":
            raise ValueError("unsupported state-space operator provider")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("state-space provider manifest has no file inventory")
        arrays: dict[str, np.ndarray] = {}
        for name, info in files.items():
            file_path = target / name
            if not file_path.is_file() or file_path.stat().st_size != info.get("bytes") or sha256_file(file_path) != info.get("sha256"):
                raise ValueError(f"state-space provider checksum mismatch: {name}")
            arrays[Path(name).stem] = np.load(file_path, allow_pickle=False)
        return cls(
            **arrays,
            metadata_payload={key: value for key, value in manifest.items() if key != "files"},
        )


@dataclass
class ResidualStateSpaceOperatorStreamProvider:
    """Persistent-memory residual adapter over a full PCA provider.

    The wrapped provider remains the exact λ-regularized state/token path;
    the state-space component contributes only low-rank latent corrections.
    Zero correction tensors therefore reproduce the wrapped provider exactly,
    which makes ablation and rollback unambiguous.
    """

    base_provider: PCAOperatorStreamProvider
    memory_input: np.ndarray
    decay: np.ndarray
    correction_head: np.ndarray
    metadata_payload: dict[str, Any]
    provider_kind: str = "state_space_residual_pca"
    _memory: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_provider, PCAOperatorStreamProvider):
            raise TypeError("base_provider must be a PCAOperatorStreamProvider")
        memory_input = _finite(self.memory_input, "memory_input")
        decay = _finite(self.decay, "decay")
        head = _finite(self.correction_head, "correction_head")
        if memory_input.ndim != 2 or memory_input.shape[0] != self.base_provider.state_dim:
            raise ValueError("memory_input must have shape [state_dim, memory_dim]")
        memory_dim = memory_input.shape[1]
        if decay.shape != (memory_dim,):
            raise ValueError("decay must match memory_dim")
        expected = (self.base_provider.num_stages, memory_dim + 1, 2 * self.base_provider.output_rank)
        if head.shape != expected:
            raise ValueError(f"correction_head must have shape {expected}")
        if not isinstance(self.metadata_payload, dict):
            raise ValueError("provider metadata must be an object")
        for name, value in (
            ("memory_input", memory_input),
            ("decay", decay),
            ("correction_head", head),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def state_dim(self) -> int:
        return self.base_provider.state_dim

    @property
    def num_stages(self) -> int:
        return self.base_provider.num_stages

    @property
    def memory_dim(self) -> int:
        return int(self.memory_input.shape[1])

    @property
    def output_rank(self) -> int:
        return self.base_provider.output_rank

    def reset(self, batch_shape: tuple[int, ...]) -> None:
        if not isinstance(batch_shape, tuple) or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0
            for value in batch_shape
        ):
            raise ValueError("batch_shape must contain positive integer dimensions")
        self._memory = np.zeros((*batch_shape, self.memory_dim), dtype=np.float32)

    def begin_token(self, token_embedding: np.ndarray) -> None:
        token = _finite(token_embedding, "token_embedding")
        if token.ndim < 1 or token.shape[-1] != self.state_dim:
            raise ValueError("token_embedding has incompatible width")
        if self._memory is None or self._memory.shape[:-1] != token.shape[:-1]:
            raise ValueError("reset must be called with the token batch shape first")
        self._memory = np.tanh(
            self._memory * self.decay + token @ self.memory_input
        ).astype(np.float32, copy=False)

    def step(
        self, state: np.ndarray, token_embedding: np.ndarray, stage: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._memory is None:
            raise ValueError("reset and begin_token must precede provider.step")
        state = _finite(state, "state")
        token = _finite(token_embedding, "token_embedding")
        if state.shape != token.shape or state.shape[:-1] != self._memory.shape[:-1]:
            raise ValueError("provider state, token, and memory shapes differ")
        if not isinstance(stage, (int, np.integer)) or isinstance(stage, bool):
            raise ValueError("stage must be an integer")
        if stage < 0 or stage >= self.num_stages:
            raise ValueError("stage is outside the provider")
        semantic, episodic = self.base_provider.step(state, token, stage)
        features = np.concatenate(
            (self._memory, np.ones((*state.shape[:-1], 1), dtype=np.float32)), axis=-1
        )
        latent = features @ self.correction_head[stage]
        semantic = semantic + latent[..., : self.output_rank] @ self.base_provider.semantic_basis[stage]
        episodic = episodic + latent[..., self.output_rank :] @ self.base_provider.episodic_basis[stage]
        return np.asarray(semantic, dtype=np.float32), np.asarray(episodic, dtype=np.float32)

    def metadata(self) -> dict[str, Any]:
        return {
            **self.metadata_payload,
            "format": OPERATOR_PROVIDER_FORMAT,
            "version": OPERATOR_PROVIDER_VERSION,
            "provider_kind": self.provider_kind,
            "state_dim": self.state_dim,
            "num_stages": self.num_stages,
            "memory_dim": self.memory_dim,
            "output_rank": self.output_rank,
            "base_provider_kind": self.base_provider.provider_kind,
            "base_provider_metadata": self.base_provider.metadata(),
            "learned": bool(self.metadata_payload.get("learned", True)),
            "transformer_layers_loaded": False,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        arrays = {
            "memory_input": self.memory_input,
            "decay": self.decay,
            "correction_head": self.correction_head,
            "base_semantic_mean": self.base_provider.semantic_mean,
            "base_semantic_basis": self.base_provider.semantic_basis,
            "base_semantic_projection": self.base_provider.semantic_projection,
            "base_episodic_mean": self.base_provider.episodic_mean,
            "base_episodic_basis": self.base_provider.episodic_basis,
            "base_episodic_projection": self.base_provider.episodic_projection,
        }
        inventory: dict[str, dict[str, Any]] = {}
        for name, value in arrays.items():
            file_path = target / f"{name}.npy"
            np.save(file_path, value, allow_pickle=False)
            inventory[file_path.name] = {
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        manifest = self.metadata()
        manifest["files"] = inventory
        atomic_json(target / "manifest.json", manifest)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ResidualStateSpaceOperatorStreamProvider":
        target = Path(path)
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != OPERATOR_PROVIDER_FORMAT or manifest.get("version") != OPERATOR_PROVIDER_VERSION or manifest.get("provider_kind") != "state_space_residual_pca":
            raise ValueError("unsupported residual state-space provider")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("residual provider manifest has no file inventory")
        arrays: dict[str, np.ndarray] = {}
        for name, info in files.items():
            file_path = target / name
            if not file_path.is_file() or file_path.stat().st_size != info.get("bytes") or sha256_file(file_path) != info.get("sha256"):
                raise ValueError(f"residual provider checksum mismatch: {name}")
            arrays[Path(name).stem] = np.load(file_path, allow_pickle=False)
        base = PCAOperatorStreamProvider(
            semantic_mean=arrays["base_semantic_mean"],
            semantic_basis=arrays["base_semantic_basis"],
            semantic_projection=arrays["base_semantic_projection"],
            episodic_mean=arrays["base_episodic_mean"],
            episodic_basis=arrays["base_episodic_basis"],
            episodic_projection=arrays["base_episodic_projection"],
            metadata_payload=dict(manifest.get("base_provider_metadata", {})),
        )
        return cls(
            base_provider=base,
            memory_input=arrays["memory_input"],
            decay=arrays["decay"],
            correction_head=arrays["correction_head"],
            metadata_payload={key: value for key, value in manifest.items() if key not in {"files", "base_provider_metadata"}},
        )


def _fit_stream(
    features: np.ndarray,
    outputs: np.ndarray,
    *,
    output_rank: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = outputs - outputs.mean(axis=0, keepdims=True)
    _u, _s, basis_t = np.linalg.svd(centered, full_matrices=False)
    rank = min(int(output_rank), basis_t.shape[0], basis_t.shape[1])
    basis = basis_t[:rank].astype(np.float32)
    latent = centered @ basis.T
    gram = features @ features.T
    scale = max(float(np.mean(np.diag(gram))), 1.0)
    gram = gram + np.eye(gram.shape[0], dtype=np.float32) * (float(ridge) * scale)
    coefficients = features.T @ np.linalg.solve(gram, latent)
    return outputs.mean(axis=0).astype(np.float32), basis, coefficients.astype(np.float32)


def fit_operator_stream_provider(
    trace: str | Path,
    out: str | Path,
    *,
    output_rank: int = 16,
    ridge: float = 1e-2,
    target: str = "streams",
) -> dict[str, Any]:
    """Fit and serialize a compact state/token-conditioned provider.

    ``target="streams"`` reconstructs the teacher's separately captured
    semantic and episodic vectors. ``target="combined_delta"`` is a
    co-adapted research arm: it learns the normalized state delta directly as
    the semantic stream and sets the episodic stream to zero. The latter tests
    whether the controller boundary, rather than the teacher's decomposition,
    is the learnable target.
    """

    if isinstance(output_rank, bool) or not isinstance(output_rank, (int, np.integer)) or output_rank <= 0:
        raise ValueError("output_rank must be a positive integer")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    if target not in {"streams", "combined_delta"}:
        raise ValueError("target must be 'streams' or 'combined_delta'")
    trace_path = Path(trace).resolve()
    from engram.training.controller_distillation import _load_trajectories

    data = _load_trajectories(trace_path)
    state = data.teacher_states[:, :-1].astype(np.float32)
    token = data.token_embedding.astype(np.float32)
    if target == "streams":
        semantic = data.semantic_outputs.astype(np.float32)
        episodic = data.episodic_outputs.astype(np.float32)
    else:
        semantic = (
            data.teacher_states[:, 1:].astype(np.float32)
            - data.teacher_states[:, :-1].astype(np.float32)
        )
        episodic = np.zeros_like(semantic)
    stages = data.num_stages
    width = data.hidden_size
    semantic_mean = np.empty((stages, width), dtype=np.float32)
    fitted_rank = min(int(output_rank), data.records, width)
    semantic_basis = np.empty((stages, fitted_rank, width), dtype=np.float32)
    semantic_projection = np.empty((stages, 2 * width + 1, fitted_rank), dtype=np.float32)
    episodic_mean = np.empty_like(semantic_mean)
    episodic_basis = np.empty_like(semantic_basis)
    episodic_projection = np.empty_like(semantic_projection)
    for stage in range(stages):
        features = np.concatenate((state[:, stage], token, np.ones((data.records, 1), dtype=np.float32)), axis=-1)
        sm, sb, sp = _fit_stream(features, semantic[:, stage], output_rank=output_rank, ridge=ridge)
        em, eb, ep = _fit_stream(features, episodic[:, stage], output_rank=output_rank, ridge=ridge)
        semantic_mean[stage], semantic_basis[stage, : sb.shape[0]], semantic_projection[stage, :, : sp.shape[1] if sp.ndim > 1 else 0] = sm, sb, sp
        episodic_mean[stage], episodic_basis[stage, : eb.shape[0]], episodic_projection[stage, :, : ep.shape[1] if ep.ndim > 1 else 0] = em, eb, ep
    provider = PCAOperatorStreamProvider(
        semantic_mean=semantic_mean,
        semantic_basis=semantic_basis,
        semantic_projection=semantic_projection,
        episodic_mean=episodic_mean,
        episodic_basis=episodic_basis,
        episodic_projection=episodic_projection,
        metadata_payload={
            "source_trace": str(trace_path),
            "source_trace_manifest_sha256": sha256_file(trace_path / "manifest.json"),
            "source_model_hash": data.manifest.get("model_hash"),
            "source_dataset_hash": data.manifest.get("dataset_hash"),
            "training_records": data.records,
            "input_order": ["controller_state", "token_embedding", "bias"],
            "ridge": float(ridge),
            "requested_output_rank": int(output_rank),
            "operator_normalization": data.manifest.get("metadata", {}).get("operator_normalization"),
            "target": target,
        },
    )
    saved = provider.save(out)
    return {
        "provider": str(saved.resolve()),
        "provider_sha256": _directory_sha256(saved),
        "metadata": provider.metadata(),
    }


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


__all__ = [
    "OPERATOR_PROVIDER_FORMAT",
    "OPERATOR_PROVIDER_VERSION",
    "OperatorStreamProvider",
    "StatefulOperatorStreamProvider",
    "TraceOperatorStreamProvider",
    "TraceSequenceOperatorStreamProvider",
    "PCAOperatorStreamProvider",
    "RecurrentContextProvider",
    "StateSpaceOperatorStreamProvider",
    "ResidualStateSpaceOperatorStreamProvider",
    "fit_operator_stream_provider",
]
