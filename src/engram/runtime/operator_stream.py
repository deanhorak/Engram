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


class OperatorStreamProvider(Protocol):
    """Protocol consumed by the transformer-free controller runtime."""

    state_dim: int
    num_stages: int
    provider_kind: str

    def step(
        self, state: np.ndarray, token_embedding: np.ndarray, stage: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return semantic and episodic vectors for one stage."""


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
    "TraceOperatorStreamProvider",
    "PCAOperatorStreamProvider",
    "fit_operator_stream_provider",
]
