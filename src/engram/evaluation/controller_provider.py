"""Held-out evaluation for a learned layer-free operator provider."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from engram.runtime.controller_only import ControllerOnlyRuntime
from engram.runtime.operator_stream import (
    TraceSequenceOperatorStreamProvider,
    load_operator_stream_provider,
)
from engram.training.controller_distillation import _load_trajectories
from engram.utils import atomic_json, sha256_file


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def evaluate_controller_provider_trace(
    trace: str | Path,
    provider: str | Path,
    controller: str | Path,
    *,
    out: str | Path,
    allow_correction: bool = False,
) -> dict[str, Any]:
    """Replay a learned provider and controller on an independent trace."""

    trace_path = Path(trace).resolve()
    provider_path = Path(provider).resolve()
    controller_path = Path(controller).resolve()
    data = _load_trajectories(trace_path)
    loaded_provider = load_operator_stream_provider(provider_path)
    runtime = ControllerOnlyRuntime(controller_path, allow_correction=allow_correction)
    provider_model_hash = loaded_provider.metadata().get("source_model_hash")
    trace_model_hash = data.manifest.get("model_hash")
    if provider_model_hash and trace_model_hash and provider_model_hash != trace_model_hash:
        raise ValueError("provider and evaluation trace source-model hashes differ")
    provider_dataset_hash = loaded_provider.metadata().get("source_dataset_hash")
    trace_dataset_hash = data.manifest.get("dataset_hash")
    if provider_dataset_hash and trace_dataset_hash and provider_dataset_hash == trace_dataset_hash:
        raise ValueError("provider training and evaluation dataset hashes must differ")
    if loaded_provider.state_dim != data.hidden_size:
        raise ValueError("provider width does not match trajectory")
    if loaded_provider.num_stages != data.num_stages:
        raise ValueError("provider stages do not match trajectory")
    if runtime.controller.state_dim != data.hidden_size:
        raise ValueError("controller width does not match trajectory")
    if runtime.controller.num_stages != data.num_stages:
        raise ValueError("controller stages do not match trajectory")

    started = time.perf_counter()
    result = runtime.run_provider(
        data.teacher_states[:, 0].astype(np.float32),
        data.token_embedding.astype(np.float32),
        loaded_provider,
        initial_is_normalized=True,
    )
    elapsed = time.perf_counter() - started
    target = data.teacher_states[:, 1:].astype(np.float32)
    error = result.stage_states - target
    stage_mse = np.mean(np.square(error), axis=(0, 2))
    report = {
        "experiment": "controller_provider_trace_replay",
        "status": "development_result",
        "trace": str(trace_path),
        "trace_manifest_sha256": sha256_file(trace_path / "manifest.json"),
        "provider": str(provider_path),
        "provider_sha256": _directory_sha256(provider_path),
        "provider_metadata": loaded_provider.metadata(),
        "controller": str(controller_path),
        "controller_sha256": _directory_sha256(controller_path),
        "records": data.records,
        "protected_validation": bool(
            provider_dataset_hash
            and trace_dataset_hash
            and provider_dataset_hash != trace_dataset_hash
        ),
        "hidden_size": data.hidden_size,
        "num_stages": data.num_stages,
        "elapsed_seconds": elapsed,
        "inference_device": "cpu",
        "transformers_model_loaded": False,
        "decoder_layer_forward_calls": 0,
        "operator_streams": {
            "provider": loaded_provider.provider_kind,
            "input_order": ["controller_state", "token_embedding", "bias"],
        },
        "metrics": {
            "mean_stage_normalized_mse": float(np.mean(stage_mse)),
            "maximum_stage_normalized_mse": float(np.max(stage_mse)),
            "terminal_normalized_mse": float(stage_mse[-1]),
            "stage_normalized_mse": [float(value) for value in stage_mse],
            "hidden_mse": float(np.mean(np.square(error))),
            "maximum_absolute_error": float(np.max(np.abs(error))),
        },
        "gate": {
            "metric": "terminal_normalized_mse",
            "threshold": 0.0225,
            "passed": bool(stage_mse[-1] <= 0.0225),
            "provider_is_learned": bool(loaded_provider.metadata().get("learned")),
            "layer_free_cpu_replay": True,
            "end_to_end_generation": False,
        },
    }
    atomic_json(Path(out), report)
    return report


def evaluate_controller_sequence_replay(
    trace: str | Path,
    provider: str | Path,
    controller: str | Path,
    *,
    out: str | Path,
) -> dict[str, Any]:
    """Validate persistent sequence replay against a captured trace.

    The provider is intentionally required to be the serialized
    ``trace_sequence_replay`` artifact.  This command proves ordering,
    reset/advance semantics, and CPU state-transition parity; it cannot
    promote a learned provider.
    """

    trace_path = Path(trace).resolve()
    provider_path = Path(provider).resolve()
    controller_path = Path(controller).resolve()
    data = _load_trajectories(trace_path)
    loaded_provider = TraceSequenceOperatorStreamProvider.load(provider_path)
    unique = np.unique(data.sample_id)
    counts = [int(np.sum(data.sample_id == sample)) for sample in unique]
    if not counts or len(set(counts)) != 1:
        raise ValueError("sequence replay trace records must have equal sample lengths")
    if loaded_provider.semantic_outputs.shape[:2] != (len(unique), counts[0]):
        raise ValueError("serialized sequence provider shape does not match trace")
    order = np.concatenate(
        [np.flatnonzero(data.sample_id == sample) for sample in unique]
    )
    initial = data.teacher_states[order, 0].astype(np.float32).reshape(
        len(unique), counts[0], data.hidden_size
    )
    token = data.token_embedding[order].astype(np.float32).reshape(
        len(unique), counts[0], data.hidden_size
    )
    target = data.teacher_states[order, 1:].astype(np.float32).reshape(
        len(unique), counts[0], data.num_stages, data.hidden_size
    )
    runtime = ControllerOnlyRuntime(controller_path)
    started = time.perf_counter()
    result = runtime.run_sequence_provider(
        initial, token, loaded_provider, initial_is_normalized=True
    )
    elapsed = time.perf_counter() - started
    error = result.stage_states - target
    stage_mse = np.mean(np.square(error), axis=(0, 1, 3))
    report = {
        "experiment": "controller_sequence_trace_replay",
        "status": "replay_boundary",
        "trace": str(trace_path),
        "trace_manifest_sha256": sha256_file(trace_path / "manifest.json"),
        "provider": str(provider_path),
        "provider_sha256": _directory_sha256(provider_path),
        "provider_metadata": loaded_provider.metadata(),
        "controller": str(controller_path),
        "controller_sha256": _directory_sha256(controller_path),
        "sequences": len(unique),
        "sequence_length": counts[0],
        "records": data.records,
        "hidden_size": data.hidden_size,
        "num_stages": data.num_stages,
        "elapsed_seconds": elapsed,
        "inference_device": "cpu",
        "transformers_model_loaded": False,
        "decoder_layer_forward_calls": 0,
        "metrics": {
            "mean_stage_normalized_mse": float(np.mean(stage_mse)),
            "maximum_stage_normalized_mse": float(np.max(stage_mse)),
            "terminal_normalized_mse": float(stage_mse[-1]),
            "stage_normalized_mse": [float(value) for value in stage_mse],
            "hidden_mse": float(np.mean(np.square(error))),
            "maximum_absolute_error": float(np.max(np.abs(error))),
        },
        "gate": {
            "metric": "terminal_normalized_mse",
            "threshold": 0.0225,
            "passed": bool(stage_mse[-1] <= 0.0225),
            "provider_is_learned": False,
            "layer_free_cpu_replay": True,
            "persistent_sequence_state": True,
            "end_to_end_generation": False,
        },
    }
    atomic_json(Path(out), report)
    return report


def evaluate_controller_stateful_provider_trace(
    trace: str | Path,
    provider: str | Path,
    controller: str | Path,
    *,
    out: str | Path,
    allow_correction: bool = False,
) -> dict[str, Any]:
    """Evaluate any authenticated stateful provider on a sequence trace."""

    trace_path = Path(trace).resolve()
    provider_path = Path(provider).resolve()
    controller_path = Path(controller).resolve()
    data = _load_trajectories(trace_path)
    loaded_provider = load_operator_stream_provider(provider_path)
    if not hasattr(loaded_provider, "reset") or not hasattr(loaded_provider, "begin_token"):
        raise TypeError("stateful provider evaluation requires reset and begin_token")
    provider_metadata = loaded_provider.metadata()
    provider_model_hash = provider_metadata.get("source_model_hash")
    trace_model_hash = data.manifest.get("model_hash")
    if provider_model_hash and trace_model_hash and provider_model_hash != trace_model_hash:
        raise ValueError("provider and evaluation trace source-model hashes differ")
    provider_dataset_hash = provider_metadata.get("source_dataset_hash")
    trace_dataset_hash = data.manifest.get("dataset_hash")
    if provider_dataset_hash and trace_dataset_hash and provider_dataset_hash == trace_dataset_hash:
        raise ValueError("provider training and evaluation dataset hashes must differ")
    unique = np.unique(data.sample_id)
    counts = [int(np.sum(data.sample_id == sample)) for sample in unique]
    if not counts or len(set(counts)) != 1:
        raise ValueError("stateful provider traces must contain equal-length sequences")
    order = np.concatenate([np.flatnonzero(data.sample_id == sample) for sample in unique])
    sequence = counts[0]
    initial = data.teacher_states[order, 0].astype(np.float32).reshape(
        len(unique), sequence, data.hidden_size
    )
    token = data.token_embedding[order].astype(np.float32).reshape(
        len(unique), sequence, data.hidden_size
    )
    target = data.teacher_states[order, 1:].astype(np.float32).reshape(
        len(unique), sequence, data.num_stages, data.hidden_size
    )
    runtime = ControllerOnlyRuntime(controller_path, allow_correction=allow_correction)
    if loaded_provider.state_dim != data.hidden_size:
        raise ValueError("provider width does not match trajectory")
    if loaded_provider.num_stages != data.num_stages:
        raise ValueError("provider stages do not match trajectory")
    started = time.perf_counter()
    result = runtime.run_sequence_provider(
        initial, token, loaded_provider, initial_is_normalized=True
    )
    elapsed = time.perf_counter() - started
    error = result.stage_states - target
    stage_mse = np.mean(np.square(error), axis=(0, 1, 3))
    report = {
        "experiment": "controller_stateful_provider_trace_replay",
        "status": "development_result",
        "trace": str(trace_path),
        "trace_manifest_sha256": sha256_file(trace_path / "manifest.json"),
        "provider": str(provider_path),
        "provider_sha256": _directory_sha256(provider_path),
        "provider_metadata": provider_metadata,
        "controller": str(controller_path),
        "controller_sha256": _directory_sha256(controller_path),
        "sequences": len(unique),
        "sequence_length": sequence,
        "records": data.records,
        "hidden_size": data.hidden_size,
        "num_stages": data.num_stages,
        "elapsed_seconds": elapsed,
        "inference_device": "cpu",
        "transformers_model_loaded": False,
        "decoder_layer_forward_calls": 0,
        "persistent_sequence_state": True,
        "metrics": {
            "mean_stage_normalized_mse": float(np.mean(stage_mse)),
            "maximum_stage_normalized_mse": float(np.max(stage_mse)),
            "terminal_normalized_mse": float(stage_mse[-1]),
            "stage_normalized_mse": [float(value) for value in stage_mse],
            "hidden_mse": float(np.mean(np.square(error))),
            "maximum_absolute_error": float(np.max(np.abs(error))),
        },
        "gate": {
            "metric": "terminal_normalized_mse",
            "threshold": 0.0225,
            "passed": bool(stage_mse[-1] <= 0.0225),
            "provider_is_learned": bool(provider_metadata.get("learned")),
            "protected_validation": bool(
                provider_dataset_hash
                and trace_dataset_hash
                and provider_dataset_hash != trace_dataset_hash
            ),
            "layer_free_cpu_replay": True,
            "end_to_end_generation": False,
        },
    }
    atomic_json(Path(out), report)
    return report


__all__ = [
    "evaluate_controller_provider_trace",
    "evaluate_controller_sequence_replay",
    "evaluate_controller_stateful_provider_trace",
]
