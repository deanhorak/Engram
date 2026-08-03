"""Held-out evaluation for the transformer-free controller runtime."""

from __future__ import annotations

import time
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from engram.runtime.controller_only import ControllerOnlyRuntime
from engram.training.controller_distillation import _load_trajectories
from engram.utils import atomic_json, sha256_file


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def evaluate_controller_only_trace(
    trace: str | Path,
    controller: str | Path,
    *,
    out: str | Path,
    allow_correction: bool = False,
) -> dict[str, Any]:
    """Replay a controller against a frozen trajectory without model layers.

    The trace contains the operator streams and normalized teacher boundaries;
    it is therefore a state-transition evaluation, not a claim of end-to-end
    generation.  The function records that no Transformers model was loaded so
    the CPU-only boundary remains auditable.
    """

    trace_path = Path(trace).resolve()
    controller_path = Path(controller).resolve()
    data = _load_trajectories(trace_path)
    runtime = ControllerOnlyRuntime(
        controller_path,
        allow_correction=allow_correction,
    )
    if runtime.controller.state_dim != data.hidden_size:
        raise ValueError("controller width does not match trajectory width")
    if runtime.controller.num_stages != data.num_stages:
        raise ValueError("controller stage count does not match trajectory")

    started = time.perf_counter()
    result = runtime.run(
        data.teacher_states[:, 0].astype(np.float32),
        data.semantic_outputs.astype(np.float32),
        data.episodic_outputs.astype(np.float32),
        initial_is_normalized=True,
    )
    elapsed = time.perf_counter() - started
    target = data.teacher_states[:, 1:].astype(np.float32)
    error = result.stage_states - target
    stage_mse = np.mean(np.square(error), axis=(0, 2))
    terminal_mse = float(stage_mse[-1])
    report = {
        "experiment": "controller_only_trace_replay",
        "status": "development_result",
        "trace": str(trace_path),
        "trace_sha256": sha256_file(trace_path / "manifest.json"),
        "controller": str(controller_path),
        "controller_sha256": _directory_sha256(controller_path),
        "controller_metadata": runtime.controller.metadata(),
        "records": data.records,
        "hidden_size": data.hidden_size,
        "num_stages": data.num_stages,
        "elapsed_seconds": elapsed,
        "inference_device": "cpu",
        "transformers_model_loaded": False,
        "decoder_layer_forward_calls": 0,
        "operator_streams": {
            "semantic": "trace.semantic_outputs",
            "episodic": "trace.episodic_outputs",
            "initial_state": "trace.teacher_states[...,0,:]",
        },
        "metrics": {
            "mean_stage_normalized_mse": float(np.mean(stage_mse)),
            "maximum_stage_normalized_mse": float(np.max(stage_mse)),
            "terminal_normalized_mse": terminal_mse,
            "stage_normalized_mse": [float(value) for value in stage_mse],
            "maximum_absolute_error": float(np.max(np.abs(error))),
        },
        "gate": {
            "metric": "terminal_normalized_mse",
            "threshold": 0.0225,
            "passed": terminal_mse <= 0.0225,
            "exact_operator_residual": bool(
                runtime.controller.has_operator_residual
                and not np.any(runtime.controller.step_scale != 0.0)
            ),
            "layer_free_cpu_replay": True,
            "end_to_end_generation": False,
        },
    }
    atomic_json(Path(out), report)
    return report


__all__ = ["evaluate_controller_only_trace"]
