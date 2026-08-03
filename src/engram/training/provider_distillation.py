"""Joint causal adaptation of a compact operator-stream provider.

This experiment keeps the provider's PCA output bases fixed and optimizes its
state/token projections through the serialized controller in free-running
mode.  It is intentionally separate from the promoted exact-residual path:
the result is a research artifact until an independent causal split passes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from engram.controller import FactorizedRecurrentController
from engram.runtime.operator_stream import (
    PCAOperatorStreamProvider,
    _directory_sha256,
)
from engram.training.controller_distillation import _load_trajectories, _torch_controller_class
from engram.utils import atomic_json


def _torch_provider_step(torch, tensors, state, token, stage):
    features = torch.cat(
        (state, token, torch.ones((*state.shape[:-1], 1), device=state.device)),
        dim=-1,
    )
    semantic = tensors["semantic_mean"][stage] + (
        features @ tensors["semantic_projection"][stage]
    ) @ tensors["semantic_basis"][stage]
    episodic = tensors["episodic_mean"][stage] + (
        features @ tensors["episodic_projection"][stage]
    ) @ tensors["episodic_basis"][stage]
    return semantic, episodic


def _rollout(torch, module, tensors, data, indices, device):
    targets = torch.as_tensor(data.teacher_states[indices], dtype=torch.float32, device=device)
    token = torch.as_tensor(data.token_embedding[indices], dtype=torch.float32, device=device)
    state = targets[:, 0]
    stage_errors = []
    for stage in range(data.num_stages):
        semantic, episodic = _torch_provider_step(torch, tensors, state, token, stage)
        supplied = torch.cat((token, semantic, episodic), dim=-1)
        state = module.step(state, supplied, stage)
        target = targets[:, stage + 1]
        rms = target.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
        stage_errors.append(((state - target) / rms).square().mean())
    return torch.stack(stage_errors).mean(), stage_errors, state


def _evaluate(torch, module, tensors, data, device, batch_size):
    module.eval()
    totals = np.zeros(data.num_stages, dtype=np.float64)
    records = 0
    with torch.inference_mode():
        for start in range(0, data.records, batch_size):
            indices = np.arange(start, min(start + batch_size, data.records))
            _loss, errors, _state = _rollout(torch, module, tensors, data, indices, device)
            count = len(indices)
            totals += np.asarray([float(value.item()) for value in errors]) * count
            records += count
    module.train()
    stage = totals / max(records, 1)
    return {
        "mean_stage_normalized_mse": float(stage.mean()),
        "terminal_normalized_mse": float(stage[-1]),
        "maximum_stage_normalized_mse": float(stage.max()),
        "stage_normalized_mse": [float(value) for value in stage],
    }


def joint_distill_operator_provider(
    provider: str | Path,
    controller: str | Path,
    trace: str | Path,
    out: str | Path,
    *,
    validation_trace: str | Path | None = None,
    steps: int = 100,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    seed: int = 37,
    device: str = "cpu",
) -> dict[str, Any]:
    """Optimize provider projections through a frozen controller rollout."""

    if steps < 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("steps, batch_size, and learning_rate must be positive")
    if device.startswith("cuda"):
        raise ValueError("provider adaptation is currently CPU-only")
    training = _load_trajectories(trace)
    validation = _load_trajectories(validation_trace) if validation_trace else training
    provider_obj = PCAOperatorStreamProvider.load(provider)
    controller_obj = FactorizedRecurrentController.load(controller)
    if provider_obj.state_dim != training.hidden_size or provider_obj.num_stages != training.num_stages:
        raise ValueError("provider and training trace dimensions differ")
    if controller_obj.state_dim != training.hidden_size or controller_obj.num_stages != training.num_stages:
        raise ValueError("controller and training trace dimensions differ")
    if validation.hidden_size != training.hidden_size or validation.num_stages != training.num_stages:
        raise ValueError("validation trace dimensions differ")

    torch, TorchFactorizedController = _torch_controller_class()
    torch.manual_seed(seed)
    module = TorchFactorizedController(controller_obj).to(device).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    tensors = {}
    trainable = []
    for name in (
        "semantic_projection",
        "episodic_projection",
    ):
        parameter = torch.nn.Parameter(
            torch.from_numpy(getattr(provider_obj, name).copy()).to(device)
        )
        tensors[name] = parameter
        trainable.append(parameter)
    for name in (
        "semantic_mean",
        "semantic_basis",
        "episodic_mean",
        "episodic_basis",
    ):
        tensors[name] = torch.as_tensor(
            getattr(provider_obj, name).copy(), dtype=torch.float32, device=device
        )
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    initial_training = _evaluate(torch, module, tensors, training, device, batch_size)
    initial_validation = _evaluate(torch, module, tensors, validation, device, batch_size)
    rng = np.random.default_rng(seed)
    history = []
    started = time.perf_counter()
    for step in range(steps):
        indices = rng.choice(training.records, size=batch_size, replace=training.records < batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss, errors, _state = _rollout(torch, module, tensors, training, indices, device)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"provider adaptation loss became non-finite at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        if step in {0, steps - 1} or (step + 1) % max(steps // 10, 1) == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach().item()),
                    "terminal_normalized_mse": float(errors[-1].detach().item()),
                }
            )
    final_training = _evaluate(torch, module, tensors, training, device, batch_size)
    final_validation = _evaluate(torch, module, tensors, validation, device, batch_size)
    arrays = {
        "semantic_mean": provider_obj.semantic_mean,
        "semantic_basis": provider_obj.semantic_basis,
        "semantic_projection": tensors["semantic_projection"].detach().cpu().numpy(),
        "episodic_mean": provider_obj.episodic_mean,
        "episodic_basis": provider_obj.episodic_basis,
        "episodic_projection": tensors["episodic_projection"].detach().cpu().numpy(),
    }
    adapted = PCAOperatorStreamProvider(
        **arrays,
        metadata_payload={
            **provider_obj.metadata_payload,
            "adapted_from": str(Path(provider).resolve()),
            "adaptation": "joint_free_running_provider_projection",
            "adaptation_steps": steps,
            "adaptation_batch_size": batch_size,
            "adaptation_learning_rate": learning_rate,
            "adaptation_seed": seed,
        },
    )
    target = Path(out)
    adapted.save(target)
    report = {
        "experiment": "joint_operator_provider_controller_adaptation",
        "status": "development_result",
        "provider": str(Path(provider).resolve()),
        "adapted_provider": str(target.resolve()),
        "adapted_provider_sha256": _directory_sha256(target),
        "controller": str(Path(controller).resolve()),
        "training_trace": str(Path(trace).resolve()),
        "validation_trace": str(Path(validation_trace).resolve()) if validation_trace else None,
        "optimizer_device": device,
        "inference_device": "cpu",
        "transformers_model_loaded": False,
        "decoder_layer_forward_calls": 0,
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "elapsed_seconds": time.perf_counter() - started,
            "history": history,
            "initial": initial_training,
            "final": final_training,
        },
        "initial_validation": initial_validation,
        "validation": final_validation,
        "gate": {
            "metric": "terminal_normalized_mse",
            "threshold": 0.0225,
            "actual": final_validation["terminal_normalized_mse"],
            "passed": bool(final_validation["terminal_normalized_mse"] <= 0.0225),
            "layer_free_cpu_replay": True,
            "end_to_end_generation": False,
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target.parent / f"{target.name}_training_report.json", report)
    return report


__all__ = ["joint_distill_operator_provider"]
