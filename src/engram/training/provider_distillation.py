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
    NonlinearResidualOperatorStreamProvider,
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


def distill_state_space_operator_provider(
    provider: str | Path,
    controller: str | Path,
    trace: str | Path,
    out: str | Path,
    *,
    validation_trace: str | Path | None = None,
    steps: int = 80,
    batch_size: int = 8,
    memory_dim: int = 64,
    projection_width: int = 64,
    learning_rate: float = 2e-3,
    seed: int = 81,
    device: str = "cpu",
) -> dict[str, Any]:
    """Distill a causal diagonal state-space operator provider.

    The provider is initialized to the best linear PCA provider, then its
    token memory, decay, and low-rank stage heads are optimized through the
    frozen exact controller in free-running sequence mode.  The serialized
    result contains NumPy tensors only and is never promoted automatically.
    """

    if steps < 0 or batch_size <= 0 or memory_dim <= 0 or projection_width <= 0:
        raise ValueError("steps, batch_size, memory_dim, and projection_width must be positive")
    if learning_rate <= 0.0 or not np.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    if device.startswith("cuda") and not __import__("torch").cuda.is_available():
        raise RuntimeError("state-space provider CUDA training requested but unavailable")
    from engram.runtime.operator_stream import StateSpaceOperatorStreamProvider

    training = _load_trajectories(trace)
    validation = _load_trajectories(validation_trace) if validation_trace else training
    base = PCAOperatorStreamProvider.load(provider)
    controller_obj = FactorizedRecurrentController.load(controller)
    if base.state_dim != training.hidden_size or base.num_stages != training.num_stages:
        raise ValueError("provider and trace dimensions differ")
    if controller_obj.state_dim != training.hidden_size or controller_obj.num_stages != training.num_stages:
        raise ValueError("controller and trace dimensions differ")
    if validation.hidden_size != training.hidden_size or validation.num_stages != training.num_stages:
        raise ValueError("validation trace dimensions differ")
    torch, TorchFactorizedController = _torch_controller_class()
    if not device.startswith("cuda"):
        torch.set_num_threads(min(4, torch.get_num_threads()))
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    def pack(data):
        sample_ids = np.unique(data.sample_id)
        counts = [int(np.sum(data.sample_id == sample)) for sample in sample_ids]
        if not counts or len(set(counts)) != 1:
            raise ValueError("state-space traces must contain equal-length sequences")
        order = np.concatenate(
            [np.flatnonzero(data.sample_id == sample) for sample in sample_ids]
        )
        sequence = counts[0]
        return (
            data.teacher_states[order].astype(np.float32).reshape(
                len(sample_ids), sequence, data.num_stages + 1, data.hidden_size
            ),
            data.token_embedding[order].astype(np.float32).reshape(
                len(sample_ids), sequence, data.hidden_size
            ),
        )

    train_states, train_tokens = pack(training)
    validation_states, validation_tokens = pack(validation)
    width = training.hidden_size
    stages = training.num_stages
    rank = base.output_rank
    state_projection = (rng.normal(size=(width, projection_width)) / np.sqrt(projection_width)).astype(np.float32)
    token_projection = (rng.normal(size=(width, projection_width)) / np.sqrt(projection_width)).astype(np.float32)
    stage_head = np.empty((stages, 2 * projection_width + memory_dim + 1, 2 * rank), dtype=np.float32)
    for stage in range(stages):
        stage_state = training.teacher_states[:, stage].astype(np.float32)
        stage_token = training.token_embedding.astype(np.float32)
        features = np.concatenate(
            (
                stage_state @ state_projection,
                stage_token @ token_projection,
                np.zeros((training.records, memory_dim), dtype=np.float32),
                np.ones((training.records, 1), dtype=np.float32),
            ),
            axis=-1,
        )
        target = np.concatenate(
            (
                (training.semantic_outputs[:, stage].astype(np.float32) - base.semantic_mean[stage]) @ base.semantic_basis[stage].T,
                (training.episodic_outputs[:, stage].astype(np.float32) - base.episodic_mean[stage]) @ base.episodic_basis[stage].T,
            ),
            axis=-1,
        )
        stage_head[stage] = np.linalg.lstsq(features, target, rcond=1e-4)[0]

    torch_controller = TorchFactorizedController(controller_obj).to(device).eval()
    for parameter in torch_controller.parameters():
        parameter.requires_grad_(False)
    state_projection_t = torch.as_tensor(state_projection, dtype=torch.float32, device=device)
    token_projection_t = torch.as_tensor(token_projection, dtype=torch.float32, device=device)
    semantic_basis_t = torch.as_tensor(np.array(base.semantic_basis, copy=True), dtype=torch.float32, device=device)
    episodic_basis_t = torch.as_tensor(np.array(base.episodic_basis, copy=True), dtype=torch.float32, device=device)
    semantic_mean_t = torch.as_tensor(np.array(base.semantic_mean, copy=True), dtype=torch.float32, device=device)
    episodic_mean_t = torch.as_tensor(np.array(base.episodic_mean, copy=True), dtype=torch.float32, device=device)
    memory_input = torch.nn.Parameter(torch.zeros((width, memory_dim), device=device))
    decay = torch.nn.Parameter(torch.full((memory_dim,), 0.8, device=device))
    heads = torch.nn.Parameter(torch.as_tensor(stage_head, dtype=torch.float32, device=device))
    trainable = [memory_input, decay, heads]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)

    def rollout(states, tokens):
        memory = torch.zeros((states.shape[0], memory_dim), dtype=torch.float32, device=device)
        stage_losses = []
        terminal = []
        for position in range(states.shape[1]):
            token = tokens[:, position]
            memory = torch.tanh(memory * decay + token @ memory_input)
            state = states[:, position, 0]
            for stage in range(stages):
                features = torch.cat(
                    (state @ state_projection_t, token @ token_projection_t, memory,
                     torch.ones((states.shape[0], 1), device=device)), dim=-1
                )
                latent = features @ heads[stage]
                semantic = semantic_mean_t[stage] + latent[..., :rank] @ semantic_basis_t[stage]
                episodic = episodic_mean_t[stage] + latent[..., rank:] @ episodic_basis_t[stage]
                state = torch_controller.step(state, torch.cat((token, semantic, episodic), dim=-1), stage)
                target = states[:, position, stage + 1]
                rms = target.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
                stage_losses.append(((state - target) / rms).square().mean())
            terminal.append(state)
        return torch.stack(stage_losses).mean(), terminal

    def evaluate(states_np, tokens_np):
        states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
        tokens = torch.as_tensor(tokens_np, dtype=torch.float32, device=device)
        totals = []
        with torch.inference_mode():
            for start in range(0, states.shape[0], batch_size):
                _, terminal = rollout(states[start : start + batch_size], tokens[start : start + batch_size])
                for offset, final in enumerate(terminal):
                    target = states[start : start + batch_size, offset, -1]
                    rms = target.square().mean(dim=-1).clamp_min(1e-8)
                    totals.append(((final - target).square().mean(dim=-1) / rms).detach().cpu().numpy())
        values = np.concatenate(totals)
        return {"terminal_normalized_mse": float(values.mean()), "maximum_terminal_normalized_mse": float(values.max())}

    initial_validation = evaluate(validation_states, validation_tokens)
    started = time.perf_counter()
    history = []
    for step in range(steps):
        indices = rng.choice(train_states.shape[0], size=batch_size, replace=train_states.shape[0] < batch_size)
        states = torch.as_tensor(train_states[indices], dtype=torch.float32, device=device)
        tokens = torch.as_tensor(train_tokens[indices], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = rollout(states, tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        if step == 0 or step == steps - 1 or (step + 1) % max(steps // 4, 1) == 0:
            history.append({"step": step + 1, "loss": float(loss.detach().cpu())})
    final_validation = evaluate(validation_states, validation_tokens)
    final_training = evaluate(train_states, train_tokens)
    trained = StateSpaceOperatorStreamProvider(
        memory_input=memory_input.detach().cpu().numpy(),
        decay=decay.detach().cpu().numpy(),
        state_projection=state_projection,
        token_projection=token_projection,
        stage_head=heads.detach().cpu().numpy(),
        semantic_mean=base.semantic_mean,
        semantic_basis=base.semantic_basis,
        episodic_mean=base.episodic_mean,
        episodic_basis=base.episodic_basis,
        metadata_payload={
            "source_provider": str(Path(provider).resolve()),
            "source_provider_sha256": _directory_sha256(Path(provider).resolve()),
            "training_trace": str(Path(trace).resolve()),
            "validation_trace": str(Path(validation_trace).resolve()) if validation_trace else None,
            "architecture": "diagonal_state_space_plus_stage_pca_heads",
            "learned": True,
            "training_steps": steps,
            "training_batch_size": batch_size,
            "training_seed": seed,
            "optimizer_device": device,
        },
    )
    target = Path(out)
    trained.save(target)
    report = {
        "experiment": "distill_state_space_operator_provider",
        "status": "development_result",
        "provider": str(Path(provider).resolve()),
        "provider_sha256": _directory_sha256(target),
        "controller": str(Path(controller).resolve()),
        "training_trace": str(Path(trace).resolve()),
        "validation_trace": str(Path(validation_trace).resolve()) if validation_trace else None,
        "optimizer_device": device,
        "inference_device": "cpu",
        "transformers_model_loaded": False,
        "decoder_layer_forward_calls": 0,
        "architecture": {"memory_dim": memory_dim, "projection_width": projection_width, "output_rank": rank},
        "training": {"steps": steps, "batch_size": batch_size, "seed": seed, "elapsed_seconds": time.perf_counter() - started, "history": history, "final": final_training},
        "initial_validation": initial_validation,
        "validation": final_validation,
        "gate": {"metric": "terminal_normalized_mse", "threshold": 0.0225, "actual": final_validation["terminal_normalized_mse"], "passed": bool(final_validation["terminal_normalized_mse"] <= 0.0225), "layer_free_cpu_replay": True, "end_to_end_generation": False},
    }
    atomic_json(target.parent / f"{target.name}_training_report.json", report)
    return report


def distill_state_space_residual_provider(
    provider: str | Path,
    controller: str | Path,
    trace: str | Path,
    out: str | Path,
    *,
    validation_trace: str | Path | None = None,
    steps: int = 40,
    batch_size: int = 8,
    memory_dim: int = 64,
    learning_rate: float = 2e-3,
    seed: int = 91,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train a persistent-memory residual over a full PCA provider."""

    if steps < 0 or batch_size <= 0 or memory_dim <= 0:
        raise ValueError("steps, batch_size, and memory_dim must be positive")
    if learning_rate <= 0.0 or not np.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    torch, TorchFactorizedController = _torch_controller_class()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("residual provider CUDA training requested but unavailable")
    if not device.startswith("cuda"):
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    from engram.runtime.operator_stream import ResidualStateSpaceOperatorStreamProvider

    training = _load_trajectories(trace)
    validation = _load_trajectories(validation_trace) if validation_trace else training
    base = PCAOperatorStreamProvider.load(provider)
    controller_obj = FactorizedRecurrentController.load(controller)
    if base.state_dim != training.hidden_size or base.num_stages != training.num_stages:
        raise ValueError("provider and trace dimensions differ")
    if controller_obj.state_dim != training.hidden_size or controller_obj.num_stages != training.num_stages:
        raise ValueError("controller and trace dimensions differ")

    def pack(data):
        sample_ids = np.unique(data.sample_id)
        counts = [int(np.sum(data.sample_id == sample)) for sample in sample_ids]
        if not counts or len(set(counts)) != 1:
            raise ValueError("residual provider traces must contain equal-length sequences")
        order = np.concatenate([np.flatnonzero(data.sample_id == sample) for sample in sample_ids])
        sequence = counts[0]
        return (
            data.teacher_states[order].astype(np.float32).reshape(
                len(sample_ids), sequence, data.num_stages + 1, data.hidden_size
            ),
            data.token_embedding[order].astype(np.float32).reshape(
                len(sample_ids), sequence, data.hidden_size
            ),
        )

    train_states, train_tokens = pack(training)
    validation_states, validation_tokens = pack(validation)
    width = training.hidden_size
    stages = training.num_stages
    rank = base.output_rank
    torch_controller = TorchFactorizedController(controller_obj).to(device).eval()
    for parameter in torch_controller.parameters():
        parameter.requires_grad_(False)
    semantic_mean = torch.as_tensor(np.array(base.semantic_mean, copy=True), dtype=torch.float32, device=device)
    semantic_basis = torch.as_tensor(np.array(base.semantic_basis, copy=True), dtype=torch.float32, device=device)
    semantic_projection = torch.as_tensor(np.array(base.semantic_projection, copy=True), dtype=torch.float32, device=device)
    episodic_mean = torch.as_tensor(np.array(base.episodic_mean, copy=True), dtype=torch.float32, device=device)
    episodic_basis = torch.as_tensor(np.array(base.episodic_basis, copy=True), dtype=torch.float32, device=device)
    episodic_projection = torch.as_tensor(np.array(base.episodic_projection, copy=True), dtype=torch.float32, device=device)
    memory_input = torch.nn.Parameter(torch.zeros((width, memory_dim), device=device))
    decay = torch.nn.Parameter(torch.full((memory_dim,), 0.8, device=device))
    correction_head = torch.nn.Parameter(torch.zeros((stages, memory_dim + 1, 2 * rank), device=device))
    trainable = [memory_input, decay, correction_head]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)

    def rollout(states, tokens):
        memory = torch.zeros((states.shape[0], memory_dim), dtype=torch.float32, device=device)
        losses = []
        terminals = []
        for position in range(states.shape[1]):
            token = tokens[:, position]
            memory = torch.tanh(memory * decay + token @ memory_input)
            state = states[:, position, 0]
            context = torch.cat((memory, torch.ones((states.shape[0], 1), device=device)), dim=-1)
            for stage in range(stages):
                base_features = torch.cat((state, token, torch.ones((states.shape[0], 1), device=device)), dim=-1)
                semantic_latent = base_features @ semantic_projection[stage] + context @ correction_head[stage, :, :rank]
                episodic_latent = base_features @ episodic_projection[stage] + context @ correction_head[stage, :, rank:]
                semantic = semantic_mean[stage] + semantic_latent @ semantic_basis[stage]
                episodic = episodic_mean[stage] + episodic_latent @ episodic_basis[stage]
                state = torch_controller.step(state, torch.cat((token, semantic, episodic), dim=-1), stage)
                target = states[:, position, stage + 1]
                rms = target.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
                losses.append(((state - target) / rms).square().mean())
            terminals.append(state)
        return torch.stack(losses).mean(), terminals

    def evaluate(states_np, tokens_np):
        states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
        tokens = torch.as_tensor(tokens_np, dtype=torch.float32, device=device)
        values = []
        with torch.inference_mode():
            for start in range(0, states.shape[0], batch_size):
                _, terminals = rollout(states[start : start + batch_size], tokens[start : start + batch_size])
                for offset, final in enumerate(terminals):
                    target = states[start : start + batch_size, offset, -1]
                    values.append((final - target).square().mean(dim=-1).div(target.square().mean(dim=-1).clamp_min(1e-8)).cpu().numpy())
        values = np.concatenate(values)
        return {"terminal_normalized_mse": float(values.mean()), "maximum_terminal_normalized_mse": float(values.max())}

    initial_validation = evaluate(validation_states, validation_tokens)
    started = time.perf_counter()
    history = []
    rng = np.random.default_rng(seed)
    for step in range(steps):
        indices = rng.choice(train_states.shape[0], size=batch_size, replace=train_states.shape[0] < batch_size)
        states = torch.as_tensor(train_states[indices], dtype=torch.float32, device=device)
        tokens = torch.as_tensor(train_tokens[indices], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = rollout(states, tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        if step == 0 or step == steps - 1 or (step + 1) % max(steps // 4, 1) == 0:
            history.append({"step": step + 1, "loss": float(loss.detach().cpu())})
    final_validation = evaluate(validation_states, validation_tokens)
    final_training = evaluate(train_states, train_tokens)
    trained = ResidualStateSpaceOperatorStreamProvider(
        base_provider=base,
        memory_input=memory_input.detach().cpu().numpy(),
        decay=decay.detach().cpu().numpy(),
        correction_head=correction_head.detach().cpu().numpy(),
        metadata_payload={
            "source_provider": str(Path(provider).resolve()),
            "source_provider_sha256": _directory_sha256(Path(provider).resolve()),
            "training_trace": str(Path(trace).resolve()),
            "validation_trace": str(Path(validation_trace).resolve()) if validation_trace else None,
            "architecture": "full_pca_provider_plus_diagonal_state_space_residual",
            "learned": True,
            "training_steps": steps,
            "training_batch_size": batch_size,
            "training_seed": seed,
            "optimizer_device": device,
        },
    )
    target = Path(out)
    trained.save(target)
    report = {
        "experiment": "distill_state_space_residual_provider",
        "status": "development_result",
        "provider": str(Path(provider).resolve()),
        "provider_sha256": _directory_sha256(target),
        "controller": str(Path(controller).resolve()),
        "training_trace": str(Path(trace).resolve()),
        "validation_trace": str(Path(validation_trace).resolve()) if validation_trace else None,
        "optimizer_device": device,
        "inference_device": "cpu",
        "transformers_model_loaded": False,
        "decoder_layer_forward_calls": 0,
        "architecture": {"memory_dim": memory_dim, "output_rank": rank, "base_provider_ridge": base.metadata().get("ridge")},
        "training": {"steps": steps, "batch_size": batch_size, "seed": seed, "elapsed_seconds": time.perf_counter() - started, "history": history, "final": final_training},
        "initial_validation": initial_validation,
        "validation": final_validation,
        "gate": {"metric": "terminal_normalized_mse", "threshold": 0.0225, "actual": final_validation["terminal_normalized_mse"], "passed": bool(final_validation["terminal_normalized_mse"] <= 0.0225), "layer_free_cpu_replay": True, "end_to_end_generation": False},
    }
    atomic_json(target.parent / f"{target.name}_training_report.json", report)
    return report


def adapt_controller_correction_for_provider(
    provider: str | Path,
    controller: str | Path,
    trace: str | Path,
    out: str | Path,
    *,
    validation_trace: str | Path | None = None,
    steps: int = 50,
    batch_size: int = 8,
    learning_rate: float = 2e-3,
    seed: int = 55,
    device: str = "cpu",
) -> dict[str, Any]:
    """Adapt only the factorized controller correction over fixed provider streams."""

    if steps < 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("steps, batch_size, and learning_rate must be positive")
    torch, TorchFactorizedController = _torch_controller_class()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("controller correction CUDA training requested but unavailable")
    if not device.startswith("cuda"):
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    training = _load_trajectories(trace)
    validation = _load_trajectories(validation_trace) if validation_trace else training
    loaded_provider = PCAOperatorStreamProvider.load(provider)
    controller_obj = FactorizedRecurrentController.load(controller)
    if loaded_provider.state_dim != training.hidden_size or loaded_provider.num_stages != training.num_stages:
        raise ValueError("provider and trace dimensions differ")
    if controller_obj.state_dim != training.hidden_size or controller_obj.num_stages != training.num_stages:
        raise ValueError("controller and trace dimensions differ")
    if validation.hidden_size != training.hidden_size or validation.num_stages != training.num_stages:
        raise ValueError("validation trace dimensions differ")
    module = TorchFactorizedController(controller_obj).to(device).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    trainable_names = (
        "step_scale",
        "adapter_down",
        "adapter_up",
        "stage_embeddings",
        "operator_residual_scale",
    )
    trainable = []
    for name in trainable_names:
        parameter = getattr(module, name)
        parameter.requires_grad_(True)
        trainable.append(parameter)
    tensors = {
        name: torch.as_tensor(
            np.array(getattr(loaded_provider, name), copy=True),
            dtype=torch.float32,
            device=device,
        )
        for name in (
            "semantic_mean",
            "semantic_basis",
            "semantic_projection",
            "episodic_mean",
            "episodic_basis",
            "episodic_projection",
        )
    }
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)

    def rollout(data, indices):
        target = torch.as_tensor(data.teacher_states[indices], dtype=torch.float32, device=device)
        token = torch.as_tensor(data.token_embedding[indices], dtype=torch.float32, device=device)
        state = target[:, 0]
        losses = []
        for stage in range(data.num_stages):
            features = torch.cat((state, token, torch.ones((len(indices), 1), device=device)), dim=-1)
            semantic = tensors["semantic_mean"][stage] + (features @ tensors["semantic_projection"][stage]) @ tensors["semantic_basis"][stage]
            episodic = tensors["episodic_mean"][stage] + (features @ tensors["episodic_projection"][stage]) @ tensors["episodic_basis"][stage]
            state = module.step(state, torch.cat((token, semantic, episodic), dim=-1), stage)
            rms = target[:, stage + 1].square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
            losses.append(((state - target[:, stage + 1]) / rms).square().mean())
        return torch.stack(losses).mean(), state

    def evaluate(data):
        values = []
        stage_values = np.zeros(data.num_stages, dtype=np.float64)
        records = 0
        with torch.inference_mode():
            for start in range(0, data.records, batch_size):
                indices = np.arange(start, min(start + batch_size, data.records))
                _, state = rollout(data, indices)
                target = torch.as_tensor(data.teacher_states[indices, 1:], dtype=torch.float32, device=device)
                # The rollout state is terminal; the stage trajectory is
                # recomputed only for the scalar gate to keep this evaluator
                # consistent with evaluate-controller-provider.
                terminal_rms = target[:, -1].square().mean(dim=-1).clamp_min(1e-8)
                values.append((state - target[:, -1]).square().mean(dim=-1).div(terminal_rms).cpu().numpy())
                records += len(indices)
        terminal = np.concatenate(values)
        return {
            "terminal_normalized_mse": float(terminal.mean()),
            "maximum_terminal_normalized_mse": float(terminal.max()),
            "records": records,
        }

    initial_validation = evaluate(validation)
    initial_training = evaluate(training)
    started = time.perf_counter()
    history = []
    rng = np.random.default_rng(seed)
    for step in range(steps):
        indices = rng.choice(training.records, size=batch_size, replace=training.records < batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = rollout(training, indices)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        if step == 0 or step == steps - 1 or (step + 1) % max(steps // 5, 1) == 0:
            history.append({"step": step + 1, "loss": float(loss.detach().cpu())})
    final_validation = evaluate(validation)
    final_training = evaluate(training)
    adapted = module.to("cpu").export()
    target = Path(out)
    adapted.save(target)
    report = {
        "experiment": "adapt_controller_correction_for_provider",
        "status": "development_result",
        "provider": str(Path(provider).resolve()),
        "provider_sha256": _directory_sha256(Path(provider).resolve()),
        "controller": str(Path(controller).resolve()),
        "adapted_controller": str(target.resolve()),
        "adapted_controller_sha256": _directory_sha256(target),
        "training_trace": str(Path(trace).resolve()),
        "validation_trace": str(Path(validation_trace).resolve()) if validation_trace else None,
        "optimizer_device": device,
        "inference_device": "cpu",
        "transformers_model_loaded": False,
        "decoder_layer_forward_calls": 0,
        "trainable_parameters": list(trainable_names),
        "training": {"steps": steps, "batch_size": batch_size, "learning_rate": learning_rate, "seed": seed, "elapsed_seconds": time.perf_counter() - started, "history": history, "initial": initial_training, "final": final_training},
        "initial_validation": initial_validation,
        "validation": final_validation,
        "gate": {"metric": "terminal_normalized_mse", "threshold": 0.0225, "actual": final_validation["terminal_normalized_mse"], "passed": bool(final_validation["terminal_normalized_mse"] <= 0.0225), "layer_free_cpu_replay": True, "end_to_end_generation": False},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target.parent / f"{target.name}_training_report.json", report)
    return report


def dagger_refit_operator_provider(
    provider: str | Path,
    controller: str | Path,
    trace: str | Path,
    out: str | Path,
    *,
    validation_trace: str | Path | None = None,
    iterations: int = 2,
    ridge: float = 1.0,
) -> dict[str, Any]:
    """Refit a provider on states visited by its own causal rollout.

    Each iteration rolls the current provider through the frozen controller,
    then fits the provider projections against the aligned teacher streams at
    those visited states.  This is a small DAgger-style intervention against
    compounding rollout error; it is deliberately an evaluator artifact until
    an independent causal split passes the controller gate.
    """

    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 0:
        raise ValueError("iterations must be a non-negative integer")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    training = _load_trajectories(trace)
    validation = _load_trajectories(validation_trace) if validation_trace else training
    loaded = PCAOperatorStreamProvider.load(provider)
    controller_obj = FactorizedRecurrentController.load(controller)
    for data in (training, validation):
        if data.hidden_size != loaded.state_dim or data.num_stages != loaded.num_stages:
            raise ValueError("provider and trajectory dimensions differ")
    if controller_obj.state_dim != training.hidden_size or controller_obj.num_stages != training.num_stages:
        raise ValueError("controller and training trace dimensions differ")

    def rollout(data, current):
        state = data.teacher_states[:, 0].astype(np.float32).copy()
        token = data.token_embedding.astype(np.float32)
        states = [state.copy()]
        for stage in range(data.num_stages):
            semantic, episodic = current.step(state, token, stage)
            state = controller_obj.step(
                state,
                np.concatenate((token, semantic, episodic), axis=-1),
                stage=stage,
            )
            states.append(state.copy())
        return np.stack(states, axis=1)

    def evaluate(data, current):
        states = rollout(data, current)
        target = data.teacher_states[:, 1:].astype(np.float32)
        rms = np.sqrt(np.mean(np.square(target), axis=-1, keepdims=True)).clip(1e-4)
        error = np.square((states[:, 1:] - target) / rms)
        stage = np.mean(error, axis=(0, 2))
        return {
            "mean_stage_normalized_mse": float(stage.mean()),
            "terminal_normalized_mse": float(stage[-1]),
            "maximum_stage_normalized_mse": float(stage.max()),
            "stage_normalized_mse": [float(value) for value in stage],
        }

    def refit(states, data, current):
        records = data.records
        token = data.token_embedding.astype(np.float32)
        semantic_mean = np.array(current.semantic_mean, copy=True)
        semantic_basis = np.array(current.semantic_basis, copy=True)
        episodic_mean = np.array(current.episodic_mean, copy=True)
        episodic_basis = np.array(current.episodic_basis, copy=True)
        semantic_projection = np.empty_like(current.semantic_projection)
        episodic_projection = np.empty_like(current.episodic_projection)
        visited = states[:, :-1]
        for stage in range(data.num_stages):
            features = np.concatenate(
                (visited[:, stage], token, np.ones((records, 1), dtype=np.float32)),
                axis=-1,
            )
            gram = features @ features.T
            scale = max(float(np.mean(np.diag(gram))), 1.0)
            gram += np.eye(records, dtype=np.float32) * (ridge * scale)
            semantic_latent = (
                data.semantic_outputs[:, stage].astype(np.float32) - semantic_mean[stage]
            ) @ semantic_basis[stage].T
            episodic_latent = (
                data.episodic_outputs[:, stage].astype(np.float32) - episodic_mean[stage]
            ) @ episodic_basis[stage].T
            semantic_projection[stage] = features.T @ np.linalg.solve(
                gram, semantic_latent
            )
            episodic_projection[stage] = features.T @ np.linalg.solve(
                gram, episodic_latent
            )
        return PCAOperatorStreamProvider(
            semantic_mean=semantic_mean,
            semantic_basis=semantic_basis,
            semantic_projection=semantic_projection,
            episodic_mean=episodic_mean,
            episodic_basis=episodic_basis,
            episodic_projection=episodic_projection,
            metadata_payload={
                **current.metadata_payload,
                "adaptation": "dagger_visited_state_refit",
                "adaptation_iterations": iterations,
                "adaptation_ridge": ridge,
            },
        )

    history = []
    current = loaded
    for iteration in range(iterations + 1):
        history.append(
            {
                "iteration": iteration,
                "training": evaluate(training, current),
                "validation": evaluate(validation, current),
            }
        )
        if iteration < iterations:
            current = refit(rollout(training, current), training, current)

    target = Path(out)
    current.save(target)
    final = history[-1]["validation"]
    report = {
        "experiment": "dagger_refit_operator_provider",
        "status": "development_result",
        "provider": str(Path(provider).resolve()),
        "provider_sha256": _directory_sha256(Path(provider).resolve()),
        "adapted_provider": str(target.resolve()),
        "adapted_provider_sha256": _directory_sha256(target),
        "controller": str(Path(controller).resolve()),
        "training_trace": str(Path(trace).resolve()),
        "validation_trace": str(Path(validation_trace).resolve()) if validation_trace else None,
        "iterations": iterations,
        "ridge": ridge,
        "optimizer_device": "cpu",
        "inference_device": "cpu",
        "transformers_model_loaded": False,
        "decoder_layer_forward_calls": 0,
        "history": history,
        "validation": final,
        "gate": {
            "metric": "terminal_normalized_mse",
            "threshold": 0.0225,
            "actual": final["terminal_normalized_mse"],
            "passed": bool(final["terminal_normalized_mse"] <= 0.0225),
            "layer_free_cpu_replay": True,
            "end_to_end_generation": False,
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target.parent / f"{target.name}_training_report.json", report)
    return report


def distill_nonlinear_residual_provider(
    provider: str | Path,
    controller: str | Path,
    trace: str | Path,
    out: str | Path,
    *,
    validation_trace: str | Path | None = None,
    steps: int = 100,
    teacher_forcing_steps: int = 0,
    teacher_forcing_decay_steps: int = 0,
    batch_size: int = 8,
    hidden_width: int = 64,
    stage_width: int = 16,
    learning_rate: float = 3e-4,
    seed: int = 403,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train a shared nonlinear latent residual through free-running rollout."""

    if (
        steps < 0
        or teacher_forcing_steps < 0
        or teacher_forcing_decay_steps < 0
        or batch_size <= 0
        or hidden_width <= 0
        or stage_width <= 0
    ):
        raise ValueError(
            "steps and teacher_forcing_steps must be non-negative; batch and widths positive"
        )
    if teacher_forcing_steps and teacher_forcing_decay_steps:
        raise ValueError(
            "teacher_forcing_steps and teacher_forcing_decay_steps are mutually exclusive"
        )
    if learning_rate <= 0.0 or not np.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    torch, TorchFactorizedController = _torch_controller_class()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("nonlinear provider CUDA training requested but unavailable")
    if not device.startswith("cuda"):
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    training = _load_trajectories(trace)
    validation = _load_trajectories(validation_trace) if validation_trace else training
    base = PCAOperatorStreamProvider.load(provider)
    controller_obj = FactorizedRecurrentController.load(controller)
    for data in (training, validation):
        if data.hidden_size != base.state_dim or data.num_stages != base.num_stages:
            raise ValueError("provider and trajectory dimensions differ")
    if controller_obj.state_dim != training.hidden_size or controller_obj.num_stages != training.num_stages:
        raise ValueError("controller and training trace dimensions differ")

    width = training.hidden_size
    stages = training.num_stages
    rank = base.output_rank
    module = TorchFactorizedController(controller_obj).to(device).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    tensors = {
        name: torch.as_tensor(
            np.array(getattr(base, name), copy=True), dtype=torch.float32, device=device
        )
        for name in (
            "semantic_mean",
            "semantic_basis",
            "semantic_projection",
            "episodic_mean",
            "episodic_basis",
            "episodic_projection",
        )
    }

    class TorchNonlinearResidual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_down = torch.nn.Linear(2 * width + 1, hidden_width)
            self.stage_embedding = torch.nn.Embedding(stages, stage_width)
            self.hidden_up = torch.nn.Linear(hidden_width + stage_width, hidden_width)
            self.output_up = torch.nn.Linear(hidden_width, 2 * rank)
            torch.nn.init.zeros_(self.output_up.weight)
            torch.nn.init.zeros_(self.output_up.bias)

        def forward(self, features, stage):
            ids = torch.full(
                (features.shape[0],), stage, dtype=torch.long, device=features.device
            )
            hidden = torch.nn.functional.silu(self.input_down(features))
            hidden = torch.nn.functional.silu(
                self.hidden_up(torch.cat((hidden, self.stage_embedding(ids)), dim=-1))
            )
            return self.output_up(hidden)

    residual = TorchNonlinearResidual().to(device)
    optimizer = torch.optim.AdamW(
        residual.parameters(), lr=learning_rate, weight_decay=1e-5
    )

    def rollout(data, indices, *, teacher_forcing: float = 0.0):
        target = torch.as_tensor(data.teacher_states[indices], dtype=torch.float32, device=device)
        token = torch.as_tensor(data.token_embedding[indices], dtype=torch.float32, device=device)
        state = target[:, 0]
        errors = []
        for stage in range(stages):
            if teacher_forcing >= 1.0:
                source = target[:, stage]
            elif teacher_forcing <= 0.0:
                source = state
            else:
                mask = torch.rand((state.shape[0], 1), device=device) < teacher_forcing
                source = torch.where(mask, target[:, stage], state)
            features = torch.cat(
                (source, token, torch.ones((len(indices), 1), device=device)), dim=-1
            )
            correction = residual(features, stage)
            semantic = tensors["semantic_mean"][stage] + (
                (features @ tensors["semantic_projection"][stage])
                + correction[:, :rank]
            ) @ tensors["semantic_basis"][stage]
            episodic = tensors["episodic_mean"][stage] + (
                (features @ tensors["episodic_projection"][stage])
                + correction[:, rank:]
            ) @ tensors["episodic_basis"][stage]
            state = module.step(
                source, torch.cat((token, semantic, episodic), dim=-1), stage
            )
            target_state = target[:, stage + 1]
            rms = target_state.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
            errors.append(((state - target_state) / rms).square().mean())
        return torch.stack(errors).mean(), errors

    def evaluate(data):
        totals = np.zeros(stages, dtype=np.float64)
        records = 0
        residual.eval()
        with torch.inference_mode():
            for start in range(0, data.records, batch_size):
                indices = np.arange(start, min(start + batch_size, data.records))
                _loss, errors = rollout(data, indices)
                count = len(indices)
                totals += np.asarray([float(value.item()) for value in errors]) * count
                records += count
        residual.train()
        values = totals / max(records, 1)
        return {
            "mean_stage_normalized_mse": float(values.mean()),
            "terminal_normalized_mse": float(values[-1]),
            "maximum_stage_normalized_mse": float(values.max()),
            "stage_normalized_mse": [float(value) for value in values],
        }

    initial_validation = evaluate(validation)
    initial_training = evaluate(training)
    rng = np.random.default_rng(seed)
    history = []
    started = time.perf_counter()
    total_steps = teacher_forcing_steps + steps
    for step in range(total_steps):
        if teacher_forcing_steps:
            teacher_forcing = 1.0 if step < teacher_forcing_steps else 0.0
        elif teacher_forcing_decay_steps:
            teacher_forcing = max(
                0.0,
                1.0 - (step / float(max(teacher_forcing_decay_steps - 1, 1))),
            )
        else:
            teacher_forcing = 0.0
        indices = rng.choice(
            training.records,
            size=batch_size,
            replace=training.records < batch_size,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, errors = rollout(training, indices, teacher_forcing=teacher_forcing)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonlinear provider loss became non-finite at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(residual.parameters(), max_norm=1.0)
        optimizer.step()
        if (
            step in {0, total_steps - 1, teacher_forcing_steps - 1}
            or (step + 1) % max(total_steps // 5, 1) == 0
        ):
            history.append(
                {
                    "step": step + 1,
                    "phase": (
                        "teacher_forcing"
                        if teacher_forcing >= 1.0
                        else "free_running"
                        if teacher_forcing <= 0.0
                        else "scheduled_sampling"
                    ),
                    "teacher_forcing_probability": float(teacher_forcing),
                    "loss": float(loss.detach().cpu()),
                    "terminal_normalized_mse": float(errors[-1].detach().cpu()),
                }
            )
    final_validation = evaluate(validation)
    final_training = evaluate(training)
    trained = NonlinearResidualOperatorStreamProvider(
        base_provider=base,
        input_down=residual.input_down.weight.detach().cpu().numpy().T,
        input_bias=residual.input_down.bias.detach().cpu().numpy(),
        stage_embedding=residual.stage_embedding.weight.detach().cpu().numpy(),
        hidden_up=residual.hidden_up.weight.detach().cpu().numpy().T,
        hidden_bias=residual.hidden_up.bias.detach().cpu().numpy(),
        output_up=residual.output_up.weight.detach().cpu().numpy().T,
        output_bias=residual.output_up.bias.detach().cpu().numpy(),
        metadata_payload={
            "source_provider": str(Path(provider).resolve()),
            "source_provider_sha256": _directory_sha256(Path(provider).resolve()),
            "training_trace": str(Path(trace).resolve()),
            "validation_trace": str(Path(validation_trace).resolve()) if validation_trace else None,
            "source_model_hash": training.manifest.get("model_hash"),
            "source_dataset_hash": training.manifest.get("dataset_hash"),
            "architecture": "shared_silu_stage_conditioned_latent_residual",
            "learned": True,
            "training_steps": steps,
            "teacher_forcing_steps": teacher_forcing_steps,
            "teacher_forcing_decay_steps": teacher_forcing_decay_steps,
            "training_batch_size": batch_size,
            "training_seed": seed,
            "optimizer_device": device,
        },
    )
    target = Path(out)
    trained.save(target)
    report = {
        "experiment": "distill_nonlinear_residual_provider",
        "status": "development_result",
        "provider": str(Path(provider).resolve()),
        "provider_sha256": _directory_sha256(Path(provider).resolve()),
        "adapted_provider": str(target.resolve()),
        "adapted_provider_sha256": _directory_sha256(target),
        "controller": str(Path(controller).resolve()),
        "training_trace": str(Path(trace).resolve()),
        "validation_trace": str(Path(validation_trace).resolve()) if validation_trace else None,
        "optimizer_device": device,
        "inference_device": "cpu",
        "transformers_model_loaded": False,
        "decoder_layer_forward_calls": 0,
        "architecture": {
            "hidden_width": hidden_width,
            "stage_width": stage_width,
            "output_rank": rank,
        },
        "training": {
            "steps": steps,
            "teacher_forcing_steps": teacher_forcing_steps,
            "teacher_forcing_decay_steps": teacher_forcing_decay_steps,
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


__all__ = [
    "joint_distill_operator_provider",
    "distill_state_space_operator_provider",
    "distill_state_space_residual_provider",
    "adapt_controller_correction_for_provider",
    "dagger_refit_operator_provider",
    "distill_nonlinear_residual_provider",
]
