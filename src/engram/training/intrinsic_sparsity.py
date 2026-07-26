"""Boundary training for exact gate-selected sparse SwiGLU memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.evaluation.intrinsic_sparsity import exact_gate_sparse_traffic
from engram.models.inspection import inspect_model, load_layer_mlp, resolve_model_path
from engram.tracing.format import TraceReader
from engram.utils import atomic_json, percentile, sha256_file


def _load_boundaries(
    reader: TraceReader, layer: int, limit: int | None
) -> tuple[np.ndarray, np.ndarray]:
    fields = [f"layer_{layer}_mlp_input", f"layer_{layer}_mlp_output"]
    inputs: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    count = 0
    for shard in reader.iter_shards(fields):
        hidden = np.asarray(shard[fields[0]], dtype=np.float32)
        target = np.asarray(shard[fields[1]], dtype=np.float32)
        if limit is not None:
            remaining = limit - count
            if remaining <= 0:
                break
            hidden = hidden[:remaining]
            target = target[:remaining]
        inputs.append(hidden)
        outputs.append(target)
        count += len(hidden)
    if not inputs:
        raise ValueError(f"trace contains no MLP boundaries for layer {layer}")
    return np.concatenate(inputs), np.concatenate(outputs)


def _stats(values: np.ndarray) -> dict[str, int | float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError("cannot summarize empty or non-finite values")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": percentile(array, 95),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _quantile(values: Any, sparsity: float, torch: Any) -> Any:
    flattened = values.detach().abs().reshape(-1)
    index = max(0, min(flattened.numel() - 1, int(sparsity * flattened.numel())))
    return torch.kthvalue(flattened, index + 1).values


def _sparse_forward(
    hidden: Any,
    gate: Any,
    up: Any,
    down: Any,
    threshold: Any,
    temperature: float,
    torch: Any,
    *,
    straight_through: bool,
) -> tuple[Any, Any]:
    gate_values = torch.nn.functional.silu(torch.nn.functional.linear(hidden, gate))
    magnitudes = gate_values.abs()
    hard = torch.where(magnitudes > threshold, gate_values, 0.0)
    if straight_through:
        soft_mask = torch.sigmoid((magnitudes - threshold) / temperature)
        soft = gate_values * soft_mask
        selected = hard + soft - soft.detach()
    else:
        soft_mask = (magnitudes > threshold).to(gate_values.dtype)
        selected = hard
    up_values = torch.nn.functional.linear(hidden, up)
    output = torch.nn.functional.linear(selected * up_values, down)
    return output, soft_mask


def _relative_rows(approximation: Any, target: Any, torch: Any) -> Any:
    return torch.linalg.vector_norm(approximation - target, dim=-1) / torch.clamp(
        torch.linalg.vector_norm(target, dim=-1), min=1e-12
    )


def train_intrinsic_sparse_boundaries(
    model: str | Path,
    training_traces: str | Path,
    validation_traces: str | Path,
    out: str | Path,
    *,
    layers: Sequence[int],
    target_sparsity: float = 0.85,
    initial_artifact: str | Path | None = None,
    steps: int = 128,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    sparsity_weight: float = 1.0,
    cosine_weight: float = 0.1,
    temperature_fraction: float = 0.1,
    warmup_steps: int = 16,
    start_threshold_fraction: float = 0.0,
    evaluation_interval: int = 16,
    maximum_mean_relative_l2: float = 0.18,
    maximum_traffic_fraction: float = 0.45,
    max_train_records: int | None = 4096,
    max_validation_records: int | None = 2048,
    seed: int = 1729,
    device: str = "cpu",
) -> dict[str, Any]:
    """Co-adapt selected MLP weights through the exact sparse forward path."""

    if not layers:
        raise ValueError("at least one layer is required")
    if not 0 < target_sparsity < 1:
        raise ValueError("target_sparsity must lie in (0, 1)")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (steps, batch_size, evaluation_interval)
    ):
        raise ValueError("steps, batch_size, and evaluation_interval must be positive")
    if (
        isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or warmup_steps < 0
        or warmup_steps >= steps
    ):
        raise ValueError("warmup_steps must be nonnegative and smaller than steps")
    if (
        not np.isfinite(start_threshold_fraction)
        or not 0 <= start_threshold_fraction <= 1
    ):
        raise ValueError("start_threshold_fraction must lie in [0, 1]")
    if (
        any(
            not np.isfinite(value) or value <= 0
            for value in (
                learning_rate,
                temperature_fraction,
                maximum_mean_relative_l2,
                maximum_traffic_fraction,
            )
        )
        or sparsity_weight < 0
        or cosine_weight < 0
    ):
        raise ValueError("training weights, rates, and thresholds are invalid")

    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for intrinsic sparse training"
        ) from exc

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    selected_layers = tuple(dict.fromkeys(int(layer) for layer in layers))
    if any(
        layer < 0 or layer >= inspection.num_hidden_layers for layer in selected_layers
    ):
        raise ValueError("layer index is outside the model")
    training = TraceReader(training_traces)
    validation = TraceReader(validation_traces)
    if (
        training.manifest["model_hash"] != inspection.source_hash
        or training.manifest["split"] != "calibration"
    ):
        raise ValueError("training trace/model provenance mismatch")
    if (
        validation.manifest["model_hash"] != inspection.source_hash
        or validation.manifest["split"] != "validation"
    ):
        raise ValueError("validation trace/model provenance mismatch")
    if training.manifest["dataset_hash"] == validation.manifest["dataset_hash"]:
        raise ValueError("training and validation datasets must be distinct")

    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    initial_tensors = (
        None
        if initial_artifact is None
        else load_file(str(Path(initial_artifact)), device="cpu")
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    layer_reports: list[dict[str, Any]] = []
    tensors: dict[str, Any] = {}
    for layer in selected_layers:
        train_inputs_np, train_targets_np = _load_boundaries(
            training, layer, max_train_records
        )
        validation_inputs_np, validation_targets_np = _load_boundaries(
            validation, layer, max_validation_records
        )
        gate_np, up_np, down_np = load_layer_mlp(model_path, layer)
        if initial_tensors is not None:
            required = [
                f"layer_{layer}.gate",
                f"layer_{layer}.up",
                f"layer_{layer}.down",
                f"layer_{layer}.threshold",
            ]
            missing = [name for name in required if name not in initial_tensors]
            if missing:
                raise ValueError(
                    f"initial artifact is missing layer {layer} tensors: {missing}"
                )
            gate_np = initial_tensors[required[0]].numpy()
            up_np = initial_tensors[required[1]].numpy()
            down_np = initial_tensors[required[2]].numpy()
        gate = torch.nn.Parameter(
            torch.tensor(gate_np, dtype=torch.float32, device=device)
        )
        up = torch.nn.Parameter(torch.tensor(up_np, dtype=torch.float32, device=device))
        down = torch.nn.Parameter(
            torch.tensor(down_np, dtype=torch.float32, device=device)
        )
        train_inputs = torch.tensor(train_inputs_np, dtype=torch.float32, device=device)
        train_targets = torch.tensor(
            train_targets_np, dtype=torch.float32, device=device
        )
        validation_inputs = torch.tensor(
            validation_inputs_np, dtype=torch.float32, device=device
        )
        validation_targets = torch.tensor(
            validation_targets_np, dtype=torch.float32, device=device
        )
        del (
            train_inputs_np,
            train_targets_np,
            validation_inputs_np,
            validation_targets_np,
            gate_np,
            up_np,
            down_np,
        )
        with torch.no_grad():
            initial_gate = torch.nn.functional.silu(
                torch.nn.functional.linear(train_inputs, gate)
            )
            threshold = (
                initial_tensors[f"layer_{layer}.threshold"]
                .to(device=device, dtype=torch.float32)
                .reshape(())
                if initial_tensors is not None
                else _quantile(initial_gate, target_sparsity, torch)
            )
            temperature = max(
                float(threshold.abs().item()) * temperature_fraction, 1e-4
            )
            del initial_gate

        def evaluate(evaluation_threshold: Any = threshold) -> dict[str, float]:
            with torch.no_grad():
                approximation, mask = _sparse_forward(
                    validation_inputs,
                    gate,
                    up,
                    down,
                    evaluation_threshold,
                    temperature,
                    torch,
                    straight_through=False,
                )
                relative = _relative_rows(approximation, validation_targets, torch)
                cosine = torch.nn.functional.cosine_similarity(
                    approximation, validation_targets, dim=-1
                )
                active_fraction = float(mask.mean().item())
                traffic = exact_gate_sparse_traffic(
                    inspection.hidden_size,
                    inspection.intermediate_size,
                    active_fraction,
                )
                return {
                    "mean_relative_l2": float(relative.mean().item()),
                    "p95_relative_l2": float(torch.quantile(relative, 0.95).item()),
                    "mean_cosine": float(cosine.mean().item()),
                    "active_fraction": active_fraction,
                    "traffic_fraction": float(traffic["fraction_of_dense"]),
                }

        initial = evaluate()
        optimizer = torch.optim.AdamW((gate, up, down), lr=learning_rate)
        history: list[dict[str, float | int]] = [{"step": 0, **initial}]
        for step in range(1, steps + 1):
            if step <= warmup_steps:
                threshold_progress = start_threshold_fraction
            else:
                linear_progress = (step - warmup_steps) / (steps - warmup_steps)
                curve = float(np.sin(0.5 * np.pi * linear_progress) ** 2)
                threshold_progress = float(
                    start_threshold_fraction + (1.0 - start_threshold_fraction) * curve
                )
            current_threshold = threshold * threshold_progress
            current_target_active = 1.0 - target_sparsity * threshold_progress
            indices = torch.randint(
                len(train_inputs),
                (min(batch_size, len(train_inputs)),),
                generator=generator,
            ).to(device)
            hidden = train_inputs[indices]
            reference = train_targets[indices]
            approximation, soft_mask = _sparse_forward(
                hidden,
                gate,
                up,
                down,
                current_threshold,
                temperature,
                torch,
                straight_through=True,
            )
            normalized_mse = torch.mean((approximation - reference) ** 2) / torch.clamp(
                torch.mean(reference**2), min=1e-8
            )
            cosine_loss = (
                1.0
                - torch.nn.functional.cosine_similarity(
                    approximation, reference, dim=-1
                ).mean()
            )
            occupancy_loss = (soft_mask.mean() - current_target_active) ** 2
            loss = (
                normalized_mse
                + cosine_weight * cosine_loss
                + sparsity_weight * occupancy_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % evaluation_interval == 0 or step == steps:
                history.append(
                    {
                        "step": step,
                        "threshold_progress": threshold_progress,
                        "training_threshold": float(current_threshold.detach().item()),
                        "loss": float(loss.detach().item()),
                        "normalized_mse": float(normalized_mse.detach().item()),
                        "occupancy_loss": float(occupancy_loss.detach().item()),
                        **evaluate(),
                    }
                )

        final = evaluate()
        tensors[f"layer_{layer}.gate"] = gate.detach().cpu().contiguous()
        tensors[f"layer_{layer}.up"] = up.detach().cpu().contiguous()
        tensors[f"layer_{layer}.down"] = down.detach().cpu().contiguous()
        tensors[f"layer_{layer}.threshold"] = threshold.detach().cpu().reshape(1)
        layer_reports.append(
            {
                "layer": layer,
                "threshold": float(threshold.item()),
                "temperature": temperature,
                "training_records": len(train_inputs),
                "validation_records": len(validation_inputs),
                "initial": initial,
                "final": final,
                "relative_l2_improvement": (
                    initial["mean_relative_l2"] - final["mean_relative_l2"]
                )
                / max(initial["mean_relative_l2"], 1e-12),
                "history": history,
                "screen": {
                    "local_quality_pass": (
                        final["mean_relative_l2"] <= maximum_mean_relative_l2
                    ),
                    "traffic_pass_before_metadata": (
                        final["traffic_fraction"] <= maximum_traffic_fraction
                    ),
                },
            }
        )

    artifact_path = target / "intrinsic_sparse_boundaries.safetensors"
    save_file(tensors, artifact_path)
    all_pass = all(
        item["screen"]["local_quality_pass"]
        and item["screen"]["traffic_pass_before_metadata"]
        for item in layer_reports
    )
    report = {
        "experiment": "intrinsic_sparse_boundary_training",
        "status": (
            "eligible_for_all_layer_causal_training"
            if all_pass
            else "stop_or_scale_boundary_training"
        ),
        "source": {
            "model_path": str(model_path),
            "model_hash": inspection.source_hash,
            "hidden_size": inspection.hidden_size,
            "intermediate_size": inspection.intermediate_size,
            "layers": inspection.num_hidden_layers,
        },
        "configuration": {
            "selected_layers": list(selected_layers),
            "target_sparsity": target_sparsity,
            "initial_artifact": (
                None
                if initial_artifact is None
                else str(Path(initial_artifact).resolve())
            ),
            "initial_artifact_sha256": (
                None if initial_artifact is None else sha256_file(initial_artifact)
            ),
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "sparsity_weight": sparsity_weight,
            "cosine_weight": cosine_weight,
            "temperature_fraction": temperature_fraction,
            "warmup_steps": warmup_steps,
            "start_threshold_fraction": start_threshold_fraction,
            "maximum_mean_relative_l2": maximum_mean_relative_l2,
            "maximum_traffic_fraction": maximum_traffic_fraction,
            "seed": seed,
            "device": device,
        },
        "provenance": {
            "training_trace": str(Path(training_traces).resolve()),
            "training_dataset_hash": training.manifest["dataset_hash"],
            "validation_trace": str(Path(validation_traces).resolve()),
            "validation_dataset_hash": validation.manifest["dataset_hash"],
        },
        "artifact": {
            "path": str(artifact_path.resolve()),
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256_file(artifact_path),
            "formal_deployment_artifact": False,
        },
        "layers": layer_reports,
        "screen": {"passed": all_pass},
        "scope_caveat": (
            "This is a cached teacher-boundary capacity screen. It is not an "
            "all-layer causal evaluation, independently quantized deployment "
            "artifact, or complete cache-line traffic measurement."
        ),
    }
    atomic_json(target / "intrinsic_sparse_boundary_training.json", report)
    return report


__all__ = ["train_intrinsic_sparse_boundaries"]
