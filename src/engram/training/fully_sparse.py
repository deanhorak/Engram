"""Q-Sparse-style exact top-K MLP boundary training.

Both gate/up projections read only the largest-magnitude coordinates of the
MLP input. The down projection reads only the largest-magnitude intermediate
activations. Selection depends only on already-resident activations, so there
is no learned router, candidate stage, or recall criterion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.models.inspection import inspect_model, load_layer_mlp, resolve_model_path
from engram.tracing.format import TraceReader
from engram.training.intrinsic_sparsity import _load_boundaries
from engram.utils import atomic_json, sha256_file

Q_SPARSE_REFERENCE = {
    "title": "Q-Sparse: All Large Language Models can be Fully Sparsely-Activated",
    "url": "https://arxiv.org/abs/2407.10969",
}


def fully_sparse_mlp_traffic(
    hidden_size: int,
    intermediate_size: int,
    input_count: int,
    intermediate_count: int,
    *,
    bytes_per_weight: float = 0.5,
) -> dict[str, int | float | bool]:
    """Return ideal Q4 traffic for sparse gate/up inputs and sparse down input."""

    values = (hidden_size, intermediate_size, input_count, intermediate_count)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("dimensions and counts must be integers")
    if hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError("dimensions must be positive")
    if not 0 < input_count <= hidden_size:
        raise ValueError("input_count must lie within hidden_size")
    if not 0 < intermediate_count <= intermediate_size:
        raise ValueError("intermediate_count must lie within intermediate_size")
    if not np.isfinite(bytes_per_weight) or bytes_per_weight <= 0:
        raise ValueError("bytes_per_weight must be finite and positive")
    gate_up = 2 * intermediate_size * input_count
    down = hidden_size * intermediate_count
    dense = 3 * hidden_size * intermediate_size
    selected = gate_up + down
    return {
        "bytes_per_weight": float(bytes_per_weight),
        "input_count": input_count,
        "intermediate_count": intermediate_count,
        "input_fraction": input_count / hidden_size,
        "intermediate_fraction": intermediate_count / intermediate_size,
        "gate_up_weights_per_token_layer": gate_up,
        "down_weights_per_token_layer": down,
        "projected_weights_per_token_layer": selected,
        "dense_weights_per_token_layer": dense,
        "projected_bytes_per_token_layer": selected * bytes_per_weight,
        "dense_bytes_per_token_layer": dense * bytes_per_weight,
        "fraction_of_dense": selected / dense,
        "candidate_recall_applicable": False,
        "metadata_included": False,
    }


def _top_k_ste(values: Any, count: int, torch: Any, *, straight_through: bool) -> Any:
    if count == values.shape[-1]:
        return values
    indices = torch.topk(values.abs(), count, dim=-1, sorted=False).indices
    mask = torch.zeros_like(values).scatter(-1, indices, 1.0)
    hard = values * mask
    return hard + values - values.detach() if straight_through else hard


def _forward(
    hidden: Any,
    gate: Any,
    up: Any,
    down: Any,
    input_count: int,
    intermediate_count: int,
    torch: Any,
    *,
    straight_through: bool,
) -> Any:
    sparse_hidden = _top_k_ste(
        hidden, input_count, torch, straight_through=straight_through
    )
    activation = torch.nn.functional.silu(
        torch.nn.functional.linear(sparse_hidden, gate)
    ) * torch.nn.functional.linear(sparse_hidden, up)
    sparse_activation = _top_k_ste(
        activation,
        intermediate_count,
        torch,
        straight_through=straight_through,
    )
    return torch.nn.functional.linear(sparse_activation, down)


def train_fully_sparse_boundaries(
    model: str | Path,
    training_traces: str | Path,
    validation_traces: str | Path,
    out: str | Path,
    *,
    layers: Sequence[int],
    input_fraction: float = 0.49,
    intermediate_fraction: float = 0.34,
    initial_artifact: str | Path | None = None,
    steps: int = 1024,
    warmup_steps: int = 128,
    start_sparse_fraction: float = 0.0,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    cosine_weight: float = 0.1,
    evaluation_interval: int = 64,
    maximum_mean_relative_l2: float = 0.18,
    maximum_traffic_fraction: float = 0.45,
    max_train_records: int | None = 32768,
    max_validation_records: int | None = 2048,
    seed: int = 2718,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train selected MLPs through the exact fully sparse deployment path."""

    if not layers:
        raise ValueError("at least one layer is required")
    if not 0 < input_fraction <= 1 or not 0 < intermediate_fraction <= 1:
        raise ValueError("sparse fractions must lie in (0, 1]")
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
    if not 0 <= start_sparse_fraction <= 1:
        raise ValueError("start_sparse_fraction must lie in [0, 1]")
    if (
        any(
            not np.isfinite(value) or value <= 0
            for value in (
                learning_rate,
                maximum_mean_relative_l2,
                maximum_traffic_fraction,
            )
        )
        or cosine_weight < 0
    ):
        raise ValueError("training rates and gate thresholds are invalid")

    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for fully sparse training"
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

    final_input_count = max(1, round(input_fraction * inspection.hidden_size))
    final_intermediate_count = max(
        1, round(intermediate_fraction * inspection.intermediate_size)
    )
    traffic = fully_sparse_mlp_traffic(
        inspection.hidden_size,
        inspection.intermediate_size,
        final_input_count,
        final_intermediate_count,
    )
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    initial_tensors = (
        None
        if initial_artifact is None
        else load_file(str(Path(initial_artifact)), device="cpu")
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    artifact_tensors: dict[str, Any] = {}
    layer_reports: list[dict[str, Any]] = []

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
            ]
            missing = [name for name in required if name not in initial_tensors]
            if missing:
                raise ValueError(
                    f"initial artifact is missing layer {layer} tensors: {missing}"
                )
            gate_np, up_np, down_np = (
                initial_tensors[required[0]].numpy(),
                initial_tensors[required[1]].numpy(),
                initial_tensors[required[2]].numpy(),
            )

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

        def evaluate() -> dict[str, float]:
            with torch.no_grad():
                approximation = _forward(
                    validation_inputs,
                    gate,
                    up,
                    down,
                    final_input_count,
                    final_intermediate_count,
                    torch,
                    straight_through=False,
                )
                relative = torch.linalg.vector_norm(
                    approximation - validation_targets, dim=-1
                ) / torch.clamp(
                    torch.linalg.vector_norm(validation_targets, dim=-1), min=1e-12
                )
                cosine = torch.nn.functional.cosine_similarity(
                    approximation, validation_targets, dim=-1
                )
                return {
                    "mean_relative_l2": float(relative.mean().item()),
                    "p95_relative_l2": float(torch.quantile(relative, 0.95).item()),
                    "mean_cosine": float(cosine.mean().item()),
                }

        initial = evaluate()
        optimizer = torch.optim.AdamW((gate, up, down), lr=learning_rate)
        history: list[dict[str, int | float]] = [{"step": 0, **initial}]
        for step in range(1, steps + 1):
            if step <= warmup_steps:
                sparse_progress = start_sparse_fraction
            else:
                linear = (step - warmup_steps) / (steps - warmup_steps)
                curve = float(np.sin(0.5 * np.pi * linear) ** 2)
                sparse_progress = (
                    start_sparse_fraction + (1.0 - start_sparse_fraction) * curve
                )
            input_count = round(
                inspection.hidden_size
                - sparse_progress * (inspection.hidden_size - final_input_count)
            )
            intermediate_count = round(
                inspection.intermediate_size
                - sparse_progress
                * (inspection.intermediate_size - final_intermediate_count)
            )
            indices = torch.randint(
                len(train_inputs),
                (min(batch_size, len(train_inputs)),),
                generator=generator,
            ).to(device)
            hidden = train_inputs[indices]
            reference = train_targets[indices]
            approximation = _forward(
                hidden,
                gate,
                up,
                down,
                input_count,
                intermediate_count,
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
            loss = normalized_mse + cosine_weight * cosine_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % evaluation_interval == 0 or step == steps:
                history.append(
                    {
                        "step": step,
                        "sparse_progress": sparse_progress,
                        "training_input_count": input_count,
                        "training_intermediate_count": intermediate_count,
                        "loss": float(loss.detach().item()),
                        "normalized_mse": float(normalized_mse.detach().item()),
                        **evaluate(),
                    }
                )

        final = evaluate()
        artifact_tensors[f"layer_{layer}.gate"] = gate.detach().cpu().contiguous()
        artifact_tensors[f"layer_{layer}.up"] = up.detach().cpu().contiguous()
        artifact_tensors[f"layer_{layer}.down"] = down.detach().cpu().contiguous()
        layer_reports.append(
            {
                "layer": layer,
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
                        traffic["fraction_of_dense"] <= maximum_traffic_fraction
                    ),
                },
            }
        )

    artifact_path = target / "fully_sparse_boundaries.safetensors"
    save_file(artifact_tensors, artifact_path)
    passed = all(
        layer["screen"]["local_quality_pass"]
        and layer["screen"]["traffic_pass_before_metadata"]
        for layer in layer_reports
    )
    report = {
        "experiment": "fully_sparse_boundary_training",
        "status": (
            "eligible_for_all_layer_causal_training"
            if passed
            else "stop_or_scale_boundary_training"
        ),
        "reference": Q_SPARSE_REFERENCE,
        "source": {
            "model_path": str(model_path),
            "model_hash": inspection.source_hash,
            "hidden_size": inspection.hidden_size,
            "intermediate_size": inspection.intermediate_size,
            "layers": inspection.num_hidden_layers,
        },
        "configuration": {
            "selected_layers": list(selected_layers),
            "input_fraction_requested": input_fraction,
            "intermediate_fraction_requested": intermediate_fraction,
            "input_count": final_input_count,
            "intermediate_count": final_intermediate_count,
            "initial_artifact": (
                None
                if initial_artifact is None
                else str(Path(initial_artifact).resolve())
            ),
            "initial_artifact_sha256": (
                None if initial_artifact is None else sha256_file(initial_artifact)
            ),
            "steps": steps,
            "warmup_steps": warmup_steps,
            "start_sparse_fraction": start_sparse_fraction,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "cosine_weight": cosine_weight,
            "maximum_mean_relative_l2": maximum_mean_relative_l2,
            "maximum_traffic_fraction": maximum_traffic_fraction,
            "seed": seed,
            "device": device,
        },
        "traffic": traffic,
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
        "screen": {"passed": passed},
        "scope_caveat": (
            "This is a cached teacher-boundary capacity screen with exact hard "
            "top-K forward execution. It is not an all-layer causal evaluation, "
            "quantized deployment artifact, or cache-line traffic measurement."
        ),
    }
    atomic_json(target / "fully_sparse_boundary_training.json", report)
    return report


__all__ = ["fully_sparse_mlp_traffic", "train_fully_sparse_boundaries"]
