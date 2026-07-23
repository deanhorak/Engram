"""Teacher-boundary local ceiling screen for fixed-width SwiGLU students."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.evaluation.mlp_intervention import _relative_and_cosine_rows
from engram.models.inspection import inspect_model, resolve_model_path
from engram.tracing.format import TraceReader
from engram.training.structured_experts import _load_trace_field, _stats
from engram.utils import atomic_json, sha256_file


def evaluate_width_pruned_local_ceiling(
    model: str | Path,
    training_traces: str | Path,
    validation_traces: str | Path,
    initial_checkpoint: str | Path,
    out: str | Path,
    *,
    layers: Sequence[int],
    target_width: int = 672,
    steps: int = 64,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    max_train_records: int | None = 4096,
    max_validation_records: int | None = 2048,
    maximum_mean_relative_l2: float = 0.15,
    minimum_improvement_fraction: float = 0.10,
    device: str = "cpu",
) -> dict[str, Any]:
    """Fit compact layers on teacher boundaries and measure their local error floor."""

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("install engram-lm[conversion] for width ceiling fits") from exc
    if not layers:
        raise ValueError("at least one layer is required")
    if steps <= 0 or batch_size <= 0 or target_width <= 0:
        raise ValueError("steps, batch_size, and target_width must be positive")
    if learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError("learning_rate must be positive and finite")
    if not 0 <= minimum_improvement_fraction < 1:
        raise ValueError("minimum_improvement_fraction must be in [0, 1)")
    if maximum_mean_relative_l2 <= 0:
        raise ValueError("maximum_mean_relative_l2 must be positive")

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    selected_layers = sorted(set(int(layer) for layer in layers))
    if selected_layers[0] < 0 or selected_layers[-1] >= inspection.num_hidden_layers:
        raise ValueError("layer index is outside the source model")
    if target_width > inspection.intermediate_size:
        raise ValueError("target_width exceeds the source intermediate size")

    training_reader = TraceReader(training_traces)
    validation_reader = TraceReader(validation_traces)
    for name, reader in (("training", training_reader), ("validation", validation_reader)):
        if reader.manifest["model_hash"] != inspection.source_hash:
            raise ValueError(f"{name} trace/model hash mismatch")
    if training_reader.manifest["dataset_hash"] == validation_reader.manifest["dataset_hash"]:
        raise ValueError("training and validation traces must use different datasets")
    if validation_reader.manifest["split"] != "validation":
        raise ValueError("validation traces must declare the validation split")

    checkpoint_path = Path(initial_checkpoint)
    checkpoint_manifest_path = checkpoint_path.with_suffix(".json")
    if not checkpoint_path.is_file() or not checkpoint_manifest_path.is_file():
        raise ValueError("width-pruned checkpoint and manifest are required")
    checkpoint_manifest = json.loads(
        checkpoint_manifest_path.read_text(encoding="utf-8")
    )
    checkpoint_configuration = checkpoint_manifest.get("configuration", {})
    if (
        checkpoint_configuration.get("source_model_hash") != inspection.source_hash
        or checkpoint_configuration.get("target_width") != target_width
    ):
        raise ValueError("width-pruned checkpoint model/width mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_parameters = checkpoint.get("parameters", {})
    del checkpoint

    class CompactSwiGLU(torch.nn.Module):
        def __init__(self, gate: Any, up: Any, down: Any) -> None:
            super().__init__()
            self.gate_weight = torch.nn.Parameter(gate.clone().float())
            self.up_weight = torch.nn.Parameter(up.clone().float())
            self.down_weight = torch.nn.Parameter(down.clone().float())

        def forward(self, hidden: Any) -> Any:
            gate = torch.nn.functional.linear(hidden, self.gate_weight)
            up = torch.nn.functional.linear(hidden, self.up_weight)
            return torch.nn.functional.linear(
                torch.nn.functional.silu(gate) * up,
                self.down_weight,
            )

    layer_reports = []
    artifact_tensors = {}
    for layer in selected_layers:
        names = {
            name: f"layers.{layer}.{name}"
            for name in ("gate_weight", "up_weight", "down_weight")
        }
        if any(full_name not in source_parameters for full_name in names.values()):
            raise ValueError(f"checkpoint is missing compact parameters for layer {layer}")
        module = CompactSwiGLU(
            source_parameters[names["gate_weight"]],
            source_parameters[names["up_weight"]],
            source_parameters[names["down_weight"]],
        ).to(device)
        train_input = torch.from_numpy(
            _load_trace_field(
                training_reader,
                f"layer_{layer}_mlp_input",
                max_train_records,
            ).astype(np.float32)
        ).to(device)
        train_target = torch.from_numpy(
            _load_trace_field(
                training_reader,
                f"layer_{layer}_mlp_output",
                max_train_records,
            ).astype(np.float32)
        ).to(device)
        validation_input = torch.from_numpy(
            _load_trace_field(
                validation_reader,
                f"layer_{layer}_mlp_input",
                max_validation_records,
            ).astype(np.float32)
        ).to(device)
        validation_target = torch.from_numpy(
            _load_trace_field(
                validation_reader,
                f"layer_{layer}_mlp_output",
                max_validation_records,
            ).astype(np.float32)
        ).to(device)
        if len(train_input) == 0 or len(validation_input) == 0:
            raise ValueError(f"layer {layer} has no usable trace records")
        module.eval()
        with torch.inference_mode():
            before_output = module(validation_input)
        before_relative, before_cosine = _relative_and_cosine_rows(
            before_output.cpu().numpy(), validation_target.cpu().numpy()
        )
        optimizer = torch.optim.AdamW(
            module.parameters(), lr=learning_rate, weight_decay=0.0
        )
        history = []
        module.train()
        for step in range(steps):
            start = (step * batch_size) % len(train_input)
            indices = torch.arange(start, start + batch_size, device=device) % len(
                train_input
            )
            output = module(train_input[indices])
            target = train_target[indices]
            loss = torch.mean((output - target) ** 2) / torch.clamp(
                torch.mean(target**2), min=1e-8
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            history.append(float(loss.detach()))
        module.eval()
        with torch.inference_mode():
            after_output = module(validation_input)
        after_relative, after_cosine = _relative_and_cosine_rows(
            after_output.cpu().numpy(), validation_target.cpu().numpy()
        )
        before_mean = float(np.mean(before_relative))
        after_mean = float(np.mean(after_relative))
        layer_reports.append(
            {
                "layer": layer,
                "training_records": len(train_input),
                "validation_records": len(validation_input),
                "relative_l2_before": _stats(before_relative.tolist()),
                "relative_l2_after": _stats(after_relative.tolist()),
                "cosine_before": _stats(before_cosine.tolist()),
                "cosine_after": _stats(after_cosine.tolist()),
                "improvement_fraction": (before_mean - after_mean)
                / max(before_mean, 1e-12),
                "training_loss_first": history[0],
                "training_loss_last": history[-1],
            }
        )
        for name, parameter in module.named_parameters():
            artifact_tensors[f"layers.{layer}.{name}"] = (
                parameter.detach().cpu().contiguous()
            )

    before_mean = float(
        np.mean([row["relative_l2_before"]["mean"] for row in layer_reports])
    )
    after_mean = float(
        np.mean([row["relative_l2_after"]["mean"] for row in layer_reports])
    )
    improvement = (before_mean - after_mean) / max(before_mean, 1e-12)
    checks = {
        "mean_relative_l2": after_mean <= maximum_mean_relative_l2,
        "mean_improvement": improvement >= minimum_improvement_fraction,
        "every_layer_improved": all(
            row["relative_l2_after"]["mean"] < row["relative_l2_before"]["mean"]
            for row in layer_reports
        ),
    }
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    artifact_path = target / "width_local_ceiling.safetensors"
    save_file(
        artifact_tensors,
        artifact_path,
        metadata={
            "format": "engram_width_local_ceiling_v1",
            "source_model_hash": inspection.source_hash,
            "target_width": str(target_width),
        },
    )
    report = {
        "schema_version": 1,
        "experiment": "width_pruned_teacher_boundary_ceiling",
        "source_model_hash": inspection.source_hash,
        "configuration": {
            "layers": selected_layers,
            "target_width": target_width,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_train_records": max_train_records,
            "max_validation_records": max_validation_records,
            "device": device,
        },
        "provenance": {
            "training_trace_dataset_hash": training_reader.manifest["dataset_hash"],
            "validation_trace_dataset_hash": validation_reader.manifest["dataset_hash"],
            "initial_checkpoint": str(checkpoint_path.resolve()),
            "initial_checkpoint_sha256": sha256_file(checkpoint_path),
        },
        "layers": layer_reports,
        "summary": {
            "mean_relative_l2_before": before_mean,
            "mean_relative_l2_after": after_mean,
            "improvement_fraction": improvement,
        },
        "screen": {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds": {
                "maximum_mean_relative_l2": maximum_mean_relative_l2,
                "minimum_improvement_fraction": minimum_improvement_fraction,
            },
            "decision": (
                "eligible_for_causal_confirmation"
                if all(checks.values())
                else "reject_width_basis"
            ),
        },
        "artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": sha256_file(artifact_path),
        },
    }
    atomic_json(target / "width_local_ceiling.json", report)
    lines = [
        "# Fixed-width teacher-boundary ceiling",
        "",
        f"Decision: **{report['screen']['decision']}**",
        "",
        f"Mean validation relative L2: {before_mean:.6f} → {after_mean:.6f} "
        f"({improvement:.2%} improvement).",
        "",
        "| Layer | Before | After | Improvement |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['layer']} | {row['relative_l2_before']['mean']:.6f} | "
        f"{row['relative_l2_after']['mean']:.6f} | "
        f"{row['improvement_fraction']:.2%} |"
        for row in layer_reports
    )
    (target / "width_local_ceiling.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["evaluate_width_pruned_local_ceiling"]
