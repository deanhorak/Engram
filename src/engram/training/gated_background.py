"""Nonlinear background fitting for the diffuse tail of sparse SwiGLU reads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.evaluation.mlp_intervention import _relative_and_cosine_rows
from engram.models.inspection import inspect_model, load_layer_mlp, resolve_model_path
from engram.semantic.swiglu import neuron_activations
from engram.tracing.format import TraceReader
from engram.training.structured_experts import _load_trace_field, _stats
from engram.utils import atomic_json, sha256_file


@dataclass(frozen=True)
class SparseBackgroundTraffic:
    selected_width: int
    background_width: int
    router_rank: int
    selected_weight_bytes: int
    background_weight_bytes: int
    router_bytes: int
    total_bytes: int
    dense_weight_bytes: int
    fraction_of_dense: float

    def to_dict(self) -> dict[str, int | float]:
        return dict(self.__dict__)


def sparse_background_traffic(
    hidden_size: int,
    intermediate_size: int,
    selected_width: int,
    background_width: int,
    router_rank: int,
    *,
    bytes_per_parameter: int = 4,
) -> SparseBackgroundTraffic:
    values = (
        hidden_size,
        intermediate_size,
        selected_width,
        background_width,
        router_rank,
        bytes_per_parameter,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("traffic dimensions must be integers")
    if hidden_size <= 0 or intermediate_size <= 0 or bytes_per_parameter <= 0:
        raise ValueError("hidden/intermediate widths and byte size must be positive")
    if not 0 < selected_width <= intermediate_size or background_width <= 0:
        raise ValueError("selected/background widths are invalid")
    if router_rank <= 0:
        raise ValueError("router_rank must be positive")
    selected_bytes = 3 * hidden_size * selected_width * bytes_per_parameter
    background_bytes = 3 * hidden_size * background_width * bytes_per_parameter
    router_bytes = (
        hidden_size * router_rank
        + router_rank * intermediate_size
        + intermediate_size
    ) * bytes_per_parameter
    dense_bytes = 3 * hidden_size * intermediate_size * bytes_per_parameter
    total = selected_bytes + background_bytes + router_bytes
    return SparseBackgroundTraffic(
        selected_width=selected_width,
        background_width=background_width,
        router_rank=router_rank,
        selected_weight_bytes=selected_bytes,
        background_weight_bytes=background_bytes,
        router_bytes=router_bytes,
        total_bytes=total,
        dense_weight_bytes=dense_bytes,
        fraction_of_dense=total / dense_bytes,
    )


def _oracle_selected_output(
    states: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    top_k: int,
) -> np.ndarray:
    activations = neuron_activations(states, gate, up)
    scores = np.abs(activations) * np.linalg.norm(down, axis=0)[None, :]
    indices = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    selected_activations = np.take_along_axis(activations, indices, axis=1)
    selected_values = down[:, indices]
    return np.einsum("nk,hnk->nh", selected_activations, selected_values).astype(
        np.float32
    )


def evaluate_gated_background_ceiling(
    model: str | Path,
    training_traces: str | Path,
    validation_traces: str | Path,
    out: str | Path,
    *,
    layers: Sequence[int],
    top_k: int = 512,
    background_width: int = 128,
    router_rank: int = 16,
    steps: int = 1024,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    max_train_records: int | None = 4096,
    max_validation_records: int | None = 2048,
    maximum_mean_relative_l2: float = 0.10,
    device: str = "cpu",
    seed: int = 97,
) -> dict[str, Any]:
    """Fit a small SwiGLU to the exact top-K residual on teacher boundaries."""

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("install engram-lm[conversion] for background fitting") from exc
    if not layers:
        raise ValueError("at least one layer is required")
    if steps <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("steps, batch_size, and learning_rate must be positive")
    if maximum_mean_relative_l2 <= 0:
        raise ValueError("maximum_mean_relative_l2 must be positive")

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    selected_layers = sorted(set(int(layer) for layer in layers))
    if selected_layers[0] < 0 or selected_layers[-1] >= inspection.num_hidden_layers:
        raise ValueError("layer index is outside the source model")
    traffic = sparse_background_traffic(
        inspection.hidden_size,
        inspection.intermediate_size,
        top_k,
        background_width,
        router_rank,
    )
    training_reader = TraceReader(training_traces)
    validation_reader = TraceReader(validation_traces)
    for name, reader in (("training", training_reader), ("validation", validation_reader)):
        if reader.manifest["model_hash"] != inspection.source_hash:
            raise ValueError(f"{name} trace/model hash mismatch")
    if training_reader.manifest["dataset_hash"] == validation_reader.manifest["dataset_hash"]:
        raise ValueError("training and validation traces must use different datasets")
    if validation_reader.manifest["split"] != "validation":
        raise ValueError("validation traces must declare the validation split")

    class GatedBackground(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = torch.nn.Linear(
                inspection.hidden_size, background_width, bias=False
            )
            self.up = torch.nn.Linear(
                inspection.hidden_size, background_width, bias=False
            )
            self.down = torch.nn.Linear(
                background_width, inspection.hidden_size, bias=False
            )
            torch.nn.init.zeros_(self.down.weight)

        def forward(self, hidden: Any) -> Any:
            return self.down(torch.nn.functional.silu(self.gate(hidden)) * self.up(hidden))

    layer_reports = []
    artifact_tensors = {}
    for layer in selected_layers:
        torch.manual_seed(seed + layer)
        gate, up, down = load_layer_mlp(model_path, layer)
        train_input = _load_trace_field(
            training_reader, f"layer_{layer}_mlp_input", max_train_records
        ).astype(np.float32)
        train_teacher = _load_trace_field(
            training_reader, f"layer_{layer}_mlp_output", max_train_records
        ).astype(np.float32)
        validation_input = _load_trace_field(
            validation_reader, f"layer_{layer}_mlp_input", max_validation_records
        ).astype(np.float32)
        validation_teacher = _load_trace_field(
            validation_reader, f"layer_{layer}_mlp_output", max_validation_records
        ).astype(np.float32)
        train_sparse = _oracle_selected_output(train_input, gate, up, down, top_k)
        validation_sparse = _oracle_selected_output(
            validation_input, gate, up, down, top_k
        )
        train_residual = train_teacher - train_sparse
        validation_residual = validation_teacher - validation_sparse
        before_relative, before_cosine = _relative_and_cosine_rows(
            validation_sparse, validation_teacher
        )

        module = GatedBackground().to(device)
        optimizer = torch.optim.AdamW(
            module.parameters(), lr=learning_rate, weight_decay=0.0
        )
        inputs = torch.from_numpy(train_input).to(device)
        residuals = torch.from_numpy(train_residual).to(device)
        teachers = torch.from_numpy(train_teacher).to(device)
        history = []
        module.train()
        for step in range(steps):
            start = (step * batch_size) % len(inputs)
            indices = torch.arange(start, start + batch_size, device=device) % len(inputs)
            prediction = module(inputs[indices])
            loss = torch.mean((prediction - residuals[indices]) ** 2) / torch.clamp(
                torch.mean(teachers[indices] ** 2), min=1e-8
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            history.append(float(loss.detach()))
        module.eval()
        with torch.inference_mode():
            prediction = (
                module(torch.from_numpy(validation_input).to(device)).cpu().numpy()
            )
        corrected = validation_sparse + prediction
        after_relative, after_cosine = _relative_and_cosine_rows(
            corrected, validation_teacher
        )
        residual_relative, _ = _relative_and_cosine_rows(
            prediction, validation_residual
        )
        layer_reports.append(
            {
                "layer": layer,
                "training_records": len(train_input),
                "validation_records": len(validation_input),
                "sparse_relative_l2": _stats(before_relative.tolist()),
                "corrected_relative_l2": _stats(after_relative.tolist()),
                "sparse_cosine": _stats(before_cosine.tolist()),
                "corrected_cosine": _stats(after_cosine.tolist()),
                "residual_prediction_relative_l2": _stats(
                    residual_relative.tolist()
                ),
                "training_loss_first": history[0],
                "training_loss_last": history[-1],
            }
        )
        for name, parameter in module.named_parameters():
            artifact_tensors[f"layers.{layer}.{name}.weight"] = (
                parameter.detach().cpu().contiguous()
            )

    before_mean = float(
        np.mean([row["sparse_relative_l2"]["mean"] for row in layer_reports])
    )
    after_mean = float(
        np.mean([row["corrected_relative_l2"]["mean"] for row in layer_reports])
    )
    checks = {
        "mean_relative_l2": after_mean <= maximum_mean_relative_l2,
        "every_layer_improved": all(
            row["corrected_relative_l2"]["mean"]
            < row["sparse_relative_l2"]["mean"]
            for row in layer_reports
        ),
        "projected_traffic": traffic.fraction_of_dense <= 0.45,
    }
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    artifact_path = target / "gated_background.safetensors"
    save_file(
        artifact_tensors,
        artifact_path,
        metadata={
            "format": "engram_gated_background_v1",
            "source_model_hash": inspection.source_hash,
            "top_k": str(top_k),
            "background_width": str(background_width),
            "router_rank": str(router_rank),
        },
    )
    report = {
        "schema_version": 1,
        "experiment": "oracle_topk_gated_background_ceiling",
        "source_model_hash": inspection.source_hash,
        "configuration": {
            "layers": selected_layers,
            "top_k": top_k,
            "background_width": background_width,
            "router_rank": router_rank,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device,
            "seed": seed,
        },
        "provenance": {
            "training_trace_dataset_hash": training_reader.manifest["dataset_hash"],
            "validation_trace_dataset_hash": validation_reader.manifest["dataset_hash"],
        },
        "layers": layer_reports,
        "summary": {
            "mean_relative_l2_before": before_mean,
            "mean_relative_l2_after": after_mean,
            "improvement_fraction": (before_mean - after_mean)
            / max(before_mean, 1e-12),
        },
        "projected_traffic": traffic.to_dict(),
        "screen": {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds": {
                "maximum_mean_relative_l2": maximum_mean_relative_l2,
                "maximum_traffic_fraction": 0.45,
            },
            "decision": (
                "eligible_for_causal_oracle_test"
                if all(checks.values())
                else "reject_gated_background"
            ),
        },
        "artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": sha256_file(artifact_path),
        },
    }
    atomic_json(target / "gated_background_ceiling.json", report)
    lines = [
        "# Exact top-K plus gated-background ceiling",
        "",
        f"Decision: **{report['screen']['decision']}**",
        "",
        f"Mean validation relative L2: {before_mean:.6f} → {after_mean:.6f}.",
        f"Projected selected + background + rank-{router_rank} router traffic: "
        f"{traffic.fraction_of_dense:.6f}× dense.",
        "",
        "| Layer | Sparse | Corrected | Residual prediction |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['layer']} | {row['sparse_relative_l2']['mean']:.6f} | "
        f"{row['corrected_relative_l2']['mean']:.6f} | "
        f"{row['residual_prediction_relative_l2']['mean']:.6f} |"
        for row in layer_reports
    )
    (target / "gated_background_ceiling.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


__all__ = [
    "SparseBackgroundTraffic",
    "evaluate_gated_background_ceiling",
    "sparse_background_traffic",
]
