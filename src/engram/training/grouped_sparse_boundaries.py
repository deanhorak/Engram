"""Teacher-boundary training for cache-aligned grouped sparse SwiGLU layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.evaluation.mlp_intervention import _relative_and_cosine_rows, _stats
from engram.evaluation.router_sweep import _membership, _sequence_hashes
from engram.models.inspection import inspect_model, load_layer_mlp, resolve_model_path
from engram.semantic.multilabel_router import LowRankMultiLabelRouter
from engram.semantic.swiglu import neuron_activations
from engram.tracing.format import TraceReader
from engram.training.sparse_teacher import (
    _aligned_router_width,
    _direct_router_traffic,
    _normalized_masked_mse,
    _wrap_sparse_mlp_class,
)
from engram.training.structured_experts import _load_trace_field
from engram.utils import atomic_json, sha256_file


def _learn_pair_permutation(membership: np.ndarray) -> np.ndarray:
    """Pair records by co-membership using deterministic matching and 2-opt."""

    labels = np.asarray(membership)
    if labels.ndim != 2 or labels.shape[1] % 2:
        raise ValueError("membership must have an even record dimension")
    if not np.issubdtype(labels.dtype, np.bool_):
        labels = labels.astype(bool)
    records = labels.shape[1]
    cooccurrence = labels.astype(np.float32).T @ labels.astype(np.float32)
    np.fill_diagonal(cooccurrence, -1.0)
    left, right = np.triu_indices(records, k=1)
    edge_order = np.argsort(-cooccurrence[left, right], kind="stable")
    used = np.zeros(records, dtype=bool)
    pairs: list[list[int]] = []
    for edge in edge_order:
        first = int(left[edge])
        second = int(right[edge])
        if not used[first] and not used[second]:
            pairs.append([first, second])
            used[first] = True
            used[second] = True
            if len(pairs) == records // 2:
                break
    if len(pairs) != records // 2:
        raise RuntimeError("failed to construct a complete pair matching")

    # Deterministic two-pair swaps recover most of the remaining matching gain
    # without introducing a heavyweight graph-optimization dependency.
    for _ in range(2):
        changed = False
        for first_pair in range(len(pairs)):
            a, b = pairs[first_pair]
            for second_pair in range(first_pair + 1, len(pairs)):
                c, d = pairs[second_pair]
                current = cooccurrence[a, b] + cooccurrence[c, d]
                crossed = cooccurrence[a, c] + cooccurrence[b, d]
                parallel = cooccurrence[a, d] + cooccurrence[b, c]
                if crossed > current and crossed >= parallel:
                    pairs[first_pair] = [a, c]
                    pairs[second_pair] = [b, d]
                    b = c
                    changed = True
                elif parallel > current:
                    pairs[first_pair] = [a, d]
                    pairs[second_pair] = [b, c]
                    b = d
                    changed = True
        if not changed:
            break
    permutation = np.asarray(pairs, dtype=np.int64).reshape(-1)
    if not np.array_equal(np.sort(permutation), np.arange(records)):
        raise RuntimeError("learned pairing is not a permutation")
    return permutation


def _pair_contribution_utility(
    states: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> np.ndarray:
    """Compute exact norms of each two-record output contribution."""

    if gate.shape != up.shape or gate.shape[0] % 2:
        raise ValueError("gate/up matrices must match and contain complete pairs")
    if down.shape != (gate.shape[1], gate.shape[0]):
        raise ValueError("down matrix has incompatible dimensions")
    activation = neuron_activations(states, gate, up).reshape(len(states), -1, 2)
    values = down.T.reshape(-1, 2, down.shape[0])
    first_norm = np.sum(values[:, 0] ** 2, axis=1)
    second_norm = np.sum(values[:, 1] ** 2, axis=1)
    cross = np.sum(values[:, 0] * values[:, 1], axis=1)
    first = activation[:, :, 0]
    second = activation[:, :, 1]
    squared = (
        first**2 * first_norm[None, :]
        + second**2 * second_norm[None, :]
        + 2.0 * first * second * cross[None, :]
    )
    return np.sqrt(np.maximum(squared, 0.0))


def _fixed_cardinality_labels(utility: np.ndarray, selected_groups: int) -> np.ndarray:
    if utility.ndim != 2 or not 0 < selected_groups <= utility.shape[1]:
        raise ValueError("selected_groups must lie within the utility width")
    order = np.argsort(-utility, axis=1, kind="stable")[:, :selected_groups]
    labels = np.zeros(utility.shape, dtype=bool)
    labels[np.arange(len(utility))[:, None], order] = True
    return labels


def train_grouped_sparse_boundaries(
    model: str | Path,
    training_traces: str | Path,
    validation_traces: str | Path,
    out: str | Path,
    *,
    layers: Sequence[int],
    top_k: int = 672,
    start_top_k: int = 768,
    router_group_size: int = 2,
    router_rank: int = 16,
    router_regularization: float = 8000.0,
    adapter_rank: int = 8,
    router_warmup_steps: int = 32,
    anneal_steps: int = 64,
    settle_steps: int = 128,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    router_learning_rate: float = 1e-3,
    route_weight: float = 0.1,
    cosine_weight: float = 0.1,
    dense_anchor_weight: float = 0.1,
    start_temperature: float = 1.0,
    temperature: float = 0.5,
    evaluation_interval: int = 32,
    maximum_mean_relative_l2: float = 0.15,
    max_train_records: int | None = 4096,
    max_validation_records: int | None = 2048,
    learned_pairing: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train selected grouped sparse layers on stable teacher MLP boundaries."""

    try:
        import torch
        import torch.nn.functional as functional
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for grouped boundary training"
        ) from exc
    if not layers:
        raise ValueError("at least one layer is required")
    if router_group_size != 2:
        raise ValueError("the cache-aligned boundary screen currently requires pairs")
    if min(top_k, start_top_k, router_rank, adapter_rank, batch_size) <= 0:
        raise ValueError("widths, ranks, and batch_size must be positive")
    if top_k > start_top_k or top_k % 2 or start_top_k % 2:
        raise ValueError("require even top_k <= even start_top_k")
    if min(router_warmup_steps, anneal_steps, settle_steps) < 0:
        raise ValueError("phase step counts must be nonnegative")
    total_steps = router_warmup_steps + anneal_steps + settle_steps
    if total_steps <= 0 or evaluation_interval <= 0:
        raise ValueError("training and evaluation intervals must be positive")
    scalar_values = (
        router_regularization,
        learning_rate,
        router_learning_rate,
        start_temperature,
        temperature,
        maximum_mean_relative_l2,
    )
    if any(not np.isfinite(value) or value <= 0 for value in scalar_values):
        raise ValueError("regularization, rates, temperatures, and threshold must be positive")
    if any(
        not np.isfinite(value) or value < 0
        for value in (route_weight, cosine_weight, dense_anchor_weight)
    ):
        raise ValueError("loss weights must be finite and nonnegative")

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    selected_layers = sorted(set(int(layer) for layer in layers))
    if selected_layers[0] < 0 or selected_layers[-1] >= inspection.num_hidden_layers:
        raise ValueError("layer index is outside the source model")
    if inspection.intermediate_size % 2 or start_top_k > inspection.intermediate_size:
        raise ValueError("model intermediate size is incompatible with the requested widths")
    traffic = _direct_router_traffic(
        inspection.hidden_size,
        inspection.intermediate_size,
        candidate_count=top_k,
        top_k=top_k,
        router_rank=router_rank,
        router_group_size=router_group_size,
    )
    if traffic["fraction_of_dense_q4"] > 0.45:
        raise ValueError("requested grouped router exceeds the traffic gate")

    training_reader = TraceReader(training_traces)
    validation_reader = TraceReader(validation_traces)
    for name, reader, split in (
        ("training", training_reader, "calibration"),
        ("validation", validation_reader, "validation"),
    ):
        if reader.manifest["model_hash"] != inspection.source_hash:
            raise ValueError(f"{name} trace/model hash mismatch")
        if reader.manifest["split"] != split:
            raise ValueError(f"expected {split!r} {name} traces")
    if training_reader.manifest["dataset_hash"] == validation_reader.manifest["dataset_hash"]:
        raise ValueError("training and validation boundary datasets must differ")
    overlap = set(_sequence_hashes(training_reader)).intersection(
        _sequence_hashes(validation_reader)
    )
    if overlap:
        raise ValueError("training and validation boundary sequences overlap")

    class DenseSwiGLU(torch.nn.Module):
        def __init__(self, gate: np.ndarray, up: np.ndarray, down: np.ndarray):
            super().__init__()
            self.gate_proj = torch.nn.Linear(
                gate.shape[1], gate.shape[0], bias=False
            )
            self.up_proj = torch.nn.Linear(up.shape[1], up.shape[0], bias=False)
            self.down_proj = torch.nn.Linear(
                down.shape[1], down.shape[0], bias=False
            )
            with torch.no_grad():
                self.gate_proj.weight.copy_(torch.from_numpy(gate).float())
                self.up_proj.weight.copy_(torch.from_numpy(up).float())
                self.down_proj.weight.copy_(torch.from_numpy(down).float())
            self.act_fn = functional.silu

        def forward(self, hidden: Any) -> Any:
            return self.down_proj(
                self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden)
            )

    Wrapper = _wrap_sparse_mlp_class(torch)
    artifact_tensors: dict[str, Any] = {}
    layer_reports: list[dict[str, Any]] = []
    for layer in selected_layers:
        gate, up, down = (
            np.asarray(value, dtype=np.float64)
            for value in load_layer_mlp(model_path, layer)
        )
        train_input = _load_trace_field(
            training_reader, f"layer_{layer}_mlp_input", max_train_records
        ).astype(np.float64)
        train_target = _load_trace_field(
            training_reader, f"layer_{layer}_mlp_output", max_train_records
        ).astype(np.float32)
        validation_input = _load_trace_field(
            validation_reader, f"layer_{layer}_mlp_input", max_validation_records
        ).astype(np.float64)
        validation_target = _load_trace_field(
            validation_reader, f"layer_{layer}_mlp_output", max_validation_records
        ).astype(np.float32)
        if not len(train_input) or not len(validation_input):
            raise ValueError(f"layer {layer} has no boundary records")

        if learned_pairing:
            record_labels = _membership(train_input, gate, up, down, top_k)
            permutation = _learn_pair_permutation(record_labels)
        else:
            permutation = np.arange(inspection.intermediate_size, dtype=np.int64)
        gate = gate[permutation]
        up = up[permutation]
        down = down[:, permutation]
        train_utility = _pair_contribution_utility(train_input, gate, up, down)
        validation_utility = _pair_contribution_utility(
            validation_input, gate, up, down
        )
        fit_labels = _fixed_cardinality_labels(
            train_utility, top_k // router_group_size
        )
        router = LowRankMultiLabelRouter.fit(
            train_input,
            fit_labels.astype(np.float64),
            rank=router_rank,
            regularization=router_regularization,
        )
        base = DenseSwiGLU(gate, up, down)
        wrapper = Wrapper(
            base,
            router,
            top_k=start_top_k,
            candidates=start_top_k,
            adapter_rank=adapter_rank,
            router_group_size=router_group_size,
            train_full_mlp=True,
        ).to(device)
        # Boundary fitting updates the canonical MLP weights directly. Keeping
        # the mergeable adapters at zero avoids redundant parameterizations.
        for name, parameter in wrapper.named_parameters():
            if "adapter_" in name:
                parameter.requires_grad_(False)
        router_parameters = [
            parameter
            for name, parameter in wrapper.named_parameters()
            if name.startswith("router_") and parameter.requires_grad
        ]
        mlp_parameters = [
            parameter
            for name, parameter in wrapper.named_parameters()
            if name.startswith("base.") and parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": router_parameters,
                    "lr": router_learning_rate,
                    "weight_decay": 0.0,
                },
                {
                    "params": mlp_parameters,
                    "lr": learning_rate,
                    "weight_decay": 0.0,
                },
            ]
        )
        train_x = torch.from_numpy(train_input.astype(np.float32)).to(device)
        train_y = torch.from_numpy(train_target).to(device)
        train_u = torch.from_numpy(train_utility.astype(np.float32)).to(device)
        validation_x = torch.from_numpy(validation_input.astype(np.float32)).to(device)
        validation_u = torch.from_numpy(
            validation_utility.astype(np.float32)
        ).to(device)
        full_mask = torch.ones(len(train_x), device=device, dtype=torch.bool)

        def evaluate(step: int) -> dict[str, Any]:
            wrapper.eval()
            wrapper.top_k = top_k
            wrapper.candidates = top_k
            wrapper.temperature = temperature
            with torch.inference_mode():
                output = wrapper(validation_x)
                logits = wrapper.last_router_logits
                selected = torch.topk(
                    logits, top_k // router_group_size, dim=1
                ).indices
                reference = torch.topk(
                    validation_u, top_k // router_group_size, dim=1
                ).indices
                selected_mask = torch.zeros_like(logits, dtype=torch.bool).scatter(
                    1, selected, True
                )
                route_recall = selected_mask.gather(1, reference).float().mean()
            relative, cosine = _relative_and_cosine_rows(
                output.detach().cpu().numpy(), validation_target
            )
            return {
                "step": step,
                "relative_l2": _stats(relative.tolist()),
                "cosine": _stats(cosine.tolist()),
                "group_target_recall": float(route_recall.cpu()),
            }

        with torch.inference_mode():
            dense_shadow = wrapper.base(validation_x)
        dense_relative, _ = _relative_and_cosine_rows(
            dense_shadow.cpu().numpy(), validation_target
        )
        evaluations = [evaluate(0)]
        best = evaluations[0]
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in wrapper.state_dict().items()
        }
        generator = np.random.default_rng(12011 + layer)
        order = generator.permutation(len(train_x))
        cursor = 0
        training_first = None
        training_last = None
        wrapper.train()
        for step in range(total_steps):
            if cursor + batch_size > len(order):
                order = generator.permutation(len(train_x))
                cursor = 0
            indices_numpy = order[cursor : cursor + batch_size]
            cursor += batch_size
            indices = torch.as_tensor(indices_numpy, device=device, dtype=torch.long)
            if step < router_warmup_steps:
                width = start_top_k
                phase_progress = 0.0
                phase = "router_warmup"
            elif step < router_warmup_steps + anneal_steps:
                offset = step - router_warmup_steps
                phase_progress = offset / max(1, anneal_steps - 1)
                width = _aligned_router_width(
                    start_top_k, top_k, phase_progress, router_group_size
                )
                phase = "anneal"
            else:
                width = top_k
                phase_progress = 1.0
                phase = "settle"
            wrapper.top_k = width
            wrapper.candidates = width
            wrapper.temperature = start_temperature + phase_progress * (
                temperature - start_temperature
            )
            x = train_x[indices]
            target_y = train_y[indices]
            target_groups = torch.topk(
                train_u[indices], width // router_group_size, dim=1
            ).indices
            output = wrapper(x)
            route_target = torch.zeros_like(wrapper.last_router_logits).scatter(
                1, target_groups, 1.0
            )
            route_loss = functional.binary_cross_entropy_with_logits(
                wrapper.last_router_logits, route_target
            )
            if phase == "router_warmup":
                reconstruction = torch.zeros((), device=device)
                cosine_loss = torch.zeros((), device=device)
                dense_anchor = torch.zeros((), device=device)
                loss = route_loss
            else:
                reconstruction = _normalized_masked_mse(
                    output, target_y, full_mask[indices], torch
                )
                cosine_loss = (
                    1.0
                    - functional.cosine_similarity(output, target_y, dim=1)
                ).mean()
                dense_output = wrapper.base(x)
                dense_anchor = _normalized_masked_mse(
                    dense_output, target_y, full_mask[indices], torch
                )
                route_scale = route_weight * (1.0 - 0.9 * phase_progress)
                anchor_scale = dense_anchor_weight * (
                    1.0 - 0.8 * phase_progress
                )
                loss = (
                    reconstruction
                    + cosine_weight * cosine_loss
                    + route_scale * route_loss
                    + anchor_scale * dense_anchor
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router_parameters + mlp_parameters, 1.0)
            optimizer.step()
            row = {
                "step": step + 1,
                "phase": phase,
                "active_records": width,
                "temperature": wrapper.temperature,
                "loss": float(loss.detach()),
                "reconstruction": float(reconstruction.detach()),
                "cosine": float(cosine_loss.detach()),
                "dense_anchor": float(dense_anchor.detach()),
                "route": float(route_loss.detach()),
            }
            training_first = row if training_first is None else training_first
            training_last = row
            if (step + 1) % evaluation_interval == 0 or step + 1 == total_steps:
                result = evaluate(step + 1)
                evaluations.append(result)
                if result["relative_l2"]["mean"] < best["relative_l2"]["mean"]:
                    best = result
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in wrapper.state_dict().items()
                    }
                wrapper.train()
        wrapper.load_state_dict(best_state)
        wrapper.to(device)

        merged_gate = wrapper.base.gate_proj.weight.detach().cpu()
        merged_up = wrapper.base.up_proj.weight.detach().cpu()
        merged_down = wrapper.base.down_proj.weight.detach().cpu()
        prefix = f"layer_{layer}."
        artifact_tensors[prefix + "deployment_gate_weight"] = merged_gate.contiguous()
        artifact_tensors[prefix + "deployment_up_weight"] = merged_up.contiguous()
        artifact_tensors[prefix + "deployment_down_weight"] = merged_down.contiguous()
        artifact_tensors[prefix + "router_input"] = (
            wrapper.router_input.detach().cpu().contiguous()
        )
        artifact_tensors[prefix + "router_output"] = (
            wrapper.router_output.detach().cpu().contiguous()
        )
        artifact_tensors[prefix + "router_bias"] = (
            wrapper.router_bias.detach().cpu().contiguous()
        )
        artifact_tensors[prefix + "router_nonlinear_scale"] = (
            wrapper.router_nonlinear_scale.detach().cpu().contiguous()
        )
        artifact_tensors[prefix + "permutation"] = torch.from_numpy(
            permutation
        ).contiguous()
        layer_reports.append(
            {
                "layer": layer,
                "training_records": len(train_input),
                "validation_records": len(validation_input),
                "dense_shadow_relative_l2": _stats(dense_relative.tolist()),
                "initial": evaluations[0],
                "best": best,
                "final": evaluations[-1],
                "evaluations": evaluations,
                "training_first": training_first,
                "training_last": training_last,
                "learned_pairing": learned_pairing,
            }
        )

    mean_best_relative_l2 = float(
        np.mean([row["best"]["relative_l2"]["mean"] for row in layer_reports])
    )
    checks = {
        "dense_permutation_parity": all(
            row["dense_shadow_relative_l2"]["maximum"] <= 1e-4
            for row in layer_reports
        ),
        "mean_relative_l2": mean_best_relative_l2 <= maximum_mean_relative_l2,
        "every_layer_improved": all(
            row["best"]["relative_l2"]["mean"]
            < row["initial"]["relative_l2"]["mean"]
            for row in layer_reports
        ),
        "projected_traffic": traffic["fraction_of_dense_q4"] <= 0.45,
    }
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    artifact_path = target / "grouped_sparse_boundaries.safetensors"
    save_file(
        artifact_tensors,
        artifact_path,
        metadata={
            "format": "engram_grouped_sparse_boundaries_v1",
            "source_model_hash": inspection.source_hash,
            "router_group_size": str(router_group_size),
            "top_k": str(top_k),
        },
    )
    report = {
        "schema_version": 1,
        "experiment": "grouped_sparse_teacher_boundary_training",
        "source_model_hash": inspection.source_hash,
        "configuration": {
            "layers": selected_layers,
            "top_k": top_k,
            "start_top_k": start_top_k,
            "router_group_size": router_group_size,
            "router_rank": router_rank,
            "router_regularization": router_regularization,
            "adapter_rank": adapter_rank,
            "router_warmup_steps": router_warmup_steps,
            "anneal_steps": anneal_steps,
            "settle_steps": settle_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "router_learning_rate": router_learning_rate,
            "route_weight": route_weight,
            "cosine_weight": cosine_weight,
            "dense_anchor_weight": dense_anchor_weight,
            "start_temperature": start_temperature,
            "temperature": temperature,
            "evaluation_interval": evaluation_interval,
            "maximum_mean_relative_l2": maximum_mean_relative_l2,
            "max_train_records": max_train_records,
            "max_validation_records": max_validation_records,
            "learned_pairing": learned_pairing,
            "device": device,
        },
        "data_separation": {
            "training_dataset_hash": training_reader.manifest["dataset_hash"],
            "validation_dataset_hash": validation_reader.manifest["dataset_hash"],
            "overlapping_sequences": 0,
            "held_out": True,
        },
        "projected_traffic": traffic,
        "layers": layer_reports,
        "summary": {"mean_best_relative_l2": mean_best_relative_l2},
        "screen": {
            "passed": all(checks.values()),
            "checks": checks,
            "decision": (
                "eligible_for_all_layer_boundary_training"
                if all(checks.values())
                else "reject_or_continue_boundary_optimization"
            ),
            "selection_caveat": (
                "Best checkpoints use the boundary-validation split for development; "
                "the causal gate still requires untouched sequence-level confirmation."
            ),
        },
        "artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": sha256_file(artifact_path),
        },
    }
    atomic_json(target / "grouped_sparse_boundaries.json", report)
    lines = [
        "# Grouped sparse teacher-boundary training",
        "",
        f"Decision: **{report['screen']['decision']}**",
        "",
        "| Layer | Initial rel-L2 | Best rel-L2 | Final rel-L2 | Group recall |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['layer']} | {row['initial']['relative_l2']['mean']:.6f} | "
        f"{row['best']['relative_l2']['mean']:.6f} | "
        f"{row['final']['relative_l2']['mean']:.6f} | "
        f"{row['best']['group_target_recall']:.6f} |"
        for row in layer_reports
    )
    lines.extend(
        (
            "",
            f"Mean best relative L2: {mean_best_relative_l2:.6f} "
            f"(target <= {maximum_mean_relative_l2:.6f}).",
            f"Projected cold traffic: {traffic['fraction_of_dense_q4']:.6f}x dense Q4.",
            "",
        )
    )
    (target / "grouped_sparse_boundaries.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


__all__ = [
    "_fixed_cardinality_labels",
    "_learn_pair_permutation",
    "_pair_contribution_utility",
    "train_grouped_sparse_boundaries",
]
