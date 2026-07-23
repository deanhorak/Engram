"""Cached-boundary training for a shared-plus-coarse-expert SwiGLU.

The layout keeps a small duplicate slab of high-utility records resident for
every token and routes whole, cache-aligned blocks from a complete expert copy
of the original intermediate dimension.  A shared record therefore occurs
twice physically.  Both down vectors start at half strength, so evaluating the
shared slab and every expert exactly reconstructs the source dense MLP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.evaluation.mlp_intervention import _relative_and_cosine_rows, _stats
from engram.evaluation.router_sweep import _sequence_hashes
from engram.models.inspection import inspect_model, load_layer_mlp, resolve_model_path
from engram.semantic.multilabel_router import MultiLabelLinearRouter
from engram.semantic.swiglu import neuron_activations
from engram.tracing.format import TraceReader
from engram.training.sparse_teacher import (
    _cardinality_preserving_top_mask,
    _normalized_masked_mse,
)
from engram.training.structured_experts import (
    _greedy_residual_experts,
    _load_trace_field,
    balanced_expert_permutation,
)
from engram.utils import atomic_json, sha256_file


def _cache_aligned(byte_count: int, cache_line_bytes: int) -> int:
    return ((byte_count + cache_line_bytes - 1) // cache_line_bytes) * cache_line_bytes


def shared_expert_traffic(
    hidden_size: int,
    intermediate_size: int,
    *,
    shared_records: int = 128,
    experts: int = 96,
    active_experts: int = 32,
    q4_bits: int = 4,
    cache_line_bytes: int = 64,
) -> dict[str, int | float]:
    """Return cache-line-honest cold bytes for the deployed Q4 layout.

    All three projection rows for one physical expert are packed together.
    The shared slab is likewise one contiguous fetch.  The full linear router
    is Q4, while its bias and selected expert IDs are FP16 and uint16.
    """

    values = (
        hidden_size,
        intermediate_size,
        shared_records,
        experts,
        active_experts,
        q4_bits,
        cache_line_bytes,
    )
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("traffic dimensions must be positive integers")
    if intermediate_size % experts:
        raise ValueError("experts must divide the intermediate size")
    if shared_records > intermediate_size:
        raise ValueError("shared_records cannot exceed the intermediate size")
    if active_experts > experts:
        raise ValueError("active_experts cannot exceed experts")
    if q4_bits != 4:
        raise ValueError("the boundary screen currently accounts Q4 weights")

    records_per_expert = intermediate_size // experts
    shared_payload_bits = 3 * shared_records * hidden_size * q4_bits
    expert_payload_bits = 3 * records_per_expert * hidden_size * q4_bits
    router_payload_bits = hidden_size * experts * q4_bits
    shared_weight_bytes = _cache_aligned(
        (shared_payload_bits + 7) // 8, cache_line_bytes
    )
    expert_block_bytes = _cache_aligned(
        (expert_payload_bits + 7) // 8, cache_line_bytes
    )
    selected_expert_weight_bytes = active_experts * expert_block_bytes
    router_weight_bytes = _cache_aligned(
        (router_payload_bits + 7) // 8, cache_line_bytes
    )
    router_bias_fp16_bytes = experts * 2
    expert_id_uint16_bytes = active_experts * 2
    router_metadata_bytes = _cache_aligned(
        router_bias_fp16_bytes + expert_id_uint16_bytes, cache_line_bytes
    )
    total_bytes = (
        shared_weight_bytes
        + selected_expert_weight_bytes
        + router_weight_bytes
        + router_metadata_bytes
    )
    dense_q4_bytes = (3 * hidden_size * intermediate_size * q4_bits + 7) // 8
    return {
        "cache_line_bytes": cache_line_bytes,
        "q4_bits": q4_bits,
        "shared_records": shared_records,
        "experts": experts,
        "active_experts": active_experts,
        "records_per_expert": records_per_expert,
        "active_physical_records": shared_records
        + active_experts * records_per_expert,
        "shared_weight_q4_bytes": shared_weight_bytes,
        "expert_block_q4_bytes": expert_block_bytes,
        "selected_expert_weight_q4_bytes": selected_expert_weight_bytes,
        "router_weight_q4_bytes": router_weight_bytes,
        "router_bias_fp16_bytes": router_bias_fp16_bytes,
        "expert_id_uint16_bytes": expert_id_uint16_bytes,
        "router_metadata_cache_aligned_bytes": router_metadata_bytes,
        "total_bytes": total_bytes,
        "dense_q4_bytes": dense_q4_bytes,
        "fraction_of_dense_q4": total_bytes / dense_q4_bytes,
    }


def _record_strength(
    states: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> np.ndarray:
    """Return exact scalar contribution magnitudes for grouping/selection."""

    activation = np.abs(neuron_activations(states, gate, up))
    return activation * np.linalg.norm(down, axis=0)[None, :]


def _select_shared_records(strength: np.ndarray, count: int) -> np.ndarray:
    matrix = np.asarray(strength, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("strength must be a non-empty state-by-record matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("strength must be finite")
    if not isinstance(count, int) or not 0 < count <= matrix.shape[1]:
        raise ValueError("count must lie within the record dimension")
    mean_strength = np.mean(matrix, axis=0)
    return np.argsort(-mean_strength, kind="stable")[:count].astype(np.int64)


def _halved_expert_down(
    down: np.ndarray, permutation: np.ndarray, shared_indices: np.ndarray
) -> np.ndarray:
    """Return expert-pool down rows with shared duplicates at half strength."""

    matrix = np.asarray(down)
    order = np.asarray(permutation, dtype=np.int64)
    shared = np.asarray(shared_indices, dtype=np.int64)
    if matrix.ndim != 2 or order.shape != (matrix.shape[1],):
        raise ValueError("down/permutation dimensions do not match")
    if not np.array_equal(np.sort(order), np.arange(matrix.shape[1])):
        raise ValueError("permutation must contain every record once")
    if shared.ndim != 1 or len(np.unique(shared)) != len(shared):
        raise ValueError("shared_indices must be unique and one-dimensional")
    if np.any(shared < 0) or np.any(shared >= matrix.shape[1]):
        raise ValueError("shared index lies outside the intermediate dimension")
    result = matrix[:, order].T.copy()
    result[np.isin(order, shared)] *= 0.5
    return result


def _expert_oracle_indices(
    states: np.ndarray,
    gate_blocks: np.ndarray,
    up_blocks: np.ndarray,
    down_blocks: np.ndarray,
    active_experts: int,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Greedily rank expert blocks by exact residual reduction in small batches."""

    hidden = np.asarray(states, dtype=np.float64)
    gate = np.asarray(gate_blocks, dtype=np.float64)
    up = np.asarray(up_blocks, dtype=np.float64)
    down = np.asarray(down_blocks, dtype=np.float64)
    if gate.ndim != 3 or gate.shape != up.shape or gate.shape != down.shape:
        raise ValueError("expert projection blocks must have matching [E, B, H] shapes")
    if hidden.ndim != 2 or hidden.shape[1] != gate.shape[2]:
        raise ValueError("states and expert hidden widths do not match")
    if not 0 < active_experts <= gate.shape[0]:
        raise ValueError("active_experts must lie within the expert count")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    result: list[np.ndarray] = []
    for start in range(0, len(hidden), batch_size):
        batch = hidden[start : start + batch_size]
        gate_values = np.einsum("nh,ebh->neb", batch, gate, optimize=True)
        up_values = np.einsum("nh,ebh->neb", batch, up, optimize=True)
        gate_values = gate_values / (1.0 + np.exp(-gate_values))
        contributions = np.einsum(
            "neb,ebh->neh", gate_values * up_values, down, optimize=True
        )
        result.append(_greedy_residual_experts(contributions, active_experts))
    return np.concatenate(result)


def _expert_membership(indices: np.ndarray, experts: int) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 2 or np.any(selected < 0) or np.any(selected >= experts):
        raise ValueError("expert indices are invalid")
    result = np.zeros((len(selected), experts), dtype=np.float64)
    result[np.arange(len(selected))[:, None], selected] = 1.0
    return result


def _curriculum_width(start: int, target: int, progress: float) -> int:
    if not 0 < target <= start:
        raise ValueError("require 0 < target <= start")
    clipped = min(1.0, max(0.0, float(progress)))
    return int(round(start + clipped * (target - start)))


def _wrap_shared_expert_mlp_class(torch: Any):
    """Build the hard-forward shared-plus-expert training module."""

    class SharedExpertMLP(torch.nn.Module):
        def __init__(
            self,
            base: Any,
            permutation: Any,
            shared_indices: Any,
            *,
            experts: int,
            active_experts: int,
            temperature: float = 1.0,
            router: MultiLabelLinearRouter | None = None,
        ):
            super().__init__()
            intermediate, hidden = base.gate_proj.weight.shape
            if intermediate % experts:
                raise ValueError("experts must divide the intermediate size")
            if not 0 < active_experts <= experts:
                raise ValueError("active_experts must lie within the expert count")
            if not np.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperature must be finite and positive")
            order = torch.as_tensor(permutation, dtype=torch.long)
            shared = torch.as_tensor(shared_indices, dtype=torch.long)
            if order.shape != (intermediate,) or not torch.equal(
                torch.sort(order).values, torch.arange(intermediate)
            ):
                raise ValueError("permutation must contain every intermediate record once")
            if (
                shared.ndim != 1
                or not len(shared)
                or len(torch.unique(shared)) != len(shared)
                or bool(torch.any(shared < 0))
                or bool(torch.any(shared >= intermediate))
            ):
                raise ValueError("shared_indices must contain unique valid records")
            if (
                base.gate_proj.bias is not None
                or base.up_proj.bias is not None
                or base.down_proj.bias is not None
            ):
                raise ValueError("bias-enabled MLP projections are not supported")
            block = intermediate // experts
            dtype = base.gate_proj.weight.dtype
            device = base.gate_proj.weight.device
            expert_gate = base.gate_proj.weight.detach()[order].reshape(
                experts, block, hidden
            ).clone()
            expert_up = base.up_proj.weight.detach()[order].reshape(
                experts, block, hidden
            ).clone()
            expert_down = base.down_proj.weight.detach().T[order].reshape(
                experts, block, hidden
            ).clone()
            shared_mask = torch.isin(order, shared).reshape(experts, block)
            expert_down[shared_mask] *= 0.5
            self.expert_gate = torch.nn.Parameter(expert_gate)
            self.expert_up = torch.nn.Parameter(expert_up)
            self.expert_down = torch.nn.Parameter(expert_down)
            self.shared_gate = torch.nn.Parameter(
                base.gate_proj.weight.detach()[shared].clone()
            )
            self.shared_up = torch.nn.Parameter(
                base.up_proj.weight.detach()[shared].clone()
            )
            self.shared_down = torch.nn.Parameter(
                0.5 * base.down_proj.weight.detach().T[shared].clone()
            )
            self.router = torch.nn.Linear(
                hidden, experts, bias=True, dtype=dtype, device=device
            )
            if router is None:
                torch.nn.init.zeros_(self.router.weight)
                torch.nn.init.zeros_(self.router.bias)
            else:
                if router.weights.shape != (hidden, experts):
                    raise ValueError("router dimensions do not match the expert layout")
                with torch.no_grad():
                    self.router.weight.copy_(
                        torch.as_tensor(router.weights.T, dtype=dtype, device=device)
                    )
                    self.router.bias.copy_(
                        torch.as_tensor(router.bias, dtype=dtype, device=device)
                    )
            self.act_fn = base.act_fn
            self.experts = experts
            self.records_per_expert = block
            self.active_experts = active_experts
            self.temperature = temperature
            self.mode = "hard"
            self.use_training_surrogate = True
            self.register_buffer("permutation", order.clone())
            self.register_buffer("shared_indices", shared.clone())
            self.last_active_experts = None
            self.last_router_logits = None
            self.last_dense_output = None
            self.last_surrogate_used = False
            self.last_output = None

        def _shared_output(self, flat: Any) -> Any:
            gate = torch.nn.functional.linear(flat, self.shared_gate)
            up = torch.nn.functional.linear(flat, self.shared_up)
            return torch.nn.functional.linear(
                self.act_fn(gate) * up, self.shared_down.T
            )

        def _dense_expert_blocks(self, flat: Any) -> Any:
            gate = torch.einsum("nh,ebh->neb", flat, self.expert_gate)
            up = torch.einsum("nh,ebh->neb", flat, self.expert_up)
            activation = self.act_fn(gate) * up
            return torch.einsum("neb,ebh->neh", activation, self.expert_down)

        def _hard_expert_output(self, flat: Any, active: Any) -> Any:
            gate_weight = self.expert_gate[active]
            up_weight = self.expert_up[active]
            down_weight = self.expert_down[active]
            gate = torch.einsum("nabh,nh->nab", gate_weight, flat)
            up = torch.einsum("nabh,nh->nab", up_weight, flat)
            activation = self.act_fn(gate) * up
            return torch.einsum("nab,nabh->nh", activation, down_weight)

        def set_active_experts(self, count: int) -> None:
            if not isinstance(count, int) or not 0 < count <= self.experts:
                raise ValueError("active expert count lies outside the layout")
            self.active_experts = count

        def forward(self, hidden: Any) -> Any:
            shape = hidden.shape
            flat = hidden.reshape(-1, shape[-1])
            shared_output = self._shared_output(flat)
            self.last_dense_output = None
            if self.mode == "dense_shadow":
                blocks = self._dense_expert_blocks(flat)
                output = shared_output + blocks.sum(dim=1)
                self.last_dense_output = output.reshape(*shape[:-1], -1)
                self.last_active_experts = None
                self.last_router_logits = None
                self.last_surrogate_used = False
            elif self.mode == "hard":
                logits = self.router(flat)
                active = torch.argsort(
                    logits, dim=1, descending=True, stable=True
                )[:, : self.active_experts]
                output = shared_output + self._hard_expert_output(flat, active)
                self.last_active_experts = active
                self.last_router_logits = logits
                self.last_surrogate_used = False
                if self.training and self.use_training_surrogate:
                    _, soft_mask = _cardinality_preserving_top_mask(
                        logits, self.active_experts, self.temperature, torch
                    )
                    blocks = self._dense_expert_blocks(flat)
                    self.last_dense_output = (
                        shared_output + blocks.sum(dim=1)
                    ).reshape(*shape[:-1], -1)
                    proxy = torch.sum(
                        soft_mask.unsqueeze(2) * blocks.detach(), dim=1
                    )
                    output = output + proxy - proxy.detach()
                    self.last_surrogate_used = True
            else:
                raise ValueError(f"unsupported shared expert mode {self.mode!r}")
            self.last_output = output.reshape(*shape[:-1], -1)
            return self.last_output

    return SharedExpertMLP


def train_shared_expert_boundaries(
    model: str | Path,
    training_traces: str | Path,
    validation_traces: str | Path,
    out: str | Path,
    *,
    layers: Sequence[int],
    shared_records: int = 128,
    experts: int = 96,
    active_experts: int = 32,
    start_active_experts: int = 48,
    grouping_iterations: int = 12,
    router_regularization: float = 3000.0,
    router_warmup_steps: int = 32,
    anneal_steps: int = 64,
    settle_steps: int = 128,
    batch_size: int = 128,
    oracle_batch_size: int = 64,
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
    device: str = "cpu",
) -> dict[str, Any]:
    """Train selected layers against cached teacher MLP input/output boundaries."""

    try:
        import torch
        import torch.nn.functional as functional
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for shared expert boundary training"
        ) from exc
    if not layers:
        raise ValueError("at least one layer is required")
    integer_values = (
        shared_records,
        experts,
        active_experts,
        start_active_experts,
        grouping_iterations,
        batch_size,
        oracle_batch_size,
        evaluation_interval,
    )
    if any(not isinstance(value, int) or value <= 0 for value in integer_values):
        raise ValueError("layout and training counts must be positive integers")
    if not 0 < active_experts <= start_active_experts <= experts:
        raise ValueError("require 0 < active_experts <= start_active_experts <= experts")
    if min(router_warmup_steps, anneal_steps, settle_steps) < 0:
        raise ValueError("phase step counts must be nonnegative")
    total_steps = router_warmup_steps + anneal_steps + settle_steps
    if total_steps <= 0:
        raise ValueError("at least one training step is required")
    positive_scalars = (
        router_regularization,
        learning_rate,
        router_learning_rate,
        start_temperature,
        temperature,
        maximum_mean_relative_l2,
    )
    if any(not np.isfinite(value) or value <= 0 for value in positive_scalars):
        raise ValueError("rates, temperatures, regularization, and threshold must be positive")
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
    if inspection.intermediate_size % experts:
        raise ValueError("experts must divide the source intermediate size")
    if not 0 < shared_records <= inspection.intermediate_size:
        raise ValueError("shared_records lies outside the source intermediate size")
    traffic = shared_expert_traffic(
        inspection.hidden_size,
        inspection.intermediate_size,
        shared_records=shared_records,
        experts=experts,
        active_experts=active_experts,
    )
    if traffic["fraction_of_dense_q4"] > 0.45:
        raise ValueError("requested shared expert layout exceeds the traffic gate")

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
            self.gate_proj = torch.nn.Linear(gate.shape[1], gate.shape[0], bias=False)
            self.up_proj = torch.nn.Linear(up.shape[1], up.shape[0], bias=False)
            self.down_proj = torch.nn.Linear(down.shape[1], down.shape[0], bias=False)
            with torch.no_grad():
                self.gate_proj.weight.copy_(torch.from_numpy(gate).float())
                self.up_proj.weight.copy_(torch.from_numpy(up).float())
                self.down_proj.weight.copy_(torch.from_numpy(down).float())
            self.act_fn = functional.silu

        def forward(self, hidden: Any) -> Any:
            return self.down_proj(
                self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden)
            )

    Wrapper = _wrap_shared_expert_mlp_class(torch)
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

        strength = _record_strength(train_input, gate, up, down)
        shared_indices = _select_shared_records(strength, shared_records)
        permutation = balanced_expert_permutation(
            strength.T, experts, iterations=grouping_iterations
        )
        block = inspection.intermediate_size // experts
        gate_blocks = gate[permutation].reshape(experts, block, inspection.hidden_size)
        up_blocks = up[permutation].reshape(experts, block, inspection.hidden_size)
        down_blocks = _halved_expert_down(
            down, permutation, shared_indices
        ).reshape(experts, block, inspection.hidden_size)
        train_oracle = _expert_oracle_indices(
            train_input,
            gate_blocks,
            up_blocks,
            down_blocks,
            start_active_experts,
            batch_size=oracle_batch_size,
        )
        validation_oracle = _expert_oracle_indices(
            validation_input,
            gate_blocks,
            up_blocks,
            down_blocks,
            active_experts,
            batch_size=oracle_batch_size,
        )
        fitted_router = MultiLabelLinearRouter.fit(
            train_input,
            _expert_membership(train_oracle[:, :active_experts], experts),
            regularization=router_regularization,
        )
        wrapper = Wrapper(
            DenseSwiGLU(gate, up, down),
            permutation,
            shared_indices,
            experts=experts,
            active_experts=start_active_experts,
            temperature=start_temperature,
            router=fitted_router,
        ).to(device)
        router_parameters = list(wrapper.router.parameters())
        mlp_parameters = [
            parameter
            for name, parameter in wrapper.named_parameters()
            if not name.startswith("router.") and parameter.requires_grad
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
        train_oracle_tensor = torch.from_numpy(train_oracle).to(device)
        validation_x = torch.from_numpy(validation_input.astype(np.float32)).to(device)
        validation_oracle_tensor = torch.from_numpy(validation_oracle).to(device)
        full_mask = torch.ones(len(train_x), dtype=torch.bool, device=device)

        wrapper.mode = "dense_shadow"
        wrapper.eval()
        with torch.inference_mode():
            initial_dense = wrapper(validation_x)
        initial_dense_relative, _ = _relative_and_cosine_rows(
            initial_dense.cpu().numpy(), validation_target
        )

        def evaluate(step: int) -> dict[str, Any]:
            wrapper.mode = "hard"
            wrapper.eval()
            wrapper.set_active_experts(active_experts)
            wrapper.temperature = temperature
            with torch.inference_mode():
                output = wrapper(validation_x)
                logits = wrapper.last_router_logits
                selected = torch.topk(logits, active_experts, dim=1).indices
                reference = validation_oracle_tensor[:, :active_experts]
                selected_mask = torch.zeros_like(logits, dtype=torch.bool).scatter(
                    1, selected, True
                )
                recall = selected_mask.gather(1, reference).float().mean()
            relative, cosine = _relative_and_cosine_rows(
                output.cpu().numpy(), validation_target
            )
            return {
                "step": step,
                "relative_l2": _stats(relative.tolist()),
                "cosine": _stats(cosine.tolist()),
                "expert_target_recall": float(recall.cpu()),
            }

        evaluations = [evaluate(0)]
        best = evaluations[0]
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in wrapper.state_dict().items()
        }
        generator = np.random.default_rng(19013 + layer)
        order = generator.permutation(len(train_x))
        cursor = 0
        training_first = None
        training_last = None
        wrapper.mode = "hard"
        wrapper.train()
        for step in range(total_steps):
            if cursor + batch_size > len(order):
                order = generator.permutation(len(train_x))
                cursor = 0
            batch_indices_numpy = order[cursor : cursor + batch_size]
            cursor += batch_size
            batch_indices = torch.as_tensor(
                batch_indices_numpy, dtype=torch.long, device=device
            )
            if step < router_warmup_steps:
                phase = "router_warmup"
                progress = 0.0
                width = start_active_experts
            elif step < router_warmup_steps + anneal_steps:
                phase = "anneal"
                offset = step - router_warmup_steps
                progress = offset / max(1, anneal_steps - 1)
                width = _curriculum_width(
                    start_active_experts, active_experts, progress
                )
            else:
                phase = "settle"
                progress = 1.0
                width = active_experts
            wrapper.set_active_experts(width)
            wrapper.temperature = start_temperature + progress * (
                temperature - start_temperature
            )
            x = train_x[batch_indices]
            target = train_y[batch_indices]
            route_indices = train_oracle_tensor[batch_indices, :width]
            route_target = torch.zeros(
                len(x), experts, dtype=x.dtype, device=device
            ).scatter(1, route_indices, 1.0)
            output = wrapper(x)
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
                    output, target, full_mask[batch_indices], torch
                )
                cosine_loss = (
                    1.0 - functional.cosine_similarity(output, target, dim=1)
                ).mean()
                dense_anchor = _normalized_masked_mse(
                    wrapper.last_dense_output,
                    target,
                    full_mask[batch_indices],
                    torch,
                )
                route_scale = route_weight * (1.0 - 0.9 * progress)
                anchor_scale = dense_anchor_weight * (1.0 - 0.8 * progress)
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
                "active_experts": width,
                "active_physical_records": shared_records + width * block,
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
                wrapper.mode = "hard"
                wrapper.train()
        wrapper.load_state_dict(best_state)
        wrapper.to(device)

        prefix = f"layer_{layer}."
        for name in (
            "shared_gate",
            "shared_up",
            "shared_down",
            "expert_gate",
            "expert_up",
            "expert_down",
        ):
            artifact_tensors[prefix + name] = (
                getattr(wrapper, name).detach().cpu().contiguous()
            )
        artifact_tensors[prefix + "router_weight"] = (
            wrapper.router.weight.detach().cpu().contiguous()
        )
        artifact_tensors[prefix + "router_bias"] = (
            wrapper.router.bias.detach().cpu().contiguous()
        )
        artifact_tensors[prefix + "permutation"] = wrapper.permutation.cpu().contiguous()
        artifact_tensors[prefix + "shared_indices"] = (
            wrapper.shared_indices.cpu().contiguous()
        )
        layer_reports.append(
            {
                "layer": layer,
                "training_records": len(train_input),
                "validation_records": len(validation_input),
                "initial_dense_parity_relative_l2": _stats(
                    initial_dense_relative.tolist()
                ),
                "initial": evaluations[0],
                "best": best,
                "final": evaluations[-1],
                "evaluations": evaluations,
                "training_first": training_first,
                "training_last": training_last,
            }
        )

    mean_best_relative_l2 = float(
        np.mean([row["best"]["relative_l2"]["mean"] for row in layer_reports])
    )
    checks = {
        "exact_initial_dense_parity": all(
            row["initial_dense_parity_relative_l2"]["maximum"] <= 1e-4
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
    target_path = Path(out)
    target_path.mkdir(parents=True, exist_ok=True)
    artifact_path = target_path / "shared_expert_boundaries.safetensors"
    save_file(
        artifact_tensors,
        artifact_path,
        metadata={
            "format": "engram_shared_expert_boundaries_v1",
            "source_model_hash": inspection.source_hash,
            "shared_records": str(shared_records),
            "experts": str(experts),
            "active_experts": str(active_experts),
        },
    )
    report = {
        "schema_version": 1,
        "experiment": "shared_expert_teacher_boundary_training",
        "source_model_hash": inspection.source_hash,
        "configuration": {
            "layers": selected_layers,
            "shared_records": shared_records,
            "experts": experts,
            "active_experts": active_experts,
            "start_active_experts": start_active_experts,
            "records_per_expert": inspection.intermediate_size // experts,
            "grouping_iterations": grouping_iterations,
            "router_regularization": router_regularization,
            "router_warmup_steps": router_warmup_steps,
            "anneal_steps": anneal_steps,
            "settle_steps": settle_steps,
            "batch_size": batch_size,
            "oracle_batch_size": oracle_batch_size,
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
            "device": device,
        },
        "data_separation": {
            "training_dataset_hash": training_reader.manifest["dataset_hash"],
            "validation_dataset_hash": validation_reader.manifest["dataset_hash"],
            "overlapping_sequences": 0,
            "held_out_from_gradient_training": True,
        },
        "layout": {
            "duplicate_shared_records": True,
            "initial_shared_down_scale": 0.5,
            "initial_expert_duplicate_down_scale": 0.5,
            "initial_dense_equivalence": "shared + all experts == source dense MLP",
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
                "The held-out boundary split selects checkpoints; the causal gate still "
                "requires untouched sequence-level intervention confirmation."
            ),
        },
        "artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": sha256_file(artifact_path),
        },
    }
    atomic_json(target_path / "shared_expert_boundaries.json", report)
    lines = [
        "# Shared-plus-coarse-expert boundary training",
        "",
        f"Decision: **{report['screen']['decision']}**",
        "",
        (
            f"The layout always reads {shared_records} shared records and routes "
            f"{active_experts} of {experts} cache-aligned experts "
            f"({traffic['active_physical_records']} physical records total)."
        ),
        "",
        "| Layer | Initial rel-L2 | Best rel-L2 | Final rel-L2 | Expert recall |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['layer']} | {row['initial']['relative_l2']['mean']:.6f} | "
        f"{row['best']['relative_l2']['mean']:.6f} | "
        f"{row['final']['relative_l2']['mean']:.6f} | "
        f"{row['best']['expert_target_recall']:.6f} |"
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
    (target_path / "shared_expert_boundaries.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


__all__ = [
    "_curriculum_width",
    "_expert_oracle_indices",
    "_halved_expert_down",
    "_select_shared_records",
    "_wrap_shared_expert_mlp_class",
    "shared_expert_traffic",
    "train_shared_expert_boundaries",
]
