"""Sparse-teacher fine-tuning with routed MLP co-adaptation and distillation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.evaluation.gates import (
    MINIMUM_EVALUATION_SEQUENCES,
    MINIMUM_NEXT_TOKEN_POSITIONS,
    MINIMUM_ROUTED_CANDIDATE_RECALL,
    MINIMUM_UNIQUE_EVALUATION_SEQUENCES,
    MLP_QUALITY_THRESHOLDS,
)
from engram.evaluation.mlp_intervention import (
    _evaluation_sequence_hashes,
    _quality_metrics,
    _relative_and_cosine_rows,
    _stats,
)
from engram.evaluation.router_sweep import _load_states, _membership, _sequence_hashes
from engram.models.inspection import inspect_model, resolve_model_path
from engram.semantic.dip import projected_dip_traffic
from engram.semantic.multilabel_router import LowRankMultiLabelRouter
from engram.semantic.swiglu import neuron_activations
from engram.tracing.format import TraceReader
from engram.training.grouped_sparse_codec import grouped_sparse_traffic
from engram.utils import atomic_json, sha256_file


def _load_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break
    if not records:
        raise ValueError("training or validation dataset contains no records")
    return records


def _ids(record: dict[str, Any], tokenizer: Any, torch: Any, device: str) -> Any:
    if "input_ids" in record:
        values = record["input_ids"]
        if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
            raise ValueError("input_ids must be a list of integers")
        return torch.tensor([values], dtype=torch.long, device=device)
    return tokenizer(str(record["text"]), return_tensors="pt")["input_ids"].to(device)


def _batch_ids(
    records: Sequence[dict[str, Any]], tokenizer: Any, torch: Any, device: str
) -> tuple[Any, Any, list[int]]:
    sequences = [_ids(record, tokenizer, torch, device).squeeze(0) for record in records]
    lengths = [int(sequence.shape[0]) for sequence in sequences]
    maximum = max(lengths)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    input_ids = torch.full(
        (len(sequences), maximum), int(pad_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        (len(sequences), maximum), dtype=torch.long, device=device
    )
    for index, sequence in enumerate(sequences):
        input_ids[index, : lengths[index]] = sequence
        attention_mask[index, : lengths[index]] = 1
    return input_ids, attention_mask, lengths


def _batches(records: Sequence[Any], size: int) -> Sequence[Sequence[Any]]:
    return [records[start : start + size] for start in range(0, len(records), size)]


def _masked_mean(values: Any, mask: Any, torch: Any) -> Any:
    weights = mask.to(dtype=values.dtype)
    return torch.sum(values * weights) / torch.clamp(torch.sum(weights), min=1.0)


def _normalized_masked_mse(
    approximation: Any, reference: Any, mask: Any, torch: Any
) -> Any:
    weights = mask.to(dtype=approximation.dtype)
    squared_error = torch.mean((approximation - reference) ** 2, dim=-1)
    squared_reference = torch.mean(reference**2, dim=-1)
    return torch.sum(squared_error * weights) / torch.clamp(
        torch.sum(squared_reference * weights), min=1e-8
    )


def _same_input_teacher_mlp_targets(
    teacher_mlps: Sequence[Any], student_inputs: Sequence[Any], torch: Any
) -> list[Any]:
    """Evaluate frozen teacher MLPs on stop-gradient student MLP inputs."""

    if len(teacher_mlps) != len(student_inputs):
        raise ValueError("teacher MLP and student input counts must agree")
    with torch.no_grad():
        return [
            teacher_mlp(student_input.detach()).detach()
            for teacher_mlp, student_input in zip(
                teacher_mlps, student_inputs, strict=True
            )
        ]


def _cardinality_preserving_top_mask(
    logits: Any, count: int, temperature: float, torch: Any
) -> tuple[Any, Any]:
    """Return hard top-C indices with a fixed-mass soft backward relaxation."""

    top = torch.topk(logits, count, dim=1, sorted=False).indices
    hard = torch.zeros_like(logits).scatter(1, top, 1.0)
    with torch.no_grad():
        lower = logits.min(dim=1, keepdim=True).values - 20.0 * temperature
        upper = logits.max(dim=1, keepdim=True).values + 20.0 * temperature
        for _ in range(32):
            threshold = (lower + upper) * 0.5
            mass = torch.sigmoid((logits - threshold) / temperature).sum(
                dim=1, keepdim=True
            )
            lower = torch.where(mass > count, threshold, lower)
            upper = torch.where(mass > count, upper, threshold)
        threshold_value = (lower + upper) * 0.5
        probabilities = torch.sigmoid(
            (logits - threshold_value) / temperature
        )
        threshold_weights = probabilities * (1.0 - probabilities)
    # Add the implicit derivative of sum(sigmoid((z-tau)/T)) = C while
    # retaining the bisection value. This makes common logit shifts a null
    # direction, matching hard top-C selection.
    weighted_logit = torch.sum(threshold_weights * logits, dim=1, keepdim=True)
    weighted_logit = weighted_logit / torch.clamp(
        torch.sum(threshold_weights, dim=1, keepdim=True), min=1e-8
    )
    threshold = threshold_value + weighted_logit - weighted_logit.detach()
    soft = torch.sigmoid((logits - threshold) / temperature)
    soft = soft * (
        float(count) / torch.clamp(soft.sum(dim=1, keepdim=True), min=1e-8)
    )
    return top, hard + soft - soft.detach()


def _group_utility_membership(
    states: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    selected_records: int,
    group_size: int,
) -> np.ndarray:
    """Return a realizable fixed-cardinality target for complete groups."""

    if group_size <= 0 or gate.shape[0] % group_size:
        raise ValueError("group_size must positively divide the intermediate size")
    if (
        selected_records <= 0
        or selected_records > gate.shape[0]
        or selected_records % group_size
    ):
        raise ValueError(
            "selected_records must lie within the intermediate size and be group aligned"
        )
    activations = neuron_activations(states, gate, up)
    record_utility = np.abs(activations) * np.linalg.norm(down, axis=0)[None, :]
    group_utility = record_utility.reshape(
        len(states), gate.shape[0] // group_size, group_size
    ).sum(axis=2)
    selected_groups = selected_records // group_size
    order = np.argsort(-group_utility, axis=1, kind="stable")[:, :selected_groups]
    membership = np.zeros(group_utility.shape, dtype=bool)
    membership[np.arange(len(states))[:, None], order] = True
    return membership


def _hard_line_fraction_with_smooth_gradient(
    line_mass: Any, hard_lines: Any, torch: Any
) -> Any:
    """Use exact occupied-line fraction forward and a concave soft gradient."""

    hard_fraction = hard_lines.sum(dim=1) / hard_lines.shape[1]
    smooth_fraction = (1.0 - torch.exp(-0.25 * line_mass)).sum(
        dim=1
    ) / line_mass.shape[1]
    return hard_fraction + smooth_fraction - smooth_fraction.detach()


def _wrap_sparse_mlp_class(torch: Any):
    class SparseStudentMLP(torch.nn.Module):
        def __init__(
            self,
            base: Any,
            router: LowRankMultiLabelRouter,
            *,
            top_k: int,
            candidates: int,
            adapter_rank: int,
            router_group_size: int = 1,
            train_full_mlp: bool = False,
        ):
            super().__init__()
            self.base = base
            for parameter in self.base.parameters():
                parameter.requires_grad_(train_full_mlp)
            self.train_full_mlp = train_full_mlp
            dtype = base.down_proj.weight.dtype
            self.router_input = torch.nn.Parameter(torch.tensor(router.input_factors, dtype=dtype))
            self.router_output = torch.nn.Parameter(torch.tensor(router.output_factors, dtype=dtype))
            self.router_bias = torch.nn.Parameter(torch.tensor(router.bias, dtype=dtype))
            self.router_nonlinear_scale = torch.nn.Parameter(
                torch.zeros(self.router_input.shape[1], dtype=dtype)
            )
            intermediate = base.down_proj.weight.shape[1]
            hidden = base.down_proj.weight.shape[0]
            if router_group_size <= 0 or intermediate % router_group_size:
                raise ValueError(
                    "router_group_size must positively divide the intermediate size"
                )
            if top_k % router_group_size or candidates % router_group_size:
                raise ValueError("top_k and candidates must be router-group aligned")
            expected_outputs = intermediate // router_group_size
            if self.router_output.shape[1] != expected_outputs:
                raise ValueError(
                    "router output width does not match intermediate/router_group_size"
                )
            generator = torch.Generator(device="cpu").manual_seed(1701 + intermediate + hidden)
            self.adapter_a = torch.nn.Parameter(
                torch.empty(intermediate, adapter_rank, dtype=dtype)
            )
            torch.nn.init.kaiming_uniform_(
                self.adapter_a, a=np.sqrt(5), generator=generator
            )
            self.adapter_b = torch.nn.Parameter(torch.zeros(adapter_rank, hidden, dtype=dtype))
            self.gate_adapter_a = torch.nn.Parameter(
                torch.empty(hidden, adapter_rank, dtype=dtype)
            )
            torch.nn.init.kaiming_uniform_(
                self.gate_adapter_a, a=np.sqrt(5), generator=generator
            )
            self.gate_adapter_b = torch.nn.Parameter(
                torch.zeros(adapter_rank, intermediate, dtype=dtype)
            )
            self.up_adapter_a = torch.nn.Parameter(
                torch.empty(hidden, adapter_rank, dtype=dtype)
            )
            torch.nn.init.kaiming_uniform_(
                self.up_adapter_a, a=np.sqrt(5), generator=generator
            )
            self.up_adapter_b = torch.nn.Parameter(
                torch.zeros(adapter_rank, intermediate, dtype=dtype)
            )
            self.adapter_scale = 1.0
            self.temperature = 1.0
            self.top_k = top_k
            self.candidates = candidates
            self.router_group_size = router_group_size
            self.mode = "trained"
            self.last_output = None
            self.last_router_logits = None
            self.last_oracle = None
            self.last_router_target = None
            self.last_recall = None
            self.last_candidate_ids = None
            self.last_active = None

        def forward(self, hidden: Any) -> Any:
            shape = hidden.shape
            flat = hidden.reshape(-1, shape[-1])
            if self.mode == "identity":
                self.last_output = self.base(flat).reshape(*shape[:-1], -1)
                self.last_router_logits = None
                self.last_recall = None
                return self.last_output
            value_norms = torch.linalg.vector_norm(
                self.base.down_proj.weight.detach(), dim=0
            )
            with torch.no_grad():
                oracle_activations = self.base.act_fn(
                    self.base.gate_proj(flat)
                ) * self.base.up_proj(flat)
                exact_scores = (
                    torch.abs(oracle_activations) * value_norms.unsqueeze(0)
                )
                oracle = torch.argsort(
                    exact_scores, dim=1, descending=True, stable=True
                )[:, : self.top_k]
            self.last_oracle = oracle
            if self.mode == "oracle":
                active = oracle
                active_values = oracle_activations.gather(1, active)
                self.last_router_logits = None
                self.last_recall = None
                soft_output = None
            else:
                router_latent = flat @ self.router_input
                router_features = router_latent + self.router_nonlinear_scale * (
                    torch.nn.functional.silu(router_latent)
                )
                logits = router_features @ self.router_output + self.router_bias
                selected_group_count = self.candidates // self.router_group_size
                group_ids, group_ste = _cardinality_preserving_top_mask(
                    logits, selected_group_count, self.temperature, torch
                )
                group_offsets = torch.arange(
                    self.router_group_size,
                    device=group_ids.device,
                    dtype=group_ids.dtype,
                )
                candidate_ids = (
                    group_ids.unsqueeze(-1) * self.router_group_size
                    + group_offsets
                ).reshape(len(flat), self.candidates)
                candidate_ste = group_ste.repeat_interleave(
                    self.router_group_size, dim=1
                )
                # Training expresses the mergeable LoRA projections densely to
                # avoid materializing [tokens, candidates, hidden] gathers on
                # modest GPUs. Deployment gathers only selected records.
                adapted_gate = self.base.gate_proj(flat) + (
                    (flat @ self.gate_adapter_a) @ self.gate_adapter_b
                ) * self.adapter_scale
                adapted_up = self.base.up_proj(flat) + (
                    (flat @ self.up_adapter_a) @ self.up_adapter_b
                ) * self.adapter_scale
                adapted_activations = self.base.act_fn(adapted_gate) * adapted_up
                candidate_activations = adapted_activations.gather(1, candidate_ids)
                candidate_scores = (
                    torch.abs(candidate_activations) * value_norms[candidate_ids]
                )
                local = torch.argsort(candidate_scores, dim=1, descending=True, stable=True)[:, : self.top_k]
                active = candidate_ids.gather(1, local)
                active_values = candidate_activations.gather(1, local)
                mask = torch.zeros_like(exact_scores, dtype=torch.bool)
                mask.scatter_(1, candidate_ids, True)
                self.last_recall = mask.gather(1, oracle).float().mean(dim=1)
                self.last_router_logits = logits
                group_utility = exact_scores.reshape(
                    len(flat),
                    exact_scores.shape[1] // self.router_group_size,
                    self.router_group_size,
                ).sum(dim=2)
                oracle_groups = torch.topk(
                    group_utility,
                    self.top_k // self.router_group_size,
                    dim=1,
                    sorted=False,
                ).indices
                self.last_router_target = torch.zeros_like(logits).scatter(
                    1, oracle_groups, 1.0
                )
                self.last_candidate_ids = candidate_ids
                self.last_active = active
                merged_down = self.base.down_proj.weight + (
                    self.adapter_b.T @ self.adapter_a.T
                ) * self.adapter_scale
                # The STE must value records under the current student, not
                # the frozen teacher decomposition. Detach values/weights so
                # this surrogate updates only the router; the exact hard path
                # above supplies MLP gradients.
                soft_output = torch.nn.functional.linear(
                    adapted_activations.detach() * candidate_ste,
                    merged_down.detach(),
                    self.base.down_proj.bias,
                )
            selected = torch.zeros(
                len(flat),
                self.base.down_proj.weight.shape[1],
                device=flat.device,
                dtype=flat.dtype,
            ).scatter(1, active, active_values)
            output = self.base.down_proj(selected)
            if self.mode == "trained":
                adapter_hidden = selected @ self.adapter_a
                output = output + (
                    adapter_hidden @ self.adapter_b
                ) * self.adapter_scale
                output = output + (soft_output - soft_output.detach())
            self.last_output = output.reshape(*shape[:-1], -1)
            return self.last_output

    return SparseStudentMLP


def _wrap_hardware_sparse_mlp_class(torch: Any):
    """Build a hard-forward/soft-backward DIP student MLP.

    Deployment uses exact top-q input coordinates, top-C proxy candidates, and
    exact top-K reranking. During training, straight-through sigmoid masks let
    local, hidden-state, and logit losses update the router factors.
    """

    class HardwareSparseStudentMLP(torch.nn.Module):
        def __init__(
            self,
            base: Any,
            router: LowRankMultiLabelRouter,
            *,
            top_k: int,
            candidates: int,
            input_fraction: float,
            adapter_rank: int,
            residual_rank: int,
            temperature: float,
            cache_line_records: int,
        ):
            super().__init__()
            self.base = base
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
            dtype = base.down_proj.weight.dtype
            self.router_input = torch.nn.Parameter(
                torch.tensor(router.input_factors, dtype=dtype)
            )
            self.router_output = torch.nn.Parameter(
                torch.tensor(router.output_factors, dtype=dtype)
            )
            self.router_bias = torch.nn.Parameter(torch.tensor(router.bias, dtype=dtype))
            self.router_blend_logit = torch.nn.Parameter(
                torch.tensor(-2.0, dtype=dtype)
            )
            intermediate = base.down_proj.weight.shape[1]
            hidden = base.down_proj.weight.shape[0]
            generator = torch.Generator(device="cpu").manual_seed(
                2903 + intermediate + hidden
            )
            self.adapter_a = torch.nn.Parameter(
                torch.empty(intermediate, adapter_rank, dtype=dtype)
            )
            torch.nn.init.kaiming_uniform_(
                self.adapter_a, a=np.sqrt(5), generator=generator
            )
            self.adapter_b = torch.nn.Parameter(
                torch.zeros(adapter_rank, hidden, dtype=dtype)
            )
            self.gate_adapter_a = torch.nn.Parameter(
                torch.empty(hidden, adapter_rank, dtype=dtype)
            )
            torch.nn.init.kaiming_uniform_(
                self.gate_adapter_a, a=np.sqrt(5), generator=generator
            )
            self.gate_adapter_b = torch.nn.Parameter(
                torch.zeros(adapter_rank, intermediate, dtype=dtype)
            )
            self.up_adapter_a = torch.nn.Parameter(
                torch.empty(hidden, adapter_rank, dtype=dtype)
            )
            torch.nn.init.kaiming_uniform_(
                self.up_adapter_a, a=np.sqrt(5), generator=generator
            )
            self.up_adapter_b = torch.nn.Parameter(
                torch.zeros(adapter_rank, intermediate, dtype=dtype)
            )
            self.adapter_scale = 1.0
            if residual_rank:
                self.residual_a = torch.nn.Parameter(
                    torch.empty(hidden, residual_rank, dtype=dtype)
                )
                torch.nn.init.kaiming_uniform_(
                    self.residual_a, a=np.sqrt(5), generator=generator
                )
                self.residual_b = torch.nn.Parameter(
                    torch.zeros(residual_rank, hidden, dtype=dtype)
                )
            else:
                self.register_parameter("residual_a", None)
                self.register_parameter("residual_b", None)
            self.residual_scale = 1.0
            self.residual_rank = residual_rank
            self.top_k = top_k
            self.candidates = candidates
            self.input_fraction = input_fraction
            self.temperature = temperature
            self.cache_line_records = cache_line_records
            self.mode = "trained"
            self.last_input = None
            self.last_output = None
            self.last_router_logits = None
            self.last_oracle = None
            self.last_recall = None
            self.last_locality_loss = None
            self.last_locality_rows = None
            self.last_occupied_lines = None
            self.last_soft_candidate = None
            self.last_candidate_ids = None
            self.last_active = None
            self.last_pre_residual = None
            self.last_residual = None

        def _soft_top_mask(self, logits: Any, count: int) -> tuple[Any, Any]:
            return _cardinality_preserving_top_mask(
                logits, count, self.temperature, torch
            )

        def forward(self, hidden: Any) -> Any:
            self.last_input = hidden.detach()
            shape = hidden.shape
            flat = hidden.reshape(-1, shape[-1])
            if self.mode == "identity":
                self.last_output = self.base(flat).reshape(*shape[:-1], -1)
                self.last_router_logits = None
                self.last_recall = None
                self.last_pre_residual = None
                self.last_residual = None
                return self.last_output
            value_norms = torch.linalg.vector_norm(
                self.base.down_proj.weight.detach(), dim=0
            )
            with torch.no_grad():
                oracle_activations = self.base.act_fn(
                    self.base.gate_proj(flat)
                ) * self.base.up_proj(flat)
                oracle_scores = (
                    torch.abs(oracle_activations) * value_norms.unsqueeze(0)
                )
                oracle = torch.argsort(
                    oracle_scores, dim=1, descending=True, stable=True
                )[:, : self.top_k]
            self.last_oracle = oracle
            if self.mode == "oracle":
                active = oracle
                active_values = oracle_activations.gather(1, active)
                self.last_router_logits = None
                self.last_recall = None
                self.last_locality_loss = None
                self.last_locality_rows = None
                self.last_occupied_lines = None
                soft_output = None
            else:
                q = max(1, min(flat.shape[1], round(self.input_fraction * flat.shape[1])))
                coordinates = torch.argsort(
                    torch.abs(flat), dim=1, descending=True, stable=True
                )[:, :q]
                partial_hidden = torch.zeros_like(flat).scatter(
                    1, coordinates, flat.gather(1, coordinates)
                )
                partial_gate = self.base.gate_proj(partial_hidden) + (
                    (partial_hidden @ self.gate_adapter_a) @ self.gate_adapter_b
                ) * self.adapter_scale
                partial_up = self.base.up_proj(partial_hidden) + (
                    (partial_hidden @ self.up_adapter_a) @ self.up_adapter_b
                ) * self.adapter_scale
                partial = self.base.act_fn(partial_gate) * partial_up
                proxy_logits = torch.log(
                    torch.abs(partial) * value_norms.unsqueeze(0) + 1e-8
                )
                learned = (
                    (flat @ self.router_input) @ self.router_output + self.router_bias
                )
                logits = proxy_logits + torch.sigmoid(self.router_blend_logit) * learned
                candidate_ids, candidate_ste = self._soft_top_mask(
                    logits, self.candidates
                )
                gate_weights = self.base.gate_proj.weight[candidate_ids]
                up_weights = self.base.up_proj.weight[candidate_ids]
                candidate_gate = torch.einsum("bch,bh->bc", gate_weights, flat)
                candidate_up = torch.einsum("bch,bh->bc", up_weights, flat)
                gate_latent = flat @ self.gate_adapter_a
                up_latent = flat @ self.up_adapter_a
                candidate_gate = candidate_gate + torch.einsum(
                    "br,bcr->bc",
                    gate_latent,
                    self.gate_adapter_b.T[candidate_ids],
                ) * self.adapter_scale
                candidate_up = candidate_up + torch.einsum(
                    "br,bcr->bc",
                    up_latent,
                    self.up_adapter_b.T[candidate_ids],
                ) * self.adapter_scale
                if self.base.gate_proj.bias is not None:
                    candidate_gate = (
                        candidate_gate + self.base.gate_proj.bias[candidate_ids]
                    )
                if self.base.up_proj.bias is not None:
                    candidate_up = candidate_up + self.base.up_proj.bias[candidate_ids]
                candidate_activations = (
                    self.base.act_fn(candidate_gate) * candidate_up
                )
                candidate_scores = (
                    torch.abs(candidate_activations)
                    * value_norms[candidate_ids]
                )
                local = torch.argsort(
                    candidate_scores, dim=1, descending=True, stable=True
                )[:, : self.top_k]
                active = candidate_ids.gather(1, local)
                active_values = candidate_activations.gather(1, local)
                if self.top_k == self.candidates:
                    soft_active = candidate_ste
                else:
                    threshold = torch.topk(
                        candidate_scores, self.top_k, dim=1, sorted=False
                    ).values.min(dim=1, keepdim=True).values.detach()
                    soft_exact = torch.sigmoid(
                        (proxy_logits - torch.log(threshold + 1e-8))
                        / self.temperature
                    )
                    soft_active = candidate_ste * soft_exact
                candidate_hard = torch.zeros_like(logits).scatter(
                    1, candidate_ids, 1.0
                )
                self.last_recall = candidate_hard.gather(1, oracle).mean(dim=1)
                line_count = (
                    candidate_hard.shape[1] + self.cache_line_records - 1
                ) // self.cache_line_records
                padded = torch.nn.functional.pad(
                    candidate_ste,
                    (0, line_count * self.cache_line_records - candidate_ste.shape[1]),
                )
                line_mass = padded.reshape(
                    len(padded), line_count, self.cache_line_records
                ).sum(dim=2)
                hard_lines = torch.zeros(
                    len(candidate_ids),
                    line_count,
                    device=hidden.device,
                    dtype=hidden.dtype,
                ).scatter(1, candidate_ids // self.cache_line_records, 1.0)
                self.last_locality_rows = _hard_line_fraction_with_smooth_gradient(
                    line_mass, hard_lines, torch
                )
                self.last_locality_loss = self.last_locality_rows.mean()
                self.last_occupied_lines = hard_lines.sum(dim=1)
                self.last_soft_candidate = candidate_ste
                self.last_router_logits = logits
                self.last_candidate_ids = candidate_ids
                soft_selected = partial * soft_active
                soft_output = self.base.down_proj(soft_selected)
            self.last_active = active
            down_weights = self.base.down_proj.weight.T[active]
            output = torch.einsum("bk,bkh->bh", active_values, down_weights)
            if self.base.down_proj.bias is not None:
                output = output + self.base.down_proj.bias
            if self.mode == "trained":
                adapter_a = self.adapter_a[active]
                adapter_hidden = torch.einsum(
                    "bk,bkr->br", active_values, adapter_a
                )
                output = output + (
                    adapter_hidden @ self.adapter_b
                ) * self.adapter_scale
                if self.residual_rank:
                    self.last_pre_residual = output.reshape(*shape[:-1], -1)
                    residual = (
                        (flat @ self.residual_a) @ self.residual_b
                    ) * self.residual_scale
                    self.last_residual = residual.reshape(*shape[:-1], -1)
                    output = output + residual
                else:
                    self.last_pre_residual = None
                    self.last_residual = None
                output = output + (soft_output - soft_output.detach())
            self.last_output = output.reshape(*shape[:-1], -1)
            return self.last_output

    return HardwareSparseStudentMLP


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _direct_router_traffic(
    hidden_size: int,
    intermediate_size: int,
    *,
    candidate_count: int,
    top_k: int,
    router_rank: int,
    router_group_size: int,
    cache_line_bytes: int = 64,
) -> dict[str, int | float | bool | str]:
    """Account exact cold bytes for the packed grouped Q4 layout.

    Requiring ``candidate_count == top_k`` means every fetched group is also
    executed and avoids pretending that a partial down-row fetch is free.  The
    delegated layout includes Q4 factor/MLP codes, all FP16 decode scales,
    router metadata, selected IDs, the format header, and cache-line padding.
    """

    if min(hidden_size, intermediate_size, candidate_count, top_k, router_rank) <= 0:
        raise ValueError("direct-router dimensions and counts must be positive")
    if router_group_size <= 0 or cache_line_bytes <= 0:
        raise ValueError("router_group_size and cache_line_bytes must be positive")
    if candidate_count != top_k:
        raise ValueError("grouped direct routing requires candidate_count == top_k")
    if (
        intermediate_size % router_group_size
        or candidate_count % router_group_size
    ):
        raise ValueError("intermediate and selected counts must be group aligned")

    return grouped_sparse_traffic(
        hidden_size,
        intermediate_size,
        selected_records=candidate_count,
        router_rank=router_rank,
        group_size=router_group_size,
        cache_line_bytes=cache_line_bytes,
    )


def _aligned_router_width(
    start: int, target: int, progress: float, group_size: int
) -> int:
    if group_size <= 0 or start % group_size or target % group_size:
        raise ValueError("router curriculum endpoints must be group aligned")
    clipped = min(1.0, max(0.0, float(progress)))
    raw = start + clipped * (target - start)
    aligned = int(round(raw / group_size)) * group_size
    return min(max(aligned, min(start, target)), max(start, target))


def _curriculum_progress(
    step: int, dense_warmup_steps: int, anneal_steps: int
) -> float:
    """Return a deterministic dense-warmup then annealing progress value."""

    if min(step, dense_warmup_steps, anneal_steps) < 0:
        raise ValueError("curriculum steps must be nonnegative")
    if step < dense_warmup_steps:
        return 0.0
    if anneal_steps == 0:
        return 1.0
    return min(1.0, (step - dense_warmup_steps) / anneal_steps)


def _progressive_hardware_schedule(
    *,
    step: int,
    dense_warmup_steps: int,
    anneal_steps: int,
    start_input_fraction: float,
    target_input_fraction: float,
    start_candidate_count: int,
    target_candidate_count: int,
    start_top_k: int,
    target_top_k: int,
    start_temperature: float,
    target_temperature: float,
) -> dict[str, int | float | bool]:
    """Interpolate an exact dense hardware graph to its deployment endpoint."""

    if not (
        0.0 < start_input_fraction <= 1.0
        and 0.0 < target_input_fraction <= 1.0
    ):
        raise ValueError("hardware input fractions must be in (0, 1]")
    if (
        min(
            start_candidate_count,
            target_candidate_count,
            start_top_k,
            target_top_k,
        )
        <= 0
        or start_top_k > start_candidate_count
        or target_top_k > target_candidate_count
    ):
        raise ValueError("hardware schedule requires 0 < K <= C at both endpoints")
    if min(start_temperature, target_temperature) <= 0.0:
        raise ValueError("hardware temperatures must be positive")
    progress = _curriculum_progress(step, dense_warmup_steps, anneal_steps)
    candidate_count = _aligned_router_width(
        start_candidate_count, target_candidate_count, progress, 1
    )
    top_k = _aligned_router_width(start_top_k, target_top_k, progress, 1)
    candidate_count = max(candidate_count, top_k)
    return {
        "progress": progress,
        "input_fraction": start_input_fraction
        + progress * (target_input_fraction - start_input_fraction),
        "candidate_count": candidate_count,
        "top_k": top_k,
        "temperature": start_temperature
        + progress * (target_temperature - start_temperature),
        "deployment_endpoint": progress >= 1.0,
    }


def _selected_layer_indices(
    layers: Sequence[int] | None, num_hidden_layers: int
) -> tuple[int, ...]:
    if num_hidden_layers <= 0:
        raise ValueError("model must contain at least one hidden layer")
    if layers is None:
        return tuple(range(num_hidden_layers))
    selected = tuple(dict.fromkeys(int(layer) for layer in layers))
    if not selected or any(
        layer < 0 or layer >= num_hidden_layers for layer in selected
    ):
        raise ValueError("layers must contain valid hidden-layer indices")
    return selected


def _train_internal_checkpoint_split(
    records: Sequence[Any], checkpoint_records: int, seed: int
) -> tuple[list[Any], list[Any], list[int], list[int]]:
    """Reserve a deterministic train-only split for checkpoint selection."""

    if checkpoint_records < 0 or checkpoint_records >= len(records):
        raise ValueError(
            "checkpoint selection records must be nonnegative and smaller "
            "than the training corpus"
        )
    if checkpoint_records == 0:
        indices = list(range(len(records)))
        return list(records), [], indices, []
    order = np.random.default_rng(seed).permutation(len(records))
    checkpoint_indices = [int(index) for index in order[:checkpoint_records]]
    fit_indices = [int(index) for index in order[checkpoint_records:]]
    return (
        [records[index] for index in fit_indices],
        [records[index] for index in checkpoint_indices],
        fit_indices,
        checkpoint_indices,
    )


def train_sparse_student(
    model: str | Path,
    calibration_dataset: str | Path,
    validation_dataset: str | Path,
    calibration_traces: str | Path,
    out: str | Path,
    *,
    top_k: int = 512,
    candidate_count: int = 512,
    router_rank: int = 16,
    router_regularization: float = 8000.0,
    adapter_rank: int = 8,
    residual_rank: int = 0,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    router_learning_rate: float = 1e-3,
    local_weight: float = 1.0,
    hidden_weight: float = 0.25,
    logit_weight: float = 0.25,
    teacher_forced_local_weight: float = 0.0,
    router_weight: float = 0.1,
    locality_weight: float = 0.05,
    routing_mode: str = "hardware_ste",
    input_fraction: float = 0.625,
    temperature: float = 1.0,
    cache_line_records: int = 16,
    batch_size: int = 4,
    gradient_diagnostics: bool = False,
    checkpoint_every: int = 0,
    resume: bool = False,
    start_top_k: int | None = None,
    anneal_steps: int = 0,
    start_temperature: float | None = None,
    router_group_size: int = 1,
    train_full_mlp: bool = False,
    training_dataset: str | Path | None = None,
    max_train_records: int | None = None,
    max_validation_records: int | None = None,
    layers: Sequence[int] | None = None,
    exact_dense_start: bool = False,
    dense_warmup_steps: int = 0,
    start_candidate_count: int | None = None,
    start_input_fraction: float | None = None,
    checkpoint_selection_records: int = 0,
    checkpoint_selection_every: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        from safetensors.torch import load_file, save_file
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as transformers_imports

        if transformers_imports.is_sklearn_available():
            try:
                import sklearn  # noqa: F401
            except ImportError:

                def sklearn_unavailable() -> bool:
                    return False

                transformers_imports.is_sklearn_available = sklearn_unavailable
                transformers_utils.is_sklearn_available = sklearn_unavailable
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install engram-lm[conversion] for sparse-teacher training") from exc
    if (
        epochs <= 0
        or learning_rate <= 0.0
        or router_learning_rate <= 0.0
        or adapter_rank <= 0
        or residual_rank < 0
    ):
        raise ValueError(
            "epochs, learning rates, and adapter_rank must be positive; residual_rank must be nonnegative"
        )
    if routing_mode not in {"hard_router", "hardware_ste"}:
        raise ValueError("routing_mode must be hard_router or hardware_ste")
    if router_group_size <= 0:
        raise ValueError("router_group_size must be positive")
    if routing_mode == "hardware_ste" and router_group_size != 1:
        raise ValueError("router_group_size is only supported by hard_router mode")
    if gradient_diagnostics and routing_mode != "hardware_ste":
        raise ValueError("gradient diagnostics require hardware_ste routing")
    if not 0.0 < input_fraction <= 1.0:
        raise ValueError("input_fraction must be in (0, 1]")
    if (
        temperature <= 0.0
        or cache_line_records <= 0
        or batch_size <= 0
        or checkpoint_every < 0
        or anneal_steps < 0
        or dense_warmup_steps < 0
        or checkpoint_selection_records < 0
        or checkpoint_selection_every < 0
        or locality_weight < 0.0
        or teacher_forced_local_weight < 0.0
    ):
        raise ValueError(
            "temperature/cache_line_records/batch_size must be positive; "
            "checkpoint/curriculum steps, checkpoint selection records, "
            "locality_weight, and teacher_forced_local_weight must be nonnegative"
        )
    if bool(checkpoint_selection_records) != bool(checkpoint_selection_every):
        raise ValueError(
            "checkpoint_selection_records and checkpoint_selection_every "
            "must either both be zero or both be positive"
        )
    if exact_dense_start and routing_mode != "hardware_ste":
        raise ValueError("exact_dense_start currently requires hardware_ste routing")
    if dense_warmup_steps and routing_mode != "hardware_ste":
        raise ValueError("dense_warmup_steps currently requires hardware_ste routing")
    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    layer_indices = _selected_layer_indices(layers, inspection.num_hidden_layers)
    if not 0 < top_k <= candidate_count <= inspection.intermediate_size:
        raise ValueError("require 0 < top_k <= candidate_count <= intermediate size")
    if routing_mode == "hard_router" and candidate_count != top_k:
        raise ValueError(
            "cache-line-honest hard_router training requires candidates == top_k"
        )
    if exact_dense_start and any(
        value is not None
        for value in (start_top_k, start_candidate_count, start_input_fraction)
    ):
        raise ValueError(
            "exact_dense_start cannot be combined with explicit hardware "
            "curriculum endpoints"
        )
    initial_top_k = (
        inspection.intermediate_size
        if exact_dense_start
        else top_k if start_top_k is None else int(start_top_k)
    )
    initial_candidate_count = (
        inspection.intermediate_size
        if exact_dense_start
        else initial_top_k
        if routing_mode == "hard_router" and start_candidate_count is None
        else candidate_count
        if start_candidate_count is None
        else int(start_candidate_count)
    )
    initial_input_fraction = (
        1.0
        if exact_dense_start
        else input_fraction
        if start_input_fraction is None
        else float(start_input_fraction)
    )
    if not top_k <= initial_top_k <= inspection.intermediate_size:
        raise ValueError("start_top_k must lie between top_k and intermediate size")
    if not candidate_count <= initial_candidate_count <= inspection.intermediate_size:
        raise ValueError(
            "start_candidate_count must lie between candidates and intermediate size"
        )
    if initial_top_k > initial_candidate_count:
        raise ValueError("start_top_k must not exceed start_candidate_count")
    if not input_fraction <= initial_input_fraction <= 1.0:
        raise ValueError(
            "start_input_fraction must lie between input_fraction and 1"
        )
    if routing_mode == "hard_router" and (
        initial_candidate_count != initial_top_k
        or initial_input_fraction != input_fraction
    ):
        raise ValueError(
            "hard_router supports only the existing top-k-only curriculum"
        )
    if train_full_mlp and routing_mode != "hard_router":
        raise ValueError("train_full_mlp currently requires hard_router mode")
    if teacher_forced_local_weight and routing_mode != "hard_router":
        raise ValueError(
            "teacher-forced local distillation currently requires hard_router mode"
        )
    if routing_mode == "hard_router" and (
        inspection.intermediate_size % router_group_size
        or top_k % router_group_size
        or candidate_count % router_group_size
        or initial_top_k % router_group_size
    ):
        raise ValueError(
            "hard_router intermediate, final, and curriculum widths must be divisible by router_group_size"
        )
    initial_temperature = (
        temperature if start_temperature is None else float(start_temperature)
    )
    if not np.isfinite(initial_temperature) or initial_temperature <= 0.0:
        raise ValueError("start_temperature must be finite and positive")
    trace = TraceReader(calibration_traces)
    if trace.manifest["model_hash"] != inspection.source_hash or trace.manifest["split"] != "calibration":
        raise ValueError("calibration trace/model provenance mismatch")
    calibration_path = Path(calibration_dataset)
    validation_path = Path(validation_dataset)
    if trace.manifest["dataset_hash"] != sha256_file(calibration_path):
        raise ValueError("calibration trace/dataset hash mismatch")
    training_path = (
        Path(training_dataset) if training_dataset is not None else calibration_path
    )
    all_train_records = _load_jsonl(training_path, max_train_records)
    training_shuffle_seed = 9103
    (
        train_records,
        checkpoint_selection_data,
        train_record_indices,
        checkpoint_selection_indices,
    ) = _train_internal_checkpoint_split(
        all_train_records,
        checkpoint_selection_records,
        training_shuffle_seed + 1,
    )
    validation_records = _load_jsonl(validation_path, max_validation_records)
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    teacher = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32).to(device)
    student = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    calibration_sequence_hashes = _sequence_hashes(trace)
    validation_sequence_hashes = _evaluation_sequence_hashes(validation_records, tokenizer)
    training_sequence_hashes = _evaluation_sequence_hashes(
        all_train_records, tokenizer
    )
    fit_sequence_hashes = _evaluation_sequence_hashes(train_records, tokenizer)
    checkpoint_selection_sequence_hashes = _evaluation_sequence_hashes(
        checkpoint_selection_data, tokenizer
    )
    if set(calibration_sequence_hashes).intersection(validation_sequence_hashes):
        raise ValueError("calibration and validation contain matching token sequences")
    if set(training_sequence_hashes).intersection(validation_sequence_hashes):
        raise ValueError("training and validation contain matching token sequences")
    if set(fit_sequence_hashes).intersection(
        checkpoint_selection_sequence_hashes
    ):
        raise ValueError(
            "training fit and checkpoint-selection splits contain matching "
            "token sequences"
        )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(False)

    SparseStudentMLP = (
        _wrap_hardware_sparse_mlp_class(torch)
        if routing_mode == "hardware_ste"
        else _wrap_sparse_mlp_class(torch)
    )
    router_cache_path = target / "router_initialization.safetensors"
    router_cache_manifest_path = target / "router_initialization.json"
    router_cache_configuration = {
        "schema_version": 3,
        "source_model_hash": inspection.source_hash,
        "calibration_dataset_hash": trace.manifest["dataset_hash"],
        "top_k": top_k,
        "router_rank": router_rank,
        "router_regularization": router_regularization,
        "router_group_size": router_group_size,
        "router_grouping": "contiguous_complete_groups_top_summed_utility",
        "router_output_groups": inspection.intermediate_size // router_group_size,
        "selected_layers": list(layer_indices),
    }
    cached_router_tensors = None
    if router_cache_path.is_file() and router_cache_manifest_path.is_file():
        cached_configuration = json.loads(
            router_cache_manifest_path.read_text(encoding="utf-8")
        )
        if cached_configuration == router_cache_configuration:
            cached_router_tensors = load_file(router_cache_path, device="cpu")
    wrappers = []
    fitted_router_tensors = {}
    for layer in layer_indices:
        decoder = student.model.layers[layer]
        teacher_mlp = teacher.model.layers[layer].mlp
        if cached_router_tensors is None:
            gate = (
                teacher_mlp.gate_proj.weight.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            up = (
                teacher_mlp.up_proj.weight.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            down = (
                teacher_mlp.down_proj.weight.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            states = _load_states(trace, layer, None)
            labels = (
                _group_utility_membership(
                    states,
                    gate,
                    up,
                    down,
                    top_k,
                    router_group_size,
                )
                if routing_mode == "hard_router"
                else _membership(states, gate, up, down, top_k)
            ).astype(np.float64)
            router = LowRankMultiLabelRouter.fit(
                states,
                labels,
                rank=router_rank,
                regularization=router_regularization,
            )
            for name, values in (
                ("input", router.input_factors),
                ("output", router.output_factors),
                ("bias", router.bias),
            ):
                fitted_router_tensors[f"layer_{layer}.{name}"] = torch.tensor(
                    values, dtype=torch.float32
                ).contiguous()
        else:
            router = LowRankMultiLabelRouter(
                cached_router_tensors[f"layer_{layer}.input"].numpy(),
                cached_router_tensors[f"layer_{layer}.output"].numpy(),
                cached_router_tensors[f"layer_{layer}.bias"].numpy(),
            )
        wrapper_kwargs = {
            "top_k": initial_top_k,
            "candidates": initial_candidate_count,
            "adapter_rank": adapter_rank,
        }
        if routing_mode == "hard_router":
            wrapper_kwargs["train_full_mlp"] = train_full_mlp
            wrapper_kwargs["router_group_size"] = router_group_size
        if routing_mode == "hardware_ste":
            wrapper_kwargs.update(
                input_fraction=initial_input_fraction,
                residual_rank=residual_rank,
                temperature=initial_temperature,
                cache_line_records=cache_line_records,
            )
        wrapper = SparseStudentMLP(decoder.mlp, router, **wrapper_kwargs).to(device)
        decoder.mlp = wrapper
        wrappers.append(wrapper)
    if cached_router_tensors is None:
        save_file(fitted_router_tensors, router_cache_path)
        atomic_json(router_cache_manifest_path, router_cache_configuration)

    teacher_inputs: dict[int, Any] = {}
    teacher_targets: dict[int, Any] = {}
    handles = []
    for index in layer_indices:
        layer = teacher.model.layers[index]
        handles.append(
            layer.mlp.register_forward_pre_hook(
                lambda module, args, index=index: teacher_inputs.__setitem__(
                    index, args[0].detach()
                )
            )
        )
        handles.append(
            layer.mlp.register_forward_hook(
                lambda module, args, output, index=index: teacher_targets.__setitem__(
                    index, output.detach()
                )
            )
        )
    router_parameters = [
        parameter
        for wrapper in wrappers
        for name, parameter in wrapper.named_parameters()
        if parameter.requires_grad and name.startswith("router_")
    ]
    adapter_parameters = [
        parameter
        for wrapper in wrappers
        for name, parameter in wrapper.named_parameters()
        if parameter.requires_grad and not name.startswith("router_")
    ]
    trainable_parameters = router_parameters + adapter_parameters
    optimizer = torch.optim.AdamW(
        [
            {"params": router_parameters, "lr": router_learning_rate},
            {"params": adapter_parameters, "lr": learning_rate},
        ]
    )
    history = []
    gradient_report = None
    trainable_named_parameters = {
        f"layer_{layer}.{name}": parameter
        for layer, wrapper in zip(layer_indices, wrappers, strict=True)
        for name, parameter in wrapper.named_parameters()
        if parameter.requires_grad
    }
    checkpoint_path = target / "training_checkpoint.pt"
    checkpoint_manifest_path = target / "training_checkpoint.json"
    local_target_policy = (
        "frozen_teacher_mlp_on_stop_gradient_student_mlp_input"
        if routing_mode == "hardware_ste"
        else "teacher_sequence_mlp_output"
    )
    checkpoint_configuration = {
        "schema_version": 4,
        "source_model_hash": inspection.source_hash,
        "training_dataset_hash": sha256_file(training_path),
        "validation_dataset_hash": sha256_file(validation_path),
        "records": len(all_train_records),
        "fit_records": len(train_records),
        "checkpoint_selection_records": len(checkpoint_selection_data),
        "checkpoint_selection_every": checkpoint_selection_every,
        "selected_layers": list(layer_indices),
        "epochs": epochs,
        "batch_size": batch_size,
        "top_k": top_k,
        "candidate_count": candidate_count,
        "router_rank": router_rank,
        "router_group_size": router_group_size,
        "adapter_rank": adapter_rank,
        "residual_rank": residual_rank,
        "learning_rate": learning_rate,
        "router_learning_rate": router_learning_rate,
        "input_fraction": input_fraction,
        "temperature": temperature,
        "start_top_k": initial_top_k,
        "start_candidate_count": initial_candidate_count,
        "start_input_fraction": initial_input_fraction,
        "exact_dense_start": exact_dense_start,
        "dense_warmup_steps": dense_warmup_steps,
        "anneal_steps": anneal_steps,
        "start_temperature": initial_temperature,
        "train_full_mlp": train_full_mlp,
        "training_order": "deterministic_epoch_shuffle",
        "training_shuffle_seed": training_shuffle_seed,
        "local_target_policy": local_target_policy,
        "loss_weights": {
            "local": local_weight,
            "hidden": hidden_weight,
            "logit": logit_weight,
            "teacher_forced_local": teacher_forced_local_weight,
            "router": router_weight,
            "locality": locality_weight,
        },
    }
    completed_steps = 0
    best_checkpoint_path = target / "best_training_checkpoint.pt"
    best_checkpoint_step: int | None = None
    best_checkpoint_metrics: dict[str, float] | None = None
    last_checkpoint_selection_step: int | None = None
    if resume:
        if not checkpoint_path.is_file() or not checkpoint_manifest_path.is_file():
            raise ValueError("resume requested but training checkpoint is missing")
        checkpoint_manifest = json.loads(
            checkpoint_manifest_path.read_text(encoding="utf-8")
        )
        if checkpoint_manifest.get("configuration") != checkpoint_configuration:
            raise ValueError("training checkpoint configuration mismatch")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        checkpoint_tensors = checkpoint.get("trainable_parameters", {})
        if set(checkpoint_tensors) != set(trainable_named_parameters):
            raise ValueError("training checkpoint parameter set mismatch")
        with torch.no_grad():
            for name, parameter in trainable_named_parameters.items():
                parameter.copy_(checkpoint_tensors[name].to(device=device))
        optimizer.load_state_dict(checkpoint["optimizer"])
        completed_steps = int(checkpoint_manifest.get("completed_steps", 0))
        history = list(checkpoint_manifest.get("history", []))
        gradient_report = checkpoint_manifest.get("gradient_diagnostics")
        checkpoint_selection_state = checkpoint_manifest.get(
            "checkpoint_selection"
        )
        if checkpoint_selection_state is not None:
            best_checkpoint_step = checkpoint_selection_state.get("best_step")
            best_checkpoint_metrics = checkpoint_selection_state.get(
                "best_metrics"
            )
            last_checkpoint_selection_step = checkpoint_selection_state.get(
                "last_evaluated_step"
            )
            if (
                best_checkpoint_step is not None
                and not best_checkpoint_path.is_file()
            ):
                raise ValueError(
                    "resume checkpoint references a missing best checkpoint"
                )

    def save_training_checkpoint(steps: int) -> None:
        temporary = target / "training_checkpoint.pt.tmp"
        torch.save(
            {
                "trainable_parameters": {
                    name: parameter.detach().cpu()
                    for name, parameter in trainable_named_parameters.items()
                },
                "optimizer": optimizer.state_dict(),
            },
            temporary,
        )
        temporary.replace(checkpoint_path)
        atomic_json(
            checkpoint_manifest_path,
            {
                "configuration": checkpoint_configuration,
                "completed_steps": steps,
                "history": history,
                "gradient_diagnostics": gradient_report,
                "checkpoint_selection": {
                    "best_step": best_checkpoint_step,
                    "best_metrics": best_checkpoint_metrics,
                    "last_evaluated_step": last_checkpoint_selection_step,
                },
            },
        )

    def evaluate_checkpoint_selection() -> dict[str, float]:
        if not checkpoint_selection_data:
            raise RuntimeError("checkpoint selection data is empty")
        wrapper_state = [
            {
                "top_k": wrapper.top_k,
                "candidates": wrapper.candidates,
                "temperature": wrapper.temperature,
                **(
                    {"input_fraction": wrapper.input_fraction}
                    if routing_mode == "hardware_ste"
                    else {}
                ),
            }
            for wrapper in wrappers
        ]
        for wrapper in wrappers:
            wrapper.top_k = top_k
            wrapper.candidates = candidate_count
            wrapper.temperature = temperature
            if routing_mode == "hardware_ste":
                wrapper.input_fraction = input_fraction
        weighted = {"local": 0.0, "hidden": 0.0, "logit": 0.0}
        token_weight = 0
        try:
            with torch.inference_mode():
                for selection_batch in _batches(
                    checkpoint_selection_data, batch_size
                ):
                    input_ids, attention_mask, lengths = _batch_ids(
                        selection_batch, tokenizer, torch, device
                    )
                    if max(lengths) < 2:
                        continue
                    valid_mask = attention_mask.bool()
                    batch_weight = int(valid_mask.sum())
                    if batch_weight == 0:
                        continue
                    teacher_inputs.clear()
                    teacher_targets.clear()
                    teacher_output = teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    student_output = student(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    selection_local_targets = (
                        _same_input_teacher_mlp_targets(
                            [
                                teacher.model.layers[layer].mlp
                                for layer in layer_indices
                            ],
                            [wrapper.last_input for wrapper in wrappers],
                            torch,
                        )
                        if routing_mode == "hardware_ste"
                        else [
                            teacher_targets[layer] for layer in layer_indices
                        ]
                    )
                    selection_local = torch.stack(
                        [
                            _normalized_masked_mse(
                                wrapper.last_output,
                                local_target,
                                valid_mask,
                                torch,
                            )
                            for wrapper, local_target in zip(
                                wrappers,
                                selection_local_targets,
                                strict=True,
                            )
                        ]
                    ).mean()
                    selection_hidden = torch.stack(
                        [
                            _normalized_masked_mse(
                                student_hidden,
                                teacher_hidden,
                                valid_mask,
                                torch,
                            )
                            for student_hidden, teacher_hidden in zip(
                                student_output.hidden_states[1:],
                                teacher_output.hidden_states[1:],
                                strict=True,
                            )
                        ]
                    ).mean()
                    teacher_logp = functional.log_softmax(
                        teacher_output.logits, dim=-1
                    )
                    student_logp = functional.log_softmax(
                        student_output.logits, dim=-1
                    )
                    selection_logit_rows = functional.kl_div(
                        student_logp,
                        teacher_logp.exp(),
                        reduction="none",
                    ).sum(dim=-1)
                    selection_logit = _masked_mean(
                        selection_logit_rows, valid_mask, torch
                    )
                    for name, value in (
                        ("local", selection_local),
                        ("hidden", selection_hidden),
                        ("logit", selection_logit),
                    ):
                        weighted[name] += float(value) * batch_weight
                    token_weight += batch_weight
        finally:
            for wrapper, state in zip(wrappers, wrapper_state, strict=True):
                wrapper.top_k = int(state["top_k"])
                wrapper.candidates = int(state["candidates"])
                wrapper.temperature = float(state["temperature"])
                if routing_mode == "hardware_ste":
                    wrapper.input_fraction = float(state["input_fraction"])
        if token_weight == 0:
            raise ValueError(
                "checkpoint selection split has no usable token positions"
            )
        means = {
            name: value / token_weight for name, value in weighted.items()
        }
        means["total"] = (
            local_weight * means["local"]
            + hidden_weight * means["hidden"]
            + logit_weight * means["logit"]
        )
        means["token_positions"] = float(token_weight)
        return means

    def select_checkpoint(steps: int) -> tuple[dict[str, float], bool]:
        nonlocal best_checkpoint_metrics
        nonlocal best_checkpoint_step
        nonlocal last_checkpoint_selection_step
        selection_metrics = evaluate_checkpoint_selection()
        last_checkpoint_selection_step = steps
        improved = (
            best_checkpoint_metrics is None
            or selection_metrics["total"] < best_checkpoint_metrics["total"]
        )
        if improved:
            best_checkpoint_metrics = selection_metrics
            best_checkpoint_step = steps
            temporary = target / "best_training_checkpoint.pt.tmp"
            torch.save(
                {
                    "trainable_parameters": {
                        name: parameter.detach().cpu()
                        for name, parameter in trainable_named_parameters.items()
                    }
                },
                temporary,
            )
            temporary.replace(best_checkpoint_path)
        return selection_metrics, improved

    # Keep frozen-model dropout disabled while retaining autograd for routers/adapters.
    student.eval()
    try:
        global_step = 0
        for epoch in range(epochs):
            epoch_order = np.random.default_rng(
                training_shuffle_seed + epoch
            ).permutation(len(train_records))
            epoch_records = [train_records[int(index)] for index in epoch_order]
            for batch_index, batch_records in enumerate(
                _batches(epoch_records, batch_size)
            ):
                input_ids, attention_mask, lengths = _batch_ids(
                    batch_records, tokenizer, torch, device
                )
                if max(lengths) < 2:
                    continue
                if global_step < completed_steps:
                    global_step += 1
                    continue
                if routing_mode == "hard_router":
                    progress = _curriculum_progress(
                        global_step, 0, anneal_steps
                    )
                    current_top_k = _aligned_router_width(
                        initial_top_k,
                        top_k,
                        progress,
                        router_group_size,
                    )
                    current_temperature = initial_temperature + progress * (
                        temperature - initial_temperature
                    )
                    for wrapper in wrappers:
                        wrapper.top_k = current_top_k
                        wrapper.candidates = current_top_k
                        wrapper.temperature = current_temperature
                else:
                    hardware_schedule = _progressive_hardware_schedule(
                        step=global_step,
                        dense_warmup_steps=dense_warmup_steps,
                        anneal_steps=anneal_steps,
                        start_input_fraction=initial_input_fraction,
                        target_input_fraction=input_fraction,
                        start_candidate_count=initial_candidate_count,
                        target_candidate_count=candidate_count,
                        start_top_k=initial_top_k,
                        target_top_k=top_k,
                        start_temperature=initial_temperature,
                        target_temperature=temperature,
                    )
                    for wrapper in wrappers:
                        wrapper.input_fraction = float(
                            hardware_schedule["input_fraction"]
                        )
                        wrapper.candidates = int(
                            hardware_schedule["candidate_count"]
                        )
                        wrapper.top_k = int(hardware_schedule["top_k"])
                        wrapper.temperature = float(
                            hardware_schedule["temperature"]
                        )
                valid_mask = attention_mask.bool()
                teacher_inputs.clear()
                teacher_targets.clear()
                with torch.no_grad():
                    teacher_output = teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                optimizer.zero_grad(set_to_none=True)
                teacher_forced_local_loss = torch.zeros(
                    (), device=input_ids.device
                )
                if teacher_forced_local_weight:
                    for layer, wrapper in zip(
                        layer_indices, wrappers, strict=True
                    ):
                        forced_output = wrapper(teacher_inputs[layer])
                        layer_forced_loss = _normalized_masked_mse(
                            forced_output,
                            teacher_targets[layer],
                            valid_mask,
                            torch,
                        )
                        (
                            teacher_forced_local_weight
                            * layer_forced_loss
                            / len(wrappers)
                        ).backward()
                        teacher_forced_local_loss = (
                            teacher_forced_local_loss
                            + layer_forced_loss.detach() / len(wrappers)
                        )
                student_output = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                local_targets = (
                    _same_input_teacher_mlp_targets(
                        [
                            teacher.model.layers[layer].mlp
                            for layer in layer_indices
                        ],
                        [wrapper.last_input for wrapper in wrappers],
                        torch,
                    )
                    if routing_mode == "hardware_ste"
                    else [
                        teacher_targets[layer]
                        for layer in layer_indices
                    ]
                )
                local_loss = torch.stack(
                    [
                        _normalized_masked_mse(
                            wrapper.last_output,
                            local_target,
                            valid_mask,
                            torch,
                        )
                        for wrapper, local_target in zip(
                            wrappers, local_targets, strict=True
                        )
                    ]
                ).mean()
                hidden_loss = torch.stack(
                    [
                        _normalized_masked_mse(
                            student_hidden, teacher_hidden, valid_mask, torch
                        )
                        for student_hidden, teacher_hidden in zip(
                            student_output.hidden_states[1:], teacher_output.hidden_states[1:], strict=True
                        )
                    ]
                ).mean()
                teacher_logp = functional.log_softmax(teacher_output.logits.detach(), dim=-1)
                student_logp = functional.log_softmax(student_output.logits, dim=-1)
                logit_rows = functional.kl_div(
                    student_logp,
                    teacher_logp.exp(),
                    reduction="none",
                ).sum(dim=-1)
                logit_loss = _masked_mean(logit_rows, valid_mask, torch)
                router_loss = torch.stack(
                    [
                        _masked_mean(
                            functional.binary_cross_entropy_with_logits(
                                wrapper.last_router_logits,
                                (
                                    wrapper.last_router_target
                                    if routing_mode == "hard_router"
                                    else torch.zeros_like(
                                        wrapper.last_router_logits
                                    ).scatter(1, wrapper.last_oracle, 1.0)
                                ),
                                reduction="none",
                            ).mean(dim=1),
                            valid_mask.reshape(-1),
                            torch,
                        )
                        for wrapper in wrappers
                    ]
                ).mean()
                locality_loss = (
                    torch.stack(
                        [
                            _masked_mean(
                                wrapper.last_locality_rows,
                                valid_mask.reshape(-1),
                                torch,
                            )
                            for wrapper in wrappers
                        ]
                    ).mean()
                    if routing_mode == "hardware_ste"
                    else torch.zeros((), device=input_ids.device)
                )
                loss = (
                    local_weight * local_loss
                    + hidden_weight * hidden_loss
                    + logit_weight * logit_loss
                    + router_weight * router_loss
                    + locality_weight * locality_loss
                )
                if gradient_diagnostics and gradient_report is None:
                    causal_objective = (
                        local_weight * local_loss
                        + hidden_weight * hidden_loss
                        + logit_weight * logit_loss
                        + router_weight * router_loss
                    )
                    causal_gradients = torch.autograd.grad(
                        causal_objective,
                        router_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    locality_gradients = torch.autograd.grad(
                        locality_loss,
                        router_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    causal_squared = torch.zeros((), device=input_ids.device)
                    locality_squared = torch.zeros((), device=input_ids.device)
                    gradient_dot = torch.zeros((), device=input_ids.device)
                    for causal_gradient, locality_gradient in zip(
                        causal_gradients, locality_gradients, strict=True
                    ):
                        if causal_gradient is not None:
                            causal_squared = causal_squared + torch.sum(
                                causal_gradient.float() ** 2
                            )
                        if locality_gradient is not None:
                            locality_squared = locality_squared + torch.sum(
                                locality_gradient.float() ** 2
                            )
                        if (
                            causal_gradient is not None
                            and locality_gradient is not None
                        ):
                            gradient_dot = gradient_dot + torch.sum(
                                causal_gradient.float()
                                * locality_gradient.float()
                            )
                    causal_norm = torch.sqrt(causal_squared)
                    locality_norm = torch.sqrt(locality_squared)
                    gradient_report = {
                        "scope": "first_training_batch_router_parameters",
                        "causal_objective_l2": float(causal_norm.detach()),
                        "locality_objective_l2": float(locality_norm.detach()),
                        "cosine": float(
                            (
                                gradient_dot
                                / torch.clamp(
                                    causal_norm * locality_norm, min=1e-20
                                )
                            ).detach()
                        ),
                        "equal_norm_locality_weight": float(
                            (
                                causal_norm
                                / torch.clamp(locality_norm, min=1e-20)
                            ).detach()
                        ),
                    }
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()
                history.append(
                    {
                        "epoch": epoch,
                        "batch": batch_index,
                        "record_start": batch_index * batch_size,
                        "records": len(batch_records),
                        "active_records": wrappers[0].top_k,
                        "candidate_records": wrappers[0].candidates,
                        "input_fraction": (
                            wrappers[0].input_fraction
                            if routing_mode == "hardware_ste"
                            else 1.0
                        ),
                        "temperature": wrappers[0].temperature,
                        "curriculum_progress": (
                            float(hardware_schedule["progress"])
                            if routing_mode == "hardware_ste"
                            else progress
                        ),
                        "total": float(loss.detach()),
                        "total_with_teacher_forced": float(
                            loss.detach()
                            + teacher_forced_local_weight
                            * teacher_forced_local_loss
                        ),
                        "local": float(local_loss.detach()),
                        "teacher_forced_local": float(
                            teacher_forced_local_loss.detach()
                        ),
                        "hidden": float(hidden_loss.detach()),
                        "logit": float(logit_loss.detach()),
                        "router": float(router_loss.detach()),
                        "locality": float(locality_loss.detach()),
                    }
                )
                global_step += 1
                endpoint_reached = (
                    bool(hardware_schedule["deployment_endpoint"])
                    if routing_mode == "hardware_ste"
                    else progress >= 1.0
                )
                if (
                    checkpoint_selection_data
                    and endpoint_reached
                    and global_step % checkpoint_selection_every == 0
                ):
                    selection_metrics, improved = select_checkpoint(
                        global_step
                    )
                    history[-1]["checkpoint_selection"] = {
                        **selection_metrics,
                        "new_best": improved,
                    }
                if checkpoint_every and global_step % checkpoint_every == 0:
                    save_training_checkpoint(global_step)
        if (
            checkpoint_selection_data
            and last_checkpoint_selection_step != global_step
        ):
            selection_metrics, improved = select_checkpoint(global_step)
            history[-1]["checkpoint_selection"] = {
                **selection_metrics,
                "new_best": improved,
            }
        if checkpoint_every and global_step != completed_steps:
            save_training_checkpoint(global_step)
    finally:
        for handle in handles:
            handle.remove()

    if checkpoint_selection_data:
        if best_checkpoint_step is None or not best_checkpoint_path.is_file():
            raise RuntimeError("train-internal checkpoint selection produced no checkpoint")
        best_checkpoint = torch.load(
            best_checkpoint_path, map_location=device, weights_only=True
        )
        best_tensors = best_checkpoint.get("trainable_parameters", {})
        if set(best_tensors) != set(trainable_named_parameters):
            raise ValueError("best checkpoint parameter set mismatch")
        with torch.no_grad():
            for name, parameter in trainable_named_parameters.items():
                parameter.copy_(best_tensors[name].to(device=device))

    tensors = {}
    for wrapper in wrappers:
        wrapper.top_k = top_k
        wrapper.candidates = candidate_count
        wrapper.temperature = temperature
        if routing_mode == "hardware_ste":
            wrapper.input_fraction = input_fraction
    tensor_names = [
        "router_input",
        "router_output",
        "router_bias",
        "adapter_a",
        "adapter_b",
        "gate_adapter_a",
        "gate_adapter_b",
        "up_adapter_a",
        "up_adapter_b",
    ]
    if routing_mode == "hard_router":
        tensor_names.append("router_nonlinear_scale")
    if routing_mode == "hardware_ste":
        tensor_names.extend(
            (
                "router_blend_logit",
            )
        )
        if residual_rank:
            tensor_names.extend(("residual_a", "residual_b"))
    for layer, wrapper in zip(layer_indices, wrappers, strict=True):
        for name in tensor_names:
            tensors[f"layer_{layer}.{name}"] = getattr(wrapper, name).detach().cpu().contiguous()
        if train_full_mlp:
            merged_gate = wrapper.base.gate_proj.weight + (
                wrapper.gate_adapter_b.T @ wrapper.gate_adapter_a.T
            ) * wrapper.adapter_scale
            merged_up = wrapper.base.up_proj.weight + (
                wrapper.up_adapter_b.T @ wrapper.up_adapter_a.T
            ) * wrapper.adapter_scale
            merged_down = wrapper.base.down_proj.weight + (
                wrapper.adapter_b.T @ wrapper.adapter_a.T
            ) * wrapper.adapter_scale
            tensors[f"layer_{layer}.deployment_gate_weight"] = (
                merged_gate.detach().cpu().contiguous()
            )
            tensors[f"layer_{layer}.deployment_up_weight"] = (
                merged_up.detach().cpu().contiguous()
            )
            tensors[f"layer_{layer}.deployment_down_weight"] = (
                merged_down.detach().cpu().contiguous()
            )
            for projection_name, projection in (
                ("gate", wrapper.base.gate_proj),
                ("up", wrapper.base.up_proj),
                ("down", wrapper.base.down_proj),
            ):
                if projection.bias is not None:
                    tensors[f"layer_{layer}.deployment_{projection_name}_bias"] = (
                        projection.bias.detach().cpu().contiguous()
                    )
    tensor_path = target / "sparse_student.safetensors"
    save_file(tensors, tensor_path)

    student.eval()
    quality = defaultdict(list)
    local_error = []
    recalls = []
    occupied_lines = []
    residual_relative_norm = []
    residual_target_cosine = []
    residual_error_reduction = []
    teacher_nll = []
    input_positions = 0
    next_positions = 0
    validation_handles = [
        teacher.model.layers[index].mlp.register_forward_hook(
            lambda module, args, output, index=index: teacher_targets.__setitem__(
                index, output.detach()
            )
        )
        for index in layer_indices
    ]
    try:
        for batch_records in _batches(validation_records, batch_size):
            input_ids, attention_mask, lengths = _batch_ids(
                batch_records, tokenizer, torch, device
            )
            teacher_targets.clear()
            with torch.inference_mode():
                teacher_output = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                student_output = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                validation_local_targets = (
                    _same_input_teacher_mlp_targets(
                        [
                            teacher.model.layers[layer].mlp
                            for layer in layer_indices
                        ],
                        [wrapper.last_input for wrapper in wrappers],
                        torch,
                    )
                    if routing_mode == "hardware_ste"
                    else [
                        teacher_targets[layer]
                        for layer in layer_indices
                    ]
                )
            for row, length in enumerate(lengths):
                if length < 2:
                    continue
                input_positions += length
                next_positions += length - 1
                metrics = _quality_metrics(
                    teacher_output.logits[row : row + 1, :length],
                    student_output.logits[row : row + 1, :length],
                    input_ids[row : row + 1, :length],
                    teacher_output.hidden_states[-1][row : row + 1, :length],
                    student_output.hidden_states[-1][row : row + 1, :length],
                    torch,
                )
                for name, values in metrics.items():
                    quality[name].extend(np.asarray(values).reshape(-1).tolist())
                teacher_nll.extend(metrics["teacher_nll"].tolist())
                for wrapper, local_target in zip(
                    wrappers, validation_local_targets, strict=True
                ):
                    relative, _ = _relative_and_cosine_rows(
                        wrapper.last_output[row : row + 1, :length]
                        .detach()
                        .cpu()
                        .numpy(),
                        local_target[row : row + 1, :length]
                        .detach()
                        .cpu()
                        .numpy(),
                    )
                    local_error.extend(relative.tolist())
                    if routing_mode == "hardware_ste" and residual_rank:
                        target_rows = (
                            local_target[row : row + 1, :length]
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(-1, inspection.hidden_size)
                        )
                        before_rows = (
                            wrapper.last_pre_residual[row : row + 1, :length]
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(-1, inspection.hidden_size)
                        )
                        correction_rows = (
                            wrapper.last_residual[row : row + 1, :length]
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(-1, inspection.hidden_size)
                        )
                        missing_rows = target_rows - before_rows
                        target_norm = np.maximum(
                            np.linalg.norm(target_rows, axis=1), 1e-12
                        )
                        correction_norm = np.linalg.norm(
                            correction_rows, axis=1
                        )
                        missing_norm = np.maximum(
                            np.linalg.norm(missing_rows, axis=1), 1e-12
                        )
                        residual_relative_norm.extend(
                            (correction_norm / target_norm).tolist()
                        )
                        residual_target_cosine.extend(
                            (
                                np.sum(correction_rows * missing_rows, axis=1)
                                / np.maximum(
                                    correction_norm * missing_norm, 1e-12
                                )
                            ).tolist()
                        )
                        residual_error_reduction.extend(
                            (
                                (
                                    missing_norm
                                    - np.linalg.norm(
                                        missing_rows - correction_rows, axis=1
                                    )
                                )
                                / target_norm
                            ).tolist()
                        )
                    recall_rows = wrapper.last_recall.reshape(
                        len(lengths), input_ids.shape[1]
                    )
                    recalls.extend(
                        recall_rows[row, :length].detach().cpu().numpy().tolist()
                    )
                    if routing_mode == "hardware_ste":
                        line_rows = wrapper.last_occupied_lines.reshape(
                            len(lengths), input_ids.shape[1]
                        )
                        occupied_lines.extend(
                            line_rows[row, :length]
                            .detach()
                            .cpu()
                            .numpy()
                            .tolist()
                        )
    finally:
        for handle in validation_handles:
            handle.remove()

    metrics_mean = {name: _mean(values) for name, values in quality.items()}
    traffic = projected_dip_traffic(
        inspection.hidden_size,
        inspection.intermediate_size,
        input_fraction=input_fraction if routing_mode == "hardware_ste" else 1.0,
        candidate_count=candidate_count,
        top_k=top_k,
    )
    cache_line_bytes = cache_line_records * 4
    cache_adjusted_bytes = (
        2 * inspection.intermediate_size * traffic.input_count * 4
        + 2
        * (inspection.hidden_size - traffic.input_count)
        * _mean(occupied_lines)
        * cache_line_bytes
        + top_k * inspection.hidden_size * 4
        if routing_mode == "hardware_ste"
        else traffic.total_bytes
    )
    residual_adapter_bytes = (
        2 * inspection.hidden_size * residual_rank * 4
        if routing_mode == "hardware_ste"
        else 0
    )
    cache_adjusted_bytes += residual_adapter_bytes
    cache_adjusted_fraction = cache_adjusted_bytes / traffic.dense_bytes
    direct_traffic = (
        _direct_router_traffic(
            inspection.hidden_size,
            inspection.intermediate_size,
            candidate_count=candidate_count,
            top_k=top_k,
            router_rank=router_rank,
            router_group_size=router_group_size,
        )
        if routing_mode == "hard_router"
        else None
    )
    checks = {
        "teacher_student_kl": metrics_mean["teacher_student_kl"] <= MLP_QUALITY_THRESHOLDS["maximum_teacher_student_kl"],
        "teacher_top1_agreement": metrics_mean["teacher_top1_agreement"] >= MLP_QUALITY_THRESHOLDS["minimum_teacher_top1_agreement"],
        "nll_delta": metrics_mean["nll_delta"] <= MLP_QUALITY_THRESHOLDS["maximum_nll_delta"],
        "final_hidden_relative_l2": metrics_mean["final_hidden_relative_l2"] <= MLP_QUALITY_THRESHOLDS["maximum_final_hidden_relative_l2"],
        "evidence_size": (
            len(validation_records) >= MINIMUM_EVALUATION_SEQUENCES
            and len(set(validation_sequence_hashes)) >= MINIMUM_UNIQUE_EVALUATION_SEQUENCES
            and next_positions >= MINIMUM_NEXT_TOKEN_POSITIONS
        ),
        **(
            {
                "hardware_budget": (
                    input_fraction <= 0.625
                    and candidate_count <= 512
                    and top_k <= 512
                ),
                "cache_adjusted_traffic": cache_adjusted_fraction < 1.0,
            }
            if routing_mode == "hardware_ste"
            else {
                "projected_traffic": direct_traffic["fraction_of_dense_q4"]
                <= 0.45,
            }
        ),
    }
    if routing_mode == "hardware_ste":
        checks["candidate_recall"] = (
            _mean(recalls) >= MINIMUM_ROUTED_CANDIDATE_RECALL
        )
    report = {
        "schema_version": 2,
        "experiment": (
            "hardware_aware_sparse_teacher_finetuning"
            if routing_mode == "hardware_ste"
            else "sparse_teacher_finetuning"
        ),
        "status": (
            "measured_all_layer_model"
            if len(layer_indices) == inspection.num_hidden_layers
            else "measured_partial_layer_causal_pilot"
        ),
        "source_model_hash": inspection.source_hash,
        "configuration": {
            "selected_layers": list(layer_indices),
            "frozen_dense_layers": [
                layer
                for layer in range(inspection.num_hidden_layers)
                if layer not in layer_indices
            ],
            "top_k": top_k,
            "candidate_count": candidate_count,
            "router_rank": router_rank,
            "router_group_size": router_group_size,
            "router_regularization": router_regularization,
            "adapter_rank": adapter_rank,
            "residual_rank": residual_rank,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "router_learning_rate": router_learning_rate,
            "routing_mode": routing_mode,
            "input_fraction": input_fraction,
            "temperature": temperature,
            "start_top_k": initial_top_k,
            "start_candidate_count": initial_candidate_count,
            "start_input_fraction": initial_input_fraction,
            "exact_dense_start": exact_dense_start,
            "dense_warmup_steps": dense_warmup_steps,
            "anneal_steps": anneal_steps,
            "start_temperature": initial_temperature,
            "train_full_mlp": train_full_mlp,
            "training_order": "deterministic_epoch_shuffle",
            "training_shuffle_seed": training_shuffle_seed,
            "local_target_policy": local_target_policy,
            "cache_line_records": cache_line_records,
            "batch_size": batch_size,
            "gradient_diagnostics": gradient_diagnostics,
            "checkpoint_every": checkpoint_every,
            "checkpoint_selection_records": checkpoint_selection_records,
            "checkpoint_selection_every": checkpoint_selection_every,
            "resumed": resume,
            "loss_weights": {
                "local": local_weight,
                "hidden": hidden_weight,
                "logit": logit_weight,
                "teacher_forced_local": teacher_forced_local_weight,
                "router": router_weight,
                "locality": locality_weight,
            },
            "adapter_scaling": "lora_alpha_equals_rank",
        },
        "training": {
            "source_records": len(all_train_records),
            "records": len(train_records),
            "checkpoint_selection_records": len(
                checkpoint_selection_data
            ),
            "steps": len(history),
            "batch_size": batch_size,
            "router_initialization_cache_reused": cached_router_tensors is not None,
            "gradient_diagnostics": gradient_report,
            "resumed_from_steps": completed_steps,
            "checkpoint_selection": {
                "policy": (
                    "minimum deployment-endpoint local_hidden_logit "
                    "distillation objective on deterministic train-only split"
                    if checkpoint_selection_data
                    else "disabled"
                ),
                "best_step": best_checkpoint_step,
                "best_metrics": best_checkpoint_metrics,
                "last_evaluated_step": last_checkpoint_selection_step,
                "development_data_used": False,
                "restored_before_development_evaluation": bool(
                    checkpoint_selection_data
                ),
            },
            "history": history,
        },
        "validation": {
            "role": "development",
            "evaluations": 1,
            "records": len(validation_records),
            "input_token_positions": input_positions,
            "next_token_positions": next_positions,
        },
        "data_separation": {
            "method": "exact_token_sequence_hashes",
            "calibration_sequences": len(calibration_sequence_hashes),
            "validation_sequences": len(validation_sequence_hashes),
            "overlapping_sequences": 0,
            "held_out": True,
            "training_dataset_hash": sha256_file(training_path),
            "training_sequences": len(training_sequence_hashes),
            "fit_sequences": len(fit_sequence_hashes),
            "checkpoint_selection_sequences": len(
                checkpoint_selection_sequence_hashes
            ),
            "fit_checkpoint_selection_overlapping_sequences": 0,
            "fit_record_indices_count": len(train_record_indices),
            "checkpoint_selection_record_indices_count": len(
                checkpoint_selection_indices
            ),
            "training_validation_overlapping_sequences": 0,
        },
        "metrics": {
            **{name: _stats(values) for name, values in quality.items()},
            "local_mlp_relative_l2": _stats(local_error),
            "candidate_recall": _stats(recalls),
            **(
                {
                    "candidate_cache_lines": _stats(occupied_lines),
                    "candidate_cache_line_fraction": _stats(
                        [
                            value
                            / int(
                                np.ceil(
                                    inspection.intermediate_size
                                    / cache_line_records
                                )
                            )
                            for value in occupied_lines
                        ]
                    ),
                    "coordinate_major_candidate_cache_lines": _stats(
                        occupied_lines
                    ),
                    "coordinate_major_candidate_cache_line_fraction": _stats(
                        [
                            value
                            / int(
                                np.ceil(
                                    inspection.intermediate_size
                                    / cache_line_records
                                )
                            )
                            for value in occupied_lines
                        ]
                    ),
                    **(
                        {
                            "residual_relative_norm": _stats(
                                residual_relative_norm
                            ),
                            "residual_target_cosine": _stats(
                                residual_target_cosine
                            ),
                            "residual_error_reduction": _stats(
                                residual_error_reduction
                            ),
                        }
                        if residual_rank
                        else {}
                    ),
                }
                if routing_mode == "hardware_ste"
                else {}
            ),
        },
        "projected_traffic": (
            {
                "completion_layout_accounting": "coordinate_major_candidate_gather",
                "input_coordinates": traffic.input_count,
                "weight_bytes_per_token_layer": traffic.total_bytes,
                "dense_weight_bytes_per_token_layer": traffic.dense_bytes,
                "fraction_of_dense": traffic.fraction_of_dense,
                "dense_over_sparse_reduction": traffic.reduction_factor,
                "cache_adjusted_weight_bytes_per_token_layer": cache_adjusted_bytes,
                "cache_adjusted_fraction_of_dense": cache_adjusted_fraction,
                "coordinate_major_cache_adjusted_fraction_of_dense": cache_adjusted_fraction,
                "residual_adapter_weight_bytes_per_token_layer": residual_adapter_bytes,
            }
            if routing_mode == "hardware_ste"
            else {
                "completion_layout_accounting": "cache_line_aligned_complete_posting_groups",
                "precision_baseline": "dense_q4_mlp",
                **direct_traffic,
                "direct_router_total_bytes_per_token_layer": direct_traffic[
                    "total_bytes"
                ],
                "direct_router_fraction_of_dense_q4": direct_traffic[
                    "fraction_of_dense_q4"
                ],
                "traffic_gate_maximum_fraction": 0.45,
                "mergeable_adapter_extra_inference_bytes": 0,
            }
        ),
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "scope": (
                "all_layers"
                if len(layer_indices) == inspection.num_hidden_layers
                else "selected_layers_causal_pilot"
            ),
            "decision": (
                (
                    "eligible_for_intervention_artifact"
                    if len(layer_indices) == inspection.num_hidden_layers
                    else "eligible_for_next_progressive_layer_stage"
                )
                if all(checks.values())
                else "stop_before_serialization"
            ),
        },
        "artifact": {
            "path": str(tensor_path.resolve()),
            "sha256": sha256_file(tensor_path),
            "format": (
                (
                    "safetensors_router_mergeable_gate_up_down_lora_and_hidden_residual"
                    if residual_rank
                    else "safetensors_router_and_mergeable_gate_up_down_lora"
                )
                if routing_mode == "hardware_ste"
                else (
                    "safetensors_router_full_mlp_and_mergeable_gate_up_down_lora"
                    if train_full_mlp
                    else "safetensors_router_and_mergeable_gate_up_down_lora"
                )
            ),
        },
    }
    atomic_json(target / "sparse_teacher_training.json", report)
    metric = report["metrics"]
    lines = [
        "# Sparse-teacher fine-tuning pilot",
        "",
        f"Status: **{report['gate']['decision']}**",
        "",
        f"Selected sparse layers: {', '.join(str(layer) for layer in layer_indices)}.",
        "",
        f"Training fit/checkpoint-selection records: {len(train_records)}/"
        f"{len(checkpoint_selection_data)}; steps: {len(history)}; "
        f"validation records: {len(validation_records)}.",
        "",
        "| Metric | Mean | Threshold | Pass |",
        "|---|---:|---:|---|",
        f"| Teacher-oracle membership recall | {metric['candidate_recall']['mean']:.6f} | "
        + (
            f"≥0.95 | {'yes' if checks['candidate_recall'] else 'no'} |"
            if routing_mode == "hardware_ste"
            else "informational after co-adaptation | — |"
        ),
        f"| Teacher-student KL | {metric['teacher_student_kl']['mean']:.6f} | ≤0.05 | "
        f"{'yes' if checks['teacher_student_kl'] else 'no'} |",
        f"| Teacher top-1 agreement | {metric['teacher_top1_agreement']['mean']:.6f} | ≥0.90 | "
        f"{'yes' if checks['teacher_top1_agreement'] else 'no'} |",
        f"| NLL delta | {metric['nll_delta']['mean']:.6f} | ≤0.05 | "
        f"{'yes' if checks['nll_delta'] else 'no'} |",
        f"| Final hidden relative L2 | {metric['final_hidden_relative_l2']['mean']:.6f} | ≤0.10 | "
        f"{'yes' if checks['final_hidden_relative_l2'] else 'no'} |",
        *(
            [
                "| Candidate cache-line fraction | "
                f"{metric['candidate_cache_line_fraction']['mean']:.6f} | minimize | informational |",
                f"| Scalar projected traffic | {traffic.fraction_of_dense:.6f}× dense | informational | — |",
                f"| Cache-adjusted traffic | {cache_adjusted_fraction:.6f}× dense | <1.0× | "
                f"{'yes' if checks['cache_adjusted_traffic'] else 'no'} |",
            ]
            if routing_mode == "hardware_ste"
            else [
                "| Projected cold MLP traffic | "
                f"{direct_traffic['fraction_of_dense_q4']:.6f}× dense Q4 | ≤0.45× | "
                f"{'yes' if checks['projected_traffic'] else 'no'} |"
            ]
        ),
        "",
        "The artifact contains router factors and mergeable sparse-MLP adapter tensors. It is not",
        (
            "eligible for package serialization unless every held-out gate check passes."
            if len(layer_indices) == inspection.num_hidden_layers
            else "eligible for package serialization because this is a selected-layer causal pilot."
        ),
        "",
    ]
    (target / "sparse_teacher_training.md").write_text("\n".join(lines), encoding="utf-8")
    return report


__all__ = ["train_sparse_student"]
