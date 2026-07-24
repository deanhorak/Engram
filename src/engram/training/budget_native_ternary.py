"""Budget-native grouped-ternary MLPs and distillation primitives.

The deployable representation is fixed before training: full-width SwiGLU
projections, five ternary coefficients per byte, and one non-learned,
MSE-refined FP16 scale per contiguous weight group. Float master weights and
optimizer state exist only while training. Evaluation can install decoded
weights from the serialized artifact so it never silently re-quantizes a
different in-memory representation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .budget_native_ternary_codec import (
    BudgetNativeTernaryLayerWeights,
    grouped_ternary_quantize,
)


def quantization_strength_for_step(
    step: int,
    *,
    total_steps: int,
    dense_warmup_steps: int,
    anneal_steps: int,
) -> float:
    """Return the continual-QAT strength for one zero-based training step."""

    for value, name in (
        (step, "step"),
        (total_steps, "total_steps"),
        (dense_warmup_steps, "dense_warmup_steps"),
        (anneal_steps, "anneal_steps"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= step < total_steps:
        raise ValueError("step must lie within the training schedule")
    if dense_warmup_steps < 0 or anneal_steps <= 0:
        raise ValueError(
            "dense_warmup_steps must be non-negative and anneal_steps positive"
        )
    if dense_warmup_steps + anneal_steps > total_steps:
        raise ValueError("warm-up and anneal phases exceed total_steps")
    if step < dense_warmup_steps:
        return 0.0
    if step >= dense_warmup_steps + anneal_steps:
        return 1.0
    progress = (step - dense_warmup_steps + 1) / anneal_steps
    # A half-cosine avoids the abrupt derivative change of a linear blend.
    return float(0.5 - 0.5 * math.cos(math.pi * progress))


def layer_quantization_strengths_for_step(
    step: int,
    *,
    total_steps: int,
    dense_warmup_steps: int,
    anneal_steps: int,
    layer_count: int,
    transition_mode: str,
) -> tuple[float, ...]:
    """Return one deployment-strength value per transformer layer.

    ``global`` applies the same half-cosine transition to every MLP.
    ``deepest_first`` spreads the transition budget across layers and
    converts one layer at a time, beginning at the output end of the model.
    """

    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count <= 0
    ):
        raise ValueError("layer_count must be a positive integer")
    if transition_mode not in {"global", "deepest_first"}:
        raise ValueError(
            "transition_mode must be 'global' or 'deepest_first'"
        )
    global_strength = quantization_strength_for_step(
        step,
        total_steps=total_steps,
        dense_warmup_steps=dense_warmup_steps,
        anneal_steps=anneal_steps,
    )
    if transition_mode == "global" or global_strength in {0.0, 1.0}:
        return (global_strength,) * layer_count

    progress = (
        (step - dense_warmup_steps + 1) * layer_count / anneal_steps
    )
    strengths = [0.0] * layer_count
    for transition_rank in range(layer_count):
        local_progress = min(
            1.0,
            max(0.0, progress - transition_rank),
        )
        local_strength = 0.5 - 0.5 * math.cos(
            math.pi * local_progress
        )
        layer = layer_count - transition_rank - 1
        strengths[layer] = float(local_strength)
    return tuple(strengths)


def confidence_weighted_kl(
    student_logits: Any,
    teacher_logits: Any,
    valid_mask: Any,
    torch: Any,
    *,
    temperature: float = 1.0,
    confidence_weight: float = 1.0,
) -> Any:
    """Teacher-to-student KL with bounded teacher-confidence weighting."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher and student logits must have matching shapes")
    if valid_mask.shape != student_logits.shape[:-1]:
        raise ValueError("valid_mask shape must match the token dimensions")
    if (
        not math.isfinite(temperature)
        or temperature <= 0
        or not math.isfinite(confidence_weight)
        or confidence_weight < 0
    ):
        raise ValueError(
            "temperature must be positive and confidence_weight non-negative"
        )
    teacher_logp = torch.nn.functional.log_softmax(
        teacher_logits.detach() / temperature,
        dim=-1,
    )
    teacher_probability = teacher_logp.exp()
    student_logp = torch.nn.functional.log_softmax(
        student_logits / temperature,
        dim=-1,
    )
    token_kl = torch.nn.functional.kl_div(
        student_logp,
        teacher_probability,
        reduction="none",
    ).sum(dim=-1)
    vocabulary_size = student_logits.shape[-1]
    if vocabulary_size <= 1:
        confidence = torch.ones_like(token_kl)
    else:
        entropy = -(teacher_probability * teacher_logp).sum(dim=-1)
        confidence = 1.0 - entropy / math.log(vocabulary_size)
        confidence = confidence.clamp(0.0, 1.0)
    weights = 1.0 + confidence_weight * confidence
    selected_weights = weights[valid_mask]
    normalized_weights = selected_weights / selected_weights.mean().clamp_min(
        1e-8
    )
    return (
        token_kl[valid_mask] * normalized_weights
    ).mean() * temperature**2


def masked_linear_cka_loss(
    student_hidden: Any,
    teacher_hidden: Any,
    valid_mask: Any,
    torch: Any,
) -> Any:
    """Return one minus linear CKA over valid token representations."""

    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError("teacher and student hidden states must match")
    if valid_mask.shape != student_hidden.shape[:-1]:
        raise ValueError("valid_mask shape must match the token dimensions")
    student = student_hidden[valid_mask].float()
    teacher = teacher_hidden.detach()[valid_mask].float()
    if student.shape[0] < 2:
        raise ValueError("linear CKA requires at least two valid tokens")
    student = student - student.mean(dim=0, keepdim=True)
    teacher = teacher - teacher.mean(dim=0, keepdim=True)
    student_gram = student @ student.transpose(0, 1)
    teacher_gram = teacher @ teacher.transpose(0, 1)
    numerator = (student_gram * teacher_gram).sum()
    denominator = (
        student_gram.square().sum()
        * teacher_gram.square().sum()
    ).sqrt().clamp_min(1e-12)
    similarity = (numerator / denominator).clamp(0.0, 1.0)
    return 1.0 - similarity


def grouped_ternary_mlp_class(torch: Any) -> type:
    """Create the Torch MLP class without importing Torch at package import."""

    functional = torch.nn.functional
    scale_fit_iterations = 2

    class GroupedTernaryProjection(torch.nn.Module):
        def __init__(self, weight: Any, *, group_size: int) -> None:
            super().__init__()
            source = torch.as_tensor(weight).detach().clone().float()
            if source.ndim != 2:
                raise ValueError("projection weight must be a matrix")
            if (
                isinstance(group_size, bool)
                or not isinstance(group_size, int)
                or group_size <= 0
            ):
                raise ValueError("group_size must be a positive integer")
            self.master = torch.nn.Parameter(source)
            self.group_size = group_size
            self.register_buffer(
                "deployment_weight",
                torch.empty(0, dtype=source.dtype),
                persistent=False,
            )

        @property
        def input_size(self) -> int:
            return int(self.master.shape[1])

        @property
        def output_size(self) -> int:
            return int(self.master.shape[0])

        def _scales_and_codes(self) -> tuple[Any, Any]:
            flat = self.master.reshape(-1)
            group_count = (
                flat.numel() + self.group_size - 1
            ) // self.group_size
            padded_size = group_count * self.group_size
            if padded_size != flat.numel():
                padded = functional.pad(flat, (0, padded_size - flat.numel()))
            else:
                padded = flat
            groups = padded.reshape(group_count, self.group_size)
            counts = torch.full(
                (group_count,),
                self.group_size,
                device=flat.device,
                dtype=flat.dtype,
            )
            tail = flat.numel() % self.group_size
            if tail:
                counts[-1] = tail
            # Scales are statistics, not optimizer parameters. Stopping their
            # gradient avoids the scale/zero-ratio collapse seen with learned
            # ternary scales while the identity STE still updates every master
            # coefficient.
            detached_groups = groups.detach()
            scales = (detached_groups.abs().sum(dim=1) / counts).clamp_min(
                2**-24
            )
            for _ in range(scale_fit_iterations):
                rounded_scale = scales.to(torch.float16).to(flat.dtype)
                codes = torch.round(
                    detached_groups / rounded_scale[:, None]
                ).clamp(-1, 1)
                denominator = codes.square().sum(dim=1)
                fitted = (detached_groups * codes).sum(
                    dim=1
                ) / denominator.clamp_min(1)
                scales = torch.where(
                    denominator > 0,
                    fitted,
                    torch.ones_like(fitted),
                )
            scales = scales.to(torch.float16).to(flat.dtype)
            codes = torch.round(groups / scales[:, None]).clamp(-1, 1)
            return scales, codes.reshape(-1)[: flat.numel()]

        def quantized_weight(self) -> Any:
            scales, codes = self._scales_and_codes()
            expanded_scales = torch.repeat_interleave(
                scales,
                self.group_size,
            )[: self.master.numel()]
            decoded = (codes * expanded_scales).reshape_as(self.master)
            return self.master + (decoded - self.master).detach()

        def effective_weight(self, quantization_strength: float) -> Any:
            if self.deployment_weight.numel():
                return self.deployment_weight
            if (
                not math.isfinite(quantization_strength)
                or not 0.0 <= quantization_strength <= 1.0
            ):
                raise ValueError(
                    "quantization_strength must be finite and in [0, 1]"
                )
            if quantization_strength == 0.0:
                return self.master
            quantized = self.quantized_weight()
            if quantization_strength == 1.0:
                return quantized
            return self.master + quantization_strength * (
                quantized - self.master
            )

        def forward(self, value: Any, *, quantization_strength: float) -> Any:
            if value.shape[-1] != self.input_size:
                raise ValueError("projection input dimension changed")
            return functional.linear(
                value,
                self.effective_weight(quantization_strength).to(value.dtype),
            )

        @torch.no_grad()
        def install_deployment_weight(self, weight: Any) -> None:
            decoded = torch.as_tensor(
                weight,
                device=self.master.device,
                dtype=self.master.dtype,
            )
            if decoded.shape != self.master.shape:
                raise ValueError("deployment weight shape changed")
            if not bool(torch.all(torch.isfinite(decoded))):
                raise ValueError("deployment weight must be finite")
            self.deployment_weight = decoded.detach().clone()

        @torch.no_grad()
        def clear_deployment_weight(self) -> None:
            self.deployment_weight = torch.empty(
                0,
                device=self.master.device,
                dtype=self.master.dtype,
            )

        @torch.no_grad()
        def zero_fraction(self) -> float:
            _, codes = self._scales_and_codes()
            return float((codes == 0).float().mean().cpu())

    class GroupedTernarySwiGLU(torch.nn.Module):
        def __init__(self, base: Any, *, group_size: int = 128) -> None:
            super().__init__()
            if (
                base.gate_proj.bias is not None
                or base.up_proj.bias is not None
                or base.down_proj.bias is not None
            ):
                raise ValueError("bias-enabled MLP projections are unsupported")
            self.gate = GroupedTernaryProjection(
                base.gate_proj.weight,
                group_size=group_size,
            )
            self.up = GroupedTernaryProjection(
                base.up_proj.weight,
                group_size=group_size,
            )
            self.down = GroupedTernaryProjection(
                base.down_proj.weight,
                group_size=group_size,
            )
            self.act_fn = base.act_fn
            self.quantization_strength = 1.0
            self.last_output = None

        @property
        def hidden_size(self) -> int:
            return self.gate.input_size

        @property
        def intermediate_size(self) -> int:
            return self.gate.output_size

        @property
        def deployment_loaded(self) -> bool:
            return all(
                projection.deployment_weight.numel()
                for projection in (self.gate, self.up, self.down)
            )

        def set_quantization_strength(self, value: float) -> None:
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    "quantization strength must be finite and in [0, 1]"
                )
            self.quantization_strength = float(value)

        def forward(self, hidden: Any) -> Any:
            gate = self.gate(
                hidden,
                quantization_strength=self.quantization_strength,
            )
            up = self.up(
                hidden,
                quantization_strength=self.quantization_strength,
            )
            activation = self.act_fn(gate) * up
            output = self.down(
                activation,
                quantization_strength=self.quantization_strength,
            )
            self.last_output = output
            return output

        @torch.no_grad()
        def zero_fractions(self) -> dict[str, float]:
            return {
                "gate": self.gate.zero_fraction(),
                "up": self.up.zero_fraction(),
                "down": self.down.zero_fraction(),
            }

        @torch.no_grad()
        def deployment_layer_weights(
            self,
        ) -> BudgetNativeTernaryLayerWeights:
            return BudgetNativeTernaryLayerWeights(
                self.gate.master.detach().cpu(),
                self.up.master.detach().cpu(),
                self.down.master.detach().cpu(),
            )

        @torch.no_grad()
        def install_deployment_weights(
            self,
            *,
            gate: Any,
            up: Any,
            down: Any,
        ) -> None:
            self.gate.install_deployment_weight(gate)
            self.up.install_deployment_weight(up)
            self.down.install_deployment_weight(down)
            self.quantization_strength = 1.0

    return GroupedTernarySwiGLU


def numpy_grouped_ternary_zero_fraction(
    values: Any,
    *,
    group_size: int,
) -> float:
    """Return the exact deployment-code zero fraction for diagnostics."""

    codes, _ = grouped_ternary_quantize(values, group_size=group_size)
    return float(np.mean(codes == 0))


__all__ = [
    "confidence_weighted_kl",
    "grouped_ternary_mlp_class",
    "layer_quantization_strengths_for_step",
    "masked_linear_cka_loss",
    "numpy_grouped_ternary_zero_fraction",
    "quantization_strength_for_step",
]
