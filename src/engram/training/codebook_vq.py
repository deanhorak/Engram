"""Strict-budget unrestricted vector-codebook quantization for SwiGLU MLPs.

The deployment representation stores one fixed-width code for each short
weight vector and a small FP16 codebook for each projection.  During QAT, each
vector has a bounded candidate set and trainable assignment logits.  The
forward pass always uses one hard codeword; a soft mixture supplies gradients
to the assignment logits and codebook without changing the deployed forward
value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .linear_constrained_vq import block_hadamard_function


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def unrestricted_codebook_vq_traffic(
    hidden_size: int,
    intermediate_size: int,
    *,
    layer_count: int,
    vector_size: int = 4,
    codebook_size: int = 128,
    cache_line_bytes: int = 64,
) -> dict[str, Any]:
    """Account complete cold bytes for fixed-width vector-codebook MLPs."""

    hidden_size = _positive_integer("hidden_size", hidden_size)
    intermediate_size = _positive_integer(
        "intermediate_size", intermediate_size
    )
    layer_count = _positive_integer("layer_count", layer_count)
    vector_size = _positive_integer("vector_size", vector_size)
    codebook_size = _positive_integer("codebook_size", codebook_size)
    cache_line_bytes = _positive_integer(
        "cache_line_bytes", cache_line_bytes
    )
    if codebook_size > 256:
        raise ValueError("codebook_size must be at most 256")
    shapes = (
        (intermediate_size, hidden_size),
        (intermediate_size, hidden_size),
        (hidden_size, intermediate_size),
    )
    elements = [output * input_ for output, input_ in shapes]
    if any(value % vector_size for value in elements):
        raise ValueError(
            "each projection element count must be vector aligned"
        )
    index_bits = max(1, math.ceil(math.log2(codebook_size)))
    code_stream_bytes = [
        ((value // vector_size) * index_bits + 7) // 8
        for value in elements
    ]
    side_scale_stream_bytes: list[int] = []
    for output_size, input_size in shapes:
        side_scale_stream_bytes.extend((2 * input_size, 2 * output_size))
    codebook_stream_bytes = [
        2 * codebook_size * vector_size for _ in shapes
    ]
    aligned_code_bytes = sum(
        _align(value, cache_line_bytes) for value in code_stream_bytes
    )
    aligned_side_scale_bytes = sum(
        _align(value, cache_line_bytes)
        for value in side_scale_stream_bytes
    )
    aligned_codebook_bytes = sum(
        _align(value, cache_line_bytes)
        for value in codebook_stream_bytes
    )
    layer_header_bytes = _align(512, cache_line_bytes)
    layer_block_bytes = _align(
        layer_header_bytes
        + aligned_code_bytes
        + aligned_side_scale_bytes
        + aligned_codebook_bytes,
        cache_line_bytes,
    )
    global_header_bytes = _align(1024, cache_line_bytes)
    directory_payload_bytes = 32 * layer_count
    directory_bytes = _align(directory_payload_bytes, cache_line_bytes)
    total_cold_bytes = (
        global_header_bytes
        + directory_bytes
        + layer_count * layer_block_bytes
    )
    dense_q4_bytes_per_layer = (
        3 * hidden_size * intermediate_size + 1
    ) // 2
    dense_q4_bytes = layer_count * dense_q4_bytes_per_layer
    fraction = total_cold_bytes / dense_q4_bytes
    return {
        "layout": "unrestricted_vector_codebook_v1",
        "projection_order": ["gate", "up", "down"],
        "vector_size": vector_size,
        "codebook_size": codebook_size,
        "index_bits": index_bits,
        "fixed_transform": (
            "normalized algorithmic block-Hadamard; no stored coefficients"
        ),
        "layer_count": layer_count,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "cache_line_bytes": cache_line_bytes,
        "code_stream_bytes_per_layer": code_stream_bytes,
        "side_scale_stream_bytes_per_layer": side_scale_stream_bytes,
        "codebook_stream_bytes_per_layer": codebook_stream_bytes,
        "code_payload_bytes_per_layer": sum(code_stream_bytes),
        "fp16_side_scale_payload_bytes_per_layer": sum(
            side_scale_stream_bytes
        ),
        "fp16_codebook_payload_bytes_per_layer": sum(
            codebook_stream_bytes
        ),
        "aligned_code_bytes_per_layer": aligned_code_bytes,
        "aligned_side_scale_bytes_per_layer": aligned_side_scale_bytes,
        "aligned_codebook_bytes_per_layer": aligned_codebook_bytes,
        "layer_header_bytes": layer_header_bytes,
        "layer_block_bytes": layer_block_bytes,
        "global_header_bytes": global_header_bytes,
        "directory_payload_bytes": directory_payload_bytes,
        "directory_cache_aligned_bytes": directory_bytes,
        "total_cold_bytes": total_cold_bytes,
        "dense_ideal_q4_bytes_per_layer": dense_q4_bytes_per_layer,
        "dense_ideal_q4_bytes": dense_q4_bytes,
        "fraction_of_dense_q4": fraction,
        "headroom_bytes_to_45_percent": math.floor(
            0.45 * dense_q4_bytes
        )
        - total_cold_bytes,
        "passes_45_percent_traffic_gate": fraction <= 0.45,
        "accounting_policy": (
            "complete serialized cold numerator including packed fixed-width "
            "codes, FP16 input/output scales, unrestricted FP16 codebooks, "
            "headers, directory, and independent cache-line padding; ideal "
            "code-only dense-Q4 denominator"
        ),
    }


def unrestricted_codebook_vq_mlp_class(torch: Any) -> type:
    """Create the hard-forward codebook-QAT module without eager Torch import."""

    functional = torch.nn.functional
    block_hadamard = block_hadamard_function(torch)

    def fake_fp16(value: Any) -> Any:
        rounded = value.to(torch.float16).to(value.dtype)
        return value + (rounded - value).detach()

    class CodebookProjection(torch.nn.Module):
        def __init__(
            self,
            teacher_weight: Any,
            state: Mapping[str, Any],
            *,
            block_size: int,
        ) -> None:
            super().__init__()
            teacher = (
                torch.as_tensor(teacher_weight).detach().clone().float()
            )
            if teacher.ndim != 2:
                raise ValueError("teacher_weight must be a matrix")
            if teacher.shape[0] % block_size or teacher.shape[1] % block_size:
                raise ValueError(
                    "projection dimensions must be divisible by block_size"
                )
            codebook = torch.as_tensor(state["codebook"]).float()
            if codebook.ndim != 2:
                raise ValueError("codebook must be a matrix")
            codebook_size, vector_size = codebook.shape
            if vector_size <= 0 or teacher.numel() % vector_size:
                raise ValueError(
                    "teacher weight must align to the codebook vector size"
                )
            groups_per_row = teacher.shape[1] // vector_size
            candidates = torch.as_tensor(state["candidates"]).long()
            expected_prefix = (teacher.shape[0], groups_per_row)
            if (
                candidates.ndim != 3
                or tuple(candidates.shape[:2]) != expected_prefix
                or candidates.shape[2] <= 0
            ):
                raise ValueError(
                    "candidates must have shape [output, groups, choices]"
                )
            if bool(torch.any(candidates < 0)) or bool(
                torch.any(candidates >= codebook_size)
            ):
                raise ValueError("candidate code is outside the codebook")
            logits = torch.as_tensor(state["logits"]).float()
            if tuple(logits.shape) != tuple(candidates.shape):
                raise ValueError("assignment logits do not match candidates")

            self.block_size = block_size
            self.vector_size = int(vector_size)
            self.codebook_size = int(codebook_size)
            self.codebook = torch.nn.Parameter(codebook.clone())
            self.assignment_logits = torch.nn.Parameter(logits.clone())
            self.input_scale = torch.nn.Parameter(
                torch.ones(teacher.shape[1], dtype=teacher.dtype)
            )
            self.output_scale = torch.nn.Parameter(
                torch.ones(teacher.shape[0], dtype=teacher.dtype)
            )
            self.register_buffer("teacher_weight", teacher)
            self.register_buffer("candidates", candidates.to(torch.int16))
            self.register_buffer("initial_codebook", codebook)
            self.register_buffer(
                "initial_codes",
                candidates[..., 0].to(torch.int16),
            )

        @property
        def input_size(self) -> int:
            return int(self.teacher_weight.shape[1])

        @property
        def output_size(self) -> int:
            return int(self.teacher_weight.shape[0])

        def hard_codes(self) -> Any:
            choice = self.assignment_logits.argmax(dim=-1, keepdim=True)
            return torch.gather(
                self.candidates.long(), -1, choice
            ).squeeze(-1)

        def decoded_weight(self) -> Any:
            candidates = self.candidates.long()
            words = fake_fp16(self.codebook)[candidates]
            probabilities = functional.softmax(
                self.assignment_logits.float(), dim=-1
            ).to(words.dtype)
            soft = torch.sum(words * probabilities[..., None], dim=-2)
            choice = self.assignment_logits.argmax(dim=-1, keepdim=True)
            hard = torch.gather(
                words,
                -2,
                choice[..., None].expand(
                    *choice.shape, self.vector_size
                ),
            ).squeeze(-2)
            straight_through = soft + (hard - soft).detach()
            return straight_through.reshape(
                self.output_size, self.input_size
            )

        def forward(self, value: Any) -> Any:
            if value.shape[-1] != self.input_size:
                raise ValueError("projection input dimension changed")
            scaled = value * fake_fp16(self.input_scale).to(value.dtype)
            transformed = block_hadamard(scaled, self.block_size)
            output = functional.linear(
                transformed, self.decoded_weight().to(transformed.dtype)
            )
            output = block_hadamard(output, self.block_size)
            return output * fake_fp16(self.output_scale).to(output.dtype)

        def anchor_loss(self) -> Any:
            reference = self.initial_codebook.square().mean().clamp_min(
                1e-12
            )
            codebook = (
                (self.codebook - self.initial_codebook).square().mean()
                / reference
            )
            scales = (
                (self.input_scale - 1.0).square().mean()
                + (self.output_scale - 1.0).square().mean()
            )
            return codebook + scales

        @torch.no_grad()
        def deployment_state(self) -> dict[str, Any]:
            return {
                "shape": tuple(self.teacher_weight.shape),
                "vector_size": self.vector_size,
                "codebook_size": self.codebook_size,
                "codes": self.hard_codes().to(torch.uint8).cpu(),
                "codebook": self.codebook.to(torch.float16).cpu(),
                "input_scale": self.input_scale.to(torch.float16).cpu(),
                "output_scale": self.output_scale.to(torch.float16).cpu(),
            }

    class UnrestrictedCodebookSwiGLU(torch.nn.Module):
        def __init__(
            self,
            gate_weight: Any,
            up_weight: Any,
            down_weight: Any,
            *,
            gate_state: Mapping[str, Any],
            up_state: Mapping[str, Any],
            down_state: Mapping[str, Any],
            block_size: int = 64,
        ) -> None:
            super().__init__()
            block_size = _positive_integer("block_size", block_size)
            if block_size & (block_size - 1):
                raise ValueError("block_size must be a power of two")
            gate = torch.as_tensor(gate_weight)
            up = torch.as_tensor(up_weight)
            down = torch.as_tensor(down_weight)
            if gate.ndim != 2 or tuple(up.shape) != tuple(gate.shape):
                raise ValueError("gate/up weights must be matching matrices")
            if tuple(down.shape) != (gate.shape[1], gate.shape[0]):
                raise ValueError(
                    "down weight must have shape [hidden, intermediate]"
                )
            self.gate = CodebookProjection(
                gate, gate_state, block_size=block_size
            )
            self.up = CodebookProjection(
                up, up_state, block_size=block_size
            )
            self.down = CodebookProjection(
                down, down_state, block_size=block_size
            )

        def forward(
            self,
            value: Any,
            *,
            return_intermediates: bool = False,
        ) -> Any:
            gate = self.gate(value)
            up = self.up(value)
            activation = functional.silu(gate) * up
            output = self.down(activation)
            if return_intermediates:
                return output, {
                    "gate": gate,
                    "up": up,
                    "activation": activation,
                }
            return output

        def teacher_forward(
            self,
            value: Any,
            *,
            return_intermediates: bool = False,
        ) -> Any:
            gate = functional.linear(value, self.gate.teacher_weight)
            up = functional.linear(value, self.up.teacher_weight)
            activation = functional.silu(gate) * up
            output = functional.linear(
                activation, self.down.teacher_weight
            )
            if return_intermediates:
                return output, {
                    "gate": gate,
                    "up": up,
                    "activation": activation,
                }
            return output

        def anchor_loss(self) -> Any:
            return (
                self.gate.anchor_loss()
                + self.up.anchor_loss()
                + self.down.anchor_loss()
            ) / 3

        @torch.no_grad()
        def deployment_state(self) -> dict[str, Any]:
            return {
                "format": "engram_unrestricted_vector_codebook_v1",
                "block_size": self.gate.block_size,
                "gate": self.gate.deployment_state(),
                "up": self.up.deployment_state(),
                "down": self.down.deployment_state(),
            }

    return UnrestrictedCodebookSwiGLU


__all__ = [
    "unrestricted_codebook_vq_mlp_class",
    "unrestricted_codebook_vq_traffic",
]
