"""High-dimensional lifted-binary quantization for SwiGLU MLPs.

A short vector of deployed 1-bit values is projected into a lower-dimensional
weight vector through a learned FP16 matrix.  The bit/vector dimension ratio
sets a fractional effective bit width while retaining sequential bit streams
and a small linear decoder rather than a large lookup table.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .linear_constrained_vq import block_hadamard_function


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def lifted_binary_traffic(
    hidden_size: int,
    intermediate_size: int,
    *,
    projection_dimensions: Sequence[tuple[int, int]],
    layer_count: int,
    cache_line_bytes: int = 64,
) -> dict[str, Any]:
    """Account complete cold bytes for lifted-binary MLP projections."""

    hidden_size = _positive_integer("hidden_size", hidden_size)
    intermediate_size = _positive_integer(
        "intermediate_size", intermediate_size
    )
    layer_count = _positive_integer("layer_count", layer_count)
    cache_line_bytes = _positive_integer(
        "cache_line_bytes", cache_line_bytes
    )
    dimensions = tuple(projection_dimensions)
    if len(dimensions) != 3:
        raise ValueError(
            "projection_dimensions must describe gate, up, and down"
        )
    checked: list[tuple[int, int]] = []
    for lifted, projected in dimensions:
        lifted = _positive_integer("lifted dimension", lifted)
        projected = _positive_integer("projected dimension", projected)
        if lifted <= projected:
            raise ValueError(
                "lifted dimension must exceed projected dimension"
            )
        checked.append((lifted, projected))
    shapes = (
        (intermediate_size, hidden_size),
        (intermediate_size, hidden_size),
        (hidden_size, intermediate_size),
    )
    elements = [output * input_ for output, input_ in shapes]
    code_stream_bytes = [
        ((value + projected - 1) // projected * lifted + 7) // 8
        for value, (lifted, projected) in zip(
            elements, checked, strict=True
        )
    ]
    side_scale_stream_bytes: list[int] = []
    for output_size, input_size in shapes:
        side_scale_stream_bytes.extend((2 * input_size, 2 * output_size))
    projection_stream_bytes = [
        2 * lifted * projected for lifted, projected in checked
    ]
    aligned_code_bytes = sum(
        _align(value, cache_line_bytes) for value in code_stream_bytes
    )
    aligned_side_scale_bytes = sum(
        _align(value, cache_line_bytes)
        for value in side_scale_stream_bytes
    )
    aligned_projection_bytes = sum(
        _align(value, cache_line_bytes)
        for value in projection_stream_bytes
    )
    layer_header_bytes = _align(512, cache_line_bytes)
    layer_block_bytes = _align(
        layer_header_bytes
        + aligned_code_bytes
        + aligned_side_scale_bytes
        + aligned_projection_bytes,
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
        "layout": "lifted_binary_projection_v1",
        "projection_order": ["gate", "up", "down"],
        "projection_dimensions": [list(value) for value in checked],
        "effective_bits_per_weight": [
            lifted / projected for lifted, projected in checked
        ],
        "fixed_transform": (
            "normalized algorithmic block-Hadamard; no stored coefficients"
        ),
        "layer_count": layer_count,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "cache_line_bytes": cache_line_bytes,
        "code_stream_bytes_per_layer": code_stream_bytes,
        "side_scale_stream_bytes_per_layer": side_scale_stream_bytes,
        "projection_stream_bytes_per_layer": projection_stream_bytes,
        "binary_code_payload_bytes_per_layer": sum(code_stream_bytes),
        "fp16_side_scale_payload_bytes_per_layer": sum(
            side_scale_stream_bytes
        ),
        "fp16_projection_payload_bytes_per_layer": sum(
            projection_stream_bytes
        ),
        "aligned_code_bytes_per_layer": aligned_code_bytes,
        "aligned_side_scale_bytes_per_layer": aligned_side_scale_bytes,
        "aligned_projection_bytes_per_layer": aligned_projection_bytes,
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
            "complete serialized cold numerator including packed binary "
            "codes, FP16 input/output scales, FP16 lifted projection maps, "
            "headers, directory, tensor-tail padding, and independent "
            "cache-line padding; ideal code-only dense-Q4 denominator"
        ),
    }


def lifted_binary_mlp_class(torch: Any) -> type:
    """Create the hard-forward lifted-binary QAT module."""

    functional = torch.nn.functional
    block_hadamard = block_hadamard_function(torch)

    def fake_fp16(value: Any) -> Any:
        rounded = value.to(torch.float16).to(value.dtype)
        return value + (rounded - value).detach()

    class LiftedBinaryProjection(torch.nn.Module):
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
            projection = torch.as_tensor(state["projection"]).float()
            if projection.ndim != 2:
                raise ValueError("projection must be a matrix")
            projected_size, lifted_size = projection.shape
            if lifted_size <= projected_size:
                raise ValueError(
                    "lifted dimension must exceed projected dimension"
                )
            groups = (teacher.numel() + projected_size - 1) // projected_size
            bits = torch.as_tensor(state["bits"]).float()
            if tuple(bits.shape) != (groups, lifted_size):
                raise ValueError(
                    "bits must have shape [padded groups, lifted dimension]"
                )
            if not bool(torch.all((bits == -1) | (bits == 1))):
                raise ValueError("initial lifted values must be -1 or +1")
            input_scale = torch.as_tensor(
                state.get(
                    "input_scale",
                    torch.ones(teacher.shape[1], dtype=teacher.dtype),
                )
            ).float()
            output_scale = torch.as_tensor(
                state.get(
                    "output_scale",
                    torch.ones(teacher.shape[0], dtype=teacher.dtype),
                )
            ).float()
            if tuple(input_scale.shape) != (teacher.shape[1],):
                raise ValueError("input_scale shape changed")
            if tuple(output_scale.shape) != (teacher.shape[0],):
                raise ValueError("output_scale shape changed")

            self.block_size = block_size
            self.projected_size = int(projected_size)
            self.lifted_size = int(lifted_size)
            self.logical_elements = int(teacher.numel())
            self.proxy_scale = math.sqrt(
                teacher.shape[0] + teacher.shape[1]
            ) / 2.0
            self.proxy = torch.nn.Parameter(bits / self.proxy_scale)
            self.projection = torch.nn.Parameter(projection.clone())
            self.input_scale = torch.nn.Parameter(input_scale.clone())
            self.output_scale = torch.nn.Parameter(output_scale.clone())
            self.register_buffer("teacher_weight", teacher)
            self.register_buffer("initial_bits", bits.to(torch.int8))
            self.register_buffer("initial_projection", projection)
            self.register_buffer("initial_input_scale", input_scale)
            self.register_buffer("initial_output_scale", output_scale)

        @property
        def input_size(self) -> int:
            return int(self.teacher_weight.shape[1])

        @property
        def output_size(self) -> int:
            return int(self.teacher_weight.shape[0])

        def quantized_bits(self) -> Any:
            continuous = self.proxy_scale * self.proxy
            hard = torch.where(
                continuous >= 0,
                torch.ones_like(continuous),
                -torch.ones_like(continuous),
            )
            return continuous + (hard - continuous).detach()

        def decoded_weight(self) -> Any:
            decoded = functional.linear(
                self.quantized_bits(),
                fake_fp16(self.projection),
            )
            weight = decoded.flatten()[: self.logical_elements].reshape(
                self.output_size, self.input_size
            )
            return weight * fake_fp16(self.output_scale)[:, None]

        def forward(self, value: Any) -> Any:
            if value.shape[-1] != self.input_size:
                raise ValueError("projection input dimension changed")
            scaled = value * fake_fp16(self.input_scale).to(value.dtype)
            transformed = block_hadamard(scaled, self.block_size)
            output = functional.linear(
                transformed, self.decoded_weight().to(transformed.dtype)
            )
            return block_hadamard(output, self.block_size)

        def anchor_loss(self) -> Any:
            projection_reference = (
                self.initial_projection.square().mean().clamp_min(1e-12)
            )
            projection = (
                (self.projection - self.initial_projection).square().mean()
                / projection_reference
            )
            input_reference = self.initial_input_scale.abs().mean().clamp_min(
                1e-12
            )
            output_reference = (
                self.initial_output_scale.abs().mean().clamp_min(1e-12)
            )
            scales = (
                (self.input_scale - self.initial_input_scale)
                .square()
                .mean()
                / input_reference.square()
                + (self.output_scale - self.initial_output_scale)
                .square()
                .mean()
                / output_reference.square()
            )
            return projection + scales

        @torch.no_grad()
        def hard_bits(self) -> Any:
            return torch.where(
                self.proxy >= 0,
                torch.ones_like(self.proxy, dtype=torch.int8),
                -torch.ones_like(self.proxy, dtype=torch.int8),
            )

        @torch.no_grad()
        def deployment_state(self) -> dict[str, Any]:
            return {
                "shape": tuple(self.teacher_weight.shape),
                "projected_size": self.projected_size,
                "lifted_size": self.lifted_size,
                "bits": self.hard_bits().cpu(),
                "projection": self.projection.to(torch.float16).cpu(),
                "input_scale": self.input_scale.to(torch.float16).cpu(),
                "output_scale": self.output_scale.to(torch.float16).cpu(),
            }

    class LiftedBinarySwiGLU(torch.nn.Module):
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
            self.gate = LiftedBinaryProjection(
                gate, gate_state, block_size=block_size
            )
            self.up = LiftedBinaryProjection(
                up, up_state, block_size=block_size
            )
            self.down = LiftedBinaryProjection(
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
                "format": "engram_lifted_binary_projection_v1",
                "block_size": self.gate.block_size,
                "gate": self.gate.deployment_state(),
                "up": self.up.deployment_state(),
                "down": self.down.deployment_state(),
            }

    return LiftedBinarySwiGLU


__all__ = [
    "lifted_binary_mlp_class",
    "lifted_binary_traffic",
]
