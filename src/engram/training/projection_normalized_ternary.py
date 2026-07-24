"""Projection-normalized ternary SwiGLU experiments.

This module implements the deployment shape used by the projection-normalized
ternary screen:

* every gate, up, and down coefficient remains present;
* five ``{-1, 0, +1}`` coefficients are packed into one byte;
* every output row has one FP16 scale; and
* every projection has a learned FP16 RMSNorm gain on its input.

The Torch module keeps float master weights during quantization-aware training.
Its forward pass can interpolate from the original dense projection to the
fully normalized ternary projection, while the hard path is exactly the
operator represented by the byte model.
"""

from __future__ import annotations

import math
from typing import Any


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _packed_ternary_bytes(output_size: int, input_size: int) -> int:
    return (output_size * input_size + 4) // 5


def projection_normalized_ternary_traffic(
    hidden_size: int,
    intermediate_size: int,
    *,
    layer_count: int,
    cache_line_bytes: int = 64,
) -> dict[str, Any]:
    """Account complete cold bytes for projection-normalized ternary MLPs.

    The denominator deliberately remains the ideal code-only dense-Q4 SwiGLU
    payload. The numerator includes packed trits, row scales, RMSNorm gains,
    per-layer headers, a global header, a layer directory, and cache-line
    padding for every independently decoded stream.
    """

    hidden_size = _positive_integer("hidden_size", hidden_size)
    intermediate_size = _positive_integer(
        "intermediate_size", intermediate_size
    )
    layer_count = _positive_integer("layer_count", layer_count)
    cache_line_bytes = _positive_integer(
        "cache_line_bytes", cache_line_bytes
    )
    shapes = (
        (intermediate_size, hidden_size),
        (intermediate_size, hidden_size),
        (hidden_size, intermediate_size),
    )
    code_stream_bytes = [
        _packed_ternary_bytes(output_size, input_size)
        for output_size, input_size in shapes
    ]
    row_scale_stream_bytes = [2 * output_size for output_size, _ in shapes]
    rms_gain_stream_bytes = [2 * input_size for _, input_size in shapes]
    aligned_code_bytes = sum(
        _align(value, cache_line_bytes) for value in code_stream_bytes
    )
    aligned_row_scale_bytes = sum(
        _align(value, cache_line_bytes) for value in row_scale_stream_bytes
    )
    aligned_rms_gain_bytes = sum(
        _align(value, cache_line_bytes) for value in rms_gain_stream_bytes
    )
    layer_header_bytes = _align(512, cache_line_bytes)
    layer_block_bytes = _align(
        layer_header_bytes
        + aligned_code_bytes
        + aligned_row_scale_bytes
        + aligned_rms_gain_bytes,
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
        "layout": "projection_normalized_row_scaled_ternary_v1",
        "packing": "five base-3 trits per byte",
        "layer_count": layer_count,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "cache_line_bytes": cache_line_bytes,
        "code_stream_bytes_per_layer": code_stream_bytes,
        "row_scale_stream_bytes_per_layer": row_scale_stream_bytes,
        "rms_gain_stream_bytes_per_layer": rms_gain_stream_bytes,
        "ternary_code_payload_bytes_per_layer": sum(code_stream_bytes),
        "fp16_row_scale_payload_bytes_per_layer": sum(
            row_scale_stream_bytes
        ),
        "fp16_rms_gain_payload_bytes_per_layer": sum(rms_gain_stream_bytes),
        "aligned_ternary_code_bytes_per_layer": aligned_code_bytes,
        "aligned_row_scale_bytes_per_layer": aligned_row_scale_bytes,
        "aligned_rms_gain_bytes_per_layer": aligned_rms_gain_bytes,
        "layer_header_bytes": layer_header_bytes,
        "layer_block_bytes": layer_block_bytes,
        "global_header_bytes": global_header_bytes,
        "directory_payload_bytes": directory_payload_bytes,
        "directory_cache_aligned_bytes": directory_bytes,
        "total_cold_bytes": total_cold_bytes,
        "dense_ideal_q4_bytes_per_layer": dense_q4_bytes_per_layer,
        "dense_ideal_q4_bytes": dense_q4_bytes,
        "fraction_of_dense_q4": fraction,
        "headroom_bytes_to_45_percent": math.floor(0.45 * dense_q4_bytes)
        - total_cold_bytes,
        "passes_45_percent_traffic_gate": fraction <= 0.45,
        "accounting_policy": (
            "complete serialized cold numerator including packed codes, FP16 "
            "row scales, FP16 projection-local RMSNorm gains, headers, "
            "directory, and independent cache-line padding; ideal code-only "
            "dense-Q4 denominator"
        ),
    }


def projection_normalized_ternary_mlp_class(torch: Any) -> type:
    """Create the QAT module without importing Torch at package import time."""

    functional = torch.nn.functional

    def fake_fp16(value: Any) -> Any:
        rounded = value.to(torch.float16).to(value.dtype)
        return value + (rounded - value).detach()

    @torch.no_grad()
    def initial_row_scale(weight: Any) -> Any:
        scale = weight.abs().mean(dim=1).clamp_min(2**-24)
        for _ in range(12):
            code = torch.round(weight / scale[:, None]).clamp(-1, 1)
            scale = (
                (weight * code).sum(dim=1)
                / code.square().sum(dim=1).clamp_min(1)
            ).abs().clamp_min(2**-24)
        return scale

    class ProjectionNormalizedTernaryLinear(torch.nn.Module):
        def __init__(
            self,
            teacher_weight: Any,
            *,
            epsilon: float = 1e-6,
        ) -> None:
            super().__init__()
            weight = torch.as_tensor(teacher_weight).detach().clone().float()
            if weight.ndim != 2:
                raise ValueError("teacher_weight must be a matrix")
            if not math.isfinite(epsilon) or epsilon <= 0:
                raise ValueError("epsilon must be positive and finite")
            self.master = torch.nn.Parameter(weight.clone())
            self.log_row_scale = torch.nn.Parameter(
                initial_row_scale(weight).log()
            )
            self.rms_gain = torch.nn.Parameter(
                torch.ones(weight.shape[1], dtype=weight.dtype)
            )
            self.register_buffer("teacher_weight", weight)
            self.epsilon = float(epsilon)

        @property
        def input_size(self) -> int:
            return int(self.master.shape[1])

        @property
        def output_size(self) -> int:
            return int(self.master.shape[0])

        def normalized_input(self, value: Any) -> Any:
            if value.shape[-1] != self.input_size:
                raise ValueError("projection input dimension changed")
            rms = torch.sqrt(
                value.float().square().mean(dim=-1, keepdim=True)
                + self.epsilon
            )
            normalized = value / rms.to(value.dtype)
            return normalized * fake_fp16(self.rms_gain).to(value.dtype)

        def quantized_weight(self) -> Any:
            scale = fake_fp16(
                self.log_row_scale.exp().clamp(2**-24, 16.0)
            )
            code = torch.round(
                self.master / scale[:, None]
            ).clamp(-1, 1).detach()
            decoded = code * scale[:, None]
            # Identity STE for the master and an explicit path to row scales.
            return (
                self.master
                + (decoded - self.master).detach()
                + decoded
                - decoded.detach()
            )

        def forward(self, value: Any, quantization_fraction: float) -> Any:
            if (
                not math.isfinite(quantization_fraction)
                or not 0.0 <= quantization_fraction <= 1.0
            ):
                raise ValueError(
                    "quantization_fraction must be finite and in [0, 1]"
                )
            quantized = functional.linear(
                self.normalized_input(value),
                self.quantized_weight(),
            )
            if quantization_fraction == 1.0:
                return quantized
            dense = functional.linear(value, self.master)
            if quantization_fraction == 0.0:
                return dense
            return dense + quantization_fraction * (quantized - dense)

        def anchor_loss(self) -> Any:
            numerator = (self.master - self.teacher_weight).square().mean()
            denominator = self.teacher_weight.square().mean().clamp_min(1e-20)
            return numerator / denominator

        @torch.no_grad()
        def hard_code(self) -> Any:
            scale = self.log_row_scale.exp().clamp(2**-24, 16.0)
            return torch.round(
                self.master / scale[:, None]
            ).clamp(-1, 1).to(torch.int8)

        @torch.no_grad()
        def deployment_state(self) -> dict[str, Any]:
            return {
                "shape": tuple(self.master.shape),
                "code": self.hard_code().cpu(),
                "row_scale": self.log_row_scale.exp()
                .clamp(2**-24, 16.0)
                .to(torch.float16)
                .cpu(),
                "rms_gain": self.rms_gain.to(torch.float16).cpu(),
            }

    class ProjectionNormalizedTernarySwiGLU(torch.nn.Module):
        def __init__(
            self,
            gate_weight: Any,
            up_weight: Any,
            down_weight: Any,
            *,
            epsilon: float = 1e-6,
        ) -> None:
            super().__init__()
            gate = torch.as_tensor(gate_weight)
            up = torch.as_tensor(up_weight)
            down = torch.as_tensor(down_weight)
            if gate.ndim != 2 or tuple(up.shape) != tuple(gate.shape):
                raise ValueError("gate/up weights must be matching matrices")
            if tuple(down.shape) != (gate.shape[1], gate.shape[0]):
                raise ValueError(
                    "down weight must have shape [hidden, intermediate]"
                )
            self.gate = ProjectionNormalizedTernaryLinear(
                gate, epsilon=epsilon
            )
            self.up = ProjectionNormalizedTernaryLinear(up, epsilon=epsilon)
            self.down = ProjectionNormalizedTernaryLinear(
                down, epsilon=epsilon
            )

        def forward(
            self,
            value: Any,
            quantization_fraction: float,
            *,
            return_intermediates: bool = False,
        ) -> Any:
            gate = self.gate(value, quantization_fraction)
            up = self.up(value, quantization_fraction)
            activation = functional.silu(gate) * up
            output = self.down(activation, quantization_fraction)
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
                "format": "engram_projection_normalized_ternary_v1",
                "gate": self.gate.deployment_state(),
                "up": self.up.deployment_state(),
                "down": self.down.deployment_state(),
            }

    return ProjectionNormalizedTernarySwiGLU


__all__ = [
    "projection_normalized_ternary_mlp_class",
    "projection_normalized_ternary_traffic",
]
