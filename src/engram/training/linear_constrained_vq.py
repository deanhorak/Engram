"""Linear-constrained low-bit vector quantization for SwiGLU MLPs.

Each matrix is surrounded by fixed, normalized block-Hadamard transforms and
learned FP16 input/output scales. Consecutive groups of four transformed
weights are decoded from low-bit symbols through one learned 4x4 affine map.
The symbols remain independently packable, while the affine map lets each
four-symbol group represent a vector codeword rather than four unrelated
scalar levels.

The QAT path uses the differentiable gradient estimator from LC-QAT. The
forward value is always the rounded, clipped symbol; only its backward
derivative is smoothed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _symbol_bytes(elements: int, levels: int) -> int:
    if levels == 3:
        return (elements + 4) // 5
    if levels == 4:
        return (elements + 3) // 4
    raise ValueError("only ternary and quaternary symbols are supported")


def linear_constrained_vq_traffic(
    hidden_size: int,
    intermediate_size: int,
    *,
    projection_levels: Sequence[int],
    layer_count: int,
    cache_line_bytes: int = 64,
) -> dict[str, Any]:
    """Account complete cold bytes for the linear-constrained VQ layout."""

    hidden_size = _positive_integer("hidden_size", hidden_size)
    intermediate_size = _positive_integer(
        "intermediate_size", intermediate_size
    )
    layer_count = _positive_integer("layer_count", layer_count)
    cache_line_bytes = _positive_integer(
        "cache_line_bytes", cache_line_bytes
    )
    levels = tuple(projection_levels)
    if len(levels) != 3 or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in (3, 4)
        for value in levels
    ):
        raise ValueError(
            "projection_levels must contain three values from {3, 4}"
        )
    shapes = (
        (intermediate_size, hidden_size),
        (intermediate_size, hidden_size),
        (hidden_size, intermediate_size),
    )
    code_stream_bytes = [
        _symbol_bytes(output_size * input_size, value_levels)
        for (output_size, input_size), value_levels in zip(
            shapes, levels, strict=True
        )
    ]
    side_scale_stream_bytes: list[int] = []
    for output_size, input_size in shapes:
        side_scale_stream_bytes.extend((2 * input_size, 2 * output_size))
    # One FP16 4x4 matrix plus one four-value FP16 bias per projection.
    affine_stream_bytes = [2 * (4 * 4 + 4)] * 3
    aligned_code_bytes = sum(
        _align(value, cache_line_bytes) for value in code_stream_bytes
    )
    aligned_side_scale_bytes = sum(
        _align(value, cache_line_bytes)
        for value in side_scale_stream_bytes
    )
    aligned_affine_bytes = sum(
        _align(value, cache_line_bytes) for value in affine_stream_bytes
    )
    layer_header_bytes = _align(512, cache_line_bytes)
    layer_block_bytes = _align(
        layer_header_bytes
        + aligned_code_bytes
        + aligned_side_scale_bytes
        + aligned_affine_bytes,
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
        "layout": "linear_constrained_vector_qat_v1",
        "projection_order": ["gate", "up", "down"],
        "projection_levels": list(levels),
        "symbol_packing": {
            "ternary": "five base-3 symbols per byte",
            "quaternary": "four two-bit symbols per byte",
        },
        "fixed_transform": (
            "normalized algorithmic block-Hadamard; no stored coefficients"
        ),
        "layer_count": layer_count,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "cache_line_bytes": cache_line_bytes,
        "code_stream_bytes_per_layer": code_stream_bytes,
        "side_scale_stream_bytes_per_layer": side_scale_stream_bytes,
        "affine_stream_bytes_per_layer": affine_stream_bytes,
        "symbol_payload_bytes_per_layer": sum(code_stream_bytes),
        "fp16_side_scale_payload_bytes_per_layer": sum(
            side_scale_stream_bytes
        ),
        "fp16_affine_payload_bytes_per_layer": sum(affine_stream_bytes),
        "aligned_symbol_bytes_per_layer": aligned_code_bytes,
        "aligned_side_scale_bytes_per_layer": aligned_side_scale_bytes,
        "aligned_affine_bytes_per_layer": aligned_affine_bytes,
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
            "complete serialized cold numerator including packed symbols, "
            "FP16 input/output scales, FP16 affine vector maps, headers, "
            "directory, and independent cache-line padding; ideal code-only "
            "dense-Q4 denominator"
        ),
    }


def block_hadamard_function(torch: Any):
    """Return a differentiable normalized block-Hadamard transform."""

    def block_hadamard(value: Any, block_size: int = 64) -> Any:
        block_size_checked = _positive_integer("block_size", block_size)
        if block_size_checked & (block_size_checked - 1):
            raise ValueError("block_size must be a power of two")
        if value.shape[-1] % block_size_checked:
            raise ValueError(
                "final tensor dimension must be divisible by block_size"
            )
        shape = value.shape
        prefix = (*shape[:-1], shape[-1] // block_size_checked)
        transformed = value.reshape(
            *prefix, block_size_checked
        )
        width = 1
        while width < block_size_checked:
            grouped = transformed.reshape(
                *prefix,
                block_size_checked // (2 * width),
                2,
                width,
            )
            left = grouped[..., 0, :]
            right = grouped[..., 1, :]
            transformed = torch.cat(
                (left + right, left - right), dim=-1
            ).reshape(*prefix, block_size_checked)
            width *= 2
        return transformed.reshape(shape) / math.sqrt(block_size_checked)

    return block_hadamard


def linear_constrained_vq_mlp_class(torch: Any) -> type:
    """Create the LC-VQ QAT module without importing Torch eagerly."""

    functional = torch.nn.functional
    block_hadamard = block_hadamard_function(torch)

    class DifferentiableSymbolEstimator(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any,
            value: Any,
            maximum: int,
            sharpness: float,
            derivative_cap: float,
        ) -> Any:
            ctx.save_for_backward(value)
            ctx.maximum = maximum
            ctx.sharpness = sharpness
            ctx.derivative_cap = derivative_cap
            return torch.round(value).clamp(0, maximum)

        @staticmethod
        def backward(ctx: Any, gradient: Any):
            (value,) = ctx.saved_tensors
            lower = torch.floor(value).clamp(0, ctx.maximum)
            relative = value - lower
            distance = torch.abs(2.0 * relative - 1.0).clamp_min(1e-6)
            derivative = (
                distance.pow(1.0 / ctx.sharpness - 1.0)
                / ctx.sharpness
            ).clamp(max=ctx.derivative_cap)
            return gradient * derivative, None, None, None

    def fake_fp16(value: Any) -> Any:
        rounded = value.to(torch.float16).to(value.dtype)
        return value + (rounded - value).detach()

    class LinearConstrainedProjection(torch.nn.Module):
        def __init__(
            self,
            teacher_weight: Any,
            state: Mapping[str, Any],
            *,
            block_size: int,
            sharpness: float,
            derivative_cap: float,
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
            symbols = torch.as_tensor(state["symbols"]).float()
            if tuple(symbols.shape) != tuple(teacher.shape):
                raise ValueError("symbol matrix shape does not match weight")
            levels = state["levels"]
            if (
                isinstance(levels, bool)
                or not isinstance(levels, int)
                or levels not in (3, 4)
            ):
                raise ValueError("levels must be 3 or 4")
            if bool(torch.any(symbols < 0)) or bool(
                torch.any(symbols > levels - 1)
            ):
                raise ValueError("initial symbols are outside their alphabet")
            matrix = torch.as_tensor(state["matrix"]).float()
            bias = torch.as_tensor(state["bias"]).float()
            if tuple(matrix.shape) != (4, 4) or tuple(bias.shape) != (4,):
                raise ValueError("affine codebook must have shapes [4,4]/[4]")
            if not math.isfinite(sharpness) or sharpness <= 0:
                raise ValueError("sharpness must be positive and finite")
            if not math.isfinite(derivative_cap) or derivative_cap <= 0:
                raise ValueError(
                    "derivative_cap must be positive and finite"
                )

            self.levels = levels
            self.block_size = block_size
            self.sharpness = float(sharpness)
            self.derivative_cap = float(derivative_cap)
            self.proxy_scale = math.sqrt(
                teacher.shape[0] + teacher.shape[1]
            ) / 2.0
            midpoint = (levels - 1) / 2.0
            self.proxy = torch.nn.Parameter(
                (symbols - midpoint) / self.proxy_scale
            )
            self.matrix = torch.nn.Parameter(matrix.clone())
            self.bias = torch.nn.Parameter(bias.clone())
            self.input_scale = torch.nn.Parameter(
                torch.ones(teacher.shape[1], dtype=teacher.dtype)
            )
            self.output_scale = torch.nn.Parameter(
                torch.ones(teacher.shape[0], dtype=teacher.dtype)
            )
            self.register_buffer("teacher_weight", teacher)
            self.register_buffer("initial_symbols", symbols.to(torch.int8))
            self.register_buffer("initial_matrix", matrix)
            self.register_buffer("initial_bias", bias)

        @property
        def input_size(self) -> int:
            return int(self.teacher_weight.shape[1])

        @property
        def output_size(self) -> int:
            return int(self.teacher_weight.shape[0])

        def quantized_symbols(self) -> Any:
            midpoint = (self.levels - 1) / 2.0
            continuous = self.proxy_scale * self.proxy + midpoint
            return DifferentiableSymbolEstimator.apply(
                continuous,
                self.levels - 1,
                self.sharpness,
                self.derivative_cap,
            )

        def decoded_weight(self) -> Any:
            symbols = self.quantized_symbols()
            groups = symbols.reshape(
                self.output_size, self.input_size // 4, 4
            )
            decoded = functional.linear(
                groups,
                fake_fp16(self.matrix),
                fake_fp16(self.bias),
            )
            return decoded.reshape(self.output_size, self.input_size)

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
            affine_reference = (
                self.initial_matrix.square().mean()
                + self.initial_bias.square().mean()
            ).clamp_min(1e-12)
            affine = (
                (self.matrix - self.initial_matrix).square().mean()
                + (self.bias - self.initial_bias).square().mean()
            ) / affine_reference
            scales = (
                (self.input_scale - 1.0).square().mean()
                + (self.output_scale - 1.0).square().mean()
            )
            return affine + scales

        @torch.no_grad()
        def hard_symbols(self) -> Any:
            midpoint = (self.levels - 1) / 2.0
            return torch.round(
                self.proxy_scale * self.proxy + midpoint
            ).clamp(0, self.levels - 1).to(torch.int8)

        @torch.no_grad()
        def deployment_state(self) -> dict[str, Any]:
            return {
                "shape": tuple(self.teacher_weight.shape),
                "levels": self.levels,
                "symbols": self.hard_symbols().cpu(),
                "matrix": self.matrix.to(torch.float16).cpu(),
                "bias": self.bias.to(torch.float16).cpu(),
                "input_scale": self.input_scale.to(torch.float16).cpu(),
                "output_scale": self.output_scale.to(torch.float16).cpu(),
            }

    class LinearConstrainedVectorSwiGLU(torch.nn.Module):
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
            sharpness: float = 5.0,
            derivative_cap: float = 3.0,
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
            kwargs = {
                "block_size": block_size,
                "sharpness": sharpness,
                "derivative_cap": derivative_cap,
            }
            self.gate = LinearConstrainedProjection(
                gate, gate_state, **kwargs
            )
            self.up = LinearConstrainedProjection(up, up_state, **kwargs)
            self.down = LinearConstrainedProjection(
                down, down_state, **kwargs
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
                "format": "engram_linear_constrained_vector_qat_v1",
                "block_size": self.gate.block_size,
                "gate": self.gate.deployment_state(),
                "up": self.up.deployment_state(),
                "down": self.down.deployment_state(),
            }

    return LinearConstrainedVectorSwiGLU


__all__ = [
    "block_hadamard_function",
    "linear_constrained_vq_mlp_class",
    "linear_constrained_vq_traffic",
]
