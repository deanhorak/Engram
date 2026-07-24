"""Cache-reused recurrent compact SwiGLU layers.

This module explores a deliberate compute-for-memory trade. One compact MLP
payload is read once, then applied repeatedly while its weights are expected to
remain resident in cache. Small cycle-specific diagonal adapters let the tied
operator perform a different refinement on each recurrence.

The byte model is intentionally conservative about serialized metadata and
adapter reads. It is optimistic only about the central hypothesis--that the
compact Q4 matrices are cold once and hot for subsequent cycles. A native
cache/latency measurement is therefore required before a recurrent artifact
can satisfy the physical-traffic gate.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from engram.training.width_pruned_codec import width_pruned_q4_traffic


def _align(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("alignment operands must be non-negative/positive")
    return ((value + alignment - 1) // alignment) * alignment


def recurrent_compact_q4_traffic(
    hidden_size: int,
    source_intermediate_size: int,
    layer_widths: Sequence[int],
    *,
    cycles: int,
    adapter_rank: int = 0,
    cache_line_bytes: int = 64,
) -> dict[str, Any]:
    """Account complete cold bytes for a tied recurrent compact MLP package.

    Each recurrence after the first stores three FP16 vectors per layer: an
    input modulation, a recurrent-feedback gain, and an output accumulation
    gain. The base compact payload is charged exactly once per token/layer.
    """

    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
        raise ValueError("hidden_size must be an integer")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if isinstance(cycles, bool) or not isinstance(cycles, int):
        raise ValueError("cycles must be an integer")
    if cycles < 2:
        raise ValueError("recurrent compact layers require at least two cycles")
    if (
        isinstance(adapter_rank, bool)
        or not isinstance(adapter_rank, int)
        or adapter_rank < 0
    ):
        raise ValueError("adapter_rank must be a non-negative integer")
    if isinstance(cache_line_bytes, bool) or not isinstance(cache_line_bytes, int):
        raise ValueError("cache_line_bytes must be an integer")
    if cache_line_bytes <= 0:
        raise ValueError("cache_line_bytes must be positive")

    widths = [int(width) for width in layer_widths]
    if not widths:
        raise ValueError("at least one layer width is required")
    base = width_pruned_q4_traffic(
        hidden_size,
        source_intermediate_size,
        widths,
        cache_line_bytes=cache_line_bytes,
    )
    diagonal_values_per_layer = 3 * (cycles - 1) * hidden_size
    low_rank_values_per_layer = 2 * (cycles - 1) * hidden_size * adapter_rank
    adapter_values_per_layer = diagonal_values_per_layer + low_rank_values_per_layer
    adapter_payload_bytes_per_layer = adapter_values_per_layer * 2
    adapter_header_bytes_per_layer = cache_line_bytes
    adapter_block_bytes_per_layer = _align(
        adapter_header_bytes_per_layer + adapter_payload_bytes_per_layer,
        cache_line_bytes,
    )
    adapter_total_bytes = adapter_block_bytes_per_layer * len(widths)
    total_cold_bytes = int(base["total_cold_bytes"]) + adapter_total_bytes
    dense_q4_bytes = int(base["dense_q4_source_mlp_bytes"])
    fraction = total_cold_bytes / dense_q4_bytes
    return {
        "layout": "recurrent_compact_q4_diagonal_cycles_v1",
        "cycles": cycles,
        "adapter_rank": adapter_rank,
        "layer_count": len(widths),
        "layer_widths": widths,
        "cache_line_bytes": cache_line_bytes,
        "base_compact_q4_bytes": int(base["total_cold_bytes"]),
        "adapter_values_per_layer": adapter_values_per_layer,
        "diagonal_adapter_values_per_layer": diagonal_values_per_layer,
        "low_rank_adapter_values_per_layer": low_rank_values_per_layer,
        "adapter_fp16_payload_bytes_per_layer": adapter_payload_bytes_per_layer,
        "adapter_header_bytes_per_layer": adapter_header_bytes_per_layer,
        "adapter_cache_aligned_bytes_per_layer": adapter_block_bytes_per_layer,
        "adapter_total_cold_bytes": adapter_total_bytes,
        "total_cold_bytes": total_cold_bytes,
        "dense_q4_source_mlp_bytes": dense_q4_bytes,
        "fraction_of_dense_q4": fraction,
        "passes_45_percent_traffic_gate": fraction <= 0.45,
        "base_weight_read_policy": (
            "one cold compact-Q4 read per layer/token; later cycles reuse the "
            "identical decoded matrix payload from CPU cache"
        ),
        "native_cache_reuse_measured": False,
        "requires_native_cache_validation": True,
        "base": base,
    }


def recurrent_compact_mlp_class(torch: Any) -> type:
    """Create a Torch module class without importing Torch at package import."""

    functional = torch.nn.functional

    class RecurrentCompactSwiGLU(torch.nn.Module):
        """A tied compact SwiGLU with stable recurrent refinements."""

        def __init__(
            self,
            gate_weight: Any,
            up_weight: Any,
            down_weight: Any,
            *,
            cycles: int,
            adapter_rank: int = 0,
            epsilon: float = 1e-6,
        ) -> None:
            super().__init__()
            gate = torch.as_tensor(gate_weight).detach().clone().float()
            up = torch.as_tensor(up_weight).detach().clone().float()
            down = torch.as_tensor(down_weight).detach().clone().float()
            if gate.ndim != 2 or gate.shape != up.shape:
                raise ValueError("gate/up weights must be matching matrices")
            if down.shape != (gate.shape[1], gate.shape[0]):
                raise ValueError("down weight must have shape [hidden, width]")
            if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 2:
                raise ValueError("cycles must be an integer of at least two")
            if (
                isinstance(adapter_rank, bool)
                or not isinstance(adapter_rank, int)
                or adapter_rank < 0
            ):
                raise ValueError("adapter_rank must be a non-negative integer")
            if not np.isfinite(epsilon) or epsilon <= 0:
                raise ValueError("epsilon must be positive and finite")

            self.gate_weight = torch.nn.Parameter(gate)
            self.up_weight = torch.nn.Parameter(up)
            self.down_weight = torch.nn.Parameter(down)
            self.cycles = cycles
            self.adapter_rank = adapter_rank
            self.epsilon = float(epsilon)
            hidden = gate.shape[1]
            refinements = cycles - 1
            self.input_delta = torch.nn.Parameter(
                torch.zeros(refinements, hidden, dtype=gate.dtype)
            )
            self.feedback_gain = torch.nn.Parameter(
                torch.zeros(refinements, hidden, dtype=gate.dtype)
            )
            self.output_gain = torch.nn.Parameter(
                torch.zeros(refinements, hidden, dtype=gate.dtype)
            )
            if adapter_rank:
                self.input_adapter_down = torch.nn.Parameter(
                    torch.empty(
                        refinements,
                        adapter_rank,
                        hidden,
                        dtype=gate.dtype,
                    )
                )
                self.input_adapter_up = torch.nn.Parameter(
                    torch.zeros(
                        refinements,
                        hidden,
                        adapter_rank,
                        dtype=gate.dtype,
                    )
                )
                torch.nn.init.kaiming_uniform_(
                    self.input_adapter_down,
                    a=np.sqrt(5),
                )
            else:
                self.register_parameter("input_adapter_down", None)
                self.register_parameter("input_adapter_up", None)

        @property
        def hidden_size(self) -> int:
            return int(self.gate_weight.shape[1])

        @property
        def width(self) -> int:
            return int(self.gate_weight.shape[0])

        def compact(self, hidden: Any) -> Any:
            gate = functional.linear(hidden, self.gate_weight)
            up = functional.linear(hidden, self.up_weight)
            return functional.linear(functional.silu(gate) * up, self.down_weight)

        def _match_rms(self, value: Any, reference: Any) -> Any:
            value_rms = torch.sqrt(
                torch.mean(value.float().square(), dim=-1, keepdim=True) + self.epsilon
            )
            reference_rms = torch.sqrt(
                torch.mean(reference.float().square(), dim=-1, keepdim=True)
                + self.epsilon
            )
            return value * (reference_rms / value_rms).to(value.dtype)

        def forward(
            self,
            hidden: Any,
            *,
            return_cycle_outputs: bool = False,
        ) -> Any:
            if hidden.shape[-1] != self.hidden_size:
                raise ValueError("hidden input dimension does not match the MLP")
            base = self.compact(hidden)
            output = base
            previous = base
            cycle_outputs = [base]
            for cycle in range(self.cycles - 1):
                normalized_feedback = self._match_rms(previous, hidden)
                input_scale = 1.0 + 0.25 * torch.tanh(self.input_delta[cycle])
                recurrent_input = (
                    hidden * input_scale
                    + torch.tanh(self.feedback_gain[cycle]) * normalized_feedback
                )
                if self.adapter_rank:
                    compressed = functional.linear(
                        hidden,
                        self.input_adapter_down[cycle],
                    )
                    recurrent_input = (
                        recurrent_input
                        + functional.linear(
                            compressed,
                            self.input_adapter_up[cycle],
                        )
                        / self.adapter_rank
                    )
                previous = self.compact(recurrent_input)
                output = output + self.output_gain[cycle] * previous
                cycle_outputs.append(previous)
            if return_cycle_outputs:
                return output, tuple(cycle_outputs)
            return output

    return RecurrentCompactSwiGLU


__all__ = [
    "recurrent_compact_mlp_class",
    "recurrent_compact_q4_traffic",
]
