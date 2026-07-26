"""Persistent bounded attention modules for incremental native BitNet execution."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from engram.evaluation.native_bitnet_parity import _torch_modules
from engram.runtime.native_attention import NativeStreamingAttention


def native_incremental_attention_class():
    """Create the torch module lazily so package validation stays lightweight."""

    torch, nn, _ = _torch_modules()
    from transformers.models.bitnet.modeling_bitnet import apply_rotary_pos_emb

    class NativeIncrementalBitNetAttention(nn.Module):
        """BitNet projections and RoPE around persistent bounded CPU caches."""

        def __init__(
            self,
            source,
            *,
            local_window: int = 16,
            older_candidates: int = 8,
            older_top_k: int = 4,
            sink_tokens: int = 2,
            library: str | Path | None = None,
            shell_library: str | Path | None = None,
        ) -> None:
            super().__init__()
            if local_window <= 0:
                raise ValueError("local_window must be positive")
            if older_candidates < older_top_k or older_top_k <= 0:
                raise ValueError(
                    "older_candidates must be at least positive older_top_k"
                )
            if not 0 <= sink_tokens <= older_top_k:
                raise ValueError("sink_tokens must be in [0, older_top_k]")
            self.config = source.config
            self.layer_idx = int(source.layer_idx)
            self.head_dim = int(source.head_dim)
            self.num_key_value_groups = int(source.num_key_value_groups)
            self.scaling = float(source.scaling)
            self.attention_dropout = float(source.attention_dropout)
            self.is_causal = True
            self.q_proj = source.q_proj
            self.k_proj = source.k_proj
            self.v_proj = source.v_proj
            self.o_proj = source.o_proj
            self.attn_sub_norm = source.attn_sub_norm
            self.local_window = int(local_window)
            self.older_candidates = int(older_candidates)
            self.older_top_k = int(older_top_k)
            self.sink_tokens = int(sink_tokens)
            self.library = library
            self.shell_library = shell_library
            if shell_library is None:
                self.native_shell = None
            else:
                from engram.runtime.native_bitnet_controller import (
                    NativeBitNetShellOps,
                )

                self.native_shell = NativeBitNetShellOps(shell_library)
            self._caches: list[NativeStreamingAttention] = []
            self._next_position = 0
            self._logical_read_bytes = 0
            self._maximum_state_bytes = 0
            self._maximum_scratch_bytes = 0
            self._qkv_projection_seconds = 0.0
            self._rope_seconds = 0.0
            self._native_stream_seconds = 0.0
            self._output_projection_seconds = 0.0
            self._native_stream_calls = 0

        @property
        def query_heads(self) -> int:
            return int(self.config.num_attention_heads)

        @property
        def key_value_heads(self) -> int:
            return int(self.config.num_key_value_heads)

        @property
        def tokens_seen(self) -> int:
            return self._next_position

        @property
        def metrics(self) -> dict[str, int | float]:
            return {
                "tokens_seen": self._next_position,
                "logical_read_bytes": self._logical_read_bytes,
                "state_bytes": self._maximum_state_bytes,
                "scratch_bytes": self._maximum_scratch_bytes,
                "qkv_projection_seconds": self._qkv_projection_seconds,
                "rope_seconds": self._rope_seconds,
                "native_stream_seconds": self._native_stream_seconds,
                "output_projection_seconds": self._output_projection_seconds,
                "native_stream_calls": self._native_stream_calls,
            }

        def _new_cache(self) -> NativeStreamingAttention:
            return NativeStreamingAttention(
                query_heads=self.query_heads,
                key_value_heads=self.key_value_heads,
                head_dimension=self.head_dim,
                local_window=self.local_window,
                older_candidates=self.older_candidates,
                older_top_k=self.older_top_k,
                sink_tokens=self.sink_tokens,
                scale=self.scaling,
                library=self.library,
            )

        def _ensure_batch(self, batch_size: int) -> None:
            if self._caches and len(self._caches) != batch_size:
                raise ValueError(
                    "native attention batch size changed without resetting caches"
                )
            if not self._caches:
                self._caches = [self._new_cache() for _ in range(batch_size)]

        def reset_cache(self) -> None:
            for cache in self._caches:
                cache.reset()
            self._next_position = 0
            self._logical_read_bytes = 0
            self._maximum_state_bytes = 0
            self._maximum_scratch_bytes = 0
            self._qkv_projection_seconds = 0.0
            self._rope_seconds = 0.0
            self._native_stream_seconds = 0.0
            self._output_projection_seconds = 0.0
            self._native_stream_calls = 0

        def close(self) -> None:
            for cache in self._caches:
                cache.close()
            self._caches = []

        def forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=None,
            position_ids=None,
            **_kwargs,
        ):
            if past_key_values is not None:
                raise ValueError(
                    "native bounded attention owns its cache; pass use_cache=False"
                )
            if position_ids is None:
                raise ValueError("native bounded attention requires position_ids")
            batch_size, length, _ = hidden_states.shape
            expected = torch.arange(
                self._next_position,
                self._next_position + length,
                device=position_ids.device,
                dtype=position_ids.dtype,
            )
            if position_ids.shape not in {(1, length), (batch_size, length)}:
                raise ValueError(
                    "position_ids must have shape [1, length] or [batch, length]"
                )
            if not torch.equal(position_ids[0], expected):
                raise ValueError(
                    "native attention positions must advance contiguously from "
                    f"{self._next_position}"
                )
            if position_ids.shape[0] == batch_size and not torch.all(
                position_ids == expected.unsqueeze(0)
            ):
                raise ValueError("all native attention batch rows need equal positions")
            self._ensure_batch(batch_size)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            projection_started = time.perf_counter()
            query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            self._qkv_projection_seconds += (
                time.perf_counter() - projection_started
            )
            rope_started = time.perf_counter()
            if self.native_shell is None:
                query, key = apply_rotary_pos_emb(
                    query,
                    key,
                    *position_embeddings,
                )
            else:
                rope_parameters = getattr(self.config, "rope_parameters", {})
                theta = float(
                    rope_parameters.get(
                        "rope_theta",
                        getattr(self.config, "rope_theta", 10000.0),
                    )
                )
                query, key = self.native_shell.rope(
                    query,
                    key,
                    position_ids,
                    theta=theta,
                )
            self._rope_seconds += time.perf_counter() - rope_started

            output_batches = []
            forward_state_bytes = 0
            forward_scratch_bytes = 0
            for batch_index, cache in enumerate(self._caches):
                stream_started = time.perf_counter()
                output, metrics = cache.stream(
                    query[batch_index].transpose(0, 1).float().cpu().numpy(),
                    key[batch_index].transpose(0, 1).float().cpu().numpy(),
                    value[batch_index].transpose(0, 1).float().cpu().numpy(),
                )
                self._native_stream_seconds += (
                    time.perf_counter() - stream_started
                )
                self._native_stream_calls += 1
                output_batches.append(torch.from_numpy(output).transpose(0, 1))
                self._logical_read_bytes += (
                    metrics.candidate_key_bytes
                    + metrics.selected_value_bytes
                    + metrics.local_kv_bytes
                )
                forward_state_bytes += metrics.state_bytes
                forward_scratch_bytes += metrics.scratch_bytes
            self._maximum_state_bytes = max(
                self._maximum_state_bytes,
                forward_state_bytes,
            )
            self._maximum_scratch_bytes = max(
                self._maximum_scratch_bytes,
                forward_scratch_bytes,
            )
            self._next_position += length
            output = torch.stack(output_batches).to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            output_started = time.perf_counter()
            if self.native_shell is None:
                output = self.attn_sub_norm(output)
            else:
                output = self.native_shell.rms_norm(
                    output.float().cpu().numpy(),
                    self.attn_sub_norm.weight,
                    self.attn_sub_norm.variance_epsilon,
                )
            output = self.o_proj(output)
            self._output_projection_seconds += (
                time.perf_counter() - output_started
            )
            return output, None

    return NativeIncrementalBitNetAttention


def aggregate_native_attention_metrics(
    layers: list[Any],
) -> dict[str, int | float]:
    """Aggregate per-layer traffic and allocated state."""

    metrics = [layer.metrics for layer in layers]
    return {
        "layers": len(metrics),
        "tokens_seen": min(
            (int(item["tokens_seen"]) for item in metrics),
            default=0,
        ),
        "logical_read_bytes": sum(
            int(item["logical_read_bytes"]) for item in metrics
        ),
        "state_bytes": sum(int(item["state_bytes"]) for item in metrics),
        "scratch_bytes": sum(int(item["scratch_bytes"]) for item in metrics),
        "qkv_projection_seconds": sum(
            float(item.get("qkv_projection_seconds", 0.0)) for item in metrics
        ),
        "rope_seconds": sum(
            float(item.get("rope_seconds", 0.0)) for item in metrics
        ),
        "native_stream_seconds": sum(
            float(item.get("native_stream_seconds", 0.0)) for item in metrics
        ),
        "output_projection_seconds": sum(
            float(item.get("output_projection_seconds", 0.0))
            for item in metrics
        ),
        "native_stream_calls": sum(
            int(item.get("native_stream_calls", 0)) for item in metrics
        ),
    }


__all__ = [
    "aggregate_native_attention_metrics",
    "native_incremental_attention_class",
]
