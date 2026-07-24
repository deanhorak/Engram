"""Persistent bounded attention modules for incremental native BitNet execution."""

from __future__ import annotations

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
            self._caches: list[NativeStreamingAttention] = []
            self._next_position = 0
            self._logical_read_bytes = 0
            self._maximum_state_bytes = 0
            self._maximum_scratch_bytes = 0

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
        def metrics(self) -> dict[str, int]:
            return {
                "tokens_seen": self._next_position,
                "logical_read_bytes": self._logical_read_bytes,
                "state_bytes": self._maximum_state_bytes,
                "scratch_bytes": self._maximum_scratch_bytes,
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
            query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            query, key = apply_rotary_pos_emb(
                query,
                key,
                *position_embeddings,
            )

            output_batches = []
            forward_state_bytes = 0
            forward_scratch_bytes = 0
            for batch_index, cache in enumerate(self._caches):
                rows = []
                cache_state_bytes = 0
                cache_scratch_bytes = 0
                for offset in range(length):
                    output, metrics = cache.step(
                        query[batch_index, :, offset].float().cpu().numpy(),
                        key[batch_index, :, offset].float().cpu().numpy(),
                        value[batch_index, :, offset].float().cpu().numpy(),
                    )
                    rows.append(torch.from_numpy(output))
                    self._logical_read_bytes += (
                        metrics.candidate_key_bytes
                        + metrics.selected_value_bytes
                        + metrics.local_kv_bytes
                    )
                    cache_state_bytes = max(
                        cache_state_bytes,
                        metrics.state_bytes,
                    )
                    cache_scratch_bytes = max(
                        cache_scratch_bytes,
                        metrics.scratch_bytes,
                    )
                output_batches.append(torch.stack(rows, dim=1))
                forward_state_bytes += cache_state_bytes
                forward_scratch_bytes += cache_scratch_bytes
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
            output = self.attn_sub_norm(output)
            return self.o_proj(output), None

    return NativeIncrementalBitNetAttention


def aggregate_native_attention_metrics(layers: list[Any]) -> dict[str, int]:
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
    }


__all__ = [
    "aggregate_native_attention_metrics",
    "native_incremental_attention_class",
]
