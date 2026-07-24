"""End-to-end long-context benchmark for a compiled native BitNet package."""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Sequence

from engram.runtime.native_bitnet import NativeBitNetRuntime
from engram.utils import atomic_json


def _fixed_length_context(tokens: Sequence[int], length: int) -> list[int]:
    if length <= 0:
        raise ValueError("context length must be positive")
    if not tokens:
        raise ValueError("benchmark prompt tokenized to an empty sequence")
    prefix = [int(tokens[0])]
    body = [int(value) for value in tokens[1:]] or prefix
    repeated = (body * ((length + len(body) - 1) // len(body)))[: max(0, length - 1)]
    return (prefix + repeated)[:length]


def benchmark_native_bitnet_generation(
    *,
    package: str | Path,
    out: str | Path,
    prompt: str,
    lengths: Sequence[int] = (33, 128, 256),
    max_new_tokens: int = 4,
    mlp_library: str | Path | None = None,
    attention_library: str | Path | None = None,
    threads: int | None = None,
    native_projections: bool = False,
    local_window: int = 16,
    older_candidates: int = 8,
    older_top_k: int = 4,
    sink_tokens: int = 2,
) -> dict:
    """Measure complete prefill and greedy decode with bounded attention."""

    checked_lengths = tuple(int(length) for length in lengths)
    if not checked_lengths or any(length <= 0 for length in checked_lengths):
        raise ValueError("benchmark lengths must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    results = []
    with NativeBitNetRuntime(
        package,
        library=mlp_library,
        threads=threads,
        native_projections=native_projections,
    ) as runtime:
        seed_tokens = runtime.encode(prompt)
        config = runtime.model.config
        query_heads = int(config.num_attention_heads)
        head_dimension = int(config.hidden_size // config.num_attention_heads)
        layers = int(config.num_hidden_layers)
        for context_length in checked_lengths:
            context = _fixed_length_context(seed_tokens, context_length)
            head_started = [0.0]
            head_elapsed = [0.0]

            def _head_pre_hook(_module, _inputs):
                head_started[0] = time.perf_counter()

            def _head_post_hook(_module, _inputs, _output):
                head_elapsed[0] += time.perf_counter() - head_started[0]

            pre_handle = runtime.model.lm_head.register_forward_pre_hook(
                _head_pre_hook
            )
            post_handle = runtime.model.lm_head.register_forward_hook(
                _head_post_hook
            )
            try:
                generated = runtime.generate_tokens_bounded(
                    context,
                    max_new_tokens=max_new_tokens,
                    attention_library=attention_library,
                    local_window=local_window,
                    older_candidates=older_candidates,
                    older_top_k=older_top_k,
                    sink_tokens=sink_tokens,
                )
            finally:
                pre_handle.remove()
                post_handle.remove()
            processed_positions = context_length + max_new_tokens - 1
            dense_logical_reads = (
                sum(range(1, processed_positions + 1))
                * layers
                * query_heads
                * head_dimension
                * 2
                * 4
            )
            results.append(
                {
                    "context_tokens": context_length,
                    "generated_tokens": list(generated.generated_tokens),
                    "generated_text": generated.text,
                    "processed_positions": processed_positions,
                    "elapsed_seconds": generated.elapsed_seconds,
                    "generated_tokens_per_second": (
                        max_new_tokens / generated.elapsed_seconds
                    ),
                    "processed_positions_per_second": (
                        processed_positions / generated.elapsed_seconds
                    ),
                    "mlp_calls": generated.mlp_calls,
                    "mlp_elapsed_seconds": generated.mlp_elapsed_seconds,
                    "scheduled_mlp_bytes": generated.scheduled_mlp_bytes,
                    "attention_logical_read_bytes": (
                        generated.attention_logical_read_bytes
                    ),
                    "dense_attention_logical_read_bytes": dense_logical_reads,
                    "attention_fraction_of_dense_logical_reads": (
                        generated.attention_logical_read_bytes / dense_logical_reads
                    ),
                    "attention_state_bytes": generated.attention_state_bytes,
                    "attention_scratch_bytes": generated.attention_scratch_bytes,
                    "qkv_projection_seconds": generated.qkv_projection_seconds,
                    "rope_seconds": generated.rope_seconds,
                    "native_attention_seconds": (
                        generated.native_attention_seconds
                    ),
                    "output_projection_seconds": (
                        generated.output_projection_seconds
                    ),
                    "native_attention_calls": generated.native_attention_calls,
                    "vocabulary_projection_seconds": head_elapsed[0],
                }
            )
    report = {
        "schema_version": 1,
        "experiment": "native_bitnet_end_to_end_long_context_generation",
        "package": str(Path(package).resolve()),
        "configuration": {
            "lengths": list(checked_lengths),
            "max_new_tokens": int(max_new_tokens),
            "local_window": int(local_window),
            "older_candidates": int(older_candidates),
            "older_top_k": int(older_top_k),
            "sink_tokens": int(sink_tokens),
            "threads": threads,
            "native_packed_attention_projections": bool(native_projections),
            "query_heads": query_heads,
            "head_dimension": head_dimension,
            "layers": layers,
            "attention_compute_dtype": "float32",
        },
        "results": results,
        "bounded_state_confirmed": (
            len({row["attention_state_bytes"] for row in results}) == 1
        ),
        "dense_hf_kv_cache_allocated": False,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "scope": (
            "Complete compiled-package prefill and greedy decode, including token "
            "embedding, all transformer layers, native packed MLP calls, bounded "
            "native attention, final normalization, vocabulary projection, and "
            "token selection. Logical reads are algorithmic float32 interface "
            "counts rather than measured DRAM transactions."
        ),
    }
    atomic_json(Path(out), report)
    return report


__all__ = ["benchmark_native_bitnet_generation"]
