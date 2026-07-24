"""Long-context benchmark for the native bounded streaming-attention cache."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np

from engram.runtime.native_attention import NativeStreamingAttention
from engram.utils import atomic_json


def benchmark_native_streaming_attention(
    *,
    out: str | Path,
    library: str | Path | None = None,
    lengths: Sequence[int] = (33, 128, 512, 2048),
    query_heads: int = 20,
    key_value_heads: int = 5,
    head_dimension: int = 128,
    local_window: int = 16,
    older_candidates: int = 8,
    older_top_k: int = 4,
    sink_tokens: int = 2,
    seed: int = 20260724,
) -> dict:
    """Benchmark fixed-state native attention against dense logical reads."""

    checked_lengths = tuple(int(length) for length in lengths)
    if not checked_lengths or any(length <= 0 for length in checked_lengths):
        raise ValueError("benchmark lengths must be positive")
    generator = np.random.default_rng(seed)
    results = []
    with NativeStreamingAttention(
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dimension=head_dimension,
        local_window=local_window,
        older_candidates=older_candidates,
        older_top_k=older_top_k,
        sink_tokens=sink_tokens,
        library=library,
    ) as attention:
        expected_state_bytes = None
        for length in checked_lengths:
            attention.reset()
            bounded_bytes = 0
            started = time.perf_counter()
            for _ in range(length):
                query = generator.standard_normal(
                    (query_heads, head_dimension), dtype=np.float32
                )
                key = generator.standard_normal(
                    (key_value_heads, head_dimension), dtype=np.float32
                )
                value = generator.standard_normal(
                    (key_value_heads, head_dimension), dtype=np.float32
                )
                _, metrics = attention.step(query, key, value)
                bounded_bytes += (
                    metrics.local_kv_bytes
                    + metrics.candidate_key_bytes
                    + metrics.selected_value_bytes
                )
            elapsed = time.perf_counter() - started
            expected_state_bytes = (
                metrics.state_bytes
                if expected_state_bytes is None
                else expected_state_bytes
            )
            if metrics.state_bytes != expected_state_bytes:
                raise RuntimeError("native attention state grew with context")
            dense_bytes = (
                sum(range(1, length + 1))
                * query_heads
                * head_dimension
                * 2
                * np.dtype(np.float32).itemsize
            )
            results.append(
                {
                    "length": length,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": length / elapsed,
                    "bounded_logical_read_bytes": bounded_bytes,
                    "dense_logical_read_bytes": dense_bytes,
                    "fraction_of_dense_logical_reads": bounded_bytes / dense_bytes,
                    "state_bytes": metrics.state_bytes,
                    "scratch_bytes": metrics.scratch_bytes,
                    "active_older_entries": metrics.active_older_entries,
                }
            )
    report = {
        "schema_version": 1,
        "experiment": "native_streaming_attention_long_context",
        "configuration": {
            "query_heads": query_heads,
            "key_value_heads": key_value_heads,
            "head_dimension": head_dimension,
            "local_window": local_window,
            "older_candidates": older_candidates,
            "older_top_k": older_top_k,
            "sink_tokens": sink_tokens,
            "dtype": "float32",
            "seed": seed,
        },
        "results": results,
        "bounded_state_confirmed": len({row["state_bytes"] for row in results}) == 1,
        "hardware_dram_counter_measured": False,
        "scope_caveat": (
            "Logical reads are counted at the query-head kernel interface. "
            "Elapsed time includes ctypes and deterministic input generation; "
            "this is not yet an end-to-end transformer benchmark."
        ),
    }
    atomic_json(Path(out), report)
    return report


__all__ = ["benchmark_native_streaming_attention"]
