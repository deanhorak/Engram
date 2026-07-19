from __future__ import annotations

import time
from typing import Any

import numpy as np

from engram.episodic.hybrid import HybridEpisodicMemory
from engram.episodic.local_attention import causal_local_attention
from engram.episodic.recurrent import normalized_recurrent_attention
from engram.utils import percentile


def _relative_rows(approximation: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.linalg.norm(approximation - reference, axis=1) / np.maximum(
        np.linalg.norm(reference, axis=1), 1e-12
    )


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": percentile(values.tolist(), 95),
    }


def _full_causal_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    output = []
    scale = np.sqrt(q.shape[1])
    for position in range(len(q)):
        scores = k[: position + 1] @ q[position] / scale
        weights = np.exp(scores - np.max(scores))
        weights /= np.sum(weights)
        output.append(weights @ v[: position + 1])
    return np.asarray(output)


def evaluate_attention_replacement(
    *,
    seed: int = 41,
    length: int = 128,
    key_width: int = 16,
    value_width: int = 16,
    local_window: int = 16,
) -> dict[str, Any]:
    """Instrument Gate 3 on deterministic synthetic attention states."""
    rng = np.random.default_rng(seed)
    keys = rng.normal(size=(length, key_width)).astype(np.float32)
    queries = (keys + 0.05 * rng.normal(size=keys.shape)).astype(np.float32)
    values = rng.normal(size=(length, value_width)).astype(np.float32)
    full = _full_causal_attention(queries, keys, values)

    started = time.perf_counter_ns()
    local = causal_local_attention(queries, keys, values, window=local_window)
    local_ns = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    recurrent = normalized_recurrent_attention(queries, keys, values, decay=0.99, dtype=np.float32)
    recurrent_ns = time.perf_counter_ns() - started
    hybrid_memory = HybridEpisodicMemory(
        key_width,
        value_width,
        local_window=local_window,
        retrieval_capacity=length,
        retrieval_candidates=min(16, length),
        retrieval_top_k=4,
        decay=0.99,
        older_weight=0.5,
    )
    hybrid_rows = []
    retrieval_hits = []
    bytes_read = []
    state_bytes = []
    started = time.perf_counter_ns()
    for position in range(length):
        read = hybrid_memory.step(queries[position], keys[position], values[position])
        hybrid_rows.append(read.output)
        bytes_read.append(read.bytes_read)
        state_bytes.append(read.state_bytes)
        if position >= local_window:
            older_scores = keys[: position - local_window + 1] @ queries[position]
            teacher_position = int(np.argmax(older_scores))
            retrieved = hybrid_memory.store.retrieve(queries[position], top_k=4)
            retrieval_hits.append(float(teacher_position in set(retrieved.positions.tolist())))
    hybrid_ns = time.perf_counter_ns() - started
    hybrid = np.asarray(hybrid_rows)

    # A controlled long-range copying case: the final query exactly repeats token zero.
    copying_store = HybridEpisodicMemory(
        key_width, value_width, local_window=4, retrieval_capacity=length, retrieval_candidates=length, retrieval_top_k=1
    )
    for position in range(length - 1):
        copying_store.step(keys[position], keys[position], values[position])
    copy_result = copying_store.store.retrieve(keys[0], top_k=1, candidate_count=length)
    copying_accuracy = float(len(copy_result.positions) == 1 and copy_result.positions[0] == 0)
    return {
        "schema_version": 1,
        "experiment": "gate_3_attention_replacement",
        "status": "synthetic_pipeline_validation",
        "configuration": {
            "seed": seed,
            "length": length,
            "key_width": key_width,
            "value_width": value_width,
            "local_window": local_window,
        },
        "relative_l2": {
            "local": _stats(_relative_rows(local, full)),
            "recurrent": _stats(_relative_rows(recurrent, full)),
            "hybrid": _stats(_relative_rows(hybrid, full)),
        },
        "retrieval_head_recall": _stats(np.asarray(retrieval_hits)),
        "copying_accuracy": copying_accuracy,
        "long_context_retrieval_accuracy": copying_accuracy,
        "memory": {
            "peak_state_bytes": max(state_bytes),
            "state_bytes_at_end": state_bytes[-1],
            "growth_model": "bounded_by_dimensions_local_window_and_configured_retrieval_capacity",
        },
        "latency_ns_per_token": {
            "local": local_ns / length,
            "recurrent": recurrent_ns / length,
            "hybrid": hybrid_ns / length,
        },
        "mean_retrieval_bytes_read": float(np.mean(bytes_read)),
        "teacher_attention_traces": {"status": "not_run"},
    }
