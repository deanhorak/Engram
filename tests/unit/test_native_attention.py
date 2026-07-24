from pathlib import Path

import numpy as np
import pytest

from engram.runtime.native_attention import NativeStreamingAttention


class _Reference:
    def __init__(self, heads, kv_heads, width, window, candidates, top_k, sinks):
        self.heads = heads
        self.groups = heads // kv_heads
        self.width = width
        self.window = window
        self.candidates = candidates
        self.top_k = top_k
        self.sinks = sinks
        self.scale = width**-0.5
        self.local = []
        self.old = [dict() for _ in range(heads)]
        self.position = 0

    def step(self, query, key, value):
        if len(self.local) == self.window:
            position, old_key, old_value, mass = self.local.pop(0)
            for head in range(self.heads):
                if position < self.sinks:
                    slot = position
                else:
                    available = [
                        slot
                        for slot in range(self.sinks, self.candidates)
                        if slot not in self.old[head]
                    ]
                    if available:
                        slot = available[0]
                    else:
                        slot = min(
                            range(self.sinks, self.candidates),
                            key=lambda candidate: (
                                self.old[head][candidate][0],
                                self.old[head][candidate][1],
                            ),
                        )
                        if float(mass[head]) < self.old[head][slot][0]:
                            continue
                kv_head = head // self.groups
                self.old[head][slot] = (
                    float(mass[head]),
                    position,
                    old_key[kv_head].copy(),
                    old_value[kv_head].copy(),
                )
        self.local.append(
            (
                self.position,
                key.copy(),
                value.copy(),
                np.zeros(self.heads, dtype=np.float32),
            )
        )
        self.position += 1
        output = np.zeros_like(query)
        for head in range(self.heads):
            kv_head = head // self.groups
            candidate_slots = list(self.old[head])
            candidate_scores = {
                slot: float(query[head] @ self.old[head][slot][2] * self.scale)
                for slot in candidate_slots
            }
            selected = sorted(
                candidate_slots,
                key=lambda slot: (
                    -candidate_scores[slot],
                    self.old[head][slot][1],
                ),
            )[: self.top_k]
            scores = [
                float(query[head] @ entry[1][kv_head] * self.scale)
                for entry in self.local
            ]
            scores.extend(candidate_scores[slot] for slot in selected)
            weights = np.exp(np.asarray(scores, dtype=np.float32) - np.max(scores))
            weights /= weights.sum()
            offset = 0
            for entry in self.local:
                entry[3][head] += weights[offset]
                output[head] += weights[offset] * entry[2][kv_head]
                offset += 1
            for slot in selected:
                score, position, old_key, old_value = self.old[head][slot]
                self.old[head][slot] = (
                    score + float(weights[offset]),
                    position,
                    old_key,
                    old_value,
                )
                output[head] += weights[offset] * old_value
                offset += 1
        return output


def test_native_streaming_attention_matches_reference_and_stays_bounded():
    library = Path("build/libengram_attention.so")
    if not library.exists():
        pytest.skip("native attention library has not been built")
    heads, kv_heads, width = 4, 2, 7
    reference = _Reference(heads, kv_heads, width, 4, 5, 3, 2)
    generator = np.random.default_rng(20260724)
    with NativeStreamingAttention(
        query_heads=heads,
        key_value_heads=kv_heads,
        head_dimension=width,
        local_window=4,
        older_candidates=5,
        older_top_k=3,
        sink_tokens=2,
        library=library,
    ) as native:
        state_bytes = None
        for position in range(40):
            query = generator.normal(size=(heads, width)).astype(np.float32)
            key = generator.normal(size=(kv_heads, width)).astype(np.float32)
            value = generator.normal(size=(kv_heads, width)).astype(np.float32)
            expected = reference.step(query, key, value)
            actual, metrics = native.step(query, key, value)
            np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
            state_bytes = metrics.state_bytes if state_bytes is None else state_bytes
            assert metrics.state_bytes == state_bytes
            assert metrics.local_entries <= 4
            assert metrics.active_older_entries <= heads * 5
            assert metrics.tokens_seen == position + 1
        native.reset()
        query = np.zeros((heads, width), dtype=np.float32)
        key = np.zeros((kv_heads, width), dtype=np.float32)
        value = np.ones((kv_heads, width), dtype=np.float32)
        output, metrics = native.step(query, key, value)
        np.testing.assert_array_equal(output, np.ones_like(output))
        assert metrics.tokens_seen == 1


def test_native_stream_call_matches_individual_steps_and_aggregates_traffic():
    library = Path("build/libengram_attention.so")
    if not library.exists():
        pytest.skip("native attention library has not been built")
    heads, kv_heads, width, length = 4, 2, 7, 23
    generator = np.random.default_rng(43)
    queries = generator.normal(size=(length, heads, width)).astype(np.float32)
    keys = generator.normal(size=(length, kv_heads, width)).astype(np.float32)
    values = generator.normal(size=(length, kv_heads, width)).astype(np.float32)
    configuration = {
        "query_heads": heads,
        "key_value_heads": kv_heads,
        "head_dimension": width,
        "local_window": 4,
        "older_candidates": 5,
        "older_top_k": 3,
        "sink_tokens": 2,
        "library": library,
    }
    with (
        NativeStreamingAttention(**configuration) as streamed,
        NativeStreamingAttention(**configuration) as stepped,
    ):
        actual, stream_metrics = streamed.stream(queries, keys, values)
        rows = []
        traffic = {
            "candidate_key_bytes": 0,
            "selected_value_bytes": 0,
            "local_kv_bytes": 0,
        }
        for position in range(length):
            row, step_metrics = stepped.step(
                queries[position],
                keys[position],
                values[position],
            )
            rows.append(row)
            for name in traffic:
                traffic[name] += getattr(step_metrics, name)

    np.testing.assert_array_equal(actual, np.stack(rows))
    assert stream_metrics.tokens_seen == length
    assert stream_metrics.candidate_key_bytes == traffic["candidate_key_bytes"]
    assert stream_metrics.selected_value_bytes == traffic["selected_value_bytes"]
    assert stream_metrics.local_kv_bytes == traffic["local_kv_bytes"]
    assert stream_metrics.state_bytes == step_metrics.state_bytes
