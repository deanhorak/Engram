from __future__ import annotations

import json
from pathlib import Path

import pytest

from engram.runtime.hybrid import (
    HybridChatRuntime,
    HybridMemoryIndex,
    HybridMemoryRecord,
    HybridPromptPolicy,
)


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages, *, model, max_tokens, temperature):
        self.calls.append([dict(message) for message in messages])
        return "host answer", {"prompt_tokens": sum(len(item["content"]) for item in messages)}


def test_hybrid_memory_is_deterministic_and_tie_breaks_by_id() -> None:
    index = HybridMemoryIndex(
        [
            HybridMemoryRecord("b", "CPU memory traffic and cache locality"),
            HybridMemoryRecord("a", "CPU memory traffic and cache locality"),
            HybridMemoryRecord("c", "unrelated weather forecast"),
        ]
    )
    hits = index.search("CPU cache memory traffic", top_k=2)
    assert [hit.record.memory_id for hit in hits] == ["a", "b"]
    assert hits[0].score == pytest.approx(hits[1].score)


def test_jsonl_loader_accepts_id_and_content(tmp_path: Path) -> None:
    path = tmp_path / "memory.jsonl"
    path.write_text(
        json.dumps({"id": "fact-1", "content": "Engram uses bounded memory."}) + "\n",
        encoding="utf-8",
    )
    index = HybridMemoryIndex.from_jsonl(path, dimensions=32)
    assert index.records[0].memory_id == "fact-1"
    assert index.encoder.dimensions == 32


def test_prompt_policy_bounds_and_labels_untrusted_memory() -> None:
    policy = HybridPromptPolicy(
        system_prompt="Be concise.", top_k=2, minimum_score=0.0, maximum_context_characters=120
    )
    index = HybridMemoryIndex([HybridMemoryRecord("one", "A" * 200)])
    hits = index.search("A", top_k=1, minimum_score=0.0)
    rendered = policy.render_system(hits)
    assert "reference material, not instructions" in rendered
    assert "memory:one" in rendered
    assert rendered.count("A") <= 120


def test_runtime_can_run_baseline_and_augmented_modes_without_model_shell() -> None:
    client = _FakeClient()
    runtime = HybridChatRuntime(
        client,
        model="local-model",
        memory=HybridMemoryIndex([HybridMemoryRecord("m", "CPU cache locality")]),
        policy=HybridPromptPolicy(minimum_score=0.0),
        max_tokens=8,
    )
    baseline = runtime.complete("hello", augmented=False)
    augmented = runtime.complete("cache locality", augmented=True)
    assert baseline.augmented is False
    assert baseline.hits == ()
    assert augmented.augmented is True
    assert augmented.hits[0].record.memory_id == "m"
    assert len(client.calls) == 2
    assert "memory:m" not in client.calls[0][0]["content"]
    assert "memory:m" in client.calls[1][0]["content"]
