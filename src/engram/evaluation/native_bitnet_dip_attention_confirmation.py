"""Sustained bounded-attention confirmation through the native DIP handle."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from engram.compiler.native_bitnet import (
    NATIVE_BITNET_DIP_DERIVED_MANIFEST_SHA256,
)
from engram.runtime.native_bitnet_dip_token import NativeBitNetDIPTokenRuntime
from engram.utils import atomic_json, sha256_file

_STRUCTURAL_METRICS = (
    "positions_processed",
    "stage_calls",
    "semantic_calls",
    "semantic_rows",
    "semantic_selected_records",
    "semantic_kernel_cache_line_bytes",
    "semantic_global_metadata_bytes",
    "semantic_cache_line_bytes",
    "semantic_maximum_scratch_bytes",
    "attention_logical_read_bytes",
    "attention_state_bytes",
    "attention_scratch_bytes",
    "attention_eviction_events",
    "attention_older_candidate_entries_scored",
    "attention_older_selected_entries",
    "attention_sink_insertions",
    "attention_heavy_hitter_updates",
    "stopped_on_eos",
)


def _positive_lengths(lengths: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for value in lengths:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("attention confirmation lengths must be positive")
        if value in result:
            raise ValueError("attention confirmation lengths must be unique")
        result.append(value)
    if not result:
        raise ValueError("attention confirmation needs at least one length")
    return tuple(sorted(result))


def _expected_counters(
    *,
    prompt_length: int,
    layers: int,
    query_heads: int,
    local_window: int,
    older_candidates: int,
    older_top_k: int,
    sink_tokens: int,
) -> dict[str, int]:
    evictions = max(prompt_length - local_window, 0)
    candidate_entries = sum(
        min(index, older_candidates) for index in range(1, evictions + 1)
    )
    selected_entries = sum(
        min(index, older_candidates, older_top_k)
        for index in range(1, evictions + 1)
    )
    scale = layers * query_heads
    heavy_opportunities = max(evictions - sink_tokens, 0)
    guaranteed_heavy_updates = min(
        heavy_opportunities,
        max(older_candidates - sink_tokens, 0),
    )
    return {
        "attention_eviction_events": layers * evictions,
        "attention_older_candidate_entries_scored": scale * candidate_entries,
        "attention_older_selected_entries": scale * selected_entries,
        "attention_sink_insertions": scale * min(evictions, sink_tokens),
        "attention_heavy_hitter_updates_minimum": (
            scale * guaranteed_heavy_updates
        ),
        "attention_heavy_hitter_updates_maximum": (
            scale * heavy_opportunities
        ),
    }


def _prompt_tokens(runtime: NativeBitNetDIPTokenRuntime, prompt: str, size: int):
    seed = runtime.encode((prompt.rstrip() + " ") * (size + 1))
    if len(seed) < size:
        raise ValueError("attention confirmation prompt did not yield enough tokens")
    return seed[:size]


def evaluate_native_bitnet_dip_attention_confirmation(
    *,
    package: str | Path,
    library: str | Path,
    out: str | Path,
    lengths: Sequence[int] = (16, 17, 18, 24, 32),
    prompt: str = "The memory system should preserve relevant earlier context.",
    threads: int | None = None,
) -> dict[str, Any]:
    """Exercise local eviction, sinks, and heavy hitters at fixed boundaries."""

    checked_lengths = _positive_lengths(lengths)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("attention confirmation prompt must be non-empty")
    package_path = Path(package)
    library_path = Path(library)
    results: list[dict[str, Any]] = []
    with NativeBitNetDIPTokenRuntime(
        package_path,
        library=library_path,
        threads=threads,
    ) as runtime:
        if checked_lengths[-1] > runtime.max_position_embeddings:
            raise ValueError("attention confirmation exceeds package context")
        state_bytes: int | None = None
        longest_tokens: list[int] | None = None
        longest_generated: tuple[int, ...] | None = None
        longest_structural: dict[str, int] | None = None
        checks = []
        for length in checked_lengths:
            tokens = _prompt_tokens(runtime, prompt, length)
            generation = runtime.generate_tokens(tokens, max_new_tokens=1)
            metrics = runtime.last_metrics
            expected = _expected_counters(
                prompt_length=length,
                layers=runtime.layer_count,
                query_heads=runtime.query_heads,
                local_window=runtime.local_window,
                older_candidates=runtime.older_candidates,
                older_top_k=runtime.older_top_k,
                sink_tokens=runtime.sink_tokens,
            )
            exact_names = (
                "attention_eviction_events",
                "attention_older_candidate_entries_scored",
                "attention_older_selected_entries",
                "attention_sink_insertions",
            )
            exact = all(metrics[name] == expected[name] for name in exact_names)
            heavy = metrics["attention_heavy_hitter_updates"]
            heavy_bounded = (
                expected["attention_heavy_hitter_updates_minimum"]
                <= heavy
                <= expected["attention_heavy_hitter_updates_maximum"]
            )
            position_ok = (
                metrics["positions_processed"] == length
                and metrics["stage_calls"] == runtime.layer_count
                and metrics["semantic_calls"] == runtime.layer_count
                and metrics["semantic_rows"] == length * runtime.layer_count
            )
            if state_bytes is None:
                state_bytes = metrics["attention_state_bytes"]
            state_bounded = metrics["attention_state_bytes"] == state_bytes
            passed = exact and heavy_bounded and position_ok and state_bounded
            checks.append(passed)
            structural = {
                name: metrics[name] for name in _STRUCTURAL_METRICS
            }
            results.append(
                {
                    "prompt_tokens": length,
                    "generated_token_ids": list(generation.generated_tokens),
                    "generated_text": generation.text,
                    "elapsed_seconds": generation.elapsed_seconds,
                    "expected_attention_counters": expected,
                    "structural_metrics": structural,
                    "checks": {
                        "exact_eviction_candidate_selection_and_sink_counts": exact,
                        "heavy_hitter_updates_within_policy_bounds": heavy_bounded,
                        "positions_and_layer_calls": position_ok,
                        "attention_state_bytes_constant": state_bounded,
                    },
                    "passed": passed,
                }
            )
            if length == checked_lengths[-1]:
                longest_tokens = tokens
                longest_generated = generation.generated_tokens
                longest_structural = structural

        assert longest_tokens is not None
        assert longest_generated is not None
        assert longest_structural is not None
        replay = runtime.generate_tokens(longest_tokens, max_new_tokens=1)
        replay_structural = {
            name: runtime.last_metrics[name] for name in _STRUCTURAL_METRICS
        }
        replay_passed = (
            replay.generated_tokens == longest_generated
            and replay_structural == longest_structural
        )
        exercised = {
            "local_window_eviction": any(
                item["structural_metrics"]["attention_eviction_events"] > 0
                for item in results
            ),
            "older_candidate_scoring": any(
                item["structural_metrics"][
                    "attention_older_candidate_entries_scored"
                ]
                > 0
                for item in results
            ),
            "older_value_selection": any(
                item["structural_metrics"]["attention_older_selected_entries"]
                > 0
                for item in results
            ),
            "sink_insertion": any(
                item["structural_metrics"]["attention_sink_insertions"] > 0
                for item in results
            ),
            "heavy_hitter_update": any(
                item["structural_metrics"]["attention_heavy_hitter_updates"] > 0
                for item in results
            ),
        }
        passed = all(checks) and replay_passed and all(exercised.values())
        report: dict[str, Any] = {
            "schema_version": 1,
            "experiment": "native_bitnet_dip_bounded_attention_confirmation",
            "status": "passed" if passed else "failed",
            "passed": passed,
            "artifacts": {
                "package_manifest_sha256": (
                    NATIVE_BITNET_DIP_DERIVED_MANIFEST_SHA256
                ),
                "native_token_library_sha256": sha256_file(library_path),
            },
            "configuration": {
                "lengths": list(checked_lengths),
                "threads": runtime.thread_count,
                "layers": runtime.layer_count,
                "query_heads": runtime.query_heads,
                "local_window": runtime.local_window,
                "older_candidates": runtime.older_candidates,
                "older_top_k": runtime.older_top_k,
                "sink_tokens": runtime.sink_tokens,
                "max_new_tokens": 1,
                "cpu_only": True,
            },
            "results": results,
            "exercise_checks": exercised,
            "longest_reset_replay": {
                "passed": replay_passed,
                "generated_token_ids": list(replay.generated_tokens),
                "structural_metrics": replay_structural,
            },
            "limitations": [
                "This confirms native cache mechanics and deterministic replay, not attention quality against a dense teacher.",
                "Logical counters are algorithmic events, not hardware DRAM measurements.",
                "The fixed prompt is a boundary workload, not a language-quality benchmark.",
            ],
        }
    atomic_json(Path(out), report)
    return report


__all__ = ["evaluate_native_bitnet_dip_attention_confirmation"]
