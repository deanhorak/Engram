from __future__ import annotations

import pytest

from engram.evaluation.native_bitnet_dip_attention_confirmation import (
    _expected_counters,
    _positive_lengths,
)


def _expected(length: int):
    return _expected_counters(
        prompt_length=length,
        layers=30,
        query_heads=20,
        local_window=16,
        older_candidates=8,
        older_top_k=4,
        sink_tokens=2,
    )


def test_attention_counter_expectations_cover_policy_boundaries():
    assert _expected(16) == {
        "attention_eviction_events": 0,
        "attention_older_candidate_entries_scored": 0,
        "attention_older_selected_entries": 0,
        "attention_sink_insertions": 0,
        "attention_heavy_hitter_updates_minimum": 0,
        "attention_heavy_hitter_updates_maximum": 0,
    }
    assert _expected(17) == {
        "attention_eviction_events": 30,
        "attention_older_candidate_entries_scored": 600,
        "attention_older_selected_entries": 600,
        "attention_sink_insertions": 600,
        "attention_heavy_hitter_updates_minimum": 0,
        "attention_heavy_hitter_updates_maximum": 0,
    }
    assert _expected(24) == {
        "attention_eviction_events": 240,
        "attention_older_candidate_entries_scored": 21_600,
        "attention_older_selected_entries": 15_600,
        "attention_sink_insertions": 1_200,
        "attention_heavy_hitter_updates_minimum": 3_600,
        "attention_heavy_hitter_updates_maximum": 3_600,
    }
    assert _expected(32) == {
        "attention_eviction_events": 480,
        "attention_older_candidate_entries_scored": 60_000,
        "attention_older_selected_entries": 34_800,
        "attention_sink_insertions": 1_200,
        "attention_heavy_hitter_updates_minimum": 3_600,
        "attention_heavy_hitter_updates_maximum": 8_400,
    }


def test_attention_confirmation_lengths_are_sorted_and_validated():
    assert _positive_lengths([32, 16, 24]) == (16, 24, 32)
    with pytest.raises(ValueError, match="unique"):
        _positive_lengths([16, 16])
    with pytest.raises(ValueError, match="positive"):
        _positive_lengths([0])
    with pytest.raises(ValueError, match="at least one"):
        _positive_lengths([])
