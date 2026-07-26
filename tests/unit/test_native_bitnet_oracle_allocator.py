import numpy as np
import pytest

from engram.evaluation.native_bitnet_oracle_allocator import (
    allocate_layer_schedule,
    robust_boundary_risk,
)


def test_robust_boundary_risk_includes_token_and_sequence_tails():
    risk = robust_boundary_risk(
        np.asarray([0.1, 0.2, 0.3, 0.8], dtype=np.float64),
        np.asarray([0, 0, 1, 1], dtype=np.int64),
    )

    assert risk["sequence_count"] == 2
    assert risk["components"]["token_mean"] == pytest.approx(0.35)
    assert risk["components"]["worst_sequence"] == pytest.approx(0.55)
    assert risk["score"] > risk["components"]["token_mean"]


def test_allocator_spends_limited_extra_records_on_sensitive_layer():
    # At the minimum 2-record arm both layers fit in a 6-record budget.
    # Only one can receive the 4-record arm, and layer zero benefits more.
    allocation = allocate_layer_schedule(
        np.asarray(
            [
                [1.0, 0.1],
                [0.5, 0.4],
            ],
            dtype=np.float64,
        ),
        (2, 4),
        intermediate_size=12,
        mean_budget=0.25,
    )

    assert allocation["layer_top_ks"] == [4, 2]
    assert allocation["record_budget_used"] == 6
    assert allocation["record_budget_available"] == 6
    assert allocation["mean_active_fraction"] == pytest.approx(0.25)


def test_allocator_rejects_impossible_or_malformed_budgets():
    with pytest.raises(ValueError, match="minimum arm"):
        allocate_layer_schedule(
            np.ones((2, 2), dtype=np.float64),
            (4, 6),
            intermediate_size=10,
            mean_budget=0.3,
        )
    with pytest.raises(ValueError, match="record_counts"):
        allocate_layer_schedule(
            np.ones((2, 2), dtype=np.float64),
            (4, 4),
            intermediate_size=10,
            mean_budget=0.5,
        )
