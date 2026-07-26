import numpy as np
import pytest

from engram.evaluation.native_bitnet_adaptive_k import (
    adaptive_k_from_candidate_utility,
    maximum_k_under_physical_limit,
)


def test_adaptive_k_uses_stable_source_index_ties_and_energy_target():
    result = adaptive_k_from_candidate_utility(
        np.asarray([[5, 2, 3]], dtype=np.int64),
        np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32),
        energy_targets=(0.5, 0.75),
        minimum_k=1,
        maximum_k=3,
    )

    assert result[0.5]["selected_k"].tolist() == [1]
    assert result[0.75]["selected_k"].tolist() == [2]
    assert result[0.5]["sorted_indices"].tolist() == [[2, 5, 3]]
    np.testing.assert_allclose(
        result[0.75]["captured_candidate_energy"],
        1.0,
    )


def test_adaptive_k_applies_minimum_and_maximum_clamps():
    result = adaptive_k_from_candidate_utility(
        np.asarray([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64),
        np.asarray(
            [[10.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
            dtype=np.float32,
        ),
        energy_targets=(0.9,),
        minimum_k=2,
        maximum_k=3,
    )[0.9]

    assert result["selected_k"].tolist() == [2, 3]
    assert result["target_attained"].tolist() == [True, False]


def test_physical_maximum_k_resolves_candidate_down_tradeoff():
    report = maximum_k_under_physical_limit(
        2560,
        6912,
        input_count=1920,
        candidate_count=5359,
    )
    adjusted = maximum_k_under_physical_limit(
        2560,
        6912,
        input_count=1920,
        candidate_count=4887,
    )

    assert report["maximum_k"] == 1995
    assert report["fraction_of_dense_q4"] <= 0.45
    assert 0 <= report["audit_reserve_bytes"] < 512
    assert adjusted["maximum_k"] >= 2938


@pytest.mark.parametrize(
    "kwargs",
    [
        {"energy_targets": (0.0,), "minimum_k": 1, "maximum_k": 2},
        {"energy_targets": (0.9,), "minimum_k": 0, "maximum_k": 2},
        {"energy_targets": (0.9,), "minimum_k": 3, "maximum_k": 2},
    ],
)
def test_adaptive_k_rejects_invalid_targets_and_clamps(kwargs):
    with pytest.raises(ValueError):
        adaptive_k_from_candidate_utility(
            np.asarray([[0, 1]], dtype=np.int64),
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            **kwargs,
        )
