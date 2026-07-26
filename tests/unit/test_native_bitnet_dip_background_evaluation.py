import numpy as np

from engram.evaluation.native_bitnet_dip_background import (
    _evaluate_cross_fit,
    maximum_candidates_with_background,
    maximum_k_for_candidates,
)


def test_layer7_boundary_counts_reserve_capsule_under_45_percent():
    candidate_count = maximum_candidates_with_background(
        2560,
        6912,
        input_count=1920,
        maximum_k=2420,
        background_bytes=5184,
    )

    assert candidate_count == 5141
    assert (
        maximum_k_for_candidates(
            2560,
            6912,
            input_count=1920,
            candidate_count=5359,
        )
        == 1995
    )


def test_cross_fit_capsule_improves_repeated_trigger_residual():
    sample_ids = np.repeat(np.arange(4), 2)
    attained = np.asarray([True, False] * 4)
    selected = np.zeros((8, 2), dtype=np.float32)
    dense = selected.copy()
    dense[~attained] = np.asarray([1.0, -0.5], dtype=np.float32)
    # Keep attained-row reference norms non-zero while preserving zero error.
    dense[attained] = 1.0
    selected[attained] = 1.0

    report = _evaluate_cross_fit(
        dense,
        selected,
        attained,
        sample_ids,
        layer=7,
    )

    assert report["parity_mean_relative_improvement"] > 0.99
    assert (
        report["leave_one_trigger_sequence_out"][
            "mean_relative_improvement"
        ]
        > 0.99
    )
