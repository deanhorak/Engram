import numpy as np
import pytest

from engram.semantic.dip import (
    dynamic_input_pruning,
    input_coordinate_count,
    partial_proxy_scores,
    projected_dip_traffic,
    stable_top_k,
)
from engram.semantic.swiglu import silu


def test_partial_proxy_is_exactly_completed_and_reranked():
    hidden = np.array([1.0, 1.0])
    gate = np.array(
        [
            [2.0, -2.0],  # Strong partial score, zero after completion.
            [1.0, 2.0],  # Weaker partial score, largest exact score.
            [0.0, 0.0],
        ]
    )
    up = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    down = np.eye(2, 3)

    result = dynamic_input_pruning(
        hidden,
        gate,
        up,
        down=down,
        input_fraction=0.5,
        candidate_count=2,
        top_k=1,
    )

    assert result.input_count == 1
    assert result.input_indices.tolist() == [0]
    assert result.candidate_indices.tolist() == [0, 1]
    assert result.selected_indices.tolist() == [1]
    assert result.oracle_indices.tolist() == [1]
    assert result.candidate_recall == 1.0
    assert result.oracle_score_mass == 1.0
    assert result.output_relative_l2 == pytest.approx(0.0)
    assert result.output_cosine == pytest.approx(1.0)
    np.testing.assert_allclose(result.selected_activations, [silu(np.array(3.0))])
    np.testing.assert_allclose(result.selected_output, result.full_output)


def test_candidate_diagnostics_measure_missed_oracle_membership_and_score_mass():
    hidden = np.array([1.0, 1.0])
    gate = np.array([[2.0, -2.0], [1.0, 2.0], [0.0, 0.0]])
    up = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    down = np.eye(2, 3)

    result = dynamic_input_pruning(
        hidden,
        gate,
        up,
        down=down,
        input_fraction=0.5,
        candidate_count=1,
        top_k=1,
    )

    assert result.candidate_indices.tolist() == [0]
    assert result.oracle_indices.tolist() == [1]
    assert result.candidate_recall == 0.0
    assert result.oracle_score_mass == 0.0
    assert result.output_relative_l2 == pytest.approx(1.0)
    assert result.output_cosine == pytest.approx(0.0)


def test_stable_ties_use_source_index_order_and_explicit_norms_work_without_down():
    result = dynamic_input_pruning(
        np.array([1.0, -1.0, 0.0]),
        np.zeros((4, 3)),
        np.zeros((4, 3)),
        value_norms=np.ones(4),
        input_fraction=2.0 / 3.0,
        candidate_count=3,
        top_k=2,
    )

    assert result.input_indices.tolist() == [0, 1]
    assert result.candidate_indices.tolist() == [0, 1, 2]
    assert result.selected_indices.tolist() == [0, 1]
    assert result.oracle_indices.tolist() == [0, 1]
    assert result.candidate_recall == 1.0
    assert result.oracle_score_mass == 1.0
    assert result.full_output is None
    assert result.output_relative_l2 is None
    assert result.output_cosine is None


def test_exact_rerank_ties_ignore_proxy_candidate_order():
    result = dynamic_input_pruning(
        np.array([1.0, 1.0]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        value_norms=np.ones(2),
        input_fraction=0.5,
        candidate_count=2,
        top_k=1,
    )

    assert result.candidate_indices.tolist() == [1, 0]
    assert result.selected_indices.tolist() == [0]
    assert result.oracle_indices.tolist() == [0]


def test_partial_proxy_exposes_one_reusable_stable_order():
    proxy = partial_proxy_scores(
        np.array([2.0, -2.0, 1.0]),
        np.zeros((4, 3)),
        np.zeros((4, 3)),
        np.ones(4),
        input_fraction=2.0 / 3.0,
    )

    assert proxy.input_indices.tolist() == [0, 1]
    assert proxy.order.tolist() == [0, 1, 2, 3]
    assert proxy.order[:2].tolist() == stable_top_k(proxy.proxy_scores, 2).tolist()


def test_projected_traffic_matches_exact_completion_formula():
    traffic = projected_dip_traffic(
        576,
        1536,
        input_fraction=0.5,
        candidate_count=1024,
        top_k=768,
        bytes_per_element=2,
    )

    partial = 2 * 1536 * 288
    completion = 2 * 1024 * (576 - 288)
    selected_down = 768 * 576
    dense = 3 * 1536 * 576
    assert traffic.input_count == 288
    assert traffic.partial_projection_elements == partial
    assert traffic.candidate_completion_elements == completion
    assert traffic.selected_down_elements == selected_down
    assert traffic.total_elements == partial + completion + selected_down
    assert traffic.dense_elements == dense
    assert traffic.total_bytes == 2 * traffic.total_elements
    assert traffic.dense_bytes == 2 * dense
    assert traffic.fraction_of_dense == pytest.approx(13.0 / 18.0)
    assert traffic.reduction_factor == pytest.approx(18.0 / 13.0)


def test_positive_fraction_always_keeps_at_least_one_coordinate():
    assert input_coordinate_count(4, 0.01) == 1


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, np.nan, True, "half"])
def test_input_fraction_validation(fraction):
    with pytest.raises(ValueError, match="input_fraction"):
        input_coordinate_count(4, fraction)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"candidate_count": 4, "top_k": 1}, "intermediate dimension"),
        ({"candidate_count": 2, "top_k": 3}, "top_k must not exceed"),
        ({"candidate_count": 0, "top_k": 1}, "positive integer"),
    ],
)
def test_selection_count_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        dynamic_input_pruning(
            np.ones(2),
            np.ones((3, 2)),
            np.ones((3, 2)),
            value_norms=np.ones(3),
            input_fraction=0.5,
            **kwargs,
        )


def test_weight_and_norm_validation():
    hidden = np.ones(2)
    gate = np.ones((3, 2))
    with pytest.raises(ValueError, match="same shape"):
        dynamic_input_pruning(
            hidden,
            gate,
            np.ones((2, 2)),
            value_norms=np.ones(3),
            input_fraction=0.5,
            candidate_count=2,
            top_k=1,
        )
    with pytest.raises(ValueError, match="either down or value_norms"):
        dynamic_input_pruning(
            hidden,
            gate,
            gate,
            input_fraction=0.5,
            candidate_count=2,
            top_k=1,
        )
    with pytest.raises(ValueError, match="down must have shape"):
        dynamic_input_pruning(
            hidden,
            gate,
            gate,
            down=np.ones((3, 2)),
            input_fraction=0.5,
            candidate_count=2,
            top_k=1,
        )
    with pytest.raises(ValueError, match="non-negative"):
        dynamic_input_pruning(
            hidden,
            gate,
            gate,
            value_norms=np.array([1.0, -1.0, 1.0]),
            input_fraction=0.5,
            candidate_count=2,
            top_k=1,
        )
