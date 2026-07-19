import numpy as np
import pytest

from engram.semantic.router import JointKeyRouter, candidate_recall
from engram.semantic.swiglu import silu


def test_joint_gate_up_candidate_routing_beats_gate_only_choice():
    hidden = np.array([1.0, 0.0])
    gate = np.array(
        [
            [1.0, 0.0],  # Best gate-only match, but its up projection is zero.
            [0.8, 0.6],  # Both keys align with the query.
            [0.2, 0.98],
        ]
    )
    up = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.866],
        ]
    )

    gate_only_choice = int(np.argmax(gate @ hidden))
    router = JointKeyRouter(gate, up, candidate_count=1, top_k=1)
    selection = router.select_candidates(hidden)
    result = router.route(hidden)

    assert gate_only_choice == 0
    assert selection.indices.tolist() == [1]
    assert result.selected_indices.tolist() == [1]
    assert result.selected_activations[0] > 0.0


def test_candidates_are_exactly_reranked_with_both_keys_and_value_norms():
    hidden = np.array([1.0, 0.0])
    gate = np.array([[0.2, 0.0], [3.0, 0.0], [1.0, 0.0]])
    up = np.array([[10.0, 0.0], [0.5, 0.0], [-2.0, 0.0]])
    values = np.array([[1.0, 0.0], [1.0, 0.0], [0.1, 0.0]])
    router = JointKeyRouter(
        gate, up, values=values, candidate_count=3, top_k=2
    )

    result = router.route(hidden)
    expected_activations = silu(gate[:, 0]) * up[:, 0]
    expected_scores = np.abs(expected_activations) * np.linalg.norm(values, axis=1)
    expected_order = np.argsort(-expected_scores, kind="stable")[:2]

    # Normalized proxy scores tie here; exact gate and up magnitudes determine order.
    assert result.candidate_indices.tolist() == [0, 1, 2]
    np.testing.assert_allclose(result.candidate_activations, expected_activations)
    assert result.selected_indices.tolist() == expected_order.tolist()
    np.testing.assert_allclose(result.selected_exact_scores, expected_scores[expected_order])


def test_candidate_recall_reports_set_metrics_and_result_convenience_method():
    metrics = candidate_recall([1, 2, 2, 4], [0, 2, 4])
    assert metrics.hits == 2
    assert metrics.candidate_count == 3
    assert metrics.oracle_count == 3
    assert metrics.recall == pytest.approx(2.0 / 3.0)
    assert metrics.precision == pytest.approx(2.0 / 3.0)

    router = JointKeyRouter(np.eye(3), np.eye(3), candidate_count=2, top_k=1)
    result = router.route(np.array([1.0, 0.0, 0.0]))
    assert result.candidate_recall([0, 2]).recall == pytest.approx(0.5)
    assert candidate_recall([], []).recall == 1.0


def test_configuration_validation_and_deterministic_zero_query_ties():
    keys = np.eye(3)
    with pytest.raises(ValueError, match="top_k must not exceed"):
        JointKeyRouter(keys, keys, candidate_count=1, top_k=2)

    router = JointKeyRouter(keys, keys, candidate_count=2, top_k=1)
    first = router.select_candidates(np.zeros(3))
    second = router.select_candidates(np.zeros(3))
    assert first.indices.tolist() == [0, 1]
    np.testing.assert_array_equal(first.indices, second.indices)
