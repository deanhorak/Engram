import numpy as np
import pytest

from engram.semantic.ivf import IVFIndexError
from engram.semantic.multilabel_router import (
    HierarchicalLowRankRouter,
    LowRankMultiLabelRouter,
    MultiLabelLinearRouter,
)


def test_multilabel_router_learns_oracle_memberships():
    states = np.array([[2, 0], [1, 0], [0, 1], [0, 2]], dtype=np.float64)
    membership = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
    router = MultiLabelLinearRouter.fit(states, membership, regularization=0.1)

    assert router.search([1, 0], candidate_count=1).indices.tolist() == [0]
    assert router.search([0, 1], candidate_count=1).indices.tolist() == [1]


def test_multilabel_router_validates_labels_and_candidate_count():
    states = np.eye(2)
    with pytest.raises(IVFIndexError, match="zero and one"):
        MultiLabelLinearRouter.fit(states, [[1, 0], [0, 0.5]])

    router = MultiLabelLinearRouter.fit(states, np.eye(2))
    with pytest.raises(IVFIndexError, match="candidate_count"):
        router.search([1, 0], candidate_count=3)


def test_low_rank_router_matches_dense_router_when_rank_is_full():
    dense = MultiLabelLinearRouter(
        [[3.0, 0.0, 1.0], [0.0, 2.0, 1.0]], [0.1, -0.2, 0.0]
    )
    compressed = LowRankMultiLabelRouter.compress(dense, rank=2)

    np.testing.assert_allclose(compressed.scores([0.5, 0.25]), [1.6, 0.3, 0.75])
    assert compressed.search([1, 0], candidate_count=2).indices.tolist() == [0, 2]
    assert compressed.parameter_bytes() == (2 * 2 + 2 * 3 + 3) * 4


def test_hierarchical_router_selects_groups_then_exact_reranks():
    dense = MultiLabelLinearRouter(
        [[4.0, 3.0, 0.0, 0.0], [0.0, 0.0, 4.0, 3.0]], np.zeros(4)
    )
    gate = np.array([[3, 0], [2, 0], [0, 3], [0, 2]], dtype=np.float64)
    up = gate.copy()
    values = np.eye(4, dtype=np.float64)
    router = HierarchicalLowRankRouter.fit(
        dense, gate, up, values, rank=2, groups=2, iterations=4
    )

    result = router.search([1, 0], groups_to_probe=1, candidate_count=1)

    assert result.indices.tolist() == [0]
    assert result.probed_record_count == 2
    assert router.router_parameter_bytes() == (2 * 2 + 2 * 2 + 2) * 4


def test_coverage_trained_hierarchy_learns_group_utility():
    states = np.array([[2, 0], [1, 0], [0, 1], [0, 2]], dtype=np.float64)
    membership = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]],
        dtype=np.float64,
    )
    gate = np.array([[3, 0], [2, 0], [0, 3], [0, 2]], dtype=np.float64)
    router = HierarchicalLowRankRouter.fit_coverage(
        states,
        membership,
        gate,
        gate,
        np.eye(4),
        rank=2,
        groups=2,
        regularization=0.1,
        iterations=4,
    )

    horizontal = router.search([1, 0], groups_to_probe=1, candidate_count=2)
    vertical = router.search([0, 1], groups_to_probe=1, candidate_count=2)

    assert set(horizontal.indices.tolist()) == {0, 1}
    assert set(vertical.indices.tolist()) == {2, 3}
