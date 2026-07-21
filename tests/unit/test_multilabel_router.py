import numpy as np
import pytest

from engram.semantic.ivf import IVFIndexError
from engram.semantic.multilabel_router import (
    HierarchicalLowRankRouter,
    LowRankMultiLabelRouter,
    MultiLabelLinearRouter,
    OverlappingCoverageRouter,
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


def test_direct_low_rank_fit_matches_dense_fit_then_compress():
    rng = np.random.default_rng(19)
    states = rng.normal(size=(7, 5))
    membership = (rng.uniform(size=(7, 9)) > 0.6).astype(np.float64)
    dense = MultiLabelLinearRouter.fit(states, membership, regularization=3.0)
    expected = LowRankMultiLabelRouter.compress(dense, rank=3)

    direct = LowRankMultiLabelRouter.fit(
        states, membership, rank=3, regularization=3.0
    )

    probes = rng.normal(size=(4, 5))
    expected_scores = np.asarray([expected.scores(row) for row in probes])
    direct_scores = np.asarray([direct.scores(row) for row in probes])
    np.testing.assert_allclose(direct_scores, expected_scores, rtol=1e-10, atol=1e-10)


def test_direct_low_rank_dual_fit_matches_dense_fit_then_compress():
    rng = np.random.default_rng(23)
    states = rng.normal(size=(4, 7))
    membership = (rng.uniform(size=(4, 9)) > 0.6).astype(np.float64)
    dense = MultiLabelLinearRouter.fit(states, membership, regularization=2.0)
    expected = LowRankMultiLabelRouter.compress(dense, rank=3)

    direct = LowRankMultiLabelRouter.fit(
        states, membership, rank=3, regularization=2.0
    )

    probes = rng.normal(size=(4, 7))
    expected_scores = np.asarray([expected.scores(row) for row in probes])
    direct_scores = np.asarray([direct.scores(row) for row in probes])
    np.testing.assert_allclose(direct_scores, expected_scores, rtol=1e-10, atol=1e-10)


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


def test_overlapping_coverage_router_bounds_replication_and_deduplicates_candidates():
    states = np.array(
        [[2, 0, 0], [1, 0, 0], [0, 2, 0], [0, 1, 0], [0, 0, 2], [0, 0, 1]],
        dtype=np.float64,
    )
    membership = np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1],
        ],
        dtype=np.float64,
    )
    gate = np.array(
        [[3, 0, 0], [2, 0, 0], [0, 3, 0], [0, 2, 0], [0, 0, 3], [0, 0, 2]],
        dtype=np.float64,
    )
    router = OverlappingCoverageRouter.fit(
        states,
        membership,
        gate,
        gate,
        np.ones((6, 3)),
        rank=2,
        groups=3,
        posting_size=4,
        candidate_count=4,
        regularization=0.1,
        iterations=3,
        max_replication=2,
    )

    replication = np.bincount(router.postings.reshape(-1), minlength=6)
    assert replication.tolist() == [2, 2, 2, 2, 2, 2]
    result = router.search([1, 0, 0], candidate_count=4)
    assert len(result.indices) == len(set(result.indices.tolist())) == 4
    assert result.probed_clusters.size >= 1
    assert result.probed_record_count >= 4
    assert router.posting_bytes == 3 * 4 * 2


def test_overlapping_coverage_router_is_deterministic():
    rng = np.random.default_rng(31)
    states = rng.normal(size=(8, 4))
    membership = np.zeros((8, 8), dtype=np.float64)
    for index in range(8):
        membership[index, [index, (index + 1) % 8]] = 1.0
    gate = rng.normal(size=(8, 4))
    arguments = dict(
        rank=2,
        groups=4,
        posting_size=4,
        candidate_count=5,
        regularization=1.0,
        iterations=2,
        max_replication=2,
    )

    first = OverlappingCoverageRouter.fit(
        states, membership, gate, gate, np.ones((8, 4)), **arguments
    )
    second = OverlappingCoverageRouter.fit(
        states, membership, gate, gate, np.ones((8, 4)), **arguments
    )

    np.testing.assert_array_equal(first.postings, second.postings)
    np.testing.assert_allclose(first.input_factors, second.input_factors)
    np.testing.assert_allclose(first.group_factors, second.group_factors)


def test_overlapping_coverage_router_rejects_incompatible_or_nonfinite_arrays():
    arguments = {
        "input_factors": np.ones((3, 2)),
        "group_factors": np.ones((2, 2)),
        "group_bias": np.zeros(2),
        "postings": np.array([[0, 1], [2, 3]]),
        "gate_keys": np.ones((4, 3)),
        "up_keys": np.ones((4, 3)),
        "value_norms": np.ones(4),
        "max_replication": 1,
    }

    with pytest.raises(IVFIndexError, match="hidden width"):
        OverlappingCoverageRouter(
            **{**arguments, "input_factors": np.ones((4, 2))}
        )
    with pytest.raises(IVFIndexError, match="non-negative"):
        OverlappingCoverageRouter(
            **{**arguments, "value_norms": np.array([1.0, 1.0, np.nan, 1.0])}
        )
