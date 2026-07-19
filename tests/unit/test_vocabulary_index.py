import numpy as np
import pytest

from engram.vocabulary import VocabularyIndex, VocabularyIndexError, exact_logits


def test_exact_logits_and_top_k_match_manual_dense_projection():
    embeddings = np.array([[1.0, 2.0], [-2.0, 0.5], [0.3, -4.0], [1.0, 2.0]])
    hidden = np.array([0.4, -0.7])
    bias = np.array([0.1, 0.2, -0.3, 0.1])
    expected = embeddings @ hidden + bias
    np.testing.assert_array_equal(exact_logits(hidden, embeddings, bias), expected)

    index = VocabularyIndex(embeddings, bias)
    token_ids, logits = index.exact_top_k(hidden, k=3)
    order = np.argsort(-expected, kind="stable")[:3]
    np.testing.assert_array_equal(token_ids, order)
    np.testing.assert_array_equal(logits, expected[order])


def test_normalized_coarse_search_exactly_rescores_selected_candidates():
    embeddings = np.array(
        [[1.0, 0.0], [10.0, 1.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64
    )
    index = VocabularyIndex(embeddings)
    result = index.search([1.0, 0.0], candidate_count=2, top_k=2, max_candidates=2)

    # Cosine ranks token 0 before token 1; exact MIPS rescoring reverses them.
    np.testing.assert_array_equal(result.candidate_ids, [1, 0])
    np.testing.assert_array_equal(result.candidate_logits, [10.0, 1.0])
    assert result.approximate
    assert not result.exact


def test_low_confidence_adaptively_expands_candidates():
    embeddings = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.3], [100.0, 100.0], [-1.0, 0.0]]
    )
    index = VocabularyIndex(embeddings)
    result = index.search(
        [1.0, 0.0],
        candidate_count=2,
        top_k=1,
        minimum_confidence_margin=0.5,
        expansion_factor=2,
        max_candidates=4,
    )
    assert result.expansions == 1
    assert result.candidate_count == 4
    assert result.token_ids.tolist() == [3]
    assert result.confidence_margin == pytest.approx(99.0)
    assert result.confidence_satisfied


def test_recall_and_logit_error_metrics_against_exact_projection():
    angles = np.arange(12) * (2.0 * np.pi / 12)
    embeddings = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    queries = embeddings[[0, 2, 5, 8, 10]]
    index = VocabularyIndex(embeddings)
    metrics = index.evaluate(queries, candidate_count=5, top_ks=(1, 3, 5))

    assert metrics.query_count == 5
    assert metrics.candidate_count == 5
    assert metrics.top1_recall == 1.0
    assert metrics.top_k_recall == {1: 1.0, 3: 1.0, 5: 1.0}
    assert metrics.mean_top1_logit_error == pytest.approx(0.0)
    assert metrics.max_top1_logit_error == pytest.approx(0.0)
    assert metrics.to_dict()["top_k_recall"] == {"1": 1.0, "3": 1.0, "5": 1.0}


def test_generation_modes_are_seeded_and_distribution_flags_are_explicit():
    rng = np.random.default_rng(55)
    embeddings = rng.normal(size=(20, 6))
    hidden = rng.normal(size=6)
    index = VocabularyIndex(embeddings)

    approximate_greedy = index.greedy(hidden, candidate_count=5)
    exact_greedy = index.greedy(hidden, exact=True)
    assert approximate_greedy.approximate_distribution
    assert not approximate_greedy.exact_distribution
    assert exact_greedy.exact_distribution
    assert exact_greedy.exact_fallback_used
    assert exact_greedy.token_id == int(np.argmax(index.exact_logits(hidden)))

    first_top_k = index.sample_top_k(
        hidden, k=4, candidate_count=8, rng=np.random.default_rng(7)
    )
    second_top_k = index.sample_top_k(
        hidden, k=4, candidate_count=8, rng=np.random.default_rng(7)
    )
    assert first_top_k == second_top_k
    assert first_top_k.method == "top_k"
    assert first_top_k.support_size == 4

    approximate_top_p = index.sample_top_p(
        hidden, top_p=0.9, candidate_count=7, rng=np.random.default_rng(3)
    )
    exact_top_p = index.sample_top_p(
        hidden, top_p=0.9, exact=True, rng=np.random.default_rng(3)
    )
    assert approximate_top_p.method == "top_p_approximate"
    assert approximate_top_p.approximate_distribution
    assert not approximate_top_p.exact_fallback_used
    assert approximate_top_p.candidate_count == 7
    assert exact_top_p.method == "top_p_exact"
    assert exact_top_p.exact_distribution
    assert exact_top_p.exact_fallback_used
    assert exact_top_p.candidate_count == len(embeddings)


def test_invalid_shapes_options_and_nonfinite_values_are_rejected():
    with pytest.raises(VocabularyIndexError, match="rank-2"):
        VocabularyIndex([1.0, 2.0])
    with pytest.raises(VocabularyIndexError, match="finite"):
        VocabularyIndex([[1.0, np.nan]])
    with pytest.raises(VocabularyIndexError, match="bias"):
        VocabularyIndex(np.zeros((3, 2)), bias=np.zeros(2))

    index = VocabularyIndex(np.eye(4))
    with pytest.raises(VocabularyIndexError, match="hidden"):
        index.search([1.0, 2.0], candidate_count=2)
    with pytest.raises(VocabularyIndexError, match="top_k"):
        index.search(np.ones(4), top_k=5)
    with pytest.raises(VocabularyIndexError, match="max_candidates"):
        index.search(np.ones(4), top_k=3, max_candidates=2)
    with pytest.raises(VocabularyIndexError, match="confidence"):
        index.search(np.ones(4), top_k=1, minimum_confidence_margin=-1.0)
    with pytest.raises(VocabularyIndexError, match="top_p"):
        index.sample_top_p(np.ones(4), top_p=0.0)
    with pytest.raises(VocabularyIndexError, match="temperature"):
        index.sample_top_k(np.ones(4), k=2, temperature=0.0)
