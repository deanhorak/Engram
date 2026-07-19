import numpy as np
import pytest

from engram.episodic.retrieval import OlderContextRetrievalStore, retrieval_recall


def test_recent_eviction_and_older_capacity_keep_the_newest_context():
    store = OlderContextRetrievalStore(
        2, 3, recent_window=2, capacity=3, candidate_count=2
    )
    for position in range(6):
        store.append(
            np.array([position + 1.0, 1.0]),
            np.full(3, position, dtype=np.float32),
            position=position,
        )

    assert store.recent_positions.tolist() == [4, 5]
    assert store.older_positions.tolist() == [1, 2, 3]
    assert store.recent_count == 2
    assert store.older_count == 3


def test_quantized_candidates_are_exactly_reranked_and_values_decode():
    store = OlderContextRetrievalStore(
        2,
        2,
        recent_window=0,
        capacity=4,
        candidate_count=2,
        value_storage="int8",
    )
    store.append([1.0, 0.0], [0.25, -0.5], position=10)
    store.append([2.0, 0.2], [1.0, -2.0], position=11)
    store.append([-1.0, 0.0], [3.0, 4.0], position=12)

    result = store.retrieve([1.0, 0.0], top_k=1)

    # Cosine search proposes position 10 first, but exact MIPS reranking selects 11.
    assert result.candidate_positions.tolist() == [10, 11]
    assert result.positions.tolist() == [11]
    assert result.scores[0] > 1.9
    np.testing.assert_allclose(result.values[0], [1.0, -2.0], atol=0.02)
    assert result.reads.candidate_search_bytes > result.reads.exact_rerank_bytes
    assert result.reads.total_bytes == (
        result.reads.candidate_search_bytes
        + result.reads.exact_rerank_bytes
        + result.reads.selected_value_bytes
    )


def test_float_value_fallback_is_lossless_and_recall_metrics_are_reported():
    store = OlderContextRetrievalStore(
        2,
        2,
        recent_window=0,
        capacity=3,
        candidate_count=3,
        value_storage="float32",
    )
    values = [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6])]
    for position, (key, value) in enumerate(zip(np.eye(3, 2), values)):
        store.append(key, value, position=position)

    result = store.retrieve([1.0, 0.0], top_k=2)
    np.testing.assert_array_equal(result.values[0], values[0].astype(np.float32))
    metrics = result.recall_against([0, 1])
    assert metrics.hits == 2
    assert metrics.recall == 1.0
    assert metrics.precision == 1.0

    partial = retrieval_recall([0, 1, 1], [0, 2])
    assert partial.recall == pytest.approx(0.5)
    assert partial.precision == pytest.approx(0.5)
    assert retrieval_recall([], []).recall == 1.0


def test_memory_metrics_are_bounded_by_configured_capacity():
    store = OlderContextRetrievalStore(
        4, 5, recent_window=2, capacity=3, candidate_count=2, value_storage="int8"
    )
    initial = store.memory_metrics()
    assert initial.older_allocated_bytes > 0
    for position in range(20):
        store.append(np.full(4, position), np.full(5, position), position=position)
    final = store.memory_metrics()

    assert store.older_count == 3
    assert store.recent_count == 2
    assert final.older_allocated_bytes == initial.older_allocated_bytes
    assert final.older_active_bytes == final.older_allocated_bytes
    assert final.total_active_bytes == final.total_allocated_bytes


def test_empty_store_and_configuration_validation():
    store = OlderContextRetrievalStore(2, 2, recent_window=1, capacity=0)
    assert store.retrieve([1.0, 0.0], top_k=1).positions.size == 0
    store.append([1.0, 0.0], [1.0, 0.0])
    store.append([0.0, 1.0], [0.0, 1.0])
    assert store.older_count == 0

    with pytest.raises(ValueError, match="top_k must not exceed"):
        OlderContextRetrievalStore(2, 2, recent_window=0, capacity=2).retrieve(
            [1.0, 0.0], top_k=2, candidate_count=1
        )
