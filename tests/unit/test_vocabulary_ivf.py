import json

import numpy as np
import pytest

from engram.vocabulary.ivf import VocabularyIVFError, VocabularyIVFIndex


def _grouped_embeddings():
    return np.repeat(
        np.array(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            dtype=np.float32,
        ),
        3,
        axis=0,
    )


def _normalized(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def test_minimum_probes_expand_until_candidate_count_and_score_only_postings():
    embeddings = _grouped_embeddings()
    index = VocabularyIVFIndex.build(
        embeddings, num_clusters=4, iterations=10
    )
    result = index.search(
        [1.0, 0.0], embeddings, candidate_count=5, minimum_probes=1
    )

    assert result.candidate_ids.tolist() == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(result.proxy_scores[:3], np.ones(3))
    assert result.probes == 2
    assert result.expansions == 1
    assert result.probed_token_count == 6
    assert result.proxy_scores_computed == 6
    assert result.proxy_scores_computed < index.vocabulary_size
    assert len(np.unique(result.candidate_ids)) == result.candidate_ids.size
    assert result.bytes_read > 0
    assert index.centroids_bytes == 4 * 2 * np.dtype(np.float32).itemsize
    assert index.postings_bytes == (5 + 12) * np.dtype(np.uint32).itemsize
    assert index.total_bytes == index.centroids_bytes + index.postings_bytes


def test_zero_query_expands_deterministically_and_returns_stable_token_ids():
    embeddings = _grouped_embeddings()
    index = VocabularyIVFIndex.build(embeddings, num_clusters=4)
    first = index.search(
        np.zeros(2), embeddings, candidate_count=12, minimum_probes=1
    )
    second = index.search(
        np.zeros(2), embeddings, candidate_count=12, minimum_probes=1
    )

    assert first.candidate_ids.tolist() == list(range(12))
    np.testing.assert_array_equal(first.candidate_ids, second.candidate_ids)
    np.testing.assert_array_equal(first.proxy_scores, np.zeros(12))
    np.testing.assert_array_equal(first.probed_clusters, second.probed_clusters)
    assert first.probes == 0
    assert first.expansions == 0

    partial = index.search(
        np.zeros(2), embeddings, candidate_count=5, minimum_probes=1
    )
    assert partial.candidate_ids.tolist() == list(range(5))
    assert partial.probes == 0
    assert partial.bytes_read == 0


def test_candidate_ids_support_exact_rescoring():
    embeddings = np.array(
        [[1.0, 0.0], [10.0, 1.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=np.float32,
    )
    index = VocabularyIVFIndex.build(embeddings, num_clusters=2)
    hidden = np.array([1.0, 0.0])
    result = index.search(
        hidden, _normalized(embeddings), candidate_count=2, minimum_probes=1
    )
    exact_logits = embeddings[result.candidate_ids] @ hidden
    exact_order = np.lexsort((result.candidate_ids, -exact_logits))

    assert result.candidate_ids[exact_order[0]] == 1
    assert exact_logits[exact_order[0]] == 10.0


def test_mmap_roundtrip_preserves_runtime_only_arrays(tmp_path):
    rng = np.random.default_rng(19)
    embeddings = rng.normal(size=(37, 6)).astype(np.float32)
    index = VocabularyIVFIndex.build(
        embeddings, num_clusters=6, iterations=12
    )
    hidden = rng.normal(size=6)
    normalized = _normalized(embeddings)
    expected = index.search(
        hidden, normalized, candidate_count=11, minimum_probes=2
    )
    directory = index.save(tmp_path / "vocabulary" / "ivf")
    loaded = VocabularyIVFIndex.load(directory)
    actual = loaded.search(
        hidden, normalized, candidate_count=11, minimum_probes=2
    )

    assert isinstance(loaded.centroid_vectors, np.memmap)
    assert isinstance(loaded.posting_offsets, np.memmap)
    assert isinstance(loaded.token_ids, np.memmap)
    assert not hasattr(loaded, "embeddings")
    np.testing.assert_array_equal(actual.candidate_ids, expected.candidate_ids)
    np.testing.assert_array_equal(actual.proxy_scores, expected.proxy_scores)
    np.testing.assert_array_equal(actual.probed_clusters, expected.probed_clusters)
    assert actual.probed_token_count == expected.probed_token_count
    assert loaded.total_bytes == index.total_bytes

    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["format"] == "engram.vocabulary.normalized_ivf"
    assert metadata["centroid_dtype"] == "float32"
    assert metadata["posting_dtype"] == "uint32"


def test_strong_build_search_and_corruption_validation(tmp_path):
    with pytest.raises(VocabularyIVFError, match="rank-2"):
        VocabularyIVFIndex.build([1.0, 2.0], num_clusters=1)
    with pytest.raises(VocabularyIVFError, match="cannot exceed"):
        VocabularyIVFIndex.build(np.eye(3), num_clusters=4)

    embeddings = np.eye(5, dtype=np.float32)
    index = VocabularyIVFIndex.build(embeddings, num_clusters=2)
    with pytest.raises(VocabularyIVFError, match="shape"):
        index.search(np.ones(4), embeddings, candidate_count=2)
    with pytest.raises(VocabularyIVFError, match="runtime embeddings shape"):
        index.search(np.ones(5), embeddings[:-1], candidate_count=2)
    with pytest.raises(VocabularyIVFError, match="minimum_probes"):
        index.search(np.ones(5), embeddings, candidate_count=2, minimum_probes=3)

    directory = index.save(tmp_path / "ivf")
    tokens = np.load(directory / "token_ids.npy")
    tokens[0] = tokens[1]
    np.save(directory / "token_ids.npy", tokens, allow_pickle=False)
    with pytest.raises(VocabularyIVFError, match="permutation"):
        VocabularyIVFIndex.load(directory)

    directory = index.save(tmp_path / "ivf-wrong-dtype")
    offsets = np.load(directory / "posting_offsets.npy").astype(np.int64)
    np.save(directory / "posting_offsets.npy", offsets, allow_pickle=False)
    with pytest.raises(VocabularyIVFError, match="dtype uint32"):
        VocabularyIVFIndex.load(directory)
