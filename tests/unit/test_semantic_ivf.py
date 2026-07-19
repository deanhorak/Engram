import json

import numpy as np
import pytest

from engram.semantic.ivf import (
    IVFIndexError,
    JointKeyIVFIndex,
    JointKeyIVFProbeIndex,
    SeparateKeyIVFIndex,
)


def test_joint_key_clustering_and_search_do_not_scan_all_records():
    # Gate geometry alone cannot distinguish these records. Alternating up keys
    # ensure stable gate-only selection would include irrelevant even IDs.
    gate = np.tile([1.0, 0.0], (8, 1))
    up = np.array([[0.0, 1.0] if i % 2 == 0 else [1.0, 0.0] for i in range(8)])
    index = JointKeyIVFIndex.build(gate, up, num_clusters=2, iterations=10)
    result = index.search([1.0, 0.0], probes=1, candidate_count=8)

    assert result.indices.tolist() == [1, 3, 5, 7]
    np.testing.assert_array_equal(result.proxy_scores, np.ones(4))
    assert result.probed_record_count == 4
    assert result.probed_record_count < index.records
    assert len(np.unique(result.indices)) == result.indices.size
    assert index.records_bytes == 2 * 8 * 2 * np.dtype(np.float32).itemsize
    assert index.centroids_bytes == 2 * 2 * 2 * np.dtype(np.float32).itemsize
    assert index.postings_bytes > 0
    assert index.total_bytes == (
        index.records_bytes + index.centroids_bytes + index.postings_bytes
    )


def test_search_ties_are_deterministic_and_probe_count_controls_work():
    gate = np.eye(6, 3, dtype=np.float32)
    up = gate.copy()
    index = JointKeyIVFIndex.build(gate, up, num_clusters=3, iterations=8)
    first = index.search(np.zeros(3), probes=1, candidate_count=10)
    second = index.search(np.zeros(3), probes=1, candidate_count=10)
    expanded = index.search(np.zeros(3), probes=3, candidate_count=10)

    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_array_equal(first.proxy_scores, second.proxy_scores)
    assert first.probed_clusters.tolist() == [0]
    assert first.indices.tolist() == sorted(first.indices.tolist())
    assert expanded.indices.tolist() == list(range(6))
    assert expanded.probed_record_count == index.records


def test_mmap_safe_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(44)
    index = JointKeyIVFIndex.build(
        rng.normal(size=(31, 7)),
        rng.normal(size=(31, 7)),
        num_clusters=5,
        iterations=12,
    )
    hidden = rng.normal(size=7)
    expected = index.search(hidden, probes=2, candidate_count=9)
    directory = index.save(tmp_path / "semantic" / "ivf")
    loaded = JointKeyIVFIndex.load(directory)
    actual = loaded.search(hidden, probes=2, candidate_count=9)

    assert isinstance(loaded.gate_records, np.memmap)
    assert isinstance(loaded.posting_indices, np.memmap)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.proxy_scores, expected.proxy_scores)
    np.testing.assert_array_equal(actual.probed_clusters, expected.probed_clusters)
    assert actual.probed_record_count == expected.probed_record_count
    assert loaded.total_bytes == index.total_bytes

    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["format"] == "engram.semantic.joint_key_ivf"
    assert metadata["records"] == 31
    assert metadata["centroids"] == 5


def test_build_search_and_corrupt_posting_validation(tmp_path):
    keys = np.eye(4, dtype=np.float32)
    with pytest.raises(IVFIndexError, match="same shape"):
        JointKeyIVFIndex.build(keys, keys[:, :-1], num_clusters=2)
    with pytest.raises(IVFIndexError, match="cannot exceed"):
        JointKeyIVFIndex.build(keys, keys, num_clusters=5)

    index = JointKeyIVFIndex.build(keys, keys, num_clusters=2)
    with pytest.raises(IVFIndexError, match="probes cannot exceed"):
        index.search(np.ones(4), probes=3, candidate_count=2)
    with pytest.raises(IVFIndexError, match="shape"):
        index.search(np.ones(3), probes=1, candidate_count=2)

    directory = index.save(tmp_path / "ivf")
    offsets = np.load(directory / "posting_offsets.npy")
    offsets[-1] -= 1
    np.save(directory / "posting_offsets.npy", offsets, allow_pickle=False)
    with pytest.raises(IVFIndexError, match="span all postings"):
        JointKeyIVFIndex.load(directory)


def test_runtime_probe_index_uses_uint32_and_expands_only_until_sufficient(tmp_path):
    gate = np.tile([1.0, 0.0], (8, 1))
    up = np.array([[0.0, 1.0] if i % 2 == 0 else [1.0, 0.0] for i in range(8)])
    index = JointKeyIVFProbeIndex.build(
        gate, up, num_clusters=4, iterations=10
    )
    result = index.probe([1.0, 0.0], probes=1, minimum_records=3)
    assert 3 <= result.indices.size < index.records
    assert result.clusters.size >= 1
    assert index.posting_offsets.dtype == np.uint32
    assert index.posting_indices.dtype == np.uint32
    assert result.index_bytes_read < index.total_bytes

    directory = index.save(tmp_path / "runtime-ivf")
    loaded = JointKeyIVFProbeIndex.load(directory)
    actual = loaded.probe([1.0, 0.0], probes=1, minimum_records=3)
    assert isinstance(loaded.joint_centroids, np.memmap)
    np.testing.assert_array_equal(actual.indices, result.indices)
    np.testing.assert_array_equal(actual.clusters, result.clusters)


def test_separate_key_index_unions_gate_and_up_postings_deterministically():
    gate = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    up = np.array([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=np.float32)
    index = SeparateKeyIVFIndex.build(gate, up, num_clusters=2, iterations=8)

    first = index.search([1, 0], probes=1, candidate_count=4)
    second = index.search([1, 0], probes=1, candidate_count=4)

    np.testing.assert_array_equal(first.indices, second.indices)
    assert 1 in first.indices
    assert first.probed_record_count >= first.indices.size
    assert first.probed_record_count <= index.records


def test_separate_key_index_reranks_with_true_key_magnitudes():
    gate = np.array([[1, 0], [3, 0]], dtype=np.float32)
    up = np.array([[1, 0], [1, 0]], dtype=np.float32)
    index = SeparateKeyIVFIndex.build(
        gate, up, num_clusters=1, iterations=2, value_norms=[1, 1]
    )

    result = index.search([1, 0], probes=1, candidate_count=1)

    assert result.indices.tolist() == [1]
    assert result.proxy_scores[0] > 0
