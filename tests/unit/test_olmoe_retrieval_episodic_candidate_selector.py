from __future__ import annotations

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_candidate_selector as selector
from engram.evaluation.olmoe_retrieval_episodic_candidate_selector import (
    evaluate_query_only_router,
)


def test_query_only_router_rejects_unbound_shapes() -> None:
    with pytest.raises(ValueError, match="training shapes changed"):
        evaluate_query_only_router(
            np.zeros((1,), dtype=np.float32),
        np.zeros((1,), dtype=np.float32),
        )


def test_actual_query_key_recall_accepts_tiny_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(selector, "_RECORDS", 2)
    monkeypatch.setattr(selector, "_READS", 2)
    monkeypatch.setattr(selector, "_LAYERS", 1)
    monkeypatch.setattr(selector, "_HEADS", 1)
    monkeypatch.setattr(selector, "_HEAD_DIMENSION", 4)
    monkeypatch.setattr(selector, "_CANDIDATES", 8)
    monkeypatch.setattr(selector.qk.full, "_READ_POSITIONS", np.array([0, 1]))
    rng = np.random.default_rng(3)
    queries = rng.normal(size=(2, 2, 1, 1, 4)).astype(np.float32)
    keys = rng.normal(size=(2, 2, 1, 1, 8, 4)).astype(np.float32)
    scores = np.einsum(
        "nrd,nrcd->nrc", queries[:, :, 0, 0, :], keys[:, :, 0, 0, :, :]
    )
    result = selector.evaluate_actual_query_key_recall(
        queries,
        np.array([0, 1], dtype=np.int64),
        keys,
        scores[:, :, None, None, :],
        ranks=(2,),
    )
    assert result["dense_exact_query_recall_mean"] >= 0.0
    assert "2" in result["results"]


def test_exact_rerank_accepts_tiny_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(selector, "_RECORDS", 2)
    monkeypatch.setattr(selector, "_READS", 2)
    monkeypatch.setattr(selector, "_LAYERS", 1)
    monkeypatch.setattr(selector, "_HEADS", 1)
    monkeypatch.setattr(selector, "_HEAD_DIMENSION", 4)
    monkeypatch.setattr(selector, "_CANDIDATES", 8)
    monkeypatch.setattr(selector.qk.full, "_READ_POSITIONS", np.array([0, 1]))
    rng = np.random.default_rng(7)
    queries = rng.normal(size=(2, 2, 1, 1, 4)).astype(np.float32)
    keys = rng.normal(size=(2, 2, 1, 1, 8, 4)).astype(np.float32)
    values = rng.normal(size=keys.shape).astype(np.float32)
    scores = np.einsum(
        "nrd,nrcd->nrc", queries[:, :, 0, 0, :], keys[:, :, 0, 0, :, :]
    )
    result = selector.evaluate_query_key_exact_rerank(
        queries,
        np.array([0, 1], dtype=np.int64),
        keys,
        values,
        scores[:, :, None, None, :],
        ranks=(2,),
        pool_sizes=(4,),
    )
    assert result["results"]["rank2_pool4"]["candidate_membership_recall_mean"] >= 0.0


def test_exact_rerank_masks_preserve_prefix_and_select_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(selector, "_RECORDS", 2)
    monkeypatch.setattr(selector, "_READS", 2)
    monkeypatch.setattr(selector, "_LAYERS", 1)
    monkeypatch.setattr(selector, "_HEADS", 1)
    monkeypatch.setattr(selector, "_HEAD_DIMENSION", 4)
    monkeypatch.setattr(selector, "_CANDIDATES", 8)
    monkeypatch.setattr(selector, "_POSITIONS", 4)
    monkeypatch.setattr(selector.qk.full, "_READ_POSITIONS", np.array([0, 1]))
    rng = np.random.default_rng(11)
    queries = rng.normal(size=(2, 2, 1, 1, 4)).astype(np.float32)
    keys = rng.normal(size=(2, 2, 1, 1, 8, 4)).astype(np.float32)
    scores = rng.normal(size=(2, 2, 1, 1, 8)).astype(np.float32)
    masks = selector.build_query_key_exact_rerank_masks(
        queries,
        np.array([0, 1], dtype=np.int64),
        keys,
        scores,
        rank=2,
        pool_size=4,
    )
    assert masks.shape == (2, 4, 1, 1, 8)
    assert np.all(masks[:, 2:] == 1)
    assert np.all(np.sum(masks[:, :2], axis=-1) == 4)


def test_cross_split_masks_do_not_require_evaluation_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(selector, "_RECORDS", 2)
    monkeypatch.setattr(selector, "_READS", 2)
    monkeypatch.setattr(selector, "_LAYERS", 1)
    monkeypatch.setattr(selector, "_HEADS", 1)
    monkeypatch.setattr(selector, "_HEAD_DIMENSION", 4)
    monkeypatch.setattr(selector, "_CANDIDATES", 8)
    monkeypatch.setattr(selector, "_POSITIONS", 4)
    monkeypatch.setattr(selector.qk.full, "_READ_POSITIONS", np.array([0, 1]))
    rng = np.random.default_rng(13)
    train_queries = rng.normal(size=(2, 2, 1, 1, 4)).astype(np.float32)
    train_keys = rng.normal(size=(2, 2, 1, 1, 8, 4)).astype(np.float32)
    evaluation_queries = rng.normal(size=(2, 2, 1, 1, 4)).astype(np.float32)
    evaluation_keys = rng.normal(size=(2, 2, 1, 1, 8, 4)).astype(np.float32)
    masks = selector.build_query_key_cross_split_masks(
        train_queries,
        train_keys,
        evaluation_queries,
        evaluation_keys,
        np.array([0, 1], dtype=np.int64),
        rank=2,
        pool_size=4,
    )
    assert masks.shape == (2, 4, 1, 1, 8)
    assert np.all(np.sum(masks[:, :2], axis=-1) == 4)


def test_frozen_pca_basis_replays_cross_split_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(selector, "_RECORDS", 2)
    monkeypatch.setattr(selector, "_READS", 2)
    monkeypatch.setattr(selector, "_LAYERS", 1)
    monkeypatch.setattr(selector, "_HEADS", 1)
    monkeypatch.setattr(selector, "_HEAD_DIMENSION", 4)
    monkeypatch.setattr(selector, "_CANDIDATES", 8)
    monkeypatch.setattr(selector, "_POSITIONS", 4)
    monkeypatch.setattr(selector.qk.full, "_READ_POSITIONS", np.array([0, 1]))
    rng = np.random.default_rng(17)
    train_keys = rng.normal(size=(2, 2, 1, 1, 8, 4)).astype(np.float32)
    evaluation_queries = rng.normal(size=(2, 2, 1, 1, 4)).astype(np.float32)
    evaluation_keys = rng.normal(size=(2, 2, 1, 1, 8, 4)).astype(np.float32)
    positions = np.array([0, 1], dtype=np.int64)
    centers, components = selector.fit_query_key_pca_basis(train_keys, rank=2)
    actual = selector.build_query_key_masks_from_pca_basis(
        evaluation_queries,
        evaluation_keys,
        positions,
        centers,
        components,
        pool_size=4,
    )
    expected = selector.build_query_key_cross_split_masks(
        evaluation_queries,
        train_keys,
        evaluation_queries,
        evaluation_keys,
        positions,
        rank=2,
        pool_size=4,
    )
    np.testing.assert_array_equal(actual, expected)
