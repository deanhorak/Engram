"""Cheap held-out screen for a query-only older-candidate router.

This is intentionally a negative-control experiment.  It fits a small
rank-``r`` ridge map from the pre-attention hidden state to the eight
pre-top-K candidate scores, holding out one complete train record at a time.
The target scores come from the authenticated candidate-QK capture; no native
attention policy is changed and no confirmation data is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from engram.evaluation import (
    olmoe_retrieval_episodic_blockwise_qk as qk,
)
from engram.utils import atomic_json, sha256_file


_HEADS = qk.full._QUERY_HEADS
_HEAD_DIMENSION = qk.full._HEAD_DIMENSION
_LAYERS = qk.full._LAYERS
_RECORDS = qk.full._RECORDS
_READS = len(qk.full._READ_POSITIONS)
_CANDIDATES = qk.full._C28_QK_CANDIDATE_ENTRIES
_POSITIONS = qk.full._POSITIONS


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("candidate selector manifest must be one object")
    return value


def _load_input_norm(manifest: str | Path, digest: str) -> np.ndarray:
    """Load and authenticate input_norm from the inherited train shards."""

    manifest_path = qk.full._checked_file(
        manifest, digest, "candidate selector residual manifest"
    )
    metadata = _read_json(manifest_path)
    shards = metadata.get("shards")
    if not isinstance(shards, list) or len(shards) != _RECORDS:
        raise ValueError("candidate selector residual shard count changed")
    rows: list[np.ndarray] = []
    for index, descriptor in enumerate(shards):
        if (
            not isinstance(descriptor, dict)
            or descriptor.get("record_index") != index
            or descriptor.get("format") != "safetensors"
            or descriptor.get("shape") != [_READS, _LAYERS, _HEADS * _HEAD_DIMENSION]
            or "input_norm" not in descriptor.get("keys", [])
        ):
            raise ValueError("candidate selector residual descriptor changed")
        source = qk.full._checked_file(
            manifest_path.parent / str(descriptor.get("file", "")),
            str(descriptor.get("file_sha256", "")),
            "candidate selector residual shard",
        )
        try:
            from safetensors.numpy import load_file
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("candidate selector requires safetensors") from error
        loaded = load_file(source)
        value = np.ascontiguousarray(loaded["input_norm"], dtype=np.float32)
        if value.shape != (_READS, _LAYERS, _HEADS * _HEAD_DIMENSION):
            raise ValueError("candidate selector input_norm shape changed")
        if not np.isfinite(value).all():
            raise ValueError("candidate selector input_norm is non-finite")
        expected_tensor_sha256 = descriptor.get("tensor_sha256", {}).get(
            "input_norm"
        )
        actual_tensor_sha256 = hashlib.sha256(
            value.tobytes(order="C")
        ).hexdigest()
        if actual_tensor_sha256 != expected_tensor_sha256:
            raise ValueError("candidate selector input_norm hash changed")
        rows.append(value)
    return np.ascontiguousarray(np.stack(rows))


def evaluate_query_only_router(
    input_norm: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    rank: int = 16,
    lambdas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> dict[str, Any]:
    """Run deterministic leave-one-record-out ridge routing."""

    hidden = np.ascontiguousarray(input_norm, dtype=np.float64)
    scores = np.ascontiguousarray(candidate_scores, dtype=np.float64)
    expected_hidden = (_RECORDS, _READS, _LAYERS, _HEADS * _HEAD_DIMENSION)
    expected_scores = (_RECORDS, _READS, _LAYERS, _HEADS, _CANDIDATES)
    if hidden.shape != expected_hidden or scores.shape != expected_scores:
        raise ValueError("candidate selector training shapes changed")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank <= 0
        or rank > _HEAD_DIMENSION
        or not np.isfinite(hidden).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("candidate selector training inputs are invalid")
    values = tuple(float(value) for value in lambdas)
    if not values or any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("candidate selector ridge grid is invalid")

    features = hidden.reshape(_RECORDS, _READS, _LAYERS, _HEADS, _HEAD_DIMENSION)
    results: dict[str, list[float]] = {
        str(value): [] for value in values
    }
    mass_results: dict[str, list[float]] = {str(value): [] for value in values}
    for held_out in range(_RECORDS):
        training = np.arange(_RECORDS) != held_out
        validation = ~training
        for layer in range(_LAYERS):
            for head in range(_HEADS):
                x_train = features[training, :, layer, head].reshape(-1, _HEAD_DIMENSION)
                x_valid = features[validation, :, layer, head].reshape(-1, _HEAD_DIMENSION)
                y_train = scores[training, :, layer, head].reshape(-1, _CANDIDATES)
                y_valid = scores[validation, :, layer, head].reshape(-1, _CANDIDATES)
                center = x_train.mean(axis=0)
                _, _, right = np.linalg.svd(
                    x_train - center, full_matrices=False
                )
                basis = right[:rank].T
                projected = (x_train - center) @ basis
                projected_valid = (x_valid - center) @ basis
                design = np.concatenate(
                    (projected, np.ones((projected.shape[0], 1))), axis=1
                )
                design_valid = np.concatenate(
                    (projected_valid, np.ones((projected_valid.shape[0], 1))), axis=1
                )
                gram = design.T @ design
                for value in values:
                    regularizer = np.eye(rank + 1)
                    regularizer[-1, -1] = 0.0
                    weights = np.linalg.solve(
                        gram + value * regularizer,
                        design.T @ y_train,
                    )
                    predicted = design_valid @ weights
                    for row, target in zip(predicted, y_valid, strict=True):
                        oracle = np.argsort(-target, kind="stable")[:4]
                        selected = np.argsort(-row, kind="stable")[:4]
                        results[str(value)].append(
                            float(np.intersect1d(oracle, selected).size / 4.0)
                        )
                        shifted = target - np.max(target)
                        mass = np.exp(shifted)
                        mass /= np.sum(mass, dtype=np.float64)
                        mass_results[str(value)].append(
                            float(np.sum(mass[selected], dtype=np.float64))
                        )
    summary: dict[str, Any] = {}
    for value in values:
        key = str(value)
        recall = np.asarray(results[key], dtype=np.float64)
        mass = np.asarray(mass_results[key], dtype=np.float64)
        summary[key] = {
            "candidate_membership_recall_mean": float(np.mean(recall)),
            "candidate_membership_recall_p10": float(np.quantile(recall, 0.10)),
            "exact_top4_fraction": float(np.mean(recall >= 1.0)),
            "oracle_mass_retention_mean": float(np.mean(mass)),
            "oracle_mass_retention_p10": float(np.quantile(mass, 0.10)),
        }
    return {
        "schema_version": 1,
        "experiment": "olmoe_q7_retrieval_episodic_query_only_candidate_router_screen",
        "router": {
            "features": "pre_attention_hidden_head_slices",
            "model": "record-held-out-rank-ridge",
            "rank": rank,
            "lambdas": list(values),
            "folds": _RECORDS,
        },
        "results": summary,
        "feature_fidelity_only": True,
        "causal_policy_changed": False,
        "confirmation_split_opened": False,
    }


def evaluate_query_key_router(
    input_norm: np.ndarray,
    candidate_keys: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    rank: int = 16,
    lambdas: Sequence[float] = (0.1, 1.0, 10.0),
) -> dict[str, Any]:
    """Fit a held-out bilinear query/key compatibility selector.

    A rank-selected diagonal compatibility map multiplies centered hidden and
    post-RoPE key coordinates, then predicts pre-top-K candidate scores with a
    ridge head.  The diagonal form keeps the screen cheap enough to run on
    CPU while still requiring both query and key side information.
    """

    hidden = np.ascontiguousarray(input_norm, dtype=np.float64)
    keys = np.ascontiguousarray(candidate_keys, dtype=np.float64)
    scores = np.ascontiguousarray(candidate_scores, dtype=np.float64)
    if (
        hidden.shape != (_RECORDS, _READS, _LAYERS, _HEADS * _HEAD_DIMENSION)
        or keys.shape
        != (_RECORDS, _READS, _LAYERS, _HEADS, _CANDIDATES, _HEAD_DIMENSION)
        or scores.shape != (_RECORDS, _READS, _LAYERS, _HEADS, _CANDIDATES)
    ):
        raise ValueError("query-key selector training shapes changed")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank <= 0
        or rank > _HEAD_DIMENSION
        or not np.isfinite(hidden).all()
        or not np.isfinite(keys).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("query-key selector training inputs are invalid")
    values = tuple(float(value) for value in lambdas)
    if not values or any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("query-key selector ridge grid is invalid")
    hidden_heads = hidden.reshape(
        _RECORDS, _READS, _LAYERS, _HEADS, _HEAD_DIMENSION
    )
    results: dict[str, list[float]] = {str(value): [] for value in values}
    mass_results: dict[str, list[float]] = {str(value): [] for value in values}
    exact_ceiling: list[float] = []
    for held_out in range(_RECORDS):
        training = np.arange(_RECORDS) != held_out
        for layer in range(_LAYERS):
            for head in range(_HEADS):
                x_train = hidden_heads[training, :, layer, head]
                x_valid = hidden_heads[held_out, :, layer, head]
                k_train = keys[training, :, layer, head]
                k_valid = keys[held_out, :, layer, head]
                y_train = scores[training, :, layer, head]
                y_valid = scores[held_out, :, layer, head]
                x_center = x_train.reshape(-1, _HEAD_DIMENSION).mean(axis=0)
                k_center = k_train.reshape(-1, _HEAD_DIMENSION).mean(axis=0)
                key_variance = np.var(
                    k_train.reshape(-1, _HEAD_DIMENSION), axis=0
                )
                dimensions = np.argsort(-key_variance, kind="stable")[:rank]
                x_train_projected = x_train[..., dimensions] - x_center[dimensions]
                x_valid_projected = x_valid[..., dimensions] - x_center[dimensions]
                k_train_projected = k_train[..., dimensions] - k_center[dimensions]
                k_valid_projected = k_valid[..., dimensions] - k_center[dimensions]
                train_features = (
                    x_train_projected[:, :, None, :]
                    * k_train_projected[:, :, :, :]
                ).reshape(-1, rank)
                valid_features = (
                    x_valid_projected[:, None, :]
                    * k_valid_projected
                ).reshape(-1, rank)
                design = np.concatenate(
                    (train_features, np.ones((train_features.shape[0], 1))), axis=1
                )
                design_valid = np.concatenate(
                    (valid_features, np.ones((valid_features.shape[0], 1))), axis=1
                )
                gram = design.T @ design
                target = y_train.reshape(-1)
                valid_target = y_valid
                exact_scores = valid_target
                exact_ceiling.extend(
                    [1.0] * int(np.prod(exact_scores.shape[:-1]))
                )
                for value in values:
                    regularizer = np.eye(rank + 1)
                    regularizer[-1, -1] = 0.0
                    weights = np.linalg.solve(
                        gram + value * regularizer,
                        design.T @ target,
                    )
                    predicted = (
                        design_valid @ weights
                    ).reshape(_READS, _CANDIDATES)
                    selected = np.argsort(-predicted, axis=-1, kind="stable")[:, :4]
                    for row, target_row, selected_row in zip(
                        predicted, valid_target, selected, strict=True
                    ):
                        oracle = np.argsort(-target_row, kind="stable")[:4]
                        results[str(value)].append(
                            float(np.intersect1d(oracle, selected_row).size / 4.0)
                        )
                        shifted = target_row - np.max(target_row)
                        mass = np.exp(shifted)
                        mass /= np.sum(mass, dtype=np.float64)
                        mass_results[str(value)].append(
                            float(np.sum(mass[selected_row], dtype=np.float64))
                        )
    summary: dict[str, Any] = {}
    for value in values:
        key = str(value)
        recall = np.asarray(results[key], dtype=np.float64)
        mass = np.asarray(mass_results[key], dtype=np.float64)
        summary[key] = {
            "candidate_membership_recall_mean": float(np.mean(recall)),
            "candidate_membership_recall_p10": float(np.quantile(recall, 0.10)),
            "exact_top4_fraction": float(np.mean(recall >= 1.0)),
            "oracle_mass_retention_mean": float(np.mean(mass)),
            "oracle_mass_retention_p10": float(np.quantile(mass, 0.10)),
        }
    return {
        "schema_version": 1,
        "experiment": "olmoe_q7_retrieval_episodic_query_key_candidate_router_screen",
        "router": {
            "features": "rank-r_centered_hidden_times_key_coordinates",
            "model": "record-held-out-diagonal-bilinear-ridge",
            "rank": rank,
            "lambdas": list(values),
            "folds": _RECORDS,
        },
        "results": summary,
        "exact_score_ceiling_membership_recall_mean": float(np.mean(exact_ceiling)),
        "feature_fidelity_only": True,
        "causal_policy_changed": False,
        "confirmation_split_opened": False,
    }


def _load_authenticated_query_features(
    path: str | Path,
    digest: str,
) -> tuple[np.ndarray, np.ndarray]:
    source = qk.full._checked_file(path, digest, "query feature artifact")
    try:
        from safetensors import safe_open
        from safetensors.numpy import load_file
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("query-key screen requires safetensors") from error
    with safe_open(source, framework="numpy") as handle:
        if sorted(handle.keys()) != ["positions", "post_qnorm_pre_rope_queries"]:
            raise ValueError("query feature artifact keys changed")
    tensors = load_file(source)
    queries = np.ascontiguousarray(tensors["post_qnorm_pre_rope_queries"])
    positions = np.ascontiguousarray(tensors["positions"])
    if (
        queries.dtype != np.float32
        or queries.shape != (_RECORDS, _READS, _LAYERS, _HEADS, _HEAD_DIMENSION)
        or positions.dtype != np.int64
        or positions.shape != (_READS,)
        or not np.array_equal(positions, qk.full._READ_POSITIONS)
        or not np.isfinite(queries).all()
    ):
        raise ValueError("query feature artifact shape or values changed")
    return queries, positions


def evaluate_actual_query_key_recall(
    queries_pre_rope: np.ndarray,
    positions: np.ndarray,
    candidate_keys: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    ranks: Sequence[int] = (4, 8, 16, 32, 64),
) -> dict[str, Any]:
    """Measure held-out candidate recall using the actual per-head queries."""

    queries = np.ascontiguousarray(queries_pre_rope, dtype=np.float64)
    positions_array = np.ascontiguousarray(positions)
    keys = np.ascontiguousarray(candidate_keys, dtype=np.float64)
    scores = np.ascontiguousarray(candidate_scores, dtype=np.float64)
    expected_query_shape = (_RECORDS, _READS, _LAYERS, _HEADS, _HEAD_DIMENSION)
    expected_key_shape = (
        _RECORDS,
        _READS,
        _LAYERS,
        _HEADS,
        _CANDIDATES,
        _HEAD_DIMENSION,
    )
    expected_score_shape = (_RECORDS, _READS, _LAYERS, _HEADS, _CANDIDATES)
    rank_values = tuple(int(value) for value in ranks)
    if (
        queries.shape != expected_query_shape
        or positions_array.shape != (_READS,)
        or not np.array_equal(positions_array, qk.full._READ_POSITIONS)
        or keys.shape != expected_key_shape
        or scores.shape != expected_score_shape
        or not rank_values
        or any(value <= 0 or value > _HEAD_DIMENSION for value in rank_values)
        or len(set(rank_values)) != len(rank_values)
        or not np.isfinite(queries).all()
        or not np.isfinite(keys).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("actual query-key recall inputs are invalid")
    half = _HEAD_DIMENSION // 2
    frequencies = np.power(
        10000.0,
        -2.0 * np.arange(half, dtype=np.float64) / _HEAD_DIMENSION,
    )
    rotated = queries.copy()
    for read, position in enumerate(positions_array):
        cosine = np.cos(float(position) * frequencies)
        sine = np.sin(float(position) * frequencies)
        first = queries[:, read, :, :, :half].copy()
        second = queries[:, read, :, :, half:].copy()
        rotated[:, read, :, :, :half] = first * cosine - second * sine
        rotated[:, read, :, :, half:] = second * cosine + first * sine
    result_rows: dict[int, list[float]] = {rank: [] for rank in rank_values}
    exact_rows: dict[int, list[bool]] = {rank: [] for rank in rank_values}
    mass_rows: dict[int, list[float]] = {rank: [] for rank in rank_values}
    dense_recall: list[float] = []
    scale = 1.0 / np.sqrt(float(_HEAD_DIMENSION))
    for held_out in range(_RECORDS):
        training = np.arange(_RECORDS) != held_out
        for layer in range(_LAYERS):
            for head in range(_HEADS):
                train_matrix = keys[training, :, layer, head].reshape(
                    -1, _HEAD_DIMENSION
                )
                center = train_matrix.mean(axis=0)
                centered = train_matrix - center
                covariance = centered.T @ centered
                eigenvalues, basis = np.linalg.eigh(covariance)
                basis = basis[:, np.argsort(eigenvalues)[::-1]]
                query = rotated[held_out, :, layer, head]
                original = keys[held_out, :, layer, head]
                oracle = np.argsort(
                    -scores[held_out, :, layer, head], axis=-1, kind="stable"
                )[:, :4]
                dense_scores = np.einsum(
                    "rd,rcd->rc", query, original, optimize=True
                ) * scale
                dense_selected = np.argsort(
                    -dense_scores, axis=-1, kind="stable"
                )[:, :4]
                dense_recall.extend(
                    [
                        float(np.intersect1d(expected, selected).size / 4.0)
                        for expected, selected in zip(
                            oracle, dense_selected, strict=True
                        )
                    ]
                )
                shifted = scores[held_out, :, layer, head]
                shifted = shifted - np.max(shifted, axis=-1, keepdims=True)
                oracle_mass = np.exp(shifted)
                oracle_mass /= np.sum(oracle_mass, axis=-1, keepdims=True)
                for rank in rank_values:
                    components = basis[:, :rank]
                    reconstructed = (
                        (original - center) @ components
                    ) @ components.T + center
                    predicted = np.einsum(
                        "rd,rcd->rc", query, reconstructed, optimize=True
                    ) * scale
                    selected = np.argsort(-predicted, axis=-1, kind="stable")[:, :4]
                    hits = [
                        float(np.intersect1d(expected, chosen).size / 4.0)
                        for expected, chosen in zip(oracle, selected, strict=True)
                    ]
                    result_rows[rank].extend(hits)
                    exact_rows[rank].extend(value >= 1.0 for value in hits)
                    mass_rows[rank].extend(
                        np.take_along_axis(
                            oracle_mass, selected, axis=-1
                        ).sum(axis=-1)
                    )
    return {
        "schema_version": 1,
        "experiment": "olmoe_q7_retrieval_episodic_actual_query_key_rank_recall",
        "query": {
            "representation": "authenticated_post_qnorm_pre_rope_query",
            "rope_theta": 10000.0,
            "rope_applied_for_scoring": True,
            "positions": positions_array.tolist(),
        },
        "ranks": list(rank_values),
        "dense_exact_query_recall_mean": float(np.mean(dense_recall)),
        "results": {
            str(rank): {
                "candidate_membership_recall_mean": float(
                    np.mean(result_rows[rank])
                ),
                "candidate_membership_recall_p10": float(
                    np.quantile(result_rows[rank], 0.10)
                ),
                "exact_top4_fraction": float(np.mean(exact_rows[rank])),
                "oracle_mass_retention_mean": float(np.mean(mass_rows[rank])),
                "oracle_mass_retention_p10": float(
                    np.quantile(mass_rows[rank], 0.10)
                ),
            }
            for rank in rank_values
        },
        "record_held_out": True,
        "feature_fidelity_only": True,
        "causal_policy_changed": False,
        "confirmation_split_opened": False,
    }


def evaluate_query_key_exact_rerank(
    queries_pre_rope: np.ndarray,
    positions: np.ndarray,
    candidate_keys: np.ndarray,
    candidate_values: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    ranks: Sequence[int] = (8, 16, 32, 64),
    pool_sizes: Sequence[int] = (4, 6, 8),
) -> dict[str, Any]:
    """Evaluate compressed candidate generation followed by exact reranking.

    The PCA basis is fit on the seven non-held-out records for each layer/head.
    The compressed score chooses a candidate pool; exact QK scores then select
    the final top-K inside that pool.  Candidate values are authenticated input
    to the downstream causal replay, but this screen intentionally reports
    membership/mass only and does not claim hidden-state parity.
    """

    queries = np.ascontiguousarray(queries_pre_rope, dtype=np.float64)
    keys = np.ascontiguousarray(candidate_keys, dtype=np.float64)
    values = np.ascontiguousarray(candidate_values, dtype=np.float64)
    scores = np.ascontiguousarray(candidate_scores, dtype=np.float64)
    expected_query_shape = (_RECORDS, _READS, _LAYERS, _HEADS, _HEAD_DIMENSION)
    expected_key_shape = (
        _RECORDS,
        _READS,
        _LAYERS,
        _HEADS,
        _CANDIDATES,
        _HEAD_DIMENSION,
    )
    expected_score_shape = (_RECORDS, _READS, _LAYERS, _HEADS, _CANDIDATES)
    rank_values = tuple(int(value) for value in ranks)
    pool_values = tuple(int(value) for value in pool_sizes)
    if (
        queries.shape != expected_query_shape
        or positions.shape != (_READS,)
        or not np.array_equal(positions, qk.full._READ_POSITIONS)
        or keys.shape != expected_key_shape
        or values.shape != expected_key_shape
        or scores.shape != expected_score_shape
        or not rank_values
        or not pool_values
        or any(value <= 0 or value > _HEAD_DIMENSION for value in rank_values)
        or any(value <= 0 or value > _CANDIDATES for value in pool_values)
        or not np.isfinite(queries).all()
        or not np.isfinite(keys).all()
        or not np.isfinite(values).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("query-key exact-rerank inputs are invalid")
    half = _HEAD_DIMENSION // 2
    frequencies = np.power(
        10000.0,
        -2.0 * np.arange(half, dtype=np.float64) / _HEAD_DIMENSION,
    )
    rotated = queries.copy()
    for read, position in enumerate(np.asarray(positions)):
        cosine = np.cos(float(position) * frequencies)
        sine = np.sin(float(position) * frequencies)
        first = queries[:, read, :, :, :half].copy()
        second = queries[:, read, :, :, half:].copy()
        rotated[:, read, :, :, :half] = first * cosine - second * sine
        rotated[:, read, :, :, half:] = second * cosine + first * sine
    scale = 1.0 / np.sqrt(float(_HEAD_DIMENSION))
    rows: dict[str, list[float]] = {
        f"rank{rank}_pool{pool}": []
        for rank in rank_values
        for pool in pool_values
    }
    mass_rows = {name: [] for name in rows}
    exact_rows = {name: [] for name in rows}
    for held_out in range(_RECORDS):
        training = np.arange(_RECORDS) != held_out
        for layer in range(_LAYERS):
            for head in range(_HEADS):
                train_matrix = keys[training, :, layer, head].reshape(
                    -1, _HEAD_DIMENSION
                )
                center = train_matrix.mean(axis=0)
                centered = train_matrix - center
                covariance = centered.T @ centered
                eigenvalues, basis = np.linalg.eigh(covariance)
                basis = basis[:, np.argsort(eigenvalues)[::-1]]
                original = keys[held_out, :, layer, head]
                query = rotated[held_out, :, layer, head]
                exact = scores[held_out, :, layer, head]
                oracle = np.argsort(-exact, axis=-1, kind="stable")[:, :4]
                shifted = exact - np.max(exact, axis=-1, keepdims=True)
                oracle_mass = np.exp(shifted)
                oracle_mass /= np.sum(oracle_mass, axis=-1, keepdims=True)
                for rank in rank_values:
                    components = basis[:, :rank]
                    reconstructed = (
                        (original - center) @ components
                    ) @ components.T + center
                    predicted = (
                        np.einsum("rd,rcd->rc", query, reconstructed)
                        * scale
                    )
                    for pool in pool_values:
                        name = f"rank{rank}_pool{pool}"
                        candidate_pool = np.argsort(
                            -predicted, axis=-1, kind="stable"
                        )[:, :pool]
                        reranked = np.take_along_axis(
                            candidate_pool,
                            np.argsort(
                                -np.take_along_axis(
                                    exact, candidate_pool, axis=-1
                                ),
                                axis=-1,
                                kind="stable",
                            )[:, :4],
                            axis=-1,
                        )
                        hits = np.asarray(
                            [
                                np.intersect1d(expected, chosen).size / 4.0
                                for expected, chosen in zip(
                                    oracle, reranked, strict=True
                                )
                            ]
                        )
                        rows[name].extend(hits.tolist())
                        exact_rows[name].extend((hits >= 1.0).tolist())
                        mass_rows[name].extend(
                            np.take_along_axis(
                                oracle_mass, reranked, axis=-1
                            ).sum(axis=-1).tolist()
                        )
    return {
        "schema_version": 1,
        "experiment": "olmoe_q7_retrieval_episodic_query_key_exact_rerank",
        "query": {
            "representation": "authenticated_post_qnorm_pre_rope_query",
            "rope_theta": 10000.0,
            "rope_applied_for_scoring": True,
        },
        "candidate_values": {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "authenticated_for_downstream_causal_replay": True,
        },
        "ranks": list(rank_values),
        "pool_sizes": list(pool_values),
        "results": {
            name: {
                "candidate_membership_recall_mean": float(np.mean(rows[name])),
                "candidate_membership_recall_p10": float(
                    np.quantile(rows[name], 0.10)
                ),
                "exact_top4_fraction": float(np.mean(exact_rows[name])),
                # When the reranked slot set equals the native dense top-4,
                # the authenticated candidate-value tensor makes the attention
                # output identical by construction. This is a lower bound,
                # not a claim about later hidden-state parity.
                "causal_attention_output_parity_lower_bound": float(
                    np.mean(exact_rows[name])
                ),
                "oracle_mass_retention_mean": float(np.mean(mass_rows[name])),
                "oracle_mass_retention_p10": float(
                    np.quantile(mass_rows[name], 0.10)
                ),
            }
            for name in rows
        },
        "record_held_out": True,
        "feature_fidelity_only": True,
        "causal_policy_changed": False,
        "confirmation_split_opened": False,
    }


def build_query_key_exact_rerank_masks(
    queries_pre_rope: np.ndarray,
    positions: np.ndarray,
    candidate_keys: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    rank: int = 16,
    pool_size: int = 6,
) -> np.ndarray:
    """Build causal older-slot masks for the native masked replay.

    A centered PCA reconstruction supplies the cheap candidate pool.  The
    native attention kernel then computes exact QK scores inside that pool and
    retains its normal top-4 entries.  Rows before the authenticated read
    positions are all ones, so this function cannot alter prefix behavior.
    The returned tensor is ``[record, position, layer, head, candidate]``.
    """

    queries = np.ascontiguousarray(queries_pre_rope, dtype=np.float64)
    position_values = np.ascontiguousarray(positions, dtype=np.int64)
    keys = np.ascontiguousarray(candidate_keys, dtype=np.float64)
    scores = np.ascontiguousarray(candidate_scores, dtype=np.float64)
    expected_query_shape = (_RECORDS, _READS, _LAYERS, _HEADS, _HEAD_DIMENSION)
    expected_key_shape = (
        _RECORDS,
        _READS,
        _LAYERS,
        _HEADS,
        _CANDIDATES,
        _HEAD_DIMENSION,
    )
    expected_score_shape = (_RECORDS, _READS, _LAYERS, _HEADS, _CANDIDATES)
    if (
        queries.shape != expected_query_shape
        or position_values.shape != (_READS,)
        or not np.array_equal(position_values, qk.full._READ_POSITIONS)
        or keys.shape != expected_key_shape
        or scores.shape != expected_score_shape
        or not (0 < rank <= _HEAD_DIMENSION)
        or not (0 < pool_size <= _CANDIDATES)
        or not np.isfinite(queries).all()
        or not np.isfinite(keys).all()
        or not np.isfinite(scores).all()
        or np.any(position_values < 0)
        or np.any(position_values >= _POSITIONS)
    ):
        raise ValueError("query-key replay mask inputs are invalid")

    half = _HEAD_DIMENSION // 2
    frequencies = np.power(
        10000.0,
        -2.0 * np.arange(half, dtype=np.float64) / _HEAD_DIMENSION,
    )
    rotated = queries.copy()
    for read, position in enumerate(position_values):
        cosine = np.cos(float(position) * frequencies)
        sine = np.sin(float(position) * frequencies)
        first = queries[:, read, :, :, :half].copy()
        second = queries[:, read, :, :, half:].copy()
        rotated[:, read, :, :, :half] = first * cosine - second * sine
        rotated[:, read, :, :, half:] = second * cosine + first * sine

    masks = np.ones(
        (_RECORDS, _POSITIONS, _LAYERS, _HEADS, _CANDIDATES),
        dtype=np.uint8,
    )
    scale = 1.0 / np.sqrt(float(_HEAD_DIMENSION))
    for held_out in range(_RECORDS):
        training = np.arange(_RECORDS) != held_out
        for layer in range(_LAYERS):
            for head in range(_HEADS):
                train_matrix = keys[training, :, layer, head].reshape(
                    -1, _HEAD_DIMENSION
                )
                center = train_matrix.mean(axis=0)
                centered = train_matrix - center
                covariance = centered.T @ centered
                eigenvalues, basis = np.linalg.eigh(covariance)
                basis = basis[:, np.argsort(eigenvalues)[::-1]]
                components = basis[:, :rank]
                original = keys[held_out, :, layer, head]
                reconstructed = (
                    (original - center) @ components
                ) @ components.T + center
                predicted = (
                    np.einsum(
                        "rd,rcd->rc",
                        rotated[held_out, :, layer, head],
                        reconstructed,
                        optimize=True,
                    )
                    * scale
                )
                pools = np.argsort(
                    -predicted,
                    axis=-1,
                    kind="stable",
                )[:, :pool_size]
                for read, position in enumerate(position_values):
                    masks[held_out, int(position), layer, head, :] = 0
                    masks[held_out, int(position), layer, head, pools[read]] = 1
    return np.ascontiguousarray(masks)


def build_query_key_cross_split_masks(
    train_queries_pre_rope: np.ndarray,
    train_candidate_keys: np.ndarray,
    evaluation_queries_pre_rope: np.ndarray,
    evaluation_candidate_keys: np.ndarray,
    positions: np.ndarray,
    *,
    rank: int = 16,
    pool_size: int = 6,
) -> np.ndarray:
    """Apply a train-fitted PCA pool to a distinct evaluation population.

    Unlike :func:`build_query_key_exact_rerank_masks`, this function never
    uses evaluation records while fitting the basis.  It is the generalization
    boundary for the native development replay.
    """

    train_queries = np.ascontiguousarray(train_queries_pre_rope, dtype=np.float64)
    train_keys = np.ascontiguousarray(train_candidate_keys, dtype=np.float64)
    evaluation_queries = np.ascontiguousarray(
        evaluation_queries_pre_rope, dtype=np.float64
    )
    evaluation_keys = np.ascontiguousarray(
        evaluation_candidate_keys, dtype=np.float64
    )
    position_values = np.ascontiguousarray(positions, dtype=np.int64)
    expected_query_tail = (_READS, _LAYERS, _HEADS, _HEAD_DIMENSION)
    expected_key_tail = (
        _READS,
        _LAYERS,
        _HEADS,
        _CANDIDATES,
        _HEAD_DIMENSION,
    )
    if (
        train_queries.ndim != 5
        or evaluation_queries.ndim != 5
        or train_queries.shape[1:] != expected_query_tail
        or evaluation_queries.shape[1:] != expected_query_tail
        or train_keys.ndim != 6
        or evaluation_keys.ndim != 6
        or train_keys.shape[1:] != expected_key_tail
        or evaluation_keys.shape[1:] != expected_key_tail
        or train_queries.shape[0] == 0
        or evaluation_queries.shape[0] == 0
        or position_values.shape != (_READS,)
        or not np.array_equal(position_values, qk.full._READ_POSITIONS)
        or not (0 < rank <= _HEAD_DIMENSION)
        or not (0 < pool_size <= _CANDIDATES)
        or not np.isfinite(train_queries).all()
        or not np.isfinite(train_keys).all()
        or not np.isfinite(evaluation_queries).all()
        or not np.isfinite(evaluation_keys).all()
        or np.any(position_values < 0)
        or np.any(position_values >= _POSITIONS)
    ):
        raise ValueError("cross-split query-key replay mask inputs are invalid")

    half = _HEAD_DIMENSION // 2
    frequencies = np.power(
        10000.0,
        -2.0 * np.arange(half, dtype=np.float64) / _HEAD_DIMENSION,
    )
    rotated = evaluation_queries.copy()
    for read, position in enumerate(position_values):
        cosine = np.cos(float(position) * frequencies)
        sine = np.sin(float(position) * frequencies)
        first = evaluation_queries[:, read, :, :, :half].copy()
        second = evaluation_queries[:, read, :, :, half:].copy()
        rotated[:, read, :, :, :half] = first * cosine - second * sine
        rotated[:, read, :, :, half:] = second * cosine + first * sine

    masks = np.ones(
        (
            evaluation_queries.shape[0],
            _POSITIONS,
            _LAYERS,
            _HEADS,
            _CANDIDATES,
        ),
        dtype=np.uint8,
    )
    scale = 1.0 / np.sqrt(float(_HEAD_DIMENSION))
    for layer in range(_LAYERS):
        for head in range(_HEADS):
            train_matrix = train_keys[:, :, layer, head].reshape(
                -1, _HEAD_DIMENSION
            )
            center = train_matrix.mean(axis=0)
            centered = train_matrix - center
            covariance = centered.T @ centered
            eigenvalues, basis = np.linalg.eigh(covariance)
            basis = basis[:, np.argsort(eigenvalues)[::-1]]
            components = basis[:, :rank]
            original = evaluation_keys[:, :, layer, head]
            reconstructed = (
                (original - center) @ components
            ) @ components.T + center
            predicted = (
                np.einsum(
                    "nrd,nrcd->nrc",
                    rotated[:, :, layer, head],
                    reconstructed,
                    optimize=True,
                )
                * scale
            )
            pools = np.argsort(
                -predicted,
                axis=-1,
                kind="stable",
            )[:, :, :pool_size]
            for record in range(evaluation_queries.shape[0]):
                for read, position in enumerate(position_values):
                    masks[record, int(position), layer, head, :] = 0
                    masks[record, int(position), layer, head, pools[record, read]] = 1
    return np.ascontiguousarray(masks)


def run_screen(
    *,
    candidate_manifest: str | Path,
    candidate_manifest_sha256: str,
    residual_manifest: str | Path,
    residual_manifest_sha256: str,
    rank: int,
    lambdas: Sequence[float],
) -> dict[str, Any]:
    candidates, _ = qk.full.load_stacked_full_visible_qk_candidate_trace(
        candidate_manifest,
        candidate_manifest_sha256,
    )
    input_norm = _load_input_norm(residual_manifest, residual_manifest_sha256)
    report = evaluate_query_only_router(
        input_norm,
        np.sum(candidates, axis=-1),
        rank=rank,
        lambdas=lambdas,
    )
    report["candidate_manifest"] = {
        "path": str(Path(candidate_manifest).resolve()),
        "sha256": candidate_manifest_sha256.lower(),
    }
    report["residual_manifest"] = {
        "path": str(Path(residual_manifest).resolve()),
        "sha256": residual_manifest_sha256.lower(),
    }
    return report


def run_query_key_screen(
    *,
    candidate_key_manifest: str | Path,
    candidate_key_manifest_sha256: str,
    candidate_manifest: str | Path,
    candidate_manifest_sha256: str,
    residual_manifest: str | Path,
    residual_manifest_sha256: str,
    rank: int,
    lambdas: Sequence[float],
) -> dict[str, Any]:
    keys, key_metadata = qk.full.load_stacked_full_visible_qk_candidate_key_trace(
        candidate_key_manifest,
        candidate_key_manifest_sha256,
    )
    candidates, candidate_metadata = qk.full.load_stacked_full_visible_qk_candidate_trace(
        candidate_manifest,
        candidate_manifest_sha256,
    )
    # The key trace was captured under the refreshed evaluator source
    # inventory, while the earlier candidate-score trace is an immutable
    # predecessor artifact.  Require shard-level provenance equality rather
    # than rejecting that intentional source-inventory revision outright.
    qk._cross_check_shards(candidate_metadata, key_metadata)
    input_norm = _load_input_norm(residual_manifest, residual_manifest_sha256)
    report = evaluate_query_key_router(
        input_norm,
        keys,
        np.sum(candidates, axis=-1),
        rank=rank,
        lambdas=lambdas,
    )
    report["candidate_key_manifest"] = {
        "path": str(Path(candidate_key_manifest).resolve()),
        "sha256": candidate_key_manifest_sha256.lower(),
    }
    report["candidate_manifest"] = {
        "path": str(Path(candidate_manifest).resolve()),
        "sha256": candidate_manifest_sha256.lower(),
    }
    report["residual_manifest"] = {
        "path": str(Path(residual_manifest).resolve()),
        "sha256": residual_manifest_sha256.lower(),
    }
    report["protocol_bindings"] = {
        "candidate": candidate_metadata.get("protocol"),
        "candidate_keys": key_metadata.get("protocol"),
        "shard_provenance_cross_bound": True,
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--residual-manifest", required=True)
    parser.add_argument("--residual-manifest-sha256", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--query-key", action="store_true")
    parser.add_argument("--actual-query-key", action="store_true")
    parser.add_argument("--query-features")
    parser.add_argument("--query-features-sha256")
    parser.add_argument("--candidate-key-manifest")
    parser.add_argument("--candidate-key-manifest-sha256")
    parser.add_argument("--candidate-value-manifest")
    parser.add_argument("--candidate-value-manifest-sha256")
    parser.add_argument("--exact-rerank", action="store_true")
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.exact_rerank:
        if not args.candidate_key_manifest or not args.candidate_key_manifest_sha256:
            raise ValueError(
                "--exact-rerank requires --candidate-key-manifest and its digest"
            )
        if not args.candidate_value_manifest or not args.candidate_value_manifest_sha256:
            raise ValueError(
                "--exact-rerank requires --candidate-value-manifest and its digest"
            )
        if not args.query_features or not args.query_features_sha256:
            raise ValueError(
                "--exact-rerank requires --query-features and its digest"
            )
        keys, key_metadata = qk.full.load_stacked_full_visible_qk_candidate_key_trace(
            args.candidate_key_manifest,
            args.candidate_key_manifest_sha256,
        )
        values, value_metadata = (
            qk.full.load_stacked_full_visible_qk_candidate_value_trace(
                args.candidate_value_manifest,
                args.candidate_value_manifest_sha256,
            )
        )
        candidates, candidate_metadata = qk.full.load_stacked_full_visible_qk_candidate_trace(
            args.candidate_manifest,
            args.candidate_manifest_sha256,
        )
        queries, positions = _load_authenticated_query_features(
            args.query_features,
            args.query_features_sha256,
        )
        qk._cross_check_shards(candidate_metadata, key_metadata)
        report = evaluate_query_key_exact_rerank(
            queries,
            positions,
            keys,
            values,
            np.sum(candidates, axis=-1),
        )
        report["query_features"] = {
            "path": str(Path(args.query_features).resolve()),
            "sha256": args.query_features_sha256.lower(),
        }
        report["candidate_key_manifest"] = {
            "path": str(Path(args.candidate_key_manifest).resolve()),
            "sha256": args.candidate_key_manifest_sha256.lower(),
        }
        report["candidate_value_manifest"] = {
            "path": str(Path(args.candidate_value_manifest).resolve()),
            "sha256": args.candidate_value_manifest_sha256.lower(),
        }
        report["candidate_manifest"] = {
            "path": str(Path(args.candidate_manifest).resolve()),
            "sha256": args.candidate_manifest_sha256.lower(),
        }
        report["protocol_bindings"] = {
            "candidate": candidate_metadata.get("protocol"),
            "candidate_keys": key_metadata.get("protocol"),
            "candidate_values": value_metadata.get("protocol"),
            "shard_provenance_cross_bound": True,
        }
    elif args.actual_query_key:
        if not args.candidate_key_manifest or not args.candidate_key_manifest_sha256:
            raise ValueError(
                "--actual-query-key requires --candidate-key-manifest and its digest"
            )
        if not args.query_features or not args.query_features_sha256:
            raise ValueError(
                "--actual-query-key requires --query-features and its digest"
            )
        keys, key_metadata = qk.full.load_stacked_full_visible_qk_candidate_key_trace(
            args.candidate_key_manifest,
            args.candidate_key_manifest_sha256,
        )
        candidates, candidate_metadata = qk.full.load_stacked_full_visible_qk_candidate_trace(
            args.candidate_manifest,
            args.candidate_manifest_sha256,
        )
        queries, positions = _load_authenticated_query_features(
            args.query_features,
            args.query_features_sha256,
        )
        qk._cross_check_shards(candidate_metadata, key_metadata)
        report = evaluate_actual_query_key_recall(
            queries,
            positions,
            keys,
            np.sum(candidates, axis=-1),
            ranks=(4, 8, 16, 32, 64),
        )
        report["query_features"] = {
            "path": str(Path(args.query_features).resolve()),
            "sha256": args.query_features_sha256.lower(),
        }
        report["candidate_key_manifest"] = {
            "path": str(Path(args.candidate_key_manifest).resolve()),
            "sha256": args.candidate_key_manifest_sha256.lower(),
        }
        report["candidate_manifest"] = {
            "path": str(Path(args.candidate_manifest).resolve()),
            "sha256": args.candidate_manifest_sha256.lower(),
        }
    elif args.query_key:
        if not args.candidate_key_manifest or not args.candidate_key_manifest_sha256:
            raise ValueError(
                "--query-key requires --candidate-key-manifest and its digest"
            )
        report = run_query_key_screen(
            candidate_key_manifest=args.candidate_key_manifest,
            candidate_key_manifest_sha256=args.candidate_key_manifest_sha256,
            candidate_manifest=args.candidate_manifest,
            candidate_manifest_sha256=args.candidate_manifest_sha256,
            residual_manifest=args.residual_manifest,
            residual_manifest_sha256=args.residual_manifest_sha256,
            rank=args.rank,
            lambdas=args.lambdas,
        )
    else:
        report = run_screen(
            candidate_manifest=args.candidate_manifest,
            candidate_manifest_sha256=args.candidate_manifest_sha256,
            residual_manifest=args.residual_manifest,
            residual_manifest_sha256=args.residual_manifest_sha256,
            rank=args.rank,
            lambdas=args.lambdas,
        )
    output = Path(args.out).expanduser()
    qk.full._reject_confirmation_path(output, "candidate selector report")
    if output.exists() or output.is_symlink():
        raise ValueError("candidate selector report already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(json.dumps({"path": str(output), "sha256": sha256_file(output)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
