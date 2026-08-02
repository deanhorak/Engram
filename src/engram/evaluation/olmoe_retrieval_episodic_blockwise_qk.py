"""Train-only audit for the authenticated C28 blockwise-QK artifact.

The native trace records eight scaled Q/K dot-product bands for each of the
twenty regular and eight episodic entries already visible to attention.  This
module checks that those bands reconstruct the authenticated native masses and
reports the mass retained by score-ranked top-k subsets.  It is deliberately a
feature-fidelity/locality screen: it does not open a confirmation split and it
does not claim causal generation quality.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

import engram.evaluation.olmoe_retrieval_episodic_full_visible_simplex_oracle as full
from engram.utils import atomic_json, sha256_file


_TOP_K = (4, 8, 12, 16, 20, 24, 28)
_CANDIDATE_TOP_K = (1, 2, 3, 4, 5, 6, 8)
_KEY_RANKS = (4, 8, 16, 32, 64)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("blockwise-QK audit input is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("blockwise-QK audit input must be one JSON object")
    return value


def _checked_manifest(value: str | Path, digest: str, label: str) -> tuple[Path, dict[str, Any]]:
    path = full._checked_file(value, digest, label)
    return path, _read_json(path)


def _cross_check_shards(
    qk_manifest: dict[str, Any],
    value_manifest: dict[str, Any],
) -> None:
    qk_shards = qk_manifest.get("shards")
    value_shards = value_manifest.get("shards")
    if (
        not isinstance(qk_shards, list)
        or not isinstance(value_shards, list)
        or len(qk_shards) != full._RECORDS
        or len(value_shards) != full._RECORDS
    ):
        raise ValueError("blockwise-QK audit shard count changed")
    for index, (qk, value) in enumerate(zip(qk_shards, value_shards, strict=True)):
        if (
            not isinstance(qk, dict)
            or not isinstance(value, dict)
            or qk.get("record_index") != index
            or value.get("record_index") != index
            or qk.get("record_id") != value.get("record_id")
            or qk.get("source_record_sha256") != value.get("source_record_sha256")
            or qk.get("output_evidence_sha256")
            != value.get("output_evidence_sha256")
            or qk.get("reset_output_evidence_sha256")
            != value.get("reset_output_evidence_sha256")
            or qk.get("schedule_rows_sha256") != value.get("schedule_rows_sha256")
        ):
            raise ValueError(f"blockwise-QK audit shard {index} is not cross-bound")


def _mass_and_kind_from_value_trace(
    manifest: str | Path,
    digest: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    arrays, metadata = full.load_stacked_full_visible_trace(manifest, digest)
    return (
        arrays["regular_entry_mass"],
        arrays["regular_entry_valid_kind"],
        arrays["slot_mass"],
        metadata,
    )


def audit_blockwise_qk_capture(
    *,
    qk_manifest: str | Path,
    qk_manifest_sha256: str,
    value_manifest: str | Path,
    value_manifest_sha256: str,
) -> dict[str, Any]:
    """Return a reproducible train-only QK fidelity and mass-retention report."""

    qk_path, qk_metadata = _checked_manifest(
        qk_manifest,
        qk_manifest_sha256,
        "blockwise-QK manifest",
    )
    value_path, value_metadata = _checked_manifest(
        value_manifest,
        value_manifest_sha256,
        "blockwise-QK value manifest",
    )
    if (
        qk_metadata.get("experiment") != full._QK_CAPTURE_EXPERIMENT
        or qk_metadata.get("confirmation_split_opened") is not False
        or value_metadata.get("confirmation_split_opened") is not False
    ):
        raise ValueError("blockwise-QK audit confirmation contract changed")
    _cross_check_shards(qk_metadata, value_metadata)
    qk, _ = full.load_stacked_full_visible_qk_trace(
        qk_path,
        qk_manifest_sha256,
        protocol=qk_metadata.get("protocol"),
    )
    regular_mass, regular_kind, slot_mass, _ = _mass_and_kind_from_value_trace(
        value_path,
        value_manifest_sha256,
    )
    logits = np.sum(qk, axis=-1, dtype=np.float32)
    native_mass = np.concatenate((regular_mass, slot_mass), axis=-1)
    valid = np.concatenate(
        (regular_kind != full._INVALID_KIND, np.ones_like(slot_mass, dtype=bool)),
        axis=-1,
    )
    errors: list[float] = []
    exact_order_rows = 0
    total_rows = 0
    top_k_rows: dict[int, list[float]] = {k: [] for k in _TOP_K}
    for row in np.ndindex(logits.shape[:4]):
        mask = valid[row]
        scores = logits[row][mask]
        masses = native_mass[row][mask]
        shifted = scores - np.max(scores)
        weights = np.exp(shifted)
        weights /= np.sum(weights, dtype=np.float32)
        errors.extend(np.abs(weights - masses).tolist())
        exact_order_rows += int(
            np.array_equal(
                np.argsort(-scores, kind="stable"),
                np.argsort(-masses, kind="stable"),
            )
        )
        total_rows += 1
        order = np.argsort(-scores, kind="stable")
        for k in _TOP_K:
            top_k_rows[k].append(float(np.sum(masses[order[:k]], dtype=np.float32)))
    error_array = np.asarray(errors, dtype=np.float64)
    retention = {
        str(k): {
            "mean": float(np.mean(values)),
            "p10": float(np.quantile(values, 0.10)),
            "minimum": float(np.min(values)),
        }
        for k, values in top_k_rows.items()
    }
    return {
        "schema_version": 1,
        "experiment": "olmoe_q7_retrieval_episodic_blockwise_qk_train_audit",
        "qk_manifest": {
            "path": str(qk_path),
            "sha256": qk_manifest_sha256.lower(),
        },
        "value_manifest": {
            "path": str(value_path),
            "sha256": value_manifest_sha256.lower(),
        },
        "qk_shape": list(qk.shape),
        "qk_partial_sum_reconstructs_native_mass": {
            "max_abs": float(np.max(error_array)),
            "mean_abs": float(np.mean(error_array)),
            "p99_abs": float(np.quantile(error_array, 0.99)),
        },
        "native_mass_order_recovered": {
            "exact_rows": exact_order_rows,
            "total_rows": total_rows,
            "fraction": float(exact_order_rows / total_rows),
        },
        "score_ranked_mass_retention": retention,
        "feature_fidelity_only": True,
        "candidate_locality_claim": False,
        "confirmation_split_opened": False,
    }


def write_audit_report(
    report: dict[str, Any],
    output: str | Path,
) -> dict[str, Any]:
    path = Path(output).expanduser()
    full._reject_confirmation_path(path, "blockwise-QK audit output")
    if path.exists() or path.is_symlink():
        raise ValueError("blockwise-QK audit output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, report)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "report": report}


def audit_candidate_qk_capture(
    *,
    candidate_manifest: str | Path,
    candidate_manifest_sha256: str,
    selected_qk_manifest: str | Path,
    selected_qk_manifest_sha256: str,
) -> dict[str, Any]:
    """Measure pre-top-K score locality against the selected older entries.

    The result is a feature/locality report only.  A score match does not by
    itself recover native slot identity or authorize causal intervention.
    """

    candidate_path, candidate_metadata = _checked_manifest(
        candidate_manifest,
        candidate_manifest_sha256,
        "blockwise-QK candidate manifest",
    )
    selected_path, selected_metadata = _checked_manifest(
        selected_qk_manifest,
        selected_qk_manifest_sha256,
        "selected blockwise-QK manifest",
    )
    if (
        candidate_metadata.get("experiment") != full._QK_CANDIDATE_CAPTURE_EXPERIMENT
        or selected_metadata.get("experiment") != full._QK_CAPTURE_EXPERIMENT
        or candidate_metadata.get("confirmation_split_opened") is not False
        or selected_metadata.get("confirmation_split_opened") is not False
    ):
        raise ValueError("candidate-QK audit confirmation contract changed")
    _cross_check_shards(candidate_metadata, selected_metadata)
    candidates, _ = full.load_stacked_full_visible_qk_candidate_trace(
        candidate_path,
        candidate_manifest_sha256,
        protocol=candidate_metadata.get("protocol"),
    )
    selected, _ = full.load_stacked_full_visible_qk_trace(
        selected_path,
        selected_qk_manifest_sha256,
    )
    candidate_scores = np.sum(candidates, axis=-1, dtype=np.float32)
    selected_scores = np.sum(selected, axis=-1, dtype=np.float32)[..., 16:20]
    rows: dict[int, list[float]] = {k: [] for k in _CANDIDATE_TOP_K}
    selected_ranks: list[int] = []
    selected_matches: list[float] = []
    for row in candidate_scores.reshape(-1, candidate_scores.shape[-1]):
        shifted = row - np.max(row)
        weights = np.exp(shifted)
        weights /= np.sum(weights, dtype=np.float32)
        order = np.argsort(-row, kind="stable")
        for k in _CANDIDATE_TOP_K:
            rows[k].append(float(np.sum(weights[order[:k]], dtype=np.float32)))
    for candidate_row, selected_row in zip(
        candidate_scores.reshape(-1, candidate_scores.shape[-1]),
        selected_scores.reshape(-1, selected_scores.shape[-1]),
        strict=True,
    ):
        for score in selected_row:
            if not np.isfinite(score):
                continue
            selected_ranks.append(int(1 + np.sum(candidate_row > score + 1.0e-5)))
            selected_matches.append(float(np.min(np.abs(candidate_row - score))))
    retention = {
        str(k): {
            "mean": float(np.mean(values)),
            "p10": float(np.quantile(values, 0.10)),
            "minimum": float(np.min(values)),
        }
        for k, values in rows.items()
    }
    ranks = np.asarray(selected_ranks, dtype=np.int64)
    matches = np.asarray(selected_matches, dtype=np.float64)
    return {
        "schema_version": 1,
        "experiment": "olmoe_q7_retrieval_episodic_blockwise_qk_candidate_train_audit",
        "candidate_manifest": {
            "path": str(candidate_path),
            "sha256": candidate_manifest_sha256.lower(),
        },
        "selected_qk_manifest": {
            "path": str(selected_path),
            "sha256": selected_qk_manifest_sha256.lower(),
        },
        "candidate_shape": list(candidates.shape),
        "candidate_score_ranked_mass_retention": retention,
        "selected_older_score_rank": {
            "count": int(ranks.size),
            "top4_fraction": float(np.mean(ranks <= 4)),
            "p50": float(np.quantile(ranks, 0.50)),
            "p90": float(np.quantile(ranks, 0.90)),
            "maximum": int(np.max(ranks)),
            "score_match_abs_error_p99": float(np.quantile(matches, 0.99)),
        },
        "feature_fidelity_only": True,
        "native_slot_membership_proven": False,
        "confirmation_split_opened": False,
    }


def audit_candidate_key_compression(
    *,
    candidate_key_manifest: str | Path,
    candidate_key_manifest_sha256: str,
    ranks: Sequence[int] = _KEY_RANKS,
) -> dict[str, Any]:
    """Measure train-only low-rank reconstruction of older candidate keys.

    A separate rank-``r`` PCA basis is fitted for each layer/query-head pair.
    The report is a key-fidelity and traffic proxy only: it does not claim
    score recall because query vectors are intentionally not exposed by the
    evaluator ABI.
    """

    key_path, key_metadata = _checked_manifest(
        candidate_key_manifest,
        candidate_key_manifest_sha256,
        "blockwise-QK candidate-key manifest",
    )
    if (
        key_metadata.get("experiment")
        != full._QK_CANDIDATE_KEY_CAPTURE_EXPERIMENT
        or key_metadata.get("confirmation_split_opened") is not False
        or key_metadata.get("head_dimension") != full._HEAD_DIMENSION
    ):
        raise ValueError("candidate-key audit confirmation contract changed")
    keys, _ = full.load_stacked_full_visible_qk_candidate_key_trace(
        key_path,
        candidate_key_manifest_sha256,
        protocol=key_metadata.get("protocol"),
    )
    rank_values = tuple(int(value) for value in ranks)
    if (
        not rank_values
        or any(value <= 0 or value > full._HEAD_DIMENSION for value in rank_values)
        or len(set(rank_values)) != len(rank_values)
    ):
        raise ValueError("candidate-key compression ranks are invalid")
    # Fit one basis per layer/head.  Keep the aggregate errors as well as
    # per-key normalized errors so a single pathological group is visible.
    total_energy = 0.0
    squared_errors = {rank: 0.0 for rank in rank_values}
    normalized_errors: dict[int, list[float]] = {
        rank: [] for rank in rank_values
    }
    for layer in range(keys.shape[2]):
        for head in range(keys.shape[3]):
            matrix = np.asarray(
                keys[:, :, layer, head], dtype=np.float64
            ).reshape(-1, full._HEAD_DIMENSION)
            center = matrix.mean(axis=0)
            centered = matrix - center
            total_energy += float(np.sum(centered * centered, dtype=np.float64))
            covariance = centered.T @ centered
            eigenvalues, basis = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            basis = basis[:, order]
            for rank in rank_values:
                components = basis[:, :rank]
                reconstructed = (centered @ components) @ components.T + center
                error = reconstructed - matrix
                squared = np.sum(error * error, axis=1, dtype=np.float64)
                squared_errors[rank] += float(np.sum(squared, dtype=np.float64))
                denominator = np.sum(matrix * matrix, axis=1, dtype=np.float64)
                normalized_errors[rank].extend(
                    np.divide(
                        squared,
                        np.maximum(denominator, 1.0e-12),
                    ).tolist()
                )
    dense_bytes_per_key = full._HEAD_DIMENSION * 4
    key_count = int(np.prod(keys.shape[:-1], dtype=np.int64))
    compression: dict[str, Any] = {}
    for rank in rank_values:
        values = np.asarray(normalized_errors[rank], dtype=np.float64)
        # The traffic estimate assumes a cached basis/mean and float16
        # coefficients per candidate key; it is deliberately labeled as a
        # model, not a benchmark.
        coefficient_bytes_per_key = rank * 2
        basis_bytes_per_group = full._HEAD_DIMENSION * rank * 2 + full._HEAD_DIMENSION * 4
        estimated_traffic = coefficient_bytes_per_key * key_count + (
            basis_bytes_per_group * full._LAYERS * full._QUERY_HEADS
        )
        compression[str(rank)] = {
            "rank": rank,
            "relative_centered_mse": float(squared_errors[rank] / max(total_energy, 1.0e-12)),
            "normalized_key_mse_mean": float(np.mean(values)),
            "normalized_key_mse_p95": float(np.quantile(values, 0.95)),
            "normalized_key_mse_max": float(np.max(values)),
            "dense_bytes_per_key": dense_bytes_per_key,
            "compressed_coefficient_bytes_per_key": coefficient_bytes_per_key,
            "basis_parameter_bytes": (
                basis_bytes_per_group * full._LAYERS * full._QUERY_HEADS
            ),
            "estimated_key_traffic_bytes": int(estimated_traffic),
            "estimated_traffic_ratio_vs_dense": float(
                estimated_traffic / max(key_count * dense_bytes_per_key, 1)
            ),
        }
    return {
        "schema_version": 1,
        "experiment": "olmoe_q7_retrieval_episodic_blockwise_qk_candidate_key_compression_audit",
        "candidate_key_manifest": {
            "path": str(key_path),
            "sha256": candidate_key_manifest_sha256.lower(),
        },
        "candidate_key_shape": list(keys.shape),
        "fitting": {
            "basis": "per_layer_query_head_centered_pca",
            "ranks": list(rank_values),
            "records": full._RECORDS,
            "confirmation_split_opened": False,
        },
        "dense_key_count": key_count,
        "dense_key_bytes": int(key_count * dense_bytes_per_key),
        "rank_results": compression,
        "score_recall_measured": False,
        "feature_fidelity_only": True,
        "causal_policy_changed": False,
        "confirmation_split_opened": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit train-only C28 QK capture")
    commands = parser.add_subparsers(dest="command", required=True)
    visible = commands.add_parser("visible")
    visible.add_argument("--qk-manifest", required=True)
    visible.add_argument("--qk-manifest-sha256", required=True)
    visible.add_argument("--value-manifest", required=True)
    visible.add_argument("--value-manifest-sha256", required=True)
    visible.add_argument("--out", required=True)
    candidate = commands.add_parser("candidates")
    candidate.add_argument("--candidate-manifest", required=True)
    candidate.add_argument("--candidate-manifest-sha256", required=True)
    candidate.add_argument("--selected-qk-manifest", required=True)
    candidate.add_argument("--selected-qk-manifest-sha256", required=True)
    candidate.add_argument("--out", required=True)
    key = commands.add_parser("candidate-key-compression")
    key.add_argument("--candidate-key-manifest", required=True)
    key.add_argument("--candidate-key-manifest-sha256", required=True)
    key.add_argument("--ranks", type=int, nargs="+", default=list(_KEY_RANKS))
    key.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "visible":
        report = audit_blockwise_qk_capture(
            qk_manifest=args.qk_manifest,
            qk_manifest_sha256=args.qk_manifest_sha256,
            value_manifest=args.value_manifest,
            value_manifest_sha256=args.value_manifest_sha256,
        )
    elif args.command == "candidates":
        report = audit_candidate_qk_capture(
            candidate_manifest=args.candidate_manifest,
            candidate_manifest_sha256=args.candidate_manifest_sha256,
            selected_qk_manifest=args.selected_qk_manifest,
            selected_qk_manifest_sha256=args.selected_qk_manifest_sha256,
        )
    else:
        report = audit_candidate_key_compression(
            candidate_key_manifest=args.candidate_key_manifest,
            candidate_key_manifest_sha256=args.candidate_key_manifest_sha256,
            ranks=args.ranks,
        )
    write_audit_report(report, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
