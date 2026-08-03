"""Sweep frozen rank/pool choices on the independent development capture.

This is feature-locality attribution only.  It never fits on development,
executes a native intervention, or opens a protected split.  The rank basis
is reconstructed from the authenticated train artifacts and applied to the
development query/key tensors; exact reranking then uses the captured native
Q/K bands.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.evaluation import olmoe_retrieval_episodic_blockwise_qk as qk
from engram.evaluation import olmoe_retrieval_episodic_candidate_selector as selector
from engram.evaluation import olmoe_retrieval_episodic_development_replay as development
from engram.evaluation import olmoe_retrieval_episodic_oracle as oracle
from engram.utils import atomic_json, sha256_file


_RECORDS = 8
_READS = 32
_ANSWER_START = 96
_POSITIONS = 128
_LAYERS = 16
_HEADS = 16
_CANDIDATES = 8
_HEAD_DIMENSION = 128
_BANDS = 8


def _load_inputs(
    *,
    records_path: str | Path,
    records_sha256: str,
    train_candidate_key_manifest: str | Path,
    train_candidate_key_manifest_sha256: str,
    train_candidate_manifest: str | Path,
    train_candidate_manifest_sha256: str,
    train_query_features: str | Path,
    train_query_features_sha256: str,
    development_capture: str | Path,
    development_manifest_sha256: str,
    development_query_features: str | Path,
    development_query_features_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    records_source = Path(records_path).expanduser().resolve()
    if sha256_file(records_source) != records_sha256.lower():
        raise ValueError("development pool sweep records digest changed")
    records = oracle.retrieval._read_split(records_source, split="development")
    if len(records) != _RECORDS:
        raise ValueError("development pool sweep record count changed")
    train_keys, train_key_meta = qk.full.load_stacked_full_visible_qk_candidate_key_trace(
        train_candidate_key_manifest,
        train_candidate_key_manifest_sha256,
    )
    _, train_candidate_meta = qk.full.load_stacked_full_visible_qk_candidate_trace(
        train_candidate_manifest,
        train_candidate_manifest_sha256,
    )
    qk._cross_check_shards(train_candidate_meta, train_key_meta)
    train_queries, _ = selector._load_authenticated_query_features(
        train_query_features,
        train_query_features_sha256,
    )
    capture_dir = Path(development_capture).expanduser().resolve()
    if sha256_file(capture_dir / "manifest.json") != development_manifest_sha256.lower():
        raise ValueError("development pool sweep manifest changed")
    dev_keys = development._load_tensor(
        capture_dir / "candidate_keys.safetensors",
        "cb06b36b252e93fb43aadbf792846a6e82476d6a200e1543b3de74a5346d2948",
        "candidate_keys",
        (_RECORDS, _READS, _LAYERS, _HEADS, _CANDIDATES, _HEAD_DIMENSION),
    )
    dev_bands = development._load_tensor(
        capture_dir / "candidate_qk_bands.safetensors",
        "5c2992f5815c627a602bca225a076f45772652ae0adb239cb3545e43034d4bb7",
        "candidate_qk_bands",
        (_RECORDS, _READS, _LAYERS, _HEADS, _CANDIDATES, _BANDS),
    )
    dev_queries = development._load_queries(
        Path(development_query_features).expanduser().resolve(),
        development_query_features_sha256,
    )
    return (
        train_queries,
        train_keys,
        dev_queries,
        dev_keys,
        dev_bands,
        {
            "records_sha256": records_sha256.lower(),
            "development_manifest_sha256": development_manifest_sha256.lower(),
            "development_query_features_sha256": development_query_features_sha256.lower(),
            "train_candidate_key_manifest_sha256": train_candidate_key_manifest_sha256.lower(),
            "train_candidate_manifest_sha256": train_candidate_manifest_sha256.lower(),
            "train_query_features_sha256": train_query_features_sha256.lower(),
        },
    )


def _score_masks(masks: np.ndarray, bands: np.ndarray) -> dict[str, float]:
    scores = np.sum(bands, axis=-1, dtype=np.float32)
    oracle_top = np.argsort(-scores, axis=-1, kind="stable")[:, :, :, :, :4]
    pool_rows: list[float] = []
    exact_rows: list[float] = []
    for record in range(_RECORDS):
        for read in range(_READS):
            position = _ANSWER_START + read
            allowed = masks[record, position]
            for layer in range(_LAYERS):
                for head in range(_HEADS):
                    chosen = np.flatnonzero(allowed[layer, head])
                    oracle_row = oracle_top[record, read, layer, head]
                    pool_rows.append(
                        float(np.intersect1d(chosen, oracle_row).size / 4.0)
                    )
                    reranked = chosen[
                        np.argsort(-scores[record, read, layer, head][chosen], kind="stable")[:4]
                    ]
                    exact_rows.append(
                        float(np.intersect1d(reranked, oracle_row).size / 4.0)
                    )
    pool_size = int(np.sum(masks[0, _ANSWER_START, 0, 0]))
    return {
        "pool_size": pool_size,
        "older_slot_fraction": float(pool_size / _CANDIDATES),
        "candidate_pool_membership_recall_mean": float(np.mean(pool_rows)),
        "candidate_pool_membership_recall_p10": float(np.quantile(pool_rows, 0.10)),
        "exact_rerank_membership_recall_mean": float(np.mean(exact_rows)),
        "exact_rerank_membership_recall_p10": float(np.quantile(exact_rows, 0.10)),
    }


def run_pool_sweep(
    *,
    records_path: str | Path,
    records_sha256: str,
    train_candidate_key_manifest: str | Path,
    train_candidate_key_manifest_sha256: str,
    train_candidate_manifest: str | Path,
    train_candidate_manifest_sha256: str,
    train_query_features: str | Path,
    train_query_features_sha256: str,
    development_capture: str | Path,
    development_manifest_sha256: str,
    development_query_features: str | Path,
    development_query_features_sha256: str,
    ranks: Sequence[int] = (8, 16, 32, 64),
    pool_sizes: Sequence[int] = (4, 6, 8),
) -> dict[str, Any]:
    started = time.perf_counter()
    train_queries, train_keys, dev_queries, dev_keys, dev_bands, artifacts = _load_inputs(
        records_path=records_path,
        records_sha256=records_sha256,
        train_candidate_key_manifest=train_candidate_key_manifest,
        train_candidate_key_manifest_sha256=train_candidate_key_manifest_sha256,
        train_candidate_manifest=train_candidate_manifest,
        train_candidate_manifest_sha256=train_candidate_manifest_sha256,
        train_query_features=train_query_features,
        train_query_features_sha256=train_query_features_sha256,
        development_capture=development_capture,
        development_manifest_sha256=development_manifest_sha256,
        development_query_features=development_query_features,
        development_query_features_sha256=development_query_features_sha256,
    )
    rows: dict[str, dict[str, float]] = {}
    positions = np.arange(_ANSWER_START, _POSITIONS, dtype=np.int64)
    for rank in sorted({int(value) for value in ranks}):
        for pool_size in sorted({int(value) for value in pool_sizes}):
            if not 1 <= rank <= _HEAD_DIMENSION or not 1 <= pool_size <= _CANDIDATES:
                raise ValueError("development pool sweep coordinate is invalid")
            masks = selector.build_query_key_cross_split_masks(
                train_queries,
                train_keys,
                dev_queries,
                dev_keys,
                positions,
                rank=rank,
                pool_size=pool_size,
            )
            rows[f"rank{rank}_pool{pool_size}"] = _score_masks(
                masks,
                # Development Q/K bands are already captured; no native model
                # execution or intervention is performed by this sweep.
                dev_bands,
            )
    return {
        "schema_version": 1,
        "experiment": "olmoe_q7_retrieval_episodic_development_pool_sweep",
        "status": "complete",
        "scope": {
            "split": "development",
            "records": _RECORDS,
            "reads": _READS,
            "selector_fit_on_development": False,
            "native_intervention_executed": False,
            "confirmation_split_opened": False,
            "traffic": "pool fraction is a candidate-stage attribution, not an end-to-end measurement",
        },
        "results": rows,
        "artifacts": artifacts,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--records-sha256", required=True)
    parser.add_argument("--train-candidate-key-manifest", required=True)
    parser.add_argument("--train-candidate-key-manifest-sha256", required=True)
    parser.add_argument("--train-candidate-manifest", required=True)
    parser.add_argument("--train-candidate-manifest-sha256", required=True)
    parser.add_argument("--train-query-features", required=True)
    parser.add_argument("--train-query-features-sha256", required=True)
    parser.add_argument("--development-capture", required=True)
    parser.add_argument("--development-manifest-sha256", required=True)
    parser.add_argument("--development-query-features", required=True)
    parser.add_argument("--development-query-features-sha256", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_pool_sweep(
        records_path=args.records,
        records_sha256=args.records_sha256,
        train_candidate_key_manifest=args.train_candidate_key_manifest,
        train_candidate_key_manifest_sha256=args.train_candidate_key_manifest_sha256,
        train_candidate_manifest=args.train_candidate_manifest,
        train_candidate_manifest_sha256=args.train_candidate_manifest_sha256,
        train_query_features=args.train_query_features,
        train_query_features_sha256=args.train_query_features_sha256,
        development_capture=args.development_capture,
        development_manifest_sha256=args.development_manifest_sha256,
        development_query_features=args.development_query_features,
        development_query_features_sha256=args.development_query_features_sha256,
        ranks=args.ranks,
        pool_sizes=args.pool_sizes,
    )
    atomic_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
