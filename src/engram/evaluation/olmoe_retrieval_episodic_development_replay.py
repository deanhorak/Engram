"""Train-fitted selector replay on the independent development split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.evaluation import olmoe_retrieval_episodic_blockwise_qk as qk
from engram.evaluation import olmoe_retrieval_episodic_candidate_selector as selector
from engram.evaluation import olmoe_retrieval_episodic_native_masked_replay as replay
from engram.evaluation import olmoe_retrieval_episodic_oracle as oracle
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.utils import atomic_json, sha256_file


_RECORDS = 8
_READS = 32
_LAYERS = 16
_HEADS = 16
_CANDIDATES = 8
_HEAD_DIMENSION = 128
_POSITIONS = 128


def _progress(message: str) -> None:
    print(f"[development-replay] {message}", file=sys.stderr, flush=True)


def _load_tensor(
    path: str | Path,
    digest: str,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    source = Path(path).expanduser().resolve()
    if sha256_file(source) != digest.lower():
        raise ValueError(f"development replay {name} file digest changed")
    try:
        from safetensors.numpy import load_file
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("development replay requires safetensors") from error
    value = np.ascontiguousarray(load_file(source)[name])
    if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"development replay {name} tensor changed")
    return value


def _load_queries(path: str | Path, digest: str) -> np.ndarray:
    source = Path(path).expanduser().resolve()
    if sha256_file(source) != digest.lower():
        raise ValueError("development replay query file digest changed")
    try:
        from safetensors.numpy import load_file
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("development replay requires safetensors") from error
    loaded = load_file(source)
    queries = np.ascontiguousarray(loaded["post_qnorm_pre_rope_queries"])
    positions = np.ascontiguousarray(loaded["positions"])
    if (
        queries.shape != (_RECORDS, _READS, _LAYERS, _HEADS, _HEAD_DIMENSION)
        or queries.dtype != np.float32
        or positions.shape != (_READS,)
        or positions.dtype != np.int64
        or not np.array_equal(positions, qk.full._READ_POSITIONS)
        or not np.isfinite(queries).all()
    ):
        raise ValueError("development replay query tensors changed")
    return queries


def _pool_metrics(masks: np.ndarray, candidate_scores: np.ndarray) -> dict[str, float]:
    oracle_top = np.argsort(-candidate_scores, axis=-1, kind="stable")[:, :, :, :, :4]
    rows: list[float] = []
    exact_rows: list[float] = []
    for record in range(_RECORDS):
        for read in range(_READS):
            position = _POSITIONS - _READS + read
            allowed = masks[record, position]
            for layer in range(_LAYERS):
                for head in range(_HEADS):
                    chosen = np.flatnonzero(allowed[layer, head])
                    oracle_row = oracle_top[record, read, layer, head]
                    rows.append(float(np.intersect1d(chosen, oracle_row).size / 4.0))
                    exact = candidate_scores[record, read, layer, head]
                    reranked = chosen[np.argsort(-exact[chosen], kind="stable")[:4]]
                    exact_rows.append(
                        float(np.intersect1d(reranked, oracle_row).size / 4.0)
                    )
    return {
        "candidate_pool_membership_recall_mean": float(np.mean(rows)),
        "candidate_pool_membership_recall_p10": float(np.quantile(rows, 0.10)),
        "exact_rerank_membership_recall_mean": float(np.mean(exact_rows)),
        "exact_rerank_membership_recall_p10": float(np.quantile(exact_rows, 0.10)),
        "older_candidate_slot_fraction": 0.75,
    }


def run_development_replay(
    *,
    records_path: str | Path,
    records_sha256: str,
    package_path: str | Path,
    native_library: str | Path,
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
    rank: int = 16,
    pool_size: int = 6,
    threads: int = 12,
) -> dict[str, Any]:
    started = time.perf_counter()
    records_source = Path(records_path).expanduser().resolve()
    if sha256_file(records_source) != records_sha256.lower():
        raise ValueError("development replay records digest changed")
    records = oracle.retrieval._read_split(records_source, split="development")
    if len(records) != _RECORDS:
        raise ValueError("development replay record count changed")
    capture_dir = Path(development_capture).expanduser().resolve()
    capture_manifest = capture_dir / "manifest.json"
    if sha256_file(capture_manifest) != development_manifest_sha256.lower():
        raise ValueError("development capture manifest digest changed")
    train_keys, train_key_meta = qk.full.load_stacked_full_visible_qk_candidate_key_trace(
        train_candidate_key_manifest,
        train_candidate_key_manifest_sha256,
    )
    train_candidates, train_candidate_meta = qk.full.load_stacked_full_visible_qk_candidate_trace(
        train_candidate_manifest,
        train_candidate_manifest_sha256,
    )
    qk._cross_check_shards(train_candidate_meta, train_key_meta)
    train_queries, _ = selector._load_authenticated_query_features(
        train_query_features,
        train_query_features_sha256,
    )
    dev_keys = _load_tensor(
        capture_dir / "candidate_keys.safetensors",
        "cb06b36b252e93fb43aadbf792846a6e82476d6a200e1543b3de74a5346d2948",
        "candidate_keys",
        (_RECORDS, _READS, _LAYERS, _HEADS, _CANDIDATES, _HEAD_DIMENSION),
    )
    dev_bands = _load_tensor(
        capture_dir / "candidate_qk_bands.safetensors",
        "5c2992f5815c627a602bca225a076f45772652ae0adb239cb3545e43034d4bb7",
        "candidate_qk_bands",
        (_RECORDS, _READS, _LAYERS, _HEADS, _CANDIDATES, 8),
    )
    dev_queries = _load_queries(
        development_query_features,
        development_query_features_sha256,
    )
    masks = selector.build_query_key_cross_split_masks(
        train_queries,
        train_keys,
        dev_queries,
        dev_keys,
        np.arange(96, 128, dtype=np.int64),
        rank=rank,
        pool_size=pool_size,
    )
    dev_scores = np.sum(dev_bands, axis=-1, dtype=np.float32)
    locality = _pool_metrics(masks, dev_scores)
    package = Path(package_path).expanduser().resolve()
    library = Path(native_library).expanduser().resolve()
    anchors = oracle._fact_anchor_ids(package / "tokenizer/tokenizer.json")
    schedules = [oracle._derive_schedule(record["input_ids"], anchors) for record in records]
    runtime = OLMoENativeTokenRuntime(
        str(package / "model/config.json"),
        str(package / "transformer/non_mlp.safetensors"),
        str(package / "mlp/experts.q7"),
        str(library),
        threads=threads,
        local_window=16,
        older_candidates=8,
        older_top_k=4,
        sink_tokens=2,
        episodic_policy={"slots": 32, "span_size": 8},
    )
    rows: list[dict[str, Any]] = []
    try:
        for index, (record, schedule) in enumerate(zip(records, schedules, strict=True)):
            _progress(f"record {index + 1}/{_RECORDS} baseline")
            baseline = replay._run_record(runtime, record=record, schedule=schedule, masks=None)
            runtime.reset()
            _progress(f"record {index + 1}/{_RECORDS} masked")
            masked = replay._run_record(
                runtime,
                record=record,
                schedule=schedule,
                masks=masks[index],
            )
            rows.append(replay._record_metrics(baseline, masked, record))
    finally:
        runtime.close()
    aggregate = {
        "answer_hidden_relative_l2_mean": float(np.mean([row["answer_hidden_relative_l2_mean"] for row in rows])),
        "answer_logits_relative_l2_mean": float(np.mean([row["answer_logits_relative_l2_mean"] for row in rows])),
        "answer_top1_agreement_mean": float(np.mean([row["answer_top1_agreement"] for row in rows])),
        "answer_nll_delta_mean": float(np.mean([row["answer_nll_delta"] for row in rows])),
        "answer_nll_delta_max": float(np.max([row["answer_nll_delta"] for row in rows])),
        "baseline_attention_logical_read_bytes_mean": float(
            np.mean(
                [
                    row["traffic"]["baseline_attention_logical_read_bytes"]
                    for row in rows
                ]
            )
        ),
        "masked_attention_logical_read_bytes_mean": float(
            np.mean(
                [
                    row["traffic"]["masked_attention_logical_read_bytes"]
                    for row in rows
                ]
            )
        ),
        "masked_over_baseline_logical_read_fraction_mean": float(
            np.mean(
                [
                    row["traffic"]["masked_over_baseline_logical_read_fraction"]
                    for row in rows
                ]
            )
        ),
        "baseline_older_candidate_entries_scored_mean": float(
            np.mean(
                [
                    row["traffic"][
                        "baseline_attention_older_candidate_entries_scored"
                    ]
                    for row in rows
                ]
            )
        ),
        "masked_older_candidate_entries_scored_mean": float(
            np.mean(
                [
                    row["traffic"][
                        "masked_attention_older_candidate_entries_scored"
                    ]
                    for row in rows
                ]
            )
        ),
    }
    passed = bool(
        locality["exact_rerank_membership_recall_mean"] >= 0.95
        and aggregate["answer_hidden_relative_l2_mean"] <= replay._THRESHOLDS["maximum_hidden_relative_l2"]
        and aggregate["answer_logits_relative_l2_mean"] <= replay._THRESHOLDS["maximum_logits_relative_l2"]
        and aggregate["answer_nll_delta_mean"] <= replay._THRESHOLDS["maximum_answer_nll_delta"]
        and aggregate["answer_top1_agreement_mean"] >= replay._THRESHOLDS["minimum_answer_top1_agreement"]
        and all(row["passed"] for row in rows)
    )
    return {
        "schema_version": 1,
        "experiment": "olmoe_q7_retrieval_episodic_development_replay",
        "status": "development_native_masked_replay_passed" if passed else "development_native_masked_replay_failed",
        "scope": {"split": "development", "records": _RECORDS, "selector_fit_on_development": False, "cpu_only": True},
        "selector": {"rank": rank, "pool_size": pool_size, "native_top_k": 4},
        "locality": locality,
        "causal_thresholds": replay._THRESHOLDS,
        "aggregate": aggregate,
        "records": rows,
        "artifacts": {
            "records_sha256": records_sha256.lower(),
            "development_capture_manifest_sha256": development_manifest_sha256.lower(),
            "development_query_features_sha256": development_query_features_sha256.lower(),
            "train_candidate_key_manifest_sha256": train_candidate_key_manifest_sha256.lower(),
            "train_candidate_manifest_sha256": train_candidate_manifest_sha256.lower(),
            "train_query_features_sha256": train_query_features_sha256.lower(),
            "native_library_sha256": sha256_file(library),
        },
        "confirmation_split_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--records-sha256", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--native-library", required=True)
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
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--pool-size", type=int, default=6)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_development_replay(
        records_path=args.records,
        records_sha256=args.records_sha256,
        package_path=args.package,
        native_library=args.native_library,
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
        rank=args.rank,
        pool_size=args.pool_size,
        threads=args.threads,
    )
    atomic_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
