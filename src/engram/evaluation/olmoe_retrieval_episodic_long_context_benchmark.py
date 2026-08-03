"""Benchmark the frozen episodic selector at longer CPU context lengths.

This is a scaling experiment, not a new selector fit.  The rank-16/pool-6
mask is reconstructed from the already frozen train artifacts and the
independent development capture.  The first 128 positions therefore retain
the authenticated causal comparison; positions after 128 use no episodic
directive and repeat the deterministic development token stream.  This keeps
the long-context result honest: quality is reported only where the selector
was authenticated, while wall time, native counters, and peak RSS measure
the bounded-state runtime as context grows.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from engram.evaluation import olmoe_retrieval_episodic_blockwise_qk as qk
from engram.evaluation import olmoe_retrieval_episodic_candidate_selector as selector
from engram.evaluation import olmoe_retrieval_episodic_development_replay as development
from engram.evaluation import olmoe_retrieval_episodic_oracle as oracle
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.utils import atomic_json, sha256_file


_BASE_POSITIONS = 128
_ANSWER_START = 96
_READS = 32
_RECORDS = 8
_LAYERS = 16
_HEADS = 16
_CANDIDATES = 8
_HEAD_DIMENSION = 128
_ALL_ONES = np.ones((_LAYERS, _HEADS, _CANDIDATES), dtype=bool)


def _progress(message: str) -> None:
    print(f"[long-context-benchmark] {message}", file=sys.stderr, flush=True)


def _cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    selected = logits[np.arange(targets.size), targets]
    maximum = np.max(logits, axis=-1)
    normalizer = maximum + np.log(np.sum(np.exp(logits - maximum[:, None]), axis=-1))
    return float(np.mean(normalizer - selected))


def _load_masks(
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
    rank: int,
    pool_size: int,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, str]]:
    records_source = Path(records_path).expanduser().resolve()
    if sha256_file(records_source) != records_sha256.lower():
        raise ValueError("long-context benchmark records digest changed")
    records = oracle.retrieval._read_split(records_source, split="development")
    if len(records) != _RECORDS:
        raise ValueError("long-context benchmark record count changed")

    train_keys, train_key_meta = qk.full.load_stacked_full_visible_qk_candidate_key_trace(
        train_candidate_key_manifest,
        train_candidate_key_manifest_sha256,
    )
    train_candidates, train_candidate_meta = qk.full.load_stacked_full_visible_qk_candidate_trace(
        train_candidate_manifest,
        train_candidate_manifest_sha256,
    )
    qk._cross_check_shards(train_candidate_meta, train_key_meta)
    del train_candidates  # The selector needs the key trace, not train scores.
    train_queries, _ = selector._load_authenticated_query_features(
        train_query_features,
        train_query_features_sha256,
    )

    capture_dir = Path(development_capture).expanduser().resolve()
    capture_manifest = capture_dir / "manifest.json"
    if sha256_file(capture_manifest) != development_manifest_sha256.lower():
        raise ValueError("long-context benchmark development manifest changed")
    dev_keys = development._load_tensor(
        capture_dir / "candidate_keys.safetensors",
        "cb06b36b252e93fb43aadbf792846a6e82476d6a200e1543b3de74a5346d2948",
        "candidate_keys",
        (_RECORDS, _READS, _LAYERS, _HEADS, _CANDIDATES, _HEAD_DIMENSION),
    )
    dev_queries = development._load_queries(
        Path(development_query_features).expanduser().resolve(),
        development_query_features_sha256,
    )
    masks = selector.build_query_key_cross_split_masks(
        train_queries,
        train_keys,
        dev_queries,
        dev_keys,
        np.arange(_ANSWER_START, _BASE_POSITIONS, dtype=np.int64),
        rank=rank,
        pool_size=pool_size,
    )
    artifacts = {
        "records_sha256": records_sha256.lower(),
        "development_manifest_sha256": development_manifest_sha256.lower(),
        "development_query_features_sha256": development_query_features_sha256.lower(),
        "train_candidate_key_manifest_sha256": train_candidate_key_manifest_sha256.lower(),
        "train_candidate_manifest_sha256": train_candidate_manifest_sha256.lower(),
        "train_query_features_sha256": train_query_features_sha256.lower(),
    }
    return records, np.ascontiguousarray(masks), artifacts


def _extended_input_ids(record: Mapping[str, Any], positions: int) -> np.ndarray:
    source = np.asarray(record["input_ids"], dtype=np.int64)
    if source.size != _BASE_POSITIONS + 1:
        raise ValueError("long-context benchmark input length changed")
    # ``np.resize`` repeats the authenticated token sequence deterministically;
    # it does not synthesize new facts or write to episodic memory.
    return np.ascontiguousarray(np.resize(source, positions + 1))


def _run_sequence(
    runtime: OLMoENativeTokenRuntime,
    *,
    input_ids: np.ndarray,
    base_schedule: Mapping[str, Any],
    selector_masks: np.ndarray | None,
) -> dict[str, Any]:
    if runtime.position != 0:
        runtime.reset()
    answer_positions = tuple(range(_ANSWER_START, _BASE_POSITIONS))
    answer_set = set(answer_positions)
    hidden_rows: list[np.ndarray] = []
    logit_rows: list[np.ndarray] = []
    next_tokens: list[int] = []
    started = time.perf_counter()
    final_metrics: dict[str, int] = {}
    for position, token_id in enumerate(input_ids[:-1]):
        if position < _BASE_POSITIONS:
            row = base_schedule["rows"][position]
            write_slot = int(row["write_slot"])
            read_span = int(row["read_span"])
        else:
            write_slot = -1
            read_span = -1
        if selector_masks is None:
            result = runtime.forward_episodic([int(token_id)], [write_slot], [read_span])
        else:
            mask = (
                selector_masks[position]
                if position < _BASE_POSITIONS
                else _ALL_ONES
            )
            result = runtime.forward_episodic_masked(
                [int(token_id)],
                [write_slot],
                [read_span],
                np.asarray(mask, dtype=bool)[None, ...],
            )
        final_metrics = dict(result.metrics)
        next_tokens.append(int(result.next_token))
        if position in answer_set:
            hidden, logits = runtime.last_diagnostics()
            hidden_rows.append(np.asarray(hidden, dtype=np.float32).copy())
            logit_rows.append(np.asarray(logits, dtype=np.float32).copy())
    if runtime.position != input_ids.size - 1:
        raise ValueError("long-context runtime did not reach requested position")
    return {
        "hidden": np.stack(hidden_rows),
        "logits": np.stack(logit_rows),
        "next_tokens": np.asarray(next_tokens, dtype=np.int64),
        "answer_positions": answer_positions,
        "final_metrics": final_metrics,
        "wall_seconds": time.perf_counter() - started,
    }


def _compare(
    baseline: Mapping[str, Any],
    masked: Mapping[str, Any],
    input_ids: np.ndarray,
) -> dict[str, Any]:
    hidden_base = np.asarray(baseline["hidden"], dtype=np.float64)
    hidden_mask = np.asarray(masked["hidden"], dtype=np.float64)
    logits_base = np.asarray(baseline["logits"], dtype=np.float64)
    logits_mask = np.asarray(masked["logits"], dtype=np.float64)
    hidden_relative = np.linalg.norm(hidden_mask - hidden_base, axis=-1) / np.maximum(
        np.linalg.norm(hidden_base, axis=-1), 1.0e-12
    )
    logits_relative = np.linalg.norm(logits_mask - logits_base, axis=-1) / np.maximum(
        np.linalg.norm(logits_base, axis=-1), 1.0e-12
    )
    answer_positions = np.asarray(baseline["answer_positions"], dtype=np.int64)
    targets = input_ids[answer_positions + 1]
    base_tokens = np.asarray(baseline["next_tokens"])[answer_positions]
    masked_tokens = np.asarray(masked["next_tokens"])[answer_positions]
    base_metrics = dict(baseline["final_metrics"])
    masked_metrics = dict(masked["final_metrics"])
    base_bytes = int(base_metrics.get("attention_logical_read_bytes", 0))
    masked_bytes = int(masked_metrics.get("attention_logical_read_bytes", 0))
    return {
        "quality_window": {"start": _ANSWER_START, "stop": _BASE_POSITIONS},
        "answer_hidden_relative_l2_mean": float(np.mean(hidden_relative)),
        "answer_hidden_relative_l2_max": float(np.max(hidden_relative)),
        "answer_logits_relative_l2_mean": float(np.mean(logits_relative)),
        "answer_logits_relative_l2_max": float(np.max(logits_relative)),
        "answer_top1_agreement": float(np.mean(base_tokens == masked_tokens)),
        "answer_nll_delta": _cross_entropy(logits_mask, targets)
        - _cross_entropy(logits_base, targets),
        "traffic": {
            "baseline_attention_logical_read_bytes": base_bytes,
            "masked_attention_logical_read_bytes": masked_bytes,
            "masked_over_baseline_logical_read_fraction": float(
                masked_bytes / base_bytes if base_bytes else 0.0
            ),
            "baseline_attention_older_candidate_entries_scored": int(
                base_metrics.get("attention_older_candidate_entries_scored", 0)
            ),
            "masked_attention_older_candidate_entries_scored": int(
                masked_metrics.get("attention_older_candidate_entries_scored", 0)
            ),
            "baseline_episodic_key_read_bytes": int(
                base_metrics.get("episodic_key_read_bytes", 0)
            ),
            "masked_episodic_key_read_bytes": int(
                masked_metrics.get("episodic_key_read_bytes", 0)
            ),
            "baseline_episodic_value_read_bytes": int(
                base_metrics.get("episodic_value_read_bytes", 0)
            ),
            "masked_episodic_value_read_bytes": int(
                masked_metrics.get("episodic_value_read_bytes", 0)
            ),
        },
    }


def run_long_context_benchmark(
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
    lengths: Sequence[int] = (128, 512, 2048),
    record_index: int = 0,
    rank: int = 16,
    pool_size: int = 6,
    threads: int = 12,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not lengths or any(int(length) < _BASE_POSITIONS for length in lengths):
        raise ValueError("benchmark lengths must be at least 128")
    records, masks, artifacts = _load_masks(
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
        rank=rank,
        pool_size=pool_size,
    )
    if not 0 <= record_index < len(records):
        raise ValueError("benchmark record index is out of range")
    package = Path(package_path).expanduser().resolve()
    library = Path(native_library).expanduser().resolve()
    anchors = oracle._fact_anchor_ids(package / "tokenizer/tokenizer.json")
    record = records[record_index]
    base_schedule = oracle._derive_schedule(record["input_ids"], anchors)
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
        for length in sorted({int(value) for value in lengths}):
            input_ids = _extended_input_ids(record, length)
            _progress(f"record {record_index} length {length} baseline")
            baseline = _run_sequence(
                runtime,
                input_ids=input_ids,
                base_schedule=base_schedule,
                selector_masks=None,
            )
            runtime.reset()
            _progress(f"record {record_index} length {length} masked")
            masked = _run_sequence(
                runtime,
                input_ids=input_ids,
                base_schedule=base_schedule,
                selector_masks=masks[record_index],
            )
            comparison = _compare(baseline, masked, input_ids)
            rows.append(
                {
                    "record_index": record_index,
                    "record_id": record["record_id"],
                    "positions": length,
                    "context_tokens": length + 1,
                    "baseline_wall_seconds": float(baseline["wall_seconds"]),
                    "masked_wall_seconds": float(masked["wall_seconds"]),
                    "baseline_tokens_per_second": float(length / baseline["wall_seconds"]),
                    "masked_tokens_per_second": float(length / masked["wall_seconds"]),
                    "comparison": comparison,
                }
            )
    finally:
        runtime.close()
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "schema_version": 1,
        "experiment": "olmoe_q7_retrieval_episodic_long_context_benchmark",
        "status": "complete",
        "scope": {
            "split": "development",
            "record_index": record_index,
            "records_measured": 1,
            "selector_fit_on_development": False,
            "cpu_only": True,
            "quality_window_authenticated_positions": [_ANSWER_START, _BASE_POSITIONS],
            "post_base_positions": "deterministic token repeat with write/read directives disabled",
        },
        "selector": {"rank": rank, "pool_size": pool_size, "native_top_k": 4},
        "records": rows,
        "peak_rss_kib": peak_rss_kib,
        "artifacts": {
            **artifacts,
            "native_library_sha256": sha256_file(library),
            "package_config_sha256": sha256_file(package / "model/config.json"),
            "package_non_mlp_sha256": sha256_file(package / "transformer/non_mlp.safetensors"),
            "package_mlp_sha256": sha256_file(package / "mlp/experts.q7"),
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
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--pool-size", type=int, default=6)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_long_context_benchmark(
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
        lengths=args.lengths,
        record_index=args.record_index,
        rank=args.rank,
        pool_size=args.pool_size,
        threads=args.threads,
    )
    atomic_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
