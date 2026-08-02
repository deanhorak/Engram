"""Native causal replay for the query/key exact-rerank selector.

The selector screen is only a feature-locality result.  This evaluator turns
its rank-16 candidate pools into the native evaluator-only older-slot mask,
replays every train record with the CPU OLMoE runtime, and compares the result
with the same runtime's unmasked episodic policy.  The comparison therefore
includes the complete downstream attention, residual, Q7 MLP, normalization,
and vocabulary projection path.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from engram.evaluation import olmoe_retrieval_episodic_candidate_selector as selector
from engram.evaluation import olmoe_retrieval_episodic_oracle as oracle
from engram.evaluation import olmoe_retrieval_episodic_blockwise_qk as qk
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.utils import atomic_json, sha256_file


_ANSWER_START = 96
_POSITIONS = 128
_THRESHOLDS = {
    "maximum_hidden_relative_l2": 0.10,
    "maximum_logits_relative_l2": 0.10,
    "maximum_answer_nll_delta": 0.05,
    "minimum_answer_top1_agreement": 0.90,
}


def _progress(message: str) -> None:
    print(f"[native-masked-replay] {message}", file=sys.stderr, flush=True)


def _checked_records(path: str | Path, *, split: str) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("native masked replay record file is unavailable")
    records = oracle.retrieval._read_split(source, split=split)
    if len(records) != selector._RECORDS:
        raise ValueError("native masked replay record count changed")
    return records


def _cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    selected = logits[np.arange(targets.size), targets]
    maximum = np.max(logits, axis=-1)
    log_normalizer = maximum + np.log(
        np.sum(np.exp(logits - maximum[:, None]), axis=-1)
    )
    return float(np.mean(log_normalizer - selected))


def _run_record(
    runtime: OLMoENativeTokenRuntime,
    *,
    record: Mapping[str, Any],
    schedule: Mapping[str, Any],
    masks: np.ndarray | None,
) -> dict[str, Any]:
    if runtime.position != 0:
        runtime.reset()
    if masks is not None and not runtime.masked_episodic_available:
        raise ValueError("native masked episodic ABI is unavailable")
    hidden_rows: list[np.ndarray] = []
    logit_rows: list[np.ndarray] = []
    next_tokens: list[int] = []
    final_metrics: dict[str, int] | None = None
    answer_positions = tuple(int(p) for p in record["answer_prediction_positions"])
    answer_set = set(answer_positions)
    for position, token_id in enumerate(record["input_ids"][:-1]):
        row = schedule["rows"][position]
        writes = [int(row["write_slot"])]
        reads = [int(row["read_span"])]
        if masks is None:
            result = runtime.forward_episodic([int(token_id)], writes, reads)
        else:
            result = runtime.forward_episodic_masked(
                [int(token_id)],
                writes,
                reads,
                masks[position : position + 1],
            )
        final_metrics = dict(result.metrics)
        next_tokens.append(int(result.next_token))
        if position in answer_set:
            hidden, logits = runtime.last_diagnostics()
            hidden_rows.append(np.asarray(hidden, dtype=np.float32).copy())
            logit_rows.append(np.asarray(logits, dtype=np.float32).copy())
    if runtime.position != _POSITIONS or len(hidden_rows) != len(answer_positions):
        raise ValueError("native masked replay did not complete the record")
    return {
        "hidden": np.stack(hidden_rows),
        "logits": np.stack(logit_rows),
        "next_tokens": np.asarray(next_tokens, dtype=np.int64),
        "answer_positions": answer_positions,
        "final_metrics": final_metrics or {},
    }


def _record_metrics(
    baseline: Mapping[str, Any],
    masked: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    hidden_base = np.asarray(baseline["hidden"], dtype=np.float64)
    hidden_mask = np.asarray(masked["hidden"], dtype=np.float64)
    logits_base = np.asarray(baseline["logits"], dtype=np.float64)
    logits_mask = np.asarray(masked["logits"], dtype=np.float64)
    hidden_delta = np.linalg.norm(hidden_mask - hidden_base, axis=-1)
    hidden_scale = np.maximum(np.linalg.norm(hidden_base, axis=-1), 1.0e-12)
    logits_delta = np.linalg.norm(logits_mask - logits_base, axis=-1)
    logits_scale = np.maximum(np.linalg.norm(logits_base, axis=-1), 1.0e-12)
    answer_positions = np.asarray(baseline["answer_positions"], dtype=np.int64)
    targets = np.asarray(
        [record["input_ids"][int(position) + 1] for position in answer_positions],
        dtype=np.int64,
    )
    baseline_answer_tokens = np.asarray(baseline["next_tokens"])[answer_positions]
    masked_answer_tokens = np.asarray(masked["next_tokens"])[answer_positions]
    prefix_count = int(np.searchsorted(answer_positions, _ANSWER_START))
    prefix_hidden_delta = np.abs(
        hidden_mask[:prefix_count] - hidden_base[:prefix_count]
    )
    prefix_logits_delta = np.abs(
        logits_mask[:prefix_count] - logits_base[:prefix_count]
    )
    hidden_relative = hidden_delta / hidden_scale
    logits_relative = logits_delta / logits_scale
    baseline_nll = _cross_entropy(logits_base, targets)
    masked_nll = _cross_entropy(logits_mask, targets)
    metrics = {
        "record_index": int(record["record_index"]),
        "record_id": record["record_id"],
        "answer_hidden_relative_l2_mean": float(np.mean(hidden_relative)),
        "answer_hidden_relative_l2_p90": float(np.quantile(hidden_relative, 0.90)),
        "answer_hidden_relative_l2_max": float(np.max(hidden_relative)),
        "answer_logits_relative_l2_mean": float(np.mean(logits_relative)),
        "answer_logits_relative_l2_p90": float(np.quantile(logits_relative, 0.90)),
        "answer_logits_relative_l2_max": float(np.max(logits_relative)),
        "answer_top1_agreement": float(
            np.mean(baseline_answer_tokens == masked_answer_tokens)
        ),
        "all_position_top1_agreement": float(
            np.mean(
                np.asarray(baseline["next_tokens"])
                == np.asarray(masked["next_tokens"])
            )
        ),
        "baseline_answer_nll": baseline_nll,
        "masked_answer_nll": masked_nll,
        "answer_nll_delta": masked_nll - baseline_nll,
        # Diagnostics are intentionally retained only on answer rows.  The
        # mask construction is all-ones before ``_ANSWER_START``; there are
        # therefore no prefix diagnostic rows to reduce here.
        "prefix_hidden_abs_max": float(np.max(prefix_hidden_delta))
        if prefix_hidden_delta.size
        else 0.0,
        "prefix_logits_abs_max": float(np.max(prefix_logits_delta))
        if prefix_logits_delta.size
        else 0.0,
        "answer_positions": answer_positions.tolist(),
    }
    baseline_metrics = dict(baseline.get("final_metrics", {}))
    masked_metrics = dict(masked.get("final_metrics", {}))
    baseline_logical = int(baseline_metrics.get("attention_logical_read_bytes", 0))
    masked_logical = int(masked_metrics.get("attention_logical_read_bytes", 0))
    metrics["traffic"] = {
        "baseline_attention_logical_read_bytes": baseline_logical,
        "masked_attention_logical_read_bytes": masked_logical,
        "masked_over_baseline_logical_read_fraction": float(
            masked_logical / baseline_logical if baseline_logical else 0.0
        ),
        "baseline_attention_older_candidate_entries_scored": int(
            baseline_metrics.get("attention_older_candidate_entries_scored", 0)
        ),
        "masked_attention_older_candidate_entries_scored": int(
            masked_metrics.get("attention_older_candidate_entries_scored", 0)
        ),
        "baseline_episodic_key_read_bytes": int(
            baseline_metrics.get("episodic_key_read_bytes", 0)
        ),
        "masked_episodic_key_read_bytes": int(
            masked_metrics.get("episodic_key_read_bytes", 0)
        ),
        "baseline_episodic_value_read_bytes": int(
            baseline_metrics.get("episodic_value_read_bytes", 0)
        ),
        "masked_episodic_value_read_bytes": int(
            masked_metrics.get("episodic_value_read_bytes", 0)
        ),
    }
    metrics["passed"] = bool(
        metrics["answer_hidden_relative_l2_mean"]
        <= _THRESHOLDS["maximum_hidden_relative_l2"]
        and metrics["answer_logits_relative_l2_mean"]
        <= _THRESHOLDS["maximum_logits_relative_l2"]
        and metrics["answer_nll_delta"]
        <= _THRESHOLDS["maximum_answer_nll_delta"]
        and metrics["answer_top1_agreement"]
        >= _THRESHOLDS["minimum_answer_top1_agreement"]
    )
    return metrics


def run_native_masked_replay(
    *,
    records_path: str | Path,
    records_sha256: str,
    package_path: str | Path,
    native_library: str | Path,
    candidate_key_manifest: str | Path,
    candidate_key_manifest_sha256: str,
    candidate_manifest: str | Path,
    candidate_manifest_sha256: str,
    query_features: str | Path,
    query_features_sha256: str,
    split: str = "train",
    rank: int = 16,
    pool_size: int = 6,
    threads: int = 12,
) -> dict[str, Any]:
    started = time.perf_counter()
    records_source = Path(records_path).expanduser().resolve()
    if sha256_file(records_source) != records_sha256.lower():
        raise ValueError("native masked replay records digest changed")
    library_source = Path(native_library).expanduser().resolve()
    if not library_source.is_file():
        raise ValueError("native masked replay library is unavailable")
    package = Path(package_path).expanduser().resolve()
    config_path = package / "model/config.json"
    non_mlp_path = package / "transformer/non_mlp.safetensors"
    q7_path = package / "mlp/experts.q7"
    for source in (config_path, non_mlp_path, q7_path):
        if not source.is_file():
            raise ValueError("native masked replay package is incomplete")
    records = _checked_records(records_source, split=split)
    keys, key_metadata = qk.full.load_stacked_full_visible_qk_candidate_key_trace(
        candidate_key_manifest,
        candidate_key_manifest_sha256,
    )
    candidates, candidate_metadata = qk.full.load_stacked_full_visible_qk_candidate_trace(
        candidate_manifest,
        candidate_manifest_sha256,
    )
    qk._cross_check_shards(candidate_metadata, key_metadata)
    queries, positions = selector._load_authenticated_query_features(
        query_features,
        query_features_sha256,
    )
    candidate_scores = np.sum(candidates, axis=-1)
    masks = selector.build_query_key_exact_rerank_masks(
        queries,
        positions,
        keys,
        candidate_scores,
        rank=rank,
        pool_size=pool_size,
    )
    anchors = oracle._fact_anchor_ids(package / "tokenizer/tokenizer.json")
    schedules = [
        oracle._derive_schedule(record["input_ids"], anchors)
        for record in records
    ]
    runtime = OLMoENativeTokenRuntime(
        str(config_path),
        str(non_mlp_path),
        str(q7_path),
        str(library_source),
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
            _progress(f"record {index + 1}/{len(records)} baseline")
            baseline = _run_record(runtime, record=record, schedule=schedule, masks=None)
            runtime.reset()
            _progress(f"record {index + 1}/{len(records)} rank{rank}/pool{pool_size}")
            masked = _run_record(
                runtime,
                record=record,
                schedule=schedule,
                masks=masks[index],
            )
            row = _record_metrics(baseline, masked, record)
            rows.append(row)
            _progress(
                f"record {index + 1}/{len(records)} agreement="
                f"{row['answer_top1_agreement']:.4f} hidden="
                f"{row['answer_hidden_relative_l2_mean']:.4f}"
            )
    finally:
        runtime.close()
    aggregate = {
        "answer_hidden_relative_l2_mean": float(
            np.mean([row["answer_hidden_relative_l2_mean"] for row in rows])
        ),
        "answer_hidden_relative_l2_p90": float(
            np.quantile([row["answer_hidden_relative_l2_p90"] for row in rows], 0.90)
        ),
        "answer_logits_relative_l2_mean": float(
            np.mean([row["answer_logits_relative_l2_mean"] for row in rows])
        ),
        "answer_logits_relative_l2_p90": float(
            np.quantile([row["answer_logits_relative_l2_p90"] for row in rows], 0.90)
        ),
        "answer_top1_agreement_mean": float(
            np.mean([row["answer_top1_agreement"] for row in rows])
        ),
        "all_position_top1_agreement_mean": float(
            np.mean([row["all_position_top1_agreement"] for row in rows])
        ),
        "answer_nll_delta_mean": float(
            np.mean([row["answer_nll_delta"] for row in rows])
        ),
        "answer_nll_delta_max": float(
            np.max([row["answer_nll_delta"] for row in rows])
        ),
        "prefix_hidden_abs_max": float(
            np.max([row["prefix_hidden_abs_max"] for row in rows])
        ),
        "prefix_logits_abs_max": float(
            np.max([row["prefix_logits_abs_max"] for row in rows])
        ),
    }
    passed = bool(
        aggregate["answer_hidden_relative_l2_mean"]
        <= _THRESHOLDS["maximum_hidden_relative_l2"]
        and aggregate["answer_logits_relative_l2_mean"]
        <= _THRESHOLDS["maximum_logits_relative_l2"]
        and aggregate["answer_nll_delta_mean"]
        <= _THRESHOLDS["maximum_answer_nll_delta"]
        and aggregate["answer_top1_agreement_mean"]
        >= _THRESHOLDS["minimum_answer_top1_agreement"]
        and all(row["passed"] for row in rows)
    )
    return {
        "schema_version": 1,
        "experiment": "olmoe_q7_retrieval_episodic_native_masked_replay",
        "status": "train_native_masked_replay_passed" if passed else "train_native_masked_replay_failed",
        "scope": {
            "split": split,
            "records": len(records),
            "record_held_out_selector": True,
            "native_runtime": True,
            "cpu_only": True,
        },
        "selector": {
            "rank": rank,
            "pool_size": pool_size,
            "native_top_k": 4,
            "prefix_rows_unmasked": _ANSWER_START,
            "older_candidate_slots_retained_fraction": float(pool_size / 8.0),
            "older_candidate_qk_value_read_fraction": float(pool_size / 8.0),
            "traffic_scope": "episodic_older_candidate_stage_only",
        },
        "thresholds": _THRESHOLDS,
        "aggregate": aggregate,
        "records": rows,
        "artifacts": {
            "records": {"path": str(records_source), "sha256": records_sha256.lower()},
            "package": str(package),
            "native_library": {
                "path": str(library_source),
                "sha256": sha256_file(library_source),
            },
            "candidate_keys": {
                "path": str(Path(candidate_key_manifest).resolve()),
                "sha256": candidate_key_manifest_sha256.lower(),
            },
            "candidate_scores": {
                "path": str(Path(candidate_manifest).resolve()),
                "sha256": candidate_manifest_sha256.lower(),
            },
            "query_features": {
                "path": str(Path(query_features).resolve()),
                "sha256": query_features_sha256.lower(),
            },
        },
        "feature_locality_only": False,
        "causal_replay_executed": True,
        "confirmation_split_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--records-sha256", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--native-library", required=True)
    parser.add_argument("--candidate-key-manifest", required=True)
    parser.add_argument("--candidate-key-manifest-sha256", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--query-features", required=True)
    parser.add_argument("--query-features-sha256", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--pool-size", type=int, default=6)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_native_masked_replay(
        records_path=args.records,
        records_sha256=args.records_sha256,
        package_path=args.package,
        native_library=args.native_library,
        candidate_key_manifest=args.candidate_key_manifest,
        candidate_key_manifest_sha256=args.candidate_key_manifest_sha256,
        candidate_manifest=args.candidate_manifest,
        candidate_manifest_sha256=args.candidate_manifest_sha256,
        query_features=args.query_features,
        query_features_sha256=args.query_features_sha256,
        split=args.split,
        rank=args.rank,
        pool_size=args.pool_size,
        threads=args.threads,
    )
    atomic_json(args.out, report)
    print(__import__("json").dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
