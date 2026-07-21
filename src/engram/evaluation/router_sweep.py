"""Recall-only screening for low-rank oracle-membership routers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from engram.models.inspection import inspect_model, load_layer_mlp, resolve_model_path
from engram.semantic.multilabel_router import LowRankMultiLabelRouter
from engram.semantic.swiglu import neuron_activations
from engram.tracing.format import TraceReader
from engram.utils import percentile, sha256_json


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError("cannot summarize empty or non-finite metrics")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": percentile(array, 95),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _load_states(reader: TraceReader, layer: int, limit: int | None) -> np.ndarray:
    field = f"layer_{layer}_mlp_input"
    batches: list[np.ndarray] = []
    count = 0
    for shard in reader.iter_shards([field]):
        batch = np.asarray(shard[field], dtype=np.float64)
        if limit is not None:
            remaining = limit - count
            if remaining <= 0:
                break
            batch = batch[:remaining]
        batches.append(batch)
        count += len(batch)
    if not batches:
        raise ValueError(f"trace contains no MLP inputs for layer {layer}")
    return np.concatenate(batches)


def _sequence_hashes(reader: TraceReader) -> list[str]:
    sequences: dict[int, list[int]] = defaultdict(list)
    for shard in reader.iter_shards(["sample_id", "token_id"]):
        sample_ids = np.asarray(shard["sample_id"])
        token_ids = np.asarray(shard["token_id"])
        if sample_ids.ndim != 1 or token_ids.ndim != 1 or sample_ids.shape != token_ids.shape:
            raise ValueError("trace sample_id/token_id fields must be matching vectors")
        for sample_id, token_id in zip(sample_ids, token_ids, strict=True):
            if not np.issubdtype(type(sample_id), np.integer) or not np.issubdtype(
                type(token_id), np.integer
            ):
                raise ValueError("trace sample_id/token_id fields must be integral")
            sequences[int(sample_id)].append(int(token_id))
    if not sequences:
        raise ValueError("trace contains no token-sequence provenance")
    return [
        sha256_json({"input_ids": sequences[index]}) for index in sorted(sequences)
    ]


def _membership(
    states: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    top_k: int,
) -> np.ndarray:
    activations = neuron_activations(states, gate, up)
    scores = np.abs(activations) * np.linalg.norm(down, axis=0)[None, :]
    order = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    membership = np.zeros(scores.shape, dtype=bool)
    membership[np.arange(scores.shape[0])[:, None], order] = True
    return membership


def _cache_metadata(
    *,
    source_hash: str,
    calibration_hash: str,
    validation_hash: str,
    layer: int,
    top_k: int,
    intermediate_size: int,
    calibration_records: int | None,
    validation_records: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_model_hash": source_hash,
        "calibration_dataset_hash": calibration_hash,
        "validation_dataset_hash": validation_hash,
        "layer": layer,
        "top_k": top_k,
        "intermediate_size": intermediate_size,
        "calibration_records_limit": calibration_records,
        "validation_records_limit": validation_records,
    }


def _load_or_build_membership(
    cache_dir: Path,
    metadata: dict[str, Any],
    calibration_states: np.ndarray,
    validation_states: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    path = cache_dir / f"layer_{metadata['layer']:03d}_top_{metadata['top_k']}.npz"
    expected = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    if path.is_file():
        with np.load(path, allow_pickle=False) as cached:
            actual = str(np.asarray(cached["metadata"]).item())
            if actual == expected:
                width = int(metadata["intermediate_size"])
                calibration = np.unpackbits(cached["calibration"], axis=1)[:, :width].astype(bool)
                validation = np.unpackbits(cached["validation"], axis=1)[:, :width].astype(bool)
                if calibration.shape[0] == len(calibration_states) and validation.shape[0] == len(
                    validation_states
                ):
                    return calibration, validation, True

    calibration = _membership(calibration_states, gate, up, down, int(metadata["top_k"]))
    validation = _membership(validation_states, gate, up, down, int(metadata["top_k"]))
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        metadata=np.asarray(expected),
        calibration=np.packbits(calibration, axis=1),
        validation=np.packbits(validation, axis=1),
    )
    temporary.replace(path)
    return calibration, validation, False


def evaluate_rank_router_regularization_sweep(
    model: str | Path,
    calibration_traces: str | Path,
    validation_traces: str | Path,
    *,
    regularizations: Sequence[float],
    top_k: int = 768,
    candidate_counts: Sequence[int] = (1280,),
    rank: int = 16,
    calibration_records: int | None = None,
    validation_records: int | None = None,
    cache_dir: str | Path,
) -> dict[str, Any]:
    """Fit several ridge strengths and measure held-out oracle-set recall only."""

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    calibration = TraceReader(calibration_traces)
    validation = TraceReader(validation_traces)
    if calibration.manifest["model_hash"] != inspection.source_hash or validation.manifest[
        "model_hash"
    ] != inspection.source_hash:
        raise ValueError("trace/model hash mismatch")
    if calibration.manifest["split"] != "calibration":
        raise ValueError("expected 'calibration' calibration traces")
    if validation.manifest["split"] != "validation":
        raise ValueError("expected 'validation' validation traces")
    if calibration_records is not None and calibration_records <= 0:
        raise ValueError("calibration_records must be positive")
    if validation_records is not None and validation_records <= 0:
        raise ValueError("validation_records must be positive")
    regularization_values = tuple(dict.fromkeys(float(value) for value in regularizations))
    if not regularization_values or any(not np.isfinite(value) or value <= 0 for value in regularization_values):
        raise ValueError("regularizations must be finite and positive")
    candidate_values = tuple(dict.fromkeys(int(value) for value in candidate_counts))
    if top_k <= 0 or top_k > inspection.intermediate_size:
        raise ValueError("top_k must lie within the intermediate size")
    if not candidate_values or any(
        value < top_k or value > inspection.intermediate_size for value in candidate_values
    ):
        raise ValueError("candidate counts must lie between top_k and intermediate size")
    if rank <= 0 or rank > min(inspection.hidden_size, inspection.intermediate_size):
        raise ValueError("rank must lie within the router dimensions")

    calibration_sequences = _sequence_hashes(calibration)
    validation_sequences = _sequence_hashes(validation)
    overlap = set(calibration_sequences).intersection(validation_sequences)
    if overlap:
        raise ValueError("calibration and validation traces contain matching token sequences")

    cache_path = Path(cache_dir)
    recalls: dict[tuple[float, int], list[float]] = {
        (value, candidates): []
        for value in regularization_values
        for candidates in candidate_values
    }
    per_layer: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    calibration_count = None
    validation_count = None
    for layer in range(inspection.num_hidden_layers):
        gate, up, down = (
            np.asarray(value, dtype=np.float64) for value in load_layer_mlp(model_path, layer)
        )
        fit_states = _load_states(calibration, layer, calibration_records)
        held_out_states = _load_states(validation, layer, validation_records)
        calibration_count = len(fit_states)
        validation_count = len(held_out_states)
        metadata = _cache_metadata(
            source_hash=inspection.source_hash,
            calibration_hash=calibration.manifest["dataset_hash"],
            validation_hash=validation.manifest["dataset_hash"],
            layer=layer,
            top_k=top_k,
            intermediate_size=inspection.intermediate_size,
            calibration_records=calibration_records,
            validation_records=validation_records,
        )
        fit_membership, held_out_membership, cache_hit = _load_or_build_membership(
            cache_path, metadata, fit_states, held_out_states, gate, up, down
        )
        cache_hits += int(cache_hit)
        cache_misses += int(not cache_hit)
        for regularization in regularization_values:
            router = LowRankMultiLabelRouter.fit(
                fit_states,
                fit_membership.astype(np.float64),
                rank=rank,
                regularization=regularization,
            )
            scores = (held_out_states @ router.input_factors) @ router.output_factors + router.bias
            layer_result: dict[str, Any] = {
                "layer": layer,
                "regularization": regularization,
                "router_parameter_bytes_float32": router.parameter_bytes(),
                "candidate_recall": {},
            }
            order = np.argsort(-scores, axis=1, kind="stable")
            for candidates in candidate_values:
                selected = order[:, :candidates]
                row_recall = np.sum(
                    np.take_along_axis(held_out_membership, selected, axis=1), axis=1
                ) / float(top_k)
                recalls[(regularization, candidates)].extend(row_recall.tolist())
                layer_result["candidate_recall"][str(candidates)] = _stats(row_recall)
            per_layer.append(layer_result)

    arms = []
    for regularization in regularization_values:
        for candidates in candidate_values:
            layer_means = [
                row["candidate_recall"][str(candidates)]["mean"]
                for row in per_layer
                if row["regularization"] == regularization
            ]
            arms.append(
                {
                    "regularization": regularization,
                    "candidate_count": candidates,
                    "candidate_recall": _stats(recalls[(regularization, candidates)]),
                    "layer_mean_candidate_recall": _stats(layer_means),
                    "meets_recall_gate": float(np.mean(recalls[(regularization, candidates)]))
                    >= 0.95,
                }
            )
    best = max(arms, key=lambda arm: (arm["candidate_recall"]["mean"], -arm["regularization"]))
    return {
        "schema_version": 1,
        "experiment": "rank_router_regularization_sweep",
        "status": "measured_local_model",
        "source_model_hash": inspection.source_hash,
        "calibration": {
            "trace_path": str(Path(calibration_traces).resolve()),
            "dataset_hash": calibration.manifest["dataset_hash"],
            "records_per_layer": calibration_count,
            "sequence_count": len(calibration_sequences),
            "unique_sequence_count": len(set(calibration_sequences)),
        },
        "validation": {
            "trace_path": str(Path(validation_traces).resolve()),
            "dataset_hash": validation.manifest["dataset_hash"],
            "records_per_layer": validation_count,
            "sequence_count": len(validation_sequences),
            "unique_sequence_count": len(set(validation_sequences)),
        },
        "data_separation": {
            "method": "exact_token_sequence_hashes",
            "overlapping_sequence_count": 0,
            "held_out": True,
        },
        "configuration": {
            "rank": rank,
            "top_k": top_k,
            "candidate_counts": list(candidate_values),
            "regularizations": list(regularization_values),
        },
        "membership_cache": {
            "path": str(cache_path.resolve()),
            "hits": cache_hits,
            "misses": cache_misses,
            "format": "per-layer packed-bit NPZ",
        },
        "arms": arms,
        "per_layer": per_layer,
        "recall_gate": 0.95,
        "best_arm": best,
        "screening_decision": (
            "eligible_for_causal_intervention"
            if best["meets_recall_gate"]
            else "reject_before_causal_intervention"
        ),
        "scope_caveat": (
            "This recall-only screen does not measure logit KL, NLL, hidden-state drift, "
            "latency, or realized memory traffic."
        ),
    }


__all__ = ["evaluate_rank_router_regularization_sweep"]
