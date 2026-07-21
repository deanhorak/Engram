from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from engram.models.inspection import inspect_model, load_layer_mlp
from engram.semantic.background import LowRankLinearBackground
from engram.semantic.memory import SemanticLayer
from engram.semantic.ivf import JointKeyIVFIndex
from engram.semantic.router import candidate_recall
from engram.tracing.format import TraceReader
from engram.utils import percentile


def _metrics(approximation: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    norm = float(np.linalg.norm(reference))
    error = float(np.linalg.norm(approximation - reference))
    relative = error / max(norm, 1e-12)
    approximation_norm = float(np.linalg.norm(approximation))
    cosine = float(np.dot(approximation, reference) / max(norm * approximation_norm, 1e-12))
    return relative, max(-1.0, min(1.0, cosine))


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": percentile(values, 95),
    }


def _load_layer_trace(reader: TraceReader, layer: int, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    hidden = []
    types = []
    count = 0
    fields = ["input_type", f"layer_{layer}_mlp_input"]
    for shard in reader.iter_shards(fields):
        batch = np.asarray(shard[fields[1]], dtype=np.float64)
        labels = np.asarray(shard["input_type"])
        if limit is not None:
            remaining = limit - count
            batch = batch[:remaining]
            labels = labels[:remaining]
        hidden.append(batch)
        types.append(labels.astype("U32"))
        count += len(batch)
        if limit is not None and count >= limit:
            break
    return np.concatenate(hidden), np.concatenate(types)


def evaluate_practical_routing(
    model: str | Path,
    calibration_traces: str | Path,
    validation_traces: str | Path,
    *,
    top_k: int,
    candidate_count: int,
    background_rank: int = 4,
    ivf_clusters: int = 8,
    ivf_probes: int = 2,
    max_records: int | None = None,
) -> dict[str, Any]:
    inspection = inspect_model(model)
    calibration = TraceReader(calibration_traces)
    validation = TraceReader(validation_traces)
    for trace, expected_split in ((calibration, "calibration"), (validation, "validation")):
        if trace.manifest["model_hash"] != inspection.source_hash:
            raise ValueError("trace/model hash mismatch")
        if trace.manifest["split"] != expected_split:
            raise ValueError(f"expected {expected_split!r} trace split")
    if top_k <= 0 or candidate_count < top_k or candidate_count > inspection.intermediate_size:
        raise ValueError("require 0 < top_k <= candidate_count <= intermediate_size")

    rows: list[dict[str, Any]] = []
    fitted_backgrounds = []
    for layer_index in range(inspection.num_hidden_layers):
        gate, up, down = load_layer_mlp(model, layer_index)
        values = down.T.astype(np.float64)
        memory = SemanticLayer(gate.astype(np.float64), up.astype(np.float64), values)
        router = JointKeyIVFIndex.build(
            gate,
            up,
            num_clusters=min(ivf_clusters, inspection.intermediate_size),
        )
        fit_hidden, _ = _load_layer_trace(calibration, layer_index, max_records)
        fit_residual = []
        for state in fit_hidden:
            candidates = router.search(
                state,
                probes=min(ivf_probes, router.centroids),
                candidate_count=candidate_count,
                expand_for_candidates=True,
            )
            sparse = memory.read(state, candidates.indices, top_k=top_k).output
            fit_residual.append(memory.full(state) - sparse)
        rank = min(background_rank, inspection.hidden_size)
        background = LowRankLinearBackground.fit(fit_hidden, np.asarray(fit_residual), rank=rank)
        fitted_backgrounds.append({"layer": layer_index, "metadata": background.metadata()})

        validation_hidden, validation_types = _load_layer_trace(validation, layer_index, max_records)
        for state, input_type in zip(validation_hidden, validation_types):
            full = memory.full(state)
            oracle_indices = memory.contribution_order(state)[:top_k]
            oracle = memory.read(state, oracle_indices, top_k=top_k)
            started = time.perf_counter_ns()
            candidate_result = router.search(
                state,
                probes=min(ivf_probes, router.centroids),
                candidate_count=candidate_count,
                expand_for_candidates=True,
            )
            routed = memory.read(state, candidate_result.indices, top_k=top_k)
            elapsed = time.perf_counter_ns() - started
            practical = routed.output
            recall = candidate_recall(candidate_result.indices, oracle_indices)
            oracle_error, oracle_cosine = _metrics(oracle.output, full)
            practical_error, practical_cosine = _metrics(practical, full)
            corrected = practical + background.predict(state)
            corrected_error, corrected_cosine = _metrics(corrected, full)
            candidate_bytes = (
                router.centroids_bytes
                + candidate_result.probed_record_count
                * 2
                * inspection.hidden_size
                * gate.dtype.itemsize
                + candidate_result.probed_record_count
                * router.posting_indices.dtype.itemsize
                + 2
                * candidate_result.probed_clusters.size
                * router.posting_offsets.dtype.itemsize
            )
            active_bytes = top_k * inspection.hidden_size * values.dtype.itemsize
            rows.append(
                {
                    "layer": layer_index,
                    "input_type": str(input_type),
                    "candidate_recall": recall.recall,
                    "candidate_precision": recall.precision,
                    "oracle_relative_l2": oracle_error,
                    "oracle_cosine": oracle_cosine,
                    "practical_relative_l2": practical_error,
                    "practical_cosine": practical_cosine,
                    "with_background_relative_l2": corrected_error,
                    "with_background_cosine": corrected_cosine,
                    "router_latency_ns": elapsed,
                    "estimated_index_bytes": candidate_bytes,
                    "index_storage_bytes": router.total_bytes,
                    "probed_clusters": candidate_result.probed_clusters.size,
                    "probed_records": candidate_result.probed_record_count,
                    "estimated_active_value_bytes": active_bytes,
                }
            )

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[("all",)].append(row)
        groups[("layer", row["layer"])].append(row)
        groups[("layer_input_type", row["layer"], row["input_type"])].append(row)
    output_groups = []
    metric_names = (
        "candidate_recall",
        "candidate_precision",
        "oracle_relative_l2",
        "oracle_cosine",
        "practical_relative_l2",
        "practical_cosine",
        "with_background_relative_l2",
        "with_background_cosine",
        "router_latency_ns",
        "estimated_index_bytes",
        "index_storage_bytes",
        "probed_clusters",
        "probed_records",
        "estimated_active_value_bytes",
    )
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        descriptor: dict[str, Any] = {"scope": key[0], "samples": len(group)}
        if key[0] != "all":
            descriptor["layer"] = key[1]
        if key[0] == "layer_input_type":
            descriptor["input_type"] = key[2]
        descriptor["metrics"] = {name: _stats([float(row[name]) for row in group]) for name in metric_names}
        output_groups.append(descriptor)
    fixture_only = bool(validation.manifest["metadata"].get("fixture_only"))
    return {
        "schema_version": 1,
        "experiment": "gate_2_practical_semantic_routing",
        "status": "pipeline_validation" if fixture_only else "measured_local_model",
        "fixture_only": fixture_only,
        "source_model_hash": inspection.source_hash,
        "calibration_dataset_hash": calibration.manifest["dataset_hash"],
        "validation_dataset_hash": validation.manifest["dataset_hash"],
        "top_k": top_k,
        "candidate_count": candidate_count,
        "router": "joint_key_ivf_plus_exact_candidate_rerank",
        "ivf_clusters": min(ivf_clusters, inspection.intermediate_size),
        "ivf_minimum_probes": min(ivf_probes, min(ivf_clusters, inspection.intermediate_size)),
        "background": "fitted_low_rank_linear_residual",
        "backgrounds": fitted_backgrounds,
        "end_to_end_logit_effect": {
            "status": "separate_evaluator_available",
            "command": "engram evaluate-mlp-intervention",
            "reason": (
                "Candidate proxy metrics do not imply language-model quality; run the "
                "trained-teacher intervention gate for the selected router and budget."
            ),
        },
        "groups": output_groups,
    }
