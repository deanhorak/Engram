"""Trace-only screening for predictor-free Dynamic Input Pruning.

The sweep evaluates an Engram extension of Dynamic Input Pruning (DIP): use
the largest-magnitude coordinates of each MLP input to approximate every
SwiGLU activation, keep a bounded candidate set, complete those candidates
with the omitted input coordinates, and then rerank them exactly.  The
reference implementation performs ordinary dense NumPy matrix products; its
byte counts describe the proposed sparse kernel and are not timing claims.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from engram.models.inspection import inspect_model, load_layer_mlp, resolve_model_path
from engram.semantic.dip import input_coordinate_count, projected_dip_traffic
from engram.semantic.swiglu import neuron_activations
from engram.tracing.format import TraceReader
from engram.utils import percentile, sha256_json

DIP_REFERENCE = {
    "title": "Efficient LLM Inference using Dynamic Input Pruning and Cache-Aware Masking",
    "url": "https://arxiv.org/abs/2412.01380",
}


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
        batch = np.asarray(shard[field], dtype=np.float32)
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


def _sequence_hashes(reader: TraceReader, limit: int | None = None) -> list[str]:
    sequences: dict[int, list[int]] = defaultdict(list)
    count = 0
    for shard in reader.iter_shards(["sample_id", "token_id"]):
        sample_ids = np.asarray(shard["sample_id"])
        token_ids = np.asarray(shard["token_id"])
        if (
            sample_ids.ndim != 1
            or token_ids.ndim != 1
            or sample_ids.shape != token_ids.shape
        ):
            raise ValueError("trace sample_id/token_id fields must be matching vectors")
        if limit is not None:
            remaining = limit - count
            if remaining <= 0:
                break
            sample_ids = sample_ids[:remaining]
            token_ids = token_ids[:remaining]
        for sample_id, token_id in zip(sample_ids, token_ids, strict=True):
            if not np.issubdtype(type(sample_id), np.integer) or not np.issubdtype(
                type(token_id), np.integer
            ):
                raise ValueError("trace sample_id/token_id fields must be integral")
            sequences[int(sample_id)].append(int(token_id))
        count += len(sample_ids)
    if not sequences:
        raise ValueError("trace contains no token-sequence provenance")
    return [sha256_json({"input_ids": sequences[index]}) for index in sorted(sequences)]


def _stable_top_k(scores: np.ndarray, count: int) -> np.ndarray:
    """Return score-descending, index-ascending top-k IDs for each row."""

    values = np.asarray(scores)
    if values.ndim != 2 or count <= 0 or count > values.shape[1]:
        raise ValueError("scores must be [N, width] and count must lie within width")
    return np.argsort(-values, axis=1, kind="stable")[:, :count]


def _partial_proxy_scores(
    states: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    value_norms: np.ndarray,
    coordinate_count: int,
) -> np.ndarray:
    coordinates = _stable_top_k(np.abs(states), coordinate_count)
    partial = np.zeros_like(states)
    np.put_along_axis(
        partial, coordinates, np.take_along_axis(states, coordinates, axis=1), axis=1
    )
    return np.abs(neuron_activations(partial, gate, up)) * value_norms[None, :]


def _rerank_candidates(
    exact_scores: np.ndarray,
    candidates: np.ndarray,
    top_k: int,
) -> np.ndarray:
    # Sort IDs first so a stable score sort has the same tie rule as a full-width
    # stable sort, independent of the proxy order used to produce candidates.
    by_id = np.sort(candidates, axis=1)
    candidate_scores = np.take_along_axis(exact_scores, by_id, axis=1)
    within = _stable_top_k(candidate_scores, top_k)
    return np.take_along_axis(by_id, within, axis=1)


def _relative_and_cosine_rows(
    approximation: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    approximate = np.asarray(approximation, dtype=np.float64)
    exact = np.asarray(reference, dtype=np.float64)
    if approximate.shape != exact.shape or approximate.ndim != 2:
        raise ValueError("approximation and reference must be matching matrices")
    exact_norm = np.linalg.norm(exact, axis=1)
    approximate_norm = np.linalg.norm(approximate, axis=1)
    relative = np.linalg.norm(approximate - exact, axis=1) / np.maximum(
        exact_norm, 1e-12
    )
    denominator = exact_norm * approximate_norm
    cosine = np.sum(approximate * exact, axis=1) / np.maximum(denominator, 1e-12)
    cosine[(exact_norm <= 1e-12) & (approximate_norm <= 1e-12)] = 1.0
    return relative, np.clip(cosine, -1.0, 1.0)


def _sparse_output(
    exact_activations: np.ndarray, selected: np.ndarray, down: np.ndarray
) -> np.ndarray:
    retained = np.zeros_like(exact_activations)
    np.put_along_axis(
        retained,
        selected,
        np.take_along_axis(exact_activations, selected, axis=1),
        axis=1,
    )
    return retained @ down.T


def _traffic(
    hidden_size: int,
    intermediate_size: int,
    *,
    coordinate_count: int,
    candidate_count: int,
    top_k: int,
    bytes_per_element: int = 4,
) -> dict[str, int | float]:
    fraction = coordinate_count / hidden_size
    traffic = projected_dip_traffic(
        hidden_size,
        intermediate_size,
        input_fraction=fraction,
        candidate_count=candidate_count,
        top_k=top_k,
        bytes_per_element=bytes_per_element,
    )
    if traffic.input_count != coordinate_count:
        raise RuntimeError("DIP traffic coordinate count differs from evaluated budget")
    return {
        "bytes_per_element": bytes_per_element,
        "partial_gate_up_elements_per_token_layer": traffic.partial_projection_elements,
        "candidate_completion_elements_per_token_layer": traffic.candidate_completion_elements,
        "selected_down_elements_per_token_layer": traffic.selected_down_elements,
        "projected_weight_elements_per_token_layer": traffic.total_elements,
        "dense_weight_elements_per_token_layer": traffic.dense_elements,
        "projected_weight_bytes_per_token_layer": traffic.total_bytes,
        "dense_weight_bytes_per_token_layer": traffic.dense_bytes,
        "projected_fraction_of_dense": traffic.fraction_of_dense,
        "projected_dense_over_sparse_reduction": traffic.reduction_factor,
    }


def _pareto_frontier(arms: Sequence[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for candidate in arms:
        dominated = False
        for other in arms:
            if other is candidate:
                continue
            no_worse = (
                other["projected_traffic"]["projected_fraction_of_dense"]
                <= candidate["projected_traffic"]["projected_fraction_of_dense"]
                and other["candidate_recall"]["mean"]
                >= candidate["candidate_recall"]["mean"]
                and other["mlp_output_relative_l2"]["mean"]
                <= candidate["mlp_output_relative_l2"]["mean"]
            )
            strictly_better = (
                other["projected_traffic"]["projected_fraction_of_dense"]
                < candidate["projected_traffic"]["projected_fraction_of_dense"]
                or other["candidate_recall"]["mean"]
                > candidate["candidate_recall"]["mean"]
                or other["mlp_output_relative_l2"]["mean"]
                < candidate["mlp_output_relative_l2"]["mean"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(candidate["name"])
    return result


def evaluate_dip_exact_completion_sweep(
    model: str | Path,
    validation_traces: str | Path,
    *,
    input_fractions: Sequence[float],
    top_k: int = 768,
    candidate_counts: Sequence[int] = (896, 1024),
    validation_records: int | None = None,
) -> dict[str, Any]:
    """Measure DIP candidate completion on held-out dense-teacher states."""

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    validation = TraceReader(validation_traces)
    if validation.manifest["model_hash"] != inspection.source_hash:
        raise ValueError("trace/model hash mismatch")
    if validation.manifest["split"] != "validation":
        raise ValueError("expected 'validation' traces")
    if validation_records is not None and validation_records <= 0:
        raise ValueError("validation_records must be positive")
    fractions = tuple(dict.fromkeys(float(value) for value in input_fractions))
    if not fractions:
        raise ValueError("at least one input fraction is required")
    coordinate_counts = {
        fraction: input_coordinate_count(inspection.hidden_size, fraction)
        for fraction in fractions
    }
    if top_k <= 0 or top_k > inspection.intermediate_size:
        raise ValueError("top_k must lie within the intermediate size")
    candidates = tuple(dict.fromkeys(int(value) for value in candidate_counts))
    if not candidates or any(
        value < top_k or value > inspection.intermediate_size for value in candidates
    ):
        raise ValueError(
            "candidate counts must lie between top_k and intermediate size"
        )

    sequence_hashes = _sequence_hashes(validation, validation_records)
    arm_keys = [(fraction, count) for fraction in fractions for count in candidates]
    aggregate: dict[tuple[float, int], dict[str, list[float]]] = {
        key: {
            "candidate_recall": [],
            "oracle_score_mass_recall": [],
            "relative_l2": [],
            "cosine": [],
        }
        for key in arm_keys
    }
    layer_means: dict[tuple[float, int], dict[str, list[float]]] = {
        key: {"candidate_recall": [], "oracle_score_mass_recall": [], "relative_l2": []}
        for key in arm_keys
    }
    oracle_relative_l2: list[float] = []
    oracle_cosine: list[float] = []
    per_layer: list[dict[str, Any]] = []
    validation_count: int | None = None

    for layer in range(inspection.num_hidden_layers):
        states = _load_states(validation, layer, validation_records)
        validation_count = len(states)
        gate, up, down = load_layer_mlp(model_path, layer)
        exact_activations = neuron_activations(states, gate, up)
        value_norms = np.linalg.norm(down, axis=0)
        exact_scores = np.abs(exact_activations) * value_norms[None, :]
        oracle_ids = _stable_top_k(exact_scores, top_k)
        oracle_membership = np.zeros(exact_scores.shape, dtype=bool)
        np.put_along_axis(oracle_membership, oracle_ids, True, axis=1)
        oracle_scores = np.take_along_axis(exact_scores, oracle_ids, axis=1)
        oracle_mass = np.maximum(np.sum(oracle_scores, axis=1), 1e-30)
        exact_output = exact_activations @ down.T
        oracle_output = _sparse_output(exact_activations, oracle_ids, down)
        oracle_relative, oracle_layer_cosine = _relative_and_cosine_rows(
            oracle_output, exact_output
        )
        oracle_relative_l2.extend(oracle_relative.tolist())
        oracle_cosine.extend(oracle_layer_cosine.tolist())
        layer_result: dict[str, Any] = {
            "layer": layer,
            "records": len(states),
            "oracle": {
                "mlp_output_relative_l2": _stats(oracle_relative),
                "mlp_output_cosine": _stats(oracle_layer_cosine),
            },
            "arms": [],
        }

        for fraction in fractions:
            coordinate_count = coordinate_counts[fraction]
            proxy_scores = _partial_proxy_scores(
                states, gate, up, value_norms, coordinate_count
            )
            proxy_order = _stable_top_k(proxy_scores, max(candidates))
            for candidate_count in candidates:
                key = (fraction, candidate_count)
                candidate_ids = proxy_order[:, :candidate_count]
                hits = np.sum(
                    np.take_along_axis(oracle_membership, candidate_ids, axis=1), axis=1
                )
                recall = hits / float(top_k)
                captured_oracle_scores = np.where(
                    np.take_along_axis(oracle_membership, candidate_ids, axis=1),
                    np.take_along_axis(exact_scores, candidate_ids, axis=1),
                    0.0,
                )
                mass_recall = np.sum(captured_oracle_scores, axis=1) / oracle_mass
                selected = _rerank_candidates(exact_scores, candidate_ids, top_k)
                sparse_output = _sparse_output(exact_activations, selected, down)
                relative, cosine = _relative_and_cosine_rows(
                    sparse_output, exact_output
                )
                aggregate[key]["candidate_recall"].extend(recall.tolist())
                aggregate[key]["oracle_score_mass_recall"].extend(mass_recall.tolist())
                aggregate[key]["relative_l2"].extend(relative.tolist())
                aggregate[key]["cosine"].extend(cosine.tolist())
                layer_means[key]["candidate_recall"].append(float(np.mean(recall)))
                layer_means[key]["oracle_score_mass_recall"].append(
                    float(np.mean(mass_recall))
                )
                layer_means[key]["relative_l2"].append(float(np.mean(relative)))
                layer_result["arms"].append(
                    {
                        "name": f"dip_q{coordinate_count}_c{candidate_count}_k{top_k}",
                        "input_fraction": fraction,
                        "input_coordinate_count": coordinate_count,
                        "candidate_count": candidate_count,
                        "candidate_recall": _stats(recall),
                        "oracle_score_mass_recall": _stats(mass_recall),
                        "mlp_output_relative_l2": _stats(relative),
                        "mlp_output_cosine": _stats(cosine),
                    }
                )
        per_layer.append(layer_result)

    arms: list[dict[str, Any]] = []
    for fraction, candidate_count in arm_keys:
        key = (fraction, candidate_count)
        recall = _stats(aggregate[key]["candidate_recall"])
        mass_recall = _stats(aggregate[key]["oracle_score_mass_recall"])
        relative = _stats(aggregate[key]["relative_l2"])
        coordinate_count = coordinate_counts[fraction]
        arm = {
            "name": f"dip_q{coordinate_count}_c{candidate_count}_k{top_k}",
            "input_fraction": fraction,
            "input_coordinate_count": coordinate_count,
            "candidate_count": candidate_count,
            "top_k": top_k,
            "candidate_recall": recall,
            "oracle_score_mass_recall": mass_recall,
            "mlp_output_relative_l2": relative,
            "mlp_output_cosine": _stats(aggregate[key]["cosine"]),
            "layer_mean_candidate_recall": _stats(layer_means[key]["candidate_recall"]),
            "layer_mean_oracle_score_mass_recall": _stats(
                layer_means[key]["oracle_score_mass_recall"]
            ),
            "layer_mean_mlp_output_relative_l2": _stats(
                layer_means[key]["relative_l2"]
            ),
            "projected_traffic": _traffic(
                inspection.hidden_size,
                inspection.intermediate_size,
                coordinate_count=coordinate_count,
                candidate_count=candidate_count,
                top_k=top_k,
            ),
            "meets_existing_candidate_recall_gate": recall["mean"] >= 0.95,
            "near_oracle_screen": (
                recall["mean"] >= 0.99 and mass_recall["mean"] >= 0.995
            ),
        }
        arms.append(arm)

    near_oracle = [arm for arm in arms if arm["near_oracle_screen"]]
    eligible = [arm for arm in arms if arm["meets_existing_candidate_recall_gate"]]
    selection_pool = near_oracle or eligible or arms
    recommended = min(
        selection_pool,
        key=lambda arm: (
            arm["projected_traffic"]["projected_fraction_of_dense"],
            arm["mlp_output_relative_l2"]["mean"],
            -arm["candidate_recall"]["mean"],
        ),
    )
    oracle_summary = {
        "mlp_output_relative_l2": _stats(oracle_relative_l2),
        "mlp_output_cosine": _stats(oracle_cosine),
    }
    return {
        "schema_version": 1,
        "experiment": "dynamic_input_pruning_exact_completion_sweep",
        "status": "measured_local_model",
        "source_model_hash": inspection.source_hash,
        "method": {
            "basis": DIP_REFERENCE,
            "engram_extension": (
                "After partial gate/up scoring, complete omitted input-coordinate products only "
                "for candidates and exactly rerank their full contribution-magnitude scores."
            ),
            "candidate_score": "abs(partial_swiglu_activation) * l2_norm(down_column)",
            "exact_score": "abs(exact_swiglu_activation) * l2_norm(down_column)",
            "tie_break": "stable ascending record index",
            "trained_parameters": 0,
        },
        "validation": {
            "trace_path": str(Path(validation_traces).resolve()),
            "dataset_hash": validation.manifest["dataset_hash"],
            "records_per_layer": validation_count,
            "sequence_count": len(sequence_hashes),
            "unique_sequence_count": len(set(sequence_hashes)),
            "split": validation.manifest["split"],
        },
        "configuration": {
            "hidden_size": inspection.hidden_size,
            "intermediate_size": inspection.intermediate_size,
            "num_hidden_layers": inspection.num_hidden_layers,
            "top_k": top_k,
            "input_fractions": list(fractions),
            "input_coordinate_counts": [
                coordinate_counts[value] for value in fractions
            ],
            "candidate_counts": list(candidates),
            "validation_records_limit": validation_records,
        },
        "oracle": oracle_summary,
        "arms": arms,
        "per_layer": per_layer,
        "pareto_frontier": _pareto_frontier(arms),
        "candidate_recall_gate": 0.95,
        "near_oracle_screen_recall": 0.99,
        "near_oracle_screen_score_mass_recall": 0.995,
        "recommended_arm": recommended,
        "screening_decision": (
            "eligible_for_causal_intervention"
            if recommended["meets_existing_candidate_recall_gate"]
            else "reject_before_causal_intervention"
        ),
        "measurement_caveat": (
            "The NumPy evaluator executes dense matrix products. Traffic is a logical float32 "
            "weight-read projection for a future sparse kernel, excludes indexes/activations/cache "
            "effects, and is not measured DRAM traffic or latency."
        ),
        "scope_caveat": (
            "This clean-state trace screen does not measure hidden-state drift, logit KL, target "
            "NLL, or top-1 agreement; the trained-teacher causal intervention remains decisive."
        ),
    }


__all__ = ["DIP_REFERENCE", "evaluate_dip_exact_completion_sweep"]
