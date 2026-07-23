"""Structured block-expert SwiGLU experiments.

This module deliberately separates two questions:

* can a fixed-width block representation preserve the dense MLP locally; and
* can a cheap router select the useful blocks without reading the others?

The trace-only shadow evaluator answers those questions before an expensive
end-to-end sparse-teacher run.  The PyTorch wrapper executes the same hard
block path that a deployed kernel would use; its optional dense surrogate is
training-only and contributes gradients without changing the forward value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from engram.evaluation.gates import (
    MINIMUM_EVALUATION_SEQUENCES,
    MINIMUM_NEXT_TOKEN_POSITIONS,
    MINIMUM_UNIQUE_EVALUATION_SEQUENCES,
    MLP_QUALITY_THRESHOLDS,
)
from engram.evaluation.mlp_intervention import (
    _evaluation_sequence_hashes,
    _quality_metrics,
    _relative_and_cosine_rows,
)
from engram.evaluation.router_sweep import _load_states, _sequence_hashes
from engram.models.inspection import inspect_model, load_layer_mlp, resolve_model_path
from engram.semantic.multilabel_router import MultiLabelLinearRouter
from engram.semantic.swiglu import neuron_activations, silu
from engram.tracing.format import TraceReader
from engram.utils import atomic_json, sha256_file


@dataclass(frozen=True)
class StructuredExpertTraffic:
    hidden_size: int
    intermediate_size: int
    experts: int
    active_experts: int
    records_per_expert: int
    active_records: int
    selected_weight_bytes: int
    router_weight_bytes: int
    total_weight_bytes: int
    dense_weight_bytes: int
    fraction_of_dense: float

    def to_dict(self) -> dict[str, int | float]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class NativeGateTraffic:
    hidden_size: int
    intermediate_size: int
    input_fraction: float
    input_coordinates: int
    active_records: int
    gate_weight_bytes: int
    selected_up_down_weight_bytes: int
    total_weight_bytes: int
    dense_weight_bytes: int
    fraction_of_dense: float

    def to_dict(self) -> dict[str, int | float]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class LowRankUtilityResidual:
    input_factors: np.ndarray
    output_factors: np.ndarray
    bias: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.input_factors.shape[1])

    def predict(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.input_factors.shape[0]:
            raise ValueError("states have an incompatible shape")
        return (values @ self.input_factors) @ self.output_factors + self.bias

    def parameter_bytes(self, bytes_per_parameter: int = 4) -> int:
        if not isinstance(bytes_per_parameter, int) or bytes_per_parameter <= 0:
            raise ValueError("bytes_per_parameter must be a positive integer")
        return int(
            (
                self.input_factors.size
                + self.output_factors.size
                + self.bias.size
            )
            * bytes_per_parameter
        )


def structured_expert_traffic(
    hidden_size: int,
    intermediate_size: int,
    *,
    experts: int = 24,
    active_experts: int = 8,
    bytes_per_parameter: int = 4,
) -> StructuredExpertTraffic:
    """Return ideal inference weight traffic for contiguous expert blocks."""

    values = (hidden_size, intermediate_size, experts, active_experts)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("sizes and expert counts must be positive integers")
    if intermediate_size % experts:
        raise ValueError("experts must divide the intermediate size")
    if active_experts > experts:
        raise ValueError("active_experts cannot exceed experts")
    if not isinstance(bytes_per_parameter, int) or bytes_per_parameter <= 0:
        raise ValueError("bytes_per_parameter must be a positive integer")
    records_per_expert = intermediate_size // experts
    active_records = active_experts * records_per_expert
    selected = 3 * hidden_size * active_records * bytes_per_parameter
    router = (hidden_size * experts + experts) * bytes_per_parameter
    dense = 3 * hidden_size * intermediate_size * bytes_per_parameter
    total = selected + router
    return StructuredExpertTraffic(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        experts=experts,
        active_experts=active_experts,
        records_per_expert=records_per_expert,
        active_records=active_records,
        selected_weight_bytes=selected,
        router_weight_bytes=router,
        total_weight_bytes=total,
        dense_weight_bytes=dense,
        fraction_of_dense=total / dense,
    )


def native_gate_channel_traffic(
    hidden_size: int,
    intermediate_size: int,
    *,
    input_fraction: float,
    active_records: int,
    bytes_per_parameter: int = 4,
) -> NativeGateTraffic:
    """Return traffic for partial gate scoring plus selected up/down rows."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (hidden_size, intermediate_size, active_records)
    ):
        raise ValueError("sizes and active_records must be positive integers")
    if active_records > intermediate_size:
        raise ValueError("active_records cannot exceed the intermediate size")
    if not np.isfinite(input_fraction) or not 0 < input_fraction <= 1:
        raise ValueError("input_fraction must lie in (0, 1]")
    if not isinstance(bytes_per_parameter, int) or bytes_per_parameter <= 0:
        raise ValueError("bytes_per_parameter must be a positive integer")
    coordinates = max(1, min(hidden_size, round(input_fraction * hidden_size)))
    gate_bytes = intermediate_size * coordinates * bytes_per_parameter
    selected_bytes = 2 * hidden_size * active_records * bytes_per_parameter
    dense_bytes = 3 * hidden_size * intermediate_size * bytes_per_parameter
    total_bytes = gate_bytes + selected_bytes
    return NativeGateTraffic(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        input_fraction=input_fraction,
        input_coordinates=coordinates,
        active_records=active_records,
        gate_weight_bytes=gate_bytes,
        selected_up_down_weight_bytes=selected_bytes,
        total_weight_bytes=total_bytes,
        dense_weight_bytes=dense_bytes,
        fraction_of_dense=total_bytes / dense_bytes,
    )


def progressive_sparse_budget(
    intermediate_size: int,
    *,
    target_input_fraction: float,
    target_top_k: int,
    step: int,
    warmup_steps: int,
    anneal_steps: int,
) -> tuple[float, int]:
    """Linearly anneal dense execution to an exact target q/K budget."""

    if not isinstance(intermediate_size, int) or intermediate_size <= 0:
        raise ValueError("intermediate_size must be a positive integer")
    if not np.isfinite(target_input_fraction) or not 0 < target_input_fraction <= 1:
        raise ValueError("target_input_fraction must lie in (0, 1]")
    if not isinstance(target_top_k, int) or not 0 < target_top_k <= intermediate_size:
        raise ValueError("target_top_k must lie within the intermediate size")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (step, warmup_steps, anneal_steps)
    ):
        raise ValueError("step and schedule lengths must be nonnegative integers")
    if step < warmup_steps:
        return 1.0, intermediate_size
    progress = min(1.0, (step - warmup_steps + 1) / max(anneal_steps, 1))
    input_fraction = 1.0 - progress * (1.0 - target_input_fraction)
    top_k = round(
        intermediate_size - progress * (intermediate_size - target_top_k)
    )
    return input_fraction, max(target_top_k, min(intermediate_size, top_k))


def fit_low_rank_utility_residual(
    states: np.ndarray,
    targets: np.ndarray,
    *,
    rank: int,
    regularization: float,
) -> LowRankUtilityResidual:
    """Fit a truncated multi-output ridge model to continuous utility residuals."""

    state_matrix = np.asarray(states, dtype=np.float64)
    target_matrix = np.asarray(targets, dtype=np.float64)
    if state_matrix.ndim != 2 or target_matrix.ndim != 2:
        raise ValueError("states and targets must be rank-2 matrices")
    if not state_matrix.shape[0] or target_matrix.shape[0] != state_matrix.shape[0]:
        raise ValueError("states and targets must contain matching nonempty rows")
    if not np.all(np.isfinite(state_matrix)) or not np.all(np.isfinite(target_matrix)):
        raise ValueError("states and targets must be finite")
    maximum_rank = min(state_matrix.shape[0], state_matrix.shape[1], target_matrix.shape[1])
    if not isinstance(rank, int) or not 0 < rank <= maximum_rank:
        raise ValueError("rank exceeds the fitted matrix dimensions")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")

    state_mean = np.mean(state_matrix, axis=0)
    centered_states = state_matrix - state_mean
    scale = np.sqrt(np.mean(centered_states**2, axis=0))
    scale = np.where(scale > 1e-8, scale, 1.0)
    features = centered_states / scale
    target_mean = np.mean(target_matrix, axis=0)
    centered_targets = target_matrix - target_mean
    if state_matrix.shape[0] <= state_matrix.shape[1]:
        gram = features @ features.T
        gram.flat[:: gram.shape[0] + 1] += regularization
        dual = np.linalg.solve(gram, centered_targets)
        left_factor = features.T / scale[:, None]
        left_basis, left_reduced = np.linalg.qr(left_factor, mode="reduced")
        right_basis, right_reduced = np.linalg.qr(dual.T, mode="reduced")
        core = left_reduced @ right_reduced.T
        core_left, singular_values, core_right = np.linalg.svd(
            core, full_matrices=False
        )
        input_factors = (
            left_basis @ core_left[:, :rank]
        ) * singular_values[:rank]
        output_factors = core_right[:rank] @ right_basis.T
    else:
        gram = features.T @ features
        gram.flat[:: gram.shape[0] + 1] += regularization
        standardized = np.linalg.solve(
            gram, features.T @ centered_targets
        )
        weights = standardized / scale[:, None]
        eigenvalues, eigenvectors = np.linalg.eigh(weights @ weights.T)
        selected = np.argsort(eigenvalues, kind="stable")[-rank:][::-1]
        singular_values = np.sqrt(np.maximum(eigenvalues[selected], 0.0))
        left = eigenvectors[:, selected]
        input_factors = left * singular_values
        output_factors = np.zeros(
            (rank, weights.shape[1]), dtype=np.float64
        )
        nonzero = singular_values > np.finfo(np.float64).eps
        output_factors[nonzero] = (
            left[:, nonzero].T @ weights
        ) / singular_values[nonzero, None]
    bias = target_mean - (state_mean @ input_factors) @ output_factors
    return LowRankUtilityResidual(input_factors, output_factors, bias)


def load_native_gate_utility_residual(
    artifact: str | Path,
    layer: int,
    *,
    expected_source_hash: str | None = None,
) -> tuple[LowRankUtilityResidual, float]:
    """Load one layer of a validated native-gate residual artifact."""

    if not isinstance(layer, int) or layer < 0:
        raise ValueError("layer must be a nonnegative integer")
    from safetensors import safe_open

    path = Path(artifact)
    with safe_open(path, framework="np") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != "engram_native_gate_utility_residual_v1":
            raise ValueError("unsupported native-gate utility residual format")
        if (
            expected_source_hash is not None
            and metadata.get("source_model_hash") != expected_source_hash
        ):
            raise ValueError("native-gate residual/model hash mismatch")
        prefix = f"layers.{layer}.utility_residual"
        try:
            predictor = LowRankUtilityResidual(
                handle.get_tensor(f"{prefix}.input_factors"),
                handle.get_tensor(f"{prefix}.output_factors"),
                handle.get_tensor(f"{prefix}.bias"),
            )
            blend = float(metadata["blend"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid native-gate residual layer {layer}") from exc
    if not np.isfinite(blend) or blend < 0:
        raise ValueError("native-gate residual blend must be finite and nonnegative")
    return predictor, blend


def balanced_expert_permutation(
    features: np.ndarray, experts: int, *, iterations: int = 12
) -> np.ndarray:
    """Cluster records with deterministic equal-capacity cosine k-means."""

    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("features must be a non-empty record-by-feature matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must be finite")
    if not isinstance(experts, int) or experts <= 0 or matrix.shape[0] % experts:
        raise ValueError("experts must be positive and divide the record count")
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")

    records = matrix.shape[0]
    capacity = records // experts
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0.0)
    selected = [0]
    nearest = np.sum((normalized - normalized[0]) ** 2, axis=1)
    while len(selected) < experts:
        nearest[np.asarray(selected)] = -1.0
        record = int(np.argmax(nearest))
        selected.append(record)
        nearest = np.minimum(
            nearest, np.sum((normalized - normalized[record]) ** 2, axis=1)
        )
    centroids = normalized[np.asarray(selected)].copy()
    assignments = np.zeros(records, dtype=np.int64)
    previous = None
    for _ in range(iterations):
        scores = normalized @ centroids.T
        preferences = np.argsort(-scores, axis=1, kind="stable")
        if experts == 1:
            priority = np.arange(records)
        else:
            confidence = scores[np.arange(records), preferences[:, 0]] - scores[
                np.arange(records), preferences[:, 1]
            ]
            priority = np.lexsort((np.arange(records), -confidence))
        populations = np.zeros(experts, dtype=np.int64)
        for record in priority:
            for expert in preferences[record]:
                if populations[expert] < capacity:
                    assignments[record] = expert
                    populations[expert] += 1
                    break
        if previous is not None and np.array_equal(assignments, previous):
            break
        for expert in range(experts):
            centroid = np.mean(normalized[assignments == expert], axis=0)
            norm = float(np.linalg.norm(centroid))
            centroids[expert] = centroid / norm if norm > 0.0 else centroid
        previous = assignments.copy()
    return np.concatenate(
        [np.flatnonzero(assignments == expert) for expert in range(experts)]
    )


def block_contributions(
    states: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    permutation: np.ndarray,
    *,
    experts: int,
) -> np.ndarray:
    """Return one output vector per state and physical expert block."""

    hidden = np.asarray(states, dtype=np.float64)
    gate_matrix = np.asarray(gate, dtype=np.float64)
    up_matrix = np.asarray(up, dtype=np.float64)
    down_matrix = np.asarray(down, dtype=np.float64)
    order = np.asarray(permutation, dtype=np.int64)
    intermediate = gate_matrix.shape[0]
    if gate_matrix.shape != up_matrix.shape:
        raise ValueError("gate and up weights must have matching shapes")
    if hidden.ndim != 2 or hidden.shape[1] != gate_matrix.shape[1]:
        raise ValueError("states and projection widths do not match")
    if down_matrix.shape != (gate_matrix.shape[1], intermediate):
        raise ValueError("down projection has an incompatible shape")
    if order.shape != (intermediate,) or not np.array_equal(
        np.sort(order), np.arange(intermediate)
    ):
        raise ValueError("permutation must contain every intermediate record once")
    if intermediate % experts:
        raise ValueError("experts must divide the intermediate size")
    block = intermediate // experts
    activations = neuron_activations(hidden, gate_matrix[order], up_matrix[order])
    activation_blocks = activations.reshape(len(hidden), experts, block)
    down_blocks = down_matrix[:, order].reshape(down_matrix.shape[0], experts, block)
    return np.einsum("neb,heb->neh", activation_blocks, down_blocks, optimize=True)


def _top_experts(contributions: np.ndarray, active_experts: int) -> np.ndarray:
    scores = np.linalg.norm(contributions, axis=2)
    return np.argsort(-scores, axis=1, kind="stable")[:, :active_experts]


def _greedy_residual_experts(
    contributions: np.ndarray, active_experts: int
) -> np.ndarray:
    """Greedily choose blocks that most reduce dense-output squared error."""

    rows, experts, _ = contributions.shape
    residual = contributions.sum(axis=1).copy()
    contribution_norm = np.sum(contributions**2, axis=2)
    selected = np.zeros((rows, experts), dtype=bool)
    result = np.empty((rows, active_experts), dtype=np.int64)
    for step in range(active_experts):
        reduction = 2.0 * np.einsum(
            "nh,neh->ne", residual, contributions, optimize=True
        ) - contribution_norm
        reduction[selected] = -np.inf
        choice = np.argmax(reduction, axis=1)
        result[:, step] = choice
        selected[np.arange(rows), choice] = True
        residual -= contributions[np.arange(rows), choice]
    return result


def _membership(indices: np.ndarray, experts: int) -> np.ndarray:
    result = np.zeros((len(indices), experts), dtype=np.float64)
    result[np.arange(len(indices))[:, None], indices] = 1.0
    return result


def _selected_sum(contributions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return contributions[np.arange(len(contributions))[:, None], indices].sum(axis=1)


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError("cannot summarize empty or non-finite metrics")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _load_trace_field(
    reader: TraceReader, field: str, limit: int | None
) -> np.ndarray:
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
        raise ValueError(f"trace contains no values for {field}")
    return np.concatenate(batches)


def _stable_top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    return np.argsort(-scores, axis=1, kind="stable")[:, :count]


def _selected_channel_output(
    gate_values: np.ndarray,
    up_values: np.ndarray,
    down: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    activations = silu(gate_values) * up_values
    sparse = np.zeros_like(activations)
    sparse[np.arange(len(sparse))[:, None], selected] = activations[
        np.arange(len(sparse))[:, None], selected
    ]
    return sparse @ down.T


def evaluate_native_gate_channel_shadow(
    model: str | Path,
    validation_traces: str | Path,
    out: str | Path,
    *,
    input_fractions: Iterable[float] = (0.625, 1.0),
    top_k: int = 512,
    validation_records: int | None = 128,
) -> dict[str, Any]:
    """Screen gate-routed channels without candidate completion or a predictor."""

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    reader = TraceReader(validation_traces)
    if reader.manifest["model_hash"] != inspection.source_hash:
        raise ValueError("validation trace/model hash mismatch")
    if reader.manifest["split"] != "validation":
        raise ValueError("expected 'validation' validation traces")
    fractions = tuple(dict.fromkeys(float(value) for value in input_fractions))
    if not fractions or any(
        not np.isfinite(value) or not 0 < value <= 1 for value in fractions
    ):
        raise ValueError("input_fractions must contain values in (0, 1]")
    if not isinstance(top_k, int) or not 0 < top_k <= inspection.intermediate_size:
        raise ValueError("top_k must lie within the intermediate size")

    dense_relative: list[float] = []
    oracle_relative: list[float] = []
    dense_gate_relative: list[float] = []
    dense_gate_cosine: list[float] = []
    dense_gate_recall: list[float] = []
    partial_relative = {fraction: [] for fraction in fractions}
    partial_cosine = {fraction: [] for fraction in fractions}
    partial_recall = {fraction: [] for fraction in fractions}
    per_layer: list[dict[str, Any]] = []
    for layer in range(inspection.num_hidden_layers):
        gate, up, down = (
            np.asarray(value, dtype=np.float64)
            for value in load_layer_mlp(model_path, layer)
        )
        states = _load_states(reader, layer, validation_records)
        targets = _load_trace_field(
            reader, f"layer_{layer}_mlp_output", validation_records
        )
        gate_values = states @ gate.T
        up_values = states @ up.T
        dense_output = (silu(gate_values) * up_values) @ down.T
        dense_error, _ = _relative_and_cosine_rows(dense_output, targets)
        dense_relative.extend(dense_error.tolist())
        value_norms = np.linalg.norm(down, axis=0)[None, :]

        exact_scores = np.abs(silu(gate_values) * up_values) * value_norms
        oracle = _stable_top_indices(exact_scores, top_k)
        oracle_output = _selected_channel_output(
            gate_values, up_values, down, oracle
        )
        oracle_error, _ = _relative_and_cosine_rows(oracle_output, targets)
        oracle_relative.extend(oracle_error.tolist())

        gate_scores = np.abs(silu(gate_values)) * value_norms
        dense_gate = _stable_top_indices(gate_scores, top_k)
        dense_gate_output = _selected_channel_output(
            gate_values, up_values, down, dense_gate
        )
        gate_error, gate_similarity = _relative_and_cosine_rows(
            dense_gate_output, targets
        )
        dense_gate_relative.extend(gate_error.tolist())
        dense_gate_cosine.extend(gate_similarity.tolist())
        oracle_membership = _membership(oracle, inspection.intermediate_size).astype(bool)
        dense_gate_membership = _membership(
            dense_gate, inspection.intermediate_size
        ).astype(bool)
        gate_recall = np.sum(
            oracle_membership & dense_gate_membership, axis=1
        ) / top_k
        dense_gate_recall.extend(gate_recall.tolist())

        layer_result: dict[str, Any] = {
            "layer": layer,
            "dense_gate_relative_l2": float(np.mean(gate_error)),
            "dense_gate_oracle_recall": float(np.mean(gate_recall)),
            "input_sparse": {},
        }
        coordinate_order = np.argsort(-np.abs(states), axis=1, kind="stable")
        for fraction in fractions:
            traffic = native_gate_channel_traffic(
                inspection.hidden_size,
                inspection.intermediate_size,
                input_fraction=fraction,
                active_records=top_k,
            )
            coordinates = coordinate_order[:, : traffic.input_coordinates]
            partial_states = np.zeros_like(states)
            partial_states[
                np.arange(len(states))[:, None], coordinates
            ] = states[np.arange(len(states))[:, None], coordinates]
            partial_gate_values = partial_states @ gate.T
            partial_scores = np.abs(silu(partial_gate_values)) * value_norms
            selected = _stable_top_indices(partial_scores, top_k)
            output = _selected_channel_output(
                partial_gate_values, up_values, down, selected
            )
            error, similarity = _relative_and_cosine_rows(output, targets)
            selected_membership = _membership(
                selected, inspection.intermediate_size
            ).astype(bool)
            recall = np.sum(
                oracle_membership & selected_membership, axis=1
            ) / top_k
            partial_relative[fraction].extend(error.tolist())
            partial_cosine[fraction].extend(similarity.tolist())
            partial_recall[fraction].extend(recall.tolist())
            layer_result["input_sparse"][str(fraction)] = {
                "input_coordinates": traffic.input_coordinates,
                "relative_l2": float(np.mean(error)),
                "oracle_recall": float(np.mean(recall)),
            }
        per_layer.append(layer_result)

    configurations = []
    for fraction in fractions:
        traffic = native_gate_channel_traffic(
            inspection.hidden_size,
            inspection.intermediate_size,
            input_fraction=fraction,
            active_records=top_k,
        )
        metrics = {
            "local_relative_l2": _stats(partial_relative[fraction]),
            "local_cosine": _stats(partial_cosine[fraction]),
            "exact_contribution_oracle_recall": _stats(partial_recall[fraction]),
        }
        checks = {
            "active_record_budget": top_k <= 512,
            "input_coordinate_budget": fraction <= 0.625,
            "projected_traffic": traffic.fraction_of_dense <= 0.45,
            "local_pretraining_screen": metrics["local_relative_l2"]["mean"] <= 0.20,
        }
        configurations.append(
            {
                "input_fraction": fraction,
                "metrics": metrics,
                "projected_traffic": traffic.to_dict(),
                "screen": {"passed": all(checks.values()), "checks": checks},
            }
        )
    baseline = {
        "dense_shadow_relative_l2": _stats(dense_relative),
        "exact_contribution_oracle_relative_l2": _stats(oracle_relative),
        "dense_gate_channel_relative_l2": _stats(dense_gate_relative),
        "dense_gate_channel_cosine": _stats(dense_gate_cosine),
        "dense_gate_exact_contribution_oracle_recall": _stats(dense_gate_recall),
    }
    viable = [item for item in configurations if item["screen"]["passed"]]
    report = {
        "schema_version": 1,
        "experiment": "native_gate_channel_shadow",
        "status": "trace_only_feasibility_screen",
        "source_model_hash": inspection.source_hash,
        "configuration": {
            "top_k": top_k,
            "input_fractions": list(fractions),
            "validation_records_limit": validation_records,
            "selection_score": "absolute_silu_gate_times_down_column_norm",
            "completion": "none",
        },
        "baseline": baseline,
        "configurations": configurations,
        "screen": {
            "passed": bool(viable),
            "decision": (
                "justify_native_gate_sparse_training"
                if viable
                else "native_gate_requires_coadaptation_before_gate_claim"
            ),
            "caveat": (
                "The exact up projection is evaluated densely only to vectorize this shadow "
                "measurement; projected inference traffic counts selected up/down rows only. "
                "This is not an all-layer causal intervention."
            ),
        },
        "layers": per_layer,
    }
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    atomic_json(target / "native_gate_channel_shadow.json", report)
    lines = [
        "# Native-gate channel shadow screen",
        "",
        f"Decision: **{report['screen']['decision']}**",
        "",
        "| Gate inputs | Local rel-L2 | Oracle recall | Traffic |",
        "|---:|---:|---:|---:|",
    ]
    for item in configurations:
        lines.append(
            f"| {item['input_fraction']:.3f} | "
            f"{item['metrics']['local_relative_l2']['mean']:.6f} | "
            f"{item['metrics']['exact_contribution_oracle_recall']['mean']:.6f} | "
            f"{item['projected_traffic']['fraction_of_dense']:.6f}× |"
        )
    lines.extend(
        [
            "",
            f"Dense-gate channel selection local relative L2 is "
            f"{baseline['dense_gate_channel_relative_l2']['mean']:.6f}; the exact contribution "
            f"top-{top_k} reference is {baseline['exact_contribution_oracle_relative_l2']['mean']:.6f}.",
            "",
            "This is a local trace-only screen and does not authorize serialization.",
            "",
        ]
    )
    (target / "native_gate_channel_shadow.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


def evaluate_native_gate_residual_shadow(
    model: str | Path,
    calibration_traces: str | Path,
    validation_traces: str | Path,
    out: str | Path,
    *,
    ranks: Iterable[int] = (8, 16),
    blends: Iterable[float] = (0.25, 0.5, 1.0),
    input_fraction: float = 0.625,
    top_k: int = 512,
    regularization: float = 1000.0,
    calibration_records: int | None = 128,
    validation_records: int | None = 128,
    active_record_limit: int = 512,
) -> dict[str, Any]:
    """Fit a cheap residual predictor for missing native-gate channel utility."""

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    calibration = TraceReader(calibration_traces)
    validation = TraceReader(validation_traces)
    for name, reader, split in (
        ("calibration", calibration, "calibration"),
        ("validation", validation, "validation"),
    ):
        if reader.manifest["model_hash"] != inspection.source_hash:
            raise ValueError(f"{name} trace/model hash mismatch")
        if reader.manifest["split"] != split:
            raise ValueError(f"expected {split!r} {name} traces")
    if set(_sequence_hashes(calibration)).intersection(
        _sequence_hashes(validation)
    ):
        raise ValueError("calibration and validation traces contain matching token sequences")
    rank_values = tuple(dict.fromkeys(int(value) for value in ranks))
    blend_values = tuple(dict.fromkeys(float(value) for value in blends))
    maximum_rank = min(
        calibration_records or 2**31,
        inspection.hidden_size,
        inspection.intermediate_size,
    )
    if not rank_values or any(value <= 0 or value > maximum_rank for value in rank_values):
        raise ValueError("ranks must fit within calibration and model dimensions")
    if not blend_values or any(
        not np.isfinite(value) or value < 0 for value in blend_values
    ):
        raise ValueError("blends must be finite and nonnegative")
    if not isinstance(active_record_limit, int) or active_record_limit <= 0:
        raise ValueError("active_record_limit must be a positive integer")
    base_traffic = native_gate_channel_traffic(
        inspection.hidden_size,
        inspection.intermediate_size,
        input_fraction=input_fraction,
        active_records=top_k,
    )
    accumulators = {
        (rank, blend): {"relative": [], "cosine": [], "recall": []}
        for rank in rank_values
        for blend in blend_values
    }
    base_relative: list[float] = []
    oracle_relative: list[float] = []
    layer_results = []
    fitted_predictors: dict[tuple[int, int], LowRankUtilityResidual] = {}
    for layer in range(inspection.num_hidden_layers):
        gate, up, down = (
            np.asarray(value, dtype=np.float64)
            for value in load_layer_mlp(model_path, layer)
        )
        fit_states = _load_states(calibration, layer, calibration_records)
        held_states = _load_states(validation, layer, validation_records)
        held_targets = _load_trace_field(
            validation, f"layer_{layer}_mlp_output", validation_records
        )
        value_norms = np.linalg.norm(down, axis=0)[None, :]

        def gate_inputs(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            coordinate_count = base_traffic.input_coordinates
            coordinates = np.argsort(
                -np.abs(states), axis=1, kind="stable"
            )[:, :coordinate_count]
            partial_states = np.zeros_like(states)
            partial_states[
                np.arange(len(states))[:, None], coordinates
            ] = states[np.arange(len(states))[:, None], coordinates]
            partial_gate = partial_states @ gate.T
            full_gate = states @ gate.T
            up_values = states @ up.T
            return partial_gate, full_gate, up_values

        fit_partial, fit_full, fit_up = gate_inputs(fit_states)
        fit_base_logits = np.log(
            np.abs(silu(fit_partial)) * value_norms + 1e-8
        )
        fit_exact_logits = np.log(
            np.abs(silu(fit_full) * fit_up) * value_norms + 1e-8
        )
        residual_targets = np.clip(
            fit_exact_logits - fit_base_logits, -8.0, 8.0
        )
        residual_targets -= np.mean(
            residual_targets, axis=1, keepdims=True
        )

        held_partial, held_full, held_up = gate_inputs(held_states)
        held_base_logits = np.log(
            np.abs(silu(held_partial)) * value_norms + 1e-8
        )
        exact_scores = np.abs(silu(held_full) * held_up) * value_norms
        oracle = _stable_top_indices(exact_scores, top_k)
        oracle_membership = _membership(
            oracle, inspection.intermediate_size
        ).astype(bool)
        oracle_output = _selected_channel_output(
            held_full, held_up, down, oracle
        )
        oracle_error, _ = _relative_and_cosine_rows(
            oracle_output, held_targets
        )
        oracle_relative.extend(oracle_error.tolist())
        base_selected = _stable_top_indices(held_base_logits, top_k)
        base_output = _selected_channel_output(
            held_partial, held_up, down, base_selected
        )
        base_error, _ = _relative_and_cosine_rows(base_output, held_targets)
        base_relative.extend(base_error.tolist())
        layer_entry = {"layer": layer, "configurations": []}
        for rank in rank_values:
            predictor = fit_low_rank_utility_residual(
                fit_states,
                residual_targets,
                rank=rank,
                regularization=regularization,
            )
            fitted_predictors[(layer, rank)] = predictor
            predicted = predictor.predict(held_states)
            for blend in blend_values:
                selected = _stable_top_indices(
                    held_base_logits + blend * predicted, top_k
                )
                output = _selected_channel_output(
                    held_partial, held_up, down, selected
                )
                error, cosine = _relative_and_cosine_rows(
                    output, held_targets
                )
                selected_membership = _membership(
                    selected, inspection.intermediate_size
                ).astype(bool)
                recall = np.sum(
                    selected_membership & oracle_membership, axis=1
                ) / top_k
                accumulator = accumulators[(rank, blend)]
                accumulator["relative"].extend(error.tolist())
                accumulator["cosine"].extend(cosine.tolist())
                accumulator["recall"].extend(recall.tolist())
                layer_entry["configurations"].append(
                    {
                        "rank": rank,
                        "blend": blend,
                        "local_relative_l2": float(np.mean(error)),
                        "oracle_recall": float(np.mean(recall)),
                    }
                )
        layer_results.append(layer_entry)

    base_stats = _stats(base_relative)
    configurations = []
    for (rank, blend), values in accumulators.items():
        predictor_bytes = (
            inspection.hidden_size * rank
            + rank * inspection.intermediate_size
            + inspection.intermediate_size
        ) * 4
        total_bytes = base_traffic.total_weight_bytes + predictor_bytes
        fraction = total_bytes / base_traffic.dense_weight_bytes
        metrics = {
            "local_relative_l2": _stats(values["relative"]),
            "local_cosine": _stats(values["cosine"]),
            "exact_contribution_oracle_recall": _stats(values["recall"]),
        }
        checks = {
            "material_local_improvement": metrics["local_relative_l2"]["mean"]
            <= 0.9 * base_stats["mean"],
            "projected_traffic": fraction <= 0.45,
            "active_record_budget": top_k <= active_record_limit,
            "input_coordinate_budget": input_fraction <= 0.625,
        }
        configurations.append(
            {
                "rank": rank,
                "blend": blend,
                "metrics": metrics,
                "projected_traffic": {
                    "base_native_gate_bytes": base_traffic.total_weight_bytes,
                    "predictor_bytes": predictor_bytes,
                    "total_weight_bytes": total_bytes,
                    "dense_weight_bytes": base_traffic.dense_weight_bytes,
                    "fraction_of_dense": fraction,
                },
                "screen": {"passed": all(checks.values()), "checks": checks},
            }
        )
    configurations.sort(
        key=lambda item: (
            item["metrics"]["local_relative_l2"]["mean"],
            item["projected_traffic"]["fraction_of_dense"],
        )
    )
    passing = [item for item in configurations if item["screen"]["passed"]]
    deployment_selection = None
    if passing:
        best_error = passing[0]["metrics"]["local_relative_l2"]["mean"]
        near_best = [
            item
            for item in passing
            if item["metrics"]["local_relative_l2"]["mean"] <= 1.01 * best_error
        ]
        deployment_selection = min(
            near_best,
            key=lambda item: (
                item["projected_traffic"]["fraction_of_dense"],
                item["metrics"]["local_relative_l2"]["mean"],
            ),
        )
    artifact = {
        "written": False,
        "selection_rule": "lowest_traffic_passing_configuration_within_one_percent_of_best_error",
    }
    if deployment_selection is not None:
        artifact.update(
            {
                "path": "native_gate_utility_residual.safetensors",
                "rank": deployment_selection["rank"],
                "blend": deployment_selection["blend"],
                "tensor_dtype": "float32",
            }
        )
    report = {
        "schema_version": 1,
        "experiment": "native_gate_low_rank_utility_residual_shadow",
        "status": "trace_only_feasibility_screen",
        "source_model_hash": inspection.source_hash,
        "configuration": {
            "ranks": list(rank_values),
            "blends": list(blend_values),
            "input_fraction": input_fraction,
            "top_k": top_k,
            "regularization": regularization,
            "calibration_records_limit": calibration_records,
            "validation_records_limit": validation_records,
            "active_record_limit": active_record_limit,
            "target": "centered_clipped_log_exact_utility_minus_partial_gate_utility",
        },
        "data_separation": {
            "method": "exact_token_sequence_hashes",
            "overlapping_sequences": 0,
            "held_out": True,
        },
        "baseline": {
            "native_gate_local_relative_l2": base_stats,
            "exact_contribution_oracle_relative_l2": _stats(oracle_relative),
            "projected_traffic": base_traffic.to_dict(),
        },
        "configurations": configurations,
        "deployment_selection": deployment_selection,
        "artifact": artifact,
        "screen": {
            "passed": bool(passing),
            "decision": (
                "integrate_low_rank_utility_residual"
                if passing
                else "reject_low_rank_utility_residual"
            ),
            "criterion": "at_least_10_percent_local_error_reduction_at_no_more_than_45_percent_dense_traffic",
        },
        "layers": layer_results,
    }
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    if deployment_selection is not None:
        from safetensors.numpy import save_file

        selected_rank = deployment_selection["rank"]
        tensors = {}
        for layer in range(inspection.num_hidden_layers):
            predictor = fitted_predictors[(layer, selected_rank)]
            prefix = f"layers.{layer}.utility_residual"
            tensors[f"{prefix}.input_factors"] = predictor.input_factors.astype(
                np.float32
            )
            tensors[f"{prefix}.output_factors"] = predictor.output_factors.astype(
                np.float32
            )
            tensors[f"{prefix}.bias"] = predictor.bias.astype(np.float32)
        artifact_path = target / artifact["path"]
        save_file(
            tensors,
            artifact_path,
            metadata={
                "source_model_hash": inspection.source_hash,
                "rank": str(selected_rank),
                "blend": str(deployment_selection["blend"]),
                "format": "engram_native_gate_utility_residual_v1",
            },
        )
        artifact["written"] = True
        artifact["sha256"] = sha256_file(artifact_path)
        artifact["bytes"] = artifact_path.stat().st_size
    atomic_json(target / "native_gate_utility_residual.json", report)
    lines = [
        "# Native-gate low-rank utility residual",
        "",
        f"Decision: **{report['screen']['decision']}**",
        "",
        f"Baseline local relative L2: {base_stats['mean']:.6f}.",
        "",
        "| Rank | Blend | Local rel-L2 | Oracle recall | Traffic | Pass |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for item in configurations:
        lines.append(
            f"| {item['rank']} | {item['blend']:.2f} | "
            f"{item['metrics']['local_relative_l2']['mean']:.6f} | "
            f"{item['metrics']['exact_contribution_oracle_recall']['mean']:.6f} | "
            f"{item['projected_traffic']['fraction_of_dense']:.6f}× | "
            f"{'yes' if item['screen']['passed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "This cached trace screen is not an all-layer causal intervention.",
            (
                f"Deployment selection: rank {deployment_selection['rank']}, blend "
                f"{deployment_selection['blend']:.2f}."
                if deployment_selection is not None
                else "No deployment artifact was written."
            ),
            "",
        ]
    )
    (target / "native_gate_utility_residual.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


def train_native_gate_trace_student(
    model: str | Path,
    calibration_traces: str | Path,
    validation_traces: str | Path,
    out: str | Path,
    *,
    layers: Iterable[int] | None = None,
    top_k: int = 512,
    input_fraction: float = 0.625,
    steps: int = 16,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    dense_shadow_weight: float = 0.25,
    utility_weight: float = 0.25,
    temperature: float = 1.0,
    calibration_records: int | None = 128,
    validation_records: int | None = 128,
    device: str = "cpu",
) -> dict[str, Any]:
    """Pretrain native-gate sparse MLP layers against cached teacher boundaries."""

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] to train native-gate layers"
        ) from exc

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    calibration = TraceReader(calibration_traces)
    validation = TraceReader(validation_traces)
    for name, reader, split in (
        ("calibration", calibration, "calibration"),
        ("validation", validation, "validation"),
    ):
        if reader.manifest["model_hash"] != inspection.source_hash:
            raise ValueError(f"{name} trace/model hash mismatch")
        if reader.manifest["split"] != split:
            raise ValueError(f"expected {split!r} {name} traces")
    overlap = set(_sequence_hashes(calibration)).intersection(
        _sequence_hashes(validation)
    )
    if overlap:
        raise ValueError("calibration and validation traces contain matching token sequences")
    selected_layers = tuple(
        range(inspection.num_hidden_layers)
        if layers is None
        else dict.fromkeys(int(value) for value in layers)
    )
    if not selected_layers or any(
        value < 0 or value >= inspection.num_hidden_layers
        for value in selected_layers
    ):
        raise ValueError("layers must contain valid model layer indices")
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(dense_shadow_weight) or dense_shadow_weight < 0:
        raise ValueError("dense_shadow_weight must be finite and nonnegative")
    if not np.isfinite(utility_weight) or utility_weight < 0:
        raise ValueError("utility_weight must be finite and nonnegative")
    traffic = native_gate_channel_traffic(
        inspection.hidden_size,
        inspection.intermediate_size,
        input_fraction=input_fraction,
        active_records=top_k,
    )

    class TraceMLP(torch.nn.Module):
        def __init__(self, gate: np.ndarray, up: np.ndarray, down: np.ndarray):
            super().__init__()
            self.gate_proj = torch.nn.Linear(
                inspection.hidden_size,
                inspection.intermediate_size,
                bias=False,
            )
            self.up_proj = torch.nn.Linear(
                inspection.hidden_size,
                inspection.intermediate_size,
                bias=False,
            )
            self.down_proj = torch.nn.Linear(
                inspection.intermediate_size,
                inspection.hidden_size,
                bias=False,
            )
            with torch.no_grad():
                self.gate_proj.weight.copy_(torch.from_numpy(gate).float())
                self.up_proj.weight.copy_(torch.from_numpy(up).float())
                self.down_proj.weight.copy_(torch.from_numpy(down).float())
            self.act_fn = torch.nn.functional.silu

    wrapper_type = _wrap_native_gate_channel_mlp_class(torch)
    tensors: dict[str, Any] = {}
    layer_reports: list[dict[str, Any]] = []
    for layer in selected_layers:
        gate, up, down = load_layer_mlp(model_path, layer)
        train_states = torch.from_numpy(
            _load_states(calibration, layer, calibration_records).astype(np.float32)
        ).to(device)
        train_targets = torch.from_numpy(
            _load_trace_field(
                calibration, f"layer_{layer}_mlp_output", calibration_records
            ).astype(np.float32)
        ).to(device)
        held_out_states = torch.from_numpy(
            _load_states(validation, layer, validation_records).astype(np.float32)
        ).to(device)
        held_out_targets = torch.from_numpy(
            _load_trace_field(
                validation, f"layer_{layer}_mlp_output", validation_records
            ).astype(np.float32)
        ).to(device)
        torch.manual_seed(4703 + layer)
        wrapper = wrapper_type(
            TraceMLP(gate, up, down),
            top_k=top_k,
            input_fraction=input_fraction,
            temperature=temperature,
        ).to(device)
        with torch.no_grad():
            teacher_gate = torch.nn.functional.linear(
                train_states, wrapper.gate_weight
            )
            teacher_up = torch.nn.functional.linear(
                train_states, wrapper.up_weight
            )
            teacher_norms = torch.linalg.vector_norm(
                wrapper.down_weight, dim=0
            )
            teacher_scores = torch.abs(
                wrapper.act_fn(teacher_gate) * teacher_up
            ) * teacher_norms.unsqueeze(0)
            train_oracle = torch.argsort(
                teacher_scores, dim=1, descending=True, stable=True
            )[:, :top_k]

        def evaluate_mode(mode: str) -> tuple[np.ndarray, np.ndarray]:
            wrapper.eval()
            wrapper.mode = mode
            with torch.inference_mode():
                output = wrapper(held_out_states)
            return _relative_and_cosine_rows(
                output.detach().cpu().numpy(),
                held_out_targets.detach().cpu().numpy(),
            )

        before_hard, before_hard_cosine = evaluate_mode("hard")
        before_dense, _ = evaluate_mode("dense_shadow")
        optimizer = torch.optim.AdamW(
            wrapper.parameters(), lr=learning_rate, weight_decay=0.0
        )
        history = []
        for step in range(steps):
            start = (step * batch_size) % len(train_states)
            indices = torch.arange(start, start + batch_size, device=device)
            indices = torch.remainder(indices, len(train_states))
            states = train_states[indices]
            targets = train_targets[indices]
            optimizer.zero_grad(set_to_none=True)
            wrapper.train()
            wrapper.mode = "hard"
            sparse_output = wrapper(states)
            selection_logits = wrapper.last_selection_logits
            sparse_loss = torch.mean((sparse_output - targets) ** 2) / torch.clamp(
                torch.mean(targets**2), min=1e-8
            )
            if utility_weight and top_k < inspection.intermediate_size:
                negative_count = min(
                    top_k, inspection.intermediate_size - top_k
                )
                positives = selection_logits.gather(
                    1, train_oracle[indices]
                )[:, :negative_count]
                positive_mask = torch.zeros_like(
                    selection_logits, dtype=torch.bool
                ).scatter(1, train_oracle[indices], True)
                negatives = torch.topk(
                    selection_logits.masked_fill(positive_mask, -torch.inf),
                    negative_count,
                    dim=1,
                    sorted=False,
                ).values
                utility_loss = torch.nn.functional.softplus(
                    negatives - positives
                ).mean()
            else:
                utility_loss = torch.zeros((), device=device)
            wrapper.mode = "dense_shadow"
            dense_output = wrapper(states)
            dense_loss = torch.mean((dense_output - targets) ** 2) / torch.clamp(
                torch.mean(targets**2), min=1e-8
            )
            loss = (
                sparse_loss
                + dense_shadow_weight * dense_loss
                + utility_weight * utility_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
            optimizer.step()
            history.append(
                {
                    "step": step + 1,
                    "total": float(loss.detach()),
                    "sparse": float(sparse_loss.detach()),
                    "dense_shadow": float(dense_loss.detach()),
                    "utility_ranking": float(utility_loss.detach()),
                }
            )
        after_hard, after_hard_cosine = evaluate_mode("hard")
        after_dense, _ = evaluate_mode("dense_shadow")
        improvement = float(
            (np.mean(before_hard) - np.mean(after_hard))
            / max(float(np.mean(before_hard)), 1e-12)
        )
        layer_reports.append(
            {
                "layer": layer,
                "training_states": len(train_states),
                "validation_states": len(held_out_states),
                "before": {
                    "hard_relative_l2": _stats(before_hard),
                    "hard_cosine": _stats(before_hard_cosine),
                    "dense_shadow_relative_l2": _stats(before_dense),
                },
                "after": {
                    "hard_relative_l2": _stats(after_hard),
                    "hard_cosine": _stats(after_hard_cosine),
                    "dense_shadow_relative_l2": _stats(after_dense),
                },
                "relative_improvement": improvement,
                "history": history,
            }
        )
        for name in ("gate_weight", "up_weight", "down_weight"):
            tensors[f"layer_{layer}.{name}"] = (
                getattr(wrapper, name).detach().cpu().contiguous()
            )

    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    tensor_path = target / "native_gate_trace_student.safetensors"
    save_file(tensors, tensor_path)
    mean_before = float(
        np.mean(
            [item["before"]["hard_relative_l2"]["mean"] for item in layer_reports]
        )
    )
    mean_after = float(
        np.mean(
            [item["after"]["hard_relative_l2"]["mean"] for item in layer_reports]
        )
    )
    mean_dense_after = float(
        np.mean(
            [
                item["after"]["dense_shadow_relative_l2"]["mean"]
                for item in layer_reports
            ]
        )
    )
    checks = {
        "held_out_sparse_improvement": mean_after <= 0.9 * mean_before,
        "dense_shadow_retention": mean_dense_after <= 0.05,
        "active_record_budget": top_k <= 512,
        "input_coordinate_budget": input_fraction <= 0.625,
        "projected_traffic": traffic.fraction_of_dense <= 0.45,
    }
    report = {
        "schema_version": 1,
        "experiment": "native_gate_cached_trace_pretraining",
        "status": "layerwise_trace_only_training",
        "source_model_hash": inspection.source_hash,
        "configuration": {
            "layers": list(selected_layers),
            "top_k": top_k,
            "input_fraction": input_fraction,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dense_shadow_weight": dense_shadow_weight,
            "utility_weight": utility_weight,
            "temperature": temperature,
            "calibration_records_limit": calibration_records,
            "validation_records_limit": validation_records,
            "device": device,
        },
        "data_separation": {
            "method": "exact_token_sequence_hashes",
            "overlapping_sequences": 0,
            "held_out": True,
        },
        "summary": {
            "hard_relative_l2_before": mean_before,
            "hard_relative_l2_after": mean_after,
            "dense_shadow_relative_l2_after": mean_dense_after,
        },
        "projected_traffic": traffic.to_dict(),
        "screen": {
            "passed": all(checks.values()),
            "checks": checks,
            "decision": (
                "justify_broader_native_gate_pretraining"
                if all(checks.values())
                else "revise_native_gate_trace_objective"
            ),
            "caveat": (
                "Cached-boundary training does not expose the student to its own transformer "
                "state drift and cannot pass the all-layer causal gate."
            ),
        },
        "layers": layer_reports,
        "artifact": {
            "path": str(tensor_path.resolve()),
            "sha256": sha256_file(tensor_path),
            "format": "safetensors_full_gate_up_down_weights_selected_layers",
        },
    }
    atomic_json(target / "native_gate_trace_training.json", report)
    lines = [
        "# Native-gate cached-trace pretraining",
        "",
        f"Decision: **{report['screen']['decision']}**",
        "",
        f"Mean held-out hard-path relative L2 changed from {mean_before:.6f} to",
        f"{mean_after:.6f}; dense-shadow error after training is {mean_dense_after:.6f}.",
        "",
        "This is a layer-boundary screen, not an end-to-end intervention result.",
        "",
    ]
    (target / "native_gate_trace_training.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


def train_native_gate_end_to_end(
    model: str | Path,
    training_dataset: str | Path,
    validation_dataset: str | Path,
    out: str | Path,
    *,
    target_top_k: int = 512,
    target_input_fraction: float = 0.625,
    steps: int = 2,
    warmup_steps: int = 0,
    anneal_steps: int = 1,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    local_weight: float = 1.0,
    dense_shadow_weight: float = 0.25,
    hidden_weight: float = 0.25,
    logit_weight: float = 0.25,
    utility_weight: float = 0.1,
    temperature: float = 1.0,
    max_train_records: int | None = None,
    max_validation_records: int | None = None,
    device: str = "cpu",
    save_artifact: bool = True,
    checkpoint_every: int = 0,
    resume: bool = False,
    utility_residual: str | Path | None = None,
) -> dict[str, Any]:
    """Progressively co-train all MLPs through the native-gate hard path."""

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for end-to-end native-gate training"
        ) from exc
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as transformers_imports

        if transformers_imports.is_sklearn_available():
            try:
                import sklearn  # noqa: F401
            except ImportError:
                # Generation imports sklearn only for optional candidate helpers.
                # A present but ABI-incompatible sklearn must not block CausalLM use.
                def sklearn_unavailable() -> bool:
                    return False

                transformers_imports.is_sklearn_available = sklearn_unavailable
                transformers_utils.is_sklearn_available = sklearn_unavailable
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "the local Transformers causal-LM stack could not be imported"
        ) from exc
    from engram.training.sparse_teacher import (
        _batch_ids,
        _batches,
        _load_jsonl,
        _masked_mean,
        _normalized_masked_mse,
    )

    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (warmup_steps, anneal_steps, checkpoint_every)
    ):
        raise ValueError(
            "warmup_steps, anneal_steps, and checkpoint_every must be nonnegative integers"
        )
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    scalar_values = (
        learning_rate,
        local_weight,
        dense_shadow_weight,
        hidden_weight,
        logit_weight,
        utility_weight,
        temperature,
    )
    if any(not np.isfinite(value) for value in scalar_values):
        raise ValueError("learning rates, weights, and temperature must be finite")
    if learning_rate <= 0 or temperature <= 0 or any(
        value < 0
        for value in (
            local_weight,
            dense_shadow_weight,
            hidden_weight,
            logit_weight,
            utility_weight,
        )
    ):
        raise ValueError("learning rate/temperature must be positive and loss weights nonnegative")

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    traffic = native_gate_channel_traffic(
        inspection.hidden_size,
        inspection.intermediate_size,
        input_fraction=target_input_fraction,
        active_records=target_top_k,
    )
    residual_path = Path(utility_residual) if utility_residual is not None else None
    residual_sha256 = sha256_file(residual_path) if residual_path is not None else None
    training_path = Path(training_dataset)
    validation_path = Path(validation_dataset)
    train_records = _load_jsonl(training_path, max_train_records)
    validation_records = _load_jsonl(validation_path, max_validation_records)
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)

    teacher = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
    ).to(device)
    student = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
    ).to(device)
    if any(
        "input_ids" not in record
        for record in (*train_records, *validation_records)
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
    else:
        class TokenIdOnlyTokenizer:
            pad_token_id = student.config.pad_token_id
            eos_token_id = student.config.eos_token_id

        tokenizer = TokenIdOnlyTokenizer()
    training_hashes = _evaluation_sequence_hashes(train_records, tokenizer)
    validation_hashes = _evaluation_sequence_hashes(validation_records, tokenizer)
    if set(training_hashes).intersection(validation_hashes):
        raise ValueError("training and validation contain matching token sequences")
    teacher.eval()
    student.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(False)

    wrapper_type = _wrap_native_gate_channel_mlp_class(torch)
    wrappers = []
    residual_rank = 0
    residual_blend = 0.0
    for layer, decoder in enumerate(student.model.layers):
        predictor = None
        if residual_path is not None:
            predictor, residual_blend = load_native_gate_utility_residual(
                residual_path,
                layer,
                expected_source_hash=inspection.source_hash,
            )
            if residual_rank and predictor.rank != residual_rank:
                raise ValueError("native-gate residual ranks differ across layers")
            residual_rank = predictor.rank
        wrapper = wrapper_type(
            decoder.mlp,
            top_k=inspection.intermediate_size,
            input_fraction=1.0,
            temperature=temperature,
            utility_residual=predictor,
            residual_blend=residual_blend,
        ).to(device)
        decoder.mlp = wrapper
        wrappers.append(wrapper)
    trainable_parameters = [
        parameter
        for wrapper in wrappers
        for parameter in wrapper.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=learning_rate, weight_decay=0.0
    )
    trainable_named_parameters = {
        f"layer_{layer}.{name}": parameter
        for layer, wrapper in enumerate(wrappers)
        for name, parameter in wrapper.named_parameters()
        if parameter.requires_grad
    }
    checkpoint_path = target / "native_gate_training_checkpoint.pt"
    checkpoint_manifest_path = target / "native_gate_training_checkpoint.json"
    checkpoint_configuration = {
        "schema_version": 1,
        "source_model_hash": inspection.source_hash,
        "training_dataset_hash": sha256_file(training_path),
        "validation_dataset_hash": sha256_file(validation_path),
        "training_records": len(train_records),
        "validation_records": len(validation_records),
        "batch_size": batch_size,
        "target_top_k": target_top_k,
        "utility_residual_sha256": residual_sha256,
        "target_input_fraction": target_input_fraction,
        "warmup_steps": warmup_steps,
        "anneal_steps": anneal_steps,
        "learning_rate": learning_rate,
        "temperature": temperature,
        "loss_weights": {
            "local": local_weight,
            "dense_shadow": dense_shadow_weight,
            "hidden": hidden_weight,
            "logit": logit_weight,
            "utility": utility_weight,
        },
    }
    completed_steps = 0
    history = []
    if resume:
        if not checkpoint_path.is_file() or not checkpoint_manifest_path.is_file():
            raise ValueError("resume requested but native-gate checkpoint is missing")
        checkpoint_manifest = json.loads(
            checkpoint_manifest_path.read_text(encoding="utf-8")
        )
        if checkpoint_manifest.get("configuration") != checkpoint_configuration:
            raise ValueError("native-gate checkpoint configuration mismatch")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        checkpoint_tensors = checkpoint.get("trainable_parameters", {})
        if set(checkpoint_tensors) != set(trainable_named_parameters):
            raise ValueError("native-gate checkpoint parameter set mismatch")
        with torch.no_grad():
            for name, parameter in trainable_named_parameters.items():
                parameter.copy_(checkpoint_tensors[name].to(device=device))
        optimizer.load_state_dict(checkpoint["optimizer"])
        completed_steps = int(checkpoint_manifest.get("completed_steps", 0))
        history = list(checkpoint_manifest.get("history", []))

    def save_training_checkpoint(completed: int) -> None:
        temporary = target / "native_gate_training_checkpoint.pt.tmp"
        torch.save(
            {
                "trainable_parameters": {
                    name: parameter.detach().cpu()
                    for name, parameter in trainable_named_parameters.items()
                },
                "optimizer": optimizer.state_dict(),
            },
            temporary,
        )
        temporary.replace(checkpoint_path)
        atomic_json(
            checkpoint_manifest_path,
            {
                "configuration": checkpoint_configuration,
                "completed_steps": completed,
                "history": history,
                "device_neutral": True,
            },
        )

    teacher_targets: dict[int, Any] = {}
    handles = [
        layer.mlp.register_forward_hook(
            lambda module, args, output, index=index: teacher_targets.__setitem__(
                index, output.detach()
            )
        )
        for index, layer in enumerate(teacher.model.layers)
    ]
    batches = list(_batches(train_records, batch_size))
    try:
        for step in range(completed_steps, steps):
            input_fraction, top_k = progressive_sparse_budget(
                inspection.intermediate_size,
                target_input_fraction=target_input_fraction,
                target_top_k=target_top_k,
                step=step,
                warmup_steps=warmup_steps,
                anneal_steps=anneal_steps,
            )
            for wrapper in wrappers:
                wrapper.set_budget(
                    top_k=top_k, input_fraction=input_fraction
                )
                wrapper.mode = "hard"
                wrapper.train()
            batch_records = batches[step % len(batches)]
            input_ids, attention_mask, lengths = _batch_ids(
                batch_records, tokenizer, torch, device
            )
            if max(lengths) < 2:
                raise ValueError("training records must contain at least two tokens")
            valid_mask = attention_mask.bool()
            valid_rows = valid_mask.reshape(-1)
            teacher_targets.clear()
            with torch.no_grad():
                teacher_output = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            optimizer.zero_grad(set_to_none=True)
            student_output = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            local_loss = torch.stack(
                [
                    _normalized_masked_mse(
                        wrapper.last_output,
                        teacher_targets[layer],
                        valid_mask,
                        torch,
                    )
                    for layer, wrapper in enumerate(wrappers)
                ]
            ).mean()
            dense_loss = torch.stack(
                [
                    _normalized_masked_mse(
                        wrapper.last_dense_output,
                        teacher_targets[layer],
                        valid_mask,
                        torch,
                    )
                    for layer, wrapper in enumerate(wrappers)
                ]
            ).mean()
            hidden_loss = torch.stack(
                [
                    _normalized_masked_mse(
                        student_hidden,
                        teacher_hidden,
                        valid_mask,
                        torch,
                    )
                    for student_hidden, teacher_hidden in zip(
                        student_output.hidden_states[1:],
                        teacher_output.hidden_states[1:],
                        strict=True,
                    )
                ]
            ).mean()
            teacher_logp = torch.nn.functional.log_softmax(
                teacher_output.logits.detach(), dim=-1
            )
            student_logp = torch.nn.functional.log_softmax(
                student_output.logits, dim=-1
            )
            logit_rows = torch.nn.functional.kl_div(
                student_logp,
                teacher_logp.exp(),
                reduction="none",
            ).sum(dim=-1)
            logit_loss = _masked_mean(logit_rows, valid_mask, torch)
            utility_rows = []
            if utility_weight and top_k < inspection.intermediate_size:
                negative_count = min(
                    top_k, inspection.intermediate_size - top_k
                )
                for wrapper in wrappers:
                    logits = wrapper.last_selection_logits[valid_rows]
                    oracle = wrapper.last_oracle[valid_rows]
                    positives = logits.gather(1, oracle)[:, :negative_count]
                    positive_mask = torch.zeros_like(
                        logits, dtype=torch.bool
                    ).scatter(1, oracle, True)
                    negatives = torch.topk(
                        logits.masked_fill(positive_mask, -torch.inf),
                        negative_count,
                        dim=1,
                        sorted=False,
                    ).values
                    utility_rows.append(
                        torch.nn.functional.softplus(
                            negatives - positives
                        ).mean()
                    )
            utility_loss = (
                torch.stack(utility_rows).mean()
                if utility_rows
                else torch.zeros((), device=input_ids.device)
            )
            loss = (
                local_weight * local_loss
                + dense_shadow_weight * dense_loss
                + hidden_weight * hidden_loss
                + logit_weight * logit_loss
                + utility_weight * utility_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
            history.append(
                {
                    "step": step + 1,
                    "input_fraction": input_fraction,
                    "input_coordinates": wrappers[0].input_coordinates,
                    "top_k": top_k,
                    "total": float(loss.detach()),
                    "local": float(local_loss.detach()),
                    "dense_shadow": float(dense_loss.detach()),
                    "hidden": float(hidden_loss.detach()),
                    "logit": float(logit_loss.detach()),
                    "utility_ranking": float(utility_loss.detach()),
                }
            )
            if checkpoint_every and (step + 1) % checkpoint_every == 0:
                save_training_checkpoint(step + 1)

        if checkpoint_every and len(history) != completed_steps:
            save_training_checkpoint(len(history))

        for wrapper in wrappers:
            wrapper.set_budget(
                top_k=target_top_k,
                input_fraction=target_input_fraction,
            )
            wrapper.mode = "hard"
            wrapper.eval()
        quality: dict[str, list[float]] = {}
        local_error: list[float] = []
        input_positions = 0
        next_positions = 0
        for batch_records in _batches(validation_records, batch_size):
            input_ids, attention_mask, lengths = _batch_ids(
                batch_records, tokenizer, torch, device
            )
            teacher_targets.clear()
            with torch.inference_mode():
                teacher_output = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                student_output = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            if any(wrapper.last_surrogate_used for wrapper in wrappers):
                raise RuntimeError("validation executed a training-only surrogate")
            for row, length in enumerate(lengths):
                if length < 2:
                    continue
                input_positions += length
                next_positions += length - 1
                metrics = _quality_metrics(
                    teacher_output.logits[row : row + 1, :length],
                    student_output.logits[row : row + 1, :length],
                    input_ids[row : row + 1, :length],
                    teacher_output.hidden_states[-1][row : row + 1, :length],
                    student_output.hidden_states[-1][row : row + 1, :length],
                    torch,
                )
                for name, values in metrics.items():
                    quality.setdefault(name, []).extend(
                        np.asarray(values).reshape(-1).tolist()
                    )
                for layer, wrapper in enumerate(wrappers):
                    relative, _ = _relative_and_cosine_rows(
                        wrapper.last_output[row : row + 1, :length]
                        .detach()
                        .cpu()
                        .numpy(),
                        teacher_targets[layer][row : row + 1, :length]
                        .detach()
                        .cpu()
                        .numpy(),
                    )
                    local_error.extend(relative.tolist())
    finally:
        for handle in handles:
            handle.remove()

    metric_means = {
        name: float(np.mean(values)) for name, values in quality.items()
    }
    residual_bytes = (
        inspection.hidden_size * residual_rank
        + residual_rank * inspection.intermediate_size
        + inspection.intermediate_size
    ) * 4
    projected_traffic_fraction = (
        traffic.total_weight_bytes + residual_bytes
    ) / traffic.dense_weight_bytes
    checks = {
        "teacher_student_kl": metric_means["teacher_student_kl"]
        <= MLP_QUALITY_THRESHOLDS["maximum_teacher_student_kl"],
        "teacher_top1_agreement": metric_means["teacher_top1_agreement"]
        >= MLP_QUALITY_THRESHOLDS["minimum_teacher_top1_agreement"],
        "nll_delta": metric_means["nll_delta"]
        <= MLP_QUALITY_THRESHOLDS["maximum_nll_delta"],
        "final_hidden_relative_l2": metric_means["final_hidden_relative_l2"]
        <= MLP_QUALITY_THRESHOLDS["maximum_final_hidden_relative_l2"],
        "evidence_size": (
            len(validation_records) >= MINIMUM_EVALUATION_SEQUENCES
            and len(set(validation_hashes)) >= MINIMUM_UNIQUE_EVALUATION_SEQUENCES
            and next_positions >= MINIMUM_NEXT_TOKEN_POSITIONS
        ),
        "active_record_budget": target_top_k <= 512,
        "input_coordinate_budget": target_input_fraction <= 0.625,
        "projected_traffic": projected_traffic_fraction <= 0.45,
    }
    artifact: dict[str, Any] = {
        "written": False,
        "format": "safetensors_full_mlp_weights_with_optional_utility_residual",
    }
    if save_artifact:
        tensors = {
            f"layer_{layer}.{name}": getattr(wrapper, name)
            .detach()
            .cpu()
            .contiguous()
            for layer, wrapper in enumerate(wrappers)
            for name in ("gate_weight", "up_weight", "down_weight")
        }
        if residual_rank:
            for layer, wrapper in enumerate(wrappers):
                prefix = f"layer_{layer}.utility_residual"
                tensors[f"{prefix}.input_factors"] = (
                    wrapper.residual_input_factors.detach().cpu().contiguous()
                )
                tensors[f"{prefix}.output_factors"] = (
                    wrapper.residual_output_factors.detach().cpu().contiguous()
                )
                tensors[f"{prefix}.bias"] = (
                    wrapper.residual_bias.detach().cpu().contiguous()
                )
        tensor_path = target / "native_gate_end_to_end.safetensors"
        save_file(tensors, tensor_path)
        artifact.update(
            written=True,
            path=str(tensor_path.resolve()),
            sha256=sha256_file(tensor_path),
        )
    report = {
        "schema_version": 1,
        "experiment": "progressive_end_to_end_native_gate_training",
        "status": "measured_local_model",
        "source_model_hash": inspection.source_hash,
        "configuration": {
            "target_top_k": target_top_k,
            "target_input_fraction": target_input_fraction,
            "steps": steps,
            "warmup_steps": warmup_steps,
            "anneal_steps": anneal_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "device": device,
            "device_neutral_semantics": True,
            "save_artifact": save_artifact,
            "checkpoint_every": checkpoint_every,
            "resumed": resume,
            "utility_residual": (
                {
                    "path": str(residual_path.resolve()),
                    "sha256": residual_sha256,
                    "rank": residual_rank,
                    "blend": residual_blend,
                }
                if residual_path is not None
                else None
            ),
            "loss_weights": {
                "local": local_weight,
                "dense_shadow": dense_shadow_weight,
                "hidden": hidden_weight,
                "logit": logit_weight,
                "utility": utility_weight,
            },
        },
        "training": {
            "records": len(train_records),
            "completed_before_resume": completed_steps,
            "requested_total_steps": steps,
            "history": history,
            "checkpoint": {
                "written": checkpoint_path.is_file(),
                "path": (
                    str(checkpoint_path.resolve())
                    if checkpoint_path.is_file()
                    else None
                ),
                "device_neutral": True,
            },
        },
        "validation": {
            "records": len(validation_records),
            "input_token_positions": input_positions,
            "next_token_positions": next_positions,
            "hard_sparse_path_only": True,
        },
        "data_separation": {
            "method": "exact_token_sequence_hashes",
            "training_sequences": len(training_hashes),
            "validation_sequences": len(validation_hashes),
            "overlapping_sequences": 0,
            "held_out": True,
            "training_dataset_hash": sha256_file(training_path),
            "validation_dataset_hash": sha256_file(validation_path),
        },
        "metrics": {
            **{name: _stats(values) for name, values in quality.items()},
            "local_mlp_relative_l2": _stats(local_error),
        },
        "projected_traffic": {
            **traffic.to_dict(),
            "utility_residual_bytes": residual_bytes,
            "with_utility_residual_fraction_of_dense": projected_traffic_fraction,
        },
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "decision": (
                "eligible_for_confirmation"
                if all(checks.values())
                else "continue_training_or_stop_before_serialization"
            ),
        },
        "artifact": artifact,
    }
    atomic_json(target / "native_gate_end_to_end.json", report)
    lines = [
        "# Progressive end-to-end native-gate training",
        "",
        f"Decision: **{report['gate']['decision']}**",
        "",
        f"Device: `{device}` (the forward semantics and artifact are device-neutral).",
        "",
        "| Metric | Mean | Threshold |",
        "|---|---:|---:|",
        f"| Teacher-student KL | {metric_means['teacher_student_kl']:.6f} | ≤0.05 |",
        f"| Teacher top-1 agreement | {metric_means['teacher_top1_agreement']:.6f} | ≥0.90 |",
        f"| NLL delta | {metric_means['nll_delta']:.6f} | ≤0.05 |",
        f"| Final hidden relative L2 | {metric_means['final_hidden_relative_l2']:.6f} | ≤0.10 |",
        f"| Local MLP relative L2 | {float(np.mean(local_error)):.6f} | diagnostic |",
        f"| Projected traffic | {projected_traffic_fraction:.6f}× | ≤0.45× |",
        "",
        "Validation executes only the target hard sparse path. A smoke run below the evidence",
        "floor validates mechanics but cannot establish model quality.",
        "",
    ]
    (target / "native_gate_end_to_end.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


def evaluate_structured_expert_shadow(
    model: str | Path,
    calibration_traces: str | Path,
    validation_traces: str | Path,
    out: str | Path,
    *,
    experts: int = 24,
    active_experts: int = 8,
    regularization: float = 1000.0,
    grouping_iterations: int = 12,
    calibration_records: int | None = 256,
    validation_records: int | None = 256,
) -> dict[str, Any]:
    """Run a trace-only block-oracle and fitted-router feasibility screen."""

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    traffic = structured_expert_traffic(
        inspection.hidden_size,
        inspection.intermediate_size,
        experts=experts,
        active_experts=active_experts,
    )
    calibration = TraceReader(calibration_traces)
    validation = TraceReader(validation_traces)
    for name, reader, expected_split in (
        ("calibration", calibration, "calibration"),
        ("validation", validation, "validation"),
    ):
        if reader.manifest["model_hash"] != inspection.source_hash:
            raise ValueError(f"{name} trace/model hash mismatch")
        if reader.manifest["split"] != expected_split:
            raise ValueError(f"expected {expected_split!r} {name} traces")
    overlap = set(_sequence_hashes(calibration)).intersection(_sequence_hashes(validation))
    if overlap:
        raise ValueError("calibration and validation traces contain matching token sequences")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")

    dense_relative: list[float] = []
    norm_oracle_relative: list[float] = []
    norm_oracle_cosine: list[float] = []
    greedy_oracle_relative: list[float] = []
    greedy_oracle_cosine: list[float] = []
    routed_relative: list[float] = []
    routed_cosine: list[float] = []
    recalls: list[float] = []
    layers: list[dict[str, Any]] = []
    for layer in range(inspection.num_hidden_layers):
        gate, up, down = (
            np.asarray(value, dtype=np.float64)
            for value in load_layer_mlp(model_path, layer)
        )
        fit_states = _load_states(calibration, layer, calibration_records)
        held_out_states = _load_states(validation, layer, validation_records)
        held_out_targets = _load_trace_field(
            validation, f"layer_{layer}_mlp_output", validation_records
        )
        fit_strength = np.abs(neuron_activations(fit_states, gate, up))
        fit_strength *= np.linalg.norm(down, axis=0)[None, :]
        permutation = balanced_expert_permutation(
            fit_strength.T, experts, iterations=grouping_iterations
        )
        fit_blocks = block_contributions(
            fit_states, gate, up, down, permutation, experts=experts
        )
        held_out_blocks = block_contributions(
            held_out_states, gate, up, down, permutation, experts=experts
        )
        fit_oracle = _greedy_residual_experts(fit_blocks, active_experts)
        held_out_oracle = _greedy_residual_experts(
            held_out_blocks, active_experts
        )
        held_out_norm_oracle = _top_experts(held_out_blocks, active_experts)
        router = MultiLabelLinearRouter.fit(
            fit_states,
            _membership(fit_oracle, experts),
            regularization=regularization,
        )
        router_scores = held_out_states @ router.weights + router.bias
        routed = np.argsort(-router_scores, axis=1, kind="stable")[:, :active_experts]
        dense_output = held_out_blocks.sum(axis=1)
        norm_oracle_output = _selected_sum(
            held_out_blocks, held_out_norm_oracle
        )
        greedy_oracle_output = _selected_sum(held_out_blocks, held_out_oracle)
        routed_output = _selected_sum(held_out_blocks, routed)
        dense_error, _ = _relative_and_cosine_rows(
            dense_output, held_out_targets
        )
        norm_oracle_error, norm_oracle_similarity = _relative_and_cosine_rows(
            norm_oracle_output, held_out_targets
        )
        greedy_oracle_error, greedy_oracle_similarity = _relative_and_cosine_rows(
            greedy_oracle_output, held_out_targets
        )
        routed_error, routed_similarity = _relative_and_cosine_rows(
            routed_output, held_out_targets
        )
        oracle_sets = _membership(held_out_oracle, experts).astype(bool)
        routed_sets = _membership(routed, experts).astype(bool)
        layer_recall = np.sum(oracle_sets & routed_sets, axis=1) / active_experts
        dense_relative.extend(dense_error.tolist())
        norm_oracle_relative.extend(norm_oracle_error.tolist())
        norm_oracle_cosine.extend(norm_oracle_similarity.tolist())
        greedy_oracle_relative.extend(greedy_oracle_error.tolist())
        greedy_oracle_cosine.extend(greedy_oracle_similarity.tolist())
        routed_relative.extend(routed_error.tolist())
        routed_cosine.extend(routed_similarity.tolist())
        recalls.extend(layer_recall.tolist())
        layers.append(
            {
                "layer": layer,
                "calibration_states": len(fit_states),
                "validation_states": len(held_out_states),
                "block_norm_oracle_local_relative_l2": float(
                    np.mean(norm_oracle_error)
                ),
                "greedy_residual_oracle_local_relative_l2": float(
                    np.mean(greedy_oracle_error)
                ),
                "routed_local_relative_l2": float(np.mean(routed_error)),
                "router_block_recall": float(np.mean(layer_recall)),
                "permutation": permutation.tolist(),
            }
        )

    metrics = {
        "dense_shadow_relative_l2": _stats(dense_relative),
        "full_information_block_norm_relative_l2": _stats(norm_oracle_relative),
        "full_information_block_norm_cosine": _stats(norm_oracle_cosine),
        "full_information_greedy_residual_relative_l2": _stats(
            greedy_oracle_relative
        ),
        "full_information_greedy_residual_cosine": _stats(
            greedy_oracle_cosine
        ),
        "routed_relative_l2": _stats(routed_relative),
        "routed_cosine": _stats(routed_cosine),
        "router_block_recall": _stats(recalls),
    }
    checks = {
        "dense_shadow_parity": metrics["dense_shadow_relative_l2"]["maximum"] <= 1e-5,
        "active_record_budget": traffic.active_records <= 512,
        "projected_traffic": traffic.fraction_of_dense <= 0.35,
        "oracle_local_screen": metrics[
            "full_information_greedy_residual_relative_l2"
        ]["mean"] <= 0.20,
    }
    report = {
        "schema_version": 1,
        "experiment": "structured_block_expert_shadow",
        "status": "trace_only_feasibility_screen",
        "source_model_hash": inspection.source_hash,
        "configuration": {
            "experts": experts,
            "active_experts": active_experts,
            "records_per_expert": traffic.records_per_expert,
            "active_records": traffic.active_records,
            "regularization": regularization,
            "grouping": "balanced_cosine_kmeans_on_absolute_contribution_profiles",
            "grouping_iterations": grouping_iterations,
            "calibration_records_limit": calibration_records,
            "validation_records_limit": validation_records,
        },
        "data_separation": {
            "method": "exact_token_sequence_hashes",
            "overlapping_sequences": 0,
            "held_out": True,
        },
        "metrics": metrics,
        "projected_traffic": traffic.to_dict(),
        "screen": {
            "passed": all(checks.values()),
            "checks": checks,
            "decision": (
                "justify_joint_block_training"
                if all(checks.values())
                else "diagnose_before_joint_block_training"
            ),
            "caveat": (
                "This is a local trace-only screen, not the all-layer causal intervention gate. "
                "Original-teacher neuron recall is intentionally not a progression criterion "
                "after the student basis becomes trainable."
            ),
        },
        "layers": layers,
    }
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    atomic_json(target / "structured_expert_shadow.json", report)
    lines = [
        "# Structured block-expert shadow screen",
        "",
        f"Decision: **{report['screen']['decision']}**",
        "",
        f"The layout has {experts} contiguous blocks of {traffic.records_per_expert} records and",
        f"executes {active_experts} blocks ({traffic.active_records} records) per token.",
        "",
        "| Metric | Mean |",
        "|---|---:|",
        f"| Block-norm reference local relative L2 | {metrics['full_information_block_norm_relative_l2']['mean']:.6f} |",
        f"| Greedy-residual reference local relative L2 | {metrics['full_information_greedy_residual_relative_l2']['mean']:.6f} |",
        f"| Fitted router local relative L2 | {metrics['routed_relative_l2']['mean']:.6f} |",
        f"| Fitted router block recall | {metrics['router_block_recall']['mean']:.6f} |",
        f"| Projected inference traffic | {traffic.fraction_of_dense:.6f}× dense |",
        "",
        "This screen uses exact dense block contributions to construct its reference. It does not",
        "pass the end-to-end intervention gate and does not authorize serialization.",
        "",
    ]
    (target / "structured_expert_shadow.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def _wrap_structured_expert_mlp_class(torch: Any):
    """Build a trainable hard-forward block-routed SwiGLU module."""

    class StructuredExpertMLP(torch.nn.Module):
        def __init__(
            self,
            base: Any,
            permutation: Any,
            *,
            experts: int,
            active_experts: int,
            temperature: float = 1.0,
        ):
            super().__init__()
            intermediate, hidden = base.gate_proj.weight.shape
            if intermediate % experts:
                raise ValueError("experts must divide the intermediate size")
            if not 0 < active_experts <= experts:
                raise ValueError("active_experts must lie within the expert count")
            if not np.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperature must be finite and positive")
            order = torch.as_tensor(permutation, dtype=torch.long)
            if order.shape != (intermediate,) or not torch.equal(
                torch.sort(order).values, torch.arange(intermediate)
            ):
                raise ValueError("permutation must contain every intermediate record once")
            if base.gate_proj.bias is not None or base.up_proj.bias is not None or base.down_proj.bias is not None:
                raise ValueError("bias-enabled MLP projections are not supported")
            block = intermediate // experts
            dtype = base.gate_proj.weight.dtype
            self.gate_blocks = torch.nn.Parameter(
                base.gate_proj.weight.detach()[order].reshape(experts, block, hidden).clone()
            )
            self.up_blocks = torch.nn.Parameter(
                base.up_proj.weight.detach()[order].reshape(experts, block, hidden).clone()
            )
            self.down_blocks = torch.nn.Parameter(
                base.down_proj.weight.detach().T[order].reshape(experts, block, hidden).clone()
            )
            self.router = torch.nn.Linear(hidden, experts, bias=True, dtype=dtype)
            torch.nn.init.zeros_(self.router.weight)
            torch.nn.init.zeros_(self.router.bias)
            self.act_fn = base.act_fn
            self.experts = experts
            self.active_experts = active_experts
            self.records_per_expert = block
            self.temperature = temperature
            self.mode = "hard"
            self.use_training_surrogate = True
            self.last_active_experts = None
            self.last_router_logits = None
            self.last_surrogate_used = False
            self.last_output = None

        def _dense_blocks(self, flat: Any) -> Any:
            gate = torch.einsum("nh,ebh->neb", flat, self.gate_blocks)
            up = torch.einsum("nh,ebh->neb", flat, self.up_blocks)
            activation = self.act_fn(gate) * up
            return torch.einsum("neb,ebh->neh", activation, self.down_blocks)

        def _hard_output(self, flat: Any, active: Any) -> Any:
            gate_weights = self.gate_blocks[active]
            up_weights = self.up_blocks[active]
            down_weights = self.down_blocks[active]
            gate = torch.einsum("nabh,nh->nab", gate_weights, flat)
            up = torch.einsum("nabh,nh->nab", up_weights, flat)
            activation = self.act_fn(gate) * up
            return torch.einsum("nab,nabh->nh", activation, down_weights)

        def forward(self, hidden: Any) -> Any:
            from engram.training.sparse_teacher import _cardinality_preserving_top_mask

            shape = hidden.shape
            flat = hidden.reshape(-1, shape[-1])
            if self.mode == "dense_shadow":
                output = self._dense_blocks(flat).sum(dim=1)
                self.last_active_experts = None
                self.last_router_logits = None
                self.last_surrogate_used = False
            elif self.mode == "hard":
                logits = self.router(flat)
                active = torch.argsort(
                    logits, dim=1, descending=True, stable=True
                )[:, : self.active_experts]
                output = self._hard_output(flat, active)
                self.last_active_experts = active
                self.last_router_logits = logits
                self.last_surrogate_used = False
                if self.training and self.use_training_surrogate:
                    _, soft_mask = _cardinality_preserving_top_mask(
                        logits, self.active_experts, self.temperature, torch
                    )
                    dense_blocks = self._dense_blocks(flat).detach()
                    proxy = torch.sum(soft_mask.unsqueeze(2) * dense_blocks, dim=1)
                    output = output + proxy - proxy.detach()
                    self.last_surrogate_used = True
            else:
                raise ValueError(f"unsupported structured expert mode {self.mode!r}")
            self.last_output = output.reshape(*shape[:-1], -1)
            return self.last_output

    return StructuredExpertMLP


def _wrap_native_gate_channel_mlp_class(torch: Any):
    """Build a co-trainable native-gate channel-sparse SwiGLU module."""

    class NativeGateChannelMLP(torch.nn.Module):
        def __init__(
            self,
            base: Any,
            *,
            top_k: int,
            input_fraction: float,
            temperature: float = 1.0,
            utility_residual: LowRankUtilityResidual | None = None,
            residual_blend: float = 0.8,
        ):
            super().__init__()
            intermediate, hidden = base.gate_proj.weight.shape
            if not 0 < top_k <= intermediate:
                raise ValueError("top_k must lie within the intermediate size")
            if not np.isfinite(input_fraction) or not 0 < input_fraction <= 1:
                raise ValueError("input_fraction must lie in (0, 1]")
            if not np.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperature must be finite and positive")
            if not np.isfinite(residual_blend) or residual_blend < 0:
                raise ValueError("residual_blend must be finite and nonnegative")
            if (
                base.gate_proj.bias is not None
                or base.up_proj.bias is not None
                or base.down_proj.bias is not None
            ):
                raise ValueError("bias-enabled MLP projections are not supported")
            self.gate_weight = torch.nn.Parameter(
                base.gate_proj.weight.detach().clone()
            )
            self.up_weight = torch.nn.Parameter(
                base.up_proj.weight.detach().clone()
            )
            self.down_weight = torch.nn.Parameter(
                base.down_proj.weight.detach().clone()
            )
            self.act_fn = base.act_fn
            self.top_k = top_k
            self.input_fraction = input_fraction
            self.input_coordinates = max(
                1, min(hidden, round(input_fraction * hidden))
            )
            self.temperature = temperature
            self.residual_blend = residual_blend
            if utility_residual is None:
                self.utility_residual_rank = 0
                self.register_buffer("residual_input_factors", None)
                self.register_buffer("residual_output_factors", None)
                self.register_buffer("residual_bias", None)
            else:
                if (
                    utility_residual.input_factors.shape[0] != hidden
                    or utility_residual.output_factors.shape
                    != (utility_residual.rank, intermediate)
                    or utility_residual.bias.shape != (intermediate,)
                ):
                    raise ValueError("utility_residual has incompatible dimensions")
                dtype = self.gate_weight.dtype
                device = self.gate_weight.device
                self.utility_residual_rank = utility_residual.rank
                self.register_buffer(
                    "residual_input_factors",
                    torch.as_tensor(
                        utility_residual.input_factors, dtype=dtype, device=device
                    ),
                )
                self.register_buffer(
                    "residual_output_factors",
                    torch.as_tensor(
                        utility_residual.output_factors, dtype=dtype, device=device
                    ),
                )
                self.register_buffer(
                    "residual_bias",
                    torch.as_tensor(
                        utility_residual.bias, dtype=dtype, device=device
                    ),
                )
            self.mode = "hard"
            self.use_training_surrogate = True
            self.last_active_records = None
            self.last_input_coordinates = None
            self.last_selection_logits = None
            self.last_surrogate_used = False
            self.last_output = None
            self.last_dense_output = None
            self.last_oracle = None

        def set_budget(self, *, top_k: int, input_fraction: float) -> None:
            intermediate, hidden = self.gate_weight.shape
            if not isinstance(top_k, int) or not 0 < top_k <= intermediate:
                raise ValueError("top_k must lie within the intermediate size")
            if not np.isfinite(input_fraction) or not 0 < input_fraction <= 1:
                raise ValueError("input_fraction must lie in (0, 1]")
            self.top_k = top_k
            self.input_fraction = input_fraction
            self.input_coordinates = max(
                1, min(hidden, round(input_fraction * hidden))
            )

        def _dense_output(self, flat: Any) -> Any:
            gate = torch.nn.functional.linear(flat, self.gate_weight)
            up = torch.nn.functional.linear(flat, self.up_weight)
            return torch.nn.functional.linear(
                self.act_fn(gate) * up, self.down_weight
            )

        def _partial_gate(self, flat: Any) -> tuple[Any, Any]:
            coordinates = torch.argsort(
                torch.abs(flat), dim=1, descending=True, stable=True
            )[:, : self.input_coordinates]
            partial = torch.zeros_like(flat).scatter(
                1, coordinates, flat.gather(1, coordinates)
            )
            return torch.nn.functional.linear(partial, self.gate_weight), coordinates

        def _hard_output(self, flat: Any, gate: Any, active: Any) -> Any:
            up_weights = self.up_weight[active]
            down_weights = self.down_weight.T[active]
            up = torch.einsum("nkh,nh->nk", up_weights, flat)
            activation = self.act_fn(gate.gather(1, active)) * up
            return torch.einsum("nk,nkh->nh", activation, down_weights)

        def forward(self, hidden: Any) -> Any:
            from engram.training.sparse_teacher import _cardinality_preserving_top_mask

            shape = hidden.shape
            flat = hidden.reshape(-1, shape[-1])
            if self.mode == "dense_shadow":
                output = self._dense_output(flat)
                self.last_active_records = None
                self.last_input_coordinates = None
                self.last_selection_logits = None
                self.last_surrogate_used = False
                self.last_dense_output = None
                self.last_oracle = None
            elif self.mode == "hard":
                partial_gate, coordinates = self._partial_gate(flat)
                value_norms = torch.linalg.vector_norm(
                    self.down_weight.detach(), dim=0
                )
                logits = torch.log(
                    torch.abs(self.act_fn(partial_gate))
                    * value_norms.unsqueeze(0)
                    + 1e-8
                )
                if self.utility_residual_rank:
                    residual = (
                        flat @ self.residual_input_factors
                    ) @ self.residual_output_factors + self.residual_bias
                    logits = logits + self.residual_blend * residual
                active = torch.argsort(
                    logits, dim=1, descending=True, stable=True
                )[:, : self.top_k]
                output = self._hard_output(flat, partial_gate, active)
                self.last_active_records = active
                self.last_input_coordinates = coordinates
                self.last_selection_logits = logits
                self.last_surrogate_used = False
                self.last_dense_output = None
                self.last_oracle = None
                if self.training and self.use_training_surrogate:
                    _, soft_mask = _cardinality_preserving_top_mask(
                        logits, self.top_k, self.temperature, torch
                    )
                    all_up = torch.nn.functional.linear(
                        flat, self.up_weight
                    )
                    full_gate = torch.nn.functional.linear(
                        flat, self.gate_weight
                    )
                    dense_values = self.act_fn(full_gate) * all_up
                    self.last_dense_output = torch.nn.functional.linear(
                        dense_values, self.down_weight
                    ).reshape(*shape[:-1], -1)
                    with torch.no_grad():
                        oracle_scores = (
                            torch.abs(dense_values)
                            * value_norms.unsqueeze(0)
                        )
                        self.last_oracle = torch.argsort(
                            oracle_scores,
                            dim=1,
                            descending=True,
                            stable=True,
                        )[:, : self.top_k]
                    values = (self.act_fn(partial_gate) * all_up).detach()
                    proxy = torch.nn.functional.linear(
                        soft_mask * values, self.down_weight.detach()
                    )
                    output = output + proxy - proxy.detach()
                    self.last_surrogate_used = True
            else:
                raise ValueError(
                    f"unsupported native gate channel mode {self.mode!r}"
                )
            self.last_output = output.reshape(*shape[:-1], -1)
            if self.mode == "dense_shadow":
                self.last_dense_output = self.last_output
            return self.last_output

    return NativeGateChannelMLP


__all__ = [
    "LowRankUtilityResidual",
    "NativeGateTraffic",
    "StructuredExpertTraffic",
    "balanced_expert_permutation",
    "block_contributions",
    "evaluate_native_gate_channel_shadow",
    "evaluate_native_gate_residual_shadow",
    "evaluate_structured_expert_shadow",
    "fit_low_rank_utility_residual",
    "load_native_gate_utility_residual",
    "native_gate_channel_traffic",
    "structured_expert_traffic",
    "train_native_gate_end_to_end",
    "train_native_gate_trace_student",
]
