"""Local trained-state screen for groupwise-Q4 OLMoE experts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.models.inspection import load_local_named_tensors
from engram.models.olmoe import audit_olmoe_source
from engram.tracing.format import TraceReader
from engram.utils import atomic_json


def round_to_bfloat16(values: ArrayLike) -> NDArray[np.float32]:
    """Round float32 values to BF16 precision with round-to-nearest-even."""

    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)
    return np.asarray(rounded, dtype=np.float32)


def symmetric_groupwise_q4_dequant(
    values: ArrayLike,
    *,
    group_size: int = 64,
) -> tuple[NDArray[np.float32], int]:
    """Simulate signed symmetric Q4 and return decoded weights plus stored bytes."""

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.size or not np.all(np.isfinite(matrix)):
        raise ValueError("values must be a non-empty finite matrix")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    rows, columns = matrix.shape
    groups = (columns + group_size - 1) // group_size
    padded_columns = groups * group_size
    padded = np.zeros((rows, padded_columns), dtype=np.float32)
    padded[:, :columns] = matrix
    blocks = padded.reshape(rows, groups, group_size)
    maximum = np.max(np.abs(blocks), axis=2, keepdims=True)
    scales = np.where(maximum == 0.0, 1.0, maximum / 7.0)
    scales = round_to_bfloat16(scales)
    codes = np.clip(np.rint(blocks / scales), -7, 7).astype(np.int8)
    decoded = (codes.astype(np.float32) * scales).reshape(
        rows, padded_columns
    )[:, :columns]
    code_bytes = (rows * columns + 1) // 2
    scale_bytes = rows * groups * 2
    return np.ascontiguousarray(decoded), code_bytes + scale_bytes


def symmetric_groupwise_dequant(
    values: ArrayLike,
    *,
    bits: int,
    group_size: int,
) -> tuple[NDArray[np.float32], int]:
    """Simulate signed symmetric 2–8 bit weights with FP16 group scales."""

    if not 2 <= bits <= 8:
        raise ValueError("bits must be in [2, 8]")
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.size or not np.all(np.isfinite(matrix)):
        raise ValueError("values must be a non-empty finite matrix")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    rows, columns = matrix.shape
    groups = (columns + group_size - 1) // group_size
    padded = np.zeros((rows, groups * group_size), dtype=np.float32)
    padded[:, :columns] = matrix
    blocks = padded.reshape(rows, groups, group_size)
    maximum_code = (1 << (bits - 1)) - 1
    maximum = np.max(np.abs(blocks), axis=2, keepdims=True)
    scales = np.where(maximum == 0.0, 1.0, maximum / maximum_code)
    scales = round_to_bfloat16(scales)
    codes = np.clip(
        np.rint(blocks / scales), -maximum_code, maximum_code
    ).astype(np.int8)
    decoded = (codes.astype(np.float32) * scales).reshape(
        rows, groups * group_size
    )[:, :columns]
    code_bytes = (rows * columns * bits + 7) // 8
    scale_bytes = rows * groups * 2
    return np.ascontiguousarray(decoded), code_bytes + scale_bytes


def _relative_l2(reference: NDArray[np.float32], candidate: NDArray[np.float32]) -> NDArray[np.float64]:
    difference = np.linalg.norm(candidate - reference, axis=1)
    denominator = np.maximum(np.linalg.norm(reference, axis=1), 1e-12)
    return difference / denominator


def evaluate_olmoe_q4_local(
    model: str | Path,
    trace: str | Path,
    out: str | Path,
    *,
    layer: int,
    group_size: int = 64,
    maximum_mean_relative_l2: float = 0.10,
) -> dict[str, Any]:
    """Measure decoded-Q4 expert output error on captured trained states."""

    model_path = Path(model).expanduser().resolve()
    audit = audit_olmoe_source(model_path)
    if audit.decision != "proceed_to_router_trace":
        raise ValueError("local OLMoE checkpoint failed exact source validation")
    layer_count = int(audit.dimensions["num_hidden_layers"] or 0)
    if layer < 0 or layer >= layer_count:
        raise ValueError("layer is outside the model")
    reader = TraceReader(trace)
    metadata = reader.manifest.get("metadata", {})
    if metadata.get("model_family") != "olmoe":
        raise ValueError("trace is not an OLMoE trace")
    fields = [
        f"layer_{layer}_mlp_input",
        f"layer_{layer}_mlp_output",
        f"layer_{layer}_expert_indices",
        f"layer_{layer}_expert_weights",
    ]
    shards = list(reader.iter_shards(fields))
    hidden = np.concatenate(
        [np.asarray(shard[fields[0]], dtype=np.float32) for shard in shards]
    )
    reference = np.concatenate(
        [np.asarray(shard[fields[1]], dtype=np.float32) for shard in shards]
    )
    indices = np.concatenate(
        [np.asarray(shard[fields[2]], dtype=np.int64) for shard in shards]
    )
    weights = np.concatenate(
        [np.asarray(shard[fields[3]], dtype=np.float32) for shard in shards]
    )
    experts = sorted(int(value) for value in np.unique(indices))
    prefix = f"model.layers.{layer}.mlp.experts"
    names = [
        f"{prefix}.{expert}.{projection}_proj.weight"
        for expert in experts
        for projection in ("gate", "up", "down")
    ]
    source = load_local_named_tensors(model_path, names)
    decoded: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    bytes_by_expert: dict[int, int] = {}
    for expert in experts:
        matrices = []
        stored_bytes = 0
        for projection in ("gate", "up", "down"):
            matrix, matrix_bytes = symmetric_groupwise_q4_dequant(
                source[f"{prefix}.{expert}.{projection}_proj.weight"],
                group_size=group_size,
            )
            matrices.append(matrix)
            stored_bytes += matrix_bytes
        decoded[expert] = (matrices[0], matrices[1], matrices[2])
        bytes_by_expert[expert] = stored_bytes

    candidate = np.zeros_like(reference)
    token_bytes = np.zeros(hidden.shape[0], dtype=np.int64)
    for row, state in enumerate(hidden):
        for position, expert_value in enumerate(indices[row]):
            expert = int(expert_value)
            gate, up, down = decoded[expert]
            gate_values = state @ gate.T
            gate_values = gate_values / (1.0 + np.exp(-gate_values))
            candidate[row] += weights[row, position] * (
                (gate_values * (state @ up.T)) @ down.T
            )
            token_bytes[row] += bytes_by_expert[expert]
    relative_l2 = _relative_l2(reference, candidate)
    dot = np.sum(reference * candidate, axis=1, dtype=np.float64)
    cosine = dot / np.maximum(
        np.linalg.norm(reference, axis=1)
        * np.linalg.norm(candidate, axis=1),
        1e-12,
    )
    dimensions = audit.dimensions
    hidden_size = int(dimensions["hidden_size"] or 0)
    intermediate = int(dimensions["intermediate_size"] or 0)
    num_experts = int(dimensions["num_experts"] or 0)
    ideal_all_expert_q4 = num_experts * 3 * hidden_size * intermediate // 2
    router_bytes = num_experts * hidden_size * 2
    complete_bytes = token_bytes + router_bytes
    mean_relative_l2 = float(np.mean(relative_l2))
    result = {
        "schema_version": 1,
        "experiment": "olmoe_trained_state_groupwise_q4_local",
        "model": str(model_path),
        "source_revision": audit.resolved_revision,
        "trace": str(Path(trace).resolve()),
        "layer": layer,
        "records": int(hidden.shape[0]),
        "unique_selected_experts": len(experts),
        "quantizer": {
            "kind": "signed_symmetric_groupwise_q4",
            "group_size": group_size,
            "code_range": [-7, 7],
            "scale_dtype": "bfloat16_executed",
        },
        "quality": {
            "mean_relative_l2": mean_relative_l2,
            "maximum_relative_l2": float(np.max(relative_l2)),
            "mean_cosine": float(np.mean(cosine)),
            "minimum_cosine": float(np.min(cosine)),
            "all_outputs_finite": bool(np.isfinite(candidate).all()),
        },
        "traffic": {
            "all_expert_ideal_q4_bytes_per_layer": ideal_all_expert_q4,
            "router_bf16_bytes_per_layer": router_bytes,
            "mean_complete_bytes_per_token": float(np.mean(complete_bytes)),
            "maximum_complete_bytes_per_token": int(np.max(complete_bytes)),
            "mean_fraction_of_all_expert_ideal_q4": float(
                np.mean(complete_bytes) / ideal_all_expert_q4
            ),
            "maximum_fraction_of_all_expert_ideal_q4": float(
                np.max(complete_bytes) / ideal_all_expert_q4
            ),
            "includes": "selected Q4 codes, BF16 group scales, and BF16 router",
            "excludes": "cache-line padding, activations, attention, and runtime overhead",
        },
        "screen": {
            "maximum_mean_relative_l2": maximum_mean_relative_l2,
            "local_quality_passed": (
                mean_relative_l2 <= maximum_mean_relative_l2
                and bool(np.isfinite(candidate).all())
            ),
            "traffic_projection_passed": (
                float(np.max(complete_bytes) / ideal_all_expert_q4) <= 0.45
            ),
            "passed": (
                mean_relative_l2 <= maximum_mean_relative_l2
                and bool(np.isfinite(candidate).all())
                and float(np.max(complete_bytes) / ideal_all_expert_q4) <= 0.45
            ),
            "scope": "local trained MLP states only; not an all-layer causal gate",
        },
    }
    output = Path(out)
    atomic_json(output, result)
    return result
