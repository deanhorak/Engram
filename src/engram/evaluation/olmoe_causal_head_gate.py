"""Causal/value-sensitive head allocation for sustained OLMoE attention.

The earlier teacher-guided head mask ranked dense attention probabilities.
That ranking cannot see value vectors or the downstream effect of replacing a
head's output.  This module instead installs one frozen scalar gate per
layer/head in an otherwise untouched OLMoE teacher:

    head = sparse_W16_C8_K4_S2 + gate * (full_causal - sparse)

Reverse-mode differentiation then exposes all 256 causal head sensitivities in
one model pass.  A deterministic hard projected update retains exactly 51
heads after every IHT step, which is the largest admissible mask
under the frozen 45-percent logical attention-read budget.

This remains a development experiment.  Gate training may read only the two
predeclared selection records.  The six internal-screen records are evaluated
only after the final mask and a second protocol have been frozen, and a pass
still requires a separately sealed confirmation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
import types
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

import engram.evaluation.olmoe_native_headwise as headwise
import engram.evaluation.olmoe_native_layer_rescue as layer_rescue
from engram.evaluation.olmoe_native_sustained import (
    _POSITIONS_PER_SEQUENCE,
    _THRESHOLDS,
    _attention_expectations,
)
from engram.models.olmoe import audit_olmoe_source
from engram.runtime.native_attention import NativeStreamingAttention
from engram.tracing.olmoe import _prepare_transformers_imports
from engram.utils import atomic_json, sha256_file, sha256_json


_PROTOCOL_EXPERIMENT = "olmoe_native_q7_causal_head_gate_protocol"
_TRAINING_EXPERIMENT = "olmoe_native_q7_causal_head_gate_training"
_SCREEN_PROTOCOL_EXPERIMENT = "olmoe_native_q7_causal_head_gate_screen_protocol"
_SCREEN_EXPERIMENT = "olmoe_native_q7_causal_head_gate_internal_screen"
_PROTOCOL_STATUS = "frozen_before_causal_gate_training"
_TRAINING_STATUS = "causal_head_gate_training_complete"
_SCREEN_PROTOCOL_STATUS = "frozen_before_causal_head_gate_native_screen"
_THREADS = 12
_LAYERS = 16
_HEADS = 16
_TOTAL_HEADS = _LAYERS * _HEADS
_RESCUED_HEADS = 51
_IHT_STEPS = 2
_SELECTION_SEQUENCE_INDICES = (0, 1)
_MASK_NAMES = ("M0", "M1", "M2")
_INTERNAL_SCREEN_SEQUENCE_INDICES = (2, 3, 4, 5, 6, 7)
_INTERNAL_SCREEN_SEQUENCE_ORDER = (3, 4, 7, 2, 5, 6)
_TRAINING_BANDS = (
    ("positions_16_31", 16, 32),
    ("positions_32_63", 32, 64),
    ("positions_64_95", 64, 96),
    ("positions_96_127", 96, 128),
)
_BASE_POLICY = {
    "local_window": 16,
    "older_candidates": 8,
    "older_top_k": 4,
    "sink_tokens": 2,
}
_FULL_POLICY = {
    "local_window": 128,
    "older_candidates": 8,
    "older_top_k": 4,
    "sink_tokens": 2,
}
_LOSS_CONTRACT = {
    "band_reduction": "equal_mean_over_four_affected_position_bands",
    "position_reduction": "mean_within_each_band",
    "teacher_to_student_kl_weight": 1.0,
    "teacher_to_student_kl_normalizer": 0.05,
    "hidden_relative_l2_weight": 1.0,
    "hidden_relative_l2_normalizer": 0.10,
    "positive_target_nll_delta_weight": 0.25,
    "positive_target_nll_delta_normalizer": 0.05,
    "teacher_top1_margin_deficit_weight": 0.05,
    "teacher_top1_margin_deficit_normalizer": 1.0,
}
_PROJECTED_GRADIENT_STEP = 1.0
_PROJECTED_GRADIENT_EPSILON = 1.0e-12
_SEED = 0
_BOUNDARY_ARTIFACT_NAMES = (
    "trace_protocol",
    "trace_metadata",
    "trace_arrays",
    "failed_head_mask",
    "failed_screen_protocol",
    "failed_screen_result",
    "headwise_library",
)
_BASE_BINDING_NAMES = (
    "package_manifest_sha256",
    "native_library_sha256",
    "dataset_sha256",
    "corpus_manifest_sha256",
    "teacher_reference_sha256",
    "teacher_arrays_sha256",
    "sustained_protocol_sha256",
    "sustained_result_sha256",
    "control_protocol_sha256",
    "control_result_sha256",
    "sweep_protocol_sha256",
    "sweep_result_sha256",
    "control_source_sha256",
    "sweep_source_sha256",
    "layer_rescue_protocol_sha256",
    "layer_rescue_result_sha256",
    "layered_native_library_sha256",
    "layer_rescue_source_sha256",
    "layer_rescue_source_inventory_sha256",
)
_LOSS_COMPONENT_NAMES = (
    "kl",
    "hidden_relative_l2",
    "positive_nll_delta",
    "top1_margin_deficit",
)
_NATIVE_METRIC_NAMES = (
    "tokens_seen",
    "local_entries",
    "active_older_entries",
    "candidate_key_bytes",
    "selected_value_bytes",
    "local_kv_bytes",
    "eviction_events",
    "older_candidate_entries_scored",
    "older_selected_entries",
    "sink_insertions",
    "heavy_hitter_updates",
    "state_bytes",
    "scratch_bytes",
)
_TRAINING_EVIDENCE_NAMES = (
    "exact_two_IHT_steps",
    "all_three_masks_executed",
    "four_backward_passes",
    "two_terminal_forward_only_passes",
    "selection_records_only",
    "no_internal_screen_record_access",
    "exact_51_after_every_IHT_step",
    "selected_mask_exact_51",
    "analytical_budget_exact",
    "teacher_weights_frozen",
    "CPU_only_native_oracle",
    "selected_mask_was_executed",
    "robust_improvement_vs_M0",
    "post_training_authentication",
)
_TRAINING_POST_AUTHENTICATION_NAMES = (
    "package",
    "reference_library",
    "layered_library",
    "dataset",
    "corpus_manifest",
    "teacher_reference",
    "teacher_arrays",
    "sustained_protocol",
    "sustained_result",
    "control_protocol",
    "control_result",
    "sweep_protocol",
    "sweep_result",
    "layer_rescue_protocol",
    "layer_rescue_result",
    "layer_rescue_historical_source_descriptors",
    "teacher_source_config",
    "teacher_source_index",
    "teacher_source_shards",
    "headwise_source_inventory",
    *_BOUNDARY_ARTIFACT_NAMES,
    "gate_protocol",
    "causal_gate_source",
    "framework_contract",
    "training_attention_library",
)
_PREREQUISITE_ARGUMENT_NAMES = frozenset(
    {
        "trace_protocol",
        "trace_protocol_sha256",
        "trace_metadata",
        "trace_metadata_sha256",
        "trace_arrays",
        "trace_arrays_sha256",
        "failed_head_mask",
        "failed_head_mask_sha256",
        "failed_screen_protocol",
        "failed_screen_protocol_sha256",
        "failed_screen_result",
        "failed_screen_result_sha256",
        "headwise_library",
        "headwise_library_sha256",
    }
)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for causal head-gate training"
        ) from exc
    return torch


def _validate_streaming_inputs(
    query_states: Any,
    key_states: Any,
    value_states: Any,
    *,
    local_window: int,
    older_candidates: int,
    older_top_k: int,
    sink_tokens: int,
) -> None:
    torch = _require_torch()
    if (
        not isinstance(query_states, torch.Tensor)
        or not isinstance(key_states, torch.Tensor)
        or not isinstance(value_states, torch.Tensor)
        or query_states.ndim != 4
        or query_states.shape != key_states.shape
        or query_states.shape != value_states.shape
        or query_states.device != key_states.device
        or query_states.device != value_states.device
        or query_states.dtype != key_states.dtype
        or query_states.dtype != value_states.dtype
        or not query_states.is_floating_point()
        or query_states.shape[0] <= 0
        or query_states.shape[1] <= 0
        or query_states.shape[2] <= 0
        or query_states.shape[3] <= 0
    ):
        raise ValueError(
            "streaming attention Q/K/V must be equal floating [B,H,T,D] tensors"
        )
    integer_values = (
        local_window,
        older_candidates,
        older_top_k,
        sink_tokens,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in integer_values
    ):
        raise ValueError("streaming attention capacities must be integers")
    if (
        local_window <= 0
        or older_candidates <= 0
        or older_top_k <= 0
        or older_top_k > older_candidates
        or sink_tokens < 0
        or sink_tokens >= older_candidates
    ):
        raise ValueError("streaming attention capacities are inconsistent")
    if not bool(torch.isfinite(query_states).all()):
        raise ValueError("streaming attention query is non-finite")
    if not bool(torch.isfinite(key_states).all()):
        raise ValueError("streaming attention key is non-finite")
    if not bool(torch.isfinite(value_states).all()):
        raise ValueError("streaming attention value is non-finite")


def _stable_descending_positions(
    scores: Sequence[Any],
    positions: Sequence[int],
) -> list[int]:
    """Return list indices by score descending, then position ascending."""

    if len(scores) != len(positions):
        raise ValueError("streaming attention score/position lengths differ")
    values = [float(score.detach().float().item()) for score in scores]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("streaming attention score is non-finite")
    return sorted(
        range(len(values)),
        key=lambda index: (-values[index], int(positions[index])),
    )


def _streaming_attention_replay(
    query_states: Any,
    key_states: Any,
    value_states: Any,
    *,
    local_window: int,
    older_candidates: int,
    older_top_k: int,
    sink_tokens: int,
) -> tuple[Any, dict[str, Any]]:
    """Replay the native heavy-hitter streaming policy over full-sequence QKV.

    Inputs have shape ``[batch, heads, positions, head_dimension]``.  OLMoE
    has equal query and KV head counts, so each head owns independent recent
    mass and older slots exactly as ``StreamingAttention`` does natively.

    This scalar implementation is retained only as a readable test/reference
    approximation.  PyTorch reductions need not reproduce the native
    sequential-float hard decisions under cancellation or ties.  Production
    gate training uses :func:`_native_sparse_straight_through` instead.
    """

    _validate_streaming_inputs(
        query_states,
        key_states,
        value_states,
        local_window=local_window,
        older_candidates=older_candidates,
        older_top_k=older_top_k,
        sink_tokens=sink_tokens,
    )
    torch = _require_torch()
    batch_size, heads, positions, head_dimension = query_states.shape
    scale = float(head_dimension) ** -0.5
    maximum_visible = local_window + older_top_k
    output = torch.empty_like(value_states)
    visible_positions = torch.full(
        (batch_size, heads, positions, maximum_visible),
        -1,
        dtype=torch.long,
        device=query_states.device,
    )
    visible_weights = torch.zeros(
        (batch_size, heads, positions, maximum_visible),
        dtype=torch.float32,
        device=query_states.device,
    )
    active_older_entries = torch.zeros(
        (batch_size, heads, positions),
        dtype=torch.long,
        device=query_states.device,
    )
    selected_older_entries = torch.zeros_like(active_older_entries)
    eviction_events = torch.zeros_like(active_older_entries)
    sink_insertions = 0
    heavy_hitter_updates = 0

    # The explicit B/H loops deliberately mirror the independently testable
    # per-query-head native state.  The production experiment is B=1,H=16.
    for batch in range(batch_size):
        for head in range(heads):
            recent_positions: list[int] = []
            recent_mass: list[float] = []
            older_positions: list[int | None] = [None] * older_candidates
            older_scores: list[float] = [0.0] * older_candidates
            for position in range(positions):
                if len(recent_positions) == local_window:
                    evicted_position = recent_positions.pop(0)
                    incoming_score = recent_mass.pop(0)
                    eviction_events[batch, head, position] = 1
                    destination: int | None = None
                    if evicted_position < sink_tokens:
                        destination = evicted_position
                    else:
                        for slot in range(sink_tokens, older_candidates):
                            if older_positions[slot] is None:
                                destination = slot
                                break
                        if destination is None:
                            destination = min(
                                range(sink_tokens, older_candidates),
                                key=lambda slot: (
                                    older_scores[slot],
                                    int(older_positions[slot]),
                                ),
                            )
                            # Equality is accepted by the native replacement
                            # rule; only a strictly smaller incoming mass loses.
                            if incoming_score < older_scores[destination]:
                                destination = None
                    if destination is not None:
                        older_positions[destination] = evicted_position
                        older_scores[destination] = incoming_score
                        if evicted_position < sink_tokens:
                            sink_insertions += 1
                        else:
                            heavy_hitter_updates += 1

                recent_positions.append(position)
                recent_mass.append(0.0)

                active_slots = [
                    slot
                    for slot, older_position in enumerate(older_positions)
                    if older_position is not None
                ]
                candidate_scores = [
                    (
                        query_states[batch, head, position].float()
                        * key_states[
                            batch,
                            head,
                            int(older_positions[slot]),
                        ].float()
                    ).sum()
                    * scale
                    for slot in active_slots
                ]
                candidate_positions = [
                    int(older_positions[slot]) for slot in active_slots
                ]
                candidate_order = _stable_descending_positions(
                    candidate_scores,
                    candidate_positions,
                )
                selected_order = candidate_order[
                    : min(older_top_k, len(candidate_order))
                ]
                selected_slots = [active_slots[index] for index in selected_order]
                selected_positions = [
                    int(older_positions[slot]) for slot in selected_slots
                ]
                visible = recent_positions + selected_positions
                score_tensors = [
                    (
                        query_states[batch, head, position].float()
                        * key_states[batch, head, key_position].float()
                    ).sum()
                    * scale
                    for key_position in visible
                ]
                scores = torch.stack(score_tensors)
                weights_f32 = torch.softmax(scores, dim=0, dtype=torch.float32)
                values = torch.stack(
                    [
                        value_states[batch, head, key_position]
                        for key_position in visible
                    ],
                    dim=0,
                )
                weights = weights_f32.to(value_states.dtype)
                output[batch, head, position] = torch.matmul(
                    weights.unsqueeze(0),
                    values,
                ).squeeze(0)

                visible_count = len(visible)
                visible_positions[batch, head, position, :visible_count] = torch.tensor(
                    visible,
                    dtype=torch.long,
                    device=query_states.device,
                )
                visible_weights[batch, head, position, :visible_count] = (
                    weights_f32.detach()
                )
                active_older_entries[batch, head, position] = len(active_slots)
                selected_older_entries[batch, head, position] = len(selected_slots)

                for local_index in range(len(recent_positions)):
                    recent_mass[local_index] = float(
                        np.float32(recent_mass[local_index])
                        + np.float32(weights_f32[local_index].detach().item())
                    )
                older_weight_start = len(recent_positions)
                for selected_index, slot in enumerate(selected_slots):
                    older_scores[slot] = float(
                        np.float32(older_scores[slot])
                        + np.float32(
                            weights_f32[older_weight_start + selected_index]
                            .detach()
                            .item()
                        )
                    )

    trace = {
        "visible_positions": visible_positions,
        "visible_weights": visible_weights,
        "active_older_entries": active_older_entries,
        "selected_older_entries": selected_older_entries,
        "eviction_events": eviction_events,
        "sink_insertions": torch.tensor(
            sink_insertions,
            dtype=torch.long,
            device=query_states.device,
        ),
        "heavy_hitter_updates": torch.tensor(
            heavy_hitter_updates,
            dtype=torch.long,
            device=query_states.device,
        ),
    }
    return output, trace


def replay_w16_c8_k4_s2(
    query_states: Any,
    key_states: Any,
    value_states: Any,
) -> Any:
    """Return only fixed-policy context; use the generalized helper for trace."""

    output, _trace = _streaming_attention_replay(
        query_states,
        key_states,
        value_states,
        **_BASE_POLICY,
    )
    return output


def full_causal_attention_context(
    query_states: Any,
    key_states: Any,
    value_states: Any,
) -> Any:
    """Return eager full-causal per-head context in ``[B,H,T,D]`` layout."""

    _validate_streaming_inputs(
        query_states,
        key_states,
        value_states,
        local_window=max(1, int(query_states.shape[2])),
        older_candidates=1,
        older_top_k=1,
        sink_tokens=0,
    )
    torch = _require_torch()
    positions = int(query_states.shape[2])
    scale = float(query_states.shape[-1]) ** -0.5
    scores = (
        torch.matmul(
            query_states,
            key_states.transpose(-1, -2),
        )
        * scale
    )
    causal = torch.ones(
        (positions, positions),
        dtype=torch.bool,
        device=query_states.device,
    ).triu(diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return torch.matmul(weights, value_states)


def _native_metric_dict(metrics: Any) -> dict[str, int]:
    fields = getattr(type(metrics), "__dataclass_fields__", {})
    if not fields:
        raise ValueError("native attention returned no metric schema")
    return {name: int(getattr(metrics, name)) for name in fields}


def _native_position_major(array: Any, name: str) -> np.ndarray:
    torch = _require_torch()
    if (
        not isinstance(array, torch.Tensor)
        or array.ndim != 4
        or array.shape[0] != 1
        or not array.is_floating_point()
        or not bool(torch.isfinite(array).all())
    ):
        raise ValueError(f"native attention {name} must be finite [1,H,T,D]")
    return np.ascontiguousarray(
        array.detach().float().cpu().numpy()[0].transpose(1, 0, 2),
        dtype=np.float32,
    )


def _expected_visible_counts(
    positions: int,
    *,
    local_window: int,
    older_candidates: int,
    older_top_k: int,
) -> np.ndarray:
    return np.asarray(
        [
            min(position + 1, local_window)
            + min(
                older_top_k,
                min(max(position + 1 - local_window, 0), older_candidates),
            )
            for position in range(positions)
        ],
        dtype=np.int64,
    )


def _native_identity_schedule(
    query_states: Any,
    key_states: Any,
    *,
    local_window: int,
    older_candidates: int,
    older_top_k: int,
    sink_tokens: int,
    attention_library: str | Path,
) -> tuple[Any, Any, dict[str, Any]]:
    """Extract exact native visible positions using token-identity values.

    With sequence length equal to head dimension (128 in the frozen OLMoE
    experiment), value at token ``t`` is basis vector ``e_t``.  The native
    output therefore is its exact float32 weight-by-position row, including
    the native sequential dot/softmax and all heavy-hitter decisions.
    """

    _validate_streaming_inputs(
        query_states,
        key_states,
        key_states,
        local_window=local_window,
        older_candidates=older_candidates,
        older_top_k=older_top_k,
        sink_tokens=sink_tokens,
    )
    torch = _require_torch()
    batch, heads, positions, head_dimension = query_states.shape
    if (
        batch != 1
        or positions != head_dimension
        or key_states.shape != query_states.shape
    ):
        raise ValueError(
            "native identity schedule requires B=1 and positions=head dimension"
        )
    library_path = Path(attention_library).expanduser().resolve()
    if not library_path.is_file():
        raise ValueError("native identity-schedule attention library is missing")
    query = _native_position_major(query_states, "query")
    key = _native_position_major(key_states, "key")
    identity = np.eye(positions, dtype=np.float32)
    identity_values = np.ascontiguousarray(
        np.broadcast_to(
            identity[:, None, :],
            (positions, heads, positions),
        )
    )
    started = time.perf_counter()
    with NativeStreamingAttention(
        query_heads=heads,
        key_value_heads=heads,
        head_dimension=head_dimension,
        local_window=local_window,
        older_candidates=older_candidates,
        older_top_k=older_top_k,
        sink_tokens=sink_tokens,
        scale=head_dimension**-0.5,
        library=library_path,
    ) as native:
        native_rows, metrics = native.stream(query, key, identity_values)
    elapsed = time.perf_counter() - started
    weights = np.ascontiguousarray(
        native_rows.transpose(1, 0, 2)[None],
        dtype=np.float32,
    )
    if (
        weights.shape != (1, heads, positions, positions)
        or not np.isfinite(weights).all()
        or float(weights.min()) < 0.0
        or float(weights.max()) > 1.0 + 2.0e-6
    ):
        raise ValueError("native identity schedule returned invalid weights")
    expected = _expected_visible_counts(
        positions,
        local_window=local_window,
        older_candidates=older_candidates,
        older_top_k=older_top_k,
    )
    positive = weights > 0.0
    observed = positive.sum(axis=-1, dtype=np.int64)
    expected_grid = np.broadcast_to(expected, observed.shape)
    if not np.array_equal(observed, expected_grid):
        raise ValueError(
            "native identity schedule has nonpositive or ambiguous visible entries"
        )
    row_sums = weights.sum(axis=-1, dtype=np.float64)
    maximum_row_error = float(np.max(np.abs(row_sums - 1.0)))
    if not math.isfinite(maximum_row_error) or maximum_row_error > 2.0e-5:
        raise ValueError("native identity schedule row sum is invalid")
    positive_weights = weights[positive]
    if positive_weights.size != int(expected.sum()) * heads:
        raise ValueError("native identity schedule population is ambiguous")
    minimum_positive = float(positive_weights.min())
    if not math.isfinite(minimum_positive) or minimum_positive <= 0.0:
        raise ValueError("native identity schedule has nonpositive weight")
    maximum_visible = local_window + older_top_k
    indices = np.full(
        (1, heads, positions, maximum_visible),
        -1,
        dtype=np.int64,
    )
    for head in range(heads):
        for position in range(positions):
            visible = np.flatnonzero(positive[0, head, position])
            count = int(expected[position])
            if visible.size != count or count > maximum_visible:
                raise ValueError("native identity schedule index count is invalid")
            expected_local = np.arange(
                max(0, position - local_window + 1),
                position + 1,
                dtype=np.int64,
            )
            if not np.all(np.isin(expected_local, visible)):
                raise ValueError(
                    "native identity schedule omits an expected local position"
                )
            if np.any(visible > position) or np.unique(visible).size != visible.size:
                raise ValueError("native identity schedule is noncausal or duplicated")
            indices[0, head, position, :count] = visible
    metric_values = _native_metric_dict(metrics)
    if (
        metric_values["tokens_seen"] != positions
        or metric_values["local_entries"] != min(positions, local_window)
        or metric_values["active_older_entries"]
        != heads * min(max(positions - local_window, 0), older_candidates)
    ):
        raise ValueError("native identity schedule counters are invalid")
    indices_tensor = torch.as_tensor(
        indices,
        dtype=torch.long,
        device=query_states.device,
    )
    weights_tensor = torch.as_tensor(
        weights,
        dtype=torch.float32,
        device=query_states.device,
    )
    diagnostics = {
        "expected_visible_counts": expected.tolist(),
        "observed_visible_count_minimum": int(observed.min()),
        "observed_visible_count_maximum": int(observed.max()),
        "maximum_row_sum_error": maximum_row_error,
        "minimum_positive_weight": minimum_positive,
        "indices_sha256": sha256_json(indices.tolist()),
        "native_metrics": metric_values,
        "elapsed_seconds": elapsed,
        "attention_library_sha256": sha256_file(library_path),
    }
    return indices_tensor, weights_tensor, diagnostics


def _differentiable_gathered_attention(
    query_states: Any,
    key_states: Any,
    value_states: Any,
    indices: Any,
) -> Any:
    """Compute a batched differentiable attention surrogate on fixed indices."""

    _validate_streaming_inputs(
        query_states,
        key_states,
        value_states,
        local_window=1,
        older_candidates=1,
        older_top_k=1,
        sink_tokens=0,
    )
    torch = _require_torch()
    batch, heads, positions, head_dimension = query_states.shape
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype != torch.long
        or indices.ndim != 4
        or indices.shape[:3] != (batch, heads, positions)
        or indices.shape[-1] <= 0
        or indices.device != query_states.device
        or bool((indices < -1).any())
        or bool((indices >= positions).any())
    ):
        raise ValueError("differentiable gathered-attention indices are invalid")
    valid = indices >= 0
    if bool((valid.sum(dim=-1) <= 0).any()):
        raise ValueError("differentiable gathered-attention row is empty")
    canonical = torch.arange(
        positions,
        dtype=torch.long,
        device=indices.device,
    ).reshape(1, 1, positions, 1)
    if bool(((indices > canonical) & valid).any()):
        raise ValueError("differentiable gathered-attention indices are noncausal")
    sorted_valid = (
        torch.where(
            valid,
            indices,
            torch.full_like(indices, positions),
        )
        .sort(dim=-1)
        .values
    )
    if bool(
        (
            (sorted_valid[..., 1:] == sorted_valid[..., :-1])
            & (sorted_valid[..., 1:] < positions)
        ).any()
    ):
        raise ValueError("differentiable gathered-attention indices repeat")
    safe = indices.clamp_min(0)
    batch_index = torch.arange(
        batch,
        device=query_states.device,
    ).reshape(batch, 1, 1, 1)
    head_index = torch.arange(
        heads,
        device=query_states.device,
    ).reshape(1, heads, 1, 1)
    gathered_keys = key_states[batch_index, head_index, safe, :]
    gathered_values = value_states[batch_index, head_index, safe, :]
    scores = (query_states.unsqueeze(3).float() * gathered_keys.float()).sum(dim=-1) * (
        head_dimension**-0.5
    )
    scores = scores.masked_fill(~valid, float("-inf"))
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
    return (weights.unsqueeze(-1) * gathered_values.float()).sum(dim=-2)


def _native_actual_value_forward(
    query_states: Any,
    key_states: Any,
    value_states: Any,
    *,
    policy: Mapping[str, int],
    attention_library: str | Path,
) -> tuple[Any, dict[str, Any]]:
    torch = _require_torch()
    _validate_streaming_inputs(
        query_states,
        key_states,
        value_states,
        **dict(policy),
    )
    batch, heads, positions, head_dimension = query_states.shape
    if batch != 1 or key_states.shape[1] != heads:
        raise ValueError("native actual-value forward requires B=1 and QH=KVH")
    library_path = Path(attention_library).expanduser().resolve()
    started = time.perf_counter()
    with NativeStreamingAttention(
        query_heads=heads,
        key_value_heads=heads,
        head_dimension=head_dimension,
        scale=head_dimension**-0.5,
        library=library_path,
        **dict(policy),
    ) as native:
        native_output, metrics = native.stream(
            _native_position_major(query_states, "query"),
            _native_position_major(key_states, "key"),
            _native_position_major(value_states, "value"),
        )
    elapsed = time.perf_counter() - started
    output = torch.as_tensor(
        np.ascontiguousarray(native_output.transpose(1, 0, 2)[None]),
        dtype=torch.float32,
        device=query_states.device,
    )
    if output.shape != query_states.shape or not bool(torch.isfinite(output).all()):
        raise ValueError("native actual-value attention output is invalid")
    return output, {
        "native_metrics": _native_metric_dict(metrics),
        "elapsed_seconds": elapsed,
        "attention_library_sha256": sha256_file(library_path),
    }


def _native_sparse_straight_through(
    query_states: Any,
    key_states: Any,
    value_states: Any,
    *,
    attention_library: str | Path,
    policy: Mapping[str, int] = _BASE_POLICY,
) -> tuple[Any, dict[str, Any]]:
    """Exact native sparse forward with a gathered differentiable backward."""

    policy_value = {name: int(value) for name, value in policy.items()}
    schedule_started = time.perf_counter()
    indices, _native_weights, schedule = _native_identity_schedule(
        query_states,
        key_states,
        attention_library=attention_library,
        **policy_value,
    )
    exact, actual = _native_actual_value_forward(
        query_states,
        key_states,
        value_states,
        policy=policy_value,
        attention_library=attention_library,
    )
    if actual["native_metrics"] != schedule["native_metrics"]:
        raise ValueError("native identity and actual-value sparse counters differ")
    surrogate_started = time.perf_counter()
    surrogate = _differentiable_gathered_attention(
        query_states,
        key_states,
        value_states,
        indices,
    )
    surrogate_seconds = time.perf_counter() - surrogate_started
    # The parenthesized difference is bit-exact zero in forward arithmetic,
    # while its derivative is the differentiable gathered surrogate.
    result = exact.detach() + (surrogate - surrogate.detach())
    diagnostics = {
        "mode": "native_exact_sparse_forward_gathered_surrogate_backward",
        "schedule": schedule,
        "actual_value": actual,
        "surrogate_elapsed_seconds": surrogate_seconds,
        "total_elapsed_seconds": time.perf_counter() - schedule_started,
        "exact_forward_sha256": hashlib.sha256(
            exact.detach().cpu().numpy().tobytes()
        ).hexdigest(),
    }
    return result, diagnostics


def _native_full_straight_through(
    query_states: Any,
    key_states: Any,
    value_states: Any,
    *,
    attention_library: str | Path,
) -> tuple[Any, dict[str, Any]]:
    """Exact native W128 forward with differentiable full-causal backward."""

    started = time.perf_counter()
    exact, actual = _native_actual_value_forward(
        query_states,
        key_states,
        value_states,
        policy=_FULL_POLICY,
        attention_library=attention_library,
    )
    surrogate_started = time.perf_counter()
    surrogate = full_causal_attention_context(
        query_states,
        key_states,
        value_states,
    )
    surrogate_seconds = time.perf_counter() - surrogate_started
    result = exact.detach() + (surrogate - surrogate.detach())
    diagnostics = {
        "mode": "native_exact_W128_forward_full_causal_surrogate_backward",
        "actual_value": actual,
        "surrogate_elapsed_seconds": surrogate_seconds,
        "total_elapsed_seconds": time.perf_counter() - started,
        "exact_forward_sha256": hashlib.sha256(
            exact.detach().cpu().numpy().tobytes()
        ).hexdigest(),
    }
    return result, diagnostics


def _mix_head_outputs(sparse: Any, full: Any, gates: Any) -> Any:
    """Mix sparse/full pre-output-projection head contexts."""

    torch = _require_torch()
    if (
        not isinstance(sparse, torch.Tensor)
        or not isinstance(full, torch.Tensor)
        or not isinstance(gates, torch.Tensor)
        or sparse.ndim != 4
        or sparse.shape != full.shape
        or sparse.device != full.device
        or sparse.dtype != full.dtype
        or not gates.is_floating_point()
        or gates.device != sparse.device
        or not bool(torch.isfinite(gates).all())
        or bool((gates < 0).any())
        or bool((gates > 1).any())
    ):
        raise ValueError("causal head output/gate tensors are invalid")
    if gates.ndim == 1 and gates.shape[0] == sparse.shape[1]:
        expanded = gates.reshape(1, -1, 1, 1)
    elif gates.ndim == 2 and gates.shape == sparse.shape[:2]:
        expanded = gates.reshape(sparse.shape[0], sparse.shape[1], 1, 1)
    elif gates.ndim == 4:
        expanded = gates
    else:
        raise ValueError("causal gates must be [H], [B,H], or broadcast [B,H,T,D]")
    try:
        torch.broadcast_shapes(tuple(sparse.shape), tuple(expanded.shape))
    except RuntimeError as exc:
        raise ValueError("causal gates do not broadcast over head outputs") from exc
    expanded = expanded.to(sparse.dtype)
    linear = sparse + expanded * (full - sparse)
    # Preserve bit-exact hard endpoints while retaining the useful linear gate
    # derivative for projected-gradient training.
    exact = torch.where(
        expanded == 0,
        sparse,
        torch.where(expanded == 1, full, linear),
    )
    return exact.detach() + (linear - linear.detach())


def mix_head_contexts(full: Any, base: Any, gates: Any) -> Any:
    """Public full/base-order alias for pre-output-projection head mixing."""

    return _mix_head_outputs(base, full, gates)


def _project_top_k(scores: Any, count: int) -> Any:
    """Project finite scores to an exact deterministic top-k Boolean mask.

    Scores rank descending.  Exact ties choose the lower C-order flattened
    index, so an all-zero 16x16 array selects flat indices 0 through 50.
    """

    torch = _require_torch()
    is_tensor = isinstance(scores, torch.Tensor)
    if is_tensor:
        if scores.requires_grad:
            values = scores.detach().float().cpu().numpy()
        else:
            values = scores.float().cpu().numpy()
    else:
        values = np.asarray(scores)
    if (
        values.ndim == 0
        or values.size == 0
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or count > values.size
        or not np.issubdtype(values.dtype, np.number)
        or not np.isfinite(values).all()
    ):
        raise ValueError("top-k projection scores/count are invalid")
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    indices = np.arange(flat.size, dtype=np.int64)
    order = np.lexsort((indices, -flat))
    mask = np.zeros(flat.size, dtype=np.bool_)
    mask[order[:count]] = True
    mask = mask.reshape(values.shape)
    if is_tensor:
        return torch.as_tensor(mask, dtype=torch.bool, device=scores.device)
    return mask


def project_exact_head_mask(scores: Any, count: int = _RESCUED_HEADS) -> Any:
    """Public exact-head projection wrapper."""

    return _project_top_k(scores, count)


def _projected_gate_step(
    current_mask: Any,
    gradient: Any,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply one frozen normalized IHT update and exact top-51 projection."""

    mask = np.asarray(current_mask)
    values = np.asarray(gradient, dtype=np.float64)
    if (
        mask.shape != (_LAYERS, _HEADS)
        or mask.dtype != np.bool_
        or values.shape != (_LAYERS, _HEADS)
        or not np.isfinite(values).all()
    ):
        raise ValueError("projected causal gate mask/gradient is invalid")
    if int(mask.sum()) not in {0, _RESCUED_HEADS}:
        raise ValueError("projected causal gate input mask has invalid count")
    gradient_rms = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
    if not math.isfinite(gradient_rms) or gradient_rms <= 0.0:
        raise ValueError("projected causal gate gradient RMS is invalid")
    scores = mask.astype(np.float64) - _PROJECTED_GRADIENT_STEP * values / (
        gradient_rms + _PROJECTED_GRADIENT_EPSILON
    )
    next_mask = np.asarray(
        _project_top_k(scores, _RESCUED_HEADS),
        dtype=np.bool_,
    )
    if next_mask.shape != (_LAYERS, _HEADS) or int(next_mask.sum()) != _RESCUED_HEADS:
        raise ValueError("projected causal gate update violated exact budget")
    return scores, next_mask, gradient_rms


def _normalized_bands(
    bands: Sequence[Any] | None,
    *,
    positions: int,
) -> tuple[tuple[str, int, int], ...]:
    source = _TRAINING_BANDS if bands is None else bands
    result: list[tuple[str, int, int]] = []
    for index, band in enumerate(source):
        if isinstance(band, Mapping):
            name = str(band.get("name", f"band_{index}"))
            start = band.get("start")
            stop = band.get("stop")
        elif isinstance(band, Sequence) and not isinstance(band, (str, bytes)):
            if len(band) == 2:
                name = f"band_{index}"
                start, stop = band
            elif len(band) == 3:
                name, start, stop = band
                name = str(name)
            else:
                raise ValueError("distillation band tuple is invalid")
        else:
            raise ValueError("distillation band is invalid")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(stop, bool)
            or not isinstance(stop, int)
            or not name
            or start < 0
            or stop <= start
            or stop > positions
        ):
            raise ValueError("distillation band bounds are invalid")
        result.append((name, start, stop))
    if not result:
        raise ValueError("distillation requires at least one band")
    occupied: set[int] = set()
    for _name, start, stop in result:
        current = set(range(start, stop))
        if occupied.intersection(current):
            raise ValueError("distillation bands overlap")
        occupied.update(current)
    return tuple(result)


def _distillation_loss(
    student_logits: Any,
    teacher_logits: Any,
    student_hidden: Any,
    teacher_hidden: Any,
    targets: Any,
    *,
    bands: Sequence[Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Band-balanced causal distillation objective.

    KL and relative hidden L2 are the primary terms.  Positive target-NLL
    drift and loss of the dense teacher's top-1 margin are lower-weight,
    differentiable safeguards aligned with the native semantic gate.
    """

    torch = _require_torch()
    if (
        not all(
            isinstance(value, torch.Tensor)
            for value in (
                student_logits,
                teacher_logits,
                student_hidden,
                teacher_hidden,
                targets,
            )
        )
        or student_logits.ndim != 3
        or student_logits.shape != teacher_logits.shape
        or student_hidden.ndim != 3
        or student_hidden.shape != teacher_hidden.shape
        or student_logits.shape[:2] != student_hidden.shape[:2]
        or targets.shape != student_logits.shape[:2]
        or student_logits.device != teacher_logits.device
        or student_hidden.device != teacher_hidden.device
        or targets.device != student_logits.device
        or student_logits.device != student_hidden.device
        or not student_logits.is_floating_point()
        or not teacher_logits.is_floating_point()
        or not student_hidden.is_floating_point()
        or not teacher_hidden.is_floating_point()
        or targets.dtype != torch.long
        or student_logits.shape[-1] < 2
    ):
        raise ValueError("causal distillation tensor contract is invalid")
    for value in (
        student_logits,
        teacher_logits,
        student_hidden,
        teacher_hidden,
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("causal distillation tensor is non-finite")
    if bool((targets < 0).any()) or bool((targets >= student_logits.shape[-1]).any()):
        raise ValueError("causal distillation target is out of range")

    normalized_bands = _normalized_bands(
        bands,
        positions=int(student_logits.shape[1]),
    )
    student_log_probability = torch.log_softmax(
        student_logits.float(),
        dim=-1,
    )
    teacher_log_probability = torch.log_softmax(
        teacher_logits.float(),
        dim=-1,
    )
    teacher_probability = teacher_log_probability.exp()
    per_position_kl = (
        teacher_probability * (teacher_log_probability - student_log_probability)
    ).sum(dim=-1)

    hidden_difference = (student_hidden.float() - teacher_hidden.float()).norm(dim=-1)
    hidden_denominator = teacher_hidden.float().norm(dim=-1).clamp_min(1.0e-12)
    per_position_hidden = hidden_difference / hidden_denominator

    gathered_student = student_log_probability.gather(
        -1,
        targets.unsqueeze(-1),
    ).squeeze(-1)
    gathered_teacher = teacher_log_probability.gather(
        -1,
        targets.unsqueeze(-1),
    ).squeeze(-1)
    per_position_positive_nll = torch.relu(-gathered_student + gathered_teacher)

    teacher_top = teacher_logits.float().argmax(dim=-1)
    teacher_top_value = (
        teacher_logits.float()
        .gather(
            -1,
            teacher_top.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    student_teacher_top_value = (
        student_logits.float()
        .gather(
            -1,
            teacher_top.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    teacher_other = teacher_logits.float().clone()
    student_other = student_logits.float().clone()
    teacher_other.scatter_(-1, teacher_top.unsqueeze(-1), float("-inf"))
    student_other.scatter_(-1, teacher_top.unsqueeze(-1), float("-inf"))
    teacher_margin = teacher_top_value - teacher_other.max(dim=-1).values
    student_margin = student_teacher_top_value - student_other.max(dim=-1).values
    per_position_margin_deficit = torch.relu(teacher_margin - student_margin)

    per_position = {
        "kl": per_position_kl,
        "hidden_relative_l2": per_position_hidden,
        "positive_nll_delta": per_position_positive_nll,
        "top1_margin_deficit": per_position_margin_deficit,
    }
    components: dict[str, Any] = {}
    band_components: dict[str, Any] = {}
    for name, values in per_position.items():
        means = []
        for band_name, start, stop in normalized_bands:
            band_mean = values[:, start:stop].mean()
            means.append(band_mean)
            band_components[f"{band_name}_{name}"] = band_mean
        components[name] = torch.stack(means).mean()

    total = (
        components["kl"]
        / _LOSS_CONTRACT["teacher_to_student_kl_normalizer"]
        * _LOSS_CONTRACT["teacher_to_student_kl_weight"]
        + components["hidden_relative_l2"]
        / _LOSS_CONTRACT["hidden_relative_l2_normalizer"]
        * _LOSS_CONTRACT["hidden_relative_l2_weight"]
        + components["positive_nll_delta"]
        / _LOSS_CONTRACT["positive_target_nll_delta_normalizer"]
        * _LOSS_CONTRACT["positive_target_nll_delta_weight"]
        + components["top1_margin_deficit"]
        / _LOSS_CONTRACT["teacher_top1_margin_deficit_normalizer"]
        * _LOSS_CONTRACT["teacher_top1_margin_deficit_weight"]
    )
    components["total"] = total
    components["bands"] = band_components
    return total, components


def band_balanced_causal_loss(
    student_logits: Any,
    teacher_logits: Any,
    student_hidden: Any,
    teacher_hidden: Any,
    targets: Any,
    *,
    bands: Sequence[Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Public alias for the frozen causal gate objective."""

    return _distillation_loss(
        student_logits,
        teacher_logits,
        student_hidden,
        teacher_hidden,
        targets,
        bands=bands,
    )


def _selection_slices_only(
    array: Any,
    selection_sequence_indices: Sequence[int],
    positions_per_sequence: int = _POSITIONS_PER_SEQUENCE,
) -> dict[int, Any]:
    """Slice only explicitly authorized training sequences from a flat array."""

    if (
        isinstance(positions_per_sequence, bool)
        or not isinstance(positions_per_sequence, int)
        or positions_per_sequence <= 0
        or not selection_sequence_indices
    ):
        raise ValueError("selection slice contract is invalid")
    normalized: list[int] = []
    for sequence_index in selection_sequence_indices:
        if (
            isinstance(sequence_index, bool)
            or not isinstance(sequence_index, int)
            or sequence_index < 0
            or sequence_index in normalized
        ):
            raise ValueError("selection sequence indices are invalid")
        normalized.append(sequence_index)
    if tuple(normalized) != _SELECTION_SEQUENCE_INDICES:
        raise ValueError("causal gate may access only selection indices [0,1]")
    length = int(array.shape[0])
    required = (max(normalized) + 1) * positions_per_sequence
    if length < required:
        raise ValueError("selection source array is too short")
    result: dict[int, Any] = {}
    for sequence_index in normalized:
        value = array[
            sequence_index * positions_per_sequence : (sequence_index + 1)
            * positions_per_sequence
        ]
        if isinstance(value, np.ndarray):
            value = value.copy()
        elif hasattr(value, "clone"):
            value = value.clone()
        elif hasattr(value, "copy"):
            value = value.copy()
        else:
            raise ValueError("selection source array cannot be copied")
        if hasattr(value, "base") and value.base is not None:
            raise ValueError("selection slice unexpectedly shares source storage")
        result[sequence_index] = value
    return result


def _current_source_inventory(
    context: Mapping[str, Any],
) -> dict[str, str]:
    inventory = headwise._current_source_inventory(
        context["layer_rescue_historical_source_inventory"]
    )
    repository = Path(__file__).resolve().parents[3]
    for relative in (
        "src/engram/evaluation/olmoe_causal_head_gate.py",
        "src/engram/runtime/native_attention.py",
        "native/include/engram/streaming_attention_c.h",
        "native/src/streaming_attention_c.cpp",
    ):
        source = repository / relative
        if not source.is_file():
            raise ValueError(f"causal gate source is missing: {relative}")
        inventory[relative] = sha256_file(source)
    return dict(sorted(inventory.items()))


def _framework_contract() -> dict[str, str]:
    torch = _require_torch()
    _prepare_transformers_imports()
    try:
        import transformers
        import transformers.models.olmoe.modeling_olmoe as modeling_olmoe
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for causal head-gate training"
        ) from exc
    modeling_path = Path(modeling_olmoe.__file__).resolve()
    return {
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "transformers_olmoe_modeling_path": str(modeling_path),
        "transformers_olmoe_modeling_sha256": sha256_file(modeling_path),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    return headwise._read_object(path, label)


def _require_digest(path: Path, supplied: str, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"{label} is missing")
    actual = sha256_file(path)
    if actual != str(supplied).lower():
        raise ValueError(f"{label} SHA-256 is invalid")
    return actual


def _authenticate_attention_library(
    attention_library: str | Path,
    attention_library_sha256: str,
) -> tuple[Path, str]:
    path = Path(attention_library).expanduser().resolve()
    digest = _require_digest(
        path,
        attention_library_sha256,
        "native streaming-attention library",
    )
    # Opening the wrapper proves this DSO exports the raw streaming ABI rather
    # than merely sharing a related token-runtime implementation.
    try:
        with NativeStreamingAttention(
            query_heads=1,
            key_value_heads=1,
            head_dimension=1,
            local_window=1,
            older_candidates=1,
            older_top_k=1,
            sink_tokens=0,
            library=path,
        ):
            pass
    except (AttributeError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            "native attention library lacks the required raw streaming ABI"
        ) from exc
    return path, digest


def _authenticate_failed_headwise_boundary(
    context: Mapping[str, Any],
    *,
    trace_protocol: str | Path,
    trace_protocol_sha256: str,
    trace_metadata: str | Path,
    trace_metadata_sha256: str,
    trace_arrays: str | Path,
    trace_arrays_sha256: str,
    failed_head_mask: str | Path,
    failed_head_mask_sha256: str,
    failed_screen_protocol: str | Path,
    failed_screen_protocol_sha256: str,
    failed_screen_result: str | Path,
    failed_screen_result_sha256: str,
    headwise_library: str | Path,
    headwise_library_sha256: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Authenticate the complete, immutable attention-mass-mask failure."""

    paths = {
        "trace_protocol": Path(trace_protocol).expanduser().resolve(),
        "trace_metadata": Path(trace_metadata).expanduser().resolve(),
        "trace_arrays": Path(trace_arrays).expanduser().resolve(),
        "failed_head_mask": Path(failed_head_mask).expanduser().resolve(),
        "failed_screen_protocol": Path(failed_screen_protocol).expanduser().resolve(),
        "failed_screen_result": Path(failed_screen_result).expanduser().resolve(),
        "headwise_library": Path(headwise_library).expanduser().resolve(),
    }
    supplied = {
        "trace_protocol": trace_protocol_sha256,
        "trace_metadata": trace_metadata_sha256,
        "trace_arrays": trace_arrays_sha256,
        "failed_head_mask": failed_head_mask_sha256,
        "failed_screen_protocol": failed_screen_protocol_sha256,
        "failed_screen_result": failed_screen_result_sha256,
        "headwise_library": headwise_library_sha256,
    }
    hashes = {
        name + "_sha256": _require_digest(paths[name], supplied[name], name)
        for name in paths
    }
    old_source_path = Path(headwise.__file__).resolve()
    old_source_hash = sha256_file(old_source_path)
    old_inventory = headwise._current_source_inventory(
        context["layer_rescue_historical_source_inventory"]
    )
    (
        trace_protocol_value,
        _trace_metadata_value,
        failed_mask,
        _attention_maps,
        trace_hashes,
    ) = headwise._validated_mask_artifacts(
        context=context,
        source_sha256=old_source_hash,
        source_inventory=old_inventory,
        trace_protocol=paths["trace_protocol"],
        trace_protocol_sha256=trace_protocol_sha256,
        trace_metadata=paths["trace_metadata"],
        trace_metadata_sha256=trace_metadata_sha256,
        trace_arrays=paths["trace_arrays"],
        trace_arrays_sha256=trace_arrays_sha256,
        head_mask=paths["failed_head_mask"],
        head_mask_sha256=failed_head_mask_sha256,
    )
    screen_protocol = _read_json(
        paths["failed_screen_protocol"],
        "failed head-wise screen protocol",
    )
    headwise._validate_screen_protocol(
        screen_protocol,
        context,
        trace_protocol=trace_protocol_value,
        mask=failed_mask,
        trace_hashes=trace_hashes,
        candidate_library_sha256=hashes["headwise_library_sha256"],
        protocol_sha256=hashes["failed_screen_protocol_sha256"],
        supplied_protocol_sha256=failed_screen_protocol_sha256,
        source_sha256=old_source_hash,
        source_inventory=old_inventory,
    )
    result = _read_json(
        paths["failed_screen_result"],
        "failed head-wise screen result",
    )
    internal_order = [
        int(row["sequence_index"]) for row in context["split"]["internal_holdout"]
    ]
    artifacts = result.get("artifacts")
    internal = result.get("internal_screen_result")
    post = result.get("post_run_authentication")
    parity = result.get("all_base_headwise_parity")
    if (
        result.get("schema_version") != 1
        or result.get("experiment") != headwise._SCREEN_EXPERIMENT
        or result.get("status") != "headwise_screen_development_complete"
        or result.get("evidence_passed") is not True
        or result.get("internal_screen_quality_passed") is not False
        or result.get("fresh_eight_sequence_confirmation_required") is not False
        or result.get("decision")
        != "investigate_value_sensitivity_or_dynamic_head_allocation"
        or not isinstance(artifacts, Mapping)
        or artifacts.get("screen_protocol_sha256")
        != hashes["failed_screen_protocol_sha256"]
        or artifacts.get("head_mask_sha256") != hashes["failed_head_mask_sha256"]
        or artifacts.get("candidate_native_library_sha256")
        != hashes["headwise_library_sha256"]
        or result.get("head_mask_identity_sha256")
        != failed_mask["attention_head_mask_sha256"]
        or not isinstance(parity, Mapping)
        or parity.get("passed") is not True
        or not isinstance(internal, Mapping)
        or internal.get("quality_passed") is not False
        or internal.get("evidence_passed") is not True
        or internal.get("sequence_indices") != internal_order
        or internal_order != list(_INTERNAL_SCREEN_SEQUENCE_ORDER)
        or not isinstance(post, Mapping)
        or not post
        or any(value is not True for value in post.values())
    ):
        raise ValueError("failed head-wise boundary conclusion is invalid")
    return hashes, {
        "failed_mask": failed_mask,
        "failed_screen_protocol": screen_protocol,
        "failed_screen_result": result,
        "paths": paths,
    }


def _training_split_contract(context: Mapping[str, Any]) -> dict[str, Any]:
    selection = list(context["split"]["selection"])
    internal = list(context["split"]["internal_holdout"])
    selection_indices = [int(row["sequence_index"]) for row in selection]
    internal_order = [int(row["sequence_index"]) for row in internal]
    if (
        selection_indices != list(_SELECTION_SEQUENCE_INDICES)
        or internal_order != list(_INTERNAL_SCREEN_SEQUENCE_ORDER)
        or set(selection_indices).intersection(internal_order)
        or sorted(internal_order) != list(_INTERNAL_SCREEN_SEQUENCE_INDICES)
    ):
        raise ValueError("causal gate record split is invalid")
    return {
        "selection_records": selection,
        "selection_sequence_indices": selection_indices,
        "gradient_sequence_indices_per_step": selection_indices,
        "iht_steps": _IHT_STEPS,
        "terminal_evaluation_sequence_indices": selection_indices,
        "internal_screen_records": internal,
        "internal_screen_sequence_order": internal_order,
        "prohibited_internal_screen_sequence_indices": sorted(internal_order),
        "internal_screen_records_used": False,
    }


def _build_gate_protocol(
    context: Mapping[str, Any],
    *,
    prerequisite_hashes: Mapping[str, str],
    source_sha256: str,
    source_inventory: Mapping[str, str],
    framework_contract: Mapping[str, str],
    attention_library_path: Path,
    attention_library_sha256: str,
    device: str,
    threads: int,
) -> dict[str, Any]:
    split = _training_split_contract(context)
    selection_inputs = [
        context["input_ids"][sequence_index][:-1]
        for sequence_index in split["selection_sequence_indices"]
    ]
    return {
        "schema_version": 1,
        "experiment": _PROTOCOL_EXPERIMENT,
        "status": _PROTOCOL_STATUS,
        **headwise._base_bindings(context),
        **dict(prerequisite_hashes),
        "causal_gate_source_sha256": source_sha256,
        "causal_gate_source_inventory_sha256": dict(source_inventory),
        "framework_contract": dict(framework_contract),
        "training_attention_library_path": str(attention_library_path),
        "training_attention_library_sha256": attention_library_sha256,
        "source_revision": context["sustained_protocol"]["source_revision"],
        "source_config_sha256": context["sustained_protocol"]["source_config_sha256"],
        "source_index_sha256": context["sustained_protocol"]["source_index_sha256"],
        "source_shard_sha256": context["sustained_protocol"]["source_shard_sha256"],
        "record_split": context["split"],
        "record_split_identity": context["split"]["split_identity"],
        "training_data_access": split,
        "selection_input_identity": sha256_json(selection_inputs),
        "model": context["model"],
        "training": {
            "device": device,
            "threads": threads,
            "dtype": "bfloat16",
            "batch_size": 1,
            "iht_steps": _IHT_STEPS,
            "gradient_sequence_indices_per_step": list(_SELECTION_SEQUENCE_INDICES),
            "terminal_evaluation_mask": "M2",
            "terminal_evaluation_sequence_indices": list(_SELECTION_SEQUENCE_INDICES),
            "backward_passes": 4,
            "terminal_forward_only_passes": 2,
            "seed": _SEED,
            "eval": True,
            "dropout": False,
            "teacher_weights_frozen": True,
            "attention_implementation": (
                "native_exact_forward_plus_batched_differentiable_surrogate"
            ),
            "native_oracle_device": "cpu",
            "native_identity_schedule": (
                "detached_float32_QK_plus_token_identity_values_D128"
            ),
            "native_sparse_exact_forward": True,
            "native_W128_exact_forward": True,
            "gate_location": "per_layer_head_pre_output_projection",
            "gate_shape": [_LAYERS, _HEADS],
            "initial_gate_mask": "all_base_zero",
            "projected_gradient_step": _PROJECTED_GRADIENT_STEP,
            "gradient_normalization": (
                "divide_by_global_float64_root_mean_square_plus_1e-12"
            ),
            "projection": ("descending_projected_score_then_ascending_flat_index"),
            "hard_head_count_after_each_IHT_step": _RESCUED_HEADS,
            "discrete_cache_decisions": (
                "exact_existing_native_streaming_attention_DSO"
            ),
            "qkv_attention_arithmetic": (
                "float32_sparse_and_full_context_then_cast_to_projection_dtype"
            ),
            "use_cache": False,
            "output_hidden_states": True,
            "full_vocabulary_logits": True,
            "candidate_masks_executed": ["M1", "M2"],
            "selection_rule": [
                "lower_maximum_per_record_composite_objective",
                "lower_mean_per_record_composite_objective",
                "M1_on_exact_tie",
            ],
            "screen_eligibility": (
                "selected maximum and mean objectives both strictly improve "
                "over executed M0 and no selected record regresses from M0"
            ),
        },
        "objective": {
            "bands": [
                {"name": name, "start": start, "stop": stop}
                for name, start, stop in _TRAINING_BANDS
            ],
            **_LOSS_CONTRACT,
        },
        "base_attention_policy": dict(_BASE_POLICY),
        "rescue_attention_policy": dict(_FULL_POLICY),
        "budget_contract": headwise._headwise_budget_contract(context["model"]),
        "decision_rule": {
            "training_evidence_failure": (
                "stop without writing a promotable head mask"
            ),
            "training_success": (
                "freeze exactly one native six-record development screen"
            ),
            "no_robust_M0_improvement": ("stop without freezing a native screen"),
            "native_screen_failure": (
                "close this frozen static causal-gate attempt without "
                "opening confirmation"
            ),
            "native_screen_pass": (
                "freeze separately sealed package-native confirmation"
            ),
        },
        "provenance": {
            "protocol_frozen_before_gradients": True,
            "attention_mass_mask_failure_authenticated": True,
            "selection_records_previously_consumed_development": True,
            "internal_screen_records_previously_consumed_by_diagnostics": True,
            "internal_screen_records_are_not_an_unseen_holdout": True,
            "internal_screen_tensors_prohibited_during_training": True,
            "confirmation_corpus_unopened": True,
        },
        "limitations": [
            "Only two development-selection records contribute gradients.",
            "The six-record native screen is previously consumed development data, not an unseen holdout.",
            "The existing native DSO fixes exact float32 hard cache decisions; the gathered surrogate does not differentiate through top-k or victim changes.",
            "The exact native attention forward plus differentiable surrogate is an attribution proxy: BF16 Hugging Face projections and dense MLPs still differ from native float32 projections and packaged Q7.",
            "Training uses the untouched BF16 dense MoE weights; the final causal screen alone measures the complete packaged Q7 path.",
            "W128 is full context only for this 128-position experiment.",
            "A development pass is not sufficient for package promotion.",
        ],
    }


def _validate_protocol_shape(protocol: Mapping[str, Any]) -> None:
    access = protocol.get("training_data_access")
    training = protocol.get("training")
    objective = protocol.get("objective")
    budget = protocol.get("budget_contract")
    if (
        protocol.get("schema_version") != 1
        or protocol.get("experiment") != _PROTOCOL_EXPERIMENT
        or protocol.get("status") != _PROTOCOL_STATUS
        or not isinstance(access, Mapping)
        or access.get("selection_sequence_indices") != list(_SELECTION_SEQUENCE_INDICES)
        or access.get("gradient_sequence_indices_per_step")
        != list(_SELECTION_SEQUENCE_INDICES)
        or access.get("iht_steps") != _IHT_STEPS
        or access.get("terminal_evaluation_sequence_indices")
        != list(_SELECTION_SEQUENCE_INDICES)
        or access.get("internal_screen_sequence_order")
        != list(_INTERNAL_SCREEN_SEQUENCE_ORDER)
        or access.get("prohibited_internal_screen_sequence_indices")
        != list(_INTERNAL_SCREEN_SEQUENCE_INDICES)
        or access.get("internal_screen_records_used") is not False
        or not isinstance(training, Mapping)
        or training.get("iht_steps") != _IHT_STEPS
        or training.get("gradient_sequence_indices_per_step")
        != list(_SELECTION_SEQUENCE_INDICES)
        or training.get("terminal_evaluation_sequence_indices")
        != list(_SELECTION_SEQUENCE_INDICES)
        or training.get("backward_passes") != 4
        or training.get("terminal_forward_only_passes") != 2
        or training.get("teacher_weights_frozen") is not True
        or training.get("device") != "cpu"
        or training.get("native_oracle_device") != "cpu"
        or training.get("hard_head_count_after_each_IHT_step") != _RESCUED_HEADS
        or not isinstance(objective, Mapping)
        or objective.get("bands")
        != [
            {"name": name, "start": start, "stop": stop}
            for name, start, stop in _TRAINING_BANDS
        ]
        or not isinstance(budget, Mapping)
        or budget.get("rescued_heads") != _RESCUED_HEADS
        or budget.get("next_head_boundary", {}).get("within_budget") is not False
        or not headwise._is_sha256(protocol.get("training_attention_library_sha256"))
        or not isinstance(
            protocol.get("training_attention_library_path"),
            str,
        )
    ):
        raise ValueError("causal head-gate protocol shape is invalid")
    if set(access["selection_sequence_indices"]).intersection(
        access["prohibited_internal_screen_sequence_indices"]
    ):
        raise ValueError("causal head-gate protocol leaks screen records")


def _validate_gate_protocol(
    protocol: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    protocol_sha256: str,
    supplied_sha256: str,
    prerequisite_hashes: Mapping[str, str],
    source_sha256: str,
    source_inventory: Mapping[str, str],
    framework_contract: Mapping[str, str],
    attention_library_path: Path,
    attention_library_sha256: str,
) -> None:
    _validate_protocol_shape(protocol)
    training = protocol["training"]
    expected = _build_gate_protocol(
        context,
        prerequisite_hashes=prerequisite_hashes,
        source_sha256=source_sha256,
        source_inventory=source_inventory,
        framework_contract=framework_contract,
        attention_library_path=attention_library_path,
        attention_library_sha256=attention_library_sha256,
        device=str(training.get("device")),
        threads=int(training.get("threads", 0)),
    )
    if (
        protocol_sha256 != str(supplied_sha256).lower()
        or protocol != expected
        or training["device"] != "cpu"
        or training["threads"] != _THREADS
    ):
        raise ValueError("causal head-gate frozen protocol is invalid")


def freeze_native_olmoe_causal_head_gate_protocol(
    *,
    out: str | Path,
    attention_library: str | Path,
    attention_library_sha256: str,
    device: str = "cpu",
    threads: int = _THREADS,
    manifest_sha256: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Freeze all data, code, framework, objective, and update decisions."""

    output_path = Path(out).expanduser().resolve()
    if output_path.exists():
        raise ValueError("causal head-gate protocol target already exists")
    if device != "cpu" or threads != _THREADS:
        raise ValueError("causal head-gate training configuration is invalid")
    attention_path, attention_hash = _authenticate_attention_library(
        attention_library,
        attention_library_sha256,
    )
    if not _PREREQUISITE_ARGUMENT_NAMES.issubset(kwargs):
        raise ValueError("causal head-gate prerequisite arguments are incomplete")
    artifact_kwargs = {name: kwargs.pop(name) for name in _PREREQUISITE_ARGUMENT_NAMES}
    _paths, context = headwise._common_context(
        manifest_sha256=manifest_sha256,
        **kwargs,
    )
    prerequisite_hashes, _boundary = _authenticate_failed_headwise_boundary(
        context,
        **artifact_kwargs,
    )
    source_hash = sha256_file(Path(__file__).resolve())
    source_inventory = _current_source_inventory(context)
    framework = _framework_contract()
    protocol = _build_gate_protocol(
        context,
        prerequisite_hashes=prerequisite_hashes,
        source_sha256=source_hash,
        source_inventory=source_inventory,
        framework_contract=framework,
        attention_library_path=attention_path,
        attention_library_sha256=attention_hash,
        device=device,
        threads=threads,
    )
    _validate_protocol_shape(protocol)
    atomic_json(output_path, protocol)
    return protocol


def _gated_attention_forward(
    module: Any,
    hidden_states: Any,
    position_embeddings: tuple[Any, Any],
    attention_mask: Any | None,
    past_key_values: Any | None = None,
    **kwargs: Any,
) -> tuple[Any, None]:
    """Evaluator-only OLMoE attention with exact sparse/full hard gates."""

    torch = _require_torch()
    state = getattr(module, "_engram_causal_gate_state", None)
    position_ids = kwargs.get("position_ids")
    use_cache = kwargs.get("use_cache", False)
    if (
        not isinstance(state, MutableMapping)
        or past_key_values is not None
        or position_embeddings is None
        or use_cache not in {False, None}
        or hidden_states.ndim != 3
        or hidden_states.shape[0] != 1
        or hidden_states.shape[1] != _POSITIONS_PER_SEQUENCE
    ):
        raise ValueError("causal gate attention invocation is invalid")
    canonical_positions = torch.arange(
        _POSITIONS_PER_SEQUENCE,
        dtype=torch.long,
        device=hidden_states.device,
    ).reshape(1, -1)
    if (
        not isinstance(position_ids, torch.Tensor)
        or position_ids.shape != canonical_positions.shape
        or not torch.equal(position_ids, canonical_positions)
    ):
        raise ValueError("causal gate requires canonical positions 0..127")
    if (
        not isinstance(attention_mask, torch.Tensor)
        or attention_mask.shape[-2:]
        != (_POSITIONS_PER_SEQUENCE, _POSITIONS_PER_SEQUENCE)
        or attention_mask.shape[0] != 1
    ):
        raise ValueError("causal gate requires a canonical unpadded causal mask")
    mask = attention_mask.detach().float()
    lower = torch.tril(torch.ones_like(mask, dtype=torch.bool))
    upper = ~lower
    if not bool((mask[lower] == 0).all()) or not bool((mask[upper] < -1.0e20).all()):
        raise ValueError("causal gate attention mask contains padding or leakage")
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query_states = module.q_norm(module.q_proj(hidden_states))
    key_states = module.k_norm(module.k_proj(hidden_states))
    value_states = module.v_proj(hidden_states)
    if module.config.clip_qkv is not None:
        clip = float(module.config.clip_qkv)
        query_states = query_states.clamp(min=-clip, max=clip)
        key_states = key_states.clamp(min=-clip, max=clip)
        value_states = value_states.clamp(min=-clip, max=clip)
    query_states = query_states.view(*hidden_shape).transpose(1, 2)
    key_states = key_states.view(*hidden_shape).transpose(1, 2)
    value_states = value_states.view(*hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = state["apply_rotary_pos_emb"](
        query_states,
        key_states,
        cos,
        sin,
    )
    if (
        query_states.shape[1] != _HEADS
        or key_states.shape != query_states.shape
        or value_states.shape != query_states.shape
        or query_states.shape[2] != _POSITIONS_PER_SEQUENCE
    ):
        raise ValueError("causal gate OLMoE Q/K/V shape is invalid")

    # Native attention consumes float32 projected Q/K/V.  Preserve that
    # arithmetic for both branches, then cast the selected context back to the
    # BF16 output-projection input type.
    query_f32 = query_states.float()
    key_f32 = key_states.float()
    value_f32 = value_states.float()
    sparse, sparse_diagnostics = _native_sparse_straight_through(
        query_f32,
        key_f32,
        value_f32,
        attention_library=state["attention_library"],
        policy=_BASE_POLICY,
    )
    full, full_diagnostics = _native_full_straight_through(
        query_f32,
        key_f32,
        value_f32,
        attention_library=state["attention_library"],
    )
    gates = state["gates"][int(module.layer_idx)]
    mixed = _mix_head_outputs(sparse, full, gates).to(query_states.dtype)
    state["diagnostics"].append(
        {
            "layer": int(module.layer_idx),
            "sparse": sparse_diagnostics,
            "full": full_diagnostics,
        }
    )
    mixed = mixed.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    return module.o_proj(mixed), None


def _install_causal_gate_attention(
    loaded: Any,
    gate_state: MutableMapping[str, Any],
) -> list[tuple[Any, Any]]:
    _prepare_transformers_imports()
    try:
        from transformers.models.olmoe.modeling_olmoe import (
            apply_rotary_pos_emb,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "installed Transformers has no OLMoE implementation"
        ) from exc
    layers = list(loaded.model.layers)
    if len(layers) != _LAYERS:
        raise ValueError("causal gate requires exactly 16 OLMoE layers")
    gate_state["apply_rotary_pos_emb"] = apply_rotary_pos_emb
    originals: list[tuple[Any, Any]] = []
    for layer_index, decoder_layer in enumerate(layers):
        attention = decoder_layer.self_attn
        if int(attention.layer_idx) != layer_index:
            raise ValueError("causal gate OLMoE layer index is invalid")
        originals.append((attention, attention.forward))
        attention._engram_causal_gate_state = gate_state
        attention.forward = types.MethodType(
            _gated_attention_forward,
            attention,
        )
    return originals


def _restore_causal_gate_attention(
    originals: Sequence[tuple[Any, Any]],
) -> None:
    for attention, original in originals:
        attention.forward = original
        if hasattr(attention, "_engram_causal_gate_state"):
            delattr(attention, "_engram_causal_gate_state")


def _component_scalars(components: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in components.items():
        if isinstance(value, Mapping):
            result[name] = {
                child_name: float(child.detach().float().item())
                for child_name, child in value.items()
            }
        else:
            result[name] = float(value.detach().float().item())
    return result


def _selected_head_rows(
    mask: np.ndarray,
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    if (
        mask.shape != (_LAYERS, _HEADS)
        or mask.dtype != np.bool_
        or int(mask.sum()) != _RESCUED_HEADS
        or scores.shape != mask.shape
        or not np.isfinite(scores).all()
    ):
        raise ValueError("causal gate final mask/score shape is invalid")
    flat_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    selected = np.flatnonzero(mask.reshape(-1))
    order = sorted(
        (int(index) for index in selected),
        key=lambda index: (-float(flat_scores[index]), index),
    )
    return [
        {
            "rank": rank,
            "layer": flat_index // _HEADS,
            "head": flat_index % _HEADS,
            "layer_major_index": flat_index,
            "projected_score": float(flat_scores[flat_index]),
        }
        for rank, flat_index in enumerate(order, start=1)
    ]


def _objective_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    if len(records) != len(_SELECTION_SEQUENCE_INDICES) or [
        int(row.get("sequence_index", -1)) for row in records
    ] != list(_SELECTION_SEQUENCE_INDICES):
        raise ValueError("causal gate objective records are invalid")
    values = np.asarray(
        [float(row["loss"]["total"]) for row in records],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("causal gate objective is non-finite")
    return {
        "maximum_per_record_composite_objective": float(values.max()),
        "mean_per_record_composite_objective": float(values.mean()),
    }


def _select_executed_mask(
    evaluations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(evaluations) != set(_MASK_NAMES):
        raise ValueError("causal gate executed-mask population is invalid")
    summaries = {
        name: _objective_summary(evaluations[name]["records"]) for name in _MASK_NAMES
    }
    selected = min(
        ("M1", "M2"),
        key=lambda name: (
            summaries[name]["maximum_per_record_composite_objective"],
            summaries[name]["mean_per_record_composite_objective"],
            0 if name == "M1" else 1,
        ),
    )
    baseline = summaries["M0"]
    chosen = summaries[selected]
    baseline_by_sequence = {
        int(row["sequence_index"]): float(row["loss"]["total"])
        for row in evaluations["M0"]["records"]
    }
    selected_deltas = [
        {
            "sequence_index": int(row["sequence_index"]),
            "selected_minus_M0_composite_objective": (
                float(row["loss"]["total"])
                - baseline_by_sequence[int(row["sequence_index"])]
            ),
            "regressed": (
                float(row["loss"]["total"])
                > baseline_by_sequence[int(row["sequence_index"])]
            ),
        }
        for row in evaluations[selected]["records"]
    ]
    eligible = (
        chosen["maximum_per_record_composite_objective"]
        < baseline["maximum_per_record_composite_objective"]
        and chosen["mean_per_record_composite_objective"]
        < baseline["mean_per_record_composite_objective"]
        and not any(row["regressed"] for row in selected_deltas)
    )
    return {
        "selected_mask_name": selected,
        "summaries": summaries,
        "screen_eligible": eligible,
        "per_record_deltas": selected_deltas,
        "selection_key": [
            chosen["maximum_per_record_composite_objective"],
            chosen["mean_per_record_composite_objective"],
            0 if selected == "M1" else 1,
        ],
    }


def _diagnostic_timing_summary(
    layers: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    if len(layers) != _LAYERS or [int(row.get("layer", -1)) for row in layers] != list(
        range(_LAYERS)
    ):
        raise ValueError("causal gate native-oracle layer diagnostics are invalid")
    return {
        "layers": len(layers),
        "native_identity_schedule_seconds": float(
            sum(float(row["sparse"]["schedule"]["elapsed_seconds"]) for row in layers)
        ),
        "native_sparse_actual_value_seconds": float(
            sum(
                float(row["sparse"]["actual_value"]["elapsed_seconds"])
                for row in layers
            )
        ),
        "native_full_actual_value_seconds": float(
            sum(float(row["full"]["actual_value"]["elapsed_seconds"]) for row in layers)
        ),
        "sparse_surrogate_seconds": float(
            sum(float(row["sparse"]["surrogate_elapsed_seconds"]) for row in layers)
        ),
        "full_surrogate_seconds": float(
            sum(float(row["full"]["surrogate_elapsed_seconds"]) for row in layers)
        ),
    }


def _run_gate_record(
    loaded: Any,
    gate_state: MutableMapping[str, Any],
    *,
    mask: np.ndarray,
    sequence_index: int,
    context: Mapping[str, Any],
    teacher_logits: np.ndarray,
    teacher_hidden: np.ndarray,
    targets: np.ndarray,
    bands: Sequence[Mapping[str, Any]],
    backward: bool,
) -> dict[str, Any]:
    torch = _require_torch()
    if (
        mask.shape != (_LAYERS, _HEADS)
        or mask.dtype != np.bool_
        or sequence_index not in _SELECTION_SEQUENCE_INDICES
    ):
        raise ValueError("causal gate record execution request is invalid")
    gates = torch.tensor(
        mask.astype(np.float32),
        dtype=torch.float32,
        device="cpu",
        requires_grad=backward,
    )
    gate_state["gates"] = gates
    gate_state["diagnostics"] = []
    tokens = torch.tensor(
        [context["input_ids"][sequence_index][:-1]],
        dtype=torch.long,
        device="cpu",
    )
    teacher_logits_tensor = torch.as_tensor(
        teacher_logits,
        dtype=torch.float32,
        device="cpu",
    ).unsqueeze(0)
    teacher_hidden_tensor = torch.as_tensor(
        teacher_hidden,
        dtype=torch.float32,
        device="cpu",
    ).unsqueeze(0)
    target_tensor = torch.as_tensor(
        targets,
        dtype=torch.long,
        device="cpu",
    ).unsqueeze(0)
    loaded.zero_grad(set_to_none=True)
    started = time.perf_counter()
    context_manager = torch.enable_grad() if backward else torch.no_grad()
    with context_manager:
        output = loaded(
            input_ids=tokens,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        student_logits = output.logits.float()
        student_hidden = output.hidden_states[-1].float()
        if (
            student_logits.shape != teacher_logits_tensor.shape
            or student_hidden.shape != teacher_hidden_tensor.shape
        ):
            raise ValueError("causal gate student/teacher shape differs")
        loss, components = _distillation_loss(
            student_logits,
            teacher_logits_tensor,
            student_hidden,
            teacher_hidden_tensor,
            target_tensor,
            bands=bands,
        )
        if backward:
            loss.backward()
    gradient: np.ndarray | None = None
    if backward:
        if gates.grad is None:
            raise ValueError("causal gate produced no gate gradient")
        gradient = (
            gates.grad.detach()
            .float()
            .cpu()
            .numpy()
            .astype(
                np.float64,
                copy=True,
            )
        )
        if gradient.shape != (_LAYERS, _HEADS) or not np.isfinite(gradient).all():
            raise ValueError("causal gate gradient is invalid")
    layer_diagnostics = list(gate_state["diagnostics"])
    timing = _diagnostic_timing_summary(layer_diagnostics)
    result = {
        "sequence_index": sequence_index,
        "record_id": context["record_ids"][sequence_index],
        "mask_sha256": sha256_json(mask.tolist()),
        "selected_head_count": int(mask.sum()),
        "loss": _component_scalars(components),
        "backward": backward,
        "gradient": None if gradient is None else gradient.tolist(),
        "native_oracle_layers": layer_diagnostics,
        "native_oracle_timing": timing,
        "elapsed_seconds": time.perf_counter() - started,
    }
    del (
        gates,
        tokens,
        teacher_logits_tensor,
        teacher_hidden_tensor,
        target_tensor,
        output,
        student_logits,
        student_hidden,
        loss,
        components,
    )
    gc.collect()
    return result


def _training_post_authentication(
    context: Mapping[str, Any],
    common_paths: Mapping[str, Path],
    *,
    manifest_sha256: str,
    source_inventory: Mapping[str, str],
    prerequisite_paths: Mapping[str, Path],
    prerequisite_hashes: Mapping[str, str],
    protocol_path: Path,
    protocol_sha256: str,
    framework_contract: Mapping[str, str],
    attention_library: Path,
    attention_library_sha256: str,
) -> dict[str, bool]:
    checks = headwise._common_post_authentication(
        context,
        common_paths,
        manifest_sha256=manifest_sha256,
        source_inventory=source_inventory,
    )
    for name, path in prerequisite_paths.items():
        checks[name] = sha256_file(path) == prerequisite_hashes[f"{name}_sha256"]
    checks.update(
        {
            "gate_protocol": sha256_file(protocol_path) == protocol_sha256,
            "causal_gate_source": (
                sha256_file(Path(__file__).resolve())
                == source_inventory["src/engram/evaluation/olmoe_causal_head_gate.py"]
            ),
            "framework_contract": _framework_contract() == dict(framework_contract),
            "training_attention_library": (
                sha256_file(attention_library) == attention_library_sha256
            ),
        }
    )
    return checks


def train_native_olmoe_causal_head_gate(
    *,
    gate_protocol: str | Path,
    gate_protocol_sha256: str,
    attention_library: str | Path,
    attention_library_sha256: str,
    out: str | Path,
    manifest_sha256: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run two full-selection-gradient IHT steps and terminal evaluation."""

    output_path = Path(out).expanduser().resolve()
    protocol_path = Path(gate_protocol).expanduser().resolve()
    if output_path.exists():
        raise ValueError("causal head-gate training target already exists")
    if not _PREREQUISITE_ARGUMENT_NAMES.issubset(kwargs):
        raise ValueError("causal head-gate prerequisite arguments are incomplete")
    attention_path, attention_hash = _authenticate_attention_library(
        attention_library,
        attention_library_sha256,
    )
    artifact_kwargs = {name: kwargs.pop(name) for name in _PREREQUISITE_ARGUMENT_NAMES}
    common_paths, context = headwise._common_context(
        manifest_sha256=manifest_sha256,
        **kwargs,
    )
    prerequisite_hashes, boundary = _authenticate_failed_headwise_boundary(
        context,
        **artifact_kwargs,
    )
    protocol_hash = _require_digest(
        protocol_path,
        gate_protocol_sha256,
        "causal head-gate protocol",
    )
    protocol = _read_json(protocol_path, "causal head-gate protocol")
    source_hash = sha256_file(Path(__file__).resolve())
    source_inventory = _current_source_inventory(context)
    framework = _framework_contract()
    _validate_gate_protocol(
        protocol,
        context,
        protocol_sha256=protocol_hash,
        supplied_sha256=gate_protocol_sha256,
        prerequisite_hashes=prerequisite_hashes,
        source_sha256=source_hash,
        source_inventory=source_inventory,
        framework_contract=framework,
        attention_library_path=attention_path,
        attention_library_sha256=attention_hash,
    )
    torch = _require_torch()
    device = str(protocol["training"]["device"])
    if device != "cpu":
        raise ValueError("causal gate native-oracle training is CPU-only")
    model_path = Path(context["reference"]["source"]["model"]).resolve()
    audit = audit_olmoe_source(model_path)
    if (
        audit.decision != "proceed_to_router_trace"
        or audit.resolved_revision != protocol["source_revision"]
        or audit.config_sha256 != protocol["source_config_sha256"]
        or audit.index_sha256 != protocol["source_index_sha256"]
    ):
        raise ValueError("causal gate dense teacher source changed")
    teacher_logits_np, teacher_hidden_np, targets_np = headwise._load_teacher_arrays(
        context,
        common_paths["arrays_path"],
    )
    allowed_indices = protocol["training_data_access"]["selection_sequence_indices"]
    teacher_logits_by_sequence = _selection_slices_only(
        teacher_logits_np,
        allowed_indices,
    )
    teacher_hidden_by_sequence = _selection_slices_only(
        teacher_hidden_np,
        allowed_indices,
    )
    targets_by_sequence = _selection_slices_only(
        targets_np,
        allowed_indices,
    )
    del teacher_logits_np, teacher_hidden_np, targets_np

    _prepare_transformers_imports()
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for causal head-gate training"
        ) from exc
    threads_before = torch.get_num_threads()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    loaded: Any | None = None
    originals: list[tuple[Any, Any]] = []
    gate_state: MutableMapping[str, Any] = {
        "attention_library": attention_path,
        "diagnostics": [],
    }
    masks: dict[str, np.ndarray] = {"M0": np.zeros((_LAYERS, _HEADS), dtype=np.bool_)}
    scores_by_mask: dict[str, np.ndarray] = {}
    step_results: list[dict[str, Any]] = []
    evaluations: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    try:
        torch.set_num_threads(int(protocol["training"]["threads"]))
        random.seed(_SEED)
        np.random.seed(_SEED)
        torch.manual_seed(_SEED)
        torch.use_deterministic_algorithms(True)
        loaded = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).eval()
        loaded.to("cpu")
        loaded.requires_grad_(False)
        if any(parameter.requires_grad for parameter in loaded.parameters()):
            raise ValueError("causal gate failed to freeze dense teacher weights")
        originals = _install_causal_gate_attention(loaded, gate_state)
        for step in range(1, _IHT_STEPS + 1):
            input_name = f"M{step - 1}"
            output_name = f"M{step}"
            input_mask = masks[input_name]
            record_results = [
                _run_gate_record(
                    loaded,
                    gate_state,
                    mask=input_mask,
                    sequence_index=sequence_index,
                    context=context,
                    teacher_logits=teacher_logits_by_sequence[sequence_index],
                    teacher_hidden=teacher_hidden_by_sequence[sequence_index],
                    targets=targets_by_sequence[sequence_index],
                    bands=protocol["objective"]["bands"],
                    backward=True,
                )
                for sequence_index in _SELECTION_SEQUENCE_INDICES
            ]
            gradients = np.stack(
                [
                    np.asarray(row["gradient"], dtype=np.float64)
                    for row in record_results
                ],
                axis=0,
            )
            mean_gradient = gradients.mean(axis=0, dtype=np.float64)
            projected_scores, output_mask, gradient_rms = _projected_gate_step(
                input_mask, mean_gradient
            )
            masks[output_name] = output_mask
            scores_by_mask[output_name] = projected_scores
            evaluations[input_name] = {
                "mask_name": input_name,
                "mask": input_mask.tolist(),
                "mask_sha256": sha256_json(input_mask.tolist()),
                "records": record_results,
                "objective_summary": _objective_summary(record_results),
                "execution_role": "gradient_and_candidate_evaluation",
            }
            step_results.append(
                {
                    "step": step,
                    "input_mask_name": input_name,
                    "output_mask_name": output_name,
                    "input_mask": input_mask.tolist(),
                    "input_mask_sha256": sha256_json(input_mask.tolist()),
                    "output_mask": output_mask.tolist(),
                    "output_mask_sha256": sha256_json(output_mask.tolist()),
                    "record_gradients": [
                        {
                            "sequence_index": row["sequence_index"],
                            "gradient": row["gradient"],
                        }
                        for row in record_results
                    ],
                    "mean_gradient": mean_gradient.tolist(),
                    "mean_gradient_root_mean_square": gradient_rms,
                    "projected_score": projected_scores.tolist(),
                    "output_selected_flat_indices": np.flatnonzero(
                        output_mask.reshape(-1)
                    )
                    .astype(int)
                    .tolist(),
                    "head_churn_from_input": int(
                        np.count_nonzero(input_mask != output_mask)
                    ),
                }
            )
        terminal_records = [
            _run_gate_record(
                loaded,
                gate_state,
                mask=masks["M2"],
                sequence_index=sequence_index,
                context=context,
                teacher_logits=teacher_logits_by_sequence[sequence_index],
                teacher_hidden=teacher_hidden_by_sequence[sequence_index],
                targets=targets_by_sequence[sequence_index],
                bands=protocol["objective"]["bands"],
                backward=False,
            )
            for sequence_index in _SELECTION_SEQUENCE_INDICES
        ]
        evaluations["M2"] = {
            "mask_name": "M2",
            "mask": masks["M2"].tolist(),
            "mask_sha256": sha256_json(masks["M2"].tolist()),
            "records": terminal_records,
            "objective_summary": _objective_summary(terminal_records),
            "execution_role": "terminal_forward_only_candidate_evaluation",
        }
    finally:
        if originals:
            _restore_causal_gate_attention(originals)
        if loaded is not None:
            del loaded
        gc.collect()
        torch.use_deterministic_algorithms(deterministic_before)
        torch.set_num_threads(threads_before)
    elapsed_seconds = time.perf_counter() - started

    selection = _select_executed_mask(evaluations)
    selected_name = str(selection["selected_mask_name"])
    selected_mask = masks[selected_name]
    final_scores = scores_by_mask[selected_name]
    selected_rows = _selected_head_rows(selected_mask, final_scores)
    selected_pairs = [(int(row["layer"]), int(row["head"])) for row in selected_rows]
    # Re-run the analytical validator independently of projection.
    expectations = headwise._headwise_expectations(
        context["model"],
        selected_pairs,
    )
    budget = protocol["budget_contract"]
    prerequisite_paths = boundary["paths"]
    post = _training_post_authentication(
        context,
        common_paths,
        manifest_sha256=manifest_sha256,
        source_inventory=source_inventory,
        prerequisite_paths=prerequisite_paths,
        prerequisite_hashes=prerequisite_hashes,
        protocol_path=protocol_path,
        protocol_sha256=protocol_hash,
        framework_contract=framework,
        attention_library=attention_path,
        attention_library_sha256=attention_hash,
    )
    evidence_checks = {
        "exact_two_IHT_steps": len(step_results) == _IHT_STEPS,
        "all_three_masks_executed": set(evaluations) == set(_MASK_NAMES),
        "four_backward_passes": sum(
            int(row["backward"])
            for value in evaluations.values()
            for row in value["records"]
        )
        == 4,
        "two_terminal_forward_only_passes": all(
            row["backward"] is False for row in evaluations["M2"]["records"]
        ),
        "selection_records_only": all(
            row["sequence_index"] in _SELECTION_SEQUENCE_INDICES
            for value in evaluations.values()
            for row in value["records"]
        ),
        "no_internal_screen_record_access": (
            not {
                row["sequence_index"]
                for value in evaluations.values()
                for row in value["records"]
            }.intersection(_INTERNAL_SCREEN_SEQUENCE_INDICES)
        ),
        "exact_51_after_every_IHT_step": all(
            int(np.asarray(row["output_mask"], dtype=np.bool_).sum()) == _RESCUED_HEADS
            for row in step_results
        ),
        "selected_mask_exact_51": (
            len(selected_pairs) == _RESCUED_HEADS
            and len(set(selected_pairs)) == _RESCUED_HEADS
        ),
        "analytical_budget_exact": (
            expectations == budget["attention_expectations_per_sequence"]
            and expectations["attention_logical_read_fraction"]
            <= _THRESHOLDS["maximum_attention_logical_read_fraction"]
        ),
        "teacher_weights_frozen": True,
        "CPU_only_native_oracle": device == "cpu",
        "selected_mask_was_executed": selected_name in {"M1", "M2"},
        "robust_improvement_vs_M0": bool(selection["screen_eligible"]),
        "post_training_authentication": all(post.values()),
    }
    evidence_without_improvement = {
        name: value
        for name, value in evidence_checks.items()
        if name != "robust_improvement_vs_M0"
    }
    execution_evidence_passed = all(evidence_without_improvement.values())
    screen_eligible = execution_evidence_passed and bool(selection["screen_eligible"])
    report = {
        "schema_version": 1,
        "experiment": _TRAINING_EXPERIMENT,
        "status": (
            _TRAINING_STATUS
            if execution_evidence_passed
            else "causal_head_gate_training_invalid"
        ),
        "artifacts": {
            **headwise._base_bindings(context),
            **prerequisite_hashes,
            "gate_protocol_sha256": protocol_hash,
            "causal_gate_source_sha256": source_hash,
            "causal_gate_source_inventory_sha256": source_inventory,
            "training_attention_library_sha256": attention_hash,
        },
        "framework_contract": framework,
        "record_split": context["split"],
        "training_data_access": protocol["training_data_access"],
        "training": protocol["training"],
        "objective": protocol["objective"],
        "budget_contract": budget,
        "attention_expectations_per_sequence": expectations,
        "IHT_step_results": step_results,
        "executed_mask_evaluations": evaluations,
        "mask_selection": selection,
        "mask_churn": {
            "M0_to_M1": int(np.count_nonzero(masks["M0"] != masks["M1"])),
            "M1_to_M2": int(np.count_nonzero(masks["M1"] != masks["M2"])),
        },
        "selected_mask_name": selected_name,
        "selected_heads": selected_rows,
        "attention_head_mask": selected_mask.tolist(),
        "attention_head_mask_sha256": sha256_json(selected_mask.tolist()),
        "selected_head_count": len(selected_rows),
        "evidence_checks": evidence_checks,
        "evidence_passed": execution_evidence_passed,
        "native_screen_eligible": screen_eligible,
        "decision": (
            "freeze_exactly_one_native_internal_development_screen"
            if screen_eligible
            else (
                "stop_causal_gate_without_native_screen_no_robust_M0_improvement"
                if execution_evidence_passed
                else "stop_and_diagnose_causal_gate_training_evidence"
            )
        ),
        "post_training_authentication": post,
        "performance": {
            "elapsed_seconds": elapsed_seconds,
            "executed_record_seconds": {
                name: [row["elapsed_seconds"] for row in value["records"]]
                for name, value in evaluations.items()
            },
        },
        "limitations": protocol["limitations"],
    }
    if not execution_evidence_passed:
        raise ValueError("causal head-gate training evidence failed")
    atomic_json(output_path, report)
    return report


def _finite_scalar(value: Any, *, nonnegative: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        return False
    number = float(value)
    return math.isfinite(number) and (not nonnegative or number >= 0.0)


def _strict_boolean_mask(value: Any, label: str) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != _LAYERS
        or any(
            not isinstance(row, list)
            or len(row) != _HEADS
            or any(not isinstance(item, bool) for item in row)
            for row in value
        )
    ):
        raise ValueError(f"causal head-gate {label} is not a Boolean 16x16 mask")
    return np.asarray(value, dtype=np.bool_)


def _strict_finite_matrix(
    value: Any,
    *,
    rows: int,
    columns: int,
    label: str,
) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != rows
        or any(
            not isinstance(row, list)
            or len(row) != columns
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in row
            )
            for row in value
        )
    ):
        raise ValueError(
            f"causal head-gate {label} is not a finite {rows}x{columns} matrix"
        )
    return np.asarray(value, dtype=np.float64)


def _validate_stored_loss(
    loss: Any,
    objective: Mapping[str, Any],
) -> None:
    if not isinstance(loss, Mapping) or set(loss) != {
        *_LOSS_COMPONENT_NAMES,
        "total",
        "bands",
    }:
        raise ValueError("causal head-gate stored loss schema is invalid")
    bands = loss["bands"]
    objective_bands = objective.get("bands")
    if not isinstance(bands, Mapping) or not isinstance(objective_bands, list):
        raise ValueError("causal head-gate stored loss bands are invalid")
    band_names = [str(row.get("name")) for row in objective_bands]
    expected_band_keys = {
        f"{band_name}_{component}"
        for band_name in band_names
        for component in _LOSS_COMPONENT_NAMES
    }
    if set(bands) != expected_band_keys:
        raise ValueError("causal head-gate stored loss band schema is invalid")
    for name, value in bands.items():
        floor = -1.0e-5 if name.endswith("_kl") else 0.0
        if not _finite_scalar(value) or float(value) < floor:
            raise ValueError("causal head-gate stored loss band is invalid")
    for component in _LOSS_COMPONENT_NAMES:
        floor = -1.0e-5 if component == "kl" else 0.0
        if not _finite_scalar(loss.get(component)) or float(loss[component]) < floor:
            raise ValueError("causal head-gate stored loss component is invalid")
        recomputed = float(
            np.mean(
                [float(bands[f"{band_name}_{component}"]) for band_name in band_names],
                dtype=np.float64,
            )
        )
        if not math.isclose(
            float(loss[component]),
            recomputed,
            rel_tol=2.0e-5,
            abs_tol=2.0e-7,
        ):
            raise ValueError("causal head-gate stored loss bands do not reduce")
    if not _finite_scalar(loss.get("total")) or float(loss["total"]) < -1.0e-4:
        raise ValueError("causal head-gate stored total loss is invalid")
    contract_prefix = {
        "kl": "teacher_to_student_kl",
        "hidden_relative_l2": "hidden_relative_l2",
        "positive_nll_delta": "positive_target_nll_delta",
        "top1_margin_deficit": "teacher_top1_margin_deficit",
    }
    recomputed_total = sum(
        float(loss[component])
        / float(objective[f"{contract_prefix[component]}_normalizer"])
        * float(objective[f"{contract_prefix[component]}_weight"])
        for component in _LOSS_COMPONENT_NAMES
    )
    if not math.isclose(
        float(loss["total"]),
        recomputed_total,
        rel_tol=2.0e-5,
        abs_tol=2.0e-6,
    ):
        raise ValueError("causal head-gate stored total loss does not recompute")


def _native_metric_expectations(
    model: Mapping[str, Any],
    policy: Mapping[str, int],
) -> tuple[dict[str, int], tuple[int, int]]:
    one_layer = {name: int(value) for name, value in model.items()}
    one_layer["layers"] = 1
    analytical = _attention_expectations(
        one_layer,
        dict(policy),
        positions=_POSITIONS_PER_SEQUENCE,
    )
    heads = int(one_layer["query_heads"])
    local_window = int(policy["local_window"])
    older_candidates = int(policy["older_candidates"])
    fixed = {
        "tokens_seen": _POSITIONS_PER_SEQUENCE,
        "local_entries": min(_POSITIONS_PER_SEQUENCE, local_window),
        "active_older_entries": (
            heads
            * min(
                max(_POSITIONS_PER_SEQUENCE - local_window, 0),
                older_candidates,
            )
        ),
        "candidate_key_bytes": int(analytical["attention_candidate_key_bytes"]),
        "selected_value_bytes": int(analytical["attention_selected_value_bytes"]),
        "local_kv_bytes": int(analytical["attention_local_kv_bytes"]),
        "eviction_events": int(analytical["attention_eviction_events"]),
        "older_candidate_entries_scored": int(
            analytical["attention_older_candidate_entries_scored"]
        ),
        "older_selected_entries": int(analytical["attention_older_selected_entries"]),
        "sink_insertions": int(analytical["attention_sink_insertions"]),
        "state_bytes": int(analytical["attention_state_bytes"]),
        "scratch_bytes": int(analytical["attention_scratch_bytes"]),
    }
    heavy_range = (
        int(analytical["attention_heavy_hitter_updates_minimum"]),
        int(analytical["attention_heavy_hitter_updates_maximum"]),
    )
    return fixed, heavy_range


def _validate_native_metrics(
    metrics: Any,
    *,
    model: Mapping[str, Any],
    policy: Mapping[str, int],
) -> None:
    if (
        not isinstance(metrics, Mapping)
        or set(metrics) != set(_NATIVE_METRIC_NAMES)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in metrics.values()
        )
    ):
        raise ValueError("causal head-gate native metric schema is invalid")
    fixed, heavy_range = _native_metric_expectations(model, policy)
    if any(metrics[name] != expected for name, expected in fixed.items()):
        raise ValueError("causal head-gate native metric contract changed")
    if not (heavy_range[0] <= int(metrics["heavy_hitter_updates"]) <= heavy_range[1]):
        raise ValueError("causal head-gate native heavy-hitter count is invalid")


def _validate_native_layer_evidence(
    layers: Any,
    *,
    protocol: Mapping[str, Any],
) -> dict[str, float | int]:
    if (
        not isinstance(layers, list)
        or len(layers) != _LAYERS
        or any(
            not isinstance(row, Mapping)
            or isinstance(row.get("layer"), bool)
            or not isinstance(row.get("layer"), int)
            or row.get("layer") != layer_index
            for layer_index, row in enumerate(layers)
        )
    ):
        raise ValueError("causal head-gate native layer order is invalid")
    library_hash = protocol["training_attention_library_sha256"]
    expected_counts = _expected_visible_counts(
        _POSITIONS_PER_SEQUENCE,
        **{
            name: int(protocol["base_attention_policy"][name])
            for name in ("local_window", "older_candidates", "older_top_k")
        },
    ).tolist()
    for layer_index, layer in enumerate(layers):
        if not isinstance(layer, Mapping) or set(layer) != {
            "layer",
            "sparse",
            "full",
        }:
            raise ValueError("causal head-gate native layer schema is invalid")
        sparse = layer["sparse"]
        full = layer["full"]
        if (
            not isinstance(sparse, Mapping)
            or set(sparse)
            != {
                "mode",
                "schedule",
                "actual_value",
                "surrogate_elapsed_seconds",
                "total_elapsed_seconds",
                "exact_forward_sha256",
            }
            or sparse.get("mode")
            != "native_exact_sparse_forward_gathered_surrogate_backward"
            or not headwise._is_sha256(sparse.get("exact_forward_sha256"))
            or not _finite_scalar(
                sparse.get("surrogate_elapsed_seconds"),
            )
            or float(sparse["surrogate_elapsed_seconds"]) <= 0.0
            or not _finite_scalar(
                sparse.get("total_elapsed_seconds"),
            )
            or float(sparse["total_elapsed_seconds"]) <= 0.0
        ):
            raise ValueError("causal head-gate sparse native evidence is invalid")
        schedule = sparse["schedule"]
        actual = sparse["actual_value"]
        if (
            not isinstance(schedule, Mapping)
            or set(schedule)
            != {
                "expected_visible_counts",
                "observed_visible_count_minimum",
                "observed_visible_count_maximum",
                "maximum_row_sum_error",
                "minimum_positive_weight",
                "indices_sha256",
                "native_metrics",
                "elapsed_seconds",
                "attention_library_sha256",
            }
            or schedule.get("expected_visible_counts") != expected_counts
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in schedule.get("expected_visible_counts", [])
            )
            or isinstance(schedule.get("observed_visible_count_minimum"), bool)
            or not isinstance(
                schedule.get("observed_visible_count_minimum"),
                int,
            )
            or isinstance(schedule.get("observed_visible_count_maximum"), bool)
            or not isinstance(
                schedule.get("observed_visible_count_maximum"),
                int,
            )
            or schedule.get("observed_visible_count_minimum") != min(expected_counts)
            or schedule.get("observed_visible_count_maximum") != max(expected_counts)
            or not _finite_scalar(
                schedule.get("maximum_row_sum_error"),
                nonnegative=True,
            )
            or float(schedule["maximum_row_sum_error"]) > 2.0e-5
            or not _finite_scalar(
                schedule.get("minimum_positive_weight"),
                nonnegative=True,
            )
            or not (0.0 < float(schedule["minimum_positive_weight"]) <= 1.0)
            or not headwise._is_sha256(schedule.get("indices_sha256"))
            or not _finite_scalar(
                schedule.get("elapsed_seconds"),
            )
            or float(schedule["elapsed_seconds"]) <= 0.0
            or schedule.get("attention_library_sha256") != library_hash
        ):
            raise ValueError("causal head-gate native schedule evidence is invalid")
        if (
            not isinstance(actual, Mapping)
            or set(actual)
            != {
                "native_metrics",
                "elapsed_seconds",
                "attention_library_sha256",
            }
            or not _finite_scalar(
                actual.get("elapsed_seconds"),
            )
            or float(actual["elapsed_seconds"]) <= 0.0
            or actual.get("attention_library_sha256") != library_hash
        ):
            raise ValueError("causal head-gate sparse actual-value evidence is invalid")
        _validate_native_metrics(
            schedule["native_metrics"],
            model=protocol["model"],
            policy=protocol["base_attention_policy"],
        )
        _validate_native_metrics(
            actual["native_metrics"],
            model=protocol["model"],
            policy=protocol["base_attention_policy"],
        )
        if schedule["native_metrics"] != actual["native_metrics"]:
            raise ValueError("causal head-gate sparse native replays diverge")
        if float(sparse["total_elapsed_seconds"]) + 1.0e-9 < (
            float(schedule["elapsed_seconds"])
            + float(actual["elapsed_seconds"])
            + float(sparse["surrogate_elapsed_seconds"])
        ):
            raise ValueError("causal head-gate sparse timing is inconsistent")
        if (
            not isinstance(full, Mapping)
            or set(full)
            != {
                "mode",
                "actual_value",
                "surrogate_elapsed_seconds",
                "total_elapsed_seconds",
                "exact_forward_sha256",
            }
            or full.get("mode")
            != "native_exact_W128_forward_full_causal_surrogate_backward"
            or not headwise._is_sha256(full.get("exact_forward_sha256"))
            or not _finite_scalar(
                full.get("surrogate_elapsed_seconds"),
            )
            or float(full["surrogate_elapsed_seconds"]) <= 0.0
            or not _finite_scalar(
                full.get("total_elapsed_seconds"),
            )
            or float(full["total_elapsed_seconds"]) <= 0.0
        ):
            raise ValueError("causal head-gate full native evidence is invalid")
        full_actual = full["actual_value"]
        if (
            not isinstance(full_actual, Mapping)
            or set(full_actual)
            != {
                "native_metrics",
                "elapsed_seconds",
                "attention_library_sha256",
            }
            or not _finite_scalar(
                full_actual.get("elapsed_seconds"),
            )
            or float(full_actual["elapsed_seconds"]) <= 0.0
            or full_actual.get("attention_library_sha256") != library_hash
        ):
            raise ValueError("causal head-gate full actual-value evidence is invalid")
        _validate_native_metrics(
            full_actual["native_metrics"],
            model=protocol["model"],
            policy=protocol["rescue_attention_policy"],
        )
        if float(full["total_elapsed_seconds"]) + 1.0e-9 < (
            float(full_actual["elapsed_seconds"])
            + float(full["surrogate_elapsed_seconds"])
        ):
            raise ValueError("causal head-gate full timing is inconsistent")
    return _diagnostic_timing_summary(layers)


def _expected_training_artifacts(
    protocol: Mapping[str, Any],
    protocol_sha256: str,
) -> dict[str, Any]:
    names = [
        *_BASE_BINDING_NAMES,
        *(f"{name}_sha256" for name in _BOUNDARY_ARTIFACT_NAMES),
        "causal_gate_source_sha256",
        "causal_gate_source_inventory_sha256",
        "training_attention_library_sha256",
    ]
    expected = {name: protocol[name] for name in names}
    expected["gate_protocol_sha256"] = protocol_sha256
    return expected


def _validate_executed_record(
    record: Any,
    *,
    sequence_index: int,
    mask: np.ndarray,
    backward: bool,
    protocol: Mapping[str, Any],
) -> None:
    if not isinstance(record, Mapping) or set(record) != {
        "sequence_index",
        "record_id",
        "mask_sha256",
        "selected_head_count",
        "loss",
        "backward",
        "gradient",
        "native_oracle_layers",
        "native_oracle_timing",
        "elapsed_seconds",
    }:
        raise ValueError("causal head-gate executed-record schema is invalid")
    selection_rows = protocol["training_data_access"]["selection_records"]
    expected_ids = {
        int(row["sequence_index"]): str(row["record_id"]) for row in selection_rows
    }
    if (
        isinstance(record.get("sequence_index"), bool)
        or not isinstance(record.get("sequence_index"), int)
        or record.get("sequence_index") != sequence_index
        or record.get("record_id") != expected_ids.get(sequence_index)
        or record.get("mask_sha256") != sha256_json(mask.tolist())
        or isinstance(record.get("selected_head_count"), bool)
        or not isinstance(record.get("selected_head_count"), int)
        or record.get("selected_head_count") != int(mask.sum())
        or record.get("backward") is not backward
        or not _finite_scalar(record.get("elapsed_seconds"))
        or float(record["elapsed_seconds"]) <= 0.0
    ):
        raise ValueError("causal head-gate executed-record identity is invalid")
    _validate_stored_loss(record["loss"], protocol["objective"])
    gradient = record["gradient"]
    if backward:
        _strict_finite_matrix(
            gradient,
            rows=_LAYERS,
            columns=_HEADS,
            label="executed gradient",
        )
    elif gradient is not None:
        raise ValueError("causal head-gate terminal run contains gradients")
    timing = _validate_native_layer_evidence(
        record["native_oracle_layers"],
        protocol=protocol,
    )
    if record["native_oracle_timing"] != timing:
        raise ValueError("causal head-gate native timing summary is invalid")
    measured_native_seconds = sum(
        float(layer["sparse"]["total_elapsed_seconds"])
        + float(layer["full"]["total_elapsed_seconds"])
        for layer in record["native_oracle_layers"]
    )
    if float(record["elapsed_seconds"]) + 1.0e-9 < measured_native_seconds:
        raise ValueError("causal head-gate record timing is inconsistent")


def _validate_training_result(
    result: Mapping[str, Any],
    *,
    result_sha256: str,
    supplied_sha256: str,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
) -> list[tuple[int, int]]:
    if (
        not headwise._is_sha256(result_sha256)
        or not headwise._is_sha256(supplied_sha256)
        or not headwise._is_sha256(protocol_sha256)
        or result_sha256 != str(supplied_sha256).lower()
    ):
        raise ValueError("causal head-gate training result hash is invalid")
    if set(result) != {
        "schema_version",
        "experiment",
        "status",
        "artifacts",
        "framework_contract",
        "record_split",
        "training_data_access",
        "training",
        "objective",
        "budget_contract",
        "attention_expectations_per_sequence",
        "IHT_step_results",
        "executed_mask_evaluations",
        "mask_selection",
        "mask_churn",
        "selected_mask_name",
        "selected_heads",
        "attention_head_mask",
        "attention_head_mask_sha256",
        "selected_head_count",
        "evidence_checks",
        "evidence_passed",
        "native_screen_eligible",
        "decision",
        "post_training_authentication",
        "performance",
        "limitations",
    }:
        raise ValueError("causal head-gate training result schema is invalid")
    steps = result.get("IHT_step_results")
    evaluations = result.get("executed_mask_evaluations")
    if (
        not isinstance(steps, list)
        or len(steps) != _IHT_STEPS
        or not isinstance(evaluations, Mapping)
        or set(evaluations) != set(_MASK_NAMES)
    ):
        raise ValueError("causal head-gate IHT result population is invalid")
    masks: dict[str, np.ndarray] = {"M0": np.zeros((_LAYERS, _HEADS), dtype=np.bool_)}
    scores_by_mask: dict[str, np.ndarray] = {}
    for step_number, step in enumerate(steps, start=1):
        input_name = f"M{step_number - 1}"
        output_name = f"M{step_number}"
        if not isinstance(step, Mapping) or set(step) != {
            "step",
            "input_mask_name",
            "output_mask_name",
            "input_mask",
            "input_mask_sha256",
            "output_mask",
            "output_mask_sha256",
            "record_gradients",
            "mean_gradient",
            "mean_gradient_root_mean_square",
            "projected_score",
            "output_selected_flat_indices",
            "head_churn_from_input",
        }:
            raise ValueError("causal head-gate IHT step schema is invalid")
        input_mask = _strict_boolean_mask(
            step.get("input_mask"),
            "IHT input",
        )
        output_mask = _strict_boolean_mask(
            step.get("output_mask"),
            "IHT output",
        )
        record_gradients = step.get("record_gradients")
        if (
            isinstance(step.get("step"), bool)
            or not isinstance(step.get("step"), int)
            or step.get("step") != step_number
            or step.get("input_mask_name") != input_name
            or step.get("output_mask_name") != output_name
            or input_mask.shape != (_LAYERS, _HEADS)
            or not np.array_equal(input_mask, masks[input_name])
            or step.get("input_mask_sha256") != sha256_json(input_mask.tolist())
            or not isinstance(record_gradients, list)
            or any(
                not isinstance(row, Mapping)
                or set(row) != {"sequence_index", "gradient"}
                for row in record_gradients
            )
            or any(
                isinstance(row.get("sequence_index"), bool)
                or not isinstance(row.get("sequence_index"), int)
                for row in record_gradients
            )
            or [row.get("sequence_index") for row in record_gradients]
            != list(_SELECTION_SEQUENCE_INDICES)
        ):
            raise ValueError("causal head-gate IHT input chain is invalid")
        gradients = np.stack(
            [
                _strict_finite_matrix(
                    row.get("gradient"),
                    rows=_LAYERS,
                    columns=_HEADS,
                    label="stored gradient",
                )
                for row in record_gradients
            ],
            axis=0,
        )
        mean_gradient = gradients.mean(axis=0, dtype=np.float64)
        stored_mean = _strict_finite_matrix(
            step.get("mean_gradient"),
            rows=_LAYERS,
            columns=_HEADS,
            label="stored mean gradient",
        )
        expected_scores, expected_mask, expected_rms = _projected_gate_step(
            input_mask,
            mean_gradient,
        )
        stored_scores = _strict_finite_matrix(
            step.get("projected_score"),
            rows=_LAYERS,
            columns=_HEADS,
            label="stored projected score",
        )
        selected_indices = step.get("output_selected_flat_indices")
        if (
            not np.array_equal(stored_mean, mean_gradient)
            or not np.array_equal(stored_scores, expected_scores)
            or not np.array_equal(output_mask, expected_mask)
            or not _finite_scalar(step.get("mean_gradient_root_mean_square"))
            or step.get("mean_gradient_root_mean_square") != expected_rms
            or step.get("output_mask_sha256") != sha256_json(expected_mask.tolist())
            or not isinstance(selected_indices, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in selected_indices
            )
            or selected_indices
            != np.flatnonzero(expected_mask.reshape(-1)).astype(int).tolist()
            or isinstance(step.get("head_churn_from_input"), bool)
            or not isinstance(step.get("head_churn_from_input"), int)
            or step.get("head_churn_from_input")
            != int(np.count_nonzero(input_mask != expected_mask))
        ):
            raise ValueError("causal head-gate IHT projection chain is invalid")
        masks[output_name] = expected_mask
        scores_by_mask[output_name] = expected_scores
    for name in _MASK_NAMES:
        evaluation = evaluations[name]
        if not isinstance(evaluation, Mapping) or set(evaluation) != {
            "mask_name",
            "mask",
            "mask_sha256",
            "records",
            "objective_summary",
            "execution_role",
        }:
            raise ValueError("causal head-gate executed-mask schema is invalid")
        mask = _strict_boolean_mask(
            evaluation.get("mask"),
            f"{name} evaluation",
        )
        records = evaluation.get("records")
        expected_backward = name in {"M0", "M1"}
        expected_role = (
            "gradient_and_candidate_evaluation"
            if expected_backward
            else "terminal_forward_only_candidate_evaluation"
        )
        if (
            evaluation.get("mask_name") != name
            or mask.shape != (_LAYERS, _HEADS)
            or not np.array_equal(mask, masks[name])
            or evaluation.get("mask_sha256") != sha256_json(mask.tolist())
            or not isinstance(records, list)
            or any(not isinstance(row, Mapping) for row in records)
            or any(
                isinstance(row.get("sequence_index"), bool)
                or not isinstance(row.get("sequence_index"), int)
                for row in records
            )
            or [row.get("sequence_index") for row in records]
            != list(_SELECTION_SEQUENCE_INDICES)
            or evaluation.get("execution_role") != expected_role
            or evaluation.get("objective_summary") != _objective_summary(records)
        ):
            raise ValueError("causal head-gate executed-mask evidence is invalid")
        for sequence_index, record in zip(
            _SELECTION_SEQUENCE_INDICES,
            records,
            strict=True,
        ):
            _validate_executed_record(
                record,
                sequence_index=sequence_index,
                mask=mask,
                backward=expected_backward,
                protocol=protocol,
            )
        if expected_backward:
            step = steps[0 if name == "M0" else 1]
            by_sequence = {
                int(row["sequence_index"]): row["gradient"]
                for row in step["record_gradients"]
            }
            if any(
                row.get("gradient") != by_sequence[int(row["sequence_index"])]
                for row in records
            ):
                raise ValueError("causal head-gate gradient evidence diverges")
    recomputed_selection = _select_executed_mask(evaluations)
    selected_name = recomputed_selection["selected_mask_name"]
    selected_mask = masks[selected_name]
    expected_rows = _selected_head_rows(
        selected_mask,
        scores_by_mask[selected_name],
    )
    selected_rows = result.get("selected_heads")
    if not isinstance(selected_rows, list) or any(
        not isinstance(row, Mapping)
        or set(row)
        != {
            "rank",
            "layer",
            "head",
            "layer_major_index",
            "projected_score",
        }
        or isinstance(row.get("projected_score"), bool)
        or not isinstance(row.get("projected_score"), float)
        or not math.isfinite(row["projected_score"])
        for row in selected_rows
    ):
        raise ValueError("causal head-gate selected-head evidence is invalid")
    selected = headwise._selected_pairs(result)
    evidence = result.get("evidence_checks")
    post = result.get("post_training_authentication")
    artifacts = result.get("artifacts")
    expected_post_names = set(_TRAINING_POST_AUTHENTICATION_NAMES)
    post_valid = (
        isinstance(post, Mapping)
        and set(post) == expected_post_names
        and all(value is True for value in post.values())
    )
    expected_expectations = protocol["budget_contract"][
        "attention_expectations_per_sequence"
    ]
    actual_expectations = result.get("attention_expectations_per_sequence")
    selected_pairs_unique = (
        len(selected) == _RESCUED_HEADS and len(set(selected)) == _RESCUED_HEADS
    )
    all_records = [row for value in evaluations.values() for row in value["records"]]
    expected_evidence = {
        "exact_two_IHT_steps": len(steps) == _IHT_STEPS,
        "all_three_masks_executed": set(evaluations) == set(_MASK_NAMES),
        "four_backward_passes": (sum(int(row["backward"]) for row in all_records) == 4),
        "two_terminal_forward_only_passes": (
            len(evaluations["M2"]["records"]) == 2
            and all(row["backward"] is False for row in evaluations["M2"]["records"])
        ),
        "selection_records_only": all(
            row["sequence_index"] in _SELECTION_SEQUENCE_INDICES for row in all_records
        ),
        "no_internal_screen_record_access": not {
            row["sequence_index"] for row in all_records
        }.intersection(_INTERNAL_SCREEN_SEQUENCE_INDICES),
        "exact_51_after_every_IHT_step": all(
            int(_strict_boolean_mask(row["output_mask"], "IHT output").sum())
            == _RESCUED_HEADS
            for row in steps
        ),
        "selected_mask_exact_51": selected_pairs_unique,
        "analytical_budget_exact": (
            actual_expectations == expected_expectations
            and float(expected_expectations["attention_logical_read_fraction"])
            <= _THRESHOLDS["maximum_attention_logical_read_fraction"]
        ),
        "teacher_weights_frozen": (
            protocol["training"]["teacher_weights_frozen"] is True
        ),
        "CPU_only_native_oracle": (
            protocol["training"]["device"] == "cpu"
            and protocol["training"]["native_oracle_device"] == "cpu"
        ),
        "selected_mask_was_executed": selected_name in {"M1", "M2"},
        "robust_improvement_vs_M0": bool(recomputed_selection["screen_eligible"]),
        "post_training_authentication": post_valid,
    }
    expected_artifacts = _expected_training_artifacts(
        protocol,
        protocol_sha256,
    )
    expected_execution_evidence = all(
        value
        for name, value in expected_evidence.items()
        if name != "robust_improvement_vs_M0"
    )
    expected_screen_eligible = (
        expected_execution_evidence and expected_evidence["robust_improvement_vs_M0"]
    )
    performance = result.get("performance")
    all_record_seconds = sum(float(row["elapsed_seconds"]) for row in all_records)
    performance_valid = (
        isinstance(performance, Mapping)
        and set(performance)
        == {
            "elapsed_seconds",
            "executed_record_seconds",
        }
        and _finite_scalar(
            performance.get("elapsed_seconds"),
        )
        and float(performance["elapsed_seconds"]) > 0.0
        and float(performance["elapsed_seconds"]) + 1.0e-9 >= all_record_seconds
        and isinstance(performance.get("executed_record_seconds"), Mapping)
        and set(performance["executed_record_seconds"]) == set(_MASK_NAMES)
        and all(
            performance["executed_record_seconds"][name]
            == [row["elapsed_seconds"] for row in evaluations[name]["records"]]
            for name in _MASK_NAMES
        )
    )
    evidence_valid = (
        isinstance(evidence, Mapping)
        and set(evidence) == set(_TRAINING_EVIDENCE_NAMES)
        and all(isinstance(value, bool) for value in evidence.values())
    )
    if (
        isinstance(result.get("schema_version"), bool)
        or not isinstance(result.get("schema_version"), int)
        or result.get("schema_version") != 1
        or result.get("experiment") != _TRAINING_EXPERIMENT
        or result.get("status") != _TRAINING_STATUS
        or result.get("evidence_passed") is not expected_execution_evidence
        or result.get("framework_contract") != protocol["framework_contract"]
        or result.get("record_split") != protocol["record_split"]
        or result.get("training_data_access") != protocol["training_data_access"]
        or result.get("training") != protocol["training"]
        or result.get("objective") != protocol["objective"]
        or result.get("budget_contract") != protocol["budget_contract"]
        or actual_expectations != expected_expectations
        or result.get("limitations") != protocol["limitations"]
        or isinstance(result.get("selected_head_count"), bool)
        or not isinstance(result.get("selected_head_count"), int)
        or result.get("selected_head_count") != _RESCUED_HEADS
        or len(selected) != _RESCUED_HEADS
        or result.get("selected_mask_name") != selected_name
        or result.get("mask_selection") != recomputed_selection
        or result.get("selected_heads") != expected_rows
        or result.get("attention_head_mask") != selected_mask.tolist()
        or result.get("mask_churn")
        != {
            "M0_to_M1": int(np.count_nonzero(masks["M0"] != masks["M1"])),
            "M1_to_M2": int(np.count_nonzero(masks["M1"] != masks["M2"])),
        }
        or not evidence_valid
        or evidence != expected_evidence
        or result.get("native_screen_eligible") is not expected_screen_eligible
        or result.get("decision")
        != (
            "freeze_exactly_one_native_internal_development_screen"
            if expected_screen_eligible
            else "stop_causal_gate_without_native_screen_no_robust_M0_improvement"
        )
        or not post_valid
        or artifacts != expected_artifacts
        or result.get("attention_head_mask_sha256")
        != sha256_json(selected_mask.tolist())
        or not performance_valid
    ):
        raise ValueError("causal head-gate training result is invalid")
    if not expected_screen_eligible:
        raise ValueError(
            "causal head-gate training did not qualify for native screening"
        )
    return selected


def _build_causal_screen_protocol(
    context: Mapping[str, Any],
    *,
    gate_protocol_sha256: str,
    training_result: Mapping[str, Any],
    training_result_sha256: str,
    prerequisite_hashes: Mapping[str, str],
    source_sha256: str,
    source_inventory: Mapping[str, str],
    threads: int,
) -> dict[str, Any]:
    selected = headwise._selected_pairs(training_result)
    internal = list(context["split"]["internal_holdout"])
    internal_order = [int(row["sequence_index"]) for row in internal]
    if internal_order != list(_INTERNAL_SCREEN_SEQUENCE_ORDER):
        raise ValueError("causal gate internal screen order changed")
    return {
        "schema_version": 1,
        "experiment": _SCREEN_PROTOCOL_EXPERIMENT,
        "status": _SCREEN_PROTOCOL_STATUS,
        **headwise._base_bindings(context),
        **dict(prerequisite_hashes),
        "gate_protocol_sha256": gate_protocol_sha256,
        "gate_training_result_sha256": training_result_sha256,
        "training_attention_library_sha256": training_result["artifacts"][
            "training_attention_library_sha256"
        ],
        "causal_gate_source_sha256": source_sha256,
        "causal_gate_source_inventory_sha256": dict(source_inventory),
        "headwise_library_sha256": prerequisite_hashes["headwise_library_sha256"],
        "record_split": context["split"],
        "record_split_identity": context["split"]["split_identity"],
        "internal_screen_records": internal,
        "internal_screen_sequence_indices": internal_order,
        "population_contract": headwise._population_contract(len(internal_order)),
        "selected_heads": training_result["selected_heads"],
        "attention_head_mask": training_result["attention_head_mask"],
        "head_mask_identity_sha256": training_result["attention_head_mask_sha256"],
        "selected_head_count": len(selected),
        "attention_head_policies": headwise._head_policies(selected),
        "budget_contract": headwise._headwise_budget_contract(context["model"]),
        "model": context["model"],
        "q7_expectations_per_sequence": context["q7_expectations"],
        "quality_bands": [
            {"name": name, "start": start, "stop": stop}
            for name, start, stop in headwise._QUALITY_BANDS
        ],
        "thresholds": _THRESHOLDS,
        "scope": {
            "candidate_device": "cpu",
            "candidate_threads": threads,
            "candidate_transformers_model_shell": False,
            "candidate_count": 1,
            "candidate_mask_fixed_before_screen_protocol": True,
            "candidate_mask_adaptation_after_freeze": False,
            "primary_sequence_order": internal_order,
            "reset_replay_sequence_index": internal_order[0],
            "reset_replay_excluded_from_semantic_metrics": True,
            "historical_all_base_parity_reused_by_hash": True,
            "q7_artifact_or_policy_changed": False,
            "package_manifest_mutated": False,
            "development_screen_only": True,
            "fresh_confirmation_required_after_pass": True,
        },
        "decision_rule": {
            "evidence_failure": ("stop and diagnose without semantic attribution"),
            "authenticated_quality_failure": (
                "close this predeclared static causal-gate attempt without "
                "opening confirmation"
            ),
            "authenticated_quality_pass": (
                "freeze a separately sealed package-native confirmation"
            ),
        },
        "provenance": {
            "training_protocol_frozen_before_gradients": True,
            "training_result_fixed_before_screen_protocol": True,
            "screen_protocol_frozen_before_native_candidate_execution": True,
            "six_screen_records_previously_consumed_by_diagnostics": True,
            "six_screen_records_are_not_an_unseen_holdout": True,
            "screen_outputs_cannot_change_mask": True,
            "confirmation_corpus_unopened": True,
        },
        "limitations": [
            "This is a reused six-record development screen, not an unseen holdout.",
            "Exactly one trained static mask may be executed under this protocol.",
            "A pass still requires a separately sealed package-native confirmation.",
            "Logical bytes are analytical native reads, not hardware DRAM counters.",
            "W128 is full context only for this 128-position experiment.",
        ],
    }


def _validate_causal_screen_protocol(
    protocol: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    protocol_sha256: str,
    supplied_sha256: str,
    gate_protocol_sha256: str,
    training_result: Mapping[str, Any],
    training_result_sha256: str,
    prerequisite_hashes: Mapping[str, str],
    source_sha256: str,
    source_inventory: Mapping[str, str],
) -> None:
    expected = _build_causal_screen_protocol(
        context,
        gate_protocol_sha256=gate_protocol_sha256,
        training_result=training_result,
        training_result_sha256=training_result_sha256,
        prerequisite_hashes=prerequisite_hashes,
        source_sha256=source_sha256,
        source_inventory=source_inventory,
        threads=int(protocol.get("scope", {}).get("candidate_threads", 0)),
    )
    if (
        protocol_sha256 != str(supplied_sha256).lower()
        or protocol != expected
        or protocol["scope"]["candidate_threads"] != _THREADS
        or protocol["internal_screen_sequence_indices"]
        != list(_INTERNAL_SCREEN_SEQUENCE_ORDER)
    ):
        raise ValueError("causal head-gate screen protocol is invalid")


def _screen_chain(
    *,
    manifest_sha256: str,
    gate_protocol: str | Path,
    gate_protocol_sha256: str,
    gate_training_result: str | Path,
    gate_training_result_sha256: str,
    attention_library: str | Path,
    attention_library_sha256: str,
    kwargs: dict[str, Any],
) -> tuple[
    dict[str, Path],
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
    Path,
    dict[str, Any],
    str,
    list[tuple[int, int]],
    dict[str, Any],
]:
    if not _PREREQUISITE_ARGUMENT_NAMES.issubset(kwargs):
        raise ValueError("causal head-gate prerequisite arguments are incomplete")
    attention_path, attention_hash = _authenticate_attention_library(
        attention_library,
        attention_library_sha256,
    )
    artifact_kwargs = {name: kwargs.pop(name) for name in _PREREQUISITE_ARGUMENT_NAMES}
    common_paths, context = headwise._common_context(
        manifest_sha256=manifest_sha256,
        **kwargs,
    )
    prerequisite_hashes, boundary = _authenticate_failed_headwise_boundary(
        context,
        **artifact_kwargs,
    )
    gate_protocol_path = Path(gate_protocol).expanduser().resolve()
    gate_protocol_hash = _require_digest(
        gate_protocol_path,
        gate_protocol_sha256,
        "causal head-gate protocol",
    )
    gate_protocol_value = _read_json(
        gate_protocol_path,
        "causal head-gate protocol",
    )
    source_hash = sha256_file(Path(__file__).resolve())
    source_inventory = _current_source_inventory(context)
    _validate_gate_protocol(
        gate_protocol_value,
        context,
        protocol_sha256=gate_protocol_hash,
        supplied_sha256=gate_protocol_sha256,
        prerequisite_hashes=prerequisite_hashes,
        source_sha256=source_hash,
        source_inventory=source_inventory,
        framework_contract=_framework_contract(),
        attention_library_path=attention_path,
        attention_library_sha256=attention_hash,
    )
    training_path = Path(gate_training_result).expanduser().resolve()
    training_hash = _require_digest(
        training_path,
        gate_training_result_sha256,
        "causal head-gate training result",
    )
    training = _read_json(
        training_path,
        "causal head-gate training result",
    )
    selected = _validate_training_result(
        training,
        result_sha256=training_hash,
        supplied_sha256=gate_training_result_sha256,
        protocol=gate_protocol_value,
        protocol_sha256=gate_protocol_hash,
    )
    return (
        common_paths,
        context,
        prerequisite_hashes,
        boundary,
        gate_protocol_path,
        gate_protocol_value,
        training_hash,
        selected,
        {
            "training_path": training_path,
            "training": training,
            "source_hash": source_hash,
            "source_inventory": source_inventory,
            "gate_protocol_hash": gate_protocol_hash,
            "attention_library_path": attention_path,
            "attention_library_sha256": attention_hash,
        },
    )


def freeze_native_olmoe_causal_head_gate_screen_protocol(
    *,
    gate_protocol: str | Path,
    gate_protocol_sha256: str,
    gate_training_result: str | Path,
    gate_training_result_sha256: str,
    attention_library: str | Path,
    attention_library_sha256: str,
    out: str | Path,
    manifest_sha256: str,
    threads: int = _THREADS,
    **kwargs: Any,
) -> dict[str, Any]:
    output_path = Path(out).expanduser().resolve()
    if output_path.exists():
        raise ValueError("causal head-gate screen protocol target exists")
    if threads != _THREADS:
        raise ValueError("causal head-gate screen requires 12 threads")
    (
        _common_paths,
        context,
        prerequisite_hashes,
        _boundary,
        _gate_protocol_path,
        _gate_protocol_value,
        training_hash,
        _selected,
        values,
    ) = _screen_chain(
        manifest_sha256=manifest_sha256,
        gate_protocol=gate_protocol,
        gate_protocol_sha256=gate_protocol_sha256,
        gate_training_result=gate_training_result,
        gate_training_result_sha256=gate_training_result_sha256,
        attention_library=attention_library,
        attention_library_sha256=attention_library_sha256,
        kwargs=dict(kwargs),
    )
    protocol = _build_causal_screen_protocol(
        context,
        gate_protocol_sha256=values["gate_protocol_hash"],
        training_result=values["training"],
        training_result_sha256=training_hash,
        prerequisite_hashes=prerequisite_hashes,
        source_sha256=values["source_hash"],
        source_inventory=values["source_inventory"],
        threads=threads,
    )
    atomic_json(output_path, protocol)
    return protocol


def _causal_screen_post_authentication(
    context: Mapping[str, Any],
    common_paths: Mapping[str, Path],
    *,
    manifest_sha256: str,
    source_inventory: Mapping[str, str],
    prerequisite_paths: Mapping[str, Path],
    prerequisite_hashes: Mapping[str, str],
    gate_protocol_path: Path,
    gate_protocol_sha256: str,
    training_path: Path,
    training_sha256: str,
    screen_protocol_path: Path,
    screen_protocol_sha256: str,
    attention_library: Path,
    attention_library_sha256: str,
) -> dict[str, bool]:
    checks = headwise._common_post_authentication(
        context,
        common_paths,
        manifest_sha256=manifest_sha256,
        source_inventory=source_inventory,
    )
    checks.update(
        {
            name: sha256_file(path) == prerequisite_hashes[f"{name}_sha256"]
            for name, path in prerequisite_paths.items()
        }
    )
    checks.update(
        {
            "gate_protocol": (sha256_file(gate_protocol_path) == gate_protocol_sha256),
            "gate_training_result": (sha256_file(training_path) == training_sha256),
            "gate_screen_protocol": (
                sha256_file(screen_protocol_path) == screen_protocol_sha256
            ),
            "causal_gate_source": (
                sha256_file(Path(__file__).resolve())
                == source_inventory["src/engram/evaluation/olmoe_causal_head_gate.py"]
            ),
            "training_attention_library": (
                sha256_file(attention_library) == attention_library_sha256
            ),
        }
    )
    return checks


def evaluate_native_olmoe_causal_head_gate_screen(
    *,
    gate_protocol: str | Path,
    gate_protocol_sha256: str,
    gate_training_result: str | Path,
    gate_training_result_sha256: str,
    attention_library: str | Path,
    attention_library_sha256: str,
    screen_protocol: str | Path,
    screen_protocol_sha256: str,
    out: str | Path,
    manifest_sha256: str,
    threads: int = _THREADS,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the one frozen final mask through the exact native per-head path."""

    output_path = Path(out).expanduser().resolve()
    if output_path.exists():
        raise ValueError("causal head-gate screen result target exists")
    if threads != _THREADS:
        raise ValueError("causal head-gate screen requires 12 threads")
    (
        common_paths,
        context,
        prerequisite_hashes,
        boundary,
        gate_protocol_path,
        gate_protocol_value,
        training_hash,
        selected,
        values,
    ) = _screen_chain(
        manifest_sha256=manifest_sha256,
        gate_protocol=gate_protocol,
        gate_protocol_sha256=gate_protocol_sha256,
        gate_training_result=gate_training_result,
        gate_training_result_sha256=gate_training_result_sha256,
        attention_library=attention_library,
        attention_library_sha256=attention_library_sha256,
        kwargs=dict(kwargs),
    )
    protocol_path = Path(screen_protocol).expanduser().resolve()
    protocol_hash = _require_digest(
        protocol_path,
        screen_protocol_sha256,
        "causal head-gate screen protocol",
    )
    protocol = _read_json(
        protocol_path,
        "causal head-gate screen protocol",
    )
    _validate_causal_screen_protocol(
        protocol,
        context,
        protocol_sha256=protocol_hash,
        supplied_sha256=screen_protocol_sha256,
        gate_protocol_sha256=values["gate_protocol_hash"],
        training_result=values["training"],
        training_result_sha256=training_hash,
        prerequisite_hashes=prerequisite_hashes,
        source_sha256=values["source_hash"],
        source_inventory=values["source_inventory"],
    )
    teacher_logits, teacher_hidden, targets = headwise._load_teacher_arrays(
        context,
        common_paths["arrays_path"],
    )
    sequence_indices = list(protocol["internal_screen_sequence_indices"])
    started = time.perf_counter()
    candidate = headwise._evaluate_headwise_candidate(
        selected,
        sequence_indices=sequence_indices,
        context=context,
        library=boundary["paths"]["headwise_library"],
        teacher_logits=teacher_logits,
        teacher_hidden=teacher_hidden,
        targets=targets,
        threads=threads,
        replay_sequence_index=protocol["scope"]["reset_replay_sequence_index"],
    )
    execution_seconds = time.perf_counter() - started
    post = _causal_screen_post_authentication(
        context,
        common_paths,
        manifest_sha256=manifest_sha256,
        source_inventory=values["source_inventory"],
        prerequisite_paths=boundary["paths"],
        prerequisite_hashes=prerequisite_hashes,
        gate_protocol_path=gate_protocol_path,
        gate_protocol_sha256=values["gate_protocol_hash"],
        training_path=values["training_path"],
        training_sha256=training_hash,
        screen_protocol_path=protocol_path,
        screen_protocol_sha256=protocol_hash,
        attention_library=values["attention_library_path"],
        attention_library_sha256=values["attention_library_sha256"],
    )
    actual_resources = candidate["attention_expectations_per_sequence"]
    expected_resources = protocol["budget_contract"][
        "attention_expectations_per_sequence"
    ]
    q7_traffic = layer_rescue._q7_traffic_contract(
        context["model"],
        context["q7_expectations"],
    )
    historical_parity = boundary["failed_screen_result"]["all_base_headwise_parity"]
    resource_checks = {
        "exact_51_of_256_heads": (
            len(selected) == _RESCUED_HEADS and len(set(selected)) == _RESCUED_HEADS
        ),
        "exact_attention_resource_contract": (actual_resources == expected_resources),
        "attention_logical_read_fraction": (
            actual_resources["attention_logical_read_fraction"]
            <= _THRESHOLDS["maximum_attention_logical_read_fraction"]
        ),
        "52_head_boundary_inadmissible": (
            protocol["budget_contract"]["next_head_boundary"]["within_budget"] is False
        ),
        "q7_expectations_unchanged": (
            candidate["q7_expectations_per_sequence"] == context["q7_expectations"]
        ),
        "q7_traffic_fraction": (
            q7_traffic["q7_fraction_of_all_expert_ideal_q4"]
            <= _THRESHOLDS["maximum_q7_traffic_fraction"]
        ),
    }
    evidence_checks = {
        "authenticated_historical_all_base_parity": (
            historical_parity.get("passed") is True
            and historical_parity.get("headwise_library_sha256")
            == prerequisite_hashes["headwise_library_sha256"]
        ),
        "one_candidate_only": True,
        "frozen_internal_sequence_order": (
            sequence_indices == list(_INTERNAL_SCREEN_SEQUENCE_ORDER)
        ),
        "mask_unchanged_by_screen": (
            values["training"]["attention_head_mask_sha256"]
            == protocol["head_mask_identity_sha256"]
        ),
        "candidate_evidence": candidate["evidence_passed"],
        "resource_contract": all(resource_checks.values()),
        "post_run_authentication": all(post.values()),
    }
    evidence_passed = all(evidence_checks.values())
    quality_passed = bool(candidate["quality_passed"])
    if not evidence_passed:
        status = "causal_head_gate_screen_invalid"
        decision = "stop_and_diagnose_causal_head_gate_screen_evidence"
    elif quality_passed:
        status = "causal_head_gate_screen_development_complete"
        decision = "freeze_separately_sealed_package_native_confirmation"
    else:
        status = "causal_head_gate_screen_development_complete"
        decision = "close_predeclared_static_causal_gate_without_opening_confirmation"
    report = {
        "schema_version": 1,
        "experiment": _SCREEN_EXPERIMENT,
        "status": status,
        "artifacts": {
            **headwise._base_bindings(context),
            **prerequisite_hashes,
            "gate_protocol_sha256": values["gate_protocol_hash"],
            "gate_training_result_sha256": training_hash,
            "gate_screen_protocol_sha256": protocol_hash,
            "causal_gate_source_sha256": values["source_hash"],
            "causal_gate_source_inventory_sha256": values["source_inventory"],
        },
        "record_split": context["split"],
        "internal_screen_sequence_indices": sequence_indices,
        "selected_heads": values["training"]["selected_heads"],
        "attention_head_mask": values["training"]["attention_head_mask"],
        "head_mask_identity_sha256": protocol["head_mask_identity_sha256"],
        "attention_head_policies": protocol["attention_head_policies"],
        "budget_contract": protocol["budget_contract"],
        "q7_traffic_contract_per_sequence": q7_traffic,
        "quality_bands": protocol["quality_bands"],
        "thresholds": protocol["thresholds"],
        "population_contract": protocol["population_contract"],
        "historical_all_base_headwise_parity": historical_parity,
        "candidate_evaluation_count": 1,
        "internal_screen_result": candidate,
        "resource_checks": resource_checks,
        "evidence_checks": evidence_checks,
        "evidence_passed": evidence_passed,
        "internal_screen_quality_passed": quality_passed,
        "fresh_confirmation_required": evidence_passed and quality_passed,
        "decision": decision,
        "post_run_authentication": post,
        "performance": {
            "execution_seconds": execution_seconds,
            "candidate_primary_sequence_seconds": candidate["performance"][
                "primary_sequence_seconds"
            ],
            "candidate_reset_replay_seconds": candidate["performance"][
                "reset_replay_seconds"
            ],
            "historical_parity_execution_seconds_reused": historical_parity[
                "elapsed_seconds"
            ],
        },
        "provenance": protocol["provenance"],
        "limitations": protocol["limitations"],
    }
    atomic_json(output_path, report)
    return report


def _add_prerequisite_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trace-protocol", required=True, type=Path)
    parser.add_argument("--trace-protocol-sha256", required=True)
    parser.add_argument("--trace-metadata", required=True, type=Path)
    parser.add_argument("--trace-metadata-sha256", required=True)
    parser.add_argument("--trace-arrays", required=True, type=Path)
    parser.add_argument("--trace-arrays-sha256", required=True)
    parser.add_argument("--failed-head-mask", required=True, type=Path)
    parser.add_argument("--failed-head-mask-sha256", required=True)
    parser.add_argument("--failed-screen-protocol", required=True, type=Path)
    parser.add_argument("--failed-screen-protocol-sha256", required=True)
    parser.add_argument("--failed-screen-result", required=True, type=Path)
    parser.add_argument("--failed-screen-result-sha256", required=True)
    parser.add_argument("--headwise-library", required=True, type=Path)
    parser.add_argument("--headwise-library-sha256", required=True)


def _prerequisite_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {name: getattr(args, name) for name in _PREREQUISITE_ARGUMENT_NAMES}


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze, train, and screen the OLMoE causal head gate"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser(
        "freeze",
        help="freeze two full-selection IHT steps and terminal evaluation",
    )
    train = commands.add_parser(
        "train",
        help="train the frozen hard-projected 51-head causal gate",
    )
    freeze_screen = commands.add_parser(
        "freeze-screen",
        help="freeze the final mask and native development screen",
    )
    evaluate = commands.add_parser(
        "evaluate",
        help="execute exactly one final mask through the native runtime",
    )
    for command in (freeze, train, freeze_screen, evaluate):
        headwise._add_common_arguments(command)
        _add_prerequisite_arguments(command)
        command.add_argument(
            "--attention-library",
            required=True,
            type=Path,
        )
        command.add_argument("--attention-library-sha256", required=True)
    freeze.add_argument("--device", choices=("cpu",), default="cpu")
    freeze.add_argument("--threads", type=int, default=_THREADS)
    freeze.add_argument("--out", required=True, type=Path)
    train.add_argument("--gate-protocol", required=True, type=Path)
    train.add_argument("--gate-protocol-sha256", required=True)
    train.add_argument("--out", required=True, type=Path)
    for command in (freeze_screen, evaluate):
        command.add_argument("--gate-protocol", required=True, type=Path)
        command.add_argument("--gate-protocol-sha256", required=True)
        command.add_argument(
            "--gate-training-result",
            required=True,
            type=Path,
        )
        command.add_argument("--gate-training-result-sha256", required=True)
        command.add_argument("--threads", type=int, default=_THREADS)
    freeze_screen.add_argument("--out", required=True, type=Path)
    evaluate.add_argument("--screen-protocol", required=True, type=Path)
    evaluate.add_argument("--screen-protocol-sha256", required=True)
    evaluate.add_argument("--out", required=True, type=Path)

    args = parser.parse_args(argv)
    common = headwise._common_from_args(args)
    manifest_sha256 = common.pop("manifest_sha256")
    prerequisites = _prerequisite_from_args(args)
    if args.command == "freeze":
        result = freeze_native_olmoe_causal_head_gate_protocol(
            **common,
            **prerequisites,
            manifest_sha256=manifest_sha256,
            attention_library=args.attention_library,
            attention_library_sha256=args.attention_library_sha256,
            device=args.device,
            threads=args.threads,
            out=args.out,
        )
    elif args.command == "train":
        result = train_native_olmoe_causal_head_gate(
            **common,
            **prerequisites,
            manifest_sha256=manifest_sha256,
            attention_library=args.attention_library,
            attention_library_sha256=args.attention_library_sha256,
            gate_protocol=args.gate_protocol,
            gate_protocol_sha256=args.gate_protocol_sha256,
            out=args.out,
        )
    elif args.command == "freeze-screen":
        result = freeze_native_olmoe_causal_head_gate_screen_protocol(
            **common,
            **prerequisites,
            manifest_sha256=manifest_sha256,
            attention_library=args.attention_library,
            attention_library_sha256=args.attention_library_sha256,
            gate_protocol=args.gate_protocol,
            gate_protocol_sha256=args.gate_protocol_sha256,
            gate_training_result=args.gate_training_result,
            gate_training_result_sha256=(args.gate_training_result_sha256),
            threads=args.threads,
            out=args.out,
        )
    else:
        result = evaluate_native_olmoe_causal_head_gate_screen(
            **common,
            **prerequisites,
            manifest_sha256=manifest_sha256,
            attention_library=args.attention_library,
            attention_library_sha256=args.attention_library_sha256,
            gate_protocol=args.gate_protocol,
            gate_protocol_sha256=args.gate_protocol_sha256,
            gate_training_result=args.gate_training_result,
            gate_training_result_sha256=(args.gate_training_result_sha256),
            screen_protocol=args.screen_protocol,
            screen_protocol_sha256=args.screen_protocol_sha256,
            threads=args.threads,
            out=args.out,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
