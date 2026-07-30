"""Certified same-state oracle over every value already read by attention.

The preceding slot-simplex experiment exposed one aggregate regular-cache
value and eight individual episodic values.  Its certified failure does not
answer whether the twenty regular values that were collapsed into that
aggregate contain useful directions.  This module defines the next
prospective experiment:

* sixteen exact local values, in native oldest-to-newest order;
* four selected older values, in native score/tie-break order; and
* eight exact BF16 episodic values, in read-span order.

The constructible arm is a 28-way product simplex initialized at the exact
native softmax masses.  The optimistic arm adds the exact native head output
as component 29 solely to absorb trace-regrouping roundoff.  Only the
constructible arm can authorize a causal selector.  Optional 10- and 16-way
top-mass nested bases are diagnostics; they never have progression authority.

Trace capture is deliberately isolated behind :class:`RegularEntryTraceAdapter`.
That keeps the numerical experiment testable without loading a native library
and prevents the evaluator from silently inventing a runtime ABI.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

import engram.evaluation.olmoe_product_simplex_active_set_solver as active_solver
import engram.evaluation.olmoe_retrieval_episodic_head_mass_oracle as mass
import engram.evaluation.olmoe_retrieval_episodic_joint_gamma_oracle as joint
from engram.utils import atomic_json, sha256_file, sha256_json

_SCHEMA_VERSION = 1
_PROTOCOL_EXPERIMENT = "olmoe_q7_retrieval_episodic_full_visible_simplex_protocol"
_PARITY_EXPERIMENT = "olmoe_q7_retrieval_episodic_full_visible_simplex_trace_parity"
_CAPTURE_EXPERIMENT = "olmoe_q7_retrieval_episodic_full_visible_simplex_capture"
_RESULT_EXPERIMENT = "olmoe_q7_retrieval_episodic_full_visible_simplex_train_screen"
_PROTOCOL_STATUS = "frozen_before_full_visible_train_capture"
_PARITY_STATUS = "full_visible_trace_parity_passed"
_RESULT_FAILED_STATUS = "train_full_visible_simplex_gate_failed"
_RESULT_PASSED_STATUS = "train_full_visible_simplex_gate_passed"
_REGULAR_ENTRY_COPY_SYMBOL = "engram_olmoe_token_copy_last_regular_entry_trace_v1"

_LOCAL_ENTRIES = 16
_OLDER_ENTRIES = 4
_REGULAR_ENTRIES = _LOCAL_ENTRIES + _OLDER_ENTRIES
_EPISODIC_ENTRIES = 8
_VISIBLE_ENTRIES = _REGULAR_ENTRIES + _EPISODIC_ENTRIES
_CONSTRUCTIBLE_COMPONENTS = _VISIBLE_ENTRIES
_OPTIMISTIC_COMPONENTS = _VISIBLE_ENTRIES + 1
_NESTED_DIAGNOSTIC_COMPONENTS = (10, 16)

_INVALID_KIND = np.uint8(0)
_LOCAL_KIND = np.uint8(1)
_OLDER_KIND = np.uint8(2)
_INVALID_POSITION = np.uint64(np.iinfo(np.uint64).max)

_QUERY_HEADS = mass._QUERY_HEADS
_HEAD_DIMENSION = mass._HEAD_DIMENSION
_LAYERS = mass._LAYERS
_HIDDEN_SIZE = mass._HIDDEN_SIZE
_RECORDS = mass._RECORDS
_POSITIONS = mass._POSITIONS
_READ_POSITIONS = mass._READ_POSITIONS

_ROW_BATCH_SIZE = 16
_MAXIMUM_ACTIVE_SET_ITERATIONS = 128
_FALLBACK_MAXIMUM_ITERATIONS = 512
_RELATIVE_GAP_TOLERANCE = 1.0e-12
_ABSOLUTE_GAP_TOLERANCE = 1.0e-13
_WORKING_SET_TOLERANCE = 1.0e-12
_KKT_RESIDUAL_TOLERANCE = 1.0e-10
_REDUCED_COST_TOLERANCE = 1.0e-12
_BASE_PROJECTION_ABSOLUTE_TOLERANCE = 2.5e-4
_BASE_PROJECTION_RELATIVE_TOLERANCE = 2.0e-5

_BASE_TRACE_KEYS = (
    "base_attention_output",
    "regular_component",
    "episodic_component",
    "regular_mass",
    "episodic_mass",
    "base_projected",
    "target_residual",
)
_REGULAR_ENTRY_TRACE_KEYS = (
    "regular_entry_mass",
    "regular_entry_values",
    "regular_entry_valid_kind",
    "regular_entry_positions",
)
_EPISODIC_SLOT_TRACE_KEYS = ("slot_mass", "slot_values")
_BASIS_TRACE_KEYS = (
    *_BASE_TRACE_KEYS,
    *_REGULAR_ENTRY_TRACE_KEYS,
    *_EPISODIC_SLOT_TRACE_KEYS,
)
_CAPTURE_TRACE_KEYS = (*_BASIS_TRACE_KEYS, "episodic_source_positions")

_SOURCE_FILES = (
    "native/include/engram/streaming_attention.h",
    "native/include/engram/streaming_attention_c.h",
    "native/src/streaming_attention.cpp",
    "native/src/streaming_attention_c.cpp",
    "native/include/engram/olmoe_token_runtime.h",
    "native/src/olmoe_token_runtime.cpp",
    "native/include/engram/olmoe_token_runtime_c.h",
    "native/src/olmoe_token_runtime_c.cpp",
    "native/src/native_bitnet_token_runtime.cpp",
    "src/engram/compiler/olmoe_native.py",
    "src/engram/runtime/olmoe_native.py",
    "src/engram/evaluation/olmoe_native_causal.py",
    "src/engram/evaluation/olmoe_native_headwise.py",
    "src/engram/evaluation/olmoe_native_sustained.py",
    "src/engram/evaluation/olmoe_joint_gamma_solver.py",
    "src/engram/evaluation/olmoe_product_simplex_solver.py",
    "src/engram/evaluation/olmoe_product_simplex_active_set_solver.py",
    "src/engram/evaluation/olmoe_retrieval_head_selector.py",
    "src/engram/evaluation/olmoe_retrieval_prefix_selector.py",
    "src/engram/evaluation/olmoe_retrieval_episodic_oracle.py",
    "src/engram/evaluation/olmoe_retrieval_episodic_rank_sweep.py",
    "src/engram/evaluation/olmoe_retrieval_episodic_logit_bias.py",
    "src/engram/evaluation/olmoe_retrieval_episodic_residual_capacity.py",
    "src/engram/evaluation/olmoe_retrieval_episodic_head_mask_oracle.py",
    "src/engram/evaluation/olmoe_retrieval_episodic_head_mass_oracle.py",
    "src/engram/evaluation/olmoe_retrieval_episodic_joint_gamma_oracle.py",
    "src/engram/evaluation/olmoe_retrieval_episodic_slot_simplex_oracle.py",
    "src/engram/evaluation/olmoe_retrieval_episodic_slot_simplex_cached.py",
    "src/engram/evaluation/olmoe_retrieval_episodic_full_visible_simplex_oracle.py",
    "src/engram/evaluation/olmoe_retrieval_episodic_full_visible_runner.py",
    "src/engram/utils.py",
)


def _source_inventory() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    return {name: sha256_file(repository / name) for name in _SOURCE_FILES}


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


@dataclass(frozen=True)
class FullVisibleBasis:
    """Validated value basis and an exactly feasible native simplex point."""

    components: np.ndarray
    correction_basis: np.ndarray
    base_coefficients: np.ndarray
    base_heads: np.ndarray
    target_residual: np.ndarray
    visible_mass: np.ndarray
    visible_values: np.ndarray
    visible_valid: np.ndarray
    component_source_indices: np.ndarray
    base_reconstruction_max_abs: float
    regular_mass_reconstruction_max_abs: float
    regular_component_reconstruction_max_abs: float
    episodic_mass_reconstruction_max_abs: float
    episodic_component_reconstruction_max_abs: float
    base_component_reconstruction_max_abs: float
    invalid_regular_entries: int
    diagnostic_only: bool


@dataclass(frozen=True)
class FullVisibleOracleResult:
    """Certified product-simplex solution for one frozen basis."""

    coefficients: np.ndarray
    target_energy: np.ndarray
    objective: np.ndarray
    objective_gap_upper_bound: np.ndarray
    direct_error_energy: np.ndarray
    iterations: np.ndarray
    converged: np.ndarray
    maximum_relative_objective_gap: float
    base_reconstruction_max_abs: float
    regular_mass_reconstruction_max_abs: float
    regular_component_reconstruction_max_abs: float
    episodic_mass_reconstruction_max_abs: float
    episodic_component_reconstruction_max_abs: float
    base_component_reconstruction_max_abs: float
    authenticated_base_projection_max_abs: float
    quadratic_direct_error_energy_max_abs: float
    deterministic_replay_exact: bool
    invalid_regular_entries: int
    component_count: int
    exact_native_anchor_included: bool
    diagnostic_only: bool
    batch_shape: tuple[int, int, int]


@dataclass(frozen=True)
class _ValidatedVisible:
    base_heads: np.ndarray
    target_residual: np.ndarray
    visible_mass: np.ndarray
    visible_values: np.ndarray
    visible_valid: np.ndarray
    regular_mass_error: float
    regular_component_error: float
    episodic_mass_error: float
    episodic_component_error: float
    base_component_error: float
    invalid_regular_entries: int


def _as_float32(array: np.ndarray, name: str) -> np.ndarray:
    value = np.ascontiguousarray(array, dtype=np.float32)
    if not value.size or not np.isfinite(value).all():
        raise ValueError(f"full-visible {name} tensor is invalid")
    return value


def _as_exact_uint8(array: np.ndarray, name: str) -> np.ndarray:
    value = np.ascontiguousarray(array)
    if not value.size or value.dtype != np.uint8:
        raise ValueError(f"full-visible {name} tensor must be uint8")
    return value


def _as_exact_uint64(array: np.ndarray, name: str) -> np.ndarray:
    value = np.ascontiguousarray(array)
    if not value.size or value.dtype != np.uint64:
        raise ValueError(f"full-visible {name} tensor must be uint64")
    return value


def _validate_kind_and_padding(
    kinds: np.ndarray,
    positions: np.ndarray,
    masses: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    local = kinds[..., :_LOCAL_ENTRIES]
    older = kinds[..., _LOCAL_ENTRIES:]
    if np.any((local != _INVALID_KIND) & (local != _LOCAL_KIND)) or np.any(
        (older != _INVALID_KIND) & (older != _OLDER_KIND)
    ):
        raise ValueError("full-visible regular entry kind changed")
    valid = kinds != _INVALID_KIND
    for section in (
        valid[..., :_LOCAL_ENTRIES],
        valid[..., _LOCAL_ENTRIES:],
    ):
        if np.any((~section[..., :-1]) & section[..., 1:]):
            raise ValueError("full-visible regular padding is not suffix-only")
    invalid = ~valid
    if (
        np.any(masses[invalid] != np.float32(0.0))
        or np.any(values[invalid] != np.float32(0.0))
        or np.any(positions[invalid] != _INVALID_POSITION)
        or np.any(masses[valid] <= np.float32(0.0))
        or np.any(positions[valid] == _INVALID_POSITION)
    ):
        raise ValueError("full-visible regular padding contract changed")
    if np.any(np.sum(valid[..., :_LOCAL_ENTRIES], axis=-1) == 0):
        raise ValueError("full-visible trace has no local fallback value")

    local_positions = positions[..., :_LOCAL_ENTRIES]
    local_valid = valid[..., :_LOCAL_ENTRIES]
    adjacent_valid = local_valid[..., :-1] & local_valid[..., 1:]
    if np.any(adjacent_valid & (local_positions[..., 1:] <= local_positions[..., :-1])):
        raise ValueError("full-visible local positions are not increasing")

    flat_positions = positions.reshape(-1, positions.shape[-1])
    flat_valid = valid.reshape(-1, valid.shape[-1])
    for row_positions, row_valid in zip(
        flat_positions,
        flat_valid,
        strict=True,
    ):
        selected = row_positions[row_valid]
        if np.unique(selected).size != selected.size:
            raise ValueError("full-visible regular positions are not unique")
    return np.ascontiguousarray(valid)


def _validate_visible_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    query_heads: int,
    regular_entries: int,
    episodic_entries: int,
) -> _ValidatedVisible:
    if not set(_BASIS_TRACE_KEYS).issubset(arrays):
        raise ValueError("full-visible trace tensors are incomplete")
    if (
        query_heads <= 0
        or regular_entries != _REGULAR_ENTRIES
        or episodic_entries != _EPISODIC_ENTRIES
    ):
        raise ValueError("full-visible component configuration changed")

    base = _as_float32(arrays["base_attention_output"], "base")
    regular = _as_float32(arrays["regular_component"], "regular component")
    episodic = _as_float32(
        arrays["episodic_component"],
        "episodic component",
    )
    base_projected = _as_float32(arrays["base_projected"], "base projected")
    target = _as_float32(arrays["target_residual"], "target residual")
    regular_mass = _as_float32(arrays["regular_mass"], "regular mass")
    episodic_mass = _as_float32(arrays["episodic_mass"], "episodic mass")
    entry_mass = _as_float32(
        arrays["regular_entry_mass"],
        "regular entry mass",
    )
    entry_values = _as_float32(
        arrays["regular_entry_values"],
        "regular entry values",
    )
    entry_kind = _as_exact_uint8(
        arrays["regular_entry_valid_kind"],
        "regular entry kind",
    )
    entry_positions = _as_exact_uint64(
        arrays["regular_entry_positions"],
        "regular entry positions",
    )
    slot_mass = _as_float32(arrays["slot_mass"], "slot mass")
    slot_values = _as_float32(arrays["slot_values"], "slot values")

    if (
        base.ndim < 2
        or base.shape != regular.shape
        or base.shape != episodic.shape
        or base.shape != base_projected.shape
        or base.shape != target.shape
        or base.shape[-1] % query_heads
    ):
        raise ValueError("full-visible base tensor shapes are invalid")
    head_dimension = base.shape[-1] // query_heads
    prefix = base.shape[:-1]
    mass_shape = prefix + (query_heads,)
    regular_shape = mass_shape + (regular_entries,)
    slot_shape = mass_shape + (episodic_entries,)
    if (
        regular_mass.shape != mass_shape
        or episodic_mass.shape != mass_shape
        or entry_mass.shape != regular_shape
        or entry_kind.shape != regular_shape
        or entry_positions.shape != regular_shape
        or entry_values.shape != regular_shape + (head_dimension,)
        or slot_mass.shape != slot_shape
        or slot_values.shape != slot_shape + (head_dimension,)
        or np.any(regular_mass <= np.float32(0.0))
        or np.any(episodic_mass <= np.float32(0.0))
        or np.any(entry_mass < np.float32(0.0))
        or np.any(slot_mass <= np.float32(0.0))
    ):
        raise ValueError("full-visible mass or entry tensor shapes are invalid")

    valid = _validate_kind_and_padding(
        entry_kind,
        entry_positions,
        entry_mass,
        entry_values,
    )
    lower_mantissa = slot_values.view(np.uint32) & np.uint32(0xFFFF)
    if np.any(lower_mantissa != 0):
        raise ValueError("full-visible episodic values are not exact BF16 decodes")

    regular_mass_error = float(
        np.max(np.abs(np.sum(entry_mass, axis=-1) - regular_mass))
    )
    episodic_mass_error = float(
        np.max(np.abs(np.sum(slot_mass, axis=-1) - episodic_mass))
    )
    partition_error = float(
        np.max(np.abs(regular_mass + episodic_mass - np.float32(1.0)))
    )
    if (
        regular_mass_error > 2.0e-5
        or episodic_mass_error > 2.0e-5
        or partition_error > 2.0e-5
    ):
        raise ValueError("full-visible masses do not reconstruct native softmax")

    head_shape = prefix + (query_heads, head_dimension)
    base_heads = base.reshape(head_shape)
    regular_heads = regular.reshape(head_shape)
    episodic_heads = episodic.reshape(head_shape)
    regular_from_entries = np.einsum(
        "...he,...hed->...hd",
        entry_mass,
        entry_values,
        optimize=True,
    )
    episodic_from_slots = np.einsum(
        "...hs,...hsd->...hd",
        slot_mass,
        slot_values,
        optimize=True,
    )
    regular_component_error = float(
        np.max(np.abs(regular_from_entries - regular_heads))
    )
    episodic_component_error = float(
        np.max(np.abs(episodic_from_slots - episodic_heads))
    )
    base_component_error = float(
        np.max(np.abs(regular_heads + episodic_heads - base_heads))
    )
    if (
        regular_component_error > 5.0e-5
        or episodic_component_error > 5.0e-5
        or base_component_error > 5.0e-5
    ):
        raise ValueError("full-visible entries do not reconstruct native output")

    first_valid_index = np.argmax(valid, axis=-1)
    fallback = np.take_along_axis(
        entry_values,
        first_valid_index[..., None, None],
        axis=-2,
    )[..., 0, :]
    safe_regular_values = np.where(
        valid[..., None],
        entry_values,
        fallback[..., None, :],
    )
    visible_values = np.concatenate(
        (safe_regular_values, slot_values),
        axis=-2,
    )
    visible_mass = np.concatenate((entry_mass, slot_mass), axis=-1)
    visible_valid = np.concatenate(
        (valid, np.ones(slot_mass.shape, dtype=bool)),
        axis=-1,
    )
    if (
        visible_values.shape[-2] != _VISIBLE_ENTRIES
        or not np.isfinite(visible_values).all()
        or not np.isfinite(visible_mass).all()
    ):
        raise ValueError("full-visible combined basis is invalid")
    return _ValidatedVisible(
        base_heads=np.ascontiguousarray(base_heads, dtype=np.float64),
        target_residual=np.ascontiguousarray(target, dtype=np.float64),
        visible_mass=np.ascontiguousarray(visible_mass, dtype=np.float64),
        visible_values=np.ascontiguousarray(visible_values, dtype=np.float64),
        visible_valid=np.ascontiguousarray(visible_valid),
        regular_mass_error=regular_mass_error,
        regular_component_error=regular_component_error,
        episodic_mass_error=episodic_mass_error,
        episodic_component_error=episodic_component_error,
        base_component_error=max(base_component_error, partition_error),
        invalid_regular_entries=int(np.count_nonzero(~valid)),
    )


def _assemble_basis(
    validated: _ValidatedVisible,
    *,
    components: np.ndarray,
    coefficients: np.ndarray,
    component_source_indices: np.ndarray,
    include_exact_native_anchor: bool,
    diagnostic_only: bool,
) -> FullVisibleBasis:
    values = np.ascontiguousarray(components, dtype=np.float64)
    native = np.ascontiguousarray(coefficients, dtype=np.float64)
    expected_components = (
        _OPTIMISTIC_COMPONENTS
        if include_exact_native_anchor
        else _CONSTRUCTIBLE_COMPONENTS
    )
    if (
        values.shape[:-1] != native.shape
        or values.shape[:-2] != validated.base_heads.shape[:-1]
        or (not diagnostic_only and values.shape[-2] != expected_components)
        or np.any(native < -1.0e-12)
        or not np.isfinite(values).all()
        or not np.isfinite(native).all()
    ):
        raise ValueError("full-visible assembled basis is invalid")
    sums = np.sum(native, axis=-1, keepdims=True, dtype=np.float64)
    if np.any(sums <= 0.0):
        raise ValueError("full-visible native point has zero mass")
    native /= sums
    if np.max(np.abs(np.sum(native, axis=-1) - 1.0)) > 1.0e-10:
        raise ValueError("full-visible native point is outside the simplex")
    reconstructed = np.einsum(
        "...hc,...hcd->...hd",
        native,
        values,
        optimize=True,
    )
    base_error = float(np.max(np.abs(reconstructed - validated.base_heads)))
    if base_error > 7.5e-5:
        raise ValueError("full-visible basis does not reconstruct native output")
    correction = values - validated.base_heads[..., None, :]
    return FullVisibleBasis(
        components=values,
        correction_basis=np.ascontiguousarray(correction),
        base_coefficients=np.ascontiguousarray(native),
        base_heads=validated.base_heads,
        target_residual=validated.target_residual,
        visible_mass=validated.visible_mass,
        visible_values=validated.visible_values,
        visible_valid=validated.visible_valid,
        component_source_indices=np.ascontiguousarray(
            component_source_indices,
            dtype=np.int16,
        ),
        base_reconstruction_max_abs=base_error,
        regular_mass_reconstruction_max_abs=validated.regular_mass_error,
        regular_component_reconstruction_max_abs=(validated.regular_component_error),
        episodic_mass_reconstruction_max_abs=validated.episodic_mass_error,
        episodic_component_reconstruction_max_abs=(validated.episodic_component_error),
        base_component_reconstruction_max_abs=validated.base_component_error,
        invalid_regular_entries=validated.invalid_regular_entries,
        diagnostic_only=diagnostic_only,
    )


def build_full_visible_basis(
    arrays: Mapping[str, np.ndarray],
    *,
    query_heads: int = _QUERY_HEADS,
    regular_entries: int = _REGULAR_ENTRIES,
    episodic_entries: int = _EPISODIC_ENTRIES,
    include_exact_native_anchor: bool = True,
) -> FullVisibleBasis:
    """Construct the authoritative 28-way or optimistic 29-way basis."""

    validated = _validate_visible_arrays(
        arrays,
        query_heads=query_heads,
        regular_entries=regular_entries,
        episodic_entries=episodic_entries,
    )
    prefix = validated.visible_mass.shape
    source = np.broadcast_to(
        np.arange(_VISIBLE_ENTRIES, dtype=np.int16),
        prefix,
    ).copy()
    if include_exact_native_anchor:
        components = np.concatenate(
            (
                validated.base_heads[..., None, :],
                validated.visible_values,
            ),
            axis=-2,
        )
        coefficients = np.zeros(components.shape[:-1], dtype=np.float64)
        coefficients[..., 0] = 1.0
        source = np.concatenate(
            (
                np.full(source.shape[:-1] + (1,), -1, dtype=np.int16),
                source,
            ),
            axis=-1,
        )
    else:
        components = validated.visible_values
        coefficients = validated.visible_mass.copy()
    return _assemble_basis(
        validated,
        components=components,
        coefficients=coefficients,
        component_source_indices=source,
        include_exact_native_anchor=include_exact_native_anchor,
        diagnostic_only=False,
    )


def build_nested_visible_basis(
    arrays: Mapping[str, np.ndarray],
    *,
    component_count: int,
    query_heads: int = _QUERY_HEADS,
    regular_entries: int = _REGULAR_ENTRIES,
    episodic_entries: int = _EPISODIC_ENTRIES,
) -> FullVisibleBasis:
    """Build a diagnostic top-mass basis with one omitted-value aggregate.

    A ``C``-component diagnostic keeps the ``C-1`` largest native-mass
    individual values and represents every omitted value by its exact
    native-mass conditional mean.  Stable native-order tie breaking makes the
    10-way selected set a subset of the 16-way selected set.
    """

    if component_count not in _NESTED_DIAGNOSTIC_COMPONENTS:
        raise ValueError("full-visible nested diagnostic must use C=10 or C=16")
    validated = _validate_visible_arrays(
        arrays,
        query_heads=query_heads,
        regular_entries=regular_entries,
        episodic_entries=episodic_entries,
    )
    selected_count = component_count - 1
    order = np.argsort(
        -validated.visible_mass,
        axis=-1,
        kind="stable",
    )
    selected = np.ascontiguousarray(order[..., :selected_count])
    selected_valid = np.take_along_axis(
        validated.visible_valid,
        selected,
        axis=-1,
    )
    if not np.all(selected_valid):
        raise ValueError("full-visible nested diagnostic selected padding")
    selected_mass = np.take_along_axis(
        validated.visible_mass,
        selected,
        axis=-1,
    )
    selected_values = np.take_along_axis(
        validated.visible_values,
        selected[..., None],
        axis=-2,
    )
    keep = np.ones(validated.visible_mass.shape, dtype=bool)
    np.put_along_axis(keep, selected, False, axis=-1)
    omitted_mass = np.sum(
        np.where(keep, validated.visible_mass, 0.0),
        axis=-1,
        keepdims=True,
        dtype=np.float64,
    )
    if np.any(omitted_mass <= 0.0):
        raise ValueError("full-visible nested diagnostic has no omitted mass")
    omitted_component = (
        np.sum(
            np.where(
                keep[..., None],
                validated.visible_mass[..., None] * validated.visible_values,
                0.0,
            ),
            axis=-2,
            keepdims=True,
            dtype=np.float64,
        )
        / omitted_mass[..., None]
    )
    components = np.concatenate(
        (omitted_component, selected_values),
        axis=-2,
    )
    coefficients = np.concatenate(
        (omitted_mass, selected_mass),
        axis=-1,
    )
    source = np.concatenate(
        (
            np.full(selected.shape[:-1] + (1,), -1, dtype=np.int16),
            selected.astype(np.int16),
        ),
        axis=-1,
    )
    return _assemble_basis(
        validated,
        components=components,
        coefficients=coefficients,
        component_source_indices=source,
        include_exact_native_anchor=False,
        diagnostic_only=True,
    )


def _project_head_basis(
    basis: np.ndarray,
    output_projection: np.ndarray,
) -> np.ndarray:
    values = np.ascontiguousarray(basis, dtype=np.float64)
    weights = np.ascontiguousarray(output_projection, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("full-visible projected basis must be rank four")
    _batch, heads, _components, head_dimension = values.shape
    hidden = heads * head_dimension
    if (
        weights.shape != (hidden, hidden)
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
    ):
        raise ValueError("full-visible output projection shape is invalid")
    head_weights = weights.reshape(hidden, heads, head_dimension).transpose(
        1,
        0,
        2,
    )
    projected = np.einsum(
        "nhci,hoi->nhco",
        values,
        head_weights,
        optimize=True,
    )
    if not np.isfinite(projected).all():
        raise ValueError("full-visible projected basis is non-finite")
    return np.ascontiguousarray(projected, dtype=np.float64)


def _quadratic_from_projected_basis(
    projected_basis: np.ndarray,
    target_residual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projected = np.ascontiguousarray(projected_basis, dtype=np.float64)
    target = np.ascontiguousarray(target_residual, dtype=np.float64)
    if (
        projected.ndim != 4
        or target.shape != (projected.shape[0], projected.shape[-1])
        or not np.isfinite(projected).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("full-visible quadratic tensors are invalid")
    gram = np.einsum(
        "nhco,nkdo->nhckd",
        projected,
        projected,
        optimize=True,
    )
    linear = np.einsum(
        "nhco,no->nhc",
        projected,
        target,
        optimize=True,
    )
    energy = np.einsum("no,no->n", target, target, optimize=True)
    if np.any(energy <= 0.0):
        raise ValueError("full-visible target energy must be positive")
    return (
        np.ascontiguousarray(gram, dtype=np.float64),
        np.ascontiguousarray(linear, dtype=np.float64),
        np.ascontiguousarray(energy, dtype=np.float64),
    )


def _direct_error_energy(
    basis: FullVisibleBasis,
    coefficients: np.ndarray,
    output_projection: np.ndarray,
) -> np.ndarray:
    selected = np.ascontiguousarray(coefficients, dtype=np.float64)
    if (
        selected.shape != basis.base_coefficients.shape
        or np.any(selected < -1.0e-12)
        or np.max(np.abs(np.sum(selected, axis=-1) - 1.0)) > 1.0e-10
    ):
        raise ValueError("full-visible selected coefficients are infeasible")
    candidate = np.einsum(
        "nhc,nhcd->nhd",
        selected,
        basis.components,
        optimize=True,
    )
    delta = (candidate - basis.base_heads).reshape(selected.shape[0], -1)
    correction = (
        delta
        @ np.ascontiguousarray(
            output_projection,
            dtype=np.float64,
        ).T
    )
    error = basis.target_residual - correction
    return np.ascontiguousarray(
        np.einsum("ni,ni->n", error, error, optimize=True),
        dtype=np.float64,
    )


def _slice_layer_batch(
    arrays: Mapping[str, np.ndarray],
    *,
    layer: int,
    begin: int,
    end: int,
) -> dict[str, np.ndarray]:
    sliced: dict[str, np.ndarray] = {}
    for name in _BASE_TRACE_KEYS:
        value = np.ascontiguousarray(arrays[name])
        if name in ("regular_mass", "episodic_mass"):
            sliced[name] = value[:, :, layer].reshape(
                -1,
                _QUERY_HEADS,
            )[begin:end]
        else:
            sliced[name] = value[:, :, layer].reshape(
                -1,
                _HIDDEN_SIZE,
            )[begin:end]
    sliced["regular_entry_mass"] = np.ascontiguousarray(
        arrays["regular_entry_mass"][:, :, layer].reshape(
            -1,
            _QUERY_HEADS,
            _REGULAR_ENTRIES,
        )[begin:end]
    )
    sliced["regular_entry_values"] = np.ascontiguousarray(
        arrays["regular_entry_values"][:, :, layer].reshape(
            -1,
            _QUERY_HEADS,
            _REGULAR_ENTRIES,
            _HEAD_DIMENSION,
        )[begin:end]
    )
    for name in ("regular_entry_valid_kind", "regular_entry_positions"):
        sliced[name] = np.ascontiguousarray(
            arrays[name][:, :, layer].reshape(
                -1,
                _QUERY_HEADS,
                _REGULAR_ENTRIES,
            )[begin:end]
        )
    sliced["slot_mass"] = np.ascontiguousarray(
        arrays["slot_mass"][:, :, layer].reshape(
            -1,
            _QUERY_HEADS,
            _EPISODIC_ENTRIES,
        )[begin:end]
    )
    sliced["slot_values"] = np.ascontiguousarray(
        arrays["slot_values"][:, :, layer].reshape(
            -1,
            _QUERY_HEADS,
            _EPISODIC_ENTRIES,
            _HEAD_DIMENSION,
        )[begin:end]
    )
    return sliced


def _run_oracle_from_arrays(
    arrays: Mapping[str, np.ndarray],
    output_projection: np.ndarray,
    *,
    basis_builder: Callable[[Mapping[str, np.ndarray]], FullVisibleBasis],
    component_count: int,
    exact_native_anchor_included: bool,
    diagnostic_only: bool,
    row_batch_size: int,
    maximum_active_set_iterations: int,
    fallback_maximum_iterations: int,
    relative_gap_tolerance: float,
    absolute_gap_tolerance: float,
) -> FullVisibleOracleResult:
    if set(arrays) != set(_BASIS_TRACE_KEYS):
        raise ValueError("full-visible stacked trace keys changed")
    base = np.ascontiguousarray(arrays["base_attention_output"])
    weights = np.ascontiguousarray(output_projection, dtype=np.float32)
    if base.ndim != 4:
        raise ValueError("full-visible stacked base tensor must be rank four")
    records, read_rows, layers, hidden = base.shape
    if (
        (records, read_rows, layers, hidden)
        != (_RECORDS, len(_READ_POSITIONS), _LAYERS, _HIDDEN_SIZE)
        or weights.shape != (_LAYERS, _HIDDEN_SIZE, _HIDDEN_SIZE)
        or row_batch_size <= 0
        or maximum_active_set_iterations <= 0
        or fallback_maximum_iterations <= 0
        or not np.isfinite(relative_gap_tolerance)
        or relative_gap_tolerance < 0.0
        or not np.isfinite(absolute_gap_tolerance)
        or absolute_gap_tolerance < 0.0
    ):
        raise ValueError("full-visible full oracle configuration is invalid")

    base_projected = np.ascontiguousarray(
        arrays["base_projected"],
        dtype=np.float32,
    )
    maximum_base_projection_error = 0.0
    for layer in range(layers):
        expected = (
            base[:, :, layer].reshape(-1, hidden).astype(np.float64)
            @ weights[layer].astype(np.float64).T
        )
        observed = base_projected[:, :, layer].reshape(-1, hidden).astype(np.float64)
        difference = np.abs(expected - observed)
        maximum_base_projection_error = max(
            maximum_base_projection_error,
            float(np.max(difference)),
        )
        tolerance = (
            _BASE_PROJECTION_ABSOLUTE_TOLERANCE
            + _BASE_PROJECTION_RELATIVE_TOLERANCE * np.abs(expected)
        )
        if np.any(difference > tolerance):
            raise ValueError(
                "full-visible authenticated base projection does not match Wo"
            )

    total_rows = records * read_rows * layers
    rows_per_layer = records * read_rows
    coefficients = np.empty(
        (total_rows, _QUERY_HEADS, component_count),
        dtype=np.float64,
    )
    target_energy = np.empty(total_rows, dtype=np.float64)
    objective = np.empty(total_rows, dtype=np.float64)
    objective_gap = np.empty(total_rows, dtype=np.float64)
    direct_energy = np.empty(total_rows, dtype=np.float64)
    iterations = np.empty(total_rows, dtype=np.int32)
    converged = np.empty(total_rows, dtype=bool)
    maxima = {
        "relative_gap": 0.0,
        "base": 0.0,
        "regular_mass": 0.0,
        "regular_component": 0.0,
        "episodic_mass": 0.0,
        "episodic_component": 0.0,
        "base_component": 0.0,
        "direct": 0.0,
    }
    invalid_regular_entries = 0
    replay_exact = True

    for layer in range(layers):
        layer_weights = weights[layer]
        for begin in range(0, rows_per_layer, row_batch_size):
            end = min(begin + row_batch_size, rows_per_layer)
            batch_arrays = _slice_layer_batch(
                arrays,
                layer=layer,
                begin=begin,
                end=end,
            )
            basis = basis_builder(batch_arrays)
            if (
                basis.components.shape[-2] != component_count
                or basis.diagnostic_only is not diagnostic_only
            ):
                raise ValueError("full-visible basis builder contract changed")
            projected = _project_head_basis(
                basis.correction_basis,
                layer_weights,
            )
            gram, linear, energy = _quadratic_from_projected_basis(
                projected,
                basis.target_residual,
            )
            solve_kwargs = {
                "max_active_set_iterations": maximum_active_set_iterations,
                "fallback_max_iterations": fallback_maximum_iterations,
                "relative_tolerance": relative_gap_tolerance,
                "absolute_tolerance": absolute_gap_tolerance,
                "working_set_tolerance": _WORKING_SET_TOLERANCE,
                "kkt_residual_tolerance": _KKT_RESIDUAL_TOLERANCE,
                "reduced_cost_tolerance": _REDUCED_COST_TOLERANCE,
            }
            solved = active_solver.solve_product_simplex_least_squares_active_set(
                gram,
                linear,
                energy,
                basis.base_coefficients,
                **solve_kwargs,
            )
            replay = active_solver.solve_product_simplex_least_squares_active_set(
                gram,
                linear,
                energy,
                basis.base_coefficients,
                **solve_kwargs,
            )
            replay_exact = replay_exact and all(
                (
                    np.array_equal(solved.coefficients, replay.coefficients),
                    np.array_equal(solved.objective, replay.objective),
                    np.array_equal(
                        solved.objective_gap_upper_bound,
                        replay.objective_gap_upper_bound,
                    ),
                    solved.iterations == replay.iterations,
                    solved.converged == replay.converged,
                    np.array_equal(
                        solved.row_converged,
                        replay.row_converged,
                    ),
                )
            )
            direct = _direct_error_energy(
                basis,
                solved.coefficients,
                layer_weights,
            )
            destination = np.arange(begin, end, dtype=np.int64) * layers + layer
            coefficients[destination] = solved.coefficients
            target_energy[destination] = energy
            objective[destination] = solved.objective
            objective_gap[destination] = solved.objective_gap_upper_bound
            direct_energy[destination] = direct
            iterations[destination] = solved.iterations
            converged[destination] = solved.row_converged
            maxima["relative_gap"] = max(
                maxima["relative_gap"],
                float(solved.max_relative_gap),
            )
            maxima["base"] = max(
                maxima["base"],
                basis.base_reconstruction_max_abs,
            )
            maxima["regular_mass"] = max(
                maxima["regular_mass"],
                basis.regular_mass_reconstruction_max_abs,
            )
            maxima["regular_component"] = max(
                maxima["regular_component"],
                basis.regular_component_reconstruction_max_abs,
            )
            maxima["episodic_mass"] = max(
                maxima["episodic_mass"],
                basis.episodic_mass_reconstruction_max_abs,
            )
            maxima["episodic_component"] = max(
                maxima["episodic_component"],
                basis.episodic_component_reconstruction_max_abs,
            )
            maxima["base_component"] = max(
                maxima["base_component"],
                basis.base_component_reconstruction_max_abs,
            )
            maxima["direct"] = max(
                maxima["direct"],
                float(np.max(np.abs(direct - solved.objective))),
            )
            invalid_regular_entries += basis.invalid_regular_entries
    if not replay_exact:
        raise ValueError("full-visible deterministic solver replay changed")
    return FullVisibleOracleResult(
        coefficients=coefficients,
        target_energy=target_energy,
        objective=objective,
        objective_gap_upper_bound=objective_gap,
        direct_error_energy=direct_energy,
        iterations=iterations,
        converged=converged,
        maximum_relative_objective_gap=maxima["relative_gap"],
        base_reconstruction_max_abs=maxima["base"],
        regular_mass_reconstruction_max_abs=maxima["regular_mass"],
        regular_component_reconstruction_max_abs=maxima["regular_component"],
        episodic_mass_reconstruction_max_abs=maxima["episodic_mass"],
        episodic_component_reconstruction_max_abs=(maxima["episodic_component"]),
        base_component_reconstruction_max_abs=maxima["base_component"],
        authenticated_base_projection_max_abs=(maximum_base_projection_error),
        quadratic_direct_error_energy_max_abs=maxima["direct"],
        deterministic_replay_exact=True,
        invalid_regular_entries=invalid_regular_entries,
        component_count=component_count,
        exact_native_anchor_included=exact_native_anchor_included,
        diagnostic_only=diagnostic_only,
        batch_shape=(records, read_rows, layers),
    )


def run_full_visible_oracle_from_arrays(
    arrays: Mapping[str, np.ndarray],
    output_projection: np.ndarray,
    *,
    include_exact_native_anchor: bool = True,
    row_batch_size: int = _ROW_BATCH_SIZE,
    maximum_active_set_iterations: int = _MAXIMUM_ACTIVE_SET_ITERATIONS,
    fallback_maximum_iterations: int = _FALLBACK_MAXIMUM_ITERATIONS,
    relative_gap_tolerance: float = _RELATIVE_GAP_TOLERANCE,
    absolute_gap_tolerance: float = _ABSOLUTE_GAP_TOLERANCE,
) -> FullVisibleOracleResult:
    """Run the authoritative 28-way or optimistic 29-way cached solve."""

    component_count = (
        _OPTIMISTIC_COMPONENTS
        if include_exact_native_anchor
        else _CONSTRUCTIBLE_COMPONENTS
    )
    return _run_oracle_from_arrays(
        arrays,
        output_projection,
        basis_builder=lambda batch: build_full_visible_basis(
            batch,
            query_heads=_QUERY_HEADS,
            regular_entries=_REGULAR_ENTRIES,
            episodic_entries=_EPISODIC_ENTRIES,
            include_exact_native_anchor=include_exact_native_anchor,
        ),
        component_count=component_count,
        exact_native_anchor_included=include_exact_native_anchor,
        diagnostic_only=False,
        row_batch_size=row_batch_size,
        maximum_active_set_iterations=maximum_active_set_iterations,
        fallback_maximum_iterations=fallback_maximum_iterations,
        relative_gap_tolerance=relative_gap_tolerance,
        absolute_gap_tolerance=absolute_gap_tolerance,
    )


def run_nested_visible_oracle_from_arrays(
    arrays: Mapping[str, np.ndarray],
    output_projection: np.ndarray,
    *,
    component_count: int,
    row_batch_size: int = _ROW_BATCH_SIZE,
    maximum_active_set_iterations: int = _MAXIMUM_ACTIVE_SET_ITERATIONS,
    fallback_maximum_iterations: int = _FALLBACK_MAXIMUM_ITERATIONS,
    relative_gap_tolerance: float = _RELATIVE_GAP_TOLERANCE,
    absolute_gap_tolerance: float = _ABSOLUTE_GAP_TOLERANCE,
) -> FullVisibleOracleResult:
    """Run a non-authoritative 10- or 16-way nested diagnostic."""

    if component_count not in _NESTED_DIAGNOSTIC_COMPONENTS:
        raise ValueError("full-visible diagnostic component count changed")
    return _run_oracle_from_arrays(
        arrays,
        output_projection,
        basis_builder=lambda batch: build_nested_visible_basis(
            batch,
            component_count=component_count,
            query_heads=_QUERY_HEADS,
            regular_entries=_REGULAR_ENTRIES,
            episodic_entries=_EPISODIC_ENTRIES,
        ),
        component_count=component_count,
        exact_native_anchor_included=False,
        diagnostic_only=True,
        row_batch_size=row_batch_size,
        maximum_active_set_iterations=maximum_active_set_iterations,
        fallback_maximum_iterations=fallback_maximum_iterations,
        relative_gap_tolerance=relative_gap_tolerance,
        absolute_gap_tolerance=absolute_gap_tolerance,
    )


def _trace_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in _CAPTURE_TRACE_KEYS:
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _trace_summary(
    arrays: Mapping[str, np.ndarray],
    query_positions: Sequence[int],
) -> dict[str, Any]:
    """Validate one record's complete trace and return immutable evidence."""

    if list(query_positions) != list(_READ_POSITIONS) or set(arrays) != set(
        _CAPTURE_TRACE_KEYS
    ):
        raise ValueError("full-visible trace support changed")
    value_shape = (len(_READ_POSITIONS), _LAYERS, _HIDDEN_SIZE)
    mass_shape = (len(_READ_POSITIONS), _LAYERS, _QUERY_HEADS)
    regular_shape = mass_shape + (_REGULAR_ENTRIES,)
    slot_shape = mass_shape + (_EPISODIC_ENTRIES,)
    tensor_hashes: dict[str, str] = {}
    for name in _BASE_TRACE_KEYS:
        value = np.ascontiguousarray(arrays[name])
        expected = (
            mass_shape if name in ("regular_mass", "episodic_mass") else value_shape
        )
        if (
            value.dtype != np.float32
            or value.shape != expected
            or not value.flags.c_contiguous
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"full-visible {name} trace is invalid")
        tensor_hashes[name] = hashlib.sha256(value.tobytes(order="C")).hexdigest()

    entry_mass = np.ascontiguousarray(arrays["regular_entry_mass"])
    entry_values = np.ascontiguousarray(arrays["regular_entry_values"])
    entry_kind = np.ascontiguousarray(arrays["regular_entry_valid_kind"])
    entry_positions = np.ascontiguousarray(arrays["regular_entry_positions"])
    slot_mass = np.ascontiguousarray(arrays["slot_mass"])
    slot_values = np.ascontiguousarray(arrays["slot_values"])
    episodic_positions = np.ascontiguousarray(arrays["episodic_source_positions"])
    if (
        entry_mass.dtype != np.float32
        or entry_mass.shape != regular_shape
        or entry_values.dtype != np.float32
        or entry_values.shape != regular_shape + (_HEAD_DIMENSION,)
        or entry_kind.dtype != np.uint8
        or entry_kind.shape != regular_shape
        or entry_positions.dtype != np.uint64
        or entry_positions.shape != regular_shape
        or slot_mass.dtype != np.float32
        or slot_mass.shape != slot_shape
        or slot_values.dtype != np.float32
        or slot_values.shape != slot_shape + (_HEAD_DIMENSION,)
        or episodic_positions.dtype != np.uint64
        or episodic_positions.shape != (len(_READ_POSITIONS), _EPISODIC_ENTRIES)
    ):
        raise ValueError("full-visible entry trace shape or dtype changed")
    for name, value in (
        ("regular_entry_mass", entry_mass),
        ("regular_entry_values", entry_values),
        ("regular_entry_valid_kind", entry_kind),
        ("regular_entry_positions", entry_positions),
        ("slot_mass", slot_mass),
        ("slot_values", slot_values),
        ("episodic_source_positions", episodic_positions),
    ):
        tensor_hashes[name] = hashlib.sha256(value.tobytes(order="C")).hexdigest()

    basis = build_full_visible_basis(
        {name: arrays[name] for name in _BASIS_TRACE_KEYS},
        query_heads=_QUERY_HEADS,
        regular_entries=_REGULAR_ENTRIES,
        episodic_entries=_EPISODIC_ENTRIES,
        include_exact_native_anchor=False,
    )
    query = np.asarray(query_positions, dtype=np.uint64)
    if np.any(episodic_positions == _INVALID_POSITION) or np.any(
        episodic_positions >= query[:, None]
    ):
        raise ValueError("full-visible episodic positions are not causal")
    for row in episodic_positions:
        if np.unique(row).size != _EPISODIC_ENTRIES:
            raise ValueError("full-visible episodic positions are not unique")

    valid = entry_kind != _INVALID_KIND
    for row_index, query_position in enumerate(query):
        row_kind = entry_kind[row_index]
        row_positions = entry_positions[row_index]
        local_valid = row_kind == _LOCAL_KIND
        older_valid = row_kind == _OLDER_KIND
        if np.any(row_positions[local_valid] > query_position) or np.any(
            row_positions[older_valid] >= query_position
        ):
            raise ValueError("full-visible regular positions are not causal")
        episode = episodic_positions[row_index]
        flat_older = row_positions[older_valid]
        if np.intersect1d(flat_older, episode).size:
            raise ValueError("full-visible selected older rows duplicate episodic rows")

    valid_counts = np.sum(valid, axis=-1, dtype=np.int64)
    visible_counts = valid_counts + _EPISODIC_ENTRIES
    return {
        "query_positions": list(query_positions),
        "value_shape": list(value_shape),
        "mass_shape": list(mass_shape),
        "regular_entry_mass_shape": list(entry_mass.shape),
        "regular_entry_value_shape": list(entry_values.shape),
        "slot_mass_shape": list(slot_mass.shape),
        "slot_value_shape": list(slot_values.shape),
        "episodic_source_position_shape": list(episodic_positions.shape),
        "tensor_sha256": tensor_hashes,
        "trace_sha256": _trace_digest(arrays),
        "regular_mass_reconstruction_max_abs": (
            basis.regular_mass_reconstruction_max_abs
        ),
        "regular_component_reconstruction_max_abs": (
            basis.regular_component_reconstruction_max_abs
        ),
        "episodic_mass_reconstruction_max_abs": (
            basis.episodic_mass_reconstruction_max_abs
        ),
        "episodic_component_reconstruction_max_abs": (
            basis.episodic_component_reconstruction_max_abs
        ),
        "base_component_reconstruction_max_abs": (
            basis.base_component_reconstruction_max_abs
        ),
        "invalid_regular_entries": basis.invalid_regular_entries,
        "visible_entries_per_head": {
            "minimum": int(np.min(visible_counts)),
            "mean": float(np.mean(visible_counts)),
            "maximum": int(np.max(visible_counts)),
        },
        "local_entries_per_head": {
            "minimum": int(np.min(np.sum(entry_kind == _LOCAL_KIND, axis=-1))),
            "maximum": int(np.max(np.sum(entry_kind == _LOCAL_KIND, axis=-1))),
        },
        "older_entries_per_head": {
            "minimum": int(np.min(np.sum(entry_kind == _OLDER_KIND, axis=-1))),
            "maximum": int(np.max(np.sum(entry_kind == _OLDER_KIND, axis=-1))),
        },
        "slot_values_exact_bf16_decodes": True,
        "regular_order": (
            "entries[0:16] local oldest_to_newest then suffix padding; "
            "entries[16:20] selected older descending native score with "
            "absolute-position tie break then suffix padding"
        ),
        "episodic_order": "read_span_offset_0_to_7",
    }


class RegularEntryTraceAdapter(Protocol):
    """Interface isolating the evaluator from the native runtime ABI."""

    def capture(self, runtime: Any) -> Mapping[str, np.ndarray]:
        """Return one last-row regular-entry trace."""


@dataclass(frozen=True)
class NativeRegularEntryTraceAdapter:
    """Adapter for the prospectively fixed Python runtime method."""

    capability_attribute: str = "regular_entry_trace_available"
    method_name: str = "last_regular_entry_trace"
    copy_symbol: str = _REGULAR_ENTRY_COPY_SYMBOL

    def capture(self, runtime: Any) -> Mapping[str, np.ndarray]:
        if getattr(runtime, self.capability_attribute, False) is not True:
            raise ValueError(
                "full-visible runtime lacks regular-entry trace capability"
            )
        method = getattr(runtime, self.method_name, None)
        if not callable(method):
            raise ValueError("full-visible regular-entry trace method is unavailable")
        trace = method()
        required = {
            "regular_entry_mass": "entry_mass",
            "regular_entry_values": "entry_values",
            "regular_entry_valid_kind": "valid_kind",
            "regular_entry_positions": "positions",
        }
        captured: dict[str, np.ndarray] = {}
        for destination, source in required.items():
            if not hasattr(trace, source):
                raise ValueError(
                    "full-visible regular-entry trace field is unavailable"
                )
            captured[destination] = np.ascontiguousarray(getattr(trace, source))
        return captured


class _FullVisibleTraceCaptureRuntime:
    """Capture wrapper for one-token episodic forwards.

    The wrapper maintains its own source-position ledger from the exact
    write/read directives.  It never asks the native implementation to expose
    private episodic state.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        regular_adapter: RegularEntryTraceAdapter | None = None,
    ) -> None:
        self._runtime = runtime
        self._regular_adapter = (
            NativeRegularEntryTraceAdapter()
            if regular_adapter is None
            else regular_adapter
        )
        slots = int(mass._EPISODIC_POLICY["slots"])
        self._slot_positions = np.full(
            slots,
            _INVALID_POSITION,
            dtype=np.uint64,
        )
        self._query_positions: list[int] = []
        self._rows: dict[str, list[np.ndarray]] = {
            name: [] for name in _CAPTURE_TRACE_KEYS
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    @property
    def position(self) -> int:
        return int(self._runtime.position)

    def forward_episodic(
        self,
        token_ids: Sequence[int],
        write_slots: Sequence[int],
        read_spans: Sequence[int],
    ) -> Any:
        if len(token_ids) != 1 or len(write_slots) != 1 or len(read_spans) != 1:
            raise ValueError("full-visible trace capture requires one-token calls")
        query_position = self.position
        write_slot = int(write_slots[0])
        read_span = int(read_spans[0])
        next_positions = self._slot_positions.copy()
        if write_slot >= 0:
            if write_slot >= next_positions.size:
                raise ValueError("full-visible write slot is outside capacity")
            next_positions[write_slot] = np.uint64(query_position)
        result = self._runtime.forward_episodic(
            token_ids,
            write_slots,
            read_spans,
        )
        if read_span >= 0:
            span_size = int(mass._EPISODIC_POLICY["span_size"])
            begin = read_span * span_size
            end = begin + span_size
            if end > next_positions.size:
                raise ValueError("full-visible read span is outside capacity")
            source_positions = np.ascontiguousarray(next_positions[begin:end])
            if (
                source_positions.size != _EPISODIC_ENTRIES
                or np.any(source_positions == _INVALID_POSITION)
                or np.any(source_positions >= np.uint64(query_position))
            ):
                raise ValueError("full-visible read span source ledger is invalid")
            _old_input, base_projected, target_residual = (
                self._runtime.last_shadow_trace()
            )
            partition = self._runtime.last_episodic_mass_trace()
            slots = self._runtime.last_episodic_slot_trace()
            regular = self._regular_adapter.capture(self._runtime)
            row = {
                "base_attention_output": partition.base_attention_output,
                "regular_component": partition.regular_component,
                "episodic_component": partition.episodic_component,
                "regular_mass": partition.regular_mass,
                "episodic_mass": partition.episodic_mass,
                "base_projected": base_projected,
                "target_residual": target_residual,
                "slot_mass": slots.slot_mass,
                "slot_values": slots.slot_values,
                "episodic_source_positions": source_positions,
                **regular,
            }
            self._query_positions.append(query_position)
            for name in _CAPTURE_TRACE_KEYS:
                value = np.ascontiguousarray(row[name])
                self._rows[name].append(value)
        self._slot_positions = next_positions
        return result

    def captured(self) -> tuple[dict[str, np.ndarray], list[int]]:
        if len(self._query_positions) != len(_READ_POSITIONS):
            raise ValueError("full-visible trace capture is incomplete")
        arrays = {
            name: np.ascontiguousarray(np.stack(rows))
            for name, rows in self._rows.items()
        }
        _trace_summary(arrays, self._query_positions)
        return arrays, list(self._query_positions)

    def reset(self) -> None:
        self._runtime.reset()
        self._slot_positions.fill(_INVALID_POSITION)
        self._query_positions.clear()
        for rows in self._rows.values():
            rows.clear()

    def close(self) -> None:
        self._runtime.close()


def _checked_file(
    value: str | Path,
    expected_sha256: str,
    label: str,
) -> Path:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{label} SHA256 is invalid")
    requested = Path(value).expanduser()
    _reject_confirmation_path(requested, label)
    resolved_parent = requested.parent.resolve(strict=False)
    source = resolved_parent / requested.name
    _reject_confirmation_path(source, label)
    if requested.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    if not source.is_file() or sha256_file(source) != expected_sha256:
        raise ValueError(f"{label} authentication failed")
    return source


def _reject_confirmation_path(value: Path, label: str) -> None:
    if any("confirmation" in part.lower() for part in value.parts):
        raise ValueError(f"{label} is lexically inside confirmation scope")


def _new_output_path(value: str | Path, label: str) -> Path:
    """Resolve an output without touching a forbidden leaf or alias target."""

    requested = Path(value).expanduser()
    _reject_confirmation_path(requested, label)
    resolved_parent = requested.parent.resolve(strict=False)
    output = resolved_parent / requested.name
    _reject_confirmation_path(output, label)
    if requested.is_symlink() or output.exists():
        raise ValueError(f"{label} already exists or is a symlink")
    return output


def _checked_directory(value: str | Path, label: str) -> Path:
    requested = Path(value).expanduser()
    _reject_confirmation_path(requested, label)
    resolved_parent = requested.parent.resolve(strict=False)
    directory = resolved_parent / requested.name
    _reject_confirmation_path(directory, label)
    if requested.is_symlink() or not directory.is_dir():
        raise ValueError(f"{label} is invalid or is a symlink")
    return directory


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _binding(path: Path, sha256: str) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256}


def _validate_predecessor_failure(
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    protocol_binding: Mapping[str, str],
) -> None:
    decision = result.get("decision")
    resource = protocol.get("resource_contract")
    if (
        protocol.get("schema_version") != 1
        or protocol.get("experiment")
        != "olmoe_q7_retrieval_episodic_slot_simplex_cached_v2_protocol"
        or protocol.get("status") != "frozen_before_authenticated_cached_v2_solve"
        or protocol.get("confirmation_split_opened") is not False
        or result.get("schema_version") != 1
        or result.get("experiment")
        != "olmoe_q7_retrieval_episodic_slot_simplex_cached_v2_train_screen"
        or result.get("status") != "train_episodic_slot_simplex_cached_v2_gate_failed"
        or result.get("protocol") != protocol_binding
        or result.get("confirmation_split_opened") is not False
        or not isinstance(decision, Mapping)
        or decision.get("failure_is_decisive") is not True
        or decision.get("certified_optimistic_gate_passed") is not False
        or decision.get("train_slot_simplex_capacity_gate_passed") is not False
        or result.get("resource_contract") != resource
        or resource
        != {
            "fixed_attention_state_bytes": 10534912,
            "fixed_combined_attention_and_episodic_traffic_bytes": 714866688,
            "fixed_fraction_of_dense_full_context_KV": (0.33030523255813954),
            "gamma_zero_earns_read_savings": False,
            "oracle_shadow_trace_and_projection_evaluator_only": True,
            "predictor_weights_features_and_execution_not_counted": True,
        }
    ):
        raise ValueError("full-visible predecessor failure is not authoritative")


def build_trace_parity_report(
    *,
    predecessor_protocol: Mapping[str, str],
    predecessor_result: Mapping[str, str],
    trace_library: Mapping[str, str],
    first_trace: Mapping[str, Any],
    reset_trace: Mapping[str, Any],
    inherited_base_trace_exact: bool,
    outputs_counters_and_loss_exact: bool,
) -> dict[str, Any]:
    """Build the prospective parity artifact from independently run evidence."""

    for name, value in (
        ("predecessor protocol", predecessor_protocol),
        ("predecessor result", predecessor_result),
        ("trace library", trace_library),
    ):
        if set(value) != {"path", "sha256"} or not _is_sha256(value["sha256"]):
            raise ValueError(f"full-visible {name} binding is invalid")
    checks = {
        "inherited_base_trace_exact": inherited_base_trace_exact is True,
        "outputs_counters_and_loss_exact": (outputs_counters_and_loss_exact is True),
        "reset_full_trace_exact": first_trace == reset_trace,
        "regular_entries_reconstruct_regular_component": (
            first_trace.get(
                "regular_component_reconstruction_max_abs",
                float("inf"),
            )
            <= 5.0e-5
        ),
        "episodic_slots_reconstruct_episodic_component": (
            first_trace.get(
                "episodic_component_reconstruction_max_abs",
                float("inf"),
            )
            <= 5.0e-5
        ),
        "base_partition_reconstructs_native_output": (
            first_trace.get(
                "base_component_reconstruction_max_abs",
                float("inf"),
            )
            <= 5.0e-5
        ),
        "slot_values_are_exact_bf16_decodes": (
            first_trace.get("slot_values_exact_bf16_decodes") is True
        ),
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise ValueError("full-visible trace parity failed")
    return {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _PARITY_EXPERIMENT,
        "status": _PARITY_STATUS,
        "predecessor_protocol": dict(predecessor_protocol),
        "predecessor_result": dict(predecessor_result),
        "trace_library": {
            **dict(trace_library),
            "regular_entry_copy_symbol": _REGULAR_ENTRY_COPY_SYMBOL,
        },
        "first_trace": dict(first_trace),
        "reset_trace": dict(reset_trace),
        "checks": checks,
        "confirmation_split_opened": False,
    }


def _validate_parity_report(
    value: Mapping[str, Any],
    *,
    predecessor_protocol: Mapping[str, str],
    predecessor_result: Mapping[str, str],
    trace_library: Mapping[str, str],
) -> None:
    checks = value.get("checks")
    if (
        value.get("schema_version") != _SCHEMA_VERSION
        or value.get("experiment") != _PARITY_EXPERIMENT
        or value.get("status") != _PARITY_STATUS
        or value.get("predecessor_protocol") != predecessor_protocol
        or value.get("predecessor_result") != predecessor_result
        or value.get("trace_library")
        != {
            **dict(trace_library),
            "regular_entry_copy_symbol": _REGULAR_ENTRY_COPY_SYMBOL,
        }
        or value.get("first_trace") != value.get("reset_trace")
        or not isinstance(checks, Mapping)
        or checks.get("passed") is not True
        or not all(check is True for check in checks.values())
        or value.get("confirmation_split_opened") is not False
    ):
        raise ValueError("full-visible parity report contract changed")


def _load_bound_json(
    binding: Mapping[str, Any],
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if (
        not isinstance(binding, Mapping)
        or set(binding) != {"path", "sha256"}
        or not isinstance(binding.get("path"), str)
        or not isinstance(binding.get("sha256"), str)
    ):
        raise ValueError(f"full-visible {label} binding is invalid")
    path = _checked_file(binding["path"], binding["sha256"], label)
    return path, _read_json(path, label)


def freeze_full_visible_protocol(
    *,
    predecessor_protocol: str | Path,
    predecessor_protocol_sha256: str,
    predecessor_result: str | Path,
    predecessor_result_sha256: str,
    trace_library: str | Path,
    trace_library_sha256: str,
    parity_report: str | Path,
    parity_report_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    """Freeze the train-only experiment without touching confirmation data."""

    predecessor_protocol_path = _checked_file(
        predecessor_protocol,
        predecessor_protocol_sha256,
        "full-visible predecessor protocol",
    )
    predecessor_result_path = _checked_file(
        predecessor_result,
        predecessor_result_sha256,
        "full-visible predecessor result",
    )
    trace_library_path = _checked_file(
        trace_library,
        trace_library_sha256,
        "full-visible trace library",
    )
    parity_path = _checked_file(
        parity_report,
        parity_report_sha256,
        "full-visible parity report",
    )
    predecessor_protocol_value = _read_json(
        predecessor_protocol_path,
        "full-visible predecessor protocol",
    )
    predecessor_result_value = _read_json(
        predecessor_result_path,
        "full-visible predecessor result",
    )
    protocol_binding = _binding(
        predecessor_protocol_path,
        predecessor_protocol_sha256,
    )
    result_binding = _binding(
        predecessor_result_path,
        predecessor_result_sha256,
    )
    library_binding = _binding(trace_library_path, trace_library_sha256)
    _validate_predecessor_failure(
        predecessor_protocol_value,
        predecessor_result_value,
        protocol_binding=protocol_binding,
    )
    parity_value = _read_json(parity_path, "full-visible parity report")
    _validate_parity_report(
        parity_value,
        predecessor_protocol=protocol_binding,
        predecessor_result=result_binding,
        trace_library=library_binding,
    )

    historical = predecessor_protocol_value.get("historical_bindings")
    if not isinstance(historical, Mapping):
        raise ValueError("full-visible historical bindings are unavailable")
    confirmation_descriptor = predecessor_protocol_value.get(
        "authenticated_confirmation_descriptor"
    )
    if not isinstance(confirmation_descriptor, Mapping):
        raise ValueError("full-visible confirmation descriptor is unavailable")
    confirmation_name = confirmation_descriptor.get("file")
    if not isinstance(confirmation_name, str) or not confirmation_name:
        raise ValueError("full-visible confirmation descriptor is invalid")
    for historical_binding in historical.values():
        if (
            isinstance(historical_binding, Mapping)
            and isinstance(historical_binding.get("path"), str)
            and Path(historical_binding["path"]).name == confirmation_name
        ):
            raise ValueError("full-visible historical root points at confirmation data")
    _v1_path, v1_protocol = _load_bound_json(
        historical.get("v1_protocol"),
        "full-visible inherited V1 protocol",
    )
    if v1_protocol.get("confirmation_split_opened") is not False or v1_protocol.get(
        "authenticated_confirmation_descriptor"
    ) != predecessor_protocol_value.get("authenticated_confirmation_descriptor"):
        raise ValueError("full-visible inherited V1 protocol changed")
    resource = dict(predecessor_protocol_value["resource_contract"])
    dense_bytes = 2_164_260_864
    cap_bytes = int(0.45 * dense_bytes)
    protocol = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _PROTOCOL_EXPERIMENT,
        "status": _PROTOCOL_STATUS,
        "predecessor_protocol": protocol_binding,
        "predecessor_result": {
            **result_binding,
            "authenticated_decisive_failure": True,
        },
        "historical_bindings": dict(historical),
        "trace_library": {
            **library_binding,
            "regular_entry_copy_symbol": _REGULAR_ENTRY_COPY_SYMBOL,
        },
        "trace_parity": {
            "path": str(parity_path),
            "sha256": parity_report_sha256,
        },
        "schedule_contract": dict(v1_protocol["schedule_contract"]),
        "output_projection": dict(predecessor_protocol_value["output_projection"]),
        "trace_contract": {
            "regular_entries": _REGULAR_ENTRIES,
            "regular_layout": ("position_layer_query_head_regular_entry_dimension"),
            "regular_order": (
                "16 local oldest-to-newest followed by four selected older "
                "in descending score/native tie-break order"
            ),
            "regular_padding": (
                "kind=0,mass=0,value=0,position=UINT64_MAX suffix only"
            ),
            "episodic_entries": _EPISODIC_ENTRIES,
            "episodic_layout": ("position_layer_query_head_read_span_offset_dimension"),
            "required_invariants": [
                "sum(regular_entry_mass)==regular_mass",
                "sum(regular_entry_mass*regular_entry_values)==regular_component",
                "sum(slot_mass)==episodic_mass",
                "sum(slot_mass*slot_values)==episodic_component",
                "regular_component+episodic_component==base_attention_output",
                "ordinary output state and resource counters remain unchanged",
            ],
            "instrumentation_state_and_traffic_counted": False,
        },
        "oracle_method": {
            "authoritative_constructible_components": (_CONSTRUCTIBLE_COMPONENTS),
            "authoritative_optimistic_components": _OPTIMISTIC_COMPONENTS,
            "constructible_components": (
                "16 local + four selected older + eight episodic values"
            ),
            "optimistic_extra_component": (
                "exact native head-output anchor; certificate only"
            ),
            "nested_diagnostic_component_counts": list(_NESTED_DIAGNOSTIC_COMPONENTS),
            "nested_diagnostics_have_progression_authority": False,
            "feasible_set": (
                "Cartesian product of one probability simplex per query "
                "head, optimized jointly after authenticated o_proj"
            ),
            "solver": (
                "deterministic active-set KKT solve with fail-closed "
                "pairwise Frank-Wolfe fallback and full product-simplex gap"
            ),
            "row_batch_size": _ROW_BATCH_SIZE,
            "maximum_active_set_iterations": (_MAXIMUM_ACTIVE_SET_ITERATIONS),
            "fallback_maximum_iterations": _FALLBACK_MAXIMUM_ITERATIONS,
            "relative_objective_gap_target": _RELATIVE_GAP_TOLERANCE,
            "absolute_objective_gap_target": _ABSOLUTE_GAP_TOLERANCE,
            "counterfactual_updates_hidden_or_cache": False,
        },
        "progression_gate": {
            "minimum_global_recovery": 0.50,
            "minimum_every_sequence_recovery": 0.25,
            "minimum_every_block_entry_position_recovery": 0.25,
            "minimum_positive_recovery_layers": 12,
            "objective_never_worse_than_native_base": True,
            "deterministic_coefficient_and_metric_replay_required": True,
            "objective_gap_certificate_required": True,
            "constructible_arm_required_for_selector_authority": True,
        },
        "resource_contract": {
            **resource,
            "dense_full_context_logical_read_bytes": dense_bytes,
            "maximum_deployable_bytes_at_45_percent": cap_bytes,
            "remaining_selector_headroom_bytes": (
                cap_bytes
                - resource["fixed_combined_attention_and_episodic_traffic_bytes"]
            ),
            "fresh_trace_and_oracle_solver_evaluator_only": True,
        },
        "scope": {
            "split": "train",
            "records": _RECORDS,
            "read_positions_per_record": len(_READ_POSITIONS),
            "fresh_native_regular_entry_trace_required": True,
            "same_state_capacity_evidence_only": True,
            "causal_selector": False,
            "causal_rollout": False,
            "semantic_or_M3_pass": False,
            "development_outcomes_used": False,
            "confirmation_file_access_permitted": False,
        },
        "authorized_next_step_on_constructible_pass": (
            "freeze a train-only causal 28-logit correction selector with "
            "packaged CPU rollout and complete selector accounting"
        ),
        "failure_interpretation": (
            "a certified optimistic 29-way failure closes all normalized "
            "same-state reweighting of values already read by W16/C8/K4/S2; "
            "the next experiment must add a different memory/value generator"
        ),
        "authenticated_confirmation_descriptor": dict(
            predecessor_protocol_value["authenticated_confirmation_descriptor"]
        ),
        "source_sha256": _source_inventory(),
        "confirmation_split_opened": False,
    }
    output = _new_output_path(out, "full-visible protocol output")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, protocol)
    return {
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
        "protocol": protocol,
    }


def _prepare_shard_directory(value: str | Path) -> Path:
    requested = _new_output_path(
        value,
        "full-visible trace shard directory",
    )
    requested.mkdir(parents=True)
    return requested


def write_full_visible_trace_shard(
    directory: str | Path,
    *,
    record_index: int,
    record_id: str,
    arrays: Mapping[str, np.ndarray],
    query_positions: Sequence[int],
    source_record_sha256: str,
    output_evidence_sha256: str,
    reset_output_evidence_sha256: str,
    reset_trace_sha256: str,
    schedule_rows_sha256: str,
) -> dict[str, Any]:
    """Persist one authenticated record trace for later CPU-only solving."""

    root = _checked_directory(
        directory,
        "full-visible trace shard directory",
    )
    if (
        isinstance(record_index, bool)
        or not isinstance(record_index, int)
        or not 0 <= record_index < _RECORDS
        or not isinstance(record_id, str)
        or not record_id
        or not all(
            _is_sha256(value)
            for value in (
                source_record_sha256,
                output_evidence_sha256,
                reset_output_evidence_sha256,
                reset_trace_sha256,
                schedule_rows_sha256,
            )
        )
    ):
        raise ValueError("full-visible shard metadata is invalid")
    summary = _trace_summary(arrays, query_positions)
    filename = f"train-{record_index:02d}.safetensors"
    path = root / filename
    if path.exists() or path.is_symlink():
        raise ValueError("full-visible trace shard already exists")
    try:
        from safetensors.numpy import save_file
    except ImportError as error:  # pragma: no cover - required dependency
        raise RuntimeError("full-visible trace shards require safetensors") from error
    payload = {name: np.ascontiguousarray(arrays[name]) for name in _CAPTURE_TRACE_KEYS}
    temporary = root / f".{filename}.tmp-{os.getpid()}"
    save_file(payload, str(temporary))
    temporary.replace(path)
    descriptor = {
        "record_index": record_index,
        "record_id": record_id,
        "file": filename,
        "file_sha256": sha256_file(path),
        "format": "safetensors",
        "keys": sorted(_CAPTURE_TRACE_KEYS),
        "query_positions": list(query_positions),
        "tensor_sha256": dict(summary["tensor_sha256"]),
        "trace_sha256": summary["trace_sha256"],
        "reset_trace_sha256": reset_trace_sha256,
        "source_record_sha256": source_record_sha256,
        "output_evidence_sha256": output_evidence_sha256,
        "reset_output_evidence_sha256": reset_output_evidence_sha256,
        "schedule_rows_sha256": schedule_rows_sha256,
        "visible_entries_per_head": dict(summary["visible_entries_per_head"]),
        "invalid_regular_entries": summary["invalid_regular_entries"],
    }
    validate_full_visible_trace_shard(path, descriptor)
    return descriptor


def validate_full_visible_trace_shard(
    path: str | Path,
    descriptor: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Authenticate and fully revalidate one persisted trace shard."""

    file_sha256 = descriptor.get("file_sha256")
    if not _is_sha256(file_sha256):
        raise ValueError("full-visible trace shard descriptor is invalid")
    source = _checked_file(
        path,
        file_sha256,
        "full-visible trace shard",
    )
    if (
        source.name != descriptor.get("file")
        or descriptor.get("format") != "safetensors"
        or descriptor.get("keys") != sorted(_CAPTURE_TRACE_KEYS)
        or descriptor.get("query_positions") != list(_READ_POSITIONS)
        or not isinstance(descriptor.get("tensor_sha256"), Mapping)
        or set(descriptor["tensor_sha256"]) != set(_CAPTURE_TRACE_KEYS)
        or not all(_is_sha256(value) for value in descriptor["tensor_sha256"].values())
        or not all(
            _is_sha256(descriptor.get(name))
            for name in (
                "trace_sha256",
                "reset_trace_sha256",
                "source_record_sha256",
                "output_evidence_sha256",
                "reset_output_evidence_sha256",
                "schedule_rows_sha256",
            )
        )
        or descriptor.get("trace_sha256") != descriptor.get("reset_trace_sha256")
    ):
        raise ValueError("full-visible trace shard descriptor is invalid")
    try:
        from safetensors import safe_open
        from safetensors.numpy import load_file
    except ImportError as error:  # pragma: no cover - required dependency
        raise RuntimeError("full-visible trace shards require safetensors") from error
    with safe_open(source, framework="numpy") as handle:
        if sorted(handle.keys()) != sorted(_CAPTURE_TRACE_KEYS):
            raise ValueError("full-visible trace shard keys changed")
    loaded = {
        name: np.ascontiguousarray(value) for name, value in load_file(source).items()
    }
    summary = _trace_summary(loaded, _READ_POSITIONS)
    if (
        summary["trace_sha256"] != descriptor["trace_sha256"]
        or summary["tensor_sha256"] != descriptor["tensor_sha256"]
        or summary["visible_entries_per_head"]
        != descriptor.get("visible_entries_per_head")
        or summary["invalid_regular_entries"]
        != descriptor.get("invalid_regular_entries")
    ):
        raise ValueError("full-visible trace shard tensor changed")
    return loaded


def write_full_visible_trace_manifest(
    directory: str | Path,
    *,
    protocol: Mapping[str, str],
    shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write the immutable eight-record capture manifest."""

    root = _checked_directory(
        directory,
        "full-visible trace shard directory",
    )
    if (
        set(protocol) != {"path", "sha256"}
        or not _is_sha256(protocol["sha256"])
        or len(shards) != _RECORDS
        or [row.get("record_index") for row in shards] != list(range(_RECORDS))
    ):
        raise ValueError("full-visible trace manifest inputs are invalid")
    for descriptor in shards:
        validate_full_visible_trace_shard(
            root / str(descriptor.get("file", "")),
            descriptor,
        )
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _CAPTURE_EXPERIMENT,
        "protocol": dict(protocol),
        "format": "safetensors",
        "stored_tensors": sorted(_CAPTURE_TRACE_KEYS),
        "record_order": list(range(_RECORDS)),
        "shards": [dict(row) for row in shards],
        "confirmation_split_opened": False,
    }
    path = root / "manifest.json"
    if path.exists() or path.is_symlink():
        raise ValueError("full-visible trace manifest already exists")
    atomic_json(path, manifest)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "manifest": manifest,
    }


def load_stacked_full_visible_trace(
    manifest: str | Path,
    manifest_sha256: str,
    *,
    protocol: Mapping[str, str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and authenticate a complete cached train capture."""

    manifest_path = _checked_file(
        manifest,
        manifest_sha256,
        "full-visible trace manifest",
    )
    value = _read_json(manifest_path, "full-visible trace manifest")
    shards = value.get("shards")
    if (
        value.get("schema_version") != _SCHEMA_VERSION
        or value.get("experiment") != _CAPTURE_EXPERIMENT
        or value.get("format") != "safetensors"
        or value.get("stored_tensors") != sorted(_CAPTURE_TRACE_KEYS)
        or value.get("record_order") != list(range(_RECORDS))
        or not isinstance(shards, list)
        or len(shards) != _RECORDS
        or value.get("confirmation_split_opened") is not False
        or (protocol is not None and value.get("protocol") != protocol)
    ):
        raise ValueError("full-visible trace manifest contract changed")
    rows: dict[str, list[np.ndarray]] = {name: [] for name in _BASIS_TRACE_KEYS}
    for index, descriptor in enumerate(shards):
        if (
            not isinstance(descriptor, Mapping)
            or descriptor.get("record_index") != index
        ):
            raise ValueError("full-visible trace record order changed")
        loaded = validate_full_visible_trace_shard(
            manifest_path.parent / str(descriptor.get("file", "")),
            descriptor,
        )
        for name in _BASIS_TRACE_KEYS:
            rows[name].append(loaded[name])
    arrays = {
        name: np.ascontiguousarray(np.stack(values)) for name, values in rows.items()
    }
    return arrays, value


def _read_authenticated_train_records(
    binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if (
        not isinstance(binding, Mapping)
        or set(binding) != {"path", "sha256"}
        or not isinstance(binding.get("path"), str)
        or not isinstance(binding.get("sha256"), str)
    ):
        raise ValueError("full-visible train binding is invalid")
    path = _checked_file(
        binding["path"],
        binding["sha256"],
        "full-visible train split",
    )
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError
                records.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("full-visible train split is invalid") from error
    if len(records) != _RECORDS:
        raise ValueError("full-visible train record count changed")
    return records


def _audit_manifest_record_bindings(
    frozen: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, bool]:
    """Cross-bind every cached shard to authenticated train evidence."""

    historical = frozen.get("historical_bindings")
    schedule = frozen.get("schedule_contract")
    shards = manifest.get("shards")
    if (
        not isinstance(historical, Mapping)
        or not isinstance(schedule, Mapping)
        or not isinstance(shards, list)
        or len(shards) != _RECORDS
    ):
        raise ValueError("full-visible manifest audit inputs changed")
    per_record_schedule = schedule.get("per_record_rows_sha256")
    if (
        not isinstance(per_record_schedule, list)
        or len(per_record_schedule) != _RECORDS
        or not all(_is_sha256(value) for value in per_record_schedule)
    ):
        raise ValueError("full-visible frozen schedule roots changed")

    records = _read_authenticated_train_records(historical.get("train_split"))
    _head_path, head_manifest = _load_bound_json(
        historical.get("inherited_head_mass_manifest"),
        "full-visible inherited head-mass manifest",
    )
    _slot_path, slot_manifest = _load_bound_json(
        historical.get("slot_trace_manifest"),
        "full-visible inherited slot manifest",
    )
    _result_path, head_result = _load_bound_json(
        historical.get("inherited_head_mass_result"),
        "full-visible inherited head-mass result",
    )
    head_shards = head_manifest.get("shards")
    slot_shards = slot_manifest.get("shards")
    output_rows = head_result.get("base_output_authentication")
    if (
        head_manifest.get("confirmation_split_opened") is not False
        or slot_manifest.get("confirmation_split_opened") is not False
        or head_result.get("confirmation_split_opened") is not False
        or not isinstance(head_shards, list)
        or not isinstance(slot_shards, list)
        or not isinstance(output_rows, list)
        or len(head_shards) != _RECORDS
        or len(slot_shards) != _RECORDS
        or len(output_rows) != _RECORDS
    ):
        raise ValueError("full-visible inherited record evidence changed")

    for index, descriptor in enumerate(shards):
        record = records[index]
        inherited = head_shards[index]
        inherited_slot = slot_shards[index]
        output = output_rows[index]
        if not all(
            isinstance(value, Mapping)
            for value in (descriptor, record, inherited, inherited_slot, output)
        ):
            raise ValueError("full-visible record evidence is invalid")
        record_id = record.get("record_id")
        source_sha256 = sha256_json(record)
        output_sha256 = output.get("observed_output_evidence_sha256")
        reset_output_sha256 = output.get("reset_output_evidence_sha256")
        tensor_sha256 = descriptor.get("tensor_sha256")
        inherited_tensors = inherited.get("tensor_sha256")
        inherited_slot_tensors = inherited_slot.get("tensor_sha256")
        if (
            descriptor.get("record_index") != index
            or inherited.get("record_index") != index
            or inherited_slot.get("record_index") != index
            or output.get("record_index") != index
            or not isinstance(record_id, str)
            or descriptor.get("record_id") != record_id
            or inherited.get("record_id") != record_id
            or inherited_slot.get("record_id") != record_id
            or output.get("record_id") != record_id
            or not _is_sha256(source_sha256)
            or descriptor.get("source_record_sha256") != source_sha256
            or inherited.get("source_record_sha256") != source_sha256
            or inherited_slot.get("source_record_sha256") != source_sha256
            or output.get("source_record_sha256") != source_sha256
            or not _is_sha256(output_sha256)
            or not _is_sha256(reset_output_sha256)
            or descriptor.get("output_evidence_sha256") != output_sha256
            or descriptor.get("reset_output_evidence_sha256") != reset_output_sha256
            or inherited.get("output_evidence_sha256") != output_sha256
            or inherited.get("reset_output_evidence_sha256") != reset_output_sha256
            or inherited_slot.get("output_evidence_sha256") != output_sha256
            or inherited_slot.get("reset_output_evidence_sha256") != reset_output_sha256
            or descriptor.get("schedule_rows_sha256") != per_record_schedule[index]
            or not isinstance(tensor_sha256, Mapping)
            or not isinstance(inherited_tensors, Mapping)
            or not isinstance(inherited_slot_tensors, Mapping)
            or any(
                tensor_sha256.get(name) != inherited_tensors.get(name)
                for name in _BASE_TRACE_KEYS
            )
            or any(
                tensor_sha256.get(name) != inherited_slot_tensors.get(name)
                for name in _EPISODIC_SLOT_TRACE_KEYS
            )
        ):
            raise ValueError(
                f"full-visible manifest record {index} is not authenticated"
            )
    return {
        "train_records_rederived": True,
        "record_identity_and_source_roots": True,
        "schedule_roots": True,
        "output_and_reset_evidence_roots": True,
        "inherited_base_tensor_roots": True,
        "inherited_slot_tensor_roots": True,
        "confirmation_not_opened": True,
    }


def _summarize_oracle_arm(
    solved: FullVisibleOracleResult,
    *,
    arm: str,
    progression_authority: bool,
) -> dict[str, Any]:
    negative_tolerance = 1.0e-10 * np.maximum(1.0, solved.target_energy)
    if np.any(solved.objective < -negative_tolerance):
        raise ValueError(f"full-visible {arm} objective became negative")
    quadratic_error = np.maximum(solved.objective, 0.0)
    feasible_metrics = joint._recovery_metrics_from_energy(
        solved.target_energy,
        solved.direct_error_energy,
        batch_shape=solved.batch_shape,
    )
    optimistic_error = np.maximum(
        quadratic_error - solved.objective_gap_upper_bound,
        0.0,
    )
    optimistic_metrics = joint._recovery_metrics_from_energy(
        solved.target_energy,
        optimistic_error,
        batch_shape=solved.batch_shape,
    )
    nonregression_tolerance = 2.0e-6 * np.maximum(
        1.0,
        solved.target_energy,
    )
    parity_tolerance = 1.0e-9 * np.maximum(1.0, solved.target_energy)
    qualification = {
        "objective_never_worse_than_native_base": bool(
            np.all(
                solved.direct_error_energy
                <= solved.target_energy + nonregression_tolerance
            )
        ),
        "quadratic_direct_parity": bool(
            np.all(
                np.abs(quadratic_error - solved.direct_error_energy) <= parity_tolerance
            )
        ),
        "deterministic_solver_replay_exact": (solved.deterministic_replay_exact),
        "objective_gap_certificate_available": bool(
            np.isfinite(solved.objective_gap_upper_bound).all()
            and np.all(solved.objective_gap_upper_bound >= 0.0)
        ),
        # The solve exists only after `_run_oracle_from_arrays` has enforced
        # the frozen elementwise absolute-plus-relative Wo tolerance.
        "authenticated_base_projection_matches_Wo": True,
    }
    qualification["passed"] = all(qualification.values())
    argmax = np.argmax(solved.coefficients, axis=-1)
    histogram = np.bincount(
        argmax.reshape(-1),
        minlength=solved.component_count,
    )
    return {
        "arm": arm,
        "components_per_head": solved.component_count,
        "progression_authority": progression_authority,
        "exact_native_anchor_included": (solved.exact_native_anchor_included),
        "diagnostic_only": solved.diagnostic_only,
        "coefficient_sha256": hashlib.sha256(
            solved.coefficients.tobytes(order="C")
        ).hexdigest(),
        "objective_sha256": hashlib.sha256(
            solved.objective.tobytes(order="C")
        ).hexdigest(),
        "objective_gap_upper_bound_sha256": hashlib.sha256(
            solved.objective_gap_upper_bound.tobytes(order="C")
        ).hexdigest(),
        "argmax_component_histogram": {
            str(index): int(count) for index, count in enumerate(histogram)
        },
        "solver": {
            "minimum_iterations": int(np.min(solved.iterations)),
            "mean_iterations": float(np.mean(solved.iterations)),
            "maximum_iterations": int(np.max(solved.iterations)),
            "converged_rows": int(np.count_nonzero(solved.converged)),
            "rows": int(solved.converged.size),
            "maximum_relative_objective_gap": (solved.maximum_relative_objective_gap),
            "maximum_objective_gap_upper_bound": float(
                np.max(solved.objective_gap_upper_bound)
            ),
        },
        "trace_qualification": {
            "base_reconstruction_max_abs": (solved.base_reconstruction_max_abs),
            "regular_mass_reconstruction_max_abs": (
                solved.regular_mass_reconstruction_max_abs
            ),
            "regular_component_reconstruction_max_abs": (
                solved.regular_component_reconstruction_max_abs
            ),
            "episodic_mass_reconstruction_max_abs": (
                solved.episodic_mass_reconstruction_max_abs
            ),
            "episodic_component_reconstruction_max_abs": (
                solved.episodic_component_reconstruction_max_abs
            ),
            "base_component_reconstruction_max_abs": (
                solved.base_component_reconstruction_max_abs
            ),
            "authenticated_base_projection_max_abs": (
                solved.authenticated_base_projection_max_abs
            ),
            "quadratic_direct_error_energy_max_abs": (
                solved.quadratic_direct_error_energy_max_abs
            ),
            "invalid_regular_entries": solved.invalid_regular_entries,
        },
        "qualification": qualification,
        "feasible_solution_metrics": feasible_metrics,
        "optimistic_recovery_upper_bound_metrics": optimistic_metrics,
        "feasible_gate_passed": bool(
            qualification["passed"] and feasible_metrics["passed"]
        ),
        "optimistic_gate_passed": bool(
            qualification["passed"] and optimistic_metrics["passed"]
        ),
    }


def run_cached_full_visible_screen(
    arrays: Mapping[str, np.ndarray],
    output_projection: np.ndarray,
    *,
    include_nested_diagnostics: bool = True,
    row_batch_size: int = _ROW_BATCH_SIZE,
) -> dict[str, Any]:
    """Run both authoritative arms and optional non-authoritative diagnostics."""

    constructible = run_full_visible_oracle_from_arrays(
        arrays,
        output_projection,
        include_exact_native_anchor=False,
        row_batch_size=row_batch_size,
    )
    optimistic = run_full_visible_oracle_from_arrays(
        arrays,
        output_projection,
        include_exact_native_anchor=True,
        row_batch_size=row_batch_size,
    )
    constructible_summary = _summarize_oracle_arm(
        constructible,
        arm="constructible_28_way",
        progression_authority=True,
    )
    optimistic_summary = _summarize_oracle_arm(
        optimistic,
        arm="optimistic_29_way_native_anchor_hull",
        progression_authority=True,
    )
    diagnostics: dict[str, Any] = {}
    if include_nested_diagnostics:
        for component_count in _NESTED_DIAGNOSTIC_COMPONENTS:
            solved = run_nested_visible_oracle_from_arrays(
                arrays,
                output_projection,
                component_count=component_count,
                row_batch_size=row_batch_size,
            )
            diagnostics[str(component_count)] = _summarize_oracle_arm(
                solved,
                arm=f"nested_{component_count}_way_diagnostic",
                progression_authority=False,
            )
    constructible_pass = constructible_summary["feasible_gate_passed"]
    optimistic_pass = optimistic_summary["optimistic_gate_passed"]
    optimistic_qualification_passed = optimistic_summary["qualification"]["passed"]
    optimistic_bound_passed = optimistic_summary[
        "optimistic_recovery_upper_bound_metrics"
    ]["passed"]
    decisive_failure = bool(
        optimistic_qualification_passed and not optimistic_bound_passed
    )
    if constructible_pass:
        status = _RESULT_PASSED_STATUS
        next_step = (
            "freeze a train-only causal 28-logit correction selector with "
            "complete CPU runtime accounting"
        )
    elif decisive_failure:
        status = _RESULT_FAILED_STATUS
        next_step = (
            "close all normalized reweighting of currently visible values "
            "and add a different memory/value generator"
        )
    else:
        status = "train_full_visible_simplex_gate_inconclusive"
        next_step = (
            "improve the certified constructible solve; the optimistic-only "
            "pass cannot authorize a selector"
        )
    return {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _RESULT_EXPERIMENT,
        "status": status,
        "authoritative_arms": {
            "constructible": constructible_summary,
            "optimistic": optimistic_summary,
        },
        "nested_diagnostics": diagnostics,
        "decision": {
            "train_full_visible_constructible_gate_passed": (constructible_pass),
            "certified_optimistic_gate_passed": optimistic_pass,
            "failure_is_decisive": decisive_failure,
            "train_only_causal_selector_authorized": constructible_pass,
            "development_authorized": False,
            "confirmation_authorized": False,
            "semantic_or_M3_gate_passed": False,
            "next_step": next_step,
        },
        "resource_contract": {
            "fixed_attention_state_bytes": 10534912,
            "fixed_combined_attention_and_episodic_traffic_bytes": 714866688,
            "fixed_fraction_of_dense_full_context_KV": (0.33030523255813954),
            "new_KV_state_or_read_traffic_bytes": 0,
            "trace_and_solver_evaluator_only": True,
            "future_selector_not_counted_by_this_capacity_screen": True,
        },
        "confirmation_split_opened": False,
    }


def _load_authenticated_output_projection(
    protocol: Mapping[str, Any],
) -> np.ndarray:
    contract = protocol.get("output_projection")
    if (
        not isinstance(contract, Mapping)
        or not isinstance(contract.get("source"), str)
        or not _is_sha256(contract.get("file_sha256"))
        or not isinstance(contract.get("tensor_sha256"), Mapping)
    ):
        raise ValueError("full-visible output projection contract is invalid")
    source = _checked_file(
        contract["source"],
        contract["file_sha256"],
        "full-visible output projection source",
    )
    projection, tensor_hashes = mass._load_output_projections({"non_mlp_path": source})
    if tensor_hashes != contract["tensor_sha256"]:
        raise ValueError("full-visible output projection tensors changed")
    return projection


def solve_cached_full_visible_capture(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    manifest: str | Path,
    manifest_sha256: str,
    out: str | Path,
    include_nested_diagnostics: bool = True,
    row_batch_size: int = _ROW_BATCH_SIZE,
) -> dict[str, Any]:
    """Authenticate a frozen capture and write the complete train screen."""

    protocol_path = _checked_file(
        protocol,
        protocol_sha256,
        "full-visible protocol",
    )
    frozen = _read_json(protocol_path, "full-visible protocol")
    if (
        frozen.get("schema_version") != _SCHEMA_VERSION
        or frozen.get("experiment") != _PROTOCOL_EXPERIMENT
        or frozen.get("status") != _PROTOCOL_STATUS
        or frozen.get("confirmation_split_opened") is not False
        or frozen.get("source_sha256") != _source_inventory()
        or frozen.get("oracle_method", {}).get("row_batch_size") != _ROW_BATCH_SIZE
        or row_batch_size != _ROW_BATCH_SIZE
    ):
        raise ValueError("full-visible frozen protocol changed")
    protocol_binding = _binding(protocol_path, protocol_sha256)
    arrays, manifest_value = load_stacked_full_visible_trace(
        manifest,
        manifest_sha256,
        protocol=protocol_binding,
    )
    record_authentication = _audit_manifest_record_bindings(
        frozen,
        manifest_value,
    )
    projection = _load_authenticated_output_projection(frozen)
    result = run_cached_full_visible_screen(
        arrays,
        projection,
        include_nested_diagnostics=include_nested_diagnostics,
        row_batch_size=row_batch_size,
    )
    result["protocol"] = protocol_binding
    result["trace_manifest"] = {
        "path": str(Path(manifest).expanduser().resolve()),
        "sha256": manifest_sha256,
        "record_count": len(manifest_value["shards"]),
    }
    result["authenticated_confirmation_descriptor"] = dict(
        frozen["authenticated_confirmation_descriptor"]
    )
    result["execution_contract"] = {
        "row_batch_size": row_batch_size,
        "mathematical_feasible_set_unchanged_by_batching": True,
        "row_batch_size_frozen_for_numerical_reproducibility": True,
    }
    result["record_authentication"] = record_authentication
    result["post_solve_authentication"] = {
        "protocol": sha256_file(protocol_path) == protocol_sha256,
        "manifest": (
            sha256_file(Path(manifest).expanduser().resolve()) == manifest_sha256
        ),
        "output_projection": True,
        "record_authentication": all(record_authentication.values()),
        "source_inventory": frozen["source_sha256"] == _source_inventory(),
        "confirmation_not_opened": True,
    }
    if not all(result["post_solve_authentication"].values()):
        raise ValueError("full-visible post-solve authentication failed")
    output = _new_output_path(out, "full-visible result output")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return result
