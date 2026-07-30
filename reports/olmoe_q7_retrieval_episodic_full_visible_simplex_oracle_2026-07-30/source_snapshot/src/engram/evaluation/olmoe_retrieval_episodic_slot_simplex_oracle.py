"""Train-only per-slot episodic value-basis capacity oracle.

This experiment is the first attention-capacity test in the retrieval branch
that can change directions inside the already-read eight-value episodic span.
For each selected query head it exposes a nine-element convex basis:

* the current regular-cache conditional mean; and
* the eight exact BF16-decoded episodic slot values, in read-span order.

An additional exact native-output anchor removes trace-regrouping roundoff from
the feasible baseline.  The oracle chooses all ten-component head mixtures
jointly after ``o_proj``.  Its feasible set is a Cartesian product of
simplices, so it is an optimistic superset of every result obtainable by
adding arbitrary finite per-slot logit biases while keeping the same value
set.  A Frank--Wolfe objective-gap certificate supplies an optimistic recovery
upper bound even when the numerical solver stops before its requested target.

The module is deliberately confirmation-blind.  It inherits the authenticated
confirmation descriptor by value and never opens, resolves, stats, or hashes
the corresponding file.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import engram.evaluation.olmoe_product_simplex_solver as simplex_solver
import engram.evaluation.olmoe_retrieval_episodic_head_mass_oracle as mass
import engram.evaluation.olmoe_retrieval_episodic_joint_gamma_oracle as joint
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.utils import atomic_json, sha256_file, sha256_json


_SCHEMA_VERSION = 1
_EXPECTED_JOINT_PROTOCOL_SHA256 = (
    "aa03a71e3dd9e1fbb413a7773d57189c41029c26b2f4372b2fb7a26744305d24"
)
_EXPECTED_JOINT_RESULT_SHA256 = (
    "1329a51bac71cb81f44494c8ef70cb23a631eacebac5540b39e2e98ed5e30ea5"
)
_SLOT_COPY_SYMBOL = "engram_olmoe_token_copy_last_episodic_slot_trace_v1"
_PROTOCOL_EXPERIMENT = "olmoe_q7_retrieval_episodic_slot_simplex_protocol"
_PARITY_EXPERIMENT = "olmoe_q7_retrieval_episodic_slot_simplex_trace_parity"
_RESULT_EXPERIMENT = "olmoe_q7_retrieval_episodic_slot_simplex_train_screen"
_PROTOCOL_STATUS = "frozen_before_per_slot_simplex_train_execution"
_PARITY_STATUS = "per_slot_trace_parity_passed"
_SLOTS = mass._EPISODIC_POLICY["span_size"]
_CONSTRUCTIBLE_COMPONENTS = _SLOTS + 1
_OPTIMISTIC_COMPONENTS = _SLOTS + 2
_QUERY_HEADS = mass._QUERY_HEADS
_HEAD_DIMENSION = mass._HEAD_DIMENSION
_LAYERS = mass._LAYERS
_HIDDEN_SIZE = mass._HIDDEN_SIZE
_RECORDS = mass._RECORDS
_POSITIONS = mass._POSITIONS
_READ_POSITIONS = mass._READ_POSITIONS
_BASE_TRACE_KEYS = (
    "base_attention_output",
    "regular_component",
    "episodic_component",
    "regular_mass",
    "episodic_mass",
    "base_projected",
    "target_residual",
)
_SLOT_TRACE_KEYS = ("slot_mass", "slot_values")
_MAXIMUM_ITERATIONS = 512
_RELATIVE_GAP_TOLERANCE = 1.0e-7
_ROW_BATCH_SIZE = 16
_SOURCE_FILES = (
    "native/include/engram/streaming_attention.h",
    "native/src/streaming_attention.cpp",
    "native/include/engram/olmoe_token_runtime.h",
    "native/src/olmoe_token_runtime.cpp",
    "native/include/engram/olmoe_token_runtime_c.h",
    "native/src/olmoe_token_runtime_c.cpp",
    "src/engram/runtime/olmoe_native.py",
    "src/engram/evaluation/olmoe_product_simplex_solver.py",
    "src/engram/evaluation/olmoe_retrieval_episodic_slot_simplex_oracle.py",
)


def _progress(message: str) -> None:
    print(
        f"[retrieval-episodic-slot-simplex] {message}",
        file=sys.stderr,
        flush=True,
    )


def _source_inventory() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    return {name: sha256_file(repository / name) for name in _SOURCE_FILES}


@dataclass(frozen=True)
class SlotBasis:
    """Validated per-head value basis and native base simplex point."""

    components: np.ndarray
    correction_basis: np.ndarray
    base_coefficients: np.ndarray
    base_heads: np.ndarray
    target_residual: np.ndarray
    base_reconstruction_max_abs: float
    traced_partition_reconstruction_max_abs: float
    episodic_component_reconstruction_max_abs: float
    mass_partition_max_abs: float


@dataclass(frozen=True)
class SlotSimplexOracleResult:
    coefficients: np.ndarray
    target_energy: np.ndarray
    objective: np.ndarray
    objective_gap_upper_bound: np.ndarray
    direct_error_energy: np.ndarray
    iterations: np.ndarray
    converged: np.ndarray
    maximum_relative_objective_gap: float
    base_reconstruction_max_abs: float
    traced_partition_reconstruction_max_abs: float
    episodic_component_reconstruction_max_abs: float
    mass_partition_max_abs: float
    quadratic_direct_error_energy_max_abs: float
    deterministic_replay_exact: bool
    batch_shape: tuple[int, int, int]


def _as_float32(array: np.ndarray, name: str) -> np.ndarray:
    value = np.ascontiguousarray(array, dtype=np.float32)
    if not value.size or not np.isfinite(value).all():
        raise ValueError(f"slot-simplex {name} tensor is invalid")
    return value


def build_slot_basis(
    arrays: Mapping[str, np.ndarray],
    *,
    query_heads: int = _QUERY_HEADS,
    slots: int = _SLOTS,
    include_exact_native_anchor: bool = True,
) -> SlotBasis:
    """Validate a batch and construct one of the two frozen value bases."""

    required = {*_BASE_TRACE_KEYS, *_SLOT_TRACE_KEYS}
    if not required.issubset(arrays):
        raise ValueError("slot-simplex trace tensors are incomplete")
    base = _as_float32(arrays["base_attention_output"], "base")
    regular = _as_float32(arrays["regular_component"], "regular component")
    episodic = _as_float32(arrays["episodic_component"], "episodic component")
    regular_mass = _as_float32(arrays["regular_mass"], "regular mass")
    episodic_mass = _as_float32(arrays["episodic_mass"], "episodic mass")
    target = _as_float32(arrays["target_residual"], "target residual")
    slot_mass = _as_float32(arrays["slot_mass"], "slot mass")
    slot_values = _as_float32(arrays["slot_values"], "slot values")
    if (
        query_heads <= 0
        or slots <= 0
        or base.ndim < 2
        or base.shape != regular.shape
        or base.shape != episodic.shape
        or base.shape != target.shape
        or base.shape[-1] % query_heads
    ):
        raise ValueError("slot-simplex base tensor shapes are invalid")
    head_dimension = base.shape[-1] // query_heads
    prefix = base.shape[:-1]
    mass_shape = prefix + (query_heads,)
    if (
        regular_mass.shape != mass_shape
        or episodic_mass.shape != mass_shape
        or slot_mass.shape != mass_shape + (slots,)
        or slot_values.shape != mass_shape + (slots, head_dimension)
        or np.any(regular_mass <= np.float32(0.0))
        or np.any(episodic_mass <= np.float32(0.0))
        or np.any(slot_mass < np.float32(0.0))
    ):
        raise ValueError("slot-simplex mass or slot tensor shapes are invalid")

    mass_partition_error = float(
        np.max(np.abs(regular_mass + episodic_mass - np.float32(1.0)))
    )
    slot_mass_error = float(
        np.max(np.abs(np.sum(slot_mass, axis=-1) - episodic_mass))
    )
    if mass_partition_error > 2.0e-5 or slot_mass_error > 2.0e-5:
        raise ValueError("slot-simplex masses do not reconstruct the partition")

    head_shape = prefix + (query_heads, head_dimension)
    base_heads = base.reshape(head_shape)
    regular_heads = regular.reshape(head_shape)
    episodic_heads = episodic.reshape(head_shape)
    episodic_from_slots = np.einsum(
        "...hs,...hsd->...hd",
        slot_mass,
        slot_values,
        optimize=True,
    )
    episodic_error = float(np.max(np.abs(episodic_from_slots - episodic_heads)))
    if episodic_error > 5.0e-5:
        raise ValueError("slot-simplex slots do not reconstruct episodic output")

    traced_partition_error = float(
        np.max(np.abs(regular_heads + episodic_heads - base_heads))
    )
    if traced_partition_error > 5.0e-5:
        raise ValueError("slot-simplex traced partition does not reconstruct base")
    regular_mean = regular_heads / regular_mass[..., None]
    if include_exact_native_anchor:
        components = np.concatenate(
            (
                base_heads[..., None, :],
                regular_mean[..., None, :],
                slot_values,
            ),
            axis=-2,
        )
    else:
        components = np.concatenate(
            (regular_mean[..., None, :], slot_values),
            axis=-2,
        )
    raw_base_coefficients = np.concatenate(
        (regular_mass[..., None], slot_mass),
        axis=-1,
    )
    simplex_error = float(
        np.max(
            np.abs(
                np.sum(raw_base_coefficients, axis=-1, dtype=np.float64)
                - 1.0
            )
        )
    )
    if (
        np.any(raw_base_coefficients < np.float32(0.0))
        or simplex_error > 2.0e-5
        or not np.isfinite(regular_mean).all()
        or not np.isfinite(components).all()
    ):
        raise ValueError("slot-simplex native base point is infeasible")
    if include_exact_native_anchor:
        base_coefficients = np.zeros(
            components.shape[:-1],
            dtype=np.float64,
        )
        # Component zero is the exact native head output.  This anchor makes
        # the untouched route exactly feasible even though evaluator-only
        # component regrouping is qualified only within a float32 tolerance.
        base_coefficients[..., 0] = 1.0
    else:
        base_coefficients = np.ascontiguousarray(
            raw_base_coefficients,
            dtype=np.float64,
        )
        base_coefficients /= np.sum(
            base_coefficients,
            axis=-1,
            keepdims=True,
            dtype=np.float64,
        )
    reconstructed = np.einsum(
        "...hc,...hcd->...hd",
        base_coefficients,
        components,
        optimize=True,
    )
    base_error = float(np.max(np.abs(reconstructed - base_heads)))
    if base_error > 7.5e-5:
        raise ValueError("slot-simplex basis does not reconstruct native output")
    correction_basis = (
        components.astype(np.float64)
        - base_heads.astype(np.float64)[..., None, :]
    )
    return SlotBasis(
        components=np.ascontiguousarray(components, dtype=np.float64),
        correction_basis=np.ascontiguousarray(
            correction_basis,
            dtype=np.float64,
        ),
        base_coefficients=np.ascontiguousarray(
            base_coefficients,
            dtype=np.float64,
        ),
        base_heads=np.ascontiguousarray(base_heads, dtype=np.float64),
        target_residual=np.ascontiguousarray(target, dtype=np.float64),
        base_reconstruction_max_abs=base_error,
        traced_partition_reconstruction_max_abs=traced_partition_error,
        episodic_component_reconstruction_max_abs=episodic_error,
        mass_partition_max_abs=max(mass_partition_error, slot_mass_error),
    )


def _project_head_basis(
    basis: np.ndarray,
    output_projection: np.ndarray,
) -> np.ndarray:
    values = np.ascontiguousarray(basis, dtype=np.float64)
    weights = np.ascontiguousarray(output_projection, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("slot-simplex projected basis must be rank four")
    _batch, heads, _components, head_dimension = values.shape
    hidden = heads * head_dimension
    if (
        weights.shape != (hidden, hidden)
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
    ):
        raise ValueError("slot-simplex output projection shape is invalid")
    head_weights = weights.reshape(hidden, heads, head_dimension).transpose(1, 0, 2)
    projected = np.einsum(
        "nhci,hoi->nhco",
        values,
        head_weights,
        optimize=True,
    )
    if not np.isfinite(projected).all():
        raise ValueError("slot-simplex projected basis is non-finite")
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
        raise ValueError("slot-simplex quadratic tensors are invalid")
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
        raise ValueError("slot-simplex target energy must be positive")
    return (
        np.ascontiguousarray(gram, dtype=np.float64),
        np.ascontiguousarray(linear, dtype=np.float64),
        np.ascontiguousarray(energy, dtype=np.float64),
    )


def _direct_error_energy(
    basis: SlotBasis,
    coefficients: np.ndarray,
    output_projection: np.ndarray,
) -> np.ndarray:
    selected = np.ascontiguousarray(coefficients, dtype=np.float64)
    expected = basis.base_coefficients.shape
    if (
        selected.shape != expected
        or np.any(selected < -1.0e-12)
        or np.max(np.abs(np.sum(selected, axis=-1) - 1.0)) > 1.0e-10
    ):
        raise ValueError("slot-simplex selected coefficients are infeasible")
    candidate = np.einsum(
        "nhc,nhcd->nhd",
        selected,
        basis.components,
        optimize=True,
    )
    delta = (candidate - basis.base_heads).reshape(
        selected.shape[0],
        -1,
    )
    correction = delta @ np.ascontiguousarray(
        output_projection,
        dtype=np.float64,
    ).T
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
            sliced[name] = value[:, :, layer].reshape(-1, _QUERY_HEADS)[begin:end]
        else:
            sliced[name] = value[:, :, layer].reshape(-1, _HIDDEN_SIZE)[begin:end]
    sliced["slot_mass"] = np.ascontiguousarray(
        arrays["slot_mass"][:, :, layer].reshape(
            -1,
            _QUERY_HEADS,
            _SLOTS,
        )[begin:end]
    )
    sliced["slot_values"] = np.ascontiguousarray(
        arrays["slot_values"][:, :, layer].reshape(
            -1,
            _QUERY_HEADS,
            _SLOTS,
            _HEAD_DIMENSION,
        )[begin:end]
    )
    return sliced


def run_slot_simplex_oracle_from_arrays(
    arrays: Mapping[str, np.ndarray],
    output_projection: np.ndarray,
    *,
    row_batch_size: int = _ROW_BATCH_SIZE,
    maximum_iterations: int = _MAXIMUM_ITERATIONS,
    relative_gap_tolerance: float = _RELATIVE_GAP_TOLERANCE,
    include_exact_native_anchor: bool = True,
) -> SlotSimplexOracleResult:
    """Solve the full train tensor in bounded batches and replay every solve."""

    if set(arrays) != {*_BASE_TRACE_KEYS, *_SLOT_TRACE_KEYS}:
        raise ValueError("slot-simplex stacked trace keys changed")
    base = np.ascontiguousarray(arrays["base_attention_output"])
    records, read_rows, layers, hidden = base.shape
    weights = np.ascontiguousarray(output_projection, dtype=np.float32)
    if (
        (records, read_rows, layers, hidden)
        != (_RECORDS, len(_READ_POSITIONS), _LAYERS, _HIDDEN_SIZE)
        or weights.shape != (_LAYERS, _HIDDEN_SIZE, _HIDDEN_SIZE)
        or row_batch_size <= 0
        or maximum_iterations <= 0
        or not np.isfinite(relative_gap_tolerance)
        or relative_gap_tolerance <= 0.0
    ):
        raise ValueError("slot-simplex full oracle configuration is invalid")

    total_rows = records * read_rows * layers
    rows_per_layer = records * read_rows
    component_count = (
        _OPTIMISTIC_COMPONENTS
        if include_exact_native_anchor
        else _CONSTRUCTIBLE_COMPONENTS
    )
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
    maximum_relative_gap = 0.0
    maximum_base_error = 0.0
    maximum_traced_partition_error = 0.0
    maximum_episodic_error = 0.0
    maximum_mass_error = 0.0
    maximum_direct_difference = 0.0
    replay_exact = True

    for layer in range(layers):
        _progress(f"solving per-slot basis layer {layer + 1}/{layers}")
        layer_weights = weights[layer]
        for begin in range(0, rows_per_layer, row_batch_size):
            end = min(begin + row_batch_size, rows_per_layer)
            batch_arrays = _slice_layer_batch(
                arrays,
                layer=layer,
                begin=begin,
                end=end,
            )
            basis = build_slot_basis(
                batch_arrays,
                query_heads=_QUERY_HEADS,
                slots=_SLOTS,
                include_exact_native_anchor=include_exact_native_anchor,
            )
            projected = _project_head_basis(
                basis.correction_basis,
                layer_weights,
            )
            gram, linear, energy = _quadratic_from_projected_basis(
                projected,
                basis.target_residual,
            )
            solved = simplex_solver.solve_product_simplex_least_squares(
                gram,
                linear,
                energy,
                basis.base_coefficients,
                max_iterations=maximum_iterations,
                relative_tolerance=relative_gap_tolerance,
            )
            replay = simplex_solver.solve_product_simplex_least_squares(
                gram,
                linear,
                energy,
                basis.base_coefficients,
                max_iterations=maximum_iterations,
                relative_tolerance=relative_gap_tolerance,
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
                    np.array_equal(solved.row_converged, replay.row_converged),
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
            maximum_relative_gap = max(
                maximum_relative_gap,
                float(solved.max_relative_gap),
            )
            maximum_base_error = max(
                maximum_base_error,
                basis.base_reconstruction_max_abs,
            )
            maximum_traced_partition_error = max(
                maximum_traced_partition_error,
                basis.traced_partition_reconstruction_max_abs,
            )
            maximum_episodic_error = max(
                maximum_episodic_error,
                basis.episodic_component_reconstruction_max_abs,
            )
            maximum_mass_error = max(
                maximum_mass_error,
                basis.mass_partition_max_abs,
            )
            maximum_direct_difference = max(
                maximum_direct_difference,
                float(np.max(np.abs(direct - solved.objective))),
            )
    if not replay_exact:
        raise ValueError("slot-simplex deterministic solver replay changed")
    return SlotSimplexOracleResult(
        coefficients=coefficients,
        target_energy=target_energy,
        objective=objective,
        objective_gap_upper_bound=objective_gap,
        direct_error_energy=direct_energy,
        iterations=iterations,
        converged=converged,
        maximum_relative_objective_gap=maximum_relative_gap,
        base_reconstruction_max_abs=maximum_base_error,
        traced_partition_reconstruction_max_abs=(
            maximum_traced_partition_error
        ),
        episodic_component_reconstruction_max_abs=maximum_episodic_error,
        mass_partition_max_abs=maximum_mass_error,
        quadratic_direct_error_energy_max_abs=maximum_direct_difference,
        deterministic_replay_exact=True,
        batch_shape=(records, read_rows, layers),
    )


def _validate_joint_failure(
    value: Any,
    *,
    protocol_path: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("slot-simplex joint-gamma result is invalid")
    decision = value.get("decision")
    continuous = value.get("continuous_box_diagnostic")
    discrete = value.get("discrete_local_oracle")
    post = value.get("post_run_authentication")
    if (
        value.get("schema_version") != joint._SCHEMA_VERSION
        or value.get("experiment") != joint._RESULT_EXPERIMENT
        or value.get("status")
        != "train_episodic_joint_gamma_oracle_gate_failed"
        or value.get("protocol")
        != {
            "path": str(protocol_path),
            "sha256": _EXPECTED_JOINT_PROTOCOL_SHA256,
        }
        or value.get("confirmation_split_opened") is not False
        or not isinstance(decision, Mapping)
        or decision.get("train_joint_gamma_capacity_gate_passed") is not False
        or decision.get("continuous_relaxation_optimistic_gate_passed") is not False
        or decision.get("semantic_or_M3_gate_passed") is not False
        or decision.get("development_authorized") is not False
        or decision.get("confirmation_authorized") is not False
        or not isinstance(continuous, Mapping)
        or continuous.get("objective_gap_certificate_available") is not True
        or continuous.get("optimistic_recovery_upper_bound_metrics", {}).get(
            "passed"
        )
        is not False
        or not isinstance(discrete, Mapping)
        or discrete.get("passed") is not False
        or not isinstance(post, Mapping)
        or not post
        or not all(check is True for check in post.values())
    ):
        raise ValueError("slot-simplex inherited joint-gamma failure changed")
    return {
        "status": value["status"],
        "continuous_optimistic_global_recovery": continuous[
            "optimistic_recovery_upper_bound_metrics"
        ]["global"]["recovery"],
        "discrete_direct_global_recovery": discrete["direct_float32_metrics"][
            "global"
        ]["recovery"],
        "confirmation_split_opened": False,
    }


def _require_slot_trace_symbols(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("slot-simplex native trace library is invalid")
    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise ValueError("slot-simplex native trace library cannot be loaded") from exc
    required = (
        mass.capacity._TRACE_OPEN_SYMBOL,
        mass.capacity._TRACE_COPY_SYMBOL,
        mass._MASS_COPY_SYMBOL,
        _SLOT_COPY_SYMBOL,
    )
    missing = [name for name in required if not hasattr(library, name)]
    if missing:
        raise ValueError(
            "slot-simplex native trace library is missing symbols: "
            + ", ".join(missing)
        )


def _authenticate_frozen_joint_evidence(
    joint_protocol: str | Path,
    joint_protocol_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate exact historical roots without replaying stale source hashes.

    The inherited protocols intentionally bound their then-current source
    inventories.  This experiment adds evaluator-only trace APIs to several of
    those files, so rebuilding an old protocol from the current checkout would
    fail even though the exact historical JSON and tensor roots remain intact.
    We instead authenticate those immutable roots directly, then revalidate
    every train input and package artifact used by the new execution.
    """

    protocol_path = mass.capacity.bias.episodic._checked_file(
        joint_protocol,
        joint_protocol_sha256,
        "slot-simplex frozen joint protocol",
    )
    frozen = mass.capacity.bias.rank.retrieval._read_json(
        protocol_path,
        "slot-simplex frozen joint protocol",
    )
    if (
        frozen.get("schema_version") != joint._SCHEMA_VERSION
        or frozen.get("experiment") != joint._PROTOCOL_EXPERIMENT
        or frozen.get("status") != joint._PROTOCOL_STATUS
        or frozen.get("confirmation_split_opened") is not False
        or frozen.get("scope", {}).get("confirmation_file_access_permitted")
        is not False
    ):
        raise ValueError("slot-simplex frozen joint protocol contract changed")

    head_mass_binding = frozen.get("head_mass_protocol")
    head_mass_result_binding = frozen.get("head_mass_result")
    manifest_binding = frozen.get("cached_trace_manifest")
    checkpoint_binding = frozen.get("training_checkpoint")
    if not all(
        isinstance(binding, Mapping)
        for binding in (
            head_mass_binding,
            head_mass_result_binding,
            manifest_binding,
            checkpoint_binding,
        )
    ):
        raise ValueError("slot-simplex inherited bindings are invalid")
    head_mass_path = mass.capacity.bias.episodic._checked_file(
        head_mass_binding.get("path"),
        head_mass_binding.get("sha256"),
        "slot-simplex head-mass protocol",
    )
    head_mass_protocol = mass.capacity.bias.rank.retrieval._read_json(
        head_mass_path,
        "slot-simplex head-mass protocol",
    )
    head_mass_result_path = mass.capacity.bias.episodic._checked_file(
        head_mass_result_binding.get("path"),
        head_mass_result_binding.get("sha256"),
        "slot-simplex head-mass result",
    )
    head_mass_result = mass.capacity.bias.rank.retrieval._read_json(
        head_mass_result_path,
        "slot-simplex head-mass result",
    )
    joint._validate_head_mass_failure(
        head_mass_result,
        protocol_path=head_mass_path,
    )
    manifest_path = mass.capacity.bias.episodic._checked_file(
        manifest_binding.get("path"),
        manifest_binding.get("sha256"),
        "slot-simplex inherited trace manifest",
    )
    manifest = mass.capacity.bias.rank.retrieval._read_json(
        manifest_path,
        "slot-simplex inherited trace manifest",
    )
    if (
        manifest.get("schema_version") != mass._SCHEMA_VERSION
        or manifest.get("experiment") != mass._RESULT_EXPERIMENT
        or manifest.get("record_order") != list(range(_RECORDS))
        or manifest.get("shards")
        != head_mass_result.get("trace_manifest", {}).get("shards")
        or manifest.get("confirmation_split_opened") is not False
    ):
        raise ValueError("slot-simplex inherited trace manifest changed")

    checkpoint_path = mass.capacity.bias.episodic._checked_file(
        checkpoint_binding.get("path"),
        checkpoint_binding.get("sha256"),
        "slot-simplex training checkpoint",
    )
    checkpoint = mass.capacity.bias.rank.retrieval._read_json(
        checkpoint_path,
        "slot-simplex training checkpoint",
    )
    training = checkpoint.get("training")
    if (
        not isinstance(training, Mapping)
        or sha256_json(training) != checkpoint_binding.get("training_sha256")
        or sha256_json(training)
        != frozen.get("training_checkpoint_payload_sha256")
        or checkpoint.get("confirmation_split_opened") is not False
    ):
        raise ValueError("slot-simplex training checkpoint changed")
    selector_binding = checkpoint.get("protocol")
    if not isinstance(selector_binding, Mapping):
        raise ValueError("slot-simplex selector protocol binding is invalid")
    selector_path = mass.capacity.bias.episodic._checked_file(
        selector_binding.get("path"),
        selector_binding.get("sha256"),
        "slot-simplex selector protocol",
    )
    selector = mass.capacity.bias.rank.retrieval._read_json(
        selector_path,
        "slot-simplex selector protocol",
    )
    corpus = selector.get("corpus")
    if not isinstance(corpus, Mapping) or corpus != head_mass_protocol.get("corpus"):
        raise ValueError("slot-simplex corpus contract changed")
    retrieval = mass.capacity.bias.rank.retrieval
    corpus_manifest_path = retrieval._safe_relative_path(
        selector_path.parent,
        corpus.get("manifest_file"),
        "slot-simplex corpus manifest",
    )
    if sha256_file(corpus_manifest_path) != corpus.get("manifest_sha256"):
        raise ValueError("slot-simplex corpus manifest changed")
    train_descriptor = corpus.get("splits", {}).get("train")
    if not isinstance(train_descriptor, Mapping):
        raise ValueError("slot-simplex train descriptor is invalid")
    train_path = retrieval._safe_relative_path(
        selector_path.parent,
        train_descriptor.get("file"),
        "slot-simplex train split",
    )
    if sha256_file(train_path) != train_descriptor.get("sha256"):
        raise ValueError("slot-simplex train split changed")
    train_records = retrieval._read_split(train_path, split="train")
    if (
        len(train_records) != _RECORDS
        or sha256_json([row["identity_sha256"] for row in train_records])
        != train_descriptor.get("record_identity_sha256")
    ):
        raise ValueError("slot-simplex train record identity changed")

    package_contract = head_mass_protocol.get("package")
    if not isinstance(package_contract, Mapping):
        raise ValueError("slot-simplex package contract is invalid")
    package_path = Path(package_contract.get("path", "")).expanduser().resolve()
    package_manifest_path = package_path / "manifest.json"
    if (
        package_path.is_symlink()
        or not package_path.is_dir()
        or sha256_file(package_manifest_path)
        != package_contract.get("manifest_sha256")
    ):
        raise ValueError("slot-simplex package manifest changed")
    package_manifest = mass.capacity.bias.rank.retrieval._read_json(
        package_manifest_path,
        "slot-simplex package manifest",
    )
    files = package_manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("slot-simplex package files are invalid")
    artifact_rows = {
        "config_path": "model/config.json",
        "non_mlp_path": "transformer/non_mlp.safetensors",
        "q7_path": "mlp/experts.q7",
    }
    artifact_paths: dict[str, Path] = {}
    for context_name, relative in artifact_rows.items():
        descriptor = files.get(relative)
        if not isinstance(descriptor, Mapping):
            raise ValueError("slot-simplex package artifact descriptor is invalid")
        artifact_paths[context_name] = mass.capacity.bias.episodic._checked_file(
            package_path / relative,
            descriptor.get("sha256"),
            f"slot-simplex package {relative}",
        )
    model = package_contract.get("model")
    q7_expectations = package_contract.get("q7_expectations_per_sequence")
    historical_rows = head_mass_result.get("base_output_authentication")
    if (
        not isinstance(model, Mapping)
        or not isinstance(q7_expectations, Mapping)
        or not isinstance(historical_rows, list)
        or len(historical_rows) != _RECORDS
        or [row.get("record_id") for row in historical_rows]
        != [row["record_id"] for row in train_records]
    ):
        raise ValueError("slot-simplex inherited train evidence changed")
    context: dict[str, Any] = {
        **artifact_paths,
        "package_path": package_path,
        "package_manifest_path": package_manifest_path,
        "model": dict(model),
        "q7_expectations": dict(q7_expectations),
        "train_records": train_records,
        "train_path": train_path,
        "historical_output_rows": historical_rows,
        "head_mass_protocol_path": head_mass_path,
        "head_mass_protocol_sha256": head_mass_binding["sha256"],
        "head_mass_protocol": head_mass_protocol,
        "head_mass_result_path": head_mass_result_path,
        "head_mass_result_sha256": head_mass_result_binding["sha256"],
        "head_mass_result": head_mass_result,
        "head_mass_manifest_path": manifest_path,
        "head_mass_manifest_sha256": manifest_binding["sha256"],
        "head_mass_manifest": manifest,
        "joint_gamma_protocol_path": protocol_path,
        "joint_gamma_protocol_sha256": joint_protocol_sha256.lower(),
        "joint_gamma_protocol": frozen,
        "training_checkpoint_path": checkpoint_path,
        "training_checkpoint": checkpoint,
    }
    return context, dict(training), frozen


def _authenticate_joint_inputs(
    *,
    joint_protocol: str | Path,
    joint_protocol_sha256: str,
    joint_result: str | Path,
    joint_result_sha256: str,
    trace_library: str | Path,
    trace_library_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if joint_protocol_sha256.lower() != _EXPECTED_JOINT_PROTOCOL_SHA256:
        raise ValueError("slot-simplex joint protocol root changed")
    if joint_result_sha256.lower() != _EXPECTED_JOINT_RESULT_SHA256:
        raise ValueError("slot-simplex joint result root changed")
    context, training, joint_frozen = _authenticate_frozen_joint_evidence(
        joint_protocol,
        joint_protocol_sha256,
    )
    protocol_path = Path(context["joint_gamma_protocol_path"]).resolve()
    result_path = mass.capacity.bias.episodic._checked_file(
        joint_result,
        joint_result_sha256,
        "slot-simplex joint-gamma result",
    )
    result = mass.capacity.bias.rank.retrieval._read_json(
        result_path,
        "slot-simplex joint-gamma result",
    )
    failure = _validate_joint_failure(result, protocol_path=protocol_path)
    library_path = mass.capacity.bias.episodic._checked_file(
        trace_library,
        trace_library_sha256,
        "slot-simplex native trace library",
    )
    _require_slot_trace_symbols(library_path)
    context = dict(context)
    context.update(
        {
            "joint_gamma_result_path": result_path,
            "joint_gamma_result_sha256": joint_result_sha256.lower(),
            "joint_gamma_result": result,
            "slot_trace_library_path": library_path,
            "slot_trace_library_sha256": trace_library_sha256.lower(),
        }
    )
    if joint_frozen["confirmation_split_opened"] is not False:
        raise ValueError("slot-simplex inherited protocol opened confirmation")
    return context, training, failure


def _open_slot_trace_runtime(context: Mapping[str, Any]) -> Any:
    return OLMoENativeTokenRuntime(
        context["config_path"],
        context["non_mlp_path"],
        context["q7_path"],
        context["slot_trace_library_path"],
        threads=mass.capacity.bias._THREADS,
        **mass._BASE_POLICY,
        episodic_policy=mass._EPISODIC_POLICY,
        episodic_head_mask=mass.capacity.bias._all_ones_mask(),
        episodic_logit_bias=0.0,
        shadow_attention_policy=mass._SHADOW_POLICY,
    )


def _validate_runtime_route(runtime: Any) -> None:
    if (
        runtime.position != 0
        or not runtime.attention_metrics_available
        or not runtime.episodic_metrics_available
        or runtime.episodic_policy != mass._EPISODIC_POLICY
        or runtime.episodic_open_abi != "shadow_trace_v1"
        or runtime.shadow_trace_available is not True
        or runtime.episodic_mass_trace_available is not True
        or runtime.episodic_slot_trace_available is not True
        or runtime.shadow_attention_policy != mass._SHADOW_POLICY
        or not mass.capacity.bias.rank.fixed._runtime_mask_matches(
            runtime,
            mass.capacity.bias._all_ones_mask(),
        )
        or mass.capacity.bias._float32_bits(runtime.episodic_logit_bias)
        != mass.capacity.bias._float32_bits(0.0)
    ):
        raise ValueError("slot-simplex native runtime route changed")


def _trace_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in (*_BASE_TRACE_KEYS, *_SLOT_TRACE_KEYS):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _trace_summary(
    arrays: Mapping[str, np.ndarray],
    positions: Sequence[int],
) -> dict[str, Any]:
    if (
        list(positions) != list(_READ_POSITIONS)
        or set(arrays) != {*_BASE_TRACE_KEYS, *_SLOT_TRACE_KEYS}
    ):
        raise ValueError("slot-simplex trace support changed")
    tensor_hashes: dict[str, str] = {}
    value_shape = (len(_READ_POSITIONS), _LAYERS, _HIDDEN_SIZE)
    mass_shape = (len(_READ_POSITIONS), _LAYERS, _QUERY_HEADS)
    for name in (
        "base_attention_output",
        "regular_component",
        "episodic_component",
        "base_projected",
        "target_residual",
    ):
        value = arrays[name]
        if (
            value.shape != value_shape
            or value.dtype != np.float32
            or not value.flags.c_contiguous
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"slot-simplex {name} trace is invalid")
        tensor_hashes[name] = hashlib.sha256(value.tobytes(order="C")).hexdigest()
    for name in ("regular_mass", "episodic_mass"):
        value = arrays[name]
        if (
            value.shape != mass_shape
            or value.dtype != np.float32
            or not value.flags.c_contiguous
            or not np.isfinite(value).all()
            or np.any(value < np.float32(0.0))
            or np.any(value > np.float32(1.0))
        ):
            raise ValueError(f"slot-simplex {name} trace is invalid")
        tensor_hashes[name] = hashlib.sha256(value.tobytes(order="C")).hexdigest()
    slot_mass = arrays["slot_mass"]
    slot_values = arrays["slot_values"]
    if (
        slot_mass.shape != mass_shape + (_SLOTS,)
        or slot_mass.dtype != np.float32
        or not slot_mass.flags.c_contiguous
        or not np.isfinite(slot_mass).all()
        or np.any(slot_mass < np.float32(0.0))
        or slot_values.shape
        != mass_shape + (_SLOTS, _HEAD_DIMENSION)
        or slot_values.dtype != np.float32
        or not slot_values.flags.c_contiguous
        or not np.isfinite(slot_values).all()
    ):
        raise ValueError("slot-simplex slot trace is invalid")
    tensor_hashes["slot_mass"] = hashlib.sha256(
        slot_mass.tobytes(order="C")
    ).hexdigest()
    tensor_hashes["slot_values"] = hashlib.sha256(
        slot_values.tobytes(order="C")
    ).hexdigest()
    lower_mantissa = slot_values.view(np.uint32) & np.uint32(0xFFFF)
    if np.any(lower_mantissa != 0):
        raise ValueError("slot-simplex values are not exact BF16 decodes")
    mass_partition_error = float(
        np.max(
            np.abs(
                arrays["regular_mass"]
                + arrays["episodic_mass"]
                - np.float32(1.0)
            )
        )
    )
    slot_mass_error = float(
        np.max(
            np.abs(
                np.sum(slot_mass, axis=-1)
                - arrays["episodic_mass"]
            )
        )
    )
    episodic_heads = arrays["episodic_component"].reshape(
        *mass_shape,
        _HEAD_DIMENSION,
    )
    slot_component = np.einsum(
        "...hs,...hsd->...hd",
        slot_mass,
        slot_values,
        optimize=True,
    )
    slot_component_error = float(np.max(np.abs(slot_component - episodic_heads)))
    base_component_error = float(
        np.max(
            np.abs(
                arrays["regular_component"]
                + arrays["episodic_component"]
                - arrays["base_attention_output"]
            )
        )
    )
    if (
        mass_partition_error > 2.0e-5
        or slot_mass_error > 2.0e-5
        or slot_component_error > 5.0e-5
        or base_component_error > 5.0e-5
    ):
        raise ValueError("slot-simplex trace reconstruction failed")
    return {
        "positions": list(positions),
        "value_shape": list(value_shape),
        "mass_shape": list(mass_shape),
        "slot_mass_shape": list(slot_mass.shape),
        "slot_value_shape": list(slot_values.shape),
        "dtype": "float32",
        "layout": "position_layer_query_head_span_dimension",
        "tensor_sha256": tensor_hashes,
        "trace_sha256": _trace_digest(arrays),
        "mass_partition_max_abs": mass_partition_error,
        "slot_mass_reconstruction_max_abs": slot_mass_error,
        "episodic_component_reconstruction_max_abs": slot_component_error,
        "base_component_reconstruction_max_abs": base_component_error,
        "slot_values_exact_bf16_decodes": True,
    }


class _SlotTraceCaptureRuntime:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._positions: list[int] = []
        self._rows: dict[str, list[np.ndarray]] = {
            name: [] for name in (*_BASE_TRACE_KEYS, *_SLOT_TRACE_KEYS)
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
        result = self._runtime.forward_episodic(token_ids, write_slots, read_spans)
        if len(read_spans) != 1:
            raise ValueError("slot-simplex trace requires one-token calls")
        if int(read_spans[0]) >= 0:
            _old_input, base_projected, target_residual = (
                self._runtime.last_shadow_trace()
            )
            partition = self._runtime.last_episodic_mass_trace()
            slots = self._runtime.last_episodic_slot_trace()
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
            }
            self._positions.append(self.position - 1)
            for name in self._rows:
                self._rows[name].append(
                    np.ascontiguousarray(row[name], dtype=np.float32)
                )
        return result

    def captured(self) -> tuple[dict[str, np.ndarray], list[int]]:
        if len(self._positions) != len(_READ_POSITIONS):
            raise ValueError("slot-simplex trace capture is incomplete")
        arrays = {
            name: np.ascontiguousarray(np.stack(rows), dtype=np.float32)
            for name, rows in self._rows.items()
        }
        _trace_summary(arrays, self._positions)
        return arrays, list(self._positions)

    def reset(self) -> None:
        self._runtime.reset()
        self._positions.clear()
        for rows in self._rows.values():
            rows.clear()

    def close(self) -> None:
        self._runtime.close()


def _execute_record(
    runtime: _SlotTraceCaptureRuntime,
    *,
    record: Mapping[str, Any],
    context: Mapping[str, Any],
    schedule: Mapping[str, Any],
    resource: Mapping[str, Any],
    progress_label: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[int]]:
    _logits, _hidden, evidence = (
        mass.capacity.bias.episodic._execute_episodic_record(
            runtime,
            record=record,
            context=context,
            schedule=schedule,
            resource=resource,
            progress_label=progress_label,
        )
    )
    arrays, positions = runtime.captured()
    return evidence, arrays, positions


def _historical_head_mass_arrays(
    context: Mapping[str, Any],
    *,
    record_index: int,
) -> dict[str, np.ndarray]:
    descriptor = context["head_mass_manifest"]["shards"][record_index]
    path = Path(context["head_mass_manifest_path"]).parent / descriptor["file"]
    return mass._validate_trace_shard(path, descriptor)


def _common_trace_exact(
    arrays: Mapping[str, np.ndarray],
    historical: Mapping[str, np.ndarray],
) -> bool:
    return all(
        np.array_equal(arrays[name], historical[name])
        for name in _BASE_TRACE_KEYS
    )


def _historical_output_exact(
    evidence: Mapping[str, Any],
    historical: Mapping[str, Any],
) -> bool:
    return bool(
        evidence.get("record_index") == historical.get("record_index")
        and evidence.get("record_id") == historical.get("record_id")
        and evidence.get("answer_cross_entropy")
        == historical.get("answer_cross_entropy")
        and evidence.get("hidden_sha256") == historical.get("hidden_sha256")
        and evidence.get("logits_sha256") == historical.get("logits_sha256")
        and evidence.get("counter_stream_sha256")
        == historical.get("counter_stream_sha256")
        and evidence.get("episodic_call_stream_sha256")
        == historical.get("episodic_call_stream_sha256")
        and evidence.get("counter_stream_passed") is True
        and sha256_json(mass.capacity._without_elapsed(evidence))
        == historical.get("observed_output_evidence_sha256")
    )


def _run_trace_parity(
    *,
    context: Mapping[str, Any],
    runtime_factory: Callable[[Mapping[str, Any]], Any] = _open_slot_trace_runtime,
) -> dict[str, Any]:
    record = context["train_records"][0]
    schedule = mass.capacity.bias.rank.fixed._derive_schedule(
        record["input_ids"],
        context["head_mass_protocol"]["schedule_contract"][
            "tokenizer_fact_anchor_ids"
        ],
    )
    raw = runtime_factory(context)
    trace = _SlotTraceCaptureRuntime(raw)
    try:
        _validate_runtime_route(raw)
        first, arrays, positions = _execute_record(
            trace,
            record=record,
            context=context,
            schedule=schedule,
            resource=context["head_mass_protocol"]["fixed_K256_arm"][
                "resource_contract"
            ],
            progress_label="slot-simplex parity first",
        )
        first_summary = _trace_summary(arrays, positions)
        trace.reset()
        replay, replay_arrays, replay_positions = _execute_record(
            trace,
            record=record,
            context=context,
            schedule=schedule,
            resource=context["head_mass_protocol"]["fixed_K256_arm"][
                "resource_contract"
            ],
            progress_label="slot-simplex parity reset",
        )
        replay_summary = _trace_summary(replay_arrays, replay_positions)
    finally:
        trace.close()
    historical_output = context["historical_output_rows"][0]
    historical_arrays = _historical_head_mass_arrays(context, record_index=0)
    checks = {
        "historical_base_outputs_counters_and_loss_exact": (
            _historical_output_exact(first, historical_output)
        ),
        "reset_outputs_counters_and_loss_exact": mass.capacity._evidence_exact(
            first,
            replay,
        ),
        "inherited_mass_and_projected_trace_exact": _common_trace_exact(
            arrays,
            historical_arrays,
        ),
        "reset_trace_exact": first_summary == replay_summary,
        "slot_mass_reconstructs_episodic_mass": (
            first_summary["slot_mass_reconstruction_max_abs"] <= 2.0e-5
        ),
        "slot_values_reconstruct_episodic_component": (
            first_summary["episodic_component_reconstruction_max_abs"] <= 5.0e-5
        ),
        "slot_values_exact_bf16_decodes": (
            first_summary["slot_values_exact_bf16_decodes"] is True
        ),
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise ValueError("slot-simplex real-model trace parity failed")
    return {
        "record_index": 0,
        "schedule_rows_sha256": schedule["rows_sha256"],
        "first_output_evidence_sha256": sha256_json(
            mass.capacity._without_elapsed(first)
        ),
        "reset_output_evidence_sha256": sha256_json(
            mass.capacity._without_elapsed(replay)
        ),
        "first_trace": first_summary,
        "reset_trace": replay_summary,
        "inherited_trace_sha256": historical_arrays
        and mass._trace_array_digest(historical_arrays),
        "checks": checks,
        "native_sequence_forwards": 2,
        "native_token_steps": 2 * _POSITIONS,
        "passed": True,
    }


def _parity_post_authentication(
    context: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "joint_gamma_protocol": (
            sha256_file(context["joint_gamma_protocol_path"])
            == context["joint_gamma_protocol_sha256"]
        ),
        "joint_gamma_result": (
            sha256_file(context["joint_gamma_result_path"])
            == context["joint_gamma_result_sha256"]
        ),
        "head_mass_manifest": (
            sha256_file(context["head_mass_manifest_path"])
            == context["head_mass_manifest_sha256"]
        ),
        "slot_trace_library": (
            sha256_file(context["slot_trace_library_path"])
            == context["slot_trace_library_sha256"]
        ),
        "confirmation_not_opened": True,
    }


def generate_trace_parity_report(
    *,
    joint_protocol: str | Path,
    joint_protocol_sha256: str,
    joint_result: str | Path,
    joint_result_sha256: str,
    trace_library: str | Path,
    trace_library_sha256: str,
    out: str | Path,
    runtime_factory: Callable[[Mapping[str, Any]], Any] = _open_slot_trace_runtime,
) -> dict[str, Any]:
    output = mass.capacity.bias.rank.retrieval._new_output(
        out,
        "slot-simplex trace parity report",
    )
    context, _training, failure = _authenticate_joint_inputs(
        joint_protocol=joint_protocol,
        joint_protocol_sha256=joint_protocol_sha256,
        joint_result=joint_result,
        joint_result_sha256=joint_result_sha256,
        trace_library=trace_library,
        trace_library_sha256=trace_library_sha256,
    )
    parity = _run_trace_parity(
        context=context,
        runtime_factory=runtime_factory,
    )
    post = _parity_post_authentication(context)
    if not all(post.values()):
        raise ValueError("slot-simplex parity post-authentication failed")
    report = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _PARITY_EXPERIMENT,
        "status": _PARITY_STATUS,
        "joint_gamma_protocol": {
            "path": str(context["joint_gamma_protocol_path"]),
            "sha256": context["joint_gamma_protocol_sha256"],
        },
        "joint_gamma_result": {
            "path": str(context["joint_gamma_result_path"]),
            "sha256": context["joint_gamma_result_sha256"],
            "authenticated_failure": failure,
        },
        "trace_library": {
            "path": str(context["slot_trace_library_path"]),
            "sha256": context["slot_trace_library_sha256"],
            "required_symbols": [
                mass.capacity._TRACE_OPEN_SYMBOL,
                mass.capacity._TRACE_COPY_SYMBOL,
                mass._MASS_COPY_SYMBOL,
                _SLOT_COPY_SYMBOL,
            ],
        },
        "parity": parity,
        "post_run_authentication": post,
        "confirmation_split_opened": False,
    }
    atomic_json(output, report)
    _progress(f"slot-simplex parity report written to {output}")
    return report


def _validate_parity_report(
    value: Any,
    *,
    context: Mapping[str, Any],
    parity_path: Path,
    parity_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("slot-simplex parity report is invalid")
    parity = value.get("parity")
    post = value.get("post_run_authentication")
    checks = parity.get("checks") if isinstance(parity, Mapping) else None
    first_trace = (
        parity.get("first_trace") if isinstance(parity, Mapping) else None
    )
    reset_trace = (
        parity.get("reset_trace") if isinstance(parity, Mapping) else None
    )
    expected_check_keys = {
        "historical_base_outputs_counters_and_loss_exact",
        "reset_outputs_counters_and_loss_exact",
        "inherited_mass_and_projected_trace_exact",
        "reset_trace_exact",
        "slot_mass_reconstructs_episodic_mass",
        "slot_values_reconstruct_episodic_component",
        "slot_values_exact_bf16_decodes",
        "passed",
    }
    expected_post_keys = {
        "joint_gamma_protocol",
        "joint_gamma_result",
        "head_mass_manifest",
        "slot_trace_library",
        "confirmation_not_opened",
    }
    if (
        value.get("schema_version") != _SCHEMA_VERSION
        or value.get("experiment") != _PARITY_EXPERIMENT
        or value.get("status") != _PARITY_STATUS
        or value.get("joint_gamma_protocol")
        != {
            "path": str(context["joint_gamma_protocol_path"]),
            "sha256": context["joint_gamma_protocol_sha256"],
        }
        or value.get("joint_gamma_result", {}).get("path")
        != str(context["joint_gamma_result_path"])
        or value.get("joint_gamma_result", {}).get("sha256")
        != context["joint_gamma_result_sha256"]
        or value.get("trace_library", {}).get("path")
        != str(context["slot_trace_library_path"])
        or value.get("trace_library", {}).get("sha256")
        != context["slot_trace_library_sha256"]
        or not isinstance(parity, Mapping)
        or parity.get("passed") is not True
        or not isinstance(checks, Mapping)
        or set(checks) != expected_check_keys
        or not all(check is True for check in checks.values())
        or not isinstance(first_trace, Mapping)
        or not isinstance(reset_trace, Mapping)
        or reset_trace != first_trace
        or parity.get("record_index") != 0
        or parity.get("schedule_rows_sha256")
        != context["head_mass_protocol"]["schedule_contract"][
            "per_record_rows_sha256"
        ][0]
        or parity.get("native_sequence_forwards") != 2
        or parity.get("native_token_steps") != 2 * _POSITIONS
        or first_trace.get("positions") != list(_READ_POSITIONS)
        or first_trace.get("value_shape")
        != [len(_READ_POSITIONS), _LAYERS, _HIDDEN_SIZE]
        or first_trace.get("mass_shape")
        != [len(_READ_POSITIONS), _LAYERS, _QUERY_HEADS]
        or first_trace.get("slot_mass_shape")
        != [len(_READ_POSITIONS), _LAYERS, _QUERY_HEADS, _SLOTS]
        or first_trace.get("slot_value_shape")
        != [
            len(_READ_POSITIONS),
            _LAYERS,
            _QUERY_HEADS,
            _SLOTS,
            _HEAD_DIMENSION,
        ]
        or first_trace.get("layout")
        != "position_layer_query_head_span_dimension"
        or first_trace.get("slot_values_exact_bf16_decodes") is not True
        or first_trace.get("mass_partition_max_abs", float("inf")) > 2.0e-5
        or first_trace.get(
            "slot_mass_reconstruction_max_abs",
            float("inf"),
        )
        > 2.0e-5
        or first_trace.get(
            "episodic_component_reconstruction_max_abs",
            float("inf"),
        )
        > 5.0e-5
        or first_trace.get(
            "base_component_reconstruction_max_abs",
            float("inf"),
        )
        > 5.0e-5
        or value.get("trace_library", {}).get("required_symbols")
        != [
            mass.capacity._TRACE_OPEN_SYMBOL,
            mass.capacity._TRACE_COPY_SYMBOL,
            mass._MASS_COPY_SYMBOL,
            _SLOT_COPY_SYMBOL,
        ]
        or not isinstance(post, Mapping)
        or set(post) != expected_post_keys
        or not all(check is True for check in post.values())
        or value.get("confirmation_split_opened") is not False
    ):
        raise ValueError("slot-simplex parity contract changed")
    return {
        "path": str(parity_path),
        "sha256": parity_sha256.lower(),
        "report": dict(value),
    }


def _build_protocol(
    *,
    context: Mapping[str, Any],
    training: Mapping[str, Any],
    failure: Mapping[str, Any],
    parity: Mapping[str, Any],
) -> dict[str, Any]:
    head_mass_protocol = context["head_mass_protocol"]
    return {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _PROTOCOL_EXPERIMENT,
        "status": _PROTOCOL_STATUS,
        "joint_gamma_protocol": {
            "path": str(context["joint_gamma_protocol_path"]),
            "sha256": context["joint_gamma_protocol_sha256"],
        },
        "joint_gamma_result": {
            "path": str(context["joint_gamma_result_path"]),
            "sha256": context["joint_gamma_result_sha256"],
            "authenticated_failure": dict(failure),
        },
        "inherited_head_mass_trace_manifest": {
            "path": str(context["head_mass_manifest_path"]),
            "sha256": context["head_mass_manifest_sha256"],
            "shards": _RECORDS,
        },
        "slot_trace_library": {
            "path": str(context["slot_trace_library_path"]),
            "sha256": context["slot_trace_library_sha256"],
            "copy_symbol": _SLOT_COPY_SYMBOL,
        },
        "trace_parity": dict(parity),
        "training_checkpoint": dict(head_mass_protocol["training_checkpoint"]),
        "training_checkpoint_payload_sha256": sha256_json(training),
        "schedule_contract": dict(head_mass_protocol["schedule_contract"]),
        "trace_contract": {
            "slot_mass_layout": (
                "position_layer_query_head_read_span_offset"
            ),
            "slot_value_layout": (
                "position_layer_query_head_read_span_offset_head_dimension"
            ),
            "slot_count": _SLOTS,
            "head_dimension": _HEAD_DIMENSION,
            "slot_value_storage": "exact_BF16_decode_in_float32",
            "required_invariants": [
                "sum(slot_mass,span)==episodic_mass",
                "sum(slot_mass*slot_values,span)==episodic_component",
                "regular_component+episodic_component==base_attention_output",
                "ordinary output state and resource counters remain unchanged",
            ],
            "instrumentation_state_and_traffic_counted": False,
        },
        "oracle_method": {
            "two_arms": True,
            "constructible_arm": {
                "components_per_selected_head": _CONSTRUCTIBLE_COMPONENTS,
                "components": [
                    "regular_cache_conditional_mean",
                    "eight_exact_BF16_episodic_slot_values_in_read_order",
                ],
                "authority": (
                    "a feasible gate pass may authorize a separately "
                    "accounted causal selector experiment"
                ),
            },
            "optimistic_hull_arm": {
                "components_per_selected_head": _OPTIMISTIC_COMPONENTS,
                "components": [
                    "exact_native_head_output_anchor",
                    "regular_cache_conditional_mean",
                    "eight_exact_BF16_episodic_slot_values_in_read_order",
                ],
                "authority": (
                    "its certified optimistic failure is decisive; a pass "
                    "cannot authorize a selector by itself"
                ),
            },
            "feasible_sets": (
                "Cartesian products of one probability simplex per query "
                "head, optimized jointly after o_proj"
            ),
            "capacity_interpretation": (
                "the nine-way arm is the traced constructible value set; the "
                "ten-way exact-native-anchor hull is its optimistic superset "
                "and absorbs trace-regrouping roundoff"
            ),
            "target": "exact same-state native W128-minus-K256 post-Wo residual",
            "quadratic_construction": (
                "float64 factor-defined Gram A.T@A from isolated head/value "
                "correction directions projected through authenticated BF16 Wo"
            ),
            "solver": (
                "deterministic block-coordinate Frank-Wolfe with pairwise "
                "away steps, exact quadratic line searches, and full "
                "product-simplex Frank-Wolfe objective-gap certificate"
            ),
            "maximum_iterations": _MAXIMUM_ITERATIONS,
            "relative_objective_gap_target": _RELATIVE_GAP_TOLERANCE,
            "row_batch_size": _ROW_BATCH_SIZE,
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
        },
        "resource_contract": dict(head_mass_protocol["resource_contract"]),
        "scope": {
            "split": "train",
            "fresh_native_slot_trace_required": True,
            "same_state_capacity_evidence_only": True,
            "learned_selector": False,
            "causal_rollout": False,
            "semantic_or_M3_pass": False,
            "development_outcomes_used": False,
            "confirmation_file_access_permitted": False,
        },
        "authorized_next_step_on_feasible_pass": (
            "freeze a train-only causal per-slot selector or logit-bias model "
            "with rolled-out native state and full selector accounting"
        ),
        "failure_interpretation": (
            "when the certified optimistic recovery upper bound fails, close "
            "all same-state reweighting over the current regular aggregate and "
            "eight episodic values at fixed K256; the next experiment must add "
            "new value directions or a different memory"
        ),
        "authenticated_confirmation_descriptor": dict(
            head_mass_protocol["authenticated_confirmation_descriptor"]
        ),
        "source_sha256": _source_inventory(),
        "confirmation_split_opened": False,
    }


def freeze_slot_simplex_protocol(
    *,
    joint_protocol: str | Path,
    joint_protocol_sha256: str,
    joint_result: str | Path,
    joint_result_sha256: str,
    trace_library: str | Path,
    trace_library_sha256: str,
    parity_report: str | Path,
    parity_report_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    output = mass.capacity.bias.rank.retrieval._new_output(
        out,
        "slot-simplex protocol",
    )
    context, training, failure = _authenticate_joint_inputs(
        joint_protocol=joint_protocol,
        joint_protocol_sha256=joint_protocol_sha256,
        joint_result=joint_result,
        joint_result_sha256=joint_result_sha256,
        trace_library=trace_library,
        trace_library_sha256=trace_library_sha256,
    )
    parity_path = mass.capacity.bias.episodic._checked_file(
        parity_report,
        parity_report_sha256,
        "slot-simplex parity report",
    )
    parity_value = mass.capacity.bias.rank.retrieval._read_json(
        parity_path,
        "slot-simplex parity report",
    )
    parity = _validate_parity_report(
        parity_value,
        context=context,
        parity_path=parity_path,
        parity_sha256=parity_report_sha256,
    )
    protocol = _build_protocol(
        context=context,
        training=training,
        failure=failure,
        parity=parity,
    )
    atomic_json(output, protocol)
    _progress(f"slot-simplex protocol written to {output}")
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "protocol": protocol,
    }


def _authenticate_protocol(
    protocol: str | Path,
    protocol_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = mass.capacity.bias.episodic._checked_file(
        protocol,
        protocol_sha256,
        "slot-simplex protocol",
    )
    value = mass.capacity.bias.rank.retrieval._read_json(
        source,
        "slot-simplex protocol",
    )
    joint_protocol = value.get("joint_gamma_protocol")
    joint_result = value.get("joint_gamma_result")
    library = value.get("slot_trace_library")
    parity_binding = value.get("trace_parity")
    if not all(
        isinstance(binding, Mapping)
        for binding in (joint_protocol, joint_result, library, parity_binding)
    ):
        raise ValueError("slot-simplex protocol bindings are invalid")
    context, training, failure = _authenticate_joint_inputs(
        joint_protocol=joint_protocol.get("path"),
        joint_protocol_sha256=joint_protocol.get("sha256"),
        joint_result=joint_result.get("path"),
        joint_result_sha256=joint_result.get("sha256"),
        trace_library=library.get("path"),
        trace_library_sha256=library.get("sha256"),
    )
    parity_path = mass.capacity.bias.episodic._checked_file(
        parity_binding.get("path"),
        parity_binding.get("sha256"),
        "slot-simplex parity report",
    )
    parity_value = mass.capacity.bias.rank.retrieval._read_json(
        parity_path,
        "slot-simplex parity report",
    )
    parity = _validate_parity_report(
        parity_value,
        context=context,
        parity_path=parity_path,
        parity_sha256=parity_binding.get("sha256"),
    )
    expected = _build_protocol(
        context=context,
        training=training,
        failure=failure,
        parity=parity,
    )
    if value != expected:
        raise ValueError("slot-simplex frozen protocol changed")
    context = dict(context)
    context.update(
        {
            "slot_simplex_protocol_path": source,
            "slot_simplex_protocol_sha256": protocol_sha256.lower(),
            "slot_simplex_protocol": expected,
            "slot_simplex_parity_path": parity_path,
            "slot_simplex_parity_sha256": parity_binding["sha256"],
        }
    )
    return context, training, expected


def _prepare_shard_directory(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    if requested.exists() or requested.is_symlink():
        raise ValueError("slot-simplex trace shard directory already exists")
    requested.mkdir(parents=True)
    return requested.resolve()


def _write_slot_shard(
    directory: Path,
    *,
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    positions: Sequence[int],
    summary: Mapping[str, Any],
    reset_trace_sha256: str,
    source_record_sha256: str,
    output_evidence_sha256: str,
    reset_output_evidence_sha256: str,
) -> dict[str, Any]:
    filename = f"train-{int(record['record_index']):02d}.safetensors"
    path = directory / filename
    if path.exists() or path.is_symlink():
        raise ValueError("slot-simplex trace shard already exists")
    try:
        from safetensors.numpy import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("slot-simplex shards require safetensors") from exc
    payload = {
        "slot_mass": np.ascontiguousarray(arrays["slot_mass"], dtype=np.float32),
        "slot_values": np.ascontiguousarray(arrays["slot_values"], dtype=np.float32),
        "positions": np.asarray(positions, dtype=np.int64),
    }
    temporary = directory / f".{filename}.tmp-{os.getpid()}"
    save_file(payload, str(temporary))
    temporary.replace(path)
    descriptor = {
        "record_index": int(record["record_index"]),
        "record_id": record["record_id"],
        "file": filename,
        "file_sha256": sha256_file(path),
        "format": "safetensors",
        "keys": ["slot_mass", "slot_values", "positions"],
        "slot_mass_shape": [
            len(_READ_POSITIONS),
            _LAYERS,
            _QUERY_HEADS,
            _SLOTS,
        ],
        "slot_value_shape": [
            len(_READ_POSITIONS),
            _LAYERS,
            _QUERY_HEADS,
            _SLOTS,
            _HEAD_DIMENSION,
        ],
        "dtype": "float32_exact_BF16_decode",
        "positions": list(_READ_POSITIONS),
        "tensor_sha256": {
            name: summary["tensor_sha256"][name] for name in _SLOT_TRACE_KEYS
        },
        "full_trace_sha256": summary["trace_sha256"],
        "reset_full_trace_sha256": reset_trace_sha256,
        "source_record_sha256": source_record_sha256,
        "output_evidence_sha256": output_evidence_sha256,
        "reset_output_evidence_sha256": reset_output_evidence_sha256,
    }
    _validate_slot_shard(path, descriptor)
    return descriptor


def _validate_slot_shard(
    path: str | Path,
    descriptor: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError("slot-simplex shard descriptor is invalid")
    source = requested.resolve()
    expected_mass_shape = [
        len(_READ_POSITIONS),
        _LAYERS,
        _QUERY_HEADS,
        _SLOTS,
    ]
    expected_value_shape = [*expected_mass_shape, _HEAD_DIMENSION]
    if (
        not source.is_file()
        or source.name != descriptor.get("file")
        or sha256_file(source) != descriptor.get("file_sha256")
        or descriptor.get("format") != "safetensors"
        or descriptor.get("keys") != ["slot_mass", "slot_values", "positions"]
        or descriptor.get("slot_mass_shape") != expected_mass_shape
        or descriptor.get("slot_value_shape") != expected_value_shape
        or descriptor.get("dtype") != "float32_exact_BF16_decode"
        or descriptor.get("positions") != list(_READ_POSITIONS)
        or set(descriptor.get("tensor_sha256", {})) != set(_SLOT_TRACE_KEYS)
        or not all(
            mass.capacity.bias.rank.retrieval._is_sha256(digest)
            for digest in descriptor.get("tensor_sha256", {}).values()
        )
        or not all(
            mass.capacity.bias.rank.retrieval._is_sha256(descriptor.get(name))
            for name in (
                "file_sha256",
                "full_trace_sha256",
                "reset_full_trace_sha256",
                "source_record_sha256",
                "output_evidence_sha256",
                "reset_output_evidence_sha256",
            )
        )
    ):
        raise ValueError("slot-simplex shard descriptor is invalid")
    try:
        from safetensors import safe_open
        from safetensors.numpy import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("slot-simplex shards require safetensors") from exc
    with safe_open(source, framework="numpy") as handle:
        if sorted(handle.keys()) != ["positions", "slot_mass", "slot_values"]:
            raise ValueError("slot-simplex shard keys changed")
    loaded = load_file(source)
    positions = loaded["positions"]
    slot_mass = np.ascontiguousarray(loaded["slot_mass"])
    slot_values = np.ascontiguousarray(loaded["slot_values"])
    if (
        positions.dtype != np.int64
        or positions.shape != (len(_READ_POSITIONS),)
        or positions.tolist() != list(_READ_POSITIONS)
        or slot_mass.dtype != np.float32
        or list(slot_mass.shape) != expected_mass_shape
        or slot_values.dtype != np.float32
        or list(slot_values.shape) != expected_value_shape
        or not np.isfinite(slot_mass).all()
        or not np.isfinite(slot_values).all()
        or np.any(slot_mass < np.float32(0.0))
        or np.any((slot_values.view(np.uint32) & np.uint32(0xFFFF)) != 0)
        or hashlib.sha256(slot_mass.tobytes(order="C")).hexdigest()
        != descriptor["tensor_sha256"]["slot_mass"]
        or hashlib.sha256(slot_values.tobytes(order="C")).hexdigest()
        != descriptor["tensor_sha256"]["slot_values"]
    ):
        raise ValueError("slot-simplex shard tensor changed")
    return {"slot_mass": slot_mass, "slot_values": slot_values}


def _load_stacked_arrays(
    context: Mapping[str, Any],
    slot_descriptors: Sequence[Mapping[str, Any]],
    slot_directory: Path,
) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {
        name: [] for name in (*_BASE_TRACE_KEYS, *_SLOT_TRACE_KEYS)
    }
    inherited_descriptors = context["head_mass_manifest"]["shards"]
    if len(slot_descriptors) != _RECORDS or len(inherited_descriptors) != _RECORDS:
        raise ValueError("slot-simplex trace manifest row count changed")
    for index, (slot_descriptor, inherited_descriptor) in enumerate(
        zip(slot_descriptors, inherited_descriptors, strict=True)
    ):
        if (
            slot_descriptor.get("record_index") != index
            or inherited_descriptor.get("record_index") != index
            or slot_descriptor.get("record_id") != inherited_descriptor.get("record_id")
        ):
            raise ValueError("slot-simplex trace record binding changed")
        inherited_path = (
            Path(context["head_mass_manifest_path"]).parent
            / inherited_descriptor["file"]
        )
        inherited = mass._validate_trace_shard(
            inherited_path,
            inherited_descriptor,
        )
        slots = _validate_slot_shard(
            slot_directory / slot_descriptor["file"],
            slot_descriptor,
        )
        combined = {
            **{name: inherited[name] for name in _BASE_TRACE_KEYS},
            **slots,
        }
        combined_summary = _trace_summary(combined, _READ_POSITIONS)
        if (
            combined_summary["trace_sha256"]
            != slot_descriptor["full_trace_sha256"]
        ):
            raise ValueError(
                "slot-simplex slot/base combined trace binding changed"
            )
        for name in _BASE_TRACE_KEYS:
            rows[name].append(inherited[name])
        for name in _SLOT_TRACE_KEYS:
            rows[name].append(slots[name])
    return {
        name: np.ascontiguousarray(np.stack(values), dtype=np.float32)
        for name, values in rows.items()
    }


def _screen_post_authentication(
    context: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "slot_simplex_protocol": (
            sha256_file(context["slot_simplex_protocol_path"])
            == context["slot_simplex_protocol_sha256"]
        ),
        "slot_simplex_parity": (
            sha256_file(context["slot_simplex_parity_path"])
            == context["slot_simplex_parity_sha256"]
        ),
        "joint_gamma_protocol": (
            sha256_file(context["joint_gamma_protocol_path"])
            == context["joint_gamma_protocol_sha256"]
        ),
        "joint_gamma_result": (
            sha256_file(context["joint_gamma_result_path"])
            == context["joint_gamma_result_sha256"]
        ),
        "head_mass_manifest": (
            sha256_file(context["head_mass_manifest_path"])
            == context["head_mass_manifest_sha256"]
        ),
        "slot_trace_library": (
            sha256_file(context["slot_trace_library_path"])
            == context["slot_trace_library_sha256"]
        ),
        "source_inventory": (
            context["slot_simplex_protocol"]["source_sha256"]
            == _source_inventory()
        ),
        "confirmation_not_opened": True,
    }


def _summarize_oracle_arm(
    solved: SlotSimplexOracleResult,
    *,
    component_count: int,
    arm: str,
) -> dict[str, Any]:
    negative_tolerance = 1.0e-10 * np.maximum(1.0, solved.target_energy)
    if np.any(solved.objective < -negative_tolerance):
        raise ValueError(f"slot-simplex {arm} quadratic objective became negative")
    quadratic_error = np.maximum(solved.objective, 0.0)
    feasible_metrics = joint._recovery_metrics_from_energy(
        solved.target_energy,
        solved.direct_error_energy,
        batch_shape=solved.batch_shape,
    )
    quadratic_metrics = joint._recovery_metrics_from_energy(
        solved.target_energy,
        quadratic_error,
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
    objective_nonregression = bool(
        np.all(
            solved.direct_error_energy
            <= solved.target_energy + nonregression_tolerance
        )
    )
    parity_tolerance = 1.0e-9 * np.maximum(1.0, solved.target_energy)
    quadratic_direct_parity = bool(
        np.all(
            np.abs(quadratic_error - solved.direct_error_energy)
            <= parity_tolerance
        )
    )
    certificate_available = bool(
        np.isfinite(solved.objective_gap_upper_bound).all()
        and np.all(solved.objective_gap_upper_bound >= 0.0)
    )
    qualification_passed = bool(
        objective_nonregression
        and quadratic_direct_parity
        and solved.deterministic_replay_exact
        and certificate_available
    )
    feasible_passed = bool(feasible_metrics["passed"] and qualification_passed)
    optimistic_passed = bool(
        optimistic_metrics["passed"] and qualification_passed
    )
    coefficient_argmax = np.argmax(solved.coefficients, axis=-1)
    argmax_histogram = np.bincount(
        coefficient_argmax.reshape(-1),
        minlength=component_count,
    )
    active_counts = np.count_nonzero(solved.coefficients > 1.0e-10, axis=-1)
    return {
        "arm": arm,
        "components_per_head": component_count,
        "coefficient_sha256": hashlib.sha256(
            solved.coefficients.tobytes(order="C")
        ).hexdigest(),
        "objective_sha256": hashlib.sha256(
            solved.objective.tobytes(order="C")
        ).hexdigest(),
        "objective_gap_upper_bound_sha256": hashlib.sha256(
            solved.objective_gap_upper_bound.tobytes(order="C")
        ).hexdigest(),
        "direct_error_energy_sha256": hashlib.sha256(
            solved.direct_error_energy.tobytes(order="C")
        ).hexdigest(),
        "argmax_component_histogram": {
            str(index): int(count)
            for index, count in enumerate(argmax_histogram)
        },
        "active_components_per_head": {
            "minimum": int(np.min(active_counts)),
            "mean": float(np.mean(active_counts)),
            "maximum": int(np.max(active_counts)),
        },
        "solver": {
            "minimum_completed_iterations": int(np.min(solved.iterations)),
            "mean_completed_iterations": float(np.mean(solved.iterations)),
            "maximum_completed_iterations": int(np.max(solved.iterations)),
            "converged_rows": int(np.count_nonzero(solved.converged)),
            "total_rows": int(solved.converged.size),
            "maximum_relative_objective_gap": (
                solved.maximum_relative_objective_gap
            ),
            "maximum_objective_gap_upper_bound": float(
                np.max(solved.objective_gap_upper_bound)
            ),
            "summed_objective_gap_upper_bound": float(
                np.sum(solved.objective_gap_upper_bound, dtype=np.float64)
            ),
            "objective_gap_certificate_available": certificate_available,
        },
        "qualification": {
            "exact_anchor_reconstruction_max_abs": (
                solved.base_reconstruction_max_abs
            ),
            "traced_partition_reconstruction_max_abs": (
                solved.traced_partition_reconstruction_max_abs
            ),
            "episodic_component_reconstruction_max_abs": (
                solved.episodic_component_reconstruction_max_abs
            ),
            "mass_partition_max_abs": solved.mass_partition_max_abs,
            "quadratic_direct_error_energy_max_abs": (
                solved.quadratic_direct_error_energy_max_abs
            ),
            "quadratic_direct_parity_tolerance_rule": (
                "1e-9*max(1,target_energy) per row"
            ),
            "quadratic_direct_parity": quadratic_direct_parity,
            "direct_objective_never_worse_than_native_base": (
                objective_nonregression
            ),
            "direct_nonregression_tolerance_rule": (
                "2e-6*max(1,native_base_target_energy) per row"
            ),
            "negative_quadratic_tolerance_rule": (
                "1e-10*max(1,target_energy) per row, then clamp to zero"
            ),
            "deterministic_solver_replay_exact": (
                solved.deterministic_replay_exact
            ),
            "factor_defined_psd_gram": True,
            "passed": qualification_passed,
        },
        "feasible_solution_metrics": feasible_metrics,
        "quadratic_solution_metrics_diagnostic": quadratic_metrics,
        "optimistic_recovery_upper_bound_metrics": optimistic_metrics,
        "feasible_gate_passed": feasible_passed,
        "optimistic_gate_passed": optimistic_passed,
    }


def screen_slot_simplex_oracle(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    shard_dir: str | Path,
    out: str | Path,
    runtime_factory: Callable[[Mapping[str, Any]], Any] = _open_slot_trace_runtime,
) -> dict[str, Any]:
    output = mass.capacity.bias.rank.retrieval._new_output(
        out,
        "slot-simplex result",
    )
    started = time.perf_counter()
    context, _training, frozen = _authenticate_protocol(
        protocol,
        protocol_sha256,
    )
    records = context["train_records"]
    schedules = [
        mass.capacity.bias.rank.fixed._derive_schedule(
            record["input_ids"],
            frozen["schedule_contract"]["tokenizer_fact_anchor_ids"],
        )
        for record in records
    ]
    if [row["rows_sha256"] for row in schedules] != frozen["schedule_contract"][
        "per_record_rows_sha256"
    ]:
        raise ValueError("slot-simplex execution schedule changed")
    directory = _prepare_shard_directory(shard_dir)

    raw = runtime_factory(context)
    trace = _SlotTraceCaptureRuntime(raw)
    manifest_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    resource = context["head_mass_protocol"]["fixed_K256_arm"][
        "resource_contract"
    ]
    try:
        _validate_runtime_route(raw)
        for index, (record, schedule, historical_output) in enumerate(
            zip(
                records,
                schedules,
                context["historical_output_rows"],
                strict=True,
            )
        ):
            _progress(f"capturing train record {index + 1}/{_RECORDS}")
            first, arrays, positions = _execute_record(
                trace,
                record=record,
                context=context,
                schedule=schedule,
                resource=resource,
                progress_label=f"slot-simplex train {index + 1}/{_RECORDS}",
            )
            if not _historical_output_exact(first, historical_output):
                raise ValueError(
                    f"slot-simplex record {index} base evidence changed"
                )
            first_summary = _trace_summary(arrays, positions)
            inherited = _historical_head_mass_arrays(
                context,
                record_index=index,
            )
            if not _common_trace_exact(arrays, inherited):
                raise ValueError(
                    f"slot-simplex record {index} inherited trace changed"
                )
            trace.reset()
            replay, replay_arrays, replay_positions = _execute_record(
                trace,
                record=record,
                context=context,
                schedule=schedule,
                resource=resource,
            )
            replay_summary = _trace_summary(replay_arrays, replay_positions)
            if (
                not mass.capacity._evidence_exact(first, replay)
                or first_summary != replay_summary
                or not _common_trace_exact(replay_arrays, inherited)
            ):
                raise ValueError(
                    f"slot-simplex record {index} reset replay changed"
                )
            source_record_sha256 = sha256_json(record)
            output_sha256 = sha256_json(mass.capacity._without_elapsed(first))
            reset_output_sha256 = sha256_json(
                mass.capacity._without_elapsed(replay)
            )
            descriptor = _write_slot_shard(
                directory,
                record=record,
                arrays=arrays,
                positions=positions,
                summary=first_summary,
                reset_trace_sha256=replay_summary["trace_sha256"],
                source_record_sha256=source_record_sha256,
                output_evidence_sha256=output_sha256,
                reset_output_evidence_sha256=reset_output_sha256,
            )
            manifest_rows.append(descriptor)
            output_rows.append(
                {
                    "record_index": index,
                    "record_id": record["record_id"],
                    "historical_output_evidence_sha256": historical_output[
                        "observed_output_evidence_sha256"
                    ],
                    "observed_output_evidence_sha256": output_sha256,
                    "reset_output_evidence_sha256": reset_output_sha256,
                    "base_outputs_counters_and_loss_exact": True,
                    "inherited_mass_and_projected_trace_exact": True,
                    "reset_outputs_counters_and_full_trace_exact": True,
                    "answer_cross_entropy": first["answer_cross_entropy"],
                    "hidden_sha256": first["hidden_sha256"],
                    "logits_sha256": first["logits_sha256"],
                    "counter_stream_sha256": first["counter_stream_sha256"],
                    "episodic_call_stream_sha256": first[
                        "episodic_call_stream_sha256"
                    ],
                    "source_record_sha256": source_record_sha256,
                    "full_trace_sha256": first_summary["trace_sha256"],
                    "reset_full_trace_sha256": replay_summary["trace_sha256"],
                    "slot_shard_file_sha256": descriptor["file_sha256"],
                    "slot_mass_reconstruction_max_abs": first_summary[
                        "slot_mass_reconstruction_max_abs"
                    ],
                    "episodic_component_reconstruction_max_abs": first_summary[
                        "episodic_component_reconstruction_max_abs"
                    ],
                }
            )
            trace.reset()
    finally:
        trace.close()

    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _RESULT_EXPERIMENT,
        "protocol": {
            "path": str(context["slot_simplex_protocol_path"]),
            "sha256": context["slot_simplex_protocol_sha256"],
        },
        "inherited_head_mass_trace_manifest": {
            "path": str(context["head_mass_manifest_path"]),
            "sha256": context["head_mass_manifest_sha256"],
        },
        "format": "safetensors",
        "stored_tensors": list(_SLOT_TRACE_KEYS),
        "record_order": list(range(_RECORDS)),
        "shards": manifest_rows,
        "confirmation_split_opened": False,
    }
    manifest_path = directory / "manifest.json"
    atomic_json(manifest_path, manifest)
    arrays = _load_stacked_arrays(context, manifest_rows, directory)
    _progress("loading authenticated BF16 output projections")
    output_projection, projection_hashes = mass._load_output_projections(context)
    expected_projection_hashes = context["joint_gamma_protocol"][
        "output_projection"
    ]["tensor_sha256"]
    if projection_hashes != expected_projection_hashes:
        raise ValueError("slot-simplex output projection hashes changed")

    _progress("running frozen constructible nine-way simplex arm")
    constructible = run_slot_simplex_oracle_from_arrays(
        arrays,
        output_projection,
        row_batch_size=frozen["oracle_method"]["row_batch_size"],
        maximum_iterations=frozen["oracle_method"]["maximum_iterations"],
        relative_gap_tolerance=frozen["oracle_method"][
            "relative_objective_gap_target"
        ],
        include_exact_native_anchor=False,
    )
    constructible_report = _summarize_oracle_arm(
        constructible,
        component_count=_CONSTRUCTIBLE_COMPONENTS,
        arm="constructible_regular_plus_eight_slots",
    )
    _progress("running frozen ten-way exact-native-anchor optimistic hull")
    optimistic_hull = run_slot_simplex_oracle_from_arrays(
        arrays,
        output_projection,
        row_batch_size=frozen["oracle_method"]["row_batch_size"],
        maximum_iterations=frozen["oracle_method"]["maximum_iterations"],
        relative_gap_tolerance=frozen["oracle_method"][
            "relative_objective_gap_target"
        ],
        include_exact_native_anchor=True,
    )
    if not np.array_equal(
        constructible.target_energy,
        optimistic_hull.target_energy,
    ):
        raise ValueError("slot-simplex arm target energies changed")
    optimistic_hull_report = _summarize_oracle_arm(
        optimistic_hull,
        component_count=_OPTIMISTIC_COMPONENTS,
        arm="optimistic_exact_native_anchor_hull",
    )
    feasible_passed = bool(constructible_report["feasible_gate_passed"])
    optimistic_passed = bool(
        optimistic_hull_report["optimistic_gate_passed"]
    )
    if feasible_passed and not optimistic_passed:
        raise ValueError(
            "slot-simplex constructible pass exceeded optimistic hull bound"
        )
    decisive_failure = bool(
        optimistic_hull_report["qualification"]["passed"]
        and not optimistic_passed
    )
    status = (
        "train_episodic_slot_simplex_gate_passed"
        if feasible_passed
        else (
            "train_episodic_slot_simplex_gate_failed"
            if decisive_failure
            else "train_episodic_slot_simplex_gate_inconclusive"
        )
    )
    post = _screen_post_authentication(context)
    if not all(post.values()):
        raise ValueError("slot-simplex post-run authentication failed")

    oracle = {
        "two_continuous_capacity_arms": True,
        "shared_solver_contract": {
            "maximum_iterations": frozen["oracle_method"]["maximum_iterations"],
            "relative_objective_gap_target": frozen["oracle_method"][
                "relative_objective_gap_target"
            ],
            "row_batch_size": frozen["oracle_method"]["row_batch_size"],
        },
        "constructible_arm": constructible_report,
        "optimistic_hull_arm": optimistic_hull_report,
        "constructible_feasible_gate_passed": feasible_passed,
        "optimistic_hull_certified_gate_passed": optimistic_passed,
        "decisive_failure": decisive_failure,
    }
    report = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _RESULT_EXPERIMENT,
        "status": status,
        "protocol": {
            "path": str(context["slot_simplex_protocol_path"]),
            "sha256": context["slot_simplex_protocol_sha256"],
        },
        "scope": {
            "split": "train",
            "records": _RECORDS,
            "positions_per_record": _POSITIONS,
            "trace_positions_per_record": len(_READ_POSITIONS),
            "same_state_capacity_evidence_only": True,
            "causal_rollout": False,
            "semantic_or_M3_gate_passed": False,
            "development_outcomes_used": False,
            "confirmation_split_opened": False,
        },
        "base_output_authentication": output_rows,
        "slot_trace_manifest": {
            "directory": str(directory),
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "shard_count": len(manifest_rows),
            "shards": manifest_rows,
        },
        "inherited_head_mass_trace_manifest": {
            "path": str(context["head_mass_manifest_path"]),
            "sha256": context["head_mass_manifest_sha256"],
            "shards": _RECORDS,
        },
        "output_projection": {
            "source": str(context["non_mlp_path"]),
            "dtype": "authenticated_BF16_loaded_as_float32",
            "orientation": "isolated_head_value_delta @ o_proj.weight.T",
            "tensor_sha256": projection_hashes,
        },
        "oracle": oracle,
        "resource_contract": dict(frozen["resource_contract"]),
        "decision": {
            "train_slot_simplex_capacity_gate_passed": feasible_passed,
            "certified_optimistic_gate_passed": optimistic_passed,
            "failure_is_decisive": decisive_failure,
            "semantic_or_M3_gate_passed": False,
            "native_causal_integration_authorized": False,
            "development_authorized": False,
            "confirmation_authorized": False,
            "train_only_causal_slot_selector_authorized": feasible_passed,
            "next_step": (
                frozen["authorized_next_step_on_feasible_pass"]
                if feasible_passed
                else (
                    frozen["failure_interpretation"]
                    if decisive_failure
                    else (
                        "improve only the certified convex solve until its "
                        "optimistic and feasible gate decisions agree"
                    )
                )
            ),
        },
        "post_run_authentication": post,
        "confirmation_split_opened": False,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output, report)
    _progress(f"slot-simplex result written to {output}")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train-only OLMoE per-slot episodic value-basis oracle",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    parity = commands.add_parser("parity")
    parity.add_argument("--joint-protocol", required=True)
    parity.add_argument("--joint-protocol-sha256", required=True)
    parity.add_argument("--joint-result", required=True)
    parity.add_argument("--joint-result-sha256", required=True)
    parity.add_argument("--trace-library", required=True)
    parity.add_argument("--trace-library-sha256", required=True)
    parity.add_argument("--out", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--joint-protocol", required=True)
    freeze.add_argument("--joint-protocol-sha256", required=True)
    freeze.add_argument("--joint-result", required=True)
    freeze.add_argument("--joint-result-sha256", required=True)
    freeze.add_argument("--trace-library", required=True)
    freeze.add_argument("--trace-library-sha256", required=True)
    freeze.add_argument("--parity-report", required=True)
    freeze.add_argument("--parity-report-sha256", required=True)
    freeze.add_argument("--out", required=True)
    screen = commands.add_parser("screen")
    screen.add_argument("--protocol", required=True)
    screen.add_argument("--protocol-sha256", required=True)
    screen.add_argument("--shard-dir", required=True)
    screen.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "parity":
        generate_trace_parity_report(
            joint_protocol=args.joint_protocol,
            joint_protocol_sha256=args.joint_protocol_sha256,
            joint_result=args.joint_result,
            joint_result_sha256=args.joint_result_sha256,
            trace_library=args.trace_library,
            trace_library_sha256=args.trace_library_sha256,
            out=args.out,
        )
    elif args.command == "freeze":
        freeze_slot_simplex_protocol(
            joint_protocol=args.joint_protocol,
            joint_protocol_sha256=args.joint_protocol_sha256,
            joint_result=args.joint_result,
            joint_result_sha256=args.joint_result_sha256,
            trace_library=args.trace_library,
            trace_library_sha256=args.trace_library_sha256,
            parity_report=args.parity_report,
            parity_report_sha256=args.parity_report_sha256,
            out=args.out,
        )
    else:
        screen_slot_simplex_oracle(
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            shard_dir=args.shard_dir,
            out=args.out,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
