"""Cached joint output-targeted per-head gamma capacity oracle.

The native head-mass experiment already captured every tensor needed by this
experiment.  This module authenticates those frozen train-only shards and
builds a quadratic objective for choosing all 16 head gamma codes jointly
after the attention output projection.  It never opens the development or
confirmation corpus and it does not execute another native teacher rollout.

For one state and layer, let ``B`` be the exact beta-zero pre-Wo attention
output, ``R/E`` the regular/episodic weighted-value numerators, and ``mr/me``
their probability masses.  Per head,

``r = R / mr``, ``e = E / me``, ``q = r - B``, and ``d = e - r``.

Every non-anchor gamma candidate has pre-Wo correction ``q + p_gamma*d``,
where ``p_gamma = gamma*me/(mr + gamma*me)``.  Code four is deliberately
represented by ``(0, 0)`` so the exact native beta-zero output remains the
anchor even in the presence of floating-point reconstruction error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import engram.evaluation.olmoe_joint_gamma_solver as gamma_solver
import engram.evaluation.olmoe_retrieval_episodic_head_mass_oracle as mass
from engram.utils import atomic_json, sha256_file, sha256_json


_SCHEMA_VERSION = 1
_EXPECTED_HEAD_MASS_PROTOCOL_SHA256 = (
    "fe09689452e6ae4f1a1b15332c61c1cc990cfc29b6a8b0d5a1758d9490a93af5"
)
_EXPECTED_HEAD_MASS_RESULT_SHA256 = (
    "f7060e7373c5faf8f154891e93efad35659723d8e3f04d83638b62fa9cf72596"
)
_EXPECTED_HEAD_MASS_MANIFEST_SHA256 = (
    "93df0a554744b97e7436b9a8b4bb71473bc21fa9f6c90985431274859164e0b6"
)
_PROTOCOL_EXPERIMENT = "olmoe_q7_retrieval_episodic_joint_gamma_oracle_protocol"
_PROTOCOL_STATUS = "frozen_before_cached_joint_gamma_execution"
_RESULT_EXPERIMENT = "olmoe_q7_retrieval_episodic_joint_gamma_oracle_train_screen"
_TIE_PRIORITY = mass._TIE_PRIORITY
_BASE_CODE = 4
_FEATURE_NAMES = ("q_anchor_offset", "episodic_direction")
_CONTINUOUS_MAXIMUM_SWEEPS = 64
_CONTINUOUS_RELATIVE_GAP_TOLERANCE = 1.0e-7
_SOURCE_FILES = (
    "src/engram/evaluation/olmoe_joint_gamma_solver.py",
    "src/engram/evaluation/olmoe_retrieval_episodic_joint_gamma_oracle.py",
)


def _progress(message: str) -> None:
    print(
        f"[retrieval-episodic-joint-gamma-oracle] {message}",
        file=sys.stderr,
        flush=True,
    )


@dataclass(frozen=True)
class JointQuadraticInputs:
    """Batched quadratic and finite candidate coefficients.

    The objective for batch row ``n`` is

    ``target_energy[n] - 2*linear[n]·x + x.T*gram[n]*x``.

    Batch rows use record-major, read-position-major, layer-minor order.
    """

    gram: np.ndarray
    linear: np.ndarray
    target_energy: np.ndarray
    candidates: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    batch_shape: tuple[int, int, int]
    feature_names: tuple[str, str]
    gram_factor_construction: str
    minimum_normalized_gram_eigenvalue: float
    maximum_gram_asymmetry: float
    base_component_reconstruction_max_abs: float
    float32_per_head_counterfactual_pre_wo_max_abs: float
    float32_uniform_code_projected_max_abs: float


def _source_inventory() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    return {name: sha256_file(repository / name) for name in _SOURCE_FILES}


def _validated_components(
    arrays: Mapping[str, np.ndarray],
    *,
    query_heads: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    required = {
        "base_attention_output",
        "regular_component",
        "episodic_component",
        "regular_mass",
        "episodic_mass",
        "target_residual",
    }
    if not required.issubset(arrays):
        raise ValueError("joint-gamma trace tensors are incomplete")
    base = np.ascontiguousarray(arrays["base_attention_output"], dtype=np.float32)
    regular = np.ascontiguousarray(arrays["regular_component"], dtype=np.float32)
    episodic = np.ascontiguousarray(arrays["episodic_component"], dtype=np.float32)
    regular_mass = np.ascontiguousarray(arrays["regular_mass"], dtype=np.float32)
    episodic_mass = np.ascontiguousarray(arrays["episodic_mass"], dtype=np.float32)
    target = np.ascontiguousarray(arrays["target_residual"], dtype=np.float32)
    if (
        query_heads <= 0
        or base.ndim != 4
        or base.shape != regular.shape
        or base.shape != episodic.shape
        or base.shape != target.shape
        or base.shape[-1] % query_heads
        or regular_mass.shape != base.shape[:-1] + (query_heads,)
        or episodic_mass.shape != regular_mass.shape
        or not all(
            np.isfinite(value).all()
            for value in (
                base,
                regular,
                episodic,
                regular_mass,
                episodic_mass,
                target,
            )
        )
        or np.any(regular_mass <= np.float32(0.0))
        or np.any(episodic_mass <= np.float32(0.0))
    ):
        raise ValueError("joint-gamma trace tensor shapes or values are invalid")
    partition_error = np.max(
        np.abs(regular_mass + episodic_mass - np.float32(1.0))
    )
    if partition_error > np.float32(2.0e-5):
        raise ValueError("joint-gamma masses do not partition one")
    return base, regular, episodic, regular_mass, episodic_mass, target


def _candidate_coefficients(
    regular_mass: np.ndarray,
    episodic_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the eight discrete coefficients and continuous box bounds."""

    regular = np.ascontiguousarray(regular_mass, dtype=np.float64)
    episodic = np.ascontiguousarray(episodic_mass, dtype=np.float64)
    if (
        regular.shape != episodic.shape
        or regular.ndim < 1
        or not regular.size
        or not np.isfinite(regular).all()
        or not np.isfinite(episodic).all()
        or np.any(regular <= 0.0)
        or np.any(episodic <= 0.0)
        or np.max(np.abs(regular + episodic - 1.0)) > 2.0e-5
    ):
        raise ValueError("joint-gamma candidate masses are invalid")
    gamma = mass._gamma_multipliers().astype(np.float64)
    denominator = regular[..., None] + episodic[..., None] * gamma
    if np.any(denominator <= 0.0):
        raise ValueError("joint-gamma candidate denominator is invalid")
    p_gamma = episodic[..., None] * gamma / denominator
    candidates = np.empty(regular.shape + (8, 2), dtype=np.float64)
    candidates[..., :, 0] = 1.0
    candidates[..., :, 1] = p_gamma
    candidates[..., _BASE_CODE, :] = 0.0
    lower = np.zeros(regular.shape + (2,), dtype=np.float64)
    upper = np.empty_like(lower)
    upper[..., 0] = 1.0
    upper[..., 1] = p_gamma[..., 7]
    if not np.isfinite(candidates).all() or np.any(upper < lower):
        raise ValueError("joint-gamma candidate coefficients are non-finite")
    return (
        np.ascontiguousarray(candidates),
        np.ascontiguousarray(lower),
        np.ascontiguousarray(upper),
    )


def _qd_basis(
    base_attention_output: np.ndarray,
    regular_component: np.ndarray,
    episodic_component: np.ndarray,
    regular_mass: np.ndarray,
    episodic_mass: np.ndarray,
    *,
    query_heads: int,
) -> np.ndarray:
    """Construct per-head ``(q, d)`` vectors in float64."""

    base = np.ascontiguousarray(base_attention_output, dtype=np.float64)
    regular = np.ascontiguousarray(regular_component, dtype=np.float64)
    episodic = np.ascontiguousarray(episodic_component, dtype=np.float64)
    regular_probability = np.ascontiguousarray(regular_mass, dtype=np.float64)
    episodic_probability = np.ascontiguousarray(episodic_mass, dtype=np.float64)
    if (
        base.shape != regular.shape
        or base.shape != episodic.shape
        or base.ndim < 1
        or query_heads <= 0
        or base.shape[-1] % query_heads
        or regular_probability.shape != base.shape[:-1] + (query_heads,)
        or episodic_probability.shape != regular_probability.shape
        or not all(
            np.isfinite(value).all()
            for value in (
                base,
                regular,
                episodic,
                regular_probability,
                episodic_probability,
            )
        )
        or np.any(regular_probability <= 0.0)
        or np.any(episodic_probability <= 0.0)
    ):
        raise ValueError("joint-gamma q,d basis inputs are invalid")
    head_dimension = base.shape[-1] // query_heads
    head_shape = base.shape[:-1] + (query_heads, head_dimension)
    base_heads = base.reshape(head_shape)
    regular_mean = regular.reshape(head_shape) / regular_probability[..., None]
    episodic_mean = episodic.reshape(head_shape) / episodic_probability[..., None]
    q = regular_mean - base_heads
    d = episodic_mean - regular_mean
    basis = np.stack((q, d), axis=-2)
    if not np.isfinite(basis).all():
        raise ValueError("joint-gamma q,d basis is non-finite")
    return np.ascontiguousarray(basis, dtype=np.float64)


def _project_qd_basis(
    basis: np.ndarray,
    output_projection: np.ndarray,
) -> np.ndarray:
    """Project isolated head bases through native ``o_proj.weight.T``."""

    values = np.ascontiguousarray(basis, dtype=np.float64)
    weights = np.ascontiguousarray(output_projection, dtype=np.float64)
    if values.ndim != 4 or values.shape[2] != 2:
        raise ValueError("joint-gamma projected basis shape is invalid")
    _batch, heads, _features, head_dimension = values.shape
    hidden = heads * head_dimension
    if (
        weights.shape != (hidden, hidden)
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
    ):
        raise ValueError("joint-gamma output projection shape is invalid")
    # W[o, i] multiplies input coordinate i.  Splitting W's input axis gives
    # one [output, head_dimension] block per attention head.
    head_weights = weights.reshape(hidden, heads, head_dimension).transpose(1, 0, 2)
    projected = np.einsum(
        "nhfi,hoi->nhfo",
        values,
        head_weights,
        optimize=True,
    )
    if not np.isfinite(projected).all():
        raise ValueError("joint-gamma projected basis is non-finite")
    return np.ascontiguousarray(projected, dtype=np.float64)


def _quadratic_from_projected_basis(
    projected_basis: np.ndarray,
    target_residual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projected = np.ascontiguousarray(projected_basis, dtype=np.float64)
    target = np.ascontiguousarray(target_residual, dtype=np.float64)
    if (
        projected.ndim != 4
        or projected.shape[2] != 2
        or target.shape != (projected.shape[0], projected.shape[3])
        or not np.isfinite(projected).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("joint-gamma quadratic basis or target is invalid")
    gram = np.einsum(
        "nhfo,nkgo->nhfkg",
        projected,
        projected,
        optimize=True,
    )
    linear = np.einsum(
        "nhfo,no->nhf",
        projected,
        target,
        optimize=True,
    )
    energy = np.einsum("no,no->n", target, target, optimize=True)
    if np.any(energy <= 0.0):
        raise ValueError("joint-gamma target energy must be positive")
    return (
        np.ascontiguousarray(gram, dtype=np.float64),
        np.ascontiguousarray(linear, dtype=np.float64),
        np.ascontiguousarray(energy, dtype=np.float64),
    )


def _reconstruct_pre_wo_delta(
    arrays: Mapping[str, np.ndarray],
    codes: np.ndarray,
    *,
    query_heads: int,
) -> np.ndarray:
    """Reconstruct a finite candidate correction from its q,d coefficients."""

    base, regular, episodic, regular_mass, episodic_mass, _target = (
        _validated_components(arrays, query_heads=query_heads)
    )
    selected = np.ascontiguousarray(codes)
    expected = base.shape[:-1] + (query_heads,)
    if (
        selected.shape != expected
        or not np.issubdtype(selected.dtype, np.integer)
        or np.any(selected < 0)
        or np.any(selected > 7)
    ):
        raise ValueError("joint-gamma selected codes are invalid")
    basis = _qd_basis(
        base,
        regular,
        episodic,
        regular_mass,
        episodic_mass,
        query_heads=query_heads,
    )
    candidates, _lower, _upper = _candidate_coefficients(
        regular_mass,
        episodic_mass,
    )
    coefficients = np.take_along_axis(
        candidates,
        selected[..., None, None].astype(np.int64),
        axis=-2,
    )[..., 0, :]
    delta_heads = np.einsum(
        "...hf,...hfi->...hi",
        coefficients,
        basis,
        optimize=True,
    )
    return np.ascontiguousarray(delta_heads.reshape(base.shape), dtype=np.float64)


def build_joint_quadratic_inputs(
    arrays: Mapping[str, np.ndarray],
    output_projection: np.ndarray,
    *,
    query_heads: int = mass._QUERY_HEADS,
    row_batch_size: int = 32,
) -> JointQuadraticInputs:
    """Build the cached joint post-Wo objective without another model rollout."""

    base, regular, episodic, regular_mass, episodic_mass, target = (
        _validated_components(arrays, query_heads=query_heads)
    )
    weights = np.ascontiguousarray(output_projection, dtype=np.float32)
    weights_float64 = np.ascontiguousarray(weights, dtype=np.float64)
    records, read_rows, layers, hidden = base.shape
    if (
        row_batch_size <= 0
        or weights.shape != (layers, hidden, hidden)
        or not np.isfinite(weights).all()
    ):
        raise ValueError("joint-gamma output projection stack is invalid")
    rows_per_layer = records * read_rows
    total_rows = rows_per_layer * layers
    gram = np.empty((total_rows, query_heads, 2, query_heads, 2), dtype=np.float64)
    linear = np.empty((total_rows, query_heads, 2), dtype=np.float64)
    target_energy = np.empty(total_rows, dtype=np.float64)
    maximum_candidate_error = 0.0
    maximum_projected_candidate_error = 0.0
    minimum_normalized_eigenvalue = float("inf")
    maximum_asymmetry = 0.0
    base_reconstruction_error = float(
        np.max(np.abs((regular + episodic) - base))
    )

    for layer in range(layers):
        _progress(f"building q,d output objective layer {layer + 1}/{layers}")
        layer_weight_float64 = weights_float64[layer]
        layer_base = base[:, :, layer].reshape(rows_per_layer, hidden)
        layer_regular = regular[:, :, layer].reshape(rows_per_layer, hidden)
        layer_episodic = episodic[:, :, layer].reshape(rows_per_layer, hidden)
        layer_regular_mass = regular_mass[:, :, layer].reshape(
            rows_per_layer,
            query_heads,
        )
        layer_episodic_mass = episodic_mass[:, :, layer].reshape(
            rows_per_layer,
            query_heads,
        )
        layer_target = target[:, :, layer].reshape(rows_per_layer, hidden)
        for begin in range(0, rows_per_layer, row_batch_size):
            end = min(begin + row_batch_size, rows_per_layer)
            basis = _qd_basis(
                layer_base[begin:end],
                layer_regular[begin:end],
                layer_episodic[begin:end],
                layer_regular_mass[begin:end],
                layer_episodic_mass[begin:end],
                query_heads=query_heads,
            )
            projected = _project_qd_basis(basis, layer_weight_float64)
            batch_gram, batch_linear, batch_energy = (
                _quadratic_from_projected_basis(
                    projected,
                    layer_target[begin:end],
                )
            )
            flat_batch_gram = batch_gram.reshape(
                end - begin,
                query_heads * 2,
                query_heads * 2,
            )
            asymmetry = np.max(
                np.abs(
                    flat_batch_gram
                    - np.swapaxes(flat_batch_gram, axis1=1, axis2=2)
                )
            )
            maximum_asymmetry = max(maximum_asymmetry, float(asymmetry))
            symmetric = 0.5 * (
                flat_batch_gram + np.swapaxes(flat_batch_gram, axis1=1, axis2=2)
            )
            eigenvalues = np.linalg.eigvalsh(symmetric)
            eigen_scale = np.maximum(
                1.0,
                np.max(np.diagonal(symmetric, axis1=1, axis2=2), axis=1),
            )
            normalized_minimum = eigenvalues[:, 0] / eigen_scale
            minimum_normalized_eigenvalue = min(
                minimum_normalized_eigenvalue,
                float(np.min(normalized_minimum)),
            )
            destination = np.arange(begin, end, dtype=np.int64) * layers + layer
            gram[destination] = batch_gram
            linear[destination] = batch_linear
            target_energy[destination] = batch_energy

            # Qualify the algebra against the established finite-grid formula
            # while the small basis batch is resident.  Code four is an exact
            # zero correction by contract; every other code uses q+p*d.
            batch_arrays = {
                "base_attention_output": layer_base[begin:end, None, None, :],
                "regular_component": layer_regular[begin:end, None, None, :],
                "episodic_component": layer_episodic[begin:end, None, None, :],
                "regular_mass": layer_regular_mass[begin:end, None, None, :],
                "episodic_mass": layer_episodic_mass[begin:end, None, None, :],
                "target_residual": layer_target[begin:end, None, None, :],
            }
            batch_candidates, _batch_lower, _batch_upper = (
                _candidate_coefficients(
                    layer_regular_mass[begin:end],
                    layer_episodic_mass[begin:end],
                )
            )
            for code in range(8):
                selected = np.full(
                    (end - begin, 1, 1, query_heads),
                    code,
                    dtype=np.uint8,
                )
                reconstructed = np.einsum(
                    "nhf,nhfi->nhi",
                    batch_candidates[:, :, code, :],
                    basis,
                    optimize=True,
                ).reshape(end - begin, hidden)
                direct = mass._counterfactual_pre_wo(
                    batch_arrays["base_attention_output"],
                    batch_arrays["regular_component"],
                    batch_arrays["episodic_component"],
                    batch_arrays["regular_mass"],
                    batch_arrays["episodic_mass"],
                    selected,
                    query_heads=query_heads,
                ).reshape(end - begin, hidden)
                direct = direct.astype(np.float64) - layer_base[begin:end].astype(
                    np.float64
                )
                maximum_candidate_error = max(
                    maximum_candidate_error,
                    float(np.max(np.abs(reconstructed - direct))),
                )
                projected_difference = (
                    (reconstructed - direct) @ layer_weight_float64.T
                )
                maximum_projected_candidate_error = max(
                    maximum_projected_candidate_error,
                    float(np.max(np.abs(projected_difference))),
                )

    flat_regular_mass = regular_mass.reshape(total_rows, query_heads)
    flat_episodic_mass = episodic_mass.reshape(total_rows, query_heads)
    candidates, lower, upper = _candidate_coefficients(
        flat_regular_mass,
        flat_episodic_mass,
    )
    if (
        not np.isfinite(gram).all()
        or not np.isfinite(linear).all()
        or not np.isfinite(target_energy).all()
        or minimum_normalized_eigenvalue < -1.0e-10
        or maximum_candidate_error > 5.0e-6
        or maximum_projected_candidate_error > 5.0e-5
    ):
        raise ValueError("joint-gamma quadratic qualification failed")
    return JointQuadraticInputs(
        gram=gram,
        linear=linear,
        target_energy=target_energy,
        candidates=candidates,
        lower=lower,
        upper=upper,
        batch_shape=(records, read_rows, layers),
        feature_names=_FEATURE_NAMES,
        gram_factor_construction=(
            "A is the float64 projection of isolated per-head q,d bases "
            "through authenticated BF16-as-float32 Wo; gram=A.T@A"
        ),
        minimum_normalized_gram_eigenvalue=minimum_normalized_eigenvalue,
        maximum_gram_asymmetry=maximum_asymmetry,
        base_component_reconstruction_max_abs=base_reconstruction_error,
        float32_per_head_counterfactual_pre_wo_max_abs=maximum_candidate_error,
        float32_uniform_code_projected_max_abs=(
            maximum_projected_candidate_error
        ),
    )


def _validate_head_mass_failure(
    value: Any,
    *,
    protocol_path: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("joint-gamma head-mass result is invalid")
    scope = value.get("scope")
    decision = value.get("decision")
    oracle = value.get("oracle")
    post = value.get("post_run_authentication")
    manifest = value.get("trace_manifest")
    if (
        value.get("schema_version") != mass._SCHEMA_VERSION
        or value.get("experiment") != mass._RESULT_EXPERIMENT
        or value.get("status") != "train_episodic_head_mass_oracle_gate_failed"
        or value.get("protocol")
        != {
            "path": str(protocol_path),
            "sha256": _EXPECTED_HEAD_MASS_PROTOCOL_SHA256,
        }
        or value.get("confirmation_split_opened") is not False
        or not isinstance(scope, Mapping)
        or scope.get("split") != "train"
        or scope.get("same_state_capacity_evidence_only") is not True
        or scope.get("development_outcomes_used") is not False
        or scope.get("confirmation_split_opened") is not False
        or not isinstance(oracle, Mapping)
        or oracle.get("passed") is not False
        or not isinstance(decision, Mapping)
        or decision.get("train_head_mass_capacity_gate_passed") is not False
        or decision.get("native_causal_integration_authorized") is not False
        or decision.get("development_authorized") is not False
        or decision.get("confirmation_authorized") is not False
        or not isinstance(post, Mapping)
        or not post
        or not all(check is True for check in post.values())
        or not isinstance(manifest, Mapping)
        or manifest.get("sha256") != _EXPECTED_HEAD_MASS_MANIFEST_SHA256
        or manifest.get("shard_count") != mass._RECORDS
        or not isinstance(manifest.get("shards"), list)
        or len(manifest["shards"]) != mass._RECORDS
    ):
        raise ValueError("joint-gamma head-mass failure contract changed")
    return {
        "status": value["status"],
        "global_recovery": oracle.get("metrics", {}).get("global", {}).get("recovery"),
        "failure_scope": decision.get("failure_scope"),
        "trace_manifest_sha256": manifest["sha256"],
        "confirmation_split_opened": False,
    }


def _authenticate_head_mass_inputs(
    *,
    head_mass_protocol: str | Path,
    head_mass_protocol_sha256: str,
    head_mass_result: str | Path,
    head_mass_result_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if head_mass_protocol_sha256.lower() != _EXPECTED_HEAD_MASS_PROTOCOL_SHA256:
        raise ValueError("joint-gamma head-mass protocol root changed")
    if head_mass_result_sha256.lower() != _EXPECTED_HEAD_MASS_RESULT_SHA256:
        raise ValueError("joint-gamma head-mass result root changed")
    context, training, frozen = mass._authenticate_protocol(
        head_mass_protocol,
        head_mass_protocol_sha256,
    )
    protocol_path = Path(context["head_mass_protocol_path"]).resolve()
    result_path = mass.capacity.bias.episodic._checked_file(
        head_mass_result,
        head_mass_result_sha256,
        "joint-gamma head-mass result",
    )
    result = mass.capacity.bias.rank.retrieval._read_json(
        result_path,
        "joint-gamma head-mass result",
    )
    failure = _validate_head_mass_failure(result, protocol_path=protocol_path)
    manifest_binding = result["trace_manifest"]
    manifest_path = mass.capacity.bias.episodic._checked_file(
        manifest_binding.get("path"),
        manifest_binding.get("sha256"),
        "joint-gamma cached trace manifest",
    )
    manifest = mass.capacity.bias.rank.retrieval._read_json(
        manifest_path,
        "joint-gamma cached trace manifest",
    )
    if (
        manifest.get("schema_version") != mass._SCHEMA_VERSION
        or manifest.get("experiment") != mass._RESULT_EXPERIMENT
        or manifest.get("format") != "safetensors"
        or manifest.get("record_order") != list(range(mass._RECORDS))
        or manifest.get("shards") != manifest_binding["shards"]
        or manifest.get("confirmation_split_opened") is not False
    ):
        raise ValueError("joint-gamma cached trace manifest changed")
    context = dict(context)
    context.update(
        {
            "head_mass_result_path": result_path,
            "head_mass_result_sha256": head_mass_result_sha256.lower(),
            "head_mass_result": result,
            "head_mass_manifest_path": manifest_path,
            "head_mass_manifest_sha256": manifest_binding["sha256"],
            "head_mass_manifest": manifest,
        }
    )
    return context, training, failure


def _build_protocol(
    *,
    context: Mapping[str, Any],
    training: Mapping[str, Any],
    failure: Mapping[str, Any],
) -> dict[str, Any]:
    head_mass_protocol = context["head_mass_protocol"]
    return {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _PROTOCOL_EXPERIMENT,
        "status": _PROTOCOL_STATUS,
        "head_mass_protocol": {
            "path": str(context["head_mass_protocol_path"]),
            "sha256": context["head_mass_protocol_sha256"],
        },
        "head_mass_result": {
            "path": str(context["head_mass_result_path"]),
            "sha256": context["head_mass_result_sha256"],
            "authenticated_failure": dict(failure),
        },
        "cached_trace_manifest": {
            "path": str(context["head_mass_manifest_path"]),
            "sha256": context["head_mass_manifest_sha256"],
            "shards": mass._RECORDS,
        },
        "training_checkpoint": dict(head_mass_protocol["training_checkpoint"]),
        "training_checkpoint_payload_sha256": sha256_json(training),
        "output_projection": {
            "source": str(context["non_mlp_path"]),
            "dtype": "authenticated_BF16_loaded_as_float32",
            "orientation": "isolated_head_delta @ o_proj.weight.T",
            "tensor_sha256": dict(
                head_mass_protocol["trace_parity"]["report"]["parity"][
                    "output_projection_tensor_sha256"
                ]
            ),
        },
        "oracle_method": {
            "single_arm": True,
            "features": list(_FEATURE_NAMES),
            "gamma_table": [dict(row) for row in mass._gamma_table()],
            "base_code": _BASE_CODE,
            "tie_priority": list(_TIE_PRIORITY),
            "candidate_coefficients": (
                "code four=(0,0) exact native base; every other code="
                "(1,gamma*me/(mr+gamma*me)) over projected (q,d)"
            ),
            "continuous_diagnostic": (
                "box relaxation z in [0,1], p in [0,p_gamma8] per head; "
                "convex diagnostic lower-error bound because its Gram is "
                "formed as A.T@A; it cannot authorize progression"
            ),
            "continuous_solver": (
                "exact two-variable box minimization inside each head with "
                "forward/reverse block sweeps and a float64 Frank-Wolfe "
                "objective-gap certificate after every sweep; fixed maximum "
                f"{_CONTINUOUS_MAXIMUM_SWEEPS} sweeps and relative gap target "
                f"{_CONTINUOUS_RELATIVE_GAP_TOLERANCE:.1e}"
            ),
            "discrete_selection": (
                "joint eight-code grouped solver including all cross-head "
                "o_proj Gram terms, deterministic multistart coordinate "
                "descent, and exhaustive candidate search within each one- "
                "or two-head move"
            ),
            "discrete_optimality_scope": (
                "verified one- and two-head-flip local optimum only; no exact "
                "or global optimum claim over the complete 8^16 grid"
            ),
            "direct_gate_metric": (
                "reconstruct selected finite candidates with the established "
                "float32 counterfactual and float32 o_proj path; quadratic "
                "objectives are diagnostics"
            ),
            "target": "exact same-state native W128-minus-K256 post-Wo residual",
            "energy_aggregation": "squared Frobenius energy before ratios",
            "counterfactual_updates_hidden_or_cache": False,
        },
        "progression_gate": {
            "finite": True,
            "minimum_global_recovery": 0.50,
            "minimum_every_sequence_recovery": 0.25,
            "minimum_every_block_entry_position_recovery": 0.25,
            "minimum_positive_recovery_layers": 12,
            "discrete_projected_objective_never_worse_than_base": True,
            "continuous_objective_gap_certificate_required": True,
            "discrete_one_and_two_flip_local_optimality_required": True,
            "deterministic_metric_and_code_replay_required": True,
        },
        "resource_contract": dict(head_mass_protocol["resource_contract"]),
        "scope": {
            "split": "train",
            "cached_same_state_capacity_evidence_only": True,
            "learned_predictor": False,
            "causal_rollout": False,
            "semantic_or_M3_pass": False,
            "development_outcomes_used": False,
            "confirmation_file_access_permitted": False,
        },
        "authorized_next_step_on_discrete_pass": (
            "freeze a train-only causal predictor for the selected 3-bit "
            "gamma codes with rolled-out native state and full traffic accounting"
        ),
        "failure_interpretation": {
            "continuous_failure": (
                "closes the cached same-state bounded per-head q,d relaxation "
                "at fixed K256 only when its optimistic recovery upper bound "
                "also fails the frozen gate"
            ),
            "continuous_pass_discrete_failure": (
                "closes only the deterministic multistart one/two-flip "
                "solution; it does not close the global eight-code grid"
            ),
        },
        # Descriptor authentication is inherited by value.  Its file is not
        # opened, resolved, stated, or hashed by this module.
        "authenticated_confirmation_descriptor": dict(
            head_mass_protocol["authenticated_confirmation_descriptor"]
        ),
        "source_sha256": _source_inventory(),
        "confirmation_split_opened": False,
    }


def freeze_joint_gamma_protocol(
    *,
    head_mass_protocol: str | Path,
    head_mass_protocol_sha256: str,
    head_mass_result: str | Path,
    head_mass_result_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    output = mass.capacity.bias.rank.retrieval._new_output(
        out,
        "joint-gamma protocol",
    )
    _progress("authenticating head-mass roots before protocol freeze")
    context, training, failure = _authenticate_head_mass_inputs(
        head_mass_protocol=head_mass_protocol,
        head_mass_protocol_sha256=head_mass_protocol_sha256,
        head_mass_result=head_mass_result,
        head_mass_result_sha256=head_mass_result_sha256,
    )
    protocol = _build_protocol(
        context=context,
        training=training,
        failure=failure,
    )
    atomic_json(output, protocol)
    _progress(f"joint-gamma protocol written to {output}")
    return {"path": str(output), "sha256": sha256_file(output), "protocol": protocol}


def _authenticate_protocol(
    protocol: str | Path,
    protocol_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = mass.capacity.bias.episodic._checked_file(
        protocol,
        protocol_sha256,
        "joint-gamma protocol",
    )
    value = mass.capacity.bias.rank.retrieval._read_json(
        source,
        "joint-gamma protocol",
    )
    protocol_binding = value.get("head_mass_protocol")
    result_binding = value.get("head_mass_result")
    if not isinstance(protocol_binding, Mapping) or not isinstance(
        result_binding,
        Mapping,
    ):
        raise ValueError("joint-gamma protocol bindings are invalid")
    context, training, failure = _authenticate_head_mass_inputs(
        head_mass_protocol=protocol_binding.get("path"),
        head_mass_protocol_sha256=protocol_binding.get("sha256"),
        head_mass_result=result_binding.get("path"),
        head_mass_result_sha256=result_binding.get("sha256"),
    )
    expected = _build_protocol(
        context=context,
        training=training,
        failure=failure,
    )
    if value != expected:
        raise ValueError("joint-gamma frozen protocol changed")
    context = dict(context)
    context.update(
        {
            "joint_gamma_protocol_path": source,
            "joint_gamma_protocol_sha256": protocol_sha256.lower(),
            "joint_gamma_protocol": expected,
        }
    )
    return context, training, expected


def load_cached_joint_quadratic(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    row_batch_size: int = 32,
) -> JointQuadraticInputs:
    """Authenticate frozen shards and build the objective; perform no rollout."""

    context, _training, frozen = _authenticate_protocol(protocol, protocol_sha256)
    stacked, output_projection = _load_cached_arrays_and_projection(
        context,
        frozen,
    )
    return build_joint_quadratic_inputs(
        stacked,
        output_projection,
        row_batch_size=row_batch_size,
    )


def _load_cached_arrays_and_projection(
    context: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {name: [] for name in mass._TRACE_KEYS}
    directory = Path(context["head_mass_manifest_path"]).parent
    for descriptor in context["head_mass_manifest"]["shards"]:
        arrays = mass._validate_trace_shard(
            directory / descriptor["file"],
            descriptor,
        )
        for name in mass._TRACE_KEYS:
            rows[name].append(arrays[name])
    stacked = {
        name: np.ascontiguousarray(np.stack(values), dtype=np.float32)
        for name, values in rows.items()
    }
    output_projection, projection_hashes = mass._load_output_projections(context)
    expected_hashes = frozen["output_projection"]["tensor_sha256"]
    if projection_hashes != expected_hashes:
        raise ValueError("joint-gamma output projections changed")
    return stacked, output_projection


def _recovery_metrics_from_energy(
    target_energy: np.ndarray,
    error_energy: np.ndarray,
    *,
    batch_shape: tuple[int, int, int],
) -> dict[str, Any]:
    target = np.ascontiguousarray(target_energy, dtype=np.float64)
    error = np.ascontiguousarray(error_energy, dtype=np.float64)
    records, read_rows, layers = batch_shape
    if (
        target.shape != (records * read_rows * layers,)
        or error.shape != target.shape
        or not np.isfinite(target).all()
        or not np.isfinite(error).all()
        or np.any(target <= 0.0)
        or np.any(error < 0.0)
    ):
        raise ValueError("joint-gamma recovery energies are invalid")
    target_grid = target.reshape(batch_shape)
    error_grid = error.reshape(batch_shape)

    def metric(target_view: np.ndarray, error_view: np.ndarray) -> dict[str, float]:
        return mass._energy_metric(
            float(np.sum(target_view, dtype=np.float64)),
            float(np.sum(error_view, dtype=np.float64)),
        )

    global_metric = metric(target_grid, error_grid)
    sequences = [
        {"record_index": record, **metric(target_grid[record], error_grid[record])}
        for record in range(records)
    ]
    layer_rows = [
        {
            "layer": layer,
            **metric(target_grid[:, :, layer], error_grid[:, :, layer]),
        }
        for layer in range(layers)
    ]
    blocks = []
    for position in mass._BLOCK_ENTRY_POSITIONS:
        offset = mass._READ_POSITIONS.index(position)
        blocks.append(
            {
                "position": position,
                **metric(target_grid[:, offset], error_grid[:, offset]),
            }
        )
    positive_layers = sum(row["recovery"] > 0.0 for row in layer_rows)
    gate = {
        "finite": True,
        "global_recovery_at_least_0_50": global_metric["recovery"] >= 0.50,
        "every_sequence_recovery_at_least_0_25": all(
            row["recovery"] >= 0.25 for row in sequences
        ),
        "every_block_entry_recovery_at_least_0_25": all(
            row["recovery"] >= 0.25 for row in blocks
        ),
        "at_least_12_of_16_layers_positive_recovery": positive_layers >= 12,
    }
    gate["passed"] = all(gate.values())
    return {
        "global": global_metric,
        "heldout_sequences": sequences,
        "layers": layer_rows,
        "block_entry_positions": blocks,
        "positive_recovery_layer_count": positive_layers,
        "gate": gate,
        "passed": bool(gate["passed"]),
    }


def _direct_float32_discrete_evaluation(
    arrays: Mapping[str, np.ndarray],
    output_projection: np.ndarray,
    codes: np.ndarray,
    inputs: JointQuadraticInputs,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    records, read_rows, layers = inputs.batch_shape
    expected_codes = (records, read_rows, layers, mass._QUERY_HEADS)
    selected = np.ascontiguousarray(codes, dtype=np.uint8).reshape(expected_codes)
    candidate = mass._counterfactual_pre_wo(
        arrays["base_attention_output"],
        arrays["regular_component"],
        arrays["episodic_component"],
        arrays["regular_mass"],
        arrays["episodic_mass"],
        selected,
    )
    correction = mass._project_counterfactual_delta(
        arrays["base_attention_output"],
        candidate,
        output_projection,
    )
    target = np.ascontiguousarray(arrays["target_residual"], dtype=np.float64)
    direct_error = (
        target - np.ascontiguousarray(correction, dtype=np.float64)
    ).reshape(-1, target.shape[-1])
    direct_error_energy = np.einsum(
        "ni,ni->n",
        direct_error,
        direct_error,
        optimize=True,
    )
    target_flat = target.reshape(-1, target.shape[-1])
    direct_target_energy = np.einsum(
        "ni,ni->n",
        target_flat,
        target_flat,
        optimize=True,
    )
    target_energy_difference = np.max(
        np.abs(direct_target_energy - inputs.target_energy)
    )
    target_energy_scale = max(1.0, float(np.max(inputs.target_energy)))
    if target_energy_difference > 2.0e-12 * target_energy_scale:
        raise ValueError("joint-gamma direct target energy changed")

    # Independently reconstruct the same selected correction through the q,d
    # factor path.  This is a diagnostic; all gates use the float32 path above.
    qd_delta = _reconstruct_pre_wo_delta(
        arrays,
        selected,
        query_heads=mass._QUERY_HEADS,
    )
    qd_projected = np.empty_like(qd_delta, dtype=np.float64)
    float64_weights = np.ascontiguousarray(output_projection, dtype=np.float64)
    for layer in range(layers):
        layer_delta = qd_delta[:, :, layer]
        qd_projected[:, :, layer] = (
            layer_delta.reshape(-1, layer_delta.shape[-1])
            @ float64_weights[layer].T
        ).reshape(layer_delta.shape)
    projected_difference = qd_projected - correction.astype(np.float64)
    pre_wo_difference = qd_delta - (
        candidate.astype(np.float64)
        - arrays["base_attention_output"].astype(np.float64)
    )
    tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * np.maximum(1.0, direct_target_energy)
    )
    excess = direct_error_energy - direct_target_energy
    nonregression = bool(np.all(excess <= tolerance))
    diagnostics = {
        "gate_path": (
            "established float32 finite counterfactual followed by established "
            "float32 candidate_delta @ o_proj.weight.T"
        ),
        "qd_path_is_diagnostic_only": True,
        "selected_pre_wo_qd_vs_float32_max_abs": float(
            np.max(np.abs(pre_wo_difference))
        ),
        "selected_projected_qd_vs_float32_max_abs": float(
            np.max(np.abs(projected_difference))
        ),
        "direct_objective_nonregression_tolerance_rule": (
            "64*float64_epsilon*max(1,base_target_energy)"
        ),
        "maximum_direct_objective_excess_over_base": float(np.max(excess)),
        "direct_objective_never_worse_than_base": nonregression,
    }
    metrics = _recovery_metrics_from_energy(
        direct_target_energy,
        direct_error_energy,
        batch_shape=inputs.batch_shape,
    )
    return metrics, diagnostics, direct_error_energy


def _screen_post_authentication(
    context: Mapping[str, Any],
) -> dict[str, bool]:
    checks = {
        "joint_gamma_protocol": (
            sha256_file(context["joint_gamma_protocol_path"])
            == context["joint_gamma_protocol_sha256"]
        ),
        "head_mass_protocol": (
            sha256_file(context["head_mass_protocol_path"])
            == context["head_mass_protocol_sha256"]
        ),
        "head_mass_result": (
            sha256_file(context["head_mass_result_path"])
            == context["head_mass_result_sha256"]
        ),
        "head_mass_manifest": (
            sha256_file(context["head_mass_manifest_path"])
            == context["head_mass_manifest_sha256"]
        ),
        "source_inventory": (
            context["joint_gamma_protocol"]["source_sha256"] == _source_inventory()
        ),
        "confirmation_not_opened": True,
    }
    return checks


def screen_cached_joint_gamma_oracle(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    out: str | Path,
    row_batch_size: int = 32,
) -> dict[str, Any]:
    """Run the frozen joint oracle entirely from authenticated cached tensors."""

    output = mass.capacity.bias.rank.retrieval._new_output(
        out,
        "joint-gamma result",
    )
    started = time.perf_counter()
    _progress("authenticating frozen protocol and inherited Q7 evidence")
    context, _training, frozen = _authenticate_protocol(protocol, protocol_sha256)
    _progress("loading and authenticating cached trace shards and BF16 Wo")
    arrays, output_projection = _load_cached_arrays_and_projection(context, frozen)
    _progress("constructing the float64 factor-defined joint objective")
    inputs = build_joint_quadratic_inputs(
        arrays,
        output_projection,
        row_batch_size=row_batch_size,
    )
    _progress("solving the continuous box relaxation")
    continuous = gamma_solver.solve_continuous_box(
        inputs.gram,
        inputs.linear,
        inputs.target_energy,
        inputs.lower,
        inputs.upper,
        maximum_sweeps=_CONTINUOUS_MAXIMUM_SWEEPS,
        kkt_relative_tolerance=_CONTINUOUS_RELATIVE_GAP_TOLERANCE,
    )
    _progress(
        "continuous relaxation "
        + (
            "met its objective-gap target"
            if continuous.converged
            else "reached its certified sweep budget"
        )
        + f" after {continuous.sweeps} sweeps; solving the discrete local oracle"
    )
    discrete = gamma_solver.solve_discrete_groups(
        inputs.gram,
        inputs.linear,
        inputs.target_energy,
        inputs.candidates,
        continuous.coefficients,
        base_code=_BASE_CODE,
        tie_priority=_TIE_PRIORITY,
    )
    if not (
        discrete.coordinate_converged
        and discrete.pair_converged
        and discrete.one_flip_locally_optimal
        and discrete.two_flip_locally_optimal
    ):
        raise ValueError("joint-gamma discrete local-optimality qualification failed")

    _progress("recomputing the selected arm through the direct float32 gate path")
    continuous_feasible_metrics = _recovery_metrics_from_energy(
        inputs.target_energy,
        continuous.objective,
        batch_shape=inputs.batch_shape,
    )
    continuous_optimistic_metrics = _recovery_metrics_from_energy(
        inputs.target_energy,
        np.maximum(
            continuous.objective - continuous.objective_gap_upper_bound,
            0.0,
        ),
        batch_shape=inputs.batch_shape,
    )
    quadratic_discrete_metrics = _recovery_metrics_from_energy(
        inputs.target_energy,
        discrete.objective,
        batch_shape=inputs.batch_shape,
    )
    direct_metrics, direct_diagnostics, direct_error_energy = (
        _direct_float32_discrete_evaluation(
            arrays,
            output_projection,
            discrete.codes,
            inputs,
        )
    )
    direct_diagnostics["quadratic_vs_direct_error_energy_max_abs"] = float(
        np.max(np.abs(discrete.objective - direct_error_energy))
    )
    direct_diagnostics["quadratic_vs_direct_error_energy_mean_abs"] = float(
        np.mean(np.abs(discrete.objective - direct_error_energy))
    )
    direct_diagnostics["quadratic_vs_direct_global_recovery_abs"] = abs(
        quadratic_discrete_metrics["global"]["recovery"]
        - direct_metrics["global"]["recovery"]
    )
    _progress("replaying both solvers and the direct gate path exactly")
    replay_continuous = gamma_solver.solve_continuous_box(
        inputs.gram,
        inputs.linear,
        inputs.target_energy,
        inputs.lower,
        inputs.upper,
        maximum_sweeps=_CONTINUOUS_MAXIMUM_SWEEPS,
        kkt_relative_tolerance=_CONTINUOUS_RELATIVE_GAP_TOLERANCE,
    )
    replay_discrete = gamma_solver.solve_discrete_groups(
        inputs.gram,
        inputs.linear,
        inputs.target_energy,
        inputs.candidates,
        replay_continuous.coefficients,
        base_code=_BASE_CODE,
        tie_priority=_TIE_PRIORITY,
    )
    (
        replay_direct_metrics,
        replay_direct_diagnostics,
        replay_direct_error_energy,
    ) = _direct_float32_discrete_evaluation(
        arrays,
        output_projection,
        replay_discrete.codes,
        inputs,
    )
    replay_direct_diagnostics["quadratic_vs_direct_error_energy_max_abs"] = float(
        np.max(np.abs(replay_discrete.objective - replay_direct_error_energy))
    )
    replay_direct_diagnostics["quadratic_vs_direct_error_energy_mean_abs"] = float(
        np.mean(np.abs(replay_discrete.objective - replay_direct_error_energy))
    )
    replay_direct_diagnostics["quadratic_vs_direct_global_recovery_abs"] = abs(
        _recovery_metrics_from_energy(
            inputs.target_energy,
            replay_discrete.objective,
            batch_shape=inputs.batch_shape,
        )["global"]["recovery"]
        - replay_direct_metrics["global"]["recovery"]
    )
    replay_checks = {
        "continuous_coefficients_exact": np.array_equal(
            continuous.coefficients,
            replay_continuous.coefficients,
        ),
        "continuous_objective_exact": np.array_equal(
            continuous.objective,
            replay_continuous.objective,
        ),
        "continuous_objective_gap_exact": np.array_equal(
            continuous.objective_gap_upper_bound,
            replay_continuous.objective_gap_upper_bound,
        ),
        "discrete_codes_exact": np.array_equal(
            discrete.codes,
            replay_discrete.codes,
        ),
        "discrete_coefficients_exact": np.array_equal(
            discrete.coefficients,
            replay_discrete.coefficients,
        ),
        "discrete_objective_exact": np.array_equal(
            discrete.objective,
            replay_discrete.objective,
        ),
        "direct_error_energy_exact": np.array_equal(
            direct_error_energy,
            replay_direct_error_energy,
        ),
        "direct_metrics_exact": direct_metrics == replay_direct_metrics,
        "direct_diagnostics_exact": (
            direct_diagnostics == replay_direct_diagnostics
        ),
    }
    replay_checks["passed"] = all(replay_checks.values())
    if not replay_checks["passed"]:
        raise ValueError("joint-gamma deterministic replay failed")
    direct_nonregression = direct_diagnostics[
        "direct_objective_never_worse_than_base"
    ]
    passed = bool(direct_metrics["passed"] and direct_nonregression)
    histogram = np.bincount(discrete.codes.reshape(-1), minlength=8)
    post = _screen_post_authentication(context)
    if not all(post.values()):
        raise ValueError("joint-gamma post-run authentication failed")
    report = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _RESULT_EXPERIMENT,
        "status": (
            "train_episodic_joint_gamma_oracle_gate_passed"
            if passed
            else "train_episodic_joint_gamma_oracle_gate_failed"
        ),
        "protocol": {
            "path": str(context["joint_gamma_protocol_path"]),
            "sha256": context["joint_gamma_protocol_sha256"],
        },
        "scope": {
            "split": "train",
            "cached_same_state_capacity_evidence_only": True,
            "native_teacher_rollout_executed": False,
            "causal_rollout": False,
            "development_outcomes_used": False,
            "confirmation_split_opened": False,
            "semantic_or_M3_gate_passed": False,
        },
        "quadratic_qualification": {
            "feature_names": list(inputs.feature_names),
            "batch_shape": list(inputs.batch_shape),
            "gram_shape": list(inputs.gram.shape),
            "gram_factor_construction": inputs.gram_factor_construction,
            "minimum_normalized_gram_eigenvalue": (
                inputs.minimum_normalized_gram_eigenvalue
            ),
            "maximum_gram_asymmetry": inputs.maximum_gram_asymmetry,
            "base_component_reconstruction_max_abs": (
                inputs.base_component_reconstruction_max_abs
            ),
            "all_codes_per_head_float32_counterfactual_pre_wo_max_abs": (
                inputs.float32_per_head_counterfactual_pre_wo_max_abs
            ),
            "uniform_all_head_code_float32_projected_max_abs": (
                inputs.float32_uniform_code_projected_max_abs
            ),
            "uniform_projected_check_is_not_mixed_assignment_bound": True,
            "psd_numerically_qualified": True,
            "passed": True,
        },
        "continuous_box_diagnostic": {
            "coefficients_sha256": hashlib.sha256(
                continuous.coefficients.tobytes(order="C")
            ).hexdigest(),
            "objective_sha256": hashlib.sha256(
                continuous.objective.tobytes(order="C")
            ).hexdigest(),
            "objective_gap_upper_bound_sha256": hashlib.sha256(
                continuous.objective_gap_upper_bound.tobytes(order="C")
            ).hexdigest(),
            "sweeps": continuous.sweeps,
            "maximum_sweeps": _CONTINUOUS_MAXIMUM_SWEEPS,
            "relative_objective_gap_target": (
                _CONTINUOUS_RELATIVE_GAP_TOLERANCE
            ),
            "objective_gap_target_met": continuous.converged,
            "objective_gap_certificate_available": True,
            "maximum_normalized_kkt_violation": (
                continuous.maximum_normalized_kkt_violation
            ),
            "maximum_relative_objective_gap": (
                continuous.maximum_relative_objective_gap
            ),
            "maximum_objective_gap_upper_bound": float(
                np.max(continuous.objective_gap_upper_bound)
            ),
            "summed_objective_gap_upper_bound": float(
                np.sum(
                    continuous.objective_gap_upper_bound,
                    dtype=np.float64,
                )
            ),
            "feasible_solution_metrics": continuous_feasible_metrics,
            "optimistic_recovery_upper_bound_metrics": (
                continuous_optimistic_metrics
            ),
            "progression_authority": False,
        },
        "discrete_local_oracle": {
            "codes_sha256": hashlib.sha256(
                discrete.codes.tobytes(order="C")
            ).hexdigest(),
            "coefficients_sha256": hashlib.sha256(
                discrete.coefficients.tobytes(order="C")
            ).hexdigest(),
            "quadratic_objective_sha256": hashlib.sha256(
                discrete.objective.tobytes(order="C")
            ).hexdigest(),
            "selected_code_histogram": {
                str(code): int(count) for code, count in enumerate(histogram)
            },
            "start_count": discrete.start_count,
            "maximum_coordinate_sweeps": discrete.maximum_coordinate_sweeps,
            "pair_sweeps": discrete.pair_sweeps,
            "coordinate_converged": discrete.coordinate_converged,
            "pair_converged": discrete.pair_converged,
            "one_flip_locally_optimal": discrete.one_flip_locally_optimal,
            "two_flip_locally_optimal": discrete.two_flip_locally_optimal,
            "optimality_scope": (
                "one/two-head-flip local optimum only; not an exact or "
                "global optimum over 8^16"
            ),
            "quadratic_metrics_diagnostic": quadratic_discrete_metrics,
            "direct_float32_metrics": direct_metrics,
            "direct_recomputation": direct_diagnostics,
            "passed": passed,
        },
        "deterministic_replay": {
            "checks": replay_checks,
            "passed": True,
        },
        "resource_contract": dict(frozen["resource_contract"]),
        "decision": {
            "train_joint_gamma_capacity_gate_passed": passed,
            "semantic_or_M3_gate_passed": False,
            "native_causal_integration_authorized": False,
            "development_authorized": False,
            "confirmation_authorized": False,
            "train_only_causal_predictor_experiment_authorized": passed,
            "continuous_relaxation_optimistic_gate_passed": (
                continuous_optimistic_metrics["passed"]
            ),
            "next_step": (
                "freeze a train-only causal 3-bit gamma predictor with "
                "rolled-out native state and full traffic accounting"
                if passed
                else (
                    (
                        "close only the cached same-state bounded affine q,d "
                        "relaxation; its optimistic recovery upper bound "
                        "failed the frozen gate"
                    )
                    if not continuous_optimistic_metrics["passed"]
                    else (
                        "close only this deterministic multistart one/two-"
                        "flip solution and do not claim global-grid failure"
                    )
                )
            ),
        },
        "post_run_authentication": post,
        "confirmation_split_opened": False,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output, report)
    _progress(f"joint-gamma result written to {output}")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cached joint output-targeted gamma oracle",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--head-mass-protocol", required=True)
    freeze.add_argument("--head-mass-protocol-sha256", required=True)
    freeze.add_argument("--head-mass-result", required=True)
    freeze.add_argument("--head-mass-result-sha256", required=True)
    freeze.add_argument("--out", required=True)
    screen = commands.add_parser("screen")
    screen.add_argument("--protocol", required=True)
    screen.add_argument("--protocol-sha256", required=True)
    screen.add_argument("--out", required=True)
    screen.add_argument("--row-batch-size", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "freeze":
        value = freeze_joint_gamma_protocol(
            head_mass_protocol=args.head_mass_protocol,
            head_mass_protocol_sha256=args.head_mass_protocol_sha256,
            head_mass_result=args.head_mass_result,
            head_mass_result_sha256=args.head_mass_result_sha256,
            out=args.out,
        )
    elif args.command == "screen":
        value = screen_cached_joint_gamma_oracle(
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            out=args.out,
            row_batch_size=args.row_batch_size,
        )
    else:  # pragma: no cover - argparse enforces commands
        raise AssertionError(f"unknown command: {args.command}")
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
