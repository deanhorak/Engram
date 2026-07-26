"""Selected-record dynamic input pruning for native BitNet MLPs.

This module is an inference-quality prototype, not a deployment kernel.  It
implements the exact data flow that a coordinate-major native kernel can later
use:

1. quantize the input with the teacher's per-row Q8 rule;
2. evaluate gate/up proxies for every record from the largest input
   coordinates;
3. choose a stable candidate set;
4. complete gate/up exactly only for those candidates;
5. correct the proxy estimate of the full intermediate RMS with the exact
   candidate activations;
6. rerank candidates with their Q8 coefficient and down-column norm; and
7. read and accumulate down records only for the selected top-K.

The intermediate RMS denominator cancels from the *integer* Q8 coefficient
codes because it is a positive row-wide scalar.  It does not cancel from the
dequantization amplitude.  The implementation exploits the former and
explicitly estimates the latter; it never claims that RMS normalization can be
discarded entirely.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.models.native_bitnet import (
    LoadedNativeBitNetArtifact,
    decode_native_bitnet_layer,
)
from engram.semantic.dip import input_coordinate_count, stable_top_k


@dataclass(frozen=True)
class NativeBitNetDIPDiagnostics:
    """Dense-oracle diagnostics, computed only when explicitly requested."""

    candidate_recall: NDArray[np.float64]
    output_relative_l2: NDArray[np.float64]
    output_cosine: NDArray[np.float64]
    routed_vs_oracle_output_relative_l2: NDArray[np.float64]
    routed_vs_oracle_output_cosine: NDArray[np.float64]
    candidate_coefficient_relative_l2: NDArray[np.float64]
    rms_variance_relative_error: NDArray[np.float64]
    normalized_absmax_relative_error: NDArray[np.float64]
    cancellation_code_mismatches: NDArray[np.int64]
    oracle_indices: NDArray[np.int64]
    dense_output: NDArray[np.float32]


@dataclass(frozen=True)
class NativeBitNetDIPResult:
    """One batched selected-record BitNet MLP result."""

    output: NDArray[np.float32]
    input_indices: NDArray[np.int64]
    candidate_indices: NDArray[np.int64]
    selected_indices: NDArray[np.int64]
    selected_counts: NDArray[np.int64]
    selected_coefficients: NDArray[np.float32]
    estimated_raw_variance: NDArray[np.float32]
    estimated_normalized_absmax: NDArray[np.float32]
    rms_q8_cancellation_applied: NDArray[np.bool_]
    diagnostics: NativeBitNetDIPDiagnostics | None


@dataclass(frozen=True)
class NativeBitNetDIPConfiguration:
    """Per-layer causal DIP budget."""

    input_fraction: float
    candidate_count: int
    top_k: int
    rms_audit_count: int = 0
    energy_target: float | None = None
    minimum_top_k: int | None = None
    maximum_top_k: int | None = None
    rms_variance_scale: float = 1.0
    rms_variance_bias: float = 0.0
    output_scale: float = 1.0
    rms_estimator: str = "corrected_proxy"
    rms_audit_strategy: str = "hashed_tail"


def _finite_matrix(
    values: ArrayLike,
    *,
    width: int,
) -> tuple[NDArray[np.float32], tuple[int, ...]]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim < 1 or array.shape[-1] != width:
        raise ValueError(f"hidden must end in dimension {width}")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("hidden must be non-empty and finite")
    return array.reshape(-1, width), array.shape


def _activation_q8_codes(
    values: NDArray[np.float32],
    *,
    absolute_max: float | None = None,
) -> tuple[NDArray[np.int8], np.float32]:
    """Return teacher-compatible Q8 codes and their dequantization scale."""

    source = np.asarray(values, dtype=np.float32)
    maximum = (
        float(np.max(np.abs(source)))
        if absolute_max is None
        else float(absolute_max)
    )
    if not np.isfinite(maximum) or maximum < 0:
        raise ValueError("absolute_max must be finite and non-negative")
    effective_maximum = np.float32(max(maximum, 1e-5))
    scale = np.float32(127.0) / effective_maximum
    codes = (
        np.rint(source * scale)
        .clip(np.float32(-128.0), np.float32(127.0))
        .astype(np.int8)
    )
    return codes, np.float32(effective_maximum / np.float32(127.0))


def _bf16_round(values: ArrayLike) -> NDArray[np.float32]:
    """Round float32 values to BF16 with the native kernel's ties-to-even rule."""

    source = np.asarray(values, dtype=np.float32)
    bits = source.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _activation_quant_bf16(
    values: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Apply the native kernel's row-wise Q8 quantization and BF16 store."""

    source = np.asarray(values, dtype=np.float32)
    maximum = np.max(np.abs(source), axis=-1, keepdims=True)
    scale = np.float32(127.0) / np.maximum(maximum, np.float32(1e-5))
    codes = (
        np.rint(source * scale)
        .clip(np.float32(-128.0), np.float32(127.0))
        .astype(np.float32)
    )
    return _bf16_round(codes / scale)


def _native_projection_activation(
    codes: NDArray[np.int8],
    state: NDArray[np.float32],
    scale: np.float32,
) -> NDArray[np.float32]:
    accumulator = np.asarray(codes @ state, dtype=np.float32)
    return _bf16_round(_bf16_round(accumulator) * scale)


def _native_raw_activation(
    gate: NDArray[np.float32],
    up: NDArray[np.float32],
) -> NDArray[np.float32]:
    positive = np.maximum(gate, np.float32(0.0))
    squared = _bf16_round(positive * positive)
    return _bf16_round(squared * up)


def _native_normalized(
    raw: NDArray[np.float32],
    gain: NDArray[np.float32],
    inverse_rms: np.float32,
) -> NDArray[np.float32]:
    return _bf16_round(_bf16_round(raw * inverse_rms) * gain)


def _activation_quant_bf16_with_maximum(
    values: NDArray[np.float32],
    *,
    absolute_max: float,
) -> tuple[NDArray[np.float32], NDArray[np.int8]]:
    maximum = np.float32(max(float(absolute_max), 1e-5))
    scale = np.float32(127.0) / maximum
    codes = (
        np.rint(np.asarray(values, dtype=np.float32) * scale)
        .clip(np.float32(-128.0), np.float32(127.0))
        .astype(np.int8)
    )
    return _bf16_round(codes.astype(np.float32) / scale), codes


def _rms_q8_candidate_coefficients(
    raw_candidates: NDArray[np.float32],
    candidate_gain: NDArray[np.float32],
    *,
    raw_variance: float,
    corrected_raw: NDArray[np.float32],
    corrected_gain: NDArray[np.float32],
    epsilon: float,
) -> tuple[NDArray[np.float32], np.float32, bool, NDArray[np.int8]]:
    """Quantize candidate coefficients while cancelling RMS from Q8 codes.

    For raw activation ``a``, gain ``g``, and row RMS denominator ``d``, the
    teacher quantizes ``z = a*g/d``.  If ``max(abs(z))`` is above the Q8 floor,

    ``round(127*z/max(abs(z))) == round(127*a*g/max(abs(a*g)))``.

    The integer codes therefore do not require ``d``.  The dequantization
    amplitude still does and is returned through the final coefficients.
    """

    variance = np.float32(max(float(raw_variance), 0.0))
    inverse_rms = np.float32(
        1.0 / np.sqrt(variance + np.float32(epsilon))
    )
    all_normalized = _native_normalized(
        np.asarray(corrected_raw, dtype=np.float32),
        np.asarray(corrected_gain, dtype=np.float32),
        inverse_rms,
    )
    normalized_absmax = np.float32(np.max(np.abs(all_normalized)))
    candidate_normalized = _native_normalized(
        np.asarray(raw_candidates, dtype=np.float32),
        np.asarray(candidate_gain, dtype=np.float32),
        inverse_rms,
    )
    coefficients, direct_codes = _activation_quant_bf16_with_maximum(
        candidate_normalized,
        absolute_max=float(normalized_absmax),
    )

    # Validate the mathematical cancellation against the teacher's BF16
    # boundaries.  BF16 rounding between RMS scaling and gain multiplication
    # can change a few codes, so cancellation is used only when it is exact.
    weighted_all = (
        np.asarray(corrected_raw, dtype=np.float32)
        * np.asarray(corrected_gain, dtype=np.float32)
    )
    weighted_maximum = np.float32(np.max(np.abs(weighted_all)))
    cancellation_applies = bool(
        normalized_absmax >= np.float32(1e-5)
        and weighted_maximum > np.float32(0.0)
    )
    if cancellation_applies:
        cancellation_scale = np.float32(127.0) / weighted_maximum
        cancelled_codes = (
            np.rint(
                np.asarray(raw_candidates, dtype=np.float32)
                * np.asarray(candidate_gain, dtype=np.float32)
                * cancellation_scale
            )
            .clip(np.float32(-128.0), np.float32(127.0))
            .astype(np.int8)
        )
    else:
        cancelled_codes = direct_codes
    cancellation_applies = bool(
        cancellation_applies
        and np.array_equal(cancelled_codes, direct_codes)
    )
    return (
        coefficients,
        normalized_absmax,
        cancellation_applies,
        cancelled_codes,
    )


def _row_similarity(
    approximation: NDArray[np.float32],
    reference: NDArray[np.float32],
) -> tuple[float, float]:
    actual = np.asarray(approximation, dtype=np.float64)
    exact = np.asarray(reference, dtype=np.float64)
    exact_norm = float(np.linalg.norm(exact))
    actual_norm = float(np.linalg.norm(actual))
    error = float(np.linalg.norm(actual - exact))
    relative = error / max(exact_norm, 1e-12)
    if exact_norm <= 1e-12 and actual_norm <= 1e-12:
        cosine = 1.0
    elif exact_norm <= 1e-12 or actual_norm <= 1e-12:
        cosine = 0.0
    else:
        cosine = float(np.dot(actual, exact) / (actual_norm * exact_norm))
    return relative, float(np.clip(cosine, -1.0, 1.0))


class NativeBitNetDIPLayer:
    """Decoded selected-record BitNet MLP suitable for causal evaluation."""

    def __init__(
        self,
        artifact: LoadedNativeBitNetArtifact,
        layer: int,
        *,
        input_fraction: float,
        candidate_count: int,
        top_k: int,
        rms_audit_count: int = 0,
        energy_target: float | None = None,
        minimum_top_k: int | None = None,
        maximum_top_k: int | None = None,
        rms_variance_scale: float = 1.0,
        rms_variance_bias: float = 0.0,
        output_scale: float = 1.0,
        rms_estimator: str = "corrected_proxy",
        rms_audit_strategy: str = "hashed_tail",
    ) -> None:
        if (
            isinstance(candidate_count, bool)
            or not isinstance(candidate_count, int)
            or candidate_count <= 0
        ):
            raise ValueError("candidate_count must be a positive integer")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if candidate_count > artifact.intermediate_size:
            raise ValueError("candidate_count exceeds the intermediate size")
        if top_k > candidate_count:
            raise ValueError("top_k exceeds candidate_count")
        if (
            isinstance(rms_audit_count, bool)
            or not isinstance(rms_audit_count, int)
            or rms_audit_count < 0
        ):
            raise ValueError("rms_audit_count must be a non-negative integer")
        if candidate_count - rms_audit_count < top_k:
            raise ValueError(
                "candidate_count minus rms_audit_count must cover top_k"
            )
        if energy_target is not None:
            if (
                isinstance(energy_target, bool)
                or not isinstance(energy_target, (int, float, np.number))
                or not np.isfinite(float(energy_target))
                or not 0 < float(energy_target) <= 1
            ):
                raise ValueError("energy_target must lie in (0, 1]")
            adaptive_minimum = (
                1 if minimum_top_k is None else minimum_top_k
            )
            adaptive_maximum = (
                top_k if maximum_top_k is None else maximum_top_k
            )
            for value, name in (
                (adaptive_minimum, "minimum_top_k"),
                (adaptive_maximum, "maximum_top_k"),
            ):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                ):
                    raise ValueError(f"{name} must be a positive integer")
            if adaptive_minimum > adaptive_maximum:
                raise ValueError("minimum_top_k exceeds maximum_top_k")
            if adaptive_maximum > candidate_count - rms_audit_count:
                raise ValueError(
                    "maximum_top_k exceeds routed candidate_count"
                )
        elif minimum_top_k is not None or maximum_top_k is not None:
            raise ValueError(
                "minimum/maximum_top_k require energy_target"
            )
        for value, name in (
            (rms_variance_scale, "rms_variance_scale"),
            (output_scale, "output_scale"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.number))
                or not np.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(rms_variance_bias, bool)
            or not isinstance(rms_variance_bias, (int, float, np.number))
            or not np.isfinite(float(rms_variance_bias))
        ):
            raise ValueError("rms_variance_bias must be finite")
        if rms_estimator not in {"corrected_proxy", "candidate_ratio"}:
            raise ValueError(
                "rms_estimator must be corrected_proxy or candidate_ratio"
            )
        if rms_estimator == "candidate_ratio" and rms_audit_count:
            raise ValueError(
                "candidate_ratio RMS estimation does not use rms_audit_count"
            )
        if rms_audit_strategy not in {"hashed_tail", "top_proxy_raw_square"}:
            raise ValueError(
                "rms_audit_strategy must be hashed_tail or top_proxy_raw_square"
            )

        self.artifact = artifact
        self.layer = int(layer)
        self.input_fraction = float(input_fraction)
        self.input_count = input_coordinate_count(
            artifact.hidden_size,
            input_fraction,
        )
        self.candidate_count = candidate_count
        self.top_k = top_k
        self.rms_audit_count = rms_audit_count
        self.energy_target = (
            None if energy_target is None else float(energy_target)
        )
        self.minimum_top_k = (
            top_k
            if self.energy_target is None
            else (1 if minimum_top_k is None else minimum_top_k)
        )
        self.maximum_top_k = (
            top_k
            if self.energy_target is None
            else (top_k if maximum_top_k is None else maximum_top_k)
        )
        self.rms_variance_scale = np.float32(rms_variance_scale)
        self.rms_variance_bias = np.float32(rms_variance_bias)
        self.output_scale = np.float32(output_scale)
        self.rms_estimator = rms_estimator
        self.rms_audit_strategy = rms_audit_strategy
        decoded = decode_native_bitnet_layer(artifact, self.layer)
        # Preserve ternary storage in the causal prototype.  NumPy promotes
        # int8-by-float32 products to float32, so materializing all 30 trained
        # layers costs one byte per trit rather than four.
        self.gate_codes = np.asarray(decoded["gate_codes"], dtype=np.int8)
        self.up_codes = np.asarray(decoded["up_codes"], dtype=np.int8)
        self.down_codes = np.asarray(decoded["down_codes"], dtype=np.int8)
        self.gate_scale = np.float32(decoded["gate_scale"])
        self.up_scale = np.float32(decoded["up_scale"])
        self.down_scale = np.float32(decoded["down_scale"])
        self.gain = np.asarray(decoded["ffn_sub_norm"], dtype=np.float32)
        self.down_norm_squared = np.count_nonzero(
            self.down_codes,
            axis=0,
        ).astype(np.float32)

    def __call__(
        self,
        hidden: ArrayLike,
        *,
        oracle_diagnostics: bool = False,
    ) -> NativeBitNetDIPResult:
        rows, original_shape = _finite_matrix(
            hidden,
            width=self.artifact.hidden_size,
        )
        quantized = _activation_quant_bf16(rows)
        row_count = rows.shape[0]
        width = self.artifact.intermediate_size

        input_indices = np.empty(
            (row_count, self.input_count),
            dtype=np.int64,
        )
        candidate_indices = np.empty(
            (row_count, self.candidate_count),
            dtype=np.int64,
        )
        selected_indices = np.full(
            (row_count, self.maximum_top_k),
            -1,
            dtype=np.int64,
        )
        selected_counts = np.empty(row_count, dtype=np.int64)
        selected_coefficients = np.empty(
            (row_count, self.maximum_top_k),
            dtype=np.float32,
        )
        selected_coefficients.fill(np.float32(0.0))
        output = np.empty(
            (row_count, self.artifact.hidden_size),
            dtype=np.float32,
        )
        estimated_variance = np.empty(row_count, dtype=np.float32)
        estimated_absmax = np.empty(row_count, dtype=np.float32)
        cancellation_applied = np.empty(row_count, dtype=np.bool_)

        diagnostic_recall: list[float] = []
        diagnostic_relative: list[float] = []
        diagnostic_cosine: list[float] = []
        diagnostic_oracle_relative: list[float] = []
        diagnostic_oracle_cosine: list[float] = []
        diagnostic_coefficient_relative: list[float] = []
        diagnostic_variance_error: list[float] = []
        diagnostic_absmax_error: list[float] = []
        diagnostic_code_mismatches: list[int] = []
        diagnostic_oracle_indices: list[NDArray[np.int64]] = []
        diagnostic_dense_outputs: list[NDArray[np.float32]] = []

        for row_index, state in enumerate(quantized):
            coordinates = stable_top_k(np.abs(state), self.input_count)
            input_indices[row_index] = coordinates

            partial_gate = _native_projection_activation(
                self.gate_codes[:, coordinates],
                state[coordinates],
                self.gate_scale,
            )
            partial_up = _native_projection_activation(
                self.up_codes[:, coordinates],
                state[coordinates],
                self.up_scale,
            )
            proxy_raw = _native_raw_activation(partial_gate, partial_up)
            proxy_utility = (
                proxy_raw
                * proxy_raw
                * self.gain
                * self.gain
                * self.down_norm_squared
            )
            proxy_order = stable_top_k(proxy_utility, width)
            routed_count = self.candidate_count - self.rms_audit_count
            routed = proxy_order[:routed_count]
            if self.rms_audit_count:
                tail = proxy_order[routed_count:]
                if self.rms_audit_strategy == "top_proxy_raw_square":
                    # Utility also includes gain and down-column norm. RMS
                    # depends only on raw activation, so reserve completion
                    # slots for records that utility routing can overlook.
                    tail_by_record = np.argsort(tail, kind="stable")
                    audit_local_by_record = stable_top_k(
                        proxy_raw[tail[tail_by_record]] ** 2,
                        self.rms_audit_count,
                    )
                    audit = tail[
                        tail_by_record[audit_local_by_record]
                    ]
                else:
                    # A state-dependent multiplicative hash gives a
                    # reproducible, approximately uniform sample of the proxy
                    # tail without carrying RNG state through generation.
                    coordinate_seed = int(
                        np.sum(
                            (coordinates[: min(16, coordinates.size)] + 1)
                            * np.arange(
                                1,
                                min(16, coordinates.size) + 1,
                                dtype=np.int64,
                            ),
                            dtype=np.int64,
                        )
                    )
                    tail_u64 = tail.astype(np.uint64)
                    hashes = (
                        tail_u64 * np.uint64(0x9E3779B185EBCA87)
                        + np.uint64(
                            (
                                coordinate_seed
                                + 0xD1B54A32D192ED03 * (self.layer + 1)
                            )
                            & ((1 << 64) - 1)
                        )
                    )
                    audit_local = np.argsort(
                        hashes,
                        kind="stable",
                    )[: self.rms_audit_count]
                    audit = tail[audit_local]
                candidates = np.concatenate((routed, audit))
            else:
                tail = proxy_order[routed_count:]
                audit = np.empty(0, dtype=np.int64)
                candidates = routed
            candidate_indices[row_index] = candidates

            exact_gate = _native_projection_activation(
                self.gate_codes[candidates],
                state,
                self.gate_scale,
            )
            exact_up = _native_projection_activation(
                self.up_codes[candidates],
                state,
                self.up_scale,
            )
            exact_raw = _native_raw_activation(exact_gate, exact_up)

            # Correct the cheap all-record proxy statistics wherever exact
            # candidate activations are available.  This estimates the one
            # row-wide amplitude that does not cancel from Q8 dequantization.
            proxy_square_sum = np.sum(
                proxy_raw * proxy_raw,
                dtype=np.float64,
            )
            candidate_proxy = proxy_raw[candidates]
            exact_candidate_square_sum = np.sum(
                exact_raw.astype(np.float64) ** 2,
                dtype=np.float64,
            )
            proxy_candidate_square_sum = np.sum(
                candidate_proxy.astype(np.float64) ** 2,
                dtype=np.float64,
            )
            if self.rms_estimator == "candidate_ratio":
                if proxy_candidate_square_sum <= 1e-30:
                    tail_scale = 1.0
                else:
                    tail_scale = (
                        exact_candidate_square_sum
                        / proxy_candidate_square_sum
                    )
                proxy_tail_square_sum = max(
                    proxy_square_sum - proxy_candidate_square_sum,
                    0.0,
                )
                corrected_square_sum = (
                    exact_candidate_square_sum
                    + tail_scale * proxy_tail_square_sum
                )
            else:
                candidate_delta = (
                    exact_raw.astype(np.float64) ** 2
                    - candidate_proxy.astype(np.float64) ** 2
                )
                routed_delta = candidate_delta[:routed_count]
                corrected_square_sum = proxy_square_sum + np.sum(
                    routed_delta,
                    dtype=np.float64,
                )
                if self.rms_audit_count:
                    audit_delta = candidate_delta[routed_count:]
                    if self.rms_audit_strategy == "hashed_tail":
                        tail_population = width - routed_count
                        corrected_square_sum += (
                            tail_population
                            / self.rms_audit_count
                            * np.sum(audit_delta, dtype=np.float64)
                        )
                    else:
                        corrected_square_sum += np.sum(
                            audit_delta,
                            dtype=np.float64,
                        )
            variance = np.float32(
                max(
                    float(self.rms_variance_scale)
                    * corrected_square_sum
                    / width
                    + float(self.rms_variance_bias),
                    0.0,
                )
            )
            estimated_variance[row_index] = variance

            corrected_raw = proxy_raw.copy()
            corrected_raw[candidates] = exact_raw
            candidate_coefficients, normalized_absmax, cancelled, _ = (
                _rms_q8_candidate_coefficients(
                    exact_raw,
                    self.gain[candidates],
                    raw_variance=float(variance),
                    corrected_raw=corrected_raw,
                    corrected_gain=self.gain,
                    epsilon=self.artifact.rms_norm_eps,
                )
            )
            estimated_absmax[row_index] = normalized_absmax
            cancellation_applied[row_index] = cancelled
            exact_candidate_utility = (
                candidate_coefficients
                * candidate_coefficients
                * self.down_norm_squared[candidates]
            )

            # Candidate proxy order must not leak into exact-score ties.
            by_record = np.argsort(candidates, kind="stable")
            exact_order = stable_top_k(
                exact_candidate_utility[by_record],
                self.candidate_count,
            )
            exact_order = by_record[exact_order]
            if self.energy_target is None:
                selected_count = self.top_k
            else:
                ordered_utility = exact_candidate_utility[exact_order]
                total_utility = float(
                    np.sum(ordered_utility, dtype=np.float64)
                )
                if total_utility <= 0.0:
                    target_count = 0
                elif self.energy_target >= 1.0:
                    target_count = int(
                        np.count_nonzero(ordered_utility > 0.0)
                    )
                else:
                    cumulative = np.cumsum(
                        ordered_utility,
                        dtype=np.float64,
                    )
                    target_count = int(
                        np.searchsorted(
                            cumulative,
                            self.energy_target * total_utility,
                            side="left",
                        )
                        + 1
                    )
                selected_count = min(
                    self.maximum_top_k,
                    max(self.minimum_top_k, target_count),
                )
            selected_local = exact_order[:selected_count]
            selected = candidates[selected_local]
            coefficients = candidate_coefficients[selected_local]
            selected_counts[row_index] = selected_count
            selected_indices[row_index, :selected_count] = selected
            selected_coefficients[row_index, :selected_count] = coefficients
            down_accumulator = np.asarray(
                self.down_codes[:, selected] @ coefficients,
                dtype=np.float32,
            )
            output[row_index] = _bf16_round(
                _bf16_round(down_accumulator) * self.down_scale
            )
            if self.output_scale != np.float32(1.0):
                output[row_index] = _bf16_round(
                    output[row_index] * self.output_scale
                )

            if oracle_diagnostics:
                dense_gate = _native_projection_activation(
                    self.gate_codes,
                    state,
                    self.gate_scale,
                )
                dense_up = _native_projection_activation(
                    self.up_codes,
                    state,
                    self.up_scale,
                )
                dense_raw = _native_raw_activation(dense_gate, dense_up)
                dense_variance = np.float32(
                    np.mean(dense_raw * dense_raw, dtype=np.float32)
                )
                dense_inverse = np.float32(
                    1.0
                    / np.sqrt(
                        dense_variance
                        + np.float32(self.artifact.rms_norm_eps)
                    )
                )
                dense_normalized = _native_normalized(
                    dense_raw,
                    self.gain,
                    dense_inverse,
                )
                dense_coefficients = _activation_quant_bf16(
                    dense_normalized[None, :]
                )[0]
                dense_utility = (
                    dense_coefficients
                    * dense_coefficients
                    * self.down_norm_squared
                )
                oracle = stable_top_k(dense_utility, selected_count)
                hits = np.intersect1d(
                    candidates,
                    oracle,
                    assume_unique=True,
                ).size
                diagnostic_recall.append(hits / selected_count)
                diagnostic_oracle_indices.append(oracle)
                exact_output = _bf16_round(
                    _bf16_round(
                        np.asarray(
                            self.down_codes @ dense_coefficients,
                            dtype=np.float32,
                        )
                    )
                    * self.down_scale
                )
                diagnostic_dense_outputs.append(exact_output)
                relative, cosine = _row_similarity(
                    output[row_index],
                    exact_output,
                )
                diagnostic_relative.append(relative)
                diagnostic_cosine.append(cosine)
                oracle_output = _bf16_round(
                    _bf16_round(
                        np.asarray(
                            self.down_codes[:, oracle]
                            @ dense_coefficients[oracle],
                            dtype=np.float32,
                        )
                    )
                    * self.down_scale
                )
                oracle_relative, oracle_cosine = _row_similarity(
                    output[row_index],
                    oracle_output,
                )
                diagnostic_oracle_relative.append(oracle_relative)
                diagnostic_oracle_cosine.append(oracle_cosine)
                coefficient_denominator = float(
                    np.linalg.norm(
                        dense_coefficients[candidates].astype(np.float64)
                    )
                )
                diagnostic_coefficient_relative.append(
                    float(
                        np.linalg.norm(
                            candidate_coefficients.astype(np.float64)
                            - dense_coefficients[candidates].astype(np.float64)
                        )
                    )
                    / max(coefficient_denominator, 1e-12)
                )
                diagnostic_variance_error.append(
                    abs(float(variance) - float(dense_variance))
                    / max(float(dense_variance), 1e-20)
                )
                dense_absmax = float(np.max(np.abs(dense_normalized)))
                diagnostic_absmax_error.append(
                    abs(float(normalized_absmax) - dense_absmax)
                    / max(dense_absmax, 1e-20)
                )
                _, direct_codes = _activation_quant_bf16_with_maximum(
                    dense_normalized,
                    absolute_max=dense_absmax,
                )
                cancelled_coefficients, _, _, cancelled_codes = (
                    _rms_q8_candidate_coefficients(
                        dense_raw,
                        self.gain,
                        raw_variance=float(dense_variance),
                        corrected_raw=dense_raw,
                        corrected_gain=self.gain,
                        epsilon=self.artifact.rms_norm_eps,
                    )
                )
                # Exercise both values so an accidental coefficient-scale
                # regression cannot hide behind equal integer codes.
                if not np.all(np.isfinite(cancelled_coefficients)):
                    raise RuntimeError(
                        "RMS-to-Q8 cancellation produced non-finite values"
                    )
                diagnostic_code_mismatches.append(
                    int(np.count_nonzero(direct_codes != cancelled_codes))
                )

        diagnostics: NativeBitNetDIPDiagnostics | None = None
        if oracle_diagnostics:
            dense_output = np.stack(diagnostic_dense_outputs)
            diagnostics = NativeBitNetDIPDiagnostics(
                candidate_recall=np.asarray(
                    diagnostic_recall,
                    dtype=np.float64,
                ),
                output_relative_l2=np.asarray(
                    diagnostic_relative,
                    dtype=np.float64,
                ),
                output_cosine=np.asarray(
                    diagnostic_cosine,
                    dtype=np.float64,
                ),
                routed_vs_oracle_output_relative_l2=np.asarray(
                    diagnostic_oracle_relative,
                    dtype=np.float64,
                ),
                routed_vs_oracle_output_cosine=np.asarray(
                    diagnostic_oracle_cosine,
                    dtype=np.float64,
                ),
                candidate_coefficient_relative_l2=np.asarray(
                    diagnostic_coefficient_relative,
                    dtype=np.float64,
                ),
                rms_variance_relative_error=np.asarray(
                    diagnostic_variance_error,
                    dtype=np.float64,
                ),
                normalized_absmax_relative_error=np.asarray(
                    diagnostic_absmax_error,
                    dtype=np.float64,
                ),
                cancellation_code_mismatches=np.asarray(
                    diagnostic_code_mismatches,
                    dtype=np.int64,
                ),
                oracle_indices=np.stack(diagnostic_oracle_indices),
                dense_output=np.asarray(dense_output, dtype=np.float32).reshape(
                    original_shape
                ),
            )

        return NativeBitNetDIPResult(
            output=output.reshape(original_shape),
            input_indices=input_indices,
            candidate_indices=candidate_indices,
            selected_indices=selected_indices,
            selected_counts=selected_counts,
            selected_coefficients=selected_coefficients,
            estimated_raw_variance=estimated_variance,
            estimated_normalized_absmax=estimated_absmax,
            rms_q8_cancellation_applied=cancellation_applied,
            diagnostics=diagnostics,
        )


def build_native_bitnet_dip_mlp(
    artifact: LoadedNativeBitNetArtifact,
    layer: int,
    *,
    input_fraction: float,
    candidate_count: int,
    top_k: int,
    rms_audit_count: int = 0,
    energy_target: float | None = None,
    minimum_top_k: int | None = None,
    maximum_top_k: int | None = None,
    rms_variance_scale: float = 1.0,
    rms_variance_bias: float = 0.0,
    output_scale: float = 1.0,
    rms_estimator: str = "corrected_proxy",
    rms_audit_strategy: str = "hashed_tail",
    oracle_diagnostics: bool = False,
) -> Any:
    """Build an inference-only ``torch.nn.Module`` for causal substitution.

    The wrapper deliberately delegates to the NumPy reference implementation.
    It is intended to establish causal behavior against a trained package
    before the same algorithm is implemented in the memory-mapped CPU kernel.
    It performs no autograd and returns results on the input device/dtype.
    """

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("causal DIP substitution requires torch") from exc

    engine = NativeBitNetDIPLayer(
        artifact,
        layer,
        input_fraction=input_fraction,
        candidate_count=candidate_count,
        top_k=top_k,
        rms_audit_count=rms_audit_count,
        energy_target=energy_target,
        minimum_top_k=minimum_top_k,
        maximum_top_k=maximum_top_k,
        rms_variance_scale=rms_variance_scale,
        rms_variance_bias=rms_variance_bias,
        output_scale=output_scale,
        rms_estimator=rms_estimator,
        rms_audit_strategy=rms_audit_strategy,
    )

    class _CausalNativeBitNetDIPMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.engine = engine
            self.last_result: NativeBitNetDIPResult | None = None

        def forward(self, hidden_states):
            if hidden_states.requires_grad:
                raise RuntimeError(
                    "native BitNet DIP causal prototype is inference-only"
                )
            states = (
                hidden_states.detach()
                .float()
                .cpu()
                .numpy()
            )
            self.last_result = self.engine(
                states,
                oracle_diagnostics=oracle_diagnostics,
            )
            return torch.from_numpy(self.last_result.output).to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

    return _CausalNativeBitNetDIPMLP()


@contextmanager
def substitute_native_bitnet_dip_mlps(
    model: Any,
    artifact: LoadedNativeBitNetArtifact,
    configurations: Mapping[int, NativeBitNetDIPConfiguration],
    *,
    oracle_diagnostics: bool = False,
) -> Iterator[dict[int, Any]]:
    """Temporarily replace trained decoder MLPs with causal DIP prototypes.

    ``model`` follows the Hugging Face causal-LM layout
    ``model.model.layers[layer].mlp``.  Every original module is restored even
    when model evaluation raises.  The yielded mapping exposes each prototype's
    ``last_result`` for evaluator-side local diagnostics.
    """

    decoder = getattr(getattr(model, "model", None), "layers", None)
    if decoder is None:
        raise ValueError("model does not expose model.layers")
    if not configurations:
        raise ValueError("configurations must not be empty")
    replacements: dict[int, Any] = {}
    originals: dict[int, Any] = {}
    try:
        for raw_layer, configuration in configurations.items():
            if (
                isinstance(raw_layer, bool)
                or not isinstance(raw_layer, int)
                or not 0 <= raw_layer < len(decoder)
            ):
                raise ValueError("DIP layer index is outside model.layers")
            if not isinstance(configuration, NativeBitNetDIPConfiguration):
                raise ValueError(
                    "DIP configurations must be NativeBitNetDIPConfiguration values"
                )
            originals[raw_layer] = decoder[raw_layer].mlp
            replacement = build_native_bitnet_dip_mlp(
                artifact,
                raw_layer,
                input_fraction=configuration.input_fraction,
                candidate_count=configuration.candidate_count,
                top_k=configuration.top_k,
                rms_audit_count=configuration.rms_audit_count,
                energy_target=configuration.energy_target,
                minimum_top_k=configuration.minimum_top_k,
                maximum_top_k=configuration.maximum_top_k,
                rms_variance_scale=configuration.rms_variance_scale,
                rms_variance_bias=configuration.rms_variance_bias,
                output_scale=configuration.output_scale,
                rms_estimator=configuration.rms_estimator,
                rms_audit_strategy=configuration.rms_audit_strategy,
                oracle_diagnostics=oracle_diagnostics,
            )
            decoder[raw_layer].mlp = replacement
            replacements[raw_layer] = replacement
        yield replacements
    finally:
        for layer, original in originals.items():
            decoder[layer].mlp = original


__all__ = [
    "NativeBitNetDIPConfiguration",
    "NativeBitNetDIPDiagnostics",
    "NativeBitNetDIPLayer",
    "NativeBitNetDIPResult",
    "build_native_bitnet_dip_mlp",
    "substitute_native_bitnet_dip_mlps",
]
