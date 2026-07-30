"""Causal mass-vector correction selector for the full-visible C28 basis.

The selector consumes only scores already produced by the bounded attention
kernel.  It predicts one additive correction for each of the 28 values that
the kernel already reads, then applies the corrected softmax before the
existing single value-accumulation pass.  No new key/value state or reads are
introduced.

This module contains the numerical reference and deterministic training core.
Protocol authentication and CLI orchestration live in
``olmoe_retrieval_episodic_mass_selector_runner``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


_PRODUCTION_LAYERS = 16
_PRODUCTION_HEADS = 16
_PRODUCTION_COMPONENTS = 28
_PRODUCTION_HEAD_DIMENSION = 128
_PRODUCTION_RANK = 16
_PRODUCTION_POSITIONS = 128
_PRODUCTION_FIXED_TRAFFIC_BYTES = 714_866_688
_PRODUCTION_DENSE_BYTES = 2_164_260_864
_PRODUCTION_INIT_SEED = 2_026_073_001
_PRODUCTION_SHUFFLE_SEED = 2_026_073_002
_PRODUCTION_FOLDS = ((0, 4), (1, 5), (2, 6), (3, 7))


@dataclass(frozen=True)
class SelectorShape:
    """Static shape of one selector artifact."""

    layers: int = _PRODUCTION_LAYERS
    heads: int = _PRODUCTION_HEADS
    components: int = _PRODUCTION_COMPONENTS
    head_dimension: int = _PRODUCTION_HEAD_DIMENSION
    rank: int = _PRODUCTION_RANK

    def validate(self) -> None:
        if (
            min(
                self.layers,
                self.heads,
                self.components,
                self.head_dimension,
                self.rank,
            )
            <= 0
        ):
            raise ValueError("mass-selector dimensions must be positive")

    @property
    def hidden_size(self) -> int:
        self.validate()
        return self.heads * self.head_dimension

    @property
    def parameter_count(self) -> int:
        self.validate()
        return (
            self.layers * self.components * self.rank
            + self.layers * self.rank * self.components
            + self.layers * self.heads * self.rank
            + self.layers * self.heads * self.components
        )


@dataclass(frozen=True)
class SelectorParameters:
    """FP32 reference parameters.

    ``U`` and ``V`` are shared across query heads within a layer.  ``E`` and
    ``B`` let each head adapt that shared map without duplicating its matrices.
    """

    U: np.ndarray
    V: np.ndarray
    E: np.ndarray
    B: np.ndarray

    def validate(self, shape: SelectorShape) -> None:
        shape.validate()
        expected = {
            "U": (shape.layers, shape.components, shape.rank),
            "V": (shape.layers, shape.rank, shape.components),
            "E": (shape.layers, shape.heads, shape.rank),
            "B": (shape.layers, shape.heads, shape.components),
        }
        for name, value in (
            ("U", self.U),
            ("V", self.V),
            ("E", self.E),
            ("B", self.B),
        ):
            array = np.asarray(value)
            if (
                array.shape != expected[name]
                or array.dtype != np.float32
                or not array.flags.c_contiguous
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"mass-selector {name} parameters are invalid")

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "U": self.U,
            "V": self.V,
            "E": self.E,
            "B": self.B,
        }


@dataclass(frozen=True)
class TrainingConfig:
    """Prospectively fixed optimizer and batching contract."""

    steps: int = 1_536
    warmup_steps: int = 96
    peak_learning_rate: float = 5.0e-3
    final_learning_rate: float = 5.0e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    uv_weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    rows_per_layer_per_step: int = 2
    epochs: int = 16
    feature_clip: float = 16.0
    delta_clamp: float = 16.0
    initial_u_standard_deviation: float = 0.02
    init_seed: int = _PRODUCTION_INIT_SEED
    shuffle_seed: int = _PRODUCTION_SHUFFLE_SEED

    def validate(self) -> None:
        finite_positive = (
            self.peak_learning_rate,
            self.final_learning_rate,
            self.epsilon,
            self.gradient_clip_norm,
            self.feature_clip,
            self.delta_clamp,
            self.initial_u_standard_deviation,
        )
        if (
            self.steps <= 0
            or self.warmup_steps <= 0
            or self.warmup_steps >= self.steps - 1
            or self.rows_per_layer_per_step <= 0
            or self.epochs <= 0
            or not all(np.isfinite(value) and value > 0.0 for value in finite_positive)
            or not np.isfinite(self.uv_weight_decay)
            or self.uv_weight_decay < 0.0
            or not (0.0 < self.beta1 < 1.0)
            or not (0.0 < self.beta2 < 1.0)
            or isinstance(self.init_seed, bool)
            or not isinstance(self.init_seed, int)
            or self.init_seed < 0
            or isinstance(self.shuffle_seed, bool)
            or not isinstance(self.shuffle_seed, int)
            or self.shuffle_seed < 0
        ):
            raise ValueError("mass-selector training configuration is invalid")


@dataclass(frozen=True)
class TrainingResult:
    """One fixed final-step fit."""

    parameters: SelectorParameters
    initial_loss: float
    final_loss: float
    learning_rate_sha256: str
    schedule_sha256: str
    steps: int
    device: str


def initialize_parameters(
    shape: SelectorShape,
    *,
    seed: int,
    standard_deviation: float = 0.02,
) -> SelectorParameters:
    """Initialize at exact native attention while keeping a gradient path."""

    shape.validate()
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or not np.isfinite(standard_deviation)
        or standard_deviation <= 0.0
    ):
        raise ValueError("mass-selector initialization is invalid")
    generator = np.random.Generator(np.random.PCG64(seed))
    U = generator.normal(
        0.0,
        standard_deviation,
        size=(shape.layers, shape.components, shape.rank),
    ).astype(np.float32)
    return SelectorParameters(
        U=np.ascontiguousarray(U),
        V=np.zeros(
            (shape.layers, shape.rank, shape.components),
            dtype=np.float32,
        ),
        E=np.zeros(
            (shape.layers, shape.heads, shape.rank),
            dtype=np.float32,
        ),
        B=np.zeros(
            (shape.layers, shape.heads, shape.components),
            dtype=np.float32,
        ),
    )


def _validate_mass_and_mask(
    native_mass: np.ndarray,
    valid: np.ndarray,
    shape: SelectorShape,
) -> tuple[np.ndarray, np.ndarray]:
    mass = np.ascontiguousarray(native_mass, dtype=np.float32)
    mask = np.ascontiguousarray(valid, dtype=bool)
    if (
        mass.shape != mask.shape
        or mass.ndim < 3
        or mass.shape[-3:] != (shape.layers, shape.heads, shape.components)
        or not np.isfinite(mass).all()
        or np.any(mass < 0.0)
        or np.any(np.sum(mask, axis=-1) <= 0)
        or np.any(mass[~mask] != 0.0)
        or np.any(mass[mask] <= 0.0)
    ):
        raise ValueError("mass-selector native mass or mask is invalid")
    sums = np.sum(mass, axis=-1, dtype=np.float32)
    if np.max(np.abs(sums - np.float32(1.0))) > 2.0e-5:
        raise ValueError("mass-selector native masses do not sum to one")
    return mass, mask


def centered_log_mass_features(
    native_mass: np.ndarray,
    valid: np.ndarray,
    shape: SelectorShape,
    *,
    clip: float = 16.0,
) -> np.ndarray:
    """Reconstruct the centered native-score gauge from cached masses."""

    mass, mask = _validate_mass_and_mask(native_mass, valid, shape)
    if not np.isfinite(clip) or clip <= 0.0:
        raise ValueError("mass-selector feature clip is invalid")
    log_mass = np.zeros_like(mass, dtype=np.float32)
    log_mass[mask] = np.log(mass[mask])
    count = np.sum(mask, axis=-1, keepdims=True).astype(np.float32)
    mean = (
        np.sum(
            np.where(mask, log_mass, np.float32(0.0)),
            axis=-1,
            keepdims=True,
            dtype=np.float32,
        )
        / count
    )
    features = np.where(mask, log_mass - mean, 0.0)
    features = np.where(mask, np.clip(features, -clip, clip), 0.0)
    return np.ascontiguousarray(features, dtype=np.float32)


def centered_score_features(
    native_scores: np.ndarray,
    valid: np.ndarray,
    shape: SelectorShape,
    *,
    clip: float = 16.0,
) -> np.ndarray:
    """Center and clip the actual production score vector in FP32."""

    scores = np.ascontiguousarray(native_scores, dtype=np.float32)
    mask = np.ascontiguousarray(valid, dtype=bool)
    if (
        scores.ndim < 3
        or scores.shape != mask.shape
        or scores.shape[-3:] != (shape.layers, shape.heads, shape.components)
        or not np.isfinite(scores[mask]).all()
        or np.any(np.sum(mask, axis=-1) <= 0)
        or not np.isfinite(clip)
        or clip <= 0.0
    ):
        raise ValueError("mass-selector native scores or mask are invalid")
    count = np.sum(mask, axis=-1, keepdims=True).astype(np.float32)
    mean = (
        np.sum(
            np.where(mask, scores, np.float32(0.0)),
            axis=-1,
            keepdims=True,
            dtype=np.float32,
        )
        / count
    )
    features = np.where(
        mask,
        np.clip(scores - mean, -clip, clip),
        np.float32(0.0),
    )
    return np.ascontiguousarray(features, dtype=np.float32)


def _gauge_and_clamp(
    delta: np.ndarray,
    valid: np.ndarray,
    *,
    clamp: float,
) -> np.ndarray:
    values = np.ascontiguousarray(delta, dtype=np.float32)
    mask = np.ascontiguousarray(valid, dtype=bool)
    if (
        values.shape != mask.shape
        or not np.isfinite(values).all()
        or not np.isfinite(clamp)
        or clamp <= 0.0
        or np.any(np.sum(mask, axis=-1) <= 0)
    ):
        raise ValueError("mass-selector correction gauge is invalid")
    count = np.sum(mask, axis=-1, keepdims=True).astype(np.float32)
    mean = (
        np.sum(
            np.where(mask, values, np.float32(0.0)),
            axis=-1,
            keepdims=True,
            dtype=np.float32,
        )
        / count
    )
    centered = np.where(mask, values - mean, 0.0)
    centered = np.where(mask, np.clip(centered, -clamp, clamp), 0.0)
    return np.ascontiguousarray(centered, dtype=np.float32)


def _masked_softmax(
    logits: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    scores = np.ascontiguousarray(logits, dtype=np.float32)
    mask = np.ascontiguousarray(valid, dtype=bool)
    if (
        scores.shape != mask.shape
        or not np.isfinite(scores[mask]).all()
        or np.any(np.sum(mask, axis=-1) <= 0)
    ):
        raise ValueError("mass-selector logits or mask is invalid")
    maximum = np.max(np.where(mask, scores, -np.inf), axis=-1, keepdims=True)
    shifted = np.where(mask, scores - maximum, np.float32(-np.inf))
    exponential = np.exp(shifted).astype(np.float32, copy=False)
    denominator = np.sum(
        exponential,
        axis=-1,
        keepdims=True,
        dtype=np.float32,
    )
    result = np.ascontiguousarray(exponential / denominator, dtype=np.float32)
    if (
        not np.isfinite(result).all()
        or np.any(result < 0.0)
        or np.any(result[~mask] != 0.0)
        or np.max(np.abs(np.sum(result, axis=-1, dtype=np.float32) - np.float32(1.0)))
        > 2.0e-6
    ):
        raise ValueError("mass-selector masked softmax failed")
    return result


def coefficients_from_mass(
    native_mass: np.ndarray,
    valid: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """Apply the cached mass route: normalize(mass * exp(delta))."""

    mass = np.ascontiguousarray(native_mass, dtype=np.float32)
    mask = np.ascontiguousarray(valid, dtype=bool)
    correction = np.ascontiguousarray(delta, dtype=np.float32)
    if (
        mass.shape != mask.shape
        or mass.shape != correction.shape
        or np.any(mass[~mask] != 0.0)
        or np.any(mass[mask] <= 0.0)
    ):
        raise ValueError("mass-selector mass route is invalid")
    logits = np.full(mass.shape, -np.inf, dtype=np.float32)
    logits[mask] = np.log(mass[mask]) + correction[mask]
    return _masked_softmax(logits, mask)


def coefficients_from_scores(
    native_scores: np.ndarray,
    valid: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """Apply the production route: softmax(native_score + delta)."""

    scores = np.ascontiguousarray(native_scores, dtype=np.float32)
    mask = np.ascontiguousarray(valid, dtype=bool)
    correction = np.ascontiguousarray(delta, dtype=np.float32)
    if scores.shape != mask.shape or scores.shape != correction.shape:
        raise ValueError("mass-selector score route is invalid")
    return _masked_softmax(scores + correction, mask)


def _selector_delta(
    features: np.ndarray,
    valid: np.ndarray,
    parameters: SelectorParameters,
    *,
    delta_clamp: float,
) -> np.ndarray:
    """Apply selector weights with production FP32 accumulation."""

    mask = np.ascontiguousarray(valid, dtype=bool)
    hidden = np.einsum(
        "...lhc,lcr->...lhr",
        features,
        parameters.U,
        optimize=True,
    )
    hidden += parameters.E
    np.maximum(hidden, 0.0, out=hidden)
    raw_delta = np.einsum(
        "...lhr,lrc->...lhc",
        hidden,
        parameters.V,
        optimize=True,
    )
    raw_delta += parameters.B
    return _gauge_and_clamp(raw_delta, mask, clamp=delta_clamp)


def selector_forward(
    native_mass: np.ndarray,
    valid: np.ndarray,
    parameters: SelectorParameters,
    shape: SelectorShape,
    *,
    feature_clip: float = 16.0,
    delta_clamp: float = 16.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the cached-mass FP32 reference used for trace evaluation."""

    parameters.validate(shape)
    mass, mask = _validate_mass_and_mask(native_mass, valid, shape)
    features = centered_log_mass_features(
        mass,
        mask,
        shape,
        clip=feature_clip,
    )
    delta = _selector_delta(
        features,
        mask,
        parameters,
        delta_clamp=delta_clamp,
    )
    coefficients = coefficients_from_mass(mass, mask, delta)
    return coefficients, delta


def selector_forward_from_scores(
    native_scores: np.ndarray,
    valid: np.ndarray,
    parameters: SelectorParameters,
    shape: SelectorShape,
    *,
    feature_clip: float = 16.0,
    delta_clamp: float = 16.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the candidate native-score path.

    Cached experiments can exercise this implementation with ``log(mass)``,
    but that does not authenticate parity with the original rounded q·k
    scores.  Native integration must establish that separately.
    """

    parameters.validate(shape)
    scores = np.ascontiguousarray(native_scores, dtype=np.float32)
    mask = np.ascontiguousarray(valid, dtype=bool)
    features = centered_score_features(
        scores,
        mask,
        shape,
        clip=feature_clip,
    )
    delta = _selector_delta(
        features,
        mask,
        parameters,
        delta_clamp=delta_clamp,
    )
    coefficients = coefficients_from_scores(scores, mask, delta)
    return coefficients, delta


def float32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round finite FP32 values to BF16 using round-to-nearest-even."""

    array = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("mass-selector BF16 source is non-finite")
    raw = array.view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((raw >> np.uint32(16)) & np.uint32(1))
    return np.ascontiguousarray(((raw + rounding) >> np.uint32(16)).astype(np.uint16))


def bf16_bits_to_float32(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.uint16)
    raw = bits.astype(np.uint32) << np.uint32(16)
    result = np.ascontiguousarray(raw.view(np.float32))
    if not np.isfinite(result).all():
        raise ValueError("mass-selector BF16 artifact is non-finite")
    return result


def quantize_parameters_bf16(
    parameters: SelectorParameters,
    shape: SelectorShape,
) -> tuple[SelectorParameters, dict[str, np.ndarray]]:
    parameters.validate(shape)
    bits = {
        name: float32_to_bf16_bits(value)
        for name, value in parameters.as_dict().items()
    }
    decoded = SelectorParameters(
        U=bf16_bits_to_float32(bits["U"]),
        V=bf16_bits_to_float32(bits["V"]),
        E=bf16_bits_to_float32(bits["E"]),
        B=bf16_bits_to_float32(bits["B"]),
    )
    decoded.validate(shape)
    return decoded, bits


def production_resource_contract(
    shape: SelectorShape = SelectorShape(),
) -> dict[str, Any]:
    """Return conservative all-token CPU accounting."""

    if shape != SelectorShape():
        raise ValueError("production resource accounting requires production shape")
    parameter_count = shape.parameter_count
    parameter_bytes = parameter_count * 2
    weight_traffic = parameter_bytes * _PRODUCTION_POSITIONS
    total = _PRODUCTION_FIXED_TRAFFIC_BYTES + weight_traffic
    scratch = (
        shape.heads * shape.components * 4
        + shape.heads * shape.rank * 4
        + shape.heads * shape.components * 4
        + shape.heads * shape.components * 4
    )
    macs_per_token = (
        shape.layers
        * shape.heads
        * (shape.components * shape.rank + shape.rank * shape.components)
    )
    if (
        parameter_count != 25_600
        or parameter_bytes != 51_200
        or weight_traffic != 6_553_600
        or total != 721_420_288
        or total * 3 != _PRODUCTION_DENSE_BYTES
        or scratch != 6_400
        or macs_per_token != 229_376
    ):
        raise AssertionError("mass-selector production accounting changed")
    return {
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_bytes,
        "selector_weight_traffic_bytes_per_128_tokens": weight_traffic,
        "combined_logical_traffic_bytes_per_128_tokens": total,
        "fraction_of_dense": total / _PRODUCTION_DENSE_BYTES,
        "remaining_bytes_below_45_percent_floor": 252_497_100,
        "selector_scratch_bytes": scratch,
        "selector_macs_per_token": macs_per_token,
        "selector_macs_per_128_tokens": macs_per_token * _PRODUCTION_POSITIONS,
        "new_KV_state_bytes": 0,
        "new_KV_read_bytes": 0,
    }


def fixed_folds(
    record_count: int = 8,
    *,
    heldout_pairs: Sequence[Sequence[int]] = _PRODUCTION_FOLDS,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return record-disjoint train/heldout folds."""

    if record_count <= 0:
        raise ValueError("mass-selector record count is invalid")
    all_records = tuple(range(record_count))
    seen: list[int] = []
    folds: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for pair in heldout_pairs:
        heldout = tuple(int(value) for value in pair)
        if (
            not heldout
            or len(set(heldout)) != len(heldout)
            or any(value < 0 or value >= record_count for value in heldout)
        ):
            raise ValueError("mass-selector heldout fold is invalid")
        heldout_set = set(heldout)
        training = tuple(value for value in all_records if value not in heldout_set)
        folds.append((training, heldout))
        seen.extend(heldout)
    if sorted(seen) != list(all_records):
        raise ValueError("mass-selector folds do not cover each record exactly once")
    return tuple(folds)


def build_training_schedule(
    training_records: Sequence[int],
    *,
    read_positions: int,
    layers: int,
    epochs: int,
    rows_per_layer_per_step: int,
    seed: int,
) -> np.ndarray:
    """Build independent per-layer epoch permutations.

    Values are flattened ``record * read_positions + read_index`` identifiers.
    The result shape is ``[steps, layers, rows_per_layer_per_step]``.
    """

    records = tuple(int(value) for value in training_records)
    if (
        not records
        or len(set(records)) != len(records)
        or min(records) < 0
        or read_positions <= 0
        or layers <= 0
        or epochs <= 0
        or rows_per_layer_per_step <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError("mass-selector training schedule configuration is invalid")
    rows = np.asarray(
        [
            record * read_positions + position
            for record in records
            for position in range(read_positions)
        ],
        dtype=np.int64,
    )
    if rows.size % rows_per_layer_per_step:
        raise ValueError("mass-selector rows do not divide into complete steps")
    steps_per_epoch = rows.size // rows_per_layer_per_step
    generator = np.random.Generator(np.random.PCG64(seed))
    schedule = np.empty(
        (epochs * steps_per_epoch, layers, rows_per_layer_per_step),
        dtype=np.int64,
    )
    for epoch in range(epochs):
        begin = epoch * steps_per_epoch
        end = begin + steps_per_epoch
        for layer in range(layers):
            schedule[begin:end, layer] = generator.permutation(rows).reshape(
                steps_per_epoch,
                rows_per_layer_per_step,
            )
    return np.ascontiguousarray(schedule)


def schedule_sha256(schedule: np.ndarray) -> str:
    array = np.ascontiguousarray(schedule, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def learning_rate_schedule(config: TrainingConfig) -> np.ndarray:
    config.validate()
    values = np.empty(config.steps, dtype=np.float64)
    for step in range(config.steps):
        if step < config.warmup_steps:
            value = config.peak_learning_rate * (step + 1) / config.warmup_steps
        else:
            progress = (step - config.warmup_steps) / (
                config.steps - config.warmup_steps - 1
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            value = (
                config.final_learning_rate
                + (config.peak_learning_rate - config.final_learning_rate) * cosine
            )
        values[step] = value
    return values


def direct_post_wo_error_energy(
    coefficients: np.ndarray,
    visible_values: np.ndarray,
    base_heads: np.ndarray,
    target_residual: np.ndarray,
    output_projection: np.ndarray,
    shape: SelectorShape,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the exact post-output-projection row/layer error energy.

    Leading dimensions before ``layers`` are arbitrary and are preserved.
    """

    alpha = np.ascontiguousarray(coefficients, dtype=np.float64)
    values = np.ascontiguousarray(visible_values, dtype=np.float64)
    base = np.ascontiguousarray(base_heads, dtype=np.float64)
    target = np.ascontiguousarray(target_residual, dtype=np.float64)
    weights = np.ascontiguousarray(output_projection, dtype=np.float64)
    prefix = alpha.shape[:-3]
    if (
        alpha.shape[-3:] != (shape.layers, shape.heads, shape.components)
        or values.shape
        != prefix
        + (
            shape.layers,
            shape.heads,
            shape.components,
            shape.head_dimension,
        )
        or base.shape != prefix + (shape.layers, shape.heads, shape.head_dimension)
        or target.shape != prefix + (shape.layers, shape.hidden_size)
        or weights.shape != (shape.layers, shape.hidden_size, shape.hidden_size)
        or not all(
            np.isfinite(array).all() for array in (alpha, values, base, target, weights)
        )
        or np.any(alpha < -1.0e-12)
        or np.max(np.abs(np.sum(alpha, axis=-1, dtype=np.float32) - np.float32(1.0)))
        > 2.0e-6
    ):
        raise ValueError("mass-selector direct-loss tensors are invalid")
    rows = int(np.prod(prefix, dtype=np.int64)) if prefix else 1
    flat_alpha = alpha.reshape(rows, shape.layers, shape.heads, shape.components)
    flat_values = values.reshape(
        rows,
        shape.layers,
        shape.heads,
        shape.components,
        shape.head_dimension,
    )
    flat_base = base.reshape(
        rows,
        shape.layers,
        shape.heads,
        shape.head_dimension,
    )
    flat_target = target.reshape(rows, shape.layers, shape.hidden_size)
    error_energy = np.empty((rows, shape.layers), dtype=np.float64)
    target_energy = np.empty((rows, shape.layers), dtype=np.float64)
    for layer in range(shape.layers):
        candidate = np.einsum(
            "rhc,rhcd->rhd",
            flat_alpha[:, layer],
            flat_values[:, layer],
            optimize=True,
        )
        delta = (candidate - flat_base[:, layer]).reshape(rows, shape.hidden_size)
        correction = delta @ weights[layer].T
        error = flat_target[:, layer] - correction
        error_energy[:, layer] = np.einsum(
            "ri,ri->r",
            error,
            error,
            optimize=True,
        )
        target_energy[:, layer] = np.einsum(
            "ri,ri->r",
            flat_target[:, layer],
            flat_target[:, layer],
            optimize=True,
        )
    return (
        np.ascontiguousarray(error_energy.reshape(prefix + (shape.layers,))),
        np.ascontiguousarray(target_energy.reshape(prefix + (shape.layers,))),
    )


def _torch_forward(
    torch: Any,
    mass: Any,
    valid: Any,
    U: Any,
    V: Any,
    E: Any,
    B: Any,
    *,
    feature_clip: float,
    delta_clamp: float,
) -> Any:
    """Torch forward for all layers: inputs are [layer, batch, head, component]."""

    log_mass = torch.where(valid, torch.log(mass), torch.zeros_like(mass))
    count = valid.sum(dim=-1, keepdim=True)
    feature_mean = (
        torch.where(valid, log_mass, 0.0).sum(
            dim=-1,
            keepdim=True,
        )
        / count
    )
    features = torch.where(
        valid,
        torch.clamp(log_mass - feature_mean, -feature_clip, feature_clip),
        0.0,
    )
    hidden = torch.relu(torch.einsum("lbhc,lcr->lbhr", features, U) + E[:, None])
    raw_delta = torch.einsum("lbhr,lrc->lbhc", hidden, V) + B[:, None]
    delta_mean = (
        torch.where(valid, raw_delta, 0.0).sum(
            dim=-1,
            keepdim=True,
        )
        / count
    )
    delta = torch.where(
        valid,
        torch.clamp(raw_delta - delta_mean, -delta_clamp, delta_clamp),
        0.0,
    )
    logits = torch.where(
        valid,
        log_mass + delta,
        torch.full_like(log_mass, -torch.inf),
    )
    return torch.softmax(logits, dim=-1)


def fit_direct_post_wo(
    native_mass: np.ndarray,
    valid: np.ndarray,
    visible_values: np.ndarray,
    base_heads: np.ndarray,
    target_residual: np.ndarray,
    output_projection: np.ndarray,
    *,
    training_records: Sequence[int],
    shape: SelectorShape,
    config: TrainingConfig,
    device: str = "cpu",
) -> TrainingResult:
    """Fit one final-step model without consulting heldout records.

    Array layout is ``[records, reads, layers, ...]``.  Every optimizer step
    samples the frozen number of complete record/read rows independently for
    every layer and always includes all query heads.
    """

    config.validate()
    shape.validate()
    mass = np.ascontiguousarray(native_mass, dtype=np.float32)
    mask = np.ascontiguousarray(valid, dtype=bool)
    values = np.ascontiguousarray(visible_values, dtype=np.float32)
    base = np.ascontiguousarray(base_heads, dtype=np.float32)
    target = np.ascontiguousarray(target_residual, dtype=np.float32)
    weights = np.ascontiguousarray(output_projection, dtype=np.float32)
    if mass.ndim != 5:
        raise ValueError("mass-selector training masses must be rank five")
    records, reads = mass.shape[:2]
    expected_mass = (records, reads, shape.layers, shape.heads, shape.components)
    if (
        mass.shape != expected_mass
        or mask.shape != expected_mass
        or values.shape != expected_mass + (shape.head_dimension,)
        or base.shape
        != (
            records,
            reads,
            shape.layers,
            shape.heads,
            shape.head_dimension,
        )
        or target.shape != (records, reads, shape.layers, shape.hidden_size)
        or weights.shape != (shape.layers, shape.hidden_size, shape.hidden_size)
    ):
        raise ValueError("mass-selector training tensor shapes are invalid")
    _validate_mass_and_mask(mass, mask, shape)
    train = tuple(int(value) for value in training_records)
    if (
        not train
        or len(set(train)) != len(train)
        or min(train) < 0
        or max(train) >= records
    ):
        raise ValueError("mass-selector training records are invalid")
    rows_per_epoch = len(train) * reads
    if (
        rows_per_epoch % config.rows_per_layer_per_step
        or config.steps
        != config.epochs * rows_per_epoch // config.rows_per_layer_per_step
    ):
        raise ValueError("mass-selector fixed epoch schedule does not match data")

    try:
        import torch
    except ImportError as error:  # pragma: no cover - project dependency
        raise RuntimeError("mass-selector training requires torch") from error
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("mass-selector CUDA device is unavailable")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise ValueError(
                "mass-selector deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG"
            )
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.init_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.init_seed)
    torch_device = torch.device(device)

    initialized = initialize_parameters(
        shape,
        seed=config.init_seed,
        standard_deviation=config.initial_u_standard_deviation,
    )
    U = torch.nn.Parameter(torch.from_numpy(initialized.U.copy()).to(torch_device))
    V = torch.nn.Parameter(torch.from_numpy(initialized.V.copy()).to(torch_device))
    E = torch.nn.Parameter(torch.from_numpy(initialized.E.copy()).to(torch_device))
    B = torch.nn.Parameter(torch.from_numpy(initialized.B.copy()).to(torch_device))
    optimizer = torch.optim.AdamW(
        [
            {"params": [U, V], "weight_decay": config.uv_weight_decay},
            {"params": [E, B], "weight_decay": 0.0},
        ],
        lr=config.peak_learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
    )
    schedule = build_training_schedule(
        train,
        read_positions=reads,
        layers=shape.layers,
        epochs=config.epochs,
        rows_per_layer_per_step=config.rows_per_layer_per_step,
        seed=config.shuffle_seed,
    )
    rates = learning_rate_schedule(config)
    flat_mass = mass.reshape(
        records * reads,
        shape.layers,
        shape.heads,
        shape.components,
    )
    flat_mask = mask.reshape(flat_mass.shape)
    flat_values = values.reshape(
        records * reads,
        shape.layers,
        shape.heads,
        shape.components,
        shape.head_dimension,
    )
    flat_base = base.reshape(
        records * reads,
        shape.layers,
        shape.heads,
        shape.head_dimension,
    )
    flat_target = target.reshape(
        records * reads,
        shape.layers,
        shape.hidden_size,
    )
    train_target = target[np.asarray(train, dtype=np.int64)]
    target_denominator = float(
        np.mean(np.sum(train_target.astype(np.float64) ** 2, axis=-1))
    )
    if not np.isfinite(target_denominator) or target_denominator <= 0.0:
        raise ValueError("mass-selector training target energy is invalid")
    # Keep the authenticated trace resident on the training device.  The
    # production screen otherwise performs thousands of tiny host-to-device
    # copies and serial layer launches even though every step visits all
    # layers.  Layer-specific schedule rows are gathered in one operation and
    # the dense output projection is evaluated as one batched matrix product.
    torch_mass = torch.from_numpy(flat_mass).to(torch_device)
    torch_mask = torch.from_numpy(flat_mask).to(torch_device)
    torch_values = torch.from_numpy(flat_values).to(torch_device)
    torch_base = torch.from_numpy(flat_base).to(torch_device)
    torch_target = torch.from_numpy(flat_target).to(torch_device)
    torch_weights = torch.from_numpy(weights).to(torch_device)
    torch_schedule = torch.from_numpy(schedule).to(torch_device)
    layer_indices = torch.arange(
        shape.layers,
        dtype=torch.long,
        device=torch_device,
    )[:, None]

    initial_loss = float("nan")
    final_loss = float("nan")
    for step in range(config.steps):
        for group in optimizer.param_groups:
            group["lr"] = float(rates[step])
        optimizer.zero_grad(set_to_none=True)
        row_ids = torch_schedule[step]
        batch_mass = torch_mass[row_ids, layer_indices]
        batch_mask = torch_mask[row_ids, layer_indices]
        batch_values = torch_values[row_ids, layer_indices]
        batch_base = torch_base[row_ids, layer_indices]
        batch_target = torch_target[row_ids, layer_indices]
        alpha = _torch_forward(
            torch,
            batch_mass,
            batch_mask,
            U,
            V,
            E,
            B,
            feature_clip=config.feature_clip,
            delta_clamp=config.delta_clamp,
        )
        candidate = torch.einsum("lbhc,lbhcd->lbhd", alpha, batch_values)
        pre_wo = (candidate - batch_base).reshape(
            shape.layers,
            config.rows_per_layer_per_step,
            shape.hidden_size,
        )
        correction = torch.bmm(pre_wo, torch_weights.transpose(1, 2))
        error = batch_target - correction
        total_error = torch.sum(error * error)
        total_rows = shape.layers * config.rows_per_layer_per_step
        loss = total_error / (total_rows * target_denominator)
        if not torch.isfinite(loss):
            raise ValueError("mass-selector training loss became non-finite")
        if step == 0:
            initial_loss = float(loss.detach().cpu())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (U, V, E, B),
            config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    fitted = SelectorParameters(
        U=np.ascontiguousarray(U.detach().cpu().numpy(), dtype=np.float32),
        V=np.ascontiguousarray(V.detach().cpu().numpy(), dtype=np.float32),
        E=np.ascontiguousarray(E.detach().cpu().numpy(), dtype=np.float32),
        B=np.ascontiguousarray(B.detach().cpu().numpy(), dtype=np.float32),
    )
    fitted.validate(shape)
    return TrainingResult(
        parameters=fitted,
        initial_loss=initial_loss,
        final_loss=final_loss,
        learning_rate_sha256=hashlib.sha256(rates.tobytes(order="C")).hexdigest(),
        schedule_sha256=schedule_sha256(schedule),
        steps=config.steps,
        device=str(torch_device),
    )


def parameters_sha256(parameters: SelectorParameters, shape: SelectorShape) -> str:
    parameters.validate(shape)
    digest = hashlib.sha256()
    for name in ("U", "V", "E", "B"):
        value = np.ascontiguousarray(parameters.as_dict()[name])
        digest.update(name.encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()
