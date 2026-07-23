"""Deterministic product/additive quantization for semantic records.

The codec treats every input row as one semantic record.  Contiguous groups of
``group_size`` columns form product-quantization subvectors.  Each additive
stage uses one codebook shared by every subvector position, and a second stage
is trained on the residual left by the first.  Codes are stored at their
declared bit width rather than in a byte-per-code convenience representation.

This module is a portable reference implementation, not an optimized inference
kernel.  Its storage accounting is exact for the encoded array payloads:
packed codes, float16 codebooks, and optional per-record float16 scales.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


PRODUCT_ADDITIVE_SCHEMA_VERSION = 1


class ProductAdditiveQuantizationError(ValueError):
    """Raised when product/additive codec input or payload data is invalid."""


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ProductAdditiveQuantizationError(f"{name} must be a positive integer")
    return int(value)


def _matrix(values: ArrayLike) -> NDArray[np.float32]:
    try:
        result = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ProductAdditiveQuantizationError("matrix must be numeric") from exc
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ProductAdditiveQuantizationError("matrix must be a non-empty rank-2 array")
    if not np.all(np.isfinite(result)):
        raise ProductAdditiveQuantizationError(
            "matrix must contain only finite values"
        )
    return np.ascontiguousarray(result)


def _pack_fixed_width(codes: NDArray[np.integer[Any]], bits: int) -> NDArray[np.uint8]:
    """Pack unsigned integer codes consecutively, least-significant bit first."""

    flat = np.asarray(codes).reshape(-1).astype(np.uint32, copy=False)
    if flat.size and int(np.max(flat)) >= 1 << bits:
        raise ProductAdditiveQuantizationError("a code exceeds its packed bit width")
    packed = np.zeros((flat.size * bits + 7) // 8, dtype=np.uint8)
    if flat.size == 0:
        return packed
    starts = np.arange(flat.size, dtype=np.int64) * bits
    for source_bit in range(bits):
        positions = starts + source_bit
        values = ((flat >> source_bit) & 1).astype(np.uint8)
        np.bitwise_or.at(
            packed,
            positions // 8,
            values << (positions % 8).astype(np.uint8),
        )
    return packed


def _unpack_fixed_width(
    packed: NDArray[np.uint8], count: int, bits: int
) -> NDArray[np.uint16]:
    """Inverse of :func:`_pack_fixed_width`."""

    source = np.asarray(packed)
    if source.dtype != np.dtype(np.uint8) or source.ndim != 1:
        raise ProductAdditiveQuantizationError(
            "packed_codes must be a one-dimensional uint8 array"
        )
    expected = (count * bits + 7) // 8
    if source.size != expected:
        raise ProductAdditiveQuantizationError(
            f"packed code payload has {source.size} bytes; expected {expected}"
        )
    values = np.zeros(count, dtype=np.uint16)
    if count:
        starts = np.arange(count, dtype=np.int64) * bits
        for target_bit in range(bits):
            positions = starts + target_bit
            bit_values = (source[positions // 8] >> (positions % 8)) & 1
            values |= bit_values.astype(np.uint16) << target_bit
    padding = source.size * 8 - count * bits
    if padding and source.size:
        used_bits = 8 - padding
        if int(source[-1]) >> used_bits:
            raise ProductAdditiveQuantizationError(
                "packed_codes has non-zero trailing padding bits"
            )
    return values


@dataclass(frozen=True)
class ProductAdditiveMetadata:
    """Shape and codec parameters required to interpret one packed payload."""

    shape: tuple[int, int]
    group_size: int
    groups: int
    padded_width: int
    num_codebooks: int
    codebook_size: int
    code_bits: int
    iterations: int
    sample_limit: int | None
    seed: int
    per_record_scale: bool
    schema_version: int = PRODUCT_ADDITIVE_SCHEMA_VERSION
    codec: str = "product_additive_residual"
    grouping: str = "contiguous_zero_padded"
    codebook_sharing: str = "all_subvector_groups"
    packed_bit_order: str = "little_within_byte"
    codebook_dtype: str = "float16"
    scale_dtype: str | None = "float16"
    scale_method: str | None = "record_max_abs"
    decoded_dtype: str = "float32"

    def validate(self) -> None:
        if self.schema_version != PRODUCT_ADDITIVE_SCHEMA_VERSION:
            raise ProductAdditiveQuantizationError(
                f"unsupported metadata version {self.schema_version}"
            )
        if self.codec != "product_additive_residual":
            raise ProductAdditiveQuantizationError(f"unsupported codec {self.codec!r}")
        if self.grouping != "contiguous_zero_padded":
            raise ProductAdditiveQuantizationError(
                f"unsupported subvector grouping {self.grouping!r}"
            )
        if self.codebook_sharing != "all_subvector_groups":
            raise ProductAdditiveQuantizationError(
                f"unsupported codebook sharing {self.codebook_sharing!r}"
            )
        if self.packed_bit_order != "little_within_byte":
            raise ProductAdditiveQuantizationError(
                f"unsupported packed bit order {self.packed_bit_order!r}"
            )
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.shape
            )
        ):
            raise ProductAdditiveQuantizationError(
                "shape must contain two positive integers"
            )
        group_size = _positive_integer(self.group_size, "group_size")
        expected_groups = math.ceil(self.shape[1] / group_size)
        if self.groups != expected_groups:
            raise ProductAdditiveQuantizationError(
                f"groups must equal ceil(width / group_size), expected {expected_groups}"
            )
        if self.padded_width != self.groups * group_size:
            raise ProductAdditiveQuantizationError(
                "padded_width must equal groups * group_size"
            )
        if self.num_codebooks not in {1, 2}:
            raise ProductAdditiveQuantizationError("num_codebooks must be 1 or 2")
        codebook_size = _positive_integer(self.codebook_size, "codebook_size")
        if codebook_size < 2 or codebook_size > 65536:
            raise ProductAdditiveQuantizationError(
                "codebook_size must be within [2, 65536]"
            )
        expected_bits = max(1, (codebook_size - 1).bit_length())
        if self.code_bits != expected_bits:
            raise ProductAdditiveQuantizationError(
                f"code_bits must be {expected_bits} for {codebook_size} entries"
            )
        _positive_integer(self.iterations, "iterations")
        if self.sample_limit is not None:
            sample_limit = _positive_integer(self.sample_limit, "sample_limit")
            if sample_limit < codebook_size:
                raise ProductAdditiveQuantizationError(
                    "sample_limit must not be smaller than codebook_size"
                )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ProductAdditiveQuantizationError(
                "seed must be a non-negative integer"
            )
        if not isinstance(self.per_record_scale, bool):
            raise ProductAdditiveQuantizationError(
                "per_record_scale must be a boolean"
            )
        if self.codebook_dtype != "float16" or self.decoded_dtype != "float32":
            raise ProductAdditiveQuantizationError(
                "codebook_dtype must be 'float16' and decoded_dtype must be 'float32'"
            )
        expected_scale_dtype = "float16" if self.per_record_scale else None
        expected_scale_method = "record_max_abs" if self.per_record_scale else None
        if self.scale_dtype != expected_scale_dtype:
            raise ProductAdditiveQuantizationError(
                "scale_dtype disagrees with per_record_scale"
            )
        if self.scale_method != expected_scale_method:
            raise ProductAdditiveQuantizationError(
                "scale_method disagrees with per_record_scale"
            )


@dataclass(frozen=True)
class ProductAdditiveEncoding:
    """Packed product/additive codes and their reconstruction data."""

    packed_codes: NDArray[np.uint8]
    codebooks: NDArray[np.float16]
    record_scales: NDArray[np.float16] | None
    metadata: ProductAdditiveMetadata

    @property
    def code_count(self) -> int:
        return (
            self.metadata.shape[0]
            * self.metadata.num_codebooks
            * self.metadata.groups
        )

    @property
    def packed_code_bits(self) -> int:
        """Information bits used by codes, excluding final byte padding."""

        return self.code_count * self.metadata.code_bits

    @property
    def packed_code_bytes(self) -> int:
        return int(self.packed_codes.nbytes)

    @property
    def storage_bytes(self) -> int:
        """Exact encoded array bytes, including codebooks and record scales."""

        scale_bytes = 0 if self.record_scales is None else self.record_scales.nbytes
        return int(self.packed_codes.nbytes + self.codebooks.nbytes + scale_bytes)

    @property
    def storage_bits(self) -> int:
        """Exact byte-addressable payload size in bits, including padding."""

        return self.storage_bytes * 8

    @property
    def bits_per_weight(self) -> float:
        records, width = self.metadata.shape
        return self.storage_bits / (records * width)

    def storage_components(self) -> dict[str, int]:
        return {
            "packed_codes": int(self.packed_codes.nbytes),
            "codebooks": int(self.codebooks.nbytes),
            "record_scales": (
                0 if self.record_scales is None else int(self.record_scales.nbytes)
            ),
            "total": self.storage_bytes,
        }

    def validate(self) -> None:
        self.metadata.validate()
        if self.packed_codes.dtype != np.dtype(np.uint8) or self.packed_codes.ndim != 1:
            raise ProductAdditiveQuantizationError(
                "packed_codes must be a one-dimensional uint8 array"
            )
        expected_codebook_shape = (
            self.metadata.num_codebooks,
            self.metadata.codebook_size,
            self.metadata.group_size,
        )
        if (
            self.codebooks.dtype != np.dtype(np.float16)
            or self.codebooks.shape != expected_codebook_shape
        ):
            raise ProductAdditiveQuantizationError(
                "codebooks must be float16 with shape "
                f"{expected_codebook_shape}"
            )
        if not np.all(np.isfinite(self.codebooks)):
            raise ProductAdditiveQuantizationError(
                "codebooks must contain only finite values"
            )
        records = self.metadata.shape[0]
        if self.metadata.per_record_scale:
            if (
                self.record_scales is None
                or self.record_scales.dtype != np.dtype(np.float16)
                or self.record_scales.shape != (records,)
            ):
                raise ProductAdditiveQuantizationError(
                    f"record_scales must be float16 with shape ({records},)"
                )
            if not np.all(np.isfinite(self.record_scales)) or np.any(
                self.record_scales < 0
            ):
                raise ProductAdditiveQuantizationError(
                    "record_scales must be finite and non-negative"
                )
        elif self.record_scales is not None:
            raise ProductAdditiveQuantizationError(
                "record_scales must be absent when per_record_scale is false"
            )
        codes = _unpack_fixed_width(
            self.packed_codes, self.code_count, self.metadata.code_bits
        )
        if codes.size and int(np.max(codes)) >= self.metadata.codebook_size:
            raise ProductAdditiveQuantizationError(
                "a packed code is outside its codebook"
            )

    def unpack_codes(self) -> NDArray[np.uint16]:
        """Return codes as ``[records, stages, groups]`` unsigned integers."""

        self.validate()
        return _unpack_fixed_width(
            self.packed_codes, self.code_count, self.metadata.code_bits
        ).reshape(
            self.metadata.shape[0],
            self.metadata.num_codebooks,
            self.metadata.groups,
        )


def _squared_distances(
    values: NDArray[np.float32], centers: NDArray[np.float32]
) -> NDArray[np.float32]:
    value_norms = np.einsum("nd,nd->n", values, values)[:, None]
    center_norms = np.einsum("kd,kd->k", centers, centers)[None, :]
    distances = value_norms + center_norms - 2.0 * (values @ centers.T)
    return np.maximum(distances, 0.0, out=distances)


def _fit_codebook(
    values: NDArray[np.float32],
    *,
    codebook_size: int,
    iterations: int,
    sample_limit: int | None,
    seed: int,
) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    if sample_limit is not None and values.shape[0] > sample_limit:
        sample_ids = np.sort(
            rng.choice(values.shape[0], size=sample_limit, replace=False)
        )
        samples = values[sample_ids]
    else:
        samples = values
    if codebook_size > samples.shape[0]:
        raise ProductAdditiveQuantizationError(
            "codebook_size cannot exceed the number of fitted records"
        )
    initial_ids = rng.choice(samples.shape[0], size=codebook_size, replace=False)
    centers = samples[initial_ids].copy()
    for _ in range(iterations):
        assignments = np.argmin(_squared_distances(samples, centers), axis=1)
        sums = np.zeros_like(centers)
        np.add.at(sums, assignments, samples)
        counts = np.bincount(assignments, minlength=codebook_size)
        updated = centers.copy()
        populated = counts > 0
        updated[populated] = sums[populated] / counts[populated, None]
        if np.array_equal(updated, centers):
            break
        centers = updated
    return np.ascontiguousarray(centers, dtype=np.float32)


def fit_product_additive(
    matrix: ArrayLike,
    *,
    group_size: int = 8,
    num_codebooks: int = 2,
    codebook_size: int = 256,
    iterations: int = 20,
    sample_limit: int | None = None,
    seed: int = 0,
    per_record_scale: bool = True,
) -> ProductAdditiveEncoding:
    """Fit deterministic grouped codebooks and return an actually packed payload.

    ``sample_limit`` bounds the number of records used to update each codebook;
    all records are still assigned and encoded.  Repeating a fit with identical
    inputs and arguments produces identical payload arrays.
    """

    values = _matrix(matrix)
    group = _positive_integer(group_size, "group_size")
    if num_codebooks not in {1, 2}:
        raise ProductAdditiveQuantizationError("num_codebooks must be 1 or 2")
    entries = _positive_integer(codebook_size, "codebook_size")
    if entries < 2 or entries > 65536:
        raise ProductAdditiveQuantizationError(
            "codebook_size must be within [2, 65536]"
        )
    rounds = _positive_integer(iterations, "iterations")
    if sample_limit is not None:
        limit = _positive_integer(sample_limit, "sample_limit")
        if limit < entries:
            raise ProductAdditiveQuantizationError(
                "sample_limit must not be smaller than codebook_size"
            )
    else:
        limit = None
    groups = math.ceil(values.shape[1] / group)
    fitted_subvectors = values.shape[0] * groups
    if entries > fitted_subvectors:
        raise ProductAdditiveQuantizationError(
            "codebook_size cannot exceed the number of fitted subvectors"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ProductAdditiveQuantizationError("seed must be a non-negative integer")
    if not isinstance(per_record_scale, bool):
        raise ProductAdditiveQuantizationError(
            "per_record_scale must be a boolean"
        )

    records, width = values.shape
    padded_width = groups * group
    working = np.zeros((records, padded_width), dtype=np.float32)
    record_scales: NDArray[np.float16] | None
    if per_record_scale:
        exact_scales = np.max(np.abs(values), axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            record_scales = exact_scales.astype(np.float16)
        if not np.all(np.isfinite(record_scales)):
            raise ProductAdditiveQuantizationError(
                "a per-record scale is outside the finite float16 range"
            )
        divisors = record_scales.astype(np.float32)
        divisors = np.where(divisors > 0.0, divisors, 1.0)
        working[:, :width] = values / divisors[:, None]
    else:
        record_scales = None
        working[:, :width] = values

    residual = working.reshape(records, groups, group).copy()
    codebooks = np.empty(
        (num_codebooks, entries, group), dtype=np.float16
    )
    codes = np.empty((records, num_codebooks, groups), dtype=np.uint16)
    for stage in range(num_codebooks):
        stage_seed = int(
            np.random.SeedSequence([seed, stage]).generate_state(1, dtype=np.uint32)[
                0
            ]
        )
        flattened_residual = residual.reshape(records * groups, group)
        fitted_centers = _fit_codebook(
            flattened_residual,
            codebook_size=entries,
            iterations=rounds,
            sample_limit=limit,
            seed=stage_seed,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            stored_centers = fitted_centers.astype(np.float16)
        if not np.all(np.isfinite(stored_centers)):
            raise ProductAdditiveQuantizationError(
                "a fitted codebook value is outside the finite float16 range"
            )
        reconstruction_centers = stored_centers.astype(np.float32)
        assignments = np.argmin(
            _squared_distances(flattened_residual, reconstruction_centers), axis=1
        ).astype(np.uint16)
        codebooks[stage] = stored_centers
        codes[:, stage, :] = assignments.reshape(records, groups)
        flattened_residual -= reconstruction_centers[assignments]

    code_bits = max(1, (entries - 1).bit_length())
    metadata = ProductAdditiveMetadata(
        shape=(records, width),
        group_size=group,
        groups=groups,
        padded_width=padded_width,
        num_codebooks=num_codebooks,
        codebook_size=entries,
        code_bits=code_bits,
        iterations=rounds,
        sample_limit=limit,
        seed=seed,
        per_record_scale=per_record_scale,
        scale_dtype="float16" if per_record_scale else None,
        scale_method="record_max_abs" if per_record_scale else None,
    )
    encoding = ProductAdditiveEncoding(
        packed_codes=_pack_fixed_width(codes, code_bits),
        codebooks=codebooks,
        record_scales=record_scales,
        metadata=metadata,
    )
    encoding.validate()
    return encoding


def decode_product_additive(
    encoding: ProductAdditiveEncoding,
) -> NDArray[np.float32]:
    """Decode all semantic records to a float32 ``[records, width]`` matrix."""

    encoding.validate()
    metadata = encoding.metadata
    codes = encoding.unpack_codes()
    decoded = np.zeros(
        (metadata.shape[0], metadata.groups, metadata.group_size),
        dtype=np.float32,
    )
    rows = np.arange(metadata.shape[0])
    for stage in range(metadata.num_codebooks):
        for group_index in range(metadata.groups):
            decoded[:, group_index, :] += encoding.codebooks[
                stage, codes[rows, stage, group_index]
            ].astype(np.float32)
    result = decoded.reshape(metadata.shape[0], metadata.padded_width)[
        :, : metadata.shape[1]
    ]
    if encoding.record_scales is not None:
        result = result * encoding.record_scales.astype(np.float32)[:, None]
    return np.ascontiguousarray(result, dtype=np.float32)


__all__ = [
    "ProductAdditiveEncoding",
    "ProductAdditiveMetadata",
    "ProductAdditiveQuantizationError",
    "decode_product_additive",
    "fit_product_additive",
]
