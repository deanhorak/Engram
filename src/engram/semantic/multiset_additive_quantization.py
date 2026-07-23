"""Serialized multi-set additive quantization for joint SwiGLU weights.

The input convention is the joint row stack ``[gate; up; down.T]``.  Equal
row ranges identify projections, and optional equal buckets of contiguous
input-position groups identify finer codebook sets.  Set selection is therefore
static: this format has no activation-dependent router or random posting-list
traffic.

Two schema-v1 profiles are supported:

``projection_2x7_scale6`` (primary)
    Two 7-bit residual codes per group, one codebook set per projection, and a
    6-bit quantized row scale.  On a ``[4608, 576]`` stack this complete artifact
    is 596,992 bytes, 44.9846% of raw dense Q4.

``position_3x4_fp16_scale`` (fallback)
    Three nibble-packed residual codes, projection/position-bucket codebook
    sets, and float16 row scales.  With 36 position buckets the same stack is
    590,296 bytes, 44.4800% of raw dense Q4.

Every artifact contains a fixed 256-byte header with a SHA-256 checksum, packed
codes, float16 codebooks, the selected scale representation, and any explicit
set mapping.  Byte accounting includes every one of those sections.  The
deterministic residual k-means fitter is an initializer; activation-aware P/V
refinement can replace codes and centroids without changing the wire format.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


MULTISET_ADDITIVE_SCHEMA_VERSION = 1
MULTISET_ADDITIVE_HEADER_BYTES = 256
MULTISET_ADDITIVE_MAGIC = b"ENMSAQ01"

PROJECTION_2X7_SCALE6 = "projection_2x7_scale6"
POSITION_3X4_FP16_SCALE = "position_3x4_fp16_scale"

PRIMARY_GROUP_SIZE = 8
PRIMARY_PROJECTION_COUNT = 3
PRIMARY_POSITION_BUCKETS = 1
FALLBACK_POSITION_BUCKETS = 36

_FLAG_QUANTIZED_SCALES = 1
_FLAG_RAW_FP16_SCALES = 2
_FLAG_EXPLICIT_SET_MAPPING = 4

# magic, schema/header, 17 uint32 fields, seven uint64 fields, checksum.
# Unused bytes in the fixed header must be zero.
_HEADER_STRUCT = struct.Struct("<8sHH" + "I" * 17 + "Q" * 7 + "32s")
_CHECKSUM_OFFSET = _HEADER_STRUCT.size - 32


class MultiSetAdditiveQuantizationError(ValueError):
    """Raised when an encoding or serialized artifact is invalid."""


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise MultiSetAdditiveQuantizationError(
            f"{name} must be a positive integer"
        )
    return int(value)


def _seed(value: object) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        or int(value) > (1 << 64) - 1
    ):
        raise MultiSetAdditiveQuantizationError(
            "seed must be an unsigned 64-bit integer"
        )
    return int(value)


def _matrix(values: ArrayLike) -> NDArray[np.float32]:
    try:
        result = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise MultiSetAdditiveQuantizationError("matrix must be numeric") from exc
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise MultiSetAdditiveQuantizationError(
            "matrix must be a non-empty rank-2 array"
        )
    if not np.all(np.isfinite(result)):
        raise MultiSetAdditiveQuantizationError(
            "matrix must contain only finite values"
        )
    return np.ascontiguousarray(result)


def _pack_fixed_width(
    codes: NDArray[np.integer[Any]], bits: int
) -> NDArray[np.uint8]:
    """Pack unsigned codes consecutively, least-significant bit first."""

    flat = np.asarray(codes).reshape(-1).astype(np.uint32, copy=False)
    if flat.size and int(np.max(flat)) >= 1 << bits:
        raise MultiSetAdditiveQuantizationError(
            "a code exceeds its packed bit width"
        )
    packed = np.zeros((flat.size * bits + 7) // 8, dtype=np.uint8)
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
    source = np.asarray(packed)
    expected = (count * bits + 7) // 8
    if source.dtype != np.dtype(np.uint8) or source.ndim != 1:
        raise MultiSetAdditiveQuantizationError(
            "packed payload must be a one-dimensional uint8 array"
        )
    if source.size != expected:
        raise MultiSetAdditiveQuantizationError(
            f"packed payload has {source.size} bytes; expected {expected}"
        )
    values = np.zeros(count, dtype=np.uint16)
    starts = np.arange(count, dtype=np.int64) * bits
    for target_bit in range(bits):
        positions = starts + target_bit
        bit_values = (source[positions // 8] >> (positions % 8)) & 1
        values |= bit_values.astype(np.uint16) << target_bit
    padding = source.size * 8 - count * bits
    if padding and source.size:
        used_bits = 8 - padding
        if int(source[-1]) >> used_bits:
            raise MultiSetAdditiveQuantizationError(
                "packed payload has non-zero trailing padding bits"
            )
    return values


@dataclass(frozen=True)
class MultiSetAdditiveMetadata:
    """Shape and codec fields represented by the fixed artifact header."""

    shape: tuple[int, int]
    group_size: int
    groups: int
    padded_width: int
    num_stages: int
    codebook_size: int
    code_bits: int
    projection_count: int
    rows_per_projection: int
    position_buckets: int
    num_codebook_sets: int
    iterations: int
    sample_limit: int | None
    seed: int
    scale_bits: int
    scale_codebook_size: int
    explicit_set_mapping: bool
    profile: str
    schema_version: int = MULTISET_ADDITIVE_SCHEMA_VERSION
    header_bytes: int = MULTISET_ADDITIVE_HEADER_BYTES
    codec: str = "multiset_additive_residual"
    stack_order: str = "gate_up_down_transpose"
    grouping: str = "contiguous_zero_padded"
    packed_order: str = "least_significant_bit_first"
    codebook_dtype: str = "float16"
    decoded_dtype: str = "float32"

    def validate(self) -> None:
        if self.schema_version != MULTISET_ADDITIVE_SCHEMA_VERSION:
            raise MultiSetAdditiveQuantizationError(
                f"unsupported metadata version {self.schema_version}"
            )
        if self.header_bytes != MULTISET_ADDITIVE_HEADER_BYTES:
            raise MultiSetAdditiveQuantizationError(
                f"header_bytes must be {MULTISET_ADDITIVE_HEADER_BYTES}"
            )
        expected_strings = {
            "codec": "multiset_additive_residual",
            "stack_order": "gate_up_down_transpose",
            "grouping": "contiguous_zero_padded",
            "packed_order": "least_significant_bit_first",
            "codebook_dtype": "float16",
            "decoded_dtype": "float32",
        }
        for name, expected in expected_strings.items():
            if getattr(self, name) != expected:
                raise MultiSetAdditiveQuantizationError(
                    f"unsupported {name} {getattr(self, name)!r}"
                )
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                or value > (1 << 32) - 1
                for value in self.shape
            )
        ):
            raise MultiSetAdditiveQuantizationError(
                "shape must contain two positive uint32 integers"
            )
        group = _positive_integer(self.group_size, "group_size")
        expected_groups = math.ceil(self.shape[1] / group)
        if self.groups != expected_groups:
            raise MultiSetAdditiveQuantizationError(
                f"groups must equal ceil(width / group_size), expected {expected_groups}"
            )
        if self.padded_width != self.groups * group:
            raise MultiSetAdditiveQuantizationError(
                "padded_width must equal groups * group_size"
            )
        projections = _positive_integer(self.projection_count, "projection_count")
        if self.shape[0] % projections:
            raise MultiSetAdditiveQuantizationError(
                "record count must be divisible by projection_count"
            )
        if self.rows_per_projection != self.shape[0] // projections:
            raise MultiSetAdditiveQuantizationError(
                "rows_per_projection disagrees with shape and projection_count"
            )
        buckets = _positive_integer(self.position_buckets, "position_buckets")
        if buckets > self.groups or self.groups % buckets:
            raise MultiSetAdditiveQuantizationError(
                "position_buckets must divide the number of groups"
            )
        if self.num_codebook_sets != projections * buckets:
            raise MultiSetAdditiveQuantizationError(
                "num_codebook_sets must equal projection_count * position_buckets"
            )
        if self.num_codebook_sets > 256:
            raise MultiSetAdditiveQuantizationError(
                "schema version 1 supports at most 256 codebook sets"
            )
        _positive_integer(self.iterations, "iterations")
        if self.sample_limit is not None:
            _positive_integer(self.sample_limit, "sample_limit")
        _seed(self.seed)
        if not isinstance(self.explicit_set_mapping, bool):
            raise MultiSetAdditiveQuantizationError(
                "explicit_set_mapping must be a boolean"
            )

        if self.profile == PROJECTION_2X7_SCALE6:
            valid = (
                self.num_stages == 2
                and self.codebook_size == 128
                and self.code_bits == 7
                and self.position_buckets == 1
                and self.scale_bits == 6
                and self.scale_codebook_size == 64
                and not self.explicit_set_mapping
            )
            if not valid:
                raise MultiSetAdditiveQuantizationError(
                    "projection_2x7_scale6 profile fields are inconsistent"
                )
        elif self.profile == POSITION_3X4_FP16_SCALE:
            valid = (
                self.num_stages == 3
                and self.codebook_size == 16
                and self.code_bits == 4
                and self.scale_bits == 0
                and self.scale_codebook_size == 0
                and self.explicit_set_mapping
            )
            if not valid:
                raise MultiSetAdditiveQuantizationError(
                    "position_3x4_fp16_scale profile fields are inconsistent"
                )
        else:
            raise MultiSetAdditiveQuantizationError(
                f"unsupported codec profile {self.profile!r}"
            )

    @property
    def quantized_scales(self) -> bool:
        return self.scale_bits > 0

    @property
    def code_count(self) -> int:
        return self.shape[0] * self.groups * self.num_stages

    @property
    def packed_code_bytes(self) -> int:
        return (self.code_count * self.code_bits + 7) // 8

    @property
    def codebook_bytes(self) -> int:
        return (
            self.num_codebook_sets
            * self.num_stages
            * self.codebook_size
            * self.group_size
            * np.dtype(np.float16).itemsize
        )

    @property
    def scale_payload_bytes(self) -> int:
        if self.quantized_scales:
            return (self.shape[0] * self.scale_bits + 7) // 8
        return self.shape[0] * np.dtype(np.float16).itemsize

    @property
    def scale_codebook_bytes(self) -> int:
        if not self.quantized_scales:
            return 0
        return (
            self.projection_count
            * self.scale_codebook_size
            * np.dtype(np.float16).itemsize
        )

    @property
    def mapping_bytes(self) -> int:
        if not self.explicit_set_mapping:
            return 0
        return self.projection_count * self.groups * np.dtype(np.uint8).itemsize

    @property
    def storage_bytes(self) -> int:
        return (
            self.header_bytes
            + self.packed_code_bytes
            + self.codebook_bytes
            + self.scale_payload_bytes
            + self.scale_codebook_bytes
            + self.mapping_bytes
        )

    @property
    def dense_q4_bytes(self) -> int:
        return (self.shape[0] * self.shape[1] * 4 + 7) // 8

    @property
    def fraction_of_dense_q4(self) -> float:
        return self.storage_bytes / self.dense_q4_bytes

    @property
    def effective_bits_per_weight(self) -> float:
        return self.storage_bytes * 8 / (self.shape[0] * self.shape[1])

    def storage_components(self) -> dict[str, int]:
        return {
            "header_and_checksum": self.header_bytes,
            "packed_codes": self.packed_code_bytes,
            "codebooks": self.codebook_bytes,
            "packed_scale_indices" if self.quantized_scales else "row_scales": (
                self.scale_payload_bytes
            ),
            "scale_codebooks": self.scale_codebook_bytes,
            "set_mapping": self.mapping_bytes,
            "total": self.storage_bytes,
        }


def make_multiset_additive_metadata(
    shape: tuple[int, int],
    *,
    profile: str = PROJECTION_2X7_SCALE6,
    group_size: int = PRIMARY_GROUP_SIZE,
    projection_count: int = PRIMARY_PROJECTION_COUNT,
    position_buckets: int | None = None,
    iterations: int = 8,
    sample_limit: int | None = 4096,
    seed: int = 0,
) -> MultiSetAdditiveMetadata:
    """Construct a validated primary or fallback layout."""

    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
    ):
        raise MultiSetAdditiveQuantizationError(
            "shape must be a tuple of two integers"
        )
    group = _positive_integer(group_size, "group_size")
    projections = _positive_integer(projection_count, "projection_count")
    rounds = _positive_integer(iterations, "iterations")
    limit = (
        None if sample_limit is None else _positive_integer(sample_limit, "sample_limit")
    )
    stable_seed = _seed(seed)
    if shape[0] <= 0 or shape[1] <= 0:
        raise MultiSetAdditiveQuantizationError("shape dimensions must be positive")
    groups = math.ceil(shape[1] / group)
    if profile == PROJECTION_2X7_SCALE6:
        buckets = 1 if position_buckets is None else position_buckets
        num_stages, entries, bits = 2, 128, 7
        scale_bits, scale_entries, explicit_mapping = 6, 64, False
    elif profile == POSITION_3X4_FP16_SCALE:
        buckets = FALLBACK_POSITION_BUCKETS if position_buckets is None else position_buckets
        num_stages, entries, bits = 3, 16, 4
        scale_bits, scale_entries, explicit_mapping = 0, 0, True
    else:
        raise MultiSetAdditiveQuantizationError(
            f"unsupported codec profile {profile!r}"
        )
    buckets = _positive_integer(buckets, "position_buckets")
    metadata = MultiSetAdditiveMetadata(
        shape=shape,
        group_size=group,
        groups=groups,
        padded_width=groups * group,
        num_stages=num_stages,
        codebook_size=entries,
        code_bits=bits,
        projection_count=projections,
        rows_per_projection=shape[0] // projections,
        position_buckets=buckets,
        num_codebook_sets=projections * buckets,
        iterations=rounds,
        sample_limit=limit,
        seed=stable_seed,
        scale_bits=scale_bits,
        scale_codebook_size=scale_entries,
        explicit_set_mapping=explicit_mapping,
        profile=profile,
    )
    metadata.validate()
    return metadata


def multiset_storage_plan(
    *, profile: str = PROJECTION_2X7_SCALE6
) -> dict[str, int | float | str | tuple[int, int]]:
    """Exact cold-byte plan for the SmolLM2-135M ``[4608, 576]`` stack."""

    metadata = make_multiset_additive_metadata((4608, 576), profile=profile)
    return {
        "profile": metadata.profile,
        "shape": metadata.shape,
        "group_size": metadata.group_size,
        "num_stages": metadata.num_stages,
        "code_bits": metadata.code_bits,
        "projection_count": metadata.projection_count,
        "position_buckets": metadata.position_buckets,
        "num_codebook_sets": metadata.num_codebook_sets,
        **metadata.storage_components(),
        "dense_q4_bytes": metadata.dense_q4_bytes,
        "fraction_of_dense_q4": metadata.fraction_of_dense_q4,
        "effective_bits_per_weight": metadata.effective_bits_per_weight,
    }


def _expected_mapping(metadata: MultiSetAdditiveMetadata) -> NDArray[np.uint8]:
    groups_per_bucket = metadata.groups // metadata.position_buckets
    group_buckets = np.arange(metadata.groups, dtype=np.int64) // groups_per_bucket
    projections = np.arange(metadata.projection_count, dtype=np.int64)[:, None]
    mapping = projections * metadata.position_buckets + group_buckets[None, :]
    return np.ascontiguousarray(mapping, dtype=np.uint8)


@dataclass(frozen=True)
class MultiSetAdditiveEncoding:
    """All learned/persisted arrays for one additive artifact."""

    packed_codes: NDArray[np.uint8]
    codebooks: NDArray[np.float16]
    metadata: MultiSetAdditiveMetadata
    row_scales: NDArray[np.float16] | None = None
    packed_scale_indices: NDArray[np.uint8] | None = None
    scale_codebooks: NDArray[np.float16] | None = None
    set_mapping: NDArray[np.uint8] | None = None

    @property
    def storage_bytes(self) -> int:
        return self.metadata.storage_bytes

    @property
    def fraction_of_dense_q4(self) -> float:
        return self.metadata.fraction_of_dense_q4

    def storage_components(self) -> dict[str, int]:
        return self.metadata.storage_components()

    def _validate_packed(self, values: object, expected_bytes: int, name: str) -> NDArray[np.uint8]:
        array = np.asarray(values)
        if (
            array.dtype != np.dtype(np.uint8)
            or array.ndim != 1
            or not array.flags.c_contiguous
        ):
            raise MultiSetAdditiveQuantizationError(
                f"{name} must be a contiguous one-dimensional uint8 array"
            )
        if array.nbytes != expected_bytes:
            raise MultiSetAdditiveQuantizationError(
                f"{name} byte length disagrees with metadata"
            )
        return array

    def validate(self) -> None:
        metadata = self.metadata
        metadata.validate()
        packed_codes = self._validate_packed(
            self.packed_codes, metadata.packed_code_bytes, "packed_codes"
        )
        codes = _unpack_fixed_width(
            packed_codes, metadata.code_count, metadata.code_bits
        )
        if codes.size and int(np.max(codes)) >= metadata.codebook_size:
            raise MultiSetAdditiveQuantizationError(
                "a packed code is outside its codebook"
            )
        expected_codebook_shape = (
            metadata.num_codebook_sets,
            metadata.num_stages,
            metadata.codebook_size,
            metadata.group_size,
        )
        if (
            self.codebooks.dtype != np.dtype(np.float16)
            or self.codebooks.shape != expected_codebook_shape
            or not self.codebooks.flags.c_contiguous
        ):
            raise MultiSetAdditiveQuantizationError(
                "codebooks must be contiguous float16 with shape "
                f"{expected_codebook_shape}"
            )
        if not np.all(np.isfinite(self.codebooks)):
            raise MultiSetAdditiveQuantizationError(
                "codebooks must contain only finite values"
            )
        if metadata.quantized_scales:
            if self.row_scales is not None or self.set_mapping is not None:
                raise MultiSetAdditiveQuantizationError(
                    "quantized-scale profile cannot contain raw scales or a set mapping"
                )
            if self.packed_scale_indices is None:
                raise MultiSetAdditiveQuantizationError(
                    "packed_scale_indices are required"
                )
            packed_scales = self._validate_packed(
                self.packed_scale_indices,
                metadata.scale_payload_bytes,
                "packed_scale_indices",
            )
            scale_indices = _unpack_fixed_width(
                packed_scales, metadata.shape[0], metadata.scale_bits
            )
            if scale_indices.size and int(np.max(scale_indices)) >= metadata.scale_codebook_size:
                raise MultiSetAdditiveQuantizationError(
                    "a scale index is outside its codebook"
                )
            expected_scale_shape = (
                metadata.projection_count,
                metadata.scale_codebook_size,
            )
            if (
                self.scale_codebooks is None
                or self.scale_codebooks.dtype != np.dtype(np.float16)
                or self.scale_codebooks.shape != expected_scale_shape
                or not self.scale_codebooks.flags.c_contiguous
            ):
                raise MultiSetAdditiveQuantizationError(
                    "scale_codebooks must be contiguous float16 with shape "
                    f"{expected_scale_shape}"
                )
            if not np.all(np.isfinite(self.scale_codebooks)) or np.any(
                self.scale_codebooks < 0
            ):
                raise MultiSetAdditiveQuantizationError(
                    "scale_codebooks must be finite and non-negative"
                )
        else:
            if self.packed_scale_indices is not None or self.scale_codebooks is not None:
                raise MultiSetAdditiveQuantizationError(
                    "raw-scale profile cannot contain quantized scale arrays"
                )
            if (
                self.row_scales is None
                or self.row_scales.dtype != np.dtype(np.float16)
                or self.row_scales.shape != (metadata.shape[0],)
                or not self.row_scales.flags.c_contiguous
            ):
                raise MultiSetAdditiveQuantizationError(
                    f"row_scales must be contiguous float16 with shape ({metadata.shape[0]},)"
                )
            if not np.all(np.isfinite(self.row_scales)) or np.any(self.row_scales < 0):
                raise MultiSetAdditiveQuantizationError(
                    "row_scales must be finite and non-negative"
                )
            expected_mapping = _expected_mapping(metadata)
            if (
                self.set_mapping is None
                or self.set_mapping.dtype != np.dtype(np.uint8)
                or self.set_mapping.shape != expected_mapping.shape
                or not self.set_mapping.flags.c_contiguous
            ):
                raise MultiSetAdditiveQuantizationError(
                    "set_mapping must be contiguous uint8 with shape "
                    f"{expected_mapping.shape}"
                )
            if not np.array_equal(self.set_mapping, expected_mapping):
                raise MultiSetAdditiveQuantizationError(
                    "set_mapping disagrees with projection/position metadata"
                )

    def unpack_codes(self) -> NDArray[np.uint16]:
        self.validate()
        return _unpack_fixed_width(
            self.packed_codes, self.metadata.code_count, self.metadata.code_bits
        ).reshape(
            self.metadata.shape[0], self.metadata.num_stages, self.metadata.groups
        )

    def decoded_scales(self) -> NDArray[np.float32]:
        self.validate()
        metadata = self.metadata
        if not metadata.quantized_scales:
            assert self.row_scales is not None
            return self.row_scales.astype(np.float32)
        assert self.packed_scale_indices is not None
        assert self.scale_codebooks is not None
        indices = _unpack_fixed_width(
            self.packed_scale_indices, metadata.shape[0], metadata.scale_bits
        )
        projections = (
            np.arange(metadata.shape[0], dtype=np.int64)
            // metadata.rows_per_projection
        )
        return self.scale_codebooks[projections, indices].astype(np.float32)


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
    if values.shape[0] == 0:
        raise MultiSetAdditiveQuantizationError(
            "every codebook set must receive at least one sample"
        )
    rng = np.random.default_rng(seed)
    if sample_limit is not None and values.shape[0] > sample_limit:
        ids = np.sort(rng.choice(values.shape[0], size=sample_limit, replace=False))
        samples = values[ids]
    else:
        samples = values
    if samples.shape[0] >= codebook_size:
        initial = rng.choice(samples.shape[0], size=codebook_size, replace=False)
    else:
        initial = np.resize(rng.permutation(samples.shape[0]), codebook_size)
    centers = samples[initial].copy()
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


def _stored_fp16(values: NDArray[np.float32], name: str) -> NDArray[np.float16]:
    with np.errstate(over="ignore", invalid="ignore"):
        stored = values.astype(np.float16)
    if not np.all(np.isfinite(stored)):
        raise MultiSetAdditiveQuantizationError(
            f"a {name} value is outside the finite float16 range"
        )
    return np.ascontiguousarray(stored)


def _fit_scales(
    exact_scales: NDArray[np.float32], metadata: MultiSetAdditiveMetadata
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float16] | None,
    NDArray[np.uint8] | None,
    NDArray[np.float16] | None,
]:
    if not metadata.quantized_scales:
        row_scales = _stored_fp16(exact_scales, "row scale")
        # Preserve nonzero subnormal rows when representable in float16.
        underflow = (exact_scales > 0) & (row_scales == 0)
        if np.any(underflow):
            row_scales[underflow] = np.nextafter(
                np.float16(0), np.float16(1), dtype=np.float16
            )
        return row_scales.astype(np.float32), row_scales, None, None

    scale_codebooks = np.empty(
        (metadata.projection_count, metadata.scale_codebook_size), dtype=np.float16
    )
    indices = np.empty(metadata.shape[0], dtype=np.uint8)
    smallest_scale = np.nextafter(
        np.float16(0), np.float16(1), dtype=np.float16
    ).astype(np.float32)
    for projection in range(metadata.projection_count):
        start = projection * metadata.rows_per_projection
        stop = start + metadata.rows_per_projection
        values = exact_scales[start:stop]
        positive = values > 0
        # Weight-row magnitudes are multiplicative.  Fitting log2(scale) and
        # assigning in that same domain minimizes relative rather than absolute
        # scale error, so small rows are not sacrificed to the largest rows.
        log_values = np.log2(np.maximum(values, smallest_scale))
        log_centers = _fit_codebook(
            log_values[:, None],
            codebook_size=metadata.scale_codebook_size,
            iterations=metadata.iterations,
            sample_limit=metadata.sample_limit,
            seed=int(
                np.random.SeedSequence([metadata.seed, 0x5343414C, projection])
                .generate_state(1, dtype=np.uint32)[0]
            ),
        )
        stored = _stored_fp16(np.exp2(log_centers[:, 0]), "scale codebook")
        zero_id: int | None = None
        if np.any(~positive):
            zero_id = int(np.argmin(stored))
            stored[zero_id] = np.float16(0)
        scale_codebooks[projection] = stored
        stored_logs = np.log2(
            np.maximum(stored.astype(np.float32), smallest_scale)
        )
        assignments = np.argmin(
            np.square(log_values[:, None] - stored_logs[None, :]), axis=1
        ).astype(np.uint8)
        if zero_id is not None:
            assignments[~positive] = zero_id
        indices[start:stop] = assignments
    projections = (
        np.arange(metadata.shape[0], dtype=np.int64) // metadata.rows_per_projection
    )
    decoded = scale_codebooks[projections, indices].astype(np.float32)
    return decoded, None, _pack_fixed_width(indices, metadata.scale_bits), scale_codebooks


def fit_multiset_additive(
    matrix: ArrayLike,
    *,
    profile: str = PROJECTION_2X7_SCALE6,
    group_size: int = PRIMARY_GROUP_SIZE,
    projection_count: int = PRIMARY_PROJECTION_COUNT,
    position_buckets: int | None = None,
    iterations: int = 8,
    sample_limit: int | None = 4096,
    seed: int = 0,
) -> MultiSetAdditiveEncoding:
    """Fit deterministic residual codebooks and return packed deployment data."""

    values = _matrix(matrix)
    metadata = make_multiset_additive_metadata(
        (values.shape[0], values.shape[1]),
        profile=profile,
        group_size=group_size,
        projection_count=projection_count,
        position_buckets=position_buckets,
        iterations=iterations,
        sample_limit=sample_limit,
        seed=seed,
    )
    records, width = metadata.shape
    exact_scales = np.max(np.abs(values), axis=1)
    decoded_scales, row_scales, packed_scale_indices, scale_codebooks = _fit_scales(
        exact_scales, metadata
    )
    divisors = np.where(decoded_scales > 0, decoded_scales, 1.0)
    working = np.zeros((records, metadata.padded_width), dtype=np.float32)
    working[:, :width] = values / divisors[:, None]
    residual = working.reshape(records, metadata.groups, metadata.group_size).copy()

    derived_mapping = _expected_mapping(metadata)
    row_projection = (
        np.arange(records, dtype=np.int64) // metadata.rows_per_projection
    )
    set_ids = derived_mapping[row_projection]
    codebooks = np.empty(
        (
            metadata.num_codebook_sets,
            metadata.num_stages,
            metadata.codebook_size,
            metadata.group_size,
        ),
        dtype=np.float16,
    )
    codes = np.empty(
        (records, metadata.num_stages, metadata.groups), dtype=np.uint16
    )
    positions = [
        np.nonzero(set_ids == set_id)
        for set_id in range(metadata.num_codebook_sets)
    ]
    for stage in range(metadata.num_stages):
        for set_id, (row_ids, group_ids) in enumerate(positions):
            vectors = np.ascontiguousarray(residual[row_ids, group_ids])
            centers = _fit_codebook(
                vectors,
                codebook_size=metadata.codebook_size,
                iterations=metadata.iterations,
                sample_limit=metadata.sample_limit,
                seed=int(
                    np.random.SeedSequence([metadata.seed, stage, set_id])
                    .generate_state(1, dtype=np.uint32)[0]
                ),
            )
            stored = _stored_fp16(centers, "codebook")
            reconstruction = stored.astype(np.float32)
            assignments = np.argmin(
                _squared_distances(vectors, reconstruction), axis=1
            ).astype(np.uint16)
            codebooks[set_id, stage] = stored
            codes[row_ids, stage, group_ids] = assignments
            residual[row_ids, group_ids] = vectors - reconstruction[assignments]

    encoding = MultiSetAdditiveEncoding(
        packed_codes=_pack_fixed_width(codes, metadata.code_bits),
        codebooks=np.ascontiguousarray(codebooks),
        metadata=metadata,
        row_scales=row_scales,
        packed_scale_indices=packed_scale_indices,
        scale_codebooks=scale_codebooks,
        set_mapping=derived_mapping if metadata.explicit_set_mapping else None,
    )
    encoding.validate()
    return encoding


def decode_multiset_additive(
    encoding: MultiSetAdditiveEncoding,
) -> NDArray[np.float32]:
    """Decode the joint stack to a contiguous float32 matrix."""

    encoding.validate()
    metadata = encoding.metadata
    codes = encoding.unpack_codes()
    mapping = (
        _expected_mapping(metadata)
        if encoding.set_mapping is None
        else encoding.set_mapping
    )
    row_projection = (
        np.arange(metadata.shape[0], dtype=np.int64)
        // metadata.rows_per_projection
    )
    set_ids = mapping[row_projection]
    decoded = np.zeros(
        (metadata.shape[0], metadata.groups, metadata.group_size), dtype=np.float32
    )
    for stage in range(metadata.num_stages):
        decoded += encoding.codebooks[
            set_ids, stage, codes[:, stage, :], :
        ].astype(np.float32)
    result = decoded.reshape(metadata.shape[0], metadata.padded_width)[
        :, : metadata.shape[1]
    ]
    result *= encoding.decoded_scales()[:, None]
    return np.ascontiguousarray(result)


def _payload(encoding: MultiSetAdditiveEncoding) -> bytes:
    sections = [
        encoding.packed_codes.tobytes(order="C"),
        encoding.codebooks.astype("<f2", copy=False).tobytes(order="C"),
    ]
    if encoding.metadata.quantized_scales:
        assert encoding.packed_scale_indices is not None
        assert encoding.scale_codebooks is not None
        sections.extend(
            (
                encoding.packed_scale_indices.tobytes(order="C"),
                encoding.scale_codebooks.astype("<f2", copy=False).tobytes(order="C"),
            )
        )
    else:
        assert encoding.row_scales is not None
        sections.extend(
            (encoding.row_scales.astype("<f2", copy=False).tobytes(order="C"), b"")
        )
    if encoding.set_mapping is not None:
        sections.append(encoding.set_mapping.tobytes(order="C"))
    return b"".join(sections)


def _flags(metadata: MultiSetAdditiveMetadata) -> int:
    result = (
        _FLAG_QUANTIZED_SCALES
        if metadata.quantized_scales
        else _FLAG_RAW_FP16_SCALES
    )
    if metadata.explicit_set_mapping:
        result |= _FLAG_EXPLICIT_SET_MAPPING
    return result


def _profile_from_fields(
    *, num_stages: int, code_bits: int, scale_bits: int
) -> str:
    if (num_stages, code_bits, scale_bits) == (2, 7, 6):
        return PROJECTION_2X7_SCALE6
    if (num_stages, code_bits, scale_bits) == (3, 4, 0):
        return POSITION_3X4_FP16_SCALE
    raise MultiSetAdditiveQuantizationError(
        "artifact fields do not identify a supported profile"
    )


def _header(
    metadata: MultiSetAdditiveMetadata, *, payload_bytes: int, checksum: bytes
) -> bytes:
    scale_payload_bytes = metadata.scale_payload_bytes
    prefix = _HEADER_STRUCT.pack(
        MULTISET_ADDITIVE_MAGIC,
        metadata.schema_version,
        metadata.header_bytes,
        _flags(metadata),
        metadata.shape[0],
        metadata.shape[1],
        metadata.group_size,
        metadata.groups,
        metadata.padded_width,
        metadata.num_stages,
        metadata.codebook_size,
        metadata.code_bits,
        metadata.projection_count,
        metadata.rows_per_projection,
        metadata.position_buckets,
        metadata.num_codebook_sets,
        metadata.iterations,
        0 if metadata.sample_limit is None else metadata.sample_limit,
        metadata.scale_bits,
        metadata.scale_codebook_size,
        metadata.seed,
        metadata.packed_code_bytes,
        metadata.codebook_bytes,
        scale_payload_bytes,
        metadata.scale_codebook_bytes,
        metadata.mapping_bytes,
        payload_bytes,
        checksum,
    )
    return prefix + bytes(MULTISET_ADDITIVE_HEADER_BYTES - len(prefix))


def serialize_multiset_additive(encoding: MultiSetAdditiveEncoding) -> bytes:
    """Serialize and checksum a complete cold artifact."""

    encoding.validate()
    payload = _payload(encoding)
    expected = encoding.storage_bytes - MULTISET_ADDITIVE_HEADER_BYTES
    if len(payload) != expected:
        raise MultiSetAdditiveQuantizationError(
            "serialized payload disagrees with byte accounting"
        )
    zero_header = _header(
        encoding.metadata, payload_bytes=len(payload), checksum=bytes(32)
    )
    checksum = hashlib.sha256(zero_header + payload).digest()
    artifact = _header(
        encoding.metadata, payload_bytes=len(payload), checksum=checksum
    ) + payload
    if len(artifact) != encoding.storage_bytes:
        raise MultiSetAdditiveQuantizationError(
            "serialized artifact disagrees with total-byte accounting"
        )
    return artifact


def deserialize_multiset_additive(
    data: bytes | bytearray | memoryview,
) -> MultiSetAdditiveEncoding:
    """Checksum and strictly decode a schema-v1 artifact."""

    try:
        artifact = bytes(data)
    except (TypeError, ValueError) as exc:
        raise MultiSetAdditiveQuantizationError("artifact must be bytes-like") from exc
    if len(artifact) < MULTISET_ADDITIVE_HEADER_BYTES:
        raise MultiSetAdditiveQuantizationError("artifact is shorter than its header")
    header = artifact[:MULTISET_ADDITIVE_HEADER_BYTES]
    fields = _HEADER_STRUCT.unpack(header[: _HEADER_STRUCT.size])
    (
        magic,
        schema_version,
        header_bytes,
        flags,
        records,
        width,
        group_size,
        groups,
        padded_width,
        num_stages,
        codebook_size,
        code_bits,
        projection_count,
        rows_per_projection,
        position_buckets,
        num_codebook_sets,
        iterations,
        stored_sample_limit,
        scale_bits,
        scale_codebook_size,
        seed,
        packed_code_bytes,
        codebook_bytes,
        scale_payload_bytes,
        scale_codebook_bytes,
        mapping_bytes,
        payload_bytes,
        checksum,
    ) = fields
    if magic != MULTISET_ADDITIVE_MAGIC:
        raise MultiSetAdditiveQuantizationError("artifact magic is invalid")
    if header_bytes != MULTISET_ADDITIVE_HEADER_BYTES:
        raise MultiSetAdditiveQuantizationError("artifact header size is invalid")
    allowed_flags = {
        _FLAG_QUANTIZED_SCALES,
        _FLAG_RAW_FP16_SCALES | _FLAG_EXPLICIT_SET_MAPPING,
    }
    if flags not in allowed_flags:
        raise MultiSetAdditiveQuantizationError("artifact flags are unsupported")
    if any(header[_HEADER_STRUCT.size :]):
        raise MultiSetAdditiveQuantizationError(
            "artifact reserved header bytes must be zero"
        )
    if len(artifact) != header_bytes + payload_bytes:
        raise MultiSetAdditiveQuantizationError(
            "artifact length disagrees with its header"
        )
    zeroed = bytearray(header)
    zeroed[_CHECKSUM_OFFSET : _CHECKSUM_OFFSET + 32] = bytes(32)
    expected_checksum = hashlib.sha256(bytes(zeroed) + artifact[header_bytes:]).digest()
    if not hmac.compare_digest(checksum, expected_checksum):
        raise MultiSetAdditiveQuantizationError("artifact checksum mismatch")

    profile = _profile_from_fields(
        num_stages=num_stages, code_bits=code_bits, scale_bits=scale_bits
    )
    metadata = MultiSetAdditiveMetadata(
        shape=(records, width),
        group_size=group_size,
        groups=groups,
        padded_width=padded_width,
        num_stages=num_stages,
        codebook_size=codebook_size,
        code_bits=code_bits,
        projection_count=projection_count,
        rows_per_projection=rows_per_projection,
        position_buckets=position_buckets,
        num_codebook_sets=num_codebook_sets,
        iterations=iterations,
        sample_limit=None if stored_sample_limit == 0 else stored_sample_limit,
        seed=seed,
        scale_bits=scale_bits,
        scale_codebook_size=scale_codebook_size,
        explicit_set_mapping=bool(flags & _FLAG_EXPLICIT_SET_MAPPING),
        profile=profile,
        schema_version=schema_version,
        header_bytes=header_bytes,
    )
    metadata.validate()
    expected_sizes = (
        metadata.packed_code_bytes,
        metadata.codebook_bytes,
        metadata.scale_payload_bytes,
        metadata.scale_codebook_bytes,
        metadata.mapping_bytes,
    )
    stored_sizes = (
        packed_code_bytes,
        codebook_bytes,
        scale_payload_bytes,
        scale_codebook_bytes,
        mapping_bytes,
    )
    if stored_sizes != expected_sizes or payload_bytes != sum(expected_sizes):
        raise MultiSetAdditiveQuantizationError(
            "artifact section sizes disagree with metadata"
        )

    payload = memoryview(artifact)[header_bytes:]
    offset = 0

    def section(size: int) -> memoryview:
        nonlocal offset
        value = payload[offset : offset + size]
        offset += size
        return value

    packed_codes = np.frombuffer(section(packed_code_bytes), dtype=np.uint8).copy()
    codebooks = np.frombuffer(section(codebook_bytes), dtype="<f2").astype(
        np.float16, copy=True
    ).reshape(
        metadata.num_codebook_sets,
        metadata.num_stages,
        metadata.codebook_size,
        metadata.group_size,
    )
    scale_payload = section(scale_payload_bytes)
    if metadata.quantized_scales:
        packed_scale_indices = np.frombuffer(scale_payload, dtype=np.uint8).copy()
        row_scales = None
        scale_codebooks = np.frombuffer(
            section(scale_codebook_bytes), dtype="<f2"
        ).astype(np.float16, copy=True).reshape(
            metadata.projection_count, metadata.scale_codebook_size
        )
    else:
        row_scales = np.frombuffer(scale_payload, dtype="<f2").astype(
            np.float16, copy=True
        )
        packed_scale_indices = None
        scale_codebooks = None
        section(scale_codebook_bytes)
    if metadata.explicit_set_mapping:
        set_mapping = np.frombuffer(section(mapping_bytes), dtype=np.uint8).copy().reshape(
            metadata.projection_count, metadata.groups
        )
    else:
        set_mapping = None
        section(mapping_bytes)
    if offset != len(payload):
        raise MultiSetAdditiveQuantizationError(
            "artifact has unconsumed payload bytes"
        )
    encoding = MultiSetAdditiveEncoding(
        packed_codes=np.ascontiguousarray(packed_codes),
        codebooks=np.ascontiguousarray(codebooks),
        metadata=metadata,
        row_scales=None if row_scales is None else np.ascontiguousarray(row_scales),
        packed_scale_indices=(
            None
            if packed_scale_indices is None
            else np.ascontiguousarray(packed_scale_indices)
        ),
        scale_codebooks=(
            None if scale_codebooks is None else np.ascontiguousarray(scale_codebooks)
        ),
        set_mapping=None if set_mapping is None else np.ascontiguousarray(set_mapping),
    )
    encoding.validate()
    return encoding


def save_multiset_additive(
    path: str | Path, encoding: MultiSetAdditiveEncoding
) -> str:
    """Atomically save an artifact and return its embedded checksum."""

    destination = Path(path)
    artifact = serialize_multiset_additive(encoding)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(artifact)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return artifact[_CHECKSUM_OFFSET : _CHECKSUM_OFFSET + 32].hex()


def load_multiset_additive(path: str | Path) -> MultiSetAdditiveEncoding:
    """Load, checksum, and validate a serialized artifact."""

    return deserialize_multiset_additive(Path(path).read_bytes())


__all__ = [
    "MULTISET_ADDITIVE_HEADER_BYTES",
    "MULTISET_ADDITIVE_SCHEMA_VERSION",
    "POSITION_3X4_FP16_SCALE",
    "PROJECTION_2X7_SCALE6",
    "MultiSetAdditiveEncoding",
    "MultiSetAdditiveMetadata",
    "MultiSetAdditiveQuantizationError",
    "decode_multiset_additive",
    "deserialize_multiset_additive",
    "fit_multiset_additive",
    "load_multiset_additive",
    "make_multiset_additive_metadata",
    "multiset_storage_plan",
    "save_multiset_additive",
    "serialize_multiset_additive",
]
