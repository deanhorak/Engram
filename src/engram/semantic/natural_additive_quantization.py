"""Strict additive quantization for naturally oriented SwiGLU matrices.

This codec deliberately does *not* form the convenient ``[gate; up; down.T]``
training stack.  It stores the three linear matrices in the orientation used by
the linear operators and groups each matrix along its real input dimension::

    gate: [intermediate, hidden]
    up:   [intermediate, hidden]
    down: [hidden, intermediate]

That distinction matters for activation-aware optimization: a group of eight
values always multiplies the same eight input activations.  Schema version 1
uses two matrix-local 7-bit residual codebooks, float16 centroids, and a
matrix-local 6-bit log-clustered scale per output row.  All arrays are packed
into a fixed-header, SHA-256-checksummed artifact.  The builder API accepts
updated assignments and centroids so AQLM beam search or PV-style refinement
can reuse the wire format without changing its byte accounting.
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
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


NATURAL_ADDITIVE_SCHEMA_VERSION = 1
NATURAL_ADDITIVE_HEADER_BYTES = 256
NATURAL_ADDITIVE_MAGIC = b"ENNAAQ01"

NATURAL_MATRIX_NAMES = ("gate", "up", "down")
NATURAL_MATRIX_COUNT = 3
NATURAL_GROUP_SIZE = 8
NATURAL_NUM_STAGES = 2
NATURAL_CODEBOOK_SIZE = 128
NATURAL_CODE_BITS = 7
NATURAL_SCALE_BITS = 6
NATURAL_SCALE_CODEBOOK_SIZE = 64

_FLAGS = 1  # matrix-local log2 row-scale codebooks

# magic, schema/header, 19 uint32 fields, eight uint64 fields, checksum.
# Remaining bytes in the fixed-size header are required to be zero.
_HEADER_STRUCT = struct.Struct("<8sHH" + "I" * 19 + "Q" * 8 + "32s")
_CHECKSUM_OFFSET = _HEADER_STRUCT.size - 32


class NaturalAdditiveQuantizationError(ValueError):
    """Raised when natural additive input or an artifact is malformed."""


def _positive_integer(value: object, name: str, *, maximum: int | None = None) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise NaturalAdditiveQuantizationError(f"{name} must be a positive integer")
    result = int(value)
    if maximum is not None and result > maximum:
        raise NaturalAdditiveQuantizationError(f"{name} exceeds {maximum}")
    return result


def _seed(value: object) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        or int(value) > (1 << 64) - 1
    ):
        raise NaturalAdditiveQuantizationError(
            "seed must be an unsigned 64-bit integer"
        )
    return int(value)


def _shape(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise NaturalAdditiveQuantizationError(
            f"{name} must be a tuple of two positive integers"
        )
    result: list[int] = []
    for dimension in value:
        result.append(_positive_integer(dimension, name, maximum=(1 << 32) - 1))
    return result[0], result[1]


def _matrix(values: ArrayLike, name: str) -> NDArray[np.float32]:
    try:
        result = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise NaturalAdditiveQuantizationError(f"{name} must be numeric") from exc
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise NaturalAdditiveQuantizationError(
            f"{name} must be a non-empty rank-2 array"
        )
    if not np.all(np.isfinite(result)):
        raise NaturalAdditiveQuantizationError(
            f"{name} must contain only finite values"
        )
    return np.ascontiguousarray(result)


def _pack_fixed_width(
    values: NDArray[np.integer[Any]], bits: int
) -> NDArray[np.uint8]:
    """Pack unsigned integers consecutively, least-significant bit first."""

    flat = np.asarray(values).reshape(-1).astype(np.uint32, copy=False)
    if flat.size and int(np.max(flat)) >= 1 << bits:
        raise NaturalAdditiveQuantizationError(
            "a value exceeds its packed bit width"
        )
    packed = np.zeros((flat.size * bits + 7) // 8, dtype=np.uint8)
    if not flat.size:
        return packed
    starts = np.arange(flat.size, dtype=np.int64) * bits
    for source_bit in range(bits):
        positions = starts + source_bit
        source = ((flat >> source_bit) & 1).astype(np.uint8)
        np.bitwise_or.at(
            packed,
            positions // 8,
            source << (positions % 8).astype(np.uint8),
        )
    return packed


def _unpack_fixed_width(
    packed: NDArray[np.uint8], count: int, bits: int, name: str
) -> NDArray[np.uint16]:
    source = np.asarray(packed)
    expected = (count * bits + 7) // 8
    if (
        source.dtype != np.dtype(np.uint8)
        or source.ndim != 1
        or not source.flags.c_contiguous
    ):
        raise NaturalAdditiveQuantizationError(
            f"{name} must be a contiguous one-dimensional uint8 array"
        )
    if source.nbytes != expected:
        raise NaturalAdditiveQuantizationError(
            f"{name} has {source.nbytes} bytes; expected {expected}"
        )
    result = np.zeros(count, dtype=np.uint16)
    if count:
        starts = np.arange(count, dtype=np.int64) * bits
        for target_bit in range(bits):
            positions = starts + target_bit
            source_bits = (source[positions // 8] >> (positions % 8)) & 1
            result |= source_bits.astype(np.uint16) << target_bit
    padding = source.size * 8 - count * bits
    if padding and source.size:
        used_bits = 8 - padding
        if int(source[-1]) >> used_bits:
            raise NaturalAdditiveQuantizationError(
                f"{name} has non-zero trailing padding bits"
            )
    return result


@dataclass(frozen=True)
class NaturalAdditiveMetadata:
    """Shape and fixed-profile fields represented by the artifact header."""

    gate_shape: tuple[int, int]
    up_shape: tuple[int, int]
    down_shape: tuple[int, int]
    group_size: int
    groups: tuple[int, int, int]
    num_stages: int
    codebook_size: int
    code_bits: int
    scale_bits: int
    scale_codebook_size: int
    iterations: int
    sample_limit: int | None
    seed: int
    schema_version: int = NATURAL_ADDITIVE_SCHEMA_VERSION
    header_bytes: int = NATURAL_ADDITIVE_HEADER_BYTES
    codec: str = "natural_matrix_additive_residual"
    matrix_order: str = "gate_up_down"
    grouping: str = "natural_input_contiguous_zero_padded"
    packed_order: str = "least_significant_bit_first"
    codebook_dtype: str = "float16"
    scale_method: str = "matrix_local_log2_kmeans_row_max_abs"
    decoded_dtype: str = "float32"

    @property
    def shapes(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        return self.gate_shape, self.up_shape, self.down_shape

    @property
    def hidden_size(self) -> int:
        return self.gate_shape[1]

    @property
    def intermediate_size(self) -> int:
        return self.gate_shape[0]

    def validate(self) -> None:
        if self.schema_version != NATURAL_ADDITIVE_SCHEMA_VERSION:
            raise NaturalAdditiveQuantizationError(
                f"unsupported metadata version {self.schema_version}"
            )
        if self.header_bytes != NATURAL_ADDITIVE_HEADER_BYTES:
            raise NaturalAdditiveQuantizationError(
                f"header_bytes must be {NATURAL_ADDITIVE_HEADER_BYTES}"
            )
        expected_strings = {
            "codec": "natural_matrix_additive_residual",
            "matrix_order": "gate_up_down",
            "grouping": "natural_input_contiguous_zero_padded",
            "packed_order": "least_significant_bit_first",
            "codebook_dtype": "float16",
            "scale_method": "matrix_local_log2_kmeans_row_max_abs",
            "decoded_dtype": "float32",
        }
        for field, expected in expected_strings.items():
            if getattr(self, field) != expected:
                raise NaturalAdditiveQuantizationError(
                    f"unsupported {field} {getattr(self, field)!r}"
                )
        gate = _shape(self.gate_shape, "gate_shape")
        up = _shape(self.up_shape, "up_shape")
        down = _shape(self.down_shape, "down_shape")
        if up != gate:
            raise NaturalAdditiveQuantizationError(
                "gate_shape and up_shape must be identical"
            )
        expected_down = (gate[1], gate[0])
        if down != expected_down:
            raise NaturalAdditiveQuantizationError(
                f"down_shape must be {expected_down} for the gate/up shapes"
            )
        group = _positive_integer(self.group_size, "group_size", maximum=(1 << 32) - 1)
        expected_groups = tuple(math.ceil(shape[1] / group) for shape in self.shapes)
        if (
            not isinstance(self.groups, tuple)
            or len(self.groups) != NATURAL_MATRIX_COUNT
            or tuple(self.groups) != expected_groups
        ):
            raise NaturalAdditiveQuantizationError(
                f"groups must equal the per-matrix input group counts {expected_groups}"
            )
        fixed = (
            self.group_size == NATURAL_GROUP_SIZE
            and self.num_stages == NATURAL_NUM_STAGES
            and self.codebook_size == NATURAL_CODEBOOK_SIZE
            and self.code_bits == NATURAL_CODE_BITS
            and self.scale_bits == NATURAL_SCALE_BITS
            and self.scale_codebook_size == NATURAL_SCALE_CODEBOOK_SIZE
        )
        if not fixed:
            raise NaturalAdditiveQuantizationError(
                "schema version 1 requires g8, two 7-bit/128-entry stages, "
                "and 6-bit/64-entry row scales"
            )
        _positive_integer(self.iterations, "iterations", maximum=(1 << 32) - 1)
        if self.sample_limit is not None:
            _positive_integer(
                self.sample_limit, "sample_limit", maximum=(1 << 32) - 1
            )
        _seed(self.seed)

    @property
    def row_count(self) -> int:
        return sum(shape[0] for shape in self.shapes)

    @property
    def weight_count(self) -> int:
        return sum(shape[0] * shape[1] for shape in self.shapes)

    @property
    def code_shapes(
        self,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        return tuple(
            (shape[0], self.num_stages, groups)
            for shape, groups in zip(self.shapes, self.groups, strict=True)
        )  # type: ignore[return-value]

    @property
    def code_count(self) -> int:
        return sum(math.prod(shape) for shape in self.code_shapes)

    @property
    def packed_code_bytes(self) -> int:
        return (self.code_count * self.code_bits + 7) // 8

    @property
    def codebook_bytes(self) -> int:
        return (
            NATURAL_MATRIX_COUNT
            * self.num_stages
            * self.codebook_size
            * self.group_size
            * np.dtype(np.float16).itemsize
        )

    @property
    def packed_scale_bytes(self) -> int:
        return (self.row_count * self.scale_bits + 7) // 8

    @property
    def scale_codebook_bytes(self) -> int:
        return (
            NATURAL_MATRIX_COUNT
            * self.scale_codebook_size
            * np.dtype(np.float16).itemsize
        )

    @property
    def storage_bytes(self) -> int:
        return (
            self.header_bytes
            + self.packed_code_bytes
            + self.codebook_bytes
            + self.packed_scale_bytes
            + self.scale_codebook_bytes
        )

    @property
    def dense_q4_bytes(self) -> int:
        return (self.weight_count * 4 + 7) // 8

    @property
    def fraction_of_dense_q4(self) -> float:
        return self.storage_bytes / self.dense_q4_bytes

    @property
    def effective_bits_per_weight(self) -> float:
        return self.storage_bytes * 8 / self.weight_count

    def storage_components(self) -> dict[str, int]:
        return {
            "header_and_checksum": self.header_bytes,
            "packed_codes": self.packed_code_bytes,
            "codebooks": self.codebook_bytes,
            "packed_scale_indices": self.packed_scale_bytes,
            "scale_codebooks": self.scale_codebook_bytes,
            "total": self.storage_bytes,
        }


def make_natural_additive_metadata(
    gate_shape: tuple[int, int],
    up_shape: tuple[int, int],
    down_shape: tuple[int, int],
    *,
    group_size: int = NATURAL_GROUP_SIZE,
    iterations: int = 8,
    sample_limit: int | None = 4096,
    seed: int = 0,
) -> NaturalAdditiveMetadata:
    """Create validated schema-v1 metadata for three natural matrices."""

    gate = _shape(gate_shape, "gate_shape")
    up = _shape(up_shape, "up_shape")
    down = _shape(down_shape, "down_shape")
    group = _positive_integer(group_size, "group_size", maximum=(1 << 32) - 1)
    rounds = _positive_integer(iterations, "iterations", maximum=(1 << 32) - 1)
    limit = (
        None
        if sample_limit is None
        else _positive_integer(sample_limit, "sample_limit", maximum=(1 << 32) - 1)
    )
    stable_seed = _seed(seed)
    metadata = NaturalAdditiveMetadata(
        gate_shape=gate,
        up_shape=up,
        down_shape=down,
        group_size=group,
        groups=tuple(math.ceil(shape[1] / group) for shape in (gate, up, down)),
        num_stages=NATURAL_NUM_STAGES,
        codebook_size=NATURAL_CODEBOOK_SIZE,
        code_bits=NATURAL_CODE_BITS,
        scale_bits=NATURAL_SCALE_BITS,
        scale_codebook_size=NATURAL_SCALE_CODEBOOK_SIZE,
        iterations=rounds,
        sample_limit=limit,
        seed=stable_seed,
    )
    metadata.validate()
    return metadata


def natural_additive_storage_plan(
    *, hidden_size: int = 576, intermediate_size: int = 1536
) -> dict[str, int | float | tuple[int, int] | tuple[int, int, int]]:
    """Return exact cold bytes for a natural SwiGLU layer."""

    hidden = _positive_integer(hidden_size, "hidden_size")
    intermediate = _positive_integer(intermediate_size, "intermediate_size")
    metadata = make_natural_additive_metadata(
        (intermediate, hidden),
        (intermediate, hidden),
        (hidden, intermediate),
    )
    return {
        "gate_shape": metadata.gate_shape,
        "up_shape": metadata.up_shape,
        "down_shape": metadata.down_shape,
        "groups": metadata.groups,
        "group_size": metadata.group_size,
        "num_stages": metadata.num_stages,
        "code_bits": metadata.code_bits,
        **metadata.storage_components(),
        "dense_q4_bytes": metadata.dense_q4_bytes,
        "fraction_of_dense_q4": metadata.fraction_of_dense_q4,
        "effective_bits_per_weight": metadata.effective_bits_per_weight,
    }


@dataclass(frozen=True)
class NaturalAdditiveEncoding:
    """Packed assignments and public centroid arrays for one SwiGLU layer."""

    packed_codes: NDArray[np.uint8]
    codebooks: NDArray[np.float16]
    packed_scale_indices: NDArray[np.uint8]
    scale_codebooks: NDArray[np.float16]
    metadata: NaturalAdditiveMetadata

    @property
    def storage_bytes(self) -> int:
        return self.metadata.storage_bytes

    @property
    def fraction_of_dense_q4(self) -> float:
        return self.metadata.fraction_of_dense_q4

    def storage_components(self) -> dict[str, int]:
        return self.metadata.storage_components()

    def validate(self) -> None:
        metadata = self.metadata
        metadata.validate()
        codes = _unpack_fixed_width(
            self.packed_codes,
            metadata.code_count,
            metadata.code_bits,
            "packed_codes",
        )
        if codes.size and int(np.max(codes)) >= metadata.codebook_size:
            raise NaturalAdditiveQuantizationError(
                "a packed code is outside its codebook"
            )
        expected_codebooks = (
            NATURAL_MATRIX_COUNT,
            metadata.num_stages,
            metadata.codebook_size,
            metadata.group_size,
        )
        if (
            self.codebooks.dtype != np.dtype(np.float16)
            or self.codebooks.shape != expected_codebooks
            or not self.codebooks.flags.c_contiguous
        ):
            raise NaturalAdditiveQuantizationError(
                f"codebooks must be contiguous float16 with shape {expected_codebooks}"
            )
        if not np.all(np.isfinite(self.codebooks)):
            raise NaturalAdditiveQuantizationError(
                "codebooks must contain only finite values"
            )
        scale_indices = _unpack_fixed_width(
            self.packed_scale_indices,
            metadata.row_count,
            metadata.scale_bits,
            "packed_scale_indices",
        )
        if (
            scale_indices.size
            and int(np.max(scale_indices)) >= metadata.scale_codebook_size
        ):
            raise NaturalAdditiveQuantizationError(
                "a packed scale index is outside its codebook"
            )
        expected_scales = (NATURAL_MATRIX_COUNT, metadata.scale_codebook_size)
        if (
            self.scale_codebooks.dtype != np.dtype(np.float16)
            or self.scale_codebooks.shape != expected_scales
            or not self.scale_codebooks.flags.c_contiguous
        ):
            raise NaturalAdditiveQuantizationError(
                "scale_codebooks must be contiguous float16 with shape "
                f"{expected_scales}"
            )
        if not np.all(np.isfinite(self.scale_codebooks)) or np.any(
            self.scale_codebooks < 0
        ):
            raise NaturalAdditiveQuantizationError(
                "scale_codebooks must be finite and non-negative"
            )

    def unpack_codes(
        self,
    ) -> tuple[NDArray[np.uint16], NDArray[np.uint16], NDArray[np.uint16]]:
        """Return ``(gate, up, down)`` assignments as ``[row, stage, group]``."""

        self.validate()
        flat = _unpack_fixed_width(
            self.packed_codes,
            self.metadata.code_count,
            self.metadata.code_bits,
            "packed_codes",
        )
        result: list[NDArray[np.uint16]] = []
        offset = 0
        for shape in self.metadata.code_shapes:
            count = math.prod(shape)
            result.append(np.ascontiguousarray(flat[offset : offset + count].reshape(shape)))
            offset += count
        return result[0], result[1], result[2]

    def unpack_scale_indices(
        self,
    ) -> tuple[NDArray[np.uint16], NDArray[np.uint16], NDArray[np.uint16]]:
        """Return matrix-local row-scale assignments in gate/up/down order."""

        self.validate()
        flat = _unpack_fixed_width(
            self.packed_scale_indices,
            self.metadata.row_count,
            self.metadata.scale_bits,
            "packed_scale_indices",
        )
        result: list[NDArray[np.uint16]] = []
        offset = 0
        for shape in self.metadata.shapes:
            result.append(np.ascontiguousarray(flat[offset : offset + shape[0]]))
            offset += shape[0]
        return result[0], result[1], result[2]

    def decoded_scales(
        self,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Decode positive per-output-row scales for all three matrices."""

        indices = self.unpack_scale_indices()
        return tuple(
            np.ascontiguousarray(
                self.scale_codebooks[matrix_index, matrix_indices].astype(np.float32)
            )
            for matrix_index, matrix_indices in enumerate(indices)
        )  # type: ignore[return-value]


def _integer_assignments(
    values: ArrayLike, expected_shape: tuple[int, ...], limit: int, name: str
) -> NDArray[np.uint16]:
    array = np.asarray(values)
    if array.shape != expected_shape or array.dtype.kind not in "iu":
        raise NaturalAdditiveQuantizationError(
            f"{name} must be an integer array with shape {expected_shape}"
        )
    if array.size and (int(np.min(array)) < 0 or int(np.max(array)) >= limit):
        raise NaturalAdditiveQuantizationError(
            f"{name} contains an assignment outside [0, {limit})"
        )
    return np.ascontiguousarray(array, dtype=np.uint16)


def _stored_float16(
    values: ArrayLike, expected_shape: tuple[int, ...], name: str
) -> NDArray[np.float16]:
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise NaturalAdditiveQuantizationError(f"{name} must be numeric") from exc
    if array.shape != expected_shape or not np.all(np.isfinite(array)):
        raise NaturalAdditiveQuantizationError(
            f"{name} must be finite with shape {expected_shape}"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        stored = array.astype(np.float16)
    if not np.all(np.isfinite(stored)):
        raise NaturalAdditiveQuantizationError(
            f"{name} contains a value outside finite float16 range"
        )
    return np.ascontiguousarray(stored)


def build_natural_additive_encoding(
    metadata: NaturalAdditiveMetadata,
    *,
    codes: Sequence[ArrayLike],
    codebooks: ArrayLike,
    scale_indices: Sequence[ArrayLike],
    scale_codebooks: ArrayLike,
) -> NaturalAdditiveEncoding:
    """Pack externally optimized codes/centroids into deployment form.

    This is the intended bridge for activation-aware P/V optimization.  Values
    are quantized to the actual persisted dtypes before the returned artifact is
    validated, so downstream metrics can be measured on deployable data.
    """

    metadata.validate()
    if len(codes) != NATURAL_MATRIX_COUNT:
        raise NaturalAdditiveQuantizationError("codes must contain gate, up, and down")
    if len(scale_indices) != NATURAL_MATRIX_COUNT:
        raise NaturalAdditiveQuantizationError(
            "scale_indices must contain gate, up, and down"
        )
    stored_codes = tuple(
        _integer_assignments(values, shape, metadata.codebook_size, f"{name}_codes")
        for name, values, shape in zip(
            NATURAL_MATRIX_NAMES, codes, metadata.code_shapes, strict=True
        )
    )
    stored_scale_indices = tuple(
        _integer_assignments(
            values,
            (shape[0],),
            metadata.scale_codebook_size,
            f"{name}_scale_indices",
        )
        for name, values, shape in zip(
            NATURAL_MATRIX_NAMES, scale_indices, metadata.shapes, strict=True
        )
    )
    stored_codebooks = _stored_float16(
        codebooks,
        (
            NATURAL_MATRIX_COUNT,
            metadata.num_stages,
            metadata.codebook_size,
            metadata.group_size,
        ),
        "codebooks",
    )
    stored_scale_codebooks = _stored_float16(
        scale_codebooks,
        (NATURAL_MATRIX_COUNT, metadata.scale_codebook_size),
        "scale_codebooks",
    )
    if np.any(stored_scale_codebooks < 0):
        raise NaturalAdditiveQuantizationError(
            "scale_codebooks must be non-negative"
        )
    encoding = NaturalAdditiveEncoding(
        packed_codes=_pack_fixed_width(
            np.concatenate([value.reshape(-1) for value in stored_codes]),
            metadata.code_bits,
        ),
        codebooks=stored_codebooks,
        packed_scale_indices=_pack_fixed_width(
            np.concatenate(stored_scale_indices), metadata.scale_bits
        ),
        scale_codebooks=stored_scale_codebooks,
        metadata=metadata,
    )
    encoding.validate()
    return encoding


def _squared_distances(
    values: NDArray[np.float32], centers: NDArray[np.float32]
) -> NDArray[np.float32]:
    value_norms = np.einsum("nd,nd->n", values, values)[:, None]
    center_norms = np.einsum("kd,kd->k", centers, centers)[None, :]
    distances = value_norms + center_norms - 2.0 * (values @ centers.T)
    return np.maximum(distances, 0.0, out=distances)


def _assign_nearest(
    values: NDArray[np.float32], centers: NDArray[np.float32], *, chunk_size: int = 16384
) -> NDArray[np.uint16]:
    result = np.empty(values.shape[0], dtype=np.uint16)
    for start in range(0, values.shape[0], chunk_size):
        stop = min(start + chunk_size, values.shape[0])
        result[start:stop] = np.argmin(
            _squared_distances(values[start:stop], centers), axis=1
        ).astype(np.uint16)
    return result


def _fit_codebook(
    values: NDArray[np.float32],
    *,
    codebook_size: int,
    iterations: int,
    sample_limit: int | None,
    seed: int,
) -> NDArray[np.float32]:
    """Deterministic residual k-means, permitting repeated initial centers."""

    rng = np.random.default_rng(seed)
    if sample_limit is not None and values.shape[0] > sample_limit:
        sample_ids = np.sort(
            rng.choice(values.shape[0], size=sample_limit, replace=False)
        )
        samples = values[sample_ids]
    else:
        samples = values
    if not samples.shape[0]:
        raise NaturalAdditiveQuantizationError("cannot fit a codebook to no vectors")
    if samples.shape[0] >= codebook_size:
        initial = rng.choice(samples.shape[0], size=codebook_size, replace=False)
    else:
        initial = np.resize(rng.permutation(samples.shape[0]), codebook_size)
    centers = np.ascontiguousarray(samples[initial].copy(), dtype=np.float32)
    for _ in range(iterations):
        assignments = _assign_nearest(samples, centers)
        sums = np.zeros_like(centers)
        np.add.at(sums, assignments, samples)
        counts = np.bincount(assignments, minlength=codebook_size)
        updated = centers.copy()
        populated = counts > 0
        updated[populated] = sums[populated] / counts[populated, None]
        if np.array_equal(updated, centers):
            break
        centers = updated
    return np.ascontiguousarray(centers)


def _derived_seed(base: int, *parts: int) -> int:
    return int(
        np.random.SeedSequence([base, *parts]).generate_state(1, dtype=np.uint32)[0]
    )


def _fit_scale_codebooks(
    matrices: tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]],
    metadata: NaturalAdditiveMetadata,
) -> tuple[
    tuple[NDArray[np.uint16], NDArray[np.uint16], NDArray[np.uint16]],
    NDArray[np.float16],
    tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]],
]:
    scale_codebooks = np.empty(
        (NATURAL_MATRIX_COUNT, metadata.scale_codebook_size), dtype=np.float16
    )
    all_indices: list[NDArray[np.uint16]] = []
    all_decoded: list[NDArray[np.float32]] = []
    smallest = np.nextafter(np.float16(0), np.float16(1), dtype=np.float16).astype(
        np.float32
    )
    for matrix_index, matrix in enumerate(matrices):
        exact = np.max(np.abs(matrix), axis=1).astype(np.float32)
        positive = exact > 0
        logs = np.log2(np.maximum(exact, smallest))
        centers = _fit_codebook(
            logs[:, None],
            codebook_size=metadata.scale_codebook_size,
            iterations=metadata.iterations,
            sample_limit=metadata.sample_limit,
            seed=_derived_seed(metadata.seed, 0x5343414C, matrix_index),
        )[:, 0]
        stored = _stored_float16(
            np.exp2(centers),
            (metadata.scale_codebook_size,),
            f"{NATURAL_MATRIX_NAMES[matrix_index]} scale codebook",
        )
        zero_id: int | None = None
        if np.any(~positive):
            zero_id = int(np.argmin(stored))
            stored[zero_id] = np.float16(0)
        stored_logs = np.log2(np.maximum(stored.astype(np.float32), smallest))
        distances = np.square(logs[:, None] - stored_logs[None, :])
        if zero_id is not None:
            distances[positive, zero_id] = np.inf
        indices = np.argmin(distances, axis=1).astype(np.uint16)
        if zero_id is not None:
            indices[~positive] = zero_id
        scale_codebooks[matrix_index] = stored
        all_indices.append(indices)
        all_decoded.append(stored[indices].astype(np.float32))
    return (
        (all_indices[0], all_indices[1], all_indices[2]),
        np.ascontiguousarray(scale_codebooks),
        (all_decoded[0], all_decoded[1], all_decoded[2]),
    )


def fit_natural_additive(
    gate: ArrayLike,
    up: ArrayLike,
    down: ArrayLike,
    *,
    group_size: int = NATURAL_GROUP_SIZE,
    iterations: int = 8,
    sample_limit: int | None = 4096,
    seed: int = 0,
) -> NaturalAdditiveEncoding:
    """Fit deterministic matrix-local residual codebooks and pack the result."""

    matrices = (
        _matrix(gate, "gate"),
        _matrix(up, "up"),
        _matrix(down, "down"),
    )
    metadata = make_natural_additive_metadata(
        matrices[0].shape,
        matrices[1].shape,
        matrices[2].shape,
        group_size=group_size,
        iterations=iterations,
        sample_limit=sample_limit,
        seed=seed,
    )
    scale_indices, scale_codebooks, decoded_scales = _fit_scale_codebooks(
        matrices, metadata
    )
    codebooks = np.empty(
        (
            NATURAL_MATRIX_COUNT,
            metadata.num_stages,
            metadata.codebook_size,
            metadata.group_size,
        ),
        dtype=np.float16,
    )
    all_codes: list[NDArray[np.uint16]] = []
    for matrix_index, (matrix, groups, scales) in enumerate(
        zip(matrices, metadata.groups, decoded_scales, strict=True)
    ):
        padded = np.zeros(
            (matrix.shape[0], groups * metadata.group_size), dtype=np.float32
        )
        divisors = np.where(scales > 0, scales, 1.0)
        padded[:, : matrix.shape[1]] = matrix / divisors[:, None]
        residual = padded.reshape(
            matrix.shape[0], groups, metadata.group_size
        ).copy()
        matrix_codes = np.empty(
            (matrix.shape[0], metadata.num_stages, groups), dtype=np.uint16
        )
        for stage in range(metadata.num_stages):
            vectors = residual.reshape(-1, metadata.group_size)
            centers = _fit_codebook(
                vectors,
                codebook_size=metadata.codebook_size,
                iterations=metadata.iterations,
                sample_limit=metadata.sample_limit,
                seed=_derived_seed(metadata.seed, 0x434F4445, matrix_index, stage),
            )
            stored = _stored_float16(
                centers,
                (metadata.codebook_size, metadata.group_size),
                f"{NATURAL_MATRIX_NAMES[matrix_index]} stage {stage} codebook",
            )
            reconstruction = stored.astype(np.float32)
            assignments = _assign_nearest(vectors, reconstruction)
            codebooks[matrix_index, stage] = stored
            matrix_codes[:, stage, :] = assignments.reshape(matrix.shape[0], groups)
            residual = (
                vectors - reconstruction[assignments]
            ).reshape(matrix.shape[0], groups, metadata.group_size)
        all_codes.append(matrix_codes)
    return build_natural_additive_encoding(
        metadata,
        codes=(all_codes[0], all_codes[1], all_codes[2]),
        codebooks=codebooks,
        scale_indices=scale_indices,
        scale_codebooks=scale_codebooks,
    )


def decode_natural_additive(
    encoding: NaturalAdditiveEncoding,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Decode and return natural ``(gate, up, down)`` float32 matrices."""

    encoding.validate()
    metadata = encoding.metadata
    codes = encoding.unpack_codes()
    scales = encoding.decoded_scales()
    decoded_matrices: list[NDArray[np.float32]] = []
    for matrix_index, (shape, groups, matrix_codes, matrix_scales) in enumerate(
        zip(metadata.shapes, metadata.groups, codes, scales, strict=True)
    ):
        decoded = np.zeros(
            (shape[0], groups, metadata.group_size), dtype=np.float32
        )
        for stage in range(metadata.num_stages):
            decoded += encoding.codebooks[
                matrix_index, stage, matrix_codes[:, stage, :], :
            ].astype(np.float32)
        matrix = decoded.reshape(shape[0], groups * metadata.group_size)[:, : shape[1]]
        matrix *= matrix_scales[:, None]
        decoded_matrices.append(np.ascontiguousarray(matrix, dtype=np.float32))
    return decoded_matrices[0], decoded_matrices[1], decoded_matrices[2]


def _payload(encoding: NaturalAdditiveEncoding) -> bytes:
    return b"".join(
        (
            encoding.packed_codes.tobytes(order="C"),
            encoding.codebooks.astype("<f2", copy=False).tobytes(order="C"),
            encoding.packed_scale_indices.tobytes(order="C"),
            encoding.scale_codebooks.astype("<f2", copy=False).tobytes(order="C"),
        )
    )


def _header(
    metadata: NaturalAdditiveMetadata, *, payload_bytes: int, checksum: bytes
) -> bytes:
    prefix = _HEADER_STRUCT.pack(
        NATURAL_ADDITIVE_MAGIC,
        metadata.schema_version,
        metadata.header_bytes,
        _FLAGS,
        metadata.gate_shape[0],
        metadata.gate_shape[1],
        metadata.up_shape[0],
        metadata.up_shape[1],
        metadata.down_shape[0],
        metadata.down_shape[1],
        metadata.group_size,
        metadata.groups[0],
        metadata.groups[1],
        metadata.groups[2],
        metadata.num_stages,
        metadata.codebook_size,
        metadata.code_bits,
        metadata.scale_bits,
        metadata.scale_codebook_size,
        metadata.iterations,
        0 if metadata.sample_limit is None else metadata.sample_limit,
        NATURAL_MATRIX_COUNT,
        metadata.seed,
        metadata.code_count,
        metadata.packed_code_bytes,
        metadata.codebook_bytes,
        metadata.row_count,
        metadata.packed_scale_bytes,
        metadata.scale_codebook_bytes,
        payload_bytes,
        checksum,
    )
    return prefix + bytes(NATURAL_ADDITIVE_HEADER_BYTES - len(prefix))


def serialize_natural_additive(encoding: NaturalAdditiveEncoding) -> bytes:
    """Serialize and checksum the complete cold artifact."""

    encoding.validate()
    payload = _payload(encoding)
    expected = encoding.storage_bytes - NATURAL_ADDITIVE_HEADER_BYTES
    if len(payload) != expected:
        raise NaturalAdditiveQuantizationError(
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
        raise NaturalAdditiveQuantizationError(
            "serialized artifact disagrees with total-byte accounting"
        )
    return artifact


def deserialize_natural_additive(
    data: bytes | bytearray | memoryview,
) -> NaturalAdditiveEncoding:
    """Checksum and strictly decode a schema-v1 artifact."""

    try:
        artifact = bytes(data)
    except (TypeError, ValueError) as exc:
        raise NaturalAdditiveQuantizationError("artifact must be bytes-like") from exc
    if len(artifact) < NATURAL_ADDITIVE_HEADER_BYTES:
        raise NaturalAdditiveQuantizationError("artifact is shorter than its header")
    header = artifact[:NATURAL_ADDITIVE_HEADER_BYTES]
    fields = _HEADER_STRUCT.unpack(header[: _HEADER_STRUCT.size])
    (
        magic,
        schema_version,
        header_bytes,
        flags,
        gate_rows,
        gate_width,
        up_rows,
        up_width,
        down_rows,
        down_width,
        group_size,
        gate_groups,
        up_groups,
        down_groups,
        num_stages,
        codebook_size,
        code_bits,
        scale_bits,
        scale_codebook_size,
        iterations,
        stored_sample_limit,
        matrix_count,
        seed,
        code_count,
        packed_code_bytes,
        codebook_bytes,
        row_count,
        packed_scale_bytes,
        scale_codebook_bytes,
        payload_bytes,
        checksum,
    ) = fields
    if magic != NATURAL_ADDITIVE_MAGIC:
        raise NaturalAdditiveQuantizationError("artifact magic is invalid")
    if header_bytes != NATURAL_ADDITIVE_HEADER_BYTES:
        raise NaturalAdditiveQuantizationError("artifact header size is invalid")
    if flags != _FLAGS:
        raise NaturalAdditiveQuantizationError("artifact flags are unsupported")
    if matrix_count != NATURAL_MATRIX_COUNT:
        raise NaturalAdditiveQuantizationError("artifact matrix count is invalid")
    if any(header[_HEADER_STRUCT.size :]):
        raise NaturalAdditiveQuantizationError(
            "artifact reserved header bytes must be zero"
        )
    if len(artifact) != header_bytes + payload_bytes:
        raise NaturalAdditiveQuantizationError(
            "artifact length disagrees with its header"
        )
    zeroed = bytearray(header)
    zeroed[_CHECKSUM_OFFSET : _CHECKSUM_OFFSET + 32] = bytes(32)
    expected_checksum = hashlib.sha256(bytes(zeroed) + artifact[header_bytes:]).digest()
    if not hmac.compare_digest(checksum, expected_checksum):
        raise NaturalAdditiveQuantizationError("artifact checksum mismatch")

    metadata = NaturalAdditiveMetadata(
        gate_shape=(gate_rows, gate_width),
        up_shape=(up_rows, up_width),
        down_shape=(down_rows, down_width),
        group_size=group_size,
        groups=(gate_groups, up_groups, down_groups),
        num_stages=num_stages,
        codebook_size=codebook_size,
        code_bits=code_bits,
        scale_bits=scale_bits,
        scale_codebook_size=scale_codebook_size,
        iterations=iterations,
        sample_limit=None if stored_sample_limit == 0 else stored_sample_limit,
        seed=seed,
        schema_version=schema_version,
        header_bytes=header_bytes,
    )
    metadata.validate()
    stored_sizes = (
        code_count,
        packed_code_bytes,
        codebook_bytes,
        row_count,
        packed_scale_bytes,
        scale_codebook_bytes,
        payload_bytes,
    )
    expected_sizes = (
        metadata.code_count,
        metadata.packed_code_bytes,
        metadata.codebook_bytes,
        metadata.row_count,
        metadata.packed_scale_bytes,
        metadata.scale_codebook_bytes,
        metadata.storage_bytes - metadata.header_bytes,
    )
    if stored_sizes != expected_sizes:
        raise NaturalAdditiveQuantizationError(
            "artifact section sizes disagree with metadata"
        )

    payload = memoryview(artifact)[header_bytes:]
    offset = 0

    def section(size: int) -> memoryview:
        nonlocal offset
        result = payload[offset : offset + size]
        offset += size
        return result

    packed_codes = np.frombuffer(section(packed_code_bytes), dtype=np.uint8).copy()
    codebooks = np.frombuffer(section(codebook_bytes), dtype="<f2").astype(
        np.float16, copy=True
    ).reshape(
        NATURAL_MATRIX_COUNT,
        metadata.num_stages,
        metadata.codebook_size,
        metadata.group_size,
    )
    packed_scale_indices = np.frombuffer(
        section(packed_scale_bytes), dtype=np.uint8
    ).copy()
    scale_codebooks = np.frombuffer(
        section(scale_codebook_bytes), dtype="<f2"
    ).astype(np.float16, copy=True).reshape(
        NATURAL_MATRIX_COUNT, metadata.scale_codebook_size
    )
    if offset != len(payload):
        raise NaturalAdditiveQuantizationError(
            "artifact has unconsumed payload bytes"
        )
    encoding = NaturalAdditiveEncoding(
        packed_codes=np.ascontiguousarray(packed_codes),
        codebooks=np.ascontiguousarray(codebooks),
        packed_scale_indices=np.ascontiguousarray(packed_scale_indices),
        scale_codebooks=np.ascontiguousarray(scale_codebooks),
        metadata=metadata,
    )
    encoding.validate()
    return encoding


def save_natural_additive(path: str | Path, encoding: NaturalAdditiveEncoding) -> str:
    """Atomically save an artifact and return its embedded checksum."""

    destination = Path(path)
    artifact = serialize_natural_additive(encoding)
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


def load_natural_additive(path: str | Path) -> NaturalAdditiveEncoding:
    """Load, checksum, and validate a serialized artifact."""

    return deserialize_natural_additive(Path(path).read_bytes())


__all__ = [
    "NATURAL_ADDITIVE_HEADER_BYTES",
    "NATURAL_ADDITIVE_SCHEMA_VERSION",
    "NATURAL_CODEBOOK_SIZE",
    "NATURAL_CODE_BITS",
    "NATURAL_GROUP_SIZE",
    "NATURAL_MATRIX_NAMES",
    "NATURAL_NUM_STAGES",
    "NATURAL_SCALE_BITS",
    "NATURAL_SCALE_CODEBOOK_SIZE",
    "NaturalAdditiveEncoding",
    "NaturalAdditiveMetadata",
    "NaturalAdditiveQuantizationError",
    "build_natural_additive_encoding",
    "decode_natural_additive",
    "deserialize_natural_additive",
    "fit_natural_additive",
    "load_natural_additive",
    "make_natural_additive_metadata",
    "natural_additive_storage_plan",
    "save_natural_additive",
    "serialize_natural_additive",
]
