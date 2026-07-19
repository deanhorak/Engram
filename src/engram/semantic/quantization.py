"""Deterministic reference quantizers for semantic keys and values.

These implementations deliberately favor an inspectable scalar baseline over
speed.  Encoded arrays and codec parameters remain separate so a later package
writer can place them in independently checksummed, memory-mappable sections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


SCHEMA_VERSION = 1


class QuantizationError(ValueError):
    """Raised when quantizer input, metadata, or encoded data is invalid."""


def _storage_dtype(bits: int) -> np.dtype[Any]:
    if not isinstance(bits, int) or isinstance(bits, bool) or not 1 <= bits <= 16:
        raise QuantizationError("bits must be an integer in [1, 16]")
    return np.dtype(np.uint8 if bits <= 8 else np.uint16)


def _matrix(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise QuantizationError(f"{name} must be numeric") from error
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise QuantizationError(f"{name} must be a non-empty rank-2 matrix")
    if not np.all(np.isfinite(array)):
        raise QuantizationError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class ScalarAffineMetadata:
    """JSON-serializable parameters for column-wise affine key coding."""

    shape: tuple[int, int]
    bits: int
    storage_dtype: str
    offsets: tuple[float, ...]
    scales: tuple[float, ...]
    schema_version: int = SCHEMA_VERSION
    codec: str = "scalar_affine_per_dimension"
    decoded_dtype: str = "float32"
    byte_order: str = "little"

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise QuantizationError(f"unsupported scalar metadata version {self.schema_version}")
        if self.codec != "scalar_affine_per_dimension":
            raise QuantizationError(f"unsupported scalar codec {self.codec!r}")
        expected_dtype = _storage_dtype(self.bits).name
        if self.storage_dtype != expected_dtype:
            raise QuantizationError(
                f"storage_dtype must be {expected_dtype!r} for {self.bits}-bit codes"
            )
        if self.decoded_dtype != "float32":
            raise QuantizationError("decoded_dtype must be 'float32'")
        if self.byte_order != "little":
            raise QuantizationError("byte_order must be 'little'")
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) != 2
            or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in self.shape)
        ):
            raise QuantizationError("shape must contain two positive integers")
        width = self.shape[1]
        if (
            not isinstance(self.offsets, tuple)
            or not isinstance(self.scales, tuple)
            or len(self.offsets) != width
            or len(self.scales) != width
        ):
            raise QuantizationError("offset and scale lengths must equal the matrix width")
        try:
            offsets = np.asarray(self.offsets, dtype=np.float64)
            scales = np.asarray(self.scales, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise QuantizationError("offsets and scales must be numeric") from error
        if not np.all(np.isfinite(offsets)):
            raise QuantizationError("offsets must be finite")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise QuantizationError("scales must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "codec": self.codec,
            "shape": list(self.shape),
            "bits": self.bits,
            "storage_dtype": self.storage_dtype,
            "decoded_dtype": self.decoded_dtype,
            "byte_order": self.byte_order,
            "offsets": list(self.offsets),
            "scales": list(self.scales),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScalarAffineMetadata":
        try:
            result = cls(
                schema_version=payload["schema_version"],
                codec=payload["codec"],
                shape=tuple(payload["shape"]),
                bits=payload["bits"],
                storage_dtype=payload["storage_dtype"],
                decoded_dtype=payload["decoded_dtype"],
                byte_order=payload["byte_order"],
                offsets=tuple(payload["offsets"]),
                scales=tuple(payload["scales"]),
            )
        except (KeyError, TypeError) as error:
            raise QuantizationError("invalid scalar affine metadata") from error
        result.validate()
        return result


@dataclass(frozen=True)
class ScalarAffineEncoding:
    codes: NDArray[np.unsignedinteger[Any]]
    metadata: ScalarAffineMetadata


def encode_scalar_affine(keys: ArrayLike, *, bits: int = 8) -> ScalarAffineEncoding:
    """Encode a key matrix with an independent affine range per dimension."""

    matrix = _matrix(keys, name="keys")
    dtype = _storage_dtype(bits)
    levels = (1 << bits) - 1
    offsets = np.min(matrix, axis=0)
    spans = np.max(matrix, axis=0) - offsets
    scales = spans / levels
    # A unit scale plus the exact constant as offset reconstructs constant
    # dimensions without a special value or a zero scale in the file format.
    scales = np.where(spans == 0.0, 1.0, scales)
    codes = np.rint((matrix - offsets) / scales)
    codes = np.clip(codes, 0, levels).astype(dtype)
    metadata = ScalarAffineMetadata(
        shape=(matrix.shape[0], matrix.shape[1]),
        bits=bits,
        storage_dtype=dtype.name,
        offsets=tuple(float(value) for value in offsets),
        scales=tuple(float(value) for value in scales),
    )
    metadata.validate()
    return ScalarAffineEncoding(codes=codes, metadata=metadata)


def decode_scalar_affine(
    encoding: ScalarAffineEncoding,
) -> NDArray[np.float32]:
    metadata = encoding.metadata
    metadata.validate()
    codes = np.asarray(encoding.codes)
    expected_dtype = _storage_dtype(metadata.bits)
    if codes.dtype != expected_dtype:
        raise QuantizationError(f"codes must have dtype {expected_dtype.name}")
    if codes.shape != metadata.shape:
        raise QuantizationError(
            f"code shape {codes.shape} does not match metadata shape {metadata.shape}"
        )
    if codes.size and int(np.max(codes)) > (1 << metadata.bits) - 1:
        raise QuantizationError("a scalar code exceeds the configured bit range")
    offsets = np.asarray(metadata.offsets, dtype=np.float64)
    scales = np.asarray(metadata.scales, dtype=np.float64)
    return (offsets + codes.astype(np.float64) * scales).astype(np.float32)


@dataclass(frozen=True)
class AdditiveVectorMetadata:
    """Description of greedy residual vector-codebook compression."""

    shape: tuple[int, int]
    num_codebooks: int
    codebook_size: int
    iterations: int
    code_dtype: str
    codebook_dtype: str = "float32"
    decoded_dtype: str = "float32"
    schema_version: int = SCHEMA_VERSION
    codec: str = "additive_residual_vector_codebook"
    assignment: str = "greedy_squared_l2"
    initialization: str = "deterministic_farthest_first"
    byte_order: str = "little"

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise QuantizationError(f"unsupported additive metadata version {self.schema_version}")
        if self.codec != "additive_residual_vector_codebook":
            raise QuantizationError(f"unsupported additive codec {self.codec!r}")
        if self.assignment != "greedy_squared_l2":
            raise QuantizationError(f"unsupported assignment {self.assignment!r}")
        if self.initialization != "deterministic_farthest_first":
            raise QuantizationError(f"unsupported initialization {self.initialization!r}")
        if self.byte_order != "little":
            raise QuantizationError("byte_order must be 'little'")
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) != 2
            or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in self.shape)
        ):
            raise QuantizationError("shape must contain two positive integers")
        for name, value in (
            ("num_codebooks", self.num_codebooks),
            ("codebook_size", self.codebook_size),
            ("iterations", self.iterations),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise QuantizationError(f"{name} must be a positive integer")
        expected_dtype = _storage_dtype(max(1, (self.codebook_size - 1).bit_length())).name
        if self.code_dtype != expected_dtype:
            raise QuantizationError(
                f"code_dtype must be {expected_dtype!r} for {self.codebook_size} entries"
            )
        if self.codebook_dtype != "float32" or self.decoded_dtype != "float32":
            raise QuantizationError("codebook_dtype and decoded_dtype must be 'float32'")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "codec": self.codec,
            "shape": list(self.shape),
            "num_codebooks": self.num_codebooks,
            "codebook_size": self.codebook_size,
            "iterations": self.iterations,
            "code_dtype": self.code_dtype,
            "codebook_dtype": self.codebook_dtype,
            "decoded_dtype": self.decoded_dtype,
            "assignment": self.assignment,
            "initialization": self.initialization,
            "byte_order": self.byte_order,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdditiveVectorMetadata":
        try:
            result = cls(
                schema_version=payload["schema_version"],
                codec=payload["codec"],
                shape=tuple(payload["shape"]),
                num_codebooks=payload["num_codebooks"],
                codebook_size=payload["codebook_size"],
                iterations=payload["iterations"],
                code_dtype=payload["code_dtype"],
                codebook_dtype=payload["codebook_dtype"],
                decoded_dtype=payload["decoded_dtype"],
                assignment=payload["assignment"],
                initialization=payload["initialization"],
                byte_order=payload["byte_order"],
            )
        except (KeyError, TypeError) as error:
            raise QuantizationError("invalid additive vector metadata") from error
        result.validate()
        return result


@dataclass(frozen=True)
class AdditiveVectorEncoding:
    codes: NDArray[np.unsignedinteger[Any]]
    codebooks: NDArray[np.float32]
    metadata: AdditiveVectorMetadata


def _squared_distances(
    values: NDArray[np.float64], centers: NDArray[np.float64]
) -> NDArray[np.float64]:
    differences = values[:, None, :] - centers[None, :, :]
    return np.einsum("nkd,nkd->nk", differences, differences)


def _initial_centers(values: NDArray[np.float64], size: int) -> NDArray[np.float64]:
    if size > values.shape[0]:
        raise QuantizationError("codebook_size cannot exceed the number of value vectors")
    selected = [int(np.argmax(np.einsum("nd,nd->n", values, values)))]
    nearest = _squared_distances(values, values[selected])[:, 0]
    while len(selected) < size:
        nearest[np.asarray(selected)] = -1.0
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(nearest, _squared_distances(values, values[[index]])[:, 0])
    return values[np.asarray(selected)].copy()


def _fit_codebook(
    values: NDArray[np.float64], size: int, iterations: int
) -> NDArray[np.float64]:
    centers = _initial_centers(values, size)
    for _ in range(iterations):
        assignments = np.argmin(_squared_distances(values, centers), axis=1)
        updated = centers.copy()
        for index in range(size):
            members = values[assignments == index]
            if members.size:
                updated[index] = np.mean(members, axis=0)
        if np.array_equal(updated, centers):
            break
        centers = updated
    return centers


def encode_additive_vectors(
    values: ArrayLike,
    *,
    num_codebooks: int = 2,
    codebook_size: int = 16,
    iterations: int = 20,
) -> AdditiveVectorEncoding:
    """Fit codebooks and greedily encode values as a sum of codewords.

    With ``num_codebooks=1`` this is ordinary vector quantization.  Additional
    codebooks are trained successively on the previous stage's residual.
    """

    matrix = _matrix(values, name="values")
    for name, value in (
        ("num_codebooks", num_codebooks),
        ("codebook_size", codebook_size),
        ("iterations", iterations),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise QuantizationError(f"{name} must be a positive integer")
    if codebook_size > matrix.shape[0]:
        raise QuantizationError("codebook_size cannot exceed the number of value vectors")
    code_bits = max(1, (codebook_size - 1).bit_length())
    code_dtype = _storage_dtype(code_bits)
    residual = matrix.copy()
    codebooks = np.empty(
        (num_codebooks, codebook_size, matrix.shape[1]), dtype=np.float64
    )
    codes = np.empty((matrix.shape[0], num_codebooks), dtype=code_dtype)
    for stage in range(num_codebooks):
        centers = _fit_codebook(residual, codebook_size, iterations)
        assignments = np.argmin(_squared_distances(residual, centers), axis=1)
        codebooks[stage] = centers
        codes[:, stage] = assignments.astype(code_dtype)
        residual -= centers[assignments]
    metadata = AdditiveVectorMetadata(
        shape=(matrix.shape[0], matrix.shape[1]),
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        iterations=iterations,
        code_dtype=code_dtype.name,
    )
    metadata.validate()
    return AdditiveVectorEncoding(
        codes=codes,
        codebooks=codebooks.astype(np.float32),
        metadata=metadata,
    )


def decode_additive_vectors(
    encoding: AdditiveVectorEncoding,
) -> NDArray[np.float32]:
    metadata = encoding.metadata
    metadata.validate()
    codes = np.asarray(encoding.codes)
    codebooks = np.asarray(encoding.codebooks)
    expected_code_dtype = np.dtype(metadata.code_dtype)
    if codes.dtype != expected_code_dtype:
        raise QuantizationError(f"codes must have dtype {expected_code_dtype.name}")
    expected_code_shape = (metadata.shape[0], metadata.num_codebooks)
    if codes.shape != expected_code_shape:
        raise QuantizationError(
            f"code shape {codes.shape} does not match expected shape {expected_code_shape}"
        )
    expected_codebook_shape = (
        metadata.num_codebooks,
        metadata.codebook_size,
        metadata.shape[1],
    )
    if codebooks.dtype != np.dtype(metadata.codebook_dtype):
        raise QuantizationError(f"codebooks must have dtype {metadata.codebook_dtype}")
    if codebooks.shape != expected_codebook_shape:
        raise QuantizationError(
            f"codebook shape {codebooks.shape} does not match expected shape {expected_codebook_shape}"
        )
    if not np.all(np.isfinite(codebooks)):
        raise QuantizationError("codebooks must contain only finite values")
    if codes.size and int(np.max(codes)) >= metadata.codebook_size:
        raise QuantizationError("a value code is outside its codebook")
    decoded = np.zeros(metadata.shape, dtype=np.float32)
    rows = np.arange(metadata.shape[0])
    for stage in range(metadata.num_codebooks):
        decoded += codebooks[stage, codes[rows, stage]]
    return decoded


__all__ = [
    "AdditiveVectorEncoding",
    "AdditiveVectorMetadata",
    "QuantizationError",
    "ScalarAffineEncoding",
    "ScalarAffineMetadata",
    "decode_additive_vectors",
    "decode_scalar_affine",
    "encode_additive_vectors",
    "encode_scalar_affine",
]
