"""Quantized-only semantic-memory reference runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.semantic.ivf import JointKeyIVFProbeIndex
from engram.semantic.quantization import (
    AdditiveVectorEncoding,
    AdditiveVectorMetadata,
    QuantizationError,
    ScalarAffineEncoding,
    ScalarAffineMetadata,
    encode_additive_vectors,
    encode_scalar_affine,
)
from engram.semantic.swiglu import silu
from engram.utils import atomic_json


@dataclass(frozen=True)
class CompressedSemanticRead:
    output: NDArray[np.float64]
    active_records: int
    candidate_records: int
    estimated_bytes_read: int
    indices: NDArray[np.int64]
    candidate_indices: NDArray[np.int64]
    candidate_proxy_scores: NDArray[np.float64]
    candidate_activations: NDArray[np.float64]
    candidate_exact_scores: NDArray[np.float64]
    probed_clusters: int
    proxy_records: int
    index_bytes_read: int


def _readonly_array(
    value: ArrayLike, *, name: str, dtype: np.dtype | type | None = None
) -> np.ndarray:
    try:
        # asanyarray preserves np.memmap loaded package sections.
        result = np.asanyarray(value, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise QuantizationError(f"{name} must be an array") from error
    if not np.all(np.isfinite(result)):
        raise QuantizationError(f"{name} must contain only finite values")
    result = result.view()
    result.flags.writeable = False
    return result


class CompressedSemanticLayer:
    """Semantic layer backed only by scalar and additive quantization data.

    Full decoded gate, up, and value matrices are never retained. Candidate
    proxy scores are computed algebraically from scalar-affine codes. Only the
    selected candidate rows are decoded transiently for exact reranking in the
    quantized representation.
    """

    def __init__(
        self,
        *,
        gate_codes: ArrayLike,
        gate_offsets: ArrayLike,
        gate_scales: ArrayLike,
        gate_metadata: ScalarAffineMetadata,
        up_codes: ArrayLike,
        up_offsets: ArrayLike,
        up_scales: ArrayLike,
        up_metadata: ScalarAffineMetadata,
        value_codes: ArrayLike,
        value_codebooks: ArrayLike,
        value_metadata: AdditiveVectorMetadata,
        ivf_index: JointKeyIVFProbeIndex | None = None,
    ) -> None:
        gate_metadata.validate()
        up_metadata.validate()
        value_metadata.validate()
        if gate_metadata.shape != up_metadata.shape:
            raise QuantizationError("gate and up codec shapes differ")
        if value_metadata.shape != gate_metadata.shape:
            raise QuantizationError("value and key codec shapes differ")

        self.gate_metadata = gate_metadata
        self.up_metadata = up_metadata
        self.value_metadata = value_metadata
        self.ivf_index = ivf_index
        self.gate_codes = _readonly_array(gate_codes, name="gate_codes")
        self.up_codes = _readonly_array(up_codes, name="up_codes")
        self.value_codes = _readonly_array(value_codes, name="value_codes")
        self.value_codebooks = _readonly_array(
            value_codebooks, name="value_codebooks"
        )
        self.gate_offsets = _readonly_array(
            gate_offsets, name="gate_offsets", dtype=np.float32
        )
        self.gate_scales = _readonly_array(
            gate_scales, name="gate_scales", dtype=np.float32
        )
        self.up_offsets = _readonly_array(
            up_offsets, name="up_offsets", dtype=np.float32
        )
        self.up_scales = _readonly_array(
            up_scales, name="up_scales", dtype=np.float32
        )
        self._validate_arrays()
        if self.ivf_index is not None and (
            self.ivf_index.records != self.records
            or self.ivf_index.hidden_size != self.hidden_size
        ):
            raise QuantizationError("IVF index dimensions do not match semantic keys")

    @classmethod
    def from_encodings(
        cls,
        gate: ScalarAffineEncoding,
        up: ScalarAffineEncoding,
        values: AdditiveVectorEncoding,
    ) -> "CompressedSemanticLayer":
        return cls(
            gate_codes=gate.codes,
            gate_offsets=np.asarray(gate.metadata.offsets, dtype=np.float32),
            gate_scales=np.asarray(gate.metadata.scales, dtype=np.float32),
            gate_metadata=gate.metadata,
            up_codes=up.codes,
            up_offsets=np.asarray(up.metadata.offsets, dtype=np.float32),
            up_scales=np.asarray(up.metadata.scales, dtype=np.float32),
            up_metadata=up.metadata,
            value_codes=values.codes,
            value_codebooks=values.codebooks,
            value_metadata=values.metadata,
        )

    @classmethod
    def compress(
        cls,
        gate_keys: ArrayLike,
        up_keys: ArrayLike,
        values: ArrayLike,
        *,
        key_bits: int = 8,
        value_codebooks: int = 2,
        value_codebook_size: int = 16,
        iterations: int = 20,
    ) -> "CompressedSemanticLayer":
        gate_array = np.asarray(gate_keys)
        up_array = np.asarray(up_keys)
        value_array = np.asarray(values)
        if (
            gate_array.ndim != 2
            or gate_array.shape != up_array.shape
            or value_array.shape != gate_array.shape
        ):
            raise QuantizationError(
                "gate, up, and value arrays must share shape [records, hidden]"
            )
        return cls.from_encodings(
            encode_scalar_affine(gate_array, bits=key_bits),
            encode_scalar_affine(up_array, bits=key_bits),
            encode_additive_vectors(
                value_array,
                num_codebooks=value_codebooks,
                codebook_size=min(value_codebook_size, gate_array.shape[0]),
                iterations=iterations,
            ),
        )

    @property
    def records(self) -> int:
        return self.gate_metadata.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.gate_metadata.shape[1]

    @property
    def compressed_bytes(self) -> int:
        return int(
            self.gate_codes.nbytes
            + self.gate_offsets.nbytes
            + self.gate_scales.nbytes
            + self.up_codes.nbytes
            + self.up_offsets.nbytes
            + self.up_scales.nbytes
            + self.value_codes.nbytes
            + self.value_codebooks.nbytes
            + (self.ivf_index.total_bytes if self.ivf_index is not None else 0)
        )

    def _validate_arrays(self) -> None:
        for name, codes, metadata in (
            ("gate", self.gate_codes, self.gate_metadata),
            ("up", self.up_codes, self.up_metadata),
        ):
            if codes.dtype != np.dtype(metadata.storage_dtype):
                raise QuantizationError(
                    f"{name}_codes must have dtype {metadata.storage_dtype}"
                )
            if codes.shape != metadata.shape:
                raise QuantizationError(
                    f"{name}_codes shape {codes.shape} does not match {metadata.shape}"
                )
            if codes.size and int(np.max(codes)) > (1 << metadata.bits) - 1:
                raise QuantizationError(f"a {name} code exceeds its configured bit range")
            offsets = getattr(self, f"{name}_offsets")
            scales = getattr(self, f"{name}_scales")
            if offsets.shape != (self.hidden_size,) or scales.shape != (
                self.hidden_size,
            ):
                raise QuantizationError(
                    f"{name} offsets and scales must have shape [{self.hidden_size}]"
                )
            if not np.array_equal(
                offsets, np.asarray(metadata.offsets, dtype=np.float32)
            ):
                raise QuantizationError(
                    f"{name} offsets disagree with codec metadata"
                )
            if not np.array_equal(
                scales, np.asarray(metadata.scales, dtype=np.float32)
            ):
                raise QuantizationError(
                    f"{name} scales disagree with codec metadata"
                )
            if np.any(scales <= 0.0):
                raise QuantizationError(f"{name} scales must be positive")

        expected_code_shape = (self.records, self.value_metadata.num_codebooks)
        if self.value_codes.dtype != np.dtype(self.value_metadata.code_dtype):
            raise QuantizationError(
                f"value_codes must have dtype {self.value_metadata.code_dtype}"
            )
        if self.value_codes.shape != expected_code_shape:
            raise QuantizationError(
                f"value_codes shape {self.value_codes.shape} does not match {expected_code_shape}"
            )
        expected_codebook_shape = (
            self.value_metadata.num_codebooks,
            self.value_metadata.codebook_size,
            self.hidden_size,
        )
        if self.value_codebooks.dtype != np.dtype(
            self.value_metadata.codebook_dtype
        ):
            raise QuantizationError(
                f"value_codebooks must have dtype {self.value_metadata.codebook_dtype}"
            )
        if self.value_codebooks.shape != expected_codebook_shape:
            raise QuantizationError(
                "value_codebooks shape does not match value codec metadata"
            )
        if self.value_codes.size and int(np.max(self.value_codes)) >= (
            self.value_metadata.codebook_size
        ):
            raise QuantizationError("a value code is outside its codebook")

    def _hidden(self, hidden: ArrayLike) -> NDArray[np.float64]:
        result = np.asarray(hidden, dtype=np.float64)
        if result.shape != (self.hidden_size,):
            raise ValueError(
                f"hidden state must have shape ({self.hidden_size},), got {result.shape}"
            )
        if not np.all(np.isfinite(result)):
            raise ValueError("hidden state must contain only finite values")
        return result

    @staticmethod
    def _proxy_alignment(
        hidden: NDArray[np.float64],
        codes: np.ndarray,
        offsets: np.ndarray,
        scales: np.ndarray,
    ) -> NDArray[np.float64]:
        offset = offsets.astype(np.float64, copy=False)
        scale = scales.astype(np.float64, copy=False)
        dot = codes @ (scale * hidden) + float(offset @ hidden)
        norm_squared = np.full(codes.shape[0], float(offset @ offset))
        norm_squared += 2.0 * (codes @ (offset * scale))
        norm_squared += np.einsum(
            "ij,j,ij->i",
            codes,
            scale * scale,
            codes,
            dtype=np.float64,
            optimize=True,
        )
        denominator = np.sqrt(np.maximum(norm_squared, 0.0)) * float(
            np.linalg.norm(hidden)
        )
        return np.divide(
            dot,
            denominator,
            out=np.zeros_like(dot, dtype=np.float64),
            where=denominator > 0.0,
        )

    @staticmethod
    def _decode_keys(
        indices: NDArray[np.int64],
        codes: np.ndarray,
        offsets: np.ndarray,
        scales: np.ndarray,
    ) -> NDArray[np.float32]:
        return (
            offsets[None, :]
            + codes[indices].astype(np.float32) * scales[None, :]
        ).astype(np.float32, copy=False)

    def decode_gate(self, indices: ArrayLike | None = None) -> NDArray[np.float32]:
        selected = self._indices(indices)
        return self._decode_keys(
            selected, self.gate_codes, self.gate_offsets, self.gate_scales
        )

    def decode_up(self, indices: ArrayLike | None = None) -> NDArray[np.float32]:
        selected = self._indices(indices)
        return self._decode_keys(
            selected, self.up_codes, self.up_offsets, self.up_scales
        )

    def decode_values(self, indices: ArrayLike | None = None) -> NDArray[np.float32]:
        selected = self._indices(indices)
        result = np.zeros((selected.size, self.hidden_size), dtype=np.float32)
        for stage in range(self.value_metadata.num_codebooks):
            result += self.value_codebooks[
                stage, self.value_codes[selected, stage]
            ]
        return result

    def _indices(self, indices: ArrayLike | None) -> NDArray[np.int64]:
        if indices is None:
            return np.arange(self.records, dtype=np.int64)
        result = np.asarray(indices)
        if result.ndim != 1 or not np.issubdtype(result.dtype, np.integer):
            raise ValueError("indices must be a one-dimensional integer array")
        result = result.astype(np.int64, copy=False)
        if np.any(result < 0) or np.any(result >= self.records):
            raise IndexError("compressed semantic index is out of range")
        return result

    def read(
        self,
        hidden: ArrayLike,
        *,
        candidate_count: int,
        top_k: int,
        probes: int | None = None,
    ) -> CompressedSemanticRead:
        state = self._hidden(hidden)
        for value, name in ((candidate_count, "candidate_count"), (top_k, "top_k")):
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if top_k > candidate_count:
            raise ValueError("top_k must not exceed candidate_count")

        count = min(int(candidate_count), self.records)
        index_bytes = 0
        probed_clusters = 0
        if self.ivf_index is not None and float(np.linalg.norm(state)) > 0.0:
            probe = self.ivf_index.probe(
                state,
                probes=1 if probes is None else probes,
                minimum_records=count,
            )
            proxy_pool = probe.indices.astype(np.int64, copy=False)
            gate_alignment = self._proxy_alignment(
                state,
                self.gate_codes[proxy_pool],
                self.gate_offsets,
                self.gate_scales,
            )
            up_alignment = self._proxy_alignment(
                state,
                self.up_codes[proxy_pool],
                self.up_offsets,
                self.up_scales,
            )
            pool_scores = np.maximum(gate_alignment, 0.0) * np.abs(up_alignment)
            order = np.lexsort((proxy_pool, -pool_scores))[:count]
            candidate_indices = proxy_pool[order]
            candidate_proxy_scores = pool_scores[order]
            proxy_records = int(proxy_pool.size)
            probed_clusters = int(probe.clusters.size)
            index_bytes = probe.index_bytes_read
            proxy_bytes = index_bytes + proxy_records * self.hidden_size * (
                self.gate_codes.dtype.itemsize + self.up_codes.dtype.itemsize
            ) + 4 * self.hidden_size * np.dtype(np.float32).itemsize
        elif self.ivf_index is not None:
            # Preserve the brute-force stable source-index result without
            # reading centroids or key rows for a zero query.
            candidate_indices = np.arange(count, dtype=np.int64)
            candidate_proxy_scores = np.zeros(count, dtype=np.float64)
            proxy_records = count
            proxy_bytes = 0
        else:
            gate_alignment = self._proxy_alignment(
                state, self.gate_codes, self.gate_offsets, self.gate_scales
            )
            up_alignment = self._proxy_alignment(
                state, self.up_codes, self.up_offsets, self.up_scales
            )
            proxy_scores = np.maximum(gate_alignment, 0.0) * np.abs(up_alignment)
            candidate_indices = np.argsort(-proxy_scores, kind="stable")[:count].astype(
                np.int64, copy=False
            )
            candidate_proxy_scores = proxy_scores[candidate_indices]
            proxy_records = self.records
            key_parameter_bytes = 4 * self.hidden_size * np.dtype(np.float32).itemsize
            proxy_bytes = (
                self.gate_codes.nbytes + self.up_codes.nbytes
            ) + key_parameter_bytes

        candidate_gate = self.decode_gate(candidate_indices)
        candidate_up = self.decode_up(candidate_indices)
        candidate_values = self.decode_values(candidate_indices)
        gate_logits = candidate_gate.astype(np.float64) @ state
        up_logits = candidate_up.astype(np.float64) @ state
        activations = np.asarray(silu(gate_logits) * up_logits, dtype=np.float64)
        exact_scores = np.abs(activations) * np.linalg.norm(
            candidate_values.astype(np.float64), axis=1
        )
        selected_count = min(int(top_k), count)
        local_order = np.argsort(-exact_scores, kind="stable")[:selected_count]
        selected = candidate_indices[local_order]
        output = activations[local_order] @ candidate_values[local_order]

        rerank_key_bytes = count * self.hidden_size * (
            self.gate_codes.dtype.itemsize + self.up_codes.dtype.itemsize
        )
        value_bytes = count * self.value_metadata.num_codebooks * self.value_codes.dtype.itemsize
        value_bytes += (
            count
            * self.value_metadata.num_codebooks
            * self.hidden_size
            * self.value_codebooks.dtype.itemsize
        )
        return CompressedSemanticRead(
            output=np.asarray(output, dtype=np.float64),
            active_records=selected_count,
            candidate_records=count,
            estimated_bytes_read=int(
                proxy_bytes + rerank_key_bytes + value_bytes
            ),
            indices=selected,
            candidate_indices=candidate_indices,
            candidate_proxy_scores=candidate_proxy_scores,
            candidate_activations=activations,
            candidate_exact_scores=exact_scores,
            probed_clusters=probed_clusters,
            proxy_records=proxy_records,
            index_bytes_read=index_bytes,
        )

    def save(self, quantized_directory: str | Path) -> Path:
        target = Path(quantized_directory)
        target.mkdir(parents=True, exist_ok=True)
        arrays = {
            "gate_codes": self.gate_codes,
            "gate_offsets": self.gate_offsets,
            "gate_scales": self.gate_scales,
            "up_codes": self.up_codes,
            "up_offsets": self.up_offsets,
            "up_scales": self.up_scales,
            "value_codes": self.value_codes,
            "value_codebooks": self.value_codebooks,
        }
        for name, array in arrays.items():
            np.save(target / f"{name}.npy", array, allow_pickle=False)
        if self.ivf_index is not None:
            self.ivf_index.save(target / "ivf")
        atomic_json(
            target / "codecs.json",
            {
                "gate": self.gate_metadata.to_dict(),
                "up": self.up_metadata.to_dict(),
                "values": self.value_metadata.to_dict(),
            },
        )
        return target

    @classmethod
    def load(
        cls,
        quantized_directory: str | Path,
        *,
        mmap_mode: str | None = "r",
    ) -> "CompressedSemanticLayer":
        source = Path(quantized_directory)
        try:
            metadata = json.loads((source / "codecs.json").read_text(encoding="utf-8"))
            gate_metadata = ScalarAffineMetadata.from_dict(metadata["gate"])
            up_metadata = ScalarAffineMetadata.from_dict(metadata["up"])
            value_metadata = AdditiveVectorMetadata.from_dict(metadata["values"])
            arrays = {
                name: np.load(
                    source / f"{name}.npy",
                    mmap_mode=mmap_mode,
                    allow_pickle=False,
                )
                for name in (
                    "gate_codes",
                    "gate_offsets",
                    "gate_scales",
                    "up_codes",
                    "up_offsets",
                    "up_scales",
                    "value_codes",
                    "value_codebooks",
                )
            }
            ivf_directory = source / "ivf"
            ivf_index = (
                JointKeyIVFProbeIndex.load(ivf_directory, mmap_mode=mmap_mode)
                if ivf_directory.is_dir()
                else None
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise QuantizationError(
                f"cannot load compressed semantic directory: {source}"
            ) from error
        return cls(
            **arrays,
            gate_metadata=gate_metadata,
            up_metadata=up_metadata,
            value_metadata=value_metadata,
            ivf_index=ivf_index,
        )


__all__ = ["CompressedSemanticLayer", "CompressedSemanticRead"]
