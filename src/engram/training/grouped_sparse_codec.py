"""Strict packed artifact for a low-rank router and grouped sparse SwiGLU.

The format is intentionally simple and auditable.  The router is stored as
two row-scaled signed-Q4 factors.  Each independently addressable MLP group is
one cache-line-aligned bundle containing signed-Q4 gate, up, and transposed
down rows plus the FP16 scales required to decode them.  Loading validates all
dimensions, padding, scales, codes, and offsets before exposing the artifact.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.training.switch_expert_boundaries import (
    PackedQ4Rows,
    _decode_symmetric_q4_rows,
    _pack_symmetric_q4_rows,
)


_MAGIC = b"ENGGRQ41"
_VERSION = 1
_HEADER_BYTES = 64
_HEADER = struct.Struct("<8s12I")
_Q4_CODEC = 1
_PERMUTATION_CODEC = 1  # little-endian uint32


def _align(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("alignment operands must be non-negative/positive")
    return ((value + alignment - 1) // alignment) * alignment


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _matrix(values: ArrayLike, name: str) -> NDArray[np.float32]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()  # type: ignore[union-attr]
    try:
        result = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if result.ndim != 2 or not result.shape[0] or not result.shape[1]:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def grouped_sparse_traffic(
    hidden_size: int,
    intermediate_size: int,
    *,
    selected_records: int,
    router_rank: int,
    group_size: int,
    cache_line_bytes: int = 64,
) -> dict[str, int | float | bool | str]:
    """Return exact cold bytes for the serialized grouped-Q4 layout.

    ``fraction_of_dense_q4`` deliberately retains the project's conservative
    code-only dense-Q4 denominator while the sparse numerator includes every
    scale and metadata read.  A second denominator that includes dense scale
    tables is reported separately for diagnostics.
    """

    for value, name in (
        (hidden_size, "hidden_size"),
        (intermediate_size, "intermediate_size"),
        (selected_records, "selected_records"),
        (router_rank, "router_rank"),
        (group_size, "group_size"),
        (cache_line_bytes, "cache_line_bytes"),
    ):
        _positive_integer(value, name)
    if cache_line_bytes < _HEADER_BYTES or cache_line_bytes % _HEADER_BYTES:
        raise ValueError("cache_line_bytes must be a positive multiple of 64")
    if intermediate_size % group_size or selected_records % group_size:
        raise ValueError("intermediate and selected records must be group aligned")
    if selected_records > intermediate_size:
        raise ValueError("selected_records cannot exceed intermediate_size")
    if router_rank > min(hidden_size, intermediate_size // group_size):
        raise ValueError("router_rank exceeds the factor dimensions")
    if intermediate_size > np.iinfo(np.uint32).max:
        raise ValueError("intermediate_size cannot be represented by the artifact")

    groups = intermediate_size // group_size
    selected_groups = selected_records // group_size
    one_matrix_group_q4_bytes = (group_size * hidden_size + 1) // 2
    group_q4_code_bytes = 3 * one_matrix_group_q4_bytes
    group_fp16_scale_bytes = 3 * group_size * 2
    group_payload_bytes = group_q4_code_bytes + group_fp16_scale_bytes
    group_block_bytes = _align(group_payload_bytes, cache_line_bytes)

    # Factors are transposed before packing so each latent dimension/router
    # output owns a scale.  These byte counts exactly match PackedQ4Rows.
    router_input_q4_code_bytes = (router_rank * hidden_size + 1) // 2
    router_input_fp16_scale_bytes = router_rank * 2
    router_output_q4_code_bytes = (groups * router_rank + 1) // 2
    router_output_fp16_scale_bytes = groups * 2
    router_bias_fp16_bytes = groups * 2
    router_nonlinear_scale_fp16_bytes = router_rank * 2
    router_payload_bytes = (
        router_input_q4_code_bytes
        + router_output_q4_code_bytes
        + router_input_fp16_scale_bytes
        + router_output_fp16_scale_bytes
        + router_bias_fp16_bytes
        + router_nonlinear_scale_fp16_bytes
    )
    router_block_bytes = _align(router_payload_bytes, cache_line_bytes)
    header_block_bytes = _align(_HEADER_BYTES, cache_line_bytes)
    selected_group_id_bytes = selected_groups * 2
    selected_group_id_block_bytes = _align(selected_group_id_bytes, cache_line_bytes)
    selected_group_block_bytes = selected_groups * group_block_bytes
    total_cold_bytes = (
        header_block_bytes
        + router_block_bytes
        + selected_group_id_block_bytes
        + selected_group_block_bytes
    )

    permutation_uint32_bytes = intermediate_size * 4
    permutation_block_bytes = _align(permutation_uint32_bytes, cache_line_bytes)
    serialized_artifact_bytes = (
        header_block_bytes
        + router_block_bytes
        + permutation_block_bytes
        + groups * group_block_bytes
    )
    dense_q4_code_bytes = (3 * hidden_size * intermediate_size + 1) // 2
    dense_q4_fp16_scale_bytes = 3 * intermediate_size * 2
    dense_q4_payload_bytes = dense_q4_code_bytes + dense_q4_fp16_scale_bytes
    conservative_fraction = total_cold_bytes / dense_q4_code_bytes
    payload_fraction = total_cold_bytes / dense_q4_payload_bytes
    return {
        "layout": "grouped_q4_low_rank_router_v1",
        "cache_line_bytes": cache_line_bytes,
        "header_cache_aligned_bytes": header_block_bytes,
        "router_group_size": group_size,
        "router_output_groups": groups,
        "selected_groups": selected_groups,
        "record_group_q4_bytes": group_q4_code_bytes,
        "record_group_fp16_scale_bytes": group_fp16_scale_bytes,
        "record_group_payload_bytes": group_payload_bytes,
        "record_group_cache_aligned_bytes": group_block_bytes,
        "selected_group_q4_code_bytes": selected_groups * group_q4_code_bytes,
        "selected_group_fp16_scale_bytes": selected_groups * group_fp16_scale_bytes,
        "selected_group_cache_padding_bytes": selected_groups
        * (group_block_bytes - group_payload_bytes),
        "selected_group_cache_aligned_bytes": selected_group_block_bytes,
        # Compatibility key: unlike the previous projection, this now includes
        # the cache-line cost of codes, scales, and per-group padding.
        "selected_record_q4_bytes": selected_group_block_bytes,
        "router_input_q4_bytes": router_input_q4_code_bytes,
        "router_input_fp16_scale_bytes": router_input_fp16_scale_bytes,
        "router_output_q4_bytes": router_output_q4_code_bytes,
        "router_output_fp16_scale_bytes": router_output_fp16_scale_bytes,
        "router_factor_q4_bytes": (
            router_input_q4_code_bytes + router_output_q4_code_bytes
        ),
        "router_factor_fp16_scale_bytes": (
            router_input_fp16_scale_bytes + router_output_fp16_scale_bytes
        ),
        "router_bias_fp16_bytes": router_bias_fp16_bytes,
        "router_nonlinear_scale_fp16_bytes": router_nonlinear_scale_fp16_bytes,
        "router_payload_bytes": router_payload_bytes,
        "router_cache_padding_bytes": router_block_bytes - router_payload_bytes,
        "router_cache_aligned_bytes": router_block_bytes,
        "uint16_group_id_bytes": selected_group_id_bytes,
        "selected_group_id_cache_aligned_bytes": selected_group_id_block_bytes,
        "permutation_uint32_bytes": permutation_uint32_bytes,
        "permutation_cache_aligned_bytes": permutation_block_bytes,
        "total_bytes": total_cold_bytes,
        "total_cold_bytes": total_cold_bytes,
        "dense_q4_bytes": dense_q4_code_bytes,
        "dense_q4_code_bytes": dense_q4_code_bytes,
        "dense_q4_fp16_scale_bytes": dense_q4_fp16_scale_bytes,
        "dense_q4_payload_bytes": dense_q4_payload_bytes,
        "fraction_of_dense_q4": conservative_fraction,
        "fraction_of_dense_q4_payload": payload_fraction,
        "passes_45_percent_traffic_gate": conservative_fraction <= 0.45,
        "serialized_artifact_bytes": serialized_artifact_bytes,
    }


@dataclass(frozen=True)
class PackedGroupedRouter:
    input_factor_t: PackedQ4Rows
    output_factor_t: PackedQ4Rows
    bias: NDArray[np.float16]
    nonlinear_scale: NDArray[np.float16]


@dataclass(frozen=True)
class PackedGroupedMLPGroup:
    gate: PackedQ4Rows
    up: PackedQ4Rows
    down_t: PackedQ4Rows


@dataclass(frozen=True)
class LoadedGroupedSparseArtifact:
    router: PackedGroupedRouter
    groups: tuple[PackedGroupedMLPGroup, ...]
    permutation: NDArray[np.uint32]
    hidden_size: int
    intermediate_size: int
    group_size: int
    selected_groups: int
    router_rank: int
    cache_line_bytes: int
    header_block_bytes: int
    router_block_bytes: int
    permutation_block_bytes: int
    group_block_bytes: int
    group_offsets: tuple[int, ...]


def _fp16_vector(values: ArrayLike, length: int, name: str) -> NDArray[np.float16]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()  # type: ignore[union-attr]
    try:
        vector = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {length} finite values")
    stored = np.ascontiguousarray(vector.astype("<f2"))
    if not np.all(np.isfinite(stored)):
        raise ValueError(f"{name} cannot be represented as FP16")
    return stored


def _permutation(values: ArrayLike | None, size: int) -> NDArray[np.uint32]:
    if values is None:
        return np.arange(size, dtype=np.uint32)
    array = np.asarray(values)
    if array.shape != (size,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("permutation must be a one-dimensional integer array")
    converted = array.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(converted), np.arange(size, dtype=np.int64)):
        raise ValueError("permutation must contain every record exactly once")
    return np.ascontiguousarray(converted.astype("<u4"))


def _router_payload(router: PackedGroupedRouter) -> bytes:
    return b"".join(
        (
            router.input_factor_t.packed.tobytes(),
            router.output_factor_t.packed.tobytes(),
            router.input_factor_t.scales.astype("<f2", copy=False).tobytes(),
            router.output_factor_t.scales.astype("<f2", copy=False).tobytes(),
            router.bias.astype("<f2", copy=False).tobytes(),
            router.nonlinear_scale.astype("<f2", copy=False).tobytes(),
        )
    )


def _group_payload(group: PackedGroupedMLPGroup) -> bytes:
    return b"".join(
        (
            group.gate.packed.tobytes(),
            group.up.packed.tobytes(),
            group.down_t.packed.tobytes(),
            group.gate.scales.astype("<f2", copy=False).tobytes(),
            group.up.scales.astype("<f2", copy=False).tobytes(),
            group.down_t.scales.astype("<f2", copy=False).tobytes(),
        )
    )


def save_grouped_sparse_artifact(
    path: str | Path,
    gate: ArrayLike,
    up: ArrayLike,
    down: ArrayLike,
    router_input: ArrayLike,
    router_output: ArrayLike,
    router_bias: ArrayLike,
    *,
    group_size: int,
    selected_records: int,
    router_nonlinear_scale: ArrayLike | None = None,
    permutation: ArrayLike | None = None,
    cache_line_bytes: int = 64,
) -> Path:
    """Quantize and atomically write one grouped sparse MLP layer."""

    gate_matrix = _matrix(gate, "gate")
    up_matrix = _matrix(up, "up")
    down_matrix = _matrix(down, "down")
    if gate_matrix.shape != up_matrix.shape:
        raise ValueError("gate and up shapes must match")
    intermediate, hidden = gate_matrix.shape
    if down_matrix.shape != (hidden, intermediate):
        raise ValueError("down must have shape [hidden, intermediate]")
    input_factor = _matrix(router_input, "router_input")
    output_factor = _matrix(router_output, "router_output")
    if input_factor.shape[0] != hidden:
        raise ValueError("router_input hidden dimension does not match the MLP")
    rank = input_factor.shape[1]
    if intermediate % group_size:
        raise ValueError("group_size must divide the intermediate dimension")
    group_count = intermediate // group_size
    if output_factor.shape != (rank, group_count):
        raise ValueError("router_output must have shape [rank, group_count]")
    traffic = grouped_sparse_traffic(
        hidden,
        intermediate,
        selected_records=selected_records,
        router_rank=rank,
        group_size=group_size,
        cache_line_bytes=cache_line_bytes,
    )
    bias = _fp16_vector(router_bias, group_count, "router_bias")
    nonlinear = _fp16_vector(
        np.zeros(rank, dtype=np.float32)
        if router_nonlinear_scale is None
        else router_nonlinear_scale,
        rank,
        "router_nonlinear_scale",
    )
    order = _permutation(permutation, intermediate)
    router = PackedGroupedRouter(
        _pack_symmetric_q4_rows(input_factor.T),
        _pack_symmetric_q4_rows(output_factor.T),
        bias,
        nonlinear,
    )
    groups = tuple(
        PackedGroupedMLPGroup(
            _pack_symmetric_q4_rows(gate_matrix[start : start + group_size]),
            _pack_symmetric_q4_rows(up_matrix[start : start + group_size]),
            _pack_symmetric_q4_rows(
                down_matrix[:, start : start + group_size].T
            ),
        )
        for start in range(0, intermediate, group_size)
    )
    router_payload = _router_payload(router)
    group_payloads = [_group_payload(group) for group in groups]
    router_block_bytes = int(traffic["router_cache_aligned_bytes"])
    group_block_bytes = int(traffic["record_group_cache_aligned_bytes"])
    permutation_block_bytes = int(traffic["permutation_cache_aligned_bytes"])
    header_block_bytes = int(traffic["header_cache_aligned_bytes"])
    if len(router_payload) != traffic["router_payload_bytes"]:
        raise AssertionError("router payload differs from traffic accounting")
    if any(len(payload) != traffic["record_group_payload_bytes"] for payload in group_payloads):
        raise AssertionError("group payload differs from traffic accounting")

    header = bytearray(header_block_bytes)
    _HEADER.pack_into(
        header,
        0,
        _MAGIC,
        _VERSION,
        hidden,
        intermediate,
        group_size,
        selected_records // group_size,
        rank,
        router_block_bytes,
        permutation_block_bytes,
        group_block_bytes,
        cache_line_bytes,
        _Q4_CODEC,
        _PERMUTATION_CODEC,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(header)
            handle.write(router_payload)
            handle.write(b"\0" * (router_block_bytes - len(router_payload)))
            permutation_payload = order.astype("<u4", copy=False).tobytes()
            handle.write(permutation_payload)
            handle.write(
                b"\0" * (permutation_block_bytes - len(permutation_payload))
            )
            for payload in group_payloads:
                handle.write(payload)
                handle.write(b"\0" * (group_block_bytes - len(payload)))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    if destination.stat().st_size != traffic["serialized_artifact_bytes"]:
        raise AssertionError("serialized grouped artifact has an unexpected size")
    return destination


def load_grouped_sparse_artifact(path: str | Path) -> LoadedGroupedSparseArtifact:
    """Strictly validate and load a packed grouped sparse MLP layer."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read grouped sparse artifact {source}") from exc
    if len(payload) < _HEADER_BYTES:
        raise ValueError("grouped sparse artifact is shorter than its header")
    (
        magic,
        version,
        hidden,
        intermediate,
        group_size,
        selected_groups,
        rank,
        router_block_bytes,
        permutation_block_bytes,
        group_block_bytes,
        cache_line_bytes,
        q4_codec,
        permutation_codec,
    ) = _HEADER.unpack_from(payload)
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("grouped sparse artifact magic/version mismatch")
    if q4_codec != _Q4_CODEC or permutation_codec != _PERMUTATION_CODEC:
        raise ValueError("grouped sparse artifact codec metadata is unsupported")
    traffic = grouped_sparse_traffic(
        hidden,
        intermediate,
        selected_records=selected_groups * group_size,
        router_rank=rank,
        group_size=group_size,
        cache_line_bytes=cache_line_bytes,
    )
    expected_blocks = (
        int(traffic["router_cache_aligned_bytes"]),
        int(traffic["permutation_cache_aligned_bytes"]),
        int(traffic["record_group_cache_aligned_bytes"]),
    )
    if (router_block_bytes, permutation_block_bytes, group_block_bytes) != expected_blocks:
        raise ValueError("grouped sparse artifact block sizes do not match dimensions")
    if len(payload) != traffic["serialized_artifact_bytes"]:
        raise ValueError("grouped sparse artifact length does not match its header")
    header_block_bytes = int(traffic["header_cache_aligned_bytes"])
    if any(payload[_HEADER.size : header_block_bytes]):
        raise ValueError("grouped sparse header padding is non-zero")

    group_count = intermediate // group_size
    input_code_bytes = int(traffic["router_input_q4_bytes"])
    output_code_bytes = int(traffic["router_output_q4_bytes"])
    cursor = header_block_bytes

    def bytes_array(count: int) -> NDArray[np.uint8]:
        nonlocal cursor
        result = np.frombuffer(payload, dtype=np.uint8, count=count, offset=cursor).copy()
        cursor += count
        return result

    def fp16_array(count: int) -> NDArray[np.float16]:
        nonlocal cursor
        result = np.frombuffer(payload, dtype="<f2", count=count, offset=cursor).copy()
        cursor += count * 2
        return result

    router_start = cursor
    input_codes = bytes_array(input_code_bytes)
    output_codes = bytes_array(output_code_bytes)
    input_scales = fp16_array(rank)
    output_scales = fp16_array(group_count)
    bias = fp16_array(group_count)
    nonlinear = fp16_array(rank)
    router_payload_end = router_start + int(traffic["router_payload_bytes"])
    if cursor != router_payload_end:
        raise AssertionError("internal grouped router offset mismatch")
    if any(payload[cursor : router_start + router_block_bytes]):
        raise ValueError("grouped sparse router padding is non-zero")
    cursor = router_start + router_block_bytes
    router = PackedGroupedRouter(
        PackedQ4Rows(input_codes, input_scales, rank, hidden),
        PackedQ4Rows(output_codes, output_scales, group_count, rank),
        bias,
        nonlinear,
    )
    _decode_symmetric_q4_rows(router.input_factor_t)
    _decode_symmetric_q4_rows(router.output_factor_t)
    if not np.all(np.isfinite(bias)) or not np.all(np.isfinite(nonlinear)):
        raise ValueError("grouped sparse router metadata contains non-finite values")

    permutation_start = cursor
    permutation = np.frombuffer(
        payload, dtype="<u4", count=intermediate, offset=cursor
    ).copy()
    cursor += intermediate * 4
    if not np.array_equal(
        np.sort(permutation.astype(np.int64)), np.arange(intermediate, dtype=np.int64)
    ):
        raise ValueError("grouped sparse artifact permutation is invalid")
    if any(payload[cursor : permutation_start + permutation_block_bytes]):
        raise ValueError("grouped sparse permutation padding is non-zero")
    cursor = permutation_start + permutation_block_bytes

    one_code_bytes = (group_size * hidden + 1) // 2
    raw_group_bytes = int(traffic["record_group_payload_bytes"])
    groups: list[PackedGroupedMLPGroup] = []
    offsets: list[int] = []
    for _ in range(group_count):
        offset = cursor
        offsets.append(offset)
        gate_codes = bytes_array(one_code_bytes)
        up_codes = bytes_array(one_code_bytes)
        down_codes = bytes_array(one_code_bytes)
        gate_scales = fp16_array(group_size)
        up_scales = fp16_array(group_size)
        down_scales = fp16_array(group_size)
        if cursor != offset + raw_group_bytes:
            raise AssertionError("internal grouped MLP offset mismatch")
        if any(payload[cursor : offset + group_block_bytes]):
            raise ValueError("grouped sparse MLP padding is non-zero")
        cursor = offset + group_block_bytes
        group = PackedGroupedMLPGroup(
            PackedQ4Rows(gate_codes, gate_scales, group_size, hidden),
            PackedQ4Rows(up_codes, up_scales, group_size, hidden),
            PackedQ4Rows(down_codes, down_scales, group_size, hidden),
        )
        _decode_symmetric_q4_rows(group.gate)
        _decode_symmetric_q4_rows(group.up)
        _decode_symmetric_q4_rows(group.down_t)
        groups.append(group)
    if cursor != len(payload):
        raise AssertionError("internal grouped artifact length mismatch")
    if any(offset % cache_line_bytes for offset in offsets):
        raise ValueError("grouped sparse MLP offset is not cache-line aligned")
    return LoadedGroupedSparseArtifact(
        router,
        tuple(groups),
        permutation,
        hidden,
        intermediate,
        group_size,
        selected_groups,
        rank,
        cache_line_bytes,
        header_block_bytes,
        router_block_bytes,
        permutation_block_bytes,
        group_block_bytes,
        tuple(offsets),
    )


def decode_grouped_sparse_artifact(
    artifact: LoadedGroupedSparseArtifact,
) -> dict[str, NDArray[np.float32] | NDArray[np.uint32]]:
    """Decode a validated artifact into its logical matrices."""

    gate = np.concatenate(
        [_decode_symmetric_q4_rows(group.gate) for group in artifact.groups], axis=0
    )
    up = np.concatenate(
        [_decode_symmetric_q4_rows(group.up) for group in artifact.groups], axis=0
    )
    down = np.concatenate(
        [_decode_symmetric_q4_rows(group.down_t).T for group in artifact.groups],
        axis=1,
    )
    router_input = _decode_symmetric_q4_rows(artifact.router.input_factor_t).T
    router_output = _decode_symmetric_q4_rows(artifact.router.output_factor_t).T
    return {
        "gate": np.ascontiguousarray(gate),
        "up": np.ascontiguousarray(up),
        "down": np.ascontiguousarray(down),
        "router_input": np.ascontiguousarray(router_input),
        "router_output": np.ascontiguousarray(router_output),
        "router_bias": artifact.router.bias.astype(np.float32),
        "router_nonlinear_scale": artifact.router.nonlinear_scale.astype(np.float32),
        "permutation": artifact.permutation.copy(),
    }


def _silu(values: NDArray[np.float32]) -> NDArray[np.float32]:
    sigmoid = np.empty_like(values)
    positive = values >= 0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    sigmoid[~positive] = exponent / (1.0 + exponent)
    return values * sigmoid


def grouped_sparse_forward(
    artifact: LoadedGroupedSparseArtifact, hidden: ArrayLike
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Execute the hard grouped path using only validated decoded payloads."""

    states = _matrix(hidden, "hidden")
    if states.shape[1] != artifact.hidden_size:
        raise ValueError("hidden width does not match the grouped artifact")
    decoded = decode_grouped_sparse_artifact(artifact)
    latent = states @ decoded["router_input"]  # type: ignore[operator]
    features = latent + decoded["router_nonlinear_scale"] * _silu(latent)  # type: ignore[operator]
    scores = features @ decoded["router_output"] + decoded["router_bias"]  # type: ignore[operator]
    selected = np.argsort(-scores, axis=1, kind="stable")[:, : artifact.selected_groups]
    output = np.zeros((len(states), artifact.hidden_size), dtype=np.float32)
    for row, state in enumerate(states):
        for group_id in selected[row]:
            group = artifact.groups[int(group_id)]
            gate = _decode_symmetric_q4_rows(group.gate)
            up = _decode_symmetric_q4_rows(group.up)
            down_t = _decode_symmetric_q4_rows(group.down_t)
            activation = _silu(state @ gate.T) * (state @ up.T)
            output[row] += activation @ down_t
    return np.ascontiguousarray(output), selected.astype(np.int64, copy=False)


__all__ = [
    "LoadedGroupedSparseArtifact",
    "PackedGroupedMLPGroup",
    "PackedGroupedRouter",
    "decode_grouped_sparse_artifact",
    "grouped_sparse_forward",
    "grouped_sparse_traffic",
    "load_grouped_sparse_artifact",
    "save_grouped_sparse_artifact",
]
