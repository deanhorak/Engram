"""Packed top-1 specialized-expert screen at cached SwiGLU boundaries.

The experiment in this module deliberately removes the record-selection
problem.  Training inputs are partitioned into deterministic nearest-centroid
regions and each region owns an independent, narrow SwiGLU.  Only one expert
is read for a token.  Every reported deployment metric is produced after the
expert weights have been symmetrically quantized to signed Q4, serialized,
reloaded, validated, and decoded.

This remains a boundary ceiling experiment: a pass justifies a causal
sequence-level intervention, but is not itself a causal-quality result.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.evaluation.mlp_intervention import _relative_and_cosine_rows, _stats
from engram.utils import atomic_json, sha256_file


_MAGIC = b"ENGSWQ41"
_VERSION = 1
_HEADER_BYTES = 64
_HEADER = struct.Struct("<8s8I")


def _align(value: int, alignment: int = 64) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("alignment operands must be non-negative/positive")
    return ((value + alignment - 1) // alignment) * alignment


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


def switch_expert_traffic(
    hidden_size: int,
    intermediate_size: int,
    *,
    experts: int = 8,
    width: int = 672,
    centroid_bits: int = 16,
    residual_rank: int = 0,
    cache_line_bytes: int = 64,
) -> dict[str, int | float | bool]:
    """Return exact selected-expert cold bytes for the serialized layout."""

    for value, name in (
        (hidden_size, "hidden_size"),
        (intermediate_size, "intermediate_size"),
        (experts, "experts"),
        (width, "width"),
        (cache_line_bytes, "cache_line_bytes"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if width > intermediate_size:
        raise ValueError("width cannot exceed the source intermediate size")
    if centroid_bits not in {8, 16}:
        raise ValueError("centroid_bits must be 8 or 16")
    if isinstance(residual_rank, bool) or not isinstance(residual_rank, int):
        raise ValueError("residual_rank must be a non-negative integer")
    if residual_rank < 0:
        raise ValueError("residual_rank must be a non-negative integer")

    projection_elements = width * hidden_size
    one_projection_q4_bytes = (projection_elements + 1) // 2
    selected_q4_bytes = 3 * one_projection_q4_bytes
    row_scale_fp16_bytes = (2 * width + hidden_size) * 2
    residual_fp16_bytes = 4 * hidden_size * residual_rank
    expert_payload_bytes = (
        selected_q4_bytes + row_scale_fp16_bytes + residual_fp16_bytes
    )
    expert_block_bytes = _align(expert_payload_bytes, cache_line_bytes)
    if centroid_bits == 16:
        centroid_payload_bytes = experts * hidden_size * 2
        centroid_scale_fp16_bytes = 0
    else:
        centroid_payload_bytes = experts * hidden_size
        centroid_scale_fp16_bytes = experts * 2
    centroid_block_bytes = _align(
        centroid_payload_bytes + centroid_scale_fp16_bytes, cache_line_bytes
    )
    runtime_header_and_id_bytes = _HEADER_BYTES
    total_cold_bytes = (
        runtime_header_and_id_bytes + centroid_block_bytes + expert_block_bytes
    )
    dense_q4_bytes = (3 * hidden_size * intermediate_size + 1) // 2
    serialized_artifact_bytes = (
        runtime_header_and_id_bytes
        + centroid_block_bytes
        + experts * expert_block_bytes
    )
    return {
        "experts": experts,
        "selected_experts": 1,
        "expert_width": width,
        "centroid_bits": centroid_bits,
        "residual_rank": residual_rank,
        "cache_line_bytes": cache_line_bytes,
        "selected_expert_q4_code_bytes": selected_q4_bytes,
        "selected_expert_fp16_scale_bytes": row_scale_fp16_bytes,
        "selected_expert_fp16_residual_bytes": residual_fp16_bytes,
        "selected_expert_payload_bytes": expert_payload_bytes,
        "selected_expert_cache_aligned_bytes": expert_block_bytes,
        "router_centroid_bytes": centroid_payload_bytes,
        "router_centroid_fp16_scale_bytes": centroid_scale_fp16_bytes,
        "router_cache_aligned_bytes": centroid_block_bytes,
        "runtime_header_and_selected_id_bytes": runtime_header_and_id_bytes,
        "total_cold_bytes": total_cold_bytes,
        "dense_q4_bytes": dense_q4_bytes,
        "fraction_of_dense_q4": total_cold_bytes / dense_q4_bytes,
        "passes_45_percent_traffic_gate": total_cold_bytes / dense_q4_bytes <= 0.45,
        "serialized_artifact_bytes": serialized_artifact_bytes,
    }


@dataclass(frozen=True)
class PackedQ4Rows:
    packed: NDArray[np.uint8]
    scales: NDArray[np.float16]
    rows: int
    columns: int


@dataclass(frozen=True)
class PackedSwitchExpert:
    gate: PackedQ4Rows
    up: PackedQ4Rows
    down: PackedQ4Rows
    residual_a: NDArray[np.float16] | None = None
    residual_b: NDArray[np.float16] | None = None


@dataclass(frozen=True)
class LoadedSwitchArtifact:
    centroids: NDArray[np.float32]
    experts: tuple[PackedSwitchExpert, ...]
    hidden_size: int
    width: int
    centroid_bits: int
    residual_rank: int
    centroid_block_bytes: int
    expert_block_bytes: int
    expert_offsets: tuple[int, ...]


def _pack_symmetric_q4_rows(values: ArrayLike) -> PackedQ4Rows:
    """Pack per-output-row symmetric Q4 using an FP16 scale and [-7, 7]."""

    matrix = _matrix(values, "Q4 matrix")
    maximum = np.max(np.abs(matrix), axis=1)
    scale32 = maximum / np.float32(7.0)
    scale32[maximum == 0.0] = np.float32(1.0)
    scales = scale32.astype(np.float16)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Q4 row scale cannot be represented as positive FP16")
    codes = np.rint(matrix / scales.astype(np.float32)[:, None])
    codes = np.clip(codes, -7, 7).astype(np.int8).reshape(-1)
    nibbles = (codes.astype(np.int16) & 0x0F).astype(np.uint8)
    if nibbles.size & 1:
        nibbles = np.concatenate((nibbles, np.zeros(1, dtype=np.uint8)))
    packed = nibbles[0::2] | (nibbles[1::2] << np.uint8(4))
    return PackedQ4Rows(
        np.ascontiguousarray(packed),
        np.ascontiguousarray(scales),
        matrix.shape[0],
        matrix.shape[1],
    )


def _decode_symmetric_q4_rows(packed: PackedQ4Rows) -> NDArray[np.float32]:
    expected = (packed.rows * packed.columns + 1) // 2
    codes_bytes = np.asarray(packed.packed)
    scales = np.asarray(packed.scales)
    if codes_bytes.dtype != np.uint8 or codes_bytes.ndim != 1:
        raise ValueError("packed Q4 codes must be a one-dimensional uint8 array")
    if codes_bytes.size != expected:
        raise ValueError("packed Q4 code length does not match its shape")
    if scales.dtype != np.float16 or scales.shape != (packed.rows,):
        raise ValueError("packed Q4 scales must be one FP16 value per row")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("packed Q4 scales must be finite and positive")
    nibbles = np.empty(codes_bytes.size * 2, dtype=np.uint8)
    nibbles[0::2] = codes_bytes & np.uint8(0x0F)
    nibbles[1::2] = codes_bytes >> np.uint8(4)
    elements = packed.rows * packed.columns
    if elements & 1 and nibbles[elements] != 0:
        raise ValueError("non-zero Q4 padding nibble")
    nibbles = nibbles[:elements]
    if np.any(nibbles == 8):
        raise ValueError("packed Q4 artifact contains forbidden -8 codes")
    signed = nibbles.astype(np.int8)
    signed[signed >= 8] -= np.int8(16)
    if np.any(signed < -7) or np.any(signed > 7):
        raise ValueError("packed Q4 artifact contains an out-of-range code")
    decoded = signed.reshape(packed.rows, packed.columns).astype(np.float32)
    decoded *= scales.astype(np.float32)[:, None]
    return np.ascontiguousarray(decoded)


def _quantize_centroids(
    centroids: NDArray[np.float32], bits: int
) -> tuple[bytes, NDArray[np.float32]]:
    if bits == 16:
        stored = np.ascontiguousarray(centroids.astype("<f2"))
        return stored.tobytes(), stored.astype(np.float32)
    maximum = np.max(np.abs(centroids), axis=1)
    scale32 = maximum / np.float32(127.0)
    scale32[maximum == 0.0] = np.float32(1.0)
    scales = scale32.astype("<f2")
    if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise ValueError("Q8 centroid scale cannot be represented as positive FP16")
    codes = np.clip(
        np.rint(centroids / scales.astype(np.float32)[:, None]), -127, 127
    ).astype(np.int8)
    decoded = codes.astype(np.float32) * scales.astype(np.float32)[:, None]
    return codes.tobytes() + scales.tobytes(), np.ascontiguousarray(decoded)


def _balanced_kmeans_regions(
    values: ArrayLike,
    experts: int,
    *,
    iterations: int = 16,
    seed: int = 0,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Fit deterministic farthest-first Lloyd regions, repairing empty cells."""

    matrix = _matrix(values, "clustering values")
    if not isinstance(experts, int) or isinstance(experts, bool) or experts <= 0:
        raise ValueError("experts must be a positive integer")
    if experts > len(matrix):
        raise ValueError("experts cannot exceed the number of clustering records")
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    # Farthest-first seeds avoid the extremely imbalanced duplicate-centroid
    # failure common to random seeding on transformer states.
    center = np.mean(matrix, axis=0)
    first = int(np.argmin(np.sum((matrix - center) ** 2, axis=1)))
    chosen = [first]
    nearest = np.sum((matrix - matrix[first]) ** 2, axis=1)
    for _ in range(1, experts):
        score = nearest.copy()
        score[np.asarray(chosen)] = -1.0
        candidate = int(np.argmax(score))
        chosen.append(candidate)
        nearest = np.minimum(
            nearest, np.sum((matrix - matrix[candidate]) ** 2, axis=1)
        )
    centroids = matrix[np.asarray(chosen)].copy()
    assignments = np.zeros(len(matrix), dtype=np.int64)
    for _ in range(iterations):
        distances = (
            np.sum(matrix * matrix, axis=1)[:, None]
            + np.sum(centroids * centroids, axis=1)[None, :]
            - 2.0 * matrix @ centroids.T
        )
        new_assignments = np.argmin(distances, axis=1).astype(np.int64)
        counts = np.bincount(new_assignments, minlength=experts)
        for empty in np.flatnonzero(counts == 0):
            owned_distance = distances[np.arange(len(matrix)), new_assignments]
            movable = counts[new_assignments] > 1
            candidate = int(np.argmax(np.where(movable, owned_distance, -1.0)))
            previous = int(new_assignments[candidate])
            new_assignments[candidate] = empty
            counts[previous] -= 1
            counts[empty] += 1
        new_centroids = np.vstack(
            [np.mean(matrix[new_assignments == index], axis=0) for index in range(experts)]
        ).astype(np.float32)
        converged = np.array_equal(assignments, new_assignments)
        assignments = new_assignments
        centroids = new_centroids
        if converged:
            break
    return np.ascontiguousarray(assignments), np.ascontiguousarray(centroids)


def _nearest_centroid_routes(
    values: ArrayLike, centroids: ArrayLike
) -> NDArray[np.int64]:
    matrix = _matrix(values, "routing values")
    centers = _matrix(centroids, "centroids")
    if matrix.shape[1] != centers.shape[1]:
        raise ValueError("routing values and centroids have different widths")
    distances = (
        np.sum(matrix * matrix, axis=1)[:, None]
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * matrix @ centers.T
    )
    return np.argmin(distances, axis=1).astype(np.int64)


def _initial_record_indices(
    inputs: NDArray[np.float32],
    gate: NDArray[np.float32],
    up: NDArray[np.float32],
    down: NDArray[np.float32],
    width: int,
    *,
    batch_size: int = 256,
) -> NDArray[np.int64]:
    """Select source neurons by their mean exact contribution norm."""

    strength = np.zeros(gate.shape[0], dtype=np.float64)
    down_norm = np.linalg.norm(down.astype(np.float64), axis=0)
    for start in range(0, len(inputs), batch_size):
        hidden = inputs[start : start + batch_size].astype(np.float64)
        gate_value = hidden @ gate.astype(np.float64).T
        gate_value = gate_value / (1.0 + np.exp(-np.clip(gate_value, -60.0, 60.0)))
        activation = gate_value * (hidden @ up.astype(np.float64).T)
        strength += np.sum(np.abs(activation), axis=0) * down_norm
    strength /= len(inputs)
    return np.argsort(-strength, kind="stable")[:width].astype(np.int64)


def _fake_q4_rows(weight: Any, torch: Any) -> Any:
    maximum = weight.detach().abs().amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 7.0, torch.ones_like(maximum))
    scale = scale.to(torch.float16).to(weight.dtype)
    quantized = torch.clamp(torch.round(weight / scale), -7, 7) * scale
    return weight + (quantized - weight).detach()


def _expert_type(torch: Any):
    class QuantizedExpert(torch.nn.Module):
        def __init__(
            self,
            gate: NDArray[np.float32],
            up: NDArray[np.float32],
            down: NDArray[np.float32],
            residual_rank: int,
            seed: int,
        ):
            super().__init__()
            self.gate = torch.nn.Parameter(torch.from_numpy(gate.copy()))
            self.up = torch.nn.Parameter(torch.from_numpy(up.copy()))
            self.down = torch.nn.Parameter(torch.from_numpy(down.copy()))
            self.residual_rank = residual_rank
            if residual_rank:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                residual_a = torch.randn(
                    residual_rank, gate.shape[1], generator=generator
                ) * (1e-3 / max(1, gate.shape[1]) ** 0.5)
                self.residual_a = torch.nn.Parameter(residual_a)
                self.residual_b = torch.nn.Parameter(
                    torch.zeros(gate.shape[1], residual_rank)
                )
            else:
                self.register_parameter("residual_a", None)
                self.register_parameter("residual_b", None)

        def forward(self, hidden: Any) -> Any:
            q_gate = _fake_q4_rows(self.gate, torch)
            q_up = _fake_q4_rows(self.up, torch)
            q_down = _fake_q4_rows(self.down, torch)
            gate_value = torch.nn.functional.linear(hidden, q_gate)
            up_value = torch.nn.functional.linear(hidden, q_up)
            output = torch.nn.functional.linear(
                torch.nn.functional.silu(gate_value) * up_value, q_down
            )
            if self.residual_rank:
                a = self.residual_a + (
                    self.residual_a.to(torch.float16).to(self.residual_a.dtype)
                    - self.residual_a
                ).detach()
                b = self.residual_b + (
                    self.residual_b.to(torch.float16).to(self.residual_b.dtype)
                    - self.residual_b
                ).detach()
                output = output + torch.nn.functional.linear(
                    torch.nn.functional.linear(hidden, a), b
                )
            return output

    return QuantizedExpert


def _pack_expert(module: Any) -> PackedSwitchExpert:
    residual_a = None
    residual_b = None
    if module.residual_rank:
        residual_a = np.ascontiguousarray(
            module.residual_a.detach().cpu().numpy().astype(np.float16)
        )
        residual_b = np.ascontiguousarray(
            module.residual_b.detach().cpu().numpy().astype(np.float16)
        )
    return PackedSwitchExpert(
        _pack_symmetric_q4_rows(module.gate.detach()),
        _pack_symmetric_q4_rows(module.up.detach()),
        _pack_symmetric_q4_rows(module.down.detach()),
        residual_a,
        residual_b,
    )


def _expert_payload(expert: PackedSwitchExpert) -> bytes:
    parts = (
        expert.gate.packed.tobytes(),
        expert.up.packed.tobytes(),
        expert.down.packed.tobytes(),
        expert.gate.scales.astype("<f2", copy=False).tobytes(),
        expert.up.scales.astype("<f2", copy=False).tobytes(),
        expert.down.scales.astype("<f2", copy=False).tobytes(),
    )
    payload = b"".join(parts)
    if expert.residual_a is not None or expert.residual_b is not None:
        if expert.residual_a is None or expert.residual_b is None:
            raise ValueError("both residual factors must be present")
        payload += expert.residual_a.astype("<f2", copy=False).tobytes()
        payload += expert.residual_b.astype("<f2", copy=False).tobytes()
    return payload


def save_switch_expert_artifact(
    path: str | Path,
    centroids: ArrayLike,
    experts: Sequence[PackedSwitchExpert],
    *,
    centroid_bits: int = 16,
) -> Path:
    """Write a cache-aligned artifact with one contiguous block per expert."""

    centers = _matrix(centroids, "centroids")
    if not experts or len(experts) != len(centers):
        raise ValueError("one packed expert is required per centroid")
    hidden = centers.shape[1]
    width = experts[0].gate.rows
    residual_rank = 0
    if experts[0].residual_a is not None:
        residual_rank = experts[0].residual_a.shape[0]
    for expert in experts:
        if (
            (expert.gate.rows, expert.gate.columns) != (width, hidden)
            or (expert.up.rows, expert.up.columns) != (width, hidden)
            or (expert.down.rows, expert.down.columns) != (hidden, width)
        ):
            raise ValueError("packed expert shapes are inconsistent")
        _decode_symmetric_q4_rows(expert.gate)
        _decode_symmetric_q4_rows(expert.up)
        _decode_symmetric_q4_rows(expert.down)
        if residual_rank:
            if expert.residual_a is None or expert.residual_b is None:
                raise ValueError("residual layout differs between experts")
            if expert.residual_a.shape != (residual_rank, hidden):
                raise ValueError("residual A has the wrong shape")
            if expert.residual_b.shape != (hidden, residual_rank):
                raise ValueError("residual B has the wrong shape")
        elif expert.residual_a is not None or expert.residual_b is not None:
            raise ValueError("residual layout differs between experts")
    centroid_payload, _ = _quantize_centroids(centers, centroid_bits)
    centroid_block_bytes = _align(len(centroid_payload))
    payloads = [_expert_payload(expert) for expert in experts]
    if len({len(payload) for payload in payloads}) != 1:
        raise ValueError("packed expert payload lengths differ")
    expert_block_bytes = _align(len(payloads[0]))
    header = bytearray(_HEADER_BYTES)
    _HEADER.pack_into(
        header,
        0,
        _MAGIC,
        _VERSION,
        len(experts),
        width,
        hidden,
        centroid_bits,
        residual_rank,
        centroid_block_bytes,
        expert_block_bytes,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(header)
            handle.write(centroid_payload)
            handle.write(b"\0" * (centroid_block_bytes - len(centroid_payload)))
            for payload in payloads:
                handle.write(payload)
                handle.write(b"\0" * (expert_block_bytes - len(payload)))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_switch_expert_artifact(path: str | Path) -> LoadedSwitchArtifact:
    """Strictly validate and load a packed switch-expert artifact."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read switch-expert artifact {source}") from exc
    if len(payload) < _HEADER_BYTES:
        raise ValueError("switch-expert artifact is shorter than its header")
    (
        magic,
        version,
        expert_count,
        width,
        hidden,
        centroid_bits,
        residual_rank,
        centroid_block_bytes,
        expert_block_bytes,
    ) = _HEADER.unpack_from(payload)
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("switch-expert artifact magic/version mismatch")
    if any(value <= 0 for value in (expert_count, width, hidden)):
        raise ValueError("switch-expert artifact has invalid dimensions")
    if centroid_bits not in {8, 16} or residual_rank < 0:
        raise ValueError("switch-expert artifact has invalid codec metadata")
    if centroid_block_bytes % 64 or expert_block_bytes % 64:
        raise ValueError("switch-expert blocks are not cache-line aligned")
    expected_length = (
        _HEADER_BYTES + centroid_block_bytes + expert_count * expert_block_bytes
    )
    if len(payload) != expected_length:
        raise ValueError("switch-expert artifact length does not match its header")
    if any(payload[_HEADER.size : _HEADER_BYTES]):
        raise ValueError("switch-expert header padding is non-zero")

    centroid_start = _HEADER_BYTES
    if centroid_bits == 16:
        centroid_payload_bytes = expert_count * hidden * 2
        centers = np.frombuffer(
            payload, dtype="<f2", count=expert_count * hidden, offset=centroid_start
        ).reshape(expert_count, hidden).astype(np.float32)
    else:
        code_bytes = expert_count * hidden
        centroid_payload_bytes = code_bytes + expert_count * 2
        codes = np.frombuffer(
            payload, dtype=np.int8, count=code_bytes, offset=centroid_start
        ).reshape(expert_count, hidden)
        scales = np.frombuffer(
            payload,
            dtype="<f2",
            count=expert_count,
            offset=centroid_start + code_bytes,
        )
        if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
            raise ValueError("switch-expert centroid scales are invalid")
        centers = codes.astype(np.float32) * scales.astype(np.float32)[:, None]
    if not np.all(np.isfinite(centers)):
        raise ValueError("switch-expert centroids are non-finite")
    if any(
        payload[
            centroid_start + centroid_payload_bytes : centroid_start + centroid_block_bytes
        ]
    ):
        raise ValueError("switch-expert centroid padding is non-zero")

    code_bytes = (width * hidden + 1) // 2
    gate_scale_bytes = width * 2
    down_scale_bytes = hidden * 2
    residual_a_bytes = residual_rank * hidden * 2
    residual_b_bytes = hidden * residual_rank * 2
    raw_expert_bytes = (
        3 * code_bytes
        + 2 * gate_scale_bytes
        + down_scale_bytes
        + residual_a_bytes
        + residual_b_bytes
    )
    if raw_expert_bytes > expert_block_bytes:
        raise ValueError("switch-expert block is too short for declared tensors")
    loaded: list[PackedSwitchExpert] = []
    offsets: list[int] = []
    for index in range(expert_count):
        offset = _HEADER_BYTES + centroid_block_bytes + index * expert_block_bytes
        offsets.append(offset)
        cursor = offset

        def codes() -> NDArray[np.uint8]:
            nonlocal cursor
            result = np.frombuffer(payload, dtype=np.uint8, count=code_bytes, offset=cursor)
            cursor += code_bytes
            return np.ascontiguousarray(result)

        gate_codes, up_codes, down_codes = codes(), codes(), codes()

        def scales(rows: int) -> NDArray[np.float16]:
            nonlocal cursor
            result = np.frombuffer(payload, dtype="<f2", count=rows, offset=cursor)
            cursor += rows * 2
            return np.ascontiguousarray(result)

        gate_scales, up_scales, down_scales = scales(width), scales(width), scales(hidden)
        residual_a = None
        residual_b = None
        if residual_rank:
            residual_a = np.frombuffer(
                payload,
                dtype="<f2",
                count=residual_rank * hidden,
                offset=cursor,
            ).reshape(residual_rank, hidden).copy()
            cursor += residual_a_bytes
            residual_b = np.frombuffer(
                payload,
                dtype="<f2",
                count=hidden * residual_rank,
                offset=cursor,
            ).reshape(hidden, residual_rank).copy()
            cursor += residual_b_bytes
            if not np.all(np.isfinite(residual_a)) or not np.all(np.isfinite(residual_b)):
                raise ValueError("switch-expert residual contains non-finite values")
        if cursor != offset + raw_expert_bytes:
            raise AssertionError("internal packed expert offset mismatch")
        if any(payload[cursor : offset + expert_block_bytes]):
            raise ValueError("switch-expert block padding is non-zero")
        expert = PackedSwitchExpert(
            PackedQ4Rows(gate_codes, gate_scales, width, hidden),
            PackedQ4Rows(up_codes, up_scales, width, hidden),
            PackedQ4Rows(down_codes, down_scales, hidden, width),
            residual_a,
            residual_b,
        )
        _decode_symmetric_q4_rows(expert.gate)
        _decode_symmetric_q4_rows(expert.up)
        _decode_symmetric_q4_rows(expert.down)
        loaded.append(expert)
    if any(offset % 64 for offset in offsets):
        raise ValueError("switch-expert block offset is not cache-line aligned")
    return LoadedSwitchArtifact(
        np.ascontiguousarray(centers),
        tuple(loaded),
        hidden,
        width,
        centroid_bits,
        residual_rank,
        centroid_block_bytes,
        expert_block_bytes,
        tuple(offsets),
    )


def _decoded_expert(expert: PackedSwitchExpert) -> tuple[NDArray[np.float32], ...]:
    gate = _decode_symmetric_q4_rows(expert.gate)
    up = _decode_symmetric_q4_rows(expert.up)
    down = _decode_symmetric_q4_rows(expert.down)
    if expert.residual_a is None:
        return gate, up, down
    return (
        gate,
        up,
        down,
        expert.residual_a.astype(np.float32),
        expert.residual_b.astype(np.float32),
    )


def _evaluate_packed(
    inputs: NDArray[np.float32],
    targets: NDArray[np.float32],
    centroids: NDArray[np.float32],
    experts: Sequence[PackedSwitchExpert],
    *,
    device: str,
) -> tuple[dict[str, Any], NDArray[np.int64]]:
    import torch

    routes = _nearest_centroid_routes(inputs, centroids)
    output = np.empty_like(targets)
    with torch.inference_mode():
        for index, expert in enumerate(experts):
            selected = np.flatnonzero(routes == index)
            if not len(selected):
                continue
            decoded = _decoded_expert(expert)
            hidden = torch.from_numpy(inputs[selected]).to(device)
            gate = torch.from_numpy(decoded[0]).to(device)
            up = torch.from_numpy(decoded[1]).to(device)
            down = torch.from_numpy(decoded[2]).to(device)
            result = torch.nn.functional.linear(
                torch.nn.functional.silu(torch.nn.functional.linear(hidden, gate))
                * torch.nn.functional.linear(hidden, up),
                down,
            )
            if len(decoded) == 5:
                a = torch.from_numpy(decoded[3]).to(device)
                b = torch.from_numpy(decoded[4]).to(device)
                result = result + torch.nn.functional.linear(
                    torch.nn.functional.linear(hidden, a), b
                )
            output[selected] = result.float().cpu().numpy()
    relative, cosine = _relative_and_cosine_rows(output, targets)
    return {
        "relative_l2": _stats(relative.tolist()),
        "cosine": _stats(cosine.tolist()),
    }, routes


def _dense_metrics(
    inputs: NDArray[np.float32],
    targets: NDArray[np.float32],
    gate: NDArray[np.float32],
    up: NDArray[np.float32],
    down: NDArray[np.float32],
    *,
    device: str,
) -> dict[str, Any]:
    import torch

    with torch.inference_mode():
        hidden = torch.from_numpy(inputs).to(device)
        result = torch.nn.functional.linear(
            torch.nn.functional.silu(
                torch.nn.functional.linear(hidden, torch.from_numpy(gate).to(device))
            )
            * torch.nn.functional.linear(hidden, torch.from_numpy(up).to(device)),
            torch.from_numpy(down).to(device),
        ).float().cpu().numpy()
    relative, cosine = _relative_and_cosine_rows(result, targets)
    return {"relative_l2": _stats(relative.tolist()), "cosine": _stats(cosine.tolist())}


def run_switch_expert_boundary_screen(
    gate_weight: ArrayLike,
    up_weight: ArrayLike,
    down_weight: ArrayLike,
    training_inputs: ArrayLike,
    training_outputs: ArrayLike,
    validation_inputs: ArrayLike,
    validation_outputs: ArrayLike,
    *,
    artifact_path: str | Path,
    report_path: str | Path | None = None,
    experts: int = 8,
    width: int = 672,
    centroid_bits: int = 16,
    residual_rank: int = 0,
    clustering_iterations: int = 16,
    steps: int = 128,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    cosine_loss_weight: float = 0.1,
    weight_decay: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
    maximum_mean_relative_l2: float = 0.20,
    enforce_traffic_gate: bool = True,
) -> dict[str, Any]:
    """Train independent regions and evaluate only the final reloaded payload."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for switch-expert boundary training"
        ) from exc
    gate = _matrix(gate_weight, "gate_weight")
    up = _matrix(up_weight, "up_weight")
    down = _matrix(down_weight, "down_weight")
    train_x = _matrix(training_inputs, "training_inputs")
    train_y = _matrix(training_outputs, "training_outputs")
    validation_x = _matrix(validation_inputs, "validation_inputs")
    validation_y = _matrix(validation_outputs, "validation_outputs")
    intermediate, hidden = gate.shape
    if up.shape != gate.shape:
        raise ValueError("up_weight must have the same shape as gate_weight")
    if down.shape != (hidden, intermediate):
        raise ValueError("down_weight must have shape [hidden, intermediate_size]")
    for name, inputs, outputs in (
        ("training", train_x, train_y),
        ("validation", validation_x, validation_y),
    ):
        if inputs.shape[1] != hidden or outputs.shape != (len(inputs), hidden):
            raise ValueError(f"{name} boundary shapes do not match the weights")
    if not 0 < width <= intermediate:
        raise ValueError("width must lie within the source intermediate size")
    if not 0 < experts <= len(train_x):
        raise ValueError("experts must not exceed the training records")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (steps, batch_size, clustering_iterations)
    ):
        raise ValueError("training and clustering counts must be positive integers")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not np.isfinite(cosine_loss_weight) or cosine_loss_weight < 0:
        raise ValueError("cosine_loss_weight must be finite and non-negative")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not np.isfinite(maximum_mean_relative_l2) or maximum_mean_relative_l2 <= 0:
        raise ValueError("maximum_mean_relative_l2 must be positive and finite")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    traffic = switch_expert_traffic(
        hidden,
        intermediate,
        experts=experts,
        width=width,
        centroid_bits=centroid_bits,
        residual_rank=residual_rank,
    )
    if enforce_traffic_gate and not traffic["passes_45_percent_traffic_gate"]:
        raise ValueError("switch-expert layout exceeds the 45% cold-traffic gate")

    _, centroids = _balanced_kmeans_regions(
        train_x, experts, iterations=clustering_iterations, seed=seed
    )
    _, deployed_centroids = _quantize_centroids(centroids, centroid_bits)
    train_routes = _nearest_centroid_routes(train_x, deployed_centroids)
    cluster_counts = np.bincount(train_routes, minlength=experts)
    if np.any(cluster_counts == 0):
        raise ValueError("deployed centroid precision produced an empty training region")

    dense_validation = _dense_metrics(
        validation_x, validation_y, gate, up, down, device=device
    )
    Expert = _expert_type(torch)
    initial_packed: list[PackedSwitchExpert] = []
    final_packed: list[PackedSwitchExpert] = []
    expert_reports: list[dict[str, Any]] = []
    for expert_index in range(experts):
        selected = np.flatnonzero(train_routes == expert_index)
        source_indices = _initial_record_indices(
            train_x[selected], gate, up, down, width, batch_size=batch_size
        )
        module = Expert(
            gate[source_indices],
            up[source_indices],
            down[:, source_indices],
            residual_rank,
            seed + 1009 * (expert_index + 1),
        ).to(device)
        initial_packed.append(_pack_expert(module))
        optimizer = torch.optim.AdamW(
            module.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        x = torch.from_numpy(train_x[selected]).to(device)
        y = torch.from_numpy(train_y[selected]).to(device)
        generator = np.random.default_rng(seed + 7919 * (expert_index + 1))
        order = generator.permutation(len(x))
        cursor = 0
        first_loss = None
        last_loss = None
        module.train()
        for _ in range(steps):
            if cursor + batch_size > len(order):
                order = generator.permutation(len(x))
                cursor = 0
            batch_numpy = order[cursor : cursor + min(batch_size, len(order))]
            cursor += len(batch_numpy)
            batch = torch.as_tensor(batch_numpy, dtype=torch.long, device=device)
            prediction = module(x[batch])
            target = y[batch]
            normalized_mse = torch.mean((prediction - target) ** 2) / torch.clamp(
                torch.mean(target**2), min=1e-8
            )
            cosine = (
                1.0
                - torch.nn.functional.cosine_similarity(prediction, target, dim=1)
            ).mean()
            loss = normalized_mse + cosine_loss_weight * cosine
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            scalar = float(loss.detach().cpu())
            first_loss = scalar if first_loss is None else first_loss
            last_loss = scalar
        final_packed.append(_pack_expert(module))
        expert_reports.append(
            {
                "expert": expert_index,
                "training_records": int(len(selected)),
                "initial_source_record_indices": source_indices.tolist(),
                "first_training_loss": first_loss,
                "final_training_loss": last_loss,
            }
        )
        del optimizer, module, x, y
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    initial_validation, initial_routes = _evaluate_packed(
        validation_x,
        validation_y,
        deployed_centroids,
        initial_packed,
        device=device,
    )
    destination = save_switch_expert_artifact(
        artifact_path, centroids, final_packed, centroid_bits=centroid_bits
    )
    reloaded = load_switch_expert_artifact(destination)
    strict_validation, validation_routes = _evaluate_packed(
        validation_x,
        validation_y,
        reloaded.centroids,
        reloaded.experts,
        device=device,
    )
    if not np.array_equal(initial_routes, validation_routes):
        raise AssertionError("centroid route changed across artifact round trip")
    expected_size = int(traffic["serialized_artifact_bytes"])
    if destination.stat().st_size != expected_size:
        raise AssertionError("serialized artifact size differs from traffic layout")

    improved = (
        strict_validation["relative_l2"]["mean"]
        < initial_validation["relative_l2"]["mean"]
    )
    checks = {
        "strict_packed_mean_relative_l2": strict_validation["relative_l2"]["mean"]
        <= maximum_mean_relative_l2,
        "training_improved_held_out_boundary": improved,
        "actual_packed_traffic": bool(traffic["passes_45_percent_traffic_gate"]),
        "artifact_exact_size": destination.stat().st_size == expected_size,
        "all_expert_offsets_cache_aligned": all(
            offset % 64 == 0 for offset in reloaded.expert_offsets
        ),
    }
    report = {
        "schema_version": 1,
        "experiment": "top1_specialized_switch_expert_boundary_screen",
        "configuration": {
            "experts": experts,
            "expert_width": width,
            "centroid_bits": centroid_bits,
            "residual_rank": residual_rank,
            "clustering_iterations": clustering_iterations,
            "steps_per_expert": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "cosine_loss_weight": cosine_loss_weight,
            "weight_decay": weight_decay,
            "seed": seed,
            "device": device,
            "validation_checkpoint_selection": False,
        },
        "router": {
            "kind": "nearest_centroid_top1",
            "training_cluster_counts": cluster_counts.astype(int).tolist(),
            "validation_route_counts": np.bincount(
                validation_routes, minlength=experts
            ).astype(int).tolist(),
        },
        "training": {"experts": expert_reports},
        "dense_reference_validation_parity": dense_validation,
        "validation": {
            "initial_packed_initialized_experts": initial_validation,
            "strict_reloaded_packed_experts": strict_validation,
            "mean_relative_l2_improvement_fraction": (
                initial_validation["relative_l2"]["mean"]
                - strict_validation["relative_l2"]["mean"]
            )
            / max(initial_validation["relative_l2"]["mean"], 1e-12),
            "source": "serialized_reloaded_q4_experts_and_deployed_centroids",
        },
        "traffic": traffic,
        "artifact": {
            "path": str(destination.resolve()),
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
            "header_bytes": _HEADER_BYTES,
            "centroid_block_bytes": reloaded.centroid_block_bytes,
            "expert_block_bytes": reloaded.expert_block_bytes,
            "expert_offsets": list(reloaded.expert_offsets),
            "all_expert_offsets_cache_aligned": all(
                offset % 64 == 0 for offset in reloaded.expert_offsets
            ),
        },
        "screen": {
            "passed": all(checks.values()),
            "checks": checks,
            "decision": (
                "eligible_for_causal_sequence_intervention"
                if all(checks.values())
                else "reject_or_continue_boundary_optimization"
            ),
            "caveat": (
                "This fixed-split boundary ceiling never selects checkpoints on "
                "validation data and makes no causal sequence-level claim."
            ),
        },
    }
    if report_path is not None:
        atomic_json(Path(report_path), report)
    return report


train_switch_expert_boundaries = run_switch_expert_boundary_screen


__all__ = [
    "LoadedSwitchArtifact",
    "PackedQ4Rows",
    "PackedSwitchExpert",
    "_balanced_kmeans_regions",
    "_decode_symmetric_q4_rows",
    "_nearest_centroid_routes",
    "_pack_symmetric_q4_rows",
    "load_switch_expert_artifact",
    "run_switch_expert_boundary_screen",
    "save_switch_expert_artifact",
    "switch_expert_traffic",
    "train_switch_expert_boundaries",
]
