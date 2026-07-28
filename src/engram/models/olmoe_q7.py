"""Immutable, directly addressable Q7 artifact for OLMoE experts.

The format is intentionally simple enough for an independent native reader:

* all integers are little endian;
* routers are BF16;
* expert matrices use signed symmetric Q7 codes in ``[-63, 63]``;
* codes are a canonical little-endian 7-bit stream with a bias of 63;
* each matrix row has BF16 scales for consecutive input groups; and
* layers, experts, phases, codes, and scales begin on cache-line boundaries.

No dense decoded expert is stored in or required to validate the artifact.
"""

from __future__ import annotations

import json
import math
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.models.inspection import load_local_named_tensors, local_tensor_inventory
from engram.models.olmoe import audit_olmoe_source


_MAGIC = b"ENGOQ711"
_LAYER_MAGIC = b"ENGOQ7L1"
_EXPERT_MAGIC = b"ENGOQ7E1"
_VERSION = 1
_ENDIAN_MARKER = 0x01020304
_HEADER_BYTES = 128
_DIRECTORY_ENTRY_BYTES = 64
_LAYER_HEADER_BYTES = 64
_EXPERT_HEADER_BYTES = 64
_CACHE_LINE_BYTES = 64
_BITS = 7
_CODE_BIAS = 63
_HEADER = struct.Struct("<8s16I4Q")
_DIRECTORY = struct.Struct("<II7Q")
_LAYER_HEADER = struct.Struct("<8s8I3Q")
_EXPERT_HEADER = struct.Struct("<8s6I4Q")


class OLMoEQ7ValidationError(ValueError):
    """Raised when a Q7 artifact or source contract is invalid."""


def _align(value: int, alignment: int = _CACHE_LINE_BYTES) -> int:
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise OLMoEQ7ValidationError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


def _bf16_bits(values: ArrayLike) -> NDArray[np.uint16]:
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return np.asarray((bits + bias) >> np.uint32(16), dtype=np.uint16)


def bf16_from_bits(values: ArrayLike) -> NDArray[np.float32]:
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32)
    return np.asarray((bits << np.uint32(16)).view(np.float32), dtype=np.float32)


def quantize_q7_matrix(
    values: ArrayLike, *, group_size: int = 64
) -> tuple[NDArray[np.int8], NDArray[np.uint16]]:
    """Quantize one row-major matrix with executed BF16 group scales."""

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.size or not np.all(np.isfinite(matrix)):
        raise OLMoEQ7ValidationError("Q7 source must be a non-empty finite matrix")
    if group_size <= 0:
        raise OLMoEQ7ValidationError("Q7 group_size must be positive")
    rows, columns = matrix.shape
    groups = (columns + group_size - 1) // group_size
    padded = np.zeros((rows, groups * group_size), dtype=np.float32)
    padded[:, :columns] = matrix
    blocks = padded.reshape(rows, groups, group_size)
    maximum = np.max(np.abs(blocks), axis=2)
    scales = np.where(maximum == 0.0, 1.0, maximum / 63.0)
    scale_bits = _bf16_bits(scales)
    executed_scales = bf16_from_bits(scale_bits)
    if (
        not np.all(np.isfinite(executed_scales))
        or np.any(executed_scales <= 0.0)
    ):
        raise OLMoEQ7ValidationError("Q7 BF16 scales are not finite and positive")
    codes = np.clip(
        np.rint(blocks / executed_scales[:, :, None]), -63, 63
    ).astype(np.int8)
    return (
        np.ascontiguousarray(codes.reshape(rows, -1)[:, :columns]),
        np.ascontiguousarray(scale_bits),
    )


def dequantize_q7_matrix(
    codes: ArrayLike, scale_bits: ArrayLike, *, group_size: int
) -> NDArray[np.float32]:
    matrix = np.asarray(codes, dtype=np.int8)
    scales = bf16_from_bits(scale_bits)
    if matrix.ndim != 2 or scales.ndim != 2:
        raise OLMoEQ7ValidationError("Q7 codes and scales must be matrices")
    expected_groups = (matrix.shape[1] + group_size - 1) // group_size
    if scales.shape != (matrix.shape[0], expected_groups):
        raise OLMoEQ7ValidationError("Q7 scale shape is inconsistent with codes")
    columns = np.arange(matrix.shape[1], dtype=np.int64) // group_size
    return np.ascontiguousarray(
        matrix.astype(np.float32) * scales[:, columns], dtype=np.float32
    )


def pack_q7_codes(codes: ArrayLike) -> bytes:
    """Pack signed Q7 codes into the canonical biased LSB-first bitstream."""

    signed = np.asarray(codes)
    if signed.size and (
        np.any(signed < -63)
        or np.any(signed > 63)
        or not np.issubdtype(signed.dtype, np.integer)
    ):
        raise OLMoEQ7ValidationError("Q7 codes must be integers in [-63, 63]")
    flat = signed.astype(np.int16, copy=False).reshape(-1)
    padded_count = _align(flat.size, 8)
    unsigned = np.zeros(padded_count, dtype=np.uint16)
    unsigned[: flat.size] = flat + _CODE_BIAS
    blocks = unsigned.reshape(-1, 8)
    packed = np.zeros((blocks.shape[0], 7), dtype=np.uint8)
    for code_index in range(8):
        bit = code_index * _BITS
        byte = bit // 8
        shift = bit % 8
        values = blocks[:, code_index]
        packed[:, byte] |= ((values << shift) & 0xFF).astype(np.uint8)
        if shift > 1:
            packed[:, byte + 1] |= (values >> (8 - shift)).astype(np.uint8)
    byte_count = (flat.size * _BITS + 7) // 8
    return packed.reshape(-1)[:byte_count].tobytes()


def unpack_q7_codes(payload: bytes | bytearray | memoryview, count: int) -> NDArray[np.int8]:
    """Validate and decode a canonical Q7 stream."""

    if count < 0:
        raise OLMoEQ7ValidationError("Q7 code count must be non-negative")
    expected_bytes = (count * _BITS + 7) // 8
    if len(payload) != expected_bytes:
        raise OLMoEQ7ValidationError("Q7 packed stream length is inconsistent")
    source = np.frombuffer(payload, dtype=np.uint8)
    padded_byte_count = ((expected_bytes + 6) // 7) * 7
    padded = np.zeros(padded_byte_count, dtype=np.uint8)
    padded[:expected_bytes] = source
    blocks = padded.reshape(-1, 7)
    decoded = np.empty((blocks.shape[0], 8), dtype=np.uint16)
    for code_index in range(8):
        bit = code_index * _BITS
        byte = bit // 8
        shift = bit % 8
        value = blocks[:, byte].astype(np.uint16) >> shift
        if shift > 1:
            value |= blocks[:, byte + 1].astype(np.uint16) << (8 - shift)
        decoded[:, code_index] = value & 0x7F
    unsigned = decoded.reshape(-1)[:count]
    if np.any(unsigned > 126):
        raise OLMoEQ7ValidationError("Q7 stream contains reserved code 127")
    if count and count * _BITS % 8:
        used = count * _BITS % 8
        if int(source[-1]) >> used:
            raise OLMoEQ7ValidationError("Q7 packed tail is not canonical")
    return np.asarray(unsigned.astype(np.int16) - _CODE_BIAS, dtype=np.int8)


@dataclass(frozen=True)
class OLMoEQ7Layout:
    layer_count: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    group_size: int
    router_bytes: int
    gate_code_bytes: int
    gate_scale_bytes: int
    down_code_bytes: int
    down_scale_bytes: int
    matrix_gate_bytes: int
    matrix_down_bytes: int
    expert_payload_bytes: int
    expert_stride: int
    layer_payload_bytes: int
    layer_block_bytes: int
    directory_bytes: int
    file_bytes: int


def olmoe_q7_layout(
    *,
    layer_count: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    group_size: int = 64,
) -> OLMoEQ7Layout:
    dimensions = (layer_count, hidden_size, intermediate_size, num_experts, top_k)
    if any(isinstance(value, bool) or value <= 0 for value in dimensions):
        raise OLMoEQ7ValidationError("Q7 dimensions must be positive integers")
    if top_k > num_experts or group_size <= 0:
        raise OLMoEQ7ValidationError("Q7 top_k/group_size is invalid")
    router_bytes = num_experts * hidden_size * 2
    gate_elements = intermediate_size * hidden_size
    down_elements = hidden_size * intermediate_size
    gate_code_bytes = (gate_elements * _BITS + 7) // 8
    down_code_bytes = (down_elements * _BITS + 7) // 8
    gate_scale_bytes = (
        intermediate_size * ((hidden_size + group_size - 1) // group_size) * 2
    )
    down_scale_bytes = (
        hidden_size * ((intermediate_size + group_size - 1) // group_size) * 2
    )
    matrix_gate_bytes = _align(gate_code_bytes) + _align(gate_scale_bytes)
    matrix_down_bytes = _align(down_code_bytes) + _align(down_scale_bytes)
    expert_payload_bytes = (
        _EXPERT_HEADER_BYTES + 2 * matrix_gate_bytes + matrix_down_bytes
    )
    expert_stride = _align(expert_payload_bytes)
    experts_offset = _align(_LAYER_HEADER_BYTES + router_bytes)
    layer_payload_bytes = experts_offset + num_experts * expert_stride
    layer_block_bytes = _align(layer_payload_bytes)
    directory_bytes = _align(layer_count * _DIRECTORY_ENTRY_BYTES)
    file_bytes = _HEADER_BYTES + directory_bytes + layer_count * layer_block_bytes
    return OLMoEQ7Layout(
        layer_count=layer_count,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        group_size=group_size,
        router_bytes=router_bytes,
        gate_code_bytes=gate_code_bytes,
        gate_scale_bytes=gate_scale_bytes,
        down_code_bytes=down_code_bytes,
        down_scale_bytes=down_scale_bytes,
        matrix_gate_bytes=matrix_gate_bytes,
        matrix_down_bytes=matrix_down_bytes,
        expert_payload_bytes=expert_payload_bytes,
        expert_stride=expert_stride,
        layer_payload_bytes=layer_payload_bytes,
        layer_block_bytes=layer_block_bytes,
        directory_bytes=directory_bytes,
        file_bytes=file_bytes,
    )


def _matrix_bytes(
    values: ArrayLike, *, group_size: int, expected_shape: tuple[int, int]
) -> tuple[bytes, bytes]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != expected_shape:
        raise OLMoEQ7ValidationError(
            f"Q7 source shape {matrix.shape} does not match {expected_shape}"
        )
    codes, scales = quantize_q7_matrix(matrix, group_size=group_size)
    return pack_q7_codes(codes), scales.astype("<u2", copy=False).tobytes()


def _write_padding(handle, count: int) -> None:
    if count < 0:
        raise AssertionError("negative Q7 padding")
    if count:
        handle.write(bytes(count))


def _write_matrix(
    handle,
    values: ArrayLike,
    *,
    group_size: int,
    shape: tuple[int, int],
    code_bytes: int,
    scale_bytes: int,
) -> None:
    packed, scales = _matrix_bytes(
        values, group_size=group_size, expected_shape=shape
    )
    if len(packed) != code_bytes or len(scales) != scale_bytes:
        raise AssertionError("Q7 matrix differs from layout accounting")
    handle.write(packed)
    _write_padding(handle, _align(code_bytes) - code_bytes)
    handle.write(scales)
    _write_padding(handle, _align(scale_bytes) - scale_bytes)


def save_olmoe_q7_artifact(
    path: str | Path,
    *,
    routers: Sequence[ArrayLike],
    experts: Sequence[Sequence[Mapping[str, ArrayLike]]],
    top_k: int,
    group_size: int = 64,
) -> Path:
    """Serialize in-memory router/expert weights, primarily for fixtures."""

    if not routers or len(routers) != len(experts) or not experts[0]:
        raise OLMoEQ7ValidationError("Q7 layers and experts must not be empty")
    first_router = np.asarray(routers[0])
    first_gate = np.asarray(experts[0][0]["gate"])
    if first_router.ndim != 2 or first_gate.ndim != 2:
        raise OLMoEQ7ValidationError("Q7 router and expert weights must be matrices")
    num_experts, hidden_size = first_router.shape
    intermediate_size, expert_hidden = first_gate.shape
    if expert_hidden != hidden_size:
        raise OLMoEQ7ValidationError("Q7 router/expert hidden sizes disagree")

    def layers() -> Iterator[
        tuple[ArrayLike, Iterable[Mapping[str, ArrayLike]]]
    ]:
        yield from zip(routers, experts, strict=True)

    return _write_olmoe_q7(
        path,
        layers=layers(),
        layer_count=len(routers),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        group_size=group_size,
    )


def _write_olmoe_q7(
    path: str | Path,
    *,
    layers: Iterator[tuple[ArrayLike, Iterable[Mapping[str, ArrayLike]]]],
    layer_count: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    group_size: int,
) -> Path:
    layout = olmoe_q7_layout(
        layer_count=layer_count,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        group_size=group_size,
    )
    header = bytearray(_HEADER_BYTES)
    _HEADER.pack_into(
        header,
        0,
        _MAGIC,
        _VERSION,
        _ENDIAN_MARKER,
        _HEADER_BYTES,
        _DIRECTORY_ENTRY_BYTES,
        _LAYER_HEADER_BYTES,
        _EXPERT_HEADER_BYTES,
        _CACHE_LINE_BYTES,
        _BITS,
        _CODE_BIAS,
        group_size,
        layer_count,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        0,
        _HEADER_BYTES,
        layout.directory_bytes,
        layout.file_bytes,
        0,
    )
    directory = bytearray(layout.directory_bytes)
    router_offset = _LAYER_HEADER_BYTES
    experts_offset = _align(router_offset + layout.router_bytes)
    first_layer = _HEADER_BYTES + layout.directory_bytes
    for layer_index in range(layer_count):
        layer_offset = first_layer + layer_index * layout.layer_block_bytes
        _DIRECTORY.pack_into(
            directory,
            layer_index * _DIRECTORY_ENTRY_BYTES,
            layer_index,
            0,
            layer_offset,
            layout.layer_block_bytes,
            router_offset,
            layout.router_bytes,
            experts_offset,
            layout.expert_stride,
            layout.layer_payload_bytes,
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    emitted = 0
    try:
        with temporary.open("wb") as handle:
            handle.write(header)
            handle.write(directory)
            for layer_index, (router_value, expert_values) in enumerate(layers):
                if layer_index >= layer_count:
                    raise OLMoEQ7ValidationError("Q7 source layer count is invalid")
                layer_start = handle.tell()
                layer_header = _LAYER_HEADER.pack(
                    _LAYER_MAGIC,
                    _VERSION,
                    layer_index,
                    hidden_size,
                    intermediate_size,
                    num_experts,
                    top_k,
                    group_size,
                    _BITS,
                    router_offset,
                    experts_offset,
                    layout.layer_block_bytes,
                )
                handle.write(layer_header)
                router = np.asarray(router_value, dtype=np.float32)
                if router.shape != (num_experts, hidden_size):
                    raise OLMoEQ7ValidationError("Q7 router shape is invalid")
                router_bits = _bf16_bits(router)
                if not np.all(np.isfinite(bf16_from_bits(router_bits))):
                    raise OLMoEQ7ValidationError("Q7 router contains non-finite values")
                handle.write(router_bits.astype("<u2", copy=False).tobytes())
                _write_padding(handle, experts_offset - (handle.tell() - layer_start))
                expert_count = 0
                for expert_index, expert in enumerate(expert_values):
                    if expert_index >= num_experts:
                        raise OLMoEQ7ValidationError(
                            "Q7 source has too many experts"
                        )
                    expert_start = handle.tell()
                    gate_offset = _EXPERT_HEADER_BYTES
                    up_offset = gate_offset + layout.matrix_gate_bytes
                    down_offset = up_offset + layout.matrix_gate_bytes
                    handle.write(
                        _EXPERT_HEADER.pack(
                            _EXPERT_MAGIC,
                            _VERSION,
                            expert_index,
                            intermediate_size,
                            hidden_size,
                            hidden_size,
                            intermediate_size,
                            gate_offset,
                            up_offset,
                            down_offset,
                            layout.expert_stride,
                        )
                    )
                    _write_matrix(
                        handle,
                        expert["gate"],
                        group_size=group_size,
                        shape=(intermediate_size, hidden_size),
                        code_bytes=layout.gate_code_bytes,
                        scale_bytes=layout.gate_scale_bytes,
                    )
                    _write_matrix(
                        handle,
                        expert["up"],
                        group_size=group_size,
                        shape=(intermediate_size, hidden_size),
                        code_bytes=layout.gate_code_bytes,
                        scale_bytes=layout.gate_scale_bytes,
                    )
                    _write_matrix(
                        handle,
                        expert["down"],
                        group_size=group_size,
                        shape=(hidden_size, intermediate_size),
                        code_bytes=layout.down_code_bytes,
                        scale_bytes=layout.down_scale_bytes,
                    )
                    _write_padding(
                        handle, layout.expert_stride - (handle.tell() - expert_start)
                    )
                    expert_count += 1
                if expert_count != num_experts:
                    raise OLMoEQ7ValidationError(
                        "Q7 source layer has the wrong expert count"
                    )
                _write_padding(
                    handle, layout.layer_block_bytes - (handle.tell() - layer_start)
                )
                emitted += 1
            if emitted != layer_count:
                raise OLMoEQ7ValidationError("Q7 source ended before all layers")
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size != layout.file_bytes:
            raise AssertionError("Q7 artifact size differs from layout accounting")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def repack_olmoe_q7_model(
    model: str | Path,
    out: str | Path,
    *,
    group_size: int = 64,
) -> Path:
    """Stream an audited local OLMoE checkpoint into the native Q7 format."""

    model_path = Path(model).expanduser().resolve()
    audit = audit_olmoe_source(model_path)
    if audit.decision != "proceed_to_router_trace":
        raise OLMoEQ7ValidationError("source does not satisfy the exact OLMoE contract")
    dimensions = audit.dimensions
    layer_count = int(dimensions["num_hidden_layers"] or 0)
    hidden_size = int(dimensions["hidden_size"] or 0)
    intermediate_size = int(dimensions["intermediate_size"] or 0)
    num_experts = int(dimensions["num_experts"] or 0)
    top_k = int(dimensions["num_experts_per_tok"] or 0)
    inventory = local_tensor_inventory(model_path)

    def source_layers():
        for layer in range(layer_count):
            prefix = f"model.layers.{layer}.mlp"
            router = load_local_named_tensors(
                model_path,
                [f"{prefix}.gate.weight"],
                inventory=inventory,
            )[f"{prefix}.gate.weight"]

            def source_experts():
                for expert in range(num_experts):
                    expert_prefix = f"{prefix}.experts.{expert}"
                    names = {
                        phase: f"{expert_prefix}.{phase}_proj.weight"
                        for phase in ("gate", "up", "down")
                    }
                    tensors = load_local_named_tensors(
                        model_path,
                        list(names.values()),
                        inventory=inventory,
                    )
                    yield {
                        phase: tensors[name] for phase, name in names.items()
                    }

            yield router, source_experts()

    return _write_olmoe_q7(
        out,
        layers=source_layers(),
        layer_count=layer_count,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        group_size=group_size,
    )


class LoadedOLMoEQ7Artifact:
    """Strict mmap-backed Q7 reader used for validation and reference parity."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = self.path.open("rb")
        try:
            self._mapping = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
            self._parse()
        except Exception:
            self._handle.close()
            raise

    def __enter__(self) -> "LoadedOLMoEQ7Artifact":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            mapping.close()
            self._mapping = None
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()
            self._handle = None

    def _parse(self) -> None:
        payload = self._mapping
        if len(payload) < _HEADER_BYTES:
            raise OLMoEQ7ValidationError("Q7 artifact is shorter than its header")
        unpacked = _HEADER.unpack_from(payload)
        (
            magic,
            version,
            endian,
            header_bytes,
            directory_entry_bytes,
            layer_header_bytes,
            expert_header_bytes,
            cache_line_bytes,
            bits,
            code_bias,
            group_size,
            layer_count,
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
            reserved,
            directory_offset,
            directory_bytes,
            file_bytes,
            reserved64,
        ) = unpacked
        if (
            magic != _MAGIC
            or version != _VERSION
            or endian != _ENDIAN_MARKER
            or header_bytes != _HEADER_BYTES
            or directory_entry_bytes != _DIRECTORY_ENTRY_BYTES
            or layer_header_bytes != _LAYER_HEADER_BYTES
            or expert_header_bytes != _EXPERT_HEADER_BYTES
            or cache_line_bytes != _CACHE_LINE_BYTES
            or bits != _BITS
            or code_bias != _CODE_BIAS
            or reserved
            or reserved64
        ):
            raise OLMoEQ7ValidationError("Q7 header contract is unsupported")
        self.layout = olmoe_q7_layout(
            layer_count=layer_count,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            group_size=group_size,
        )
        if (
            directory_offset != _HEADER_BYTES
            or directory_bytes != self.layout.directory_bytes
            or file_bytes != self.layout.file_bytes
            or len(payload) != file_bytes
            or any(payload[_HEADER.size : _HEADER_BYTES])
        ):
            raise OLMoEQ7ValidationError("Q7 header sizes or padding are invalid")
        directory_used = layer_count * _DIRECTORY_ENTRY_BYTES
        if any(
            payload[
                directory_offset + directory_used : directory_offset + directory_bytes
            ]
        ):
            raise OLMoEQ7ValidationError("Q7 directory padding is non-zero")
        self._layer_offsets: list[int] = []
        expected = _HEADER_BYTES + directory_bytes
        for layer in range(layer_count):
            entry = _DIRECTORY.unpack_from(
                payload, directory_offset + layer * _DIRECTORY_ENTRY_BYTES
            )
            (
                entry_layer,
                entry_reserved,
                offset,
                block_bytes,
                router_offset,
                router_bytes,
                experts_offset,
                expert_stride,
                payload_bytes,
            ) = entry
            if (
                entry_layer != layer
                or entry_reserved
                or offset != expected
                or offset % _CACHE_LINE_BYTES
                or block_bytes != self.layout.layer_block_bytes
                or router_offset != _LAYER_HEADER_BYTES
                or router_bytes != self.layout.router_bytes
                or experts_offset != _align(_LAYER_HEADER_BYTES + router_bytes)
                or expert_stride != self.layout.expert_stride
                or payload_bytes != self.layout.layer_payload_bytes
            ):
                raise OLMoEQ7ValidationError("Q7 directory entry is invalid")
            self._validate_layer(layer, offset, router_offset, experts_offset)
            self._layer_offsets.append(offset)
            expected += block_bytes

    def _validate_layer(
        self, layer: int, offset: int, router_offset: int, experts_offset: int
    ) -> None:
        payload = self._mapping
        header = _LAYER_HEADER.unpack_from(payload, offset)
        if header != (
            _LAYER_MAGIC,
            _VERSION,
            layer,
            self.layout.hidden_size,
            self.layout.intermediate_size,
            self.layout.num_experts,
            self.layout.top_k,
            self.layout.group_size,
            _BITS,
            router_offset,
            experts_offset,
            self.layout.layer_block_bytes,
        ):
            raise OLMoEQ7ValidationError("Q7 layer header is invalid")
        router_start = offset + router_offset
        router_end = router_start + self.layout.router_bytes
        router = np.frombuffer(
            payload,
            dtype="<u2",
            count=self.layout.router_bytes // 2,
            offset=router_start,
        )
        if not np.all(np.isfinite(bf16_from_bits(router))):
            raise OLMoEQ7ValidationError("Q7 router is not finite")
        if any(payload[router_end : offset + experts_offset]):
            raise OLMoEQ7ValidationError("Q7 router padding is non-zero")
        for expert in range(self.layout.num_experts):
            expert_start = offset + experts_offset + expert * self.layout.expert_stride
            self._validate_expert(expert, expert_start)

    def _validate_expert(self, expert: int, offset: int) -> None:
        gate_offset = _EXPERT_HEADER_BYTES
        up_offset = gate_offset + self.layout.matrix_gate_bytes
        down_offset = up_offset + self.layout.matrix_gate_bytes
        header = _EXPERT_HEADER.unpack_from(self._mapping, offset)
        if header != (
            _EXPERT_MAGIC,
            _VERSION,
            expert,
            self.layout.intermediate_size,
            self.layout.hidden_size,
            self.layout.hidden_size,
            self.layout.intermediate_size,
            gate_offset,
            up_offset,
            down_offset,
            self.layout.expert_stride,
        ):
            raise OLMoEQ7ValidationError("Q7 expert header is invalid")
        for phase_offset, rows, columns, code_bytes, scale_bytes in (
            (
                gate_offset,
                self.layout.intermediate_size,
                self.layout.hidden_size,
                self.layout.gate_code_bytes,
                self.layout.gate_scale_bytes,
            ),
            (
                up_offset,
                self.layout.intermediate_size,
                self.layout.hidden_size,
                self.layout.gate_code_bytes,
                self.layout.gate_scale_bytes,
            ),
            (
                down_offset,
                self.layout.hidden_size,
                self.layout.intermediate_size,
                self.layout.down_code_bytes,
                self.layout.down_scale_bytes,
            ),
        ):
            start = offset + phase_offset
            packed = self._mapping[start : start + code_bytes]
            unpack_q7_codes(packed, rows * columns)
            code_end = start + code_bytes
            scale_start = start + _align(code_bytes)
            scale_end = scale_start + scale_bytes
            scales = np.frombuffer(
                self._mapping,
                dtype="<u2",
                count=scale_bytes // 2,
                offset=scale_start,
            )
            decoded_scales = bf16_from_bits(scales)
            if not np.all(np.isfinite(decoded_scales)) or np.any(
                decoded_scales <= 0
            ):
                raise OLMoEQ7ValidationError("Q7 expert scales are invalid")
            matrix_end = start + _align(code_bytes) + _align(scale_bytes)
            if any(self._mapping[code_end:scale_start]) or any(
                self._mapping[scale_end:matrix_end]
            ):
                raise OLMoEQ7ValidationError("Q7 matrix padding is non-zero")

    def router(self, layer: int) -> NDArray[np.float32]:
        offset = self._layer_offsets[layer] + _LAYER_HEADER_BYTES
        bits = np.frombuffer(
            self._mapping,
            dtype="<u2",
            count=self.layout.num_experts * self.layout.hidden_size,
            offset=offset,
        ).copy()
        return bf16_from_bits(bits).reshape(
            self.layout.num_experts, self.layout.hidden_size
        )

    def expert(self, layer: int, expert: int) -> dict[str, NDArray[np.float32]]:
        if not 0 <= expert < self.layout.num_experts:
            raise IndexError("Q7 expert index is out of range")
        layer_offset = self._layer_offsets[layer]
        experts_offset = _align(_LAYER_HEADER_BYTES + self.layout.router_bytes)
        base = layer_offset + experts_offset + expert * self.layout.expert_stride
        result = {}
        phase_specs = (
            (
                "gate",
                _EXPERT_HEADER_BYTES,
                self.layout.intermediate_size,
                self.layout.hidden_size,
                self.layout.gate_code_bytes,
                self.layout.gate_scale_bytes,
            ),
            (
                "up",
                _EXPERT_HEADER_BYTES + self.layout.matrix_gate_bytes,
                self.layout.intermediate_size,
                self.layout.hidden_size,
                self.layout.gate_code_bytes,
                self.layout.gate_scale_bytes,
            ),
            (
                "down",
                _EXPERT_HEADER_BYTES + 2 * self.layout.matrix_gate_bytes,
                self.layout.hidden_size,
                self.layout.intermediate_size,
                self.layout.down_code_bytes,
                self.layout.down_scale_bytes,
            ),
        )
        for name, phase, rows, columns, code_bytes, scale_bytes in phase_specs:
            start = base + phase
            codes = unpack_q7_codes(
                self._mapping[start : start + code_bytes], rows * columns
            ).reshape(rows, columns)
            scale_start = start + _align(code_bytes)
            scales = np.frombuffer(
                self._mapping,
                dtype="<u2",
                count=scale_bytes // 2,
                offset=scale_start,
            ).copy()
            result[name] = dequantize_q7_matrix(
                codes,
                scales.reshape(rows, -1),
                group_size=self.layout.group_size,
            )
        return result

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "format": "olmoe_native_groupwise_q7_v1",
            "path": str(self.path.resolve()),
            "file_bytes": self.layout.file_bytes,
            "dimensions": {
                "layers": self.layout.layer_count,
                "hidden_size": self.layout.hidden_size,
                "intermediate_size": self.layout.intermediate_size,
                "experts": self.layout.num_experts,
                "top_k": self.layout.top_k,
            },
            "quantizer": {
                "bits": _BITS,
                "code_range": [-63, 63],
                "group_size": self.layout.group_size,
                "scale_dtype": "bfloat16_executed",
                "packing": "biased_lsb_first_7bit",
            },
            "addressability": {
                "cache_line_bytes": _CACHE_LINE_BYTES,
                "expert_stride": self.layout.expert_stride,
                "expert_payload_bytes": self.layout.expert_payload_bytes,
            },
        }


def inspect_olmoe_q7_artifact(path: str | Path) -> dict[str, object]:
    with LoadedOLMoEQ7Artifact(path) as artifact:
        return artifact.metadata()


def write_olmoe_q7_report(path: str | Path, out: str | Path) -> Path:
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = inspect_olmoe_q7_artifact(path)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


__all__ = [
    "LoadedOLMoEQ7Artifact",
    "OLMoEQ7Layout",
    "OLMoEQ7ValidationError",
    "bf16_from_bits",
    "dequantize_q7_matrix",
    "inspect_olmoe_q7_artifact",
    "olmoe_q7_layout",
    "pack_q7_codes",
    "quantize_q7_matrix",
    "repack_olmoe_q7_model",
    "save_olmoe_q7_artifact",
    "unpack_q7_codes",
    "write_olmoe_q7_report",
]
