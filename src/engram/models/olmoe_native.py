"""Mapped non-MLP weight artifact for the native OLMoE token runtime."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

from engram.models.inspection import (
    load_local_named_tensors,
    local_tensor_inventory,
)
from engram.models.olmoe import audit_olmoe_source
from engram.utils import sha256_file


class OLMoENativeWeightError(ValueError):
    """Raised when native OLMoE non-MLP weights cannot be compiled."""


def _bf16_payload(values: np.ndarray) -> bytes:
    array = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise OLMoENativeWeightError("non-MLP tensor contains non-finite values")
    bits = array.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return np.asarray((bits + bias) >> np.uint32(16), dtype="<u2").tobytes()


def repack_olmoe_non_mlp_weights(
    model: str | Path, out: str | Path
) -> dict[str, Any]:
    """Stream all required non-MLP tensors into one mmap-safe BF16 file."""

    model_path = Path(model).expanduser().resolve()
    audit = audit_olmoe_source(model_path)
    if audit.decision != "proceed_to_router_trace":
        raise OLMoENativeWeightError(
            "source does not satisfy the exact OLMoE tensor contract"
        )
    inventory = local_tensor_inventory(model_path)
    names = sorted(tensor.name for tensor in inventory if ".mlp." not in tensor.name)
    layer_count = int(audit.dimensions["num_hidden_layers"] or 0)
    expected_count = 3 + 8 * layer_count
    if len(names) != expected_count:
        raise OLMoENativeWeightError(
            f"expected {expected_count} non-MLP tensors, found {len(names)}"
        )
    by_name = {tensor.name: tensor for tensor in inventory}
    offset = 0
    header: dict[str, dict[str, object]] = {}
    for name in names:
        shape = list(by_name[name].shape)
        elements = int(np.prod(shape, dtype=np.int64))
        size = elements * 2
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    header_payload = json.dumps(
        header, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    header_payload += b" " * ((8 - len(header_payload) % 8) % 8)

    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(struct.pack("<Q", len(header_payload)))
            handle.write(header_payload)
            for name in names:
                tensor = load_local_named_tensors(
                    model_path, [name], inventory=inventory
                )[name]
                expected_shape = tuple(by_name[name].shape)
                if tensor.shape != expected_shape:
                    raise OLMoENativeWeightError(
                        f"{name} changed shape during non-MLP compilation"
                    )
                handle.write(_bf16_payload(tensor))
            handle.flush()
            os.fsync(handle.fileno())
        expected_bytes = 8 + len(header_payload) + offset
        if temporary.stat().st_size != expected_bytes:
            raise AssertionError("native OLMoE non-MLP byte accounting failed")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema_version": 1,
        "format": "olmoe_native_non_mlp_bf16_v1",
        "source": str(model_path),
        "path": str(destination.resolve()),
        "tensor_count": len(names),
        "tensor_payload_bytes": offset,
        "file_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "dtype": "BF16",
    }


__all__ = [
    "OLMoENativeWeightError",
    "repack_olmoe_non_mlp_weights",
]
