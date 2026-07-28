"""ctypes binding for the CPU-only native OLMoE Q7 expert kernel."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


class OLMoEQ7NativeError(RuntimeError):
    """Raised when the native Q7 library rejects an operation."""


class _Metrics(ctypes.Structure):
    _fields_ = [
        ("elapsed_ns", ctypes.c_uint64),
        ("router_stream_bytes", ctypes.c_uint64),
        ("selected_expert_stream_bytes", ctypes.c_uint64),
        ("scheduled_stream_bytes", ctypes.c_uint64),
        ("scratch_bytes", ctypes.c_uint64),
        ("rows", ctypes.c_uint64),
        ("threads", ctypes.c_uint64),
        ("selected_experts", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class OLMoEQ7NativeResult:
    output: NDArray[np.float32]
    selected_experts: NDArray[np.uint32]
    metrics: dict[str, int]


def _configure(library: ctypes.CDLL) -> None:
    library.engram_olmoe_q7_open.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.engram_olmoe_q7_open.restype = ctypes.c_void_p
    library.engram_olmoe_q7_close.argtypes = [ctypes.c_void_p]
    for name in (
        "layer_count",
        "hidden_size",
        "intermediate_size",
        "expert_count",
        "top_k",
        "group_size",
        "artifact_bytes",
    ):
        function = getattr(library, f"engram_olmoe_q7_{name}")
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_size_t
    library.engram_olmoe_q7_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(_Metrics),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.engram_olmoe_q7_forward.restype = ctypes.c_int


class OLMoEQ7NativeKernel:
    def __init__(
        self, artifact: str | Path, library: str | Path, *, threads: int = 1
    ):
        self._library = ctypes.CDLL(str(Path(library).resolve()))
        _configure(self._library)
        error = ctypes.create_string_buffer(1024)
        self._handle = self._library.engram_olmoe_q7_open(
            str(Path(artifact).resolve()).encode(),
            threads,
            error,
            len(error),
        )
        if not self._handle:
            raise OLMoEQ7NativeError(error.value.decode(errors="replace"))
        self.layer_count = self._value("layer_count")
        self.hidden_size = self._value("hidden_size")
        self.intermediate_size = self._value("intermediate_size")
        self.expert_count = self._value("expert_count")
        self.top_k = self._value("top_k")
        self.group_size = self._value("group_size")
        self.artifact_bytes = self._value("artifact_bytes")

    def _value(self, name: str) -> int:
        return int(getattr(self._library, f"engram_olmoe_q7_{name}")(self._handle))

    def close(self) -> None:
        if self._handle:
            self._library.engram_olmoe_q7_close(self._handle)
            self._handle = None

    def __enter__(self) -> "OLMoEQ7NativeKernel":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def forward(self, layer: int, values: ArrayLike) -> OLMoEQ7NativeResult:
        source = np.asarray(values, dtype=np.float32)
        if source.ndim == 1:
            source = source[None, :]
        if source.ndim != 2 or source.shape[1] != self.hidden_size:
            raise ValueError("native Q7 input must have shape [rows, hidden_size]")
        source = np.ascontiguousarray(source)
        output = np.empty_like(source)
        selected = np.empty((source.shape[0], self.top_k), dtype=np.uint32)
        metrics = _Metrics()
        error = ctypes.create_string_buffer(1024)
        status = self._library.engram_olmoe_q7_forward(
            self._handle,
            layer,
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            source.shape[0],
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            selected.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.byref(metrics),
            error,
            len(error),
        )
        if status:
            raise OLMoEQ7NativeError(error.value.decode(errors="replace"))
        return OLMoEQ7NativeResult(
            output=output,
            selected_experts=selected,
            metrics={
                name: int(getattr(metrics, name))
                for name, _ctype in metrics._fields_
            },
        )


__all__ = [
    "OLMoEQ7NativeError",
    "OLMoEQ7NativeKernel",
    "OLMoEQ7NativeResult",
]
