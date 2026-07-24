"""ctypes binding for packed official-layout BitNet projections."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from engram.evaluation.native_bitnet_parity import _torch_modules


class _Metrics(ctypes.Structure):
    _fields_ = [
        ("elapsed_ns", ctypes.c_uint64),
        ("packed_weight_bytes", ctypes.c_uint64),
        ("scratch_bytes", ctypes.c_uint64),
        ("rows", ctypes.c_uint64),
    ]


class NativeTernaryProjectionKernel:
    def __init__(
        self,
        *,
        threads: int = 12,
        library: str | Path | None = None,
    ) -> None:
        if threads <= 0:
            raise ValueError("threads must be positive")
        default = Path(__file__).resolve().parents[3] / "build/libengram_bitnet.so"
        self.library_path = Path(default if library is None else library).resolve()
        self._library = ctypes.CDLL(str(self.library_path))
        self._configure()
        error = ctypes.create_string_buffer(512)
        self._handle = self._library.engram_ternary_projection_create(
            threads, error, len(error)
        )
        if not self._handle:
            raise RuntimeError(error.value.decode("utf-8", "replace"))
        self.shapes: list[tuple[int, int]] = []
        self.calls: list[dict[str, int]] = []

    def _configure(self) -> None:
        library = self._library
        library.engram_ternary_projection_create.argtypes = [
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.engram_ternary_projection_create.restype = ctypes.c_void_p
        library.engram_ternary_projection_destroy.argtypes = [ctypes.c_void_p]
        library.engram_ternary_projection_add.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.engram_ternary_projection_add.restype = ctypes.c_int
        library.engram_ternary_projection_forward_bf16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.POINTER(_Metrics),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.engram_ternary_projection_forward_bf16.restype = ctypes.c_int

    def add(self, packed, *, output_features: int, scale: float) -> int:
        source = np.ascontiguousarray(packed, dtype=np.uint8)
        if source.ndim != 2 or output_features != source.shape[0] * 4:
            raise ValueError("packed projection shape is invalid")
        index = ctypes.c_size_t()
        error = ctypes.create_string_buffer(512)
        status = self._library.engram_ternary_projection_add(
            self._handle,
            ctypes.c_void_p(source.ctypes.data),
            source.nbytes,
            source.shape[1],
            output_features,
            float(scale),
            ctypes.byref(index),
            error,
            len(error),
        )
        if status:
            raise RuntimeError(error.value.decode("utf-8", "replace"))
        self.shapes.append((source.shape[1], output_features))
        return int(index.value)

    def forward(self, projection: int, hidden):
        torch, _, _ = _torch_modules()
        if hidden.device.type != "cpu" or hidden.dtype != torch.bfloat16:
            raise ValueError("native projection requires a CPU BF16 tensor")
        input_features, output_features = self.shapes[int(projection)]
        if hidden.shape[-1] != input_features:
            raise ValueError("native projection input shape is invalid")
        source = hidden.contiguous()
        rows = source.numel() // input_features
        output = torch.empty(
            (*source.shape[:-1], output_features),
            dtype=torch.bfloat16,
            device="cpu",
        )
        metrics = _Metrics()
        error = ctypes.create_string_buffer(512)
        status = self._library.engram_ternary_projection_forward_bf16(
            self._handle,
            int(projection),
            ctypes.c_void_p(source.data_ptr()),
            rows,
            ctypes.c_void_p(output.data_ptr()),
            ctypes.byref(metrics),
            error,
            len(error),
        )
        if status:
            raise RuntimeError(error.value.decode("utf-8", "replace"))
        self.calls.append(
            {name: int(getattr(metrics, name)) for name, _ in metrics._fields_}
        )
        return output

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.engram_ternary_projection_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def native_projection_module_class():
    _, nn, _ = _torch_modules()

    class NativeProjection(nn.Module):
        def __init__(self, kernel, projection: int) -> None:
            super().__init__()
            self.kernel = kernel
            self.projection = int(projection)

        def forward(self, values):
            return self.kernel.forward(self.projection, values)

    return NativeProjection


__all__ = ["NativeTernaryProjectionKernel", "native_projection_module_class"]
