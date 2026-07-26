"""NumPy/ctypes binding for the CPU-only native BitNet DIP kernel."""

from __future__ import annotations

import ctypes
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.utils import sha256_file


class NativeBitNetDIPKernelError(RuntimeError):
    """Raised when the native DIP artifact or request fails validation."""


class _NativeDIPMetrics(ctypes.Structure):
    _fields_ = [
        ("elapsed_ns", ctypes.c_uint64),
        ("coordinate_stream_bytes", ctypes.c_uint64),
        ("candidate_completion_bytes", ctypes.c_uint64),
        ("gain_stream_bytes", ctypes.c_uint64),
        ("down_norm_stream_bytes", ctypes.c_uint64),
        ("selected_down_stream_bytes", ctypes.c_uint64),
        ("layer_metadata_bytes", ctypes.c_uint64),
        ("scheduled_cache_line_bytes", ctypes.c_uint64),
        ("scratch_bytes", ctypes.c_uint64),
        ("rows", ctypes.c_uint64),
        ("threads", ctypes.c_uint64),
        ("input_coordinates", ctypes.c_uint64),
        ("candidate_count", ctypes.c_uint64),
        ("selected_count_total", ctypes.c_uint64),
        ("selected_count_min", ctypes.c_uint64),
        ("selected_count_max", ctypes.c_uint64),
    ]

    def to_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name, _ in self._fields_}


@dataclass(frozen=True)
class NativeBitNetDIPKernelPolicy:
    input_coordinates: int
    candidate_count: int
    minimum_top_k: int
    maximum_top_k: int
    energy_target: float
    rms_audit_count: int
    rms_estimator: str
    rms_audit_strategy: str


@dataclass(frozen=True)
class NativeBitNetDIPKernelResult:
    output: NDArray[np.float32]
    output_bf16_bits: NDArray[np.uint16]
    selected_counts: NDArray[np.uint32]
    metrics: dict[str, int]
    input_coordinate_ids: NDArray[np.uint32] | None = None
    candidate_ids: NDArray[np.uint32] | None = None
    selected_record_ids: NDArray[np.uint32] | None = None


@dataclass(frozen=True)
class NativeBitNetDIPTorchDiagnostics:
    selected_counts: NDArray[np.uint32]
    metrics: dict[str, int]
    input_coordinate_ids: NDArray[np.uint32] | None = None
    candidate_ids: NDArray[np.uint32] | None = None
    selected_record_ids: NDArray[np.uint32] | None = None


def _default_library_path() -> Path:
    configured = os.environ.get("ENGRAM_BITNET_DIP_LIBRARY")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path(__file__).resolve().parents[3]
        / "build"
        / "libengram_bitnet_dip.so"
    )


def _load_library(path: str | Path | None):
    library_path = (
        Path(path).resolve() if path is not None else _default_library_path()
    )
    if not library_path.is_file():
        raise NativeBitNetDIPKernelError(
            f"native BitNet DIP library is missing: {library_path}; "
            "configure and build the CMake target `engram_bitnet_dip`"
        )
    library = ctypes.CDLL(str(library_path))
    library.engram_bitnet_dip_open.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.engram_bitnet_dip_open.restype = ctypes.c_void_p
    library.engram_bitnet_dip_close.argtypes = [ctypes.c_void_p]
    library.engram_bitnet_dip_close.restype = None
    for name in (
        "engram_bitnet_dip_layer_count",
        "engram_bitnet_dip_hidden_size",
        "engram_bitnet_dip_intermediate_size",
        "engram_bitnet_dip_thread_count",
        "engram_bitnet_dip_record_artifact_bytes",
        "engram_bitnet_dip_coordinate_index_bytes",
    ):
        function = getattr(library, name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_size_t
    library.engram_bitnet_dip_layer_policy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.engram_bitnet_dip_layer_policy.restype = ctypes.c_int
    library.engram_bitnet_dip_forward_bf16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(_NativeDIPMetrics),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.engram_bitnet_dip_forward_bf16.restype = ctypes.c_int
    library.engram_bitnet_dip_forward_debug_bf16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(_NativeDIPMetrics),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.engram_bitnet_dip_forward_debug_bf16.restype = ctypes.c_int
    library.engram_bitnet_dip_teacher_top_k_bf16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.engram_bitnet_dip_teacher_top_k_bf16.restype = ctypes.c_int
    library.engram_bitnet_dip_teacher_top_k_positive_bf16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.engram_bitnet_dip_teacher_top_k_positive_bf16.restype = ctypes.c_int
    return library, library_path


def _float32_to_bf16_bits(values: ArrayLike) -> NDArray[np.uint16]:
    source = np.ascontiguousarray(values, dtype=np.float32)
    bits = source.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + (
        (bits >> np.uint32(16)) & np.uint32(1)
    )
    return np.ascontiguousarray(rounded >> np.uint32(16), dtype=np.uint16)


def _bf16_bits_to_float32(values: NDArray[np.uint16]) -> NDArray[np.float32]:
    return np.ascontiguousarray(
        (np.asarray(values, dtype=np.uint32) << np.uint32(16)).view(np.float32)
    )


class NativeBitNetDIPCPUKernel:
    """Persistent, mmap-backed CPU kernel with no PyTorch dependency."""

    def __init__(
        self,
        record_artifact: str | Path,
        coordinate_index: str | Path,
        *,
        threads: int = 1,
        library: str | Path | None = None,
        expected_record_sha256: str | None = None,
        expected_index_sha256: str | None = None,
    ) -> None:
        if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
            raise ValueError("threads must be a positive integer")
        self.record_artifact = Path(record_artifact).resolve()
        self.coordinate_index = Path(coordinate_index).resolve()
        self.record_sha256 = sha256_file(self.record_artifact)
        self.index_sha256 = sha256_file(self.coordinate_index)
        for actual, expected, label in (
            (self.record_sha256, expected_record_sha256, "record artifact"),
            (self.index_sha256, expected_index_sha256, "coordinate index"),
        ):
            if expected is not None and actual != expected.lower():
                raise NativeBitNetDIPKernelError(f"{label} SHA-256 mismatch")
        self._library, self.library_path = _load_library(library)
        error = ctypes.create_string_buffer(1024)
        self._handle = self._library.engram_bitnet_dip_open(
            os.fsencode(self.record_artifact),
            os.fsencode(self.coordinate_index),
            threads,
            error,
            len(error),
        )
        if not self._handle:
            raise NativeBitNetDIPKernelError(
                error.value.decode("utf-8", "replace")
            )
        self.layer_count = int(
            self._library.engram_bitnet_dip_layer_count(self._handle)
        )
        self.hidden_size = int(
            self._library.engram_bitnet_dip_hidden_size(self._handle)
        )
        self.intermediate_size = int(
            self._library.engram_bitnet_dip_intermediate_size(self._handle)
        )
        self.thread_count = int(
            self._library.engram_bitnet_dip_thread_count(self._handle)
        )
        self.record_artifact_bytes = int(
            self._library.engram_bitnet_dip_record_artifact_bytes(self._handle)
        )
        self.coordinate_index_bytes = int(
            self._library.engram_bitnet_dip_coordinate_index_bytes(self._handle)
        )
        self.policies = tuple(
            self._read_policy(layer) for layer in range(self.layer_count)
        )
        self.calls: list[dict[str, int]] = []

    def _read_policy(self, layer: int) -> NativeBitNetDIPKernelPolicy:
        fields = [ctypes.c_size_t() for _ in range(4)]
        energy = ctypes.c_float()
        audit_count = ctypes.c_size_t()
        estimator = ctypes.c_uint32()
        audit_strategy = ctypes.c_uint32()
        error = ctypes.create_string_buffer(1024)
        status = self._library.engram_bitnet_dip_layer_policy(
            self._handle,
            layer,
            *(ctypes.byref(field) for field in fields),
            ctypes.byref(energy),
            ctypes.byref(audit_count),
            ctypes.byref(estimator),
            ctypes.byref(audit_strategy),
            error,
            len(error),
        )
        if status:
            raise NativeBitNetDIPKernelError(
                error.value.decode("utf-8", "replace")
            )
        estimator_name = {1: "corrected_proxy", 2: "candidate_ratio"}.get(
            estimator.value
        )
        audit_name = {0: "none", 2: "top_proxy_raw_square"}.get(
            audit_strategy.value
        )
        if estimator_name is None or audit_name is None:
            raise NativeBitNetDIPKernelError(
                "native BitNet DIP returned an unsupported RMS policy"
            )
        return NativeBitNetDIPKernelPolicy(
            input_coordinates=fields[0].value,
            candidate_count=fields[1].value,
            minimum_top_k=fields[2].value,
            maximum_top_k=fields[3].value,
            energy_target=float(energy.value),
            rms_audit_count=audit_count.value,
            rms_estimator=estimator_name,
            rms_audit_strategy=audit_name,
        )

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.engram_bitnet_dip_close(self._handle)
            self._handle = None

    def __enter__(self) -> "NativeBitNetDIPCPUKernel":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def forward_bf16_bits(
        self,
        layer: int,
        hidden_bits: ArrayLike,
    ) -> NativeBitNetDIPKernelResult:
        return self._forward_bf16_bits(layer, hidden_bits, debug=False)

    def forward_debug_bf16_bits(
        self,
        layer: int,
        hidden_bits: ArrayLike,
    ) -> NativeBitNetDIPKernelResult:
        """Return exact route identities outside the timed sparse region."""

        return self._forward_bf16_bits(layer, hidden_bits, debug=True)

    def _forward_bf16_bits(
        self,
        layer: int,
        hidden_bits: ArrayLike,
        *,
        debug: bool,
    ) -> NativeBitNetDIPKernelResult:
        if not self._handle:
            raise NativeBitNetDIPKernelError("native BitNet DIP kernel is closed")
        source = np.ascontiguousarray(hidden_bits, dtype=np.uint16)
        if source.ndim < 1 or source.shape[-1] != self.hidden_size:
            raise NativeBitNetDIPKernelError(
                "native BitNet DIP hidden shape is invalid"
            )
        if not 0 <= int(layer) < self.layer_count:
            raise NativeBitNetDIPKernelError(
                "native BitNet DIP layer index is invalid"
            )
        rows = source.size // self.hidden_size
        output = np.empty_like(source)
        selected_counts = np.empty(rows, dtype=np.uint32)
        policy = self.policies[int(layer)]
        input_coordinate_ids = (
            np.empty((rows, policy.input_coordinates), dtype=np.uint32)
            if debug
            else None
        )
        candidate_ids = (
            np.empty((rows, policy.candidate_count), dtype=np.uint32)
            if debug
            else None
        )
        selected_record_ids = (
            np.empty((rows, policy.maximum_top_k), dtype=np.uint32)
            if debug
            else None
        )
        metrics = _NativeDIPMetrics()
        error = ctypes.create_string_buffer(1024)
        common = (
            self._handle,
            int(layer),
            ctypes.c_void_p(source.ctypes.data),
            rows,
            ctypes.c_void_p(output.ctypes.data),
            ctypes.c_void_p(selected_counts.ctypes.data),
        )
        if debug:
            assert input_coordinate_ids is not None
            assert candidate_ids is not None
            assert selected_record_ids is not None
            status = self._library.engram_bitnet_dip_forward_debug_bf16(
                *common,
                ctypes.c_void_p(input_coordinate_ids.ctypes.data),
                ctypes.c_void_p(candidate_ids.ctypes.data),
                ctypes.c_void_p(selected_record_ids.ctypes.data),
                ctypes.byref(metrics),
                error,
                len(error),
            )
        else:
            status = self._library.engram_bitnet_dip_forward_bf16(
                *common,
                ctypes.byref(metrics),
                error,
                len(error),
            )
        if status:
            raise NativeBitNetDIPKernelError(
                error.value.decode("utf-8", "replace")
            )
        call = metrics.to_dict()
        call["layer"] = int(layer)
        self.calls.append(call)
        return NativeBitNetDIPKernelResult(
            output=_bf16_bits_to_float32(output).reshape(source.shape),
            output_bf16_bits=output.reshape(source.shape),
            selected_counts=selected_counts.reshape(source.shape[:-1]),
            metrics=call,
            input_coordinate_ids=(
                input_coordinate_ids.reshape(
                    (*source.shape[:-1], policy.input_coordinates)
                )
                if input_coordinate_ids is not None
                else None
            ),
            candidate_ids=(
                candidate_ids.reshape(
                    (*source.shape[:-1], policy.candidate_count)
                )
                if candidate_ids is not None
                else None
            ),
            selected_record_ids=(
                selected_record_ids.reshape(
                    (*source.shape[:-1], policy.maximum_top_k)
                )
                if selected_record_ids is not None
                else None
            ),
        )

    def forward(
        self,
        layer: int,
        hidden: ArrayLike,
    ) -> NativeBitNetDIPKernelResult:
        source = np.asarray(hidden, dtype=np.float32)
        return self.forward_bf16_bits(layer, _float32_to_bf16_bits(source))

    def forward_debug(
        self,
        layer: int,
        hidden: ArrayLike,
    ) -> NativeBitNetDIPKernelResult:
        source = np.asarray(hidden, dtype=np.float32)
        return self.forward_debug_bf16_bits(
            layer,
            _float32_to_bf16_bits(source),
        )

    def teacher_top_k_bf16_bits(
        self,
        layer: int,
        hidden_bits: ArrayLike,
        *,
        top_k: int,
    ) -> NDArray[np.uint32]:
        """Return canonical exact native-BF16 teacher utility IDs."""

        if not self._handle:
            raise NativeBitNetDIPKernelError("native BitNet DIP kernel is closed")
        source = np.ascontiguousarray(hidden_bits, dtype=np.uint16)
        if source.ndim < 1 or source.shape[-1] != self.hidden_size:
            raise NativeBitNetDIPKernelError(
                "native BitNet DIP hidden shape is invalid"
            )
        if not 0 <= int(layer) < self.layer_count:
            raise NativeBitNetDIPKernelError(
                "native BitNet DIP layer index is invalid"
            )
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 0 < top_k <= self.intermediate_size
        ):
            raise NativeBitNetDIPKernelError(
                "native BitNet DIP teacher top-K is invalid"
            )
        rows = source.size // self.hidden_size
        result = np.empty((rows, top_k), dtype=np.uint32)
        error = ctypes.create_string_buffer(1024)
        status = self._library.engram_bitnet_dip_teacher_top_k_bf16(
            self._handle,
            int(layer),
            ctypes.c_void_p(source.ctypes.data),
            rows,
            top_k,
            ctypes.c_void_p(result.ctypes.data),
            error,
            len(error),
        )
        if status:
            raise NativeBitNetDIPKernelError(
                error.value.decode("utf-8", "replace")
            )
        return result.reshape((*source.shape[:-1], top_k))

    def teacher_top_k(
        self,
        layer: int,
        hidden: ArrayLike,
        *,
        top_k: int,
    ) -> NDArray[np.uint32]:
        return self.teacher_top_k_bf16_bits(
            layer,
            _float32_to_bf16_bits(np.asarray(hidden, dtype=np.float32)),
            top_k=top_k,
        )

    def teacher_top_k_with_positive_counts_bf16_bits(
        self,
        layer: int,
        hidden_bits: ArrayLike,
        *,
        top_k: int,
    ) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
        """Return exact teacher IDs and each row's positive-utility count."""

        if not self._handle:
            raise NativeBitNetDIPKernelError("native BitNet DIP kernel is closed")
        source = np.ascontiguousarray(hidden_bits, dtype=np.uint16)
        if source.ndim < 1 or source.shape[-1] != self.hidden_size:
            raise NativeBitNetDIPKernelError(
                "native BitNet DIP hidden shape is invalid"
            )
        if not 0 <= int(layer) < self.layer_count:
            raise NativeBitNetDIPKernelError(
                "native BitNet DIP layer index is invalid"
            )
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 0 < top_k <= self.intermediate_size
        ):
            raise NativeBitNetDIPKernelError(
                "native BitNet DIP teacher top-K is invalid"
            )
        rows = source.size // self.hidden_size
        result = np.empty((rows, top_k), dtype=np.uint32)
        positive_counts = np.empty(rows, dtype=np.uint32)
        error = ctypes.create_string_buffer(1024)
        status = (
            self._library.engram_bitnet_dip_teacher_top_k_positive_bf16(
                self._handle,
                int(layer),
                ctypes.c_void_p(source.ctypes.data),
                rows,
                top_k,
                ctypes.c_void_p(result.ctypes.data),
                ctypes.c_void_p(positive_counts.ctypes.data),
                error,
                len(error),
            )
        )
        if status:
            raise NativeBitNetDIPKernelError(
                error.value.decode("utf-8", "replace")
            )
        return (
            result.reshape((*source.shape[:-1], top_k)),
            positive_counts.reshape(source.shape[:-1]),
        )

    def forward_torch(
        self,
        layer: int,
        hidden,
        *,
        debug_routes: bool = False,
    ):
        """Execute directly on a contiguous CPU BF16 torch tensor."""

        try:
            import torch
        except ImportError as exc:
            raise NativeBitNetDIPKernelError(
                "native BitNet DIP torch substitution requires torch"
            ) from exc
        if not self._handle:
            raise NativeBitNetDIPKernelError("native BitNet DIP kernel is closed")
        if hidden.device.type != "cpu" or hidden.dtype != torch.bfloat16:
            raise NativeBitNetDIPKernelError(
                "native BitNet DIP torch input must be CPU BF16"
            )
        if hidden.ndim < 1 or hidden.shape[-1] != self.hidden_size:
            raise NativeBitNetDIPKernelError(
                "native BitNet DIP hidden shape is invalid"
            )
        if not 0 <= int(layer) < self.layer_count:
            raise NativeBitNetDIPKernelError(
                "native BitNet DIP layer index is invalid"
            )
        source = hidden.contiguous()
        output = torch.empty_like(source)
        rows = source.numel() // self.hidden_size
        prefix = tuple(source.shape[:-1])
        policy = self.policies[int(layer)]
        selected_counts = np.empty(rows, dtype=np.uint32)
        input_coordinate_ids = (
            np.empty((rows, policy.input_coordinates), dtype=np.uint32)
            if debug_routes
            else None
        )
        candidate_ids = (
            np.empty((rows, policy.candidate_count), dtype=np.uint32)
            if debug_routes
            else None
        )
        selected_record_ids = (
            np.empty((rows, policy.maximum_top_k), dtype=np.uint32)
            if debug_routes
            else None
        )
        metrics = _NativeDIPMetrics()
        error = ctypes.create_string_buffer(1024)
        common = (
            self._handle,
            int(layer),
            ctypes.c_void_p(source.data_ptr()),
            rows,
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(selected_counts.ctypes.data),
        )
        if debug_routes:
            assert input_coordinate_ids is not None
            assert candidate_ids is not None
            assert selected_record_ids is not None
            status = self._library.engram_bitnet_dip_forward_debug_bf16(
                *common,
                ctypes.c_void_p(input_coordinate_ids.ctypes.data),
                ctypes.c_void_p(candidate_ids.ctypes.data),
                ctypes.c_void_p(selected_record_ids.ctypes.data),
                ctypes.byref(metrics),
                error,
                len(error),
            )
        else:
            status = self._library.engram_bitnet_dip_forward_bf16(
                *common,
                ctypes.byref(metrics),
                error,
                len(error),
            )
        if status:
            raise NativeBitNetDIPKernelError(
                error.value.decode("utf-8", "replace")
            )
        call = metrics.to_dict()
        call["layer"] = int(layer)
        self.calls.append(call)
        diagnostics = NativeBitNetDIPTorchDiagnostics(
            selected_counts=selected_counts.reshape(prefix),
            metrics=call,
            input_coordinate_ids=(
                input_coordinate_ids.reshape((*prefix, policy.input_coordinates))
                if input_coordinate_ids is not None
                else None
            ),
            candidate_ids=(
                candidate_ids.reshape((*prefix, policy.candidate_count))
                if candidate_ids is not None
                else None
            ),
            selected_record_ids=(
                selected_record_ids.reshape((*prefix, policy.maximum_top_k))
                if selected_record_ids is not None
                else None
            ),
        )
        return output.reshape(hidden.shape), diagnostics


def build_native_bitnet_dip_kernel_mlp(
    kernel: NativeBitNetDIPCPUKernel,
    layer: int,
    *,
    debug_routes: bool = False,
):
    """Build an inference-only torch module over one persistent native kernel."""

    try:
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "native BitNet DIP torch substitution requires torch"
        ) from exc
    if not isinstance(kernel, NativeBitNetDIPCPUKernel):
        raise TypeError("kernel must be NativeBitNetDIPCPUKernel")
    if not 0 <= int(layer) < kernel.layer_count:
        raise ValueError("native BitNet DIP layer index is invalid")

    class _NativeBitNetDIPKernelMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kernel = kernel
            self.layer = int(layer)
            self.debug_routes = bool(debug_routes)
            self.last_result: NativeBitNetDIPTorchDiagnostics | None = None

        def forward(self, hidden_states):
            if hidden_states.requires_grad:
                raise RuntimeError(
                    "native BitNet DIP kernel module is inference-only"
                )
            output, self.last_result = self.kernel.forward_torch(
                self.layer,
                hidden_states,
                debug_routes=self.debug_routes,
            )
            return output

    return _NativeBitNetDIPKernelMLP()


@contextmanager
def substitute_native_bitnet_dip_kernel_mlps(
    model,
    kernel: NativeBitNetDIPCPUKernel,
    *,
    debug_routes: bool = False,
):
    """Temporarily replace every model-shell MLP with the native DIP kernel."""

    decoder = getattr(getattr(model, "model", None), "layers", None)
    if decoder is None:
        raise ValueError("model does not expose model.layers")
    if len(decoder) != kernel.layer_count:
        raise ValueError("model layer count differs from native DIP policy")
    originals = {layer: decoder[layer].mlp for layer in range(len(decoder))}
    replacements = {}
    try:
        for layer in range(len(decoder)):
            replacement = build_native_bitnet_dip_kernel_mlp(
                kernel,
                layer,
                debug_routes=debug_routes,
            )
            decoder[layer].mlp = replacement
            replacements[layer] = replacement
        yield replacements
    finally:
        for layer, original in originals.items():
            decoder[layer].mlp = original


__all__ = [
    "NativeBitNetDIPCPUKernel",
    "NativeBitNetDIPKernelError",
    "NativeBitNetDIPKernelPolicy",
    "NativeBitNetDIPKernelResult",
    "NativeBitNetDIPTorchDiagnostics",
    "build_native_bitnet_dip_kernel_mlp",
    "substitute_native_bitnet_dip_kernel_mlps",
]
