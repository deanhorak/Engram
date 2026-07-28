"""ctypes wrapper for the bounded native streaming-attention kernel."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class _Config(ctypes.Structure):
    _fields_ = [
        ("query_heads", ctypes.c_size_t),
        ("key_value_heads", ctypes.c_size_t),
        ("head_dimension", ctypes.c_size_t),
        ("local_window", ctypes.c_size_t),
        ("older_candidates", ctypes.c_size_t),
        ("older_top_k", ctypes.c_size_t),
        ("sink_tokens", ctypes.c_size_t),
        ("scale", ctypes.c_float),
    ]


class _Metrics(ctypes.Structure):
    _fields_ = [
        ("tokens_seen", ctypes.c_uint64),
        ("local_entries", ctypes.c_uint64),
        ("active_older_entries", ctypes.c_uint64),
        ("candidate_key_bytes", ctypes.c_uint64),
        ("selected_value_bytes", ctypes.c_uint64),
        ("local_kv_bytes", ctypes.c_uint64),
        ("eviction_events", ctypes.c_uint64),
        ("older_candidate_entries_scored", ctypes.c_uint64),
        ("older_selected_entries", ctypes.c_uint64),
        ("sink_insertions", ctypes.c_uint64),
        ("heavy_hitter_updates", ctypes.c_uint64),
        ("state_bytes", ctypes.c_uint64),
        ("scratch_bytes", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class NativeStreamingAttentionMetrics:
    tokens_seen: int
    local_entries: int
    active_older_entries: int
    candidate_key_bytes: int
    selected_value_bytes: int
    local_kv_bytes: int
    eviction_events: int
    older_candidate_entries_scored: int
    older_selected_entries: int
    sink_insertions: int
    heavy_hitter_updates: int
    state_bytes: int
    scratch_bytes: int


class NativeStreamingAttention:
    """Stateful float32 C++ implementation of W/C/K streaming attention."""

    def __init__(
        self,
        *,
        query_heads: int,
        key_value_heads: int,
        head_dimension: int,
        local_window: int = 16,
        older_candidates: int = 8,
        older_top_k: int = 4,
        sink_tokens: int = 2,
        scale: float | None = None,
        library: str | Path | None = None,
    ) -> None:
        self.query_heads = int(query_heads)
        self.key_value_heads = int(key_value_heads)
        self.head_dimension = int(head_dimension)
        default_library = (
            Path(__file__).resolve().parents[3] / "build" / "libengram_attention.so"
        )
        self.library_path = Path(
            default_library if library is None else library
        ).resolve()
        self._library = ctypes.CDLL(str(self.library_path))
        self._configure_signatures()
        config = _Config(
            self.query_heads,
            self.key_value_heads,
            self.head_dimension,
            int(local_window),
            int(older_candidates),
            int(older_top_k),
            int(sink_tokens),
            float(self.head_dimension**-0.5 if scale is None else scale),
        )
        error = ctypes.create_string_buffer(512)
        self._handle = self._library.engram_streaming_attention_create(
            ctypes.byref(config),
            error,
            len(error),
        )
        if not self._handle:
            raise ValueError(error.value.decode("utf-8", errors="replace"))

    def _configure_signatures(self) -> None:
        pointer = ctypes.POINTER(ctypes.c_float)
        self._library.engram_streaming_attention_create.argtypes = [
            ctypes.POINTER(_Config),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._library.engram_streaming_attention_create.restype = ctypes.c_void_p
        self._library.engram_streaming_attention_destroy.argtypes = [ctypes.c_void_p]
        self._library.engram_streaming_attention_reset.argtypes = [ctypes.c_void_p]
        self._library.engram_streaming_attention_step_f32.argtypes = [
            ctypes.c_void_p,
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.POINTER(_Metrics),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._library.engram_streaming_attention_step_f32.restype = ctypes.c_int
        self._library.engram_streaming_attention_stream_f32.argtypes = [
            ctypes.c_void_p,
            pointer,
            pointer,
            pointer,
            ctypes.c_size_t,
            pointer,
            ctypes.POINTER(_Metrics),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._library.engram_streaming_attention_stream_f32.restype = ctypes.c_int

    @staticmethod
    def _array(values, shape: tuple[int, int], name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32)
        if result.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be finite")
        return np.ascontiguousarray(result)

    def step(
        self,
        query,
        key,
        value,
    ) -> tuple[np.ndarray, NativeStreamingAttentionMetrics]:
        if not self._handle:
            raise RuntimeError("native streaming attention is closed")
        query_array = self._array(
            query,
            (self.query_heads, self.head_dimension),
            "query",
        )
        key_array = self._array(
            key,
            (self.key_value_heads, self.head_dimension),
            "key",
        )
        value_array = self._array(
            value,
            (self.key_value_heads, self.head_dimension),
            "value",
        )
        output = np.empty_like(query_array)
        metrics = _Metrics()
        error = ctypes.create_string_buffer(512)
        pointer = ctypes.POINTER(ctypes.c_float)
        status = self._library.engram_streaming_attention_step_f32(
            self._handle,
            query_array.ctypes.data_as(pointer),
            key_array.ctypes.data_as(pointer),
            value_array.ctypes.data_as(pointer),
            output.ctypes.data_as(pointer),
            ctypes.byref(metrics),
            error,
            len(error),
        )
        if status:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        values = {
            name: int(getattr(metrics, name))
            for name in NativeStreamingAttentionMetrics.__dataclass_fields__
        }
        return output, NativeStreamingAttentionMetrics(**values)

    def stream(
        self,
        queries,
        keys,
        values,
    ) -> tuple[np.ndarray, NativeStreamingAttentionMetrics]:
        """Advance a position-major stream through one native ABI call."""

        if not self._handle:
            raise RuntimeError("native streaming attention is closed")
        query_array = np.asarray(queries, dtype=np.float32)
        key_array = np.asarray(keys, dtype=np.float32)
        value_array = np.asarray(values, dtype=np.float32)
        if query_array.ndim != 3:
            raise ValueError("queries must have shape [length, heads, dimension]")
        length = int(query_array.shape[0])
        expected_query = (
            length,
            self.query_heads,
            self.head_dimension,
        )
        expected_kv = (
            length,
            self.key_value_heads,
            self.head_dimension,
        )
        if length <= 0 or query_array.shape != expected_query:
            raise ValueError(f"queries must have shape {expected_query}")
        if key_array.shape != expected_kv:
            raise ValueError(f"keys must have shape {expected_kv}")
        if value_array.shape != expected_kv:
            raise ValueError(f"values must have shape {expected_kv}")
        for name, array in (
            ("queries", query_array),
            ("keys", key_array),
            ("values", value_array),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite")
        query_array = np.ascontiguousarray(query_array)
        key_array = np.ascontiguousarray(key_array)
        value_array = np.ascontiguousarray(value_array)
        output = np.empty_like(query_array)
        metrics = _Metrics()
        error = ctypes.create_string_buffer(512)
        pointer = ctypes.POINTER(ctypes.c_float)
        status = self._library.engram_streaming_attention_stream_f32(
            self._handle,
            query_array.ctypes.data_as(pointer),
            key_array.ctypes.data_as(pointer),
            value_array.ctypes.data_as(pointer),
            length,
            output.ctypes.data_as(pointer),
            ctypes.byref(metrics),
            error,
            len(error),
        )
        if status:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        metric_values = {
            name: int(getattr(metrics, name))
            for name in NativeStreamingAttentionMetrics.__dataclass_fields__
        }
        return output, NativeStreamingAttentionMetrics(**metric_values)

    def reset(self) -> None:
        if self._handle:
            self._library.engram_streaming_attention_reset(self._handle)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.engram_streaming_attention_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> NativeStreamingAttention:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["NativeStreamingAttention", "NativeStreamingAttentionMetrics"]
