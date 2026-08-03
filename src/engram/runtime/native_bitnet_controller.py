"""Controller-driven BitNet execution without decoder-layer forward calls."""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from engram.controller import FactorizedRecurrentController
from engram.evaluation.native_bitnet_parity import _torch_modules


@dataclass(frozen=True)
class ControllerForward:
    logits: Any
    next_token: int | None
    normalized_state: np.ndarray
    residual_rms: np.ndarray
    elapsed_seconds: float
    controller_seconds: float


class _StageMLPMetrics(ctypes.Structure):
    _fields_ = [
        ("elapsed_ns", ctypes.c_uint64),
        ("gate_up_stream_bytes", ctypes.c_uint64),
        ("norm_stream_bytes", ctypes.c_uint64),
        ("down_stream_bytes", ctypes.c_uint64),
        ("layer_metadata_bytes", ctypes.c_uint64),
        ("scheduled_cache_line_bytes", ctypes.c_uint64),
        ("scratch_bytes", ctypes.c_uint64),
        ("rows", ctypes.c_uint64),
        ("threads", ctypes.c_uint64),
    ]


class _FusedStreamingMetrics(ctypes.Structure):
    _fields_ = [
        ("tokens_seen", ctypes.c_uint64),
        ("local_entries", ctypes.c_uint64),
        ("active_older_entries", ctypes.c_uint64),
        ("candidate_key_bytes", ctypes.c_uint64),
        ("selected_value_bytes", ctypes.c_uint64),
        ("local_kv_bytes", ctypes.c_uint64),
        ("state_bytes", ctypes.c_uint64),
        ("scratch_bytes", ctypes.c_uint64),
    ]


class _FusedAttentionMetrics(ctypes.Structure):
    _fields_ = [
        ("qkv_projection_ns", ctypes.c_uint64),
        ("rope_ns", ctypes.c_uint64),
        ("native_attention_ns", ctypes.c_uint64),
        ("output_projection_ns", ctypes.c_uint64),
        ("packed_weight_bytes", ctypes.c_uint64),
        ("projection_scratch_bytes", ctypes.c_uint64),
        ("attention", _FusedStreamingMetrics),
    ]


class _NativeStageDescriptor(ctypes.Structure):
    _fields_ = [
        ("projection_handle", ctypes.c_void_p),
        ("query_projection", ctypes.c_size_t),
        ("key_projection", ctypes.c_size_t),
        ("value_projection", ctypes.c_size_t),
        ("output_projection", ctypes.c_size_t),
        ("attention_handles", ctypes.POINTER(ctypes.c_void_p)),
        ("input_norm_weight", ctypes.c_void_p),
        ("input_norm_epsilon", ctypes.c_float),
        ("attention_norm_weight", ctypes.c_void_p),
        ("attention_norm_epsilon", ctypes.c_float),
        ("semantic_norm_weight", ctypes.c_void_p),
        ("semantic_norm_epsilon", ctypes.c_float),
        ("semantic_scale", ctypes.c_float),
        ("episodic_scale", ctypes.c_float),
        ("semantic_layer", ctypes.c_size_t),
    ]


class _NativeControllerWeights(ctypes.Structure):
    _fields_ = [
        ("input_dim", ctypes.c_size_t),
        ("state_dim", ctypes.c_size_t),
        ("rank", ctypes.c_size_t),
        ("adapter_rank", ctypes.c_size_t),
        ("input_adapter_rank", ctypes.c_size_t),
        ("input_down", ctypes.c_void_p),
        ("recurrent_down", ctypes.c_void_p),
        ("gate_up", ctypes.c_void_p),
        ("bias", ctypes.c_void_p),
        ("stage_embedding", ctypes.c_void_p),
        ("adapter_down", ctypes.c_void_p),
        ("adapter_up", ctypes.c_void_p),
        ("input_adapter_down", ctypes.c_void_p),
        ("input_adapter_up", ctypes.c_void_p),
        ("step_scale", ctypes.c_float),
    ]


class NativeOperatorResidual:
    """ctypes binding to the float32 exact residual/RMS kernel."""

    def __init__(self, library) -> None:
        self.library_path = str(library)
        self._library = ctypes.CDLL(self.library_path)
        function = self._library.engram_operator_residual_step_f32
        pointer = ctypes.POINTER(ctypes.c_float)
        function.argtypes = [
            pointer,
            pointer,
            pointer,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            pointer,
            pointer,
        ]
        function.restype = ctypes.c_int
        self._step = function

    def step(
        self,
        state: np.ndarray,
        semantic: np.ndarray,
        episodic: np.ndarray,
        *,
        semantic_scale: float = 1.0,
        episodic_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        current = np.ascontiguousarray(state, dtype=np.float32)
        semantic_values = np.ascontiguousarray(semantic, dtype=np.float32)
        episodic_values = np.ascontiguousarray(episodic, dtype=np.float32)
        if (
            current.ndim < 2
            or semantic_values.shape != current.shape
            or episodic_values.shape != current.shape
        ):
            raise ValueError("native residual operands must have matching shapes")
        vectors = int(np.prod(current.shape[:-1]))
        width = current.shape[-1]
        output = np.empty_like(current)
        relative_rms = np.empty((*current.shape[:-1], 1), dtype=np.float32)
        pointer = ctypes.POINTER(ctypes.c_float)
        status = self._step(
            current.ctypes.data_as(pointer),
            semantic_values.ctypes.data_as(pointer),
            episodic_values.ctypes.data_as(pointer),
            vectors,
            width,
            semantic_scale,
            episodic_scale,
            output.ctypes.data_as(pointer),
            relative_rms.ctypes.data_as(pointer),
        )
        if status:
            raise RuntimeError(f"native operator residual failed with status {status}")
        return output, relative_rms


class NativeBitNetShellOps:
    """Native BF16 embedding and RMSNorm operations used around stage kernels."""

    def __init__(self, library) -> None:
        self.library_path = str(library)
        self._library = ctypes.CDLL(self.library_path)
        uint16_pointer = ctypes.POINTER(ctypes.c_uint16)
        int64_pointer = ctypes.POINTER(ctypes.c_int64)
        float_pointer = ctypes.POINTER(ctypes.c_float)

        embedding = self._library.engram_embedding_lookup_bf16
        embedding.argtypes = [
            uint16_pointer,
            ctypes.c_size_t,
            ctypes.c_size_t,
            int64_pointer,
            ctypes.c_size_t,
            uint16_pointer,
        ]
        embedding.restype = ctypes.c_int
        self._embedding = embedding

        rms_norm = self._library.engram_rms_norm_f32_to_bf16
        rms_norm.argtypes = [
            float_pointer,
            uint16_pointer,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            uint16_pointer,
        ]
        rms_norm.restype = ctypes.c_int
        self._rms_norm = rms_norm

        vocab_argmax = self._library.engram_vocab_argmax_bf16
        vocab_argmax.argtypes = [
            uint16_pointer,
            uint16_pointer,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            float_pointer,
        ]
        vocab_argmax.restype = ctypes.c_int
        self._vocab_argmax = vocab_argmax

        rope = self._library.engram_rope_bf16
        rope.argtypes = [
            uint16_pointer,
            ctypes.c_size_t,
            uint16_pointer,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            int64_pointer,
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        rope.restype = ctypes.c_int
        self._rope = rope

    @staticmethod
    def _bf16_bits(tensor) -> np.ndarray:
        torch, _, _ = _torch_modules()
        values = tensor.detach().cpu().contiguous()
        if values.dtype != torch.bfloat16:
            values = values.to(torch.bfloat16)
        return values.view(torch.uint16).numpy()

    def embedding(self, weight, token_ids):
        """Look up token rows without executing a Torch embedding operator."""

        torch, _, _ = _torch_modules()
        table = self._bf16_bits(weight)
        indices = np.ascontiguousarray(
            token_ids.detach().cpu().numpy(), dtype=np.int64
        )
        if table.ndim != 2:
            raise ValueError("embedding weight must be rank-2")
        output = np.empty((*indices.shape, table.shape[1]), dtype=np.uint16)
        uint16_pointer = ctypes.POINTER(ctypes.c_uint16)
        status = self._embedding(
            table.ctypes.data_as(uint16_pointer),
            table.shape[0],
            table.shape[1],
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            indices.size,
            output.ctypes.data_as(uint16_pointer),
        )
        if status:
            raise RuntimeError(f"native embedding lookup failed with status {status}")
        return torch.from_numpy(output).view(torch.bfloat16)

    def rms_norm(self, values: np.ndarray, weight, epsilon: float):
        """Apply BitNet's BF16 RMSNorm ordering to float32 state vectors."""

        torch, _, _ = _torch_modules()
        inputs = np.ascontiguousarray(values, dtype=np.float32)
        scales = self._bf16_bits(weight)
        if inputs.ndim < 2 or scales.ndim != 1:
            raise ValueError("RMSNorm requires vectors and a rank-1 weight")
        if inputs.shape[-1] != scales.shape[0]:
            raise ValueError("RMSNorm input and weight widths differ")
        output = np.empty(inputs.shape, dtype=np.uint16)
        vectors = int(np.prod(inputs.shape[:-1]))
        uint16_pointer = ctypes.POINTER(ctypes.c_uint16)
        status = self._rms_norm(
            inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            scales.ctypes.data_as(uint16_pointer),
            vectors,
            inputs.shape[-1],
            epsilon,
            output.ctypes.data_as(uint16_pointer),
        )
        if status:
            raise RuntimeError(f"native RMSNorm failed with status {status}")
        return torch.from_numpy(output).view(torch.bfloat16)

    def vocab_argmax(self, hidden, weight, *, threads: int) -> tuple[int, float]:
        """Return the best tied-vocabulary row without materializing logits."""

        if threads <= 0:
            raise ValueError("vocabulary projection threads must be positive")
        inputs = self._bf16_bits(hidden)
        table = self._bf16_bits(weight)
        if inputs.ndim != 1 or table.ndim != 2:
            raise ValueError("vocabulary argmax requires a vector and matrix")
        if inputs.shape[0] != table.shape[1]:
            raise ValueError("vocabulary input and weight widths differ")
        uint16_pointer = ctypes.POINTER(ctypes.c_uint16)
        token = ctypes.c_int64()
        score = ctypes.c_float()
        status = self._vocab_argmax(
            inputs.ctypes.data_as(uint16_pointer),
            table.ctypes.data_as(uint16_pointer),
            table.shape[0],
            table.shape[1],
            threads,
            ctypes.byref(token),
            ctypes.byref(score),
        )
        if status:
            raise RuntimeError(f"native vocabulary argmax failed with status {status}")
        return int(token.value), float(score.value)

    def rope(self, query, key, position_ids, *, theta: float):
        """Apply default RoPE in place to contiguous CPU BF16 projections."""

        torch, _, _ = _torch_modules()
        if (
            query.device.type != "cpu"
            or key.device.type != "cpu"
            or query.dtype != torch.bfloat16
            or key.dtype != torch.bfloat16
            or query.ndim != 4
            or key.ndim != 4
        ):
            raise ValueError("native RoPE requires rank-4 CPU BF16 query/key")
        if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
            raise ValueError("native RoPE query/key dimensions differ")
        query = query.contiguous()
        key = key.contiguous()
        positions = np.ascontiguousarray(
            position_ids.detach().cpu().numpy(), dtype=np.int64
        )
        if positions.shape[0] not in (1, query.shape[0]):
            raise ValueError("native RoPE position batch differs")
        uint16_pointer = ctypes.POINTER(ctypes.c_uint16)
        status = self._rope(
            ctypes.cast(query.data_ptr(), uint16_pointer),
            query.shape[1],
            ctypes.cast(key.data_ptr(), uint16_pointer),
            key.shape[1],
            query.shape[0],
            query.shape[2],
            query.shape[3],
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            positions.shape[0],
            theta,
        )
        if status:
            raise RuntimeError(f"native RoPE failed with status {status}")
        return query, key


class NativeStageState:
    """Persistent C++ normalized residual state for one prefill/decode call."""

    def __init__(self, library, *, vectors: int, width: int) -> None:
        self._library = ctypes.CDLL(str(library))
        self.vectors = int(vectors)
        self.width = int(width)
        lib = self._library
        lib.engram_native_stage_create.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.engram_native_stage_create.restype = ctypes.c_void_p
        lib.engram_native_stage_destroy.argtypes = [ctypes.c_void_p]
        pointer_args = [ctypes.c_void_p, ctypes.c_void_p]
        for name in (
            "engram_native_stage_begin_bf16",
            "engram_native_stage_accept_attention_bf16",
        ):
            function = getattr(lib, name)
            function.argtypes = [*pointer_args, ctypes.c_char_p, ctypes.c_size_t]
            function.restype = ctypes.c_int
        for name in (
            "engram_native_stage_attention_input_bf16",
            "engram_native_stage_semantic_input_bf16",
            "engram_native_stage_final_norm_bf16",
        ):
            function = getattr(lib, name)
            function.argtypes = [
                *pointer_args,
                ctypes.c_float,
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            function.restype = ctypes.c_int
        lib.engram_native_stage_accept_semantic_bf16.argtypes = [
            *pointer_args,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.engram_native_stage_accept_semantic_bf16.restype = ctypes.c_int
        lib.engram_native_stage_accept_controller_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(_NativeControllerWeights),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.engram_native_stage_accept_controller_f32.restype = ctypes.c_int
        lib.engram_native_stage_copy_state_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.engram_native_stage_copy_state_f32.restype = ctypes.c_int
        lib.engram_bitnet_stage_semantic_bf16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(_StageMLPMetrics),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.engram_bitnet_stage_semantic_bf16.restype = ctypes.c_int
        lib.engram_native_stage_attention_bf16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.POINTER(_FusedAttentionMetrics),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.engram_native_stage_attention_bf16.restype = ctypes.c_int
        lib.engram_native_run_stages_bf16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(_NativeStageDescriptor),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.POINTER(_FusedAttentionMetrics),
            ctypes.POINTER(_StageMLPMetrics),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.engram_native_run_stages_bf16.restype = ctypes.c_int
        error = ctypes.create_string_buffer(512)
        self._handle = lib.engram_native_stage_create(
            self.vectors, self.width, error, len(error)
        )
        if not self._handle:
            raise RuntimeError(error.value.decode("utf-8", "replace"))

    def _call(self, name: str, *arguments) -> None:
        error = ctypes.create_string_buffer(512)
        status = getattr(self._library, name)(
            self._handle, *arguments, error, len(error)
        )
        if status:
            raise RuntimeError(error.value.decode("utf-8", "replace"))

    def _output(self):
        torch, _, _ = _torch_modules()
        return torch.empty(
            (self.vectors, self.width), dtype=torch.bfloat16, device="cpu"
        )

    def begin(self, embedding) -> None:
        source = embedding.contiguous().view(self.vectors, self.width)
        self._call(
            "engram_native_stage_begin_bf16",
            ctypes.c_void_p(source.data_ptr()),
        )

    def norm(self, name: str, weight, epsilon: float):
        output = self._output()
        scale = weight.detach().cpu().contiguous()
        self._call(
            name,
            ctypes.c_void_p(scale.data_ptr()),
            float(epsilon),
            ctypes.c_void_p(output.data_ptr()),
        )
        return output

    def attention_input(self, weight, epsilon: float):
        return self.norm(
            "engram_native_stage_attention_input_bf16", weight, epsilon
        )

    def accept_attention(self, output) -> None:
        source = output.contiguous().view(self.vectors, self.width)
        self._call(
            "engram_native_stage_accept_attention_bf16",
            ctypes.c_void_p(source.data_ptr()),
        )

    def semantic_input(self, weight, epsilon: float):
        return self.norm(
            "engram_native_stage_semantic_input_bf16", weight, epsilon
        )

    def accept_semantic(
        self, output, *, semantic_scale: float, episodic_scale: float
    ) -> None:
        source = output.contiguous().view(self.vectors, self.width)
        self._call(
            "engram_native_stage_accept_semantic_bf16",
            ctypes.c_void_p(source.data_ptr()),
            float(semantic_scale),
            float(episodic_scale),
        )

    def accept_controller(
        self,
        output,
        controller: FactorizedRecurrentController,
        *,
        stage: int,
        semantic_scale: float,
        episodic_scale: float,
    ) -> None:
        """Apply a schema-v3 factorized correction in the native stage state."""

        source = output.contiguous().view(self.vectors, self.width)
        input_down = np.ascontiguousarray(controller.input_down, dtype=np.float32)
        recurrent_down = np.ascontiguousarray(
            controller.recurrent_down, dtype=np.float32
        )
        gate_up = np.ascontiguousarray(controller.gate_up, dtype=np.float32)
        bias = np.ascontiguousarray(controller.bias, dtype=np.float32)
        stage_embedding = np.ascontiguousarray(
            controller.stage_embeddings[stage], dtype=np.float32
        )
        adapter_down = np.ascontiguousarray(
            controller.adapter_down[stage], dtype=np.float32
        )
        adapter_up = np.ascontiguousarray(
            controller.adapter_up[stage], dtype=np.float32
        )
        input_adapter_down = np.ascontiguousarray(
            controller.input_adapter_down[stage], dtype=np.float32
        )
        input_adapter_up = np.ascontiguousarray(
            controller.input_adapter_up[stage], dtype=np.float32
        )
        weights = _NativeControllerWeights(
            input_dim=controller.input_dim,
            state_dim=controller.state_dim,
            rank=controller.rank,
            adapter_rank=controller.adapter_rank,
            input_adapter_rank=controller.input_adapter_rank,
            input_down=input_down.ctypes.data,
            recurrent_down=recurrent_down.ctypes.data,
            gate_up=gate_up.ctypes.data,
            bias=bias.ctypes.data,
            stage_embedding=stage_embedding.ctypes.data,
            adapter_down=adapter_down.ctypes.data,
            adapter_up=adapter_up.ctypes.data,
            input_adapter_down=input_adapter_down.ctypes.data,
            input_adapter_up=input_adapter_up.ctypes.data,
            step_scale=float(controller.step_scale[stage]),
        )
        self._call(
            "engram_native_stage_accept_controller_f32",
            ctypes.c_void_p(source.data_ptr()),
            float(semantic_scale),
            float(episodic_scale),
            ctypes.byref(weights),
        )

    def run_semantic(
        self,
        mlp,
        weight,
        epsilon: float,
        *,
        semantic_scale: float,
        episodic_scale: float,
    ) -> float:
        """Normalize, run a packed MLP, and update state in one native call."""

        scale = weight.detach().cpu().contiguous()
        metrics = _StageMLPMetrics()
        error = ctypes.create_string_buffer(512)
        started = time.perf_counter()
        status = self._library.engram_bitnet_stage_semantic_bf16(
            mlp.kernel._handle,
            self._handle,
            mlp.layer,
            ctypes.c_void_p(scale.data_ptr()),
            float(epsilon),
            self.vectors,
            float(semantic_scale),
            float(episodic_scale),
            ctypes.byref(metrics),
            error,
            len(error),
        )
        elapsed = time.perf_counter() - started
        if status:
            raise RuntimeError(error.value.decode("utf-8", "replace"))
        call = {
            name: int(getattr(metrics, name))
            for name, _ in metrics._fields_
        }
        call["layer"] = int(mlp.layer)
        mlp.kernel.calls.append(call)
        return max(elapsed - metrics.elapsed_ns / 1.0e9, 0.0)

    def run_attention(self, attention, position_ids, weight, epsilon: float) -> None:
        """Execute packed projections and persistent attention in one C call."""

        torch, _, _ = _torch_modules()
        batch = int(position_ids.shape[0])
        length = int(position_ids.shape[1])
        expected = torch.arange(
            attention._next_position,
            attention._next_position + length,
            dtype=position_ids.dtype,
            device=position_ids.device,
        )
        if not torch.all(position_ids == expected.unsqueeze(0)):
            raise ValueError("native attention positions must advance contiguously")
        attention._ensure_batch(batch)
        handles = (ctypes.c_void_p * batch)(
            *(cache._handle for cache in attention._caches)
        )
        positions = np.ascontiguousarray(
            position_ids.detach().cpu().numpy(), dtype=np.int64
        )
        scale = weight.detach().cpu().contiguous()
        sub_scale = attention.attn_sub_norm.weight.detach().cpu().contiguous()
        rope_parameters = getattr(attention.config, "rope_parameters", {})
        theta = float(
            rope_parameters.get(
                "rope_theta",
                getattr(attention.config, "rope_theta", 10000.0),
            )
        )
        projections = (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            attention.o_proj,
        )
        if any(
            projection.kernel is not projections[0].kernel
            for projection in projections
        ):
            raise ValueError("native attention projections must share one kernel")
        metrics = _FusedAttentionMetrics()
        error = ctypes.create_string_buffer(512)
        status = self._library.engram_native_stage_attention_bf16(
            self._handle,
            projections[0].kernel._handle,
            projections[0].projection,
            projections[1].projection,
            projections[2].projection,
            projections[3].projection,
            handles,
            batch,
            length,
            self.width,
            attention.query_heads,
            attention.key_value_heads,
            attention.head_dim,
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            positions.shape[0],
            theta,
            ctypes.c_void_p(scale.data_ptr()),
            float(epsilon),
            ctypes.c_void_p(sub_scale.data_ptr()),
            float(attention.attn_sub_norm.variance_epsilon),
            ctypes.byref(metrics),
            error,
            len(error),
        )
        if status:
            raise RuntimeError(error.value.decode("utf-8", "replace"))
        native = metrics.attention
        attention._next_position += length
        attention._qkv_projection_seconds += metrics.qkv_projection_ns / 1.0e9
        attention._rope_seconds += metrics.rope_ns / 1.0e9
        attention._native_stream_seconds += metrics.native_attention_ns / 1.0e9
        attention._output_projection_seconds += (
            metrics.output_projection_ns / 1.0e9
        )
        attention._native_stream_calls += batch
        attention._logical_read_bytes += (
            native.candidate_key_bytes
            + native.selected_value_bytes
            + native.local_kv_bytes
        )
        attention._maximum_state_bytes = max(
            attention._maximum_state_bytes,
            int(native.state_bytes),
        )
        attention._maximum_scratch_bytes = max(
            attention._maximum_scratch_bytes,
            int(native.scratch_bytes + metrics.projection_scratch_bytes),
        )

    @staticmethod
    def _record_attention(attention, metrics: _FusedAttentionMetrics, batch: int):
        native = metrics.attention
        attention._qkv_projection_seconds += metrics.qkv_projection_ns / 1.0e9
        attention._rope_seconds += metrics.rope_ns / 1.0e9
        attention._native_stream_seconds += metrics.native_attention_ns / 1.0e9
        attention._output_projection_seconds += (
            metrics.output_projection_ns / 1.0e9
        )
        attention._native_stream_calls += batch
        attention._logical_read_bytes += (
            native.candidate_key_bytes
            + native.selected_value_bytes
            + native.local_kv_bytes
        )
        attention._maximum_state_bytes = max(
            attention._maximum_state_bytes,
            int(native.state_bytes),
        )
        attention._maximum_scratch_bytes = max(
            attention._maximum_scratch_bytes,
            int(native.scratch_bytes + metrics.projection_scratch_bytes),
        )

    def run_stages(self, layers, position_ids, controller) -> float:
        """Execute the complete transformer-depth loop in one native call."""

        torch, _, _ = _torch_modules()
        batch, length = (int(value) for value in position_ids.shape)
        positions = np.ascontiguousarray(
            position_ids.detach().cpu().numpy(), dtype=np.int64
        )
        descriptors = (_NativeStageDescriptor * len(layers))()
        attention_metrics = (_FusedAttentionMetrics * len(layers))()
        semantic_metrics = (_StageMLPMetrics * len(layers))()
        cache_arrays = []
        retained_weights = []
        semantic_kernel = None
        theta = None
        for stage, layer in enumerate(layers):
            attention = layer.self_attn
            expected = torch.arange(
                attention._next_position,
                attention._next_position + length,
                dtype=position_ids.dtype,
                device=position_ids.device,
            )
            if not torch.all(position_ids == expected.unsqueeze(0)):
                raise ValueError(
                    "native attention positions must advance contiguously"
                )
            attention._ensure_batch(batch)
            handles = (ctypes.c_void_p * batch)(
                *(cache._handle for cache in attention._caches)
            )
            cache_arrays.append(handles)
            projections = (
                attention.q_proj,
                attention.k_proj,
                attention.v_proj,
                attention.o_proj,
            )
            if any(
                projection.kernel is not projections[0].kernel
                for projection in projections
            ):
                raise ValueError(
                    "native attention projections must share one kernel"
                )
            current_semantic_kernel = layer.mlp.kernel
            if semantic_kernel is None:
                semantic_kernel = current_semantic_kernel
            elif current_semantic_kernel is not semantic_kernel:
                raise ValueError("native MLP stages must share one kernel")
            weights = (
                layer.input_layernorm.weight.detach().cpu().contiguous(),
                attention.attn_sub_norm.weight.detach().cpu().contiguous(),
                layer.post_attention_layernorm.weight.detach().cpu().contiguous(),
            )
            retained_weights.extend(weights)
            rope_parameters = getattr(attention.config, "rope_parameters", {})
            current_theta = float(
                rope_parameters.get(
                    "rope_theta",
                    getattr(attention.config, "rope_theta", 10000.0),
                )
            )
            if theta is None:
                theta = current_theta
            elif current_theta != theta:
                raise ValueError("native stages require one shared RoPE theta")
            descriptors[stage] = _NativeStageDescriptor(
                projections[0].kernel._handle,
                projections[0].projection,
                projections[1].projection,
                projections[2].projection,
                projections[3].projection,
                handles,
                ctypes.c_void_p(weights[0].data_ptr()),
                float(layer.input_layernorm.variance_epsilon),
                ctypes.c_void_p(weights[1].data_ptr()),
                float(attention.attn_sub_norm.variance_epsilon),
                ctypes.c_void_p(weights[2].data_ptr()),
                float(layer.post_attention_layernorm.variance_epsilon),
                float(controller.operator_residual_scale[stage, 0]),
                float(controller.operator_residual_scale[stage, 1]),
                layer.mlp.layer,
            )
        assert semantic_kernel is not None and theta is not None
        error = ctypes.create_string_buffer(512)
        started = time.perf_counter()
        status = self._library.engram_native_run_stages_bf16(
            self._handle,
            semantic_kernel._handle,
            descriptors,
            len(layers),
            batch,
            length,
            self.width,
            layers[0].self_attn.query_heads,
            layers[0].self_attn.key_value_heads,
            layers[0].self_attn.head_dim,
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            positions.shape[0],
            theta,
            attention_metrics,
            semantic_metrics,
            error,
            len(error),
        )
        elapsed = time.perf_counter() - started
        if status:
            raise RuntimeError(error.value.decode("utf-8", "replace"))
        measured_ns = 0
        for stage, layer in enumerate(layers):
            attention = layer.self_attn
            attention._next_position += length
            self._record_attention(attention, attention_metrics[stage], batch)
            metric = semantic_metrics[stage]
            call = {
                name: int(getattr(metric, name))
                for name, _ in metric._fields_
            }
            call["layer"] = int(layer.mlp.layer)
            semantic_kernel.calls.append(call)
            measured_ns += (
                attention_metrics[stage].qkv_projection_ns
                + attention_metrics[stage].rope_ns
                + attention_metrics[stage].native_attention_ns
                + attention_metrics[stage].output_projection_ns
                + metric.elapsed_ns
            )
        return max(elapsed - measured_ns / 1.0e9, 0.0)

    def final_norm(self, weight, epsilon: float):
        return self.norm("engram_native_stage_final_norm_bf16", weight, epsilon)

    def copy_state(self, shape) -> tuple[np.ndarray, np.ndarray]:
        state = np.empty((self.vectors, self.width), dtype=np.float32)
        rms = np.empty((self.vectors, 1), dtype=np.float32)
        self._call(
            "engram_native_stage_copy_state_f32",
            ctypes.c_void_p(state.ctypes.data),
            ctypes.c_void_p(rms.ctypes.data),
        )
        return state.reshape(shape), rms.reshape((*shape[:-1], 1))

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.engram_native_stage_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class ControllerDrivenBitNet:
    """Dispatch native stage operators from normalized controller state.

    The controller owns the vector residual transition. A separate scalar RMS
    is carried for every token because BitNet's normalized operators are
    approximately scale invariant while their residual outputs are not.
    """

    def __init__(
        self,
        model,
        controller: FactorizedRecurrentController,
        *,
        native_library=None,
        native_vocab_argmax: bool = False,
        vocab_threads: int = 1,
    ) -> None:
        layers = model.model.layers
        hidden_size = int(model.config.hidden_size)
        if len(layers) != controller.num_stages:
            raise ValueError("controller stage count does not match BitNet")
        if hidden_size != controller.state_dim:
            raise ValueError("controller state width does not match BitNet")
        if not controller.has_operator_residual:
            raise ValueError("controller must preserve operator residuals")
        self.model = model
        self.controller = controller
        self.native_residual = (
            NativeOperatorResidual(native_library)
            if native_library is not None
            else None
        )
        self.native_shell = (
            NativeBitNetShellOps(native_library)
            if native_library is not None
            else None
        )
        if native_vocab_argmax and self.native_shell is None:
            raise ValueError("native vocabulary argmax requires a native library")
        self.native_vocab_argmax = bool(native_vocab_argmax)
        self.vocab_threads = int(vocab_threads)
        self.decoder_layer_forward_calls = 0

    @staticmethod
    def _rms(values: np.ndarray) -> np.ndarray:
        return np.sqrt(
            np.mean(np.square(values), axis=-1, keepdims=True)
        ).clip(1e-6)

    def forward(self, input_ids, *, position_ids) -> ControllerForward:
        """Run prefill or decode tokens through explicit stage operators."""

        torch, _, _ = _torch_modules()
        if input_ids.ndim != 2 or position_ids.ndim != 2:
            raise ValueError("input_ids and position_ids must be rank-2")
        if input_ids.shape != position_ids.shape:
            raise ValueError("input_ids and position_ids shapes must match")
        started = time.perf_counter()
        with torch.inference_mode():
            if self.native_shell is None:
                embedded = self.model.model.embed_tokens(input_ids)
            else:
                embedded = self.native_shell.embedding(
                    self.model.model.embed_tokens.weight,
                    input_ids,
                )
            initial = embedded.float().cpu().numpy()
            residual_rms = self._rms(initial)
            state = initial / residual_rms
            token_embedding = state.copy()
            position_embeddings = (
                None
                if self.native_shell is not None
                else self.model.model.rotary_emb(
                    embedded,
                    position_ids=position_ids,
                )
            )
            if self.native_shell is not None and self.native_residual is not None:
                stage_state = NativeStageState(
                    self.native_shell.library_path,
                    vectors=input_ids.numel(),
                    width=embedded.shape[-1],
                )
                stage_state.begin(embedded)
                layers = self.model.model.layers
                fully_native = all(
                    hasattr(layer.self_attn.q_proj, "kernel")
                    and hasattr(layer.self_attn, "_caches")
                    and hasattr(layer.mlp, "kernel")
                    for layer in layers
                )
                if fully_native and not np.any(self.controller.step_scale != 0.0):
                    controller_seconds = stage_state.run_stages(
                        layers,
                        position_ids,
                        self.controller,
                    )
                else:
                    controller_seconds = 0.0
                    for stage, layer in enumerate(layers):
                        if (
                            hasattr(layer.self_attn.q_proj, "kernel")
                            and hasattr(layer.self_attn, "_caches")
                        ):
                            stage_state.run_attention(
                                layer.self_attn,
                                position_ids,
                                layer.input_layernorm.weight,
                                layer.input_layernorm.variance_epsilon,
                            )
                        else:
                            attention_input = stage_state.attention_input(
                                layer.input_layernorm.weight,
                                layer.input_layernorm.variance_epsilon,
                            ).view_as(embedded)
                            attention_output, _ = layer.self_attn(
                                hidden_states=attention_input,
                                attention_mask=None,
                                position_ids=position_ids,
                                past_key_values=None,
                                use_cache=False,
                                position_embeddings=position_embeddings,
                            )
                            stage_state.accept_attention(attention_output)
                        assert (
                            self.controller.operator_residual_scale is not None
                        )
                        semantic_scale = float(
                            self.controller.operator_residual_scale[stage, 0]
                        )
                        episodic_scale = float(
                            self.controller.operator_residual_scale[stage, 1]
                        )
                        if (
                            hasattr(layer.mlp, "kernel")
                            and not np.any(self.controller.step_scale != 0.0)
                        ):
                            controller_seconds += stage_state.run_semantic(
                                layer.mlp,
                                layer.post_attention_layernorm.weight,
                                layer.post_attention_layernorm.variance_epsilon,
                                semantic_scale=semantic_scale,
                                episodic_scale=episodic_scale,
                            )
                        else:
                            controller_started = time.perf_counter()
                            semantic_input = stage_state.semantic_input(
                                layer.post_attention_layernorm.weight,
                                layer.post_attention_layernorm.variance_epsilon,
                            ).view_as(embedded)
                            semantic_output = layer.mlp(semantic_input)
                            if np.any(self.controller.step_scale != 0.0):
                                stage_state.accept_controller(
                                    semantic_output,
                                    self.controller,
                                    stage=stage,
                                    semantic_scale=semantic_scale,
                                    episodic_scale=episodic_scale,
                                )
                            else:
                                stage_state.accept_semantic(
                                    semantic_output,
                                    semantic_scale=semantic_scale,
                                    episodic_scale=episodic_scale,
                                )
                            controller_seconds += (
                                time.perf_counter() - controller_started
                            )
                hidden = stage_state.final_norm(
                    self.model.model.norm.weight,
                    self.model.model.norm.variance_epsilon,
                ).view_as(embedded)
                state, residual_rms = stage_state.copy_state(
                    tuple(embedded.shape)
                )
                stage_state.close()
                if self.native_vocab_argmax:
                    next_token, _ = self.native_shell.vocab_argmax(
                        hidden[0, -1],
                        self.model.lm_head.weight,
                        threads=self.vocab_threads,
                    )
                    logits = None
                else:
                    logits = self.model.lm_head(hidden).float()
                return ControllerForward(
                    logits=logits,
                    next_token=(
                        next_token if self.native_vocab_argmax else None
                    ),
                    normalized_state=state,
                    residual_rms=residual_rms,
                    elapsed_seconds=time.perf_counter() - started,
                    controller_seconds=controller_seconds,
                )
            controller_seconds = 0.0
            for stage, layer in enumerate(self.model.model.layers):
                state_tensor = torch.from_numpy(state).to(
                    device=embedded.device,
                    dtype=embedded.dtype,
                )
                if self.native_shell is None:
                    attention_input = layer.input_layernorm(state_tensor)
                else:
                    attention_input = self.native_shell.rms_norm(
                        state,
                        layer.input_layernorm.weight,
                        layer.input_layernorm.variance_epsilon,
                    )
                attention_output, _ = layer.self_attn(
                    hidden_states=attention_input,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )
                attention_normalized = (
                    attention_output.float().cpu().numpy() / residual_rms
                )
                post_attention = state + attention_normalized
                post_attention_tensor = torch.from_numpy(post_attention).to(
                    device=embedded.device,
                    dtype=embedded.dtype,
                )
                if self.native_shell is None:
                    semantic_input = layer.post_attention_layernorm(
                        post_attention_tensor
                    )
                else:
                    semantic_input = self.native_shell.rms_norm(
                        post_attention,
                        layer.post_attention_layernorm.weight,
                        layer.post_attention_layernorm.variance_epsilon,
                    )
                semantic_output = layer.mlp(semantic_input)
                semantic_normalized = (
                    semantic_output.float().cpu().numpy() / residual_rms
                )
                supplied = np.concatenate(
                    (
                        token_embedding,
                        semantic_normalized,
                        attention_normalized,
                    ),
                    axis=-1,
                )
                controller_started = time.perf_counter()
                if self.native_residual is None:
                    next_state = self.controller.step(
                        state,
                        supplied,
                        stage=stage,
                    )
                    relative_rms = self._rms(
                        state + attention_normalized + semantic_normalized
                    )
                else:
                    assert self.controller.operator_residual_scale is not None
                    next_state, relative_rms = self.native_residual.step(
                        state,
                        semantic_normalized,
                        attention_normalized,
                        semantic_scale=float(
                            self.controller.operator_residual_scale[stage, 0]
                        ),
                        episodic_scale=float(
                            self.controller.operator_residual_scale[stage, 1]
                        ),
                    )
                controller_seconds += time.perf_counter() - controller_started
                residual_rms = residual_rms * relative_rms
                state = next_state

            final_state = torch.from_numpy(state).to(
                device=embedded.device,
                dtype=embedded.dtype,
            )
            if self.native_shell is None:
                hidden = self.model.model.norm(final_state)
            else:
                hidden = self.native_shell.rms_norm(
                    state,
                    self.model.model.norm.weight,
                    self.model.model.norm.variance_epsilon,
                )
            if self.native_vocab_argmax:
                if hidden.shape[0] != 1:
                    raise ValueError("native vocabulary argmax requires batch size one")
                next_token, _ = self.native_shell.vocab_argmax(
                    hidden[0, -1],
                    self.model.lm_head.weight,
                    threads=self.vocab_threads,
                )
                logits = None
            else:
                logits = self.model.lm_head(hidden).float()
        return ControllerForward(
            logits=logits,
            next_token=next_token if self.native_vocab_argmax else None,
            normalized_state=state,
            residual_rms=residual_rms,
            elapsed_seconds=time.perf_counter() - started,
            controller_seconds=controller_seconds,
        )


__all__ = [
    "ControllerDrivenBitNet",
    "ControllerForward",
    "NativeBitNetShellOps",
    "NativeOperatorResidual",
    "NativeStageState",
]
