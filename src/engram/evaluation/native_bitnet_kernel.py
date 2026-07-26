"""Direct CPU execution for ``native_bitnet_phase_base3_v1`` artifacts."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Sequence

from engram.evaluation.native_bitnet_parity import (
    _disable_bitnet_torch_compile,
    _disable_broken_optional_transformers_dependencies,
    _load_reference_model,
    _logit_metrics,
    _tensor_metrics,
    _torch_modules,
)
from engram.models.native_bitnet import (
    OFFICIAL_NATIVE_BITNET_REPO,
    OFFICIAL_NATIVE_BITNET_REVISION,
    OFFICIAL_NATIVE_BITNET_WEIGHT_SHA256,
    NativeBitNetValidationError,
    _resolve_full_source,
    load_native_bitnet_artifact,
    native_bitnet_repack_traffic,
)
from engram.utils import atomic_json, sha256_file


class NativeBitNetKernelError(RuntimeError):
    """Raised when the direct native kernel rejects or fails a request."""


class _NativeMetrics(ctypes.Structure):
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

    def to_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name, _ in self._fields_}


def _default_library_path() -> Path:
    configured = os.environ.get("ENGRAM_BITNET_LIBRARY")
    if configured:
        return Path(configured).expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    return repository / "build" / "libengram_bitnet.so"


def _load_library(path: str | Path | None = None):
    library_path = Path(path).resolve() if path is not None else _default_library_path()
    if not library_path.is_file():
        raise NativeBitNetKernelError(
            f"native BitNet library is missing: {library_path}; "
            "run `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release` "
            "and `cmake --build build`"
        )
    library = ctypes.CDLL(str(library_path))
    library.engram_bitnet_open.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.engram_bitnet_open.restype = ctypes.c_void_p
    library.engram_bitnet_close.argtypes = [ctypes.c_void_p]
    library.engram_bitnet_close.restype = None
    for name in (
        "engram_bitnet_layer_count",
        "engram_bitnet_hidden_size",
        "engram_bitnet_intermediate_size",
        "engram_bitnet_thread_count",
        "engram_bitnet_artifact_bytes",
    ):
        function = getattr(library, name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_size_t
    library.engram_bitnet_forward_bf16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.POINTER(_NativeMetrics),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.engram_bitnet_forward_bf16.restype = ctypes.c_int
    library.engram_bitnet_forward_oracle_bf16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.POINTER(_NativeMetrics),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.engram_bitnet_forward_oracle_bf16.restype = ctypes.c_int
    return library, library_path


class NativeBitNetCPUKernel:
    """Persistent ctypes binding to the memory-mapped C++ phase-stream kernel."""

    def __init__(
        self,
        artifact: str | Path,
        *,
        threads: int = 12,
        library: str | Path | None = None,
        expected_sha256: str | None = None,
    ) -> None:
        self.artifact = Path(artifact).resolve()
        if threads <= 0:
            raise ValueError("threads must be positive")
        self.artifact_sha256 = sha256_file(self.artifact)
        if expected_sha256 is not None:
            expected = expected_sha256.lower()
            if expected != self.artifact_sha256:
                raise NativeBitNetKernelError("native BitNet artifact SHA-256 mismatch")
        self._library, self.library_path = _load_library(library)
        error = ctypes.create_string_buffer(1024)
        self._handle = self._library.engram_bitnet_open(
            os.fsencode(self.artifact),
            threads,
            error,
            len(error),
        )
        if not self._handle:
            raise NativeBitNetKernelError(error.value.decode("utf-8", "replace"))
        self.layer_count = int(self._library.engram_bitnet_layer_count(self._handle))
        self.hidden_size = int(self._library.engram_bitnet_hidden_size(self._handle))
        self.intermediate_size = int(
            self._library.engram_bitnet_intermediate_size(self._handle)
        )
        self.thread_count = int(self._library.engram_bitnet_thread_count(self._handle))
        self.artifact_bytes = int(
            self._library.engram_bitnet_artifact_bytes(self._handle)
        )
        self.calls: list[dict[str, int]] = []

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.engram_bitnet_close(self._handle)
            self._handle = None

    def __enter__(self) -> NativeBitNetCPUKernel:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def clear_metrics(self) -> None:
        self.calls.clear()

    def forward(self, layer: int, hidden):
        torch, _, _ = _torch_modules()
        if not self._handle:
            raise NativeBitNetKernelError("native BitNet kernel is closed")
        if hidden.device.type != "cpu" or hidden.dtype != torch.bfloat16:
            raise NativeBitNetKernelError(
                "native BitNet kernel requires a CPU BF16 tensor"
            )
        if hidden.ndim < 1 or hidden.shape[-1] != self.hidden_size:
            raise NativeBitNetKernelError("native BitNet hidden shape is invalid")
        if not 0 <= int(layer) < self.layer_count:
            raise NativeBitNetKernelError("native BitNet layer index is invalid")
        source = hidden.contiguous()
        rows = source.numel() // self.hidden_size
        output = torch.empty_like(source)
        metrics = _NativeMetrics()
        error = ctypes.create_string_buffer(1024)
        status = self._library.engram_bitnet_forward_bf16(
            self._handle,
            int(layer),
            ctypes.c_void_p(source.data_ptr()),
            rows,
            ctypes.c_void_p(output.data_ptr()),
            ctypes.byref(metrics),
            error,
            len(error),
        )
        if status:
            raise NativeBitNetKernelError(error.value.decode("utf-8", "replace"))
        call = metrics.to_dict()
        call["layer"] = int(layer)
        self.calls.append(call)
        return output.reshape(hidden.shape)

    def forward_oracle(self, layer: int, hidden, *, top_k: int):
        torch, _, _ = _torch_modules()
        if not self._handle:
            raise NativeBitNetKernelError("native BitNet kernel is closed")
        if hidden.device.type != "cpu" or hidden.dtype != torch.bfloat16:
            raise NativeBitNetKernelError(
                "native BitNet kernel requires a CPU BF16 tensor"
            )
        if hidden.ndim < 1 or hidden.shape[-1] != self.hidden_size:
            raise NativeBitNetKernelError("native BitNet hidden shape is invalid")
        if not 0 <= int(layer) < self.layer_count:
            raise NativeBitNetKernelError("native BitNet layer index is invalid")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 0 < top_k <= self.intermediate_size
        ):
            raise NativeBitNetKernelError("native BitNet oracle top-K is invalid")
        source = hidden.contiguous()
        rows = source.numel() // self.hidden_size
        output = torch.empty_like(source)
        metrics = _NativeMetrics()
        error = ctypes.create_string_buffer(1024)
        status = self._library.engram_bitnet_forward_oracle_bf16(
            self._handle,
            int(layer),
            ctypes.c_void_p(source.data_ptr()),
            rows,
            int(top_k),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.byref(metrics),
            error,
            len(error),
        )
        if status:
            raise NativeBitNetKernelError(error.value.decode("utf-8", "replace"))
        call = metrics.to_dict()
        call["layer"] = int(layer)
        call["oracle_top_k"] = int(top_k)
        self.calls.append(call)
        return output.reshape(hidden.shape)


def _kernel_mlp_class():
    _, nn, _ = _torch_modules()

    class NativeKernelBitNetMLP(nn.Module):
        def __init__(self, kernel: NativeBitNetCPUKernel, layer: int) -> None:
            super().__init__()
            self.kernel = kernel
            self.layer = int(layer)

        def forward(self, hidden_states):
            return self.kernel.forward(self.layer, hidden_states)

    return NativeKernelBitNetMLP


def _validate_expected_sha256(actual: str, expected: str | None, label: str) -> str:
    if expected is None:
        raise NativeBitNetValidationError(f"{label} expected SHA-256 is required")
    normalized = expected.lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise NativeBitNetValidationError(f"{label} SHA-256 is malformed")
    if actual != normalized:
        raise NativeBitNetValidationError(f"{label} SHA-256 mismatch")
    return normalized


def _load_frozen_sequences(
    dataset: str | Path,
    *,
    sequence_count: int,
    record_offset: int = 0,
) -> tuple[list[str], dict[str, Any]]:
    if record_offset < 0:
        raise ValueError("record_offset must be nonnegative")
    source = Path(dataset).resolve()
    texts: list[str] = []
    usable_records = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise NativeBitNetValidationError(
                    f"invalid confirmation JSONL at line {line_number}"
                ) from exc
            text = record.get("text") if isinstance(record, dict) else None
            if not isinstance(text, str) or not text:
                raise NativeBitNetValidationError(
                    f"confirmation record {line_number} has no text"
                )
            if usable_records < record_offset:
                usable_records += 1
                continue
            usable_records += 1
            texts.append(text)
            if len(texts) == sequence_count:
                break
    if len(texts) != sequence_count:
        raise NativeBitNetValidationError(
            f"confirmation corpus has {len(texts)} usable sequences; "
            f"required {sequence_count}"
        )
    return texts, {
        "path": str(source),
        "sha256": sha256_file(source),
        "selection_rule": (
            f"{sequence_count}_nonempty_records_in_file_order_after_"
            f"offset_{record_offset}"
        ),
        "record_offset": record_offset,
        "unique_sequences": len(set(texts)),
    }


def evaluate_native_bitnet_kernel_confirmation(
    model: str | Path,
    artifact: str | Path,
    dataset: str | Path,
    *,
    out: str | Path,
    artifact_sha256: str,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    library: str | Path | None = None,
    threads: int = 12,
    sequence_count: int = 8,
    prediction_positions: int = 256,
    record_offset: int = 0,
    parity_layers: Sequence[int] = (0, 14, 29),
    parity_states: int = 2,
) -> dict[str, Any]:
    """Run direct-kernel parity and the frozen causal confirmation protocol."""

    if sequence_count <= 0 or prediction_positions <= 0:
        raise ValueError("confirmation evidence counts must be positive")
    if prediction_positions % sequence_count:
        raise ValueError("prediction_positions must divide evenly across sequences")
    predictions_per_sequence = prediction_positions // sequence_count
    tokens_per_sequence = predictions_per_sequence + 1
    artifact_path = Path(artifact).resolve()
    loaded_artifact = load_native_bitnet_artifact(artifact_path)
    _validate_expected_sha256(
        loaded_artifact.payload_sha256,
        artifact_sha256,
        "artifact",
    )
    model_path, repo_id, resolved_revision = _resolve_full_source(
        model,
        revision=revision,
        cache_dir=cache_dir,
    )
    official_reference = (
        repo_id == OFFICIAL_NATIVE_BITNET_REPO
        and resolved_revision == OFFICIAL_NATIVE_BITNET_REVISION
    )
    if not official_reference:
        raise NativeBitNetValidationError(
            "formal native BitNet confirmation requires the pinned official reference"
        )
    weight_path = model_path / "model.safetensors"
    weight_sha_before = sha256_file(weight_path)
    if weight_sha_before != OFFICIAL_NATIVE_BITNET_WEIGHT_SHA256:
        raise NativeBitNetValidationError("pinned reference weight SHA-256 mismatch")

    texts, dataset_evidence = _load_frozen_sequences(
        dataset,
        sequence_count=sequence_count,
        record_offset=record_offset,
    )
    if dataset_evidence["unique_sequences"] != sequence_count:
        raise NativeBitNetValidationError("confirmation sequences must be unique")

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    _disable_broken_optional_transformers_dependencies()
    _disable_bitnet_torch_compile()
    torch, _, functional = _torch_modules()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    encoded: list[list[int]] = []
    for index, text in enumerate(texts):
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        if len(token_ids) < tokens_per_sequence:
            raise NativeBitNetValidationError(
                f"confirmation sequence {index} has only {len(token_ids)} tokens; "
                f"required {tokens_per_sequence}"
            )
        encoded.append([int(value) for value in token_ids[:tokens_per_sequence]])
    input_ids = torch.tensor(encoded, dtype=torch.long)

    reference_model, load_seconds, materialization = _load_reference_model(model_path)
    weight_sha_after = sha256_file(weight_path)
    if weight_sha_after != weight_sha_before:
        raise NativeBitNetValidationError("pinned reference changed while loading")
    if (
        reference_model.config.hidden_size != loaded_artifact.hidden_size
        or reference_model.config.intermediate_size != loaded_artifact.intermediate_size
        or len(reference_model.model.layers) != len(loaded_artifact.layers)
    ):
        raise NativeBitNetValidationError("reference/artifact dimensions differ")

    with NativeBitNetCPUKernel(
        artifact_path,
        threads=threads,
        library=library,
        expected_sha256=artifact_sha256,
    ) as kernel:
        if (
            kernel.hidden_size != loaded_artifact.hidden_size
            or kernel.intermediate_size != loaded_artifact.intermediate_size
            or kernel.layer_count != len(loaded_artifact.layers)
            or kernel.artifact_bytes != loaded_artifact.serialized_artifact_bytes
        ):
            raise NativeBitNetValidationError("native kernel/artifact metadata differs")

        NativeKernelBitNetMLP = _kernel_mlp_class()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260724)
        parity_input = torch.randn(
            parity_states,
            1,
            kernel.hidden_size,
            generator=generator,
            dtype=torch.bfloat16,
        )
        parity: dict[str, Any] = {}
        kernel.clear_metrics()
        with torch.inference_mode():
            for layer_index in dict.fromkeys(int(value) for value in parity_layers):
                if not 0 <= layer_index < kernel.layer_count:
                    raise ValueError("parity layer is outside the artifact")
                expected = reference_model.model.layers[layer_index].mlp(parity_input)
                actual = kernel.forward(layer_index, parity_input)
                parity[str(layer_index)] = _tensor_metrics(expected, actual)
        parity_calls = list(kernel.calls)

        with torch.inference_mode():
            started = time.perf_counter()
            reference_output = reference_model(
                input_ids=input_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            reference_seconds = time.perf_counter() - started

        for layer_index, layer in enumerate(reference_model.model.layers):
            layer.mlp = NativeKernelBitNetMLP(kernel, layer_index)
        kernel.clear_metrics()
        with torch.inference_mode():
            started = time.perf_counter()
            candidate_output = reference_model(
                input_ids=input_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            kernel_seconds = time.perf_counter() - started
        causal_calls = list(kernel.calls)

    reference_logits = reference_output.logits[:, :-1, :]
    candidate_logits = candidate_output.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    if reference_logits.shape[:2] != (sequence_count, predictions_per_sequence):
        raise AssertionError("confirmation prediction shape is inconsistent")
    logit_metrics = _logit_metrics(reference_logits, candidate_logits)
    reference_nll = functional.cross_entropy(
        reference_logits.float().reshape(-1, reference_logits.shape[-1]),
        labels.reshape(-1),
    )
    candidate_nll = functional.cross_entropy(
        candidate_logits.float().reshape(-1, candidate_logits.shape[-1]),
        labels.reshape(-1),
    )
    hidden_metrics = _tensor_metrics(
        reference_output.hidden_states[-1][:, :-1, :],
        candidate_output.hidden_states[-1][:, :-1, :],
    )
    nll_delta = float((candidate_nll - reference_nll).item())
    traffic = native_bitnet_repack_traffic(
        loaded_artifact.hidden_size,
        loaded_artifact.intermediate_size,
        layer_count=len(loaded_artifact.layers),
        cache_line_bytes=loaded_artifact.cache_line_bytes,
    )
    scheduled_layer_bytes = sum(
        int(call["scheduled_cache_line_bytes"]) for call in causal_calls
    )
    expected_layer_bytes = sum(loaded_artifact.layer_block_bytes)
    if len(causal_calls) != len(loaded_artifact.layers):
        raise AssertionError("native kernel did not execute exactly once per layer")
    if scheduled_layer_bytes != expected_layer_bytes:
        raise AssertionError("native kernel byte schedule differs from artifact")
    complete_cold_bytes = loaded_artifact.serialized_artifact_bytes
    dense_q4_bytes = int(traffic["dense_q4_source_mlp_bytes"])
    cold_fraction = complete_cold_bytes / dense_q4_bytes
    thresholds = {
        "maximum_teacher_student_kl": 0.05,
        "minimum_teacher_top1_agreement": 0.9,
        "maximum_nll_delta": 0.05,
        "maximum_final_hidden_relative_l2": 0.1,
        "minimum_unique_sequences": 8,
        "minimum_prediction_positions": 256,
        "maximum_physical_cold_mlp_traffic_fraction": 0.45,
    }
    checks = {
        "teacher_student_kl": logit_metrics["mean_kl_divergence"]
        <= thresholds["maximum_teacher_student_kl"],
        "teacher_top1_agreement": logit_metrics["top1_agreement"]
        >= thresholds["minimum_teacher_top1_agreement"],
        "nll_delta": nll_delta <= thresholds["maximum_nll_delta"],
        "final_hidden_relative_l2": hidden_metrics["relative_l2"]
        <= thresholds["maximum_final_hidden_relative_l2"],
        "unique_sequences": dataset_evidence["unique_sequences"]
        >= thresholds["minimum_unique_sequences"],
        "prediction_positions": prediction_positions
        >= thresholds["minimum_prediction_positions"],
        "physical_cold_mlp_traffic": cold_fraction
        <= thresholds["maximum_physical_cold_mlp_traffic_fraction"],
        "direct_packed_execution": True,
        "artifact_integrity": loaded_artifact.payload_sha256 == artifact_sha256.lower(),
        "source_integrity": weight_sha_before == OFFICIAL_NATIVE_BITNET_WEIGHT_SHA256
        and weight_sha_after == weight_sha_before,
    }
    gate_passed = all(checks.values())
    report = {
        "schema_version": 1,
        "decision": (
            "native_bitnet_milestone_2_semantic_gate_pass"
            if gate_passed
            else "native_bitnet_confirmation_gate_fail"
        ),
        "source_track": "low_bit_native_not_dense_llama_conversion",
        "source_model": str(model),
        "repository": repo_id,
        "resolved_revision": resolved_revision,
        "source_weight_integrity": {
            "expected_sha256": OFFICIAL_NATIVE_BITNET_WEIGHT_SHA256,
            "sha256_before_load": weight_sha_before,
            "sha256_after_load": weight_sha_after,
            "verified": checks["source_integrity"],
            "materialization": materialization,
        },
        "artifact": {
            "path": str(artifact_path),
            "encoding": "native_bitnet_phase_base3_v1",
            "sha256": loaded_artifact.payload_sha256,
            "serialized_bytes": complete_cold_bytes,
            "integrity_verified": checks["artifact_integrity"],
        },
        "dataset": {
            **dataset_evidence,
            "role": "frozen_confirmation",
            "tokenizer_policy": "pinned_model_tokenizer_with_mistral_regex_fix",
            "sequences": sequence_count,
            "tokens_per_sequence": tokens_per_sequence,
            "prediction_positions_per_sequence": predictions_per_sequence,
            "prediction_positions": prediction_positions,
        },
        "local_kernel_parity": {
            "layers": [int(value) for value in parity_layers],
            "states": parity_states,
            "metrics": parity,
            "calls": parity_calls,
            "exact": all(value["exact"] for value in parity.values()),
        },
        "causal_confirmation": {
            "teacher_student_kl": logit_metrics["mean_kl_divergence"],
            "teacher_top1_agreement": logit_metrics["top1_agreement"],
            "nll_delta": nll_delta,
            "teacher_nll": float(reference_nll.item()),
            "student_nll": float(candidate_nll.item()),
            "final_hidden_relative_l2": hidden_metrics["relative_l2"],
            "maximum_logit_absolute_error": logit_metrics["maximum_absolute_error"],
            "maximum_hidden_absolute_error": hidden_metrics["maximum_absolute_error"],
        },
        "kernel": {
            "library": str(Path(library).resolve())
            if library is not None
            else str(_default_library_path()),
            "threads": threads,
            "direct_packed_execution": True,
            "dense_weight_materialization_bytes": 0,
            "reference_seconds": reference_seconds,
            "kernel_substituted_seconds": kernel_seconds,
            "calls": causal_calls,
            "maximum_scratch_bytes": max(
                int(call["scratch_bytes"]) for call in causal_calls
            ),
        },
        "traffic": {
            **traffic,
            "kernel_layer_scheduled_bytes": scheduled_layer_bytes,
            "global_header_and_directory_bytes": complete_cold_bytes
            - expected_layer_bytes,
            "complete_cold_bytes": complete_cold_bytes,
            "complete_cold_fraction_of_dense_q4": cold_fraction,
            "kernel_schedule_matches_artifact": True,
            "hardware_dram_counter_measured": False,
        },
        "thresholds": thresholds,
        "checks": checks,
        "gate_passed": gate_passed,
        "execution": {
            "device": "cpu",
            "dtype": "bfloat16",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "reference_model_load_seconds": load_seconds,
            "gpu_used": False,
        },
        "scope_caveat": (
            "This result qualifies only the pinned low-bit-native BitNet source "
            "track. It is not a dense-Llama conversion result. Cold traffic is "
            "the exact cache-line schedule of the executed phase streams; no "
            "hardware DRAM performance counter was available."
        ),
    }
    atomic_json(Path(out), report)
    return report


__all__ = [
    "NativeBitNetCPUKernel",
    "NativeBitNetKernelError",
    "evaluate_native_bitnet_kernel_confirmation",
]
