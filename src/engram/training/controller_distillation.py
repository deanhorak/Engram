"""CUDA-assisted distillation for the shared recurrent controller.

CUDA is used only by the optimizer.  Teacher trajectories are durable,
checksummed traces, and the resulting controller is serialized as plain FP32
NumPy tensors consumed by :class:`FactorizedRecurrentController` on CPU.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.controller import FactorizedRecurrentController
from engram.models.inspection import inspect_model, resolve_model_path
from engram.runtime.native_bitnet import NativeBitNetRuntime
from engram.tracing.format import TraceReader, TraceWriter
from engram.utils import atomic_json, sha256_file

CONTROLLER_TRACE_CONTRACT = "engram.controller.teacher_trajectory"
CONTROLLER_TRACE_CONTRACT_VERSION = 1
CONTROLLER_INPUT_ORDER = ("token_embedding", "semantic_output", "episodic_output")
CONTROLLER_SUBSTITUTION_NMSE_GATE = 0.0225


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL record {line_number} must be an object")
            records.append(item)
    if not records:
        raise ValueError("controller trace dataset contains no records")
    return records


def _hidden_tensor(output):
    if isinstance(output, tuple):
        output = output[0]
    if not hasattr(output, "detach") or output.ndim != 3:
        raise RuntimeError("teacher hook did not receive a hidden-state tensor")
    return output.detach()


def _captured_sample_ids(path: Path) -> set[int]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return set()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    captured: set[int] = set()
    for shard in manifest.get("shards", []):
        field = shard.get("fields", {}).get("sample_id")
        if not isinstance(field, dict):
            raise ValueError("existing controller trace shard has no sample_id")
        values = np.load(
            path / shard["name"] / field["file"],
            mmap_mode="r",
            allow_pickle=False,
        )
        captured.update(int(value) for value in np.unique(values))
    return captured


def _controller_trace_arrays(
    torch,
    *,
    input_ids,
    attention_mask,
    sample_ids: list[int],
    layer_inputs: dict[int, Any],
    layer_outputs: dict[int, Any],
    attention_outputs: dict[int, Any],
    mlp_outputs: dict[int, Any],
    layer_count: int,
    hidden_size: int,
    causal_top_ids=None,
    causal_top_logits=None,
    causal_target_ids=None,
) -> dict[str, np.ndarray]:
    """Normalize and flatten one padded teacher batch into valid positions."""

    states = [layer_inputs[0]]
    states.extend(layer_outputs[index] for index in range(layer_count))
    teacher_states = torch.stack(states, dim=2)
    semantic = torch.stack([mlp_outputs[index] for index in range(layer_count)], dim=2)
    episodic = torch.stack(
        [attention_outputs[index] for index in range(layer_count)], dim=2
    )
    expected = (
        input_ids.shape[0],
        input_ids.shape[1],
        layer_count + 1,
        hidden_size,
    )
    if teacher_states.shape != expected:
        raise RuntimeError("teacher state trace has wrong dimensions")
    state_rms = (
        teacher_states.float()
        .square()
        .mean(dim=-1, keepdim=True)
        .sqrt()
        .clamp_min(1e-6)
    )
    normalized_states = teacher_states.float() / state_rms
    normalized_semantic = semantic.float() / state_rms[:, :, :-1]
    normalized_episodic = episodic.float() / state_rms[:, :, :-1]
    for label, value in (
        ("teacher states", normalized_states),
        ("semantic outputs", normalized_semantic),
        ("episodic outputs", normalized_episodic),
    ):
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"normalized {label} contain non-finite values")
        if float(value.abs().max()) > np.finfo(np.float16).max:
            raise RuntimeError(f"normalized {label} exceed FP16 trace range")

    valid = attention_mask.to(dtype=torch.bool)
    positions = torch.arange(input_ids.shape[1], dtype=torch.int64)[None, :].expand(
        input_ids.shape[0], -1
    )
    sample_matrix = torch.as_tensor(sample_ids, dtype=torch.int64)[:, None].expand(
        -1, input_ids.shape[1]
    )
    result = {
        "sample_id": sample_matrix[valid].numpy(),
        "token_id": input_ids[valid].numpy(),
        "token_position": positions[valid].numpy(),
        "token_embedding": normalized_states[:, :, 0, :][valid]
        .numpy()
        .astype(np.float16),
        "teacher_states": normalized_states[valid].numpy().astype(np.float16),
        "semantic_outputs": normalized_semantic[valid].numpy().astype(np.float16),
        "episodic_outputs": normalized_episodic[valid].numpy().astype(np.float16),
    }
    if causal_top_ids is not None or causal_top_logits is not None:
        if causal_top_ids is None or causal_top_logits is None or causal_target_ids is None:
            raise ValueError(
                "causal top-k ids, logits, and target ids must be supplied together"
            )
        if (
            causal_top_ids.ndim != 3
            or causal_top_logits.shape != causal_top_ids.shape
            or causal_target_ids.shape != input_ids.shape
            or causal_top_ids.shape[:2] != input_ids.shape
        ):
            raise ValueError("causal top-k arrays do not match token boundaries")
        result.update(
            {
                "causal_top_ids": causal_top_ids[valid]
                .numpy()
                .astype(np.int32),
                "causal_top_logits": causal_top_logits[valid]
                .numpy()
                .astype(np.float16),
                "causal_target_ids": causal_target_ids[valid]
                .numpy()
                .astype(np.int64),
            }
        )
    return result


def capture_native_bitnet_controller_traces(
    package: str | Path,
    dataset: str | Path,
    out: str | Path,
    *,
    split: str,
    samples: int = 8,
    max_tokens: int = 64,
    causal_top_k: int = 0,
    batch_size: int = 1,
    record_offset: int = 0,
    seed: int = 31,
    library: str | Path | None = None,
    threads: int | None = None,
    native_projections: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Capture stage-by-stage BitNet teacher trajectories on CPU.

    The teacher is the already qualified packaged BitNet runtime.  Capturing on
    CPU avoids introducing CUDA-only teacher behavior and keeps GPU memory free
    for the subsequent student optimization.
    """

    if samples <= 0 or max_tokens <= 0 or batch_size <= 0:
        raise ValueError("samples, max_tokens, and batch_size must be positive")
    if causal_top_k < 0:
        raise ValueError("causal_top_k must be non-negative")
    if record_offset < 0:
        raise ValueError("record_offset must be non-negative")
    package_path = Path(package).resolve()
    dataset_path = Path(dataset).resolve()
    output_path = Path(out)
    all_records = _load_jsonl(dataset_path)
    selected = list(
        enumerate(
            all_records[record_offset : record_offset + samples],
            start=record_offset,
        )
    )
    if not selected:
        raise ValueError("record_offset is beyond the controller trace dataset")
    captured_sample_ids = _captured_sample_ids(output_path) if resume else set()
    pending = [
        (sample_id, record)
        for sample_id, record in selected
        if sample_id not in captured_sample_ids
    ]
    started = time.perf_counter()
    with NativeBitNetRuntime(
        package_path,
        library=library,
        threads=threads,
        native_projections=native_projections,
    ) as runtime:
        model = runtime.model
        layers = model.model.layers
        manifest_model = runtime.manifest["model"]
        hidden_size = int(manifest_model["hidden_size"])
        layer_count = int(manifest_model["num_hidden_layers"])
        if len(layers) != layer_count:
            raise RuntimeError("packaged teacher layer count changed while loading")

        layer_inputs: dict[int, Any] = {}
        layer_outputs: dict[int, Any] = {}
        attention_outputs: dict[int, Any] = {}
        mlp_outputs: dict[int, Any] = {}
        hooks = []

        def pre_hook(index: int):
            def capture(_module, args, kwargs):
                hidden = args[0] if args else kwargs.get("hidden_states")
                layer_inputs[index] = _hidden_tensor(hidden)

            return capture

        def output_hook(destination: dict[int, Any], index: int):
            def capture(_module, _args, output):
                destination[index] = _hidden_tensor(output)

            return capture

        for layer_index, layer in enumerate(layers):
            hooks.append(
                layer.register_forward_pre_hook(pre_hook(layer_index), with_kwargs=True)
            )
            hooks.append(
                layer.register_forward_hook(output_hook(layer_outputs, layer_index))
            )
            hooks.append(
                layer.self_attn.register_forward_hook(
                    output_hook(attention_outputs, layer_index)
                )
            )
            hooks.append(
                layer.mlp.register_forward_hook(output_hook(mlp_outputs, layer_index))
            )

        try:
            import torch

            with (
                TraceWriter(
                    output_path,
                    model_hash=str(runtime.manifest["source"]["weight_sha256"]),
                    dataset_hash=sha256_file(dataset_path),
                    split=split,
                    seed=seed,
                    metadata={
                        "contract": CONTROLLER_TRACE_CONTRACT,
                        "contract_version": CONTROLLER_TRACE_CONTRACT_VERSION,
                        "source_package": str(package_path),
                        "source_repository": runtime.manifest["source"]["repository"],
                        "source_revision": runtime.manifest["source"]["revision"],
                        "teacher_runtime": "packaged_native_bitnet_cpu",
                        "teacher_device": "cpu",
                        "hidden_size": hidden_size,
                        "num_stages": layer_count,
                        "input_order": list(CONTROLLER_INPUT_ORDER),
                        "boundary_dtype": "float16",
                        "state_normalization": "per_token_rms",
                        "operator_normalization": "divide_by_stage_input_rms",
                        "max_tokens": max_tokens,
                        "batch_size": batch_size,
                        "record_offset": record_offset,
                        "requested_samples": samples,
                        "causal_top_k": causal_top_k,
                        "sequence_boundaries_preserved": True,
                    },
                    resume=resume,
                ) as writer,
                torch.inference_mode(),
            ):
                encoded: list[tuple[int, list[int]]] = []
                for sample_id, record in pending:
                    if "input_ids" in record:
                        raw_ids = record["input_ids"]
                        if not isinstance(raw_ids, list) or not all(
                            isinstance(value, int) for value in raw_ids
                        ):
                            raise ValueError(
                                f"record {sample_id} input_ids must be integers"
                            )
                        token_ids = [int(value) for value in raw_ids]
                    else:
                        token_ids = runtime.encode(str(record.get("text", "")))
                    if len(token_ids) > max_tokens:
                        # Per-record seeding makes crop selection independent
                        # of batching and restart boundaries. Sample zero also
                        # preserves the original single-sequence behavior.
                        record_rng = np.random.default_rng(seed + sample_id)
                        start = int(
                            record_rng.integers(0, len(token_ids) - max_tokens + 1)
                        )
                        token_ids = token_ids[start : start + max_tokens]
                    if not token_ids:
                        raise ValueError(
                            f"record {sample_id} tokenized to an empty sequence"
                        )
                    encoded.append((sample_id, token_ids))

                pad_token = runtime.tokenizer.pad_token_id
                if pad_token is None:
                    pad_token = runtime.tokenizer.eos_token_id
                if isinstance(pad_token, (tuple, list)):
                    pad_token = pad_token[0]
                if pad_token is None:
                    pad_token = 0
                completed_batches = writer.shard_count
                for batch_start in range(0, len(encoded), batch_size):
                    batch = encoded[batch_start : batch_start + batch_size]
                    sample_ids = [sample_id for sample_id, _ in batch]
                    maximum = max(len(token_ids) for _, token_ids in batch)
                    input_ids = torch.full(
                        (len(batch), maximum),
                        int(pad_token),
                        dtype=torch.long,
                    )
                    attention_mask = torch.zeros(
                        (len(batch), maximum), dtype=torch.long
                    )
                    for row, (_, token_ids) in enumerate(batch):
                        length = len(token_ids)
                        input_ids[row, :length] = torch.as_tensor(
                            token_ids, dtype=torch.long
                        )
                        attention_mask[row, :length] = 1
                    layer_inputs.clear()
                    layer_outputs.clear()
                    attention_outputs.clear()
                    mlp_outputs.clear()
                    model_output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    causal_top_ids = None
                    causal_top_logits = None
                    causal_target_ids = None
                    if causal_top_k:
                        logits = (
                            model_output.logits
                            if hasattr(model_output, "logits")
                            else model_output[0]
                        )
                        vocabulary = int(logits.shape[-1])
                        width = min(causal_top_k, vocabulary)
                        causal_top_logits, causal_top_ids = torch.topk(
                            logits.float(), width, dim=-1
                        )
                        causal_target_ids = torch.full_like(input_ids, -1)
                        if input_ids.shape[1] > 1:
                            causal_target_ids[:, :-1] = input_ids[:, 1:]
                    expected = set(range(layer_count))
                    for label, captured in (
                        ("layer inputs", layer_inputs),
                        ("layer outputs", layer_outputs),
                        ("attention outputs", attention_outputs),
                        ("MLP outputs", mlp_outputs),
                    ):
                        if set(captured) != expected:
                            raise RuntimeError(f"teacher did not capture all {label}")
                    writer.append(
                        _controller_trace_arrays(
                            torch,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            sample_ids=sample_ids,
                            layer_inputs=layer_inputs,
                            layer_outputs=layer_outputs,
                            attention_outputs=attention_outputs,
                            mlp_outputs=mlp_outputs,
                            layer_count=layer_count,
                            hidden_size=hidden_size,
                            causal_top_ids=causal_top_ids,
                            causal_top_logits=causal_top_logits,
                            causal_target_ids=causal_target_ids,
                        )
                    )
                    completed_batches += 1
                    captured_sample_ids.update(sample_ids)
                    atomic_json(
                        output_path / "capture_progress.json",
                        {
                            "complete": False,
                            "completed_batches": completed_batches,
                            "completed_sequences": len(captured_sample_ids),
                            "requested_sequences": len(selected),
                            "last_sample_ids": sample_ids,
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                    )
        finally:
            for hook in hooks:
                hook.remove()
    manifest = json.loads((output_path / "manifest.json").read_text(encoding="utf-8"))
    result = {
        "trace": str(output_path.resolve()),
        "split": split,
        "sequences": len(selected),
        "batches": len(manifest["shards"]),
        "batch_size": batch_size,
        "record_offset": record_offset,
        "causal_top_k": causal_top_k,
        "token_positions": sum(int(shard["records"]) for shard in manifest["shards"]),
        "hidden_size": hidden_size,
        "num_stages": layer_count,
        "elapsed_seconds": time.perf_counter() - started,
        "teacher_device": "cpu",
        "optimizer_device": None,
    }
    atomic_json(output_path / "capture_report.json", result)
    atomic_json(
        output_path / "capture_progress.json",
        {
            "complete": True,
            "completed_batches": result["batches"],
            "completed_sequences": result["sequences"],
            "requested_sequences": result["sequences"],
            "elapsed_seconds": result["elapsed_seconds"],
        },
    )
    return result


def capture_hf_controller_traces(
    model: str | Path,
    dataset: str | Path,
    out: str | Path,
    *,
    split: str,
    samples: int = 8,
    max_tokens: int = 64,
    causal_top_k: int = 0,
    batch_size: int = 1,
    record_offset: int = 0,
    seed: int = 31,
    resume: bool = False,
) -> dict[str, Any]:
    """Capture the controller trajectory contract from a dense HF teacher.

    This is the source-family comparison path for Qwen3 and other dense
    decoder checkpoints accepted by :func:`inspect_model`.  It deliberately
    runs the untouched teacher on CPU and writes the same normalized fields as
    the qualified native-BitNet capture.  It does not make the dense source
    compilable, and it does not alter the protected BitNet trace protocol.
    """

    if samples <= 0 or max_tokens <= 0 or batch_size <= 0:
        raise ValueError("samples, max_tokens, and batch_size must be positive")
    if causal_top_k < 0:
        raise ValueError("causal_top_k must be non-negative")
    if record_offset < 0:
        raise ValueError("record_offset must be non-negative")
    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    dataset_path = Path(dataset).resolve()
    all_records = _load_jsonl(dataset_path)
    selected = list(
        enumerate(
            all_records[record_offset : record_offset + samples],
            start=record_offset,
        )
    )
    if not selected:
        raise ValueError("record_offset is beyond the controller trace dataset")
    output_path = Path(out)
    captured_sample_ids = _captured_sample_ids(output_path) if resume else set()
    pending = [
        (sample_id, record)
        for sample_id, record in selected
        if sample_id not in captured_sample_ids
    ]
    started = time.perf_counter()

    try:
        import torch
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as transformers_imports

        if transformers_imports.is_sklearn_available():
            try:
                import sklearn  # noqa: F401
            except ImportError:

                def sklearn_unavailable() -> bool:
                    return False

                transformers_imports.is_sklearn_available = sklearn_unavailable
                transformers_utils.is_sklearn_available = sklearn_unavailable
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] to capture dense Hugging Face controller traces"
        ) from exc

    tokenizer = None
    if any("input_ids" not in record for _, record in pending):
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
        device_map=None,
    )
    teacher.eval()
    layers = getattr(getattr(teacher, "model", None), "layers", None)
    if layers is None or len(layers) != inspection.num_hidden_layers:
        raise RuntimeError(
            "dense teacher does not expose model.layers with the inspected layer count"
        )
    hidden_size = inspection.hidden_size
    layer_count = inspection.num_hidden_layers
    layer_inputs: dict[int, Any] = {}
    layer_outputs: dict[int, Any] = {}
    attention_outputs: dict[int, Any] = {}
    mlp_outputs: dict[int, Any] = {}
    hooks = []

    def pre_hook(index: int):
        def capture(_module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            layer_inputs[index] = _hidden_tensor(hidden)

        return capture

    def output_hook(destination: dict[int, Any], index: int):
        def capture(_module, _args, output):
            destination[index] = _hidden_tensor(output)

        return capture

    for layer_index, layer in enumerate(layers):
        hooks.append(
            layer.register_forward_pre_hook(pre_hook(layer_index), with_kwargs=True)
        )
        hooks.append(layer.register_forward_hook(output_hook(layer_outputs, layer_index)))
        hooks.append(
            layer.self_attn.register_forward_hook(
                output_hook(attention_outputs, layer_index)
            )
        )
        hooks.append(
            layer.mlp.register_forward_hook(output_hook(mlp_outputs, layer_index))
        )

    try:
        with (
            TraceWriter(
                output_path,
                model_hash=inspection.source_hash,
                dataset_hash=sha256_file(dataset_path),
                split=split,
                seed=seed,
                metadata={
                    "contract": CONTROLLER_TRACE_CONTRACT,
                    "contract_version": CONTROLLER_TRACE_CONTRACT_VERSION,
                    "source_model": str(model_path),
                    "source_model_type": inspection.model_type,
                    "source_architecture": inspection.architecture,
                    "teacher_runtime": "transformers_dense_cpu",
                    "teacher_device": "cpu",
                    "hidden_size": hidden_size,
                    "num_stages": layer_count,
                    "input_order": list(CONTROLLER_INPUT_ORDER),
                    "boundary_dtype": "float16",
                    "state_normalization": "per_token_rms",
                    "operator_normalization": "divide_by_stage_input_rms",
                    "max_tokens": max_tokens,
                    "batch_size": batch_size,
                    "record_offset": record_offset,
                    "requested_samples": samples,
                    "causal_top_k": causal_top_k,
                    "sequence_boundaries_preserved": True,
                    "native_package_compilation": False,
                },
                resume=resume,
            ) as writer,
            torch.inference_mode(),
        ):
            encoded: list[tuple[int, list[int]]] = []
            for sample_id, record in pending:
                if "input_ids" in record:
                    raw_ids = record["input_ids"]
                    if not isinstance(raw_ids, list) or not all(
                        isinstance(value, int) for value in raw_ids
                    ):
                        raise ValueError(f"record {sample_id} input_ids must be integers")
                    token_ids = [int(value) for value in raw_ids]
                else:
                    assert tokenizer is not None
                    token_ids = tokenizer(
                        str(record.get("text", "")), add_special_tokens=True
                    )["input_ids"]
                if len(token_ids) > max_tokens:
                    record_rng = np.random.default_rng(seed + sample_id)
                    start = int(
                        record_rng.integers(0, len(token_ids) - max_tokens + 1)
                    )
                    token_ids = token_ids[start : start + max_tokens]
                if not token_ids:
                    raise ValueError(f"record {sample_id} tokenized to an empty sequence")
                encoded.append((sample_id, token_ids))

            pad_token = getattr(tokenizer, "pad_token_id", None) if tokenizer else None
            if pad_token is None:
                pad_token = getattr(tokenizer, "eos_token_id", None) if tokenizer else 0
            if isinstance(pad_token, (tuple, list)):
                pad_token = pad_token[0]
            if pad_token is None:
                pad_token = 0
            completed_batches = writer.shard_count
            for batch_start in range(0, len(encoded), batch_size):
                batch = encoded[batch_start : batch_start + batch_size]
                sample_ids = [sample_id for sample_id, _ in batch]
                maximum = max(len(token_ids) for _, token_ids in batch)
                input_ids = torch.full(
                    (len(batch), maximum), int(pad_token), dtype=torch.long
                )
                attention_mask = torch.zeros(
                    (len(batch), maximum), dtype=torch.long
                )
                for row, (_, token_ids) in enumerate(batch):
                    length = len(token_ids)
                    input_ids[row, :length] = torch.as_tensor(token_ids, dtype=torch.long)
                    attention_mask[row, :length] = 1
                layer_inputs.clear()
                layer_outputs.clear()
                attention_outputs.clear()
                mlp_outputs.clear()
                model_output = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                causal_top_ids = None
                causal_top_logits = None
                causal_target_ids = None
                if causal_top_k:
                    logits = (
                        model_output.logits
                        if hasattr(model_output, "logits")
                        else model_output[0]
                    )
                    width = min(causal_top_k, int(logits.shape[-1]))
                    causal_top_logits, causal_top_ids = torch.topk(
                        logits.float(), width, dim=-1
                    )
                    causal_target_ids = torch.full_like(input_ids, -1)
                    if input_ids.shape[1] > 1:
                        causal_target_ids[:, :-1] = input_ids[:, 1:]
                expected = set(range(layer_count))
                for label, captured in (
                    ("layer inputs", layer_inputs),
                    ("layer outputs", layer_outputs),
                    ("attention outputs", attention_outputs),
                    ("MLP outputs", mlp_outputs),
                ):
                    if set(captured) != expected:
                        raise RuntimeError(f"dense teacher did not capture all {label}")
                writer.append(
                    _controller_trace_arrays(
                        torch,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        sample_ids=sample_ids,
                        layer_inputs=layer_inputs,
                        layer_outputs=layer_outputs,
                        attention_outputs=attention_outputs,
                        mlp_outputs=mlp_outputs,
                        layer_count=layer_count,
                        hidden_size=hidden_size,
                        causal_top_ids=causal_top_ids,
                        causal_top_logits=causal_top_logits,
                        causal_target_ids=causal_target_ids,
                    )
                )
                completed_batches += 1
                captured_sample_ids.update(sample_ids)
                atomic_json(
                    output_path / "capture_progress.json",
                    {
                        "complete": False,
                        "completed_batches": completed_batches,
                        "completed_sequences": len(captured_sample_ids),
                        "requested_sequences": len(selected),
                        "last_sample_ids": sample_ids,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
    finally:
        for hook in hooks:
            hook.remove()
        del teacher
    manifest = json.loads((output_path / "manifest.json").read_text(encoding="utf-8"))
    result = {
        "trace": str(output_path.resolve()),
        "split": split,
        "sequences": len(selected),
        "batches": len(manifest["shards"]),
        "batch_size": batch_size,
        "record_offset": record_offset,
        "causal_top_k": causal_top_k,
        "token_positions": sum(int(shard["records"]) for shard in manifest["shards"]),
        "hidden_size": hidden_size,
        "num_stages": layer_count,
        "elapsed_seconds": time.perf_counter() - started,
        "teacher_device": "cpu",
        "optimizer_device": None,
        "source_model_type": inspection.model_type,
        "source_hash": inspection.source_hash,
    }
    atomic_json(output_path / "capture_report.json", result)
    atomic_json(
        output_path / "capture_progress.json",
        {
            "complete": True,
            "completed_batches": result["batches"],
            "completed_sequences": result["sequences"],
            "requested_sequences": result["sequences"],
            "elapsed_seconds": result["elapsed_seconds"],
        },
    )
    return result


def merge_controller_traces(
    traces: Sequence[str | Path], out: str | Path
) -> dict[str, Any]:
    """Merge independently captured, checksummed controller-trace chunks.

    Chunked capture is useful on CPU hosts where loading the native teacher and
    retaining a large batch of trace tensors in one process is unsafe.  The
    merger is intentionally strict: all inputs must share the same model,
    dataset, split, seed, and trace contract, and sample IDs may not overlap.
    It emits a new authenticated trace with the original shard fields and no
    source-model tensors beyond the captured boundary arrays.
    """

    paths = [Path(value).resolve() for value in traces]
    target = Path(out).resolve()
    if len(paths) < 2:
        raise ValueError("at least two controller traces are required to merge")
    if len(set(paths)) != len(paths):
        raise ValueError("controller trace inputs must be distinct")
    if target in paths:
        raise ValueError("merge output must differ from every input trace")
    if target.exists():
        raise FileExistsError(f"merge output already exists: {target}")

    readers = [TraceReader(path) for path in paths]
    first = readers[0].manifest
    contract_fields = (
        "schema_version",
        "engram_version",
        "model_hash",
        "dataset_hash",
        "split",
        "seed",
    )
    for path, reader in zip(paths[1:], readers[1:]):
        for field in contract_fields:
            if reader.manifest.get(field) != first.get(field):
                raise ValueError(
                    f"controller trace {path} differs in contract field {field}"
                )

    def normalized_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(manifest.get("metadata", {}))
        # These fields are expected to differ for offset-based chunks.
        metadata.pop("record_offset", None)
        metadata.pop("requested_samples", None)
        return metadata

    expected_metadata = normalized_metadata(first)
    for path, reader in zip(paths[1:], readers[1:]):
        if normalized_metadata(reader.manifest) != expected_metadata:
            raise ValueError(f"controller trace {path} metadata contract differs")

    all_sample_ids: set[int] = set()
    shard_counts: list[int] = []
    for path, reader in zip(paths, readers):
        for shard in reader.iter_shards(fields=["sample_id"]):
            sample_ids = np.asarray(shard["sample_id"], dtype=np.int64)
            if sample_ids.ndim != 1:
                raise ValueError(f"sample_id field is not one-dimensional in {path}")
            unique = {int(value) for value in np.unique(sample_ids)}
            overlap = unique.intersection(all_sample_ids)
            if overlap:
                raise ValueError(
                    f"controller trace sample IDs overlap: {sorted(overlap)[:8]}"
                )
            all_sample_ids.update(unique)
            shard_counts.append(int(sample_ids.shape[0]))
    if not all_sample_ids:
        raise ValueError("controller traces contain no records")

    metadata = dict(first.get("metadata", {}))
    metadata["record_offset"] = 0
    metadata["requested_samples"] = len(all_sample_ids)
    metadata["merged_trace_chunks"] = len(paths)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = TraceWriter(
        target,
        model_hash=str(first["model_hash"]),
        dataset_hash=str(first["dataset_hash"]),
        split=str(first["split"]),
        seed=int(first["seed"]),
        metadata=metadata,
    )
    # Do not close in a ``finally`` block: ``TraceWriter.close`` marks the
    # manifest complete, and an append failure must leave an explicitly
    # incomplete artifact rather than a falsely authenticated partial merge.
    for reader in readers:
        for shard in reader.iter_shards():
            writer.append(shard)
    writer.close()

    manifest_path = target / "manifest.json"
    report = {
        "experiment": "merge_controller_traces",
        "status": "complete",
        "inputs": [
            {
                "trace": str(path),
                "manifest_sha256": sha256_file(path / "manifest.json"),
            }
            for path in paths
        ],
        "output": str(target),
        "output_manifest_sha256": sha256_file(manifest_path),
        "chunks": len(paths),
        "records": int(sum(shard_counts)),
        "sample_count": len(all_sample_ids),
        "model_hash": first["model_hash"],
        "dataset_hash": first["dataset_hash"],
        "split": first["split"],
        "sample_ids_disjoint": True,
    }
    atomic_json(target / "merge_report.json", report)
    return report


@dataclass(frozen=True)
class _TrajectoryArrays:
    token_embedding: np.ndarray
    teacher_states: np.ndarray
    semantic_outputs: np.ndarray
    episodic_outputs: np.ndarray
    sample_id: np.ndarray
    manifest: dict[str, Any]
    token_id: np.ndarray | None = None
    causal_top_ids: np.ndarray | None = None
    causal_top_logits: np.ndarray | None = None
    causal_target_ids: np.ndarray | None = None

    @property
    def records(self) -> int:
        return int(self.token_embedding.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.token_embedding.shape[1])

    @property
    def num_stages(self) -> int:
        return int(self.semantic_outputs.shape[1])


def _load_trajectories(path: str | Path) -> _TrajectoryArrays:
    reader = TraceReader(path)
    metadata = reader.manifest.get("metadata", {})
    if (
        metadata.get("contract") != CONTROLLER_TRACE_CONTRACT
        or metadata.get("contract_version") != CONTROLLER_TRACE_CONTRACT_VERSION
    ):
        raise ValueError("not a supported controller trajectory trace")
    if (
        metadata.get("state_normalization") != "per_token_rms"
        or metadata.get("operator_normalization") != "divide_by_stage_input_rms"
    ):
        raise ValueError(
            "controller trajectory trace has an unsupported normalization contract"
        )
    fields = [
        "token_embedding",
        "teacher_states",
        "semantic_outputs",
        "episodic_outputs",
        "sample_id",
    ]
    available_fields = [
        set(shard.get("fields", {})) for shard in reader.manifest.get("shards", [])
    ]
    optional_fields = (
        "token_id",
        "causal_top_ids",
        "causal_top_logits",
        "causal_target_ids",
    )
    for field in optional_fields:
        if available_fields and all(field in shard for shard in available_fields):
            fields.append(field)
    shards = list(reader.iter_shards(fields))
    arrays = {
        field: np.concatenate([np.asarray(shard[field]) for shard in shards], axis=0)
        for field in fields
    }
    result = _TrajectoryArrays(
        token_embedding=arrays["token_embedding"],
        teacher_states=arrays["teacher_states"],
        semantic_outputs=arrays["semantic_outputs"],
        episodic_outputs=arrays["episodic_outputs"],
        sample_id=arrays["sample_id"],
        manifest=reader.manifest,
        token_id=arrays.get("token_id"),
        causal_top_ids=arrays.get("causal_top_ids"),
        causal_top_logits=arrays.get("causal_top_logits"),
        causal_target_ids=arrays.get("causal_target_ids"),
    )
    expected_states = (
        result.records,
        result.num_stages + 1,
        result.hidden_size,
    )
    expected_operators = (
        result.records,
        result.num_stages,
        result.hidden_size,
    )
    if result.teacher_states.shape != expected_states:
        raise ValueError("controller trace teacher_states shape is inconsistent")
    if (
        result.semantic_outputs.shape != expected_operators
        or result.episodic_outputs.shape != expected_operators
    ):
        raise ValueError("controller trace operator-output shape is inconsistent")
    causal_fields = (
        result.causal_top_ids,
        result.causal_top_logits,
        result.causal_target_ids,
    )
    if any(value is not None for value in causal_fields):
        if any(value is None for value in causal_fields):
            raise ValueError("causal top-k trace fields must be present together")
        assert result.causal_top_ids is not None
        assert result.causal_top_logits is not None
        assert result.causal_target_ids is not None
        if (
            result.causal_top_ids.ndim != 2
            or result.causal_top_logits.shape != result.causal_top_ids.shape
            or result.causal_target_ids.shape != (result.records,)
            or result.causal_top_ids.shape[0] != result.records
        ):
            raise ValueError("causal top-k trace arrays are inconsistent")
    return result


def _torch_controller_class():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] to distill a controller"
        ) from exc

    class TorchFactorizedController(nn.Module):
        def __init__(self, source: FactorizedRecurrentController) -> None:
            super().__init__()
            for name, value in source.tensors().items():
                self.register_parameter(
                    name,
                    nn.Parameter(torch.from_numpy(value.copy())),
                )
            self.state_dim = source.state_dim
            self.num_stages = source.num_stages
            self.adapter_rank = source.adapter_rank
            self.input_adapter_rank = source.input_adapter_rank
            self.has_operator_residual = source.has_operator_residual

        def step(self, state, supplied, stage: int):
            input_feature = supplied @ self.input_down
            if self.input_adapter_rank:
                input_feature = (
                    input_feature
                    + (supplied @ self.input_adapter_down[stage])
                    @ self.input_adapter_up[stage]
                )
            feature = torch.nn.functional.silu(
                input_feature + state @ self.recurrent_down
            )
            projected = feature @ self.gate_up + self.bias
            gate = torch.sigmoid(projected[..., : self.state_dim])
            candidate = projected[..., self.state_dim :] + self.stage_embeddings[stage]
            if self.adapter_rank:
                candidate = (
                    candidate
                    + (state @ self.adapter_down[stage]) @ self.adapter_up[stage]
                )
            residual = state
            if self.has_operator_residual:
                semantic = supplied[
                    ..., self.state_dim : 2 * self.state_dim
                ]
                episodic = supplied[..., 2 * self.state_dim :]
                residual = (
                    residual
                    + self.operator_residual_scale[stage, 0] * semantic
                    + self.operator_residual_scale[stage, 1] * episodic
                )
            residual = (
                residual
                + self.step_scale[stage] * gate * torch.tanh(candidate)
            )
            rms = residual.square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            return residual / rms

        def export(self) -> FactorizedRecurrentController:
            tensors = {
                name: parameter.detach().float().cpu().numpy()
                for name, parameter in self.named_parameters()
            }
            return FactorizedRecurrentController(**tensors)

    return torch, TorchFactorizedController


def _controller_inputs(torch, data: _TrajectoryArrays, indices, device):
    embedding = torch.as_tensor(
        data.token_embedding[indices], dtype=torch.float32, device=device
    )
    semantic = torch.as_tensor(
        data.semantic_outputs[indices], dtype=torch.float32, device=device
    )
    episodic = torch.as_tensor(
        data.episodic_outputs[indices], dtype=torch.float32, device=device
    )
    embedding = embedding[:, None, :].expand(-1, data.num_stages, -1)
    return torch.cat((embedding, semantic, episodic), dim=-1)


def _causal_topk_loss(
    torch,
    state,
    data: _TrajectoryArrays,
    indices,
    device,
    *,
    lm_head,
    norm_weight,
):
    """Distill teacher top-k logits through a frozen vocabulary readout.

    Only the final controller state is differentiated.  The trace supplies
    teacher top-k logits and target IDs, while ``lm_head`` and the final RMSNorm
    weight are immutable CPU/CUDA tensors loaded from the source package.  This
    keeps the objective causal without constructing decoder layers.
    """

    if (
        data.causal_top_ids is None
        or data.causal_top_logits is None
        or data.causal_target_ids is None
    ):
        raise ValueError("causal top-k targets are missing from the controller trace")
    top_ids = torch.as_tensor(data.causal_top_ids[indices], dtype=torch.long, device=device)
    teacher_logits = torch.as_tensor(
        data.causal_top_logits[indices], dtype=torch.float32, device=device
    )
    targets = torch.as_tensor(
        data.causal_target_ids[indices], dtype=torch.long, device=device
    )
    valid = targets >= 0
    if not bool(valid.any()):
        zero = state.sum() * 0.0
        return zero, {"causal_topk_kl": zero, "causal_target_ce": zero}
    state = state[valid]
    top_ids = top_ids[valid]
    teacher_logits = teacher_logits[valid]
    targets = targets[valid]
    rms = state.square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    normalized = state / rms * norm_weight
    rows = lm_head[top_ids]
    student_logits = (normalized[:, None, :] * rows).sum(dim=-1)
    teacher_probabilities = torch.softmax(teacher_logits, dim=-1)
    student_log_probabilities = torch.log_softmax(student_logits, dim=-1)
    topk_kl = -(
        teacher_probabilities * student_log_probabilities
    ).sum(dim=-1).mean()
    matched = top_ids == targets[:, None]
    matched_rows = matched.any(dim=-1)
    if bool(matched_rows.any()):
        positions = matched[matched_rows].to(dtype=torch.float32).argmax(dim=-1)
        target_ce = -student_log_probabilities[matched_rows, positions].mean()
    else:
        target_ce = topk_kl * 0.0
    return topk_kl + 0.25 * target_ce, {
        "causal_topk_kl": topk_kl,
        "causal_target_ce": target_ce,
    }


def _rollout_loss(
    torch,
    module,
    data: _TrajectoryArrays,
    indices,
    device,
    *,
    teacher_forcing: float,
    causal_lm_head=None,
    causal_norm_weight=None,
    causal_weight: float = 0.0,
):
    targets = torch.as_tensor(
        data.teacher_states[indices], dtype=torch.float32, device=device
    )
    supplied = _controller_inputs(torch, data, indices, device)
    state = targets[:, 0]
    losses = []
    delta_losses = []
    cosine_losses = []
    for stage in range(data.num_stages):
        source = targets[:, stage] if teacher_forcing >= 1.0 else state
        if 0.0 < teacher_forcing < 1.0:
            mask = torch.rand((state.shape[0], 1), device=device) < teacher_forcing
            source = torch.where(mask, targets[:, stage], state)
        predicted = module.step(source, supplied[:, stage], stage)
        target = targets[:, stage + 1]
        target_rms = target.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
        losses.append(((predicted - target) / target_rms).square().mean())
        target_delta = target - targets[:, stage]
        predicted_delta = predicted - source
        delta_rms = (
            target_delta.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
        )
        delta_losses.append(
            ((predicted_delta - target_delta) / delta_rms).square().mean()
        )
        cosine_losses.append(
            1.0
            - torch.nn.functional.cosine_similarity(predicted, target, dim=-1).mean()
        )
        state = predicted
    hidden_loss = torch.stack(losses).mean()
    delta_loss = torch.stack(delta_losses).mean()
    cosine_loss = torch.stack(cosine_losses).mean()
    terminal_rms = (
        targets[:, -1].square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
    )
    terminal_loss = (((state - targets[:, -1]) / terminal_rms) ** 2).mean()
    total = hidden_loss + 0.25 * delta_loss + 0.1 * cosine_loss + terminal_loss
    causal_metrics = {}
    if causal_lm_head is not None:
        causal_loss, causal_metrics = _causal_topk_loss(
            torch,
            state,
            data,
            indices,
            device,
            lm_head=causal_lm_head,
            norm_weight=causal_norm_weight,
        )
        total = total + causal_weight * causal_loss
    return (
        total,
        {
            "hidden_normalized_mse": hidden_loss,
            "delta_normalized_mse": delta_loss,
            "cosine_loss": cosine_loss,
            "terminal_normalized_mse": terminal_loss,
            **causal_metrics,
        },
        state,
    )


def _evaluate(
    torch,
    module,
    data: _TrajectoryArrays,
    device,
    *,
    batch_size: int,
    causal_lm_head=None,
    causal_norm_weight=None,
    causal_weight: float = 0.0,
) -> dict[str, float]:
    module.eval()
    totals: dict[str, float] = {}
    records = 0
    with torch.inference_mode():
        for start in range(0, data.records, batch_size):
            indices = np.arange(start, min(start + batch_size, data.records))
            loss, metrics, _ = _rollout_loss(
                torch,
                module,
                data,
                indices,
                device,
                teacher_forcing=0.0,
                causal_lm_head=causal_lm_head,
                causal_norm_weight=causal_norm_weight,
                causal_weight=causal_weight,
            )
            count = len(indices)
            values = {"loss": loss, **metrics}
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value.item()) * count
            records += count
    module.train()
    return {name: value / records for name, value in totals.items()}


def _validate_cpu_parity(
    torch,
    module,
    controller_path: Path,
    data: _TrajectoryArrays,
) -> dict[str, Any]:
    reloaded = FactorizedRecurrentController.load(controller_path)
    count = min(2, data.records)
    indices = np.arange(count)
    supplied = np.concatenate(
        (
            np.repeat(
                data.token_embedding[indices, None, :],
                data.num_stages,
                axis=1,
            ),
            data.semantic_outputs[indices],
            data.episodic_outputs[indices],
        ),
        axis=-1,
    ).astype(np.float32)
    initial = data.teacher_states[indices, 0].astype(np.float32)
    cpu_result = reloaded.run_staged(initial, supplied)
    module_cpu = module.to("cpu").eval()
    with torch.inference_mode():
        state = torch.from_numpy(initial)
        torch_inputs = torch.from_numpy(supplied)
        for stage in range(data.num_stages):
            state = module_cpu.step(state, torch_inputs[:, stage], stage)
        torch_result = state.numpy()
    absolute = np.abs(cpu_result - torch_result)
    return {
        "passed": bool(np.allclose(cpu_result, torch_result, rtol=2e-5, atol=2e-5)),
        "records": count,
        "max_absolute_error": float(np.max(absolute)),
        "mean_absolute_error": float(np.mean(absolute)),
        "runtime_device": "cpu",
        "torch_required_by_serialized_runtime": False,
    }


def distill_factorized_controller(
    trace: str | Path,
    out: str | Path,
    *,
    validation_trace: str | Path | None = None,
    initial_controller: str | Path | None = None,
    device: str = "cuda",
    rank: int = 128,
    adapter_rank: int = 8,
    input_adapter_rank: int = 0,
    operator_residual: bool = False,
    steps: int = 1000,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-3,
    teacher_forcing_schedule: str = "scheduled",
    causal_lm_head: str | Path | None = None,
    causal_norm_weight: str | Path | None = None,
    causal_weight: float = 0.0,
    seed: int = 37,
) -> dict[str, Any]:
    """Fit a compact controller and prove independent NumPy CPU reload."""

    if steps < 0 or batch_size <= 0:
        raise ValueError("steps must be non-negative and batch_size must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if teacher_forcing_schedule not in {"scheduled", "none"}:
        raise ValueError("teacher_forcing_schedule must be 'scheduled' or 'none'")
    if not math.isfinite(causal_weight) or causal_weight < 0.0:
        raise ValueError("causal_weight must be finite and non-negative")
    if (causal_lm_head is None) != (causal_norm_weight is None):
        raise ValueError(
            "causal_lm_head and causal_norm_weight must be supplied together"
        )
    training = _load_trajectories(trace)
    validation = (
        _load_trajectories(validation_trace)
        if validation_trace is not None
        else training
    )
    protected_validation = validation_trace is not None
    if protected_validation:
        training_path = Path(trace).resolve()
        validation_path = Path(validation_trace).resolve()
        if training_path == validation_path:
            raise ValueError("training and validation traces must be distinct")
        if training.manifest.get("dataset_hash") == validation.manifest.get(
            "dataset_hash"
        ):
            raise ValueError("protected validation requires a different dataset hash")
    if (
        validation.hidden_size != training.hidden_size
        or validation.num_stages != training.num_stages
        or validation.manifest.get("model_hash") != training.manifest.get("model_hash")
    ):
        raise ValueError("training and validation trace contracts differ")

    torch, TorchFactorizedController = _torch_controller_class()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA controller distillation requested but unavailable")
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    causal_head = None
    causal_norm = None
    if causal_lm_head is not None:
        assert causal_norm_weight is not None
        causal_head_array = np.asarray(np.load(causal_lm_head), dtype=np.float32)
        causal_norm_array = np.asarray(np.load(causal_norm_weight), dtype=np.float32)
        if causal_head_array.ndim != 2 or causal_head_array.shape[1] != training.hidden_size:
            raise ValueError("causal lm_head must have shape [vocabulary, hidden_size]")
        if causal_norm_array.shape != (training.hidden_size,):
            raise ValueError("causal norm weight must have shape [hidden_size]")
        for data, name in ((training, "training"), (validation, "validation")):
            if (
                data.causal_top_ids is None
                or data.causal_top_logits is None
                or data.causal_target_ids is None
            ):
                raise ValueError(f"causal top-k targets are missing from {name} trace")
        causal_head = torch.as_tensor(causal_head_array, dtype=torch.float32, device=device)
        causal_norm = torch.as_tensor(causal_norm_array, dtype=torch.float32, device=device)
    np_rng = np.random.default_rng(seed)
    target_deltas = training.teacher_states[:, 1:].astype(
        np.float32
    ) - training.teacher_states[:, :-1].astype(np.float32)
    stage_scales = np.sqrt(np.mean(np.square(target_deltas), axis=(0, 2)))
    stage_scales = np.maximum(stage_scales * 2.0, 1e-3)
    if initial_controller is None:
        initial = FactorizedRecurrentController.initialize(
            input_dim=3 * training.hidden_size,
            state_dim=training.hidden_size,
            num_stages=training.num_stages,
            rank=rank,
            adapter_rank=adapter_rank,
            input_adapter_rank=input_adapter_rank,
            operator_residual=operator_residual,
            seed=seed,
            residual_scale=float(np.median(stage_scales)),
        )
        tensors = initial.tensors()
        tensors["step_scale"] = (
            np.zeros(training.num_stages, dtype=np.float32)
            if operator_residual
            else stage_scales.astype(np.float32)
        )
        initial = FactorizedRecurrentController(**tensors)
    else:
        initial = FactorizedRecurrentController.load(initial_controller)
        expected = {
            "input_dim": 3 * training.hidden_size,
            "state_dim": training.hidden_size,
            "num_stages": training.num_stages,
            "rank": rank,
            "adapter_rank": adapter_rank,
            "input_adapter_rank": input_adapter_rank,
            "operator_residual": operator_residual,
        }
        actual = {
            "input_dim": initial.input_dim,
            "state_dim": initial.state_dim,
            "num_stages": initial.num_stages,
            "rank": initial.rank,
            "adapter_rank": initial.adapter_rank,
            "input_adapter_rank": initial.input_adapter_rank,
            "operator_residual": initial.has_operator_residual,
        }
        if actual != expected:
            raise ValueError(
                "initial controller dimensions do not match the requested run"
            )
    module = TorchFactorizedController(initial).to(device)
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    initial_train_metrics = _evaluate(
        torch,
        module,
        training,
        device,
        batch_size=batch_size,
        causal_lm_head=causal_head,
        causal_norm_weight=causal_norm,
        causal_weight=causal_weight,
    )
    initial_validation_metrics = _evaluate(
        torch,
        module,
        validation,
        device,
        batch_size=batch_size,
        causal_lm_head=causal_head,
        causal_norm_weight=causal_norm,
        causal_weight=causal_weight,
    )
    started = time.perf_counter()
    history: list[dict[str, float]] = []
    module.train()
    for step in range(steps):
        replace = training.records < batch_size
        indices = np_rng.choice(training.records, size=batch_size, replace=replace)
        progress = step / max(steps - 1, 1)
        if teacher_forcing_schedule == "none":
            teacher_forcing = 0.0
        elif progress < 0.4:
            teacher_forcing = 1.0
        elif progress < 0.8:
            teacher_forcing = 1.0 - (progress - 0.4) / 0.4
        else:
            teacher_forcing = 0.0
        optimizer.zero_grad(set_to_none=True)
        loss, metrics, _ = _rollout_loss(
            torch,
            module,
            training,
            indices,
            device,
            teacher_forcing=teacher_forcing,
            causal_lm_head=causal_head,
            causal_norm_weight=causal_norm,
            causal_weight=causal_weight,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"controller loss became non-finite at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
        optimizer.step()
        if step in {0, steps - 1} or (step + 1) % max(steps // 10, 1) == 0:
            history.append(
                {
                    "step": step + 1,
                    "teacher_forcing": teacher_forcing,
                    "loss": float(loss.detach().item()),
                    **{
                        name: float(value.detach().item())
                        for name, value in metrics.items()
                    },
                }
            )

    train_metrics = _evaluate(
        torch,
        module,
        training,
        device,
        batch_size=batch_size,
        causal_lm_head=causal_head,
        causal_norm_weight=causal_norm,
        causal_weight=causal_weight,
    )
    validation_metrics = _evaluate(
        torch,
        module,
        validation,
        device,
        batch_size=batch_size,
        causal_lm_head=causal_head,
        causal_norm_weight=causal_norm,
        causal_weight=causal_weight,
    )
    target = Path(out)
    controller_path = target / "controller"
    exported = module.export()
    exported.save(controller_path)
    parity = _validate_cpu_parity(torch, module, controller_path, validation)
    if not parity["passed"]:
        raise RuntimeError("serialized NumPy controller failed CPU parity")
    validation_improvement = {
        name: initial_validation_metrics[name] - validation_metrics[name]
        for name in validation_metrics
    }
    trajectory_improved = (
        validation_metrics["terminal_normalized_mse"]
        < initial_validation_metrics["terminal_normalized_mse"]
        and validation_metrics["cosine_loss"]
        < initial_validation_metrics["cosine_loss"]
    )
    report = {
        "experiment": "shared_controller_distillation",
        "status": (
            "development_result"
            if protected_validation
            else "smoke_only_unprotected_validation"
        ),
        "source_model_hash": training.manifest["model_hash"],
        "trace": str(Path(trace).resolve()),
        "initial_controller": (
            str(Path(initial_controller).resolve())
            if initial_controller is not None
            else None
        ),
        "validation_trace": (
            str(Path(validation_trace).resolve())
            if validation_trace is not None
            else None
        ),
        "protected_validation": protected_validation,
        "teacher_runtime": training.manifest["metadata"].get("teacher_runtime"),
        "teacher_device": training.manifest["metadata"].get("teacher_device"),
        "optimizer_device": str(device),
        "inference_device": "cpu",
        "torch_required_for_inference": False,
        "causal_objective": {
            "enabled": causal_head is not None,
            "lm_head": (
                str(Path(causal_lm_head).resolve())
                if causal_lm_head is not None
                else None
            ),
            "norm_weight": (
                str(Path(causal_norm_weight).resolve())
                if causal_norm_weight is not None
                else None
            ),
            "weight": causal_weight,
            "target": "teacher_topk_logits_and_next_token",
            "decoder_layers_loaded": False,
        },
        "records": {
            "training": training.records,
            "validation": validation.records,
        },
        "controller": exported.metadata(),
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "teacher_forcing_schedule": teacher_forcing_schedule,
            "causal_weight": causal_weight,
            "seed": seed,
            "history": history,
            "elapsed_seconds": time.perf_counter() - started,
            "initial_metrics": initial_train_metrics,
            "metrics": train_metrics,
        },
        "initial_validation": initial_validation_metrics,
        "validation": validation_metrics,
        "validation_improvement": validation_improvement,
        "trajectory_development_gate_passed": bool(
            protected_validation and trajectory_improved and parity["passed"]
        ),
        "fixed_substitution_gate": {
            "metric": "protected_validation_terminal_normalized_mse",
            "threshold": CONTROLLER_SUBSTITUTION_NMSE_GATE,
            "actual": validation_metrics["terminal_normalized_mse"],
            "passed": bool(
                protected_validation
                and parity["passed"]
                and validation_metrics["terminal_normalized_mse"]
                <= CONTROLLER_SUBSTITUTION_NMSE_GATE
            ),
        },
        "cpu_reload_parity": parity,
        "scope": {
            "controller_inputs": list(CONTROLLER_INPUT_ORDER),
            "teacher_operator_outputs_used": True,
            "compiled_semantic_operator_substitution_tested": False,
            "compiled_episodic_operator_substitution_tested": False,
            "original_transformer_layers_removed_from_controller_runtime": True,
            "end_to_end_generation_qualified": False,
        },
    }
    target.mkdir(parents=True, exist_ok=True)
    atomic_json(target / "training_report.json", report)
    return report


__all__ = [
    "CONTROLLER_INPUT_ORDER",
    "CONTROLLER_SUBSTITUTION_NMSE_GATE",
    "CONTROLLER_TRACE_CONTRACT",
    "capture_native_bitnet_controller_traces",
    "merge_controller_traces",
    "distill_factorized_controller",
]
