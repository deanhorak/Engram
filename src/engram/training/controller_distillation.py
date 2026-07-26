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
from typing import Any

import numpy as np

from engram.controller import FactorizedRecurrentController
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
    return {
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


def capture_native_bitnet_controller_traces(
    package: str | Path,
    dataset: str | Path,
    out: str | Path,
    *,
    split: str,
    samples: int = 8,
    max_tokens: int = 64,
    batch_size: int = 1,
    record_offset: int = 0,
    seed: int = 31,
    library: str | Path | None = None,
    threads: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Capture stage-by-stage BitNet teacher trajectories on CPU.

    The teacher is the already qualified packaged BitNet runtime.  Capturing on
    CPU avoids introducing CUDA-only teacher behavior and keeps GPU memory free
    for the subsequent student optimization.
    """

    if samples <= 0 or max_tokens <= 0 or batch_size <= 0:
        raise ValueError("samples, max_tokens, and batch_size must be positive")
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
    with NativeBitNetRuntime(package_path, library=library, threads=threads) as runtime:
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
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
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


@dataclass(frozen=True)
class _TrajectoryArrays:
    token_embedding: np.ndarray
    teacher_states: np.ndarray
    semantic_outputs: np.ndarray
    episodic_outputs: np.ndarray
    sample_id: np.ndarray
    manifest: dict[str, Any]

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


def _rollout_loss(
    torch,
    module,
    data: _TrajectoryArrays,
    indices,
    device,
    *,
    teacher_forcing: float,
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
    return (
        total,
        {
            "hidden_normalized_mse": hidden_loss,
            "delta_normalized_mse": delta_loss,
            "cosine_loss": cosine_loss,
            "terminal_normalized_mse": terminal_loss,
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
    seed: int = 37,
) -> dict[str, Any]:
    """Fit a compact controller and prove independent NumPy CPU reload."""

    if steps < 0 or batch_size <= 0:
        raise ValueError("steps must be non-negative and batch_size must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if teacher_forcing_schedule not in {"scheduled", "none"}:
        raise ValueError("teacher_forcing_schedule must be 'scheduled' or 'none'")
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
        torch, module, training, device, batch_size=batch_size
    )
    initial_validation_metrics = _evaluate(
        torch, module, validation, device, batch_size=batch_size
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

    train_metrics = _evaluate(torch, module, training, device, batch_size=batch_size)
    validation_metrics = _evaluate(
        torch, module, validation, device, batch_size=batch_size
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
    "distill_factorized_controller",
]
