from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from engram.models.inspection import inspect_model, load_layer_mlp, resolve_model_path
from engram.semantic.swiglu import swiglu
from engram.tracing.format import TraceWriter
from engram.utils import sha256_file, sha256_json


def _fixture_trace(
    model_path: Path,
    out: Path,
    *,
    dataset: Path | None,
    split: str,
    seed: int,
    samples: int,
) -> None:
    inspection = inspect_model(model_path)
    if dataset is not None:
        dataset_hash = sha256_file(dataset)
        records = []
        with dataset.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    records.append(str(item.get("input_type", "unspecified")))
        if not records:
            raise ValueError("dataset contains no JSONL records")
        input_types = np.asarray([records[index % len(records)] for index in range(samples)], dtype="U32")
    else:
        dataset_hash = sha256_json({"fixture_samples": samples, "seed": seed})
        labels = ("prose", "code", "structured", "conversation")
        input_types = np.asarray([labels[index % len(labels)] for index in range(samples)], dtype="U32")
    rng = np.random.default_rng(seed)
    with TraceWriter(
        out,
        model_hash=inspection.source_hash,
        dataset_hash=dataset_hash,
        split=split,
        seed=seed,
        metadata={"fixture_only": True, "capture_boundary": "mlp_input_and_output"},
    ) as writer:
        arrays: dict[str, np.ndarray] = {"input_type": input_types, "sample_id": np.arange(samples)}
        for layer in range(inspection.num_hidden_layers):
            hidden = rng.normal(size=(samples, inspection.hidden_size)).astype(np.float32)
            gate, up, down = load_layer_mlp(model_path, layer)
            arrays[f"layer_{layer}_mlp_input"] = hidden
            arrays[f"layer_{layer}_mlp_output"] = swiglu(hidden, gate, up, down).astype(np.float32)
        writer.append(arrays)


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
    if not records:
        raise ValueError("dataset contains no JSONL records")
    return records


def _hf_trace(
    model_path: Path,
    dataset: Path,
    out: Path,
    *,
    split: str,
    seed: int,
    samples: int,
) -> None:
    """Capture exact MLP-boundary traces from a cached Hugging Face checkpoint."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install engram-lm[conversion] to trace Hugging Face models") from exc

    inspection = inspect_model(model_path)
    records = _load_dataset(dataset)[:samples]
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float32, device_map=None
    )
    model.eval()
    layers = model.model.layers
    if len(layers) != inspection.num_hidden_layers:
        raise RuntimeError("loaded layer count differs from inspected configuration")
    tokenizer = None
    if any("input_ids" not in record for record in records):
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    captured_in: dict[int, torch.Tensor] = {}
    captured_out: dict[int, torch.Tensor] = {}
    captured_attention_in: dict[int, torch.Tensor] = {}
    captured_attention_out: dict[int, torch.Tensor] = {}

    def capture_input(destination, index):
        def hook(module, args, kwargs):
            del module
            if args:
                hidden_states = args[0]
            else:
                hidden_states = kwargs.get("hidden_states")
            if hidden_states is None:
                raise RuntimeError("trace hook could not find the module hidden_states input")
            destination[index] = hidden_states.detach().cpu()

        return hook

    hooks = []
    for layer_index, layer in enumerate(layers):
        hooks.append(
            layer.mlp.register_forward_pre_hook(
                capture_input(captured_in, layer_index), with_kwargs=True
            )
        )
        hooks.append(
            layer.self_attn.register_forward_pre_hook(
                capture_input(captured_attention_in, layer_index), with_kwargs=True
            )
        )
        hooks.append(
            layer.self_attn.register_forward_hook(
                lambda module, args, output, index=layer_index: captured_attention_out.__setitem__(
                    index,
                    (output[0] if isinstance(output, tuple) else output).detach().cpu(),
                )
            )
        )
        hooks.append(
            layer.mlp.register_forward_hook(
                lambda module, args, output, index=layer_index: captured_out.__setitem__(
                    index, output.detach().cpu()
                )
            )
        )
    try:
        with TraceWriter(
            out,
            model_hash=inspection.source_hash,
            dataset_hash=sha256_file(dataset),
            split=split,
            seed=seed,
            metadata={"fixture_only": False, "capture_boundary": "exact_mlp_module"},
        ) as writer, torch.inference_mode():
            sample_offset = 0
            for record in records:
                if "input_ids" in record:
                    input_ids = torch.tensor([record["input_ids"]], dtype=torch.long)
                else:
                    assert tokenizer is not None
                    input_ids = tokenizer(str(record["text"]), return_tensors="pt")["input_ids"]
                captured_in.clear()
                captured_out.clear()
                captured_attention_in.clear()
                captured_attention_out.clear()
                model(input_ids=input_ids, use_cache=False)
                token_count = input_ids.shape[1]
                arrays: dict[str, np.ndarray] = {
                    "input_type": np.asarray([record.get("input_type", "unspecified")] * token_count, dtype="U32"),
                    "sample_id": np.full(token_count, sample_offset, dtype=np.int64),
                    "token_id": input_ids[0].numpy(),
                }
                for layer_index in range(inspection.num_hidden_layers):
                    arrays[f"layer_{layer_index}_mlp_input"] = captured_in[layer_index][0].float().numpy()
                    arrays[f"layer_{layer_index}_mlp_output"] = captured_out[layer_index][0].float().numpy()
                    arrays[f"layer_{layer_index}_attention_input"] = (
                        captured_attention_in[layer_index][0].float().numpy()
                    )
                    arrays[f"layer_{layer_index}_attention_output"] = (
                        captured_attention_out[layer_index][0].float().numpy()
                    )
                writer.append(arrays)
                sample_offset += 1
    finally:
        for hook in hooks:
            hook.remove()


def capture_teacher_traces(
    model: str | Path,
    out: str | Path,
    *,
    dataset: str | Path | None = None,
    split: str = "calibration",
    seed: int = 17,
    samples: int = 32,
) -> None:
    model_path = resolve_model_path(model)
    with (model_path / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    dataset_path = Path(dataset) if dataset is not None else None
    if config.get("engram_fixture"):
        _fixture_trace(model_path, Path(out), dataset=dataset_path, split=split, seed=seed, samples=samples)
    else:
        if dataset_path is None:
            raise ValueError("--dataset is required for a real model")
        _hf_trace(model_path, dataset_path, Path(out), split=split, seed=seed, samples=samples)
