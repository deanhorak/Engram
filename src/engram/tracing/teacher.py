from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.models.inspection import inspect_model, load_layer_mlp, resolve_model_path
from engram.semantic.swiglu import swiglu
from engram.tracing.format import TraceWriter
from engram.utils import sha256_file, sha256_json


def _selected_layer_indices(
    layers: Sequence[int] | None, num_hidden_layers: int
) -> tuple[int, ...]:
    if num_hidden_layers <= 0:
        raise ValueError("model must contain at least one hidden layer")
    if layers is None:
        return tuple(range(num_hidden_layers))
    selected = tuple(dict.fromkeys(int(layer) for layer in layers))
    if not selected or any(
        layer < 0 or layer >= num_hidden_layers for layer in selected
    ):
        raise ValueError("layers must contain valid hidden-layer indices")
    return selected


def _fixture_trace(
    model_path: Path,
    out: Path,
    *,
    dataset: Path | None,
    split: str,
    seed: int,
    samples: int,
    layers: Sequence[int] | None,
) -> None:
    inspection = inspect_model(model_path)
    selected_layers = _selected_layer_indices(
        layers, inspection.num_hidden_layers
    )
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
        metadata={
            "fixture_only": True,
            "capture_boundary": "mlp_input_and_output",
            "selected_layers": list(selected_layers),
            "all_layers_captured": (
                len(selected_layers) == inspection.num_hidden_layers
            ),
        },
    ) as writer:
        arrays: dict[str, np.ndarray] = {"input_type": input_types, "sample_id": np.arange(samples)}
        for layer in selected_layers:
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
    include_attention: bool,
    tokens_per_sequence: int | None,
    selected_layers: Sequence[int] | None,
) -> None:
    """Capture exact MLP-boundary traces from a cached Hugging Face checkpoint."""
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
        raise RuntimeError("install engram-lm[conversion] to trace Hugging Face models") from exc

    inspection = inspect_model(model_path)
    layer_indices = _selected_layer_indices(
        selected_layers, inspection.num_hidden_layers
    )
    records = _load_dataset(dataset)[:samples]
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32, device_map=None
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
    for layer_index in layer_indices:
        layer = layers[layer_index]
        hooks.append(
            layer.mlp.register_forward_pre_hook(
                capture_input(captured_in, layer_index), with_kwargs=True
            )
        )
        if include_attention:
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
        rng = np.random.default_rng(seed)
        with TraceWriter(
            out,
            model_hash=inspection.source_hash,
            dataset_hash=sha256_file(dataset),
            split=split,
            seed=seed,
            metadata={
                "fixture_only": False,
                "capture_boundary": "exact_mlp_module",
                "included_boundaries": (
                    ["mlp", "attention"] if include_attention else ["mlp"]
                ),
                "selected_layers": list(layer_indices),
                "all_layers_captured": (
                    len(layer_indices) == inspection.num_hidden_layers
                ),
                "tokens_per_sequence": tokens_per_sequence,
                "token_sampling": (
                    "seeded_without_replacement"
                    if tokens_per_sequence is not None
                    else "all"
                ),
            },
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
                selected = np.arange(token_count)
                if tokens_per_sequence is not None and token_count > tokens_per_sequence:
                    selected = np.sort(
                        rng.choice(token_count, size=tokens_per_sequence, replace=False)
                    )
                selected_tensor = torch.from_numpy(selected)
                arrays: dict[str, np.ndarray] = {
                    "input_type": np.asarray(
                        [record.get("input_type", "unspecified")] * len(selected),
                        dtype="U32",
                    ),
                    "sample_id": np.full(len(selected), sample_offset, dtype=np.int64),
                    "token_id": input_ids[0, selected_tensor].numpy(),
                    "token_position": selected.astype(np.int64),
                }
                for layer_index in layer_indices:
                    arrays[f"layer_{layer_index}_mlp_input"] = (
                        captured_in[layer_index][0, selected_tensor].float().numpy()
                    )
                    arrays[f"layer_{layer_index}_mlp_output"] = (
                        captured_out[layer_index][0, selected_tensor].float().numpy()
                    )
                    if include_attention:
                        arrays[f"layer_{layer_index}_attention_input"] = (
                            captured_attention_in[layer_index][0, selected_tensor]
                            .float()
                            .numpy()
                        )
                        arrays[f"layer_{layer_index}_attention_output"] = (
                            captured_attention_out[layer_index][0, selected_tensor]
                            .float()
                            .numpy()
                        )
                writer.append(arrays)
                sample_offset += 1
    finally:
        for hook in hooks:
            hook.remove()


def plan_teacher_trace_capture(
    model: str | Path,
    dataset: str | Path,
    *,
    samples: int = 32,
    include_attention: bool = True,
    tokens_per_sequence: int | None = None,
    layers: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Plan selected-layer capture without loading the model or writing traces."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    if tokens_per_sequence is not None and tokens_per_sequence <= 0:
        raise ValueError("tokens_per_sequence must be positive when provided")
    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    selected_layers = _selected_layer_indices(
        layers, inspection.num_hidden_layers
    )
    dataset_path = Path(dataset)
    records = _load_dataset(dataset_path)[:samples]
    tokenizer = None
    if any("input_ids" not in record for record in records):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "install engram-lm[conversion] to plan text trace capture"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
    sequence_tokens = []
    captured_tokens = []
    for index, record in enumerate(records):
        if "input_ids" in record:
            values = record["input_ids"]
            if not isinstance(values, list) or not all(
                isinstance(value, int) for value in values
            ):
                raise ValueError(
                    f"record {index} input_ids must be a list of integers"
                )
        else:
            assert tokenizer is not None
            values = tokenizer(
                str(record["text"]), add_special_tokens=True
            )["input_ids"]
        sequence_tokens.append(len(values))
        captured_tokens.append(
            min(len(values), tokens_per_sequence)
            if tokens_per_sequence is not None
            else len(values)
        )
    token_positions = sum(captured_tokens)
    float_bytes = np.dtype(np.float32).itemsize
    boundary_kinds = 2 + (2 if include_attention else 0)
    boundary_bytes = (
        token_positions
        * len(selected_layers)
        * boundary_kinds
        * inspection.hidden_size
        * float_bytes
    )
    token_metadata_bytes = token_positions * (
        np.dtype("U32").itemsize
        + 3 * np.dtype(np.int64).itemsize
    )
    return {
        "schema_version": 1,
        "mode": "dry_run",
        "model": str(model_path),
        "model_hash": inspection.source_hash,
        "dataset": str(dataset_path.resolve()),
        "dataset_hash": sha256_file(dataset_path),
        "requested_samples": samples,
        "captured_sequences": len(records),
        "source_token_positions": sum(sequence_tokens),
        "captured_token_positions": token_positions,
        "tokens_per_sequence": tokens_per_sequence,
        "selected_layers": list(selected_layers),
        "all_layers_captured": (
            len(selected_layers) == inspection.num_hidden_layers
        ),
        "included_boundaries": (
            ["mlp", "attention"] if include_attention else ["mlp"]
        ),
        "boundary_tensor_dtype": "float32",
        "estimated_boundary_tensor_bytes": boundary_bytes,
        "estimated_token_metadata_bytes": token_metadata_bytes,
        "estimated_npy_payload_bytes": (
            boundary_bytes + token_metadata_bytes
        ),
        "estimate_excludes": [
            "NPY headers",
            "manifest JSON",
            "filesystem directory entries",
        ],
        "execution_note": (
            "selected layers limit hooks, retained tensors, and trace writes; "
            "the Hugging Face forward still evaluates every transformer layer"
        ),
    }


def capture_teacher_traces(
    model: str | Path,
    out: str | Path,
    *,
    dataset: str | Path | None = None,
    split: str = "calibration",
    seed: int = 17,
    samples: int = 32,
    include_attention: bool = True,
    tokens_per_sequence: int | None = None,
    layers: Sequence[int] | None = None,
) -> None:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if tokens_per_sequence is not None and tokens_per_sequence <= 0:
        raise ValueError("tokens_per_sequence must be positive when provided")
    model_path = resolve_model_path(model)
    with (model_path / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    dataset_path = Path(dataset) if dataset is not None else None
    if config.get("engram_fixture"):
        _fixture_trace(
            model_path,
            Path(out),
            dataset=dataset_path,
            split=split,
            seed=seed,
            samples=samples,
            layers=layers,
        )
    else:
        if dataset_path is None:
            raise ValueError("--dataset is required for a real model")
        _hf_trace(
            model_path,
            dataset_path,
            Path(out),
            split=split,
            seed=seed,
            samples=samples,
            include_attention=include_attention,
            tokens_per_sequence=tokens_per_sequence,
            selected_layers=layers,
        )
