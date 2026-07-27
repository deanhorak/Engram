"""OLMoE router/expert trace contract and deterministic fixture capture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from engram.models.inspection import load_local_named_tensors
from engram.models.olmoe import audit_olmoe_source
from engram.semantic.olmoe import olmoe_sparse_mlp
from engram.tracing.format import TraceWriter
from engram.utils import sha256_file, sha256_json


def _prepare_transformers_imports() -> None:
    """Disable an advertised but ABI-broken optional sklearn installation."""

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


def capture_olmoe_router_batch(
    model: object,
    input_ids: object,
    *,
    layers: Sequence[int],
) -> dict[str, np.ndarray]:
    """Run one HF OLMoE batch and capture unmodified router/MLP boundaries."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("install engram-lm[conversion] for OLMoE tracing") from exc
    decoder_layers = model.model.layers  # type: ignore[attr-defined]
    selected = tuple(dict.fromkeys(int(layer) for layer in layers))
    if not selected or any(
        layer < 0 or layer >= len(decoder_layers) for layer in selected
    ):
        raise ValueError("layers contain an invalid layer index")
    captured: dict[str, torch.Tensor] = {}
    hooks = []

    def mlp_input_hook(layer: int):
        def hook(module, args, kwargs):
            del module
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                raise RuntimeError("OLMoE MLP hook could not find hidden states")
            captured[f"layer_{layer}_mlp_input"] = hidden.detach().cpu()

        return hook

    def router_hook(layer: int):
        def hook(module, args, output):
            del module, args
            if not isinstance(output, tuple) or len(output) != 3:
                raise RuntimeError("unexpected OLMoE router output contract")
            probabilities, weights, indices = output
            captured[f"layer_{layer}_router_probabilities"] = (
                probabilities.detach().cpu()
            )
            captured[f"layer_{layer}_expert_weights"] = weights.detach().cpu()
            captured[f"layer_{layer}_expert_indices"] = indices.detach().cpu()

        return hook

    def mlp_output_hook(layer: int):
        def hook(module, args, output):
            del module, args
            captured[f"layer_{layer}_mlp_output"] = output.detach().cpu()

        return hook

    for layer in selected:
        mlp = decoder_layers[layer].mlp
        hooks.append(
            mlp.register_forward_pre_hook(mlp_input_hook(layer), with_kwargs=True)
        )
        hooks.append(mlp.gate.register_forward_hook(router_hook(layer)))
        hooks.append(mlp.register_forward_hook(mlp_output_hook(layer)))
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    missing = {
        f"layer_{layer}_{field}"
        for layer in selected
        for field in (
            "mlp_input",
            "router_probabilities",
            "expert_weights",
            "expert_indices",
            "mlp_output",
        )
    } - set(captured)
    if missing:
        raise RuntimeError(f"OLMoE trace hooks did not capture {sorted(missing)}")
    return {
        name: tensor.float().numpy()
        if name.endswith(("mlp_input", "mlp_output", "probabilities", "weights"))
        else tensor.to(torch.int64).numpy()
        for name, tensor in captured.items()
    }


def capture_olmoe_router_traces(
    model: str | Path,
    dataset: str | Path,
    out: str | Path,
    *,
    samples: int = 8,
    layers: Sequence[int] | None = None,
    tokens_per_sequence: int | None = None,
    seed: int = 37,
) -> None:
    """Capture trained local OLMoE routing without altering expert selection."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    if tokens_per_sequence is not None and tokens_per_sequence <= 0:
        raise ValueError("tokens_per_sequence must be positive")
    model_path = Path(model).expanduser().resolve()
    audit = audit_olmoe_source(model_path)
    if audit.decision != "proceed_to_router_trace":
        raise ValueError("local OLMoE checkpoint failed exact source validation")
    dataset_path = Path(dataset).expanduser().resolve()
    records = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at line {line_number}: {exc}"
                ) from exc
            records.append(item)
            if len(records) == samples:
                break
    if not records:
        raise ValueError("dataset contains no JSONL records")
    try:
        import torch
        _prepare_transformers_imports()
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install engram-lm[conversion] for OLMoE tracing") from exc
    torch.manual_seed(seed)
    loaded = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    loaded.eval()
    layer_count = len(loaded.model.layers)
    selected = (
        tuple(range(layer_count))
        if layers is None
        else tuple(dict.fromkeys(int(layer) for layer in layers))
    )
    tokenizer = None
    if any("input_ids" not in record for record in records):
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model_hash = sha256_json(
        {
            "config_sha256": audit.config_sha256,
            "index_sha256": audit.index_sha256,
            "resolved_revision": audit.resolved_revision,
        }
    )
    rng = np.random.default_rng(seed)
    with TraceWriter(
        out,
        model_hash=model_hash,
        dataset_hash=sha256_file(dataset_path),
        split="calibration",
        seed=seed,
        metadata={
            "fixture_only": False,
            "model_family": "olmoe",
            "capture_boundary": "trained_router_selection_and_mlp",
            "selected_layers": list(selected),
            "tokens_per_sequence": tokens_per_sequence,
            "source_revision": audit.resolved_revision,
        },
    ) as writer:
        for sample_id, record in enumerate(records):
            if "input_ids" in record:
                input_ids = torch.tensor([record["input_ids"]], dtype=torch.long)
            else:
                assert tokenizer is not None
                input_ids = tokenizer(
                    str(record.get("text", "")), return_tensors="pt"
                )["input_ids"]
            captured = capture_olmoe_router_batch(
                loaded, input_ids, layers=selected
            )
            token_count = input_ids.shape[1]
            positions = np.arange(token_count)
            if (
                tokens_per_sequence is not None
                and token_count > tokens_per_sequence
            ):
                positions = np.sort(
                    rng.choice(
                        token_count, size=tokens_per_sequence, replace=False
                    )
                )
            arrays: dict[str, np.ndarray] = {
                "sample_id": np.full(len(positions), sample_id, dtype=np.int64),
                "token_id": input_ids[0, positions].numpy(),
                "token_position": positions.astype(np.int64),
            }
            for name, value in captured.items():
                if name.endswith(("mlp_input", "mlp_output")):
                    arrays[name] = value[0, positions]
                else:
                    arrays[name] = value[positions]
            writer.append(arrays)


def capture_olmoe_fixture_router_traces(
    model: str | Path,
    out: str | Path,
    *,
    samples: int = 16,
    layers: Sequence[int] | None = None,
    seed: int = 23,
) -> None:
    """Capture exact synthetic states through a locally validated OLMoE fixture."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    model_path = Path(model).expanduser().resolve()
    audit = audit_olmoe_source(model_path)
    if audit.decision != "proceed_to_router_trace":
        raise ValueError("OLMoE fixture must pass exact local tensor validation")
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if not config.get("engram_fixture"):
        raise ValueError("this capture path is restricted to deterministic fixtures")
    layer_count = int(config["num_hidden_layers"])
    selected = tuple(range(layer_count)) if layers is None else tuple(dict.fromkeys(layers))
    if not selected or any(layer < 0 or layer >= layer_count for layer in selected):
        raise ValueError("layers contain an invalid layer index")
    experts = int(config["num_experts"])
    top_k = int(config["num_experts_per_tok"])
    width = int(config["hidden_size"])
    rng = np.random.default_rng(seed)
    weights_path = model_path / "weights.npz"
    model_hash = sha256_json(
        {
            "config": sha256_file(model_path / "config.json"),
            "weights": sha256_file(weights_path),
        }
    )
    with TraceWriter(
        out,
        model_hash=model_hash,
        dataset_hash=sha256_json({"synthetic_states": samples, "seed": seed}),
        split="fixture",
        seed=seed,
        metadata={
            "fixture_only": True,
            "model_family": "olmoe",
            "capture_boundary": "router_and_weighted_expert_contributions",
            "selected_layers": list(selected),
            "num_experts": experts,
            "top_k": top_k,
        },
    ) as writer:
        arrays: dict[str, np.ndarray] = {
            "sample_id": np.arange(samples, dtype=np.int64)
        }
        for layer in selected:
            prefix = f"model.layers.{layer}.mlp"
            names = [f"{prefix}.gate.weight"]
            for expert in range(experts):
                expert_prefix = f"{prefix}.experts.{expert}"
                names.extend(
                    [
                        f"{expert_prefix}.gate_proj.weight",
                        f"{expert_prefix}.up_proj.weight",
                        f"{expert_prefix}.down_proj.weight",
                    ]
                )
            tensors = load_local_named_tensors(model_path, names)
            hidden = rng.normal(size=(samples, width)).astype(np.float32)
            gate = np.stack(
                [
                    tensors[f"{prefix}.experts.{expert}.gate_proj.weight"]
                    for expert in range(experts)
                ]
            )
            up = np.stack(
                [
                    tensors[f"{prefix}.experts.{expert}.up_proj.weight"]
                    for expert in range(experts)
                ]
            )
            down = np.stack(
                [
                    tensors[f"{prefix}.experts.{expert}.down_proj.weight"]
                    for expert in range(experts)
                ]
            )
            result = olmoe_sparse_mlp(
                hidden,
                tensors[f"{prefix}.gate.weight"],
                gate,
                up,
                down,
                top_k=top_k,
                norm_topk_prob=bool(config["norm_topk_prob"]),
            )
            field = f"layer_{layer}"
            arrays[f"{field}_mlp_input"] = hidden
            arrays[f"{field}_router_probabilities"] = result.router_probabilities
            arrays[f"{field}_expert_indices"] = result.expert_indices
            arrays[f"{field}_expert_weights"] = result.expert_weights
            arrays[f"{field}_expert_contributions"] = result.expert_contributions
            arrays[f"{field}_mlp_output"] = result.output
        writer.append(arrays)
