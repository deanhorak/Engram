"""Reproducible real-layer benchmark for the frozen OLMoE expert proxy.

This benchmark deliberately loads one expert layer instead of constructing an
entire Transformers model.  The installed OLMoE expert module is allocated on
the meta device, materialized directly as CPU BF16, and populated one expert
slice at a time from safetensors.  That keeps the real 64-expert fixture near
its 768 MiB final size and avoids allocating either a full model or FP32 expert
parameters before converting them.

The benchmark compares the installed eager expert implementation with
``frozen_olmoe_expert_backward_proxy`` twice at every requested worker count.
Outputs, hidden-state gradients, and routing-weight gradients must be bitwise
identical.  Wall time and process CPU time are measured separately for forward
and backward so expert-level CPU utilization is visible without hiding the
proxy's intentionally serial forward path.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import re
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from engram.evaluation.olmoe_expert_proxy import (
    frozen_olmoe_expert_backward_proxy,
)
from engram.tracing.olmoe import _prepare_transformers_imports
from engram.utils import atomic_json, sha256_file


_EXPERIMENT = "olmoe_frozen_expert_proxy_real_layer_parity"
_SCHEMA_VERSION = 2
_DEFAULT_WORKERS = (1, 4, 8, 12)
_DEFAULT_TOKENS = 128
_DEFAULT_REPEATS = 2
_DEFAULT_SEED = 20260728
_SUPPORTED_SOURCE_DTYPES = ("torch.bfloat16", "torch.float16", "torch.float32")
_SOURCE_KEYS = ("benchmark", "proxy", "transformers_olmoe")
_AUTHENTICATION_CHECK_KEYS = (
    "benchmark_source_unchanged",
    "proxy_source_unchanged",
    "transformers_oracle_unchanged",
    "versions_unchanged",
    "model_path_unchanged",
    "config_unchanged",
    "index_unchanged",
    "selected_shards_unchanged",
    "loaded_gate_up_unchanged",
    "loaded_down_unchanged",
)
_STATS_CHECK_KEYS = (
    "workers",
    "patched_one_layer",
    "serial_forward_calls",
    "parallel_backward_calls",
    "expert_backward_tasks",
    "restored_one_layer",
    "context_inactive_after_exit",
    "executor_shutdown",
    "serial_forward_time_nested",
    "parallel_backward_time_nested",
)
_TIMING_CLASSIFICATION = "host_specific_non_counterbalanced_microbenchmark"


@dataclass(frozen=True)
class LoadedOLMoEExpertLayer:
    """One real, frozen OLMoE expert layer and its bounded source inventory."""

    owner: Any
    experts: Any
    layer_index: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    model_info: Mapping[str, Any]


@dataclass(frozen=True)
class _MeasuredRun:
    output: Any
    hidden_gradient: Any
    routing_weight_gradient: Any
    timing: Mapping[str, Mapping[str, float]]
    frozen_expert_gradients_absent: bool


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for the OLMoE expert benchmark"
        ) from exc
    return torch


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_nonfinite(constant: str) -> Any:
        raise ValueError(f"{label} contains non-finite number {constant}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _safe_shard_path(model_path: Path, shard: Any) -> Path:
    if not isinstance(shard, str) or not shard:
        raise ValueError("safetensors index contains an invalid shard name")
    relative = Path(shard)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != shard:
        raise ValueError("safetensors shard name must be a local file name")
    path = model_path / relative
    if not path.is_file():
        raise ValueError(f"safetensors shard is not a regular file: {path}")
    return path


def _source_tensor(
    torch: Any,
    handle: Any,
    name: str,
    expected_shape: tuple[int, ...],
    *,
    expert_index: int | None = None,
) -> Any:
    try:
        if expert_index is None:
            source = handle.get_tensor(name)
        else:
            source = handle.get_slice(name)[expert_index]
    except Exception as exc:
        raise ValueError(
            f"could not read selected expert tensor {name}: {exc}"
        ) from exc
    if tuple(source.shape) != expected_shape:
        raise ValueError(
            f"selected expert tensor {name} has shape {tuple(source.shape)}, "
            f"expected {expected_shape}"
        )
    if str(source.dtype) not in _SUPPORTED_SOURCE_DTYPES:
        raise ValueError(
            f"selected expert tensor {name} has unsupported dtype {source.dtype}"
        )
    if source.device.type != "cpu" or not bool(torch.isfinite(source).all()):
        raise ValueError(f"selected expert tensor {name} is not finite CPU state")
    return source


def _update_source_digest(
    torch: Any,
    digest: Any,
    *,
    name: str,
    source: Any,
    expert_index: int | None = None,
) -> None:
    identity = json.dumps(
        {
            "dtype": str(source.dtype),
            "expert_index": expert_index,
            "name": name,
            "shape": list(source.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest.update(len(identity).to_bytes(8, "little"))
    digest.update(identity)
    raw = source.detach().contiguous().view(torch.uint8).numpy()
    digest.update(memoryview(raw))


def _legacy_tensor_names(layer: int, expert: int) -> dict[str, str]:
    prefix = f"model.layers.{layer}.mlp.experts.{expert}"
    return {
        "gate": f"{prefix}.gate_proj.weight",
        "up": f"{prefix}.up_proj.weight",
        "down": f"{prefix}.down_proj.weight",
    }


def _packed_tensor_names(layer: int) -> dict[str, str]:
    prefix = f"model.layers.{layer}.mlp.experts"
    return {
        "gate_up": f"{prefix}.gate_up_proj",
        "down": f"{prefix}.down_proj",
    }


def _detect_source_layout(
    weight_map: Mapping[str, Any],
    *,
    layer: int,
    num_experts: int,
) -> tuple[str, tuple[str, ...]]:
    legacy = tuple(
        name
        for expert in range(num_experts)
        for name in _legacy_tensor_names(layer, expert).values()
    )
    packed = tuple(_packed_tensor_names(layer).values())
    has_legacy = all(name in weight_map for name in legacy)
    has_packed = all(name in weight_map for name in packed)
    if has_legacy == has_packed:
        raise ValueError(
            "selected OLMoE layer must have exactly one complete supported "
            "expert tensor layout"
        )
    names = legacy if has_legacy else packed
    for name in names:
        if not isinstance(weight_map[name], str) or not weight_map[name]:
            raise ValueError(f"safetensors index has an invalid mapping for {name}")
    return ("legacy_per_expert" if has_legacy else "packed_experts"), names


def _file_inventory(path: Path, *, name: str | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"authenticated file is not a regular file: {resolved}")
    result = {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if name is not None:
        result["name"] = name
    return result


def _authentication_inventory(
    model: str | Path,
    *,
    layer: int,
) -> dict[str, Any]:
    """Hash every executable oracle and selected model file for one run."""

    model_path = Path(model).resolve()
    if not model_path.is_dir():
        raise ValueError(f"OLMoE model path is not a directory: {model_path}")
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    config_json = _read_json_object(config_path, "OLMoE config")
    index_json = _read_json_object(index_path, "safetensors index")
    layer_index = _nonnegative_int(layer, "layer")
    num_experts = _positive_int(config_json.get("num_experts"), "num_experts")
    layers = _positive_int(config_json.get("num_hidden_layers"), "num_hidden_layers")
    if layer_index >= layers:
        raise ValueError("selected expert layer is outside the model")
    weight_map = index_json.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("safetensors index has an invalid weight_map")
    _source_layout, selected_names = _detect_source_layout(
        weight_map,
        layer=layer_index,
        num_experts=num_experts,
    )
    shard_paths = sorted(
        {_safe_shard_path(model_path, weight_map[name]) for name in selected_names},
        key=lambda path: path.name,
    )

    _prepare_transformers_imports()
    try:
        import safetensors
        import transformers
        from transformers.models.olmoe.modeling_olmoe import OlmoeExperts
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for the OLMoE expert benchmark"
        ) from exc
    oracle_source_value = inspect.getsourcefile(OlmoeExperts)
    if not oracle_source_value:
        raise RuntimeError("could not locate the installed OLMoE oracle source")
    oracle_source = Path(oracle_source_value).resolve()
    benchmark_source = Path(__file__).resolve()
    proxy_source = benchmark_source.with_name("olmoe_expert_proxy.py")
    return {
        "sources": {
            "benchmark": _file_inventory(benchmark_source),
            "proxy": _file_inventory(proxy_source),
            "transformers_olmoe": _file_inventory(oracle_source),
        },
        "versions": {
            "transformers": str(transformers.__version__),
            "safetensors": str(safetensors.__version__),
        },
        "model": {
            "path": str(model_path),
            "config": _file_inventory(config_path),
            "index": _file_inventory(index_path),
            "selected_shards": [
                _file_inventory(path, name=path.name) for path in shard_paths
            ],
        },
    }


def _sha256_tensor_state(torch: Any, tensor: Any, *, name: str) -> str:
    digest = hashlib.sha256()
    identity = json.dumps(
        {
            "dtype": str(tensor.dtype),
            "name": name,
            "shape": list(tensor.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest.update(len(identity).to_bytes(8, "little"))
    digest.update(identity)
    raw = tensor.detach().contiguous().view(torch.uint8).numpy()
    digest.update(memoryview(raw))
    return digest.hexdigest()


def load_frozen_olmoe_expert_layer(
    model: str | Path,
    *,
    layer: int = 0,
) -> LoadedOLMoEExpertLayer:
    """Stream one selected real OLMoE expert layer into CPU BF16 state."""

    torch = _require_torch()
    layer_index = _nonnegative_int(layer, "layer")
    model_path = Path(model).resolve()
    if not model_path.is_dir():
        raise ValueError(f"OLMoE model path is not a directory: {model_path}")
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    config_json = _read_json_object(config_path, "OLMoE config")
    index_json = _read_json_object(index_path, "safetensors index")
    if config_json.get("model_type") != "olmoe":
        raise ValueError("selected model is not an OLMoE checkpoint")
    hidden_size = _positive_int(config_json.get("hidden_size"), "hidden_size")
    intermediate_size = _positive_int(
        config_json.get("intermediate_size"),
        "intermediate_size",
    )
    num_experts = _positive_int(config_json.get("num_experts"), "num_experts")
    top_k = _positive_int(
        config_json.get("num_experts_per_tok"),
        "num_experts_per_tok",
    )
    layers = _positive_int(config_json.get("num_hidden_layers"), "num_hidden_layers")
    if layer_index >= layers:
        raise ValueError("selected expert layer is outside the model")
    if top_k > num_experts:
        raise ValueError("OLMoE top-k exceeds its expert count")
    if config_json.get("hidden_act", "silu") != "silu":
        raise ValueError("expert proxy benchmark requires SwiGLU/SiLU OLMoE")
    weight_map = index_json.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("safetensors index has an invalid weight_map")
    source_layout, selected_names = _detect_source_layout(
        weight_map,
        layer=layer_index,
        num_experts=num_experts,
    )
    selected_shards = {
        name: _safe_shard_path(model_path, weight_map[name]) for name in selected_names
    }

    _prepare_transformers_imports()
    try:
        from safetensors import safe_open
        from transformers import AutoConfig
        from transformers.models.olmoe.modeling_olmoe import OlmoeExperts
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for the OLMoE expert benchmark"
        ) from exc
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if (
        int(config.hidden_size) != hidden_size
        or int(config.intermediate_size) != intermediate_size
        or int(config.num_local_experts) != num_experts
        or int(config.num_experts_per_tok) != top_k
        or int(config.num_hidden_layers) != layers
        or str(config.hidden_act) != "silu"
    ):
        raise ValueError("Transformers config disagrees with audited OLMoE metadata")
    config._experts_implementation = "eager"
    with torch.device("meta"):
        experts = OlmoeExperts(config)
    experts.to(dtype=torch.bfloat16)
    experts.to_empty(device="cpu")
    experts.config = config
    experts.eval()
    experts.requires_grad_(False)

    source_dtype_counts: dict[str, int] = {}
    selected_state_digest = hashlib.sha256()
    streamed_slices = 0
    unique_shards = sorted(set(selected_shards.values()), key=str)
    with ExitStack() as stack:
        handles = {
            shard: stack.enter_context(
                safe_open(str(shard), framework="pt", device="cpu")
            )
            for shard in unique_shards
        }
        with torch.no_grad():
            if source_layout == "legacy_per_expert":
                for expert_index in range(num_experts):
                    names = _legacy_tensor_names(layer_index, expert_index)
                    gate = _source_tensor(
                        torch,
                        handles[selected_shards[names["gate"]]],
                        names["gate"],
                        (intermediate_size, hidden_size),
                    )
                    source_dtype_counts[str(gate.dtype)] = (
                        source_dtype_counts.get(str(gate.dtype), 0) + 1
                    )
                    _update_source_digest(
                        torch,
                        selected_state_digest,
                        name=names["gate"],
                        source=gate,
                    )
                    experts.gate_up_proj[
                        expert_index,
                        :intermediate_size,
                    ].copy_(gate)
                    streamed_slices += 1
                    del gate

                    up = _source_tensor(
                        torch,
                        handles[selected_shards[names["up"]]],
                        names["up"],
                        (intermediate_size, hidden_size),
                    )
                    source_dtype_counts[str(up.dtype)] = (
                        source_dtype_counts.get(str(up.dtype), 0) + 1
                    )
                    _update_source_digest(
                        torch,
                        selected_state_digest,
                        name=names["up"],
                        source=up,
                    )
                    experts.gate_up_proj[
                        expert_index,
                        intermediate_size:,
                    ].copy_(up)
                    streamed_slices += 1
                    del up

                    down = _source_tensor(
                        torch,
                        handles[selected_shards[names["down"]]],
                        names["down"],
                        (hidden_size, intermediate_size),
                    )
                    source_dtype_counts[str(down.dtype)] = (
                        source_dtype_counts.get(str(down.dtype), 0) + 1
                    )
                    _update_source_digest(
                        torch,
                        selected_state_digest,
                        name=names["down"],
                        source=down,
                    )
                    experts.down_proj[expert_index].copy_(down)
                    streamed_slices += 1
                    del down
            else:
                names = _packed_tensor_names(layer_index)
                packed_shapes = {
                    names["gate_up"]: (
                        num_experts,
                        2 * intermediate_size,
                        hidden_size,
                    ),
                    names["down"]: (
                        num_experts,
                        hidden_size,
                        intermediate_size,
                    ),
                }
                for name, expected_shape in packed_shapes.items():
                    try:
                        outer_shape = tuple(
                            handles[selected_shards[name]].get_slice(name).get_shape()
                        )
                    except Exception as exc:
                        raise ValueError(
                            f"could not inspect packed expert tensor {name}: {exc}"
                        ) from exc
                    if outer_shape != expected_shape:
                        raise ValueError(
                            f"packed expert tensor {name} has shape {outer_shape}, "
                            f"expected {expected_shape}"
                        )
                for expert_index in range(num_experts):
                    gate_up = _source_tensor(
                        torch,
                        handles[selected_shards[names["gate_up"]]],
                        names["gate_up"],
                        (2 * intermediate_size, hidden_size),
                        expert_index=expert_index,
                    )
                    source_dtype_counts[str(gate_up.dtype)] = (
                        source_dtype_counts.get(str(gate_up.dtype), 0) + 1
                    )
                    _update_source_digest(
                        torch,
                        selected_state_digest,
                        name=names["gate_up"],
                        source=gate_up,
                        expert_index=expert_index,
                    )
                    experts.gate_up_proj[expert_index].copy_(gate_up)
                    streamed_slices += 1
                    del gate_up

                    down = _source_tensor(
                        torch,
                        handles[selected_shards[names["down"]]],
                        names["down"],
                        (hidden_size, intermediate_size),
                        expert_index=expert_index,
                    )
                    source_dtype_counts[str(down.dtype)] = (
                        source_dtype_counts.get(str(down.dtype), 0) + 1
                    )
                    _update_source_digest(
                        torch,
                        selected_state_digest,
                        name=names["down"],
                        source=down,
                        expert_index=expert_index,
                    )
                    experts.down_proj[expert_index].copy_(down)
                    streamed_slices += 1
                    del down

    expected_slices = num_experts * (3 if source_layout == "legacy_per_expert" else 2)
    if streamed_slices != expected_slices:
        raise AssertionError("selected OLMoE expert state was not loaded completely")
    if experts.gate_up_proj.requires_grad or experts.down_proj.requires_grad:
        raise AssertionError("loaded OLMoE expert state is not frozen")
    if experts.gate_up_proj.dtype != torch.bfloat16:
        raise AssertionError("loaded OLMoE gate/up state is not BF16")
    if experts.down_proj.dtype != torch.bfloat16:
        raise AssertionError("loaded OLMoE down state is not BF16")
    expected_gate_up_shape = (
        num_experts,
        2 * intermediate_size,
        hidden_size,
    )
    expected_down_shape = (
        num_experts,
        hidden_size,
        intermediate_size,
    )
    if tuple(experts.gate_up_proj.shape) != expected_gate_up_shape:
        raise AssertionError("loaded OLMoE gate/up state has the wrong shape")
    if tuple(experts.down_proj.shape) != expected_down_shape:
        raise AssertionError("loaded OLMoE down state has the wrong shape")
    if not bool(torch.isfinite(experts.gate_up_proj).all()) or not bool(
        torch.isfinite(experts.down_proj).all()
    ):
        raise AssertionError("loaded OLMoE expert state is not finite")
    owner = torch.nn.Module()
    owner.add_module("experts", experts)
    owner.eval()
    parameter_bytes = int(
        experts.gate_up_proj.numel() * experts.gate_up_proj.element_size()
        + experts.down_proj.numel() * experts.down_proj.element_size()
    )
    expected_parameter_bytes = num_experts * (3 * hidden_size * intermediate_size) * 2
    if parameter_bytes != expected_parameter_bytes:
        raise AssertionError("loaded OLMoE expert parameter bytes are inconsistent")
    shard_inventory = [
        {
            "name": shard.name,
            "size_bytes": shard.stat().st_size,
        }
        for shard in unique_shards
    ]
    model_info = {
        "path": str(model_path),
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "source_layout": source_layout,
        "selected_tensor_keys": len(selected_names),
        "streamed_expert_slices": streamed_slices,
        "selected_source_state_sha256": selected_state_digest.hexdigest(),
        "source_dtype_counts": dict(sorted(source_dtype_counts.items())),
        "selected_shards": shard_inventory,
        "loaded_dtype": str(experts.gate_up_proj.dtype),
        "loaded_parameter_bytes": parameter_bytes,
        "loaded_shapes": {
            "gate_up_proj": list(experts.gate_up_proj.shape),
            "down_proj": list(experts.down_proj.shape),
        },
        "loaded_state_sha256": {
            "gate_up_proj": _sha256_tensor_state(
                torch,
                experts.gate_up_proj,
                name="gate_up_proj",
            ),
            "down_proj": _sha256_tensor_state(
                torch,
                experts.down_proj,
                name="down_proj",
            ),
        },
    }
    return LoadedOLMoEExpertLayer(
        owner=owner,
        experts=experts,
        layer_index=layer_index,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        model_info=model_info,
    )


def _elapsed(
    wall_start_ns: int,
    cpu_start_ns: int,
    wall_end_ns: int,
    cpu_end_ns: int,
) -> dict[str, float]:
    wall_seconds = (wall_end_ns - wall_start_ns) / 1_000_000_000.0
    cpu_seconds = (cpu_end_ns - cpu_start_ns) / 1_000_000_000.0
    if wall_seconds <= 0.0 or cpu_seconds < 0.0:
        raise RuntimeError("benchmark timing clock did not advance")
    return {
        "wall_seconds": wall_seconds,
        "process_cpu_seconds": cpu_seconds,
        "effective_cores": cpu_seconds / wall_seconds,
    }


def _measure_run(
    torch: Any,
    experts: Any,
    hidden: Any,
    top_k_index: Any,
    top_k_weights: Any,
    probe: Any,
) -> _MeasuredRun:
    hidden_leaf = hidden.detach().clone().requires_grad_(True)
    routing_leaf = top_k_weights.detach().clone().requires_grad_(True)
    total_wall_start = time.perf_counter_ns()
    total_cpu_start = time.process_time_ns()
    forward_wall_start = total_wall_start
    forward_cpu_start = total_cpu_start
    output = experts(hidden_leaf, top_k_index, routing_leaf)
    forward_wall_end = time.perf_counter_ns()
    forward_cpu_end = time.process_time_ns()
    backward_wall_start = forward_wall_end
    backward_cpu_start = forward_cpu_end
    hidden_gradient, routing_gradient = torch.autograd.grad(
        output,
        (hidden_leaf, routing_leaf),
        grad_outputs=probe,
        create_graph=False,
        retain_graph=False,
    )
    backward_wall_end = time.perf_counter_ns()
    backward_cpu_end = time.process_time_ns()
    timing = {
        "forward": _elapsed(
            forward_wall_start,
            forward_cpu_start,
            forward_wall_end,
            forward_cpu_end,
        ),
        "backward": _elapsed(
            backward_wall_start,
            backward_cpu_start,
            backward_wall_end,
            backward_cpu_end,
        ),
        "total": _elapsed(
            total_wall_start,
            total_cpu_start,
            backward_wall_end,
            backward_cpu_end,
        ),
    }
    result = _MeasuredRun(
        output=output.detach().clone(),
        hidden_gradient=hidden_gradient.detach().clone(),
        routing_weight_gradient=routing_gradient.detach().clone(),
        timing=timing,
        frozen_expert_gradients_absent=(
            experts.gate_up_proj.grad is None and experts.down_proj.grad is None
        ),
    )
    del output, hidden_gradient, routing_gradient, hidden_leaf, routing_leaf
    return result


def _tensor_comparison(torch: Any, actual: Any, expected: Any) -> dict[str, Any]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {
            "exact": False,
            "shape_and_dtype_match": False,
            "mismatch_count": max(int(actual.numel()), int(expected.numel())),
            "max_absolute_difference": None,
        }
    difference = (actual.float() - expected.float()).abs()
    maximum = float(difference.max().item()) if difference.numel() else 0.0
    exact = bool(torch.equal(actual, expected))
    return {
        "exact": exact,
        "shape_and_dtype_match": True,
        "mismatch_count": int(torch.count_nonzero(actual != expected).item()),
        "max_absolute_difference": maximum,
    }


def _parity(
    torch: Any,
    actual: _MeasuredRun,
    expected: _MeasuredRun,
) -> dict[str, Any]:
    values = {
        "output": _tensor_comparison(torch, actual.output, expected.output),
        "hidden_gradient": _tensor_comparison(
            torch,
            actual.hidden_gradient,
            expected.hidden_gradient,
        ),
        "routing_weight_gradient": _tensor_comparison(
            torch,
            actual.routing_weight_gradient,
            expected.routing_weight_gradient,
        ),
    }
    return {
        "values": values,
        "exact": all(value["exact"] for value in values.values()),
    }


def _timing_summary(runs: Sequence[_MeasuredRun]) -> dict[str, Any]:
    summary: dict[str, Any] = {"runs": [run.timing for run in runs]}
    for phase in ("forward", "backward", "total"):
        wall = sum(float(run.timing[phase]["wall_seconds"]) for run in runs)
        cpu = sum(float(run.timing[phase]["process_cpu_seconds"]) for run in runs)
        summary[f"mean_{phase}_wall_seconds"] = wall / len(runs)
        summary[f"mean_{phase}_process_cpu_seconds"] = cpu / len(runs)
        summary[f"aggregate_{phase}_effective_cores"] = cpu / wall
    return summary


def _sha256_tensor(torch: Any, tensor: Any) -> str:
    value = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def _workload(
    torch: Any,
    loaded: LoadedOLMoEExpertLayer,
    *,
    tokens: int,
    seed: int,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    token_count = _positive_int(tokens, "tokens")
    seed_value = _nonnegative_int(seed, "seed")
    if loaded.num_experts % loaded.top_k != 0:
        raise ValueError("benchmark requires expert count divisible by top-k")
    route_period = loaded.num_experts // loaded.top_k
    if token_count % route_period != 0:
        raise ValueError(
            "tokens must be a positive multiple of num_experts / top_k "
            "to cover every expert uniformly"
        )
    token_indices = torch.arange(token_count, dtype=torch.int64)[:, None]
    slots = torch.arange(loaded.top_k, dtype=torch.int64)[None, :]
    top_k_index = (token_indices + route_period * slots) % loaded.num_experts
    route_counts = torch.bincount(
        top_k_index.reshape(-1),
        minlength=loaded.num_experts,
    )
    expected_count = token_count * loaded.top_k // loaded.num_experts
    if not bool((route_counts == expected_count).all()):
        raise AssertionError("balanced benchmark routes do not cover experts uniformly")
    sorted_routes = torch.sort(top_k_index, dim=-1).values
    if bool((sorted_routes[:, 1:] == sorted_routes[:, :-1]).any()):
        raise AssertionError("balanced benchmark routes repeat an expert per token")
    generator = torch.Generator(device="cpu").manual_seed(seed_value)
    hidden = torch.randn(
        token_count,
        loaded.hidden_size,
        generator=generator,
        dtype=torch.bfloat16,
    )
    routing_weights = torch.softmax(
        torch.randn(
            token_count,
            loaded.top_k,
            generator=generator,
            dtype=torch.float32,
        )
        * 3.0,
        dim=-1,
    )
    probe = torch.randn(
        token_count,
        loaded.hidden_size,
        generator=generator,
        dtype=torch.bfloat16,
    )
    metadata = {
        "tokens": token_count,
        "top_k": loaded.top_k,
        "route_pairs": int(top_k_index.numel()),
        "active_experts": int(torch.unique(top_k_index).numel()),
        "assignments_per_expert": route_counts.tolist(),
        "hidden_dtype": str(hidden.dtype),
        "routing_weight_dtype": str(routing_weights.dtype),
        "probe_dtype": str(probe.dtype),
        "seed": seed_value,
        "hidden_sha256": _sha256_tensor(torch, hidden),
        "top_k_index_sha256": _sha256_tensor(torch, top_k_index),
        "routing_weight_sha256": _sha256_tensor(torch, routing_weights),
        "probe_sha256": _sha256_tensor(torch, probe),
    }
    return hidden, top_k_index, routing_weights, probe, metadata


def _stats_checks(
    snapshot: Mapping[str, Any],
    *,
    workers: int,
    repeats: int,
    active_experts: int,
    warmup: _MeasuredRun,
    measured: Sequence[_MeasuredRun],
) -> dict[str, bool]:
    call_count = repeats + 1
    forward_wall = float(warmup.timing["forward"]["wall_seconds"]) + sum(
        float(run.timing["forward"]["wall_seconds"]) for run in measured
    )
    backward_wall = float(warmup.timing["backward"]["wall_seconds"]) + sum(
        float(run.timing["backward"]["wall_seconds"]) for run in measured
    )
    serial_forward_seconds = snapshot.get("serial_forward_seconds")
    parallel_task_seconds = snapshot.get("parallel_backward_task_seconds")
    reduction_seconds = snapshot.get("ordered_reduction_seconds")
    return {
        "workers": snapshot.get("workers") == workers,
        "patched_one_layer": snapshot.get("patched_layers") == 1,
        "serial_forward_calls": snapshot.get("serial_forward_calls") == call_count,
        "parallel_backward_calls": (
            snapshot.get("parallel_backward_calls") == call_count
        ),
        "expert_backward_tasks": (
            snapshot.get("expert_backward_tasks") == call_count * active_experts
        ),
        "restored_one_layer": snapshot.get("restored_layers") == 1,
        "context_inactive_after_exit": snapshot.get("context_active") is False,
        "executor_shutdown": snapshot.get("executor_shutdown") is True,
        "serial_forward_time_nested": (
            type(serial_forward_seconds) is float
            and math.isfinite(serial_forward_seconds)
            and 0.0 <= serial_forward_seconds <= forward_wall
        ),
        "parallel_backward_time_nested": (
            type(parallel_task_seconds) is float
            and type(reduction_seconds) is float
            and math.isfinite(parallel_task_seconds)
            and math.isfinite(reduction_seconds)
            and parallel_task_seconds >= 0.0
            and reduction_seconds >= 0.0
            and parallel_task_seconds + reduction_seconds <= backward_wall
        ),
    }


def _report_object(
    value: Any,
    *,
    keys: Sequence[str],
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} has invalid keys; missing={missing}, extra={extra}")
    return value


def _report_list(value: Any, *, label: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{label} must be an array")
    return value


def _report_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _report_int(
    value: Any,
    *,
    label: str,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _report_float(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    positive: bool = False,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite JSON float")
    if positive and value <= 0.0:
        raise ValueError(f"{label} must be positive")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _report_string(
    value: Any,
    *,
    label: str,
    allow_empty: bool = False,
) -> str:
    if type(value) is not str or (not allow_empty and not value):
        raise ValueError(f"{label} must be a string")
    return value


def _report_optional_string(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    return _report_string(value, label=label, allow_empty=True)


def _report_sha256(value: Any, *, label: str) -> str:
    text = _report_string(value, label=label)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _report_close(
    actual: float,
    expected: float,
    *,
    label: str,
) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label} is inconsistent")


def _validate_file_inventory(
    value: Any,
    *,
    label: str,
    named: bool,
) -> dict[str, Any]:
    keys = (
        ("path", "size_bytes", "sha256", "name")
        if named
        else (
            "path",
            "size_bytes",
            "sha256",
        )
    )
    item = _report_object(value, keys=keys, label=label)
    path_text = _report_string(item["path"], label=f"{label}.path")
    if not Path(path_text).is_absolute():
        raise ValueError(f"{label}.path must be absolute")
    _report_int(item["size_bytes"], label=f"{label}.size_bytes", minimum=1)
    _report_sha256(item["sha256"], label=f"{label}.sha256")
    if named:
        name = _report_string(item["name"], label=f"{label}.name")
        if Path(name).name != name:
            raise ValueError(f"{label}.name must be a local file name")
    return item


def _validate_authentication_inventory(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    inventory = _report_object(
        value,
        keys=("sources", "versions", "model"),
        label=label,
    )
    sources = _report_object(
        inventory["sources"],
        keys=_SOURCE_KEYS,
        label=f"{label}.sources",
    )
    for key in _SOURCE_KEYS:
        _validate_file_inventory(
            sources[key],
            label=f"{label}.sources.{key}",
            named=False,
        )
    versions = _report_object(
        inventory["versions"],
        keys=("transformers", "safetensors"),
        label=f"{label}.versions",
    )
    _report_string(
        versions["transformers"],
        label=f"{label}.versions.transformers",
    )
    _report_string(
        versions["safetensors"],
        label=f"{label}.versions.safetensors",
    )
    model = _report_object(
        inventory["model"],
        keys=("path", "config", "index", "selected_shards"),
        label=f"{label}.model",
    )
    model_path = _report_string(model["path"], label=f"{label}.model.path")
    if not Path(model_path).is_absolute():
        raise ValueError(f"{label}.model.path must be absolute")
    _validate_file_inventory(
        model["config"],
        label=f"{label}.model.config",
        named=False,
    )
    _validate_file_inventory(
        model["index"],
        label=f"{label}.model.index",
        named=False,
    )
    shards = _report_list(
        model["selected_shards"],
        label=f"{label}.model.selected_shards",
    )
    if not shards:
        raise ValueError(f"{label}.model.selected_shards must not be empty")
    names: list[str] = []
    for index, shard in enumerate(shards):
        validated = _validate_file_inventory(
            shard,
            label=f"{label}.model.selected_shards[{index}]",
            named=True,
        )
        names.append(validated["name"])
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError(f"{label}.model.selected_shards must be sorted and unique")
    return inventory


def _validate_model_report(
    value: Any,
    *,
    pre_authentication: Mapping[str, Any],
    post_loaded_state: Mapping[str, Any],
) -> dict[str, Any]:
    model = _report_object(
        value,
        keys=(
            "layer_index",
            "hidden_size",
            "intermediate_size",
            "num_experts",
            "top_k",
            "path",
            "config_sha256",
            "index_sha256",
            "source_layout",
            "selected_tensor_keys",
            "streamed_expert_slices",
            "selected_source_state_sha256",
            "source_dtype_counts",
            "selected_shards",
            "loaded_dtype",
            "loaded_parameter_bytes",
            "loaded_shapes",
            "loaded_state_sha256",
        ),
        label="report.model",
    )
    _report_int(model["layer_index"], label="report.model.layer_index", minimum=0)
    hidden_size = _report_int(
        model["hidden_size"],
        label="report.model.hidden_size",
        minimum=1,
    )
    intermediate_size = _report_int(
        model["intermediate_size"],
        label="report.model.intermediate_size",
        minimum=1,
    )
    num_experts = _report_int(
        model["num_experts"],
        label="report.model.num_experts",
        minimum=1,
    )
    top_k = _report_int(model["top_k"], label="report.model.top_k", minimum=1)
    if top_k > num_experts or num_experts % top_k:
        raise ValueError("report.model top-k/expert relationship is invalid")
    model_path = _report_string(model["path"], label="report.model.path")
    if not Path(model_path).is_absolute():
        raise ValueError("report.model.path must be absolute")
    config_sha256 = _report_sha256(
        model["config_sha256"],
        label="report.model.config_sha256",
    )
    index_sha256 = _report_sha256(
        model["index_sha256"],
        label="report.model.index_sha256",
    )
    source_layout = _report_string(
        model["source_layout"],
        label="report.model.source_layout",
    )
    if source_layout not in {"legacy_per_expert", "packed_experts"}:
        raise ValueError("report.model.source_layout is unsupported")
    expected_slices = num_experts * (3 if source_layout == "legacy_per_expert" else 2)
    selected_tensor_keys = _report_int(
        model["selected_tensor_keys"],
        label="report.model.selected_tensor_keys",
        minimum=1,
    )
    expected_keys = expected_slices if source_layout == "legacy_per_expert" else 2
    if selected_tensor_keys != expected_keys:
        raise ValueError("report.model.selected_tensor_keys is inconsistent")
    streamed_slices = _report_int(
        model["streamed_expert_slices"],
        label="report.model.streamed_expert_slices",
        minimum=1,
    )
    if streamed_slices != expected_slices:
        raise ValueError("report.model.streamed_expert_slices is inconsistent")
    _report_sha256(
        model["selected_source_state_sha256"],
        label="report.model.selected_source_state_sha256",
    )
    dtype_counts = model["source_dtype_counts"]
    if type(dtype_counts) is not dict or not dtype_counts:
        raise ValueError("report.model.source_dtype_counts must be an object")
    source_count = 0
    for dtype, count in dtype_counts.items():
        if type(dtype) is not str or dtype not in _SUPPORTED_SOURCE_DTYPES:
            raise ValueError("report.model.source_dtype_counts has invalid dtype")
        source_count += _report_int(
            count,
            label=f"report.model.source_dtype_counts.{dtype}",
            minimum=1,
        )
    if source_count != expected_slices:
        raise ValueError("report.model.source_dtype_counts is inconsistent")
    shards = _report_list(
        model["selected_shards"],
        label="report.model.selected_shards",
    )
    if not shards:
        raise ValueError("report.model.selected_shards must not be empty")
    shard_names: list[str] = []
    for index, shard_value in enumerate(shards):
        shard = _report_object(
            shard_value,
            keys=("name", "size_bytes", "sha256"),
            label=f"report.model.selected_shards[{index}]",
        )
        name = _report_string(
            shard["name"],
            label=f"report.model.selected_shards[{index}].name",
        )
        if Path(name).name != name:
            raise ValueError("report.model selected shard name is invalid")
        shard_names.append(name)
        _report_int(
            shard["size_bytes"],
            label=f"report.model.selected_shards[{index}].size_bytes",
            minimum=1,
        )
        _report_sha256(
            shard["sha256"],
            label=f"report.model.selected_shards[{index}].sha256",
        )
    if shard_names != sorted(shard_names) or len(shard_names) != len(set(shard_names)):
        raise ValueError("report.model.selected_shards must be sorted and unique")
    if model["loaded_dtype"] != "torch.bfloat16":
        raise ValueError("report.model.loaded_dtype must be torch.bfloat16")
    expected_bytes = num_experts * 3 * hidden_size * intermediate_size * 2
    if (
        _report_int(
            model["loaded_parameter_bytes"],
            label="report.model.loaded_parameter_bytes",
            minimum=1,
        )
        != expected_bytes
    ):
        raise ValueError("report.model.loaded_parameter_bytes is inconsistent")
    shapes = _report_object(
        model["loaded_shapes"],
        keys=("gate_up_proj", "down_proj"),
        label="report.model.loaded_shapes",
    )
    expected_shapes = {
        "gate_up_proj": [num_experts, 2 * intermediate_size, hidden_size],
        "down_proj": [num_experts, hidden_size, intermediate_size],
    }
    for name, expected in expected_shapes.items():
        shape = _report_list(
            shapes[name],
            label=f"report.model.loaded_shapes.{name}",
        )
        for dimension_index, dimension in enumerate(shape):
            _report_int(
                dimension,
                label=(f"report.model.loaded_shapes.{name}[{dimension_index}]"),
                minimum=1,
            )
        if shape != expected:
            raise ValueError(f"report.model.loaded_shapes.{name} is inconsistent")
    loaded_state = _report_object(
        model["loaded_state_sha256"],
        keys=("gate_up_proj", "down_proj"),
        label="report.model.loaded_state_sha256",
    )
    post_state = _report_object(
        post_loaded_state,
        keys=("gate_up_proj", "down_proj"),
        label="report.authentication.post_loaded_state_sha256",
    )
    for name in ("gate_up_proj", "down_proj"):
        _report_sha256(
            loaded_state[name],
            label=f"report.model.loaded_state_sha256.{name}",
        )
        _report_sha256(
            post_state[name],
            label=f"report.authentication.post_loaded_state_sha256.{name}",
        )

    auth_model = pre_authentication["model"]
    if model_path != auth_model["path"]:
        raise ValueError("report.model.path does not match authentication")
    if config_sha256 != auth_model["config"]["sha256"]:
        raise ValueError("report.model.config_sha256 does not match authentication")
    if index_sha256 != auth_model["index"]["sha256"]:
        raise ValueError("report.model.index_sha256 does not match authentication")
    authenticated_shards = [
        {
            "name": shard["name"],
            "size_bytes": shard["size_bytes"],
            "sha256": shard["sha256"],
        }
        for shard in auth_model["selected_shards"]
    ]
    if shards != authenticated_shards:
        raise ValueError("report.model.selected_shards does not match authentication")
    return model


def _validate_tensor_comparison(value: Any, *, label: str) -> dict[str, Any]:
    comparison = _report_object(
        value,
        keys=(
            "exact",
            "shape_and_dtype_match",
            "mismatch_count",
            "max_absolute_difference",
        ),
        label=label,
    )
    exact = _report_bool(comparison["exact"], label=f"{label}.exact")
    shape_match = _report_bool(
        comparison["shape_and_dtype_match"],
        label=f"{label}.shape_and_dtype_match",
    )
    mismatch_count = _report_int(
        comparison["mismatch_count"],
        label=f"{label}.mismatch_count",
        minimum=0,
    )
    maximum = comparison["max_absolute_difference"]
    if shape_match:
        max_difference = _report_float(
            maximum,
            label=f"{label}.max_absolute_difference",
            minimum=0.0,
        )
        expected_exact = mismatch_count == 0 and max_difference == 0.0
        if exact != expected_exact:
            raise ValueError(f"{label}.exact is inconsistent")
        if not exact and (mismatch_count == 0 or max_difference <= 0.0):
            raise ValueError(f"{label} mismatch evidence is inconsistent")
    else:
        if maximum is not None or exact or mismatch_count <= 0:
            raise ValueError(f"{label} shape-mismatch evidence is inconsistent")
    return comparison


def _validate_parity(value: Any, *, label: str) -> dict[str, Any]:
    parity = _report_object(
        value,
        keys=("values", "exact"),
        label=label,
    )
    values = _report_object(
        parity["values"],
        keys=("output", "hidden_gradient", "routing_weight_gradient"),
        label=f"{label}.values",
    )
    exact_values: list[bool] = []
    for name in ("output", "hidden_gradient", "routing_weight_gradient"):
        comparison = _validate_tensor_comparison(
            values[name],
            label=f"{label}.values.{name}",
        )
        exact_values.append(comparison["exact"])
    exact = _report_bool(parity["exact"], label=f"{label}.exact")
    if exact != all(exact_values):
        raise ValueError(f"{label}.exact is inconsistent")
    return parity


def _validate_timing_run(value: Any, *, label: str) -> dict[str, Any]:
    timing = _report_object(
        value,
        keys=("forward", "backward", "total"),
        label=label,
    )
    phases: dict[str, dict[str, float]] = {}
    for phase in ("forward", "backward", "total"):
        item = _report_object(
            timing[phase],
            keys=("wall_seconds", "process_cpu_seconds", "effective_cores"),
            label=f"{label}.{phase}",
        )
        wall = _report_float(
            item["wall_seconds"],
            label=f"{label}.{phase}.wall_seconds",
            positive=True,
        )
        cpu = _report_float(
            item["process_cpu_seconds"],
            label=f"{label}.{phase}.process_cpu_seconds",
            minimum=0.0,
        )
        effective = _report_float(
            item["effective_cores"],
            label=f"{label}.{phase}.effective_cores",
            minimum=0.0,
        )
        _report_close(
            effective,
            cpu / wall,
            label=f"{label}.{phase}.effective_cores",
        )
        phases[phase] = {"wall": wall, "cpu": cpu}
    _report_close(
        phases["total"]["wall"],
        phases["forward"]["wall"] + phases["backward"]["wall"],
        label=f"{label}.total.wall_seconds",
    )
    _report_close(
        phases["total"]["cpu"],
        phases["forward"]["cpu"] + phases["backward"]["cpu"],
        label=f"{label}.total.process_cpu_seconds",
    )
    return timing


def _validate_timing_summary(
    value: Any,
    *,
    label: str,
    repeats: int,
) -> dict[str, Any]:
    keys = ["runs"]
    for phase in ("forward", "backward", "total"):
        keys.extend(
            (
                f"mean_{phase}_wall_seconds",
                f"mean_{phase}_process_cpu_seconds",
                f"aggregate_{phase}_effective_cores",
            )
        )
    summary = _report_object(value, keys=keys, label=label)
    runs = _report_list(summary["runs"], label=f"{label}.runs")
    if len(runs) != repeats:
        raise ValueError(f"{label}.runs has the wrong cardinality")
    validated_runs = [
        _validate_timing_run(run, label=f"{label}.runs[{index}]")
        for index, run in enumerate(runs)
    ]
    for phase in ("forward", "backward", "total"):
        wall_values = [float(run[phase]["wall_seconds"]) for run in validated_runs]
        cpu_values = [
            float(run[phase]["process_cpu_seconds"]) for run in validated_runs
        ]
        expected_wall = sum(wall_values) / repeats
        expected_cpu = sum(cpu_values) / repeats
        expected_effective = sum(cpu_values) / sum(wall_values)
        wall = _report_float(
            summary[f"mean_{phase}_wall_seconds"],
            label=f"{label}.mean_{phase}_wall_seconds",
            positive=True,
        )
        cpu = _report_float(
            summary[f"mean_{phase}_process_cpu_seconds"],
            label=f"{label}.mean_{phase}_process_cpu_seconds",
            minimum=0.0,
        )
        effective = _report_float(
            summary[f"aggregate_{phase}_effective_cores"],
            label=f"{label}.aggregate_{phase}_effective_cores",
            minimum=0.0,
        )
        _report_close(
            wall,
            expected_wall,
            label=f"{label}.mean_{phase}_wall_seconds",
        )
        _report_close(
            cpu,
            expected_cpu,
            label=f"{label}.mean_{phase}_process_cpu_seconds",
        )
        _report_close(
            effective,
            expected_effective,
            label=f"{label}.aggregate_{phase}_effective_cores",
        )
    return summary


def _validate_workload_report(
    value: Any,
    *,
    model: Mapping[str, Any],
) -> dict[str, Any]:
    workload = _report_object(
        value,
        keys=(
            "tokens",
            "top_k",
            "route_pairs",
            "active_experts",
            "assignments_per_expert",
            "hidden_dtype",
            "routing_weight_dtype",
            "probe_dtype",
            "seed",
            "hidden_sha256",
            "top_k_index_sha256",
            "routing_weight_sha256",
            "probe_sha256",
        ),
        label="report.workload",
    )
    tokens = _report_int(
        workload["tokens"],
        label="report.workload.tokens",
        minimum=1,
    )
    top_k = _report_int(
        workload["top_k"],
        label="report.workload.top_k",
        minimum=1,
    )
    if top_k != model["top_k"]:
        raise ValueError("report.workload.top_k does not match model")
    route_pairs = _report_int(
        workload["route_pairs"],
        label="report.workload.route_pairs",
        minimum=1,
    )
    if route_pairs != tokens * top_k:
        raise ValueError("report.workload.route_pairs is inconsistent")
    active_experts = _report_int(
        workload["active_experts"],
        label="report.workload.active_experts",
        minimum=1,
    )
    if active_experts != model["num_experts"]:
        raise ValueError("report.workload does not activate every expert")
    assignments = _report_list(
        workload["assignments_per_expert"],
        label="report.workload.assignments_per_expert",
    )
    if len(assignments) != active_experts:
        raise ValueError("report.workload assignment cardinality is inconsistent")
    assignment_values = [
        _report_int(
            count,
            label=f"report.workload.assignments_per_expert[{index}]",
            minimum=1,
        )
        for index, count in enumerate(assignments)
    ]
    if sum(assignment_values) != route_pairs or len(set(assignment_values)) != 1:
        raise ValueError("report.workload assignments are not uniform")
    if workload["hidden_dtype"] != "torch.bfloat16":
        raise ValueError("report.workload.hidden_dtype is invalid")
    if workload["routing_weight_dtype"] != "torch.float32":
        raise ValueError("report.workload.routing_weight_dtype is invalid")
    if workload["probe_dtype"] != "torch.bfloat16":
        raise ValueError("report.workload.probe_dtype is invalid")
    _report_int(workload["seed"], label="report.workload.seed", minimum=0)
    for name in (
        "hidden_sha256",
        "top_k_index_sha256",
        "routing_weight_sha256",
        "probe_sha256",
    ):
        _report_sha256(workload[name], label=f"report.workload.{name}")
    return workload


def _validate_stats_report(
    value: Any,
    *,
    checks_value: Any,
    label: str,
    workers: int,
    repeats: int,
    active_experts: int,
    warmup_timing: Mapping[str, Any],
    measured_timing: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    stats = _report_object(
        value,
        keys=(
            "workers",
            "patched_layers",
            "serial_forward_calls",
            "serial_forward_seconds",
            "parallel_backward_calls",
            "expert_backward_tasks",
            "parallel_backward_task_seconds",
            "ordered_reduction_seconds",
            "restored_layers",
            "context_active",
            "executor_shutdown",
        ),
        label=label,
    )
    call_count = repeats + 1
    typed_workers = _report_int(
        stats["workers"],
        label=f"{label}.workers",
        minimum=1,
    )
    patched = _report_int(
        stats["patched_layers"],
        label=f"{label}.patched_layers",
        minimum=0,
    )
    forward_calls = _report_int(
        stats["serial_forward_calls"],
        label=f"{label}.serial_forward_calls",
        minimum=0,
    )
    backward_calls = _report_int(
        stats["parallel_backward_calls"],
        label=f"{label}.parallel_backward_calls",
        minimum=0,
    )
    tasks = _report_int(
        stats["expert_backward_tasks"],
        label=f"{label}.expert_backward_tasks",
        minimum=0,
    )
    restored = _report_int(
        stats["restored_layers"],
        label=f"{label}.restored_layers",
        minimum=0,
    )
    serial_seconds = _report_float(
        stats["serial_forward_seconds"],
        label=f"{label}.serial_forward_seconds",
        minimum=0.0,
    )
    task_seconds = _report_float(
        stats["parallel_backward_task_seconds"],
        label=f"{label}.parallel_backward_task_seconds",
        minimum=0.0,
    )
    reduction_seconds = _report_float(
        stats["ordered_reduction_seconds"],
        label=f"{label}.ordered_reduction_seconds",
        minimum=0.0,
    )
    context_active = _report_bool(
        stats["context_active"],
        label=f"{label}.context_active",
    )
    executor_shutdown = _report_bool(
        stats["executor_shutdown"],
        label=f"{label}.executor_shutdown",
    )
    forward_wall = float(warmup_timing["forward"]["wall_seconds"]) + sum(
        float(run["forward"]["wall_seconds"]) for run in measured_timing["runs"]
    )
    backward_wall = float(warmup_timing["backward"]["wall_seconds"]) + sum(
        float(run["backward"]["wall_seconds"]) for run in measured_timing["runs"]
    )
    expected_checks = {
        "workers": typed_workers == workers,
        "patched_one_layer": patched == 1,
        "serial_forward_calls": forward_calls == call_count,
        "parallel_backward_calls": backward_calls == call_count,
        "expert_backward_tasks": tasks == call_count * active_experts,
        "restored_one_layer": restored == 1,
        "context_inactive_after_exit": context_active is False,
        "executor_shutdown": executor_shutdown is True,
        "serial_forward_time_nested": 0.0 <= serial_seconds <= forward_wall,
        "parallel_backward_time_nested": (
            task_seconds + reduction_seconds <= backward_wall
        ),
    }
    checks = _report_object(
        checks_value,
        keys=_STATS_CHECK_KEYS,
        label=f"{label}_checks",
    )
    for key, expected in expected_checks.items():
        actual = _report_bool(
            checks[key],
            label=f"{label}_checks.{key}",
        )
        if actual != expected:
            raise ValueError(f"{label}_checks.{key} is inconsistent")
    return stats, checks


def _validate_eager_report(
    value: Any,
    *,
    repeats: int,
) -> tuple[dict[str, Any], bool]:
    eager = _report_object(
        value,
        keys=(
            "warmup",
            "timing",
            "repeat_parity",
            "frozen_expert_gradients_absent",
            "passed",
        ),
        label="report.eager",
    )
    warmup = _report_object(
        eager["warmup"],
        keys=(
            "timing_excluded_from_summary",
            "first_measured_parity",
            "frozen_expert_gradients_absent",
        ),
        label="report.eager.warmup",
    )
    _validate_timing_run(
        warmup["timing_excluded_from_summary"],
        label="report.eager.warmup.timing_excluded_from_summary",
    )
    warmup_parity = _validate_parity(
        warmup["first_measured_parity"],
        label="report.eager.warmup.first_measured_parity",
    )
    warmup_frozen = _report_bool(
        warmup["frozen_expert_gradients_absent"],
        label="report.eager.warmup.frozen_expert_gradients_absent",
    )
    _validate_timing_summary(
        eager["timing"],
        label="report.eager.timing",
        repeats=repeats,
    )
    repeat_parity = _report_list(
        eager["repeat_parity"],
        label="report.eager.repeat_parity",
    )
    if len(repeat_parity) != repeats:
        raise ValueError("report.eager.repeat_parity has the wrong cardinality")
    repeated = [
        _validate_parity(
            parity,
            label=f"report.eager.repeat_parity[{index}]",
        )
        for index, parity in enumerate(repeat_parity)
    ]
    frozen = _report_bool(
        eager["frozen_expert_gradients_absent"],
        label="report.eager.frozen_expert_gradients_absent",
    )
    expected_passed = (
        warmup_frozen
        and frozen
        and warmup_parity["exact"]
        and all(parity["exact"] for parity in repeated)
    )
    passed = _report_bool(eager["passed"], label="report.eager.passed")
    if passed != expected_passed:
        raise ValueError("report.eager.passed is inconsistent")
    return eager, passed


def _validate_proxy_report(
    value: Any,
    *,
    repeats: int,
    expected_workers: Sequence[int],
    active_experts: int,
    eager_timing: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    rows = _report_list(value, label="report.proxy")
    if len(rows) != len(expected_workers):
        raise ValueError("report.proxy has the wrong cardinality")
    validated: list[dict[str, Any]] = []
    row_passes: list[bool] = []
    for row_index, (row_value, worker_count) in enumerate(
        zip(rows, expected_workers, strict=True)
    ):
        label = f"report.proxy[{row_index}]"
        row = _report_object(
            row_value,
            keys=(
                "workers",
                "warmup",
                "timing",
                "eager_parity",
                "repeat_parity",
                "stats",
                "stats_checks",
                "frozen_expert_gradients_absent",
                "passed",
                "speedup_vs_eager",
            ),
            label=label,
        )
        workers = _report_int(
            row["workers"],
            label=f"{label}.workers",
            minimum=1,
        )
        if workers != worker_count:
            raise ValueError(f"{label}.workers does not match execution contract")
        warmup = _report_object(
            row["warmup"],
            keys=(
                "timing_excluded_from_summary",
                "eager_parity",
                "first_measured_parity",
                "frozen_expert_gradients_absent",
            ),
            label=f"{label}.warmup",
        )
        warmup_timing = _validate_timing_run(
            warmup["timing_excluded_from_summary"],
            label=f"{label}.warmup.timing_excluded_from_summary",
        )
        warmup_eager = _validate_parity(
            warmup["eager_parity"],
            label=f"{label}.warmup.eager_parity",
        )
        warmup_repeat = _validate_parity(
            warmup["first_measured_parity"],
            label=f"{label}.warmup.first_measured_parity",
        )
        warmup_frozen = _report_bool(
            warmup["frozen_expert_gradients_absent"],
            label=f"{label}.warmup.frozen_expert_gradients_absent",
        )
        timing = _validate_timing_summary(
            row["timing"],
            label=f"{label}.timing",
            repeats=repeats,
        )
        eager_parity = _report_list(
            row["eager_parity"],
            label=f"{label}.eager_parity",
        )
        repeat_parity = _report_list(
            row["repeat_parity"],
            label=f"{label}.repeat_parity",
        )
        if len(eager_parity) != repeats or len(repeat_parity) != repeats:
            raise ValueError(f"{label} parity cardinality is inconsistent")
        eager_values = [
            _validate_parity(
                parity,
                label=f"{label}.eager_parity[{index}]",
            )
            for index, parity in enumerate(eager_parity)
        ]
        repeat_values = [
            _validate_parity(
                parity,
                label=f"{label}.repeat_parity[{index}]",
            )
            for index, parity in enumerate(repeat_parity)
        ]
        _stats, checks = _validate_stats_report(
            row["stats"],
            checks_value=row["stats_checks"],
            label=f"{label}.stats",
            workers=workers,
            repeats=repeats,
            active_experts=active_experts,
            warmup_timing=warmup_timing,
            measured_timing=timing,
        )
        frozen = _report_bool(
            row["frozen_expert_gradients_absent"],
            label=f"{label}.frozen_expert_gradients_absent",
        )
        speedup = _report_object(
            row["speedup_vs_eager"],
            keys=("total_wall", "backward_wall"),
            label=f"{label}.speedup_vs_eager",
        )
        total_speedup = _report_float(
            speedup["total_wall"],
            label=f"{label}.speedup_vs_eager.total_wall",
            positive=True,
        )
        backward_speedup = _report_float(
            speedup["backward_wall"],
            label=f"{label}.speedup_vs_eager.backward_wall",
            positive=True,
        )
        _report_close(
            total_speedup,
            float(eager_timing["mean_total_wall_seconds"])
            / float(timing["mean_total_wall_seconds"]),
            label=f"{label}.speedup_vs_eager.total_wall",
        )
        _report_close(
            backward_speedup,
            float(eager_timing["mean_backward_wall_seconds"])
            / float(timing["mean_backward_wall_seconds"]),
            label=f"{label}.speedup_vs_eager.backward_wall",
        )
        expected_passed = (
            warmup_eager["exact"]
            and warmup_repeat["exact"]
            and warmup_frozen
            and frozen
            and all(parity["exact"] for parity in eager_values)
            and all(parity["exact"] for parity in repeat_values)
            and all(checks.values())
        )
        passed = _report_bool(row["passed"], label=f"{label}.passed")
        if passed != expected_passed:
            raise ValueError(f"{label}.passed is inconsistent")
        validated.append(row)
        row_passes.append(passed)
    return validated, all(row_passes)


def _validate_execution_contract(
    value: Any,
) -> tuple[dict[str, Any], int, list[int]]:
    contract = _report_object(
        value,
        keys=(
            "repeats",
            "warmups_per_implementation",
            "workers",
            "torch_intraop_threads",
            "torch_interop_threads_observed",
            "deterministic_algorithms",
            "experts_implementation",
            "forward_is_serial",
            "expert_weight_gradients_prohibited",
            "exact_parity_required",
            "measurement_order",
            "counterbalanced",
            "timing_classification",
        ),
        label="report.execution_contract",
    )
    repeats = _report_int(
        contract["repeats"],
        label="report.execution_contract.repeats",
        minimum=1,
    )
    if (
        _report_int(
            contract["warmups_per_implementation"],
            label="report.execution_contract.warmups_per_implementation",
            minimum=0,
        )
        != 1
    ):
        raise ValueError("report must contain exactly one warmup per implementation")
    worker_values = _report_list(
        contract["workers"],
        label="report.execution_contract.workers",
    )
    if not worker_values:
        raise ValueError("report.execution_contract.workers must not be empty")
    workers = [
        _report_int(
            worker,
            label=f"report.execution_contract.workers[{index}]",
            minimum=1,
        )
        for index, worker in enumerate(worker_values)
    ]
    if len(workers) != len(set(workers)):
        raise ValueError("report.execution_contract.workers must be unique")
    if (
        _report_int(
            contract["torch_intraop_threads"],
            label="report.execution_contract.torch_intraop_threads",
            minimum=1,
        )
        != 1
    ):
        raise ValueError("report.execution_contract must use one intraop thread")
    _report_int(
        contract["torch_interop_threads_observed"],
        label="report.execution_contract.torch_interop_threads_observed",
        minimum=1,
    )
    if contract["experts_implementation"] != "eager":
        raise ValueError(
            "report.execution_contract.experts_implementation must be eager"
        )
    for name in (
        "deterministic_algorithms",
        "forward_is_serial",
        "expert_weight_gradients_prohibited",
        "exact_parity_required",
    ):
        if (
            _report_bool(
                contract[name],
                label=f"report.execution_contract.{name}",
            )
            is not True
        ):
            raise ValueError(f"report.execution_contract.{name} must be true")
    measurement_order = _report_list(
        contract["measurement_order"],
        label="report.execution_contract.measurement_order",
    )
    expected_order = [
        "eager",
        *(f"proxy_workers_{worker}" for worker in workers),
    ]
    for index, item in enumerate(measurement_order):
        _report_string(
            item,
            label=f"report.execution_contract.measurement_order[{index}]",
        )
    if measurement_order != expected_order:
        raise ValueError("report.execution_contract.measurement_order is invalid")
    if (
        _report_bool(
            contract["counterbalanced"],
            label="report.execution_contract.counterbalanced",
        )
        is not False
    ):
        raise ValueError("report.execution_contract.counterbalanced must be false")
    if contract["timing_classification"] != _TIMING_CLASSIFICATION:
        raise ValueError("report.execution_contract timing classification is invalid")
    return contract, repeats, workers


def _validate_environment(
    value: Any,
    *,
    pre_authentication: Mapping[str, Any],
) -> dict[str, Any]:
    environment = _report_object(
        value,
        keys=(
            "python",
            "platform",
            "processor",
            "torch",
            "torch_cpu_capability",
            "cpu_affinity",
            "transformers",
            "safetensors",
            "transformers_olmoe_source",
            "omp_num_threads",
            "mkl_num_threads",
            "openblas_num_threads",
        ),
        label="report.environment",
    )
    for name in ("python", "platform", "torch", "transformers", "safetensors"):
        _report_string(environment[name], label=f"report.environment.{name}")
    _report_string(
        environment["processor"],
        label="report.environment.processor",
        allow_empty=True,
    )
    if environment["torch_cpu_capability"] is not None:
        _report_string(
            environment["torch_cpu_capability"],
            label="report.environment.torch_cpu_capability",
            allow_empty=True,
        )
    affinity = environment["cpu_affinity"]
    if affinity is not None:
        affinity_values = _report_list(
            affinity,
            label="report.environment.cpu_affinity",
        )
        parsed_affinity = [
            _report_int(
                cpu,
                label=f"report.environment.cpu_affinity[{index}]",
                minimum=0,
            )
            for index, cpu in enumerate(affinity_values)
        ]
        if parsed_affinity != sorted(set(parsed_affinity)):
            raise ValueError("report.environment.cpu_affinity is invalid")
    for name in ("omp_num_threads", "mkl_num_threads", "openblas_num_threads"):
        _report_optional_string(
            environment[name],
            label=f"report.environment.{name}",
        )
    oracle_source = _report_string(
        environment["transformers_olmoe_source"],
        label="report.environment.transformers_olmoe_source",
    )
    if environment["transformers"] != pre_authentication["versions"]["transformers"]:
        raise ValueError("report.environment.transformers is unauthenticated")
    if environment["safetensors"] != pre_authentication["versions"]["safetensors"]:
        raise ValueError("report.environment.safetensors is unauthenticated")
    if oracle_source != pre_authentication["sources"]["transformers_olmoe"]["path"]:
        raise ValueError("report.environment OLMoE source is unauthenticated")
    return environment


def _validate_authentication(
    value: Any,
) -> tuple[dict[str, Any], bool]:
    authentication = _report_object(
        value,
        keys=(
            "pre_run",
            "post_run",
            "post_loaded_state_sha256",
            "checks",
            "passed",
        ),
        label="report.authentication",
    )
    pre = _validate_authentication_inventory(
        authentication["pre_run"],
        label="report.authentication.pre_run",
    )
    post = _validate_authentication_inventory(
        authentication["post_run"],
        label="report.authentication.post_run",
    )
    post_loaded = _report_object(
        authentication["post_loaded_state_sha256"],
        keys=("gate_up_proj", "down_proj"),
        label="report.authentication.post_loaded_state_sha256",
    )
    for name in ("gate_up_proj", "down_proj"):
        _report_sha256(
            post_loaded[name],
            label=f"report.authentication.post_loaded_state_sha256.{name}",
        )
    checks = _report_object(
        authentication["checks"],
        keys=_AUTHENTICATION_CHECK_KEYS,
        label="report.authentication.checks",
    )
    for key in _AUTHENTICATION_CHECK_KEYS:
        _report_bool(
            checks[key],
            label=f"report.authentication.checks.{key}",
        )
    expected_without_loaded = {
        "benchmark_source_unchanged": (
            pre["sources"]["benchmark"] == post["sources"]["benchmark"]
        ),
        "proxy_source_unchanged": (pre["sources"]["proxy"] == post["sources"]["proxy"]),
        "transformers_oracle_unchanged": (
            pre["sources"]["transformers_olmoe"]
            == post["sources"]["transformers_olmoe"]
        ),
        "versions_unchanged": pre["versions"] == post["versions"],
        "model_path_unchanged": pre["model"]["path"] == post["model"]["path"],
        "config_unchanged": pre["model"]["config"] == post["model"]["config"],
        "index_unchanged": pre["model"]["index"] == post["model"]["index"],
        "selected_shards_unchanged": (
            pre["model"]["selected_shards"] == post["model"]["selected_shards"]
        ),
    }
    for key, expected in expected_without_loaded.items():
        if checks[key] != expected:
            raise ValueError(f"report.authentication.checks.{key} is inconsistent")
    passed = _report_bool(
        authentication["passed"],
        label="report.authentication.passed",
    )
    return authentication, passed


def _validate_report_object(value: Any) -> dict[str, Any]:
    report = _report_object(
        value,
        keys=(
            "schema_version",
            "experiment",
            "status",
            "parity_passed",
            "evidence_passed",
            "authentication",
            "source_sha256",
            "model",
            "workload",
            "execution_contract",
            "environment",
            "eager",
            "proxy",
            "best_proxy_workers_by_mean_total_wall",
            "decision",
        ),
        label="report",
    )
    if (
        _report_int(
            report["schema_version"],
            label="report.schema_version",
            minimum=1,
        )
        != _SCHEMA_VERSION
    ):
        raise ValueError("report.schema_version is unsupported")
    if report["experiment"] != _EXPERIMENT:
        raise ValueError("report.experiment is invalid")
    authentication, reported_auth_passed = _validate_authentication(
        report["authentication"]
    )
    pre_authentication = authentication["pre_run"]
    post_loaded_state = authentication["post_loaded_state_sha256"]
    model = _validate_model_report(
        report["model"],
        pre_authentication=pre_authentication,
        post_loaded_state=post_loaded_state,
    )
    expected_loaded_checks = {
        "loaded_gate_up_unchanged": (
            model["loaded_state_sha256"]["gate_up_proj"]
            == post_loaded_state["gate_up_proj"]
        ),
        "loaded_down_unchanged": (
            model["loaded_state_sha256"]["down_proj"] == post_loaded_state["down_proj"]
        ),
    }
    checks = authentication["checks"]
    for key, expected in expected_loaded_checks.items():
        if checks[key] != expected:
            raise ValueError(f"report.authentication.checks.{key} is inconsistent")
    expected_auth_passed = all(checks.values())
    if reported_auth_passed != expected_auth_passed:
        raise ValueError("report.authentication.passed is inconsistent")
    source_sha256 = _report_object(
        report["source_sha256"],
        keys=(
            "src/engram/evaluation/olmoe_expert_proxy.py",
            "src/engram/evaluation/olmoe_expert_proxy_benchmark.py",
        ),
        label="report.source_sha256",
    )
    expected_source_sha256 = {
        "src/engram/evaluation/olmoe_expert_proxy.py": (
            pre_authentication["sources"]["proxy"]["sha256"]
        ),
        "src/engram/evaluation/olmoe_expert_proxy_benchmark.py": (
            pre_authentication["sources"]["benchmark"]["sha256"]
        ),
    }
    for name, digest in source_sha256.items():
        _report_sha256(digest, label=f"report.source_sha256.{name}")
    if source_sha256 != expected_source_sha256:
        raise ValueError("report.source_sha256 does not match authentication")
    _validate_workload_report(report["workload"], model=model)
    _contract, repeats, workers = _validate_execution_contract(
        report["execution_contract"]
    )
    _validate_environment(
        report["environment"],
        pre_authentication=pre_authentication,
    )
    eager, eager_passed = _validate_eager_report(
        report["eager"],
        repeats=repeats,
    )
    proxy, proxy_passed = _validate_proxy_report(
        report["proxy"],
        repeats=repeats,
        expected_workers=workers,
        active_experts=report["workload"]["active_experts"],
        eager_timing=eager["timing"],
    )
    expected_parity_passed = eager_passed and proxy_passed
    parity_passed = _report_bool(
        report["parity_passed"],
        label="report.parity_passed",
    )
    if parity_passed != expected_parity_passed:
        raise ValueError("report.parity_passed is inconsistent")
    expected_evidence_passed = parity_passed and expected_auth_passed
    evidence_passed = _report_bool(
        report["evidence_passed"],
        label="report.evidence_passed",
    )
    if evidence_passed != expected_evidence_passed:
        raise ValueError("report.evidence_passed is inconsistent")
    expected_status = (
        "exact_parity_passed"
        if evidence_passed
        else ("authentication_failed" if not expected_auth_passed else "parity_failed")
    )
    status = _report_string(report["status"], label="report.status")
    if status != expected_status:
        raise ValueError("report.status is inconsistent")
    best_worker = _report_int(
        report["best_proxy_workers_by_mean_total_wall"],
        label="report.best_proxy_workers_by_mean_total_wall",
        minimum=1,
    )
    expected_best = int(
        min(
            proxy,
            key=lambda row: float(row["timing"]["mean_total_wall_seconds"]),
        )["workers"]
    )
    if best_worker != expected_best:
        raise ValueError("report best proxy worker is inconsistent")
    expected_decision = (
        "proxy_exact_for_real_layer_performance_is_measured"
        if evidence_passed
        else (
            "do_not_use_proxy_authentication_failed"
            if not expected_auth_passed
            else "do_not_use_proxy_real_layer_parity_failed"
        )
    )
    decision = _report_string(report["decision"], label="report.decision")
    if decision != expected_decision:
        raise ValueError("report.decision is inconsistent")
    return report


def validate_frozen_olmoe_expert_proxy_report(
    report_path: str | Path,
) -> dict[str, Any]:
    """Read and strictly validate one durable expert-proxy report."""

    report = _read_json_object(Path(report_path), "expert-proxy report")
    return _validate_report_object(report)


def benchmark_frozen_olmoe_expert_proxy(
    model: str | Path,
    *,
    out: str | Path,
    layer: int = 0,
    tokens: int = _DEFAULT_TOKENS,
    repeats: int = _DEFAULT_REPEATS,
    workers: Sequence[int] = _DEFAULT_WORKERS,
    seed: int = _DEFAULT_SEED,
) -> dict[str, Any]:
    """Run exact eager/proxy parity and atomically write the benchmark report."""

    torch = _require_torch()
    repeat_count = _positive_int(repeats, "repeats")
    worker_counts = tuple(_positive_int(value, "workers") for value in workers)
    if not worker_counts or len(set(worker_counts)) != len(worker_counts):
        raise ValueError("workers must contain distinct positive integers")
    pre_authentication = _authentication_inventory(model, layer=layer)
    loaded = load_frozen_olmoe_expert_layer(model, layer=layer)
    hidden, top_k_index, routing_weights, probe, workload = _workload(
        torch,
        loaded,
        tokens=tokens,
        seed=seed,
    )
    prior_threads = torch.get_num_threads()
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    eager_warmup: _MeasuredRun | None = None
    eager_runs: list[_MeasuredRun] = []
    proxy_rows: list[dict[str, Any]] = []
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        eager_warmup = _measure_run(
            torch,
            loaded.experts,
            hidden,
            top_k_index,
            routing_weights,
            probe,
        )
        eager_runs = [
            _measure_run(
                torch,
                loaded.experts,
                hidden,
                top_k_index,
                routing_weights,
                probe,
            )
            for _ in range(repeat_count)
        ]
        eager_repeat_parity = [_parity(torch, run, eager_runs[0]) for run in eager_runs]
        eager_warmup_parity = _parity(torch, eager_warmup, eager_runs[0])
        for worker_count in worker_counts:
            with frozen_olmoe_expert_backward_proxy(
                loaded.owner,
                workers=worker_count,
            ) as stats:
                warmup = _measure_run(
                    torch,
                    loaded.experts,
                    hidden,
                    top_k_index,
                    routing_weights,
                    probe,
                )
                measured = [
                    _measure_run(
                        torch,
                        loaded.experts,
                        hidden,
                        top_k_index,
                        routing_weights,
                        probe,
                    )
                    for _ in range(repeat_count)
                ]
            snapshot = stats.snapshot()
            checks = _stats_checks(
                snapshot,
                workers=worker_count,
                repeats=repeat_count,
                active_experts=int(workload["active_experts"]),
                warmup=warmup,
                measured=measured,
            )
            parity = [_parity(torch, run, eager_runs[0]) for run in measured]
            repeat_parity = [_parity(torch, run, measured[0]) for run in measured]
            warmup_eager_parity = _parity(torch, warmup, eager_warmup)
            warmup_repeat_parity = _parity(torch, warmup, measured[0])
            warmup_frozen = warmup.frozen_expert_gradients_absent
            frozen = warmup_frozen and all(
                run.frozen_expert_gradients_absent for run in measured
            )
            proxy_rows.append(
                {
                    "workers": worker_count,
                    "warmup": {
                        "timing_excluded_from_summary": warmup.timing,
                        "eager_parity": warmup_eager_parity,
                        "first_measured_parity": warmup_repeat_parity,
                        "frozen_expert_gradients_absent": warmup_frozen,
                    },
                    "timing": _timing_summary(measured),
                    "eager_parity": parity,
                    "repeat_parity": repeat_parity,
                    "stats": snapshot,
                    "stats_checks": checks,
                    "frozen_expert_gradients_absent": frozen,
                    "passed": (
                        warmup_eager_parity["exact"]
                        and warmup_repeat_parity["exact"]
                        and all(row["exact"] for row in parity)
                        and all(row["exact"] for row in repeat_parity)
                        and all(checks.values())
                        and frozen
                    ),
                }
            )
    finally:
        torch.use_deterministic_algorithms(prior_deterministic)
        torch.set_num_threads(prior_threads)

    if eager_warmup is None:
        raise AssertionError("eager warmup did not execute")
    eager_frozen = eager_warmup.frozen_expert_gradients_absent and all(
        run.frozen_expert_gradients_absent for run in eager_runs
    )
    eager_exact = eager_warmup_parity["exact"] and all(
        row["exact"] for row in eager_repeat_parity
    )
    proxy_passed = all(row["passed"] for row in proxy_rows)
    parity_passed = eager_frozen and eager_exact and proxy_passed
    eager_timing = _timing_summary(eager_runs)
    eager_total = float(eager_timing["mean_total_wall_seconds"])
    for row in proxy_rows:
        proxy_total = float(row["timing"]["mean_total_wall_seconds"])
        proxy_backward = float(row["timing"]["mean_backward_wall_seconds"])
        row["speedup_vs_eager"] = {
            "total_wall": eager_total / proxy_total,
            "backward_wall": (
                float(eager_timing["mean_backward_wall_seconds"]) / proxy_backward
            ),
        }
    best = min(
        proxy_rows,
        key=lambda row: float(row["timing"]["mean_total_wall_seconds"]),
    )
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    post_loaded_state_sha256 = {
        "gate_up_proj": _sha256_tensor_state(
            torch,
            loaded.experts.gate_up_proj,
            name="gate_up_proj",
        ),
        "down_proj": _sha256_tensor_state(
            torch,
            loaded.experts.down_proj,
            name="down_proj",
        ),
    }
    post_authentication = _authentication_inventory(model, layer=layer)
    authentication_checks = {
        "benchmark_source_unchanged": (
            pre_authentication["sources"]["benchmark"]
            == post_authentication["sources"]["benchmark"]
        ),
        "proxy_source_unchanged": (
            pre_authentication["sources"]["proxy"]
            == post_authentication["sources"]["proxy"]
        ),
        "transformers_oracle_unchanged": (
            pre_authentication["sources"]["transformers_olmoe"]
            == post_authentication["sources"]["transformers_olmoe"]
        ),
        "versions_unchanged": (
            pre_authentication["versions"] == post_authentication["versions"]
        ),
        "model_path_unchanged": (
            pre_authentication["model"]["path"] == post_authentication["model"]["path"]
        ),
        "config_unchanged": (
            pre_authentication["model"]["config"]
            == post_authentication["model"]["config"]
        ),
        "index_unchanged": (
            pre_authentication["model"]["index"]
            == post_authentication["model"]["index"]
        ),
        "selected_shards_unchanged": (
            pre_authentication["model"]["selected_shards"]
            == post_authentication["model"]["selected_shards"]
        ),
        "loaded_gate_up_unchanged": (
            loaded.model_info["loaded_state_sha256"]["gate_up_proj"]
            == post_loaded_state_sha256["gate_up_proj"]
        ),
        "loaded_down_unchanged": (
            loaded.model_info["loaded_state_sha256"]["down_proj"]
            == post_loaded_state_sha256["down_proj"]
        ),
    }
    authentication_passed = all(authentication_checks.values())
    evidence_passed = parity_passed and authentication_passed
    authenticated_shards = [
        {
            "name": item["name"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in pre_authentication["model"]["selected_shards"]
    ]
    model_info = dict(loaded.model_info)
    model_info.update(
        {
            "path": pre_authentication["model"]["path"],
            "config_sha256": pre_authentication["model"]["config"]["sha256"],
            "index_sha256": pre_authentication["model"]["index"]["sha256"],
            "selected_shards": authenticated_shards,
        }
    )
    report = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _EXPERIMENT,
        "status": (
            "exact_parity_passed"
            if evidence_passed
            else (
                "authentication_failed"
                if not authentication_passed
                else "parity_failed"
            )
        ),
        "parity_passed": parity_passed,
        "evidence_passed": evidence_passed,
        "authentication": {
            "pre_run": pre_authentication,
            "post_run": post_authentication,
            "post_loaded_state_sha256": post_loaded_state_sha256,
            "checks": authentication_checks,
            "passed": authentication_passed,
        },
        "source_sha256": {
            "src/engram/evaluation/olmoe_expert_proxy.py": (
                pre_authentication["sources"]["proxy"]["sha256"]
            ),
            "src/engram/evaluation/olmoe_expert_proxy_benchmark.py": (
                pre_authentication["sources"]["benchmark"]["sha256"]
            ),
        },
        "model": {
            "layer_index": loaded.layer_index,
            "hidden_size": loaded.hidden_size,
            "intermediate_size": loaded.intermediate_size,
            "num_experts": loaded.num_experts,
            "top_k": loaded.top_k,
            **model_info,
        },
        "workload": workload,
        "execution_contract": {
            "repeats": repeat_count,
            "warmups_per_implementation": 1,
            "workers": list(worker_counts),
            "torch_intraop_threads": 1,
            "torch_interop_threads_observed": torch.get_num_interop_threads(),
            "deterministic_algorithms": True,
            "experts_implementation": "eager",
            "forward_is_serial": True,
            "expert_weight_gradients_prohibited": True,
            "exact_parity_required": True,
            "measurement_order": [
                "eager",
                *(f"proxy_workers_{value}" for value in worker_counts),
            ],
            "counterbalanced": False,
            "timing_classification": _TIMING_CLASSIFICATION,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": str(torch.__version__),
            "torch_cpu_capability": (
                torch.backends.cpu.get_cpu_capability()
                if callable(
                    getattr(
                        getattr(torch.backends, "cpu", None),
                        "get_cpu_capability",
                        None,
                    )
                )
                else None
            ),
            "cpu_affinity": affinity,
            "transformers": pre_authentication["versions"]["transformers"],
            "safetensors": pre_authentication["versions"]["safetensors"],
            "transformers_olmoe_source": (
                pre_authentication["sources"]["transformers_olmoe"]["path"]
            ),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "eager": {
            "warmup": {
                "timing_excluded_from_summary": eager_warmup.timing,
                "first_measured_parity": eager_warmup_parity,
                "frozen_expert_gradients_absent": (
                    eager_warmup.frozen_expert_gradients_absent
                ),
            },
            "timing": eager_timing,
            "repeat_parity": eager_repeat_parity,
            "frozen_expert_gradients_absent": eager_frozen,
            "passed": eager_frozen and eager_exact,
        },
        "proxy": proxy_rows,
        "best_proxy_workers_by_mean_total_wall": int(best["workers"]),
        "decision": (
            "proxy_exact_for_real_layer_performance_is_measured"
            if evidence_passed
            else (
                "do_not_use_proxy_authentication_failed"
                if not authentication_passed
                else "do_not_use_proxy_real_layer_parity_failed"
            )
        ),
    }
    for value in (
        eager_total,
        *(float(row["timing"]["mean_total_wall_seconds"]) for row in proxy_rows),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError("benchmark produced invalid performance evidence")
    _validate_report_object(report)
    atomic_json(out, report)
    disk_report = validate_frozen_olmoe_expert_proxy_report(out)
    if disk_report != report:
        raise RuntimeError("atomic benchmark report changed during serialization")
    return disk_report


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark exact eager/proxy parity for one streamed real OLMoE "
            "expert layer"
        )
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=_DEFAULT_TOKENS)
    parser.add_argument("--repeats", type=int, default=_DEFAULT_REPEATS)
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=list(_DEFAULT_WORKERS),
    )
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    report = benchmark_frozen_olmoe_expert_proxy(
        args.model,
        out=args.out,
        layer=args.layer,
        tokens=args.tokens,
        repeats=args.repeats,
        workers=args.workers,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["evidence_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
