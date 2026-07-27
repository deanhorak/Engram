"""Fail-closed source audit for OLMoE's addressable sparse experts.

This is deliberately separate from the dense Llama inspector.  OLMoE layers
contain a learned router and independently stored SwiGLU experts, which is the
model contract Engram needs to evaluate before attempting semantic compilation.
The Hub path downloads only config and tensor-index metadata.
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from engram.models.inspection import (
    ModelValidationError,
    TensorInfo,
    local_tensor_inventory,
)
from engram.utils import sha256_file


OFFICIAL_OLMOE_REPO = "allenai/OLMoE-1B-7B-0125"
OFFICIAL_OLMOE_MODEL_CARD = f"https://huggingface.co/{OFFICIAL_OLMOE_REPO}"
OFFICIAL_OLMOE_PAPER = "https://arxiv.org/abs/2409.02060"
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024


class OLMoEValidationError(ValueError):
    """Raised when OLMoE metadata cannot be read safely."""


def _snapshot_revision(path: Path) -> str | None:
    if path.parent.parent.name != "snapshots":
        return None
    revision = path.parent.name
    if len(revision) == 40 and all(c in "0123456789abcdef" for c in revision):
        return revision
    return None


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OLMoEValidationError(f"invalid {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OLMoEValidationError(f"{label} must be a JSON object")
    return payload


def _resolve_metadata(
    model: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
) -> tuple[Path, Path | None, str | None, str | None]:
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        model_path = candidate.resolve()
        config = model_path / "config.json"
        if not config.is_file():
            raise OLMoEValidationError(f"missing model config: {config}")
        index = model_path / "model.safetensors.index.json"
        return (
            config,
            index if index.is_file() else None,
            None,
            _snapshot_revision(config),
        )
    if candidate.exists():
        raise OLMoEValidationError(
            f"model path is not a directory: {candidate.resolve()}"
        )
    model_id = str(model)
    if (
        Path(model_id).is_absolute()
        or model_id.startswith(("./", "../", "~"))
        or "/" not in model_id
    ):
        raise OLMoEValidationError(
            f"model path is not a directory or Hub repository ID: {model_id!r}"
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise OLMoEValidationError(
            "install engram-lm[conversion] to audit a Hugging Face source"
        ) from exc
    kwargs = {
        "repo_id": model_id,
        "revision": revision,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
    }
    try:
        downloaded_config = Path(hf_hub_download(filename="config.json", **kwargs))
        downloaded_index = Path(
            hf_hub_download(filename="model.safetensors.index.json", **kwargs)
        )
    except Exception as exc:
        raise OLMoEValidationError(
            f"could not download OLMoE metadata for {model_id!r}: {exc}"
        ) from exc
    resolved = _snapshot_revision(downloaded_config) or revision
    return (
        downloaded_config.resolve(),
        downloaded_index.resolve(),
        model_id,
        resolved,
    )


def _parse_safetensors_header(payload: bytes, shard: str) -> tuple[TensorInfo, ...]:
    if len(payload) < 8:
        raise OLMoEValidationError(f"truncated safetensors prefix for {shard}")
    header_bytes = struct.unpack("<Q", payload[:8])[0]
    if not 2 <= header_bytes <= _MAX_SAFETENSORS_HEADER_BYTES:
        raise OLMoEValidationError(
            f"unsafe safetensors header length {header_bytes} for {shard}"
        )
    if len(payload) != 8 + header_bytes:
        raise OLMoEValidationError(
            f"incomplete safetensors header for {shard}: "
            f"received {len(payload) - 8}, expected {header_bytes} bytes"
        )
    try:
        header = json.loads(payload[8:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OLMoEValidationError(
            f"invalid safetensors header JSON for {shard}: {exc}"
        ) from exc
    if not isinstance(header, dict):
        raise OLMoEValidationError(f"safetensors header for {shard} is not an object")
    tensors = []
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise OLMoEValidationError(
                f"invalid tensor entry in safetensors header for {shard}"
            )
        shape = metadata.get("shape")
        dtype = metadata.get("dtype")
        if (
            not isinstance(shape, list)
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in shape
            )
            or not isinstance(dtype, str)
        ):
            raise OLMoEValidationError(
                f"invalid shape/dtype for tensor {name!r} in {shard}"
            )
        tensors.append(TensorInfo(name, tuple(shape), dtype, shard))
    return tuple(sorted(tensors, key=lambda item: item.name))


def _remote_safetensors_inventory(
    repo_id: str,
    revision: str,
    shards: set[str],
) -> tuple[TensorInfo, ...]:
    """Read only bounded safetensors headers using verified range responses."""

    try:
        import httpx
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers
    except ImportError as exc:
        raise OLMoEValidationError(
            "install engram-lm[conversion] for remote shape auditing"
        ) from exc

    base_headers = dict(build_hf_headers())
    result: list[TensorInfo] = []
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        for shard in sorted(shards):
            url = hf_hub_url(repo_id, shard, revision=revision)
            prefix_headers = {**base_headers, "Range": "bytes=0-7"}
            with client.stream("GET", url, headers=prefix_headers) as response:
                if response.status_code != 206:
                    raise OLMoEValidationError(
                        f"refusing unbounded response for {shard}: "
                        f"HTTP {response.status_code}, expected 206"
                    )
                prefix = response.read()
            if len(prefix) != 8:
                raise OLMoEValidationError(
                    f"invalid range length {len(prefix)} for {shard} prefix"
                )
            header_bytes = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_bytes <= _MAX_SAFETENSORS_HEADER_BYTES:
                raise OLMoEValidationError(
                    f"unsafe safetensors header length {header_bytes} for {shard}"
                )
            last_byte = 8 + header_bytes - 1
            header_headers = {
                **base_headers,
                "Range": f"bytes=0-{last_byte}",
            }
            with client.stream("GET", url, headers=header_headers) as response:
                if response.status_code != 206:
                    raise OLMoEValidationError(
                        f"refusing unbounded response for {shard}: "
                        f"HTTP {response.status_code}, expected 206"
                    )
                payload = response.read()
            result.extend(_parse_safetensors_header(payload, shard))
    return tuple(sorted(result, key=lambda item: item.name))


def _positive_int(config: dict[str, Any], key: str) -> int | None:
    value = config.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def required_olmoe_tensor_shapes(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    num_heads: int,
    num_key_value_heads: int,
    num_experts: int,
    vocab_size: int,
) -> dict[str, tuple[int, ...]]:
    """Return the exact tensor/shape contract for the supported OLMoE layout."""

    head_dim = hidden_size // num_heads
    kv_width = num_key_value_heads * head_dim
    result = {
        "model.embed_tokens.weight": (vocab_size, hidden_size),
        "model.norm.weight": (hidden_size,),
        "lm_head.weight": (vocab_size, hidden_size),
    }
    for layer in range(num_layers):
        base = f"model.layers.{layer}"
        result[f"{base}.input_layernorm.weight"] = (hidden_size,)
        result[f"{base}.post_attention_layernorm.weight"] = (hidden_size,)
        result[f"{base}.self_attn.q_proj.weight"] = (hidden_size, hidden_size)
        result[f"{base}.self_attn.k_proj.weight"] = (kv_width, hidden_size)
        result[f"{base}.self_attn.v_proj.weight"] = (kv_width, hidden_size)
        result[f"{base}.self_attn.o_proj.weight"] = (hidden_size, hidden_size)
        # OLMoE normalizes flattened Q/K projections before splitting heads.
        result[f"{base}.self_attn.q_norm.weight"] = (hidden_size,)
        result[f"{base}.self_attn.k_norm.weight"] = (kv_width,)
        result[f"{base}.mlp.gate.weight"] = (num_experts, hidden_size)
        for expert in range(num_experts):
            prefix = f"{base}.mlp.experts.{expert}"
            result[f"{prefix}.gate_proj.weight"] = (
                intermediate_size,
                hidden_size,
            )
            result[f"{prefix}.up_proj.weight"] = (
                intermediate_size,
                hidden_size,
            )
            result[f"{prefix}.down_proj.weight"] = (
                hidden_size,
                intermediate_size,
            )
    return result


def olmoe_projected_expert_traffic(
    hidden_size: int,
    intermediate_size: int,
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
) -> dict[str, Any]:
    """Model cold expert/router bytes relative to all-expert dense Q4."""

    expert_parameters = 3 * hidden_size * intermediate_size
    all_expert_parameters = num_layers * num_experts * expert_parameters
    active_expert_parameters = num_layers * top_k * expert_parameters
    all_expert_q4_bytes = (all_expert_parameters + 1) // 2
    active_expert_q4_bytes = (active_expert_parameters + 1) // 2
    router_bf16_bytes = num_layers * num_experts * hidden_size * 2
    selected_plus_router = active_expert_q4_bytes + router_bf16_bytes
    fraction = selected_plus_router / all_expert_q4_bytes
    return {
        "per_expert_parameters": expert_parameters,
        "all_expert_parameters": all_expert_parameters,
        "active_expert_parameters": active_expert_parameters,
        "active_expert_fraction": top_k / num_experts,
        "all_expert_dense_q4_bytes": all_expert_q4_bytes,
        "selected_expert_dense_q4_bytes": active_expert_q4_bytes,
        "router_bf16_bytes": router_bf16_bytes,
        "selected_experts_plus_router_bytes": selected_plus_router,
        "fraction_of_all_expert_dense_q4": fraction,
        "passes_45_percent_projection": fraction <= 0.45,
        "measured_hardware_traffic": False,
        "scope": (
            "expert weights and BF16 router matrices only; attention, norms, "
            "embeddings, activations, cache lines, and runtime overhead excluded"
        ),
    }


@dataclass(frozen=True)
class OLMoESourceAudit:
    source: str
    source_kind: str
    config_path: str
    index_path: str | None
    requested_revision: str | None
    resolved_revision: str | None
    config_sha256: str
    index_sha256: str | None
    adapter: str
    model_type: str
    architecture: str
    dimensions: dict[str, int | None]
    checks: dict[str, bool]
    tensor_contract: dict[str, Any]
    projected_traffic: dict[str, Any] | None
    provenance: dict[str, Any]
    capabilities: dict[str, bool]
    decision: str
    combined_gate_status: str
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["caveats"] = list(self.caveats)
        return result


def audit_olmoe_source(
    model: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    verify_remote_shapes: bool = False,
) -> OLMoESourceAudit:
    """Audit OLMoE metadata, optionally reading bounded remote shape headers."""

    config_path, index_path, repo_id, resolved_revision = _resolve_metadata(
        model, revision=revision, cache_dir=cache_dir
    )
    config = _read_object(config_path, "model config")
    architectures = config.get("architectures")
    architecture = (
        str(architectures[0])
        if isinstance(architectures, list) and architectures
        else ""
    )
    keys = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "num_experts",
        "num_experts_per_tok",
        "vocab_size",
    )
    dimensions = {key: _positive_int(config, key) for key in keys}
    h = dimensions["hidden_size"]
    heads = dimensions["num_attention_heads"]
    kv_heads = dimensions["num_key_value_heads"]
    experts = dimensions["num_experts"]
    top_k = dimensions["num_experts_per_tok"]
    checks = {
        "model_type_olmoe": config.get("model_type") == "olmoe",
        "architecture_olmoe_causal_lm": architecture == "OlmoeForCausalLM",
        "activation_silu": config.get("hidden_act") in {"silu", "swish"},
        "attention_bias_disabled": config.get("attention_bias", False) is False,
        "router_probabilities_not_renormalized": config.get("norm_topk_prob") is False,
        "positive_dimensions": all(value is not None for value in dimensions.values()),
        "valid_attention_partition": bool(
            h and heads and kv_heads and h % heads == 0 and heads % kv_heads == 0
        ),
        "valid_expert_selection": bool(experts and top_k and top_k <= experts),
    }
    config_valid = all(checks.values())

    required: dict[str, tuple[int, ...]] = {}
    if config_valid:
        required = required_olmoe_tensor_shapes(
            hidden_size=int(h),
            intermediate_size=int(dimensions["intermediate_size"]),
            num_layers=int(dimensions["num_hidden_layers"]),
            num_heads=int(heads),
            num_key_value_heads=int(kv_heads),
            num_experts=int(experts),
            vocab_size=int(dimensions["vocab_size"]),
        )
    indexed_names: set[str] = set()
    indexed_shards: set[str] = set()
    if index_path is not None:
        index = _read_object(index_path, "safetensors index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not all(
            isinstance(name, str) and isinstance(shard, str)
            for name, shard in weight_map.items()
        ):
            raise OLMoEValidationError("safetensors index has an invalid weight_map")
        indexed_names = set(weight_map)
        indexed_shards = set(weight_map.values())

    inventory: tuple[TensorInfo, ...] = ()
    inventory_error: str | None = None
    if repo_id is None:
        try:
            inventory = local_tensor_inventory(config_path.parent)
        except ModelValidationError as exc:
            inventory_error = str(exc)
    elif verify_remote_shapes:
        if not resolved_revision:
            raise OLMoEValidationError(
                "remote shape audit requires a resolved immutable revision"
            )
        inventory = _remote_safetensors_inventory(
            repo_id,
            resolved_revision,
            indexed_shards,
        )
    actual_names = {item.name for item in inventory}
    contract_names = actual_names or indexed_names
    missing = sorted(set(required) - contract_names) if required else []
    unexpected = sorted(contract_names - set(required)) if required else sorted(contract_names)
    shape_errors = []
    if inventory and required:
        for item in inventory:
            expected = required.get(item.name)
            if expected is not None and item.shape != expected:
                shape_errors.append(
                    {"name": item.name, "actual": list(item.shape), "expected": list(expected)}
                )
    names_complete = bool(required) and not missing
    shapes_validated = bool(inventory) and names_complete and not shape_errors
    contract_valid = (
        config_valid and names_complete and not unexpected and not shape_errors
    )
    traffic = None
    if config_valid:
        traffic = olmoe_projected_expert_traffic(
            int(h),
            int(dimensions["intermediate_size"]),
            num_layers=int(dimensions["num_hidden_layers"]),
            num_experts=int(experts),
            top_k=int(top_k),
        )
    if not contract_valid:
        decision = "reject_incompatible_olmoe_contract"
    elif shapes_validated:
        decision = "proceed_to_router_trace"
    else:
        decision = "proceed_to_exact_weight_shape_audit"

    official = repo_id == OFFICIAL_OLMOE_REPO
    return OLMoESourceAudit(
        source=str(model),
        source_kind="huggingface_hub" if repo_id is not None else "local",
        config_path=str(config_path),
        index_path=str(index_path) if index_path is not None else None,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        config_sha256=sha256_file(config_path),
        index_sha256=sha256_file(index_path) if index_path is not None else None,
        adapter="olmoe_sparse_expert_v1",
        model_type=str(config.get("model_type", "")),
        architecture=architecture,
        dimensions=dimensions,
        checks=checks,
        tensor_contract={
            "required_tensor_count": len(required),
            "indexed_tensor_count": len(indexed_names),
            "local_tensor_count": len(inventory) if repo_id is None else 0,
            "remote_header_tensor_count": len(inventory) if repo_id is not None else 0,
            "required_names_complete": names_complete,
            "unexpected_tensor_names": unexpected,
            "missing_tensor_names": missing,
            "exact_shapes_validated": shapes_validated,
            "shape_inventory_source": (
                "local_weights"
                if repo_id is None and inventory
                else "remote_bounded_safetensors_headers"
                if inventory
                else "not_validated"
            ),
            "shape_errors": shape_errors,
            "inventory_error": inventory_error,
        },
        projected_traffic=traffic,
        provenance={
            "status": (
                "resolved_official_repository"
                if official and resolved_revision
                else "format_only_unverified"
            ),
            "repository": repo_id,
            "resolved_revision": resolved_revision,
            "evidence": (
                [OFFICIAL_OLMOE_MODEL_CARD, OFFICIAL_OLMOE_PAPER] if official else []
            ),
            "policy": (
                "repository identity and resolved revision establish source "
                "provenance; quality and traffic gates require separate experiments"
            ),
        },
        capabilities={
            "metadata_tensor_inventory": contract_valid,
            "exact_shape_validation": shapes_validated,
            "expert_level_addressability": contract_valid,
            "existing_dense_llama_compiler": False,
            "router_trace_adapter": False,
            "compiled_sparse_expert_runtime": False,
        },
        decision=decision,
        combined_gate_status="not_evaluated_source_audit_only",
        caveats=(
            "The projected byte fraction is neither measured runtime traffic nor causal quality.",
            "OLMoE requires a new expert/router trace and compiler path.",
            (
                "Remote shape verification reads bounded safetensors header "
                "ranges only; it does not download checkpoint payloads."
            ),
            "A top-k MoE still executes attention and selected expert arithmetic.",
        ),
    )
