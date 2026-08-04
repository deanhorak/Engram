"""Structural adapter for Qwen3 dense decoder checkpoints.

Qwen3's feed-forward block is the same bias-free SiLU/SwiGLU contract used by
Engram's generic teacher tracer::

    down_proj(silu(gate_proj(x)) * up_proj(x))

This module intentionally stops at a source audit.  It does not claim that a
Qwen3 checkpoint can be emitted by the BitNet-native compiler: attention,
embeddings, and dense projection serialization remain a separate compiler
track.  A successful audit is therefore a useful, falsifiable boundary before
we download or distill a full checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from engram.models.inspection import ModelInspection, ModelValidationError, inspect_model


QWEN3_MODEL_TYPE = "qwen3"
QWEN3_ARCHITECTURE = "Qwen3ForCausalLM"


class Qwen3ValidationError(ModelValidationError):
    """Raised when a checkpoint is not within the supported Qwen3 contract."""


@dataclass(frozen=True)
class Qwen3SourceAudit:
    model_path: str
    source_hash: str
    model_type: str
    architecture: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    tie_word_embeddings: bool
    tensor_count: int
    weight_bytes: int
    capabilities: dict[str, bool]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_config(model_path: Path) -> dict[str, Any]:
    try:
        value = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Qwen3ValidationError(f"invalid Qwen3 config: {exc}") from exc
    if not isinstance(value, dict):
        raise Qwen3ValidationError("Qwen3 config must be a JSON object")
    return value


def _positive_config_int(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Qwen3ValidationError(f"Qwen3 config field {name!r} must be a positive integer")
    return value


def _rope_theta(config: dict[str, Any]) -> float:
    # Transformers 4.53 checkpoints used the top-level field; newer configs
    # may normalize it into rope_parameters.
    value = config.get("rope_theta")
    if value is None:
        parameters = config.get("rope_parameters")
        if isinstance(parameters, dict):
            value = parameters.get("rope_theta")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise Qwen3ValidationError(
            "Qwen3 config must provide a positive rope_theta or rope_parameters.rope_theta"
        )
    return float(value)


def qwen3_mlp_tensor_names(layer: int) -> tuple[str, str, str]:
    """Return the canonical gate/up/down tensor names for one Qwen3 layer."""

    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ValueError("layer must be a non-negative integer")
    prefix = f"model.layers.{layer}.mlp"
    return (
        f"{prefix}.gate_proj.weight",
        f"{prefix}.up_proj.weight",
        f"{prefix}.down_proj.weight",
    )


def audit_qwen3_source(
    model: str | Path,
    *,
    hash_weights: bool = True,
) -> Qwen3SourceAudit:
    """Validate a local Qwen3 checkpoint against the generic trace contract.

    ``inspect_model`` performs the complete tensor inventory and shape checks;
    this function adds Qwen-specific configuration checks and makes the
    unsupported native-compiler boundary explicit in the returned report.
    """

    try:
        inspection: ModelInspection = inspect_model(model, hash_weights=hash_weights)
    except ModelValidationError as exc:
        raise Qwen3ValidationError(str(exc)) from exc
    if inspection.model_type != QWEN3_MODEL_TYPE:
        raise Qwen3ValidationError(
            f"expected model_type {QWEN3_MODEL_TYPE!r}, got {inspection.model_type!r}"
        )
    model_path = Path(inspection.model_path)
    config = _read_config(model_path)
    architecture = inspection.architecture
    if architecture != QWEN3_ARCHITECTURE:
        raise Qwen3ValidationError(
            f"unsupported Qwen3 architecture {architecture!r}; expected {QWEN3_ARCHITECTURE!r}"
        )
    hidden_act = config.get("hidden_act", "silu")
    if hidden_act not in {"silu", "swish"}:
        raise Qwen3ValidationError(f"Qwen3 hidden_act must be SiLU, got {hidden_act!r}")
    if config.get("mlp_bias", False) or config.get("attention_bias", False):
        raise Qwen3ValidationError(
            "the current Qwen3 trace contract requires bias-free MLP and attention projections"
        )
    num_key_value_heads = _positive_config_int(config, "num_key_value_heads")
    if num_key_value_heads > inspection.num_attention_heads:
        raise Qwen3ValidationError(
            "num_key_value_heads cannot exceed num_attention_heads"
        )
    head_dim = _positive_config_int(config, "head_dim")
    max_positions = _positive_config_int(config, "max_position_embeddings")
    tie_word_embeddings = bool(config.get("tie_word_embeddings", False))
    warnings = list(inspection.warnings)
    warnings.append(
        "Qwen3 is structurally traceable; native BitNet package compilation is not implemented for dense Qwen3"
    )
    return Qwen3SourceAudit(
        model_path=inspection.model_path,
        source_hash=inspection.source_hash,
        model_type=inspection.model_type,
        architecture=architecture,
        hidden_size=inspection.hidden_size,
        intermediate_size=inspection.intermediate_size,
        num_hidden_layers=inspection.num_hidden_layers,
        num_attention_heads=inspection.num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        vocab_size=inspection.vocab_size,
        max_position_embeddings=max_positions,
        rope_theta=_rope_theta(config),
        tie_word_embeddings=tie_word_embeddings,
        tensor_count=inspection.tensor_count,
        weight_bytes=inspection.weight_bytes,
        capabilities={
            "exact_swiglu_decomposition": True,
            "generic_hf_teacher_trace": True,
            "native_bitnet_compilation": False,
            "cpu_teacher_execution": True,
        },
        warnings=tuple(warnings),
    )


__all__ = [
    "QWEN3_ARCHITECTURE",
    "QWEN3_MODEL_TYPE",
    "Qwen3SourceAudit",
    "Qwen3ValidationError",
    "audit_qwen3_source",
    "qwen3_mlp_tensor_names",
]
