"""Fail-closed support for the native offline BitNet source format.

This module deliberately does not add ``bitnet`` to the generic Engram model
inspector.  The existing semantic compiler assumes a dense SiLU/SwiGLU source;
native BitNet instead uses packed ternary projections, ReLU-squared gating,
activation quantization, and an intermediate RMS normalization.

The adapter here has two narrow jobs:

* audit a BitNet configuration without downloading model weights; and
* losslessly repack an already-ternary MLP from four trits per byte to a
  record-addressable five-trits-per-byte artifact.

The repack is a separate low-bit-native research track.  It is not evidence
that an arbitrary dense Llama checkpoint can be converted at the same quality.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.utils import atomic_json, sha256_file


OFFICIAL_NATIVE_BITNET_REPO = "microsoft/bitnet-b1.58-2B-4T"
OFFICIAL_NATIVE_BITNET_REVISION = "04c3b9ad9361b824064a1f25ea60a8be9599b127"
OFFICIAL_NATIVE_BITNET_WEIGHT_SHA256 = (
    "8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b"
)
OFFICIAL_NATIVE_BITNET_MODEL_CARD = (
    "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T"
)
OFFICIAL_NATIVE_BITNET_TECHNICAL_REPORT = "https://arxiv.org/abs/2504.12285"

_MAGIC = b"ENGBNP11"
_LAYER_MAGIC = b"ENGBNPL1"
_VERSION = 1
_HEADER_BYTES = 64
_LAYER_HEADER_BYTES = 64
_HEADER = struct.Struct("<8s8If")
_DIRECTORY_ENTRY = struct.Struct("<IIQQII")
_LAYER_HEADER = struct.Struct("<8s10I")
_BF16_BYTES = 2
_PROJECTION_SCALE_COUNT = 3
_HF_TRITS_PER_BYTE = 4
_ENGRAM_TRITS_PER_BYTE = 5
_MAX_MODEL_DIMENSION = 1 << 20
_MAX_LAYER_COUNT = 1 << 12
_MAX_CACHE_LINE_BYTES = 1 << 12
_UINT32_MAX = (1 << 32) - 1


class NativeBitNetValidationError(ValueError):
    """Raised when a source or artifact violates the narrow native contract."""


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NativeBitNetValidationError(f"{name} must be a positive integer")
    return value


def _align(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("alignment operands must be non-negative/positive")
    return ((value + alignment - 1) // alignment) * alignment


def _resolved_snapshot_revision(path: Path) -> str | None:
    """Extract a Hub snapshot revision without trusting arbitrary path text."""

    try:
        if path.parent.parent.name == "snapshots":
            revision = path.parent.name
            if len(revision) == 40 and all(
                character in "0123456789abcdef" for character in revision
            ):
                return revision
    except IndexError:
        pass
    return None


def _resolve_config_only(
    model: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
) -> tuple[Path, str | None, str | None]:
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        config_path = candidate / "config.json"
        if not config_path.is_file():
            raise NativeBitNetValidationError(
                f"missing model config: {config_path.resolve()}"
            )
        return config_path.resolve(), None, None
    if candidate.exists():
        raise NativeBitNetValidationError(
            f"model path is not a directory: {candidate.resolve()}"
        )
    model_id = str(model)
    if (
        Path(model_id).is_absolute()
        or model_id.startswith(("./", "../", "~"))
        or "/" not in model_id
    ):
        raise NativeBitNetValidationError(
            f"model path is not a directory or Hub repository ID: {model_id!r}"
        )
    selected_revision = revision
    if model_id == OFFICIAL_NATIVE_BITNET_REPO and selected_revision is None:
        selected_revision = OFFICIAL_NATIVE_BITNET_REVISION
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise NativeBitNetValidationError(
            "install engram-lm[conversion] to audit a Hugging Face source"
        ) from exc
    try:
        downloaded = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            revision=selected_revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    except Exception as exc:
        raise NativeBitNetValidationError(
            f"could not download config for {model_id!r}: {exc}"
        ) from exc
    path = Path(downloaded).resolve()
    return path, model_id, _resolved_snapshot_revision(path) or selected_revision


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeBitNetValidationError(
            f"invalid model config {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise NativeBitNetValidationError("model config must be a JSON object")
    return payload


def _cache_lines_for_range(
    offset: int,
    length: int,
    cache_line_bytes: int,
) -> int:
    phase = offset % cache_line_bytes
    return (phase + length + cache_line_bytes - 1) // cache_line_bytes


def native_bitnet_repack_traffic(
    hidden_size: int,
    intermediate_size: int,
    *,
    layer_count: int,
    cache_line_bytes: int = 64,
) -> dict[str, Any]:
    """Return complete dual-baseline traffic for the exact native repack."""

    hidden_size = _positive_integer(hidden_size, "hidden_size")
    intermediate_size = _positive_integer(intermediate_size, "intermediate_size")
    layer_count = _positive_integer(layer_count, "layer_count")
    cache_line_bytes = _positive_integer(cache_line_bytes, "cache_line_bytes")
    if hidden_size > _MAX_MODEL_DIMENSION:
        raise NativeBitNetValidationError("hidden_size exceeds the supported maximum")
    if intermediate_size > _MAX_MODEL_DIMENSION:
        raise NativeBitNetValidationError(
            "intermediate_size exceeds the supported maximum"
        )
    if layer_count > _MAX_LAYER_COUNT:
        raise NativeBitNetValidationError("layer_count exceeds the supported maximum")
    if cache_line_bytes > _MAX_CACHE_LINE_BYTES:
        raise NativeBitNetValidationError(
            "cache_line_bytes exceeds the supported maximum"
        )
    if cache_line_bytes < 64 or cache_line_bytes % 64:
        raise NativeBitNetValidationError(
            "cache_line_bytes must be a positive multiple of 64"
        )
    if hidden_size % _HF_TRITS_PER_BYTE:
        raise NativeBitNetValidationError(
            "native offline BitNet requires hidden_size divisible by four"
        )
    if intermediate_size % _HF_TRITS_PER_BYTE:
        raise NativeBitNetValidationError(
            "native offline BitNet requires intermediate_size divisible by four"
        )

    packed_key_bytes = (
        hidden_size + _ENGRAM_TRITS_PER_BYTE - 1
    ) // _ENGRAM_TRITS_PER_BYTE
    record_payload_bytes = 3 * packed_key_bytes + _BF16_BYTES
    projection_scale_bytes_per_layer = _PROJECTION_SCALE_COUNT * _BF16_BYTES
    record_payload_bytes_per_layer = intermediate_size * record_payload_bytes
    layer_metadata_bytes = _LAYER_HEADER_BYTES + projection_scale_bytes_per_layer
    gate_stream_offset = _align(layer_metadata_bytes, cache_line_bytes)
    gate_stream_bytes = intermediate_size * packed_key_bytes
    up_stream_offset = _align(
        gate_stream_offset + gate_stream_bytes,
        cache_line_bytes,
    )
    up_stream_bytes = gate_stream_bytes
    norm_stream_offset = _align(
        up_stream_offset + up_stream_bytes,
        cache_line_bytes,
    )
    norm_stream_bytes = intermediate_size * _BF16_BYTES
    down_stream_offset = _align(
        norm_stream_offset + norm_stream_bytes,
        cache_line_bytes,
    )
    down_stream_bytes = gate_stream_bytes
    layer_payload_bytes = down_stream_offset + down_stream_bytes
    if layer_payload_bytes > _UINT32_MAX:
        raise NativeBitNetValidationError(
            "layer payload is not representable in the artifact header"
        )
    layer_block_bytes = _align(layer_payload_bytes, cache_line_bytes)
    layer_internal_padding_bytes = (
        gate_stream_offset
        - layer_metadata_bytes
        + up_stream_offset
        - (gate_stream_offset + gate_stream_bytes)
        + norm_stream_offset
        - (up_stream_offset + up_stream_bytes)
        + down_stream_offset
        - (norm_stream_offset + norm_stream_bytes)
    )
    header_block_bytes = _align(_HEADER_BYTES, cache_line_bytes)
    directory_payload_bytes = layer_count * _DIRECTORY_ENTRY.size
    directory_block_bytes = _align(directory_payload_bytes, cache_line_bytes)
    serialized_artifact_bytes = (
        header_block_bytes + directory_block_bytes + layer_count * layer_block_bytes
    )

    elements_per_projection = hidden_size * intermediate_size
    source_projection_bytes = 3 * elements_per_projection // _HF_TRITS_PER_BYTE
    source_norm_bytes = intermediate_size * _BF16_BYTES
    source_native_bytes_per_layer = (
        source_projection_bytes + source_norm_bytes + projection_scale_bytes_per_layer
    )
    source_native_mlp_bytes = layer_count * source_native_bytes_per_layer
    dense_q4_source_mlp_bytes = (layer_count * 3 * elements_per_projection + 1) // 2
    fraction_of_dense_q4 = serialized_artifact_bytes / dense_q4_source_mlp_bytes
    fraction_of_native_source = serialized_artifact_bytes / source_native_mlp_bytes
    record_phase_period = math.lcm(
        cache_line_bytes // math.gcd(packed_key_bytes, cache_line_bytes),
        cache_line_bytes // math.gcd(_BF16_BYTES, cache_line_bytes),
    )
    cycle_record_bytes = tuple(
        cache_line_bytes
        * (
            _cache_lines_for_range(
                gate_stream_offset + record * packed_key_bytes,
                packed_key_bytes,
                cache_line_bytes,
            )
            + _cache_lines_for_range(
                up_stream_offset + record * packed_key_bytes,
                packed_key_bytes,
                cache_line_bytes,
            )
            + _cache_lines_for_range(
                norm_stream_offset + record * _BF16_BYTES,
                _BF16_BYTES,
                cache_line_bytes,
            )
            + _cache_lines_for_range(
                down_stream_offset + record * packed_key_bytes,
                packed_key_bytes,
                cache_line_bytes,
            )
        )
        for record in range(record_phase_period)
    )
    complete_cycles, tail_records = divmod(
        intermediate_size,
        record_phase_period,
    )
    independent_records_bytes_per_layer = complete_cycles * sum(
        cycle_record_bytes
    ) + sum(cycle_record_bytes[:tail_records])
    observed_cycle = cycle_record_bytes[: min(intermediate_size, record_phase_period)]
    worst_case_independent_record_bytes = max(observed_cycle)
    worst_case_independent_records_bytes = (
        layer_count * independent_records_bytes_per_layer
        + header_block_bytes
        + directory_block_bytes
        + layer_count * gate_stream_offset
    )
    worst_case_fraction = (
        worst_case_independent_records_bytes / dense_q4_source_mlp_bytes
    )
    gate_bytes = math.floor(0.45 * dense_q4_source_mlp_bytes)

    return {
        "layout": "native_bitnet_phase_base3_v1",
        "packing": "five base-3 ternary digits per byte",
        "record_layout": (
            "logical records addressed across cache-aligned gate, up, "
            "ffn_sub_norm-gain, and transposed-down phase streams"
        ),
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "layer_count": layer_count,
        "cache_line_bytes": cache_line_bytes,
        "elements_per_projection": elements_per_projection,
        "packed_bytes_per_record_vector": packed_key_bytes,
        "record_payload_bytes": record_payload_bytes,
        "independently_scattered_record_max_cache_lines": (
            worst_case_independent_record_bytes // cache_line_bytes
        ),
        "independently_scattered_record_max_bytes": (
            worst_case_independent_record_bytes
        ),
        "projection_scale_bytes_per_layer": (projection_scale_bytes_per_layer),
        "record_payload_bytes_per_layer": record_payload_bytes_per_layer,
        "layer_header_bytes": _LAYER_HEADER_BYTES,
        "layer_metadata_aligned_bytes": gate_stream_offset,
        "layer_internal_padding_bytes": layer_internal_padding_bytes,
        "gate_stream_offset": gate_stream_offset,
        "gate_stream_bytes": gate_stream_bytes,
        "up_stream_offset": up_stream_offset,
        "up_stream_bytes": up_stream_bytes,
        "norm_stream_offset": norm_stream_offset,
        "norm_stream_bytes": norm_stream_bytes,
        "down_stream_offset": down_stream_offset,
        "down_stream_bytes": down_stream_bytes,
        "layer_payload_bytes": layer_payload_bytes,
        "layer_cache_padding_bytes": layer_block_bytes - layer_payload_bytes,
        "layer_block_bytes": layer_block_bytes,
        "header_cache_aligned_bytes": header_block_bytes,
        "directory_entry_bytes": _DIRECTORY_ENTRY.size,
        "directory_payload_bytes": directory_payload_bytes,
        "directory_cache_aligned_bytes": directory_block_bytes,
        "serialized_artifact_bytes": serialized_artifact_bytes,
        "total_cold_bytes": serialized_artifact_bytes,
        "traffic_numerator_bytes": serialized_artifact_bytes,
        "hf_native_two_bit_mlp_bytes": source_native_mlp_bytes,
        "dense_q4_source_mlp_bytes": dense_q4_source_mlp_bytes,
        "fraction_of_hf_native_two_bit": fraction_of_native_source,
        "fraction_of_dense_q4": fraction_of_dense_q4,
        "serialized_layout_passes_45_percent_gate": fraction_of_dense_q4 <= 0.45,
        "modelled_full_phase_schedule_bytes": serialized_artifact_bytes,
        "modelled_full_phase_schedule_fraction_of_dense_q4": (fraction_of_dense_q4),
        "modelled_full_phase_schedule_passes_45_percent_gate": (
            fraction_of_dense_q4 <= 0.45
        ),
        "modelled_full_phase_schedule": (
            "stream gate and up, then contiguous normalization gains, then "
            "transposed down; each serialized cache line is charged once"
        ),
        "measured_hardware_traffic": False,
        "headroom_bytes_to_45_percent": (gate_bytes - serialized_artifact_bytes),
        "independently_scattered_records_bytes": (worst_case_independent_records_bytes),
        "independently_scattered_records_fraction_of_dense_q4": (worst_case_fraction),
        "independently_scattered_records_passes_45_percent_gate": (
            worst_case_fraction <= 0.45
        ),
        "accounting_policy": (
            "the primary numerator is the exact cache-aligned serialized MLP "
            "artifact, including all ternary streams, BF16 projection scales, "
            "BF16 intermediate-normalization gains, headers, directory, and "
            "padding. Phase-segregated streams permit an exact dense schedule "
            "that charges every serialized cache line once; this remains a "
            "modelled cold-byte result until the direct kernel measures "
            "hardware traffic. The frozen denominator is ideal code-only "
            "dense-Q4 gate/up/down weights. The actual Hugging Face two-bit "
            "MLP payload is reported separately and is never substituted for "
            "the frozen denominator."
        ),
    }


@dataclass(frozen=True)
class NativeBitNetSourceAudit:
    source: str
    source_kind: str
    config_path: str
    requested_revision: str | None
    resolved_revision: str | None
    config_sha256: str
    adapter: str
    model_type: str
    architecture: str
    hidden_act: str
    hidden_size: int | None
    intermediate_size: int | None
    num_hidden_layers: int | None
    rms_norm_eps: float | None
    quantization_config: dict[str, Any]
    checks: dict[str, bool]
    capabilities: dict[str, bool]
    provenance: dict[str, Any]
    projected_traffic: dict[str, Any] | None
    decision: str
    combined_gate_status: str
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["caveats"] = list(self.caveats)
        return result


def audit_native_bitnet_source(
    model: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> NativeBitNetSourceAudit:
    """Audit only configuration/provenance; never download model weights."""

    config_path, repo_id, resolved_revision = _resolve_config_only(
        model,
        revision=revision,
        cache_dir=cache_dir,
    )
    config = _read_json(config_path)
    architectures = config.get("architectures")
    architecture = (
        str(architectures[0])
        if isinstance(architectures, list) and architectures
        else ""
    )
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        quantization = {}

    dimensions: dict[str, int | None] = {}
    for field in ("hidden_size", "intermediate_size", "num_hidden_layers"):
        value = config.get(field)
        dimensions[field] = (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
            else None
        )
    epsilon = config.get("rms_norm_eps")
    rms_norm_eps = (
        float(epsilon)
        if isinstance(epsilon, (int, float))
        and not isinstance(epsilon, bool)
        and math.isfinite(float(epsilon))
        and float(epsilon) > 0
        else None
    )

    checks = {
        "model_type_bitnet": config.get("model_type") == "bitnet",
        "architecture_bitnet_causal_lm": (architecture == "BitNetForCausalLM"),
        "activation_relu2": config.get("hidden_act") == "relu2",
        "quant_method_bitnet": (quantization.get("quant_method") == "bitnet"),
        "offline_quantization": (quantization.get("quantization_mode") == "offline"),
        "autobitlinear_storage": (quantization.get("linear_class") == "autobitlinear"),
        "positive_dimensions": all(value is not None for value in dimensions.values()),
        "rms_norm_epsilon_present": rms_norm_eps is not None,
        "packed_dimensions_divisible_by_four": (
            dimensions["hidden_size"] is not None
            and dimensions["intermediate_size"] is not None
            and dimensions["hidden_size"] % _HF_TRITS_PER_BYTE == 0
            and dimensions["intermediate_size"] % _HF_TRITS_PER_BYTE == 0
        ),
    }
    format_valid = all(checks.values())
    is_pinned_official = (
        repo_id == OFFICIAL_NATIVE_BITNET_REPO
        and resolved_revision == OFFICIAL_NATIVE_BITNET_REVISION
    )
    provenance = {
        "status": (
            "pinned_official_attestation"
            if is_pinned_official
            else "format_only_unverified"
        ),
        "native_training_claim_accepted": is_pinned_official,
        "repository": repo_id,
        "pinned_revision": (
            OFFICIAL_NATIVE_BITNET_REVISION if is_pinned_official else None
        ),
        "expected_model_weight_sha256": (
            OFFICIAL_NATIVE_BITNET_WEIGHT_SHA256 if is_pinned_official else None
        ),
        "evidence": (
            [
                OFFICIAL_NATIVE_BITNET_MODEL_CARD,
                OFFICIAL_NATIVE_BITNET_TECHNICAL_REPORT,
            ]
            if is_pinned_official
            else []
        ),
        "policy": (
            "quantization metadata alone does not prove that a model was "
            "trained natively at low bit width"
        ),
    }
    traffic = None
    if format_valid:
        assert dimensions["hidden_size"] is not None
        assert dimensions["intermediate_size"] is not None
        assert dimensions["num_hidden_layers"] is not None
        traffic = native_bitnet_repack_traffic(
            dimensions["hidden_size"],
            dimensions["intermediate_size"],
            layer_count=dimensions["num_hidden_layers"],
        )
    projected_pass = bool(
        traffic and traffic["modelled_full_phase_schedule_passes_45_percent_gate"]
    )
    decision = (
        "proceed_to_exact_weight_repack"
        if format_valid and is_pinned_official and projected_pass
        else "reject_or_require_explicit_provenance"
    )
    return NativeBitNetSourceAudit(
        source=str(model),
        source_kind="huggingface_hub" if repo_id is not None else "local",
        config_path=str(config_path),
        requested_revision=revision,
        resolved_revision=resolved_revision,
        config_sha256=sha256_file(config_path),
        adapter="bitnet_offline_autobitlinear_v1",
        model_type=str(config.get("model_type", "")),
        architecture=architecture,
        hidden_act=str(config.get("hidden_act", "")),
        hidden_size=dimensions["hidden_size"],
        intermediate_size=dimensions["intermediate_size"],
        num_hidden_layers=dimensions["num_hidden_layers"],
        rms_norm_eps=rms_norm_eps,
        quantization_config=dict(quantization),
        checks=checks,
        capabilities={
            "metadata_inventory": format_valid,
            "exact_native_repack": format_valid,
            "hf_boundary_trace": False,
            "exact_swiglu_decomposition": False,
            "existing_semantic_compiler": False,
            "dense_llama_conversion": False,
        },
        provenance=provenance,
        projected_traffic=traffic,
        decision=decision,
        combined_gate_status="not_evaluated_metadata_only",
        caveats=(
            "The projected traffic pass is not a causal-quality or runtime pass.",
            "Native BitNet is a separate source track, not a dense-Llama conversion.",
            (
                "ReLU-squared gating and ffn_sub_norm couple record magnitudes; "
                "the current SiLU/SwiGLU semantic selector is incompatible."
            ),
            (
                "Standard Transformers execution does not demonstrate the "
                "specialized CPU-kernel latency benefit."
            ),
            (
                "The generic teacher-trace pipeline does not yet support this "
                "native BitNet adapter."
            ),
        ),
    )


def pack_hf_bitnet_codes(codes: ArrayLike) -> NDArray[np.uint8]:
    """Pack rows in the Hugging Face four-trits-per-byte orientation."""

    values = np.asarray(codes)
    if values.ndim != 2:
        raise NativeBitNetValidationError("ternary codes must be a matrix")
    if values.shape[0] % _HF_TRITS_PER_BYTE:
        raise NativeBitNetValidationError(
            "Hugging Face packed output width must be divisible by four"
        )
    if not np.all((values == -1) | (values == 0) | (values == 1)):
        raise NativeBitNetValidationError(
            "ternary codes must contain only -1, 0, and +1"
        )
    row_count = values.shape[0] // _HF_TRITS_PER_BYTE
    digits = values.astype(np.uint8, copy=False) + 1
    packed = np.zeros((row_count, values.shape[1]), dtype=np.uint8)
    for index in range(_HF_TRITS_PER_BYTE):
        start = index * row_count
        packed |= digits[start : start + row_count] << (2 * index)
    return np.ascontiguousarray(packed)


def unpack_hf_bitnet_codes(
    packed: ArrayLike,
    *,
    out_features: int,
) -> NDArray[np.int8]:
    """Decode and strictly validate the official two-bit checkpoint layout."""

    out_features = _positive_integer(out_features, "out_features")
    source = np.asarray(packed)
    if source.dtype != np.uint8 or source.ndim != 2:
        raise NativeBitNetValidationError(
            "packed BitNet weights must be a rank-2 uint8 matrix"
        )
    if out_features % _HF_TRITS_PER_BYTE:
        raise NativeBitNetValidationError(
            "packed BitNet out_features must be divisible by four"
        )
    expected_rows = out_features // _HF_TRITS_PER_BYTE
    if source.shape[0] != expected_rows:
        raise NativeBitNetValidationError(
            f"packed BitNet row count {source.shape[0]} does not match "
            f"out_features={out_features}"
        )
    result = np.empty((out_features, source.shape[1]), dtype=np.int8)
    for index in range(_HF_TRITS_PER_BYTE):
        digits = (source >> (2 * index)) & 0b11
        if np.any(digits == 0b11):
            raise NativeBitNetValidationError(
                "packed BitNet weight contains invalid two-bit code 3"
            )
        start = index * expected_rows
        result[start : start + expected_rows] = digits.astype(np.int8) - 1
    return np.ascontiguousarray(result)


def pack_base3_rows(codes: ArrayLike) -> NDArray[np.uint8]:
    """Pack each semantic record independently using five base-3 digits."""

    values = np.asarray(codes)
    if values.ndim != 2:
        raise NativeBitNetValidationError("ternary records must be a matrix")
    if not np.all((values == -1) | (values == 0) | (values == 1)):
        raise NativeBitNetValidationError(
            "ternary records must contain only -1, 0, and +1"
        )
    rows, width = values.shape
    packed_width = (width + _ENGRAM_TRITS_PER_BYTE - 1) // (_ENGRAM_TRITS_PER_BYTE)
    padded = np.zeros(
        (rows, packed_width * _ENGRAM_TRITS_PER_BYTE),
        dtype=np.uint8,
    )
    padded[:, :width] = values.astype(np.int8, copy=False) + 1
    groups = padded.reshape(rows, packed_width, _ENGRAM_TRITS_PER_BYTE).astype(
        np.uint16
    )
    packed = (
        groups[:, :, 0]
        + 3 * groups[:, :, 1]
        + 9 * groups[:, :, 2]
        + 27 * groups[:, :, 3]
        + 81 * groups[:, :, 4]
    )
    return np.ascontiguousarray(packed.astype(np.uint8))


def unpack_base3_rows(
    packed: ArrayLike,
    *,
    logical_width: int,
) -> NDArray[np.int8]:
    """Decode record-local base-3 streams and reject noncanonical tails."""

    logical_width = _positive_integer(logical_width, "logical_width")
    source = np.asarray(packed)
    if source.dtype != np.uint8 or source.ndim != 2:
        raise NativeBitNetValidationError(
            "packed base-3 records must be a rank-2 uint8 matrix"
        )
    expected_width = (
        logical_width + _ENGRAM_TRITS_PER_BYTE - 1
    ) // _ENGRAM_TRITS_PER_BYTE
    if source.shape[1] != expected_width:
        raise NativeBitNetValidationError("packed base-3 record width is inconsistent")
    if np.any(source > 242):
        raise NativeBitNetValidationError(
            "packed base-3 byte is outside the canonical range"
        )
    working = source.astype(np.uint16)
    digits = np.empty(
        (*source.shape, _ENGRAM_TRITS_PER_BYTE),
        dtype=np.uint8,
    )
    for column in range(_ENGRAM_TRITS_PER_BYTE):
        digits[:, :, column] = working % 3
        working //= 3
    flat = digits.reshape(source.shape[0], -1)
    if np.any(flat[:, logical_width:] != 0):
        raise NativeBitNetValidationError("packed base-3 record tail is not canonical")
    return np.ascontiguousarray(flat[:, :logical_width].astype(np.int8) - 1)


def _bf16_bits_exact(values: ArrayLike, name: str) -> NDArray[np.uint16]:
    source = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(source)):
        raise NativeBitNetValidationError(f"{name} must be finite")
    bits = np.ascontiguousarray(source).view(np.uint32)
    if np.any(bits & np.uint32(0xFFFF)):
        raise NativeBitNetValidationError(
            f"{name} is not exactly representable as BF16"
        )
    return np.ascontiguousarray((bits >> np.uint32(16)).astype("<u2"))


def _bf16_from_bits(values: ArrayLike) -> NDArray[np.float32]:
    bits = np.asarray(values, dtype="<u2")
    expanded = bits.astype(np.uint32) << np.uint32(16)
    return np.ascontiguousarray(expanded.view(np.float32))


@dataclass(frozen=True)
class NativeBitNetLayerWeights:
    gate_codes: ArrayLike
    up_codes: ArrayLike
    down_codes: ArrayLike
    gate_scale: float
    up_scale: float
    down_scale: float
    ffn_sub_norm: ArrayLike


@dataclass(frozen=True)
class PackedNativeBitNetLayer:
    gate_records: NDArray[np.uint8]
    up_records: NDArray[np.uint8]
    down_records: NDArray[np.uint8]
    scale_bits: NDArray[np.uint16]
    norm_bits: NDArray[np.uint16]


@dataclass(frozen=True)
class LoadedNativeBitNetArtifact:
    layers: tuple[PackedNativeBitNetLayer, ...]
    hidden_size: int
    intermediate_size: int
    rms_norm_eps: float
    cache_line_bytes: int
    record_payload_bytes: int
    header_block_bytes: int
    directory_block_bytes: int
    layer_offsets: tuple[int, ...]
    layer_block_bytes: tuple[int, ...]
    serialized_artifact_bytes: int
    payload_sha256: str


def _codes(values: ArrayLike, shape: tuple[int, int], name: str) -> NDArray[np.int8]:
    result = np.asarray(values)
    if result.shape != shape:
        raise NativeBitNetValidationError(
            f"{name} has shape {result.shape}, expected {shape}"
        )
    if not np.all((result == -1) | (result == 0) | (result == 1)):
        raise NativeBitNetValidationError(f"{name} must contain only -1, 0, and +1")
    return np.ascontiguousarray(result.astype(np.int8))


def _pack_layer(
    layer: NativeBitNetLayerWeights,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> PackedNativeBitNetLayer:
    gate = _codes(
        layer.gate_codes,
        (intermediate_size, hidden_size),
        "gate_codes",
    )
    up = _codes(
        layer.up_codes,
        (intermediate_size, hidden_size),
        "up_codes",
    )
    down = _codes(
        layer.down_codes,
        (hidden_size, intermediate_size),
        "down_codes",
    )
    scales = np.asarray(
        [layer.gate_scale, layer.up_scale, layer.down_scale],
        dtype=np.float32,
    )
    if np.any(scales <= 0):
        raise NativeBitNetValidationError(
            "native BitNet projection scales must be positive"
        )
    norm = np.asarray(layer.ffn_sub_norm, dtype=np.float32)
    if norm.shape != (intermediate_size,):
        raise NativeBitNetValidationError("ffn_sub_norm has an incompatible shape")
    return PackedNativeBitNetLayer(
        gate_records=pack_base3_rows(gate),
        up_records=pack_base3_rows(up),
        down_records=pack_base3_rows(down.T),
        scale_bits=_bf16_bits_exact(scales, "projection scales"),
        norm_bits=_bf16_bits_exact(norm, "ffn_sub_norm"),
    )


def _write_native_bitnet_artifact(
    path: str | Path,
    layers: Iterable[NativeBitNetLayerWeights],
    *,
    hidden_size: int,
    intermediate_size: int,
    layer_count: int,
    rms_norm_eps: float,
    cache_line_bytes: int,
) -> Path:
    traffic = native_bitnet_repack_traffic(
        hidden_size,
        intermediate_size,
        layer_count=layer_count,
        cache_line_bytes=cache_line_bytes,
    )
    if not math.isfinite(rms_norm_eps) or rms_norm_eps <= 0:
        raise NativeBitNetValidationError("rms_norm_eps must be finite and positive")
    header_block_bytes = int(traffic["header_cache_aligned_bytes"])
    directory_block_bytes = int(traffic["directory_cache_aligned_bytes"])
    layer_block_bytes = int(traffic["layer_block_bytes"])
    layer_payload_bytes = int(traffic["layer_payload_bytes"])
    record_payload_bytes = int(traffic["record_payload_bytes"])
    packed_width = int(traffic["packed_bytes_per_record_vector"])
    gate_stream_offset = int(traffic["gate_stream_offset"])
    gate_stream_bytes = int(traffic["gate_stream_bytes"])
    up_stream_offset = int(traffic["up_stream_offset"])
    up_stream_bytes = int(traffic["up_stream_bytes"])
    norm_stream_offset = int(traffic["norm_stream_offset"])
    norm_stream_bytes = int(traffic["norm_stream_bytes"])
    down_stream_offset = int(traffic["down_stream_offset"])
    down_stream_bytes = int(traffic["down_stream_bytes"])
    offsets = [
        header_block_bytes + directory_block_bytes + index * layer_block_bytes
        for index in range(layer_count)
    ]

    header = bytearray(header_block_bytes)
    _HEADER.pack_into(
        header,
        0,
        _MAGIC,
        _VERSION,
        layer_count,
        hidden_size,
        intermediate_size,
        cache_line_bytes,
        _DIRECTORY_ENTRY.size,
        directory_block_bytes,
        record_payload_bytes,
        float(rms_norm_eps),
    )
    directory = bytearray(directory_block_bytes)
    for index, offset in enumerate(offsets):
        _DIRECTORY_ENTRY.pack_into(
            directory,
            index * _DIRECTORY_ENTRY.size,
            index,
            0,
            offset,
            layer_block_bytes,
            layer_payload_bytes,
            0,
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    emitted = 0
    try:
        with temporary.open("wb") as handle:
            handle.write(header)
            handle.write(directory)
            for index, logical_layer in enumerate(layers):
                if index >= layer_count:
                    raise NativeBitNetValidationError(
                        "more source layers were provided than declared"
                    )
                layer = _pack_layer(
                    logical_layer,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                )
                layer_bytes = bytearray(layer_block_bytes)
                _LAYER_HEADER.pack_into(
                    layer_bytes,
                    0,
                    _LAYER_MAGIC,
                    _VERSION,
                    index,
                    hidden_size,
                    intermediate_size,
                    intermediate_size,
                    packed_width,
                    record_payload_bytes,
                    layer_payload_bytes,
                    _PROJECTION_SCALE_COUNT,
                    0,
                )
                scale_bytes = layer.scale_bits.astype(
                    "<u2",
                    copy=False,
                ).tobytes()
                gate_bytes = layer.gate_records.tobytes()
                up_bytes = layer.up_records.tobytes()
                norm_bytes = layer.norm_bits.astype("<u2", copy=False).tobytes()
                down_bytes = layer.down_records.tobytes()
                if (
                    len(gate_bytes) != gate_stream_bytes
                    or len(up_bytes) != up_stream_bytes
                    or len(norm_bytes) != norm_stream_bytes
                    or len(down_bytes) != down_stream_bytes
                ):
                    raise AssertionError(
                        "native BitNet phase stream differs from traffic accounting"
                    )
                layer_bytes[
                    _LAYER_HEADER_BYTES : _LAYER_HEADER_BYTES + len(scale_bytes)
                ] = scale_bytes
                layer_bytes[
                    gate_stream_offset : gate_stream_offset + gate_stream_bytes
                ] = gate_bytes
                layer_bytes[up_stream_offset : up_stream_offset + up_stream_bytes] = (
                    up_bytes
                )
                layer_bytes[
                    norm_stream_offset : norm_stream_offset + norm_stream_bytes
                ] = norm_bytes
                layer_bytes[
                    down_stream_offset : down_stream_offset + down_stream_bytes
                ] = down_bytes
                if layer_payload_bytes > layer_block_bytes:
                    raise AssertionError(
                        "native BitNet layer payload exceeds its block"
                    )
                handle.write(layer_bytes)
                emitted += 1
            if emitted != layer_count:
                raise NativeBitNetValidationError(
                    f"expected {layer_count} source layers, received {emitted}"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    if destination.stat().st_size != traffic["serialized_artifact_bytes"]:
        raise AssertionError("serialized native BitNet artifact has an unexpected size")
    return destination


def save_native_bitnet_artifact(
    path: str | Path,
    layers: Sequence[NativeBitNetLayerWeights],
    *,
    rms_norm_eps: float,
    cache_line_bytes: int = 64,
) -> Path:
    """Serialize an in-memory native BitNet fixture or model."""

    if not layers:
        raise NativeBitNetValidationError("layers must not be empty")
    first_gate = np.asarray(layers[0].gate_codes)
    if first_gate.ndim != 2:
        raise NativeBitNetValidationError("gate_codes must be a matrix")
    intermediate_size, hidden_size = first_gate.shape
    return _write_native_bitnet_artifact(
        path,
        iter(layers),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        layer_count=len(layers),
        rms_norm_eps=rms_norm_eps,
        cache_line_bytes=cache_line_bytes,
    )


def load_native_bitnet_artifact(
    path: str | Path,
) -> LoadedNativeBitNetArtifact:
    """Strictly validate and load a record-addressable native artifact."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise NativeBitNetValidationError(
            f"cannot read native BitNet artifact {source}"
        ) from exc
    if len(payload) < _HEADER_BYTES:
        raise NativeBitNetValidationError(
            "native BitNet artifact is shorter than its header"
        )
    (
        magic,
        version,
        layer_count,
        hidden_size,
        intermediate_size,
        cache_line_bytes,
        directory_entry_bytes,
        directory_block_bytes,
        record_payload_bytes,
        rms_norm_eps,
    ) = _HEADER.unpack_from(payload)
    if magic != _MAGIC or version != _VERSION:
        raise NativeBitNetValidationError(
            "native BitNet artifact magic/version mismatch"
        )
    for value, name in (
        (layer_count, "layer_count"),
        (hidden_size, "hidden_size"),
        (intermediate_size, "intermediate_size"),
        (cache_line_bytes, "cache_line_bytes"),
        (record_payload_bytes, "record_payload_bytes"),
    ):
        _positive_integer(value, name)
    if not math.isfinite(rms_norm_eps) or rms_norm_eps <= 0:
        raise NativeBitNetValidationError("native BitNet RMS epsilon is invalid")
    if directory_entry_bytes != _DIRECTORY_ENTRY.size:
        raise NativeBitNetValidationError(
            "native BitNet directory entry size is unsupported"
        )
    traffic = native_bitnet_repack_traffic(
        hidden_size,
        intermediate_size,
        layer_count=layer_count,
        cache_line_bytes=cache_line_bytes,
    )
    if record_payload_bytes != traffic["record_payload_bytes"]:
        raise NativeBitNetValidationError("native BitNet record size is inconsistent")
    if len(payload) != traffic["serialized_artifact_bytes"]:
        raise NativeBitNetValidationError(
            "native BitNet artifact length does not match its header"
        )
    header_block_bytes = int(traffic["header_cache_aligned_bytes"])
    if any(payload[_HEADER.size : header_block_bytes]):
        raise NativeBitNetValidationError("native BitNet header padding is non-zero")
    expected_directory_bytes = int(traffic["directory_cache_aligned_bytes"])
    if directory_block_bytes != expected_directory_bytes:
        raise NativeBitNetValidationError(
            "native BitNet directory block size is invalid"
        )
    directory_payload_end = header_block_bytes + layer_count * _DIRECTORY_ENTRY.size
    directory_end = header_block_bytes + directory_block_bytes
    if any(payload[directory_payload_end:directory_end]):
        raise NativeBitNetValidationError("native BitNet directory padding is non-zero")

    packed_width = int(traffic["packed_bytes_per_record_vector"])
    layer_payload_bytes = int(traffic["layer_payload_bytes"])
    layer_block_bytes = int(traffic["layer_block_bytes"])
    gate_stream_offset = int(traffic["gate_stream_offset"])
    gate_stream_bytes = int(traffic["gate_stream_bytes"])
    up_stream_offset = int(traffic["up_stream_offset"])
    up_stream_bytes = int(traffic["up_stream_bytes"])
    norm_stream_offset = int(traffic["norm_stream_offset"])
    norm_stream_bytes = int(traffic["norm_stream_bytes"])
    down_stream_offset = int(traffic["down_stream_offset"])
    down_stream_bytes = int(traffic["down_stream_bytes"])
    expected_offset = directory_end
    layers: list[PackedNativeBitNetLayer] = []
    offsets: list[int] = []
    block_sizes: list[int] = []
    for index in range(layer_count):
        (
            entry_index,
            reserved,
            offset,
            block_bytes,
            entry_payload_bytes,
            reserved_2,
        ) = _DIRECTORY_ENTRY.unpack_from(
            payload,
            header_block_bytes + index * _DIRECTORY_ENTRY.size,
        )
        if (
            entry_index != index
            or reserved
            or reserved_2
            or offset != expected_offset
            or offset % cache_line_bytes
            or block_bytes != layer_block_bytes
            or entry_payload_bytes != layer_payload_bytes
        ):
            raise NativeBitNetValidationError(
                "native BitNet directory entry is invalid"
            )
        (
            layer_magic,
            layer_version,
            layer_index,
            layer_hidden,
            layer_intermediate,
            record_count,
            layer_packed_width,
            layer_record_bytes,
            header_payload_bytes,
            scale_count,
            layer_reserved,
        ) = _LAYER_HEADER.unpack_from(payload, offset)
        if (
            layer_magic != _LAYER_MAGIC
            or layer_version != _VERSION
            or layer_index != index
            or layer_hidden != hidden_size
            or layer_intermediate != intermediate_size
            or record_count != intermediate_size
            or layer_packed_width != packed_width
            or layer_record_bytes != record_payload_bytes
            or header_payload_bytes != layer_payload_bytes
            or scale_count != _PROJECTION_SCALE_COUNT
            or layer_reserved
        ):
            raise NativeBitNetValidationError("native BitNet layer header is invalid")
        scale_start = offset + _LAYER_HEADER_BYTES
        scale_end = scale_start + _PROJECTION_SCALE_COUNT * _BF16_BYTES
        scale_bits = np.frombuffer(
            payload,
            dtype="<u2",
            count=_PROJECTION_SCALE_COUNT,
            offset=scale_start,
        ).copy()
        scales = _bf16_from_bits(scale_bits)
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise NativeBitNetValidationError(
                "native BitNet projection scales are invalid"
            )
        gate_start = offset + gate_stream_offset
        gate_end = gate_start + gate_stream_bytes
        up_start = offset + up_stream_offset
        up_end = up_start + up_stream_bytes
        norm_start = offset + norm_stream_offset
        norm_end = norm_start + norm_stream_bytes
        down_start = offset + down_stream_offset
        down_end = down_start + down_stream_bytes
        for padding_start, padding_end in (
            (scale_end, gate_start),
            (gate_end, up_start),
            (up_end, norm_start),
            (norm_end, down_start),
            (down_end, offset + layer_payload_bytes),
            (offset + layer_payload_bytes, offset + block_bytes),
        ):
            if any(payload[padding_start:padding_end]):
                raise NativeBitNetValidationError(
                    "native BitNet phase-stream padding is non-zero"
                )
        gate = (
            np.frombuffer(
                payload,
                dtype=np.uint8,
                count=intermediate_size * packed_width,
                offset=gate_start,
            )
            .reshape(intermediate_size, packed_width)
            .copy()
        )
        up = (
            np.frombuffer(
                payload,
                dtype=np.uint8,
                count=intermediate_size * packed_width,
                offset=up_start,
            )
            .reshape(intermediate_size, packed_width)
            .copy()
        )
        norm_bits = np.frombuffer(
            payload,
            dtype="<u2",
            count=intermediate_size,
            offset=norm_start,
        ).copy()
        down = (
            np.frombuffer(
                payload,
                dtype=np.uint8,
                count=intermediate_size * packed_width,
                offset=down_start,
            )
            .reshape(intermediate_size, packed_width)
            .copy()
        )
        norm = _bf16_from_bits(norm_bits)
        if not np.all(np.isfinite(norm)):
            raise NativeBitNetValidationError(
                "native BitNet normalization gains are invalid"
            )
        unpack_base3_rows(gate, logical_width=hidden_size)
        unpack_base3_rows(up, logical_width=hidden_size)
        unpack_base3_rows(down, logical_width=hidden_size)
        layers.append(
            PackedNativeBitNetLayer(
                gate_records=gate,
                up_records=up,
                down_records=down,
                scale_bits=scale_bits,
                norm_bits=np.ascontiguousarray(norm_bits),
            )
        )
        offsets.append(offset)
        block_sizes.append(block_bytes)
        expected_offset += block_bytes
    if expected_offset != len(payload):
        raise NativeBitNetValidationError("native BitNet artifact has trailing bytes")
    return LoadedNativeBitNetArtifact(
        layers=tuple(layers),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        rms_norm_eps=float(rms_norm_eps),
        cache_line_bytes=cache_line_bytes,
        record_payload_bytes=record_payload_bytes,
        header_block_bytes=header_block_bytes,
        directory_block_bytes=directory_block_bytes,
        layer_offsets=tuple(offsets),
        layer_block_bytes=tuple(block_sizes),
        serialized_artifact_bytes=len(payload),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
    )


def decode_native_bitnet_layer(
    artifact: LoadedNativeBitNetArtifact,
    layer: int,
) -> dict[str, NDArray[np.float32] | NDArray[np.int8]]:
    """Decode one layer without materializing the full model."""

    if isinstance(layer, bool) or not isinstance(layer, int):
        raise NativeBitNetValidationError("layer must be an integer")
    if not 0 <= layer < len(artifact.layers):
        raise NativeBitNetValidationError("layer index is outside the artifact")
    packed = artifact.layers[layer]
    scales = _bf16_from_bits(packed.scale_bits)
    return {
        "gate_codes": unpack_base3_rows(
            packed.gate_records,
            logical_width=artifact.hidden_size,
        ),
        "up_codes": unpack_base3_rows(
            packed.up_records,
            logical_width=artifact.hidden_size,
        ),
        "down_codes": unpack_base3_rows(
            packed.down_records,
            logical_width=artifact.hidden_size,
        ).T.copy(),
        "gate_scale": np.asarray(scales[0], dtype=np.float32),
        "up_scale": np.asarray(scales[1], dtype=np.float32),
        "down_scale": np.asarray(scales[2], dtype=np.float32),
        "ffn_sub_norm": _bf16_from_bits(packed.norm_bits),
    }


def _activation_quant(values: NDArray[np.float32]) -> NDArray[np.float32]:
    maximum = np.max(np.abs(values), axis=-1, keepdims=True)
    scale = np.float32(127.0) / np.maximum(maximum, np.float32(1e-5))
    return (
        np.rint(values * scale)
        .clip(np.float32(-128.0), np.float32(127.0))
        .astype(np.float32)
        / scale
    )


def native_bitnet_mlp_forward(
    artifact: LoadedNativeBitNetArtifact,
    layer: int,
    hidden: ArrayLike,
) -> NDArray[np.float32]:
    """Reference float32 forward preserving native BitNet operations."""

    state = np.asarray(hidden, dtype=np.float32)
    if state.ndim < 1 or state.shape[-1] != artifact.hidden_size:
        raise NativeBitNetValidationError("hidden states have an incompatible shape")
    decoded = decode_native_bitnet_layer(artifact, layer)
    quantized_state = _activation_quant(state)
    gate = (
        quantized_state
        @ np.asarray(decoded["gate_codes"], dtype=np.float32).T
        * np.asarray(decoded["gate_scale"], dtype=np.float32)
    )
    up = (
        quantized_state
        @ np.asarray(decoded["up_codes"], dtype=np.float32).T
        * np.asarray(decoded["up_scale"], dtype=np.float32)
    )
    activation = np.maximum(gate, np.float32(0.0)) ** 2 * up
    variance = np.mean(
        activation.astype(np.float32) ** 2,
        axis=-1,
        keepdims=True,
    )
    normalized = activation * np.reciprocal(
        np.sqrt(variance + np.float32(artifact.rms_norm_eps))
    )
    normalized *= np.asarray(decoded["ffn_sub_norm"], dtype=np.float32)
    quantized_normalized = _activation_quant(normalized)
    return np.asarray(
        quantized_normalized
        @ np.asarray(decoded["down_codes"], dtype=np.float32).T
        * np.asarray(decoded["down_scale"], dtype=np.float32),
        dtype=np.float32,
    )


def _resolve_full_source(
    model: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
) -> tuple[Path, str | None, str | None]:
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        return candidate.resolve(), None, None
    if candidate.exists():
        raise NativeBitNetValidationError(
            f"model path is not a directory: {candidate.resolve()}"
        )
    model_id = str(model)
    selected_revision = revision
    if model_id == OFFICIAL_NATIVE_BITNET_REPO and selected_revision is None:
        selected_revision = OFFICIAL_NATIVE_BITNET_REVISION
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise NativeBitNetValidationError(
            "install engram-lm[conversion] to download native BitNet weights"
        ) from exc
    try:
        downloaded = snapshot_download(
            repo_id=model_id,
            revision=selected_revision,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "*.safetensors",
                "*.safetensors.index.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
            ],
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    except Exception as exc:
        raise NativeBitNetValidationError(
            f"could not download native BitNet model {model_id!r}: {exc}"
        ) from exc
    path = Path(downloaded).resolve()
    return path, model_id, _resolved_snapshot_revision(path / "config.json")


def _safetensor_inventory(model_path: Path) -> dict[str, Path]:
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise NativeBitNetValidationError(
            "native BitNet source has no safetensors weights"
        )
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise NativeBitNetValidationError(
            "install engram-lm[conversion] to read safetensors"
        ) from exc
    inventory: dict[str, Path] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in inventory:
                    raise NativeBitNetValidationError(
                        f"duplicate source tensor {name!r}"
                    )
                inventory[name] = path
    return inventory


def _load_source_layer(
    inventory: dict[str, Path],
    layer: int,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> NativeBitNetLayerWeights:
    prefix = f"model.layers.{layer}.mlp"
    names = {
        "gate_weight": f"{prefix}.gate_proj.weight",
        "gate_scale": f"{prefix}.gate_proj.weight_scale",
        "up_weight": f"{prefix}.up_proj.weight",
        "up_scale": f"{prefix}.up_proj.weight_scale",
        "down_weight": f"{prefix}.down_proj.weight",
        "down_scale": f"{prefix}.down_proj.weight_scale",
        "norm": f"{prefix}.ffn_sub_norm.weight",
    }
    missing = [name for name in names.values() if name not in inventory]
    if missing:
        raise NativeBitNetValidationError(
            f"missing native BitNet layer-{layer} tensors: {missing}"
        )
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise NativeBitNetValidationError(
            "install engram-lm[conversion] to load native BitNet tensors"
        ) from exc

    tensors: dict[str, Any] = {}
    by_shard: dict[Path, list[tuple[str, str]]] = {}
    for role, name in names.items():
        by_shard.setdefault(inventory[name], []).append((role, name))
    for shard, entries in by_shard.items():
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for role, name in entries:
                tensors[role] = handle.get_tensor(name)

    expected_packed = {
        "gate_weight": (
            intermediate_size // _HF_TRITS_PER_BYTE,
            hidden_size,
        ),
        "up_weight": (
            intermediate_size // _HF_TRITS_PER_BYTE,
            hidden_size,
        ),
        "down_weight": (
            hidden_size // _HF_TRITS_PER_BYTE,
            intermediate_size,
        ),
    }
    for role, expected_shape in expected_packed.items():
        tensor = tensors[role]
        if tensor.dtype != torch.uint8 or tuple(tensor.shape) != expected_shape:
            raise NativeBitNetValidationError(
                f"{names[role]!r} must be uint8 with shape {expected_shape}"
            )
    for role in ("gate_scale", "up_scale", "down_scale"):
        tensor = tensors[role]
        if tensor.dtype != torch.bfloat16 or tensor.numel() != 1:
            raise NativeBitNetValidationError(
                f"{names[role]!r} must be a scalar BF16 tensor"
            )
    norm = tensors["norm"]
    if norm.dtype != torch.bfloat16 or tuple(norm.shape) != (intermediate_size,):
        raise NativeBitNetValidationError(
            f"{names['norm']!r} must be BF16 with shape ({intermediate_size},)"
        )

    gate_packed = tensors["gate_weight"].cpu().numpy()
    up_packed = tensors["up_weight"].cpu().numpy()
    down_packed = tensors["down_weight"].cpu().numpy()
    return NativeBitNetLayerWeights(
        gate_codes=unpack_hf_bitnet_codes(
            gate_packed,
            out_features=intermediate_size,
        ),
        up_codes=unpack_hf_bitnet_codes(
            up_packed,
            out_features=intermediate_size,
        ),
        down_codes=unpack_hf_bitnet_codes(
            down_packed,
            out_features=hidden_size,
        ),
        gate_scale=float(tensors["gate_scale"].float().item()),
        up_scale=float(tensors["up_scale"].float().item()),
        down_scale=float(tensors["down_scale"].float().item()),
        ffn_sub_norm=norm.float().cpu().numpy(),
    )


def _verify_repacked_layers(
    inventory: dict[str, Path],
    artifact: LoadedNativeBitNetArtifact,
) -> dict[str, Any]:
    """Compare every reconstructed logical value with the packed source."""

    source_digest = hashlib.sha256()
    artifact_digest = hashlib.sha256()
    coefficients = 0
    bf16_values = 0
    for layer_index in range(len(artifact.layers)):
        source = _load_source_layer(
            inventory,
            layer_index,
            hidden_size=artifact.hidden_size,
            intermediate_size=artifact.intermediate_size,
        )
        decoded = decode_native_bitnet_layer(artifact, layer_index)
        for role in ("gate_codes", "up_codes", "down_codes"):
            expected = np.asarray(getattr(source, role), dtype=np.int8)
            actual = np.asarray(decoded[role], dtype=np.int8)
            if not np.array_equal(expected, actual):
                raise NativeBitNetValidationError(
                    f"repacked layer-{layer_index} {role} differs from source"
                )
            source_digest.update(expected.tobytes(order="C"))
            artifact_digest.update(actual.tobytes(order="C"))
            coefficients += expected.size
        expected_scales = _bf16_bits_exact(
            np.asarray(
                [source.gate_scale, source.up_scale, source.down_scale],
                dtype=np.float32,
            ),
            f"source layer-{layer_index} scales",
        )
        actual_scales = artifact.layers[layer_index].scale_bits
        if not np.array_equal(expected_scales, actual_scales):
            raise NativeBitNetValidationError(
                f"repacked layer-{layer_index} scales differ from source"
            )
        expected_norm = _bf16_bits_exact(
            source.ffn_sub_norm,
            f"source layer-{layer_index} ffn_sub_norm",
        )
        actual_norm = artifact.layers[layer_index].norm_bits
        if not np.array_equal(expected_norm, actual_norm):
            raise NativeBitNetValidationError(
                f"repacked layer-{layer_index} ffn_sub_norm differs from source"
            )
        for expected, actual in (
            (expected_scales, actual_scales),
            (expected_norm, actual_norm),
        ):
            source_digest.update(expected.astype("<u2", copy=False).tobytes(order="C"))
            artifact_digest.update(actual.astype("<u2", copy=False).tobytes(order="C"))
            bf16_values += expected.size
    source_sha256 = source_digest.hexdigest()
    artifact_sha256 = artifact_digest.hexdigest()
    if source_sha256 != artifact_sha256:
        raise AssertionError("logical source/artifact digests unexpectedly differ")
    return {
        "passed": True,
        "layers_compared": len(artifact.layers),
        "ternary_coefficients_compared": coefficients,
        "bf16_values_compared": bf16_values,
        "logical_source_sha256": source_sha256,
        "logical_artifact_sha256": artifact_sha256,
    }


def repack_native_bitnet_model(
    model: str | Path,
    out: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    verify_official_weight_hash: bool = True,
) -> dict[str, Any]:
    """Download if needed, validate, and losslessly repack a native source."""

    model_path, repo_id, resolved_revision = _resolve_full_source(
        model,
        revision=revision,
        cache_dir=cache_dir,
    )
    config_path = model_path / "config.json"
    _read_json(config_path)
    audit = audit_native_bitnet_source(model_path)
    if not all(audit.checks.values()):
        raise NativeBitNetValidationError(
            "native BitNet source configuration failed the format audit"
        )
    assert audit.hidden_size is not None
    assert audit.intermediate_size is not None
    assert audit.num_hidden_layers is not None
    assert audit.rms_norm_eps is not None
    inventory = _safetensor_inventory(model_path)
    weight_files = sorted(set(inventory.values()))
    weight_hashes = {
        path.name: sha256_file(path)
        for path in weight_files
        if verify_official_weight_hash
    }
    official_revision_pinned = (
        repo_id == OFFICIAL_NATIVE_BITNET_REPO
        and resolved_revision == OFFICIAL_NATIVE_BITNET_REVISION
    )
    official_weight_hash_verified = bool(
        official_revision_pinned
        and verify_official_weight_hash
        and weight_hashes.get("model.safetensors")
        == OFFICIAL_NATIVE_BITNET_WEIGHT_SHA256
    )
    if (
        official_revision_pinned
        and verify_official_weight_hash
        and not (official_weight_hash_verified)
    ):
        raise NativeBitNetValidationError(
            "pinned official model.safetensors SHA-256 mismatch"
        )

    def source_layers() -> Iterator[NativeBitNetLayerWeights]:
        for layer in range(audit.num_hidden_layers or 0):
            yield _load_source_layer(
                inventory,
                layer,
                hidden_size=audit.hidden_size or 0,
                intermediate_size=audit.intermediate_size or 0,
            )

    artifact_path = Path(out)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{artifact_path.name}.verify-",
        suffix=".bin",
        dir=artifact_path.parent,
        delete=False,
    ) as staging_handle:
        staging_path = Path(staging_handle.name)
    try:
        _write_native_bitnet_artifact(
            staging_path,
            source_layers(),
            hidden_size=audit.hidden_size,
            intermediate_size=audit.intermediate_size,
            layer_count=audit.num_hidden_layers,
            rms_norm_eps=audit.rms_norm_eps,
            cache_line_bytes=64,
        )
        loaded = load_native_bitnet_artifact(staging_path)
        traffic = native_bitnet_repack_traffic(
            audit.hidden_size,
            audit.intermediate_size,
            layer_count=audit.num_hidden_layers,
        )
        if loaded.serialized_artifact_bytes != traffic["serialized_artifact_bytes"]:
            raise AssertionError("reloaded artifact traffic differs from projection")
        reconstruction = _verify_repacked_layers(inventory, loaded)
        if verify_official_weight_hash:
            stable_weight_hashes = {
                path.name: sha256_file(path) for path in weight_files
            }
            if stable_weight_hashes != weight_hashes:
                raise NativeBitNetValidationError(
                    "source weights changed during native BitNet repacking"
                )
        staging_path.replace(artifact_path)
    finally:
        staging_path.unlink(missing_ok=True)
    result = {
        "schema_version": 2,
        "report_stage": "exact_repack_before_parity",
        "status_is_stage_local": True,
        "source_track": "low_bit_native",
        "source_model": str(model),
        "resolved_model_path": str(model_path),
        "repository": repo_id,
        "resolved_revision": resolved_revision,
        "config_sha256": sha256_file(config_path),
        "weight_hashes": weight_hashes,
        "provenance_verification": {
            "official_revision_pinned": official_revision_pinned,
            "weight_hash_verification_requested": verify_official_weight_hash,
            "official_weight_hash_verified": official_weight_hash_verified,
            "source_stability_rechecked": verify_official_weight_hash,
        },
        "artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": sha256_file(artifact_path),
            "serialized_bytes": artifact_path.stat().st_size,
            "reloaded": True,
            "encoding": "native_bitnet_phase_base3_v1",
        },
        "traffic": traffic,
        "representation_checks": {
            "all_hf_two_bit_codes_validated": True,
            "base3_streams_canonical_after_reload": True,
            "projection_scales_preserved_as_bf16": True,
            "ffn_sub_norm_preserved_as_bf16": True,
            "phase_streams_are_fixed_stride": True,
            "logical_reconstruction": reconstruction,
        },
        "quality_status": (
            "weight_representation_exact; causal substitution not yet run"
        ),
        "combined_gate_status": "not_yet_evaluated",
        "dense_llama_conversion_status": "not_applicable",
        "next_step": (
            "run layer-local parity and a small all-layer causal substitution "
            "against the pinned BitNet reference after strict packed-weight "
            "materialization"
        ),
    }
    destination_report = (
        Path(report_path)
        if report_path is not None
        else artifact_path.with_suffix(artifact_path.suffix + ".json")
    )
    result["report_path"] = str(destination_report.resolve())
    atomic_json(destination_report, result)
    return result


__all__ = [
    "LoadedNativeBitNetArtifact",
    "NativeBitNetLayerWeights",
    "NativeBitNetSourceAudit",
    "NativeBitNetValidationError",
    "OFFICIAL_NATIVE_BITNET_REPO",
    "OFFICIAL_NATIVE_BITNET_REVISION",
    "audit_native_bitnet_source",
    "decode_native_bitnet_layer",
    "load_native_bitnet_artifact",
    "native_bitnet_mlp_forward",
    "native_bitnet_repack_traffic",
    "pack_base3_rows",
    "pack_hf_bitnet_codes",
    "repack_native_bitnet_model",
    "save_native_bitnet_artifact",
    "unpack_base3_rows",
    "unpack_hf_bitnet_codes",
]
