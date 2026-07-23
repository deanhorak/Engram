"""Authenticated shared-input-basis artifacts and physical sparse MLP modules.

The ESIB v1 representation stores one converted SwiGLU layer as:

* a Q3 shared input basis;
* Q2 gate and up coefficients;
* FP16 down-row norms used for full-width hard top-K selection; and
* independently addressable, 64-byte-aligned Q4 down rows.

This module deliberately implements decoding, authentication, traffic
accounting, and artifact-only execution.  It does not contain a fitting path
and it never consults the source model's dense MLP weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from engram.training.canonical_rans import rans_decode

try:  # Torch is an optional conversion/evaluation dependency.
    import torch
    import torch.nn.functional as torch_functional
except ImportError:  # pragma: no cover - exercised by minimal installations.
    torch = None
    torch_functional = None


ESIB_MAGIC = b"ESIBQ344"
ESIB_VERSION = 1
ESIB_LINE_BYTES = 64
ESIB_HEADER_BYTES = 256
ESIB_SCALE_BITS = 12
ESIB_BASIS_BITS = 3
ESIB_COEFFICIENT_BITS = 2
ESIB_DOWN_BITS = 4
ESIB_MANIFEST_FORMAT = "engram_shared_basis_artifact_set_v1"
ESIB_MANIFEST_VERSION = 1

_HEADER_PREFIX = struct.Struct("<8sHHIIIIIHHHHff")
_SECTION_DESCRIPTOR = struct.Struct("<QQ")
_DOWN_RECORD_HEADER = struct.Struct("<HH")
_SECTION_NAMES = (
    "frequency_models",
    "basis_scales",
    "gate_scales",
    "up_scales",
    "down_norms",
    "down_index",
    "basis_payload",
    "gate_payload",
    "up_payload",
    "down_records",
)
_HEADER_CORE_BYTES = (
    _HEADER_PREFIX.size + len(_SECTION_NAMES) * _SECTION_DESCRIPTOR.size
)
_CHECKSUM_OFFSET = _HEADER_CORE_BYTES
_CHECKSUM_BYTES = 32
_HEADER_PADDING_OFFSET = _CHECKSUM_OFFSET + _CHECKSUM_BYTES

if _HEADER_CORE_BYTES != 208 or _HEADER_PADDING_OFFSET > ESIB_HEADER_BYTES:
    raise AssertionError("ESIB v1 header layout no longer fits its fixed header")


def _align(value: int) -> int:
    if value < 0:
        raise ValueError("cannot align a negative byte count")
    return (value + ESIB_LINE_BYTES - 1) // ESIB_LINE_BYTES * ESIB_LINE_BYTES


def _sha256(payload: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _zero_checksum(payload: bytes | bytearray | memoryview) -> bytearray:
    result = bytearray(payload)
    result[_CHECKSUM_OFFSET : _CHECKSUM_OFFSET + _CHECKSUM_BYTES] = bytes(
        _CHECKSUM_BYTES
    )
    return result


def _content_checksum(payload: bytes | bytearray | memoryview) -> bytes:
    return hashlib.sha256(_zero_checksum(payload)).digest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return value == value.lower()


@dataclass(frozen=True)
class SharedBasisSection:
    """One canonical, cache-line-aligned region in an ESIB artifact."""

    offset: int
    logical_bytes: int

    @property
    def physical_bytes(self) -> int:
        return _align(self.logical_bytes)


@dataclass(frozen=True)
class SharedBasisArtifact:
    """Strictly decoded weights and physical layout for one ESIB layer."""

    payload: bytes
    layer: int
    rank: int
    top_k: int
    width: int
    hidden: int
    gate_ratio: float
    up_ratio: float
    sections: Mapping[str, SharedBasisSection]
    basis: np.ndarray
    gate_coeff: np.ndarray
    up_coeff: np.ndarray
    down: np.ndarray
    down_norms: np.ndarray
    record_bytes: np.ndarray
    content_checksum: str
    artifact_sha256: str


@dataclass(frozen=True)
class SharedBasisArtifactSet:
    """Manifest-authenticated collection of per-layer ESIB artifacts."""

    source_model_hash: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    artifacts: Mapping[int, SharedBasisArtifact]
    manifest_path: Path
    manifest_sha256: str
    artifact_set_sha256: str
    provenance: Mapping[str, Any]


def _require_zero(
    view: memoryview,
    start: int,
    stop: int,
    label: str,
) -> None:
    if start < 0 or stop < start or stop > len(view):
        raise ValueError(f"{label} padding range is invalid")
    if any(view[start:stop]):
        raise ValueError(f"nonzero padding in {label}")


def _parse_models(
    view: memoryview,
    section: SharedBasisSection,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    expected_entries = 7 + 4 + 4 + 15
    if section.logical_bytes != expected_entries * 2:
        raise ValueError("frequency-model region has the wrong logical size")
    flat = np.frombuffer(
        view[section.offset : section.offset + section.logical_bytes],
        dtype="<u2",
    )
    boundaries = (0, 7, 11, 15, 30)
    models = tuple(
        tuple(int(value) for value in flat[start:stop])
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
    )
    for model in models:
        if any(value <= 0 for value in model) or sum(model) != 1 << ESIB_SCALE_BITS:
            raise ValueError("invalid canonical-rANS frequency model")
    return models  # type: ignore[return-value]


def _read_positive_fp16(
    view: memoryview,
    section: SharedBasisSection,
    count: int,
    label: str,
) -> np.ndarray:
    if section.logical_bytes != count * 2:
        raise ValueError(f"{label} region has the wrong logical size")
    result = np.frombuffer(
        view[section.offset : section.offset + section.logical_bytes],
        dtype="<f2",
    ).astype(np.float32)
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ValueError(f"{label} contains an invalid value")
    return result


def _decode_fixed_stream(
    view: memoryview,
    section: SharedBasisSection,
    count: int,
    model: tuple[int, ...],
) -> np.ndarray:
    return rans_decode(
        view[section.offset : section.offset + section.logical_bytes],
        count,
        model,
        scale_bits=ESIB_SCALE_BITS,
    )


def decode_shared_basis_artifact(
    payload: bytes | bytearray | memoryview,
) -> SharedBasisArtifact:
    """Authenticate and strictly decode one canonical ESIB v1 artifact."""

    immutable = bytes(payload)
    view = memoryview(immutable)
    if len(view) < ESIB_HEADER_BYTES:
        raise ValueError("ESIB artifact is shorter than its fixed header")
    (
        magic,
        version,
        header_bytes,
        layer,
        rank,
        top_k,
        width,
        hidden,
        basis_bits,
        coefficient_bits,
        down_bits,
        scale_bits,
        gate_ratio,
        up_ratio,
    ) = _HEADER_PREFIX.unpack_from(view, 0)
    if (
        magic != ESIB_MAGIC
        or version != ESIB_VERSION
        or header_bytes != ESIB_HEADER_BYTES
        or rank <= 0
        or width <= 0
        or hidden <= 0
        or top_k <= 0
        or top_k > width
        or basis_bits != ESIB_BASIS_BITS
        or coefficient_bits != ESIB_COEFFICIENT_BITS
        or down_bits != ESIB_DOWN_BITS
        or scale_bits != ESIB_SCALE_BITS
        or not math.isfinite(gate_ratio)
        or not math.isfinite(up_ratio)
        or not 0 < gate_ratio < 1
        or not 0 < up_ratio < 1
    ):
        raise ValueError("invalid ESIB artifact header")

    descriptor_cursor = _HEADER_PREFIX.size
    sections: dict[str, SharedBasisSection] = {}
    expected_offset = ESIB_HEADER_BYTES
    for name in _SECTION_NAMES:
        offset, logical_bytes = _SECTION_DESCRIPTOR.unpack_from(view, descriptor_cursor)
        descriptor_cursor += _SECTION_DESCRIPTOR.size
        if (
            offset != expected_offset
            or offset % ESIB_LINE_BYTES
            or logical_bytes <= 0
            or offset + logical_bytes > len(view)
        ):
            raise ValueError(f"invalid canonical ESIB section layout for {name}")
        section = SharedBasisSection(int(offset), int(logical_bytes))
        sections[name] = section
        expected_offset = section.offset + section.physical_bytes
    if descriptor_cursor != _CHECKSUM_OFFSET or expected_offset != len(view):
        raise ValueError("ESIB size does not match its canonical section layout")

    expected_checksum = bytes(
        view[_CHECKSUM_OFFSET : _CHECKSUM_OFFSET + _CHECKSUM_BYTES]
    )
    if _content_checksum(view) != expected_checksum:
        raise ValueError("ESIB content checksum mismatch")
    _require_zero(
        view,
        _HEADER_PADDING_OFFSET,
        ESIB_HEADER_BYTES,
        "ESIB header",
    )
    for name in _SECTION_NAMES[:-1]:
        section = sections[name]
        _require_zero(
            view,
            section.offset + section.logical_bytes,
            section.offset + section.physical_bytes,
            f"ESIB {name}",
        )

    basis_model, gate_model, up_model, down_model = _parse_models(
        view, sections["frequency_models"]
    )
    basis_scales = _read_positive_fp16(
        view, sections["basis_scales"], int(rank), "basis scales"
    )
    gate_scales = _read_positive_fp16(
        view, sections["gate_scales"], int(width), "gate scales"
    )
    up_scales = _read_positive_fp16(
        view, sections["up_scales"], int(width), "up scales"
    )
    down_norms = _read_positive_fp16(
        view, sections["down_norms"], int(width), "down norms"
    )

    index_section = sections["down_index"]
    if index_section.logical_bytes != (width + 1) * 2:
        raise ValueError("ESIB down index has the wrong logical size")
    line_offsets = np.frombuffer(
        view[index_section.offset : index_section.offset + index_section.logical_bytes],
        dtype="<u2",
    ).astype(np.int64)
    down_section = sections["down_records"]
    if (
        line_offsets[0] != 0
        or np.any(np.diff(line_offsets) <= 0)
        or int(line_offsets[-1]) * ESIB_LINE_BYTES != down_section.logical_bytes
    ):
        raise ValueError("ESIB down index is not a canonical increasing line index")
    record_bytes = np.diff(line_offsets) * ESIB_LINE_BYTES

    basis_codes = _decode_fixed_stream(
        view,
        sections["basis_payload"],
        int(rank * hidden),
        basis_model,
    ).reshape(rank, hidden)
    gate_codes = _decode_fixed_stream(
        view,
        sections["gate_payload"],
        int(width * rank),
        gate_model,
    ).reshape(width, rank)
    up_codes = _decode_fixed_stream(
        view,
        sections["up_payload"],
        int(width * rank),
        up_model,
    ).reshape(width, rank)
    basis = (basis_codes.astype(np.float32) - 3.0) * basis_scales[:, None]
    gate_levels = np.asarray([-1.0, -gate_ratio, gate_ratio, 1.0], dtype=np.float32)
    up_levels = np.asarray([-1.0, -up_ratio, up_ratio, 1.0], dtype=np.float32)
    gate_coeff = gate_levels[gate_codes] * gate_scales[:, None]
    up_coeff = up_levels[up_codes] * up_scales[:, None]

    down = np.empty((width, hidden), dtype=np.float32)
    for row in range(width):
        start = down_section.offset + int(line_offsets[row]) * ESIB_LINE_BYTES
        stop = down_section.offset + int(line_offsets[row + 1]) * ESIB_LINE_BYTES
        scale_bits_raw, payload_bytes = _DOWN_RECORD_HEADER.unpack_from(view, start)
        scale = float(np.asarray([scale_bits_raw], dtype=np.uint16).view(np.float16)[0])
        payload_start = start + _DOWN_RECORD_HEADER.size
        payload_stop = payload_start + payload_bytes
        if (
            not math.isfinite(scale)
            or scale <= 0
            or payload_bytes < 4
            or payload_stop > stop
        ):
            raise ValueError(f"ESIB down record {row} has an invalid header")
        codes = rans_decode(
            view[payload_start:payload_stop],
            int(hidden),
            down_model,
            scale_bits=ESIB_SCALE_BITS,
        )
        down[row] = (codes.astype(np.float32) - 7.0) * scale
        _require_zero(view, payload_stop, stop, f"ESIB down record {row}")

    arrays = (basis, gate_coeff, up_coeff, down, down_norms, record_bytes)
    for array in arrays:
        if not np.all(np.isfinite(array)):
            raise ValueError("decoded ESIB artifact contains a non-finite value")
        array.setflags(write=False)
    return SharedBasisArtifact(
        payload=immutable,
        layer=int(layer),
        rank=int(rank),
        top_k=int(top_k),
        width=int(width),
        hidden=int(hidden),
        gate_ratio=float(gate_ratio),
        up_ratio=float(up_ratio),
        sections=sections,
        basis=np.ascontiguousarray(basis),
        gate_coeff=np.ascontiguousarray(gate_coeff),
        up_coeff=np.ascontiguousarray(up_coeff),
        down=np.ascontiguousarray(down),
        down_norms=np.ascontiguousarray(down_norms),
        record_bytes=np.ascontiguousarray(record_bytes),
        content_checksum=expected_checksum.hex(),
        artifact_sha256=_sha256(immutable),
    )


def load_shared_basis_artifact(path: str | Path) -> SharedBasisArtifact:
    """Read, authenticate, and decode one ESIB file."""

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise ValueError(f"ESIB artifact is not a file: {artifact_path}")
    return decode_shared_basis_artifact(artifact_path.read_bytes())


def _positive_manifest_integer(manifest: Mapping[str, Any], name: str) -> int:
    value = manifest.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"ESIB manifest {name} must be a positive integer")
    return value


def _manifest_artifact_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("ESIB manifest artifact path must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("ESIB manifest artifact paths must stay below the manifest")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            "ESIB manifest artifact path escapes the manifest directory"
        ) from exc
    return resolved


def load_shared_basis_artifact_set(
    manifest_path: str | Path,
    *,
    expected_source_model_hash: str | None = None,
    expected_hidden_size: int | None = None,
    expected_intermediate_size: int | None = None,
    expected_num_hidden_layers: int | None = None,
    required_layers: Sequence[int] | None = None,
    require_all_layers: bool = True,
) -> SharedBasisArtifactSet:
    """Load a manifest and verify every exact ESIB artifact it names.

    ESIB v1 authenticates its own byte stream, but its fixed header intentionally
    does not contain model identity.  The manifest closes that provenance gap:
    callers bind it to the independently inspected source-model hash, and each
    manifest entry binds a layer to both the exact file SHA-256 and the embedded
    ESIB content checksum.
    """

    path = Path(manifest_path)
    if not path.is_file():
        raise ValueError(f"ESIB artifact-set manifest is not a file: {path}")
    raw_manifest = path.read_bytes()
    try:
        manifest = json.loads(raw_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid ESIB artifact-set JSON manifest") from exc
    if not isinstance(manifest, dict):
        raise ValueError("ESIB artifact-set manifest must be a JSON object")
    if (
        manifest.get("format") != ESIB_MANIFEST_FORMAT
        or manifest.get("version") != ESIB_MANIFEST_VERSION
    ):
        raise ValueError("unsupported ESIB artifact-set manifest format")
    source_model_hash = manifest.get("source_model_hash")
    if not _is_sha256(source_model_hash):
        raise ValueError("ESIB manifest source_model_hash must be lowercase SHA-256")
    if (
        expected_source_model_hash is not None
        and source_model_hash != expected_source_model_hash
    ):
        raise ValueError("ESIB manifest/source-model hash mismatch")

    hidden_size = _positive_manifest_integer(manifest, "hidden_size")
    intermediate_size = _positive_manifest_integer(manifest, "intermediate_size")
    num_hidden_layers = _positive_manifest_integer(manifest, "num_hidden_layers")
    expected_values = (
        ("hidden_size", hidden_size, expected_hidden_size),
        ("intermediate_size", intermediate_size, expected_intermediate_size),
        ("num_hidden_layers", num_hidden_layers, expected_num_hidden_layers),
    )
    for name, actual, expected in expected_values:
        if expected is not None and actual != expected:
            raise ValueError(f"ESIB manifest/source-model {name} mismatch")

    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("ESIB manifest artifacts must be a non-empty list")
    artifacts: dict[int, SharedBasisArtifact] = {}
    canonical_entries: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each ESIB manifest artifact entry must be an object")
        layer = entry.get("layer")
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer < 0
            or layer >= num_hidden_layers
            or layer in artifacts
        ):
            raise ValueError("ESIB manifest contains an invalid or duplicate layer")
        declared_sha = entry.get("sha256")
        declared_checksum = entry.get("content_checksum")
        if not _is_sha256(declared_sha) or not _is_sha256(declared_checksum):
            raise ValueError("ESIB artifact hashes must be lowercase SHA-256")
        artifact_path = _manifest_artifact_path(path.parent, entry.get("path"))
        if artifact_path in seen_paths:
            raise ValueError("ESIB manifest contains a duplicate artifact path")
        seen_paths.add(artifact_path)
        artifact = load_shared_basis_artifact(artifact_path)
        if (
            artifact.artifact_sha256 != declared_sha
            or artifact.content_checksum != declared_checksum
        ):
            raise ValueError(f"ESIB manifest authentication failed for layer {layer}")
        if (
            artifact.layer != layer
            or artifact.hidden != hidden_size
            or artifact.width != intermediate_size
        ):
            raise ValueError(f"ESIB manifest/header mismatch for layer {layer}")
        artifacts[layer] = artifact
        canonical_entries.append(
            {
                "layer": layer,
                "sha256": artifact.artifact_sha256,
                "content_checksum": artifact.content_checksum,
            }
        )

    raw_requested: Sequence[int] = (
        tuple(range(num_hidden_layers))
        if require_all_layers and required_layers is None
        else tuple(artifacts if required_layers is None else required_layers)
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < 0
        or value >= num_hidden_layers
        for value in raw_requested
    ):
        raise ValueError("required ESIB layers contain an invalid layer index")
    requested = {int(value) for value in raw_requested}
    if require_all_layers:
        requested = set(range(num_hidden_layers))
    if set(artifacts) != requested:
        missing = sorted(requested - set(artifacts))
        unexpected = sorted(set(artifacts) - requested)
        raise ValueError(
            "ESIB manifest does not exactly cover required layers "
            f"(missing={missing}, unexpected={unexpected})"
        )

    provenance = manifest.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("ESIB manifest provenance must be an object when provided")
    artifact_set_identity = {
        "format": ESIB_MANIFEST_FORMAT,
        "version": ESIB_MANIFEST_VERSION,
        "source_model_hash": source_model_hash,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "artifacts": sorted(canonical_entries, key=lambda item: item["layer"]),
    }
    identity_payload = json.dumps(
        artifact_set_identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return SharedBasisArtifactSet(
        source_model_hash=source_model_hash,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        artifacts=dict(sorted(artifacts.items())),
        manifest_path=path.resolve(),
        manifest_sha256=_sha256(raw_manifest),
        artifact_set_sha256=_sha256(identity_payload),
        provenance=provenance,
    )


def shared_basis_fixed_traffic(artifact: SharedBasisArtifact) -> int:
    """Physical fixed-prefix bytes read for every token at one layer."""

    return artifact.sections["down_records"].offset


def shared_basis_traffic(
    artifact: SharedBasisArtifact,
    selected: np.ndarray | None = None,
) -> dict[str, Any]:
    """Account exact physical cold reads against an ideal dense Q4 MLP."""

    dense_q4 = (3 * artifact.width * artifact.hidden + 1) // 2
    traffic_limit = math.floor(0.45 * dense_q4)
    fixed_bytes = shared_basis_fixed_traffic(artifact)
    adversarial_selected = int(np.sort(artifact.record_bytes)[-artifact.top_k :].sum())
    adversarial_total = fixed_bytes + adversarial_selected
    report: dict[str, Any] = {
        "dense_ideal_q4_bytes": dense_q4,
        "strict_45_percent_limit_floor_bytes": traffic_limit,
        "fixed_bytes": fixed_bytes,
        "adversarial_selected_down_record_bytes": adversarial_selected,
        "adversarial_total_cold_bytes": adversarial_total,
        "adversarial_fraction_of_dense_q4": adversarial_total / dense_q4,
        "adversarial_passes_45_percent": adversarial_total <= traffic_limit,
    }
    if selected is not None:
        indices = np.asarray(selected)
        if (
            indices.ndim != 2
            or indices.shape[1] != artifact.top_k
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= artifact.width)
        ):
            raise ValueError("selected ESIB indices have the wrong shape or range")
        if any(len(np.unique(row)) != artifact.top_k for row in indices):
            raise ValueError("selected ESIB indices contain a duplicate row")
        selected_bytes = artifact.record_bytes[indices].sum(axis=1)
        totals = selected_bytes + fixed_bytes
        report["observed"] = {
            "count": int(len(totals)),
            "minimum_total_cold_bytes": int(totals.min()),
            "mean_total_cold_bytes": float(totals.mean()),
            "maximum_total_cold_bytes": int(totals.max()),
            "mean_fraction_of_dense_q4": float(totals.mean() / dense_q4),
            "maximum_fraction_of_dense_q4": float(totals.max() / dense_q4),
            "passes_45_percent_at_maximum": bool(totals.max() <= traffic_limit),
        }
    return report


if torch is not None:

    class SharedBasisMLP(torch.nn.Module):
        """Physical SwiGLU replacement whose weights come only from one ESIB."""

        def __init__(
            self,
            artifact: SharedBasisArtifact,
            *,
            capture: bool = False,
            execution_chunk_size: int = 64,
        ) -> None:
            super().__init__()
            if (
                isinstance(execution_chunk_size, bool)
                or not isinstance(execution_chunk_size, int)
                or execution_chunk_size <= 0
            ):
                raise ValueError("execution_chunk_size must be a positive integer")
            self.layer = artifact.layer
            self.rank = artifact.rank
            self.top_k = artifact.top_k
            self.width = artifact.width
            self.hidden = artifact.hidden
            self.artifact_sha256 = artifact.artifact_sha256
            self.content_checksum = artifact.content_checksum
            self.capture = bool(capture)
            self.execution_chunk_size = execution_chunk_size
            self.register_buffer(
                "basis", torch.tensor(np.array(artifact.basis, copy=True))
            )
            self.register_buffer(
                "gate_coeff", torch.tensor(np.array(artifact.gate_coeff, copy=True))
            )
            self.register_buffer(
                "up_coeff", torch.tensor(np.array(artifact.up_coeff, copy=True))
            )
            self.register_buffer(
                "down", torch.tensor(np.array(artifact.down, copy=True))
            )
            self.register_buffer(
                "down_norms",
                torch.tensor(np.array(artifact.down_norms, copy=True)),
            )
            self._last_output: Any | None = None
            self._last_selected: Any | None = None

        def forward(self, hidden_states: Any) -> Any:
            if hidden_states.shape[-1] != self.hidden:
                raise ValueError(
                    "ESIB MLP input width differs from the artifact hidden size"
                )
            original_shape = hidden_states.shape
            output_dtype = hidden_states.dtype
            flat = hidden_states.reshape(-1, self.hidden).float()
            output_chunks = []
            selected_chunks = []
            for start in range(0, len(flat), self.execution_chunk_size):
                values = flat[start : start + self.execution_chunk_size]
                latent = torch_functional.linear(values, self.basis)
                gate = torch_functional.linear(latent, self.gate_coeff)
                up = torch_functional.linear(latent, self.up_coeff)
                activation = torch_functional.silu(gate) * up
                scores = activation.abs() * self.down_norms.unsqueeze(0)
                selected = torch.topk(
                    scores,
                    self.top_k,
                    dim=1,
                    largest=True,
                    sorted=False,
                ).indices
                active_values = activation.gather(1, selected)
                output_chunks.append(
                    torch.bmm(
                        active_values.unsqueeze(1),
                        self.down[selected],
                    ).squeeze(1)
                )
                if self.capture:
                    selected_chunks.append(selected.detach())
            output = torch.cat(output_chunks, dim=0).reshape(
                *original_shape[:-1], self.hidden
            )
            if self.capture:
                self._last_output = output.detach()
                self._last_selected = torch.cat(selected_chunks, dim=0)
            return output.to(dtype=output_dtype)

        def pop_capture(self) -> tuple[Any, Any]:
            """Return and clear the most recent diagnostic output/selection."""

            if not self.capture:
                raise RuntimeError("capture is disabled for this ESIB MLP")
            if self._last_output is None or self._last_selected is None:
                raise RuntimeError("ESIB MLP has no completed forward to capture")
            output = self._last_output
            selected = self._last_selected
            self._last_output = None
            self._last_selected = None
            return output, selected

else:

    class SharedBasisMLP:  # pragma: no cover - only used without conversion extras.
        """Placeholder that explains the optional dependency at construction."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "install engram-lm[conversion] to execute shared-basis artifacts"
            )


__all__ = [
    "ESIB_MANIFEST_FORMAT",
    "ESIB_MANIFEST_VERSION",
    "SharedBasisArtifact",
    "SharedBasisArtifactSet",
    "SharedBasisMLP",
    "SharedBasisSection",
    "decode_shared_basis_artifact",
    "load_shared_basis_artifact",
    "load_shared_basis_artifact_set",
    "shared_basis_fixed_traffic",
    "shared_basis_traffic",
]
