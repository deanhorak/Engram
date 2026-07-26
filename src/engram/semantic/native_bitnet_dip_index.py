"""Durable coordinate-major index for practical native-BitNet DIP.

The native BitNet MLP artifact remains the record-major source used for exact
candidate completion and selected down-record reads.  This companion artifact
duplicates only gate/up in coordinate-major order and stores down-column
nonzero counts.  Every coordinate row is padded to one or more complete
64-byte cache lines, matching
``native_bitnet_dip_physical_accounting`` exactly.

Format version 2 also authenticates the selected joint routing policy.  The
policy is intentionally small and fixed:

* one input-coordinate count per layer (currently common across layers);
* one candidate count and maximum adaptive top-K per layer;
* one minimum adaptive top-K and energy target.

Format version 2 embeds the complete frozen RMS policy instead of deriving it
from a layer number in the runtime.  The approved policy report must provide
one explicit estimator/audit record per layer, and those fields are included
in each layer checksum.

The binary format is fail closed.  Dimensions, byte order, encodings,
directory offsets, policy bounds, padding, canonical base-3 tails, and
per-layer SHA-256 digests are all checked before a memory-mapped view is
returned.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from engram.models.native_bitnet import (
    NativeBitNetValidationError,
    load_native_bitnet_artifact,
    pack_base3_rows,
    unpack_base3_rows,
)
from engram.utils import sha256_file


NATIVE_BITNET_DIP_INDEX_FORMAT = "engram-native-bitnet-dip-index"
NATIVE_BITNET_DIP_INDEX_VERSION = 2

_MAGIC = b"ENGBDI12"
_LAYER_MAGIC = b"ENGBDIL2"
_ENDIAN_MARKER = 0x01020304
_VERSION = NATIVE_BITNET_DIP_INDEX_VERSION
_CACHE_LINE_BYTES = 64
_TRITS_PER_BYTE = 5
_COORDINATE_ENCODING_BASE3_U8 = 1
_NORM_DTYPE_LITTLE_UINT16 = 1
_NORM_DTYPE_LITTLE_UINT32 = 2
_CHECKSUM_SHA256 = 1
_HEADER = struct.Struct("<8s14I")
_DIRECTORY_ENTRY = struct.Struct("<4I2Q")
_LAYER_HEADER = struct.Struct("<8sIf32s7I")
_HEADER_BYTES = 128
_DIRECTORY_ENTRY_BYTES = 32
_LAYER_HEADER_BYTES = 128
_POLICY_CHECKSUM = struct.Struct("<8If")
_RMS_ESTIMATOR_CORRECTED_PROXY = 1
_RMS_ESTIMATOR_CANDIDATE_RATIO = 2
_RMS_AUDIT_NONE = 0
_RMS_AUDIT_TOP_PROXY_RAW_SQUARE = 2
_MAX_MODEL_DIMENSION = 1 << 20
_MAX_LAYER_COUNT = 1 << 12


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _integer(value: Any, name: str, *, maximum: int | None = None) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or (maximum is not None and value > maximum)
    ):
        suffix = "" if maximum is None else f" and at most {maximum}"
        raise NativeBitNetValidationError(
            f"{name} must be a positive integer{suffix}"
        )
    return value


def _layout(
    hidden_size: int,
    intermediate_size: int,
    layer_count: int,
) -> dict[str, int]:
    hidden = _integer(
        hidden_size,
        "hidden_size",
        maximum=_MAX_MODEL_DIMENSION,
    )
    intermediate = _integer(
        intermediate_size,
        "intermediate_size",
        maximum=_MAX_MODEL_DIMENSION,
    )
    layers = _integer(
        layer_count,
        "layer_count",
        maximum=_MAX_LAYER_COUNT,
    )
    coordinate_payload = math.ceil(intermediate / _TRITS_PER_BYTE)
    coordinate_stride = _align(coordinate_payload, _CACHE_LINE_BYTES)
    coordinate_stream = hidden * coordinate_stride
    norm_value_bytes = 2 if hidden <= np.iinfo(np.uint16).max else 4
    norm_payload = intermediate * norm_value_bytes
    norm_stream = _align(norm_payload, _CACHE_LINE_BYTES)
    gate_offset = _LAYER_HEADER_BYTES
    up_offset = gate_offset + coordinate_stream
    norm_offset = up_offset + coordinate_stream
    layer_payload = norm_offset + norm_stream
    layer_block = _align(layer_payload, _CACHE_LINE_BYTES)
    directory_block = _align(
        layers * _DIRECTORY_ENTRY_BYTES,
        _CACHE_LINE_BYTES,
    )
    serialized = _HEADER_BYTES + directory_block + layers * layer_block
    return {
        "coordinate_payload_bytes": coordinate_payload,
        "coordinate_stride_bytes": coordinate_stride,
        "coordinate_stream_bytes": coordinate_stream,
        "norm_value_bytes": norm_value_bytes,
        "norm_payload_bytes": norm_payload,
        "norm_stream_bytes": norm_stream,
        "gate_stream_offset": gate_offset,
        "up_stream_offset": up_offset,
        "down_norm_stream_offset": norm_offset,
        "layer_payload_bytes": layer_payload,
        "layer_block_bytes": layer_block,
        "directory_block_bytes": directory_block,
        "serialized_artifact_bytes": serialized,
    }


@dataclass(frozen=True)
class NativeBitNetDIPPolicy:
    """Validated policy for one indexed MLP layer."""

    input_coordinates: int
    candidate_count: int
    minimum_top_k: int
    maximum_top_k: int
    energy_target: float
    rms_audit_count: int
    rms_estimator: str
    rms_audit_strategy: str


def _rms_policy_codes(policy: NativeBitNetDIPPolicy) -> tuple[int, int]:
    estimator = {
        "corrected_proxy": _RMS_ESTIMATOR_CORRECTED_PROXY,
        "candidate_ratio": _RMS_ESTIMATOR_CANDIDATE_RATIO,
    }.get(policy.rms_estimator)
    audit = {
        "none": _RMS_AUDIT_NONE,
        "top_proxy_raw_square": _RMS_AUDIT_TOP_PROXY_RAW_SQUARE,
    }.get(policy.rms_audit_strategy)
    if estimator is None or audit is None:
        raise NativeBitNetValidationError(
            "native BitNet DIP RMS policy is unsupported"
        )
    return estimator, audit


@dataclass(frozen=True)
class MappedNativeBitNetDIPLayer:
    """Read-only mmap views and routing policy for one layer."""

    layer: int
    offset: int
    block_bytes: int
    gate_stream_offset: int
    up_stream_offset: int
    down_norm_stream_offset: int
    gate_coordinates: NDArray[np.uint8]
    up_coordinates: NDArray[np.uint8]
    down_norm_squared: NDArray[np.unsignedinteger[Any]]
    policy: NativeBitNetDIPPolicy
    payload_sha256: str


def _policy_bytes(
    layer: int,
    policy: NativeBitNetDIPPolicy,
) -> bytes:
    estimator, audit_strategy = _rms_policy_codes(policy)
    return _POLICY_CHECKSUM.pack(
        layer,
        policy.input_coordinates,
        policy.candidate_count,
        policy.minimum_top_k,
        policy.maximum_top_k,
        policy.rms_audit_count,
        estimator,
        audit_strategy,
        float(policy.energy_target),
    )


def _layer_digest(
    layer: int,
    policy: NativeBitNetDIPPolicy,
    payload: memoryview | bytes | bytearray,
) -> bytes:
    digest = hashlib.sha256(_policy_bytes(layer, policy))
    digest.update(payload)
    return digest.digest()


def _read_selected_policy(
    path: Path,
    *,
    artifact_sha256: str,
    hidden_size: int,
    intermediate_size: int,
    layer_count: int,
) -> tuple[NativeBitNetDIPPolicy, ...]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeBitNetValidationError(
            f"invalid native BitNet DIP policy report: {exc}"
        ) from exc
    if not isinstance(report, dict):
        raise NativeBitNetValidationError(
            "native BitNet DIP policy report must be a JSON object"
        )
    if report.get("format") == "engram-native-bitnet-dip-policy":
        # A frozen manifest is reconstructed from all of its bound evidence,
        # including the already-built source-bound index, before it may be
        # consumed to reproduce that index for a package/final runner.
        from engram.semantic.native_bitnet_dip_policy_manifest import (
            load_native_bitnet_dip_policy_manifest,
        )

        try:
            frozen = load_native_bitnet_dip_policy_manifest(path)
        except ValueError as exc:
            raise NativeBitNetValidationError(
                f"native BitNet DIP frozen policy is invalid: {exc}"
            ) from exc
        bound_artifact = frozen.payload["bindings"]["base_record_artifact"]
        if bound_artifact.get("sha256") != artifact_sha256:
            raise NativeBitNetValidationError(
                "native BitNet DIP policy source-artifact SHA-256 mismatch"
            )
        if len(frozen.layers) != layer_count:
            raise NativeBitNetValidationError(
                "native BitNet DIP frozen policy has the wrong layer count"
            )
        policies = tuple(
            NativeBitNetDIPPolicy(
                input_coordinates=layer.input_coordinates,
                candidate_count=layer.candidate_count,
                minimum_top_k=layer.minimum_top_k,
                maximum_top_k=layer.maximum_top_k,
                energy_target=layer.energy_target,
                rms_audit_count=layer.rms_audit_count,
                rms_estimator=layer.rms_estimator,
                rms_audit_strategy=layer.rms_audit_strategy,
            )
            for layer in frozen.layers
        )
        if any(
            policy.input_coordinates > hidden_size
            or policy.candidate_count > intermediate_size
            or policy.maximum_top_k > intermediate_size
            for policy in policies
        ):
            raise NativeBitNetValidationError(
                "native BitNet DIP frozen policy dimensions are inconsistent"
            )
        return policies
    if (
        report.get("experiment")
        != "native_bitnet_dip_joint_candidate_adaptive_k_policy"
        or report.get("decision")
        != "use_joint_policy_for_candidate_only_causal_development"
        or report.get("progression_screen", {}).get("passed") is not True
    ):
        raise NativeBitNetValidationError(
            "native BitNet DIP policy is not an approved selected joint policy"
        )
    if report.get("artifact_sha256") != artifact_sha256:
        raise NativeBitNetValidationError(
            "native BitNet DIP policy source-artifact SHA-256 mismatch"
        )
    configuration = report.get("configuration")
    selected = report.get("selected_policy")
    if not isinstance(configuration, dict) or not isinstance(selected, dict):
        raise NativeBitNetValidationError(
            "native BitNet DIP policy report has no selected policy"
        )
    input_coordinates = _integer(
        configuration.get("input_coordinates"),
        "input_coordinates",
        maximum=hidden_size,
    )
    input_fraction = configuration.get("input_fraction")
    if (
        isinstance(input_fraction, bool)
        or not isinstance(input_fraction, (int, float))
        or not math.isfinite(float(input_fraction))
        or not 0 < float(input_fraction) <= 1
        or math.ceil(float(input_fraction) * hidden_size) != input_coordinates
    ):
        raise NativeBitNetValidationError(
            "native BitNet DIP input fraction/count are inconsistent"
        )
    minimum_top_k = _integer(
        configuration.get("minimum_k"),
        "minimum_k",
        maximum=intermediate_size,
    )
    energy_target = configuration.get("energy_target")
    if (
        isinstance(energy_target, bool)
        or not isinstance(energy_target, (int, float))
        or not math.isfinite(float(energy_target))
        or not 0 < float(energy_target) <= 1
    ):
        raise NativeBitNetValidationError(
            "native BitNet DIP energy_target must lie in (0, 1]"
        )
    candidate_counts = selected.get("candidate_counts")
    maximum_top_ks = selected.get("maximum_ks")
    rms_policies = selected.get("rms_policies")
    if (
        not isinstance(candidate_counts, list)
        or not isinstance(maximum_top_ks, list)
        or not isinstance(rms_policies, list)
        or len(candidate_counts) != layer_count
        or len(maximum_top_ks) != layer_count
        or len(rms_policies) != layer_count
    ):
        raise NativeBitNetValidationError(
            "native BitNet DIP selected schedules/RMS policies have the "
            "wrong layer count"
        )
    result: list[NativeBitNetDIPPolicy] = []
    for layer, (candidate_value, maximum_value, rms_value) in enumerate(
        zip(candidate_counts, maximum_top_ks, rms_policies, strict=True)
    ):
        candidate = _integer(
            candidate_value,
            f"candidate_counts[{layer}]",
            maximum=intermediate_size,
        )
        maximum = _integer(
            maximum_value,
            f"maximum_ks[{layer}]",
            maximum=intermediate_size,
        )
        if minimum_top_k > maximum or maximum > candidate:
            raise NativeBitNetValidationError(
                f"native BitNet DIP layer {layer} policy bounds are inconsistent"
            )
        if not isinstance(rms_value, dict):
            raise NativeBitNetValidationError(
                f"native BitNet DIP rms_policies[{layer}] must be an object"
            )
        rms_audit_count_value = rms_value.get("audit_count")
        if (
            isinstance(rms_audit_count_value, bool)
            or not isinstance(rms_audit_count_value, int)
            or rms_audit_count_value < 0
            or rms_audit_count_value > candidate
        ):
            raise NativeBitNetValidationError(
                f"native BitNet DIP rms_policies[{layer}].audit_count is invalid"
            )
        rms_audit_count = rms_audit_count_value
        rms_estimator = rms_value.get("estimator")
        rms_audit_strategy = rms_value.get("audit_strategy")
        if (
            rms_estimator not in {"candidate_ratio", "corrected_proxy"}
            or rms_audit_strategy
            not in {"none", "top_proxy_raw_square"}
            or (rms_estimator == "candidate_ratio" and rms_audit_count != 0)
            or (
                rms_estimator == "corrected_proxy"
                and (
                    rms_audit_count == 0
                    or rms_audit_strategy != "top_proxy_raw_square"
                )
            )
            or (rms_audit_count == 0 and rms_audit_strategy != "none")
        ):
            raise NativeBitNetValidationError(
                f"native BitNet DIP rms_policies[{layer}] is inconsistent"
            )
        if maximum > candidate - rms_audit_count:
            raise NativeBitNetValidationError(
                f"native BitNet DIP layer {layer} maximum K exceeds its "
                "non-audit candidate budget"
            )
        result.append(
            NativeBitNetDIPPolicy(
                input_coordinates=input_coordinates,
                candidate_count=candidate,
                minimum_top_k=minimum_top_k,
                maximum_top_k=maximum,
                energy_target=float(energy_target),
                rms_audit_count=rms_audit_count,
                rms_estimator=rms_estimator,
                rms_audit_strategy=rms_audit_strategy,
            )
        )
    return tuple(result)


def build_native_bitnet_dip_index(
    artifact: str | Path,
    policy_report: str | Path,
    out: str | Path,
) -> Path:
    """Build a coordinate-major DIP index from a validated native artifact."""

    artifact_path = Path(artifact).resolve()
    policy_path = Path(policy_report).resolve()
    destination = Path(out).resolve()
    if destination == artifact_path:
        raise NativeBitNetValidationError(
            "native BitNet DIP index must not overwrite its source artifact"
        )
    source = load_native_bitnet_artifact(artifact_path)
    if source.cache_line_bytes != _CACHE_LINE_BYTES:
        raise NativeBitNetValidationError(
            "native BitNet DIP index v2 requires 64-byte source alignment"
        )
    policies = _read_selected_policy(
        policy_path,
        artifact_sha256=source.payload_sha256,
        hidden_size=source.hidden_size,
        intermediate_size=source.intermediate_size,
        layer_count=len(source.layers),
    )
    layout = _layout(
        source.hidden_size,
        source.intermediate_size,
        len(source.layers),
    )
    norm_dtype = (
        np.dtype("<u2")
        if layout["norm_value_bytes"] == 2
        else np.dtype("<u4")
    )
    norm_dtype_code = (
        _NORM_DTYPE_LITTLE_UINT16
        if norm_dtype.itemsize == 2
        else _NORM_DTYPE_LITTLE_UINT32
    )
    header_core = _HEADER.pack(
        _MAGIC,
        _VERSION,
        _ENDIAN_MARKER,
        _HEADER_BYTES,
        _DIRECTORY_ENTRY_BYTES,
        layout["directory_block_bytes"],
        _LAYER_HEADER_BYTES,
        _CACHE_LINE_BYTES,
        source.hidden_size,
        source.intermediate_size,
        len(source.layers),
        _TRITS_PER_BYTE,
        _COORDINATE_ENCODING_BASE3_U8,
        norm_dtype_code,
        _CHECKSUM_SHA256,
    )
    header = (
        header_core
        + bytes.fromhex(source.payload_sha256)
        + bytes(_HEADER_BYTES - len(header_core) - 32)
    )
    directory = bytearray(layout["directory_block_bytes"])
    first_layer_offset = _HEADER_BYTES + layout["directory_block_bytes"]
    for layer, policy in enumerate(policies):
        offset = first_layer_offset + layer * layout["layer_block_bytes"]
        _DIRECTORY_ENTRY.pack_into(
            directory,
            layer * _DIRECTORY_ENTRY_BYTES,
            layer,
            policy.input_coordinates,
            policy.candidate_count,
            policy.maximum_top_k,
            offset,
            layout["layer_block_bytes"],
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(header)
            handle.write(directory)
            for layer, (packed, policy) in enumerate(
                zip(source.layers, policies, strict=True)
            ):
                gate = unpack_base3_rows(
                    packed.gate_records,
                    logical_width=source.hidden_size,
                )
                up = unpack_base3_rows(
                    packed.up_records,
                    logical_width=source.hidden_size,
                )
                down = unpack_base3_rows(
                    packed.down_records,
                    logical_width=source.hidden_size,
                )
                gate_packed = pack_base3_rows(gate.T)
                up_packed = pack_base3_rows(up.T)
                down_norm = np.count_nonzero(down, axis=1).astype(
                    norm_dtype,
                    copy=False,
                )
                body = bytearray(
                    layout["layer_block_bytes"] - _LAYER_HEADER_BYTES
                )
                coordinate_payload = layout["coordinate_payload_bytes"]
                coordinate_stride = layout["coordinate_stride_bytes"]
                coordinate_stream = layout["coordinate_stream_bytes"]
                for rows, stream_offset in (
                    (gate_packed, layout["gate_stream_offset"]),
                    (up_packed, layout["up_stream_offset"]),
                ):
                    if rows.shape != (
                        source.hidden_size,
                        coordinate_payload,
                    ):
                        raise AssertionError(
                            "coordinate-major packing has an unexpected shape"
                        )
                    relative = stream_offset - _LAYER_HEADER_BYTES
                    padded = np.zeros(
                        (source.hidden_size, coordinate_stride),
                        dtype=np.uint8,
                    )
                    padded[:, :coordinate_payload] = rows
                    stream_bytes = padded.tobytes()
                    if len(stream_bytes) != coordinate_stream:
                        raise AssertionError(
                            "coordinate stream differs from physical accounting"
                        )
                    body[relative : relative + coordinate_stream] = stream_bytes
                norm_relative = (
                    layout["down_norm_stream_offset"] - _LAYER_HEADER_BYTES
                )
                norm_bytes = down_norm.tobytes()
                if len(norm_bytes) != layout["norm_payload_bytes"]:
                    raise AssertionError(
                        "down-norm stream differs from physical accounting"
                    )
                body[norm_relative : norm_relative + len(norm_bytes)] = norm_bytes
                checksum = _layer_digest(layer, policy, body)
                estimator, audit_strategy = _rms_policy_codes(policy)
                layer_header_core = _LAYER_HEADER.pack(
                    _LAYER_MAGIC,
                    policy.minimum_top_k,
                    float(policy.energy_target),
                    checksum,
                    layout["gate_stream_offset"],
                    layout["up_stream_offset"],
                    layout["down_norm_stream_offset"],
                    layout["layer_payload_bytes"],
                    policy.rms_audit_count,
                    estimator,
                    audit_strategy,
                )
                handle.write(
                    layer_header_core
                    + bytes(_LAYER_HEADER_BYTES - len(layer_header_core))
                )
                handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size != layout["serialized_artifact_bytes"]:
            raise AssertionError(
                "serialized DIP index differs from physical accounting"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    # Do not publish an artifact that this implementation cannot reload.
    with load_native_bitnet_dip_index(destination):
        pass
    try:
        policy_payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        destination.unlink(missing_ok=True)
        raise NativeBitNetValidationError(
            f"cannot re-read native BitNet DIP policy report: {exc}"
        ) from exc
    if policy_payload.get("format") == "engram-native-bitnet-dip-policy":
        expected_sha256 = (
            policy_payload.get("bindings", {})
            .get("coordinate_index", {})
            .get("sha256")
        )
        if (
            not isinstance(expected_sha256, str)
            or sha256_file(destination) != expected_sha256
        ):
            destination.unlink(missing_ok=True)
            raise NativeBitNetValidationError(
                "rebuilt native BitNet DIP index differs from its frozen hash"
            )
    return destination


class LoadedNativeBitNetDIPIndex:
    """Owner of a validated read-only mmap and its typed layer views."""

    def __init__(
        self,
        path: Path,
        handle: Any,
        mapping: mmap.mmap,
        *,
        hidden_size: int,
        intermediate_size: int,
        cache_line_bytes: int,
        directory_block_bytes: int,
        layers: tuple[MappedNativeBitNetDIPLayer, ...],
        payload_sha256: str,
        source_artifact_sha256: str,
    ) -> None:
        self.path = path
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.cache_line_bytes = cache_line_bytes
        self.directory_block_bytes = directory_block_bytes
        self.layers = layers
        self.serialized_artifact_bytes = len(mapping)
        self.payload_sha256 = payload_sha256
        self.source_artifact_sha256 = source_artifact_sha256
        self._handle = handle
        self._mapping = mapping

    @property
    def closed(self) -> bool:
        return self._mapping.closed

    def metadata(self) -> dict[str, Any]:
        return {
            "format": NATIVE_BITNET_DIP_INDEX_FORMAT,
            "version": _VERSION,
            "endianness": "little",
            "endianness_marker": _ENDIAN_MARKER,
            "coordinate_dtype": "uint8",
            "coordinate_encoding": "base3_five_trits_per_byte",
            "down_norm_dtype": (
                "little_endian_uint16"
                if self.hidden_size <= np.iinfo(np.uint16).max
                else "little_endian_uint32"
            ),
            "checksum": "sha256_per_layer_data_and_policy",
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "layer_count": len(self.layers),
            "cache_line_bytes": self.cache_line_bytes,
            "directory_block_bytes": self.directory_block_bytes,
            "serialized_artifact_bytes": self.serialized_artifact_bytes,
            "payload_sha256": self.payload_sha256,
            "source_artifact_sha256": self.source_artifact_sha256,
            "layers": [
                {
                    "layer": layer.layer,
                    "offset": layer.offset,
                    "block_bytes": layer.block_bytes,
                    "gate_stream_offset": layer.gate_stream_offset,
                    "up_stream_offset": layer.up_stream_offset,
                    "down_norm_stream_offset": (
                        layer.down_norm_stream_offset
                    ),
                    "payload_sha256": layer.payload_sha256,
                    "policy": {
                        "input_coordinates": layer.policy.input_coordinates,
                        "candidate_count": layer.policy.candidate_count,
                        "minimum_top_k": layer.policy.minimum_top_k,
                        "maximum_top_k": layer.policy.maximum_top_k,
                        "energy_target": layer.policy.energy_target,
                        "rms_audit_count": layer.policy.rms_audit_count,
                        "rms_estimator": layer.policy.rms_estimator,
                        "rms_audit_strategy": layer.policy.rms_audit_strategy,
                    },
                }
                for layer in self.layers
            ],
        }

    def close(self) -> None:
        if self.closed:
            return
        # Drop our NumPy exports before closing the mmap.  A caller retaining
        # a layer array must release it before closing and receives a clear
        # error instead of silently keeping an unauthenticated mapping alive.
        self.layers = ()
        try:
            self._mapping.close()
        except BufferError as exc:
            raise RuntimeError(
                "cannot close DIP index while mapped array views are retained"
            ) from exc
        finally:
            self._handle.close()

    def __enter__(self) -> "LoadedNativeBitNetDIPIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def load_native_bitnet_dip_index(
    path: str | Path,
) -> LoadedNativeBitNetDIPIndex:
    """Validate and memory-map a native BitNet DIP index."""

    source = Path(path).resolve()
    try:
        handle = source.open("rb")
    except OSError as exc:
        raise NativeBitNetValidationError(
            f"cannot open native BitNet DIP index {source}"
        ) from exc
    mapping: mmap.mmap | None = None
    view: memoryview | None = None
    layers: list[MappedNativeBitNetDIPLayer] = []
    gate_rows: NDArray[np.uint8] | None = None
    up_rows: NDArray[np.uint8] | None = None
    norms: NDArray[np.unsignedinteger[Any]] | None = None
    padded: NDArray[np.uint8] | None = None
    rows: NDArray[np.uint8] | None = None
    try:
        mapping = mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ)
        if len(mapping) < _HEADER_BYTES:
            raise NativeBitNetValidationError(
                "native BitNet DIP index is shorter than its header"
            )
        (
            magic,
            version,
            endian_marker,
            header_bytes,
            directory_entry_bytes,
            directory_block_bytes,
            layer_header_bytes,
            cache_line_bytes,
            hidden_size,
            intermediate_size,
            layer_count,
            trits_per_byte,
            coordinate_encoding,
            norm_dtype_code,
            checksum_code,
        ) = _HEADER.unpack_from(mapping)
        if magic != _MAGIC or version != _VERSION:
            raise NativeBitNetValidationError(
                "native BitNet DIP index magic/version mismatch"
            )
        if (
            endian_marker != _ENDIAN_MARKER
            or header_bytes != _HEADER_BYTES
            or directory_entry_bytes != _DIRECTORY_ENTRY_BYTES
            or layer_header_bytes != _LAYER_HEADER_BYTES
            or cache_line_bytes != _CACHE_LINE_BYTES
            or trits_per_byte != _TRITS_PER_BYTE
            or coordinate_encoding != _COORDINATE_ENCODING_BASE3_U8
            or checksum_code != _CHECKSUM_SHA256
        ):
            raise NativeBitNetValidationError(
                "native BitNet DIP index metadata is unsupported"
            )
        source_artifact_sha256 = mapping[64:96].hex()
        if source_artifact_sha256 == "0" * 64 or any(
            mapping[96:_HEADER_BYTES]
        ):
            raise NativeBitNetValidationError(
                "native BitNet DIP source binding/header padding is invalid"
            )
        layout = _layout(hidden_size, intermediate_size, layer_count)
        expected_norm_code = (
            _NORM_DTYPE_LITTLE_UINT16
            if layout["norm_value_bytes"] == 2
            else _NORM_DTYPE_LITTLE_UINT32
        )
        if (
            norm_dtype_code != expected_norm_code
            or directory_block_bytes != layout["directory_block_bytes"]
            or len(mapping) != layout["serialized_artifact_bytes"]
        ):
            raise NativeBitNetValidationError(
                "native BitNet DIP index length/dtype is inconsistent"
            )
        directory_payload_end = (
            _HEADER_BYTES + layer_count * _DIRECTORY_ENTRY_BYTES
        )
        directory_end = _HEADER_BYTES + directory_block_bytes
        if any(mapping[directory_payload_end:directory_end]):
            raise NativeBitNetValidationError(
                "native BitNet DIP directory padding is non-zero"
            )
        norm_dtype = (
            np.dtype("<u2")
            if layout["norm_value_bytes"] == 2
            else np.dtype("<u4")
        )
        first_layer_offset = directory_end
        view = memoryview(mapping)
        for layer in range(layer_count):
            (
                entry_layer,
                input_coordinates,
                candidate_count,
                maximum_top_k,
                offset,
                block_bytes,
            ) = _DIRECTORY_ENTRY.unpack_from(
                mapping,
                _HEADER_BYTES + layer * _DIRECTORY_ENTRY_BYTES,
            )
            expected_offset = (
                first_layer_offset + layer * layout["layer_block_bytes"]
            )
            if (
                entry_layer != layer
                or input_coordinates <= 0
                or input_coordinates > hidden_size
                or candidate_count <= 0
                or candidate_count > intermediate_size
                or maximum_top_k <= 0
                or maximum_top_k > candidate_count
                or offset != expected_offset
                or offset % _CACHE_LINE_BYTES
                or block_bytes != layout["layer_block_bytes"]
            ):
                raise NativeBitNetValidationError(
                    "native BitNet DIP directory entry is invalid"
                )
            (
                layer_magic,
                minimum_top_k,
                energy_target,
                expected_digest,
                gate_offset,
                up_offset,
                norm_offset,
                payload_bytes,
                rms_audit_count,
                rms_estimator_code,
                rms_audit_strategy_code,
            ) = _LAYER_HEADER.unpack_from(mapping, offset)
            rms_estimator = {
                _RMS_ESTIMATOR_CORRECTED_PROXY: "corrected_proxy",
                _RMS_ESTIMATOR_CANDIDATE_RATIO: "candidate_ratio",
            }.get(rms_estimator_code)
            rms_audit_strategy = {
                _RMS_AUDIT_NONE: "none",
                _RMS_AUDIT_TOP_PROXY_RAW_SQUARE: "top_proxy_raw_square",
            }.get(rms_audit_strategy_code)
            if (
                layer_magic != _LAYER_MAGIC
                or minimum_top_k <= 0
                or minimum_top_k > maximum_top_k
                or not math.isfinite(float(energy_target))
                or not 0 < float(energy_target) <= 1
                or gate_offset != layout["gate_stream_offset"]
                or up_offset != layout["up_stream_offset"]
                or norm_offset != layout["down_norm_stream_offset"]
                or payload_bytes != layout["layer_payload_bytes"]
                or rms_estimator is None
                or rms_audit_strategy is None
                or rms_audit_count > candidate_count
                or maximum_top_k > candidate_count - rms_audit_count
                or (
                    rms_estimator == "candidate_ratio"
                    and rms_audit_count != 0
                )
                or (
                    rms_estimator == "corrected_proxy"
                    and (
                        rms_audit_count == 0
                        or rms_audit_strategy != "top_proxy_raw_square"
                    )
                )
                or (
                    rms_audit_count == 0
                    and rms_audit_strategy != "none"
                )
            ):
                raise NativeBitNetValidationError(
                    "native BitNet DIP layer header/policy is invalid"
                )
            policy = NativeBitNetDIPPolicy(
                input_coordinates=input_coordinates,
                candidate_count=candidate_count,
                minimum_top_k=minimum_top_k,
                maximum_top_k=maximum_top_k,
                energy_target=float(energy_target),
                rms_audit_count=rms_audit_count,
                rms_estimator=rms_estimator,
                rms_audit_strategy=rms_audit_strategy,
            )
            if any(
                mapping[
                    offset + _LAYER_HEADER.size :
                    offset + _LAYER_HEADER_BYTES
                ]
            ):
                raise NativeBitNetValidationError(
                    "native BitNet DIP layer-header padding is non-zero"
                )
            body_start = offset + _LAYER_HEADER_BYTES
            body_end = offset + block_bytes
            actual_digest = _layer_digest(
                layer,
                policy,
                view[body_start:body_end],
            )
            if actual_digest != expected_digest:
                raise NativeBitNetValidationError(
                    f"native BitNet DIP layer {layer} checksum mismatch"
                )
            coordinate_payload = layout["coordinate_payload_bytes"]
            coordinate_stride = layout["coordinate_stride_bytes"]
            gate_rows = np.ndarray(
                shape=(hidden_size, coordinate_payload),
                dtype=np.uint8,
                buffer=mapping,
                offset=offset + gate_offset,
                strides=(coordinate_stride, 1),
            )
            up_rows = np.ndarray(
                shape=(hidden_size, coordinate_payload),
                dtype=np.uint8,
                buffer=mapping,
                offset=offset + up_offset,
                strides=(coordinate_stride, 1),
            )
            for name, rows, stream_offset in (
                ("gate", gate_rows, gate_offset),
                ("up", up_rows, up_offset),
            ):
                if np.any(rows > 242):
                    raise NativeBitNetValidationError(
                        f"native BitNet DIP {name} stream has invalid base-3 bytes"
                    )
                tail_digits = intermediate_size % _TRITS_PER_BYTE
                if tail_digits and np.any(
                    rows[:, -1] >= 3**tail_digits
                ):
                    raise NativeBitNetValidationError(
                        f"native BitNet DIP {name} stream has noncanonical tails"
                    )
                padding_start = coordinate_payload
                padding_bytes = coordinate_stride - coordinate_payload
                if padding_bytes:
                    padded = np.ndarray(
                        shape=(hidden_size, padding_bytes),
                        dtype=np.uint8,
                        buffer=mapping,
                        offset=offset + stream_offset + padding_start,
                        strides=(coordinate_stride, 1),
                    )
                    if np.any(padded):
                        raise NativeBitNetValidationError(
                            f"native BitNet DIP {name} row padding is non-zero"
                        )
            norms = np.ndarray(
                shape=(intermediate_size,),
                dtype=norm_dtype,
                buffer=mapping,
                offset=offset + norm_offset,
            )
            if np.any(norms > hidden_size):
                raise NativeBitNetValidationError(
                    "native BitNet DIP down norms exceed the hidden width"
                )
            norm_padding_start = (
                offset + norm_offset + layout["norm_payload_bytes"]
            )
            norm_padding_end = (
                offset + norm_offset + layout["norm_stream_bytes"]
            )
            if any(mapping[norm_padding_start:norm_padding_end]):
                raise NativeBitNetValidationError(
                    "native BitNet DIP down-norm padding is non-zero"
                )
            layers.append(
                MappedNativeBitNetDIPLayer(
                    layer=layer,
                    offset=offset,
                    block_bytes=block_bytes,
                    gate_stream_offset=gate_offset,
                    up_stream_offset=up_offset,
                    down_norm_stream_offset=norm_offset,
                    gate_coordinates=gate_rows,
                    up_coordinates=up_rows,
                    down_norm_squared=norms,
                    policy=policy,
                    payload_sha256=actual_digest.hex(),
                )
            )
        payload_sha256 = hashlib.sha256(view).hexdigest()
        view.release()
        view = None
        return LoadedNativeBitNetDIPIndex(
            source,
            handle,
            mapping,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            cache_line_bytes=cache_line_bytes,
            directory_block_bytes=directory_block_bytes,
            layers=tuple(layers),
            payload_sha256=payload_sha256,
            source_artifact_sha256=source_artifact_sha256,
        )
    except Exception:
        if view is not None:
            view.release()
        layers.clear()
        gate_rows = None
        up_rows = None
        norms = None
        padded = None
        rows = None
        if mapping is not None:
            mapping.close()
        handle.close()
        raise


__all__ = [
    "LoadedNativeBitNetDIPIndex",
    "MappedNativeBitNetDIPLayer",
    "NATIVE_BITNET_DIP_INDEX_FORMAT",
    "NATIVE_BITNET_DIP_INDEX_VERSION",
    "NativeBitNetDIPPolicy",
    "build_native_bitnet_dip_index",
    "load_native_bitnet_dip_index",
]
