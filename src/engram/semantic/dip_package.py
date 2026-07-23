from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.models.inspection import inspect_model, load_layer_mlp
from engram.semantic.dip import input_coordinate_count, stable_top_k
from engram.semantic.swiglu import silu
from engram.utils import atomic_json, npy_file_metadata, sha256_file


DIP_PACKAGE_FORMAT = "engram-dip-experimental"
DIP_PACKAGE_VERSION = 3
CACHE_LINE_BYTES = 64


@dataclass(frozen=True)
class SerializedDIPMetrics:
    selected_input_coordinates: int
    partial_projection_bytes: int
    candidate_completion_bytes: int
    selected_down_bytes: int
    logical_weight_bytes: int
    cache_line_weight_bytes: int
    dense_weight_bytes: int

    @property
    def logical_fraction_of_dense(self) -> float:
        return self.logical_weight_bytes / self.dense_weight_bytes

    @property
    def cache_line_fraction_of_dense(self) -> float:
        return self.cache_line_weight_bytes / self.dense_weight_bytes


@dataclass(frozen=True)
class SerializedDIPRead:
    output: NDArray[np.float32]
    input_coordinates: NDArray[np.int64]
    candidate_indices: NDArray[np.int64]
    selected_indices: NDArray[np.int64]
    selected_activations: NDArray[np.float32]
    metrics: SerializedDIPMetrics


def _descriptor(path: Path) -> dict[str, object]:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **npy_file_metadata(path),
    }


def write_serialized_dip_layer(
    target: str | Path,
    gate: ArrayLike,
    up: ArrayLike,
    down: ArrayLike,
    *,
    dual_layout: bool = False,
) -> Path:
    """Write one mmap-friendly DIP layer.

    The default v2 format stores gate/up as ``[coordinate, record]``. The
    rejected v3 diagnostic can be requested explicitly to duplicate gate/up
    as ``[record, coordinate]`` for record-major completion experiments.
    """

    gate_array = np.asarray(gate, dtype="<f4")
    up_array = np.asarray(up, dtype="<f4")
    down_array = np.asarray(down, dtype="<f4")
    if gate_array.ndim != 2 or gate_array.shape != up_array.shape:
        raise ValueError("gate and up must have the same shape [records, hidden]")
    records, hidden = gate_array.shape
    if records == 0 or hidden == 0 or down_array.shape != (hidden, records):
        raise ValueError(
            "down must have shape [hidden, records] and dimensions must be positive"
        )
    if not (
        np.all(np.isfinite(gate_array))
        and np.all(np.isfinite(up_array))
        and np.all(np.isfinite(down_array))
    ):
        raise ValueError("DIP weights must contain only finite values")

    directory = Path(target)
    directory.mkdir(parents=True, exist_ok=True)
    version = DIP_PACKAGE_VERSION if dual_layout else 2
    arrays: dict[str, NDArray[np.generic]] = {
        "config": np.asarray(
            [version, records, hidden, CACHE_LINE_BYTES], dtype="<u4"
        ),
        "gate_coordinates": np.ascontiguousarray(gate_array.T),
        "up_coordinates": np.ascontiguousarray(up_array.T),
        "down_rows": np.ascontiguousarray(down_array.T),
        "value_norms": np.asarray(np.linalg.norm(down_array, axis=0), dtype="<f4"),
    }
    if dual_layout:
        arrays.update(
            gate_rows=np.ascontiguousarray(gate_array),
            up_rows=np.ascontiguousarray(up_array),
        )
    files: dict[str, dict[str, object]] = {}
    for name, array in arrays.items():
        path = directory / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        files[name] = _descriptor(path)
    atomic_json(
        directory / "metadata.json",
        {
            "format": DIP_PACKAGE_FORMAT,
            "version": version,
            "experimental": True,
            "records": records,
            "hidden_size": hidden,
            "cache_line_bytes": CACHE_LINE_BYTES,
            "dtype": "float32",
            "gate_up_layout": (
                "dual_coordinate_record_and_record_coordinate"
                if dual_layout
                else "coordinate_record"
            ),
            "dual_layout_diagnostic": dual_layout,
            "down_layout": "record_output",
            "selection": "descending_abs_hidden_stable_coordinate_index",
            "accounting": {
                "logical_scalar_reads": True,
                "cache_line_amplification_estimate": True,
                "measured_dram_bytes": False,
            },
            "files": files,
        },
    )
    return directory


def build_serialized_dip_package(
    model: str | Path,
    out: str | Path,
    *,
    layers: Iterable[int] | None = None,
    dual_layout: bool = False,
) -> Path:
    inspection = inspect_model(model)
    requested = list(range(inspection.num_hidden_layers)) if layers is None else list(layers)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("layers must be a non-empty sequence without duplicates")
    if any(layer < 0 or layer >= inspection.num_hidden_layers for layer in requested):
        raise ValueError("layer index is outside the model")
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    descriptors = []
    for layer in requested:
        gate, up, down = load_layer_mlp(inspection.model_path, layer)
        layer_dir = write_serialized_dip_layer(
            target / f"layer-{layer:04d}",
            gate,
            up,
            down,
            dual_layout=dual_layout,
        )
        descriptors.append(
            {
                "layer": layer,
                "directory": layer_dir.name,
                "metadata_sha256": sha256_file(layer_dir / "metadata.json"),
            }
        )
    atomic_json(
        target / "manifest.json",
        {
            "format": DIP_PACKAGE_FORMAT,
            "version": DIP_PACKAGE_VERSION if dual_layout else 2,
            "experimental": True,
            "source_model_hash": inspection.source_hash,
            "hidden_size": inspection.hidden_size,
            "intermediate_size": inspection.intermediate_size,
            "cache_line_bytes": CACHE_LINE_BYTES,
            "dual_layout_diagnostic": dual_layout,
            "layers": descriptors,
        },
    )
    return target


class SerializedDIPLayer:
    def __init__(self, directory: str | Path, *, verify: bool = True) -> None:
        self.directory = Path(directory)
        try:
            metadata = json.loads((self.directory / "metadata.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid DIP metadata: {exc}") from exc
        if metadata.get("format") != DIP_PACKAGE_FORMAT:
            raise ValueError("unsupported DIP package format")
        metadata_version = metadata.get("version")
        if metadata_version not in {2, DIP_PACKAGE_VERSION}:
            raise ValueError("unsupported DIP package version")
        names = [
            "config",
            "gate_coordinates",
            "up_coordinates",
            "down_rows",
            "value_norms",
        ]
        if metadata_version >= 3:
            names.extend(("gate_rows", "up_rows"))
        arrays: dict[str, NDArray[np.generic]] = {}
        for name in names:
            descriptor = metadata.get("files", {}).get(name)
            if not isinstance(descriptor, dict) or not isinstance(
                descriptor.get("file"), str
            ):
                raise ValueError(f"DIP metadata is missing {name}")
            path = self.directory / descriptor["file"]
            if verify and sha256_file(path) != descriptor.get("sha256"):
                raise ValueError(f"DIP checksum mismatch: {path}")
            arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        config = np.asarray(arrays["config"])
        if config.shape != (4,) or config.dtype != np.dtype("uint32"):
            raise ValueError("DIP config must be uint32[4]")
        binary_version, records, hidden, cache_line = map(int, config)
        if (
            binary_version not in {2, DIP_PACKAGE_VERSION}
            or cache_line != CACHE_LINE_BYTES
        ):
            raise ValueError("unsupported DIP binary configuration")
        if binary_version != metadata_version:
            raise ValueError("DIP metadata and binary configuration versions differ")
        expected = {
            "gate_coordinates": (hidden, records),
            "up_coordinates": (hidden, records),
            "down_rows": (records, hidden),
            "value_norms": (records,),
        }
        if binary_version >= 3:
            expected.update(
                gate_rows=(records, hidden), up_rows=(records, hidden)
            )
        for name, shape in expected.items():
            if arrays[name].dtype != np.dtype("float32") or arrays[name].shape != shape:
                raise ValueError(f"DIP {name} must be float32 with shape {shape}")
        self.records = records
        self.hidden_size = hidden
        self.gate_coordinates = arrays["gate_coordinates"]
        self.up_coordinates = arrays["up_coordinates"]
        self.gate_rows = arrays.get("gate_rows")
        self.up_rows = arrays.get("up_rows")
        self.down_rows = arrays["down_rows"]
        self.value_norms = arrays["value_norms"]

    def read(
        self,
        hidden: ArrayLike,
        *,
        input_fraction: float,
        candidate_count: int,
        top_k: int,
    ) -> SerializedDIPRead:
        state = np.asarray(hidden, dtype=np.float32)
        if state.shape != (self.hidden_size,) or not np.all(np.isfinite(state)):
            raise ValueError(f"hidden must be finite with shape ({self.hidden_size},)")
        if candidate_count <= 0 or candidate_count > self.records:
            raise ValueError("candidate_count must be within the record count")
        if top_k <= 0 or top_k > candidate_count:
            raise ValueError("top_k must be within candidate_count")
        coordinate_count = input_coordinate_count(self.hidden_size, input_fraction)
        coordinates = stable_top_k(np.abs(state), coordinate_count)
        partial_gate = np.asarray(
            state[coordinates] @ self.gate_coordinates[coordinates], dtype=np.float32
        )
        partial_up = np.asarray(
            state[coordinates] @ self.up_coordinates[coordinates], dtype=np.float32
        )
        proxy = np.abs(silu(partial_gate) * partial_up) * self.value_norms
        candidates = stable_top_k(proxy, candidate_count)
        candidate_gate = partial_gate[candidates].copy()
        candidate_up = partial_up[candidates].copy()
        omitted = np.setdiff1d(
            np.arange(self.hidden_size, dtype=np.int64), coordinates, assume_unique=True
        )
        if omitted.size:
            if self.gate_rows is not None and self.up_rows is not None:
                candidate_gate += self.gate_rows[candidates][:, omitted] @ state[omitted]
                candidate_up += self.up_rows[candidates][:, omitted] @ state[omitted]
            else:
                candidate_gate += state[omitted] @ self.gate_coordinates[omitted][
                    :, candidates
                ]
                candidate_up += state[omitted] @ self.up_coordinates[omitted][
                    :, candidates
                ]
        activations = np.asarray(silu(candidate_gate) * candidate_up, dtype=np.float32)
        exact_scores = np.abs(activations) * self.value_norms[candidates]
        by_index = np.argsort(candidates, kind="stable")
        local = by_index[stable_top_k(exact_scores[by_index], top_k)]
        selected = candidates[local]
        selected_activations = activations[local]
        output = np.asarray(selected_activations @ self.down_rows[selected], dtype=np.float32)

        item_bytes = np.dtype("float32").itemsize
        partial_bytes = 2 * self.records * coordinate_count * item_bytes
        completion_bytes = 2 * candidate_count * len(omitted) * item_bytes
        down_bytes = top_k * self.hidden_size * item_bytes
        logical = partial_bytes + completion_bytes + down_bytes
        values_per_line = CACHE_LINE_BYTES // item_bytes
        if self.gate_rows is not None:
            omitted_lines = np.unique(omitted // values_per_line).size
            completion_cache_bytes = (
                2 * candidate_count * omitted_lines * CACHE_LINE_BYTES
            )
        else:
            candidate_lines = np.unique(candidates // values_per_line).size
            completion_cache_bytes = (
                2 * len(omitted) * candidate_lines * CACHE_LINE_BYTES
            )
        cache_line_bytes = partial_bytes + completion_cache_bytes + down_bytes
        dense_bytes = 3 * self.records * self.hidden_size * item_bytes
        return SerializedDIPRead(
            output=output,
            input_coordinates=coordinates,
            candidate_indices=candidates,
            selected_indices=selected,
            selected_activations=selected_activations,
            metrics=SerializedDIPMetrics(
                selected_input_coordinates=coordinate_count,
                partial_projection_bytes=partial_bytes,
                candidate_completion_bytes=completion_bytes,
                selected_down_bytes=down_bytes,
                logical_weight_bytes=logical,
                cache_line_weight_bytes=cache_line_bytes,
                dense_weight_bytes=dense_bytes,
            ),
        )
