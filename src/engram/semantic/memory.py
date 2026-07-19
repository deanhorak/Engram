from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from engram.models.inspection import inspect_model, load_layer_mlp
from engram.semantic.swiglu import neuron_activations
from engram.semantic.quantization import encode_additive_vectors, encode_scalar_affine
from engram.semantic.ivf import JointKeyIVFProbeIndex
from engram.utils import atomic_json, npy_file_metadata, sha256_file


SEMANTIC_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SemanticRead:
    output: np.ndarray
    active_records: int
    candidate_records: int
    estimated_bytes_read: int
    indices: np.ndarray


class SemanticLayer:
    def __init__(self, gate_keys: np.ndarray, up_keys: np.ndarray, values: np.ndarray) -> None:
        self.gate_keys = np.asarray(gate_keys)
        self.up_keys = np.asarray(up_keys)
        self.values = np.asarray(values)
        if self.gate_keys.ndim != 2 or self.gate_keys.shape != self.up_keys.shape:
            raise ValueError("gate and up keys must have shape [records, hidden]")
        records, hidden = self.gate_keys.shape
        if self.values.shape != (records, hidden):
            raise ValueError("values must have shape [records, hidden]")

    @property
    def records(self) -> int:
        return self.gate_keys.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.gate_keys.shape[1]

    def full(self, hidden: np.ndarray) -> np.ndarray:
        activation = neuron_activations(hidden, self.gate_keys, self.up_keys)
        return activation @ self.values

    def contribution_order(self, hidden: np.ndarray) -> np.ndarray:
        activation = neuron_activations(hidden, self.gate_keys, self.up_keys)
        scores = np.abs(activation) * np.linalg.norm(self.values, axis=1)
        return np.argsort(-scores, kind="stable")

    def read(self, hidden: np.ndarray, candidates: Iterable[int], *, top_k: int) -> SemanticRead:
        state = np.asarray(hidden)
        if state.shape != (self.hidden_size,):
            raise ValueError(f"hidden state must have shape ({self.hidden_size},)")
        candidate_array = np.asarray(list(candidates), dtype=np.int64)
        if candidate_array.ndim != 1 or np.any(candidate_array < 0) or np.any(candidate_array >= self.records):
            raise ValueError("candidate index outside semantic memory")
        candidate_array = np.unique(candidate_array)
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        activation = neuron_activations(
            state, self.gate_keys[candidate_array], self.up_keys[candidate_array]
        )
        scores = np.abs(activation) * np.linalg.norm(self.values[candidate_array], axis=1)
        count = min(top_k, candidate_array.size)
        local_order = np.argsort(-scores, kind="stable")[:count]
        selected = candidate_array[local_order]
        selected_activation = activation[local_order]
        output = selected_activation @ self.values[selected] if count else np.zeros(self.hidden_size, dtype=state.dtype)
        scalar_bytes = max(self.gate_keys.dtype.itemsize, self.up_keys.dtype.itemsize)
        value_bytes = self.values.dtype.itemsize
        estimated_bytes = candidate_array.size * 2 * self.hidden_size * scalar_bytes
        estimated_bytes += count * self.hidden_size * value_bytes
        return SemanticRead(
            output=np.asarray(output),
            active_records=count,
            candidate_records=candidate_array.size,
            estimated_bytes_read=estimated_bytes,
            indices=selected,
        )

    def oracle_read(self, hidden: np.ndarray, *, top_k: int) -> SemanticRead:
        order = self.contribution_order(hidden)
        return self.read(hidden, order[:top_k], top_k=top_k)


def _array_metadata(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "file": path.name,
        **npy_file_metadata(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def build_semantic_package(
    model: str | Path,
    out: str | Path,
    *,
    key_bits: int = 8,
    value_codebooks: int = 2,
    value_codebook_size: int = 16,
    ivf_clusters: int = 32,
    ivf_iterations: int = 20,
    include_reference: bool = True,
) -> Path:
    if ivf_clusters <= 0 or ivf_iterations <= 0:
        raise ValueError("ivf_clusters and ivf_iterations must be positive")
    inspection = inspect_model(model)
    target = Path(out)
    semantic_dir = target / "semantic"
    semantic_dir.mkdir(parents=True, exist_ok=True)
    layers = []
    for layer in range(inspection.num_hidden_layers):
        layer_dir = semantic_dir / f"layer-{layer:04d}"
        layer_dir.mkdir(exist_ok=True)
        gate, up, down = load_layer_mlp(model, layer)
        arrays = {"gate_keys": gate, "up_keys": up, "values": down.T.copy()}
        files = {}
        if include_reference:
            for name, array in arrays.items():
                path = layer_dir / f"{name}.npy"
                np.save(path, array, allow_pickle=False)
                files[name] = _array_metadata(path, array)
        metadata = np.zeros(
            inspection.intermediate_size,
            dtype=[("source_layer", "<i4"), ("source_neuron", "<i4"), ("stage", "<i4")],
        )
        metadata["source_layer"] = layer
        metadata["source_neuron"] = np.arange(inspection.intermediate_size)
        metadata["stage"] = layer
        metadata_path = layer_dir / "metadata.npy"
        np.save(metadata_path, metadata, allow_pickle=False)
        files["metadata"] = _array_metadata(metadata_path, metadata)
        quantized_dir = layer_dir / "quantized"
        quantized_dir.mkdir(exist_ok=True)
        gate_encoding = encode_scalar_affine(gate, bits=key_bits)
        up_encoding = encode_scalar_affine(up, bits=key_bits)
        value_encoding = encode_additive_vectors(
            arrays["values"],
            num_codebooks=value_codebooks,
            codebook_size=min(value_codebook_size, inspection.intermediate_size),
        )
        quantized_arrays = {
            "gate_codes": gate_encoding.codes,
            "gate_offsets": np.asarray(gate_encoding.metadata.offsets, dtype=np.float32),
            "gate_scales": np.asarray(gate_encoding.metadata.scales, dtype=np.float32),
            "up_codes": up_encoding.codes,
            "up_offsets": np.asarray(up_encoding.metadata.offsets, dtype=np.float32),
            "up_scales": np.asarray(up_encoding.metadata.scales, dtype=np.float32),
            "value_codes": value_encoding.codes,
            "value_codebooks": value_encoding.codebooks,
        }
        quantized_files = {}
        for name, array in quantized_arrays.items():
            path = quantized_dir / f"{name}.npy"
            np.save(path, array, allow_pickle=False)
            quantized_files[name] = _array_metadata(path, array)
        codec_metadata = {
            "gate": gate_encoding.metadata.to_dict(),
            "up": up_encoding.metadata.to_dict(),
            "values": value_encoding.metadata.to_dict(),
        }
        atomic_json(quantized_dir / "codecs.json", codec_metadata)
        quantized_files["codecs"] = {
            "file": "codecs.json",
            "sha256": sha256_file(quantized_dir / "codecs.json"),
            "bytes": (quantized_dir / "codecs.json").stat().st_size,
        }
        cluster_count = min(max(1, int(ivf_clusters)), inspection.intermediate_size)
        decoded_gate = (
            quantized_arrays["gate_offsets"][None, :]
            + quantized_arrays["gate_codes"].astype(np.float32)
            * quantized_arrays["gate_scales"][None, :]
        )
        decoded_up = (
            quantized_arrays["up_offsets"][None, :]
            + quantized_arrays["up_codes"].astype(np.float32)
            * quantized_arrays["up_scales"][None, :]
        )
        ivf = JointKeyIVFProbeIndex.build(
            decoded_gate,
            decoded_up,
            num_clusters=cluster_count,
            iterations=ivf_iterations,
        )
        ivf_dir = ivf.save(quantized_dir / "ivf")
        ivf_files = {}
        for path in sorted(ivf_dir.iterdir()):
            if path.is_file():
                ivf_files[path.name] = {
                    "file": f"ivf/{path.name}",
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    **(npy_file_metadata(path) if path.suffix == ".npy" else {}),
                }
        layers.append(
            {
                "layer": layer,
                "records": inspection.intermediate_size,
                "files": files,
                "quantized_files": quantized_files,
                "ivf_files": ivf_files,
            }
        )
    manifest = {
        "format": "engram-semantic-memory",
        "version": SEMANTIC_FORMAT_VERSION,
        "source_model_hash": inspection.source_hash,
        "hidden_size": inspection.hidden_size,
        "layers": layers,
        "compression": {
            "status": "reference_and_quantized" if include_reference else "quantized_only",
            "key_bits": key_bits,
            "value_codebooks": value_codebooks,
            "value_codebook_size": min(value_codebook_size, inspection.intermediate_size),
        },
        "routing": {
            "type": "joint_key_ivf",
            "clusters": min(max(1, int(ivf_clusters)), inspection.intermediate_size),
            "iterations": ivf_iterations,
            "record_proxy": "quantized_joint_gate_up_cosine",
            "exact_candidate_rerank": True,
        },
    }
    atomic_json(semantic_dir / "manifest.json", manifest)
    return semantic_dir


def load_semantic_layer(package: str | Path, layer: int, *, verify: bool = True) -> SemanticLayer:
    root = Path(package)
    semantic_dir = root if root.name == "semantic" else root / "semantic"
    with (semantic_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("version") != SEMANTIC_FORMAT_VERSION:
        raise ValueError("unsupported semantic package version")
    descriptor = next((item for item in manifest["layers"] if item["layer"] == layer), None)
    if descriptor is None:
        raise IndexError(f"semantic layer {layer} is absent")
    layer_dir = semantic_dir / f"layer-{layer:04d}"
    arrays = {}
    for name in ("gate_keys", "up_keys", "values"):
        file_info = descriptor["files"][name]
        path = layer_dir / file_info["file"]
        if verify and sha256_file(path) != file_info["sha256"]:
            raise ValueError(f"semantic checksum mismatch: {path}")
        arrays[name] = np.load(path, mmap_mode="r")
    return SemanticLayer(arrays["gate_keys"], arrays["up_keys"], arrays["values"])
