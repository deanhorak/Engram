"""Deterministic inverted-file index for joint SwiGLU gate/up keys."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.utils import atomic_json


IVF_FORMAT = "engram.semantic.joint_key_ivf"
IVF_SCHEMA_VERSION = 1


class IVFIndexError(ValueError):
    """Raised for invalid joint-key indexes or searches."""


@dataclass(frozen=True)
class IVFCandidateResult:
    indices: NDArray[np.int64]
    proxy_scores: NDArray[np.float64]
    probed_clusters: NDArray[np.int64]
    probed_record_count: int


@dataclass(frozen=True)
class IVFProbeResult:
    indices: NDArray[np.uint32]
    clusters: NDArray[np.uint32]
    centroid_scores: NDArray[np.float64]
    index_bytes_read: int


def _positive_integer(value: object, name: str) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or value <= 0
    ):
        raise IVFIndexError(f"{name} must be a positive integer")
    return int(value)


def _matrix(value: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise IVFIndexError(f"{name} must be numeric") from error
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise IVFIndexError(f"{name} must be a non-empty rank-2 matrix")
    if not np.all(np.isfinite(result)):
        raise IVFIndexError(f"{name} must contain only finite values")
    return result


def _normalize_rows(values: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0.0)


def _squared_distances(
    records: NDArray[np.float64], centroids: NDArray[np.float64]
) -> NDArray[np.float64]:
    record_norms = np.einsum("ij,ij->i", records, records)[:, None]
    centroid_norms = np.einsum("ij,ij->i", centroids, centroids)[None, :]
    return np.maximum(record_norms + centroid_norms - 2.0 * (records @ centroids.T), 0.0)


def _initial_centroids(
    records: NDArray[np.float64], count: int
) -> NDArray[np.float64]:
    norms = np.einsum("ij,ij->i", records, records)
    selected = [int(np.argmax(norms))]
    nearest = _squared_distances(records, records[selected])[:, 0]
    while len(selected) < count:
        nearest[np.asarray(selected, dtype=np.int64)] = -1.0
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(
            nearest, _squared_distances(records, records[[index]])[:, 0]
        )
    return records[np.asarray(selected, dtype=np.int64)].copy()


def _repair_empty_clusters(
    assignments: NDArray[np.int64], distances: NDArray[np.float64], count: int
) -> None:
    populations = np.bincount(assignments, minlength=count)
    for empty in np.flatnonzero(populations == 0):
        assigned_distances = distances[np.arange(assignments.size), assignments]
        eligible = populations[assignments] > 1
        scores = np.where(eligible, assigned_distances, -1.0)
        donor_record = int(np.argmax(scores))
        donor_cluster = int(assignments[donor_record])
        populations[donor_cluster] -= 1
        assignments[donor_record] = int(empty)
        populations[empty] += 1


def _fit_joint_centroids(
    records: NDArray[np.float64], count: int, iterations: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    centroids = _initial_centroids(records, count)
    previous: NDArray[np.int64] | None = None
    assignments = np.zeros(records.shape[0], dtype=np.int64)
    for _ in range(iterations):
        distances = _squared_distances(records, centroids)
        assignments = np.argmin(distances, axis=1).astype(np.int64, copy=False)
        _repair_empty_clusters(assignments, distances, count)
        if previous is not None and np.array_equal(assignments, previous):
            break
        for cluster in range(count):
            centroids[cluster] = np.mean(records[assignments == cluster], axis=0)
        previous = assignments.copy()
    return centroids, assignments


def _readonly(value: ArrayLike, dtype: np.dtype | type, name: str) -> np.ndarray:
    try:
        result = np.asanyarray(value, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise IVFIndexError(f"{name} has an invalid dtype") from error
    if not np.all(np.isfinite(result)):
        raise IVFIndexError(f"{name} must contain only finite values")
    result = result.view()
    result.flags.writeable = False
    return result


class JointKeyIVFIndex:
    """CSR IVF index built from both normalized gate and up key geometry."""

    def __init__(
        self,
        *,
        gate_records: ArrayLike,
        up_records: ArrayLike,
        gate_centroids: ArrayLike,
        up_centroids: ArrayLike,
        posting_offsets: ArrayLike,
        posting_indices: ArrayLike,
        build_iterations: int,
    ) -> None:
        self.gate_records = _readonly(gate_records, np.float32, "gate_records")
        self.up_records = _readonly(up_records, np.float32, "up_records")
        self.gate_centroids = _readonly(
            gate_centroids, np.float32, "gate_centroids"
        )
        self.up_centroids = _readonly(up_centroids, np.float32, "up_centroids")
        self.posting_offsets = _readonly(
            posting_offsets, np.int64, "posting_offsets"
        )
        self.posting_indices = _readonly(
            posting_indices, np.int64, "posting_indices"
        )
        self.build_iterations = _positive_integer(
            build_iterations, "build_iterations"
        )
        self._validate()

    @classmethod
    def build(
        cls,
        gate_keys: ArrayLike,
        up_keys: ArrayLike,
        *,
        num_clusters: int,
        iterations: int = 20,
    ) -> "JointKeyIVFIndex":
        gate = _matrix(gate_keys, "gate_keys")
        up = _matrix(up_keys, "up_keys")
        if gate.shape != up.shape:
            raise IVFIndexError("gate_keys and up_keys must have the same shape")
        clusters = _positive_integer(num_clusters, "num_clusters")
        iterations = _positive_integer(iterations, "iterations")
        if clusters > gate.shape[0]:
            raise IVFIndexError("num_clusters cannot exceed record count")

        normalized_gate = _normalize_rows(gate)
        normalized_up = _normalize_rows(up)
        joint = np.concatenate([normalized_gate, normalized_up], axis=1)
        centroids, assignments = _fit_joint_centroids(
            joint, clusters, iterations
        )
        counts = np.bincount(assignments, minlength=clusters)
        offsets = np.empty(clusters + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        # cluster is the primary key and record ID the deterministic secondary key.
        postings = np.lexsort(
            (np.arange(gate.shape[0], dtype=np.int64), assignments)
        ).astype(np.int64, copy=False)
        hidden = gate.shape[1]
        return cls(
            gate_records=normalized_gate.astype(np.float32),
            up_records=normalized_up.astype(np.float32),
            gate_centroids=centroids[:, :hidden].astype(np.float32),
            up_centroids=centroids[:, hidden:].astype(np.float32),
            posting_offsets=offsets,
            posting_indices=postings,
            build_iterations=iterations,
        )

    @property
    def records(self) -> int:
        return int(self.gate_records.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.gate_records.shape[1])

    @property
    def centroids(self) -> int:
        return int(self.gate_centroids.shape[0])

    @property
    def records_bytes(self) -> int:
        return int(self.gate_records.nbytes + self.up_records.nbytes)

    @property
    def centroids_bytes(self) -> int:
        return int(self.gate_centroids.nbytes + self.up_centroids.nbytes)

    @property
    def postings_bytes(self) -> int:
        return int(self.posting_offsets.nbytes + self.posting_indices.nbytes)

    @property
    def total_bytes(self) -> int:
        return self.records_bytes + self.centroids_bytes + self.postings_bytes

    def _validate(self) -> None:
        if self.gate_records.ndim != 2 or self.gate_records.shape[0] == 0:
            raise IVFIndexError("gate_records must be a non-empty rank-2 matrix")
        if self.up_records.shape != self.gate_records.shape:
            raise IVFIndexError("gate_records and up_records shapes differ")
        if self.gate_centroids.ndim != 2 or self.gate_centroids.shape[0] == 0:
            raise IVFIndexError("gate_centroids must be a non-empty rank-2 matrix")
        if self.up_centroids.shape != self.gate_centroids.shape:
            raise IVFIndexError("gate and up centroid shapes differ")
        if self.gate_centroids.shape[1] != self.gate_records.shape[1]:
            raise IVFIndexError("centroid and record hidden widths differ")
        if self.posting_offsets.shape != (self.centroids + 1,):
            raise IVFIndexError("posting_offsets must have centroids + 1 entries")
        if self.posting_indices.shape != (self.records,):
            raise IVFIndexError("posting_indices must contain every record exactly once")
        if self.posting_offsets[0] != 0 or self.posting_offsets[-1] != self.records:
            raise IVFIndexError("posting_offsets must span all postings")
        if np.any(np.diff(self.posting_offsets) <= 0):
            raise IVFIndexError("every centroid must have a non-empty CSR posting")
        if np.any(self.posting_indices < 0) or np.any(
            self.posting_indices >= self.records
        ):
            raise IVFIndexError("posting index is out of range")
        if not np.array_equal(
            np.sort(self.posting_indices), np.arange(self.records, dtype=np.int64)
        ):
            raise IVFIndexError("posting indices must be a permutation of records")
        for cluster in range(self.centroids):
            start, end = self.posting_offsets[cluster : cluster + 2]
            posting = self.posting_indices[start:end]
            if np.any(np.diff(posting) <= 0):
                raise IVFIndexError("records within a posting must be strictly ordered")

    def cluster_records(self, cluster: int) -> NDArray[np.int64]:
        if (
            not isinstance(cluster, (int, np.integer))
            or isinstance(cluster, (bool, np.bool_))
            or cluster < 0
            or cluster >= self.centroids
        ):
            raise IVFIndexError("cluster index is out of range")
        start, end = self.posting_offsets[int(cluster) : int(cluster) + 2]
        return self.posting_indices[start:end]

    @staticmethod
    def _joint_scores(
        query: NDArray[np.float64],
        gate: np.ndarray,
        up: np.ndarray,
    ) -> NDArray[np.float64]:
        gate_norms = np.linalg.norm(gate, axis=1)
        up_norms = np.linalg.norm(up, axis=1)
        gate_dot = gate @ query
        up_dot = up @ query
        gate_alignment = np.divide(
            gate_dot,
            gate_norms,
            out=np.zeros_like(gate_dot, dtype=np.float64),
            where=gate_norms > 0.0,
        )
        up_alignment = np.divide(
            up_dot,
            up_norms,
            out=np.zeros_like(up_dot, dtype=np.float64),
            where=up_norms > 0.0,
        )
        return np.maximum(gate_alignment, 0.0) * np.abs(up_alignment)

    def search(
        self,
        hidden: ArrayLike,
        *,
        probes: int,
        candidate_count: int,
        expand_for_candidates: bool = False,
    ) -> IVFCandidateResult:
        try:
            query = np.asarray(hidden, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise IVFIndexError("hidden must be numeric") from error
        if query.shape != (self.hidden_size,):
            raise IVFIndexError(
                f"hidden must have shape [{self.hidden_size}], got {query.shape}"
            )
        if not np.all(np.isfinite(query)):
            raise IVFIndexError("hidden must contain only finite values")
        probe_count = _positive_integer(probes, "probes")
        candidate_count = _positive_integer(candidate_count, "candidate_count")
        if probe_count > self.centroids:
            raise IVFIndexError("probes cannot exceed centroid count")
        query_norm = float(np.linalg.norm(query))
        unit_query = query / query_norm if query_norm > 0.0 else np.zeros_like(query)

        centroid_scores = self._joint_scores(
            unit_query, self.gate_centroids, self.up_centroids
        )
        centroid_ids = np.arange(self.centroids, dtype=np.int64)
        cluster_order = np.lexsort((centroid_ids, -centroid_scores))
        selected_probes = probe_count
        if expand_for_candidates:
            while selected_probes < self.centroids:
                gathered = sum(
                    self.cluster_records(int(cluster)).size
                    for cluster in cluster_order[:selected_probes]
                )
                if gathered >= candidate_count:
                    break
                selected_probes += 1
        cluster_order = cluster_order[:selected_probes]
        probed_clusters = centroid_ids[cluster_order]
        posting_parts = [self.cluster_records(int(cluster)) for cluster in probed_clusters]
        probed_records = np.concatenate(posting_parts).astype(np.int64, copy=False)
        # Each record belongs to exactly one CSR posting, so uniqueness is structural.
        record_scores = self._joint_scores(
            unit_query,
            self.gate_records[probed_records],
            self.up_records[probed_records],
        )
        record_order = np.lexsort((probed_records, -record_scores))
        selected_count = min(candidate_count, probed_records.size)
        selected_order = record_order[:selected_count]
        return IVFCandidateResult(
            indices=probed_records[selected_order],
            proxy_scores=record_scores[selected_order],
            probed_clusters=probed_clusters,
            probed_record_count=int(probed_records.size),
        )

    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        arrays = {
            "gate_records": self.gate_records,
            "up_records": self.up_records,
            "gate_centroids": self.gate_centroids,
            "up_centroids": self.up_centroids,
            "posting_offsets": self.posting_offsets,
            "posting_indices": self.posting_indices,
        }
        for name, array in arrays.items():
            np.save(target / f"{name}.npy", array, allow_pickle=False)
        atomic_json(
            target / "metadata.json",
            {
                "format": IVF_FORMAT,
                "schema_version": IVF_SCHEMA_VERSION,
                "records": self.records,
                "hidden_size": self.hidden_size,
                "centroids": self.centroids,
                "build_iterations": self.build_iterations,
                "record_dtype": "float32",
                "centroid_dtype": "float32",
                "posting_dtype": "int64",
                "files": {name: f"{name}.npy" for name in arrays},
            },
        )
        return target

    @classmethod
    def load(
        cls, directory: str | Path, *, mmap_mode: str | None = "r"
    ) -> "JointKeyIVFIndex":
        source = Path(directory)
        try:
            metadata = json.loads(
                (source / "metadata.json").read_text(encoding="utf-8")
            )
            if metadata.get("format") != IVF_FORMAT:
                raise IVFIndexError("unsupported IVF index format")
            if metadata.get("schema_version") != IVF_SCHEMA_VERSION:
                raise IVFIndexError("unsupported IVF index schema version")
            if metadata.get("record_dtype") != "float32":
                raise IVFIndexError("unsupported IVF record dtype")
            if metadata.get("centroid_dtype") != "float32":
                raise IVFIndexError("unsupported IVF centroid dtype")
            if metadata.get("posting_dtype") != "int64":
                raise IVFIndexError("unsupported IVF posting dtype")
            files = metadata["files"]
            expected_names = {
                "gate_records",
                "up_records",
                "gate_centroids",
                "up_centroids",
                "posting_offsets",
                "posting_indices",
            }
            if set(files) != expected_names:
                raise IVFIndexError("IVF metadata has an unexpected file set")
            for name in expected_names:
                if files[name] != f"{name}.npy":
                    raise IVFIndexError(
                        f"IVF metadata has an unsafe or unexpected path for {name}"
                    )
            arrays = {
                name: np.load(
                    source / files[name], mmap_mode=mmap_mode, allow_pickle=False
                )
                for name in expected_names
            }
            result = cls(
                **arrays, build_iterations=metadata["build_iterations"]
            )
            if (
                metadata.get("records") != result.records
                or metadata.get("hidden_size") != result.hidden_size
                or metadata.get("centroids") != result.centroids
            ):
                raise IVFIndexError("IVF metadata dimensions disagree with arrays")
            return result
        except IVFIndexError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise IVFIndexError(f"cannot load IVF index: {source}") from error


class JointKeyIVFProbeIndex:
    """Runtime IVF routing data without duplicate full-precision record keys."""

    def __init__(
        self,
        *,
        joint_centroids: ArrayLike,
        posting_offsets: ArrayLike,
        posting_indices: ArrayLike,
        hidden_size: int,
        build_iterations: int,
    ) -> None:
        self.hidden_size = _positive_integer(hidden_size, "hidden_size")
        self.build_iterations = _positive_integer(
            build_iterations, "build_iterations"
        )
        self.joint_centroids = _readonly(
            joint_centroids, np.float32, "joint_centroids"
        )
        self.posting_offsets = _readonly(
            posting_offsets, np.uint32, "posting_offsets"
        )
        self.posting_indices = _readonly(
            posting_indices, np.uint32, "posting_indices"
        )
        self._validate()

    @classmethod
    def from_full(cls, index: JointKeyIVFIndex) -> "JointKeyIVFProbeIndex":
        return cls(
            joint_centroids=np.concatenate(
                [index.gate_centroids, index.up_centroids], axis=1
            ).astype(np.float32),
            posting_offsets=index.posting_offsets.astype(np.uint32),
            posting_indices=index.posting_indices.astype(np.uint32),
            hidden_size=index.hidden_size,
            build_iterations=index.build_iterations,
        )

    @classmethod
    def build(
        cls,
        gate_keys: ArrayLike,
        up_keys: ArrayLike,
        *,
        num_clusters: int,
        iterations: int = 20,
    ) -> "JointKeyIVFProbeIndex":
        return cls.from_full(
            JointKeyIVFIndex.build(
                gate_keys,
                up_keys,
                num_clusters=num_clusters,
                iterations=iterations,
            )
        )

    @property
    def clusters(self) -> int:
        return int(self.joint_centroids.shape[0])

    @property
    def records(self) -> int:
        return int(self.posting_indices.size)

    @property
    def total_bytes(self) -> int:
        return int(
            self.joint_centroids.nbytes
            + self.posting_offsets.nbytes
            + self.posting_indices.nbytes
        )

    def _validate(self) -> None:
        if self.joint_centroids.ndim != 2 or self.joint_centroids.shape != (
            self.joint_centroids.shape[0],
            2 * self.hidden_size,
        ) or self.joint_centroids.shape[0] == 0:
            raise IVFIndexError(
                "joint_centroids must have shape [clusters, 2 * hidden_size]"
            )
        if self.posting_offsets.shape != (self.clusters + 1,):
            raise IVFIndexError("posting_offsets must have clusters + 1 entries")
        if self.posting_offsets[0] != 0 or self.posting_offsets[-1] != self.records:
            raise IVFIndexError("posting_offsets must span all postings")
        if np.any(np.diff(self.posting_offsets.astype(np.int64)) < 0):
            raise IVFIndexError("posting_offsets must be monotonic")
        if self.records == 0:
            raise IVFIndexError("posting_indices must not be empty")
        if not np.array_equal(
            np.sort(self.posting_indices),
            np.arange(self.records, dtype=np.uint32),
        ):
            raise IVFIndexError("posting_indices must be a permutation of records")
        for cluster in range(self.clusters):
            start = int(self.posting_offsets[cluster])
            end = int(self.posting_offsets[cluster + 1])
            posting = self.posting_indices[start:end]
            if posting.size > 1 and np.any(np.diff(posting.astype(np.int64)) <= 0):
                raise IVFIndexError("records within a posting must be strictly ordered")

    def probe(
        self,
        hidden: ArrayLike,
        *,
        probes: int,
        minimum_records: int,
    ) -> IVFProbeResult:
        query = np.asarray(hidden, dtype=np.float64)
        if query.shape != (self.hidden_size,) or not np.all(np.isfinite(query)):
            raise IVFIndexError(
                f"hidden must be finite with shape [{self.hidden_size}]"
            )
        probes = _positive_integer(probes, "probes")
        minimum_records = _positive_integer(minimum_records, "minimum_records")
        if probes > self.clusters:
            raise IVFIndexError("probes cannot exceed cluster count")
        if minimum_records > self.records:
            raise IVFIndexError("minimum_records cannot exceed record count")
        norm = float(np.linalg.norm(query))
        unit_query = query / norm if norm > 0.0 else np.zeros_like(query)
        gate = self.joint_centroids[:, : self.hidden_size]
        up = self.joint_centroids[:, self.hidden_size :]
        scores = JointKeyIVFIndex._joint_scores(unit_query, gate, up)
        ids = np.arange(self.clusters, dtype=np.int64)
        order = np.lexsort((ids, -scores))
        selected = probes
        gathered = 0
        while True:
            selected_clusters = order[:selected]
            gathered = sum(
                int(self.posting_offsets[c + 1])
                - int(self.posting_offsets[c])
                for c in selected_clusters
            )
            if gathered >= minimum_records or selected == self.clusters:
                break
            selected += 1
        parts = [
            self.posting_indices[
                int(self.posting_offsets[c]) : int(self.posting_offsets[c + 1])
            ]
            for c in selected_clusters
        ]
        records = np.concatenate(parts).astype(np.uint32, copy=False)
        bytes_read = (
            (0 if norm == 0.0 else self.joint_centroids.nbytes)
            + 2 * selected * np.dtype(np.uint32).itemsize
            + records.nbytes
        )
        return IVFProbeResult(
            indices=records,
            clusters=selected_clusters.astype(np.uint32),
            centroid_scores=scores[selected_clusters],
            index_bytes_read=int(bytes_read),
        )

    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "centroids.npy", self.joint_centroids, allow_pickle=False)
        np.save(target / "posting_offsets.npy", self.posting_offsets, allow_pickle=False)
        np.save(target / "posting_indices.npy", self.posting_indices, allow_pickle=False)
        atomic_json(
            target / "metadata.json",
            {
                "format": IVF_FORMAT,
                "schema_version": IVF_SCHEMA_VERSION,
                "records": self.records,
                "hidden_size": self.hidden_size,
                "clusters": self.clusters,
                "build_iterations": self.build_iterations,
                "centroid_layout": "gate_then_up",
                "posting_dtype": "uint32",
            },
        )
        return target

    @classmethod
    def load(
        cls, directory: str | Path, *, mmap_mode: str | None = "r"
    ) -> "JointKeyIVFProbeIndex":
        source = Path(directory)
        try:
            metadata = json.loads((source / "metadata.json").read_text())
            if metadata.get("format") != IVF_FORMAT or metadata.get(
                "schema_version"
            ) != IVF_SCHEMA_VERSION:
                raise IVFIndexError("unsupported IVF runtime index format")
            result = cls(
                joint_centroids=np.load(
                    source / "centroids.npy", mmap_mode=mmap_mode, allow_pickle=False
                ),
                posting_offsets=np.load(
                    source / "posting_offsets.npy", mmap_mode=mmap_mode, allow_pickle=False
                ),
                posting_indices=np.load(
                    source / "posting_indices.npy", mmap_mode=mmap_mode, allow_pickle=False
                ),
                hidden_size=metadata["hidden_size"],
                build_iterations=metadata["build_iterations"],
            )
            if metadata.get("records") != result.records or metadata.get(
                "clusters"
            ) != result.clusters:
                raise IVFIndexError("IVF runtime metadata dimensions disagree")
            return result
        except IVFIndexError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise IVFIndexError(f"cannot load IVF runtime index: {source}") from error


__all__ = [
    "IVFCandidateResult",
    "IVFIndexError",
    "IVFProbeResult",
    "JointKeyIVFIndex",
    "JointKeyIVFProbeIndex",
]
