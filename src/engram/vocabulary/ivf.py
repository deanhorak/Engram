"""Deterministic IVF candidate index for vocabulary projection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.utils import atomic_json


VOCABULARY_IVF_FORMAT = "engram.vocabulary.normalized_ivf"
VOCABULARY_IVF_SCHEMA_VERSION = 1


class VocabularyIVFError(ValueError):
    """Raised for invalid vocabulary IVF indexes and searches."""


@dataclass(frozen=True)
class VocabularyIVFResult:
    candidate_ids: NDArray[np.uint32]
    proxy_scores: NDArray[np.float64]
    probed_clusters: NDArray[np.uint32]
    probes: int
    expansions: int
    probed_token_count: int
    proxy_scores_computed: int
    bytes_read: int


def _positive_integer(value: object, name: str) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or value <= 0
    ):
        raise VocabularyIVFError(f"{name} must be a positive integer")
    return int(value)


def _embedding_matrix(values: ArrayLike) -> NDArray[np.float64]:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise VocabularyIVFError("embeddings must be numeric") from error
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise VocabularyIVFError("embeddings must be a non-empty rank-2 matrix")
    if not np.all(np.isfinite(result)):
        raise VocabularyIVFError("embeddings must contain only finite values")
    if result.shape[0] > np.iinfo(np.uint32).max:
        raise VocabularyIVFError("vocabulary exceeds uint32 token range")
    return result


def _normalize_rows(values: NDArray[np.float64]) -> NDArray[np.float32]:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.divide(
        values, norms, out=np.zeros_like(values), where=norms > 0.0
    )
    return normalized.astype(np.float32)


def _squared_distances(
    values: NDArray[np.float64], centers: NDArray[np.float64]
) -> NDArray[np.float64]:
    value_norms = np.einsum("ij,ij->i", values, values)[:, None]
    center_norms = np.einsum("ij,ij->i", centers, centers)[None, :]
    return np.maximum(value_norms + center_norms - 2.0 * (values @ centers.T), 0.0)


def _initial_centroids(
    values: NDArray[np.float64], count: int
) -> NDArray[np.float64]:
    norms = np.einsum("ij,ij->i", values, values)
    selected = [int(np.argmax(norms))]
    nearest = _squared_distances(values, values[selected])[:, 0]
    while len(selected) < count:
        nearest[np.asarray(selected, dtype=np.int64)] = -1.0
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(
            nearest, _squared_distances(values, values[[index]])[:, 0]
        )
    return values[np.asarray(selected, dtype=np.int64)].copy()


def _repair_empty_clusters(
    assignments: NDArray[np.int64], distances: NDArray[np.float64], count: int
) -> None:
    populations = np.bincount(assignments, minlength=count)
    for empty in np.flatnonzero(populations == 0):
        assigned_distance = distances[np.arange(assignments.size), assignments]
        eligible = populations[assignments] > 1
        donor = int(np.argmax(np.where(eligible, assigned_distance, -1.0)))
        previous = int(assignments[donor])
        populations[previous] -= 1
        assignments[donor] = int(empty)
        populations[empty] += 1


def _fit_centroids(
    values: NDArray[np.float64], count: int, iterations: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    centroids = _initial_centroids(values, count)
    previous: NDArray[np.int64] | None = None
    assignments = np.zeros(values.shape[0], dtype=np.int64)
    for _ in range(iterations):
        distances = _squared_distances(values, centroids)
        assignments = np.argmin(distances, axis=1).astype(np.int64, copy=False)
        _repair_empty_clusters(assignments, distances, count)
        if previous is not None and np.array_equal(assignments, previous):
            break
        for cluster in range(count):
            centroids[cluster] = np.mean(values[assignments == cluster], axis=0)
        previous = assignments.copy()
    return centroids, assignments


def _readonly(value: ArrayLike, dtype: np.dtype | type, name: str) -> np.ndarray:
    try:
        result = np.asanyarray(value)
    except (TypeError, ValueError) as error:
        raise VocabularyIVFError(f"{name} has an invalid dtype") from error
    expected_dtype = np.dtype(dtype)
    if result.dtype != expected_dtype:
        raise VocabularyIVFError(
            f"{name} must have dtype {expected_dtype.name}, got {result.dtype}"
        )
    if not np.all(np.isfinite(result)):
        raise VocabularyIVFError(f"{name} must contain only finite values")
    result = result.view()
    result.flags.writeable = False
    return result


class VocabularyIVFIndex:
    """Runtime IVF data without a duplicate vocabulary embedding matrix."""

    def __init__(
        self,
        *,
        centroids: ArrayLike,
        posting_offsets: ArrayLike,
        token_ids: ArrayLike,
        build_iterations: int,
    ) -> None:
        self.centroid_vectors = _readonly(centroids, np.float32, "centroids")
        self.posting_offsets = _readonly(
            posting_offsets, np.uint32, "posting_offsets"
        )
        self.token_ids = _readonly(token_ids, np.uint32, "token_ids")
        self.build_iterations = _positive_integer(
            build_iterations, "build_iterations"
        )
        self._validate()

    @classmethod
    def build(
        cls,
        embeddings: ArrayLike,
        *,
        num_clusters: int,
        iterations: int = 20,
    ) -> "VocabularyIVFIndex":
        matrix = _embedding_matrix(embeddings)
        clusters = _positive_integer(num_clusters, "num_clusters")
        iterations = _positive_integer(iterations, "iterations")
        if clusters > matrix.shape[0]:
            raise VocabularyIVFError("num_clusters cannot exceed vocabulary size")
        normalized = _normalize_rows(matrix)
        centroids, assignments = _fit_centroids(
            normalized.astype(np.float64), clusters, iterations
        )
        populations = np.bincount(assignments, minlength=clusters)
        offsets = np.empty(clusters + 1, dtype=np.uint32)
        offsets[0] = 0
        offsets[1:] = np.cumsum(populations, dtype=np.uint64).astype(np.uint32)
        tokens = np.lexsort(
            (np.arange(matrix.shape[0], dtype=np.uint32), assignments)
        ).astype(np.uint32, copy=False)
        return cls(
            centroids=centroids.astype(np.float32),
            posting_offsets=offsets,
            token_ids=tokens,
            build_iterations=iterations,
        )

    @property
    def vocabulary_size(self) -> int:
        return int(self.token_ids.size)

    @property
    def hidden_size(self) -> int:
        return int(self.centroid_vectors.shape[1])

    @property
    def centroids(self) -> int:
        return int(self.centroid_vectors.shape[0])

    @property
    def centroids_bytes(self) -> int:
        return int(self.centroid_vectors.nbytes)

    @property
    def postings_bytes(self) -> int:
        return int(self.posting_offsets.nbytes + self.token_ids.nbytes)

    @property
    def total_bytes(self) -> int:
        return self.centroids_bytes + self.postings_bytes

    def _validate(self) -> None:
        if self.centroid_vectors.ndim != 2 or not all(
            self.centroid_vectors.shape
        ):
            raise VocabularyIVFError("centroids must be a non-empty rank-2 matrix")
        if self.posting_offsets.shape != (self.centroids + 1,):
            raise VocabularyIVFError(
                "posting_offsets must contain centroids + 1 entries"
            )
        if self.token_ids.ndim != 1 or self.token_ids.size == 0:
            raise VocabularyIVFError("token_ids must be a non-empty vector")
        if self.posting_offsets[0] != 0 or self.posting_offsets[-1] != (
            self.vocabulary_size
        ):
            raise VocabularyIVFError("posting_offsets must span every token")
        if np.any(self.posting_offsets[1:] <= self.posting_offsets[:-1]):
            raise VocabularyIVFError("every centroid must have a non-empty posting")
        if np.any(self.token_ids >= self.vocabulary_size):
            raise VocabularyIVFError("a token ID is outside the vocabulary")
        if not np.array_equal(
            np.sort(self.token_ids),
            np.arange(self.vocabulary_size, dtype=np.uint32),
        ):
            raise VocabularyIVFError("token IDs must be a vocabulary permutation")
        for cluster in range(self.centroids):
            start, end = self.posting_offsets[cluster : cluster + 2]
            posting = self.token_ids[int(start) : int(end)]
            if np.any(np.diff(posting.astype(np.uint64)) == 0) or not np.array_equal(
                posting, np.sort(posting)
            ):
                raise VocabularyIVFError(
                    "token IDs within each posting must be strictly ordered"
                )

    def cluster_tokens(self, cluster: int) -> NDArray[np.uint32]:
        if (
            not isinstance(cluster, (int, np.integer))
            or isinstance(cluster, (bool, np.bool_))
            or cluster < 0
            or cluster >= self.centroids
        ):
            raise VocabularyIVFError("cluster index is out of range")
        start, end = self.posting_offsets[int(cluster) : int(cluster) + 2]
        return self.token_ids[int(start) : int(end)]

    def _runtime_embeddings(self, embeddings: ArrayLike) -> np.ndarray:
        matrix = np.asanyarray(embeddings)
        if matrix.ndim != 2 or matrix.shape != (
            self.vocabulary_size,
            self.hidden_size,
        ):
            raise VocabularyIVFError(
                "runtime embeddings shape does not match the IVF index"
            )
        if matrix.dtype != np.float32:
            raise VocabularyIVFError("runtime normalized embeddings must have dtype float32")
        return matrix

    def search(
        self,
        hidden: ArrayLike,
        embeddings: ArrayLike,
        *,
        candidate_count: int,
        minimum_probes: int = 1,
    ) -> VocabularyIVFResult:
        try:
            query = np.asarray(hidden, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise VocabularyIVFError("hidden must be numeric") from error
        if query.shape != (self.hidden_size,):
            raise VocabularyIVFError(
                f"hidden must have shape [{self.hidden_size}], got {query.shape}"
            )
        if not np.all(np.isfinite(query)):
            raise VocabularyIVFError("hidden must contain only finite values")
        matrix = self._runtime_embeddings(embeddings)
        requested = min(
            _positive_integer(candidate_count, "candidate_count"),
            self.vocabulary_size,
        )
        minimum = _positive_integer(minimum_probes, "minimum_probes")
        if minimum > self.centroids:
            raise VocabularyIVFError("minimum_probes cannot exceed centroid count")

        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            candidate_ids = np.arange(requested, dtype=np.uint32)
            return VocabularyIVFResult(
                candidate_ids=candidate_ids,
                proxy_scores=np.zeros(requested, dtype=np.float64),
                probed_clusters=np.empty(0, dtype=np.uint32),
                probes=0,
                expansions=0,
                probed_token_count=requested,
                proxy_scores_computed=0,
                bytes_read=0,
            )
        unit_query = query / query_norm
        centroid_norms = np.linalg.norm(self.centroid_vectors, axis=1)
        centroid_dots = self.centroid_vectors @ unit_query
        centroid_scores = np.divide(
            centroid_dots,
            centroid_norms,
            out=np.zeros_like(centroid_dots, dtype=np.float64),
            where=centroid_norms > 0.0,
        )
        cluster_ids = np.arange(self.centroids, dtype=np.uint32)
        cluster_order = np.lexsort((cluster_ids, -centroid_scores))

        probes = minimum
        posting_total = int(self.posting_offsets[cluster_order[:probes] + 1].sum())
        posting_total -= int(self.posting_offsets[cluster_order[:probes]].sum())
        while posting_total < requested and probes < self.centroids:
            cluster = int(cluster_order[probes])
            posting_total += int(
                self.posting_offsets[cluster + 1] - self.posting_offsets[cluster]
            )
            probes += 1
        selected_clusters = cluster_ids[cluster_order[:probes]]
        posting_parts = [self.cluster_tokens(int(cluster)) for cluster in selected_clusters]
        probed_ids = np.concatenate(posting_parts).astype(np.uint32, copy=False)

        selected_embeddings = np.asarray(matrix[probed_ids], dtype=np.float32)
        if not np.all(np.isfinite(selected_embeddings)):
            raise VocabularyIVFError(
                "a probed runtime embedding contains a non-finite value"
            )
        scores = selected_embeddings.astype(np.float64) @ unit_query
        order = np.lexsort((probed_ids, -scores))
        chosen = order[:requested]

        centroid_bytes = self.centroid_vectors.nbytes
        posting_bytes = probes * 2 * np.dtype(np.uint32).itemsize
        posting_bytes += probed_ids.size * np.dtype(np.uint32).itemsize
        embedding_bytes = (
            probed_ids.size * self.hidden_size * matrix.dtype.itemsize
        )
        return VocabularyIVFResult(
            candidate_ids=probed_ids[chosen],
            proxy_scores=np.asarray(scores[chosen], dtype=np.float64),
            probed_clusters=selected_clusters,
            probes=probes,
            expansions=probes - minimum,
            probed_token_count=int(probed_ids.size),
            proxy_scores_computed=int(probed_ids.size),
            bytes_read=int(centroid_bytes + posting_bytes + embedding_bytes),
        )

    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        arrays = {
            "centroids": self.centroid_vectors,
            "posting_offsets": self.posting_offsets,
            "token_ids": self.token_ids,
        }
        for name, array in arrays.items():
            np.save(target / f"{name}.npy", array, allow_pickle=False)
        atomic_json(
            target / "metadata.json",
            {
                "format": VOCABULARY_IVF_FORMAT,
                "schema_version": VOCABULARY_IVF_SCHEMA_VERSION,
                "vocabulary_size": self.vocabulary_size,
                "hidden_size": self.hidden_size,
                "centroids": self.centroids,
                "build_iterations": self.build_iterations,
                "centroid_dtype": "float32",
                "posting_dtype": "uint32",
                "assignment": "l2_normalized_embeddings",
                "centroid_score": "cosine",
                "record_score": "compiled_normalized_dot",
                "tie_breaks": {"cluster": "cluster_id", "token": "token_id"},
                "probe_policy": "expand_until_candidate_count",
                "zero_query": "global_ascending_token_ids",
                "files": {name: f"{name}.npy" for name in arrays},
            },
        )
        return target

    @classmethod
    def load(
        cls, directory: str | Path, *, mmap_mode: str | None = "r"
    ) -> "VocabularyIVFIndex":
        source = Path(directory)
        try:
            metadata = json.loads(
                (source / "metadata.json").read_text(encoding="utf-8")
            )
            if metadata.get("format") != VOCABULARY_IVF_FORMAT:
                raise VocabularyIVFError("unsupported vocabulary IVF format")
            if metadata.get("schema_version") != VOCABULARY_IVF_SCHEMA_VERSION:
                raise VocabularyIVFError("unsupported vocabulary IVF schema version")
            if metadata.get("centroid_dtype") != "float32":
                raise VocabularyIVFError("unsupported centroid dtype")
            if metadata.get("posting_dtype") != "uint32":
                raise VocabularyIVFError("unsupported posting dtype")
            if (
                metadata.get("assignment") != "l2_normalized_embeddings"
                or metadata.get("record_score") != "compiled_normalized_dot"
                or metadata.get("probe_policy")
                != "expand_until_candidate_count"
                or metadata.get("zero_query")
                != "global_ascending_token_ids"
            ):
                raise VocabularyIVFError("unsupported vocabulary IVF search contract")
            files = metadata["files"]
            expected = {"centroids", "posting_offsets", "token_ids"}
            if set(files) != expected:
                raise VocabularyIVFError("vocabulary IVF metadata file set is invalid")
            for name in expected:
                if files[name] != f"{name}.npy":
                    raise VocabularyIVFError(
                        f"unsafe or unexpected vocabulary IVF path for {name}"
                    )
            arrays = {
                name: np.load(
                    source / files[name], mmap_mode=mmap_mode, allow_pickle=False
                )
                for name in expected
            }
            result = cls(
                centroids=arrays["centroids"],
                posting_offsets=arrays["posting_offsets"],
                token_ids=arrays["token_ids"],
                build_iterations=metadata["build_iterations"],
            )
            if (
                metadata.get("vocabulary_size") != result.vocabulary_size
                or metadata.get("hidden_size") != result.hidden_size
                or metadata.get("centroids") != result.centroids
            ):
                raise VocabularyIVFError(
                    "vocabulary IVF metadata dimensions disagree with arrays"
                )
            return result
        except VocabularyIVFError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise VocabularyIVFError(
                f"cannot load vocabulary IVF index: {source}"
            ) from error


__all__ = ["VocabularyIVFError", "VocabularyIVFIndex", "VocabularyIVFResult"]
