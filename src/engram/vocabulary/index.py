"""NumPy reference vocabulary maximum-inner-product index.

The coarse pass ranks normalized embeddings, while every selected candidate is
rescored with the original embedding and optional bias.  Normalized search is
only an approximation to maximum inner product when embedding norms differ, so
all public results retain an explicit exact/approximate marker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.vocabulary.ivf import VocabularyIVFIndex


class VocabularyIndexError(ValueError):
    """Raised for invalid vocabulary-index inputs or options."""


def _finite_matrix(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise VocabularyIndexError(f"{name} must be numeric") from error
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise VocabularyIndexError(f"{name} must be a non-empty rank-2 matrix")
    if not np.all(np.isfinite(array)):
        raise VocabularyIndexError(f"{name} must contain only finite values")
    return array


def _finite_vector(
    values: ArrayLike, dimension: int, *, name: str
) -> NDArray[np.float64]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise VocabularyIndexError(f"{name} must be numeric") from error
    if array.shape != (dimension,):
        raise VocabularyIndexError(f"{name} must have shape ({dimension},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise VocabularyIndexError(f"{name} must contain only finite values")
    return array


def _positive_integer(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise VocabularyIndexError(f"{name} must be a positive integer")
    return value


def exact_logits(
    hidden: ArrayLike, embeddings: ArrayLike, bias: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Compute the exact dense vocabulary projection in float64."""

    embedding_matrix = _finite_matrix(embeddings, name="embeddings")
    hidden_vector = _finite_vector(hidden, embedding_matrix.shape[1], name="hidden")
    logits = embedding_matrix @ hidden_vector
    if bias is not None:
        bias_vector = _finite_vector(bias, embedding_matrix.shape[0], name="bias")
        logits = logits + bias_vector
    return np.asarray(logits, dtype=np.float64)


def _descending_order(values: np.ndarray) -> NDArray[np.int64]:
    return np.asarray(np.argsort(-values, kind="stable"), dtype=np.int64)


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise VocabularyIndexError("temperature must be finite and positive")
    shifted = logits / temperature
    shifted -= np.max(shifted)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


@dataclass(frozen=True)
class SearchResult:
    token_ids: NDArray[np.int64]
    logits: NDArray[np.float64]
    candidate_ids: NDArray[np.int64]
    candidate_logits: NDArray[np.float64]
    candidate_count: int
    expansions: int
    confidence_margin: float
    confidence_satisfied: bool
    exact: bool
    proxy_count: int = 0
    probed_clusters: int = 0
    index_bytes_read: int = 0

    @property
    def approximate(self) -> bool:
        return not self.exact


@dataclass(frozen=True)
class VocabularyMetrics:
    query_count: int
    candidate_count: int
    top1_recall: float
    top_k_recall: dict[int, float]
    mean_top1_logit_error: float
    max_top1_logit_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_count": self.query_count,
            "candidate_count": self.candidate_count,
            "top1_recall": self.top1_recall,
            "top_k_recall": {str(k): value for k, value in self.top_k_recall.items()},
            "mean_top1_logit_error": self.mean_top1_logit_error,
            "max_top1_logit_error": self.max_top1_logit_error,
        }


@dataclass(frozen=True)
class GenerationResult:
    token_id: int
    logit: float
    method: str
    exact_distribution: bool
    approximate_distribution: bool
    exact_fallback_used: bool
    candidate_count: int
    support_size: int
    proxy_count: int = 0
    probed_clusters: int = 0
    index_bytes_read: int = 0


class VocabularyIndex:
    """Deterministic normalized coarse search with exact candidate rescoring."""

    def __init__(
        self,
        embeddings: ArrayLike,
        bias: ArrayLike | None = None,
        *,
        normalized_embeddings: ArrayLike | None = None,
        ivf_index: VocabularyIVFIndex | None = None,
    ) -> None:
        matrix = _finite_matrix(embeddings, name="embeddings")
        self.embeddings = np.asarray(matrix, dtype=np.float64)
        self.vocabulary_size, self.hidden_dimension = self.embeddings.shape
        if bias is None:
            self.bias = None
        else:
            self.bias = _finite_vector(bias, self.vocabulary_size, name="bias")
        if normalized_embeddings is None:
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.normalized_embeddings = np.divide(
                self.embeddings,
                norms,
                out=np.zeros_like(self.embeddings),
                where=norms > 0.0,
            )
        else:
            normalized = np.asanyarray(normalized_embeddings)
            if normalized.dtype != np.float32 or normalized.shape != self.embeddings.shape:
                raise VocabularyIndexError(
                    "normalized_embeddings must be float32 with the embedding shape"
                )
            if not np.all(np.isfinite(normalized)):
                raise VocabularyIndexError("normalized_embeddings must be finite")
            self.normalized_embeddings = normalized
        self.ivf_index = ivf_index
        if self.ivf_index is not None and (
            self.ivf_index.vocabulary_size != self.vocabulary_size
            or self.ivf_index.hidden_size != self.hidden_dimension
        ):
            raise VocabularyIndexError("vocabulary IVF dimensions disagree with embeddings")

    def exact_logits(self, hidden: ArrayLike) -> NDArray[np.float64]:
        hidden_vector = self._hidden(hidden)
        logits = self.embeddings @ hidden_vector
        if self.bias is not None:
            logits = logits + self.bias
        return np.asarray(logits, dtype=np.float64)

    def exact_top_k(
        self, hidden: ArrayLike, *, k: int = 1
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        k = _positive_integer(k, name="k")
        if k > self.vocabulary_size:
            raise VocabularyIndexError("k cannot exceed vocabulary size")
        logits = self.exact_logits(hidden)
        token_ids = _descending_order(logits)[:k]
        return token_ids, logits[token_ids]

    def _hidden(self, hidden: ArrayLike) -> NDArray[np.float64]:
        return _finite_vector(hidden, self.hidden_dimension, name="hidden")

    def _coarse_candidates(
        self,
        hidden: NDArray[np.float64],
        candidate_count: int,
        minimum_probes: int,
    ) -> tuple[NDArray[np.int64], int, int, int]:
        if self.ivf_index is not None:
            result = self.ivf_index.search(
                hidden,
                self.normalized_embeddings,
                candidate_count=candidate_count,
                minimum_probes=minimum_probes,
            )
            return (
                result.candidate_ids.astype(np.int64, copy=False),
                result.proxy_scores_computed,
                result.probes,
                result.bytes_read,
            )
        norm = float(np.linalg.norm(hidden))
        normalized_hidden = hidden / norm if norm > 0.0 else np.zeros_like(hidden)
        coarse_scores = self.normalized_embeddings @ normalized_hidden
        return (
            _descending_order(coarse_scores)[:candidate_count],
            self.vocabulary_size,
            0,
            int(self.normalized_embeddings.nbytes),
        )

    def _rescore(
        self, hidden: NDArray[np.float64], candidate_ids: NDArray[np.int64]
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        logits = self.embeddings[candidate_ids] @ hidden
        if self.bias is not None:
            logits = logits + self.bias[candidate_ids]
        order = _descending_order(logits)
        return candidate_ids[order], np.asarray(logits[order], dtype=np.float64)

    def search(
        self,
        hidden: ArrayLike,
        *,
        candidate_count: int = 64,
        top_k: int = 5,
        minimum_confidence_margin: float | None = None,
        expansion_factor: int = 2,
        max_candidates: int | None = None,
        minimum_probes: int = 1,
    ) -> SearchResult:
        """Retrieve candidates and exactly rescore them.

        Confidence is the exact-rescored gap between the best two retrieved
        logits.  If it is below ``minimum_confidence_margin``, the coarse set is
        geometrically expanded up to ``max_candidates``.
        """

        hidden_vector = self._hidden(hidden)
        candidate_count = _positive_integer(candidate_count, name="candidate_count")
        top_k = _positive_integer(top_k, name="top_k")
        expansion_factor = _positive_integer(expansion_factor, name="expansion_factor")
        if expansion_factor < 2:
            raise VocabularyIndexError("expansion_factor must be at least 2")
        if top_k > self.vocabulary_size:
            raise VocabularyIndexError("top_k cannot exceed vocabulary size")
        if max_candidates is None:
            limit = self.vocabulary_size
        else:
            limit = _positive_integer(max_candidates, name="max_candidates")
            limit = min(limit, self.vocabulary_size)
        if limit < top_k:
            raise VocabularyIndexError("max_candidates cannot be smaller than top_k")
        count = min(max(candidate_count, top_k), limit)
        if minimum_confidence_margin is not None:
            try:
                threshold = float(minimum_confidence_margin)
            except (TypeError, ValueError) as error:
                raise VocabularyIndexError(
                    "minimum_confidence_margin must be finite and nonnegative"
                ) from error
            if not np.isfinite(threshold) or threshold < 0.0:
                raise VocabularyIndexError(
                    "minimum_confidence_margin must be finite and nonnegative"
                )
        else:
            threshold = None

        expansions = 0
        while True:
            coarse_ids, proxy_count, probed_clusters, index_bytes = self._coarse_candidates(
                hidden_vector, count, minimum_probes
            )
            ranked_ids, ranked_logits = self._rescore(hidden_vector, coarse_ids)
            margin = (
                float(ranked_logits[0] - ranked_logits[1])
                if ranked_logits.size >= 2
                else 0.0
            )
            satisfied = threshold is None or margin >= threshold
            if satisfied or count >= limit:
                break
            next_count = min(limit, max(count + 1, count * expansion_factor))
            if next_count == count:
                break
            count = next_count
            expansions += 1
        return SearchResult(
            token_ids=ranked_ids[:top_k],
            logits=ranked_logits[:top_k],
            candidate_ids=ranked_ids,
            candidate_logits=ranked_logits,
            candidate_count=count,
            expansions=expansions,
            confidence_margin=margin,
            confidence_satisfied=satisfied,
            exact=count == self.vocabulary_size,
            proxy_count=proxy_count,
            probed_clusters=probed_clusters,
            index_bytes_read=index_bytes,
        )

    def evaluate(
        self,
        hidden_states: ArrayLike,
        *,
        candidate_count: int = 64,
        top_ks: Iterable[int] = (1, 5),
    ) -> VocabularyMetrics:
        states = _finite_matrix(hidden_states, name="hidden_states")
        if states.shape[1] != self.hidden_dimension:
            raise VocabularyIndexError(
                f"hidden_states width must be {self.hidden_dimension}"
            )
        requested = sorted(set(_positive_integer(k, name="top_k") for k in top_ks))
        if not requested:
            raise VocabularyIndexError("top_ks must not be empty")
        if requested[-1] > self.vocabulary_size:
            raise VocabularyIndexError("top_k cannot exceed vocabulary size")
        candidate_count = _positive_integer(candidate_count, name="candidate_count")
        actual_candidate_count = min(
            max(candidate_count, requested[-1]), self.vocabulary_size
        )
        recalls = {k: [] for k in requested}
        top1_hits: list[float] = []
        logit_errors: list[float] = []
        for hidden in states:
            exact_values = self.exact_logits(hidden)
            exact_order = _descending_order(exact_values)
            result = self.search(
                hidden,
                candidate_count=actual_candidate_count,
                top_k=requested[-1],
                max_candidates=actual_candidate_count,
            )
            candidate_set = set(int(index) for index in result.candidate_ids)
            top1_hits.append(float(int(exact_order[0]) in candidate_set))
            for k in requested:
                exact_ids = set(int(index) for index in exact_order[:k])
                recalls[k].append(len(exact_ids & candidate_set) / k)
            logit_errors.append(
                abs(float(exact_values[exact_order[0]]) - float(result.candidate_logits[0]))
            )
        return VocabularyMetrics(
            query_count=states.shape[0],
            candidate_count=actual_candidate_count,
            top1_recall=float(np.mean(top1_hits)),
            top_k_recall={k: float(np.mean(values)) for k, values in recalls.items()},
            mean_top1_logit_error=float(np.mean(logit_errors)),
            max_top1_logit_error=float(np.max(logit_errors)),
        )

    def greedy(
        self,
        hidden: ArrayLike,
        *,
        exact: bool = False,
        candidate_count: int = 64,
        minimum_confidence_margin: float | None = None,
        minimum_probes: int = 1,
    ) -> GenerationResult:
        if exact:
            token_ids, logits = self.exact_top_k(hidden, k=1)
            count = self.vocabulary_size
            is_exact = True
            proxy_count = 0
            probes = 0
            index_bytes = 0
        else:
            result = self.search(
                hidden,
                candidate_count=candidate_count,
                top_k=1,
                minimum_confidence_margin=minimum_confidence_margin,
                minimum_probes=minimum_probes,
            )
            token_ids, logits = result.token_ids, result.logits
            count, is_exact = result.candidate_count, result.exact
            proxy_count = result.proxy_count
            probes = result.probed_clusters
            index_bytes = result.index_bytes_read
        return GenerationResult(
            token_id=int(token_ids[0]),
            logit=float(logits[0]),
            method="greedy",
            exact_distribution=is_exact,
            approximate_distribution=not is_exact,
            exact_fallback_used=exact,
            candidate_count=count,
            support_size=1,
            proxy_count=proxy_count,
            probed_clusters=probes,
            index_bytes_read=index_bytes,
        )

    def sample_top_k(
        self,
        hidden: ArrayLike,
        *,
        k: int,
        exact: bool = False,
        candidate_count: int = 64,
        temperature: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> GenerationResult:
        k = _positive_integer(k, name="k")
        if k > self.vocabulary_size:
            raise VocabularyIndexError("k cannot exceed vocabulary size")
        if exact:
            token_ids, logits = self.exact_top_k(hidden, k=k)
            count = self.vocabulary_size
            is_exact = True
        else:
            result = self.search(hidden, candidate_count=candidate_count, top_k=k)
            token_ids, logits = result.token_ids, result.logits
            count, is_exact = result.candidate_count, result.exact
        probabilities = _softmax(logits, temperature)
        generator = np.random.default_rng(0) if rng is None else rng
        selected = int(generator.choice(len(token_ids), p=probabilities))
        return GenerationResult(
            token_id=int(token_ids[selected]),
            logit=float(logits[selected]),
            method="top_k",
            exact_distribution=is_exact,
            approximate_distribution=not is_exact,
            exact_fallback_used=exact,
            candidate_count=count,
            support_size=len(token_ids),
        )

    def sample_top_p(
        self,
        hidden: ArrayLike,
        *,
        top_p: float,
        exact: bool = False,
        candidate_count: int = 256,
        temperature: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> GenerationResult:
        try:
            probability_mass = float(top_p)
        except (TypeError, ValueError) as error:
            raise VocabularyIndexError("top_p must be finite and in (0, 1]") from error
        if not np.isfinite(probability_mass) or not 0.0 < probability_mass <= 1.0:
            raise VocabularyIndexError("top_p must be finite and in (0, 1]")
        if exact:
            logits = self.exact_logits(hidden)
            token_ids = _descending_order(logits)
            ranked_logits = logits[token_ids]
            count = self.vocabulary_size
            is_exact = True
        else:
            requested = min(_positive_integer(candidate_count, name="candidate_count"), self.vocabulary_size)
            result = self.search(
                hidden,
                candidate_count=requested,
                top_k=requested,
                max_candidates=requested,
            )
            token_ids = result.candidate_ids
            ranked_logits = result.candidate_logits
            count, is_exact = result.candidate_count, result.exact
        probabilities = _softmax(ranked_logits, temperature)
        cumulative = np.cumsum(probabilities)
        support_size = min(
            len(probabilities), int(np.searchsorted(cumulative, probability_mass, side="left")) + 1
        )
        nucleus_probabilities = probabilities[:support_size]
        nucleus_probabilities = nucleus_probabilities / np.sum(nucleus_probabilities)
        generator = np.random.default_rng(0) if rng is None else rng
        selected = int(generator.choice(support_size, p=nucleus_probabilities))
        return GenerationResult(
            token_id=int(token_ids[selected]),
            logit=float(ranked_logits[selected]),
            method="top_p_exact" if is_exact else "top_p_approximate",
            exact_distribution=is_exact,
            approximate_distribution=not is_exact,
            exact_fallback_used=exact,
            candidate_count=count,
            support_size=support_size,
        )


__all__ = [
    "GenerationResult",
    "SearchResult",
    "VocabularyIndex",
    "VocabularyIndexError",
    "VocabularyMetrics",
    "exact_logits",
]
