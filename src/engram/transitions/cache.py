from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.utils import atomic_json, sha256_json


CACHE_SCHEMA_VERSION = 1
PutSource = Literal["offline", "online"]


class CacheFormatError(ValueError):
    """Raised for malformed or checksum-invalid persistent caches."""


@dataclass(frozen=True)
class StateFingerprint:
    """Hashable product of fixed-step scalar-quantized state blocks."""

    input_token: int
    blocks: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Transition:
    next_state: NDArray[np.float32]
    output_candidates: tuple[tuple[int, float], ...]
    confidence: float


@dataclass(frozen=True)
class CacheLookup:
    hit: bool
    reason: Literal["hit", "miss", "bypass", "radius_rejection"]
    transition: Transition | None
    state_distance: float | None


@dataclass(frozen=True)
class CacheMetrics:
    entries: int
    capacity: int
    lookups: int
    hits: int
    misses: int
    hit_rate: float
    radius_rejections: int
    collisions: int
    evictions: int
    offline_puts: int
    online_puts: int
    bypassed_lookups: int
    bypassed_puts: int
    approximation_error_samples: int
    mean_approximation_error: float | None
    max_approximation_error: float | None


@dataclass
class _MutableMetrics:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    radius_rejections: int = 0
    collisions: int = 0
    evictions: int = 0
    offline_puts: int = 0
    online_puts: int = 0
    bypassed_lookups: int = 0
    bypassed_puts: int = 0
    approximation_error_samples: int = 0
    approximation_error_sum: float = 0.0
    max_approximation_error: float = 0.0


@dataclass(frozen=True)
class _CacheEntry:
    reference_state: NDArray[np.float32]
    transition: Transition
    source: PutSource


def _relative_distance(left: NDArray[np.float32], right: NDArray[np.float32]) -> float:
    numerator = float(np.linalg.norm(left.astype(np.float64) - right.astype(np.float64)))
    denominator = max(
        float(np.linalg.norm(left.astype(np.float64))),
        float(np.linalg.norm(right.astype(np.float64))),
        1e-12,
    )
    return numerator / denominator


class TransitionCache:
    """Validated in-memory LRU keyed by quantized state and input token.

    Each state dimension is rounded to a configured scalar step and clipped to
    int16. Codes are grouped into fixed-width blocks, making the fingerprint a
    deterministic product of subvector codes. The exact float32 reference state
    remains in the entry: a fingerprint match is reused only when its symmetric
    relative L2 distance does not exceed ``similarity_radius``.
    """

    def __init__(
        self,
        state_width: int,
        *,
        capacity: int = 1024,
        quantization_step: float = 0.125,
        subvector_width: int = 8,
        similarity_radius: float = 0.05,
        bypass: bool = False,
    ) -> None:
        for value, name in (
            (state_width, "state_width"),
            (capacity, "capacity"),
            (subvector_width, "subvector_width"),
        ):
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be an integer")
        if state_width <= 0 or capacity <= 0 or subvector_width <= 0:
            raise ValueError("state_width, capacity, and subvector_width must be positive")
        if not np.isfinite(quantization_step) or quantization_step <= 0.0:
            raise ValueError("quantization_step must be finite and positive")
        if not np.isfinite(similarity_radius) or similarity_radius < 0.0:
            raise ValueError("similarity_radius must be finite and non-negative")

        self.state_width = int(state_width)
        self.capacity = int(capacity)
        self.quantization_step = float(quantization_step)
        self.subvector_width = int(subvector_width)
        self.similarity_radius = float(similarity_radius)
        self.bypass = bool(bypass)
        self._entries: OrderedDict[StateFingerprint, _CacheEntry] = OrderedDict()
        self._metrics = _MutableMetrics()

    def _state(self, state: ArrayLike, name: str) -> NDArray[np.float32]:
        try:
            result = np.asarray(state, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be numeric") from error
        if result.ndim != 1 or result.shape[0] != self.state_width:
            raise ValueError(f"{name} must have shape [{self.state_width}], got {result.shape}")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain only finite values")
        return result.copy()

    @staticmethod
    def _token(input_token: int) -> int:
        if (
            not isinstance(input_token, (int, np.integer))
            or isinstance(input_token, (bool, np.bool_))
            or input_token < 0
        ):
            raise ValueError("input_token must be a non-negative integer")
        return int(input_token)

    def fingerprint(self, state: ArrayLike, input_token: int) -> StateFingerprint:
        state_array = self._state(state, "state")
        token = self._token(input_token)
        codes = np.clip(
            np.rint(state_array.astype(np.float64) / self.quantization_step),
            np.iinfo(np.int16).min,
            np.iinfo(np.int16).max,
        ).astype(np.int16)
        blocks = tuple(
            tuple(int(value) for value in codes[start : start + self.subvector_width])
            for start in range(0, self.state_width, self.subvector_width)
        )
        return StateFingerprint(input_token=token, blocks=blocks)

    def _transition(
        self,
        next_state: ArrayLike,
        output_candidates: Iterable[Sequence[float | int]],
        confidence: float,
    ) -> Transition:
        state = self._state(next_state, "next_state")
        if not np.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence must be finite and lie in [0, 1]")
        candidates: list[tuple[int, float]] = []
        for item in output_candidates:
            try:
                pair = tuple(item)
            except TypeError as error:
                raise ValueError(
                    "each output candidate must be a (token, score) pair"
                ) from error
            if len(pair) != 2:
                raise ValueError("each output candidate must be a (token, score) pair")
            token = self._token(pair[0])
            score = float(pair[1])
            if not np.isfinite(score):
                raise ValueError("candidate scores must be finite")
            candidates.append((token, score))
        if not candidates:
            raise ValueError("output_candidates must not be empty")
        state.setflags(write=False)
        return Transition(
            next_state=state,
            output_candidates=tuple(candidates),
            confidence=float(confidence),
        )

    def put(
        self,
        state: ArrayLike,
        input_token: int,
        next_state: ArrayLike,
        output_candidates: Iterable[Sequence[float | int]],
        confidence: float,
        *,
        source: PutSource = "online",
    ) -> bool:
        """Insert or update an entry; return false when bypass suppresses it."""

        if source not in {"offline", "online"}:
            raise ValueError("source must be 'offline' or 'online'")
        reference = self._state(state, "state")
        fingerprint = self.fingerprint(reference, input_token)
        transition = self._transition(next_state, output_candidates, confidence)
        if self.bypass:
            self._metrics.bypassed_puts += 1
            return False

        previous = self._entries.get(fingerprint)
        if previous is not None:
            if _relative_distance(reference, previous.reference_state) > self.similarity_radius:
                self._metrics.collisions += 1
            del self._entries[fingerprint]
        elif len(self._entries) >= self.capacity:
            self._entries.popitem(last=False)
            self._metrics.evictions += 1

        reference.setflags(write=False)
        self._entries[fingerprint] = _CacheEntry(reference, transition, source)
        if source == "offline":
            self._metrics.offline_puts += 1
        else:
            self._metrics.online_puts += 1
        return True

    def put_offline(
        self,
        state: ArrayLike,
        input_token: int,
        next_state: ArrayLike,
        output_candidates: Iterable[Sequence[float | int]],
        confidence: float,
    ) -> bool:
        return self.put(
            state,
            input_token,
            next_state,
            output_candidates,
            confidence,
            source="offline",
        )

    def put_online(
        self,
        state: ArrayLike,
        input_token: int,
        next_state: ArrayLike,
        output_candidates: Iterable[Sequence[float | int]],
        confidence: float,
    ) -> bool:
        return self.put(
            state,
            input_token,
            next_state,
            output_candidates,
            confidence,
            source="online",
        )

    def lookup(
        self,
        state: ArrayLike,
        input_token: int,
        *,
        actual_next_state: ArrayLike | None = None,
    ) -> CacheLookup:
        """Look up and radius-validate an entry, optionally measuring reuse error."""

        query = self._state(state, "state")
        token = self._token(input_token)
        self._metrics.lookups += 1
        if self.bypass:
            self._metrics.misses += 1
            self._metrics.bypassed_lookups += 1
            return CacheLookup(False, "bypass", None, None)

        fingerprint = self.fingerprint(query, token)
        entry = self._entries.get(fingerprint)
        if entry is None:
            self._metrics.misses += 1
            return CacheLookup(False, "miss", None, None)
        distance = _relative_distance(query, entry.reference_state)
        if distance > self.similarity_radius:
            self._metrics.misses += 1
            self._metrics.radius_rejections += 1
            self._metrics.collisions += 1
            return CacheLookup(False, "radius_rejection", None, distance)

        self._entries.move_to_end(fingerprint)
        self._metrics.hits += 1
        if actual_next_state is not None:
            actual = self._state(actual_next_state, "actual_next_state")
            error = _relative_distance(actual, entry.transition.next_state)
            self._metrics.approximation_error_samples += 1
            self._metrics.approximation_error_sum += error
            self._metrics.max_approximation_error = max(
                self._metrics.max_approximation_error, error
            )
        return CacheLookup(True, "hit", entry.transition, distance)

    get = lookup

    def clear(self) -> None:
        self._entries.clear()

    def set_bypass(self, enabled: bool) -> None:
        self.bypass = bool(enabled)

    @property
    def metrics(self) -> CacheMetrics:
        samples = self._metrics.approximation_error_samples
        mean_error = (
            self._metrics.approximation_error_sum / samples if samples else None
        )
        return CacheMetrics(
            entries=len(self._entries),
            capacity=self.capacity,
            lookups=self._metrics.lookups,
            hits=self._metrics.hits,
            misses=self._metrics.misses,
            hit_rate=self._metrics.hits / self._metrics.lookups if self._metrics.lookups else 0.0,
            radius_rejections=self._metrics.radius_rejections,
            collisions=self._metrics.collisions,
            evictions=self._metrics.evictions,
            offline_puts=self._metrics.offline_puts,
            online_puts=self._metrics.online_puts,
            bypassed_lookups=self._metrics.bypassed_lookups,
            bypassed_puts=self._metrics.bypassed_puts,
            approximation_error_samples=samples,
            mean_approximation_error=mean_error,
            max_approximation_error=(self._metrics.max_approximation_error if samples else None),
        )

    def _body(self) -> dict[str, object]:
        entries = []
        for fingerprint, entry in self._entries.items():
            entries.append(
                {
                    "input_token": fingerprint.input_token,
                    "fingerprint_blocks": [list(block) for block in fingerprint.blocks],
                    "reference_state": entry.reference_state.tolist(),
                    "next_state": entry.transition.next_state.tolist(),
                    "output_candidates": [list(item) for item in entry.transition.output_candidates],
                    "confidence": entry.transition.confidence,
                    "source": entry.source,
                }
            )
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "config": {
                "state_width": self.state_width,
                "capacity": self.capacity,
                "quantization_step": self.quantization_step,
                "subvector_width": self.subvector_width,
                "similarity_radius": self.similarity_radius,
            },
            "entries": entries,
        }

    def save(self, path: str | Path) -> Path:
        """Atomically save LRU order and entries with a canonical JSON checksum."""

        destination = Path(path)
        body = self._body()
        atomic_json(destination, {**body, "sha256": sha256_json(body)})
        return destination

    @classmethod
    def load(cls, path: str | Path, *, bypass: bool = False) -> "TransitionCache":
        source = Path(path)
        try:
            import json

            document = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as error:
            raise CacheFormatError(f"cannot read transition cache: {source}") from error
        if not isinstance(document, Mapping):
            raise CacheFormatError("transition cache root must be an object")
        checksum = document.get("sha256")
        body = {key: value for key, value in document.items() if key != "sha256"}
        if not isinstance(checksum, str) or sha256_json(body) != checksum:
            raise CacheFormatError("transition cache checksum mismatch")
        if body.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise CacheFormatError(
                f"unsupported transition cache schema {body.get('schema_version')}"
            )
        config = body.get("config")
        entries = body.get("entries")
        if not isinstance(config, Mapping) or not isinstance(entries, list):
            raise CacheFormatError("transition cache config or entries are malformed")
        try:
            cache = cls(
                state_width=config["state_width"],
                capacity=config["capacity"],
                quantization_step=config["quantization_step"],
                subvector_width=config["subvector_width"],
                similarity_radius=config["similarity_radius"],
                bypass=False,
            )
            if len(entries) > cache.capacity:
                raise CacheFormatError("transition cache has more entries than capacity")
            for item in entries:
                if not isinstance(item, Mapping):
                    raise CacheFormatError("transition cache entry is malformed")
                expected = StateFingerprint(
                    input_token=item["input_token"],
                    blocks=tuple(tuple(block) for block in item["fingerprint_blocks"]),
                )
                actual = cache.fingerprint(item["reference_state"], item["input_token"])
                if expected != actual:
                    raise CacheFormatError("transition cache fingerprint mismatch")
                cache.put(
                    item["reference_state"],
                    item["input_token"],
                    item["next_state"],
                    item["output_candidates"],
                    item["confidence"],
                    source=item["source"],
                )
        except CacheFormatError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise CacheFormatError("transition cache payload is malformed") from error
        cache._metrics = _MutableMetrics()
        cache.bypass = bool(bypass)
        return cache
