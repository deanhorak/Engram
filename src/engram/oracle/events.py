"""Append-only revisioned event streams for Oracle sessions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Any, Iterable, Protocol

from .models import _identifier


class RevisionConflict(RuntimeError):
    """Raised when an append loses an optimistic-concurrency race."""


class EventConflict(RuntimeError):
    """Raised when an event ID is reused with different content."""


class EventKind(str, Enum):
    SESSION_STARTED = "session_started"
    ACTION_SELECTED = "action_selected"
    OUTCOME_OBSERVED = "outcome_observed"


@dataclass(frozen=True)
class PendingEvent:
    event_id: str
    kind: EventKind
    payload: Any
    schema_version: int = 1

    def __post_init__(self) -> None:
        _identifier(self.event_id, "event_id")
        if not isinstance(self.kind, EventKind):
            raise ValueError("kind must be an EventKind")
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int) or self.schema_version <= 0:
            raise ValueError("schema_version must be a positive integer")


@dataclass(frozen=True)
class OracleEvent:
    stream_id: str
    revision: int
    event_id: str
    kind: EventKind
    payload: Any
    schema_version: int = 1

    def __post_init__(self) -> None:
        _identifier(self.stream_id, "stream_id")
        _identifier(self.event_id, "event_id")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision <= 0:
            raise ValueError("event revision must be a positive integer")
        if not isinstance(self.kind, EventKind):
            raise ValueError("kind must be an EventKind")
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int) or self.schema_version <= 0:
            raise ValueError("schema_version must be a positive integer")


class OracleEventStore(Protocol):
    """Storage contract required by ``OracleSession``."""

    def read(self, stream_id: str) -> tuple[OracleEvent, ...]: ...

    def append(
        self,
        stream_id: str,
        *,
        expected_revision: int,
        events: Iterable[PendingEvent],
    ) -> tuple[OracleEvent, ...]: ...


class InMemoryOracleEventStore:
    """Thread-safe compare-and-swap event store used by the reference executive."""

    def __init__(self) -> None:
        self._streams: dict[str, tuple[OracleEvent, ...]] = {}
        self._lock = RLock()

    def read(self, stream_id: str) -> tuple[OracleEvent, ...]:
        _identifier(stream_id, "stream_id")
        with self._lock:
            return self._streams.get(stream_id, ())

    def append(
        self,
        stream_id: str,
        *,
        expected_revision: int,
        events: Iterable[PendingEvent],
    ) -> tuple[OracleEvent, ...]:
        _identifier(stream_id, "stream_id")
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("expected_revision must be a non-negative integer")
        pending = tuple(events)
        if not pending:
            raise ValueError("append requires at least one event")
        if len({event.event_id for event in pending}) != len(pending):
            raise EventConflict("an append batch cannot repeat an event ID")
        with self._lock:
            current = self._streams.get(stream_id, ())
            by_id = {event.event_id: event for event in current}
            duplicates = [event for event in pending if event.event_id in by_id]
            if duplicates:
                if len(duplicates) != len(pending):
                    raise EventConflict("an append cannot mix existing and new event IDs")
                existing = tuple(by_id[event.event_id] for event in pending)
                if all(
                    old.kind is new.kind
                    and old.payload == new.payload
                    and old.schema_version == new.schema_version
                    for old, new in zip(existing, pending)
                ):
                    return existing
                raise EventConflict("event ID was reused with different content")
            if expected_revision != len(current):
                raise RevisionConflict(
                    f"expected stream revision {expected_revision}, found {len(current)}"
                )
            appended = tuple(
                OracleEvent(
                    stream_id=stream_id,
                    revision=len(current) + offset,
                    event_id=event.event_id,
                    kind=event.kind,
                    payload=event.payload,
                    schema_version=event.schema_version,
                )
                for offset, event in enumerate(pending, start=1)
            )
            self._streams[stream_id] = current + appended
            return appended


__all__ = [
    "EventConflict",
    "EventKind",
    "InMemoryOracleEventStore",
    "OracleEvent",
    "OracleEventStore",
    "PendingEvent",
    "RevisionConflict",
]
