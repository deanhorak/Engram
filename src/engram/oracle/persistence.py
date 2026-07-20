"""Durable SQLite and JSONL event stores for Oracle sessions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Iterator

from .events import (
    EventConflict,
    EventKind,
    OracleEvent,
    PendingEvent,
    RevisionConflict,
)
from .models import (
    ActionProposal,
    AttentionCandidate,
    AttentionSelection,
    ConfidenceEstimate,
    DecisionPolicy,
    Evidence,
    Goal,
    GoalGraph,
    GoalStatus,
    MemoryCandidate,
    MemoryDecision,
    MemoryDisposition,
    MonitorDecision,
    MonitorStatus,
    OracleDecision,
    ProgressObservation,
    ScoredAction,
    _identifier,
)
from .session import (
    ActionOutcome,
    ActionSelected,
    OutcomeObserved,
    OutcomeStatus,
    PendingAttempt,
    ResourceBudget,
    SessionStarted,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts
    fcntl = None


EVENT_STORE_FORMAT = "engram.oracle.events"
EVENT_STORE_VERSION = 1


class EventLogCorruption(RuntimeError):
    """Raised when durable event data fails structural validation."""


_DATACLASSES = {
    cls.__name__: cls
    for cls in (
        ActionOutcome,
        ActionProposal,
        ActionSelected,
        AttentionCandidate,
        AttentionSelection,
        ConfidenceEstimate,
        DecisionPolicy,
        Evidence,
        Goal,
        GoalGraph,
        MemoryCandidate,
        MemoryDecision,
        MonitorDecision,
        OracleDecision,
        OutcomeObserved,
        PendingAttempt,
        ProgressObservation,
        ResourceBudget,
        ScoredAction,
        SessionStarted,
    )
}
_ENUMS = {
    cls.__name__: cls
    for cls in (GoalStatus, MemoryDisposition, MonitorStatus, OutcomeStatus)
}


def _encode(value: Any) -> Any:
    if isinstance(value, Enum):
        name = type(value).__name__
        if name not in _ENUMS or _ENUMS[name] is not type(value):
            raise TypeError(f"unsupported event enum: {type(value).__name__}")
        return {"$enum": name, "value": value.value}
    if is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if name not in _DATACLASSES or _DATACLASSES[name] is not type(value):
            raise TypeError(f"unsupported event dataclass: {type(value).__name__}")
        return {
            "$type": name,
            "fields": {field.name: _encode(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, tuple):
        return {"$tuple": [_encode(item) for item in value]}
    if isinstance(value, list):
        return {"$list": [_encode(item) for item in value]}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("event dictionaries must use string keys")
        return {"$dict": {key: _encode(item) for key, item in sorted(value.items())}}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("event floats must be finite")
        return value
    raise TypeError(f"unsupported event value: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if not isinstance(value, dict):
        raise EventLogCorruption("encoded event value must be an object or scalar")
    if set(value) == {"$enum", "value"}:
        enum_type = _ENUMS.get(value["$enum"])
        if enum_type is None:
            raise EventLogCorruption(f"unknown event enum {value['$enum']!r}")
        try:
            return enum_type(value["value"])
        except (TypeError, ValueError) as error:
            raise EventLogCorruption("invalid event enum value") from error
    if set(value) == {"$type", "fields"}:
        data_type = _DATACLASSES.get(value["$type"])
        if data_type is None or not isinstance(value["fields"], dict):
            raise EventLogCorruption(f"unknown event dataclass {value['$type']!r}")
        expected = {field.name for field in fields(data_type)}
        if set(value["fields"]) != expected:
            raise EventLogCorruption(f"event fields do not match {value['$type']}")
        try:
            return data_type(**{key: _decode(item) for key, item in value["fields"].items()})
        except (TypeError, ValueError) as error:
            raise EventLogCorruption(f"invalid {value['$type']} payload") from error
    if set(value) == {"$tuple"} and isinstance(value["$tuple"], list):
        return tuple(_decode(item) for item in value["$tuple"])
    if set(value) == {"$list"} and isinstance(value["$list"], list):
        return [_decode(item) for item in value["$list"]]
    if set(value) == {"$dict"} and isinstance(value["$dict"], dict):
        return {key: _decode(item) for key, item in value["$dict"].items()}
    raise EventLogCorruption("event value contains an unknown encoding marker")


def encode_event_payload(payload: Any) -> str:
    return json.dumps(_encode(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def decode_event_payload(payload: str) -> Any:
    try:
        encoded = json.loads(payload)
    except (json.JSONDecodeError, TypeError) as error:
        raise EventLogCorruption("event payload is not valid JSON") from error
    return _decode(encoded)


def _validate_append(
    stream_id: str,
    current: tuple[OracleEvent, ...],
    expected_revision: int,
    pending: tuple[PendingEvent, ...],
) -> tuple[OracleEvent, ...]:
    if (
        isinstance(expected_revision, bool)
        or not isinstance(expected_revision, int)
        or expected_revision < 0
    ):
        raise ValueError("expected_revision must be a non-negative integer")
    if not pending:
        raise ValueError("append requires at least one event")
    if any(event.schema_version != 1 for event in pending):
        raise ValueError("unsupported Oracle event schema version")
    if len({event.event_id for event in pending}) != len(pending):
        raise EventConflict("an append batch cannot repeat an event ID")
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
    return tuple(
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


def _validate_stream(events: tuple[OracleEvent, ...], stream_id: str) -> None:
    event_ids: set[str] = set()
    for expected, event in enumerate(events, start=1):
        if event.stream_id != stream_id or event.revision != expected:
            raise EventLogCorruption("event stream revisions are not contiguous")
        if event.event_id in event_ids:
            raise EventLogCorruption("event stream repeats an event ID")
        event_ids.add(event.event_id)


class SQLiteOracleEventStore:
    """Transactional SQLite event store with per-stream optimistic concurrency."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS oracle_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS oracle_events (
                    stream_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    event_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (stream_id, revision),
                    UNIQUE (stream_id, event_id)
                );
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO oracle_metadata(key, value) VALUES('format', ?)",
                (EVENT_STORE_FORMAT,),
            )
            connection.execute(
                "INSERT OR IGNORE INTO oracle_metadata(key, value) VALUES('version', ?)",
                (str(EVENT_STORE_VERSION),),
            )
            metadata = dict(connection.execute("SELECT key, value FROM oracle_metadata"))
            if metadata.get("format") != EVENT_STORE_FORMAT or metadata.get("version") != str(
                EVENT_STORE_VERSION
            ):
                raise EventLogCorruption("unsupported SQLite Oracle event-store format")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30.0)

    @staticmethod
    def _rows_to_events(rows: Iterable[tuple[Any, ...]]) -> tuple[OracleEvent, ...]:
        events: list[OracleEvent] = []
        for stream_id, revision, event_id, kind, schema_version, payload_json in rows:
            try:
                event_kind = EventKind(kind)
            except ValueError as error:
                raise EventLogCorruption(f"unknown Oracle event kind {kind!r}") from error
            events.append(
                OracleEvent(
                    stream_id,
                    int(revision),
                    event_id,
                    event_kind,
                    decode_event_payload(payload_json),
                    int(schema_version),
                )
            )
            if events[-1].schema_version != 1:
                raise EventLogCorruption("unsupported Oracle event schema version")
        return tuple(events)

    def read(self, stream_id: str) -> tuple[OracleEvent, ...]:
        _identifier(stream_id, "stream_id")
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT stream_id, revision, event_id, kind, schema_version, payload_json "
                "FROM oracle_events WHERE stream_id = ? ORDER BY revision",
                (stream_id,),
            )
            events = self._rows_to_events(rows)
        _validate_stream(events, stream_id)
        return events

    def append(
        self,
        stream_id: str,
        *,
        expected_revision: int,
        events: Iterable[PendingEvent],
    ) -> tuple[OracleEvent, ...]:
        _identifier(stream_id, "stream_id")
        pending = tuple(events)
        # Encode before opening a transaction so unsupported payloads cannot
        # leave a partially mutated database.
        encoded = {event.event_id: encode_event_payload(event.payload) for event in pending}
        with self._lock, self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                rows = connection.execute(
                    "SELECT stream_id, revision, event_id, kind, schema_version, payload_json "
                    "FROM oracle_events WHERE stream_id = ? ORDER BY revision",
                    (stream_id,),
                )
                current = self._rows_to_events(rows)
                _validate_stream(current, stream_id)
                appended = _validate_append(stream_id, current, expected_revision, pending)
                if all(event.revision <= len(current) for event in appended):
                    connection.rollback()
                    return appended
                for event in appended:
                    connection.execute(
                        "INSERT INTO oracle_events "
                        "(stream_id, revision, event_id, kind, schema_version, payload_json) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            event.stream_id,
                            event.revision,
                            event.event_id,
                            event.kind.value,
                            event.schema_version,
                            encoded[event.event_id],
                        ),
                    )
                connection.commit()
                return appended
            except Exception:
                connection.rollback()
                raise


class JSONLOracleEventStore:
    """Inspectable JSONL store with one fsynced record per append transaction."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = RLock()

    @contextmanager
    def _file(self, *, exclusive: bool) -> Iterator[Any]:
        with self._lock, self.path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield handle
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _read_all(handle: Any) -> tuple[OracleEvent, ...]:
        handle.seek(0)
        events: list[OracleEvent] = []
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                transaction = json.loads(line)
            except json.JSONDecodeError as error:
                raise EventLogCorruption(f"invalid JSONL transaction on line {line_number}") from error
            if (
                not isinstance(transaction, dict)
                or set(transaction) != {"format", "version", "events", "checksum"}
                or transaction.get("format") != EVENT_STORE_FORMAT
                or transaction.get("version") != EVENT_STORE_VERSION
                or not isinstance(transaction.get("events"), list)
                or not isinstance(transaction.get("checksum"), str)
            ):
                raise EventLogCorruption(f"invalid JSONL transaction envelope on line {line_number}")
            encoded_events = json.dumps(
                transaction["events"], sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            if hashlib.sha256(encoded_events).hexdigest() != transaction["checksum"]:
                raise EventLogCorruption(f"JSONL transaction checksum mismatch on line {line_number}")
            for item in transaction["events"]:
                if not isinstance(item, dict) or set(item) != {
                    "stream_id",
                    "revision",
                    "event_id",
                    "kind",
                    "schema_version",
                    "payload",
                }:
                    raise EventLogCorruption(f"invalid event record on line {line_number}")
                try:
                    kind = EventKind(item["kind"])
                    payload = _decode(item["payload"])
                    events.append(
                        OracleEvent(
                            item["stream_id"],
                            item["revision"],
                            item["event_id"],
                            kind,
                            payload,
                            item["schema_version"],
                        )
                    )
                    if events[-1].schema_version != 1:
                        raise EventLogCorruption("unsupported Oracle event schema version")
                except (KeyError, TypeError, ValueError) as error:
                    raise EventLogCorruption(f"invalid event record on line {line_number}") from error
        by_stream: dict[str, list[OracleEvent]] = {}
        for event in events:
            by_stream.setdefault(event.stream_id, []).append(event)
        for stream_id, stream_events in by_stream.items():
            _validate_stream(tuple(stream_events), stream_id)
        return tuple(events)

    def read(self, stream_id: str) -> tuple[OracleEvent, ...]:
        _identifier(stream_id, "stream_id")
        with self._file(exclusive=False) as handle:
            events = self._read_all(handle)
        return tuple(event for event in events if event.stream_id == stream_id)

    def append(
        self,
        stream_id: str,
        *,
        expected_revision: int,
        events: Iterable[PendingEvent],
    ) -> tuple[OracleEvent, ...]:
        _identifier(stream_id, "stream_id")
        pending = tuple(events)
        # Validate encodability before locking or writing.
        for event in pending:
            _encode(event.payload)
        with self._file(exclusive=True) as handle:
            all_events = self._read_all(handle)
            current = tuple(event for event in all_events if event.stream_id == stream_id)
            appended = _validate_append(stream_id, current, expected_revision, pending)
            if all(event.revision <= len(current) for event in appended):
                return appended
            encoded_events = [
                {
                    "stream_id": event.stream_id,
                    "revision": event.revision,
                    "event_id": event.event_id,
                    "kind": event.kind.value,
                    "schema_version": event.schema_version,
                    "payload": _encode(event.payload),
                }
                for event in appended
            ]
            canonical_events = json.dumps(
                encoded_events, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            transaction = {
                "format": EVENT_STORE_FORMAT,
                "version": EVENT_STORE_VERSION,
                "events": encoded_events,
                "checksum": hashlib.sha256(canonical_events).hexdigest(),
            }
            line = json.dumps(
                transaction, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            handle.seek(0, os.SEEK_END)
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return appended


__all__ = [
    "EVENT_STORE_FORMAT",
    "EVENT_STORE_VERSION",
    "EventLogCorruption",
    "JSONLOracleEventStore",
    "SQLiteOracleEventStore",
    "decode_event_payload",
    "encode_event_payload",
]
