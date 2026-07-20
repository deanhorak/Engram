import json
import sqlite3
from dataclasses import replace

import pytest

from engram.oracle import (
    ActionProposal,
    EventConflict,
    EventKind,
    EventLogCorruption,
    FunctionWorkerAdapter,
    Goal,
    GoalGraph,
    JSONLOracleEventStore,
    OracleDispatcher,
    OracleSession,
    OutcomeStatus,
    PendingEvent,
    ResourceBudget,
    RevisionConflict,
    SQLiteOracleEventStore,
    WorkerAdapterRegistry,
    WorkerCapability,
    WorkerRegistry,
    WorkerResult,
)


def _store(kind, tmp_path):
    path = tmp_path / ("oracle.sqlite" if kind == "sqlite" else "oracle.jsonl")
    factory = SQLiteOracleEventStore if kind == "sqlite" else JSONLOracleEventStore
    return factory(path), factory, path


def _worker_registry():
    capability = WorkerCapability(
        "research-worker",
        1,
        ("research",),
        adapter="test-adapter",
        local=True,
        trust=0.9,
    )
    return WorkerRegistry().register(capability), capability


def _proposal():
    return ActionProposal(
        "research-action",
        "research-goal",
        "research",
        0.8,
        0.6,
        2.0,
        token_cost=100,
        compute_cost=1.0,
    )


def _plan(session, stream_id, registry):
    state = session.start(
        stream_id,
        GoalGraph((Goal("research-goal", "Research the question"),)),
        ResourceBudget(20.0, 1000, 10.0, 3),
    )
    return session.plan(
        stream_id,
        expected_revision=state.revision,
        attempt_id="attempt-1",
        proposals=(_proposal(),),
        registry=registry,
    )


@pytest.mark.parametrize("kind", ("sqlite", "jsonl"))
def test_durable_store_reopens_and_replays_typed_session(kind, tmp_path):
    store, factory, path = _store(kind, tmp_path)
    registry, capability = _worker_registry()
    session = OracleSession(store)
    planned = _plan(session, "durable-task", registry)
    adapters = WorkerAdapterRegistry()
    requests = []

    def execute(request):
        requests.append(request)
        return WorkerResult(
            OutcomeStatus.SUCCEEDED,
            goal_progress=1.0,
            confidence=0.9,
            goal_completed=True,
            information_gain=0.7,
            information_measurement="evidence-validator-v1",
            latency_seconds=2.5,
            token_cost=90,
            compute_cost=1.2,
        )

    adapters.register(FunctionWorkerAdapter("test-adapter", execute))
    outcome = OracleDispatcher(adapters).dispatch(planned, capability, outcome_id="outcome-1")
    observed = session.observe(
        "durable-task", expected_revision=planned.revision, outcome=outcome
    )

    reopened = OracleSession(factory(path)).state("durable-task")
    assert reopened == observed
    assert requests[0].attempt_id == "attempt-1"
    assert reopened.calibration[0].actual_information_gain == 0.7
    assert reopened.ledger.charged_token_cost == 90


@pytest.mark.parametrize("kind", ("sqlite", "jsonl"))
def test_durable_store_contract_is_idempotent_and_stream_isolated(kind, tmp_path):
    store, factory, path = _store(kind, tmp_path)
    payload = GoalGraph((Goal("g", "Goal"),))
    first = PendingEvent("same-id", EventKind.SESSION_STARTED, payload)
    a = store.append("stream-a", expected_revision=0, events=(first,))
    b = store.append("stream-b", expected_revision=0, events=(first,))

    assert store.append("stream-a", expected_revision=0, events=(first,)) == a
    assert a[0].revision == b[0].revision == 1
    assert factory(path).read("stream-a") == a
    assert factory(path).read("stream-b") == b
    with pytest.raises(EventConflict):
        store.append(
            "stream-a",
            expected_revision=1,
            events=(PendingEvent("same-id", EventKind.SESSION_STARTED, {"changed": True}),),
        )
    with pytest.raises(RevisionConflict):
        store.append(
            "stream-a",
            expected_revision=0,
            events=(PendingEvent("new", EventKind.SESSION_STARTED, payload),),
        )


@pytest.mark.parametrize("kind", ("sqlite", "jsonl"))
def test_failed_batch_encoding_leaves_durable_store_unchanged(kind, tmp_path):
    store, _, _ = _store(kind, tmp_path)
    with pytest.raises(TypeError, match="unsupported event value"):
        store.append(
            "stream",
            expected_revision=0,
            events=(
                PendingEvent("valid", EventKind.SESSION_STARTED, "ok"),
                PendingEvent("invalid", EventKind.SESSION_STARTED, object()),
            ),
        )
    assert store.read("stream") == ()


def test_sqlite_detects_revision_corruption(tmp_path):
    store, _, path = _store("sqlite", tmp_path)
    payload = GoalGraph((Goal("g", "Goal"),))
    store.append(
        "stream",
        expected_revision=0,
        events=(
            PendingEvent("one", EventKind.SESSION_STARTED, payload),
            PendingEvent("two", EventKind.SESSION_STARTED, payload),
        ),
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "DELETE FROM oracle_events WHERE stream_id = ? AND revision = ?", ("stream", 1)
        )
    with pytest.raises(EventLogCorruption, match="contiguous"):
        store.read("stream")


def test_jsonl_detects_checksum_corruption_and_torn_tail(tmp_path):
    store, factory, path = _store("jsonl", tmp_path)
    payload = GoalGraph((Goal("g", "Goal"),))
    store.append(
        "stream",
        expected_revision=0,
        events=(PendingEvent("one", EventKind.SESSION_STARTED, payload),),
    )
    transaction = json.loads(path.read_text())
    transaction["events"][0]["event_id"] = "tampered"
    path.write_text(json.dumps(transaction) + "\n")
    with pytest.raises(EventLogCorruption, match="checksum"):
        factory(path).read("stream")

    path.write_text('{"format":"engram.oracle.events"')
    with pytest.raises(EventLogCorruption, match="invalid JSONL"):
        factory(path).read("stream")


def test_dispatcher_stamps_identity_and_converts_adapter_failure(tmp_path):
    store, _, _ = _store("sqlite", tmp_path)
    registry, capability = _worker_registry()
    planned = _plan(OracleSession(store), "dispatch-task", registry)
    adapters = WorkerAdapterRegistry()

    def fail(_request):
        raise RuntimeError("secret internal detail")

    adapters.register(FunctionWorkerAdapter("test-adapter", fail))
    outcome = OracleDispatcher(adapters).dispatch(planned, capability, outcome_id="failed-1")

    assert outcome.status is OutcomeStatus.FAILED
    assert outcome.error_code == "adapter_exception"
    assert outcome.attempt_id == planned.pending.attempt_id
    assert outcome.worker_generation == planned.pending.worker_generation
    assert outcome.latency_seconds >= 0.0


def test_dispatcher_rejects_wrong_generation_before_invocation(tmp_path):
    store, _, _ = _store("jsonl", tmp_path)
    registry, capability = _worker_registry()
    planned = _plan(OracleSession(store), "wrong-worker", registry)
    calls = []
    adapters = WorkerAdapterRegistry()
    adapters.register(
        FunctionWorkerAdapter(
            "test-adapter", lambda request: calls.append(request) or WorkerResult(
                OutcomeStatus.SUCCEEDED, 1.0, 1.0, goal_completed=True
            )
        )
    )

    with pytest.raises(ValueError, match="does not match"):
        OracleDispatcher(adapters).dispatch(
            planned, replace(capability, generation=2), outcome_id="wrong"
        )
    assert calls == []
