import math
from dataclasses import replace

import pytest

from engram.oracle import (
    ActionOutcome,
    ActionProposal,
    EventConflict,
    EventKind,
    Goal,
    GoalGraph,
    GoalStatus,
    InMemoryOracleEventStore,
    OracleSession,
    OutcomeStatus,
    PendingEvent,
    PredictionOutcome,
    ResourceBudget,
    RevisionConflict,
    WorkerCapability,
    WorkerRegistry,
    replay_oracle_events,
    summarize_calibration,
)


def _registry():
    return WorkerRegistry().register(
        WorkerCapability(
            "local-research",
            1,
            ("local_lookup", "multi_source"),
            model_ids=("engram-small",),
            local=True,
            trust=0.8,
        )
    )


def _budget():
    return ResourceBudget(latency_seconds=100.0, token_cost=10_000, compute_cost=20.0, max_actions=4)


def _proposal():
    return ActionProposal(
        "careful",
        "research",
        "multi_source",
        0.8,
        0.6,
        10.0,
        token_cost=1000,
        compute_cost=2.0,
        model_id="engram-small",
        predictor_id="baseline-v1",
    )


def _planned_session():
    session = OracleSession()
    state = session.start("task-1", GoalGraph((Goal("research", "Research"),)), _budget())
    state = session.plan(
        "task-1",
        expected_revision=state.revision,
        attempt_id="attempt-1",
        proposals=(_proposal(),),
        registry=_registry(),
    )
    return session, state


def _outcome(state, **changes):
    values = {
        "outcome_id": "outcome-1",
        "attempt_id": state.pending.attempt_id,
        "decision_id": state.pending.decision_id,
        "action_id": state.pending.proposal.action_id,
        "worker_id": state.pending.worker_id,
        "worker_generation": state.pending.worker_generation,
        "state_revision": state.revision,
        "status": OutcomeStatus.SUCCEEDED,
        "goal_progress": 1.0,
        "confidence": 0.9,
        "goal_completed": True,
        "information_gain": 0.7,
        "information_measurement": "validated-evidence-v1",
        "latency_seconds": 12.0,
        "token_cost": 900,
        "compute_cost": 2.5,
    }
    values.update(changes)
    return ActionOutcome(**values)


def test_registry_pins_worker_generation_and_disable_creates_a_new_generation():
    registry = _registry()
    worker = registry.resolve(_proposal())
    disabled = registry.disable(worker.worker_id)

    assert worker.generation == 1
    assert disabled.revision == 2
    assert disabled.latest(worker.worker_id).generation == 2
    assert disabled.latest(worker.worker_id).enabled is False
    with pytest.raises(ValueError, match="supports"):
        disabled.resolve(_proposal())
    with pytest.raises(ValueError, match="no longer current"):
        disabled.resolve(
            replace(_proposal(), worker_id="local-research", worker_generation=1)
        )


def test_event_store_compare_and_swap_and_idempotency_are_atomic():
    store = InMemoryOracleEventStore()
    first = PendingEvent("e1", EventKind.SESSION_STARTED, "payload")
    recorded = store.append("stream", expected_revision=0, events=(first,))

    assert recorded[0].revision == 1
    assert store.append("stream", expected_revision=0, events=(first,)) == recorded
    with pytest.raises(EventConflict):
        store.append(
            "stream",
            expected_revision=1,
            events=(PendingEvent("e1", EventKind.SESSION_STARTED, "changed"),),
        )
    with pytest.raises(RevisionConflict):
        store.append(
            "stream",
            expected_revision=0,
            events=(PendingEvent("e2", EventKind.SESSION_STARTED, "new"),),
        )
    assert store.read("stream") == recorded


def test_session_reserves_predicted_cost_and_observes_actual_outcome_once():
    session, planned = _planned_session()

    assert planned.revision == 2
    assert planned.goals.get("research").status is GoalStatus.ACTIVE
    assert planned.ledger.reserved_token_cost == 1000
    assert planned.pending.worker_id == "local-research"
    assert planned.pending.worker_generation == 1
    assert planned.pending.registry_revision == 1

    outcome = _outcome(planned)
    observed = session.observe("task-1", expected_revision=planned.revision, outcome=outcome)
    duplicate = session.observe("task-1", expected_revision=planned.revision, outcome=outcome)

    assert duplicate == observed
    assert observed.revision == 3
    assert observed.pending is None
    assert observed.goals.get("research").status is GoalStatus.COMPLETE
    assert observed.ledger.reserved_token_cost == 0
    assert observed.ledger.charged_token_cost == 900
    assert observed.ledger.completed_actions == 1
    assert len(observed.calibration) == 1
    assert replay_oracle_events(session.store.read("task-1")) == observed


def test_plan_retry_is_idempotent_only_when_inputs_are_identical():
    session, planned = _planned_session()
    repeated = session.plan(
        "task-1",
        expected_revision=planned.pending.base_revision,
        attempt_id="attempt-1",
        proposals=(_proposal(),),
        registry=_registry(),
    )
    assert repeated == planned

    with pytest.raises(ValueError, match="different planning inputs"):
        session.plan(
            "task-1",
            expected_revision=planned.revision,
            attempt_id="attempt-1",
            proposals=(replace(_proposal(), predicted_success=0.1),),
            registry=_registry(),
        )


def test_conflicting_or_mismatched_outcomes_do_not_modify_the_log():
    session, planned = _planned_session()
    before = session.store.read("task-1")
    wrong = _outcome(planned, worker_generation=2)

    with pytest.raises(ValueError, match="does not match"):
        session.observe("task-1", expected_revision=planned.revision, outcome=wrong)
    assert session.store.read("task-1") == before

    accepted = _outcome(planned)
    session.observe("task-1", expected_revision=planned.revision, outcome=accepted)
    with pytest.raises(ValueError, match="reused"):
        session.observe(
            "task-1",
            expected_revision=planned.revision,
            outcome=_outcome(planned, latency_seconds=99.0),
        )


def test_budget_admission_rejects_prediction_before_event_append():
    session = OracleSession()
    state = session.start(
        "tight", GoalGraph((Goal("research", "Research"),)), ResourceBudget(5.0, 500, 1.0, 1)
    )
    before = session.store.read("tight")

    with pytest.raises(ValueError, match="no proposal can be dispatched"):
        session.plan(
            "tight",
            expected_revision=state.revision,
            attempt_id="too-large",
            proposals=(_proposal(),),
            registry=_registry(),
        )
    assert session.store.read("tight") == before


def test_outcome_after_registry_disable_is_accepted_against_pinned_generation():
    session, planned = _planned_session()
    disabled = _registry().disable("local-research")
    assert disabled.latest("local-research").enabled is False

    observed = session.observe(
        "task-1", expected_revision=planned.revision, outcome=_outcome(planned)
    )
    assert observed.calibration[0].worker_generation == 1


def test_missing_telemetry_is_not_calibrated_as_zero_and_reservation_is_charged():
    session, planned = _planned_session()
    outcome = _outcome(
        planned,
        information_gain=None,
        information_measurement=None,
        latency_seconds=None,
        token_cost=None,
        compute_cost=None,
    )
    observed = session.observe("task-1", expected_revision=planned.revision, outcome=outcome)
    summary = summarize_calibration(observed.calibration)

    assert observed.ledger.charged_token_cost == planned.pending.proposal.token_cost
    assert observed.ledger.unmeasured_outcomes == 1
    assert summary.success.count == 1
    assert summary.information_gain.count == 0
    assert summary.latency.count == 0
    assert summary.tokens.count == 0
    assert summary.compute.count == 0


def test_calibration_metrics_keep_units_separate_and_handle_final_probability_bin():
    common = {
        "attempt_id": "a",
        "strategy": "test",
        "worker_id": "worker",
        "worker_generation": 1,
        "predictor_id": "p1",
        "predicted_information_gain": 0.2,
        "actual_information_gain": 0.4,
        "predicted_latency_seconds": 1.0,
        "actual_latency_seconds": 2.0,
        "predicted_token_cost": 10,
        "actual_token_cost": 12,
        "predicted_compute_cost": 1.0,
        "actual_compute_cost": 2.0,
    }
    records = (
        PredictionOutcome(predicted_success=0.0, actual_success=0.0, **common),
        PredictionOutcome(predicted_success=0.5, actual_success=1.0, **{**common, "attempt_id": "b"}),
        PredictionOutcome(predicted_success=1.0, actual_success=1.0, **{**common, "attempt_id": "c"}),
    )
    summary = summarize_calibration(records, bins=10)

    assert summary.success.mean_squared_error == pytest.approx(1.0 / 12.0)
    assert summary.success.bias == pytest.approx(1.0 / 6.0)
    assert summary.success.expected_calibration_error == pytest.approx(1.0 / 6.0)
    assert summary.latency.mean_absolute_error == 1.0
    assert summary.latency.bias == 1.0
    assert summary.latency.mean_absolute_log_error == pytest.approx(
        abs(math.log1p(2.0) - math.log1p(1.0))
    )
    assert summary.tokens.mean_absolute_error == 2.0
    assert summary.compute.mean_absolute_error == 1.0


def test_cancelled_outcome_is_excluded_from_success_calibration():
    session, planned = _planned_session()
    cancelled = _outcome(
        planned,
        status=OutcomeStatus.CANCELLED,
        goal_completed=False,
        goal_progress=0.2,
    )
    observed = session.observe("task-1", expected_revision=planned.revision, outcome=cancelled)

    assert summarize_calibration(observed.calibration).success.count == 0
