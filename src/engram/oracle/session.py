"""Revisioned Oracle session state, replay, budgeting, and outcome observation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable

from .calibration import PredictionOutcome
from .events import EventKind, InMemoryOracleEventStore, OracleEvent, OracleEventStore, PendingEvent
from .executive import CognitiveExecutive
from .models import (
    ActionProposal,
    DecisionPolicy,
    Evidence,
    GoalGraph,
    GoalStatus,
    OracleDecision,
    ProgressObservation,
    _identifier,
    _nonnegative,
    _probability,
    ensure_unique_ids,
)
from .registry import WorkerRegistry


@dataclass(frozen=True)
class ResourceBudget:
    latency_seconds: float
    token_cost: int
    compute_cost: float
    max_actions: int

    def __post_init__(self) -> None:
        _nonnegative(self.latency_seconds, "latency_seconds")
        _nonnegative(self.compute_cost, "compute_cost")
        for value, name in ((self.token_cost, "token_cost"), (self.max_actions, "max_actions")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class ResourceLedger:
    charged_latency_seconds: float = 0.0
    charged_token_cost: int = 0
    charged_compute_cost: float = 0.0
    reserved_latency_seconds: float = 0.0
    reserved_token_cost: int = 0
    reserved_compute_cost: float = 0.0
    completed_actions: int = 0
    unmeasured_outcomes: int = 0

    def available(self, budget: ResourceBudget) -> tuple[float, int, float, int]:
        return (
            max(0.0, budget.latency_seconds - self.charged_latency_seconds - self.reserved_latency_seconds),
            max(0, budget.token_cost - self.charged_token_cost - self.reserved_token_cost),
            max(0.0, budget.compute_cost - self.charged_compute_cost - self.reserved_compute_cost),
            max(0, budget.max_actions - self.completed_actions),
        )

    def exhausted(self, budget: ResourceBudget) -> bool:
        latency, tokens, compute, actions = self.available(budget)
        return latency <= 0.0 or tokens <= 0 or compute <= 0.0 or actions <= 0


class OutcomeStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ActionOutcome:
    outcome_id: str
    attempt_id: str
    decision_id: str
    action_id: str
    worker_id: str
    worker_generation: int
    state_revision: int
    status: OutcomeStatus
    goal_progress: float
    confidence: float
    goal_completed: bool = False
    information_gain: float | None = None
    latency_seconds: float | None = None
    token_cost: int | None = None
    compute_cost: float | None = None
    information_measurement: str | None = None
    error_code: str | None = None
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        for value, name in (
            (self.outcome_id, "outcome_id"),
            (self.attempt_id, "attempt_id"),
            (self.decision_id, "decision_id"),
            (self.action_id, "action_id"),
            (self.worker_id, "worker_id"),
        ):
            _identifier(value, name)
        if (
            isinstance(self.worker_generation, bool)
            or not isinstance(self.worker_generation, int)
            or self.worker_generation <= 0
        ):
            raise ValueError("worker_generation must be a positive integer")
        if isinstance(self.state_revision, bool) or not isinstance(self.state_revision, int) or self.state_revision <= 0:
            raise ValueError("state_revision must be a positive integer")
        if not isinstance(self.status, OutcomeStatus):
            raise ValueError("status must be an OutcomeStatus")
        _probability(self.goal_progress, "goal_progress")
        _probability(self.confidence, "confidence")
        if not isinstance(self.goal_completed, bool):
            raise ValueError("goal_completed must be a boolean")
        if self.goal_completed and self.status is not OutcomeStatus.SUCCEEDED:
            raise ValueError("only a succeeded outcome can complete a goal")
        if self.information_gain is not None:
            _probability(self.information_gain, "information_gain")
            if self.information_measurement is None:
                raise ValueError("measured information gain requires measurement provenance")
        if self.information_measurement is not None:
            _identifier(self.information_measurement, "information_measurement")
        if self.error_code is not None:
            _identifier(self.error_code, "error_code")
        if self.latency_seconds is not None:
            _nonnegative(self.latency_seconds, "latency_seconds")
        if self.compute_cost is not None:
            _nonnegative(self.compute_cost, "compute_cost")
        if self.token_cost is not None and (
            isinstance(self.token_cost, bool) or not isinstance(self.token_cost, int) or self.token_cost < 0
        ):
            raise ValueError("token_cost must be a non-negative integer")
        ensure_unique_ids((item.evidence_id for item in self.evidence), "outcome evidence")

    @property
    def calibration_success(self) -> float | None:
        if self.status is OutcomeStatus.CANCELLED:
            return None
        return 1.0 if self.status is OutcomeStatus.SUCCEEDED else 0.0


@dataclass(frozen=True)
class SessionStarted:
    goals: GoalGraph
    budget: ResourceBudget


@dataclass(frozen=True)
class PendingAttempt:
    attempt_id: str
    decision_id: str
    base_revision: int
    registry_revision: int
    worker_id: str
    worker_generation: int
    decision: OracleDecision
    submitted_proposals: tuple[ActionProposal, ...]
    policy: DecisionPolicy

    @property
    def proposal(self) -> ActionProposal:
        return self.decision.selected.proposal


@dataclass(frozen=True)
class ActionSelected:
    pending: PendingAttempt


@dataclass(frozen=True)
class OutcomeObserved:
    outcome: ActionOutcome


@dataclass(frozen=True)
class OracleState:
    stream_id: str
    revision: int
    goals: GoalGraph
    budget: ResourceBudget
    ledger: ResourceLedger
    evidence: tuple[Evidence, ...]
    progress: tuple[ProgressObservation, ...]
    calibration: tuple[PredictionOutcome, ...]
    pending: PendingAttempt | None
    events: tuple[OracleEvent, ...]


def _reserve(ledger: ResourceLedger, proposal: ActionProposal) -> ResourceLedger:
    return replace(
        ledger,
        reserved_latency_seconds=ledger.reserved_latency_seconds + proposal.latency_seconds,
        reserved_token_cost=ledger.reserved_token_cost + proposal.token_cost,
        reserved_compute_cost=ledger.reserved_compute_cost + proposal.compute_cost,
    )


def _settle(
    ledger: ResourceLedger, proposal: ActionProposal, outcome: ActionOutcome
) -> ResourceLedger:
    missing = any(
        value is None
        for value in (outcome.latency_seconds, outcome.token_cost, outcome.compute_cost)
    )
    # Missing telemetry is conservatively charged at the reservation, but stays
    # absent from calibration rather than being mislabeled as zero.
    latency = proposal.latency_seconds if outcome.latency_seconds is None else outcome.latency_seconds
    tokens = proposal.token_cost if outcome.token_cost is None else outcome.token_cost
    compute = proposal.compute_cost if outcome.compute_cost is None else outcome.compute_cost
    return ResourceLedger(
        charged_latency_seconds=ledger.charged_latency_seconds + latency,
        charged_token_cost=ledger.charged_token_cost + tokens,
        charged_compute_cost=ledger.charged_compute_cost + compute,
        reserved_latency_seconds=ledger.reserved_latency_seconds - proposal.latency_seconds,
        reserved_token_cost=ledger.reserved_token_cost - proposal.token_cost,
        reserved_compute_cost=ledger.reserved_compute_cost - proposal.compute_cost,
        completed_actions=ledger.completed_actions + 1,
        unmeasured_outcomes=ledger.unmeasured_outcomes + int(missing),
    )


def replay_oracle_events(events: Iterable[OracleEvent]) -> OracleState:
    history = tuple(events)
    if not history:
        raise ValueError("an Oracle session requires a start event")
    stream_id = history[0].stream_id
    for expected, event in enumerate(history, start=1):
        if event.stream_id != stream_id or event.revision != expected:
            raise ValueError("event stream revisions must be contiguous and share one stream ID")
    first = history[0]
    if first.kind is not EventKind.SESSION_STARTED or not isinstance(first.payload, SessionStarted):
        raise ValueError("the first Oracle event must start the session")
    state = OracleState(
        stream_id=stream_id,
        revision=1,
        goals=first.payload.goals,
        budget=first.payload.budget,
        ledger=ResourceLedger(),
        evidence=(),
        progress=(),
        calibration=(),
        pending=None,
        events=(first,),
    )
    outcome_ids: dict[str, ActionOutcome] = {}
    attempt_ids: set[str] = set()
    for event in history[1:]:
        if event.kind is EventKind.ACTION_SELECTED and isinstance(event.payload, ActionSelected):
            pending = event.payload.pending
            if state.pending is not None:
                raise ValueError("cannot select another action while an attempt is pending")
            if pending.base_revision != state.revision:
                raise ValueError("selected action has a stale base revision")
            if pending.attempt_id in attempt_ids:
                raise ValueError("attempt IDs must be unique")
            attempt_ids.add(pending.attempt_id)
            proposal = pending.proposal
            available_latency, available_tokens, available_compute, available_actions = state.ledger.available(
                state.budget
            )
            if (
                proposal.latency_seconds > available_latency
                or proposal.token_cost > available_tokens
                or proposal.compute_cost > available_compute
                or available_actions <= 0
            ):
                raise ValueError("selected action exceeds the remaining budget")
            state = replace(
                state,
                revision=event.revision,
                goals=state.goals.with_status(pending.decision.goal_id, GoalStatus.ACTIVE),
                ledger=_reserve(state.ledger, proposal),
                pending=pending,
                events=state.events + (event,),
            )
        elif event.kind is EventKind.OUTCOME_OBSERVED and isinstance(event.payload, OutcomeObserved):
            outcome = event.payload.outcome
            if outcome.outcome_id in outcome_ids:
                if outcome_ids[outcome.outcome_id] == outcome:
                    raise ValueError("duplicate outcome events must be removed by the event store")
                raise ValueError("outcome ID was reused with different content")
            outcome_ids[outcome.outcome_id] = outcome
            pending = state.pending
            if pending is None:
                raise ValueError("cannot observe an outcome without a pending attempt")
            if outcome.state_revision != state.revision:
                raise ValueError("outcome targets a stale state revision")
            expected_identity = (
                pending.attempt_id,
                pending.decision_id,
                pending.proposal.action_id,
                pending.worker_id,
                pending.worker_generation,
            )
            actual_identity = (
                outcome.attempt_id,
                outcome.decision_id,
                outcome.action_id,
                outcome.worker_id,
                outcome.worker_generation,
            )
            if actual_identity != expected_identity:
                raise ValueError("outcome does not match the selected attempt")
            existing_evidence = {item.evidence_id for item in state.evidence}
            if existing_evidence & {item.evidence_id for item in outcome.evidence}:
                raise ValueError("outcome evidence IDs must not already exist in the session")
            failures = (state.progress[-1].failures if state.progress else 0) + int(
                outcome.status in {OutcomeStatus.FAILED, OutcomeStatus.TIMED_OUT}
            )
            observation = ProgressObservation(
                step=state.ledger.completed_actions + 1,
                progress=outcome.goal_progress,
                confidence=outcome.confidence,
                failures=failures,
            )
            proposal = pending.proposal
            calibration = PredictionOutcome(
                attempt_id=pending.attempt_id,
                strategy=proposal.strategy,
                worker_id=pending.worker_id,
                worker_generation=pending.worker_generation,
                predictor_id=proposal.predictor_id,
                predicted_success=proposal.predicted_success,
                actual_success=outcome.calibration_success,
                predicted_information_gain=proposal.information_gain,
                actual_information_gain=outcome.information_gain,
                predicted_latency_seconds=proposal.latency_seconds,
                actual_latency_seconds=outcome.latency_seconds,
                predicted_token_cost=proposal.token_cost,
                actual_token_cost=outcome.token_cost,
                predicted_compute_cost=proposal.compute_cost,
                actual_compute_cost=outcome.compute_cost,
            )
            goals = (
                state.goals.with_status(pending.decision.goal_id, GoalStatus.COMPLETE)
                if outcome.goal_completed
                else state.goals
            )
            state = replace(
                state,
                revision=event.revision,
                goals=goals,
                ledger=_settle(state.ledger, proposal, outcome),
                evidence=state.evidence + outcome.evidence,
                progress=state.progress + (observation,),
                calibration=state.calibration + (calibration,),
                pending=None,
                events=state.events + (event,),
            )
        else:
            raise ValueError(f"unsupported or malformed Oracle event: {event.kind}")
    return state


class OracleSession:
    """Reference event-sourced lifecycle around the pure CognitiveExecutive policy."""

    def __init__(
        self,
        store: OracleEventStore | None = None,
        executive: CognitiveExecutive | None = None,
    ) -> None:
        self.store = InMemoryOracleEventStore() if store is None else store
        self.executive = CognitiveExecutive() if executive is None else executive

    def start(self, stream_id: str, goals: GoalGraph, budget: ResourceBudget) -> OracleState:
        self.store.append(
            stream_id,
            expected_revision=0,
            events=(PendingEvent(f"{stream_id}:start", EventKind.SESSION_STARTED, SessionStarted(goals, budget)),),
        )
        return self.state(stream_id)

    def state(self, stream_id: str) -> OracleState:
        return replay_oracle_events(self.store.read(stream_id))

    def plan(
        self,
        stream_id: str,
        *,
        expected_revision: int,
        attempt_id: str,
        proposals: Iterable[ActionProposal],
        registry: WorkerRegistry,
        policy: DecisionPolicy = DecisionPolicy(),
    ) -> OracleState:
        _identifier(attempt_id, "attempt_id")
        state = self.state(stream_id)
        submitted = tuple(proposals)
        ensure_unique_ids((proposal.action_id for proposal in submitted), "action proposals")
        if state.pending is not None:
            if state.pending.attempt_id == attempt_id:
                if (
                    state.pending.submitted_proposals == submitted
                    and state.pending.policy == policy
                    and state.pending.registry_revision == registry.revision
                    and expected_revision
                    in {state.pending.base_revision, state.revision}
                ):
                    return state
                raise ValueError("attempt_id was retried with different planning inputs")
            raise ValueError("an action attempt is already pending")
        if state.revision != expected_revision:
            raise ValueError(f"expected state revision {expected_revision}, found {state.revision}")
        if any(
            isinstance(event.payload, ActionSelected)
            and event.payload.pending.attempt_id == attempt_id
            for event in state.events
        ):
            raise ValueError("attempt_id has already been used")
        available_latency, available_tokens, available_compute, available_actions = state.ledger.available(
            state.budget
        )
        if available_actions <= 0:
            raise ValueError("the session action budget is exhausted")

        eligible: list[ActionProposal] = []
        rejected: list[tuple[str, str]] = []
        for proposal in submitted:
            try:
                worker = registry.resolve(proposal)
            except (KeyError, ValueError) as error:
                rejected.append((proposal.action_id, str(error)))
                continue
            pinned = replace(
                proposal, worker_id=worker.worker_id, worker_generation=worker.generation
            )
            if pinned.latency_seconds > available_latency:
                rejected.append((pinned.action_id, "predicted latency exceeds remaining budget"))
            elif pinned.token_cost > available_tokens:
                rejected.append((pinned.action_id, "predicted token cost exceeds remaining budget"))
            elif pinned.compute_cost > available_compute:
                rejected.append((pinned.action_id, "predicted compute cost exceeds remaining budget"))
            else:
                eligible.append(pinned)
        if not eligible:
            detail = "; ".join(f"{action}: {reason}" for action, reason in rejected)
            raise ValueError(f"no proposal can be dispatched: {detail}")
        decision = self.executive.decide(state.goals, eligible, policy=policy)
        decision = replace(decision, rejected=decision.rejected + tuple(rejected))
        proposal = decision.selected.proposal
        if proposal.worker_id is None or proposal.worker_generation is None:
            raise AssertionError("registry resolution must pin a worker generation")
        decision_id = f"{stream_id}:decision:{attempt_id}"
        pending = PendingAttempt(
            attempt_id=attempt_id,
            decision_id=decision_id,
            base_revision=state.revision,
            registry_revision=registry.revision,
            worker_id=proposal.worker_id,
            worker_generation=proposal.worker_generation,
            decision=decision,
            submitted_proposals=submitted,
            policy=policy,
        )
        pending_event = PendingEvent(
            f"{stream_id}:selected:{attempt_id}",
            EventKind.ACTION_SELECTED,
            ActionSelected(pending),
        )
        replay_oracle_events(
            state.events
            + (
                OracleEvent(
                    stream_id,
                    expected_revision + 1,
                    pending_event.event_id,
                    pending_event.kind,
                    pending_event.payload,
                ),
            )
        )
        self.store.append(
            stream_id,
            expected_revision=expected_revision,
            events=(pending_event,),
        )
        return self.state(stream_id)

    def observe(
        self,
        stream_id: str,
        *,
        expected_revision: int,
        outcome: ActionOutcome,
    ) -> OracleState:
        event = PendingEvent(
            f"{stream_id}:outcome:{outcome.outcome_id}",
            EventKind.OUTCOME_OBSERVED,
            OutcomeObserved(outcome),
        )
        existing = self.store.read(stream_id)
        for recorded in existing:
            if recorded.event_id == event.event_id:
                if recorded.kind is event.kind and recorded.payload == event.payload:
                    return replay_oracle_events(existing)
                raise ValueError("outcome ID was reused with different content")
        state = replay_oracle_events(existing)
        if state.revision != expected_revision:
            raise ValueError(f"expected state revision {expected_revision}, found {state.revision}")
        if state.pending is None:
            raise ValueError("there is no pending attempt to observe")
        replay_oracle_events(
            existing
            + (
                OracleEvent(
                    stream_id,
                    expected_revision + 1,
                    event.event_id,
                    event.kind,
                    event.payload,
                ),
            )
        )
        self.store.append(
            stream_id,
            expected_revision=expected_revision,
            events=(event,),
        )
        return self.state(stream_id)


__all__ = [
    "ActionOutcome",
    "OracleSession",
    "OracleState",
    "OutcomeStatus",
    "PendingAttempt",
    "ResourceBudget",
    "ResourceLedger",
    "replay_oracle_events",
]
