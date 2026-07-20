"""Validated worker-adapter boundary for selected Oracle attempts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

from .models import ActionProposal, Evidence, _identifier, _nonnegative, _probability
from .registry import WorkerCapability
from .session import ActionOutcome, OracleState, OutcomeStatus, PendingAttempt


@dataclass(frozen=True)
class WorkerRequest:
    stream_id: str
    state_revision: int
    attempt_id: str
    decision_id: str
    goal_id: str
    worker_id: str
    worker_generation: int
    proposal: ActionProposal


@dataclass(frozen=True)
class WorkerResult:
    status: OutcomeStatus
    goal_progress: float
    confidence: float
    goal_completed: bool = False
    information_gain: float | None = None
    information_measurement: str | None = None
    latency_seconds: float | None = None
    token_cost: int | None = None
    compute_cost: float | None = None
    error_code: str | None = None
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, OutcomeStatus):
            raise ValueError("status must be an OutcomeStatus")
        _probability(self.goal_progress, "goal_progress")
        _probability(self.confidence, "confidence")
        if not isinstance(self.goal_completed, bool):
            raise ValueError("goal_completed must be a boolean")
        if self.goal_completed and self.status is not OutcomeStatus.SUCCEEDED:
            raise ValueError("only a succeeded worker result can complete a goal")
        if self.information_gain is not None:
            _probability(self.information_gain, "information_gain")
            if self.information_measurement is None:
                raise ValueError("information gain requires measurement provenance")
        if self.information_measurement is not None:
            _identifier(self.information_measurement, "information_measurement")
        if self.latency_seconds is not None:
            _nonnegative(self.latency_seconds, "latency_seconds")
        if self.compute_cost is not None:
            _nonnegative(self.compute_cost, "compute_cost")
        if self.token_cost is not None and (
            isinstance(self.token_cost, bool) or not isinstance(self.token_cost, int) or self.token_cost < 0
        ):
            raise ValueError("token_cost must be a non-negative integer")
        if self.error_code is not None:
            _identifier(self.error_code, "error_code")


class WorkerAdapter(Protocol):
    adapter_id: str

    def dispatch(self, request: WorkerRequest) -> WorkerResult: ...


class FunctionWorkerAdapter:
    """Small adapter for local callables and integration tests."""

    def __init__(self, adapter_id: str, function: Callable[[WorkerRequest], WorkerResult]) -> None:
        self.adapter_id = _identifier(adapter_id, "adapter_id")
        if not callable(function):
            raise ValueError("function must be callable")
        self._function = function

    def dispatch(self, request: WorkerRequest) -> WorkerResult:
        return self._function(request)


class WorkerAdapterRegistry:
    """Explicit adapter registry; registration never dispatches work."""

    def __init__(self) -> None:
        self._adapters: dict[str, WorkerAdapter] = {}

    def register(self, adapter: WorkerAdapter) -> None:
        adapter_id = _identifier(adapter.adapter_id, "adapter_id")
        if adapter_id in self._adapters:
            raise ValueError(f"adapter {adapter_id!r} is already registered")
        self._adapters[adapter_id] = adapter

    def get(self, adapter_id: str) -> WorkerAdapter:
        try:
            return self._adapters[adapter_id]
        except KeyError as error:
            raise KeyError(f"worker adapter {adapter_id!r} is not registered") from error


class OracleDispatcher:
    """Dispatch a pinned attempt and construct an identity-safe structured outcome."""

    def __init__(self, adapters: WorkerAdapterRegistry) -> None:
        self.adapters = adapters

    @staticmethod
    def _request(state: OracleState, pending: PendingAttempt) -> WorkerRequest:
        return WorkerRequest(
            stream_id=state.stream_id,
            state_revision=state.revision,
            attempt_id=pending.attempt_id,
            decision_id=pending.decision_id,
            goal_id=pending.decision.goal_id,
            worker_id=pending.worker_id,
            worker_generation=pending.worker_generation,
            proposal=pending.proposal,
        )

    def dispatch(
        self,
        state: OracleState,
        capability: WorkerCapability,
        *,
        outcome_id: str,
    ) -> ActionOutcome:
        _identifier(outcome_id, "outcome_id")
        pending = state.pending
        if pending is None:
            raise ValueError("Oracle state has no selected attempt to dispatch")
        if (capability.worker_id, capability.generation) != (
            pending.worker_id,
            pending.worker_generation,
        ):
            raise ValueError("worker capability does not match the selected generation")
        if not capability.enabled or not capability.supports(pending.proposal):
            raise ValueError("selected worker capability cannot execute the proposal")
        adapter = self.adapters.get(capability.adapter)
        request = self._request(state, pending)
        started = time.monotonic()
        try:
            result = adapter.dispatch(request)
            if not isinstance(result, WorkerResult):
                raise TypeError("worker adapter must return WorkerResult")
        except TimeoutError:
            result = WorkerResult(
                status=OutcomeStatus.TIMED_OUT,
                goal_progress=state.progress[-1].progress if state.progress else 0.0,
                confidence=0.0,
                error_code="adapter_timeout",
            )
        except Exception:
            result = WorkerResult(
                status=OutcomeStatus.FAILED,
                goal_progress=state.progress[-1].progress if state.progress else 0.0,
                confidence=0.0,
                error_code="adapter_exception",
            )
        elapsed = time.monotonic() - started
        return ActionOutcome(
            outcome_id=outcome_id,
            attempt_id=pending.attempt_id,
            decision_id=pending.decision_id,
            action_id=pending.proposal.action_id,
            worker_id=pending.worker_id,
            worker_generation=pending.worker_generation,
            state_revision=state.revision,
            status=result.status,
            goal_progress=result.goal_progress,
            confidence=result.confidence,
            goal_completed=result.goal_completed,
            information_gain=result.information_gain,
            information_measurement=result.information_measurement,
            latency_seconds=result.latency_seconds if result.latency_seconds is not None else elapsed,
            token_cost=result.token_cost,
            compute_cost=result.compute_cost,
            error_code=result.error_code,
            evidence=result.evidence,
        )


__all__ = [
    "FunctionWorkerAdapter",
    "OracleDispatcher",
    "WorkerAdapter",
    "WorkerAdapterRegistry",
    "WorkerRequest",
    "WorkerResult",
]
