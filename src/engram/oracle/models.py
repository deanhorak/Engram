"""Typed state and decisions for Engram's system-level cognitive executive."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable


def _probability(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return result


def _nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


class GoalStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE = "complete"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class Goal:
    goal_id: str
    description: str
    dependencies: tuple[str, ...] = ()
    status: GoalStatus = GoalStatus.PENDING
    priority: float = 0.5

    def __post_init__(self) -> None:
        _identifier(self.goal_id, "goal_id")
        _identifier(self.description, "description")
        if not isinstance(self.dependencies, tuple):
            raise ValueError("goal dependencies must be a tuple")
        for dependency in self.dependencies:
            _identifier(dependency, "goal dependency")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("goal dependencies must be unique")
        if self.goal_id in self.dependencies:
            raise ValueError("a goal cannot depend on itself")
        if not isinstance(self.status, GoalStatus):
            raise ValueError("status must be a GoalStatus")
        _probability(self.priority, "priority")


@dataclass(frozen=True)
class GoalGraph:
    goals: tuple[Goal, ...]

    def __post_init__(self) -> None:
        if not self.goals:
            raise ValueError("a goal graph must contain at least one goal")
        by_id = {goal.goal_id: goal for goal in self.goals}
        if len(by_id) != len(self.goals):
            raise ValueError("goal IDs must be unique")
        for goal in self.goals:
            missing = set(goal.dependencies) - set(by_id)
            if missing:
                raise ValueError(f"goal {goal.goal_id!r} has unknown dependencies: {sorted(missing)}")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(goal_id: str) -> None:
            if goal_id in visiting:
                raise ValueError("goal graph must be acyclic")
            if goal_id in visited:
                return
            visiting.add(goal_id)
            for dependency in by_id[goal_id].dependencies:
                visit(dependency)
            visiting.remove(goal_id)
            visited.add(goal_id)

        for goal_id in by_id:
            visit(goal_id)

    def get(self, goal_id: str) -> Goal:
        for goal in self.goals:
            if goal.goal_id == goal_id:
                return goal
        raise KeyError(goal_id)

    def runnable(self) -> tuple[Goal, ...]:
        completed = {goal.goal_id for goal in self.goals if goal.status is GoalStatus.COMPLETE}
        runnable = [
            goal
            for goal in self.goals
            if goal.status in {GoalStatus.PENDING, GoalStatus.ACTIVE}
            and set(goal.dependencies) <= completed
        ]
        return tuple(sorted(runnable, key=lambda goal: (-goal.priority, goal.goal_id)))

    def with_status(self, goal_id: str, status: GoalStatus) -> "GoalGraph":
        self.get(goal_id)
        if not isinstance(status, GoalStatus):
            raise ValueError("status must be a GoalStatus")
        return GoalGraph(
            tuple(
                replace(goal, status=status) if goal.goal_id == goal_id else goal
                for goal in self.goals
            )
        )


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source_id: str
    supports: bool
    reliability: float
    age_seconds: float = 0.0

    def __post_init__(self) -> None:
        _identifier(self.evidence_id, "evidence_id")
        _identifier(self.source_id, "source_id")
        if not isinstance(self.supports, bool):
            raise ValueError("supports must be a boolean")
        _probability(self.reliability, "reliability")
        _nonnegative(self.age_seconds, "age_seconds")


@dataclass(frozen=True)
class ConfidenceEstimate:
    score: float
    supporting_observations: int
    conflicting_observations: int
    independent_sources: int
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _probability(self.score, "score")


@dataclass(frozen=True)
class AttentionCandidate:
    item_id: str
    relevance: float
    confidence: float
    byte_cost: int
    required: bool = False
    category: str = "memory"

    def __post_init__(self) -> None:
        _identifier(self.item_id, "item_id")
        _identifier(self.category, "category")
        _probability(self.relevance, "relevance")
        _probability(self.confidence, "confidence")
        if not isinstance(self.required, bool):
            raise ValueError("required must be a boolean")
        if isinstance(self.byte_cost, bool) or not isinstance(self.byte_cost, int) or self.byte_cost < 0:
            raise ValueError("byte_cost must be a non-negative integer")


@dataclass(frozen=True)
class AttentionSelection:
    active_ids: tuple[str, ...]
    dormant_ids: tuple[str, ...]
    total_bytes: int
    budget_bytes: int


class MemoryDisposition(str, Enum):
    KEEP = "keep"
    MERGE = "merge"
    EXPIRE = "expire"
    DISCARD = "discard"


@dataclass(frozen=True)
class MemoryCandidate:
    memory_id: str
    kind: str
    salience: float
    expected_reuse: float
    redundancy: float = 0.0
    expires_at: float | None = None
    observations: int = 1

    def __post_init__(self) -> None:
        _identifier(self.memory_id, "memory_id")
        _identifier(self.kind, "kind")
        _probability(self.salience, "salience")
        _probability(self.expected_reuse, "expected_reuse")
        _probability(self.redundancy, "redundancy")
        if self.expires_at is not None and not math.isfinite(self.expires_at):
            raise ValueError("expires_at must be finite")
        if isinstance(self.observations, bool) or not isinstance(self.observations, int) or self.observations <= 0:
            raise ValueError("observations must be a positive integer")


@dataclass(frozen=True)
class MemoryDecision:
    memory_id: str
    disposition: MemoryDisposition
    reason: str


@dataclass(frozen=True)
class ActionProposal:
    action_id: str
    goal_id: str
    strategy: str
    predicted_success: float
    information_gain: float
    latency_seconds: float
    token_cost: int = 0
    compute_cost: float = 0.0
    risk: float = 0.0
    model_id: str | None = None
    tool_id: str | None = None
    attention_ids: tuple[str, ...] = ()
    worker_id: str | None = None
    worker_generation: int | None = None
    predictor_id: str = "manual"

    def __post_init__(self) -> None:
        _identifier(self.action_id, "action_id")
        _identifier(self.goal_id, "goal_id")
        _identifier(self.strategy, "strategy")
        _probability(self.predicted_success, "predicted_success")
        _probability(self.information_gain, "information_gain")
        _nonnegative(self.latency_seconds, "latency_seconds")
        _nonnegative(self.compute_cost, "compute_cost")
        _probability(self.risk, "risk")
        if isinstance(self.token_cost, bool) or not isinstance(self.token_cost, int) or self.token_cost < 0:
            raise ValueError("token_cost must be a non-negative integer")
        if self.model_id is not None:
            _identifier(self.model_id, "model_id")
        if self.tool_id is not None:
            _identifier(self.tool_id, "tool_id")
        if self.worker_id is not None:
            _identifier(self.worker_id, "worker_id")
        if self.worker_generation is not None and (
            isinstance(self.worker_generation, bool)
            or not isinstance(self.worker_generation, int)
            or self.worker_generation <= 0
        ):
            raise ValueError("worker_generation must be a positive integer")
        if self.worker_generation is not None and self.worker_id is None:
            raise ValueError("worker_generation requires worker_id")
        _identifier(self.predictor_id, "predictor_id")
        if len(set(self.attention_ids)) != len(self.attention_ids):
            raise ValueError("attention_ids must be unique")


@dataclass(frozen=True)
class DecisionPolicy:
    success_weight: float = 1.0
    information_weight: float = 0.4
    latency_weight: float = 0.05
    token_weight: float = 0.02
    compute_weight: float = 0.1
    risk_weight: float = 0.8
    max_latency_seconds: float | None = None
    max_token_cost: int | None = None
    max_risk: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "success_weight",
            "information_weight",
            "latency_weight",
            "token_weight",
            "compute_weight",
            "risk_weight",
        ):
            _nonnegative(getattr(self, name), name)
        if self.max_latency_seconds is not None:
            _nonnegative(self.max_latency_seconds, "max_latency_seconds")
        if self.max_token_cost is not None and (
            isinstance(self.max_token_cost, bool)
            or not isinstance(self.max_token_cost, int)
            or self.max_token_cost < 0
        ):
            raise ValueError("max_token_cost must be a non-negative integer")
        if self.max_risk is not None:
            _probability(self.max_risk, "max_risk")


@dataclass(frozen=True)
class ScoredAction:
    proposal: ActionProposal
    utility: float
    utility_terms: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class OracleDecision:
    goal_id: str
    selected: ScoredAction
    alternatives: tuple[ScoredAction, ...]
    rejected: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ProgressObservation:
    step: int
    progress: float
    confidence: float
    failures: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        _probability(self.progress, "progress")
        _probability(self.confidence, "confidence")
        if isinstance(self.failures, bool) or not isinstance(self.failures, int) or self.failures < 0:
            raise ValueError("failures must be a non-negative integer")


class MonitorStatus(str, Enum):
    PROGRESSING = "progressing"
    STALLED = "stalled"
    REGRESSING = "regressing"
    UNCERTAIN = "uncertain"
    COMPLETE = "complete"


@dataclass(frozen=True)
class MonitorDecision:
    status: MonitorStatus
    recommended_action: str
    progress_delta: float


def ensure_unique_ids(values: Iterable[str], name: str) -> None:
    collected = tuple(values)
    if len(set(collected)) != len(collected):
        raise ValueError(f"{name} must have unique IDs")
