"""Deterministic policy scaffold for Engram's system-level Oracle."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from .models import (
    ActionProposal,
    AttentionCandidate,
    AttentionSelection,
    ConfidenceEstimate,
    DecisionPolicy,
    Evidence,
    GoalGraph,
    MemoryCandidate,
    MemoryDecision,
    MemoryDisposition,
    MonitorDecision,
    MonitorStatus,
    OracleDecision,
    ProgressObservation,
    ScoredAction,
    ensure_unique_ids,
)


class CognitiveExecutive:
    """Plan and audit actions without executing models, tools, or memory writes."""

    def estimate_confidence(
        self, evidence: Iterable[Evidence], *, half_life_seconds: float | None = None
    ) -> ConfidenceEstimate:
        observations = tuple(evidence)
        ensure_unique_ids((item.evidence_id for item in observations), "evidence")
        if half_life_seconds is not None and (
            not math.isfinite(half_life_seconds) or half_life_seconds <= 0.0
        ):
            raise ValueError("half_life_seconds must be finite and positive")

        # Repeated observations from one source are not treated as independent.
        by_source: dict[str, list[Evidence]] = defaultdict(list)
        for item in observations:
            by_source[item.source_id].append(item)
        support_weight = 0.0
        conflict_weight = 0.0
        for source_items in by_source.values():
            best_support = 0.0
            best_conflict = 0.0
            for item in source_items:
                decay = (
                    1.0
                    if half_life_seconds is None
                    else 0.5 ** (item.age_seconds / half_life_seconds)
                )
                weight = item.reliability * decay
                if item.supports:
                    best_support = max(best_support, weight)
                else:
                    best_conflict = max(best_conflict, weight)
            support_weight += best_support
            conflict_weight += best_conflict
        total = support_weight + conflict_weight
        # A symmetric unit prior prevents one weak observation from producing
        # unjustified certainty. This is still a policy score until calibrated
        # against outcomes for a particular domain.
        score = (1.0 + support_weight) / (2.0 + total)
        support_count = sum(item.supports for item in observations)
        conflict_count = len(observations) - support_count
        reasons = (
            f"{support_count} supporting observations",
            f"{conflict_count} conflicting observations",
            f"{len(by_source)} independent source IDs",
        )
        return ConfidenceEstimate(
            score=score,
            supporting_observations=support_count,
            conflicting_observations=conflict_count,
            independent_sources=len(by_source),
            reasons=reasons,
        )

    def select_attention(
        self, candidates: Iterable[AttentionCandidate], *, budget_bytes: int
    ) -> AttentionSelection:
        items = tuple(candidates)
        ensure_unique_ids((item.item_id for item in items), "attention candidates")
        if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int) or budget_bytes < 0:
            raise ValueError("budget_bytes must be a non-negative integer")
        required = tuple(sorted((item for item in items if item.required), key=lambda item: item.item_id))
        required_bytes = sum(item.byte_cost for item in required)
        if required_bytes > budget_bytes:
            raise ValueError("required attention items exceed the byte budget")
        optional = sorted(
            (item for item in items if not item.required),
            key=lambda item: (
                -(item.relevance * item.confidence / max(item.byte_cost, 1)),
                -(item.relevance * item.confidence),
                item.item_id,
            ),
        )
        active = list(required)
        used = required_bytes
        for item in optional:
            if used + item.byte_cost <= budget_bytes:
                active.append(item)
                used += item.byte_cost
        active_ids = tuple(item.item_id for item in active)
        active_set = set(active_ids)
        return AttentionSelection(
            active_ids=active_ids,
            dormant_ids=tuple(item.item_id for item in items if item.item_id not in active_set),
            total_bytes=used,
            budget_bytes=budget_bytes,
        )

    def curate_memory(
        self,
        candidates: Iterable[MemoryCandidate],
        *,
        now: float,
        discard_threshold: float = 0.15,
        merge_threshold: float = 0.8,
    ) -> tuple[MemoryDecision, ...]:
        items = tuple(candidates)
        ensure_unique_ids((item.memory_id for item in items), "memory candidates")
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        if (
            not math.isfinite(discard_threshold)
            or not 0.0 <= discard_threshold <= 1.0
            or not math.isfinite(merge_threshold)
            or not 0.0 <= merge_threshold <= 1.0
        ):
            raise ValueError("memory thresholds must lie in [0, 1]")
        decisions: list[MemoryDecision] = []
        for item in items:
            if item.expires_at is not None and item.expires_at <= now:
                disposition = MemoryDisposition.EXPIRE
                reason = "configured expiry has passed"
            elif item.redundancy >= merge_threshold and item.observations > 1:
                disposition = MemoryDisposition.MERGE
                reason = "repeated, highly redundant evidence"
            elif item.salience * item.expected_reuse < discard_threshold:
                disposition = MemoryDisposition.DISCARD
                reason = "low salience and expected reuse"
            else:
                disposition = MemoryDisposition.KEEP
                reason = "retained by salience/reuse policy"
            decisions.append(MemoryDecision(item.memory_id, disposition, reason))
        return tuple(decisions)

    def decide(
        self,
        goals: GoalGraph,
        proposals: Iterable[ActionProposal],
        *,
        policy: DecisionPolicy = DecisionPolicy(),
    ) -> OracleDecision:
        actions = tuple(proposals)
        ensure_unique_ids((action.action_id for action in actions), "action proposals")
        runnable = goals.runnable()
        if not runnable:
            raise ValueError("goal graph has no runnable goals")
        goal = runnable[0]
        rejected: list[tuple[str, str]] = []
        scored: list[ScoredAction] = []
        for action in actions:
            if action.goal_id != goal.goal_id:
                rejected.append((action.action_id, "proposal does not target the selected goal"))
                continue
            if policy.max_latency_seconds is not None and action.latency_seconds > policy.max_latency_seconds:
                rejected.append((action.action_id, "latency exceeds policy limit"))
                continue
            if policy.max_token_cost is not None and action.token_cost > policy.max_token_cost:
                rejected.append((action.action_id, "token cost exceeds policy limit"))
                continue
            if policy.max_risk is not None and action.risk > policy.max_risk:
                rejected.append((action.action_id, "risk exceeds policy limit"))
                continue
            terms = (
                ("success", policy.success_weight * action.predicted_success),
                ("information", policy.information_weight * action.information_gain),
                ("latency", -policy.latency_weight * math.log1p(action.latency_seconds)),
                ("tokens", -policy.token_weight * action.token_cost / 1000.0),
                ("compute", -policy.compute_weight * action.compute_cost),
                ("risk", -policy.risk_weight * action.risk),
            )
            scored.append(ScoredAction(action, sum(value for _, value in terms), terms))
        if not scored:
            raise ValueError("no action proposal satisfies the selected goal and policy")
        ranked = sorted(scored, key=lambda item: (-item.utility, item.proposal.action_id))
        return OracleDecision(
            goal_id=goal.goal_id,
            selected=ranked[0],
            alternatives=tuple(ranked[1:]),
            rejected=tuple(rejected),
        )

    def monitor(
        self,
        observations: Iterable[ProgressObservation],
        *,
        window: int = 3,
        minimum_progress: float = 0.01,
        confidence_threshold: float = 0.4,
    ) -> MonitorDecision:
        history = tuple(observations)
        if not history:
            raise ValueError("monitoring requires at least one observation")
        if any(later.step <= earlier.step for earlier, later in zip(history, history[1:])):
            raise ValueError("progress observation steps must be strictly increasing")
        if isinstance(window, bool) or not isinstance(window, int) or window < 2:
            raise ValueError("window must be an integer of at least two")
        if not math.isfinite(minimum_progress) or minimum_progress < 0.0:
            raise ValueError("minimum_progress must be finite and non-negative")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must lie in [0, 1]")
        recent = history[-window:]
        delta = recent[-1].progress - recent[0].progress
        if recent[-1].progress >= 1.0:
            return MonitorDecision(MonitorStatus.COMPLETE, "close_goal", delta)
        if recent[-1].confidence < confidence_threshold:
            return MonitorDecision(MonitorStatus.UNCERTAIN, "seek_evidence", delta)
        if delta < 0.0:
            return MonitorDecision(MonitorStatus.REGRESSING, "rollback_or_change_strategy", delta)
        if len(recent) >= window and (delta < minimum_progress or recent[-1].failures > recent[0].failures):
            return MonitorDecision(MonitorStatus.STALLED, "change_strategy", delta)
        return MonitorDecision(MonitorStatus.PROGRESSING, "continue", delta)
