import pytest

from engram.oracle import (
    ActionProposal,
    AttentionCandidate,
    CognitiveExecutive,
    DecisionPolicy,
    Evidence,
    Goal,
    GoalGraph,
    GoalStatus,
    MemoryCandidate,
    MemoryDisposition,
    MonitorStatus,
    ProgressObservation,
)


def test_goal_graph_exposes_only_dependency_ready_goals():
    graph = GoalGraph(
        (
            Goal("gather", "Gather sources", priority=0.7),
            Goal("outline", "Create outline", dependencies=("gather",), priority=0.9),
            Goal("style", "Load writing preferences", priority=0.5),
        )
    )

    assert [goal.goal_id for goal in graph.runnable()] == ["gather", "style"]
    completed = graph.with_status("gather", GoalStatus.COMPLETE)
    assert [goal.goal_id for goal in completed.runnable()] == ["outline", "style"]

    with pytest.raises(ValueError, match="acyclic"):
        GoalGraph((Goal("a", "A", ("b",)), Goal("b", "B", ("a",))))


def test_confidence_counts_sources_conservatively_and_applies_age_decay():
    executive = CognitiveExecutive()
    estimate = executive.estimate_confidence(
        (
            Evidence("a", "paper-1", True, 0.9),
            Evidence("b", "paper-1", True, 0.8),
            Evidence("c", "paper-2", False, 0.6),
            Evidence("d", "old-paper", True, 1.0, age_seconds=100.0),
        ),
        half_life_seconds=100.0,
    )

    assert estimate.independent_sources == 3
    assert estimate.supporting_observations == 3
    assert estimate.conflicting_observations == 1
    assert estimate.score == pytest.approx(2.4 / 4.0)


def test_attention_selection_respects_required_items_and_budget():
    executive = CognitiveExecutive()
    selection = executive.select_attention(
        (
            AttentionCandidate("goal", 1.0, 1.0, 20, required=True, category="goal"),
            AttentionCandidate("physics", 0.9, 0.9, 40),
            AttentionCandidate("music", 0.2, 0.8, 40),
            AttentionCandidate("style", 0.7, 1.0, 20),
        ),
        budget_bytes=80,
    )

    assert selection.active_ids == ("goal", "style", "physics")
    assert selection.dormant_ids == ("music",)
    assert selection.total_bytes == 80


def test_memory_curation_returns_policy_decisions_without_mutating_storage():
    decisions = CognitiveExecutive().curate_memory(
        (
            MemoryCandidate("weather", "transient", 0.1, 0.1),
            MemoryCandidate("preference", "user", 0.9, 0.8),
            MemoryCandidate("repeated", "fact", 0.7, 0.8, redundancy=0.95, observations=20),
            MemoryCandidate("shopping", "task", 0.6, 0.3, expires_at=10.0),
        ),
        now=20.0,
    )

    assert [decision.disposition for decision in decisions] == [
        MemoryDisposition.DISCARD,
        MemoryDisposition.KEEP,
        MemoryDisposition.MERGE,
        MemoryDisposition.EXPIRE,
    ]


def test_predictive_decision_selects_best_allowed_action_and_audits_terms():
    executive = CognitiveExecutive()
    goals = GoalGraph((Goal("research", "Research the claim"),))
    actions = (
        ActionProposal("quick", "research", "local_lookup", 0.65, 0.4, 2.0, token_cost=200),
        ActionProposal("careful", "research", "multi_source", 0.95, 0.8, 20.0, token_cost=2000),
        ActionProposal("unsafe", "research", "untrusted_tool", 0.99, 0.9, 1.0, risk=0.9),
    )
    decision = executive.decide(
        goals,
        actions,
        policy=DecisionPolicy(max_latency_seconds=30.0, max_token_cost=3000, max_risk=0.5),
    )

    assert decision.selected.proposal.action_id == "careful"
    assert dict(decision.selected.utility_terms)["success"] == pytest.approx(0.95)
    assert decision.rejected == (("unsafe", "risk exceeds policy limit"),)


@pytest.mark.parametrize(
    ("history", "status", "action"),
    (
        (
            (ProgressObservation(0, 0.2, 0.8), ProgressObservation(1, 0.4, 0.8)),
            MonitorStatus.PROGRESSING,
            "continue",
        ),
        (
            (
                ProgressObservation(0, 0.2, 0.8),
                ProgressObservation(1, 0.2, 0.8),
                ProgressObservation(2, 0.2, 0.8),
            ),
            MonitorStatus.STALLED,
            "change_strategy",
        ),
        (
            (ProgressObservation(0, 0.5, 0.8), ProgressObservation(1, 0.4, 0.8)),
            MonitorStatus.REGRESSING,
            "rollback_or_change_strategy",
        ),
        (
            (ProgressObservation(0, 0.2, 0.8), ProgressObservation(1, 0.3, 0.2)),
            MonitorStatus.UNCERTAIN,
            "seek_evidence",
        ),
    ),
)
def test_self_monitor_selects_a_response(history, status, action):
    decision = CognitiveExecutive().monitor(history)
    assert decision.status is status
    assert decision.recommended_action == action
