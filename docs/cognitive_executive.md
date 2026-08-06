# The Oracle cognitive executive

## Final conclusion

The Oracle scaffolding is an independent systems experiment and did not change the central outcome:
Engram did not produce an LLM runtime superior or equivalent to current practical technology. Its
event stores, policies, and adapters should be treated as research infrastructure, not evidence of
successful model replacement.

Status: **durable policy/session and adapter scaffolds implemented; production workers and learned predictors are not**

This subsystem is tracked separately from the compiled model worker's
native-BitNet semantic gate, which now passes by postmortem adjudication. See
[Project status](status.md) for the precise evidence boundary and the remaining
Milestone 2 work.

Engram now distinguishes two very different control problems. Inside a compiled model, the
token-level recurrent controller updates a numeric state during generation. Above one or more
models, an optional system-level **Cognitive Executive**, called the Oracle, manages goals,
evidence, attention budgets, memory policy, strategies, and worker selection.

```text
user / application
        |
        v
Oracle cognitive executive
  goal graph | evidence confidence | attention | memory policy
  strategy/action proposals | cost/risk policy | progress monitor
        |
        v
typed action selected for dispatch
        |
        +----------+------------+--------------+
        |          |            |              |
   Engram model   LLM      symbolic solver   external tool
        |
        v
validated outcome -> evidence, progress, and policy updates
```

The Oracle does not answer questions or generate prose. It produces typed, auditable decisions.
A worker may generate prose, run a search, execute a planner, or invoke a model, but dispatch is a
separate integration boundary. This makes it possible to test executive policy without silently
performing external actions.

## Why this is separate from the compiled runtime

An `.engram` package is one possible cognitive worker. It contains compiled model weights,
token-level state machinery, and bounded sequence memory. It does not contain user goal graphs,
durable personal memory, tool credentials, a model registry, or system-wide policy.

The word *oracle* also appears in the semantic-memory experiments. There it means the
**magnitude-oracle baseline**: an offline full scan used to measure which MLP records matter. That
measurement baseline is unrelated to the Cognitive Executive. Likewise:

| Term | Compiled worker meaning | Executive meaning |
|---|---|---|
| Controller | GRU-like token-state transition | Goal and action policy layer |
| Semantic memory | Immutable compiled MLP records | Evidence or factual knowledge store |
| Episodic memory | Bounded state for the current sequence | Potential durable event history |
| Attention | Token/context computation | Selection of memories, workers, and tools |
| Confidence | Residual proxy or logit margin | Evidence-backed, outcome-calibrated belief |

These mechanisms may eventually exchange telemetry, but they must not be described as already
equivalent.

## Responsibilities

### Goal manager

`GoalGraph` represents work as an acyclic dependency graph. A goal becomes runnable only after
all dependencies complete. This turns a request such as writing a research paper into explicit
steps—gather evidence, verify it, outline, draft, and review—rather than one opaque prompt.

The current implementation validates dependencies, rejects cycles, prioritizes runnable goals,
and returns immutable status updates. Decomposition from natural language remains a worker task;
the scaffold does not pretend to infer a correct graph by itself.

### Attention controller

`AttentionCandidate` gives every memory or context item a relevance score, confidence score,
byte cost, category, and required flag. The executive activates required items first and then
selects optional items under a byte budget. Dormant items remain addressable without entering the
worker context.

The initial selector is a deterministic density heuristic, not a learned retrieval policy or an
optimal knapsack solver. Its purpose is to make the budget and exclusions visible.

### Confidence estimator

`Evidence` records provenance, direction, reliability, and age. Repeated observations sharing a
source ID do not count as independent support. Optional time decay reduces stale evidence, and a
symmetric prior prevents one observation from producing certainty.

The resulting value is a policy score, **not calibrated epistemic probability**. Before it can
justify values such as 0.97, the estimator must be evaluated against real outcomes using Brier
score, calibration error, and domain-specific reliability models. Source counts alone do not
establish truth, and independence cannot be inferred merely from different URLs or model names.

### Memory curator

`curate_memory` proposes `keep`, `merge`, `expire`, or `discard` dispositions from explicit
salience, reuse, redundancy, and expiry metadata. It never deletes data. A storage layer must
apply retention rules, user consent, provenance requirements, legal holds, and recoverability
before executing any disposition. Forgetting an index entry must also remain distinct from
deleting its source record.

### Strategy and worker selector

`ActionProposal` describes a candidate strategy and optional model/tool, along with predicted
success, information gain, latency, token cost, compute cost, risk, and required attention IDs.
`DecisionPolicy` applies user- or application-supplied constraints and weights. The selected
action includes every utility term, ranked alternatives, and reasons for rejected actions.

Cost units are deliberately explicit. A policy should be calibrated for its deployment rather
than assuming that a token, second, joule, dollar, or privacy exposure is interchangeable.

### Self-monitor

`ProgressObservation` records progress, confidence, and failures at event or budget checkpoints.
The monitor returns `progressing`, `stalled`, `regressing`, `uncertain`, or `complete`, together
with a typed recommendation such as continuing, seeking evidence, or changing strategy.

This is not hidden chain-of-thought inspection. It monitors observable task state and outcomes.

### Predictive action choice

Predictive selection cuts across all six responsibilities. Before dispatch, the executive asks:

- How likely is the action to succeed?
- How much decision-relevant information should it provide?
- What latency, token, compute, monetary, and privacy costs will it incur?
- Does it satisfy user policy and risk constraints?
- Is a cheaper action adequate for this goal?

The current utility function is deterministic and hand-configured. Learned outcome, information,
latency, and cost predictors remain future work.

## Typed execution loop

The intended request-level loop is:

1. Observe the request, user policy, available workers, and current task state.
2. Create or update an explicit goal graph.
3. Retrieve scoped evidence and select attention under a budget.
4. Enumerate typed action proposals from registered strategies and workers.
5. Predict success, information gain, cost, latency, and risk.
6. Select an allowed action and persist its complete decision record.
7. Dispatch through an adapter and validate the structured outcome.
8. Update evidence, confidence, goal progress, and proposed memory dispositions.
9. Continue, clarify, retry, change strategy, escalate, or stop at a checkpoint.

Today the repository implements steps 2, 3, 6, structured outcome validation and state updates in
step 8, and the policy decision in step 9. It also implements in-memory, SQLite, and JSONL event
stores; a worker capability registry; and an explicit adapter/dispatcher boundary. It does not yet
implement production model or tool adapters, a strategy registry, domain-specific output-content
validators, or learned predictors.

## Revisioned sessions and event replay

`OracleSession` wraps the pure policy in an append-only lifecycle. A stream begins with
`session_started`, then alternates between `action_selected` and `outcome_observed` events. Every
event receives a contiguous stream revision from `InMemoryOracleEventStore`. Appends use an
expected revision, so two writers racing from the same state cannot both succeed.

All three stores provide strict idempotency: repeating an event ID with identical content is a no-op;
reusing it with different content is a conflict. `replay_oracle_events` folds only recorded event
content and does not consult the current clock, registry, or external services. Replaying the same
log therefore reconstructs the same goals, resource ledger, evidence, progress, pending attempt,
and calibration records.

The reference lifecycle allows one pending attempt per session. Selection reserves its predicted
latency, token, and compute budget before dispatch authorization. A terminal outcome must match
the exact attempt, decision, action, worker ID, worker generation, and selected state revision.
Settlement releases the reservation and charges measured usage. If telemetry is absent, the
reservation is charged conservatively while the missing value remains absent from calibration.
Actual overruns are preserved in the ledger and block work that no longer fits; they are never
rewritten to match the prediction.

### Durable backends

`SQLiteOracleEventStore` stores events in a transactional table with unique stream/revision and
stream/event-ID constraints. Appends run under `BEGIN IMMEDIATE`, so revision comparison and the
entire event batch commit atomically across connections. WAL mode supports durable restart and
concurrent readers.

`JSONLOracleEventStore` writes one complete append batch per line. Each line has a format version
and SHA-256 checksum over its canonical event array. Writers use an OS file lock where available,
then flush and `fsync` before returning. A torn line, checksum mismatch, unknown schema/type, gap,
duplicate revision, or malformed payload fails loudly rather than being skipped.

Both formats use an allow-listed, versioned JSON codec for the executive's dataclasses, enums, and
tuples. They never load pickle data or dynamically import a type named by the log. SQLite is the
recommended operational backend; JSONL is useful when direct inspection and simple archival are
more important than write throughput.

## Worker capability registry

`WorkerRegistry` is immutable and revisioned. A `WorkerCapability` declares a worker generation,
supported strategies, model and tool IDs, adapter kind, locality, trust, and enabled state.
Updating or disabling a worker creates a new generation. New selections may use only the latest
enabled generation, while outcomes from already selected attempts remain valid against the
generation pinned in their event. Each decision also records the registry revision it used.

The registry describes capability and policy eligibility, not transient health or concurrency.
Availability, credentials, data-boundary enforcement, and adapter execution remain dispatch-layer
responsibilities.

## Worker adapters and structured outcomes

`WorkerAdapter` receives a `WorkerRequest` whose attempt, decision, state revision, goal, worker
generation, and immutable proposal come from the selected event. An adapter returns only a
validated `WorkerResult`. `OracleDispatcher` stamps authoritative identity onto the resulting
`ActionOutcome`; a worker cannot choose which attempt or revision its output settles.

Adapters are registered explicitly by adapter ID. There is no dynamic import, `eval`, shell
command construction, or credential material in the event payload. The included
`FunctionWorkerAdapter` is intended for local integrations and tests. Adapter exceptions and
timeouts become structured failure outcomes with stable error codes rather than leaking exception
text into the log.

Selection is durably recorded before invocation, and the stable `attempt_id` is passed to every
worker as its idempotency key. There is nevertheless an unavoidable crash boundary between an
external side effect and recording its outcome. The dispatcher therefore provides **at-least-once
attempt semantics**, not magical exactly-once execution. Production adapters must make retries
idempotent by attempt ID or query the worker for an earlier result before repeating side effects.

## Outcome calibration

Every observed attempt preserves the original prediction beside independently measured outcomes.
`summarize_calibration` reports:

- success Brier score, absolute error, signed bias, observed rate, and fixed-bin calibration error;
- information-gain MSE, absolute error, signed bias, and calibration error;
- latency absolute error, signed bias, and log-space absolute error;
- token and compute error separately so incompatible units are never combined.

Cancelled attempts are excluded from success calibration. Failures and dispatched timeouts count
as unsuccessful. Missing measurements affect only their own metric count and are never treated as
zero. Information gain requires a named measurement method; a worker's unsupported self-report is
not sufficient provenance.

These metrics cover selected actions only. They cannot estimate whether an unselected alternative
would have done better, so utility regret and counterfactual worker comparisons require controlled
exploration or randomized evaluation data.

## Minimal API example

```python
from engram.oracle import ActionProposal, CognitiveExecutive, DecisionPolicy, Goal, GoalGraph

executive = CognitiveExecutive()
goals = GoalGraph((Goal("research", "Verify the technical claim"),))
decision = executive.decide(
    goals,
    (
        ActionProposal("local", "research", "local_retrieval", 0.65, 0.4, 2.0),
        ActionProposal("deep", "research", "multi_source_review", 0.95, 0.8, 20.0),
    ),
    policy=DecisionPolicy(max_latency_seconds=30.0, max_risk=0.2),
)

assert decision.selected.proposal.action_id == "deep"
```

The returned proposal is a request to dispatch, not evidence that dispatch occurred.

A revisioned lifecycle pins and observes the selected worker explicitly:

```python
from engram.oracle import (
    OracleSession,
    ResourceBudget,
    SQLiteOracleEventStore,
    WorkerCapability,
    WorkerRegistry,
)

registry = WorkerRegistry().register(
    WorkerCapability("local-model", 1, ("multi_source_review",), local=True)
)
session = OracleSession(SQLiteOracleEventStore("oracle-events.sqlite"))
state = session.start("task-42", goals, ResourceBudget(60.0, 4000, 10.0, 4))
state = session.plan(
    "task-42",
    expected_revision=state.revision,
    attempt_id="review-1",
    proposals=(decision.selected.proposal,),
    registry=registry,
)
```

`plan` records and reserves the action but still does not invoke the worker. `OracleDispatcher`
uses the registered adapter and returns a matching `ActionOutcome`, which is committed through
`session.observe`.

## Safety and privacy requirements

Before external integrations are added, the executive needs:

- per-worker capability and trust declarations;
- explicit local/cloud routing policy and data-boundary enforcement;
- tool authorization, parameter validation, and least-privilege credentials;
- access control, backup, retention, and encryption policy for the durable decision/outcome log;
- user-visible memory retention and deletion controls;
- provenance and conflict preservation during memory merging;
- hard budget, time, and retry limits;
- human confirmation for high-impact actions.

## Research gates

The Cognitive Executive has a separate evaluation program from model compilation:

1. **Goal execution:** completion rate, dependency correctness, and unnecessary-step rate.
2. **Confidence:** Brier score, expected calibration error, selective accuracy, and conflict tests.
3. **Predictive policy:** success/latency/cost prediction error and utility regret versus known
   alternatives.
4. **Attention:** relevant-evidence coverage, irrelevant-context rate, byte cost, and downstream
   task impact.
5. **Memory:** bounded growth, retention correctness, merge provenance, expiry correctness, and
   recoverability.
6. **Monitoring:** stall detection precision/recall, productive strategy-switch rate, and retry
   waste.
7. **Safety:** policy-violation rate, unauthorized dispatch attempts, data-boundary violations,
   and audit completeness.

Passing compiler or token-generation gates does not pass these executive gates, and a successful
executive scaffold does not establish that the compiled language model preserves quality.
