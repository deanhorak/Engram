# Evaluation

## Gate 1 definition

For each traced state and layer, all neuron activations are computed. Records are sorted by
the norm of their individual contribution, `abs(a_j) * ||v_j||₂`. Every cumulative prefix
is evaluated. The first prefix satisfying the residual-energy criterion is recorded for
90%, 95%, and 99% targets.

Reports include mean, median, and p95 required neuron fraction, relative L2 error, and
cosine similarity. Results are grouped globally, by layer, and by layer plus input type.
The reconstruction error between extracted weights and captured teacher MLP output is also
reported to catch boundary or extraction errors.

## Trained-teacher MLP intervention gate

Proxy reconstruction and candidate recall are screening metrics. They do not show how errors
propagate through later transformer layers. `engram evaluate-mlp-intervention` therefore runs
held-out sequences through the trained Hugging Face teacher under fixed-token teacher forcing,
then replaces selected `layer.mlp` outputs with one of five arms:

- `identity`: return the exact output to validate hook and metric instrumentation;
- `oracle`: retain the contribution-magnitude top-K records using full activation information;
- `rank16`: use a low-rank multi-label candidate router and exact reranking inside its candidates;
- `overlap`: select a learned combination of coverage-trained overlapping postings, deduplicate
  their records, and exactly rerank the resulting candidates.
- `dip`: select large-magnitude input coordinates, compute predictor-free partial gate/up scores,
  exactly complete only a bounded candidate set, and rerank those candidates to top-K.

The evaluator can run all-layer interventions separately from one-layer-at-a-time attribution. Local
MLP error compares the replacement with the exact MLP at the same, possibly drifted, input. Final
normalized-hidden-state drift and logits compare with a separate untouched teacher pass. The
checked reports currently use all-layer mode; one-layer attribution has not yet been checked in.
Next-token metrics use logits at positions `[:-1]` and targets at `[1:]`. Final-hidden metrics use
all input-token states, while local MLP error and candidate recall use all input-token/layer states;
the JSON statistics record each population count explicitly.

An all-layer arm passes the current quality prerequisite only when all mean held-out checks pass:

| Check | Threshold |
|---|---:|
| Teacher-to-intervention KL | at most 0.05 nat/token |
| Teacher top-1 agreement | at least 0.90 |
| Target NLL delta | at most +0.05 nat/token |
| Final normalized hidden-state relative L2 | at most 0.10 |
| Candidate recall, routed arms only | at least 0.95 |

Progression additionally requires at least 8 unique sequences and 256 next-token positions, a
passing all-layer identity arm, and an all-layer magnitude reference measured at the routed arm's
active K. The standard experiment order waits for a passing reference before spending effort on a
router. The final gate does not require that reference to pass if the routed arm itself passes the
causal quality checks, because a non-magnitude subset can be better. For a learned router,
calibration and evaluation must be disjoint under exact token-sequence hashing; different raw file
hashes alone are insufficient. DIP has no fitted predictor and therefore no training split, though
its hyperparameters are still development choices. A report labeled `confirmation` must supply
the configuration-selection trace corpus and prove zero exact token-sequence overlap with the
evaluation corpus; otherwise the gate rejects that label.

These are engineering progression thresholds, not a proof of user-visible equivalence. A passing
magnitude reference justifies router experiments at that active K. It is not a theoretical
ceiling because magnitude top-K need not be the optimal subset. Only a passing routed arm is
eligible for an experimental serialization step. Traffic, latency, task accuracy, long-context
behavior, and replication on another model remain separate gates.

The checked SmolLM2-135M study uses 16 held-out sequences and 491 next-token positions. K=256 and
K=512 fail the all-layer magnitude-reference gate. K=768 is the first tested pass; it keeps 50% of
every MLP.
Neither the flat rank-16 router nor the checked 192-by-32 overlapping-posting router passes at
K=768 with 1,280 candidates after refitting on all 1,112 calibration states per layer. Their
recalls are 0.889 and 0.868, respectively, and both fail every routed-arm causal-quality check.
They are stopped before serialization and distillation; the result rules out these particular
full-corpus fits, not a different representation or training objective. See
the [decision summary](../reports/smollm2_mlp_intervention/decision.md).

A recall-only screen reuses cached held-out dense-teacher states and packed oracle memberships.
At C=1,280, corpus-scaled regularization peaks at λ=8,000 with 0.900 recall. Increasing C to
1,408 and 1,472 produces 0.954 and 0.978 recall, respectively, and therefore triggers causal
checks. Neither passes downstream quality: even C=1,472 has KL 0.085, top-1 agreement 0.866,
NLL delta +0.055, and final-hidden relative L2 0.131. Recall screening is therefore a useful
cost filter, not a substitute for the intervention gate.

A subsequent trace-only screen fits affine low-rank correction capsules against the exact residual
left by the C=1,280 routed read. Capsules are seeded from the largest residuals and can be limited
to tight failure-region radii. The uncorrected mean local relative L2 is 0.207. The best global
capsule raises it to 0.259; the best tight targeted capsule still raises it to 0.233 while matching
only 7.1% of held-out states. Because no arm improves held-out local error, none proceeds to a
transformer intervention.

Sparse-teacher fine-tuning uses the same exact sequence separation, evidence floor, and held-out
quality thresholds. The dense teacher is frozen. The student executes its sparse route during
training, with local MLP-output, layer-hidden, logit-KL, and oracle-membership losses. The first
rank-8 adapter pilot uses all 32 calibration sequences and all 16 validation sequences. Training
loss falls from 0.436 to 0.326, but held-out recall is 0.900, KL 0.448, top-1 agreement 0.721, NLL
delta +0.343, and final-hidden relative L2 0.250. It therefore remains stopped before package
serialization. A later gradient audit found that the hard top-K route prevents the local, hidden,
and logit losses from updating router scores, while the adapter update is negligible; this pilot
does not test a differentiable soft-to-hard sparse student.

The next experiment replaced learned membership prediction with a predictor-free, DIP-inspired
selector. The published DIP method motivates top-magnitude input pruning and partial activation
scoring; candidate-only exact completion and contribution-norm reranking are Engram extensions. A
trace-only sweep retains the largest absolute MLP-input coordinates, evaluates partial gate/up
projections for all records, exactly completes only candidate records, and reranks their full
contribution scores. It reports both oracle membership recall and retained oracle score mass,
because the failed rank router showed that membership recall alone is not a causal-quality proxy.

Across 507 configuration-selection states per layer, the
75%-input/1,024-candidate/K=768 point reaches 0.9971 recall, 0.9989 score-mass recall, and mean
local relative L2 0.1044 versus 0.1040 for the matched full-information reference. The causal
frontier then establishes the checked-grid boundary:

| Input fraction | Candidates | Projected dense traffic | Recall | KL | Top-1 | NLL delta | Final hidden rel-L2 | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.625 | 896 | 0.7292 | 0.9599 | 0.0378 | 0.8961 | +0.0091 | 0.1006 | fail |
| 0.625 | 1,024 | 0.7500 | 0.9821 | 0.0351 | 0.9084 | +0.0295 | 0.0958 | pass |
| 0.750 | 896 | 0.7639 | 0.9899 | 0.0339 | 0.9124 | +0.0262 | 0.0938 | pass |
| 0.750 | 1,024 | 0.7778 | 0.9971 | 0.0316 | 0.9246 | +0.0308 | 0.0921 | pass |

The recommended development point is 75%/896: its small traffic premium over the cheapest pass
buys better margin on recall, top-1 agreement, and hidden drift. That configuration was then frozen
and run once on a new 16-sequence corpus: 1,184 input states, 1,168 next-token positions, and zero
exact token-sequence overlap with the configuration-selection corpus. The confirmation arm passes
with recall 0.9897, score-mass recall 0.9961, KL 0.0286, top-1 agreement 0.9101, NLL delta +0.0326,
and final-hidden relative L2 0.0905. The machine-readable decision is now
`eligible_for_selector_serialization`. This does not yet
authorize a performance claim: both evaluators express the algorithm using dense operations, and
the byte formula excludes index traffic, activations, cache-line waste, and sorting. A packed
cache-aware kernel and hardware measurement remain required.

The experiments were run in stages. The checked
[composite report](../reports/smollm2_mlp_intervention_composite/mlp_intervention.json) verifies
that source-model hash, evaluation hash, layer scope, baseline counts, and routed calibration
provenance agree before applying one machine-readable decision across all arms.

The forward-hook harness executes the original dense MLP before replacing its output and also
recomputes activations for exact measurement. Its quality metrics are valid, but its wall time is
not an Engram inference benchmark.

## Evidence labels

- `pipeline_validation`: deterministic random fixture; no model-quality conclusion.
- `measured_local_model`: a user-supplied trained checkpoint and held-out trace data.

The original energy-prefix Gate 1 study includes a fitted background ablation, which failed to
improve held-out error. The intervention gate above supersedes proxy error as the progression
decision. Gates 3–4 still have only random/synthetic pipeline reports. `engram evaluate-e2e` measures
student NLL/perplexity, teacher KL, top-1/top-5 agreement, category accuracy, repetition, and
fixed examples against a cached Hugging Face teacher. Model IDs are downloaded automatically,
while local model directories remain offline-capable. Trained SmolLM2 semantic interventions have
run, but no trained compiled-package Gate 5 evaluation has, so no Gate 5 quality target is claimed.

The system-level Cognitive Executive has separate goal, confidence-calibration, action-utility,
attention, memory, monitoring, and safety gates defined in
[its design document](cognitive_executive.md). Compiler gates do not imply executive success.
