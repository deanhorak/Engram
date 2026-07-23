# Evaluation

## Current combined-gate decision

As of 2026-07-23, no candidate passes the causal quality thresholds and the
complete physical cold-traffic threshold together. Predictor-free DIP passes
quality on an untouched confirmation corpus but reaches 83.33% cache-line
traffic and is slower than dense in the checked native benchmark. The
serialized mild-width compact-Q4 student reaches 44.9334% traffic but fails
quality after 3,000,093 training positions. The latest 1M-prototype
output-memory experiment is layer-local only and fails its predeclared
progression screen.

See [Project status](status.md) and the
[machine-readable snapshot](../reports/semantic_gate_status_2026-07-23/summary.json).
The sections below define the individual experiments and preserve their
historical evidence.

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
checked adaptive-budget selection report uses one-layer attribution at five active counts.
`--layer-top-k` adds one all-layer magnitude-reference arm with a fixed per-layer schedule; its
mean active count must be integral, and confirmation reports require a sequence-disjoint
configuration-selection trace corpus.
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

The replacement hardware-aware trainer does test that missing mechanism. Unit tests require
nonzero router gradients from causal output and locality losses with membership BCE absent. Its
real-model smoke arm fixes `q=62.5%`, `C=K=512`, reports both scalar and 64-byte-line-adjusted
traffic, and retains the same held-out causal thresholds. One training record is sufficient only
to validate execution and reporting; it is explicitly below the evidence floor for a quality
decision.

The complete hardware-aware run uses all 32 training and 16 held-out sequences. It passes the
evidence and hardware-budget checks but fails recall and every causal-quality check: recall 0.8959,
KL 0.1659, top-1 0.7678, NLL delta +0.1261, and final-hidden relative L2 0.1988. Its candidates
occupy 99.86% of physical gate/up line groups. Follow-up storage-permutation, complete-line,
blend-scale, three-projection LoRA, and broader-corpus screens do not justify another full run.

Later diagnostics close the remaining low-budget variants. Exact top-512 membership itself touches
95.86/96 contiguous record lines; a perfect 80-line selector can cover only 91.75% of those records.
Correcting LoRA initialization/scaling and adding a rank-32 residual improves the complete run to
KL 0.152, top-1 0.780, NLL +0.100, and hidden L2 0.193, still a full failure. The residual has
negligible alignment with the omitted output, and higher learning rates are unstable. Finally, a
same-total layer-adaptive schedule chosen from individual-layer causal measurements on four
separate sequences fails confirmation with KL 0.134, top-1 0.786, NLL +0.110, and hidden L2 0.185.
The next experiment must co-train structured sparsity with the MLP basis rather than retune this
post-hoc selector.

Before committing to that expensive run, `engram evaluate-structured-experts` now performs a
trace-only shadow screen. It clusters records by held-out-safe calibration contribution profiles,
applies one lossless gate/up/down permutation, constructs contiguous expert blocks, and compares a
full-information greedy-residual block reference with a fitted linear block router. Dense-shadow
parity, exact active records, and projected physical traffic are checked separately. This is not an
all-layer intervention and cannot pass the causal gate.

At exactly 512 active records, 64-, 32-, and 16-record blocks produce greedy-reference local
relative-L2 errors of 0.547, 0.497, and 0.438. Their fitted-router errors are 0.655, 0.638, and
0.624. All fail the 0.20 local pretraining screen. The finest layout also reaches 35.42% projected
dense traffic with its full router, just above the 35% screen. Static grouping is therefore stopped;
the next feasibility experiment must train native channel routing and the MLP basis together.

The native-gate shadow evaluates that alternative without a predictor or exact-completion pass.
It computes either the full gate or the gate on top-magnitude input coordinates, selects 512
channels from gate utility, and evaluates exact up/down projections only for those channels. The
exact contribution top-512 reference has local relative L2 0.190. Dense-gate selection is 0.375;
q=62.5% and q=50% are 0.386 and 0.402 at 43.06% and 38.89% ideal traffic. This isolates learned
channel utility as the main problem.

`engram train-native-gate-traces` trains selected MLP layers on cached teacher boundaries through
the exact hard sparse forward and uses a dense surrogate only for backward selection gradients.
Its representative layer-14 arm improves held-out error from 0.4146 to 0.4040 after 64 steps and
keeps dense-shadow error at 0.0339, but misses the declared 10% improvement screen. The artifact is
diagnostic and cannot enter serialization. Cached-boundary tuning is stopped before an all-layer
run; only end-to-end causal training can test the remaining hypothesis.

`engram train-native-gate-e2e` performs that causal experiment on either CPU or an optional CUDA
device with identical semantics. It progressively anneals q/K, freezes non-MLP parameters, and
validates only at the final hard budget. Full MLP/optimizer checkpoints are device-neutral and may
resume to a larger requested total step count. A `steps=0` run supplies the matched control.

On all 16 expanded-validation sequences, the untrained q=62.5%/K=512 control has KL 1.235, top-1
0.460, NLL +1.202, hidden L2 0.508, and local L2 0.702. Eight progressive CPU steps change these to
1.254, 0.481, +1.211, 0.510, and 0.700. The run passes evidence and traffic checks but not causal
quality; because the metrics move in opposing directions, it does not trigger a longer run.

`engram evaluate-native-gate-residual` fits a continuous low-rank correction to the difference
between partial-gate log utility and exact contribution log utility. Predictor parameters count as
traffic. On 512 calibration states per layer, rank 16/blend 0.8 reaches local L2 0.338 and exact
top-512 recall 0.643 at 44.39% of dense traffic, passing the declared 10% local-improvement screen.
The evaluator writes provenance-bound per-layer tensors; `train-native-gate-e2e
--utility-residual ...` consumes them in the actual hard sparse path.

On the same 16-sequence causal set, that untrained residual path reaches KL 0.629, top-1 0.599,
NLL +0.583, hidden L2 0.363, and local L2 0.625. These are large improvements over the native-gate
control but remain outside the final thresholds. Eight matched progressive steps produce
0.640/0.605/+0.604/0.363/0.626, rejecting longer training with the unchanged objective. The next
screen must refit residuals on sparse-student states and then repeat this exact held-out gate.

That refit has now been screened and rejected: on sequence-disjoint sparse-student trajectories it
changes same-state local L2 only from 0.35117 to 0.34983. The larger causal local metric also
contains accumulated state drift, but this controlled comparison shows that state-distribution
mismatch is not the main selector limitation. No causal evaluation of the refitted artifact is
justified.

A development-only q=43.75%/K=640/rank-23 composition uses 44.25% projected traffic and passes its
local screen, but worsens KL and NLL relative to K=512 while remaining outside every final quality
threshold. The original K cap is not revised. Together with the failing full-information K=512
oracle, this closes the frozen-basis routing branch. Further Milestone 2 evaluation must concern a
co-adapted structured/width-pruned MLP trained on a materially larger corpus.

### Fixed-width co-adapted student

The next controlled experiment replaces every 1,536-wide SmolLM2 SwiGLU with a trainable
672-wide layer. This is a router-free, contiguous representation at 43.75% of dense MLP weight
traffic. The student freezes non-MLP transformer components and trains local-MLP, hidden-state,
and logit-distillation losses. A deterministic corpus builder round-robins 2,048 sequences
(258,899 token positions) across 129 repository prose/code files. Exact token-sequence hashes
confirm no overlap with the expanded validation set.

Parameter-only checkpoint transfer from the 128-sequence pilot prevents optimizer/history leakage
into the new corpus. All held-out metrics improve through a complete 2,048-step epoch, but remain
far outside the gate:

| Training state | KL | Top-1 | NLL delta | Hidden rel-L2 | Local MLP rel-L2 |
|---|---:|---:|---:|---:|---:|
| 128-sequence pilot, 128 steps | 1.5499 | 0.4175 | +1.5254 | 0.4896 | 0.7636 |
| Expanded corpus, 512 steps | 1.3445 | 0.4460 | +1.2723 | 0.4537 | 0.7310 |
| Expanded corpus, 1,024 steps | 1.2660 | 0.4521 | +1.1604 | 0.4418 | 0.7189 |
| Expanded corpus, 2,048 steps | 1.1773 | 0.4745 | +1.0553 | 0.4260 | 0.7053 |
| Required gate | <=0.05 | >=0.90 | <=+0.05 | <=0.10 | diagnostic |

This rejects additional blind epochs of the same fixed-width configuration. Before another causal
run, Engram should fit compact layers on a larger sample of cached teacher boundaries and measure
the attainable per-layer approximation ceiling. If that ceiling remains poor, the next design must
spend the same byte budget on a more expressive structured basis rather than more optimization of
width 672.

The follow-up ceiling screen uses MLP-only traces: 4,096 training boundaries sampled from 256
full-context sequences and 446 validation boundaries from a separate 16-sequence split. Layers
0, 7, 14, 21, and 29 are initialized from the full-epoch checkpoint and independently trained for
2,048 cached-boundary steps. The declared screen requires at least 10% mean improvement and final
mean relative L2 no greater than 0.15.

| Layer | Initial rel-L2 | Fitted rel-L2 | Improvement |
|---:|---:|---:|---:|
| 0 | 0.3059 | 0.2221 | 27.4% |
| 7 | 0.5476 | 0.5049 | 7.8% |
| 14 | 0.4905 | 0.4558 | 7.1% |
| 21 | 0.4805 | 0.4497 | 6.4% |
| 29 | 0.1007 | 0.0961 | 4.6% |
| Mean | 0.3851 | 0.3457 | 10.2% |

The relative-improvement check passes, but the absolute ceiling fails by 0.1957. Uniform width 672
is rejected before another causal run. The next architecture should test layer-adaptive capacity or
a more expressive structured basis under an aggregate, rather than per-layer, 45% traffic budget.

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
