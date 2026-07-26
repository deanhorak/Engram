# Evaluation

## Current combined-gate decision

As of 2026-07-26, no dense-Llama conversion passes the causal quality
thresholds and the complete physical cold-traffic threshold together. The
older dense-SmolLM predictor-free DIP experiment passed quality but reached
83.33% cache-line traffic and was slower than dense. The
serialized mild-width compact-Q4 student reaches 44.9334% traffic but fails
quality after 3,000,093 training positions. The latest 1M-prototype
output-memory experiment is layer-local only and fails its predeclared
progression screen. Five later recurrent/low-bit representations also fit the
traffic policy but miss the 0.20 layer-local ceiling; the strongest trained
point reaches 0.308254. The subsequent all-layer budget-native
grouped-ternary artifact reaches 43.1353% traffic, but after 1,014,225 training
positions still has KL 2.2844, top-1 0.3198, NLL delta +2.2770, and
final-hidden relative L2 0.6036. It fails its frozen pre-3M progression rule.

A separate low-bit-native source track first passed the causal quality and
cold-byte checks while executing every record. Its direct CPU kernel
memory-maps the 318,924,544-byte base-3 phase artifact, materializes no dense
weights, and schedules 40.0527% of dense ideal-Q4 cold bytes. On the frozen
8-sequence/256-position corpus it measures KL 0.00371, top-1 0.96094, NLL
delta +0.00224, and final-hidden relative L2 0.04678 while executing every MLP
record. See the
[direct-kernel result](../reports/semantic_gate_native_bitnet_2026-07-24/summary.md).

The subsequent exact-membership oracle established that the BitNet source does
have a viable semantic subset. A development-only layer sweep chose a 15–35%
schedule averaging 24.8375% selected records. On the frozen
8-sequence/256-position protocol it passes with KL 0.02543, top-1 0.94531, NLL
delta +0.02386, and final-hidden relative L2 0.09205. Fixed 25% missed only the
hidden-state limit (0.10448). These are oracle results: dense gate/up
coefficients still determine membership, so neither candidate recall nor
practical traffic was claimed. See the
[oracle report](../reports/native_bitnet_oracle_2026-07-26/summary.md).

That practical route now exists. A source-bound v2 coordinate index and
CPU-only native DIP kernel use the largest 1,920/2,560 live-BF16 input
coordinates, per-layer candidate and maximum-K schedules, token-adaptive
nonzero selection, candidate-ratio RMS estimation except for a layer-9
eight-record audit, and selected down-row reads. On the declared
8-sequence/256-position development corpus it passes:

| Measure | Result | Requirement |
|---|---:|---:|
| Mean KL | 0.0044706883 | <= 0.05 |
| Top-1 agreement | 0.94921875 | >= 0.90 |
| NLL delta | +0.0013608933 | <= +0.05 |
| Final-hidden relative L2 | 0.0498965010 | <= 0.10 |
| Mean active fraction | 0.2008071899 | <= 0.25 |
| Modeled physical cold traffic | 0.4096389557 | <= 0.45 |
| Global micro candidate recall | 0.9995917258 | >= 0.95 |
| Worst-layer mean recall | 0.9939353303 | >= 0.95 |

The timed sparse pass makes no dense full-record calls. Dense teacher
membership is computed only in a separate untimed diagnostic pass against a
fixed per-layer top-K schedule. Six rows in each of all 30 layers have
bit-exact Python/native coordinate, candidate, selected-record, selected-count,
and BF16-output parity. The frozen policy authorizes one sealed final
confirmation; therefore this is a development-gate pass, not a Milestone 2
pass. Its sparse end-to-end elapsed time is 1.1565x dense (15.65% slower), and
the traffic ratio is modeled from the v2 serialized layout rather than
measured with DRAM counters.

See [Project status](status.md) and the
[machine-readable snapshot](../reports/semantic_gate_status_2026-07-23/summary.json).
Exact budget-edge screen results are in the
[low-bit/recurrent summary](../reports/semantic_gate_lowbit_2026-07-23/summary.json).
The exact hard-forward training protocol and scale-up decision are in the
[budget-native summary](../reports/semantic_gate_budget_native_2026-07-23/summary.json).
The sections below define the individual experiments and preserve their
historical evidence.

## Native BitNet repack and parity protocol

`engram audit-native-bitnet` downloads configuration only and accepts the
narrow `bitnet_offline_autobitlinear_v1` contract. It requires BitNet causal
architecture, ReLU-squared gating, offline `AutoBitLinear` storage, and
dimensions compatible with the official four-trits-per-byte layout. It does
not add `bitnet` to the generic SiLU/SwiGLU compiler. Native-training
provenance is accepted only for the pinned official source attestation; a
local config with matching fields remains unverified.

`engram repack-native-bitnet` then verifies the pinned safetensors SHA-256,
rejects invalid two-bit code `3`, writes cache-aligned
five-trits-per-byte gate/up/gain/down phase streams, reloads the complete
artifact, and compares every reconstructed logical value. Each channel
remains O(1)-addressable as a logical record. The physical phase layout avoids
the compulsory rereads an interleaved record would incur around the shared
RMS normalization. Traffic is reported against both the unchanged dense-Q4
denominator and the actual Hugging Face native payload. The latter is never
used to weaken the 45% threshold.

`engram evaluate-native-bitnet-parity` preserves native per-token activation
quantization, ReLU-squared gating, `ffn_sub_norm`, and BF16 projection scales.
Its dense decode exists only as a correctness oracle. Progression to the
combined gate produced:

| Check | Current result | Required next result |
|---|---:|---:|
| Logical reconstruction | exact | exact |
| Selected-layer BF16 parity | exact | exact |
| Bounded all-layer causal parity | exact | exact |
| Complete serialized/modelled phase traffic | 40.0527% | at most 45% |
| Direct packed CPU execution | implemented; zero dense-weight bytes | parity-correct |
| Evidence | 8 sequences / 256 positions | at least 8 sequences / 256 positions |
| Causal quality | KL 0.00371; top-1 0.96094; NLL +0.00224; hidden L2 0.04678 | pass frozen thresholds |
| Packed-kernel latency and traffic | 9.737 s summed MLP time; exact 40.0527% scheduled cold bytes | report against dense baseline |

`engram evaluate-native-bitnet-kernel` is the full-record systems command. It verifies
the pinned source and artifact hashes, loads the pinned tokenizer with its
regex compatibility fix, selects the frozen records deterministically,
checks layers 0/14/29 against the dense oracle, substitutes the direct kernel
into all 30 transformer layers, and writes every per-layer byte/time counter.
Official BF16 layer outputs are not bit-identical because PyTorch GEMM and the
stream kernel reduce in different orders; their maximum checked relative L2
is 0.00982. The causal thresholds, rather than bit identity, determine the
formal outcome.

## Native BitNet practical-routing protocol

The practical DIP evaluator is stricter than the earlier trace screens:

1. It reloads the source-bound base record artifact and v2 coordinate index.
2. It substitutes the native CPU DIP kernel into all 30 MLPs at live BF16
   boundaries. No dense MLP fallback is permitted.
3. The timed sparse pass records selected counts and cache-line traffic but
   does not call the dense teacher.
4. An untimed debug pass exposes the exact route and evaluates recall against
   a frozen, router-independent dense-teacher top-K schedule on the same actual
   sparse causal states.
5. Global micro recall and every layer's mean recall must each reach 0.95.
   Adaptive K is not used as the recall denominator.
6. Quality, mean activity, and physical traffic must pass on at least eight
   unique sequences and 256 positions.

The frozen route uses `q=1920`, `minK=346`, and energy target 1.0 in all
layers. `C` and `Kmax` are layer-specific and authenticated in the
[policy manifest](../reports/native_bitnet_m2_2026-07-26/frozen_dip_policy.json).
Energy target 1.0 means the token's target K is its number of positive exact
candidate utilities, clipped to `[346,Kmax]`; it is not a fixed-density
selection. Candidate-ratio RMS estimates the unseen tail by the ratio between
exact and proxy squared energy inside the candidate set. Layer 9 instead uses
corrected proxy energy and eight top-proxy-raw-square audit candidates inside
the same fixed `C=4480`.

The policy manifest binds the base artifact, coordinate index, native
libraries, package manifest, tokenizer, protocol, development report, and
parity report by SHA-256. The original float16 trace proposal cannot approve
the route because it does not reproduce live BF16 boundaries or native
accumulation. The v2 coordinate index, not that proposal, is the executable
policy authority.

After this development pass, no configuration field may change. The final
runner may open the independent 8-sequence/256-position holdout once. A policy
change after opening requires a new holdout. The holdout is plaintext in the
repository, so this protection is procedural and honor-system-based rather
than cryptographic; the fail-closed runner and committed hashes make misuse
auditable.

## Budget-edge local progression protocol

The final bounded representation screens are deliberately cheaper than a
formal all-layer intervention. They use sequence-disjoint development-role
teacher boundaries at representative layer 14 and require:

| Check | Threshold |
|---|---:|
| Complete modeled cold MLP traffic | at most 45% of dense ideal Q4 |
| Mean layer-local relative L2 | at most 0.20 |
| Formal or external data opened before a local pass | no |

Initialization guards may stop a representation before training. Recurrent
cache reuse must additionally be demonstrated in a native benchmark before
its modeled byte result can count as a physical systems pass. These screens
can reject an arm but cannot qualify one for compilation; a local pass would
only authorize the existing all-layer causal gate on a serialized and
independently reloaded artifact.

## Budget-native causal progression protocol

`engram train-budget-native-ternary` keeps the deployable representation in
the student forward path. It supports a global or deepest-layer-first
continual transition, hard-forward straight-through quantization, direct
hidden/logit distillation, optional CKA and teacher-top-1 losses, co-adaptation
of already-resident backbone tensors, fresh-record offsets, device-neutral
checkpoint/resume, and initialization from an earlier checkpoint when the
objective or trainable set changes.

Every scored result forces all 30 MLPs to hard ternary. The MLP artifact is
serialized, strictly reloaded, decoded, and installed before validation.
Co-adapted attention, normalization, embedding, or head tensors are separately
written as safetensors and reloaded. The complete MLP file size must exactly
equal the byte model. Training hardware is not part of the inference claim:
the one-million-position run used the local RTX 3050 as an accelerator, while
its checkpoint and artifact remain CPU compatible.

Short objective screens used fresh record ranges and predeclared improvement
rules. The promoted one-million-position rung used 8,192 records and had to
close at least half of every remaining formal gap from the preceding
head-coadaptation checkpoint:

| Metric | Baseline | Required after 1M | Measured | Gap closed |
|---|---:|---:|---:|---:|
| KL | 6.14955 | ≤3.09977 | 2.28436 | 63.37% |
| Top-1 | 0.05499 | ≥0.47749 | 0.31976 | 31.33% |
| NLL delta | +6.03256 | ≤+3.04128 | +2.27704 | 62.77% |
| Hidden L2 | 0.91613 | ≤0.50807 | 0.60361 | 38.29% |

Because top-1 and hidden state fail, this configuration is not eligible for a
3M or 10M continuation. The rule prevents strong KL/NLL movement from being
mistaken for broad semantic recovery.

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

Exact activation-sparse screens use a separate accounting model because they
do not predict source-record membership. For CATS/FATReLU gating, the runtime
reads the complete gate projection and only active up/down records; ideal
traffic is `(1 + 2a) / 3`. For Q-Sparse-style execution, the top-magnitude
input coordinates are already resident and select columns of both gate and
up, while a second exact top-K selects the down input; ideal traffic is
`(2q + k) / 3`. Candidate recall is not applicable to either mechanism.
Thresholds, where used, are fitted on calibration traces only. The
development boundary screen still requires mean relative L2 at most 0.18 and
traffic at most 45% before permitting an all-layer causal run. Metadata,
indices, scales, alignment, and cache-line amplification must be added by a
serialized artifact before a formal systems pass.

The later whole-model campaign executes exact hard top-K at all 30 MLPs and
uses a training-only identity STE. CUDA is permitted only as a training
accelerator. Candidate tensors are saved device-neutral and independently
reloaded for a CPU hard-path execution check; this check is necessary but does
not turn the float training artifact into a formal packed-Q4 runtime.

Configuration selection uses 16 development sequences. Unbiased development
evaluation uses an authenticated 128-sequence/15,559-position tail shard from
the pinned pretraining-mixture corpus, disjoint from its 81,647-record training
prefix by exact token-sequence hash. Confirmation remains sealed.

The selected causal per-layer schedule keeps `q <= 360/576` and `K <= 512` and
uses exactly 45% ideal traffic before metadata. Its unseen baseline is KL
0.4574, top-1 0.6694, NLL delta +0.4744, and final-hidden relative L2 0.3281.
The best verified attention/normalization co-adaptation reaches KL 0.4517,
top-1 0.6714, NLL +0.4585, and hidden L2 0.3272. It therefore fails every
semantic threshold. Label-only continuation, token-adaptive concentration
thresholds, and a rank-24 correction charged against the same traffic budget
are rejected. See the
[whole-model campaign report](../reports/semantic_gate_fully_sparse_2026-07-24/summary.md).

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

### Earlier dense-SmolLM DIP experiment

This historical experiment replaced learned membership prediction with a
predictor-free, DIP-inspired selector. The published DIP method motivates
top-magnitude input pruning and partial activation
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

The selected development point was 75%/896: its small traffic premium over the cheapest pass
buys better margin on recall, top-1 agreement, and hidden drift. That configuration was then frozen
and run once on a new 16-sequence corpus: 1,184 input states, 1,168 next-token positions, and zero
exact token-sequence overlap with the configuration-selection corpus. The confirmation arm passes
with recall 0.9897, score-mass recall 0.9961, KL 0.0286, top-1 agreement 0.9101, NLL delta +0.0326,
and final-hidden relative L2 0.0905. The machine-readable decision was
`eligible_for_selector_serialization`. The later packed cache-aware benchmark
raised traffic to 83.33% and ran slower than dense, so this dense-source arm
did not pass the systems gate. It must not be confused with the newer
native-BitNet DIP policy above.

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

## Shared-controller distillation protocol

Controller evaluation is staged so exact teacher signals cannot be confused
with deployable compiled signals:

1. Capture the packaged BitNet teacher on CPU. Each checksummed shard records
   token identity, token position, token embedding, all 31 residual
   boundaries, 30 MLP outputs, and 30 attention outputs.
2. Normalize every residual state to unit per-token RMS. Divide each operator
   output by the RMS of the residual entering its stage. Capture fails rather
   than clipping if the normalized values are non-finite or exceed FP16.
3. Train the shared factorized controller on CUDA with intermediate hidden,
   transition-delta, cosine, and terminal rollout losses. Teacher forcing is
   held at 100%, annealed, then removed for the final 20% of steps.
4. Keep training and validation traces on different dataset hashes. Reusing
   the same trace or dataset hash is rejected when protected validation is
   requested.
5. Serialize FP32 `.npy` factors, load them through the independent NumPy CPU
   implementation, and compare a complete 30-stage rollout with Torch.
6. Report teacher versus compiled-operator inputs explicitly. Results using
   exact teacher MLP/attention outputs may open the next development rung but
   cannot qualify transformer-free generation.

The next protected development run uses 128 training and 64 validation
positions across eight and four sequences. Its rank-128 artifact reduces
terminal validation normalized MSE from 1.998608 to 0.245010 and cosine loss
from 0.973363 to 0.333417. Serialized CPU parity passes at 7.45e-6 maximum
absolute error. A fully self-fed 500-step continuation regresses terminal
validation error to 0.260050 despite improving its training error, so the
pre-continuation artifact is retained. These numbers justify broader
trajectory coverage; they do not open compiled-operator substitution.

The next frozen scale rung uses 1,024 training and 256 protected validation
positions. A fresh 1,000-step CUDA fit reaches terminal normalized MSE
0.159440, cosine loss 0.272803 averaged across stages, and total loss 0.931534.
It fails the fixed 0.0225 substitution gate.

A controlled rank-4 stage input adapter changes terminal normalized MSE only
to 0.157431. The passing architecture instead preserves the teacher's known
residual algebra: current state plus semantic output plus episodic output,
then RMS normalization. With the factorized correction disabled, schema v3
reaches protected terminal normalized MSE 0.000020801 and mean hidden
normalized MSE 0.000017685. Independent NumPy reload matches Torch within
5.72e-6. This passes the fixed controller-only gate. The semantic outputs
already come from the packaged CPU MLP kernel, while attention remains dense;
native bounded-attention substitution is the next evaluation boundary.

The required stagewise diagnostic evaluates the self-fed state after every
controller cycle against the corresponding RMS-normalized teacher boundary.
For the 1,024-position artifact, NMSE is 1.077929 at stage 1, 0.679043 at
stage 10, 0.419096 at stage 20, and 0.159440 at stage 30. Declining error
shows that exact later teacher operator outputs are correcting an initially
poor transition; it is not evidence that the recurrence itself becomes more
accurate. The rank-4 input-adapter result confirms that this learned transition
should not be promoted. Compiled-operator substitution is now open only under
the schema-v3 exact residual controller; the following frozen experiment
measures that compiled-input boundary.

The compiled-input result now exists. Controller traces were produced by
`NativeBitNetRuntime`, so their semantic outputs already came from the direct
packed CPU MLP kernel. The frozen joint evaluator replaces dense attention
with native W16/C8/K4/S2 streaming attention, captures both compiled operator
outputs, replays them through schema v3 without decoder residual scaffolding,
and applies the package final norm/head.

On the unchanged eight-sequence, 256-position confirmation split at offset 8,
controller replay versus the dense-attention package baseline reaches KL
0.011125, top-1 agreement 0.957031, NLL delta -0.008285, and final hidden
relative L2 0.075893. Replay versus the compiled candidate reaches hidden
relative L2 0.006810 and terminal trajectory normalized MSE 0.000026666.
Every quality and sample-size check passes. This opens direct incremental
controller dispatch; it does not yet claim generation without decoder-layer
operator capture.

Direct incremental controller dispatch is now measured. The candidate invokes
stage normalization, native bounded attention, native packed MLP, and the
schema-v3 controller explicitly; it never calls a decoder layer. Absolute
position IDs advance through prefill and one-token decode calls, and each
native attention layer retains its bounded cache.

The fixed eight-prompt suite generates four greedy tokens per prompt. All 32
tokens exactly match the bounded decoder-scaffold reference, all eight prompts
have exact sequence parity, every reported attention position count equals
prompt length plus decoded inputs, and decoder-layer forward calls are zero.
The predeclared 90% token and 75% exact-prompt thresholds therefore pass at
100% each. This is an incremental Python/Torch-shell result; native controller
serialization and a C++ residual/RMS loop remain.

## Native-BitNet package and Milestone 3 attention evidence

The native-BitNet package runtime has exact output parity with the
source-backed direct-kernel model on the fixed prompt: final hidden states and
logits match bit-for-bit. Two-token greedy generation yields tokens
`[12366, 13]` (` Paris.`), invokes the direct packed MLP 60 times, and never
loads source MLP tensors. The package report records the source revision,
artifact hash, non-MLP tensor count, package inventory, and runtime metrics.

Attention development rejected an all-layer 16-token local replacement (KL
0.2031, top-1 0.8594) and normalized recurrent attention (KL 2.6883, top-1
0.1875). The promoted hybrid performs one causal softmax over the local window
and the four exact best older keys. The frozen confirmation uses records 8–15,
disjoint from the records used for operator selection:

| Check | Threshold | Frozen result |
|---|---:|---:|
| KL | <= 0.05 | 0.002494 |
| Teacher top-1 agreement | >= 0.90 | 0.996094 |
| NLL delta | <= +0.05 | +0.007099 |
| Final-hidden relative L2 | <= 0.10 | 0.043498 |
| Evidence | >= 8 sequences / 256 positions | 8 / 256 |

That exact result passed semantic progression only because selection scanned
the complete older history. Follow-up random sign-LSH reached only
58.8–65.6% exact top-k recall. Exact bounding-box and centroid-radius page
indexes preserved recall but opened about 94% of pages and exceeded dense
logical traffic after metadata.

The promoted bounded streaming cache uses W=16, eight retained old keys (two
sinks and six online heavy hitters), and exact-reranks four values. On the
unchanged frozen records 8–15 it reaches KL 0.01409, top-1 0.94141, NLL delta
−0.00613, and hidden L2 0.08559. All evidence and semantic checks pass. At 33
tokens its modeled logical traffic is 93.34% of dense, but old-context storage
and reads no longer grow with sequence length. This advances Milestone 3 to
native implementation and long-context traffic validation; it does not claim
native latency or measured DRAM reduction.
