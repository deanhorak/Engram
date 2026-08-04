# Engram milestone and gate report

Status date: 2026-08-04

## Transformer-free controller replay boundary

The exact schema-v3 controller now has a standalone CPU runtime and CLI:
`engram evaluate-controller-only` loads only the serialized controller and
operator streams from a checksummed trajectory trace. It does not construct a
Transformers model, import decoder layers, or read MLP/attention weights. The
runtime rejects nonzero factorized corrections unless an explicit evaluator
override is supplied.

On the sequence-disjoint 16-sequence validation trace (256 token records and
30 stages), the exact operator-residual artifact replays with terminal
normalized MSE **0.0000208009** against the stored teacher boundaries, below
the 0.0225 controller threshold. The report records zero decoder-layer calls,
CPU-only execution, and the trace-manifest hash:
`reports/controller_only_replay_2026-08-03/validation.json` (SHA-256
`212d894b86de3b2e4e28481aa65091f276c74193b3031d53a4e9baeb4615ae4f`). The
serialized controller directory is bound by SHA-256
`51eb056d5755a69a5aa9a99336877e12a90531963abf4ef8b49a7a4a51716ee3`.

This closes the layer-free **state-transition** boundary only. The operator
streams are still supplied by a captured/compiled provider, so it is not yet a
layer-free end-to-end generator and does not promote a learned nonzero
correction.

### Stage-local causal-memory follow-up (2026-08-03)

The first token-level causal provider used one shared K/V memory for every
controller stage.  That is not structurally faithful: each stage consumes a
different hidden state and has a different operator boundary.  The follow-up
`stage_causal_attention_pca` provider therefore keeps one compact key/value
prefix per stage, attends over that prefix, and emits a residual before
recording the current stage state.  It is serialized with checksummed NumPy
arrays and runs through the existing CPU-only stateful evaluator.

The protected 8-sequence/128-record training and 16-sequence/256-record
validation split was held fixed.  Twenty free-running steps after ten
teacher-forced steps reduced terminal normalized MSE from **0.1767018** to
**0.1714502** for a rank-64 latent residual.  Replacing the base with the
full-corpus rank-256 provider reached **0.1692925**, but this is still 7.52×
above the fixed **0.0225** gate.  Direct 2×hidden-size residual heads were
also screened: the 10-step arm reached **0.1817167** and the lower-rate
20-step arm **0.1770538**, showing overfit/instability rather than a route to
the gate.  No Transformers model or decoder layer was loaded in any arm.

The authenticated screen record is
`reports/controller_provider_pca_2026-08-03/stage_causal_attention_screens.json`.
The stage-local latent artifact is useful implementation progress, but all
learned-provider promotion remains blocked.  More isolated cache-shape, rank,
or teacher-forcing sweeps are closed.  A joint arm that unfroze only step
scale, stage embeddings, and low-rank controller adapters reached **0.1714471**,
effectively unchanged.  Unfreezing every controller tensor reaches
**0.1714721**, also a null result; the next experiment must use a materially
different model or expand the independent causal corpus.  Repeating the
latent stage-local fit with all 64 training sequences reaches **0.1707001**,
which is a modest data effect but still fails decisively; the rank-256
full-corpus arm remains best at **0.1692925**.  A larger nonlinear rank-256
residual (hidden width 256, stage width 32) reaches **0.1666128** after 80
free-running steps, still 7.4× above the threshold.  This closes the current
provider-only capacity family; evidence is recorded in
`reports/controller_provider_pca_2026-08-03/nonlinear_rank256_h256_full.json`.

The trace contract now has an opt-in causal extension: `--causal-top-k` stores
teacher top-k logits and next-token IDs, while `distill-controller` can consume
those fields with a frozen vocabulary matrix and final RMSNorm vector. A
synthetic CPU smoke confirmed that the objective backpropagates and the
serialized controller still reloads with parity. No real BitNet causal trace
was used for promotion at that point, so the smoke was plumbing evidence rather
than a semantic gate result.

A real CPU smoke now exercises the complete path: two sequence-disjoint
1-sequence/4-position BitNet traces with top-8 logits were captured (about 41 s
per sequence), and a rank-2 one-step controller fit consumed the frozen
vocabulary head. The causal objective ran and CPU reload parity passed, but the
tiny arm failed the fixed controller gate (validation terminal normalized MSE
**1.96610**, top-k KL **10.11891**). It is retained as execution evidence only:
`reports/controller_causal_smoke_2026-08-03/training_report.json` (SHA-256
`c42501f4ae6896dba3704a66fedbf6b9dd88468b073592f3d360eba197b938b6`).

The first evidence-sized causal fit is now also complete. It uses 8 training
sequences/128 records, 16 held-out sequences/256 positions, top-32 teacher
logits, and 500 CPU steps at rank 128. CPU reload parity passes, but free-run
validation fails badly: terminal normalized MSE **0.2624663**, hidden MSE
**0.7221664**, and top-k KL **8.7518297**. The result rejects this factorized
controller configuration for promotion; more epochs of the same objective are
not justified. Report:
`reports/controller_causal_2026-08-03/rank128_8x16_500cpu.json` (SHA-256
`47a1c94ae4797c9e0653f1b88d129620db691f3ad3306ca1ed85f2c234267a32`).
A rank-256/adapter-8 arm at the same 500-step budget is no better: terminal
normalized MSE **0.2710301**, hidden MSE **0.7289818**, and top-k KL
**8.9769297**. Capacity alone is therefore rejected; the next change must
co-adapt or replace the operator-stream provider.

### Operator-provider boundary (2026-08-03)

The layer-free seam is now explicit in `engram.runtime.operator_stream` and
`engram evaluate-controller-provider`. A provider receives only the current
controller state, token embedding, and stage, and returns semantic and
episodic vectors; the controller runtime performs no Transformer imports or
decoder-layer calls. The trace-backed provider is labeled replay-only. A
serialized PCA/ridge provider is conditioned on state and token embedding and
contains no source-model tensors. `engram fit-operator-provider` creates it
with a checksummed manifest and an explicit target (`streams` or the
co-adapted `combined_delta` arm).

The first independent causal replay uses 128 training records, 256 held-out
records, rank-16 output factors, and the exact operator-residual controller.
CPU reload and layer-free execution pass structurally, but terminal
normalized MSE is **0.2536094** (threshold 0.0225), hidden MSE **0.8770986**,
and the quality gate fails. Increasing the provider output rank to 64 and 128
does not recover the threshold (terminal MSE **0.2404522** and **0.2478832**);
the combined-delta rank-16 arm is worse (**0.5889627**). These are provider
change experiments, not promotion claims. The canonical rank-16 report is
`reports/controller_provider_pca_2026-08-03/rank16_train8x16_validation16x16.json`
(provider SHA-256 `3bc7c9afd9996f26673e408328b9d35e77986f66fdd6492025967dacaeb0a3a9`).
Twenty steps of joint free-running projection adaptation reduce held-out
terminal MSE to **0.1970640** from **0.2536094**, while the training terminal
reaches **0.1501944**; the gate still fails. The retained adaptation report is
`reports/controller_provider_pca_2026-08-03/joint20_train8x16_validation16x16.json`
(SHA-256 `7edd36f97fbbeb196869e3e5b5d4cc435c94bbf3c475c036c86790371844b1d7`).
Resuming from that checkpoint for another 20 steps with a different seed
regresses slightly to **0.1997519**, so the adaptation is not a monotonic path
to promotion. That retained result is
`reports/controller_provider_pca_2026-08-03/joint20_resume_seed101.json`
(SHA-256 `ec461fe948a059d4d9926e6c5a65e10c1b173a06310432a857e2bb29e0b47cbb`).
The next M4 attempt must add temporal/context features or a stronger jointly
trained provider/controller model; another isolated rank sweep is not
justified.

### Direct transition hypothesis (2026-08-04)

A materially different CPU pilot replaced the fixed operator-stream basis with
a stage-conditioned residual MLP that predicts the complete next controller
state directly from the current state and token embedding.  This tests whether
the PCA output seam, rather than the controller dynamics, is the limiting
factor.  After 100 bounded updates on the protected training trace, the
held-out terminal normalized MSE was **1.0855621** (initial **1.9863384**),
still 48× above the **0.0225** threshold.  A wider 512-unit pilot was stopped
by the host after its 100-update checkpoint at **1.2892941**.  Neither run
produced a promotion artifact.  The direct-transition family is rejected until
a model with explicit sequence memory and a larger independent corpus is
available; this result does not alter the exact state-transition pass.

A scalar-gated residual transition was screened as a stability alternative.
After 60 CPU updates (30 teacher-forced), held-out terminal normalized MSE was
**1.1115413** from an initial **1.9863384**; it remains 49× above the fixed
threshold and produced no artifact.  This closes the current low-capacity
gated-transition family.  Evidence is recorded in
`reports/controller_provider_pca_2026-08-03/gated_transition_screen.json`.

### Stateful sequence-provider boundary (2026-08-03)

The runtime now exposes `run_sequence_provider`, which resets a provider once
per sequence batch, advances token context exactly once per token, and then
executes all depth stages against that persistent context. The
`RecurrentContextProvider` defines the CPU artifact contract for a compact
token-memory state, while `TraceSequenceOperatorStreamProvider` provides an
explicit replay-only reference implementation.

On the 16-sequence/256-position validation trace, sequence-preserving replay
matches the earlier flat replay at terminal normalized MSE **0.0000208009**,
with zero decoder-layer calls and CPU-only execution. This proves cache/state
advancement and sequence boundaries, not learned-provider quality: the
provider is backed by captured teacher streams and remains excluded from
promotion. Evidence:
`reports/controller_provider_pca_2026-08-03/sequence_trace_replay.json`
(SHA-256 `eec4ebda338cbd604bd7c47ac3ea64801da5457bfe573d3f39282ece379725fa`).

The replay boundary is now durable. A serialized `trace_sequence_replay`
provider records both sequence-shaped operator arrays with per-file hashes,
and `engram evaluate-controller-sequence` reloads it, reconstructs the
sample ordering, and runs the persistent reset/advance contract without a
Transformer model. The CLI report reproduces terminal normalized MSE
**0.0000208009**, with zero decoder-layer calls and provider SHA-256
`cdd5f27c503491d02083878d823804a9e76fb5917d8478201b1c4b2748237313`:
`reports/controller_provider_pca_2026-08-03/sequence_provider_cli_replay.json`.
The artifact is marked `learned: false`; this is a durable replay/package
boundary, not a learned-provider promotion.

The 64-sequence fit was also measured. Rank-16 state/token regression reaches
held-out terminal normalized MSE **0.2127623**, and 20-step free-running
projection adaptation reaches **0.2120011**, both failing the **0.0225**
causal gate. Previous-state context, nearest-neighbor retrieval, and residual
context correction screens regress to **0.4178558**, **0.3247504**, and
**0.4624460**; a shared nonlinear latent provider is worse at **2.4635784**
validation mean normalized MSE. The negative results are
preserved in `reports/controller_provider_pca_2026-08-03/context_and_retrieval_screens.json`.

The next provider architecture is a diagonal state-space recurrence with a
64-wide token memory and rank-16 stage heads. Starting from the linear
provider and running 80 free-running CPU optimization steps lowers held-out
terminal normalized MSE to **0.1926129** (training **0.1439981**). This is a
material but insufficient improvement over the **0.0225** gate. The
`StateSpaceOperatorStreamProvider` class now serializes the memory, decay,
projection, and output tensors with checksums, making longer training and
CPU-only replay reproducible. The screen remains unpromoted:
`reports/controller_provider_pca_2026-08-03/state_space_screen.json`.
The reproducible training entry point is `engram distill-state-space-provider`;
a one-step CPU smoke produced a checksummed reloadable artifact and recorded
zero decoder-layer calls. CUDA may accelerate distillation when available,
but the serialized inference path remains CPU-only.

The 64-sequence rank-16 regularization sweep selects λ=1: held-out terminal
normalized MSE is **0.1789347** (λ=0.1: **0.1841841**, λ=3: **0.1797703**,
λ=10: **0.1832849**). A full-width residual state-space adapter over that
provider preserves the λ=1 path when its correction is zero and reaches
**0.1777104** after 40 free-running CPU steps, with training error
**0.1343896**. This is the best learned-provider result so far, but it still
fails the **0.0225** causal gate. The resumable artifact is
`work/controller_provider_state_space_residual_train64_ridge1_40`; report:
`reports/controller_provider_pca_2026-08-03/residual_state_space_train64_ridge1_40.json`.

Keeping that λ=1 provider fixed, correction-only adaptation of the factorized
controller (`step_scale`, stage embeddings, low-rank adapters, and operator
scales) reaches held-out terminal normalized MSE **0.1759220** after 50 CPU
steps (training **0.1336495**). Jointly adapting provider projections is
worse (**0.1810400**) and is rejected. The command is
`engram adapt-controller-correction`; the resulting nonzero controller is
retained as evaluator-only evidence in
`work/controller_provider_ridge1_controller_correction50` and
`reports/controller_provider_pca_2026-08-03/controller_correction_ridge1_50.json`.
It does not pass the **0.0225** promotion gate.

The remaining rank-only hypothesis is now closed. A randomized stage-wise
rank-64 provider fit on the same 64-sequence training trace reaches held-out
terminal normalized MSE **0.1767018** (mean **0.7871890**, maximum stage
**1.5251367**). This is only a marginal improvement over the rank-16 λ=1
result (**0.1789347**) and remains 7.85 times above the **0.0225** gate. A
matched rank-64 provider trained on the normalized combined transition is
worse (terminal **0.5177052**), confirming that the teacher's
semantic/episodic decomposition should be kept. The screen used no
Transformers model or decoder-layer calls and is recorded in
`reports/controller_provider_pca_2026-08-03/high_rank_stream_screen.json`.
Output rank alone is not a defensible route to M4 promotion; the next attempt
must change the causal training signal or replace the provider/controller
architecture.

The exact-residual algebra was tested directly as well. A rank-16 provider
trained on the combined teacher residual (`semantic_output + episodic_output`)
with the episodic stream serialized as zero reaches held-out terminal
normalized MSE **0.1793060** (mean **0.8155644**, maximum stage **1.6025280**).
It is slightly worse than the separate-stream λ=1 provider (**0.1789347**),
so stream separation is not the dominant source of causal error. The
transformer-free CPU result is preserved in
`reports/controller_provider_pca_2026-08-03/combined_stream_rank16.json`.

A DAgger-style visited-state refit was then implemented as
`engram dagger-refit-operator-provider`. On the smaller 8-sequence training /
16-sequence validation split, two refits lower held-out terminal normalized
MSE from **0.2536107** to **0.1849852** (validation mean **0.6448557**), while
the training terminal falls from **0.1302454** to **0.1226217**. The gain
stops after the second refit and remains 8.22 times above the **0.0225** gate,
so the artifact is retained as development evidence only. It uses CPU-only
causal rollouts, no Transformers model, and zero decoder-layer calls. Report:
`reports/controller_provider_pca_2026-08-03/dagger_refit_train8x16_validation16x16.json`.
This closes the first visited-state correction attempt; a passing M4 provider
still requires a materially more expressive teacher signal or architecture.

The first durable nonlinear provider was then implemented. A shared SiLU MLP
with a learned stage embedding adds latent corrections to the rank-16 PCA
provider and is serialized as `nonlinear_residual_pca`. A 100-step CPU run on
8 training sequences improves the independent 16-sequence validation terminal
normalized MSE from **0.2536107** to **0.1966997**; reloading the serialized
artifact reproduces **0.1966993**. The improvement is real but remains 8.74
times above the **0.0225** gate. The provider loader, CLI, checksum-authenticated
artifact path, and zero-output parity test are now integrated. Evidence:
`reports/controller_provider_pca_2026-08-03/nonlinear_residual_train8x16_validation16x16.json`.

A larger hidden-128/stage-32 arm trained for 300 CPU steps reaches terminal
normalized MSE **0.2107506**, worse than the smaller 100-step arm. The
capacity/epoch increase is therefore rejected rather than promoted; its
controlled comparison is recorded in
`reports/controller_provider_pca_2026-08-03/nonlinear_capacity_screen.json`.

Combining the full-train randomized rank-64 provider with the nonlinear
residual gives the best learned-provider screen so far. A 100-step CPU
residual fit on the 8-sequence arm improves held-out terminal normalized MSE
from **0.1767018** to **0.1731798** (mean stage **0.6788608**). The gain is
small and the result remains 7.70 times above the **0.0225** gate. It is
preserved as a transformer-free, protected-validation development result in
`reports/controller_provider_pca_2026-08-03/nonlinear_highrank_screen.json`;
no learned provider is promoted.

Scheduled causal distillation is now exposed by the nonlinear-provider CLI.
One hundred teacher-forced steps followed by one hundred free-running steps
on the same high-rank base reduce held-out terminal normalized MSE to
**0.1717456**; serialized reload gives **0.1717457**. This is the best
learned-provider result so far, but remains 7.63 times above the **0.0225**
gate. The controlled result is recorded in
`reports/controller_provider_pca_2026-08-03/nonlinear_highrank_scheduled.json`.

Extending the free-running phase to 300 steps after the same 100-step
teacher-forced warm-up regresses to terminal normalized MSE **0.1740203**.
The shorter 100-step free-running checkpoint remains the selected research
point; longer optimization is closed as an overfitting/oscillation failure.
Evidence:
`reports/controller_provider_pca_2026-08-03/nonlinear_schedule_length_screen.json`.

Finally, a stateful nonlinear provider with a 64-wide token memory reset once
per sequence was trained on the same causal split. Averaged over all 16
validation positions, terminal normalized MSE changes only from **0.1765443**
to **0.1760611** after 100 CPU steps; the last-position error is **0.1965011**.
Persistent token memory is therefore not the missing factor at this capacity.
The negative result is preserved in
`reports/controller_provider_pca_2026-08-03/stateful_nonlinear_memory_screen.json`.

Adapting the controller correction tensors over the rank-64 provider was also
tested (100 CPU steps). Held-out terminal normalized MSE changes from
**0.1767018** to **0.1718648**, slightly worse than the scheduled nonlinear
provider (**0.1717456**). This controller-only compensator is rejected and
remains evaluator-only; evidence:
`reports/controller_provider_pca_2026-08-03/controller_correction_highrank_screen.json`.

Two additional causal screens closed the most plausible remaining capacity and
optimization loopholes. A 128-wide persistent-memory residual trained for 20
CPU steps changes held-out terminal MSE only from **0.1767018** to
**0.1760428**. A shared nonlinear residual trained on all 64 training
sequences with teacher-forcing probability decayed smoothly to zero reaches
**0.1749395**. Both use the independent 16-sequence validation trace and are
rejected; evidence:
`reports/controller_provider_pca_2026-08-03/state_space_residual_memory128_screen.json`
and
`reports/controller_provider_pca_2026-08-03/nonlinear_train64_decay.json`.
The teacher-forced training loss can be below the gate while free-running
validation remains above it, so this is a causal compounding failure rather
than a missing optimizer schedule.

Increasing the full-corpus provider output rank from 64 to 256 lowers held-out
terminal MSE from **0.1767018** to **0.1710317**, but remains **7.6x** above
the gate. This rules out rank-64 truncation as the sole cause; the protected
screen is preserved in
`reports/controller_provider_pca_2026-08-03/rank256_fullcorpus_screen.json`.

A direct state-space provider with 256 memory dimensions and rank-64 heads
was also screened for 20 CPU steps. It improves its own untrained validation
baseline from **0.5126387** to **0.2416596**, but is materially worse than the
rank-64 PCA provider and fails the gate. This closes the “residual adapter
placement” hypothesis for this capacity; evidence:
`reports/controller_provider_pca_2026-08-03/state_space_direct_memory256_screen.json`.

The context-aware replacement keeps compact keys and values for every prior
token and attends to that prefix with a stage-conditioned query. Its new
stateful evaluator and authenticated loader pass CPU reload parity, but a
20-step screen (key 16, value 32, query width 32) changes protected terminal
MSE only from **0.1767018** to **0.1758242**. The provider is therefore
retained as an implemented research boundary, not promoted; evidence:
`reports/controller_provider_pca_2026-08-03/causal_attention_provider_screen.json`.
Adding a 10-step teacher-forced warm-up before 10 free-running steps reaches
**0.1764556**, slightly worse than the all-free-running screen; that result is
preserved in
`reports/controller_provider_pca_2026-08-03/causal_attention_teacher_forcing_screen.json`.

The stage-local follow-up keeps a separate compact K/V prefix for each of the
30 controller stages.  This is the correct cache topology for stage-specific
hidden states and is implemented as the CPU-serializable
`stage_causal_attention_pca` provider.  On the same protected split, 20
free-running steps after 10 teacher-forced steps reduce terminal MSE from
**0.1767018** to **0.1714502** at rank 64; a full-rank-256 base reaches
**0.1692925**.  Both remain 7.52× above **0.0225**.  Direct hidden-size
residual heads reach **0.1817167** and **0.1770538** in two scheduled arms,
so they overfit rather than improve generalization.  The complete screen and
hashes are in
`reports/controller_provider_pca_2026-08-03/stage_causal_attention_screens.json`.
This closes provider-only cache/rank sweeps; joint controller/provider
training at the small-adapter capacity was screened (0.1714471), followed by
full-controller training (0.1714721), followed by the full 64-sequence latent
screen (0.1707001) and the larger nonlinear residual (0.1666128); none is
sufficient.  A materially different model or independent causal corpus is
required next.

## Native recurrent-controller implementation boundary

The native token runtime now has a direct implementation of the schema-v3
factorized recurrent correction. The C++ stage ABI validates and consumes the
shared projections, gate/up projection, stage embeddings, low-rank adapters,
and optional input adapters, then applies operator-residual scaling and the
per-token RMS transition. Native tests establish exact zero-correction
parity, a deterministic nonzero transition, and fail-closed handling of
missing tensors.

This does not yet promote a learned controller. The authenticated package
continues to require zero `step_scale` and uses exact operator residuals. An
explicit evaluator-only CLI override can run an unauthenticated nonzero
controller directory; an existing trained artifact completed a six-position,
30-stage CPU smoke generation through it. That run demonstrates native
execution only—no semantic quality, held-out generalization, or package
promotion claim. A trained nonzero artifact and an independently sealed
causal gate remain required for the layer-free Milestone 4 boundary.

The Python `ControllerDrivenBitNet` path now dispatches nonzero corrections
through the same native stage ABI. An eight-prompt, one-token development run
with zero operator scales and an existing nonzero controller reached 0.0%
token agreement and 0/8 exact prompts against the exact residual package;
cache positions and zero decoder-layer calls passed. Controller-stage work
averaged 12.50 seconds per prompt, and the first exact token `12366` became
`36306`. This validates the cross-language dispatch but also demonstrates why
the correction remains unpromoted: the run is a decisive quality failure, not
a held-out generalization gate. Report:
`reports/controller_native_recurrent_2026-08-03/development_8x1.json`.

## Executive result

The latest Milestone 3 attention-substitution boundary is now a passing
prospective causal gate. On the frozen eight-sequence/128-position corpus, the
native CPU runtime retains the full W128 local context while storing local keys
and values as per-vector symmetric INT8 with FP32 scales. The independently
rerun report passes every semantic band and structural check: overall KL
0.00417982, top-1 0.974609, target-NLL delta +0.000391, hidden L2 0.048147,
and 25.0% logical attention traffic versus dense. The final 96–127 band also
passes (KL 0.00301720, top-1 0.964844, hidden L2 0.043127). This is an
evaluator-only CPU compression result, not a protected Milestone 2 replay or a
claim that the ordinary W16 package is already replaced.

Protocol SHA-256: `953b83cead9e722e6228c5d79252ecaa0c0c8980343459c62f5567455d33bda7`.
Report SHA-256: `7cd55514efab021b2109f835310eb14aff64664e972fa6458266f20d1d17df80`.

An authenticated package-level benchmark over 128 teacher-forced context
tokens plus eight native steps measured 75.17 s for ordinary W16/FP32 and
82.06 s for W128/INT8. Counted attention reads fell from 45,834,829,824 to
28,105,506,816 bytes, but total latency rose 9.17% because the current scalar
dequantization path is not fused or vectorized. The result is therefore a
memory-traffic/quality pass, not an end-to-end speedup claim. Artifact:
`work/olmoe_q7/local_attention_package_benchmark_2026-08-03.json`, SHA-256
`e8d634a1e08ba12da01cc968e87a8d7b031fb510fd763ddd68a957e44e723bdb`.

The compiler/runtime boundary is now exercised as an authenticated W128/INT8
package, not only as an evaluator override.  Package manifest SHA-256 is
`0b370a5b47913cade8255056b835ecf6d55f3a5aaf183290808ad22aa3ab6e8f`.
On an identical eight-token prompt, W128/INT8 and the ordinary W16/FP32
package produce identical token IDs.  Attention reads fall from 31,457,280 to
7,864,320 bytes (75%), while elapsed time changes from 4.7367 s to 4.8412 s
(+2.21%).  Q7 MLP work dominates this short run, so this validates package
assembly, CPU execution, and quality parity but is not a general speedup
claim.  Evidence:
`reports/olmoe_q7/package_w128_int8_generation_2026-08-03.json`.

The rank-16 compressed query/key selector followed by native exact reranking
now passes both the eight-record train screen and the independently captured
eight-record development screen. The development result is a causal CPU
replay, not a feature-only estimate:

- candidate-pool and exact-rerank membership recall: 99.8192% mean, 100% p10;
- answer-token agreement against the unmasked native runtime: 100%;
- mean hidden-state relative L2: 0.013236;
- mean logit relative L2: 0.006520;
- mean answer NLL delta: −0.003980, maximum +0.000574.

The authenticated result is
`work/olmoe_q7/retrieval_episodic_development_replay_rank16_pool6.json`,
SHA-256
`0c5cb2273f63b930148c78070da68ae57bb821969a68e2a6a038ee7ac5d04bb6`.

The selector remains an evaluator-only artifact and is disabled by default in
ordinary packages. The separately authorized protected replay now passes all
eight records: 100% answer-token agreement, 0.009133 mean hidden relative L2,
0.004416 mean logit relative L2, −0.000460 mean answer NLL delta (maximum
+0.005879), and 99.8501% mean candidate/exact-rerank recall. Native counters
show mean logical attention reads falling from 710,664,576 to 702,166,784
bytes (1.1958%); the older-candidate scoring stage falls 7.3733%. This is a
real semantic and locality result, but not yet a convincing end-to-end
speedup.

The protected policy was also copied into an authenticated CPU-only opt-in
package and validated. A one-token `Hello` generation produced the same token
ID 13 (`,`) through the opt-in and ordinary packages. This proves package
assembly and native parity without silently changing the ordinary runtime
default.

The required long-context CPU scaling boundary is now measured on development
record 0.  The exact same native package was run unmasked and masked at 512
and 2,048 positions (the first 128 positions remain the authenticated
selector window; later positions repeat tokens with episodic directives
disabled).  Answer quality remains 100% top-1 agreement with mean hidden/logit
relative L2 0.008138/0.003919 and NLL delta −0.001520.  At 512 positions the
masked logical-read fraction is 0.997079 (3.04 versus 3.09 tokens/s); at 2,048
it is 0.999275 (3.083 versus 3.085 tokens/s).  Peak resident set was 6.28 GiB
and did not grow with the repeated context.  These results establish bounded
state and honest scaling, but do not claim a speedup: the selector saves a
fixed 8.55 MiB of logical attention reads while the rest of the full model
continues to run.  The report is
`work/olmoe_q7/retrieval_episodic_long_context_rank16_pool6.json`, SHA-256
`fa205bd2ab4c91de27170247e7669f44c9def8bccea22d94558f8caa4b26bf71`.

The follow-up frozen development rank/pool sweep confirms the operating point:
rank-16/pool-4 falls to 95.1389% exact membership recall at 50% of the older
slots, while rank-16/pool-6 reaches 99.8192% at 75%; rank-16/pool-8 is exact
but reads all eight slots. Increasing rank to 32 or 64 changes pool-6 recall
only to 99.8867% and 99.9367%, respectively, so the extra projection cost has
no current causal justification. This is a feature-only sweep (no native
intervention); its report is
`work/olmoe_q7/retrieval_episodic_development_pool_sweep_rank16_pool6.json`,
SHA-256
`b170625c9802019a62be0c88a64799bef35bc8246befc000637dcb75e864f0ce`.

The selected policy is now serialized as a disabled-by-default evaluator
artifact.  It contains the train-only per-layer/head rank-16 key PCA basis,
the rank-16/pool-6/top-4 geometry, source hashes, and validation report hashes;
it contains no protected data.  Loading the artifact and rebuilding all
development masks reproduces the previous cross-split masks bit-for-bit
(shape `[8, 128, 16, 16, 8]`).  The local artifact is
`work/olmoe_q7/selector_policy_rank16_pool6_2026-08-03/` with policy SHA-256
`6d22205b543fd2d2ef986dd1a60b3f517b95a1c1218f1fd834c5cb3a6f16f46b` and basis
SHA-256
`1897767a6b22801faad4267c22152b5029285091e53fc3f36078ad6aa849f813`.
The loader deliberately refuses any artifact that is not evaluator-only and
disabled by default; native package defaults remain unchanged.
The native package compiler can copy this artifact under `selector/` and bind
both files into the authenticated package manifest when explicitly requested;
omitting that option preserves the previous package layout.

That opt-in path has now been exercised end to end on development record 0:
the serialized policy generated the native masked replay, preserving 100%
answer top-1 agreement while reducing logical attention reads from 710,668,288
to 702,124,544 bytes.  The tiny-package compiler/validator also authenticated
the copied policy and basis under `selector/`.  This closes the implementation
boundary without silently enabling the policy in ordinary generation.

## Milestones

| Milestone | Status | Evidence and remaining work |
|---|---|---|
| 1. Repository, inspection, fixtures, teacher traces, exact MLP decomposition, oracle top-K, tests | Complete | Build system, source inspection, teacher traces, exact SwiGLU/MLP decomposition, oracle experiments, and regression reports are present. |
| 2. Semantic memory, practical routing, quantization, Python runtime, substituted-MLP evaluation | Protected promotion passed; opt-in only | Train-to-development causal replay, 512/2,048-position CPU scaling, frozen pool frontier, authenticated opt-in package generation, and the separately authorized protected rank-16/pool-6 replay pass. Protected aggregate: 100% top-1, hidden L2 0.009133, logit L2 0.004416, NLL delta −0.000460. The policy remains disabled by default and requires explicit package opt-in. |
| 3. Attention analysis, local/recurrent/retrieval heads, hybrid episodic memory, attention substitution | Sustained gate passed; authenticated opt-in package validated | W128 full-context local attention with per-vector INT8 K/V and FP32 scales passes all frozen bands at 25% logical attention traffic. An authenticated W128/INT8 package now reproduces W16 token IDs and 75% lower attention reads; the ordinary default remains W16, and broader corpora/performance tuning remain. |
| 4. Shared recurrent controller and layer-free Engram runtime | Partial; state-transition gate passed | The standalone CPU controller runtime replays exact semantic/episodic streams with zero decoder-layer calls and terminal normalized MSE 0.0000208009 on the held-out trajectory. Stage-local causal K/V memory, normalized-residual, prefix-context, and full-corpus nonlinear targets remain negative; the best original-corpus learned-provider terminal MSE is 0.1666128 versus the fixed 0.0225 promotion threshold. The new native-projection capture path completed an independent 8/8 auxiliary split, but rank-128 PCA (0.2438876) and nonlinear (0.2530605) arms also failed. Causal provider/controller promotion remains blocked. |
| 5. Vocabulary/transition/correction artifacts, compiler, validation and CLI | Partial/usable | Native package generation, mapped weights, evaluator recurrent-correction dispatch, validation, greedy generation, and chat CLI work. The source-independent DIP package validates cleanly with 18 authenticated files; authenticated nonzero-controller package promotion remains gated. |
| 6. Native C++ runtime, kernels, mapping, parity, generation, benchmarks | Partial/usable | CPU scalar/vector kernels, memory mapping, C ABI parity, native generation, and tests are operational. The full native CTest suite passes 20/20 outside the sandbox; authenticated DIP package runs generate four tokens in 33.16 s and eight tokens in 50.43 s through the native CPU token-step runtime at about 1.30 GiB peak RSS. Python provider artifacts reload read-only via NumPy memmap. AVX2 tuning, lower-memory packaging, and an apples-to-apples `llama.cpp` comparison remain. |
| 7. Comprehensive evaluation, ablations, tuning, documentation, final report | In progress | Protected promotion and its documentation are complete. Broad model/task coverage, end-to-end performance tuning, ablations, and the reproducible final study remain. |

## Goals versus current reality

Engram is no longer handing inference to `llama.cpp` or a Transformers model
shell for the native path. The packaged OLMoE/BitNet route performs token
execution, Q7 expert reads, bounded attention, controller operations, and
greedy selection in the native CPU runtime. CUDA remains an optional training
and distillation accelerator; it is not required for inference.

The strongest demonstrated advantage is architectural: bounded semantic and
episodic state with authenticated causal replay and a path to CPU-only
execution. The measured traffic advantage of the current selector is small,
so it is not yet defensible to claim that Engram is faster than a highly tuned
`llama.cpp` implementation on general workloads. That comparison requires the
same model, context lengths, quantization, thread count, and output-quality
thresholds.

## Next development order

1. Keep rank-16/pool-6 as the frozen research point: pool 4 is below the
   desired recall margin and higher ranks do not materially improve it.
2. If the frontier remains near a 1% total-read reduction, keep the selector
   as a validated research path and prioritize fused projection/MLP kernels or
   a more aggressive learned residual selector.
3. The protected replay and authenticated opt-in package boundary now pass.
   Keep native defaults selector-disabled while deciding whether to implement
   full runtime policy consumption and broader end-to-end benchmarking.
4. Keep the exact operator-residual controller as the production boundary.
   The standalone controller-only replay passes the state-transition
   threshold, but learned provider and joint adaptation arms remain far above
   the causal threshold. Rank-256, persistent-memory, stage-local K/V,
   direct-hidden, full-corpus, smooth scheduled-sampling, normalized-residual,
   prefix-context, and independent auxiliary-corpus screens are now negative.
   The next M4 attempt must be a materially different jointly trained
   provider/controller or a different teacher/export model; isolated rank,
   context-feature, or epoch sweeps are closed.  The `merge-controller-traces`
   CLI now permits future low-memory corpus expansion by combining independently
   authenticated chunks without changing the protected split contract.

## Verification

- Python: 1,131 passed, 1 skipped (CUDA unavailable on the test runner).
- Native CTest: 20/20 passed outside the sandbox in 26.94 s (including the
  26.81 s C ABI lifecycle test).
- Lint and `git diff --check`: passed.
### Auxiliary learned-provider corpus screen (2026-08-04)

An independently pinned WikiText-2 raw train/validation split was captured
through the authenticated chunk/merge path to separate corpus size from trace
integrity. The 16-record-per-split screen produced 256 trace records per split.
Rank-16 and rank-128 PCA state/token providers scored terminal normalized MSE
`0.184007` and `0.186352` on validation, both failing the fixed `0.0225`
learned-provider promotion threshold. This is not the protected benchmark and
does not change the native fixed-controller semantic result. It closes the
specific hypothesis that a modest expansion of ordinary prose data would solve
the learned-provider failure.
