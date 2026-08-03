# Project status

Snapshot date: **2026-08-03**

Engram is an operational research prototype, not a general quality-preserving
dense-Llama compiler. The repository can inspect and trace a Llama-compatible
teacher, decompose SwiGLU MLPs, run routing and compression experiments, and
execute fixture packages in Python and C++. The native-BitNet track now also
compiles and validates a source-independent trained-model package and performs
real greedy generation with its direct C++ MLP kernel.

**The native-BitNet Milestone 2 semantic-memory gate has passed by postmortem
adjudication.** The low-bit-native track has a practical, CPU-only native DIP
implementation, not just the earlier dense-membership oracle. Its consumed
8-sequence/256-position final attempt substituted all 30 MLPs with no dense
fallback. The preserved raw report passes quality, activity, modeled physical
cold-traffic, and candidate recall.

The final raw result is KL 0.00404129, top-1 0.98828125, NLL delta
+0.00482893, final-hidden relative L2 0.0477494, 21.38001% mean active records,
41.13713% modeled traffic, 99.94058% global candidate recall, and 99.39429%
worst-layer mean recall. End-to-end sparse evaluation took 1.1449x the dense
elapsed time, so the kernel is 14.49% slower; latency was disclosed but was
not part of the frozen semantic gate. The dense-Llama conversion track remains
blocked.

The OLMoE rank-16/pool-6 selector has now also passed its separately
authorized protected semantic replay. The exact protocol-defined split was
regenerated and hash-matched before capture; all eight native CPU records pass
with 100% answer-token agreement, 0.009133 hidden relative L2, 0.004416 logit
relative L2, −0.000460 mean answer NLL delta (maximum +0.005879), and 99.8501%
mean candidate/exact-rerank recall. An authenticated CPU-only opt-in package
was assembled and one-token native generation matched the ordinary package.
The policy remains disabled by default, and this result is not an end-to-end
speedup claim.

For OLMoE, the earlier bounded W16 attention screen failed because older
context was evicted, while W128 dense attention passed attribution but exceeded
the traffic budget. The follow-up CPU-only local-cache compression experiment
now passes the sustained eight-sequence/128-position semantic gate: a full W128
cache stored as per-vector symmetric INT8 with FP32 scales reaches KL
0.00417982, top-1 0.974609, target-NLL delta +0.000391, and hidden L2
0.048147 across all 1,024 positions. Every frozen band passes, including
positions 96–127 (KL 0.00301720, top-1 0.964844, hidden L2 0.043127), while
logical attention reads fall to 25% of dense. This is an evaluator-only
Milestone 3 attention-substitution pass; the ordinary package remains W16 and
the recurrent/episodic bounded policy is not yet promoted. The authenticated
protocol and report are recorded in the milestone report.

The compressed cache is now reachable through an explicit package-runtime
override (`local_window=128, local_int8=True`) without changing the authenticated
manifest or production default. This verifies package assembly plus native
INT8 state allocation. The compiler now also accepts the explicit opt-in pair
`attention_local_window=128, attention_storage="int8"` and records that mode in
the authenticated runtime manifest; the default compiler arguments still emit
W16/FP32. A package-level 136-token benchmark measured 75.17 s for the
ordinary W16/FP32 path versus 82.06 s for W128/INT8: counted attention reads
fell from 45.83 GB to 28.11 GB, but total latency increased 9.17%. The current
scalar INT8 implementation therefore makes no end-to-end speed claim; fused
SIMD/dequantized kernels are the next optimization boundary. The benchmark
artifact is `work/olmoe_q7/local_attention_package_benchmark_2026-08-03.json`
(SHA-256 `e8d634a1e08ba12da01cc968e87a8d7b031fb510fd763ddd68a957e44e723bdb`).
The native vector-kernel layer now includes a guarded AVX2 INT8 dot-product
implementation with scalar fallback and parity coverage. This host reports
`avx2_available=false`, so its measured behavior remains the scalar path; AVX2
performance and hardware-counter traffic are still unvalidated.

### Compiled native OLMoE Q7 source track

The repository now has a separate `olmoe_sparse_expert_v1` source adapter. It
does not route OLMoE through the dense-Llama inspector. The official
`allenai/OLMoE-1B-7B-0125` revision
`9b0c1aa87e34a20052389dce1f0cf01da783f654` passed config, complete-name, and
exact remote-header shape validation: 3,219 required tensors, 3,219 present,
no missing or unexpected names, and no shape errors. The bounded verifier read
only safetensors header byte ranges and rejected unbounded responses; the
27.68 GB checkpoint payload was not downloaded by that initial audit. The
checkpoint was subsequently downloaded, authenticated, compiled, and used for
the frozen evidence below.

The native topology is promising for Milestone 2: 64 addressable experts per
layer and top-8 learned routing yield a 12.5% active-expert fraction. Selected
Q4 expert weights plus BF16 router matrices initially projected to 12.6302% of
the all-expert Q4 baseline. Trained traces then showed that post-training Q4
errors compound causally and fail. A frozen Q7/group-64 candidate subsequently
passed an all-layer 8-sequence/256-position confirmation: KL 0.00900774, top-1
0.9765625, NLL delta +0.00391912, and final-hidden relative L2 0.0460273.
Selected Q7 codes, BF16 scales, and BF16 routers project to 22.7865% of the
all-expert ideal-Q4 baseline.

The native Q7 systems gate now also passes. The compiler emitted one immutable
5,842,733,184-byte artifact containing all BF16 routers and 1,024 directly
addressable packed experts. Independent Python and C++ readers strictly
validated every code, scale, tail, header, directory entry, and zero-padding
region. The direct CPU mmap kernel exactly matched the production top-eight
route and reached output relative L2 1.94718e-6. It scheduled 45,875,200
unique packed bytes per layer/state, or 22.7865% of all-expert ideal Q4, and
needed no dense expert materialization or Transformers model. The single-row
path now dispatches the selected experts across the native thread pool. A
canonical block decoder improves representative layers 0, 7, and 15 from
108.49/106.24/117.09 ms to 16.53/12.55/12.67 ms at 12 threads
(6.56×–9.24×), with bit-identical routes and outputs.

The complete token boundary now maps a 949,242,368-byte BF16 non-MLP artifact
alongside Q7. It executes embeddings, normalization, dense attention
projections, full-width Q/K normalization, RoPE, persistent bounded attention,
residuals, Q7 experts, final normalization, and `lm_head` without constructing
Transformers. Fixture NumPy parity, batch/incremental cache equivalence,
position advancement, and reset replay pass. The production prompt `The
capital of France is` predicts ` Paris`.

An atomic package compiler now installs both artifacts, model configuration,
and tokenizer into a symlink-free exact inventory. Runtime authentication
requires the externally supplied manifest SHA-256 and rejects manifest,
content, size, inventory, path, or symlink tampering. The production package
root is `861e9cc472f9e1245db5d64e9253411d0b656a0f08df2f58264e9c708ed750db`.
Package-only generation reproduces ` Paris`; its five-position run spends
1.91 seconds in Q7 and 2.17 seconds in native execution, down from 13.08 and
13.33 seconds respectively, while scheduling the same 3,670,016,000 Q7 bytes.
Parallel 16-layer structural validation falls from about 16.5 to 2.29
seconds. Parallel six-shard and package-inventory hashing take 23.65 and 27.19
seconds respectively; these phases are now storage-bandwidth dominated.
For the untouched teacher, four concurrent sequence forwards share one model,
remain byte-exact to serial execution, and reduce the BF16 8×33 compute pass
from 366.14 to 94.78 seconds (3.86×).

The frozen package-generation integration passes against the untouched BF16
teacher: 60/60 teacher-forced top-1 decisions, 29/32 greedy tokens, and 7/8
complete prompts agree. Those short prompts do not cross the W=16 attention
window.

The complete native causal protocol does cross it and now closes the
OLMoE-specific Milestone 2 gate. Across eight sequences and 256 positions the
CPU-only, shell-free package reaches KL 0.0129809, top-1 0.960938, NLL delta
+0.0168240, and final-hidden relative L2 0.0620471. The 128 exact-local
positions and 128 bounded-retrieval positions are gated separately under the
same thresholds. The post-window half passes with KL 0.0106424, top-1
0.960938, NLL +0.0136896, and hidden L2 0.0752018. Scheduled Q7 reads are
22.7865% of the all-expert ideal-Q4 reference. See the
[complete native causal report](../reports/olmoe_q7_native_causal_2026-07-28/summary.md).
A source-bound hardened replay, explicitly labeled non-independent, reproduces
every metric and check exactly while authenticating all seven post-run roots.
It measures 88.79 seconds of native execution, of which 72.17 seconds is Q7.

### OLMoE sustained-context attribution

The formal OLMoE Milestone 2 result remains the authenticated complete-native
**8×32** confirmation above. A subsequent prospectively frozen **8×128**
development gate asked a different question: whether the combined Q7 semantic
substitution and W16/C8/K4/S2 attention policy remain stable over 1,024
positions of newly authored natural prose. All runtime/evidence checks passed,
including exact counters, Q7 and attention traffic, reset replay, source and
artifact authentication, teacher identity, and post-run rehashing. Quality
failed overall and in every band beginning at offsets 32–63:

| Policy and range | Mean KL | Top-1 | NLL delta | Hidden L2 | Quality |
|---|---:|---:|---:|---:|---|
| W16, overall | 0.14357762246730044 | 0.802734375 | +0.15929241067956923 | 0.23826045083114877 | Fail |
| W16, 0–15 | 0.011373728091207624 | 0.9453125 | +0.0006674100286545581 | 0.05591745024139527 | Pass |
| W16, 16–31 | 0.00825166942991018 | 0.9375 | −0.003416883628233336 | 0.07565470178087708 | Pass |
| W16, 32–63 | 0.08385673793038251 | 0.828125 | +0.07557724783760023 | 0.2185442634508945 | Fail |
| W16, 64–95 | 0.22342167909778254 | 0.75390625 | +0.23844351908473982 | 0.3145873202593066 | Fail |
| W16, 96–127 | 0.2572193740804778 | 0.6875 | +0.32452361259572626 | 0.35412414360325783 | Fail |

The sustained protocol SHA-256 is `82189276…eb599`; the authenticated failed
result is `673523c2…97eb`. Because that result combined Q7 and bounded-attention
drift, a matched control was frozen after the failure but before its own
execution. It retained the exact package, Q7 artifact and execution policy,
corpus, teacher reference/arrays, native library, 12-thread setting, and
evaluator identities. Its only intervention was `local_window: 16 → 128`.

The W128 control passed every semantic band and every evidence check. It also
matched all **128** pre-intervention position rows—eight sequences times
offsets 0–15—exactly, which verifies that the two candidates are identical
before W16 begins eviction:

| W128 range | Mean KL | Top-1 | NLL delta | Hidden L2 |
|---|---:|---:|---:|---:|
| Overall | 0.0034381193102017704 | 0.974609375 | +0.0014586126028746094 | 0.041389157548110234 |
| 0–15 | 0.011373728091207624 | 0.9453125 | +0.0006674100286545581 | 0.05591745024139527 |
| 16–31 | 0.001968802186894436 | 0.9765625 | +0.0036758615460712463 | 0.038717562216334045 |
| 32–63 | 0.002101623871460845 | 0.9921875 | +0.002262885092477518 | 0.039487719419412315 |
| 64–95 | 0.002359753387329633 | 0.96875 | +0.006798400573984509 | 0.038475771420053206 |
| 96–127 | 0.002619834842965574 | 0.9765625 | −0.0053984710423264914 | 0.040275633124110755 |

The control protocol SHA-256 is `1619cd5f…4dd9`; its authenticated result is
`3d099ffd…d345`. This matched attribution vindicates Q7 semantic substitution
and preserves the formal M2 pass. The remaining OLMoE blocker is **Milestone 3
bounded attention**, not semantic routing or Q7.

W128 is deliberately nondeployable. It reads 2,164,260,864 logical attention
bytes per sequence—**100%** of dense full-context reads—and holds
35,825,664 bytes (35.8 MB) of attention state, versus W16's 31.2863% read
fraction and 6,336,512-byte state. The attribution protocol deliberately
exempts deployability; W128 would violate the 45% attention-read requirement.

The prospectively frozen matched **8×128** development sweep is now complete.
It ran all three predeclared arms in fixed order against the consumed
sustained-development corpus. Each arm read exactly 968,753,152 logical bytes
per sequence (44.7613856589% of dense), exposed 32 values per mature step, and
retained the same Q7 schedule at 22.7864583333% of the all-expert ideal-Q4
reference:

| Policy | Mean KL | Top-1 | NLL delta | Hidden L2 | Evidence | Quality |
|---|---:|---:|---:|---:|---|---|
| W16/C18/K16/S2 | 0.06388655 | 0.8671875 | +0.05170082 | 0.15771664 | Pass | **Fail** |
| W24/C10/K8/S2 | 0.06591232 | 0.8779297 | +0.05847984 | 0.15975482 | Pass | **Fail** |
| W30/C4/K2/S2 | 0.09581344 | 0.8408203 | +0.07572840 | 0.18842230 | Pass | **Fail** |

All three exact 0–15 and 16–31 bands passed. Hidden-state drift failed the
32–63 band, and quality failed broadly at 64–95 and 96–127. All artifact,
source, counter, replay, post-run authentication, and pre-eviction identity
checks passed. Because zero arms passed every frozen quality check, the
predeclared rule selected **no arm** and did not consume the separately sealed
fresh-confirmation corpus.

The sweep protocol is `2853de54…cef0`, the result is `813bac5b…7658`, and
the frozen evaluator is source commit `102bda2` with source SHA
`cf2e4be0…fa60`. The intervention used the raw native runtime to override the
immutable package's W16/C8/K4/S2 attention policy; it did not mutate or promote
that installed package. Milestone 2 remains passed, while Milestone 3 remains
blocked. Another global W/C/K trade is not justified.

The prospectively frozen layer-rescue follow-up tested whether concentrating
that budget in a few entire layers could recover quality. The added layered
native ABI first passed exact old-scalar/all-base-layered parity. A
three-round greedy search then evaluated 45 causal candidates—16, 15, and 14
remaining layers—on a deterministic two-record selection split and chose
layers **11, 6, and 10**. The selected schedule applies W128/C8/K4/S2 to those
three layers and the base W16/C8/K4/S2 policy to the other 13. It reads
**955,957,248 logical attention bytes per sequence** (**44.1701489826%**),
holds **11,865,728 bytes** of attention state, needs **6,528 bytes** of
scratch, and leaves Q7 traffic at **22.7864583333%**.

The six remaining development records were an internal screen, not a fresh
confirmation. Its overall result failed all four quality thresholds:

| Layer-rescue range | Mean KL | Top-1 | NLL delta | Hidden L2 | Quality |
|---|---:|---:|---:|---:|---|
| Overall | 0.10232094998 | 0.84505208333 | +0.11677564952 | 0.20603686522 | **Fail** |
| 0–15 | Pass | Pass | Pass | Pass | Pass |
| 16–31 | Pass | Pass | Pass | Pass | Pass |
| 32–63 | Fail | Fail | Fail | Fail | **Fail** |
| 64–95 | Fail | Fail | Fail | Fail | **Fail** |
| 96–127 | Fail | Fail | Fail | Fail | **Fail** |

Every evidence, exact-resource, reset-replay, ABI-parity, and post-run
authentication check passed. The evaluator source commit is `708782b`; the
protocol SHA-256 is
`9514e90bd5d14ae01ea27185763e5a833d4f1963e6bffd0ec0c81848f35b0c3e`,
the result SHA-256 is
`97ce800bd855c1f16248cada696936c7c56acd49c02d6f1c9ce9885dc44f7c49`,
and the layered candidate DSO SHA-256 is
`fe4dfdcc7e87a3cd5e36074e07d297f838ba345c37e939eeb0d796cb39cce409`.
Because the corpus was already consumed for development, the schedule was not
promoted and no fresh package confirmation was attempted.

This closes the frozen greedy three-layer W128 path, while its disclosed
search limitation leaves interacting whole-layer combinations untested. The
prospectively frozen teacher-attention-mass experiment is now complete. It
captured dense BF16 attention maps from only the deterministic two-record
selection split, measured for each layer-head pair the older-context mass not
covered by the four largest older attention weights, and froze the first
**51 of 256 pairs** before examining the six internal-screen outputs. Those
51 heads use W128/C8/K4/S2; the rest retain W16/C8/K4/S2.

The resulting policy reads **973,384,704 logical attention bytes per
sequence**, or exactly **44.975387218386625%** of dense. A 52-pair mask would
consume **45.2437999637%** and is inadmissible. Q7 remains unchanged at
**93,952,409,600 scheduled bytes per sequence**. Exact all-base headwise ABI
parity passed, as did every resource, reset-replay, evidence, and
authentication check.

The six reused 128-position development records contributed 768 prediction
positions. They were an internal screen, not an untouched holdout or fresh
confirmation:

| Fixed 51-head screen | Result | Required | Outcome |
|---|---:|---:|---|
| Mean KL | 0.07371992968429097 | ≤ 0.05 | **Fail** |
| Teacher top-1 agreement | 0.8671875 | ≥ 0.90 | **Fail** |
| Target NLL delta | +0.05345554334600896 | ≤ 0.05 | **Fail** |
| Final-hidden relative L2 | 0.1675178178168911 | ≤ 0.10 | **Fail** |

The 0–15 and 16–31 bands passed, with degradation returning after position
32. Relative to the three-layer rescue at 44.170% reads (KL 0.10232095,
top-1 0.845052, NLL +0.11677565, hidden L2 0.20603687), the head-wise mask
improves every overall metric, but not enough to pass. No fresh confirmation
was run and no package policy was promoted.

This closes the tested fixed **teacher-attention-mass ranking**, not all
head-wise allocation. It motivated the prospectively frozen
causal/value-sensitive static follow-up below. Milestone 2 remains passed; the
Milestone 3 attention gate remains blocked.

### Causal/value-sensitive 51-head follow-up

The follow-up fit used exact native W16/C8/K4/S2 decisions, a differentiable
gathered surrogate, two iterative-hard-thresholding steps, and the same fixed
budget of exactly 51 W128 rescue heads. On the two frozen selection records,
the chosen M1 mask reduced the maximum composite objective from
**7.867116928100586** to **4.755991458892822** and the mean from
**6.917216062545776** to **4.328476905822754**. Both records improved, the
training evidence gate passed, and fitting took **6,930.099 seconds**. CPU
fitting was the dominant systems bottleneck. That boundary is now resolved
for development fits: a deterministic frozen-expert proxy preserves the
installed serial `grouped_mm` CPU forward and parallelizes only independent
expert backwards across 12 workers. One authenticated full M0/sequence-0
record matched loss, all 256 gate gradients, non-timing native diagnostics,
and the projected 51-head mask exactly. Record time fell from 1,564.347 to
809.168 seconds, a 1.933× speedup and 48.274% reduction. The proxy is
authorized for larger fits but does not alter the sealed native-Q7 reference
or count as new semantic evidence. See the
[qualification report](../reports/olmoe_q7_expert_proxy_2026-07-28/summary.md).
The comparison used one previously consumed record from separate executions;
it is not a controlled repeated benchmark or a measured runtime for the full
two-record, two-step fit.

The sealed artifact chain is rooted at source commit `483c62f`:

- protocol SHA-256:
  `037ebfd7d4e40af898ece7f353654eb8a41dc1883f191cbdf05fc34bf50bf4ba`
- training-result SHA-256:
  `bacb0e31899f514a8b2b517987566e8bca68d39cabfd50b3c9e7ecf83bc756ea`
- screen-protocol SHA-256:
  `282bfe0b9e1da86577f0187112a4a444b0f36d7f84e10f4f9bb67730676807c2`
- screen-result SHA-256:
  `437d0de4ce4da37e69ca13279b76627d6f7721e766b8f1b4371fb318e7cbeb59`

The complete native-Q7 screen reused the six already-consumed development
records and 768 prediction positions. Its evidence and resource gates passed,
including the **44.9753872%** logical-read budget, but its quality gate failed:

| Causal/value-sensitive 51-head screen | Result | Required | Outcome |
|---|---:|---:|---|
| Mean KL | 0.07913208059 | ≤ 0.05 | **Fail** |
| Teacher top-1 agreement | 0.8645833333 | ≥ 0.90 | **Fail** |
| Target NLL delta | +0.08119899696 | ≤ 0.05 | **Fail** |
| Final-hidden relative L2 | 0.18264718059 | ≤ 0.10 | **Fail** |

The 0–15 and 16–31 bands passed. The 32–63 band failed top-1 agreement and
hidden-state error; both later bands failed all four quality measures. The
static causal/value-sensitive mask is worse than the earlier attention-mass
mask on every overall metric. No fresh confirmation was opened and no package
policy was promoted.

This closes the tested two-record natural-prose causal/value-sensitive
selector, not every static selector. Milestone 2 remains passed and Milestone
3 remains blocked.

### Retrieval-targeted selector result

The next experiment is now implemented as a separate evaluator rather than an
extension of the consumed natural-prose fit. Its newly generated synthetic
passkey corpus has an exact **8/8/8 train/development/sealed-confirmation**
split. Every record contains 129 tokens, of which 128 are model inputs; only
the 32 ground-truth answer targets at logit rows **96–127** contribute to the
training objective. Four eight-token passkeys occupy four balanced source
depths. The 24 records use **768 globally unique numeric singleton tokens**,
with disjoint record identities and no passkey-token overlap among splits.

The training contract uses complete packaged native Q7 forward logits and
answer-only cross-entropy. A straight-through boundary supplies gradients
through a frozen BF16 shell with exact native attention, and the previously
qualified frozen-expert backward proxy supplies 12-worker expert backwards.
Two IHT steps form `M0 → M1 → M2`; `M1` and `M2` are each hard-projected to
exactly 51 rescued layer-head pairs, with a prospective worst-record,
mean-record, and no-regression selection rule.

The exact candidate budget remains 973,384,704 logical attention bytes per
128-position sequence, or **44.9753872184%** of full causal attention, with
12,284,864 bytes of persistent attention state. The 52-head boundary is
**45.2437999637%** and therefore inadmissible. A selected mask must then pass
the full-W128 packaged-Q7 development control and its own packaged-Q7
development screen both overall and separately at all four source depths.

The fail-closed protocol is frozen at
`work/olmoe_q7/retrieval_selector_2026-07-29_frozen/protocol.json`, SHA-256
`f6961ec6bffadb306a75ae8aaa8c400d1282cbb096876532d75e22a295660580`.
The complete fit and development screen ran. `M2` passed the training
progression rule, the teacher demonstrated retrieval, and the full-W128 Q7
control passed. The exact-51 candidate did not: overall KL 0.186610,
target-NLL delta 0.283658, and hidden relative L2 0.335103 all exceeded their
thresholds; top-1 agreement alone passed at 0.929688. All resource,
reset/replay, proxy-lifecycle, and artifact-authentication checks passed.
Confirmation remained unopened and unauthorized. The
[archived evidence](../reports/olmoe_q7_retrieval_selector_2026-07-29/summary.md)
therefore closes this static mask. The train-only attribution sequence that
followed is summarized below. Milestone 3 remains blocked.

### Episodic and head-gated retrieval boundary

The first causal two-prototype follow-up did not improve global `M2`. Its
assigned mean answer cross-entropy was 1.046825 versus 1.005444 for `M2`, and
five of eight training records regressed. An exact episodic oracle then removed
selector error by storing and rereading the known eight-token payload for each
answer block. It also failed: mean/worst answer CE was 1.224460/1.327343, seven
records regressed, and the largest divergence occurred on the first prediction
of each retrieved span. Adding the source label immediately before each
payload produced mean/worst CE 1.231254/1.321619 and again regressed seven
records. This rejects both cheap cache representations as sufficient repairs,
not episodic retrieval as a class.

These were semantic failures. The payload-only and label-plus-payload paths
passed their exact native counters, reset replay, artifact authentication, and
traffic checks. Label-plus-payload used 719,585,280 upper-bound total bytes
(33.2485% of dense full-context K/V), 11,059,712 state bytes, and 4,992 scratch
bytes. Its frozen protocol/result SHA-256 values are
`1812a6ba72afe0c5f32e459867c29f3d8dbd609a3d0ddf59ac52ae6859ce4d3d`
and
`e1ec5a2bde8b9ce7198fe1571a7670c45a3bc7a712cdf9a856f869b6429fe69d`.
Development and confirmation remained unopened.

The native runtime now has a versioned head-gated episodic ABI. Inactive layers
execute the exact legacy attention step without allocating an episodic bank;
active layers retain full K/V rows, but only mask-enabled query heads include
the episodic span in their softmax. The all-ones mask has exact all-head parity,
while missing, malformed, and all-zero policies fail closed.

The first use of that ABI transferred the fixed retrieval `M2` mask—51
layer-head pairs over 14 active layers—to the exact payload oracle. All
systems checks passed at 687,472,640 total traffic bytes (31.7648% of dense)
and 10,010,112 state bytes. Quality did not: mean/worst answer CE was
1.400569/1.694034, and only one of eight records improved over `M2`. See the
[authenticated K51 evidence](../reports/olmoe_q7_retrieval_episodic_head_mask_2026-07-29/summary.md).

The frozen ranked head-prefix screen at **K64, K96, K128, and K165**
completed all four candidates. Mean/worst answer CE was
1.379699/1.639418, 1.328848/1.618843, 1.337958/1.621764, and
1.331006/1.608617 respectively. Each K improved only one of eight records and
failed the strict gate. The total-failure rule retained K165 for diagnostic
reset replay because it had the lowest worst CE; replay passed, but no policy
was promoted. Protocol/result SHA-256 values are
`e238b5cfc4359422abc96c227f4010ade747ff161c946b2edae14c171a7bf04c`
and
`a6fb045bbad526411b8318e7de2412ea56d69b3f2be0d977ca6819635c0718da`.
All systems checks passed and confirmation remained unopened.

This closes larger cardinalities under the transferred `M2` ranking. The
authenticated K256 all-head payload was a better fixed starting point than
K165—1.224460 mean and 1.327343 worst CE at 33.0305% upper-bound traffic.

The completed V2 logit-mass screen then held K256 and its schedule fixed and
tested `gamma=1/2,1/4,3/16,1/8`. All four arms executed all eight training
records, passed the exact systems contract, and failed. Mean/worst answer CE
was respectively 1.461414/1.669250, 1.883818/2.288258,
2.161750/2.595642, and 2.725091/3.430532. The total-failure rule replayed
`gamma=1/2` exactly, but it is only the best failed nonzero arm and is worse
than historical `beta=0`. Development was not authorized and the reserved
confirmation split remained unopened. See the
[archived V2 evidence](../reports/olmoe_q7_retrieval_episodic_logit_bias_2026-07-29/summary.md),
result SHA-256
`19d08ce9eb4b673d423e9781a491e25ec03bdec09467a43e7be1881874ef2287`.

Shared scalar calibration is closed. The subsequent same-state W128-shadow
residual-capacity screen fixed `beta=0` as the K256 base and measured an
optimistic leave-one-sequence-out output-subspace ceiling with oracle held-out
coefficients. Ranks 2/4/8 recovered 0.400470/0.428686/0.469253 globally.
Each passed the sequence, block-entry, finite, and positive-layer conditions,
but each missed the frozen 0.50 global gate. Replay and all post-run
authentication checks passed; confirmation remained unopened. The
[archived capacity evidence](../reports/olmoe_q7_retrieval_episodic_residual_capacity_2026-07-29/summary.md)
closes only rank-at-most-8 global per-layer output subspaces. No correction
fit was authorized.

The subsequent dynamic per-head mass oracle selected one of eight gamma codes
at every state/layer/head coordinate to match the W128 teacher's scheduled
source mass. Mean mass error improved from 0.0445126662 to 0.0084754603
without any coordinate regression, but global post-`W_o` recovery was
**-0.10891245427020602**. Every sequence and block-entry recovery was
negative, and only 1/16 layers was positive. Protocol/result SHA-256 roots are
`fe09689452e6ae4f1a1b15332c61c1cc990cfc29b6a8b0d5a1758d9490a93af5`
and
`f7060e7373c5faf8f154891e93efad35659723d8e3f04d83638b62fa9cf72596`.
The [archived evidence](../reports/olmoe_q7_retrieval_episodic_head_mass_oracle_2026-07-29/summary.md)
closes exact scheduled-source-mass matching as an objective.

The next experiment removed that objective mismatch by optimizing all heads
jointly against the exact W128-minus-K256 post-output-projection residual.
Its continuous box relaxation is an optimistic superset of the discrete gamma
family:

| Joint-gamma result | Global recovery | Sequences ≥0.25 | Blocks ≥0.25 | Positive layers |
|---|---:|---:|---:|---:|
| Continuous optimistic relaxation | **0.22738059544921096** | 1/8 | 0/4 | 16/16 |
| Discrete direct float32 | **0.1997680396822742** | 0/8 | 0/4 | 16/16 |

Both failed the frozen 0.50 global, every-sequence, and every-block
requirements. Protocol SHA-256 is
`aa03a71e3dd9e1fbb413a7773d57189c41029c26b2f4372b2fb7a26744305d24`;
result SHA-256 is
`1329a51bac71cb81f44494c8ef70cb23a631eacebac5540b39e2e98ed5e30ea5`.
The [archived joint-gamma evidence](../reports/olmoe_q7_retrieval_episodic_joint_gamma_oracle_2026-07-30/summary.md)
passed replay and authentication, but authorized no predictor, Milestone 3
promotion, or confirmation access. Scalar mass retuning at fixed K256 is now
closed.

The next capacity screen exposed the eight exact BF16 episodic values
individually. Its constructible per-head simplex recovered 0.3844378107
globally; an optimistic superset that also contained the exact native head
output recovered 0.3844378142. Both passed every sequence, block-entry, and
positive-layer condition but missed the frozen 0.50 global gate. The
optimistic maximum objective-gap bound was only `5.90e-11`; exact replay,
direct/quadratic parity, and all post-run authentication checks passed. This
is therefore a decisive closure of same-state reweighting over the current
regular aggregate plus eight episodic values. See the
[archived per-slot evidence](../reports/olmoe_q7_retrieval_episodic_slot_simplex_oracle_2026-07-30/summary.md),
rooted by result SHA-256
`2e8e9b7d5f33d33c0e8c642a50359da785d0690d8260ed9f65837b14cd93a5bf`.

The subsequent frozen full-visible oracle exposed those regular entries
without adding reads: 16 local values plus four selected-older values and the
eight episodic values form constructible C28. An optimistic C29 arm adds the
exact native head output as an extra anchor.

| Full-visible capacity check | Constructible C28 | Optimistic C29 | Requirement |
|---|---:|---:|---:|
| Global recovery | **0.6653937751** | **0.6653865288** | >=0.50 |
| Minimum sequence recovery | **0.6447006551** | — | >=0.25 |
| Minimum block-entry recovery | **0.6306278392** | — | >=0.25 |
| Positive-recovery layers | **16/16** | **16/16** | >=12/16 |

All qualification, gate, deterministic-replay, source/artifact
authentication, and post-solve checks passed. Nested C10 and C16 diagnostics
recovered 0.5335805245 and 0.6021187653, but they were diagnostics only and
had no progression authority. The native resource contract stayed at
10,534,912 bytes of state and 714,866,688 logical traffic bytes, or 33.0305%
of dense full-context KV; the experiment added no KV reads. Result SHA-256 is
`a8711f07fdcbe48b16ebe962aae1962e4613873640eba20bc7fa88238216aee1`;
see the
[archived full-visible evidence](../reports/olmoe_q7_retrieval_episodic_full_visible_simplex_oracle_2026-07-30/README.md).

This passes the train-only same-state capacity gate and authorizes a causal
28-logit selector. It does not establish native causal learnability, promote
an attention policy, pass Milestone 3, authorize development, or authorize
confirmation. The confirmation split remained sealed.

The two frozen learned successors are both negative:

| Train-only selector | FP32 global | BF16 global | Minimum sequence | Minimum block | Positive layers | Traffic |
|---|---:|---:|---:|---:|---:|---:|
| Rank-4 query content plus mass | 0.2542615526 | 0.25422074198 | 0.23161600085 | 0.18371154473 | 16/16 | 36.8096% |
| Eight-phase table plus mass | 0.2618976463 | 0.2618728353 | 0.2405241062 | 0.2244750908 | 16/16 | 34.0116% |

The content-selector protocol/result roots are
`0a58ba3a59d2f0f816046ca28aac304baf7663ef890a6b298f0cc7277613d051`
and
`9ea504f83a487584cb9ae2127565674a8e341ca58f6777a03514b0c9a281995c`.
The phase-selector protocol/result roots are
`8cb1c7b0e9a6bc2d23839cdbf4de973e66616cccc86e980e6a151d4f2b773987`
and
`52360cf47cb2eeab52e595961f436e4c1e7b79db6cdaa339b7f699d3290883ed`.
Its BF16 block-entry recoveries are
0.314588398/0.228395562/0.261696236/0.224475091. The deployable phase artifact
has 82,944 parameters, occupies 165,888 BF16 bytes, and accounts for
736,100,352 logical bytes, or 34.0116% of dense. It improves the preceding
mass-only BF16 result by only 0.0040699.

All systems, resource, parity, deterministic replay, and authentication checks
passed. The semantic gates did not. These are train-only model-selection
outcomes on the exposed training corpus, not independent generalization
evidence. No development, confirmation, native integration, or package
promotion was authorized. Query-content and phase-on-mass are closed; the
next directional boundary is a blockwise-QK feature controller. Milestone 2
remains passed; Milestone 3 remains blocked.

### Blockwise-QK feature capture (train-only, 2026-07-31)

The next boundary is now implemented and captured against a refreshed,
source-bound full-visible protocol. The native evaluator can optionally emit
eight scaled Q/K partial dot-product bands for each of the 28 entries already
visible to the C28 attention step. The feature path has its own C ABI,
Python binding, reset-proven safetensor shards, authenticated manifest, and a
reusable train-only audit. It is evaluator-only; production generation does
not enable it.

The eight-record, 32-layer, 16-head, 32-read capture has shape
`[8, 32, 16, 16, 28, 8]`. Summing the bands reconstructs the authenticated
native masses with maximum absolute error `1.90735e-6`, mean absolute error
`1.36212e-8`, and 99th-percentile error `2.08616e-7`. Score ordering agrees
with native mass ordering on 65,529 of 65,536 rows (99.9893%). These are
feature-fidelity results, not a causal quality or Milestone 3 pass.

The score-ranked mass-retention screen is still too weak to authorize a
bounded selector: top-20 retains 96.26% mean mass but only 91.20% at the
10th percentile, while top-24 reaches 96.52% at that percentile. The QK
trace is therefore a viable compatibility feature for a new locality model,
not evidence that a cheaper candidate policy already exists. The confirmation
split remains sealed.

### Native-BitNet package integration and evidence caveat

The qualification is not a pristine runner pass. After the evaluator
completed, the original wrapper marked the consumed attempt `error`: the
verifier compared the protocol's frozen full-record hashes, made with the
canonical `input_ids` object envelope, against raw-evaluator hashes of the
first 33 scored tokens made from bare lists. A separate no-model postmortem
adjudicator verified the corrected identities, the frozen bindings, and all
primitive measurements. The raw report was prospectively sealed about 13
minutes after the error, rather than being contemporaneously hash-bound by
the original result. See the
[final evidence summary](../reports/native_bitnet_m2_final_audit/49df50cc01c96844ab3e7015d66c8899025dad4e1f7f01a450f97677751b36f2/summary.md).

That decision is now integrated into a derived native package and the complete
C++ token-step path. The installer authenticates the frozen policy
(`c572754e…3768e`), adjudication (`ebb5ca95…a5cc`), base artifact
(`4fcf598a…ab55`), and v2 index (`b98ce4e4…0e15`), and never mutates the
policy-bound source package. `NativeBitNetTokenRuntime` is DIP-only and
fail-closed: every layer directly executes attention, normalized semantic
input extraction, DIP, and semantic-output acceptance, with no dense semantic
backend or fallback.

The promoted manifest SHA is `707bbe06…26926`; the rebuilt standalone
executable SHA is `c6c5b05b…a15b`; and the versioned chat-runtime DSO SHA is
`4b732beb…72a`. The executable and shared ABI both authenticate the exact,
symlink-free inventory and all semantic trust roots before model mapping and
derive architecture and EOS (including `128009`) from packaged files. The
standalone executable has no Engram shared-library dependency. The chat DSO
depends only on system libraries and exports six versioned C symbols.

The fixed non-holdout eight-prompt/32-token integration confirmation has now
passed with 32/32 greedy token-ID agreement and 8/8 exact prompts. Global mean
activity is 21.56017%, with a 22.58916% maximum prompt mean. Complete modeled
cold traffic is 30,153,074,432 bytes, including 194,304 global-metadata bytes;
its global mean is 41.16116% of dense ideal Q4 and its maximum prompt mean is
41.29835%. All absolute-position, stage/semantic-call, semantic-row, backend,
traffic-recomputation, generated-budget, and reset-replay checks passed on
CPU.

The rebuilt-core confirmation took 390.4183 seconds across first runs, reset
replays, and per-process package authentication; native counters/timings are
first-run snapshots. Exact means greedy tokens, not hidden or logit parity.
Reset proves token replay, zeroed counters, and structural metric parity, not
hidden-state identity. The frozen suite still stops at 14 positions. A real
interactive chat smoke processed 17 prompt tokens and crossed W=16, but it
does not establish sustained older-memory quality. A separate 16/17/18/24/32
position protocol now proves exact eviction, older-candidate, older-selection,
sink, heavy-hitter, fixed-state, and reset mechanics. At 32 positions it
records 480 evictions, 60,000 older keys scored, 34,800 older values selected,
1,200 sink insertions, and 5,654 accepted heavy-hitter updates while state
remains 7,477,440 bytes. This is integration correctness, not speed or dense-
teacher long-context quality evidence. See the
[native attention report](../reports/native_bitnet_dip_attention_confirmation_2026-07-27/summary.md).

The frozen practical-routing policy is
[machine-readable](../reports/native_bitnet_m2_2026-07-26/frozen_dip_policy.json).
The strongest earlier full-record systems evidence is the
[2026-07-24 direct-kernel confirmation](../reports/semantic_gate_native_bitnet_2026-07-24/kernel_confirmation.json);
the prior cross-track snapshot remains the
[2026-07-23 semantic-gate summary](../reports/semantic_gate_status_2026-07-23/summary.json).
Large corpora, checkpoints, and scratch experiments remain under ignored
`work/` paths and are not source-control artifacts.

## Milestone 2 ledger

Native BitNet and OLMoE Q7 can proceed beyond their source-specific Milestone
2 gates. OLMoE now passes its causal quality screen, packed native systems
gate, complete native token boundary, authenticated package boundary, frozen
short generation integration, and frozen complete native causal
confirmation. Generic dense-Llama conversion remains blocked:

| Milestone 2 deliverable | Native BitNet | OLMoE Q7 | Generic dense Llama |
|---|---|---|---|
| Background operators | Exact packaged residual; learned correction is zero | Native top-8 mixture requires no added residual in the passing simulation | Current fitted background hurts held-out quality |
| Semantic key/value package | Complete ternary records plus authenticated DIP-v2 index | **Authenticated package with 5.84 GB immutable packed-Q7 artifact** | Format/runtime exist; no qualifying trained artifact |
| Practical routing | **Passed**, all 30 MLPs, no fallback | **Direct native top-8 CPU route passed** | **Blocked** |
| Quantization | Native packed ternary | **Canonical Q7/group-64 and BF16 scales validated** | Experimental product/additive codecs exist |
| Python runtime | Persistent native DIP handle | **Complete native token owner and persistent caches pass** | Research runtime exists |
| End-to-end substituted MLPs | Native evaluation, generation, and chat passed | **Formal complete shell-free native 8×32 causal confirmation plus frozen generation integration pass; matched W128 attribution vindicates Q7 at 8×128** | No gate-passing compilation candidate |

These are track-specific results. Neither successful source track erases the
original dense-source failures. OLMoE does not need another semantic/Q7
selection experiment: its next boundary is a deployable Milestone 3
bounded-attention policy that passes the authenticated 8×128 bands at no more
than 45% logical attention reads. The completed ranked episodic head-prefix
screen was a train-only capacity test within that boundary; all four
candidates failed. The subsequent fixed-K256 logit-mass screen also failed
all four nonzero candidates while passing its systems contract. The completed
same-state shadow residual screen then rejected ranks 2/4/8 because their
40.05%/42.87%/46.93% global recovery missed the frozen 50% gate, even though
all other conditions passed. That closes only global per-layer subspaces with
oracle coefficients. The subsequent per-head mass-matching oracle failed with
-0.108912 global recovery, and its joint output-targeted successor failed even
under a continuous optimistic relaxation at 0.227381 global recovery. The
direct discrete result was 0.199768. Both joint arms missed every-block and
nearly every-sequence requirements despite 16/16 positive layers. Scalar mass
retuning is closed. The exact per-slot successor raised the same-state
ceiling to 0.384438 and passed every non-global condition, but its
exact-native-anchor hull still missed the 0.50 global gate with a maximum
certificate gap of only `5.90e-11`. The no-extra-read full-visible successor
then passed its frozen train-only capacity gate: C28 recovered
0.6653937751 globally, with 0.6447006551 minimum sequence recovery,
0.6306278392 minimum block recovery, and 16/16 positive layers; C29 recovered
0.6653865288. The result keeps traffic at 33.0305% with no new KV reads. It
authorized learned causal selection, but the subsequent content and
phase-on-mass selectors recovered only 0.25422074198 and 0.2618728353 in BF16.
Both frozen train-only screens failed semantically while passing their
systems contracts. No native causal, development, confirmation, package, or
Milestone 3 result exists; the next directional class is blockwise-QK.

## Semantic gate definition and outcome

A candidate must use one serialized and independently reloaded artifact and
pass all of the following on an all-layer causal evaluation:

| Requirement | Threshold |
|---|---:|
| Teacher-to-student KL | at most 0.05 nat/token |
| Teacher top-1 agreement | at least 0.90 |
| Target NLL delta | at most +0.05 nat/token |
| Final normalized hidden-state relative L2 | at most 0.10 |
| Evidence | at least 8 unique sequences and 256 prediction positions |
| Complete physical cold MLP traffic | at most 45% of dense ideal Q4 |
| Mean selected records | at most 25% of the 6,912 records |
| Candidate recall | global micro and every layer mean at least 0.95 |

Configuration selection used development-only data. The consumed final
confirmation used the identical frozen artifact on a sequence-disjoint corpus
that was not used for fitting or selection. The checked-in holdout is
plaintext, so this is procedural/honor-system separation, not cryptographic
secrecy. The passing decision is by the postmortem adjudication described
above, not by the original runner result.

## Strongest measured frontiers

No row in the original dense-Llama track passes both sides of the gate. The
native BitNet row now passes direct packed execution, the evidence floor, all
semantic thresholds, and exact scheduled cold-byte accounting.

| Representation | Quality result | Systems result | Decision |
|---|---|---|---|
| Native BitNet DIP, frozen practical policy | Final 8-sequence/256-position raw result: KL 0.00404129, top-1 0.98828125, NLL +0.00482893, hidden L2 0.0477494; global recall 0.9994058, worst-layer mean 0.9939429 | 21.3800% mean active records; 41.1371% modeled physical cold traffic; CPU-only native kernel; 1.1449x dense elapsed time | **Semantic gate passed by postmortem adjudication; original wrapper ended in error** |
| Native BitNet layer-adaptive exact-membership oracle | Historical oracle result: KL 0.02543, top-1 0.94531, NLL +0.02386, hidden L2 0.09205 | 15–35% per layer, 24.8375% mean selected down records; dense gate/up coefficient scan remains | **Oracle ceiling passed** and motivated practical DIP |
| Native BitNet phase-stream base-3 records | Frozen 8-sequence/256-position result: KL 0.00371, top-1 0.96094, NLL +0.00224, hidden L2 0.04678 | Direct memory-mapped CPU kernel; 318,924,544 scheduled cold bytes, 40.0527% of dense Q4; all MLP records execute | **Systems substrate only**; not routed semantic memory or a Milestone 2 pass |
| Exact float magnitude oracle, K=768 | KL 0.0336, top-1 0.9124, NLL +0.0153, hidden L2 0.0953 | More than 4x the dense-Q4 payload before a practical selector | Semantic capacity exists, but this is not deployable |
| Predictor-free DIP, q=432/C=896/K=768 | Untouched confirmation: recall 0.9897, KL 0.0286, top-1 0.9101, NLL +0.0326, hidden L2 0.0905 | 76.39% scalar traffic, 83.33% cache-line traffic; native kernel is 0.863x dense throughput | Quality pass, systems fail |
| Mild layer-adaptive compact Q4 student | At 3,000,093 training positions: KL 0.8866, top-1 0.5659, NLL +0.8838, hidden L2 0.4245 | Serialized/reloaded artifact is 44.9334% of dense ideal Q4 | Traffic pass, quality fail; stopped at the frozen 3M rule |
| Budget-native full-width grouped ternary | At 1,014,225 training positions: KL 2.2844, top-1 0.3198, NLL +2.2770, hidden L2 0.6036 | Serialized/reloaded artifact is 43.1353% of dense ideal Q4 | Traffic pass, quality fail; top-1/hidden miss frozen 50%-gap-closure rule |
| Exact nonparametric output memory | Layer-14 LLE-32 error 0.3275 with 233,005 local records; 0.3219 after adding 1,000,000 pretraining records | Exact search and FP16 values are not a deployable traffic result | Only 1.73% improvement; density-scaling branch closed |
| Mixed affine LC-VQ | Development-only layer-14 hard-QAT error 0.3364 after 8,192 steps | Complete modeled cold traffic is 44.3482% | Traffic pass, local-quality fail; no causal run |
| Fully sparse top-K activation path | Best unseen all-layer result after causal schedule fitting and verified attention/norm co-adaptation: KL 0.4517, top-1 0.6714, NLL +0.4585, hidden L2 0.3272 | Fixed per-layer q/K schedule is exactly 45% ideal traffic before metadata; every artifact reloads and executes on CPU | Whole-model hypothesis tested and stopped; far from every semantic threshold |

The native-BitNet DIP result is the first tested practical mechanism to clear
the complete final quality, recall, mean-activity, and modeled physical
traffic gate. Its status is pass-by-adjudication, with the evidence-integrity
caveats above. This does not promote it into the generic dense-Llama compiler.
The older SmolLM DIP quality result in the table is a separate dense-source
experiment that failed systems traffic and latency.

### Native-BitNet oracle semantic ceiling

The corrected Milestone-2 restart first tested the actual BitNet teacher rather
than treating lossless full-record execution as semantic routing. A new direct
CPU oracle ranks additive records after the teacher's Q8 activation,
ReLU-squared gate/up product, intermediate RMS normalization, gain, and second
Q8 quantization. Fixed 25% selection passed 32-position development but missed
the frozen final-hidden limit at 0.10448.

A development-only all-layer sweep then allocated 15–35% per layer while
holding the exact aggregate below 25%. The selected schedule averages 24.8375%
and passes the untouched frozen protocol: KL 0.02543, top-1 0.94531, NLL delta
+0.02386, and final-hidden relative L2 0.09205. The report is
[here](../reports/native_bitnet_oracle_2026-07-26/summary.md).

This establishes semantic concentration and the target membership schedule.
By itself it did not close Gate 2 because selection still consulted all dense
gate/up coefficients.

The subsequent practical-router screen was positive at representative depths.
A direct nonlinear rank-256 membership predictor is rejected at only 77.75%
recall with 1.5x candidates. BitNet-specific Dynamic Input Pruning instead
scores ternary gate/up keys from the largest 75% of input coordinates and
reaches 96.23%, 98.06%, and 96.78% recall at layers 0, 14, and 29. Its
provisional modeled traffic was about 35–41% of dense Q4. See the historical
[router screen](../reports/native_bitnet_router_2026-07-26/summary.md).

### Native-BitNet practical DIP development pass

The all-layer follow-up is now implemented as a memory-mapped C++ CPU kernel.
It accepts live BF16 MLP inputs, applies native Q8 quantization, keeps the
largest `q=1920` of 2,560 coordinates, and scans packed coordinate-major
ternary gate/up streams across all 6,912 records. It exactly completes the
frozen per-layer candidate set, estimates the coupled RMS, computes exact
candidate utilities, and reads only the down rows selected for that token.
The adaptive count is the number of nonzero candidate utilities, clipped to
`minK=346` and each layer's `Kmax`.

```text
C    = [4224,5504,4224,4224,4224,4224,4224,4224,4480,4480,
        4736,4992,4480,4992,4992,4736,4992,4992,5248,4736,
        3456,5248,5248,5248,4992,3968,3200,4992,4224,4992]
Kmax = [4224,1705,4224,4224,4224,4224,4224,4224,3753,3753,
        3241,2729,3753,2729,2729,3241,2729,2729,2217,3241,
        3456,2217,2217,2217,2729,3968,3200,2729,4224,2729]
```

Layers other than 9 use the candidate-ratio RMS estimate. Layer 9 uses a
corrected-proxy estimate and reserves 8 records inside `C=4480` for a
top-proxy-raw-square audit; this is not extra candidate traffic. The v2
coordinate index is source-hash-bound, independently reloaded, and stores the
complete RMS and q/C/K policy. Its 216,688,448 bytes plus the 318,924,544-byte
base record artifact total 535,612,992 bytes, or 67.2659% of the dense-Q4
reference as stored data. Storage size is disclosed separately from per-token
cold traffic.

On the declared development corpus, the live native BF16 substitution passes
all frozen thresholds:

| Measure | Development result | Threshold |
|---|---:|---:|
| KL | 0.0044707 | <= 0.05 |
| Top-1 agreement | 0.94921875 | >= 0.90 |
| NLL delta | +0.0013609 | <= +0.05 |
| Final-hidden relative L2 | 0.0498965 | <= 0.10 |
| Mean active fraction | 0.2008072 | <= 0.25 |
| Modeled physical cold traffic | 0.409639 | <= 0.45 |
| Global micro candidate recall | 0.9995917 | >= 0.95 |
| Worst-layer mean recall | 0.9939353 | >= 0.95 |

The candidate recall denominator is a fixed, router-independent per-layer
dense-teacher top-K schedule, not the adaptive selected count. A separate
untimed diagnostic pass computes that reference; the timed sparse pass makes
no dense full-record calls. Python/native route fields and BF16 outputs are
bit-exact for six rows in all 30 layers. The end-to-end development sparse
pass takes 1.1565x the dense elapsed time, so it is not a speedup. The traffic
fraction is modeled from touched 64-byte lines and metadata, not measured
DRAM.

The identical [frozen policy](../reports/native_bitnet_m2_2026-07-26/frozen_dip_policy.json)
then produced the following raw final-holdout measurements:

| Measure | Final raw result | Threshold |
|---|---:|---:|
| KL | 0.0040412880 | <= 0.05 |
| Top-1 agreement | 0.98828125 | >= 0.90 |
| NLL delta | +0.0048289299 | <= +0.05 |
| Final-hidden relative L2 | 0.0477494113 | <= 0.10 |
| Mean active fraction | 0.2138000677 | <= 0.25 |
| Modeled physical cold traffic | 0.4113713394 | <= 0.45 |
| Global micro candidate recall | 0.9994058295 | >= 0.95 |
| Worst-layer mean recall | 0.9939428640 | >= 0.95 |

The slow path took 295.3364 seconds versus 257.9552 seconds dense, or 1.1449x
dense. This is measured CPU latency but was not an upper-bound gate. The
semantic decision is a postmortem adjudication because the original runner's
full-record/object-versus-33-token/list hash verifier defect fired after the
raw evaluator completed. No model or evaluator was executed during
adjudication. The host-bound binaries and artifacts, delayed prospective
evidence seal, modeled rather than measured DRAM traffic, and small 8x32
confirmation scale remain material limitations.

## What the latest experiments changed

### Exact activation-sparse training paths

The latest dense-source experiment removes approximate routing from the
critical path. CATS-style execution reads the full gate matrix, thresholds
its activation exactly, and reads up/down records only for nonzero gates. Its
ideal traffic fraction is `(1 + 2a) / 3`, where `a` is active-record
fraction. At the traffic boundary, zero-shot layer-local error is 0.511 and a
progressive plus fixed-budget boundary fit reaches only 0.470.

The stronger Q-Sparse-style path selects already-resident activation
coordinates directly. It reads `q` input columns of both gate and up and `k`
input columns of down, for ideal traffic `(2q + k) / 3`. The selected integer
point uses 282 of 576 input coordinates and 522 of 1,536 intermediate
coordinates, or 43.967% before metadata. It has no router, candidate stage, or
recall gate. On the representative layer-14 development boundary set, its
error improves from 0.3426 to a best 0.3228 and then plateaus.

Both local screens fail the unchanged 0.18 progression ceiling. The distinct
whole-model Q-Sparse hypothesis was then tested with CUDA used only for
training. A causal single-layer sensitivity fit improved the fixed all-layer
baseline from KL 0.742 to 0.457 at exactly 45% ideal traffic, but verified
attention/normalization co-adaptation moved the unseen result only to KL
0.452, top-1 0.671, NLL +0.458, and hidden L2 0.327. Label-only full-model
continuation, per-token concentration budgets, and a traffic-charged rank-24
residual did not improve the frontier. Every artifact independently reloaded
and executed on CPU; confirmation remained sealed. The dense-source
activation-sparse branch is therefore stopped at the available scale.

### Lossless native-BitNet semantic records

The selected new program starts from Microsoft's natively trained
`bitnet-b1.58-2B-4T` rather than post-hoc quantizing SmolLM2. It is pinned to
revision `04c3b9ad9361b824064a1f25ea60a8be9599b127`; the checked
1,178,623,988-byte safetensors file has SHA-256
`8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b`.
Engram keeps this source family outside the generic SiLU/SwiGLU compiler:
BitNet uses ReLU-squared gating, per-token activation quantization, and an
intermediate RMS normalization that couples channel magnitudes.

The official checkpoint's four-trits-per-byte MLP payload is 50.0521% of the
frozen dense-Q4 denominator, so changing source models alone does not pass.
The new format losslessly stores five base-3 trits per byte. Each channel has
a 1,538-byte logical payload: a 512-byte gate key, 512-byte up key, 512-byte
transposed down value, and a BF16 normalization gain. Those fields live in
four independently fixed-stride phase streams rather than one contiguous
record. Three BF16 projection scales remain layer-global. The complete
30-layer file is
318,924,544 bytes or 40.0527%, with 39,393,536 bytes of headroom. Its logical
source and reconstructed streams have the same SHA-256.

A CPU BF16 evaluation oracle reproduces the decoded artifact exactly. The new
C++ kernel instead memory-maps and executes the packed phase streams directly,
without creating dense weights. Its tiny-fixture integration test is
bit-exact. Official-layer outputs differ from PyTorch dense BF16 by at most
0.00982 relative L2 because their reduction orders differ, so the decisive
test is all-layer causal substitution. On the frozen 8-sequence/256-position
corpus it reaches KL 0.00371, top-1 0.96094, NLL +0.00224, and final-hidden L2
0.04678 while scheduling 40.0527% of dense-Q4 cold bytes. Exact details are in
the [direct-kernel result](../reports/semantic_gate_native_bitnet_2026-07-24/summary.md).

### Budget-native grouped-ternary distillation

The new path fixes the deployment representation before training: all 30 MLPs
remain width 1,536, five ternary coefficients are packed per byte, and each
128-weight group has one non-learned FP16 scale refined by two least-squares
code/scale iterations. The exact 17,173,504-byte file includes codes, scales,
headers, directory entries, and cache-line padding. It uses 43.1353% of dense
ideal-Q4 MLP traffic and leaves 742,400 bytes below the 45% limit.

Training uses an immutable dense teacher and a copied student whose float
master weights exist only for optimization. The forward path uses the exact
hard grouped-ternary decode with straight-through gradients. A
deepest-layer-first transition avoids quantizing all layers at once; later
rungs stay hard throughout. Local MLP, all-layer hidden, direct final-hidden,
confidence-weighted KL, teacher-top-1, and label losses are implemented.
Attention, normalization, and optionally the already-resident tied
embedding/output head may co-adapt without adding MLP cold bytes. Checkpoints
resume across CPU/CUDA, and validation reloads both the MLP binary and any
co-adapted backbone tensors.

The final bounded rung trained on 8,192 fresh sequences containing 1,014,225
input positions. Relative to its preceding head-coadaptation checkpoint, it
closed 63.37% of the remaining KL gap and 62.77% of the NLL gap, but only
31.33% of the top-1 gap and 38.29% of the final-hidden gap. The predeclared
rule required 50% on every metric before spending 3M or 10M positions. The
configuration therefore stops at KL 2.2844, top-1 0.3198, NLL +2.2770, and
hidden L2 0.6036 despite its valid traffic result. Exact hashes and intermediate
rungs are in the
[budget-native result](../reports/semantic_gate_budget_native_2026-07-23/summary.md).

### Compact-model distillation

The final bounded compact-Q4 program used a predeclared 10-million-position
sample of the released SmolLM2 training mixture. The student kept hidden size
576, used 11 MLPs of width 704 and 19 of width 672, executed fake-decoded Q4
weights during training, and serialized all MLP codes, scales, source IDs,
headers, and alignment.

At the frozen 3M checkpoint it had closed 78.2% of the KL gap but only 50.0%
of the top-1 gap and 56.7% of the hidden-state gap. The protocol required at
least 80% closure on every metric before spending the remaining seven million
positions. All four checks failed, so the run stopped without opening formal
development or confirmation data.

This rejects the tested compact architecture, width vector, initialization,
loss, and training budget. It does not prove that a compact model trained from
scratch or distilled on a much larger token budget cannot work.

### Low-bit and structured representations

Post-hoc low-bit representations do not provide the missing bridge:

- full-width Q3 QAT plateaus at layer-local relative L2 0.2174 while consuming
  far more than the traffic budget;
- a traffic-feasible asymmetric ternary plus rank-12 Q4 residual reaches
  0.4938;
- the best strict shared-input-basis rate-constrained artifact reaches 0.3554;
- structured shared-basis dictionaries, source-coordinate block sparsity,
  affine experts, and conditional compact experts all miss their local screens.

These results are why another small bit-allocation or router sweep is not the
default next step.

The final bounded representation campaign tested additional mechanisms at the
same byte edge:

| Screen | Complete traffic | Best layer-14 mean rel-L2 | Progression |
|---|---:|---:|---|
| Four-cycle width-640 recurrent compact Q4 | 44.9293% | 0.308254 | fail; later-cycle cache reuse is not measured |
| Projection-normalized full-width ternary | 41.0013% | 0.631323 | fail |
| Mixed affine LC-VQ | 44.3482% | 0.336396 | fail |
| Unrestricted 128-entry vector codebook | 44.9799% | 0.576865 initial | stopped at initialization guard |
| Mixed LiftQuant-style lifted-binary lattice | 44.4012% | 0.556958 initial | stopped at initialization guard |

The recurrent and affine-LC arms were trained on sequence-disjoint
development-role boundaries. The codebook and lifted-binary arms failed their
predeclared 0.55 initialization guard and were not trained. None reached the
0.20 local ceiling required to expose an expensive causal or external
evaluation. Exact metrics and scratch-report hashes are checked in under the
[budget-edge representation summary](../reports/semantic_gate_lowbit_2026-07-23/summary.json).

### Nonparametric output memory

The most recent branch asked whether the MLP could be represented by stored
input/output examples rather than compressed source weights.

The exact nested local curve was:

| Prototypes | Layer-14 LLE-32 mean relative L2 |
|---:|---:|
| 16,384 | 0.490340 |
| 65,536 | 0.401270 |
| 233,005 | 0.327526 |

Several local-linear variants exposed the same limitation:

- reconstructing a query state from 512 neighbors is accurate (0.0496
  relative L2), but interpolating the nonlinear MLP output remains at 0.3275;
- token-conditioned and two-region token-conditioned Jacobians improve to
  0.2360 and 0.2164, but require tens of GiB before an index or all-layer
  package;
- an exact per-query nearest-prototype Jacobian calculation reaches 0.1725,
  demonstrating a local capacity ceiling, but finite shared Jacobian banks
  regress above 0.22 and are not byte-feasible.

The frozen final pilot then captured exactly one million finite FP16 layer-14
input/output pairs from 8,192 independent pretraining sequences. Combining
them with all 233,005 local fitting records improved exact LLE-32 error only
from 0.327526 to 0.321854. The predeclared progression rule required error at
most 0.28 and at least 10% improvement; the measured improvement was 1.73%.
The ten-million-record capture was therefore not run.

## Milestone position

“Implemented” below means that code and tests exist. It does not mean the
scientific exit criterion has passed.

| Milestone | Implementation status | Evidence status |
|---|---|---|
| 1. Inspection, tracing, exact MLP decomposition, oracle experiment | Complete | Complete for the fixture and exercised on SmolLM2 |
| 2. Semantic package, routing, quantization, Python substitution runtime | Source-bound native-BitNet DIP and OLMoE learned-expert routes, authenticated packages, CPU kernels, native token runtimes, and causal evaluators exist | **Native-BitNet passed** by postmortem adjudication and **OLMoE Q7 formally passed** an authenticated frozen complete-native 8×32 protocol. The matched W128 8×128 control passes every band and attributes the later sustained failure to attention, preserving the Q7/M2 decision. Generic dense-Llama conversion and broader replication remain incomplete |
| 3. Local/recurrent/retrieval attention and hybrid episodic memory | Bounded W=16/C=8/K=4 streaming hybrid, stateful C++20 cache/rerank kernel, incremental package integration, exact per-layer/per-head native policy ABIs, an exact episodic K/V bank, and learned content/phase selectors implemented | The earlier W16 policy fails sustained older-context quality, but the prospective W128/per-vector-INT8 local-cache replay now passes all 1,024 semantic positions at 25% logical attention traffic (protocol `953b83ce…33bda7`, report `7cd55514…d17df80`). This clears the sustained attention-quality boundary. Package metadata/default integration, broader corpora, and end-to-end speedup remain. |
| 4. Shared recurrent controller, adapters, adaptive cycles, transformer-free Python runtime | Versioned exact residual controller, authenticated package installation, persistent native stage state, and a one-call 30-stage C++ attention/semantic runner implemented | **Controller, compiled-substitution, incremental-generation, and C++ orchestration gates pass**; frozen generation reaches 96.875% token agreement, 87.5% exact prompts, correct cache positions, and zero decoder-layer calls |
| 5. Vocabulary index, transition cache, corrections, compiler, validation, generation CLI | Generic infrastructure plus native-BitNet package compiler, validator, native vocabulary argmax, and generation CLI implemented | Native-BitNet package excludes all source MLP tensors and passes source/package parity; generic vocabulary/cache/correction paths are not all active in the promoted native-BitNet runtime |
| 6. C++ runtime, scalar/AVX2 paths, mmap, parity, generation, benchmarks | Fixture runtime, memory-mapped BitNet DIP/projection kernels, streaming attention, native shell operators, authenticated C++ package mapping, manifest-derived model configuration/EOS, token-step control, greedy argmax, reset, standalone generation, and a versioned shared ABI implemented | **Partial**: model execution is native and chat uses the shared handle; tokenizer/template/history orchestration remains Python-side, the compressed INT8 attention path is opt-in, and AVX2 tuning plus hardware-counter traffic remain |
| 7. Evaluation, ablations, tuning, documentation, final report | In progress | Many negative ablations exist; no successful reproducible final report |

The optional Oracle cognitive executive is a separate request-level subsystem.
Its revisioned SQLite/JSONL event stores, worker registry, dispatch adapters,
outcome observation, and calibration summaries are implemented, but they are
independent of the model-worker semantic-gate evidence.

## What is intentionally not claimed

- There is no quality-preserving `.engram` conversion of SmolLM2.
- The native BitNet result is a separate source track, not evidence that a
  dense Llama model can be losslessly repacked.
- There is no hardware-counter demonstration of DRAM traffic; the final
  41.1371% practical-DIP result is a modeled cache-line schedule, not measured
  DRAM.
- The native-BitNet practical DIP arm has passed its semantic gate by
  postmortem adjudication, not by a pristine final-runner result. This is not a
  blanket claim that all generic Milestone 2 packaging and conversion work is
  complete.
- The matched OLMoE W128 control is an attribution diagnostic, not a promoted
  Milestone 3 policy. Its 100% logical attention reads and 35.8 MB state violate
  the deployable bounded-attention objective.
- The OLMoE layer-rescue screen is not a Milestone 3 confirmation. Forty-five
  adaptive comparisons used two sequences, and the six internal-screen
  sequences came from an already consumed development corpus. Its valid
  negative result closes this whole-layer schedule, not every possible
  layer/head-adaptive design.
- The fixed 51-head OLMoE screen is also not a Milestone 3 confirmation. Its
  attention maps came from two development records and its causal metrics from
  six reused records. Its valid negative result closes only the frozen
  teacher-attention-mass ranking.
- The causal/value-sensitive 51-head follow-up is not a Milestone 3
  confirmation either. It fit two development records and reused the same six
  records for its native-Q7 screen. It closes the tested static
  causal/value-sensitive mask, not prefix-conditioned dynamic allocation on a
  new corpus.
- The retrieval-targeted selector is negative Milestone 3 development
  evidence, not a confirmation. Its full 8-record fit selected `M2`, but the
  exact-51 candidate failed KL, NLL-delta, and hidden-state gates on the
  separate 8-record development split. The confirmation file was created and
  hash-bound before execution and remained unopened by the fit path.
- The K2 prefix selector and payload, label-plus-payload, and K51 episodic
  screens are train-only attribution evidence. They cannot promote a policy.
  The completed K64/96/128/165 sweep is also train-only: all four failed, and
  its diagnostic K165 replay is not a promotion.
- The derived DIP package, standalone runtime, and shared chat handle pass
  their integration checks. Python still owns tokenization, template
  rendering, and history, but it no longer constructs or executes a
  Transformers model shell.
- The final holdout is plaintext in the repository. Its separation is
  procedural and honor-system-based, with a fail-closed runner; it is not
  cryptographically hidden from developers.
- Random-fixture generation is pipeline evidence, not language quality.
- CUDA was used for bounded training and trace capture, but no deployment
  format or inference path requires a GPU.
- Failure on SmolLM2-135M is not a theorem about every model or every possible
  architecture.

## Current development decision

For the active OLMoE boundary, Milestone 2 remains passed and Milestone 3
remains blocked. The tested static global family and frozen greedy three-layer
W128 path have failed valid development screens under the 45% logical-read
cap. The completed fixed 51-of-256 teacher-attention-mass mask improved on the
layer rescue but also failed all four overall gates at 44.975387218386625%
reads. The follow-on causal/value-sensitive static mask improved its
two-record training objective but was worse than the attention-mass mask on
all four overall native-Q7 metrics at the same read budget. No fresh
confirmation or package promotion is justified. Those two static objectives
are closed. The retrieval-specific 8/8/8 selector has now also completed: its
training objective passed decisively, but its static exact-51 candidate failed
KL, NLL-delta, and hidden-state development gates while the W128 control
passed. The K2 prefix screen and exact payload, label-plus-payload, and K51
episodic train screens then failed without crossing their systems budgets.
Confirmation remained unopened. The frozen K64/96/128/165 ranked episodic
head-prefix screen then rejected all four candidates; deterministic K165
diagnostic replay passed, but K165 remained worse than the historical K256
all-head payload result. The following fixed-K256 logit-bias screen executed
all four nonzero arms and failed them all; its replayed `gamma=1/2` candidate
was also worse than `beta=0`. The same-state W128-shadow capacity screen that
followed measured ranks 2/4/8 at 40.05%/42.87%/46.93% global residual
recovery. All passed their sequence, block-entry, finite, and positive-layer
conditions, but all missed the frozen 50% global threshold. Replay and every
post-run authentication check passed; confirmation stayed unopened. This
closes only rank-at-most-8 global per-layer output subspaces with oracle
coefficients. The per-head scheduled-source-mass oracle that followed failed
at -0.108912 global recovery despite improving its mass-matching objective.
The subsequent joint output-targeted gamma oracle also failed: its continuous
optimistic relaxation recovered exactly **0.22738059544921096** globally,
with only 1/8 sequences and 0/4 block entries at or above 0.25; direct
discrete replay recovered exactly **0.1997680396822742**, with 0/8 sequences
and 0/4 blocks at or above 0.25. Both had 16/16 positive layers. The joint
protocol/result SHA-256 roots are
`aa03a71e3dd9e1fbb413a7773d57189c41029c26b2f4372b2fb7a26744305d24`
and
`1329a51bac71cb81f44494c8ef70cb23a631eacebac5540b39e2e98ed5e30ea5`.
No predictor, Milestone 3 promotion, or confirmation access was authorized.
The continuous superset failure closes scalar mass retuning at fixed K256.
The exact per-slot successor then optimized the eight episodic values
individually. Its constructible and exact-native-anchor optimistic arms
recovered 0.3844378107 and 0.3844378142 globally. Both passed every
sequence, block-entry, and positive-layer condition, but decisively missed the
0.50 global gate; the optimistic maximum objective-gap bound was
`5.90e-11`. Result SHA-256 is
`2e8e9b7d5f33d33c0e8c642a50359da785d0690d8260ed9f65837b14cd93a5bf`.
This closes the current regular aggregate plus eight episodic values.

That frozen capacity boundary has now completed successfully. It exposed all
28 values already read by the kernel—16 local, four selected older, and eight
episodic—in deterministic order. Constructible C28 recovered 0.6653937751
globally, with minimum sequence/block recovery
0.6447006551/0.6306278392 and 16/16 positive layers. Exact-native-anchor C29
recovered 0.6653865288. All qualification, gate, replay, and post-run
authentication checks passed at the unchanged 33.0305% traffic fraction, and
confirmation stayed sealed. This is a train-only same-state capacity pass. It
authorized a causal 28-logit selector experiment, but the later content and
phase-on-mass screens failed at 0.25422074198 and 0.2618728353 BF16 global
recovery. Neither passed Milestone 3 or authorized native integration,
development, or confirmation.

The shared-controller stage has started with a deployable low-rank design
rather than the original dense FP64 GRU fixture. At width 2,560, the original
two dense GRU kernels would occupy about 629 MB and impose that shared-weight
traffic on every depth cycle. The new rank-128 controller has 2,662,430 FP32
parameters (10,649,720 bytes), an identity-biased residual path, per-token RMS
normalization, 30 stage embeddings, and rank-4 stage adapters.

The next protected run expanded to 128 training and 64 validation positions
from different dataset hashes. It reduced validation terminal normalized MSE
from 1.998608 to 0.245010, cosine loss from 0.973363 to 0.333417, and total
rollout loss from 4.292672 to 1.156465. A lower-rate 500-step continuation
loaded from the serialized artifact and used no teacher forcing. It improved
training terminal error from 0.060627 to 0.029324 but worsened validation
terminal error to 0.260050; that continuation is rejected. Independent NumPy
reload of the retained pre-continuation artifact matches Torch within 7.45e-6
maximum absolute error. Sixty-four states complete 30 CPU cycles at 79.7
states/s in the measured NumPy batch. Exact teacher MLP and attention outputs
are still supplied at the controller boundary, so compiled-operator
substitution remains sealed.

The controller-only prerequisite for substitution is terminal normalized MSE
at most 0.0225, corresponding to the existing hidden relative L2 limit of
0.15. A controlled rank-4 stage input adapter improves the 1,024/256-position
result only from 0.159440 to 0.157431, showing that another learned input
alignment is not the solution.

The host was subsequently rebooted with matching NVIDIA 580.173.02 kernel and
userspace components. An exact CUDA reproduction reaches protected terminal
normalized MSE 0.246530 and cosine loss 0.335160, compared with 0.245010 and
0.333417 for the CPU-optimizer run. Serialized CPU parity passes at 5.36e-6.
CUDA reduces the 500-step fit from 131.8 to 112.6 seconds but does not alter
the scientific decision; the slightly better CPU artifact remains retained.

A subsequent frozen 1,024/256-position rung trained a fresh rank-128
controller for 1,000 CUDA steps. Protected terminal normalized MSE improves
to 0.159440 and total rollout loss to 0.931534, but the artifact still fails
the 0.0225 substitution gate by 7.1 times. The artifact reloads on CPU within
5.90e-6 and processes a batch of 256 states through 30 cycles at 101.3
states/s.

The trace contract exposes a stronger architectural fact: teacher layer output
is the incoming residual plus the captured attention and MLP outputs. A new
schema-v3 controller preserves those additions exactly and reserves the shared
factorized recurrence for corrections only. With correction scales zero, the
protected terminal NMSE is 0.000020801, passing the gate with 1,081.7 times
margin. CPU reload parity is 5.72e-6, and the matrix-skipping NumPy path runs
41,575.9 stage transitions/s. Exact teacher operator outputs are still
supplied, so the next gate is compiled semantic and episodic substitution,
first independently and then jointly.

Operator provenance was then corrected: `NativeBitNetRuntime` already replaces
every MLP with the packaged direct CPU phase-stream kernel, so captured
semantic outputs were compiled. A frozen controller replay now combines those
packed semantic outputs with native W16/C8/K4/S2 attention outputs outside the
decoder residual scaffold. Across eight held-out sequences and 256 prediction
positions it passes every check: KL 0.011125, top-1 agreement 0.957031, NLL
delta -0.008285, final hidden relative L2 0.075893, controller-to-candidate
hidden L2 0.006810, and terminal trajectory NMSE 0.000026666.

The next boundary is no longer another substitution-quality experiment. It is
incremental runtime integration: controller state must directly feed the
native MLP and attention operators, advance RoPE/cache positions, persist
bounded episodic state, and produce logits without decoder-layer forwards.

That incremental boundary now passes. `ControllerDrivenBitNet` explicitly
dispatches all 30 normalized attention/MLP stages and advances schema-v3
controller state while native attention owns persistent cross-token memory.
It carries one residual RMS scalar per token so operator outputs retain their
correct relative scale. Across the fixed eight-prompt suite, all 32 generated
tokens match the bounded decoder reference, cache positions are exact, and
decoder-layer forward calls are zero. Controller arithmetic is only 0.0427
seconds per prompt versus 22.581 seconds complete execution.

Package-native installation and residual execution are now implemented. The
manifest owns and authenticates every controller tensor, generation can select
the installed controller without an external path, and the residual/RMS loop
runs through `libengram_bitnet.so`.

The next native-shell cut is also complete. BF16 embedding lookup, all 92
RMSNorm sites, default RoPE construction/application, and tied-vocabulary
greedy argmax now execute through the C ABI. The vocabulary path returns only
the selected token and no longer allocates 128,256 logits. The four-token
`The capital of France is` smoke test remains ` Paris. Paris is` and improves
from about 21.3 to 18.6 seconds. The frozen eight-prompt/32-token protocol
passes at 96.875% weighted agreement and 87.5% exact prompts, with exact cache
positions and zero decoder-layer calls. Its one fourth-token mismatch is a
BF16 near-tie caused by native scalar versus PyTorch/oneDNN accumulation order,
not a hidden-state or cache gate failure.

The remaining Milestone 4 systems boundary is an all-C++ stage orchestrator.
Python still sequences the 30 stages and creates Torch tensor views for the
already-native packed projections, MLPs, and streaming attention. The next
implementation should load the authenticated non-MLP/controller state into a
single native runtime handle and dispatch the full stage loop without Python
or Torch.

The first part of that orchestrator is now live. A persistent C++ stage-state
handle owns normalized residual state, attention contribution, post-attention
state, and physical RMS, enforces attention-before-semantic call ordering, and
produces BF16 normalized inputs for both operators. Package generation uses
this handle instead of the NumPy state loop and retains ` Paris. Paris is`;
measured controller bookkeeping falls from roughly 26–30 ms to 11.4 ms on
that smoke prompt. The complete suite passes at 452 Python and 16 native
tests. Python still invokes each attention and MLP operator, so this is the
buffer/state foundation for the all-C++ loop rather than completion of it.

The semantic half of each stage is now fused across that boundary.
`engram_bitnet_stage_semantic_bf16` asks the stage handle for its normalized
post-attention input, executes the selected packed phase-stream MLP directly,
records the existing traffic/timing metrics, and inserts the semantic result
back into normalized residual state in one native call. The smoke sequence
remains exact, completes in 18.15 seconds, and reports only 10.4 ms of
controller/orchestration overhead. Python no longer materializes semantic
inputs or outputs. Attention still crosses the Python/Torch boundary and is
the next half to fuse.

The attention half and depth loop are now fused as well. One descriptor-driven
C++ call executes input normalization, packed Q/K/V, RoPE, persistent bounded
attention, attention sub-normalization, packed O projection, packed MLP,
residual insertion, and renormalization for all 30 stages. The unchanged
eight-prompt/32-token protocol passes at 96.875% weighted agreement and 87.5%
exact prompts, with every cache position correct and zero decoder-layer calls.
Mean complete controller runtime is 16.50 seconds and measured
controller/orchestration overhead is 18.0 ms per prompt. This closes the
Milestone 4 systems-orchestration gate. Python still loads the non-MLP
safetensors into a Transformers-shaped holder and drives per-token generation;
moving package loading and generation control into the native runtime is the
next Milestone 6 boundary.

Native package loading has now advanced past its largest storage risk. A strict
read-only safetensors mapper validates the complete header, dtype/shape,
contiguous offsets, payload length, and typed views. It maps the real
780,054,616-byte non-MLP file with all 332 tensors. `NativeBitNetWeights` binds
the 128,256x2,560 tied embedding, final and per-layer norm vectors, and all 120
packed Q/K/V/O projections. The projection kernel now supports explicit
lifetime-bound mapped registration, so the complete 30-layer binding reports
zero copied projection bytes. The next loader step is to construct attention
caches/stage descriptors plus the MLP/controller handles and expose one native
token-step runtime.

That token-step runtime now exists. `NativeBitNetTokenRuntime` owns the mapped
weights, memory-mapped packed MLP artifact and DIP index, validated
zero-correction controller scales, 30 persistent streaming-attention caches,
position counter, final norm, and tied-vocabulary argmax. The standalone
`engram-bitnet-token-generate` executable accepts raw token IDs and performs
greedy generation without Python, Torch, the Python `safetensors` package,
Transformers, or an Engram shared library. It reads the packaged
`transformer/non_mlp.safetensors` file through Engram's C++ parser. Its package
preflight authenticates the promoted manifest, exact inventory, semantic trust
roots, controller, model/tokenizer configuration, attention policy, and EOS
IDs before deriving every runtime architecture parameter. Native
tokenizer/chat-template support remains outside the binary; model execution
is native from packaged token IDs to generated token IDs.

The low-bit-native hypothesis has passed source validation, exact
reconstruction, the unchanged MLP byte limit, direct CPU execution, frozen
causal confirmation, source-independent package compilation, and exact
generation parity. Its package contains a checksummed 780,054,616-byte
non-MLP tensor file plus the 318,924,544-byte packed MLP artifact, installed
controller, config, and tokenizer assets. The derived semantic package adds
the authenticated 216,688,448-byte DIP index and its promotion descriptor.

Milestone 3 development rejected local-only and recurrent-only replacements.
An exact W=16/K=4 hybrid passed semantics but scanned all older keys. Random
sign-LSH then missed too many top keys, and exact page bounds pruned too few
pages. The promoted streaming policy retains 16 local tokens, two sinks, and
six cumulative-attention heavy hitters, exact-reranking eight old keys to four
values. On frozen records 8–15 it passes every semantic threshold: KL 0.01409,
top-1 0.94141, NLL delta −0.00613, and hidden L2 0.08559 over 256 positions.
Its old-context state and reads are constant in context length. At the short
33-token test point it still models at 93.34% of dense KV traffic.

The same state machine now has a stateful C++20 implementation and C ABI.
Randomized 40-token native/NumPy parity passes through eviction and
heavy-hitter replacement. Trained development substitution also passes
quality (KL 0.00528, top-1 0.96875, NLL +0.01239, hidden L2 0.04210). The
standalone native benchmark holds state at 249,248 bytes per layer and reduces
logical reads to 31.29%/8.40%/2.14% of dense at context
128/512/2,048.

Incremental compiled-package integration is now complete. The runtime resets
one persistent native cache per layer and batch item, processes prompt tokens
in order, advances absolute positions during decode, applies normal BitNet
RoPE at those positions, and does not allocate a Hugging Face KV cache.
Full-sequence and uneven-chunk execution are bit-identical for the bounded
operator, and a position discontinuity is rejected. Complete generation at
33/128/256 prompt tokens holds all-layer attention state at 7,477,440 bytes
and uses 86.55%/31.07%/16.35% of dense logical attention reads. It processes
about 0.87/0.98/1.01 input positions per second. Only 5.69/8.97/12.65 seconds
of the corresponding 39.06/131.97/255.64 seconds occur inside packed MLP
calls, so the next systems work should move Q/K/V/output projection and cache
orchestration across the native boundary and avoid a full vocabulary
projection on every decode step. Hardware DRAM counters remain unmeasured.

A fused position-major stream ABI has since reduced prompt-time native calls
from one per token per layer to one per layer. At 256 prompt tokens it changes
elapsed time only from 255.64 to 254.23 seconds (0.55%), so call-loop overhead
is rejected as the main problem. A measured 33-token phase profile assigns
11.60 seconds to Q/K/V projections, 7.71 to output projection, 12.62 to the
vocabulary head, 5.94 to the packed MLP, 0.12 to native attention, and 0.06 to
RoPE. The next implementation should therefore reuse the existing
threaded/base-3 machinery for packed native Q/K/V/O projections, then address
the tied vocabulary head separately.

That packed projection path is now implemented. It retains the official
four-codes-per-byte tensors, shares one 12-thread native kernel across all 120
Q/K/V/O modules, and does not materialize their BF16 matrices. The 33-token
end-to-end run falls from 38.51 to 22.29 seconds, with projection time falling
from 19.31 to 3.01 seconds. A direct 32-position comparison against the
materialized-projection package has KL 0.00394, top-1 0.96875, NLL delta
−0.00037, and hidden L2 0.03532. This is a development semantic pass, not yet a
frozen confirmation. The subsequent frozen 8-sequence/256-position result
passes with KL 0.00548, top-1 0.95703, NLL delta +0.00200, and hidden L2
0.05887. Native projection execution is 111.38 seconds versus 256.56 seconds
materialized on the same confirmation tensor. The projection path is promoted;
the vocabulary head now dominates at 13.00 seconds.

The vocabulary bottleneck was primarily redundant work rather than search
quality: BitNet exposes `logits_to_keep`, but package generation had left it at
zero and projected every prompt position. Requesting the final prompt logit
only preserves exact full-vocabulary selection. The 33-token run falls from
22.29 to 10.16 seconds and vocabulary time from 13.00 to 0.83 seconds. At 256
tokens, total generation falls from 254.23 to 20.72 seconds (91.8%) with
unchanged output tokens, traffic, and bounded state. A bounded vocabulary
index is stopped for generation because it would add recall risk after the
exact head ceased to dominate. The packed MLP now consumes 13.07 of 20.72
seconds and is again the principal target.

## Complete inference validation

The optimized components now pass one combined frozen test. On records 8–15
(8 sequences, 256 positions), packed native MLPs and projections plus bounded
native attention reach KL 0.01315, top-1 0.92969, NLL delta +0.00365, and
hidden L2 0.08436. All thresholds pass.

An eight-prompt, 16-token greedy suite produces recognizable factual,
explanatory, narrative, and procedural text with no identical-token runs and
consistent 7,477,440-byte attention state. The code prompt is not a good code
completion and the testing prompt drifts into a multiple-choice format, so
this proves generation works rather than broad task quality.

Full-prompt versus split-prompt final logits are bit-identical. Reset
generation returns the same tokens and stable cache counters, and EOS
termination is unit-tested. Complete prefill succeeds at 512 and 2,048 tokens
in 24.10 and 81.77 seconds. Peak process RSS is 2.14 and 2.57 GB.

The main performance blocker is still decode speed. The older shell measured
about 5.47 seconds per decoded token; the current native-DIP chat smoke took
5.16 seconds for one generated token after startup. Removing the Transformers
model shell therefore closed an architectural dependency, not the
single-token CPU optimization problem. A dedicated single-row
MLP/projection/vocabulary path remains justified.

An interactive `chat-native-bitnet` CLI now applies the authenticated
packaged tokenizer's template to structured history and re-prefills that
complete history through the DIP-only shared handle on every turn. It
supports history display, reset, clean exit, EOF, and interrupt handling. A
real default-system `Hello` smoke rendered 17 prompt tokens, generated token
`9906` (`Hello`) in 5.16 seconds, and reported 7,477,440 attention-state
bytes. The earlier two-turn poem session remains historical evidence from the
retired Transformers model shell; it has not yet been repeated as a scripted
multi-turn DIP confirmation.

The C++ token-step runtime and its versioned shared ABI now move the
adjudicated semantic operator through both the model-core and chat boundaries.
Package derivation authenticates the policy, adjudication, base records, and
coordinate index; the runtime maps both semantic artifacts and has no dense
MLP object or fallback. The rebuilt non-holdout 8×4 confirmation matches all
reference tokens and reset structure. The standalone binary remains
self-contained; the chat DSO exports only the narrow C ABI and depends on no
other Engram library. The frozen suite stays within W=16, while the 17-token
chat smoke crosses the boundary without constituting a sustained
older-retrieval test. Persistent chat caching, streaming, separately
adjudicated trust roots, measured DRAM traffic, and performance optimization
remain later work.

CUDA remains an optional training accelerator only; the serialized format and
inference mechanism are CPU-native. Repeating IVF, candidate-count,
regularization, prototype-density, small residual, post-hoc bit allocation,
loss-reweighting, or short grouped-ternary continuation sweeps is no longer
supported by the accumulated evidence.

## Current OLMoE candidate locality boundary

The evaluator-only native shadow now records all eight older-cache candidate
Q/K score bands before native top-K truncation. The reset-proven train tensor
has shape `[8, 32, 16, 16, 8, 8]` (records, reads, layers, query heads,
candidates, bands), and is stored under
`work/olmoe_q7/retrieval_episodic_blockwise_qk_candidates_2026-07-31/` with a
separate authenticated manifest. Candidate score-ranked top-4 retains 92.27%
mean candidate softmax mass (77.85% p10); the selected older scores map into
candidate top-4 95.74% of the time. These are train-only locality features,
not causal recall or native slot-membership proof. The next experiment is a
fresh-development candidate/group selector with exact reranking and measured
traffic. Milestone 3 remains blocked.

A record-held-out rank-16 ridge negative-control is now recorded beside the
candidate capture.  It predicts candidate scores from pre-attention hidden
head slices only and reaches 68.93% membership recall (50.0% p10; 13.52% exact
top-4).  Query state alone is insufficient for a pre-read selector, so the
next implementation must add stable key/group side information or a learned
residual key summary before any causal reranking attempt.

The next evaluator primitive is now present: the native shadow route can emit
the exact post-RoPE key vectors for all eight older candidate slots, without
changing attention output or traffic.  The reset-proven train tensor has shape
`[8, 32, 16, 16, 8, 128]` and is stored in the separate
`retrieval_episodic_blockwise_qk_candidate_keys_2026-07-31` artifact.  The
candidate-key manifest is authenticated and the native route replays exactly
after reset; the capture remains train-only and the confirmation split is
closed.  The key manifest SHA-256 is
`9df57182d3e537b241a19b4aa1917f981b0ac5aded128c866a3ec17c207f8620`.

A per-layer/head centered-PCA audit gives the first memory-traffic boundary.
Rank 16 has 2.07% centered reconstruction MSE, 0.99% mean normalized key MSE,
4.50% p95 normalized MSE, and an estimated 6.69% of dense key traffic under a
cached float16-basis/coefficients model.  Rank 32 reduces mean normalized MSE
to 0.14% at 13.33% estimated traffic; rank 64 is effectively lossless on this
corpus.  These are key-fidelity and traffic proxies, not score recall or a
causal gate, because query vectors are not exposed by this evaluator ABI.  The
next boundary is a held-out query/key compatibility selector (with exact
reranking inside selected groups), followed by a fresh causal run only if its
feature recall is strong.

The first held-out query/key compatibility screen is now complete.  A cheap
rank-8 diagonal bilinear router combines centered pre-attention hidden
coordinates with the captured post-RoPE candidate keys and predicts the
pre-top-K scores.  It reaches only 51.13% mean candidate membership recall
(25.0% p10; 6.32% exact top-4) and 51.52% mean oracle-mass retention.  The
exact-score ceiling is 100%, so key availability alone does not repair the
query representation.  This is a reproducible feature-only negative result;
no causal policy changed.  The next justified experiment must expose or learn
the actual per-head query projection (or a distillation target for it) before
another selector claim.

Using the existing authenticated per-head query-feature artifact fixes the
representation problem for a feature-only rank sweep.  After native RoPE,
record-held-out PCA reconstruction of the candidate keys reaches 87.19%,
92.88%, **95.17%**, 96.25%, and 97.17% mean candidate membership recall at
ranks 4, 8, 16, 32, and 64 respectively.  Rank 16 has 81.10% exact-top-4
rows, 75.0% p10 recall, and 92.11% mean oracle-mass retention; dense
query/key scoring is 100% recall.  Rank 16 therefore crosses the 95% mean
feature threshold but is not yet a causal pass: tail recall and exact-row
coverage remain weak, and exact reranking inside selected groups has not been
implemented.  The next boundary is a rank-16 grouped selector with exact
reranking and a fresh development causal replay.

The compression audit SHA-256 is
`ca301b74277b064d84b86ce412af1cc22dafecfbc011f868ae1ab5db318607f1`; the
query/key screen SHA-256 is
`3fec2173f637c54a02e9203c9d03d6aee51637f3e9b2ce60be19284372455c3e`.

The authenticated rank-recall report is
`actual_query_key_rank_recall.json` (SHA-256
`332df4670167a6ca351201419c53906bd8bbed7d3dcfa8bb343e00a610c4b4cd`).

The value-side companion trace is now captured and authenticated.  It has
shape `[8, 32, 16, 16, 8, 128]`, exact first/reset parity, and manifest SHA-256
`8c13a25f1070fc0fba2b032fc7e84a24229aaecbb3fb35c55c51a20414865a1d`.  Values
are copied from the same older-cache candidate slots as the key trace and are
available for an offline causal replay; the trace route remains evaluator-only.

The research-guided query-aware selector then uses rank-16 PCA scores for
candidate generation and exact native Q/K scores for reranking.  With a pool
of six candidates it reaches 99.804% mean membership recall, 100% p10 recall,
99.220% exact-top-4 rows, and 92.266% mean candidate-mass retention.  Pool
eight is the exact-score ceiling at 100% recall.  The authenticated report is
`query_key_exact_rerank_v2.json` (SHA-256
`6567ec5ec272c8d18ff7f661f5f917aae307780ab668e27507cc27e73e0d86e1`).  This
passes the candidate-locality boundary but not the semantic gate: the
remaining 0.78% non-exact rows still need native intervention replay with
complete hidden-state/logit measurements.

The native intervention boundary is now exercised.  The evaluator-only
`forward_episodic_masked` ABI converts each rank-16/pool-6 group into an
older-candidate allow-list, while the native kernel still performs its exact
top-4 Q/K rerank and reads the corresponding cached values.  On all eight
record-held-out train sequences, CPU replay against the identical unmasked
native runtime gives 99.6094% answer-position top-1 agreement, 99.9023% over
all positions, mean answer hidden relative L2 **0.013111**, mean logit relative
L2 **0.006376**, and mean answer NLL delta **+0.002404** (maximum +0.018012).
Every record passes the declared 10% hidden/logit, 0.05 NLL, and 90% top-1
thresholds.  The authenticated report is
`work/olmoe_q7/retrieval_episodic_native_masked_replay_rank16_pool6.json`
(SHA-256
`2e6e017fc507e915cc96948deb53de0112062c208834b899643e04abe587baa7`).

This is the first complete causal replay for the selector, but it is still a
train-only result: the independent development split has not been opened and
the selector remains evaluator-only.  Thus the semantic gate is materially
strengthened, not promoted to a generalization or production claim.  The next
defensible step is a newly authenticated development replay with the same
rank/pool, then measured traffic and long-context scaling if it passes.

The independent development replay now passes as well.  The development
capture was made after the selector was frozen, and PCA bases were fit only on
the train candidate-key/query artifacts.  On all eight development sequences,
the rank-16/pool-6 mask retains **99.8192%** of exact candidate top-4
membership (100% p10); native exact reranking inside the pool has the same
recall.  Full CPU replay reaches **100% answer top-1 agreement**, mean hidden
relative L2 **0.013236**, mean logit relative L2 **0.006520**, and mean answer
NLL delta **−0.003980** (maximum +0.000574).  Every record passes the causal
thresholds.  The development report is
`work/olmoe_q7/retrieval_episodic_development_replay_rank16_pool6.json`
(SHA-256
`0c5cb2273f63b930148c78070da68ae57bb821969a68e2a6a038ee7ac5d04bb6`).

This closes the train-to-development semantic gate for the bounded selector.
The separately authorized protected replay has now passed as well.  The
authenticated opt-in package boundary has now been validated; native defaults
remain selector-disabled.  The long-context CPU scaling boundary is now
complete.  Any change to the ordinary runtime default still requires a
separate production decision and broader end-to-end benchmarking.

The frozen rank-16/pool-6 selector was replayed on development record 0 at
512 and 2,048 positions. The first 128 positions remain the authenticated
selector window; later positions repeat the deterministic token stream with
episodic directives disabled. Answer quality remains 100% top-1 agreement,
with mean hidden/logit relative L2 0.008138/0.003919 and NLL delta −0.001520.
Masked/unmasked logical-read fractions are 0.997079 at 512 positions and
0.999275 at 2,048, with 3.04/3.09 and 3.083/3.085 tokens/s. Peak resident
memory was 6.28 GiB and did not grow with context. Report:
`work/olmoe_q7/retrieval_episodic_long_context_rank16_pool6.json` (SHA-256
`fa205bd2ab4c91de27170247e7669f44c9def8bccea22d94558f8caa4b26bf71`).

The selected train-fitted policy is also serialized as a disabled-by-default
evaluator artifact containing its rank-16 key PCA basis and rank-16/pool-6
geometry. The loader reproduces the development mask tensor bit-for-bit and
fails closed if the artifact is enabled by default or its basis digest changes.
Native package defaults remain unmodified.

The serialized policy has now driven the native masked ABI on development
record 0 with 100% answer top-1 agreement and the expected logical-read
reduction (710,668,288 to 702,124,544 bytes). The opt-in package compiler also
copied and authenticated the policy and PCA basis.

The separately authorized protected replay has now been completed against the
exact protocol-defined split. The split was regenerated from the frozen
tokenizer/seed/generator and matched its expected SHA-256
`c74aa6532e94bfa4dd10bbc1e13c27a06c35f79edbc9bc921a4219409f903baa` before
native traces were captured. Using the required authenticated post-QNorm/
pre-RoPE query representation, the frozen rank-16/pool-6 policy passed all
eight protected records: aggregate answer top-1 **1.000000**, hidden relative
L2 **0.009133**, logit relative L2 **0.004416**, and answer NLL delta
**−0.000460** (maximum +0.005879). Candidate-pool and exact-rerank recall
were 99.8501% mean and 100% p10. Logical attention reads fell 1.1958%.
The report is
`work/olmoe_q7/retrieval_episodic_protected_replay_rank16_pool6_2026-08-03/replay.json`
(SHA-256
`526202b68e2a283482f73a018479f75057435b8576b4784573caa8459b18176a`).
The selector remains disabled by default; explicit package opt-in is now
authorized.

The opt-in package was assembled and validated at
`work/olmoe_q7/package_selector_optin_2026-08-03` (manifest SHA-256
`3c014679f2c626b68f73f8eebadbde8cb2421d4e174d8d69c27ebac774f3383c`).  Native
one-token generation for `Hello` returned token ID 13 (`,`) through both the
opt-in and ordinary packages; the parity record is
`work/olmoe_q7/selector_package_optin_generation_2026-08-03.json`.

The same development replay now records native counters.  Mean logical
attention reads fall from **710,667,264** bytes unmasked to **702,166,336**
bytes masked (**98.8038%** retained, **1.1962%** reduction); older candidate
entries scored fall from **222,208** to **205,824** (**7.3733%** reduction).
This confirms that the semantic win is not yet a large end-to-end traffic win:
episodic value reads and the dominant local/projection work are unchanged.
