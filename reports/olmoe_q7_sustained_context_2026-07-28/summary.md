# OLMoE Q7 sustained-context gate, attribution, and static rescue experiments

Date: 2026-07-28

## Outcome

The prospectively frozen sustained-context gate **failed semantic quality** with
the deployable bounded-attention policy `W16/C8/K4/S2`. The run was authentic,
reproducible, structurally exact, and within both traffic limits, but the model
drifted sharply after position 31. This is a real gate failure, not an
infrastructure failure.

A second protocol was frozen after that failure and before its own execution.
It changed only the local attention window from 16 to 128, making attention
exactly causal over this 128-position experiment while retaining the identical
package, native library, Q7 artifact and policy, corpus, teacher arrays, and
12-thread CPU runtime. This matched W128 control passed every overall and
per-band semantic threshold. Its first 16 offsets were exactly identical to the
bounded run, as required by the manipulation check.

The defensible attribution is therefore:

> Bounded attention is the dominant source of sustained-context drift on this
> frozen corpus. The Q7 MoE path is compatible with the semantic thresholds
> when attention is exact.

This does **not** pass the deployable attention gate. W128 consumed 100% of the
dense logical attention reads and was specified after the W16 failure as an
attribution diagnostic, not as a replacement gate. It also does not establish
task-sensitive retrieval quality beyond this fixed corpus.

The subsequently frozen exact-traffic-matched development sweep also
completed. All three <=45%-read policies passed every evidence check, but zero
arms passed semantic quality. The frozen rule therefore selected no policy:
there is no defensible “best failure” to promote.

The following layer-adaptive experiment is now complete as well. An exact
old-scalar/new-layered parity prerequisite passed, and a frozen 45-candidate
greedy search selected layers 11, 6, and 10 for W128 rescue. The resulting
44.1701%-read schedule passed every execution and authentication check but
failed all four overall semantic metrics on its six-sequence internal screen;
all four metrics also failed in every band from position 32 onward. No package
policy was promoted and no fresh confirmation was run. The combined evidence
preserves the Milestone 2 Q7 pass, closes the tested global policy family and
this frozen greedy three-layer path under the current cap, leaves interacting
layer combinations untested, keeps Milestone 3 blocked, and motivated the
fixed teacher-guided 51-of-256 head-mask experiment below.

That head-wise experiment is now complete. A dense untouched teacher exposed
attention maps for only the two development-selection records, and a frozen
ranking selected 51 of 256 layer-head pairs for `W128/C8/K4/S2` rescue. The
other 205 heads retained `W16/C8/K4/S2`. Exact all-base parity, the fixed-mask
resource contract, six-sequence execution evidence, reset replay, and all
post-run authentication checks passed. Semantic quality nevertheless failed:
overall KL was 0.073719930, top-1 agreement 0.867187500, target-NLL delta
+0.053455543, and hidden relative L2 0.167517818. No fresh confirmation was
run.

The final static experiment replaced attention-mass ranking with gradients
from a causal/value-sensitive objective. The frozen CPU trainer kept exact
native sparse and W128 forward results, used differentiable fixed-support
surrogates only for backward propagation, and projected two averaged-gradient
IHT steps to exactly 51 heads. The selected `M1` mask improved the two-record
maximum/mean composite objective from 7.8671169/6.9172161 to
4.7559915/4.3284769, with no record regression. The fit took 6,930.099 seconds.
Nevertheless, its one complete-native six-record screen was worse than the
attention-mass mask: overall KL/top-1/NLL-delta/hidden-L2 were
0.07913208059/0.8645833333/+0.08119899696/0.18264718059. Evidence and exact
44.9753872%-read resources passed; semantic quality did not. No confirmation
or promotion occurred. This closes the two tested static objectives, not every
possible static selector. The next semantic experiment is Q7-aware,
retrieval-targeted training on a new synthetic corpus; prefix-conditioned
dynamic allocation follows if it cannot pass.

## Frozen experimental contract

- Corpus: eight newly Engram-authored, distinct-domain development texts fixed
  without inspecting teacher or candidate outputs and without reusing prior
  Engram calibration or confirmation text. This is a prospective development
  corpus, not an independent external benchmark or cryptographically hidden
  holdout.
- Population: 129 tokens per sequence, 128 prediction positions per sequence,
  1,024 prediction positions total.
- Candidate: authenticated native OLMoE package, CPU only, 12 threads, with no
  Transformers model shell.
- Bounded policy: local window 16, eight older candidates, exact reranking of
  four older entries, and two sink tokens (`W16/C8/K4/S2`).
- Control policy: `W128/C8/K4/S2`; because every evaluated offset is below 128,
  no eviction or approximate retrieval occurs.
- Quality thresholds, applied to the overall population and every frozen band:
  mean teacher-to-native KL at most 0.05, teacher top-1 agreement at least
  0.90, mean target-NLL delta at most +0.05, and mean final-hidden relative L2
  at most 0.10.
- Deployability thresholds: Q7 scheduled traffic and logical attention reads
  must each be at most 45% of their stated dense reference. These are
  algorithmic byte counters, not hardware-counter measurements.

## Authentication roots

All hashes are SHA-256 and are written in full.

| Object | SHA-256 |
|---|---|
| Frozen W16 protocol | `82189276ed0e555c2737f4842b1d1ed625f54d9ceaa2c63fe41fe71c5c6eb599` |
| W16 result | `673523c29b12154f98916b8ce6f203b4967842e4bcae8f5c02ad4d197aab97eb` |
| Frozen W128 control protocol | `1619cd5f3cb607a7d0e2b5cde2e61a83dba3f1615884462a30570d62c7764dd9` |
| W128 control result | `3d099ffd3121e47bdf61ed8772e5e9d08b01b8c6041e9a963b409a502808d345` |
| W128 control evaluator source | `3dd8ca0be3c64fb8fb5f3971f5b073b30febc09e8be9f0ed277337522f39714d` |
| Frozen matched-sweep protocol | `2853de54119f4218c165ebebfe560162f76f99b552fdfe84c803a5ca8acfcef0` |
| Matched-sweep result | `813bac5b1d38af7653cf49d8c7b7ca278df8aac5402fdd28692e905bebfc7658` |
| Matched-sweep source commit | `102bda2` |
| Matched-sweep evaluator source | `cf2e4be0bc4d8e6da54aebcb11b94e7c4ecde2d56e12831fe8de835a342ffa60` |
| Frozen layer-rescue protocol | `9514e90bd5d14ae01ea27185763e5a833d4f1963e6bffd0ec0c81848f35b0c3e` |
| Layer-rescue result | `97ce800bd855c1f16248cada696936c7c56acd49c02d6f1c9ce9885dc44f7c49` |
| Layer-rescue source commit | `708782b` |
| Layer-rescue evaluator source | `77dafe8fc1fb6ca317ad7b99d5d86122e26b94b477f5befcf6184ce14080dff0` |
| Layered candidate native DSO | `fe4dfdcc7e87a3cd5e36074e07d297f838ba345c37e939eeb0d796cb39cce409` |
| Frozen head-wise trace protocol | `47f2c6bd7d467130ac492e7a5dc35b05b95ac0da5e71567a508951bd9754ad05` |
| Head-wise trace metadata | `0445220b967be99208ee703bfd12a421de1a0721ff51b1d6404bc3db5305e08a` |
| Dense attention-map NPZ, retained in work tree | `06e72ff9e9a03b58afd2197f5f40cd4e673867ecd5c566e038ce7e3dc8e38a55` |
| Teacher-guided head-mask file | `dbb220384bade6793950d92a88528de3039c8efdb3b1e65964d368abaac90f48` |
| Head-mask identity | `18854c256af3fc68326a2e9fa9173d943db838c116523d5c4057e3f1efe9c278` |
| Frozen head-wise screen protocol | `b863a6620f269bfe1dafec023c2e9742d9605510b113174b0eadbf64dc5cc850` |
| Head-wise screen result | `16bc2f8c11751612023145a36ace32b44bd082b77179a3c5753cb081424daa06` |
| Head-wise candidate native DSO | `cb72b31e7afbf9b9986f1ed107ca2b0d893947aac2c96b58b95298e2bfc12d36` |
| Head-wise evaluator source | `303862d4f2151f6c554fd3605c46100603fb9d37de44537bf866d1e892a9fe00` |
| Causal-gate source commit | `483c62f` |
| Causal-gate evaluator source | `442169060860257e78bbc0068bfdf9e5cf6edd93ff2b392c75ed333687765590` |
| Frozen causal-gate training protocol (`causal_head_gate_protocol.json`) | `037ebfd7d4e40af898ece7f353654eb8a41dc1883f191cbdf05fc34bf50bf4ba` |
| Causal-gate training result (`causal_head_gate_training.json`) | `bacb0e31899f514a8b2b517987566e8bca68d39cabfd50b3c9e7ecf83bc756ea` |
| Frozen causal-gate screen protocol (`causal_head_gate_screen_protocol.json`) | `282bfe0b9e1da86577f0187112a4a444b0f36d7f84e10f4f9bb67730676807c2` |
| Causal-gate screen result (`causal_head_gate_screen_result.json`) | `437d0de4ce4da37e69ca13279b76627d6f7721e766b8f1b4371fb318e7cbeb59` |
| Causal-gate raw streaming-attention DSO | `153e91d9d1fdb964b678eec0f22498d397888781edfe2d531eda8933c3fe87c5` |
| Native package manifest | `861e9cc472f9e1245db5d64e9253411d0b656a0f08df2f58264e9c708ed750db` |
| Native runtime library | `4cd4de8f3e3cefad59d7b9e6e23a0d1d06a26abc10af2e0c4f9242a2b5876ca7` |
| Sustained-context dataset | `0fb513ea29bdae760b91932d60cf942df047ec5ce578d1c21bbf9438a777abeb` |
| Corpus manifest | `e59beabe4cf526c69e531a12eb61ce34cfa2cc8378c3c059ec6e61192d781ab4` |
| Teacher reference JSON | `6b9d103006cee6b4ab92c016a2e1c57c68d1cc01f3837b24c1972e38a5e5cc37` |
| Teacher arrays NPZ | `b5798f662148b1b338ad0ae04c7bf72a9b3c4e04fa86262edad0183de0133009` |

Both runs re-authenticated the package, native library, dataset, corpus
manifest, teacher JSON and arrays, teacher config/index/shards, and frozen
evaluator sources after execution. The W16 run also re-authenticated its
protocol; the control re-authenticated both protocols, the W16 result, and its
own evaluator source. The sweep additionally re-authenticated both prerequisite
protocols and results, its evaluator source (committed at `102bda2`), all
package/DSO/corpus/teacher roots, per-arm counter streams, and its own protocol
after executing all arms. The layer-rescue result additionally re-authenticated
the historical and layered DSOs, every prerequisite protocol and result, the
complete 26-file current execution-source inventory, the six teacher shards,
its own frozen protocol, and its evaluator source after all 45 candidates,
the internal screen, and reset replay. Every recorded post-run authentication
check passed. The head-wise trace additionally authenticated that exactly two
selection records, and no internal-screen records, contributed dense attention
maps. The head-wise screen bound the trace protocol, metadata, map array, mask,
new DSO, complete source inventory, split, and all inherited roots before
execution, then re-authenticated 27 post-run roots. Every head-wise evidence,
parity, replay, resource, and authentication check passed.
The causal-gate trainer additionally bound the exact raw streaming-attention
DSO, evaluator and 33-file source inventory, installed Transformers
implementation, two-record gradient population, loss formula, complete
native-oracle diagnostics, two IHT transitions, and all three executed masks.
Its result revalidated every stored gradient, projection, mask hash, record
identity, loss reduction, native metric contract, and 31 post-training roots.
The screen then bound the selected `M1` mask before reading the six reused
screen records and passed all seven evidence checks, all six resource checks,
and all 32 post-run authentication checks.

## Semantic results

The status column applies all four semantic thresholds to the indicated
population.

| Policy and population | Positions | Mean KL | Top-1 agreement | Target NLL delta | Hidden rel. L2 | Status |
|---|---:|---:|---:|---:|---:|---|
| Threshold | — | <=0.050000 | >=0.900000 | <=+0.050000 | <=0.100000 | — |
| W16 overall | 1,024 | 0.143577622 | 0.802734375 | +0.159292411 | 0.238260451 | **fail** |
| W16 offsets 0–15 | 128 | 0.011373728 | 0.945312500 | +0.000667410 | 0.055917450 | pass |
| W16 offsets 16–31 | 128 | 0.008251669 | 0.937500000 | -0.003416884 | 0.075654702 | pass |
| W16 offsets 32–63 | 256 | 0.083856738 | 0.828125000 | +0.075577248 | 0.218544263 | **fail** |
| W16 offsets 64–95 | 256 | 0.223421679 | 0.753906250 | +0.238443519 | 0.314587320 | **fail** |
| W16 offsets 96–127 | 256 | 0.257219374 | 0.687500000 | +0.324523613 | 0.354124144 | **fail** |
| W128 control overall | 1,024 | 0.003438119 | 0.974609375 | +0.001458613 | 0.041389158 | **pass** |
| W128 offsets 0–15 | 128 | 0.011373728 | 0.945312500 | +0.000667410 | 0.055917450 | pass |
| W128 offsets 16–31 | 128 | 0.001968802 | 0.976562500 | +0.003675862 | 0.038717562 | pass |
| W128 offsets 32–63 | 256 | 0.002101624 | 0.992187500 | +0.002262885 | 0.039487719 | pass |
| W128 offsets 64–95 | 256 | 0.002359753 | 0.968750000 | +0.006798401 | 0.038475771 | pass |
| W128 offsets 96–127 | 256 | 0.002619835 | 0.976562500 | -0.005398471 | 0.040275633 | pass |

The W16 maximum per-position KL was 3.036844015 and its 95th-percentile KL was
0.503347072. Under W128 these fell to 0.467020452 and 0.008838504,
respectively. Overall, W128 reduced mean KL by 0.140139503, improved top-1
agreement by 0.171875, reduced target-NLL delta by 0.157833798, and reduced
hidden relative L2 by 0.196871293.

Offsets 0–15 comprise 128 cross-sequence positions. All position-level metrics
for those offsets matched exactly between W16 and W128, proving that the
control had no effect before the first possible W16 eviction. The shared first
top-1 mismatch at sequence 0, offset 0 is consequently attributable to the
unchanged Q7/native path rather than the attention intervention.

## Exact structural and traffic evidence

The following values are per 128-position sequence. Each deterministic counter
matched exactly for all eight sequences.

| Counter | W16 bounded | W128 control |
|---|---:|---:|
| Positions processed | 128 | 128 |
| Persistent attention state | 6,336,512 bytes | 35,825,664 bytes |
| Attention scratch | 3,840 bytes | 18,176 bytes |
| Eviction events | 1,792 | 0 |
| Older candidate entries scored | 222,208 | 0 |
| Older entries selected for exact values | 113,152 | 0 |
| Sink insertions | 512 | 0 |
| Heavy-hitter updates | data-dependent; 1,536–28,160 allowed | 0 |
| Local/exact KV reads | 505,413,632 bytes | 2,164,260,864 bytes |
| Candidate-key reads | 113,770,496 bytes | 0 |
| Selected-value reads | 57,933,824 bytes | 0 |
| Total logical attention reads | 677,117,952 bytes | 2,164,260,864 bytes |
| Dense logical attention reference | 2,164,260,864 bytes | 2,164,260,864 bytes |
| Logical attention fraction | 31.2863372093% | 100% |
| Q7 scheduled bytes | 93,952,409,600 bytes | 93,952,409,600 bytes |

The eight observed W16 heavy-hitter update counts were 4,085, 4,016, 4,352,
4,015, 4,063, 4,086, 3,901, and 4,172; all were inside the analytically
permitted range. Q7 scheduled 751,619,276,800 bytes across the eight sequences,
22.7864583333% of the all-expert ideal-Q4 reference, in both runs. W16 therefore
passed both 45% traffic limits, whereas W128 intentionally violated the
attention limit.

Every W16 and W128 per-sequence structural check passed: cache position,
positions processed, attention state and scratch sizes, eviction, candidate,
selection, sink, heavy-hitter, logical-read, and Q7-byte counters. Reset and
replay of sequence 0 reproduced top-1 tokens, diagnostic hashes, metrics, and
structural counters exactly in both experiments. Consequently:

- W16: evidence passed, resource checks passed, quality failed, gate failed.
- W128: evidence passed and all quality checks passed, but it was a
  post-failure attribution control at dense attention traffic and was never a
  deployable gate candidate.

For context, the eight W16 sequence executions took 362.225 seconds of wall
time, including 354.546 native seconds and 287.181 Q7 seconds. W128 took
371.321 seconds of sequence wall time, including 363.938 native seconds and
290.956 Q7 seconds. These runs establish correctness and attribution; their
logical byte counts are not direct DRAM measurements.

## Matched attention sweep under 45%

The prospectively frozen development sweep executed all three policies in its
predeclared order. Each arm read exactly 968,753,152 logical attention bytes
per sequence (44.7613856589% of the dense reference), exposed 32 values per
mature step, and scheduled the unchanged Q7 path at 22.7864583333% of the
all-expert ideal-Q4 reference:

| Policy | State bytes | Mean KL | Top-1 | NLL delta | Hidden L2 | Evidence | Quality |
|---|---:|---:|---:|---:|---:|---|---|
| `W16/C18/K16/S2` | 8,991,232 | 0.06388655 | 0.8671875 | +0.05170082 | 0.15771664 | pass | **fail** |
| `W24/C10/K8/S2` | 8,973,824 | 0.06591232 | 0.8779297 | +0.05847984 | 0.15975482 | pass | **fail** |
| `W30/C4/K2/S2` | 8,960,768 | 0.09581344 | 0.8408203 | +0.07572840 | 0.18842230 | pass | **fail** |

The frozen band results show where the shared failure begins:

| Policy | Positions | Mean KL | Top-1 | NLL delta | Hidden L2 |
|---|---:|---:|---:|---:|---:|
| W16/C18/K16 | 0–15 | 0.011374 | 0.945312 | +0.000667 | 0.055917 |
| W16/C18/K16 | 16–31 | 0.001969 | 0.976562 | +0.003676 | 0.038718 |
| W16/C18/K16 | 32–63 | 0.026165 | 0.898438 | +0.015436 | 0.116066 |
| W16/C18/K16 | 64–95 | 0.097770 | 0.828125 | +0.054984 | 0.213969 |
| W16/C18/K16 | 96–127 | 0.124940 | 0.781250 | +0.134212 | 0.253515 |
| W24/C10/K8 | 0–15 | 0.011374 | 0.945312 | +0.000667 | 0.055917 |
| W24/C10/K8 | 16–31 | 0.001969 | 0.976562 | +0.003676 | 0.038718 |
| W24/C10/K8 | 32–63 | 0.026789 | 0.925781 | +0.017731 | 0.123064 |
| W24/C10/K8 | 64–95 | 0.098504 | 0.859375 | +0.085019 | 0.208041 |
| W24/C10/K8 | 96–127 | 0.131684 | 0.765625 | +0.128998 | 0.260596 |
| W30/C4/K2 | 0–15 | 0.011374 | 0.945312 | +0.000667 | 0.055917 |
| W30/C4/K2 | 16–31 | 0.001969 | 0.976562 | +0.003676 | 0.038718 |
| W30/C4/K2 | 32–63 | 0.039842 | 0.890625 | -0.008932 | 0.150861 |
| W30/C4/K2 | 64–95 | 0.145345 | 0.796875 | +0.092438 | 0.251079 |
| W30/C4/K2 | 96–127 | 0.191395 | 0.714844 | +0.217236 | 0.304432 |

Every arm passed source/artifact authentication, exact structural and traffic
counters, reset replay, post-run rehashing, and its exact pre-eviction identity
check against W128. The identity populations were 128, 192, and 240 positions
for W16, W24, and W30 respectively. All three arms passed every quality
threshold in the 0–15 and 16–31 bands. Final-hidden drift failed at 32–63 for
all three; failures were broad across KL, top-1, NLL, and hidden state in the
64–95 and 96–127 bands.

The systems counters also matched the frozen analytical table exactly:

| Policy | State bytes | Scratch bytes | Evictions | Older keys scored | Older values selected | Sink insertions | Heavy-hitter updates |
|---|---:|---:|---:|---:|---:|---:|---:|
| W16/C18/K16 | 8,991,232 | 7,424 | 1,792 | 476,928 | 428,032 | 512 | 4,096–28,160 |
| W24/C10/K8 | 8,973,824 | 5,888 | 1,664 | 254,720 | 205,824 | 512 | 2,048–26,112 |
| W30/C4/K2 | 8,960,768 | 4,736 | 1,568 | 98,816 | 49,920 | 512 | 512–24,576 |

The three arms took 1,183.50 seconds together. Their eight-sequence wall
times were 352.90, 352.73, and 345.54 seconds, with deterministic replay times
of 41.60, 41.03, and 42.53 seconds. Summed native/Q7 times over the eight
measured sequences were 345.13/277.55, 347.57/279.76, and 340.30/272.77
seconds. These are controlled run timings, not a hardware-counter benchmark.

The experiment used the raw native token runtime to override the attention
policy because the authenticated package is immutably bound to W16/C8/K4/S2.
That intervention was explicit in the protocol and did not mutate the package.
The sweep consumed the already designated sustained-development corpus; since
no arm passed, the separately sealed fresh-confirmation corpus remains unused.

The predeclared ranking admitted only evidence-valid arms that passed overall
and every frozen position band. Its eligible set was therefore empty,
`selected_arm` is null, and the decision is
`investigate_layer_adaptive_or_learned_selector`. Selecting the numerically
least-bad failure after observing these results would violate the protocol.

Ordinary global window/candidate reallocation under the 45% ceiling is now a
closed development branch for this policy family. This result motivated the
prospectively frozen whole-layer test below. W128 remains the diagnostic
ceiling, Milestone 2 remains passed, and Milestone 3 remains blocked.

## Greedy three-layer dense-attention rescue

The next prospectively frozen development protocol tested a layer-adaptive
upper bound. It used the new per-layer native attention ABI to keep 13 layers
at the base `W16/C8/K4/S2` policy while changing exactly three layers to
`W128/C8/K4/S2`. It did not change the Q7 artifact or policy, construct a
Transformers model shell, or mutate the authenticated package manifest.

The already-consumed eight-sequence sustained-development corpus was split
deterministically by ascending `sha256(utf8(record_id))`, then `record_id`, then
original sequence index. The first two records were used for greedy selection;
the remaining six were kept output-blind until the final internal screen. The
split identity was
`c267dd96c121b5baf9d229b4e6a2a880f396361ae9565020813d7e2e279ed310`.
This is an auditable development split, not a fresh or independent holdout.

### Authentication and parity

The implementation was frozen at source commit `708782b`. The new layered DSO
was copied to an immutable hash-named path before the protocol was frozen.

| Object | SHA-256 |
|---|---|
| Frozen layer-rescue protocol | `9514e90bd5d14ae01ea27185763e5a833d4f1963e6bffd0ec0c81848f35b0c3e` |
| Layer-rescue result | `97ce800bd855c1f16248cada696936c7c56acd49c02d6f1c9ce9885dc44f7c49` |
| Layer-rescue evaluator source | `77dafe8fc1fb6ca317ad7b99d5d86122e26b94b477f5befcf6184ce14080dff0` |
| Layered native DSO | `fe4dfdcc7e87a3cd5e36074e07d297f838ba345c37e939eeb0d796cb39cce409` |
| Deterministic record split | `c267dd96c121b5baf9d229b4e6a2a880f396361ae9565020813d7e2e279ed310` |

The mandatory pre-candidate parity run compared the historical scalar DSO with
an all-base policy opened through the new layered DSO over a complete
128-position sequence. Tokens, normalized hidden states, complete logits,
cache positions, deterministic counter streams, and historical diagnostic
hashes all matched exactly. Every one of its twelve parity checks passed.

The search then passed all exact evidence checks: all 45 candidate contracts,
the `16/15/14` candidate counts, all three round-resource contracts, the final
schedule and traffic contract, the six-sequence screen population and counter
checks, deterministic reset replay, and all 21 post-run authentication roots.
The result is therefore an authenticated semantic failure rather than an
infrastructure, ABI, artifact, or counter failure.

### Frozen search and selected schedule

Every candidate in a round was executed before the frozen scoring rule selected
a winner. No early stop or score adaptation occurred:

| Round | Candidates evaluated | Winning layer | Cumulative rescued layers |
|---|---:|---:|---|
| 1 | 16 | 11 | 11 |
| 2 | 15 | 6 | 11, 6 |
| 3 | 14 | 10 | 11, 6, 10 |
| Total | **45** | — | layers 6, 10, and 11 |

The resulting schedule has 13 `W16/C8/K4/S2` layers and three
`W128/C8/K4/S2` layers. Its frozen per-sequence analytical and observed
resource contract was:

| Resource or deterministic counter | Exact value |
|---|---:|
| Positions processed | 128 |
| Persistent attention state | 11,865,728 bytes |
| Attention scratch | 6,528 bytes |
| Local/exact KV reads | 816,447,488 bytes |
| Candidate-key reads | 92,438,528 bytes |
| Selected-value reads | 47,071,232 bytes |
| Total logical attention reads | 955,957,248 bytes |
| Dense full-context logical-KV reference | 2,164,260,864 bytes |
| Logical attention fraction | 44.1701489826% |
| Eviction events | 1,456 |
| Older candidate entries scored | 180,544 |
| Older entries selected for exact values | 91,936 |
| Sink insertions | 416 |
| Permitted heavy-hitter updates | 1,248–22,880 |
| Q7 scheduled bytes | 93,952,409,600 bytes |
| All-expert ideal-Q4 Q7 reference | 412,316,860,416 bytes |
| Q7 traffic fraction | 22.7864583333% |

These byte counts are deterministic logical native-interface counts, not
measured DRAM transactions.

### Six-sequence internal screen

The selected schedule was evaluated on 768 prediction positions from the six
records whose outputs were not used during greedy selection. The inherited
quality thresholds applied to the overall population and every position band:

| Population | Positions | Mean KL | Top-1 agreement | Target NLL delta | Hidden rel. L2 | Status |
|---|---:|---:|---:|---:|---:|---|
| Threshold | — | <=0.050000 | >=0.900000 | <=+0.050000 | <=0.100000 | — |
| Overall | 768 | 0.102320950 | 0.845052083 | +0.116775650 | 0.206036865 | **fail** |
| Offsets 0–15 | 96 | 0.012377540 | 0.947916667 | -0.006711043 | 0.057957455 | pass |
| Offsets 16–31 | 96 | 0.006325668 | 0.979166667 | +0.010194634 | 0.072582537 | pass |
| Offsets 32–63 | 192 | 0.065331080 | 0.854166667 | +0.072611753 | 0.193418547 | **fail** |
| Offsets 64–95 | 192 | 0.147380969 | 0.796875000 | +0.150818163 | 0.261827487 | **fail** |
| Offsets 96–127 | 192 | 0.187220146 | 0.765625000 | +0.241930887 | 0.303631431 | **fail** |

Offsets 0–15 and 16–31 passed all four thresholds. The overall population and
each of the three later bands failed all four. The overall maximum
per-position KL was 2.818039656 and p95 KL was 0.367445458.

The fully authenticated command took 4,443.916 seconds. Of that,
3,938.180 seconds were the primary sequence executions for 45 candidates,
86.643 seconds were the layered/scalar parity run, 260.321 seconds were the
six-sequence internal screen, and 42.033 seconds were deterministic holdout
reset replay.

### Decision and next boundary

This is a valid negative result. The internal screen failed, so no fresh
eight-sequence confirmation was run and the development-only layer schedule
was not integrated into or promoted as a package policy. The experiment closes
this frozen greedy three-layer `W128` path under the 45% attention-read
ceiling; as disclosed prospectively, greedy selection can miss interacting
layer combinations. Milestone 2's authenticated Q7 conclusion remains passed
and unchanged; Milestone 3 remains blocked on deployable bounded attention.

That result motivated a teacher-guided fixed head mask. Exactly 51 of the
model's 256 layer-head pairs could be rescued while reading 973,384,704 logical
bytes per sequence, 44.9754% of dense attention. Rescuing 52 pairs would
require 979,193,856 bytes, 45.2438%, and therefore exceed the frozen cap.

## Teacher-guided 51-head rescue

The next prospective protocol was split into two independently frozen phases.
First, the untouched dense BF16 teacher captured eager attention maps from only
the two established development-selection records. The persisted float32
tensor had shape `2 x 16 x 16 x 128 x 128`; every shape, finiteness,
non-negativity, causal-triangle, row-normalization, source, and artifact check
passed. Capture took 365.847 seconds.

The 33.6 MB map array is intentionally not duplicated into this report
directory. It remains at
`work/olmoe_q7/sustained_2026-07-28/headwise_trace.npz`, with SHA-256
`06e72ff9e9a03b58afd2197f5f40cd4e673867ecd5c566e038ce7e3dc8e38a55`.
The much smaller trace protocol, trace metadata, deterministic mask, screen
protocol, and screen result are archived alongside this summary.

For each head and each query offset 16–127, the frozen ranking measured dense
teacher attention mass on keys older than the 16-token local window that could
not be retained by an ideal four-key older selection. Scores accumulated in
float64. Heads were ranked by descending total deficit, then descending
minimum record-by-band mean deficit, then ascending layer and head. This
selected one immutable 51-head prefix without inspecting any attention map or
candidate output from the six internal-screen records. Attention mass is a
teacher attribution heuristic, not a causal or value-sensitive guarantee.

### Head-wise parity and exact resources

Before the fixed-mask candidate could run, the new per-head ABI had to prove
that 256 independent all-base head engines were semantically identical to the
authenticated layered all-base runtime over a complete 128-position sequence.
Tokens, normalized hidden states, full logits, cache positions, deterministic
logical/Q7 counters, and archived layered diagnostic and counter hashes all
matched. Every parity check passed.

The selected policy rescued 51 heads with `W128/C8/K4/S2` and retained 205
heads at `W16/C8/K4/S2`. Its exact per-sequence contract was:

| Resource or deterministic counter | Exact value |
|---|---:|
| Positions processed | 128 |
| Persistent attention state | 12,284,864 bytes |
| Attention scratch | 107,136 bytes |
| Local/exact KV reads | 835,887,104 bytes |
| Candidate-key reads | 91,105,280 bytes |
| Selected-value reads | 46,392,320 bytes |
| Total logical attention reads | 973,384,704 bytes |
| Dense full-context logical-KV reference | 2,164,260,864 bytes |
| Logical attention fraction | 44.9753872184% |
| Eviction events | 22,960 |
| Older candidate entries scored | 177,940 |
| Older entries selected for exact values | 90,610 |
| Sink insertions | 410 |
| Permitted heavy-hitter updates | 1,230–22,550 |
| Q7 scheduled bytes | 93,952,409,600 bytes |
| Q7 traffic fraction | 22.7864583333% |

The six observed heavy-hitter update counts were 3,490, 3,497, 3,576, 3,792,
3,480, and 3,309, all within the analytical range. These are logical
native-interface byte counts rather than measured hardware DRAM transactions.
At this horizon, 52 rescued heads would read 45.2437999637% and violate the
45% cap. `W128` is full-context only for this 128-position protocol; it remains
bounded beyond that horizon.

### Six-sequence internal screen

The fixed mask was evaluated once on the six records not used to rank heads,
covering 768 prediction positions. These records had already been consumed by
earlier diagnostics, so this is an internal development screen rather than a
fresh or independent holdout.

| Population | Positions | Mean KL | Top-1 agreement | Target NLL delta | Hidden rel. L2 | Status |
|---|---:|---:|---:|---:|---:|---|
| Threshold | — | <=0.050000 | >=0.900000 | <=+0.050000 | <=0.100000 | — |
| Overall | 768 | 0.073719930 | 0.867187500 | +0.053455543 | 0.167517818 | **fail** |
| Offsets 0–15 | 96 | 0.012377540 | 0.947916667 | -0.006711043 | 0.057957455 | pass |
| Offsets 16–31 | 96 | 0.003680607 | 0.979166667 | -0.000053456 | 0.056517532 | pass |
| Offsets 32–63 | 192 | 0.046755770 | 0.880208333 | +0.038320826 | 0.160129544 | **fail** |
| Offsets 64–95 | 192 | 0.116722707 | 0.817708333 | +0.095345491 | 0.211740463 | **fail** |
| Offsets 96–127 | 192 | 0.123372168 | 0.807291667 | +0.083538106 | 0.240963771 | **fail** |

The overall population failed all four thresholds. Offsets 0–31 passed all
four; offsets 32–63 passed KL and NLL but failed top-1 and hidden state; both
later bands failed all four. The overall maximum per-position KL was
2.172196865 and p95 KL was 0.285592937.

The complete authenticated screen took 478.542 seconds: 103.876 seconds for
all-base parity, 325.300 seconds for the six primary sequence executions, and
45.742 seconds for deterministic reset replay. The candidate passed every
counter and population check, reproduced replay exactly, retained the mask
identity
`18854c256af3fc68326a2e9fa9173d943db838c116523d5c4057e3f1efe9c278`,
and passed all 27 post-run authentication checks.

### Decision

This is another valid negative semantic result, not an infrastructure or
resource failure. The mask was selected without the six screen records and
could not adapt after freeze, but teacher attention mass alone did not identify
a static 51-head allocation that met the causal quality gate. No package policy
was promoted and no fresh eight-sequence confirmation was run. Milestone 2's
Q7 result remains passed and unchanged; Milestone 3 remains blocked on
deployable bounded attention.

That failure justified the causal/value-sensitive static experiment documented
next. It has now completed and failed as well; neither another attention-mass
ranking nor more fitting of the same two-record natural-prose objective is
justified.

## Causal/value-sensitive 51-head gate

The last static experiment tested the remaining causal/value-sensitive
hypothesis directly. Its implementation was frozen at source commit `483c62f`;
the evaluator SHA-256 was
`442169060860257e78bbc0068bfdf9e5cf6edd93ff2b392c75ed333687765590`.
The training protocol was frozen before gradients under SHA-256
`037ebfd7d4e40af898ece7f353654eb8a41dc1883f191cbdf05fc34bf50bf4ba`.
It inherited and authenticated the failed attention-mass experiment rather
than silently replacing that historical boundary.

### Exact-forward training protocol

The trainer installed one float32 scalar gate for every layer-head pair before
the attention output projection of an otherwise frozen BF16 OLMoE teacher.
Gate zero meant `W16/C8/K4/S2`; gate one meant `W128/C8/K4/S2`. Each layer
computed both branches as follows:

1. Detached float32 Q/K and token-identity D128 values were sent through the
   existing raw streaming-attention DSO. Because sequence length and head
   dimension were both 128, each output coordinate exposed the native
   schedule's exact selected token and weight.
2. The DSO ran again with the real values, preserving its exact sequential
   dot products, softmax, top-k selection, eviction victims, counters, and
   output.
3. Backward propagation used differentiable gathered attention over the fixed
   native W16 support and differentiable full causal attention for W128. A
   straight-through expression retained the exact native forward values. The
   surrogate did not differentiate through top-k or cache-victim decisions.

This separation matters: hard cache behavior and measured candidate quality
were native, while gradients were explicitly an attribution proxy. The
training shell still used BF16 Hugging Face projections and dense MLPs, so it
could select a mask but could not itself establish packaged Q7 quality. Only
the later complete-native screen could do that.

Only selection sequences 0 and 1 contributed gradients. The six internal
screen records were prohibited during fitting. The loss was an equal mean over
bands 16–31, 32–63, 64–95, and 96–127. It combined normalized
teacher-to-student KL and final-hidden relative L2 with lower-weight positive
target-NLL drift and dense-teacher top-1 margin deficit. Two IHT steps each:

- executed both selection records with gradients under the current hard mask;
- averaged the two 16×16 gradients in float64;
- divided by global RMS plus `1e-12`;
- took the frozen unit projected-gradient step; and
- retained exactly the top 51 layer-head scores with deterministic
  layer-major tie-breaking.

Both `M1` and `M2` were therefore real executed masks. The terminal `M2`
evaluation was forward-only. The predeclared rule chose the lower maximum
per-record composite objective, then lower mean, then `M1` on an exact tie,
and allowed screening only if maximum and mean both improved over `M0` with no
individual record regression.

### Training result

The authenticated training result has SHA-256
`bacb0e31899f514a8b2b517987566e8bca68d39cabfd50b3c9e7ecf83bc756ea`.

| Executed mask | Role | Maximum objective | Mean objective | Selected heads |
|---|---|---:|---:|---:|
| `M0` | all-W16 baseline and first gradient | 7.8671169281 | 6.9172160625 | 0 |
| `M1` | first projection and second gradient | **4.7559914589** | **4.3284769058** | 51 |
| `M2` | second projection, terminal forward only | 6.2355780602 | 5.3186683655 | 51 |

`M1` improved the two record objectives by -2.0663528442 and -3.1111254692;
neither record regressed. The first projection changed 51 gates and the second
changed 42, demonstrating that the second IHT step was not a duplicate.
`M1` therefore won and was eligible for exactly one native development screen.
All training evidence and all post-training authentication checks passed.

The CPU-only fit took 6,930.099236 seconds. The four backward-bearing record
runs took 1,564.347, 1,650.704, 1,656.802, and 1,662.156 seconds; the two
terminal forwards took 179.301 and 178.861 seconds. These timings expose a
performance limitation of the BF16 proxy, not additional semantic evidence.

### Frozen complete-native screen

The selected mask was sealed by screen protocol SHA-256
`282bfe0b9e1da86577f0187112a4a444b0f36d7f84e10f4f9bb67730676807c2`
before the six reused internal records were executed. The screen used the
complete mapped native Q7 runtime, not the Transformers training shell. It
kept exactly 51 heads at W128 and 205 at W16:

| Resource or deterministic boundary | Exact value |
|---|---:|
| Positions per sequence | 128 |
| Rescued/base heads | 51 / 205 |
| Persistent attention state | 12,284,864 bytes |
| Attention scratch | 107,136 bytes |
| Total logical attention reads | 973,384,704 bytes |
| Dense full-context reference | 2,164,260,864 bytes |
| Logical attention fraction | 44.9753872184% |
| Inadmissible 52-head fraction | 45.2437999637% |
| Q7 scheduled bytes | 93,952,409,600 bytes |
| Q7 traffic fraction | 22.7864583333% |

The result SHA-256 is
`437d0de4ce4da37e69ca13279b76627d6f7721e766b8f1b4371fb318e7cbeb59`.
All historical parity, one-candidate, sequence-order, immutable-mask, native
candidate, resource, and post-run authentication evidence passed. The screen
took 319.674715 seconds, including 272.304325 seconds for the six primary
sequences and 44.940173 seconds for deterministic reset replay.

The semantic result was an authenticated failure:

| Population | Positions | Mean KL | Top-1 agreement | Target NLL delta | Hidden rel. L2 | Status |
|---|---:|---:|---:|---:|---:|---|
| Threshold | — | <=0.050000 | >=0.900000 | <=+0.050000 | <=0.100000 | — |
| Overall | 768 | 0.079132081 | 0.864583333 | +0.081198997 | 0.182647181 | **fail** |
| Offsets 0–15 | 96 | 0.012377540 | 0.947916667 | -0.006711043 | 0.057957455 | pass |
| Offsets 16–31 | 96 | 0.005119230 | 0.968750000 | +0.007758211 | 0.065067085 | pass |
| Offsets 32–63 | 192 | 0.049241551 | 0.869791667 | +0.028802983 | 0.172636807 | **fail** |
| Offsets 64–95 | 192 | 0.111879486 | 0.859375000 | +0.115367470 | 0.230262711 | **fail** |
| Offsets 96–127 | 192 | 0.146658900 | 0.770833333 | +0.180101951 | 0.266176934 | **fail** |

Both early bands through offset 31 passed every threshold. Offsets 32–63
passed KL and NLL but failed top-1 and hidden state; both later bands failed
all four metrics. Overall maximum per-position KL was 1.509838343 and p95 KL
was 0.268069629.

The learned causal mask is worse than the earlier attention-mass mask on every
overall measure:

| Static 51-head selector | Mean KL | Top-1 | NLL delta | Hidden L2 |
|---|---:|---:|---:|---:|
| Dense-teacher attention mass | 0.073719930 | 0.867187500 | +0.053455543 | 0.167517818 |
| Causal/value-sensitive IHT | 0.079132081 | 0.864583333 | +0.081198997 | 0.182647181 |

The training improvement was real but did not transfer through the packaged
Q7 runtime. This closes the tested fixed causal/value-sensitive static path,
not merely one failed optimizer run. The confirmation corpus remains unopened,
no fresh confirmation was frozen, and no head policy or package schema was
promoted.

### Next semantic boundary

The next experiment must change the supervision rather than merely refine this
mask. The defensible target is a Q7-aware retrieval-head selector trained
against complete packaged behavior on a new and substantially larger
synthetic retrieval corpus. Its loss must concentrate on the answer positions:
[DuoAttention](https://arxiv.org/abs/2410.10819) reports that synthetic
retrieval supervision is more effective for identifying retrieval heads than
ordinary language-model examples. Its protocol must:

- preserve an exact 51-head-equivalent ceiling of 973,384,704 logical
  attention bytes per 128-position sequence and the 12,284,864-byte attention
  state contract, with the 52-head boundary remaining inadmissible;
- predeclare training/development/confirmation splits and forbid adaptive
  reuse of this exhausted eight-record development corpus; and
- pass the complete native Q7 development gate before a separately sealed
  confirmation or any package-format proposal.

If this retrieval-targeted static selector cannot pass under the exact
resource contract, the next allocation class is a prompt/prefix-conditioned
dynamic policy. It must make every allocation causally from the prefix and
commit it before the first eviction it can affect. Adaptive per-head
[Ada-KV](https://arxiv.org/abs/2407.11550) and task-aware
[Task-KV](https://arxiv.org/abs/2501.15113) results motivate that fallback;
arbitrary token-wise promotion is not valid because an unselected head cannot
recover K/V state that was already evicted.

Two engineering changes can make that research cheaper: a deterministic
expert-parallel BF16 proxy and a native streaming-attention API that returns
its selected support during the real-value pass. Both require parity evidence,
but neither is semantic evidence and neither reopens the closed static result.
