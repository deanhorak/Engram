# OLMoE Q7 sustained-context gate, attribution, sweep, and layer rescue

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
layer combinations untested, keeps Milestone 3 blocked, and moves the next
prospective experiment to a fixed teacher-guided 51-of-256 head mask.

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
check passed.

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

The next prospective experiment is a teacher-guided fixed head mask. Exactly
51 of the model's 256 layer-head pairs can be rescued while reading
973,384,704 logical bytes per sequence, 44.9754% of dense attention. Rescuing
52 pairs would require 979,193,856 bytes, 45.2438%, and therefore exceed the
frozen cap.
