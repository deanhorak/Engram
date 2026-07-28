# OLMoE Q7 sustained-context gate and attention attribution

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
own evaluator source. Every recorded post-run authentication check passed.

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

## Next prospective experiment: matched attention under 45%

The next experiment should freeze all non-attention identities above and
compare three policies that allocate essentially the same permissible logical
traffic differently:

| Policy | Intended bias | Logical reads/sequence | Dense fraction | State bytes |
|---|---|---:|---:|---:|
| `W16/C18/K16/S2` | maximum older retrieval | 968,753,152 | 44.7613856589% | 8,991,232 |
| `W24/C10/K8/S2` | balanced locality and retrieval | 968,753,152 | 44.7613856589% | 8,973,824 |
| `W30/C4/K2/S2` | maximum exact locality under budget | 968,753,152 | 44.7613856589% | 8,960,768 |

This is a controlled allocation experiment rather than another unstructured
budget guess. All three arms are exactly matched in logical reads, expose 32
values per mature attention step, and differ by less than 0.35% in persistent
state. Freeze the policies and selection rule before executing any arm, reuse
the identical 1,024-position population and thresholds, and require both
overall and every-band quality to pass. This development sweep may rank
passing arms by their worst normalized band margin, with state bytes as the
predeclared tie-breaker, but promotion still requires a fresh confirmation.

If none passes, ordinary window/candidate reallocation under the 45% ceiling is
unlikely to close the gap. The next boundary should then be learned or
distilled older-context selection trained against the exact-attention teacher,
while retaining W128 only as the diagnostic ceiling.
