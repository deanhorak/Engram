# Evaluation

## Current combined-gate decision

As of 2026-07-30, no generic dense-Llama conversion passes the causal quality
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

The OLMoE Milestone 3 boundary has a positive train-only capacity result but
negative learned-selector results. Constructible C28 passes its frozen
same-state recovery gate at 0.6653937751 without additional KV reads. The
subsequent rank-4 query-content selector recovers only 0.25422074198 in BF16,
and its phase-conditioned mass successor recovers 0.2618728353. Both pass
systems, parity, and authentication checks but fail the semantic gate. There
is still no native causal, development, confirmation, package, or Milestone 3
progression.

The separate OLMoE branch now passes the same semantic thresholds and evidence
floor using the source model's trained top-8 router. The authoritative
Q7/group-64 confirmation executes BF16-rounded group scales and scores exactly
8 unique sequences and 256 positions:

| Measure | Result | Requirement |
|---|---:|---:|
| Mean KL | 0.00900774 | <= 0.05 |
| Top-1 agreement | 0.9765625 | >= 0.90 |
| Target NLL delta | +0.00391912 | <= +0.05 |
| Final-hidden relative L2 | 0.0460273 | <= 0.10 |
| Modeled expert/router traffic | 22.7865% | <= 45% |

The maximum single-position KL is 0.587149 and remains disclosed. That
all-layer causal confirmation executes decoded Q7 values inside Transformers.
The subsequent native systems confirmation reloads one 5,842,733,184-byte
artifact through the CPU-only kernel: route identity is exact, relative output
L2 is 1.94718e-6, and scheduled packed traffic is 22.7865%. See the
[causal confirmation](../reports/olmoe_q7_confirmation_2026-07-27/summary.md)
and [native systems confirmation](../reports/olmoe_q7_native_systems_2026-07-27/summary.md).

The subsequent [native token-boundary confirmation](../reports/olmoe_q7_native_token_boundary_2026-07-27/summary.md)
adds the mapped BF16 non-MLP state and complete CPU token loop. Fixture tests
match an independent NumPy reference and prove batch/incremental cache
equivalence. The pinned production smoke maps both artifacts and predicts
` Paris` after `The capital of France is` without constructing Transformers.
The authenticated package-only frontend reproduces the same token after
checking its external manifest root and exact inventory. Its selected-expert
parallel kernel plus canonical packed-block decoder improves representative
production layers by 6.56×–9.24× with bit-identical output. A five-position
prompt now takes 2.17 seconds of native execution, including 1.91 seconds in
Q7, down from 13.33 and 13.08 seconds.

The CPU teacher path has a separate safe parallel mode: four independent
sequence forwards share one read-only model. It reproduces the serial teacher
arrays byte-for-byte and reduces 8×33 teacher compute from 366.14 to 94.78
seconds (3.86×). Parallelizing experts directly is not used for sealed
references because concurrent BF16 kernels alter rounding.

The next two frozen evaluations exercise the complete package rather than
decoded Q7 inside Transformers. The short generation integration agrees on
60/60 teacher-forced decisions, 29/32 greedy tokens, and 7/8 complete prompts.
All remain within W=16.

The stronger complete native causal confirmation scores 8 sequences and 256
positions, deliberately split into 128 exact-local positions and 128
post-window positions. Both splits use the same frozen semantic thresholds:

| Population | Mean KL | Top-1 | NLL delta | Hidden relative L2 |
|---|---:|---:|---:|---:|
| Overall, 256 positions | 0.0129809 | 0.960938 | +0.0168240 | 0.0620471 |
| Offsets 0–15, exact local | 0.0153193 | 0.960938 | +0.0199584 | 0.0488925 |
| Offsets 16–31, bounded retrieval | 0.0106424 | 0.960938 | +0.0136896 | 0.0752018 |
| Frozen threshold | <= 0.05 | >= 0.90 | <= +0.05 | <= 0.10 |

The CPU-only candidate constructs no Transformers model shell. It passes
cache-position, diagnostic-argmax, reset, package, DSO, and post-run
authentication checks. Q7 schedules 187,904,819,200 bytes, 22.7865% of the
all-expert ideal-Q4 reference. The original candidate/metric loop takes 93.37
seconds and the fully authenticated command 183.57 seconds. A source-bound,
explicitly non-independent hardened replay reproduces all metrics exactly and
separates 88.79 seconds of native execution, 72.17 seconds of Q7 execution,
92.14 seconds of candidate-plus-metric wall time, and 184.55 seconds for all
authentication and execution.

This is an aggregate-threshold pass, not uniform positional parity. Maximum
KL is 0.606769 and p95 is 0.083556. Offset 31 alone has KL 0.051834, top-1
0.75, NLL +0.265473, and hidden L2 0.124504, missing all four per-offset
limits; per-offset limits were not part of the frozen gate. The corpus is too
small for broad language-quality claims. See the
[short generation report](../reports/olmoe_q7_native_generation_2026-07-28/summary.md)
and [complete native causal report](../reports/olmoe_q7_native_causal_2026-07-28/summary.md).

### Sustained-context W16 failure and matched W128 control

The later prospectively frozen sustained protocol is the stronger
bounded-attention result. It scores 8 distinct natural-text sequences at 128
prediction positions each (1,024 positions total), with the native
W16/C8/K4/S2 attention policy and the same Q7/group-64 MLP artifact used
throughout. All structural, reset-replay, traffic, artifact, teacher, source,
and post-run authentication checks pass, but semantic quality does not:

| Population | Mean KL | Top-1 | NLL delta | Hidden relative L2 | Result |
|---|---:|---:|---:|---:|---|
| Overall, 1,024 positions | 0.143578 | 0.802734 | +0.159292 | 0.238260 | fail |
| Positions 0–15 | 0.011374 | 0.945312 | +0.000667 | 0.055917 | pass |
| Positions 16–31 | 0.008252 | 0.937500 | -0.003417 | 0.075655 | pass |
| Positions 32–63 | 0.083857 | 0.828125 | +0.075577 | 0.218544 | fail |
| Positions 64–95 | 0.223422 | 0.753906 | +0.238444 | 0.314587 | fail |
| Positions 96–127 | 0.257219 | 0.687500 | +0.324524 | 0.354124 | fail |
| Frozen threshold | <= 0.05 | >= 0.90 | <= +0.05 | <= 0.10 | every row required |

Per sequence, the authenticated W16 run processes 128 positions with
6,336,512 bytes of attention state and 3,840 scratch bytes. It records exactly
1,792 evictions, 222,208 older-candidate entries scored, 113,152 older entries
selected, and 512 sink insertions; accepted heavy-hitter updates range from
3,901 to 4,352 across sequences. Its 677,117,952 logical attention-read bytes
are 31.2863% of full-context logical KV reads. Q7 schedules
93,952,409,600 bytes per sequence, or 751,619,276,800 bytes total, unchanged
from the previously validated 22.7865% expert/router traffic ratio.

Because the W16 result is an authenticated quality failure, a separate
post-failure protocol froze a matched attribution control before executing
it. The control changes only `local_window` from 16 to 128: it retains C8,
K4, S2, the same package, DSO, Q7 artifact and policy, corpus, teacher arrays,
12 CPU threads, and transformer-shell-free token runtime. W128 is exact full
causal attention for this protocol's 128 positions. The control passes every
frozen overall and per-band semantic threshold:

| Population | Mean KL | Top-1 | NLL delta | Hidden relative L2 | Result |
|---|---:|---:|---:|---:|---|
| Overall, 1,024 positions | 0.003438 | 0.974609 | +0.001459 | 0.041389 | pass |
| Positions 0–15 | 0.011374 | 0.945312 | +0.000667 | 0.055917 | pass |
| Positions 16–31 | 0.001969 | 0.976562 | +0.003676 | 0.038718 | pass |
| Positions 32–63 | 0.002102 | 0.992188 | +0.002263 | 0.039488 | pass |
| Positions 64–95 | 0.002360 | 0.968750 | +0.006798 | 0.038476 | pass |
| Positions 96–127 | 0.002620 | 0.976562 | -0.005398 | 0.040276 | pass |

The 128 pre-intervention rows—positions 0–15 in each of eight sequences—match
the bounded run exactly, including every recorded per-position metric. This
localizes the behavioral change to the frozen intervention boundary instead
of a package, execution, or metric mismatch. W128 has 35,825,664 bytes of
state and 18,176 scratch bytes per sequence, with zero evictions, older
candidates, selections, sink insertions, and heavy-hitter updates. It reads
2,164,260,864 logical attention bytes per sequence, exactly 100% of the dense
full-context logical-KV reference, while Q7 scheduling remains unchanged.

The authenticated roots are:

- sustained protocol `82189276ed0e555c2737f4842b1d1ed625f54d9ceaa2c63fe41fe71c5c6eb599`;
- bounded W16 result `673523c29b12154f98916b8ce6f203b4967842e4bcae8f5c02ad4d197aab97eb`;
- matched-control protocol `1619cd5f3cb607a7d0e2b5cde2e61a83dba3f1615884462a30570d62c7764dd9`;
- matched W128 result `3d099ffd3121e47bdf61ed8772e5e9d08b01b8c6041e9a963b409a502808d345`;
- package manifest `861e9cc472f9e1245db5d64e9253411d0b656a0f08df2f58264e9c708ed750db`;
- native DSO `4cd4de8f3e3cefad59d7b9e6e23a0d1d06a26abc10af2e0c4f9242a2b5876ca7`.

This matched result attributes the sustained drift primarily to bounded
attention on this corpus; it does not turn W128 into a deployable policy, prove
task-sensitive long-context retrieval, or establish broad language quality.
The byte counts above are deterministic logical interface counts, not measured
DRAM traffic. W128 intentionally consumes 100% of full-context logical reads
and is a diagnostic ceiling rather than a candidate for the <=45% attention
traffic gate. See the
[sustained-context and control evidence](../reports/olmoe_q7_sustained_context_2026-07-28/summary.md).

### Exactly traffic-matched attention sweep

The prospectively frozen follow-up ran all three predeclared arms in fixed
order under the 45% modeled logical-read cap. Each arm read exactly
968,753,152 logical bytes per sequence (44.7613856589% of dense), exposed 32
values per mature step, and retained the Q7 schedule at 22.7864583333% of the
all-expert ideal-Q4 reference. Persistent state ranged only from 8,960,768 to
8,991,232 bytes, so the controlled treatment was how the fixed attention
budget was divided between exact locality and older retrieval:

| Policy | Mean KL | Top-1 | NLL delta | Hidden relative L2 | Evidence | Quality |
|---|---:|---:|---:|---:|---|---|
| W16/C18/K16/S2 | 0.06388655 | 0.8671875 | +0.05170082 | 0.15771664 | pass | **fail** |
| W24/C10/K8/S2 | 0.06591232 | 0.8779297 | +0.05847984 | 0.15975482 | pass | **fail** |
| W30/C4/K2/S2 | 0.09581344 | 0.8408203 | +0.07572840 | 0.18842230 | pass | **fail** |

All arms passed every exact evidence, source/artifact authentication,
structural-counter, reset-replay, and pre-eviction identity check. All also
passed every threshold in the 0–15 and 16–31 bands. Final-hidden drift exceeded
the threshold at 32–63 for all arms, and the 64–95 and 96–127 bands failed
broadly. Thus zero arms passed the required overall-plus-every-band rule. The
predeclared selector returned no arm, deliberately avoiding a post hoc “best
failure,” and the separately sealed fresh-confirmation corpus was not used.

The frozen protocol SHA-256 is
`2853de54119f4218c165ebebfe560162f76f99b552fdfe84c803a5ca8acfcef0`;
the authenticated result is
`813bac5b1d38af7653cf49d8c7b7ca278df8aac5402fdd28692e905bebfc7658`.
The evaluator was frozen at source commit `102bda2` with source SHA-256
`cf2e4be0bc4d8e6da54aebcb11b94e7c4ecde2d56e12831fe8de835a342ffa60`.
Because the installed package immutably binds W16/C8/K4/S2, the experiment
used an explicit raw-runtime policy override against the consumed
sustained-development corpus; it neither modified nor promoted the package.

This closes global W/C/K reallocation as the next useful search axis under the
current budget. Milestone 2 remains passed because the matched exact-attention
control isolates the Q7 path; Milestone 3 remains blocked. It motivated the
prospectively frozen whole-layer experiment below. W128 remains only the
diagnostic ceiling.

### Authenticated three-layer dense-attention rescue

The next prospectively frozen development experiment tested whether the
attention error was concentrated in a few layers. It added a backward-compatible
per-layer native attention ABI, then greedily changed exactly three of OLMoE's
16 layers from `W16/C8/K4/S2` to `W128/C8/K4/S2`. The selection corpus was the
same already-consumed sustained-development corpus, deterministically divided
by SHA-256 of `record_id`: two sequences selected layers and six sequences
formed an output-blind internal screen. The split identity was
`c267dd96c121b5baf9d229b4e6a2a880f396361ae9565020813d7e2e279ed310`.
This was development selection, not an independent confirmation.

All candidates in each round ran before selection. The fixed greedy search
therefore executed `16 + 15 + 14 = 45` candidate schedules:

| Round | Candidates | Winning layer | Cumulative rescued layers |
|---|---:|---:|---|
| 1 | 16 | 11 | 11 |
| 2 | 15 | 6 | 11, 6 |
| 3 | 14 | 10 | 11, 6, 10 |

The final schedule retained 13 base layers and rescued layers 6, 10, and 11.
Its exact per-sequence native resource contract was:

| Resource or counter | Final schedule |
|---|---:|
| Persistent attention state | 11,865,728 bytes |
| Attention scratch | 6,528 bytes |
| Local/exact KV reads | 816,447,488 bytes |
| Candidate-key reads | 92,438,528 bytes |
| Selected-value reads | 47,071,232 bytes |
| Total logical attention reads | 955,957,248 bytes |
| Dense logical attention reference | 2,164,260,864 bytes |
| Logical attention fraction | 44.1701489826% |
| Evictions | 1,456 |
| Older entries scored | 180,544 |
| Older values selected | 91,936 |
| Sink insertions | 416 |
| Permitted heavy-hitter updates | 1,248–22,880 |
| Q7 scheduled bytes | 93,952,409,600 bytes |
| Q7 all-expert ideal-Q4 fraction | 22.7864583333% |

Before any layer-rescue candidate ran, the new all-base layered ABI was checked
against the historical scalar DSO over one complete 128-position sequence.
Tokens, normalized hidden states, full logits, cache positions, deterministic
counter streams, and historical diagnostic hashes were exactly equal. All 45
candidate evidence contracts, all three round-resource contracts, the final
traffic checks, the six-sequence screen evidence, deterministic reset replay,
and all 21 post-run authentication roots passed.

The selected schedule nevertheless failed semantic quality on the six-sequence,
768-position internal screen:

| Population | Mean KL | Top-1 | NLL delta | Hidden relative L2 | Result |
|---|---:|---:|---:|---:|---|
| Overall, 768 positions | 0.102320950 | 0.845052083 | +0.116775650 | 0.206036865 | **fail** |
| Positions 0–15, 96 positions | 0.012377540 | 0.947916667 | -0.006711043 | 0.057957455 | pass |
| Positions 16–31, 96 positions | 0.006325668 | 0.979166667 | +0.010194634 | 0.072582537 | pass |
| Positions 32–63, 192 positions | 0.065331080 | 0.854166667 | +0.072611753 | 0.193418547 | **fail** |
| Positions 64–95, 192 positions | 0.147380969 | 0.796875000 | +0.150818163 | 0.261827487 | **fail** |
| Positions 96–127, 192 positions | 0.187220146 | 0.765625000 | +0.241930887 | 0.303631431 | **fail** |
| Frozen threshold | <= 0.05 | >= 0.90 | <= +0.05 | <= 0.10 | every row required |

Both early bands passed all four checks, but the overall population and every
band from position 32 onward failed all four. This is an authenticated quality
failure, not an execution or provenance failure. No fresh confirmation was
run, and the development schedule was not integrated into or promoted as a
package policy.

The authenticated command took 4,443.916 seconds: 3,938.180 seconds for
candidate primary-sequence execution, 86.643 seconds for layered/scalar parity,
260.321 seconds for the internal holdout, and 42.033 seconds for its reset
replay. The implementation was frozen at source commit `708782b`. Authentication
roots are:

- layer-rescue protocol
  `9514e90bd5d14ae01ea27185763e5a833d4f1963e6bffd0ec0c81848f35b0c3e`;
- layer-rescue result
  `97ce800bd855c1f16248cada696936c7c56acd49c02d6f1c9ce9885dc44f7c49`;
- evaluator source
  `77dafe8fc1fb6ca317ad7b99d5d86122e26b94b477f5befcf6184ce14080dff0`;
- immutable layered native DSO
  `fe4dfdcc7e87a3cd5e36074e07d297f838ba345c37e939eeb0d796cb39cce409`.

This closes the tested frozen greedy three-layer `W128` path under the 45%
attention-read cap; the search can miss interacting layer combinations. It
does not change the passed Milestone 2 Q7 result; Milestone 3 remains blocked
on bounded attention.

### Prospective teacher-attention-mass 51-head rescue

The next experiment moved the allocation unit from a whole layer to one
layer-head pair. OLMoE has 16 layers and 16 query heads per layer, for 256
pairs. A prospectively fixed dense-teacher attention-mass heuristic selected
exactly 51 pairs for `W128/C8/K4/S2`; the remaining 205 pairs retained
`W16/C8/K4/S2`. The additive experimental head-wise runtime passed exact
all-base output parity, deterministic accounting, cache-position, reset-replay,
resource, and authentication checks. Version 1 requires equal query and
key/value head counts so each selected pair owns an independent K/V cache.

The 51-head schedule is the largest admissible fixed mask under the declared
attention criterion. It reads 973,384,704 logical attention bytes per
128-position sequence, or 44.975387218386625% of the dense reference. A
52-head mask would read 979,193,856 bytes, or 45.2437999637%, and is therefore
inadmissible. Q7 expert scheduling is unchanged at 93,952,409,600 bytes per
sequence and 22.7864583333% of the all-expert ideal-Q4 reference. Here W128
means exact full causal context only for the tested 128-position horizon; it
is not a general unbounded-attention claim.

The fixed mask was then scored on the six reused internal development records
that were not used to derive the teacher-mass ranking: 768 predictions in
total. All execution and provenance evidence passed, but semantic quality did
not:

| Population | Mean KL | Top-1 | NLL delta | Hidden relative L2 | Result |
|---|---:|---:|---:|---:|---|
| Overall, 768 positions | 0.07371992968429097 | 0.8671875 | +0.05345554334600896 | 0.1675178178168911 | **fail** |
| Positions 0–15, 96 positions | pass | pass | pass | pass | pass |
| Positions 16–31, 96 positions | pass | pass | pass | pass | pass |
| Positions 32–63, 192 positions | pass | **fail** | pass | **fail** | **fail** |
| Positions 64–95, 192 positions | **fail** | **fail** | **fail** | **fail** | **fail** |
| Positions 96–127, 192 positions | **fail** | **fail** | **fail** | **fail** | **fail** |
| Frozen threshold | <= 0.05 | >= 0.90 | <= +0.05 | <= 0.10 | every row required |

This materially improves all four overall metrics over the prior greedy
three-layer rescue, but it still misses every overall threshold. No fresh
confirmation or package promotion was justified. The negative conclusion is
narrow: this closes the fixed 51-head attention-mass heuristic, not all
head-wise allocation. The next useful boundary is a causal/value-sensitivity
selector or a dynamic teacher-distilled allocation policy. Milestone 2
remains passed for Q7; Milestone 3 remains blocked on bounded attention.

### Causal/value-sensitive 51-head gate

The follow-up replaced attention-mass ranking with a direct causal objective
while retaining the same exact 51-of-256 head budget. For each layer and head,
the evaluator mixed the native `W16/C8/K4/S2` output with an exact native
`W128/C8/K4/S2` output. Native float32 execution supplied the forward value;
a fixed-membership gathered/full-attention surrogate supplied gradients only.
The frozen dense BF16 model contributed projections, the unchanged MoE path,
hidden states, and vocabulary logits. This is therefore a serial BF16
attribution/training proxy with exact native attention forward, not the final
Q7 runtime measurement.

Two hard-projected IHT steps used only selection records 0 and 1. Each step
averaged both complete-record gradients and projected back to exactly 51
heads. All three executed masks were scored, and the predeclared
maximum-objective, then mean-objective, then M1-tie-break rule selected M1:

| Selection objective | All-sparse M0 | Selected M1 |
|---|---:|---:|
| Maximum per-record composite objective | 7.8671169 | 4.7559915 |
| Mean per-record composite objective | 6.9172161 | 4.3284769 |

Neither selection record regressed. All training evidence, projection-chain,
native-oracle, artifact, framework, and post-run checks passed. The CPU fit
took 6,930.10 seconds, or about 115.5 minutes. That development result
authorized exactly one package-native screen; it did not itself establish Q7
quality.

The subsequent systems qualification establishes a way to reduce that
bottleneck without changing the reference arithmetic. Transformers' installed
`grouped_mm` dispatcher uses a serial CPU fallback on the frozen Torch 2.5.1
stack; this is numerically distinct from the eager expert loop. The proxy
therefore executes the installed forward dispatcher unchanged, replays frozen
expert backwards on 12 workers, and reduces hidden gradients in the exact
backend-specific order. One complete archived M0/sequence-0 record matched
loss, all 256 gradients, native non-timing diagnostics, and the projected
51-head mask bit for bit. It reduced record time from 1,564.347 to 809.168
seconds (1.933×; 48.274% less wall time), exceeding the predeclared 10%
qualification boundary. The
[authenticated proxy report](../reports/olmoe_q7_expert_proxy_2026-07-28/summary.md)
authorizes larger development fits only; it neither changes the failed mask
nor advances the Milestone 3 semantic gate.
This is one previously consumed development record measured across separate
executions, not a controlled repeated benchmark or a measured speedup for the
complete 6,930-second fit.

The selected static mask then ran through the complete native Q7 path on the
six reused development-screen records, totaling 768 positions. Every
execution, resource, reset, package, source, and authentication check passed,
and the schedule remained within the frozen budget at
44.9753872184% of dense logical attention reads. Semantic quality failed:

| Population | Mean KL | Top-1 | NLL delta | Hidden relative L2 | Result |
|---|---:|---:|---:|---:|---|
| Overall, 768 positions | 0.07913208059 | 0.8645833333 | +0.08119899696 | 0.18264718059 | **fail** |
| Positions 0–15 | pass | pass | pass | pass | pass |
| Positions 16–31 | pass | pass | pass | pass | pass |
| Positions 32–63 | pass | **fail** | pass | **fail** | **fail** |
| Positions 64–95 | **fail** | **fail** | **fail** | **fail** | **fail** |
| Positions 96–127 | **fail** | **fail** | **fail** | **fail** | **fail** |
| Frozen threshold | <= 0.05 | >= 0.90 | <= +0.05 | <= 0.10 | every row required |

The causal/value-selected mask transferred worse on all four overall metrics
than the earlier fixed attention-mass mask. This closes the tested fixed
static causal/value selector as well as the attention-mass selector; it does
not close head-wise execution or dynamic allocation. No fresh confirmation
was opened, and neither mask nor package format was promoted. Milestone 2
remains passed for the qualified Q7 semantic path, while Milestone 3 remains
blocked on bounded long-context attention.

The evaluator was frozen at source commit `483c62f`. Authenticated artifact
roots for that historical causal/value-sensitive screen are:

- [training protocol](../reports/olmoe_q7_sustained_context_2026-07-28/causal_head_gate_protocol.json)
  `037ebfd7d4e40af898ece7f353654eb8a41dc1883f191cbdf05fc34bf50bf4ba`;
- [training result](../reports/olmoe_q7_sustained_context_2026-07-28/causal_head_gate_training.json)
  `bacb0e31899f514a8b2b517987566e8bca68d39cabfd50b3c9e7ecf83bc756ea`;
- [screen protocol](../reports/olmoe_q7_sustained_context_2026-07-28/causal_head_gate_screen_protocol.json)
  `282bfe0b9e1da86577f0187112a4a444b0f36d7f84e10f4f9bb67730676807c2`;
- [native Q7 screen result](../reports/olmoe_q7_sustained_context_2026-07-28/causal_head_gate_screen_result.json)
  `437d0de4ce4da37e69ca13279b76627d6f7721e766b8f1b4371fb318e7cbeb59`.

### Synthetic-retrieval 51-head protocol and result

The retrieval-specific follow-up is now implemented and frozen independently
of the consumed natural-prose experiment. It creates deterministic synthetic
passkey records in three identity- and token-disjoint splits: **8 train, 8
development, and 8 sealed confirmation**. Each record contains 129 tokenizer
IDs: IDs 0–127 are model inputs, IDs 1–128 are their causal targets, and only
the final 32 targets contribute to training. In logit coordinates, the
answer-only ground-truth cross-entropy therefore scores rows **96–127**.

Each record places four eight-token passkeys at the source starts 8, 28, 48,
and 68, followed by a query and the 32-token answer. The eight records in each
split balance the A/B/C/D answer labels over the four source depths twice.
The generator selects numeric strings that are singleton tokens under the
packaged tokenizer and assigns **768 globally unique passkey token IDs**:
4 passkeys × 8 tokens × 24 records. Record identities are disjoint, split
token intersections are empty, and round-trip tokenization is required.

The training forward value is the complete packaged native Q7 result, not a
dense-MLP approximation. For gradients only, a straight-through boundary
reuses a frozen BF16 Transformers shell with exact native attention; the
previously qualified frozen-expert backward proxy executes independent expert
backwards on 12 workers while preserving the installed `grouped_mm` forward.
All source weights remain frozen. Training performs two
iterative-hard-thresholding steps, `M0 → M1 → M2`, from the all-bounded
baseline. `M1` and `M2` are projected to exactly **51 heads**. Selection first
minimizes the worst per-record answer cross-entropy, then its mean, preferring
M1 on a tie, and additionally requires strict worst/mean improvement over M0
with no training-record regression.

Exactly 51 W128 rescue heads schedule **973,384,704 logical attention bytes
per 128-position sequence**, or **44.9753872184%** of the 2,164,260,864-byte
full-context reference, and retain 12,284,864 bytes of attention state. A
52-head candidate would schedule **45.2437999637%**, so it is prospectively
inadmissible. Q7 scheduling remains 93,952,409,600 bytes per sequence, or
22.7864583333% of all-expert ideal Q4.

Only a training-qualified M1 or M2 reaches development. The development gate
then evaluates both a full-W128 packaged-Q7 control and the exact 51-head
packaged-Q7 candidate over all eight development records. Both must pass KL,
top-1, NLL-delta, and hidden-state thresholds overall and independently for
each of the four source depths. The fit path authenticates train and
development but deliberately does not open or hash the confirmation file. A
development pass would authorize a separate one-shot confirmation command;
that command is not yet implemented.

The protocol is frozen at
`work/olmoe_q7/retrieval_selector_2026-07-29_frozen/protocol.json`, with
SHA-256
`f6961ec6bffadb306a75ae8aaa8c400d1282cbb096876532d75e22a295660580`.
The complete fit/development screen ran for 14,025.138 seconds. `M2` passed the
training-selection rule: worst answer cross-entropy fell from 7.976308 to
1.227907 and mean answer cross-entropy fell from 7.647114 to 1.005444, with no
record regression. The dense teacher passed retrieval evidence overall and at
all four source depths. The full-W128 packaged-Q7 control also passed, with KL
0.002000, top-1 agreement 0.984375, target-NLL delta 0.004333, and hidden
relative L2 0.048957.

The exact-51 candidate failed development. Overall KL was 0.186610, target-NLL
delta 0.283658, and hidden relative L2 0.335103; only top-1 agreement passed at
0.929688. Every resource check and both reset/replay checks passed. The
candidate failed KL, NLL-delta, and hidden-state thresholds at every source
depth, and also failed top-1 at the middle depth. Confirmation remained
unopened and was not authorized. The authenticated
[result](../reports/olmoe_q7_retrieval_selector_2026-07-29/development_result.json),
[training checkpoint](../reports/olmoe_q7_retrieval_selector_2026-07-29/development_result.training_checkpoint.json),
and [summary](../reports/olmoe_q7_retrieval_selector_2026-07-29/summary.md)
close this static selector at the declared budget; Milestone 3 remains
blocked.

Before development, `fit-screen` atomically writes and rereads a complete
protocol-bound training checkpoint. Resume is explicit and requires both
`--resume-training-checkpoint` and
`--resume-training-checkpoint-sha256`; the loader reconstructs the complete
`M0 → M1 → M2` chain before skipping training.

To reproduce the freeze in a new directory:

```bash
PYTHONPATH=src python -m engram.evaluation.olmoe_retrieval_head_selector \
  freeze \
  --package work/olmoe_q7/package \
  --manifest-sha256 861e9cc472f9e1245db5d64e9253411d0b656a0f08df2f58264e9c708ed750db \
  --layered-library work/olmoe_q7/libengram_olmoe_token_runtime.layered-fe4dfdcc.so \
  --headwise-library work/olmoe_q7/libengram_olmoe_token_runtime.headwise-cb72b31e.so \
  --attention-library work/olmoe_q7/libengram_attention.causal-gate-153e91d9.so \
  --proxy-qualifier work/olmoe_q7/sustained_2026-07-28/causal_head_gate_proxy_record.json \
  --out work/olmoe_q7/retrieval_selector_reproduction/protocol.json
```

To execute the frozen train/development boundary:

```bash
PYTHONPATH=src python -m engram.evaluation.olmoe_retrieval_head_selector \
  fit-screen \
  --protocol work/olmoe_q7/retrieval_selector_2026-07-29_frozen/protocol.json \
  --protocol-sha256 f6961ec6bffadb306a75ae8aaa8c400d1282cbb096876532d75e22a295660580 \
  --out work/olmoe_q7/retrieval_selector_2026-07-29_frozen/development_result.json
```

The answer-only design follows
[DuoAttention](https://arxiv.org/abs/2410.10819), which reports that
retrieval-specific supervision identifies retrieval heads more effectively
than ordinary language-model examples. The conditioned and episodic follow-ups
retain the causality and task-aware allocation constraints motivated by
adaptive per-head [Ada-KV](https://arxiv.org/abs/2407.11550) and task-aware
[Task-KV](https://arxiv.org/abs/2501.15113) allocation.

### Train-only retrieval attribution after the static-selector failure

All experiments in this subsection reuse the eight retrieval-training records
and the authenticated `M2` checkpoint. They are diagnostic progression
screens, not new development or confirmation evidence. Their strict reference
is the full-context `M2` training population:

| Candidate | Mean answer CE | Worst answer CE | Records no worse than `M2` | Progression |
|---|---:|---:|---:|---|
| Full-context `M2` reference | 1.005444 | 1.227907 | 8/8 | Reference |
| Causal K2 prefix prototypes | 1.046825 | 1.224952 | 3/8 | Fail |
| All-head payload-only episodic oracle | 1.224460 | 1.327343 | 1/8 | Fail |
| All-head label-plus-payload episodic oracle | 1.231254 | 1.321619 | 1/8 | Fail |
| K51 head-gated payload oracle | 1.400569 | 1.694034 | 1/8 | Fail |

#### Causal K2 prefix selector

The two-prototype selector reused the completed `M0 → M1 → M2` transfer
matrix and never repeated the expensive surrogate backwards. It selected an
earlier-half or later-half 51-head prototype using only fact order already
visible in the causal prefix. Assigned mean answer CE regressed by 0.041381;
the worst CE improved by only 0.002955; five records regressed; and the
later-half cluster regressed in both mean and worst loss. Every native counter,
resource, split-separation, and artifact-authentication check passed.
Confirmation remained unopened. Result SHA-256:
`dacb3f37886d1207bc6b9a5717b3015174c4edc4947b89dd12ef35ff67ae8814`.

#### Exact episodic-capacity oracles

The first episodic ABI adds a per-layer BF16 K/V bank without changing the
legacy token-step path. The evaluator prospectively validates an entire
multi-token schedule before mutating runtime state; reset clears the cache,
directive shadow state, and counters. A causal oracle wrote four known
eight-token source payloads into 32 canonical slots and exposed only the
correct span at the 32 corresponding answer rows. This eliminates selector
error and tests whether the representation itself is sufficient.

It was not. The full-W128 packaged-Q7 control passed with KL 0.001892, top-1
1.0, target-NLL delta -0.002330, and hidden relative L2 0.047297. The
payload-only candidate measured KL 0.446656, top-1 0.921875, NLL delta
+0.557528, and hidden relative L2 0.428062. Mean/worst answer CE was
1.224460/1.327343, and seven records regressed against `M2`. The first row of
each answer block accounted for 55.26% of total KL and 62.12% of positive NLL
regression; its mean KL was 1.974513 and top-1 only 0.4375. All systems checks
passed at an upper bound of 710,672,384 read bytes, 714,866,688 total
read-plus-write bytes, and 10,534,912 state bytes. Protocol/result SHA-256:
`1e7b89e5b376430b82456bf306e50a0fb7c0cb9ed75b0d4e400ad7950b517cce`
and
`b2daa5eff271b6f030c01e8a4854a602f7b2907f7af487d38ab750468bbc42cc`.
See the [payload-only report](../reports/olmoe_q7_retrieval_episodic_oracle_2026-07-29/summary.md).

A second cheap capacity screen stored the immediately preceding label token
with each payload: four nine-token spans, 36 canonical slots. It too failed.
Mean/worst answer CE was 1.231254/1.321619, respectively 0.225811 and
0.093712 worse than `M2`, and seven records regressed. The exact counters,
reset replay, resource gates, and post-run authentication all passed.
Upper-bound combined reads were 714,866,688 bytes; total traffic was
719,585,280 bytes (33.2485% of dense full-context K/V); state and scratch were
11,059,712 and 4,992 bytes. The fit path performed no dense-teacher forward
and opened neither development nor confirmation. Frozen protocol SHA-256:
`1812a6ba72afe0c5f32e459867c29f3d8dbd609a3d0ddf59ac52ae6859ce4d3d`;
result SHA-256:
`e1ec5a2bde8b9ce7198fe1571a7670c45a3bc7a712cdf9a856f869b6429fe69d`.
This rejects the label-plus-payload representation at all-head exposure.

#### Head-gated episodic ABI and fixed K51 result

The runtime now exposes
`engram_olmoe_token_open_episodic_headwise_v1` through the C ABI and
`episodic_head_mask` through Python. Inactive layers call the exact legacy
attention step and allocate no episodic bank. Active layers write the complete
causal BF16 K/V row, but only enabled query heads deduplicate, score, join the
softmax, and read the selected episodic span. A malformed, missing, or
all-zero mask fails closed. An all-ones mask is bit-exact with the original
all-head episodic ABI. Reset and all counter streams retain exact parity.

The fixed K51 screen applied the existing `M2` mask, rather than fitting a new
one. Its mask SHA-256 is
`49802a2d37abd44e4015e87633c9a321e333315b9400f6a69d4713ec2270b446`;
the per-layer head counts are
`[3,3,1,4,0,7,7,4,1,6,4,1,5,3,2,0]`, spanning 14 active layers. It failed
the strict answer-loss gate: mean CE regressed by 0.395125 to 1.400569, worst
CE regressed by 0.466127 to 1.694034, and only one record improved. All
systems checks passed with 683,802,624 upper-bound read bytes, 687,472,640
total traffic bytes (31.7648%), 10,010,112 state bytes, and 4,736 scratch
bytes. Confirmation remained unopened.

The [K51 protocol, result, and summary](../reports/olmoe_q7_retrieval_episodic_head_mask_2026-07-29/summary.md)
have SHA-256 roots
`38ceb03c5ab8a18038aea57728bdca9f405ec46cd5311a0ba8569059843a5fd6`
and
`18bc2ec7ee55712f85d237ab0159ff160add37dc840dfcea3028b216f0062852`.
The negative result rejects transferring the old full-context K51
cardinality directly to the cheaper episodic cache; the earlier all-head
payload result was materially better and still far below the traffic ceiling.

#### Frozen ranked-prefix screen result

The train-only experiment was frozen independently at
`work/olmoe_q7/retrieval_episodic_rank_sweep_2026-07-29_frozen/protocol.json`,
SHA-256
`e238b5cfc4359422abc96c227f4010ade747ff161c946b2edae14c171a7bf04c`.
It authenticated the failed K51 prerequisite, the historical all-head
attribution, the native head-gated DSO, the checkpoint, and the complete
projected-score order before execution.

| Ordered candidate | Active layers | State bytes | Scratch bytes | Read bytes | Total traffic bytes |
|---:|---:|---:|---:|---:|---:|
| K64 | 14 | 10,010,112 | 4,736 | 685,506,560 | 689,176,576 |
| K96 | 15 | 10,272,512 | 4,800 | 689,700,864 | 693,633,024 |
| K128 | 16 | 10,534,912 | 4,864 | 693,895,168 | 698,089,472 |
| K165 | 16 | 10,534,912 | 4,864 | 698,744,832 | 702,939,136 |

The evaluator completed all eight records for each candidate before applying
strict mean improvement, strict worst improvement, and no-record-regression
checks against authenticated `M2`. No candidate passed:

| Candidate | Mean answer CE | Worst answer CE | Records improved versus `M2` |
|---:|---:|---:|---:|
| K64 | 1.379699 | 1.639418 | 1/8 |
| K96 | 1.328848 | 1.618843 | 1/8 |
| K128 | 1.337958 | 1.621764 | 1/8 |
| K165 | 1.331006 | 1.608617 | 1/8 |

The prospectively frozen total-failure key—worst CE, then mean CE, then
K—retained K165 for diagnostic reset replay. Replay passed exactly, but K165
remains a failed candidate. All resource, counter-stream, replay, and
post-run authentication checks passed; no development screen was authorized
and confirmation remained unopened. The immutable
[protocol](../reports/olmoe_q7_retrieval_episodic_head_mask_2026-07-29/rank_sweep_protocol.json)
and
[result](../reports/olmoe_q7_retrieval_episodic_head_mask_2026-07-29/rank_sweep_train_screen.json)
have SHA-256 roots
`e238b5cfc4359422abc96c227f4010ade747ff161c946b2edae14c171a7bf04c`
and
`a6fb045bbad526411b8318e7de2412ea56d69b3f2be0d977ca6819635c0718da`.

The failure closes cardinality expansion under the transferred `M2` ranking.
It does not make diagnostic K165 the next base: the authenticated K256
all-head payload result is strictly better at 1.224460 mean and 1.327343
worst CE, with 33.0305% upper-bound traffic versus K165's 32.4794%.

#### Fixed-K256 V2 logit-bias result

The next prospectively frozen V2 screen fixed K256, the all-head payload,
the oracle schedule, and `W16/C8/K4/S2`, changing only episodic logit mass.
It first established exact `beta=0` V1/V2 parity, then evaluated
`gamma={1/2,1/4,3/16,1/8}` in least-intervention-first order. Each candidate
added `float32(log(gamma))` to the episodic scores and added no state,
scratch, or logical traffic.

All four candidates executed all eight training records and failed the strict
mean/worst/no-regression gate:

| Candidate | Mean answer CE | Worst answer CE | Records improved versus `M2` |
|---|---:|---:|---:|
| `gamma=1/2` | 1.461414 | 1.669250 | 1/8 |
| `gamma=1/4` | 1.883818 | 2.288258 | 0/8 |
| `gamma=3/16` | 2.161750 | 2.595642 | 0/8 |
| `gamma=1/8` | 2.725091 | 3.430532 | 0/8 |

Every candidate passed the exact counter and resource contract at 714,866,688
upper-bound traffic bytes, 33.030523% of dense full-context K/V, with
10,534,912 state bytes and 4,864 scratch bytes. The total-failure rule
replayed `gamma=1/2`; replay passed exactly, but this is only the best failed
nonzero arm. The historical `beta=0` attribution remained substantially
better at 1.224460 mean and 1.327343 worst CE. No dense-teacher forward ran,
development was not authorized, and the reserved confirmation split remained
unopened.

The archived [V2 parity, protocol, result, and summary](../reports/olmoe_q7_retrieval_episodic_logit_bias_2026-07-29/summary.md)
are rooted by result SHA-256
`19d08ce9eb4b673d423e9781a491e25ec03bdec09467a43e7be1881874ef2287`.
This closes shared scalar logit-mass calibration.

The same-state W128-shadow residual-capacity boundary then completed. It fixed
the `beta=0` K256 base and captured the post-`W_o` difference while the base
and shadow consumed identical candidate-produced Q/K/V. Leave-one-sequence-out
global per-layer bases used oracle projection coefficients on the held-out
residual:

| Rank | Global recovery | Minimum sequence | Minimum block entry | Positive layers |
|---:|---:|---:|---:|---:|
| 2 | 0.4004695221 | 0.3157818897 | 0.2520495994 | 16/16 |
| 4 | 0.4286862133 | 0.3469467122 | 0.3253174554 | 16/16 |
| 8 | 0.4692526182 | 0.3874984380 | 0.4439671669 | 16/16 |

All candidates passed the finite, every-sequence, every-block-entry, and
positive-layer conditions. All failed only the prospectively frozen global
recovery threshold of 0.50. Reset replay and every post-run authentication
check passed; confirmation remained unopened. Result SHA-256 is
`c636ad124d570f3675a36f0a23b276ba2e4cd4f5efc23dbf98cc10cd2cfd8e33`;
trace-manifest SHA-256 is
`1f255a59a20089abe4d6805c625a119c167b71153bc21f5edbfcf0fd8050f461`.
See the
[archived capacity evidence](../reports/olmoe_q7_retrieval_episodic_residual_capacity_2026-07-29/summary.md).

This closes only rank-at-most-8 global per-layer subspaces with oracle
coefficients. No causal coefficient fit or runtime correction was authorized.
The next train-only capacity boundary is a dynamic per-head episodic
logit-mass oracle under the existing state and logical-read ceilings.

#### Dynamic per-head episodic-mass oracle

The prospectively frozen train-only capacity experiment held the authenticated
K256 representation, all-head payload, oracle schedule, cache state, and
logical-read contract fixed. At every record/read-row/layer/head coordinate it
selected one multiplier from
`gamma={0,1/8,1/4,1/2,1,2,4,8}` by minimizing error to the W128 teacher's
probability mass on the eight scheduled source positions. The resulting
pre-`W_o` counterfactual was projected through the authenticated BF16 output
projection and compared with the exact native W128-minus-K256 residual.

The mass objective behaved as intended: mean absolute mass error improved from
0.0445126662 at the gamma-one base to 0.0084754603 and did not regress at any
coordinate. Output recovery nevertheless moved in the wrong direction:

| Frozen recovery check | Result | Requirement |
|---|---:|---:|
| Global | -0.1089124543 | >=0.50 |
| Sequences passing | 0/8 | 8/8 at >=0.25 |
| Position 96 block entry | -0.0838671661 | >=0.25 |
| Position 104 block entry | -0.1344610650 | >=0.25 |
| Position 112 block entry | -0.0262677422 | >=0.25 |
| Position 120 block entry | -0.0686255750 | >=0.25 |
| Positive-recovery layers | 1/16 | >=12/16 |

All eight sequence recoveries were negative. Base output and counters,
historical trace tensors, exact metric/code replay, reset replay, and every
post-run authentication check passed. The real-model direct-gamma
qualification is deliberately limited to layer zero, whose input state is
identical before the intervention; its analytic mass/output/projected
counterfactual agreed to about `1.2e-7`. Directly changing layer zero makes
later-layer input states different, so later direct comparisons are causal
diagnostics rather than same-state parity. The shared native attention kernel
has separate unit coverage over the complete gamma grid.

The archived
[head-mass evidence](../reports/olmoe_q7_retrieval_episodic_head_mass_oracle_2026-07-29/summary.md)
is rooted by parity
`569218dbb3ba8667c15ff932e0f9dffe5575774418080cc4e4239062fe2c6d01`,
protocol
`fe09689452e6ae4f1a1b15332c61c1cc990cfc29b6a8b0d5a1758d9490a93af5`,
result
`f7060e7373c5faf8f154891e93efad35659723d8e3f04d83638b62fa9cf72596`,
trace manifest
`93df0a554744b97e7436b9a8b4bb71473bc21fa9f6c90985431274859164e0b6`,
and immutable trace DSO
`6c466bb75a508bd7f8b9173667e7bd9d8433d91c3be818db25f01a495be6d2da`.
Confirmation remained unopened. The result closes only independent-head
scheduled-source-mass matching on this exact grid. It authorized no gamma
predictor, causal integration, or Milestone 3 progression.

#### Joint output-targeted per-head gamma oracle

The next cached, same-state train-only screen targeted the exact
post-`W_o` residual directly. For each head it derived the two value directions
`q = R/mr - B` and `d = E/me - R/mr` from authenticated beta-zero traces.
Every non-base gamma code contributed `q + p_gamma*d`, where
`p_gamma = gamma*me/(mr+gamma*me)`; code 4 remained anchored to the exact
native base. All 16 heads were optimized jointly through the BF16 output
projection, including cross-head terms.

Two prospectively frozen arms were reported. A continuous per-head box
relaxation is an optimistic superset of the eight-code choices and therefore
provides the decisive capacity ceiling. A deterministic discrete solver used
multistart coordinate descent and exhaustive one- and two-head moves; its
claim is local optimality, not global optimization over `8^16`.

| Frozen recovery check | Continuous optimistic bound | Discrete direct float32 | Requirement |
|---|---:|---:|---:|
| Global | 0.22738059544921096 | 0.1997680396822742 | >=0.50 |
| Sequences passing | 1/8 | 0/8 | 8/8 at >=0.25 |
| Block entries passing | 0/4 | 0/4 | 4/4 at >=0.25 |
| Positive-recovery layers | 16/16 | 16/16 | >=12/16 |

Continuous sequence recoveries were
`0.218541, 0.222868, 0.245852, 0.263555, 0.221380, 0.235512, 0.208633,
0.207381`; its position-96/104/112/120 block recoveries were
`0.187912, 0.165185, 0.184378, 0.186928`. Discrete direct sequence
recoveries were
`0.181123, 0.200645, 0.224646, 0.227986, 0.189272, 0.207653, 0.186098,
0.184277`; its block recoveries were
`0.171932, 0.150288, 0.170368, 0.163029`.

The continuous certificate's maximum relative objective gap was
`3.010281e-08` and its summed absolute bound was `7.465143e-07`, far too
small to affect the gate. Gram asymmetry was zero; the minimum normalized
eigenvalue was `-7.069599e-16`, consistent with factor-construction
roundoff. Maximum per-head q/d versus float32 pre-`W_o` discrepancy was
`6.199955e-08`, selected mixed-code projected discrepancy was
`1.409902e-07`, and quadratic versus direct global recovery differed by
`4.147716e-11`. Deterministic replay, exact non-regression against code 4,
one-/two-head local optimality, and all post-run authentication checks passed.

The archived
[joint-gamma evidence](../reports/olmoe_q7_retrieval_episodic_joint_gamma_oracle_2026-07-30/summary.md)
has frozen protocol SHA-256
`aa03a71e3dd9e1fbb413a7773d57189c41029c26b2f4372b2fb7a26744305d24`
and result SHA-256
`1329a51bac71cb81f44494c8ef70cb23a631eacebac5540b39e2e98ed5e30ea5`.
It inherits trace-manifest root
`93df0a554744b97e7436b9a8b4bb71473bc21fa9f6c90985431274859164e0b6`
and parity root
`569218dbb3ba8667c15ff932e0f9dffe5575774418080cc4e4239062fe2c6d01`;
solver/evaluator source roots are
`5c6bf5c4680349b8127ed9dca1bb1ad2f92f3691110eb2025e812cd84c235395`
and
`084e513d78ab4a9c996e351a0927685bd7fbe02ffe2a4f3f08ebff91ba9e094e`.

The optimistic continuous superset fails the global, every-sequence, and
every-block thresholds, so further optimization of the discrete scalar-mass
grid is not justified. This closes only the cached, same-state bounded affine
per-head `(q,d)` family at fixed K256. Confirmation remained unopened; no
predictor, native causal integration, Milestone 3 pass, or end-to-end
attention substitution was authorized. The next attention experiment must
add new value directions or use a different memory mechanism.

#### Per-slot product-simplex value oracle

The next train-only screen added real value directions without changing the
fixed K256 state or KV-read schedule. A native trace recorded the eight exact
BF16 episodic values and their normalized masses for every read
row/layer/head. The constructible arm optimized a nine-way simplex containing
the regular-cache conditional mean and those eight values. A ten-way
optimistic hull added the exact native head output, making it a superset of
every per-slot-logit result over the same value set.

| Frozen recovery check | Constructible 9-way | Optimistic 10-way | Requirement |
|---|---:|---:|---:|
| Global | 0.3844378107 | 0.3844378142 | >=0.50 |
| Sequences passing | 8/8 | 8/8 | 8/8 at >=0.25 |
| Block entries passing | 4/4 | 4/4 | 4/4 at >=0.25 |
| Positive-recovery layers | 16/16 | 16/16 | >=12/16 |

The cached V2 solve used a deterministic active-set accelerator with the
pairwise block Frank-Wolfe solver as a fail-closed fallback. Every accepted
row retained a full product-simplex objective-gap certificate. The
constructible arm's maximum gap was `7.03e-14`; the optimistic arm's was
`5.90e-11`, far too small to bridge the failed global threshold. Exact task
replay, float32 direct/quadratic parity, non-regression, all artifact/source
post-checks, and confirmation blindness passed. Eight CPU workers completed
both arms and their full replays in 91.59 seconds.

The archived
[per-slot evidence](../reports/olmoe_q7_retrieval_episodic_slot_simplex_oracle_2026-07-30/summary.md)
binds capture-report SHA-256
`18218d3a7dbcae731ae42b85cefc09a20ab738ad15531bae3be74c17368d8258`,
cached protocol SHA-256
`f3be957ec0c13d0f49c85a2fa149611307de756f2be82165098a43263bb78ce3`,
and result SHA-256
`2e8e9b7d5f33d33c0e8c642a50359da785d0690d8260ed9f65837b14cd93a5bf`.

This is a decisive failure for reweighting the current regular aggregate and
eight episodic values, not for all same-state attention changes. The smallest
remaining same-state expansion is to expose separately the local and selected
older values that the regular aggregate currently hides. Any later mechanism
that reads or stores additional values must account for those resources
prospectively.

#### Full-visible C28/C29 product-simplex oracle

The prospectively frozen successor exposed every value the bounded native
runtime already reads. For each query head, constructible C28 contains 16
chronological local values, four selected-older values in native score order,
and eight episodic values. Optimistic C29 adds the exact native head output as
one extra anchor. The solver optimizes the per-head simplexes jointly through
the authenticated output projection against the same-state W128-minus-K256
residual.

| Frozen recovery check | Constructible C28 | Optimistic C29 | Requirement |
|---|---:|---:|---:|
| Global | **0.6653937751** | **0.6653865288** | >=0.50 |
| Minimum sequence | **0.6447006551** | — | >=0.25 |
| Minimum block entry | **0.6306278392** | — | >=0.25 |
| Positive-recovery layers | **16/16** | **16/16** | >=12/16 |

The authoritative C28 arm passes the global, every-sequence,
every-block-entry, and positive-layer conditions. C29 also passes its
optimistic qualification. Deterministic replay, certificate checks, trace and
projection authentication, source/manifest authentication, and every
post-solve check passed. Nested C10 and C16 top-mass diagnostics recovered
0.5335805245 and 0.6021187653, respectively, but the frozen protocol gives
them no progression authority.

This expansion adds no deployed KV state or KV-read traffic. The fixed
attention state remains 10,534,912 bytes and logical attention-plus-episodic
traffic remains 714,866,688 bytes, or 33.0305% of the dense full-context
reference. The confirmation split remained sealed. The
[frozen report](../reports/olmoe_q7_retrieval_episodic_full_visible_simplex_oracle_2026-07-30/README.md)
is rooted by result SHA-256
`a8711f07fdcbe48b16ebe962aae1962e4613873640eba20bc7fa88238216aee1`.

This is a train-only same-state **capacity pass**. It authorizes a causal
28-logit selector trained from inference-available state. It does not show
that the coefficients are causally learnable, pass a native counterfactual
rollout, generalize to development or confirmation, qualify a package policy,
or pass Milestone 3.

#### Learned content and phase selectors

The first learned successor reconstructed the packaged post-QNorm, pre-RoPE
query and combined a rank-4 content projection with inference-available source
masses. The second retained the cheaper mass selector and added an eight-step
schedule-relative table. Both were evaluated out-of-fold on the same eight
training records used for model selection.

| Frozen train-only result | FP32 global | BF16 global | Minimum sequence | Minimum block | Positive layers |
|---|---:|---:|---:|---:|---:|
| Rank-4 query content plus mass | 0.2542615526 | **0.25422074198** | 0.23161600085 | 0.18371154473 | 16/16 |
| Eight-phase table plus mass | **0.2618976463** | **0.2618728353** | 0.2405241062 | 0.2244750908 | 16/16 |

The content protocol/result SHA-256 roots are
`0a58ba3a59d2f0f816046ca28aac304baf7663ef890a6b298f0cc7277613d051`
and
`9ea504f83a487584cb9ae2127565674a8e341ca58f6777a03514b0c9a281995c`.
Its conservative traffic fraction is 36.8096% of dense. The phase
protocol/result roots are
`8cb1c7b0e9a6bc2d23839cdbf4de973e66616cccc86e980e6a151d4f2b773987`
and
`52360cf47cb2eeab52e595961f436e4c1e7b79db6cdaa339b7f699d3290883ed`.
Its four BF16 block-entry recoveries are 0.314588398, 0.228395562,
0.261696236, and 0.224475091.

The phase deployment artifact contains 82,944 parameters in 165,888 BF16
bytes. Its conservative total is 736,100,352 logical bytes per 128-token
sequence, or 34.0116% of dense, below the exact-51-head ceiling. All resource,
authentication, finite-value, masking, zero-model, deterministic replay,
schedule-shift, and FP32/BF16 parity checks passed. Phase conditioning
improved the preceding mass-only BF16 recovery by only **0.0040699**, however,
and remained far below the frozen 0.50 global requirement.

These are train-only model-selection outcomes, not independent generalization
evidence. Both learned classes are closed. They authorize no native
integration, development, confirmation, or package change. The next
directional experiment is a blockwise-QK feature controller that directly
models query-to-key compatibility rather than only source mass, query
content, or answer-span phase.

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
and BF16-output parity. This development result froze the policy for the final
attempt. Its sparse end-to-end elapsed time was 1.1565x dense (15.65% slower),
and the traffic ratio is modeled from the v2 serialized layout rather than
measured with DRAM counters.

The independent 8-sequence/256-position final raw report subsequently passed
every frozen threshold:

| Measure | Final raw result | Requirement |
|---|---:|---:|
| Mean KL | 0.0040412880 | <= 0.05 |
| Top-1 agreement | 0.98828125 | >= 0.90 |
| NLL delta | +0.0048289299 | <= +0.05 |
| Final-hidden relative L2 | 0.0477494113 | <= 0.10 |
| Mean active fraction | 0.2138000677 | <= 0.25 |
| Modeled physical cold traffic | 0.4113713394 | <= 0.45 |
| Global micro candidate recall | 0.9994058295 | >= 0.95 |
| Worst-layer mean recall | 0.9939428640 | >= 0.95 |

The original wrapper nevertheless recorded
`final_holdout_consumed_with_error`. Its verifier compared hashes made from
two different representations: full records with the canonical `input_ids`
object envelope versus the evaluator's first 33 scored tokens in a bare list.
A separate no-model postmortem adjudicator reconstructed the canonical token
identities and verified all raw primitive evidence, frozen artifacts, and
attestations. The semantic-memory gate therefore passes **by postmortem
adjudication**, not by a pristine runner result.

The raw report was prospectively hash-sealed about 13 minutes after the error;
the original result did not contemporaneously bind it. Artifacts and native
libraries are host-bound, the 41.1371% traffic number is cache-line modeling
rather than measured DRAM, and the final sparse pass was 1.1449x dense
(295.3364 versus 257.9552 seconds). Latency was measured but was not a frozen
gate. The confirmation covers only 8 sequences and 256 positions. See the
[preserved evidence summary](../reports/native_bitnet_m2_final_audit/49df50cc01c96844ab3e7015d66c8899025dad4e1f7f01a450f97677751b36f2/summary.md).

## Packaged DIP token-runtime confirmation

The semantic-gate result has now been tested inside the complete C++ token
runtime rather than only through the all-layer substitution evaluator. This
is a separate, fixed **non-holdout** integration suite; it does not reuse the
consumed Milestone-2 holdout.

Before execution, `install-native-bitnet-semantic-memory` authenticates the
frozen policy, passing adjudication, source-package inventory, packed base
artifact, and v2 coordinate index. It derives a new package and never changes
the policy-bound source package. The installed index is the runtime policy:
the C++ loader obtains q/C/K, RMS, and audit behavior from its authenticated
layer headers. The derived manifest declares all 30 MLPs substituted,
`dense_fallback: false`, and
`native_bitnet_dynamic_input_pruning_v2` as its MLP mode.

The standalone executable adds an independent native preflight. Before model
mapping or thread-pool construction, it matches the exact manifest digest and
byte count against compiled deployment trust roots, rejects symlinks and
inventory drift, hashes every packaged file, and cross-checks the source
package, base records, index, policy, and adjudication. Model dimensions,
heads, context/vocabulary limits, paths, attention settings, RoPE/RMS values,
and EOS IDs `128001` and `128009` are then derived from those authenticated
files. The executable directly links its kernels and has no Engram
shared-library dependency. The Python harness also requires the exact manifest
and executable hashes and rechecks them after all prompt processes before
publishing the report.

The evaluator runs eight unique prompts for four greedy tokens each and, by
default, resets the runtime and repeats each prompt. It compares generated IDs
against the previously frozen native C++ controller-stage reference and
checks:

- the runtime reports the DIP backend;
- absolute prefill/decode positions advance exactly;
- stage calls and semantic calls equal generated tokens × 30 layers;
- semantic rows equal processed positions × 30 layers;
- selected-record counts are bounded;
- kernel and global-metadata traffic are independently recomputed;
- reset yields the same token IDs, zeroes counters, and reproduces structural
  counters; and
- global and per-prompt mean activity and complete modeled traffic remain below
  25% and 45%.

The recorded result passes:

| Measure | Result | Requirement |
|---|---:|---:|
| Prompts / reference tokens | 8 / 32 | at least 8 / 32 |
| Greedy token-ID agreement | 32/32 | at least 90% |
| Exact prompts | 8/8 | at least 75% |
| Global / maximum-prompt mean active fraction | 0.2156017260 / 0.2258916324 | both at most 0.25 |
| Complete modeled cold bytes | 30,153,074,432 | independently recomputed |
| Global metadata bytes | 194,304 | included above |
| Global / maximum-prompt mean traffic fraction | 0.4116115605 / 0.4129835480 | both at most 0.45 |
| Runtime invariants | all pass | all pass |

The rebuilt-core run used 12 CPU threads and took 390.4183 seconds across first runs, reset
replays, and per-process package authentication. Semantic and attention
counters/timings describe the first generation; replay structural counters
are compared with that snapshot. No latency threshold was applied, and the
traffic counter is a deterministic cache-line model rather than measured
DRAM.

Exact match means greedy token IDs, not hidden-state or logit parity. Reset
similarly proves token/counter structure, not hidden-state identity. The
longest processed context is 14 positions, below W=16, so this suite does not
test eviction or older-context retrieval. The small 8×4 suite proves
package/runtime integration, not broad model quality.
The full report and reproduction commands are in the
[native chat-binding summary](../reports/native_bitnet_dip_chat_runtime_2026-07-27/summary.md).

The same production runtime is now exposed through a versioned native C ABI
and the `NativeBitNetDIPTokenRuntime` Python owner. A direct raw-token
comparison generated the same token and identical structural semantic metrics
as the standalone executable, and reset replay on the persistent mapped
handle also matched. The actual `chat-native-bitnet` CLI then processed a
17-token rendered conversation and generated `Hello`, crossing W=16 without a
Transformers model shell or dense semantic fallback.

The subsequent bounded-attention protocol tests exact lengths
16/17/18/24/32. At length 32 the complete runtime reports 480 evictions,
60,000 older candidates scored, 34,800 older entries selected, 1,200 sink
insertions, and 5,654 accepted heavy-hitter updates, with constant
7,477,440-byte state and exact reset replay. All analytical counter bounds
pass. See the
[native attention summary](../reports/native_bitnet_dip_attention_confirmation_2026-07-27/summary.md).
This establishes cache mechanics, not long-context quality relative to dense
attention.

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
decision. The original generic Gates 3–4 retain random/synthetic fixture
reports, but the separate native-BitNet track now has trained-model bounded
attention, exact-residual controller, compiled-operator, incremental
generation, C++ orchestration, and DIP package/token evidence. These
source-specific results do not retroactively qualify the generic fixture
pipeline. `engram evaluate-e2e` measures student NLL/perplexity, teacher KL,
top-1/top-5 agreement, category accuracy, repetition, and fixed examples
against a cached Hugging Face teacher. Model IDs are downloaded automatically,
while local model directories remain offline-capable. Trained SmolLM2 semantic
interventions have run, but no trained generic compiled-package Gate 5
evaluation has, so no generic Gate 5 quality target is claimed.

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
