# Native OLMoE frozen generation and Q7 performance boundary

The authenticated OLMoE package passes the frozen eight-prompt generation
integration protocol against the untouched BF16 teacher:

| Measure | Requirement | Result |
|---|---:|---:|
| Prompts | at least 8 | 8 |
| Teacher-forced positions | at least 60 | 60 |
| Teacher-forced top-1 agreement | at least 90% | **100%** |
| Greedy reference tokens | at least 32 | 32 |
| Weighted greedy-token agreement | at least 90% | **90.625%** |
| Exact four-token prompts | at least 75% | **87.5%** |
| Cache positions, reset replay, post-run authentication | all pass | **pass** |

Seven prompts are exact. `A healthy software testing strategy should` agrees
on the first generated token, then takes a different valid-looking greedy
branch: teacher ` include a combination of`, native
` include the following:`. All 60 teacher-forced top-1 decisions are exact.

The protocol was frozen before the complete candidate suite under SHA-256
`f731b002e4ca0cfe42e14a220798cab5ced1d1ab7c7e484d07ca3e3855d803e7`.
It authenticates the six untouched teacher shards, teacher reference, prompt
suite, package manifest, and native library. The candidate re-authenticates
the package and library after execution. The Paris prompt was already used by
an earlier one-token smoke; this is disclosed in the protocol and result.

## Performance

The Q7 matrix loop now decodes the canonical eight-code/seven-byte blocks and
loads each BF16 group scale once. It preserves accumulation order. Against the
frozen pre-change library, routes and float outputs are bit-identical at
layers 0, 7, and 15 while median layer time improves by **6.56×–9.24×**.

The five-position `The capital of France is` smoke remains token-identical.
Native execution falls from 13.33 to 2.17 seconds and Q7 time from 13.08 to
1.91 seconds. Cold wall time falls from 61.78 to 32.06 seconds after also
parallelizing package/shard hashing and independent Q7 layer validation.
Strict Q7 structural validation falls from about 16.5 to 2.29 seconds.

## Scope

All prompts plus generation remain within the exact W16 local attention
window. This closes package/generation integration, not long-context
retrieval quality or the full 256-position causal reproduction. Native logits
and final hidden-state diagnostics were the next boundary at the time of this
run. That subsequent boundary now passes; see the
[complete native causal confirmation](../olmoe_q7_native_causal_2026-07-28/summary.md).
