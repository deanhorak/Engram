# Budget-edge representation screens — 2026-07-23

Decision: **close the bounded recurrent and post-hoc low-bit search.**

These screens asked whether a materially different representation could fit
within 45% of dense ideal-Q4 MLP traffic and clear a conservative layer-14
progression ceiling before an expensive all-layer causal run. None reached the
required mean relative L2 of 0.20. Formal development, confirmation, and
external evaluation data remained unopened.

| Representation | Complete traffic | Best layer-14 mean rel-L2 | Scope | Decision |
|---|---:|---:|---|---|
| Four-cycle width-640 recurrent compact Q4 | 44.9293% | 0.308254 | trained; optimistic cache reuse | close |
| Projection-normalized full-width ternary | 41.0013% | 0.631323 | trained hard QAT | close |
| Mixed affine LC-VQ, Q4/ternary/ternary | 44.3482% | 0.336396 | trained hard QAT | close |
| Unrestricted 128-entry VQ, four weights/code | 44.9799% | 0.576865 | initialization | close before QAT |
| Mixed LiftQuant-style 16-to-10/8/10 lattice | 44.4012% | 0.556958 | initialization | close before QAT |

The recurrent arm improved its jointly fitted width-640 base by only 4.84%,
and its byte result assumes later cycles reuse the same decoded compact-Q4
payload from cache. The mixed affine vector arm was the strongest low-bit
training result: 8,192 hard-QAT steps reduced error by 46.44%, but the final
0.336396 result still misses the progression ceiling by a wide margin.

The unrestricted codebook and lifted-binary arms were stopped at their
predeclared initialization guards. This is important: training a representation
that already starts above the guard would spend more compute without evidence
that it can reach the local ceiling, and would expose formal data prematurely.

The checked machine-readable [summary](summary.json) records exact traffic
accounting, metrics, official-asset provenance, and hashes of the ignored
scratch reports. At this point the evidence supported one of two materially
different programs:

1. train a student whose representation and byte budget are native from the
   beginning, using a substantially larger and more representative token
   budget; or
2. explicitly relax the 45% policy toward the quality-passing DIP frontier and
   optimize that contiguous runtime.

Another small post-hoc bit allocation, router, basis, codebook, or residual
sweep was not the next supported experiment. The first program was
subsequently implemented and reached its frozen stop condition; see the
[budget-native grouped-ternary result](../semantic_gate_budget_native_2026-07-23/summary.md).
