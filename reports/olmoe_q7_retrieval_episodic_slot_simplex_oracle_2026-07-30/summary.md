# OLMoE Q7 per-slot value-simplex capacity oracle

## Question

Can fixed-K256 episodic attention recover the exact W128-minus-K256
post-output-projection residual if every query head may independently
reweight the eight exact values that K256 already read?

This is a train-only, cached, same-state capacity experiment. It does not
train a selector, alter hidden or cache state, execute a causal
counterfactual, load the native runtime during the solve, or open the
confirmation split.

## Method

The native trace records, for every read row, layer, and query head:

- the regular-cache weighted value and probability mass;
- the episodic weighted value and probability mass;
- the eight normalized episodic slot masses; and
- the eight exact BF16-decoded episodic value vectors in read order.

The constructible arm exposes a nine-way simplex per head: the regular-cache
conditional mean plus the eight stored episodic values. A second ten-way
optimistic hull adds the exact native head output as an anchor. The optimistic
arm is a superset of every result obtainable by changing per-slot logits while
retaining the same values, so its certified failure is decisive for this
value family.

All 16 heads are optimized jointly after the authenticated BF16 output
projection. A deterministic bulk active-set KKT solver handles the fast path,
with the earlier pairwise block Frank-Wolfe solver retained as a fail-closed
fallback. Every result is judged by the full product-simplex Frank-Wolfe gap,
not by active-set termination. The frozen solve used eight CPU workers,
replayed every layer/arm task exactly, and completed in 91.59 seconds.

## Frozen roots

- V1 native-capture protocol:
  `56b33472ee23353e945abb9741f3b5b16e70965450d023a45cd6223a8d85c4cb`
- V1 trace-parity report:
  `4565a5fcaa2039f4229422243e0f121b3444c89495b4141b39f6358c19645a02`
- Completed slot-trace manifest:
  `0ac40bfa8f41d23627ce9e3ee89283f68828ae02d482c77a43be0d4d17129b04`
- Authenticated cached-capture report:
  `18218d3a7dbcae731ae42b85cefc09a20ab738ad15531bae3be74c17368d8258`
- Cached V2 solve protocol:
  `f3be957ec0c13d0f49c85a2fa149611307de756f2be82165098a43263bb78ce3`
- Cached V2 train result:
  `2e8e9b7d5f33d33c0e8c642a50359da785d0690d8260ed9f65837b14cd93a5bf`
- Native trace DSO:
  `deeee538cbf651793aff8a003b35dca9b64b39796454830b6fb7d57954938f96`
- Active-set solver source:
  `057c2f70f5b154564743648d20fb4d8df796cff89cac63b438181a915af3325e`
- Cached evaluator source:
  `b676b2b8c657ce41777616d9b4f41054555f5975738aa5131d15dc0a8f9d2f13`

The archived JSON roots are under [`v1/`](v1/) and [`v2/`](v2/). The eight
large tensor shards remain in the authenticated work directory and are bound
by the archived manifest rather than duplicated here. The exact seven-file
native/runtime source state that produced this boundary is preserved under
[`source_snapshot/`](source_snapshot/); its `SHA256SUMS` entries match the
source roots frozen in the V2 protocol.

## Result

| Frozen recovery check | Constructible 9-way simplex | Optimistic 10-way hull | Requirement |
|---|---:|---:|---:|
| Global recovery | **0.3844378107** | **0.3844378142** | >=0.50 |
| Sequences passing | 8/8 | 8/8 | 8/8 at >=0.25 |
| Block entries passing | 4/4 | 4/4 | 4/4 at >=0.25 |
| Positive-recovery layers | 16/16 | 16/16 | >=12/16 |
| Frozen gate passed | no | no | yes |

The exact-native anchor improves global recovery by only about
`3.49e-9`, confirming that trace regrouping roundoff is immaterial. The
constructible arm certified all 4096 rows at the strict target; its maximum
objective-gap bound was `7.03e-14`. The optimistic arm certified 4078 rows at
the strict `1e-12` relative target. Its remaining valid per-row certificates
had a maximum gap of `5.90e-11`, and the resulting optimistic global recovery
upper bound was still only `0.3844378142`.

Qualification passed:

- deterministic coefficient and metric replay was exact;
- quadratic and direct float32 error energies agreed within the frozen rule;
- neither direct arm regressed against the exact native base;
- all Gram matrices were factor-defined positive semidefinite;
- slot values remained exact BF16 decodes;
- maximum slot-mass reconstruction error was `1.79e-7`;
- maximum regular-plus-episodic reconstruction error was `8.94e-8`;
- every post-run artifact and source authentication check passed; and
- confirmation remained unopened.

The resource boundary stayed fixed at 10,534,912 attention-state bytes and
714,866,688 combined attention/episodic logical traffic bytes, or 33.03% of
dense full-context KV traffic under the inherited accounting.

## Decision

The gate failed decisively. Even the optimistic convex hull over the current
regular aggregate, all eight already-read episodic values, and the exact
native anchor cannot reach the required 50% global recovery. No per-slot
selector, logit-bias predictor, native causal integration, development run,
confirmation access, package field, Milestone 3 pass, or end-to-end attention
substitution is authorized.

This closes same-state reweighting over the present fixed-K256 value set; it
does not close episodic memory generally. The next defensible experiment must
introduce genuinely new value directions—such as multiple separately
addressable regular/retrieved summaries—or replace the bounded memory
mechanism. More optimization of logits over these same nine directions cannot
cross the gate.
