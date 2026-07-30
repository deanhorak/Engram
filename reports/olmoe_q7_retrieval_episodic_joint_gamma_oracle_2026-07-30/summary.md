# OLMoE Q7 retrieval episodic joint-gamma oracle

## Question

Can the fixed-K256 episodic attention arm recover the exact W128-minus-K256
post-output-projection residual by choosing one of eight episodic-mass gamma
codes independently for every state, layer, and attention head?

This is a train-only, cached, same-state capacity experiment. It does not
train a predictor, update hidden or cache state, execute a causal rollout, or
open the confirmation split.

## Method

For each head, the authenticated native trace supplies the beta-zero attention
output `B`, regular and episodic weighted-value numerators `R` and `E`, and
their masses `mr` and `me`. The evaluator forms

```text
r = R / mr
e = E / me
q = r - B
d = e - r
```

Every non-base gamma code is represented as the correction
`q + p_gamma*d`, where
`p_gamma = gamma*me / (mr + gamma*me)`. Code 4 is anchored to the exact native
base correction `(0, 0)`.

All 16 heads are optimized jointly after the authenticated BF16 output
projection, including cross-head Gram terms. A continuous box relaxation
provides an optimistic capacity upper bound. Its exact two-variable per-head
block solver runs for at most 64 sweeps and reports a rigorous float64
Frank-Wolfe objective-gap certificate. The discrete eight-code arm uses
deterministic multistart coordinate descent followed by exhaustive one- and
two-head moves. Its optimality claim is local, not global over `8^16`.

The actual progression metrics are recomputed through the established float32
counterfactual and output-projection path, not taken from the quadratic
surrogate.

## Frozen roots

- Protocol SHA-256:
  `aa03a71e3dd9e1fbb413a7773d57189c41029c26b2f4372b2fb7a26744305d24`
- Result SHA-256:
  `1329a51bac71cb81f44494c8ef70cb23a631eacebac5540b39e2e98ed5e30ea5`
- Cached trace manifest SHA-256:
  `93df0a554744b97e7436b9a8b4bb71473bc21fa9f6c90985431274859164e0b6`
- Inherited trace-parity SHA-256:
  `569218dbb3ba8667c15ff932e0f9dffe5575774418080cc4e4239062fe2c6d01`
- Solver source SHA-256:
  `5c6bf5c4680349b8127ed9dca1bb1ad2f92f3691110eb2025e812cd84c235395`
- Evaluator source SHA-256:
  `084e513d78ab4a9c996e351a0927685bd7fbe02ffe2a4f3f08ebff91ba9e094e`

## Result

| Metric | Continuous optimistic upper bound | Discrete direct float32 |
|---|---:|---:|
| Global recovery | 0.227381 | 0.199768 |
| Sequences at or above 0.25 | 1/8 | 0/8 |
| Block entries at or above 0.25 | 0/4 | 0/4 |
| Positive-recovery layers | 16/16 | 16/16 |
| Frozen gate passed | no | no |

Continuous optimistic sequence recoveries were
`0.218541, 0.222868, 0.245852, 0.263555, 0.221380, 0.235512,
0.208633, 0.207381`. Block-entry recoveries at positions
`96/104/112/120` were
`0.187912, 0.165185, 0.184378, 0.186928`.

Discrete direct sequence recoveries were
`0.181123, 0.200645, 0.224646, 0.227986, 0.189272, 0.207653,
0.186098, 0.184277`. Block-entry recoveries were
`0.171932, 0.150288, 0.170368, 0.163029`.

The continuous objective-gap uncertainty is negligible relative to the failed
thresholds: its maximum relative bound is `3.010281e-08`, and its summed
absolute bound is `7.465143e-07`. The discrete solution is deterministically
replayed, one- and two-head-flip locally optimal, and exactly non-regressive
against the code-4 base under the frozen float64 comparison tolerance.

Numerical qualification passed:

- Gram construction is explicit float64 `A.T@A`.
- Maximum Gram asymmetry is zero.
- Minimum normalized eigenvalue is `-7.069599e-16`, consistent with
  roundoff around a factor-defined positive-semidefinite matrix.
- Maximum per-head q/d versus float32 pre-Wo discrepancy is
  `6.199955e-08`.
- Selected mixed-code projected discrepancy is `1.409902e-07`.
- Quadratic versus direct global recovery differs by
  `4.147716e-11`.
- Every post-run authentication check passed.
- The confirmation split remained unopened.

## Decision

The gate failed. Even the more permissive continuous relaxation cannot reach
the required `0.50` global recovery, `0.25` on every sequence, or `0.25` on
every block entry. This closes only the cached same-state bounded affine
per-head `(q,d)` reweighting family at fixed K256. It does not prove that the
global `8^16` discrete grid is worse than the reported local solution, but the
continuous superset failure makes further optimization of that grid
scientifically unjustified.

No gamma predictor, native causal integration, confirmation access, Milestone
3 pass, or end-to-end attention substitution is authorized. The next attention
experiment must introduce new value directions or a different memory
mechanism; retuning scalar episodic mass is no longer a defensible path.
