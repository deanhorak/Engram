# Fixed-K256 episodic logit-bias train screen

This report archives the completed V2 train-only branch that tested whether a
single shared scalar could repair the fixed-K256 all-head episodic payload
cache. The cache, oracle write/read schedule, `W16/C8/K4/S2` base attention,
and all 256 layer-head pairs stayed fixed. Candidate `gamma` changed only the
episodic softmax partition:

```text
beta = float32(log(gamma))
episodic_score' = episodic_score + beta
```

The operation adds no K/V state, reads, writes, or scratch. It does not change
the episodic keys or values.

## Authenticated evidence

The JSON files in this directory are byte-for-byte copies of the active V2
parity, protocol, and result.

| Artifact | SHA-256 |
|---|---|
| [Beta-zero V1/V2 parity](beta_zero_parity.json) | `8e3c75de7fbb156a6d1e2f4f8053ae6bd4dccd35b73995dec810f3dc75911234` |
| [Frozen V2 protocol](protocol.json) | `025ff45e41966faf033338ffcac0c3fc1f93b40ed7676c36f189ba57485e8be7` |
| [V2 train screen](train_screen.json) | `19d08ce9eb4b673d423e9781a491e25ec03bdec09467a43e7be1881874ef2287` |
| Evaluator source | `8ab1ec4aaab30a8e218a33f52f81637e22918e3cc9f0773988a97d09936b2802` |
| Frozen native V2 DSO | `612f1d5c2b86f20574285039a1e2110638ceaa16337b6ad8c0f00913b0add383` |

Beta-zero parity covered four native sequence forwards and 512 token steps.
Outputs, counters, and reset behavior matched the V1 all-head episodic ABI
exactly. The V2 run then reauthenticated every bound source, package, model,
library, checkpoint, protocol, parity, and train-corpus artifact after
execution.

## Result

The strict reference was the authenticated full-context `M2` training
population at mean answer cross-entropy 1.005444 and worst-record
cross-entropy 1.227907. A candidate had to improve both values strictly and
regress no training record.

| Arm | Mean answer CE | Worst answer CE | Records no worse than `M2` | Gate |
|---|---:|---:|---:|---|
| Full-context `M2` reference | 1.005444 | 1.227907 | 8/8 | Reference |
| Historical K256, `beta=0` | 1.224460 | 1.327343 | 1/8 | Attribution only |
| `gamma=1/2`, `beta bits=0xbf317218` | 1.461414 | 1.669250 | 1/8 | Fail |
| `gamma=1/4`, `beta bits=0xbfb17218` | 1.883818 | 2.288258 | 0/8 | Fail |
| `gamma=3/16`, `beta bits=0xbfd644dc` | 2.161750 | 2.595642 | 0/8 | Fail |
| `gamma=1/8`, `beta bits=0xc0051592` | 2.725091 | 3.430532 | 0/8 | Fail |

All four nonzero candidates executed all eight training records; none was
skipped. All four passed their counter and resource contracts and failed
semantic quality. The total-failure rule retained `gamma=1/2` solely as the
lexicographically best failed candidate under `(worst CE, mean CE, candidate
order)`. Its reset replay passed exactly. This diagnostic retention is not a
promotion, and it does not make `gamma=1/2` preferable to the stronger
historical `beta=0` base.

Each arm remained at the fixed K256 upper bounds: 714,866,688 logical
read-plus-write bytes per sequence, 33.030523% of the dense full-context K/V
reference; 10,534,912 bytes of attention state; and 4,864 bytes of scratch.
The sweep used 32 population sequence forwards plus one failed-candidate
replay. No dense-teacher forward ran. Development was not authorized, and the
reserved confirmation split remained unopened.

## Decision and next boundary

The result closes shared fixed-K256 episodic logit-mass calibration. Reducing
the episodic partition monotonically worsened this train population, so another
scalar `gamma` guess is not justified.

The next bounded experiment is a **same-state shadow residual capacity
screen**, not another cache-layout or scalar-calibration sweep:

1. Run the deployable K256 path beside a train-only W128 shadow that consumes
   the exact same candidate-produced Q/K/V at every layer and token.
2. Fix `beta=0` prospectively as the K256 base. The authenticated comparison
   already rejects `gamma=1/2` as worse; it remains diagnostic-only and does
   not consume a second shadow trace.
3. Measure the output-subspace ceiling of the post-`W_o` residual before
   implementing a production correction loader or kernel. Only evidence that a
   small, traffic-bounded subspace can recover the shadow output can authorize
   a prospectively frozen low-rank fit.

This next screen remains train-only. Its operator, ranks, resource accounting,
selection rule, and causal gate must be frozen before fitting. Development and
confirmation remain unavailable unless that new capacity and causal
progression boundary passes.

That screen has since completed. Ranks 2, 4, and 8 passed every
per-sequence, block-entry, and positive-layer condition, but all missed the
frozen 50% global-recovery gate. See the
[same-state residual-capacity report](../olmoe_q7_retrieval_episodic_residual_capacity_2026-07-29/summary.md).
The recommendation above is retained as the prospective record and is
superseded by that later negative result.
