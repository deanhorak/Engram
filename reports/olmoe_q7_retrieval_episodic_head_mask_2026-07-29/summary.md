# Head-gated episodic payload cache: K51 and ranked-prefix train screens

This experiment tests whether the payload-only episodic cache failed because
it forced retrieved K/V entries into every layer/head softmax. It reuses the
exact causal four-span oracle schedule, but exposes each span only to the
authenticated 51 layer/head pairs selected by the existing `M2` answer-loss
fit. No mask is fitted here, and neither development nor confirmation is
opened.

## Authenticated evidence

| Artifact | SHA-256 |
|---|---|
| [Frozen K51 protocol](k51_protocol.json) | `38ceb03c5ab8a18038aea57728bdca9f405ec46cd5311a0ba8569059843a5fd6` |
| [K51 train screen](k51_train_screen.json) | `18bc2ec7ee55712f85d237ab0159ff160add37dc840dfcea3028b216f0062852` |
| [Frozen ranked-prefix protocol](rank_sweep_protocol.json) | `e238b5cfc4359422abc96c227f4010ade747ff161c946b2edae14c171a7bf04c` |
| [Ranked-prefix train screen](rank_sweep_train_screen.json) | `a6fb045bbad526411b8318e7de2412ea56d69b3f2be0d977ca6819635c0718da` |
| Frozen native DSO | `7f9e3319adc07e647e5479179af19d9a7ff89fa02daec5c9bbf09692c059b715` |
| Fixed `M2` mask | `49802a2d37abd44e4015e87633c9a321e333315b9400f6a69d4713ec2270b446` |

The DSO adds a versioned head-gated episodic-open ABI. Inactive layers execute
the exact legacy attention step and allocate no episodic bank. Active layers
store full causal BF16 K/V rows, but only selected query heads deduplicate,
score, normalize against, and read the cached span. The previous all-head ABI
is unchanged and has exact all-ones parity with the new path.

## Result

K51 failed the train-only progression gate:

| Candidate | Mean answer CE | Worst answer CE |
|---|---:|---:|
| Same-policy no-cache `M0` | 7.647114 | 7.976308 |
| Existing full-context `M2` reference | **1.005444** | **1.227907** |
| K51 head-gated episodic payload | 1.400569 | 1.694034 |

Only one of eight records improved relative to `M2`. Mean CE regressed by
0.395125 and worst CE by 0.466127. The result still improves dramatically on
the exact same-policy no-cache baseline, so it does not reject episodic
memory; it rejects transferring the old 51-head cardinality directly to this
cheaper cache.

Every systems check passed:

- exact per-position and final counters;
- deterministic reset replay;
- source, package, DSO, checkpoint, and protocol reauthentication;
- confirmation remained unopened;
- 10,010,112 bytes of reported attention state and 4,736 bytes of scratch;
- at most 683,802,624 logical read bytes plus 3,670,016 write bytes;
- 31.7648% upper-bound total traffic relative to dense full-context K/V.

## Ranked-prefix result

The number 51 was imposed by the old full-context 45% traffic boundary, not
learned as the best cardinality for episodic retrieval. The all-head payload
cache previously reached mean CE 1.224460, materially better than K51, while
still using only 33.03% upper-bound total traffic. A separate protocol
therefore froze the complete projected-score order and evaluated larger
prefixes at K64, K96, K128, and K165.

All four candidates executed all eight training records and failed:

| Candidate | Mean answer CE | Worst answer CE | Records improved versus `M2` |
|---|---:|---:|---:|
| K64 | 1.379699 | 1.639418 | 1/8 |
| K96 | 1.328848 | 1.618843 | 1/8 |
| K128 | 1.337958 | 1.621764 | 1/8 |
| K165 | 1.331006 | 1.608617 | 1/8 |

The frozen total-failure rule retained K165 for diagnostic reset replay
because it had the lowest worst-record CE; replay passed exactly. K96 had the
lowest mean, but neither candidate approached the `M2` reference at
1.005444 mean and 1.227907 worst. Every candidate passed its counter,
resource, and authentication contracts. K165 used 702,939,136 total
attention-traffic bytes, 32.4794% of dense full-context K/V. Confirmation
remained unopened, and no development or semantic screen was authorized.

## Decision

The ranked prefixes close cardinality expansion under the transferred `M2`
ordering. They do not close episodic memory: the authenticated K256 all-head
payload result remains better than every ranked prefix at 1.224460 mean and
1.327343 worst while using only 33.0305% upper-bound total traffic.

The next bounded direction therefore fixes that K256 payload, schedule, and
all-head exposure and changes only its native softmax calibration. The kernel
currently inserts episodic scores directly into a much smaller
W16/C8/K4-plus-payload normalization set. A prospectively frozen logit-mass
screen will add `log(gamma)` to every episodic score before the joint softmax,
leaving K/V state and traffic unchanged. Only strict mean and worst
improvement over `M2`, with no record regression, may authorize a later
dense-teacher semantic screen. This remains train-only work; confirmation is
sealed.

## Supersession

The logit-mass experiment proposed above has completed. Its active V2
parity, protocol, result, and decision are archived in the
[fixed-K256 episodic logit-bias report](../olmoe_q7_retrieval_episodic_logit_bias_2026-07-29/summary.md).
This report remains the historical K51/ranked-prefix evidence, but its
next-step recommendation is superseded by that later result.
