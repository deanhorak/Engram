# Same-state episodic residual-capacity train screen

This report archives the prospectively frozen, train-only capacity test that
followed the fixed-K256 episodic logit-bias result. The deployable branch was
fixed to the stronger historical `beta=0` K256 payload. At every layer and
token, a non-intervening W128 shadow consumed the exact same post-RoPE Q/K/V
as that branch. If `b` is the base attention output after `W_o` and `f` is the
shadow output after the same projection, the target was the same-state
residual `f-b`.

This is deliberately an optimistic capacity ceiling. For each leave-one-
sequence-out fold, the per-layer output basis was learned from the other seven
training records, while coefficients for the held-out residual were chosen by
oracle projection. It tests whether the missing output lies in a small global
per-layer subspace; it does not test whether a causal predictor can infer those
coefficients.

## Authenticated evidence

The JSON files in this directory are byte-for-byte copies of the active
parity, protocol, result, and compact trace manifest.

| Artifact | SHA-256 |
|---|---|
| [Same-state trace parity](trace_parity.json) | `56e4b730dc7580895e952a5746d105f5ca01ec36d83f6b37044c5f331061f8dd` |
| [Frozen capacity protocol](protocol.json) | `584302d17a3224cda1b61dfe1f62685497fa5a0dc335cfc0a074439456ee1606` |
| [Capacity train screen](train_screen.json) | `c636ad124d570f3675a36f0a23b276ba2e4cd4f5efc23dbf98cc10cd2cfd8e33` |
| [Trace manifest](trace_manifest.json) | `1f255a59a20089abe4d6805c625a119c167b71153bc21f5edbfcf0fd8050f461` |

The parity gate passed before freezing the screen. The run then captured all
eight training records, reauthenticated every frozen input after execution,
and passed its deterministic reset replay. The reserved confirmation split
remained unopened.

The eight local `safetensors` trace shards are intentionally not copied into
the repository. They are ignored working data, approximately 12 MiB per
record. Their identities and shapes remain bound by
[`trace_manifest.json`](trace_manifest.json), so the archived report discloses
exactly which local traces produced the result without adding roughly 96 MiB
of derived arrays to Git.

## Frozen gate and result

The screen evaluated ranks 2, 4, and 8. A candidate had to satisfy all of the
following:

- finite measurements;
- at least 50% global squared-Frobenius residual recovery;
- at least 25% recovery on every held-out sequence;
- at least 25% recovery at every answer-block entry position
  (`96`, `104`, `112`, and `120`);
- positive recovery in at least 12 of 16 layers.

| Rank | Global recovery | Minimum sequence recovery | Minimum block-entry recovery | Positive layers | Gate |
|---:|---:|---:|---:|---:|---|
| 2 | 0.4004695221 | 0.3157818897 | 0.2520495994 | 16/16 | Fail: global recovery |
| 4 | 0.4286862133 | 0.3469467122 | 0.3253174554 | 16/16 | Fail: global recovery |
| 8 | 0.4692526182 | 0.3874984380 | 0.4439671669 | 16/16 | Fail: global recovery |

Every rank passed the finite, per-sequence, block-entry, and positive-layer
conditions. Rank 8 was the strongest candidate, but its 46.9253% global
recovery remained below the prospectively frozen 50% threshold. Thus the
screen failed solely on the global capacity condition. No rank was promoted,
no causal coefficient predictor was authorized, and no correction artifact
or package-format change followed.

## Decision and next boundary

This result closes only **rank-at-most-8 global per-layer output subspaces
with oracle held-out coefficients** for this fixed K256 same-state residual.
It does not close higher ranks, input-conditioned or token-varying bases,
per-head control, learned causal coefficient prediction, or episodic memory as
a class.

The next bounded experiment is a **dynamic per-head episodic logit-mass
oracle**. It should hold the authenticated K256 payload, schedule, and
`beta=0` base fixed, then measure whether choosing episodic mass separately by
layer, head, and causal query can recover the W128 answer behavior under the
same logical-read and state ceilings. The oracle must be frozen as a
train-only capacity test before any learned controller is fit. A negative
oracle result closes that mechanism; a positive oracle result only authorizes
a causal predictor experiment.

The OLMoE Q7 Milestone 2 semantic path remains passed. Milestone 3 attention
substitution remains blocked, and development or confirmation promotion is
not authorized by this failure.
