# OLMoE Q7 full-visible simplex capacity oracle

## Question

Can the bounded native attention state already supply enough value directions
to recover the exact W128-minus-bounded-attention projected residual, before
training a causal selector?

This is a frozen, train-only capacity experiment. It does not train or execute
a selector, run a causal counterfactual, authorize development or confirmation,
or establish a Milestone 3 attention-substitution pass.

## Method

The constructible C28 arm gives every query head a simplex over the 16 local
values, four selected older regular-cache values, and eight episodic values
already read by the native runtime. The optimistic C29 arm adds the exact
native head output as an extra anchor. Both arms are solved jointly through the
authenticated output projection, with deterministic replay and certified
objective-gap bounds.

The frozen run covers eight training sequences, 32 evaluated read positions
per sequence, 16 layers, and 16 query heads. It uses no additional KV state or
KV-read traffic. The fixed native attention state is 10,534,912 bytes; combined
attention and episodic logical traffic is 714,866,688 bytes, or 33.03% of the
dense full-context KV reference.

## Result

| Check | Constructible C28 | Optimistic C29 | Requirement |
|---|---:|---:|---:|
| Certified feasible global recovery | **0.6653937751** | **0.6653865288** | >=0.50 |
| Certified optimistic recovery bound | **0.6654271515** | **0.6654553013** | >=0.50 |
| Every sequence recovery | passed | passed | >=0.25 |
| Every block-entry recovery | passed | passed | >=0.25 |
| Positive-recovery layers | 16/16 | 16/16 | >=12/16 |
| Frozen capacity gate | **passed** | **passed** | pass |

Qualification, deterministic replay, projection authentication, source
authentication, manifest authentication, and all post-solve checks passed.
The protected confirmation split remained unopened.

## Decision

This passes the train-only value-capacity boundary that the earlier nine-value
experiment failed. It shows that the native C28 value set contains enough
information in principle. It does **not** pass Milestone 3: the coefficients
were selected by an oracle using the target residual, not by a causal model
available during inference.

The authorized next experiment is a frozen, train-only causal 28-logit
correction selector with complete CPU state, traffic, and latency accounting.
Development and confirmation remain unauthorized until that selector passes
its own progression gate.

## Frozen artifacts

- `protocol.json`:
  `61e4b6da682bb501b21a0fa41d961f1f092e92e750ff913057c2f20ff4f34734`
- `trace_parity.json`:
  `2e8a42e7bdef2a632cc9fd1796c7e67c44c677ea53067c3df3a64c41d6711ccf`
- `manifest.json`:
  `ec80fb57e1c9c684b9d1e95ad672b69afa30a7915ea427cec9c76ca15219fef4`
- `train_screen.json`:
  `a8711f07fdcbe48b16ebe962aae1962e4613873640eba20bc7fa88238216aee1`
- native trace DSO:
  `6a01d54ef1e57857550219d951dd515dc16001a7e7243425aadd1ea096cece94`

The large trace tensor shards remain in the authenticated work directory and
are bound by the archived manifest rather than duplicated here. The 31 exact
protocol-bound source files are preserved under `source_snapshot/`.

Verify the archive from this directory:

```bash
sha256sum --check SHA256SUMS
(
  cd source_snapshot
  sha256sum --check SHA256SUMS
)
```
