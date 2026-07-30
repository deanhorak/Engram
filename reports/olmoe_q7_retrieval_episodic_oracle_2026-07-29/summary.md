# Payload-only episodic oracle: train-screen result

This report tests a deliberately stronger retrieval mechanism than a learned
head selector. The native `W16/C8/K4/S2` attention path receives an exact,
causal oracle schedule: it writes the four eight-token source payloads into a
32-slot per-layer BF16 K/V cache, then reads the correct eight-token span into
the same attention softmax while predicting each corresponding answer block.
Selection error is therefore removed from the experiment.

## Frozen evidence

- Protocol:
  `protocol.json`
- Protocol SHA-256:
  `1e7b89e5b376430b82456bf306e50a0fb7c0cb9ed75b0d4e400ad7950b517cce`
- Train screen:
  `train_screen.json`
- Train-screen SHA-256:
  `b2daa5eff271b6f030c01e8a4854a602f7b2907f7af487d38ab750468bbc42cc`
- Frozen native library SHA-256:
  `877dac1dcff9ab6749a2abc5fe2f9f21b9660fadafe8dc7714e26ea77629221e`
- Confirmation split opened: **no**
- Runtime: 12 CPU threads; no GPU inference

The full-`W128` packaged-Q7 control passed every overall and source-depth
semantic check:

| Metric | Control | Gate |
|---|---:|---:|
| Teacher-to-native KL | 0.001892 | <= 0.05 |
| Teacher top-1 agreement | 1.000000 | >= 0.90 |
| Target NLL delta | -0.002330 | <= +0.05 |
| Final hidden relative L2 | 0.047297 | <= 0.10 |

The payload-only episodic candidate passed every systems, resource,
counter-stream, reset-replay, and post-run authentication check, but failed
three of the four semantic metrics:

| Metric | Candidate | Gate | Result |
|---|---:|---:|---|
| Teacher-to-native KL | 0.446656 | <= 0.05 | fail |
| Teacher top-1 agreement | 0.921875 | >= 0.90 | pass |
| Target NLL delta | +0.557528 | <= +0.05 | fail |
| Final hidden relative L2 | 0.428062 | <= 0.10 | fail |

The candidate answer cross-entropy was 1.224460 mean and 1.327343 worst,
versus 1.005444 mean and 1.227907 worst for the existing exact-51-head M2
training checkpoint. It regressed on seven of eight records.

## Failure localization

The error is concentrated at the first prediction of each eight-token answer
block:

| Answer positions | Mean KL | Mean NLL delta | Hidden L2 | Top-1 |
|---|---:|---:|---:|---:|
| Four block-entry rows | 1.974513 | +2.750141 | 0.638946 | 0.437500 |
| Remaining 28 rows | 0.228391 | +0.244297 | 0.397936 | 0.991071 |

Those 32 block-entry observations across the eight records account for
55.26% of total answer-position KL and 62.12% of positive answer-position NLL
regression. The pattern is causal and interpretable. At each block boundary,
the model must use the requested key identity to enter the right source
record. Once the first code token has been supplied, recent answer tokens
provide a strong continuation cue. The cache retained only numeric payload
K/V rows, omitting the source label token that associates a payload with
`A`, `B`, `C`, or `D`.

## Resource result

The measured upper-bound attention reads were 710,672,384 bytes per
128-token sequence, 32.8367% of dense full-context K/V reads. Including
4,194,304 cache-write bytes gives 714,866,688 bytes, or 33.0305%. Total
attention state was 10,534,912 bytes and scratch was 4,864 bytes. These are
comfortably inside the frozen 45% traffic ceiling.

## Decision

Do not train a selector for this payload-only cache. Oracle selection has
already shown that its representational contract is insufficient.

The next bounded capacity test should store the source label token followed by
the eight payload tokens for each fact: four nine-token spans, 36 slots total.
That directly tests the missing identity cue while remaining below both the
state and traffic ceilings. It should first run as a train-only,
candidate-only cross-entropy screen against the authenticated M2 checkpoint.
Only a strict mean and worst-case improvement with no record regression
justifies another expensive dense-teacher semantic run.
