# Budget-native grouped-ternary result — 2026-07-23

Decision: **traffic pass, quality fail; stop this configuration before 3M.**

Engram now has an exact full-width grouped-ternary training and artifact path.
Five ternary coefficients are packed per byte, each 128-weight group has one
non-learned FP16 scale, and all headers, scales, directories, and alignment are
charged to traffic. The independently reloaded 30-layer artifact is 17,173,504
bytes, or **43.1353%** of dense ideal Q4.

The student is trained through that hard representation with an untouched
dense teacher. Checkpoints are device-neutral. CUDA accelerated the long
training rung, but the artifact and CPU inference contract do not depend on a
GPU.

## One-million-position result

The final rung used 8,192 fresh training sequences (1,014,225 input positions)
and evaluated one frozen, serialized/reloaded artifact on 16 sequences and 491
next-token positions.

| Metric | Required gate | Measured |
|---|---:|---:|
| Teacher-to-student KL | ≤0.05 | 2.28436 |
| Teacher top-1 agreement | ≥0.90 | 0.31976 |
| NLL delta | ≤+0.05 | +2.27704 |
| Final-hidden relative L2 | ≤0.10 | 0.60361 |
| Physical cold MLP traffic | ≤45% | 43.1353% |
| Serialized/reloaded | yes | yes |

The experiment had a second, cheaper progression rule before any 3M or 10M
spend: close at least 50% of every remaining causal gap from the preceding
head-coadaptation checkpoint.

| Metric | 50%-closure threshold | Measured | Gap closed | Result |
|---|---:|---:|---:|---|
| KL | ≤3.09977 | 2.28436 | 63.37% | pass |
| Top-1 | ≥0.47749 | 0.31976 | 31.33% | fail |
| NLL delta | ≤3.04128 | 2.27704 | 62.77% | pass |
| Hidden L2 | ≤0.50807 | 0.60361 | 38.29% | fail |

KL and NLL respond strongly to more data, but vocabulary decisions and hidden
geometry do not improve quickly enough. The exact configuration is therefore
stopped rather than promoted to 3M.

## What was tested before the long rung

- A deepest-layer-first transition removed the abrupt all-layer quantization
  cliff and reduced initial KL by 61.2%, but hidden error barely moved.
- Rebalancing local/all-layer hidden losses, adding a direct final-hidden
  loss, CKA geometry distillation, hard teacher-top-1 distillation, and
  co-adapting the already-resident embedding/output head were each screened on
  fresh records.
- CKA did not break the top-1 plateau, and head co-adaptation moved top-1 only
  from 5.09% to 5.50% in its bounded screen.
- A nearby 87.5%-width affine-Q2 idea failed its cheap layer-14 initialization
  screen (0.713 mean relative L2) and was not promoted to a production codec.
- The remaining 742,400 traffic bytes are too small to justify repeating the
  previously failed post-hoc correction-capsule program without a new
  hypothesis.

The machine-readable [`summary.json`](summary.json) records exact thresholds,
metrics, artifact hashes, report hashes, and scope. This is a negative
scientific result, not a claim that every low-bit model or pretraining strategy
must fail.
