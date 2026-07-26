# BitNet controller: 1,024-position scale rung

## Decision

The larger corpus improves protected trajectory quality, but the controller
still fails the fixed substitution gate by 7.1 times. Do not collect another
blind scale rung and do not substitute compiled semantic or episodic outputs.
The error profile instead justifies stage-conditioned input adapters.

## Frozen protocol

| Property | Training | Validation |
|---|---:|---:|
| Sequences | 64 | 16 |
| Positions per sequence | 16 | 16 |
| Total positions | 1,024 | 256 |
| Capture batch | 4 | 4 |
| Trace shards | 16 | 4 |
| Capture seconds | 1,044.24 | 276.21 |

The dataset hashes differ and all trace shards pass checksum verification.
The controller is a fresh rank-128, adapter-rank-4 model trained for 1,000
steps on the RTX 3050 with batch size 32 and learning rate 3e-4.

## Protected result

| Metric | Initial | Trained |
|---|---:|---:|
| Total rollout loss | 4.275405 | 0.931534 |
| Hidden normalized MSE | 1.941286 | 0.545606 |
| Terminal normalized MSE | 1.987054 | 0.159440 |
| Mean stage cosine loss | 0.970642 | 0.272803 |
| Delta normalized MSE | 1.000004 | 0.796833 |

The prior 128-position result had terminal normalized MSE 0.245010. Eight
times more training data reduces it to 0.159440. The fixed substitution
threshold is 0.0225, so this rung fails by a factor of 7.086.

## Stagewise diagnosis

| Stage | Normalized MSE | Relative L2 | Cosine loss |
|---:|---:|---:|---:|
| 1 | 1.077929 | 1.038234 | 0.538964 |
| 5 | 0.848588 | 0.921188 | 0.424295 |
| 10 | 0.679043 | 0.824041 | 0.339521 |
| 15 | 0.530822 | 0.728575 | 0.265411 |
| 20 | 0.419096 | 0.647376 | 0.209548 |
| 25 | 0.293215 | 0.541493 | 0.146607 |
| 30 | 0.159440 | 0.399300 | 0.079720 |

Error is largest immediately and then declines. Exact later teacher MLP and
attention outputs repeatedly steer the controller back toward the teacher
trajectory. This is evidence against accumulated recurrent drift and against
proceeding to approximate compiled inputs.

The current controller's stage adapters transform recurrent state only. Every
stage's token embedding, semantic output, and episodic output pass through the
same rank-128 input projection. Layer-specific coordinate systems therefore
have no stage-conditioned alignment before the shared bottleneck.

## Scaling check

Using only the 128→1,024 position improvement gives a crude empirical exponent
of approximately 0.207. Extrapolating that two-point slope to MSE 0.0225 would
suggest about 13.4 million training positions. This is an inference, not a
forecast, and is too expensive and weakly founded to justify blind capture.

## Artifact evidence

- CUDA training time: 235.09 seconds
- Serialized FP32 bytes: 10,649,720
- Torch/NumPy maximum absolute error: `5.900860e-6`
- PyTorch-free CPU batch: 256 states × 30 cycles in median 2.5281 seconds
- Batched CPU throughput: 101.26 states/s

## Next experiment

Add a small stage-conditioned low-rank adapter from the 7,680-wide controller
input into the shared 128-wide bottleneck. At input-adapter rank four this
adds approximately 0.94 million FP32 parameters, or about 3.8 MB, while
preserving a shared recurrent core and CPU-only inference. Train it on the
same frozen 1,024/256 traces first; this isolates architectural effect from
additional data.
