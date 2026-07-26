# BitNet controller scaling and self-fed continuation

## Decision

The factorized controller benefits materially from broader trajectory data,
but 128 training positions are still insufficient for compiled-operator
substitution. Retain the first 500-step rank-128 artifact. Reject the
subsequent fully self-fed continuation because it improves training while
regressing every protected validation metric.

## Capture evidence

| Property | Training | Validation |
|---|---:|---:|
| Sequences | 8 | 4 |
| Token positions | 128 | 64 |
| Shards | 4 | 2 |
| Tokens per sequence | 16 | 16 |
| Batch size | 2 | 2 |

The splits have different dataset hashes. All shards are complete and
checksum-valid. For the same sample, batched and single-sequence captures are
bit-identical across token IDs, embeddings, 31 residual states, 30 MLP
outputs, and 30 attention outputs.

## Retained rank-128 result

| Protected metric | Before | After |
|---|---:|---:|
| Total rollout loss | 4.292672 | 1.156465 |
| Hidden normalized MSE | 1.946727 | 0.666834 |
| Terminal normalized MSE | 1.998608 | 0.245010 |
| Cosine loss | 0.973363 | 0.333417 |
| Delta normalized MSE | 1.000004 | 0.845118 |

Training used 500 steps, batch size 16, rank 128, adapter rank 4, and learning
rate 3e-4. The optimizer ran on CPU because the host's loaded NVIDIA
580.159.03 kernel module did not match the installed 580.173.02 userspace
library at the time of this run.

After reboot, both driver components report 580.173.02 and the exact
configuration was reproduced on the RTX 3050:

| Metric | CPU optimizer | CUDA optimizer |
|---|---:|---:|
| Validation terminal normalized MSE | 0.245010 | 0.246530 |
| Validation cosine loss | 0.333417 | 0.335160 |
| Validation total loss | 1.156465 | 1.163494 |
| Training time | 131.84 s | 112.63 s |
| Serialized CPU parity max error | 7.45e-6 | 5.36e-6 |

The close results show that optimizer device is not the present quality
limiter. The marginally better CPU artifact remains retained.

## Rejected self-fed continuation

The retained artifact was reloaded from its NumPy serialization and trained
for another 500 steps at learning rate 1e-4 with zero teacher forcing.

| Metric | Initial | Continued | Direction |
|---|---:|---:|---|
| Training terminal normalized MSE | 0.060627 | 0.029324 | improved |
| Validation terminal normalized MSE | 0.245010 | 0.260050 | regressed |
| Validation cosine loss | 0.333417 | 0.342041 | regressed |
| Validation total loss | 1.156465 | 1.191359 | regressed |

The continuation fails the protected development gate. More epochs on the
same trace are not justified.

## CPU artifact

- Parameters: 2,662,430
- Serialized FP32 bytes: 10,649,720
- Torch/NumPy maximum absolute error: `7.450581e-6`
- PyTorch-free CPU benchmark: 64 states × 30 cycles in median `0.803207 s`
- Batched throughput: `79.68 states/s`
- Output RMS range: `0.99999952`–`0.99999964`

## Next gate

Do not introduce compiled semantic-memory or episodic-attention error while
the controller alone has terminal normalized MSE 0.245. The next justified
experiment is a substantially broader, sequence-diverse protected trajectory
corpus after the host CUDA driver stack is made consistent. Because the
project's existing hidden-state gate is relative L2 at most 0.15, the
controller-only prerequisite is terminal normalized MSE at most
`0.15² = 0.0225`; compiled operators will need additional headroom. The next
data-scale rung should therefore report both that fixed target and its learning
curve rather than redefining success after training. Additional optimization
on these 128 positions is stopped.
