# BitNet shared-controller distillation: first protected development run

## Decision

The CUDA-training-to-CPU-artifact path is operational. The checked rank-128
controller improves an untouched micro-validation trajectory and independently
reloads in the NumPy CPU runtime. This passes the controller
infrastructure/development gate, not the Milestone 4 semantic or generation
gate.

## Contract

- Frozen teacher: packaged `microsoft/bitnet-b1.58-2B-4T`
- Source weight SHA-256:
  `8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b`
- Teacher execution: native packaged BitNet, CPU only
- Optimizer execution: RTX 3050 CUDA
- Deployment execution: NumPy, CPU only; Torch not imported
- Training/validation: 8/8 token positions with different dataset hashes
- Controller inputs: token embedding, exact teacher MLP output, exact teacher
  attention output
- State contract: per-token RMS normalized at every depth stage
- Operator contract: divided by the incoming teacher residual RMS

## Controller

| Property | Value |
|---|---:|
| Hidden width | 2,560 |
| Stages | 30 |
| Shared rank | 128 |
| Stage-adapter rank | 4 |
| Parameters | 2,662,430 |
| Serialized FP32 bytes | 10,649,720 |
| Training steps | 200 |
| CUDA training time | 43.82 s |

The replaced dense FP64 GRU target would require approximately 629 MB for its
two large kernels alone. The factorized artifact is about 59 times smaller.

## Protected validation

| Metric | Before | After | Improvement |
|---|---:|---:|---:|
| Total rollout loss | 4.306441 | 1.881813 | 2.424628 |
| Hidden normalized MSE | 1.954873 | 1.064729 | 0.890144 |
| Terminal normalized MSE | 2.003824 | 0.522350 | 1.481474 |
| Cosine loss | 0.977434 | 0.532363 | 0.445070 |
| Delta normalized MSE | 1.000003 | 0.965990 | 0.034013 |

Training terminal normalized MSE reaches 0.005545, while protected validation
is 0.522350. The gap is large and expected for an eight-position fitting set;
it is direct evidence that corpus scale is now the next requirement.

## CPU artifact evidence

- Serialized Torch/NumPy maximum absolute error: `7.152557e-6`
- Mean absolute error: `6.981588e-7`
- Eight states, 30 NumPy CPU cycles: `0.147532 s`
- Throughput for that batched interface: `54.23 states/s`
- Output RMS range: `0.99999946` to `0.99999958`
- Torch imported by the measured inference process: no

## What this does not prove

The controller is still supplied exact teacher MLP and attention outputs.
Consequently, it has not yet demonstrated inference from compiled semantic
memory and episodic attention, vocabulary/logit agreement, autoregressive
generation, or acceptable language quality. The next experiment must first
scale disjoint traces, then train and evaluate the controller self-fed with
the compiled operator outputs that will exist at deployment.
