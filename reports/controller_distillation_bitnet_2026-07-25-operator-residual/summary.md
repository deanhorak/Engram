# BitNet controller: exact operator-residual transition

## Decision

The controller-only fixed substitution gate passes. The protected terminal
normalized MSE is **0.000020801** against a maximum of **0.0225**, giving
1,081.7 times margin. This opens the next experimental boundary: replace dense
attention with the native bounded operator and measure the complete rollout.
The semantic output is already produced by the packaged CPU MLP kernel. This
result does not itself qualify transformer-free generation.

## Why the learned transition was wrong

The packaged BitNet teacher layer has a known residual contract:

```text
next_state = RMSNormalize(
    current_state + attention_output + mlp_output
)
```

The trace stores all three terms in the same incoming-state RMS coordinate
system. The earlier controller nevertheless compressed the two operator
outputs through one learned 7,680-to-128 projection and tried to reconstruct
the complete transition. Its rank-4 stage-conditioned input adapter improved
protected terminal NMSE only from 0.159440 to 0.157431 (1.26%), falsifying
stage alignment as the main limitation.

Schema-v3 controllers now preserve the two known residual additions exactly.
The factorized recurrent network remains available only as a correction path:

```text
residual = state + semantic_output + episodic_output
residual += correction_scale[stage] * factorized_correction(...)
next = RMSNormalize(residual)
```

The exact artifact initializes every correction scale to zero. The NumPy CPU
runtime detects that condition and does not read or multiply the factorized
correction tensors.

## Frozen protected result

The protocol is unchanged from the prior scale rung: 1,024 training positions
and 256 protected validation positions from distinct dataset hashes, 30
stages, width 2,560, and exact packaged-teacher operator outputs.

| Metric | Protected result |
|---|---:|
| Terminal normalized MSE | 0.000020801 |
| Mean hidden normalized MSE | 0.000017685 |
| Mean stage cosine loss | 0.000008841 |
| Delta normalized MSE | 0.000274950 |
| Total rollout loss | 0.000108108 |
| Fixed gate | **pass** |

The maximum stage NMSE is 0.000022630. The remaining error is consistent with
the FP16 trace boundary and repeated FP32 normalization, rather than a learned
transition approximation.

## Artifact and CPU evidence

- Artifact format: schema v3 `operator_residual_with_factorized_correction`
- Serialized FP32 bytes: 10,649,960
- Torch/NumPy maximum absolute error: `5.722046e-6`
- PyTorch-free benchmark: 256 states x 30 stages in median 0.1847 seconds
- CPU throughput: 41,575.9 stage transitions/s
- Correction matrices read in the measured hot path: no

The v1 learned-transition and v2 input-adapter artifact formats remain
loadable and retain their original metadata and tensor layouts.

## Scope and next gate

This result uses dense-attention outputs but already uses compiled MLP outputs:
`NativeBitNetRuntime` replaces every MLP with the packaged direct CPU
phase-stream kernel before trace capture. It proves that the shared controller
does not need to relearn known residual algebra and that its CPU transition
mechanism has ample numerical headroom. It does **not** measure native bounded
attention through the controller. That next experiment is recorded in the
[compiled substitution report](../controller_compiled_substitution_2026-07-25/summary.md).
