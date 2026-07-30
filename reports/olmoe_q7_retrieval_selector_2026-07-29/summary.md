# OLMoE Q7 retrieval-targeted selector development result

## Decision

The prospectively frozen static exact-51-head selector **failed** its
development semantic gate. Training selected `M2` and passed every declared
training progression rule, the dense teacher demonstrated retrieval, the
full-W128 packaged-Q7 control passed, all resource and replay contracts passed,
and every post-run artifact binding reauthenticated. The static `M2` candidate
nevertheless exceeded the KL, target-NLL-delta, and hidden-state thresholds
overall and at every source depth.

Confirmation was not opened or hashed, and confirmation was not authorized.
This closes this static-mask configuration at the declared 44.9754% logical
attention-read budget. The next allocation class is a causally valid,
prefix- or phase-conditioned policy.

## Authenticated artifacts

| Artifact | SHA-256 |
|---|---|
| [Frozen protocol](protocol.json) | `f6961ec6bffadb306a75ae8aaa8c400d1282cbb096876532d75e22a295660580` |
| [Development result](development_result.json) | `66c9c03d04e191865c4acc783c9fa73679a40f602777fb7a8aa56ccb9b61e4a6` |
| [Training checkpoint](development_result.training_checkpoint.json) | `9edf4fe7e7a1340c6f34bfe4d04544525947935116d662e94d98ccea4c282aa4` |

The checkpoint was written, reread, SHA-authenticated, and structurally
reconstructed before development began. It contains the complete `M0`, `M1`,
and `M2` training evidence and permits an explicit SHA-bound resume without
repeating the 16 surrogate backwards.

## Training

All eight records improved under the selected `M2` mask; none regressed.

| Mask | Worst answer CE | Mean answer CE |
|---|---:|---:|
| `M0` | 7.976308 | 7.647114 |
| `M1` | 4.080642 | 2.921855 |
| `M2` | 1.227907 | 1.005444 |

The 12-worker proxy completed 256 serial forward calls, 256 parallel backward
calls, and 14,603 expert-backward tasks. All proxy lifecycle checks passed.
Training took 11,747.166 seconds.

## Teacher and control evidence

The dense teacher passed the retrieval-evidence check overall and at every
source depth. Its mean ground-truth-versus-counterfactual log-probability
advantage was 12.457870 across 256 answer positions.

The full-W128 packaged-Q7 control passed every quality threshold:

| KL | Top-1 | Target NLL delta | Hidden relative L2 |
|---:|---:|---:|---:|
| 0.002000 | 0.984375 | 0.004333 | 0.048957 |

This establishes that the development failure is caused by the bounded
attention allocation rather than by Q7 execution or an incapable teacher.

## Exact-51 candidate

The candidate used exactly 51 of 256 layer/head pairs. It scheduled
973,384,704 logical attention-read bytes per sequence, 44.9753872184% of the
full-context reference, with 12,284,864 bytes of attention state. Every
resource check passed; a 52-head static mask remained inadmissible at
45.2437999637%.

Declared thresholds were KL at most 0.05, top-1 agreement at least 0.90,
target-NLL delta at most 0.05, and hidden relative L2 at most 0.10.

| Scope | KL | Top-1 | Target NLL delta | Hidden relative L2 |
|---|---:|---:|---:|---:|
| Overall | 0.186610 | 0.929688 | 0.283658 | 0.335103 |
| Earliest | 0.165740 | 0.968750 | 0.297904 | 0.305395 |
| Early | 0.155776 | 0.953125 | 0.157642 | 0.319725 |
| Middle | 0.194748 | 0.890625 | 0.358004 | 0.355587 |
| Late | 0.230177 | 0.906250 | 0.321084 | 0.359705 |

Only overall top-1 agreement passed. All four measures failed at the middle
depth; the other depths failed KL, target NLL, and hidden-state error.

A post-result development diagnosis shows that the first token of each
eight-token passkey is especially sensitive: its mean KL was 0.605066,
target-NLL delta 1.450437, and top-1 agreement 0.59375. Later within-passkey
tokens usually retained top-1 agreement, but hidden relative L2 remained about
0.31–0.35 throughout. That combination argues against merely adding a
first-token exception; the next experiment must test whether conditioned head
allocation can preserve the full hidden trajectory at the same total budget.

Both full-control and candidate reset/replay checks passed for hidden hashes,
logit hashes, answer cross-entropy, and all non-timing native counters.
Total wall time was 14,025.138 seconds.
