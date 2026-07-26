# Dense-source exact activation-sparse screen

Date: **2026-07-24**

Decision: **local retrofit rejected; whole-model Q-Sparse co-adaptation is the
remaining distinct activation-sparse hypothesis.**

Two exact-selector representations were implemented and screened without
opening confirmation data.

| Representation | Exact selection | Ideal-Q4 traffic | Layer-14 initial L2 | Best/final L2 | Decision |
|---|---|---:|---:|---:|---|
| CATS-style gate threshold | Full gate, then exact nonzero up/down records | 43.48% final | 0.6151 | 0.4698 final | Reject local retrofit |
| Q-Sparse top-K activations | 282/576 gate/up inputs, 522/1,536 down inputs | 43.967% | 0.3426 | 0.3228/0.3233 | Reject local retrofit |

The traffic numbers exclude metadata, scales, alignment, and cache-line
amplification. Both mechanisms avoid approximate routing and therefore have no
candidate-recall requirement. The unchanged local progression threshold is
mean relative L2 at most 0.18; neither arm qualifies for an all-layer causal
run.

The threshold arm used 1,024 progressive updates followed by 4,096
fixed-budget updates on as many as 65,536 cached layer-14 teacher boundaries.
The Q-Sparse arm used the same schedule and data ceiling. Both were evaluated
on the same disjoint 446-boundary development set.

Published ProSparse and Q-Sparse results depend on whole-model continued
training at very large token scales. The local result rejects short
source-coordinate fitting, not whole-model coordinate co-adaptation. The next
eligible implementation is therefore a copied student trained end to end
through the exact Q-Sparse forward path against an untouched dense teacher,
with causal development checkpoints and the frozen confirmation corpus kept
sealed.

Scratch evidence:

- intrinsic sweep JSON SHA-256:
  `0a8b78ce153c43e593efe294b6b56d34d160e77c1ee0dfddee4656b2a1ca6d89`
- fixed-budget threshold-training JSON SHA-256:
  `2418232c5043cac3d0601130e9bb751f9135fcd58ec4f8609bc5875f9e2a9c63`
- fixed-budget Q-Sparse training JSON SHA-256:
  `51d1320788509edbbc21853bfcf52e08466cfea6eaae71fcea14992fdc6ae21f`

Primary references:

- [ProSparse](https://arxiv.org/abs/2402.13516)
- [CATS](https://arxiv.org/abs/2404.08763)
- [Q-Sparse](https://arxiv.org/abs/2407.10969)
- [MOHAWK transformer-to-SSM distillation](https://arxiv.org/abs/2408.10189)
