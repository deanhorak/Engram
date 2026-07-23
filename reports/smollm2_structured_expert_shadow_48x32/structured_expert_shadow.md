# Structured block-expert shadow screen

Decision: **diagnose_before_joint_block_training**

The layout has 48 contiguous blocks of 32 records and
executes 16 blocks (512 records) per token.

| Metric | Mean |
|---|---:|
| Block-norm reference local relative L2 | 0.522172 |
| Greedy-residual reference local relative L2 | 0.496813 |
| Fitted router local relative L2 | 0.637928 |
| Fitted router block recall | 0.540788 |
| Projected inference traffic | 0.343768× dense |

This screen uses exact dense block contributions to construct its reference. It does not
pass the end-to-end intervention gate and does not authorize serialization.
