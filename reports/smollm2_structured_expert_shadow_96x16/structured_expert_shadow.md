# Structured block-expert shadow screen

Decision: **diagnose_before_joint_block_training**

The layout has 96 contiguous blocks of 16 records and
executes 32 blocks (512 records) per token.

| Metric | Mean |
|---|---:|
| Block-norm reference local relative L2 | 0.473910 |
| Greedy-residual reference local relative L2 | 0.437657 |
| Fitted router local relative L2 | 0.623527 |
| Fitted router block recall | 0.526042 |
| Projected inference traffic | 0.354203× dense |

This screen uses exact dense block contributions to construct its reference. It does not
pass the end-to-end intervention gate and does not authorize serialization.
