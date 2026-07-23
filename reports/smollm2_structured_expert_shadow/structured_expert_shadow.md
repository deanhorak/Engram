# Structured block-expert shadow screen

Decision: **diagnose_before_joint_block_training**

The layout has 24 contiguous blocks of 64 records and
executes 8 blocks (512 records) per token.

| Metric | Mean |
|---|---:|
| Block-norm reference local relative L2 | 0.562435 |
| Greedy-residual reference local relative L2 | 0.546505 |
| Fitted router local relative L2 | 0.654594 |
| Fitted router block recall | 0.557910 |
| Projected inference traffic | 0.338551× dense |

This screen uses exact dense block contributions to construct its reference. It does not
pass the end-to-end intervention gate and does not authorize serialization.
