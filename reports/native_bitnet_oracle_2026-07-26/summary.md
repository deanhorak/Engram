# Native BitNet semantic-record oracle

Date: **2026-07-26**

Decision: the trained BitNet teacher has a qualifying layer-adaptive semantic
subset, but Milestone 2 remains blocked at practical routing.

## Frozen result

The schedule was selected on two development sequences and then frozen. It
uses 15–35% of the 6,912 records per layer and 24.8375% on average. The frozen
confirmation uses eight unique sequences and 256 prediction positions.

| Metric | Threshold | Result |
|---|---:|---:|
| Teacher-to-student KL | at most 0.05 | 0.025428 |
| Teacher top-1 agreement | at least 0.90 | 0.945312 |
| NLL delta | at most +0.05 | +0.023855 |
| Final hidden relative L2 | at most 0.10 | 0.092049 |
| Mean active-record fraction | at most 25% | 24.8375% |
| Evidence | at least 8 sequences / 256 positions | 8 / 256 |

The machine-readable result is
[frozen_causal_adaptive_8x256.json](frozen_causal_adaptive_8x256.json).
The fixed 25% comparison is
[frozen_causal_25pct_8x256.json](frozen_causal_25pct_8x256.json); it missed only
the hidden-state threshold at 0.104475.

## What the oracle proves

The kernel preserves native BitNet activation quantization, ternary gate/up
accumulation, ReLU-squared activation, intermediate RMS normalization and
gain, second activation quantization, and additive down records. It ranks
records by the exact norm of each coefficient-times-down-column contribution.
The passing result therefore establishes that the selected semantic values can
carry the teacher's causal behavior at the intended active-record budget.

## What remains open

The oracle still executes the full gate/up coefficient path before selecting
records. It therefore does not provide practical candidate recall, complete
routed cold-traffic evidence, or a deployable latency result. Milestone 2
requires a compact router trained against these memberships, coefficient
reconstruction from selected gate/up keys only, a serialized/reloaded
router/index, and a repeat of the frozen causal, traffic, recall, and CPU
latency gates.
