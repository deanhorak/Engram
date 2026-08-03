# Rank-128 causal controller experiment

This is the first evidence-sized causal controller trial, and it fails the
fixed controller gate.

- Teacher: packaged BitNet, source model hash
  `8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b`.
- Training: 8 sequences, 128 token records, top-32 teacher logits, CPU capture
  243.67 s.
- Validation: 16 sequence-disjoint records, 256 token positions, top-32 logits,
  CPU capture 402.13 s.
- Fit: rank 128, adapter rank 4, 500 CPU steps, hidden-state plus causal
  top-k objective (causal weight 0.1).
- CPU reload parity: passed, maximum absolute error `4.2915344e-06`.
- Validation terminal normalized MSE: **0.2624663** versus threshold `0.0225`.
- Validation causal top-k KL: **8.7518297**.
- Validation hidden normalized MSE: **0.7221664**.

The causal loss falls substantially on the training split, but free-running
validation still drifts. Teacher forcing and the fixed operator streams do not
produce a deployable nonzero controller. This rejects the current rank-128
factorized architecture for promotion; the next experiment must change the
provider/model capacity or objective rather than add blind epochs.

Machine-readable report:
`rank128_8x16_500cpu.json` (SHA-256
`47a1c94ae4797c9e0653f1b88d129620db691f3ad3306ca1ed85f2c234267a32`).
