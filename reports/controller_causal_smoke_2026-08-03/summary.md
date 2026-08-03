# Real BitNet causal-controller smoke

This is an execution smoke for the opt-in causal trace/objective path, not a
promotion or confirmation result.

- Teacher: packaged `microsoft/bitnet-b1.58-2B-4T` source hash
  `8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b`.
- Training trace: one sequence and four positions, top-8 logits, CPU capture
  in 41.36 seconds.
- Validation trace: a different source JSONL and one sequence/four positions,
  top-8 logits, CPU capture in 41.78 seconds.
- Controller: rank 2, adapter rank 1, one CPU optimization step, causal weight
  0.1; frozen vocabulary head and final RMSNorm were used for the readout.
- CPU reload parity: passed, maximum absolute error `2.3841858e-06`.
- Validation causal top-k KL: `10.1189083`.
- Validation terminal normalized MSE: `1.9661006` versus the fixed threshold
  `0.0225`; the gate failed.

The result proves that real native CPU traces can carry causal targets and that
the controller objective runs without decoder layers. The tiny arm is far below
the evidence floor and does not authorize a learned correction or layer-free
generation.

Machine-readable report:
`training_report.json` (SHA-256
`c42501f4ae6896dba3704a66fedbf6b9dd88468b073592f3d360eba197b938b6`).
