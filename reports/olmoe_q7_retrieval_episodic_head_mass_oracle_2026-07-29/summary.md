# OLMoE Q7 dynamic per-head episodic-mass oracle

This train-only, same-state capacity experiment held the authenticated K256
attention/episodic representation fixed and selected one of eight episodic
mass multipliers independently at every record/read-row/layer/head coordinate.
The multiplier codes were `0, 1/8, 1/4, 1/2, 1, 2, 4, 8`. Selection minimized
absolute error to the W128 teacher's probability mass on the eight scheduled
source positions. The selected pre-`W_o` counterfactual was then projected
through the authenticated BF16 output projection and compared with the exact
native W128-minus-K256 residual.

The experiment failed its prospectively frozen gate:

- global recovery: **-0.1089124543** (required at least 0.50);
- sequence recovery: negative on all 8 records (required at least 0.25 each);
- block-entry recovery at positions 96/104/112/120:
  **-0.0838671661 / -0.1344610650 / -0.0262677422 / -0.0686255750**
  (required at least 0.25 each);
- positive-recovery layers: **1/16** (required at least 12/16).

Mass matching itself worked: mean selected mass error fell from the gamma-one
baseline's 0.0445126662 to 0.0084754603, and it never regressed at any
coordinate. That improvement translated in the wrong direction after value
aggregation and `W_o`. The result therefore closes this exact fixed-K256,
independent-head, scheduled-source-mass matching grid. It does not close a
joint selector trained directly against the projected output residual.

Systems evidence passed:

- all 8 base executions exactly matched the historical outputs and counters;
- every new `base_projected` and `target_residual` tensor matched its
  authenticated historical shard byte for byte;
- every reset replay reproduced outputs, counters, and traces exactly;
- the metric/code replay was exact;
- every post-run package, source, checkpoint, DSO, corpus, and protocol check
  passed;
- confirmation data remained unopened.

The real-model gamma qualification is intentionally layer-zero-only. Layer
zero has an identical input state in beta-zero and biased execution and
matched the analytic mass/output/projected counterfactual to at worst about
`1.2e-7`. Applying the bias directly to all layers changes layer-zero output
and therefore changes the input to layers 1–15; those differences are causal
diagnostics, not valid same-state parity checks. The shared native attention
kernel is separately unit-tested across the entire gamma grid.

Authenticated roots:

- parity report: `569218dbb3ba8667c15ff932e0f9dffe5575774418080cc4e4239062fe2c6d01`
- protocol: `fe09689452e6ae4f1a1b15332c61c1cc990cfc29b6a8b0d5a1758d9490a93af5`
- result: `f7060e7373c5faf8f154891e93efad35659723d8e3f04d83638b62fa9cf72596`
- trace manifest: `93df0a554744b97e7436b9a8b4bb71473bc21fa9f6c90985431274859164e0b6`
- immutable trace DSO:
  `6c466bb75a508bd7f8b9173667e7bd9d8433d91c3be818db25f01a495be6d2da`

Milestone 2 remains passed for the Q7 semantic path. Milestone 3 bounded
attention remains blocked. The next defensible cached capacity boundary is a
joint output-targeted per-head gamma oracle that optimizes the exact projected
residual, includes a continuous bounded upper bound, and accounts for
cross-head coupling through `W_o`.
