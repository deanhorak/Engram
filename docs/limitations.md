# Limitations

- The Cognitive Executive currently has deterministic policy, revisioned SQLite/JSONL/in-memory
  event stores, a worker capability registry, resource accounting, matched outcome observation,
  adapter protocols, and calibration metrics. It has no durable knowledge store, strategy
  registry, production model/tool adapters, domain content validators, or learned success/cost
  predictors. Its reference session permits only one in-flight attempt.
- Executive evidence confidence is a conservative policy score, not calibrated epistemic
  probability. Existing controller-residual and vocabulary-margin confidence fields are also
  proxies and must not be interpreted as knowing whether a claim is true.
- Outcome calibration covers dispatched actions only. It cannot estimate counterfactual quality
  for rejected alternatives without controlled exploration, and information-gain measurements
  are comparable only when they share a declared validator.
- External dispatch is at-least-once across crashes: selection is durable before invocation, but a
  process can fail after a worker side effect and before recording its outcome. Production workers
  must deduplicate the stable attempt ID or recover their prior result. SQLite/JSONL durability
  does not make external side effects exactly once.

- The checked-in measurements use random weights and do not establish trained-model MLP
  sparsity.
- The oracle computes every activation. The practical router now uses deterministic joint-key
  IVF and avoids scanning all record keys, but still scans every coarse centroid.
- The low-rank background is implemented, but the checked random-fixture Gate 2 experiment
  overfits badly (mean relative L2 rises from 0.693 without it to 7.09 with it).
- Real-model tracing loads the source model in CPU float32. Layer-at-a-time source execution
  and activation checkpointing remain future compiler work.
- Attention replacement primitives exist, but trained teacher head analysis/distillation has
  not run. The shared controller is initialized, not distilled from source residual trajectories.
- Python and native generation work without source transformer tensors, but generated fixture
  token IDs are not meaningful language.
- Compiled runtimes consume quantized-only semantic arrays and scan only IVF-posted key codes.
  The tiny fixture still misses the active-fraction and logical-traffic goals, and the claimed
  10x hardware DRAM-traffic reduction is unproven.
- Correction capsules and adaptive escalation policies are operational primitives but are not
  fitted from teacher divergence regions by the compiler.
- The development Xeon E5-2695 v2 lacks AVX2. AVX2 code must use runtime dispatch and be
  executed on a Haswell-or-newer host or suitable CI runner.
- Hardware performance counters are unavailable on the development host. DRAM and energy
  claims cannot be measured here.
- Trained SmolLM2 semantic-routing reports are checked in, but no trained end-to-end compiled
  Gate 5 run exists. Engram downloads a model when a Hub ID is supplied explicitly; this can
  require substantial disk space and gated models still require Hugging Face authentication and
  license acceptance.
- Synthetic Gate 3 mean relative L2 is 0.456 for the heuristic hybrid; retrieval/copying
  accuracy of 1.0 comes from a controlled synthetic case and must not be generalized.
- Synthetic Gate 4 adaptive control averaged 7.98/8 cycles and therefore did not demonstrate
  useful early exit.
- The native fixture benchmark is not a large-model benchmark. llama.cpp is unavailable locally,
  and no trained Hugging Face or llama.cpp comparison has run.
- Semantic and vocabulary IVF both reduce proxy-scored fixture rows from 64 to 32 per token,
  but coarse-centroid overhead makes the tiny logical-byte estimate worse, not better.
- The installed optional conversion stack currently has Transformers 5.5.4 and an older
  scikit-learn incompatible with NumPy 2.4.6; two Hugging Face integration tests skip
  until that external dependency set is made compatible. Those tests passed at an earlier
  compatible checkpoint in this workspace.
