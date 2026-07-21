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

- Random fixtures still validate most of the systems pipeline, but trained SmolLM2-135M
  interventions now measure MLP replacement quality on only 16 held-out sequences (491
  next-token positions). All 16 token sequences are unique, which clears the declared evidence
  floor for rejecting these artifacts; this is not a broad model, task, or language claim.
- Bias-enabled MLP projections (`mlp_bias=true`) are rejected during inspection because the
  current extraction and semantic-record formats do not represent those biases.
- The oracle computes every activation. The practical router now uses deterministic joint-key
  IVF and avoids scanning all record keys, but still scans every coarse centroid.
- Exact all-layer top-256 and top-512 magnitude-reference substitutions fail the declared quality
  gate. Top-768 is the first tested pass and retains half of all MLP records; intermediate counts
  between 640 and 768 were not tested. Magnitude ranking is not guaranteed to be the optimal
  K-record subset.
- At top-768 with 1,280 candidates, flat rank-16 and overlapping-posting routers both fail the
  downstream quality and 95% recall gates after training on all 1,112 calibration states per
  layer. Full-corpus recall is 0.889 for the flat router and 0.868 for the overlap router; the
  latter scans about 1,667 posting entries per layer on average to form 1,280 unique candidates.
  No learned router has been serialized into the package format.
- A cached rank-16 sweep peaks at λ=8,000. Candidate counts of 1,408 and 1,472 pass the 95% recall
  screen, but both fail causal quality. The 1,472 arm reads 95.8% of record keys and still has
  KL 0.085, top-1 agreement 0.866, NLL delta +0.055, and final-hidden relative L2 0.131. Further
  candidate expansion would approach a dense scan and is not considered a viable routing result.
- A predictor-free, DIP-inspired algorithm is the first realizable semantic selector to pass the
  all-layer gate. Engram's candidate-only completion and exact contribution reranking extend the
  published DIP method. After selecting 75%-input/896-candidate/K=768 on the development grid, a
  sequence-disjoint 16-sequence confirmation run has KL 0.029, top-1 agreement 0.910, NLL delta
  +0.033, final-hidden relative L2 0.090, and 0.990 candidate recall. This still covers only one
  135M-parameter model and a small generated corpus; another model and broader natural data are
  required before generalization.
- DIP currently exists only as a mathematical selection primitive and dense quality evaluator.
  Its 76.4%-of-dense float32 weight-read figure assumes a packed sparse kernel can read each
  requested scalar once. It excludes activations, index/sort traffic, cache-line amplification,
  and weight-repacking costs. There is no native kernel, measured latency, or measured DRAM result.
  The projected 1.31x reduction is also far short of Engram's long-term 10x target.
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
  fitted by the compiler. The research fitter now targets exact routed-read residuals and hard
  regions, but all checked global and targeted layouts worsen held-out local MLP error; no fitted
  capsule is serialized.
- Sparse-teacher fine-tuning is implemented only as a pilot trainer. It freezes all base weights
  and trains router factors plus sparse down adapters. The checked run has only 32 optimizer steps,
  no hyperparameter/seed replication, and fails every routed quality check. The hard
  `argsort`/gather route also prevents local, hidden, and logit losses from reaching the router;
  only its auxiliary membership loss trains router scores, and the adapter movement is negligible.
  Its safetensors file is an experiment artifact, not a supported compiled-package input.
- The development Xeon E5-2695 v2 lacks AVX2. AVX2 code must use runtime dispatch and be
  executed on a Haswell-or-newer host or suitable CI runner.
- Hardware performance counters are unavailable on the development host. DRAM and energy
  claims cannot be measured here.
- Trained SmolLM2 semantic-routing and causal MLP-intervention reports are checked in, but no
  trained end-to-end compiled Gate 5 run exists. Learned-router artifacts remain blocked; the
  predictor-free DIP arm is quality-eligible but has not been serialized or implemented in the
  compiled runtimes. Attention/controller distillation and trained-package compilation remain
  pending.
  Engram downloads a model when a Hub ID is supplied explicitly; this can require substantial
  disk space, and gated models still require Hugging Face authentication and license acceptance.
- Synthetic Gate 3 mean relative L2 is 0.456 for the heuristic hybrid; retrieval/copying
  accuracy of 1.0 comes from a controlled synthetic case and must not be generalized.
- Synthetic Gate 4 adaptive control averaged 7.98/8 cycles and therefore did not demonstrate
  useful early exit.
- The native fixture benchmark is not a large-model benchmark. llama.cpp is unavailable locally,
  and no trained Hugging Face or llama.cpp comparison has run.
- Semantic and vocabulary IVF both reduce proxy-scored fixture rows from 64 to 32 per token,
  but coarse-centroid overhead makes the tiny logical-byte estimate worse, not better.
- The base Anaconda environment still has an older scikit-learn incompatible with NumPy 2.4.6,
  so Hugging Face integration tests skip there. The repository's conversion environment uses
  PyTorch 2.7.1 and Transformers 5.14.1; the local-Llama intervention tests pass when
  `LD_LIBRARY_PATH` is unset so `/opt/libtorch` does not override the wheel libraries.
