# Evaluation

## Gate 1 definition

For each traced state and layer, all neuron activations are computed. Records are sorted by
the norm of their individual contribution, `abs(a_j) * ||v_j||₂`. Every cumulative prefix
is evaluated. The first prefix satisfying the residual-energy criterion is recorded for
90%, 95%, and 99% targets.

Reports include mean, median, and p95 required neuron fraction, relative L2 error, and
cosine similarity. Results are grouped globally, by layer, and by layer plus input type.
The reconstruction error between extracted weights and captured teacher MLP output is also
reported to catch boundary or extraction errors.

## Evidence labels

- `pipeline_validation`: deterministic random fixture; no model-quality conclusion.
- `measured_local_model`: a user-supplied trained checkpoint and held-out trace data.

Gate 1 is not complete until the same trained-model study includes fitted background
operators. Gates 2–4 have only random/synthetic pipeline reports. `engram evaluate-e2e` measures
student NLL/perplexity, teacher KL, top-1/top-5 agreement, category accuracy, repetition, and
fixed examples against a cached Hugging Face teacher. Model IDs are downloaded automatically,
while local model directories remain offline-capable. Trained SmolLM2 semantic-routing experiments
have run, but no trained end-to-end Gate 5 evaluation has, so no Gate 5 quality target is claimed.

The system-level Cognitive Executive has separate goal, confidence-calibration, action-utility,
attention, memory, monitoring, and safety gates defined in
[its design document](cognitive_executive.md). Compiler gates do not imply executive success.
