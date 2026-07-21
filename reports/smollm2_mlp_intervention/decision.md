# SmolLM2 trained-teacher MLP intervention decision

Status: **stop before router serialization**

This study asks whether sparse semantic reads preserve the behavior of
`HuggingFaceTB/SmolLM2-135M` when substituted into the original transformer. It uses 16 held-out
sequences, 491 next-token positions, all 30 layers, and both 128-state pilot fits and full-corpus
fits using all 1,112 available calibration states per layer. The source hash is
`907880e8955e6ef4072314e9032bc938d36ebcd49554a499b334bb441ba4deb7`, and validation dataset hash
`48c3e84202a13a4c2eac8f5bcda0d52a0b70ce5b7dc439bbb34338b928a45a4c`.
Exact token-sequence hashing found 32 unique calibration sequences, 16 unique evaluation
sequences, and zero overlap. The gate's minimum evidence floor is 8 unique sequences and 256
next-token positions; this study clears that floor but remains a single-model pilot.
Logit and NLL means use the 491 next-token positions. Final-hidden means use all 507 input-token
positions, and local MLP/recall means use 15,210 input-token/layer states.

The identity arm matches the teacher exactly. The full-information magnitude reference establishes
a reproducible active-set frontier before candidate routing. It is not a theoretical ceiling:
because vector contributions can cancel, another K-record subset can in principle do better.

| Active K | Active fraction | MLP rel-L2 | Final hidden rel-L2 | KL | Top-1 agreement | NLL delta | Gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 128 | 8.3% | 0.469 | 0.559 | 2.053 | 0.350 | +2.105 | fail |
| 256 | 16.7% | 0.341 | 0.357 | 0.648 | 0.605 | +0.668 | fail |
| 512 | 33.3% | 0.193 | 0.179 | 0.132 | 0.809 | +0.085 | fail |
| 640 | 41.7% | 0.144 | 0.131 | 0.066 | 0.857 | +0.043 | fail |
| 768 | 50.0% | 0.104 | 0.092 | 0.032 | 0.923 | +0.022 | pass |
| 1,024 | 66.7% | 0.047 | 0.041 | 0.006 | 0.976 | -0.002 | pass |

The declared all-layer progression thresholds are KL at most 0.05, top-1 agreement at least 0.90,
NLL delta at most +0.05, final normalized-hidden-state relative L2 at most 0.10, and candidate recall
at least 0.95 for routed arms. K=768 is the first tested magnitude-reference pass, so practical
routing was evaluated at that active count.

| Router | Candidates | Recall | MLP rel-L2 | Final hidden rel-L2 | KL | Top-1 agreement | NLL delta | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Flat rank 16 | 1,024 | 0.722 | 0.374 | 0.495 | 1.634 | 0.399 | +1.679 | fail |
| Flat rank 16 | 1,280 | 0.867 | 0.270 | 0.323 | 0.650 | 0.619 | +0.670 | fail |
| Overlap rank 16, 192×32 | 1,280 | 0.858 | 0.289 | 0.359 | 0.794 | 0.619 | +0.825 | fail |
| Flat rank 16, full corpus | 1,024 | 0.751 | 0.312 | 0.575 | 1.969 | 0.371 | +1.978 | fail |
| Flat rank 16, full corpus | 1,280 | 0.889 | 0.218 | 0.361 | 0.789 | 0.615 | +0.764 | fail |
| Overlap rank 16, 192×32, full corpus | 1,280 | 0.868 | 0.263 | 0.431 | 1.149 | 0.521 | +1.095 | fail |
| Flat rank 16, λ=8,000 | 1,408 | 0.954 | 0.156 | 0.163 | 0.146 | 0.829 | +0.116 | fail |
| Flat rank 16, λ=8,000 | 1,472 | 0.978 | 0.130 | 0.131 | 0.085 | 0.866 | +0.055 | fail |

The overlap arm uses four posting slots per record and selects about 61 groups per layer. Because
of duplicates it scans about 1,954 posting entries to produce 1,280 unique candidates—more posting
IDs than the layer's 1,536 records. In the full-corpus refit it scans about 1,667 entries and
selects about 52 groups. Its learned recall and downstream quality remain worse than the flat
rank-16 baseline.

Projected logical float32 read-once key-plus-value traffic for K=768/C=1,280 is about 230.0 MB per token across 30
layers (176.9 MB candidate keys plus 53.1 MB selected values), excluding caches and other model
components. A dense three-matrix MLP read is about 318.5 MB, so this operating point offers only
about 1.38× reduction in this accounting—not the long-term 10× target—even before its quality
failure.

The trace-only regularization screen peaks at λ=8,000. Expanding its candidate budget clears the
recall gate at C=1,408 and C=1,472, but the causal rows above still fail KL, top-1, NLL, and
hidden-state checks. C=1,472 examines 95.8% of every layer's record keys. Its candidate-key plus
selected-value traffic is about 256.6 MB/token across 30 layers, only 1.24× below the dense MLP's
318.5 MB before router overhead.

## Decision

No tested routed arm is eligible for serialization. Attention/episodic distillation, controller/output
distillation, trained package compilation, native benchmarking, and cross-model replication remain
gated off in that order. Full-corpus fitting, corpus-scaled regularization, and candidate expansion
are complete. Near-dense expansion clears recall but not causal quality and defeats the intended
traffic reduction, so the flat rank-16 configuration is abandoned. The representation or
training process must now change. Global and targeted low-rank correction capsules were fitted
next, but every layout worsened held-out local MLP error and was rejected before causal
integration. Sparse-teacher fine-tuning was therefore selected as the next major semantic
experiment. Any new artifact must rerun this intervention gate before compilation.

That initial sparse-teacher pilot is now complete. It trains router factors and rank-8 sparse down
adapters for 32 calibration steps while freezing the teacher and all base student tensors. On the
same 16-sequence held-out set it reaches recall 0.900, KL 0.448, top-1 agreement 0.721, NLL delta
+0.343, and final-hidden relative L2 0.250. It fails every routed quality check and remains outside
the package format. Longer training or wider adapters require a declared follow-up experiment;
this result is not evidence that sparse fine-tuning succeeds.

Machine-readable results and full per-layer statistics:

- [provenance-checked composite decision](../smollm2_mlp_intervention_composite/mlp_intervention.json)
- [initial oracle frontier](../smollm2_mlp_intervention_oracle/mlp_intervention.json)
- [extended oracle frontier](../smollm2_mlp_intervention_oracle_extended/mlp_intervention.json)
- [flat rank-16 intervention](../smollm2_mlp_intervention_rank16/mlp_intervention.json)
- [overlapping-posting intervention](../smollm2_mlp_intervention_overlap/mlp_intervention.json)
- [flat rank-16 full-corpus intervention](../smollm2_mlp_intervention_rank16_full/mlp_intervention.json)
- [overlapping-posting full-corpus intervention](../smollm2_mlp_intervention_overlap_full/mlp_intervention.json)
- [cached regularization sweep](../smollm2_rank_router_regularization_sweep/rank_router_regularization_sweep.json)
- [cached candidate frontier](../smollm2_rank_router_candidate_frontier/rank_router_regularization_sweep.json)
- [λ=8,000 near-dense causal intervention](../smollm2_mlp_intervention_rank16_lambda8000_frontier/mlp_intervention.json)
- [global correction-capsule sweep](../smollm2_correction_capsule_sweep/correction_capsule_sweep.json)
- [targeted correction-capsule sweep](../smollm2_correction_capsule_targeted_sweep/correction_capsule_sweep.json)
- [tight targeted correction-capsule sweep](../smollm2_correction_capsule_targeted_tight_sweep/correction_capsule_sweep.json)
- [sparse-teacher fine-tuning pilot](../smollm2_sparse_teacher_epoch1/sparse_teacher_training.json)

The hook evaluator executes the dense teacher MLP before replacing its output. Its quality metrics
are causal for the substitution, but its wall time is not an Engram runtime benchmark.
The composite decision validates matching model, evaluation, layer, baseline, and calibration
provenance across the separately executed arms before issuing `stop_before_serialization`.
