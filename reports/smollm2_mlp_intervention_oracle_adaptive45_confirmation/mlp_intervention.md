# Trained-teacher MLP intervention

Status: **local_teacher_intervention_measurement**

Evaluation role: **confirmation**.

Sequences: 16 (16 unique); next-token positions: 1168; input-token positions: 1184.

Exact-teacher NLL: 2.658047; perplexity: 14.268391.

Configuration-selection separation: 4 selection sequences, 16 evaluation sequences, 0 exact overlaps.

| Arm | Scope | Layers | Input fraction | Candidates | Recall | Score mass | MLP rel-L2 | Final hidden rel-L2 | KL | Top-1 agreement | NLL delta | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| identity_all_selected_layers | all | 30 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_layer_adaptive_mean_691_all_selected_layers | all | 30 | - | - | - | - | 0.130111 | 0.112317 | 0.042574 | 0.897260 | 0.033276 | fail |

> Forward hooks replace MLP outputs after the dense Hugging Face MLP executes; quality metrics are valid, but evaluator wall time is not an inference benchmark.

> The magnitude oracle uses all neuron activations to choose top-K and is not a realizable candidate-selection algorithm. Magnitude ranking is not the mathematically optimal K-subset when vector contributions can cancel, so this is a full-information reference rather than a theoretical quality ceiling.

Development decision: **increase_active_budget_or_stop**.
