# Trained-teacher MLP intervention

Status: **local_teacher_intervention_measurement**

Sequences: 16 (16 unique); next-token positions: 491; input-token positions: 507.

Exact-teacher NLL: 3.176500; perplexity: 23.962732.

| Arm | Scope | Layers | Candidates | Recall | MLP rel-L2 | Final hidden rel-L2 | KL | Top-1 agreement | NLL delta | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| identity_all_selected_layers | all | 30 | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_768_all_selected_layers | all | 30 | - | - | 0.104149 | 0.092162 | 0.032106 | 0.922607 | 0.021622 | pass |
| rank16_candidates_1408_top_768_all_selected_layers | all | 30 | 1408 | 0.953657 | 0.155503 | 0.162666 | 0.145702 | 0.828921 | 0.115845 | fail |
| rank16_candidates_1472_top_768_all_selected_layers | all | 30 | 1472 | 0.978280 | 0.129924 | 0.130723 | 0.085243 | 0.865580 | 0.055005 | fail |

> Forward hooks replace MLP outputs after the dense Hugging Face MLP executes; quality metrics are valid, but evaluator wall time is not an inference benchmark.

> The magnitude oracle uses all neuron activations to choose top-K and is not a realizable candidate-selection algorithm. Magnitude ranking is not the mathematically optimal K-subset when vector contributions can cancel, so this is a full-information reference rather than a theoretical quality ceiling.

Development decision: **stop_before_serialization**.
