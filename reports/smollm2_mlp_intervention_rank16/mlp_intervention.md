# Trained-teacher MLP intervention

Status: **local_teacher_intervention_measurement**

Sequences: 16 (16 unique); next-token positions: 491; input-token positions: 507.

Exact-teacher NLL: 3.176500; perplexity: 23.962732.

| Arm | Scope | Layers | Candidates | Recall | MLP rel-L2 | Final hidden rel-L2 | KL | Top-1 agreement | NLL delta | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| identity_all_selected_layers | all | 30 | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| rank16_candidates_1024_top_768_all_selected_layers | all | 30 | 1024 | 0.722389 | 0.373788 | 0.495427 | 1.634455 | 0.399185 | 1.678786 | fail |
| rank16_candidates_1280_top_768_all_selected_layers | all | 30 | 1280 | 0.867489 | 0.269613 | 0.323149 | 0.649639 | 0.619145 | 0.669567 | fail |

> Forward hooks replace MLP outputs after the dense Hugging Face MLP executes; quality metrics are valid, but evaluator wall time is not an inference benchmark.

> The magnitude oracle uses all neuron activations to choose top-K and is not a realizable candidate-selection algorithm. Magnitude ranking is not the mathematically optimal K-subset when vector contributions can cancel, so this is a full-information reference rather than a theoretical quality ceiling.

Development decision: **stop_before_serialization**.
