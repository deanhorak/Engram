# Trained-teacher MLP intervention

Status: **local_teacher_intervention_measurement**

Sequences: 16 (16 unique); next-token positions: 491; input-token positions: 507.

Exact-teacher NLL: 3.176500; perplexity: 23.962732.

| Arm | Scope | Layers | Candidates | Recall | MLP rel-L2 | Final hidden rel-L2 | KL | Top-1 agreement | NLL delta | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| identity_all_selected_layers | all | 30 | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_128_all_selected_layers | all | 30 | - | - | 0.468940 | 0.558986 | 2.053423 | 0.350305 | 2.104710 | fail |
| oracle_top_256_all_selected_layers | all | 30 | - | - | 0.340719 | 0.356503 | 0.648368 | 0.604888 | 0.667717 | fail |
| oracle_top_512_all_selected_layers | all | 30 | - | - | 0.192689 | 0.179080 | 0.132203 | 0.808554 | 0.084959 | fail |

> Forward hooks replace MLP outputs after the dense Hugging Face MLP executes; quality metrics are valid, but evaluator wall time is not an inference benchmark.

> The magnitude oracle uses all neuron activations to choose top-K and is not a realizable candidate-selection algorithm. Magnitude ranking is not the mathematically optimal K-subset when vector contributions can cancel, so this is a full-information reference rather than a theoretical quality ceiling.

Development decision: **increase_active_budget_or_stop**.
