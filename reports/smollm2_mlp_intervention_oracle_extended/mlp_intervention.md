# Trained-teacher MLP intervention

Status: **local_teacher_intervention_measurement**

Sequences: 16 (16 unique); next-token positions: 491; input-token positions: 507.

Exact-teacher NLL: 3.176500; perplexity: 23.962732.

| Arm | Scope | Layers | Candidates | Recall | MLP rel-L2 | Final hidden rel-L2 | KL | Top-1 agreement | NLL delta | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| identity_all_selected_layers | all | 30 | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_640_all_selected_layers | all | 30 | - | - | 0.143566 | 0.130626 | 0.066099 | 0.857434 | 0.042641 | fail |
| oracle_top_768_all_selected_layers | all | 30 | - | - | 0.104149 | 0.092162 | 0.032106 | 0.922607 | 0.021622 | pass |
| oracle_top_1024_all_selected_layers | all | 30 | - | - | 0.047340 | 0.040970 | 0.006151 | 0.975560 | -0.001800 | pass |

> Forward hooks replace MLP outputs after the dense Hugging Face MLP executes; quality metrics are valid, but evaluator wall time is not an inference benchmark.

> The magnitude oracle uses all neuron activations to choose top-K and is not a realizable candidate-selection algorithm. Magnitude ranking is not the mathematically optimal K-subset when vector contributions can cancel, so this is a full-information reference rather than a theoretical quality ceiling.

Development decision: **router_experiments_justified**.
