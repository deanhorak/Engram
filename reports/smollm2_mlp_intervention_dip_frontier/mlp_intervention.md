# Trained-teacher MLP intervention

Status: **local_teacher_intervention_measurement**

Evaluation role: **development**.

Sequences: 16 (16 unique); next-token positions: 491; input-token positions: 507.

Exact-teacher NLL: 3.176500; perplexity: 23.962732.

| Arm | Scope | Layers | Input fraction | Candidates | Recall | Score mass | MLP rel-L2 | Final hidden rel-L2 | KL | Top-1 agreement | NLL delta | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| identity_all_selected_layers | all | 30 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_768_all_selected_layers | all | 30 | - | - | - | - | 0.104149 | 0.092162 | 0.032106 | 0.922607 | 0.021622 | pass |
| dip_input_0p625_candidates_768_top_768_all_selected_layers | all | 30 | 0.625 | 768 | 0.906436 | 0.960961 | 0.125974 | 0.114731 | 0.049040 | 0.879837 | 0.039928 | fail |
| dip_input_0p75_candidates_768_top_768_all_selected_layers | all | 30 | 0.750 | 768 | 0.949593 | 0.981210 | 0.110814 | 0.100662 | 0.038376 | 0.898167 | 0.021954 | fail |
| dip_input_0p625_candidates_896_top_768_all_selected_layers | all | 30 | 0.625 | 896 | 0.959938 | 0.983225 | 0.111586 | 0.100646 | 0.037807 | 0.896130 | 0.009127 | fail |
| dip_input_0p75_candidates_896_top_768_all_selected_layers | all | 30 | 0.750 | 896 | 0.989930 | 0.996226 | 0.105121 | 0.093834 | 0.033855 | 0.912424 | 0.026164 | pass |
| dip_input_0p625_candidates_1024_top_768_all_selected_layers | all | 30 | 0.625 | 1024 | 0.982119 | 0.992424 | 0.107594 | 0.095813 | 0.035135 | 0.908350 | 0.029525 | pass |
| dip_input_0p75_candidates_1024_top_768_all_selected_layers | all | 30 | 0.750 | 1024 | 0.997067 | 0.998890 | 0.104509 | 0.092079 | 0.031643 | 0.924644 | 0.030814 | pass |

> Forward hooks replace MLP outputs after the dense Hugging Face MLP executes; quality metrics are valid, but evaluator wall time is not an inference benchmark.

> The magnitude oracle uses all neuron activations to choose top-K and is not a realizable candidate-selection algorithm. Magnitude ranking is not the mathematically optimal K-subset when vector contributions can cancel, so this is a full-information reference rather than a theoretical quality ceiling.

Development decision: **eligible_for_selector_serialization**.
