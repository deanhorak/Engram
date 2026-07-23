# Trained-teacher MLP intervention

Status: **local_teacher_intervention_measurement**

Evaluation role: **development**.

Sequences: 4 (4 unique); next-token positions: 71; input-token positions: 75.

Exact-teacher NLL: 2.387150; perplexity: 10.882434.

| Arm | Scope | Layers | Input fraction | Candidates | Recall | Score mass | MLP rel-L2 | Final hidden rel-L2 | KL | Top-1 agreement | NLL delta | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| identity_all_selected_layers | all | 30 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_896_layer_0 | individual | 1 | - | - | - | - | 0.019976 | 0.012373 | 0.000682 | 0.985915 | 0.006018 | pass |
| oracle_top_896_layer_1 | individual | 1 | - | - | - | - | 0.059220 | 0.011840 | 0.000720 | 1.000000 | -0.001031 | pass |
| oracle_top_896_layer_2 | individual | 1 | - | - | - | - | 0.070197 | 0.011487 | 0.000585 | 1.000000 | -0.001068 | pass |
| oracle_top_896_layer_3 | individual | 1 | - | - | - | - | 0.089252 | 0.010112 | 0.000380 | 1.000000 | 0.001164 | pass |
| oracle_top_896_layer_4 | individual | 1 | - | - | - | - | 0.095157 | 0.009897 | 0.000285 | 1.000000 | 0.002760 | pass |
| oracle_top_896_layer_5 | individual | 1 | - | - | - | - | 0.102801 | 0.010456 | 0.000453 | 0.985915 | -0.000213 | pass |
| oracle_top_896_layer_6 | individual | 1 | - | - | - | - | 0.090496 | 0.010636 | 0.000391 | 0.985915 | 0.000074 | pass |
| oracle_top_896_layer_7 | individual | 1 | - | - | - | - | 0.086875 | 0.011579 | 0.000434 | 1.000000 | 0.002760 | pass |
| oracle_top_896_layer_8 | individual | 1 | - | - | - | - | 0.069001 | 0.011125 | 0.000360 | 1.000000 | 0.003495 | pass |
| oracle_top_896_layer_9 | individual | 1 | - | - | - | - | 0.078631 | 0.012203 | 0.000441 | 1.000000 | 0.001007 | pass |
| oracle_top_896_layer_10 | individual | 1 | - | - | - | - | 0.077216 | 0.012881 | 0.000659 | 1.000000 | 0.002589 | pass |
| oracle_top_896_layer_11 | individual | 1 | - | - | - | - | 0.085447 | 0.012818 | 0.000597 | 1.000000 | 0.007162 | pass |
| oracle_top_896_layer_12 | individual | 1 | - | - | - | - | 0.088682 | 0.011517 | 0.000509 | 1.000000 | 0.004255 | pass |
| oracle_top_896_layer_13 | individual | 1 | - | - | - | - | 0.084059 | 0.010899 | 0.000383 | 1.000000 | 0.003141 | pass |
| oracle_top_896_layer_14 | individual | 1 | - | - | - | - | 0.072866 | 0.009888 | 0.000284 | 1.000000 | 0.002242 | pass |
| oracle_top_896_layer_15 | individual | 1 | - | - | - | - | 0.075467 | 0.010550 | 0.000389 | 1.000000 | 0.004990 | pass |
| oracle_top_896_layer_16 | individual | 1 | - | - | - | - | 0.080827 | 0.009985 | 0.000377 | 1.000000 | -0.000869 | pass |
| oracle_top_896_layer_17 | individual | 1 | - | - | - | - | 0.077839 | 0.010537 | 0.000407 | 1.000000 | 0.003018 | pass |
| oracle_top_896_layer_18 | individual | 1 | - | - | - | - | 0.065264 | 0.010027 | 0.000269 | 1.000000 | -0.001352 | pass |
| oracle_top_896_layer_19 | individual | 1 | - | - | - | - | 0.076372 | 0.011340 | 0.000314 | 1.000000 | 0.002053 | pass |
| oracle_top_896_layer_20 | individual | 1 | - | - | - | - | 0.062616 | 0.012924 | 0.000424 | 1.000000 | 0.005445 | pass |
| oracle_top_896_layer_21 | individual | 1 | - | - | - | - | 0.069357 | 0.012246 | 0.000391 | 1.000000 | -0.001454 | pass |
| oracle_top_896_layer_22 | individual | 1 | - | - | - | - | 0.062370 | 0.013442 | 0.000413 | 1.000000 | 0.000235 | pass |
| oracle_top_896_layer_23 | individual | 1 | - | - | - | - | 0.056900 | 0.012427 | 0.000390 | 1.000000 | -0.005818 | pass |
| oracle_top_896_layer_24 | individual | 1 | - | - | - | - | 0.053383 | 0.011342 | 0.000277 | 1.000000 | 0.001907 | pass |
| oracle_top_896_layer_25 | individual | 1 | - | - | - | - | 0.060549 | 0.010251 | 0.000251 | 0.985915 | 0.000389 | pass |
| oracle_top_896_layer_26 | individual | 1 | - | - | - | - | 0.057595 | 0.010557 | 0.000259 | 1.000000 | -0.000643 | pass |
| oracle_top_896_layer_27 | individual | 1 | - | - | - | - | 0.050269 | 0.010171 | 0.000284 | 1.000000 | 0.004300 | pass |
| oracle_top_896_layer_28 | individual | 1 | - | - | - | - | 0.037619 | 0.009474 | 0.000262 | 1.000000 | -0.003737 | pass |
| oracle_top_896_layer_29 | individual | 1 | - | - | - | - | 0.015829 | 0.012702 | 0.001507 | 1.000000 | -0.007204 | pass |
| oracle_top_1024_layer_0 | individual | 1 | - | - | - | - | 0.012648 | 0.007789 | 0.000399 | 1.000000 | 0.000789 | pass |
| oracle_top_1024_layer_1 | individual | 1 | - | - | - | - | 0.037936 | 0.007880 | 0.000269 | 1.000000 | -0.000435 | pass |
| oracle_top_1024_layer_2 | individual | 1 | - | - | - | - | 0.045444 | 0.006876 | 0.000163 | 1.000000 | -0.003224 | pass |
| oracle_top_1024_layer_3 | individual | 1 | - | - | - | - | 0.057788 | 0.006994 | 0.000135 | 1.000000 | 0.001047 | pass |
| oracle_top_1024_layer_4 | individual | 1 | - | - | - | - | 0.062447 | 0.006594 | 0.000141 | 1.000000 | -0.000339 | pass |
| oracle_top_1024_layer_5 | individual | 1 | - | - | - | - | 0.068115 | 0.007048 | 0.000163 | 1.000000 | 0.002016 | pass |
| oracle_top_1024_layer_6 | individual | 1 | - | - | - | - | 0.059087 | 0.007000 | 0.000148 | 0.985915 | 0.001354 | pass |
| oracle_top_1024_layer_7 | individual | 1 | - | - | - | - | 0.056626 | 0.007683 | 0.000174 | 1.000000 | 0.003645 | pass |
| oracle_top_1024_layer_8 | individual | 1 | - | - | - | - | 0.045172 | 0.007331 | 0.000164 | 1.000000 | -0.004193 | pass |
| oracle_top_1024_layer_9 | individual | 1 | - | - | - | - | 0.051260 | 0.007485 | 0.000210 | 1.000000 | 0.000284 | pass |
| oracle_top_1024_layer_10 | individual | 1 | - | - | - | - | 0.050235 | 0.008180 | 0.000276 | 1.000000 | 0.000883 | pass |
| oracle_top_1024_layer_11 | individual | 1 | - | - | - | - | 0.055541 | 0.008324 | 0.000274 | 1.000000 | 0.002835 | pass |
| oracle_top_1024_layer_12 | individual | 1 | - | - | - | - | 0.057605 | 0.007496 | 0.000176 | 1.000000 | 0.001740 | pass |
| oracle_top_1024_layer_13 | individual | 1 | - | - | - | - | 0.054684 | 0.007597 | 0.000223 | 1.000000 | 0.001410 | pass |
| oracle_top_1024_layer_14 | individual | 1 | - | - | - | - | 0.047681 | 0.006375 | 0.000129 | 1.000000 | 0.003332 | pass |
| oracle_top_1024_layer_15 | individual | 1 | - | - | - | - | 0.049183 | 0.006601 | 0.000115 | 1.000000 | 0.003250 | pass |
| oracle_top_1024_layer_16 | individual | 1 | - | - | - | - | 0.052555 | 0.006661 | 0.000134 | 1.000000 | 0.001889 | pass |
| oracle_top_1024_layer_17 | individual | 1 | - | - | - | - | 0.050020 | 0.006594 | 0.000143 | 1.000000 | 0.003391 | pass |
| oracle_top_1024_layer_18 | individual | 1 | - | - | - | - | 0.042286 | 0.006478 | 0.000103 | 1.000000 | 0.000202 | pass |
| oracle_top_1024_layer_19 | individual | 1 | - | - | - | - | 0.049547 | 0.007310 | 0.000124 | 1.000000 | 0.001349 | pass |
| oracle_top_1024_layer_20 | individual | 1 | - | - | - | - | 0.040862 | 0.008578 | 0.000192 | 1.000000 | 0.002351 | pass |
| oracle_top_1024_layer_21 | individual | 1 | - | - | - | - | 0.045157 | 0.007880 | 0.000124 | 1.000000 | -0.000271 | pass |
| oracle_top_1024_layer_22 | individual | 1 | - | - | - | - | 0.040602 | 0.008633 | 0.000158 | 1.000000 | 0.002682 | pass |
| oracle_top_1024_layer_23 | individual | 1 | - | - | - | - | 0.037290 | 0.008262 | 0.000173 | 1.000000 | -0.002421 | pass |
| oracle_top_1024_layer_24 | individual | 1 | - | - | - | - | 0.034701 | 0.007314 | 0.000148 | 1.000000 | -0.000029 | pass |
| oracle_top_1024_layer_25 | individual | 1 | - | - | - | - | 0.039390 | 0.006681 | 0.000117 | 1.000000 | -0.000260 | pass |
| oracle_top_1024_layer_26 | individual | 1 | - | - | - | - | 0.037360 | 0.006789 | 0.000108 | 1.000000 | 0.000468 | pass |
| oracle_top_1024_layer_27 | individual | 1 | - | - | - | - | 0.032781 | 0.006587 | 0.000131 | 1.000000 | 0.002051 | pass |
| oracle_top_1024_layer_28 | individual | 1 | - | - | - | - | 0.024448 | 0.006159 | 0.000115 | 1.000000 | 0.000374 | pass |
| oracle_top_1024_layer_29 | individual | 1 | - | - | - | - | 0.010306 | 0.008323 | 0.000651 | 1.000000 | 0.000509 | pass |
| oracle_top_1280_layer_0 | individual | 1 | - | - | - | - | 0.003393 | 0.002000 | 0.000019 | 1.000000 | -0.000090 | pass |
| oracle_top_1280_layer_1 | individual | 1 | - | - | - | - | 0.010681 | 0.002144 | 0.000018 | 1.000000 | 0.000306 | pass |
| oracle_top_1280_layer_2 | individual | 1 | - | - | - | - | 0.012845 | 0.001947 | 0.000018 | 1.000000 | 0.000876 | pass |
| oracle_top_1280_layer_3 | individual | 1 | - | - | - | - | 0.016164 | 0.001628 | 0.000009 | 1.000000 | -0.000550 | pass |
| oracle_top_1280_layer_4 | individual | 1 | - | - | - | - | 0.018185 | 0.001819 | 0.000012 | 1.000000 | -0.000672 | pass |
| oracle_top_1280_layer_5 | individual | 1 | - | - | - | - | 0.019873 | 0.002024 | 0.000015 | 1.000000 | -0.000478 | pass |
| oracle_top_1280_layer_6 | individual | 1 | - | - | - | - | 0.017051 | 0.002083 | 0.000015 | 1.000000 | 0.000221 | pass |
| oracle_top_1280_layer_7 | individual | 1 | - | - | - | - | 0.016210 | 0.002325 | 0.000023 | 1.000000 | -0.000067 | pass |
| oracle_top_1280_layer_8 | individual | 1 | - | - | - | - | 0.012875 | 0.002131 | 0.000013 | 1.000000 | 0.000044 | pass |
| oracle_top_1280_layer_9 | individual | 1 | - | - | - | - | 0.014572 | 0.002124 | 0.000014 | 1.000000 | 0.000077 | pass |
| oracle_top_1280_layer_10 | individual | 1 | - | - | - | - | 0.014272 | 0.002397 | 0.000028 | 1.000000 | 0.000930 | pass |
| oracle_top_1280_layer_11 | individual | 1 | - | - | - | - | 0.015809 | 0.002295 | 0.000021 | 1.000000 | 0.001080 | pass |
| oracle_top_1280_layer_12 | individual | 1 | - | - | - | - | 0.016181 | 0.002104 | 0.000017 | 1.000000 | -0.000358 | pass |
| oracle_top_1280_layer_13 | individual | 1 | - | - | - | - | 0.015597 | 0.002029 | 0.000012 | 1.000000 | 0.000011 | pass |
| oracle_top_1280_layer_14 | individual | 1 | - | - | - | - | 0.013191 | 0.001887 | 0.000011 | 1.000000 | -0.000284 | pass |
| oracle_top_1280_layer_15 | individual | 1 | - | - | - | - | 0.014061 | 0.001820 | 0.000009 | 1.000000 | 0.001063 | pass |
| oracle_top_1280_layer_16 | individual | 1 | - | - | - | - | 0.014888 | 0.001813 | 0.000008 | 1.000000 | 0.000709 | pass |
| oracle_top_1280_layer_17 | individual | 1 | - | - | - | - | 0.014084 | 0.001648 | 0.000008 | 1.000000 | 0.000120 | pass |
| oracle_top_1280_layer_18 | individual | 1 | - | - | - | - | 0.011773 | 0.001808 | 0.000009 | 1.000000 | -0.000698 | pass |
| oracle_top_1280_layer_19 | individual | 1 | - | - | - | - | 0.014090 | 0.002083 | 0.000010 | 1.000000 | 0.000043 | pass |
| oracle_top_1280_layer_20 | individual | 1 | - | - | - | - | 0.011452 | 0.002415 | 0.000017 | 1.000000 | 0.001004 | pass |
| oracle_top_1280_layer_21 | individual | 1 | - | - | - | - | 0.012835 | 0.002241 | 0.000012 | 1.000000 | -0.000416 | pass |
| oracle_top_1280_layer_22 | individual | 1 | - | - | - | - | 0.011433 | 0.002454 | 0.000015 | 1.000000 | 0.001091 | pass |
| oracle_top_1280_layer_23 | individual | 1 | - | - | - | - | 0.010705 | 0.002364 | 0.000010 | 1.000000 | -0.000305 | pass |
| oracle_top_1280_layer_24 | individual | 1 | - | - | - | - | 0.009894 | 0.002110 | 0.000010 | 1.000000 | 0.000284 | pass |
| oracle_top_1280_layer_25 | individual | 1 | - | - | - | - | 0.011091 | 0.001855 | 0.000011 | 1.000000 | -0.000375 | pass |
| oracle_top_1280_layer_26 | individual | 1 | - | - | - | - | 0.010600 | 0.001915 | 0.000009 | 1.000000 | 0.000065 | pass |
| oracle_top_1280_layer_27 | individual | 1 | - | - | - | - | 0.009214 | 0.001874 | 0.000011 | 1.000000 | -0.000795 | pass |
| oracle_top_1280_layer_28 | individual | 1 | - | - | - | - | 0.006925 | 0.001716 | 0.000009 | 1.000000 | -0.000033 | pass |
| oracle_top_1280_layer_29 | individual | 1 | - | - | - | - | 0.002896 | 0.002343 | 0.000053 | 1.000000 | 0.002966 | pass |
| oracle_top_1536_layer_0 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_1 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_2 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_3 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_4 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_5 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_6 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_7 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_8 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_9 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_10 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_11 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_12 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_13 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_14 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_15 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_16 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_17 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_18 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_19 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_20 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_21 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_22 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_23 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_24 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_25 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_26 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_27 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_28 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_1536_layer_29 | individual | 1 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |

> Forward hooks replace MLP outputs after the dense Hugging Face MLP executes; quality metrics are valid, but evaluator wall time is not an inference benchmark.

> The magnitude oracle uses all neuron activations to choose top-K and is not a realizable candidate-selection algorithm. Magnitude ranking is not the mathematically optimal K-subset when vector contributions can cancel, so this is a full-information reference rather than a theoretical quality ceiling.

Development decision: **insufficient_all_layer_evidence**.
