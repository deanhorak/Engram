# Trained-teacher MLP intervention

Status: **local_teacher_intervention_measurement**

Evaluation role: **development**.

Sequences: 4 (4 unique); next-token positions: 71; input-token positions: 75.

Exact-teacher NLL: 2.387150; perplexity: 10.882434.

| Arm | Scope | Layers | Input fraction | Candidates | Recall | Score mass | MLP rel-L2 | Final hidden rel-L2 | KL | Top-1 agreement | NLL delta | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| identity_all_selected_layers | all | 30 | - | - | - | - | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | pass |
| oracle_top_256_layer_0 | individual | 1 | - | - | - | - | 0.121798 | 0.085516 | 0.025529 | 0.943662 | 0.027265 | pass |
| oracle_top_256_layer_1 | individual | 1 | - | - | - | - | 0.282663 | 0.060389 | 0.020205 | 0.957746 | 0.003302 | pass |
| oracle_top_256_layer_2 | individual | 1 | - | - | - | - | 0.327580 | 0.057711 | 0.012623 | 0.957746 | -0.004100 | pass |
| oracle_top_256_layer_3 | individual | 1 | - | - | - | - | 0.412068 | 0.050438 | 0.008647 | 0.985915 | 0.005178 | pass |
| oracle_top_256_layer_4 | individual | 1 | - | - | - | - | 0.438258 | 0.051959 | 0.009686 | 1.000000 | -0.003487 | pass |
| oracle_top_256_layer_5 | individual | 1 | - | - | - | - | 0.442841 | 0.049243 | 0.008773 | 0.971831 | -0.014338 | pass |
| oracle_top_256_layer_6 | individual | 1 | - | - | - | - | 0.403171 | 0.057298 | 0.011783 | 0.957746 | 0.030865 | pass |
| oracle_top_256_layer_7 | individual | 1 | - | - | - | - | 0.391693 | 0.052194 | 0.006937 | 0.985915 | 0.043628 | pass |
| oracle_top_256_layer_8 | individual | 1 | - | - | - | - | 0.316458 | 0.052530 | 0.011088 | 0.971831 | -0.000876 | pass |
| oracle_top_256_layer_9 | individual | 1 | - | - | - | - | 0.354312 | 0.061640 | 0.014183 | 0.929577 | -0.017142 | pass |
| oracle_top_256_layer_10 | individual | 1 | - | - | - | - | 0.359714 | 0.072309 | 0.045732 | 0.943662 | -0.002526 | pass |
| oracle_top_256_layer_11 | individual | 1 | - | - | - | - | 0.400295 | 0.062007 | 0.012372 | 0.943662 | 0.021281 | pass |
| oracle_top_256_layer_12 | individual | 1 | - | - | - | - | 0.416964 | 0.054985 | 0.009826 | 0.985915 | 0.017076 | pass |
| oracle_top_256_layer_13 | individual | 1 | - | - | - | - | 0.383645 | 0.052597 | 0.009187 | 0.971831 | 0.016258 | pass |
| oracle_top_256_layer_14 | individual | 1 | - | - | - | - | 0.347281 | 0.051171 | 0.008664 | 0.957746 | 0.033137 | pass |
| oracle_top_256_layer_15 | individual | 1 | - | - | - | - | 0.358529 | 0.053044 | 0.009600 | 0.971831 | -0.002456 | pass |
| oracle_top_256_layer_16 | individual | 1 | - | - | - | - | 0.387326 | 0.051968 | 0.008472 | 0.985915 | -0.004615 | pass |
| oracle_top_256_layer_17 | individual | 1 | - | - | - | - | 0.386473 | 0.050566 | 0.007545 | 1.000000 | 0.004373 | pass |
| oracle_top_256_layer_18 | individual | 1 | - | - | - | - | 0.323098 | 0.055135 | 0.007076 | 0.985915 | -0.000323 | pass |
| oracle_top_256_layer_19 | individual | 1 | - | - | - | - | 0.365853 | 0.059116 | 0.010557 | 0.985915 | 0.003038 | pass |
| oracle_top_256_layer_20 | individual | 1 | - | - | - | - | 0.309793 | 0.068383 | 0.010409 | 0.971831 | 0.012196 | pass |
| oracle_top_256_layer_21 | individual | 1 | - | - | - | - | 0.336913 | 0.064402 | 0.008540 | 0.971831 | 0.008498 | pass |
| oracle_top_256_layer_22 | individual | 1 | - | - | - | - | 0.307012 | 0.070119 | 0.012671 | 0.985915 | 0.028361 | pass |
| oracle_top_256_layer_23 | individual | 1 | - | - | - | - | 0.278396 | 0.060621 | 0.009627 | 0.985915 | 0.019045 | pass |
| oracle_top_256_layer_24 | individual | 1 | - | - | - | - | 0.257528 | 0.056651 | 0.009389 | 0.971831 | 0.004640 | pass |
| oracle_top_256_layer_25 | individual | 1 | - | - | - | - | 0.287901 | 0.049987 | 0.007512 | 0.985915 | 0.017794 | pass |
| oracle_top_256_layer_26 | individual | 1 | - | - | - | - | 0.269859 | 0.049729 | 0.006418 | 0.971831 | 0.020026 | pass |
| oracle_top_256_layer_27 | individual | 1 | - | - | - | - | 0.248864 | 0.051472 | 0.007865 | 0.971831 | 0.008965 | pass |
| oracle_top_256_layer_28 | individual | 1 | - | - | - | - | 0.186740 | 0.048673 | 0.007376 | 0.971831 | 0.005292 | pass |
| oracle_top_256_layer_29 | individual | 1 | - | - | - | - | 0.089550 | 0.074555 | 0.077010 | 0.915493 | 0.048170 | fail |
| oracle_top_384_layer_0 | individual | 1 | - | - | - | - | 0.083827 | 0.052188 | 0.010028 | 0.971831 | 0.015488 | pass |
| oracle_top_384_layer_1 | individual | 1 | - | - | - | - | 0.213271 | 0.044951 | 0.010391 | 0.985915 | 0.003414 | pass |
| oracle_top_384_layer_2 | individual | 1 | - | - | - | - | 0.246241 | 0.038594 | 0.005722 | 0.985915 | 0.000461 | pass |
| oracle_top_384_layer_3 | individual | 1 | - | - | - | - | 0.311834 | 0.037613 | 0.003496 | 1.000000 | -0.007355 | pass |
| oracle_top_384_layer_4 | individual | 1 | - | - | - | - | 0.331745 | 0.036890 | 0.004858 | 0.985915 | -0.003824 | pass |
| oracle_top_384_layer_5 | individual | 1 | - | - | - | - | 0.340429 | 0.036790 | 0.004807 | 0.985915 | -0.010286 | pass |
| oracle_top_384_layer_6 | individual | 1 | - | - | - | - | 0.308058 | 0.041482 | 0.006430 | 0.971831 | 0.011544 | pass |
| oracle_top_384_layer_7 | individual | 1 | - | - | - | - | 0.298321 | 0.039182 | 0.005257 | 0.957746 | 0.032529 | pass |
| oracle_top_384_layer_8 | individual | 1 | - | - | - | - | 0.239722 | 0.036970 | 0.004582 | 0.985915 | -0.001872 | pass |
| oracle_top_384_layer_9 | individual | 1 | - | - | - | - | 0.268675 | 0.042391 | 0.005561 | 1.000000 | -0.008879 | pass |
| oracle_top_384_layer_10 | individual | 1 | - | - | - | - | 0.271236 | 0.045404 | 0.009504 | 0.957746 | -0.007456 | pass |
| oracle_top_384_layer_11 | individual | 1 | - | - | - | - | 0.301080 | 0.048394 | 0.007605 | 0.985915 | 0.020602 | pass |
| oracle_top_384_layer_12 | individual | 1 | - | - | - | - | 0.314806 | 0.041309 | 0.005130 | 1.000000 | 0.014874 | pass |
| oracle_top_384_layer_13 | individual | 1 | - | - | - | - | 0.291402 | 0.039979 | 0.005514 | 0.985915 | 0.005770 | pass |
| oracle_top_384_layer_14 | individual | 1 | - | - | - | - | 0.260961 | 0.038416 | 0.004209 | 0.971831 | 0.023174 | pass |
| oracle_top_384_layer_15 | individual | 1 | - | - | - | - | 0.269880 | 0.039296 | 0.005737 | 0.985915 | 0.001582 | pass |
| oracle_top_384_layer_16 | individual | 1 | - | - | - | - | 0.290922 | 0.037471 | 0.004385 | 0.985915 | 0.000982 | pass |
| oracle_top_384_layer_17 | individual | 1 | - | - | - | - | 0.287502 | 0.036937 | 0.003961 | 1.000000 | 0.002310 | pass |
| oracle_top_384_layer_18 | individual | 1 | - | - | - | - | 0.236954 | 0.039197 | 0.003704 | 0.985915 | -0.007257 | pass |
| oracle_top_384_layer_19 | individual | 1 | - | - | - | - | 0.272991 | 0.042757 | 0.006091 | 0.985915 | -0.013749 | pass |
| oracle_top_384_layer_20 | individual | 1 | - | - | - | - | 0.229140 | 0.050887 | 0.005686 | 0.985915 | 0.009568 | pass |
| oracle_top_384_layer_21 | individual | 1 | - | - | - | - | 0.251754 | 0.046598 | 0.005377 | 0.985915 | 0.007521 | pass |
| oracle_top_384_layer_22 | individual | 1 | - | - | - | - | 0.229071 | 0.051584 | 0.006798 | 0.985915 | 0.023553 | pass |
| oracle_top_384_layer_23 | individual | 1 | - | - | - | - | 0.204657 | 0.044680 | 0.004833 | 0.985915 | -0.003326 | pass |
| oracle_top_384_layer_24 | individual | 1 | - | - | - | - | 0.190540 | 0.041029 | 0.004522 | 1.000000 | -0.000943 | pass |
| oracle_top_384_layer_25 | individual | 1 | - | - | - | - | 0.215455 | 0.037111 | 0.003234 | 1.000000 | -0.000699 | pass |
| oracle_top_384_layer_26 | individual | 1 | - | - | - | - | 0.202108 | 0.036745 | 0.003292 | 0.971831 | 0.013707 | pass |
| oracle_top_384_layer_27 | individual | 1 | - | - | - | - | 0.182761 | 0.037667 | 0.003961 | 1.000000 | 0.009124 | pass |
| oracle_top_384_layer_28 | individual | 1 | - | - | - | - | 0.137866 | 0.035543 | 0.003955 | 0.971831 | -0.001852 | pass |
| oracle_top_384_layer_29 | individual | 1 | - | - | - | - | 0.062112 | 0.051901 | 0.031384 | 0.957746 | 0.010823 | pass |
| oracle_top_512_layer_0 | individual | 1 | - | - | - | - | 0.059060 | 0.037047 | 0.005478 | 1.000000 | 0.011200 | pass |
| oracle_top_512_layer_1 | individual | 1 | - | - | - | - | 0.159982 | 0.033080 | 0.004670 | 1.000000 | -0.002091 | pass |
| oracle_top_512_layer_2 | individual | 1 | - | - | - | - | 0.185573 | 0.029410 | 0.003197 | 0.985915 | -0.004561 | pass |
| oracle_top_512_layer_3 | individual | 1 | - | - | - | - | 0.236096 | 0.028994 | 0.002604 | 1.000000 | 0.002808 | pass |
| oracle_top_512_layer_4 | individual | 1 | - | - | - | - | 0.252588 | 0.027130 | 0.003336 | 0.971831 | -0.004293 | pass |
| oracle_top_512_layer_5 | individual | 1 | - | - | - | - | 0.261664 | 0.027447 | 0.002655 | 0.985915 | 0.002158 | pass |
| oracle_top_512_layer_6 | individual | 1 | - | - | - | - | 0.234306 | 0.032453 | 0.003102 | 0.985915 | 0.004790 | pass |
| oracle_top_512_layer_7 | individual | 1 | - | - | - | - | 0.227463 | 0.030147 | 0.002681 | 0.985915 | 0.013005 | pass |
| oracle_top_512_layer_8 | individual | 1 | - | - | - | - | 0.182755 | 0.029217 | 0.003014 | 1.000000 | -0.001327 | pass |
| oracle_top_512_layer_9 | individual | 1 | - | - | - | - | 0.204537 | 0.031440 | 0.003023 | 0.985915 | -0.011244 | pass |
| oracle_top_512_layer_10 | individual | 1 | - | - | - | - | 0.206235 | 0.037099 | 0.005890 | 0.985915 | -0.003835 | pass |
| oracle_top_512_layer_11 | individual | 1 | - | - | - | - | 0.226568 | 0.034232 | 0.003820 | 0.971831 | 0.023595 | pass |
| oracle_top_512_layer_12 | individual | 1 | - | - | - | - | 0.237748 | 0.030871 | 0.002991 | 1.000000 | 0.004911 | pass |
| oracle_top_512_layer_13 | individual | 1 | - | - | - | - | 0.223060 | 0.030742 | 0.003098 | 0.985915 | 0.004471 | pass |
| oracle_top_512_layer_14 | individual | 1 | - | - | - | - | 0.196511 | 0.028695 | 0.002782 | 1.000000 | 0.019109 | pass |
| oracle_top_512_layer_15 | individual | 1 | - | - | - | - | 0.201515 | 0.028835 | 0.002895 | 0.985915 | 0.010221 | pass |
| oracle_top_512_layer_16 | individual | 1 | - | - | - | - | 0.219609 | 0.029101 | 0.002418 | 1.000000 | 0.000445 | pass |
| oracle_top_512_layer_17 | individual | 1 | - | - | - | - | 0.216030 | 0.028515 | 0.002334 | 0.985915 | 0.004979 | pass |
| oracle_top_512_layer_18 | individual | 1 | - | - | - | - | 0.177463 | 0.029143 | 0.002566 | 0.985915 | -0.009744 | pass |
| oracle_top_512_layer_19 | individual | 1 | - | - | - | - | 0.204359 | 0.030833 | 0.002670 | 0.985915 | -0.005939 | pass |
| oracle_top_512_layer_20 | individual | 1 | - | - | - | - | 0.171983 | 0.037031 | 0.002975 | 1.000000 | 0.007644 | pass |
| oracle_top_512_layer_21 | individual | 1 | - | - | - | - | 0.188104 | 0.033519 | 0.002595 | 0.985915 | 0.008027 | pass |
| oracle_top_512_layer_22 | individual | 1 | - | - | - | - | 0.170113 | 0.036772 | 0.003661 | 0.985915 | 0.005719 | pass |
| oracle_top_512_layer_23 | individual | 1 | - | - | - | - | 0.152390 | 0.033805 | 0.002365 | 1.000000 | 0.004963 | pass |
| oracle_top_512_layer_24 | individual | 1 | - | - | - | - | 0.143088 | 0.030681 | 0.002736 | 0.985915 | 0.003349 | pass |
| oracle_top_512_layer_25 | individual | 1 | - | - | - | - | 0.162417 | 0.027699 | 0.001767 | 0.985915 | 0.002444 | pass |
| oracle_top_512_layer_26 | individual | 1 | - | - | - | - | 0.152059 | 0.027896 | 0.001885 | 0.971831 | 0.005972 | pass |
| oracle_top_512_layer_27 | individual | 1 | - | - | - | - | 0.136100 | 0.027833 | 0.002221 | 1.000000 | 0.007648 | pass |
| oracle_top_512_layer_28 | individual | 1 | - | - | - | - | 0.103077 | 0.026451 | 0.002041 | 0.985915 | -0.000350 | pass |
| oracle_top_512_layer_29 | individual | 1 | - | - | - | - | 0.044425 | 0.036503 | 0.013890 | 0.971831 | -0.005486 | pass |
| oracle_top_640_layer_0 | individual | 1 | - | - | - | - | 0.042345 | 0.027680 | 0.004079 | 0.985915 | 0.003659 | pass |
| oracle_top_640_layer_1 | individual | 1 | - | - | - | - | 0.118882 | 0.023584 | 0.002030 | 1.000000 | -0.005907 | pass |
| oracle_top_640_layer_2 | individual | 1 | - | - | - | - | 0.138591 | 0.022026 | 0.001968 | 1.000000 | -0.004785 | pass |
| oracle_top_640_layer_3 | individual | 1 | - | - | - | - | 0.175748 | 0.021428 | 0.001626 | 1.000000 | 0.005805 | pass |
| oracle_top_640_layer_4 | individual | 1 | - | - | - | - | 0.189020 | 0.020202 | 0.001894 | 0.985915 | -0.001052 | pass |
| oracle_top_640_layer_5 | individual | 1 | - | - | - | - | 0.197870 | 0.021119 | 0.001869 | 0.985915 | 0.004014 | pass |
| oracle_top_640_layer_6 | individual | 1 | - | - | - | - | 0.175578 | 0.022183 | 0.001522 | 0.971831 | 0.001444 | pass |
| oracle_top_640_layer_7 | individual | 1 | - | - | - | - | 0.170401 | 0.023450 | 0.001557 | 1.000000 | 0.006691 | pass |
| oracle_top_640_layer_8 | individual | 1 | - | - | - | - | 0.136039 | 0.020969 | 0.001727 | 0.985915 | 0.001126 | pass |
| oracle_top_640_layer_9 | individual | 1 | - | - | - | - | 0.154382 | 0.022909 | 0.001900 | 0.971831 | -0.009205 | pass |
| oracle_top_640_layer_10 | individual | 1 | - | - | - | - | 0.153594 | 0.027637 | 0.002771 | 0.985915 | -0.012954 | pass |
| oracle_top_640_layer_11 | individual | 1 | - | - | - | - | 0.169014 | 0.025314 | 0.002362 | 1.000000 | 0.008707 | pass |
| oracle_top_640_layer_12 | individual | 1 | - | - | - | - | 0.178219 | 0.023654 | 0.001939 | 1.000000 | 0.005067 | pass |
| oracle_top_640_layer_13 | individual | 1 | - | - | - | - | 0.166577 | 0.022465 | 0.001620 | 0.985915 | 0.001523 | pass |
| oracle_top_640_layer_14 | individual | 1 | - | - | - | - | 0.144992 | 0.020709 | 0.001296 | 1.000000 | 0.009775 | pass |
| oracle_top_640_layer_15 | individual | 1 | - | - | - | - | 0.149317 | 0.021011 | 0.001566 | 0.985915 | 0.009108 | pass |
| oracle_top_640_layer_16 | individual | 1 | - | - | - | - | 0.161608 | 0.020225 | 0.001498 | 1.000000 | 0.000709 | pass |
| oracle_top_640_layer_17 | individual | 1 | - | - | - | - | 0.158935 | 0.021497 | 0.001463 | 1.000000 | 0.010571 | pass |
| oracle_top_640_layer_18 | individual | 1 | - | - | - | - | 0.131335 | 0.020794 | 0.001164 | 1.000000 | -0.003089 | pass |
| oracle_top_640_layer_19 | individual | 1 | - | - | - | - | 0.152063 | 0.023006 | 0.001273 | 0.985915 | -0.003982 | pass |
| oracle_top_640_layer_20 | individual | 1 | - | - | - | - | 0.126595 | 0.026669 | 0.001586 | 1.000000 | 0.003888 | pass |
| oracle_top_640_layer_21 | individual | 1 | - | - | - | - | 0.138184 | 0.024254 | 0.001298 | 1.000000 | -0.004799 | pass |
| oracle_top_640_layer_22 | individual | 1 | - | - | - | - | 0.125651 | 0.026977 | 0.002009 | 1.000000 | -0.001218 | pass |
| oracle_top_640_layer_23 | individual | 1 | - | - | - | - | 0.112975 | 0.024978 | 0.001341 | 1.000000 | -0.003644 | pass |
| oracle_top_640_layer_24 | individual | 1 | - | - | - | - | 0.105999 | 0.022720 | 0.001290 | 1.000000 | -0.002734 | pass |
| oracle_top_640_layer_25 | individual | 1 | - | - | - | - | 0.121145 | 0.020811 | 0.000993 | 0.985915 | 0.001903 | pass |
| oracle_top_640_layer_26 | individual | 1 | - | - | - | - | 0.113018 | 0.020476 | 0.000872 | 0.985915 | 0.003176 | pass |
| oracle_top_640_layer_27 | individual | 1 | - | - | - | - | 0.101131 | 0.020665 | 0.001306 | 0.985915 | 0.007253 | pass |
| oracle_top_640_layer_28 | individual | 1 | - | - | - | - | 0.076068 | 0.019437 | 0.001042 | 0.985915 | -0.000402 | pass |
| oracle_top_640_layer_29 | individual | 1 | - | - | - | - | 0.032040 | 0.026261 | 0.007404 | 0.985915 | -0.010932 | pass |
| oracle_top_768_layer_0 | individual | 1 | - | - | - | - | 0.029599 | 0.018696 | 0.001831 | 1.000000 | 0.003024 | pass |
| oracle_top_768_layer_1 | individual | 1 | - | - | - | - | 0.085464 | 0.016158 | 0.001083 | 1.000000 | -0.005058 | pass |
| oracle_top_768_layer_2 | individual | 1 | - | - | - | - | 0.100351 | 0.015481 | 0.000995 | 1.000000 | -0.002400 | pass |
| oracle_top_768_layer_3 | individual | 1 | - | - | - | - | 0.127861 | 0.014003 | 0.000771 | 1.000000 | 0.001776 | pass |
| oracle_top_768_layer_4 | individual | 1 | - | - | - | - | 0.137284 | 0.013540 | 0.000713 | 1.000000 | 0.000568 | pass |
| oracle_top_768_layer_5 | individual | 1 | - | - | - | - | 0.145715 | 0.014792 | 0.000900 | 0.985915 | 0.000047 | pass |
| oracle_top_768_layer_6 | individual | 1 | - | - | - | - | 0.128886 | 0.015501 | 0.000939 | 0.985915 | -0.001708 | pass |
| oracle_top_768_layer_7 | individual | 1 | - | - | - | - | 0.124724 | 0.016815 | 0.000855 | 1.000000 | 0.004563 | pass |
| oracle_top_768_layer_8 | individual | 1 | - | - | - | - | 0.098841 | 0.015409 | 0.000826 | 1.000000 | 0.000135 | pass |
| oracle_top_768_layer_9 | individual | 1 | - | - | - | - | 0.112323 | 0.016301 | 0.000779 | 0.985915 | -0.001940 | pass |
| oracle_top_768_layer_10 | individual | 1 | - | - | - | - | 0.110761 | 0.018550 | 0.001285 | 1.000000 | 0.003831 | pass |
| oracle_top_768_layer_11 | individual | 1 | - | - | - | - | 0.122977 | 0.017622 | 0.001084 | 1.000000 | 0.008251 | pass |
| oracle_top_768_layer_12 | individual | 1 | - | - | - | - | 0.128114 | 0.016200 | 0.000944 | 1.000000 | 0.006252 | pass |
| oracle_top_768_layer_13 | individual | 1 | - | - | - | - | 0.121804 | 0.015945 | 0.000778 | 1.000000 | 0.004428 | pass |
| oracle_top_768_layer_14 | individual | 1 | - | - | - | - | 0.105306 | 0.014849 | 0.000756 | 1.000000 | 0.006017 | pass |
| oracle_top_768_layer_15 | individual | 1 | - | - | - | - | 0.108347 | 0.014956 | 0.000757 | 1.000000 | 0.005826 | pass |
| oracle_top_768_layer_16 | individual | 1 | - | - | - | - | 0.116850 | 0.014416 | 0.000865 | 1.000000 | -0.000290 | pass |
| oracle_top_768_layer_17 | individual | 1 | - | - | - | - | 0.113928 | 0.015418 | 0.000759 | 1.000000 | 0.005980 | pass |
| oracle_top_768_layer_18 | individual | 1 | - | - | - | - | 0.094255 | 0.014544 | 0.000620 | 1.000000 | -0.001986 | pass |
| oracle_top_768_layer_19 | individual | 1 | - | - | - | - | 0.110334 | 0.016324 | 0.000726 | 0.985915 | 0.000637 | pass |
| oracle_top_768_layer_20 | individual | 1 | - | - | - | - | 0.091356 | 0.019022 | 0.000839 | 1.000000 | 0.001385 | pass |
| oracle_top_768_layer_21 | individual | 1 | - | - | - | - | 0.099865 | 0.017395 | 0.000674 | 1.000000 | -0.001773 | pass |
| oracle_top_768_layer_22 | individual | 1 | - | - | - | - | 0.090035 | 0.019264 | 0.000785 | 1.000000 | -0.001068 | pass |
| oracle_top_768_layer_23 | individual | 1 | - | - | - | - | 0.081681 | 0.017987 | 0.000731 | 0.985915 | -0.006934 | pass |
| oracle_top_768_layer_24 | individual | 1 | - | - | - | - | 0.077019 | 0.016584 | 0.000593 | 1.000000 | 0.000950 | pass |
| oracle_top_768_layer_25 | individual | 1 | - | - | - | - | 0.087383 | 0.014868 | 0.000523 | 0.985915 | -0.001448 | pass |
| oracle_top_768_layer_26 | individual | 1 | - | - | - | - | 0.082239 | 0.014888 | 0.000502 | 1.000000 | -0.003034 | pass |
| oracle_top_768_layer_27 | individual | 1 | - | - | - | - | 0.072753 | 0.014782 | 0.000603 | 1.000000 | 0.006337 | pass |
| oracle_top_768_layer_28 | individual | 1 | - | - | - | - | 0.054780 | 0.013916 | 0.000630 | 1.000000 | -0.002452 | pass |
| oracle_top_768_layer_29 | individual | 1 | - | - | - | - | 0.022894 | 0.018594 | 0.003353 | 1.000000 | -0.016093 | pass |

> Forward hooks replace MLP outputs after the dense Hugging Face MLP executes; quality metrics are valid, but evaluator wall time is not an inference benchmark.

> The magnitude oracle uses all neuron activations to choose top-K and is not a realizable candidate-selection algorithm. Magnitude ranking is not the mathematically optimal K-subset when vector contributions can cancel, so this is a full-information reference rather than a theoretical quality ceiling.

Development decision: **insufficient_all_layer_evidence**.
