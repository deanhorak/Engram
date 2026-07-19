# Milestone 1: MLP magnitude-oracle sparsity

Status: **pipeline_validation**

> These measurements use deterministic random fixture weights. They validate the
> experiment pipeline and make no claim about sparsity in a trained language model.

Source hash: `e4e93594bda24ab2894800d137de8f688fd818baf2cd1c378573ed70ffd52d12`

Oracle: all neuron activations are computed; neurons are ranked by abs(activation_j) * L2(value_j), and every ranked prefix is scanned.

Caveat: magnitude oracle; not a combinatorial optimum under vector cancellation.

Background comparison: not run (Milestone 2). Gate 1 is therefore incomplete.

| Scope | Layer | Input type | Target | Mean active fraction | Median rel-L2 | p95 rel-L2 | Mean cosine |
|---|---:|---|---:|---:|---:|---:|---:|
| all | - | - | 90% | 0.268555 | 0.290413 | 0.313656 | 0.960297 |
| all | - | - | 95% | 0.354492 | 0.200798 | 0.222067 | 0.980752 |
| all | - | - | 99% | 0.550293 | 0.091095 | 0.099332 | 0.996197 |
| layer | 0 | - | 90% | 0.273438 | 0.292045 | 0.313518 | 0.958918 |
| layer | 0 | - | 95% | 0.356445 | 0.203505 | 0.221324 | 0.980611 |
| layer | 0 | - | 99% | 0.548828 | 0.092164 | 0.099529 | 0.996070 |
| layer | 1 | - | 90% | 0.263672 | 0.288422 | 0.312752 | 0.961676 |
| layer | 1 | - | 95% | 0.352539 | 0.196434 | 0.222490 | 0.980893 |
| layer | 1 | - | 99% | 0.551758 | 0.088814 | 0.098633 | 0.996323 |
| layer_input_type | 0 | code | 90% | 0.300781 | 0.280658 | 0.306644 | 0.961761 |
| layer_input_type | 0 | code | 95% | 0.382812 | 0.209503 | 0.220811 | 0.981128 |
| layer_input_type | 0 | code | 99% | 0.550781 | 0.091117 | 0.098309 | 0.996014 |
| layer_input_type | 0 | conversation | 90% | 0.253906 | 0.292937 | 0.307093 | 0.960326 |
| layer_input_type | 0 | conversation | 95% | 0.320312 | 0.211582 | 0.215398 | 0.979464 |
| layer_input_type | 0 | conversation | 99% | 0.554688 | 0.094004 | 0.099075 | 0.996391 |
| layer_input_type | 0 | prose | 90% | 0.265625 | 0.295550 | 0.312717 | 0.957445 |
| layer_input_type | 0 | prose | 95% | 0.343750 | 0.195547 | 0.222291 | 0.981031 |
| layer_input_type | 0 | prose | 99% | 0.519531 | 0.095199 | 0.098377 | 0.995877 |
| layer_input_type | 0 | structured | 90% | 0.273438 | 0.300214 | 0.312702 | 0.956140 |
| layer_input_type | 0 | structured | 95% | 0.378906 | 0.195076 | 0.219103 | 0.980822 |
| layer_input_type | 0 | structured | 99% | 0.570312 | 0.091547 | 0.098933 | 0.995998 |
| layer_input_type | 1 | code | 90% | 0.277344 | 0.291623 | 0.307811 | 0.960751 |
| layer_input_type | 1 | code | 95% | 0.343750 | 0.196574 | 0.216198 | 0.980754 |
| layer_input_type | 1 | code | 99% | 0.535156 | 0.084659 | 0.094538 | 0.996562 |
| layer_input_type | 1 | conversation | 90% | 0.289062 | 0.283546 | 0.300314 | 0.963926 |
| layer_input_type | 1 | conversation | 95% | 0.363281 | 0.187608 | 0.220493 | 0.982363 |
| layer_input_type | 1 | conversation | 99% | 0.597656 | 0.081701 | 0.093477 | 0.996948 |
| layer_input_type | 1 | prose | 90% | 0.265625 | 0.284922 | 0.311317 | 0.962295 |
| layer_input_type | 1 | prose | 95% | 0.363281 | 0.194225 | 0.220328 | 0.980742 |
| layer_input_type | 1 | prose | 99% | 0.542969 | 0.094204 | 0.098415 | 0.995792 |
| layer_input_type | 1 | structured | 90% | 0.222656 | 0.289684 | 0.313331 | 0.959731 |
| layer_input_type | 1 | structured | 95% | 0.339844 | 0.207389 | 0.221888 | 0.979712 |
| layer_input_type | 1 | structured | 99% | 0.531250 | 0.092994 | 0.099147 | 0.995990 |

The JSON companion contains mean, median, and p95 for every reported metric.
