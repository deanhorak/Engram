# Dynamic Input Pruning exact-completion sweep

Status: **eligible_for_causal_intervention**

Held-out evidence: 507 states per layer from 16 unique sequences across 30 layers.

Method: retain the largest-magnitude input coordinates, use the source model's partial gate/up projections to score every intermediate record, exactly complete only the candidate records, then rerank candidates at full precision.

Published DIP motivates input pruning and partial scoring; candidate-only exact completion plus contribution-norm reranking is an Engram extension.

Top-K magnitude reference mean local relative L2: 0.104020.

| Input coordinates | Candidates | Top-K | Recall | Oracle score mass | Local rel-L2 | Projected dense traffic | Reduction | Trace screen |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 288 (0.500) | 896 | 768 | 0.915021 | 0.960167 | 0.129891 | 0.694444 | 1.440x | reject |
| 288 (0.500) | 1024 | 768 | 0.950523 | 0.976717 | 0.119028 | 0.722222 | 1.385x | recall pass |
| 288 (0.500) | 1152 | 768 | 0.971748 | 0.986629 | 0.112748 | 0.750000 | 1.333x | recall pass |
| 288 (0.500) | 1280 | 768 | 0.985038 | 0.992883 | 0.108762 | 0.777778 | 1.286x | recall pass |
| 360 (0.625) | 896 | 768 | 0.960089 | 0.983302 | 0.111444 | 0.729167 | 1.371x | recall pass |
| 360 (0.625) | 1024 | 768 | 0.982055 | 0.992407 | 0.107515 | 0.750000 | 1.333x | recall pass |
| 360 (0.625) | 1152 | 768 | 0.991247 | 0.996253 | 0.105853 | 0.770833 | 1.297x | near-oracle |
| 360 (0.625) | 1280 | 768 | 0.995704 | 0.998144 | 0.104988 | 0.791667 | 1.263x | near-oracle |
| 432 (0.750) | 896 | 768 | 0.989899 | 0.996211 | 0.105008 | 0.763889 | 1.309x | recall pass |
| 432 (0.750) | 1024 | 768 | 0.997080 | 0.998895 | 0.104376 | 0.777778 | 1.286x | near-oracle |
| 432 (0.750) | 1152 | 768 | 0.998786 | 0.999539 | 0.104191 | 0.791667 | 1.263x | near-oracle |
| 432 (0.750) | 1280 | 768 | 0.999439 | 0.999787 | 0.104100 | 0.805556 | 1.241x | near-oracle |

Lowest-traffic near-oracle trace arm: **dip_q360_c1152_k768**, with 0.991247 mean recall and 0.770833 projected dense weight traffic.

> The NumPy evaluator executes dense matrix products. Traffic is a logical float32 weight-read projection for a future sparse kernel, excludes indexes/activations/cache effects, and is not measured DRAM traffic or latency.

> This clean-state trace screen does not measure hidden-state drift, logit KL, target NLL, or top-1 agreement; the trained-teacher causal intervention remains decisive.
