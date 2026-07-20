# SmolLM2-135M coverage-trained hierarchical routing

Status: **measured improvement over post-hoc groups, but still below flat routing**

This experiment trains the hierarchy directly against oracle coverage. Records are first placed
into balanced groups by clustering their oracle-membership columns across 128 calibration states.
The target for each state and group is then the number of oracle top-256 records contained in that
group. A ridge model predicts those group utilities and is compressed to rank 16. The highest
scoring groups provide exactly 512 or 640 records, which are then eligible for exact SwiGLU
reranking.

Evaluation uses 32 disjoint validation states from all 30 layers of
`HuggingFaceTB/SmolLM2-135M`, giving 960 measurements per configuration.

## Direct coverage results

| Groups | Records/group | Mean recall @512 | Mean recall @640 | Rank-16 bytes/layer | Posting bytes/layer |
|---:|---:|---:|---:|---:|---:|
| 48 | 32 | 0.5437 | 0.6201 | 40,128 | 3,072 |
| **96** | **16** | **0.5459** | **0.6221** | **43,392** | **3,072** |
| 192 | 8 | 0.5411 | 0.6217 | 49,920 | 3,072 |
| Flat rank 16 | 1 | 0.6328 | 0.6990 | 141,312 | 0 |

Ridge regularization 1,000 was best across the tested values 100, 1,000, and 10,000. Compared with
post-hoc embedding groups, direct coverage training improves recall at 512 records from 0.4778 to
0.5437 for 48 groups, from 0.5008 to 0.5459 for 96 groups, and from 0.5282 to 0.5411 for 192 groups.

## Multiple representatives

Each group was also divided into one, two, or four balanced sub-postings. A separate binary target
predicted whether each sub-posting contained any oracle record; representative scores were
combined by either their sum or maximum.

| Groups | Representatives | Aggregation | Mean recall @512 | Router bytes/layer |
|---:|---:|---|---:|---:|
| 48 | 4 | Sum | 0.5386 | 49,920 |
| 96 | 4 | Sum | **0.5450** | 62,976 |
| 192 | 4 | Sum | 0.5400 | 89,088 |

Maximum aggregation was consistently worse than summation. Multiple representatives did not beat
the simpler count target: the best result, 0.5450, is statistically and practically tied with but
slightly below the single group-utility model's 0.5459, while requiring 45% more router bytes at
96 groups.

## Decision

Training for coverage fixes part of the objective mismatch in post-hoc grouping, especially for
large groups. It does not close the gap to flat rank-16 scoring: the best hierarchy still loses
8.69 percentage points of recall at 512 records. Exact reranking cannot repair this candidate-set
miss rate because absent groups have already discarded those oracle records.

The balanced partition constraint is now the likely bottleneck. The next candidate-generation
experiment should allow overlapping learned postings and optimize their contents with a greedy
coverage objective, then train the low-rank query model to select a small posting combination.
