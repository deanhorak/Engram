# Oracle cache-line coverage bound

Status: **candidate locality rejected for the current static record order**

This diagnostic uses exact top-512 contribution membership on all 15,210 held-out
state/layer pairs from the expanded validation traces. It grants a hypothetical group selector
perfect knowledge of the oracle set, sorts the 96 contiguous 16-record lines by their oracle
occupancy, and measures the best coverage possible at each line budget. A real router cannot
exceed this bound.

Exact top-512 membership touches 95.858 of 96 lines on average; the median is all 96 lines.

| Selected lines | Perfect-selector mean recall | 5th percentile | Minimum |
|---:|---:|---:|---:|
| 32 | 0.4603 | 0.4434 | 0.4121 |
| 48 | 0.6387 | 0.6211 | 0.5957 |
| 64 | 0.7924 | 0.7773 | 0.7520 |
| 80 | 0.9175 | 0.9062 | 0.8867 |
| 88 | 0.9665 | 0.9590 | 0.9492 |
| 96 | 1.0000 | 1.0000 | 1.0000 |

Therefore a 95% mean-recall target needs between 80 and 88 of the 96 lines even with oracle
group ranking. Optimizing the present router for contiguous candidate locality cannot provide a
material completion-traffic reduction. This is a representation/layout bound, not a statement
that all learned or dynamic physical layouts are impossible.
