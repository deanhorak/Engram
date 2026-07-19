# Gate 5 random-fixture quality report

This evaluation used a deterministic, randomly initialized one-layer Llama fixture. It validates
the local-teacher comparison path and does not measure useful language ability.

| Metric | Measured value |
|---|---:|
| Evaluated next-token positions | 18 |
| Student perplexity | 31.999 |
| Teacher/student KL | 0.00163 |
| Teacher top-1 agreement | 61.1% |
| Teacher top-5 agreement | 88.9% |
| Code/reasoning/factual/long-context target accuracy | 0% each |
| Repetition fraction over 32 generated tokens | 93.75% |

The small KL and moderate teacher agreement are artifacts of two near-uniform random models;
they do not indicate successful compilation. Zero next-token task accuracy and severe repetition
are the important negative outcomes. Trained-model targets remain unevaluated because no trained
local checkpoint is available.
