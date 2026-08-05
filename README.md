# Engram

Engram asks a practical research question: can we take knowledge and behavior learned by a
Llama-family transformer and reorganize them into a much smaller, CPU-native inference system?

A normal transformer evaluates every layer and most model weights for every token. Engram's
target design does something different:

1. A small recurrent **controller** maintains the current language-model state and allocates
   bounded internal update cycles.
2. A sparse **semantic memory** stores the useful records extracted from transformer MLPs and
   retrieves only a small relevant subset for each token.
3. A bounded **episodic memory** combines exact recent context with compressed older context.
4. A CPU-native runtime executes the compiled representation without PyTorch or the original
   transformer layers.

The intended result is not a wrapper, cache, or quantized copy of the source transformer. It is
a different inference architecture compiled from a trained model.

## Goals

- Preserve useful next-token behavior from a trained Llama-compatible teacher.
- Avoid reading the full transformer parameter set for every generated token.
- Bound working memory as context grows instead of retaining an unlimited attention cache.
- Run efficiently on ordinary CPUs with an inspectable C++20 implementation.
- Measure quality, latency, memory traffic, and failure modes honestly at every research gate.

The long-term systems target is a substantial reduction in DRAM traffic—ideally around 10x—while
retaining useful model quality. That target is a hypothesis, not a result. Engram will not claim
success from random fixtures, synthetic tasks, proxy byte counts, or a runnable compiler alone.

## Practical direction: the hybrid architecture

The original layer-free replacement remains a research result, not a validated product path. The
current practical direction is therefore hybrid: keep a conventional quantized model as the
quality anchor and run Engram as a CPU sidecar around it.

The first hybrid boundary is deliberately model-agnostic. Engram reads a JSONL memory file,
selects a small number of relevant records with a deterministic CPU hashing index, bounds their
total context size, labels them as untrusted reference material, and sends the resulting messages
to an OpenAI-compatible chat endpoint. A `llama.cpp` server is one supported host, but the host
could be any local compatible implementation. Engram does not inspect or replace the host's
hidden states, decoder layers, or logits in this mode.

Start a compatible host, then run:

```bash
PYTHONPATH=src python -m engram.cli chat-hybrid \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --model local-model \
  --memory path/to/memory.jsonl \
  --top-k 4 \
  --min-score 0.15 \
  --max-context-chars 4000 \
  --max-tokens 128
```

Ollama's native endpoint is also supported. For reasoning-capable Qwen3
models, `--no-think` makes the endpoint return answer content within the
requested token budget:

```bash
PYTHONPATH=src python -m engram.cli chat-hybrid \
  --endpoint http://127.0.0.1:11434/api/chat \
  --model qwen3:latest \
  --memory path/to/memory.jsonl \
  --no-think
```

Each memory line is an object such as
`{"id":"project-goal","text":"...","metadata":{"source":"notes"}}`.
Use `--mode baseline` to bypass retrieval while keeping the same host and conversation loop.
`benchmark-hybrid` sends the same prompt set through baseline and augmented modes and records
host latency, usage, and retrieved IDs. It intentionally reports
`quality_claim: not_established`; answer quality must be scored with an independent task rubric.

This first sidecar is a reproducible lexical retrieval baseline, not a claim that hashed text
embeddings are the final Engram memory. A later experiment can replace the encoder with a frozen
model embedding or a model-specific semantic index without changing the host protocol. The
hybrid go/no-go question is now concrete: does bounded retrieval improve a fixed task-quality
score at equal or lower end-to-end cost than the same host without Engram?

The first CPU-only Ollama screen answered only the plumbing question. With
Qwen3 on 100% CPU, one 16-token prompt took 5.47 s baseline versus 13.80 s
with retrieved context (33 versus 99 prompt tokens). Retrieval was correct,
but this sidecar configuration is not yet a performance improvement; see
[`hybrid_ollama_cpu_smoke_2026-08-04.json`](reports/hybrid_ollama_cpu_smoke_2026-08-04.json).

For an external, reproducible stress test, the project also freezes the public
[LongEmbed Passkey auxiliary benchmark](docs/auxiliary_benchmarks.md) at an upstream Git revision
and SHA-256 manifest. This benchmark is evaluation-only and explicitly does **not** replace or
authorize the separately protected Engram gate.

## Two levels of control

Engram's compiled runtime and its longer-term system architecture solve different problems. The
runtime controller is a low-level numeric mechanism inside one model worker. Above it, Engram is
developing an optional **Oracle cognitive executive** that represents goals, scopes attention,
estimates evidence confidence, proposes memory retention, selects strategies and workers, and
monitors progress under explicit cost and risk policy.

The executive produces typed decisions rather than prose and does not sit in the per-token hot
loop. Its deterministic policy, revisioned SQLite/JSONL/in-memory event stores, versioned worker
registry, resource ledger, worker-adapter boundary, outcome-observation loop, and calibration
metrics are implemented. Production model/tool adapters, content validators, deployment security,
and learned predictors are not. See
[The Oracle cognitive executive](docs/cognitive_executive.md) for its contracts, boundaries,
safety requirements, and separate research gates.

## Where the project stands

**Current decision:** the native-BitNet practical semantic-memory gate has
**passed by postmortem adjudication** on its independent, frozen
8-sequence/256-position holdout. The CPU-only native Dynamic Input Pruning
(DIP) kernel substituted all 30 MLPs with live BF16 boundaries and no dense
fallback. The final raw evaluator report measured KL **0.00404129**, teacher
top-1 agreement **0.98828125**, NLL delta **+0.00482893**, final-hidden
relative L2 **0.0477494**, mean active-record fraction **0.2138001**, modeled
physical cold traffic **0.4113713** of dense ideal Q4, global candidate recall
**0.9994058**, and worst-layer mean recall **0.9939429**.

This is not a pristine one-shot runner pass. The original runner consumed the
holdout and ended in `error` after the completed raw evaluator report because
its verifier compared the protocol's frozen full-record hashes, made with the
canonical `input_ids` object envelope, against evaluator hashes of the first
33 scored tokens made with a bare-list envelope. A separate no-model
postmortem adjudicator corrected that hash contract, checked the preserved
evidence and every frozen threshold, and returned
`milestone_2_semantic_gate_passed_by_postmortem_adjudication`. The
raw report was prospectively hash-sealed about 13 minutes after the runner
error, not contemporaneously bound by the original result. The evidence is
therefore sufficient for this repository's semantic-gate decision, but weaker
than a clean independently sealed rerun.

The practical selector keeps the largest 1,920 of 2,560 BF16 input
coordinates, scans their coordinate-major packed ternary gate/up keys, exactly
completes a frozen per-layer candidate budget, estimates the coupled
intermediate RMS, and reads down rows only for token-adaptive nonzero
candidates. This is now a real routed semantic-memory implementation, not the
earlier dense-membership oracle. Its complete final sparse run was still
**1.1449x the dense elapsed time** (14.49% slower), however, and latency was
not a frozen gate. The traffic result is deterministic cache-line accounting,
not a hardware-counter measurement of DRAM. The artifact and native-library
bindings are also host-bound; broader replication remains required.

The original dense-Llama conversion track remains blocked; this pass belongs
to the separately trained native-BitNet source track. The holdout is a
checked-in plaintext fixture whose non-use before the attempt was enforced by
project procedure, not by cryptographic secrecy. This adjudicated semantic
result does not establish a quality-preserving dense-Llama conversion and does
not, by itself, certify every broader Milestone 2 deliverable as complete.

### Current controller boundary

The exact operator-residual controller also has a standalone CPU-only replay
runtime. `engram evaluate-controller-only` consumes a serialized controller
and captured semantic/episodic operator streams, without constructing a
Transformers model or calling decoder layers. On the sequence-disjoint
16-sequence validation trace it reaches terminal normalized MSE **0.0000208009**
and records zero decoder-layer calls. This is a state-transition pass, not yet
layer-free generation: an independent semantic/episodic provider and a
causally qualified learned correction are still required.

The first evidence-sized causal controller trial has now been run on CPU
(8-sequence training split, 16-sequence/256-position held-out split, top-32
teacher logits, rank 128, 500 steps). It preserves CPU serialization parity but
fails free-run validation (terminal normalized MSE **0.2624663**), so this
factorized controller is not promoted and the next experiment must change
capacity or the operator-stream provider.
The matched rank-256 capacity arm also fails (terminal normalized MSE
**0.2710301**), so rank expansion alone is not the next direction.

The provider seam is now executable and versioned. `engram fit-operator-provider`
creates a CPU/NumPy PCA-ridge provider conditioned on the current controller
state and token embedding; `engram evaluate-controller-provider` replays it
without a Transformers model or decoder-layer calls. The trace provider is
explicitly replay-only. The first learned rank-16 provider reaches held-out
terminal normalized MSE **0.2536094** (hidden MSE **0.8770986**) and therefore
fails the 0.0225 causal gate. Rank-64 and rank-128 providers also fail, so
these artifacts are research evidence rather than package inputs. The next
controller experiment must add temporal/context features or jointly optimize
provider and controller against the causal objective.
The first bounded joint projection adaptation improved the held-out terminal
MSE to **0.1970640** after 20 CPU steps, but remains well above the gate; its
report is retained under
`reports/controller_provider_pca_2026-08-03/joint20_train8x16_validation16x16.json`.

The runtime also has a sequence-preserving provider path: it resets context
once, advances it once per token, and then runs all controller stages. The
trace-backed reference reproduces the exact 16-sequence replay at terminal
MSE **0.0000208009**, but it is explicitly replay-only; a learned recurrent
provider still needs to pass the causal gate.

### Qwen3 as an alternative dense teacher

The original generic path was described as “Llama-compatible,” but the
semantic extraction contract is actually the canonical bias-free SiLU/SwiGLU
block: `down_proj(silu(gate_proj(x)) * up_proj(x))` with tensors named
`model.layers.<n>.mlp.{gate,up,down}_proj.weight`. Qwen3 uses that same block,
so Engram now has a separate, fail-closed Qwen3 source audit and does not
pretend that Qwen3 is a BitNet package.

The first pinned source experiment used the official
`Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca`. The CPU teacher has hidden size
1024, intermediate size 3072, 28 layers, 16 query heads, 8 KV heads, head
dimension 128, vocabulary 151,936, and RoPE theta 1,000,000. A two-sequence,
eight-position trace at layers 0/13/27 reconstructed the hooked Hugging Face
MLP outputs to maximum relative L2 **6.1929e-7**. This is a structural
source/trace pass only: no Qwen3 controller, causal quality result, or native
Qwen3 runtime exists, and the native BitNet compiler still rejects dense Qwen3
as a source.

Run the audit on a local downloaded checkpoint with:

```bash
PYTHONPATH=src python -m engram.cli audit-qwen3 \
  --model /path/to/Qwen3-0.6B \
  --out reports/qwen3_audit.json
```

The captured evidence and immutable hashes are in
`reports/qwen3_teacher_trace_2026-08-04.json`. That structural result called
for a larger frozen Qwen3 causal trace and a teacher-family comparison against
the failed BitNet provider—not another isolated provider rank sweep.

The Qwen3 adapter is covered by the full current regression checkpoint:
**1,145 Python tests passed, 1 skipped**, and native CTest **20/20 passed**.

That causal comparison is now complete. The new `trace-hf-controller` command
captured all 28 Qwen3 stages for disjoint 8-sequence/128-record training and
validation splits, preserving normalized controller states, semantic and
episodic outputs, and top-32 causal targets. A rank-16 operator-residual
controller exactly replays the validation transition (terminal normalized MSE
**6.91e-8**), but its rank-16 state/token PCA provider reaches **0.41916** on
the held-out terminal MSE versus the **0.0225** provider gate. This is a
CPU-only, zero-decoder-layer evaluation and a decisive negative result for
“change the teacher family only”; it does not alter the protected BitNet gate.
See `reports/qwen3_controller_provider_screen_2026-08-04.json`.

The remaining isolated causal-supervision test also failed to move this
boundary. A rank-16 controller trained with Qwen3 top-k loss (weight 0.25)
kept CPU reload parity, but held-out top-k KL changed from **2.24974** to
**2.27380** and the learned provider reached **0.42226** terminal MSE. This
is worse than the exact-controller provider's **0.41916**, so adding a causal
loss without jointly changing the provider is closed. See
`reports/qwen3_causal_supervision_screen_2026-08-04.json`.

The first joint architecture—stage-specific causal key/value memory plus
controller correction—improves the Qwen3 provider to **0.37301** terminal MSE
after 20 free-running steps (from **0.41916**), but a 60-step continuation
regresses to **0.38828**. The best arm is still 16.6× above the **0.0225**
gate. It is retained as evidence only; further tuning of this exact arm is
closed until a new representation, supervision signal, or larger causal corpus
is justified. See `reports/qwen3_joint_causal_provider_screen_2026-08-04.json`.

The replay provider is now a durable, checksummed artifact. For a frozen
trace, `engram evaluate-controller-sequence` reloads the sequence-shaped
semantic/episodic arrays, restores sample ordering, and executes the
transformer-free controller with one context advance per token. The CLI
reproduces terminal normalized MSE **0.0000208009** and zero decoder-layer
calls. This proves package/state durability, not learned generalization; the
artifact is explicitly marked `learned: false`.

Example validation command:

```bash
PYTHONPATH=src python -m engram.cli evaluate-controller-sequence \
  --trace work/controller_distillation/bitnet_validation_16x16_b4 \
  --provider work/controller_provider_trace_sequence_validation \
  --controller work/controller_distillation/bitnet_rank128_operator_residual_1024x256_exact/controller \
  --out reports/controller_provider_pca_2026-08-03/sequence_provider_cli_replay.json
```

The larger 64-sequence provider fit and 20-step causal adaptation were also
screened (terminal normalized MSE **0.2127623** and **0.2120011**). Compact
previous-state context, nearest-neighbor retrieval, and residual context
correction were rejected after worse held-out results. The current learned
provider therefore remains outside the authenticated package until the
0.0225 causal gate is met.

The next architecture screen is a diagonal state-space provider: a 64-wide
token memory feeds rank-16 semantic/episodic stage heads. Initialized from
the linear provider, 80 free-running CPU steps reach held-out terminal
normalized MSE **0.1926129**. This is measurable progress but not a
promotion; `StateSpaceOperatorStreamProvider` is serialized for longer
distillation while the **0.0225** causal gate remains open.

The resumable training command is `engram distill-state-space-provider`; it
can use CUDA for optimization when available, while the resulting provider
artifact and sequence runtime remain NumPy/CPU-only.

The current best provider uses the 64-sequence corpus with rank-16 λ=1
regularization and a full-width 64-memory residual state-space adapter.
Forty free-running CPU steps reach held-out terminal normalized MSE
**0.1777104**. The residual path is checksummed and exposed through
`engram distill-state-space-residual-provider`, but remains outside the
authenticated package until the **0.0225** causal gate is met.

Correction-only adaptation over the fixed λ=1 provider reaches held-out
terminal normalized MSE **0.1759220** after 50 CPU steps. It is available as
`engram adapt-controller-correction`; the nonzero controller is retained for
evaluation only and is not an authenticated promotion artifact.

The current learned-provider frontier is documented in
`docs/milestone_report.md`. Full-corpus rank-256 fitting reaches terminal MSE
**0.1710317**; a 128-wide persistent-memory residual reaches **0.1760428**;
and smooth scheduled sampling reaches **0.1749395**. None passes the fixed
**0.0225** causal gate. An explicit causal key/value provider is now also
implemented: it keeps compact keys and values for every prior token, forms a
stage-specific query from the controller state and token, and predicts a
low-rank residual over the PCA streams. It is CPU-serializable and evaluated
with the stateful command below, but its bounded protected screen reaches only
**0.1758242**, so it remains research-only.

The follow-up stage-local provider keeps a separate causal K/V prefix for each
of the 30 controller stages, matching their distinct hidden-state and operator
spaces. Its rank-64 latent screen reaches **0.1714502** after 20 free-running
CPU steps (from **0.1767018**); using the full-rank-256 base reaches
**0.1692925**. Both remain far above **0.0225**. A direct hidden-size residual
head does not generalize on this split (**0.1817167** and **0.1770538** in two
scheduled screens), so it is not promoted. Results and artifact hashes are
recorded in
`reports/controller_provider_pca_2026-08-03/stage_causal_attention_screens.json`.
Jointly unfreezing the controller's step scale, stage embeddings, and
low-rank adapters changes the latent result only to **0.1714471**, so that
small co-adaptation arm is also closed. Unfreezing every controller tensor
reaches **0.1714721**, also a null result. The provider/controller class is
closed for this model. Repeating the latent screen with all 64 training
sequences reaches **0.1707001**, still far above the gate; the full-corpus
rank-256 arm remains best at **0.1692925**. A larger nonlinear rank-256
residual reaches **0.1666128**, but still fails the gate; report:
`reports/controller_provider_pca_2026-08-03/nonlinear_rank256_h256_full.json`.
The current provider-only capacity family is closed; the next justified
attempt requires a materially different model or an independent causal corpus.

```bash
PYTHONPATH=src python -m engram.cli distill-causal-attention-provider \
  --provider work/provider_train8_rank64 \
  --controller work/controller_distillation/bitnet_rank128_operator_residual_1024x256_exact/controller \
  --trace work/controller_distillation/bitnet_train_8x16_b2 \
  --validation-trace work/controller_distillation/bitnet_validation_16x16_b4 \
  --out work/controller_provider_causal_attention \
  --steps 100 --key-dim 32 --value-dim 64 --query-width 64 --device cpu

PYTHONPATH=src python -m engram.cli evaluate-controller-stateful-provider \
  --trace work/controller_distillation/bitnet_validation_16x16_b4 \
  --provider work/controller_provider_causal_attention \
  --controller work/controller_distillation/bitnet_rank128_operator_residual_1024x256_exact/controller \
  --out reports/controller_provider_causal_attention/evaluation.json
```

The evaluator checks source-model identity, rejects training/validation
dataset reuse, advances provider context exactly once per token, and reports
zero decoder-layer calls. A passing replay or fixture is not a learned-model
promotion; only an independent causal result below **0.0225** can authorize
the nonzero provider in a package.

### Milestone 2 ledger

### Current Milestone 3 attention boundary (2026-08-03)

The sustained bounded-attention failure has now been resolved at the
evaluator level without giving up CPU-only execution. The native runtime keeps
the complete 128-position local context (W128), but stores each local key and
value vector as symmetric INT8 with one FP32 scale. On the frozen eight-record,
1,024-position causal replay, this reaches KL **0.00417982**, top-1 agreement
**0.974609**, target-NLL delta **+0.000391**, and hidden-state relative L2
**0.048147**. Every position band passes, including offsets 96–127 (KL
**0.00301720**, top-1 **0.964844**, hidden L2 **0.043127**). Actual native
attention reads are **4,328,521,728 bytes**, exactly **25%** of the dense
reference, and the run uses no Transformers model shell or GPU.

This is not a protected Milestone 2 replay, and W16 remains the ordinary
package default. The quality gate was first evaluator-only and is now also
available as an authenticated W128/INT8 opt-in package; broader corpora and
end-to-end performance tuning remain. The frozen protocol is
`work/olmoe_q7/local_int8_w128_2026-08-03_protocol.json` (SHA-256
`953b83cead9e722e6228c5d79252ecaa0c0c8980343459c62f5567455d33bda7`) and the
authenticated result is
`work/olmoe_q7/local_int8_w128_2026-08-03_report.json` (SHA-256
`7cd55514efab021b2109f835310eb14aff64664e972fa6458266f20d1d17df80`).

The explicit compiler path is now exercised end to end as
`work/olmoe_q7/package_w128_int8_2026-08-03` (manifest SHA-256
`0b370a5b47913cade8255056b835ecf6d55f3a5aaf183290808ad22aa3ab6e8f`).  On an
identical eight-token prompt it produces the same token IDs as the W16/FP32
package, reduces attention reads by 75% (31,457,280 → 7,864,320 bytes), and
changes elapsed time by +2.21% in this short CPU run.  This is an authenticated
opt-in package boundary; W16/FP32 remains the ordinary default.  The complete
comparison is recorded in
`reports/olmoe_q7/package_w128_int8_generation_2026-08-03.json`.

The native Python package frontend can exercise this path explicitly without
changing the authenticated package manifest:

```python
from engram.runtime.olmoe_native import OLMoENativePackageRuntime
from engram.utils import sha256_file

with OLMoENativePackageRuntime(
    "work/olmoe_q7/package",
    manifest_sha256=sha256_file("work/olmoe_q7/package/manifest.json"),
    library="build/libengram_olmoe_token_runtime.so",
    threads=12,
    local_window=128,
    local_int8=True,
) as runtime:
    result = runtime.runtime.forward([token_id])
```

This opt-in is intentionally separate from the production manifest policy;
the default package still uses W16 FP32 attention storage.

For a separately authenticated research package, the compiler records the
same choice in its manifest:

```python
compile_olmoe_native_package(
    model, q7_artifact, non_mlp_safetensors, output,
    attention_local_window=128,
    attention_storage="int8",
)
```

The validator accepts this mode only as the explicit W128/INT8 pair; all
other packages retain the W16/C8/K4/S2 FP32 contract.

The first package benchmark shows why this remains opt-in: 128 teacher-forced
context tokens plus eight native steps took **75.17 s** with W16/FP32 and
**82.06 s** with W128/INT8. Counted attention reads fell from **45.83 GB** to
**28.11 GB**, but scalar INT8 dequantization made the complete run **9.17%
slower**. SIMD/fused attention kernels are required before claiming a latency
advantage over a tuned CPU implementation. The native kernel library now also
contains a runtime-dispatched AVX2 INT8 dot kernel plus the portable scalar
fallback. This host reports `avx2_available=false`, so AVX2 performance remains
unvalidated until an AVX2-equipped CPU is used.

### Native recurrent-controller boundary (2026-08-03)

The native token runtime now executes the schema-v3 factorized recurrent
correction directly. It loads the shared `input_down` and `recurrent_down`
projections, `gate_up`/`bias`, per-stage embeddings and low-rank adapters,
checks their float32 shapes, and applies the same operator-residual plus
per-token RMS transition as the NumPy controller reference. Optional
stage-specific input adapters are supported too.

The authenticated package deliberately remains on the exact residual path:
its `step_scale` values are zero and the normal runtime rejects nonzero
corrections. This is a security and scientific boundary. A nonzero learned
controller must first pass causal quality and held-out generalization before
it can replace the exact operator additions or support the layer-free
Milestone 4 claim.

For implementation testing only, the native token CLI exposes an explicit
unauthenticated evaluator override:

```bash
./build/engram-bitnet-token-generate \
  work/native_bitnet/model.engram-bitnet-dip 1 12 \
  --enable-recurrent-correction \
  --controller-directory work/controller_distillation/<controller>/controller \
  TOKEN_ID [TOKEN_ID ...]
```

It prints a warning because the override directory is not covered by the
package manifest. Native tests prove zero-correction parity, a known nonzero
transition, and fail-closed rejection of incomplete tensors. An existing
trained controller completed a short six-position CPU smoke run through this
path; that is execution evidence only, not a semantic gate or production
deployment.

The latest native parity check closes an important implementation detail: when
the authenticated controller has `step_scale == 0`, evaluator-controller mode
delegates to the exact residual implementation. On the frozen eight-token
package run, exact and evaluator modes produce identical token IDs, selected
records, semantic traffic counters, and cache positions. The sealed result is
`reports/native_bitnet_controller_zero_step_parity_2026-08-04.json`. This is
systems parity only; nonzero learned corrections still fail the causal
promotion threshold.

The Python `ControllerDrivenBitNet` path also uses this ABI for nonzero
corrections. An eight-prompt, one-token development screen reached **0.0%
token agreement** and **0/8 exact prompts** against the exact residual package;
cache positions and zero decoder-layer calls passed. Controller-stage work
averaged 12.50 seconds per prompt, and the first exact token `12366` became
`36306`. This confirms native dispatch while exposing a decisive quality
failure; it is intentionally not a gate result. The preserved report is
`reports/controller_native_recurrent_2026-08-03/development_8x1.json`.

Milestone 2 now has three source-track outcomes that should not be conflated:

| Deliverable | Native-BitNet | OLMoE Q7 | Generic dense Llama |
|---|---|---|---|
| Background/residual operators | Exact packaged residual; learned correction is zero | Native top-8 mixture needs no fitted residual in the passing simulation | Experimental fitted background worsened held-out error |
| Semantic key/value package | Complete ternary records plus authenticated DIP-v2 index | **Complete immutable 5.84 GB packed-Q7 expert/router artifact** | Quantized research package exists; no qualifying artifact |
| Practical routing | **Passed** in the native CPU kernel | **Passed** using the learned top-8 router in the direct packed CPU kernel | **Blocked** by quality/traffic tradeoffs |
| Quantization | Native packed ternary representation | **Canonical Q7/group-64 plus executed BF16 scales** | Product/additive codecs implemented experimentally |
| Python semantic-memory runtime | Tokenizer/chat drives a persistent native DIP handle | **Persistent complete native OLMoE token runtime implemented** | Implemented for research packages |
| End-to-end substituted-MLP evaluation | Complete through native token generation and chat | **Formal frozen complete-native 8×32 causal confirmation and package generation pass** | Evaluation path exists; no qualifying compiled candidate |

Therefore the separately trained **native-BitNet and OLMoE Q7 Milestone 2
paths are operational and may advance**. Engram still cannot claim that it
converts an arbitrary dense Llama checkpoint into a gate-passing
semantic-memory model.

The learned-provider boundary has also been screened against a pinned public
WikiText-2 raw corpus (`Salesforce/wikitext`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`). Sixteen-record train and
validation subsets were captured in authenticated chunks. Separate-stream
rank-16/rank-128 providers reached terminal normalized MSE `0.184007` and
`0.186352`; a rank-128 normalized-residual target reached `0.179788`, all
versus the fixed `0.0225` gate. Ordinary prose expansion and bounded target
changes therefore remain closed as solutions. See
`reports/controller_provider_pca_2026-08-03/auxiliary_wikitext2_screen.json`.

The authenticated native CPU package is multithreaded in attention: a
controlled eight-token sweep reached 1.47x wall and 2.32x attention speedup at
12 threads over one thread, while semantic time remained flat. This points to
semantic coordinate/record parallelism as the next systems optimization, not
to a missing thread-pool configuration. The full sweep is in
`reports/native_bitnet_cpu_generation_2026-08-04_long.json`.

The earlier Milestone 3 selector screens remain useful negative controls: a
rank-4 query-content selector recovered only **25.42%** of the same-state
residual and a phase-conditioned mass selector **26.19%**. The newer INT8
full-context cache above addresses the actual failure mode—older-context
eviction—while retaining a substantially lower memory-traffic budget.

### OLMoE source-track experiment

The current controlled source-family experiment is
[`allenai/OLMoE-1B-7B-0125`](https://huggingface.co/allenai/OLMoE-1B-7B-0125),
pinned at revision `9b0c1aa87e34a20052389dce1f0cf01da783f654`.
Unlike dense Llama or dense Qwen, each OLMoE layer already contains a learned
64-way router and 64 separately stored SwiGLU experts, of which eight are
selected per token. This gives Engram a trained, natively addressable semantic
substrate instead of asking a router to recover useful records from one
monolithic dense MLP after training. The topology alone did not solve
Milestone 2; the compiled causal evidence below is what now closes the
OLMoE-specific gate.

The new fail-closed audit reads the config and weight index, then optionally
uses bounded HTTP range requests to read only the six safetensors headers. It
never accepts a full-shard response during that shape audit. On the pinned
checkpoint, all **3,219/3,219** names and shapes match the Engram OLMoE
contract. Selected expert Q4 plus BF16 router matrices project to **12.6302%**
of an all-expert dense-Q4 MLP baseline. That is a structural screen only; it
excludes attention, cache-line amplification, runtime overhead, and causal
quality.

```bash
PYTHONPATH=src python -m engram.cli audit-olmoe \
  --model allenai/OLMoE-1B-7B-0125 \
  --revision 9b0c1aa87e34a20052389dce1f0cf01da783f654 \
  --verify-remote-shapes \
  --out reports/olmoe_source_audit_2026-07-27/audit.json
```

Engram now also has an exact NumPy decomposition of OLMoE routing and weighted
expert contributions, trained router traces, and an all-layer quantization
intervention. The frozen Q7/group-64 confirmation on 8 sequences and 256
positions passed the semantic thresholds: KL **0.00900774**, top-1 agreement
**0.9765625**, NLL delta **+0.00391912**, and final-hidden relative L2
**0.0460273**. Selected packed Q7 experts, BF16 group scales, and BF16 routers
project to **22.7865%** of the all-expert ideal-Q4 baseline.

The systems follow-up now serializes all 16 layers and 1,024 experts into a
strictly validated **5,842,733,184-byte** artifact. Codes use canonical biased,
LSB-first seven-bit packing; scales and routers are BF16; every phase and
expert is cache-line aligned and directly addressable. A CPU-only mmap kernel
computes the learned router and executes only the selected top-eight experts
without constructing dense matrices or a Transformers model.

On the production artifact, the native route exactly matches the independent
decoded reference. Output relative L2 is **1.94718e-6**, maximum absolute error
is **1.63913e-7**, and one layer/state schedules **45,875,200 bytes**, or
**22.7865%** of all-expert ideal Q4. The
[native systems report](reports/olmoe_q7_native_systems_2026-07-27/summary.md)
passes. This closes the remaining OLMoE Q7 native systems gate.

The next integration boundary now passes too. A separate 949,242,368-byte BF16
artifact maps embeddings, all attention projections and norms, the final norm,
and the independent language head. The CPU runtime combines it with Q7 experts
and performs a complete token step with RMS normalization, Q/K normalization,
RoPE/cache advancement, bounded attention, residuals, and vocabulary argmax,
without constructing Transformers. On `The capital of France is`, it predicts
` Paris`. See the
[native token-boundary report](reports/olmoe_q7_native_token_boundary_2026-07-27/summary.md).

That pair is now assembled into an authenticated, CPU-only generation package.
Its manifest covers the exact seven-file inventory, fixes the attention and
MLP policies, and has external authentication root
`861e9cc472f9e1245db5d64e9253411d0b656a0f08df2f58264e9c708ed750db`.
Package loading rejects a changed manifest, changed file, extra file, or
symlink. The package-only runtime loads its own config and tokenizer and
reproduces token `7785` (` Paris`) without a Transformers model shell.

The single-row Q7 kernel now parallelizes the eight selected experts. On
production layers 0, 7, and 15, canonical eight-code/seven-byte block decoding
reduces medians from **108.49/106.24/117.09 ms** to
**16.53/12.55/12.67 ms** (**6.56×–9.24×**) with bit-identical routes and
outputs. The complete five-position prompt falls from 13.33 to **2.17
seconds** of native execution, including Q7 time falling from 13.08 to
**1.91 seconds**. Parallel structural validation and inventory hashing reduce
cold wall time from 61.78 to **32.06 seconds**.

Teacher/reference capture is also parallelized safely. Four concurrent
sequence forwards share one read-only model and reproduce the serial teacher
arrays byte-for-byte, while reducing the 8×33 BF16 teacher pass from 366.14
to **94.78 seconds** (**3.86×**). Direct expert threading is faster than
serial but changes BF16 rounding, so it remains opt-in rather than redefining
the sealed reference.

The frozen eight-prompt package-generation protocol also passes: all **60/60**
teacher-forced top-1 decisions, **29/32** greedy reference tokens, and **7/8**
complete four-token prompts agree with the untouched BF16 teacher. Those
sequences remain inside W=16, so this is an integration result rather than
older-context evidence. See the
[generation and performance report](reports/olmoe_q7_native_generation_2026-07-28/summary.md).

The formal frozen causal protocol crosses that boundary. The complete CPU-only
package passes **8×32**—eight sequences and 256 prediction positions—with
overall KL **0.012981**, top-1 agreement **0.960938**, NLL delta
**+0.016824**, and final-hidden relative L2 **0.062047**. Positions 16–31,
after the exact W=16 window begins evicting context, independently pass the
same thresholds with KL **0.010642**, top-1 **0.960938**, NLL **+0.013690**,
and hidden L2 **0.075202**. Scheduled Q7 reads are **22.7865%** of the
all-expert ideal-Q4 reference. This remains the formal OLMoE Milestone 2
qualification. See the
[complete native causal report](reports/olmoe_q7_native_causal_2026-07-28/summary.md).
A separately frozen, explicitly non-independent source-bound replay reproduces
every metric and check exactly, authenticates all post-run roots, and measures
**88.79 seconds** inside native execution, including **72.17 seconds** in Q7.

A stronger prospectively frozen **8×128** follow-up used eight newly authored
natural-prose records, 1,024 prediction positions, the same authenticated
package and Q7 policy, and W16/C8/K4/S2 bounded attention. Every evidence,
counter, reset, traffic, and post-run authentication check passed, but semantic
quality did not: overall KL was **0.1435776225**, top-1 agreement
**0.802734375**, NLL delta **+0.1592924107**, and hidden L2
**0.2382604508**. The 0–15 and 16–31 bands still passed; failure first appeared
at offsets 32–63 (KL **0.0838567379**, top-1 **0.828125**, NLL
**+0.0755772478**, hidden L2 **0.2185442635**) and worsened thereafter.

That failure did not reopen Milestone 2. A post-failure matched attribution
control changed only the local attention window from 16 to 128 while retaining
the exact package, Q7 artifact and policy, corpus, teacher arrays, native
library, thread count, and evaluator identities. W128 full causal attention
matched all 128 pre-intervention rows exactly and passed every position band
and evidence check. Overall KL was **0.00343811931**, top-1 agreement
**0.974609375**, NLL delta **+0.00145861260**, and hidden L2
**0.04138915755**. The result attributes the sustained failure to bounded
attention and vindicates the Q7 semantic substitution underlying the formal M2
pass; it motivated the compressed full-context INT8 experiment reported at
the top of this section.

The uncompressed W128 control is a diagnostic, not a deployable solution: it reads **100%** of dense
causal attention bytes (**2,164,260,864 bytes per sequence**) and holds
**35,825,664 bytes** of attention state. The prospectively frozen follow-up
therefore compared W16/C18/K16/S2, W24/C10/K8/S2, and W30/C4/K2/S2 at an
exactly matched **968,753,152 logical bytes per sequence** (**44.7614%**) and
32 visible values per mature step. Every arm passed its evidence,
authentication, exact-counter, reset-replay, and pre-eviction identity checks,
but **none passed semantic quality**:

| Policy | Mean KL | Top-1 | NLL delta | Hidden L2 | Decision |
|---|---:|---:|---:|---:|---|
| W16/C18/K16/S2 | 0.063887 | 0.867188 | +0.051701 | 0.157717 | No selection |
| W24/C10/K8/S2 | 0.065912 | 0.877930 | +0.058480 | 0.159755 | No selection |
| W30/C4/K2/S2 | 0.095813 | 0.840820 | +0.075728 | 0.188422 | No selection |

All three passed the 0–15 and 16–31 bands. Hidden-state drift appeared in
32–63, and the 64–95 and 96–127 bands failed broadly. The frozen rule forbids
promoting a “best failure,” so there is no selected arm and the reserved fresh
confirmation corpus remains unconsumed. The sweep used an explicit raw-runtime
intervention because the immutable installed package remains bound to
W16/C8/K4/S2; it consumed the designated sustained-development corpus but did
not silently rewrite or promote the package policy.

The next authenticated experiment tested the layer-adaptive part of that
boundary. A new native ABI accepts one attention policy per layer; its
all-base layered configuration matched the old scalar W16/C8/K4/S2 ABI
exactly for tokens, hidden states, logits, counters, and diagnostics. Starting
from that base, a prospectively frozen three-round greedy search evaluated
**45** causal candidates (**16 + 15 + 14**) on a deterministic two-sequence
selection split. It chose layers **11, 6, and 10** for full W128 rescue, then
evaluated the resulting schedule on the other six development sequences. The
schedule uses W16/C8/K4/S2 in 13 layers and W128/C8/K4/S2 in three layers. It
reads **955,957,248 logical attention bytes per sequence** (**44.1701489826%**
of dense), holds **11,865,728 bytes** of attention state, uses **6,528 bytes**
of scratch, and leaves Q7 traffic unchanged at **22.7864583333%**.

Every evidence, exact-resource, replay, old/new-ABI parity, and post-run
authentication check passed, but all four overall quality metrics failed on
the six-sequence internal screen: KL **0.10232094998**, top-1
**0.84505208333**, NLL delta **+0.11677564952**, and hidden L2
**0.20603686522**. The 0–15 and 16–31 bands passed every metric; all four
metrics failed in each of 32–63, 64–95, and 96–127. This was a
development-only use of the already consumed corpus, so the schedule was not
promoted and no fresh confirmation was run. The authenticated evaluator is
source commit `708782b`; the protocol SHA-256 is
`9514e90bd5d14ae01ea27185763e5a833d4f1963e6bffd0ec0c81848f35b0c3e`,
the result SHA-256 is
`97ce800bd855c1f16248cada696936c7c56acd49c02d6f1c9ce9885dc44f7c49`,
and the layered candidate DSO SHA-256 is
`fe4dfdcc7e87a3cd5e36074e07d297f838ba345c37e939eeb0d796cb39cce409`.

Global W/C/K reallocation in the tested policy family and this frozen greedy
three-layer W128 path are now closed. The negative greedy result does not rule
out every interacting whole-layer combination.

The prospectively frozen teacher-attention-mass follow-up is now complete.
Dense BF16 teacher attention maps from only the deterministic two-record
selection split ranked all 256 layer-head pairs by older-context attention
mass not covered by the largest four older weights. The frozen prefix gives
W128/C8/K4/S2 rescue to exactly **51 of 256 heads** and leaves every other
head at W16/C8/K4/S2. It reads **973,384,704 logical attention bytes per
sequence** (**44.975387218386625%** of dense); a 52-head prefix would exceed
the 45% cap. Q7 is unchanged at **93,952,409,600 scheduled bytes per
sequence**.

The new per-head native ABI passed exact all-base semantic parity, and every
frozen evidence, resource, reset-replay, and authentication check passed.
Quality still failed over the six reused internal-development records and
768 prediction positions:

| Head-wise screen | Result | Gate |
|---|---:|---:|
| Mean KL | 0.07371992968429097 | ≤ 0.05 |
| Teacher top-1 agreement | 0.8671875 | ≥ 0.90 |
| Target NLL delta | +0.05345554334600896 | ≤ 0.05 |
| Final-hidden relative L2 | 0.1675178178168911 | ≤ 0.10 |

The 0–15 and 16–31 bands passed; degradation resumed after position 32. This
is materially better than the three-layer rescue at slightly higher traffic
(KL 0.10232095, top-1 0.845052, NLL +0.11677565, hidden L2 0.20603687 at
44.170% reads), but it remains outside all four overall gates. No fresh
confirmation was run and no package policy was promoted.

This result closes only the tested **fixed teacher-attention-mass ranking**,
not all head-wise allocation. It motivated the causal/value-sensitive static
follow-up below. Milestone 2 remains passed; Milestone 3 attention remains
blocked. See the
[sustained-context evidence and attribution report](reports/olmoe_q7_sustained_context_2026-07-28/summary.md).

The prospectively frozen causal/value-sensitive follow-up is also complete.
Its two-record fit used the exact native W16/C8/K4/S2 decisions with a
differentiable gathered surrogate, two iterative-hard-thresholding steps, and
an invariant budget of exactly 51 rescued heads. The selected M1 mask improved
the maximum training composite objective from **7.867116928100586** to
**4.755991458892822** and the mean from **6.917216062545776** to
**4.328476905822754**; both selection records improved. Training evidence
passed after **6,930.099 seconds**. That CPU fit was the experiment's dominant
bottleneck. A deterministic frozen-expert backward proxy now preserves the
installed `grouped_mm` CPU forward exactly while replaying independent expert
backwards on 12 workers. On the authenticated M0/sequence-0 full-record
qualification it matched the loss, all 256 gate gradients, native diagnostics,
and projected 51-head mask bit for bit after removing timing fields. Record
time fell from **1,564.347 seconds** to **809.168 seconds** (**1.933×**,
**48.274% less wall time**), so
the proxy is authorized for larger development fits. It changes offline
training performance only; the sealed native Q7 reference and failed semantic
result are unchanged. See the
[expert-proxy qualification](reports/olmoe_q7_expert_proxy_2026-07-28/summary.md).
This was one previously consumed record measured in separate executions, not
a controlled repeated benchmark or a claim that the complete 6,930-second fit
will obtain the same speedup.

The sealed chain is rooted at source commit `483c62f`:

- training protocol:
  `037ebfd7d4e40af898ece7f353654eb8a41dc1883f191cbdf05fc34bf50bf4ba`
- training result:
  `bacb0e31899f514a8b2b517987566e8bca68d39cabfd50b3c9e7ecf83bc756ea`
- native-screen protocol:
  `282bfe0b9e1da86577f0187112a4a444b0f36d7f84e10f4f9bb67730676807c2`
- native-screen result:
  `437d0de4ce4da37e69ca13279b76627d6f7721e766b8f1b4371fb318e7cbeb59`

The complete packaged-Q7 screen then reused the same six consumed development
records and 768 prediction positions. Its evidence and resource gates passed,
including the **44.9753872%** logical-read budget, but quality did not:

| Causal/value-sensitive 51-head screen | Result | Required | Outcome |
|---|---:|---:|---|
| Mean KL | 0.07913208059 | ≤ 0.05 | **Fail** |
| Teacher top-1 agreement | 0.8645833333 | ≥ 0.90 | **Fail** |
| Target NLL delta | +0.08119899696 | ≤ 0.05 | **Fail** |
| Final-hidden relative L2 | 0.18264718059 | ≤ 0.10 | **Fail** |

The 0–15 and 16–31 bands passed. The 32–63 band failed top-1 agreement
and hidden-state error, and every later band failed all four quality measures.
This mask is worse than the earlier attention-mass mask on every overall
metric. No fresh confirmation was opened and no policy was promoted.

The tested two-record natural-prose causal/value-sensitive selector is
therefore closed, not merely awaiting more CPU fitting. This does **not** close
every static selector: the fit never trained specifically on retrieval
failures.

### Retrieval-targeted 51-head selector

A separate retrieval-targeted selector and fail-closed protocol are now
implemented. They use a new synthetic passkey corpus with **8 training, 8
development, and 8 sealed-confirmation records**. Each record has 129 tokens:
128 causal prediction rows and 32 ground-truth answer targets scored only at
logit rows **96–127**. Four eight-token passkeys are placed at four balanced
source depths. Across all 24 records, the corpus uses **768 globally unique
numeric singleton tokenizer tokens**, so no passkey token leaks between
splits.

Training minimizes answer-only ground-truth cross-entropy. The loss value
comes from the complete packaged native Q7 forward path; gradients cross a
straight-through boundary into a frozen BF16 shell with exact native
attention, while the qualified frozen-expert backward proxy uses 12 workers.
No teacher parameter is updated. Two iterative-hard-thresholding steps produce
`M0 → M1 → M2`, projecting each learned candidate to exactly **51 of 256
layer-head pairs**.

That is the largest admissible mask: it reads **973,384,704 logical attention
bytes per sequence**, or **44.9753872184%** of full causal attention. A
52-head mask would read **45.2437999637%** and is rejected. If training
qualifies a mask, development must still pass both a full-W128 packaged-Q7
control and the 51-head candidate, overall and separately at each of the four
source depths.

The protocol is frozen at
`work/olmoe_q7/retrieval_selector_2026-07-29_frozen/protocol.json`, SHA-256
`f6961ec6bffadb306a75ae8aaa8c400d1282cbb096876532d75e22a295660580`.
The full 8-record fit and 8-record development screen completed. Training
selected `M2`: mean answer cross-entropy fell from 7.647114 at `M0` to
1.005444, with no record regression. The teacher retrieval check and full-W128
packaged-Q7 control both passed. The exact-51 candidate still failed the
semantic gate: KL was 0.186610, target-NLL delta 0.283658, and hidden relative
L2 0.335103, although top-1 agreement reached 0.929688. Every resource,
reset/replay, and post-run authentication check passed. Confirmation remained
unopened and was not authorized.

The complete [result and evidence summary](reports/olmoe_q7_retrieval_selector_2026-07-29/summary.md)
are archived with the source-bound protocol. A SHA-authenticated training
checkpoint was durably written before development, so future diagnostic
reruns can skip all 16 expensive surrogate backwards. This closes the static
retrieval-targeted mask at the declared budget. Milestone 2 remains passed and
Milestone 3 remains blocked.

```bash
PYTHONPATH=src python -m engram.evaluation.olmoe_retrieval_head_selector \
  fit-screen \
  --protocol work/olmoe_q7/retrieval_selector_2026-07-29_frozen/protocol.json \
  --protocol-sha256 f6961ec6bffadb306a75ae8aaa8c400d1282cbb096876532d75e22a295660580 \
  --out work/olmoe_q7/retrieval_selector_2026-07-29_frozen/development_result.json
```

An interrupted development-only rerun can explicitly authenticate and reuse
the completed training state with
`--resume-training-checkpoint <path>` and
`--resume-training-checkpoint-sha256 <sha256>`. The checkpoint is never
selected implicitly.

The experiment follows the retrieval-specific identification result in
[DuoAttention](https://arxiv.org/abs/2410.10819) instead of treating ordinary
language-model loss as sufficient supervision. The subsequent conditioned and
episodic experiments retain the causality and task-aware allocation principles
described by [Ada-KV](https://arxiv.org/abs/2407.11550) and
[Task-KV](https://arxiv.org/abs/2501.15113).

### Episodic retrieval and head-gated follow-ups

The next experiments asked a narrower question: is the retrieval failure
caused by choosing the wrong static heads, or does the bounded attention path
lack the right information at answer time? These are **train-only diagnostic
screens** over the same eight retrieval-training records. They do not consume
development or confirmation evidence. The full-context `M2` training result,
mean answer cross-entropy **1.005444** and worst **1.227907**, remains their
strict reference.

| Diagnostic | Mean answer CE | Worst answer CE | Result |
|---|---:|---:|---|
| Two causal prefix prototypes, 51 heads each | 1.046825 | 1.224952 | Fail: mean regressed and 5/8 records regressed |
| Exact payload-only episodic oracle, all heads | 1.224460 | 1.327343 | Fail: 7/8 records regressed |
| Exact label-plus-payload episodic oracle, all heads | 1.231254 | 1.321619 | Fail: 7/8 records regressed |
| Exact payload oracle gated to the `M2` K51 heads | 1.400569 | 1.694034 | Fail: 7/8 records regressed |

The two-prototype allocator used only causally observed fact order, but its
mean loss was 0.041381 worse than global `M2`; result SHA-256
`dacb3f37886d1207bc6b9a5717b3015174c4edc4947b89dd12ef35ff67ae8814`.
The payload-only oracle then removed selector error completely by writing each
known eight-token source payload and reading the correct span at its answer
rows. It still failed, with the damage concentrated at the first row of each
answer block. See the
[payload-oracle evidence](reports/olmoe_q7_retrieval_episodic_oracle_2026-07-29/summary.md).

Adding the immediately preceding label token produced a 36-slot, four-span
label-plus-payload cache and did not repair the failure. Mean answer CE was
1.231254, 0.225811 worse than `M2`, and worst CE was 1.321619. Every resource,
counter, replay, and authentication check passed. Its upper-bound traffic was
719,585,280 bytes, **33.2485%** of dense full-context K/V traffic, with
11,059,712 bytes of state and 4,992 bytes of scratch. Protocol SHA-256 is
`1812a6ba72afe0c5f32e459867c29f3d8dbd609a3d0ddf59ac52ae6859ce4d3d`;
result SHA-256 is
`e1ec5a2bde8b9ce7198fe1571a7670c45a3bc7a712cdf9a856f869b6429fe69d`.

The native runtime now also exposes the versioned
`engram_olmoe_token_open_episodic_headwise_v1` ABI. Inactive layers use the
exact legacy attention step and allocate no episodic bank. Active layers store
the complete causal K/V rows, while only enabled query heads deduplicate,
score, normalize against, and read an episodic span. All-zero, malformed, and
policy-less masks fail closed; an all-ones mask is exactly equivalent to the
legacy all-head episodic path.

That ABI enabled an exact K51 attribution screen using the frozen `M2` mask
(`49802a2d37abd44e4015e87633c9a321e333315b9400f6a69d4713ec2270b446`).
It passed every systems check at 687,472,640 total traffic bytes
(**31.7648%**) and 10,010,112 state bytes, but semantic loss became worse:
mean CE 1.400569 and worst CE 1.694034, with only one of eight records
improving. The authenticated
[K51 result](reports/olmoe_q7_retrieval_episodic_head_mask_2026-07-29/summary.md)
therefore rejects transferring the old 51-head cardinality directly to the
episodic cache; it does not reject episodic memory at larger head budgets.

The prospectively frozen **K64 → K96 → K128 → K165** ranked-prefix train
screen has now completed. All four candidates executed all eight records,
passed their systems contracts, and failed the strict
mean/worst/no-regression gate:

| Ranked payload prefix | Mean answer CE | Worst answer CE |
|---|---:|---:|
| K64 | 1.379699 | 1.639418 |
| K96 | 1.328848 | 1.618843 |
| K128 | 1.337958 | 1.621764 |
| K165 | 1.331006 | 1.608617 |

Only one of eight records improved at every K. The frozen total-failure rule
retained K165 for diagnostic reset replay because it had the lowest worst
loss; replay passed, but it is not a promotion. The
[archived protocol and result](reports/olmoe_q7_retrieval_episodic_head_mask_2026-07-29/summary.md)
have SHA-256 roots
`e238b5cfc4359422abc96c227f4010ade747ff161c946b2edae14c171a7bf04c`
and
`a6fb045bbad526411b8318e7de2412ea56d69b3f2be0d977ca6819635c0718da`.
Confirmation remained unopened.

This closes larger prefixes under the transferred `M2` ordering, not episodic
memory. The stronger fixed base was the authenticated K256 all-head payload
result at 1.224460 mean, 1.327343 worst, and 33.0305% upper-bound traffic.
The subsequent V2 logit-mass screen held that cache and schedule fixed and
tested `gamma={1/2,1/4,3/16,1/8}` by adding `float32(log(gamma))` only to
episodic logits. All four candidates completed all eight training records,
passed their systems contracts, and failed:

| Fixed-K256 bias arm | Mean answer CE | Worst answer CE |
|---|---:|---:|
| Historical `beta=0` attribution | 1.224460 | 1.327343 |
| `gamma=1/2` | 1.461414 | 1.669250 |
| `gamma=1/4` | 1.883818 | 2.288258 |
| `gamma=3/16` | 2.161750 | 2.595642 |
| `gamma=1/8` | 2.725091 | 3.430532 |

The total-failure rule replayed `gamma=1/2` only because it was the best
failed nonzero candidate; replay passed exactly, but this is not a promotion
and `beta=0` remained materially better. The immutable
[V2 protocol, parity, result, and summary](reports/olmoe_q7_retrieval_episodic_logit_bias_2026-07-29/summary.md)
are rooted by result SHA-256
`19d08ce9eb4b673d423e9781a491e25ec03bdec09467a43e7be1881874ef2287`.
No development run was authorized, and the reserved confirmation split
remained unopened.

Shared scalar logit calibration is therefore closed. The subsequent
**same-state shadow residual capacity screen** fixed the stronger `beta=0`
K256 base, fed its exact post-RoPE Q/K/V into a non-intervening train-only
W128 shadow, and measured an optimistic leave-one-sequence-out ceiling for the
post-`W_o` residual. Per-layer output bases were learned from seven records;
the held-out coefficients were chosen by oracle projection.

| Residual rank | Global recovery | Minimum sequence | Minimum block entry | Positive layers |
|---:|---:|---:|---:|---:|
| 2 | 0.400470 | 0.315782 | 0.252050 | 16/16 |
| 4 | 0.428686 | 0.346947 | 0.325317 | 16/16 |
| 8 | 0.469253 | 0.387498 | 0.443967 | 16/16 |

Every rank passed the frozen sequence, block-entry, finite, and positive-layer
conditions, but every rank missed the required 50% global recovery. Result
SHA-256 is
`c636ad124d570f3675a36f0a23b276ba2e4cd4f5efc23dbf98cc10cd2cfd8e33`;
the compact trace-manifest SHA-256 is
`1f255a59a20089abe4d6805c625a119c167b71153bc21f5edbfcf0fd8050f461`.
Replay and all post-run authentication checks passed, and confirmation
remained unopened. See the
[archived capacity evidence](reports/olmoe_q7_retrieval_episodic_residual_capacity_2026-07-29/summary.md).

This closes only rank-at-most-8 global per-layer output subspaces with oracle
held-out coefficients. No causal coefficient predictor, correction artifact,
or package change is authorized.

### Per-head mass and joint output-targeted gamma oracles

The next cached capacity experiment selected one of
`gamma={0,1/8,1/4,1/2,1,2,4,8}` independently at every
record/read-row/layer/head coordinate to match the W128 teacher's probability
mass on the eight scheduled source positions. It successfully reduced mean
mass error from 0.0445126662 to 0.0084754603 without a coordinate regression,
but the reconstructed post-`W_o` result moved in the wrong direction:
global recovery was **-0.10891245427020602**, all eight sequence recoveries
were negative, all four block-entry recoveries were negative, and only 1/16
layers had positive recovery. The
[archived head-mass evidence](reports/olmoe_q7_retrieval_episodic_head_mass_oracle_2026-07-29/summary.md)
has protocol/result SHA-256 roots
`fe09689452e6ae4f1a1b15332c61c1cc990cfc29b6a8b0d5a1758d9490a93af5`
and
`f7060e7373c5faf8f154891e93efad35659723d8e3f04d83638b62fa9cf72596`.
This closes exact scheduled-source-mass matching as an objective, not
output-targeted per-head reweighting.

The follow-up therefore optimized all 16 heads jointly against the exact
W128-minus-K256 post-output-projection residual, including cross-head
coupling through `W_o`. Its continuous box relaxation is an optimistic
superset of the eight-code family; its discrete arm directly replayed the
selected mixed codes through the established float32 counterfactual path.

| Joint-gamma result | Global recovery | Sequences ≥0.25 | Blocks ≥0.25 | Positive layers |
|---|---:|---:|---:|---:|
| Continuous optimistic relaxation | **0.22738059544921096** | 1/8 | 0/4 | 16/16 |
| Discrete direct float32 | **0.1997680396822742** | 0/8 | 0/4 | 16/16 |

Both arms failed the frozen 0.50 global, every-sequence, and every-block
requirements. The continuous relaxation's failure makes further search over
the contained scalar gamma grid unjustified even though the reported discrete
solution claims only one- and two-head local optimality. The
[archived joint-gamma evidence](reports/olmoe_q7_retrieval_episodic_joint_gamma_oracle_2026-07-30/summary.md)
is rooted by protocol SHA-256
`aa03a71e3dd9e1fbb413a7773d57189c41029c26b2f4372b2fb7a26744305d24`
and result SHA-256
`1329a51bac71cb81f44494c8ef70cb23a631eacebac5540b39e2e98ed5e30ea5`.
Replay and authentication passed, but no predictor, Milestone 3 promotion, or
confirmation access was authorized.

The follow-up then removed the scalar-mass restriction. A native trace exposed
all eight exact BF16-decoded episodic values already read by K256. A
constructible nine-way simplex per head combined those values with the
regular-cache conditional mean; an optimistic ten-way hull also included the
exact native head output. Both arms were optimized jointly after `W_o` with a
certified product-simplex solver.

| Per-slot value result | Global recovery | Sequences ≥0.25 | Blocks ≥0.25 | Positive layers |
|---|---:|---:|---:|---:|
| Constructible regular + 8 slots | **0.3844378107** | 8/8 | 4/4 | 16/16 |
| Optimistic exact-native-anchor hull | **0.3844378142** | 8/8 | 4/4 | 16/16 |

The only failed condition was the prospectively frozen 50% global threshold,
but it failed decisively: the optimistic hull's maximum per-row objective-gap
bound was `5.90e-11`, direct/quadratic parity and exact replay passed, and the
native anchor changed global recovery by only about `3.49e-9`. The immutable
[per-slot capacity evidence](reports/olmoe_q7_retrieval_episodic_slot_simplex_oracle_2026-07-30/summary.md)
is rooted by cached protocol SHA-256
`f3be957ec0c13d0f49c85a2fa149611307de756f2be82165098a43263bb78ce3`
and result SHA-256
`2e8e9b7d5f33d33c0e8c642a50359da785d0690d8260ed9f65837b14cd93a5bf`.
The solve used eight CPU workers, took 91.59 seconds, and never reopened the
native runtime or confirmation split.

The frozen **full-visible C28/C29 capacity oracle** then exposed the regular
path's 16 chronological local values and four selected-older values
separately, alongside the eight episodic values. C28 is fully constructible
from data the kernel already reads; C29 is an optimistic superset that adds
the exact native head output as an anchor.

| Full-visible result | Global recovery | Minimum sequence | Minimum block entry | Positive layers |
|---|---:|---:|---:|---:|
| Constructible C28 | **0.6653937751** | **0.6447006551** | **0.6306278392** | 16/16 |
| Optimistic C29 | **0.6653865288** | — | — | 16/16 |

C28 passes the frozen 0.50 global, every-sequence, every-block, and
positive-layer conditions. Qualification, deterministic replay, source and
artifact authentication, and post-solve checks all pass. Nested C10 and C16
diagnostics recover 0.5335805245 and 0.6021187653, but they have no
progression authority. The experiment retains the fixed 10,534,912-byte
attention state and 714,866,688 logical traffic bytes—**33.0305%** of dense—
and adds no KV reads.

This passes the train-only same-state value-capacity gate and authorizes a
causal 28-logit selector. It does **not** pass Milestone 3, demonstrate native
causal learnability, promote a package policy, authorize development, or open
confirmation. The sealed confirmation split remained unopened. The
[archived full-visible evidence](reports/olmoe_q7_retrieval_episodic_full_visible_simplex_oracle_2026-07-30/README.md)
is rooted by result SHA-256
`a8711f07fdcbe48b16ebe962aae1962e4613873640eba20bc7fa88238216aee1`.

Two prospectively frozen causal screens have now tested that authorization:

| Train-only selector | FP32 global | BF16 global | Minimum sequence | Minimum block | Positive layers | Logical traffic |
|---|---:|---:|---:|---:|---:|---:|
| Rank-4 query content plus mass | 0.2542615526 | **0.25422074198** | 0.23161600085 | 0.18371154473 | 16/16 | 36.8096% |
| Eight-phase table plus mass | **0.2618976463** | **0.2618728353** | 0.2405241062 | 0.2244750908 | 16/16 | 34.0116% |

The content arm reconstructs each packaged post-QNorm, pre-RoPE query and
combines a rank-4 content projection with inference-available source masses.
Its protocol/result SHA-256 roots are
`0a58ba3a59d2f0f816046ca28aac304baf7663ef890a6b298f0cc7277613d051`
and
`9ea504f83a487584cb9ae2127565674a8e341ca58f6777a03514b0c9a281995c`.
The phase arm instead adds an eight-step schedule-relative table to the
smaller learned mass selector. Its four BF16 block-entry recoveries were
0.314588398, 0.228395562, 0.261696236, and 0.224475091. It contains 82,944
parameters serialized in 165,888 BF16 bytes; total logical traffic is
736,100,352 bytes, or 34.0116% of dense. Its protocol/result SHA-256 roots are
`8cb1c7b0e9a6bc2d23839cdbf4de973e66616cccc86e980e6a151d4f2b773987`
and
`52360cf47cb2eeab52e595961f436e4c1e7b79db6cdaa339b7f699d3290883ed`.

All systems, authentication, deterministic replay, zero-model, masking, and
BF16 parity checks passed. Semantically, however, phase conditioning improved
the preceding mass-only BF16 recovery by only **0.0040699**, far short of the
0.50 global requirement, and both learned classes are closed. These are
train-only model-selection outcomes on an already exposed corpus, not
independent generalization evidence. No development, confirmation, native
integration, or package promotion was authorized. The next justified model
class is a directional blockwise-QK feature controller that can represent
query-to-key compatibility rather than only source mass, query content, or
schedule phase. Milestone 2 remains passed; Milestone 3 remains blocked.

The blockwise-QK feature boundary is now implemented as an evaluator-only
trace. On the eight-record train capture, its `[8, 32, 16, 16, 28, 8]`
partial tensor reconstructs native attention masses with max error
`1.90735e-6` and recovers their ordering on 65,529/65,536 rows. A score-ranked
top-20 subset retains 96.26% mean mass but only 91.20% at p10, so this is a
validated compatibility feature—not yet a cheaper causal attention policy.
The reusable train-only audit is
`engram.evaluation.olmoe_retrieval_episodic_blockwise_qk`; the confirmation
split remains sealed.

The next locality boundary is also implemented: the native shadow can now
copy all eight older-cache candidate Q/K scores before the native four-entry
top-K decision.  This is a separate evaluator-only ABI and manifest, so the
existing C28 visible-entry artifact remains unchanged.  The authenticated
candidate capture is
`work/olmoe_q7/retrieval_episodic_blockwise_qk_candidates_2026-07-31`, with
tensor shape `[8, 32, 16, 16, 8, 8]` (records, reads, layers, heads,
candidates, bands).  On train-only data, score-ranked candidate top-4 retains
92.27% of candidate softmax mass (p10 77.85%); the selected older-entry scores
fall in the candidate top four 95.74% of the time.  These are locality
features, not native slot-membership or causal recall proof.  The next
experiment is a learned candidate/group selector on a fresh development split,
followed by exact reranking and traffic accounting; Milestone 3 remains
blocked.

A cheap record-held-out negative-control fit is also recorded in
`query_only_router_screen.json`.  A rank-16 ridge map from pre-attention
hidden head slices to the eight candidate scores reaches only 68.93%
candidate-membership recall (p10 50.0%; exact top-4 13.52%).  Query state alone
is therefore not a defensible pre-read selector; the next model must add
stable key/group side information or a learned residual key summary.

Python owns packaged tokenization and prompt text handling. From token IDs
through recurrent state, Q7 routing/expert execution, final logits, and
argmax, this confirmation uses the native runtime without a Transformers
model shell. This OLMoE Milestone 2 result substitutes the MLPs at a native
token boundary but still executes the source model's embeddings, norms,
Q/K/V/O attention projections, and `lm_head`; it is not the Milestone 4
controller-only architecture with the original transformer operators removed.
Remaining OLMoE work is now bounded-attention repair at the Milestone 3
boundary, followed by broader generation quality, chat UX, whole-system
hardware-counter traffic, and lower authentication latency. The earlier
[authenticated package report](reports/olmoe_q7_native_package_2026-07-27/summary.md)
remains the package-integrity boundary.

```bash
PYTHONPATH=src python -m engram.cli repack-olmoe-q7 \
  --model work/huggingface/models--allenai--OLMoE-1B-7B-0125/snapshots/9b0c1aa87e34a20052389dce1f0cf01da783f654 \
  --out work/olmoe_q7/model.engram-olmoe-q7 \
  --group-size 64 --report work/olmoe_q7/repack.json

PYTHONPATH=src python -m engram.cli evaluate-native-olmoe-q7 \
  --artifact work/olmoe_q7/model.engram-olmoe-q7 \
  --library build/libengram_olmoe_q7.so \
  --out reports/olmoe_q7_native_systems_2026-07-27/result.json

PYTHONPATH=src python -m engram.cli repack-olmoe-non-mlp \
  --model work/huggingface/models--allenai--OLMoE-1B-7B-0125/snapshots/9b0c1aa87e34a20052389dce1f0cf01da783f654 \
  --out work/olmoe_q7/non_mlp.safetensors

PYTHONPATH=src python -m engram.cli run-native-olmoe-token \
  --config work/huggingface/models--allenai--OLMoE-1B-7B-0125/snapshots/9b0c1aa87e34a20052389dce1f0cf01da783f654/config.json \
  --non-mlp work/olmoe_q7/non_mlp.safetensors \
  --q7-artifact work/olmoe_q7/model.engram-olmoe-q7 \
  --library build/libengram_olmoe_token_runtime.so \
  --prompt "The capital of France is" \
  --tokenizer work/huggingface/models--allenai--OLMoE-1B-7B-0125/snapshots/9b0c1aa87e34a20052389dce1f0cf01da783f654 \
  --max-new-tokens 1 --threads 12

PYTHONPATH=src python -m engram.cli compile-native-olmoe \
  --model work/huggingface/models--allenai--OLMoE-1B-7B-0125/snapshots/9b0c1aa87e34a20052389dce1f0cf01da783f654 \
  --q7-artifact work/olmoe_q7/model.engram-olmoe-q7 \
  --non-mlp work/olmoe_q7/non_mlp.safetensors \
  --out work/olmoe_q7/package --threads 12 \
  --report work/olmoe_q7/package-report.json

PYTHONPATH=src python -m engram.cli generate-native-olmoe-package \
  --package work/olmoe_q7/package \
  --manifest-sha256 861e9cc472f9e1245db5d64e9253411d0b656a0f08df2f58264e9c708ed750db \
  --library build/libengram_olmoe_token_runtime.so \
  --prompt "The capital of France is" --max-new-tokens 1
```

CUDA is permitted for training and distillation only. Packaged inference,
including the passing BitNet MLP and attention kernels, remains CPU-only and
does not call llama.cpp.

The adjudicated DIP memory is now promoted into a real derived package and the
complete C++ token-step runtime. The installer authenticates the frozen policy,
adjudication, base record artifact, and v2 coordinate index, copies the
policy-bound package rather than modifying it, and records the DIP operator as
the package's only MLP mode. The runtime then executes each layer as native
attention → normalized semantic input → DIP → residual acceptance. It does not
construct a dense MLP backend and cannot fall back to one.

On the fixed non-holdout eight-prompt/32-token integration suite, the packaged
DIP runtime reproduced **32/32 greedy token IDs** and all **8/8** exact
four-token continuations. Global mean activity was **0.2156017260**, with a
maximum prompt mean of **0.2258916324**. Complete modeled cold traffic was
**30,153,074,432 bytes**, including **194,304 bytes** of global metadata;
the global mean fraction was **0.4116115605** of dense ideal Q4 and the
maximum prompt mean was **0.4129835480**. Position, stage/semantic-call,
semantic-row, backend, traffic-recomputation, and reset-replay checks passed
on CPU.

This is exact greedy token agreement, not hidden-state or logit parity. Reset
proves repeated tokens, zeroed counters, and structural metric parity—not
hidden-state identity. After adding the chat ABI, the rebuilt native core
repeated the same 32/32 and 8/8 result in **390.4183 seconds** including reset
replays and per-process authentication. The frozen suite still stops at 14
positions, below W=16.

A separate reproducible boundary protocol now runs the same authenticated
handle at 16, 17, 18, 24, and 32 prompt positions. At 32 it records 480 layer
evictions, 60,000 older-key scores, 34,800 older-value selections, 1,200 sink
insertions, and 5,654 accepted heavy-hitter updates while attention state stays
fixed at 7,477,440 bytes. Reset reproduces the token and all structural
counters. This passes bounded-attention mechanics, not long-context quality
against a dense teacher. Traffic is modeled rather than measured DRAM. See the
[native DIP attention report](reports/native_bitnet_dip_attention_confirmation_2026-07-27/summary.md).

The shared-controller path now passes its fixed transition gate. The decisive
change was architectural, not another corpus scale-up: a BitNet layer already
defines its next residual as the current state plus its attention and MLP
outputs, followed by normalization. Those operator outputs were present in the
trace, but the old controller unnecessarily compressed them through a learned
rank-128 bottleneck. A controlled rank-4 stage input adapter improved terminal
NMSE only from 0.159440 to 0.157431, confirming that this was not a capacity
problem.

Schema-v3 controllers preserve the known operator additions exactly and keep
the shared factorized recurrence only as an optional learned correction.
Across the unchanged 1,024/256-position split, the zero-correction CPU artifact
reaches protected terminal normalized MSE **0.000020801** against the
**0.0225** gate, passes Torch/NumPy reload parity within 5.72e-6, and executes
41,575.9 stage transitions/s without importing Torch or reading the correction
matrices. The trace provenance is stronger than initially reported: its
semantic outputs already come from the packaged direct CPU MLP kernel; only
attention was still dense.

The subsequent frozen compiled-operator replay also passes. On eight held-out
sequences and 256 prediction positions, packed semantic output plus native
W16/C8/K4/S2 attention replayed through the controller reaches KL 0.01113,
95.70% top-1 agreement, NLL delta -0.00829, and final-hidden relative L2
0.07589 against the dense-attention package baseline. Controller replay tracks
the compiled candidate at hidden L2 0.00681 and terminal trajectory NMSE
0.00002667. The next boundary is incremental generation driven directly by
controller state; the passing replay still executes decoder layers to obtain
operator outputs before independently replacing their residual scaffold.

Incremental controller-driven generation now passes as well. The explicit
runtime calls normalization, native attention, native MLP, and the controller
stage by stage without invoking `decoder_layer.forward`. It carries one scalar
RMS per token, advances absolute RoPE/cache positions, and preserves native
bounded-attention state across decode calls. On the fixed eight-prompt suite,
all 32 greedy tokens match the bounded decoder reference exactly, all cache
positions match, and decoder-layer calls remain zero. Controller arithmetic
averages 42.7 ms per prompt, about 0.19% of complete runtime.

The controller is now package-owned and native at its hot boundary. An
authenticated installer copies the schema-v3 tensors into `controller/`, adds
every file hash and controller contract to the native BitNet manifest, and
refuses incompatible or conflicting artifacts. The float32 residual/RMS step
now executes through `libengram_bitnet.so`; package-owned generation reproduces
` Paris. Paris is`, advances all positions, and still reports zero decoder
layer calls.

The surrounding decode shell has now moved substantially farther across the
native boundary. `libengram_bitnet.so` performs BF16 embedding lookup, all
RMSNorm operations, RoPE, exact residual/RMS advancement, and a threaded
tied-vocabulary argmax that does not materialize full logits. Together with
the existing packed MLP/projection and streaming-attention kernels, the
controller command invokes no decoder-layer forwards. A frozen eight-prompt,
32-token confirmation passes its progression gate at 96.875% token agreement,
87.5% exact prompts, and exact cache positions. One BF16 near-tie differs from
PyTorch/oneDNN because scalar native and library GEMM accumulation orders are
not identical. Python/Torch still orchestrates stage dispatch and tensor
views in that older path. The DIP-only token runtime described below has now
crossed the C++ package-runtime boundary.

The model core can also generate greedily without constructing a Python,
Torch, or Transformers model shell. The native BitNet token CLI is now
fail-closed on the DIP package: it accepts already-tokenized IDs, maps the
authenticated base artifact and v2 coordinate index, owns all 30 attention
caches, executes the DIP-only C++ token runtime, and prints generated IDs:

```bash
./build-runtime/engram-bitnet-token-generate \
  work/native_bitnet/model.engram-bitnet-dip 4 12 \
  128000 791 6864 315 9822 374
```

Add `--verify-reset` before the prompt IDs to repeat generation after clearing
all native caches and require identical output plus zeroed-counter and
structural-metric replay. Before any model mapping, the executable authenticates
the exact manifest and symlink-free file inventory against compiled deployment
trust roots. It derives architecture, paths, attention policy, context and
vocabulary bounds, RoPE/RMS settings, and EOS IDs—including `128009`—from the
authenticated package. The executable links the kernels directly and does not
load an Engram shared library. It reports semantic calls, rows, selected
records, kernel and global-metadata traffic, and semantic/attention time. Text
tokenization and chat-template handling remain outside this C++ command.

Derive, build, and evaluate the DIP package with:

```bash
PYTHONPATH=src python -m engram.cli install-native-bitnet-semantic-memory \
  --model work/native_bitnet/model.engram-bitnet \
  --index work/native_bitnet/model.provisional.bitnet-dip-index.bin \
  --policy reports/native_bitnet_m2_2026-07-26/frozen_dip_policy.json \
  --adjudication reports/native_bitnet_m2_final_audit/49df50cc01c96844ab3e7015d66c8899025dad4e1f7f01a450f97677751b36f2/0fdce5a3eb2eaaec5e3b47587d7693bfdc0fed6fde53e3a40a74b8d2a76d4aa3.adjudication.json \
  --out work/native_bitnet/model.engram-bitnet-dip \
  --index-sha256 b98ce4e46c8ae67d9d92d4d13f5de3d4fe45ef2c76400bd9d50be08b2bd60e15 \
  --policy-sha256 c572754e597a760bc5ea6ba337bdaaf092e4ae1d5b5e90b6a2a14cbfbea3768e \
  --adjudication-sha256 ebb5ca9568387ffd3c5b187f8e17f3ce706aaee86f4bbe9e314bf1760a7da5cc

cmake -S . -B build-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build-runtime \
  --target engram-bitnet-token-generate engram_bitnet_token_runtime -j

PYTHONPATH=src python -m engram.cli \
  evaluate-native-bitnet-dip-token-generation \
  --model work/native_bitnet/model.engram-bitnet-dip \
  --executable build-runtime/engram-bitnet-token-generate \
  --prompts tests/fixtures/inference_prompts.jsonl \
  --reference reports/controller_cpp_stage_runner_2026-07-26/frozen_8x4.json \
  --package-manifest-sha256 707bbe069ef6892ce9bfe98258f3289e28af15a400922e950c4386f56dd26926 \
  --executable-sha256 c6c5b05b6d8be72edd7f9e12e5e66c615859b74268143a5b2023b8dae423a15b \
  --out reports/native_bitnet_dip_attention_confirmation_2026-07-27/frozen_8x4.json \
  --max-tokens 4 --threads 12 --timeout 300

PYTHONPATH=src python -m engram.cli \
  evaluate-native-bitnet-dip-attention \
  --model work/native_bitnet/model.engram-bitnet-dip \
  --library build-runtime/libengram_bitnet_token_runtime.so \
  --out reports/native_bitnet_dip_attention_confirmation_2026-07-27/confirmation.json \
  --threads 12
```

`chat-native-bitnet` now crosses this boundary through
`libengram_bitnet_token_runtime.so`. The versioned C ABI accepts only the
authenticated package root, owns one mapped runtime, and requires a reset
before each full-history re-prefill. Python uses the packaged tokenizer and
chat template but does not construct a Transformers model or execute Torch.

The latest dense-source campaign implements whole-model exact Q-Sparse
co-adaptation rather than another router guess. A causally fitted per-layer
schedule improves the unseen 128-sequence baseline to KL 0.457, top-1 66.9%,
NLL delta +0.474, and hidden L2 0.328 at exactly 45% ideal traffic. Verified
attention/normalization co-adaptation moves it only to
0.452/67.1%/+0.458/0.327. Label-only continuation, token-adaptive budgets,
and a traffic-charged rank-24 residual do not improve the frontier, so this
dense-source arm remains stopped and confirmation stays sealed. The detailed
results are in the
[whole-model fully sparse report](reports/semantic_gate_fully_sparse_2026-07-24/summary.md).
Predictor-free DIP
passes the causal quality thresholds on an untouched confirmation corpus, but
requires 83.33% of dense MLP traffic after cache-line accounting and its native
kernel is slower than dense. A separately trained compact-Q4 student fits the
physical traffic limit at 44.9334%, but after 3,000,093 pretraining positions still has KL
0.887, top-1 agreement 56.6%, NLL delta +0.884, and final-hidden relative L2
0.425. The latest exact output-memory pilot improved layer-14 error only 1.73%
after adding one million independent prototypes, so that density-scaling path
is also closed. A final budget-edge campaign then tested recurrent reuse,
projection-normalized ternary weights, affine constrained vectors,
unrestricted codebooks, and LiftQuant-style lifted-binary lattices. Every arm
fit within 45% modeled cold traffic, but the best trained layer-local result
was still 0.308 relative L2 against a 0.20 progression ceiling.

The follow-up budget-native implementation trains all 30 full-width MLPs
through an exact grouped-ternary representation and independently reloads its
17,173,504-byte artifact before validation. It passes physical traffic at
43.1353%. After 1,014,225 fresh training positions, however, it reaches KL
2.284, top-1 agreement 32.0%, NLL delta +2.277, and final-hidden relative L2
0.604. A frozen scale-up rule required at least 50% closure of every remaining
quality gap; KL and NLL passed, while top-1 and hidden state did not. This
configuration is stopped before 3M rather than scaled on partial progress.

The new source track pins Microsoft's natively trained
`bitnet-b1.58-2B-4T`, validates its official two-bit checkpoint, and
losslessly repacks every ternary MLP coefficient as five base-3 trits per
byte. Each logical semantic record still contains one gate row, one up row,
one transposed down column, and the channel's BF16
intermediate-normalization gain, but the physical file groups those fields
into cache-aligned phase streams. That layout matches BitNet's gate/up,
normalization, and down execution order without rereading interleaved cache
lines. The independently reloaded 318,924,544-byte artifact and modeled
one-read-per-line phase schedule are 40.0527% of dense ideal Q4; charging
every scattered logical record independently is 41.6673%. All 1,592,524,800
ternary coefficients and 207,450 BF16 values
reconstruct exactly. A memory-mapped C++ kernel now executes those streams
directly without materializing dense weights. On the pinned tokenizer and
frozen 8-sequence/256-position corpus it reaches KL 0.00371, 96.09% teacher
top-1 agreement, NLL delta +0.00224, and final-hidden relative L2 0.04678.
The exact scheduled cold bytes remain 40.0527% of dense ideal Q4, so every
predeclared causal-quality and cold-byte checks pass on this source track.
Because every MLP record executes, this is not a Milestone 2 routing pass.

The renewed BitNet semantic experiment no longer executes every down record.
An exact-membership CPU kernel retains a development-fitted 15–35% per-layer
schedule, averaging **24.84%**. On the frozen 256-position protocol it reaches
KL **0.02543**, teacher top-1 **94.53%**, NLL delta **+0.02386**, and
final-hidden relative L2 **0.09205**. This proves that a small routed subset
can carry the teacher semantics, but it is an oracle ceiling rather than a
Milestone 2 pass: that oracle selector still scans dense gate/up coefficients.
See the [oracle report](reports/native_bitnet_oracle_2026-07-26/summary.md).

The practical follow-up now removes that dense coefficient path. The frozen
native DIP policy uses `q=1920` input coordinates in every layer, `minK=346`,
an energy target of 1.0, and the following layer-0-through-29 candidate and
maximum adaptive-K schedules:

```text
C    = [4224,5504,4224,4224,4224,4224,4224,4224,4480,4480,
        4736,4992,4480,4992,4992,4736,4992,4992,5248,4736,
        3456,5248,5248,5248,4992,3968,3200,4992,4224,4992]
Kmax = [4224,1705,4224,4224,4224,4224,4224,4224,3753,3753,
        3241,2729,3753,2729,2729,3241,2729,2729,2217,3241,
        3456,2217,2217,2217,2729,3968,3200,2729,4224,2729]
```

After exact candidate completion, the kernel selects the number of
positive-utility records for the current token, clipped to `[346,Kmax]`.
All layers except layer 9 estimate missing RMS energy by applying the
exact-to-proxy candidate-energy ratio to the proxy tail. Layer 9 uses
corrected proxy energy and reserves eight positions inside its unchanged
`C=4480` union for a top-proxy-raw-square audit. The qualifying live-BF16
development run passed every quality, activity, modeled-traffic, and recall
threshold. The source-bound v2 index and policy were independently reloaded,
and six rows in each layer have bit-exact Python/native input-coordinate,
candidate, selected-record, selected-count, and BF16-output parity. The
same [frozen policy](reports/native_bitnet_m2_2026-07-26/frozen_dip_policy.json)
was used for the consumed final attempt. Its raw report passes every
threshold; the original wrapper errored on the token-hash schema defect
described above, and the
[preserved evidence and adjudication](reports/native_bitnet_m2_final_audit/49df50cc01c96844ab3e7015d66c8899025dad4e1f7f01a450f97677751b36f2/summary.md)
support the semantic-gate pass-by-adjudication.

This is not evidence that a dense Llama checkpoint can be converted
losslessly. It also does not claim measured hardware DRAM events: the traffic
result is the exact cache-line schedule of the serialized streams. The concise,
milestone-by-milestone account is [Project status](docs/status.md), with prior
machine-readable metrics in the
[2026-07-23 status snapshot](reports/semantic_gate_status_2026-07-23/summary.json).
The exact budget-native protocol and stop rule are in the
[grouped-ternary report](reports/semantic_gate_budget_native_2026-07-23/summary.md).
The source audit, repack, and dense-oracle evidence is in the
[native BitNet report](reports/semantic_gate_native_bitnet_2026-07-23/summary.md).
The qualifying direct-kernel evidence is in the
[2026-07-24 confirmation](reports/semantic_gate_native_bitnet_2026-07-24/summary.md).
That artifact is now integrated into a checksummed, source-independent
1,108,116,808-byte package. The package excludes all 210 source MLP tensors,
loads only the embedding/attention/normalization/head tensors, installs the
memory-mapped C++ MLP kernel, and generates through the pinned tokenizer.
Package-backed and source-backed direct-kernel models have bit-exact final
hidden states and logits on the parity prompt; greedy generation produces
` Paris.` after `The capital of France is`.

For the native-BitNet source track, Milestone 3 attention substitution has a
bounded trained-model pass. The
initial exact hybrid established semantic capacity but scanned all older keys.
Random sign-LSH recalled only 58.8–65.6% of the exact older top-k, while exact
box and sphere page bounds opened about 94% of pages; those index branches are
rejected. The promoted streaming hybrid keeps 16 exact local tokens, two
attention sinks, and six online heavy hitters. It exact-scores those eight old
keys and transfers the best four values. On the sequence-disjoint frozen
8-sequence/256-position confirmation it reaches KL 0.01409, 94.14% top-1
agreement, NLL delta −0.00613, and final-hidden relative L2 0.08559. Old-context
storage and reads are fixed as context grows. The current 33-token protocol
models at 93.34% of dense KV traffic.

The state transition and eight-to-four rerank are now also implemented behind
a C ABI in C++20. Randomized eviction parity passes against an independent
NumPy state machine, and trained one-sequence substitution reaches KL 0.00528,
top-1 0.96875, NLL +0.01239, and hidden L2 0.04210. A standalone long-context
run keeps per-layer state fixed at 249,248 bytes while logical reads fall from
87.88% of dense at 33 tokens to 31.29% at 128, 8.40% at 512, and 2.14% at
2,048.

That kernel is now wired into compiled-package prefill and incremental greedy
generation. Every transformer layer owns a persistent bounded cache; the
runtime supplies monotonic absolute positions to normal BitNet RoPE and keeps
the Hugging Face dense KV cache disabled. Full-sequence and uneven incremental
chunks are bit-identical for the same bounded operator. On complete 30-layer
package generation, total attention state remains 7,477,440 bytes while
logical attention reads are 86.55%, 31.07%, and 16.35% of dense at 33, 128,
and 256 prompt tokens. End-to-end processing is only about one position per
second: Python-side projection/orchestration and the full vocabulary path
dominate, so these results establish bounded memory scaling, not production
latency or measured hardware DRAM traffic.

Collapsing prompt attention into one native stream call per layer preserves
outputs exactly but improves the 256-token run by only 0.55%. A phase profile
shows why: Q/K/V and O projections consume 19.31 seconds of a 38.51-second
33-token run, the full vocabulary projection consumes 12.62 seconds, the
packed MLP consumes 5.94 seconds, and the bounded cache itself consumes only
0.12 seconds. The next native work is packed ternary Q/K/V/O execution, not
further cache or call-loop tuning.

The packed-projection implementation now consumes the official
four-codes-per-byte package tensors directly through a shared threaded C++
kernel. On the controlled 33-token run it reduces Q/K/V/O time from 19.31 to
3.01 seconds and total time from 38.51 to 22.29 seconds (42.1%), with identical
generated tokens. Against the materialized-projection model on 32 trained
next-token positions it measures KL 0.00394, top-1 agreement 0.96875, target
NLL delta −0.00037, and final-hidden relative L2 0.03532. This clears the
development semantic thresholds. The sequence-disjoint frozen
8-sequence/256-position confirmation also passes: KL 0.00548, top-1 0.95703,
NLL delta +0.00200, and hidden L2 0.05887. Native projection execution takes
111.38 seconds versus 256.56 seconds materialized on that identical batch.
The full vocabulary projection, now about 13 seconds of the 22-second
generation run, is the dominant next target.

Generation now requests only the final prompt logit from the existing
Transformers API. This preserves the exact full-vocabulary argmax and avoids
projecting every prompt position. At 33 tokens, vocabulary time falls from
13.00 to 0.83 seconds and total time from 22.29 to 10.16 seconds. At 256
tokens, the fully optimized path takes 20.72 seconds versus the earlier
254.23-second stream-fused run, a 91.8% reduction, while generating the same
tokens and retaining the same 7,477,440-byte bounded attention state. An
approximate vocabulary index is therefore not justified for greedy package
generation; the packed MLP is again the dominant measured phase.

The first complete inference validation now passes. With packed MLPs, packed
Q/K/V/O, bounded native attention, incremental RoPE, and the exact last-row
vocabulary head enabled together, the frozen 8-sequence/256-position result is
KL 0.01315, top-1 0.92969, NLL delta +0.00365, and hidden L2 0.08436. Eight
natural prompts generated 16-token continuations without collapse; factual
and procedural completions are generally coherent, although the code prompt
drifts and the testing prompt adopts an exam-question format.

Full versus split-prompt logits are bit-identical, resets reproduce identical
tokens and stable state, and EOS termination has a unit-tested control path.
Complete prefill reaches 21.24 positions/s at 512 tokens and 25.05 positions/s
at 2,048 tokens. Attention state remains 7,477,440 bytes, but process peak RSS
is 2.14–2.57 GB because the Python/Transformers shell and prompt tensors
remain. Seven autoregressive steps take 38.26 seconds, about 5.47 seconds per
step. This is a working research inference engine, not an interactive runtime.

### Interactive native BitNet chat

Build the versioned token-runtime DSO and start the authenticated DIP package:

```console
cmake --build build-runtime --target engram_bitnet_token_runtime -j

PYTHONPATH=src python -m engram.cli chat-native-bitnet \
  --model work/native_bitnet/model.engram-bitnet-dip \
  --library build-runtime/libengram_bitnet_token_runtime.so \
  --threads 12 \
  --max-tokens 1
Engram native BitNet DIP chat. Commands: /reset, /history, /quit
You> Hello
Engram> Hello
[5.16s; 1 tokens; 7477440 attention-state bytes]
You>
```

This real smoke rendered the default system message and `Hello` to 17 tokens,
crossing the W=16 local-attention boundary. The native result matched the
standalone executable, and a second generation on the same mapped handle
after reset reproduced token `9906` and every non-timing structural metric.
Python performs local packaged tokenization and template rendering only; all
model execution is in the CPU-native DIP runtime.

The earlier pre-DIP Transformers-shell transcript is retained below as
historical behavioral evidence. Its command and timings do **not** describe
the current backend:

```console
PYTHONPATH=src python -m engram.cli chat-native-bitnet \
  --model work/native_bitnet/model.engram-bitnet \
  --library build/libengram_bitnet.so \
  --attention-library build/libengram_attention.so \
  --threads 12 \
  --max-tokens 32
Engram native BitNet chat. Commands: /reset, /history, /quit
You> write a random poem.
Engram> In the heart of the forest, where the trees whisper,
Lies a secret, a tale of a time.
Of ancient roots, of earth and sky,
[166.43s; 32 tokens; 7477440 attention-state bytes]
You> awesome!
Engram> I'm glad you liked it! Here's another one:

In the land of the sun and the moon,
Where the rivers run and the mountains loom,
[153.15s; 32 tokens; 7477440 attention-state bytes]
You>
```

Each turn is appended to structured user/assistant history, rendered with the
chat template stored in the authenticated package, then re-prefilled through a
fresh native cache. `/history` displays the current conversation, `/reset`
clears it while retaining the system message, and `/quit` exits. The initial
implementation does not stream tokens or preserve cache state between turns.
The old two-turn transcript showed that complete-history rendering conditioned
the second response, but this behavior still needs a new scripted multi-turn
confirmation on the DIP binding.
The detailed history below is retained so negative results remain auditable.

The repository contains an end-to-end research prototype: Hugging Face model inspection and
download, exact teacher tracing, SwiGLU decomposition, sparsity oracles, semantic routing and
quantization, recurrent and retrieval memory primitives, compiled packages, and PyTorch-free
Python and native C++20 generation.

The first trained-model experiments used `HuggingFaceTB/SmolLM2-135M`:

- Retaining 90% of each MLP output's energy required 16.6% of neurons on average; retaining 99%
  required 44.8%. This suggests useful sparsity, but not extreme sparsity at high fidelity.
- The current joint-key IVF router is the main bottleneck. With 256 active neurons and 512
  candidates, candidate recall was only 40.6% and practical relative error was 0.673 versus the
  oracle's 0.335.
- An experimental trace-calibrated router improves mean top-256 recall to 61.0% with 512
  candidates, and to 66.8% while examining about 641 records. Increasing calibration coverage
  fourfold did not improve that router, which motivated learned multi-label and
  coverage-optimized follow-ups.
- A learned multi-label ridge router reaches 65.9% recall with 512 candidates and 72.2% with 640.
  This confirms that direct oracle-membership supervision helps, but its dense scoring matrix is
  too expensive for production.
- Low-rank compression preserves most of that gain: rank 16 reaches 63.3% recall with 512
  candidates using 141 KB of float32 router parameters per layer, 4.0% of the dense router. Rank
  32 reaches 64.4% using 276 KB.
- Hierarchical rank-16 group selection followed by exact local reranking was not successful. Its
  best configuration reached only 52.8% recall at 512 records, and the router-weight saving is
  small beside the selected key traffic.
- Training groups directly for oracle coverage improves hierarchical recall to 54.6%. Multiple
  representatives do not improve that result.
- A trained-teacher intervention harness now replaces MLP outputs inside the original transformer
  and measures final normalized-hidden-state drift, logit KL, top-1/top-5 agreement, and held-out
  NLL. It verifies the identity path exactly before testing sparse arms.
- The old top-256 target fails under the full-information magnitude reference: replacing all 30
  MLPs raises KL by 0.648 and NLL by 0.668 nats/token, while preserving only 60.5% of teacher
  top-1 choices. Magnitude top-K is not guaranteed to be the optimal K-record subset.
  K=768—half of every layer's 1,536 records—is the first tested active count that passes the
  declared progression thresholds (KL 0.032, top-1 92.3%, NLL +0.022, final-hidden rel-L2 0.092).
- At K=768, full-corpus refits using all 1,112 calibration states per layer still fail after
  examining 1,280 candidates. The flat rank-16 router reaches 88.9% recall, KL 0.789, and NLL
  +0.764. Coverage-trained overlapping postings reach 86.8% recall, KL 1.149, and NLL +1.095,
  while scanning about 1,667 posting entries to form 1,280 unique candidates. More calibration
  data modestly improved recall but did not close the 95% recall or downstream-quality gaps.
- A cached regularization sweep found a shallow optimum near λ=8,000. Raising the candidate
  budget to 1,408 and 1,472 clears the recall gate at 95.4% and 97.8%, but causal substitution
  still fails: the 1,472-candidate arm has KL 0.085, top-1 agreement 86.6%, NLL +0.055, and
  final-hidden rel-L2 0.131. It reads 95.8% of record keys, leaving only about a 1.24× projected
  key/value traffic reduction before router overhead. This rank-16 configuration is abandoned.
- A predictor-free, DIP-inspired path now uses the source model's own gate/up weights on the
  largest-magnitude input coordinates. Engram then exactly completes only its candidate records
  and reranks them to K=768; that completion/reranking stage is an Engram extension to the
  published DIP method. It requires no learned membership router. The recommended
  75%-input/896-candidate arm passes both the development frontier and a separate untouched
  confirmation corpus. Confirmation metrics are 99.0% candidate recall, KL 0.029, 91.0% top-1
  agreement, NLL +0.033, and final-hidden rel-L2 0.090. Its logical float32 weight-read model is
  76.4% of dense MLP traffic, a projected 1.31x reduction before indexes and cache effects.
- The selector now has an experimental version-2 coordinate-major package and a candidate-only
  native kernel. Cache-line accounting raises the same arm to 83.3% of dense bytes. After
  structure-of-arrays, partition selection, sorted gathers, and float32 parity work, the best
  30-layer streamed kernel is still about 15.4% slower than dense (`0.863x`). It is explicitly
  rejected before default-runtime integration. A spatial 16-float-block layout was
  also rejected: confirmation recall fell to 85.2%. The semantic quality pass therefore survives,
  but the current systems implementation does not pass.
- A second-generation sparse-teacher path now targets the systems failure directly. Its default
  budget is `q=62.5%`, `C=K=512`; the student evaluates all records on only the retained input
  coordinates, completes exactly 512 candidates, and reads 512 down records. Straight-through
  candidate masks let local-MLP, hidden-state, and logit losses train routing, while a cache-line
  occupancy loss penalizes scattered candidates. The first one-record SmolLM2 smoke run verifies
  this gradient path and sparse execution. It starts at 90.0% candidate recall, but its 512
  candidates touch 95.84/96 gate/up cache-line groups on average. Cache-line amplification raises
  the optimistic 61.1% scalar estimate to about 77.7% of dense traffic. This is a training starting
  point, not a quality or speed pass; a full-corpus training run must materially improve both recall
  and locality before compilation.
- The complete 32-sequence/16-held-out run at that budget fails despite meeting the evidence and
  hardware-budget checks: recall is 89.59%, KL 0.166, top-1 agreement 76.78%, NLL delta +0.126,
  and final-hidden relative L2 0.199. Candidates still touch 95.86/96 gate/up line groups.
  Balanced storage permutations reduce this only to 94.66 lines; forcing selection of 32 complete
  line groups cuts recall to at most 48.73% and raises local MLP error above 0.47.
- The trainer now supports masked sequence batching, provenance-checked router initialization
  caches, separate calibration/training corpora, and mergeable rank-8 LoRA updates for gate, up,
  and down projections. A deterministic local-source corpus builder produced 128 sequences and
  15,991 token positions. A bounded 16-sequence LoRA stage modestly improved KL/top-1 but worsened
  NLL and left hidden error, recall, and locality effectively unchanged, so it was not scaled.
- A held-out oracle bound now explains the locality failure rather than treating it as an optimizer
  mystery: exact top-512 membership already touches 95.86/96 contiguous 16-record lines. Even a
  perfect group selector limited to 80 lines can cover only 91.75% of the oracle set; reaching
  96.65% requires 88 lines. A duplicated record-major v3 package is also rejected: it grows MLP
  storage by 66.7% and its tested kernels are slower. Version 2 remains the default research
  package.
- The locality relaxation was audited and replaced with an exact-hard-value, fixed-cardinality
  soft-backward objective. Gradient diagnostics show its unweighted router gradient is about 269x
  smaller than the causal gradient, but a balanced 16-step trial still does not reduce hard line
  occupancy. Standard LoRA scaling and resumable checkpoints are now implemented. A full
  128-sequence rank-32 residual run improves KL/top-1/NLL/hidden L2 to
  0.152/0.780/+0.100/0.193, but still fails every causal threshold. The residual has essentially
  zero alignment with the missing output and is disabled by default; higher adapter learning rates
  are unstable.
- A fixed-total layer-adaptive magnitude oracle was selected from individual-layer interventions
  on a separate four-sequence split and frozen before confirmation. At the same mean K=512 it is
  slightly worse than uniform K=512 on the untouched 16-sequence set (KL 0.134, top-1 0.786, NLL
  +0.110, hidden L2 0.185). Layer adaptation is also stopped. The next justified direction is a
  co-trained structured expert/block representation, not more tuning of the frozen neuron basis.
- A new structured-expert shadow path tests that direction before expensive training. Balanced
  24×64/top-8, 48×32/top-16, and 96×16/top-32 layouts all execute exactly 512 records and preserve
  the dense all-block output to below 8.6e-7 maximum relative L2. However, even a non-deployable
  greedy-residual block oracle has mean local error 0.547/0.497/0.438, and fitted routers worsen
  those to 0.655/0.638/0.624. Static grouping is therefore stopped before end-to-end training. The
  next bounded design is co-trained native gate-based channel sparsity with hardware-aligned
  grouped selection, not a larger static expert router.
- The native-gate follow-up removes candidate completion entirely. At K=512, the exact
  contribution reference has local relative L2 0.190, while dense-gate channel selection is 0.375;
  q=62.5% input pruning moves it only to 0.386 at 43.06% ideal traffic. A hard-forward/soft-backward
  full-weight wrapper and cached-trace pretrainer are implemented, but a controlled 64-step
  representative-layer run improves held-out error only 2.55% and fails its 10% screen. This local
  pretraining path is stopped; the next credible run requires progressive end-to-end co-training
  on materially more data. The implementation remains CPU-capable; CUDA is an optional training
  accelerator, not an inference or format dependency.
- Progressive end-to-end native-gate co-training is now implemented and runs on CPU. It anneals
  dense execution to q=62.5%/K=512, co-trains full MLP weights while freezing non-MLP transformer
  components, validates through the hard path only, and supports resumable device-neutral
  checkpoints. The full-evidence untrained baseline is KL 1.235/top-1 0.460/NLL +1.202/hidden L2
  0.508. An eight-step CPU stage reaches 1.254/0.481/+1.211/0.510: mixed movement, not justification
  for a longer run on the same objective. The trainer is ready for controlled CPU slices or optional
  CUDA acceleration once a better training curriculum/data scale is justified.
- A low-rank utility-residual router now predicts the missing up-projection-dependent channel
  utility from the current hidden state. With 512 calibration states, rank 16/blend 0.8 lowers the
  trace-local error from 0.386 to 0.338 and raises exact-oracle recall to 0.643 at 44.39% projected
  dense traffic. The full all-layer hard-path control confirms that this is causal: KL falls from
  1.235 to 0.629, top-1 agreement rises from 0.460 to 0.599, NLL delta falls from +1.202 to +0.583,
  and hidden L2 falls from 0.508 to 0.363. It still misses the final quality gate. A matched
  eight-step run slightly improves top-1 but regresses the other metrics, so the next bounded
  experiment is on-policy residual recalibration on sparse-student states, not longer training.
- That on-policy screen is now complete and negative: same-state local L2 changes only
  0.35117→0.34983. A 44.25%-traffic q=43.75%/K=640 alternative also fails causally
  (KL 0.684, top-1 0.603, NLL +0.616, hidden L2 0.358). Since even the exact K=512 oracle misses
  the gate, the frozen-basis router search is closed. The remaining Milestone 2 path is structured
  sparse upcycling/width pruning with full MLP adaptation and materially more real-token data.
- The first all-layer fixed-width student now tests that path directly. It replaces every
  1,536-wide SwiGLU with a trainable 672-wide contiguous SwiGLU, freezes attention and
  normalization, and stays at 43.75% of dense MLP weight traffic. Training is CPU-capable,
  checkpointed, and supports parameter-only transfer to a fresh corpus. One full epoch over 2,048
  sequence-disjoint local-source examples improves the held-out result to KL 1.177, top-1 47.5%,
  NLL delta +1.055, hidden relative L2 0.426, and local MLP relative L2 0.705. This is a decisive
  semantic-gate failure, not a compilation candidate. More epochs on the same narrow fixed-width
  objective are stopped; the next experiment must measure the teacher-boundary local approximation
  ceiling before spending on another causal run.
- That ceiling is now measured on 4,096 sampled training boundaries and 446 sequence-disjoint
  validation boundaries. Five representative compact layers trained for 2,048 cached-boundary
  steps improve mean local L2 from 0.3851 to 0.3457, but miss the declared 0.15 ceiling. Middle
  layers remain between 0.45 and 0.50. Width 672 is therefore rejected as a uniform all-layer
  representation; the next design must allocate capacity by layer or use a more expressive
  structured basis while retaining the same aggregate traffic cap.
- The fitted rank-4 background operator worsened mean held-out error, so it is not currently a
  viable correction.

The generic dense-Llama compiler still installs its controller initializer.
The separate native-BitNet package now embeds the schema-v3 exact residual
controller, whose learned correction is disabled; this integration preserves
known operator additions rather than claiming a learned recurrent replacement
for them. Learned rank-16,
posting-group, residual-capsule, and first sparse-teacher artifacts
remain blocked. The older dense-SmolLM DIP arm was the first realizable
selector to clear its semantic quality prerequisite, but it failed systems
traffic and latency. The newer native-BitNet DIP implementation passes the
complete final quality/recall/activity/modeled-traffic gate by postmortem
adjudication. Its frozen index has now been installed into a derived,
authenticated `model.engram-bitnet-dip` package and exercised through the
DIP-only native token runtime. It is not the generic dense-Llama `.engram`
format and its timing is still not a speedup. The original sparse-teacher
pilot's disconnected routing gradient is fixed in the hardware-aware trainer, but the complete
low-budget evaluation, corrected LoRA/residual run, locality bound, and layer-adaptive confirmation
all fail. These artifacts remain blocked; structured sparsity must be learned jointly with the MLP
weights before another compilation attempt. The first 672-wide jointly adapted student also fails
after complete exposure to its 2,048-sequence corpus, so compilation remains blocked.
Subsequent adaptive low-bit, structured-basis, compact-Q4, conditional-expert,
and nonparametric output-memory experiments also fail their frozen progression
rules. In particular, the serialized mild-width Q4 student passes the 45%
physical-byte check but remains far outside every causal threshold after 3M
pretraining positions; scaling exact output memory from 233,005 local records
to 1,233,005 combined records changes layer-14 error only from 0.327526 to
0.321854. Recurrent compact, normalized ternary, affine constrained-vector,
unrestricted-codebook, and lifted-binary follow-ups also fail their local
screens despite modeled traffic of 41.00%–44.98%. No dense-source converted
semantic artifact is currently eligible for default package compilation; the
separately trained native-BitNet artifact is the current teacher and CPU
substrate. Its DIP index and native kernel now form an adjudicated
semantic-gate-passing routed-memory candidate. This closes the practical
native-BitNet semantic evidence gate, not the dense-Llama conversion problem
or every remaining integration and replication item in Milestone 2.
See [architecture](docs/architecture.md), [evaluation](docs/evaluation.md),
and [limitations](docs/limitations.md) for the precise design and caveats. The latest routing
measurement is documented in the [trace-calibrated recall report](reports/smollm2_calibrated_router/recall.md).
The directly supervised follow-up is in the
[multi-label routing report](reports/smollm2_multilabel_router/recall.md).
The compression frontier is measured in the
[low-rank routing report](reports/smollm2_lowrank_router/recall.md).
The hierarchical follow-up and its negative result are in the
[hierarchical routing report](reports/smollm2_hierarchical_router/recall.md).
The direct coverage and multiple-representative experiments are in the
[coverage-trained group report](reports/smollm2_coverage_groups/recall.md).
The causal intervention frontier and router decisions are summarized in the
[trained-teacher intervention decision](reports/smollm2_mlp_intervention/decision.md), with the
machine-readable arm reports linked there. A
[provenance-checked composite report](reports/smollm2_mlp_intervention_composite/mlp_intervention.json)
applies the final gate across the separately executed arms.

### Distill the shared controller

Teacher capture is CPU-only and uses the already packaged native-BitNet model.
Use different corpora for training and validation so development evidence is
not fitted and measured on the same text:

```bash
PYTHONPATH=src python -m engram.cli trace-native-bitnet-controller \
  --model work/native_bitnet/model.engram-bitnet \
  --dataset work/controller_train.jsonl \
  --out work/controller/train-trace \
  --split training --samples 64 --max-tokens 64 --batch-size 2 \
  --library build/libengram_bitnet.so --threads 12

PYTHONPATH=src python -m engram.cli trace-native-bitnet-controller \
  --model work/native_bitnet/model.engram-bitnet \
  --dataset work/controller_validation.jsonl \
  --out work/controller/validation-trace \
  --split validation --samples 16 --max-tokens 64 --batch-size 2 \
  --library build/libengram_bitnet.so --threads 12

# On memory-constrained CPU hosts, capture independent chunks and merge only
# after their authenticated contracts and sample IDs have been checked.
PYTHONPATH=src python -m engram.cli merge-controller-traces \
  --traces work/controller/train-chunk0 work/controller/train-chunk1 \
  --out work/controller/train-trace

PYTHONPATH=src python -m engram.cli distill-controller \
  --trace work/controller/train-trace \
  --validation-trace work/controller/validation-trace \
  --out work/controller/rank128 \
  --device cuda --rank 128 --adapter-rank 4 \
  --operator-residual --steps 0 --batch-size 16

PYTHONPATH=src python -m engram.cli evaluate-native-bitnet-controller \
  --model work/native_bitnet/model.engram-bitnet \
  --dataset tests/fixtures/confirmation_expanded.jsonl \
  --controller work/controller/rank128/controller \
  --out reports/controller_compiled_substitution/frozen.json \
  --library build/libengram_bitnet.so \
  --attention-library build/libengram_attention.so \
  --threads 12 --sequence-count 8 --prediction-positions 256 \
  --record-offset 8

PYTHONPATH=src python -m engram.cli \
  evaluate-native-bitnet-controller-generation \
  --model work/native_bitnet/model.engram-bitnet \
  --controller work/controller/rank128/controller \
  --prompts tests/fixtures/inference_prompts.jsonl \
  --out reports/controller_generation/frozen.json \
  --max-tokens 4 \
  --mlp-library build/libengram_bitnet.so \
  --attention-library build/libengram_attention.so \
  --threads 12

PYTHONPATH=src python -m engram.cli install-native-bitnet-controller \
  --model work/native_bitnet/model.engram-bitnet \
  --controller work/controller/rank128/controller

PYTHONPATH=src python -m engram.cli generate-native-bitnet-controller \
  --model work/native_bitnet/model.engram-bitnet \
  --prompt "The capital of France is" --max-tokens 4 \
  --library build/libengram_bitnet.so \
  --attention-library build/libengram_attention.so --threads 12
```

An interrupted capture can be restarted with the identical arguments plus
`--resume`; completed sample IDs and checksummed shards are not duplicated.
CUDA is used only to accelerate fitting or evaluation when available. The
deployable artifact is
`work/controller/rank128/controller/`: a metadata file plus FP32 `.npy`
tensors loaded by `FactorizedRecurrentController` on CPU. See the
[latest controller transition report](reports/controller_distillation_bitnet_2026-07-25-operator-residual/summary.md).
The frozen joint operator result is in the
[compiled controller substitution report](reports/controller_compiled_substitution_2026-07-25/summary.md).
The incremental parity result is in the
[controller generation report](reports/controller_incremental_generation_2026-07-25/summary.md).
The static structured-expert screens are recorded for
[64-record blocks](reports/smollm2_structured_expert_shadow/structured_expert_shadow.md),
[32-record blocks](reports/smollm2_structured_expert_shadow_48x32/structured_expert_shadow.md),
and [16-record blocks](reports/smollm2_structured_expert_shadow_96x16/structured_expert_shadow.md).
The native-gate diagnosis is in the
[channel shadow report](reports/smollm2_native_gate_channel_shadow/native_gate_channel_shadow.md),
and the bounded training stop is in the
[64-step layer report](reports/smollm2_native_gate_trace_layer14_utility64/native_gate_trace_training.md).
The device-neutral end-to-end controls are the
[untrained CPU baseline](reports/smollm2_native_gate_e2e_cpu_baseline/native_gate_end_to_end.md)
and [eight-step CPU stage](reports/smollm2_native_gate_e2e_cpu_stage8/native_gate_end_to_end.md).
The passing local residual screen and device-neutral router tensors are in the
[expanded residual report](reports/smollm2_native_gate_utility_residual_expanded/native_gate_utility_residual.md).
Its all-layer controls are the
[untrained residual run](reports/smollm2_native_gate_e2e_cpu_residual_baseline/native_gate_end_to_end.md)
and [matched eight-step run](reports/smollm2_native_gate_e2e_cpu_residual_stage8/native_gate_end_to_end.md).
The cached [regularization sweep](reports/smollm2_rank_router_regularization_sweep/rank_router_regularization_sweep.md),
[candidate frontier](reports/smollm2_rank_router_candidate_frontier/rank_router_regularization_sweep.md),
and [near-dense causal check](reports/smollm2_mlp_intervention_rank16_lambda8000_frontier/mlp_intervention.md)
record why the flat rank-16 configuration is no longer being pursued.
The [global correction-capsule sweep](reports/smollm2_correction_capsule_sweep/correction_capsule_sweep.md)
and [targeted tight-radius sweep](reports/smollm2_correction_capsule_targeted_tight_sweep/correction_capsule_sweep.md)
record the negative residual-correction result.
The [sparse-teacher pilot](reports/smollm2_sparse_teacher_epoch1/sparse_teacher_training.md)
records the first trainable sparse-student result and its unchanged stop decision.
The [hardware-aware sparse-teacher smoke run](reports/smollm2_hardware_sparse_smoke/sparse_teacher_training.md)
checks the replacement gradient path and low-budget/cache-line reporting without claiming a
full-corpus result.
The [complete low-budget gate](reports/smollm2_hardware_sparse_full/sparse_teacher_training.md),
[three-projection LoRA stage](reports/smollm2_hardware_sparse_lora_stage/sparse_teacher_training.md),
and [broader-corpus stage](reports/smollm2_hardware_sparse_corpus_stage/sparse_teacher_training.md)
record the subsequent stop decision and bounded follow-ups.
The [oracle locality bound](reports/smollm2_locality_oracle_bound/oracle_line_coverage.md),
[dual-layout diagnostic](reports/smollm2_dip_dual_layout/dual_layout_benchmark.md),
[full corrected-LoRA/residual run](reports/smollm2_residual_r32_scaled_full/sparse_teacher_training.md),
and [layer-adaptive confirmation](reports/smollm2_mlp_intervention_oracle_adaptive512_causal/mlp_intervention.md)
record the final low-budget representation tests.
The predictor-free [DIP trace sweep](reports/smollm2_dip_exact_completion_sweep/dip_exact_completion_sweep.md)
and [causal frontier](reports/smollm2_mlp_intervention_dip_frontier/mlp_intervention.md)
record its development selection and measured quality/projected scalar-read frontier. The
[untouched confirmation report](reports/smollm2_mlp_intervention_dip_confirmation/mlp_intervention.md)
freezes the 75%/896 configuration and verifies zero exact sequence overlap with the selection set.
The [blocked-layout confirmation](reports/smollm2_dip_blocked_confirmation/dip_exact_completion_sweep.md)
and [native layer benchmark](reports/smollm2_dip_native_layer10/dip_native_benchmark.md) record the
subsequent negative systems results.

## How conversion and inference work

A Llama model alternates attention blocks, which move information between token positions, and
SwiGLU MLP blocks, which transform each position independently. Engram treats every MLP neuron as
a memory record with two lookup keys and one output value. It reads those tensors directly from
the Hugging Face checkpoint, records the real inputs and outputs seen at each layer, and measures
which records matter for each state. The converter then quantizes the records, builds indexes for
sparse lookup, copies tokenizer and embedding data, and writes a checksummed `.engram` directory.
The original transformer layers are not needed to load that directory.

At inference time, the runtime tokenizes the prompt and maintains one fixed-width recurrent state.
For each token it retrieves semantic records, updates bounded short- and long-context memory,
runs a shared recurrent controller for a small number of cycles, and searches the vocabulary for
the next token. The intended trained system will distill the controller and episodic mechanisms
from teacher traces. The current runnable baseline exercises this dataflow but uses an initialized
controller and heuristic episodic memory, so it is infrastructure for the research rather than a
quality-preserving conversion.

For a detailed explanation written for readers who know general computer science but not language
models, see [How Engram works](docs/how_engram_works.md). It covers the source Llama computation,
extraction process, compiled format, inference loop, and the work still required.

## Quick start

Python 3.10+, NumPy, CMake 3.20+, and a C++20 compiler are required. PyTorch,
Transformers, and safetensors are only required for real Hugging Face checkpoints.

```bash
python -m pip install -e '.[dev,conversion]'
engram create-fixture --out work/tiny-llama --seed 7
engram inspect --model work/tiny-llama --out work/inspection.json
engram trace \
  --model work/tiny-llama \
  --dataset tests/fixtures/calibration.jsonl \
  --out work/traces \
  --samples 32
engram analyze-mlp \
  --model work/tiny-llama \
  --traces work/traces \
  --out reports/generated/milestone1
engram build-semantic --model work/tiny-llama --out work/tiny.engram
engram trace --model work/tiny-llama --out work/validation-traces \
  --split validation --samples 32 --seed 18
engram evaluate-semantic --model work/tiny-llama \
  --calibration-traces work/traces \
  --validation-traces work/validation-traces \
  --out reports/generated/milestone2
engram evaluate-attention --out reports/generated/milestone3
engram evaluate-controller --out reports/generated/milestone4
engram compile --model work/tiny-llama --out work/tiny.engram
engram validate --model work/tiny.engram
engram generate --model work/tiny.engram --prompt "hello" --max-tokens 16

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/engram-inspect work/tiny.engram
./build/engram-run work/tiny.engram --prompt "hello" --max-tokens 16 --greedy
./build/engram-bench work/tiny.engram 512
```

The fixture is random and only validates the pipeline. To produce meaningful evidence,
use a trained Llama-compatible Hugging Face model. Pass either a local directory or a
Hub model ID. Hub models are downloaded automatically into the standard Hugging Face
cache and reused on subsequent commands. The current semantic format requires bias-free
SwiGLU MLP projections and rejects checkpoints with `mlp_bias=true`:

```bash
engram inspect --model HuggingFaceTB/SmolLM2-135M
engram trace \
  --model HuggingFaceTB/SmolLM2-135M \
  --dataset /absolute/path/to/calibration.jsonl \
  --out work/real-traces \
  --samples 128
engram analyze-mlp \
  --model HuggingFaceTB/SmolLM2-135M \
  --traces work/real-traces \
  --out reports/generated/real-model
engram evaluate-mlp-intervention \
  --model HuggingFaceTB/SmolLM2-135M \
  --dataset /absolute/path/to/held-out.jsonl \
  --out reports/generated/mlp-quality \
  --variants identity oracle \
  --top-k 256 512 768 \
  --layer-mode all
engram sweep-dip \
  --model HuggingFaceTB/SmolLM2-135M \
  --validation-traces /absolute/path/to/validation-traces \
  --out reports/generated/dip-sweep \
  --input-fractions 0.5 0.625 0.75 \
  --top-k 768 \
  --candidates 896 1024 1152
engram evaluate-mlp-intervention \
  --model HuggingFaceTB/SmolLM2-135M \
  --dataset /absolute/path/to/untouched-confirmation.jsonl \
  --out reports/generated/dip-confirmation \
  --variants identity oracle dip \
  --input-fractions 0.75 \
  --top-k 768 \
  --candidates 896 \
  --layer-mode all \
  --evaluation-role confirmation \
  --configuration-selection-traces /absolute/path/to/validation-traces
engram build-distillation-corpus \
  --model HuggingFaceTB/SmolLM2-135M \
  --input README.md docs src native tests \
  --out work/sparse-distillation.jsonl \
  --sequence-length 128 \
  --max-sequences 128
engram train-sparse-student \
  --model HuggingFaceTB/SmolLM2-135M \
  --calibration-dataset /absolute/path/to/calibration.jsonl \
  --training-dataset work/sparse-distillation.jsonl \
  --validation-dataset /absolute/path/to/held-out.jsonl \
  --calibration-traces /absolute/path/to/calibration-traces \
  --out reports/generated/hardware-sparse-student \
  --routing-mode hardware_ste \
  --input-fraction 0.625 \
  --top-k 512 \
  --candidates 512 \
  --locality-weight 0.05
engram train-budget-native-ternary \
  --model HuggingFaceTB/SmolLM2-135M \
  --training-dataset /absolute/path/to/pretraining-distillation.jsonl \
  --validation-dataset /absolute/path/to/held-out.jsonl \
  --out work/budget-native-ternary \
  --steps 128 \
  --anneal-steps 96 \
  --transition-mode deepest_first \
  --coadapt-backbone \
  --backbone-start-step 96 \
  --checkpoint-every 32 \
  --device cpu
```

A successful command exit is not a compilation claim. The generated intervention report applies
explicit quality gates; routed arms must pass before their parameters are eligible for
serialization. The grouped-ternary command writes an exact byte-accounted research artifact, but
the checked SmolLM2 configuration is stopped by its progression rule and is not a supported
compiler input. `engram gate-mlp-intervention --report PATH` reapplies the current declared
thresholds to an existing report. Supplying several `--report` paths plus `--out DIRECTORY`
creates a provenance-checked composite gate, which is useful when expensive arms were run in
stages.

For gated repositories, authenticate first with `hf auth login` or set `HF_TOKEN`.
Existing local directories continue to work without network access.
The cache location follows Hugging Face defaults and can be changed with `HF_HOME` or
`HF_HUB_CACHE`. See the [conversion pipeline](docs/conversion_pipeline.md) for model-source
resolution details and supported commands.

Dataset records may contain either `{"text": "...", "input_type": "prose"}` or
pretokenized `{"input_ids": [1, 2, 3], "input_type": "code"}`. Pretokenized input is
useful for tiny local checkpoints without tokenizer assets.

## Verification

```bash
python -m pytest -q
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
/usr/bin/ctest --test-dir build --output-on-failure
```

The explicit `/usr/bin/ctest` avoids a broken user-local Python wrapper observed on the
development host. Ordinary `ctest` is correct when it resolves to the CMake executable.

## Scientific interpretation

The “oracle” computes every SwiGLU activation and ranks neuron records by
`abs(activation_j) * ||value_j||₂`; it then scans every prefix because vector cancellation
can make reconstruction error non-monotonic. It is a strong full-information
contribution-magnitude baseline, not the mathematically optimal K-subset and not a
production router. A target of 90% means
`||full - approximation||² / ||full||² <= 0.10`.

The checked-in [fixture report](reports/milestone1_fixture/oracle_topk.md) is pipeline
evidence only. A subsequent SmolLM2-135M experiment measured trained-model sparsity and a
fitted-background ablation; the background failed to improve held-out mean error. Those pilot
corpora remain too small for a broad model-family claim.

The [Gate 2 fixture report](reports/milestone2_fixture/practical_routing.md) preserves a
negative result: joint-key IVF scored 18.25 of 32 records on average, but candidate recall
was only 0.578 and reconstruction error trailed the oracle. The low-rank background also
overfit the small random calibration set. This is instrumentation evidence, not a reason
to claim the semantic-memory hypothesis works.

The [Gate 3 synthetic report](reports/milestone3_fixture/attention_replacement.md) covers
bounded local, recurrent, and older-context retrieval memory. It is not teacher-attention
distillation evidence. It is superseded for the native-BitNet track by the
trained-model W16/C8/K4/S2 attention confirmation and its incremental package
integration.

For the OLMoE track, the trained-model evidence is source-specific and
negative at sustained context: W16 fails after offset 31, a 100%-read W128
control passes, three matched global allocations fail, and the authenticated
three-layer W128 rescue also fails its six-sequence internal screen at 44.17%
logical reads. A prospectively frozen fixed teacher-attention-mass mask then
rescued 51 of 256 heads at 44.975387218386625% reads and improved every
overall metric relative to that layer rescue, but still failed all four
quality gates after position 31. A subsequent causal/value-sensitive
two-record fit improved its training objective on both records, yet its
static 51-head native-Q7 mask was worse than the attention-mass mask on all
four overall screen metrics. These six-record screens are development
evidence from an already consumed corpus, not fresh Milestone 3
confirmations. Those two static objectives are closed. The separate
retrieval-targeted implementation and its new synthetic 8/8/8 passkey
protocol completed its full fit and development screen. `M2` passed the
training-selection rule and the full-W128 control passed, but the static
exact-51 candidate failed KL, NLL-delta, and hidden-state gates. The sealed
confirmation remains unopened. Subsequent K2 prefix, exact payload,
label-plus-payload, and head-gated K51 train screens also failed semantically
while passing their systems budgets. The prospectively frozen
K64/96/128/165 ranked episodic head-prefix screen then executed every
candidate and failed: its deterministic best failed replay was K165 at
1.331006 mean and 1.608617 worst CE, while only one record improved. The
following fixed-K256 V2 logit-bias screen also failed all four nonzero arms;
`gamma=1/2` was merely its replayed best failure and was worse than the
historical `beta=0` result. Shared scalar calibration is closed. The
same-state W128-shadow capacity screen that followed also failed: ranks
2/4/8 recovered 40.05%/42.87%/46.93% globally, so each missed the frozen 50%
gate despite passing every other condition. This closes only those global
per-layer output subspaces with oracle coefficients. The dynamic per-head
mass-matching oracle then recovered -10.89% globally, with all eight sequences
negative. Its joint output-targeted successor also failed: even the continuous
optimistic relaxation recovered only **0.22738059544921096** globally
(1/8 sequences and 0/4 block entries at or above 0.25), while direct discrete
replay recovered **0.1997680396822742** (0/8 and 0/4); both had 16/16
positive layers. Protocol/result SHA-256 roots are
`aa03a71e3dd9e1fbb413a7773d57189c41029c26b2f4372b2fb7a26744305d24`
and
`1329a51bac71cb81f44494c8ef70cb23a631eacebac5540b39e2e98ed5e30ea5`.
No predictor, Milestone 3 promotion, or confirmation access was authorized.
The exact per-slot successor then optimized the eight episodic values
individually. Its constructible and exact-native-anchor optimistic hulls
recovered **0.3844378107** and **0.3844378142** globally. Both passed all
sequence, block-entry, and positive-layer checks but decisively missed the
50% global requirement; the optimistic maximum objective-gap bound was only
`5.90e-11`. Scalar mass retuning and this current nine-direction value family
are closed. The no-extra-read full-visible successor then passed its frozen
train-only capacity gate: constructible C28 recovered **0.6653937751**
globally, with minimum sequence/block recovery
**0.6447006551/0.6306278392** and 16/16 positive layers; optimistic C29
recovered **0.6653865288**. It retained 10,534,912 bytes of state and
714,866,688 logical traffic bytes (33.0305% of dense), with no new KV reads.
This authorized a causal 28-logit selector, not Milestone 3 progression,
development, confirmation, or a claim that the selector was learnable. The
subsequent rank-4 query-content-plus-mass selector recovered only
**0.25422074198** in BF16, and the smaller phase-conditioned mass selector
recovered **0.2618728353**. Both are frozen train-only model-selection
failures; all systems checks passed, but neither advanced to development,
confirmation, or native integration. Phase improved the preceding mass-only
result by only **0.0040699**, so query-content and phase-on-mass are closed.
The next directional boundary is a blockwise-QK feature controller. See the
[frozen full-visible report](reports/olmoe_q7_retrieval_episodic_full_visible_simplex_oracle_2026-07-30/README.md).

[Gate 4](reports/milestone4_fixture/controller_gate.json) is also synthetic; adaptive
execution averaged 7.98 of 8 allowed cycles, so it found essentially no compute saving.
The [runtime benchmark](reports/runtime_fixture/benchmark.md) is a tiny-fixture systems
measurement. Native-BitNet now has separate trained-model controller,
compiled-operator, incremental-generation, C++ orchestration, and packaged
DIP token-runtime evidence; those results do not turn the original synthetic
fixture into trained evidence or qualify the generic dense-Llama compiler.

The checked [Gate 5 random-fixture report](reports/milestone5_fixture/end_to_end_quality.md)
validates that evaluator and records a negative result: zero category target accuracy and 93.75%
repetition. Its small KL is an artifact of near-uniform random logits.

The trained-teacher MLP intervention is narrower and more diagnostic than Gate 5: it keeps the
trained transformer's attention, residual path, normalization, and vocabulary head exact while
replacing only selected MLP outputs. The checked SmolLM2 result finds that full-information
magnitude top-768 is the first tested selection that passes the declared progression thresholds,
and all learned practical routers fail. Predictor-free DIP subsequently passes with 75% of input
coordinates and 896 candidates. Experimental serialization and a native kernel now exist, but the
kernel fails its isolated latency gate and remains outside the compiled runtime. There is still no
trained controller or trained-package Gate 5 result.

### Key-side candidate trace (train-only)

The native C28 evaluator can now record the exact post-RoPE keys for each of
the eight older candidates before native top-K truncation.  The authenticated
tensor is `[8, 32, 16, 16, 8, 128]` and lives in the separate
`work/olmoe_q7/retrieval_episodic_blockwise_qk_candidate_keys_2026-07-31/`
artifact.  It is a shadow trace only: normal attention output and traffic are
unchanged, reset replay is exact.  The separately authorized protected replay
now passes for the frozen policy; this trace remains an evaluator artifact.

An offline per-layer/head PCA audit gives a concrete compression boundary:
rank 16 has 0.99% mean normalized key error (4.50% p95) with 6.69% modeled
dense-key traffic, while rank 32 has 0.14% mean error at 13.33% modeled
traffic.  These figures measure key reconstruction, not routing recall.  The
next experiment must combine the compressed key summaries with held-out query
features and exact reranking inside selected groups before a causal policy is
considered.

The first held-out query/key compatibility screen is deliberately recorded as
a negative control.  A rank-8 diagonal bilinear router reaches 51.13% mean
candidate membership recall (25.0% p10; 6.32% exact top-4) and 51.52% mean
oracle-mass retention, while the exact-score ceiling is 100%.  Capturing keys
without the actual per-head query projection is not enough; the next selector
experiment must expose or distill query vectors before any causal change.

With the authenticated per-head query features (including the native RoPE
convention), record-held-out key reconstruction reaches 87.19%, 92.88%,
**95.17%**, 96.25%, and 97.17% mean candidate membership recall at ranks 4,
8, 16, 32, and 64.  Rank 16 still has 75.0% p10 recall and 81.10%
exact-top-4 rows, so it is a promising feature boundary rather than a causal
pass.  The next implementation is rank-16 group selection followed by exact
reranking only inside selected groups.

### Query-aware exact reranking (train-only)

The next research-guided experiment is now complete.  It follows the
retrieval pattern used by recent query-aware KV-cache work: a compressed
query/key representation proposes a small candidate pool, then the original
query/key scores perform exact reranking inside that pool.  The native shadow
ABI also captured the corresponding older-candidate value vectors, so the
selected slots are bound to the values that a causal replay would read.

The authenticated value tensor is `[8, 32, 16, 16, 8, 128]` and its manifest is
`work/olmoe_q7/retrieval_episodic_full_visible_value_trace_2026-07-31/value_shards/qk-candidate-value-manifest.json`
(SHA-256 `8c13a25f1070fc0fba2b032fc7e84a24229aaecbb3fb35c55c51a20414865a1d`).
It is an evaluator artifact only; it does not alter production attention.

On record-held-out folds, rank-16 compressed scoring followed by exact
reranking in a six-candidate pool reaches **99.804% mean candidate membership
recall**, 100% p10 recall, 99.220% exact-top-4 rows, and 92.266% mean oracle
candidate mass retention.  A pool of eight is the exact-score ceiling (100%
recall).  The v2 report is
`work/olmoe_q7/retrieval_episodic_blockwise_qk_candidate_keys_2026-07-31/query_key_exact_rerank_v2.json`
(SHA-256 `6567ec5ec272c8d18ff7f661f5f917aae307780ab668e27507cc27e73e0d86e1`).

This is a strong candidate-locality result, not yet a full semantic gate:
the 0.78% non-exact rows still require a native intervention replay measuring
the complete hidden-state/logit trajectory.  The defensible next boundary is
to make the six-pool selector an explicit evaluator policy, replay it through
the native attention path, and fail closed on any output divergence.

The native replay has now been completed on all eight train records.  The
evaluator-only `forward_episodic_masked` ABI applies the rank-16/pool-6
allow-list, and the CPU kernel exact-reranks Q/K scores inside the pool before
reading cached values.  Compared with the identical unmasked native runtime,
the intervention reaches **99.6094% answer top-1 agreement**, **0.013111 mean
hidden relative L2**, **0.006376 mean logit relative L2**, and **+0.002404 mean
answer NLL delta** (maximum +0.018012).  All eight records pass the 10% hidden
and logit, 0.05 NLL, and 90% top-1 thresholds.  The report is
`work/olmoe_q7/retrieval_episodic_native_masked_replay_rank16_pool6.json`
(SHA-256
`2e6e017fc507e915cc96948deb53de0112062c208834b899643e04abe587baa7`).

This is a causal train-screen pass, not yet a generalization or production
claim.  The independent development split, traffic measurement, and
long-context scaling remain before the Milestone 2 semantic gate can be
promoted.

### Independent development replay

The selector has now been evaluated on a separately captured development
split.  The rank-16 PCA basis was fit only on the train artifacts; development
queries, keys, and Q/K bands were captured afterward by the native CPU shadow
route.  The six-slot pool achieves **99.8192%** candidate-pool and exact-
rerank membership recall (100% p10).  Full native masked replay reaches
**100% answer top-1 agreement**, **0.013236 mean hidden relative L2**,
**0.006520 mean logit relative L2**, and **−0.003980 mean answer NLL delta**
(maximum +0.000574) across all eight development records.  The authenticated
report is
`work/olmoe_q7/retrieval_episodic_development_replay_rank16_pool6.json`
(SHA-256
`0c5cb2273f63b930148c78070da68ae57bb821969a68e2a6a038ee7ac5d04bb6`).

This closes the train-to-development semantic gate for the bounded selector.
The separately authorized protected replay has now passed for the frozen
rank-16/pool-6 policy: all eight records reached 100% answer top-1 agreement,
0.009133 mean hidden relative L2, 0.004416 mean logit relative L2, and
−0.000460 mean answer NLL delta (maximum +0.005879). Candidate-pool and
exact-rerank recall were 99.8501% mean and 100% p10. The authenticated report
is
`work/olmoe_q7/retrieval_episodic_protected_replay_rank16_pool6_2026-08-03/replay.json`.
The selector remains disabled by default and is now eligible only for explicit
authenticated package opt-in. An authenticated opt-in package was then built
and native-generated successfully; its manifest SHA-256 is
`3c014679f2c626b68f73f8eebadbde8cb2421d4e174d8d69c27ebac774f3383c`. A one-token
`Hello` generation matched the ordinary package exactly (token ID 13, `,`).

### Long-context CPU scaling

The frozen rank-16/pool-6 selector was replayed on development record 0 at
512 and 2,048 positions using the native CPU package. The first 128 positions
are the authenticated selector window; later positions repeat the deterministic
token stream with episodic directives disabled, so this is a bounded-state
scaling measurement rather than a new retrieval claim. Answer quality remains
100% top-1 agreement (mean hidden/logit relative L2 0.008138/0.003919; NLL
delta −0.001520). The masked/unmasked logical-read fractions are 0.997079 at
512 positions and 0.999275 at 2,048, with 3.04/3.09 and 3.083/3.085 tokens/s
respectively. Peak resident memory was 6.28 GiB and did not grow with the
repeated context. The selector therefore has authenticated semantics and
bounded state, but not yet a demonstrated end-to-end speedup: most model work
is unchanged. Reproducible report:
`work/olmoe_q7/retrieval_episodic_long_context_rank16_pool6.json` (SHA-256
`fa205bd2ab4c91de27170247e7669f44c9def8bccea22d94558f8caa4b26bf71`).

The replay's native counters show the current tradeoff: mean logical
attention reads fall from 710,667,264 to 702,166,336 bytes (1.1962%), while
older candidate entries scored fall 7.3733%.  Episodic value reads and the
dominant local/projection/MLP work are unchanged, so this is not yet a claimed
end-to-end speedup.

## Documentation

- [Milestone and gate report](docs/milestone_report.md)
- [Current project and milestone status](docs/status.md)
- [Architecture](docs/architecture.md)
- [How Engram works](docs/how_engram_works.md)
- [Conversion pipeline](docs/conversion_pipeline.md)
- [Model format](docs/model_format.md)
- [Evaluation](docs/evaluation.md)
- [Limitations](docs/limitations.md)
- [Research log](docs/research_log.md)
