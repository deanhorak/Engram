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

### Milestone 2 ledger

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
attention, vindicates the Q7 semantic substitution underlying the formal M2
pass, and makes **Milestone 3 bounded attention** the remaining OLMoE blocker.

W128 is a diagnostic, not a deployable solution: it reads **100%** of dense
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
not all head-wise allocation. The next justified direction is value- or
sensitivity-guided selection, or a dynamic allocation policy. Milestone 2
remains passed; Milestone 3 attention remains blocked. See the
[sustained-context evidence and attribution report](reports/olmoe_q7_sustained_context_2026-07-28/summary.md).

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
quality gates after position 31. Both six-record screens are development
evidence from an already consumed corpus, not fresh Milestone 3
confirmations. The fixed attention-mass result motivates value/sensitivity
ranking or dynamic head allocation; it does not qualify an OLMoE attention
package or close every head-wise design.

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

## Documentation

- [Current project and milestone status](docs/status.md)
- [Architecture](docs/architecture.md)
- [How Engram works](docs/how_engram_works.md)
- [Conversion pipeline](docs/conversion_pipeline.md)
- [Model format](docs/model_format.md)
- [Evaluation](docs/evaluation.md)
- [Limitations](docs/limitations.md)
- [Research log](docs/research_log.md)
