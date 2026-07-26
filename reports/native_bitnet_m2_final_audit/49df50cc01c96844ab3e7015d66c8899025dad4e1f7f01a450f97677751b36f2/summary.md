# Native BitNet Milestone 2 final-attempt adjudication

Date: **2026-07-26**

Decision: **the native-BitNet practical semantic-memory gate passed by
postmortem adjudication**. This is not a pristine final-runner pass and is not
a claim that dense-Llama conversion or every broader Milestone 2 deliverable
is complete.

## What happened

The single authorized 8-sequence/256-position final attempt ran the frozen
CPU-only DIP route across all 30 MLPs with no dense fallback. The raw evaluator
completed and passed every frozen threshold. The original wrapper then marked
the consumed attempt `error` because its token verifier compared incompatible
hash contracts:

- the protocol stored full token sequences hashed with the canonical
  `input_ids` object envelope;
- the evaluator stored the first 33 scored tokens hashed as a bare list.

The original error result and opened marker were preserved. A separate
postmortem adjudicator ran no model and no evaluator. It reconstructed both
historical hash schemas, verified the full-sequence identities and scored
prefixes, and checked the frozen authorization, implementation identity,
artifacts, raw primitive evidence, parity, and evaluator attestations.

## Final raw measurements

| Measure | Result | Frozen threshold |
|---|---:|---:|
| Teacher-to-student KL | 0.0040412880 | <= 0.05 |
| Teacher top-1 agreement | 0.98828125 | >= 0.90 |
| NLL delta | +0.0048289299 | <= +0.05 |
| Final-hidden relative L2 | 0.0477494113 | <= 0.10 |
| Mean active-record fraction | 0.2138000677 | <= 0.25 |
| Modeled physical cold traffic | 0.4113713394 | <= 0.45 |
| Global micro candidate recall | 0.9994058295 | >= 0.95 |
| Worst-layer mean candidate recall | 0.9939428640 | >= 0.95 |

The raw report covers 7,680 layer rows. The worst individual layer/token
modeled traffic is 0.4499903549 of dense ideal Q4, and the worst complete token
is 0.4210874003. Python/native parity passed, the serialized index was
reloaded, execution was CPU-only, and all MLP layers were substituted.

Sparse execution took 295.3364 seconds versus 257.9552 seconds dense, or
1.1449x dense. This is **not a speedup**. Latency was measured and disclosed
but was not a frozen upper-bound gate.

## Evidence

- [Original consumed-attempt result](0fdce5a3eb2eaaec5e3b47587d7693bfdc0fed6fde53e3a40a74b8d2a76d4aa3.result.json)
- [Opened marker](opened.json)
- [Raw native causal evaluator report](0fdce5a3eb2eaaec5e3b47587d7693bfdc0fed6fde53e3a40a74b8d2a76d4aa3.native-causal.raw.json)
- [Archived authorization](0fdce5a3eb2eaaec5e3b47587d7693bfdc0fed6fde53e3a40a74b8d2a76d4aa3.authorization.json)
- [Prospective evidence seal](0fdce5a3eb2eaaec5e3b47587d7693bfdc0fed6fde53e3a40a74b8d2a76d4aa3.evidence-seal.json)
- [No-model postmortem adjudication](0fdce5a3eb2eaaec5e3b47587d7693bfdc0fed6fde53e3a40a74b8d2a76d4aa3.adjudication.json)

## Limits on the claim

- The original one-shot result remains an execution error; it was not
  rewritten into a pass.
- The raw report was prospectively hash-sealed about 13 minutes after that
  error. The original result did not contemporaneously bind the raw report,
  so evidence custody is weaker than a clean independently sealed run.
- The holdout was a checked-in plaintext fixture. Its separation was
  procedural/honor-system-based, not cryptographic.
- Artifact and native-library bindings are host-specific. Independent-host
  reproduction has not been demonstrated.
- The traffic result is deterministic serialized-layout/cache-line accounting,
  not measured hardware DRAM traffic.
- The confirmation scale is 8 sequences by 32 scored positions. It is not a
  broad language-quality, workload, or hardware study.
- The adjudicated pass applies to the native-BitNet semantic-memory gate. It
  does not establish quality-preserving dense-Llama conversion, runtime
  speedup, or blanket completion of all Milestone 2 integration work.
