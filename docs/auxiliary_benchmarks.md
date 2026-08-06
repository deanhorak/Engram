# Auxiliary external benchmarks

## Final conclusion

These auxiliary benchmarks document reproducibility and failure boundaries only. They do not show
that Engram improves on or equals current LLM technology, and none authorizes the protected Engram
gate or a production-quality claim.

The project now freezes one public long-context retrieval corpus for an
independent, reproducible stress test. It is deliberately separate from the
project-specific protected gate.

## BEIR SciFact

The hybrid sidecar's first public natural-language retrieval boundary uses BEIR
SciFact. The official `scifact.zip` archive has MD5
`5f7d1de60b170fc8027bb7898e2efca1` and locally verified SHA-256
`536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`.
The extracted file SHA-256 values are:

- `corpus.jsonl`: `dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6`
- `queries.jsonl`: `8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313`
- `qrels/test.tsv`: `0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c`

The run uses all 5,183 corpus records and the 300 test queries with positive
judgments (339 binary-positive pairs). It is a public auxiliary retrieval and host
benchmark; it is not the protected Engram semantic gate. The checked-in reports
record the source URL, archive hashes, extracted hashes, encoder identity, thresholds,
and per-query results. SciFact is distributed as part of the BEIR benchmark under the
license recorded by its public dataset card.

## LongEmbed Passkey

The selected source is the `passkey` configuration of
[dwzhu/LongEmbed](https://huggingface.co/datasets/dwzhu/LongEmbed), pinned to
Git revision
`10039a580487dacecf79db69166e17ace3ede392`. The exact file sizes, Git objects,
LFS object IDs, and SHA-256 values are frozen in
[`auxiliary_longembed_passkey.json`](auxiliary_longembed_passkey.json).

The downloaded working copy is intentionally ignored by Git at:

```text
work/auxiliary/longembed_passkey_10039a580487dacecf79db69166e17ace3ede392/passkey/
```

Recreate it from the frozen revision with Git LFS:

```bash
git clone --filter=blob:none --no-checkout \
  https://huggingface.co/datasets/dwzhu/LongEmbed.git /tmp/engram-longembed
git -C /tmp/engram-longembed checkout \
  10039a580487dacecf79db69166e17ace3ede392 -- \
  passkey/corpus.jsonl passkey/qrels.jsonl passkey/queries.jsonl
git -C /tmp/engram-longembed lfs pull --include='passkey/*'
sha256sum /tmp/engram-longembed/passkey/*.jsonl
```

The frozen files contain 800 documents, 400 queries, and 400 relevance rows
over context lengths 256 through 32,768 tokens. Each context length has 100
documents and 50 queries.

This corpus is an auxiliary retrieval stress test only. It is public, uses a
different record and query schema, and is not tokenized or traced against the
Engram source model. It must not be used to claim that the separately
protected Engram gate passed. Running it through Engram requires an adapter
that converts the corpus/query/qrels representation and fresh native teacher
traces; the adapter must remain evaluation-only and must not feed selector
training.

The benchmark is frozen by upstream revision and file hashes, not by secrecy.
Any report using it must include the manifest, source revision, and all three
file hashes.
