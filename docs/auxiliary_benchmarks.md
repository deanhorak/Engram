# Auxiliary external benchmarks

The project now freezes one public long-context retrieval corpus for an
independent, reproducible stress test. It is deliberately separate from the
project-specific protected gate.

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
