"""Deterministic local-source corpus construction for sparse distillation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from engram.models.inspection import resolve_model_path
from engram.utils import atomic_json, sha256_file


SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp", ".md", ".py", ".rst"}


def _source_files(inputs: Sequence[str | Path], output: Path) -> list[Path]:
    files: set[Path] = set()
    for value in inputs:
        source = Path(value).resolve()
        if source.is_file() and source.suffix.lower() in SOURCE_SUFFIXES:
            files.add(source)
        elif source.is_dir():
            files.update(
                path.resolve()
                for path in source.rglob("*")
                if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
            )
        else:
            raise ValueError(f"corpus input is not a supported file or directory: {source}")
    files.discard(output.resolve())
    if not files:
        raise ValueError("corpus inputs contain no supported source files")
    return sorted(files)


def build_distillation_corpus(
    model: str | Path,
    inputs: Sequence[str | Path],
    out: str | Path,
    *,
    sequence_length: int = 128,
    max_sequences: int = 128,
    minimum_tokens: int = 16,
) -> dict[str, Any]:
    """Tokenize local prose/code into round-robin, fixed-maximum-length records."""

    if sequence_length < 2 or max_sequences <= 0 or minimum_tokens < 2:
        raise ValueError("sequence_length/minimum_tokens must be >=2 and max_sequences positive")
    if minimum_tokens > sequence_length:
        raise ValueError("minimum_tokens must not exceed sequence_length")
    try:
        import transformers

        if getattr(transformers, "__path__", None):
            import transformers.utils as transformers_utils
            import transformers.utils.import_utils as transformers_imports

            if transformers_imports.is_sklearn_available():
                try:
                    import sklearn  # noqa: F401
                except ImportError:

                    def sklearn_unavailable() -> bool:
                        return False

                    transformers_imports.is_sklearn_available = sklearn_unavailable
                    transformers_utils.is_sklearn_available = sklearn_unavailable
        AutoTokenizer = transformers.AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install engram-lm[conversion] to build a corpus") from exc
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    files = _source_files(inputs, target)
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_path(model), local_files_only=True
    )
    tokenizer.model_max_length = max(int(tokenizer.model_max_length), 2**31)
    by_file: list[list[list[int]]] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        token_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        chunks = [
            [int(value) for value in token_ids[start : start + sequence_length]]
            for start in range(0, len(token_ids), sequence_length)
            if len(token_ids[start : start + sequence_length]) >= minimum_tokens
        ]
        if chunks:
            by_file.append(chunks)
    records: list[dict[str, Any]] = []
    seen_sequences: set[tuple[int, ...]] = set()
    duplicate_sequences = 0
    round_index = 0
    while len(records) < max_sequences:
        advanced = False
        for chunks in by_file:
            if round_index < len(chunks):
                advanced = True
                chunk = chunks[round_index]
                fingerprint = tuple(chunk)
                if fingerprint in seen_sequences:
                    duplicate_sequences += 1
                    continue
                seen_sequences.add(fingerprint)
                records.append({"input_ids": chunk})
                if len(records) >= max_sequences:
                    break
        if not advanced:
            break
        round_index += 1
    if not records:
        raise ValueError("corpus sources did not produce any sufficiently long sequences")
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    temporary.replace(target)
    report = {
        "schema_version": 1,
        "model": str(model),
        "sequence_length": sequence_length,
        "minimum_tokens": minimum_tokens,
        "max_sequences": max_sequences,
        "sequences": len(records),
        "unique_sequences": len(seen_sequences),
        "duplicate_sequences_skipped": duplicate_sequences,
        "deduplication": "exact_token_sequence_first_occurrence",
        "token_positions": sum(len(record["input_ids"]) for record in records),
        "source_files": len(files),
        "dataset_path": str(target.resolve()),
        "dataset_sha256": sha256_file(target),
    }
    atomic_json(target.with_suffix(target.suffix + ".manifest.json"), report)
    return report


__all__ = ["build_distillation_corpus"]
