"""Sustained generation evaluation for the optimized native BitNet package."""

from __future__ import annotations

import json
from pathlib import Path

from engram.runtime.native_bitnet import NativeBitNetRuntime
from engram.utils import atomic_json, sha256_file


def _longest_run(tokens: tuple[int, ...]) -> int:
    longest = current = 0
    previous = None
    for token in tokens:
        current = current + 1 if token == previous else 1
        longest = max(longest, current)
        previous = token
    return longest


def evaluate_native_bitnet_generation(
    *,
    package: str | Path,
    prompts: str | Path,
    out: str | Path,
    max_new_tokens: int = 16,
    mlp_library: str | Path | None = None,
    attention_library: str | Path | None = None,
    threads: int | None = None,
) -> dict:
    source = Path(prompts).resolve()
    records = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        prompt = value.get("prompt") if isinstance(value, dict) else None
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"generation prompt {line_number} is invalid")
        records.append(prompt)
    if not records:
        raise ValueError("generation prompt suite is empty")

    results = []
    with NativeBitNetRuntime(
        package,
        library=mlp_library,
        threads=threads,
        native_projections=True,
    ) as runtime:
        for prompt in records:
            result = runtime.generate_bounded(
                prompt,
                max_new_tokens=max_new_tokens,
                attention_library=attention_library,
            )
            unique_fraction = len(set(result.generated_tokens)) / len(
                result.generated_tokens
            )
            results.append(
                {
                    "prompt": prompt,
                    "prompt_tokens": len(result.prompt_tokens),
                    "generated_tokens": list(result.generated_tokens),
                    "generated_text": result.text,
                    "generated_count": len(result.generated_tokens),
                    "stopped_on_eos": result.stopped_on_eos,
                    "elapsed_seconds": result.elapsed_seconds,
                    "prefill_seconds": result.prefill_seconds,
                    "decode_seconds": result.decode_seconds,
                    "tokens_per_second": (
                        len(result.generated_tokens) / result.elapsed_seconds
                    ),
                    "unique_token_fraction": unique_fraction,
                    "longest_identical_token_run": _longest_run(
                        result.generated_tokens
                    ),
                    "attention_state_bytes": result.attention_state_bytes,
                }
            )
    report = {
        "schema_version": 1,
        "experiment": "native_bitnet_sustained_generation",
        "package": str(Path(package).resolve()),
        "prompt_suite": {
            "path": str(source),
            "sha256": sha256_file(source),
            "prompts": len(records),
        },
        "configuration": {
            "max_new_tokens": max_new_tokens,
            "greedy": True,
            "native_packed_projections": True,
            "bounded_attention": "W16/C8/K4/sinks2",
            "exact_last_row_vocabulary": True,
        },
        "results": results,
        "summary": {
            "completed_prompts": len(results),
            "eos_terminations": sum(row["stopped_on_eos"] for row in results),
            "mean_tokens_per_second": sum(
                row["tokens_per_second"] for row in results
            )
            / len(results),
            "maximum_identical_token_run": max(
                row["longest_identical_token_run"] for row in results
            ),
            "minimum_unique_token_fraction": min(
                row["unique_token_fraction"] for row in results
            ),
            "state_bytes_consistent": len(
                {row["attention_state_bytes"] for row in results}
            )
            == 1,
        },
    }
    atomic_json(Path(out), report)
    return report


__all__ = ["evaluate_native_bitnet_generation"]
