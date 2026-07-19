from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import psutil

from engram.runtime.reference import EngramRuntime
from engram.utils import sha256_file


def validate_package(path: str | Path) -> dict[str, Any]:
    try:
        runtime = EngramRuntime(path)
    except (OSError, ValueError) as error:
        return {"valid": False, "errors": [str(error)], "smoke_tokens": []}
    root = Path(path)
    errors = []
    for relative, descriptor in runtime.manifest["files"].items():
        file_path = root / relative
        if not file_path.is_file() or sha256_file(file_path) != descriptor["sha256"]:
            errors.append(relative)
    first = runtime.generate_tokens([1, 2, 3], max_tokens=4)[0]
    second = EngramRuntime(path).generate_tokens([1, 2, 3], max_tokens=4)[0]
    if first != second:
        errors.append("deterministic_generation")
    return {"valid": not errors, "errors": errors, "smoke_tokens": first}


def benchmark_runtime(
    path: str | Path, *, tokens: int = 32, bypass_transition_cache: bool = True
) -> dict[str, Any]:
    runtime = EngramRuntime(path)
    runtime.cache.set_bypass(bypass_transition_cache)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    started = time.perf_counter_ns()
    generated, metrics = runtime.generate_tokens([1, 2, 3], max_tokens=tokens)
    elapsed = time.perf_counter_ns() - started
    return {
        "status": "fixture_runtime_measurement" if runtime.manifest["fixture_only"] else "local_model_runtime_measurement",
        "generated_tokens": len(generated),
        "elapsed_ns": elapsed,
        "decode_tokens_per_second": tokens / (elapsed / 1e9),
        "rss_delta_bytes": process.memory_info().rss - rss_before,
        "mean_cycles": sum(item.cycles for item in metrics) / len(metrics),
        "mean_semantic_records": sum(item.semantic_records for item in metrics) / len(metrics),
        "mean_semantic_proxy_records": sum(item.semantic_proxy_records for item in metrics) / len(metrics),
        "mean_semantic_probed_clusters": sum(item.semantic_probed_clusters for item in metrics) / len(metrics),
        "mean_vocabulary_candidates": sum(item.vocabulary_candidates for item in metrics) / len(metrics),
        "mean_vocabulary_proxy_records": sum(item.vocabulary_proxy_records for item in metrics) / len(metrics),
        "mean_vocabulary_probed_clusters": sum(item.vocabulary_probed_clusters for item in metrics) / len(metrics),
        "transition_cache_bypassed": bypass_transition_cache,
        "measured": ["elapsed_ns", "rss_delta_bytes"],
        "estimated": ["active semantic record payload bytes"],
        "quality_claim": None,
    }
