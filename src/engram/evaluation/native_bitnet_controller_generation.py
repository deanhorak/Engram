"""Prompt-suite parity for controller-driven incremental BitNet generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engram.controller import FactorizedRecurrentController
from engram.runtime.native_bitnet import NativeBitNetRuntime
from engram.utils import atomic_json, sha256_file


def _load_prompts(path: Path) -> list[str]:
    prompts = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        prompt = value.get("prompt") if isinstance(value, dict) else None
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"generation prompt {line_number} is invalid")
        prompts.append(prompt)
    if not prompts:
        raise ValueError("generation prompt suite is empty")
    return prompts


def _token_agreement(reference: tuple[int, ...], candidate: tuple[int, ...]) -> float:
    length = max(len(reference), len(candidate))
    if not length:
        return 1.0
    return sum(
        index < len(reference)
        and index < len(candidate)
        and reference[index] == candidate[index]
        for index in range(length)
    ) / length


def evaluate_native_bitnet_controller_generation(
    *,
    package: str | Path,
    controller: str | Path,
    prompts: str | Path,
    out: str | Path,
    max_new_tokens: int = 4,
    mlp_library: str | Path | None = None,
    attention_library: str | Path | None = None,
    threads: int | None = None,
) -> dict[str, Any]:
    """Compare decoder-scaffold and controller-driven bounded generation."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    prompt_path = Path(prompts).resolve()
    records = _load_prompts(prompt_path)
    controller_runtime = FactorizedRecurrentController.load(controller)
    results = []
    with NativeBitNetRuntime(
        package,
        library=mlp_library,
        threads=threads,
        native_projections=True,
    ) as runtime:
        for prompt in records:
            tokens = runtime.encode(prompt)
            baseline = runtime.generate_tokens_bounded(
                tokens,
                max_new_tokens=max_new_tokens,
                attention_library=attention_library,
            )
            candidate = runtime.generate_tokens_controller_bounded(
                tokens,
                controller_runtime,
                max_new_tokens=max_new_tokens,
                attention_library=attention_library,
            )
            expected_positions = len(tokens) + len(candidate.generated_tokens) - 1
            results.append(
                {
                    "prompt": prompt,
                    "prompt_tokens": len(tokens),
                    "baseline_tokens": list(baseline.generated_tokens),
                    "controller_tokens": list(candidate.generated_tokens),
                    "baseline_text": baseline.text,
                    "controller_text": candidate.text,
                    "exact_token_parity": (
                        baseline.generated_tokens == candidate.generated_tokens
                    ),
                    "token_agreement": _token_agreement(
                        baseline.generated_tokens,
                        candidate.generated_tokens,
                    ),
                    "baseline_seconds": baseline.elapsed_seconds,
                    "controller_runtime_seconds": candidate.elapsed_seconds,
                    "controller_math_seconds": candidate.controller_seconds,
                    "controller_state_bytes": candidate.controller_state_bytes,
                    "attention_tokens_seen": candidate.attention_tokens_seen,
                    "expected_attention_tokens_seen": expected_positions,
                    "cache_position_passed": (
                        candidate.attention_tokens_seen == expected_positions
                    ),
                    "decoder_layer_forward_calls": (
                        candidate.decoder_layer_forward_calls
                    ),
                }
            )

    total_reference_tokens = sum(len(row["baseline_tokens"]) for row in results)
    weighted_agreement = (
        sum(
            row["token_agreement"] * len(row["baseline_tokens"])
            for row in results
        )
        / total_reference_tokens
        if total_reference_tokens
        else 1.0
    )
    exact_prompt_fraction = sum(
        row["exact_token_parity"] for row in results
    ) / len(results)
    thresholds = {
        "minimum_prompts": 8,
        "minimum_generated_reference_tokens": 32,
        "minimum_weighted_token_agreement": 0.9,
        "minimum_exact_prompt_fraction": 0.75,
    }
    checks = {
        "prompt_count": len(results) >= thresholds["minimum_prompts"],
        "generated_reference_tokens": (
            total_reference_tokens
            >= thresholds["minimum_generated_reference_tokens"]
        ),
        "weighted_token_agreement": (
            weighted_agreement
            >= thresholds["minimum_weighted_token_agreement"]
        ),
        "exact_prompt_fraction": (
            exact_prompt_fraction
            >= thresholds["minimum_exact_prompt_fraction"]
        ),
        "cache_positions": all(
            row["cache_position_passed"] for row in results
        ),
        "decoder_layers_bypassed": all(
            row["decoder_layer_forward_calls"] == 0 for row in results
        ),
    }
    gate_passed = all(checks.values())
    report = {
        "schema_version": 1,
        "experiment": "native_bitnet_controller_incremental_generation",
        "status": (
            "frozen_controller_generation_confirmation"
            if gate_passed
            else "controller_generation_development_result"
        ),
        "package": str(Path(package).resolve()),
        "controller": {
            "path": str(Path(controller).resolve()),
            **controller_runtime.metadata(),
        },
        "prompt_suite": {
            "path": str(prompt_path),
            "sha256": sha256_file(prompt_path),
            "prompts": len(records),
        },
        "configuration": {
            "max_new_tokens": max_new_tokens,
            "greedy": True,
            "semantic_operator": "native_packed_bitnet_phase_stream",
            "episodic_operator": "native_streaming_w16_c8_k4_sinks2",
            "native_packed_attention_projections": True,
            "controller_correction_enabled": False,
            "native_embedding_lookup": True,
            "native_rms_norm": True,
            "native_rope": True,
            "native_vocabulary_argmax": True,
        },
        "results": results,
        "summary": {
            "prompts": len(results),
            "generated_reference_tokens": total_reference_tokens,
            "weighted_token_agreement": weighted_agreement,
            "exact_prompt_fraction": exact_prompt_fraction,
            "mean_controller_runtime_seconds": sum(
                row["controller_runtime_seconds"] for row in results
            )
            / len(results),
            "mean_controller_math_seconds": sum(
                row["controller_math_seconds"] for row in results
            )
            / len(results),
            "maximum_controller_state_bytes": max(
                row["controller_state_bytes"] for row in results
            ),
        },
        "thresholds": thresholds,
        "checks": checks,
        "gate_passed": gate_passed,
        "scope": {
            "decoder_layer_forward_used": False,
            "persistent_native_attention_cache_used": True,
            "absolute_rope_positions_advanced": True,
            "source_mlp_tensors_loaded": False,
            "torch_module_shell_still_used": True,
            "fully_native_cpp_controller_used": False,
            "torch_embedding_used": False,
            "torch_rms_norm_used": False,
            "torch_rope_used": False,
            "torch_vocabulary_projection_used": False,
        },
        "decision": (
            "incremental_controller_generation_pass_cpp_stage_orchestration_next"
            if gate_passed
            else "diagnose_controller_generation_divergence"
        ),
    }
    atomic_json(Path(out), report)
    return report


__all__ = [
    "_token_agreement",
    "evaluate_native_bitnet_controller_generation",
]
