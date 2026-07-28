"""Frozen generation confirmation for authenticated native OLMoE packages."""

from __future__ import annotations

import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from engram.compiler.olmoe_native import validate_olmoe_native_package
from engram.models.olmoe import audit_olmoe_source
from engram.runtime.olmoe_native import OLMoENativePackageRuntime
from engram.tracing.olmoe import _prepare_transformers_imports
from engram.utils import atomic_json, sha256_file, sha256_json


def _load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"prompt suite line {line_number} is invalid JSON"
                ) from exc
            prompt = record.get("prompt") if isinstance(record, dict) else None
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"prompt suite line {line_number} has no prompt")
            prompts.append(prompt)
    if not prompts or len(prompts) != len(set(prompts)):
        raise ValueError("prompt suite must contain unique non-empty prompts")
    return prompts


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _agreement(reference: list[int], candidate: list[int]) -> float:
    length = max(len(reference), len(candidate))
    if length == 0:
        return 1.0
    return (
        sum(
            index < len(reference)
            and index < len(candidate)
            and reference[index] == candidate[index]
            for index in range(length)
        )
        / length
    )


def capture_olmoe_teacher_generation(
    *,
    model: str | Path,
    prompts: str | Path,
    out: str | Path,
    max_new_tokens: int = 4,
    device: str = "cpu",
    threads: int = 12,
) -> dict[str, Any]:
    """Capture greedy and teacher-forced top-1 tokens from untouched OLMoE."""

    if max_new_tokens <= 0 or threads <= 0:
        raise ValueError("max_new_tokens and threads must be positive")
    if device not in {"cpu", "cuda"}:
        raise ValueError("teacher device must be cpu or cuda")
    model_path = Path(model).expanduser().resolve()
    prompt_path = Path(prompts).expanduser().resolve()
    prompt_values = _load_prompts(prompt_path)
    audit = audit_olmoe_source(model_path)
    if audit.decision != "proceed_to_router_trace":
        raise ValueError("local OLMoE checkpoint failed exact source validation")
    try:
        import torch
        from tokenizers import Tokenizer

        _prepare_transformers_imports()
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for OLMoE teacher capture"
        ) from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for teacher capture but is unavailable")
    torch.set_num_threads(threads)
    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    encoded = [tokenizer.encode(prompt).ids for prompt in prompt_values]
    if any(not token_ids for token_ids in encoded):
        raise ValueError("teacher prompt tokenization produced an empty sequence")
    loaded = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    loaded.to(device)
    eos_token_id = int(loaded.config.eos_token_id)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for prompt, token_ids in zip(prompt_values, encoded, strict=True):
            inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
            output = loaded(input_ids=inputs, use_cache=True)
            teacher_forced = (
                torch.argmax(output.logits[0].float(), dim=-1).cpu().tolist()
            )
            generated: list[int] = []
            past = output.past_key_values
            next_token = int(torch.argmax(output.logits[0, -1].float()).item())
            for step in range(max_new_tokens):
                generated.append(next_token)
                if next_token == eos_token_id or step + 1 == max_new_tokens:
                    break
                output = loaded(
                    input_ids=torch.tensor(
                        [[next_token]], dtype=torch.long, device=device
                    ),
                    past_key_values=past,
                    use_cache=True,
                )
                past = output.past_key_values
                next_token = int(torch.argmax(output.logits[0, -1].float()).item())
            results.append(
                {
                    "prompt": prompt,
                    "input_ids": token_ids,
                    "teacher_forced_top1": teacher_forced,
                    "generated_token_ids": generated,
                    "generated_text": tokenizer.decode(generated),
                }
            )
    elapsed = time.perf_counter() - started
    del loaded
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    reference = {
        "schema_version": 1,
        "experiment": "olmoe_untouched_teacher_generation_reference",
        "source": {
            "model": str(model_path),
            "revision": audit.resolved_revision,
            "config_sha256": audit.config_sha256,
            "index_sha256": audit.index_sha256,
            "adapter": audit.adapter,
        },
        "prompt_suite": {
            "path": str(prompt_path),
            "sha256": sha256_file(prompt_path),
            "prompts": len(prompt_values),
            "input_identity": sha256_json(encoded),
        },
        "configuration": {
            "max_new_tokens": max_new_tokens,
            "greedy": True,
            "dtype": "bfloat16",
            "device": device,
            "threads": threads,
            "attention_implementation": "eager",
            "use_cache": True,
            "weights_modified": False,
        },
        "results": results,
        "summary": {
            "prompt_positions": sum(len(value) for value in encoded),
            "generated_tokens": sum(
                len(result["generated_token_ids"]) for result in results
            ),
            "elapsed_seconds": elapsed,
        },
    }
    atomic_json(Path(out), reference)
    return reference


def _validate_protocol(
    protocol: dict[str, Any],
    *,
    protocol_sha256: str,
    protocol_path: Path,
    prompt_sha256: str,
    reference_sha256: str,
    manifest_sha256: str,
    library_sha256: str,
) -> dict[str, Any]:
    if sha256_file(protocol_path) != protocol_sha256.lower():
        raise ValueError("OLMoE generation protocol authentication failed")
    thresholds = protocol.get("thresholds")
    expected_thresholds = {
        "minimum_prompts": 8,
        "minimum_prompt_positions": 60,
        "minimum_generated_reference_tokens": 32,
        "minimum_teacher_forced_top1_agreement": 0.90,
        "minimum_weighted_greedy_token_agreement": 0.90,
        "minimum_exact_prompt_fraction": 0.75,
    }
    if (
        protocol.get("schema_version") != 1
        or protocol.get("experiment") != "olmoe_native_package_generation_confirmation"
        or protocol.get("status") != "frozen_before_candidate_execution"
        or protocol.get("prompt_suite_sha256") != prompt_sha256
        or protocol.get("teacher_reference_sha256") != reference_sha256
        or protocol.get("package_manifest_sha256") != manifest_sha256
        or protocol.get("native_library_sha256") != library_sha256
        or protocol.get("max_new_tokens") != 4
        or thresholds != expected_thresholds
    ):
        raise ValueError("OLMoE generation protocol contract is invalid")
    return expected_thresholds


def _validate_teacher_source(
    reference: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    source = reference.get("source")
    if (
        not isinstance(source, dict)
        or source.get("revision") != protocol.get("source_revision")
        or source.get("config_sha256") != protocol.get("source_config_sha256")
        or source.get("index_sha256") != protocol.get("source_index_sha256")
    ):
        raise ValueError("teacher source identity differs from frozen protocol")
    model = source.get("model")
    shard_hashes = protocol.get("source_shard_sha256")
    if (
        not isinstance(model, str)
        or not isinstance(shard_hashes, dict)
        or not shard_hashes
    ):
        raise ValueError("teacher source shard contract is missing")
    model_path = Path(model)
    if sha256_file(model_path / "config.json") != protocol.get(
        "source_config_sha256"
    ) or sha256_file(model_path / "model.safetensors.index.json") != protocol.get(
        "source_index_sha256"
    ):
        raise ValueError("teacher source config or index authentication failed")
    index = _read_object(
        model_path / "model.safetensors.index.json",
        "teacher weight index",
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("teacher weight index has no weight map")
    indexed_shards = set(weight_map.values())
    if indexed_shards != set(shard_hashes):
        raise ValueError("teacher shard inventory differs from frozen protocol")
    descriptors = list(shard_hashes.items())
    for name, expected_hash in descriptors:
        if not isinstance(name, str) or not isinstance(expected_hash, str):
            raise ValueError("teacher shard descriptor is invalid")

    def authenticate(descriptor: tuple[str, str]) -> tuple[str, bool]:
        name, expected_hash = descriptor
        return name, sha256_file(model_path / name) == expected_hash

    with ThreadPoolExecutor(max_workers=min(6, len(descriptors))) as executor:
        results = executor.map(authenticate, descriptors)
    for name, passed in results:
        if not passed:
            raise ValueError(f"teacher source shard authentication failed: {name}")


def evaluate_native_olmoe_generation(
    *,
    package: str | Path,
    manifest_sha256: str,
    library: str | Path,
    prompts: str | Path,
    teacher_reference: str | Path,
    protocol: str | Path,
    protocol_sha256: str,
    out: str | Path,
    threads: int | None = None,
) -> dict[str, Any]:
    """Evaluate package-only native generation against a sealed teacher."""

    package_path = Path(package).expanduser().resolve()
    library_path = Path(library).expanduser().resolve()
    prompt_path = Path(prompts).expanduser().resolve()
    reference_path = Path(teacher_reference).expanduser().resolve()
    protocol_path = Path(protocol).expanduser().resolve()
    prompt_values = _load_prompts(prompt_path)
    reference = _read_object(reference_path, "teacher reference")
    protocol_value = _read_object(protocol_path, "generation protocol")
    prompt_hash = sha256_file(prompt_path)
    reference_hash = sha256_file(reference_path)
    library_hash = sha256_file(library_path)
    thresholds = _validate_protocol(
        protocol_value,
        protocol_sha256=protocol_sha256,
        protocol_path=protocol_path,
        prompt_sha256=prompt_hash,
        reference_sha256=reference_hash,
        manifest_sha256=manifest_sha256.lower(),
        library_sha256=library_hash,
    )
    reference_results = reference.get("results")
    if (
        reference.get("schema_version") != 1
        or reference.get("experiment") != "olmoe_untouched_teacher_generation_reference"
        or reference.get("configuration", {}).get("max_new_tokens") != 4
        or reference.get("configuration", {}).get("greedy") is not True
        or reference.get("configuration", {}).get("weights_modified") is not False
        or reference.get("prompt_suite", {}).get("sha256") != prompt_hash
        or not isinstance(reference_results, list)
        or len(reference_results) != len(prompt_values)
    ):
        raise ValueError("OLMoE teacher reference contract is invalid")
    _validate_teacher_source(reference, protocol_value)
    for teacher in reference_results:
        if (
            not isinstance(teacher, dict)
            or not isinstance(teacher.get("teacher_forced_top1"), list)
            or not isinstance(teacher.get("generated_token_ids"), list)
            or not teacher["teacher_forced_top1"]
            or not teacher["generated_token_ids"]
            or teacher["generated_token_ids"][0] != teacher["teacher_forced_top1"][-1]
        ):
            raise ValueError("teacher greedy and forced references are inconsistent")

    results: list[dict[str, Any]] = []
    all_teacher_forced: list[int] = []
    all_native_forced: list[int] = []
    longest_index = max(
        range(len(reference_results)),
        key=lambda index: len(reference_results[index].get("input_ids", [])),
    )
    replay_expected: list[int] | None = None
    replay_prompt = ""
    candidate_started = time.perf_counter()
    load_started = time.perf_counter()
    runtime = OLMoENativePackageRuntime(
        package_path,
        manifest_sha256=manifest_sha256,
        library=library_path,
        threads=threads,
    )
    cold_load_seconds = time.perf_counter() - load_started
    try:
        if (
            runtime.manifest.get("source", {}).get("revision")
            != protocol_value["source_revision"]
            or runtime.manifest["files"]["model/config.json"]["sha256"]
            != protocol_value["source_config_sha256"]
        ):
            raise ValueError("package source identity differs from frozen protocol")
        for index, (prompt, teacher) in enumerate(
            zip(prompt_values, reference_results, strict=True)
        ):
            if not isinstance(teacher, dict) or teacher.get("prompt") != prompt:
                raise ValueError("teacher reference prompt order is invalid")
            input_ids = runtime.tokenizer.encode(prompt).ids
            if input_ids != teacher.get("input_ids"):
                raise ValueError("package and teacher tokenization differ")
            teacher_forced = teacher.get("teacher_forced_top1")
            teacher_generated = teacher.get("generated_token_ids")
            if (
                not isinstance(teacher_forced, list)
                or len(teacher_forced) != len(input_ids)
                or not isinstance(teacher_generated, list)
                or not teacher_generated
            ):
                raise ValueError("teacher reference token arrays are invalid")

            runtime.reset()
            forced_started = time.perf_counter()
            native_forced = [
                runtime.runtime.forward([token_id]).next_token for token_id in input_ids
            ]
            forced_elapsed = time.perf_counter() - forced_started
            forced_position = runtime.runtime.position
            runtime.reset()
            started = time.perf_counter()
            candidate = runtime.generate(prompt, max_new_tokens=4)
            elapsed = time.perf_counter() - started
            native_generated = list(candidate["generated_token_ids"])
            expected_position = len(input_ids) + len(native_generated) - 1
            forced_agreement = _agreement(teacher_forced, native_forced)
            greedy_agreement = _agreement(
                teacher_generated,
                native_generated,
            )
            all_teacher_forced.extend(teacher_forced)
            all_native_forced.extend(native_forced)
            results.append(
                {
                    "prompt": prompt,
                    "input_ids": input_ids,
                    "prompt_tokens": len(input_ids),
                    "teacher_forced_top1": teacher_forced,
                    "native_forced_top1": native_forced,
                    "teacher_forced_top1_agreement": forced_agreement,
                    "teacher_generated_token_ids": teacher_generated,
                    "native_generated_token_ids": native_generated,
                    "teacher_generated_text": teacher.get("generated_text"),
                    "native_generated_text": candidate["completion"],
                    "greedy_token_agreement": greedy_agreement,
                    "exact_greedy_tokens": teacher_generated == native_generated,
                    "native_forced_elapsed_seconds": forced_elapsed,
                    "native_elapsed_seconds": elapsed,
                    "forced_cache_position": forced_position,
                    "greedy_cache_position": candidate["position"],
                    "expected_greedy_cache_position": expected_position,
                    "cache_positions_passed": (
                        forced_position == len(input_ids)
                        and candidate["position"] == expected_position
                    ),
                    "last_native_metrics": candidate["metrics"],
                }
            )
            if index == longest_index:
                replay_prompt = prompt
                replay_expected = native_generated

        assert replay_expected is not None
        runtime.reset()
        replay = runtime.generate(replay_prompt, max_new_tokens=4)
        replay_passed = (
            list(replay["generated_token_ids"]) == replay_expected
            and replay["position"]
            == len(runtime.tokenizer.encode(replay_prompt).ids)
            + len(replay_expected)
            - 1
        )
    finally:
        runtime.close()
    post_manifest = validate_olmoe_native_package(
        package_path,
        expected_manifest_sha256=manifest_sha256,
    )
    post_run_authentication_passed = (
        post_manifest == runtime.manifest and sha256_file(library_path) == library_hash
    )
    candidate_wall_seconds = time.perf_counter() - candidate_started

    prompt_positions = len(all_teacher_forced)
    generated_reference_tokens = sum(
        len(result["teacher_generated_token_ids"]) for result in results
    )
    teacher_forced_agreement = _agreement(
        all_teacher_forced,
        all_native_forced,
    )
    weighted_greedy_agreement = (
        sum(
            result["greedy_token_agreement"]
            * len(result["teacher_generated_token_ids"])
            for result in results
        )
        / generated_reference_tokens
    )
    exact_prompt_fraction = sum(
        result["exact_greedy_tokens"] for result in results
    ) / len(results)
    checks = {
        "prompt_count": len(results) >= thresholds["minimum_prompts"],
        "prompt_positions": (
            prompt_positions >= thresholds["minimum_prompt_positions"]
        ),
        "generated_reference_tokens": (
            generated_reference_tokens
            >= thresholds["minimum_generated_reference_tokens"]
        ),
        "teacher_forced_top1_agreement": (
            teacher_forced_agreement
            >= thresholds["minimum_teacher_forced_top1_agreement"]
        ),
        "weighted_greedy_token_agreement": (
            weighted_greedy_agreement
            >= thresholds["minimum_weighted_greedy_token_agreement"]
        ),
        "exact_prompt_fraction": (
            exact_prompt_fraction >= thresholds["minimum_exact_prompt_fraction"]
        ),
        "cache_positions": all(result["cache_positions_passed"] for result in results),
        "longest_prompt_reset_replay": replay_passed,
        "post_run_artifact_authentication": post_run_authentication_passed,
    }
    gate_passed = all(checks.values())
    report = {
        "schema_version": 1,
        "experiment": "olmoe_native_package_generation_confirmation",
        "status": (
            "frozen_confirmation_passed"
            if gate_passed
            else "frozen_confirmation_failed"
        ),
        "artifacts": {
            "package": str(package_path),
            "package_manifest_sha256": manifest_sha256.lower(),
            "native_library": str(library_path),
            "native_library_sha256": library_hash,
            "teacher_reference": str(reference_path),
            "teacher_reference_sha256": reference_hash,
            "protocol": str(protocol_path),
            "protocol_sha256": protocol_sha256.lower(),
            "prompt_suite": str(prompt_path),
            "prompt_suite_sha256": prompt_hash,
        },
        "configuration": {
            "max_new_tokens": 4,
            "greedy": True,
            "cpu_only_candidate": True,
            "transformers_model_shell_used_by_candidate": False,
            "attention_policy": "native_streaming_w16_c8_k4_sinks2",
            "all_sequences_within_exact_local_window": all(
                len(result["input_ids"]) + len(result["native_generated_token_ids"])
                <= 16
                for result in results
            ),
        },
        "results": results,
        "summary": {
            "prompts": len(results),
            "prompt_positions": prompt_positions,
            "generated_reference_tokens": generated_reference_tokens,
            "teacher_forced_top1_agreement": teacher_forced_agreement,
            "weighted_greedy_token_agreement": weighted_greedy_agreement,
            "exact_prompt_fraction": exact_prompt_fraction,
            "cold_authentication_and_load_seconds": cold_load_seconds,
            "total_native_forced_seconds": sum(
                result["native_forced_elapsed_seconds"] for result in results
            ),
            "total_native_greedy_seconds": sum(
                result["native_elapsed_seconds"] for result in results
            ),
            "candidate_wall_seconds_including_post_authentication": (
                candidate_wall_seconds
            ),
            "mean_native_seconds": sum(
                result["native_elapsed_seconds"] for result in results
            )
            / len(results),
            "total_native_seconds": sum(
                result["native_elapsed_seconds"] for result in results
            ),
        },
        "thresholds": thresholds,
        "checks": checks,
        "gate_passed": gate_passed,
        "decision": (
            "promote_package_generation_boundary_and_optimize_q7"
            if gate_passed
            else "stop_and_diagnose_first_native_teacher_divergence"
        ),
        "limitations": [
            "Token identity is measured; full native logits are not exported.",
            "All prompts stay within the exact local attention window, so this does not validate older-token retrieval quality.",
            "The suite is a frozen integration confirmation, not a broad language benchmark.",
            "The Paris prompt had been used by an earlier one-token smoke; the complete eight-prompt protocol and remaining prompts were not executed before freezing this protocol.",
        ],
    }
    atomic_json(Path(out), report)
    return report


__all__ = [
    "capture_olmoe_teacher_generation",
    "evaluate_native_olmoe_generation",
]
