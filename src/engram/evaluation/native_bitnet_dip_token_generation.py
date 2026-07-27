"""Authenticated native-token confirmation for packaged BitNet DIP inference."""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

from engram.compiler.native_bitnet import (
    NATIVE_BITNET_DIP_DERIVED_MANIFEST_SHA256,
    NATIVE_BITNET_DIP_OPERATOR,
    NATIVE_BITNET_M2_ADJUDICATION_SHA256,
    NATIVE_BITNET_M2_BASE_ARTIFACT_SHA256,
    NATIVE_BITNET_M2_COORDINATE_INDEX_SHA256,
    NATIVE_BITNET_M2_PACKAGE_MANIFEST_SHA256,
    NATIVE_BITNET_M2_POLICY_MANIFEST_SHA256,
)
from engram.evaluation.native_bitnet_parity import (
    _disable_broken_optional_transformers_dependencies,
)
from engram.models.native_bitnet import native_bitnet_repack_traffic
from engram.runtime.native_bitnet import validate_native_bitnet_package
from engram.semantic.native_bitnet_dip_index import (
    load_native_bitnet_dip_index,
)
from engram.utils import atomic_json, sha256_file


_CANONICAL_PROMPT_SUITE_SHA256 = (
    "dd38c4ce92045d333edd572f23bad3f41f331393edfc58796a7cf2af01554fd2"
)
_CANONICAL_DENSE_REFERENCE_SHA256 = (
    "3078cd2c36d54fa55380e2550f0176a37027390699ef248257785bc203665e02"
)
_TRUSTED_NATIVE_EXECUTABLE_SHA256 = (
    "29526c9838ea484d8a21887dafeaba99a57348e7377e0de4138e0631dde10fad"
)
_REFERENCE_THRESHOLDS = {
    "minimum_prompts": 8,
    "minimum_generated_reference_tokens": 32,
    "minimum_weighted_token_agreement": 0.9,
    "minimum_exact_prompt_fraction": 0.75,
}
_MAXIMUM_MEAN_ACTIVE_FRACTION = 0.25
_MAXIMUM_MEAN_TRAFFIC_FRACTION = 0.45


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_prompts(path: Path) -> list[str]:
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
                raise ValueError(
                    f"prompt suite line {line_number} has no prompt"
                )
            prompts.append(prompt)
    if not prompts or len(set(prompts)) != len(prompts):
        raise ValueError("prompt suite must contain unique non-empty prompts")
    return prompts


def _parse_native_metrics(stderr: str) -> dict[str, int | float | str]:
    integer_fields = {
        "positions",
        "stage_calls",
        "semantic_calls",
        "semantic_rows",
        "selected_records",
        "semantic_kernel_cache_line_bytes",
        "semantic_global_metadata_bytes",
        "semantic_cache_line_bytes",
        "reset_verified",
        "reset_counters_zeroed",
        "replay_metrics_match",
    }
    float_fields = {"semantic_seconds", "attention_seconds"}
    fields: dict[str, int | float | str] = {}
    for item in stderr.split():
        if "=" not in item:
            continue
        name, value = item.split("=", 1)
        if name not in integer_fields | float_fields | {"semantic_backend"}:
            continue
        if name in fields:
            raise ValueError(f"duplicate native token metric: {name}")
        if name in integer_fields:
            parsed: int | float | str = int(value)
        elif name in float_fields:
            parsed = float(value)
            if not math.isfinite(parsed) or parsed < 0:
                raise ValueError(f"native token timing is invalid: {name}")
        else:
            parsed = value
        fields[name] = parsed
    required = integer_fields | float_fields | {"semantic_backend"}
    if set(fields) != required:
        missing = sorted(required - set(fields))
        raise ValueError(f"native token metrics are incomplete: {missing}")
    return fields


def _validate_dense_reference(
    reference: dict[str, Any],
    *,
    prompts: list[str],
    prompt_sha256: str,
    max_new_tokens: int,
) -> dict[str, list[int]]:
    if (
        reference.get("schema_version") != 1
        or reference.get("experiment")
        != "native_bitnet_controller_incremental_generation"
        or reference.get("status") != "frozen_controller_generation_confirmation"
        or reference.get("gate_passed") is not True
        or reference.get("thresholds") != _REFERENCE_THRESHOLDS
    ):
        raise ValueError("generation reference is not the frozen dense baseline")
    configuration = reference.get("configuration")
    required_configuration = {
        "controller_correction_enabled": False,
        "episodic_operator": "native_streaming_w16_c8_k4_sinks2",
        "greedy": True,
        "max_new_tokens": max_new_tokens,
        "native_embedding_lookup": True,
        "native_packed_attention_projections": True,
        "native_rms_norm": True,
        "native_rope": True,
        "native_vocabulary_argmax": True,
        "semantic_operator": "native_packed_bitnet_phase_stream",
    }
    if not isinstance(configuration, dict) or any(
        configuration.get(name) != value
        for name, value in required_configuration.items()
    ):
        raise ValueError("generation reference runtime provenance is invalid")
    scope = reference.get("scope")
    required_scope = {
        "absolute_rope_positions_advanced": True,
        "decoder_layer_forward_used": False,
        "persistent_native_attention_cache_used": True,
        "source_mlp_tensors_loaded": False,
        "torch_embedding_used": False,
        "torch_rms_norm_used": False,
        "torch_rope_used": False,
        "torch_vocabulary_projection_used": False,
    }
    if not isinstance(scope, dict) or any(
        scope.get(name) is not value for name, value in required_scope.items()
    ):
        raise ValueError("generation reference scope is invalid")
    checks = reference.get("checks")
    if (
        not isinstance(checks, dict)
        or set(checks)
        != {
            "cache_positions",
            "decoder_layers_bypassed",
            "exact_prompt_fraction",
            "generated_reference_tokens",
            "prompt_count",
            "weighted_token_agreement",
        }
        or any(value is not True for value in checks.values())
    ):
        raise ValueError("generation reference checks are not all passing")
    prompt_descriptor = reference.get("prompt_suite")
    reference_results = reference.get("results")
    if (
        not isinstance(prompt_descriptor, dict)
        or prompt_descriptor.get("sha256") != prompt_sha256
        or prompt_descriptor.get("prompts") != len(prompts)
        or not isinstance(reference_results, list)
        or len(reference_results) != len(prompts)
    ):
        raise ValueError("generation reference does not match the prompt suite")
    references: dict[str, list[int]] = {}
    for result in reference_results:
        if not isinstance(result, dict):
            raise ValueError("generation reference result is malformed")
        prompt = result.get("prompt")
        tokens = result.get("baseline_tokens")
        if (
            not isinstance(prompt, str)
            or prompt in references
            or not isinstance(tokens, list)
            or len(tokens) != max_new_tokens
            or any(
                isinstance(token, bool) or not isinstance(token, int)
                for token in tokens
            )
        ):
            raise ValueError("generation reference tokens are malformed")
        references[prompt] = tokens
    if list(references) != prompts:
        raise ValueError("generation reference prompt order differs from the suite")
    return references


def _expected_traffic(
    *,
    index_path: Path,
    layers: int,
    hidden: int,
    intermediate: int,
    positions: int,
    selected_records: int,
) -> dict[str, int]:
    cache_line = 64
    packed_record = (hidden + 4) // 5
    coordinate_stride = (
        ((intermediate + 4) // 5 + cache_line - 1) // cache_line
    ) * cache_line
    gain_bytes = (
        (intermediate * 2 + cache_line - 1) // cache_line
    ) * cache_line
    norm_value_bytes = 2 if hidden <= 65535 else 4
    norm_bytes = (
        (intermediate * norm_value_bytes + cache_line - 1) // cache_line
    ) * cache_line
    fixed_per_position = 0
    minimum_selected_per_position = 0
    maximum_selected_per_position = 0
    with load_native_bitnet_dip_index(index_path) as index:
        if len(index.layers) != layers:
            raise ValueError("DIP index layer count changed during evaluation")
        for layer in index.layers:
            policy = layer.policy
            completion = 2 * policy.candidate_count * packed_record
            fixed_per_position += (
                2 * policy.input_coordinates * coordinate_stride
                + completion
                + gain_bytes
                + norm_bytes
                + 4 * cache_line
            )
            minimum_selected_per_position += policy.minimum_top_k
            maximum_selected_per_position += policy.maximum_top_k
    kernel = fixed_per_position * positions + selected_records * packed_record
    base = native_bitnet_repack_traffic(
        hidden,
        intermediate,
        layer_count=layers,
        cache_line_bytes=cache_line,
    )
    base_global = int(base["header_cache_aligned_bytes"]) + int(
        base["directory_cache_aligned_bytes"]
    )
    index_global = 128 + (
        (layers * 32 + cache_line - 1) // cache_line
    ) * cache_line
    global_metadata = positions * (base_global + index_global)
    return {
        "kernel_cache_line_bytes": kernel,
        "global_metadata_bytes": global_metadata,
        "complete_modelled_cold_bytes": kernel + global_metadata,
        "minimum_selected_records": (
            positions * minimum_selected_per_position
        ),
        "maximum_selected_records": (
            positions * maximum_selected_per_position
        ),
    }


def evaluate_native_bitnet_dip_token_generation(
    package: str | Path,
    executable: str | Path,
    prompts: str | Path,
    reference_report: str | Path,
    out: str | Path,
    *,
    package_manifest_sha256: str,
    executable_sha256: str,
    max_new_tokens: int = 4,
    threads: int = 12,
    verify_reset: bool = True,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """Run the fixed, authenticated non-holdout DIP confirmation protocol."""

    if max_new_tokens <= 0 or threads <= 0 or timeout_seconds <= 0:
        raise ValueError("token count, threads, and timeout must be positive")
    package_input = Path(package)
    binary_input = Path(executable)
    if package_input.is_symlink():
        raise ValueError("native BitNet DIP package root cannot be a symlink")
    if binary_input.is_symlink():
        raise ValueError("native token executable cannot be a symlink")
    root = package_input.resolve()
    binary = binary_input.resolve()
    prompt_path = Path(prompts).resolve()
    reference_path = Path(reference_report).resolve()
    output_path = Path(out).resolve()
    if not binary.is_file():
        raise ValueError(f"native token executable is missing: {binary}")

    expected_package_sha256 = package_manifest_sha256.lower()
    expected_executable_sha256 = executable_sha256.lower()
    if expected_package_sha256 != NATIVE_BITNET_DIP_DERIVED_MANIFEST_SHA256:
        raise ValueError("package manifest is not the promoted DIP trust root")
    if expected_executable_sha256 != _TRUSTED_NATIVE_EXECUTABLE_SHA256:
        raise ValueError("native executable is not the reviewed build")
    manifest_path = root / "manifest.json"
    if sha256_file(manifest_path) != expected_package_sha256:
        raise ValueError("native BitNet DIP package manifest SHA-256 mismatch")
    if sha256_file(binary) != expected_executable_sha256:
        raise ValueError("native token executable SHA-256 mismatch")
    package_validation = validate_native_bitnet_package(root)
    if package_validation.get("valid") is not True:
        raise ValueError(
            "native BitNet DIP package is invalid: "
            + "; ".join(package_validation.get("errors", []))
        )
    manifest = _read_json_object(manifest_path, "package manifest")
    semantic = manifest.get("semantic_memory")
    trusted_semantic_binding = {
        "operator": NATIVE_BITNET_DIP_OPERATOR,
        "dense_fallback": False,
        "all_mlp_layers_substituted": True,
        "source_package_manifest_sha256": (
            NATIVE_BITNET_M2_PACKAGE_MANIFEST_SHA256
        ),
        "source_artifact_sha256": NATIVE_BITNET_M2_BASE_ARTIFACT_SHA256,
        "sha256": NATIVE_BITNET_M2_COORDINATE_INDEX_SHA256,
        "policy_manifest_sha256": NATIVE_BITNET_M2_POLICY_MANIFEST_SHA256,
        "adjudication_sha256": NATIVE_BITNET_M2_ADJUDICATION_SHA256,
    }
    if not isinstance(semantic, dict) or any(
        semantic.get(name) != value
        for name, value in trusted_semantic_binding.items()
    ):
        raise ValueError("package is not the approved fail-closed DIP derivation")
    model = manifest.get("model", {})
    layers = int(model.get("num_hidden_layers", 0))
    hidden = int(model.get("hidden_size", 0))
    intermediate = int(model.get("intermediate_size", 0))
    if layers <= 0 or hidden <= 0 or intermediate <= 0:
        raise ValueError("native BitNet package dimensions are invalid")
    index_path = root / semantic["path"]

    prompt_sha256 = sha256_file(prompt_path)
    reference_sha256 = sha256_file(reference_path)
    if prompt_sha256 != _CANONICAL_PROMPT_SUITE_SHA256:
        raise ValueError("prompt suite is not the canonical DIP integration suite")
    if reference_sha256 != _CANONICAL_DENSE_REFERENCE_SHA256:
        raise ValueError("generation reference SHA-256 is not canonical")
    prompt_values = _read_prompts(prompt_path)
    reference = _read_json_object(reference_path, "generation reference")
    references = _validate_dense_reference(
        reference,
        prompts=prompt_values,
        prompt_sha256=prompt_sha256,
        max_new_tokens=max_new_tokens,
    )

    _disable_broken_optional_transformers_dependencies()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] to tokenize the prompt suite"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        root / manifest["tokenizer"]["path"],
        local_files_only=True,
        fix_mistral_regex=bool(
            manifest.get("tokenizer", {}).get("fix_mistral_regex", False)
        ),
    )

    results: list[dict[str, Any]] = []
    total_hits = 0
    total_reference = 0
    total_rows = 0
    total_selected = 0
    total_complete_traffic = 0
    total_kernel_traffic = 0
    total_global_metadata = 0
    all_runtime_checks = True
    for prompt_index, prompt in enumerate(prompt_values):
        token_ids = [
            int(token)
            for token in tokenizer.encode(prompt, add_special_tokens=True)
        ]
        if not token_ids:
            raise ValueError(f"prompt {prompt_index} tokenized to an empty input")
        reference_row = reference["results"][prompt_index]
        if reference_row.get("prompt_tokens") != len(token_ids):
            raise ValueError("reference prompt tokenization differs from package")
        command = [str(binary), str(root), str(max_new_tokens), str(threads)]
        if verify_reset:
            command.append("--verify-reset")
        command.extend(str(token) for token in token_ids)
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        wall_seconds = time.perf_counter() - started
        generated = [int(token) for token in completed.stdout.split()]
        expected = references[prompt]
        metrics = _parse_native_metrics(completed.stderr)
        hits = sum(
            actual == reference_token
            for actual, reference_token in zip(generated, expected, strict=False)
        )
        exact = generated == expected
        total_hits += hits
        total_reference += len(expected)
        expected_positions = len(token_ids) + max(len(generated) - 1, 0)
        expected_semantic_rows = expected_positions * layers
        expected_calls = len(generated) * layers
        rows = int(metrics["semantic_rows"])
        selected = int(metrics["selected_records"])
        traffic = _expected_traffic(
            index_path=index_path,
            layers=layers,
            hidden=hidden,
            intermediate=intermediate,
            positions=expected_positions,
            selected_records=selected,
        )
        runtime_checks = {
            "backend_is_dip": (
                metrics["semantic_backend"] == NATIVE_BITNET_DIP_OPERATOR
            ),
            "position_advancement": metrics["positions"] == expected_positions,
            "stage_calls": metrics["stage_calls"] == expected_calls,
            "semantic_calls": metrics["semantic_calls"] == expected_calls,
            "semantic_rows": rows == expected_semantic_rows,
            "selected_records_bounded": (
                traffic["minimum_selected_records"]
                <= selected
                <= traffic["maximum_selected_records"]
            ),
            "kernel_traffic_recomputed": (
                metrics["semantic_kernel_cache_line_bytes"]
                == traffic["kernel_cache_line_bytes"]
            ),
            "global_metadata_recomputed": (
                metrics["semantic_global_metadata_bytes"]
                == traffic["global_metadata_bytes"]
            ),
            "complete_traffic_recomputed": (
                metrics["semantic_cache_line_bytes"]
                == traffic["complete_modelled_cold_bytes"]
            ),
            "generated_exact_token_budget": (
                len(generated) == max_new_tokens
            ),
            "reset_requested": verify_reset,
            "reset_replay_tokens_verified": metrics["reset_verified"] == 1,
            "reset_counters_zeroed": metrics["reset_counters_zeroed"] == 1,
            "reset_replay_metrics_match": (
                metrics["replay_metrics_match"] == 1
            ),
        }
        runtime_passed = all(runtime_checks.values())
        all_runtime_checks &= runtime_passed
        dense_q4_bytes = rows * ((3 * hidden * intermediate + 1) // 2)
        active_fraction = selected / (rows * intermediate)
        traffic_fraction = (
            traffic["complete_modelled_cold_bytes"] / dense_q4_bytes
        )
        total_rows += rows
        total_selected += selected
        total_kernel_traffic += traffic["kernel_cache_line_bytes"]
        total_global_metadata += traffic["global_metadata_bytes"]
        total_complete_traffic += traffic["complete_modelled_cold_bytes"]
        results.append(
            {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "prompt_tokens": token_ids,
                "dense_reference_tokens": expected,
                "generated_tokens": generated,
                "token_hits": hits,
                "token_agreement": hits / len(expected),
                "exact_greedy_token_match_to_dense_reference": exact,
                "wall_seconds_including_reset_replay": wall_seconds,
                "metrics": metrics,
                "independently_recomputed_traffic": traffic,
                "mean_active_fraction": active_fraction,
                "mean_modelled_traffic_fraction_of_dense_q4": traffic_fraction,
                "runtime_checks": runtime_checks,
                "runtime_checks_passed": runtime_passed,
            }
        )

    if (
        sha256_file(binary) != expected_executable_sha256
        or sha256_file(manifest_path) != expected_package_sha256
        or validate_native_bitnet_package(root).get("valid") is not True
    ):
        raise ValueError("an authenticated runtime input changed during evaluation")
    exact_prompts = sum(
        result["exact_greedy_token_match_to_dense_reference"]
        for result in results
    )
    weighted_agreement = total_hits / total_reference
    exact_fraction = exact_prompts / len(results)
    global_active_fraction = total_selected / (total_rows * intermediate)
    dense_q4_bytes = total_rows * ((3 * hidden * intermediate + 1) // 2)
    global_traffic_fraction = total_complete_traffic / dense_q4_bytes
    maximum_prompt_active_fraction = max(
        result["mean_active_fraction"] for result in results
    )
    maximum_prompt_traffic_fraction = max(
        result["mean_modelled_traffic_fraction_of_dense_q4"]
        for result in results
    )
    gate_checks = {
        "minimum_prompts": (
            len(results) >= _REFERENCE_THRESHOLDS["minimum_prompts"]
        ),
        "minimum_reference_tokens": (
            total_reference
            >= _REFERENCE_THRESHOLDS[
                "minimum_generated_reference_tokens"
            ]
        ),
        "minimum_weighted_token_agreement": (
            weighted_agreement
            >= _REFERENCE_THRESHOLDS["minimum_weighted_token_agreement"]
        ),
        "minimum_exact_prompt_fraction": (
            exact_fraction
            >= _REFERENCE_THRESHOLDS["minimum_exact_prompt_fraction"]
        ),
        "all_runtime_invariants": all_runtime_checks,
        "maximum_global_mean_active_fraction": (
            global_active_fraction <= _MAXIMUM_MEAN_ACTIVE_FRACTION
        ),
        "maximum_prompt_mean_active_fraction": (
            maximum_prompt_active_fraction
            <= _MAXIMUM_MEAN_ACTIVE_FRACTION
        ),
        "maximum_global_mean_modelled_traffic_fraction": (
            global_traffic_fraction <= _MAXIMUM_MEAN_TRAFFIC_FRACTION
        ),
        "maximum_prompt_mean_modelled_traffic_fraction": (
            maximum_prompt_traffic_fraction
            <= _MAXIMUM_MEAN_TRAFFIC_FRACTION
        ),
    }
    gate_passed = all(gate_checks.values())
    report = {
        "schema_version": 1,
        "experiment": "native_bitnet_dip_packaged_token_generation",
        "status": (
            "integrated_native_dip_generation_passed"
            if gate_passed
            else "integrated_native_dip_generation_failed"
        ),
        "scope": "canonical_non_holdout_fixed_prompt_suite",
        "configuration": {
            "semantic_operator": NATIVE_BITNET_DIP_OPERATOR,
            "runtime": "authenticated_cpp_native_token_runtime",
            "max_new_tokens": max_new_tokens,
            "threads": threads,
            "verify_reset": verify_reset,
            "dense_semantic_fallback": False,
            "cpu_only": True,
            "metrics_scope": "first_generation_run",
            "wall_time_scope": "first_run_plus_reset_replay",
        },
        "inputs": {
            "package": str(root),
            "executable": str(binary),
            "prompt_suite": str(prompt_path),
            "dense_reference_report": str(reference_path),
        },
        "artifacts": {
            "package_manifest_sha256": expected_package_sha256,
            "coordinate_index_sha256": semantic["sha256"],
            "base_artifact_sha256": semantic["source_artifact_sha256"],
            "source_package_manifest_sha256": (
                semantic["source_package_manifest_sha256"]
            ),
            "policy_manifest_sha256": semantic["policy_manifest_sha256"],
            "adjudication_sha256": semantic["adjudication_sha256"],
            "executable_sha256": expected_executable_sha256,
            "prompt_suite_sha256": prompt_sha256,
            "reference_report_sha256": reference_sha256,
            "authenticated_before_and_after": True,
        },
        "thresholds": {
            **_REFERENCE_THRESHOLDS,
            "maximum_global_and_prompt_mean_active_fraction": (
                _MAXIMUM_MEAN_ACTIVE_FRACTION
            ),
            "maximum_global_and_prompt_mean_modelled_traffic_fraction_of_dense_q4": (
                _MAXIMUM_MEAN_TRAFFIC_FRACTION
            ),
        },
        "summary": {
            "prompts": len(results),
            "generated_reference_tokens": total_reference,
            "weighted_token_agreement": weighted_agreement,
            "exact_prompts": exact_prompts,
            "exact_prompt_fraction": exact_fraction,
            "semantic_rows": total_rows,
            "selected_records": total_selected,
            "global_mean_active_fraction": global_active_fraction,
            "maximum_prompt_mean_active_fraction": (
                maximum_prompt_active_fraction
            ),
            "kernel_modelled_cache_line_bytes": total_kernel_traffic,
            "global_metadata_cache_line_bytes": total_global_metadata,
            "complete_modelled_cold_bytes": total_complete_traffic,
            "global_mean_modelled_traffic_fraction_of_dense_q4": (
                global_traffic_fraction
            ),
            "maximum_prompt_mean_modelled_traffic_fraction_of_dense_q4": (
                maximum_prompt_traffic_fraction
            ),
            "total_wall_seconds_including_reset_replays": sum(
                float(result["wall_seconds_including_reset_replay"])
                for result in results
            ),
        },
        "gate_checks": gate_checks,
        "gate_passed": gate_passed,
        "decision": (
            "promote_dip_native_token_runtime_to_chat_binding_boundary"
            if gate_passed
            else "diagnose_packaged_dip_generation_failure"
        ),
        "limitations": {
            "traffic_is_modelled_not_measured_dram": True,
            "latency_is_not_an_upper_bound_gate": True,
            "prompt_suite_is_small": True,
            "milestone_2_holdout_reused": False,
            "python_chat_runtime_uses_this_backend": False,
            "reset_proves_token_and_structural_counter_replay_not_hidden_state_identity": (
                True
            ),
            "attention_contexts_exceed_local_window": False,
            "exact_match_is_greedy_token_match_not_hidden_or_logit_parity": True,
        },
        "results": results,
    }
    atomic_json(output_path, report)
    return report


__all__ = ["evaluate_native_bitnet_dip_token_generation"]
