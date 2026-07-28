"""Full causal confirmation for the authenticated native OLMoE runtime."""

from __future__ import annotations

import gc
import json
import os
import tempfile
import time
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from engram.compiler.olmoe_native import validate_olmoe_native_package
from engram.evaluation.olmoe_native_generation import (
    _read_object,
    _validate_teacher_source,
)
from engram.models.olmoe import audit_olmoe_source
from engram.runtime.olmoe_native import OLMoENativePackageRuntime
from engram.tracing.olmoe import _prepare_transformers_imports
from engram.utils import atomic_json, sha256_file, sha256_json


_THRESHOLDS = {
    "maximum_mean_kl": 0.05,
    "minimum_top1_agreement": 0.90,
    "maximum_mean_target_nll_delta": 0.05,
    "maximum_mean_final_hidden_relative_l2": 0.10,
    "maximum_q7_traffic_fraction": 0.45,
    "minimum_sequences": 8,
    "minimum_prediction_positions": 256,
    "minimum_split_prediction_positions": 128,
}


def _authenticate_evaluator_sources(
    protocol: dict[str, Any],
) -> dict[str, str]:
    inventory = protocol.get("evaluator_source_sha256")
    if inventory is None:
        return {}
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("causal evaluator source inventory is invalid")
    repository = Path(__file__).resolve().parents[3]
    authenticated: dict[str, str] = {}
    for relative, expected_hash in inventory.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected_hash, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("causal evaluator source descriptor is invalid")
        source = repository / relative
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(
                f"causal evaluator source authentication failed: {relative}"
            )
        authenticated[relative] = expected_hash
    return authenticated


def _load_inputs(
    dataset: Path,
    tokenizer: object,
    *,
    sequences: int,
    tokens_per_sequence: int,
) -> list[list[int]]:
    inputs: list[list[int]] = []
    with dataset.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"causal dataset line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"causal dataset line {line_number} is invalid")
            if "input_ids" in record:
                token_ids = record["input_ids"]
            else:
                text = record.get("text")
                if not isinstance(text, str) or not text:
                    raise ValueError(f"causal dataset line {line_number} has no input")
                token_ids = tokenizer.encode(text).ids
            if (
                not isinstance(token_ids, list)
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in token_ids
                )
                or len(token_ids) < tokens_per_sequence
            ):
                raise ValueError(
                    f"causal dataset line {line_number} has too few tokens"
                )
            inputs.append(token_ids[:tokens_per_sequence])
            if len(inputs) == sequences:
                break
    if len(inputs) != sequences or len({tuple(row) for row in inputs}) != sequences:
        raise ValueError("causal dataset does not contain enough unique sequences")
    return inputs


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp.npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _threaded_expert_forward(
    module: object,
    hidden_states: object,
    top_k_index: object,
    top_k_weights: object,
    executor: ThreadPoolExecutor,
) -> object:
    """Execute independent OLMoE experts concurrently, then reduce in source order."""

    import torch
    from torch.nn import functional

    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(
            top_k_index,
            num_classes=module.num_experts,
        )
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_indices = [
            int(value[0].item())
            for value in torch.greater(
                expert_mask.sum(dim=(-1, -2)),
                0,
            ).nonzero()
        ]

    def calculate(expert_idx: int) -> tuple[object, object]:
        with torch.inference_mode():
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = functional.linear(
                current_state,
                module.gate_up_proj[expert_idx],
            ).chunk(2, dim=-1)
            current_hidden_states = module.act_fn(gate) * up
            current_hidden_states = functional.linear(
                current_hidden_states,
                module.down_proj[expert_idx],
            )
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            return token_idx, current_hidden_states

    for token_idx, current_hidden_states in executor.map(
        calculate,
        expert_indices,
    ):
        final_hidden_states.index_add_(
            0,
            token_idx,
            current_hidden_states.to(final_hidden_states.dtype),
        )
    return final_hidden_states


@contextmanager
def _threaded_olmoe_experts(model: object, workers: int):
    if workers == 1:
        yield 0
        return
    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="engram-olmoe-expert",
    )
    patched: list[tuple[object, object]] = []
    try:
        for module in model.modules():
            if module.__class__.__name__ != "OlmoeExperts":
                continue
            original = module.forward

            def forward(
                self,
                hidden_states,
                top_k_index,
                top_k_weights,
                *,
                _executor=executor,
            ):
                return _threaded_expert_forward(
                    self,
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                    _executor,
                )

            module.forward = types.MethodType(forward, module)
            patched.append((module, original))
        if not patched:
            raise RuntimeError("no OLMoE expert modules were available for threading")
        yield len(patched)
    finally:
        for module, original in patched:
            module.forward = original
        executor.shutdown(wait=True)


def capture_olmoe_teacher_causal_reference(
    *,
    model: str | Path,
    dataset: str | Path,
    out: str | Path,
    arrays_out: str | Path,
    sequences: int = 8,
    tokens_per_sequence: int = 33,
    device: str = "cpu",
    threads: int = 12,
    batch_size: int = 1,
    expert_workers: int = 1,
    sequence_workers: int | None = None,
) -> dict[str, Any]:
    """Capture untouched-teacher logits and final normalized hidden states."""

    if sequence_workers is None:
        sequence_workers = (
            1 if device != "cpu" or expert_workers != 1 else min(4, sequences)
        )
    if (
        sequences <= 0
        or tokens_per_sequence < 2
        or threads <= 0
        or batch_size <= 0
        or expert_workers <= 0
        or sequence_workers <= 0
        or device not in {"cpu", "cuda"}
        or (device != "cpu" and expert_workers != 1)
        or (device != "cpu" and sequence_workers != 1)
        or (expert_workers != 1 and sequence_workers != 1)
    ):
        raise ValueError("teacher causal capture configuration is invalid")
    model_path = Path(model).expanduser().resolve()
    dataset_path = Path(dataset).expanduser().resolve()
    arrays_path = Path(arrays_out).expanduser().resolve()
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
    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    input_ids = _load_inputs(
        dataset_path,
        tokenizer,
        sequences=sequences,
        tokens_per_sequence=tokens_per_sequence,
    )
    torch.set_num_threads(threads)
    loaded = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    loaded.to(device)
    logits: list[np.ndarray] = []
    hidden: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    batches = [
        input_ids[batch_start : batch_start + batch_size]
        for batch_start in range(0, len(input_ids), batch_size)
    ]

    def capture_batch(
        batch_ids: list[list[int]],
        _model=loaded,
    ) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
        with torch.inference_mode():
            tokens = torch.tensor(batch_ids, dtype=torch.long, device=device)
            output = _model(
                input_ids=tokens,
                use_cache=False,
                output_hidden_states=True,
            )
            return (
                output.logits[:, :-1].float().cpu().numpy(),
                output.hidden_states[-1][:, :-1].float().cpu().numpy(),
                batch_ids,
            )

    started = time.perf_counter()
    with _threaded_olmoe_experts(loaded, expert_workers) as patched_layers:
        if sequence_workers == 1:
            batch_results = map(capture_batch, batches)
            sequence_executor = None
        else:
            sequence_executor = ThreadPoolExecutor(
                max_workers=sequence_workers,
                thread_name_prefix="engram-olmoe-sequence",
            )
            batch_results = sequence_executor.map(capture_batch, batches)
        try:
            for batch_logits, batch_hidden, batch_ids in batch_results:
                for batch_offset, token_ids in enumerate(batch_ids):
                    logits.append(batch_logits[batch_offset])
                    hidden.append(batch_hidden[batch_offset])
                    targets.append(np.asarray(token_ids[1:], dtype=np.int64))
        finally:
            if sequence_executor is not None:
                sequence_executor.shutdown(wait=True)
    elapsed = time.perf_counter() - started
    del batch_results
    del capture_batch
    del loaded
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    logits_array = np.ascontiguousarray(np.concatenate(logits), dtype=np.float32)
    hidden_array = np.ascontiguousarray(np.concatenate(hidden), dtype=np.float32)
    targets_array = np.ascontiguousarray(np.concatenate(targets), dtype=np.int64)
    _write_npz_atomic(
        arrays_path,
        logits=logits_array,
        hidden=hidden_array,
        targets=targets_array,
    )
    reference = {
        "schema_version": 1,
        "experiment": "olmoe_untouched_teacher_causal_reference",
        "source": {
            "model": str(model_path),
            "revision": audit.resolved_revision,
            "config_sha256": audit.config_sha256,
            "index_sha256": audit.index_sha256,
            "adapter": audit.adapter,
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "sequences": sequences,
            "tokens_per_sequence": tokens_per_sequence,
            "prediction_positions": sequences * (tokens_per_sequence - 1),
            "input_identity": sha256_json(input_ids),
            "input_ids": input_ids,
        },
        "configuration": {
            "dtype": "bfloat16",
            "device": device,
            "threads": threads,
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "batch_size": min(batch_size, sequences),
            "batches": (sequences + batch_size - 1) // batch_size,
            "expert_workers": expert_workers,
            "sequence_workers": sequence_workers,
            "threaded_expert_layers": patched_layers,
            "expert_backend": (
                "thread_pool_source_order_v1"
                if expert_workers > 1
                else "transformers_reference"
            ),
            "sequence_backend": (
                "thread_pool_shared_model_v1"
                if sequence_workers > 1
                else "serial_batches"
            ),
            "attention_implementation": "eager",
            "use_cache": False,
            "output_hidden_states": True,
            "weights_modified": False,
        },
        "arrays": {
            "path": str(arrays_path),
            "sha256": sha256_file(arrays_path),
            "keys": {
                "logits": {
                    "dtype": "float32",
                    "shape": list(logits_array.shape),
                },
                "hidden": {
                    "dtype": "float32",
                    "shape": list(hidden_array.shape),
                },
                "targets": {
                    "dtype": "int64",
                    "shape": list(targets_array.shape),
                },
            },
        },
        "elapsed_seconds": elapsed,
    }
    atomic_json(Path(out), reference)
    return reference


def _position_metrics(
    teacher_logits: np.ndarray,
    native_logits: np.ndarray,
    teacher_hidden: np.ndarray,
    native_hidden: np.ndarray,
    target: int,
) -> dict[str, float | bool | int]:
    import torch

    teacher_scores = torch.from_numpy(np.asarray(teacher_logits, dtype=np.float32))
    native_scores = torch.from_numpy(np.asarray(native_logits, dtype=np.float32))
    teacher_state = torch.from_numpy(np.asarray(teacher_hidden, dtype=np.float32))
    native_state = torch.from_numpy(np.asarray(native_hidden, dtype=np.float32))
    teacher_log_probabilities = torch.log_softmax(teacher_scores, dim=-1)
    native_log_probabilities = torch.log_softmax(native_scores, dim=-1)
    teacher_probabilities = teacher_log_probabilities.exp()
    teacher_top1 = int(torch.argmax(teacher_scores).item())
    native_top1 = int(torch.argmax(native_scores).item())
    hidden_denominator = torch.linalg.vector_norm(teacher_state).clamp_min(1.0e-12)
    return {
        "kl": float(
            torch.sum(
                teacher_probabilities
                * (teacher_log_probabilities - native_log_probabilities)
            ).item()
        ),
        "top1_match": teacher_top1 == native_top1,
        "teacher_top1": teacher_top1,
        "native_top1": native_top1,
        "target_nll_delta": float(
            (
                -native_log_probabilities[target] + teacher_log_probabilities[target]
            ).item()
        ),
        "hidden_relative_l2": float(
            (
                torch.linalg.vector_norm(native_state - teacher_state)
                / hidden_denominator
            ).item()
        ),
    }


def _aggregate(rows: list[dict[str, float | bool | int]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty causal metrics")
    return {
        "prediction_positions": len(rows),
        "teacher_to_native_kl": float(np.mean([float(row["kl"]) for row in rows])),
        "teacher_top1_agreement": float(
            np.mean([bool(row["top1_match"]) for row in rows])
        ),
        "target_nll_delta": float(
            np.mean([float(row["target_nll_delta"]) for row in rows])
        ),
        "final_hidden_relative_l2": float(
            np.mean([float(row["hidden_relative_l2"]) for row in rows])
        ),
        "maximum_position_kl": max(float(row["kl"]) for row in rows),
        "p95_position_kl": float(np.percentile([float(row["kl"]) for row in rows], 95)),
    }


def evaluate_native_olmoe_causal(
    *,
    package: str | Path,
    manifest_sha256: str,
    library: str | Path,
    dataset: str | Path,
    teacher_reference: str | Path,
    teacher_arrays: str | Path,
    protocol: str | Path,
    protocol_sha256: str,
    out: str | Path,
    threads: int | None = None,
) -> dict[str, Any]:
    """Run the complete 8x32 causal gate through the native package."""

    package_path = Path(package).expanduser().resolve()
    library_path = Path(library).expanduser().resolve()
    dataset_path = Path(dataset).expanduser().resolve()
    reference_path = Path(teacher_reference).expanduser().resolve()
    arrays_path = Path(teacher_arrays).expanduser().resolve()
    protocol_path = Path(protocol).expanduser().resolve()
    reference = _read_object(reference_path, "causal teacher reference")
    protocol_value = _read_object(protocol_path, "causal protocol")
    protocol_hash = sha256_file(protocol_path)
    library_hash = sha256_file(library_path)
    dataset_hash = sha256_file(dataset_path)
    reference_hash = sha256_file(reference_path)
    arrays_hash = sha256_file(arrays_path)
    scope = protocol_value.get("scope")
    frozen_threads = scope.get("candidate_threads") if isinstance(scope, dict) else None
    if protocol_hash != protocol_sha256.lower():
        raise ValueError("native OLMoE causal protocol authentication failed")
    evaluator_sources = _authenticate_evaluator_sources(protocol_value)
    if (
        protocol_value.get("schema_version") != 1
        or protocol_value.get("experiment")
        != "olmoe_native_package_causal_confirmation"
        or protocol_value.get("status") != "frozen_before_candidate_execution"
        or protocol_value.get("package_manifest_sha256") != manifest_sha256.lower()
        or protocol_value.get("native_library_sha256") != library_hash
        or protocol_value.get("dataset_sha256") != dataset_hash
        or protocol_value.get("teacher_reference_sha256") != reference_hash
        or protocol_value.get("teacher_arrays_sha256") != arrays_hash
        or protocol_value.get("thresholds") != _THRESHOLDS
        or protocol_value.get("sequences") != 8
        or protocol_value.get("tokens_per_sequence") != 33
        or protocol_value.get("post_window_uses_same_quality_thresholds") is not True
        or isinstance(frozen_threads, bool)
        or not isinstance(frozen_threads, int)
        or frozen_threads <= 0
        or (threads is not None and threads != frozen_threads)
    ):
        raise ValueError("native OLMoE causal protocol contract is invalid")
    if (
        reference.get("schema_version") != 1
        or reference.get("experiment") != "olmoe_untouched_teacher_causal_reference"
        or reference.get("configuration", {}).get("weights_modified") is not False
        or reference.get("dataset", {}).get("sha256")
        != protocol_value.get("dataset_sha256")
        or reference.get("dataset", {}).get("input_identity")
        != protocol_value.get("input_identity")
        or reference.get("dataset", {}).get("sequences")
        != protocol_value.get("sequences")
        or reference.get("dataset", {}).get("tokens_per_sequence")
        != protocol_value.get("tokens_per_sequence")
        or reference.get("arrays", {}).get("sha256")
        != protocol_value.get("teacher_arrays_sha256")
    ):
        raise ValueError("native OLMoE causal teacher contract is invalid")
    _validate_teacher_source(reference, protocol_value)
    input_ids = reference["dataset"]["input_ids"]
    if sha256_json(input_ids) != protocol_value["input_identity"]:
        raise ValueError("causal teacher input identity authentication failed")
    sequences = int(reference["dataset"]["sequences"])
    tokens_per_sequence = int(reference["dataset"]["tokens_per_sequence"])
    prediction_positions = sequences * (tokens_per_sequence - 1)
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"logits", "hidden", "targets"}:
            raise ValueError("causal teacher arrays have unexpected keys")
        teacher_logits = np.asarray(arrays["logits"], dtype=np.float32)
        teacher_hidden = np.asarray(arrays["hidden"], dtype=np.float32)
        targets = np.asarray(arrays["targets"], dtype=np.int64)
    expected_targets = np.asarray(
        [token for sequence in input_ids for token in sequence[1:]],
        dtype=np.int64,
    )
    model = protocol_value["model"]
    if (
        teacher_logits.shape != (prediction_positions, int(model["vocab_size"]))
        or teacher_hidden.shape != (prediction_positions, int(model["hidden_size"]))
        or targets.shape != (prediction_positions,)
        or not np.array_equal(targets, expected_targets)
    ):
        raise ValueError("causal teacher array shapes are invalid")

    all_rows: list[dict[str, float | bool | int]] = []
    exact_window_rows: list[dict[str, float | bool | int]] = []
    retrieval_rows: list[dict[str, float | bool | int]] = []
    sequence_results: list[dict[str, Any]] = []
    total_q7_bytes = 0
    load_started = time.perf_counter()
    runtime = OLMoENativePackageRuntime(
        package_path,
        manifest_sha256=manifest_sha256,
        library=library_path,
        threads=threads,
    )
    cold_load_seconds = time.perf_counter() - load_started
    try:
        effective_threads = (
            int(runtime.manifest["runtime"]["kernel_threads"])
            if threads is None
            else int(threads)
        )
        if (
            runtime.manifest.get("source", {}).get("revision")
            != protocol_value["source_revision"]
            or runtime.manifest["files"]["model/config.json"]["sha256"]
            != protocol_value["source_config_sha256"]
            or effective_threads != frozen_threads
        ):
            raise ValueError("causal package source or thread identity is invalid")
        offset = 0
        for sequence_index, sequence in enumerate(input_ids):
            runtime.reset()
            sequence_rows = []
            started = time.perf_counter()
            for position, token_id in enumerate(sequence[:-1]):
                native_result = runtime.runtime.forward([token_id])
                native_hidden, native_logits = runtime.runtime.last_diagnostics()
                diagnostic_top1 = int(np.argmax(native_logits))
                if diagnostic_top1 != native_result.next_token:
                    raise ValueError(
                        "native diagnostic argmax differs from the returned token"
                    )
                row = _position_metrics(
                    teacher_logits[offset],
                    native_logits,
                    teacher_hidden[offset],
                    native_hidden,
                    int(targets[offset]),
                )
                row.update(
                    {
                        "sequence": sequence_index,
                        "position": position,
                        "target": int(targets[offset]),
                    }
                )
                all_rows.append(row)
                sequence_rows.append(row)
                if position < 16:
                    exact_window_rows.append(row)
                else:
                    retrieval_rows.append(row)
                offset += 1
            elapsed = time.perf_counter() - started
            metrics = runtime.runtime.last_result.metrics
            total_q7_bytes += metrics["q7_scheduled_bytes"]
            sequence_results.append(
                {
                    "sequence": sequence_index,
                    "input_ids": sequence,
                    "elapsed_seconds": elapsed,
                    "metrics": _aggregate(sequence_rows),
                    "q7_scheduled_bytes": metrics["q7_scheduled_bytes"],
                    "q7_elapsed_seconds": metrics["q7_elapsed_ns"] / 1.0e9,
                    "native_elapsed_seconds": metrics["elapsed_ns"] / 1.0e9,
                    "attention_state_bytes": metrics["attention_state_bytes"],
                    "cache_position": runtime.runtime.position,
                    "cache_position_passed": (
                        runtime.runtime.position == tokens_per_sequence - 1
                    ),
                }
            )
    finally:
        runtime.close()
    post_manifest = validate_olmoe_native_package(
        package_path,
        expected_manifest_sha256=manifest_sha256,
    )
    post_root_authentication = {
        "package": post_manifest == runtime.manifest,
        "library": sha256_file(library_path) == library_hash,
        "protocol": sha256_file(protocol_path) == protocol_hash,
        "dataset": sha256_file(dataset_path) == dataset_hash,
        "teacher_reference": sha256_file(reference_path) == reference_hash,
        "teacher_arrays": sha256_file(arrays_path) == arrays_hash,
        "evaluator_sources": all(
            sha256_file(Path(__file__).resolve().parents[3] / relative) == expected_hash
            for relative, expected_hash in evaluator_sources.items()
        ),
    }
    post_authentication = all(post_root_authentication.values())
    aggregate = _aggregate(all_rows)
    exact_window = _aggregate(exact_window_rows)
    retrieval = _aggregate(retrieval_rows)
    layers = int(model["layers"])
    hidden = int(model["hidden_size"])
    intermediate = int(model["intermediate_size"])
    experts = int(model["experts"])
    ideal_q4_bytes_per_position = layers * experts * 3 * hidden * intermediate // 2
    q7_traffic_fraction = total_q7_bytes / (
        prediction_positions * ideal_q4_bytes_per_position
    )
    checks = {
        "sequence_count": sequences >= _THRESHOLDS["minimum_sequences"],
        "prediction_positions": (
            prediction_positions >= _THRESHOLDS["minimum_prediction_positions"]
        ),
        "exact_window_positions": exact_window["prediction_positions"]
        >= _THRESHOLDS["minimum_split_prediction_positions"],
        "post_window_positions": retrieval["prediction_positions"]
        >= _THRESHOLDS["minimum_split_prediction_positions"],
        "mean_kl": aggregate["teacher_to_native_kl"] <= _THRESHOLDS["maximum_mean_kl"],
        "top1_agreement": aggregate["teacher_top1_agreement"]
        >= _THRESHOLDS["minimum_top1_agreement"],
        "target_nll_delta": aggregate["target_nll_delta"]
        <= _THRESHOLDS["maximum_mean_target_nll_delta"],
        "hidden_relative_l2": aggregate["final_hidden_relative_l2"]
        <= _THRESHOLDS["maximum_mean_final_hidden_relative_l2"],
        "exact_window_mean_kl": exact_window["teacher_to_native_kl"]
        <= _THRESHOLDS["maximum_mean_kl"],
        "exact_window_top1_agreement": exact_window["teacher_top1_agreement"]
        >= _THRESHOLDS["minimum_top1_agreement"],
        "exact_window_target_nll_delta": exact_window["target_nll_delta"]
        <= _THRESHOLDS["maximum_mean_target_nll_delta"],
        "exact_window_hidden_relative_l2": exact_window["final_hidden_relative_l2"]
        <= _THRESHOLDS["maximum_mean_final_hidden_relative_l2"],
        "post_window_mean_kl": retrieval["teacher_to_native_kl"]
        <= _THRESHOLDS["maximum_mean_kl"],
        "post_window_top1_agreement": retrieval["teacher_top1_agreement"]
        >= _THRESHOLDS["minimum_top1_agreement"],
        "post_window_target_nll_delta": retrieval["target_nll_delta"]
        <= _THRESHOLDS["maximum_mean_target_nll_delta"],
        "post_window_hidden_relative_l2": retrieval["final_hidden_relative_l2"]
        <= _THRESHOLDS["maximum_mean_final_hidden_relative_l2"],
        "q7_traffic": q7_traffic_fraction <= _THRESHOLDS["maximum_q7_traffic_fraction"],
        "cache_positions": all(
            result["cache_position_passed"] for result in sequence_results
        ),
        "post_run_authentication": post_authentication,
    }
    gate_passed = all(checks.values())
    report = {
        "schema_version": 1,
        "experiment": "olmoe_native_package_causal_confirmation",
        "status": (
            "frozen_confirmation_passed"
            if gate_passed
            else "frozen_confirmation_failed"
        ),
        "artifacts": {
            "package_manifest_sha256": manifest_sha256.lower(),
            "native_library_sha256": library_hash,
            "dataset_sha256": sha256_file(dataset_path),
            "teacher_reference_sha256": sha256_file(reference_path),
            "teacher_arrays_sha256": sha256_file(arrays_path),
            "protocol_sha256": protocol_sha256.lower(),
            "evaluator_source_sha256": evaluator_sources,
        },
        "configuration": {
            "sequences": sequences,
            "tokens_per_sequence": tokens_per_sequence,
            "prediction_positions": prediction_positions,
            "cpu_only_candidate": True,
            "candidate_threads": effective_threads,
            "transformers_model_shell_used_by_candidate": False,
            "attention_policy": "native_streaming_w16_c8_k4_sinks2",
        },
        "metrics": aggregate,
        "position_splits": {
            "exact_local_positions_0_15": exact_window,
            "bounded_retrieval_positions_16_31": retrieval,
        },
        "per_position_offset": {
            str(position): _aggregate(
                [row for row in all_rows if row["position"] == position]
            )
            for position in range(tokens_per_sequence - 1)
        },
        "first_top1_divergence": next(
            (
                {
                    "sequence": int(row["sequence"]),
                    "position": int(row["position"]),
                    "teacher_top1": int(row["teacher_top1"]),
                    "native_top1": int(row["native_top1"]),
                    "kl": float(row["kl"]),
                }
                for row in all_rows
                if not row["top1_match"]
            ),
            None,
        ),
        "traffic": {
            "q7_scheduled_bytes": total_q7_bytes,
            "all_expert_ideal_q4_bytes": (
                prediction_positions * ideal_q4_bytes_per_position
            ),
            "fraction_of_all_expert_ideal_q4": q7_traffic_fraction,
            "measured_hardware_traffic": False,
        },
        "performance": {
            "cold_authentication_and_load_seconds": cold_load_seconds,
            "sequence_seconds": [
                result["elapsed_seconds"] for result in sequence_results
            ],
            "total_sequence_seconds": sum(
                result["elapsed_seconds"] for result in sequence_results
            ),
            "total_native_elapsed_seconds": sum(
                result["native_elapsed_seconds"] for result in sequence_results
            ),
            "total_q7_elapsed_seconds": sum(
                result["q7_elapsed_seconds"] for result in sequence_results
            ),
        },
        "sequence_results": sequence_results,
        "position_results": all_rows,
        "post_run_authentication": post_root_authentication,
        "thresholds": _THRESHOLDS,
        "checks": checks,
        "gate_passed": gate_passed,
        "decision": (
            "promote_complete_native_causal_boundary"
            if gate_passed
            else "stop_and_diagnose_native_attention_or_numeric_divergence"
        ),
        "limitations": [
            "Scheduled packed bytes are exact algorithmic reads, not hardware-counter DRAM traffic.",
            "This fixed corpus is a causal confirmation, not a broad language benchmark.",
        ],
    }
    atomic_json(Path(out), report)
    return report


__all__ = [
    "capture_olmoe_teacher_causal_reference",
    "evaluate_native_olmoe_causal",
]
