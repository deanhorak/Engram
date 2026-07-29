"""Qualify the frozen OLMoE expert proxy on one complete causal-gate record.

The causal-head-gate archive already contains an authenticated serial execution
of M0 on selection sequence 0.  Its model used eager attention and the
``grouped_mm`` expert dispatcher, which resolves to Transformers' serial CPU
fallback on the frozen host stack.  Repeating that 26-minute reference record
would add no new semantic evidence.  This evaluator therefore:

* authenticates the frozen causal-gate protocol and complete training result;
* executes only the matching M0/sequence-0 record with the frozen-expert
  backward proxy;
* requires bit-exact loss, all 256 gate gradients, non-timing native-attention
  evidence, and the resulting single-record projected score/mask; and
* compares the new proxy record time with the archived reference-record time.

This is a performance qualification, not a new head-mask experiment.  The
single-record projection is diagnostic only and cannot replace the archived
two-record IHT projection.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

import engram.evaluation.olmoe_causal_head_gate as causal_gate
import engram.evaluation.olmoe_expert_proxy as expert_proxy
import engram.evaluation.olmoe_native_headwise as headwise
from engram.models.olmoe import audit_olmoe_source
from engram.tracing.olmoe import _prepare_transformers_imports
from engram.utils import atomic_json, sha256_file, sha256_json


_EXPERIMENT = "olmoe_causal_head_gate_frozen_expert_proxy_full_record"
_SCHEMA_VERSION = 1
_SEQUENCE_INDEX = 0
_MASK_NAME = "M0"
_WORKERS = 12
_MINIMUM_IMPROVEMENT_FRACTION = 0.10


def _new_output_path(out: str | Path) -> Path:
    raw = Path(out).expanduser()
    if raw.exists() or raw.is_symlink():
        raise ValueError("expert-proxy record benchmark target already exists")
    return raw.resolve()


def _write_new_report(path: Path, report: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("expert-proxy record benchmark target already exists")
    atomic_json(path, dict(report))


def _authenticated_source(
    path: Path,
    supplied_sha256: str,
    label: str,
) -> str:
    return causal_gate._require_digest(path, supplied_sha256, label)


def _transformers_expert_dispatch_contract() -> dict[str, str]:
    """Hash the installed files that resolve and execute expert dispatch."""

    _prepare_transformers_imports()
    try:
        import transformers.integrations.moe as transformers_moe
        import transformers.modeling_utils as transformers_modeling_utils
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "the installed Transformers expert dispatcher is unavailable"
        ) from exc
    moe_path = Path(transformers_moe.__file__).resolve()
    modeling_utils_path = Path(transformers_modeling_utils.__file__).resolve()
    return {
        "transformers_moe_path": str(moe_path),
        "transformers_moe_sha256": sha256_file(moe_path),
        "transformers_modeling_utils_path": str(modeling_utils_path),
        "transformers_modeling_utils_sha256": sha256_file(modeling_utils_path),
    }


def _strip_timing_fields(value: Any) -> Any:
    """Copy JSON-like evidence while removing only ``*_seconds`` fields."""

    if isinstance(value, Mapping):
        return {
            key: _strip_timing_fields(child)
            for key, child in value.items()
            if not (isinstance(key, str) and key.endswith("_seconds"))
        }
    if isinstance(value, list):
        return [_strip_timing_fields(child) for child in value]
    return value


def _matrix_comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return {
            "shape_exact": False,
            "exact": False,
            "mismatch_count": max(int(actual.size), int(expected.size)),
            "maximum_absolute_difference": None,
        }
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "shape_exact": True,
        "exact": bool(np.array_equal(actual, expected)),
        "mismatch_count": int(np.count_nonzero(actual != expected)),
        "maximum_absolute_difference": (
            float(difference.max()) if difference.size else 0.0
        ),
    }


def _record_projection(record: Mapping[str, Any], mask: np.ndarray) -> dict[str, Any]:
    gradient = causal_gate._strict_finite_matrix(
        record.get("gradient"),
        rows=causal_gate._LAYERS,
        columns=causal_gate._HEADS,
        label="proxy qualification gradient",
    )
    score, projected_mask, gradient_rms = causal_gate._projected_gate_step(
        mask,
        gradient,
    )
    selected = np.flatnonzero(projected_mask.reshape(-1)).astype(int).tolist()
    return {
        "gradient": gradient,
        "score": score,
        "mask": projected_mask,
        "gradient_sha256": sha256_json(gradient.tolist()),
        "score_sha256": sha256_json(score.tolist()),
        "mask_sha256": sha256_json(projected_mask.tolist()),
        "gradient_root_mean_square": gradient_rms,
        "selected_flat_indices": selected,
    }


def _record_parity(
    archived: Mapping[str, Any],
    proxy: Mapping[str, Any],
    *,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Compare all semantic record evidence and the diagnostic projection."""

    archived_projection = _record_projection(archived, mask)
    proxy_projection = _record_projection(proxy, mask)
    gradient = _matrix_comparison(
        proxy_projection["gradient"],
        archived_projection["gradient"],
    )
    projected_score = _matrix_comparison(
        proxy_projection["score"],
        archived_projection["score"],
    )
    archived_native = _strip_timing_fields(archived.get("native_oracle_layers"))
    proxy_native = _strip_timing_fields(proxy.get("native_oracle_layers"))
    archived_semantic = _strip_timing_fields(dict(archived))
    proxy_semantic = _strip_timing_fields(dict(proxy))
    identity_fields = (
        "sequence_index",
        "record_id",
        "mask_sha256",
        "selected_head_count",
        "backward",
    )
    checks = {
        "record_identity_exact": all(
            proxy.get(name) == archived.get(name) for name in identity_fields
        ),
        "loss_exact": proxy.get("loss") == archived.get("loss"),
        "gate_gradient_exact": gradient["exact"],
        "native_non_timing_diagnostics_exact": proxy_native == archived_native,
        "complete_non_timing_record_exact": proxy_semantic == archived_semantic,
        "projected_score_exact": projected_score["exact"],
        "projected_mask_exact": bool(
            np.array_equal(
                proxy_projection["mask"],
                archived_projection["mask"],
            )
        ),
        "projected_mask_has_exactly_51_heads": (
            int(proxy_projection["mask"].sum()) == causal_gate._RESCUED_HEADS
            and int(archived_projection["mask"].sum()) == causal_gate._RESCUED_HEADS
        ),
        "projected_flat_indices_exact": (
            proxy_projection["selected_flat_indices"]
            == archived_projection["selected_flat_indices"]
        ),
    }
    return {
        "checks": checks,
        "exact": all(checks.values()),
        "gradient_comparison": gradient,
        "projected_score_comparison": projected_score,
        "archived": {
            "semantic_record_sha256": sha256_json(archived_semantic),
            "native_non_timing_sha256": sha256_json(archived_native),
            "gradient_sha256": archived_projection["gradient_sha256"],
            "projected_score_sha256": archived_projection["score_sha256"],
            "projected_mask_sha256": archived_projection["mask_sha256"],
            "projected_flat_indices": archived_projection["selected_flat_indices"],
            "gradient_root_mean_square": archived_projection[
                "gradient_root_mean_square"
            ],
        },
        "proxy": {
            "semantic_record_sha256": sha256_json(proxy_semantic),
            "native_non_timing_sha256": sha256_json(proxy_native),
            "gradient_sha256": proxy_projection["gradient_sha256"],
            "projected_score_sha256": proxy_projection["score_sha256"],
            "projected_mask_sha256": proxy_projection["mask_sha256"],
            "projected_flat_indices": proxy_projection["selected_flat_indices"],
            "gradient_root_mean_square": proxy_projection["gradient_root_mean_square"],
        },
    }


def _positive_finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _nonnegative_finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _proxy_stats_checks(
    snapshot: Mapping[str, Any],
    *,
    record_elapsed_seconds: float,
) -> dict[str, bool]:
    tasks = snapshot.get("expert_backward_tasks")
    maximum_tasks = causal_gate._LAYERS * 64
    nested_seconds = sum(
        float(snapshot.get(name, math.inf))
        for name in (
            "serial_forward_seconds",
            "parallel_backward_task_seconds",
            "ordered_reduction_seconds",
        )
        if _nonnegative_finite(snapshot.get(name))
    )
    return {
        "workers_exact": snapshot.get("workers") == _WORKERS,
        "patched_all_layers": (snapshot.get("patched_layers") == causal_gate._LAYERS),
        "one_serial_forward_per_layer": (
            snapshot.get("serial_forward_calls") == causal_gate._LAYERS
        ),
        "one_parallel_backward_per_layer": (
            snapshot.get("parallel_backward_calls") == causal_gate._LAYERS
        ),
        "expert_task_population_valid": (
            not isinstance(tasks, bool)
            and isinstance(tasks, int)
            and causal_gate._LAYERS <= tasks <= maximum_tasks
        ),
        "serial_forward_time_positive": _positive_finite(
            snapshot.get("serial_forward_seconds")
        ),
        "parallel_backward_time_positive": _positive_finite(
            snapshot.get("parallel_backward_task_seconds")
        ),
        "reduction_time_nonnegative": _nonnegative_finite(
            snapshot.get("ordered_reduction_seconds")
        ),
        "component_times_nested_in_record": (
            _positive_finite(record_elapsed_seconds)
            and math.isfinite(nested_seconds)
            and nested_seconds <= record_elapsed_seconds + 1.0e-9
        ),
        "restored_all_layers": (snapshot.get("restored_layers") == causal_gate._LAYERS),
        "context_inactive_after_exit": snapshot.get("context_active") is False,
        "executor_shutdown": snapshot.get("executor_shutdown") is True,
    }


def _performance_comparison(
    archived_elapsed_seconds: Any,
    proxy_elapsed_seconds: Any,
) -> dict[str, Any]:
    if not _positive_finite(archived_elapsed_seconds) or not _positive_finite(
        proxy_elapsed_seconds
    ):
        raise ValueError("expert-proxy record timing is invalid")
    archived_seconds = float(archived_elapsed_seconds)
    proxy_seconds = float(proxy_elapsed_seconds)
    ratio = proxy_seconds / archived_seconds
    improvement = 1.0 - ratio
    material = ratio <= 1.0 - _MINIMUM_IMPROVEMENT_FRACTION
    return {
        "archived_reference_record_seconds": archived_seconds,
        "proxy_record_seconds": proxy_seconds,
        "proxy_to_archived_reference_ratio": ratio,
        "speedup_vs_archived_reference": archived_seconds / proxy_seconds,
        "improvement_fraction": improvement,
        "minimum_improvement_fraction": _MINIMUM_IMPROVEMENT_FRACTION,
        "material_improvement": material,
    }


def _elapsed(
    wall_start_ns: int,
    cpu_start_ns: int,
    wall_end_ns: int,
    cpu_end_ns: int,
) -> dict[str, float]:
    wall = (wall_end_ns - wall_start_ns) / 1_000_000_000.0
    cpu = (cpu_end_ns - cpu_start_ns) / 1_000_000_000.0
    if wall <= 0.0 or cpu < 0.0:
        raise RuntimeError("expert-proxy record timing clock did not advance")
    return {
        "wall_seconds": wall,
        "process_cpu_seconds": cpu,
        "effective_cores": cpu / wall,
    }


def _execute_proxy_record(
    *,
    torch: Any,
    loaded: Any,
    gate_state: MutableMapping[str, Any],
    mask: np.ndarray,
    context: Mapping[str, Any],
    teacher_logits: np.ndarray,
    teacher_hidden: np.ndarray,
    targets: np.ndarray,
    bands: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any], bool]:
    with expert_proxy.frozen_olmoe_expert_backward_proxy(
        loaded,
        workers=_WORKERS,
    ) as stats:
        wall_start = time.perf_counter_ns()
        cpu_start = time.process_time_ns()
        record = causal_gate._run_gate_record(
            loaded,
            gate_state,
            mask=mask,
            sequence_index=_SEQUENCE_INDEX,
            context=context,
            teacher_logits=teacher_logits,
            teacher_hidden=teacher_hidden,
            targets=targets,
            bands=bands,
            backward=True,
        )
        wall_end = time.perf_counter_ns()
        cpu_end = time.process_time_ns()
    frozen_gradients_absent = all(
        parameter.grad is None for parameter in loaded.parameters()
    )
    return (
        record,
        _elapsed(wall_start, cpu_start, wall_end, cpu_end),
        stats.snapshot(),
        frozen_gradients_absent,
    )


def _post_authentication(
    *,
    context: Mapping[str, Any],
    common_paths: Mapping[str, Path],
    manifest_sha256: str,
    causal_source_inventory: Mapping[str, str],
    prerequisite_paths: Mapping[str, Path],
    prerequisite_hashes: Mapping[str, str],
    gate_protocol_path: Path,
    gate_protocol_sha256: str,
    gate_training_result_path: Path,
    gate_training_result_sha256: str,
    framework_contract: Mapping[str, str],
    attention_library: Path,
    attention_library_sha256: str,
    proxy_source: Path,
    proxy_source_sha256: str,
    benchmark_source: Path,
    benchmark_source_sha256: str,
    expert_dispatch_contract: Mapping[str, str],
) -> dict[str, bool]:
    checks = causal_gate._training_post_authentication(
        context,
        common_paths,
        manifest_sha256=manifest_sha256,
        source_inventory=causal_source_inventory,
        prerequisite_paths=prerequisite_paths,
        prerequisite_hashes=prerequisite_hashes,
        protocol_path=gate_protocol_path,
        protocol_sha256=gate_protocol_sha256,
        framework_contract=framework_contract,
        attention_library=attention_library,
        attention_library_sha256=attention_library_sha256,
    )
    checks.update(
        {
            "gate_training_result": (
                sha256_file(gate_training_result_path) == gate_training_result_sha256
            ),
            "expert_proxy_source": (sha256_file(proxy_source) == proxy_source_sha256),
            "proxy_record_benchmark_source": (
                sha256_file(benchmark_source) == benchmark_source_sha256
            ),
            "transformers_expert_dispatch_contract": (
                _transformers_expert_dispatch_contract()
                == dict(expert_dispatch_contract)
            ),
        }
    )
    return checks


def benchmark_frozen_olmoe_expert_proxy_full_record(
    *,
    gate_protocol: str | Path,
    gate_protocol_sha256: str,
    gate_training_result: str | Path,
    gate_training_result_sha256: str,
    attention_library: str | Path,
    attention_library_sha256: str,
    expert_proxy_source_sha256: str,
    benchmark_source_sha256: str,
    transformers_moe_source_sha256: str,
    transformers_modeling_utils_source_sha256: str,
    out: str | Path,
    manifest_sha256: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one proxy record against the authenticated archived serial record."""

    output_path = _new_output_path(out)
    if not causal_gate._PREREQUISITE_ARGUMENT_NAMES.issubset(kwargs):
        raise ValueError("expert-proxy record prerequisites are incomplete")
    artifact_kwargs = {
        name: kwargs.pop(name) for name in causal_gate._PREREQUISITE_ARGUMENT_NAMES
    }
    attention_path, attention_hash = causal_gate._authenticate_attention_library(
        attention_library,
        attention_library_sha256,
    )
    proxy_source_path = Path(expert_proxy.__file__).resolve()
    benchmark_source_path = Path(__file__).resolve()
    proxy_source_hash = _authenticated_source(
        proxy_source_path,
        expert_proxy_source_sha256,
        "frozen OLMoE expert-proxy source",
    )
    benchmark_source_hash = _authenticated_source(
        benchmark_source_path,
        benchmark_source_sha256,
        "expert-proxy record benchmark source",
    )
    expert_dispatch_contract = _transformers_expert_dispatch_contract()
    _authenticated_source(
        Path(expert_dispatch_contract["transformers_moe_path"]),
        transformers_moe_source_sha256,
        "Transformers expert-dispatch source",
    )
    _authenticated_source(
        Path(expert_dispatch_contract["transformers_modeling_utils_path"]),
        transformers_modeling_utils_source_sha256,
        "Transformers expert-resolution source",
    )
    common_paths, context = headwise._common_context(
        manifest_sha256=manifest_sha256,
        **kwargs,
    )
    prerequisite_hashes, boundary = causal_gate._authenticate_failed_headwise_boundary(
        context,
        **artifact_kwargs,
    )
    protocol_path = Path(gate_protocol).expanduser().resolve()
    protocol_hash = causal_gate._require_digest(
        protocol_path,
        gate_protocol_sha256,
        "causal head-gate protocol",
    )
    protocol = causal_gate._read_json(
        protocol_path,
        "causal head-gate protocol",
    )
    causal_source_hash = sha256_file(Path(causal_gate.__file__).resolve())
    causal_source_inventory = causal_gate._current_source_inventory(context)
    framework = causal_gate._framework_contract()
    causal_gate._validate_gate_protocol(
        protocol,
        context,
        protocol_sha256=protocol_hash,
        supplied_sha256=gate_protocol_sha256,
        prerequisite_hashes=prerequisite_hashes,
        source_sha256=causal_source_hash,
        source_inventory=causal_source_inventory,
        framework_contract=framework,
        attention_library_path=attention_path,
        attention_library_sha256=attention_hash,
    )

    training_result_path = Path(gate_training_result).expanduser().resolve()
    training_result_hash = causal_gate._require_digest(
        training_result_path,
        gate_training_result_sha256,
        "causal head-gate training result",
    )
    training_result = causal_gate._read_json(
        training_result_path,
        "causal head-gate training result",
    )
    causal_gate._validate_training_result(
        training_result,
        result_sha256=training_result_hash,
        supplied_sha256=gate_training_result_sha256,
        protocol=protocol,
        protocol_sha256=protocol_hash,
    )
    archived_evaluation = training_result["executed_mask_evaluations"][_MASK_NAME]
    mask = causal_gate._strict_boolean_mask(
        archived_evaluation["mask"],
        "archived M0 qualification",
    )
    archived_record = archived_evaluation["records"][_SEQUENCE_INDEX]
    if int(mask.sum()) != 0:
        raise ValueError("expert-proxy qualification requires the all-base M0 mask")
    causal_gate._validate_executed_record(
        archived_record,
        sequence_index=_SEQUENCE_INDEX,
        mask=mask,
        backward=True,
        protocol=protocol,
    )

    model_path = Path(context["reference"]["source"]["model"]).resolve()
    audit = audit_olmoe_source(model_path)
    if (
        audit.decision != "proceed_to_router_trace"
        or audit.resolved_revision != protocol["source_revision"]
        or audit.config_sha256 != protocol["source_config_sha256"]
        or audit.index_sha256 != protocol["source_index_sha256"]
    ):
        raise ValueError("expert-proxy qualification dense source changed")
    teacher_logits_np, teacher_hidden_np, targets_np = headwise._load_teacher_arrays(
        context,
        common_paths["arrays_path"],
    )
    allowed_indices = protocol["training_data_access"]["selection_sequence_indices"]
    teacher_logits_by_sequence = causal_gate._selection_slices_only(
        teacher_logits_np,
        allowed_indices,
    )
    teacher_hidden_by_sequence = causal_gate._selection_slices_only(
        teacher_hidden_np,
        allowed_indices,
    )
    targets_by_sequence = causal_gate._selection_slices_only(
        targets_np,
        allowed_indices,
    )
    del teacher_logits_np, teacher_hidden_np, targets_np

    torch = causal_gate._require_torch()
    _prepare_transformers_imports()
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for the expert-proxy qualification"
        ) from exc
    threads_before = torch.get_num_threads()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    torch_random_state = torch.random.get_rng_state()
    loaded: Any | None = None
    originals: list[tuple[Any, Any]] = []
    proxy_record: dict[str, Any]
    external_timing: dict[str, float]
    proxy_stats: dict[str, Any]
    frozen_gradients_absent: bool
    experts_implementation: str
    gate_state: MutableMapping[str, Any] = {
        "attention_library": attention_path,
        "diagnostics": [],
    }
    execution_started = time.perf_counter()
    try:
        torch.set_num_threads(int(protocol["training"]["threads"]))
        random.seed(causal_gate._SEED)
        np.random.seed(causal_gate._SEED)
        torch.manual_seed(causal_gate._SEED)
        torch.use_deterministic_algorithms(True)
        loaded = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).eval()
        loaded.to("cpu")
        loaded.requires_grad_(False)
        experts_implementation = getattr(
            loaded.config,
            "_experts_implementation",
            None,
        )
        if experts_implementation != "grouped_mm":
            raise ValueError(
                "expert-proxy qualification requires the archived grouped_mm "
                "expert-dispatch contract"
            )
        if any(parameter.requires_grad for parameter in loaded.parameters()):
            raise ValueError("expert-proxy qualification teacher is not frozen")
        originals = causal_gate._install_causal_gate_attention(
            loaded,
            gate_state,
        )
        (
            proxy_record,
            external_timing,
            proxy_stats,
            frozen_gradients_absent,
        ) = _execute_proxy_record(
            torch=torch,
            loaded=loaded,
            gate_state=gate_state,
            mask=mask,
            context=context,
            teacher_logits=teacher_logits_by_sequence[_SEQUENCE_INDEX],
            teacher_hidden=teacher_hidden_by_sequence[_SEQUENCE_INDEX],
            targets=targets_by_sequence[_SEQUENCE_INDEX],
            bands=protocol["objective"]["bands"],
        )
    finally:
        if originals:
            causal_gate._restore_causal_gate_attention(originals)
        if loaded is not None:
            del loaded
        gc.collect()
        torch.random.set_rng_state(torch_random_state)
        np.random.set_state(numpy_random_state)
        random.setstate(python_random_state)
        torch.use_deterministic_algorithms(deterministic_before)
        torch.set_num_threads(threads_before)
    execution_seconds = time.perf_counter() - execution_started

    causal_gate._validate_executed_record(
        proxy_record,
        sequence_index=_SEQUENCE_INDEX,
        mask=mask,
        backward=True,
        protocol=protocol,
    )
    parity = _record_parity(
        archived_record,
        proxy_record,
        mask=mask,
    )
    performance = _performance_comparison(
        archived_record["elapsed_seconds"],
        proxy_record["elapsed_seconds"],
    )
    stats_checks = _proxy_stats_checks(
        proxy_stats,
        record_elapsed_seconds=float(proxy_record["elapsed_seconds"]),
    )
    post = _post_authentication(
        context=context,
        common_paths=common_paths,
        manifest_sha256=manifest_sha256,
        causal_source_inventory=causal_source_inventory,
        prerequisite_paths=boundary["paths"],
        prerequisite_hashes=prerequisite_hashes,
        gate_protocol_path=protocol_path,
        gate_protocol_sha256=protocol_hash,
        gate_training_result_path=training_result_path,
        gate_training_result_sha256=training_result_hash,
        framework_contract=framework,
        attention_library=attention_path,
        attention_library_sha256=attention_hash,
        proxy_source=proxy_source_path,
        proxy_source_sha256=proxy_source_hash,
        benchmark_source=benchmark_source_path,
        benchmark_source_sha256=benchmark_source_hash,
        expert_dispatch_contract=expert_dispatch_contract,
    )
    if not post or not all(post.values()):
        raise ValueError("expert-proxy qualification post-authentication failed")
    evidence_checks = {
        "archived_training_result_validated": True,
        "archived_M0_sequence_0_only": (
            archived_record["sequence_index"] == _SEQUENCE_INDEX
            and archived_record["selected_head_count"] == 0
        ),
        "exactly_one_new_backward_record": (
            proxy_stats.get("parallel_backward_calls") == causal_gate._LAYERS
        ),
        "teacher_weights_frozen": frozen_gradients_absent,
        "proxy_lifecycle_and_counts_valid": all(stats_checks.values()),
        "post_run_authentication": all(post.values()),
    }
    evidence_passed = all(evidence_checks.values())
    parity_passed = evidence_passed and bool(parity["exact"])
    authorized = parity_passed and bool(performance["material_improvement"])
    if not evidence_passed:
        status = "proxy_record_execution_invalid"
        decision = "reject_proxy_and_diagnose_execution_evidence"
    elif not parity_passed:
        status = "proxy_record_exact_parity_failed"
        decision = "reject_proxy_exact_full_record_parity_failed"
    elif not performance["material_improvement"]:
        status = "proxy_record_exact_but_not_materially_faster"
        decision = "retain_proxy_as_experimental_not_faster_enough"
    else:
        status = "proxy_record_exact_and_materially_faster"
        decision = "authorize_proxy_for_larger_development_fits"

    repository = Path(__file__).resolve().parents[3]
    source_inventory = {
        **dict(causal_source_inventory),
        str(proxy_source_path.relative_to(repository)): proxy_source_hash,
        str(benchmark_source_path.relative_to(repository)): benchmark_source_hash,
    }
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    report = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _EXPERIMENT,
        "status": status,
        "artifacts": {
            **headwise._base_bindings(context),
            **prerequisite_hashes,
            "gate_protocol_sha256": protocol_hash,
            "gate_training_result_sha256": training_result_hash,
            "causal_gate_source_sha256": causal_source_hash,
            "causal_gate_source_inventory_sha256": dict(causal_source_inventory),
            "training_attention_library_sha256": attention_hash,
            "expert_proxy_source_sha256": proxy_source_hash,
            "proxy_record_benchmark_source_sha256": benchmark_source_hash,
            "transformers_expert_dispatch_contract": dict(expert_dispatch_contract),
            "complete_proxy_benchmark_source_inventory_sha256": dict(
                sorted(source_inventory.items())
            ),
        },
        "framework_contract": framework,
        "transformers_expert_dispatch_contract": expert_dispatch_contract,
        "record_contract": {
            "mask_name": _MASK_NAME,
            "mask_sha256": sha256_json(mask.tolist()),
            "selected_head_count": int(mask.sum()),
            "sequence_index": _SEQUENCE_INDEX,
            "record_id": archived_record["record_id"],
            "positions": causal_gate._POSITIONS_PER_SEQUENCE,
            "backward": True,
            "archived_serial_record_is_reference": True,
            "new_serial_reference_execution_prohibited": True,
            "attention_implementation": "eager",
            "experts_implementation": experts_implementation,
            "single_record_projection_is_diagnostic_only": True,
        },
        "execution_contract": {
            "device": "cpu",
            "dtype": "bfloat16",
            "torch_intraop_threads": int(protocol["training"]["threads"]),
            "torch_interop_threads_observed": torch.get_num_interop_threads(),
            "proxy_workers": _WORKERS,
            "deterministic_algorithms": True,
            "model_eval": True,
            "teacher_weights_frozen": True,
            "proxy_forward_uses_installed_dispatcher_serially": True,
            "proxy_backward_is_expert_parallel": True,
            "nested_intraop_oversubscription_not_disabled": True,
            "exact_parity_required": True,
            "minimum_total_record_improvement_fraction": (
                _MINIMUM_IMPROVEMENT_FRACTION
            ),
        },
        "archived_serial_reference": {
            "training_result_sha256": training_result_hash,
            "record_elapsed_seconds": archived_record["elapsed_seconds"],
            "loss": archived_record["loss"],
            "gradient_sha256": parity["archived"]["gradient_sha256"],
            "native_non_timing_sha256": parity["archived"]["native_non_timing_sha256"],
            "semantic_record_sha256": parity["archived"]["semantic_record_sha256"],
            "projected_score_sha256": parity["archived"]["projected_score_sha256"],
            "projected_mask_sha256": parity["archived"]["projected_mask_sha256"],
        },
        "proxy_execution": {
            "record": proxy_record,
            "outer_timing": external_timing,
            "expert_proxy_stats": proxy_stats,
            "expert_proxy_stats_checks": stats_checks,
            "frozen_parameter_gradients_absent": frozen_gradients_absent,
        },
        "parity": parity,
        "exact_parity_passed": parity_passed,
        "performance": {
            **performance,
            "complete_execution_seconds_including_model_load_and_cleanup": (
                execution_seconds
            ),
        },
        "evidence_checks": evidence_checks,
        "evidence_passed": evidence_passed,
        "authorized_for_larger_development_fits": authorized,
        "post_run_authentication": post,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": str(torch.__version__),
            "torch_cpu_capability": (
                torch.backends.cpu.get_cpu_capability()
                if callable(
                    getattr(
                        getattr(torch.backends, "cpu", None),
                        "get_cpu_capability",
                        None,
                    )
                )
                else None
            ),
            "cpu_affinity": affinity,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "decision": decision,
        "limitations": [
            "The reference is the authenticated archived M0 sequence-0 record; its eager-attention/grouped-mm-expert execution is not rerun.",
            "The elapsed-time comparison spans separate executions and is a qualification boundary, not a controlled repeated benchmark.",
            "Only one previously consumed development-selection record is executed.",
            "The single-record projected mask is diagnostic and cannot replace the archived two-record IHT mask.",
            "The archive fixes Torch intra-op threads at 12 while the proxy uses 12 expert workers; nested scheduling can reduce the gain, so the full-record 10-percent rule adjudicates usefulness without changing the archived execution configuration.",
            "This proxy changes offline backward execution only; packaged inference remains CPU-only and unchanged.",
        ],
    }
    _write_new_report(output_path, report)
    return report


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify the frozen OLMoE expert proxy against one authenticated "
            "archived causal-head-gate serial reference record"
        )
    )
    headwise._add_common_arguments(parser)
    causal_gate._add_prerequisite_arguments(parser)
    parser.add_argument("--attention-library", required=True, type=Path)
    parser.add_argument("--attention-library-sha256", required=True)
    parser.add_argument("--gate-protocol", required=True, type=Path)
    parser.add_argument("--gate-protocol-sha256", required=True)
    parser.add_argument("--gate-training-result", required=True, type=Path)
    parser.add_argument("--gate-training-result-sha256", required=True)
    parser.add_argument("--expert-proxy-source-sha256", required=True)
    parser.add_argument("--benchmark-source-sha256", required=True)
    parser.add_argument("--transformers-moe-source-sha256", required=True)
    parser.add_argument(
        "--transformers-modeling-utils-source-sha256",
        required=True,
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    common = headwise._common_from_args(args)
    manifest_sha256 = common.pop("manifest_sha256")
    prerequisites = causal_gate._prerequisite_from_args(args)
    report = benchmark_frozen_olmoe_expert_proxy_full_record(
        **common,
        **prerequisites,
        manifest_sha256=manifest_sha256,
        gate_protocol=args.gate_protocol,
        gate_protocol_sha256=args.gate_protocol_sha256,
        gate_training_result=args.gate_training_result,
        gate_training_result_sha256=args.gate_training_result_sha256,
        attention_library=args.attention_library,
        attention_library_sha256=args.attention_library_sha256,
        expert_proxy_source_sha256=args.expert_proxy_source_sha256,
        benchmark_source_sha256=args.benchmark_source_sha256,
        transformers_moe_source_sha256=args.transformers_moe_source_sha256,
        transformers_modeling_utils_source_sha256=(
            args.transformers_modeling_utils_source_sha256
        ),
        out=args.out,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["authorized_for_larger_development_fits"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
