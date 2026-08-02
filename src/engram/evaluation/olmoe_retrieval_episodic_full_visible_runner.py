"""Native capture orchestration for the full-visible attention-value oracle.

The numerical experiment lives in
``olmoe_retrieval_episodic_full_visible_simplex_oracle``.  This module keeps
native execution separate from that cached solver:

* ``parity`` executes train record zero twice around a real reset and proves
  exact agreement with the authenticated historical head-mass trace and
  output evidence;
* ``freeze`` delegates the prospective protocol freeze to the oracle;
* ``capture`` executes all eight authenticated train records twice and writes
  only reset-proven safetensor shards plus their immutable manifest; and
* ``solve`` delegates to the oracle's native-free cached solve.

All user-supplied paths are lexically rejected if any component is named
``confirmation.jsonl``.  That rejection happens before any filesystem
operation.  The authenticated confirmation descriptor is copied by value
inside inherited artifacts and is never interpreted as a path here.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import engram.evaluation.olmoe_retrieval_episodic_full_visible_simplex_oracle as full
import engram.evaluation.olmoe_retrieval_episodic_slot_simplex_cached as cached
import engram.evaluation.olmoe_retrieval_episodic_slot_simplex_oracle as slot
from engram.utils import atomic_json, sha256_file, sha256_json

_EXPECTED_PREDECESSOR_PROTOCOL_SHA256 = (
    "f3be957ec0c13d0f49c85a2fa149611307de756f2be82165098a43263bb78ce3"
)
_EXPECTED_PREDECESSOR_RESULT_SHA256 = (
    "2e8e9b7d5f33d33c0e8c642a50359da785d0690d8260ed9f65837b14cd93a5bf"
)
_EXPECTED_CAPTURE_REPORT_SHA256 = (
    "18218d3a7dbcae731ae42b85cefc09a20ab738ad15531bae3be74c17368d8258"
)
_CONFIRMATION_FILENAME = "confirmation.jsonl"
_RUNNER_SOURCE = "src/engram/evaluation/olmoe_retrieval_episodic_full_visible_runner.py"


def _progress(message: str) -> None:
    print(
        f"[retrieval-episodic-full-visible] {message}",
        file=sys.stderr,
        flush=True,
    )


def _reject_confirmation_paths(
    values: Sequence[tuple[str, str | Path]],
) -> None:
    """Reject forbidden paths lexically, before touching the filesystem."""

    for label, value in values:
        requested = Path(value)
        if any(part.casefold() == _CONFIRMATION_FILENAME for part in requested.parts):
            raise ValueError(f"{label} cannot name the confirmation split")


def _guard_paths(
    values: Sequence[tuple[str, str | Path]],
) -> None:
    """Apply requested-path and resolved parent-alias confirmation guards."""

    # Complete the purely lexical pass for every input before resolving even
    # one path.  This guarantees a literal confirmation path cannot induce a
    # stat through another otherwise-valid argument.
    _reject_confirmation_paths(values)
    for label, value in values:
        requested = Path(value).expanduser()
        resolved_parent = requested.parent.resolve(strict=False)
        resolved = resolved_parent / requested.name
        if any(part.casefold() == _CONFIRMATION_FILENAME for part in resolved.parts):
            raise ValueError(f"{label} resolves inside the confirmation split")
        # Do not resolve the leaf: a leaf symlink could itself target the
        # forbidden split.  Inputs and outputs in this protocol must be
        # regular lexical leaves beneath the already-checked parent.
        if requested.is_symlink():
            raise ValueError(f"{label} must not be a symlink")


def _binding(path: Path, digest: str) -> dict[str, str]:
    return {"path": str(path), "sha256": digest.lower()}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _new_output(value: str | Path, label: str) -> Path:
    _guard_paths(((label, value),))
    requested = Path(value).expanduser()
    if requested.exists() or requested.is_symlink():
        raise ValueError(f"{label} already exists")
    requested.parent.mkdir(parents=True, exist_ok=True)
    return requested.resolve()


def _require_regular_trace_symbol(path: Path) -> None:
    try:
        library = ctypes.CDLL(str(path))
    except OSError as error:
        raise ValueError(
            "full-visible native trace library cannot be loaded"
        ) from error
    if not hasattr(library, full._REGULAR_ENTRY_COPY_SYMBOL):
        raise ValueError(
            "full-visible native trace library lacks the regular-entry symbol"
        )


def _validate_predecessor_result(
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    protocol_binding: Mapping[str, str],
    capture_binding: Mapping[str, Any],
) -> None:
    full._validate_predecessor_failure(
        protocol,
        result,
        protocol_binding=protocol_binding,
    )
    post = result.get("post_run_authentication")
    result_capture = result.get("capture_report")
    scope = result.get("scope")
    if (
        not isinstance(post, Mapping)
        or not post
        or not all(check is True for check in post.values())
        or not isinstance(result_capture, Mapping)
        or result_capture.get("path") != capture_binding.get("path")
        or result_capture.get("sha256") != capture_binding.get("sha256")
        or not isinstance(scope, Mapping)
        or scope.get("split") != "train"
        or scope.get("native_execution_performed") is not False
        or scope.get("confirmation_split_opened") is not False
    ):
        raise ValueError("full-visible cached predecessor authentication changed")


def _authenticate_predecessor_inputs(
    *,
    predecessor_protocol: str | Path,
    predecessor_protocol_sha256: str,
    predecessor_result: str | Path,
    predecessor_result_sha256: str,
    trace_library: str | Path,
    trace_library_sha256: str,
) -> dict[str, Any]:
    """Authenticate the exact cached V2 failure and native train inputs."""

    if (
        predecessor_protocol_sha256.lower() != _EXPECTED_PREDECESSOR_PROTOCOL_SHA256
        or predecessor_result_sha256.lower() != _EXPECTED_PREDECESSOR_RESULT_SHA256
    ):
        raise ValueError("full-visible cached predecessor root changed")
    protocol_path = full._checked_file(
        predecessor_protocol,
        predecessor_protocol_sha256,
        "full-visible predecessor protocol",
    )
    result_path = full._checked_file(
        predecessor_result,
        predecessor_result_sha256,
        "full-visible predecessor result",
    )
    library_path = full._checked_file(
        trace_library,
        trace_library_sha256,
        "full-visible trace library",
    )
    protocol = _read_json(protocol_path, "full-visible predecessor protocol")
    result = _read_json(result_path, "full-visible predecessor result")
    protocol_binding = _binding(protocol_path, predecessor_protocol_sha256)
    capture_binding = protocol.get("capture_report")
    historical = protocol.get("historical_bindings")
    if (
        not isinstance(capture_binding, Mapping)
        or capture_binding.get("sha256") != _EXPECTED_CAPTURE_REPORT_SHA256
        or not isinstance(historical, Mapping)
    ):
        raise ValueError("full-visible predecessor bindings changed")
    _validate_predecessor_result(
        protocol,
        result,
        protocol_binding=protocol_binding,
        capture_binding=capture_binding,
    )

    capture_path, capture_report, cached_context = cached._authenticate_capture_report(
        capture_binding.get("path"),
        capture_binding.get("sha256"),
        stack_arrays=False,
    )
    if (
        _binding(capture_path, capture_binding["sha256"])
        != {
            "path": capture_binding.get("path"),
            "sha256": capture_binding.get("sha256"),
        }
        or capture_binding.get("capture_rows_sha256")
        != capture_report.get("capture_rows_sha256")
        or capture_report.get("bindings") != historical
        or capture_report.get("output_projection") != protocol.get("output_projection")
        or capture_report.get("authenticated_confirmation_descriptor")
        != protocol.get("authenticated_confirmation_descriptor")
        or capture_report.get("confirmation_split_opened") is not False
    ):
        raise ValueError("full-visible predecessor capture binding changed")

    joint_protocol = historical.get("joint_gamma_protocol")
    joint_result = historical.get("joint_gamma_result")
    if not isinstance(joint_protocol, Mapping) or not isinstance(
        joint_result,
        Mapping,
    ):
        raise ValueError("full-visible historical joint bindings changed")
    runtime_context, _training, _joint_failure = slot._authenticate_joint_inputs(
        joint_protocol=joint_protocol.get("path"),
        joint_protocol_sha256=joint_protocol.get("sha256"),
        joint_result=joint_result.get("path"),
        joint_result_sha256=joint_result.get("sha256"),
        trace_library=library_path,
        trace_library_sha256=trace_library_sha256,
    )
    _require_regular_trace_symbol(library_path)

    path_bindings = {
        "joint_gamma_protocol": (
            runtime_context["joint_gamma_protocol_path"],
            runtime_context["joint_gamma_protocol_sha256"],
        ),
        "joint_gamma_result": (
            runtime_context["joint_gamma_result_path"],
            runtime_context["joint_gamma_result_sha256"],
        ),
        "inherited_head_mass_protocol": (
            runtime_context["head_mass_protocol_path"],
            runtime_context["head_mass_protocol_sha256"],
        ),
        "inherited_head_mass_result": (
            runtime_context["head_mass_result_path"],
            runtime_context["head_mass_result_sha256"],
        ),
        "inherited_head_mass_manifest": (
            runtime_context["head_mass_manifest_path"],
            runtime_context["head_mass_manifest_sha256"],
        ),
        "train_split": (
            runtime_context["train_path"],
            cached_context["train_sha256"],
        ),
    }
    for name, (path, digest) in path_bindings.items():
        if historical.get(name) != _binding(Path(path), digest):
            raise ValueError(f"full-visible historical {name} binding changed")
    if (
        runtime_context["train_records"] != cached_context["train_records"]
        or runtime_context["historical_output_rows"]
        != cached_context["head_mass_result"]["base_output_authentication"]
        or Path(runtime_context["non_mlp_path"])
        != Path(cached_context["projection_path"])
        or sha256_file(runtime_context["non_mlp_path"])
        != protocol["output_projection"]["file_sha256"]
        or protocol.get("resource_contract") != result.get("resource_contract")
    ):
        raise ValueError("full-visible native and cached evidence diverged")

    context = dict(runtime_context)
    context.update(
        {
            "predecessor_protocol_path": protocol_path,
            "predecessor_protocol_sha256": predecessor_protocol_sha256.lower(),
            "predecessor_protocol": protocol,
            "predecessor_result_path": result_path,
            "predecessor_result_sha256": predecessor_result_sha256.lower(),
            "predecessor_result": result,
            "predecessor_capture_path": capture_path,
            "predecessor_capture_sha256": capture_binding["sha256"],
            "predecessor_capture": capture_report,
            "cached_context": cached_context,
            "full_visible_trace_library_path": library_path,
            "full_visible_trace_library_sha256": trace_library_sha256.lower(),
        }
    )
    return context


def _open_full_visible_runtime(context: Mapping[str, Any]) -> Any:
    return slot._open_slot_trace_runtime(
        context,
        c28_qk_partial_trace=bool(context.get("c28_qk_partial_trace", False)),
        c28_qk_candidate_trace=bool(context.get("c28_qk_candidate_trace", False)),
        c28_qk_candidate_key_trace=bool(
            context.get("c28_qk_candidate_key_trace", False)
        ),
        c28_qk_candidate_value_trace=bool(
            context.get("c28_qk_candidate_value_trace", False)
        ),
    )


def _validate_runtime_route(runtime: Any) -> None:
    slot._validate_runtime_route(
        runtime,
        allow_c28_qk=(
            getattr(runtime, "c28_qk_partial_trace_available", False) is True
        ),
        allow_c28_qk_candidates=(
            getattr(runtime, "c28_qk_candidate_trace_available", False) is True
        ),
        allow_c28_qk_candidate_keys=(
            getattr(runtime, "c28_qk_candidate_key_trace_available", False) is True
        ),
        allow_c28_qk_candidate_values=(
            getattr(runtime, "c28_qk_candidate_value_trace_available", False)
            is True
        ),
    )
    if runtime.regular_entry_trace_available is not True or not callable(
        getattr(runtime, "last_regular_entry_trace", None)
    ):
        raise ValueError("full-visible native runtime route changed")


def _historical_base_arrays(
    context: Mapping[str, Any],
    *,
    record_index: int,
) -> dict[str, np.ndarray]:
    return slot._historical_head_mass_arrays(
        context,
        record_index=record_index,
    )


def _record_schedule(
    context: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    return slot.mass.capacity.bias.rank.fixed._derive_schedule(
        record["input_ids"],
        context["head_mass_protocol"]["schedule_contract"]["tokenizer_fact_anchor_ids"],
    )


def _execute_record_pair(
    trace: Any,
    *,
    context: Mapping[str, Any],
    record_index: int,
    progress_prefix: str,
) -> dict[str, Any]:
    """Execute one authenticated train record before and after a real reset."""

    record = context["train_records"][record_index]
    historical_output = context["historical_output_rows"][record_index]
    historical_arrays = _historical_base_arrays(
        context,
        record_index=record_index,
    )
    schedule = _record_schedule(context, record)
    resource = context["head_mass_protocol"]["fixed_K256_arm"]["resource_contract"]
    first, arrays, positions = slot._execute_record(
        trace,
        record=record,
        context=context,
        schedule=schedule,
        resource=resource,
        progress_label=f"{progress_prefix} first",
    )
    first_summary = full._trace_summary(arrays, positions)
    qk_capture = (
        getattr(trace, "qk_captured", None)
        if getattr(trace, "c28_qk_partial_trace_available", False) is True
        else None
    )
    first_qk = (
        np.ascontiguousarray(qk_capture()) if callable(qk_capture) else None
    )
    qk_candidate_capture = (
        getattr(trace, "qk_candidates_captured", None)
        if getattr(trace, "c28_qk_candidate_trace_available", False) is True
        else None
    )
    first_qk_candidates = (
        np.ascontiguousarray(qk_candidate_capture())
        if callable(qk_candidate_capture)
        else None
    )
    qk_candidate_key_capture = (
        getattr(trace, "qk_candidate_keys_captured", None)
        if getattr(trace, "c28_qk_candidate_key_trace_available", False) is True
        else None
    )
    first_qk_candidate_keys = (
        np.ascontiguousarray(qk_candidate_key_capture())
        if callable(qk_candidate_key_capture)
        else None
    )
    qk_candidate_value_capture = (
        getattr(trace, "qk_candidate_values_captured", None)
        if getattr(trace, "c28_qk_candidate_value_trace_available", False) is True
        else None
    )
    first_qk_candidate_values = (
        np.ascontiguousarray(qk_candidate_value_capture())
        if callable(qk_candidate_value_capture)
        else None
    )
    trace.reset()
    replay, replay_arrays, replay_positions = slot._execute_record(
        trace,
        record=record,
        context=context,
        schedule=schedule,
        resource=resource,
        progress_label=f"{progress_prefix} reset",
    )
    replay_summary = full._trace_summary(replay_arrays, replay_positions)
    reset_qk = (
        np.ascontiguousarray(qk_capture()) if callable(qk_capture) else None
    )
    reset_qk_candidates = (
        np.ascontiguousarray(qk_candidate_capture())
        if callable(qk_candidate_capture)
        else None
    )
    reset_qk_candidate_keys = (
        np.ascontiguousarray(qk_candidate_key_capture())
        if callable(qk_candidate_key_capture)
        else None
    )
    reset_qk_candidate_values = (
        np.ascontiguousarray(qk_candidate_value_capture())
        if callable(qk_candidate_value_capture)
        else None
    )

    source_sha256 = sha256_json(record)
    first_output_sha256 = sha256_json(slot.mass.capacity._without_elapsed(first))
    reset_output_sha256 = sha256_json(slot.mass.capacity._without_elapsed(replay))
    inherited_first = slot._common_trace_exact(arrays, historical_arrays)
    inherited_reset = slot._common_trace_exact(
        replay_arrays,
        historical_arrays,
    )
    checks = {
        "source_record_root_exact": (
            source_sha256 == historical_output.get("source_record_sha256")
        ),
        "historical_first_outputs_counters_and_loss_exact": (
            slot._historical_output_exact(first, historical_output)
        ),
        "historical_reset_outputs_counters_and_loss_exact": (
            slot._historical_output_exact(replay, historical_output)
        ),
        "first_reset_outputs_counters_and_loss_exact": (
            slot.mass.capacity._evidence_exact(first, replay)
        ),
        "inherited_first_base_trace_exact": inherited_first,
        "inherited_reset_base_trace_exact": inherited_reset,
        "first_reset_trace_summary_exact": first_summary == replay_summary,
        "first_reset_trace_tensors_exact": all(
            np.array_equal(arrays[name], replay_arrays[name])
            for name in full._CAPTURE_TRACE_KEYS
        ),
        "historical_output_root_exact": (
            first_output_sha256
            == historical_output.get("observed_output_evidence_sha256")
        ),
        "reset_output_root_exact": reset_output_sha256 == first_output_sha256,
        "reset_trace_root_exact": (
            replay_summary["trace_sha256"] == first_summary["trace_sha256"]
        ),
    }
    if first_qk is not None or reset_qk is not None:
        if first_qk is None or reset_qk is None:
            raise ValueError("full-visible blockwise-QK capture was incomplete")
        first_qk_summary = full._qk_trace_summary(first_qk, positions)
        reset_qk_summary = full._qk_trace_summary(reset_qk, replay_positions)
        checks["first_reset_qk_trace_exact"] = bool(
            np.array_equal(first_qk, reset_qk)
            and first_qk_summary["trace_sha256"] == reset_qk_summary["trace_sha256"]
        )
    if first_qk_candidates is not None or reset_qk_candidates is not None:
        if first_qk_candidates is None or reset_qk_candidates is None:
            raise ValueError("full-visible QK candidate capture was incomplete")
        first_qk_candidate_summary = full._qk_candidate_trace_summary(
            first_qk_candidates, positions
        )
        reset_qk_candidate_summary = full._qk_candidate_trace_summary(
            reset_qk_candidates, replay_positions
        )
        checks["first_reset_qk_candidate_trace_exact"] = bool(
            np.array_equal(first_qk_candidates, reset_qk_candidates)
            and first_qk_candidate_summary["trace_sha256"]
            == reset_qk_candidate_summary["trace_sha256"]
        )
    if first_qk_candidate_keys is not None or reset_qk_candidate_keys is not None:
        if first_qk_candidate_keys is None or reset_qk_candidate_keys is None:
            raise ValueError("full-visible QK candidate-key capture was incomplete")
        first_key_summary = full._qk_candidate_key_trace_summary(
            first_qk_candidate_keys, positions
        )
        reset_key_summary = full._qk_candidate_key_trace_summary(
            reset_qk_candidate_keys, replay_positions
        )
        checks["first_reset_qk_candidate_key_trace_exact"] = bool(
            np.array_equal(first_qk_candidate_keys, reset_qk_candidate_keys)
            and first_key_summary["trace_sha256"]
            == reset_key_summary["trace_sha256"]
        )
    if first_qk_candidate_values is not None or reset_qk_candidate_values is not None:
        if first_qk_candidate_values is None or reset_qk_candidate_values is None:
            raise ValueError("full-visible QK candidate-value capture was incomplete")
        first_value_summary = full._qk_candidate_value_trace_summary(
            first_qk_candidate_values, positions
        )
        reset_value_summary = full._qk_candidate_value_trace_summary(
            reset_qk_candidate_values, replay_positions
        )
        checks["first_reset_qk_candidate_value_trace_exact"] = bool(
            np.array_equal(first_qk_candidate_values, reset_qk_candidate_values)
            and first_value_summary["trace_sha256"]
            == reset_value_summary["trace_sha256"]
        )
    checks["passed"] = all(checks.values())
    if checks["passed"] is not True:
        raise ValueError(f"full-visible record {record_index} parity failed")
    return {
        "record_index": record_index,
        "record_id": record["record_id"],
        "schedule_rows_sha256": schedule["rows_sha256"],
        "source_record_sha256": source_sha256,
        "first_output_evidence_sha256": first_output_sha256,
        "reset_output_evidence_sha256": reset_output_sha256,
        "first_trace": first_summary,
        "reset_trace": replay_summary,
        "arrays": arrays,
        "query_positions": positions,
        "checks": checks,
        **(
            {
                "qk_partials": first_qk,
                "reset_qk_partials": reset_qk,
                "first_qk_trace": full._qk_trace_summary(first_qk, positions),
                "reset_qk_trace": full._qk_trace_summary(reset_qk, replay_positions),
            }
            if first_qk is not None and reset_qk is not None
            else {}
        ),
        **(
            {
                "qk_candidate_keys": first_qk_candidate_keys,
                "reset_qk_candidate_keys": reset_qk_candidate_keys,
                "first_qk_candidate_key_trace": full._qk_candidate_key_trace_summary(
                    first_qk_candidate_keys, positions
                ),
                "reset_qk_candidate_key_trace": full._qk_candidate_key_trace_summary(
                    reset_qk_candidate_keys, replay_positions
                ),
            }
            if first_qk_candidate_keys is not None
            and reset_qk_candidate_keys is not None
            else {}
        ),
        **(
            {
                "qk_candidates": first_qk_candidates,
                "reset_qk_candidates": reset_qk_candidates,
                "first_qk_candidate_trace": full._qk_candidate_trace_summary(
                    first_qk_candidates, positions
                ),
                "reset_qk_candidate_trace": full._qk_candidate_trace_summary(
                    reset_qk_candidates, replay_positions
                ),
            }
            if first_qk_candidates is not None and reset_qk_candidates is not None
            else {}
        ),
        **(
            {
                "qk_candidate_values": first_qk_candidate_values,
                "reset_qk_candidate_values": reset_qk_candidate_values,
                "first_qk_candidate_value_trace": full._qk_candidate_value_trace_summary(
                    first_qk_candidate_values, positions
                ),
                "reset_qk_candidate_value_trace": full._qk_candidate_value_trace_summary(
                    reset_qk_candidate_values, replay_positions
                ),
            }
            if first_qk_candidate_values is not None
            and reset_qk_candidate_values is not None
            else {}
        ),
    }


def _post_run_authentication(context: Mapping[str, Any]) -> dict[str, bool]:
    cached_context = context["cached_context"]
    checks = {
        "predecessor_protocol": (
            sha256_file(context["predecessor_protocol_path"])
            == context["predecessor_protocol_sha256"]
        ),
        "predecessor_result": (
            sha256_file(context["predecessor_result_path"])
            == context["predecessor_result_sha256"]
        ),
        "predecessor_capture_report": (
            sha256_file(context["predecessor_capture_path"])
            == context["predecessor_capture_sha256"]
        ),
        "joint_gamma_protocol": (
            sha256_file(context["joint_gamma_protocol_path"])
            == context["joint_gamma_protocol_sha256"]
        ),
        "joint_gamma_result": (
            sha256_file(context["joint_gamma_result_path"])
            == context["joint_gamma_result_sha256"]
        ),
        "head_mass_protocol": (
            sha256_file(context["head_mass_protocol_path"])
            == context["head_mass_protocol_sha256"]
        ),
        "head_mass_result": (
            sha256_file(context["head_mass_result_path"])
            == context["head_mass_result_sha256"]
        ),
        "head_mass_manifest": (
            sha256_file(context["head_mass_manifest_path"])
            == context["head_mass_manifest_sha256"]
        ),
        "training_checkpoint": (
            sha256_file(context["training_checkpoint_path"])
            == cached_context["training_checkpoint_sha256"]
        ),
        "selector_protocol": (
            sha256_file(cached_context["selector_protocol_path"])
            == cached_context["selector_protocol_sha256"]
        ),
        "corpus_manifest": (
            sha256_file(cached_context["corpus_manifest_path"])
            == cached_context["corpus_manifest_sha256"]
        ),
        "train_split": (
            sha256_file(context["train_path"]) == cached_context["train_sha256"]
        ),
        "package_manifest": (
            sha256_file(context["package_manifest_path"])
            == context["head_mass_protocol"]["package"]["manifest_sha256"]
        ),
        "config_artifact": (
            sha256_file(context["config_path"])
            == context["head_mass_protocol"]["package"]["files"]["model/config.json"][
                "sha256"
            ]
            if "files" in context["head_mass_protocol"]["package"]
            else True
        ),
        "output_projection_file": (
            sha256_file(context["non_mlp_path"])
            == context["predecessor_protocol"]["output_projection"]["file_sha256"]
        ),
        "trace_library": (
            sha256_file(context["full_visible_trace_library_path"])
            == context["full_visible_trace_library_sha256"]
        ),
        "confirmation_not_opened": True,
    }
    return checks


def _runner_source_sha256() -> str:
    repository = Path(__file__).resolve().parents[3]
    return sha256_file(repository / _RUNNER_SOURCE)


def generate_full_visible_trace_parity_report(
    *,
    predecessor_protocol: str | Path,
    predecessor_protocol_sha256: str,
    predecessor_result: str | Path,
    predecessor_result_sha256: str,
    trace_library: str | Path,
    trace_library_sha256: str,
    out: str | Path,
    runtime_factory: Callable[[Mapping[str, Any]], Any] = (_open_full_visible_runtime),
) -> dict[str, Any]:
    """Run the real first/reset parity protocol on train record zero."""

    _guard_paths(
        (
            ("predecessor protocol", predecessor_protocol),
            ("predecessor result", predecessor_result),
            ("trace library", trace_library),
            ("parity output", out),
        )
    )
    output = _new_output(out, "full-visible parity output")
    context = _authenticate_predecessor_inputs(
        predecessor_protocol=predecessor_protocol,
        predecessor_protocol_sha256=predecessor_protocol_sha256,
        predecessor_result=predecessor_result,
        predecessor_result_sha256=predecessor_result_sha256,
        trace_library=trace_library,
        trace_library_sha256=trace_library_sha256,
    )
    raw = runtime_factory(context)
    trace = full._FullVisibleTraceCaptureRuntime(raw)
    try:
        _validate_runtime_route(raw)
        evidence = _execute_record_pair(
            trace,
            context=context,
            record_index=0,
            progress_prefix="full-visible parity",
        )
    finally:
        trace.close()
    post = _post_run_authentication(context)
    if not all(post.values()):
        raise ValueError("full-visible parity post-run authentication failed")
    report = full.build_trace_parity_report(
        predecessor_protocol=_binding(
            context["predecessor_protocol_path"],
            context["predecessor_protocol_sha256"],
        ),
        predecessor_result=_binding(
            context["predecessor_result_path"],
            context["predecessor_result_sha256"],
        ),
        trace_library=_binding(
            context["full_visible_trace_library_path"],
            context["full_visible_trace_library_sha256"],
        ),
        first_trace=evidence["first_trace"],
        reset_trace=evidence["reset_trace"],
        inherited_base_trace_exact=(
            evidence["checks"]["inherited_first_base_trace_exact"]
            and evidence["checks"]["inherited_reset_base_trace_exact"]
        ),
        outputs_counters_and_loss_exact=(
            evidence["checks"]["historical_first_outputs_counters_and_loss_exact"]
            and evidence["checks"]["historical_reset_outputs_counters_and_loss_exact"]
            and evidence["checks"]["first_reset_outputs_counters_and_loss_exact"]
        ),
    )
    report["execution_evidence"] = {
        name: evidence[name]
        for name in (
            "record_index",
            "record_id",
            "schedule_rows_sha256",
            "source_record_sha256",
            "first_output_evidence_sha256",
            "reset_output_evidence_sha256",
            "checks",
        )
    }
    report["instrumentation_source_sha256"] = full._source_inventory()
    report["runner_source_sha256"] = {_RUNNER_SOURCE: _runner_source_sha256()}
    report["post_run_authentication"] = post
    atomic_json(output, report)
    _progress(f"parity report written to {output}")
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "report": report,
    }


def _validate_runner_parity_extension(
    parity: Mapping[str, Any],
    *,
    expected_source_inventory: Mapping[str, str] | None = None,
) -> None:
    execution = parity.get("execution_evidence")
    post = parity.get("post_run_authentication")
    runner = parity.get("runner_source_sha256")
    inventory = parity.get("instrumentation_source_sha256")
    if (
        not isinstance(execution, Mapping)
        or execution.get("record_index") != 0
        or execution.get("first_output_evidence_sha256")
        != execution.get("reset_output_evidence_sha256")
        or not isinstance(execution.get("checks"), Mapping)
        or execution["checks"].get("passed") is not True
        or not all(check is True for check in execution["checks"].values())
        or not isinstance(post, Mapping)
        or not post
        or not all(check is True for check in post.values())
        or runner != {_RUNNER_SOURCE: _runner_source_sha256()}
        or inventory != full._source_inventory()
        or (
            expected_source_inventory is not None
            and inventory != expected_source_inventory
        )
    ):
        raise ValueError("full-visible runner parity extension changed")


def freeze_full_visible_protocol(
    *,
    predecessor_protocol: str | Path,
    predecessor_protocol_sha256: str,
    predecessor_result: str | Path,
    predecessor_result_sha256: str,
    trace_library: str | Path,
    trace_library_sha256: str,
    parity_report: str | Path,
    parity_report_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    """Validate runner evidence, then delegate the protocol freeze."""

    _guard_paths(
        (
            ("predecessor protocol", predecessor_protocol),
            ("predecessor result", predecessor_result),
            ("trace library", trace_library),
            ("parity report", parity_report),
            ("protocol output", out),
        )
    )
    parity_path = full._checked_file(
        parity_report,
        parity_report_sha256,
        "full-visible parity report",
    )
    parity = _read_json(parity_path, "full-visible parity report")
    _validate_runner_parity_extension(parity)
    result = full.freeze_full_visible_protocol(
        predecessor_protocol=predecessor_protocol,
        predecessor_protocol_sha256=predecessor_protocol_sha256,
        predecessor_result=predecessor_result,
        predecessor_result_sha256=predecessor_result_sha256,
        trace_library=trace_library,
        trace_library_sha256=trace_library_sha256,
        parity_report=parity_path,
        parity_report_sha256=parity_report_sha256,
        out=out,
    )
    protocol = result["protocol"]
    if (
        protocol.get("source_sha256") != full._source_inventory()
        or protocol.get("trace_library", {}).get("sha256")
        != trace_library_sha256.lower()
        or protocol.get("trace_parity", {}).get("sha256")
        != parity_report_sha256.lower()
    ):
        raise ValueError("full-visible delegated protocol binding changed")
    return result


def _authenticate_frozen_capture_context(
    *,
    protocol: str | Path,
    protocol_sha256: str,
) -> dict[str, Any]:
    protocol_path = full._checked_file(
        protocol,
        protocol_sha256,
        "full-visible protocol",
    )
    frozen = _read_json(protocol_path, "full-visible protocol")
    predecessor = frozen.get("predecessor_protocol")
    predecessor_result = frozen.get("predecessor_result")
    library = frozen.get("trace_library")
    parity_binding = frozen.get("trace_parity")
    if (
        frozen.get("schema_version") != full._SCHEMA_VERSION
        or frozen.get("experiment") != full._PROTOCOL_EXPERIMENT
        or frozen.get("status") != full._PROTOCOL_STATUS
        or frozen.get("confirmation_split_opened") is not False
        or frozen.get("source_sha256") != full._source_inventory()
        or not all(
            isinstance(value, Mapping)
            for value in (
                predecessor,
                predecessor_result,
                library,
                parity_binding,
            )
        )
    ):
        raise ValueError("full-visible frozen protocol changed")
    context = _authenticate_predecessor_inputs(
        predecessor_protocol=predecessor.get("path"),
        predecessor_protocol_sha256=predecessor.get("sha256"),
        predecessor_result=predecessor_result.get("path"),
        predecessor_result_sha256=predecessor_result.get("sha256"),
        trace_library=library.get("path"),
        trace_library_sha256=library.get("sha256"),
    )
    parity_path = full._checked_file(
        parity_binding.get("path"),
        parity_binding.get("sha256"),
        "full-visible parity report",
    )
    parity = _read_json(parity_path, "full-visible parity report")
    full._validate_parity_report(
        parity,
        predecessor_protocol=_binding(
            context["predecessor_protocol_path"],
            context["predecessor_protocol_sha256"],
        ),
        predecessor_result=_binding(
            context["predecessor_result_path"],
            context["predecessor_result_sha256"],
        ),
        trace_library=_binding(
            context["full_visible_trace_library_path"],
            context["full_visible_trace_library_sha256"],
        ),
    )
    _validate_runner_parity_extension(
        parity,
        expected_source_inventory=frozen["source_sha256"],
    )
    if (
        frozen.get("historical_bindings")
        != context["predecessor_protocol"]["historical_bindings"]
        or frozen.get("output_projection")
        != context["predecessor_protocol"]["output_projection"]
        or frozen.get("resource_contract", {}).get(
            "fixed_combined_attention_and_episodic_traffic_bytes"
        )
        != context["predecessor_protocol"]["resource_contract"][
            "fixed_combined_attention_and_episodic_traffic_bytes"
        ]
    ):
        raise ValueError("full-visible frozen inherited binding changed")
    context.update(
        {
            "full_visible_protocol_path": protocol_path,
            "full_visible_protocol_sha256": protocol_sha256.lower(),
            "full_visible_protocol": frozen,
            "full_visible_parity_path": parity_path,
            "full_visible_parity_sha256": parity_binding["sha256"],
        }
    )
    return context


def _capture_post_authentication(
    context: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
) -> dict[str, bool]:
    inherited = _post_run_authentication(context)
    return {
        **inherited,
        "full_visible_protocol": (
            sha256_file(context["full_visible_protocol_path"])
            == context["full_visible_protocol_sha256"]
        ),
        "full_visible_parity": (
            sha256_file(context["full_visible_parity_path"])
            == context["full_visible_parity_sha256"]
        ),
        "full_visible_source_inventory": (
            context["full_visible_protocol"]["source_sha256"]
            == full._source_inventory()
        ),
        "full_visible_manifest": (sha256_file(manifest_path) == manifest_sha256),
        "confirmation_not_opened": True,
    }


def capture_full_visible_train_traces(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    shard_directory: str | Path,
    runtime_factory: Callable[[Mapping[str, Any]], Any] = (_open_full_visible_runtime),
) -> dict[str, Any]:
    """Capture reset-proven full-visible traces for all eight train records."""

    _guard_paths(
        (
            ("full-visible protocol", protocol),
            ("trace shard directory", shard_directory),
        )
    )
    context = _authenticate_frozen_capture_context(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    directory = full._prepare_shard_directory(shard_directory)
    raw = runtime_factory(context)
    trace = full._FullVisibleTraceCaptureRuntime(raw)
    descriptors: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    try:
        _validate_runtime_route(raw)
        for record_index in range(full._RECORDS):
            if trace.position != 0:
                trace.reset()
            _progress(f"capturing train record {record_index + 1}/{full._RECORDS}")
            evidence = _execute_record_pair(
                trace,
                context=context,
                record_index=record_index,
                progress_prefix=f"full-visible train {record_index}",
            )
            descriptor = full.write_full_visible_trace_shard(
                directory,
                record_index=record_index,
                record_id=evidence["record_id"],
                arrays=evidence["arrays"],
                query_positions=evidence["query_positions"],
                source_record_sha256=evidence["source_record_sha256"],
                output_evidence_sha256=(evidence["first_output_evidence_sha256"]),
                reset_output_evidence_sha256=(evidence["reset_output_evidence_sha256"]),
                reset_trace_sha256=evidence["reset_trace"]["trace_sha256"],
                schedule_rows_sha256=evidence["schedule_rows_sha256"],
            )
            descriptors.append(descriptor)
            capture_rows.append(
                {
                    name: evidence[name]
                    for name in (
                        "record_index",
                        "record_id",
                        "schedule_rows_sha256",
                        "source_record_sha256",
                        "first_output_evidence_sha256",
                        "reset_output_evidence_sha256",
                        "checks",
                    )
                }
            )
    finally:
        trace.close()
    protocol_binding = _binding(
        context["full_visible_protocol_path"],
        context["full_visible_protocol_sha256"],
    )
    manifest = full.write_full_visible_trace_manifest(
        directory,
        protocol=protocol_binding,
        shards=descriptors,
    )
    manifest_path = Path(manifest["path"])
    full.load_stacked_full_visible_trace(
        manifest_path,
        manifest["sha256"],
        protocol=protocol_binding,
    )
    post = _capture_post_authentication(
        context,
        manifest_path=manifest_path,
        manifest_sha256=manifest["sha256"],
    )
    if not all(post.values()):
        raise ValueError("full-visible capture post-run authentication failed")
    _progress(f"capture manifest written to {manifest_path}")
    return {
        **manifest,
        "capture_rows": capture_rows,
        "capture_rows_sha256": sha256_json(capture_rows),
        "post_run_authentication": post,
        "confirmation_split_opened": False,
    }


def capture_full_visible_train_qk_traces(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    shard_directory: str | Path,
    runtime_factory: Callable[[Mapping[str, Any]], Any] = (_open_full_visible_runtime),
) -> dict[str, Any]:
    """Capture reset-proven C28 blockwise-QK features for the train split.

    This deliberately reuses the authenticated full-visible execution pair.
    The ordinary value trace is still checked against its historical roots on
    every record, while the QK tensor is written to a separate manifest so it
    cannot alter the existing causal value-basis solver.
    """

    _guard_paths(
        (
            ("full-visible protocol", protocol),
            ("blockwise-QK shard directory", shard_directory),
        )
    )
    context = _authenticate_frozen_capture_context(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    qk_context = dict(context)
    qk_context["c28_qk_partial_trace"] = True
    directory = full._prepare_shard_directory(shard_directory)
    raw = runtime_factory(qk_context)
    trace = full._FullVisibleTraceCaptureRuntime(raw)
    descriptors: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    try:
        _validate_runtime_route(raw)
        if getattr(raw, "c28_qk_partial_trace_available", False) is not True:
            raise ValueError("full-visible native runtime lacks C28 QK trace route")
        for record_index in range(full._RECORDS):
            if trace.position != 0:
                trace.reset()
            _progress(
                f"capturing train blockwise-QK record "
                f"{record_index + 1}/{full._RECORDS}"
            )
            evidence = _execute_record_pair(
                trace,
                context=context,
                record_index=record_index,
                progress_prefix=f"full-visible blockwise-QK train {record_index}",
            )
            if "qk_partials" not in evidence or "reset_qk_partials" not in evidence:
                raise ValueError("full-visible blockwise-QK capture was unavailable")
            descriptor = full.write_full_visible_qk_trace_shard(
                directory,
                record_index=record_index,
                record_id=evidence["record_id"],
                qk_partials=evidence["qk_partials"],
                reset_qk_partials=evidence["reset_qk_partials"],
                query_positions=evidence["query_positions"],
                source_record_sha256=evidence["source_record_sha256"],
                output_evidence_sha256=evidence["first_output_evidence_sha256"],
                reset_output_evidence_sha256=evidence["reset_output_evidence_sha256"],
                schedule_rows_sha256=evidence["schedule_rows_sha256"],
            )
            descriptors.append(descriptor)
            capture_rows.append(
                {
                    name: evidence[name]
                    for name in (
                        "record_index",
                        "record_id",
                        "schedule_rows_sha256",
                        "source_record_sha256",
                        "first_output_evidence_sha256",
                        "reset_output_evidence_sha256",
                        "checks",
                        "first_qk_trace",
                        "reset_qk_trace",
                    )
                }
            )
    finally:
        trace.close()
    protocol_binding = _binding(
        context["full_visible_protocol_path"],
        context["full_visible_protocol_sha256"],
    )
    manifest = full.write_full_visible_qk_trace_manifest(
        directory,
        protocol=protocol_binding,
        shards=descriptors,
    )
    manifest_path = Path(manifest["path"])
    full.load_stacked_full_visible_qk_trace(
        manifest_path,
        manifest["sha256"],
        protocol=protocol_binding,
    )
    post = _post_run_authentication(context)
    post.update(
        {
            "blockwise_qk_manifest": (
                sha256_file(manifest_path) == manifest["sha256"]
            ),
            "blockwise_qk_source_inventory": (
                context["full_visible_protocol"]["source_sha256"]
                == full._source_inventory()
            ),
        }
    )
    if not all(post.values()):
        raise ValueError("full-visible blockwise-QK capture post-run authentication failed")
    _progress(f"blockwise-QK capture manifest written to {manifest_path}")
    return {
        **manifest,
        "capture_rows": capture_rows,
        "capture_rows_sha256": sha256_json(capture_rows),
        "post_run_authentication": post,
        "confirmation_split_opened": False,
    }


def capture_full_visible_train_qk_candidate_traces(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    shard_directory: str | Path,
    runtime_factory: Callable[[Mapping[str, Any]], Any] = (_open_full_visible_runtime),
) -> dict[str, Any]:
    """Capture every older candidate score before native top-K selection."""

    _guard_paths(
        (
            ("full-visible protocol", protocol),
            ("QK candidate shard directory", shard_directory),
        )
    )
    context = _authenticate_frozen_capture_context(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    candidate_context = dict(context)
    candidate_context["c28_qk_candidate_trace"] = True
    directory = full._prepare_shard_directory(shard_directory)
    raw = runtime_factory(candidate_context)
    trace = full._FullVisibleTraceCaptureRuntime(raw)
    descriptors: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    try:
        _validate_runtime_route(raw)
        if getattr(raw, "c28_qk_candidate_trace_available", False) is not True:
            raise ValueError("full-visible native runtime lacks QK candidate route")
        for record_index in range(full._RECORDS):
            if trace.position != 0:
                trace.reset()
            _progress(
                f"capturing train QK candidates record "
                f"{record_index + 1}/{full._RECORDS}"
            )
            evidence = _execute_record_pair(
                trace,
                context=context,
                record_index=record_index,
                progress_prefix=f"full-visible QK candidate train {record_index}",
            )
            if (
                "qk_candidates" not in evidence
                or "reset_qk_candidates" not in evidence
            ):
                raise ValueError("full-visible QK candidate capture was unavailable")
            descriptor = full.write_full_visible_qk_candidate_trace_shard(
                directory,
                record_index=record_index,
                record_id=evidence["record_id"],
                qk_candidates=evidence["qk_candidates"],
                reset_qk_candidates=evidence["reset_qk_candidates"],
                query_positions=evidence["query_positions"],
                source_record_sha256=evidence["source_record_sha256"],
                output_evidence_sha256=evidence["first_output_evidence_sha256"],
                reset_output_evidence_sha256=evidence["reset_output_evidence_sha256"],
                schedule_rows_sha256=evidence["schedule_rows_sha256"],
            )
            descriptors.append(descriptor)
            capture_rows.append(
                {
                    name: evidence[name]
                    for name in (
                        "record_index",
                        "record_id",
                        "schedule_rows_sha256",
                        "source_record_sha256",
                        "first_output_evidence_sha256",
                        "reset_output_evidence_sha256",
                        "checks",
                        "first_qk_candidate_trace",
                        "reset_qk_candidate_trace",
                    )
                }
            )
    finally:
        trace.close()
    protocol_binding = _binding(
        context["full_visible_protocol_path"],
        context["full_visible_protocol_sha256"],
    )
    manifest = full.write_full_visible_qk_candidate_trace_manifest(
        directory,
        protocol=protocol_binding,
        shards=descriptors,
    )
    manifest_path = Path(manifest["path"])
    full.load_stacked_full_visible_qk_candidate_trace(
        manifest_path,
        manifest["sha256"],
        protocol=protocol_binding,
    )
    post = _post_run_authentication(context)
    post.update(
        {
            "qk_candidate_manifest": (
                sha256_file(manifest_path) == manifest["sha256"]
            ),
            "qk_candidate_source_inventory": (
                context["full_visible_protocol"]["source_sha256"]
                == full._source_inventory()
            ),
        }
    )
    if not all(post.values()):
        raise ValueError("full-visible QK candidate capture authentication failed")
    _progress(f"QK candidate capture manifest written to {manifest_path}")
    return {
        **manifest,
        "capture_rows": capture_rows,
        "capture_rows_sha256": sha256_json(capture_rows),
        "post_run_authentication": post,
        "confirmation_split_opened": False,
    }


def capture_full_visible_train_qk_candidate_key_traces(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    shard_directory: str | Path,
    runtime_factory: Callable[[Mapping[str, Any]], Any] = (_open_full_visible_runtime),
) -> dict[str, Any]:
    """Capture exact post-RoPE keys for every older candidate slot."""

    _guard_paths(
        (
            ("full-visible protocol", protocol),
            ("QK candidate-key shard directory", shard_directory),
        )
    )
    context = _authenticate_frozen_capture_context(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    key_context = dict(context)
    key_context["c28_qk_candidate_key_trace"] = True
    directory = full._prepare_shard_directory(shard_directory)
    raw = runtime_factory(key_context)
    trace = full._FullVisibleTraceCaptureRuntime(raw)
    descriptors: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    try:
        _validate_runtime_route(raw)
        if getattr(raw, "c28_qk_candidate_key_trace_available", False) is not True:
            raise ValueError("full-visible native runtime lacks QK candidate-key route")
        for record_index in range(full._RECORDS):
            if trace.position != 0:
                trace.reset()
            _progress(
                f"capturing train QK candidate keys record "
                f"{record_index + 1}/{full._RECORDS}"
            )
            evidence = _execute_record_pair(
                trace,
                context=context,
                record_index=record_index,
                progress_prefix=f"full-visible QK candidate-key train {record_index}",
            )
            if (
                "qk_candidate_keys" not in evidence
                or "reset_qk_candidate_keys" not in evidence
            ):
                raise ValueError("full-visible QK candidate-key capture was unavailable")
            descriptor = full.write_full_visible_qk_candidate_key_trace_shard(
                directory,
                record_index=record_index,
                record_id=evidence["record_id"],
                candidate_keys=evidence["qk_candidate_keys"],
                reset_candidate_keys=evidence["reset_qk_candidate_keys"],
                query_positions=evidence["query_positions"],
                source_record_sha256=evidence["source_record_sha256"],
                output_evidence_sha256=evidence["first_output_evidence_sha256"],
                reset_output_evidence_sha256=evidence["reset_output_evidence_sha256"],
                schedule_rows_sha256=evidence["schedule_rows_sha256"],
            )
            descriptors.append(descriptor)
            capture_rows.append(
                {
                    name: evidence[name]
                    for name in (
                        "record_index",
                        "record_id",
                        "schedule_rows_sha256",
                        "source_record_sha256",
                        "first_output_evidence_sha256",
                        "reset_output_evidence_sha256",
                        "checks",
                        "first_qk_candidate_key_trace",
                        "reset_qk_candidate_key_trace",
                    )
                }
            )
    finally:
        trace.close()
    protocol_binding = _binding(
        context["full_visible_protocol_path"],
        context["full_visible_protocol_sha256"],
    )
    manifest = full.write_full_visible_qk_candidate_key_trace_manifest(
        directory,
        protocol=protocol_binding,
        shards=descriptors,
    )
    manifest_path = Path(manifest["path"])
    full.load_stacked_full_visible_qk_candidate_key_trace(
        manifest_path,
        manifest["sha256"],
        protocol=protocol_binding,
    )
    post = _post_run_authentication(context)
    post.update(
        {
            "qk_candidate_key_manifest": (
                sha256_file(manifest_path) == manifest["sha256"]
            ),
            "qk_candidate_key_source_inventory": (
                context["full_visible_protocol"]["source_sha256"]
                == full._source_inventory()
            ),
        }
    )
    if not all(post.values()):
        raise ValueError(
            "full-visible QK candidate-key capture post-run authentication failed"
        )
    _progress(f"QK candidate-key capture manifest written to {manifest_path}")
    return {
        **manifest,
        "capture_rows": capture_rows,
        "capture_rows_sha256": sha256_json(capture_rows),
        "post_run_authentication": post,
        "confirmation_split_opened": False,
    }


def capture_full_visible_train_qk_candidate_value_traces(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    shard_directory: str | Path,
    runtime_factory: Callable[[Mapping[str, Any]], Any] = (_open_full_visible_runtime),
) -> dict[str, Any]:
    """Capture exact older-candidate values for causal replay."""

    _guard_paths(
        (
            ("full-visible protocol", protocol),
            ("QK candidate-value shard directory", shard_directory),
        )
    )
    context = _authenticate_frozen_capture_context(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    value_context = dict(context)
    value_context["c28_qk_candidate_value_trace"] = True
    directory = full._prepare_shard_directory(shard_directory)
    raw = runtime_factory(value_context)
    trace = full._FullVisibleTraceCaptureRuntime(raw)
    descriptors: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    try:
        _validate_runtime_route(raw)
        if getattr(raw, "c28_qk_candidate_value_trace_available", False) is not True:
            raise ValueError("full-visible native runtime lacks QK candidate-value route")
        for record_index in range(full._RECORDS):
            if trace.position != 0:
                trace.reset()
            _progress(
                f"capturing train QK candidate values record "
                f"{record_index + 1}/{full._RECORDS}"
            )
            evidence = _execute_record_pair(
                trace,
                context=context,
                record_index=record_index,
                progress_prefix=f"full-visible QK candidate-value train {record_index}",
            )
            if (
                "qk_candidate_values" not in evidence
                or "reset_qk_candidate_values" not in evidence
            ):
                raise ValueError("full-visible QK candidate-value capture was unavailable")
            descriptor = full.write_full_visible_qk_candidate_value_trace_shard(
                directory,
                record_index=record_index,
                record_id=evidence["record_id"],
                candidate_values=evidence["qk_candidate_values"],
                reset_candidate_values=evidence["reset_qk_candidate_values"],
                query_positions=evidence["query_positions"],
                source_record_sha256=evidence["source_record_sha256"],
                output_evidence_sha256=evidence["first_output_evidence_sha256"],
                reset_output_evidence_sha256=evidence["reset_output_evidence_sha256"],
                schedule_rows_sha256=evidence["schedule_rows_sha256"],
            )
            descriptors.append(descriptor)
            capture_rows.append(
                {
                    name: evidence[name]
                    for name in (
                        "record_index",
                        "record_id",
                        "schedule_rows_sha256",
                        "source_record_sha256",
                        "first_output_evidence_sha256",
                        "reset_output_evidence_sha256",
                        "checks",
                        "first_qk_candidate_value_trace",
                        "reset_qk_candidate_value_trace",
                    )
                }
            )
    finally:
        trace.close()
    protocol_binding = _binding(
        context["full_visible_protocol_path"],
        context["full_visible_protocol_sha256"],
    )
    manifest = full.write_full_visible_qk_candidate_value_trace_manifest(
        directory,
        protocol=protocol_binding,
        shards=descriptors,
    )
    manifest_path = Path(manifest["path"])
    full.load_stacked_full_visible_qk_candidate_value_trace(
        manifest_path,
        manifest["sha256"],
        protocol=protocol_binding,
    )
    post = _post_run_authentication(context)
    post.update(
        {
            "qk_candidate_value_manifest": (
                sha256_file(manifest_path) == manifest["sha256"]
            ),
            "qk_candidate_value_source_inventory": (
                context["full_visible_protocol"]["source_sha256"]
                == full._source_inventory()
            ),
        }
    )
    if not all(post.values()):
        raise ValueError(
            "full-visible QK candidate-value capture post-run authentication failed"
        )
    _progress(f"QK candidate-value capture manifest written to {manifest_path}")
    return {
        **manifest,
        "capture_rows": capture_rows,
        "capture_rows_sha256": sha256_json(capture_rows),
        "post_run_authentication": post,
        "confirmation_split_opened": False,
    }


def _authenticate_manifest_provenance(
    context: Mapping[str, Any],
    *,
    manifest: str | Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Cross-bind every cached shard descriptor to authenticated history."""

    manifest_path = full._checked_file(
        manifest,
        manifest_sha256,
        "full-visible trace manifest",
    )
    value = _read_json(manifest_path, "full-visible trace manifest")
    descriptors = value.get("shards")
    protocol_binding = _binding(
        context["full_visible_protocol_path"],
        context["full_visible_protocol_sha256"],
    )
    schedule_rows = (
        context["full_visible_protocol"]
        .get(
            "schedule_contract",
            {},
        )
        .get("per_record_rows_sha256")
    )
    inherited_descriptors = context["head_mass_manifest"].get("shards")
    slot_descriptors = context["cached_context"]["slot_manifest"].get("shards")
    records = context["train_records"]
    historical_rows = context["historical_output_rows"]
    if (
        value.get("schema_version") != full._SCHEMA_VERSION
        or value.get("experiment") != full._CAPTURE_EXPERIMENT
        or value.get("protocol") != protocol_binding
        or value.get("record_order") != list(range(full._RECORDS))
        or value.get("confirmation_split_opened") is not False
        or not isinstance(descriptors, list)
        or len(descriptors) != full._RECORDS
        or not isinstance(schedule_rows, list)
        or len(schedule_rows) != full._RECORDS
        or not isinstance(inherited_descriptors, list)
        or len(inherited_descriptors) != full._RECORDS
        or not isinstance(slot_descriptors, list)
        or len(slot_descriptors) != full._RECORDS
        or len(records) != full._RECORDS
        or len(historical_rows) != full._RECORDS
    ):
        raise ValueError("full-visible manifest provenance contract changed")

    for index, (
        descriptor,
        inherited,
        slot_descriptor,
        record,
        historical,
        schedule_sha256,
    ) in enumerate(
        zip(
            descriptors,
            inherited_descriptors,
            slot_descriptors,
            records,
            historical_rows,
            schedule_rows,
            strict=True,
        )
    ):
        if not all(
            isinstance(row, Mapping)
            for row in (
                descriptor,
                inherited,
                slot_descriptor,
                record,
                historical,
            )
        ):
            raise ValueError(f"full-visible manifest record {index} provenance changed")
        source_sha256 = sha256_json(record)
        output_sha256 = historical.get("observed_output_evidence_sha256")
        tensor_hashes = descriptor.get("tensor_sha256")
        inherited_hashes = inherited.get("tensor_sha256")
        slot_hashes = slot_descriptor.get("tensor_sha256")
        base_tensor_roots_exact = (
            isinstance(tensor_hashes, Mapping)
            and isinstance(inherited_hashes, Mapping)
            and all(
                tensor_hashes.get(name) == inherited_hashes.get(name)
                for name in full._BASE_TRACE_KEYS
            )
        )
        slot_tensor_roots_exact = (
            isinstance(tensor_hashes, Mapping)
            and isinstance(slot_hashes, Mapping)
            and all(
                tensor_hashes.get(name) == slot_hashes.get(name)
                for name in full._EPISODIC_SLOT_TRACE_KEYS
            )
        )
        if (
            descriptor.get("record_index") != index
            or inherited.get("record_index") != index
            or slot_descriptor.get("record_index") != index
            or historical.get("record_index") != index
            or descriptor.get("record_id") != record.get("record_id")
            or descriptor.get("record_id") != inherited.get("record_id")
            or descriptor.get("record_id") != slot_descriptor.get("record_id")
            or descriptor.get("record_id") != historical.get("record_id")
            or descriptor.get("source_record_sha256") != source_sha256
            or inherited.get("source_record_sha256") != source_sha256
            or slot_descriptor.get("source_record_sha256") != source_sha256
            or historical.get("source_record_sha256") != source_sha256
            or descriptor.get("output_evidence_sha256") != output_sha256
            or descriptor.get("reset_output_evidence_sha256") != output_sha256
            or inherited.get("output_evidence_sha256") != output_sha256
            or inherited.get("reset_output_evidence_sha256") != output_sha256
            or slot_descriptor.get("output_evidence_sha256") != output_sha256
            or slot_descriptor.get("reset_output_evidence_sha256") != output_sha256
            or historical.get("historical_output_evidence_sha256") != output_sha256
            or historical.get("reset_output_evidence_sha256") != output_sha256
            or descriptor.get("schedule_rows_sha256") != schedule_sha256
            or not full._is_sha256(schedule_sha256)
            or descriptor.get("query_positions") != list(full._READ_POSITIONS)
            or inherited.get("positions") != list(full._READ_POSITIONS)
            or slot_descriptor.get("positions") != list(full._READ_POSITIONS)
            or not base_tensor_roots_exact
            or not slot_tensor_roots_exact
        ):
            raise ValueError(f"full-visible manifest record {index} provenance changed")
    return value


def solve_cached_full_visible_capture(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    manifest: str | Path,
    manifest_sha256: str,
    out: str | Path,
    include_nested_diagnostics: bool = True,
    row_batch_size: int = full._ROW_BATCH_SIZE,
) -> dict[str, Any]:
    """Lexically guard paths, then delegate the cached numerical solve."""

    if row_batch_size != full._ROW_BATCH_SIZE:
        raise ValueError("full-visible row batch size differs from the frozen protocol")
    _guard_paths(
        (
            ("full-visible protocol", protocol),
            ("trace manifest", manifest),
            ("result output", out),
        )
    )
    context = _authenticate_frozen_capture_context(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    _authenticate_manifest_provenance(
        context,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )
    return full.solve_cached_full_visible_capture(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        out=out,
        include_nested_diagnostics=include_nested_diagnostics,
        row_batch_size=row_batch_size,
    )


def _add_predecessor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--predecessor-protocol", required=True)
    parser.add_argument("--predecessor-protocol-sha256", required=True)
    parser.add_argument("--predecessor-result", required=True)
    parser.add_argument("--predecessor-result-sha256", required=True)
    parser.add_argument("--trace-library", required=True)
    parser.add_argument("--trace-library-sha256", required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the train-only full-visible native/cached handoff",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    parity = commands.add_parser("parity")
    _add_predecessor_arguments(parity)
    parity.add_argument("--out", required=True)

    freeze = commands.add_parser("freeze")
    _add_predecessor_arguments(freeze)
    freeze.add_argument("--parity-report", required=True)
    freeze.add_argument("--parity-report-sha256", required=True)
    freeze.add_argument("--out", required=True)

    capture = commands.add_parser("capture")
    capture.add_argument("--protocol", required=True)
    capture.add_argument("--protocol-sha256", required=True)
    capture.add_argument("--shard-directory", required=True)

    qk_capture = commands.add_parser("capture-qk")
    qk_capture.add_argument("--protocol", required=True)
    qk_capture.add_argument("--protocol-sha256", required=True)
    qk_capture.add_argument("--shard-directory", required=True)

    qk_candidate_capture = commands.add_parser("capture-qk-candidates")
    qk_candidate_capture.add_argument("--protocol", required=True)
    qk_candidate_capture.add_argument("--protocol-sha256", required=True)
    qk_candidate_capture.add_argument("--shard-directory", required=True)

    qk_candidate_key_capture = commands.add_parser("capture-qk-candidate-keys")
    qk_candidate_key_capture.add_argument("--protocol", required=True)
    qk_candidate_key_capture.add_argument("--protocol-sha256", required=True)
    qk_candidate_key_capture.add_argument("--shard-directory", required=True)

    qk_candidate_value_capture = commands.add_parser(
        "capture-qk-candidate-values"
    )
    qk_candidate_value_capture.add_argument("--protocol", required=True)
    qk_candidate_value_capture.add_argument("--protocol-sha256", required=True)
    qk_candidate_value_capture.add_argument("--shard-directory", required=True)

    solve = commands.add_parser("solve")
    solve.add_argument("--protocol", required=True)
    solve.add_argument("--protocol-sha256", required=True)
    solve.add_argument("--manifest", required=True)
    solve.add_argument("--manifest-sha256", required=True)
    solve.add_argument("--out", required=True)
    solve.add_argument(
        "--no-nested-diagnostics",
        action="store_true",
    )
    solve.add_argument(
        "--row-batch-size",
        type=int,
        default=full._ROW_BATCH_SIZE,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "parity":
        generate_full_visible_trace_parity_report(
            predecessor_protocol=args.predecessor_protocol,
            predecessor_protocol_sha256=(args.predecessor_protocol_sha256),
            predecessor_result=args.predecessor_result,
            predecessor_result_sha256=args.predecessor_result_sha256,
            trace_library=args.trace_library,
            trace_library_sha256=args.trace_library_sha256,
            out=args.out,
        )
    elif args.command == "freeze":
        freeze_full_visible_protocol(
            predecessor_protocol=args.predecessor_protocol,
            predecessor_protocol_sha256=(args.predecessor_protocol_sha256),
            predecessor_result=args.predecessor_result,
            predecessor_result_sha256=args.predecessor_result_sha256,
            trace_library=args.trace_library,
            trace_library_sha256=args.trace_library_sha256,
            parity_report=args.parity_report,
            parity_report_sha256=args.parity_report_sha256,
            out=args.out,
        )
    elif args.command == "capture":
        capture_full_visible_train_traces(
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            shard_directory=args.shard_directory,
        )
    elif args.command == "capture-qk":
        capture_full_visible_train_qk_traces(
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            shard_directory=args.shard_directory,
        )
    elif args.command == "capture-qk-candidates":
        capture_full_visible_train_qk_candidate_traces(
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            shard_directory=args.shard_directory,
        )
    elif args.command == "capture-qk-candidate-keys":
        capture_full_visible_train_qk_candidate_key_traces(
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            shard_directory=args.shard_directory,
        )
    elif args.command == "capture-qk-candidate-values":
        capture_full_visible_train_qk_candidate_value_traces(
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            shard_directory=args.shard_directory,
        )
    elif args.command == "solve":
        solve_cached_full_visible_capture(
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            manifest=args.manifest,
            manifest_sha256=args.manifest_sha256,
            out=args.out,
            include_nested_diagnostics=not args.no_nested_diagnostics,
            row_batch_size=args.row_batch_size,
        )
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError("unreachable full-visible command")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
