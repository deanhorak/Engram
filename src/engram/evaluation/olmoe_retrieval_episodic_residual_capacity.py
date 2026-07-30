"""Train-only same-state residual output-subspace capacity experiment.

The fixed-K256 shared episodic-logit-bias family failed all four prospective
training arms.  This experiment does not fit another correction.  It asks a
strictly narrower question: does the exact projected-attention residual from
the same-state W128 shadow have a small global, per-layer output subspace?

The base is fixed prospectively to beta=0 K256 because its authenticated
historical mean and worst training losses both strictly beat the diagnostic
gamma=1/2 failure.  A trace-only native ABI executes the ordinary
W16/C8/K4/S2 beta=0 K256 path and a same-state W128/C8/K4/S2 shadow.  Only
positions 96..127 are persisted.  Eight leave-one-sequence-out folds fit a
training-fold mean and per-layer right-singular output subspace; held-out
coefficients are oracle coefficients.  This is capacity evidence, not a
predictor, native correction, semantic/Milestone-3 pass, or unbounded-context
claim.

No command in this module opens the development or confirmation corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import engram.evaluation.olmoe_retrieval_episodic_logit_bias as bias
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.utils import atomic_json, sha256_file, sha256_json


_SCHEMA_VERSION = 1
_PARITY_EXPERIMENT = "olmoe_q7_retrieval_episodic_residual_capacity_parity"
_PROTOCOL_EXPERIMENT = "olmoe_q7_retrieval_episodic_residual_capacity_protocol"
_RESULT_EXPERIMENT = "olmoe_q7_retrieval_episodic_residual_capacity_train_screen"
_PARITY_STATUS = "same_state_shadow_trace_parity_passed"
_PROTOCOL_STATUS = "frozen_before_residual_capacity_trace_execution"
_EXPECTED_BIAS_PROTOCOL_SHA256 = (
    "025ff45e41966faf033338ffcac0c3fc1f93b40ed7676c36f189ba57485e8be7"
)
_EXPECTED_BIAS_RESULT_SHA256 = (
    "19d08ce9eb4b673d423e9781a491e25ec03bdec09467a43e7be1881874ef2287"
)
_EXPECTED_BETA0_MEAN = 1.2244600802659988
_EXPECTED_BETA0_WORST = 1.3273429870605469
_EXPECTED_GAMMA_HALF_MEAN = 1.4614136666059494
_EXPECTED_GAMMA_HALF_WORST = 1.66925048828125
_EXPECTED_POST_AUTHENTICATION_KEYS = frozenset(
    {
        *bias._EXPECTED_POST_AUTHENTICATION_KEYS,
        "beta_zero_parity_report",
        "logit_bias_protocol",
        "logit_bias_source_inventory",
    }
)
_EXPECTED_BASE_POST_AUTHENTICATION_KEYS = frozenset(
    {
        *bias._EXPECTED_POST_AUTHENTICATION_KEYS,
        "logit_bias_protocol",
        "logit_bias_result",
        "trace_library",
        "historical_K256_result",
    }
)
_TRACE_OPEN_SYMBOL = "engram_olmoe_token_open_episodic_shadow_trace_v1"
_TRACE_COPY_SYMBOL = "engram_olmoe_token_copy_last_shadow_trace_v1"
_REQUIRED_TRACE_SYMBOLS = (
    bias._REQUIRED_V2_SYMBOL,
    _TRACE_OPEN_SYMBOL,
    _TRACE_COPY_SYMBOL,
)
_RECORDS = bias._RECORDS
_POSITIONS = bias._POSITIONS
_READ_POSITIONS = tuple(range(96, 128))
_BLOCK_ENTRY_POSITIONS = (96, 104, 112, 120)
_LAYERS = 16
_HIDDEN_SIZE = 2048
_RANKS = (2, 4, 8)
_BASE_POLICY = {
    "local_window": 16,
    "older_candidates": 8,
    "older_top_k": 4,
    "sink_tokens": 2,
}
_SHADOW_POLICY = {
    "local_window": 128,
    "older_candidates": 8,
    "older_top_k": 4,
    "sink_tokens": 2,
}
_EPISODIC_POLICY = {"slots": 32, "span_size": 8}
_TRACE_KEYS = ("input_norm", "base_projected", "target_residual")
_SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *bias._SOURCE_FILES,
            "src/engram/evaluation/olmoe_retrieval_episodic_residual_capacity.py",
        )
    )
)


def _progress(message: str) -> None:
    print(
        f"[retrieval-episodic-residual-capacity] {message}",
        file=sys.stderr,
        flush=True,
    )


def _source_inventory() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    return {name: sha256_file(repository / name) for name in _SOURCE_FILES}


def _require_trace_symbols(path: Path) -> None:
    import ctypes

    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise ValueError("residual-capacity trace library could not be loaded") from exc
    missing = [name for name in _REQUIRED_TRACE_SYMBOLS if not hasattr(library, name)]
    if missing:
        raise ValueError(
            "residual-capacity trace library lacks required symbols: "
            + ", ".join(missing)
        )


def _without_elapsed(value: Any) -> Any:
    """Remove explicitly non-deterministic elapsed fields recursively."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_elapsed(item)
            for key, item in value.items()
            if "elapsed" not in str(key).lower()
        }
    if isinstance(value, list):
        return [_without_elapsed(item) for item in value]
    return value


def _validate_bias_total_failure(
    value: Any,
    *,
    protocol_path: Path,
    protocol_sha256: str,
    frozen_bias: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on the exact four-arm shared-bias total failure."""

    if not isinstance(value, Mapping):
        raise ValueError("residual-capacity bias result is invalid")
    sweep = value.get("logit_bias_sweep")
    scope = value.get("scope")
    decision = value.get("decision")
    post = value.get("post_run_authentication")
    candidates = bias._validated_bias_candidates(frozen_bias.get("candidates"))
    candidate_ids = [row["candidate_id"] for row in candidates]
    if (
        value.get("schema_version") != bias._SCHEMA_VERSION
        or value.get("experiment") != bias._RESULT_EXPERIMENT
        or value.get("status") != "train_episodic_logit_bias_gate_failed"
        or value.get("protocol")
        != {"path": str(protocol_path), "sha256": protocol_sha256}
        or not isinstance(scope, Mapping)
        or scope.get("split") != "train"
        or scope.get("dense_teacher_forwards") != 0
        or scope.get("fixed_K") != 256
        or scope.get("development_outcomes_used") is not False
        or scope.get("confirmation_split_opened") is not False
        or not isinstance(sweep, Mapping)
        or sweep.get("candidate_order") != candidate_ids
        or sweep.get("executed_candidates") != candidate_ids
        or sweep.get("skipped_candidates") != []
        or sweep.get("passed") is not False
        or sweep.get("selected_candidate_id") != "gamma_1_2"
        or sweep.get("selection_role") != "best_failed_candidate_for_diagnostic_replay"
        or not isinstance(decision, Mapping)
        or decision.get("train_progression_gate_passed") is not False
        or decision.get("semantic_gate_passed") is not False
        or decision.get("development_authorized") is not False
        or decision.get("confirmation_authorized") is not False
        or not isinstance(post, Mapping)
        or set(post) != _EXPECTED_POST_AUTHENTICATION_KEYS
        or not all(check is True for check in post.values())
        or value.get("confirmation_split_opened") is not False
    ):
        raise ValueError("residual-capacity bias total failure changed")
    outcomes = sweep.get("candidate_outcomes")
    if not isinstance(outcomes, Mapping) or set(outcomes) != set(candidate_ids):
        raise ValueError("residual-capacity bias candidate population changed")
    fixed_arm = frozen_bias.get("fixed_arm")
    if not isinstance(fixed_arm, Mapping):
        raise ValueError("residual-capacity fixed K256 arm is missing")
    resource = fixed_arm.get("resource_contract")
    mask = fixed_arm.get("head_mask")
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        outcome = outcomes[candidate_id]
        checks = (
            outcome.get("population_resource_checks")
            if isinstance(outcome, Mapping)
            else None
        )
        replay = outcome.get("reset_replay") if isinstance(outcome, Mapping) else None
        selected = candidate_id == "gamma_1_2"
        if (
            not isinstance(outcome, Mapping)
            or outcome.get("candidate") != candidate
            or outcome.get("head_mask") != mask
            or outcome.get("resource_contract") != resource
            or not isinstance(checks, Mapping)
            or not checks
            or not all(check is True for check in checks.values())
            or outcome.get("population_resource_passed") is not True
            or outcome.get("loss_gate", {}).get("passed") is not False
            or outcome.get("pre_replay_passed") is not False
            or outcome.get("passed") is not False
            or not isinstance(replay, Mapping)
            or replay.get("executed") is not selected
            or (selected and replay.get("passed") is not True)
            or (not selected and replay.get("native_sequence_forwards") != 0)
        ):
            raise ValueError(
                f"residual-capacity bias failure changed for {candidate_id}"
            )
    selected = outcomes["gamma_1_2"]
    selected_summary = selected["loss_gate"]["summaries"]["candidate"]
    selection_key = [
        float(selected_summary["maximum_answer_cross_entropy"]),
        float(selected_summary["mean_answer_cross_entropy"]),
        0,
    ]
    if (
        selected_summary.get("mean_answer_cross_entropy") != _EXPECTED_GAMMA_HALF_MEAN
        or selected_summary.get("maximum_answer_cross_entropy")
        != _EXPECTED_GAMMA_HALF_WORST
        or sweep.get("selection_key") != selection_key
        or sweep.get("selected_candidate") != candidates[0]
    ):
        raise ValueError("residual-capacity diagnostic gamma=1/2 evidence changed")
    return {
        "status": value["status"],
        "candidate_order": candidate_ids,
        "all_candidates_executed": True,
        "all_candidates_failed": True,
        "systems_clean": True,
        "selected_candidate_id": "gamma_1_2",
        "selection_role": sweep["selection_role"],
        "selection_key": selection_key,
        "selected_reset_replay_passed": True,
        "post_authentication_keys": sorted(post),
        "development_outcomes_used": False,
        "confirmation_split_opened": False,
    }


def _select_beta0_base(
    frozen_bias: Mapping[str, Any],
    failure: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the prospective, outcome-independent beta=0 selection rule."""

    historical = frozen_bias.get("fixed_arm", {}).get("historical_K256_attribution")
    if not isinstance(historical, Mapping):
        raise ValueError("residual-capacity historical beta=0 attribution is missing")
    losses = historical.get("record_answer_cross_entropy")
    if (
        not isinstance(losses, list)
        or len(losses) != _RECORDS
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in losses
        )
    ):
        raise ValueError("residual-capacity historical beta=0 losses changed")
    mean = float(np.mean(losses, dtype=np.float64))
    worst = float(max(losses))
    diagnostic_key = failure.get("selection_key")
    if (
        mean != _EXPECTED_BETA0_MEAN
        or worst != _EXPECTED_BETA0_WORST
        or diagnostic_key != [_EXPECTED_GAMMA_HALF_WORST, _EXPECTED_GAMMA_HALF_MEAN, 0]
        or not (mean < _EXPECTED_GAMMA_HALF_MEAN and worst < _EXPECTED_GAMMA_HALF_WORST)
    ):
        raise ValueError("residual-capacity beta=0 selection basis changed")
    return {
        "selected_base": "historical_beta0_K256",
        "beta_float32": 0.0,
        "beta_float32_bits": "0x00000000",
        "K": 256,
        "historical_mean_answer_cross_entropy": mean,
        "historical_worst_answer_cross_entropy": worst,
        "diagnostic_gamma_1_2_mean_answer_cross_entropy": (_EXPECTED_GAMMA_HALF_MEAN),
        "diagnostic_gamma_1_2_worst_answer_cross_entropy": (_EXPECTED_GAMMA_HALF_WORST),
        "selection_rule": (
            "choose beta=0 K256 only because both authenticated historical "
            "mean and worst train CE strictly beat diagnostic gamma=1/2"
        ),
        "base_tuning_permitted": False,
        "fixed_before_trace_execution": True,
    }


def _authenticate_bias_inputs(
    *,
    bias_protocol: str | Path,
    bias_protocol_sha256: str,
    bias_result: str | Path,
    bias_result_sha256: str,
    trace_library: str | Path,
    trace_library_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate frozen bias roots while separately binding current sources.

    The bias protocol predates the additive shadow-trace implementation.  Its
    embedded source inventory is therefore passed back to its own exact
    reconstruction validator, while this experiment independently binds the
    current (extended) inventory in parity and protocol artifacts.
    """

    if bias_protocol_sha256.lower() != _EXPECTED_BIAS_PROTOCOL_SHA256:
        raise ValueError("residual-capacity bias protocol root changed")
    if bias_result_sha256.lower() != _EXPECTED_BIAS_RESULT_SHA256:
        raise ValueError("residual-capacity bias result root changed")
    protocol_path = bias.episodic._checked_file(
        bias_protocol,
        bias_protocol_sha256,
        "residual-capacity bias protocol",
    )
    result_path = bias.episodic._checked_file(
        bias_result,
        bias_result_sha256,
        "residual-capacity bias result",
    )
    trace_path = bias.episodic._checked_file(
        trace_library,
        trace_library_sha256,
        "residual-capacity trace library",
    )
    _require_trace_symbols(trace_path)
    raw_protocol = bias.rank.retrieval._read_json(
        protocol_path,
        "residual-capacity bias protocol",
    )
    frozen_inventory = raw_protocol.get("source_sha256")
    if not isinstance(frozen_inventory, Mapping) or not frozen_inventory:
        raise ValueError("residual-capacity frozen bias source inventory is invalid")
    original_source_inventory = bias._source_inventory
    bias._source_inventory = lambda: dict(frozen_inventory)
    try:
        context, training, frozen_bias = bias._authenticate_protocol(
            protocol_path,
            bias_protocol_sha256,
        )
    finally:
        bias._source_inventory = original_source_inventory
    result_value = bias.rank.retrieval._read_json(
        result_path,
        "residual-capacity bias result",
    )
    failure = _validate_bias_total_failure(
        result_value,
        protocol_path=protocol_path,
        protocol_sha256=bias_protocol_sha256.lower(),
        frozen_bias=frozen_bias,
    )
    base_choice = _select_beta0_base(frozen_bias, failure)
    historical_binding = frozen_bias["fixed_arm"]["historical_K256_attribution"][
        "result"
    ]
    historical_path = bias.episodic._checked_file(
        historical_binding["path"],
        historical_binding["sha256"],
        "residual-capacity historical K256 result",
    )
    historical_value = bias.rank.retrieval._read_json(
        historical_path,
        "residual-capacity historical K256 result",
    )
    historical_evidence = historical_value.get("episodic_candidate", {}).get(
        "sequence_evidence"
    )
    record_ids = frozen_bias["fixed_arm"]["historical_K256_attribution"]["record_ids"]
    if (
        not isinstance(historical_evidence, list)
        or len(historical_evidence) != _RECORDS
        or [row.get("record_id") for row in historical_evidence] != record_ids
        or [row.get("record_index") for row in historical_evidence]
        != list(range(_RECORDS))
    ):
        raise ValueError("residual-capacity historical K256 evidence changed")
    context = dict(context)
    context.update(
        {
            "bias_protocol_path": protocol_path,
            "bias_protocol_sha256": bias_protocol_sha256.lower(),
            "bias_result_path": result_path,
            "bias_result_sha256": bias_result_sha256.lower(),
            "trace_library_path": trace_path,
            "trace_library_sha256": trace_library_sha256.lower(),
            "bias_failure": failure,
            "base_choice": base_choice,
            "historical_k256_result_path": historical_path,
            "historical_k256_result_sha256": historical_binding["sha256"],
            "historical_k256_evidence": historical_evidence,
        }
    )
    return context, training, frozen_bias, failure


def _open_base_runtime(context: Mapping[str, Any]) -> Any:
    return OLMoENativeTokenRuntime(
        context["config_path"],
        context["non_mlp_path"],
        context["q7_path"],
        context["trace_library_path"],
        threads=bias._THREADS,
        **_BASE_POLICY,
        episodic_policy=_EPISODIC_POLICY,
        episodic_head_mask=bias._all_ones_mask(),
        episodic_logit_bias=0.0,
    )


def _open_trace_runtime(context: Mapping[str, Any]) -> Any:
    return OLMoENativeTokenRuntime(
        context["config_path"],
        context["non_mlp_path"],
        context["q7_path"],
        context["trace_library_path"],
        threads=bias._THREADS,
        **_BASE_POLICY,
        episodic_policy=_EPISODIC_POLICY,
        episodic_head_mask=bias._all_ones_mask(),
        episodic_logit_bias=0.0,
        shadow_attention_policy=_SHADOW_POLICY,
    )


def _validate_runtime_route(runtime: Any, *, shadow: bool) -> None:
    if (
        runtime.position != 0
        or not runtime.attention_metrics_available
        or not getattr(runtime, "episodic_metrics_available", False)
        or getattr(runtime, "episodic_policy", None) != _EPISODIC_POLICY
        or not bias.rank.fixed._runtime_mask_matches(runtime, bias._all_ones_mask())
        or bias._float32_bits(
            float(getattr(runtime, "episodic_logit_bias", float("nan")))
        )
        != "0x00000000"
    ):
        raise ValueError("residual-capacity beta=0 K256 runtime route changed")
    if shadow:
        if (
            getattr(runtime, "episodic_open_abi", None) != "shadow_trace_v1"
            or getattr(runtime, "shadow_trace_available", False) is not True
            or getattr(runtime, "shadow_attention_policy", None) != _SHADOW_POLICY
        ):
            raise ValueError("residual-capacity shadow runtime route changed")
    elif (
        getattr(runtime, "episodic_open_abi", None) != "v2"
        or getattr(runtime, "shadow_trace_available", False) is not False
    ):
        raise ValueError("residual-capacity ordinary runtime route changed")


def _trace_array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in _TRACE_KEYS:
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _trace_summary(
    arrays: Mapping[str, np.ndarray],
    positions: Sequence[int],
) -> dict[str, Any]:
    expected_shape = (len(_READ_POSITIONS), _LAYERS, _HIDDEN_SIZE)
    if list(positions) != list(_READ_POSITIONS) or set(arrays) != set(_TRACE_KEYS):
        raise ValueError("residual-capacity trace support changed")
    tensor_sha256: dict[str, str] = {}
    for name in _TRACE_KEYS:
        array = arrays[name]
        if (
            not isinstance(array, np.ndarray)
            or array.shape != expected_shape
            or array.dtype != np.float32
            or not array.flags.c_contiguous
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"residual-capacity {name} trace is invalid")
        tensor_sha256[name] = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    residual = arrays["target_residual"]
    nonzero_rows = [
        bool(np.any(residual[index] != np.float32(0.0)))
        for index in range(residual.shape[0])
    ]
    if not all(nonzero_rows):
        raise ValueError("residual-capacity trace has a zero residual read row")
    return {
        "positions": list(positions),
        "shape": list(expected_shape),
        "dtype": "float32",
        "layout": "position_layer_hidden",
        "tensor_sha256": tensor_sha256,
        "trace_sha256": _trace_array_digest(arrays),
        "nonzero_residual_read_rows": int(sum(nonzero_rows)),
        "all_residual_read_rows_nonzero": True,
    }


class _TraceCaptureRuntime:
    """Capture trace tensors only on explicit episodic read rows."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._positions: list[int] = []
        self._rows: dict[str, list[np.ndarray]] = {name: [] for name in _TRACE_KEYS}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    @property
    def position(self) -> int:
        return int(self._runtime.position)

    def forward_episodic(
        self,
        token_ids: Sequence[int],
        write_slots: Sequence[int],
        read_spans: Sequence[int],
    ) -> Any:
        result = self._runtime.forward_episodic(
            token_ids,
            write_slots,
            read_spans,
        )
        if len(read_spans) != 1:
            raise ValueError("residual-capacity trace requires one-token calls")
        if int(read_spans[0]) >= 0:
            position = self.position - 1
            traces = self._runtime.last_shadow_trace()
            if not isinstance(traces, tuple) or len(traces) != len(_TRACE_KEYS):
                raise ValueError("residual-capacity native trace tuple changed")
            self._positions.append(position)
            for name, array in zip(_TRACE_KEYS, traces, strict=True):
                copied = np.ascontiguousarray(array)
                if (
                    copied.shape != (_LAYERS, _HIDDEN_SIZE)
                    or copied.dtype != np.float32
                    or not np.isfinite(copied).all()
                ):
                    raise ValueError(f"residual-capacity native {name} row is invalid")
                self._rows[name].append(copied)
        return result

    def captured(self) -> tuple[dict[str, np.ndarray], list[int]]:
        if len(self._positions) != len(_READ_POSITIONS):
            raise ValueError("residual-capacity trace capture is incomplete")
        arrays = {
            name: np.ascontiguousarray(np.stack(rows), dtype=np.float32)
            for name, rows in self._rows.items()
        }
        _trace_summary(arrays, self._positions)
        return arrays, list(self._positions)

    def reset(self) -> None:
        self._runtime.reset()
        self._positions.clear()
        for rows in self._rows.values():
            rows.clear()

    def close(self) -> None:
        self._runtime.close()


def _execute_record(
    runtime: Any,
    *,
    record: Mapping[str, Any],
    context: Mapping[str, Any],
    schedule: Mapping[str, Any],
    resource: Mapping[str, Any],
    progress_label: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None, list[int] | None]:
    trace_runtime = runtime if isinstance(runtime, _TraceCaptureRuntime) else None
    _logits, _hidden, evidence = bias.episodic._execute_episodic_record(
        runtime,
        record=record,
        context=context,
        schedule=schedule,
        resource=resource,
        progress_label=progress_label,
    )
    if trace_runtime is None:
        return evidence, None, None
    arrays, positions = trace_runtime.captured()
    return evidence, arrays, positions


def _evidence_exact(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    return _without_elapsed(left) == _without_elapsed(right)


def _run_trace_parity(
    *,
    context: Mapping[str, Any],
    record: Mapping[str, Any],
    schedule: Mapping[str, Any],
    resource: Mapping[str, Any],
    base_factory: Callable[[Mapping[str, Any]], Any] = _open_base_runtime,
    trace_factory: Callable[[Mapping[str, Any]], Any] = _open_trace_runtime,
) -> dict[str, Any]:
    base = base_factory(context)
    try:
        _validate_runtime_route(base, shadow=False)
        base_first, _arrays, _positions = _execute_record(
            base,
            record=record,
            context=context,
            schedule=schedule,
            resource=resource,
        )
        base.reset()
        if base.position != 0:
            raise ValueError("residual-capacity ordinary reset failed")
        base_replay, _arrays, _positions = _execute_record(
            base,
            record=record,
            context=context,
            schedule=schedule,
            resource=resource,
        )
    finally:
        base.close()
    raw_trace = trace_factory(context)
    trace = _TraceCaptureRuntime(raw_trace)
    try:
        _validate_runtime_route(raw_trace, shadow=True)
        trace_first, first_arrays, first_positions = _execute_record(
            trace,
            record=record,
            context=context,
            schedule=schedule,
            resource=resource,
        )
        assert first_arrays is not None and first_positions is not None
        first_summary = _trace_summary(first_arrays, first_positions)
        trace.reset()
        if trace.position != 0:
            raise ValueError("residual-capacity shadow reset failed")
        trace_replay, replay_arrays, replay_positions = _execute_record(
            trace,
            record=record,
            context=context,
            schedule=schedule,
            resource=resource,
        )
        assert replay_arrays is not None and replay_positions is not None
        replay_summary = _trace_summary(replay_arrays, replay_positions)
    finally:
        trace.close()
    checks = {
        "ordinary_reset_outputs_counters_and_loss_exact": _evidence_exact(
            base_first, base_replay
        ),
        "shadow_reset_outputs_counters_and_loss_exact": _evidence_exact(
            trace_first, trace_replay
        ),
        "ordinary_shadow_first_outputs_counters_and_loss_exact": _evidence_exact(
            base_first, trace_first
        ),
        "ordinary_shadow_replay_outputs_counters_and_loss_exact": _evidence_exact(
            base_replay, trace_replay
        ),
        "trace_reset_digest_exact": first_summary == replay_summary,
        "trace_shape_dtype_finite_and_nonzero": (
            first_summary["all_residual_read_rows_nonzero"] is True
            and first_summary["nonzero_residual_read_rows"] == len(_READ_POSITIONS)
        ),
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise ValueError("residual-capacity same-state trace parity failed")
    return {
        "ordinary_v2_beta0_first": base_first,
        "ordinary_v2_beta0_reset_replay": base_replay,
        "shadow_trace_beta0_first": trace_first,
        "shadow_trace_beta0_reset_replay": trace_replay,
        "first_trace": first_summary,
        "reset_trace": replay_summary,
        "checks": checks,
        "native_sequence_forwards": 4,
        "native_token_steps": 4 * _POSITIONS,
        "passed": True,
    }


def _base_post_authentication(
    context: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
) -> dict[str, bool]:
    checks = bias._post_input_authentication(context, checkpoint=checkpoint)
    checks.update(
        {
            "logit_bias_protocol": (
                sha256_file(context["bias_protocol_path"])
                == context["bias_protocol_sha256"]
            ),
            "logit_bias_result": (
                sha256_file(context["bias_result_path"])
                == context["bias_result_sha256"]
            ),
            "trace_library": (
                sha256_file(context["trace_library_path"])
                == context["trace_library_sha256"]
            ),
            "historical_K256_result": (
                sha256_file(context["historical_k256_result_path"])
                == context["historical_k256_result_sha256"]
            ),
        }
    )
    return checks


def _validated_base_post_authentication(
    value: Any,
    *,
    context: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, bool]:
    fresh = _base_post_authentication(context, checkpoint=checkpoint)
    if (
        not isinstance(value, Mapping)
        or set(value) != _EXPECTED_BASE_POST_AUTHENTICATION_KEYS
        or set(fresh) != _EXPECTED_BASE_POST_AUTHENTICATION_KEYS
        or dict(value) != fresh
        or not all(check is True for check in fresh.values())
    ):
        raise ValueError("residual-capacity parity post-authentication changed")
    return dict(fresh)


def _trace_report_summaries_valid(first: Any, reset: Any) -> bool:
    return (
        isinstance(first, Mapping)
        and first == reset
        and first.get("positions") == list(_READ_POSITIONS)
        and first.get("shape") == [len(_READ_POSITIONS), _LAYERS, _HIDDEN_SIZE]
        and first.get("dtype") == "float32"
        and first.get("layout") == "position_layer_hidden"
        and first.get("all_residual_read_rows_nonzero") is True
        and first.get("nonzero_residual_read_rows") == len(_READ_POSITIONS)
        and bias.rank.retrieval._is_sha256(first.get("trace_sha256"))
        and set(first.get("tensor_sha256", {})) == set(_TRACE_KEYS)
        and all(
            bias.rank.retrieval._is_sha256(digest)
            for digest in first.get("tensor_sha256", {}).values()
        )
    )


def generate_trace_parity_report(
    *,
    bias_protocol: str | Path,
    bias_protocol_sha256: str,
    bias_result: str | Path,
    bias_result_sha256: str,
    trace_library: str | Path,
    trace_library_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    output = bias.rank.retrieval._new_output(
        out,
        "residual-capacity parity report",
    )
    context, _training, frozen_bias, failure = _authenticate_bias_inputs(
        bias_protocol=bias_protocol,
        bias_protocol_sha256=bias_protocol_sha256,
        bias_result=bias_result,
        bias_result_sha256=bias_result_sha256,
        trace_library=trace_library,
        trace_library_sha256=trace_library_sha256,
    )
    record = context["train_records"][0]
    schedule = bias.rank.fixed._derive_schedule(
        record["input_ids"],
        frozen_bias["tokenizer_fact_anchor_ids"],
    )
    _progress("running ordinary V2 and same-state trace first/reset parity")
    parity = _run_trace_parity(
        context=context,
        record=record,
        schedule=schedule,
        resource=frozen_bias["fixed_arm"]["resource_contract"],
    )
    post = _base_post_authentication(
        context,
        checkpoint=frozen_bias["training_checkpoint"],
    )
    if not post or not all(post.values()):
        raise ValueError("residual-capacity parity post-authentication failed")
    report = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _PARITY_EXPERIMENT,
        "status": _PARITY_STATUS,
        "bias_protocol": {
            "path": str(context["bias_protocol_path"]),
            "sha256": context["bias_protocol_sha256"],
        },
        "bias_result": {
            "path": str(context["bias_result_path"]),
            "sha256": context["bias_result_sha256"],
            "authenticated_total_failure": failure,
        },
        "trace_library": {
            "path": str(context["trace_library_path"]),
            "sha256": context["trace_library_sha256"],
            "required_symbols": list(_REQUIRED_TRACE_SYMBOLS),
        },
        "base_choice": context["base_choice"],
        "base_policy": dict(_BASE_POLICY),
        "shadow_policy": dict(_SHADOW_POLICY),
        "episodic_policy": dict(_EPISODIC_POLICY),
        "fixed_mask": bias._fixed_mask_descriptor(),
        "scope": {
            "split": "train",
            "record_index": 0,
            "positions": _POSITIONS,
            "trace_positions": list(_READ_POSITIONS),
            "capacity_only": True,
            "development_outcomes_used": False,
            "confirmation_split_opened": False,
        },
        "schedule_rows_sha256": schedule["rows_sha256"],
        "parity": parity,
        "source_sha256": _source_inventory(),
        "post_run_authentication": post,
        "confirmation_split_opened": False,
    }
    atomic_json(output, report)
    _progress(f"trace parity report written to {output}")
    return report


def _validate_parity_report(
    *,
    path: str | Path,
    expected_sha256: str,
    context: Mapping[str, Any],
    frozen_bias: Mapping[str, Any],
) -> dict[str, Any]:
    source = bias.episodic._checked_file(
        path,
        expected_sha256,
        "residual-capacity parity report",
    )
    value = bias.rank.retrieval._read_json(
        source,
        "residual-capacity parity report",
    )
    parity = value.get("parity")
    post = value.get("post_run_authentication")
    schedule = bias.rank.fixed._derive_schedule(
        context["train_records"][0]["input_ids"],
        frozen_bias["tokenizer_fact_anchor_ids"],
    )
    if not isinstance(parity, Mapping):
        raise ValueError("residual-capacity parity report is invalid")
    evidence_names = (
        "ordinary_v2_beta0_first",
        "ordinary_v2_beta0_reset_replay",
        "shadow_trace_beta0_first",
        "shadow_trace_beta0_reset_replay",
    )
    evidence = [parity.get(name) for name in evidence_names]
    evidence_valid = all(
        isinstance(row, Mapping)
        and row.get("record_index") == 0
        and row.get("record_id") == context["train_records"][0]["record_id"]
        and row.get("final_position") == _POSITIONS
        and row.get("schedule_rows_sha256") == schedule["rows_sha256"]
        and isinstance(row.get("top1_tokens"), list)
        and len(row["top1_tokens"]) == _POSITIONS
        and isinstance(row.get("answer_cross_entropy"), (int, float))
        and not isinstance(row.get("answer_cross_entropy"), bool)
        and math.isfinite(float(row["answer_cross_entropy"]))
        and isinstance(row.get("counter_stream"), list)
        and len(row["counter_stream"]) == _POSITIONS
        and all(
            isinstance(counter, Mapping)
            and counter.get("position") == position
            and counter.get("passed") is True
            and isinstance(counter.get("checks"), Mapping)
            and counter["checks"]
            and all(check is True for check in counter["checks"].values())
            for position, counter in enumerate(row["counter_stream"])
        )
        and row.get("counter_stream_passed") is True
        and all(
            bias.episodic._counter_checks(
                row.get("final_metrics", {}),
                context=context,
                schedule=schedule,
                positions=_POSITIONS,
                resource=frozen_bias["fixed_arm"]["resource_contract"],
            ).values()
        )
        and bias.rank.retrieval._is_sha256(row.get("hidden_sha256"))
        and bias.rank.retrieval._is_sha256(row.get("logits_sha256"))
        and bias.rank.retrieval._is_sha256(row.get("counter_stream_sha256"))
        and bias.rank.retrieval._is_sha256(row.get("episodic_call_stream_sha256"))
        for row in evidence
    )
    first_trace = parity.get("first_trace")
    reset_trace = parity.get("reset_trace")
    trace_valid = _trace_report_summaries_valid(
        first_trace,
        reset_trace,
    )
    cross_exact = (
        evidence_valid
        and _evidence_exact(evidence[0], evidence[1])
        and _evidence_exact(evidence[0], evidence[2])
        and _evidence_exact(evidence[2], evidence[3])
    )
    expected_checks = {
        "ordinary_reset_outputs_counters_and_loss_exact": True,
        "shadow_reset_outputs_counters_and_loss_exact": True,
        "ordinary_shadow_first_outputs_counters_and_loss_exact": True,
        "ordinary_shadow_replay_outputs_counters_and_loss_exact": True,
        "trace_reset_digest_exact": True,
        "trace_shape_dtype_finite_and_nonzero": True,
        "passed": True,
    }
    if (
        value.get("schema_version") != _SCHEMA_VERSION
        or value.get("experiment") != _PARITY_EXPERIMENT
        or value.get("status") != _PARITY_STATUS
        or value.get("bias_protocol")
        != {
            "path": str(context["bias_protocol_path"]),
            "sha256": context["bias_protocol_sha256"],
        }
        or value.get("bias_result", {}).get("path") != str(context["bias_result_path"])
        or value.get("bias_result", {}).get("sha256") != context["bias_result_sha256"]
        or value.get("bias_result", {}).get("authenticated_total_failure")
        != context["bias_failure"]
        or value.get("trace_library")
        != {
            "path": str(context["trace_library_path"]),
            "sha256": context["trace_library_sha256"],
            "required_symbols": list(_REQUIRED_TRACE_SYMBOLS),
        }
        or value.get("base_choice") != context["base_choice"]
        or value.get("base_policy") != _BASE_POLICY
        or value.get("shadow_policy") != _SHADOW_POLICY
        or value.get("episodic_policy") != _EPISODIC_POLICY
        or value.get("fixed_mask") != bias._fixed_mask_descriptor()
        or value.get("scope")
        != {
            "split": "train",
            "record_index": 0,
            "positions": _POSITIONS,
            "trace_positions": list(_READ_POSITIONS),
            "capacity_only": True,
            "development_outcomes_used": False,
            "confirmation_split_opened": False,
        }
        or value.get("schedule_rows_sha256") != schedule["rows_sha256"]
        or parity.get("checks") != expected_checks
        or parity.get("passed") is not True
        or parity.get("native_sequence_forwards") != 4
        or parity.get("native_token_steps") != 4 * _POSITIONS
        or not cross_exact
        or not trace_valid
        or value.get("source_sha256") != _source_inventory()
        or value.get("confirmation_split_opened") is not False
    ):
        raise ValueError("residual-capacity parity report is invalid")
    _validated_base_post_authentication(
        post,
        context=context,
        checkpoint=frozen_bias["training_checkpoint"],
    )
    return {
        "path": str(source),
        "sha256": expected_sha256.lower(),
        "status": value["status"],
        "outputs_counters_losses_and_reset_exact": True,
        "trace_reset_digest_exact": True,
        "trace_sha256": first_trace["trace_sha256"],
        "native_sequence_forwards": 4,
        "native_token_steps": 4 * _POSITIONS,
    }


def _build_protocol(
    *,
    context: Mapping[str, Any],
    training: Mapping[str, Any],
    frozen_bias: Mapping[str, Any],
    failure: Mapping[str, Any],
    parity: Mapping[str, Any],
) -> dict[str, Any]:
    base_protocol = context["protocol"]
    schedules = [
        bias.rank.fixed._derive_schedule(
            record["input_ids"],
            frozen_bias["tokenizer_fact_anchor_ids"],
        )
        for record in context["train_records"]
    ]
    if _BASE_POLICY != bias.rank.retrieval._BASE_POLICY:
        raise ValueError("residual-capacity base attention policy changed")
    model = context["model"]
    if int(model["layers"]) != _LAYERS or int(model["hidden_size"]) != _HIDDEN_SIZE:
        raise ValueError("residual-capacity trace dimensions changed")
    return {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _PROTOCOL_EXPERIMENT,
        "status": _PROTOCOL_STATUS,
        "bias_protocol": {
            "path": str(context["bias_protocol_path"]),
            "sha256": context["bias_protocol_sha256"],
        },
        "bias_result": {
            "path": str(context["bias_result_path"]),
            "sha256": context["bias_result_sha256"],
            "authenticated_total_failure": dict(failure),
        },
        "trace_library": {
            "path": str(context["trace_library_path"]),
            "sha256": context["trace_library_sha256"],
            "required_symbols": list(_REQUIRED_TRACE_SYMBOLS),
        },
        "trace_parity": dict(parity),
        "training_checkpoint": dict(frozen_bias["training_checkpoint"]),
        "training_checkpoint_payload_sha256": sha256_json(training),
        "package": dict(base_protocol["package"]),
        "corpus": dict(base_protocol["corpus"]),
        "source_model": dict(base_protocol["source_model"]),
        "libraries": dict(base_protocol["libraries"]),
        "base_choice": dict(context["base_choice"]),
        "fixed_K256_arm": {
            "head_mask": bias._fixed_mask_descriptor(),
            "resource_contract": frozen_bias["fixed_arm"]["resource_contract"],
            "historical_K256_attribution": frozen_bias["fixed_arm"][
                "historical_K256_attribution"
            ],
        },
        "policies": {
            "base": dict(_BASE_POLICY),
            "shadow": dict(_SHADOW_POLICY),
            "episodic": dict(_EPISODIC_POLICY),
            "mask": "all_ones_K256",
            "episodic_logit_bias_float32": 0.0,
            "episodic_logit_bias_bits": "0x00000000",
        },
        "trace_schema": {
            "native_step_shape": [_LAYERS, _HIDDEN_SIZE],
            "stored_shape": [len(_READ_POSITIONS), _LAYERS, _HIDDEN_SIZE],
            "dtype": "float32",
            "layout": "position_layer_hidden",
            "keys": list(_TRACE_KEYS),
            "definitions": {
                "input_norm": "pre-attention normalized hidden input",
                "base_projected": "beta0 K256 base attention after W_o",
                "target_residual": "same-state W128 projected minus base_projected",
            },
            "support_rule": "capture iff schedule read_span >= 0",
            "positions": list(_READ_POSITIONS),
            "block_entry_positions": list(_BLOCK_ENTRY_POSITIONS),
            "W128_exact_full_context_only_for_position_horizon": _POSITIONS,
            "unbounded_context_claim": False,
            "storage": "safetensors",
            "pickle_permitted": False,
        },
        "schedule_contract": {
            "records": _RECORDS,
            "positions_per_record": _POSITIONS,
            "tokenizer_fact_anchor_ids": {
                label: list(values)
                for label, values in frozen_bias["tokenizer_fact_anchor_ids"].items()
            },
            "per_record_rows_sha256": [
                schedule["rows_sha256"] for schedule in schedules
            ],
            "read_positions": list(_READ_POSITIONS),
            "read_rows_per_record": len(_READ_POSITIONS),
        },
        "capacity_method": {
            "ranks": list(_RANKS),
            "rank_zero_baseline": "training-fold per-layer residual mean",
            "folds": "eight leave-one-sequence-out folds",
            "fit": (
                "per-layer training-fold mean plus right singular output "
                "subspace of centered training-fold target residuals"
            ),
            "heldout_coefficients": "oracle least-squares projection",
            "position_leakage_across_folds_permitted": False,
            "energy_aggregation": "squared Frobenius energy before ratios",
            "error_ratio": "sqrt(sum(error^2)/sum(target_residual^2))",
            "recovery": "1-sum(error^2)/sum(target_residual^2)",
        },
        "progression_gate": {
            "finite": True,
            "minimum_global_recovery": 0.50,
            "minimum_every_sequence_recovery": 0.25,
            "minimum_every_block_entry_position_recovery": 0.25,
            "minimum_positive_recovery_layers": 12,
            "smallest_passing_rank_selected": True,
            "failure_selection_key": [
                "worst_sequence_error_ratio",
                "global_error_ratio",
                "rank",
            ],
            "selected_metrics_deterministic_replay_required": True,
        },
        "scope": {
            "split": "train",
            "capacity_evidence_only": True,
            "learned_predictor": False,
            "native_correction": False,
            "semantic_or_M3_pass": False,
            "development_outcomes_used": False,
            "confirmation_file_access_permitted": False,
            "trace_instrumentation_excluded_from_inference_claims": True,
        },
        "authorized_next_step_on_pass": (
            "train-only predictor fit with rank0, input-only, base-only, and "
            "input-plus-base controls"
        ),
        "authenticated_confirmation_descriptor": dict(
            frozen_bias["authenticated_confirmation_descriptor"]
        ),
        "source_sha256": _source_inventory(),
        "confirmation_split_opened": False,
    }


def freeze_residual_capacity_protocol(
    *,
    bias_protocol: str | Path,
    bias_protocol_sha256: str,
    bias_result: str | Path,
    bias_result_sha256: str,
    trace_library: str | Path,
    trace_library_sha256: str,
    parity_report: str | Path,
    parity_report_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    output = bias.rank.retrieval._new_output(
        out,
        "residual-capacity protocol",
    )
    context, training, frozen_bias, failure = _authenticate_bias_inputs(
        bias_protocol=bias_protocol,
        bias_protocol_sha256=bias_protocol_sha256,
        bias_result=bias_result,
        bias_result_sha256=bias_result_sha256,
        trace_library=trace_library,
        trace_library_sha256=trace_library_sha256,
    )
    parity = _validate_parity_report(
        path=parity_report,
        expected_sha256=parity_report_sha256,
        context=context,
        frozen_bias=frozen_bias,
    )
    protocol = _build_protocol(
        context=context,
        training=training,
        frozen_bias=frozen_bias,
        failure=failure,
        parity=parity,
    )
    atomic_json(output, protocol)
    return {"path": str(output), "sha256": sha256_file(output), "protocol": protocol}


def _authenticate_protocol(
    protocol: str | Path,
    protocol_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = bias.episodic._checked_file(
        protocol,
        protocol_sha256,
        "residual-capacity protocol",
    )
    value = bias.rank.retrieval._read_json(
        source,
        "residual-capacity protocol",
    )
    bindings = (
        value.get("bias_protocol"),
        value.get("bias_result"),
        value.get("trace_library"),
        value.get("trace_parity"),
    )
    if not all(isinstance(item, Mapping) for item in bindings):
        raise ValueError("residual-capacity protocol bindings are invalid")
    bias_binding, result_binding, library_binding, parity_binding = bindings
    context, training, frozen_bias, failure = _authenticate_bias_inputs(
        bias_protocol=bias_binding.get("path"),
        bias_protocol_sha256=bias_binding.get("sha256"),
        bias_result=result_binding.get("path"),
        bias_result_sha256=result_binding.get("sha256"),
        trace_library=library_binding.get("path"),
        trace_library_sha256=library_binding.get("sha256"),
    )
    parity = _validate_parity_report(
        path=parity_binding.get("path"),
        expected_sha256=parity_binding.get("sha256"),
        context=context,
        frozen_bias=frozen_bias,
    )
    expected = _build_protocol(
        context=context,
        training=training,
        frozen_bias=frozen_bias,
        failure=failure,
        parity=parity,
    )
    if value != expected:
        raise ValueError("residual-capacity frozen protocol changed")
    context = dict(context)
    context.update(
        {
            "capacity_protocol_path": source,
            "capacity_protocol_sha256": protocol_sha256.lower(),
            "capacity_protocol": expected,
            "capacity_parity_path": Path(parity["path"]).resolve(),
            "capacity_parity_sha256": parity["sha256"],
        }
    )
    return context, training, expected


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _write_trace_shard(
    directory: Path,
    *,
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    positions: Sequence[int],
    source_record_sha256: str,
    output_sha256: str,
    reset_output_sha256: str,
    reset_trace_sha256: str,
) -> dict[str, Any]:
    summary = _trace_summary(arrays, positions)
    filename = f"train-{int(record['record_index']):02d}.safetensors"
    path = directory / filename
    if path.exists() or path.is_symlink():
        raise ValueError("residual-capacity trace shard already exists")
    try:
        from safetensors.numpy import save_file
    except ImportError as exc:  # pragma: no cover - required project dependency
        raise RuntimeError("residual-capacity shards require safetensors") from exc
    payload = {
        name: np.ascontiguousarray(arrays[name], dtype=np.float32)
        for name in _TRACE_KEYS
    }
    payload["positions"] = np.asarray(positions, dtype=np.int64)
    temporary = directory / f".{filename}.tmp-{os.getpid()}"
    save_file(payload, str(temporary))
    temporary.replace(path)
    descriptor = {
        "record_index": int(record["record_index"]),
        "record_id": record["record_id"],
        "file": filename,
        "file_sha256": sha256_file(path),
        "format": "safetensors",
        "keys": [*_TRACE_KEYS, "positions"],
        "shape": [len(_READ_POSITIONS), _LAYERS, _HIDDEN_SIZE],
        "dtype": "float32",
        "positions": list(_READ_POSITIONS),
        "tensor_sha256": summary["tensor_sha256"],
        "trace_sha256": summary["trace_sha256"],
        "reset_trace_sha256": reset_trace_sha256,
        "source_record_sha256": source_record_sha256,
        "output_evidence_sha256": output_sha256,
        "reset_output_evidence_sha256": reset_output_sha256,
    }
    _validate_trace_shard(path, descriptor)
    return descriptor


def _validate_trace_shard(
    path: str | Path,
    descriptor: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError("residual-capacity trace shard descriptor is invalid")
    source = requested.resolve()
    if (
        not source.is_file()
        or source.name != descriptor.get("file")
        or sha256_file(source) != descriptor.get("file_sha256")
        or descriptor.get("format") != "safetensors"
        or descriptor.get("keys") != [*_TRACE_KEYS, "positions"]
        or descriptor.get("shape") != [len(_READ_POSITIONS), _LAYERS, _HIDDEN_SIZE]
        or descriptor.get("dtype") != "float32"
        or descriptor.get("positions") != list(_READ_POSITIONS)
        or set(descriptor.get("tensor_sha256", {})) != set(_TRACE_KEYS)
        or not all(
            bias.rank.retrieval._is_sha256(digest)
            for digest in descriptor.get("tensor_sha256", {}).values()
        )
        or not all(
            bias.rank.retrieval._is_sha256(descriptor.get(name))
            for name in (
                "file_sha256",
                "trace_sha256",
                "reset_trace_sha256",
                "source_record_sha256",
                "output_evidence_sha256",
                "reset_output_evidence_sha256",
            )
        )
    ):
        raise ValueError("residual-capacity trace shard descriptor is invalid")
    try:
        from safetensors import safe_open
        from safetensors.numpy import load_file
    except ImportError as exc:  # pragma: no cover - required project dependency
        raise RuntimeError("residual-capacity shards require safetensors") from exc
    with safe_open(source, framework="numpy") as handle:
        if sorted(handle.keys()) != sorted([*_TRACE_KEYS, "positions"]):
            raise ValueError("residual-capacity trace shard keys changed")
    loaded = load_file(source)
    positions = loaded["positions"]
    arrays = {name: np.ascontiguousarray(loaded[name]) for name in _TRACE_KEYS}
    if (
        positions.dtype != np.int64
        or positions.shape != (len(_READ_POSITIONS),)
        or positions.tolist() != list(_READ_POSITIONS)
    ):
        raise ValueError("residual-capacity trace shard positions changed")
    summary = _trace_summary(arrays, positions.tolist())
    if summary["trace_sha256"] != descriptor.get("trace_sha256") or summary[
        "tensor_sha256"
    ] != descriptor.get("tensor_sha256"):
        raise ValueError("residual-capacity trace shard tensor hash changed")
    return arrays


def _prepare_shard_directory(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    if requested.is_symlink():
        raise ValueError("residual-capacity shard directory is invalid")
    path = requested.resolve()
    if path.exists() and not path.is_dir():
        raise ValueError("residual-capacity shard directory is invalid")
    if path.exists() and any(path.iterdir()):
        raise ValueError("residual-capacity shard directory is not empty")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _energy_metric(target_squared: float, error_squared: float) -> dict[str, float]:
    target = float(target_squared)
    error = float(error_squared)
    if (
        not math.isfinite(target)
        or not math.isfinite(error)
        or target < 0.0
        or error < 0.0
    ):
        raise ValueError("residual-capacity energy is invalid")
    if target == 0.0:
        if error > 1e-20:
            raise ValueError("residual-capacity zero target has nonzero error")
        return {
            "target_squared_frobenius": 0.0,
            "error_squared_frobenius": 0.0,
            "error_ratio": 0.0,
            "recovery": 0.0,
        }
    ratio_squared = error / target
    ratio = math.sqrt(max(0.0, ratio_squared))
    return {
        "target_squared_frobenius": target,
        "error_squared_frobenius": error,
        "error_ratio": ratio,
        "recovery": 1.0 - ratio_squared,
    }


def _right_singular_subspace(
    centered: np.ndarray,
    rank_value: int,
) -> np.ndarray:
    """Return an exact small right-singular subspace via the sample Gram matrix."""

    matrix = np.asarray(centered, dtype=np.float64)
    if matrix.ndim != 2 or rank_value < 0:
        raise ValueError("residual-capacity subspace input is invalid")
    if rank_value == 0 or matrix.size == 0:
        return np.empty((0, matrix.shape[1]), dtype=np.float64)
    feature_space = matrix.shape[1] <= matrix.shape[0]
    gram = matrix.T @ matrix if feature_space else matrix @ matrix.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    largest = float(max(eigenvalues[0], 0.0)) if eigenvalues.size else 0.0
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * largest
    positive = np.flatnonzero(eigenvalues > tolerance)[:rank_value]
    if not len(positive):
        return np.empty((0, matrix.shape[1]), dtype=np.float64)
    if feature_space:
        vectors = eigenvectors[:, positive].T
    else:
        vectors = (eigenvectors[:, positive].T @ matrix) / np.sqrt(
            eigenvalues[positive]
        )[:, None]
    # Sequential QR keeps every leading-k span nested while removing numerical
    # drift from the Gram reconstruction.
    q, _r = np.linalg.qr(vectors.T, mode="reduced")
    return np.ascontiguousarray(q.T)


def _fit_fold_layer_subspace(
    targets: np.ndarray,
    *,
    heldout: int,
    layer: int,
    rank_value: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    residual = np.asarray(targets)
    if (
        residual.ndim != 4
        or residual.shape[0] != _RECORDS
        or not 0 <= heldout < _RECORDS
        or not 0 <= layer < residual.shape[2]
        or rank_value < 0
    ):
        raise ValueError("residual-capacity fold request is invalid")
    training_indices = [index for index in range(_RECORDS) if index != heldout]
    hidden_size = int(residual.shape[3])
    train = residual[training_indices, :, layer, :].reshape(-1, hidden_size)
    train64 = np.asarray(train, dtype=np.float64)
    mean = np.mean(train64, axis=0, dtype=np.float64)
    basis = _right_singular_subspace(train64 - mean, rank_value)
    return mean, basis, training_indices


def _capacity_metrics(
    targets: np.ndarray,
    ranks: Sequence[int],
) -> dict[int, dict[str, Any]]:
    residual = np.asarray(targets)
    normalized_ranks = tuple(int(value) for value in ranks)
    if (
        residual.ndim != 4
        or residual.shape[:3] != (_RECORDS, len(_READ_POSITIONS), _LAYERS)
        or residual.shape[3] < max(normalized_ranks, default=0)
        or residual.dtype != np.float32
        or not residual.flags.c_contiguous
        or not np.isfinite(residual).all()
        or not normalized_ranks
        or any(value < 0 for value in normalized_ranks)
        or len(set(normalized_ranks)) != len(normalized_ranks)
    ):
        raise ValueError("residual-capacity target tensor or ranks changed")
    accumulators: dict[int, dict[str, np.ndarray | float]] = {}
    for rank_value in normalized_ranks:
        accumulators[rank_value] = {
            "global_target": 0.0,
            "global_error": 0.0,
            "sequence_target": np.zeros(_RECORDS, dtype=np.float64),
            "sequence_error": np.zeros(_RECORDS, dtype=np.float64),
            "layer_target": np.zeros(_LAYERS, dtype=np.float64),
            "layer_error": np.zeros(_LAYERS, dtype=np.float64),
            "offset_target": np.zeros(len(_BLOCK_ENTRY_POSITIONS), dtype=np.float64),
            "offset_error": np.zeros(len(_BLOCK_ENTRY_POSITIONS), dtype=np.float64),
        }
    maximum_rank = max(normalized_ranks)
    for heldout in range(_RECORDS):
        for layer in range(_LAYERS):
            mean, basis, _training_indices = _fit_fold_layer_subspace(
                residual,
                heldout=heldout,
                layer=layer,
                rank_value=maximum_rank,
            )
            target = np.asarray(residual[heldout, :, layer, :], dtype=np.float64)
            centered_heldout = target - mean
            target_per_position = np.einsum(
                "ij,ij->i", target, target, dtype=np.float64
            )
            for rank_value in normalized_ranks:
                active = basis[:rank_value]
                if active.size:
                    reconstruction = mean + (centered_heldout @ active.T) @ active
                else:
                    reconstruction = np.broadcast_to(mean, target.shape)
                error = target - reconstruction
                error_per_position = np.einsum(
                    "ij,ij->i", error, error, dtype=np.float64
                )
                target_sum = float(np.sum(target_per_position, dtype=np.float64))
                error_sum = float(np.sum(error_per_position, dtype=np.float64))
                accumulator = accumulators[rank_value]
                accumulator["global_target"] = (
                    float(accumulator["global_target"]) + target_sum
                )
                accumulator["global_error"] = (
                    float(accumulator["global_error"]) + error_sum
                )
                sequence_target = accumulator["sequence_target"]
                sequence_error = accumulator["sequence_error"]
                layer_target = accumulator["layer_target"]
                layer_error = accumulator["layer_error"]
                assert isinstance(sequence_target, np.ndarray)
                assert isinstance(sequence_error, np.ndarray)
                assert isinstance(layer_target, np.ndarray)
                assert isinstance(layer_error, np.ndarray)
                sequence_target[heldout] += target_sum
                sequence_error[heldout] += error_sum
                layer_target[layer] += target_sum
                layer_error[layer] += error_sum
                offset_target = accumulator["offset_target"]
                offset_error = accumulator["offset_error"]
                assert isinstance(offset_target, np.ndarray)
                assert isinstance(offset_error, np.ndarray)
                for offset_index, position in enumerate(_BLOCK_ENTRY_POSITIONS):
                    row = position - _READ_POSITIONS[0]
                    offset_target[offset_index] += target_per_position[row]
                    offset_error[offset_index] += error_per_position[row]
    outcomes: dict[int, dict[str, Any]] = {}
    for rank_value, accumulator in accumulators.items():
        sequence_target = accumulator["sequence_target"]
        sequence_error = accumulator["sequence_error"]
        layer_target = accumulator["layer_target"]
        layer_error = accumulator["layer_error"]
        offset_target = accumulator["offset_target"]
        offset_error = accumulator["offset_error"]
        assert isinstance(sequence_target, np.ndarray)
        assert isinstance(sequence_error, np.ndarray)
        assert isinstance(layer_target, np.ndarray)
        assert isinstance(layer_error, np.ndarray)
        assert isinstance(offset_target, np.ndarray)
        assert isinstance(offset_error, np.ndarray)
        outcomes[rank_value] = {
            "rank": rank_value,
            "global": _energy_metric(
                float(accumulator["global_target"]),
                float(accumulator["global_error"]),
            ),
            "heldout_sequences": [
                {
                    "record_index": index,
                    **_energy_metric(
                        float(sequence_target[index]),
                        float(sequence_error[index]),
                    ),
                }
                for index in range(_RECORDS)
            ],
            "layers": [
                {
                    "layer": layer,
                    **_energy_metric(
                        float(layer_target[layer]),
                        float(layer_error[layer]),
                    ),
                }
                for layer in range(_LAYERS)
            ],
            "block_entry_positions": [
                {
                    "position": position,
                    **_energy_metric(
                        float(offset_target[index]),
                        float(offset_error[index]),
                    ),
                }
                for index, position in enumerate(_BLOCK_ENTRY_POSITIONS)
            ],
            "folds": [
                {
                    "heldout_record_index": heldout,
                    "training_record_indices": [
                        index for index in range(_RECORDS) if index != heldout
                    ],
                    "training_positions_per_record": len(_READ_POSITIONS),
                    "heldout_coefficients": "oracle",
                }
                for heldout in range(_RECORDS)
            ],
        }
    return outcomes


def _metric_tree_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_metric_tree_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_metric_tree_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _attach_capacity_gate(outcome: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(outcome)
    sequence_rows = value["heldout_sequences"]
    layer_rows = value["layers"]
    offset_rows = value["block_entry_positions"]
    checks = {
        "finite": _metric_tree_finite(value),
        "global_recovery_at_least_0_50": (float(value["global"]["recovery"]) >= 0.50),
        "every_sequence_recovery_at_least_0_25": all(
            float(row["recovery"]) >= 0.25 for row in sequence_rows
        ),
        "every_block_entry_recovery_at_least_0_25": all(
            float(row["recovery"]) >= 0.25 for row in offset_rows
        ),
        "at_least_12_of_16_layers_positive_recovery": (
            sum(float(row["recovery"]) > 0.0 for row in layer_rows) >= 12
        ),
    }
    checks["passed"] = all(checks.values())
    value["gate"] = checks
    value["positive_recovery_layer_count"] = sum(
        float(row["recovery"]) > 0.0 for row in layer_rows
    )
    value["worst_sequence_error_ratio"] = max(
        float(row["error_ratio"]) for row in sequence_rows
    )
    return value


def _select_capacity_outcome(
    outcomes: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(outcomes) != set(_RANKS):
        raise ValueError("residual-capacity rank population changed")
    gated = {
        rank_value: _attach_capacity_gate(outcomes[rank_value]) for rank_value in _RANKS
    }
    passing = [
        rank_value for rank_value in _RANKS if gated[rank_value]["gate"]["passed"]
    ]
    if passing:
        selected_rank = passing[0]
        role = "smallest_passing_rank"
        passed = True
    else:
        selected_rank = min(
            _RANKS,
            key=lambda rank_value: (
                float(gated[rank_value]["worst_sequence_error_ratio"]),
                float(gated[rank_value]["global"]["error_ratio"]),
                rank_value,
            ),
        )
        role = "best_failed_rank_for_diagnostic_replay"
        passed = False
    selected = gated[selected_rank]
    return {
        "rank_order": list(_RANKS),
        "rank_outcomes": {str(rank_value): gated[rank_value] for rank_value in _RANKS},
        "selected_rank": selected_rank,
        "selection_role": role,
        "selection_key": [
            float(selected["worst_sequence_error_ratio"]),
            float(selected["global"]["error_ratio"]),
            selected_rank,
        ],
        "passed": passed,
    }


def _capacity_screen(targets: np.ndarray) -> dict[str, Any]:
    all_metrics = _capacity_metrics(targets, (0, *_RANKS))
    rank_zero = all_metrics.pop(0)
    selection = _select_capacity_outcome(all_metrics)
    selected_rank = int(selection["selected_rank"])
    replay = _capacity_metrics(targets, (0, *_RANKS))[selected_rank]
    reference = all_metrics[selected_rank]
    checks = {
        "selected_rank_exact_metric_recomputation": replay == reference,
        "selected_rank_metric_sha256": (sha256_json(replay) == sha256_json(reference)),
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise ValueError("residual-capacity selected metric replay failed")
    selection["rank_zero_mean_baseline"] = rank_zero
    selection["selected_metric_replay"] = {
        "rank": selected_rank,
        "reference_sha256": sha256_json(reference),
        "recomputed_sha256": sha256_json(replay),
        "checks": checks,
        "passed": True,
    }
    return selection


def _screen_post_authentication(
    context: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
) -> dict[str, bool]:
    checks = _base_post_authentication(context, checkpoint=checkpoint)
    checks.update(
        {
            "capacity_protocol": (
                sha256_file(context["capacity_protocol_path"])
                == context["capacity_protocol_sha256"]
            ),
            "capacity_parity": (
                sha256_file(context["capacity_parity_path"])
                == context["capacity_parity_sha256"]
            ),
            "capacity_source_inventory": (
                context["capacity_protocol"]["source_sha256"] == _source_inventory()
            ),
        }
    )
    return checks


def screen_residual_capacity(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    shard_dir: str | Path,
    out: str | Path,
    runtime_factory: Callable[[Mapping[str, Any]], Any] = _open_trace_runtime,
) -> dict[str, Any]:
    output = bias.rank.retrieval._new_output(
        out,
        "residual-capacity result",
    )
    directory = _prepare_shard_directory(shard_dir)
    started = time.perf_counter()
    context, _training, frozen = _authenticate_protocol(
        protocol,
        protocol_sha256,
    )
    records = context["train_records"]
    schedules = [
        bias.rank.fixed._derive_schedule(
            record["input_ids"],
            frozen["schedule_contract"]["tokenizer_fact_anchor_ids"],
        )
        for record in records
    ]
    if [row["rows_sha256"] for row in schedules] != frozen["schedule_contract"][
        "per_record_rows_sha256"
    ]:
        raise ValueError("residual-capacity execution schedule changed")
    runtime = runtime_factory(context)
    manifest_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    try:
        _validate_runtime_route(runtime, shadow=True)
        trace_runtime = _TraceCaptureRuntime(runtime)
        for index, (record, schedule, historical) in enumerate(
            zip(
                records,
                schedules,
                context["historical_k256_evidence"],
                strict=True,
            )
        ):
            _progress(f"capturing train record {index + 1}/{_RECORDS}")
            first, arrays, positions = _execute_record(
                trace_runtime,
                record=record,
                context=context,
                schedule=schedule,
                resource=frozen["fixed_K256_arm"]["resource_contract"],
                progress_label=f"trace train record {index + 1}/{_RECORDS}",
            )
            assert arrays is not None and positions is not None
            if not _evidence_exact(first, historical):
                raise ValueError(
                    f"residual-capacity record {index} base evidence changed"
                )
            first_summary = _trace_summary(arrays, positions)
            trace_runtime.reset()
            replay, replay_arrays, replay_positions = _execute_record(
                trace_runtime,
                record=record,
                context=context,
                schedule=schedule,
                resource=frozen["fixed_K256_arm"]["resource_contract"],
            )
            assert replay_arrays is not None and replay_positions is not None
            replay_summary = _trace_summary(replay_arrays, replay_positions)
            if not _evidence_exact(first, replay) or first_summary != replay_summary:
                raise ValueError(
                    f"residual-capacity record {index} reset replay changed"
                )
            source_record_sha256 = sha256_json(record)
            output_sha256 = sha256_json(_without_elapsed(first))
            reset_output_sha256 = sha256_json(_without_elapsed(replay))
            shard = _write_trace_shard(
                directory,
                record=record,
                arrays=arrays,
                positions=positions,
                source_record_sha256=source_record_sha256,
                output_sha256=output_sha256,
                reset_output_sha256=reset_output_sha256,
                reset_trace_sha256=replay_summary["trace_sha256"],
            )
            manifest_rows.append(shard)
            output_rows.append(
                {
                    "record_index": index,
                    "record_id": record["record_id"],
                    "historical_output_evidence_sha256": sha256_json(
                        _without_elapsed(historical)
                    ),
                    "observed_output_evidence_sha256": output_sha256,
                    "reset_output_evidence_sha256": reset_output_sha256,
                    "base_outputs_counters_and_loss_exact": True,
                    "reset_outputs_counters_loss_and_trace_exact": True,
                    "answer_cross_entropy": first["answer_cross_entropy"],
                    "hidden_sha256": first["hidden_sha256"],
                    "logits_sha256": first["logits_sha256"],
                    "counter_stream_sha256": first["counter_stream_sha256"],
                    "episodic_call_stream_sha256": first["episodic_call_stream_sha256"],
                    "source_record_sha256": source_record_sha256,
                    "trace_sha256": first_summary["trace_sha256"],
                    "reset_trace_sha256": replay_summary["trace_sha256"],
                    "shard_file_sha256": shard["file_sha256"],
                }
            )
            trace_runtime.reset()
    finally:
        runtime.close()
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _RESULT_EXPERIMENT,
        "protocol": {
            "path": str(context["capacity_protocol_path"]),
            "sha256": context["capacity_protocol_sha256"],
        },
        "format": "safetensors",
        "record_order": list(range(_RECORDS)),
        "shards": manifest_rows,
        "confirmation_split_opened": False,
    }
    manifest_path = directory / "manifest.json"
    atomic_json(manifest_path, manifest)
    validated_targets: list[np.ndarray] = []
    for descriptor in manifest_rows:
        arrays = _validate_trace_shard(
            directory / descriptor["file"],
            descriptor,
        )
        validated_targets.append(arrays["target_residual"])
    targets = np.ascontiguousarray(
        np.stack(validated_targets),
        dtype=np.float32,
    )
    _progress("running leave-one-sequence-out rank-0/2/4/8 capacity ceilings")
    capacity = _capacity_screen(targets)
    post = _screen_post_authentication(
        context,
        checkpoint=frozen["training_checkpoint"],
    )
    if not post or not all(post.values()):
        raise ValueError("residual-capacity post-run authentication failed")
    passed = bool(capacity["passed"])
    report = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _RESULT_EXPERIMENT,
        "status": (
            "train_residual_capacity_gate_passed"
            if passed
            else "train_residual_capacity_gate_failed"
        ),
        "protocol": {
            "path": str(context["capacity_protocol_path"]),
            "sha256": context["capacity_protocol_sha256"],
        },
        "scope": {
            "split": "train",
            "records": _RECORDS,
            "positions_per_record": _POSITIONS,
            "trace_positions_per_record": len(_READ_POSITIONS),
            "capacity_evidence_only": True,
            "predictor_fitted": False,
            "native_correction_integrated": False,
            "semantic_or_M3_gate_passed": False,
            "development_outcomes_used": False,
            "confirmation_split_opened": False,
        },
        "base_choice": frozen["base_choice"],
        "base_output_authentication": output_rows,
        "trace_manifest": {
            "directory": str(directory),
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "shard_count": len(manifest_rows),
            "shards": manifest_rows,
        },
        "capacity": capacity,
        "decision": {
            "capacity_gate_passed": passed,
            "semantic_or_M3_gate_passed": False,
            "native_integration_authorized": False,
            "development_authorized": False,
            "confirmation_authorized": False,
            "train_only_predictor_fit_authorized": passed,
            "next_step": (
                "fit train-only residual predictors with rank0, input-only, "
                "base-only, and input-plus-base controls"
                if passed
                else "close rank<=8 global per-layer output-subspace residuals"
            ),
            "failure_scope": (
                None
                if passed
                else (
                    "rank<=8 global per-layer output-subspace residuals only; "
                    "no broader residual or selector family is rejected"
                )
            ),
        },
        "post_run_authentication": post,
        "confirmation_split_opened": False,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output, report)
    _progress(f"residual-capacity result written to {output}")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train-only OLMoE same-state residual capacity screen",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    parity = commands.add_parser("parity")
    parity.add_argument("--bias-protocol", required=True)
    parity.add_argument("--bias-protocol-sha256", required=True)
    parity.add_argument("--bias-result", required=True)
    parity.add_argument("--bias-result-sha256", required=True)
    parity.add_argument("--trace-library", required=True)
    parity.add_argument("--trace-library-sha256", required=True)
    parity.add_argument("--out", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--bias-protocol", required=True)
    freeze.add_argument("--bias-protocol-sha256", required=True)
    freeze.add_argument("--bias-result", required=True)
    freeze.add_argument("--bias-result-sha256", required=True)
    freeze.add_argument("--trace-library", required=True)
    freeze.add_argument("--trace-library-sha256", required=True)
    freeze.add_argument("--parity-report", required=True)
    freeze.add_argument("--parity-report-sha256", required=True)
    freeze.add_argument("--out", required=True)
    screen = commands.add_parser("screen")
    screen.add_argument("--protocol", required=True)
    screen.add_argument("--protocol-sha256", required=True)
    screen.add_argument("--shard-dir", required=True)
    screen.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "parity":
        value = generate_trace_parity_report(
            bias_protocol=args.bias_protocol,
            bias_protocol_sha256=args.bias_protocol_sha256,
            bias_result=args.bias_result,
            bias_result_sha256=args.bias_result_sha256,
            trace_library=args.trace_library,
            trace_library_sha256=args.trace_library_sha256,
            out=args.out,
        )
    elif args.command == "freeze":
        value = freeze_residual_capacity_protocol(
            bias_protocol=args.bias_protocol,
            bias_protocol_sha256=args.bias_protocol_sha256,
            bias_result=args.bias_result,
            bias_result_sha256=args.bias_result_sha256,
            trace_library=args.trace_library,
            trace_library_sha256=args.trace_library_sha256,
            parity_report=args.parity_report,
            parity_report_sha256=args.parity_report_sha256,
            out=args.out,
        )
    elif args.command == "screen":
        value = screen_residual_capacity(
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            shard_dir=args.shard_dir,
            out=args.out,
        )
    else:  # pragma: no cover - argparse owns this boundary
        raise AssertionError("unknown residual-capacity command")
    print(json.dumps(value, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
