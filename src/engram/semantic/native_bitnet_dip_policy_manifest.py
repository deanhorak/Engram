"""Fail-closed freeze manifest for the native BitNet DIP policy.

The validation-only float16 trace can propose a q/C/K schedule, but it cannot
approve the live policy: float16 trace boundaries do not preserve the BF16
operator boundary or the native accumulation order.  This module therefore
freezes a policy only after an unchanged, all-layer native CPU kernel has
passed the declared 8-sequence/32-position development protocol.

Approval is intentionally stricter than copying booleans from an experiment
report.  The builder:

* serializes every effective field for every layer;
* authenticates the package, record artifact, coordinate index, both native
  libraries, protocol, proposal, and development report;
* recomputes activity from all per-token/per-layer selected counts;
* recomputes v2 cache-line traffic for every token; and
* reloads the source-bound coordinate index and compares its embedded policy.

It never opens the sealed final dataset.  A successful manifest authorizes
one final-confirmation run; it does not claim that Milestone 2 has passed.
Compiler-build provenance is intentionally frozen by the later authorization
manifest, after this policy is committed; binding it here would create an
impossible commit/hash cycle.  Both produced library bytes are bound here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from engram.evaluation.native_bitnet_dip_traffic import (
    native_bitnet_dip_physical_accounting,
)
from engram.models.native_bitnet import load_native_bitnet_artifact
from engram.utils import atomic_json, sha256_file, sha256_json


NATIVE_BITNET_DIP_POLICY_FORMAT = "engram-native-bitnet-dip-policy"
NATIVE_BITNET_DIP_POLICY_VERSION = 1
NATIVE_BITNET_DIP_POLICY_STATUS = "approved"
NATIVE_BITNET_DIP_TRAFFIC_FORMAT = "native_bitnet_dip_dual_layout_v2"


class NativeBitNetDIPPolicyManifestError(ValueError):
    """Raised when policy-freeze evidence is absent or inconsistent."""


@dataclass(frozen=True)
class FrozenNativeBitNetDIPLayerPolicy:
    """Complete effective routing and normalization policy for one layer."""

    layer: int
    input_fraction: float
    input_coordinates: int
    candidate_count: int
    minimum_top_k: int
    maximum_top_k: int
    energy_target: float
    rms_estimator: str
    rms_audit_count: int
    rms_audit_strategy: str
    rms_variance_scale: float
    rms_variance_bias: float
    output_scale: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "input_fraction": self.input_fraction,
            "input_coordinates": self.input_coordinates,
            "candidate_count": self.candidate_count,
            "minimum_top_k": self.minimum_top_k,
            "maximum_top_k": self.maximum_top_k,
            "energy_target": self.energy_target,
            "rms_estimator": self.rms_estimator,
            "rms_audit_count": self.rms_audit_count,
            "rms_audit_strategy": self.rms_audit_strategy,
            "rms_variance_scale": self.rms_variance_scale,
            "rms_variance_bias": self.rms_variance_bias,
            "output_scale": self.output_scale,
        }


@dataclass(frozen=True)
class LoadedNativeBitNetDIPPolicyManifest:
    """A fully reconstructed and authenticated frozen policy manifest."""

    path: Path
    manifest_sha256: str
    layers: tuple[FrozenNativeBitNetDIPLayerPolicy, ...]
    payload: Mapping[str, Any]


def _fail(message: str) -> None:
    raise NativeBitNetDIPPolicyManifestError(message)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeBitNetDIPPolicyManifestError(
            f"cannot read {label}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _integer(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        _fail(f"{label} must be at least {minimum}")
    if maximum is not None and value > maximum:
        _fail(f"{label} must be at most {maximum}")
    return value


def _number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail(f"{label} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        _fail(f"{label} must be at least {minimum}")
    if maximum is not None and result > maximum:
        _fail(f"{label} must be at most {maximum}")
    return result


def _require_true(value: Any, label: str) -> None:
    if value is not True:
        _fail(f"{label} must be true")


def _require_false(value: Any, label: str) -> None:
    if value is not False:
        _fail(f"{label} must be false")


def _close(actual: float, expected: float, label: str) -> None:
    # Causal NLL values are emitted from separate float32 reductions while
    # traffic/activity summaries are float64 ratios.  One float32 ULP is
    # tolerated; byte counts and selected counts remain exact integers.
    if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-7):
        _fail(f"{label} does not reconcile with executed schedules")


def _close_ratio(actual: float, expected: float, label: str) -> None:
    """Reconcile ratios derived deterministically from integer evidence."""

    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        _fail(f"{label} does not reconcile with integer evidence")


def _file_descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        _fail(f"bound file is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _reported_file_matches(
    reported: Any,
    actual: Mapping[str, Any],
    *,
    report_directory: Path,
    label: str,
) -> None:
    descriptor = _object(reported, f"development artifacts.{label}")
    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        _fail(f"development artifacts.{label}.path must be a string")
    reported_path = Path(raw_path).expanduser()
    if not reported_path.is_absolute():
        reported_path = report_directory / reported_path
    if reported_path.resolve() != Path(str(actual["path"])):
        _fail(f"development artifacts.{label}.path mismatch")
    if descriptor.get("sha256") != actual["sha256"]:
        _fail(f"development artifacts.{label}.sha256 mismatch")
    if descriptor.get("bytes") != actual["bytes"]:
        _fail(f"development artifacts.{label}.bytes mismatch")


def _parse_layers(
    development: Mapping[str, Any],
    *,
    hidden_size: int,
    intermediate_size: int,
    layer_count: int,
) -> tuple[FrozenNativeBitNetDIPLayerPolicy, ...]:
    configuration = _object(
        development.get("configuration"),
        "development configuration",
    )
    expected_keys = {str(layer) for layer in range(layer_count)}
    if set(configuration) != expected_keys:
        _fail(
            "development configuration must contain exactly one entry for "
            f"each of {layer_count} layers"
        )

    layers: list[FrozenNativeBitNetDIPLayerPolicy] = []
    for layer in range(layer_count):
        raw = _object(
            configuration[str(layer)],
            f"development configuration[{layer}]",
        )
        input_coordinates = raw.get("input_coordinates")
        input_coordinates = _integer(
            input_coordinates,
            f"layer {layer} input_coordinates",
            minimum=1,
            maximum=hidden_size,
        )
        raw_fraction = raw.get("input_fraction")
        fraction = (
            input_coordinates / hidden_size
            if raw_fraction is None
            else _number(
                raw_fraction,
                f"layer {layer} input_fraction",
                minimum=0.0,
                maximum=1.0,
            )
        )
        if fraction <= 0.0:
            _fail(f"layer {layer} input_fraction must be positive")
        if math.ceil(fraction * hidden_size) != input_coordinates:
            _fail(f"layer {layer} q fraction/count are inconsistent")
        candidate_count = _integer(
            raw.get("candidate_count"),
            f"layer {layer} candidate_count",
            minimum=1,
            maximum=intermediate_size,
        )
        minimum_top_k = _integer(
            raw.get("minimum_top_k"),
            f"layer {layer} minimum_top_k",
            minimum=1,
            maximum=intermediate_size,
        )
        maximum_top_k = _integer(
            raw.get("maximum_top_k"),
            f"layer {layer} maximum_top_k",
            minimum=1,
            maximum=intermediate_size,
        )
        if raw.get("top_k") is not None and raw.get("top_k") != maximum_top_k:
            _fail(f"layer {layer} top_k must equal maximum_top_k")
        energy_target = _number(
            raw.get("energy_target"),
            f"layer {layer} energy_target",
            minimum=0.0,
            maximum=1.0,
        )
        if energy_target <= 0.0:
            _fail(f"layer {layer} energy_target must be positive")
        audit_count = _integer(
            raw.get("rms_audit_count"),
            f"layer {layer} rms_audit_count",
            minimum=0,
            maximum=candidate_count,
        )
        estimator = raw.get("rms_estimator")
        audit_strategy = raw.get("rms_audit_strategy")
        if estimator not in {"candidate_ratio", "corrected_proxy"}:
            _fail(f"layer {layer} rms_estimator is unsupported")
        if audit_strategy not in {"none", "top_proxy_raw_square"}:
            _fail(f"layer {layer} rms_audit_strategy is unsupported")
        if estimator == "candidate_ratio":
            if audit_count or audit_strategy != "none":
                _fail(
                    f"layer {layer} candidate_ratio RMS policy cannot audit"
                )
        elif audit_count <= 0 or audit_strategy != "top_proxy_raw_square":
            _fail(
                f"layer {layer} corrected_proxy RMS policy requires a "
                "top_proxy_raw_square audit"
            )
        if not minimum_top_k <= maximum_top_k <= candidate_count - audit_count:
            _fail(f"layer {layer} C/minK/maxK/audit bounds are inconsistent")

        rms_variance_scale = _number(
            raw.get("rms_variance_scale"),
            f"layer {layer} rms_variance_scale",
        )
        rms_variance_bias = _number(
            raw.get("rms_variance_bias"),
            f"layer {layer} rms_variance_bias",
        )
        output_scale = _number(
            raw.get("output_scale"),
            f"layer {layer} output_scale",
        )
        if (
            rms_variance_scale != 1.0
            or rms_variance_bias != 0.0
            or output_scale != 1.0
        ):
            _fail(
                f"layer {layer} contains an unapproved fitted normalization "
                "or output scale"
            )
        layers.append(
            FrozenNativeBitNetDIPLayerPolicy(
                layer=layer,
                input_fraction=fraction,
                input_coordinates=input_coordinates,
                candidate_count=candidate_count,
                minimum_top_k=minimum_top_k,
                maximum_top_k=maximum_top_k,
                energy_target=energy_target,
                rms_estimator=str(estimator),
                rms_audit_count=audit_count,
                rms_audit_strategy=str(audit_strategy),
                rms_variance_scale=rms_variance_scale,
                rms_variance_bias=rms_variance_bias,
                output_scale=output_scale,
            )
        )
    return tuple(layers)


def _validate_proposal(
    proposal: Mapping[str, Any],
    protocol: Mapping[str, Any],
    layers: Sequence[FrozenNativeBitNetDIPLayerPolicy],
    *,
    artifact_sha256: str,
) -> dict[str, Any]:
    if (
        proposal.get("experiment")
        != "native_bitnet_dip_joint_candidate_adaptive_k_policy"
    ):
        _fail("proposal report has an unsupported experiment")
    if proposal.get("artifact_sha256") != artifact_sha256:
        _fail("proposal report record-artifact SHA-256 mismatch")
    validation = _object(
        proposal.get("validation_trace"),
        "proposal validation_trace",
    )
    fit = _object(protocol.get("configuration_fit"), "configuration_fit")
    if validation.get("dataset_hash") != fit.get("dataset_sha256"):
        _fail("proposal trace does not match the declared fit corpus")
    _require_false(
        validation.get("causal_or_final_confirmation_corpus_used"),
        "proposal causal_or_final_confirmation_corpus_used",
    )
    if fit.get("role") != "noncanonical_policy_proposal_only":
        _fail("protocol does not demote the fit trace to proposal-only")
    if fit.get("stored_dtype") != "float16":
        _fail("proposal provenance must identify the float16 trace boundary")

    configuration = _object(
        proposal.get("configuration"),
        "proposal configuration",
    )
    selected = _object(
        proposal.get("selected_policy"),
        "proposal selected_policy",
    )
    candidates = selected.get("candidate_counts")
    maximums = selected.get("maximum_ks")
    rms_policies = selected.get("rms_policies")
    if not all(
        isinstance(value, list) and len(value) == len(layers)
        for value in (candidates, maximums, rms_policies)
    ):
        _fail("proposal schedules do not cover every layer")
    proposal_input = _integer(
        configuration.get("input_coordinates"),
        "proposal input_coordinates",
        minimum=1,
    )
    proposal_fraction = _number(
        configuration.get("input_fraction"),
        "proposal input_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    proposal_minimum = _integer(
        configuration.get("minimum_k"),
        "proposal minimum_k",
        minimum=1,
    )
    proposal_energy = _number(
        configuration.get("energy_target"),
        "proposal energy_target",
        minimum=0.0,
        maximum=1.0,
    )
    for layer, candidate, maximum, rms in zip(
        layers,
        candidates,
        maximums,
        rms_policies,
        strict=True,
    ):
        rms = _object(rms, f"proposal rms_policies[{layer.layer}]")
        if (
            proposal_input != layer.input_coordinates
            or proposal_fraction != layer.input_fraction
            or proposal_minimum != layer.minimum_top_k
            or proposal_energy != layer.energy_target
            or candidate != layer.candidate_count
            or maximum != layer.maximum_top_k
            or rms.get("estimator") != layer.rms_estimator
            or rms.get("audit_count") != layer.rms_audit_count
            or rms.get("audit_strategy") != layer.rms_audit_strategy
        ):
            _fail(
                f"live native policy differs from proposal at layer "
                f"{layer.layer}; a new proposal/protocol revision is required"
            )

    physical = proposal.get("physical_layout")
    proposal_traffic_format = (
        physical.get("format")
        if isinstance(physical, dict)
        else None
    )
    if proposal_traffic_format is None and isinstance(physical, dict):
        proposal_traffic_format = physical.get("accounting")
    reported_layout = (
        physical.get("layout")
        if isinstance(physical, dict)
        and isinstance(physical.get("layout"), dict)
        else {}
    )
    return {
        "classification": "float16_trace_policy_proposal_only",
        "approval_authority": False,
        "traffic_evidence_accepted": False,
        "stored_boundary_dtype": "float16",
        "proposal_traffic_format": proposal_traffic_format,
        "reported_index_header_bytes": reported_layout.get(
            "index_header_bytes"
        ),
        "reported_index_layer_header_bytes": reported_layout.get(
            "index_layer_header_bytes"
        ),
        "demotion_reason": (
            "stored float16 states cannot reproduce live BF16 boundaries or "
            "native accumulation; proposal traffic predating the source-bound "
            "128-byte v2 headers is also non-qualifying"
        ),
        "effective_fields_revalidated_on_live_native_bf16": True,
    }


def _validate_protocol_and_development_scope(
    protocol: Mapping[str, Any],
    development: Mapping[str, Any],
    *,
    layer_count: int,
) -> tuple[dict[str, float], dict[str, float]]:
    if (
        protocol.get("experiment")
        != "native_bitnet_milestone_2_practical_semantic_memory_confirmation"
    ):
        _fail("frozen protocol has an unsupported experiment")
    if protocol.get("configuration") is not None:
        _fail("frozen protocol already contains a configuration")
    if protocol.get("final_result") is not None:
        _fail("frozen protocol already contains a final result")

    causal = _object(protocol.get("causal_development"), "causal_development")
    declared = _object(
        causal.get("full_length_dataset"),
        "causal_development.full_length_dataset",
    )
    final = _object(protocol.get("final_confirmation"), "final_confirmation")
    dataset = _object(development.get("dataset"), "development dataset")
    if dataset.get("sha256") != declared.get("sha256"):
        _fail("development report does not use the declared full-length corpus")
    if dataset.get("sha256") == final.get("dataset_sha256"):
        _fail("sealed final corpus cannot be used as development evidence")
    allowed_offsets = declared.get("allowed_record_offsets")
    if (
        not isinstance(allowed_offsets, list)
        or dataset.get("record_offset") not in allowed_offsets
    ):
        _fail("development record offset is not protocol-authorized")
    for label, actual, required in (
        (
            "sequence_count",
            dataset.get("sequence_count", dataset.get("samples")),
            8,
        ),
        (
            "predictions_per_sequence",
            dataset.get("predictions_per_sequence"),
            32,
        ),
        ("prediction_positions", dataset.get("prediction_positions"), 256),
    ):
        if actual != required:
            _fail(f"development {label} must equal {required}")

    evidence = _object(
        development.get("evidence_observed"),
        "development evidence_observed",
    )
    expected_evidence = {
        "sequences": 8,
        "unique_sequences": 8,
        "predictions_per_sequence": 32,
        "prediction_positions": 256,
        "all_mlp_layers": True,
        "layer_count": layer_count,
        "layers_executed": list(range(layer_count)),
    }
    for key, expected in expected_evidence.items():
        if evidence.get(key) != expected:
            _fail(f"development evidence_observed.{key} is not qualifying")

    quality = _object(
        protocol.get("quality_thresholds"),
        "quality_thresholds",
    )
    practical = _object(
        protocol.get("practical_router_thresholds"),
        "practical_router_thresholds",
    )
    thresholds = {
        "maximum_mean_kl_divergence": _number(
            quality.get("maximum_teacher_student_kl"),
            "maximum_teacher_student_kl",
            minimum=0.0,
        ),
        "minimum_top1_agreement": _number(
            quality.get("minimum_top1_agreement"),
            "minimum_top1_agreement",
            minimum=0.0,
            maximum=1.0,
        ),
        "maximum_nll_delta": _number(
            quality.get("maximum_nll_delta"),
            "maximum_nll_delta",
        ),
        "maximum_final_hidden_relative_l2": _number(
            quality.get("maximum_final_hidden_relative_l2"),
            "maximum_final_hidden_relative_l2",
            minimum=0.0,
        ),
    }
    router_thresholds = {
        "maximum_mean_active_record_fraction": _number(
            practical.get("maximum_mean_active_record_fraction"),
            "maximum_mean_active_record_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
        "maximum_complete_physical_cold_traffic_fraction_of_dense_q4": (
            _number(
                practical.get(
                    "maximum_complete_physical_cold_traffic_fraction_of_dense_q4"
                ),
                "maximum_complete_physical_cold_traffic_fraction_of_dense_q4",
                minimum=0.0,
                maximum=1.0,
            )
        ),
    }
    _require_true(
        practical.get("cpu_only_inference_required"),
        "cpu_only_inference_required",
    )
    _require_false(
        practical.get("dense_gate_up_or_down_fallback_allowed"),
        "dense_gate_up_or_down_fallback_allowed",
    )
    return thresholds, router_thresholds


def _validate_execution(
    development: Mapping[str, Any],
) -> dict[str, Any]:
    if development.get("experiment") != "native_bitnet_dip_native_causal":
        _fail("development report is not native BitNet DIP causal evidence")
    if development.get("dataset_role") != "development":
        _fail("native BF16 policy evidence must have dataset_role=development")
    if (
        development.get("milestone_2_status")
        != "development_gate_passed_pending_final"
        or development.get("decision")
        != "freeze_policy_and_run_protected_final_confirmation"
    ):
        _fail(
            "development report must remain pending sealed final confirmation"
        )
    execution = _object(development.get("execution"), "development execution")
    expected = {
        "input_boundary": "live_native_bf16",
        "kernel": "native_cpu",
        "device": "cpu",
        "dense_fallback": False,
        "all_mlp_layers_substituted": True,
        "serialized_index_reloaded": True,
        "python_native_parity_passed": True,
    }
    for key, value in expected.items():
        if execution.get(key) != value:
            _fail(f"development execution.{key} must equal {value!r}")
    return dict(expected)


def _validate_development_progression(
    development: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    for key in (
        "candidate_recall_passed",
        "systems_evidence_passed",
        "protocol_qualifying",
        "overall_gate_passed",
    ):
        _require_true(development.get(key), f"development {key}")
    reference = _object(
        development.get("reference_top_ks"),
        "development reference_top_ks",
    )
    definition = _object(
        protocol.get("candidate_recall_definition"),
        "candidate_recall_definition",
    )
    practical = _object(
        protocol.get("practical_router_thresholds"),
        "practical_router_thresholds",
    )
    required_recall = _number(
        practical.get("minimum_held_out_candidate_recall"),
        "minimum_held_out_candidate_recall",
        minimum=0.0,
        maximum=1.0,
    )
    if required_recall != 0.95:
        _fail("Milestone 2 candidate-recall threshold must remain 0.95")
    expected = definition.get("reference_top_ks")
    if (
        not isinstance(expected, list)
        or len(expected) != 30
        or reference.get("values") != expected
        or reference.get("sha256") != sha256_json(expected)
        or reference.get("role")
        != "frozen_fixed_per_layer_candidate_recall_denominator"
    ):
        _fail("development candidate-recall reference schedule is not frozen")
    debug = _object(development.get("debug_recall"), "development debug_recall")
    global_recall = _object(
        debug.get("global"),
        "development debug_recall.global",
    )
    layer_reports = _object(
        debug.get("layers"),
        "development debug_recall.layers",
    )
    if set(layer_reports) != {str(layer) for layer in range(len(expected))}:
        _fail("development recall must contain exactly one report per layer")
    total_targets = 0
    total_hits = 0
    layer_means: list[float] = []
    for layer, reference_top_k in enumerate(expected):
        reference_top_k = _integer(
            reference_top_k,
            f"candidate_recall_definition.reference_top_ks[{layer}]",
            minimum=1,
        )
        report = _object(
            layer_reports[str(layer)],
            f"development debug_recall.layers[{layer}]",
        )
        rows = _integer(
            report.get("rows"),
            f"development recall layer {layer} rows",
            minimum=1,
        )
        if rows != 256:
            _fail(
                f"development recall layer {layer} must contain exactly "
                "256 scored rows"
            )
        if report.get("layer") != layer:
            _fail(f"development recall layer {layer} identity mismatch")
        if report.get("reference_top_k") != reference_top_k:
            _fail(f"development recall layer {layer} reference K mismatch")
        targets = _integer(
            report.get("target_records"),
            f"development recall layer {layer} target_records",
            minimum=1,
        )
        hits = _integer(
            report.get("candidate_hits"),
            f"development recall layer {layer} candidate_hits",
            minimum=0,
            maximum=targets,
        )
        if targets != rows * reference_top_k:
            _fail(
                f"development recall layer {layer} target_records do not "
                "reconcile with rows and fixed K"
            )
        layer_mean = hits / targets
        for key in (
            "candidate_micro_recall",
            "candidate_mean_row_recall",
        ):
            _close_ratio(
                _number(
                    report.get(key),
                    f"development recall layer {layer} {key}",
                    minimum=0.0,
                    maximum=1.0,
                ),
                layer_mean,
                f"development recall layer {layer} {key}",
            )
        total_targets += targets
        total_hits += hits
        layer_means.append(layer_mean)

    micro = total_hits / total_targets
    macro = sum(layer_means) / len(layer_means)
    minimum_layer = min(layer_means)
    if (
        global_recall.get("rows") != 256 * len(expected)
        or global_recall.get("target_records") != total_targets
        or global_recall.get("candidate_hits") != total_hits
    ):
        _fail("development global recall integer totals do not reconcile")
    _close_ratio(
        _number(
            global_recall.get("minimum_candidate_recall"),
            "development candidate recall minimum_candidate_recall",
            minimum=0.0,
            maximum=1.0,
        ),
        required_recall,
        "development candidate recall minimum_candidate_recall",
    )
    for key, expected_value in (
        ("candidate_micro_recall", micro),
        ("macro_mean_layer_recall", macro),
        ("candidate_minimum_layer_mean_recall", minimum_layer),
    ):
        _close_ratio(
            _number(
                global_recall.get(key),
                f"development candidate recall {key}",
                minimum=0.0,
                maximum=1.0,
            ),
            expected_value,
            f"development candidate recall {key}",
        )
    expected_global_pass = micro >= required_recall
    expected_layers_pass = minimum_layer >= required_recall
    expected_pass = expected_global_pass and expected_layers_pass
    if (
        global_recall.get("global_micro_passes_95_percent")
        is not expected_global_pass
        or global_recall.get("every_layer_mean_passes_95_percent")
        is not expected_layers_pass
        or global_recall.get("passes_95_percent") is not expected_pass
        or development.get("candidate_recall_passed") is not expected_pass
    ):
        _fail("development candidate-recall pass booleans are forged")
    if micro < required_recall or minimum_layer < required_recall:
        _fail("development candidate recall is below 95 percent")
    return {
        "reference_top_ks": list(expected),
        "reference_top_ks_sha256": reference["sha256"],
        "target_records": total_targets,
        "candidate_hits": total_hits,
        "global_micro_membership_recall": micro,
        "macro_mean_layer_recall": macro,
        "minimum_layer_mean_recall": minimum_layer,
        "passed": True,
        "final_holdout_recall_still_required": True,
    }


def _validate_parity(
    parity: Mapping[str, Any],
    development: Mapping[str, Any],
    *,
    layer_count: int,
    actual_artifacts: Mapping[str, Mapping[str, Any]],
    report_directory: Path,
) -> dict[str, Any]:
    if (
        parity.get("experiment")
        != "native_bitnet_dip_full_artifact_parity"
        or parity.get("scope") != "all_30_layers_live_bf16_development"
    ):
        _fail("native parity report has an unsupported experiment or scope")
    _require_true(parity.get("passed"), "native parity passed")
    _require_false(
        parity.get("protected_holdout_used"),
        "native parity protected_holdout_used",
    )
    execution = _object(parity.get("execution"), "native parity execution")
    expected_execution = {
        "device": "cpu",
        "input_boundary": "live_native_bf16",
        "python_reference": "native_bitnet_dip_bf16_reference",
        "native_kernel": "native_cpu",
    }
    for key, expected in expected_execution.items():
        if execution.get(key) != expected:
            _fail(f"native parity execution.{key} must equal {expected!r}")
    evidence = _object(parity.get("evidence"), "native parity evidence")
    if (
        evidence.get("layer_count") != layer_count
        or evidence.get("layers_executed") != list(range(layer_count))
    ):
        _fail("native parity evidence must cover exactly all 30 layers")
    rows_per_layer = _integer(
        evidence.get("rows_per_layer"),
        "native parity rows_per_layer",
        minimum=6,
    )
    total_rows = _integer(
        evidence.get("total_rows"),
        "native parity total_rows",
        minimum=layer_count,
    )
    if total_rows != rows_per_layer * layer_count:
        _fail("native parity total_rows does not reconcile")
    input_tokens = _integer(
        evidence.get("input_tokens"),
        "native parity input_tokens",
        minimum=rows_per_layer,
    )
    dataset = _object(parity.get("dataset"), "native parity dataset")
    development_dataset = _object(
        development.get("dataset"),
        "development dataset",
    )
    if dataset.get("sha256") != development_dataset.get("sha256"):
        _fail("native parity did not use the declared development corpus")
    equality = _object(parity.get("equality"), "native parity equality")
    required_equality = (
        "input_coordinate_ids",
        "candidate_ids",
        "selected_record_ids",
        "selected_counts",
        "output_bf16_bits",
    )
    for key in required_equality:
        _require_true(equality.get(key), f"native parity equality.{key}")
    layer_reports = parity.get("layers")
    if not isinstance(layer_reports, list) or len(layer_reports) != layer_count:
        _fail("native parity must contain exactly 30 layer proofs")
    for layer, raw_report in enumerate(layer_reports):
        report = _object(raw_report, f"native parity layers[{layer}]")
        if report.get("layer") != layer or report.get("rows") != rows_per_layer:
            _fail(f"native parity layer {layer} identity/row count mismatch")
        row_indices = report.get("row_indices")
        if (
            not isinstance(row_indices, list)
            or len(row_indices) != rows_per_layer
            or any(
                isinstance(row, bool)
                or not isinstance(row, int)
                or not 0 <= row < input_tokens
                for row in row_indices
            )
            or len(set(row_indices)) != rows_per_layer
        ):
            _fail(f"native parity layer {layer} row_indices are invalid")
        selected_counts = report.get("selected_counts")
        if (
            not isinstance(selected_counts, list)
            or len(selected_counts) != rows_per_layer
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in selected_counts
            )
        ):
            _fail(f"native parity layer {layer} selected_counts are invalid")
        _require_true(
            report.get("includes_observed_minimum_k"),
            f"native parity layer {layer} includes_observed_minimum_k",
        )
        _require_true(
            report.get("includes_observed_maximum_k"),
            f"native parity layer {layer} includes_observed_maximum_k",
        )
        layer_equality = _object(
            report.get("equality"),
            f"native parity layer {layer} equality",
        )
        for key in required_equality:
            _require_true(
                layer_equality.get(key),
                f"native parity layer {layer} equality.{key}",
            )
    reported_artifacts = _object(
        parity.get("artifacts"),
        "native parity artifacts",
    )
    for label, descriptor in actual_artifacts.items():
        _reported_file_matches(
            reported_artifacts.get(label),
            descriptor,
            report_directory=report_directory,
            label=label,
        )
    return {
        "execution": expected_execution,
        "dataset_sha256": dataset["sha256"],
        "layer_count": layer_count,
        "layers_executed": list(range(layer_count)),
        "rows_per_layer": rows_per_layer,
        "total_rows": total_rows,
        "input_tokens": input_tokens,
        "bit_exact_fields": list(required_equality),
        "passed": True,
    }


def _validate_quality(
    development: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    quality = _object(development.get("quality"), "development quality")
    mean_kl = _number(
        quality.get("mean_kl_divergence"),
        "quality.mean_kl_divergence",
        minimum=0.0,
    )
    top1 = _number(
        quality.get("top1_agreement"),
        "quality.top1_agreement",
        minimum=0.0,
        maximum=1.0,
    )
    reference_nll = _number(
        quality.get("reference_nll"),
        "quality.reference_nll",
        minimum=0.0,
    )
    candidate_nll = _number(
        quality.get("candidate_nll"),
        "quality.candidate_nll",
        minimum=0.0,
    )
    nll_delta = _number(quality.get("nll_delta"), "quality.nll_delta")
    hidden = _number(
        quality.get("final_hidden_relative_l2"),
        "quality.final_hidden_relative_l2",
        minimum=0.0,
    )
    _close(candidate_nll - reference_nll, nll_delta, "quality.nll_delta")
    passed = bool(
        mean_kl <= thresholds["maximum_mean_kl_divergence"]
        and top1 >= thresholds["minimum_top1_agreement"]
        and nll_delta <= thresholds["maximum_nll_delta"]
        and hidden <= thresholds["maximum_final_hidden_relative_l2"]
    )
    _require_true(quality.get("passed"), "development quality.passed")
    _require_true(
        development.get("quality_passed"),
        "development quality_passed",
    )
    if not passed:
        _fail("native BF16 development quality does not pass frozen thresholds")
    return {
        "mean_kl_divergence": mean_kl,
        "top1_agreement": top1,
        "reference_nll": reference_nll,
        "candidate_nll": candidate_nll,
        "nll_delta": nll_delta,
        "final_hidden_relative_l2": hidden,
        "passed": True,
    }


def _validate_index_policy(
    index_path: Path,
    artifact_sha256: str,
    layers: Sequence[FrozenNativeBitNetDIPLayerPolicy],
) -> None:
    # Import lazily so the index builder can consume an approved manifest
    # without creating a module-import cycle.
    from engram.semantic.native_bitnet_dip_index import (
        load_native_bitnet_dip_index,
    )

    with load_native_bitnet_dip_index(index_path) as index:
        if index.source_artifact_sha256 != artifact_sha256:
            _fail("coordinate index is not bound to the record artifact")
        if len(index.layers) != len(layers):
            _fail("coordinate index layer count differs from frozen policy")
        for expected, mapped in zip(layers, index.layers, strict=True):
            actual = mapped.policy
            if (
                actual.input_coordinates != expected.input_coordinates
                or actual.candidate_count != expected.candidate_count
                or actual.minimum_top_k != expected.minimum_top_k
                or actual.maximum_top_k != expected.maximum_top_k
                or actual.energy_target != expected.energy_target
                or actual.rms_estimator != expected.rms_estimator
                or actual.rms_audit_count != expected.rms_audit_count
                or actual.rms_audit_strategy
                != expected.rms_audit_strategy
            ):
                _fail(
                    f"coordinate index policy differs at layer "
                    f"{expected.layer}"
                )


def _validate_selected_schedules_and_traffic(
    development: Mapping[str, Any],
    layers: Sequence[FrozenNativeBitNetDIPLayerPolicy],
    *,
    hidden_size: int,
    intermediate_size: int,
    router_thresholds: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selected = _object(
        development.get("selected_records"),
        "development selected_records",
    )
    raw_schedules = selected.get("per_token_layer_k")
    if not isinstance(raw_schedules, list) or len(raw_schedules) != 256:
        _fail("selected_records.per_token_layer_k must contain 256 rows")
    schedules: list[list[int]] = []
    for token, raw_schedule in enumerate(raw_schedules):
        if not isinstance(raw_schedule, list) or len(raw_schedule) != len(layers):
            _fail(
                f"selected_records.per_token_layer_k[{token}] must cover "
                "every layer"
            )
        schedule: list[int] = []
        for layer, raw_k in zip(layers, raw_schedule, strict=True):
            value = _integer(
                raw_k,
                f"selected K token {token} layer {layer.layer}",
                minimum=layer.minimum_top_k,
                maximum=layer.maximum_top_k,
            )
            schedule.append(value)
        schedules.append(schedule)

    flattened = [value for schedule in schedules for value in schedule]
    selected_sum = sum(flattened)
    selected_count = len(flattened)
    denominator = 256 * len(layers) * intermediate_size
    active_fraction = selected_sum / denominator
    observed_global = _object(
        selected.get("global"),
        "selected_records.global",
    )
    expected_global: dict[str, int | float] = {
        "sum": selected_sum,
        "count": selected_count,
        "minimum": min(flattened),
        "maximum": max(flattened),
        "mean": selected_sum / selected_count,
        "active_fraction": active_fraction,
    }
    for key in ("sum", "count", "minimum", "maximum"):
        if observed_global.get(key) != expected_global[key]:
            _fail(f"selected_records.global.{key} does not reconcile")
    for key in ("mean", "active_fraction"):
        _close(
            _number(
                observed_global.get(key),
                f"selected_records.global.{key}",
            ),
            float(expected_global[key]),
            f"selected_records.global.{key}",
        )
    active_limit = router_thresholds["maximum_mean_active_record_fraction"]
    if active_fraction > active_limit:
        _fail(
            "actual native selected-record activity exceeds the frozen "
            f"{active_limit:.1%} budget"
        )

    input_counts = [layer.input_coordinates for layer in layers]
    candidate_counts = [layer.candidate_count for layer in layers]
    accounting = [
        native_bitnet_dip_physical_accounting(
            hidden_size,
            intermediate_size,
            input_counts=input_counts,
            candidate_counts=candidate_counts,
            top_ks=schedule,
        )
        for schedule in schedules
    ]
    if any(item.get("format") != NATIVE_BITNET_DIP_TRAFFIC_FORMAT for item in accounting):
        _fail("physical accounting implementation is not v2")
    token_bytes = [
        int(item["traffic"]["complete_modelled_cold_bytes"])
        for item in accounting
    ]
    dense_bytes_per_token = int(accounting[0]["traffic"]["dense_q4_bytes"])
    token_fractions = [
        value / dense_bytes_per_token for value in token_bytes
    ]
    worst_token_index = max(
        range(len(token_fractions)),
        key=token_fractions.__getitem__,
    )
    worst_layer_token = 0
    worst_layer_index = 0
    worst_layer_fraction = -1.0
    for token, item in enumerate(accounting):
        for layer_report in item["traffic"]["layers"]:
            fraction = float(layer_report["fraction_of_dense_q4"])
            if fraction > worst_layer_fraction:
                worst_layer_fraction = fraction
                worst_layer_token = token
                worst_layer_index = int(layer_report["layer"])
    total_bytes = sum(token_bytes)
    total_dense_bytes = dense_bytes_per_token * len(token_bytes)
    global_fraction = total_bytes / total_dense_bytes
    traffic_limit = router_thresholds[
        "maximum_complete_physical_cold_traffic_fraction_of_dense_q4"
    ]
    if (
        max(token_fractions) > traffic_limit
        or global_fraction > traffic_limit
        or worst_layer_fraction > traffic_limit
    ):
        _fail(
            "recomputed native cache-line traffic exceeds the frozen "
            f"{traffic_limit:.1%} budget"
        )

    reported = _object(
        development.get("physical_cold_traffic"),
        "development physical_cold_traffic",
    )
    if reported.get("accounting_version") != NATIVE_BITNET_DIP_TRAFFIC_FORMAT:
        _fail("development traffic report is not v2 cache-line evidence")
    reported_per_token = reported.get("per_token")
    if not isinstance(reported_per_token, list) or len(reported_per_token) != 256:
        _fail("development traffic per_token must contain 256 rows")
    for token, (row, expected_bytes, expected_fraction) in enumerate(
        zip(
            reported_per_token,
            token_bytes,
            token_fractions,
            strict=True,
        )
    ):
        row = _object(row, f"physical_cold_traffic.per_token[{token}]")
        if (
            row.get("token") != token
            or row.get("scheduled_cache_line_bytes") != expected_bytes
            or row.get("dense_q4_bytes") != dense_bytes_per_token
        ):
            _fail(f"physical traffic token {token} byte counts do not reconcile")
        _close(
            _number(
                row.get("fraction_of_dense_q4"),
                f"physical traffic token {token} fraction",
            ),
            expected_fraction,
            f"physical traffic token {token} fraction",
        )

    reported_global = _object(
        reported.get("global"),
        "physical_cold_traffic.global",
    )
    if (
        reported_global.get("scheduled_cache_line_bytes") != total_bytes
        or reported_global.get("dense_q4_bytes") != total_dense_bytes
    ):
        _fail("physical_cold_traffic.global byte counts do not reconcile")
    _close(
        _number(
            reported_global.get("fraction_of_dense_q4"),
            "physical_cold_traffic.global.fraction_of_dense_q4",
        ),
        global_fraction,
        "physical_cold_traffic.global.fraction_of_dense_q4",
    )
    reported_worst_token = _object(
        reported.get("worst_token"),
        "physical_cold_traffic.worst_token",
    )
    if reported_worst_token.get("token") != worst_token_index:
        _fail("physical_cold_traffic.worst_token token does not reconcile")
    _close(
        _number(
            reported_worst_token.get("fraction_of_dense_q4"),
            "physical_cold_traffic.worst_token.fraction_of_dense_q4",
        ),
        token_fractions[worst_token_index],
        "physical_cold_traffic.worst_token.fraction_of_dense_q4",
    )
    reported_worst_layer = _object(
        reported.get("worst_layer"),
        "physical_cold_traffic.worst_layer",
    )
    if (
        reported_worst_layer.get("token") != worst_layer_token
        or reported_worst_layer.get("layer") != worst_layer_index
    ):
        _fail("physical_cold_traffic.worst_layer identity does not reconcile")
    _close(
        _number(
            reported_worst_layer.get("fraction_of_dense_q4"),
            "physical_cold_traffic.worst_layer.fraction_of_dense_q4",
        ),
        worst_layer_fraction,
        "physical_cold_traffic.worst_layer.fraction_of_dense_q4",
    )
    _require_true(
        reported.get("passes_45_percent"),
        "physical_cold_traffic.passes_45_percent",
    )

    activity_summary = {
        **expected_global,
        "denominator_records": denominator,
        "maximum_mean_active_fraction": active_limit,
        "passed": True,
    }
    traffic_summary = {
        "accounting_version": NATIVE_BITNET_DIP_TRAFFIC_FORMAT,
        "tokens": len(token_bytes),
        "scheduled_cache_line_bytes": total_bytes,
        "dense_q4_bytes": total_dense_bytes,
        "global_fraction_of_dense_q4": global_fraction,
        "minimum_token_fraction_of_dense_q4": min(token_fractions),
        "maximum_token_fraction_of_dense_q4": max(token_fractions),
        "worst_token": worst_token_index,
        "worst_layer": worst_layer_index,
        "worst_layer_token": worst_layer_token,
        "worst_layer_fraction_of_dense_q4": worst_layer_fraction,
        "maximum_allowed_fraction_of_dense_q4": traffic_limit,
        "passed": True,
    }
    layout = accounting[0]
    return activity_summary, traffic_summary, layout


def _package_bindings(
    package_path: Path,
    record_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    package_manifest_path = package_path / "manifest.json"
    package_manifest = _read_json_object(
        package_manifest_path,
        "native BitNet package manifest",
    )
    if package_manifest.get("format") != "engram-native-bitnet":
        _fail("package manifest is not an Engram native BitNet package")
    mlp = _object(package_manifest.get("mlp"), "package mlp")
    relative = mlp.get("path")
    if not isinstance(relative, str) or not relative:
        _fail("package MLP path is invalid")
    if (package_path / relative).resolve() != record_path:
        _fail("record artifact is not the package-declared MLP artifact")
    record_descriptor = _file_descriptor(record_path)
    if (
        mlp.get("sha256") != record_descriptor["sha256"]
        or mlp.get("serialized_bytes") != record_descriptor["bytes"]
    ):
        _fail("package MLP descriptor does not match the record artifact")
    files = _object(package_manifest.get("files"), "package files")
    inventory_record = _object(
        files.get(relative),
        f"package files[{relative}]",
    )
    if (
        inventory_record.get("sha256") != record_descriptor["sha256"]
        or inventory_record.get("bytes") != record_descriptor["bytes"]
    ):
        _fail("package inventory does not authenticate the record artifact")

    tokenizer_path = package_path / "tokenizer" / "tokenizer.json"
    tokenizer = _file_descriptor(tokenizer_path)
    tokenizer_relative = "tokenizer/tokenizer.json"
    tokenizer_inventory = _object(
        files.get(tokenizer_relative),
        f"package files[{tokenizer_relative}]",
    )
    if (
        tokenizer_inventory.get("sha256") != tokenizer["sha256"]
        or tokenizer_inventory.get("bytes") != tokenizer["bytes"]
    ):
        _fail("package inventory does not authenticate tokenizer.json")
    inventory_bytes = sum(
        _integer(
            _object(descriptor, f"package files[{name}]").get("bytes"),
            f"package files[{name}].bytes",
            minimum=0,
        )
        for name, descriptor in files.items()
    )
    return (
        _file_descriptor(package_manifest_path),
        record_descriptor,
        tokenizer,
        inventory_bytes,
    )


def _build_payload(
    *,
    proposal_report: Path,
    development_report: Path,
    native_parity_report: Path,
    frozen_protocol: Path,
    package: Path,
    record_artifact: Path,
    coordinate_index: Path,
    dense_library: Path,
    dip_library: Path,
) -> tuple[dict[str, Any], tuple[FrozenNativeBitNetDIPLayerPolicy, ...]]:
    paths = {
        "proposal report": proposal_report.resolve(),
        "development report": development_report.resolve(),
        "native parity report": native_parity_report.resolve(),
        "frozen protocol": frozen_protocol.resolve(),
        "package": package.resolve(),
        "record artifact": record_artifact.resolve(),
        "coordinate index": coordinate_index.resolve(),
        "dense library": dense_library.resolve(),
        "DIP library": dip_library.resolve(),
    }
    package_path = paths["package"]
    if not package_path.is_dir():
        _fail(f"native BitNet package is missing: {package_path}")
    proposal = _read_json_object(paths["proposal report"], "proposal report")
    development = _read_json_object(
        paths["development report"],
        "development report",
    )
    parity = _read_json_object(
        paths["native parity report"],
        "native parity report",
    )
    protocol = _read_json_object(paths["frozen protocol"], "frozen protocol")
    (
        package_descriptor,
        record_descriptor,
        tokenizer_descriptor,
        package_inventory_bytes,
    ) = _package_bindings(package_path, paths["record artifact"])
    index_descriptor = _file_descriptor(paths["coordinate index"])
    dense_library_descriptor = _file_descriptor(paths["dense library"])
    dip_library_descriptor = _file_descriptor(paths["DIP library"])
    proposal_descriptor = _file_descriptor(paths["proposal report"])
    development_descriptor = _file_descriptor(paths["development report"])
    parity_descriptor = _file_descriptor(paths["native parity report"])
    protocol_descriptor = _file_descriptor(paths["frozen protocol"])

    artifact = load_native_bitnet_artifact(paths["record artifact"])
    if artifact.payload_sha256 != record_descriptor["sha256"]:
        _fail("record artifact payload hash differs from its file hash")
    package_manifest = _read_json_object(
        Path(package_descriptor["path"]),
        "native BitNet package manifest",
    )
    model = _object(package_manifest.get("model"), "package model")
    dimensions = {
        "hidden_size": artifact.hidden_size,
        "intermediate_size": artifact.intermediate_size,
        "layer_count": len(artifact.layers),
    }
    expected_dimensions = {
        "hidden_size": model.get("hidden_size"),
        "intermediate_size": model.get("intermediate_size"),
        "layer_count": model.get("num_hidden_layers"),
    }
    if dimensions != expected_dimensions:
        _fail("package model dimensions differ from the record artifact")
    if dimensions["layer_count"] != 30:
        _fail("Milestone 2 policy freeze requires exactly 30 MLP layers")

    layers = _parse_layers(
        development,
        hidden_size=artifact.hidden_size,
        intermediate_size=artifact.intermediate_size,
        layer_count=len(artifact.layers),
    )
    thresholds, router_thresholds = _validate_protocol_and_development_scope(
        protocol,
        development,
        layer_count=len(layers),
    )
    execution = _validate_execution(development)
    quality = _validate_quality(development, thresholds)
    progression = _validate_development_progression(
        development,
        protocol,
    )
    proposal_classification = _validate_proposal(
        proposal,
        protocol,
        layers,
        artifact_sha256=artifact.payload_sha256,
    )
    _validate_index_policy(
        paths["coordinate index"],
        artifact.payload_sha256,
        layers,
    )

    reported_artifacts = _object(
        development.get("artifacts"),
        "development artifacts",
    )
    report_directory = paths["development report"].parent
    for label, descriptor in (
        ("package_manifest", package_descriptor),
        ("base_record_artifact", record_descriptor),
        ("coordinate_index", index_descriptor),
        ("dense_kernel_library", dense_library_descriptor),
        ("dip_kernel_library", dip_library_descriptor),
    ):
        _reported_file_matches(
            reported_artifacts.get(label),
            descriptor,
            report_directory=report_directory,
            label=label,
        )

    parity_summary = _validate_parity(
        parity,
        development,
        layer_count=len(layers),
        actual_artifacts={
            "package_manifest": package_descriptor,
            "base_record_artifact": record_descriptor,
            "coordinate_index": index_descriptor,
            "dip_kernel_library": dip_library_descriptor,
        },
        report_directory=paths["native parity report"].parent,
    )
    activity, traffic, physical = _validate_selected_schedules_and_traffic(
        development,
        layers,
        hidden_size=artifact.hidden_size,
        intermediate_size=artifact.intermediate_size,
        router_thresholds=router_thresholds,
    )
    serialization = physical["serialization"]
    layout = physical["layout"]
    if serialization["base_record_artifact_bytes"] != record_descriptor["bytes"]:
        _fail("physical accounting base bytes differ from bound artifact")
    if serialization["coordinate_index_bytes"] != index_descriptor["bytes"]:
        _fail("physical accounting index bytes differ from bound index")
    semantic_bytes = record_descriptor["bytes"] + index_descriptor["bytes"]
    if serialization["combined_serialized_bytes"] != semantic_bytes:
        _fail("combined semantic-memory storage does not reconcile")

    layer_payloads = [layer.to_dict() for layer in layers]
    final = _object(protocol.get("final_confirmation"), "final_confirmation")
    storage = {
        "base_record_artifact_bytes": record_descriptor["bytes"],
        "coordinate_index_bytes": index_descriptor["bytes"],
        "combined_semantic_mlp_bytes": semantic_bytes,
        "combined_semantic_mlp_fraction_of_dense_q4": serialization[
            "combined_fraction_of_dense_q4"
        ],
        "dense_q4_reference_bytes": serialization["dense_q4_bytes"],
        "dip_native_library_bytes": dip_library_descriptor["bytes"],
        "dense_reference_library_bytes": dense_library_descriptor["bytes"],
        "package_inventory_bytes": package_inventory_bytes,
        "complete_final_runtime_payload_bytes": (
            package_inventory_bytes
            + index_descriptor["bytes"]
            + dip_library_descriptor["bytes"]
        ),
        "development_proof_payload_bytes": (
            package_inventory_bytes
            + index_descriptor["bytes"]
            + dip_library_descriptor["bytes"]
            + dense_library_descriptor["bytes"]
        ),
        "coordinate_index_layout": {
            "cache_line_bytes": layout["cache_line_bytes"],
            "index_header_bytes": layout["index_header_bytes"],
            "index_directory_entry_bytes": layout[
                "index_directory_entry_bytes"
            ],
            "index_layer_header_bytes": layout["index_layer_header_bytes"],
            "coordinate_stride_bytes": layout["coordinate_stride_bytes"],
        },
    }
    payload: dict[str, Any] = {
        "format": NATIVE_BITNET_DIP_POLICY_FORMAT,
        "version": NATIVE_BITNET_DIP_POLICY_VERSION,
        "status": NATIVE_BITNET_DIP_POLICY_STATUS,
        "scope": "native_bitnet_milestone_2_semantic_memory_policy",
        "milestone_2_status": "blocked_pending_sealed_final_confirmation",
        "policy": {
            "layer_count": len(layers),
            "layers": layer_payloads,
            "sha256": sha256_json(layer_payloads),
        },
        "proposal_provenance": {
            **proposal_descriptor,
            **proposal_classification,
        },
        "development_evidence": {
            "report": development_descriptor,
            "native_parity_report": parity_descriptor,
            "execution": execution,
            "python_native_parity": parity_summary,
            "dataset": {
                "sha256": development["dataset"]["sha256"],
                "record_offset": development["dataset"]["record_offset"],
                "sequences": 8,
                "unique_sequences": 8,
                "predictions_per_sequence": 32,
                "prediction_positions": 256,
            },
            "quality": quality,
            "quality_thresholds": thresholds,
            "candidate_recall": progression,
            "activity": activity,
            "physical_cold_traffic": traffic,
        },
        "bindings": {
            "frozen_protocol": protocol_descriptor,
            "package_manifest": package_descriptor,
            "base_record_artifact": record_descriptor,
            "coordinate_index": index_descriptor,
            "tokenizer_json": tokenizer_descriptor,
            "dense_reference_library": dense_library_descriptor,
            "dip_native_library": dip_library_descriptor,
        },
        "storage": storage,
        "approval": {
            "passed": True,
            "authorizes": "one_sealed_final_confirmation_run",
            "does_not_claim_milestone_2_passed": True,
            "checks": {
                "all_30_layers_explicit": True,
                "proposal_demoted_to_nonapproving_float16_provenance": True,
                "live_native_bf16_cpu_8x32_development_passed": True,
                "serialized_source_bound_index_reloaded": True,
                "python_native_parity_passed": True,
                "dense_fallback_absent": True,
                "actual_sum_k_within_25_percent": True,
                "v2_per_token_and_global_cache_line_traffic_within_45_percent": (
                    True
                ),
                "artifact_hashes_reconciled": True,
                "storage_disclosed": True,
            },
        },
        "sealed_final_guard": {
            "dataset_sha256": final.get("dataset_sha256"),
            "dataset_was_opened_by_builder": False,
            "result": None,
            "policy_changes_after_opening_require_a_new_holdout": True,
        },
    }
    return payload, layers


def build_native_bitnet_dip_policy_manifest(
    *,
    proposal_report: str | Path,
    development_report: str | Path,
    native_parity_report: str | Path,
    frozen_protocol: str | Path,
    package: str | Path,
    record_artifact: str | Path,
    coordinate_index: str | Path,
    dense_library: str | Path,
    dip_library: str | Path,
    out: str | Path,
) -> Path:
    """Validate all freeze evidence and atomically write an approved manifest.

    The destination is untouched on validation failure.  In particular, a
    2x16 smoke, Python sparse path, float16 boundary replay, v1 traffic report,
    or asserted-but-unreconciled summary can never produce a manifest.
    """

    output_path = Path(out).resolve()
    payload, _ = _build_payload(
        proposal_report=Path(proposal_report),
        development_report=Path(development_report),
        native_parity_report=Path(native_parity_report),
        frozen_protocol=Path(frozen_protocol),
        package=Path(package),
        record_artifact=Path(record_artifact),
        coordinate_index=Path(coordinate_index),
        dense_library=Path(dense_library),
        dip_library=Path(dip_library),
    )
    atomic_json(output_path, payload)
    # Reconstruct from bound inputs before returning; this catches write-time
    # corruption and guarantees the public loader accepts what was emitted.
    load_native_bitnet_dip_policy_manifest(output_path)
    return output_path


def load_native_bitnet_dip_policy_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> LoadedNativeBitNetDIPPolicyManifest:
    """Reload and fully reconstruct an approved policy from its hash bindings."""

    manifest_path = Path(path).resolve()
    manifest_sha256 = sha256_file(manifest_path)
    if expected_sha256 is not None and manifest_sha256 != expected_sha256.lower():
        _fail("native BitNet DIP policy-manifest SHA-256 mismatch")
    payload = _read_json_object(manifest_path, "native BitNet DIP policy manifest")
    if (
        payload.get("format") != NATIVE_BITNET_DIP_POLICY_FORMAT
        or payload.get("version") != NATIVE_BITNET_DIP_POLICY_VERSION
        or payload.get("status") != NATIVE_BITNET_DIP_POLICY_STATUS
        or payload.get("approval", {}).get("passed") is not True
    ):
        _fail("native BitNet DIP policy manifest is not approved")
    bindings = _object(payload.get("bindings"), "policy bindings")
    provenance = _object(
        payload.get("proposal_provenance"),
        "proposal_provenance",
    )
    development = _object(
        payload.get("development_evidence"),
        "development_evidence",
    )
    development_report = _object(
        development.get("report"),
        "development_evidence.report",
    )
    parity_report = _object(
        development.get("native_parity_report"),
        "development_evidence.native_parity_report",
    )
    required = {
        "frozen_protocol": bindings.get("frozen_protocol"),
        "package_manifest": bindings.get("package_manifest"),
        "base_record_artifact": bindings.get("base_record_artifact"),
        "coordinate_index": bindings.get("coordinate_index"),
        "dense_reference_library": bindings.get("dense_reference_library"),
        "dip_native_library": bindings.get("dip_native_library"),
    }
    descriptors = {
        name: _object(value, f"bindings.{name}")
        for name, value in required.items()
    }
    package_manifest_path = Path(
        str(descriptors["package_manifest"].get("path"))
    ).resolve()
    if package_manifest_path.name != "manifest.json":
        _fail("bound package manifest path is invalid")
    reconstructed, layers = _build_payload(
        proposal_report=Path(str(provenance.get("path"))),
        development_report=Path(str(development_report.get("path"))),
        native_parity_report=Path(str(parity_report.get("path"))),
        frozen_protocol=Path(
            str(descriptors["frozen_protocol"].get("path"))
        ),
        package=package_manifest_path.parent,
        record_artifact=Path(
            str(descriptors["base_record_artifact"].get("path"))
        ),
        coordinate_index=Path(
            str(descriptors["coordinate_index"].get("path"))
        ),
        dense_library=Path(
            str(descriptors["dense_reference_library"].get("path"))
        ),
        dip_library=Path(
            str(descriptors["dip_native_library"].get("path"))
        ),
    )
    if reconstructed != payload:
        _fail(
            "native BitNet DIP policy manifest differs from reconstructed "
            "bound evidence"
        )
    return LoadedNativeBitNetDIPPolicyManifest(
        path=manifest_path,
        manifest_sha256=manifest_sha256,
        layers=layers,
        payload=payload,
    )


__all__ = [
    "FrozenNativeBitNetDIPLayerPolicy",
    "LoadedNativeBitNetDIPPolicyManifest",
    "NATIVE_BITNET_DIP_POLICY_FORMAT",
    "NATIVE_BITNET_DIP_POLICY_STATUS",
    "NATIVE_BITNET_DIP_POLICY_VERSION",
    "NATIVE_BITNET_DIP_TRAFFIC_FORMAT",
    "NativeBitNetDIPPolicyManifestError",
    "build_native_bitnet_dip_policy_manifest",
    "load_native_bitnet_dip_policy_manifest",
]
