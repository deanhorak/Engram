"""Explicit go/no-go criteria for trained-teacher intervention reports."""

from __future__ import annotations

import copy
import math
from typing import Any

MLP_QUALITY_THRESHOLDS = {
    "maximum_teacher_student_kl": 0.05,
    "minimum_teacher_top1_agreement": 0.90,
    "maximum_nll_delta": 0.05,
    "maximum_final_hidden_relative_l2": 0.10,
}
MINIMUM_ROUTED_CANDIDATE_RECALL = 0.95
IDENTITY_TOLERANCE = 1e-6
MINIMUM_EVALUATION_SEQUENCES = 8
MINIMUM_UNIQUE_EVALUATION_SEQUENCES = 8
MINIMUM_NEXT_TOKEN_POSITIONS = 256
GATED_VARIANTS = {"identity", "oracle", "rank16", "overlap", "dip"}
MAGNITUDE_REFERENCE_CAVEAT = (
    "The magnitude oracle uses all neuron activations to choose top-K and is not a "
    "realizable candidate-selection algorithm. Magnitude ranking is not the "
    "mathematically optimal K-subset when vector contributions can cancel, so this is "
    "a full-information reference rather than a theoretical quality ceiling."
)
_COMPOSITE_MATCH_FIELDS = (
    "schema_version",
    "experiment",
    "source_model_hash",
    "num_hidden_layers",
    "intermediate_size",
    "dataset_hash",
    "evaluation_role",
    "configuration_selection",
    "selected_layers",
    "layer_mode",
    "baseline",
)
_CALIBRATION_CONSENSUS_FIELDS = (
    "dataset_hash",
    "dataset_files_differ",
    "trace_path",
    "records_per_layer_limit",
    "regularization",
    "rank",
    "separation_method",
    "calibration_sequence_count",
    "calibration_unique_sequence_count",
    "evaluation_sequence_count",
    "evaluation_unique_sequence_count",
    "overlapping_sequence_count",
    "record_level_disjoint",
    "held_out_from_evaluation",
)


def _check(
    name: str, actual: float, threshold: float, comparison: str
) -> dict[str, Any]:
    if not math.isfinite(actual):
        raise ValueError(f"metric {name!r} must be finite")
    if comparison == "maximum":
        passed = actual <= threshold
    elif comparison == "minimum":
        passed = actual >= threshold
    else:
        raise ValueError(f"unsupported gate comparison {comparison!r}")
    return {
        "metric": name,
        "actual": actual,
        "comparison": comparison,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _mean_metric(
    container: dict[str, Any],
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        value = container[name]["mean"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"metric {name!r} must contain a mean") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {name!r} mean must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"metric {name!r} mean must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"metric {name!r} mean must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"metric {name!r} mean must be at most {maximum}")
    return result


def _ensure_stat_count(metric: Any, name: str, expected: int) -> None:
    if not isinstance(metric, dict):
        raise ValueError(f"metric {name!r} must be a statistics object")
    count = metric.setdefault("count", expected)
    if isinstance(count, bool) or not isinstance(count, int) or count != expected:
        raise ValueError(f"metric {name!r} count must equal {expected}")


def evaluate_mlp_arm_gate(arm: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one intervention arm against declared quality prerequisites."""

    variant = arm.get("variant")
    if variant not in GATED_VARIANTS:
        raise ValueError(f"unsupported gated intervention variant {variant!r}")
    quality = arm["quality"]
    kl = _mean_metric(quality, "teacher_student_kl", minimum=0.0)
    top1 = _mean_metric(quality, "teacher_top1_agreement", minimum=0.0, maximum=1.0)
    nll_delta = _mean_metric(quality, "nll_delta")
    hidden_relative_l2 = _mean_metric(quality, "final_hidden_relative_l2", minimum=0.0)
    if variant == "identity":
        checks = [
            _check(
                "teacher_student_kl",
                kl,
                IDENTITY_TOLERANCE,
                "maximum",
            ),
            _check(
                "final_hidden_relative_l2",
                hidden_relative_l2,
                IDENTITY_TOLERANCE,
                "maximum",
            ),
            _check(
                "teacher_top1_agreement",
                top1,
                1.0 - IDENTITY_TOLERANCE,
                "minimum",
            ),
            _check(
                "absolute_nll_delta",
                abs(nll_delta),
                IDENTITY_TOLERANCE,
                "maximum",
            ),
        ]
        gate_type = "instrumentation_sanity"
    else:
        checks = [
            _check(
                "teacher_student_kl",
                kl,
                MLP_QUALITY_THRESHOLDS["maximum_teacher_student_kl"],
                "maximum",
            ),
            _check(
                "teacher_top1_agreement",
                top1,
                MLP_QUALITY_THRESHOLDS["minimum_teacher_top1_agreement"],
                "minimum",
            ),
            _check(
                "nll_delta",
                nll_delta,
                MLP_QUALITY_THRESHOLDS["maximum_nll_delta"],
                "maximum",
            ),
            _check(
                "final_hidden_relative_l2",
                hidden_relative_l2,
                MLP_QUALITY_THRESHOLDS["maximum_final_hidden_relative_l2"],
                "maximum",
            ),
        ]
        gate_type = (
            "full_information_magnitude_reference"
            if variant == "oracle"
            else "routed_quality"
        )
        if variant not in {"oracle", "identity"}:
            recall = _mean_metric(
                arm.get("local_mlp", {}),
                "candidate_recall",
                minimum=0.0,
                maximum=1.0,
            )
            checks.append(
                _check(
                    "candidate_recall",
                    recall,
                    MINIMUM_ROUTED_CANDIDATE_RECALL,
                    "minimum",
                )
            )
    failed = [item["metric"] for item in checks if not item["passed"]]
    return {
        "type": gate_type,
        "passed": not failed,
        "checks": checks,
        "failed_metrics": failed,
    }


def apply_mlp_intervention_gates(report: dict[str, Any]) -> dict[str, Any]:
    """Return a report copy annotated with arm-level and development gates."""

    result = copy.deepcopy(report)
    if result.get("schema_version") != 1:
        raise ValueError("unsupported or missing intervention report schema_version")
    if result.get("experiment") != "trained_teacher_mlp_intervention":
        raise ValueError("report is not a trained-teacher MLP intervention experiment")
    baseline = result.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("intervention report must contain baseline evidence counts")
    sequences = baseline.get("sequences")
    unique_sequences = baseline.get("unique_sequences")
    next_token_positions = baseline.get("next_token_positions")
    if (
        isinstance(sequences, bool)
        or not isinstance(sequences, int)
        or sequences <= 0
        or isinstance(unique_sequences, bool)
        or not isinstance(unique_sequences, int)
        or unique_sequences <= 0
        or unique_sequences > sequences
        or isinstance(next_token_positions, bool)
        or not isinstance(next_token_positions, int)
        or next_token_positions <= 0
    ):
        raise ValueError("baseline evidence counts must be positive integers")
    evidence_size_verified = (
        sequences >= MINIMUM_EVALUATION_SEQUENCES
        and unique_sequences >= MINIMUM_UNIQUE_EVALUATION_SEQUENCES
        and next_token_positions >= MINIMUM_NEXT_TOKEN_POSITIONS
    )
    evaluation_role = result.get("evaluation_role", "development")
    if evaluation_role not in {"development", "confirmation"}:
        raise ValueError("evaluation_role must be 'development' or 'confirmation'")
    selection = result.get("configuration_selection")
    configuration_selection_verified: bool | None = None
    if selection is not None:
        if not isinstance(selection, dict):
            raise ValueError("configuration_selection must be an object")
        configuration_selection_verified = bool(
            selection.get("separation_method") == "exact_token_sequence_hashes"
            and selection.get("held_out_from_configuration_selection") is True
            and selection.get("overlapping_sequence_count") == 0
            and selection.get("evaluation_sequence_count") == sequences
            and selection.get("evaluation_unique_sequence_count") == unique_sequences
        )
        if not configuration_selection_verified:
            raise ValueError(
                "configuration_selection must prove exact sequence separation"
            )
    if (
        evaluation_role == "confirmation"
        and configuration_selection_verified is not True
    ):
        raise ValueError(
            "confirmation evaluation requires verified configuration-selection separation"
        )
    result["evaluation_role"] = evaluation_role
    result["configuration_selection"] = copy.deepcopy(selection)
    input_token_positions = baseline.get("input_token_positions")
    if input_token_positions is not None and (
        isinstance(input_token_positions, bool)
        or not isinstance(input_token_positions, int)
        or input_token_positions <= 0
    ):
        raise ValueError("baseline input_token_positions must be a positive integer")
    arms = result.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("intervention report must contain at least one arm")
    required_layers = result.get("num_hidden_layers")
    if (
        isinstance(required_layers, bool)
        or not isinstance(required_layers, int)
        or required_layers <= 0
    ):
        raise ValueError(
            "intervention report must declare a positive num_hidden_layers"
        )
    arm_names: set[str] = set()
    intermediate_size = result.get("intermediate_size")
    if (
        isinstance(intermediate_size, bool)
        or not isinstance(intermediate_size, int)
        or intermediate_size <= 0
    ):
        raise ValueError(
            "intervention report must declare a positive intermediate_size"
        )
    identity_arms = []
    oracle_arms = []
    routed_arms = []
    for arm in arms:
        name = arm.get("name")
        if not isinstance(name, str) or not name or name in arm_names:
            raise ValueError("intervention arm names must be non-empty and unique")
        arm_names.add(name)
        variant = arm.get("variant")
        if variant not in GATED_VARIANTS:
            raise ValueError(f"unsupported gated intervention variant {variant!r}")
        scope = arm.get("scope")
        layer_indices = arm.get("layer_indices")
        if scope not in {"all", "individual"}:
            raise ValueError(f"arm {name!r} has an invalid scope")
        if (
            not isinstance(layer_indices, list)
            or not layer_indices
            or any(not isinstance(value, int) for value in layer_indices)
            or len(set(layer_indices)) != len(layer_indices)
            or any(value < 0 or value >= required_layers for value in layer_indices)
        ):
            raise ValueError(f"arm {name!r} has invalid layer indices")
        if scope == "individual" and len(layer_indices) != 1:
            raise ValueError(f"individual arm {name!r} must contain exactly one layer")
        local_mlp = arm.get("local_mlp")
        if not isinstance(local_mlp, dict):
            raise ValueError(f"arm {name!r} must contain local_mlp statistics")
        local_tokens = local_mlp.get("tokens")
        if (
            isinstance(local_tokens, bool)
            or not isinstance(local_tokens, int)
            or local_tokens <= 0
            or local_tokens % len(layer_indices)
        ):
            raise ValueError(
                f"arm {name!r} local token count must divide evenly across its layers"
            )
        arm_input_token_positions = local_tokens // len(layer_indices)
        if input_token_positions is None:
            input_token_positions = arm_input_token_positions
            baseline["input_token_positions"] = input_token_positions
        elif arm_input_token_positions != input_token_positions:
            raise ValueError(f"arm {name!r} uses a different input-token population")
        for metric_name, metric in local_mlp.items():
            if metric_name != "tokens":
                _ensure_stat_count(metric, f"local_mlp.{metric_name}", local_tokens)
        for layer_summary in arm.get("local_mlp_by_layer", []):
            layer_tokens = layer_summary.get("tokens")
            if layer_tokens != input_token_positions:
                raise ValueError(
                    f"arm {name!r} has an inconsistent per-layer token population"
                )
            for metric_name, metric in layer_summary.items():
                if metric_name not in {"layer", "tokens"}:
                    _ensure_stat_count(
                        metric,
                        f"local_mlp_by_layer.{metric_name}",
                        input_token_positions,
                    )
        if variant != "identity":
            top_k = arm.get("top_k")
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
                raise ValueError(f"arm {name!r} must declare a positive top_k")
            if top_k > intermediate_size:
                raise ValueError(f"arm {name!r} top_k exceeds intermediate_size")
        if variant not in {"identity", "oracle"}:
            candidate_count = arm.get("candidate_count")
            if (
                isinstance(candidate_count, bool)
                or not isinstance(candidate_count, int)
                or candidate_count < arm["top_k"]
            ):
                raise ValueError(
                    f"routed arm {name!r} must declare candidate_count >= top_k"
                )
            if candidate_count > intermediate_size:
                raise ValueError(
                    f"routed arm {name!r} candidate_count exceeds intermediate_size"
                )
        if variant == "dip":
            input_fraction = arm.get("input_fraction")
            if (
                isinstance(input_fraction, bool)
                or not isinstance(input_fraction, (int, float))
                or not math.isfinite(float(input_fraction))
                or not 0.0 < float(input_fraction) <= 1.0
            ):
                raise ValueError(
                    f"DIP arm {name!r} must declare input_fraction in (0, 1]"
                )
        quality = arm.get("quality", {})
        if (
            "final_hidden_relative_l2" not in quality
            and "final_residual_relative_l2" in quality
        ):
            quality["final_hidden_relative_l2"] = quality.pop(
                "final_residual_relative_l2"
            )
        if "final_hidden_cosine" not in quality and "final_residual_cosine" in quality:
            quality["final_hidden_cosine"] = quality.pop("final_residual_cosine")
        for metric_name, metric in quality.items():
            if isinstance(metric, dict):
                expected = (
                    input_token_positions
                    if metric_name.startswith("final_hidden_")
                    else next_token_positions
                )
                _ensure_stat_count(metric, f"quality.{metric_name}", expected)
        arm["gate"] = evaluate_mlp_arm_gate(arm)
        if arm["variant"] == "identity":
            identity_arms.append(arm)
        elif arm["variant"] == "oracle":
            oracle_arms.append(arm)
        else:
            routed_arms.append(arm)

    def progression_eligible(arm: dict[str, Any]) -> bool:
        return (
            arm.get("scope") == "all"
            and len(arm.get("layer_indices", [])) == required_layers
        )

    eligible_identities = [arm for arm in identity_arms if progression_eligible(arm)]
    eligible_oracles = [arm for arm in oracle_arms if progression_eligible(arm)]
    eligible_routers = [arm for arm in routed_arms if progression_eligible(arm)]
    eligible_learned_routers = [
        arm for arm in eligible_routers if arm.get("variant") in {"rank16", "overlap"}
    ]
    identity_passed = any(arm["gate"]["passed"] for arm in eligible_identities)
    passing_oracles = [arm["name"] for arm in eligible_oracles if arm["gate"]["passed"]]
    passing_router_arms = [arm for arm in eligible_routers if arm["gate"]["passed"]]
    matched_passing_router_arms = []
    for router_arm in passing_router_arms:
        matched_reference = any(
            oracle_arm.get("top_k") == router_arm.get("top_k")
            for oracle_arm in eligible_oracles
        )
        if matched_reference:
            matched_passing_router_arms.append(router_arm)
    data_separation: bool | None = None
    if eligible_learned_routers:
        calibration = result.get("calibration") or {}
        data_separation = bool(
            calibration.get("separation_method") == "exact_token_sequence_hashes"
            and calibration.get("record_level_disjoint") is True
            and calibration.get("overlapping_sequence_count") == 0
            and calibration.get("evaluation_sequence_count") == sequences
            and calibration.get("evaluation_unique_sequence_count") == unique_sequences
        )
    elif eligible_routers:
        # DIP has no fitted predictor and therefore no train/evaluation split to leak.
        data_separation = True
    passing_predictor_free_arms = [
        arm for arm in matched_passing_router_arms if arm.get("variant") == "dip"
    ]
    matched_passing_learned_arms = [
        arm
        for arm in matched_passing_router_arms
        if arm.get("variant") in {"rank16", "overlap"}
    ]
    passing_learned_arms = (
        matched_passing_learned_arms if data_separation is True else []
    )
    accepted_passing_arms = [*passing_predictor_free_arms, *passing_learned_arms]
    passing_routers = [arm["name"] for arm in accepted_passing_arms]
    if not eligible_oracles and not eligible_routers:
        decision = "insufficient_all_layer_evidence"
        targets_met: bool | None = None
    elif not evidence_size_verified:
        decision = "insufficient_evaluation_corpus"
        targets_met = None
    elif not identity_passed:
        decision = "insufficient_identity_sanity"
        targets_met = None
    elif eligible_routers:
        if passing_routers:
            decision = (
                "eligible_for_selector_serialization"
                if passing_predictor_free_arms and not passing_learned_arms
                else "eligible_for_router_serialization"
            )
            targets_met = True
        elif eligible_learned_routers and data_separation is not True:
            decision = "invalid_data_separation"
            targets_met = None
        elif passing_router_arms:
            decision = "insufficient_matched_magnitude_reference"
            targets_met = None
        else:
            decision = "stop_before_serialization"
            targets_met = False
    elif eligible_oracles:
        decision = (
            "router_experiments_justified"
            if passing_oracles
            else "increase_active_budget_or_stop"
        )
        targets_met = None
    else:
        decision = "insufficient_non_identity_arms"
        targets_met = None
    result["gate_definition"] = {
        **MLP_QUALITY_THRESHOLDS,
        "minimum_routed_candidate_recall": MINIMUM_ROUTED_CANDIDATE_RECALL,
        "minimum_evaluation_sequences": MINIMUM_EVALUATION_SEQUENCES,
        "minimum_unique_evaluation_sequences": MINIMUM_UNIQUE_EVALUATION_SEQUENCES,
        "minimum_next_token_positions": MINIMUM_NEXT_TOKEN_POSITIONS,
        "matched_magnitude_reference_policy": (
            "a routed arm requires an all-layer magnitude-reference arm at the same top_k; "
            "the reference need not pass when the routed arm itself passes causal quality"
        ),
        "scope": (
            "logit and NLL means use held-out next-token positions; final-hidden, local MLP, "
            "and recall means use held-out input-token states; all-layer arms are required "
            "for progression"
        ),
        "confirmation_policy": (
            "confirmation reports must prove exact token-sequence separation from the corpus "
            "used to select their configuration"
        ),
    }
    result["gate_summary"] = {
        "passing_oracle_arms": passing_oracles,
        "passing_routed_arms": passing_routers,
        "passing_predictor_free_arms": [
            arm["name"] for arm in passing_predictor_free_arms
        ],
        "passing_learned_router_arms": [arm["name"] for arm in passing_learned_arms],
        "instrumentation_sanity": identity_passed if eligible_identities else None,
        "evidence_size_verified": evidence_size_verified,
        "evaluation_role": evaluation_role,
        "configuration_selection_separation_verified": (
            configuration_selection_verified
        ),
        "data_separation_verified": data_separation,
        "learned_router_data_separation_verified": (
            data_separation if eligible_learned_routers else None
        ),
        "oracle_viability": (
            bool(passing_oracles)
            if eligible_oracles and identity_passed and evidence_size_verified
            else None
        ),
        "routing_viability": (
            True
            if passing_routers and identity_passed and evidence_size_verified
            else (
                None
                if (
                    eligible_learned_routers
                    and data_separation is not True
                    and identity_passed
                    and evidence_size_verified
                )
                else (
                    False
                    if eligible_routers and identity_passed and evidence_size_verified
                    else None
                )
            )
        ),
        "development_decision": decision,
    }
    result["quality_targets_met"] = targets_met
    result["oracle_caveat"] = MAGNITUDE_REFERENCE_CAVEAT
    return result


def combine_mlp_intervention_reports(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compose staged arms only when their experimental provenance agrees exactly."""

    if len(reports) < 2:
        raise ValueError("composite gating requires at least two intervention reports")
    normalized = [apply_mlp_intervention_gates(report) for report in reports]
    reference = normalized[0]
    for report in normalized[1:]:
        for field in _COMPOSITE_MATCH_FIELDS:
            if report.get(field) != reference.get(field):
                raise ValueError(
                    f"cannot compose intervention reports with different {field}"
                )

    routed_calibrations = []
    for report in normalized:
        if any(arm["variant"] in {"rank16", "overlap"} for arm in report["arms"]):
            calibration = report.get("calibration")
            if not isinstance(calibration, dict):
                raise ValueError(
                    "routed source report is missing calibration provenance"
                )
            routed_calibrations.append(calibration)
    calibration = None
    if routed_calibrations:
        first = routed_calibrations[0]
        for candidate in routed_calibrations[1:]:
            for field in _CALIBRATION_CONSENSUS_FIELDS:
                if candidate.get(field) != first.get(field):
                    raise ValueError(
                        "cannot compose routed reports with different calibration "
                        f"{field}"
                    )
        calibration = {
            field: copy.deepcopy(first.get(field))
            for field in _CALIBRATION_CONSENSUS_FIELDS
        }
        calibration["router_configurations"] = copy.deepcopy(routed_calibrations)

    combined_arms: list[dict[str, Any]] = []
    arms_by_name: dict[str, dict[str, Any]] = {}
    for report in normalized:
        for source_arm in report["arms"]:
            arm = copy.deepcopy(source_arm)
            arm.pop("gate", None)
            if arm.get("router_training") is None:
                arm.pop("router_training", None)
            previous = arms_by_name.get(arm["name"])
            if previous is not None:
                if previous != arm:
                    raise ValueError(
                        f"duplicate intervention arm {arm['name']!r} differs across reports"
                    )
                continue
            arms_by_name[arm["name"]] = arm
            combined_arms.append(arm)

    combined = copy.deepcopy(reference)
    for field in ("gate_definition", "gate_summary", "quality_targets_met"):
        combined.pop(field, None)
    combined.update(
        {
            "status": "composite_local_teacher_intervention_measurement",
            "arms": combined_arms,
            "variants": list(dict.fromkeys(arm["variant"] for arm in combined_arms)),
            "top_ks": sorted(
                {
                    int(arm["top_k"])
                    for arm in combined_arms
                    if arm.get("top_k") is not None
                }
            ),
            "candidate_counts": sorted(
                {
                    int(arm["candidate_count"])
                    for arm in combined_arms
                    if arm.get("candidate_count") is not None
                }
            ),
            "calibration": calibration,
            "composite_source_count": len(normalized),
        }
    )
    return apply_mlp_intervention_gates(combined)


__all__ = [
    "IDENTITY_TOLERANCE",
    "GATED_VARIANTS",
    "MAGNITUDE_REFERENCE_CAVEAT",
    "MINIMUM_EVALUATION_SEQUENCES",
    "MINIMUM_UNIQUE_EVALUATION_SEQUENCES",
    "MINIMUM_NEXT_TOKEN_POSITIONS",
    "MINIMUM_ROUTED_CANDIDATE_RECALL",
    "MLP_QUALITY_THRESHOLDS",
    "apply_mlp_intervention_gates",
    "combine_mlp_intervention_reports",
    "evaluate_mlp_arm_gate",
]
