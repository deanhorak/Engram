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
MAXIMUM_PROJECTED_MLP_TRAFFIC_FRACTION = 0.45
IDENTITY_TOLERANCE = 1e-6
MINIMUM_EVALUATION_SEQUENCES = 8
MINIMUM_UNIQUE_EVALUATION_SEQUENCES = 8
MINIMUM_NEXT_TOKEN_POSITIONS = 256
FULL_CONVERTED_WIDTH_SELECTION_SCOPE = "full_converted_width"
ROUTED_CANDIDATE_SELECTION_SCOPE = "routed_candidates"
GATED_VARIANTS = {
    "identity",
    "oracle",
    "rank16",
    "overlap",
    "dip",
    "dip_paq",
    "shared_basis",
}
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


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _candidate_recall_policy(
    arm: dict[str, Any],
    *,
    intermediate_size: int | None = None,
) -> dict[str, Any]:
    """Authenticate whether oracle candidate recall applies to one deployable arm."""

    variant = arm.get("variant")
    scope = arm.get("selection_scope")
    if scope is None:
        if variant == "shared_basis":
            raise ValueError(
                "shared_basis arm must authenticate selection_scope as "
                "full_converted_width"
            )
        return {
            "applicable": True,
            "status": "required",
            "selection_scope": ROUTED_CANDIDATE_SELECTION_SCOPE,
            "selection_scope_authenticated": False,
            "reason": "arm_selects_from_a_candidate_shortlist",
        }
    if not isinstance(scope, dict):
        raise ValueError("selection_scope must be an authenticated object")
    mode = scope.get("mode")
    if mode == ROUTED_CANDIDATE_SELECTION_SCOPE:
        if variant == "shared_basis":
            raise ValueError(
                "shared_basis arm must use full_converted_width selection"
            )
        if scope.get("candidate_shortlist") is not True:
            raise ValueError(
                "routed_candidates selection_scope must declare "
                "candidate_shortlist=true"
            )
        return {
            "applicable": True,
            "status": "required",
            "selection_scope": ROUTED_CANDIDATE_SELECTION_SCOPE,
            "selection_scope_authenticated": True,
            "reason": "arm_selects_from_a_candidate_shortlist",
        }
    if mode != FULL_CONVERTED_WIDTH_SELECTION_SCOPE:
        raise ValueError(
            "selection_scope mode must be 'routed_candidates' or "
            "'full_converted_width'"
        )
    if variant != "shared_basis":
        raise ValueError(
            "full_converted_width selection_scope is only valid for shared_basis arms"
        )
    converted_width = scope.get("converted_width")
    scored_width = scope.get("scored_width")
    if (
        isinstance(converted_width, bool)
        or not isinstance(converted_width, int)
        or converted_width <= 0
        or isinstance(scored_width, bool)
        or not isinstance(scored_width, int)
        or scored_width != converted_width
    ):
        raise ValueError(
            "full_converted_width selection_scope must declare equal positive "
            "converted_width and scored_width"
        )
    if intermediate_size is not None and converted_width != intermediate_size:
        raise ValueError(
            "full_converted_width selection_scope must equal intermediate_size"
        )
    if scope.get("candidate_shortlist") is not False:
        raise ValueError(
            "full_converted_width selection_scope must declare "
            "candidate_shortlist=false"
        )
    authentication = scope.get("authentication")
    if not isinstance(authentication, dict):
        raise ValueError(
            "full_converted_width selection_scope requires artifact authentication"
        )
    if (
        authentication.get("source")
        != "strict_reloaded_artifact_header_and_manifest"
        or authentication.get("verified") is not True
        or not _valid_sha256(authentication.get("sha256"))
    ):
        raise ValueError(
            "full_converted_width selection_scope requires verified strict-reload "
            "artifact headers, an authenticated manifest, and a SHA-256"
        )
    return {
        "applicable": False,
        "status": "not_applicable",
        "selection_scope": FULL_CONVERTED_WIDTH_SELECTION_SCOPE,
        "selection_scope_authenticated": True,
        "reason": "all_converted_records_are_scored_without_a_candidate_shortlist",
    }


def evaluate_mlp_arm_gate(
    arm: dict[str, Any],
    *,
    intermediate_size: int | None = None,
) -> dict[str, Any]:
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
        if variant == "oracle":
            gate_type = "full_information_magnitude_reference"
        elif variant == "shared_basis":
            gate_type = "full_width_sparse_quality"
        else:
            gate_type = "routed_quality"
        candidate_recall_policy = None
        if variant not in {"oracle", "identity"}:
            candidate_recall_policy = _candidate_recall_policy(
                arm,
                intermediate_size=intermediate_size,
            )
            local_mlp = arm.get("local_mlp", {})
            reported_applicability = arm.get("candidate_recall_applicable")
            if (
                reported_applicability is not None
                and reported_applicability
                is not candidate_recall_policy["applicable"]
            ):
                raise ValueError(
                    "candidate_recall_applicable contradicts authenticated "
                    "selection_scope"
                )
            if candidate_recall_policy["applicable"]:
                recall = _mean_metric(
                    local_mlp,
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
            else:
                if "candidate_recall" in local_mlp:
                    raise ValueError(
                        "candidate_recall must be omitted when authenticated "
                        "selection_scope is full_converted_width"
                    )
                checks.append(
                    {
                        "metric": "candidate_recall",
                        "actual": None,
                        "comparison": "not_applicable",
                        "threshold": None,
                        "passed": True,
                        "applicable": False,
                        "reason": candidate_recall_policy["reason"],
                    }
                )
        if variant in {"dip_paq", "shared_basis"}:
            projected = arm.get("projected_accounting", {})
            traffic = projected.get("cold_fraction_of_dense_q4")
            if isinstance(traffic, bool) or not isinstance(traffic, (int, float)):
                raise ValueError(
                    f"{variant} arm must report cold_fraction_of_dense_q4"
                )
            if not math.isfinite(float(traffic)) or float(traffic) < 0.0:
                raise ValueError(
                    "cold_fraction_of_dense_q4 must be finite and non-negative"
                )
            checks.append(
                _check(
                    "cold_fraction_of_dense_q4",
                    float(traffic),
                    MAXIMUM_PROJECTED_MLP_TRAFFIC_FRACTION,
                    "maximum",
                )
            )
    failed = [item["metric"] for item in checks if not item["passed"]]
    result = {
        "type": gate_type,
        "passed": not failed,
        "checks": checks,
        "failed_metrics": failed,
    }
    if variant not in {"oracle", "identity"}:
        result["candidate_recall"] = candidate_recall_policy
    return result


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
    full_width_arms = []
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
        candidate_recall_policy = None
        if variant not in {"identity", "oracle"}:
            candidate_recall_policy = _candidate_recall_policy(
                arm,
                intermediate_size=intermediate_size,
            )
            candidate_count = arm.get("candidate_count")
            if candidate_recall_policy["applicable"]:
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
            elif candidate_count is not None:
                raise ValueError(
                    f"full-width arm {name!r} must omit candidate_count because it "
                    "does not use a candidate shortlist"
                )
            reported_applicability = arm.get("candidate_recall_applicable")
            if reported_applicability is not None and (
                not isinstance(reported_applicability, bool)
                or reported_applicability
                is not candidate_recall_policy["applicable"]
            ):
                raise ValueError(
                    "candidate_recall_applicable contradicts authenticated "
                    "selection_scope"
                )
            arm["candidate_recall_applicable"] = candidate_recall_policy["applicable"]
        if variant in {"dip", "dip_paq"}:
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
        arm["gate"] = evaluate_mlp_arm_gate(
            arm,
            intermediate_size=intermediate_size,
        )
        if arm["variant"] == "identity":
            identity_arms.append(arm)
        elif arm["variant"] == "oracle":
            oracle_arms.append(arm)
        elif candidate_recall_policy is not None and not candidate_recall_policy[
            "applicable"
        ]:
            full_width_arms.append(arm)
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
    eligible_full_width_arms = [
        arm for arm in full_width_arms if progression_eligible(arm)
    ]
    eligible_learned_routers = [
        arm for arm in eligible_routers if arm.get("variant") in {"rank16", "overlap"}
    ]
    eligible_fitted_arms = [*eligible_learned_routers, *eligible_full_width_arms]
    identity_passed = any(arm["gate"]["passed"] for arm in eligible_identities)
    passing_oracles = [arm["name"] for arm in eligible_oracles if arm["gate"]["passed"]]
    passing_router_arms = [arm for arm in eligible_routers if arm["gate"]["passed"]]
    passing_full_width_gate_arms = [
        arm for arm in eligible_full_width_arms if arm["gate"]["passed"]
    ]
    matched_passing_router_arms = []
    for router_arm in passing_router_arms:
        matched_reference = any(
            oracle_arm.get("top_k") == router_arm.get("top_k")
            for oracle_arm in eligible_oracles
        )
        if matched_reference:
            matched_passing_router_arms.append(router_arm)
    matched_passing_full_width_arms = []
    for full_width_arm in passing_full_width_gate_arms:
        matched_reference = any(
            oracle_arm.get("top_k") == full_width_arm.get("top_k")
            for oracle_arm in eligible_oracles
        )
        if matched_reference:
            matched_passing_full_width_arms.append(full_width_arm)
    data_separation: bool | None = None
    if eligible_fitted_arms:
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
        arm
        for arm in matched_passing_router_arms
        if arm.get("variant") in {"dip", "dip_paq"}
    ]
    matched_passing_learned_arms = [
        arm
        for arm in matched_passing_router_arms
        if arm.get("variant") in {"rank16", "overlap"}
    ]
    passing_learned_arms = (
        matched_passing_learned_arms if data_separation is True else []
    )
    passing_full_width_arms = (
        matched_passing_full_width_arms if data_separation is True else []
    )
    accepted_passing_arms = [
        *passing_predictor_free_arms,
        *passing_learned_arms,
        *passing_full_width_arms,
    ]
    passing_routers = [
        arm["name"] for arm in [*passing_predictor_free_arms, *passing_learned_arms]
    ]
    if not eligible_oracles and not eligible_routers and not eligible_full_width_arms:
        decision = "insufficient_all_layer_evidence"
        targets_met: bool | None = None
    elif not evidence_size_verified:
        decision = "insufficient_evaluation_corpus"
        targets_met = None
    elif not identity_passed:
        decision = "insufficient_identity_sanity"
        targets_met = None
    elif eligible_routers or eligible_full_width_arms:
        if accepted_passing_arms:
            if passing_full_width_arms and not (
                passing_predictor_free_arms or passing_learned_arms
            ):
                decision = "eligible_for_full_width_artifact_confirmation"
            elif passing_predictor_free_arms and not (
                passing_learned_arms or passing_full_width_arms
            ):
                decision = "eligible_for_selector_serialization"
            elif passing_learned_arms and not (
                passing_predictor_free_arms or passing_full_width_arms
            ):
                decision = "eligible_for_router_serialization"
            else:
                decision = "eligible_for_deployable_artifact_confirmation"
            targets_met = True
        elif eligible_fitted_arms and data_separation is not True:
            decision = "invalid_data_separation"
            targets_met = None
        elif passing_router_arms or passing_full_width_gate_arms:
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
        "maximum_projected_mlp_traffic_fraction": (
            MAXIMUM_PROJECTED_MLP_TRAFFIC_FRACTION
        ),
        "minimum_evaluation_sequences": MINIMUM_EVALUATION_SEQUENCES,
        "minimum_unique_evaluation_sequences": MINIMUM_UNIQUE_EVALUATION_SEQUENCES,
        "minimum_next_token_positions": MINIMUM_NEXT_TOKEN_POSITIONS,
        "candidate_recall_applicability_policy": (
            "candidate recall is mandatory for routed-candidate arms and is "
            "not applicable only for a shared_basis arm whose strict-reloaded "
            "artifact authenticates full_converted_width scoring with no "
            "candidate shortlist"
        ),
        "matched_magnitude_reference_policy": (
            "a deployable arm requires an all-layer magnitude-reference arm at the same "
            "top_k; the reference need not pass when the deployable arm itself passes "
            "causal quality"
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
        "passing_full_width_arms": [
            arm["name"] for arm in passing_full_width_arms
        ],
        "passing_deployable_arms": [
            arm["name"] for arm in accepted_passing_arms
        ],
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
        "full_width_data_separation_verified": (
            data_separation if eligible_full_width_arms else None
        ),
        "oracle_viability": (
            bool(passing_oracles)
            if eligible_oracles and identity_passed and evidence_size_verified
            else None
        ),
        "routing_viability": (
            True
            if (passing_predictor_free_arms or passing_learned_arms)
            and identity_passed
            and evidence_size_verified
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
        "full_width_viability": (
            True
            if passing_full_width_arms
            and identity_passed
            and evidence_size_verified
            else (
                False
                if eligible_full_width_arms
                and data_separation is True
                and identity_passed
                and evidence_size_verified
                else None
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
        if any(
            arm["variant"] in {"rank16", "overlap", "shared_basis"}
            for arm in report["arms"]
        ):
            calibration = report.get("calibration")
            if not isinstance(calibration, dict):
                raise ValueError(
                    "fitted source report is missing calibration provenance"
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
    "FULL_CONVERTED_WIDTH_SELECTION_SCOPE",
    "IDENTITY_TOLERANCE",
    "GATED_VARIANTS",
    "MAGNITUDE_REFERENCE_CAVEAT",
    "MINIMUM_EVALUATION_SEQUENCES",
    "MINIMUM_UNIQUE_EVALUATION_SEQUENCES",
    "MINIMUM_NEXT_TOKEN_POSITIONS",
    "MINIMUM_ROUTED_CANDIDATE_RECALL",
    "MAXIMUM_PROJECTED_MLP_TRAFFIC_FRACTION",
    "MLP_QUALITY_THRESHOLDS",
    "ROUTED_CANDIDATE_SELECTION_SCOPE",
    "apply_mlp_intervention_gates",
    "combine_mlp_intervention_reports",
    "evaluate_mlp_arm_gate",
]
