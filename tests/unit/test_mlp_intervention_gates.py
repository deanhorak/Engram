import pytest

from engram.evaluation.gates import (
    apply_mlp_intervention_gates,
    combine_mlp_intervention_reports,
    evaluate_mlp_arm_gate,
)
from engram.evaluation.report import write_mlp_intervention_report


def _arm(
    variant, *, kl, top1, nll_delta, residual, recall=None, scope="all", layers=(0,)
):
    arm = {
        "name": variant,
        "variant": variant,
        "scope": scope,
        "layer_indices": list(layers),
        "quality": {
            "teacher_student_kl": {"mean": kl},
            "teacher_top1_agreement": {"mean": top1},
            "nll_delta": {"mean": nll_delta},
            "final_hidden_relative_l2": {"mean": residual},
        },
        "local_mlp": {"tokens": 256},
    }
    if recall is not None:
        arm["local_mlp"]["candidate_recall"] = {"mean": recall}
    if variant != "identity":
        arm["top_k"] = 1
    if variant not in {"identity", "oracle", "shared_basis"}:
        arm["candidate_count"] = 1
    if variant in {"dip", "dip_paq"}:
        arm["input_fraction"] = 0.75
    if variant == "dip_paq":
        arm["projected_accounting"] = {"cold_fraction_of_dense_q4": 0.44}
    return arm


def _full_width_scope(width=10):
    return {
        "mode": "full_converted_width",
        "converted_width": width,
        "scored_width": width,
        "candidate_shortlist": False,
        "authentication": {
            "source": "strict_reloaded_artifact_header_and_manifest",
            "verified": True,
            "sha256": "a" * 64,
        },
    }


def _base(num_hidden_layers=1):
    return {
        "schema_version": 1,
        "experiment": "trained_teacher_mlp_intervention",
        "num_hidden_layers": num_hidden_layers,
        "intermediate_size": 10,
        "baseline": {
            "sequences": 8,
            "unique_sequences": 8,
            "next_token_positions": 256,
        },
    }


def _held_out_calibration():
    return {
        "dataset_hash": "calibration",
        "separation_method": "exact_token_sequence_hashes",
        "calibration_sequence_count": 8,
        "calibration_unique_sequence_count": 8,
        "evaluation_sequence_count": 8,
        "evaluation_unique_sequence_count": 8,
        "record_level_disjoint": True,
        "overlapping_sequence_count": 0,
    }


def test_oracle_gate_enforces_all_quality_thresholds():
    passing = _arm("oracle", kl=0.04, top1=0.91, nll_delta=0.03, residual=0.09)
    failing = _arm("oracle", kl=0.06, top1=0.91, nll_delta=0.03, residual=0.09)

    assert evaluate_mlp_arm_gate(passing)["passed"] is True
    result = evaluate_mlp_arm_gate(failing)
    assert result["passed"] is False
    assert result["failed_metrics"] == ["teacher_student_kl"]


def test_routed_gate_requires_candidate_recall_and_sets_stop_decision():
    identity = _arm("identity", kl=0.0, top1=1.0, nll_delta=0.0, residual=0.0)
    routed = _arm(
        "overlap", kl=0.01, top1=0.99, nll_delta=0.01, residual=0.02, recall=0.90
    )
    report = {
        **_base(),
        "dataset_hash": "validation",
        "calibration": _held_out_calibration(),
        "arms": [identity, routed],
    }

    gated = apply_mlp_intervention_gates(report)

    assert gated["quality_targets_met"] is False
    assert gated["gate_summary"]["development_decision"] == "stop_before_serialization"
    assert gated["arms"][1]["gate"]["failed_metrics"] == ["candidate_recall"]


def test_shared_basis_authenticates_full_width_and_marks_recall_not_applicable(
    tmp_path,
):
    identity = _arm("identity", kl=0.0, top1=1.0, nll_delta=0.0, residual=0.0)
    oracle = _arm("oracle", kl=0.01, top1=0.99, nll_delta=0.01, residual=0.02)
    shared = _arm(
        "shared_basis",
        kl=0.01,
        top1=0.99,
        nll_delta=0.01,
        residual=0.02,
    )
    shared["selection_scope"] = _full_width_scope()
    shared["projected_accounting"] = {"cold_fraction_of_dense_q4": 0.44}
    report = apply_mlp_intervention_gates(
        {
            **_base(),
            "status": "test",
            "calibration": _held_out_calibration(),
            "measurement_caveat": "test",
            "arms": [identity, oracle, shared],
        }
    )

    gated = report["arms"][2]
    recall = gated["gate"]["candidate_recall"]
    recall_check = next(
        check
        for check in gated["gate"]["checks"]
        if check["metric"] == "candidate_recall"
    )
    assert recall == {
        "applicable": False,
        "status": "not_applicable",
        "selection_scope": "full_converted_width",
        "selection_scope_authenticated": True,
        "reason": "all_converted_records_are_scored_without_a_candidate_shortlist",
    }
    assert recall_check["actual"] is None
    assert recall_check["comparison"] == "not_applicable"
    assert recall_check["passed"] is True
    assert gated["candidate_recall_applicable"] is False
    assert "candidate_recall" not in gated["local_mlp"]
    assert report["gate_summary"]["passing_routed_arms"] == []
    assert report["gate_summary"]["passing_full_width_arms"] == ["shared_basis"]
    assert report["quality_targets_met"] is True
    assert report["gate_summary"]["development_decision"] == (
        "eligible_for_full_width_artifact_confirmation"
    )

    report["baseline"].update(
        {
            "negative_log_likelihood": 1.0,
            "perplexity": 2.0,
        }
    )
    for arm in report["arms"]:
        arm["local_mlp"].setdefault("mlp_output_relative_l2", {"mean": 0.0})
    _, markdown_path = write_mlp_intervention_report(report, tmp_path)
    assert "N/A (full converted width)" in markdown_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda arm: arm.pop("selection_scope"),
        lambda arm: arm.__setitem__("selection_scope", "full_converted_width"),
        lambda arm: arm["selection_scope"]["authentication"].__setitem__(
            "verified", False
        ),
        lambda arm: arm["selection_scope"].__setitem__(
            "candidate_shortlist", True
        ),
        lambda arm: arm["selection_scope"].__setitem__("scored_width", 9),
        lambda arm: arm["selection_scope"]["authentication"].__setitem__(
            "sha256", "not-a-sha"
        ),
    ],
)
def test_shared_basis_recall_exemption_requires_authenticated_full_width(mutate):
    shared = _arm(
        "shared_basis",
        kl=0.01,
        top1=0.99,
        nll_delta=0.01,
        residual=0.02,
    )
    shared["selection_scope"] = _full_width_scope()
    shared["projected_accounting"] = {"cold_fraction_of_dense_q4": 0.44}
    mutate(shared)

    with pytest.raises(ValueError, match="selection_scope|artifact|strict-reload"):
        apply_mlp_intervention_gates(
            {
                **_base(),
                "calibration": _held_out_calibration(),
                "arms": [shared],
            }
        )


def test_full_width_scope_rejects_fabricated_recall_or_candidate_shortlist():
    shared = _arm(
        "shared_basis",
        kl=0.01,
        top1=0.99,
        nll_delta=0.01,
        residual=0.02,
    )
    shared["selection_scope"] = _full_width_scope()
    shared["projected_accounting"] = {"cold_fraction_of_dense_q4": 0.44}
    shared["local_mlp"]["candidate_recall"] = {"mean": 1.0}
    with pytest.raises(ValueError, match="candidate_recall must be omitted"):
        apply_mlp_intervention_gates(
            {
                **_base(),
                "calibration": _held_out_calibration(),
                "arms": [shared],
            }
        )

    shared["local_mlp"].pop("candidate_recall")
    shared["candidate_count"] = 10
    with pytest.raises(ValueError, match="must omit candidate_count"):
        apply_mlp_intervention_gates(
            {
                **_base(),
                "calibration": _held_out_calibration(),
                "arms": [shared],
            }
        )


def test_routed_candidate_scope_cannot_opt_out_of_recall():
    routed = _arm(
        "rank16", kl=0.01, top1=0.99, nll_delta=0.01, residual=0.02
    )
    routed["selection_scope"] = {
        "mode": "routed_candidates",
        "candidate_shortlist": True,
    }
    routed["candidate_recall_applicable"] = False

    with pytest.raises(ValueError, match="candidate_recall_applicable"):
        apply_mlp_intervention_gates(
            {
                **_base(),
                "calibration": _held_out_calibration(),
                "arms": [routed],
            }
        )

    routed.pop("candidate_recall_applicable")
    with pytest.raises(ValueError, match="candidate_recall"):
        apply_mlp_intervention_gates(
            {
                **_base(),
                "calibration": _held_out_calibration(),
                "arms": [routed],
            }
        )


def test_shared_basis_gate_requires_worst_case_physical_traffic_pass():
    shared = _arm(
        "shared_basis",
        kl=0.01,
        top1=0.99,
        nll_delta=0.01,
        residual=0.02,
    )
    shared["selection_scope"] = _full_width_scope()
    shared["projected_accounting"] = {"cold_fraction_of_dense_q4": 0.451}

    result = evaluate_mlp_arm_gate(shared, intermediate_size=10)

    assert result["passed"] is False
    assert result["failed_metrics"] == ["cold_fraction_of_dense_q4"]
    assert result["candidate_recall"]["applicable"] is False


def test_predictor_free_dip_does_not_require_calibration_provenance():
    identity = _arm("identity", kl=0.0, top1=1.0, nll_delta=0.0, residual=0.0)
    oracle = _arm("oracle", kl=0.01, top1=0.99, nll_delta=0.01, residual=0.02)
    dip = _arm("dip", kl=0.01, top1=0.99, nll_delta=0.01, residual=0.02, recall=0.99)
    report = apply_mlp_intervention_gates(
        {
            **_base(),
            "arms": [identity, oracle, dip],
        }
    )

    assert report.get("calibration") is None
    assert report["gate_summary"]["data_separation_verified"] is True
    assert report["gate_summary"]["development_decision"] == (
        "eligible_for_selector_serialization"
    )
    assert report["quality_targets_met"] is True


def test_dip_gate_rejects_missing_input_fraction():
    dip = _arm("dip", kl=0.01, top1=0.99, nll_delta=0.01, residual=0.02, recall=0.99)
    del dip["input_fraction"]

    with pytest.raises(ValueError, match="input_fraction"):
        apply_mlp_intervention_gates({**_base(), "arms": [dip]})


def test_quantized_dip_gate_requires_cold_traffic_pass():
    passing = _arm(
        "dip_paq", kl=0.01, top1=0.99, nll_delta=0.01, residual=0.02, recall=0.99
    )
    assert evaluate_mlp_arm_gate(passing)["passed"] is True

    passing["projected_accounting"]["cold_fraction_of_dense_q4"] = 0.451
    result = evaluate_mlp_arm_gate(passing)
    assert result["passed"] is False
    assert result["failed_metrics"] == ["cold_fraction_of_dense_q4"]


def test_invalid_learned_calibration_does_not_block_passing_dip():
    identity = _arm("identity", kl=0.0, top1=1.0, nll_delta=0.0, residual=0.0)
    oracle = _arm("oracle", kl=0.01, top1=0.99, nll_delta=0.01, residual=0.02)
    learned = _arm(
        "rank16", kl=0.01, top1=0.99, nll_delta=0.01, residual=0.02, recall=0.99
    )
    dip = _arm("dip", kl=0.01, top1=0.99, nll_delta=0.01, residual=0.02, recall=0.99)

    report = apply_mlp_intervention_gates(
        {
            **_base(),
            "calibration": {"dataset_hash": "unverified"},
            "arms": [identity, oracle, learned, dip],
        }
    )

    assert report["quality_targets_met"] is True
    assert report["gate_summary"]["data_separation_verified"] is False
    assert report["gate_summary"]["passing_learned_router_arms"] == []
    assert report["gate_summary"]["passing_predictor_free_arms"] == ["dip"]
    assert report["gate_summary"]["development_decision"] == (
        "eligible_for_selector_serialization"
    )


def test_confirmation_gate_requires_selection_separation_evidence():
    identity = _arm("identity", kl=0.0, top1=1.0, nll_delta=0.0, residual=0.0)

    with pytest.raises(ValueError, match="configuration-selection separation"):
        apply_mlp_intervention_gates(
            {**_base(), "evaluation_role": "confirmation", "arms": [identity]}
        )


def test_identity_gate_is_instrumentation_sanity_not_router_evidence():
    identity = _arm("identity", kl=0.0, top1=1.0, nll_delta=0.0, residual=0.0)
    report = apply_mlp_intervention_gates(
        {
            **_base(),
            "arms": [identity],
        }
    )

    assert report["arms"][0]["gate"]["passed"] is True
    assert report["quality_targets_met"] is None
    assert (
        report["gate_summary"]["development_decision"]
        == "insufficient_all_layer_evidence"
    )


def test_router_cannot_progress_without_passing_identity_sanity():
    broken_identity = _arm("identity", kl=0.01, top1=0.9, nll_delta=0.0, residual=0.01)
    routed = _arm(
        "rank16", kl=0.01, top1=0.99, nll_delta=0.01, residual=0.02, recall=0.99
    )
    gated = apply_mlp_intervention_gates(
        {
            **_base(),
            "dataset_hash": "validation",
            "calibration": _held_out_calibration(),
            "arms": [broken_identity, routed],
        }
    )

    assert gated["quality_targets_met"] is None
    assert (
        gated["gate_summary"]["development_decision"] == "insufficient_identity_sanity"
    )


def test_individual_layer_only_report_is_insufficient_not_a_failure():
    identity = _arm(
        "identity", kl=0.0, top1=1.0, nll_delta=0.0, residual=0.0, scope="individual"
    )
    routed = _arm(
        "rank16",
        kl=1.0,
        top1=0.0,
        nll_delta=1.0,
        residual=1.0,
        recall=0.0,
        scope="individual",
    )
    gated = apply_mlp_intervention_gates(
        {
            **_base(2),
            "arms": [identity, routed],
        }
    )

    assert gated["quality_targets_met"] is None
    assert (
        gated["gate_summary"]["development_decision"]
        == "insufficient_all_layer_evidence"
    )


def test_passing_router_requires_matched_magnitude_reference():
    identity = _arm("identity", kl=0.0, top1=1.0, nll_delta=0.0, residual=0.0)
    routed = _arm(
        "rank16", kl=0.01, top1=0.99, nll_delta=0.01, residual=0.02, recall=0.99
    )
    routed["top_k"] = 4
    routed["candidate_count"] = 4
    base = {
        **_base(),
        "dataset_hash": "validation",
        "calibration": _held_out_calibration(),
    }

    unmatched = apply_mlp_intervention_gates({**base, "arms": [identity, routed]})
    assert unmatched["quality_targets_met"] is None
    assert (
        unmatched["gate_summary"]["development_decision"]
        == "insufficient_matched_magnitude_reference"
    )

    oracle = _arm("oracle", kl=0.01, top1=0.99, nll_delta=0.01, residual=0.02)
    oracle["top_k"] = 4
    matched = apply_mlp_intervention_gates({**base, "arms": [identity, oracle, routed]})
    assert matched["quality_targets_met"] is True
    assert (
        matched["gate_summary"]["development_decision"]
        == "eligible_for_router_serialization"
    )

    oracle["quality"]["teacher_student_kl"]["mean"] = 1.0
    direct_quality_wins = apply_mlp_intervention_gates(
        {**base, "arms": [identity, oracle, routed]}
    )
    assert direct_quality_wins["quality_targets_met"] is True
    assert direct_quality_wins["gate_summary"]["oracle_viability"] is False


def test_oracle_only_pass_does_not_claim_overall_quality_target():
    identity = _arm("identity", kl=0.0, top1=1.0, nll_delta=0.0, residual=0.0)
    oracle = _arm("oracle", kl=0.01, top1=0.99, nll_delta=0.01, residual=0.02)
    report = apply_mlp_intervention_gates(
        {
            **_base(),
            "arms": [identity, oracle],
        }
    )

    assert report["gate_summary"]["oracle_viability"] is True
    assert report["quality_targets_met"] is None


def test_gate_rejects_unknown_variant():
    unknown = _arm(
        "future_magic", kl=0.0, top1=1.0, nll_delta=0.0, residual=0.0, recall=1.0
    )
    with pytest.raises(ValueError, match="unsupported gated"):
        evaluate_mlp_arm_gate(unknown)
    with pytest.raises(ValueError, match="unsupported gated"):
        apply_mlp_intervention_gates(
            {
                **_base(),
                "arms": [unknown],
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("teacher_student_kl", -1.0),
        ("teacher_student_kl", float("nan")),
        ("teacher_top1_agreement", 1.01),
        ("nll_delta", float("-inf")),
        ("final_hidden_relative_l2", -0.1),
    ],
)
def test_gate_rejects_nonfinite_or_impossible_quality_metrics(field, value):
    arm = _arm("oracle", kl=0.01, top1=0.99, nll_delta=0.0, residual=0.01)
    arm["quality"][field]["mean"] = value

    with pytest.raises(ValueError, match="metric"):
        apply_mlp_intervention_gates({**_base(), "arms": [arm]})


def test_gate_rejects_impossible_recall_and_missing_budgets():
    routed = _arm(
        "rank16", kl=0.01, top1=0.99, nll_delta=0.0, residual=0.01, recall=2.0
    )
    with pytest.raises(ValueError, match="candidate_recall"):
        apply_mlp_intervention_gates(
            {
                **_base(),
                "calibration": _held_out_calibration(),
                "arms": [routed],
            }
        )

    routed["local_mlp"]["candidate_recall"]["mean"] = 0.99
    routed["top_k"] = None
    with pytest.raises(ValueError, match="positive top_k"):
        apply_mlp_intervention_gates(
            {
                **_base(),
                "calibration": _held_out_calibration(),
                "arms": [routed],
            }
        )


def test_duplicate_only_corpus_cannot_authorize_progression():
    identity = _arm("identity", kl=0.0, top1=1.0, nll_delta=0.0, residual=0.0)
    oracle = _arm("oracle", kl=0.01, top1=0.99, nll_delta=0.01, residual=0.02)
    report = apply_mlp_intervention_gates(
        {
            **_base(),
            "baseline": {
                "sequences": 8,
                "unique_sequences": 1,
                "next_token_positions": 256,
            },
            "arms": [identity, oracle],
        }
    )

    assert report["quality_targets_met"] is None
    assert report["gate_summary"]["evidence_size_verified"] is False
    assert (
        report["gate_summary"]["development_decision"]
        == "insufficient_evaluation_corpus"
    )


def test_unverified_data_separation_is_unknown_not_router_failure():
    identity = _arm("identity", kl=0.0, top1=1.0, nll_delta=0.0, residual=0.0)
    routed = _arm(
        "rank16", kl=0.01, top1=0.99, nll_delta=0.01, residual=0.02, recall=0.99
    )
    report = apply_mlp_intervention_gates(
        {
            **_base(),
            "calibration": {"dataset_hash": "different-file-only"},
            "arms": [identity, routed],
        }
    )

    assert report["quality_targets_met"] is None
    assert report["gate_summary"]["data_separation_verified"] is False
    assert report["gate_summary"]["routing_viability"] is None
    assert report["gate_summary"]["development_decision"] == "invalid_data_separation"


def test_composite_gate_validates_provenance_and_deduplicates_identity():
    identity = _arm("identity", kl=0.0, top1=1.0, nll_delta=0.0, residual=0.0)
    oracle = _arm("oracle", kl=0.01, top1=0.99, nll_delta=0.01, residual=0.02)
    routed = _arm(
        "rank16", kl=0.01, top1=0.99, nll_delta=0.01, residual=0.02, recall=0.99
    )
    common = {
        **_base(),
        "source_model_hash": "model",
        "dataset_hash": "validation",
        "selected_layers": [0],
        "layer_mode": "all",
    }
    calibration = {
        **_held_out_calibration(),
        "trace_path": "/calibration",
        "records_per_layer_limit": 8,
        "regularization": 1.0,
        "rank": 1,
        "calibration_sequence_count": 8,
        "evaluation_sequence_count": 8,
    }

    combined = combine_mlp_intervention_reports(
        [
            {**common, "arms": [identity, oracle]},
            {**common, "calibration": calibration, "arms": [identity, routed]},
        ]
    )

    assert [arm["variant"] for arm in combined["arms"]] == [
        "identity",
        "oracle",
        "rank16",
    ]
    assert combined["gate_summary"]["development_decision"] == (
        "eligible_for_router_serialization"
    )

    with pytest.raises(ValueError, match="different dataset_hash"):
        combine_mlp_intervention_reports(
            [
                {**common, "arms": [identity, oracle]},
                {
                    **common,
                    "dataset_hash": "other-validation",
                    "calibration": calibration,
                    "arms": [identity, routed],
                },
            ]
        )
