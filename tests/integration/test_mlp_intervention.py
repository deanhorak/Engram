import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
try:
    LlamaForCausalLM = transformers.LlamaForCausalLM
except (ImportError, RuntimeError) as error:
    pytest.skip(
        f"local Transformers Llama stack is unavailable: {error}",
        allow_module_level=True,
    )

from engram.evaluation.mlp_intervention import evaluate_mlp_interventions
from engram.tracing.teacher import capture_teacher_traces


def _tiny_teacher(path):
    torch.manual_seed(211)
    config = transformers.LlamaConfig(
        vocab_size=48,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    LlamaForCausalLM(config).save_pretrained(path, safe_serialization=True)


def _dataset(path, *, offset=0):
    path.write_text(
        json.dumps(
            {
                "input_ids": [1, 5 + offset, 9 + offset, 3 + offset, 2],
                "input_type": "prose",
            }
        )
        + "\n"
        + json.dumps(
            {"input_ids": [1, 7 + offset, 4 + offset, 8 + offset], "input_type": "code"}
        )
        + "\n",
        encoding="utf-8",
    )


def test_identity_and_full_oracle_match_tiny_teacher(tmp_path):
    model_path = tmp_path / "teacher"
    dataset = tmp_path / "validation.jsonl"
    _tiny_teacher(model_path)
    _dataset(dataset)

    report = evaluate_mlp_interventions(
        model_path,
        dataset,
        variants=("identity", "oracle"),
        top_ks=(12,),
        layer_mode="all",
    )

    assert report["baseline"]["next_token_positions"] == 7
    assert [arm["variant"] for arm in report["arms"]] == ["identity", "oracle"]
    for arm in report["arms"]:
        assert arm["quality"]["teacher_student_kl"]["mean"] == pytest.approx(
            0.0, abs=1e-7
        )
        assert arm["quality"]["teacher_top1_agreement"]["mean"] == 1.0
        assert arm["quality"]["final_hidden_relative_l2"]["mean"] == pytest.approx(
            0.0, abs=1e-7
        )
        assert arm["local_mlp"]["mlp_output_relative_l2"]["mean"] == pytest.approx(
            0.0, abs=1e-7
        )


def test_partial_oracle_and_rank_router_produce_finite_quality_metrics(tmp_path):
    model_path = tmp_path / "teacher"
    calibration_data = tmp_path / "calibration.jsonl"
    validation_data = tmp_path / "validation.jsonl"
    _tiny_teacher(model_path)
    _dataset(calibration_data)
    _dataset(validation_data, offset=1)
    traces = tmp_path / "calibration-traces"
    capture_teacher_traces(
        model_path,
        traces,
        dataset=calibration_data,
        split="calibration",
        samples=2,
    )

    report = evaluate_mlp_interventions(
        model_path,
        validation_data,
        calibration_traces=traces,
        variants=("identity", "oracle", "rank16", "overlap"),
        top_ks=(2,),
        candidate_counts=(6, 12),
        rank=2,
        regularization=1.0,
        calibration_records=8,
        posting_groups=3,
        posting_size=8,
        overlap_iterations=2,
        max_replication=2,
        layers=(0,),
        layer_mode="both",
    )

    assert len(report["arms"]) == 11
    rank_arms = [arm for arm in report["arms"] if arm["variant"] == "rank16"]
    overlap_arms = [arm for arm in report["arms"] if arm["variant"] == "overlap"]
    assert rank_arms
    assert overlap_arms
    for arm in report["arms"]:
        for name in (
            "teacher_student_kl",
            "student_nll",
            "nll_delta",
            "final_hidden_relative_l2",
        ):
            assert math_is_finite(arm["quality"][name]["mean"])
    for arm in rank_arms:
        recall = arm["local_mlp"]["candidate_recall"]["mean"]
        assert 0.0 <= recall <= 1.0
    for arm in overlap_arms:
        assert arm["local_mlp"]["posting_groups_selected"]["mean"] >= 1.0
    full_candidate_arms = [
        arm
        for arm in report["arms"]
        if arm["variant"] in {"rank16", "overlap"} and arm["candidate_count"] == 12
    ]
    for arm in full_candidate_arms:
        oracle = next(
            item
            for item in report["arms"]
            if item["variant"] == "oracle"
            and item["scope"] == arm["scope"]
            and item["layer_indices"] == arm["layer_indices"]
        )
        assert arm["local_mlp"]["candidate_recall"]["mean"] == 1.0
        assert arm["local_mlp"]["mlp_output_relative_l2"]["mean"] == pytest.approx(
            oracle["local_mlp"]["mlp_output_relative_l2"]["mean"], abs=1e-7
        )
        assert arm["quality"]["teacher_student_kl"]["mean"] == pytest.approx(
            oracle["quality"]["teacher_student_kl"]["mean"], abs=1e-7
        )


def test_dip_is_predictor_free_deterministic_and_accounts_projected_reads(tmp_path):
    model_path = tmp_path / "teacher"
    dataset = tmp_path / "validation.jsonl"
    _tiny_teacher(model_path)
    _dataset(dataset)

    arguments = {
        "variants": ("identity", "oracle", "dip"),
        "top_ks": (2,),
        "candidate_counts": (6, 12),
        "input_fractions": (0.5, 1.0),
        "layers": (0,),
        "layer_mode": "all",
    }
    first = evaluate_mlp_interventions(model_path, dataset, **arguments)
    second = evaluate_mlp_interventions(model_path, dataset, **arguments)

    assert first["calibration"] is None
    assert first["candidate_counts"] == [6, 12]
    assert first["input_fractions"] == [0.5, 1.0]
    dip_arms = [arm for arm in first["arms"] if arm["variant"] == "dip"]
    assert len(dip_arms) == 4
    assert [arm["name"] for arm in dip_arms] == [
        "dip_input_0p5_candidates_6_top_2_selected_layer_0",
        "dip_input_1_candidates_6_top_2_selected_layer_0",
        "dip_input_0p5_candidates_12_top_2_selected_layer_0",
        "dip_input_1_candidates_12_top_2_selected_layer_0",
    ]
    for arm in dip_arms:
        recall = arm["local_mlp"]["candidate_recall"]["mean"]
        score_mass = arm["local_mlp"]["oracle_score_mass_recall"]["mean"]
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= score_mass <= 1.0

    half_six = next(
        arm
        for arm in dip_arms
        if arm["input_fraction"] == 0.5 and arm["candidate_count"] == 6
    )
    accounting = half_six["projected_accounting"]
    assert accounting["input_coordinate_count"] == 4
    assert accounting["partial_key_bytes_per_token"] == 2 * 12 * 4 * 4
    assert accounting["candidate_completion_key_bytes_per_token"] == 2 * 6 * 4 * 4
    assert accounting["selected_value_bytes_per_token"] == 2 * 8 * 4
    assert accounting["projected_weight_scalar_reads_per_token"] == 160
    assert accounting["dense_mlp_weight_scalar_reads_per_token"] == 288
    assert accounting["total_mlp_weight_bytes_per_token"] == 640
    assert accounting["dense_mlp_weight_bytes_per_token"] == 1152
    assert accounting["projected_weight_traffic_fraction"] == pytest.approx(640 / 1152)

    full_candidate = next(
        arm
        for arm in dip_arms
        if arm["input_fraction"] == 1.0 and arm["candidate_count"] == 12
    )
    oracle = next(arm for arm in first["arms"] if arm["variant"] == "oracle")
    assert full_candidate["local_mlp"]["candidate_recall"]["mean"] == 1.0
    assert full_candidate["local_mlp"]["oracle_score_mass_recall"]["mean"] == 1.0
    assert full_candidate["local_mlp"]["mlp_output_relative_l2"][
        "mean"
    ] == pytest.approx(oracle["local_mlp"]["mlp_output_relative_l2"]["mean"], abs=1e-7)
    second_by_name = {arm["name"]: arm for arm in second["arms"]}
    for arm in dip_arms:
        repeated = second_by_name[arm["name"]]
        assert repeated["local_mlp"] == arm["local_mlp"]
        assert repeated["quality"] == arm["quality"]


@pytest.mark.parametrize("input_fraction", (0.0, -0.1, 1.1, float("nan")))
def test_dip_rejects_invalid_input_fraction(tmp_path, input_fraction):
    model_path = tmp_path / "teacher"
    dataset = tmp_path / "validation.jsonl"
    _tiny_teacher(model_path)
    _dataset(dataset)

    with pytest.raises(ValueError, match="input_fractions"):
        evaluate_mlp_interventions(
            model_path,
            dataset,
            variants=("dip",),
            top_ks=(2,),
            candidate_counts=(6,),
            input_fractions=(input_fraction,),
            layers=(0,),
            layer_mode="all",
        )


def test_dip_confirmation_verifies_configuration_selection_separation(tmp_path):
    model_path = tmp_path / "teacher"
    selection_data = tmp_path / "selection.jsonl"
    confirmation_data = tmp_path / "confirmation.jsonl"
    _tiny_teacher(model_path)
    _dataset(selection_data, offset=1)
    _dataset(confirmation_data)
    selection_traces = tmp_path / "selection-traces"
    capture_teacher_traces(
        model_path,
        selection_traces,
        dataset=selection_data,
        split="validation",
        samples=2,
    )

    report = evaluate_mlp_interventions(
        model_path,
        confirmation_data,
        variants=("identity", "oracle", "dip"),
        top_ks=(2,),
        candidate_counts=(6,),
        input_fractions=(0.75,),
        layers=(0,),
        layer_mode="all",
        evaluation_role="confirmation",
        configuration_selection_traces=selection_traces,
    )

    assert report["evaluation_role"] == "confirmation"
    assert report["configuration_selection"]["overlapping_sequence_count"] == 0
    assert (
        report["configuration_selection"]["held_out_from_configuration_selection"]
        is True
    )
    assert report["gate_summary"]["configuration_selection_separation_verified"] is True

    with pytest.raises(ValueError, match="configuration-selection and evaluation"):
        evaluate_mlp_interventions(
            model_path,
            selection_data,
            variants=("dip",),
            top_ks=(2,),
            candidate_counts=(6,),
            input_fractions=(0.75,),
            layers=(0,),
            layer_mode="all",
            evaluation_role="confirmation",
            configuration_selection_traces=selection_traces,
        )


def test_router_evaluation_rejects_calibration_dataset_reuse(tmp_path):
    model_path = tmp_path / "teacher"
    dataset = tmp_path / "data.jsonl"
    _tiny_teacher(model_path)
    _dataset(dataset)
    traces = tmp_path / "calibration-traces"
    capture_teacher_traces(
        model_path, traces, dataset=dataset, split="calibration", samples=2
    )

    with pytest.raises(ValueError, match="matching token sequences"):
        evaluate_mlp_interventions(
            model_path,
            dataset,
            calibration_traces=traces,
            variants=("identity", "rank16"),
            top_ks=(2,),
            candidate_counts=(6,),
            rank=2,
            calibration_records=8,
            layers=(0,),
            layer_mode="all",
        )

    reordered = tmp_path / "same-records-different-file.jsonl"
    lines = dataset.read_text(encoding="utf-8").splitlines()
    reordered.write_text("\n".join(reversed(lines)) + "\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="matching token sequences"):
        evaluate_mlp_interventions(
            model_path,
            reordered,
            calibration_traces=traces,
            variants=("identity", "rank16"),
            top_ks=(2,),
            candidate_counts=(6,),
            rank=2,
            calibration_records=8,
            layers=(0,),
            layer_mode="all",
        )


def math_is_finite(value):
    return value == value and abs(value) != float("inf")
