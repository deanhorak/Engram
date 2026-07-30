from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_mass_selector as selector
import engram.evaluation.olmoe_retrieval_episodic_mass_selector_runner as runner
from engram.utils import sha256_file


def _descriptor() -> dict[str, object]:
    return {
        "file": "confirmation.jsonl",
        "sha256": "a" * 64,
        "record_identity_sha256": "b" * 64,
        "records": 8,
        "tokens_per_record": 129,
        "prediction_positions_per_record": 128,
        "answer_prediction_positions_per_record": 32,
    }


def _freeze_context(tmp_path: Path) -> dict[str, object]:
    descriptor = _descriptor()
    return {
        "capacity_protocol_path": tmp_path / "capacity_protocol.json",
        "capacity_protocol_sha256": "1" * 64,
        "capacity_protocol": {
            "historical_bindings": {
                "train": {"path": "train.jsonl", "sha256": "2" * 64}
            },
            "schedule_contract": {"records": 8, "positions_per_record": 128},
            "output_projection": {"file_sha256": "3" * 64},
            "source_sha256": {"old.py": "4" * 64},
            "authenticated_confirmation_descriptor": descriptor,
        },
        "capacity_result_path": tmp_path / "capacity_result.json",
        "capacity_result_sha256": "5" * 64,
        "capacity_result": {
            "authenticated_confirmation_descriptor": descriptor,
        },
        "trace_manifest_path": tmp_path / "manifest.json",
        "trace_manifest_sha256": "6" * 64,
        "trace_manifest": {},
    }


def _valid_capacity_artifacts() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, str],
    dict[str, str],
]:
    protocol_binding = {"path": "/tmp/capacity.json", "sha256": "1" * 64}
    manifest_binding = {"path": "/tmp/manifest.json", "sha256": "2" * 64}
    descriptor = _descriptor()
    protocol: dict[str, object] = {
        "schema_version": 1,
        "experiment": runner._CAPACITY_PROTOCOL_EXPERIMENT,
        "status": runner._CAPACITY_PROTOCOL_STATUS,
        "confirmation_split_opened": False,
        "scope": {
            "split": "train",
            "records": 8,
            "read_positions_per_record": 32,
            "causal_selector": False,
            "semantic_or_M3_pass": False,
            "confirmation_file_access_permitted": False,
        },
        "oracle_method": {"authoritative_constructible_components": 28},
        "resource_contract": {
            "dense_full_context_logical_read_bytes": 2_164_260_864,
            "fixed_attention_state_bytes": 10_534_912,
            "fixed_combined_attention_and_episodic_traffic_bytes": 714_866_688,
            "fixed_fraction_of_dense_full_context_KV": 0.33030523255813954,
            "fresh_trace_and_oracle_solver_evaluator_only": True,
            "gamma_zero_earns_read_savings": False,
            "maximum_deployable_bytes_at_45_percent": 973_917_388,
            "oracle_shadow_trace_and_projection_evaluator_only": True,
            "predictor_weights_features_and_execution_not_counted": True,
            "remaining_selector_headroom_bytes": 259_050_700,
        },
        "source_sha256": {"legacy.py": "3" * 64},
        "historical_bindings": {
            "train": {"path": "/tmp/train.jsonl", "sha256": "4" * 64}
        },
        "authenticated_confirmation_descriptor": descriptor,
    }
    qualification = {
        "authenticated_base_projection_matches_Wo": True,
        "deterministic_solver_replay_exact": True,
        "objective_gap_certificate_available": True,
        "objective_never_worse_than_native_base": True,
        "passed": True,
        "quadratic_direct_parity": True,
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "experiment": runner._CAPACITY_RESULT_EXPERIMENT,
        "status": runner._CAPACITY_RESULT_STATUS,
        "protocol": protocol_binding,
        "confirmation_split_opened": False,
        "decision": {
            "train_full_visible_constructible_gate_passed": True,
            "train_only_causal_selector_authorized": True,
            "certified_optimistic_gate_passed": True,
            "failure_is_decisive": False,
            "confirmation_authorized": False,
            "development_authorized": False,
            "semantic_or_M3_gate_passed": False,
        },
        "post_solve_authentication": {"protocol": True, "manifest": True},
        "record_authentication": {"records": True, "schedule": True},
        "resource_contract": {
            "fixed_attention_state_bytes": 10_534_912,
            "fixed_combined_attention_and_episodic_traffic_bytes": 714_866_688,
            "fixed_fraction_of_dense_full_context_KV": 0.33030523255813954,
            "future_selector_not_counted_by_this_capacity_screen": True,
            "new_KV_state_or_read_traffic_bytes": 0,
            "trace_and_solver_evaluator_only": True,
        },
        "authoritative_arms": {
            "constructible": {
                "arm": "constructible_28_way",
                "components_per_head": 28,
                "diagnostic_only": False,
                "exact_native_anchor_included": False,
                "progression_authority": True,
                "feasible_gate_passed": True,
                "optimistic_gate_passed": True,
                "passed": True,
                "feasible_solution_metrics": {
                    "passed": True,
                    "gate": {"passed": True},
                },
                "qualification": qualification,
            }
        },
        "trace_manifest": {**manifest_binding, "record_count": 8},
        "authenticated_confirmation_descriptor": descriptor,
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "experiment": runner._TRACE_MANIFEST_EXPERIMENT,
        "format": "safetensors",
        "protocol": protocol_binding,
        "record_order": list(range(8)),
        "confirmation_split_opened": False,
        "shards": [{"record_index": value} for value in range(8)],
    }
    return protocol, result, manifest, protocol_binding, manifest_binding


def test_freeze_binds_exact_model_training_gate_and_resource_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _freeze_context(tmp_path)
    descriptor = context["capacity_result"]["authenticated_confirmation_descriptor"]
    monkeypatch.setattr(
        runner,
        "_authenticate_capacity_inputs",
        lambda **_kwargs: context,
    )
    monkeypatch.setattr(
        runner,
        "_source_inventory",
        lambda: {"core.py": "7" * 64, "runner.py": "8" * 64},
    )
    monkeypatch.setattr(
        runner,
        "_post_freeze_authentication",
        lambda _context: {
            "capacity_protocol": True,
            "capacity_result": True,
            "trace_manifest": True,
            "capacity_source_inventory": True,
            "mass_selector_source_inventory": True,
            "confirmation_not_opened": True,
        },
    )
    output = tmp_path / "selector_protocol.json"
    frozen = runner.freeze_mass_selector_protocol(
        capacity_protocol=tmp_path / "capacity_protocol.json",
        capacity_protocol_sha256="1" * 64,
        capacity_result=tmp_path / "capacity_result.json",
        capacity_result_sha256="5" * 64,
        trace_manifest=tmp_path / "manifest.json",
        trace_manifest_sha256="6" * 64,
        out=output,
    )
    protocol = frozen["protocol"]
    assert output.is_file()
    assert frozen["sha256"] == sha256_file(output)
    assert protocol["selector_model"]["components_per_query_head"] == 28
    assert protocol["selector_model"]["rank"] == 16
    assert protocol["selector_model"]["delta_clamp"] == [-16.0, 16.0]
    assert protocol["selector_model"]["parameter_shapes"] == {
        "U": [16, 28, 16],
        "V": [16, 16, 28],
        "E": [16, 16, 16],
        "B": [16, 16, 28],
    }
    training = protocol["training"]
    assert training["steps"] == 1_536
    assert training["warmup_steps"] == 96
    assert training["peak_learning_rate"] == 0.005
    assert training["final_learning_rate"] == 0.0005
    assert training["training_rows_per_layer_per_fold"] == 192
    assert training["steps_per_epoch"] == 96
    assert training["epochs"] == 16
    assert training["shuffle_generator"] == (
        "numpy.random.Generator(numpy.random.PCG64)"
    )
    assert training["training_device"] == "cuda"
    assert training["CUDA_deterministic_algorithms_required"] is True
    assert training["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert training["weight_decay"] == {
        "U": 1.0e-4,
        "V": 1.0e-4,
        "E": 0.0,
        "B": 0.0,
    }
    assert training["cross_validation"]["folds"] == [
        {
            "fold_index": 0,
            "training_record_indices": [1, 2, 3, 5, 6, 7],
            "heldout_record_indices": [0, 4],
            "initialization_seed": 2_026_073_001,
            "shuffle_seed": 2_026_073_002,
        },
        {
            "fold_index": 1,
            "training_record_indices": [0, 2, 3, 4, 6, 7],
            "heldout_record_indices": [1, 5],
            "initialization_seed": 2_026_073_002,
            "shuffle_seed": 2_026_073_003,
        },
        {
            "fold_index": 2,
            "training_record_indices": [0, 1, 3, 4, 5, 7],
            "heldout_record_indices": [2, 6],
            "initialization_seed": 2_026_073_003,
            "shuffle_seed": 2_026_073_004,
        },
        {
            "fold_index": 3,
            "training_record_indices": [0, 1, 2, 4, 5, 6],
            "heldout_record_indices": [3, 7],
            "initialization_seed": 2_026_073_004,
            "shuffle_seed": 2_026_073_005,
        },
    ]
    gate = protocol["progression_gate"]
    assert gate["arms_required"] == ["FP32", "BF16_RNE"]
    assert gate["minimum_global_recovery"] == 0.50
    assert gate["maximum_BF16_global_recovery_drop_from_FP32"] == 0.005
    resource = protocol["resource_contract"]
    assert resource["parameter_count"] == 25_600
    assert resource["serialized_parameter_bytes"] == 51_200
    assert resource["deployment_artifact_contains_BF16_only"] is True
    assert resource["deployment_artifact_BF16_tensor_bytes"] == 51_200
    assert resource["FP32_training_audit_copy_loaded_by_runtime"] is False
    assert resource["fixed_attention_state_bytes"] == 10_534_912
    assert resource["combined_attention_and_selector_state_bytes"] == 10_586_112
    assert resource["conservative_selector_scratch_bytes"] == 6_400
    assert resource["selector_multiply_accumulates_per_token"] == 229_376
    assert (
        resource["selector_multiply_accumulates_per_128_token_sequence"] == 29_360_128
    )
    assert (
        resource["conservative_selector_weight_traffic_bytes_per_128_token_sequence"]
        == 6_553_600
    )
    assert resource["total_logical_traffic_bytes_per_128_token_sequence"] == 721_420_288
    assert resource["fraction_of_dense_full_context_logical_reads"] == 1.0 / 3.0
    assert resource["remaining_headroom_bytes"] == 252_497_100
    assert resource["new_KV_read_traffic_bytes"] == 0
    assert resource["future_native_fused_single_value_pass_required"] is True
    assert protocol["authenticated_confirmation_descriptor"] == descriptor
    assert protocol["authenticated_confirmation_descriptor"] is not descriptor
    assert protocol["confirmation_split_opened"] is False


def test_capacity_validation_accepts_only_bound_authoritative_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, result, manifest, protocol_binding, manifest_binding = (
        _valid_capacity_artifacts()
    )
    monkeypatch.setattr(
        runner.full,
        "_source_inventory",
        lambda: {"legacy.py": "3" * 64},
    )
    runner._validate_capacity_artifacts(
        protocol,
        result,
        manifest,
        protocol_binding=protocol_binding,
        manifest_binding=manifest_binding,
    )

    mutations = (
        lambda _p, r, _m: r["post_solve_authentication"].update(manifest=False),
        lambda _p, r, _m: r["decision"].update(
            train_full_visible_constructible_gate_passed=False
        ),
        lambda _p, r, _m: r.update(status="failed"),
        lambda _p, r, _m: r["trace_manifest"].update(sha256="9" * 64),
        lambda _p, _r, m: m.update(protocol={"path": "wrong", "sha256": "1" * 64}),
    )
    for mutate in mutations:
        changed_protocol = deepcopy(protocol)
        changed_result = deepcopy(result)
        changed_manifest = deepcopy(manifest)
        mutate(changed_protocol, changed_result, changed_manifest)
        with pytest.raises(ValueError):
            runner._validate_capacity_artifacts(
                changed_protocol,
                changed_result,
                changed_manifest,
                protocol_binding=protocol_binding,
                manifest_binding=manifest_binding,
            )


def test_confirmation_argument_is_rejected_before_any_authentication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False

    def forbidden_auth(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("authentication must not run")

    monkeypatch.setattr(runner, "_authenticate_capacity_inputs", forbidden_auth)
    with pytest.raises(ValueError, match="confirmation split"):
        runner.freeze_mass_selector_protocol(
            capacity_protocol=tmp_path / "capacity.json",
            capacity_protocol_sha256="1" * 64,
            capacity_result=tmp_path / "result.json",
            capacity_result_sha256="2" * 64,
            trace_manifest=tmp_path / "manifest.json",
            trace_manifest_sha256="3" * 64,
            out=tmp_path / "confirmation.jsonl",
        )
    assert not called


def test_authentication_helper_guards_all_paths_before_checking_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False

    def forbidden_check(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("file check must not run")

    monkeypatch.setattr(runner, "_checked_file", forbidden_check)
    with pytest.raises(ValueError, match="confirmation split"):
        runner._authenticate_capacity_inputs(
            capacity_protocol=tmp_path / "capacity.json",
            capacity_protocol_sha256=runner._EXPECTED_CAPACITY_PROTOCOL_SHA256,
            capacity_result=tmp_path / "result.json",
            capacity_result_sha256=runner._EXPECTED_CAPACITY_RESULT_SHA256,
            trace_manifest=tmp_path / "confirmation.jsonl",
            trace_manifest_sha256=runner._EXPECTED_TRACE_MANIFEST_SHA256,
        )
    assert not called


def test_leaf_symlink_is_rejected_without_following_it(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        runner._guard_paths((("capacity result", alias),))


def test_existing_output_is_not_replaced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _freeze_context(tmp_path)
    monkeypatch.setattr(
        runner,
        "_authenticate_capacity_inputs",
        lambda **_kwargs: context,
    )
    monkeypatch.setattr(runner, "_source_inventory", lambda: {"runner.py": "8" * 64})
    monkeypatch.setattr(
        runner,
        "_post_freeze_authentication",
        lambda _context: {"all_roots": True},
    )
    output = tmp_path / "protocol.json"
    output.write_text('{"owner":"user"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        runner.freeze_mass_selector_protocol(
            capacity_protocol=tmp_path / "capacity.json",
            capacity_protocol_sha256="1" * 64,
            capacity_result=tmp_path / "result.json",
            capacity_result_sha256="5" * 64,
            trace_manifest=tmp_path / "manifest.json",
            trace_manifest_sha256="6" * 64,
            out=output,
        )
    assert output.read_text(encoding="utf-8") == '{"owner":"user"}\n'


def test_atomic_publication_never_replaces_a_racing_output(tmp_path: Path) -> None:
    output = runner._new_output(tmp_path / "result.json", "result")
    output.write_text('{"owner":"user"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        runner._atomic_json_new(output, {"owner": "selector"})
    assert output.read_text(encoding="utf-8") == '{"owner":"user"}\n'


def test_source_closure_is_deduplicated_and_includes_new_modules() -> None:
    assert len(runner._SOURCE_FILES) == len(set(runner._SOURCE_FILES))
    assert runner._CORE_SOURCE in runner._SOURCE_FILES
    assert runner._RUNNER_SOURCE in runner._SOURCE_FILES
    assert set(runner.full._SOURCE_FILES).issubset(runner._SOURCE_FILES)


def test_cli_exposes_freeze(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        runner,
        "freeze_mass_selector_protocol",
        lambda **kwargs: observed.update(kwargs),
    )
    assert (
        runner.main(
            [
                "freeze",
                "--capacity-protocol",
                str(tmp_path / "capacity.json"),
                "--capacity-protocol-sha256",
                "1" * 64,
                "--capacity-result",
                str(tmp_path / "result.json"),
                "--capacity-result-sha256",
                "2" * 64,
                "--trace-manifest",
                str(tmp_path / "manifest.json"),
                "--trace-manifest-sha256",
                "3" * 64,
                "--out",
                str(tmp_path / "protocol.json"),
            ]
        )
        == 0
    )
    assert observed["capacity_protocol_sha256"] == "1" * 64
    assert observed["capacity_result_sha256"] == "2" * 64
    assert observed["trace_manifest_sha256"] == "3" * 64


def test_model_artifact_round_trips_fp32_and_bf16(tmp_path: Path) -> None:
    shape = selector.SelectorShape(
        layers=2,
        heads=2,
        components=3,
        head_dimension=2,
        rank=2,
    )
    parameters = selector.initialize_parameters(shape, seed=71)
    path = tmp_path / "selector.safetensors"
    descriptor = runner._write_model_artifact(path, parameters, shape)
    assert descriptor["file_sha256"] == sha256_file(path)
    assert descriptor["embedded_BF16_tensor_bytes"] == shape.parameter_count * 2
    assert descriptor["artifact_role"] == "training_audit"
    assert descriptor["authorized_for_runtime_loading"] is False

    fp32 = runner._load_model_artifact(
        path,
        descriptor,
        shape,
        bf16=False,
    )
    bf16 = runner._load_model_artifact(
        path,
        descriptor,
        shape,
        bf16=True,
    )
    for name in ("U", "V", "E", "B"):
        np.testing.assert_array_equal(
            fp32.as_dict()[name],
            parameters.as_dict()[name],
        )
    expected_bf16, _bits = selector.quantize_parameters_bf16(parameters, shape)
    for name in ("U", "V", "E", "B"):
        np.testing.assert_array_equal(
            bf16.as_dict()[name],
            expected_bf16.as_dict()[name],
        )


def test_bf16_deployment_artifact_has_no_fp32_copy(tmp_path: Path) -> None:
    shape = selector.SelectorShape(
        layers=2,
        heads=2,
        components=3,
        head_dimension=2,
        rank=2,
    )
    parameters = selector.initialize_parameters(shape, seed=73)
    path = tmp_path / "selector-bf16.safetensors"
    descriptor = runner._write_bf16_deployment_artifact(
        path,
        parameters,
        shape,
    )
    assert descriptor["artifact_role"] == "native_BF16_deployment"
    assert descriptor["authorized_for_runtime_loading"] is True
    assert descriptor["contains_FP32_training_copy"] is False
    assert descriptor["BF16_tensor_bytes"] == shape.parameter_count * 2
    replay = runner._load_bf16_deployment_artifact(
        path,
        descriptor,
        shape,
    )
    expected, _bits = selector.quantize_parameters_bf16(parameters, shape)
    for name in ("U", "V", "E", "B"):
        np.testing.assert_array_equal(
            replay.as_dict()[name],
            expected.as_dict()[name],
        )


def test_selector_gate_requires_both_serialized_arms_and_parity() -> None:
    coefficients = np.asarray([[[[[0.25, 0.75]]]]], dtype=np.float64)
    valid = np.ones(coefficients.shape, dtype=bool)
    metrics = {"passed": True, "global": {"recovery": 0.55}}
    passed = runner._selector_gate(
        metrics,
        metrics,
        fp32_coefficients=coefficients,
        bf16_coefficients=coefficients,
        valid=valid,
        mass_score_max_abs=0.0,
        mass_score_delta_max_abs=0.0,
        zero_coefficient_max_abs=0.0,
        zero_output_max_abs=0.0,
        deterministic_replay_exact=True,
    )
    assert passed["passed"] is True

    failed = runner._selector_gate(
        metrics,
        metrics,
        fp32_coefficients=coefficients,
        bf16_coefficients=coefficients,
        valid=valid,
        mass_score_max_abs=1.1e-6,
        mass_score_delta_max_abs=0.0,
        zero_coefficient_max_abs=0.0,
        zero_output_max_abs=0.0,
        deterministic_replay_exact=True,
    )
    assert failed["mass_vs_reconstructed_score_coefficient_parity"] is False
    assert failed["passed"] is False


def test_cli_exposes_fit_screen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        runner,
        "fit_screen_mass_selector",
        lambda **kwargs: observed.update(kwargs),
    )
    assert (
        runner.main(
            [
                "fit-screen",
                "--protocol",
                str(tmp_path / "protocol.json"),
                "--protocol-sha256",
                "1" * 64,
                "--artifact-directory",
                str(tmp_path / "artifacts"),
                "--out",
                str(tmp_path / "result.json"),
                "--device",
                "cuda",
            ]
        )
        == 0
    )
    assert observed == {
        "protocol": str(tmp_path / "protocol.json"),
        "protocol_sha256": "1" * 64,
        "artifact_directory": str(tmp_path / "artifacts"),
        "out": str(tmp_path / "result.json"),
        "device": "cuda",
    }
