from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_phase_selector as selector
import engram.evaluation.olmoe_retrieval_episodic_phase_selector_runner as runner
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


def _metrics(
    *,
    global_recovery: float,
    minimum_sequence: float,
    minimum_block: float,
) -> dict[str, object]:
    return {
        "global": {"recovery": global_recovery},
        "heldout_sequences": [
            {"record_index": index, "recovery": minimum_sequence + index * 0.001}
            for index in range(8)
        ],
        "block_entry_positions": [
            {"position": position, "recovery": minimum_block + index * 0.001}
            for index, position in enumerate(runner._BLOCK_POSITIONS)
        ],
        "positive_recovery_layer_count": 16,
    }


def _content_failure(protocol_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": runner.content._RESULT_EXPERIMENT,
        "status": runner.content._FAILED_STATUS,
        "protocol": {
            "path": str(protocol_path),
            "sha256": runner._EXPECTED_CONTENT_PROTOCOL_SHA256,
        },
        "confirmation_split_opened": False,
        "gate": {
            "passed": False,
            "FP32_gate_passed": False,
            "BF16_gate_passed": False,
        },
        "decision": {
            "train_content_selector_oof_gate_passed": False,
            "train_only_native_integration_implementation_authorized": False,
            "development_authorized": False,
            "confirmation_authorized": False,
            "semantic_or_M3_gate_passed": False,
        },
        "arms": {
            "FP32": {
                "metrics": _metrics(
                    global_recovery=0.25426155258896843,
                    minimum_sequence=0.23154313101688695,
                    minimum_block=0.18379959630601683,
                )
            },
            "BF16_RNE_parameters_and_sidecars": {
                "metrics": _metrics(
                    global_recovery=0.2542207419770499,
                    minimum_sequence=0.2316160008520175,
                    minimum_block=0.1837115447332116,
                )
            },
        },
        "post_fit_authentication": {"all": True},
    }


def _freeze_context(tmp_path: Path) -> dict[str, object]:
    descriptor = _descriptor()
    mass_folds = []
    for fold in runner._fold_contract():
        mass_folds.append(
            {
                "fold_index": fold["fold_index"],
                "path": tmp_path / f"mass-fold-{fold['fold_index']}.safetensors",
                "descriptor": {
                    "file_sha256": str(fold["fold_index"]) * 64,
                    "fp32_parameter_sha256": "c" * 64,
                    "bf16_decoded_parameter_sha256": "d" * 64,
                },
            }
        )
    capacity = {
        "capacity_protocol_path": tmp_path / "capacity-protocol.json",
        "capacity_protocol_sha256": "1" * 64,
        "capacity_result_path": tmp_path / "capacity-result.json",
        "capacity_result_sha256": "2" * 64,
        "trace_manifest_path": tmp_path / "trace-manifest.json",
        "trace_manifest_sha256": "3" * 64,
    }
    predecessor = {
        "capacity": capacity,
        "residual_manifest_path": tmp_path / "residual-manifest.json",
        "package": {
            "manifest_path": tmp_path / "package-manifest.json",
            "non_mlp_path": tmp_path / "non-mlp.safetensors",
        },
        "record_join": {f"record_{index:02d}": True for index in range(8)},
    }
    content_protocol = {
        "predecessor_mass_protocol": {
            "path": str(tmp_path / "mass-protocol.json"),
            "sha256": runner._EXPECTED_MASS_PROTOCOL_SHA256,
        },
        "predecessor_mass_result": {
            "path": str(tmp_path / "mass-result.json"),
            "sha256": runner._EXPECTED_MASS_RESULT_SHA256,
        },
        "authenticated_confirmation_descriptor": descriptor,
    }
    return {
        "content_protocol_path": tmp_path / "content-protocol.json",
        "content_protocol": content_protocol,
        "content_result_path": tmp_path / "content-result.json",
        "content_result": {},
        "content_failure": {"authenticated": True},
        "predecessor": predecessor,
        "mass_fold_artifacts": mass_folds,
    }


def test_phase_schedule_is_relative_active_and_shift_equivariant() -> None:
    positions = np.arange(94, 130, dtype=np.int64)
    phase, active = runner._derive_phase_schedule(positions)
    expected_active = (positions >= 96) & (positions < 128)
    assert np.array_equal(active, expected_active)
    assert np.array_equal(phase[active], np.tile(np.arange(8), 4))
    assert np.all(phase[~active] == 0)

    shifted_phase, shifted_active = runner._derive_phase_schedule(
        positions + 37,
        block_positions=tuple(value + 37 for value in runner._BLOCK_POSITIONS),
    )
    assert np.array_equal(shifted_phase, phase)
    assert np.array_equal(shifted_active, active)


def test_phase_schedule_rejects_overlapping_spans() -> None:
    with pytest.raises(ValueError, match="overlap"):
        runner._derive_phase_schedule(
            np.arange(96, 112, dtype=np.int64),
            block_positions=(96, 100),
        )


def test_resource_contract_is_exact_and_below_51_head_ceiling() -> None:
    resource = runner._resource_contract()
    assert resource["phase_table_parameter_count"] == 57_344
    assert resource["mass_selector_parameter_count"] == 25_600
    assert resource["parameter_count"] == 82_944
    assert resource["serialized_parameter_bytes"] == 165_888
    assert resource["combined_attention_and_selector_state_bytes"] == 10_700_800
    assert resource["total_logical_traffic_bytes_per_128_token_sequence"] == 736_100_352
    assert resource["fraction_of_dense_full_context_logical_reads"] == pytest.approx(
        0.34011627906976744
    )
    assert resource["exact_51_head_equivalent_ceiling_bytes"] == 973_384_704
    assert resource["remaining_headroom_bytes"] == 237_284_352
    assert resource["persistent_value_sidecar_bytes"] == 0
    assert resource["new_KV_read_traffic_bytes"] == 0
    assert resource["full_KV_sidecar_or_second_value_read_pass"] is False


def test_content_failure_validation_is_exact() -> None:
    protocol_path = Path("/tmp/content-protocol.json")
    expected = runner._validate_content_failure(
        _content_failure(protocol_path),
        protocol_path=protocol_path,
    )
    assert expected["FP32_global_recovery"] == 0.25426155258896843
    assert expected["BF16_minimum_block_recovery"] == 0.1837115447332116
    changed = _content_failure(protocol_path)
    changed["arms"]["BF16_RNE_parameters_and_sidecars"]["metrics"]["global"][
        "recovery"
    ] += 0.001
    with pytest.raises(ValueError, match="metrics"):
        runner._validate_content_failure(changed, protocol_path=protocol_path)


def test_freeze_binds_sequential_model_schedule_resources_and_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _freeze_context(tmp_path)
    descriptor = context["content_protocol"]["authenticated_confirmation_descriptor"]
    monkeypatch.setattr(runner, "_authenticate_inputs", lambda **_kwargs: context)
    monkeypatch.setattr(
        runner,
        "_source_inventory",
        lambda: {"phase.py": "e" * 64, "runner.py": "f" * 64},
    )
    monkeypatch.setattr(
        runner.content,
        "_query_positions",
        lambda _context: np.arange(96, 128, dtype=np.int64),
    )
    output = tmp_path / "phase-protocol.json"
    frozen = runner.freeze_phase_selector_protocol(
        content_protocol=tmp_path / "content-protocol.json",
        content_protocol_sha256=runner._EXPECTED_CONTENT_PROTOCOL_SHA256,
        content_result=tmp_path / "content-result.json",
        content_result_sha256=runner._EXPECTED_CONTENT_RESULT_SHA256,
        out=output,
    )
    protocol = frozen["protocol"]
    assert frozen["sha256"] == sha256_file(output)
    assert protocol["predecessor_mass_protocol"]["sha256"] == (
        runner._EXPECTED_MASS_PROTOCOL_SHA256
    )
    assert len(protocol["predecessor_mass_fold_artifacts"]) == 4
    assert protocol["selector_model"]["parameter_shapes"]["T"] == [8, 16, 16, 28]
    assert protocol["phase_schedule"]["phase_values_on_authenticated_reads"] == (
        list(range(8)) * 4
    )
    assert protocol["training"]["sequential_fit"]["no_joint_mass_phase_updates"]
    assert protocol["training"]["weight_decay"]["T"] == 1.0e-4
    assert protocol["scope"]["independent_generalization_claim"] is False
    assert (
        protocol["resource_contract"][
            "total_logical_traffic_bytes_per_128_token_sequence"
        ]
        == 736_100_352
    )
    assert protocol["authenticated_confirmation_descriptor"] == descriptor
    assert protocol["authenticated_confirmation_descriptor"] is not descriptor
    assert protocol["confirmation_split_opened"] is False


def test_audit_and_deployment_artifacts_round_trip_exactly(
    tmp_path: Path,
) -> None:
    shape = selector.PhaseSelectorShape(
        layers=2,
        heads=2,
        components=3,
        head_dimension=2,
        rank=2,
        phases=2,
    )
    parameters = selector.initialize_parameters(shape, seed=17)
    table = np.linspace(
        -0.5,
        0.5,
        num=shape.phase_parameter_count,
        dtype=np.float32,
    ).reshape(
        shape.phases,
        shape.layers,
        shape.heads,
        shape.components,
    )
    parameters = selector.PhaseSelectorParameters(
        U=parameters.U,
        V=parameters.V,
        E=parameters.E,
        B=parameters.B,
        T=np.ascontiguousarray(table),
    )
    audit_path = tmp_path / "audit.safetensors"
    audit = runner._write_audit_artifact(audit_path, parameters, shape)
    fp32 = runner._load_audit_artifact(
        audit_path,
        audit,
        shape,
        bf16=False,
    )
    bf16 = runner._load_audit_artifact(
        audit_path,
        audit,
        shape,
        bf16=True,
    )
    assert selector.parameters_sha256(fp32, shape) == (
        selector.parameters_sha256(parameters, shape)
    )
    assert audit["BF16_tensor_bytes"] == shape.parameter_count * 2

    deployment_path = tmp_path / "deployment.safetensors"
    deployment = runner._write_deployment_artifact(
        deployment_path,
        parameters,
        shape,
    )
    replay = runner._load_deployment_artifact(
        deployment_path,
        deployment,
        shape,
    )
    assert selector.parameters_sha256(replay, shape) == (
        selector.parameters_sha256(bf16, shape)
    )
    assert deployment["contains_FP32_training_copy"] is False


def test_gate_requires_exact_schedule_and_inactive_equivariance() -> None:
    coefficients = np.array([[[[[0.5, 0.5]]]]], dtype=np.float32)
    valid = np.ones_like(coefficients, dtype=bool)
    metrics = {"passed": True, "global": {"recovery": 0.6}}
    gate = runner._selector_gate(
        metrics,
        metrics,
        fp32_coefficients=coefficients,
        bf16_coefficients=coefficients,
        valid=valid,
        score_coefficient_max_abs=0.0,
        score_delta_max_abs=0.0,
        zero_coefficient_max_abs=0.0,
        zero_output_max_abs=0.0,
        deterministic_replay_exact=True,
        schedule_shift_coefficient_max_abs=0.0,
        schedule_shift_phase_exact=True,
        inactive_disable_coefficient_max_abs=0.0,
    )
    assert gate["passed"] is True
    failed = deepcopy(gate)
    failed = runner._selector_gate(
        metrics,
        metrics,
        fp32_coefficients=coefficients,
        bf16_coefficients=coefficients,
        valid=valid,
        score_coefficient_max_abs=0.0,
        score_delta_max_abs=0.0,
        zero_coefficient_max_abs=0.0,
        zero_output_max_abs=0.0,
        deterministic_replay_exact=True,
        schedule_shift_coefficient_max_abs=1.0e-8,
        schedule_shift_phase_exact=True,
        inactive_disable_coefficient_max_abs=0.0,
    )
    assert failed["schedule_shift_coefficient_equivariance_exact"] is False
    assert failed["passed"] is False
