import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import engram.evaluation.olmoe_native_attention_sweep as sweep
from engram.compiler.olmoe_native import compile_olmoe_native_package
from engram.evaluation.olmoe_native_causal import _write_npz_atomic
from engram.evaluation.olmoe_native_dense_control import (
    evaluate_native_olmoe_dense_attention_control,
    freeze_native_olmoe_dense_attention_control_protocol,
)
from engram.evaluation.olmoe_native_sustained import (
    _TEACHER_CONFIGURATION,
    _THRESHOLDS,
    _attention_expectations,
    _q7_expectations,
    evaluate_native_olmoe_sustained_context,
    freeze_olmoe_sustained_context_protocol,
)
from engram.models.fixture import create_tiny_olmoe_fixture
from engram.models.olmoe_native import repack_olmoe_non_mlp_weights
from engram.models.olmoe_q7 import repack_olmoe_q7_model
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.utils import atomic_json, sha256_file, sha256_json


_PRODUCTION_MODEL = {
    "layers": 16,
    "hidden_size": 2_048,
    "intermediate_size": 1_024,
    "experts": 64,
    "vocab_size": 50_304,
    "query_heads": 16,
    "key_value_heads": 16,
    "head_dimension": 128,
    "top_k": 8,
    "q7_group_size": 64,
}

_EXPECTED_POLICIES = [
    {
        "name": "w16_c18_k16_s2",
        "attention_policy": {
            "local_window": 16,
            "older_candidates": 18,
            "older_top_k": 16,
            "sink_tokens": 2,
        },
    },
    {
        "name": "w24_c10_k8_s2",
        "attention_policy": {
            "local_window": 24,
            "older_candidates": 10,
            "older_top_k": 8,
            "sink_tokens": 2,
        },
    },
    {
        "name": "w30_c4_k2_s2",
        "attention_policy": {
            "local_window": 30,
            "older_candidates": 4,
            "older_top_k": 2,
            "sink_tokens": 2,
        },
    },
]

_EXPECTED_ARM_COUNTERS = {
    "w16_c18_k16_s2": {
        "attention_state_bytes": 8_991_232,
        "attention_scratch_bytes": 7_424,
        "attention_eviction_events": 1_792,
        "attention_older_candidate_entries_scored": 476_928,
        "attention_older_selected_entries": 428_032,
        "attention_sink_insertions": 512,
        "attention_heavy_hitter_updates_minimum": 4_096,
        "attention_heavy_hitter_updates_maximum": 28_160,
        "attention_local_kv_bytes": 505_413_632,
        "attention_candidate_key_bytes": 244_187_136,
        "attention_selected_value_bytes": 219_152_384,
    },
    "w24_c10_k8_s2": {
        "attention_state_bytes": 8_973_824,
        "attention_scratch_bytes": 5_888,
        "attention_eviction_events": 1_664,
        "attention_older_candidate_entries_scored": 254_720,
        "attention_older_selected_entries": 205_824,
        "attention_sink_insertions": 512,
        "attention_heavy_hitter_updates_minimum": 2_048,
        "attention_heavy_hitter_updates_maximum": 26_112,
        "attention_local_kv_bytes": 732_954_624,
        "attention_candidate_key_bytes": 130_416_640,
        "attention_selected_value_bytes": 105_381_888,
    },
    "w30_c4_k2_s2": {
        "attention_state_bytes": 8_960_768,
        "attention_scratch_bytes": 4_736,
        "attention_eviction_events": 1_568,
        "attention_older_candidate_entries_scored": 98_816,
        "attention_older_selected_entries": 49_920,
        "attention_sink_insertions": 512,
        "attention_heavy_hitter_updates_minimum": 512,
        "attention_heavy_hitter_updates_maximum": 24_576,
        "attention_local_kv_bytes": 892_600_320,
        "attention_candidate_key_bytes": 50_593_792,
        "attention_selected_value_bytes": 25_559_040,
    },
}


def _metric_population(
    *,
    kl: float = 0.025,
    top1: float = 0.95,
    nll: float = 0.025,
    hidden: float = 0.05,
) -> dict[str, float]:
    return {
        "teacher_to_native_kl": kl,
        "teacher_top1_agreement": top1,
        "target_nll_delta": nll,
        "final_hidden_relative_l2": hidden,
    }


def _position_rows() -> list[dict[str, int | float]]:
    return [
        {
            "sequence": sequence,
            "position": position,
            "teacher_to_native_kl": (sequence * 128 + position) / 100_000,
        }
        for sequence in range(8)
        for position in range(128)
    ]


def test_production_arm_policies_state_and_aggregate_counters_are_exact():
    assert sweep._arms() == _EXPECTED_POLICIES
    assert sweep._per_position_read_contract() == {
        "key_rows": "min(position + 1, 34)",
        "value_rows": "min(position + 1, 32)",
        "first_full_causal_value_omission_offset": 32,
        "row_bytes": 131_072,
    }

    descriptors = sweep._arm_descriptors(_PRODUCTION_MODEL)
    assert [descriptor["ordinal"] for descriptor in descriptors] == [0, 1, 2]
    assert [descriptor["name"] for descriptor in descriptors] == [
        arm["name"] for arm in _EXPECTED_POLICIES
    ]
    for descriptor in descriptors:
        assert descriptor["mature_visible_values"] == 32
        assert descriptor["mature_visible_key_rows"] == 34
        expectations = descriptor["attention_expectations_per_sequence"]
        assert expectations["positions_processed"] == 128
        assert expectations["attention_logical_read_bytes"] == 968_753_152
        assert expectations["dense_full_context_logical_kv_bytes"] == 2_164_260_864
        assert expectations["attention_logical_read_fraction"] == pytest.approx(
            0.44761385658914726
        )
        for name, expected in _EXPECTED_ARM_COUNTERS[descriptor["name"]].items():
            assert expectations[name] == expected


def test_all_arms_match_every_per_position_read_delta_and_aggregate():
    row_bytes = 131_072
    for arm in _EXPECTED_POLICIES:
        previous = 0
        policy = arm["attention_policy"]
        for position in range(1, 129):
            expectations = _attention_expectations(
                _PRODUCTION_MODEL,
                policy,
                positions=position,
            )
            logical_bytes = int(expectations["attention_logical_read_bytes"])
            expected_delta = row_bytes * (min(position, 34) + min(position, 32))
            assert logical_bytes - previous == expected_delta
            previous = logical_bytes
        assert previous == 968_753_152


def test_final_and_incremental_structural_counter_checks_are_exact():
    q7 = _q7_expectations(_PRODUCTION_MODEL)
    for descriptor in sweep._arm_descriptors(_PRODUCTION_MODEL):
        expectations = descriptor["attention_expectations_per_sequence"]
        metrics = {
            name: int(value)
            for name, value in expectations.items()
            if isinstance(value, int)
        }
        metrics["attention_heavy_hitter_updates"] = int(
            expectations["attention_heavy_hitter_updates_minimum"]
        )
        metrics["q7_scheduled_bytes"] = q7["scheduled_bytes_per_sequence"]
        assert all(
            sweep._structural_checks(
                metrics,
                expectations,
                q7,
                position=128,
            ).values()
        )
        wrong_cache_position = sweep._structural_checks(
            metrics,
            expectations,
            q7,
            position=127,
        )
        assert wrong_cache_position["cache_position"] is False
        assert sum(not passed for passed in wrong_cache_position.values()) == 1

        policy = descriptor["attention_policy"]
        for position in (1, 16, 32, 33, 34, 128):
            cumulative = _attention_expectations(
                _PRODUCTION_MODEL,
                policy,
                positions=position,
            )
            snapshot = {
                name: int(value)
                for name, value in cumulative.items()
                if isinstance(value, int)
            }
            snapshot["attention_heavy_hitter_updates"] = int(
                cumulative["attention_heavy_hitter_updates_minimum"]
            )
            snapshot["q7_scheduled_bytes"] = (
                position * q7["scheduled_bytes_per_position"]
            )
            previous = (
                0
                if position == 1
                else int(
                    _attention_expectations(
                        _PRODUCTION_MODEL,
                        policy,
                        positions=position - 1,
                    )["attention_logical_read_bytes"]
                )
            )
            delta = 131_072 * (min(position, 34) + min(position, 32))
            checks = sweep._counter_snapshot_checks(
                snapshot,
                cumulative,
                q7,
                position=position,
                previous_logical_read_bytes=previous,
                expected_logical_read_delta_bytes=delta,
            )
            assert all(checks.values())

            wrong_delta = sweep._counter_snapshot_checks(
                snapshot,
                cumulative,
                q7,
                position=position,
                previous_logical_read_bytes=previous,
                expected_logical_read_delta_bytes=delta + 1,
            )
            assert wrong_delta["logical_read_delta_bytes"] is False


def test_q7_traffic_fraction_uses_all_expert_ideal_q4_reference():
    q7 = _q7_expectations(_PRODUCTION_MODEL)
    prediction_positions = 8 * 128
    q7_scheduled_bytes = 8 * q7["scheduled_bytes_per_sequence"]
    all_expert_ideal_q4_bytes = prediction_positions * (
        _PRODUCTION_MODEL["layers"]
        * _PRODUCTION_MODEL["experts"]
        * 3
        * _PRODUCTION_MODEL["hidden_size"]
        * _PRODUCTION_MODEL["intermediate_size"]
        // 2
    )
    q7_fraction = q7_scheduled_bytes / all_expert_ideal_q4_bytes

    assert q7_scheduled_bytes == 751_619_276_800
    assert all_expert_ideal_q4_bytes == 3_298_534_883_328
    assert q7_fraction == pytest.approx(0.22786458333333334)
    assert q7_fraction <= _THRESHOLDS["maximum_q7_traffic_fraction"]


def test_quality_margin_formula_excludes_common_prefix_and_finds_worst_metric():
    bands = {
        "positions_0_15": _metric_population(
            kl=10.0,
            top1=0.0,
            nll=10.0,
            hidden=10.0,
        ),
        "positions_16_31": _metric_population(top1=0.91),
        "positions_32_63": _metric_population(),
        "positions_64_95": _metric_population(),
        "positions_96_127": _metric_population(),
    }
    margin = sweep._quality_margin(_metric_population(), bands)

    assert "positions_0_15" not in margin["normalized_margins"]
    assert margin["normalized_margins"]["overall"] == pytest.approx(
        {
            "teacher_to_native_kl": 0.5,
            "teacher_top1_agreement": 0.5,
            "target_nll_delta": 0.5,
            "final_hidden_relative_l2": 0.5,
        }
    )
    assert margin["worst"]["population"] == "positions_16_31"
    assert margin["worst"]["metric"] == "teacher_top1_agreement"
    assert margin["worst"]["normalized_margin"] == pytest.approx(0.1)
    assert margin["worst_normalized_quality_margin"] == pytest.approx(0.1)
    assert sweep._ranking_rule()["ordering"] == [
        "descending_worst_normalized_quality_margin",
        "ascending_attention_state_bytes",
        "frozen_arm_order",
    ]


@pytest.mark.parametrize("local_window", [16, 24, 30])
def test_pre_eviction_identity_is_exact_and_rejects_prefix_tamper(local_window):
    control = _position_rows()
    arm = deepcopy(control)
    identity = sweep._pre_eviction_identity(
        control,
        arm,
        local_window=local_window,
    )
    assert identity == {
        "local_window": local_window,
        "expected_positions": 8 * local_window,
        "control_positions": 8 * local_window,
        "arm_positions": 8 * local_window,
        "exact_position_metrics_match": True,
    }

    after_prefix = deepcopy(arm)
    after_prefix[local_window]["teacher_to_native_kl"] += 1.0
    assert sweep._pre_eviction_identity(
        control,
        after_prefix,
        local_window=local_window,
    )["exact_position_metrics_match"]

    within_prefix = deepcopy(arm)
    within_prefix[local_window - 1]["teacher_to_native_kl"] += 1.0
    assert not sweep._pre_eviction_identity(
        control,
        within_prefix,
        local_window=local_window,
    )["exact_position_metrics_match"]


def _protocol_context_and_value() -> tuple[dict, dict, str, str]:
    protocol_hash = "protocol-hash"
    sweep_source_hash = "sweep-source-hash"
    hashes = {
        "sustained_protocol_sha256": "sustained-protocol",
        "sustained_result_sha256": "sustained-result",
        "control_protocol_sha256": "control-protocol",
        "control_result_sha256": "control-result",
    }
    identities = {
        "package_manifest_sha256": "package",
        "native_library_sha256": "library",
        "dataset_sha256": "dataset",
        "corpus_manifest_sha256": "corpus",
        "teacher_reference_sha256": "teacher-reference",
        "teacher_arrays_sha256": "teacher-arrays",
    }
    input_ids = [[sequence] + [1] * 128 for sequence in range(8)]
    sustained_protocol = {
        "source_revision": "revision",
        "source_config_sha256": "config",
        "source_index_sha256": "index",
        "source_shard_sha256": {"shard": "hash"},
        "input_identity": sha256_json(input_ids),
    }
    context = {
        "sustained_protocol": sustained_protocol,
        "hashes": hashes,
        "identities": identities,
        "control_source_hash": "control-source",
        "evaluator_sources": {"source.py": "source-hash"},
        "input_ids": input_ids,
        "model": _PRODUCTION_MODEL,
        "q7_expectations": _q7_expectations(_PRODUCTION_MODEL),
    }
    value = {
        "schema_version": 1,
        "experiment": "olmoe_native_q7_bounded_attention_development_sweep",
        "status": "frozen_after_dense_attribution_before_sweep_execution",
        "source_revision": "revision",
        **hashes,
        **identities,
        "control_source_sha256": "control-source",
        "sweep_source_sha256": sweep_source_hash,
        "frozen_evaluator_source_sha256": {"source.py": "source-hash"},
        "source_config_sha256": "config",
        "source_index_sha256": "index",
        "source_shard_sha256": {"shard": "hash"},
        "input_identity": sha256_json(input_ids),
        "input_ids": input_ids,
        "model": _PRODUCTION_MODEL,
        "q7_expectations_per_sequence": _q7_expectations(_PRODUCTION_MODEL),
        "quality_bands": sweep._expected_bands(),
        "thresholds": _THRESHOLDS,
        "arms": sweep._arm_descriptors(_PRODUCTION_MODEL),
        "per_position_read_contract": sweep._per_position_read_contract(),
        "ranking_rule": sweep._ranking_rule(),
        "scope": {
            "candidate_device": "cpu",
            "candidate_threads": 12,
            "candidate_transformers_model_shell": False,
            "execution_interface": sweep._EXECUTION_INTERFACE,
            "source_package_attention_policy": (sweep._SOURCE_PACKAGE_ATTENTION_POLICY),
            "attention_policy_overridden_for_development": True,
            "package_manifest_mutated": False,
            "arms_execute_sequentially_in_frozen_order": True,
            "intermediate_outputs_inspected_or_used_to_adapt_later_arms": False,
            "q7_artifact_or_policy_changed": False,
            "corpus_or_teacher_changed": False,
            "mature_visible_values_per_arm": 32,
            "mature_visible_key_rows_per_arm": 34,
            "attention_logical_read_bytes_per_sequence": 968_753_152,
            "maximum_attention_logical_read_fraction": 0.45,
            "reset_replay_sequence_per_arm": 0,
            "development_selection_only": True,
            "fresh_confirmation_required": True,
            "protocol_frozen_before_any_arm_execution": True,
            "dense_attribution_result_known_before_sweep_freeze": True,
            "teacher_configuration": _TEACHER_CONFIGURATION,
        },
    }
    return context, value, protocol_hash, sweep_source_hash


def test_sweep_protocol_contract_rejects_policy_scope_and_hash_tamper():
    context, value, protocol_hash, sweep_source_hash = _protocol_context_and_value()
    assert (
        sweep._validate_sweep_protocol(
            value,
            context,
            sweep_protocol_hash=protocol_hash,
            supplied_sweep_protocol_hash=protocol_hash,
            sweep_source_hash=sweep_source_hash,
        )
        is None
    )

    bad_policy = deepcopy(value)
    bad_policy["arms"][0]["attention_policy"]["older_top_k"] = 15
    with pytest.raises(ValueError, match="protocol contract"):
        sweep._validate_sweep_protocol(
            bad_policy,
            context,
            sweep_protocol_hash=protocol_hash,
            supplied_sweep_protocol_hash=protocol_hash,
            sweep_source_hash=sweep_source_hash,
        )

    bad_scope = deepcopy(value)
    bad_scope["scope"]["attention_policy_overridden_for_development"] = False
    with pytest.raises(ValueError, match="protocol contract"):
        sweep._validate_sweep_protocol(
            bad_scope,
            context,
            sweep_protocol_hash=protocol_hash,
            supplied_sweep_protocol_hash=protocol_hash,
            sweep_source_hash=sweep_source_hash,
        )

    with pytest.raises(ValueError, match="protocol contract"):
        sweep._validate_sweep_protocol(
            value,
            context,
            sweep_protocol_hash="changed",
            supplied_sweep_protocol_hash=protocol_hash,
            sweep_source_hash=sweep_source_hash,
        )


def test_protocol_and_result_outputs_are_non_overwriting(tmp_path):
    existing = tmp_path / "existing.json"
    existing.write_text("preserve", encoding="utf-8")
    common = {
        "package": tmp_path / "package",
        "manifest_sha256": "manifest",
        "library": tmp_path / "library.so",
        "dataset": tmp_path / "dataset.jsonl",
        "corpus_manifest": tmp_path / "corpus.json",
        "teacher_reference": tmp_path / "teacher.json",
        "teacher_arrays": tmp_path / "teacher.npz",
        "sustained_protocol": tmp_path / "sustained-protocol.json",
        "sustained_protocol_sha256": "sustained-protocol",
        "sustained_result": tmp_path / "sustained-result.json",
        "sustained_result_sha256": "sustained-result",
        "control_protocol": tmp_path / "control-protocol.json",
        "control_protocol_sha256": "control-protocol",
        "control_result": tmp_path / "control-result.json",
        "control_result_sha256": "control-result",
        "out": existing,
        "threads": 12,
    }
    with pytest.raises(ValueError, match="protocol target already exists"):
        sweep.freeze_native_olmoe_attention_sweep_protocol(**common)
    with pytest.raises(ValueError, match="result target already exists"):
        sweep.evaluate_native_olmoe_attention_sweep(
            **common,
            sweep_protocol=tmp_path / "sweep-protocol.json",
            sweep_protocol_sha256="sweep-protocol",
        )
    assert existing.read_text(encoding="utf-8") == "preserve"


def _evaluate_with_mocked_arms(
    tmp_path,
    monkeypatch,
    *,
    quality_passed=(True, True, True),
    evidence_passed=(True, True, True),
    state_bytes=(200, 100, 100),
    post_authentication=None,
):
    input_ids = [[0] * 129 for _ in range(8)]
    arrays_path = tmp_path / "teacher.npz"
    np.savez(
        arrays_path,
        logits=np.zeros((1_024, 2), dtype=np.float32),
        hidden=np.zeros((1_024, 2), dtype=np.float32),
        targets=np.zeros(1_024, dtype=np.int64),
    )
    protocol_path = tmp_path / "sweep-protocol.json"
    atomic_json(protocol_path, {"arms": _EXPECTED_POLICIES})
    context = {
        "input_ids": input_ids,
        "model": {"vocab_size": 2, "hidden_size": 2},
        "hashes": {},
        "identities": {},
        "control_source_hash": "control",
    }
    monkeypatch.setattr(
        sweep,
        "_authenticate_prerequisites",
        lambda **_kwargs: context,
    )
    monkeypatch.setattr(
        sweep,
        "_validate_sweep_protocol",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        sweep,
        "_post_authentication",
        lambda **_kwargs: (
            {"all": True} if post_authentication is None else post_authentication
        ),
    )

    execution_order = []

    def fake_evaluate_arm(descriptor, **_kwargs):
        ordinal = next(
            index
            for index, arm in enumerate(_EXPECTED_POLICIES)
            if arm["name"] == descriptor["name"]
        )
        execution_order.append(descriptor["name"])
        return {
            "ordinal": ordinal,
            "name": descriptor["name"],
            "quality_passed": quality_passed[ordinal],
            "quality_margin": {"worst_normalized_quality_margin": 0.25},
            "attention_expectations_per_sequence": {
                "attention_state_bytes": state_bytes[ordinal],
            },
            "evidence_checks": {"local": evidence_passed[ordinal]},
            "performance": {
                "total_sequence_seconds": 0.0,
                "reset_replay_seconds": 0.0,
            },
        }

    monkeypatch.setattr(sweep, "_evaluate_arm", fake_evaluate_arm)
    result = sweep.evaluate_native_olmoe_attention_sweep(
        package=tmp_path / "package",
        manifest_sha256="manifest",
        library=tmp_path / "library.so",
        dataset=tmp_path / "dataset.jsonl",
        corpus_manifest=tmp_path / "corpus.json",
        teacher_reference=tmp_path / "teacher.json",
        teacher_arrays=arrays_path,
        sustained_protocol=tmp_path / "sustained-protocol.json",
        sustained_protocol_sha256="sustained-protocol",
        sustained_result=tmp_path / "sustained-result.json",
        sustained_result_sha256="sustained-result",
        control_protocol=tmp_path / "control-protocol.json",
        control_protocol_sha256="control-protocol",
        control_result=tmp_path / "control-result.json",
        control_result_sha256="control-result",
        sweep_protocol=protocol_path,
        sweep_protocol_sha256=sha256_file(protocol_path),
        out=tmp_path / "result.json",
        threads=12,
    )
    return result, execution_order


def test_ranking_ties_use_state_bytes_then_frozen_order(tmp_path, monkeypatch):
    result, execution_order = _evaluate_with_mocked_arms(
        tmp_path,
        monkeypatch,
    )

    assert execution_order == [arm["name"] for arm in _EXPECTED_POLICIES]
    assert [entry["name"] for entry in result["ranking"]] == [
        "w24_c10_k8_s2",
        "w30_c4_k2_s2",
        "w16_c18_k16_s2",
    ]
    assert result["selected_arm"] == "w24_c10_k8_s2"
    assert result["diagnostic_quality_passing_arm_count"] == 3
    assert result["decision"] == (
        "integrate_selected_attention_policy_then_freeze_fresh_"
        "package_native_confirmation"
    )
    for section in (result["provenance"], result["configuration"]):
        assert section["execution_interface"] == "raw_native_token_runtime"
        assert section["source_package_attention_policy"] == {
            "local_window": 16,
            "older_candidates": 8,
            "older_top_k": 4,
            "sink_tokens": 2,
        }
        assert section["attention_policy_overridden_for_development"] is True
        assert section["package_manifest_mutated"] is False


def test_all_arms_execute_before_no_quality_pass_decision(tmp_path, monkeypatch):
    result, execution_order = _evaluate_with_mocked_arms(
        tmp_path,
        monkeypatch,
        quality_passed=(False, False, False),
    )

    assert execution_order == [arm["name"] for arm in _EXPECTED_POLICIES]
    assert result["status"] == "development_sweep_complete"
    assert result["evidence_passed"] is True
    assert result["diagnostic_quality_passing_arm_count"] == 0
    assert result["ranking"] == []
    assert result["selected_arm"] is None
    assert result["selection_is_development_only"] is False
    assert result["fresh_confirmation_required"] is False
    assert result["decision"] == "investigate_layer_adaptive_or_learned_selector"


def test_arm_evidence_failure_invalidates_selection_after_all_arms(
    tmp_path,
    monkeypatch,
):
    result, execution_order = _evaluate_with_mocked_arms(
        tmp_path,
        monkeypatch,
        evidence_passed=(True, False, True),
    )

    assert execution_order == [arm["name"] for arm in _EXPECTED_POLICIES]
    assert result["status"] == "development_sweep_invalid"
    assert result["evidence_passed"] is False
    assert result["diagnostic_quality_passing_arm_count"] == 3
    assert result["ranking"] == []
    assert result["selected_arm"] is None
    assert result["decision"] == "stop_and_diagnose_evidence"
    assert result["arm_results"][1]["evidence_passed"] is False


def test_post_authentication_failure_invalidates_every_arm(
    tmp_path,
    monkeypatch,
):
    result, execution_order = _evaluate_with_mocked_arms(
        tmp_path,
        monkeypatch,
        post_authentication={"package_manifest": True, "sweep_source": False},
    )

    assert execution_order == [arm["name"] for arm in _EXPECTED_POLICIES]
    assert result["status"] == "development_sweep_invalid"
    assert result["evidence_passed"] is False
    assert result["ranking"] == []
    assert result["selected_arm"] is None
    assert result["decision"] == "stop_and_diagnose_evidence"
    assert all(
        arm["evidence_checks"]["post_run_authentication"] is False
        and arm["evidence_passed"] is False
        for arm in result["arm_results"]
    )


def _common_cli_arguments(tmp_path, *, out):
    return [
        "--package",
        str(tmp_path / "package"),
        "--manifest-sha256",
        "manifest",
        "--library",
        str(tmp_path / "library.so"),
        "--dataset",
        str(tmp_path / "dataset.jsonl"),
        "--corpus-manifest",
        str(tmp_path / "corpus.json"),
        "--teacher-reference",
        str(tmp_path / "teacher.json"),
        "--teacher-arrays",
        str(tmp_path / "teacher.npz"),
        "--sustained-protocol",
        str(tmp_path / "sustained-protocol.json"),
        "--sustained-protocol-sha256",
        "sustained-protocol",
        "--sustained-result",
        str(tmp_path / "sustained-result.json"),
        "--sustained-result-sha256",
        "sustained-result",
        "--control-protocol",
        str(tmp_path / "control-protocol.json"),
        "--control-protocol-sha256",
        "control-protocol",
        "--control-result",
        str(tmp_path / "control-result.json"),
        "--control-result-sha256",
        "control-result",
        "--out",
        str(out),
        "--threads",
        "12",
    ]


def test_cli_parses_freeze_and_evaluate_commands(
    tmp_path,
    monkeypatch,
    capsys,
):
    freeze_calls = []

    def fake_freeze(**kwargs):
        freeze_calls.append(kwargs)
        return {
            "status": "frozen",
            "arms": deepcopy(_EXPECTED_POLICIES),
        }

    monkeypatch.setattr(
        sweep,
        "freeze_native_olmoe_attention_sweep_protocol",
        fake_freeze,
    )
    monkeypatch.setattr(sweep, "sha256_file", lambda _path: "output-hash")
    freeze_out = tmp_path / "frozen.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "olmoe-native-attention-sweep",
            "freeze",
            *_common_cli_arguments(tmp_path, out=freeze_out),
        ],
    )
    assert sweep._main() == 0
    freeze_output = json.loads(capsys.readouterr().out)
    assert freeze_output == {
        "status": "frozen",
        "arms": deepcopy(_EXPECTED_POLICIES),
        "protocol_sha256": "output-hash",
    }
    assert freeze_calls[0]["out"] == freeze_out
    assert freeze_calls[0]["threads"] == 12

    evaluate_calls = []

    def fake_evaluate(**kwargs):
        evaluate_calls.append(kwargs)
        return {
            "status": "development_sweep_invalid",
            "evidence_passed": False,
            "diagnostic_quality_passing_arm_count": 0,
            "selected_arm": None,
            "decision": "stop_and_diagnose_evidence",
            "ranking": [],
        }

    monkeypatch.setattr(
        sweep,
        "evaluate_native_olmoe_attention_sweep",
        fake_evaluate,
    )
    evaluate_out = tmp_path / "result.json"
    sweep_protocol = tmp_path / "sweep-protocol.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "olmoe-native-attention-sweep",
            "evaluate",
            *_common_cli_arguments(tmp_path, out=evaluate_out),
            "--sweep-protocol",
            str(sweep_protocol),
            "--sweep-protocol-sha256",
            "sweep-protocol",
        ],
    )
    assert sweep._main() == 2
    evaluate_output = json.loads(capsys.readouterr().out)
    assert evaluate_output == {
        "status": "development_sweep_invalid",
        "evidence_passed": False,
        "diagnostic_quality_passing_arm_count": 0,
        "selected_arm": None,
        "decision": "stop_and_diagnose_evidence",
        "ranking": [],
        "result_sha256": "output-hash",
    }
    assert evaluate_calls[0]["out"] == evaluate_out
    assert evaluate_calls[0]["sweep_protocol"] == sweep_protocol
    assert evaluate_calls[0]["sweep_protocol_sha256"] == "sweep-protocol"
    assert evaluate_calls[0]["threads"] == 12


def _build_tiny_control_prerequisites(
    tmp_path: Path,
    library: Path,
) -> dict[str, Path | str | dict]:
    model = create_tiny_olmoe_fixture(
        tmp_path / "model",
        num_experts=64,
        num_experts_per_token=1,
    )
    vocabulary = {"[UNK]": 0, "x": 1}
    vocabulary.update({f"marker{index}": index + 2 for index in range(8)})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(model / "tokenizer.json"))
    q7 = repack_olmoe_q7_model(model, tmp_path / "model.q7", group_size=64)
    non_mlp = tmp_path / "non_mlp.safetensors"
    repack_olmoe_non_mlp_weights(model, non_mlp)
    package = tmp_path / "package"
    compiled = compile_olmoe_native_package(
        model,
        q7,
        non_mlp,
        package,
        kernel_threads=12,
    )
    manifest_hash = compiled["manifest_sha256"]

    dataset = tmp_path / "sustained.jsonl"
    records = [
        {
            "record_id": f"fixture-{index}",
            "source_kind": "engram_authored_holdout",
            "domain": f"fixture-domain-{index}",
            "text": " ".join([f"marker{index}", *(["x"] * 128)]),
        }
        for index in range(8)
    ]
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    inputs = [tokenizer.encode(record["text"]).ids[:129] for record in records]
    corpus_manifest = {
        "schema_version": 1,
        "experiment": "olmoe_sustained_context_authored_holdout",
        "status": "authored_and_fixed_before_candidate_execution",
        "created_utc": "2026-07-28",
        "dataset_sha256": sha256_file(dataset),
        "tokenizer_sha256": sha256_file(model / "tokenizer.json"),
        "input_identity": sha256_json(inputs),
        "sequences": 8,
        "tokens_per_sequence": 129,
        "selection": {
            "source_kind": "engram_authored_holdout",
            "rule": "fixed attention-sweep fixture records",
            "candidate_or_teacher_outputs_inspected_during_selection": False,
            "previous_engram_calibration_or_confirmation_text_reused": False,
        },
        "records": [
            {
                "record_id": record["record_id"],
                "domain": record["domain"],
                "full_token_count": len(tokenizer.encode(record["text"]).ids),
                "window_identity": sha256_json({"input_ids": inputs[index]}),
            }
            for index, record in enumerate(records)
        ],
    }
    corpus_manifest_path = tmp_path / "corpus-manifest.json"
    atomic_json(corpus_manifest_path, corpus_manifest)

    logits: list[np.ndarray] = []
    hidden: list[np.ndarray] = []
    targets: list[int] = []
    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=12,
        local_window=128,
        older_candidates=8,
        older_top_k=4,
        sink_tokens=2,
    ) as runtime:
        for sequence in inputs:
            runtime.reset()
            for position, token_id in enumerate(sequence[:-1]):
                runtime.forward([token_id])
                state, scores = runtime.last_diagnostics()
                hidden.append(state)
                logits.append(scores)
                targets.append(sequence[position + 1])
    arrays_path = tmp_path / "teacher.npz"
    _write_npz_atomic(
        arrays_path,
        logits=np.asarray(logits, dtype=np.float32),
        hidden=np.asarray(hidden, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.int64),
    )
    reference = {
        "schema_version": 1,
        "experiment": "olmoe_untouched_teacher_causal_reference",
        "source": {
            "model": str(model),
            "revision": None,
            "config_sha256": sha256_file(model / "config.json"),
            "index_sha256": sha256_file(model / "model.safetensors.index.json"),
            "adapter": "olmoe_sparse_expert_v1",
        },
        "dataset": {
            "path": str(dataset),
            "sha256": sha256_file(dataset),
            "sequences": 8,
            "tokens_per_sequence": 129,
            "prediction_positions": 1_024,
            "input_identity": sha256_json(inputs),
            "input_ids": inputs,
        },
        "configuration": dict(_TEACHER_CONFIGURATION),
        "arrays": {
            "path": str(arrays_path),
            "sha256": sha256_file(arrays_path),
        },
    }
    reference_path = tmp_path / "teacher.json"
    atomic_json(reference_path, reference)

    sustained_protocol_path = tmp_path / "sustained-protocol.json"
    sustained_protocol = freeze_olmoe_sustained_context_protocol(
        package=package,
        manifest_sha256=manifest_hash,
        library=library,
        dataset=dataset,
        corpus_manifest=corpus_manifest_path,
        teacher_reference=reference_path,
        teacher_arrays=arrays_path,
        out=sustained_protocol_path,
        threads=12,
    )
    failed_result_path = tmp_path / "sustained-result.json"
    sustained_result = evaluate_native_olmoe_sustained_context(
        package=package,
        manifest_sha256=manifest_hash,
        library=library,
        dataset=dataset,
        corpus_manifest=corpus_manifest_path,
        teacher_reference=reference_path,
        teacher_arrays=arrays_path,
        protocol=sustained_protocol_path,
        protocol_sha256=sha256_file(sustained_protocol_path),
        out=failed_result_path,
        threads=12,
    )
    assert sustained_result["evidence_passed"]
    assert not sustained_result["quality_passed"]
    assert not sustained_result["gate_passed"]
    assert sustained_result["decision"] == "run_matched_q7_dense_attention_control"

    control_protocol_path = tmp_path / "control-protocol.json"
    freeze_native_olmoe_dense_attention_control_protocol(
        package=package,
        manifest_sha256=manifest_hash,
        library=library,
        dataset=dataset,
        corpus_manifest=corpus_manifest_path,
        teacher_reference=reference_path,
        teacher_arrays=arrays_path,
        sustained_protocol=sustained_protocol_path,
        sustained_protocol_sha256=sha256_file(sustained_protocol_path),
        sustained_result=failed_result_path,
        sustained_result_sha256=sha256_file(failed_result_path),
        out=control_protocol_path,
        threads=12,
    )
    control_result_path = tmp_path / "control-result.json"
    control_result = evaluate_native_olmoe_dense_attention_control(
        package=package,
        manifest_sha256=manifest_hash,
        library=library,
        dataset=dataset,
        corpus_manifest=corpus_manifest_path,
        teacher_reference=reference_path,
        teacher_arrays=arrays_path,
        sustained_protocol=sustained_protocol_path,
        sustained_protocol_sha256=sha256_file(sustained_protocol_path),
        sustained_result=failed_result_path,
        sustained_result_sha256=sha256_file(failed_result_path),
        control_protocol=control_protocol_path,
        control_protocol_sha256=sha256_file(control_protocol_path),
        out=control_result_path,
        threads=12,
    )
    assert control_result["evidence_passed"]
    assert control_result["quality_passed"]
    return {
        "package": package,
        "manifest_hash": manifest_hash,
        "library": library,
        "dataset": dataset,
        "corpus_manifest": corpus_manifest_path,
        "teacher_reference": reference_path,
        "teacher_arrays": arrays_path,
        "sustained_protocol": sustained_protocol_path,
        "sustained_result": failed_result_path,
        "control_protocol": control_protocol_path,
        "control_result": control_result_path,
        "model": sustained_protocol["model"],
    }


def _sweep_arguments(
    artifacts: dict[str, Path | str | dict],
    *,
    out: Path,
) -> dict[str, Path | str | int]:
    sustained_protocol = Path(artifacts["sustained_protocol"])
    sustained_result = Path(artifacts["sustained_result"])
    control_protocol = Path(artifacts["control_protocol"])
    control_result = Path(artifacts["control_result"])
    return {
        "package": Path(artifacts["package"]),
        "manifest_sha256": str(artifacts["manifest_hash"]),
        "library": Path(artifacts["library"]),
        "dataset": Path(artifacts["dataset"]),
        "corpus_manifest": Path(artifacts["corpus_manifest"]),
        "teacher_reference": Path(artifacts["teacher_reference"]),
        "teacher_arrays": Path(artifacts["teacher_arrays"]),
        "sustained_protocol": sustained_protocol,
        "sustained_protocol_sha256": sha256_file(sustained_protocol),
        "sustained_result": sustained_result,
        "sustained_result_sha256": sha256_file(sustained_result),
        "control_protocol": control_protocol,
        "control_protocol_sha256": sha256_file(control_protocol),
        "control_result": control_result,
        "control_result_sha256": sha256_file(control_result),
        "out": out,
        "threads": 12,
    }


def test_tiny_native_freeze_and_three_arm_sweep_smoke(tmp_path, monkeypatch):
    library = Path("build/libengram_olmoe_token_runtime.so")
    if not library.is_file():
        pytest.skip("native OLMoE token runtime has not been built")
    artifacts = _build_tiny_control_prerequisites(tmp_path, library)
    model = artifacts["model"]
    assert isinstance(model, dict)

    row_bytes = (
        int(model["layers"])
        * int(model["query_heads"])
        * int(model["head_dimension"])
        * 4
    )
    tiny_expectations = _attention_expectations(
        model,
        _EXPECTED_POLICIES[0]["attention_policy"],
    )
    expected_logical_bytes = int(tiny_expectations["attention_logical_read_bytes"])
    expected_fraction = float(tiny_expectations["attention_logical_read_fraction"])
    monkeypatch.setattr(
        sweep,
        "_EXPECTED_LOGICAL_READ_BYTES",
        expected_logical_bytes,
    )
    monkeypatch.setattr(
        sweep,
        "_EXPECTED_LOGICAL_READ_FRACTION",
        expected_fraction,
    )
    monkeypatch.setattr(
        sweep,
        "_per_position_read_contract",
        lambda: {
            "key_rows": "min(position + 1, 34)",
            "value_rows": "min(position + 1, 32)",
            "first_full_causal_value_omission_offset": 32,
            "row_bytes": row_bytes,
        },
    )

    tampered_control = json.loads(
        Path(artifacts["control_result"]).read_text(encoding="utf-8")
    )
    tampered_control["evidence_checks"]["prediction_positions"] = False
    tampered_control_path = tmp_path / "tampered-control-result.json"
    atomic_json(tampered_control_path, tampered_control)
    bad_arguments = _sweep_arguments(
        artifacts,
        out=tmp_path / "bad-sweep-protocol.json",
    )
    bad_arguments["control_result"] = tampered_control_path
    bad_arguments["control_result_sha256"] = sha256_file(tampered_control_path)
    with pytest.raises(ValueError, match="control result prerequisite"):
        sweep.freeze_native_olmoe_attention_sweep_protocol(**bad_arguments)

    sweep_protocol_path = tmp_path / "sweep-protocol.json"
    sweep_protocol = sweep.freeze_native_olmoe_attention_sweep_protocol(
        **_sweep_arguments(artifacts, out=sweep_protocol_path)
    )
    assert [arm["name"] for arm in sweep_protocol["arms"]] == [
        arm["name"] for arm in _EXPECTED_POLICIES
    ]
    assert sweep_protocol["scope"]["execution_interface"] == (
        "raw_native_token_runtime"
    )
    assert sweep_protocol["scope"]["source_package_attention_policy"] == {
        "local_window": 16,
        "older_candidates": 8,
        "older_top_k": 4,
        "sink_tokens": 2,
    }
    assert (
        sweep_protocol["scope"]["attention_policy_overridden_for_development"] is True
    )
    assert sweep_protocol["scope"]["package_manifest_mutated"] is False

    result = sweep.evaluate_native_olmoe_attention_sweep(
        **_sweep_arguments(artifacts, out=tmp_path / "sweep-result.json"),
        sweep_protocol=sweep_protocol_path,
        sweep_protocol_sha256=sha256_file(sweep_protocol_path),
    )

    assert result["evidence_passed"]
    assert result["configuration"]["execution_order"] == [
        arm["name"] for arm in _EXPECTED_POLICIES
    ]
    for section in (result["provenance"], result["configuration"]):
        assert section["execution_interface"] == "raw_native_token_runtime"
        assert section["source_package_attention_policy"] == {
            "local_window": 16,
            "older_candidates": 8,
            "older_top_k": 4,
            "sink_tokens": 2,
        }
        assert section["attention_policy_overridden_for_development"] is True
        assert section["package_manifest_mutated"] is False
    assert all(result["post_run_authentication"].values())
    expected_q7_bytes = 8 * _q7_expectations(model)["scheduled_bytes_per_sequence"]
    all_expert_ideal_q4_bytes = (
        8
        * 128
        * (
            int(model["layers"])
            * int(model["experts"])
            * 3
            * int(model["hidden_size"])
            * int(model["intermediate_size"])
            // 2
        )
    )
    expected_q7_fraction = expected_q7_bytes / all_expert_ideal_q4_bytes
    for arm_result, expected in zip(
        result["arm_results"],
        _EXPECTED_POLICIES,
        strict=True,
    ):
        assert arm_result["name"] == expected["name"]
        assert arm_result["evidence_passed"]
        assert all(arm_result["evidence_checks"].values())
        assert arm_result["reset_replay"]["passed"]
        assert all(
            sequence["counter_stream_passed"] and sequence["structural_passed"]
            for sequence in arm_result["sequence_results"]
        )
        local_window = expected["attention_policy"]["local_window"]
        assert arm_result["pre_eviction_identity"] == {
            "local_window": local_window,
            "expected_positions": 8 * local_window,
            "control_positions": 8 * local_window,
            "arm_positions": 8 * local_window,
            "exact_position_metrics_match": True,
        }
        assert (
            arm_result["traffic"]["attention_logical_read_bytes_per_sequence"]
            == expected_logical_bytes
        )
        assert arm_result["traffic"][
            "attention_logical_read_fraction"
        ] == pytest.approx(expected_fraction)
        assert arm_result["traffic"]["q7_scheduled_bytes"] == expected_q7_bytes
        assert (
            arm_result["traffic"]["all_expert_ideal_q4_bytes"]
            == all_expert_ideal_q4_bytes
        )
        assert arm_result["traffic"][
            "q7_fraction_of_all_expert_ideal_q4"
        ] == pytest.approx(expected_q7_fraction)
        assert (
            arm_result["traffic"]["q7_fraction_of_all_expert_ideal_q4"]
            <= _THRESHOLDS["maximum_q7_traffic_fraction"]
        )
        assert arm_result["evidence_checks"]["q7_traffic_fraction"] is True

    tampered_protocol = deepcopy(sweep_protocol)
    tampered_protocol["arms"][0]["attention_policy"]["older_candidates"] = 17
    tampered_protocol_path = tmp_path / "tampered-sweep-protocol.json"
    atomic_json(tampered_protocol_path, tampered_protocol)
    with pytest.raises(ValueError, match="protocol contract"):
        sweep.evaluate_native_olmoe_attention_sweep(
            **_sweep_arguments(artifacts, out=tmp_path / "tampered-result.json"),
            sweep_protocol=tampered_protocol_path,
            sweep_protocol_sha256=sha256_file(tampered_protocol_path),
        )
