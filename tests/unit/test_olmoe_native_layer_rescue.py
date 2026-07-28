import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import engram.evaluation.olmoe_native_layer_rescue as rescue
from engram.evaluation.olmoe_native_sustained import (
    _THRESHOLDS,
    _q7_expectations,
)
from engram.models.fixture import create_tiny_olmoe_fixture
from engram.models.olmoe_native import repack_olmoe_non_mlp_weights
from engram.models.olmoe_q7 import repack_olmoe_q7_model
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.utils import atomic_json, sha256_file


_MODEL = {
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


def _population(
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


def _bands(**overrides):
    return {
        name: _population(**overrides.get(name, {}))
        for name in (
            "positions_0_15",
            "positions_16_31",
            "positions_32_63",
            "positions_64_95",
            "positions_96_127",
        )
    }


def test_sha_record_split_is_deterministic_and_fixes_two_six_membership():
    record_ids = [f"record-{index}" for index in range(8)]
    split = rescue._record_split(record_ids)
    expected = sorted(
        range(8),
        key=lambda index: (
            hashlib.sha256(record_ids[index].encode()).hexdigest(),
            record_ids[index],
            index,
        ),
    )

    assert [
        row["sequence_index"] for row in split["ranked_records"]
    ] == expected
    assert [
        row["sequence_index"] for row in split["selection"]
    ] == expected[:2]
    assert [
        row["sequence_index"] for row in split["internal_holdout"]
    ] == expected[2:]
    assert rescue._record_split(record_ids) == split
    assert len(split["split_identity"]) == 64

    with pytest.raises(ValueError, match="distinct record IDs"):
        rescue._record_split(["duplicate"] * 8)


def test_exact_round_resources_final_budget_and_policy_layout():
    contracts = rescue._round_resource_contracts(_MODEL)
    assert [row["candidate_count"] for row in contracts] == [16, 15, 14]
    expected = [
        {
            "attention_state_bytes": 8_179_584,
            "attention_scratch_bytes": 4_736,
            "attention_eviction_events": 1_680,
            "attention_older_candidate_entries_scored": 208_320,
            "attention_older_selected_entries": 106_080,
            "attention_sink_insertions": 480,
            "attention_heavy_hitter_updates_minimum": 1_440,
            "attention_heavy_hitter_updates_maximum": 26_400,
            "attention_local_kv_bytes": 609_091_584,
            "attention_candidate_key_bytes": 106_659_840,
            "attention_selected_value_bytes": 54_312_960,
            "attention_logical_read_bytes": 770_064_384,
            "attention_logical_read_fraction": 0.3558094113372093,
        },
        {
            "attention_state_bytes": 10_022_656,
            "attention_scratch_bytes": 5_632,
            "attention_eviction_events": 1_568,
            "attention_older_candidate_entries_scored": 194_432,
            "attention_older_selected_entries": 99_008,
            "attention_sink_insertions": 448,
            "attention_heavy_hitter_updates_minimum": 1_344,
            "attention_heavy_hitter_updates_maximum": 24_640,
            "attention_local_kv_bytes": 712_769_536,
            "attention_candidate_key_bytes": 99_549_184,
            "attention_selected_value_bytes": 50_692_096,
            "attention_logical_read_bytes": 863_010_816,
            "attention_logical_read_fraction": 0.39875545058139533,
        },
        {
            "attention_state_bytes": 11_865_728,
            "attention_scratch_bytes": 6_528,
            "attention_eviction_events": 1_456,
            "attention_older_candidate_entries_scored": 180_544,
            "attention_older_selected_entries": 91_936,
            "attention_sink_insertions": 416,
            "attention_heavy_hitter_updates_minimum": 1_248,
            "attention_heavy_hitter_updates_maximum": 22_880,
            "attention_local_kv_bytes": 816_447_488,
            "attention_candidate_key_bytes": 92_438_528,
            "attention_selected_value_bytes": 47_071_232,
            "attention_logical_read_bytes": 955_957_248,
            "attention_logical_read_fraction": 0.4417014898255814,
        },
    ]
    for contract, expected_round in zip(contracts, expected, strict=True):
        actual = contract["attention_expectations_per_sequence"]
        assert actual["positions_processed"] == 128
        assert actual["dense_full_context_logical_kv_bytes"] == 2_164_260_864
        for name, value in expected_round.items():
            if name == "attention_logical_read_fraction":
                assert actual[name] == pytest.approx(value)
            else:
                assert actual[name] == value

    final = rescue._final_schedule_contract(_MODEL)
    assert final["base_layer_count"] == 13
    assert final["rescued_layer_count"] == 3
    assert final["attention_logical_read_bytes_per_sequence"] == 955_957_248
    assert final["attention_logical_read_fraction"] == pytest.approx(
        0.4417014898255814
    )
    assert final["attention_logical_read_fraction"] <= 0.45

    policies = rescue._schedule_policies([1, 7, 15])
    assert len(policies) == 16
    assert [
        index
        for index, policy in enumerate(policies)
        if policy == rescue._RESCUE_POLICY
    ] == [
        1,
        7,
        15,
    ]
    assert sum(policy == rescue._BASE_POLICY for policy in policies) == 13


@pytest.mark.parametrize(
    "rescued_layers,layers",
    [
        ([0, 0], 16),
        ([True], 16),
        ([1.0], 16),
        ([-1], 16),
        ([16], 16),
        ([0, 1, 2, 3], 16),
        ([0], 15),
    ],
)
def test_schedule_policy_validation_is_strict(rescued_layers, layers):
    with pytest.raises(ValueError, match="schedule is invalid"):
        rescue._schedule_policies(rescued_layers, layers=layers)


def test_schedule_algebra_rejects_non_olmoe_layer_count():
    wrong = deepcopy(_MODEL)
    wrong["layers"] = 15
    with pytest.raises(ValueError, match="16-layer OLMoE"):
        rescue._schedule_expectations(wrong, [0])


def test_q7_traffic_contract_is_unchanged_and_below_gate():
    contract = rescue._q7_traffic_contract(
        _MODEL,
        _q7_expectations(_MODEL),
    )
    assert contract == {
        "q7_scheduled_bytes_per_sequence": 93_952_409_600,
        "all_expert_ideal_q4_bytes_per_sequence": 412_316_860_416,
        "q7_fraction_of_all_expert_ideal_q4": pytest.approx(
            0.22786458333333334
        ),
    }
    assert (
        contract["q7_fraction_of_all_expert_ideal_q4"]
        <= _THRESHOLDS["maximum_q7_traffic_fraction"]
    )


def test_score_excludes_only_invariant_prefix_and_uses_frozen_ties():
    bands = _bands(
        positions_0_15={"kl": 100.0},
        positions_16_31={"top1": 0.91},
    )
    score = rescue._normalized_quality_score(_population(), bands)

    assert "positions_0_15" not in score["normalized_margins"]
    assert "positions_16_31" in score["normalized_margins"]
    assert score["worst"] == {
        "population": "positions_16_31",
        "metric": "teacher_top1_agreement",
        "normalized_margin": pytest.approx(0.1),
    }
    candidates = [
        {
            "candidate_layer": 9,
            "selection_score": {
                "worst_normalized_margin": 0.2,
                "mean_normalized_margin": 0.4,
            },
        },
        {
            "candidate_layer": 3,
            "selection_score": {
                "worst_normalized_margin": 0.2,
                "mean_normalized_margin": 0.4,
            },
        },
        {
            "candidate_layer": 1,
            "selection_score": {
                "worst_normalized_margin": 0.2,
                "mean_normalized_margin": 0.3,
            },
        },
    ]
    assert [
        row["candidate_layer"]
        for row in sorted(candidates, key=rescue._candidate_sort_key)
    ] == [3, 9, 1]


def test_score_rejects_non_finite_teacher_comparison():
    overall = _population(kl=float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        rescue._normalized_quality_score(overall, _bands())


def _fake_protocol_context():
    record_ids = [f"record-{index}" for index in range(8)]
    input_ids = [[index] + [0] * 128 for index in range(8)]
    model = deepcopy(_MODEL)
    context = {
        "sustained_protocol": {
            "source_revision": "revision",
            "source_config_sha256": "config",
            "source_index_sha256": "index",
            "source_shard_sha256": {"shard": "hash"},
            "input_identity": "input",
        },
        "identities": {
            "package_manifest_sha256": "package",
            "native_library_sha256": "reference-library",
            "dataset_sha256": "dataset",
            "corpus_manifest_sha256": "corpus",
            "teacher_reference_sha256": "teacher-reference",
            "teacher_arrays_sha256": "teacher-arrays",
        },
        "hashes": {
            "sustained_protocol_sha256": "sustained-protocol",
            "sustained_result_sha256": "sustained-result",
            "control_protocol_sha256": "control-protocol",
            "control_result_sha256": "control-result",
        },
        "sweep_hashes": {
            "sweep_protocol_sha256": "sweep-protocol",
            "sweep_result_sha256": "sweep-result",
        },
        "control_source_hash": "control-source",
        "sweep_source_hash": "sweep-source",
        "historical_evaluator_sources": {"old.py": "a" * 64},
        "rescue_source_inventory": {
            "src/engram/runtime/olmoe_native.py": "b" * 64,
            "src/engram/evaluation/olmoe_native_layer_rescue.py": "c" * 64,
        },
        "candidate_library_sha256": "candidate-library",
        "input_ids": input_ids,
        "record_ids": record_ids,
        "split": rescue._record_split(record_ids),
        "model": model,
        "q7_expectations": _q7_expectations(model),
        "final_schedule_contract": rescue._final_schedule_contract(model),
        "evaluator_sources": {"old.py": "a" * 64},
    }
    return context


def test_protocol_freezes_separate_dsos_sources_rounds_and_fourth_layer_boundary():
    context = _fake_protocol_context()
    protocol = rescue._build_protocol(
        context,
        rescue_source_hash="rescue-source",
    )
    assert protocol["native_library_sha256"] == "reference-library"
    assert (
        protocol["candidate_native_library_sha256"] == "candidate-library"
    )
    assert protocol["historical_frozen_evaluator_source_sha256"] == {
        "old.py": "a" * 64
    }
    assert protocol["rescue_source_inventory_sha256"] == context[
        "rescue_source_inventory"
    ]
    assert [row["candidate_count"] for row in protocol["round_resource_contracts"]] == [
        16,
        15,
        14,
    ]
    boundary = protocol["next_rescue_budget_boundary"]
    assert boundary["rescued_layer_count"] == 4
    assert boundary["attention_logical_read_bytes_per_sequence"] == 1_048_903_680
    assert boundary["attention_logical_read_fraction"] == pytest.approx(
        0.48464752906976744
    )
    assert boundary["within_budget"] is False
    assert (
        protocol["scope"]["internal_holdout_is_not_sustained_gate_confirmation"]
        is True
    )

    assert (
        rescue._validate_protocol(
            protocol,
            context,
            protocol_hash="protocol",
            supplied_protocol_hash="protocol",
            rescue_source_hash="rescue-source",
        )
        is None
    )
    tampered = deepcopy(protocol)
    tampered["candidate_counts"][0] = 15
    with pytest.raises(ValueError, match="protocol contract"):
        rescue._validate_protocol(
            tampered,
            context,
            protocol_hash="protocol",
            supplied_protocol_hash="protocol",
            rescue_source_hash="rescue-source",
        )

    tampered_sources = deepcopy(protocol)
    tampered_sources["rescue_source_inventory_sha256"][
        "src/engram/runtime/olmoe_native.py"
    ] = "d" * 64
    with pytest.raises(ValueError, match="protocol contract"):
        rescue._validate_protocol(
            tampered_sources,
            context,
            protocol_hash="protocol",
            supplied_protocol_hash="protocol",
            rescue_source_hash="rescue-source",
        )

    with pytest.raises(ValueError, match="protocol contract"):
        rescue._validate_protocol(
            protocol,
            context,
            protocol_hash="changed",
            supplied_protocol_hash="protocol",
            rescue_source_hash="rescue-source",
        )


def test_source_inventory_descriptors_are_strict_and_cover_layered_boundary():
    historical_path = "src/engram/evaluation/olmoe_native_sustained.py"
    historical = rescue._historical_source_inventory(
        {"evaluator_source_sha256": {historical_path: "a" * 64}}
    )
    assert historical == {historical_path: "a" * 64}

    inventory = rescue._rescue_source_inventory(historical)
    required = {
        historical_path,
        "native/include/engram/olmoe_token_runtime.h",
        "native/include/engram/olmoe_token_runtime_c.h",
        "native/src/olmoe_token_runtime.cpp",
        "native/src/olmoe_token_runtime_c.cpp",
        "src/engram/runtime/olmoe_native.py",
        "src/engram/evaluation/olmoe_native_dense_control.py",
        "src/engram/evaluation/olmoe_native_attention_sweep.py",
        "src/engram/evaluation/olmoe_native_layer_rescue.py",
    }
    assert required <= inventory.keys()
    assert list(inventory) == sorted(inventory)
    assert all(
        len(digest) == 64
        and digest == digest.lower()
        and set(digest) <= set("0123456789abcdef")
        for digest in inventory.values()
    )

    for descriptor in (
        {},
        {"/absolute.py": "a" * 64},
        {"../escape.py": "a" * 64},
        {"source.py": "not-a-sha256"},
        {"source.py": "A" * 64},
    ):
        with pytest.raises(ValueError, match="source"):
            rescue._historical_source_inventory(
                {"evaluator_source_sha256": descriptor}
            )


def test_protocol_and_result_outputs_are_non_overwriting(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("preserve", encoding="utf-8")
    common = {
        "package": tmp_path / "package",
        "manifest_sha256": "manifest",
        "reference_library": tmp_path / "reference.so",
        "candidate_library": tmp_path / "candidate.so",
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
        "sweep_protocol": tmp_path / "sweep-protocol.json",
        "sweep_protocol_sha256": "sweep-protocol",
        "sweep_result": tmp_path / "sweep-result.json",
        "sweep_result_sha256": "sweep-result",
        "out": output,
        "threads": 12,
    }
    with pytest.raises(ValueError, match="protocol target already exists"):
        rescue.freeze_native_olmoe_layer_rescue_protocol(**common)
    with pytest.raises(ValueError, match="result target already exists"):
        rescue.evaluate_native_olmoe_layer_rescue(
            **common,
            rescue_protocol=tmp_path / "rescue-protocol.json",
            rescue_protocol_sha256="rescue-protocol",
        )
    assert output.read_text(encoding="utf-8") == "preserve"


def _mocked_evaluation(
    tmp_path,
    monkeypatch,
    *,
    invalid_candidate=None,
    parity_passed=True,
):
    model = deepcopy(_MODEL)
    model["vocab_size"] = 2
    model["hidden_size"] = 2
    input_ids = [[0] * 129 for _ in range(8)]
    record_ids = [f"record-{index}" for index in range(8)]
    split = rescue._record_split(record_ids)
    q7 = _q7_expectations(model)
    arrays_path = tmp_path / "teacher.npz"
    np.savez(
        arrays_path,
        logits=np.zeros((1_024, 2), dtype=np.float32),
        hidden=np.zeros((1_024, 2), dtype=np.float32),
        targets=np.zeros(1_024, dtype=np.int64),
    )
    rescue_protocol_path = tmp_path / "rescue-protocol.json"
    atomic_json(
        rescue_protocol_path,
        {
            "record_split": split,
            "candidate_layer_order": list(range(16)),
            "round_resource_contracts": rescue._round_resource_contracts(model),
            "population_contracts": {
                "selection": rescue._population_contract(2),
                "internal_holdout": rescue._population_contract(6),
            },
            "limitations": ["development-only"],
        },
    )
    context = {
        "input_ids": input_ids,
        "record_ids": record_ids,
        "split": split,
        "model": model,
        "q7_expectations": q7,
        "config_path": tmp_path / "config.json",
        "non_mlp_path": tmp_path / "weights.safetensors",
        "q7_path": tmp_path / "q7.bin",
        "hashes": {},
        "sweep_hashes": {},
        "identities": {"native_library_sha256": "reference"},
        "control_source_hash": "control",
        "sweep_source_hash": "sweep-source",
        "candidate_library_sha256": "candidate",
        "rescue_source_inventory": {},
        "final_schedule_contract": rescue._final_schedule_contract(model),
    }
    monkeypatch.setattr(
        rescue,
        "_authenticate_prerequisites",
        lambda **_kwargs: context,
    )
    monkeypatch.setattr(rescue, "_validate_protocol", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        rescue,
        "_evaluate_layered_abi_parity",
        lambda **_kwargs: {
            "passed": parity_passed,
            "elapsed_seconds": 0.0,
        },
    )
    monkeypatch.setattr(
        rescue,
        "_post_authentication",
        lambda **_kwargs: {"all": True},
    )
    calls = []
    winners = {1: 2, 2: 5, 3: 1}

    def fake_schedule(
        rescued_layers,
        *,
        sequence_indices,
        split_name,
        **_kwargs,
    ):
        calls.append((list(rescued_layers), list(sequence_indices), split_name))
        candidate = rescued_layers[-1]
        round_number = len(rescued_layers)
        margin = 1.0 if winners.get(round_number) == candidate else 0.0
        evidence = invalid_candidate != (round_number, candidate)
        return {
            "rescued_layers": list(rescued_layers),
            "attention_expectations_per_sequence": rescue._schedule_expectations(
                model,
                rescued_layers,
            ),
            "q7_expectations_per_sequence": q7,
            "q7_traffic_contract_per_sequence": rescue._q7_traffic_contract(
                model,
                q7,
            ),
            "selection_score": {
                "worst_normalized_margin": margin,
                "mean_normalized_margin": margin,
            },
            "quality_passed": split_name == "internal_holdout",
            "evidence_passed": evidence,
            "performance": {
                "primary_sequence_seconds": 0.0,
                "reset_replay_seconds": 0.0,
            },
        }

    monkeypatch.setattr(rescue, "_evaluate_schedule", fake_schedule)
    common = {
        "package": tmp_path / "package",
        "manifest_sha256": "manifest",
        "reference_library": tmp_path / "reference.so",
        "candidate_library": tmp_path / "candidate.so",
        "dataset": tmp_path / "dataset.jsonl",
        "corpus_manifest": tmp_path / "corpus.json",
        "teacher_reference": tmp_path / "teacher.json",
        "teacher_arrays": arrays_path,
        "sustained_protocol": tmp_path / "sustained-protocol.json",
        "sustained_protocol_sha256": "sustained-protocol",
        "sustained_result": tmp_path / "sustained-result.json",
        "sustained_result_sha256": "sustained-result",
        "control_protocol": tmp_path / "control-protocol.json",
        "control_protocol_sha256": "control-protocol",
        "control_result": tmp_path / "control-result.json",
        "control_result_sha256": "control-result",
        "sweep_protocol": tmp_path / "sweep-protocol.json",
        "sweep_protocol_sha256": "sweep-protocol",
        "sweep_result": tmp_path / "sweep-result.json",
        "sweep_result_sha256": "sweep-result",
        "rescue_protocol": rescue_protocol_path,
        "rescue_protocol_sha256": sha256_file(rescue_protocol_path),
        "out": tmp_path / "result.json",
        "threads": 12,
    }
    result = rescue.evaluate_native_olmoe_layer_rescue(**common)
    return result, calls


def test_all_45_candidates_execute_before_holdout_and_greedy_ties_are_frozen(
    tmp_path,
    monkeypatch,
):
    result, calls = _mocked_evaluation(tmp_path, monkeypatch)

    assert [row["candidate_count"] for row in result["greedy_round_results"]] == [
        16,
        15,
        14,
    ]
    assert result["candidate_evaluation_count"] == 45
    assert result["selected_rescued_layers"] == [2, 5, 1]
    assert len(calls) == 46
    selection_indices = [
        row["sequence_index"] for row in result["record_split"]["selection"]
    ]
    holdout_indices = [
        row["sequence_index"]
        for row in result["record_split"]["internal_holdout"]
    ]
    assert all(call[1] == selection_indices for call in calls[:45])
    assert calls[-1] == ([2, 5, 1], holdout_indices, "internal_holdout")
    assert result["evidence_passed"] is True
    assert result["internal_holdout_quality_passed"] is True
    assert result["fresh_confirmation_required"] is True
    assert result["decision"] == (
        "integrate_selected_schedule_then_freeze_fresh_package_native_"
        "confirmation"
    )


def test_candidate_evidence_failure_invalidates_whole_selection_after_45_runs(
    tmp_path,
    monkeypatch,
):
    result, calls = _mocked_evaluation(
        tmp_path,
        monkeypatch,
        invalid_candidate=(1, 9),
    )

    assert len(calls) == 46
    assert result["candidate_evaluation_count"] == 45
    assert result["evidence_passed"] is False
    assert result["status"] == "layer_rescue_invalid"
    assert result["fresh_confirmation_required"] is False
    assert result["decision"] == "stop_and_diagnose_layer_rescue_evidence"


def test_failed_layered_abi_parity_persists_invalid_result_before_candidate_search(
    tmp_path,
    monkeypatch,
):
    result, calls = _mocked_evaluation(
        tmp_path,
        monkeypatch,
        parity_passed=False,
    )

    assert calls == []
    assert result["status"] == "layer_rescue_invalid"
    assert result["candidate_evaluation_count"] == 0
    assert result["greedy_round_results"] == []
    assert result["selected_rescued_layers"] is None
    assert result["layered_abi_all_base_parity"]["passed"] is False
    assert result["evidence_checks"] == {
        "layered_abi_all_base_parity": False,
        "no_candidate_outputs_inspected": True,
        "post_run_authentication": True,
    }
    assert result["evidence_passed"] is False
    assert result["decision"] == (
        "stop_and_diagnose_layered_abi_all_base_parity"
    )
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == (
        result
    )


class _WrongPositionRuntime:
    def __init__(self, *_args, **_kwargs):
        self.calls = 0
        self.attention_metrics_available = True

    @property
    def position(self):
        return max(0, self.calls - 1)

    def reset(self):
        self.calls = 0

    def forward(self, _tokens):
        self.calls += 1
        expectations = rescue._schedule_expectations(
            _MODEL,
            [0],
            positions=self.calls,
        )
        metrics = {
            name: int(value)
            for name, value in expectations.items()
            if isinstance(value, int)
        }
        q7 = _q7_expectations(_MODEL)
        metrics["q7_scheduled_bytes"] = (
            self.calls * q7["scheduled_bytes_per_position"]
        )
        metrics["attention_heavy_hitter_updates"] = expectations[
            "attention_heavy_hitter_updates_minimum"
        ]
        return type("Result", (), {"next_token": 0, "metrics": metrics})()

    def last_diagnostics(self):
        return np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)

    def close(self):
        return None


def test_schedule_evidence_uses_runtime_position_not_loop_index(monkeypatch):
    monkeypatch.setattr(rescue, "OLMoENativeTokenRuntime", _WrongPositionRuntime)
    monkeypatch.setattr(
        rescue,
        "_position_metrics",
        lambda *_args: {
            "kl": 0.0,
            "top1_match": True,
            "teacher_top1": 0,
            "native_top1": 0,
            "target_nll_delta": 0.0,
            "hidden_relative_l2": 0.0,
        },
    )
    context = {
        "model": _MODEL,
        "q7_expectations": _q7_expectations(_MODEL),
        "input_ids": [[0] * 129 for _ in range(8)],
        "record_ids": [f"record-{index}" for index in range(8)],
        "config_path": Path("config.json"),
        "non_mlp_path": Path("weights.safetensors"),
        "q7_path": Path("q7.bin"),
    }
    result = rescue._evaluate_schedule(
        [0],
        sequence_indices=[0],
        split_name="selection",
        context=context,
        library_path=Path("candidate.so"),
        teacher_logits=np.zeros((1_024, 2), dtype=np.float32),
        teacher_hidden=np.zeros((1_024, 2), dtype=np.float32),
        targets=np.zeros(1_024, dtype=np.int64),
        threads=12,
    )
    assert result["evidence_checks"]["per_token_counter_checks"] is False
    assert result["evidence_checks"]["final_counter_checks"] is False
    assert result["evidence_passed"] is False


def test_native_layered_dso_all_base_parity_and_mixed_schedule_smoke(tmp_path):
    library = Path("build/libengram_olmoe_token_runtime.so")
    if not library.is_file():
        pytest.skip("native layered OLMoE runtime has not been built")
    model_path = create_tiny_olmoe_fixture(
        tmp_path / "model",
        hidden_size=8,
        intermediate_size=8,
        num_layers=16,
        num_heads=2,
        num_experts=2,
        num_experts_per_token=1,
        vocab_size=32,
    )
    q7_path = repack_olmoe_q7_model(
        model_path,
        tmp_path / "model.q7",
        group_size=8,
    )
    non_mlp_path = tmp_path / "non-mlp.safetensors"
    repack_olmoe_non_mlp_weights(model_path, non_mlp_path)
    tokens = [1 + (index % 30) for index in range(18)]

    with OLMoENativeTokenRuntime(
        model_path / "config.json",
        non_mlp_path,
        q7_path,
        library,
        threads=2,
        **rescue._BASE_POLICY,
    ) as scalar:
        scalar_result = scalar.forward(tokens)
        scalar_hidden, scalar_logits = scalar.last_diagnostics()
    with OLMoENativeTokenRuntime(
        model_path / "config.json",
        non_mlp_path,
        q7_path,
        library,
        threads=2,
        attention_policies=rescue._schedule_policies([]),
    ) as all_base:
        all_base_result = all_base.forward(tokens)
        all_base_hidden, all_base_logits = all_base.last_diagnostics()

    assert all_base_result.next_token == scalar_result.next_token
    assert {
        name: value
        for name, value in all_base_result.metrics.items()
        if name not in {"elapsed_ns", "q7_elapsed_ns"}
    } == {
        name: value
        for name, value in scalar_result.metrics.items()
        if name not in {"elapsed_ns", "q7_elapsed_ns"}
    }
    np.testing.assert_array_equal(all_base_hidden, scalar_hidden)
    np.testing.assert_array_equal(all_base_logits, scalar_logits)

    tiny_model = {
        "layers": 16,
        "query_heads": 2,
        "key_value_heads": 2,
        "head_dimension": 4,
    }
    expected = rescue._schedule_expectations(
        tiny_model,
        [0],
        positions=len(tokens),
    )
    with OLMoENativeTokenRuntime(
        model_path / "config.json",
        non_mlp_path,
        q7_path,
        library,
        threads=2,
        attention_policies=rescue._schedule_policies([0]),
    ) as mixed:
        mixed_result = mixed.forward(tokens)
        assert mixed.position == len(tokens)
    for name in (
        "attention_logical_read_bytes",
        "attention_state_bytes",
        "attention_scratch_bytes",
        "attention_eviction_events",
        "attention_older_candidate_entries_scored",
        "attention_older_selected_entries",
        "attention_sink_insertions",
    ):
        assert mixed_result.metrics[name] == expected[name]
    assert (
        expected["attention_heavy_hitter_updates_minimum"]
        <= mixed_result.metrics["attention_heavy_hitter_updates"]
        <= expected["attention_heavy_hitter_updates_maximum"]
    )


def test_cli_exposes_separate_reference_and_candidate_libraries(
    tmp_path,
    monkeypatch,
    capsys,
):
    calls = []
    monkeypatch.setattr(
        rescue,
        "freeze_native_olmoe_layer_rescue_protocol",
        lambda **kwargs: calls.append(kwargs) or {"status": "frozen"},
    )
    argv = [
        "layer-rescue",
        "freeze",
        "--package",
        str(tmp_path / "package"),
        "--manifest-sha256",
        "manifest",
        "--reference-library",
        str(tmp_path / "reference.so"),
        "--candidate-library",
        str(tmp_path / "candidate.so"),
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
        "--sweep-protocol",
        str(tmp_path / "sweep-protocol.json"),
        "--sweep-protocol-sha256",
        "sweep-protocol",
        "--sweep-result",
        str(tmp_path / "sweep-result.json"),
        "--sweep-result-sha256",
        "sweep-result",
        "--out",
        str(tmp_path / "protocol.json"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert rescue._main() == 0
    assert calls[0]["reference_library"] == tmp_path / "reference.so"
    assert calls[0]["candidate_library"] == tmp_path / "candidate.so"
    assert json.loads(capsys.readouterr().out)["status"] == "frozen"

    evaluate_calls = []
    monkeypatch.setattr(
        rescue,
        "evaluate_native_olmoe_layer_rescue",
        lambda **kwargs: (
            evaluate_calls.append(kwargs)
            or {
                "status": "layer_rescue_invalid",
                "decision": "stop_and_diagnose_layered_abi_all_base_parity",
            }
        ),
    )
    result_path = tmp_path / "result.json"
    rescue_protocol = tmp_path / "rescue-protocol.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "layer-rescue",
            "evaluate",
            *argv[2:-2],
            "--out",
            str(result_path),
            "--rescue-protocol",
            str(rescue_protocol),
            "--rescue-protocol-sha256",
            "rescue-protocol",
        ],
    )

    assert rescue._main() == 0
    assert evaluate_calls[0]["reference_library"] == tmp_path / "reference.so"
    assert evaluate_calls[0]["candidate_library"] == tmp_path / "candidate.so"
    assert evaluate_calls[0]["rescue_protocol"] == rescue_protocol
    assert (
        evaluate_calls[0]["rescue_protocol_sha256"] == "rescue-protocol"
    )
    assert evaluate_calls[0]["out"] == result_path
    assert json.loads(capsys.readouterr().out) == {
        "status": "layer_rescue_invalid",
        "decision": "stop_and_diagnose_layered_abi_all_base_parity",
    }
