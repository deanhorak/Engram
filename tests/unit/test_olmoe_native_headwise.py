import inspect
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import engram.evaluation.olmoe_native_headwise as headwise
import engram.evaluation.olmoe_native_layer_rescue as layer_rescue
from engram.evaluation.olmoe_native_sustained import _q7_expectations
from engram.utils import atomic_json, sha256_file, sha256_json


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


def _fake_context():
    record_ids = [f"record-{index}" for index in range(8)]
    split = layer_rescue._record_split(record_ids)
    layer_source_hash = "d" * 64
    return {
        "sustained_protocol": {
            "source_revision": "revision",
            "source_config_sha256": "config",
            "source_index_sha256": "index",
            "source_shard_sha256": {"model.safetensors": "shard"},
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
        "layer_rescue_protocol_sha256": "layer-protocol",
        "layer_rescue_result_sha256": "layer-result",
        "layer_rescue_protocol": {
            "rescue_source_sha256": layer_source_hash,
            "rescue_source_inventory_sha256": {
                "src/engram/evaluation/olmoe_native_layer_rescue.py": (
                    layer_source_hash
                )
            },
        },
        "layer_rescue_historical_source_inventory": {
            "src/engram/evaluation/olmoe_native_layer_rescue.py": (
                layer_source_hash
            )
        },
        "candidate_library_sha256": "layered-library",
        "record_ids": record_ids,
        "split": split,
        "input_ids": [[index] + [0] * 128 for index in range(8)],
        "model": deepcopy(_MODEL),
        "q7_expectations": _q7_expectations(_MODEL),
    }


def _mask_from_layer_major_prefix(count=51):
    pairs = [(index // 16, index % 16) for index in range(count)]
    selected = [
        {
            "rank": rank,
            "layer": layer,
            "head": head,
            "layer_major_index": layer * 16 + head,
        }
        for rank, (layer, head) in enumerate(pairs, start=1)
    ]
    pair_set = set(pairs)
    nested = [
        [(layer, head) in pair_set for head in range(16)]
        for layer in range(16)
    ]
    return {
        "selected_heads": selected,
        "attention_head_mask": nested,
        "attention_head_mask_sha256": sha256_json(nested),
    }


def test_exact_51_head_contract_and_52_head_cap_boundary():
    contract = headwise._headwise_budget_contract(_MODEL)
    expectations = contract["attention_expectations_per_sequence"]

    assert expectations == {
        "positions_processed": 128,
        "attention_state_bytes": 12_284_864,
        "attention_scratch_bytes": 107_136,
        "attention_eviction_events": 22_960,
        "attention_older_candidate_entries_scored": 177_940,
        "attention_older_selected_entries": 90_610,
        "attention_sink_insertions": 410,
        "attention_heavy_hitter_updates_minimum": 1_230,
        "attention_heavy_hitter_updates_maximum": 22_550,
        "attention_local_kv_bytes": 835_887_104,
        "attention_candidate_key_bytes": 91_105_280,
        "attention_selected_value_bytes": 46_392_320,
        "attention_logical_read_bytes": 973_384_704,
        "dense_full_context_logical_kv_bytes": 2_164_260_864,
        "attention_logical_read_fraction": pytest.approx(
            0.44975387218386625
        ),
    }
    assert contract["rescued_heads"] == 51
    assert contract["attention_logical_read_fraction"] <= 0.45
    assert contract["next_head_boundary"] == {
        "rescued_heads": 52,
        "attention_logical_read_fraction": pytest.approx(
            0.4524379996366279
        ),
        "within_budget": False,
    }
    assert contract["next_head_boundary"]["attention_logical_read_fraction"] > 0.45


def test_head_policy_mask_is_nested_layer_major_and_strict():
    selected = [(0, 0), (3, 7), (15, 15)]
    policies = headwise._head_policies(selected)

    assert len(policies) == 16
    assert all(len(layer) == 16 for layer in policies)
    assert policies[0][0] == headwise._RESCUE_POLICY
    assert policies[3][7] == headwise._RESCUE_POLICY
    assert policies[15][15] == headwise._RESCUE_POLICY
    assert policies[0][1] == headwise._BASE_POLICY
    assert (
        sum(
            policy == headwise._RESCUE_POLICY
            for layer in policies
            for policy in layer
        )
        == 3
    )

    with pytest.raises(ValueError, match="duplicates"):
        headwise._head_policies([(0, 0), (0, 0)])
    with pytest.raises(ValueError, match="mask is invalid"):
        headwise._head_policies([(16, 0)])


def _toy_attentions():
    values = np.zeros((2, 2, 2, 8, 8), dtype=np.float32)
    for sequence in range(2):
        for layer in range(2):
            for head in range(2):
                for query in range(8):
                    values[sequence, layer, head, query, query] = 1.0

    def set_deficit(sequence, layer, head, deficit):
        for query in range(3, 8):
            values[sequence, layer, head, query, :] = 0.0
            values[sequence, layer, head, query, 0] = deficit
            values[sequence, layer, head, query, 1] = deficit
            values[sequence, layer, head, query, query] = 1.0 - 2 * deficit

    # (1,0) wins primary. (0,0) and (0,1) tie on total; the former
    # wins the frozen minimum-record-band secondary score.
    set_deficit(0, 1, 0, 0.11)
    set_deficit(1, 1, 0, 0.11)
    set_deficit(0, 0, 0, 0.10)
    set_deficit(1, 0, 0, 0.10)
    set_deficit(0, 0, 1, 0.20)
    return values


def test_deficit_ranking_uses_primary_secondary_and_layer_head_ties():
    ranking = headwise._derive_head_scores(
        _toy_attentions(),
        local_window=2,
        top_k=1,
        bands=(("early", 3, 5), ("late", 5, 8)),
    )

    assert [
        (row["layer"], row["head"]) for row in ranking
    ] == [
        (1, 0),
        (0, 0),
        (0, 1),
        (1, 1),
    ]
    assert ranking[1]["total_deficit"] == pytest.approx(
        ranking[2]["total_deficit"]
    )
    assert ranking[1]["minimum_sequence_band_mean_deficit"] == pytest.approx(
        0.1
    )
    assert ranking[2]["minimum_sequence_band_mean_deficit"] == 0.0
    assert [row["rank"] for row in ranking] == [1, 2, 3, 4]


def test_deficit_ranking_rejects_nan_and_mask_prefix_rejects_duplicates():
    values = _toy_attentions()
    values[0, 0, 0, 3, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        headwise._derive_head_scores(
            values,
            local_window=2,
            top_k=1,
            bands=(("all", 3, 8),),
        )

    ranking = [
        {"layer": index // 2, "head": index % 2}
        for index in range(4)
    ]
    selected, mask = headwise._mask_from_ranking(
        ranking,
        layers=2,
        heads=2,
        selected_count=3,
    )
    assert selected == [(0, 0), (0, 1), (1, 0)]
    assert mask == [[True, True], [True, False]]
    duplicate = deepcopy(ranking)
    duplicate[1] = dict(duplicate[0])
    with pytest.raises(ValueError, match="prefix"):
        headwise._mask_from_ranking(
            duplicate,
            layers=2,
            heads=2,
            selected_count=3,
        )


def test_attention_trace_shape_dtype_causality_and_normalization_are_strict():
    values = np.zeros((2, 1, 1, 4, 4), dtype=np.float32)
    for query in range(4):
        values[..., query, query] = 1.0
    checks, observations = headwise._attention_array_evidence(
        values,
        expected_shape=(2, 1, 1, 4, 4),
    )
    assert all(checks.values())
    assert observations["maximum_attention_row_sum_absolute_error"] == 0.0

    wrong_shape, _ = headwise._attention_array_evidence(
        values,
        expected_shape=(2, 1, 1, 5, 5),
    )
    assert wrong_shape["shape"] is False
    wrong_dtype, _ = headwise._attention_array_evidence(
        values.astype(np.float64),
        expected_shape=(2, 1, 1, 4, 4),
    )
    assert wrong_dtype["dtype_float32"] is False
    future = values.copy()
    future[..., 0, 1] = 0.1
    future[..., 0, 0] = 0.9
    causal, _ = headwise._attention_array_evidence(
        future,
        expected_shape=(2, 1, 1, 4, 4),
    )
    assert causal["causal_upper_triangle_zero_within_tolerance"] is False
    negative = values.copy()
    negative[..., 2, 0] = -0.1
    negative[..., 2, 2] = 1.1
    nonnegative, _ = headwise._attention_array_evidence(
        negative,
        expected_shape=(2, 1, 1, 4, 4),
    )
    assert nonnegative["nonnegative_within_tolerance"] is False


def test_trace_protocol_is_prospective_and_binds_full_split_and_exact_derivation():
    context = _fake_context()
    protocol = headwise._build_trace_protocol(
        context,
        source_sha256="e" * 64,
        source_inventory={"source.py": "f" * 64},
        device="cuda",
        threads=12,
    )

    assert protocol["status"] == "frozen_before_dense_attention_map_capture"
    assert protocol["record_split"] == context["split"]
    assert protocol["record_split_identity"] == context["split"]["split_identity"]
    assert protocol["selection_records"] == context["split"]["selection"]
    assert protocol["teacher_capture"] == {
        "model_role": "untouched_dense_teacher",
        "dtype": "bfloat16",
        "device": "cuda",
        "threads": 12,
        "attention_implementation": "eager",
        "eval": True,
        "inference_mode": True,
        "use_cache": False,
        "output_attentions": True,
        "output_hidden_states": False,
        "return_dict": True,
        "batch_size": 1,
    }
    assert protocol["derivation"]["stable_ranking"] == [
        "descending_total_deficit",
        "descending_minimum_sequence_band_mean_deficit",
        "ascending_layer",
        "ascending_head",
    ]
    assert protocol["derivation"]["maps_from_internal_screen_records"] is False
    assert (
        protocol["provenance"]["protocol_frozen_before_attention_map_capture"]
        is True
    )
    assert protocol["attention_array_contract"]["shape"] == [
        2,
        16,
        16,
        128,
        128,
    ]
    assert (
        protocol["attention_array_contract"]["expected_uncompressed_bytes"]
        == 33_554_432
    )
    assert any(
        "beyond it" in limitation and "W128/C8/K4/S2" in limitation
        for limitation in protocol["limitations"]
    )


def test_trace_protocol_from_another_authenticated_context_is_rejected():
    context_a = _fake_context()
    source_hash = "e" * 64
    inventory = {"source.py": "f" * 64}
    protocol = headwise._build_trace_protocol(
        context_a,
        source_sha256=source_hash,
        source_inventory=inventory,
        device="cpu",
        threads=12,
    )
    context_b = deepcopy(context_a)
    selection_index = context_b["split"]["selection"][0]["sequence_index"]
    context_b["input_ids"][selection_index][0] += 1

    with pytest.raises(ValueError, match="trace protocol contract"):
        headwise._validate_trace_protocol(
            protocol,
            context_b,
            protocol_sha256="protocol",
            supplied_sha256="protocol",
            source_sha256=source_hash,
            source_inventory=inventory,
        )


def test_trace_artifact_chain_rejects_identity_and_shape_drift(
    tmp_path,
):
    context = _fake_context()
    source_path = Path(headwise.__file__).resolve()
    source_hash = sha256_file(source_path)
    inventory = {
        "src/engram/evaluation/olmoe_native_headwise.py": source_hash
    }
    protocol = headwise._build_trace_protocol(
        context,
        source_sha256=source_hash,
        source_inventory=inventory,
        device="cpu",
        threads=12,
    )
    protocol_path = tmp_path / "protocol.json"
    arrays_path = tmp_path / "trace.npz"
    metadata_path = tmp_path / "metadata.json"
    atomic_json(protocol_path, protocol)
    # Deliberately violates the frozen 128x128 tensor shape.
    np.savez(
        arrays_path,
        attentions=np.zeros((2, 16, 16, 8, 8), dtype=np.float32),
    )
    metadata = {
        "schema_version": 1,
        "experiment": headwise._TRACE_CAPTURE_EXPERIMENT,
        "status": "dense_attention_trace_complete",
        "evidence_passed": True,
        "internal_record_attention_maps_captured": False,
        "selection_records": protocol["selection_records"],
        "selection_sequence_indices": protocol["selection_sequence_indices"],
        "artifacts": {
            **headwise._base_bindings(context),
            "trace_protocol_sha256": sha256_file(protocol_path),
            "trace_arrays_sha256": sha256_file(arrays_path),
            "headwise_source_sha256": source_hash,
            "headwise_source_inventory_sha256": inventory,
        },
        "attention_array": {
            **protocol["attention_array_contract"],
            "sha256": sha256_file(arrays_path),
        },
        "evidence_checks": {"all": True},
        "post_capture_authentication": {"all": True},
    }
    atomic_json(metadata_path, metadata)
    common = {
        "trace_protocol": protocol_path,
        "trace_protocol_sha256": sha256_file(protocol_path),
        "trace_metadata": metadata_path,
        "trace_metadata_sha256": sha256_file(metadata_path),
        "trace_arrays": arrays_path,
        "trace_arrays_sha256": sha256_file(arrays_path),
    }
    with pytest.raises(ValueError, match="attention arrays"):
        headwise._trace_artifacts(**common)
    with pytest.raises(ValueError, match="artifact chain"):
        headwise._trace_artifacts(
            **{**common, "trace_arrays_sha256": "0" * 64}
        )


def test_mocked_trace_to_screen_flow_cannot_use_six_records_for_selection():
    context_a = _fake_context()
    mask = _mask_from_layer_major_prefix()
    trace = {
        "provenance": {
            "protocol_frozen_before_attention_map_capture": True,
        }
    }
    hashes = {
        "trace_protocol_sha256": "1" * 64,
        "trace_metadata_sha256": "2" * 64,
        "trace_arrays_sha256": "3" * 64,
        "head_mask_sha256": "4" * 64,
    }
    protocol_a = headwise._build_screen_protocol(
        context_a,
        trace_protocol=trace,
        mask=mask,
        trace_hashes=hashes,
        candidate_library_sha256="5" * 64,
        source_sha256="6" * 64,
        source_inventory={"source.py": "7" * 64},
    )

    # Simulate arbitrary changes to all six screen records after the mask has
    # already been fixed. The screen binding changes, but the selected head
    # prefix and identity cannot.
    context_b = deepcopy(context_a)
    for ordinal, row in enumerate(
        context_b["split"]["internal_holdout"],
        start=1,
    ):
        row["record_id"] = f"changed-screen-{ordinal}"
    context_b["split"]["split_identity"] = sha256_json(context_b["split"])
    protocol_b = headwise._build_screen_protocol(
        context_b,
        trace_protocol=trace,
        mask=mask,
        trace_hashes=hashes,
        candidate_library_sha256="5" * 64,
        source_sha256="6" * 64,
        source_inventory={"source.py": "7" * 64},
    )

    assert protocol_a["internal_screen_records"] != protocol_b[
        "internal_screen_records"
    ]
    assert protocol_a["selected_heads"] == protocol_b["selected_heads"]
    assert protocol_a["attention_head_mask"] == protocol_b[
        "attention_head_mask"
    ]
    assert protocol_a["head_mask_identity_sha256"] == protocol_b[
        "head_mask_identity_sha256"
    ]
    assert protocol_a["scope"]["attention_maps_from_internal_screen_records"] is False
    assert protocol_a["provenance"]["six_internal_records_cannot_change_mask"] is True
    assert protocol_a["analytical_byte_components"] == {
        "attention_local_kv_bytes": 835_887_104,
        "attention_candidate_key_bytes": 91_105_280,
        "attention_selected_value_bytes": 46_392_320,
        "runtime_observes_only_total_logical_read_bytes": True,
    }


def test_parity_counter_rule_allows_separate_state_scratch_and_scaled_evictions():
    layered_expectations = layer_rescue._schedule_expectations(_MODEL, [])
    headwise_expectations = headwise._headwise_expectations(_MODEL, [])
    q7 = _q7_expectations(_MODEL)

    def metrics(expectations):
        value = {
            name: int(expected)
            for name, expected in expectations.items()
            if isinstance(expected, int)
        }
        value["attention_heavy_hitter_updates"] = expectations[
            "attention_heavy_hitter_updates_minimum"
        ]
        value["attention_weight_bytes"] = 123
        value["q7_scheduled_bytes"] = q7["scheduled_bytes_per_sequence"]
        return value

    layered = metrics(layered_expectations)
    split = metrics(headwise_expectations)
    checks = headwise._parity_counter_checks(
        layered,
        split,
        layered_expectations=layered_expectations,
        headwise_expectations=headwise_expectations,
        q7_expectations=q7,
        position=128,
    )

    assert all(checks.values())
    assert layered["attention_state_bytes"] != split["attention_state_bytes"]
    assert layered["attention_scratch_bytes"] != split["attention_scratch_bytes"]
    assert (
        split["attention_eviction_events"]
        == layered["attention_eviction_events"] * 16
    )
    wrong = dict(split)
    wrong["attention_state_bytes"] = layered["attention_state_bytes"]
    assert (
        headwise._parity_counter_checks(
            layered,
            wrong,
            layered_expectations=layered_expectations,
            headwise_expectations=headwise_expectations,
            q7_expectations=q7,
            position=128,
        )["headwise_state_analytical"]
        is False
    )


class _WrongCachePositionRuntime:
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
        expectations = headwise._headwise_expectations(
            _MODEL,
            [(index // 16, index % 16) for index in range(51)],
            positions=self.calls,
        )
        metrics = {
            name: int(value)
            for name, value in expectations.items()
            if isinstance(value, int)
        }
        q7 = _q7_expectations(_MODEL)
        metrics["attention_heavy_hitter_updates"] = expectations[
            "attention_heavy_hitter_updates_minimum"
        ]
        metrics["q7_scheduled_bytes"] = (
            self.calls * q7["scheduled_bytes_per_position"]
        )
        metrics["attention_weight_bytes"] = 0
        return type("Result", (), {"next_token": 0, "metrics": metrics})()

    def last_diagnostics(self):
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)

    def close(self):
        return None


def test_candidate_evidence_rejects_runtime_cache_position_drift(monkeypatch):
    monkeypatch.setattr(
        headwise,
        "OLMoENativeTokenRuntime",
        _WrongCachePositionRuntime,
    )
    monkeypatch.setattr(
        headwise,
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
    selected = [(index // 16, index % 16) for index in range(51)]
    context = {
        "model": deepcopy(_MODEL),
        "q7_expectations": _q7_expectations(_MODEL),
        "config_path": Path("config.json"),
        "non_mlp_path": Path("weights.safetensors"),
        "q7_path": Path("q7.bin"),
        "input_ids": [[0] * 129],
        "record_ids": ["record-0"],
    }
    result = headwise._evaluate_headwise_candidate(
        selected,
        sequence_indices=[0],
        context=context,
        library=Path("headwise.so"),
        teacher_logits=np.zeros((128, 1), dtype=np.float32),
        teacher_hidden=np.zeros((128, 1), dtype=np.float32),
        targets=np.zeros(128, dtype=np.int64),
        threads=12,
        replay_sequence_index=0,
    )

    assert result["evidence_checks"]["cache_positions"] is False
    assert result["sequence_results"][0]["cache_positions_passed"] is False
    assert result["reset_replay"]["cache_positions_passed"] is False
    assert result["evidence_passed"] is False


def test_capture_prepares_transformers_compatibility_before_model_import():
    source = inspect.getsource(
        headwise.capture_native_olmoe_headwise_dense_attention
    )
    assert source.index("_prepare_transformers_imports()") < source.index(
        "from transformers import AutoModelForCausalLM"
    )


def test_non_overwriting_boundaries_fail_before_authentication(tmp_path):
    existing = tmp_path / "existing.json"
    existing.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="trace protocol target already exists"):
        headwise.freeze_native_olmoe_headwise_trace_protocol(out=existing)
    with pytest.raises(ValueError, match="mask target already exists"):
        headwise.derive_native_olmoe_headwise_mask(
            trace_protocol=tmp_path / "protocol",
            trace_protocol_sha256="x",
            trace_metadata=tmp_path / "metadata",
            trace_metadata_sha256="x",
            trace_arrays=tmp_path / "arrays",
            trace_arrays_sha256="x",
            out=existing,
        )
    shared = tmp_path / "shared-output"
    with pytest.raises(ValueError, match="trace output target already exists"):
        headwise.capture_native_olmoe_headwise_dense_attention(
            trace_protocol=tmp_path / "protocol",
            trace_protocol_sha256="x",
            arrays_out=shared,
            trace_out=shared,
            manifest_sha256="x",
        )
    assert existing.read_text(encoding="utf-8") == "preserve"


def test_cli_exposes_all_two_phase_commands(capsys):
    with pytest.raises(SystemExit) as raised:
        headwise._main(["--help"])
    assert raised.value.code == 0
    output = capsys.readouterr().out
    for command in (
        "freeze-trace",
        "capture",
        "derive",
        "freeze-screen",
        "evaluate",
    ):
        assert command in output
