from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_head_mass_oracle as oracle


def test_prior_residual_roots_are_frozen() -> None:
    assert oracle._EXPECTED_CAPACITY_PROTOCOL_SHA256 == (
        "584302d17a3224cda1b61dfe1f62685497fa5a0dc335cfc0a074439456ee1606"
    )
    assert oracle._EXPECTED_CAPACITY_RESULT_SHA256 == (
        "c636ad124d570f3675a36f0a23b276ba2e4cd4f5efc23dbf98cc10cd2cfd8e33"
    )


def test_prior_residual_failure_contract_uses_rank_mapping(tmp_path) -> None:
    protocol = tmp_path / "protocol.json"
    outcomes = {}
    for rank, recovery in (
        (2, 0.40046952208141817),
        (4, 0.4286862133341903),
        (8, 0.469252618228868),
    ):
        outcomes[str(rank)] = {
            "rank": rank,
            "global": {"recovery": recovery},
            "gate": {
                "global_recovery_at_least_0_50": False,
                "every_sequence_recovery_at_least_0_25": True,
                "every_block_entry_recovery_at_least_0_25": True,
                "at_least_12_of_16_layers_positive_recovery": True,
                "passed": False,
            },
        }
    value = {
        "schema_version": oracle.capacity._SCHEMA_VERSION,
        "experiment": oracle.capacity._RESULT_EXPERIMENT,
        "status": "train_residual_capacity_gate_failed",
        "protocol": {
            "path": str(protocol),
            "sha256": oracle._EXPECTED_CAPACITY_PROTOCOL_SHA256,
        },
        "confirmation_split_opened": False,
        "decision": {
            "capacity_gate_passed": False,
            "train_only_predictor_fit_authorized": False,
            "native_integration_authorized": False,
            "development_authorized": False,
            "confirmation_authorized": False,
        },
        "capacity": {
            "passed": False,
            "selected_rank": 8,
            "selection_role": "best_failed_rank_for_diagnostic_replay",
            "selected_metric_replay": {"passed": True},
            "rank_order": [2, 4, 8],
            "rank_outcomes": outcomes,
        },
        "post_run_authentication": {"all": True},
        "trace_manifest": {
            "sha256": oracle._EXPECTED_CAPACITY_MANIFEST_SHA256,
            "shard_count": oracle._RECORDS,
        },
    }
    result = oracle._validate_capacity_failure(value, protocol)
    assert result["selected_rank"] == 8
    assert result["failure_condition"] == "global_recovery_below_0_50_only"


def test_frozen_gamma_table_has_exact_bits_multipliers_and_tie_order() -> None:
    rows = oracle._gamma_table()
    assert [row["code"] for row in rows] == list(range(8))
    assert [row["beta_bits"] for row in rows] == [
        None,
        "0xc0051592",
        "0xbfb17218",
        "0xbf317218",
        "0x00000000",
        "0x3f317218",
        "0x3fb17218",
        "0x40051592",
    ]
    np.testing.assert_allclose(
        [row["multiplier_float32"] for row in rows],
        [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        rtol=1.0e-6,
        atol=0.0,
    )
    assert oracle._TIE_PRIORITY == (4, 3, 5, 2, 6, 1, 7, 0)


def test_mass_selection_uses_dense_teacher_target_and_natural_tie_break() -> None:
    regular = np.asarray([[0.5, 0.75, 0.5]], dtype=np.float32)
    episodic = np.asarray([[0.5, 0.25, 0.5]], dtype=np.float32)
    candidates = oracle._candidate_masses(regular, episodic)
    target = np.asarray(
        [
            [
                candidates[0, 0, 4],
                candidates[0, 1, 3],
                0.0,
            ]
        ],
        dtype=np.float32,
    )
    codes, selected, distances = oracle._select_gamma_codes(
        regular,
        episodic,
        target,
    )
    assert codes.tolist() == [[4, 3, 0]]
    np.testing.assert_array_equal(
        selected,
        np.take_along_axis(candidates, codes[..., None], axis=-1)[..., 0],
    )
    assert np.all(
        np.take_along_axis(distances, codes[..., None], axis=-1)[..., 0]
        <= distances[..., 4]
    )


def test_counterfactual_pre_wo_anchors_gamma_one_and_zero_endpoint() -> None:
    base = np.asarray([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    regular = np.asarray([[1.0, 2.0, 6.0, 8.0]], dtype=np.float32)
    episodic = np.asarray([[3.0, 4.0, 9.0, 12.0]], dtype=np.float32)
    regular_mass = np.asarray([[0.25, 0.4]], dtype=np.float32)
    episodic_mass = np.asarray([[0.75, 0.6]], dtype=np.float32)
    codes = np.asarray([[4, 0]], dtype=np.uint8)
    result = oracle._counterfactual_pre_wo(
        base,
        regular,
        episodic,
        regular_mass,
        episodic_mass,
        codes,
        query_heads=2,
    )
    np.testing.assert_array_equal(result[0, :2], base[0, :2])
    np.testing.assert_allclose(result[0, 2:], regular[0, 2:] / 0.4)


def test_projection_uses_native_o_proj_orientation() -> None:
    base = np.zeros((1, 1, 1, 2), dtype=np.float32)
    candidate = np.asarray([[[[2.0, 3.0]]]], dtype=np.float32)
    weight = np.asarray([[[1.0, 10.0], [100.0, 1000.0]]], dtype=np.float32)
    correction = oracle._project_counterfactual_delta(base, candidate, weight)
    np.testing.assert_array_equal(
        correction,
        np.asarray([[[[32.0, 3200.0]]]], dtype=np.float32),
    )


def test_load_output_projections_accepts_packaged_single_file(
    tmp_path,
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    monkeypatch.setattr(oracle, "_LAYERS", 2)
    monkeypatch.setattr(oracle, "_HIDDEN_SIZE", 2)
    path = tmp_path / "non_mlp.safetensors"
    tensors = {
        f"model.layers.{layer}.self_attn.o_proj.weight": torch.asarray(
            [[1.0 + layer, 2.0], [3.0, 4.0 + layer]],
            dtype=torch.bfloat16,
        )
        for layer in range(2)
    }
    save_file(tensors, path)
    loaded, hashes = oracle._load_output_projections({"non_mlp_path": path})
    assert loaded.shape == (2, 2, 2)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(
        loaded[0],
        tensors[next(iter(tensors))].float().numpy(),
    )
    assert len(hashes) == 2
    assert all(len(digest) == 64 for digest in hashes.values())


def test_direct_grid_qualifies_same_input_layer_zero_only(monkeypatch) -> None:
    monkeypatch.setattr(oracle, "_READ_POSITIONS", (0,))
    monkeypatch.setattr(oracle, "_LAYERS", 2)
    monkeypatch.setattr(oracle, "_HIDDEN_SIZE", 2)
    monkeypatch.setattr(oracle, "_QUERY_HEADS", 1)
    monkeypatch.setattr(oracle, "_validate_runtime_route", lambda *_args, **_kw: None)
    regular = np.asarray([[[0.25, 0.5], [0.25, 0.5]]], dtype=np.float32)
    episodic = np.asarray([[[0.75, 1.5], [0.75, 1.5]]], dtype=np.float32)
    regular_mass = np.asarray([[[0.25], [0.25]]], dtype=np.float32)
    episodic_mass = np.asarray([[[0.75], [0.75]]], dtype=np.float32)
    base = regular + episodic
    beta_zero_arrays = {
        "base_attention_output": base,
        "regular_component": regular,
        "episodic_component": episodic,
        "regular_mass": regular_mass,
        "episodic_mass": episodic_mass,
        "base_projected": base.copy(),
    }
    gamma_by_bits = {
        row["beta_bits"]: row for row in oracle._gamma_table() if row["beta_bits"]
    }

    class FakeRuntime:
        def __init__(self, beta: float) -> None:
            row = gamma_by_bits[oracle._float32_bits(beta)]
            code = int(row["code"])
            codes = np.full((1, 2, 1), code, dtype=np.uint8)
            candidate = oracle._counterfactual_pre_wo(
                base,
                regular,
                episodic,
                regular_mass,
                episodic_mass,
                codes,
            )[0]
            mass = oracle._candidate_masses(regular_mass, episodic_mass)[0, ..., code]
            projected = candidate.copy()
            candidate[1] += np.float32(1.0)
            mass[1] += np.float32(0.1)
            projected[1] += np.float32(1.0)
            self._mass = SimpleNamespace(
                base_attention_output=candidate,
                episodic_mass=mass,
            )
            self._projected = projected

        def forward_episodic(self, *_args) -> None:
            return None

        def last_episodic_mass_trace(self):
            return self._mass

        def last_shadow_trace(self):
            zeros = np.zeros((2, 2), dtype=np.float32)
            return zeros, self._projected, zeros

        def close(self) -> None:
            return None

    result = oracle._direct_first_read_qualification(
        context={},
        record={"input_ids": [1]},
        schedule={"rows": [{"write_slot": -1, "read_span": 0}]},
        beta_zero_arrays=beta_zero_arrays,
        output_projection=np.stack(
            [np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32)]
        ),
        runtime_factory=lambda _context, beta: FakeRuntime(beta),
    )
    assert result["passed"] is True
    assert result["qualified_layer"] == 0
    assert all(row["downstream_causal_output_max_abs"] > 0.0 for row in result["rows"])


def test_oracle_metrics_apply_every_frozen_gate(monkeypatch) -> None:
    monkeypatch.setattr(oracle, "_RECORDS", 2)
    monkeypatch.setattr(oracle, "_READ_POSITIONS", (96, 97))
    monkeypatch.setattr(oracle, "_BLOCK_ENTRY_POSITIONS", (96,))
    monkeypatch.setattr(oracle, "_LAYERS", 12)
    monkeypatch.setattr(oracle, "_HIDDEN_SIZE", 4)
    monkeypatch.setattr(oracle, "_QUERY_HEADS", 2)
    target = np.ones((2, 2, 12, 4), dtype=np.float32)
    correction = target.copy()
    selected = np.zeros((2, 2, 12, 2), dtype=np.float32)
    baseline = np.ones_like(selected)
    metrics = oracle._oracle_metrics(target, correction, selected, baseline)
    assert metrics["global"]["recovery"] == 1.0
    assert metrics["positive_recovery_layer_count"] == 12
    assert metrics["gate"]["passed"] is True
    assert metrics["passed"] is True


def test_oracle_metrics_reject_selected_mass_regression(monkeypatch) -> None:
    monkeypatch.setattr(oracle, "_RECORDS", 1)
    monkeypatch.setattr(oracle, "_READ_POSITIONS", (96,))
    monkeypatch.setattr(oracle, "_BLOCK_ENTRY_POSITIONS", (96,))
    monkeypatch.setattr(oracle, "_LAYERS", 1)
    monkeypatch.setattr(oracle, "_HIDDEN_SIZE", 2)
    monkeypatch.setattr(oracle, "_QUERY_HEADS", 1)
    target = np.ones((1, 1, 1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="metric inputs"):
        oracle._oracle_metrics(
            target,
            target,
            np.ones((1, 1, 1, 1), dtype=np.float32),
            np.zeros((1, 1, 1, 1), dtype=np.float32),
        )


def _tiny_trace_arrays(monkeypatch) -> dict[str, np.ndarray]:
    monkeypatch.setattr(oracle, "_READ_POSITIONS", (96,))
    monkeypatch.setattr(oracle, "_BLOCK_ENTRY_POSITIONS", (96,))
    monkeypatch.setattr(oracle, "_LAYERS", 1)
    monkeypatch.setattr(oracle, "_HIDDEN_SIZE", 2)
    monkeypatch.setattr(oracle, "_QUERY_HEADS", 1)
    monkeypatch.setattr(oracle, "_HEAD_DIMENSION", 2)
    base = np.asarray([[[1.0, 2.0]]], dtype=np.float32)
    regular = np.asarray([[[0.25, 0.5]]], dtype=np.float32)
    episodic = np.asarray([[[0.75, 1.5]]], dtype=np.float32)
    return {
        "base_attention_output": base,
        "regular_component": regular,
        "episodic_component": episodic,
        "regular_mass": np.asarray([[[0.25]]], dtype=np.float32),
        "episodic_mass": np.asarray([[[0.75]]], dtype=np.float32),
        "shadow_scheduled_mass": np.asarray([[[0.5]]], dtype=np.float32),
        "base_projected": base.copy(),
        "target_residual": np.asarray([[[0.1, -0.2]]], dtype=np.float32),
    }


def test_trace_summary_hashes_codes_and_partition(monkeypatch) -> None:
    arrays = _tiny_trace_arrays(monkeypatch)
    summary = oracle._trace_summary(arrays, [96])
    assert summary["gamma_one_exact_anchor"] is True
    assert summary["component_reconstruction_max_abs"] == 0.0
    assert summary["mass_partition_max_abs"] == 0.0
    assert summary["trace_sha256"] == oracle._trace_array_digest(arrays)
    assert len(summary["selected_code_stream_sha256"]) == 64


def test_trace_shard_is_create_only_and_hash_authenticated(
    tmp_path,
    monkeypatch,
) -> None:
    arrays = _tiny_trace_arrays(monkeypatch)
    record = {"record_index": 0, "record_id": "tiny"}
    descriptor = oracle._write_trace_shard(
        tmp_path,
        record=record,
        arrays=arrays,
        positions=[96],
        source_record_sha256=hashlib.sha256(b"record").hexdigest(),
        output_sha256=hashlib.sha256(b"output").hexdigest(),
        reset_output_sha256=hashlib.sha256(b"reset-output").hexdigest(),
        reset_trace_sha256=hashlib.sha256(b"reset-trace").hexdigest(),
    )
    loaded = oracle._validate_trace_shard(tmp_path / descriptor["file"], descriptor)
    np.testing.assert_array_equal(loaded["target_residual"], arrays["target_residual"])
    with pytest.raises(ValueError, match="already exists"):
        oracle._write_trace_shard(
            tmp_path,
            record=record,
            arrays=arrays,
            positions=[96],
            source_record_sha256=hashlib.sha256(b"record").hexdigest(),
            output_sha256=hashlib.sha256(b"output").hexdigest(),
            reset_output_sha256=hashlib.sha256(b"reset-output").hexdigest(),
            reset_trace_sha256=hashlib.sha256(b"reset-trace").hexdigest(),
        )


def test_prepare_shard_directory_rejects_symlink(tmp_path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="directory is invalid"):
        oracle._prepare_shard_directory(link)
