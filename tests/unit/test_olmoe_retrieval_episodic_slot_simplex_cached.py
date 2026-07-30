from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_slot_simplex_cached as cached
from engram.utils import atomic_json, sha256_file


def _small_arrays(
    records: int = 1,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    regular_mean = np.asarray(
        [[[1.0, -0.5], [0.25, 0.75]]],
        dtype=np.float32,
    )
    slot_values = np.asarray(
        [
            [
                [[0.0, 1.0], [2.0, 0.5]],
                [[-0.5, 0.0], [1.0, 1.5]],
            ]
        ],
        dtype=np.float32,
    )
    regular_mass = np.asarray([[0.5, 0.5]], dtype=np.float32)
    slot_mass = np.asarray(
        [[[0.2, 0.3], [0.2, 0.3]]],
        dtype=np.float32,
    )
    episodic_mass = np.sum(slot_mass, axis=-1, dtype=np.float32)
    regular_component = regular_mean * regular_mass[..., None]
    episodic_component = np.einsum(
        "nhs,nhsd->nhd",
        slot_mass,
        slot_values,
        optimize=True,
    ).astype(np.float32)
    base = (regular_component + episodic_component).reshape(1, 4)
    target = np.asarray([[0.25, -0.5, 0.75, 0.125]], dtype=np.float32)
    row = {
        "base_attention_output": base,
        "regular_component": regular_component.reshape(1, 4),
        "episodic_component": episodic_component.reshape(1, 4),
        "regular_mass": regular_mass,
        "episodic_mass": episodic_mass,
        "base_projected": base.copy(),
        "target_residual": target,
        "slot_mass": slot_mass,
        "slot_values": slot_values,
    }
    arrays = {}
    for name, value in row.items():
        repeated = np.repeat(value, records, axis=0)
        arrays[name] = np.ascontiguousarray(
            repeated.reshape((records, 1, 1) + repeated.shape[1:])
        )
    projection = np.eye(4, dtype=np.float32)[None]
    return arrays, projection


def _patch_small_shape(
    monkeypatch: pytest.MonkeyPatch,
    *,
    records: int = 1,
) -> None:
    values = {
        "_RECORDS": records,
        "_READ_POSITIONS": (3,),
        "_LAYERS": 1,
        "_HIDDEN_SIZE": 4,
        "_QUERY_HEADS": 2,
        "_SLOTS": 2,
        "_HEAD_DIMENSION": 2,
        "_CONSTRUCTIBLE_COMPONENTS": 3,
        "_OPTIMISTIC_COMPONENTS": 4,
    }
    for name, value in values.items():
        monkeypatch.setattr(cached.slot, name, value)


def _fake_result(
    *,
    layers: int = 1,
    records: int = 1,
    positions: int = 1,
    heads: int = 1,
    components: int = 2,
    offset: float = 0.0,
) -> cached.slot.SlotSimplexOracleResult:
    rows = records * positions * layers
    values = np.arange(rows, dtype=np.float64) + offset
    coefficients = np.zeros((rows, heads, components), dtype=np.float64)
    coefficients[..., 0] = 1.0
    return cached.slot.SlotSimplexOracleResult(
        coefficients=coefficients,
        target_energy=values + 1.0,
        objective=values,
        objective_gap_upper_bound=np.zeros(rows, dtype=np.float64),
        direct_error_energy=values,
        iterations=np.ones(rows, dtype=np.int32),
        converged=np.ones(rows, dtype=bool),
        maximum_relative_objective_gap=0.0,
        base_reconstruction_max_abs=0.0,
        traced_partition_reconstruction_max_abs=0.0,
        episodic_component_reconstruction_max_abs=0.0,
        mass_partition_max_abs=0.0,
        quadratic_direct_error_energy_max_abs=0.0,
        deterministic_replay_exact=True,
        batch_shape=(records, positions, layers),
    )


def test_cli_exposes_only_cached_handoff_surfaces() -> None:
    parser = cached._build_parser()
    solve = parser.parse_args(
        [
            "solve-cached",
            "--protocol",
            "protocol.json",
            "--protocol-sha256",
            "0" * 64,
            "--out",
            "result.json",
        ]
    )
    assert vars(solve) == {
        "command": "solve-cached",
        "protocol": "protocol.json",
        "protocol_sha256": "0" * 64,
        "out": "result.json",
    }
    freeze = parser.parse_args(
        [
            "freeze-cached",
            "--capture-report",
            "capture.json",
            "--capture-report-sha256",
            "1" * 64,
            "--workers",
            "3",
            "--row-batch-size",
            "128",
            "--out",
            "protocol.json",
        ]
    )
    assert freeze.workers == 3
    assert freeze.row_batch_size == 128
    source = inspect.getsource(cached)
    assert "import ctypes" not in source
    assert "OLMoENativeTokenRuntime" not in source
    assert "CDLL(" not in source


def test_checked_file_rejects_confirmation_before_authentication() -> None:
    with pytest.raises(ValueError, match="cannot name the confirmation"):
        cached._checked_file(
            "/path/that/must/not/be/inspected/confirmation.jsonl",
            "0" * 64,
            "malformed cached binding",
        )


def test_capture_report_labels_reset_as_descriptor_attested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    roots = {
        "v1_protocol_path": tmp_path / "v1.json",
        "v1_protocol": {
            "authenticated_confirmation_descriptor": {
                "file": "confirmation.jsonl",
                "sha256": "a" * 64,
            }
        },
        "parity_path": tmp_path / "parity.json",
        "parity_sha256": "b" * 64,
        "slot_manifest_path": tmp_path / "slot.json",
        "inherited_manifest_path": tmp_path / "base.json",
        "inherited_manifest_sha256": "c" * 64,
        "head_mass_protocol_path": tmp_path / "head-protocol.json",
        "head_mass_protocol_sha256": "d" * 64,
        "head_mass_result_path": tmp_path / "head-result.json",
        "training_checkpoint_path": tmp_path / "checkpoint.json",
        "training_checkpoint_sha256": "7" * 64,
        "selector_protocol_path": tmp_path / "selector.json",
        "selector_protocol_sha256": "8" * 64,
        "corpus_manifest_path": tmp_path / "corpus.json",
        "corpus_manifest_sha256": "9" * 64,
        "train_path": tmp_path / "train.jsonl",
        "train_sha256": "0" * 64,
        "joint_protocol_path": tmp_path / "joint-protocol.json",
        "joint_protocol_sha256": "e" * 64,
        "joint_result_path": tmp_path / "joint-result.json",
        "joint_result_sha256": "f" * 64,
        "projection_path": tmp_path / "non-mlp.safetensors",
        "projection_file_sha256": "1" * 64,
        "projection_hashes": {"layer": "2" * 64},
        "output_projection": np.zeros((1, 1, 1), dtype=np.float32),
    }
    rows = [{"record_index": 0, "checks": {"authenticated": True}}]
    monkeypatch.setattr(
        cached,
        "_authenticate_historical_roots",
        lambda **_kwargs: roots,
    )
    monkeypatch.setattr(
        cached,
        "_audit_cached_rows",
        lambda _context, *, stack_arrays: (
            rows,
            {} if stack_arrays else None,
        ),
    )
    monkeypatch.setattr(cached.slot, "_RECORDS", 1)
    monkeypatch.setattr(cached.slot, "_READ_POSITIONS", (3,))
    context, report = cached._build_capture_report(
        v1_protocol="unused",
        v1_protocol_sha256="0" * 64,
        slot_manifest="unused",
        slot_manifest_sha256="0" * 64,
        head_mass_result="unused",
        head_mass_result_sha256="0" * 64,
    )
    assert context["stacked_arrays"] is None
    assert report["capture"]["native_rerun_performed"] is False
    reset = report["reset_replay_evidence"]
    assert reset["status"] == "descriptor_attested_not_independently_rederived"
    assert reset["reset_tensors_persisted"] is False
    assert reset["native_rerun_performed"] is False
    assert report["post_capture_authentication"]["confirmation_not_opened"]


def test_capture_report_authentication_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = {
        "schema_version": cached._SCHEMA_VERSION,
        "experiment": cached._CAPTURE_EXPERIMENT,
        "status": cached._CAPTURE_STATUS,
        "bindings": {
            "v1_protocol": {"path": "v1", "sha256": "1" * 64},
            "slot_trace_manifest": {"path": "slot", "sha256": "2" * 64},
            "inherited_head_mass_result": {
                "path": "result",
                "sha256": "3" * 64,
            },
        },
        "capture": {"native_rerun_performed": False},
        "reset_replay_evidence": {
            "status": "descriptor_attested_not_independently_rederived"
        },
        "confirmation_split_opened": False,
    }
    path = tmp_path / "capture.json"
    atomic_json(path, expected)
    monkeypatch.setattr(
        cached,
        "_build_capture_report",
        lambda **_kwargs: ({"validated": True}, expected),
    )
    source, observed, context = cached._authenticate_capture_report(
        path,
        sha256_file(path),
        stack_arrays=False,
    )
    assert source == path
    assert observed == expected
    assert context["validated"] is True

    changed = dict(expected)
    changed["reset_replay_evidence"] = {
        "status": "independently_rederived"
    }
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="contract changed"):
        cached._authenticate_capture_report(
            path,
            sha256_file(path),
            stack_arrays=False,
        )


def test_cached_protocol_freezes_solver_transition_and_process_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    solver_name = cached._REFERENCE_SOLVER_SOURCE
    active_name = cached._ACTIVE_SET_SOLVER_SOURCE
    capture = {
        "bindings": {"slot_trace_manifest": {"sha256": "2" * 64}},
        "capture_rows_sha256": "3" * 64,
        "output_projection": {
            "source": "non-mlp.safetensors",
            "file_sha256": "4" * 64,
            "tensor_sha256": {"layer": "5" * 64},
        },
        "reset_replay_evidence": {
            "status": "descriptor_attested_not_independently_rederived"
        },
        "authenticated_confirmation_descriptor": {
            "file": "confirmation.jsonl",
            "sha256": "6" * 64,
        },
    }
    context = {
        "v1_protocol": {
            "oracle_method": {
                "two_arms": True,
                "row_batch_size": 16,
                "maximum_iterations": 512,
                "relative_objective_gap_target": 1.0e-7,
            },
            "progression_gate": {"minimum_global_recovery": 0.5},
            "resource_contract": {"fixed_attention_state_bytes": 1},
            "source_sha256": {solver_name: "7" * 64},
            "authorized_next_step_on_feasible_pass": "next",
            "failure_interpretation": "closed",
        }
    }
    current = {
        name: "8" * 64 for name in cached._SOURCE_FILES
    }
    monkeypatch.setattr(cached, "_source_inventory", lambda: current)
    monkeypatch.setattr(cached.slot, "_LAYERS", 16)
    protocol = cached._build_cached_protocol(
        capture_path=tmp_path / "capture.json",
        capture_sha256="9" * 64,
        capture=capture,
        context=context,
        workers=8,
        row_batch_size=256,
    )
    transition = protocol["solver_source_transition"]
    assert transition["v1_frozen_solver_sha256"] == "7" * 64
    assert transition["cached_v2_reference_solver_sha256"] == "8" * 64
    assert transition["cached_v2_active_set_solver_sha256"] == "8" * 64
    assert transition["exact_v1_solver_source_replay_claimed"] is False
    assert active_name in protocol["source_sha256"]
    assert protocol["parallel_execution"]["workers"] == 8
    assert protocol["parallel_execution"]["task_unit"] == "one_layer_one_arm"
    assert protocol["oracle_method"]["row_batch_size"] == 256
    assert protocol["oracle_method"]["v1_capture_protocol_row_batch_size"] == 16
    assert protocol["oracle_method"]["maximum_active_set_iterations"] == 128
    assert protocol["oracle_method"]["fallback_maximum_iterations"] == 512
    assert protocol["oracle_method"]["relative_objective_gap_target"] == 1.0e-12
    assert protocol["oracle_method"]["absolute_objective_gap_target"] == 1.0e-13


def test_source_inventory_closes_transitive_oracle_and_solver_sources() -> None:
    expected = tuple(
        dict.fromkeys(
            (
                *cached.mass._SOURCE_FILES,
                *cached.slot._SOURCE_FILES,
                cached._REFERENCE_SOLVER_SOURCE,
                cached._ACTIVE_SET_SOLVER_SOURCE,
                cached._CACHED_SOURCE,
            )
        )
    )
    assert cached._SOURCE_FILES == expected
    assert len(cached._SOURCE_FILES) == len(set(cached._SOURCE_FILES))


def test_post_solve_authentication_rechecks_historical_protocol_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = {"source": "1" * 64}
    context = {
        "slot_manifest": {"shards": []},
        "inherited_manifest": {"shards": []},
        "slot_manifest_path": Path("slot/manifest.json"),
        "inherited_manifest_path": Path("inherited/manifest.json"),
        "cached_protocol_path": Path("cached-protocol.json"),
        "cached_protocol_sha256": "2" * 64,
        "capture_report_path": Path("capture.json"),
        "capture_report_sha256": "3" * 64,
        "v1_protocol_path": Path("v1-protocol.json"),
        "parity_path": Path("parity.json"),
        "parity_sha256": "4" * 64,
        "inherited_manifest_sha256": cached._EXPECTED_HEAD_MASS_MANIFEST_SHA256,
        "head_mass_protocol_path": Path("head-protocol.json"),
        "head_mass_result_path": Path("head-result.json"),
        "joint_protocol_path": Path("joint-protocol.json"),
        "joint_result_path": Path("joint-result.json"),
        "training_checkpoint_path": Path("checkpoint.json"),
        "training_checkpoint_sha256": "5" * 64,
        "selector_protocol_path": Path("selector.json"),
        "selector_protocol_sha256": "6" * 64,
        "corpus_manifest_path": Path("corpus.json"),
        "corpus_manifest_sha256": "7" * 64,
        "train_path": Path("train.jsonl"),
        "train_sha256": "8" * 64,
        "projection_path": Path("non-mlp.safetensors"),
        "projection_file_sha256": "9" * 64,
        "cached_protocol": {"source_sha256": sources},
    }
    expected = {
        str(context["cached_protocol_path"]): context["cached_protocol_sha256"],
        str(context["capture_report_path"]): context["capture_report_sha256"],
        str(context["v1_protocol_path"]): cached._EXPECTED_V1_PROTOCOL_SHA256,
        str(context["parity_path"]): context["parity_sha256"],
        str(context["slot_manifest_path"]): cached._EXPECTED_SLOT_MANIFEST_SHA256,
        str(context["inherited_manifest_path"]): (
            cached._EXPECTED_HEAD_MASS_MANIFEST_SHA256
        ),
        str(context["head_mass_protocol_path"]): (
            cached._EXPECTED_HEAD_MASS_PROTOCOL_SHA256
        ),
        str(context["head_mass_result_path"]): (
            cached._EXPECTED_HEAD_MASS_RESULT_SHA256
        ),
        str(context["joint_protocol_path"]): (
            cached._EXPECTED_JOINT_PROTOCOL_SHA256
        ),
        str(context["joint_result_path"]): cached._EXPECTED_JOINT_RESULT_SHA256,
        str(context["training_checkpoint_path"]): (
            context["training_checkpoint_sha256"]
        ),
        str(context["selector_protocol_path"]): (
            context["selector_protocol_sha256"]
        ),
        str(context["corpus_manifest_path"]): context["corpus_manifest_sha256"],
        str(context["train_path"]): context["train_sha256"],
        str(context["projection_path"]): context["projection_file_sha256"],
    }
    monkeypatch.setattr(cached, "sha256_file", lambda path: expected[str(path)])
    monkeypatch.setattr(cached, "_source_inventory", lambda: sources)
    checks = cached._post_solve_authentication(context)
    assert checks["head_mass_protocol"]
    assert checks["joint_gamma_protocol"]
    assert checks["joint_gamma_result"]
    assert all(checks.values())


def test_inconclusive_decision_does_not_claim_frozen_failure() -> None:
    frozen = {
        "authorized_next_step_on_feasible_pass": "selector",
        "failure_interpretation": "abandon this capacity direction",
    }
    assert (
        cached._decision_next_step(
            frozen,
            feasible_passed=True,
            decisive_failure=False,
        )
        == "selector"
    )
    assert (
        cached._decision_next_step(
            frozen,
            feasible_passed=False,
            decisive_failure=True,
        )
        == "abandon this capacity direction"
    )
    inconclusive = cached._decision_next_step(
        frozen,
        feasible_passed=False,
        decisive_failure=False,
    )
    assert "inconclusive" in inconclusive
    assert inconclusive != frozen["failure_interpretation"]


def test_layer_aggregation_restores_record_position_layer_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cached.slot, "_RECORDS", 2)
    monkeypatch.setattr(cached.slot, "_READ_POSITIONS", (3,))
    monkeypatch.setattr(cached.slot, "_LAYERS", 2)
    monkeypatch.setattr(cached.slot, "_QUERY_HEADS", 1)
    monkeypatch.setattr(cached.slot, "_CONSTRUCTIBLE_COMPONENTS", 2)
    layers = {
        0: _fake_result(records=2, offset=0.0),
        1: _fake_result(records=2, offset=10.0),
    }
    result = cached._aggregate_layer_results(
        layers,
        include_exact_native_anchor=False,
    )
    np.testing.assert_array_equal(
        result.objective,
        np.asarray([0.0, 10.0, 1.0, 11.0]),
    )
    assert result.batch_shape == (2, 1, 2)


def test_forked_active_set_solver_is_exact_across_row_batching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_small_shape(monkeypatch, records=2)
    arrays, projection = _small_arrays(records=2)
    constructible, optimistic = cached._run_parallel_two_arms(
        arrays,
        projection,
        workers=1,
        row_batch_size=2,
        maximum_active_set_iterations=64,
        fallback_maximum_iterations=64,
        relative_gap_tolerance=1.0e-9,
    )
    expected_constructible, expected_optimistic = cached._run_parallel_two_arms(
        arrays,
        projection,
        workers=1,
        row_batch_size=1,
        maximum_active_set_iterations=64,
        fallback_maximum_iterations=64,
        relative_gap_tolerance=1.0e-9,
    )
    for observed, expected in (
        (constructible, expected_constructible),
        (optimistic, expected_optimistic),
    ):
        np.testing.assert_array_equal(observed.coefficients, expected.coefficients)
        np.testing.assert_array_equal(observed.objective, expected.objective)
        np.testing.assert_array_equal(
            observed.objective_gap_upper_bound,
            expected.objective_gap_upper_bound,
        )
        np.testing.assert_array_equal(
            observed.direct_error_energy,
            expected.direct_error_energy,
        )
        assert observed.deterministic_replay_exact


def test_solve_cached_uses_only_authenticated_arrays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _fake_result(components=2)
    protocol = {
        "oracle_method": {
            "row_batch_size": 1,
            "maximum_active_set_iterations": 2,
            "fallback_maximum_iterations": 2,
            "relative_objective_gap_target": 1.0e-7,
            "absolute_objective_gap_target": 1.0e-11,
            "working_set_tolerance": 1.0e-12,
            "kkt_residual_tolerance": 1.0e-10,
            "reduced_cost_tolerance": 1.0e-12,
        },
        "parallel_execution": {"workers": 1, "task_unit": "one_layer_one_arm"},
        "resource_contract": {"bytes": 1},
        "authorized_next_step_on_feasible_pass": "next",
        "failure_interpretation": "closed",
        "reset_replay_evidence": {
            "status": "descriptor_attested_not_independently_rederived"
        },
        "output_projection": {"tensor_sha256": {"layer": "a" * 64}},
    }
    capture = {"capture_rows_sha256": "b" * 64}
    context = {
        "stacked_arrays": {"tensor": np.zeros(1, dtype=np.float32)},
        "output_projection": np.zeros((1, 1, 1), dtype=np.float32),
        "cached_protocol_path": tmp_path / "protocol.json",
        "cached_protocol_sha256": "c" * 64,
        "capture_report_path": tmp_path / "capture.json",
        "capture_report_sha256": "d" * 64,
    }
    monkeypatch.setattr(
        cached,
        "_authenticate_cached_protocol",
        lambda *_args, **_kwargs: (
            context["cached_protocol_path"],
            protocol,
            capture,
            context,
        ),
    )
    monkeypatch.setattr(
        cached,
        "_run_parallel_two_arms",
        lambda *_args, **_kwargs: (result, result),
    )

    def summarize(_result, *, arm: str, component_count: int):
        return {
            "arm": arm,
            "components_per_head": component_count,
            "qualification": {"passed": True},
            "feasible_gate_passed": False,
            "optimistic_gate_passed": False,
        }

    monkeypatch.setattr(cached.slot, "_summarize_oracle_arm", summarize)
    monkeypatch.setattr(
        cached,
        "_post_solve_authentication",
        lambda _context: {"all_cached_roots": True},
    )
    monkeypatch.setattr(cached.slot, "_RECORDS", 1)
    monkeypatch.setattr(cached.slot, "_READ_POSITIONS", (3,))
    monkeypatch.setattr(cached.slot, "_CONSTRUCTIBLE_COMPONENTS", 2)
    monkeypatch.setattr(cached.slot, "_OPTIMISTIC_COMPONENTS", 2)
    monkeypatch.setattr(
        cached.slot.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native library must not be opened")
        ),
    )
    report = cached.solve_cached_slot_simplex(
        protocol="ignored",
        protocol_sha256="e" * 64,
        out=tmp_path / "result.json",
    )
    assert report["status"] == "train_episodic_slot_simplex_cached_v2_gate_failed"
    assert report["scope"]["native_execution_performed"] is False
    assert report["cache_authentication"]["partial_v1_solver_output_used"] is False
    assert report["confirmation_split_opened"] is False
