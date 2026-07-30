from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_full_visible_runner as runner
from engram.utils import sha256_file, sha256_json


class _FakeTrace:
    def __init__(self) -> None:
        self.position = 0
        self.reset_calls = 0
        self.closed = False

    def reset(self) -> None:
        self.position = 0
        self.reset_calls += 1

    def close(self) -> None:
        self.closed = True


def test_confirmation_path_is_rejected_before_authentication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False

    def forbidden_auth(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("authentication must not run")

    monkeypatch.setattr(
        runner,
        "_authenticate_predecessor_inputs",
        forbidden_auth,
    )
    with pytest.raises(ValueError, match="confirmation split"):
        runner.generate_full_visible_trace_parity_report(
            predecessor_protocol=tmp_path / "confirmation.jsonl",
            predecessor_protocol_sha256="a" * 64,
            predecessor_result=tmp_path / "result.json",
            predecessor_result_sha256="b" * 64,
            trace_library=tmp_path / "trace.so",
            trace_library_sha256="c" * 64,
            out=tmp_path / "parity.json",
        )
    assert not called
    assert not (tmp_path / "parity.json").exists()


def test_execute_record_pair_checks_historical_and_reset_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {"record_id": "train-00", "input_ids": [1, 2, 3]}
    evidence = {
        "record_index": 0,
        "record_id": "train-00",
        "answer_cross_entropy": 1.25,
        "hidden_sha256": "a" * 64,
        "logits_sha256": "b" * 64,
        "counter_stream_sha256": "c" * 64,
        "episodic_call_stream_sha256": "d" * 64,
        "counter_stream_passed": True,
    }
    output_root = sha256_json(runner.slot.mass.capacity._without_elapsed(evidence))
    context = {
        "train_records": [record],
        "historical_output_rows": [
            {
                "source_record_sha256": sha256_json(record),
                "observed_output_evidence_sha256": output_root,
            }
        ],
        "head_mass_protocol": {"fixed_K256_arm": {"resource_contract": {"bytes": 1}}},
    }
    arrays = {
        name: np.array([1], dtype=np.float32)
        for name in runner.full._CAPTURE_TRACE_KEYS
    }
    summary = {
        "trace_sha256": "e" * 64,
        "regular_component_reconstruction_max_abs": 0.0,
        "episodic_component_reconstruction_max_abs": 0.0,
        "base_component_reconstruction_max_abs": 0.0,
        "slot_values_exact_bf16_decodes": True,
    }
    executions = iter(
        (
            (dict(evidence), arrays, [96]),
            (
                dict(evidence),
                {name: value.copy() for name, value in arrays.items()},
                [96],
            ),
        )
    )
    monkeypatch.setattr(
        runner,
        "_historical_base_arrays",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        runner,
        "_record_schedule",
        lambda *_args, **_kwargs: {"rows_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        runner.slot,
        "_execute_record",
        lambda *_args, **_kwargs: next(executions),
    )
    monkeypatch.setattr(
        runner.full,
        "_trace_summary",
        lambda *_args, **_kwargs: dict(summary),
    )
    monkeypatch.setattr(
        runner.slot,
        "_common_trace_exact",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        runner.slot,
        "_historical_output_exact",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        runner.slot.mass.capacity,
        "_evidence_exact",
        lambda *_args, **_kwargs: True,
    )
    trace = _FakeTrace()
    result = runner._execute_record_pair(
        trace,
        context=context,
        record_index=0,
        progress_prefix="test",
    )
    assert result["checks"]["passed"]
    assert result["source_record_sha256"] == sha256_json(record)
    assert result["first_output_evidence_sha256"] == output_root
    assert result["reset_output_evidence_sha256"] == output_root
    assert trace.reset_calls == 1


def test_parity_report_binds_runner_and_execution_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = SimpleNamespace()
    trace = _FakeTrace()
    context = {
        "predecessor_protocol_path": tmp_path / "predecessor.json",
        "predecessor_protocol_sha256": "a" * 64,
        "predecessor_result_path": tmp_path / "result.json",
        "predecessor_result_sha256": "b" * 64,
        "full_visible_trace_library_path": tmp_path / "trace.so",
        "full_visible_trace_library_sha256": "c" * 64,
    }
    summary = {
        "trace_sha256": "d" * 64,
        "regular_component_reconstruction_max_abs": 0.0,
        "episodic_component_reconstruction_max_abs": 0.0,
        "base_component_reconstruction_max_abs": 0.0,
        "slot_values_exact_bf16_decodes": True,
    }
    checks = {
        "inherited_first_base_trace_exact": True,
        "inherited_reset_base_trace_exact": True,
        "historical_first_outputs_counters_and_loss_exact": True,
        "historical_reset_outputs_counters_and_loss_exact": True,
        "first_reset_outputs_counters_and_loss_exact": True,
        "passed": True,
    }
    evidence = {
        "record_index": 0,
        "record_id": "train-00",
        "schedule_rows_sha256": "e" * 64,
        "source_record_sha256": "f" * 64,
        "first_output_evidence_sha256": "1" * 64,
        "reset_output_evidence_sha256": "1" * 64,
        "first_trace": summary,
        "reset_trace": dict(summary),
        "checks": checks,
    }
    monkeypatch.setattr(
        runner,
        "_authenticate_predecessor_inputs",
        lambda **_kwargs: context,
    )
    monkeypatch.setattr(
        runner.full,
        "_FullVisibleTraceCaptureRuntime",
        lambda _raw: trace,
    )
    monkeypatch.setattr(runner, "_validate_runtime_route", lambda _raw: None)
    monkeypatch.setattr(
        runner,
        "_execute_record_pair",
        lambda *_args, **_kwargs: evidence,
    )
    monkeypatch.setattr(
        runner,
        "_post_run_authentication",
        lambda _context: {"all_roots": True},
    )
    monkeypatch.setattr(
        runner.full,
        "_source_inventory",
        lambda: {"native.cpp": "2" * 64},
    )
    monkeypatch.setattr(runner, "_runner_source_sha256", lambda: "3" * 64)
    result = runner.generate_full_visible_trace_parity_report(
        predecessor_protocol=tmp_path / "predecessor.json",
        predecessor_protocol_sha256="a" * 64,
        predecessor_result=tmp_path / "result.json",
        predecessor_result_sha256="b" * 64,
        trace_library=tmp_path / "trace.so",
        trace_library_sha256="c" * 64,
        out=tmp_path / "parity.json",
        runtime_factory=lambda _context: raw,
    )
    report = result["report"]
    assert report["execution_evidence"]["checks"]["passed"]
    assert report["runner_source_sha256"] == {runner._RUNNER_SOURCE: "3" * 64}
    assert report["instrumentation_source_sha256"] == {"native.cpp": "2" * 64}
    assert report["confirmation_split_opened"] is False
    assert trace.closed


def test_capture_writes_every_record_and_reauthenticates_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    context = {
        "full_visible_protocol_path": protocol_path,
        "full_visible_protocol_sha256": "a" * 64,
    }
    trace = _FakeTrace()
    written: list[int] = []
    monkeypatch.setattr(runner.full, "_RECORDS", 2)
    monkeypatch.setattr(
        runner,
        "_authenticate_frozen_capture_context",
        lambda **_kwargs: context,
    )
    monkeypatch.setattr(
        runner.full,
        "_prepare_shard_directory",
        lambda value: Path(value),
    )
    monkeypatch.setattr(
        runner.full,
        "_FullVisibleTraceCaptureRuntime",
        lambda _raw: trace,
    )
    monkeypatch.setattr(runner, "_validate_runtime_route", lambda _raw: None)

    def fake_pair(
        _trace,
        *,
        record_index: int,
        **_kwargs,
    ):
        root = str(record_index) * 64
        return {
            "record_index": record_index,
            "record_id": f"train-{record_index:02d}",
            "schedule_rows_sha256": "b" * 64,
            "source_record_sha256": "c" * 64,
            "first_output_evidence_sha256": "d" * 64,
            "reset_output_evidence_sha256": "d" * 64,
            "reset_trace": {"trace_sha256": root},
            "arrays": {},
            "query_positions": [96],
            "checks": {"passed": True},
        }

    monkeypatch.setattr(runner, "_execute_record_pair", fake_pair)

    def fake_write(_directory, *, record_index: int, **_kwargs):
        written.append(record_index)
        return {"record_index": record_index}

    monkeypatch.setattr(
        runner.full,
        "write_full_visible_trace_shard",
        fake_write,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        runner.full,
        "write_full_visible_trace_manifest",
        lambda *_args, **_kwargs: {
            "path": str(manifest_path),
            "sha256": "e" * 64,
            "manifest": {},
        },
    )
    monkeypatch.setattr(
        runner.full,
        "load_stacked_full_visible_trace",
        lambda *_args, **_kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        runner,
        "_capture_post_authentication",
        lambda *_args, **_kwargs: {"all_roots": True},
    )
    result = runner.capture_full_visible_train_traces(
        protocol=protocol_path,
        protocol_sha256="a" * 64,
        shard_directory=tmp_path / "shards",
        runtime_factory=lambda _context: SimpleNamespace(),
    )
    assert written == [0, 1]
    assert [row["record_index"] for row in result["capture_rows"]] == [0, 1]
    assert result["post_run_authentication"] == {"all_roots": True}
    assert result["confirmation_split_opened"] is False
    assert trace.closed


def test_solve_delegates_after_lexical_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def fake_solve(**kwargs):
        observed.update(kwargs)
        return {"status": "done"}

    monkeypatch.setattr(
        runner.full,
        "solve_cached_full_visible_capture",
        fake_solve,
    )
    monkeypatch.setattr(
        runner,
        "_authenticate_frozen_capture_context",
        lambda **_kwargs: {"authenticated": True},
    )
    monkeypatch.setattr(
        runner,
        "_authenticate_manifest_provenance",
        lambda *_args, **_kwargs: {"authenticated": True},
    )
    result = runner.solve_cached_full_visible_capture(
        protocol=tmp_path / "protocol.json",
        protocol_sha256="a" * 64,
        manifest=tmp_path / "manifest.json",
        manifest_sha256="b" * 64,
        out=tmp_path / "result.json",
        include_nested_diagnostics=False,
        row_batch_size=runner.full._ROW_BATCH_SIZE,
    )
    assert result == {"status": "done"}
    assert observed["include_nested_diagnostics"] is False
    assert observed["row_batch_size"] == runner.full._ROW_BATCH_SIZE

    observed.clear()
    with pytest.raises(ValueError, match="frozen protocol"):
        runner.solve_cached_full_visible_capture(
            protocol=tmp_path / "protocol.json",
            protocol_sha256="a" * 64,
            manifest=tmp_path / "manifest.json",
            manifest_sha256="b" * 64,
            out=tmp_path / "result.json",
            row_batch_size=3,
        )
    assert not observed

    with pytest.raises(ValueError, match="confirmation split"):
        runner.solve_cached_full_visible_capture(
            protocol=tmp_path / "protocol.json",
            protocol_sha256="a" * 64,
            manifest=tmp_path / "confirmation.jsonl",
            manifest_sha256="b" * 64,
            out=tmp_path / "result.json",
        )
    assert not observed


def test_manifest_provenance_cross_binds_history_and_schedule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner.full, "_RECORDS", 1)
    monkeypatch.setattr(runner.full, "_READ_POSITIONS", (96,))
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    record = {"record_id": "train-00", "input_ids": [1]}
    source_root = sha256_json(record)
    output_root = "a" * 64
    schedule_root = "b" * 64
    tensor_roots = {
        name: str(index % 10) * 64
        for index, name in enumerate(runner.full._CAPTURE_TRACE_KEYS)
    }
    descriptor = {
        "record_index": 0,
        "record_id": "train-00",
        "source_record_sha256": source_root,
        "output_evidence_sha256": output_root,
        "reset_output_evidence_sha256": output_root,
        "schedule_rows_sha256": schedule_root,
        "query_positions": [96],
        "tensor_sha256": tensor_roots,
    }
    inherited = {
        "record_index": 0,
        "record_id": "train-00",
        "source_record_sha256": source_root,
        "output_evidence_sha256": output_root,
        "reset_output_evidence_sha256": output_root,
        "positions": [96],
        "tensor_sha256": {
            name: tensor_roots[name] for name in runner.full._BASE_TRACE_KEYS
        },
    }
    slot_descriptor = {
        "record_index": 0,
        "record_id": "train-00",
        "source_record_sha256": source_root,
        "output_evidence_sha256": output_root,
        "reset_output_evidence_sha256": output_root,
        "positions": [96],
        "tensor_sha256": {
            name: tensor_roots[name] for name in runner.full._EPISODIC_SLOT_TRACE_KEYS
        },
    }
    historical = {
        "record_index": 0,
        "record_id": "train-00",
        "source_record_sha256": source_root,
        "historical_output_evidence_sha256": output_root,
        "observed_output_evidence_sha256": output_root,
        "reset_output_evidence_sha256": output_root,
    }
    context = {
        "full_visible_protocol_path": protocol_path.resolve(),
        "full_visible_protocol_sha256": "c" * 64,
        "full_visible_protocol": {
            "schedule_contract": {"per_record_rows_sha256": [schedule_root]}
        },
        "head_mass_manifest": {"shards": [inherited]},
        "cached_context": {"slot_manifest": {"shards": [slot_descriptor]}},
        "train_records": [record],
        "historical_output_rows": [historical],
    }
    manifest = {
        "schema_version": runner.full._SCHEMA_VERSION,
        "experiment": runner.full._CAPTURE_EXPERIMENT,
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": "c" * 64,
        },
        "record_order": [0],
        "shards": [descriptor],
        "confirmation_split_opened": False,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    result = runner._authenticate_manifest_provenance(
        context,
        manifest=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
    )
    assert result["shards"][0]["schedule_rows_sha256"] == schedule_root

    manifest["shards"][0]["schedule_rows_sha256"] = "d" * 64
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="provenance changed"):
        runner._authenticate_manifest_provenance(
            context,
            manifest=manifest_path,
            manifest_sha256=sha256_file(manifest_path),
        )
