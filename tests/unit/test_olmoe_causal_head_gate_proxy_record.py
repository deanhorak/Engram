from __future__ import annotations

import json
from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import pytest

import engram.evaluation.olmoe_causal_head_gate_proxy_record as proxy_record
from engram.utils import sha256_file


def _mask() -> np.ndarray:
    return np.zeros((16, 16), dtype=np.bool_)


def _record() -> dict:
    gradient = np.arange(1, 257, dtype=np.float64).reshape(16, 16)
    return {
        "sequence_index": 0,
        "record_id": "clock-tower-restoration",
        "mask_sha256": "a" * 64,
        "selected_head_count": 0,
        "loss": {
            "kl": 0.1,
            "hidden_relative_l2": 0.2,
            "positive_nll_delta": 0.3,
            "top1_margin_deficit": 0.4,
            "total": 5.0,
            "bands": {"positions_16_31_kl": 0.01},
        },
        "backward": True,
        "gradient": gradient.tolist(),
        "native_oracle_layers": [
            {
                "layer": layer,
                "sparse": {
                    "exact_forward_sha256": f"{layer:064x}",
                    "elapsed_seconds": 1.0 + layer,
                    "schedule": {
                        "indices_sha256": f"{layer + 16:064x}",
                        "native_metrics": {"tokens_seen": 128},
                        "elapsed_seconds": 0.1,
                    },
                },
                "full": {
                    "exact_forward_sha256": f"{layer + 32:064x}",
                    "total_elapsed_seconds": 0.2,
                },
            }
            for layer in range(16)
        ],
        "native_oracle_timing": {
            "layers": 16,
            "native_identity_schedule_seconds": 1.0,
            "native_sparse_actual_value_seconds": 2.0,
            "native_full_actual_value_seconds": 3.0,
            "sparse_surrogate_seconds": 4.0,
            "full_surrogate_seconds": 5.0,
        },
        "elapsed_seconds": 100.0,
    }


def _valid_stats() -> dict:
    return {
        "workers": 12,
        "patched_layers": 16,
        "serial_forward_calls": 16,
        "serial_forward_seconds": 12.0,
        "parallel_backward_calls": 16,
        "expert_backward_tasks": 900,
        "parallel_backward_task_seconds": 30.0,
        "ordered_reduction_seconds": 1.0,
        "restored_layers": 16,
        "context_active": False,
        "executor_shutdown": True,
    }


def test_timing_only_differences_preserve_complete_exact_parity():
    archived = _record()
    proxy = deepcopy(archived)
    proxy["elapsed_seconds"] = 42.0
    proxy["native_oracle_timing"]["native_identity_schedule_seconds"] = 9.0
    proxy["native_oracle_layers"][0]["sparse"]["elapsed_seconds"] = 8.0
    proxy["native_oracle_layers"][0]["full"]["total_elapsed_seconds"] = 7.0

    parity = proxy_record._record_parity(
        archived,
        proxy,
        mask=_mask(),
    )

    assert parity["exact"]
    assert all(parity["checks"].values())
    assert (
        parity["archived"]["semantic_record_sha256"]
        == parity["proxy"]["semantic_record_sha256"]
    )


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            lambda value: value["loss"].__setitem__("total", 5.5),
            "loss_exact",
        ),
        (
            lambda value: value["gradient"][0].__setitem__(0, 1.25),
            "gate_gradient_exact",
        ),
        (
            lambda value: value["native_oracle_layers"][0]["sparse"].__setitem__(
                "exact_forward_sha256",
                "f" * 64,
            ),
            "native_non_timing_diagnostics_exact",
        ),
        (
            lambda value: value.__setitem__("record_id", "different-record"),
            "record_identity_exact",
        ),
    ],
)
def test_semantic_tampering_fails_exact_parity(mutation, failed_check):
    archived = _record()
    proxy = deepcopy(archived)
    mutation(proxy)

    parity = proxy_record._record_parity(
        archived,
        proxy,
        mask=_mask(),
    )

    assert not parity["exact"]
    assert not parity["checks"][failed_check]
    assert not parity["checks"]["complete_non_timing_record_exact"]


def test_gradient_tamper_that_changes_projection_fails_score_and_mask():
    archived = _record()
    proxy = deepcopy(archived)
    proxy["gradient"][15][15] = -10000.0

    parity = proxy_record._record_parity(
        archived,
        proxy,
        mask=_mask(),
    )

    assert not parity["checks"]["gate_gradient_exact"]
    assert not parity["checks"]["projected_score_exact"]
    assert not parity["checks"]["projected_mask_exact"]
    assert (
        parity["archived"]["projected_mask_sha256"]
        != parity["proxy"]["projected_mask_sha256"]
    )
    assert len(parity["proxy"]["projected_flat_indices"]) == 51


def test_strip_timing_fields_removes_only_seconds_suffix_recursively():
    value = {
        "elapsed_seconds": 1.0,
        "seconds": 2.0,
        "semantic_timing": 3.0,
        "children": [
            {
                "total_elapsed_seconds": 4.0,
                "tokens_seen": 128,
            }
        ],
    }

    assert proxy_record._strip_timing_fields(value) == {
        "seconds": 2.0,
        "semantic_timing": 3.0,
        "children": [{"tokens_seen": 128}],
    }


def test_performance_rule_requires_at_least_ten_percent_improvement():
    boundary = proxy_record._performance_comparison(100.0, 90.0)
    insufficient = proxy_record._performance_comparison(100.0, 90.000001)

    assert boundary["material_improvement"]
    assert boundary["improvement_fraction"] == pytest.approx(0.1)
    assert not insufficient["material_improvement"]
    assert boundary["speedup_vs_archived_reference"] == pytest.approx(10.0 / 9.0)


@pytest.mark.parametrize(
    "archived,proxy",
    [
        (0.0, 1.0),
        (1.0, 0.0),
        (float("nan"), 1.0),
        (1.0, float("inf")),
        (True, 1.0),
    ],
)
def test_performance_rule_rejects_invalid_timing(archived, proxy):
    with pytest.raises(ValueError, match="timing"):
        proxy_record._performance_comparison(archived, proxy)


def test_proxy_stats_require_full_lifecycle_and_nested_components():
    checks = proxy_record._proxy_stats_checks(
        _valid_stats(),
        record_elapsed_seconds=100.0,
    )
    assert all(checks.values())

    invalid = _valid_stats()
    invalid["restored_layers"] = 15
    invalid["expert_backward_tasks"] = True
    invalid["parallel_backward_task_seconds"] = 100.0
    invalid_checks = proxy_record._proxy_stats_checks(
        invalid,
        record_elapsed_seconds=100.0,
    )
    assert not invalid_checks["restored_all_layers"]
    assert not invalid_checks["expert_task_population_valid"]
    assert not invalid_checks["component_times_nested_in_record"]


def test_proxy_execution_enters_one_context_and_runs_one_record(monkeypatch):
    calls = []
    snapshot = _valid_stats()

    class Stats:
        def snapshot(self):
            calls.append("snapshot")
            return snapshot

    @contextmanager
    def install(loaded, *, workers):
        calls.append(("enter", loaded, workers))
        yield Stats()
        calls.append("exit")

    expected_record = _record()

    def run(loaded, gate_state, **kwargs):
        calls.append(("run", loaded, gate_state, kwargs))
        return expected_record

    class Parameter:
        grad = None

    class Loaded:
        def parameters(self):
            return [Parameter(), Parameter()]

    loaded = Loaded()
    monkeypatch.setattr(
        proxy_record.expert_proxy,
        "frozen_olmoe_expert_backward_proxy",
        install,
    )
    monkeypatch.setattr(proxy_record.causal_gate, "_run_gate_record", run)

    record, timing, actual_snapshot, frozen = proxy_record._execute_proxy_record(
        torch=object(),
        loaded=loaded,
        gate_state={"state": True},
        mask=_mask(),
        context={"record": True},
        teacher_logits=np.zeros((128, 3), dtype=np.float32),
        teacher_hidden=np.zeros((128, 2), dtype=np.float32),
        targets=np.zeros(128, dtype=np.int64),
        bands=[],
    )

    assert record is expected_record
    assert actual_snapshot == snapshot
    assert frozen
    assert timing["wall_seconds"] > 0.0
    assert calls[0] == ("enter", loaded, 12)
    assert calls[1][0] == "run"
    assert calls[2:] == ["exit", "snapshot"]
    assert calls[1][3]["sequence_index"] == 0
    assert calls[1][3]["backward"] is True


def test_source_authentication_rejects_wrong_digest(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("sentinel\n", encoding="utf-8")

    assert proxy_record._authenticated_source(
        source,
        sha256_file(source),
        "test source",
    ) == sha256_file(source)
    with pytest.raises(ValueError, match="SHA-256"):
        proxy_record._authenticated_source(source, "0" * 64, "test source")


def test_atomic_report_write_rejects_existing_target(tmp_path):
    output = tmp_path / "nested" / "result.json"
    resolved = proxy_record._new_output_path(output)
    proxy_record._write_new_report(resolved, {"passed": True})

    assert json.loads(output.read_text(encoding="utf-8")) == {"passed": True}
    assert not list(output.parent.glob(f".{output.name}.tmp-*"))
    with pytest.raises(ValueError, match="already exists"):
        proxy_record._new_output_path(output)
    with pytest.raises(ValueError, match="already exists"):
        proxy_record._write_new_report(resolved, {"passed": False})


def test_public_benchmark_rejects_existing_output_before_authentication(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        proxy_record.benchmark_frozen_olmoe_expert_proxy_full_record(
            gate_protocol="missing",
            gate_protocol_sha256="0" * 64,
            gate_training_result="missing",
            gate_training_result_sha256="0" * 64,
            attention_library="missing",
            attention_library_sha256="0" * 64,
            expert_proxy_source_sha256="0" * 64,
            benchmark_source_sha256="0" * 64,
            transformers_moe_source_sha256="0" * 64,
            transformers_modeling_utils_source_sha256="0" * 64,
            out=output,
            manifest_sha256="0" * 64,
        )
