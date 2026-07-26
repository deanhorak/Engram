import numpy as np
import pytest

from engram.evaluation.native_bitnet_dip_kernel import (
    NativeBitNetDIPKernelPolicy,
    NativeBitNetDIPTorchDiagnostics,
)
from engram.evaluation.native_bitnet_dip_native_causal import (
    _aggregate_recall,
    _candidate_recall,
    _file_descriptor,
    _native_scored_schedule_bytes,
    _physical_traffic_report,
    _selection_report,
    _validate_native_calls,
)
from engram.utils import sha256_file


def _policies():
    return (
        NativeBitNetDIPKernelPolicy(
            input_coordinates=240,
            candidate_count=14,
            minimum_top_k=2,
            maximum_top_k=10,
            energy_target=1.0,
            rms_audit_count=0,
            rms_estimator="candidate_ratio",
            rms_audit_strategy="none",
        ),
        NativeBitNetDIPKernelPolicy(
            input_coordinates=240,
            candidate_count=15,
            minimum_top_k=2,
            maximum_top_k=10,
            energy_target=1.0,
            rms_audit_count=3,
            rms_estimator="corrected_proxy",
            rms_audit_strategy="top_proxy_raw_square",
        ),
    )


def test_native_causal_file_descriptor_is_hash_bound(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"native-dip-evidence")

    descriptor = _file_descriptor(path)

    assert descriptor == {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": len(b"native-dip-evidence"),
    }


def test_native_causal_selection_report_preserves_exact_token_layer_k():
    counts = {
        0: np.asarray([[2, 3], [4, 5]], dtype=np.uint32),
        1: np.asarray([[6, 7], [8, 9]], dtype=np.uint32),
    }

    report = _selection_report(
        counts,
        sequence_count=2,
        predictions_per_sequence=2,
        intermediate_size=20,
    )

    assert report["per_token_layer_k"] == [
        [2, 6],
        [3, 7],
        [4, 8],
        [5, 9],
    ]
    assert report["global"]["sum"] == 44
    assert report["global"]["count"] == 8
    assert report["global"]["active_fraction"] == pytest.approx(44 / 160)
    assert report["layers"]["0"]["sum"] == 14
    assert report["layers"]["1"]["sum"] == 30


def test_native_causal_physical_report_is_recomputable_per_token():
    schedules = [[2, 3], [4, 5]]

    report = _physical_traffic_report(
        hidden_size=320,
        intermediate_size=20,
        policies=_policies(),
        schedules=schedules,
        predictions_per_sequence=2,
    )

    assert report["accounting_version"] == (
        "native_bitnet_dip_dual_layout_v2"
    )
    assert [item["token"] for item in report["per_token"]] == [0, 1]
    assert report["global"]["scheduled_cache_line_bytes"] == sum(
        item["scheduled_cache_line_bytes"] for item in report["per_token"]
    )
    assert report["global"]["dense_q4_bytes"] == sum(
        item["dense_q4_bytes"] for item in report["per_token"]
    )
    assert report["global"]["fraction_of_dense_q4"] == pytest.approx(
        report["global"]["scheduled_cache_line_bytes"]
        / report["global"]["dense_q4_bytes"]
    )
    assert sum(
        item["scheduled_cache_line_bytes"] for item in report["layers"]
    ) < report["global"]["scheduled_cache_line_bytes"]
    assert report["worst_layer"]["top_k"] == 5


def test_native_causal_call_contract_fails_closed():
    calls = [
        {
            "layer": 0,
            "rows": 3,
            "input_coordinates": 240,
            "candidate_count": 14,
            "selected_count_total": 9,
            "selected_count_min": 2,
            "selected_count_max": 4,
        },
        {
            "layer": 1,
            "rows": 3,
            "input_coordinates": 240,
            "candidate_count": 15,
            "selected_count_total": 10,
            "selected_count_min": 2,
            "selected_count_max": 5,
        },
    ]
    _validate_native_calls(calls, policies=_policies(), rows=3)

    invalid = [dict(call) for call in calls]
    invalid[1]["rows"] = 2
    with pytest.raises(RuntimeError, match="rows"):
        _validate_native_calls(invalid, policies=_policies(), rows=3)


def test_native_scored_bytes_exclude_n_plus_one_context_rows():
    # Each layer has 1,000 fixed bytes per row plus 64 bytes per selected
    # record. The third row is the context-only N+1 row and must be excluded.
    calls = [
        {
            "scheduled_cache_line_bytes": 3_000 + 9 * 64,
            "selected_count_total": 9,
        },
        {
            "scheduled_cache_line_bytes": 3_000 + 12 * 64,
            "selected_count_total": 12,
        },
    ]
    counts = {
        0: np.asarray([[2, 3]], dtype=np.uint32),
        1: np.asarray([[4, 5]], dtype=np.uint32),
    }

    report = _native_scored_schedule_bytes(
        calls,
        layer_counts=counts,
        hidden_size=320,
        all_rows=3,
    )

    assert report["scored_rows_per_layer"] == 2
    assert report["layers"][0]["scheduled_cache_line_bytes"] == 2_000 + 5 * 64
    assert report["layers"][1]["scheduled_cache_line_bytes"] == 2_000 + 9 * 64
    assert report["scheduled_cache_line_bytes"] == 4_000 + 14 * 64


def test_native_causal_debug_recall_uses_fixed_reference_k():
    sentinel = np.iinfo(np.uint32).max

    class FakeKernel:
        hidden_size = 4
        intermediate_size = 6
        policies = (
            NativeBitNetDIPKernelPolicy(
                input_coordinates=2,
                candidate_count=4,
                minimum_top_k=2,
                maximum_top_k=3,
                energy_target=1.0,
                rms_audit_count=0,
                rms_estimator="candidate_ratio",
                rms_audit_strategy="none",
            ),
        )

        def teacher_top_k_bf16_bits(self, layer, hidden, *, top_k):
            assert layer == 0
            assert top_k == 3
            return np.tile(
                np.asarray([[0, 1, 2]], dtype=np.uint32),
                (hidden.shape[0], 1),
            )

        def teacher_top_k_with_positive_counts_bf16_bits(
            self,
            layer,
            hidden,
            *,
            top_k,
        ):
            return (
                self.teacher_top_k_bf16_bits(
                    layer,
                    hidden,
                    top_k=top_k,
                ),
                np.asarray([1, 6], dtype=np.uint32),
            )

    diagnostics = NativeBitNetDIPTorchDiagnostics(
        selected_counts=np.asarray([[2, 3, 2]], dtype=np.uint32),
        metrics={},
        input_coordinate_ids=np.asarray(
            [[[0, 1], [1, 2], [2, 3]]],
            dtype=np.uint32,
        ),
        candidate_ids=np.asarray(
            [[[0, 1, 2, 4], [0, 2, 4, 5], [0, 1, 3, 5]]],
            dtype=np.uint32,
        ),
        selected_record_ids=np.asarray(
            [[[0, 1, sentinel], [0, 2, 4], [0, 1, sentinel]]],
            dtype=np.uint32,
        ),
    )

    layer = _candidate_recall(
        FakeKernel(),
        layer=0,
        reference_top_k=3,
        hidden_bf16_bits=np.zeros((1, 3, 4), dtype=np.uint16),
        diagnostics=diagnostics,
        predictions_per_sequence=2,
    )
    aggregate = _aggregate_recall([layer])

    assert layer["target_records"] == 6
    assert layer["candidate_hits"] == 5
    assert layer["candidate_micro_recall"] == pytest.approx(5 / 6)
    assert layer["candidate_mean_row_recall"] == pytest.approx(5 / 6)
    assert layer["candidate_p05_row_recall"] == pytest.approx(
        2 / 3 + 0.05 * (1 - 2 / 3)
    )
    assert layer["selected_fixed_reference_hits"] == 4
    assert layer[
        "selected_fixed_reference_clipped_micro_recall"
    ] == pytest.approx(4 / 6)
    secondary = layer[
        "secondary_teacher_positive_utility_recall_clipped_to_"
        "frozen_minimum_and_maximum_k"
    ]
    assert secondary["clipped_target_count_sum"] == 5
    assert secondary["candidate_micro_recall"] == pytest.approx(4 / 5)
    assert secondary["selected_micro_recall"] == pytest.approx(4 / 5)
    assert aggregate["candidate_micro_recall"] == pytest.approx(5 / 6)
    assert aggregate["macro_mean_layer_recall"] == pytest.approx(5 / 6)
    assert aggregate[
        "secondary_teacher_positive_utility_recall_clipped_to_"
        "frozen_minimum_and_maximum_k"
    ]["candidate_micro_recall"] == pytest.approx(4 / 5)
    assert not aggregate["passes_95_percent"]


def test_native_causal_recall_requires_every_layer_mean_to_pass():
    aggregate = _aggregate_recall(
        [
            {
                "rows": 1,
                "target_records": 100,
                "candidate_hits": 100,
                "candidate_mean_row_recall": 1.0,
                "secondary_teacher_positive_utility_recall_clipped_to_"
                "frozen_minimum_and_maximum_k": {
                    "target_records": 50,
                    "candidate_hits": 50,
                    "candidate_mean_row_recall": 1.0,
                    "selected_hits": 25,
                    "selected_mean_row_recall": 0.5,
                },
            },
            {
                "rows": 1,
                "target_records": 10,
                "candidate_hits": 9,
                "candidate_mean_row_recall": 0.9,
                "secondary_teacher_positive_utility_recall_clipped_to_"
                "frozen_minimum_and_maximum_k": {
                    "target_records": 5,
                    "candidate_hits": 4,
                    "candidate_mean_row_recall": 0.8,
                    "selected_hits": 2,
                    "selected_mean_row_recall": 0.4,
                },
            },
        ]
    )

    assert aggregate["candidate_micro_recall"] == pytest.approx(109 / 110)
    assert aggregate["global_micro_passes_95_percent"]
    assert not aggregate["every_layer_mean_passes_95_percent"]
    assert not aggregate["passes_95_percent"]
