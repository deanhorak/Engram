import json

import numpy as np
import pytest

from engram.semantic.product_quantization import fit_product_additive
from engram.training.activation_aware_aq import (
    load_activation_aware_aq_encoding,
    project_activation_aware_aq_storage,
    run_activation_aware_aq_boundary_screen,
    save_activation_aware_aq_encoding,
)


def test_two_by_seven_projection_is_below_cold_q4_traffic_gate():
    projection = project_activation_aware_aq_storage(576, 1536)

    assert projection == {
        "records": 4608,
        "groups_per_record": 72,
        "code_bits": 7,
        "information_bits": 4_644_864,
        "packed_code_bytes": 580_608,
        "fp16_codebook_bytes": 4_096,
        "fp16_record_scale_bytes": 9_216,
        "total_payload_bytes": 593_920,
        "runtime_header_bytes": 64,
        "total_cold_bytes": 593_984,
        "dense_q4_bytes": 1_327_104,
        "fraction_of_dense_q4": pytest.approx(0.44757908719135804),
        "passes_45_percent_traffic_gate": True,
    }


def test_encoding_save_load_round_trip_is_exact_and_pickle_free(tmp_path):
    rng = np.random.default_rng(12)
    encoding = fit_product_additive(
        rng.normal(size=(24, 8)).astype(np.float32),
        group_size=4,
        num_codebooks=2,
        codebook_size=8,
        iterations=3,
        seed=4,
        per_record_scale=True,
    )
    path = tmp_path / "layer.aq.npz"

    save_activation_aware_aq_encoding(path, encoding)
    loaded = load_activation_aware_aq_encoding(path)

    assert loaded.metadata == encoding.metadata
    np.testing.assert_array_equal(loaded.packed_codes, encoding.packed_codes)
    np.testing.assert_array_equal(loaded.codebooks, encoding.codebooks)
    np.testing.assert_array_equal(loaded.record_scales, encoding.record_scales)


def test_p_only_screen_improves_cached_boundary_and_reports_reloaded_payload(
    tmp_path,
):
    pytest.importorskip("torch")
    rng = np.random.default_rng(91)
    hidden_size = 8
    intermediate_size = 12
    gate = (rng.normal(size=(intermediate_size, hidden_size)) * 0.28).astype(
        np.float32
    )
    up = (rng.normal(size=(intermediate_size, hidden_size)) * 0.25).astype(
        np.float32
    )
    down = (rng.normal(size=(hidden_size, intermediate_size)) * 0.22).astype(
        np.float32
    )

    def dense_output(inputs):
        gate_projection = inputs @ gate.T
        silu = gate_projection / (1.0 + np.exp(-gate_projection))
        return ((silu * (inputs @ up.T)) @ down.T).astype(np.float32)

    training_inputs = rng.normal(size=(96, hidden_size)).astype(np.float32)
    validation_inputs = rng.normal(size=(48, hidden_size)).astype(np.float32)
    artifact = tmp_path / "optimized.aq.npz"
    report_path = tmp_path / "optimized.report.json"
    result = run_activation_aware_aq_boundary_screen(
        gate,
        up,
        down,
        training_inputs,
        dense_output(training_inputs),
        validation_inputs,
        dense_output(validation_inputs),
        artifact_path=artifact,
        report_path=report_path,
        group_size=4,
        num_codebooks=2,
        codebook_size=8,
        fit_iterations=4,
        steps=24,
        batch_size=32,
        learning_rate=1e-2,
        checkpoint_interval=4,
        seed=7,
    )

    report = result.report
    initial = report["validation"]["initial_fixed_code_decoded"]
    strict = report["validation"]["strict_reloaded_decoded"]
    assert report["optimization_mode"] == "P_only_fixed_assignments"
    assert report["training"]["best_full_objective"] < report["training"][
        "initial_full_objective"
    ]
    assert strict["output_nmse"] < initial["output_nmse"]
    assert report["artifact"]["exact_array_round_trip"] is True
    assert report["artifact"]["validation_source"] == (
        "serialized_reloaded_fp16_payload"
    )
    assert result.artifact_path == artifact
    assert artifact.is_file()
    assert report_path.is_file()
    assert json.loads(report_path.read_text())["traffic"] == report["traffic"]
    assert result.encoding.storage_bytes == report["traffic"]["total_payload_bytes"]
    assert result.encoding.record_scales is not None
    assert np.all(result.encoding.record_scales > 0)

    # Assignment indices are genuinely fixed; only P (codebooks/scales) moved.
    initial_encoding = fit_product_additive(
        np.concatenate((gate, up, down.T), axis=0),
        group_size=4,
        num_codebooks=2,
        codebook_size=8,
        iterations=4,
        seed=7,
        per_record_scale=True,
    )
    np.testing.assert_array_equal(
        result.encoding.unpack_codes(), initial_encoding.unpack_codes()
    )


def test_screen_rejects_inconsistent_swiglu_shapes(tmp_path):
    pytest.importorskip("torch")
    values = np.ones((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="down_weight"):
        run_activation_aware_aq_boundary_screen(
            values,
            values,
            np.ones((2, 4), dtype=np.float32),
            np.ones((5, 3), dtype=np.float32),
            np.ones((5, 3), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
            artifact_path=tmp_path / "unused.npz",
            codebook_size=2,
            steps=1,
        )
