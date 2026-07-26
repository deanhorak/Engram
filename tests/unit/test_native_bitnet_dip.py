import numpy as np
import pytest

from engram.models.native_bitnet import (
    NativeBitNetLayerWeights,
    load_native_bitnet_artifact,
    native_bitnet_mlp_forward,
    save_native_bitnet_artifact,
)
from engram.semantic.native_bitnet_dip import (
    NativeBitNetDIPConfiguration,
    NativeBitNetDIPLayer,
    build_native_bitnet_dip_mlp,
    substitute_native_bitnet_dip_mlps,
)


def _artifact(tmp_path):
    layer = NativeBitNetLayerWeights(
        gate_codes=np.asarray(
            [
                [1, 1, 1, 1],
                [1, -1, 0, 0],
                [-1, -1, -1, -1],
                [0, 1, 0, -1],
            ],
            dtype=np.int8,
        ),
        up_codes=np.asarray(
            [
                [1, 1, 0, 0],
                [1, 1, 1, 1],
                [1, 0, -1, 0],
                [0, 1, 1, 0],
            ],
            dtype=np.int8,
        ),
        down_codes=np.asarray(
            [
                [1, 0, 1, 0],
                [0, 1, 0, -1],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
            ],
            dtype=np.int8,
        ),
        gate_scale=0.5,
        up_scale=0.25,
        down_scale=0.5,
        ffn_sub_norm=np.asarray([1.0, 0.75, 1.25, -0.5], dtype=np.float32),
    )
    path = tmp_path / "native-dip.bitnet-records.bin"
    save_native_bitnet_artifact(path, [layer], rms_norm_eps=1e-5)
    return load_native_bitnet_artifact(path)


def test_full_candidate_and_selection_path_matches_dense_teacher(tmp_path):
    artifact = _artifact(tmp_path)
    states = np.asarray(
        [[1.0, 0.5, -0.25, 0.125], [-0.5, 0.75, 0.25, -1.0]],
        dtype=np.float32,
    )
    layer = NativeBitNetDIPLayer(
        artifact,
        0,
        input_fraction=0.5,
        candidate_count=4,
        top_k=4,
    )

    result = layer(states, oracle_diagnostics=True)

    assert result.diagnostics is not None
    np.testing.assert_array_equal(
        result.output,
        result.diagnostics.dense_output,
    )
    # The analytical model oracle retains float32 between operators.  The
    # deployed native teacher deliberately stores BF16 at each boundary.
    np.testing.assert_allclose(
        result.output,
        native_bitnet_mlp_forward(artifact, 0, states),
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(result.diagnostics.candidate_recall, 1.0)
    np.testing.assert_allclose(result.diagnostics.output_relative_l2, 0.0, atol=2e-6)
    np.testing.assert_array_equal(
        result.diagnostics.cancellation_code_mismatches,
        0,
    )
    assert np.all(result.rms_q8_cancellation_applied)


def test_selected_down_path_uses_stable_exact_candidate_rerank(tmp_path):
    artifact = _artifact(tmp_path)
    layer = NativeBitNetDIPLayer(
        artifact,
        0,
        input_fraction=0.5,
        candidate_count=3,
        top_k=2,
    )

    first = layer(np.ones((1, 4), dtype=np.float32), oracle_diagnostics=True)
    second = layer(np.ones((1, 4), dtype=np.float32), oracle_diagnostics=True)

    np.testing.assert_array_equal(first.input_indices, second.input_indices)
    np.testing.assert_array_equal(first.candidate_indices, second.candidate_indices)
    np.testing.assert_array_equal(first.selected_indices, second.selected_indices)
    assert len(np.unique(first.candidate_indices[0])) == 3
    assert len(np.unique(first.selected_indices[0])) == 2
    assert set(first.selected_indices[0]).issubset(first.candidate_indices[0])
    assert np.all(np.isfinite(first.output))
    assert first.diagnostics is not None


def test_rms_tail_audit_is_deterministic_and_stays_inside_candidate_budget(tmp_path):
    artifact = _artifact(tmp_path)
    layer = NativeBitNetDIPLayer(
        artifact,
        0,
        input_fraction=0.5,
        candidate_count=3,
        top_k=1,
        rms_audit_count=1,
    )

    first = layer(np.asarray([[1.0, 0.5, -0.25, 0.125]], dtype=np.float32))
    second = layer(np.asarray([[1.0, 0.5, -0.25, 0.125]], dtype=np.float32))

    np.testing.assert_array_equal(first.candidate_indices, second.candidate_indices)
    assert len(np.unique(first.candidate_indices[0])) == 3
    assert first.selected_indices[0, 0] in first.candidate_indices[0]
    assert np.all(np.isfinite(first.estimated_raw_variance))


def test_adaptive_energy_selection_drops_zero_utility_records_with_fixed_padding(
    tmp_path,
):
    artifact = _artifact(tmp_path)
    layer = NativeBitNetDIPLayer(
        artifact,
        0,
        input_fraction=1.0,
        candidate_count=4,
        top_k=4,
        energy_target=1.0,
        minimum_top_k=1,
        maximum_top_k=4,
    )

    result = layer(np.ones((1, 4), dtype=np.float32))

    assert result.selected_counts.tolist() == [1]
    assert result.selected_indices[0, 0] == 0
    assert result.selected_indices[0, 1:].tolist() == [-1, -1, -1]
    np.testing.assert_array_equal(result.selected_coefficients[0, 1:], 0.0)


def test_adaptive_energy_selection_respects_minimum_and_maximum(tmp_path):
    artifact = _artifact(tmp_path)
    state = np.asarray([[1.0, 0.5, -0.25, 0.125]], dtype=np.float32)
    minimum = NativeBitNetDIPLayer(
        artifact,
        0,
        input_fraction=1.0,
        candidate_count=4,
        top_k=3,
        energy_target=0.01,
        minimum_top_k=2,
        maximum_top_k=3,
    )(state)
    maximum = NativeBitNetDIPLayer(
        artifact,
        0,
        input_fraction=1.0,
        candidate_count=4,
        top_k=1,
        energy_target=1.0,
        minimum_top_k=1,
        maximum_top_k=1,
    )(state)

    assert minimum.selected_counts.tolist() == [2]
    assert maximum.selected_counts.tolist() == [1]


def test_causal_torch_wrapper_preserves_shape_dtype_and_dense_full_width(tmp_path):
    torch = pytest.importorskip("torch")
    artifact = _artifact(tmp_path)
    module = build_native_bitnet_dip_mlp(
        artifact,
        0,
        input_fraction=0.5,
        candidate_count=4,
        top_k=4,
        oracle_diagnostics=True,
    )
    states = torch.tensor(
        [[[1.0, 0.5, -0.25, 0.125]]],
        dtype=torch.bfloat16,
    )

    with torch.inference_mode():
        actual = module(states)

    expected = native_bitnet_mlp_forward(
        artifact,
        0,
        states.float().numpy(),
    )
    assert actual.shape == states.shape
    assert actual.dtype == states.dtype
    np.testing.assert_allclose(
        actual.float().numpy(),
        expected,
        rtol=5e-3,
        atol=5e-3,
    )
    assert module.last_result is not None
    assert module.last_result.diagnostics is not None


def test_causal_substitution_hook_restores_original_mlp(tmp_path):
    torch = pytest.importorskip("torch")
    from torch import nn

    artifact = _artifact(tmp_path)

    class DecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Identity()

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([DecoderLayer()])

    class CausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()

    model = CausalLM()
    original = model.model.layers[0].mlp
    with substitute_native_bitnet_dip_mlps(
        model,
        artifact,
        {
            0: NativeBitNetDIPConfiguration(
                input_fraction=0.5,
                candidate_count=4,
                top_k=4,
            )
        },
    ) as replacements:
        assert model.model.layers[0].mlp is replacements[0]
        with torch.inference_mode():
            output = model.model.layers[0].mlp(
                torch.ones((1, 1, 4), dtype=torch.bfloat16)
            )
        assert output.shape == (1, 1, 4)

    assert model.model.layers[0].mlp is original


@pytest.mark.parametrize(
    ("candidate_count", "top_k", "message"),
    [
        (0, 1, "positive"),
        (5, 1, "intermediate"),
        (2, 3, "exceeds"),
    ],
)
def test_native_bitnet_dip_validates_selection_counts(
    tmp_path,
    candidate_count,
    top_k,
    message,
):
    with pytest.raises(ValueError, match=message):
        NativeBitNetDIPLayer(
            _artifact(tmp_path),
            0,
            input_fraction=0.5,
            candidate_count=candidate_count,
            top_k=top_k,
        )


def test_native_bitnet_dip_validates_rms_audit_budget(tmp_path):
    with pytest.raises(ValueError, match="minus rms_audit_count"):
        NativeBitNetDIPLayer(
            _artifact(tmp_path),
            0,
            input_fraction=0.5,
            candidate_count=2,
            top_k=2,
            rms_audit_count=1,
        )


def test_native_bitnet_dip_validates_adaptive_energy_budget(tmp_path):
    artifact = _artifact(tmp_path)
    with pytest.raises(ValueError, match="energy_target"):
        NativeBitNetDIPLayer(
            artifact,
            0,
            input_fraction=0.5,
            candidate_count=4,
            top_k=2,
            energy_target=0.0,
        )
    with pytest.raises(ValueError, match="minimum_top_k exceeds"):
        NativeBitNetDIPLayer(
            artifact,
            0,
            input_fraction=0.5,
            candidate_count=4,
            top_k=2,
            energy_target=1.0,
            minimum_top_k=3,
            maximum_top_k=2,
        )


def test_native_bitnet_dip_applies_bounded_output_scale_and_validates_calibration(
    tmp_path,
):
    artifact = _artifact(tmp_path)
    state = np.asarray([[1.0, 0.5, -0.25, 0.125]], dtype=np.float32)
    baseline = NativeBitNetDIPLayer(
        artifact,
        0,
        input_fraction=1.0,
        candidate_count=4,
        top_k=4,
    )(state)
    scaled = NativeBitNetDIPLayer(
        artifact,
        0,
        input_fraction=1.0,
        candidate_count=4,
        top_k=4,
        output_scale=1.5,
    )(state)

    np.testing.assert_allclose(
        scaled.output,
        baseline.output * 1.5,
        rtol=5e-3,
        atol=5e-3,
    )
    with pytest.raises(ValueError, match="rms_variance_scale"):
        NativeBitNetDIPLayer(
            artifact,
            0,
            input_fraction=1.0,
            candidate_count=4,
            top_k=4,
            rms_variance_scale=0.0,
        )
    with pytest.raises(ValueError, match="rms_variance_bias"):
        NativeBitNetDIPLayer(
            artifact,
            0,
            input_fraction=1.0,
            candidate_count=4,
            top_k=4,
            rms_variance_bias=np.nan,
        )
    with pytest.raises(ValueError, match="rms_estimator"):
        NativeBitNetDIPLayer(
            artifact,
            0,
            input_fraction=1.0,
            candidate_count=4,
            top_k=4,
            rms_estimator="unknown",
        )
    with pytest.raises(ValueError, match="rms_audit_strategy"):
        NativeBitNetDIPLayer(
            artifact,
            0,
            input_fraction=1.0,
            candidate_count=4,
            top_k=3,
            rms_audit_count=1,
            rms_audit_strategy="unknown",
        )
