import pytest

from engram.training.projection_normalized_ternary import (
    projection_normalized_ternary_mlp_class,
    projection_normalized_ternary_traffic,
)


def _fixture(torch):
    generator = torch.Generator().manual_seed(2203)
    gate = torch.randn(7, 5, generator=generator) * 0.2
    up = torch.randn(7, 5, generator=generator) * 0.2
    down = torch.randn(5, 7, generator=generator) * 0.2
    hidden = torch.randn(6, 5, generator=generator)
    return gate, up, down, hidden


def test_zero_fraction_is_exact_teacher_mlp():
    torch = pytest.importorskip("torch")
    gate, up, down, hidden = _fixture(torch)
    Module = projection_normalized_ternary_mlp_class(torch)
    module = Module(gate, up, down)

    expected = module.teacher_forward(hidden)
    actual = module(hidden, 0.0)

    torch.testing.assert_close(actual, expected)


def test_hard_forward_uses_ternary_codes_and_projection_norms():
    torch = pytest.importorskip("torch")
    gate, up, down, hidden = _fixture(torch)
    Module = projection_normalized_ternary_mlp_class(torch)
    module = Module(gate, up, down)

    output = module(hidden, 1.0)
    state = module.deployment_state()

    assert output.shape == (6, 5)
    assert state["format"] == "engram_projection_normalized_ternary_v1"
    for name, input_size in (("gate", 5), ("up", 5), ("down", 7)):
        projection = state[name]
        assert set(projection["code"].unique().tolist()) <= {-1, 0, 1}
        assert projection["row_scale"].dtype == torch.float16
        assert projection["rms_gain"].dtype == torch.float16
        assert projection["rms_gain"].numel() == input_size


def test_ste_trains_master_scales_and_norm_gains():
    torch = pytest.importorskip("torch")
    gate, up, down, hidden = _fixture(torch)
    Module = projection_normalized_ternary_mlp_class(torch)
    module = Module(gate, up, down)

    module(hidden, 1.0).square().mean().backward()

    for projection in (module.gate, module.up, module.down):
        for parameter in (
            projection.master,
            projection.log_row_scale,
            projection.rms_gain,
        ):
            assert parameter.grad is not None
            assert bool(torch.all(torch.isfinite(parameter.grad)))
            assert bool(torch.any(parameter.grad != 0))


def test_traffic_is_complete_and_below_gate():
    traffic = projection_normalized_ternary_traffic(
        576,
        1536,
        layer_count=30,
    )

    assert traffic["ternary_code_payload_bytes_per_layer"] == 530844
    assert traffic["fp16_row_scale_payload_bytes_per_layer"] == 7296
    assert traffic["fp16_rms_gain_payload_bytes_per_layer"] == 5376
    assert traffic["layer_block_bytes"] == 544064
    assert traffic["total_cold_bytes"] == 16323904
    assert traffic["fraction_of_dense_q4"] < 0.411
    assert traffic["passes_45_percent_traffic_gate"]
    assert traffic["headroom_bytes_to_45_percent"] > 0


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("hidden_size", {"hidden_size": 0, "intermediate_size": 8, "layer_count": 2}),
        (
            "intermediate_size",
            {"hidden_size": 4, "intermediate_size": True, "layer_count": 2},
        ),
        ("layer_count", {"hidden_size": 4, "intermediate_size": 8, "layer_count": 0}),
    ],
)
def test_traffic_rejects_invalid_dimensions(field, kwargs):
    with pytest.raises(ValueError, match=field):
        projection_normalized_ternary_traffic(**kwargs)
