import pytest

from engram.training.linear_constrained_vq import (
    block_hadamard_function,
    linear_constrained_vq_mlp_class,
    linear_constrained_vq_traffic,
)


def _state(torch, shape, levels, seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        "symbols": torch.randint(
            levels, shape, generator=generator, dtype=torch.int8
        ),
        "matrix": torch.randn(4, 4, generator=generator) * 0.04,
        "bias": torch.randn(4, generator=generator) * 0.01,
        "levels": levels,
    }


def _fixture(torch):
    generator = torch.Generator().manual_seed(2204)
    gate = torch.randn(12, 8, generator=generator) * 0.2
    up = torch.randn(12, 8, generator=generator) * 0.2
    down = torch.randn(8, 12, generator=generator) * 0.2
    hidden = torch.randn(6, 8, generator=generator)
    return gate, up, down, hidden


def test_block_hadamard_is_normalized_involution():
    torch = pytest.importorskip("torch")
    transform = block_hadamard_function(torch)
    generator = torch.Generator().manual_seed(18)
    value = torch.randn(7, 12, generator=generator)

    transformed = transform(value, 4)
    restored = transform(transformed, 4)

    torch.testing.assert_close(restored, value)
    torch.testing.assert_close(
        transformed.square().sum(), value.square().sum()
    )


def test_hard_forward_has_expected_shape_and_symbol_alphabets():
    torch = pytest.importorskip("torch")
    gate, up, down, hidden = _fixture(torch)
    Module = linear_constrained_vq_mlp_class(torch)
    module = Module(
        gate,
        up,
        down,
        gate_state=_state(torch, gate.shape, 3, 1),
        up_state=_state(torch, up.shape, 3, 2),
        down_state=_state(torch, down.shape, 4, 3),
        block_size=4,
    )

    output = module(hidden)
    state = module.deployment_state()

    assert output.shape == hidden.shape
    assert state["format"] == "engram_linear_constrained_vector_qat_v1"
    assert state["block_size"] == 4
    for name, levels in (("gate", 3), ("up", 3), ("down", 4)):
        symbols = state[name]["symbols"]
        assert int(symbols.min()) >= 0
        assert int(symbols.max()) < levels
        assert state[name]["matrix"].dtype == torch.float16
        assert state[name]["input_scale"].dtype == torch.float16


def test_dge_trains_proxy_affine_map_and_side_scales():
    torch = pytest.importorskip("torch")
    gate, up, down, hidden = _fixture(torch)
    Module = linear_constrained_vq_mlp_class(torch)
    module = Module(
        gate,
        up,
        down,
        gate_state=_state(torch, gate.shape, 3, 4),
        up_state=_state(torch, up.shape, 3, 5),
        down_state=_state(torch, down.shape, 4, 6),
        block_size=4,
    )

    module(hidden).square().mean().backward()

    for projection in (module.gate, module.up, module.down):
        for parameter in (
            projection.proxy,
            projection.matrix,
            projection.bias,
            projection.input_scale,
            projection.output_scale,
        ):
            assert parameter.grad is not None
            assert bool(torch.all(torch.isfinite(parameter.grad)))
            assert bool(torch.any(parameter.grad != 0))


@pytest.mark.parametrize(
    "levels", [(3, 3, 4), (3, 4, 3), (4, 3, 3)]
)
def test_mixed_symbol_traffic_passes_gate(levels):
    traffic = linear_constrained_vq_traffic(
        576,
        1536,
        projection_levels=levels,
        layer_count=30,
    )

    assert traffic["symbol_payload_bytes_per_layer"] == 575080
    assert traffic["fp16_side_scale_payload_bytes_per_layer"] == 12672
    assert traffic["fp16_affine_payload_bytes_per_layer"] == 120
    assert traffic["layer_block_bytes"] == 588480
    assert traffic["total_cold_bytes"] == 17656384
    assert traffic["fraction_of_dense_q4"] < 0.445
    assert traffic["passes_45_percent_traffic_gate"]
    assert traffic["headroom_bytes_to_45_percent"] > 0


def test_traffic_rejects_unsupported_alphabet():
    with pytest.raises(ValueError, match="projection_levels"):
        linear_constrained_vq_traffic(
            8,
            12,
            projection_levels=(3, 2, 4),
            layer_count=1,
        )
