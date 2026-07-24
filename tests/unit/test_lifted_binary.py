import pytest

from engram.training.lifted_binary import (
    lifted_binary_mlp_class,
    lifted_binary_traffic,
)


torch = pytest.importorskip("torch")


def _state(output: int, input_: int, *, projected: int = 2, lifted: int = 3):
    groups = (output * input_ + projected - 1) // projected
    bits = torch.ones(groups, lifted)
    bits[:, 1::2] = -1
    return {
        "bits": bits,
        "projection": torch.arange(
            1, projected * lifted + 1, dtype=torch.float32
        ).reshape(projected, lifted)
        / 10,
        "input_scale": torch.ones(input_),
        "output_scale": torch.ones(output),
    }


def test_hard_forward_is_invariant_to_proxy_magnitude():
    Module = lifted_binary_mlp_class(torch)
    state = _state(4, 4)
    module = Module(
        torch.zeros(4, 4),
        torch.zeros(4, 4),
        torch.zeros(4, 4),
        gate_state=state,
        up_state=state,
        down_state=state,
        block_size=2,
    )
    value = torch.randn(3, 4)
    first = module(value)
    with torch.no_grad():
        for projection in (module.gate, module.up, module.down):
            projection.proxy.mul_(100)
    second = module(value)
    torch.testing.assert_close(first, second)


def test_proxy_and_projection_receive_gradients():
    Module = lifted_binary_mlp_class(torch)
    state = _state(4, 4)
    module = Module(
        torch.randn(4, 4),
        torch.randn(4, 4),
        torch.randn(4, 4),
        gate_state=state,
        up_state=state,
        down_state=state,
        block_size=2,
    )
    module(torch.randn(3, 4)).square().mean().backward()
    assert module.gate.proxy.grad is not None
    assert torch.count_nonzero(module.gate.proxy.grad) > 0
    assert module.gate.projection.grad is not None
    deployment = module.deployment_state()
    assert deployment["format"] == "engram_lifted_binary_projection_v1"
    assert set(deployment["gate"]["bits"].unique().tolist()) <= {-1, 1}


def test_mixed_16_to_8_and_16_to_10_layout_fits_gate():
    traffic = lifted_binary_traffic(
        576,
        1536,
        projection_dimensions=((16, 8), (16, 10), (16, 10)),
        layer_count=30,
    )
    assert traffic["effective_bits_per_weight"] == [2.0, 1.6, 1.6]
    assert traffic["code_stream_bytes_per_layer"] == [
        221_184,
        176_948,
        176_948,
    ]
    assert traffic["binary_code_payload_bytes_per_layer"] == 575_080
    assert traffic["fp16_projection_payload_bytes_per_layer"] == 896
    assert traffic["layer_block_bytes"] == 589_184
    assert traffic["total_cold_bytes"] == 17_677_504
    assert traffic["fraction_of_dense_q4"] == pytest.approx(
        0.44401202417695473
    )
    assert traffic["headroom_bytes_to_45_percent"] == 238_400
    assert traffic["passes_45_percent_traffic_gate"] is True


@pytest.mark.parametrize(
    "dimensions",
    [
        ((16, 8), (16, 10)),
        ((8, 8), (16, 10), (16, 10)),
    ],
)
def test_traffic_rejects_invalid_dimensions(dimensions):
    with pytest.raises(ValueError):
        lifted_binary_traffic(
            576,
            1536,
            projection_dimensions=dimensions,
            layer_count=30,
        )


def test_module_rejects_nonbinary_initial_values():
    Module = lifted_binary_mlp_class(torch)
    state = _state(4, 4)
    state["bits"][0, 0] = 0
    with pytest.raises(ValueError, match="must be -1 or"):
        Module(
            torch.zeros(4, 4),
            torch.zeros(4, 4),
            torch.zeros(4, 4),
            gate_state=state,
            up_state=_state(4, 4),
            down_state=_state(4, 4),
            block_size=2,
        )
