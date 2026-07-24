import pytest

from engram.training.codebook_vq import (
    unrestricted_codebook_vq_mlp_class,
    unrestricted_codebook_vq_traffic,
)


torch = pytest.importorskip("torch")


def _state(output: int, input_: int, *, vector: int = 2):
    groups = input_ // vector
    candidates = torch.zeros(output, groups, 2, dtype=torch.long)
    candidates[..., 1] = 1
    logits = torch.zeros(output, groups, 2)
    logits[..., 0] = 2.0
    return {
        "codebook": torch.tensor(
            [[0.25, -0.5], [1.0, 0.75]], dtype=torch.float32
        ),
        "candidates": candidates,
        "logits": logits,
    }


def test_hard_forward_is_invariant_to_soft_logits_with_same_winner():
    Module = unrestricted_codebook_vq_mlp_class(torch)
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
            projection.assignment_logits[..., 0] = 20.0
            projection.assignment_logits[..., 1] = 19.0
    second = module(value)
    torch.testing.assert_close(first, second)


def test_assignment_logits_receive_straight_through_gradients():
    Module = unrestricted_codebook_vq_mlp_class(torch)
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
    assert module.gate.assignment_logits.grad is not None
    assert torch.count_nonzero(module.gate.assignment_logits.grad) > 0
    assert module.gate.codebook.grad is not None
    deployment = module.deployment_state()
    assert deployment["format"] == "engram_unrestricted_vector_codebook_v1"
    assert deployment["gate"]["codes"].dtype == torch.uint8


def test_strict_7_bit_four_weight_layout_fits_gate():
    traffic = unrestricted_codebook_vq_traffic(
        576,
        1536,
        layer_count=30,
        vector_size=4,
        codebook_size=128,
    )
    assert traffic["index_bits"] == 7
    assert traffic["code_stream_bytes_per_layer"] == [
        193_536,
        193_536,
        193_536,
    ]
    assert traffic["fp16_codebook_payload_bytes_per_layer"] == 3_072
    assert traffic["layer_block_bytes"] == 596_864
    assert traffic["total_cold_bytes"] == 17_907_904
    assert traffic["fraction_of_dense_q4"] == pytest.approx(
        0.44979906121399177
    )
    assert traffic["headroom_bytes_to_45_percent"] == 8_000
    assert traffic["passes_45_percent_traffic_gate"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vector_size", 0),
        ("codebook_size", 257),
    ],
)
def test_traffic_rejects_invalid_layout(field, value):
    kwargs = {
        "hidden_size": 576,
        "intermediate_size": 1536,
        "layer_count": 30,
        "vector_size": 4,
        "codebook_size": 128,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        unrestricted_codebook_vq_traffic(**kwargs)


def test_module_rejects_out_of_range_candidate():
    Module = unrestricted_codebook_vq_mlp_class(torch)
    state = _state(4, 4)
    state["candidates"][0, 0, 0] = 2
    with pytest.raises(ValueError, match="outside"):
        Module(
            torch.zeros(4, 4),
            torch.zeros(4, 4),
            torch.zeros(4, 4),
            gate_state=state,
            up_state=_state(4, 4),
            down_state=_state(4, 4),
            block_size=2,
        )
