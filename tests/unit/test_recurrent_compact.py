import pytest

from engram.training.recurrent_compact import (
    recurrent_compact_mlp_class,
    recurrent_compact_q4_traffic,
)


def _fixture(torch):
    generator = torch.Generator().manual_seed(1701)
    gate = torch.randn(7, 5, generator=generator) * 0.2
    up = torch.randn(7, 5, generator=generator) * 0.2
    down = torch.randn(5, 7, generator=generator) * 0.2
    hidden = torch.randn(6, 5, generator=generator)
    return gate, up, down, hidden


def test_zero_initialized_recurrence_preserves_single_compact_forward():
    torch = pytest.importorskip("torch")
    gate, up, down, hidden = _fixture(torch)
    Module = recurrent_compact_mlp_class(torch)
    module = Module(gate, up, down, cycles=3)

    expected = module.compact(hidden)
    actual, cycle_outputs = module(hidden, return_cycle_outputs=True)

    torch.testing.assert_close(actual, expected)
    assert len(cycle_outputs) == 3
    torch.testing.assert_close(cycle_outputs[0], expected)


def test_recurrent_forward_matches_explicit_refinement_and_trains_adapters():
    torch = pytest.importorskip("torch")
    gate, up, down, hidden = _fixture(torch)
    Module = recurrent_compact_mlp_class(torch)
    module = Module(gate, up, down, cycles=2, adapter_rank=2)
    with torch.no_grad():
        module.input_delta.fill_(0.2)
        module.feedback_gain.fill_(0.15)
        module.output_gain.fill_(0.1)
        module.input_adapter_up.fill_(0.03)

    base = module.compact(hidden)
    value_rms = torch.sqrt(
        torch.mean(base.float().square(), dim=-1, keepdim=True) + 1e-6
    )
    hidden_rms = torch.sqrt(
        torch.mean(hidden.float().square(), dim=-1, keepdim=True) + 1e-6
    )
    matched = base * (hidden_rms / value_rms)
    recurrent_input = hidden * (1.0 + 0.25 * torch.tanh(module.input_delta[0]))
    recurrent_input = recurrent_input + torch.tanh(module.feedback_gain[0]) * matched
    compressed = torch.nn.functional.linear(hidden, module.input_adapter_down[0])
    recurrent_input = (
        recurrent_input
        + torch.nn.functional.linear(compressed, module.input_adapter_up[0]) / 2
    )
    expected = base + module.output_gain[0] * module.compact(recurrent_input)

    actual = module(hidden)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    for parameter in (
        module.input_delta,
        module.feedback_gain,
        module.output_gain,
        module.input_adapter_down,
        module.input_adapter_up,
    ):
        assert parameter.grad is not None
        assert bool(torch.any(parameter.grad != 0))


def test_recurrent_traffic_charges_adapters_and_bounds_supported_cycles():
    two = recurrent_compact_q4_traffic(576, 1536, [672] * 30, cycles=2)
    four = recurrent_compact_q4_traffic(576, 1536, [672] * 30, cycles=4)
    five = recurrent_compact_q4_traffic(576, 1536, [672] * 30, cycles=5)
    relaxed = recurrent_compact_q4_traffic(
        576,
        1536,
        [640] * 30,
        cycles=4,
        adapter_rank=4,
    )

    assert two["adapter_fp16_payload_bytes_per_layer"] == 3 * 576 * 2
    assert two["base_compact_q4_bytes"] < two["total_cold_bytes"]
    assert two["fraction_of_dense_q4"] < four["fraction_of_dense_q4"]
    assert four["passes_45_percent_traffic_gate"]
    assert not five["passes_45_percent_traffic_gate"]
    assert relaxed["low_rank_adapter_values_per_layer"] == 2 * 3 * 576 * 4
    assert relaxed["passes_45_percent_traffic_gate"]
    assert relaxed["fraction_of_dense_q4"] < 0.45
    assert four["requires_native_cache_validation"]
    assert not four["native_cache_reuse_measured"]


@pytest.mark.parametrize("cycles", [0, 1, True])
def test_recurrent_traffic_rejects_non_recurrent_cycle_counts(cycles):
    with pytest.raises(ValueError, match="integer|at least two"):
        recurrent_compact_q4_traffic(8, 16, [4], cycles=cycles)
