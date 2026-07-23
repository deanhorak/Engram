import numpy as np
import pytest
import torch

from engram.training.width_pruning import (
    _fake_signed_q4_rows,
    _resolve_target_widths,
    _wrap_width_pruned_mlp_class,
    select_width_channels,
    width_pruned_schedule_traffic,
    width_pruned_traffic,
)
from engram.training.switch_expert_boundaries import (
    _decode_symmetric_q4_rows,
    _pack_symmetric_q4_rows,
)


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 8, bias=False)
        self.up_proj = torch.nn.Linear(4, 8, bias=False)
        self.down_proj = torch.nn.Linear(8, 4, bias=False)
        self.act_fn = torch.nn.functional.silu

    def forward(self, hidden):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden)
        )


def test_width_pruned_traffic_uses_contiguous_target_width():
    traffic = width_pruned_traffic(576, 1536, 672)
    assert traffic.weight_bytes == 4_644_864
    assert traffic.dense_weight_bytes == 10_616_832
    assert traffic.fraction_of_dense == 0.4375


def test_width_pruned_schedule_traffic_accounts_each_layer_width():
    traffic = width_pruned_schedule_traffic(8, 12, [3, 5])

    assert traffic["layer_widths"] == [3, 5]
    assert traffic["weight_bytes"] == 768
    assert traffic["dense_weight_bytes"] == 2304
    assert traffic["fraction_of_dense"] == 1 / 3
    assert [layer["weight_bytes"] for layer in traffic["layers"]] == [288, 480]


@pytest.mark.parametrize("widths", ([], [3, 13], [3, True]))
def test_width_pruned_schedule_traffic_rejects_invalid_widths(widths):
    with pytest.raises(ValueError):
        width_pruned_schedule_traffic(8, 12, widths)


def test_target_width_schedule_requires_one_width_per_layer():
    with pytest.raises(ValueError, match="one width per transformer layer"):
        _resolve_target_widths(
            4,
            [3],
            layer_count=2,
            source_intermediate_size=12,
        )


def test_fake_signed_q4_rows_matches_deployment_pack_decode_and_uses_ste():
    weight = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.25, -0.7, -0.1, 0.45, 1.6],
            [7.0, -6.0, 3.25, -2.0, 0.125],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    actual = _fake_signed_q4_rows(weight, torch)
    packed = _pack_symmetric_q4_rows(weight.detach().numpy())
    expected = torch.from_numpy(_decode_symmetric_q4_rows(packed))

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    actual.sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight))


def test_fake_q4_wrapper_quantizes_down_in_codec_orientation():
    torch.manual_seed(49)
    base = _MLP()
    wrapper_type = _wrap_width_pruned_mlp_class(torch)
    wrapper = wrapper_type(
        base,
        np.asarray([1, 3, 4, 7]),
        fake_q4_training=True,
    )
    wrapper.mode = "compact"
    hidden = torch.randn(2, 3, 4)

    output = wrapper(hidden)
    gate = torch.from_numpy(
        _decode_symmetric_q4_rows(
            _pack_symmetric_q4_rows(wrapper.gate_weight.detach().numpy())
        )
    )
    up = torch.from_numpy(
        _decode_symmetric_q4_rows(
            _pack_symmetric_q4_rows(wrapper.up_weight.detach().numpy())
        )
    )
    down = torch.from_numpy(
        _decode_symmetric_q4_rows(
            _pack_symmetric_q4_rows(wrapper.down_weight.detach().T.numpy())
        ).T
    )
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(hidden, gate))
        * torch.nn.functional.linear(hidden, up),
        down,
    )

    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)


def test_width_channel_selection_is_deterministic_and_unique():
    rng = np.random.default_rng(43)
    gate = rng.normal(size=(8, 4))
    up = rng.normal(size=(8, 4))
    down = rng.normal(size=(4, 8))
    states = rng.normal(size=(12, 4))
    first = select_width_channels(gate, up, down, 5, states=states)
    second = select_width_channels(gate, up, down, 5, states=states)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 5


def test_width_wrapper_preserves_dense_path_and_executes_compact_path():
    torch.manual_seed(47)
    base = _MLP()
    wrapper_type = _wrap_width_pruned_mlp_class(torch)
    wrapper = wrapper_type(base, np.asarray([1, 3, 4, 7]))
    hidden = torch.randn(2, 3, 4)

    wrapper.mode = "dense"
    torch.testing.assert_close(wrapper(hidden), base(hidden))

    wrapper.mode = "compact"
    output = wrapper(hidden)
    gate = torch.nn.functional.linear(hidden, wrapper.gate_weight)
    up = torch.nn.functional.linear(hidden, wrapper.up_weight)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(gate) * up, wrapper.down_weight
    )
    torch.testing.assert_close(output, expected)
    assert wrapper.target_width == 4
    assert wrapper.last_output is output
