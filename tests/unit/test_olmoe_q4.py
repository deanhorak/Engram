import numpy as np
import pytest

from engram.evaluation.olmoe_q4 import (
    round_to_bfloat16,
    symmetric_groupwise_dequant,
    symmetric_groupwise_q4_dequant,
)
from engram.evaluation.olmoe_q4_causal import _quantize_olmoe_experts_in_place
from engram.tracing.olmoe import _prepare_transformers_imports


def test_groupwise_q4_is_deterministic_bounded_and_accounts_scales():
    rng = np.random.default_rng(12)
    values = rng.normal(size=(5, 70)).astype(np.float32)

    first, first_bytes = symmetric_groupwise_q4_dequant(values, group_size=64)
    second, second_bytes = symmetric_groupwise_q4_dequant(values, group_size=64)

    np.testing.assert_array_equal(first, second)
    assert first_bytes == second_bytes == (5 * 70 + 1) // 2 + 5 * 2 * 2
    assert np.isfinite(first).all()
    assert np.linalg.norm(first - values) / np.linalg.norm(values) < 0.15
    # The reference path must execute the same rounded scales charged in storage.
    maximum = np.max(np.abs(values[:, :64]), axis=1) / 7.0
    assert np.all(round_to_bfloat16(maximum) != maximum)


def test_groupwise_q4_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="finite matrix"):
        symmetric_groupwise_q4_dequant([[np.nan]])
    with pytest.raises(ValueError, match="group_size"):
        symmetric_groupwise_q4_dequant([[1.0]], group_size=0)


def test_six_bit_groupwise_reduces_error_and_accounts_packed_codes():
    rng = np.random.default_rng(22)
    values = rng.normal(size=(7, 35)).astype(np.float32)
    q4, _ = symmetric_groupwise_dequant(
        values, bits=4, group_size=32
    )
    q6, q6_bytes = symmetric_groupwise_dequant(
        values, bits=6, group_size=32
    )

    assert np.linalg.norm(q6 - values) < np.linalg.norm(q4 - values)
    assert q6_bytes == (7 * 35 * 6 + 7) // 8 + 7 * 2 * 2
    with pytest.raises(ValueError, match="bits"):
        symmetric_groupwise_dequant(values, bits=9, group_size=32)


def test_bfloat16_scale_rounding_preserves_tiny_nonzero_values():
    values = np.array([1e-20, 1.0, 1.00390625], dtype=np.float32)
    rounded = round_to_bfloat16(values)

    assert rounded[0] > 0.0
    assert rounded.dtype == np.float32
    np.testing.assert_allclose(rounded[:2], values[:2], rtol=4e-3)


def test_hf_olmoe_experts_are_quantized_in_place():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    _prepare_transformers_imports()
    config = transformers.OlmoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
    )
    model = transformers.OlmoeForCausalLM(config).eval()
    before = model.model.layers[0].mlp.experts.gate_up_proj.detach().clone()

    report = _quantize_olmoe_experts_in_place(model, bits=4, group_size=8)

    after = model.model.layers[0].mlp.experts.gate_up_proj.detach()
    assert report["expert_matrices"] == 8
    assert report["expert_parameters"] > 0
    assert torch.isfinite(after).all()
    assert not torch.equal(before, after)
