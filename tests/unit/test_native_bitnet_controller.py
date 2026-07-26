from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from engram.controller import FactorizedRecurrentController
from engram.evaluation.native_bitnet_controller_generation import (
    _token_agreement,
)
from engram.runtime.native_bitnet_controller import (
    ControllerDrivenBitNet,
    NativeBitNetShellOps,
    NativeOperatorResidual,
)


def test_controller_driven_forward_never_calls_decoder_layer():
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

    class ZeroAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.positions = []

        def forward(self, hidden_states, *, position_ids, **_kwargs):
            self.positions.append(position_ids.clone())
            return torch.zeros_like(hidden_states), None

    class ZeroMLP(nn.Module):
        def forward(self, hidden_states):
            return torch.zeros_like(hidden_states)

    class ForbiddenDecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.Identity()
            self.self_attn = ZeroAttention()
            self.post_attention_layernorm = nn.Identity()
            self.mlp = ZeroMLP()

        def forward(self, *_args, **_kwargs):
            raise AssertionError("decoder layer forward must not run")

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(3, 2)
            self.layers = nn.ModuleList(
                [ForbiddenDecoderLayer(), ForbiddenDecoderLayer()]
            )
            self.norm = nn.Identity()

        def rotary_emb(self, hidden_states, *, position_ids):
            shape = (hidden_states.shape[0], hidden_states.shape[1], 1)
            return (
                torch.ones(shape, dtype=hidden_states.dtype),
                torch.zeros(shape, dtype=hidden_states.dtype),
            )

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=2)
            self.model = FakeBackbone()
            self.lm_head = nn.Identity()

    model = FakeModel()
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(
            torch.tensor([[3.0, 4.0], [5.0, 12.0], [8.0, 15.0]])
        )
    controller = FactorizedRecurrentController.initialize(
        input_dim=6,
        state_dim=2,
        num_stages=2,
        rank=1,
        adapter_rank=0,
        operator_residual=True,
        seed=53,
    )
    tensors = controller.tensors()
    tensors["step_scale"][:] = 0.0
    controller = FactorizedRecurrentController(**tensors)
    runner = ControllerDrivenBitNet(model, controller)
    token_ids = torch.tensor([[0, 1]])
    position_ids = torch.tensor([[7, 8]])

    result = runner.forward(token_ids, position_ids=position_ids)

    embedded = model.model.embed_tokens(token_ids).detach().numpy()
    expected_rms = np.sqrt(np.mean(np.square(embedded), axis=-1, keepdims=True))
    expected = embedded / expected_rms
    np.testing.assert_allclose(
        result.normalized_state,
        expected,
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(result.residual_rms, expected_rms, rtol=2e-3)
    np.testing.assert_allclose(result.logits.numpy(), expected, atol=0.01)
    assert runner.decoder_layer_forward_calls == 0
    for layer in model.model.layers:
        assert torch.equal(layer.self_attn.positions[0], position_ids)


def test_controller_driven_forward_validates_position_shape():
    torch = pytest.importorskip("torch")
    controller = FactorizedRecurrentController.initialize(
        input_dim=3,
        state_dim=1,
        num_stages=1,
        rank=1,
        adapter_rank=0,
        operator_residual=True,
    )
    tensors = controller.tensors()
    tensors["step_scale"][:] = 0.0
    controller = FactorizedRecurrentController(**tensors)
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=1),
        model=SimpleNamespace(layers=[object()]),
    )
    runner = ControllerDrivenBitNet(model, controller)

    with pytest.raises(ValueError, match="shapes must match"):
        runner.forward(
            torch.tensor([[1, 2]]),
            position_ids=torch.tensor([[0]]),
        )


def test_controller_generation_token_agreement_counts_length_mismatch():
    assert _token_agreement((1, 2, 3), (1, 4, 3)) == pytest.approx(2 / 3)
    assert _token_agreement((1, 2), (1, 2, 3)) == pytest.approx(2 / 3)
    assert _token_agreement((), ()) == 1.0


def test_native_operator_residual_matches_numpy_controller():
    library = Path("build/libengram_bitnet.so")
    if not library.is_file():
        pytest.skip("native BitNet library has not been built")
    rng = np.random.default_rng(59)
    state = rng.normal(size=(2, 3, 4)).astype(np.float32)
    semantic = rng.normal(scale=0.2, size=state.shape).astype(np.float32)
    episodic = rng.normal(scale=0.2, size=state.shape).astype(np.float32)
    native_state, native_rms = NativeOperatorResidual(library).step(
        state,
        semantic,
        episodic,
    )
    residual = state + semantic + episodic
    expected_rms = np.sqrt(
        np.mean(np.square(residual), axis=-1, keepdims=True)
    ).clip(1e-6)
    expected_state = residual / np.sqrt(
        np.mean(np.square(residual), axis=-1, keepdims=True) + 1e-6
    )

    np.testing.assert_allclose(native_state, expected_state, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(native_rms, expected_rms, rtol=2e-6, atol=2e-6)


def test_native_shell_embedding_and_rms_norm_match_torch():
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    library = Path("build/libengram_bitnet.so")
    if not library.is_file():
        pytest.skip("native BitNet library has not been built")
    native = NativeBitNetShellOps(library)
    rng = np.random.default_rng(61)
    embedding = nn.Embedding(17, 12, dtype=torch.bfloat16)
    with torch.no_grad():
        embedding.weight.copy_(
            torch.from_numpy(rng.normal(size=(17, 12)).astype(np.float32))
        )
    token_ids = torch.tensor([[2, 11, 4], [16, 0, 7]], dtype=torch.long)
    native_embedding = native.embedding(embedding.weight, token_ids)
    assert torch.equal(native_embedding, embedding(token_ids))

    norm = nn.RMSNorm(12, eps=1e-6, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(
            torch.from_numpy(rng.normal(size=(12,)).astype(np.float32))
        )
    values = rng.normal(size=(2, 3, 12)).astype(np.float32)
    rounded = torch.from_numpy(values).to(torch.bfloat16)
    variance = rounded.float().pow(2).mean(-1, keepdim=True)
    expected = norm.weight * (
        rounded.float() * torch.rsqrt(variance + norm.eps)
    ).to(torch.bfloat16)
    actual = native.rms_norm(values, norm.weight, norm.eps)
    # Reduction order may differ by one BF16 unit from Torch's vectorized mean.
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.01, atol=0.01)

    hidden = torch.from_numpy(rng.normal(size=(12,)).astype(np.float32)).to(
        torch.bfloat16
    )
    vocabulary = torch.from_numpy(
        rng.normal(size=(31, 12)).astype(np.float32)
    ).to(torch.bfloat16)
    token, score = native.vocab_argmax(hidden, vocabulary, threads=3)
    expected_logits = (hidden.float() @ vocabulary.float().T).to(torch.bfloat16)
    assert token == int(expected_logits.argmax())
    assert score == pytest.approx(float(expected_logits[token]), rel=2e-6)

    query = torch.from_numpy(
        rng.normal(size=(2, 3, 4, 12)).astype(np.float32)
    ).to(torch.bfloat16)
    key = torch.from_numpy(
        rng.normal(size=(2, 1, 4, 12)).astype(np.float32)
    ).to(torch.bfloat16)
    positions = torch.tensor([[0, 1, 7, 31], [2, 3, 11, 47]])
    half = query.shape[-1] // 2
    frequencies = 1.0 / (
        500000.0
        ** (torch.arange(0, query.shape[-1], 2).float() / query.shape[-1])
    )
    angles = positions.float().unsqueeze(-1) * frequencies
    cosine = torch.cat((angles, angles), dim=-1).cos().to(torch.bfloat16)
    sine = torch.cat((angles, angles), dim=-1).sin().to(torch.bfloat16)

    def expected_rope(values):
        rotated = torch.cat((-values[..., half:], values[..., :half]), dim=-1)
        return values * cosine.unsqueeze(1) + rotated * sine.unsqueeze(1)

    expected_query = expected_rope(query)
    expected_key = expected_rope(key)
    actual_query, actual_key = native.rope(
        query.clone(), key.clone(), positions, theta=500000.0
    )
    torch.testing.assert_close(
        actual_query.float(), expected_query.float(), rtol=0.02, atol=0.02
    )
    torch.testing.assert_close(
        actual_key.float(), expected_key.float(), rtol=0.02, atol=0.02
    )
