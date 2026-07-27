import numpy as np

from engram.models.fixture import create_tiny_olmoe_fixture
from engram.semantic.olmoe import olmoe_sparse_mlp
from engram.tracing.format import TraceReader
from engram.tracing.olmoe import capture_olmoe_fixture_router_traces
from engram.tracing.olmoe import (
    _prepare_transformers_imports,
    capture_olmoe_router_batch,
)


def test_exact_olmoe_decomposition_sums_weighted_contributions():
    rng = np.random.default_rng(9)
    hidden = rng.normal(size=(5, 6)).astype(np.float32)
    router = rng.normal(size=(4, 6)).astype(np.float32)
    gate = rng.normal(size=(4, 3, 6)).astype(np.float32)
    up = rng.normal(size=(4, 3, 6)).astype(np.float32)
    down = rng.normal(size=(4, 6, 3)).astype(np.float32)

    result = olmoe_sparse_mlp(hidden, router, gate, up, down, top_k=2)

    np.testing.assert_allclose(
        result.output,
        result.expert_contributions.sum(axis=1),
        rtol=1e-6,
        atol=1e-6,
    )
    expected = np.argsort(
        -result.router_probabilities, axis=1, kind="stable"
    )[:, :2]
    np.testing.assert_array_equal(result.expert_indices, expected)
    assert np.all(result.expert_weights.sum(axis=1) < 1.0)


def test_olmoe_fixture_trace_preserves_router_and_expert_fields(tmp_path):
    model = create_tiny_olmoe_fixture(tmp_path / "model")
    out = tmp_path / "trace"

    capture_olmoe_fixture_router_traces(
        model, out, samples=7, layers=[1], seed=31
    )

    reader = TraceReader(out)
    assert reader.manifest["metadata"]["model_family"] == "olmoe"
    shard = next(reader.iter_shards())
    assert shard["layer_1_router_probabilities"].shape == (7, 4)
    assert shard["layer_1_expert_indices"].shape == (7, 2)
    assert shard["layer_1_expert_contributions"].shape == (7, 2, 16)
    np.testing.assert_allclose(
        shard["layer_1_mlp_output"],
        shard["layer_1_expert_contributions"].sum(axis=1),
        rtol=1e-6,
        atol=1e-6,
    )


def test_hf_olmoe_hook_captures_unmodified_router_contract():
    torch = __import__("pytest").importorskip("torch")
    transformers = __import__("pytest").importorskip("transformers")
    _prepare_transformers_imports()
    config = transformers.OlmoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
    )
    model = transformers.OlmoeForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 4, 7, 3]], dtype=torch.long)

    captured = capture_olmoe_router_batch(model, input_ids, layers=[0, 1])

    for layer in (0, 1):
        prefix = f"layer_{layer}"
        assert captured[f"{prefix}_mlp_input"].shape == (1, 4, 16)
        assert captured[f"{prefix}_router_probabilities"].shape == (4, 4)
        assert captured[f"{prefix}_expert_indices"].shape == (4, 2)
        assert captured[f"{prefix}_expert_weights"].shape == (4, 2)
        assert captured[f"{prefix}_mlp_output"].shape == (1, 4, 16)
        selected = np.take_along_axis(
            captured[f"{prefix}_router_probabilities"],
            captured[f"{prefix}_expert_indices"],
            axis=1,
        )
        np.testing.assert_allclose(
            captured[f"{prefix}_expert_weights"], selected
        )
        probabilities = captured[f"{prefix}_router_probabilities"].copy()
        np.put_along_axis(
            probabilities,
            captured[f"{prefix}_expert_indices"],
            -np.inf,
            axis=1,
        )
        assert np.all(selected.min(axis=1) >= probabilities.max(axis=1))
