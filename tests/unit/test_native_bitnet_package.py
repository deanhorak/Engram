from pathlib import Path
from types import SimpleNamespace

import pytest

from engram.compiler.native_bitnet import (
    _inventory,
    _write_non_mlp_weights,
)
from engram.evaluation.native_bitnet_attention import (
    _attention_replacement_class,
)
from engram.evaluation.native_bitnet_generation_benchmark import (
    _fixed_length_context,
)
from engram.evaluation.native_bitnet_parity import (
    _disable_broken_optional_transformers_dependencies,
)
from engram.runtime.native_bitnet import (
    NativeBitNetRuntime,
    _materialized_bitlinear_class,
    _safe_package_path,
)
from engram.runtime.native_bitnet_attention import (
    native_incremental_attention_class,
)


def test_package_paths_are_confined_to_the_package(tmp_path):
    assert _safe_package_path(tmp_path, "mlp/model.bin") == (tmp_path / "mlp/model.bin")
    for unsafe in ("", "../model.bin", "/tmp/model.bin"):
        with pytest.raises(ValueError, match="unsafe"):
            _safe_package_path(tmp_path, unsafe)


def test_long_context_benchmark_repeats_prompt_to_exact_requested_length():
    assert _fixed_length_context([1, 2, 3], 1) == [1]
    assert _fixed_length_context([1, 2, 3], 6) == [1, 2, 3, 2, 3, 2]


def test_non_mlp_package_excludes_source_mlp_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "transformer" / "non_mlp.safetensors"
    safetensors.save_file(
        {
            "model.embed_tokens.weight": torch.arange(12).reshape(3, 4),
            "model.layers.0.self_attn.q_proj.weight": torch.ones(1, 4),
            "model.layers.0.mlp.gate_proj.weight": torch.zeros(1, 4),
        },
        source,
    )

    report = _write_non_mlp_weights(source, destination)
    loaded = safetensors.load_file(destination)

    assert report["source_tensors"] == 3
    assert report["packaged_tensors"] == 2
    assert report["excluded_mlp_tensors"] == 1
    assert "model.embed_tokens.weight" in loaded
    assert not any(".mlp." in name for name in loaded)
    inventory = _inventory(tmp_path)
    assert inventory["transformer/non_mlp.safetensors"]["bytes"] == (
        destination.stat().st_size
    )


def test_materialized_attention_linear_matches_bitnet_activation_quant():
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torch.nn.functional")
    Linear = _materialized_bitlinear_class()
    codes = torch.tensor([[1, 0, -1], [-1, 1, 0]], dtype=torch.int8)
    scale = torch.tensor([0.25], dtype=torch.bfloat16)
    values = torch.tensor(
        [[[0.2, -0.4, 0.7], [1.0, -0.1, 0.3]]],
        dtype=torch.bfloat16,
    )
    activation = values.float()
    quant_scale = 127 / activation.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    quantized = ((activation * quant_scale).round().clamp(-128, 127) / quant_scale).to(
        torch.bfloat16
    )
    expected = functional.linear(quantized, codes.to(torch.bfloat16)) * scale

    assert torch.equal(Linear(codes, scale)(values), expected)


def test_joint_local_retrieval_is_exact_when_topk_covers_all_older_keys():
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    _disable_broken_optional_transformers_dependencies()
    Replacement = _attention_replacement_class()
    source = SimpleNamespace(
        config=SimpleNamespace(),
        layer_idx=0,
        head_dim=2,
        num_key_value_groups=1,
        scaling=2**-0.5,
        attention_dropout=0.0,
        q_proj=nn.Identity(),
        k_proj=nn.Identity(),
        v_proj=nn.Identity(),
        o_proj=nn.Identity(),
        attn_sub_norm=nn.Identity(),
    )
    replacement = Replacement(
        source,
        mode="hybrid",
        local_window=2,
        recurrent_decay=0.99,
        retrieval_top_k=8,
        older_weight=0.5,
    )
    generator = torch.Generator().manual_seed(20260724)
    query = torch.randn(1, 1, 5, 2, generator=generator, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 5, 2, generator=generator, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 5, 2, generator=generator, dtype=torch.bfloat16)
    actual = replacement._local_retrieval(query, key, value)
    expected_rows = []
    for position in range(5):
        scores = (
            torch.einsum(
                "bhd,bhtd->bht",
                query[:, :, position],
                key[:, :, : position + 1],
            )
            * replacement.scaling
        )
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        expected_rows.append(
            torch.einsum(
                "bht,bhtv->bhv",
                weights,
                value[:, :, : position + 1],
            )
        )
    expected = torch.stack(expected_rows, dim=2)

    assert torch.equal(actual, expected)


def test_indexed_hybrid_recovers_exact_attention_when_postings_cover_history():
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    _disable_broken_optional_transformers_dependencies()
    Replacement = _attention_replacement_class()
    source = SimpleNamespace(
        config=SimpleNamespace(),
        layer_idx=0,
        head_dim=2,
        num_key_value_groups=1,
        scaling=2**-0.5,
        attention_dropout=0.0,
        q_proj=nn.Identity(),
        k_proj=nn.Identity(),
        v_proj=nn.Identity(),
        o_proj=nn.Identity(),
        attn_sub_norm=nn.Identity(),
    )
    replacement = Replacement(
        source,
        mode="indexed_hybrid",
        local_window=2,
        recurrent_decay=0.99,
        retrieval_top_k=8,
        older_weight=0.5,
        retrieval_candidates=8,
        lsh_tables=1,
        lsh_bits=1,
        lsh_radius=1,
    )
    generator = torch.Generator().manual_seed(19)
    query = torch.randn(1, 1, 5, 2, generator=generator, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 5, 2, generator=generator, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 5, 2, generator=generator, dtype=torch.bfloat16)

    actual = replacement._indexed_local_retrieval(query, key, value)
    expected = replacement._local_retrieval(query, key, value)

    # Candidate order can change BF16 reduction order while covering the same keys.
    assert torch.allclose(actual.float(), expected.float(), atol=0.005, rtol=0.005)
    assert (
        replacement.index_stats["oracle_hits"]
        == replacement.index_stats["oracle_slots"]
    )


def test_page_bounds_recover_exact_older_topk_attention():
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    _disable_broken_optional_transformers_dependencies()
    Replacement = _attention_replacement_class()
    source = SimpleNamespace(
        config=SimpleNamespace(),
        layer_idx=0,
        head_dim=4,
        num_key_value_groups=1,
        scaling=0.5,
        attention_dropout=0.0,
        q_proj=nn.Identity(),
        k_proj=nn.Identity(),
        v_proj=nn.Identity(),
        o_proj=nn.Identity(),
        attn_sub_norm=nn.Identity(),
    )
    replacement = Replacement(
        source,
        mode="bounded_hybrid",
        local_window=2,
        recurrent_decay=0.99,
        retrieval_top_k=2,
        older_weight=0.5,
        page_size=2,
    )
    generator = torch.Generator().manual_seed(23)
    query = torch.randn(1, 1, 9, 4, generator=generator, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 9, 4, generator=generator, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 9, 4, generator=generator, dtype=torch.bfloat16)

    actual = replacement._bounded_local_retrieval(query, key, value)
    expected = replacement._local_retrieval(query, key, value)

    assert torch.allclose(actual.float(), expected.float(), atol=0.01, rtol=0.01)
    assert replacement.index_stats["opened_pages_per_head"] > 0


def test_streaming_cache_is_exact_while_capacity_covers_older_context():
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    _disable_broken_optional_transformers_dependencies()
    Replacement = _attention_replacement_class()
    source = SimpleNamespace(
        config=SimpleNamespace(),
        layer_idx=0,
        head_dim=2,
        num_key_value_groups=1,
        scaling=2**-0.5,
        attention_dropout=0.0,
        q_proj=nn.Identity(),
        k_proj=nn.Identity(),
        v_proj=nn.Identity(),
        o_proj=nn.Identity(),
        attn_sub_norm=nn.Identity(),
    )
    replacement = Replacement(
        source,
        mode="streaming_hybrid",
        local_window=2,
        recurrent_decay=0.99,
        retrieval_top_k=8,
        older_weight=0.5,
        sink_tokens=2,
    )
    generator = torch.Generator().manual_seed(29)
    query = torch.randn(1, 1, 6, 2, generator=generator, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 6, 2, generator=generator, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 6, 2, generator=generator, dtype=torch.bfloat16)

    actual = replacement._streaming_local_retrieval(query, key, value)
    expected = replacement._local_retrieval(query, key, value)

    assert torch.allclose(actual.float(), expected.float(), atol=0.005, rtol=0.005)


def test_native_streaming_replacement_matches_python_state_machine():
    library = Path("build/libengram_attention.so")
    if not library.exists():
        pytest.skip("native attention library has not been built")
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    _disable_broken_optional_transformers_dependencies()
    Replacement = _attention_replacement_class()
    source = SimpleNamespace(
        config=SimpleNamespace(),
        layer_idx=0,
        head_dim=4,
        num_key_value_groups=2,
        scaling=0.5,
        attention_dropout=0.0,
        q_proj=nn.Identity(),
        k_proj=nn.Identity(),
        v_proj=nn.Identity(),
        o_proj=nn.Identity(),
        attn_sub_norm=nn.Identity(),
    )
    native = Replacement(
        source,
        mode="native_streaming",
        local_window=4,
        recurrent_decay=0.99,
        retrieval_top_k=3,
        older_weight=0.5,
        retrieval_candidates=5,
        sink_tokens=2,
        native_attention_library=library,
    )
    reference = Replacement(
        source,
        mode="streaming_hybrid",
        local_window=4,
        recurrent_decay=0.99,
        retrieval_top_k=3,
        older_weight=0.5,
        retrieval_candidates=5,
        sink_tokens=2,
    )
    generator = torch.Generator().manual_seed(31)
    query = torch.randn(1, 4, 12, 4, generator=generator)
    key = torch.randn(1, 2, 12, 4, generator=generator)
    value = torch.randn(1, 2, 12, 4, generator=generator)

    actual = native._native_streaming(query, key, value)
    expected = reference._streaming_local_retrieval(
        query,
        key.repeat_interleave(2, dim=1),
        value.repeat_interleave(2, dim=1),
    )

    assert torch.allclose(actual.float(), expected.float(), atol=0.015, rtol=0.015)


def test_incremental_native_attention_matches_chunked_execution_and_tracks_positions():
    library = Path("build/libengram_attention.so")
    if not library.exists():
        pytest.skip("native attention library has not been built")
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    _disable_broken_optional_transformers_dependencies()
    Replacement = native_incremental_attention_class()
    source = SimpleNamespace(
        config=SimpleNamespace(
            num_attention_heads=1,
            num_key_value_heads=1,
        ),
        layer_idx=0,
        head_dim=4,
        num_key_value_groups=1,
        scaling=0.5,
        attention_dropout=0.0,
        q_proj=nn.Identity(),
        k_proj=nn.Identity(),
        v_proj=nn.Identity(),
        o_proj=nn.Identity(),
        attn_sub_norm=nn.Identity(),
    )
    full = Replacement(
        source,
        local_window=4,
        older_candidates=5,
        older_top_k=3,
        sink_tokens=2,
        library=library,
    )
    chunked = Replacement(
        source,
        local_window=4,
        older_candidates=5,
        older_top_k=3,
        sink_tokens=2,
        library=library,
    )
    generator = torch.Generator().manual_seed(37)
    hidden = torch.randn(1, 11, 4, generator=generator)
    positions = torch.arange(11)
    frequencies = positions.float().reshape(1, 11, 1) * torch.tensor(
        [0.1, 0.3, 0.1, 0.3]
    ).reshape(1, 1, 4)
    embeddings = (frequencies.cos(), frequencies.sin())

    expected, _ = full(
        hidden,
        embeddings,
        None,
        position_ids=positions.unsqueeze(0),
    )
    rows = []
    for start, end in ((0, 6), (6, 10), (10, 11)):
        output, _ = chunked(
            hidden[:, start:end],
            tuple(item[:, start:end] for item in embeddings),
            None,
            position_ids=positions[start:end].unsqueeze(0),
        )
        rows.append(output)
    actual = torch.cat(rows, dim=1)

    assert torch.equal(actual, expected)
    assert chunked.tokens_seen == 11
    assert chunked.metrics["state_bytes"] > 0
    with pytest.raises(ValueError, match="advance contiguously"):
        chunked(
            hidden[:, :1],
            tuple(item[:, :1] for item in embeddings),
            None,
            position_ids=torch.tensor([[13]]),
        )
    full.close()
    chunked.close()


def test_bounded_generation_advances_absolute_positions_without_hf_cache():
    torch = pytest.importorskip("torch")

    class FakeKernel:
        calls = ()

        def clear_metrics(self):
            return None

    class FakeModel:
        def __init__(self):
            self.calls = []

        def __call__(
            self,
            *,
            input_ids,
            position_ids,
            use_cache,
            logits_to_keep,
        ):
            assert logits_to_keep == 1
            self.calls.append((input_ids.clone(), position_ids.clone(), use_cache))
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)
            logits[:, -1, len(self.calls)] = 1
            return SimpleNamespace(logits=logits)

    runtime = object.__new__(NativeBitNetRuntime)
    runtime.model = FakeModel()
    runtime.kernel = FakeKernel()
    runtime._native_attention_layers = [
        SimpleNamespace(
            metrics={
                "tokens_seen": 5,
                "logical_read_bytes": 100,
                "state_bytes": 20,
                "scratch_bytes": 10,
            }
        )
    ]
    runtime.enable_bounded_attention = lambda **_kwargs: None
    runtime.reset_bounded_attention = lambda: None
    runtime.decode = lambda tokens: " ".join(map(str, tokens))

    result = runtime.generate_tokens_bounded(
        [4, 5, 6],
        max_new_tokens=3,
    )

    calls = runtime.model.calls
    assert calls[0][1].tolist() == [[0, 1, 2]]
    assert calls[1][1].tolist() == [[3]]
    assert calls[2][1].tolist() == [[4]]
    assert all(use_cache is False for _, _, use_cache in calls)
    assert result.generated_tokens == (1, 2, 3)
    assert result.attention_tokens_seen == 5
