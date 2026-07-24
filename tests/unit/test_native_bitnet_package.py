from types import SimpleNamespace

import pytest

from engram.compiler.native_bitnet import (
    _inventory,
    _write_non_mlp_weights,
)
from engram.evaluation.native_bitnet_attention import (
    _attention_replacement_class,
)
from engram.evaluation.native_bitnet_parity import (
    _disable_broken_optional_transformers_dependencies,
)
from engram.runtime.native_bitnet import (
    _materialized_bitlinear_class,
    _safe_package_path,
)


def test_package_paths_are_confined_to_the_package(tmp_path):
    assert _safe_package_path(tmp_path, "mlp/model.bin") == (tmp_path / "mlp/model.bin")
    for unsafe in ("", "../model.bin", "/tmp/model.bin"):
        with pytest.raises(ValueError, match="unsafe"):
            _safe_package_path(tmp_path, unsafe)


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
