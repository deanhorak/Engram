import ctypes
import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from engram.compiler.olmoe_native import (
    OLMoENativePackageError,
    compile_olmoe_native_package,
    validate_olmoe_native_package,
)
from engram.models.fixture import create_tiny_olmoe_fixture
from engram.models.inspection import load_local_named_tensors
from engram.models.olmoe_native import repack_olmoe_non_mlp_weights
from engram.models.olmoe_q7 import (
    LoadedOLMoEQ7Artifact,
    bf16_from_bits,
    repack_olmoe_q7_model,
)
from engram.runtime.olmoe_native import (
    OLMoENativePackageRuntime,
    OLMoENativeRuntimeError,
    OLMoENativeTokenRuntime,
    _validate_attention_head_policies,
    _validate_attention_policies,
)
from engram.runtime import olmoe_native as olmoe_native_runtime


def _bf16(values):
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return bf16_from_bits(((bits + bias) >> np.uint32(16)).astype(np.uint16))


def _norm(values, weight, epsilon=1e-6):
    return values / np.sqrt(np.mean(values * values) + epsilon) * weight


def _rope(values, heads, position):
    result = values.reshape(heads, -1).copy()
    dimension = result.shape[1]
    half = dimension // 2
    for index in range(half):
        frequency = 10000.0 ** (-2.0 * index / dimension)
        cosine = np.cos(position * frequency)
        sine = np.sin(position * frequency)
        first = result[:, index].copy()
        second = result[:, index + half].copy()
        result[:, index] = first * cosine - second * sine
        result[:, index + half] = second * cosine + first * sine
    return result


def _attention_capacity_bytes(
    *,
    query_heads,
    key_value_heads,
    head_dimension,
    local_window,
    older_candidates,
    older_top_k,
):
    state = (
        2 * local_window * key_value_heads * head_dimension * 4
        + query_heads * local_window * 4
        + local_window * 8
        + 2 * query_heads * older_candidates * head_dimension * 4
        + query_heads * older_candidates * (4 + 8 + 1)
    )
    scratch = (
        (local_window + older_candidates) * 4
        + older_candidates * 4
        + (local_window + older_top_k) * 4
        + older_top_k * 8
    )
    return state, scratch


def test_per_layer_attention_policy_validation_is_strict():
    policy = {
        "local_window": 16,
        "older_candidates": 8,
        "older_top_k": 4,
        "sink_tokens": 2,
    }
    assert _validate_attention_policies([policy, policy], layers=2) == (
        policy,
        policy,
    )
    with pytest.raises(ValueError, match="count must equal"):
        _validate_attention_policies([policy], layers=2)
    with pytest.raises(ValueError, match="invalid fields"):
        _validate_attention_policies(
            [{**policy, "unexpected": 1}, policy],
            layers=2,
        )
    with pytest.raises(ValueError, match="must contain integers"):
        _validate_attention_policies(
            [{**policy, "local_window": True}, policy],
            layers=2,
        )
    with pytest.raises(ValueError, match="is inconsistent"):
        _validate_attention_policies(
            [{**policy, "older_top_k": 9}, policy],
            layers=2,
        )
    with pytest.raises(ValueError, match="is inconsistent"):
        _validate_attention_policies(
            [{**policy, "sink_tokens": -1}, policy],
            layers=2,
        )


def test_per_head_attention_policy_validation_is_strict():
    policy = {
        "local_window": 16,
        "older_candidates": 8,
        "older_top_k": 4,
        "sink_tokens": 2,
    }
    expected = tuple(
        tuple(dict(policy) for _head in range(4)) for _layer in range(2)
    )
    assert (
        _validate_attention_head_policies(
            [[policy] * 4, [policy] * 4],
            layers=2,
            query_heads=4,
        )
        == expected
    )
    with pytest.raises(ValueError, match="layer count must equal"):
        _validate_attention_head_policies(
            [[policy] * 4],
            layers=2,
            query_heads=4,
        )
    with pytest.raises(ValueError, match="layer count must equal"):
        _validate_attention_head_policies(
            "not policies",
            layers=2,
            query_heads=4,
        )
    with pytest.raises(ValueError, match="count for layer 1"):
        _validate_attention_head_policies(
            [[policy] * 4, [policy] * 3],
            layers=2,
            query_heads=4,
        )
    with pytest.raises(ValueError, match="count for layer 0"):
        _validate_attention_head_policies(
            ["not head policies", [policy] * 4],
            layers=2,
            query_heads=4,
        )
    with pytest.raises(ValueError, match="invalid fields"):
        _validate_attention_head_policies(
            [
                [{**policy, "unexpected": 1}, policy, policy, policy],
                [policy] * 4,
            ],
            layers=2,
            query_heads=4,
        )
    with pytest.raises(ValueError, match="must contain integers"):
        _validate_attention_head_policies(
            [
                [{**policy, "local_window": True}, policy, policy, policy],
                [policy] * 4,
            ],
            layers=2,
            query_heads=4,
        )
    with pytest.raises(ValueError, match="must contain integers"):
        _validate_attention_head_policies(
            [
                [{**policy, "older_top_k": 1.0}, policy, policy, policy],
                [policy] * 4,
            ],
            layers=2,
            query_heads=4,
        )
    with pytest.raises(ValueError, match="is inconsistent"):
        _validate_attention_head_policies(
            [
                [{**policy, "older_top_k": 9}, policy, policy, policy],
                [policy] * 4,
            ],
            layers=2,
            query_heads=4,
        )


def test_headwise_attention_requires_additive_native_symbol(
    tmp_path,
    monkeypatch,
):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "hidden_size": 16,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_hidden_layers": 2,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10_000.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        olmoe_native_runtime.ctypes,
        "CDLL",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        olmoe_native_runtime,
        "_configure",
        lambda _library: (False, True, False),
    )
    policy = {
        "local_window": 3,
        "older_candidates": 2,
        "older_top_k": 1,
        "sink_tokens": 1,
    }
    with pytest.raises(
        OLMoENativeRuntimeError,
        match="no headwise-attention ABI",
    ):
        OLMoENativeTokenRuntime(
            config,
            tmp_path / "non-mlp.safetensors",
            tmp_path / "model.q7",
            tmp_path / "legacy.so",
            attention_head_policies=[
                [policy] * 4,
                [policy] * 4,
            ],
        )


def _prompt_reference(model, q7_path, tokens, *, diagnostics=False):
    base_names = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    for layer in range(2):
        base = f"model.layers.{layer}"
        base_names.extend(
            [
                f"{base}.input_layernorm.weight",
                f"{base}.post_attention_layernorm.weight",
                f"{base}.self_attn.q_proj.weight",
                f"{base}.self_attn.k_proj.weight",
                f"{base}.self_attn.v_proj.weight",
                f"{base}.self_attn.o_proj.weight",
                f"{base}.self_attn.q_norm.weight",
                f"{base}.self_attn.k_norm.weight",
            ]
        )
    tensors = {
        name: _bf16(value)
        for name, value in load_local_named_tensors(model, base_names).items()
    }
    hidden = tensors["model.embed_tokens.weight"][tokens].copy()
    with LoadedOLMoEQ7Artifact(q7_path) as q7:
        for layer in range(2):
            base = f"model.layers.{layer}"
            normalized = np.stack(
                [
                    _norm(row, tensors[f"{base}.input_layernorm.weight"])
                    for row in hidden
                ]
            )
            query = normalized @ tensors[f"{base}.self_attn.q_proj.weight"].T
            key = normalized @ tensors[f"{base}.self_attn.k_proj.weight"].T
            value = normalized @ tensors[f"{base}.self_attn.v_proj.weight"].T
            query = np.stack(
                [
                    _norm(row, tensors[f"{base}.self_attn.q_norm.weight"])
                    for row in query
                ]
            )
            key = np.stack(
                [_norm(row, tensors[f"{base}.self_attn.k_norm.weight"]) for row in key]
            )
            query = np.stack(
                [_rope(row, 4, position) for position, row in enumerate(query)]
            )
            key = np.stack(
                [_rope(row, 4, position) for position, row in enumerate(key)]
            )
            values = value.reshape(len(tokens), 4, 4)
            attention_rows = []
            for position in range(len(tokens)):
                heads = []
                for head in range(4):
                    scores = (
                        key[: position + 1, head] @ query[position, head] / np.sqrt(4.0)
                    )
                    probabilities = np.exp(scores - scores.max())
                    probabilities /= probabilities.sum()
                    heads.append(probabilities @ values[: position + 1, head])
                attention_rows.append(np.concatenate(heads))
            attention = (
                np.stack(attention_rows) @ tensors[f"{base}.self_attn.o_proj.weight"].T
            )
            hidden += attention
            semantic_input = np.stack(
                [
                    _norm(
                        row,
                        tensors[f"{base}.post_attention_layernorm.weight"],
                    )
                    for row in hidden
                ]
            )
            router = q7.router(layer)
            semantic = np.zeros_like(hidden)
            for row, state in enumerate(semantic_input):
                logits = state @ router.T
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                selected = np.argsort(-probabilities, kind="stable")[:2]
                for expert in selected:
                    weights = q7.expert(layer, int(expert))
                    gate = weights["gate"] @ state
                    activation = (gate / (1.0 + np.exp(-gate))) * (
                        weights["up"] @ state
                    )
                    semantic[row] += probabilities[expert] * (
                        weights["down"] @ activation
                    )
            hidden += semantic
    final = _norm(hidden[-1], tensors["model.norm.weight"])
    logits = tensors["lm_head.weight"] @ final
    next_token = int(np.argmax(logits))
    if diagnostics:
        return next_token, final, logits
    return next_token


def test_native_olmoe_token_step_matches_single_position_reference(tmp_path):
    library = Path("build/libengram_olmoe_token_runtime.so")
    if not library.is_file():
        pytest.skip("native OLMoE token runtime has not been built")
    model = create_tiny_olmoe_fixture(tmp_path / "model")
    q7 = repack_olmoe_q7_model(model, tmp_path / "model.q7", group_size=8)
    non_mlp = tmp_path / "non_mlp.safetensors"
    report = repack_olmoe_non_mlp_weights(model, non_mlp)
    expected, expected_hidden, expected_logits = _prompt_reference(
        model,
        q7,
        [1],
        diagnostics=True,
    )
    expected_prompt = _prompt_reference(model, q7, [1, 2, 3])

    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
    ) as runtime:
        first = runtime.forward([1])
        assert first.next_token == expected
        assert runtime.position == 1
        assert first.metrics["positions_processed"] == 1
        assert first.metrics["q7_scheduled_bytes"] > 0
        assert first.metrics["attention_state_bytes"] > 0
        diagnostic_hidden, diagnostic_logits = runtime.last_diagnostics()
        np.testing.assert_allclose(
            diagnostic_hidden,
            expected_hidden,
            rtol=2e-5,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            diagnostic_logits,
            expected_logits,
            rtol=2e-5,
            atol=2e-6,
        )
        runtime.reset()
        with pytest.raises(OLMoENativeRuntimeError, match="diagnostic storage"):
            runtime.last_diagnostics()
        replay = runtime.forward([1])
        assert replay.next_token == first.next_token
        assert runtime.position == 1
        runtime.reset()
        prompt = runtime.forward([1, 2, 3])
        assert prompt.next_token == expected_prompt
        assert runtime.position == 3
        runtime.reset()
        runtime.forward([1, 2])
        incremental = runtime.forward([3])
        assert incremental.next_token == prompt.next_token
        assert runtime.position == 3
        runtime.reset()
        generated = runtime.generate([1], max_new_tokens=2)
        assert len(generated) == 2
        assert generated[0] == expected
        assert runtime.position == 2
        runtime.reset()
        for _position in range(17):
            sustained = runtime.forward([1])
        assert runtime.position == 17
        assert sustained.metrics["attention_eviction_events"] == 2
        assert sustained.metrics["attention_older_candidate_entries_scored"] == 8
        assert sustained.metrics["attention_older_selected_entries"] == 8
        assert sustained.metrics["attention_sink_insertions"] == 8
        assert sustained.metrics["attention_heavy_hitter_updates"] == 0
        assert sustained.metrics["attention_logical_read_bytes"] > 0
        assert sustained.metrics["attention_scratch_bytes"] > 0
        runtime.reset()
        reset = runtime.forward([1])
        assert reset.metrics["attention_eviction_events"] == 0
        assert reset.metrics["attention_older_candidate_entries_scored"] == 0
        assert reset.metrics["attention_sink_insertions"] == 0

    assert report["tensor_count"] == 19
    assert report["file_bytes"] == non_mlp.stat().st_size


def test_native_olmoe_layered_attention_matches_scalar_and_sums_state(
    tmp_path,
):
    library = Path("build/libengram_olmoe_token_runtime.so")
    if not library.is_file():
        pytest.skip("native OLMoE token runtime has not been built")
    model = create_tiny_olmoe_fixture(tmp_path / "model")
    q7 = repack_olmoe_q7_model(model, tmp_path / "model.q7", group_size=8)
    non_mlp = tmp_path / "non_mlp.safetensors"
    repack_olmoe_non_mlp_weights(model, non_mlp)
    scalar_policy = {
        "local_window": 3,
        "older_candidates": 2,
        "older_top_k": 1,
        "sink_tokens": 1,
    }
    tokens = [1, 2, 3, 4]
    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
        **scalar_policy,
    ) as scalar:
        scalar_result = scalar.forward(tokens)
        scalar_hidden, scalar_logits = scalar.last_diagnostics()
    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
        attention_policies=[scalar_policy, scalar_policy],
    ) as layered:
        layered_result = layered.forward(tokens)
        layered_hidden, layered_logits = layered.last_diagnostics()
    assert layered_result.next_token == scalar_result.next_token
    assert {
        name: value
        for name, value in layered_result.metrics.items()
        if name not in {"elapsed_ns", "q7_elapsed_ns"}
    } == {
        name: value
        for name, value in scalar_result.metrics.items()
        if name not in {"elapsed_ns", "q7_elapsed_ns"}
    }
    np.testing.assert_array_equal(layered_hidden, scalar_hidden)
    np.testing.assert_array_equal(layered_logits, scalar_logits)

    heterogeneous = [
        {
            "local_window": 1,
            "older_candidates": 2,
            "older_top_k": 1,
            "sink_tokens": 1,
        },
        {
            "local_window": 3,
            "older_candidates": 1,
            "older_top_k": 1,
            "sink_tokens": 1,
        },
    ]
    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
        attention_policies=heterogeneous,
    ) as layered:
        result = layered.forward(tokens)
        assert layered.position == 4
        assert layered.attention_policies == tuple(heterogeneous)
        assert result.metrics["attention_state_bytes"] == 1_148
        assert result.metrics["attention_scratch_bytes"] == 80
        assert result.metrics["attention_eviction_events"] == 4
        assert result.metrics[
            "attention_older_candidate_entries_scored"
        ] == 24
        assert result.metrics["attention_older_selected_entries"] == 16
        assert result.metrics["attention_sink_insertions"] == 8
        layered.reset()
        reset = layered.forward([1])
        assert reset.metrics["attention_eviction_events"] == 0
        assert reset.metrics["attention_older_candidate_entries_scored"] == 0

    with pytest.raises(ValueError, match="cannot be combined"):
        OLMoENativeTokenRuntime(
            model / "config.json",
            non_mlp,
            q7,
            library,
            attention_policies=heterogeneous,
            local_window=3,
        )


def test_native_olmoe_headwise_attention_matches_layered_and_reports_counters(
    tmp_path,
):
    library = Path("build/libengram_olmoe_token_runtime.so")
    if not library.is_file():
        pytest.skip("native OLMoE token runtime has not been built")
    native = ctypes.CDLL(str(library.resolve()))
    if not hasattr(native, "engram_olmoe_token_open_headwise_v1"):
        pytest.skip("native OLMoE headwise-attention ABI has not been built")
    model = create_tiny_olmoe_fixture(tmp_path / "model")
    q7 = repack_olmoe_q7_model(model, tmp_path / "model.q7", group_size=8)
    non_mlp = tmp_path / "non_mlp.safetensors"
    repack_olmoe_non_mlp_weights(model, non_mlp)
    homogeneous = {
        "local_window": 3,
        "older_candidates": 2,
        "older_top_k": 1,
        "sink_tokens": 1,
    }
    tokens = [1, 2, 3, 4]
    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
        attention_policies=[homogeneous, homogeneous],
    ) as layered:
        layered_result = layered.forward(tokens)
        layered_hidden, layered_logits = layered.last_diagnostics()
    nested_homogeneous = tuple(
        tuple(dict(homogeneous) for _head in range(4))
        for _layer in range(2)
    )
    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
        attention_head_policies=nested_homogeneous,
    ) as headwise:
        headwise_result = headwise.forward(tokens)
        headwise_hidden, headwise_logits = headwise.last_diagnostics()
        assert headwise.attention_policies is None
        assert headwise.attention_head_policies == nested_homogeneous
    assert headwise_result.next_token == layered_result.next_token
    for name in (
        "positions_processed",
        "attention_weight_bytes",
        "q7_scheduled_bytes",
        "attention_logical_read_bytes",
        "attention_older_candidate_entries_scored",
        "attention_older_selected_entries",
        "attention_sink_insertions",
        "attention_heavy_hitter_updates",
    ):
        assert headwise_result.metrics[name] == layered_result.metrics[name]
    assert headwise_result.metrics["attention_eviction_events"] == (
        layered_result.metrics["attention_eviction_events"] * 4
    )
    layered_per_layer = _attention_capacity_bytes(
        query_heads=4,
        key_value_heads=4,
        head_dimension=4,
        local_window=3,
        older_candidates=2,
        older_top_k=1,
    )
    headwise_per_head = _attention_capacity_bytes(
        query_heads=1,
        key_value_heads=1,
        head_dimension=4,
        local_window=3,
        older_candidates=2,
        older_top_k=1,
    )
    assert layered_result.metrics["attention_state_bytes"] == (
        2 * layered_per_layer[0]
    )
    assert layered_result.metrics["attention_scratch_bytes"] == (
        2 * layered_per_layer[1]
    )
    assert headwise_result.metrics["attention_state_bytes"] == (
        2 * 4 * headwise_per_head[0]
    )
    assert headwise_result.metrics["attention_scratch_bytes"] == (
        2 * 4 * headwise_per_head[1]
    )
    np.testing.assert_array_equal(headwise_hidden, layered_hidden)
    np.testing.assert_array_equal(headwise_logits, layered_logits)

    compact = {
        "local_window": 1,
        "older_candidates": 1,
        "older_top_k": 1,
        "sink_tokens": 0,
    }
    mixed = (
        (compact, homogeneous, compact, homogeneous),
        (homogeneous, compact, homogeneous, compact),
    )
    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
        attention_head_policies=mixed,
    ) as headwise:
        result = headwise.forward(tokens)
        assert headwise.position == len(tokens)
        assert headwise.attention_head_policies == mixed
        compact_per_head = _attention_capacity_bytes(
            query_heads=1,
            key_value_heads=1,
            head_dimension=4,
            local_window=1,
            older_candidates=1,
            older_top_k=1,
        )
        assert result.metrics["attention_state_bytes"] == (
            4 * compact_per_head[0] + 4 * headwise_per_head[0]
        )
        assert result.metrics["attention_scratch_bytes"] == (
            4 * compact_per_head[1] + 4 * headwise_per_head[1]
        )
        assert result.metrics["attention_eviction_events"] == 16
        assert result.metrics[
            "attention_older_candidate_entries_scored"
        ] == 16
        assert result.metrics["attention_older_selected_entries"] == 16
        assert result.metrics["attention_sink_insertions"] == 4
        assert result.metrics["attention_heavy_hitter_updates"] == 4
        assert result.metrics["attention_logical_read_bytes"] == 2_176
        headwise.reset()
        reset = headwise.forward([1])
        assert reset.metrics["attention_eviction_events"] == 0
        assert reset.metrics[
            "attention_older_candidate_entries_scored"
        ] == 0
        assert reset.metrics["attention_sink_insertions"] == 0

    with pytest.raises(ValueError, match="cannot be combined"):
        OLMoENativeTokenRuntime(
            model / "config.json",
            non_mlp,
            q7,
            library,
            attention_policies=[homogeneous, homogeneous],
            attention_head_policies=nested_homogeneous,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        OLMoENativeTokenRuntime(
            model / "config.json",
            non_mlp,
            q7,
            library,
            local_window=3,
            attention_head_policies=nested_homogeneous,
        )


def test_authenticated_native_olmoe_package_generation_and_tamper_rejection(
    tmp_path,
):
    library = Path("build/libengram_olmoe_token_runtime.so")
    if not library.is_file():
        pytest.skip("native OLMoE token runtime has not been built")
    model = create_tiny_olmoe_fixture(tmp_path / "model")
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.save(str(model / "tokenizer.json"))
    q7 = repack_olmoe_q7_model(model, tmp_path / "model.q7", group_size=8)
    non_mlp = tmp_path / "non_mlp.safetensors"
    repack_olmoe_non_mlp_weights(model, non_mlp)
    package = tmp_path / "package"
    compiled = compile_olmoe_native_package(
        model,
        q7,
        non_mlp,
        package,
        kernel_threads=2,
    )
    manifest_hash = compiled["manifest_sha256"]

    manifest = validate_olmoe_native_package(
        package,
        expected_manifest_sha256=manifest_hash,
    )
    assert manifest["does_not_require_transformers"]
    assert manifest["runtime"]["device"] == "cpu"
    expected = _prompt_reference(model, q7, [1])
    with OLMoENativePackageRuntime(
        package,
        manifest_sha256=manifest_hash,
        library=library,
    ) as runtime:
        result = runtime.generate("hello", max_new_tokens=1)
    assert result["prompt_token_ids"] == [1]
    assert result["generated_token_ids"] == [expected]

    tokenizer_path = package / "tokenizer" / "tokenizer.json"
    tokenizer_path.write_bytes(tokenizer_path.read_bytes() + b" ")
    with pytest.raises(OLMoENativePackageError, match="package file is invalid"):
        validate_olmoe_native_package(
            package,
            expected_manifest_sha256=manifest_hash,
        )
