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
)


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
