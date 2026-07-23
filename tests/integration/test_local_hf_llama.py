import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")
try:
    LlamaForCausalLM = transformers.LlamaForCausalLM
except (ImportError, RuntimeError) as error:
    pytest.skip(f"local Transformers Llama stack is unavailable: {error}", allow_module_level=True)

from engram.models.inspection import inspect_model, load_layer_mlp  # noqa: E402
from engram.semantic.oracle import analyze_magnitude_oracle  # noqa: E402
from engram.tracing.format import TraceReader  # noqa: E402
from engram.tracing.teacher import capture_teacher_traces  # noqa: E402


def test_local_hf_llama_exact_mlp_boundary(tmp_path):
    torch.manual_seed(101)
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config)
    model_path = tmp_path / "model"
    model.save_pretrained(model_path, safe_serialization=True)
    del model

    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps({"input_ids": [1, 5, 9, 2], "input_type": "prose"}) + "\n",
        encoding="utf-8",
    )
    traces = tmp_path / "traces"
    capture_teacher_traces(model_path, traces, dataset=dataset, samples=1)
    report = analyze_magnitude_oracle(model_path, traces)

    assert inspect_model(model_path).model_type == "llama"
    assert report["status"] == "measured_local_model"
    assert report["record_count"] == 8
    overall = next(group for group in report["groups"] if group["scope"] == "all")
    assert overall["teacher_reconstruction_relative_l2"]["p95"] < 1e-5


def test_local_hf_bfloat16_mlp_weights_load_as_float32(tmp_path):
    config = transformers.LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
    )
    model_path = tmp_path / "bf16-model"
    LlamaForCausalLM(config).to(dtype=torch.bfloat16).save_pretrained(
        model_path, safe_serialization=True
    )

    gate, up, down = load_layer_mlp(model_path, 0)

    assert gate.dtype == np.float32
    assert up.dtype == np.float32
    assert down.dtype == np.float32


def test_local_hf_mlp_only_trace_samples_positions(tmp_path):
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model_path = tmp_path / "model"
    LlamaForCausalLM(config).save_pretrained(model_path, safe_serialization=True)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps({"input_ids": [1, 3, 5, 7, 9, 2]}) + "\n",
        encoding="utf-8",
    )
    traces = tmp_path / "traces"

    capture_teacher_traces(
        model_path,
        traces,
        dataset=dataset,
        samples=1,
        include_attention=False,
        tokens_per_sequence=3,
        layers=[1],
    )

    reader = TraceReader(traces)
    shard = next(reader.iter_shards())
    assert reader.manifest["metadata"]["included_boundaries"] == ["mlp"]
    assert reader.manifest["metadata"]["selected_layers"] == [1]
    assert shard["token_position"].shape == (3,)
    assert shard["layer_1_mlp_input"].shape == (3, 8)
    assert not any(field.startswith("layer_0_") for field in shard)
    assert not any("attention" in field for field in shard)
