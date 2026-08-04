"""Structural Qwen3 teacher-trace gate.

The fixture is intentionally tiny and randomly initialized.  It proves the
loader and exact MLP-boundary contract only; it is not a language-quality
benchmark and does not exercise the native BitNet compiler.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

# Some environments have a scikit-learn binary built for an older NumPy.  The
# Qwen modeling module only needs the availability flag for optional generation
# helpers, so make the same narrow compatibility patch as the production tracer.
try:
    import transformers.utils as transformers_utils
    import transformers.utils.import_utils as transformers_imports

    if transformers_imports.is_sklearn_available():
        try:
            import sklearn  # noqa: F401
        except ImportError:
            transformers_imports.is_sklearn_available = lambda: False
            transformers_utils.is_sklearn_available = lambda: False
    Qwen3Config = transformers.Qwen3Config
    Qwen3ForCausalLM = transformers.Qwen3ForCausalLM
except (ImportError, RuntimeError, AttributeError) as error:
    pytest.skip(f"local Transformers Qwen3 stack is unavailable: {error}", allow_module_level=True)

from engram.models.qwen3 import audit_qwen3_source  # noqa: E402
from engram.semantic.swiglu import swiglu  # noqa: E402
from engram.tracing.format import TraceReader  # noqa: E402
from engram.tracing.teacher import capture_teacher_traces  # noqa: E402
from engram.models.inspection import load_layer_mlp  # noqa: E402


def test_local_hf_qwen3_exact_mlp_boundary(tmp_path):
    torch.manual_seed(707)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        rope_theta=1_000_000.0,
    )
    model_path = tmp_path / "model"
    Qwen3ForCausalLM(config).save_pretrained(model_path, safe_serialization=True)

    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps({"input_ids": [1, 5, 9, 2], "input_type": "prose"}) + "\n",
        encoding="utf-8",
    )
    traces = tmp_path / "traces"
    capture_teacher_traces(
        model_path,
        traces,
        dataset=dataset,
        samples=1,
        include_attention=False,
    )

    audit = audit_qwen3_source(model_path)
    shard = next(TraceReader(traces).iter_shards())
    assert audit.capabilities["exact_swiglu_decomposition"]
    assert audit.capabilities["generic_hf_teacher_trace"]
    assert audit.capabilities["native_bitnet_compilation"] is False
    assert shard["layer_0_mlp_input"].shape == (4, 16)

    for layer in range(2):
        gate, up, down = load_layer_mlp(model_path, layer)
        reconstructed = swiglu(
            shard[f"layer_{layer}_mlp_input"], gate, up, down
        )
        np.testing.assert_allclose(
            reconstructed,
            shard[f"layer_{layer}_mlp_output"],
            rtol=2e-5,
            atol=2e-6,
        )
