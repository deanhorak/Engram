import json

import numpy as np
import pytest

from engram.models.fixture import create_tiny_fixture
from engram.models.inspection import ModelValidationError, inspect_model
from engram.models.qwen3 import audit_qwen3_source, qwen3_mlp_tensor_names


def _make_qwen3_fixture(path):
    model = create_tiny_fixture(
        path,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=4,
        vocab_size=64,
    )
    config_path = model / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 128,
            "rope_theta": 1_000_000.0,
            "tie_word_embeddings": True,
        }
    )
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return model


def test_qwen3_structural_audit_accepts_canonical_dense_swiglu(tmp_path):
    model = _make_qwen3_fixture(tmp_path / "qwen3")
    inspection = inspect_model(model)
    audit = audit_qwen3_source(model)

    assert inspection.model_type == "qwen3"
    assert audit.architecture == "Qwen3ForCausalLM"
    assert audit.num_key_value_heads == 2
    assert audit.head_dim == 4
    assert audit.rope_theta == 1_000_000.0
    assert audit.capabilities["exact_swiglu_decomposition"]
    assert not audit.capabilities["native_bitnet_compilation"]
    assert qwen3_mlp_tensor_names(1) == (
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.1.mlp.up_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    )


def test_qwen3_audit_rejects_non_qwen_architecture(tmp_path):
    model = _make_qwen3_fixture(tmp_path / "qwen3")
    config_path = model / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["architectures"] = ["LlamaForCausalLM"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ModelValidationError, match="unsupported Qwen3 architecture"):
        audit_qwen3_source(model, hash_weights=False)


def test_qwen3_inspector_still_checks_projection_shapes(tmp_path):
    model = _make_qwen3_fixture(tmp_path / "qwen3")
    with np.load(model / "weights.npz", allow_pickle=False) as archive:
        weights = {name: archive[name] for name in archive.files}
    weights["model.layers.0.mlp.down_proj.weight"] = np.zeros((31, 16), dtype=np.float32)
    np.savez(model / "weights.npz", **weights)
    with pytest.raises(ModelValidationError, match="has shape"):
        audit_qwen3_source(model, hash_weights=False)
