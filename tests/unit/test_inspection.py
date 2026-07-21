import json

import numpy as np
import pytest

from engram.models.fixture import create_tiny_fixture
from engram.models.inspection import ModelValidationError, inspect_model


def test_fixture_and_inspection_are_deterministic(tmp_path):
    first = create_tiny_fixture(tmp_path / "first", seed=11)
    second = create_tiny_fixture(tmp_path / "second", seed=11)
    first_info = inspect_model(first)
    second_info = inspect_model(second)
    assert first_info.source_hash == second_info.source_hash
    assert first_info.file_hashes["weights.npz"] == second_info.file_hashes["weights.npz"]
    assert first_info.tensor_count == 21
    assert first_info.hidden_size == 16
    assert first_info.intermediate_size == 32


def test_inspector_rejects_unsupported_model(tmp_path):
    model = create_tiny_fixture(tmp_path / "model")
    config_path = model / "config.json"
    config = json.loads(config_path.read_text())
    config["model_type"] = "bert"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ModelValidationError, match="unsupported model_type"):
        inspect_model(model)


def test_inspector_rejects_wrong_mlp_shape(tmp_path):
    model = create_tiny_fixture(tmp_path / "model")
    with np.load(model / "weights.npz", allow_pickle=False) as archive:
        weights = {name: archive[name] for name in archive.files}
    weights["model.layers.0.mlp.gate_proj.weight"] = np.zeros((31, 16), dtype=np.float32)
    np.savez(model / "weights.npz", **weights)
    with pytest.raises(ModelValidationError, match="has shape"):
        inspect_model(model)


def test_inspector_rejects_bias_enabled_mlp_configuration(tmp_path):
    model = create_tiny_fixture(tmp_path / "model")
    config_path = model / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["mlp_bias"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ModelValidationError, match="mlp_bias=false"):
        inspect_model(model)
