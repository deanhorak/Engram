import sys
import types

import pytest

from engram.models.inspection import ModelValidationError, resolve_model_path


def test_existing_model_directory_is_used_without_hub(tmp_path):
    model = tmp_path / "model"
    model.mkdir()

    assert resolve_model_path(model) == model.resolve()


def test_model_id_is_downloaded_through_hugging_face_cache(tmp_path, monkeypatch):
    cached = tmp_path / "hub" / "snapshot"
    cached.mkdir(parents=True)
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(cached)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=snapshot_download),
    )

    assert resolve_model_path("org/llama-model") == cached.resolve()
    assert calls[0]["repo_id"] == "org/llama-model"
    assert "*.safetensors" in calls[0]["allow_patterns"]


def test_missing_explicit_local_path_is_not_treated_as_model_id(tmp_path):
    missing = tmp_path / "missing-model"

    with pytest.raises(ModelValidationError, match="model path is not a directory"):
        resolve_model_path(missing)
