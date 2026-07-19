from engram.compiler import compile_model
from engram.models import create_tiny_fixture
from engram.runtime import EngramRuntime
from engram.runtime.validation import validate_package
import pytest


def test_compiler_creates_source_independent_generating_runtime(tmp_path):
    source = create_tiny_fixture(tmp_path / "source")
    package = compile_model(source, tmp_path / "tiny.engram")
    runtime = EngramRuntime(package)
    tokens, metrics = runtime.generate_tokens([1, 4, 7], max_tokens=5)
    assert len(tokens) == 5
    assert all(0 <= token < 64 for token in tokens)
    assert all(item.cycles == 2 for item in metrics)
    assert runtime.manifest["does_not_require_source_transformer"] is True
    assert (package / "tokenizer" / "metadata.json").is_file()
    assert not (package / "semantic" / "layer-0000" / "gate_keys.npy").exists()
    assert (package / "semantic" / "layer-0000" / "quantized" / "gate_codes.npy").is_file()
    assert (package / "semantic" / "layer-0000" / "quantized" / "ivf" / "centroids.npy").is_file()
    assert all(item.semantic_proxy_records < 64 for item in metrics)
    assert all(item.semantic_probed_clusters > 0 for item in metrics)
    assert (package / "vocabulary" / "ivf" / "centroids.npy").is_file()
    assert all(item.vocabulary_proxy_records < 64 for item in metrics)
    assert all(item.vocabulary_probed_clusters > 0 for item in metrics)

    # The package remains usable when the source directory is no longer in scope.
    moved = tmp_path / "source-hidden"
    source.rename(moved)
    second = EngramRuntime(package)
    assert second.generate_tokens([1, 4, 7], max_tokens=5)[0] == tokens


def test_python_runtime_rejects_corrupt_index_before_mmap(tmp_path):
    source = create_tiny_fixture(tmp_path / "source")
    package = compile_model(source, tmp_path / "tiny.engram")
    index_path = package / "semantic" / "layer-0000" / "quantized" / "ivf" / "posting_indices.npy"
    payload = bytearray(index_path.read_bytes())
    payload[-1] ^= 0x01
    index_path.write_bytes(payload)

    with pytest.raises(ValueError, match="checksum mismatch"):
        EngramRuntime(package)
    assert validate_package(package)["valid"] is False
