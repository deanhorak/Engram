import numpy as np

from engram.models.fixture import create_tiny_fixture
from engram.semantic.memory import SemanticLayer, build_semantic_package, load_semantic_layer
from engram.semantic.swiglu import swiglu


def test_semantic_read_and_package_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    gate = rng.normal(size=(7, 3)).astype(np.float32)
    up = rng.normal(size=(7, 3)).astype(np.float32)
    values = rng.normal(size=(7, 3)).astype(np.float32)
    state = rng.normal(size=3).astype(np.float32)
    memory = SemanticLayer(gate, up, values)
    full = memory.read(state, range(7), top_k=7)
    np.testing.assert_allclose(full.output, swiglu(state, gate, up, values.T), rtol=1e-6, atol=1e-6)
    assert full.active_records == 7
    assert full.estimated_bytes_read > 0

    model = create_tiny_fixture(tmp_path / "model")
    package = tmp_path / "model.engram"
    build_semantic_package(model, package)
    loaded = load_semantic_layer(package, 0)
    assert loaded.records == 32
    assert loaded.gate_keys.flags.writeable is False
