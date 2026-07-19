import pytest

from engram.compiler import compile_model
from engram.models import create_tiny_fixture


def test_compile_resumes_valid_package_and_rejects_changed_options(tmp_path):
    source = create_tiny_fixture(tmp_path / "source")
    package = compile_model(source, tmp_path / "model.engram", cycles=2)
    manifest_time = (package / "manifest.json").stat().st_mtime_ns
    assert compile_model(source, package, cycles=2) == package
    assert (package / "manifest.json").stat().st_mtime_ns == manifest_time
    with pytest.raises(ValueError, match="options changed"):
        compile_model(source, package, cycles=3)
