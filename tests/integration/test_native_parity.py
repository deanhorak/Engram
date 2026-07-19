import json
import subprocess
from pathlib import Path

import pytest

from engram.compiler import compile_model
from engram.models import create_tiny_fixture
from engram.runtime import EngramRuntime


def test_native_and_python_greedy_tokens_match(tmp_path):
    executable = Path("build/engram-run").resolve()
    if not executable.is_file():
        pytest.skip("build/engram-run has not been built")
    source = create_tiny_fixture(tmp_path / "source")
    package = compile_model(source, tmp_path / "model.engram")
    runtime = EngramRuntime(package)
    expected, _ = runtime.generate_tokens(runtime.tokenize_fixture("hello"), max_tokens=8, exact_vocab=True)
    output = subprocess.check_output(
        [str(executable), str(package), "--prompt", "hello", "--max-tokens", "8", "--exact-vocab"],
        text=True,
    )
    native = [int(line.split(",", 1)[0]) for line in output.splitlines()[1:]]
    assert native == expected

    runtime.reset()
    runtime.cache.set_bypass(True)
    approximate, metrics = runtime.generate_tokens(
        runtime.tokenize_fixture("hello"), max_tokens=8, exact_vocab=False
    )
    output = subprocess.check_output(
        [
            str(executable),
            str(package),
            "--prompt",
            "hello",
            "--max-tokens",
            "8",
            "--no-cache",
        ],
        text=True,
    )
    native_approximate = [int(line.split(",", 1)[0]) for line in output.splitlines()[1:]]
    assert native_approximate == approximate
    assert all(item.vocabulary_proxy_records < runtime.manifest["vocab_size"] for item in metrics)
