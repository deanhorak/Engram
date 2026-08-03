from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from engram.compiler.olmoe_native import (
    compile_olmoe_native_package,
    validate_olmoe_native_package,
)
from engram.models.fixture import create_tiny_olmoe_fixture
from engram.models.olmoe_native import repack_olmoe_non_mlp_weights
from engram.models.olmoe_q7 import repack_olmoe_q7_model
from engram.runtime.olmoe_native import OLMoENativePackageRuntime
from engram.utils import sha256_file


def test_compiled_int8_attention_manifest_selects_native_mode(tmp_path: Path):
    model = create_tiny_olmoe_fixture(
        tmp_path / "model", num_experts=4, num_experts_per_token=1
    )
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "x": 1}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(model / "tokenizer.json"))
    q7 = repack_olmoe_q7_model(model, tmp_path / "experts.q7", group_size=64)
    non_mlp = tmp_path / "non_mlp.safetensors"
    repack_olmoe_non_mlp_weights(model, non_mlp)
    package = tmp_path / "package"
    compiled = compile_olmoe_native_package(
        model,
        q7,
        non_mlp,
        package,
        kernel_threads=1,
        attention_local_window=128,
        attention_storage="int8",
    )
    manifest = validate_olmoe_native_package(
        package, expected_manifest_sha256=compiled["manifest_sha256"]
    )
    assert manifest["runtime"]["attention_mode"] == (
        "native_streaming_w128_int8_c8_k4_sinks2"
    )
    assert manifest["runtime"]["attention_storage"] == "int8"
    assert manifest["runtime"]["attention_policy"]["local_window"] == 128
    with OLMoENativePackageRuntime(
        package,
        manifest_sha256=sha256_file(package / "manifest.json"),
        library="build/libengram_olmoe_token_runtime.so",
        threads=1,
    ) as runtime:
        result = runtime.runtime.forward([1])
        assert result.metrics["attention_state_bytes"] > 0
        assert runtime.runtime.local_int8 is True
