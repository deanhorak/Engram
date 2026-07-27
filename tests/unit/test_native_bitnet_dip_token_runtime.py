from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

import engram.runtime.native_bitnet_dip_token as token_runtime
from engram.compiler.native_bitnet import (
    NATIVE_BITNET_ATTENTION_OPERATOR,
    NATIVE_BITNET_DIP_DERIVED_MANIFEST_SHA256,
    NATIVE_BITNET_DIP_OPERATOR,
)
from engram.runtime.native_bitnet_dip_token import (
    NativeBitNetDIPTokenRuntime,
    NativeBitNetDIPTokenRuntimeError,
)


class _Tokenizer:
    chat_template = "fixture"

    def __init__(self) -> None:
        self.encoded = []
        self.decoded = []

    def encode(self, text, *, add_special_tokens):
        self.encoded.append((text, add_special_tokens))
        return [128000, 1502]

    def decode(self, tokens, *, skip_special_tokens):
        self.decoded.append((list(tokens), skip_special_tokens))
        return "native answer"


class _FakeLibrary:
    def __init__(self) -> None:
        self.created = 0
        self.destroyed = 0
        self.resets = 0
        self.generates = 0
        self.generate_status = 0
        self.fresh = True

    def engram_native_bitnet_token_create_v1(
        self,
        config_pointer,
        output_pointer,
        _error,
        _error_capacity,
    ):
        config = config_pointer._obj
        assert config.abi_version == 1
        assert config.struct_size == ctypes.sizeof(token_runtime._ConfigV1)
        assert config.flags == 0
        self.created += 1
        output_pointer._obj.value = 123
        return 0

    def engram_native_bitnet_token_destroy_v1(self, handle):
        assert handle.value == 123
        self.destroyed += 1

    def engram_native_bitnet_token_get_info_v1(
        self,
        handle,
        info_pointer,
        _error,
        _error_capacity,
    ):
        assert handle.value == 123
        info = info_pointer._obj
        info.layers = 30
        info.hidden_size = 2560
        info.intermediate_size = 6912
        info.vocabulary_size = 128256
        info.max_position_embeddings = 64
        info.query_heads = 20
        info.key_value_heads = 5
        info.head_dimension = 128
        info.thread_count = 12
        info.local_window = 16
        info.older_candidates = 8
        info.older_top_k = 4
        info.sink_tokens = 2
        info.eos_token_count = 2
        info.eos_token_ids[0] = 128001
        info.eos_token_ids[1] = 128009
        info.semantic_backend = NATIVE_BITNET_DIP_OPERATOR.encode()
        info.package_manifest_sha256 = (
            NATIVE_BITNET_DIP_DERIVED_MANIFEST_SHA256.encode()
        )
        return 0

    def engram_native_bitnet_token_reset_v1(
        self,
        handle,
        _error,
        _error_capacity,
    ):
        assert handle.value == 123
        self.resets += 1
        self.fresh = True
        return 0

    def engram_native_bitnet_token_generate_v1(
        self,
        handle,
        prompt,
        prompt_count,
        max_new_tokens,
        output,
        output_capacity,
        output_count,
        metrics_pointer,
        error,
        _error_capacity,
    ):
        assert handle.value == 123
        assert self.fresh
        self.generates += 1
        if self.generate_status:
            self.fresh = False
            error.value = b"forced native generation failure"
            return self.generate_status
        assert prompt_count == 2
        assert [prompt[index] for index in range(prompt_count)] == [
            128000,
            1502,
        ]
        assert max_new_tokens == 2
        assert output_capacity == 2
        output[0] = 7
        output[1] = 128009
        output_count._obj.value = 2
        metrics = metrics_pointer._obj
        metrics.prompt_tokens = 2
        metrics.generated_tokens = 2
        metrics.positions_processed = 3
        metrics.stage_calls = 60
        metrics.semantic_calls = 60
        metrics.semantic_rows = 90
        metrics.semantic_selected_records = 120
        metrics.semantic_kernel_cache_line_bytes = 1000
        metrics.semantic_global_metadata_bytes = 12
        metrics.semantic_cache_line_bytes = 1012
        metrics.semantic_maximum_scratch_bytes = 256
        metrics.semantic_elapsed_ns = 2_000_000_000
        metrics.attention_elapsed_ns = 3_000_000_000
        metrics.attention_logical_read_bytes = 400
        metrics.attention_state_bytes = 500
        metrics.attention_scratch_bytes = 600
        metrics.attention_eviction_events = 7
        metrics.attention_older_candidate_entries_scored = 80
        metrics.attention_older_selected_entries = 40
        metrics.attention_sink_insertions = 20
        metrics.attention_heavy_hitter_updates = 60
        metrics.qkv_projection_ns = 100
        metrics.rope_ns = 200
        metrics.native_attention_ns = 300
        metrics.output_projection_ns = 400
        metrics.call_elapsed_ns = 5_000_000_000
        metrics.prefill_elapsed_ns = 3_000_000_000
        metrics.decode_elapsed_ns = 2_000_000_000
        metrics.stopped_on_eos = 1
        self.fresh = False
        return 0


def _manifest():
    return {
        "format": "engram-native-bitnet",
        "version": 1,
        "runtime": {
            "mlp_mode": NATIVE_BITNET_DIP_OPERATOR,
            "attention_mode": NATIVE_BITNET_ATTENTION_OPERATOR,
        },
        "tokenizer": {
            "path": "tokenizer",
            "files": ["tokenizer.json"],
            "fix_mistral_regex": True,
        },
        "files": {},
    }


def _make_runtime(tmp_path: Path, monkeypatch, library=None):
    fake_library = _FakeLibrary() if library is None else library
    tokenizer = _Tokenizer()
    monkeypatch.setattr(
        token_runtime,
        "_load_pinned_manifest",
        lambda _root: _manifest(),
    )
    monkeypatch.setattr(
        token_runtime,
        "_verify_tokenizer_assets",
        lambda _root, _manifest_value: tmp_path,
    )
    monkeypatch.setattr(
        token_runtime,
        "_load_packaged_tokenizer",
        lambda _directory, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(
        token_runtime,
        "_load_library",
        lambda _path: (fake_library, tmp_path / "runtime.so"),
    )
    return (
        NativeBitNetDIPTokenRuntime(tmp_path, threads=12),
        fake_library,
        tokenizer,
    )


def test_native_dip_token_runtime_maps_generation_and_metrics(
    tmp_path,
    monkeypatch,
):
    runtime, library, tokenizer = _make_runtime(tmp_path, monkeypatch)
    with runtime:
        result = runtime.generate("rendered chat", max_new_tokens=2)

    assert result.prompt_tokens == (128000, 1502)
    assert result.generated_tokens == (7, 128009)
    assert result.text == "native answer"
    assert result.elapsed_seconds == 5.0
    assert result.prefill_seconds == 3.0
    assert result.decode_seconds == 2.0
    assert result.mlp_calls == 60
    assert result.scheduled_mlp_bytes == 1012
    assert result.attention_state_bytes == 500
    assert runtime.last_metrics["attention_eviction_events"] == 7
    assert (
        runtime.last_metrics["attention_older_candidate_entries_scored"] == 80
    )
    assert runtime.last_metrics["attention_older_selected_entries"] == 40
    assert runtime.last_metrics["attention_sink_insertions"] == 20
    assert runtime.last_metrics["attention_heavy_hitter_updates"] == 60
    assert result.stopped_on_eos
    assert result.controller_mode == "native_exact_operator_residual"
    assert result.decoder_layer_forward_calls == 0
    assert tokenizer.encoded == [("rendered chat", True)]
    assert tokenizer.decoded == [([7, 128009], True)]
    assert library.resets == 1
    assert library.generates == 1
    assert library.destroyed == 1


def test_native_dip_token_runtime_recovers_after_native_failure(
    tmp_path,
    monkeypatch,
):
    library = _FakeLibrary()
    library.generate_status = 6
    runtime, library, _tokenizer = _make_runtime(
        tmp_path,
        monkeypatch,
        library,
    )

    with pytest.raises(
        NativeBitNetDIPTokenRuntimeError,
        match="forced native generation failure",
    ):
        runtime.generate_tokens([1], max_new_tokens=2)

    assert library.resets == 2
    assert library.fresh
    runtime.close()


def test_native_dip_token_runtime_rejects_invalid_requests_before_native_call(
    tmp_path,
    monkeypatch,
):
    runtime, library, _tokenizer = _make_runtime(tmp_path, monkeypatch)

    for prompt, budget, message in (
        ([], 1, "must not be empty"),
        ([128256], 1, "outside authenticated vocabulary"),
        ([1], 0, "must be positive"),
        ([1] * 64, 2, "exceed authenticated context"),
    ):
        with pytest.raises(ValueError, match=message):
            runtime.generate_tokens(prompt, max_new_tokens=budget)

    assert library.resets == 0
    assert library.generates == 0
    runtime.close()


def test_native_dip_token_runtime_close_is_idempotent_and_closed_fails(
    tmp_path,
    monkeypatch,
):
    runtime, library, _tokenizer = _make_runtime(tmp_path, monkeypatch)
    runtime.close()
    runtime.close()

    assert library.destroyed == 1
    with pytest.raises(NativeBitNetDIPTokenRuntimeError, match="closed"):
        runtime.generate_tokens([1], max_new_tokens=1)


@pytest.mark.parametrize("threads", [True, 0, -1, 257, 1.5])
def test_native_dip_token_runtime_rejects_invalid_thread_counts(
    tmp_path,
    monkeypatch,
    threads,
):
    monkeypatch.setattr(
        token_runtime,
        "_load_pinned_manifest",
        lambda _root: pytest.fail("package should not be inspected"),
    )
    with pytest.raises(ValueError, match="threads"):
        NativeBitNetDIPTokenRuntime(tmp_path, threads=threads)


def test_pinned_manifest_rejects_a_symlink_package_root(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(
        NativeBitNetDIPTokenRuntimeError,
        match="non-symlink directory",
    ):
        token_runtime._load_pinned_manifest(linked)
