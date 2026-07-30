import ctypes
import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from engram.compiler.olmoe_native import (
    OLMoENativePackageError,
    compile_olmoe_native_package,
    validate_olmoe_native_package,
)
from engram.models.fixture import create_tiny_olmoe_fixture
from engram.models.inspection import load_local_named_tensors
from engram.models.olmoe_native import repack_olmoe_non_mlp_weights
from engram.models.olmoe_q7 import (
    LoadedOLMoEQ7Artifact,
    bf16_from_bits,
    repack_olmoe_q7_model,
)
from engram.runtime.olmoe_native import (
    OLMoENativePackageRuntime,
    OLMoENativeRuntimeError,
    OLMoENativeTokenRuntime,
    _validate_attention_head_policies,
    _validate_attention_policies,
)
from engram.runtime import olmoe_native as olmoe_native_runtime


def _bf16(values):
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return bf16_from_bits(((bits + bias) >> np.uint32(16)).astype(np.uint16))


def _norm(values, weight, epsilon=1e-6):
    return values / np.sqrt(np.mean(values * values) + epsilon) * weight


def _rope(values, heads, position):
    result = values.reshape(heads, -1).copy()
    dimension = result.shape[1]
    half = dimension // 2
    for index in range(half):
        frequency = 10000.0 ** (-2.0 * index / dimension)
        cosine = np.cos(position * frequency)
        sine = np.sin(position * frequency)
        first = result[:, index].copy()
        second = result[:, index + half].copy()
        result[:, index] = first * cosine - second * sine
        result[:, index + half] = second * cosine + first * sine
    return result


def _attention_capacity_bytes(
    *,
    query_heads,
    key_value_heads,
    head_dimension,
    local_window,
    older_candidates,
    older_top_k,
):
    state = (
        2 * local_window * key_value_heads * head_dimension * 4
        + query_heads * local_window * 4
        + local_window * 8
        + 2 * query_heads * older_candidates * head_dimension * 4
        + query_heads * older_candidates * (4 + 8 + 1)
    )
    scratch = (
        (local_window + older_candidates) * 4
        + older_candidates * 4
        + (local_window + older_top_k) * 4
        + older_top_k * 8
    )
    return state, scratch


def test_per_layer_attention_policy_validation_is_strict():
    policy = {
        "local_window": 16,
        "older_candidates": 8,
        "older_top_k": 4,
        "sink_tokens": 2,
    }
    assert _validate_attention_policies([policy, policy], layers=2) == (
        policy,
        policy,
    )
    with pytest.raises(ValueError, match="count must equal"):
        _validate_attention_policies([policy], layers=2)
    with pytest.raises(ValueError, match="invalid fields"):
        _validate_attention_policies(
            [{**policy, "unexpected": 1}, policy],
            layers=2,
        )
    with pytest.raises(ValueError, match="must contain integers"):
        _validate_attention_policies(
            [{**policy, "local_window": True}, policy],
            layers=2,
        )
    with pytest.raises(ValueError, match="is inconsistent"):
        _validate_attention_policies(
            [{**policy, "older_top_k": 9}, policy],
            layers=2,
        )
    with pytest.raises(ValueError, match="is inconsistent"):
        _validate_attention_policies(
            [{**policy, "sink_tokens": -1}, policy],
            layers=2,
        )


def test_per_head_attention_policy_validation_is_strict():
    policy = {
        "local_window": 16,
        "older_candidates": 8,
        "older_top_k": 4,
        "sink_tokens": 2,
    }
    expected = tuple(tuple(dict(policy) for _head in range(4)) for _layer in range(2))
    assert (
        _validate_attention_head_policies(
            [[policy] * 4, [policy] * 4],
            layers=2,
            query_heads=4,
        )
        == expected
    )
    with pytest.raises(ValueError, match="layer count must equal"):
        _validate_attention_head_policies(
            [[policy] * 4],
            layers=2,
            query_heads=4,
        )
    with pytest.raises(ValueError, match="layer count must equal"):
        _validate_attention_head_policies(
            "not policies",
            layers=2,
            query_heads=4,
        )
    with pytest.raises(ValueError, match="count for layer 1"):
        _validate_attention_head_policies(
            [[policy] * 4, [policy] * 3],
            layers=2,
            query_heads=4,
        )
    with pytest.raises(ValueError, match="count for layer 0"):
        _validate_attention_head_policies(
            ["not head policies", [policy] * 4],
            layers=2,
            query_heads=4,
        )
    with pytest.raises(ValueError, match="invalid fields"):
        _validate_attention_head_policies(
            [
                [{**policy, "unexpected": 1}, policy, policy, policy],
                [policy] * 4,
            ],
            layers=2,
            query_heads=4,
        )
    with pytest.raises(ValueError, match="must contain integers"):
        _validate_attention_head_policies(
            [
                [{**policy, "local_window": True}, policy, policy, policy],
                [policy] * 4,
            ],
            layers=2,
            query_heads=4,
        )
    with pytest.raises(ValueError, match="must contain integers"):
        _validate_attention_head_policies(
            [
                [{**policy, "older_top_k": 1.0}, policy, policy, policy],
                [policy] * 4,
            ],
            layers=2,
            query_heads=4,
        )
    with pytest.raises(ValueError, match="is inconsistent"):
        _validate_attention_head_policies(
            [
                [{**policy, "older_top_k": 9}, policy, policy, policy],
                [policy] * 4,
            ],
            layers=2,
            query_heads=4,
        )


def test_headwise_attention_requires_additive_native_symbol(
    tmp_path,
    monkeypatch,
):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "hidden_size": 16,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_hidden_layers": 2,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10_000.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        olmoe_native_runtime.ctypes,
        "CDLL",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        olmoe_native_runtime,
        "_configure",
        lambda _library: (
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ),
    )
    policy = {
        "local_window": 3,
        "older_candidates": 2,
        "older_top_k": 1,
        "sink_tokens": 1,
    }
    with pytest.raises(
        OLMoENativeRuntimeError,
        match="no headwise-attention ABI",
    ):
        OLMoENativeTokenRuntime(
            config,
            tmp_path / "non-mlp.safetensors",
            tmp_path / "model.q7",
            tmp_path / "legacy.so",
            attention_head_policies=[
                [policy] * 4,
                [policy] * 4,
            ],
        )


class _FakeNativeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class _FakeEpisodicTokenLibrary:
    def __init__(
        self,
        *,
        episodic=True,
        episodic_headwise=True,
        episodic_headwise_v2=True,
        shadow_trace=True,
        episodic_mass_trace=True,
        episodic_slot_trace=True,
        regular_entry_trace=True,
    ):
        self.position = 0
        self.normal_forward_calls = 0
        self.episodic_forward_calls = 0
        self.closed = False
        self.open_config = None
        self.open_policy = None
        self.open_head_mask = None
        self.open_logit_bias = None
        self.headwise_v1_open_calls = 0
        self.headwise_v2_open_calls = 0
        self.shadow_trace_open_calls = 0
        self.open_shadow_policy = None
        self.last_tokens = None
        self.last_write_slots = None
        self.last_read_spans = None
        self.mass_trace_valid = False
        self.engram_olmoe_token_open = _FakeNativeFunction(self._open)
        self.engram_olmoe_token_close = _FakeNativeFunction(self._close)
        self.engram_olmoe_token_reset = _FakeNativeFunction(self._reset)
        self.engram_olmoe_token_position = _FakeNativeFunction(
            lambda _handle: self.position
        )
        self.engram_olmoe_token_vocabulary_size = _FakeNativeFunction(
            lambda _handle: 97
        )
        self.engram_olmoe_token_forward = _FakeNativeFunction(self._forward)
        self.engram_olmoe_token_copy_last_diagnostics = _FakeNativeFunction(
            lambda *_args: 0
        )
        if episodic:
            self.engram_olmoe_token_open_episodic_v1 = _FakeNativeFunction(
                self._open_episodic
            )
            self.engram_olmoe_token_forward_episodic_v1 = _FakeNativeFunction(
                self._forward_episodic
            )
            self.engram_olmoe_token_copy_episodic_metrics_v1 = _FakeNativeFunction(
                self._copy_episodic_metrics
            )
            if episodic_headwise:
                self.engram_olmoe_token_open_episodic_headwise_v1 = _FakeNativeFunction(
                    self._open_episodic_headwise
                )
            if episodic_headwise_v2:
                self.engram_olmoe_token_open_episodic_headwise_v2 = _FakeNativeFunction(
                    self._open_episodic_headwise_v2
                )
            if shadow_trace:
                self.engram_olmoe_token_open_episodic_shadow_trace_v1 = (
                    _FakeNativeFunction(self._open_episodic_shadow_trace)
                )
                self.engram_olmoe_token_copy_last_shadow_trace_v1 = _FakeNativeFunction(
                    self._copy_last_shadow_trace
                )
        if episodic_mass_trace:
            self.engram_olmoe_token_copy_last_episodic_mass_trace_v1 = (
                _FakeNativeFunction(self._copy_last_episodic_mass_trace)
            )
        if episodic_slot_trace:
            self.engram_olmoe_token_copy_last_episodic_slot_trace_v1 = (
                _FakeNativeFunction(self._copy_last_episodic_slot_trace)
            )
        if regular_entry_trace:
            self.engram_olmoe_token_copy_last_regular_entry_trace_v1 = (
                _FakeNativeFunction(self._copy_last_regular_entry_trace)
            )

    @staticmethod
    def _contents(pointer, structure):
        return ctypes.cast(pointer, ctypes.POINTER(structure)).contents

    def _capture_config(self, pointer):
        config = self._contents(pointer, olmoe_native_runtime._Config)
        self.open_config = {
            name: getattr(config, name) for name, _ctype in config._fields_
        }

    def _open(self, config, _error, _capacity):
        self._capture_config(config)
        return 41

    def _open_episodic(self, config, policy, _error, _capacity):
        self._capture_config(config)
        native_policy = self._contents(
            policy,
            olmoe_native_runtime._EpisodicPolicyV1,
        )
        self.open_policy = {
            name: int(getattr(native_policy, name))
            for name, _ctype in native_policy._fields_
        }
        return 43

    def _open_episodic_headwise(
        self,
        config,
        policy,
        head_mask,
        mask_count,
        _error,
        _capacity,
    ):
        self.headwise_v1_open_calls += 1
        self._open_episodic(config, policy, _error, _capacity)
        count = int(mask_count)
        self.open_head_mask = np.ctypeslib.as_array(
            head_mask,
            shape=(count,),
        ).copy()
        return 47

    def _open_episodic_headwise_v2(
        self,
        config,
        policy,
        head_mask,
        mask_count,
        episodic_logit_bias,
        _error,
        _capacity,
    ):
        self.headwise_v2_open_calls += 1
        self._open_episodic(config, policy, _error, _capacity)
        count = int(mask_count)
        self.open_head_mask = np.ctypeslib.as_array(
            head_mask,
            shape=(count,),
        ).copy()
        self.open_logit_bias = float(episodic_logit_bias)
        return 53

    def _open_episodic_shadow_trace(
        self,
        config,
        policy,
        head_mask,
        mask_count,
        episodic_logit_bias,
        shadow_policy,
        _error,
        _capacity,
    ):
        self.shadow_trace_open_calls += 1
        self._open_episodic(config, policy, _error, _capacity)
        count = int(mask_count)
        self.open_head_mask = np.ctypeslib.as_array(
            head_mask,
            shape=(count,),
        ).copy()
        self.open_logit_bias = float(episodic_logit_bias)
        native_shadow = self._contents(
            shadow_policy,
            olmoe_native_runtime._AttentionPolicyV1,
        )
        self.open_shadow_policy = {
            name: int(getattr(native_shadow, name))
            for name, _ctype in native_shadow._fields_
        }
        return 59

    @staticmethod
    def _copy_last_shadow_trace(
        _handle,
        input_norm,
        input_count,
        base_projected,
        base_count,
        target_residual,
        target_count,
        _error,
        _capacity,
    ):
        counts = (int(input_count), int(base_count), int(target_count))
        if len(set(counts)) != 1:
            return 1
        count = counts[0]
        np.ctypeslib.as_array(input_norm, shape=(count,))[:] = np.arange(
            count,
            dtype=np.float32,
        )
        np.ctypeslib.as_array(base_projected, shape=(count,))[:] = (
            np.arange(count, dtype=np.float32) + 100.0
        )
        np.ctypeslib.as_array(target_residual, shape=(count,))[:] = (
            np.arange(count, dtype=np.float32) - 100.0
        )
        return 0

    def _copy_last_episodic_mass_trace(
        self,
        _handle,
        base_attention_output,
        base_count,
        regular_component,
        regular_count,
        episodic_component,
        episodic_count,
        regular_mass,
        regular_mass_count,
        episodic_mass,
        episodic_mass_count,
        shadow_scheduled_mass,
        shadow_mass_count,
        _error,
        _capacity,
    ):
        if not self.mass_trace_valid:
            return 1
        output_counts = (
            int(base_count),
            int(regular_count),
            int(episodic_count),
        )
        mass_counts = (
            int(regular_mass_count),
            int(episodic_mass_count),
            int(shadow_mass_count),
        )
        if len(set(output_counts)) != 1 or len(set(mass_counts)) != 1:
            return 1
        output_count = output_counts[0]
        mass_count = mass_counts[0]
        np.ctypeslib.as_array(
            base_attention_output,
            shape=(output_count,),
        )[:] = np.arange(output_count, dtype=np.float32)
        np.ctypeslib.as_array(
            regular_component,
            shape=(output_count,),
        )[:] = np.arange(output_count, dtype=np.float32) + 100.0
        np.ctypeslib.as_array(
            episodic_component,
            shape=(output_count,),
        )[:] = np.arange(output_count, dtype=np.float32) - 100.0
        np.ctypeslib.as_array(
            regular_mass,
            shape=(mass_count,),
        )[:] = np.arange(mass_count, dtype=np.float32) + 200.0
        np.ctypeslib.as_array(
            episodic_mass,
            shape=(mass_count,),
        )[:] = np.arange(mass_count, dtype=np.float32) + 300.0
        np.ctypeslib.as_array(
            shadow_scheduled_mass,
            shape=(mass_count,),
        )[:] = np.arange(mass_count, dtype=np.float32) + 400.0
        return 0

    def _copy_last_episodic_slot_trace(
        self,
        _handle,
        slot_mass,
        slot_mass_count,
        slot_values,
        slot_value_count,
        _error,
        _capacity,
    ):
        if not self.mass_trace_valid:
            return 1
        mass_count = int(slot_mass_count)
        value_count = int(slot_value_count)
        if value_count != mass_count * 4:
            return 1
        np.ctypeslib.as_array(
            slot_mass,
            shape=(mass_count,),
        )[:] = np.arange(mass_count, dtype=np.float32) + 500.0
        np.ctypeslib.as_array(
            slot_values,
            shape=(value_count,),
        )[:] = np.arange(value_count, dtype=np.float32) + 600.0
        return 0

    def _copy_last_regular_entry_trace(
        self,
        _handle,
        entry_mass,
        entry_mass_count,
        entry_values,
        entry_value_count,
        valid_kind,
        valid_kind_count,
        positions,
        position_count,
        _error,
        _capacity,
    ):
        if not self.mass_trace_valid:
            return 1
        mass_count = int(entry_mass_count)
        value_count = int(entry_value_count)
        if (
            int(valid_kind_count) != mass_count
            or int(position_count) != mass_count
            or value_count != mass_count * 4
        ):
            return 1
        np.ctypeslib.as_array(entry_mass, shape=(mass_count,))[:] = (
            np.arange(mass_count, dtype=np.float32) + 700.0
        )
        np.ctypeslib.as_array(entry_values, shape=(value_count,))[:] = (
            np.arange(value_count, dtype=np.float32) + 800.0
        )
        np.ctypeslib.as_array(valid_kind, shape=(mass_count,))[:] = (
            np.arange(mass_count, dtype=np.uint8) % np.uint8(3)
        )
        np.ctypeslib.as_array(positions, shape=(mass_count,))[:] = np.arange(
            mass_count,
            dtype=np.uint64,
        )
        return 0

    def _close(self, _handle):
        self.closed = True

    def _reset(self, _handle):
        self.position = 0
        self.mass_trace_valid = False

    @staticmethod
    def _fill_base_metrics(pointer, positions):
        metrics = _FakeEpisodicTokenLibrary._contents(
            pointer,
            olmoe_native_runtime._Metrics,
        )
        for index, (name, _ctype) in enumerate(metrics._fields_, start=1):
            setattr(metrics, name, positions * 100 + index)

    def _forward(
        self,
        _handle,
        tokens,
        length,
        next_token,
        metrics,
        _error,
        _capacity,
    ):
        count = int(length)
        self.normal_forward_calls += 1
        self.last_tokens = np.ctypeslib.as_array(
            tokens,
            shape=(count,),
        ).copy()
        self.position += count
        self._contents(next_token, ctypes.c_int64).value = 71
        self._fill_base_metrics(metrics, self.position)
        return 0

    def _forward_episodic(
        self,
        _handle,
        tokens,
        length,
        write_slots,
        read_spans,
        next_token,
        metrics,
        _error,
        _capacity,
    ):
        count = int(length)
        self.episodic_forward_calls += 1
        self.last_tokens = np.ctypeslib.as_array(
            tokens,
            shape=(count,),
        ).copy()
        self.last_write_slots = np.ctypeslib.as_array(
            write_slots,
            shape=(count,),
        ).copy()
        self.last_read_spans = np.ctypeslib.as_array(
            read_spans,
            shape=(count,),
        ).copy()
        self.mass_trace_valid = bool(self.last_read_spans[-1] >= 0)
        self.position += count
        self._contents(next_token, ctypes.c_int64).value = 73
        self._fill_base_metrics(metrics, self.position)
        return 0

    def _copy_episodic_metrics(
        self,
        _handle,
        metrics,
        _error,
        _capacity,
    ):
        native = self._contents(
            metrics,
            olmoe_native_runtime._EpisodicMetricsV1,
        )
        for index, (name, _ctype) in enumerate(native._fields_, start=1):
            setattr(native, name, index * 11)
        return 0


def _mock_runtime_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "hidden_size": 16,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_hidden_layers": 2,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10_000.0,
            }
        ),
        encoding="utf-8",
    )
    return path


def _mock_token_runtime(
    tmp_path,
    monkeypatch,
    *,
    episodic=True,
    episodic_headwise=True,
    episodic_headwise_v2=True,
    shadow_trace=True,
    episodic_mass_trace=True,
    episodic_slot_trace=True,
    regular_entry_trace=True,
    **kwargs,
):
    library = _FakeEpisodicTokenLibrary(
        episodic=episodic,
        episodic_headwise=episodic_headwise,
        episodic_headwise_v2=episodic_headwise_v2,
        shadow_trace=shadow_trace,
        episodic_mass_trace=episodic_mass_trace,
        episodic_slot_trace=episodic_slot_trace,
        regular_entry_trace=regular_entry_trace,
    )
    monkeypatch.setattr(
        olmoe_native_runtime.ctypes,
        "CDLL",
        lambda _path: library,
    )
    runtime = OLMoENativeTokenRuntime(
        _mock_runtime_config(tmp_path),
        tmp_path / "non-mlp.safetensors",
        tmp_path / "model.q7",
        tmp_path / "runtime.so",
        **kwargs,
    )
    return runtime, library


def test_mocked_episodic_binding_preserves_old_forward_and_reports_metrics(
    tmp_path,
    monkeypatch,
):
    runtime, library = _mock_token_runtime(
        tmp_path,
        monkeypatch,
        threads=3,
        local_window=7,
        older_candidates=6,
        older_top_k=2,
        sink_tokens=1,
        episodic_policy={"slots": 32, "span_size": 8},
    )
    assert library.open_policy == {"slots": 32, "span_size": 8}
    assert library.open_config["threads"] == 3
    assert library.open_config["local_window"] == 7
    assert library.open_config["older_candidates"] == 6
    assert library.open_config["older_top_k"] == 2
    assert library.open_config["sink_tokens"] == 1
    assert runtime.episodic_policy == {"slots": 32, "span_size": 8}
    assert runtime.episodic_spans == 4
    assert runtime.episodic_logit_bias == 0.0
    assert runtime.episodic_open_abi == "v1"
    assert runtime.episodic_metrics_available is True
    assert library.engram_olmoe_token_forward_episodic_v1.argtypes[3] == ctypes.POINTER(
        ctypes.c_int32
    )
    assert library.engram_olmoe_token_forward_episodic_v1.argtypes[4] == ctypes.POINTER(
        ctypes.c_int32
    )

    ordinary = runtime.forward([9])
    assert ordinary.next_token == 71
    assert library.normal_forward_calls == 1
    assert library.episodic_forward_calls == 0
    assert not any(name.startswith("episodic_") for name in ordinary.metrics)

    result = runtime.forward_episodic(
        [10, 11, 12],
        write_slots=[0, -1, 31],
        read_spans=[-1, 0, 3],
    )
    assert result.next_token == 73
    assert library.normal_forward_calls == 1
    assert library.episodic_forward_calls == 1
    np.testing.assert_array_equal(library.last_tokens, [10, 11, 12])
    np.testing.assert_array_equal(library.last_write_slots, [0, -1, 31])
    np.testing.assert_array_equal(library.last_read_spans, [-1, 0, 3])
    episodic_names = [
        name for name, _ctype in olmoe_native_runtime._EpisodicMetricsV1._fields_
    ]
    assert {name: result.metrics[name] for name in episodic_names} == {
        name: index * 11 for index, name in enumerate(episodic_names, start=1)
    }
    assert runtime.position == 4
    runtime.reset()
    assert runtime.position == 0
    runtime.close()
    assert library.closed


@pytest.mark.parametrize(
    "policy, message",
    [
        ({"slots": 32}, "invalid fields"),
        ({"slots": True, "span_size": 8}, "must contain integers"),
        ({"slots": 0, "span_size": 8}, "is inconsistent"),
        ({"slots": 31, "span_size": 8}, "is inconsistent"),
        (
            {"slots": np.iinfo(np.int32).max + 1, "span_size": 1},
            "is inconsistent",
        ),
    ],
)
def test_episodic_policy_validation_is_strict(policy, message):
    with pytest.raises(ValueError, match=message):
        olmoe_native_runtime._normalize_episodic_policy(policy)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        "1.0",
        float("nan"),
        float("inf"),
        -float("inf"),
        float(np.finfo(np.float32).max) * 2.0,
        10**1000,
    ],
)
def test_episodic_logit_bias_validation_is_strict(value):
    with pytest.raises(ValueError, match="episodic logit bias"):
        olmoe_native_runtime._normalize_episodic_logit_bias(value)


def test_episodic_logit_bias_is_canonical_float32():
    assert olmoe_native_runtime._normalize_episodic_logit_bias(-0.0) == 0.0
    assert olmoe_native_runtime._normalize_episodic_logit_bias(2) == 2.0
    assert olmoe_native_runtime._normalize_episodic_logit_bias(1.2) == float(
        np.float32(1.2)
    )


def test_episodic_head_mask_validation_is_strict():
    expected = np.asarray(
        [[1, 0, 1, 0], [0, 1, 0, 1]],
        dtype=np.uint8,
    )
    actual = olmoe_native_runtime._normalize_episodic_head_mask(
        expected.astype(bool),
        layers=2,
        query_heads=4,
    )
    assert actual.dtype == np.uint8
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)
    invalid = (
        ([1] * 8, "shape must equal"),
        ([[1, 0, 1], [0, 1, 0]], "shape must equal"),
        (np.ones((2, 4), dtype=np.float32), "booleans or integers"),
        ([[1, 0, 2, 0], [0, 1, 0, 1]], "zero or one"),
        ([[0, 0, 0, 0], [0, 0, 0, 0]], "at least one"),
    )
    for mask, message in invalid:
        with pytest.raises(ValueError, match=message):
            olmoe_native_runtime._normalize_episodic_head_mask(
                mask,
                layers=2,
                query_heads=4,
            )


def test_mocked_head_gated_episodic_binding_and_missing_abi(
    tmp_path,
    monkeypatch,
):
    head_mask = (
        (True, False, False, False),
        (False, False, True, False),
    )
    runtime, library = _mock_token_runtime(
        tmp_path,
        monkeypatch,
        episodic_policy={"slots": 32, "span_size": 8},
        episodic_head_mask=head_mask,
    )
    np.testing.assert_array_equal(
        library.open_head_mask,
        [1, 0, 0, 0, 0, 0, 1, 0],
    )
    assert runtime.episodic_head_mask == head_mask
    assert runtime.episodic_logit_bias == 0.0
    assert runtime.episodic_open_abi == "v1"
    assert library.headwise_v1_open_calls == 1
    assert library.headwise_v2_open_calls == 0
    assert library.engram_olmoe_token_open_episodic_headwise_v1.argtypes[
        2
    ] == ctypes.POINTER(ctypes.c_uint8)
    result = runtime.forward_episodic(
        [1],
        write_slots=[0],
        read_spans=[-1],
    )
    assert result.next_token == 73
    runtime.close()

    with pytest.raises(ValueError, match="requires an episodic policy"):
        _mock_token_runtime(
            tmp_path,
            monkeypatch,
            episodic_head_mask=head_mask,
        )
    with pytest.raises(
        OLMoENativeRuntimeError,
        match="no head-gated episodic-memory ABI",
    ):
        _mock_token_runtime(
            tmp_path,
            monkeypatch,
            episodic_headwise=False,
            episodic_policy={"slots": 32, "span_size": 8},
            episodic_head_mask=head_mask,
        )


def test_mocked_additive_episodic_binding_uses_v2_for_explicit_zero_and_nonzero(
    tmp_path,
    monkeypatch,
):
    head_mask = (
        (True, False, False, False),
        (False, False, True, False),
    )
    explicit_zero, zero_library = _mock_token_runtime(
        tmp_path,
        monkeypatch,
        episodic_policy={"slots": 32, "span_size": 8},
        episodic_head_mask=head_mask,
        episodic_logit_bias=0.0,
    )
    assert explicit_zero.episodic_logit_bias == 0.0
    assert explicit_zero.episodic_open_abi == "v2"
    assert zero_library.headwise_v1_open_calls == 0
    assert zero_library.headwise_v2_open_calls == 1
    assert zero_library.open_logit_bias == 0.0
    assert (
        zero_library.engram_olmoe_token_open_episodic_headwise_v2.argtypes[4]
        == ctypes.c_float
    )
    explicit_zero.close()

    biased, biased_library = _mock_token_runtime(
        tmp_path,
        monkeypatch,
        episodic_policy={"slots": 32, "span_size": 8},
        episodic_head_mask=head_mask,
        episodic_logit_bias=1.2,
    )
    expected = float(np.float32(1.2))
    assert biased.episodic_logit_bias == expected
    assert biased.episodic_open_abi == "v2"
    assert biased_library.headwise_v1_open_calls == 0
    assert biased_library.headwise_v2_open_calls == 1
    assert biased_library.open_logit_bias == expected
    biased.close()


def test_additive_episodic_binding_requires_policy_mask_and_v2(
    tmp_path,
    monkeypatch,
):
    head_mask = (
        (True, False, False, False),
        (False, False, True, False),
    )
    with pytest.raises(ValueError, match="requires an episodic policy"):
        _mock_token_runtime(
            tmp_path,
            monkeypatch,
            episodic_logit_bias=0.0,
        )
    with pytest.raises(ValueError, match="requires an explicit head mask"):
        _mock_token_runtime(
            tmp_path,
            monkeypatch,
            episodic_policy={"slots": 32, "span_size": 8},
            episodic_logit_bias=0.0,
        )
    with pytest.raises(
        OLMoENativeRuntimeError,
        match="no additive head-gated episodic-memory ABI",
    ):
        _mock_token_runtime(
            tmp_path,
            monkeypatch,
            episodic_headwise_v2=False,
            episodic_policy={"slots": 32, "span_size": 8},
            episodic_head_mask=head_mask,
            episodic_logit_bias=0.0,
        )


def test_mocked_shadow_trace_binding_is_explicit_and_copies_layer_major_arrays(
    tmp_path,
    monkeypatch,
):
    head_mask = (
        (True, True, True, True),
        (True, True, True, True),
    )
    shadow_policy = {
        "local_window": 128,
        "older_candidates": 8,
        "older_top_k": 4,
        "sink_tokens": 2,
    }
    runtime, library = _mock_token_runtime(
        tmp_path,
        monkeypatch,
        episodic_policy={"slots": 32, "span_size": 8},
        episodic_head_mask=head_mask,
        episodic_logit_bias=0.0,
        shadow_attention_policy=shadow_policy,
    )
    assert runtime.episodic_open_abi == "shadow_trace_v1"
    assert runtime.episodic_logit_bias == 0.0
    assert runtime.shadow_attention_policy == shadow_policy
    assert runtime.shadow_trace_available is True
    assert library.shadow_trace_open_calls == 1
    assert library.headwise_v1_open_calls == 0
    assert library.headwise_v2_open_calls == 0
    assert library.open_shadow_policy == shadow_policy
    assert (
        library.engram_olmoe_token_open_episodic_shadow_trace_v1.argtypes[4]
        == ctypes.c_float
    )
    assert library.engram_olmoe_token_open_episodic_shadow_trace_v1.argtypes[
        5
    ] == ctypes.POINTER(olmoe_native_runtime._AttentionPolicyV1)
    runtime.forward_episodic([1], [0], [-1])
    input_norm, base_projected, target_residual = runtime.last_shadow_trace()
    assert input_norm.shape == (2, 16)
    assert base_projected.shape == (2, 16)
    assert target_residual.shape == (2, 16)
    assert input_norm.dtype == np.float32
    np.testing.assert_array_equal(
        input_norm.reshape(-1),
        np.arange(32, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        base_projected.reshape(-1),
        np.arange(32, dtype=np.float32) + 100.0,
    )
    np.testing.assert_array_equal(
        target_residual.reshape(-1),
        np.arange(32, dtype=np.float32) - 100.0,
    )
    runtime.close()


def test_shadow_trace_binding_requires_all_inputs_and_additive_symbols(
    tmp_path,
    monkeypatch,
):
    head_mask = (
        (True, True, True, True),
        (True, True, True, True),
    )
    shadow_policy = {
        "local_window": 128,
        "older_candidates": 8,
        "older_top_k": 4,
        "sink_tokens": 2,
    }
    required = {
        "episodic_policy": {"slots": 32, "span_size": 8},
        "episodic_head_mask": head_mask,
        "episodic_logit_bias": 0.0,
    }
    for missing in required:
        kwargs = {**required, "shadow_attention_policy": shadow_policy}
        del kwargs[missing]
        with pytest.raises(ValueError, match="shadow attention tracing requires"):
            _mock_token_runtime(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(
        OLMoENativeRuntimeError,
        match="no same-state shadow-trace ABI",
    ):
        _mock_token_runtime(
            tmp_path,
            monkeypatch,
            shadow_trace=False,
            shadow_attention_policy=shadow_policy,
            **required,
        )
    runtime, _library = _mock_token_runtime(
        tmp_path,
        monkeypatch,
        **required,
    )
    with pytest.raises(
        OLMoENativeRuntimeError,
        match="was not opened for shadow tracing",
    ):
        runtime.last_shadow_trace()
    runtime.close()


def test_mocked_episodic_mass_trace_binding_copies_typed_layer_major_arrays(
    tmp_path,
    monkeypatch,
):
    head_mask = (
        (True, True, True, True),
        (True, True, True, True),
    )
    shadow_policy = {
        "local_window": 128,
        "older_candidates": 8,
        "older_top_k": 4,
        "sink_tokens": 2,
    }
    runtime, library = _mock_token_runtime(
        tmp_path,
        monkeypatch,
        episodic_policy={"slots": 32, "span_size": 8},
        episodic_head_mask=head_mask,
        episodic_logit_bias=0.0,
        shadow_attention_policy=shadow_policy,
    )
    assert runtime.episodic_mass_trace_available is True
    binding = library.engram_olmoe_token_copy_last_episodic_mass_trace_v1
    assert len(binding.argtypes) == 15
    for pointer_index in (1, 3, 5, 7, 9, 11):
        assert binding.argtypes[pointer_index] == ctypes.POINTER(ctypes.c_float)
        assert binding.argtypes[pointer_index + 1] == ctypes.c_size_t
    with pytest.raises(OLMoENativeRuntimeError):
        runtime.last_episodic_mass_trace()

    runtime.forward_episodic([1], [0], [-1])
    with pytest.raises(OLMoENativeRuntimeError):
        runtime.last_episodic_mass_trace()
    runtime.forward_episodic([2], [-1], [0])
    trace = runtime.last_episodic_mass_trace()
    assert isinstance(trace, olmoe_native_runtime.OLMoENativeEpisodicMassTrace)
    for array in (
        trace.base_attention_output,
        trace.regular_component,
        trace.episodic_component,
    ):
        assert array.shape == (2, 16)
        assert array.dtype == np.float32
        assert array.flags.c_contiguous
    for array in (
        trace.regular_mass,
        trace.episodic_mass,
        trace.shadow_scheduled_mass,
    ):
        assert array.shape == (2, 4)
        assert array.dtype == np.float32
        assert array.flags.c_contiguous
    np.testing.assert_array_equal(
        trace.base_attention_output.reshape(-1),
        np.arange(32, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        trace.regular_component.reshape(-1),
        np.arange(32, dtype=np.float32) + 100.0,
    )
    np.testing.assert_array_equal(
        trace.episodic_component.reshape(-1),
        np.arange(32, dtype=np.float32) - 100.0,
    )
    np.testing.assert_array_equal(
        trace.regular_mass.reshape(-1),
        np.arange(8, dtype=np.float32) + 200.0,
    )
    np.testing.assert_array_equal(
        trace.episodic_mass.reshape(-1),
        np.arange(8, dtype=np.float32) + 300.0,
    )
    np.testing.assert_array_equal(
        trace.shadow_scheduled_mass.reshape(-1),
        np.arange(8, dtype=np.float32) + 400.0,
    )

    runtime.reset()
    with pytest.raises(OLMoENativeRuntimeError):
        runtime.last_episodic_mass_trace()
    runtime.close()


def test_episodic_mass_trace_capability_requires_symbol_and_shadow_handle(
    tmp_path,
    monkeypatch,
):
    head_mask = (
        (True, True, True, True),
        (True, True, True, True),
    )
    required = {
        "episodic_policy": {"slots": 32, "span_size": 8},
        "episodic_head_mask": head_mask,
        "episodic_logit_bias": 0.0,
    }
    shadow_runtime, _library = _mock_token_runtime(
        tmp_path,
        monkeypatch,
        episodic_mass_trace=False,
        shadow_attention_policy={
            "local_window": 128,
            "older_candidates": 8,
            "older_top_k": 4,
            "sink_tokens": 2,
        },
        **required,
    )
    assert shadow_runtime.shadow_trace_available is True
    assert shadow_runtime.episodic_mass_trace_available is False
    with pytest.raises(
        OLMoENativeRuntimeError,
        match="no episodic mass trace for this handle",
    ):
        shadow_runtime.last_episodic_mass_trace()
    shadow_runtime.close()

    additive_runtime, _library = _mock_token_runtime(
        tmp_path,
        monkeypatch,
        **required,
    )
    assert additive_runtime.episodic_open_abi == "v2"
    assert additive_runtime.episodic_mass_trace_available is False
    with pytest.raises(
        OLMoENativeRuntimeError,
        match="no episodic mass trace for this handle",
    ):
        additive_runtime.last_episodic_mass_trace()
    additive_runtime.close()


def test_mocked_episodic_slot_trace_binding_copies_typed_layer_major_arrays(
    tmp_path,
    monkeypatch,
):
    head_mask = (
        (True, True, True, True),
        (True, True, True, True),
    )
    runtime, library = _mock_token_runtime(
        tmp_path,
        monkeypatch,
        episodic_policy={"slots": 32, "span_size": 8},
        episodic_head_mask=head_mask,
        episodic_logit_bias=0.0,
        shadow_attention_policy={
            "local_window": 128,
            "older_candidates": 8,
            "older_top_k": 4,
            "sink_tokens": 2,
        },
    )
    assert runtime.episodic_slot_trace_available is True
    binding = library.engram_olmoe_token_copy_last_episodic_slot_trace_v1
    assert len(binding.argtypes) == 7
    for pointer_index in (1, 3):
        assert binding.argtypes[pointer_index] == ctypes.POINTER(ctypes.c_float)
        assert binding.argtypes[pointer_index + 1] == ctypes.c_size_t

    with pytest.raises(OLMoENativeRuntimeError):
        runtime.last_episodic_slot_trace()
    runtime.forward_episodic([1], [0], [-1])
    with pytest.raises(OLMoENativeRuntimeError):
        runtime.last_episodic_slot_trace()
    runtime.forward_episodic([2], [-1], [0])
    trace = runtime.last_episodic_slot_trace()
    assert isinstance(trace, olmoe_native_runtime.OLMoENativeEpisodicSlotTrace)
    assert trace.slot_mass.shape == (2, 4, 8)
    assert trace.slot_values.shape == (2, 4, 8, 4)
    assert trace.slot_mass.dtype == np.float32
    assert trace.slot_values.dtype == np.float32
    assert trace.slot_mass.flags.c_contiguous
    assert trace.slot_values.flags.c_contiguous
    np.testing.assert_array_equal(
        trace.slot_mass.reshape(-1),
        np.arange(64, dtype=np.float32) + 500.0,
    )
    np.testing.assert_array_equal(
        trace.slot_values.reshape(-1),
        np.arange(256, dtype=np.float32) + 600.0,
    )

    runtime.reset()
    with pytest.raises(OLMoENativeRuntimeError):
        runtime.last_episodic_slot_trace()
    runtime.close()


def test_episodic_slot_trace_capability_requires_complete_trace_symbols_and_handle(
    tmp_path,
    monkeypatch,
):
    head_mask = (
        (True, True, True, True),
        (True, True, True, True),
    )
    required = {
        "episodic_policy": {"slots": 32, "span_size": 8},
        "episodic_head_mask": head_mask,
        "episodic_logit_bias": 0.0,
    }
    for missing_capability in (
        {"episodic_slot_trace": False},
        {"episodic_mass_trace": False},
    ):
        runtime, _library = _mock_token_runtime(
            tmp_path,
            monkeypatch,
            shadow_attention_policy={
                "local_window": 128,
                "older_candidates": 8,
                "older_top_k": 4,
                "sink_tokens": 2,
            },
            **missing_capability,
            **required,
        )
        assert runtime.episodic_slot_trace_available is False
        with pytest.raises(
            OLMoENativeRuntimeError,
            match="no episodic slot trace for this handle",
        ):
            runtime.last_episodic_slot_trace()
        runtime.close()

    additive_runtime, _library = _mock_token_runtime(
        tmp_path,
        monkeypatch,
        **required,
    )
    assert additive_runtime.episodic_slot_trace_available is False
    with pytest.raises(
        OLMoENativeRuntimeError,
        match="no episodic slot trace for this handle",
    ):
        additive_runtime.last_episodic_slot_trace()
    additive_runtime.close()


def test_mocked_regular_entry_trace_binding_copies_fixed_typed_layout(
    tmp_path,
    monkeypatch,
):
    head_mask = (
        (True, True, True, True),
        (True, True, True, True),
    )
    runtime, library = _mock_token_runtime(
        tmp_path,
        monkeypatch,
        episodic_policy={"slots": 32, "span_size": 8},
        episodic_head_mask=head_mask,
        episodic_logit_bias=0.0,
        shadow_attention_policy={
            "local_window": 128,
            "older_candidates": 8,
            "older_top_k": 4,
            "sink_tokens": 2,
        },
    )
    assert runtime.regular_entry_trace_available is True
    binding = library.engram_olmoe_token_copy_last_regular_entry_trace_v1
    assert len(binding.argtypes) == 11
    assert binding.argtypes[1] == ctypes.POINTER(ctypes.c_float)
    assert binding.argtypes[3] == ctypes.POINTER(ctypes.c_float)
    assert binding.argtypes[5] == ctypes.POINTER(ctypes.c_uint8)
    assert binding.argtypes[7] == ctypes.POINTER(ctypes.c_uint64)
    for count_index in (2, 4, 6, 8):
        assert binding.argtypes[count_index] == ctypes.c_size_t

    with pytest.raises(OLMoENativeRuntimeError):
        runtime.last_regular_entry_trace()
    runtime.forward_episodic([1], [0], [-1])
    with pytest.raises(OLMoENativeRuntimeError):
        runtime.last_regular_entry_trace()
    runtime.forward_episodic([2], [-1], [0])
    trace = runtime.last_regular_entry_trace()
    assert isinstance(trace, olmoe_native_runtime.OLMoENativeRegularEntryTrace)
    entry_shape = (2, 4, 20)
    assert trace.entry_mass.shape == entry_shape
    assert trace.entry_values.shape == (*entry_shape, 4)
    assert trace.valid_kind.shape == entry_shape
    assert trace.positions.shape == entry_shape
    assert trace.entry_mass.dtype == np.float32
    assert trace.entry_values.dtype == np.float32
    assert trace.valid_kind.dtype == np.uint8
    assert trace.positions.dtype == np.uint64
    assert all(
        array.flags.c_contiguous
        for array in (
            trace.entry_mass,
            trace.entry_values,
            trace.valid_kind,
            trace.positions,
        )
    )
    np.testing.assert_array_equal(
        trace.entry_mass.reshape(-1),
        np.arange(160, dtype=np.float32) + 700.0,
    )
    np.testing.assert_array_equal(
        trace.entry_values.reshape(-1),
        np.arange(640, dtype=np.float32) + 800.0,
    )
    np.testing.assert_array_equal(
        trace.valid_kind.reshape(-1),
        np.arange(160, dtype=np.uint8) % np.uint8(3),
    )
    np.testing.assert_array_equal(
        trace.positions.reshape(-1),
        np.arange(160, dtype=np.uint64),
    )

    runtime.reset()
    with pytest.raises(OLMoENativeRuntimeError):
        runtime.last_regular_entry_trace()
    runtime.close()


@pytest.mark.parametrize(
    "overrides",
    (
        {"regular_entry_trace": False},
        {"episodic_slot_trace": False},
        {"episodic_mass_trace": False},
        {"local_window": 15},
        {"older_top_k": 3},
    ),
)
def test_regular_entry_trace_capability_fails_closed(
    tmp_path,
    monkeypatch,
    overrides,
):
    runtime_options = {
        "episodic_policy": {"slots": 32, "span_size": 8},
        "episodic_head_mask": np.ones((2, 4), dtype=np.uint8),
        "episodic_logit_bias": 0.0,
        "shadow_attention_policy": {
            "local_window": 128,
            "older_candidates": 8,
            "older_top_k": 4,
            "sink_tokens": 2,
        },
    }
    library_options = {}
    for name, value in overrides.items():
        if name.endswith("_trace"):
            library_options[name] = value
        else:
            runtime_options[name] = value
    runtime, _library = _mock_token_runtime(
        tmp_path,
        monkeypatch,
        **library_options,
        **runtime_options,
    )
    assert runtime.regular_entry_trace_available is False
    with pytest.raises(
        OLMoENativeRuntimeError,
        match="no regular-entry trace for this handle",
    ):
        runtime.last_regular_entry_trace()
    runtime.close()


def test_episodic_policy_is_mutually_exclusive_with_structured_attention(
    tmp_path,
    monkeypatch,
):
    policy = {
        "local_window": 3,
        "older_candidates": 2,
        "older_top_k": 1,
        "sink_tokens": 1,
    }
    for structured in (
        {"attention_policies": [policy, policy]},
        {
            "attention_head_policies": [
                [policy] * 4,
                [policy] * 4,
            ]
        },
    ):
        with pytest.raises(
            ValueError,
            match="episodic and structured attention policies",
        ):
            _mock_token_runtime(
                tmp_path,
                monkeypatch,
                episodic_policy={"slots": 32, "span_size": 8},
                **structured,
            )


def test_mocked_episodic_forward_validation_and_missing_abi(
    tmp_path,
    monkeypatch,
):
    runtime, library = _mock_token_runtime(
        tmp_path,
        monkeypatch,
        episodic_policy={"slots": 32, "span_size": 8},
    )
    invalid = (
        ([], [], [], "must not be empty"),
        ([1, 2], [0], [-1, -1], "lengths must be equal"),
        ([1], [0], [-1, 0], "lengths must be equal"),
        ([1], [0.0], [-1], "must contain integers"),
        ([1], [-2], [-1], "write slots are outside"),
        ([1], [32], [-1], "write slots are outside"),
        ([1], [-1], [-2], "read spans are outside"),
        ([1], [-1], [4], "read spans are outside"),
    )
    for tokens, writes, reads, message in invalid:
        with pytest.raises(ValueError, match=message):
            runtime.forward_episodic(
                tokens,
                write_slots=writes,
                read_spans=reads,
            )
    assert library.episodic_forward_calls == 0

    scalar, _scalar_library = _mock_token_runtime(
        tmp_path,
        monkeypatch,
    )
    with pytest.raises(
        OLMoENativeRuntimeError,
        match="not opened with an episodic policy",
    ):
        scalar.forward_episodic(
            [1],
            write_slots=[-1],
            read_spans=[-1],
        )

    with pytest.raises(
        OLMoENativeRuntimeError,
        match="no episodic-memory ABI",
    ):
        _mock_token_runtime(
            tmp_path,
            monkeypatch,
            episodic=False,
            episodic_policy={"slots": 32, "span_size": 8},
        )


def _prompt_reference(model, q7_path, tokens, *, diagnostics=False):
    base_names = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    for layer in range(2):
        base = f"model.layers.{layer}"
        base_names.extend(
            [
                f"{base}.input_layernorm.weight",
                f"{base}.post_attention_layernorm.weight",
                f"{base}.self_attn.q_proj.weight",
                f"{base}.self_attn.k_proj.weight",
                f"{base}.self_attn.v_proj.weight",
                f"{base}.self_attn.o_proj.weight",
                f"{base}.self_attn.q_norm.weight",
                f"{base}.self_attn.k_norm.weight",
            ]
        )
    tensors = {
        name: _bf16(value)
        for name, value in load_local_named_tensors(model, base_names).items()
    }
    hidden = tensors["model.embed_tokens.weight"][tokens].copy()
    with LoadedOLMoEQ7Artifact(q7_path) as q7:
        for layer in range(2):
            base = f"model.layers.{layer}"
            normalized = np.stack(
                [
                    _norm(row, tensors[f"{base}.input_layernorm.weight"])
                    for row in hidden
                ]
            )
            query = normalized @ tensors[f"{base}.self_attn.q_proj.weight"].T
            key = normalized @ tensors[f"{base}.self_attn.k_proj.weight"].T
            value = normalized @ tensors[f"{base}.self_attn.v_proj.weight"].T
            query = np.stack(
                [
                    _norm(row, tensors[f"{base}.self_attn.q_norm.weight"])
                    for row in query
                ]
            )
            key = np.stack(
                [_norm(row, tensors[f"{base}.self_attn.k_norm.weight"]) for row in key]
            )
            query = np.stack(
                [_rope(row, 4, position) for position, row in enumerate(query)]
            )
            key = np.stack(
                [_rope(row, 4, position) for position, row in enumerate(key)]
            )
            values = value.reshape(len(tokens), 4, 4)
            attention_rows = []
            for position in range(len(tokens)):
                heads = []
                for head in range(4):
                    scores = (
                        key[: position + 1, head] @ query[position, head] / np.sqrt(4.0)
                    )
                    probabilities = np.exp(scores - scores.max())
                    probabilities /= probabilities.sum()
                    heads.append(probabilities @ values[: position + 1, head])
                attention_rows.append(np.concatenate(heads))
            attention = (
                np.stack(attention_rows) @ tensors[f"{base}.self_attn.o_proj.weight"].T
            )
            hidden += attention
            semantic_input = np.stack(
                [
                    _norm(
                        row,
                        tensors[f"{base}.post_attention_layernorm.weight"],
                    )
                    for row in hidden
                ]
            )
            router = q7.router(layer)
            semantic = np.zeros_like(hidden)
            for row, state in enumerate(semantic_input):
                logits = state @ router.T
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                selected = np.argsort(-probabilities, kind="stable")[:2]
                for expert in selected:
                    weights = q7.expert(layer, int(expert))
                    gate = weights["gate"] @ state
                    activation = (gate / (1.0 + np.exp(-gate))) * (
                        weights["up"] @ state
                    )
                    semantic[row] += probabilities[expert] * (
                        weights["down"] @ activation
                    )
            hidden += semantic
    final = _norm(hidden[-1], tensors["model.norm.weight"])
    logits = tensors["lm_head.weight"] @ final
    next_token = int(np.argmax(logits))
    if diagnostics:
        return next_token, final, logits
    return next_token


def test_native_olmoe_token_step_matches_single_position_reference(tmp_path):
    library = Path("build/libengram_olmoe_token_runtime.so")
    if not library.is_file():
        pytest.skip("native OLMoE token runtime has not been built")
    model = create_tiny_olmoe_fixture(tmp_path / "model")
    q7 = repack_olmoe_q7_model(model, tmp_path / "model.q7", group_size=8)
    non_mlp = tmp_path / "non_mlp.safetensors"
    report = repack_olmoe_non_mlp_weights(model, non_mlp)
    expected, expected_hidden, expected_logits = _prompt_reference(
        model,
        q7,
        [1],
        diagnostics=True,
    )
    expected_prompt = _prompt_reference(model, q7, [1, 2, 3])

    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
    ) as runtime:
        first = runtime.forward([1])
        assert first.next_token == expected
        assert runtime.position == 1
        assert first.metrics["positions_processed"] == 1
        assert first.metrics["q7_scheduled_bytes"] > 0
        assert first.metrics["attention_state_bytes"] > 0
        diagnostic_hidden, diagnostic_logits = runtime.last_diagnostics()
        np.testing.assert_allclose(
            diagnostic_hidden,
            expected_hidden,
            rtol=2e-5,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            diagnostic_logits,
            expected_logits,
            rtol=2e-5,
            atol=2e-6,
        )
        runtime.reset()
        with pytest.raises(OLMoENativeRuntimeError, match="diagnostic storage"):
            runtime.last_diagnostics()
        replay = runtime.forward([1])
        assert replay.next_token == first.next_token
        assert runtime.position == 1
        runtime.reset()
        prompt = runtime.forward([1, 2, 3])
        assert prompt.next_token == expected_prompt
        assert runtime.position == 3
        runtime.reset()
        runtime.forward([1, 2])
        incremental = runtime.forward([3])
        assert incremental.next_token == prompt.next_token
        assert runtime.position == 3
        runtime.reset()
        generated = runtime.generate([1], max_new_tokens=2)
        assert len(generated) == 2
        assert generated[0] == expected
        assert runtime.position == 2
        runtime.reset()
        for _position in range(17):
            sustained = runtime.forward([1])
        assert runtime.position == 17
        assert sustained.metrics["attention_eviction_events"] == 2
        assert sustained.metrics["attention_older_candidate_entries_scored"] == 8
        assert sustained.metrics["attention_older_selected_entries"] == 8
        assert sustained.metrics["attention_sink_insertions"] == 8
        assert sustained.metrics["attention_heavy_hitter_updates"] == 0
        assert sustained.metrics["attention_logical_read_bytes"] > 0
        assert sustained.metrics["attention_scratch_bytes"] > 0
        runtime.reset()
        reset = runtime.forward([1])
        assert reset.metrics["attention_eviction_events"] == 0
        assert reset.metrics["attention_older_candidate_entries_scored"] == 0
        assert reset.metrics["attention_sink_insertions"] == 0

    assert report["tensor_count"] == 19
    assert report["file_bytes"] == non_mlp.stat().st_size


def test_native_head_gated_episodic_parity_counters_and_reset(tmp_path):
    library = Path("build/libengram_olmoe_token_runtime.so")
    if not library.is_file():
        pytest.skip("native OLMoE token runtime has not been built")
    model = create_tiny_olmoe_fixture(tmp_path / "model")
    q7 = repack_olmoe_q7_model(model, tmp_path / "model.q7", group_size=8)
    non_mlp = tmp_path / "non_mlp.safetensors"
    repack_olmoe_non_mlp_weights(model, non_mlp)
    scalar_policy = {
        "local_window": 3,
        "older_candidates": 2,
        "older_top_k": 1,
        "sink_tokens": 1,
    }
    episodic_policy = {"slots": 4, "span_size": 2}
    tokens = [1, 2, 1, 2]
    writes = [0, 1, -1, -1]
    reads = [-1, -1, 0, 0]
    all_ones = np.ones((2, 4), dtype=bool)
    mixed = np.asarray(
        [[1, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.uint8,
    )

    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
        episodic_policy=episodic_policy,
        **scalar_policy,
    ) as legacy:
        legacy_result = legacy.forward_episodic(tokens, writes, reads)
        legacy_hidden, legacy_logits = legacy.last_diagnostics()
    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
        episodic_policy=episodic_policy,
        episodic_head_mask=all_ones,
        **scalar_policy,
    ) as explicit_all:
        explicit_result = explicit_all.forward_episodic(
            tokens,
            writes,
            reads,
        )
        explicit_hidden, explicit_logits = explicit_all.last_diagnostics()
        assert explicit_all.episodic_head_mask == tuple(
            tuple(True for _head in range(4)) for _layer in range(2)
        )
        assert explicit_all.episodic_open_abi == "v1"
    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
        episodic_policy=episodic_policy,
        episodic_head_mask=all_ones,
        episodic_logit_bias=0.0,
        **scalar_policy,
    ) as explicit_zero:
        explicit_zero_result = explicit_zero.forward_episodic(
            tokens,
            writes,
            reads,
        )
        explicit_zero_hidden, explicit_zero_logits = explicit_zero.last_diagnostics()
        assert explicit_zero.episodic_open_abi == "v2"
        assert explicit_zero.episodic_logit_bias == 0.0
    assert explicit_result.next_token == legacy_result.next_token
    assert {
        name: value
        for name, value in explicit_result.metrics.items()
        if name not in {"elapsed_ns", "q7_elapsed_ns"}
    } == {
        name: value
        for name, value in legacy_result.metrics.items()
        if name not in {"elapsed_ns", "q7_elapsed_ns"}
    }
    np.testing.assert_array_equal(explicit_hidden, legacy_hidden)
    np.testing.assert_array_equal(explicit_logits, legacy_logits)
    assert explicit_zero_result.next_token == explicit_result.next_token
    assert {
        name: value
        for name, value in explicit_zero_result.metrics.items()
        if name not in {"elapsed_ns", "q7_elapsed_ns"}
    } == {
        name: value
        for name, value in explicit_result.metrics.items()
        if name not in {"elapsed_ns", "q7_elapsed_ns"}
    }
    np.testing.assert_array_equal(explicit_zero_hidden, explicit_hidden)
    np.testing.assert_array_equal(explicit_zero_logits, explicit_logits)

    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
        episodic_policy=episodic_policy,
        episodic_head_mask=all_ones,
        episodic_logit_bias=1.25,
        **scalar_policy,
    ) as biased:
        biased_result = biased.forward_episodic(tokens, writes, reads)
        biased_hidden, biased_logits = biased.last_diagnostics()
        assert biased.episodic_open_abi == "v2"
    assert not (
        np.array_equal(biased_hidden, explicit_zero_hidden)
        and np.array_equal(biased_logits, explicit_zero_logits)
    )
    stable_metrics = {
        "positions_processed",
        "attention_weight_bytes",
        "attention_state_bytes",
        "attention_scratch_bytes",
        "q7_scheduled_bytes",
        "episodic_slots_written",
        "episodic_read_events",
        "episodic_active_slots",
        "episodic_entries_read",
        "episodic_write_bytes",
        "episodic_key_read_bytes",
        "episodic_value_read_bytes",
        "episodic_state_bytes",
        "episodic_scratch_bytes",
    }
    assert {name: biased_result.metrics[name] for name in stable_metrics} == {
        name: explicit_zero_result.metrics[name] for name in stable_metrics
    }

    base_state, base_scratch = _attention_capacity_bytes(
        query_heads=4,
        key_value_heads=4,
        head_dimension=4,
        local_window=3,
        older_candidates=2,
        older_top_k=1,
    )
    episodic_state_per_active_layer = 4 * (2 * 16 * 2 + 8)
    episodic_scratch_per_active_layer = 2 * 2 * 4
    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
        episodic_policy=episodic_policy,
        episodic_head_mask=mixed,
        **scalar_policy,
    ) as gated:
        result = gated.forward_episodic(tokens, writes, reads)
        expected = {
            "episodic_slots_written": 2,
            "episodic_read_events": 2,
            "episodic_active_slots": 2,
            "episodic_entries_read": 4,
            "episodic_write_bytes": 128,
            "episodic_key_read_bytes": 32,
            "episodic_value_read_bytes": 32,
            "episodic_state_bytes": (2 * base_state + episodic_state_per_active_layer),
            "episodic_scratch_bytes": (
                2 * base_scratch + episodic_scratch_per_active_layer
            ),
        }
        assert {name: result.metrics[name] for name in expected} == expected
        assert (
            0
            <= result.metrics["episodic_duplicate_older_entries_suppressed"]
            <= result.metrics["episodic_entries_read"]
        )
        gated.reset()
        reset = gated.forward_episodic([1], [0], [-1])
        assert reset.metrics["episodic_slots_written"] == 1
        assert reset.metrics["episodic_read_events"] == 0
        assert reset.metrics["episodic_active_slots"] == 1
        assert reset.metrics["episodic_entries_read"] == 0
        assert reset.metrics["episodic_write_bytes"] == 64
        assert reset.metrics["episodic_key_read_bytes"] == 0
        assert reset.metrics["episodic_value_read_bytes"] == 0


def test_native_olmoe_layered_attention_matches_scalar_and_sums_state(
    tmp_path,
):
    library = Path("build/libengram_olmoe_token_runtime.so")
    if not library.is_file():
        pytest.skip("native OLMoE token runtime has not been built")
    model = create_tiny_olmoe_fixture(tmp_path / "model")
    q7 = repack_olmoe_q7_model(model, tmp_path / "model.q7", group_size=8)
    non_mlp = tmp_path / "non_mlp.safetensors"
    repack_olmoe_non_mlp_weights(model, non_mlp)
    scalar_policy = {
        "local_window": 3,
        "older_candidates": 2,
        "older_top_k": 1,
        "sink_tokens": 1,
    }
    tokens = [1, 2, 3, 4]
    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
        **scalar_policy,
    ) as scalar:
        scalar_result = scalar.forward(tokens)
        scalar_hidden, scalar_logits = scalar.last_diagnostics()
    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
        attention_policies=[scalar_policy, scalar_policy],
    ) as layered:
        layered_result = layered.forward(tokens)
        layered_hidden, layered_logits = layered.last_diagnostics()
    assert layered_result.next_token == scalar_result.next_token
    assert {
        name: value
        for name, value in layered_result.metrics.items()
        if name not in {"elapsed_ns", "q7_elapsed_ns"}
    } == {
        name: value
        for name, value in scalar_result.metrics.items()
        if name not in {"elapsed_ns", "q7_elapsed_ns"}
    }
    np.testing.assert_array_equal(layered_hidden, scalar_hidden)
    np.testing.assert_array_equal(layered_logits, scalar_logits)

    heterogeneous = [
        {
            "local_window": 1,
            "older_candidates": 2,
            "older_top_k": 1,
            "sink_tokens": 1,
        },
        {
            "local_window": 3,
            "older_candidates": 1,
            "older_top_k": 1,
            "sink_tokens": 1,
        },
    ]
    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
        attention_policies=heterogeneous,
    ) as layered:
        result = layered.forward(tokens)
        assert layered.position == 4
        assert layered.attention_policies == tuple(heterogeneous)
        assert result.metrics["attention_state_bytes"] == 1_148
        assert result.metrics["attention_scratch_bytes"] == 80
        assert result.metrics["attention_eviction_events"] == 4
        assert result.metrics["attention_older_candidate_entries_scored"] == 24
        assert result.metrics["attention_older_selected_entries"] == 16
        assert result.metrics["attention_sink_insertions"] == 8
        layered.reset()
        reset = layered.forward([1])
        assert reset.metrics["attention_eviction_events"] == 0
        assert reset.metrics["attention_older_candidate_entries_scored"] == 0

    with pytest.raises(ValueError, match="cannot be combined"):
        OLMoENativeTokenRuntime(
            model / "config.json",
            non_mlp,
            q7,
            library,
            attention_policies=heterogeneous,
            local_window=3,
        )


def test_native_olmoe_headwise_attention_matches_layered_and_reports_counters(
    tmp_path,
):
    library = Path("build/libengram_olmoe_token_runtime.so")
    if not library.is_file():
        pytest.skip("native OLMoE token runtime has not been built")
    native = ctypes.CDLL(str(library.resolve()))
    if not hasattr(native, "engram_olmoe_token_open_headwise_v1"):
        pytest.skip("native OLMoE headwise-attention ABI has not been built")
    model = create_tiny_olmoe_fixture(tmp_path / "model")
    q7 = repack_olmoe_q7_model(model, tmp_path / "model.q7", group_size=8)
    non_mlp = tmp_path / "non_mlp.safetensors"
    repack_olmoe_non_mlp_weights(model, non_mlp)
    homogeneous = {
        "local_window": 3,
        "older_candidates": 2,
        "older_top_k": 1,
        "sink_tokens": 1,
    }
    tokens = [1, 2, 3, 4]
    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
        attention_policies=[homogeneous, homogeneous],
    ) as layered:
        layered_result = layered.forward(tokens)
        layered_hidden, layered_logits = layered.last_diagnostics()
    nested_homogeneous = tuple(
        tuple(dict(homogeneous) for _head in range(4)) for _layer in range(2)
    )
    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
        attention_head_policies=nested_homogeneous,
    ) as headwise:
        headwise_result = headwise.forward(tokens)
        headwise_hidden, headwise_logits = headwise.last_diagnostics()
        assert headwise.attention_policies is None
        assert headwise.attention_head_policies == nested_homogeneous
    assert headwise_result.next_token == layered_result.next_token
    for name in (
        "positions_processed",
        "attention_weight_bytes",
        "q7_scheduled_bytes",
        "attention_logical_read_bytes",
        "attention_older_candidate_entries_scored",
        "attention_older_selected_entries",
        "attention_sink_insertions",
        "attention_heavy_hitter_updates",
    ):
        assert headwise_result.metrics[name] == layered_result.metrics[name]
    assert headwise_result.metrics["attention_eviction_events"] == (
        layered_result.metrics["attention_eviction_events"] * 4
    )
    layered_per_layer = _attention_capacity_bytes(
        query_heads=4,
        key_value_heads=4,
        head_dimension=4,
        local_window=3,
        older_candidates=2,
        older_top_k=1,
    )
    headwise_per_head = _attention_capacity_bytes(
        query_heads=1,
        key_value_heads=1,
        head_dimension=4,
        local_window=3,
        older_candidates=2,
        older_top_k=1,
    )
    assert layered_result.metrics["attention_state_bytes"] == (2 * layered_per_layer[0])
    assert layered_result.metrics["attention_scratch_bytes"] == (
        2 * layered_per_layer[1]
    )
    assert headwise_result.metrics["attention_state_bytes"] == (
        2 * 4 * headwise_per_head[0]
    )
    assert headwise_result.metrics["attention_scratch_bytes"] == (
        2 * 4 * headwise_per_head[1]
    )
    np.testing.assert_array_equal(headwise_hidden, layered_hidden)
    np.testing.assert_array_equal(headwise_logits, layered_logits)

    compact = {
        "local_window": 1,
        "older_candidates": 1,
        "older_top_k": 1,
        "sink_tokens": 0,
    }
    mixed = (
        (compact, homogeneous, compact, homogeneous),
        (homogeneous, compact, homogeneous, compact),
    )
    with OLMoENativeTokenRuntime(
        model / "config.json",
        non_mlp,
        q7,
        library,
        threads=2,
        attention_head_policies=mixed,
    ) as headwise:
        result = headwise.forward(tokens)
        assert headwise.position == len(tokens)
        assert headwise.attention_head_policies == mixed
        compact_per_head = _attention_capacity_bytes(
            query_heads=1,
            key_value_heads=1,
            head_dimension=4,
            local_window=1,
            older_candidates=1,
            older_top_k=1,
        )
        assert result.metrics["attention_state_bytes"] == (
            4 * compact_per_head[0] + 4 * headwise_per_head[0]
        )
        assert result.metrics["attention_scratch_bytes"] == (
            4 * compact_per_head[1] + 4 * headwise_per_head[1]
        )
        assert result.metrics["attention_eviction_events"] == 16
        assert result.metrics["attention_older_candidate_entries_scored"] == 16
        assert result.metrics["attention_older_selected_entries"] == 16
        assert result.metrics["attention_sink_insertions"] == 4
        assert result.metrics["attention_heavy_hitter_updates"] == 4
        assert result.metrics["attention_logical_read_bytes"] == 2_176
        headwise.reset()
        reset = headwise.forward([1])
        assert reset.metrics["attention_eviction_events"] == 0
        assert reset.metrics["attention_older_candidate_entries_scored"] == 0
        assert reset.metrics["attention_sink_insertions"] == 0

    with pytest.raises(ValueError, match="cannot be combined"):
        OLMoENativeTokenRuntime(
            model / "config.json",
            non_mlp,
            q7,
            library,
            attention_policies=[homogeneous, homogeneous],
            attention_head_policies=nested_homogeneous,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        OLMoENativeTokenRuntime(
            model / "config.json",
            non_mlp,
            q7,
            library,
            local_window=3,
            attention_head_policies=nested_homogeneous,
        )


def test_authenticated_native_olmoe_package_generation_and_tamper_rejection(
    tmp_path,
):
    library = Path("build/libengram_olmoe_token_runtime.so")
    if not library.is_file():
        pytest.skip("native OLMoE token runtime has not been built")
    model = create_tiny_olmoe_fixture(tmp_path / "model")
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.save(str(model / "tokenizer.json"))
    q7 = repack_olmoe_q7_model(model, tmp_path / "model.q7", group_size=8)
    non_mlp = tmp_path / "non_mlp.safetensors"
    repack_olmoe_non_mlp_weights(model, non_mlp)
    package = tmp_path / "package"
    compiled = compile_olmoe_native_package(
        model,
        q7,
        non_mlp,
        package,
        kernel_threads=2,
    )
    manifest_hash = compiled["manifest_sha256"]

    manifest = validate_olmoe_native_package(
        package,
        expected_manifest_sha256=manifest_hash,
    )
    assert manifest["does_not_require_transformers"]
    assert manifest["runtime"]["device"] == "cpu"
    expected = _prompt_reference(model, q7, [1])
    with OLMoENativePackageRuntime(
        package,
        manifest_sha256=manifest_hash,
        library=library,
    ) as runtime:
        result = runtime.generate("hello", max_new_tokens=1)
    assert result["prompt_token_ids"] == [1]
    assert result["generated_token_ids"] == [expected]

    tokenizer_path = package / "tokenizer" / "tokenizer.json"
    tokenizer_path.write_bytes(tokenizer_path.read_bytes() + b" ")
    with pytest.raises(OLMoENativePackageError, match="package file is invalid"):
        validate_olmoe_native_package(
            package,
            expected_manifest_sha256=manifest_hash,
        )
