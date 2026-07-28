"""Python owner for the transformer-shell-free native OLMoE token runtime."""

from __future__ import annotations

import ctypes
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from numpy.typing import ArrayLike

from engram.compiler.olmoe_native import (
    OLMOE_CONFIG_PATH,
    OLMOE_NON_MLP_PATH,
    OLMOE_Q7_PATH,
    OLMoENativePackageError,
    validate_olmoe_native_package,
)
from engram.utils import sha256_file


class OLMoENativeRuntimeError(RuntimeError):
    """Raised when the native OLMoE token runtime rejects an operation."""


class _Config(ctypes.Structure):
    _fields_ = [
        ("non_mlp_safetensors", ctypes.c_char_p),
        ("q7_artifact", ctypes.c_char_p),
        ("layers", ctypes.c_size_t),
        ("hidden_size", ctypes.c_size_t),
        ("query_heads", ctypes.c_size_t),
        ("key_value_heads", ctypes.c_size_t),
        ("head_dimension", ctypes.c_size_t),
        ("threads", ctypes.c_size_t),
        ("local_window", ctypes.c_size_t),
        ("older_candidates", ctypes.c_size_t),
        ("older_top_k", ctypes.c_size_t),
        ("sink_tokens", ctypes.c_size_t),
        ("rms_norm_epsilon", ctypes.c_float),
        ("rope_theta", ctypes.c_float),
    ]


class _Metrics(ctypes.Structure):
    _fields_ = [
        ("positions_processed", ctypes.c_uint64),
        ("attention_weight_bytes", ctypes.c_uint64),
        ("q7_scheduled_bytes", ctypes.c_uint64),
        ("q7_elapsed_ns", ctypes.c_uint64),
        ("attention_state_bytes", ctypes.c_uint64),
        ("elapsed_ns", ctypes.c_uint64),
    ]


class _AttentionPolicyV1(ctypes.Structure):
    _fields_ = [
        ("local_window", ctypes.c_size_t),
        ("older_candidates", ctypes.c_size_t),
        ("older_top_k", ctypes.c_size_t),
        ("sink_tokens", ctypes.c_size_t),
    ]


class _AttentionMetricsV1(ctypes.Structure):
    _fields_ = [
        ("attention_logical_read_bytes", ctypes.c_uint64),
        ("attention_state_bytes", ctypes.c_uint64),
        ("attention_scratch_bytes", ctypes.c_uint64),
        ("attention_eviction_events", ctypes.c_uint64),
        ("attention_older_candidate_entries_scored", ctypes.c_uint64),
        ("attention_older_selected_entries", ctypes.c_uint64),
        ("attention_sink_insertions", ctypes.c_uint64),
        ("attention_heavy_hitter_updates", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class OLMoENativeTokenResult:
    next_token: int
    metrics: dict[str, int]


def _configure(library: ctypes.CDLL) -> tuple[bool, bool, bool]:
    library.engram_olmoe_token_open.argtypes = [
        ctypes.POINTER(_Config),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.engram_olmoe_token_open.restype = ctypes.c_void_p
    library.engram_olmoe_token_close.argtypes = [ctypes.c_void_p]
    library.engram_olmoe_token_reset.argtypes = [ctypes.c_void_p]
    library.engram_olmoe_token_position.argtypes = [ctypes.c_void_p]
    library.engram_olmoe_token_position.restype = ctypes.c_size_t
    library.engram_olmoe_token_vocabulary_size.argtypes = [ctypes.c_void_p]
    library.engram_olmoe_token_vocabulary_size.restype = ctypes.c_size_t
    library.engram_olmoe_token_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(_Metrics),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.engram_olmoe_token_forward.restype = ctypes.c_int
    library.engram_olmoe_token_copy_last_diagnostics.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.engram_olmoe_token_copy_last_diagnostics.restype = ctypes.c_int
    has_attention_metrics = hasattr(
        library, "engram_olmoe_token_copy_attention_metrics_v1"
    )
    if has_attention_metrics:
        library.engram_olmoe_token_copy_attention_metrics_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_AttentionMetricsV1),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.engram_olmoe_token_copy_attention_metrics_v1.restype = ctypes.c_int
    has_layered_open = hasattr(library, "engram_olmoe_token_open_layered_v1")
    if has_layered_open:
        library.engram_olmoe_token_open_layered_v1.argtypes = [
            ctypes.POINTER(_Config),
            ctypes.POINTER(_AttentionPolicyV1),
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.engram_olmoe_token_open_layered_v1.restype = ctypes.c_void_p
    has_headwise_open = hasattr(
        library, "engram_olmoe_token_open_headwise_v1"
    )
    if has_headwise_open:
        library.engram_olmoe_token_open_headwise_v1.argtypes = [
            ctypes.POINTER(_Config),
            ctypes.POINTER(_AttentionPolicyV1),
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.engram_olmoe_token_open_headwise_v1.restype = ctypes.c_void_p
    return has_attention_metrics, has_layered_open, has_headwise_open


def _normalize_attention_policy(
    policy: Mapping[str, int],
    *,
    coordinate: str,
) -> dict[str, int]:
    names = {
        "local_window",
        "older_candidates",
        "older_top_k",
        "sink_tokens",
    }
    if not isinstance(policy, Mapping) or set(policy) != names:
        raise ValueError(
            f"attention policy for {coordinate} has invalid fields"
        )
    if any(
        isinstance(policy[name], bool) or not isinstance(policy[name], int)
        for name in names
    ):
        raise ValueError(
            f"attention policy for {coordinate} must contain integers"
        )
    value = {name: int(policy[name]) for name in names}
    if (
        value["local_window"] <= 0
        or value["older_candidates"] <= 0
        or value["older_top_k"] <= 0
        or value["older_top_k"] > value["older_candidates"]
        or value["sink_tokens"] < 0
        or value["sink_tokens"] > value["older_candidates"]
    ):
        raise ValueError(
            f"attention policy for {coordinate} is inconsistent"
        )
    return value


def _validate_attention_policies(
    policies: Sequence[Mapping[str, int]],
    *,
    layers: int,
) -> tuple[dict[str, int], ...]:
    if isinstance(policies, (str, bytes)) or len(policies) != layers:
        raise ValueError(
            "per-layer attention policy count must equal model layers"
        )
    return tuple(
        _normalize_attention_policy(policy, coordinate=f"layer {layer}")
        for layer, policy in enumerate(policies)
    )


def _validate_attention_head_policies(
    policies: Sequence[Sequence[Mapping[str, int]]],
    *,
    layers: int,
    query_heads: int,
) -> tuple[tuple[dict[str, int], ...], ...]:
    if (
        isinstance(policies, (str, bytes))
        or not isinstance(policies, Sequence)
        or len(policies) != layers
    ):
        raise ValueError(
            "per-head attention policy layer count must equal model layers"
        )
    normalized: list[tuple[dict[str, int], ...]] = []
    for layer, head_policies in enumerate(policies):
        if (
            isinstance(head_policies, (str, bytes))
            or not isinstance(head_policies, Sequence)
            or len(head_policies) != query_heads
        ):
            raise ValueError(
                "per-head attention policy count for layer "
                f"{layer} must equal model query heads"
            )
        normalized.append(
            tuple(
                _normalize_attention_policy(
                    policy,
                    coordinate=f"layer {layer}, head {head}",
                )
                for head, policy in enumerate(head_policies)
            )
        )
    return tuple(normalized)


class OLMoENativeTokenRuntime:
    def __init__(
        self,
        model_config: str | Path,
        non_mlp_safetensors: str | Path,
        q7_artifact: str | Path,
        library: str | Path,
        *,
        threads: int = 1,
        local_window: int | None = None,
        older_candidates: int | None = None,
        older_top_k: int | None = None,
        sink_tokens: int | None = None,
        attention_policies: Sequence[Mapping[str, int]] | None = None,
        attention_head_policies: (
            Sequence[Sequence[Mapping[str, int]]] | None
        ) = None,
    ):
        config_path = Path(model_config)
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        hidden = int(config_value["hidden_size"])
        heads = int(config_value["num_attention_heads"])
        kv_heads = int(config_value["num_key_value_heads"])
        if hidden % heads:
            raise ValueError("OLMoE hidden size is not divisible by heads")
        layers = int(config_value["num_hidden_layers"])
        self._library = ctypes.CDLL(str(Path(library).resolve()))
        (
            self._has_attention_metrics,
            has_layered_open,
            has_headwise_open,
        ) = _configure(self._library)
        non_mlp_bytes = str(Path(non_mlp_safetensors).resolve()).encode()
        q7_bytes = str(Path(q7_artifact).resolve()).encode()
        scalar_values = (
            local_window,
            older_candidates,
            older_top_k,
            sink_tokens,
        )
        if (
            attention_policies is not None
            and attention_head_policies is not None
        ):
            raise ValueError(
                "per-layer and per-head attention policies cannot be combined"
            )
        if (
            attention_policies is not None
            or attention_head_policies is not None
        ) and any(value is not None for value in scalar_values):
            raise ValueError(
                "structured attention policies cannot be combined with "
                "scalar attention overrides"
            )
        scalar_policy = {
            "local_window": 16 if local_window is None else int(local_window),
            "older_candidates": (
                8 if older_candidates is None else int(older_candidates)
            ),
            "older_top_k": 4 if older_top_k is None else int(older_top_k),
            "sink_tokens": 2 if sink_tokens is None else int(sink_tokens),
        }
        native_config = _Config(
            non_mlp_bytes,
            q7_bytes,
            layers,
            hidden,
            heads,
            kv_heads,
            hidden // heads,
            threads,
            scalar_policy["local_window"],
            scalar_policy["older_candidates"],
            scalar_policy["older_top_k"],
            scalar_policy["sink_tokens"],
            float(config_value["rms_norm_eps"]),
            float(config_value["rope_theta"]),
        )
        error = ctypes.create_string_buffer(1024)
        if attention_policies is None and attention_head_policies is None:
            self._handle = self._library.engram_olmoe_token_open(
                ctypes.byref(native_config), error, len(error)
            )
            self.attention_policies = tuple(
                dict(scalar_policy) for _layer in range(layers)
            )
            self.attention_head_policies = tuple(
                tuple(dict(scalar_policy) for _head in range(heads))
                for _layer in range(layers)
            )
        elif attention_policies is not None:
            policies = _validate_attention_policies(
                attention_policies,
                layers=layers,
            )
            if not has_layered_open:
                raise OLMoENativeRuntimeError(
                    "native OLMoE library has no layered-attention ABI"
                )
            native_policies = (_AttentionPolicyV1 * layers)(
                *(
                    _AttentionPolicyV1(
                        policy["local_window"],
                        policy["older_candidates"],
                        policy["older_top_k"],
                        policy["sink_tokens"],
                    )
                    for policy in policies
                )
            )
            self._handle = self._library.engram_olmoe_token_open_layered_v1(
                ctypes.byref(native_config),
                native_policies,
                layers,
                error,
                len(error),
            )
            self.attention_policies = policies
            self.attention_head_policies = tuple(
                tuple(dict(policy) for _head in range(heads))
                for policy in policies
            )
        else:
            assert attention_head_policies is not None
            head_policies = _validate_attention_head_policies(
                attention_head_policies,
                layers=layers,
                query_heads=heads,
            )
            if not has_headwise_open:
                raise OLMoENativeRuntimeError(
                    "native OLMoE library has no headwise-attention ABI"
                )
            flattened = tuple(
                policy
                for layer_policies in head_policies
                for policy in layer_policies
            )
            native_policies = (_AttentionPolicyV1 * len(flattened))(
                *(
                    _AttentionPolicyV1(
                        policy["local_window"],
                        policy["older_candidates"],
                        policy["older_top_k"],
                        policy["sink_tokens"],
                    )
                    for policy in flattened
                )
            )
            self._handle = self._library.engram_olmoe_token_open_headwise_v1(
                ctypes.byref(native_config),
                native_policies,
                len(flattened),
                error,
                len(error),
            )
            self.attention_policies = None
            self.attention_head_policies = head_policies
        if not self._handle:
            raise OLMoENativeRuntimeError(error.value.decode(errors="replace"))
        self.last_result: OLMoENativeTokenResult | None = None
        self.hidden_size = hidden
        self.vocabulary_size = int(
            self._library.engram_olmoe_token_vocabulary_size(self._handle)
        )

    @property
    def position(self) -> int:
        return int(self._library.engram_olmoe_token_position(self._handle))

    @property
    def attention_metrics_available(self) -> bool:
        return self._has_attention_metrics

    def forward(self, token_ids: ArrayLike) -> OLMoENativeTokenResult:
        tokens = np.ascontiguousarray(token_ids, dtype=np.int64).reshape(-1)
        if not tokens.size:
            raise ValueError("native OLMoE token input must not be empty")
        next_token = ctypes.c_int64()
        metrics = _Metrics()
        error = ctypes.create_string_buffer(1024)
        status = self._library.engram_olmoe_token_forward(
            self._handle,
            tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            tokens.size,
            ctypes.byref(next_token),
            ctypes.byref(metrics),
            error,
            len(error),
        )
        if status:
            raise OLMoENativeRuntimeError(error.value.decode(errors="replace"))
        metric_values = {
            name: int(getattr(metrics, name)) for name, _ctype in metrics._fields_
        }
        if self._has_attention_metrics:
            attention = _AttentionMetricsV1()
            status = self._library.engram_olmoe_token_copy_attention_metrics_v1(
                self._handle,
                ctypes.byref(attention),
                error,
                len(error),
            )
            if status:
                raise OLMoENativeRuntimeError(error.value.decode(errors="replace"))
            metric_values.update(
                {
                    name: int(getattr(attention, name))
                    for name, _ctype in attention._fields_
                }
            )
        result = OLMoENativeTokenResult(
            next_token=int(next_token.value),
            metrics=metric_values,
        )
        self.last_result = result
        return result

    def last_diagnostics(self) -> tuple[np.ndarray, np.ndarray]:
        """Copy the last normalized hidden state and full vocabulary logits."""

        hidden = np.empty(self.hidden_size, dtype=np.float32)
        logits = np.empty(self.vocabulary_size, dtype=np.float32)
        error = ctypes.create_string_buffer(1024)
        status = self._library.engram_olmoe_token_copy_last_diagnostics(
            self._handle,
            hidden.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            hidden.size,
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            logits.size,
            error,
            len(error),
        )
        if status:
            raise OLMoENativeRuntimeError(error.value.decode(errors="replace"))
        return hidden, logits

    def reset(self) -> None:
        self._library.engram_olmoe_token_reset(self._handle)

    def generate(
        self,
        prompt: ArrayLike,
        *,
        max_new_tokens: int,
        eos_token_ids: tuple[int, ...] = (),
    ) -> list[int]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        first = self.forward(prompt).next_token
        generated = [first]
        while len(generated) < max_new_tokens and generated[-1] not in eos_token_ids:
            generated.append(self.forward([generated[-1]]).next_token)
        return generated

    def close(self) -> None:
        if self._handle:
            self._library.engram_olmoe_token_close(self._handle)
            self._handle = None

    def __enter__(self) -> "OLMoENativeTokenRuntime":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _package_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise OLMoENativePackageError(f"unsafe authenticated path: {relative!r}")
    return root.joinpath(*pure.parts)


class OLMoENativePackageRuntime:
    """Authenticated package-only frontend for native OLMoE generation."""

    def __init__(
        self,
        package: str | Path,
        *,
        manifest_sha256: str,
        library: str | Path,
        threads: int | None = None,
    ):
        self.path = Path(os.path.abspath(os.fspath(Path(package).expanduser())))
        self.manifest_sha256 = manifest_sha256.lower()
        self.manifest = validate_olmoe_native_package(
            self.path, expected_manifest_sha256=self.manifest_sha256
        )
        runtime = self.manifest["runtime"]
        selected_threads = (
            int(runtime["kernel_threads"]) if threads is None else int(threads)
        )
        if selected_threads <= 0 or selected_threads > 256:
            raise OLMoENativePackageError("runtime thread override is invalid")
        tokenizer_path = _package_path(
            self.path,
            f"{self.manifest['tokenizer']['path']}/tokenizer.json",
        )
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise OLMoENativePackageError(
                "install engram-lm[conversion] for packaged tokenization"
            ) from exc
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        eos = self.tokenizer.token_to_id("<|endoftext|>")
        self.eos_token_ids = () if eos is None else (int(eos),)
        if sha256_file(self.path / "manifest.json") != self.manifest_sha256:
            raise OLMoENativePackageError(
                "native OLMoE manifest changed while loading tokenizer"
            )
        self.runtime = OLMoENativeTokenRuntime(
            _package_path(self.path, OLMOE_CONFIG_PATH.as_posix()),
            _package_path(self.path, OLMOE_NON_MLP_PATH.as_posix()),
            _package_path(self.path, OLMOE_Q7_PATH.as_posix()),
            library,
            threads=selected_threads,
            local_window=int(runtime["attention_policy"]["local_window"]),
            older_candidates=int(runtime["attention_policy"]["older_candidates"]),
            older_top_k=int(runtime["attention_policy"]["older_top_k"]),
            sink_tokens=int(runtime["attention_policy"]["sink_tokens"]),
        )

    def generate(self, prompt: str, *, max_new_tokens: int) -> dict[str, object]:
        token_ids = self.tokenizer.encode(prompt).ids
        if not token_ids:
            raise ValueError("packaged OLMoE prompt produced no tokens")
        generated = self.runtime.generate(
            token_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=self.eos_token_ids,
        )
        return {
            "prompt_token_ids": token_ids,
            "generated_token_ids": generated,
            "completion": self.tokenizer.decode(generated),
            "position": self.runtime.position,
            "metrics": (
                self.runtime.last_result.metrics
                if self.runtime.last_result is not None
                else {}
            ),
        }

    def reset(self) -> None:
        self.runtime.reset()

    def close(self) -> None:
        self.runtime.close()

    def __enter__(self) -> "OLMoENativePackageRuntime":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


__all__ = [
    "OLMoENativePackageRuntime",
    "OLMoENativeRuntimeError",
    "OLMoENativeTokenResult",
    "OLMoENativeTokenRuntime",
]
