"""Python owner for the transformer-shell-free native OLMoE token runtime."""

from __future__ import annotations

import ctypes
import json
import os
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


@dataclass(frozen=True)
class OLMoENativeTokenResult:
    next_token: int
    metrics: dict[str, int]


def _configure(library: ctypes.CDLL) -> None:
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


class OLMoENativeTokenRuntime:
    def __init__(
        self,
        model_config: str | Path,
        non_mlp_safetensors: str | Path,
        q7_artifact: str | Path,
        library: str | Path,
        *,
        threads: int = 1,
        local_window: int = 16,
        older_candidates: int = 8,
        older_top_k: int = 4,
        sink_tokens: int = 2,
    ):
        config_path = Path(model_config)
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        hidden = int(config_value["hidden_size"])
        heads = int(config_value["num_attention_heads"])
        kv_heads = int(config_value["num_key_value_heads"])
        if hidden % heads:
            raise ValueError("OLMoE hidden size is not divisible by heads")
        self._library = ctypes.CDLL(str(Path(library).resolve()))
        _configure(self._library)
        non_mlp_bytes = str(Path(non_mlp_safetensors).resolve()).encode()
        q7_bytes = str(Path(q7_artifact).resolve()).encode()
        native_config = _Config(
            non_mlp_bytes,
            q7_bytes,
            int(config_value["num_hidden_layers"]),
            hidden,
            heads,
            kv_heads,
            hidden // heads,
            threads,
            local_window,
            older_candidates,
            older_top_k,
            sink_tokens,
            float(config_value["rms_norm_eps"]),
            float(config_value["rope_theta"]),
        )
        error = ctypes.create_string_buffer(1024)
        self._handle = self._library.engram_olmoe_token_open(
            ctypes.byref(native_config), error, len(error)
        )
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
        result = OLMoENativeTokenResult(
            next_token=int(next_token.value),
            metrics={
                name: int(getattr(metrics, name)) for name, _ctype in metrics._fields_
            },
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
