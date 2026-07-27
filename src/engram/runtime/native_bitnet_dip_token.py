"""ctypes binding for the authenticated native BitNet DIP token runtime."""

from __future__ import annotations

import ctypes
import json
import operator
import os
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from engram.compiler.native_bitnet import (
    NATIVE_BITNET_ATTENTION_OPERATOR,
    NATIVE_BITNET_DIP_DERIVED_MANIFEST_BYTES,
    NATIVE_BITNET_DIP_DERIVED_MANIFEST_SHA256,
    NATIVE_BITNET_DIP_OPERATOR,
    NATIVE_BITNET_PACKAGE_FORMAT,
    NATIVE_BITNET_PACKAGE_VERSION,
)
from engram.runtime.native_bitnet import NativeBitNetGeneration
from engram.utils import sha256_file

_ABI_VERSION = 1
_ERROR_CAPACITY = 2048
_MAXIMUM_THREADS = 256
_MAXIMUM_EOS_IDS = 8


class NativeBitNetDIPTokenRuntimeError(RuntimeError):
    """Raised when the authenticated token runtime rejects a request."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class _ConfigV1(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("package_path", ctypes.c_char_p),
        ("threads", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _InfoV1(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("max_position_embeddings", ctypes.c_uint64),
        ("vocabulary_size", ctypes.c_uint64),
        ("layers", ctypes.c_uint64),
        ("hidden_size", ctypes.c_uint64),
        ("intermediate_size", ctypes.c_uint64),
        ("query_heads", ctypes.c_uint64),
        ("key_value_heads", ctypes.c_uint64),
        ("head_dimension", ctypes.c_uint64),
        ("local_window", ctypes.c_uint64),
        ("older_candidates", ctypes.c_uint64),
        ("older_top_k", ctypes.c_uint64),
        ("sink_tokens", ctypes.c_uint64),
        ("thread_count", ctypes.c_uint32),
        ("eos_token_count", ctypes.c_uint32),
        ("rms_norm_epsilon", ctypes.c_float),
        ("rope_theta", ctypes.c_float),
        ("eos_token_ids", ctypes.c_int64 * _MAXIMUM_EOS_IDS),
        ("semantic_backend", ctypes.c_char * 64),
        ("package_manifest_sha256", ctypes.c_char * 65),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _MetricsV1(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("prompt_tokens", ctypes.c_uint64),
        ("generated_tokens", ctypes.c_uint64),
        ("positions_processed", ctypes.c_uint64),
        ("stage_calls", ctypes.c_uint64),
        ("semantic_calls", ctypes.c_uint64),
        ("semantic_rows", ctypes.c_uint64),
        ("semantic_selected_records", ctypes.c_uint64),
        ("semantic_kernel_cache_line_bytes", ctypes.c_uint64),
        ("semantic_global_metadata_bytes", ctypes.c_uint64),
        ("semantic_cache_line_bytes", ctypes.c_uint64),
        ("semantic_maximum_scratch_bytes", ctypes.c_uint64),
        ("attention_logical_read_bytes", ctypes.c_uint64),
        ("attention_state_bytes", ctypes.c_uint64),
        ("attention_scratch_bytes", ctypes.c_uint64),
        ("qkv_projection_ns", ctypes.c_uint64),
        ("rope_ns", ctypes.c_uint64),
        ("native_attention_ns", ctypes.c_uint64),
        ("output_projection_ns", ctypes.c_uint64),
        ("semantic_elapsed_ns", ctypes.c_uint64),
        ("attention_elapsed_ns", ctypes.c_uint64),
        ("call_elapsed_ns", ctypes.c_uint64),
        ("prefill_elapsed_ns", ctypes.c_uint64),
        ("decode_elapsed_ns", ctypes.c_uint64),
        ("stopped_on_eos", ctypes.c_uint32),
        ("reserved32", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 4),
    ]

    def to_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name, _ctype in self._fields_
            if name != "reserved"
        }


def _default_library_path() -> Path:
    configured = os.environ.get("ENGRAM_BITNET_TOKEN_RUNTIME_LIBRARY")
    if configured:
        return Path(configured).expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    return repository / "build" / "libengram_bitnet_token_runtime.so"


def _configure_library(library) -> None:
    library.engram_native_bitnet_token_abi_version_v1.argtypes = []
    library.engram_native_bitnet_token_abi_version_v1.restype = ctypes.c_uint32
    library.engram_native_bitnet_token_create_v1.argtypes = [
        ctypes.POINTER(_ConfigV1),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.engram_native_bitnet_token_create_v1.restype = ctypes.c_int32
    library.engram_native_bitnet_token_destroy_v1.argtypes = [ctypes.c_void_p]
    library.engram_native_bitnet_token_destroy_v1.restype = None
    library.engram_native_bitnet_token_reset_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.engram_native_bitnet_token_reset_v1.restype = ctypes.c_int32
    library.engram_native_bitnet_token_get_info_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_InfoV1),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.engram_native_bitnet_token_get_info_v1.restype = ctypes.c_int32
    library.engram_native_bitnet_token_generate_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(_MetricsV1),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.engram_native_bitnet_token_generate_v1.restype = ctypes.c_int32


def _load_library(path: str | Path | None):
    library_path = (
        _default_library_path()
        if path is None
        else Path(path).expanduser().resolve()
    )
    if not library_path.is_file():
        raise NativeBitNetDIPTokenRuntimeError(
            f"native BitNet token-runtime library is missing: {library_path}; "
            "run `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` and "
            "`cmake --build build --target engram_bitnet_token_runtime`"
        )
    try:
        library = ctypes.CDLL(str(library_path))
        _configure_library(library)
    except (AttributeError, OSError) as exc:
        raise NativeBitNetDIPTokenRuntimeError(
            "failed to load the version-1 native BitNet token-runtime ABI "
            f"from {library_path}: {exc}"
        ) from exc
    actual_abi = int(library.engram_native_bitnet_token_abi_version_v1())
    if actual_abi != _ABI_VERSION:
        raise NativeBitNetDIPTokenRuntimeError(
            f"native BitNet token-runtime ABI {actual_abi} is unsupported"
        )
    return library, library_path


def _absolute_without_symlink_resolution(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _safe_package_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str):
        raise NativeBitNetDIPTokenRuntimeError(
            f"unsafe native BitNet package path: {relative!r}"
        )
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise NativeBitNetDIPTokenRuntimeError(
            f"unsafe native BitNet package path: {relative!r}"
        )
    return root.joinpath(*pure.parts)


def _load_pinned_manifest(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise NativeBitNetDIPTokenRuntimeError(
            "native BitNet DIP package root must be a non-symlink directory"
        )
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise NativeBitNetDIPTokenRuntimeError(
            "native BitNet DIP manifest must be a regular non-symlink file"
        )
    if (
        manifest_path.stat().st_size
        != NATIVE_BITNET_DIP_DERIVED_MANIFEST_BYTES
        or sha256_file(manifest_path)
        != NATIVE_BITNET_DIP_DERIVED_MANIFEST_SHA256
    ):
        raise NativeBitNetDIPTokenRuntimeError(
            "native BitNet DIP manifest does not match the promoted trust root"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeBitNetDIPTokenRuntimeError(
            f"cannot read authenticated native BitNet manifest: {exc}"
        ) from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != NATIVE_BITNET_PACKAGE_FORMAT
        or manifest.get("version") != NATIVE_BITNET_PACKAGE_VERSION
        or manifest.get("runtime", {}).get("mlp_mode")
        != NATIVE_BITNET_DIP_OPERATOR
        or manifest.get("runtime", {}).get("attention_mode")
        != NATIVE_BITNET_ATTENTION_OPERATOR
    ):
        raise NativeBitNetDIPTokenRuntimeError(
            "authenticated manifest does not declare the promoted DIP runtime"
        )
    return manifest


def _verify_tokenizer_assets(root: Path, manifest: dict[str, Any]) -> Path:
    tokenizer = manifest.get("tokenizer")
    inventory = manifest.get("files")
    if not isinstance(tokenizer, dict) or not isinstance(inventory, dict):
        raise NativeBitNetDIPTokenRuntimeError(
            "authenticated package tokenizer descriptor is malformed"
        )
    tokenizer_relative = tokenizer.get("path")
    listed = tokenizer.get("files")
    if not isinstance(listed, list) or not listed:
        raise NativeBitNetDIPTokenRuntimeError(
            "authenticated package has no tokenizer inventory"
        )
    directory = _safe_package_path(root, tokenizer_relative)
    if directory.is_symlink() or not directory.is_dir():
        raise NativeBitNetDIPTokenRuntimeError(
            "authenticated tokenizer path is not a regular directory"
        )
    expected_names: set[str] = set()
    for name in listed:
        pure = PurePosixPath(name) if isinstance(name, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or len(pure.parts) != 1
            or pure.name in {"", ".", ".."}
        ):
            raise NativeBitNetDIPTokenRuntimeError(
                "authenticated tokenizer file name is unsafe"
            )
        expected_names.add(pure.name)
        relative = f"{tokenizer_relative}/{pure.name}"
        descriptor = inventory.get(relative)
        path = directory / pure.name
        if (
            not isinstance(descriptor, dict)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != descriptor.get("bytes")
            or sha256_file(path) != descriptor.get("sha256")
        ):
            raise NativeBitNetDIPTokenRuntimeError(
                f"authenticated tokenizer asset is corrupt: {relative}"
            )
    actual_names = {path.name for path in directory.iterdir()}
    if actual_names != expected_names:
        raise NativeBitNetDIPTokenRuntimeError(
            "authenticated tokenizer directory inventory is not exact"
        )
    return directory


def _load_packaged_tokenizer(directory: Path, *, fix_mistral_regex: bool):
    try:
        from engram.evaluation.native_bitnet_parity import (
            _disable_broken_optional_transformers_dependencies,
        )

        _disable_broken_optional_transformers_dependencies()
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise NativeBitNetDIPTokenRuntimeError(
            "install engram-lm[conversion] for packaged chat tokenization"
        ) from exc
    return AutoTokenizer.from_pretrained(
        directory,
        local_files_only=True,
        fix_mistral_regex=fix_mistral_regex,
    )


def _decode_c_string(value) -> str:
    return bytes(value).split(b"\0", 1)[0].decode("utf-8", "strict")


class NativeBitNetDIPTokenRuntime:
    """Persistent CPU-only handle for authenticated DIP token generation."""

    def __init__(
        self,
        package: str | Path,
        *,
        library: str | Path | None = None,
        threads: int | None = None,
    ) -> None:
        if threads is None:
            configured_threads = 0
        elif (
            isinstance(threads, bool)
            or not isinstance(threads, int)
            or not 1 <= threads <= _MAXIMUM_THREADS
        ):
            raise ValueError(
                f"threads must be in [1, {_MAXIMUM_THREADS}] or omitted"
            )
        else:
            configured_threads = threads
        self.path = _absolute_without_symlink_resolution(package)
        self.manifest = _load_pinned_manifest(self.path)
        self._library, self.library_path = _load_library(library)
        self._handle = ctypes.c_void_p()
        self._last_metrics: dict[str, int] = {}
        package_bytes = os.fsencode(self.path)
        config = _ConfigV1(
            abi_version=_ABI_VERSION,
            struct_size=ctypes.sizeof(_ConfigV1),
            package_path=package_bytes,
            threads=configured_threads,
            flags=0,
        )
        error = ctypes.create_string_buffer(_ERROR_CAPACITY)
        status = int(
            self._library.engram_native_bitnet_token_create_v1(
                ctypes.byref(config),
                ctypes.byref(self._handle),
                error,
                len(error),
            )
        )
        if status or not self._handle.value:
            self._handle = ctypes.c_void_p()
            self._raise_status(status, error, "create")
        try:
            self._info = self._read_info()
            self._validate_info()
            tokenizer_directory = _verify_tokenizer_assets(
                self.path,
                self.manifest,
            )
            self.tokenizer = _load_packaged_tokenizer(
                tokenizer_directory,
                fix_mistral_regex=bool(
                    self.manifest["tokenizer"].get(
                        "fix_mistral_regex",
                        False,
                    )
                ),
            )
            # Detect mutation while the tokenizer frontend was loading.
            if _load_pinned_manifest(self.path) != self.manifest:
                raise NativeBitNetDIPTokenRuntimeError(
                    "native BitNet package changed while loading the tokenizer"
                )
            _verify_tokenizer_assets(self.path, self.manifest)
        except Exception:
            self.close()
            raise

    def _raise_status(self, status: int, error, operation: str) -> None:
        detail = error.value.decode("utf-8", "replace").strip()
        if not detail:
            detail = f"native BitNet token-runtime {operation} failed"
        raise NativeBitNetDIPTokenRuntimeError(detail, status=status)

    def _require_open(self) -> None:
        if not getattr(self, "_handle", None) or not self._handle.value:
            raise NativeBitNetDIPTokenRuntimeError(
                "native BitNet token runtime is closed"
            )

    def _read_info(self) -> _InfoV1:
        self._require_open()
        info = _InfoV1(
            abi_version=_ABI_VERSION,
            struct_size=ctypes.sizeof(_InfoV1),
        )
        error = ctypes.create_string_buffer(_ERROR_CAPACITY)
        status = int(
            self._library.engram_native_bitnet_token_get_info_v1(
                self._handle,
                ctypes.byref(info),
                error,
                len(error),
            )
        )
        if status:
            self._raise_status(status, error, "get-info")
        return info

    def _validate_info(self) -> None:
        info = self._info
        if (
            int(info.abi_version) != _ABI_VERSION
            or int(info.struct_size) != ctypes.sizeof(_InfoV1)
            or _decode_c_string(info.semantic_backend)
            != NATIVE_BITNET_DIP_OPERATOR
            or _decode_c_string(info.package_manifest_sha256)
            != NATIVE_BITNET_DIP_DERIVED_MANIFEST_SHA256
            or int(info.local_window) != 16
            or int(info.older_candidates) != 8
            or int(info.older_top_k) != 4
            or int(info.sink_tokens) != 2
            or not 1 <= int(info.thread_count) <= _MAXIMUM_THREADS
            or not 0 < int(info.eos_token_count) <= _MAXIMUM_EOS_IDS
        ):
            raise NativeBitNetDIPTokenRuntimeError(
                "native BitNet token-runtime metadata is unsupported"
            )
        self.semantic_backend = _decode_c_string(info.semantic_backend)
        self.thread_count = int(info.thread_count)
        self.vocabulary_size = int(info.vocabulary_size)
        self.max_position_embeddings = int(info.max_position_embeddings)
        self.eos_token_ids = tuple(
            int(info.eos_token_ids[index])
            for index in range(int(info.eos_token_count))
        )
        if set(self.eos_token_ids) != {128001, 128009}:
            raise NativeBitNetDIPTokenRuntimeError(
                "native BitNet token runtime returned an unexpected EOS set"
            )
        self.attention_mode = NATIVE_BITNET_ATTENTION_OPERATOR

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None and handle.value:
            self._library.engram_native_bitnet_token_destroy_v1(handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> NativeBitNetDIPTokenRuntime:
        self._require_open()
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def last_metrics(self) -> dict[str, int]:
        return dict(self._last_metrics)

    def reset(self) -> None:
        self._require_open()
        error = ctypes.create_string_buffer(_ERROR_CAPACITY)
        status = int(
            self._library.engram_native_bitnet_token_reset_v1(
                self._handle,
                error,
                len(error),
            )
        )
        if status:
            self._raise_status(status, error, "reset")
        self._last_metrics = {}

    @staticmethod
    def _token_ids(values: Sequence[int]) -> list[int]:
        tokens = []
        for value in values:
            if isinstance(value, bool):
                raise ValueError("token ids must be integers, not booleans")
            try:
                token = operator.index(value)
            except TypeError as exc:
                raise ValueError("token ids must be integers") from exc
            if not 0 <= token <= (1 << 63) - 1:
                raise ValueError("token id is outside signed 64-bit range")
            tokens.append(token)
        return tokens

    def encode(self, prompt: str) -> list[int]:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        if not tokens:
            raise ValueError("native BitNet prompt tokenized to an empty sequence")
        return self._token_ids(tokens)

    def decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(
            self._token_ids(tokens),
            skip_special_tokens=True,
        )

    def generate_tokens(
        self,
        prompt_tokens: Sequence[int],
        *,
        max_new_tokens: int,
    ) -> NativeBitNetGeneration:
        self._require_open()
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be positive")
        prompt = self._token_ids(prompt_tokens)
        if not prompt:
            raise ValueError("prompt_tokens must not be empty")
        if any(token >= self.vocabulary_size for token in prompt):
            raise ValueError("prompt token id is outside authenticated vocabulary")
        if (
            len(prompt) > self.max_position_embeddings
            or max_new_tokens - 1
            > self.max_position_embeddings - len(prompt)
        ):
            raise ValueError(
                "prompt and generation budget exceed authenticated context length"
            )

        self.reset()
        prompt_array = (ctypes.c_int64 * len(prompt))(*prompt)
        output_array = (ctypes.c_int64 * max_new_tokens)()
        output_count = ctypes.c_uint64()
        metrics = _MetricsV1(
            abi_version=_ABI_VERSION,
            struct_size=ctypes.sizeof(_MetricsV1),
        )
        error = ctypes.create_string_buffer(_ERROR_CAPACITY)
        status = int(
            self._library.engram_native_bitnet_token_generate_v1(
                self._handle,
                prompt_array,
                len(prompt),
                max_new_tokens,
                output_array,
                max_new_tokens,
                ctypes.byref(output_count),
                ctypes.byref(metrics),
                error,
                len(error),
            )
        )
        if status:
            try:
                self.reset()
            except NativeBitNetDIPTokenRuntimeError:
                self.close()
            self._raise_status(status, error, "generate")
        count = int(output_count.value)
        if (
            count <= 0
            or count > max_new_tokens
            or int(metrics.generated_tokens) != count
            or int(metrics.prompt_tokens) != len(prompt)
        ):
            self.close()
            raise NativeBitNetDIPTokenRuntimeError(
                "native BitNet token runtime returned inconsistent output"
            )
        generated = tuple(int(output_array[index]) for index in range(count))
        self._last_metrics = metrics.to_dict()
        return NativeBitNetGeneration(
            prompt_tokens=tuple(prompt),
            generated_tokens=generated,
            text=self.decode(generated),
            elapsed_seconds=int(metrics.call_elapsed_ns) / 1e9,
            mlp_calls=int(metrics.semantic_calls),
            mlp_elapsed_seconds=int(metrics.semantic_elapsed_ns) / 1e9,
            scheduled_mlp_bytes=int(metrics.semantic_cache_line_bytes),
            maximum_scratch_bytes=int(
                metrics.semantic_maximum_scratch_bytes
            ),
            attention_mode=self.attention_mode,
            attention_tokens_seen=int(metrics.positions_processed),
            attention_logical_read_bytes=int(
                metrics.attention_logical_read_bytes
            ),
            attention_state_bytes=int(metrics.attention_state_bytes),
            attention_scratch_bytes=int(metrics.attention_scratch_bytes),
            qkv_projection_seconds=int(metrics.qkv_projection_ns) / 1e9,
            rope_seconds=int(metrics.rope_ns) / 1e9,
            native_attention_seconds=int(metrics.native_attention_ns) / 1e9,
            output_projection_seconds=(
                int(metrics.output_projection_ns) / 1e9
            ),
            native_attention_calls=int(metrics.stage_calls),
            stopped_on_eos=bool(metrics.stopped_on_eos),
            prefill_seconds=int(metrics.prefill_elapsed_ns) / 1e9,
            decode_seconds=int(metrics.decode_elapsed_ns) / 1e9,
            controller_mode="native_exact_operator_residual",
            controller_seconds=0.0,
            controller_state_bytes=0,
            decoder_layer_forward_calls=0,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> NativeBitNetGeneration:
        return self.generate_tokens(
            self.encode(prompt),
            max_new_tokens=max_new_tokens,
        )


__all__ = [
    "NativeBitNetDIPTokenRuntime",
    "NativeBitNetDIPTokenRuntimeError",
]
