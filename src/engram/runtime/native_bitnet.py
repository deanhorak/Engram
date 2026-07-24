"""Generation runtime for source-independent native BitNet packages."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from engram.compiler.native_bitnet import (
    NATIVE_BITNET_PACKAGE_FORMAT,
    NATIVE_BITNET_PACKAGE_VERSION,
)
from engram.evaluation.native_bitnet_kernel import (
    NativeBitNetCPUKernel,
    _kernel_mlp_class,
)
from engram.evaluation.native_bitnet_parity import (
    _disable_bitnet_torch_compile,
    _disable_broken_optional_transformers_dependencies,
    _torch_modules,
)
from engram.models.native_bitnet import unpack_hf_bitnet_codes
from engram.models.native_bitnet import load_native_bitnet_artifact
from engram.utils import sha256_file


@dataclass(frozen=True)
class NativeBitNetGeneration:
    prompt_tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    text: str
    elapsed_seconds: float
    mlp_calls: int
    mlp_elapsed_seconds: float
    scheduled_mlp_bytes: int
    maximum_scratch_bytes: int


def _safe_package_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe native BitNet package path: {relative!r}")
    return root.joinpath(*pure.parts)


def _activation_quant(values):
    input_dtype = values.dtype
    activation = values.float()
    scale = 127 / activation.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-5)
    return ((activation * scale).round().clamp(-128, 127) / scale).to(input_dtype)


def _materialized_bitlinear_class():
    torch, nn, functional = _torch_modules()

    class MaterializedBitLinear(nn.Module):
        def __init__(self, codes, scale) -> None:
            super().__init__()
            self.register_buffer("weight", codes.to(torch.bfloat16).contiguous())
            self.register_buffer(
                "weight_scale",
                scale.to(torch.bfloat16).reshape(1).contiguous(),
            )

        def forward(self, values):
            return functional.linear(_activation_quant(values), self.weight) * (
                self.weight_scale
            )

    return MaterializedBitLinear


class NativeBitNetRuntime:
    """CPU BF16 transformer runtime backed by the direct packed MLP kernel."""

    def __init__(
        self,
        package: str | Path,
        *,
        library: str | Path | None = None,
        threads: int | None = None,
        verify_checksums: bool = True,
    ) -> None:
        self.path = Path(package).resolve()
        self.manifest = json.loads(
            (self.path / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            self.manifest.get("format") != NATIVE_BITNET_PACKAGE_FORMAT
            or self.manifest.get("version") != NATIVE_BITNET_PACKAGE_VERSION
        ):
            raise ValueError("not a supported native BitNet package")
        self._verify_package_files(verify_checksums)
        self.artifact_path = _safe_package_path(
            self.path,
            self.manifest["mlp"]["path"],
        )
        configured_threads = int(self.manifest["runtime"]["kernel_threads"])
        self.kernel = NativeBitNetCPUKernel(
            self.artifact_path,
            threads=configured_threads if threads is None else threads,
            library=library,
            expected_sha256=self.manifest["mlp"]["sha256"],
        )
        try:
            self.model, self.tokenizer = self._load_transformer()
        except Exception:
            self.kernel.close()
            raise

    def close(self) -> None:
        if getattr(self, "kernel", None) is not None:
            self.kernel.close()

    def __enter__(self) -> NativeBitNetRuntime:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _verify_package_files(self, verify_checksums: bool) -> None:
        files = self.manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("native BitNet manifest has no file inventory")
        for relative, descriptor in files.items():
            path = _safe_package_path(self.path, relative)
            if (
                not isinstance(descriptor, dict)
                or not path.is_file()
                or path.stat().st_size != descriptor.get("bytes")
            ):
                raise ValueError(
                    f"native BitNet package file is missing or malformed: {relative}"
                )
            if verify_checksums and sha256_file(path) != descriptor.get("sha256"):
                raise ValueError(f"native BitNet package checksum mismatch: {relative}")

    def _load_transformer(self):
        _disable_broken_optional_transformers_dependencies()
        _disable_bitnet_torch_compile()
        torch, nn, _ = _torch_modules()
        try:
            from accelerate import init_empty_weights
            from safetensors import safe_open
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "install engram-lm[conversion] to load native BitNet packages"
            ) from exc

        config_dir = self.path / "config"
        tokenizer_dir = self.path / self.manifest["tokenizer"]["path"]
        config = AutoConfig.from_pretrained(config_dir, local_files_only=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        weights_path = _safe_package_path(
            self.path,
            self.manifest["transformer"]["non_mlp_path"],
        )
        MaterializedBitLinear = _materialized_bitlinear_class()
        ordinary_state = {}
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            names = set(handle.keys())
            for layer_index, layer in enumerate(model.model.layers):
                for projection_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    prefix = f"model.layers.{layer_index}.self_attn.{projection_name}"
                    weight_name = f"{prefix}.weight"
                    scale_name = f"{prefix}.weight_scale"
                    if weight_name not in names or scale_name not in names:
                        raise ValueError(
                            f"native BitNet package is missing {projection_name} "
                            f"for layer {layer_index}"
                        )
                    projection = getattr(layer.self_attn, projection_name)
                    packed = handle.get_tensor(weight_name).cpu().numpy()
                    codes = unpack_hf_bitnet_codes(
                        packed,
                        out_features=projection.out_features,
                    )
                    decoded = torch.from_numpy(codes)
                    scale = handle.get_tensor(scale_name)
                    setattr(
                        layer.self_attn,
                        projection_name,
                        MaterializedBitLinear(decoded, scale),
                    )
            projection_keys = {
                f"model.layers.{layer_index}.self_attn.{projection_name}.{suffix}"
                for layer_index in range(len(model.model.layers))
                for projection_name in ("q_proj", "k_proj", "v_proj", "o_proj")
                for suffix in ("weight", "weight_scale")
            }
            for name in names - projection_keys:
                ordinary_state[name] = handle.get_tensor(name)

        incompatible = model.load_state_dict(
            ordinary_state,
            strict=False,
            assign=True,
        )
        unexpected = sorted(incompatible.unexpected_keys)
        if unexpected:
            raise ValueError(
                f"native BitNet package has unexpected tensors: {unexpected[:4]}"
            )
        NativeKernelBitNetMLP = _kernel_mlp_class()
        for layer_index, layer in enumerate(model.model.layers):
            layer.mlp = NativeKernelBitNetMLP(self.kernel, layer_index)
        model.tie_weights()
        meta_parameters = [
            name
            for name, parameter in model.named_parameters()
            if parameter.device.type == "meta"
        ]
        if meta_parameters:
            raise ValueError(
                f"native BitNet package left meta tensors: {meta_parameters[:4]}"
            )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            local_files_only=True,
            fix_mistral_regex=bool(
                self.manifest["tokenizer"].get("fix_mistral_regex", False)
            ),
        )
        return model, tokenizer

    def encode(self, prompt: str) -> list[int]:
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        if not tokens:
            raise ValueError("native BitNet prompt tokenized to an empty sequence")
        return [int(value) for value in tokens]

    def decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(list(tokens), skip_special_tokens=True)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids=input_ids, **kwargs)

    def generate_tokens(
        self,
        prompt_tokens: Sequence[int],
        *,
        max_new_tokens: int,
    ) -> NativeBitNetGeneration:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not prompt_tokens:
            raise ValueError("prompt_tokens must not be empty")
        torch, _, _ = _torch_modules()
        input_ids = torch.tensor([list(prompt_tokens)], dtype=torch.long)
        generated: list[int] = []
        self.kernel.clear_metrics()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(input_ids=input_ids, use_cache=True)
            past_key_values = output.past_key_values
            next_token = int(output.logits[0, -1].argmax().item())
            generated.append(next_token)
            for _ in range(1, max_new_tokens):
                output = self.model(
                    input_ids=torch.tensor([[next_token]], dtype=torch.long),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
                next_token = int(output.logits[0, -1].argmax().item())
                generated.append(next_token)
        elapsed = time.perf_counter() - started
        calls = list(self.kernel.calls)
        return NativeBitNetGeneration(
            prompt_tokens=tuple(int(value) for value in prompt_tokens),
            generated_tokens=tuple(generated),
            text=self.decode(generated),
            elapsed_seconds=elapsed,
            mlp_calls=len(calls),
            mlp_elapsed_seconds=sum(int(call["elapsed_ns"]) for call in calls) / 1e9,
            scheduled_mlp_bytes=sum(
                int(call["scheduled_cache_line_bytes"]) for call in calls
            ),
            maximum_scratch_bytes=max(
                (int(call["scratch_bytes"]) for call in calls),
                default=0,
            ),
        )

    def generate(self, prompt: str, *, max_new_tokens: int) -> NativeBitNetGeneration:
        return self.generate_tokens(
            self.encode(prompt),
            max_new_tokens=max_new_tokens,
        )


def validate_native_bitnet_package(
    package: str | Path,
    *,
    verify_checksums: bool = True,
) -> dict[str, Any]:
    """Validate a package without materializing transformer tensors."""

    root = Path(package).resolve()
    errors = []
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("format") != NATIVE_BITNET_PACKAGE_FORMAT
            or manifest.get("version") != NATIVE_BITNET_PACKAGE_VERSION
        ):
            raise ValueError("unsupported native BitNet package manifest")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("native BitNet package has no file inventory")
        for relative, descriptor in files.items():
            path = _safe_package_path(root, relative)
            if (
                not isinstance(descriptor, dict)
                or not path.is_file()
                or path.stat().st_size != descriptor.get("bytes")
            ):
                raise ValueError(f"invalid package file: {relative}")
            if verify_checksums and sha256_file(path) != descriptor.get("sha256"):
                raise ValueError(f"package checksum mismatch: {relative}")
        artifact_path = _safe_package_path(root, manifest["mlp"]["path"])
        artifact = load_native_bitnet_artifact(artifact_path)
        if artifact.payload_sha256 != manifest["mlp"]["sha256"]:
            raise ValueError("package MLP artifact hash differs from manifest")
        weights_path = _safe_package_path(
            root,
            manifest["transformer"]["non_mlp_path"],
        )
        from safetensors import safe_open

        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            names = list(handle.keys())
        if not names or any(".mlp." in name for name in names):
            raise ValueError("package non-MLP tensor boundary is invalid")
        if "model.embed_tokens.weight" not in names:
            raise ValueError("package token embedding is missing")
    except Exception as exc:
        errors.append(str(exc))
        manifest = {}
    return {
        "valid": not errors,
        "format": manifest.get("format"),
        "version": manifest.get("version"),
        "source_independent": bool(manifest.get("does_not_require_source_transformer")),
        "file_count": len(manifest.get("files", {})),
        "errors": errors,
    }


__all__ = [
    "NativeBitNetGeneration",
    "NativeBitNetRuntime",
    "validate_native_bitnet_package",
]
