"""CPU-only parity evaluation for the separate native BitNet source track."""

from __future__ import annotations

import gc
import os
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.models.native_bitnet import (
    LoadedNativeBitNetArtifact,
    NativeBitNetValidationError,
    OFFICIAL_NATIVE_BITNET_REPO,
    OFFICIAL_NATIVE_BITNET_REVISION,
    OFFICIAL_NATIVE_BITNET_WEIGHT_SHA256,
    _resolve_full_source,
    decode_native_bitnet_layer,
    load_native_bitnet_artifact,
    native_bitnet_repack_traffic,
    unpack_hf_bitnet_codes,
)
from engram.utils import atomic_json, sha256_file


def _disable_broken_optional_transformers_dependencies() -> None:
    """Keep unrelated local sklearn ABI problems out of text-model imports."""

    import transformers.utils as transformers_utils
    import transformers.utils.import_utils as transformers_imports

    if transformers_imports.is_sklearn_available():
        try:
            import sklearn  # noqa: F401
        except Exception:

            def sklearn_unavailable() -> bool:
                return False

            transformers_imports.is_sklearn_available = sklearn_unavailable
            transformers_utils.is_sklearn_available = sklearn_unavailable


def _torch_modules():
    try:
        import torch
        import torch.nn.functional as functional
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for native BitNet parity"
        ) from exc
    return torch, nn, functional


def _disable_bitnet_torch_compile() -> None:
    """Use eager reference functions when local Inductor is unavailable."""

    import transformers.integrations.bitnet as bitnet_integration

    unpack = bitnet_integration.unpack_weights
    original_unpack = getattr(unpack, "_torchdynamo_orig_callable", None)
    if original_unpack is not None:
        bitnet_integration.unpack_weights = original_unpack
    for quantizer in (
        bitnet_integration.ActQuant,
        bitnet_integration.WeightQuant,
    ):
        forward = quantizer.forward
        original_forward = getattr(
            forward,
            "_torchdynamo_orig_callable",
            None,
        )
        if original_forward is not None:
            quantizer.forward = staticmethod(original_forward)


def _artifact_mlp_class():
    torch, nn, functional = _torch_modules()

    class ArtifactBitNetMLP(nn.Module):
        """Dense parity oracle decoded from the packed Engram artifact.

        This class is intentionally an evaluation oracle, not the deployment
        kernel.  It performs the same activation quantization, ReLU-squared
        gate, intermediate RMS normalization, and scaled ternary linears as
        the Hugging Face reference after decoding a single artifact layer.
        """

        def __init__(
            self,
            artifact: LoadedNativeBitNetArtifact,
            layer_index: int,
        ) -> None:
            super().__init__()
            decoded = decode_native_bitnet_layer(artifact, layer_index)
            self.register_buffer(
                "gate_weight",
                torch.from_numpy(np.asarray(decoded["gate_codes"], dtype=np.int8)).to(
                    torch.bfloat16
                ),
            )
            self.register_buffer(
                "up_weight",
                torch.from_numpy(np.asarray(decoded["up_codes"], dtype=np.int8)).to(
                    torch.bfloat16
                ),
            )
            self.register_buffer(
                "down_weight",
                torch.from_numpy(np.asarray(decoded["down_codes"], dtype=np.int8)).to(
                    torch.bfloat16
                ),
            )
            self.register_buffer(
                "gate_scale",
                torch.as_tensor(decoded["gate_scale"], dtype=torch.bfloat16).reshape(1),
            )
            self.register_buffer(
                "up_scale",
                torch.as_tensor(decoded["up_scale"], dtype=torch.bfloat16).reshape(1),
            )
            self.register_buffer(
                "down_scale",
                torch.as_tensor(decoded["down_scale"], dtype=torch.bfloat16).reshape(1),
            )
            self.register_buffer(
                "ffn_sub_norm",
                torch.from_numpy(
                    np.asarray(decoded["ffn_sub_norm"], dtype=np.float32)
                ).to(torch.bfloat16),
            )
            self.rms_norm_eps = float(artifact.rms_norm_eps)

        @staticmethod
        def _activation_quant(values):
            input_dtype = values.dtype
            activation = values.float()
            scale = 127 / activation.abs().max(dim=-1, keepdim=True).values.clamp_(
                min=1e-5
            )
            activation = (activation * scale).round().clamp(-128, 127) / scale
            return activation.to(input_dtype)

        def _linear(self, values, weight, scale):
            output = functional.linear(
                self._activation_quant(values),
                weight,
            )
            return output * scale

        def forward(self, hidden_states):
            gate = self._linear(
                hidden_states,
                self.gate_weight,
                self.gate_scale,
            )
            up = self._linear(
                hidden_states,
                self.up_weight,
                self.up_scale,
            )
            activation = torch.square(functional.relu(gate)) * up
            input_dtype = activation.dtype
            normalized = activation.float()
            variance = normalized.pow(2).mean(-1, keepdim=True)
            normalized = normalized * torch.rsqrt(variance + self.rms_norm_eps)
            normalized = self.ffn_sub_norm * normalized.to(input_dtype)
            return self._linear(
                normalized,
                self.down_weight,
                self.down_scale,
            )

    return ArtifactBitNetMLP


def _tensor_metrics(reference, candidate) -> dict[str, Any]:
    torch, _, _ = _torch_modules()
    reference32 = reference.float()
    candidate32 = candidate.float()
    delta = candidate32 - reference32
    denominator = torch.linalg.vector_norm(reference32).clamp_min(1e-12)
    return {
        "exact": bool(torch.equal(reference, candidate)),
        "maximum_absolute_error": float(delta.abs().max().item()),
        "mean_absolute_error": float(delta.abs().mean().item()),
        "relative_l2": float((torch.linalg.vector_norm(delta) / denominator).item()),
    }


def _logit_metrics(reference, candidate) -> dict[str, Any]:
    torch, _, functional = _torch_modules()
    reference32 = reference.float()
    candidate32 = candidate.float()
    reference_log_probs = functional.log_softmax(reference32, dim=-1)
    candidate_log_probs = functional.log_softmax(candidate32, dim=-1)
    probabilities = reference_log_probs.exp()
    kl = torch.sum(
        probabilities * (reference_log_probs - candidate_log_probs),
        dim=-1,
    ).mean()
    top1 = (reference32.argmax(dim=-1) == candidate32.argmax(dim=-1)).float().mean()
    result = _tensor_metrics(reference, candidate)
    result.update(
        {
            "mean_kl_divergence": float(kl.item()),
            "top1_agreement": float(top1.item()),
        }
    )
    return result


def _load_reference_model(model_path: Path):
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    _disable_broken_optional_transformers_dependencies()
    _disable_bitnet_torch_compile()
    torch, _, _ = _torch_modules()
    from transformers import AutoModelForCausalLM

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model.eval()
    load_seconds = time.perf_counter() - started
    materialization = _materialize_reference_bitnet_weights(model)
    return model, load_seconds, materialization


def _materialize_reference_bitnet_weights(model) -> dict[str, Any]:
    """Repair packed AutoBitLinear tensors in incompatible Transformers builds."""

    torch, nn, _ = _torch_modules()
    from transformers.integrations.bitnet import AutoBitLinear

    converted_modules = 0
    wrapped_uint8_modules = 0
    coefficients = 0
    already_materialized_modules = 0
    for name, module in model.named_modules():
        if not isinstance(module, AutoBitLinear):
            continue
        weight = module.weight
        expected_shape = (module.out_features, module.in_features)
        if weight.dtype == torch.uint8 and tuple(weight.shape) == (
            module.out_features // 4,
            module.in_features,
        ):
            codes = unpack_hf_bitnet_codes(
                weight.detach().cpu().numpy(),
                out_features=module.out_features,
            )
            materialized = torch.from_numpy(codes).to(torch.bfloat16)
            module.weight = nn.Parameter(
                materialized,
                requires_grad=False,
            )
            if module.bias is not None:
                module.bias.data = module.bias.data.to(torch.bfloat16)
            converted_modules += 1
            coefficients += materialized.numel()
        elif tuple(weight.shape) == expected_shape and weight.dtype == torch.uint8:
            raw = weight.detach().cpu().numpy()
            if not np.all(
                (raw == np.uint8(0)) | (raw == np.uint8(1)) | (raw == np.uint8(255))
            ):
                raise NativeBitNetValidationError(
                    f"reference module {name!r} contains invalid unpacked "
                    "uint8 ternary values"
                )
            materialized = torch.from_numpy(raw.view(np.int8)).to(torch.bfloat16)
            module.weight = nn.Parameter(
                materialized,
                requires_grad=False,
            )
            if module.bias is not None:
                module.bias.data = module.bias.data.to(torch.bfloat16)
            converted_modules += 1
            wrapped_uint8_modules += 1
            coefficients += materialized.numel()
        elif tuple(weight.shape) == expected_shape and weight.dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ):
            already_materialized_modules += 1
        else:
            raise NativeBitNetValidationError(
                f"reference module {name!r} has unsupported weight "
                f"dtype/shape {weight.dtype}/{tuple(weight.shape)}"
            )
    if not converted_modules and not already_materialized_modules:
        raise NativeBitNetValidationError(
            "reference model contains no AutoBitLinear modules"
        )
    return {
        "policy": (
            "strictly decode official packed uint8 tensors when the installed "
            "Transformers build leaves AutoBitLinear weights unmaterialized"
        ),
        "converted_modules": converted_modules,
        "wrapped_uint8_modules": wrapped_uint8_modules,
        "already_materialized_modules": already_materialized_modules,
        "ternary_coefficients_materialized": coefficients,
    }


def evaluate_native_bitnet_parity(
    model: str | Path,
    artifact_path: str | Path,
    *,
    out: str | Path,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_layers: Sequence[int] = (0, 14, 29),
    local_states: int = 2,
    input_ids: Sequence[int] = (128000,),
    run_causal_substitution: bool = True,
    expected_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Compare artifact execution with the pinned native BitNet reference."""

    if local_states <= 0:
        raise ValueError("local_states must be positive")
    artifact_source = Path(artifact_path)
    artifact = load_native_bitnet_artifact(artifact_source)
    actual_artifact_sha256 = artifact.payload_sha256
    normalized_expected_sha256 = None
    if expected_artifact_sha256 is not None:
        normalized_expected_sha256 = expected_artifact_sha256.lower()
        if len(normalized_expected_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in normalized_expected_sha256
        ):
            raise NativeBitNetValidationError(
                "expected artifact SHA-256 must contain 64 hexadecimal characters"
            )
        if actual_artifact_sha256 != normalized_expected_sha256:
            raise NativeBitNetValidationError("native BitNet artifact SHA-256 mismatch")
    model_path, repo_id, resolved_revision = _resolve_full_source(
        model,
        revision=revision,
        cache_dir=cache_dir,
    )
    official_reference = (
        repo_id == OFFICIAL_NATIVE_BITNET_REPO
        and resolved_revision == OFFICIAL_NATIVE_BITNET_REVISION
    )
    reference_weight_path = model_path / "model.safetensors"
    reference_weight_sha256_before = None
    if official_reference:
        reference_weight_sha256_before = sha256_file(reference_weight_path)
        if reference_weight_sha256_before != OFFICIAL_NATIVE_BITNET_WEIGHT_SHA256:
            raise NativeBitNetValidationError(
                "pinned reference model.safetensors SHA-256 mismatch"
            )
    selected_layers = tuple(dict.fromkeys(int(layer) for layer in local_layers))
    if not selected_layers or any(
        layer < 0 or layer >= len(artifact.layers) for layer in selected_layers
    ):
        raise ValueError("local_layers contains an invalid layer index")
    if not input_ids:
        raise ValueError("input_ids must not be empty")
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in input_ids
    ):
        raise ValueError("input_ids must contain nonnegative integers")

    torch, _, _ = _torch_modules()
    ArtifactBitNetMLP = _artifact_mlp_class()
    reference_model, load_seconds, reference_materialization = _load_reference_model(
        model_path
    )
    reference_weight_sha256_after = None
    if official_reference:
        reference_weight_sha256_after = sha256_file(reference_weight_path)
        if reference_weight_sha256_after != reference_weight_sha256_before:
            raise NativeBitNetValidationError(
                "pinned reference weights changed while loading"
            )
    reference_layers = reference_model.model.layers
    if len(reference_layers) != len(artifact.layers):
        raise NativeBitNetValidationError("reference/artifact layer counts differ")
    if (
        reference_model.config.hidden_size != artifact.hidden_size
        or reference_model.config.intermediate_size != artifact.intermediate_size
    ):
        raise NativeBitNetValidationError("reference/artifact dimensions differ")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260723)
    states = torch.randn(
        local_states,
        1,
        artifact.hidden_size,
        generator=generator,
        dtype=torch.bfloat16,
    )
    local_results: dict[str, Any] = {}
    with torch.inference_mode():
        for layer_index in selected_layers:
            artifact_mlp = ArtifactBitNetMLP(artifact, layer_index)
            reference_output = reference_layers[layer_index].mlp(states)
            artifact_output = artifact_mlp(states)
            local_results[str(layer_index)] = _tensor_metrics(
                reference_output,
                artifact_output,
            )
            del artifact_mlp, reference_output, artifact_output
            gc.collect()

    causal = {
        "ran": False,
        "input_ids": list(input_ids),
    }
    if run_causal_substitution:
        ids = torch.tensor([list(input_ids)], dtype=torch.long)
        with torch.inference_mode():
            started = time.perf_counter()
            reference_output = reference_model(
                input_ids=ids,
                use_cache=False,
                output_hidden_states=True,
            )
            reference_seconds = time.perf_counter() - started
        for layer_index, layer in enumerate(reference_layers):
            layer.mlp = ArtifactBitNetMLP(artifact, layer_index)
        gc.collect()
        with torch.inference_mode():
            started = time.perf_counter()
            artifact_output = reference_model(
                input_ids=ids,
                use_cache=False,
                output_hidden_states=True,
            )
            artifact_seconds = time.perf_counter() - started
        final_hidden_reference = reference_output.hidden_states[-1]
        final_hidden_artifact = artifact_output.hidden_states[-1]
        causal = {
            "ran": True,
            "input_ids": list(input_ids),
            "reference_seconds": reference_seconds,
            "artifact_dense_oracle_seconds": artifact_seconds,
            "logits": _logit_metrics(
                reference_output.logits,
                artifact_output.logits,
            ),
            "final_hidden": _tensor_metrics(
                final_hidden_reference,
                final_hidden_artifact,
            ),
            "scope": (
                "one bounded CPU causal smoke sequence; the artifact path "
                "decodes to dense BF16 tensors and is a parity oracle, not "
                "the packed deployment kernel"
            ),
        }

    traffic = native_bitnet_repack_traffic(
        artifact.hidden_size,
        artifact.intermediate_size,
        layer_count=len(artifact.layers),
        cache_line_bytes=artifact.cache_line_bytes,
    )
    local_pass = all(metrics["exact"] for metrics in local_results.values())
    causal_pass = bool(
        causal.get("ran")
        and causal["logits"]["exact"]
        and causal["final_hidden"]["exact"]
    )
    serialized_layout_pass = bool(
        traffic["modelled_full_phase_schedule_passes_45_percent_gate"]
    )
    smoke_pass = (
        local_pass
        and (causal_pass if run_causal_substitution else True)
        and serialized_layout_pass
    )
    report = {
        "schema_version": 4,
        "source_track": "low_bit_native",
        "source_model": str(model),
        "repository": repo_id,
        "resolved_revision": resolved_revision,
        "source_model_path": str(model_path),
        "source_weight_integrity": {
            "official_verification_required": official_reference,
            "expected_sha256": (
                OFFICIAL_NATIVE_BITNET_WEIGHT_SHA256 if official_reference else None
            ),
            "sha256_before_load": reference_weight_sha256_before,
            "sha256_after_load": reference_weight_sha256_after,
            "verified": (
                official_reference
                and reference_weight_sha256_before
                == OFFICIAL_NATIVE_BITNET_WEIGHT_SHA256
                and reference_weight_sha256_after == reference_weight_sha256_before
            ),
        },
        "artifact_path": str(artifact_source.resolve()),
        "artifact_sha256": actual_artifact_sha256,
        "artifact_integrity": {
            "expected_sha256": normalized_expected_sha256,
            "verification_requested": normalized_expected_sha256 is not None,
            "verified": (
                normalized_expected_sha256 is not None
                and actual_artifact_sha256 == normalized_expected_sha256
            ),
        },
        "execution": {
            "device": "cpu",
            "dtype": "bfloat16",
            "torchdynamo_disabled": True,
            "reference_model_load_seconds": load_seconds,
            "reference_weight_materialization": reference_materialization,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "local_parity": {
            "layers": list(selected_layers),
            "states": local_states,
            "metrics": local_results,
            "passed": local_pass,
        },
        "causal_parity": causal,
        "traffic": traffic,
        "smoke_gate": {
            "passed": smoke_pass,
            "local_exact": local_pass,
            "causal_exact": causal_pass if run_causal_substitution else None,
            "serialized_cold_bytes_within_limit": serialized_layout_pass,
            "direct_packed_execution": False,
            "measured_hardware_traffic": False,
        },
        "combined_gate_status": (
            "low_bit_native_smoke_pass_not_dense_llama_conversion"
            if smoke_pass
            else "not_passed"
        ),
        "dense_llama_conversion_status": "not_applicable",
        "limitations": [
            (
                "The causal sample is a mechanical parity smoke test, not the "
                "frozen multi-sequence confirmation corpus."
            ),
            (
                "The evaluation module materializes dense BF16 matrices; "
                "native packed-kernel throughput remains unmeasured."
            ),
            (
                "Passing this separate source track does not establish that "
                "a dense Llama checkpoint can be converted losslessly."
            ),
        ],
    }
    atomic_json(Path(out), report)
    return report


__all__ = ["evaluate_native_bitnet_parity"]
