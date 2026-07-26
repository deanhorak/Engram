"""Sequence-disjoint output-scale calibration for practical BitNet DIP."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from engram.evaluation.native_bitnet_router import _trace_mlp_states
from engram.models.native_bitnet import (
    decode_native_bitnet_layer,
    load_native_bitnet_artifact,
)
from engram.training.controller_distillation import _load_trajectories
from engram.utils import atomic_json, sha256_file


def _fit_output_scale(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    actual = np.asarray(prediction, dtype=np.float64)
    exact = np.asarray(target, dtype=np.float64)
    if actual.shape != exact.shape or actual.size == 0:
        raise ValueError("prediction and target must have one non-empty shape")
    denominator = float(np.sum(actual * actual))
    if denominator <= 1e-24:
        return 1.0
    return max(1e-4, float(np.sum(actual * exact)) / denominator)


def _row_relative_l2(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    actual = np.asarray(prediction, dtype=np.float64)
    exact = np.asarray(target, dtype=np.float64)
    return np.linalg.norm(actual - exact, axis=1) / np.maximum(
        np.linalg.norm(exact, axis=1),
        1e-12,
    )


def _summary(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def fit_native_bitnet_dip_output_scales(
    package: str | Path,
    validation_trace: str | Path,
    router_policy: str | Path,
    *,
    out: str | Path,
    energy_target: float = 1.0,
    minimum_top_k: int = 346,
    maximum_top_k: int | None = None,
    sequence_length: int = 16,
    device: str = "cuda",
    rms_estimator: str = "corrected_proxy",
) -> dict[str, Any]:
    """Fit negligible-state per-layer scales with even/odd sequence cross-fit.

    Labels are exact candidate-only adaptive semantic outputs at the same
    practical candidate membership.  No causal logits or reserved corpus are
    used to fit the scales.
    """

    if (
        not 0 < energy_target <= 1
        or minimum_top_k <= 0
        or (
            maximum_top_k is not None
            and not minimum_top_k <= maximum_top_k
        )
        or sequence_length <= 0
        or rms_estimator not in {"corrected_proxy", "candidate_ratio"}
    ):
        raise ValueError("invalid adaptive calibration budget")
    package_path = Path(package).resolve()
    trace_path = Path(validation_trace).resolve()
    policy_path = Path(router_policy).resolve()
    trace = _load_trajectories(trace_path)
    manifest = json.loads(
        (package_path / "manifest.json").read_text(encoding="utf-8")
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    artifact = load_native_bitnet_artifact(
        package_path / manifest["mlp"]["path"]
    )
    layer_count = len(artifact.layers)
    joint_policy = policy.get("selected_policy", {})
    candidate_counts = joint_policy.get("candidate_counts")
    maximum_top_ks = joint_policy.get("maximum_ks")
    if candidate_counts is None:
        candidate_counts = policy.get("aggregate", {}).get(
            "selected_candidate_counts"
        )
    if maximum_top_ks is None and maximum_top_k is not None:
        maximum_top_ks = [maximum_top_k] * layer_count
    input_fraction = float(
        policy.get("configuration", {}).get("input_fraction", 0.0)
    )
    if (
        not isinstance(candidate_counts, list)
        or not isinstance(maximum_top_ks, list)
        or len(candidate_counts) != layer_count
        or len(maximum_top_ks) != layer_count
        or any(
            not isinstance(value, int)
            or not minimum_top_k <= value <= artifact.intermediate_size
            for value in maximum_top_ks
        )
        or any(
            not isinstance(candidate, int)
            or not maximum <= candidate <= artifact.intermediate_size
            for candidate, maximum in zip(
                candidate_counts,
                maximum_top_ks,
                strict=True,
            )
        )
        or any(
            not isinstance(value, int)
            or value <= 0
            for value in candidate_counts
        )
        or not 0 < input_fraction <= 1
    ):
        raise ValueError("router policy has no compatible candidate schedule")
    if trace.records % sequence_length:
        raise ValueError("trace records are not divisible by sequence_length")

    try:
        import torch
        import torch.nn.functional as functional
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "BitNet DIP calibration requires torch and safetensors"
        ) from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA calibration was requested but is unavailable")

    def activation_quant(values):
        dtype = values.dtype
        values32 = values.float()
        scale = 127.0 / values32.abs().amax(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-5)
        return (
            (values32 * scale).round().clamp(-128, 127) / scale
        ).to(dtype)

    hidden = artifact.hidden_size
    width = artifact.intermediate_size
    input_count = min(
        hidden,
        max(1, int(np.ceil(input_fraction * hidden))),
    )
    weights_path = package_path / manifest["transformer"]["non_mlp_path"]
    norm_weights = []
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for layer in range(layer_count):
            norm_weights.append(
                handle.get_tensor(
                    f"model.layers.{layer}.post_attention_layernorm.weight"
                )
                .float()
                .numpy()
            )
    epsilon = float(manifest["model"]["rms_norm_eps"])
    sequence_ids = np.arange(trace.records) // sequence_length
    even = sequence_ids % 2 == 0
    odd = ~even
    layer_reports = []
    final_scales = []
    crossfit_before: list[float] = []
    crossfit_after: list[float] = []
    selected_counts_all: list[int] = []
    started = time.perf_counter()

    for layer, (candidate_count, layer_maximum_k) in enumerate(
        zip(candidate_counts, maximum_top_ks, strict=True)
    ):
        states = _trace_mlp_states(
            trace,
            layer,
            norm_weights[layer],
            epsilon,
        )
        decoded = decode_native_bitnet_layer(artifact, layer)
        state = torch.from_numpy(states).to(device=device, dtype=torch.bfloat16)
        gate = torch.from_numpy(
            np.asarray(decoded["gate_codes"], dtype=np.int8)
        ).to(device=device, dtype=torch.bfloat16)
        up = torch.from_numpy(
            np.asarray(decoded["up_codes"], dtype=np.int8)
        ).to(device=device, dtype=torch.bfloat16)
        down = torch.from_numpy(
            np.asarray(decoded["down_codes"], dtype=np.int8)
        ).to(device=device, dtype=torch.bfloat16)
        gain = torch.from_numpy(
            np.asarray(decoded["ffn_sub_norm"], dtype=np.float32)
        ).to(device=device, dtype=torch.bfloat16)
        gate_scale = torch.as_tensor(
            decoded["gate_scale"],
            device=device,
            dtype=torch.bfloat16,
        )
        up_scale = torch.as_tensor(
            decoded["up_scale"],
            device=device,
            dtype=torch.bfloat16,
        )
        down_scale = torch.as_tensor(
            decoded["down_scale"],
            device=device,
            dtype=torch.bfloat16,
        )
        down_norm = torch.count_nonzero(down, dim=0).float()

        with torch.inference_mode():
            quantized = activation_quant(state)
            coordinates = torch.argsort(
                quantized.abs(),
                dim=1,
                descending=True,
                stable=True,
            )[:, :input_count]
            masked = torch.zeros_like(quantized)
            masked.scatter_(
                1,
                coordinates,
                quantized.gather(1, coordinates),
            )
            partial_gate = functional.linear(masked, gate) * gate_scale
            partial_up = functional.linear(masked, up) * up_scale
            proxy_raw = torch.relu(partial_gate).square() * partial_up
            proxy_utility = (
                proxy_raw.float().square()
                * gain.float().square()[None, :]
                * down_norm[None, :]
            )
            candidates = torch.argsort(
                proxy_utility,
                dim=1,
                descending=True,
                stable=True,
            )[:, :candidate_count]

            exact_gate = functional.linear(quantized, gate) * gate_scale
            exact_up = functional.linear(quantized, up) * up_scale
            exact_raw = torch.relu(exact_gate).square() * exact_up
            exact_variance = exact_raw.float().square().mean(dim=1)
            exact_normalized = (
                exact_raw.float()
                * torch.rsqrt(exact_variance[:, None] + epsilon)
            ).to(torch.bfloat16) * gain[None, :]
            exact_coefficients = activation_quant(exact_normalized)

            candidate_raw = exact_raw.gather(1, candidates)
            candidate_proxy_raw = proxy_raw.gather(1, candidates)
            proxy_square_sum = proxy_raw.float().square().sum(dim=1)
            candidate_square_sum = candidate_raw.float().square().sum(dim=1)
            candidate_proxy_square_sum = (
                candidate_proxy_raw.float().square().sum(dim=1)
            )
            if rms_estimator == "candidate_ratio":
                tail_scale = candidate_square_sum / (
                    candidate_proxy_square_sum.clamp_min(1e-30)
                )
                estimated_variance = (
                    candidate_square_sum
                    + tail_scale
                    * (proxy_square_sum - candidate_proxy_square_sum).clamp_min(0)
                ) / width
            else:
                estimated_variance = (
                    proxy_square_sum
                    + candidate_square_sum
                    - candidate_proxy_square_sum
                ) / width
            corrected_raw = proxy_raw.clone()
            corrected_raw.scatter_(1, candidates, candidate_raw)
            practical_normalized = (
                corrected_raw.float()
                * torch.rsqrt(estimated_variance[:, None] + epsilon)
            ).to(torch.bfloat16) * gain[None, :]
            practical_coefficients = activation_quant(
                practical_normalized
            ).gather(1, candidates)
            exact_candidate_coefficients = exact_coefficients.gather(
                1,
                candidates,
            )

            practical_utility = (
                practical_coefficients.float().square()
                * down_norm[candidates]
            )
            order = torch.argsort(
                practical_utility,
                dim=1,
                descending=True,
                stable=True,
            )
            sorted_utility = practical_utility.gather(1, order)
            cumulative = sorted_utility.double().cumsum(dim=1)
            total = cumulative[:, -1]
            required = (
                (cumulative < energy_target * total[:, None]).sum(dim=1)
                + 1
            )
            required[total <= 1e-30] = minimum_top_k
            selected_k = required.clamp(
                minimum_top_k,
                layer_maximum_k,
            )
            selected_local = order[:, :layer_maximum_k]
            selected_ids = candidates.gather(1, selected_local)
            rank = torch.arange(
                layer_maximum_k,
                device=device,
            )[None, :]
            active = rank < selected_k[:, None]

            practical_sparse = torch.zeros_like(exact_coefficients)
            practical_values = practical_coefficients.gather(
                1,
                selected_local,
            ) * active
            practical_sparse.scatter_(1, selected_ids, practical_values)
            target_sparse = torch.zeros_like(exact_coefficients)
            target_values = exact_candidate_coefficients.gather(
                1,
                selected_local,
            ) * active
            target_sparse.scatter_(1, selected_ids, target_values)
            practical_output = (
                functional.linear(practical_sparse, down) * down_scale
            ).float().cpu().numpy()
            target_output = (
                functional.linear(target_sparse, down) * down_scale
            ).float().cpu().numpy()
            selected_k_numpy = selected_k.cpu().numpy()

        scale_even = _fit_output_scale(
            practical_output[even],
            target_output[even],
        )
        scale_odd = _fit_output_scale(
            practical_output[odd],
            target_output[odd],
        )
        before = _row_relative_l2(practical_output, target_output)
        after = np.empty_like(before)
        after[odd] = _row_relative_l2(
            practical_output[odd] * scale_even,
            target_output[odd],
        )
        after[even] = _row_relative_l2(
            practical_output[even] * scale_odd,
            target_output[even],
        )
        final_scale = _fit_output_scale(practical_output, target_output)
        final_scales.append(final_scale)
        crossfit_before.extend(before.tolist())
        crossfit_after.extend(after.tolist())
        selected_counts_all.extend(selected_k_numpy.tolist())
        layer_reports.append(
            {
                "layer": layer,
                "candidate_count": candidate_count,
                "maximum_top_k": layer_maximum_k,
                "fit_even_validate_odd_scale": scale_even,
                "fit_odd_validate_even_scale": scale_odd,
                "full_validation_scale": final_scale,
                "selected_k": _summary(selected_k_numpy),
                "crossfit_relative_l2_before": _summary(before),
                "crossfit_relative_l2_after": _summary(after),
                "crossfit_improved": float(np.mean(after))
                < float(np.mean(before)),
            }
        )
        del (
            state,
            gate,
            up,
            down,
            gain,
            quantized,
            exact_raw,
            proxy_raw,
            practical_output,
            target_output,
        )
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    before_summary = _summary(crossfit_before)
    after_summary = _summary(crossfit_after)
    passed = (
        after_summary["mean"] < before_summary["mean"]
        and sum(
            bool(report["crossfit_improved"]) for report in layer_reports
        )
        >= int(np.ceil(0.8 * layer_count))
    )
    result = {
        "experiment": "native_bitnet_dip_sequence_disjoint_output_calibration",
        "scope": "validation_trace_local_outputs_only",
        "package": str(package_path),
        "validation_trace": {
            "path": str(trace_path),
            "manifest_sha256": sha256_file(trace_path / "manifest.json"),
            "records": trace.records,
            "sequence_length": sequence_length,
            "split": "even_sequence_ids_vs_odd_sequence_ids",
        },
        "router_policy": {
            "path": str(policy_path),
            "sha256": sha256_file(policy_path),
        },
        "configuration": {
            "energy_target": energy_target,
            "minimum_top_k": minimum_top_k,
            "maximum_top_ks": maximum_top_ks,
            "device": device,
            "rms_estimator": rms_estimator,
        },
        "layers": layer_reports,
        "output_scales": final_scales,
        "aggregate": {
            "crossfit_relative_l2_before": before_summary,
            "crossfit_relative_l2_after": after_summary,
            "layers_improved": sum(
                bool(report["crossfit_improved"])
                for report in layer_reports
            ),
            "selected_k": _summary(selected_counts_all),
        },
        "crossfit_passed": passed,
        "decision": (
            "use_output_scales_in_candidate_only_causal_development"
            if passed
            else "reject_global_per_layer_output_scaling"
        ),
        "causal_labels_used_for_fit": False,
        "reserved_confirmation_corpus_used": False,
        "milestone_2_status": "blocked",
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(Path(out), result)
    return result


__all__ = [
    "_fit_output_scale",
    "_row_relative_l2",
    "fit_native_bitnet_dip_output_scales",
]
