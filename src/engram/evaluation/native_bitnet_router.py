"""Learned practical-router probes for the native BitNet semantic oracle."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.models.native_bitnet import (
    _activation_quant,
    decode_native_bitnet_layer,
    load_native_bitnet_artifact,
)
from engram.training.controller_distillation import _load_trajectories
from engram.utils import atomic_json, sha256_file


def native_bitnet_router_traffic(
    hidden_size: int,
    intermediate_size: int,
    *,
    rank: int,
    candidate_count: int,
    bytes_per_router_parameter: int = 2,
) -> dict[str, Any]:
    """Model cold bytes for a nonlinear low-rank router plus candidate records."""

    if min(hidden_size, intermediate_size, rank, candidate_count) <= 0:
        raise ValueError("router traffic dimensions must be positive")
    if candidate_count > intermediate_size:
        raise ValueError("candidate_count exceeds intermediate size")
    router_parameters = (
        hidden_size * rank + rank * intermediate_size + intermediate_size
    )
    router_bytes = router_parameters * bytes_per_router_parameter
    packed_width = (hidden_size + 4) // 5
    candidate_record_bytes = candidate_count * (3 * packed_width + 2)
    dense_q4_bytes = (3 * hidden_size * intermediate_size + 1) // 2
    total = router_bytes + candidate_record_bytes
    return {
        "router_parameters": router_parameters,
        "router_bytes": router_bytes,
        "candidate_record_bytes": candidate_record_bytes,
        "complete_modelled_bytes": total,
        "dense_q4_bytes": dense_q4_bytes,
        "fraction_of_dense_q4": total / dense_q4_bytes,
        "passes_45_percent": total / dense_q4_bytes <= 0.45,
        "includes": (
            "FP16 input/output router factors and bias plus complete packed "
            "gate/up/gain/down payload for every candidate"
        ),
        "excludes": "headers, alignment, and runtime scratch",
    }


def _rms_norm(states: np.ndarray, weight: np.ndarray, epsilon: float) -> np.ndarray:
    values = np.asarray(states, dtype=np.float32)
    inverse = np.reciprocal(
        np.sqrt(np.mean(values * values, axis=1, keepdims=True) + epsilon)
    )
    return values * inverse * np.asarray(weight, dtype=np.float32)[None, :]


def _oracle_membership(
    artifact,
    layer: int,
    states: np.ndarray,
    top_k: int,
) -> np.ndarray:
    decoded = decode_native_bitnet_layer(artifact, layer)
    quantized = _activation_quant(np.asarray(states, dtype=np.float32))
    gate = (
        quantized @ np.asarray(decoded["gate_codes"], dtype=np.float32).T
        * np.asarray(decoded["gate_scale"], dtype=np.float32)
    )
    up = (
        quantized @ np.asarray(decoded["up_codes"], dtype=np.float32).T
        * np.asarray(decoded["up_scale"], dtype=np.float32)
    )
    activation = np.maximum(gate, 0.0) ** 2 * up
    inverse = np.reciprocal(
        np.sqrt(
            np.mean(activation * activation, axis=1, keepdims=True)
            + artifact.rms_norm_eps
        )
    )
    normalized = (
        activation
        * inverse
        * np.asarray(decoded["ffn_sub_norm"], dtype=np.float32)[None, :]
    )
    coefficients = _activation_quant(normalized)
    down = np.asarray(decoded["down_codes"], dtype=np.float32)
    utility = coefficients * coefficients * np.sum(down * down, axis=0)[None, :]
    # Stable ordering is part of the oracle contract. It is especially
    # important for BitNet because ReLU-squared gating can leave a large
    # exactly-zero tail; argpartition would choose that tail arbitrarily and
    # turn recall into a test of nondeterministic zero ties.
    selected = np.argsort(-utility, axis=1, kind="stable")[:, :top_k]
    membership = np.zeros(utility.shape, dtype=bool)
    np.put_along_axis(membership, selected, True, axis=1)
    return membership


def _trace_mlp_states(trace, layer: int, norm_weight, epsilon: float) -> np.ndarray:
    # Both terms were divided by the same incoming-state RMS at capture time.
    # RMSNorm removes that common scale; only the negligible epsilon scaling
    # differs from the uncaptured raw boundary.
    post_attention = (
        trace.teacher_states[:, layer].astype(np.float32)
        + trace.episodic_outputs[:, layer].astype(np.float32)
    )
    return _rms_norm(post_attention, norm_weight, epsilon)


def evaluate_native_bitnet_low_rank_router(
    package: str | Path,
    training_trace: str | Path,
    validation_trace: str | Path,
    *,
    out: str | Path,
    layers: Sequence[int] = (0, 14, 29),
    top_ks: Sequence[int] = (1728, 1728, 2074),
    rank: int = 128,
    steps: int = 500,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    device: str = "cuda",
    seed: int = 20260726,
) -> dict[str, Any]:
    """Train nonlinear low-rank membership routers and measure held-out recall."""

    if len(layers) != len(top_ks) or not layers:
        raise ValueError("layers and top_ks must be non-empty and aligned")
    if min(rank, steps, batch_size) <= 0 or learning_rate <= 0:
        raise ValueError("router training hyperparameters must be positive")
    package_path = Path(package).resolve()
    train = _load_trajectories(training_trace)
    validation = _load_trajectories(validation_trace)
    if train.manifest["model_hash"] != validation.manifest["model_hash"]:
        raise ValueError("router traces use different teachers")
    if train.manifest["dataset_hash"] == validation.manifest["dataset_hash"]:
        raise ValueError("router training and validation datasets must differ")

    manifest = __import__("json").loads(
        (package_path / "manifest.json").read_text(encoding="utf-8")
    )
    artifact = load_native_bitnet_artifact(
        package_path / manifest["mlp"]["path"]
    )
    hidden = artifact.hidden_size
    width = artifact.intermediate_size
    selected_layers = tuple(int(value) for value in layers)
    selected_top_ks = tuple(int(value) for value in top_ks)
    if any(not 0 <= layer < len(artifact.layers) for layer in selected_layers):
        raise ValueError("router layer is outside the artifact")
    if any(not 0 < value <= width for value in selected_top_ks):
        raise ValueError("router top-K is outside the intermediate width")

    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("router probe requires torch and safetensors") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    weights_path = package_path / manifest["transformer"]["non_mlp_path"]
    norm_weights = {}
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for layer in selected_layers:
            name = f"model.layers.{layer}.post_attention_layernorm.weight"
            norm_weights[layer] = handle.get_tensor(name).float().numpy()
    epsilon = float(manifest["model"]["rms_norm_eps"])
    started = time.perf_counter()
    layer_reports = []

    for layer, top_k in zip(selected_layers, selected_top_ks, strict=True):
        train_states = _trace_mlp_states(train, layer, norm_weights[layer], epsilon)
        validation_states = _trace_mlp_states(
            validation, layer, norm_weights[layer], epsilon
        )
        train_membership = _oracle_membership(
            artifact, layer, train_states, top_k
        )
        validation_membership = _oracle_membership(
            artifact, layer, validation_states, top_k
        )
        mean = train_states.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = train_states.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-5] = 1.0
        train_x = torch.from_numpy((train_states - mean) / scale).to(device)
        train_y = torch.from_numpy(train_membership.astype(np.float32)).to(device)
        validation_x = torch.from_numpy(
            (validation_states - mean) / scale
        ).to(device)

        model = torch.nn.Sequential(
            torch.nn.Linear(hidden, rank, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(rank, width, bias=True),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        positive_weight = torch.tensor(
            (width - top_k) / top_k, dtype=torch.float32, device=device
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + layer)
        final_loss = None
        model.train()
        for _ in range(steps):
            indices = torch.randint(
                len(train_x),
                (min(batch_size, len(train_x)),),
                generator=generator,
            ).to(device)
            logits = model(train_x[indices])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                train_y[indices],
                pos_weight=positive_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())

        model.eval()
        with torch.inference_mode():
            scores = model(validation_x).float().cpu().numpy()
        order = np.argsort(-scores, axis=1, kind="stable")
        candidate_counts = tuple(
            sorted(
                {
                    top_k,
                    min(width, int(math.ceil(1.25 * top_k))),
                    min(width, int(math.ceil(1.5 * top_k))),
                }
            )
        )
        recalls = {}
        for candidates in candidate_counts:
            selected = order[:, :candidates]
            row_recall = (
                np.take_along_axis(validation_membership, selected, axis=1).sum(axis=1)
                / top_k
            )
            traffic = native_bitnet_router_traffic(
                hidden,
                width,
                rank=rank,
                candidate_count=candidates,
            )
            recalls[str(candidates)] = {
                "mean": float(np.mean(row_recall)),
                "minimum": float(np.min(row_recall)),
                "p05": float(np.percentile(row_recall, 5)),
                "meets_95_percent": float(np.mean(row_recall)) >= 0.95,
                "traffic": traffic,
            }
        layer_reports.append(
            {
                "layer": layer,
                "top_k": top_k,
                "rank": rank,
                "training_records": len(train_states),
                "validation_records": len(validation_states),
                "final_training_loss": final_loss,
                "candidate_recall": recalls,
            }
        )

    eligible = all(
        any(
            arm["meets_95_percent"] and arm["traffic"]["passes_45_percent"]
            for arm in report["candidate_recall"].values()
        )
        for report in layer_reports
    )
    result = {
        "experiment": "native_bitnet_nonlinear_low_rank_membership_router",
        "package": str(package_path),
        "artifact_sha256": artifact.payload_sha256,
        "training_trace": {
            "path": str(Path(training_trace).resolve()),
            "dataset_hash": train.manifest["dataset_hash"],
            "records": train.records,
        },
        "validation_trace": {
            "path": str(Path(validation_trace).resolve()),
            "dataset_hash": validation.manifest["dataset_hash"],
            "records": validation.records,
        },
        "configuration": {
            "layers": list(selected_layers),
            "top_ks": list(selected_top_ks),
            "rank": rank,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device,
            "seed": seed,
        },
        "layers": layer_reports,
        "recall_gate": 0.95,
        "eligible_for_all_layer_training": eligible,
        "decision": (
            "expand_router_training_to_all_layers"
            if eligible
            else "reject_or_revise_router_before_causal_work"
        ),
        "milestone_2_status": "blocked",
        "elapsed_seconds": time.perf_counter() - started,
        "trace_reconstruction_caveat": (
            "MLP inputs are reconstructed from normalized stage and attention "
            "traces; RMSNorm removes their shared scale up to epsilon."
        ),
    }
    atomic_json(out, result)
    return result


def evaluate_native_bitnet_dip_router(
    package: str | Path,
    validation_trace: str | Path,
    *,
    out: str | Path,
    layer: int = 14,
    top_k: int = 1728,
    input_fractions: Sequence[float] = (0.25, 0.5, 0.75),
    candidate_multipliers: Sequence[float] = (1.0, 1.25, 1.5),
) -> dict[str, Any]:
    """Screen top-magnitude input-coordinate routing on held-out BitNet states."""

    fractions = tuple(dict.fromkeys(float(value) for value in input_fractions))
    multipliers = tuple(dict.fromkeys(float(value) for value in candidate_multipliers))
    if not fractions or any(not 0 < value <= 1 for value in fractions):
        raise ValueError("input_fractions must lie in (0, 1]")
    if not multipliers or any(value < 1 for value in multipliers):
        raise ValueError("candidate_multipliers must be at least one")
    package_path = Path(package).resolve()
    trace = _load_trajectories(validation_trace)
    import json
    from safetensors import safe_open

    manifest = json.loads((package_path / "manifest.json").read_text(encoding="utf-8"))
    artifact = load_native_bitnet_artifact(package_path / manifest["mlp"]["path"])
    if not 0 <= layer < len(artifact.layers) or not 0 < top_k <= artifact.intermediate_size:
        raise ValueError("layer or top_k is outside the artifact")
    weights_path = package_path / manifest["transformer"]["non_mlp_path"]
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        norm_weight = (
            handle.get_tensor(
                f"model.layers.{layer}.post_attention_layernorm.weight"
            )
            .float()
            .numpy()
        )
    states = _trace_mlp_states(
        trace,
        layer,
        norm_weight,
        float(manifest["model"]["rms_norm_eps"]),
    )
    oracle = _oracle_membership(artifact, layer, states, top_k)
    decoded = decode_native_bitnet_layer(artifact, layer)
    gate_codes = np.asarray(decoded["gate_codes"], dtype=np.float32)
    up_codes = np.asarray(decoded["up_codes"], dtype=np.float32)
    gain_squared = np.asarray(decoded["ffn_sub_norm"], dtype=np.float32) ** 2
    down = np.asarray(decoded["down_codes"], dtype=np.float32)
    down_norm_squared = np.sum(down * down, axis=0)
    quantized = _activation_quant(states)
    hidden = artifact.hidden_size
    width = artifact.intermediate_size
    dense_q4 = (3 * hidden * width + 1) // 2
    started = time.perf_counter()
    arms = []

    for fraction in fractions:
        count = min(hidden, max(1, int(math.ceil(fraction * hidden))))
        selected_coordinates = np.argpartition(
            -np.abs(quantized), count - 1, axis=1
        )[:, :count]
        masked = np.zeros_like(quantized)
        np.put_along_axis(
            masked,
            selected_coordinates,
            np.take_along_axis(quantized, selected_coordinates, axis=1),
            axis=1,
        )
        gate = (
            masked @ gate_codes.T
            * np.asarray(decoded["gate_scale"], dtype=np.float32)
        )
        up = (
            masked @ up_codes.T
            * np.asarray(decoded["up_scale"], dtype=np.float32)
        )
        raw = np.maximum(gate, 0.0) ** 2 * up
        proxy = raw * raw * gain_squared[None, :] * down_norm_squared[None, :]
        order = np.argsort(-proxy, axis=1, kind="stable")
        coordinate_index_bytes = math.ceil(2 * count * width / 5)
        for multiplier in multipliers:
            candidates = min(width, int(math.ceil(multiplier * top_k)))
            selected = order[:, :candidates]
            row_recall = (
                np.take_along_axis(oracle, selected, axis=1).sum(axis=1) / top_k
            )
            candidate_bytes = candidates * (3 * ((hidden + 4) // 5) + 2)
            complete = coordinate_index_bytes + candidate_bytes
            arms.append(
                {
                    "input_fraction": fraction,
                    "input_coordinates": count,
                    "candidate_multiplier": multiplier,
                    "candidate_count": candidates,
                    "candidate_recall": {
                        "mean": float(np.mean(row_recall)),
                        "minimum": float(np.min(row_recall)),
                        "p05": float(np.percentile(row_recall, 5)),
                    },
                    "traffic": {
                        "coordinate_major_gate_up_bytes": coordinate_index_bytes,
                        "complete_candidate_record_bytes": candidate_bytes,
                        "complete_modelled_bytes": complete,
                        "dense_q4_bytes": dense_q4,
                        "fraction_of_dense_q4": complete / dense_q4,
                        "passes_45_percent": complete / dense_q4 <= 0.45,
                        "metadata_included": False,
                    },
                    "meets_joint_screen": (
                        float(np.mean(row_recall)) >= 0.95
                        and complete / dense_q4 <= 0.45
                    ),
                }
            )
    best = max(
        arms,
        key=lambda arm: (
            arm["meets_joint_screen"],
            arm["candidate_recall"]["mean"],
            -arm["traffic"]["fraction_of_dense_q4"],
        ),
    )
    result = {
        "experiment": "native_bitnet_dynamic_input_pruning_router",
        "package": str(package_path),
        "validation_trace": str(Path(validation_trace).resolve()),
        "layer": layer,
        "top_k": top_k,
        "records": len(states),
        "arms": arms,
        "best_arm": best,
        "recall_gate": 0.95,
        "decision": (
            "implement_selected_record_causal_dip"
            if best["meets_joint_screen"]
            else "dip_recall_or_traffic_insufficient"
        ),
        "milestone_2_status": "blocked",
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(out, result)
    return result


__all__ = [
    "evaluate_native_bitnet_low_rank_router",
    "native_bitnet_router_traffic",
    "evaluate_native_bitnet_dip_router",
]
