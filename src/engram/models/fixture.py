from __future__ import annotations

import json
import io
import zipfile
from pathlib import Path

import numpy as np


def _normal(rng: np.random.Generator, shape: tuple[int, ...], scale: float = 0.08) -> np.ndarray:
    return rng.normal(0.0, scale, size=shape).astype(np.float32)


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an uncompressed NPZ with stable ordering and ZIP metadata."""
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue())


def create_tiny_fixture(
    out: str | Path,
    *,
    seed: int = 7,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    num_layers: int = 2,
    num_heads: int = 4,
    vocab_size: int = 64,
) -> Path:
    """Create deterministic Llama-shaped random weights without network access.

    The fixture validates conversion mechanics only. It is not a trained language model.
    """
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    if hidden_size % num_heads:
        raise ValueError("hidden_size must be divisible by num_heads")
    config = {
        "architectures": ["LlamaForCausalLM"],
        "engram_fixture": True,
        "fixture_seed": seed,
        "hidden_act": "silu",
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "attention_bias": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": hidden_size // num_heads,
        "max_position_embeddings": 256,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": num_heads,
        "num_hidden_layers": num_layers,
        "num_key_value_heads": num_heads,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "vocab_size": vocab_size,
    }
    with (target / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    rng = np.random.default_rng(seed)
    weights: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": _normal(rng, (vocab_size, hidden_size)),
        "model.norm.weight": np.ones(hidden_size, dtype=np.float32),
        "lm_head.weight": _normal(rng, (vocab_size, hidden_size)),
    }
    for layer in range(num_layers):
        base = f"model.layers.{layer}"
        weights[f"{base}.input_layernorm.weight"] = np.ones(hidden_size, dtype=np.float32)
        weights[f"{base}.post_attention_layernorm.weight"] = np.ones(hidden_size, dtype=np.float32)
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weights[f"{base}.self_attn.{projection}.weight"] = _normal(rng, (hidden_size, hidden_size))
        weights[f"{base}.mlp.gate_proj.weight"] = _normal(rng, (intermediate_size, hidden_size))
        weights[f"{base}.mlp.up_proj.weight"] = _normal(rng, (intermediate_size, hidden_size))
        weights[f"{base}.mlp.down_proj.weight"] = _normal(rng, (hidden_size, intermediate_size))
    _write_deterministic_npz(target / "weights.npz", weights)
    return target
