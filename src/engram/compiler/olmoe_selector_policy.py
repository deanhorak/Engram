"""Compile and authenticate a frozen OLMoE episodic selector artifact.

The artifact contains only the train-fitted key PCA basis and an explicit
evaluator-only policy contract.  It does not enable the selector in the native
runtime, and it cannot authorize a protected evaluation.  A runtime adapter
can load it to construct candidate masks when authenticated query/key traces
are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.evaluation import olmoe_retrieval_episodic_blockwise_qk as qk
from engram.evaluation import olmoe_retrieval_episodic_candidate_selector as selector
from engram.utils import atomic_json, sha256_file


_LAYERS = 16
_HEADS = 16
_DIMENSION = 128
_CANDIDATES = 8
_TOP_K = 4


def _save_basis(path: Path, centers: np.ndarray, components: np.ndarray) -> dict[str, Any]:
    try:
        from safetensors.numpy import save_file
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("selector policy compilation requires safetensors") from error
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    components = np.ascontiguousarray(components, dtype=np.float32)
    save_file({"centers": centers, "components": components}, str(path))
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "tensors": {
            "centers": {
                "shape": list(centers.shape),
                "dtype": str(centers.dtype),
                "sha256": hashlib.sha256(centers.tobytes(order="C")).hexdigest(),
            },
            "components": {
                "shape": list(components.shape),
                "dtype": str(components.dtype),
                "sha256": hashlib.sha256(components.tobytes(order="C")).hexdigest(),
            },
        },
    }


def compile_olmoe_selector_policy(
    *,
    train_candidate_key_manifest: str | Path,
    train_candidate_key_manifest_sha256: str,
    train_candidate_manifest: str | Path,
    train_candidate_manifest_sha256: str,
    out: str | Path,
    rank: int = 16,
    pool_size: int = 6,
    development_replay_sha256: str | None = None,
    long_context_sha256: str | None = None,
    pool_sweep_sha256: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not 0 < rank <= _DIMENSION or not 0 < pool_size <= _CANDIDATES:
        raise ValueError("selector policy rank or pool size is invalid")
    train_keys, key_metadata = qk.full.load_stacked_full_visible_qk_candidate_key_trace(
        train_candidate_key_manifest,
        train_candidate_key_manifest_sha256,
    )
    _, candidate_metadata = qk.full.load_stacked_full_visible_qk_candidate_trace(
        train_candidate_manifest,
        train_candidate_manifest_sha256,
    )
    qk._cross_check_shards(candidate_metadata, key_metadata)
    centers, components = selector.fit_query_key_pca_basis(train_keys, rank=rank)
    output = Path(out).expanduser().resolve()
    if output.exists():
        raise ValueError("selector policy output already exists")
    output.mkdir(parents=True, exist_ok=False)
    basis_metadata = _save_basis(output / "key_pca.safetensors", centers, components)
    report = {
        "schema_version": 1,
        "kind": "engram.olmoe.episodic_selector",
        "status": "compiled_evaluator_only",
        "runtime_mode": "evaluator_only",
        "enabled_by_default": False,
        "selector": {
            "method": "train_key_pca_candidate_pool_exact_rerank",
            "rank": rank,
            "pool_size": pool_size,
            "native_top_k": _TOP_K,
            "older_candidates": _CANDIDATES,
            "layers": _LAYERS,
            "query_heads": _HEADS,
            "head_dimension": _DIMENSION,
        },
        "basis": basis_metadata,
        "training": {
            "split": "train",
            "candidate_key_manifest_sha256": train_candidate_key_manifest_sha256.lower(),
            "candidate_manifest_sha256": train_candidate_manifest_sha256.lower(),
            "development_fit": False,
        },
        "validation": {
            "development_replay_sha256": development_replay_sha256.lower()
            if development_replay_sha256
            else None,
            "long_context_sha256": long_context_sha256.lower()
            if long_context_sha256
            else None,
            "pool_sweep_sha256": pool_sweep_sha256.lower()
            if pool_sweep_sha256
            else None,
            "protected_evaluation_opened": False,
        },
        "artifacts": {
            "candidate_key_manifest": str(Path(train_candidate_key_manifest).resolve()),
            "candidate_manifest": str(Path(train_candidate_manifest).resolve()),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "policy.json", report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-candidate-key-manifest", required=True)
    parser.add_argument("--train-candidate-key-manifest-sha256", required=True)
    parser.add_argument("--train-candidate-manifest", required=True)
    parser.add_argument("--train-candidate-manifest-sha256", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--pool-size", type=int, default=6)
    parser.add_argument("--development-replay-sha256")
    parser.add_argument("--long-context-sha256")
    parser.add_argument("--pool-sweep-sha256")
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = compile_olmoe_selector_policy(
        train_candidate_key_manifest=args.train_candidate_key_manifest,
        train_candidate_key_manifest_sha256=args.train_candidate_key_manifest_sha256,
        train_candidate_manifest=args.train_candidate_manifest,
        train_candidate_manifest_sha256=args.train_candidate_manifest_sha256,
        out=args.out,
        rank=args.rank,
        pool_size=args.pool_size,
        development_replay_sha256=args.development_replay_sha256,
        long_context_sha256=args.long_context_sha256,
        pool_sweep_sha256=args.pool_sweep_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
