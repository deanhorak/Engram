"""Load and apply a frozen evaluator-only OLMoE selector artifact."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from engram.evaluation.olmoe_retrieval_episodic_candidate_selector import (
    build_query_key_masks_from_pca_basis,
)
from engram.utils import sha256_file


class OLMoESelectorPolicyError(ValueError):
    """Raised when a selector policy artifact fails closed validation."""


@dataclass(frozen=True)
class OLMoESelectorPolicy:
    """Validated policy and train-fitted basis for evaluator mask generation."""

    policy_path: Path
    metadata: dict[str, Any]
    centers: np.ndarray
    components: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.metadata["selector"]["rank"])

    @property
    def pool_size(self) -> int:
        return int(self.metadata["selector"]["pool_size"])

    @property
    def evaluator_only(self) -> bool:
        return bool(self.metadata["runtime_mode"] == "evaluator_only")

    def build_masks(
        self,
        evaluation_queries_pre_rope: np.ndarray,
        evaluation_candidate_keys: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Build masks without changing native runtime defaults."""

        if not self.evaluator_only or self.metadata.get("enabled_by_default") is not False:
            raise OLMoESelectorPolicyError(
                "selector policy is not an evaluator-only disabled-by-default artifact"
            )
        return build_query_key_masks_from_pca_basis(
            evaluation_queries_pre_rope,
            evaluation_candidate_keys,
            positions,
            self.centers,
            self.components,
            pool_size=self.pool_size,
        )


def _tensor_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def load_olmoe_selector_policy(path: str | Path) -> OLMoESelectorPolicy:
    """Load a policy and verify its local basis file and tensor contracts."""

    policy_path = Path(path).expanduser().resolve()
    try:
        metadata = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OLMoESelectorPolicyError("selector policy metadata is unreadable") from error
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != 1
        or metadata.get("kind") != "engram.olmoe.episodic_selector"
        or metadata.get("runtime_mode") != "evaluator_only"
        or metadata.get("enabled_by_default") is not False
        or metadata.get("training", {}).get("development_fit") is not False
        or metadata.get("validation", {}).get("protected_evaluation_opened") is not False
    ):
        raise OLMoESelectorPolicyError("selector policy contract changed")
    selector_metadata = metadata.get("selector")
    if not isinstance(selector_metadata, dict):
        raise OLMoESelectorPolicyError("selector policy geometry is missing")
    rank = selector_metadata.get("rank")
    pool_size = selector_metadata.get("pool_size")
    if (
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or not 0 < rank <= 128
        or not isinstance(pool_size, int)
        or isinstance(pool_size, bool)
        or not 0 < pool_size <= 8
        or selector_metadata.get("native_top_k") != 4
        or selector_metadata.get("older_candidates") != 8
        or selector_metadata.get("layers") != 16
        or selector_metadata.get("query_heads") != 16
        or selector_metadata.get("head_dimension") != 128
    ):
        raise OLMoESelectorPolicyError("selector policy geometry changed")
    basis = metadata.get("basis")
    if not isinstance(basis, dict) or not isinstance(basis.get("path"), str):
        raise OLMoESelectorPolicyError("selector policy basis descriptor is missing")
    basis_name = Path(basis["path"])
    if basis_name.is_absolute() or basis_name.name != basis["path"]:
        raise OLMoESelectorPolicyError("selector policy basis path must be local")
    basis_path = policy_path.parent / basis_name
    if not basis_path.is_file() or sha256_file(basis_path) != basis.get("sha256"):
        raise OLMoESelectorPolicyError("selector policy basis digest changed")
    try:
        from safetensors.numpy import load_file
    except ImportError as error:  # pragma: no cover
        raise OLMoESelectorPolicyError("selector policy requires safetensors") from error
    loaded = load_file(basis_path)
    centers = np.ascontiguousarray(loaded.get("centers"), dtype=np.float64)
    components = np.ascontiguousarray(loaded.get("components"), dtype=np.float64)
    if (
        centers.shape != (16, 16, 128)
        or components.shape != (16, 16, 128, rank)
        or not np.isfinite(centers).all()
        or not np.isfinite(components).all()
    ):
        raise OLMoESelectorPolicyError("selector policy basis shape or values changed")
    tensor_metadata = basis.get("tensors", {})
    if (
        tensor_metadata.get("centers", {}).get("sha256") != _tensor_hash(centers.astype(np.float32))
        or tensor_metadata.get("components", {}).get("sha256")
        != _tensor_hash(components.astype(np.float32))
    ):
        raise OLMoESelectorPolicyError("selector policy tensor digest changed")
    return OLMoESelectorPolicy(policy_path, metadata, centers, components)


__all__ = [
    "OLMoESelectorPolicy",
    "OLMoESelectorPolicyError",
    "load_olmoe_selector_policy",
]
