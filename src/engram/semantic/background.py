"""Background operators for diffuse SwiGLU residual contributions.

The fitting target is expected to be the residual left after a sparse semantic
read, rather than the complete FFN output.  Keeping that convention here makes
the operator usable with different top-K policies without coupling it to a
particular router.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


BACKGROUND_FORMAT = "engram.semantic.background"
BACKGROUND_SCHEMA_VERSION = 1


def _matrix(value: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must have shape [samples, features]")
    if result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _predict_input(value: ArrayLike, input_dim: int) -> tuple[np.ndarray, tuple[int, ...]]:
    result = np.asarray(value)
    if result.ndim < 1 or result.shape[-1] != input_dim:
        raise ValueError(f"inputs must have trailing dimension {input_dim}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("inputs must contain only finite values")
    return result, result.shape[:-1]


@dataclass(frozen=True)
class NoBackground:
    """The explicit no-background ablation."""

    input_dim: int
    output_dim: int
    fit_samples: int = 0

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if self.fit_samples < 0:
            raise ValueError("fit_samples must be non-negative")

    @classmethod
    def fit(cls, inputs: ArrayLike, residuals: ArrayLike) -> "NoBackground":
        x = _matrix(inputs, "inputs")
        y = _matrix(residuals, "residuals")
        if x.shape[0] != y.shape[0]:
            raise ValueError("inputs and residuals must have the same sample count")
        return cls(input_dim=x.shape[1], output_dim=y.shape[1], fit_samples=x.shape[0])

    def predict(self, inputs: ArrayLike) -> np.ndarray:
        x, leading_shape = _predict_input(inputs, self.input_dim)
        dtype = np.result_type(x.dtype, np.float32)
        return np.zeros((*leading_shape, self.output_dim), dtype=dtype)

    def metadata(self) -> dict[str, Any]:
        return {
            "format": BACKGROUND_FORMAT,
            "schema_version": BACKGROUND_SCHEMA_VERSION,
            "operator": "none",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "rank": 0,
            "fit_samples": self.fit_samples,
            "storage_dtype": None,
            "tensor_layout": {},
        }

    def tensors(self) -> dict[str, NDArray[np.float64]]:
        return {}

    @classmethod
    def from_state(
        cls, metadata: Mapping[str, Any], tensors: Mapping[str, ArrayLike]
    ) -> "NoBackground":
        _validate_common_metadata(metadata, operator="none")
        if tensors:
            raise ValueError("no-background state must not contain tensors")
        return cls(
            input_dim=int(metadata["input_dim"]),
            output_dim=int(metadata["output_dim"]),
            fit_samples=int(metadata.get("fit_samples", 0)),
        )


@dataclass(frozen=True)
class LowRankLinearBackground:
    """Ridge-fitted, rank-truncated linear model of the FFN residual.

    The fitted prediction is::

        output_mean + (hidden - input_mean) @ input_factor @ output_factor

    Ridge regression is performed before the coefficient matrix is truncated by
    SVD.  NumPy's deterministic dense linear algebra and the absence of random
    initialization make repeated fits deterministic for fixed inputs.
    """

    input_mean: NDArray[np.float64]
    output_mean: NDArray[np.float64]
    input_factor: NDArray[np.float64]
    output_factor: NDArray[np.float64]
    ridge: float
    fit_samples: int

    def __post_init__(self) -> None:
        input_mean = np.asarray(self.input_mean, dtype=np.float64)
        output_mean = np.asarray(self.output_mean, dtype=np.float64)
        input_factor = np.asarray(self.input_factor, dtype=np.float64)
        output_factor = np.asarray(self.output_factor, dtype=np.float64)
        if input_mean.ndim != 1 or output_mean.ndim != 1:
            raise ValueError("means must be rank-1")
        if input_factor.ndim != 2 or output_factor.ndim != 2:
            raise ValueError("factors must be rank-2")
        rank = input_factor.shape[1]
        if input_factor.shape[0] != input_mean.size:
            raise ValueError("input factor and mean dimensions differ")
        if output_factor.shape != (rank, output_mean.size):
            raise ValueError("output factor has incompatible dimensions")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if self.ridge < 0.0 or not np.isfinite(self.ridge):
            raise ValueError("ridge must be finite and non-negative")
        if self.fit_samples <= 0:
            raise ValueError("fit_samples must be positive")
        for name, value in (
            ("input_mean", input_mean),
            ("output_mean", output_mean),
            ("input_factor", input_factor),
            ("output_factor", output_factor),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
            value.setflags(write=False)
        object.__setattr__(self, "input_mean", input_mean)
        object.__setattr__(self, "output_mean", output_mean)
        object.__setattr__(self, "input_factor", input_factor)
        object.__setattr__(self, "output_factor", output_factor)

    @property
    def input_dim(self) -> int:
        return int(self.input_mean.size)

    @property
    def output_dim(self) -> int:
        return int(self.output_mean.size)

    @property
    def rank(self) -> int:
        return int(self.input_factor.shape[1])

    @classmethod
    def fit(
        cls,
        inputs: ArrayLike,
        residuals: ArrayLike,
        *,
        rank: int,
        ridge: float = 1e-6,
    ) -> "LowRankLinearBackground":
        x = _matrix(inputs, "inputs")
        y = _matrix(residuals, "residuals")
        if x.shape[0] != y.shape[0]:
            raise ValueError("inputs and residuals must have the same sample count")
        max_rank = min(x.shape[1], y.shape[1])
        if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
            raise ValueError("rank must be an integer")
        if rank <= 0 or rank > max_rank:
            raise ValueError(f"rank must lie in [1, {max_rank}]")
        if ridge < 0.0 or not np.isfinite(ridge):
            raise ValueError("ridge must be finite and non-negative")

        input_mean = x.mean(axis=0)
        output_mean = y.mean(axis=0)
        centered_x = x - input_mean
        centered_y = y - output_mean
        if ridge == 0.0:
            coefficient, _, _, _ = np.linalg.lstsq(centered_x, centered_y, rcond=None)
        else:
            gram = centered_x.T @ centered_x
            gram.flat[:: gram.shape[0] + 1] += ridge
            coefficient = np.linalg.solve(gram, centered_x.T @ centered_y)

        left, singular_values, right = np.linalg.svd(coefficient, full_matrices=False)
        input_factor = left[:, :rank] * singular_values[:rank]
        output_factor = right[:rank, :]
        return cls(
            input_mean=input_mean,
            output_mean=output_mean,
            input_factor=input_factor,
            output_factor=output_factor,
            ridge=float(ridge),
            fit_samples=x.shape[0],
        )

    def predict(self, inputs: ArrayLike) -> np.ndarray:
        x, leading_shape = _predict_input(inputs, self.input_dim)
        flat = np.asarray(x, dtype=np.float64).reshape(-1, self.input_dim)
        prediction = (flat - self.input_mean) @ self.input_factor @ self.output_factor
        prediction += self.output_mean
        return prediction.reshape(*leading_shape, self.output_dim)

    def metadata(self) -> dict[str, Any]:
        return {
            "format": BACKGROUND_FORMAT,
            "schema_version": BACKGROUND_SCHEMA_VERSION,
            "operator": "low_rank_linear_residual",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "rank": self.rank,
            "ridge": self.ridge,
            "fit_intercept": True,
            "fit_samples": self.fit_samples,
            "storage_dtype": "float64",
            "tensor_layout": {
                "input_mean": [self.input_dim],
                "output_mean": [self.output_dim],
                "input_factor": [self.input_dim, self.rank],
                "output_factor": [self.rank, self.output_dim],
            },
        }

    def tensors(self) -> dict[str, NDArray[np.float64]]:
        return {
            "input_mean": self.input_mean.copy(),
            "output_mean": self.output_mean.copy(),
            "input_factor": self.input_factor.copy(),
            "output_factor": self.output_factor.copy(),
        }

    @classmethod
    def from_state(
        cls, metadata: Mapping[str, Any], tensors: Mapping[str, ArrayLike]
    ) -> "LowRankLinearBackground":
        _validate_common_metadata(metadata, operator="low_rank_linear_residual")
        expected = {"input_mean", "output_mean", "input_factor", "output_factor"}
        if set(tensors) != expected:
            raise ValueError(f"background tensors must be exactly {sorted(expected)}")
        result = cls(
            input_mean=np.asarray(tensors["input_mean"], dtype=np.float64),
            output_mean=np.asarray(tensors["output_mean"], dtype=np.float64),
            input_factor=np.asarray(tensors["input_factor"], dtype=np.float64),
            output_factor=np.asarray(tensors["output_factor"], dtype=np.float64),
            ridge=float(metadata["ridge"]),
            fit_samples=int(metadata["fit_samples"]),
        )
        if result.metadata() != dict(metadata):
            raise ValueError("background metadata does not match tensor dimensions")
        return result


def _validate_common_metadata(metadata: Mapping[str, Any], *, operator: str) -> None:
    if metadata.get("format") != BACKGROUND_FORMAT:
        raise ValueError("unsupported background format")
    if metadata.get("schema_version") != BACKGROUND_SCHEMA_VERSION:
        raise ValueError("unsupported background schema version")
    if metadata.get("operator") != operator:
        raise ValueError(f"expected {operator!r} background operator")
