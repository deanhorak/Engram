"""Dependency-free calibration metrics for executed Oracle actions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .models import _identifier, _nonnegative, _probability


@dataclass(frozen=True)
class PredictionOutcome:
    attempt_id: str
    strategy: str
    worker_id: str
    worker_generation: int
    predictor_id: str
    predicted_success: float
    actual_success: float | None
    predicted_information_gain: float
    actual_information_gain: float | None
    predicted_latency_seconds: float
    actual_latency_seconds: float | None
    predicted_token_cost: int
    actual_token_cost: int | None
    predicted_compute_cost: float
    actual_compute_cost: float | None

    def __post_init__(self) -> None:
        for value, name in (
            (self.attempt_id, "attempt_id"),
            (self.strategy, "strategy"),
            (self.worker_id, "worker_id"),
            (self.predictor_id, "predictor_id"),
        ):
            _identifier(value, name)
        if (
            isinstance(self.worker_generation, bool)
            or not isinstance(self.worker_generation, int)
            or self.worker_generation <= 0
        ):
            raise ValueError("worker_generation must be a positive integer")
        _probability(self.predicted_success, "predicted_success")
        _probability(self.predicted_information_gain, "predicted_information_gain")
        if self.actual_success is not None and self.actual_success not in {0.0, 1.0}:
            raise ValueError("actual_success must be zero, one, or None")
        if self.actual_information_gain is not None:
            _probability(self.actual_information_gain, "actual_information_gain")
        _nonnegative(self.predicted_latency_seconds, "predicted_latency_seconds")
        _nonnegative(self.predicted_compute_cost, "predicted_compute_cost")
        if self.actual_latency_seconds is not None:
            _nonnegative(self.actual_latency_seconds, "actual_latency_seconds")
        if self.actual_compute_cost is not None:
            _nonnegative(self.actual_compute_cost, "actual_compute_cost")
        for value, name in (
            (self.predicted_token_cost, "predicted_token_cost"),
            (self.actual_token_cost, "actual_token_cost"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")


@dataclass(frozen=True)
class ProbabilityCalibration:
    count: int
    mean_squared_error: float | None
    mean_absolute_error: float | None
    bias: float | None
    expected_calibration_error: float | None
    observed_mean: float | None


@dataclass(frozen=True)
class CostCalibration:
    count: int
    mean_absolute_error: float | None
    bias: float | None
    mean_absolute_log_error: float | None = None


@dataclass(frozen=True)
class CalibrationSummary:
    samples: int
    success: ProbabilityCalibration
    information_gain: ProbabilityCalibration
    latency: CostCalibration
    tokens: CostCalibration
    compute: CostCalibration


def _probability_metrics(
    pairs: list[tuple[float, float]], bins: int
) -> ProbabilityCalibration:
    if not pairs:
        return ProbabilityCalibration(0, None, None, None, None, None)
    errors = [actual - predicted for predicted, actual in pairs]
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for predicted, actual in pairs:
        buckets[min(int(predicted * bins), bins - 1)].append((predicted, actual))
    ece = math.fsum(
        len(bucket)
        / len(pairs)
        * abs(
            math.fsum(actual for _, actual in bucket) / len(bucket)
            - math.fsum(predicted for predicted, _ in bucket) / len(bucket)
        )
        for bucket in buckets
        if bucket
    )
    return ProbabilityCalibration(
        count=len(pairs),
        mean_squared_error=math.fsum(error * error for error in errors) / len(errors),
        mean_absolute_error=math.fsum(abs(error) for error in errors) / len(errors),
        bias=math.fsum(errors) / len(errors),
        expected_calibration_error=ece,
        observed_mean=math.fsum(actual for _, actual in pairs) / len(pairs),
    )


def _cost_metrics(
    pairs: list[tuple[float, float]], *, logarithmic: bool = False
) -> CostCalibration:
    if not pairs:
        return CostCalibration(0, None, None, None)
    errors = [actual - predicted for predicted, actual in pairs]
    log_error = (
        math.fsum(abs(math.log1p(actual) - math.log1p(predicted)) for predicted, actual in pairs)
        / len(pairs)
        if logarithmic
        else None
    )
    return CostCalibration(
        count=len(pairs),
        mean_absolute_error=math.fsum(abs(error) for error in errors) / len(errors),
        bias=math.fsum(errors) / len(errors),
        mean_absolute_log_error=log_error,
    )


def summarize_calibration(
    records: Iterable[PredictionOutcome], *, bins: int = 10
) -> CalibrationSummary:
    values = tuple(records)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    success = [
        (record.predicted_success, record.actual_success)
        for record in values
        if record.actual_success is not None
    ]
    information = [
        (record.predicted_information_gain, record.actual_information_gain)
        for record in values
        if record.actual_information_gain is not None
    ]
    latency = [
        (record.predicted_latency_seconds, record.actual_latency_seconds)
        for record in values
        if record.actual_latency_seconds is not None
    ]
    tokens = [
        (float(record.predicted_token_cost), float(record.actual_token_cost))
        for record in values
        if record.actual_token_cost is not None
    ]
    compute = [
        (record.predicted_compute_cost, record.actual_compute_cost)
        for record in values
        if record.actual_compute_cost is not None
    ]
    return CalibrationSummary(
        samples=len(values),
        success=_probability_metrics(success, bins),
        information_gain=_probability_metrics(information, bins),
        latency=_cost_metrics(latency, logarithmic=True),
        tokens=_cost_metrics(tokens),
        compute=_cost_metrics(compute),
    )


__all__ = [
    "CalibrationSummary",
    "CostCalibration",
    "PredictionOutcome",
    "ProbabilityCalibration",
    "summarize_calibration",
]
