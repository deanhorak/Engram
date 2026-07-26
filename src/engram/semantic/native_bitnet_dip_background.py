"""Conditional background capsules for adaptive native-BitNet DIP.

The adaptive DIP selector already computes whether its requested cumulative
candidate-energy target was attained before the per-token K clamp.  That
boolean is a useful, free trigger for a deliberately narrow correction:

* do nothing when the adaptive selection attained its target; and
* add one learned hidden-width residual vector when the K clamp prevented it.

The capsule never reads an omitted down record.  It is therefore materially
cheaper than a general low-rank proxy-to-hidden map and, unlike an unconditional
background, cannot perturb the large exact-zero residual population.

This module contains the small inference primitive and its validation-only
fitter.  It does not decide which layer deserves a capsule; that decision must
come from sequence-disjoint evidence and causal confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


_CAPSULE_HEADER_BYTES = 64
_BF16_BYTES = 2


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _bf16_round(values: ArrayLike) -> NDArray[np.float32]:
    """Round float32 values to BF16 using ties-to-even."""

    source = np.asarray(values, dtype=np.float32)
    bits = source.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + (
        (bits >> np.uint32(16)) & np.uint32(1)
    )
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


@dataclass(frozen=True)
class NativeBitNetConditionalBackground:
    """One BF16 residual capsule gated by adaptive target attainment."""

    layer: int
    residual: NDArray[np.float32]
    fitting_trigger_count: int
    trigger: str = "adaptive_candidate_energy_target_unattained"

    def __post_init__(self) -> None:
        residual = np.asarray(self.residual, dtype=np.float32)
        if (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, int)
            or self.layer < 0
        ):
            raise ValueError("layer must be a non-negative integer")
        if (
            residual.ndim != 1
            or residual.size == 0
            or not np.all(np.isfinite(residual))
        ):
            raise ValueError("residual must be a non-empty finite vector")
        if (
            isinstance(self.fitting_trigger_count, bool)
            or not isinstance(self.fitting_trigger_count, int)
            or self.fitting_trigger_count <= 0
        ):
            raise ValueError("fitting_trigger_count must be positive")
        object.__setattr__(self, "residual", _bf16_round(residual))

    @property
    def hidden_size(self) -> int:
        return int(self.residual.size)

    def apply(
        self,
        selected_output: ArrayLike,
        *,
        target_attained: ArrayLike,
    ) -> NDArray[np.float32]:
        """Add the capsule only to rows whose adaptive target was missed."""

        output = np.asarray(selected_output, dtype=np.float32)
        attained = np.asarray(target_attained)
        if output.ndim < 1 or output.shape[-1] != self.hidden_size:
            raise ValueError(
                "selected_output must end in the capsule hidden dimension"
            )
        if attained.dtype != np.bool_:
            raise ValueError("target_attained must be boolean")
        if attained.shape != output.shape[:-1]:
            raise ValueError(
                "target_attained must match selected_output leading dimensions"
            )
        return output + (~attained)[..., None] * self.residual

    def traffic(
        self,
        *,
        cache_line_bytes: int = 64,
    ) -> dict[str, int | str]:
        """Return conservative serialized and triggered cold-read bytes."""

        if (
            isinstance(cache_line_bytes, bool)
            or not isinstance(cache_line_bytes, int)
            or cache_line_bytes < 64
            or cache_line_bytes % 64
        ):
            raise ValueError(
                "cache_line_bytes must be a positive multiple of 64"
            )
        payload = self.hidden_size * _BF16_BYTES
        payload_block = _align(payload, cache_line_bytes)
        header_block = _align(_CAPSULE_HEADER_BYTES, cache_line_bytes)
        complete = header_block + payload_block
        return {
            "format": "native_bitnet_conditional_background_v1",
            "header_bytes": _CAPSULE_HEADER_BYTES,
            "header_block_bytes": header_block,
            "bf16_payload_bytes": payload,
            "payload_block_bytes": payload_block,
            "serialized_bytes": complete,
            "worst_case_triggered_cold_bytes": complete,
            "omitted_down_record_bytes": 0,
        }


def fit_native_bitnet_conditional_background(
    dense_output: ArrayLike,
    selected_output: ArrayLike,
    *,
    target_attained: ArrayLike,
    layer: int,
) -> NativeBitNetConditionalBackground:
    """Fit the mean missed-target residual and immediately BF16-quantize it."""

    dense = np.asarray(dense_output, dtype=np.float32)
    selected = np.asarray(selected_output, dtype=np.float32)
    attained = np.asarray(target_attained)
    if (
        dense.shape != selected.shape
        or dense.ndim != 2
        or dense.shape[1] == 0
        or not np.all(np.isfinite(dense))
        or not np.all(np.isfinite(selected))
    ):
        raise ValueError(
            "dense_output and selected_output must be aligned finite matrices"
        )
    if attained.dtype != np.bool_ or attained.shape != dense.shape[:1]:
        raise ValueError(
            "target_attained must be a boolean vector aligned to output rows"
        )
    triggered = ~attained
    trigger_count = int(np.count_nonzero(triggered))
    if trigger_count == 0:
        raise ValueError("at least one missed-target row is required to fit")
    residual = np.mean(dense[triggered] - selected[triggered], axis=0)
    return NativeBitNetConditionalBackground(
        layer=layer,
        residual=residual,
        fitting_trigger_count=trigger_count,
    )


__all__ = [
    "NativeBitNetConditionalBackground",
    "fit_native_bitnet_conditional_background",
]
