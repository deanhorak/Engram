import numpy as np
import pytest

from engram.semantic.native_bitnet_dip_background import (
    NativeBitNetConditionalBackground,
    fit_native_bitnet_conditional_background,
)


def test_conditional_background_only_changes_missed_target_rows():
    capsule = NativeBitNetConditionalBackground(
        layer=7,
        residual=np.asarray([0.5, -0.25], dtype=np.float32),
        fitting_trigger_count=3,
    )
    selected = np.asarray(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=np.float32,
    )

    actual = capsule.apply(
        selected,
        target_attained=np.asarray([True, False, True]),
    )

    np.testing.assert_array_equal(actual[0], selected[0])
    np.testing.assert_array_equal(actual[2], selected[2])
    np.testing.assert_allclose(actual[1], [3.5, 3.75])


def test_fitter_uses_only_missed_target_residuals_and_rounds_to_bf16():
    selected = np.zeros((4, 2), dtype=np.float32)
    dense = np.asarray(
        [[100.0, 100.0], [1.001, -2.002], [100.0, 100.0], [3.003, -4.004]],
        dtype=np.float32,
    )

    capsule = fit_native_bitnet_conditional_background(
        dense,
        selected,
        target_attained=np.asarray([True, False, True, False]),
        layer=7,
    )

    assert capsule.layer == 7
    assert capsule.fitting_trigger_count == 2
    np.testing.assert_allclose(capsule.residual, [2.0, -3.0], atol=0.02)


def test_hidden_2560_capsule_is_one_header_plus_80_cache_lines():
    capsule = NativeBitNetConditionalBackground(
        layer=7,
        residual=np.zeros(2560, dtype=np.float32),
        fitting_trigger_count=8,
    )

    traffic = capsule.traffic()

    assert traffic["header_block_bytes"] == 64
    assert traffic["bf16_payload_bytes"] == 5120
    assert traffic["serialized_bytes"] == 5184
    assert traffic["worst_case_triggered_cold_bytes"] == 5184
    assert traffic["omitted_down_record_bytes"] == 0


def test_conditional_background_rejects_misaligned_trigger():
    capsule = NativeBitNetConditionalBackground(
        layer=0,
        residual=np.ones(2, dtype=np.float32),
        fitting_trigger_count=1,
    )

    with pytest.raises(ValueError, match="leading dimensions"):
        capsule.apply(
            np.ones((2, 2), dtype=np.float32),
            target_attained=np.asarray([True]),
        )
