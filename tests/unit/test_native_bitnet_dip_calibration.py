import numpy as np

from engram.evaluation.native_bitnet_dip_calibration import (
    _fit_output_scale,
    _row_relative_l2,
)


def test_output_scale_recovers_multiplicative_local_residual():
    prediction = np.asarray([[1.0, 2.0], [-1.0, 0.5]])
    target = 1.5 * prediction

    scale = _fit_output_scale(prediction, target)

    assert scale == 1.5
    np.testing.assert_allclose(
        _row_relative_l2(scale * prediction, target),
        0.0,
    )


def test_output_scale_is_identity_for_zero_prediction():
    prediction = np.zeros((2, 3))
    target = np.ones((2, 3))

    assert _fit_output_scale(prediction, target) == 1.0
