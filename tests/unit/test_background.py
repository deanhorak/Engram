import numpy as np
import pytest

from engram.semantic.background import LowRankLinearBackground, NoBackground


def test_low_rank_background_reconstructs_constructed_residual_and_beats_none():
    rng = np.random.default_rng(41)
    inputs = rng.normal(size=(256, 9))
    validation_inputs = rng.normal(size=(64, 9))
    left = rng.normal(size=(9, 2))
    right = rng.normal(size=(2, 7))
    bias = rng.normal(size=7)
    residuals = inputs @ left @ right + bias
    validation_residuals = validation_inputs @ left @ right + bias

    none = NoBackground.fit(inputs, residuals)
    fitted = LowRankLinearBackground.fit(inputs, residuals, rank=2, ridge=0.0)
    none_error = np.linalg.norm(validation_residuals - none.predict(validation_inputs))
    fitted_error = np.linalg.norm(validation_residuals - fitted.predict(validation_inputs))

    assert fitted_error < none_error * 1e-10
    np.testing.assert_allclose(
        fitted.predict(validation_inputs), validation_residuals, rtol=1e-11, atol=1e-11
    )


def test_fit_is_deterministic_and_state_round_trips():
    rng = np.random.default_rng(7)
    inputs = rng.normal(size=(80, 5))
    residuals = inputs @ rng.normal(size=(5, 4)) + rng.normal(size=4)

    first = LowRankLinearBackground.fit(inputs, residuals, rank=3, ridge=1e-4)
    second = LowRankLinearBackground.fit(inputs, residuals, rank=3, ridge=1e-4)
    assert first.metadata() == second.metadata()
    for name in first.tensors():
        np.testing.assert_array_equal(first.tensors()[name], second.tensors()[name])

    restored = LowRankLinearBackground.from_state(first.metadata(), first.tensors())
    np.testing.assert_array_equal(restored.predict(inputs), first.predict(inputs))
    assert restored.metadata() == first.metadata()


def test_no_background_shape_and_state_metadata():
    inputs = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    operator = NoBackground(input_dim=4, output_dim=6, fit_samples=12)
    prediction = operator.predict(inputs)

    assert prediction.shape == (2, 3, 6)
    assert prediction.dtype == np.float32
    assert np.count_nonzero(prediction) == 0
    assert NoBackground.from_state(operator.metadata(), {}).metadata() == operator.metadata()


def test_fit_validation_rejects_invalid_rank_and_mismatched_samples():
    inputs = np.ones((4, 3))
    residuals = np.ones((4, 2))
    with pytest.raises(ValueError, match="rank must lie"):
        LowRankLinearBackground.fit(inputs, residuals, rank=3)
    with pytest.raises(ValueError, match="sample count"):
        LowRankLinearBackground.fit(inputs, residuals[:3], rank=1)
