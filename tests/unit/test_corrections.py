import numpy as np

from engram.corrections import (
    CorrectionCapsule,
    CorrectionManager,
    fit_correction_capsules,
)


def test_capsule_selection_and_uncertainty_escalation():
    capsule = CorrectionCapsule(np.ones(4), 0.2, np.ones((4, 1)) * 0.1, np.ones((1, 4)) * 0.2)
    manager = CorrectionManager([capsule], uncertainty_threshold=0.4)
    decision = manager.decide(np.ones(4), 0.8)
    assert decision.capsule_index == 0
    assert decision.extra_cycles == 1 and decision.expand_vocabulary
    assert not np.array_equal(manager.apply(np.ones(4), decision), np.ones(4))


def test_fitted_capsule_predicts_affine_low_rank_residual():
    rng = np.random.default_rng(17)
    states = rng.normal(size=(64, 6))
    left = rng.normal(size=(6, 2))
    right = rng.normal(size=(2, 6))
    bias = rng.normal(size=6)
    residual = states @ left @ right + bias

    fitted = fit_correction_capsules(
        states,
        residual,
        capsules=1,
        rank=2,
        ridge=0.0,
        radius_scale=2.0,
    )
    predicted, matched = fitted.predict(states)

    assert np.all(matched)
    np.testing.assert_allclose(predicted, residual, rtol=1e-10, atol=1e-10)
    assert fitted.parameter_bytes() > 0


def test_capsule_seeds_prioritize_large_residual_and_fit_is_deterministic():
    rng = np.random.default_rng(23)
    states = rng.normal(size=(20, 4))
    residual = rng.normal(scale=0.01, size=(20, 4))
    residual[7] *= 1000.0

    first = fit_correction_capsules(states, residual, capsules=3, rank=2, ridge=1.0)
    second = fit_correction_capsules(states, residual, capsules=3, rank=2, ridge=1.0)

    assert first.assignments[7] == second.assignments[7]
    np.testing.assert_array_equal(first.assignments, second.assignments)
    for left_capsule, right_capsule in zip(
        first.manager.capsules, second.manager.capsules, strict=True
    ):
        np.testing.assert_allclose(left_capsule.center, right_capsule.center)
        np.testing.assert_allclose(left_capsule.down, right_capsule.down)
        np.testing.assert_allclose(left_capsule.up, right_capsule.up)


def test_capsule_rejects_nonfinite_bias():
    with np.testing.assert_raises_regex(ValueError, "finite"):
        CorrectionCapsule(
            np.ones(3),
            1.0,
            np.ones((3, 1)),
            np.ones((1, 3)),
            bias=np.array([0.0, np.nan, 0.0]),
            centered_input=True,
        )


def test_targeted_capsules_fit_only_high_residual_fraction():
    rng = np.random.default_rng(29)
    states = rng.normal(size=(20, 4))
    residual = np.zeros((20, 4))
    residual[:4] = rng.normal(size=(4, 4))

    fitted = fit_correction_capsules(
        states,
        residual,
        capsules=2,
        rank=1,
        ridge=1.0,
        priority_fraction=0.2,
        radius_quantile=0.8,
    )

    assert np.count_nonzero(fitted.assignments >= 0) == 4
    assert np.all(fitted.assignments[:4] >= 0)
