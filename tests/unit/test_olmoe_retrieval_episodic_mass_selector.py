from __future__ import annotations

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_mass_selector as selector


def _shape() -> selector.SelectorShape:
    return selector.SelectorShape(
        layers=1,
        heads=1,
        components=3,
        head_dimension=2,
        rank=2,
    )


def _mass() -> tuple[np.ndarray, np.ndarray]:
    mass = np.asarray([[[[0.25, 0.75, 0.0]]]], dtype=np.float32)
    valid = np.asarray([[[[True, True, False]]]])
    return mass, valid


def test_zero_initialized_selector_is_exact_native_baseline() -> None:
    shape = _shape()
    mass, valid = _mass()
    parameters = selector.initialize_parameters(shape, seed=17)
    coefficients, delta = selector.selector_forward(
        mass,
        valid,
        parameters,
        shape,
    )
    np.testing.assert_allclose(coefficients, mass, atol=1.0e-7, rtol=0.0)
    np.testing.assert_array_equal(delta, np.zeros_like(delta))
    assert coefficients[..., 2].item() == 0.0


def test_mass_and_score_routes_are_equivalent() -> None:
    mass, valid = _mass()
    delta = np.asarray([[[[0.5, -0.25, 0.0]]]], dtype=np.float32)
    score = np.full(mass.shape, -123.0, dtype=np.float64)
    score[valid] = np.log(mass[valid]) + 9.5
    from_mass = selector.coefficients_from_mass(mass, valid, delta)
    from_score = selector.coefficients_from_scores(score, valid, delta)
    np.testing.assert_allclose(from_mass, from_score, atol=1.0e-6, rtol=0.0)


def test_mass_and_reconstructed_score_paths_are_equivalent_in_fp32() -> None:
    shape = _shape()
    mass, valid = _mass()
    initialized = selector.initialize_parameters(shape, seed=29)
    parameters = selector.SelectorParameters(
        U=initialized.U,
        V=np.asarray(
            [[[0.4, -0.2, 0.1], [-0.3, 0.5, 0.2]]],
            dtype=np.float32,
        ),
        E=np.asarray([[[0.2, -0.1]]], dtype=np.float32),
        B=np.asarray([[[0.1, -0.2, 7.0]]], dtype=np.float32),
    )
    score = np.zeros(mass.shape, dtype=np.float32)
    score[valid] = np.log(mass[valid]) + np.float32(6.25)
    mass_coefficients, mass_delta = selector.selector_forward(
        mass,
        valid,
        parameters,
        shape,
    )
    score_coefficients, score_delta = selector.selector_forward_from_scores(
        score,
        valid,
        parameters,
        shape,
    )
    assert mass_coefficients.dtype == np.float32
    assert score_coefficients.dtype == np.float32
    np.testing.assert_allclose(mass_delta, score_delta, atol=1.0e-6, rtol=0.0)
    np.testing.assert_allclose(
        mass_coefficients,
        score_coefficients,
        atol=1.0e-6,
        rtol=0.0,
    )


def test_features_mask_padding_and_gauge_delta() -> None:
    shape = _shape()
    mass, valid = _mass()
    parameters = selector.SelectorParameters(
        U=np.zeros((1, 3, 2), dtype=np.float32),
        V=np.zeros((1, 2, 3), dtype=np.float32),
        E=np.zeros((1, 1, 2), dtype=np.float32),
        B=np.asarray([[[4.0, 2.0, 100.0]]], dtype=np.float32),
    )
    features = selector.centered_log_mass_features(mass, valid, shape)
    coefficients, delta = selector.selector_forward(
        mass,
        valid,
        parameters,
        shape,
    )
    assert features[..., 2].item() == 0.0
    np.testing.assert_allclose(
        np.mean(delta[..., :2], axis=-1),
        0.0,
        atol=0.0,
        rtol=0.0,
    )
    assert delta[..., 2].item() == 0.0
    assert coefficients[..., 2].item() == 0.0


def test_bf16_round_to_nearest_even_and_replay() -> None:
    shape = _shape()
    parameters = selector.initialize_parameters(shape, seed=23)
    decoded, bits = selector.quantize_parameters_bf16(parameters, shape)
    assert sum(value.nbytes for value in bits.values()) == shape.parameter_count * 2
    decoded.validate(shape)
    for name, value in parameters.as_dict().items():
        expected = (
            np.asarray(value, dtype=np.float32)
            .view(np.uint32)
            .astype(np.uint32, copy=False)
        )
        assert bits[name].dtype == np.uint16
        assert expected.shape == bits[name].shape
    mass, valid = _mass()
    first, _ = selector.selector_forward(mass, valid, decoded, shape)
    second, _ = selector.selector_forward(mass, valid, decoded, shape)
    np.testing.assert_array_equal(first, second)


def test_production_resource_contract_is_exact() -> None:
    resource = selector.production_resource_contract()
    assert resource["parameter_count"] == 25_600
    assert resource["parameter_bytes"] == 51_200
    assert resource["selector_weight_traffic_bytes_per_128_tokens"] == 6_553_600
    assert resource["combined_logical_traffic_bytes_per_128_tokens"] == 721_420_288
    assert resource["fraction_of_dense"] == pytest.approx(1.0 / 3.0)
    assert resource["remaining_bytes_below_45_percent_floor"] == 252_497_100
    assert resource["selector_scratch_bytes"] == 6_400
    assert resource["selector_macs_per_token"] == 229_376


def test_fixed_folds_are_record_disjoint_and_cover_all_records() -> None:
    folds = selector.fixed_folds()
    heldout = []
    for training, testing in folds:
        assert set(training).isdisjoint(testing)
        assert len(training) == 6
        heldout.extend(testing)
    assert sorted(heldout) == list(range(8))
    assert [testing for _training, testing in folds] == [
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]


def test_schedule_is_deterministic_and_layer_independent() -> None:
    first = selector.build_training_schedule(
        (0, 2),
        read_positions=4,
        layers=3,
        epochs=2,
        rows_per_layer_per_step=2,
        seed=41,
    )
    second = selector.build_training_schedule(
        (0, 2),
        read_positions=4,
        layers=3,
        epochs=2,
        rows_per_layer_per_step=2,
        seed=41,
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (8, 3, 2)
    assert selector.schedule_sha256(first) == selector.schedule_sha256(second)
    # Each layer sees every row exactly once in each four-step epoch.
    expected = [0, 1, 2, 3, 8, 9, 10, 11]
    for layer in range(3):
        assert sorted(first[:4, layer].reshape(-1).tolist()) == expected


def test_direct_post_wo_error_matches_manual_scalar_case() -> None:
    shape = selector.SelectorShape(
        layers=1,
        heads=1,
        components=2,
        head_dimension=1,
        rank=1,
    )
    coefficients = np.asarray([[[[0.25, 0.75]]]], dtype=np.float64)
    values = np.asarray([[[[[0.0], [2.0]]]]], dtype=np.float64)
    base = np.asarray([[[[1.0]]]], dtype=np.float64)
    target = np.asarray([[[1.0]]], dtype=np.float64)
    projection = np.asarray([[[2.0]]], dtype=np.float64)
    error, energy = selector.direct_post_wo_error_energy(
        coefficients,
        values,
        base,
        target,
        projection,
        shape,
    )
    # Selected value 1.5, pre-Wo delta .5, projected correction 1.0.
    np.testing.assert_allclose(error, 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(energy, 1.0, atol=0.0, rtol=0.0)


def test_direct_post_wo_accepts_declared_fp32_simplex_closure() -> None:
    shape = selector.SelectorShape(
        layers=1,
        heads=1,
        components=2,
        head_dimension=1,
        rank=1,
    )
    coefficients = np.asarray([[[[0.2, 0.8000002]]]], dtype=np.float32)
    values = np.asarray([[[[[0.0], [2.0]]]]], dtype=np.float32)
    base = np.asarray([[[[1.0]]]], dtype=np.float32)
    target = np.asarray([[[1.0]]], dtype=np.float32)
    projection = np.asarray([[[2.0]]], dtype=np.float32)
    error, energy = selector.direct_post_wo_error_energy(
        coefficients,
        values,
        base,
        target,
        projection,
        shape,
    )
    assert np.isfinite(error).all()
    assert np.isfinite(energy).all()


def test_tiny_direct_training_reduces_final_step_loss() -> None:
    pytest.importorskip("torch")
    shape = selector.SelectorShape(
        layers=1,
        heads=1,
        components=2,
        head_dimension=1,
        rank=2,
    )
    records, reads = 4, 2
    mass = np.full((records, reads, 1, 1, 2), 0.5, dtype=np.float32)
    valid = np.ones(mass.shape, dtype=bool)
    values = np.zeros(mass.shape + (1,), dtype=np.float32)
    values[..., 1, 0] = 2.0
    base = np.ones((records, reads, 1, 1, 1), dtype=np.float32)
    target = np.ones((records, reads, 1, 1), dtype=np.float32)
    projection = np.ones((1, 1, 1), dtype=np.float32)
    config = selector.TrainingConfig(
        steps=32,
        warmup_steps=2,
        peak_learning_rate=0.05,
        final_learning_rate=0.01,
        uv_weight_decay=0.0,
        rows_per_layer_per_step=2,
        epochs=16,
        init_seed=7,
        shuffle_seed=8,
    )
    result = selector.fit_direct_post_wo(
        mass,
        valid,
        values,
        base,
        target,
        projection,
        training_records=(0, 1),
        shape=shape,
        config=config,
    )
    assert np.isfinite(result.initial_loss)
    assert np.isfinite(result.final_loss)
    assert result.final_loss < result.initial_loss * 0.5
    result.parameters.validate(shape)


def test_learning_rate_requires_two_post_warmup_steps() -> None:
    with pytest.raises(ValueError, match="configuration"):
        selector.learning_rate_schedule(
            selector.TrainingConfig(steps=3, warmup_steps=2)
        )
