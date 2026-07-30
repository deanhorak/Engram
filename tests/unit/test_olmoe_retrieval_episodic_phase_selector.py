from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_mass_selector as mass
import engram.evaluation.olmoe_retrieval_episodic_phase_selector as selector


def _shape() -> selector.PhaseSelectorShape:
    return selector.PhaseSelectorShape(
        layers=1,
        heads=1,
        components=3,
        head_dimension=2,
        rank=2,
        phases=2,
    )


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    native_mass = np.asarray(
        [
            [[[0.25, 0.75, 0.0]]],
            [[[0.6, 0.3, 0.1]]],
        ],
        dtype=np.float32,
    )
    valid = np.asarray(
        [
            [[[True, True, False]]],
            [[[True, True, True]]],
        ],
        dtype=bool,
    )
    phase = np.asarray([0, 1], dtype=np.int64)
    active = np.asarray([True, True], dtype=bool)
    return native_mass, valid, phase, active


def test_zero_initialization_is_exact_native_baseline() -> None:
    shape = _shape()
    native_mass, valid, phase, active = _fixture()
    parameters = selector.initialize_parameters(shape, seed=7)
    coefficients, delta = selector.selector_forward(
        native_mass,
        valid,
        phase,
        active,
        parameters,
        shape,
    )
    np.testing.assert_array_equal(delta, np.zeros_like(delta))
    np.testing.assert_allclose(
        coefficients,
        native_mass,
        rtol=0.0,
        atol=6.0e-8,
    )


def test_phase_table_is_phase_specific_and_disabled_when_inactive() -> None:
    shape = _shape()
    native_mass, valid, phase, active = _fixture()
    initialized = selector.initialize_parameters(shape, seed=11)
    table = np.zeros_like(initialized.T)
    table[0, 0, 0] = np.asarray([1.0, -1.0, 200.0], dtype=np.float32)
    table[1, 0, 0] = np.asarray([-0.5, 0.5, 0.0], dtype=np.float32)
    parameters = replace(initialized, T=table)
    coefficients, delta = selector.selector_forward(
        native_mass,
        valid,
        phase,
        active,
        parameters,
        shape,
    )
    assert delta[0, 0, 0, 0] > 0.0
    assert delta[1, 0, 0, 0] < 0.0
    assert np.max(np.abs(coefficients - native_mass)) > 0.0
    np.testing.assert_array_equal(delta[0, ..., 2], 0.0)

    inactive_coefficients, inactive_delta = selector.selector_forward(
        native_mass,
        valid,
        phase,
        np.zeros_like(active),
        parameters,
        shape,
    )
    np.testing.assert_array_equal(inactive_delta, np.zeros_like(inactive_delta))
    np.testing.assert_allclose(
        inactive_coefficients,
        native_mass,
        rtol=0.0,
        atol=6.0e-8,
    )


def test_joint_position_shift_preserves_phase_and_forward_result() -> None:
    shape = _shape()
    native_mass, valid, _phase, active = _fixture()
    positions = np.asarray([96, 105], dtype=np.int64)
    starts = np.asarray([96, 104], dtype=np.int64)
    phase = selector.causal_read_phase(
        positions,
        starts,
        active,
        phase_count=shape.phases,
    )
    shifted_phase = selector.causal_read_phase(
        positions + 37,
        starts + 37,
        active,
        phase_count=shape.phases,
    )
    np.testing.assert_array_equal(phase, shifted_phase)
    initialized = selector.initialize_parameters(shape, seed=13)
    table = np.zeros_like(initialized.T)
    table[0, 0, 0] = np.asarray([0.3, -0.2, 0.1], dtype=np.float32)
    table[1, 0, 0] = np.asarray([-0.1, 0.4, -0.3], dtype=np.float32)
    parameters = replace(initialized, T=table)
    first = selector.selector_forward(
        native_mass,
        valid,
        phase,
        active,
        parameters,
        shape,
    )
    second = selector.selector_forward(
        native_mass,
        valid,
        shifted_phase,
        active,
        parameters,
        shape,
    )
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_invalid_component_stays_zero_with_large_table_payload() -> None:
    shape = _shape()
    native_mass, valid, phase, active = _fixture()
    initialized = selector.initialize_parameters(shape, seed=17)
    table = np.zeros_like(initialized.T)
    table[..., 2] = 10_000.0
    parameters = replace(
        initialized,
        V=np.full_like(initialized.V, 0.2),
        B=np.full_like(initialized.B, 0.3),
        T=table,
    )
    coefficients, delta = selector.selector_forward(
        native_mass[:1],
        valid[:1],
        phase[:1],
        active[:1],
        parameters,
        shape,
    )
    np.testing.assert_array_equal(coefficients[..., 2], 0.0)
    np.testing.assert_array_equal(delta[..., 2], 0.0)
    np.testing.assert_allclose(
        np.sum(coefficients, axis=-1),
        1.0,
        rtol=0.0,
        atol=2.0e-6,
    )


def test_mass_and_native_score_routes_match() -> None:
    shape = _shape()
    native_mass, valid, phase, active = _fixture()
    initialized = selector.initialize_parameters(shape, seed=19)
    parameters = replace(
        initialized,
        V=np.full_like(initialized.V, 0.05),
        T=np.full_like(initialized.T, 0.03125),
    )
    from_mass = selector.selector_forward(
        native_mass,
        valid,
        phase,
        active,
        parameters,
        shape,
    )
    scores = np.full(native_mass.shape, -123.0, dtype=np.float32)
    scores[valid] = np.log(native_mass[valid]) + np.float32(4.5)
    from_scores = selector.selector_forward_from_scores(
        scores,
        valid,
        phase,
        active,
        parameters,
        shape,
    )
    for left, right in zip(from_mass, from_scores, strict=True):
        np.testing.assert_allclose(left, right, rtol=0.0, atol=1.0e-6)


def test_bf16_serialization_is_explicit_and_deterministic() -> None:
    shape = _shape()
    native_mass, valid, phase, active = _fixture()
    initialized = selector.initialize_parameters(shape, seed=23)
    parameters = replace(
        initialized,
        V=np.full_like(initialized.V, 0.0317),
        T=np.full_like(initialized.T, 0.0273),
    )
    decoded, bits = selector.quantize_parameters_bf16(parameters, shape)
    assert set(bits) == {"U", "V", "E", "B", "T"}
    assert all(value.dtype == np.uint16 for value in bits.values())
    assert sum(value.nbytes for value in bits.values()) == shape.parameter_count * 2
    first = selector.selector_forward(
        native_mass,
        valid,
        phase,
        active,
        decoded,
        shape,
    )
    second = selector.selector_forward(
        native_mass,
        valid,
        phase,
        active,
        decoded,
        shape,
    )
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_shape_phase_and_parameter_validation_fail_closed() -> None:
    with pytest.raises(ValueError):
        selector.PhaseSelectorShape(phases=0).validate()
    with pytest.raises(ValueError):
        selector.TrainingConfig(t_weight_decay=-1.0).validate()
    shape = _shape()
    native_mass, valid, phase, active = _fixture()
    parameters = selector.initialize_parameters(shape, seed=29)
    with pytest.raises(ValueError, match="phase"):
        selector.selector_forward(
            native_mass,
            valid,
            phase.astype(np.float32),
            active,
            parameters,
            shape,
        )
    with pytest.raises(ValueError, match="phase"):
        selector.selector_forward(
            native_mass,
            valid,
            np.asarray([0, shape.phases], dtype=np.int64),
            active,
            parameters,
            shape,
        )
    with pytest.raises(ValueError):
        selector.selector_forward(
            native_mass,
            valid,
            phase,
            active,
            replace(parameters, T=parameters.T.astype(np.float64)),
            shape,
        )
    with pytest.raises(ValueError, match="out of range"):
        selector.causal_read_phase(
            np.asarray([8], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([True], dtype=bool),
            phase_count=shape.phases,
        )


def test_production_resource_contract_is_exact_and_below_ceiling() -> None:
    resource = selector.production_resource_contract()
    assert resource["mass_selector_parameter_count"] == 25_600
    assert resource["phase_table_parameter_count"] == 57_344
    assert resource["parameter_count"] == 82_944
    assert resource["serialized_parameter_bytes"] == 165_888
    assert (
        resource["total_logical_traffic_bytes_per_128_token_sequence"]
        == 736_100_352
    )
    assert (
        resource["remaining_headroom_below_exact_51_head_ceiling_bytes"]
        == 237_284_352
    )
    assert resource["combined_attention_and_selector_state_bytes"] == 10_700_800
    assert resource["mass_selector_multiply_accumulates_per_token"] == 229_376
    assert resource["phase_table_additions_per_active_token"] == 7_168
    assert resource["new_KV_state_bytes"] == 0
    assert resource["single_full_value_accumulation_pass"] is True


def test_frozen_mass_phase_fit_reduces_loss_and_is_deterministic() -> None:
    pytest.importorskip("torch")
    shape = selector.PhaseSelectorShape(
        layers=1,
        heads=1,
        components=2,
        head_dimension=1,
        rank=1,
        phases=2,
    )
    records, reads = 2, 2
    native_mass = np.full(
        (records, reads, 1, 1, 2),
        0.5,
        dtype=np.float32,
    )
    valid = np.ones(native_mass.shape, dtype=bool)
    phase = np.tile(
        np.asarray([0, 1], dtype=np.int64),
        (records, 1),
    )
    active = np.ones((records, reads), dtype=bool)
    values = np.zeros(native_mass.shape + (1,), dtype=np.float32)
    values[..., 1, 0] = 2.0
    base = np.einsum(
        "...c,...cd->...d",
        native_mass,
        values,
    ).astype(np.float32)
    target = np.tile(
        np.asarray([1.0, -1.0], dtype=np.float32)[None, :, None, None],
        (records, 1, 1, 1),
    )
    projection = np.ones((1, 1, 1), dtype=np.float32)
    base_mass = mass.initialize_parameters(shape.mass_shape, seed=31)
    config = selector.TrainingConfig(
        steps=32,
        warmup_steps=2,
        peak_learning_rate=0.2,
        final_learning_rate=0.05,
        uv_weight_decay=1.0e-4,
        t_weight_decay=1.0e-4,
        rows_per_layer_per_step=1,
        epochs=8,
        init_seed=37,
        shuffle_seed=41,
    )
    first = selector.fit_phase_table_direct_post_wo(
        native_mass,
        valid,
        phase,
        active,
        values,
        base,
        target,
        projection,
        base_parameters=base_mass,
        training_records=(0, 1),
        shape=shape,
        config=config,
    )
    second = selector.fit_phase_table_direct_post_wo(
        native_mass,
        valid,
        phase,
        active,
        values,
        base,
        target,
        projection,
        base_parameters=base_mass,
        training_records=(0, 1),
        shape=shape,
        config=config,
    )
    assert first.train_mass_branch is False
    assert first.final_loss < first.initial_loss * 0.25
    assert first.final_loss == second.final_loss
    assert (
        selector.parameters_sha256(first.parameters, shape)
        == selector.parameters_sha256(second.parameters, shape)
    )
    for name in ("U", "V", "E", "B"):
        np.testing.assert_array_equal(
            first.parameters.as_dict()[name],
            base_mass.as_dict()[name],
        )
    assert np.max(np.abs(first.parameters.T)) > 0.0
