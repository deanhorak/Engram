from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_content_selector as content
import engram.evaluation.olmoe_retrieval_episodic_mass_selector as mass


def _fixture() -> tuple[
    content.ContentSelectorShape,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    shape = content.ContentSelectorShape(
        layers=1,
        heads=2,
        components=3,
        head_dimension=2,
        mass_rank=2,
        content_rank=2,
    )
    native_mass = np.asarray(
        [
            [
                [
                    [
                        [0.25, 0.5, 0.25],
                        [0.6, 0.3, 0.1],
                    ]
                ]
            ]
        ],
        dtype=np.float32,
    )
    valid = np.ones(native_mass.shape, dtype=bool)
    query = np.asarray(
        [[[[[1.0, -0.5], [0.25, 2.0]]]]],
        dtype=np.float32,
    )
    values = np.asarray(
        [
            [
                [
                    [
                        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]],
                        [[0.5, 0.5], [-0.5, 1.0], [2.0, -1.0]],
                    ]
                ]
            ]
        ],
        dtype=np.float32,
    )
    return shape, native_mass, valid, query, values


def test_zero_initialization_is_exact_native_attention() -> None:
    shape, native_mass, valid, query, values = _fixture()
    parameters = content.initialize_parameters(shape, seed=7)
    coefficients, delta, sidecars = content.selector_forward(
        native_mass,
        valid,
        query,
        values,
        parameters,
        shape,
        quantize_sidecars=True,
    )
    np.testing.assert_array_equal(delta, np.zeros_like(delta))
    np.testing.assert_array_equal(sidecars, np.zeros_like(sidecars))
    np.testing.assert_allclose(coefficients, native_mass, rtol=0.0, atol=6.0e-8)


def test_content_branch_changes_coefficients_and_preserves_simplex() -> None:
    shape, native_mass, valid, query, values = _fixture()
    initial = content.initialize_parameters(shape, seed=11)
    Q = np.zeros_like(initial.Q)
    P = np.zeros_like(initial.P)
    Q[..., 0, 0] = 1.0
    Q[..., 1, 1] = 1.0
    P[..., 0, 0] = 0.5
    P[..., 1, 1] = -0.25
    parameters = replace(initial, Q=Q, P=P)
    coefficients, delta, sidecars = content.selector_forward(
        native_mass,
        valid,
        query,
        values,
        parameters,
        shape,
    )
    assert np.max(np.abs(delta)) > 0.0
    assert np.max(np.abs(sidecars)) > 0.0
    assert np.max(np.abs(coefficients - native_mass)) > 0.0
    np.testing.assert_allclose(
        np.sum(coefficients, axis=-1),
        1.0,
        rtol=0.0,
        atol=2.0e-6,
    )
    assert np.all(coefficients >= 0.0)


def test_mass_and_reconstructed_score_routes_match() -> None:
    shape, native_mass, valid, query, values = _fixture()
    parameters = content.initialize_parameters(shape, seed=17)
    P = np.full_like(parameters.P, 0.125)
    V = np.full_like(parameters.V, 0.05)
    parameters = replace(parameters, P=P, V=V)
    from_mass = content.selector_forward(
        native_mass,
        valid,
        query,
        values,
        parameters,
        shape,
        quantize_sidecars=True,
    )
    scores = np.log(native_mass).astype(np.float32)
    from_scores = content.selector_forward_from_scores(
        scores,
        valid,
        query,
        values,
        parameters,
        shape,
        quantize_sidecars=True,
    )
    for left, right in zip(from_mass, from_scores, strict=True):
        np.testing.assert_allclose(left, right, rtol=0.0, atol=1.0e-6)


def test_bf16_parameters_and_sidecars_are_explicit() -> None:
    shape, native_mass, valid, query, values = _fixture()
    parameters = content.initialize_parameters(shape, seed=23)
    parameters = replace(
        parameters,
        V=np.full_like(parameters.V, 0.0317),
        P=np.full_like(parameters.P, 0.0273),
    )
    decoded, bits = content.quantize_parameters_bf16(parameters, shape)
    assert set(bits) == {"U", "V", "E", "B", "Q", "P"}
    assert all(value.dtype == np.uint16 for value in bits.values())
    coefficients, delta, sidecars = content.selector_forward(
        native_mass,
        valid,
        query,
        values,
        decoded,
        shape,
        quantize_sidecars=True,
    )
    assert np.isfinite(coefficients).all()
    assert np.isfinite(delta).all()
    np.testing.assert_array_equal(
        sidecars,
        mass.bf16_bits_to_float32(mass.float32_to_bf16_bits(sidecars)),
    )


def test_invalid_component_remains_masked() -> None:
    shape, native_mass, valid, query, values = _fixture()
    native_mass = native_mass.copy()
    valid = valid.copy()
    values = values.copy()
    native_mass[..., 2] = 0.0
    native_mass[..., :2] /= np.sum(native_mass[..., :2], axis=-1, keepdims=True)
    valid[..., 2] = False
    values[..., 2, :] = 123.0
    parameters = content.initialize_parameters(shape, seed=29)
    parameters = replace(
        parameters,
        V=np.full_like(parameters.V, 0.2),
        B=np.full_like(parameters.B, 0.3),
        P=np.full_like(parameters.P, 0.1),
    )
    coefficients, delta, sidecars = content.selector_forward(
        native_mass,
        valid,
        query,
        values,
        parameters,
        shape,
    )
    np.testing.assert_array_equal(coefficients[..., 2], 0.0)
    np.testing.assert_array_equal(delta[..., 2], 0.0)
    np.testing.assert_array_equal(sidecars[..., 2, :], 0.0)


def test_production_resource_contract_is_below_exact_51_head_ceiling() -> None:
    resource = content.production_resource_contract()
    assert resource["parameter_count"] == 287_744
    assert resource["serialized_parameter_bytes"] == 575_488
    assert resource["value_sidecar_state_bytes"] == 114_688
    assert (
        resource["total_logical_traffic_bytes_per_128_token_sequence"]
        == 796_655_616
    )
    assert (
        resource["total_logical_traffic_bytes_per_128_token_sequence"]
        < resource["exact_51_head_equivalent_ceiling_bytes"]
    )
    assert resource["single_full_value_accumulation_pass"] is True


def test_shape_and_parameter_validation_fail_closed() -> None:
    with pytest.raises(ValueError):
        content.ContentSelectorShape(content_rank=0).validate()
    shape, native_mass, valid, query, values = _fixture()
    parameters = content.initialize_parameters(shape, seed=31)
    with pytest.raises(ValueError):
        content.selector_forward(
            native_mass,
            valid,
            query[..., :-1],
            values,
            parameters,
            shape,
        )
    with pytest.raises(ValueError):
        content.selector_forward(
            native_mass,
            valid,
            query,
            values,
            replace(parameters, P=parameters.P.astype(np.float64)),
            shape,
        )


def test_tiny_cpu_fit_is_finite_and_deterministic() -> None:
    shape = content.ContentSelectorShape(
        layers=1,
        heads=1,
        components=3,
        head_dimension=2,
        mass_rank=2,
        content_rank=2,
    )
    generator = np.random.Generator(np.random.PCG64(37))
    scores = generator.normal(size=(2, 2, 1, 1, 3)).astype(np.float32)
    scores -= np.max(scores, axis=-1, keepdims=True)
    native_mass = np.exp(scores).astype(np.float32)
    native_mass /= np.sum(native_mass, axis=-1, keepdims=True, dtype=np.float32)
    valid = np.ones(native_mass.shape, dtype=bool)
    query = generator.normal(size=(2, 2, 1, 1, 2)).astype(np.float32)
    values = generator.normal(size=(2, 2, 1, 1, 3, 2)).astype(np.float32)
    base = np.einsum("...c,...cd->...d", native_mass, values).astype(np.float32)
    target = generator.normal(size=(2, 2, 1, 2)).astype(np.float32)
    output_projection = np.eye(2, dtype=np.float32)[None]
    config = mass.TrainingConfig(
        steps=8,
        warmup_steps=2,
        rows_per_layer_per_step=1,
        epochs=2,
        init_seed=41,
        shuffle_seed=43,
    )
    first = content.fit_direct_post_wo(
        native_mass,
        valid,
        query,
        values,
        base,
        target,
        output_projection,
        training_records=(0, 1),
        shape=shape,
        config=config,
    )
    second = content.fit_direct_post_wo(
        native_mass,
        valid,
        query,
        values,
        base,
        target,
        output_projection,
        training_records=(0, 1),
        shape=shape,
        config=config,
    )
    assert np.isfinite(first.initial_loss)
    assert np.isfinite(first.final_loss)
    assert content.parameters_sha256(
        first.parameters,
        shape,
    ) == content.parameters_sha256(second.parameters, shape)
    assert first.schedule_sha256 == second.schedule_sha256
