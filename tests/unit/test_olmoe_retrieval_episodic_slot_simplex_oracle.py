from __future__ import annotations

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_slot_simplex_oracle as slot


def _basis_arrays() -> dict[str, np.ndarray]:
    regular_mean = np.asarray(
        [[[1.0, -0.5], [0.25, 0.75]]],
        dtype=np.float32,
    )
    slot_values = np.asarray(
        [
            [
                [[0.0, 1.0], [2.0, 0.5]],
                [[-0.5, 0.0], [1.0, 1.5]],
            ]
        ],
        dtype=np.float32,
    )
    regular_mass = np.asarray([[0.5, 0.5]], dtype=np.float32)
    slot_mass = np.asarray(
        [[[0.2, 0.3], [0.2, 0.3]]],
        dtype=np.float32,
    )
    episodic_mass = np.sum(slot_mass, axis=-1, dtype=np.float32)
    regular_component = regular_mean * regular_mass[..., None]
    episodic_component = np.einsum(
        "nhs,nhsd->nhd",
        slot_mass,
        slot_values,
        optimize=True,
    ).astype(np.float32)
    base_heads = regular_component + episodic_component
    base = base_heads.reshape(1, 4)
    return {
        "base_attention_output": base,
        "regular_component": regular_component.reshape(1, 4),
        "episodic_component": episodic_component.reshape(1, 4),
        "regular_mass": regular_mass,
        "episodic_mass": episodic_mass,
        "base_projected": np.zeros_like(base),
        "target_residual": np.ones_like(base),
        "slot_mass": slot_mass,
        "slot_values": slot_values,
    }


def test_build_slot_basis_reconstructs_native_point() -> None:
    arrays = _basis_arrays()
    basis = slot.build_slot_basis(arrays, query_heads=2, slots=2)
    assert basis.components.shape == (1, 2, 4, 2)
    assert basis.correction_basis.shape == basis.components.shape
    np.testing.assert_allclose(
        np.sum(basis.base_coefficients, axis=-1),
        1.0,
        rtol=0.0,
        atol=1.0e-7,
    )
    reconstructed = np.einsum(
        "nhc,nhcd->nhd",
        basis.base_coefficients,
        basis.components,
        optimize=True,
    )
    np.testing.assert_allclose(
        reconstructed,
        basis.base_heads,
        rtol=0.0,
        atol=1.0e-7,
    )
    assert basis.base_reconstruction_max_abs <= 1.0e-7
    assert basis.episodic_component_reconstruction_max_abs <= 1.0e-7


@pytest.mark.parametrize(
    ("name", "mutation"),
    (
        (
            "slot_mass",
            lambda value: value.__setitem__((0, 0, 0), np.float32(0.1)),
        ),
        (
            "slot_values",
            lambda value: value.__setitem__((0, 0, 0, 0), np.float32(np.nan)),
        ),
        (
            "regular_mass",
            lambda value: value.__setitem__((0, 0), np.float32(0.4)),
        ),
    ),
)
def test_build_slot_basis_rejects_broken_trace(
    name: str,
    mutation,
) -> None:
    arrays = _basis_arrays()
    arrays[name] = arrays[name].copy()
    mutation(arrays[name])
    with pytest.raises(ValueError):
        slot.build_slot_basis(arrays, query_heads=2, slots=2)


def _small_full_arrays() -> tuple[dict[str, np.ndarray], np.ndarray]:
    records = 2
    positions = 2
    layers = 2
    heads = 2
    slots = 2
    dimension = 2
    hidden = heads * dimension
    regular_mean = np.asarray(
        [[1.0, -0.5], [0.25, 0.75]],
        dtype=np.float32,
    )
    values = np.asarray(
        [
            [[0.0, 1.0], [2.0, 0.5]],
            [[-0.5, 0.0], [1.0, 1.5]],
        ],
        dtype=np.float32,
    )
    native = np.asarray([0.5, 0.2, 0.3], dtype=np.float32)
    desired = np.asarray([0.2, 0.5, 0.3], dtype=np.float32)
    component = np.concatenate((regular_mean[:, None, :], values), axis=1)
    base_heads = np.einsum("c,hcd->hd", native, component, optimize=True)
    target_heads = np.einsum("c,hcd->hd", desired, component, optimize=True)
    target = (target_heads - base_heads).reshape(hidden)
    prefix = (records, positions, layers)
    base = np.broadcast_to(base_heads.reshape(hidden), prefix + (hidden,)).copy()
    regular_component = np.broadcast_to(
        (native[0] * regular_mean).reshape(hidden),
        prefix + (hidden,),
    ).copy()
    episodic_component = np.broadcast_to(
        np.einsum(
            "s,hsd->hd",
            native[1:],
            values,
            optimize=True,
        ).reshape(hidden),
        prefix + (hidden,),
    ).copy()
    regular_mass = np.full(prefix + (heads,), native[0], dtype=np.float32)
    slot_mass = np.broadcast_to(
        native[1:],
        prefix + (heads, slots),
    ).copy()
    episodic_mass = np.sum(slot_mass, axis=-1, dtype=np.float32)
    slot_values = np.broadcast_to(
        values,
        prefix + (heads, slots, dimension),
    ).copy()
    target_scale = (
        np.arange(1, records * positions * layers + 1, dtype=np.float32)
        .reshape(prefix)
        / np.float32(10.0)
    )
    target_residual = target_scale[..., None] * target
    arrays = {
        "base_attention_output": base,
        "regular_component": regular_component,
        "episodic_component": episodic_component,
        "regular_mass": regular_mass,
        "episodic_mass": episodic_mass,
        "base_projected": base.copy(),
        "target_residual": target_residual,
        "slot_mass": slot_mass,
        "slot_values": slot_values,
    }
    projection = np.broadcast_to(
        np.eye(hidden, dtype=np.float32),
        (layers, hidden, hidden),
    ).copy()
    return arrays, projection


def test_full_oracle_recovers_known_product_simplex_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(slot, "_RECORDS", 2)
    monkeypatch.setattr(slot, "_READ_POSITIONS", (3, 4))
    monkeypatch.setattr(slot, "_LAYERS", 2)
    monkeypatch.setattr(slot, "_QUERY_HEADS", 2)
    monkeypatch.setattr(slot, "_SLOTS", 2)
    monkeypatch.setattr(slot, "_CONSTRUCTIBLE_COMPONENTS", 3)
    monkeypatch.setattr(slot, "_OPTIMISTIC_COMPONENTS", 4)
    monkeypatch.setattr(slot, "_HEAD_DIMENSION", 2)
    monkeypatch.setattr(slot, "_HIDDEN_SIZE", 4)
    arrays, projection = _small_full_arrays()
    result = slot.run_slot_simplex_oracle_from_arrays(
        arrays,
        projection,
        row_batch_size=2,
        maximum_iterations=256,
        relative_gap_tolerance=1.0e-10,
    )
    constructible = slot.run_slot_simplex_oracle_from_arrays(
        arrays,
        projection,
        row_batch_size=2,
        maximum_iterations=256,
        relative_gap_tolerance=1.0e-10,
        include_exact_native_anchor=False,
    )
    assert result.batch_shape == (2, 2, 2)
    assert result.deterministic_replay_exact
    assert np.all(result.converged)
    assert np.max(result.direct_error_energy) <= 2.0e-16
    assert np.max(result.objective_gap_upper_bound) <= 1.0e-8
    assert constructible.coefficients.shape == (8, 2, 3)
    assert constructible.deterministic_replay_exact
    assert np.max(constructible.direct_error_energy) <= 5.0e-16
    np.testing.assert_allclose(
        result.objective,
        result.direct_error_energy,
        rtol=0.0,
        atol=1.0e-14,
    )
    expected_energy = np.einsum(
        "...d,...d->...",
        arrays["target_residual"].astype(np.float64),
        arrays["target_residual"].astype(np.float64),
        optimize=True,
    ).reshape(-1)
    np.testing.assert_array_equal(result.target_energy, expected_energy)


def test_trace_summary_requires_exact_bf16_decodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(slot, "_READ_POSITIONS", (3,))
    monkeypatch.setattr(slot, "_LAYERS", 1)
    monkeypatch.setattr(slot, "_QUERY_HEADS", 2)
    monkeypatch.setattr(slot, "_SLOTS", 2)
    monkeypatch.setattr(slot, "_HEAD_DIMENSION", 2)
    monkeypatch.setattr(slot, "_HIDDEN_SIZE", 4)
    row = _basis_arrays()
    arrays = {
        name: np.ascontiguousarray(value[None], dtype=np.float32)
        for name, value in row.items()
    }
    arrays["slot_values"] = np.ascontiguousarray(
        arrays["slot_values"].astype(np.float32)
    )
    # The fixture's halves and integers are all exactly BF16 representable.
    summary = slot._trace_summary(arrays, [3])
    assert summary["slot_values_exact_bf16_decodes"]
    arrays["slot_values"] = arrays["slot_values"].copy()
    arrays["slot_values"][0, 0, 0, 0, 0] = np.float32(1.0001)
    with pytest.raises(ValueError, match="BF16"):
        slot._trace_summary(arrays, [3])
