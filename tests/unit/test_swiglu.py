import numpy as np
import pytest

from engram.semantic.swiglu import neuron_contributions, swiglu, swiglu_decomposed


def test_neuron_decomposition_is_exact_float64():
    rng = np.random.default_rng(123)
    hidden = rng.normal(size=(5, 7))
    gate = rng.normal(size=(11, 7))
    up = rng.normal(size=(11, 7))
    down = rng.normal(size=(7, 11))
    direct = swiglu(hidden, gate, up, down)
    decomposed = swiglu_decomposed(hidden, gate, up, down)
    np.testing.assert_allclose(decomposed, direct, rtol=1e-13, atol=1e-13)
    assert neuron_contributions(hidden, gate, up, down).shape == (5, 11, 7)


def test_down_projection_orientation_is_validated():
    hidden = np.zeros((2, 3))
    gate = np.zeros((5, 3))
    up = np.zeros((5, 3))
    wrong_down = np.zeros((5, 3))
    with pytest.raises(ValueError, match="incompatible shapes"):
        swiglu(hidden, gate, up, wrong_down)
