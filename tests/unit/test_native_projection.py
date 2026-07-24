from pathlib import Path

import numpy as np
import pytest

from engram.models.native_bitnet import pack_hf_bitnet_codes
from engram.runtime.native_bitnet import _materialized_bitlinear_class
from engram.runtime.native_projection import NativeTernaryProjectionKernel


def test_native_ternary_projection_matches_materialized_bitlinear():
    library = Path("build/libengram_bitnet.so")
    if not library.exists():
        pytest.skip("native BitNet library has not been built")
    torch = pytest.importorskip("torch")
    generator = np.random.default_rng(47)
    codes = generator.integers(-1, 2, size=(12, 9), dtype=np.int8)
    packed = pack_hf_bitnet_codes(codes)
    hidden = torch.from_numpy(
        generator.normal(size=(5, 9)).astype(np.float32)
    ).to(torch.bfloat16)
    scale = 0.03125
    expected = _materialized_bitlinear_class()(
        torch.from_numpy(codes),
        torch.tensor([scale], dtype=torch.bfloat16),
    )(hidden)

    kernel = NativeTernaryProjectionKernel(threads=2, library=library)
    projection = kernel.add(packed, output_features=12, scale=scale)
    actual = kernel.forward(projection, hidden)
    kernel.close()

    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.01)
