import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from engram.semantic.dip_package import SerializedDIPLayer, write_serialized_dip_layer


def test_native_serialized_dip_matches_python_reference(tmp_path):
    executable = Path("build/engram-dip-bench").resolve()
    if not executable.is_file():
        pytest.skip("build/engram-dip-bench has not been built")
    rng = np.random.default_rng(99)
    gate = rng.normal(size=(11, 32)).astype(np.float32)
    up = rng.normal(size=(11, 32)).astype(np.float32)
    down = rng.normal(size=(32, 11)).astype(np.float32)
    write_serialized_dip_layer(tmp_path, gate, up, down, dual_layout=True)
    hidden = np.asarray(
        [(index % 29 - 14) * 0.03125 for index in range(32)], dtype=np.float32
    )
    expected = SerializedDIPLayer(tmp_path).read(
        hidden, input_fraction=0.5, candidate_count=7, top_k=5
    )

    output = subprocess.check_output(
        [
            str(executable),
            str(tmp_path),
            "1",
            "16",
            "7",
            "5",
            "--record-completion",
        ],
        text=True,
    )
    native = json.loads(output)

    assert native["reference_selected"] == expected.selected_indices.tolist()
    assert native["reference_output_checksum"] == pytest.approx(
        float(np.sum(expected.output, dtype=np.float64)), rel=2e-5, abs=2e-5
    )
    assert native["logical_weight_bytes"] == expected.metrics.logical_weight_bytes
    assert native["dense_weight_bytes"] == expected.metrics.dense_weight_bytes
    assert native["cache_line_weight_bytes"] == expected.metrics.cache_line_weight_bytes
    assert native["record_completion"] is True
