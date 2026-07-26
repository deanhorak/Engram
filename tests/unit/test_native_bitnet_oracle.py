import numpy as np

from engram.evaluation.native_bitnet_oracle import (
    analyze_native_bitnet_layer_oracle,
)
from engram.models.native_bitnet import (
    NativeBitNetLayerWeights,
    load_native_bitnet_artifact,
    save_native_bitnet_artifact,
)


def _artifact(tmp_path):
    # One record is active for the chosen positive input. The remaining gate
    # rows are strictly negative, giving an exactly sparse known ceiling.
    layer = NativeBitNetLayerWeights(
        gate_codes=np.asarray(
            [
                [1, 1, 1, 1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
            dtype=np.int8,
        ),
        up_codes=np.ones((4, 4), dtype=np.int8),
        down_codes=np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.int8,
        ),
        gate_scale=0.5,
        up_scale=0.5,
        down_scale=0.5,
        ffn_sub_norm=np.ones(4, dtype=np.float32),
    )
    path = tmp_path / "oracle.bitnet-records.bin"
    save_native_bitnet_artifact(path, [layer], rms_norm_eps=1e-5)
    return load_native_bitnet_artifact(path)


def test_native_bitnet_oracle_recovers_exact_single_active_record(tmp_path):
    report = analyze_native_bitnet_layer_oracle(
        _artifact(tmp_path),
        0,
        np.ones((2, 4), dtype=np.float32),
        fractions=(0.25, 1.0),
    )

    assert report["coefficient_zero_fraction"]["mean"] == 0.75
    assert report["fractions"][0]["record_count"] == 1
    assert report["fractions"][0]["relative_l2"]["maximum"] == 0.0
    assert report["fractions"][0]["cosine_similarity"]["minimum"] == 1.0
    assert report["fractions"][1]["relative_l2"]["maximum"] == 0.0


def test_native_bitnet_oracle_validates_fraction_and_hidden_shape(tmp_path):
    artifact = _artifact(tmp_path)
    try:
        analyze_native_bitnet_layer_oracle(
            artifact,
            0,
            np.ones((1, 3), dtype=np.float32),
        )
    except ValueError as exc:
        assert "hidden" in str(exc)
    else:
        raise AssertionError("invalid hidden shape was accepted")

    try:
        analyze_native_bitnet_layer_oracle(
            artifact,
            0,
            np.ones((1, 4), dtype=np.float32),
            fractions=(0.0,),
        )
    except ValueError as exc:
        assert "fractions" in str(exc)
    else:
        raise AssertionError("invalid fraction was accepted")
