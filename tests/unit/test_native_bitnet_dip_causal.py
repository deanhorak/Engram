import pytest

from engram.evaluation.native_bitnet_dip_causal import (
    _causal_evidence_passed,
    _json_native,
    _load_causal_records,
    _prediction_views,
    _validate_configurations,
)
from engram.semantic.native_bitnet_dip import NativeBitNetDIPConfiguration


def test_causal_record_loader_honors_offset_and_count(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"input_ids":[1]}\n'
        '{"input_ids":[2]}\n'
        '{"input_ids":[3]}\n',
        encoding="utf-8",
    )

    records = _load_causal_records(path, offset=1, count=2)

    assert records == [{"input_ids": [2]}, {"input_ids": [3]}]


def test_causal_configuration_requires_full_schedule_by_default():
    configuration = NativeBitNetDIPConfiguration(
        input_fraction=0.75,
        candidate_count=3,
        top_k=2,
    )
    with pytest.raises(ValueError, match="one configuration per layer"):
        _validate_configurations(
            {0: configuration},
            layer_count=2,
            require_all_layers=True,
        )

    assert _validate_configurations(
        {0: configuration},
        layer_count=2,
        require_all_layers=False,
    ) == {0: configuration}


def test_causal_configuration_rejects_invalid_layer_and_value():
    configuration = NativeBitNetDIPConfiguration(
        input_fraction=0.75,
        candidate_count=3,
        top_k=2,
    )
    with pytest.raises(ValueError, match="outside"):
        _validate_configurations(
            {2: configuration},
            layer_count=2,
            require_all_layers=False,
        )
    with pytest.raises(ValueError, match="Configuration"):
        _validate_configurations(
            {0: object()},
            layer_count=2,
            require_all_layers=False,
        )


def test_causal_report_normalizes_numpy_scalars_for_json():
    import json

    report = _json_native(
        {
            "passed": __import__("numpy").bool_(True),
            "metric": __import__("numpy").float64(0.25),
            "count": __import__("numpy").int64(3),
            "nested": [__import__("numpy").float32(0.5)],
        }
    )

    assert json.loads(json.dumps(report, allow_nan=False)) == {
        "passed": True,
        "metric": 0.25,
        "count": 3,
        "nested": [0.5],
    }


def test_prediction_views_score_n_positions_from_n_plus_one_tokens():
    torch = pytest.importorskip("torch")

    class Result:
        def __init__(self, offset):
            self.logits = (
                torch.arange(2 * 4 * 5).reshape(2, 4, 5) + offset
            )
            self.hidden_states = (
                torch.arange(2 * 4 * 3).reshape(2, 4, 3) + offset,
            )

    input_ids = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    dense_logits, sparse_logits, dense_hidden, sparse_hidden, labels = (
        _prediction_views(Result(0), Result(1), input_ids)
    )

    assert dense_logits.shape == sparse_logits.shape == (2, 3, 5)
    assert dense_hidden.shape == sparse_hidden.shape == (2, 3, 3)
    assert labels.tolist() == [[11, 12, 13], [21, 22, 23]]
    assert torch.equal(dense_logits, Result(0).logits[:, :-1].float())


def test_causal_evidence_requires_eight_unique_32_prediction_sequences():
    assert not _causal_evidence_passed(
        sequences=2,
        unique_sequences=2,
        predictions_per_sequence=16,
        prediction_positions=32,
        all_mlp_layers=True,
    )
    assert not _causal_evidence_passed(
        sequences=8,
        unique_sequences=7,
        predictions_per_sequence=32,
        prediction_positions=256,
        all_mlp_layers=True,
    )
    assert _causal_evidence_passed(
        sequences=8,
        unique_sequences=8,
        predictions_per_sequence=32,
        prediction_positions=256,
        all_mlp_layers=True,
    )
