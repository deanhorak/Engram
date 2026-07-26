import json
import sys
from types import SimpleNamespace

from engram.training.corpus import (
    build_distillation_corpus,
    build_distillation_tail_holdout,
)


class _Tokenizer:
    model_max_length = 1024

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens
        return {"input_ids": list(range(1, len(text) + 1))}


class _AutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _Tokenizer()


def test_build_distillation_corpus_round_robins_sources(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=_AutoTokenizer),
    )
    model = tmp_path / "model"
    model.mkdir()
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "a.md").write_text("a" * 24, encoding="utf-8")
    (sources / "b.py").write_text("b" * 24, encoding="utf-8")
    (sources / "ignored.txt").write_text("x" * 100, encoding="utf-8")
    output = tmp_path / "corpus.jsonl"

    report = build_distillation_corpus(
        model,
        [sources],
        output,
        sequence_length=8,
        minimum_tokens=4,
        max_sequences=4,
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["input_ids"] for record in records] == [
        list(range(1, 9)),
        list(range(9, 17)),
        list(range(17, 25)),
    ]
    assert report["source_files"] == 2
    assert report["sequences"] == 3
    assert report["unique_sequences"] == 3
    assert report["duplicate_sequences_skipped"] == 3
    assert report["deduplication"] == "exact_token_sequence_first_occurrence"
    assert report["token_positions"] == 24
    assert output.with_suffix(".jsonl.manifest.json").is_file()


def test_build_distillation_tail_holdout_authenticates_disjoint_prefix(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps({"input_ids": [index, index + 1, index + 2]}) + "\n"
            for index in range(6)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "development.jsonl"

    report = build_distillation_tail_holdout(source, output, records=2)

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records == [
        {"input_ids": [4, 5, 6]},
        {"input_ids": [5, 6, 7]},
    ]
    assert report["partition"]["training_prefix_records"] == 4
    assert report["partition"]["holdout_records"] == 2
    assert report["partition"]["exact_token_sequence_overlap"] == 0
    assert report["holdout"]["prediction_token_positions"] == 4
    assert output.with_suffix(".jsonl.manifest.json").is_file()
