import pytest

from engram.cli import main


def test_composite_gate_cannot_overwrite_an_input_report(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "mlp_intervention.json"
    second = second_dir / "mlp_intervention.json"
    first.write_text("{}\n", encoding="utf-8")
    second.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not overwrite"):
        main(
            [
                "gate-mlp-intervention",
                "--report",
                str(first),
                str(second),
                "--out",
                str(first_dir),
            ]
        )
