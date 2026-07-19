from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from engram.runtime import EngramRuntime
from engram.utils import sha256_file


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    return shifted - math.log(float(np.sum(np.exp(shifted))))


def evaluate_end_to_end(
    package: str | Path,
    teacher: str | Path,
    dataset: str | Path,
    *,
    max_records: int | None = None,
) -> dict[str, Any]:
    """Compare compiled Engram with a local HF teacher; never accesses the Hub."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install engram-lm[conversion] for teacher evaluation") from exc
    runtime = EngramRuntime(package)
    teacher_path = Path(teacher)
    model = AutoModelForCausalLM.from_pretrained(
        teacher_path, local_files_only=True, dtype=torch.float32, device_map=None
    ).eval()
    records = [json.loads(line) for line in Path(dataset).read_text().splitlines() if line.strip()]
    if max_records is not None:
        records = records[:max_records]
    tokenizer = None
    if any("input_ids" not in record for record in records):
        tokenizer = AutoTokenizer.from_pretrained(teacher_path, local_files_only=True)
    nlls: list[float] = []
    kls: list[float] = []
    top1: list[float] = []
    top5: list[float] = []
    categories: dict[str, list[float]] = defaultdict(list)
    examples = []
    with torch.inference_mode():
        for record in records:
            if "input_ids" in record:
                tokens = [int(value) for value in record["input_ids"]]
            else:
                assert tokenizer is not None
                tokens = tokenizer(str(record["text"]), return_tensors="pt")["input_ids"][0].tolist()
            if len(tokens) < 2:
                continue
            teacher_logits = model(input_ids=torch.tensor([tokens]))["logits"][0, :-1].float().cpu().numpy()
            runtime.reset()
            student_predictions = []
            for current in tokens[:-1]:
                result = runtime.step(current, exact_vocab=True)
                student_predictions.append(result.token_id)
                student_logits = runtime.vocabulary.exact_logits(runtime.state)
                teacher_logp = _log_softmax(teacher_logits[len(student_predictions) - 1])
                student_logp = _log_softmax(student_logits)
                probability = np.exp(teacher_logp)
                kls.append(float(np.sum(probability * (teacher_logp - student_logp))))
                target = tokens[len(student_predictions)]
                nlls.append(float(-student_logp[target]))
                teacher_order = np.argsort(-teacher_logits[len(student_predictions) - 1], kind="stable")
                top1.append(float(result.token_id == int(teacher_order[0])))
                top5.append(float(result.token_id in set(teacher_order[:5].tolist())))
                categories[str(record.get("input_type", "unspecified"))].append(
                    float(result.token_id == target)
                )
            examples.append(
                {
                    "input_tokens": tokens,
                    "student_greedy_next_tokens": student_predictions,
                    "teacher_greedy_next_tokens": np.argmax(teacher_logits, axis=1).tolist(),
                }
            )
    if not nlls:
        raise ValueError("evaluation dataset produced no next-token positions")
    generated, _ = runtime.generate_tokens(records[0].get("input_ids", [1]), max_tokens=32, exact_vocab=True)
    repetition = 1.0 - len(set(generated)) / len(generated)
    mean_nll = float(np.mean(nlls))
    return {
        "schema_version": 1,
        "experiment": "gate_5_end_to_end_quality",
        "status": "local_teacher_measurement",
        "package_fixture_only": bool(runtime.manifest["fixture_only"]),
        "teacher_path": str(teacher_path.resolve()),
        "dataset_hash": sha256_file(Path(dataset)),
        "tokens_evaluated": len(nlls),
        "student_negative_log_likelihood": mean_nll,
        "student_perplexity": float(math.exp(min(mean_nll, 50.0))),
        "teacher_student_kl": float(np.mean(kls)),
        "teacher_top1_agreement": float(np.mean(top1)),
        "teacher_top5_agreement": float(np.mean(top5)),
        "category_next_token_accuracy": {
            name: float(np.mean(values)) for name, values in sorted(categories.items())
        },
        "generation_repetition_fraction": repetition,
        "generation_examples": examples,
        "long_context_retrieval": {"status": "use Gate 3 controlled test unless dataset supplies a long_context category"},
        "quality_targets_met": None,
        "note": "No trained-model claim is inferred from a local checkpoint; inspect its provenance separately.",
    }
