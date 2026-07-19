from __future__ import annotations

from pathlib import Path
from typing import Any

from engram.utils import atomic_json


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def write_oracle_report(report: dict[str, Any], out: str | Path) -> tuple[Path, Path]:
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "oracle_topk.json"
    markdown_path = target / "oracle_topk.md"
    atomic_json(json_path, report)

    lines = [
        "# Milestone 1: MLP magnitude-oracle sparsity",
        "",
        f"Status: **{report['status']}**",
        "",
    ]
    if report["fixture_only"]:
        lines.extend(
            [
                "> These measurements use deterministic random fixture weights. They validate the",
                "> experiment pipeline and make no claim about sparsity in a trained language model.",
                "",
            ]
        )
    lines.extend(
        [
            f"Source hash: `{report['source_model_hash']}`",
            "",
            f"Oracle: {report['oracle_definition']}.",
            "",
            f"Caveat: {report['oracle_limit']}.",
            "",
            "Background comparison: not run (Milestone 2). Gate 1 is therefore incomplete.",
            "",
            "| Scope | Layer | Input type | Target | Mean active fraction | Median rel-L2 | p95 rel-L2 | Mean cosine |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for group in report["groups"]:
        for target_name, metrics in group["targets"].items():
            lines.append(
                "| {scope} | {layer} | {input_type} | {target} | {fraction} | {median_error} | "
                "{p95_error} | {cosine} |".format(
                    scope=group["scope"],
                    layer=group.get("layer", "-"),
                    input_type=group.get("input_type", "-"),
                    target=target_name.replace("pct", "%"),
                    fraction=_fmt(metrics["required_neuron_fraction"]["mean"]),
                    median_error=_fmt(metrics["relative_l2"]["median"]),
                    p95_error=_fmt(metrics["relative_l2"]["p95"]),
                    cosine=_fmt(metrics["cosine_similarity"]["mean"]),
                )
            )
    lines.extend(
        [
            "",
            "The JSON companion contains mean, median, and p95 for every reported metric.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def write_semantic_routing_report(report: dict[str, Any], out: str | Path) -> tuple[Path, Path]:
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "practical_routing.json"
    markdown_path = target / "practical_routing.md"
    atomic_json(json_path, report)
    overall = next(group for group in report["groups"] if group["scope"] == "all")
    metrics = overall["metrics"]
    lines = [
        "# Gate 2: practical semantic routing",
        "",
        f"Status: **{report['status']}**",
        "",
    ]
    if report["fixture_only"]:
        lines.extend(
            [
                "> Random fixture result: this validates routing instrumentation and is not",
                "> evidence of trained-model quality or production speed.",
                "",
            ]
        )
    lines.extend(
        [
            f"Top-K: {report['top_k']}; candidate count: {report['candidate_count']}.",
            "",
            f"Mean candidate recall: {_fmt(metrics['candidate_recall']['mean'])}",
            "",
            f"Mean oracle relative L2: {_fmt(metrics['oracle_relative_l2']['mean'])}",
            "",
            f"Mean practical relative L2: {_fmt(metrics['practical_relative_l2']['mean'])}",
            "",
            f"Mean practical + background relative L2: {_fmt(metrics['with_background_relative_l2']['mean'])}",
            "",
            f"Mean IVF clusters probed: {_fmt(metrics['probed_clusters']['mean'])}",
            "",
            f"Mean semantic records proxy-scored: {_fmt(metrics['probed_records']['mean'])}",
            "",
            "Router timing is Python wall-clock instrumentation, not a production CPU benchmark.",
            "End-to-end logit effect remains explicitly not run.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def write_attention_report(report: dict[str, Any], out: str | Path) -> tuple[Path, Path]:
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "attention_replacement.json"
    markdown_path = target / "attention_replacement.md"
    atomic_json(json_path, report)
    lines = [
        "# Gate 3: attention replacement",
        "",
        f"Status: **{report['status']}**",
        "",
        "> Synthetic states validate the memory algorithms and metrics only. Teacher attention",
        "> traces from a trained model have not been evaluated.",
        "",
        "| Path | Mean rel-L2 | Median rel-L2 | p95 rel-L2 | ns/token |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("local", "recurrent", "hybrid"):
        metric = report["relative_l2"][name]
        lines.append(
            f"| {name} | {_fmt(metric['mean'])} | {_fmt(metric['median'])} | "
            f"{_fmt(metric['p95'])} | {_fmt(report['latency_ns_per_token'][name])} |"
        )
    lines.extend(
        [
            "",
            f"Retrieval recall: {_fmt(report['retrieval_head_recall']['mean'])}",
            "",
            f"Controlled long-context copying accuracy: {_fmt(report['copying_accuracy'])}",
            "",
            f"Peak configured state bytes: {report['memory']['peak_state_bytes']}",
            "",
            "Timing is Python wall-clock instrumentation, not a native production benchmark.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path
