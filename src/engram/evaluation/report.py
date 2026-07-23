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


def write_semantic_routing_report(
    report: dict[str, Any], out: str | Path
) -> tuple[Path, Path]:
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
            "Use `engram evaluate-mlp-intervention` to measure downstream logits and NLL;",
            "candidate recall alone is not a quality result.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def write_rank_router_sweep_report(
    report: dict[str, Any], out: str | Path
) -> tuple[Path, Path]:
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "rank_router_regularization_sweep.json"
    markdown_path = target / "rank_router_regularization_sweep.md"
    atomic_json(json_path, report)
    configuration = report["configuration"]
    lines = [
        "# Rank-router regularization sweep",
        "",
        f"Status: **{report['screening_decision']}**",
        "",
        f"Rank: {configuration['rank']}; top-K: {configuration['top_k']}.",
        "",
        "| Regularization | Candidates | Mean recall | Minimum layer mean | Recall gate |",
        "|---:|---:|---:|---:|---|",
    ]
    for arm in report["arms"]:
        lines.append(
            f"| {arm['regularization']:g} | {arm['candidate_count']} | "
            f"{_fmt(arm['candidate_recall']['mean'])} | "
            f"{_fmt(arm['layer_mean_candidate_recall']['minimum'])} | "
            f"{'pass' if arm['meets_recall_gate'] else 'fail'} |"
        )
    cache = report["membership_cache"]
    lines.extend(
        [
            "",
            f"Packed membership cache: `{cache['path']}` ({cache['hits']} hits, "
            f"{cache['misses']} misses).",
            "",
            report["scope_caveat"],
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def write_dip_sweep_report(
    report: dict[str, Any], out: str | Path
) -> tuple[Path, Path]:
    """Write the predictor-free DIP exact-completion screening report."""

    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "dip_exact_completion_sweep.json"
    markdown_path = target / "dip_exact_completion_sweep.md"
    atomic_json(json_path, report)
    configuration = report["configuration"]
    validation = report["validation"]
    recommended = report["recommended_arm"]
    lines = [
        "# Dynamic Input Pruning exact-completion sweep",
        "",
        f"Status: **{report['screening_decision']}**",
        "",
        (
            f"Held-out evidence: {validation['records_per_layer']} states per layer from "
            f"{validation['unique_sequence_count']} unique sequences across "
            f"{configuration['num_hidden_layers']} layers."
        ),
        "",
        (
            "Method: retain the largest-magnitude input coordinates, use the source model's "
            "partial gate/up projections to score every intermediate record, exactly complete "
            "only the candidate records, then rerank candidates at full precision."
        ),
        "",
        (
            "Published DIP motivates input pruning and partial scoring; candidate-only exact "
            "completion plus contribution-norm reranking is an Engram extension."
        ),
        "",
        (
            f"Top-K magnitude reference mean local relative L2: "
            f"{_fmt(report['oracle']['mlp_output_relative_l2']['mean'])}."
        ),
        "",
        "| Input coordinates | Candidates | Top-K | Recall | Oracle score mass | Local rel-L2 | Projected dense traffic | Reduction | Trace screen |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm in report["arms"]:
        traffic = arm["projected_traffic"]
        lines.append(
            f"| {arm['input_coordinate_count']} "
            f"({arm['input_fraction']:.3f}) | {arm['candidate_count']} | {arm['top_k']} | "
            f"{_fmt(arm['candidate_recall']['mean'])} | "
            f"{_fmt(arm['oracle_score_mass_recall']['mean'])} | "
            f"{_fmt(arm['mlp_output_relative_l2']['mean'])} | "
            f"{_fmt(traffic['projected_fraction_of_dense'])} | "
            f"{traffic['projected_dense_over_sparse_reduction']:.3f}x | "
            f"{'near-oracle' if arm['near_oracle_screen'] else ('recall pass' if arm['meets_existing_candidate_recall_gate'] else 'reject')} |"
        )
    lines.extend(
        [
            "",
            (
                f"Lowest-traffic near-oracle trace arm: **{recommended['name']}**, with "
                f"{_fmt(recommended['candidate_recall']['mean'])} mean recall and "
                f"{_fmt(recommended['projected_traffic']['projected_fraction_of_dense'])} "
                "projected dense weight traffic."
            ),
            "",
            f"> {report['measurement_caveat']}",
            "",
            f"> {report['scope_caveat']}",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def write_correction_capsule_sweep_report(
    report: dict[str, Any], out: str | Path
) -> tuple[Path, Path]:
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "correction_capsule_sweep.json"
    markdown_path = target / "correction_capsule_sweep.md"
    atomic_json(json_path, report)
    lines = [
        "# Correction-capsule residual sweep",
        "",
        f"Status: **{report['screening_decision']}**",
        "",
        f"Uncorrected mean local relative L2: {_fmt(report['baseline_relative_l2']['mean'])}",
        "",
        "| Priority | Capsules | Rank | Corrected rel-L2 | Improvement | Hard-subset rel-L2 | Match | Correction MB/token |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in report["arms"]:
        lines.append(
            f"| {arm['priority_fraction']:.2f} | {arm['capsules']} | {arm['rank']} | "
            f"{_fmt(arm['corrected_relative_l2']['mean'])} | "
            f"{_fmt(arm['relative_l2_improvement'])} | "
            f"{_fmt(arm['hard_subset_corrected_relative_l2']['mean'])} | "
            f"{_fmt(arm['match_fraction'])} | "
            f"{arm['logical_correction_bytes_per_token_all_layers'] / 1_000_000:.3f} |"
        )
    lines.extend(["", report["scope_caveat"], ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def write_attention_report(
    report: dict[str, Any], out: str | Path
) -> tuple[Path, Path]:
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


def write_mlp_intervention_report(
    report: dict[str, Any], out: str | Path
) -> tuple[Path, Path]:
    """Write the trained-teacher MLP intervention result and a compact comparison."""

    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "mlp_intervention.json"
    markdown_path = target / "mlp_intervention.md"
    atomic_json(json_path, report)
    baseline = report["baseline"]
    lines = [
        "# Trained-teacher MLP intervention",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Evaluation role: **{report.get('evaluation_role', 'development')}**.",
        "",
        f"Sequences: {baseline['sequences']} ({baseline['unique_sequences']} unique); "
        f"next-token positions: "
        f"{baseline['next_token_positions']}; input-token positions: "
        f"{baseline['input_token_positions']}.",
        "",
        f"Exact-teacher NLL: {_fmt(baseline['negative_log_likelihood'])}; "
        f"perplexity: {_fmt(baseline['perplexity'])}.",
        "",
    ]
    selection = report.get("configuration_selection")
    if selection is not None:
        lines.extend(
            [
                (
                    "Configuration-selection separation: "
                    f"{selection['selection_unique_sequence_count']} selection sequences, "
                    f"{selection['evaluation_unique_sequence_count']} evaluation sequences, "
                    f"{selection['overlapping_sequence_count']} exact overlaps."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "| Arm | Scope | Layers | Input fraction | Candidates | Recall | Score mass | MLP rel-L2 | Final hidden rel-L2 | KL | Top-1 agreement | NLL delta | Gate |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for arm in report["arms"]:
        quality = arm["quality"]
        recall_policy = arm.get("gate", {}).get("candidate_recall", {})
        if recall_policy.get("applicable") is False:
            recall = "N/A (full converted width)"
        elif "candidate_recall" in arm["local_mlp"]:
            recall = _fmt(arm["local_mlp"]["candidate_recall"]["mean"])
        else:
            recall = "-"
        lines.append(
            "| {name} | {scope} | {layers} | {input_fraction} | {candidates} | {recall} | {score_mass} | {mlp} | {hidden} | {kl} | {top1} | {nll} | {gate} |".format(
                name=arm["name"],
                scope=arm["scope"],
                layers=len(arm["layer_indices"]),
                input_fraction=(
                    f"{arm['input_fraction']:.3f}"
                    if arm.get("input_fraction") is not None
                    else "-"
                ),
                candidates=arm.get("candidate_count") or "-",
                recall=recall,
                score_mass=(
                    _fmt(arm["local_mlp"]["oracle_score_mass_recall"]["mean"])
                    if "oracle_score_mass_recall" in arm["local_mlp"]
                    else "-"
                ),
                mlp=_fmt(arm["local_mlp"]["mlp_output_relative_l2"]["mean"]),
                hidden=_fmt(quality["final_hidden_relative_l2"]["mean"]),
                kl=_fmt(quality["teacher_student_kl"]["mean"]),
                top1=_fmt(quality["teacher_top1_agreement"]["mean"]),
                nll=_fmt(quality["nll_delta"]["mean"]),
                gate=(
                    ("pass" if arm.get("gate", {}).get("passed") else "fail")
                    if "gate" in arm
                    else "not applied"
                ),
            )
        )
    lines.extend(
        [
            "",
            f"> {report['measurement_caveat']}",
            "",
            f"> {report['oracle_caveat']}",
            "",
            "Development decision: **{}**.".format(
                report.get("gate_summary", {}).get(
                    "development_decision", "gate not applied"
                )
            ),
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path
