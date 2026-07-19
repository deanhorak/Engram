from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from engram.evaluation.report import write_attention_report, write_oracle_report, write_semantic_routing_report
from engram.episodic.evaluate import evaluate_attention_replacement
from engram.models.fixture import create_tiny_fixture
from engram.models.inspection import inspect_model
from engram.semantic.oracle import analyze_magnitude_oracle
from engram.semantic.evaluate import evaluate_practical_routing
from engram.semantic.memory import build_semantic_package
from engram.tracing.teacher import capture_teacher_traces
from engram.compiler import compile_model
from engram.runtime import EngramRuntime
from engram.runtime.validation import benchmark_runtime, validate_package
from engram.evaluation.end_to_end import evaluate_end_to_end
from engram.evaluation.controller_gate import evaluate_controller_gate
from engram.utils import atomic_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="engram", description="Engram research compiler")
    parser.add_argument("--version", action="version", version="engram 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="validate and inventory a local Llama-compatible model")
    inspect.add_argument("--model", required=True)
    inspect.add_argument("--no-weight-hash", action="store_true")
    inspect.add_argument("--out", type=Path)

    fixture = commands.add_parser("create-fixture", help="create deterministic random Llama-shaped weights")
    fixture.add_argument("--out", required=True, type=Path)
    fixture.add_argument("--seed", type=int, default=7)

    trace = commands.add_parser("trace", help="capture exact MLP-boundary teacher traces")
    trace.add_argument("--model", required=True)
    trace.add_argument("--dataset", type=Path)
    trace.add_argument("--out", required=True, type=Path)
    trace.add_argument("--split", default="calibration")
    trace.add_argument("--seed", type=int, default=17)
    trace.add_argument("--samples", type=int, default=32)

    analyze = commands.add_parser("analyze-mlp", help="run Gate 1 contribution-magnitude oracle")
    analyze.add_argument("--model", required=True)
    analyze.add_argument("--traces", required=True)
    analyze.add_argument("--out", required=True, type=Path)
    analyze.add_argument("--max-records", type=int)

    semantic = commands.add_parser("build-semantic", help="extract and quantize semantic memory")
    semantic.add_argument("--model", required=True)
    semantic.add_argument("--out", required=True, type=Path)
    semantic.add_argument("--key-bits", type=int, default=8)
    semantic.add_argument("--value-codebooks", type=int, default=2)
    semantic.add_argument("--value-codebook-size", type=int, default=16)
    semantic.add_argument("--ivf-clusters", type=int, default=32)
    semantic.add_argument("--ivf-iterations", type=int, default=20)

    routing = commands.add_parser("evaluate-semantic", help="run Gate 2 practical routing evaluation")
    routing.add_argument("--model", required=True)
    routing.add_argument("--calibration-traces", required=True)
    routing.add_argument("--validation-traces", required=True)
    routing.add_argument("--out", required=True, type=Path)
    routing.add_argument("--top-k", type=int, default=8)
    routing.add_argument("--candidates", type=int, default=16)
    routing.add_argument("--background-rank", type=int, default=4)
    routing.add_argument("--ivf-clusters", type=int, default=8)
    routing.add_argument("--ivf-probes", type=int, default=2)
    routing.add_argument("--max-records", type=int)

    attention = commands.add_parser("evaluate-attention", help="run synthetic Gate 3 attention replacement")
    attention.add_argument("--out", required=True, type=Path)
    attention.add_argument("--seed", type=int, default=41)
    attention.add_argument("--length", type=int, default=128)
    attention.add_argument("--key-width", type=int, default=16)
    attention.add_argument("--value-width", type=int, default=16)
    attention.add_argument("--local-window", type=int, default=16)

    compile_command = commands.add_parser("compile", help="compile a runnable Engram package")
    compile_command.add_argument("--model", required=True)
    compile_command.add_argument("--out", required=True, type=Path)
    compile_command.add_argument("--config", type=Path)
    compile_command.add_argument("--seed", type=int)
    compile_command.add_argument("--semantic-top-k", type=int)
    compile_command.add_argument("--semantic-candidates", type=int)
    compile_command.add_argument("--semantic-ivf-clusters", type=int)
    compile_command.add_argument("--semantic-ivf-probes", type=int)
    compile_command.add_argument("--semantic-ivf-iterations", type=int)
    compile_command.add_argument("--vocabulary-candidates", type=int)
    compile_command.add_argument("--vocabulary-ivf-clusters", type=int)
    compile_command.add_argument("--vocabulary-ivf-probes", type=int)
    compile_command.add_argument("--vocabulary-ivf-iterations", type=int)
    compile_command.add_argument("--local-window", type=int)
    compile_command.add_argument("--cycles", type=int)

    generate = commands.add_parser("generate", help="generate with the PyTorch-free reference runtime")
    generate.add_argument("--model", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-tokens", type=int, default=16)
    generate.add_argument("--exact-vocab", action="store_true")

    validate = commands.add_parser("validate", help="verify package checksums and deterministic generation")
    validate.add_argument("--model", required=True)

    benchmark = commands.add_parser("benchmark", help="benchmark the Python reference runtime")
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--tokens", type=int, default=32)
    benchmark.add_argument("--enable-transition-cache", action="store_true")

    quality = commands.add_parser("evaluate-e2e", help="measure Gate 5 against a local HF teacher")
    quality.add_argument("--model", required=True, help="compiled .engram package")
    quality.add_argument("--teacher", required=True, help="local teacher checkpoint")
    quality.add_argument("--dataset", required=True, type=Path)
    quality.add_argument("--out", required=True, type=Path)
    quality.add_argument("--max-records", type=int)

    controller_gate = commands.add_parser("evaluate-controller", help="run synthetic Gate 4 controller instrumentation")
    controller_gate.add_argument("--out", required=True, type=Path)
    controller_gate.add_argument("--seed", type=int, default=73)
    controller_gate.add_argument("--samples", type=int, default=64)
    controller_gate.add_argument("--width", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        result = inspect_model(args.model, hash_weights=not args.no_weight_hash).to_dict()
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(payload, encoding="utf-8")
        print(payload, end="")
    elif args.command == "create-fixture":
        print(create_tiny_fixture(args.out, seed=args.seed))
    elif args.command == "trace":
        capture_teacher_traces(
            args.model,
            args.out,
            dataset=args.dataset,
            split=args.split,
            seed=args.seed,
            samples=args.samples,
        )
        print(args.out)
    elif args.command == "analyze-mlp":
        report = analyze_magnitude_oracle(args.model, args.traces, max_records=args.max_records)
        json_path, markdown_path = write_oracle_report(report, args.out)
        print(json_path)
        print(markdown_path)
    elif args.command == "build-semantic":
        path = build_semantic_package(
            args.model,
            args.out,
            key_bits=args.key_bits,
            value_codebooks=args.value_codebooks,
            value_codebook_size=args.value_codebook_size,
            ivf_clusters=args.ivf_clusters,
            ivf_iterations=args.ivf_iterations,
        )
        print(path)
    elif args.command == "evaluate-semantic":
        report = evaluate_practical_routing(
            args.model,
            args.calibration_traces,
            args.validation_traces,
            top_k=args.top_k,
            candidate_count=args.candidates,
            background_rank=args.background_rank,
            ivf_clusters=args.ivf_clusters,
            ivf_probes=args.ivf_probes,
            max_records=args.max_records,
        )
        json_path, markdown_path = write_semantic_routing_report(report, args.out)
        print(json_path)
        print(markdown_path)
    elif args.command == "evaluate-attention":
        report = evaluate_attention_replacement(
            seed=args.seed,
            length=args.length,
            key_width=args.key_width,
            value_width=args.value_width,
            local_window=args.local_window,
        )
        json_path, markdown_path = write_attention_report(report, args.out)
        print(json_path)
        print(markdown_path)
    elif args.command == "compile":
        compile_config = {}
        if args.config:
            import yaml

            loaded = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
            compile_config = loaded.get("compile", {})
            if not isinstance(compile_config, dict):
                raise ValueError("config 'compile' section must be a mapping")
        def configured(name, default):
            value = getattr(args, name)
            return value if value is not None else compile_config.get(name, default)
        print(
            compile_model(
                args.model,
                args.out,
                seed=int(configured("seed", 71)),
                semantic_top_k=int(configured("semantic_top_k", 8)),
                semantic_candidates=int(configured("semantic_candidates", 16)),
                semantic_ivf_clusters=int(configured("semantic_ivf_clusters", 32)),
                semantic_ivf_probes=int(configured("semantic_ivf_probes", 4)),
                semantic_ivf_iterations=int(configured("semantic_ivf_iterations", 20)),
                vocabulary_candidates=int(configured("vocabulary_candidates", 32)),
                vocabulary_ivf_clusters=int(configured("vocabulary_ivf_clusters", 64)),
                vocabulary_ivf_probes=int(configured("vocabulary_ivf_probes", 4)),
                vocabulary_ivf_iterations=int(configured("vocabulary_ivf_iterations", 20)),
                local_window=int(configured("local_window", 16)),
                cycles=int(configured("cycles", 2)),
            )
        )
    elif args.command == "generate":
        runtime = EngramRuntime(args.model)
        prompt_tokens = runtime.tokenize(args.prompt)
        tokens, metrics = runtime.generate_tokens(
            prompt_tokens, max_tokens=args.max_tokens, exact_vocab=args.exact_vocab
        )
        print(runtime.detokenize(tokens))
        print(json.dumps({"tokens": tokens, "metrics": [item.__dict__ for item in metrics]}, indent=2))
    elif args.command == "validate":
        result = validate_package(args.model)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    elif args.command == "benchmark":
        print(
            json.dumps(
                benchmark_runtime(
                    args.model,
                    tokens=args.tokens,
                    bypass_transition_cache=not args.enable_transition_cache,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "evaluate-e2e":
        report = evaluate_end_to_end(
            args.model, args.teacher, args.dataset, max_records=args.max_records
        )
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "end_to_end_quality.json"
        atomic_json(path, report)
        print(path)
    elif args.command == "evaluate-controller":
        report = evaluate_controller_gate(seed=args.seed, samples=args.samples, width=args.width)
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "controller_gate.json"
        atomic_json(path, report)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
