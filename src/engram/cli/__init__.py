from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from engram.evaluation.report import (
    write_attention_report,
    write_correction_capsule_sweep_report,
    write_dip_sweep_report,
    write_mlp_intervention_report,
    write_oracle_report,
    write_rank_router_sweep_report,
    write_semantic_routing_report,
)
from engram.evaluation.mlp_intervention import evaluate_mlp_interventions
from engram.evaluation.olmoe_q4 import evaluate_olmoe_q4_local
from engram.evaluation.olmoe_q4_causal import evaluate_olmoe_q4_causal
from engram.evaluation.olmoe_q7_systems import (
    evaluate_olmoe_q7_native_systems,
)
from engram.evaluation.olmoe_native_generation import (
    capture_olmoe_teacher_generation,
    evaluate_native_olmoe_generation,
)
from engram.evaluation.olmoe_native_causal import (
    capture_olmoe_teacher_causal_reference,
    evaluate_native_olmoe_causal,
)
from engram.evaluation.olmoe_native_sustained import (
    evaluate_native_olmoe_sustained_context,
    freeze_olmoe_sustained_context_protocol,
)
from engram.evaluation.native_bitnet_parity import (
    evaluate_native_bitnet_parity,
)
from engram.evaluation.native_bitnet_kernel import (
    evaluate_native_bitnet_kernel_confirmation,
)
from engram.evaluation.native_bitnet_oracle import (
    evaluate_native_bitnet_oracle,
    evaluate_native_bitnet_oracle_causal,
    evaluate_native_bitnet_oracle_layer_sweep,
)
from engram.evaluation.native_bitnet_router import (
    evaluate_native_bitnet_dip_all_layers,
    evaluate_native_bitnet_dip_router,
    evaluate_native_bitnet_low_rank_router,
)
from engram.evaluation.native_bitnet_adaptive_k import (
    evaluate_native_bitnet_dip_adaptive_k,
    evaluate_native_bitnet_dip_joint_policy,
)
from engram.evaluation.native_bitnet_attention import (
    evaluate_native_bitnet_attention_substitution,
)
from engram.evaluation.controller_substitution import (
    evaluate_native_bitnet_controller_substitution,
)
from engram.evaluation.controller_only import evaluate_controller_only_trace
from engram.evaluation.controller_provider import (
    evaluate_controller_provider_trace,
    evaluate_controller_sequence_replay,
)
from engram.evaluation.native_attention_benchmark import (
    benchmark_native_streaming_attention,
)
from engram.evaluation.native_bitnet_generation_benchmark import (
    benchmark_native_bitnet_generation,
)
from engram.evaluation.native_bitnet_generation import (
    evaluate_native_bitnet_generation,
)
from engram.evaluation.native_bitnet_controller_generation import (
    evaluate_native_bitnet_controller_generation,
)
from engram.evaluation.native_bitnet_dip_token_generation import (
    evaluate_native_bitnet_dip_token_generation,
)
from engram.evaluation.native_bitnet_dip_attention_confirmation import (
    evaluate_native_bitnet_dip_attention_confirmation,
)
from engram.evaluation.router_sweep import evaluate_rank_router_regularization_sweep
from engram.evaluation.dip_sweep import evaluate_dip_exact_completion_sweep
from engram.evaluation.intrinsic_sparsity import (
    evaluate_intrinsic_sparse_gate_sweep,
    write_intrinsic_sparse_gate_report,
)
from engram.evaluation.correction_sweep import evaluate_correction_capsule_sweep
from engram.evaluation.gates import (
    apply_mlp_intervention_gates,
    combine_mlp_intervention_reports,
)
from engram.episodic.evaluate import evaluate_attention_replacement
from engram.models.fixture import create_tiny_fixture, create_tiny_olmoe_fixture
from engram.models.inspection import inspect_model
from engram.models.native_bitnet import (
    audit_native_bitnet_source,
    repack_native_bitnet_model,
)
from engram.models.olmoe import audit_olmoe_source
from engram.models.olmoe_q7 import (
    inspect_olmoe_q7_artifact,
    repack_olmoe_q7_model,
)
from engram.models.olmoe_native import repack_olmoe_non_mlp_weights
from engram.semantic.oracle import analyze_magnitude_oracle
from engram.semantic.evaluate import evaluate_practical_routing
from engram.semantic.memory import build_semantic_package
from engram.semantic.dip_package import build_serialized_dip_package
from engram.tracing.teacher import (
    capture_teacher_traces,
    plan_teacher_trace_capture,
)
from engram.tracing.olmoe import (
    capture_olmoe_fixture_router_traces,
    capture_olmoe_router_traces,
)
from engram.compiler import (
    compile_olmoe_native_package,
    compile_model,
    compile_native_bitnet_package,
    install_native_bitnet_controller,
    install_native_bitnet_semantic_memory,
)
from engram.runtime import (
    EngramRuntime,
    NativeBitNetDIPTokenRuntime,
    NativeBitNetRuntime,
    OLMoENativePackageRuntime,
    run_native_bitnet_chat,
    validate_native_bitnet_package,
    OLMoENativeTokenRuntime,
)
from engram.runtime.operator_stream import fit_operator_stream_provider
from engram.runtime.validation import benchmark_runtime, validate_package
from engram.evaluation.end_to_end import evaluate_end_to_end
from engram.evaluation.controller_gate import evaluate_controller_gate
from engram.training import (
    adapt_controller_correction_for_provider,
    dagger_refit_operator_provider,
    build_distillation_corpus,
    build_distillation_tail_holdout,
    capture_native_bitnet_controller_traces,
    distill_factorized_controller,
    distill_nonlinear_residual_provider,
    distill_state_space_operator_provider,
    distill_state_space_residual_provider,
    joint_distill_operator_provider,
    evaluate_native_gate_channel_shadow,
    evaluate_native_gate_residual_shadow,
    evaluate_structured_expert_shadow,
    evaluate_gated_background_ceiling,
    evaluate_oracle_residual_ceiling,
    evaluate_width_pruned_local_ceiling,
    evaluate_width_residual_sweep,
    recalibrate_native_gate_residual,
    train_activation_aware_aq_boundaries,
    train_budget_native_ternary_student,
    train_projection_aq_layers,
    train_native_gate_end_to_end,
    train_native_gate_trace_student,
    train_grouped_sparse_boundaries,
    train_fully_sparse_boundaries,
    train_fully_sparse_student,
    train_intrinsic_sparse_boundaries,
    train_shared_expert_boundaries,
    train_sparse_student,
    train_width_pruned_student,
)
from engram.utils import atomic_json, sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engram", description="Engram research compiler"
    )
    parser.add_argument("--version", action="version", version="engram 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser(
        "inspect", help="validate a local or Hugging Face Llama-compatible model"
    )
    inspect.add_argument("--model", required=True)
    inspect.add_argument("--no-weight-hash", action="store_true")
    inspect.add_argument("--out", type=Path)

    bitnet_audit = commands.add_parser(
        "audit-native-bitnet",
        help=("metadata-only audit of the separate native low-bit BitNet source track"),
    )
    bitnet_audit.add_argument("--model", required=True)
    bitnet_audit.add_argument("--revision")
    bitnet_audit.add_argument("--cache-dir", type=Path)
    bitnet_audit.add_argument("--out", type=Path)

    olmoe_audit = commands.add_parser(
        "audit-olmoe",
        help="metadata-first audit of OLMoE router and sparse-expert tensors",
    )
    olmoe_audit.add_argument("--model", required=True)
    olmoe_audit.add_argument("--revision")
    olmoe_audit.add_argument("--cache-dir", type=Path)
    olmoe_audit.add_argument("--out", type=Path)
    olmoe_audit.add_argument(
        "--verify-remote-shapes",
        action="store_true",
        help=("read only bounded safetensors header ranges and validate tensor shapes"),
    )

    olmoe_q7_repack = commands.add_parser(
        "repack-olmoe-q7",
        help="compile local OLMoE routers and experts into the native packed-Q7 artifact",
    )
    olmoe_q7_repack.add_argument("--model", required=True, type=Path)
    olmoe_q7_repack.add_argument("--out", required=True, type=Path)
    olmoe_q7_repack.add_argument("--group-size", type=int, default=64)
    olmoe_q7_repack.add_argument("--report", type=Path)

    olmoe_q7_inspect = commands.add_parser(
        "inspect-olmoe-q7",
        help="strictly validate and describe a native packed OLMoE Q7 artifact",
    )
    olmoe_q7_inspect.add_argument("--artifact", required=True, type=Path)

    olmoe_q7_systems = commands.add_parser(
        "evaluate-native-olmoe-q7",
        help="prove native packed-Q7 route/output parity and exact scheduled traffic",
    )
    olmoe_q7_systems.add_argument("--artifact", required=True, type=Path)
    olmoe_q7_systems.add_argument("--library", required=True, type=Path)
    olmoe_q7_systems.add_argument("--out", required=True, type=Path)
    olmoe_q7_systems.add_argument("--layer", type=int, default=0)
    olmoe_q7_systems.add_argument("--states", type=int, default=1)
    olmoe_q7_systems.add_argument("--threads", type=int, default=1)
    olmoe_q7_systems.add_argument("--seed", type=int, default=7)
    olmoe_q7_systems.add_argument("--maximum-relative-l2", type=float, default=1e-5)
    olmoe_q7_systems.add_argument(
        "--maximum-traffic-fraction", type=float, default=0.45
    )

    olmoe_non_mlp = commands.add_parser(
        "repack-olmoe-non-mlp",
        help="stream OLMoE embedding, attention, norms, and output head into one BF16 mmap file",
    )
    olmoe_non_mlp.add_argument("--model", required=True, type=Path)
    olmoe_non_mlp.add_argument("--out", required=True, type=Path)
    olmoe_non_mlp.add_argument("--report", type=Path)

    olmoe_token = commands.add_parser(
        "run-native-olmoe-token",
        help="run a transformer-shell-free native OLMoE token step",
    )
    olmoe_token.add_argument("--config", required=True, type=Path)
    olmoe_token.add_argument("--non-mlp", required=True, type=Path)
    olmoe_token.add_argument("--q7-artifact", required=True, type=Path)
    olmoe_token.add_argument("--library", required=True, type=Path)
    olmoe_input = olmoe_token.add_mutually_exclusive_group(required=True)
    olmoe_input.add_argument("--token-ids", nargs="+", type=int)
    olmoe_input.add_argument("--prompt")
    olmoe_token.add_argument(
        "--tokenizer",
        type=Path,
        help="tokenizer.json or directory containing it; required with --prompt",
    )
    olmoe_token.add_argument("--threads", type=int, default=1)
    olmoe_token.add_argument("--max-new-tokens", type=int, default=1)

    olmoe_package = commands.add_parser(
        "compile-native-olmoe",
        help="assemble an authenticated package for native OLMoE generation",
    )
    olmoe_package.add_argument("--model", required=True, type=Path)
    olmoe_package.add_argument("--q7-artifact", required=True, type=Path)
    olmoe_package.add_argument("--non-mlp", required=True, type=Path)
    olmoe_package.add_argument("--out", required=True, type=Path)
    olmoe_package.add_argument("--threads", type=int, default=12)
    olmoe_package.add_argument("--report", type=Path)

    olmoe_package_generate = commands.add_parser(
        "generate-native-olmoe-package",
        help="authenticate a native OLMoE package and generate without Transformers",
    )
    olmoe_package_generate.add_argument("--package", required=True, type=Path)
    olmoe_package_generate.add_argument("--manifest-sha256", required=True)
    olmoe_package_generate.add_argument("--library", required=True, type=Path)
    olmoe_package_generate.add_argument("--prompt", required=True)
    olmoe_package_generate.add_argument("--max-new-tokens", type=int, default=1)
    olmoe_package_generate.add_argument("--threads", type=int)

    olmoe_teacher = commands.add_parser(
        "capture-olmoe-teacher-generation",
        help="capture a sealed greedy/top-1 reference from untouched OLMoE",
    )
    olmoe_teacher.add_argument("--model", required=True, type=Path)
    olmoe_teacher.add_argument("--prompts", required=True, type=Path)
    olmoe_teacher.add_argument("--out", required=True, type=Path)
    olmoe_teacher.add_argument("--max-new-tokens", type=int, default=4)
    olmoe_teacher.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    olmoe_teacher.add_argument("--threads", type=int, default=12)

    olmoe_generation = commands.add_parser(
        "evaluate-native-olmoe-generation",
        help="run the frozen authenticated-package versus teacher confirmation",
    )
    olmoe_generation.add_argument("--package", required=True, type=Path)
    olmoe_generation.add_argument("--manifest-sha256", required=True)
    olmoe_generation.add_argument("--library", required=True, type=Path)
    olmoe_generation.add_argument("--prompts", required=True, type=Path)
    olmoe_generation.add_argument("--teacher-reference", required=True, type=Path)
    olmoe_generation.add_argument("--protocol", required=True, type=Path)
    olmoe_generation.add_argument("--protocol-sha256", required=True)
    olmoe_generation.add_argument("--out", required=True, type=Path)
    olmoe_generation.add_argument("--threads", type=int)

    olmoe_causal_teacher = commands.add_parser(
        "capture-olmoe-teacher-causal",
        help="capture sealed BF16 OLMoE causal logits and hidden states",
    )
    olmoe_causal_teacher.add_argument("--model", required=True, type=Path)
    olmoe_causal_teacher.add_argument("--dataset", required=True, type=Path)
    olmoe_causal_teacher.add_argument("--out", required=True, type=Path)
    olmoe_causal_teacher.add_argument("--arrays-out", required=True, type=Path)
    olmoe_causal_teacher.add_argument("--sequences", type=int, default=8)
    olmoe_causal_teacher.add_argument("--tokens-per-sequence", type=int, default=33)
    olmoe_causal_teacher.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu"
    )
    olmoe_causal_teacher.add_argument("--threads", type=int, default=12)
    olmoe_causal_teacher.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="teacher sequences per individual forward pass",
    )
    olmoe_causal_teacher.add_argument(
        "--expert-workers",
        type=int,
        default=1,
        help="capture-only worker threads for independent OLMoE experts",
    )
    olmoe_causal_teacher.add_argument(
        "--sequence-workers",
        type=int,
        default=None,
        help="capture-only concurrent CPU teacher forwards sharing one model",
    )

    olmoe_causal = commands.add_parser(
        "evaluate-native-olmoe-causal",
        help="run the frozen complete 8-sequence/256-position native gate",
    )
    olmoe_causal.add_argument("--package", required=True, type=Path)
    olmoe_causal.add_argument("--manifest-sha256", required=True)
    olmoe_causal.add_argument("--library", required=True, type=Path)
    olmoe_causal.add_argument("--dataset", required=True, type=Path)
    olmoe_causal.add_argument("--teacher-reference", required=True, type=Path)
    olmoe_causal.add_argument("--teacher-arrays", required=True, type=Path)
    olmoe_causal.add_argument("--protocol", required=True, type=Path)
    olmoe_causal.add_argument("--protocol-sha256", required=True)
    olmoe_causal.add_argument("--out", required=True, type=Path)
    olmoe_causal.add_argument("--threads", type=int)

    olmoe_sustained_freeze = commands.add_parser(
        "freeze-olmoe-sustained-protocol",
        help="prospectively freeze the authenticated OLMoE 8x128 gate",
    )
    olmoe_sustained_freeze.add_argument("--package", required=True, type=Path)
    olmoe_sustained_freeze.add_argument("--manifest-sha256", required=True)
    olmoe_sustained_freeze.add_argument("--library", required=True, type=Path)
    olmoe_sustained_freeze.add_argument("--dataset", required=True, type=Path)
    olmoe_sustained_freeze.add_argument("--corpus-manifest", required=True, type=Path)
    olmoe_sustained_freeze.add_argument("--teacher-reference", required=True, type=Path)
    olmoe_sustained_freeze.add_argument("--teacher-arrays", required=True, type=Path)
    olmoe_sustained_freeze.add_argument("--out", required=True, type=Path)
    olmoe_sustained_freeze.add_argument("--threads", type=int, default=12)

    olmoe_sustained = commands.add_parser(
        "evaluate-native-olmoe-sustained",
        help="run the frozen 8-sequence/1,024-position sustained-context gate",
    )
    olmoe_sustained.add_argument("--package", required=True, type=Path)
    olmoe_sustained.add_argument("--manifest-sha256", required=True)
    olmoe_sustained.add_argument("--library", required=True, type=Path)
    olmoe_sustained.add_argument("--dataset", required=True, type=Path)
    olmoe_sustained.add_argument("--corpus-manifest", required=True, type=Path)
    olmoe_sustained.add_argument("--teacher-reference", required=True, type=Path)
    olmoe_sustained.add_argument("--teacher-arrays", required=True, type=Path)
    olmoe_sustained.add_argument("--protocol", required=True, type=Path)
    olmoe_sustained.add_argument("--protocol-sha256", required=True)
    olmoe_sustained.add_argument("--out", required=True, type=Path)
    olmoe_sustained.add_argument("--threads", type=int)

    bitnet_repack = commands.add_parser(
        "repack-native-bitnet",
        help=(
            "download if needed and losslessly repack native BitNet MLP "
            "records as five trits per byte"
        ),
    )
    bitnet_repack.add_argument("--model", required=True)
    bitnet_repack.add_argument("--revision")
    bitnet_repack.add_argument("--cache-dir", type=Path)
    bitnet_repack.add_argument("--out", required=True, type=Path)
    bitnet_repack.add_argument("--report", type=Path)
    bitnet_repack.add_argument(
        "--skip-official-weight-hash",
        action="store_true",
        help="skip the pinned official checkpoint SHA-256 verification",
    )

    bitnet_parity = commands.add_parser(
        "evaluate-native-bitnet-parity",
        help=(
            "run CPU-only local and causal parity against a repacked native "
            "BitNet artifact"
        ),
    )
    bitnet_parity.add_argument("--model", required=True)
    bitnet_parity.add_argument("--artifact", required=True, type=Path)
    bitnet_parity.add_argument(
        "--artifact-sha256",
        help="expected artifact SHA-256 from the repack report",
    )
    bitnet_parity.add_argument("--out", required=True, type=Path)
    bitnet_parity.add_argument("--revision")
    bitnet_parity.add_argument("--cache-dir", type=Path)
    bitnet_parity.add_argument(
        "--local-layers",
        nargs="+",
        type=int,
        default=(0, 14, 29),
    )
    bitnet_parity.add_argument("--local-states", type=int, default=2)
    bitnet_parity.add_argument(
        "--input-ids",
        nargs="+",
        type=int,
        default=(128000,),
    )
    bitnet_parity.add_argument(
        "--no-causal-substitution",
        action="store_true",
    )

    bitnet_kernel = commands.add_parser(
        "evaluate-native-bitnet-kernel",
        help=(
            "run direct packed CPU parity and the frozen native-BitNet "
            "confirmation gate"
        ),
    )
    bitnet_kernel.add_argument("--model", required=True)
    bitnet_kernel.add_argument("--artifact", required=True, type=Path)
    bitnet_kernel.add_argument("--artifact-sha256", required=True)
    bitnet_kernel.add_argument("--dataset", required=True, type=Path)
    bitnet_kernel.add_argument("--out", required=True, type=Path)
    bitnet_kernel.add_argument("--revision")
    bitnet_kernel.add_argument("--cache-dir", type=Path)
    bitnet_kernel.add_argument("--library", type=Path)
    bitnet_kernel.add_argument("--threads", type=int, default=12)
    bitnet_kernel.add_argument("--sequence-count", type=int, default=8)
    bitnet_kernel.add_argument("--prediction-positions", type=int, default=256)
    bitnet_kernel.add_argument("--record-offset", type=int, default=0)
    bitnet_kernel.add_argument(
        "--parity-layers",
        nargs="+",
        type=int,
        default=(0, 14, 29),
    )
    bitnet_kernel.add_argument("--parity-states", type=int, default=2)

    bitnet_oracle = commands.add_parser(
        "analyze-native-bitnet-oracle",
        help=(
            "measure the trained BitNet teacher's exact additive-record "
            "concentration ceiling"
        ),
    )
    bitnet_oracle.add_argument("--model", required=True, type=Path)
    bitnet_oracle.add_argument("--dataset", required=True, type=Path)
    bitnet_oracle.add_argument("--out", required=True, type=Path)
    bitnet_oracle.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=(0, 14, 29),
    )
    bitnet_oracle.add_argument("--samples", type=int, default=2)
    bitnet_oracle.add_argument("--max-tokens", type=int, default=8)
    bitnet_oracle.add_argument("--record-offset", type=int, default=0)
    bitnet_oracle.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=(0.05, 0.1, 0.15, 0.175, 0.25, 0.5, 1.0),
    )
    bitnet_oracle.add_argument("--library", type=Path)
    bitnet_oracle.add_argument("--threads", type=int)

    bitnet_oracle_causal = commands.add_parser(
        "evaluate-native-bitnet-oracle-causal",
        help=(
            "substitute exact top-record oracle reads into every trained "
            "BitNet MLP and measure causal quality"
        ),
    )
    bitnet_oracle_causal.add_argument("--model", required=True, type=Path)
    bitnet_oracle_causal.add_argument("--dataset", required=True, type=Path)
    bitnet_oracle_causal.add_argument("--out", required=True, type=Path)
    bitnet_oracle_causal.add_argument("--fraction", type=float, default=0.25)
    bitnet_oracle_causal.add_argument(
        "--layer-fractions",
        nargs="+",
        type=float,
        help="optional one active-record fraction for each transformer layer",
    )
    bitnet_oracle_causal.add_argument("--sequence-count", type=int, default=1)
    bitnet_oracle_causal.add_argument(
        "--predictions-per-sequence",
        type=int,
        default=8,
    )
    bitnet_oracle_causal.add_argument("--record-offset", type=int, default=0)
    bitnet_oracle_causal.add_argument("--library", type=Path)
    bitnet_oracle_causal.add_argument("--threads", type=int)

    bitnet_oracle_sweep = commands.add_parser(
        "sweep-native-bitnet-oracle-layers",
        help="fit a layer-adaptive exact-record allocation under a mean budget",
    )
    bitnet_oracle_sweep.add_argument("--model", required=True, type=Path)
    bitnet_oracle_sweep.add_argument("--dataset", required=True, type=Path)
    bitnet_oracle_sweep.add_argument("--out", required=True, type=Path)
    bitnet_oracle_sweep.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=(0.15, 0.20, 0.25, 0.30, 0.35),
    )
    bitnet_oracle_sweep.add_argument("--mean-budget", type=float, default=0.25)
    bitnet_oracle_sweep.add_argument("--sequence-count", type=int, default=2)
    bitnet_oracle_sweep.add_argument("--tokens-per-sequence", type=int, default=16)
    bitnet_oracle_sweep.add_argument("--record-offset", type=int, default=0)
    bitnet_oracle_sweep.add_argument("--library", type=Path)
    bitnet_oracle_sweep.add_argument("--threads", type=int)

    bitnet_router = commands.add_parser(
        "evaluate-native-bitnet-router",
        help="fit compact BitNet routers against exact oracle memberships",
    )
    bitnet_router.add_argument("--model", required=True, type=Path)
    bitnet_router.add_argument("--training-trace", required=True, type=Path)
    bitnet_router.add_argument("--validation-trace", required=True, type=Path)
    bitnet_router.add_argument("--out", required=True, type=Path)
    bitnet_router.add_argument("--layers", nargs="+", type=int, default=(0, 14, 29))
    bitnet_router.add_argument(
        "--top-ks", nargs="+", type=int, default=(1728, 1728, 2074)
    )
    bitnet_router.add_argument("--rank", type=int, default=128)
    bitnet_router.add_argument("--steps", type=int, default=500)
    bitnet_router.add_argument("--batch-size", type=int, default=128)
    bitnet_router.add_argument("--learning-rate", type=float, default=2e-3)
    bitnet_router.add_argument("--device", default="cuda")
    bitnet_router.add_argument("--seed", type=int, default=20260726)

    bitnet_dip = commands.add_parser(
        "evaluate-native-bitnet-dip-router",
        help="screen coordinate-pruned BitNet gate/up routing on held-out states",
    )
    bitnet_dip.add_argument("--model", required=True, type=Path)
    bitnet_dip.add_argument("--validation-trace", required=True, type=Path)
    bitnet_dip.add_argument("--out", required=True, type=Path)
    bitnet_dip.add_argument("--layer", type=int, default=14)
    bitnet_dip.add_argument("--top-k", type=int, default=1728)
    bitnet_dip.add_argument(
        "--input-fractions", nargs="+", type=float, default=(0.25, 0.5, 0.75)
    )
    bitnet_dip.add_argument(
        "--candidate-multipliers", nargs="+", type=float, default=(1.0, 1.25, 1.5)
    )

    bitnet_dip_all = commands.add_parser(
        "sweep-native-bitnet-dip-all-layers",
        help=(
            "measure held-out DIP recall for every frozen adaptive BitNet "
            "oracle layer budget"
        ),
    )
    bitnet_dip_all.add_argument("--model", required=True, type=Path)
    bitnet_dip_all.add_argument("--validation-trace", required=True, type=Path)
    bitnet_dip_all.add_argument("--oracle-schedule", required=True, type=Path)
    bitnet_dip_all.add_argument("--out", required=True, type=Path)
    bitnet_dip_all.add_argument("--input-fraction", type=float, default=0.75)
    bitnet_dip_all.add_argument(
        "--candidate-multipliers",
        nargs="+",
        type=float,
        default=(
            1.0,
            1.25,
            1.5,
            1.75,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
        ),
    )
    bitnet_dip_all.add_argument("--maximum-traffic-fraction", type=float, default=0.45)
    bitnet_dip_all.add_argument("--recall-gate", type=float, default=0.95)
    bitnet_dip_all.add_argument("--tail-recall-preference", type=float, default=0.99)
    bitnet_dip_all.add_argument(
        "--worst-row-recall-preference", type=float, default=0.95
    )

    bitnet_adaptive_k = commands.add_parser(
        "sweep-native-bitnet-dip-adaptive-k",
        help=(
            "sweep token-adaptive exact candidate-energy K on protected "
            "BitNet validation states"
        ),
    )
    bitnet_adaptive_k.add_argument("--model", required=True, type=Path)
    bitnet_adaptive_k.add_argument("--validation-trace", required=True, type=Path)
    bitnet_adaptive_k.add_argument("--router-policy", required=True, type=Path)
    bitnet_adaptive_k.add_argument("--out", required=True, type=Path)
    bitnet_adaptive_k.add_argument(
        "--energy-targets",
        nargs="+",
        type=float,
        default=(0.90, 0.95, 0.975, 0.99, 0.995, 0.999),
    )
    bitnet_adaptive_k.add_argument("--minimum-fraction", type=float, default=0.05)
    bitnet_adaptive_k.add_argument("--maximum-fraction", type=float, default=0.425)
    bitnet_adaptive_k.add_argument("--mean-budget-fraction", type=float, default=0.25)
    bitnet_adaptive_k.add_argument("--device", default="cuda")

    bitnet_joint_policy = commands.add_parser(
        "optimize-native-bitnet-dip-joint-policy",
        help=(
            "jointly optimize practical DIP candidate C and target=1 "
            "adaptive K under physical traffic"
        ),
    )
    bitnet_joint_policy.add_argument("--model", required=True, type=Path)
    bitnet_joint_policy.add_argument("--validation-trace", required=True, type=Path)
    bitnet_joint_policy.add_argument("--out", required=True, type=Path)
    bitnet_joint_policy.add_argument(
        "--candidate-counts",
        nargs="+",
        type=int,
        default=(3200, 3456, 3712, 3968, 4224, 4480, 4736, 4992, 5248, 5504),
    )
    bitnet_joint_policy.add_argument("--input-fraction", type=float, default=0.75)
    bitnet_joint_policy.add_argument("--minimum-fraction", type=float, default=0.05)
    bitnet_joint_policy.add_argument("--mean-budget-fraction", type=float, default=0.25)
    bitnet_joint_policy.add_argument(
        "--maximum-traffic-fraction", type=float, default=0.45
    )
    bitnet_joint_policy.add_argument("--device", default="cuda")

    fixture = commands.add_parser(
        "create-fixture", help="create deterministic random Llama-shaped weights"
    )
    fixture.add_argument("--out", required=True, type=Path)
    fixture.add_argument("--seed", type=int, default=7)

    olmoe_fixture = commands.add_parser(
        "create-olmoe-fixture",
        help="create deterministic random OLMoE router/expert weights",
    )
    olmoe_fixture.add_argument("--out", required=True, type=Path)
    olmoe_fixture.add_argument("--seed", type=int, default=17)

    olmoe_trace = commands.add_parser(
        "trace-olmoe-fixture",
        help="capture exact router selection and expert contributions on a fixture",
    )
    olmoe_trace.add_argument("--model", required=True, type=Path)
    olmoe_trace.add_argument("--out", required=True, type=Path)
    olmoe_trace.add_argument("--samples", type=int, default=16)
    olmoe_trace.add_argument("--layers", nargs="+", type=int)
    olmoe_trace.add_argument("--seed", type=int, default=23)

    trained_olmoe_trace = commands.add_parser(
        "trace-olmoe-router",
        help="capture trained OLMoE router selections and exact MLP boundaries",
    )
    trained_olmoe_trace.add_argument("--model", required=True, type=Path)
    trained_olmoe_trace.add_argument("--dataset", required=True, type=Path)
    trained_olmoe_trace.add_argument("--out", required=True, type=Path)
    trained_olmoe_trace.add_argument("--samples", type=int, default=8)
    trained_olmoe_trace.add_argument("--layers", nargs="+", type=int)
    trained_olmoe_trace.add_argument("--tokens-per-sequence", type=int)
    trained_olmoe_trace.add_argument("--seed", type=int, default=37)

    olmoe_q4 = commands.add_parser(
        "evaluate-olmoe-q4-local",
        help="screen groupwise-Q4 OLMoE experts on captured trained states",
    )
    olmoe_q4.add_argument("--model", required=True, type=Path)
    olmoe_q4.add_argument("--trace", required=True, type=Path)
    olmoe_q4.add_argument("--out", required=True, type=Path)
    olmoe_q4.add_argument("--layer", required=True, type=int)
    olmoe_q4.add_argument("--group-size", type=int, default=64)
    olmoe_q4.add_argument("--maximum-mean-relative-l2", type=float, default=0.10)

    olmoe_q4_causal = commands.add_parser(
        "evaluate-olmoe-quantized-causal",
        aliases=["evaluate-olmoe-q4-causal"],
        help="compare BF16 OLMoE against all-layer in-place groupwise low-bit experts",
    )
    olmoe_q4_causal.add_argument("--model", required=True, type=Path)
    olmoe_q4_causal.add_argument("--dataset", required=True, type=Path)
    olmoe_q4_causal.add_argument("--out", required=True, type=Path)
    olmoe_q4_causal.add_argument("--samples", type=int, default=2)
    olmoe_q4_causal.add_argument("--max-tokens", type=int, default=16)
    olmoe_q4_causal.add_argument("--bits", type=int, default=4)
    olmoe_q4_causal.add_argument("--group-size", type=int, default=8)
    olmoe_q4_causal.add_argument("--threads", type=int, default=12)

    trace = commands.add_parser(
        "trace", help="capture exact MLP-boundary teacher traces"
    )
    trace.add_argument("--model", required=True)
    trace.add_argument("--dataset", type=Path)
    trace.add_argument("--out", required=True, type=Path)
    trace.add_argument("--split", default="calibration")
    trace.add_argument("--seed", type=int, default=17)
    trace.add_argument("--samples", type=int, default=32)
    trace.add_argument(
        "--layers",
        nargs="+",
        type=int,
        help="capture only these transformer-layer boundaries",
    )
    trace.add_argument(
        "--mlp-only",
        action="store_true",
        help="omit attention boundary tensors from real-model traces",
    )
    trace.add_argument(
        "--tokens-per-sequence",
        type=int,
        help="sample at most this many token positions from each full-context sequence",
    )
    trace.add_argument(
        "--dry-run",
        action="store_true",
        help="plan fields, token count, and payload bytes without model inference",
    )
    trace.add_argument(
        "--plan-out",
        type=Path,
        help="write the dry-run plan as JSON",
    )

    controller_trace = commands.add_parser(
        "trace-native-bitnet-controller",
        help="capture CPU BitNet trajectories for shared-controller distillation",
    )
    controller_trace.add_argument("--model", required=True, type=Path)
    controller_trace.add_argument("--dataset", required=True, type=Path)
    controller_trace.add_argument("--out", required=True, type=Path)
    controller_trace.add_argument(
        "--split", required=True, choices=("training", "validation", "test")
    )
    controller_trace.add_argument("--samples", type=int, default=8)
    controller_trace.add_argument("--max-tokens", type=int, default=64)
    controller_trace.add_argument(
        "--causal-top-k",
        type=int,
        default=0,
        help=(
            "optionally store teacher top-k logits and next-token targets in "
            "the trace for causal controller distillation"
        ),
    )
    controller_trace.add_argument("--batch-size", type=int, default=1)
    controller_trace.add_argument("--record-offset", type=int, default=0)
    controller_trace.add_argument("--seed", type=int, default=31)
    controller_trace.add_argument("--library", type=Path)
    controller_trace.add_argument("--threads", type=int)
    controller_trace.add_argument(
        "--resume",
        action="store_true",
        help="continue a checksummed incomplete capture without duplicate samples",
    )

    controller_distill = commands.add_parser(
        "distill-controller",
        help=(
            "fit a factorized shared controller on CUDA and export a "
            "PyTorch-free CPU artifact"
        ),
    )
    controller_distill.add_argument("--trace", required=True, type=Path)
    controller_distill.add_argument("--validation-trace", type=Path)
    controller_distill.add_argument("--initial-controller", type=Path)
    controller_distill.add_argument("--out", required=True, type=Path)
    controller_distill.add_argument("--device", default="cuda")
    controller_distill.add_argument("--rank", type=int, default=128)
    controller_distill.add_argument("--adapter-rank", type=int, default=8)
    controller_distill.add_argument("--input-adapter-rank", type=int, default=0)
    controller_distill.add_argument(
        "--operator-residual",
        action="store_true",
        help=(
            "preserve semantic and episodic outputs through their exact "
            "residual-addition path and learn only a factorized correction"
        ),
    )
    controller_distill.add_argument("--steps", type=int, default=1000)
    controller_distill.add_argument("--batch-size", type=int, default=16)
    controller_distill.add_argument("--learning-rate", type=float, default=3e-4)
    controller_distill.add_argument("--weight-decay", type=float, default=1e-3)
    controller_distill.add_argument(
        "--teacher-forcing",
        choices=("scheduled", "none"),
        default="scheduled",
    )
    controller_distill.add_argument(
        "--causal-lm-head",
        type=Path,
        help="optional frozen vocabulary matrix for top-k causal distillation",
    )
    controller_distill.add_argument(
        "--causal-norm-weight",
        type=Path,
        help="final RMSNorm weight paired with --causal-lm-head",
    )
    controller_distill.add_argument(
        "--causal-weight",
        type=float,
        default=0.0,
        help="weight for the optional trace top-k causal objective",
    )
    controller_distill.add_argument("--seed", type=int, default=37)

    analyze = commands.add_parser(
        "analyze-mlp", help="run Gate 1 contribution-magnitude oracle"
    )
    analyze.add_argument("--model", required=True)
    analyze.add_argument("--traces", required=True)
    analyze.add_argument("--out", required=True, type=Path)
    analyze.add_argument("--max-records", type=int)

    semantic = commands.add_parser(
        "build-semantic", help="extract and quantize semantic memory"
    )
    semantic.add_argument("--model", required=True)
    semantic.add_argument("--out", required=True, type=Path)
    semantic.add_argument("--key-bits", type=int, default=8)
    semantic.add_argument("--value-codebooks", type=int, default=2)
    semantic.add_argument("--value-codebook-size", type=int, default=16)
    semantic.add_argument("--ivf-clusters", type=int, default=32)
    semantic.add_argument("--ivf-iterations", type=int, default=20)

    dip_package = commands.add_parser(
        "build-dip-package",
        help="extract an experimental coordinate-major DIP package",
    )
    dip_package.add_argument("--model", required=True)
    dip_package.add_argument("--out", required=True, type=Path)
    dip_package.add_argument("--layers", nargs="+", type=int)
    dip_package.add_argument(
        "--dual-layout-experimental",
        action="store_true",
        help="also duplicate gate/up rows for the rejected record-major diagnostic",
    )

    routing = commands.add_parser(
        "evaluate-semantic", help="run Gate 2 practical routing evaluation"
    )
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

    attention = commands.add_parser(
        "evaluate-attention", help="run synthetic Gate 3 attention replacement"
    )
    attention.add_argument("--out", required=True, type=Path)
    attention.add_argument("--seed", type=int, default=41)
    attention.add_argument("--length", type=int, default=128)
    attention.add_argument("--key-width", type=int, default=16)
    attention.add_argument("--value-width", type=int, default=16)
    attention.add_argument("--local-window", type=int, default=16)

    bitnet_attention = commands.add_parser(
        "evaluate-native-bitnet-attention",
        help="run trained-model Milestone 3 substitutions on a BitNet package",
    )
    bitnet_attention.add_argument("--model", required=True, type=Path)
    bitnet_attention.add_argument("--dataset", required=True, type=Path)
    bitnet_attention.add_argument("--out", required=True, type=Path)
    bitnet_attention.add_argument("--library", type=Path)
    bitnet_attention.add_argument("--threads", type=int)
    bitnet_attention.add_argument("--native-projections", action="store_true")
    bitnet_attention.add_argument("--sequence-count", type=int, default=2)
    bitnet_attention.add_argument("--prediction-positions", type=int, default=32)
    bitnet_attention.add_argument("--record-offset", type=int, default=0)
    bitnet_attention.add_argument(
        "--modes",
        nargs="+",
        choices=(
            "local",
            "recurrent",
            "retrieval",
            "hybrid",
            "indexed_hybrid",
            "bounded_hybrid",
            "streaming_hybrid",
            "native_streaming",
        ),
        default=("local", "recurrent", "retrieval", "hybrid"),
    )
    bitnet_attention.add_argument("--layers", nargs="+", type=int)
    bitnet_attention.add_argument("--local-window", type=int, default=16)
    bitnet_attention.add_argument("--recurrent-decay", type=float, default=0.99)
    bitnet_attention.add_argument("--retrieval-top-k", type=int, default=4)
    bitnet_attention.add_argument("--older-weight", type=float, default=0.5)
    bitnet_attention.add_argument("--retrieval-candidates", type=int, default=12)
    bitnet_attention.add_argument("--lsh-tables", type=int, default=4)
    bitnet_attention.add_argument("--lsh-bits", type=int, default=8)
    bitnet_attention.add_argument("--lsh-radius", type=int, default=1)
    bitnet_attention.add_argument("--lsh-seed", type=int, default=314159)
    bitnet_attention.add_argument("--page-size", type=int, default=8)
    bitnet_attention.add_argument(
        "--page-bound", choices=("box", "sphere"), default="sphere"
    )
    bitnet_attention.add_argument("--sink-tokens", type=int, default=2)
    bitnet_attention.add_argument("--attention-library", type=Path)

    controller_substitution = commands.add_parser(
        "evaluate-native-bitnet-controller",
        help=(
            "replay compiled BitNet semantic and episodic outputs through "
            "the transformer-free controller boundary"
        ),
    )
    controller_substitution.add_argument("--model", required=True, type=Path)
    controller_substitution.add_argument("--dataset", required=True, type=Path)
    controller_substitution.add_argument("--controller", required=True, type=Path)
    controller_substitution.add_argument("--out", required=True, type=Path)
    controller_substitution.add_argument("--library", type=Path)
    controller_substitution.add_argument("--attention-library", type=Path)
    controller_substitution.add_argument("--threads", type=int)
    controller_substitution.add_argument(
        "--no-native-projections",
        action="store_true",
    )
    controller_substitution.add_argument("--sequence-count", type=int, default=2)
    controller_substitution.add_argument("--prediction-positions", type=int, default=32)
    controller_substitution.add_argument("--record-offset", type=int, default=0)
    controller_substitution.add_argument("--local-window", type=int, default=16)
    controller_substitution.add_argument("--retrieval-candidates", type=int, default=8)
    controller_substitution.add_argument("--retrieval-top-k", type=int, default=4)
    controller_substitution.add_argument("--sink-tokens", type=int, default=2)

    controller_only = commands.add_parser(
        "evaluate-controller-only",
        help=(
            "replay a serialized controller on operator streams without loading "
            "a Transformers model"
        ),
    )
    controller_only.add_argument("--trace", required=True, type=Path)
    controller_only.add_argument("--controller", required=True, type=Path)
    controller_only.add_argument("--out", required=True, type=Path)
    controller_only.add_argument(
        "--allow-correction",
        action="store_true",
        help="allow an unauthenticated nonzero factorized correction",
    )

    fit_operator_provider = commands.add_parser(
        "fit-operator-provider",
        help=(
            "fit a compact CPU operator-stream provider from a controller trace"
        ),
    )
    fit_operator_provider.add_argument("--trace", required=True, type=Path)
    fit_operator_provider.add_argument("--out", required=True, type=Path)
    fit_operator_provider.add_argument("--output-rank", type=int, default=16)
    fit_operator_provider.add_argument("--ridge", type=float, default=1e-2)
    fit_operator_provider.add_argument(
        "--target",
        choices=("streams", "combined_stream", "combined_delta"),
        default="streams",
    )

    distill_operator_provider = commands.add_parser(
        "distill-operator-provider",
        help="jointly adapt provider projections through a frozen controller",
    )
    distill_operator_provider.add_argument("--provider", required=True, type=Path)
    distill_operator_provider.add_argument("--controller", required=True, type=Path)
    distill_operator_provider.add_argument("--trace", required=True, type=Path)
    distill_operator_provider.add_argument("--validation-trace", type=Path)
    distill_operator_provider.add_argument("--out", required=True, type=Path)
    distill_operator_provider.add_argument("--steps", type=int, default=100)
    distill_operator_provider.add_argument("--batch-size", type=int, default=4)
    distill_operator_provider.add_argument("--learning-rate", type=float, default=1e-3)
    distill_operator_provider.add_argument("--seed", type=int, default=37)
    distill_operator_provider.add_argument("--device", default="cpu")

    distill_state_space = commands.add_parser(
        "distill-state-space-provider",
        help="distill a causal diagonal state-space operator provider",
    )
    distill_state_space.add_argument("--provider", required=True, type=Path)
    distill_state_space.add_argument("--controller", required=True, type=Path)
    distill_state_space.add_argument("--trace", required=True, type=Path)
    distill_state_space.add_argument("--validation-trace", type=Path)
    distill_state_space.add_argument("--out", required=True, type=Path)
    distill_state_space.add_argument("--steps", type=int, default=80)
    distill_state_space.add_argument("--batch-size", type=int, default=8)
    distill_state_space.add_argument("--memory-dim", type=int, default=64)
    distill_state_space.add_argument("--projection-width", type=int, default=64)
    distill_state_space.add_argument("--learning-rate", type=float, default=2e-3)
    distill_state_space.add_argument("--seed", type=int, default=81)
    distill_state_space.add_argument("--device", default="cpu")

    distill_state_space_residual = commands.add_parser(
        "distill-state-space-residual-provider",
        help="distill a persistent-memory residual over a full PCA provider",
    )
    distill_state_space_residual.add_argument("--provider", required=True, type=Path)
    distill_state_space_residual.add_argument("--controller", required=True, type=Path)
    distill_state_space_residual.add_argument("--trace", required=True, type=Path)
    distill_state_space_residual.add_argument("--validation-trace", type=Path)
    distill_state_space_residual.add_argument("--out", required=True, type=Path)
    distill_state_space_residual.add_argument("--steps", type=int, default=40)
    distill_state_space_residual.add_argument("--batch-size", type=int, default=8)
    distill_state_space_residual.add_argument("--memory-dim", type=int, default=64)
    distill_state_space_residual.add_argument("--learning-rate", type=float, default=2e-3)
    distill_state_space_residual.add_argument("--seed", type=int, default=91)
    distill_state_space_residual.add_argument("--device", default="cpu")

    adapt_controller = commands.add_parser(
        "adapt-controller-correction",
        help="adapt only controller correction tensors over fixed provider streams",
    )
    adapt_controller.add_argument("--provider", required=True, type=Path)
    adapt_controller.add_argument("--controller", required=True, type=Path)
    adapt_controller.add_argument("--trace", required=True, type=Path)
    adapt_controller.add_argument("--validation-trace", type=Path)
    adapt_controller.add_argument("--out", required=True, type=Path)
    adapt_controller.add_argument("--steps", type=int, default=50)
    adapt_controller.add_argument("--batch-size", type=int, default=8)
    adapt_controller.add_argument("--learning-rate", type=float, default=2e-3)
    adapt_controller.add_argument("--seed", type=int, default=55)
    adapt_controller.add_argument("--device", default="cpu")

    dagger_provider = commands.add_parser(
        "dagger-refit-operator-provider",
        help="refit provider projections on states visited by causal rollout",
    )
    dagger_provider.add_argument("--provider", required=True, type=Path)
    dagger_provider.add_argument("--controller", required=True, type=Path)
    dagger_provider.add_argument("--trace", required=True, type=Path)
    dagger_provider.add_argument("--validation-trace", type=Path)
    dagger_provider.add_argument("--out", required=True, type=Path)
    dagger_provider.add_argument("--iterations", type=int, default=2)
    dagger_provider.add_argument("--ridge", type=float, default=1.0)

    nonlinear_provider = commands.add_parser(
        "distill-nonlinear-residual-provider",
        help="distill a shared nonlinear latent residual over a PCA provider",
    )
    nonlinear_provider.add_argument("--provider", required=True, type=Path)
    nonlinear_provider.add_argument("--controller", required=True, type=Path)
    nonlinear_provider.add_argument("--trace", required=True, type=Path)
    nonlinear_provider.add_argument("--validation-trace", type=Path)
    nonlinear_provider.add_argument("--out", required=True, type=Path)
    nonlinear_provider.add_argument("--steps", type=int, default=100)
    nonlinear_provider.add_argument("--teacher-forcing-steps", type=int, default=0)
    nonlinear_provider.add_argument(
        "--teacher-forcing-decay-steps",
        type=int,
        default=0,
        help="linearly decay teacher-forcing probability to zero over this many steps",
    )
    nonlinear_provider.add_argument("--batch-size", type=int, default=8)
    nonlinear_provider.add_argument("--hidden-width", type=int, default=64)
    nonlinear_provider.add_argument("--stage-width", type=int, default=16)
    nonlinear_provider.add_argument("--learning-rate", type=float, default=3e-4)
    nonlinear_provider.add_argument("--seed", type=int, default=403)
    nonlinear_provider.add_argument("--device", default="cpu")

    evaluate_operator_provider = commands.add_parser(
        "evaluate-controller-provider",
        help="evaluate a learned operator provider with a transformer-free controller",
    )
    evaluate_operator_provider.add_argument("--trace", required=True, type=Path)
    evaluate_operator_provider.add_argument("--provider", required=True, type=Path)
    evaluate_operator_provider.add_argument("--controller", required=True, type=Path)
    evaluate_operator_provider.add_argument("--out", required=True, type=Path)
    evaluate_operator_provider.add_argument(
        "--allow-correction",
        action="store_true",
        help="allow an unauthenticated nonzero factorized correction",
    )

    evaluate_sequence_provider = commands.add_parser(
        "evaluate-controller-sequence",
        help="validate persistent sequence replay without a Transformer model",
    )
    evaluate_sequence_provider.add_argument("--trace", required=True, type=Path)
    evaluate_sequence_provider.add_argument("--provider", required=True, type=Path)
    evaluate_sequence_provider.add_argument("--controller", required=True, type=Path)
    evaluate_sequence_provider.add_argument("--out", required=True, type=Path)

    attention_benchmark = commands.add_parser(
        "benchmark-native-attention",
        help="benchmark the bounded native attention cache at increasing contexts",
    )
    attention_benchmark.add_argument("--out", required=True, type=Path)
    attention_benchmark.add_argument("--library", type=Path)
    attention_benchmark.add_argument(
        "--lengths", nargs="+", type=int, default=(33, 128, 512, 2048)
    )
    attention_benchmark.add_argument("--local-window", type=int, default=16)
    attention_benchmark.add_argument("--candidates", type=int, default=8)
    attention_benchmark.add_argument("--top-k", type=int, default=4)
    attention_benchmark.add_argument("--sink-tokens", type=int, default=2)

    generation_benchmark = commands.add_parser(
        "benchmark-native-bitnet-generation",
        help="benchmark complete bounded-attention generation from a BitNet package",
    )
    generation_benchmark.add_argument("--model", required=True, type=Path)
    generation_benchmark.add_argument("--out", required=True, type=Path)
    generation_benchmark.add_argument(
        "--prompt",
        default="The purpose of a semantic memory system is",
    )
    generation_benchmark.add_argument(
        "--lengths", nargs="+", type=int, default=(33, 128, 256)
    )
    generation_benchmark.add_argument("--max-tokens", type=int, default=4)
    generation_benchmark.add_argument("--mlp-library", type=Path)
    generation_benchmark.add_argument("--attention-library", type=Path)
    generation_benchmark.add_argument("--threads", type=int)
    generation_benchmark.add_argument("--native-projections", action="store_true")
    generation_benchmark.add_argument("--local-window", type=int, default=16)
    generation_benchmark.add_argument("--candidates", type=int, default=8)
    generation_benchmark.add_argument("--top-k", type=int, default=4)
    generation_benchmark.add_argument("--sink-tokens", type=int, default=2)

    generation_evaluation = commands.add_parser(
        "evaluate-native-bitnet-generation",
        help="run sustained greedy generation over a JSONL prompt suite",
    )
    generation_evaluation.add_argument("--model", required=True, type=Path)
    generation_evaluation.add_argument("--prompts", required=True, type=Path)
    generation_evaluation.add_argument("--out", required=True, type=Path)
    generation_evaluation.add_argument("--max-tokens", type=int, default=16)
    generation_evaluation.add_argument("--mlp-library", type=Path)
    generation_evaluation.add_argument("--attention-library", type=Path)
    generation_evaluation.add_argument("--threads", type=int)

    dip_token_generation = commands.add_parser(
        "evaluate-native-bitnet-dip-token-generation",
        help="confirm packaged DIP generation through the pure C++ token runtime",
    )
    dip_token_generation.add_argument("--model", required=True, type=Path)
    dip_token_generation.add_argument("--executable", required=True, type=Path)
    dip_token_generation.add_argument("--prompts", required=True, type=Path)
    dip_token_generation.add_argument("--reference", required=True, type=Path)
    dip_token_generation.add_argument(
        "--package-manifest-sha256",
        required=True,
        help="expected SHA-256 of the complete derived package manifest",
    )
    dip_token_generation.add_argument(
        "--executable-sha256",
        required=True,
        help="expected SHA-256 of the statically bound native executable",
    )
    dip_token_generation.add_argument("--out", required=True, type=Path)
    dip_token_generation.add_argument("--max-tokens", type=int, default=4)
    dip_token_generation.add_argument("--threads", type=int, default=12)
    dip_token_generation.add_argument(
        "--no-verify-reset",
        action="store_true",
    )
    dip_token_generation.add_argument("--timeout", type=float, default=300.0)

    dip_attention = commands.add_parser(
        "evaluate-native-bitnet-dip-attention",
        help="confirm sustained native DIP attention eviction and retrieval",
    )
    dip_attention.add_argument("--model", required=True, type=Path)
    dip_attention.add_argument("--library", required=True, type=Path)
    dip_attention.add_argument("--out", required=True, type=Path)
    dip_attention.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=(16, 17, 18, 24, 32),
    )
    dip_attention.add_argument(
        "--prompt",
        default="The memory system should preserve relevant earlier context.",
    )
    dip_attention.add_argument("--threads", type=int)

    controller_generation = commands.add_parser(
        "evaluate-native-bitnet-controller-generation",
        help=(
            "compare incremental bounded generation with and without "
            "decoder-layer residual scaffolding"
        ),
    )
    controller_generation.add_argument("--model", required=True, type=Path)
    controller_generation.add_argument("--controller", required=True, type=Path)
    controller_generation.add_argument("--prompts", required=True, type=Path)
    controller_generation.add_argument("--out", required=True, type=Path)
    controller_generation.add_argument("--max-tokens", type=int, default=4)
    controller_generation.add_argument("--mlp-library", type=Path)
    controller_generation.add_argument("--attention-library", type=Path)
    controller_generation.add_argument("--threads", type=int)

    compile_command = commands.add_parser(
        "compile", help="compile a runnable Engram package"
    )
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

    compile_bitnet = commands.add_parser(
        "compile-native-bitnet",
        help="compile a source-independent package around the direct BitNet kernel",
    )
    compile_bitnet.add_argument("--model", required=True)
    compile_bitnet.add_argument("--artifact", required=True, type=Path)
    compile_bitnet.add_argument("--artifact-sha256", required=True)
    compile_bitnet.add_argument("--out", required=True, type=Path)
    compile_bitnet.add_argument("--revision")
    compile_bitnet.add_argument("--cache-dir", type=Path)
    compile_bitnet.add_argument("--threads", type=int, default=12)

    install_bitnet_controller = commands.add_parser(
        "install-native-bitnet-controller",
        help="install an authenticated schema-v3 controller into a BitNet package",
    )
    install_bitnet_controller.add_argument("--model", required=True, type=Path)
    install_bitnet_controller.add_argument("--controller", required=True, type=Path)

    install_bitnet_semantic = commands.add_parser(
        "install-native-bitnet-semantic-memory",
        help="derive a BitNet package with the adjudicated DIP v2 index",
    )
    install_bitnet_semantic.add_argument("--model", required=True, type=Path)
    install_bitnet_semantic.add_argument("--index", required=True, type=Path)
    install_bitnet_semantic.add_argument("--policy", required=True, type=Path)
    install_bitnet_semantic.add_argument("--adjudication", required=True, type=Path)
    install_bitnet_semantic.add_argument("--out", required=True, type=Path)
    install_bitnet_semantic.add_argument("--index-sha256", required=True)
    install_bitnet_semantic.add_argument("--policy-sha256", required=True)
    install_bitnet_semantic.add_argument("--adjudication-sha256", required=True)

    generate = commands.add_parser(
        "generate", help="generate with the PyTorch-free reference runtime"
    )
    generate.add_argument("--model", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-tokens", type=int, default=16)
    generate.add_argument("--exact-vocab", action="store_true")

    generate_bitnet = commands.add_parser(
        "generate-native-bitnet",
        help="generate from a compiled native BitNet package on CPU",
    )
    generate_bitnet.add_argument("--model", required=True, type=Path)
    generate_bitnet.add_argument("--prompt", required=True)
    generate_bitnet.add_argument("--max-tokens", type=int, default=16)
    generate_bitnet.add_argument("--library", type=Path)
    generate_bitnet.add_argument("--threads", type=int)

    generate_bitnet_controller = commands.add_parser(
        "generate-native-bitnet-controller",
        help="generate through the package-owned native residual controller",
    )
    generate_bitnet_controller.add_argument("--model", required=True, type=Path)
    generate_bitnet_controller.add_argument("--prompt", required=True)
    generate_bitnet_controller.add_argument("--max-tokens", type=int, default=16)
    generate_bitnet_controller.add_argument("--library", type=Path)
    generate_bitnet_controller.add_argument("--attention-library", type=Path)
    generate_bitnet_controller.add_argument("--threads", type=int)
    generate_bitnet.add_argument("--native-projections", action="store_true")
    generate_bitnet.add_argument("--bounded-attention", action="store_true")
    generate_bitnet.add_argument("--attention-library", type=Path)
    generate_bitnet.add_argument("--local-window", type=int, default=16)
    generate_bitnet.add_argument("--candidates", type=int, default=8)
    generate_bitnet.add_argument("--top-k", type=int, default=4)
    generate_bitnet.add_argument("--sink-tokens", type=int, default=2)

    chat_bitnet = commands.add_parser(
        "chat-native-bitnet",
        help=(
            "chat through the authenticated CPU-only native BitNet DIP token runtime"
        ),
    )
    chat_bitnet.add_argument("--model", required=True, type=Path)
    chat_bitnet.add_argument("--max-tokens", type=int, default=32)
    chat_bitnet.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="system message rendered by the packaged tokenizer chat template",
    )
    chat_bitnet.add_argument(
        "--library",
        type=Path,
        help=(
            "libengram_bitnet_token_runtime shared library; defaults to "
            "build/libengram_bitnet_token_runtime.so"
        ),
    )
    chat_bitnet.add_argument("--threads", type=int)

    validate = commands.add_parser(
        "validate", help="verify package checksums and deterministic generation"
    )
    validate.add_argument("--model", required=True)

    benchmark = commands.add_parser(
        "benchmark", help="benchmark the Python reference runtime"
    )
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--tokens", type=int, default=32)
    benchmark.add_argument("--enable-transition-cache", action="store_true")

    quality = commands.add_parser(
        "evaluate-e2e", help="measure Gate 5 against a Hugging Face teacher"
    )
    quality.add_argument("--model", required=True, help="compiled .engram package")
    quality.add_argument(
        "--teacher", required=True, help="local checkpoint or Hugging Face model ID"
    )
    quality.add_argument("--dataset", required=True, type=Path)
    quality.add_argument("--out", required=True, type=Path)
    quality.add_argument("--max-records", type=int)

    controller_gate = commands.add_parser(
        "evaluate-controller", help="run synthetic Gate 4 controller instrumentation"
    )
    controller_gate.add_argument("--out", required=True, type=Path)
    controller_gate.add_argument("--seed", type=int, default=73)
    controller_gate.add_argument("--samples", type=int, default=64)
    controller_gate.add_argument("--width", type=int, default=16)

    intervention = commands.add_parser(
        "evaluate-mlp-intervention",
        help="substitute sparse MLP outputs inside a Hugging Face teacher",
    )
    intervention.add_argument("--model", required=True)
    intervention.add_argument("--dataset", required=True, type=Path)
    intervention.add_argument("--calibration-traces")
    intervention.add_argument("--out", required=True, type=Path)
    intervention.add_argument(
        "--variants",
        nargs="+",
        choices=("identity", "oracle", "rank16", "overlap", "dip", "dip_paq"),
        default=("identity", "oracle"),
    )
    intervention.add_argument("--top-k", nargs="+", type=int, default=(256,))
    intervention.add_argument(
        "--layer-top-k",
        nargs="+",
        type=int,
        help="evaluate one all-layer oracle arm with a per-layer active budget",
    )
    intervention.add_argument("--candidates", nargs="+", type=int, default=(512,))
    intervention.add_argument(
        "--input-fractions", nargs="+", type=float, default=(0.75,)
    )
    intervention.add_argument("--rank", type=int, default=16)
    intervention.add_argument("--regularization", type=float, default=1000.0)
    intervention.add_argument("--calibration-records", type=int, default=128)
    intervention.add_argument("--posting-groups", type=int, default=96)
    intervention.add_argument("--posting-size", type=int, default=32)
    intervention.add_argument("--overlap-iterations", type=int, default=8)
    intervention.add_argument("--max-replication", type=int, default=4)
    intervention.add_argument("--layers", nargs="+", type=int)
    intervention.add_argument(
        "--layer-mode", choices=("all", "individual", "both"), default="both"
    )
    intervention.add_argument("--max-records", type=int)
    intervention.add_argument("--device", default="cpu")
    intervention.add_argument("--allow-calibration-overlap", action="store_true")
    intervention.add_argument(
        "--evaluation-role",
        choices=("development", "confirmation"),
        default="development",
    )
    intervention.add_argument("--configuration-selection-traces")
    intervention.add_argument("--paq-group-size", type=int, default=8)
    intervention.add_argument("--paq-codebooks", type=int, default=2)
    intervention.add_argument("--paq-codebook-size", type=int, default=128)
    intervention.add_argument("--paq-iterations", type=int, default=8)
    intervention.add_argument("--paq-sample-limit", type=int, default=8192)
    intervention.add_argument("--paq-seed", type=int, default=73)
    intervention.add_argument(
        "--paq-cacheline-amplification",
        type=float,
        default=12.0 / 11.0,
        help="physical/logical packed-code traffic multiplier",
    )

    router_sweep = commands.add_parser(
        "sweep-rank-router",
        help="screen low-rank router regularization using cached held-out traces",
    )
    router_sweep.add_argument("--model", required=True)
    router_sweep.add_argument("--calibration-traces", required=True)
    router_sweep.add_argument("--validation-traces", required=True)
    router_sweep.add_argument("--out", required=True, type=Path)
    router_sweep.add_argument("--cache", required=True, type=Path)
    router_sweep.add_argument("--regularization", nargs="+", type=float, required=True)
    router_sweep.add_argument("--top-k", type=int, default=768)
    router_sweep.add_argument("--candidates", nargs="+", type=int, default=(1280,))
    router_sweep.add_argument("--rank", type=int, default=16)
    router_sweep.add_argument("--calibration-records", type=int)
    router_sweep.add_argument("--validation-records", type=int)

    dip_sweep = commands.add_parser(
        "sweep-dip",
        help="screen predictor-free Dynamic Input Pruning with exact candidate completion",
    )
    dip_sweep.add_argument("--model", required=True)
    dip_sweep.add_argument("--validation-traces", required=True)
    dip_sweep.add_argument("--out", required=True, type=Path)
    dip_sweep.add_argument(
        "--input-fractions", nargs="+", type=float, default=(0.5, 0.625, 0.75)
    )
    dip_sweep.add_argument("--top-k", type=int, default=768)
    dip_sweep.add_argument(
        "--candidates", nargs="+", type=int, default=(896, 1024, 1152)
    )
    dip_sweep.add_argument("--validation-records", type=int)
    dip_sweep.add_argument(
        "--input-block-size",
        type=int,
        help="select contiguous input blocks (16 float32 values = one cache line)",
    )

    intrinsic_sparse = commands.add_parser(
        "sweep-intrinsic-sparsity",
        help="screen exact full-gate selection with sparse SiLU/ReLU activations",
    )
    intrinsic_sparse.add_argument("--model", required=True)
    intrinsic_sparse.add_argument("--calibration-traces", required=True)
    intrinsic_sparse.add_argument("--validation-traces", required=True)
    intrinsic_sparse.add_argument("--out", required=True, type=Path)
    intrinsic_sparse.add_argument(
        "--sparsities",
        nargs="+",
        type=float,
        default=(0.5, 0.7, 0.8, 0.825, 0.85, 0.9),
    )
    intrinsic_sparse.add_argument(
        "--activations",
        nargs="+",
        choices=("cats_silu", "fatrelu"),
        default=("cats_silu", "fatrelu"),
    )
    intrinsic_sparse.add_argument("--calibration-records", type=int)
    intrinsic_sparse.add_argument("--validation-records", type=int)
    intrinsic_sparse.add_argument(
        "--maximum-mean-relative-l2", type=float, default=0.18
    )
    intrinsic_sparse.add_argument(
        "--maximum-traffic-fraction", type=float, default=0.45
    )

    intrinsic_sparse_train = commands.add_parser(
        "train-intrinsic-sparse-boundaries",
        help="co-adapt exact thresholded SwiGLU records on cached teacher boundaries",
    )
    intrinsic_sparse_train.add_argument("--model", required=True)
    intrinsic_sparse_train.add_argument("--training-traces", required=True)
    intrinsic_sparse_train.add_argument("--validation-traces", required=True)
    intrinsic_sparse_train.add_argument("--out", required=True, type=Path)
    intrinsic_sparse_train.add_argument("--layers", nargs="+", type=int, required=True)
    intrinsic_sparse_train.add_argument("--target-sparsity", type=float, default=0.85)
    intrinsic_sparse_train.add_argument("--initial-artifact", type=Path)
    intrinsic_sparse_train.add_argument("--steps", type=int, default=128)
    intrinsic_sparse_train.add_argument("--batch-size", type=int, default=64)
    intrinsic_sparse_train.add_argument("--learning-rate", type=float, default=1e-4)
    intrinsic_sparse_train.add_argument("--sparsity-weight", type=float, default=1.0)
    intrinsic_sparse_train.add_argument("--cosine-weight", type=float, default=0.1)
    intrinsic_sparse_train.add_argument(
        "--temperature-fraction", type=float, default=0.1
    )
    intrinsic_sparse_train.add_argument("--warmup-steps", type=int, default=16)
    intrinsic_sparse_train.add_argument(
        "--start-threshold-fraction", type=float, default=0.0
    )
    intrinsic_sparse_train.add_argument("--evaluation-interval", type=int, default=16)
    intrinsic_sparse_train.add_argument(
        "--maximum-mean-relative-l2", type=float, default=0.18
    )
    intrinsic_sparse_train.add_argument(
        "--maximum-traffic-fraction", type=float, default=0.45
    )
    intrinsic_sparse_train.add_argument("--max-train-records", type=int, default=4096)
    intrinsic_sparse_train.add_argument(
        "--max-validation-records", type=int, default=2048
    )
    intrinsic_sparse_train.add_argument("--seed", type=int, default=1729)
    intrinsic_sparse_train.add_argument("--device", default="cpu")

    fully_sparse_train = commands.add_parser(
        "train-fully-sparse-boundaries",
        help="train exact top-K input/intermediate sparse MLPs on teacher boundaries",
    )
    fully_sparse_train.add_argument("--model", required=True)
    fully_sparse_train.add_argument("--training-traces", required=True)
    fully_sparse_train.add_argument("--validation-traces", required=True)
    fully_sparse_train.add_argument("--out", required=True, type=Path)
    fully_sparse_train.add_argument("--layers", nargs="+", type=int, required=True)
    fully_sparse_train.add_argument("--input-fraction", type=float, default=0.49)
    fully_sparse_train.add_argument("--intermediate-fraction", type=float, default=0.34)
    fully_sparse_train.add_argument("--initial-artifact", type=Path)
    fully_sparse_train.add_argument("--steps", type=int, default=1024)
    fully_sparse_train.add_argument("--warmup-steps", type=int, default=128)
    fully_sparse_train.add_argument("--start-sparse-fraction", type=float, default=0.0)
    fully_sparse_train.add_argument("--batch-size", type=int, default=128)
    fully_sparse_train.add_argument("--learning-rate", type=float, default=1e-4)
    fully_sparse_train.add_argument("--cosine-weight", type=float, default=0.1)
    fully_sparse_train.add_argument("--evaluation-interval", type=int, default=64)
    fully_sparse_train.add_argument(
        "--maximum-mean-relative-l2", type=float, default=0.18
    )
    fully_sparse_train.add_argument(
        "--maximum-traffic-fraction", type=float, default=0.45
    )
    fully_sparse_train.add_argument("--max-train-records", type=int, default=32768)
    fully_sparse_train.add_argument("--max-validation-records", type=int, default=2048)
    fully_sparse_train.add_argument("--seed", type=int, default=2718)
    fully_sparse_train.add_argument("--device", default="cpu")

    correction_sweep = commands.add_parser(
        "sweep-correction-capsules",
        help="fit state-selected low-rank corrections to routed MLP residuals",
    )
    correction_sweep.add_argument("--model", required=True)
    correction_sweep.add_argument("--calibration-traces", required=True)
    correction_sweep.add_argument("--validation-traces", required=True)
    correction_sweep.add_argument("--membership-cache", required=True, type=Path)
    correction_sweep.add_argument("--out", required=True, type=Path)
    correction_sweep.add_argument("--router-rank", type=int, default=16)
    correction_sweep.add_argument("--router-regularization", type=float, default=8000.0)
    correction_sweep.add_argument("--top-k", type=int, default=768)
    correction_sweep.add_argument("--candidates", type=int, default=1280)
    correction_sweep.add_argument("--capsules", nargs="+", type=int, default=(1, 4, 8))
    correction_sweep.add_argument(
        "--capsule-ranks", nargs="+", type=int, default=(8, 16)
    )
    correction_sweep.add_argument("--capsule-ridge", type=float, default=1000.0)
    correction_sweep.add_argument("--capsule-iterations", type=int, default=8)
    correction_sweep.add_argument("--radius-scale", type=float, default=1.25)
    correction_sweep.add_argument(
        "--priority-fractions", nargs="+", type=float, default=(1.0,)
    )
    correction_sweep.add_argument("--radius-quantile", type=float, default=1.0)
    correction_sweep.add_argument("--calibration-records", type=int)
    correction_sweep.add_argument("--validation-records", type=int)

    corpus = commands.add_parser(
        "build-distillation-corpus",
        help="tokenize local prose/code into a deterministic student-training corpus",
    )
    corpus.add_argument("--model", required=True)
    corpus.add_argument("--input", required=True, nargs="+", type=Path)
    corpus.add_argument("--out", required=True, type=Path)
    corpus.add_argument("--sequence-length", type=int, default=128)
    corpus.add_argument("--max-sequences", type=int, default=128)
    corpus.add_argument("--minimum-tokens", type=int, default=16)

    holdout = commands.add_parser(
        "build-distillation-holdout",
        help="reserve an authenticated tail shard from a pretokenized corpus",
    )
    holdout.add_argument("--source", required=True, type=Path)
    holdout.add_argument("--out", required=True, type=Path)
    holdout.add_argument("--records", type=int, default=128)

    sparse_train = commands.add_parser(
        "train-sparse-student",
        help="distill a frozen-base sparse student with trainable routers and MLP adapters",
    )
    sparse_train.add_argument("--model", required=True)
    sparse_train.add_argument("--calibration-dataset", required=True, type=Path)
    sparse_train.add_argument("--training-dataset", type=Path)
    sparse_train.add_argument("--validation-dataset", required=True, type=Path)
    sparse_train.add_argument("--calibration-traces", required=True)
    sparse_train.add_argument("--out", required=True, type=Path)
    sparse_train.add_argument("--top-k", type=int, default=512)
    sparse_train.add_argument("--candidates", type=int, default=512)
    sparse_train.add_argument("--router-rank", type=int, default=16)
    sparse_train.add_argument("--router-regularization", type=float, default=8000.0)
    sparse_train.add_argument("--adapter-rank", type=int, default=8)
    sparse_train.add_argument("--residual-rank", type=int, default=0)
    sparse_train.add_argument("--epochs", type=int, default=1)
    sparse_train.add_argument("--learning-rate", type=float, default=1e-4)
    sparse_train.add_argument("--router-learning-rate", type=float, default=1e-3)
    sparse_train.add_argument("--local-weight", type=float, default=1.0)
    sparse_train.add_argument("--hidden-weight", type=float, default=0.25)
    sparse_train.add_argument("--logit-weight", type=float, default=0.25)
    sparse_train.add_argument("--teacher-forced-local-weight", type=float, default=0.0)
    sparse_train.add_argument("--router-weight", type=float, default=0.1)
    sparse_train.add_argument("--locality-weight", type=float, default=0.05)
    sparse_train.add_argument(
        "--routing-mode",
        choices=("hard_router", "hardware_ste"),
        default="hardware_ste",
    )
    sparse_train.add_argument("--input-fraction", type=float, default=0.625)
    sparse_train.add_argument("--temperature", type=float, default=1.0)
    sparse_train.add_argument("--cache-line-records", type=int, default=16)
    sparse_train.add_argument("--batch-size", type=int, default=4)
    sparse_train.add_argument("--gradient-diagnostics", action="store_true")
    sparse_train.add_argument("--checkpoint-every", type=int, default=0)
    sparse_train.add_argument("--resume", action="store_true")
    sparse_train.add_argument("--layers", nargs="+", type=int)
    sparse_train.add_argument("--exact-dense-start", action="store_true")
    sparse_train.add_argument("--dense-warmup-steps", type=int, default=0)
    sparse_train.add_argument("--start-top-k", type=int)
    sparse_train.add_argument("--start-candidates", type=int)
    sparse_train.add_argument("--start-input-fraction", type=float)
    sparse_train.add_argument("--anneal-steps", type=int, default=0)
    sparse_train.add_argument("--start-temperature", type=float)
    sparse_train.add_argument("--checkpoint-selection-records", type=int, default=0)
    sparse_train.add_argument("--checkpoint-selection-every", type=int, default=0)
    sparse_train.add_argument("--router-group-size", type=int, default=1)
    sparse_train.add_argument("--train-full-mlp", action="store_true")
    sparse_train.add_argument("--max-train-records", type=int)
    sparse_train.add_argument("--max-validation-records", type=int)
    sparse_train.add_argument("--device", default="cpu")

    grouped_boundaries = commands.add_parser(
        "train-grouped-sparse-boundaries",
        help="fit cache-aligned grouped sparse MLPs on stable teacher boundaries",
    )
    grouped_boundaries.add_argument("--model", required=True)
    grouped_boundaries.add_argument("--training-traces", required=True)
    grouped_boundaries.add_argument("--validation-traces", required=True)
    grouped_boundaries.add_argument("--out", required=True, type=Path)
    grouped_boundaries.add_argument("--layers", nargs="+", type=int, required=True)
    grouped_boundaries.add_argument("--top-k", type=int, default=672)
    grouped_boundaries.add_argument("--start-top-k", type=int, default=768)
    grouped_boundaries.add_argument("--router-rank", type=int, default=16)
    grouped_boundaries.add_argument(
        "--router-regularization", type=float, default=8000.0
    )
    grouped_boundaries.add_argument("--adapter-rank", type=int, default=8)
    grouped_boundaries.add_argument("--router-warmup-steps", type=int, default=32)
    grouped_boundaries.add_argument("--anneal-steps", type=int, default=64)
    grouped_boundaries.add_argument("--settle-steps", type=int, default=128)
    grouped_boundaries.add_argument("--batch-size", type=int, default=256)
    grouped_boundaries.add_argument("--learning-rate", type=float, default=1e-4)
    grouped_boundaries.add_argument("--router-learning-rate", type=float, default=1e-3)
    grouped_boundaries.add_argument("--route-weight", type=float, default=0.1)
    grouped_boundaries.add_argument("--cosine-weight", type=float, default=0.1)
    grouped_boundaries.add_argument("--dense-anchor-weight", type=float, default=0.1)
    grouped_boundaries.add_argument("--start-temperature", type=float, default=1.0)
    grouped_boundaries.add_argument("--temperature", type=float, default=0.5)
    grouped_boundaries.add_argument("--evaluation-interval", type=int, default=32)
    grouped_boundaries.add_argument(
        "--maximum-mean-relative-l2", type=float, default=0.15
    )
    grouped_boundaries.add_argument("--max-train-records", type=int, default=4096)
    grouped_boundaries.add_argument("--max-validation-records", type=int, default=2048)
    grouped_boundaries.add_argument("--identity-pairing", action="store_true")
    grouped_boundaries.add_argument("--device", default="cpu")

    shared_expert_boundaries = commands.add_parser(
        "train-shared-expert-boundaries",
        help="fit an exact-upcycled shared-plus-expert MLP on teacher boundaries",
    )
    shared_expert_boundaries.add_argument("--model", required=True)
    shared_expert_boundaries.add_argument("--training-traces", required=True)
    shared_expert_boundaries.add_argument("--validation-traces", required=True)
    shared_expert_boundaries.add_argument("--out", required=True, type=Path)
    shared_expert_boundaries.add_argument(
        "--layers", nargs="+", type=int, required=True
    )
    shared_expert_boundaries.add_argument("--shared-records", type=int, default=128)
    shared_expert_boundaries.add_argument("--experts", type=int, default=96)
    shared_expert_boundaries.add_argument("--active-experts", type=int, default=32)
    shared_expert_boundaries.add_argument(
        "--start-active-experts", type=int, default=48
    )
    shared_expert_boundaries.add_argument("--grouping-iterations", type=int, default=12)
    shared_expert_boundaries.add_argument(
        "--router-regularization", type=float, default=3000.0
    )
    shared_expert_boundaries.add_argument("--router-warmup-steps", type=int, default=32)
    shared_expert_boundaries.add_argument("--anneal-steps", type=int, default=64)
    shared_expert_boundaries.add_argument("--settle-steps", type=int, default=128)
    shared_expert_boundaries.add_argument("--batch-size", type=int, default=128)
    shared_expert_boundaries.add_argument("--oracle-batch-size", type=int, default=64)
    shared_expert_boundaries.add_argument("--learning-rate", type=float, default=1e-4)
    shared_expert_boundaries.add_argument(
        "--router-learning-rate", type=float, default=1e-3
    )
    shared_expert_boundaries.add_argument("--route-weight", type=float, default=0.1)
    shared_expert_boundaries.add_argument("--cosine-weight", type=float, default=0.1)
    shared_expert_boundaries.add_argument(
        "--dense-anchor-weight", type=float, default=0.1
    )
    shared_expert_boundaries.add_argument(
        "--start-temperature", type=float, default=1.0
    )
    shared_expert_boundaries.add_argument("--temperature", type=float, default=0.5)
    shared_expert_boundaries.add_argument("--evaluation-interval", type=int, default=32)
    shared_expert_boundaries.add_argument(
        "--maximum-mean-relative-l2", type=float, default=0.15
    )
    shared_expert_boundaries.add_argument("--max-train-records", type=int, default=4096)
    shared_expert_boundaries.add_argument(
        "--max-validation-records", type=int, default=2048
    )
    shared_expert_boundaries.add_argument("--device", default="cpu")

    activation_aware_aq = commands.add_parser(
        "train-activation-aware-aq",
        help="fit a packed 2x7 additive MLP encoding on cached teacher boundaries",
    )
    activation_aware_aq.add_argument("--model", required=True)
    activation_aware_aq.add_argument("--training-traces", required=True)
    activation_aware_aq.add_argument("--validation-traces", required=True)
    activation_aware_aq.add_argument("--out", required=True, type=Path)
    activation_aware_aq.add_argument("--layers", nargs="+", type=int, required=True)
    activation_aware_aq.add_argument("--fit-iterations", type=int, default=12)
    activation_aware_aq.add_argument("--fit-sample-limit", type=int, default=65_536)
    activation_aware_aq.add_argument("--steps", type=int, default=128)
    activation_aware_aq.add_argument("--batch-size", type=int, default=64)
    activation_aware_aq.add_argument("--learning-rate", type=float, default=2e-3)
    activation_aware_aq.add_argument("--cosine-loss-weight", type=float, default=0.1)
    activation_aware_aq.add_argument(
        "--projection-loss-weight", type=float, default=0.01
    )
    activation_aware_aq.add_argument("--checkpoint-interval", type=int)
    activation_aware_aq.add_argument(
        "--maximum-mean-relative-l2", type=float, default=0.10
    )
    activation_aware_aq.add_argument("--max-train-records", type=int, default=4096)
    activation_aware_aq.add_argument("--max-validation-records", type=int, default=2048)
    activation_aware_aq.add_argument("--seed", type=int, default=0)
    activation_aware_aq.add_argument("--device", default="cpu")

    projection_aq = commands.add_parser(
        "train-projection-aq",
        help="fit projection-local packed AQ weights with discrete P/V refinement",
    )
    projection_aq.add_argument("--model", required=True)
    projection_aq.add_argument("--training-traces", required=True)
    projection_aq.add_argument("--validation-traces", required=True)
    projection_aq.add_argument("--out", required=True, type=Path)
    projection_aq.add_argument("--layers", nargs="+", type=int, required=True)
    projection_aq.add_argument("--p-steps-per-cycle", type=int, default=64)
    projection_aq.add_argument("--v-cycles", type=int, default=2)
    projection_aq.add_argument("--batch-size", type=int, default=128)
    projection_aq.add_argument("--learning-rate", type=float, default=2e-3)
    projection_aq.add_argument("--checkpoint-interval", type=int, default=16)
    projection_aq.add_argument("--fit-iterations", type=int, default=12)
    projection_aq.add_argument("--fit-sample-limit", type=int, default=65_536)
    projection_aq.add_argument("--v-max-records", type=int, default=1024)
    projection_aq.add_argument("--v-change-fraction", type=float, default=0.01)
    projection_aq.add_argument("--selection-records", type=int, default=512)
    projection_aq.add_argument("--maximum-mean-relative-l2", type=float, default=0.08)
    projection_aq.add_argument("--maximum-p95-relative-l2", type=float, default=0.18)
    projection_aq.add_argument("--minimum-mean-cosine", type=float, default=0.99)
    projection_aq.add_argument("--max-train-records", type=int, default=4096)
    projection_aq.add_argument("--max-validation-records", type=int, default=2048)
    projection_aq.add_argument("--seed", type=int, default=0)
    projection_aq.add_argument("--device", default="cpu")

    structured_shadow = commands.add_parser(
        "evaluate-structured-experts",
        help="screen a contiguous block-routed SwiGLU on held-out teacher traces",
    )
    structured_shadow.add_argument("--model", required=True)
    structured_shadow.add_argument("--calibration-traces", required=True)
    structured_shadow.add_argument("--validation-traces", required=True)
    structured_shadow.add_argument("--out", required=True, type=Path)
    structured_shadow.add_argument("--experts", type=int, default=24)
    structured_shadow.add_argument("--active-experts", type=int, default=8)
    structured_shadow.add_argument("--regularization", type=float, default=1000.0)
    structured_shadow.add_argument("--grouping-iterations", type=int, default=12)
    structured_shadow.add_argument("--calibration-records", type=int, default=256)
    structured_shadow.add_argument("--validation-records", type=int, default=256)

    native_gate_shadow = commands.add_parser(
        "evaluate-native-gate-channels",
        help="screen input-sparse native-gate channel routing on teacher traces",
    )
    native_gate_shadow.add_argument("--model", required=True)
    native_gate_shadow.add_argument("--validation-traces", required=True)
    native_gate_shadow.add_argument("--out", required=True, type=Path)
    native_gate_shadow.add_argument(
        "--input-fractions", nargs="+", type=float, default=(0.625, 1.0)
    )
    native_gate_shadow.add_argument("--top-k", type=int, default=512)
    native_gate_shadow.add_argument("--validation-records", type=int, default=128)

    native_gate_residual = commands.add_parser(
        "evaluate-native-gate-residual",
        help="screen a low-rank correction to native-gate channel utility",
    )
    native_gate_residual.add_argument("--model", required=True)
    native_gate_residual.add_argument("--calibration-traces", required=True)
    native_gate_residual.add_argument("--validation-traces", required=True)
    native_gate_residual.add_argument("--out", required=True, type=Path)
    native_gate_residual.add_argument("--ranks", nargs="+", type=int, default=(8, 16))
    native_gate_residual.add_argument(
        "--blends", nargs="+", type=float, default=(0.25, 0.5, 1.0)
    )
    native_gate_residual.add_argument("--input-fraction", type=float, default=0.625)
    native_gate_residual.add_argument("--top-k", type=int, default=512)
    native_gate_residual.add_argument("--regularization", type=float, default=1000.0)
    native_gate_residual.add_argument("--calibration-records", type=int, default=128)
    native_gate_residual.add_argument("--validation-records", type=int, default=128)
    native_gate_residual.add_argument("--active-record-limit", type=int, default=512)

    on_policy_residual = commands.add_parser(
        "recalibrate-native-gate-residual",
        help="refit native-gate utility residuals on hard sparse-student states",
    )
    on_policy_residual.add_argument("--model", required=True)
    on_policy_residual.add_argument("--calibration-dataset", required=True, type=Path)
    on_policy_residual.add_argument("--initial-residual", required=True, type=Path)
    on_policy_residual.add_argument("--out", required=True, type=Path)
    on_policy_residual.add_argument("--rank", type=int, default=16)
    on_policy_residual.add_argument(
        "--blends", nargs="+", type=float, default=(0.5, 0.65, 0.8, 1.0)
    )
    on_policy_residual.add_argument("--input-fraction", type=float, default=0.625)
    on_policy_residual.add_argument("--top-k", type=int, default=512)
    on_policy_residual.add_argument("--regularization", type=float, default=4000.0)
    on_policy_residual.add_argument("--fit-fraction", type=float, default=0.75)
    on_policy_residual.add_argument("--fit-states", type=int, default=512)
    on_policy_residual.add_argument("--validation-states", type=int, default=128)
    on_policy_residual.add_argument("--device", default="cpu")

    native_gate_train = commands.add_parser(
        "train-native-gate-traces",
        help="pretrain selected native-gate sparse MLP layers on cached boundaries",
    )
    native_gate_train.add_argument("--model", required=True)
    native_gate_train.add_argument("--calibration-traces", required=True)
    native_gate_train.add_argument("--validation-traces", required=True)
    native_gate_train.add_argument("--out", required=True, type=Path)
    native_gate_train.add_argument("--layers", nargs="+", type=int, required=True)
    native_gate_train.add_argument("--top-k", type=int, default=512)
    native_gate_train.add_argument("--input-fraction", type=float, default=0.625)
    native_gate_train.add_argument("--steps", type=int, default=16)
    native_gate_train.add_argument("--batch-size", type=int, default=8)
    native_gate_train.add_argument("--learning-rate", type=float, default=1e-4)
    native_gate_train.add_argument("--dense-shadow-weight", type=float, default=0.25)
    native_gate_train.add_argument("--utility-weight", type=float, default=0.25)
    native_gate_train.add_argument("--temperature", type=float, default=1.0)
    native_gate_train.add_argument("--calibration-records", type=int, default=128)
    native_gate_train.add_argument("--validation-records", type=int, default=128)
    native_gate_train.add_argument("--device", default="cpu")

    native_gate_e2e = commands.add_parser(
        "train-native-gate-e2e",
        help="progressively distill all MLPs through the hard native-gate path",
    )
    native_gate_e2e.add_argument("--model", required=True)
    native_gate_e2e.add_argument("--training-dataset", required=True, type=Path)
    native_gate_e2e.add_argument("--validation-dataset", required=True, type=Path)
    native_gate_e2e.add_argument("--out", required=True, type=Path)
    native_gate_e2e.add_argument("--target-top-k", type=int, default=512)
    native_gate_e2e.add_argument("--target-input-fraction", type=float, default=0.625)
    native_gate_e2e.add_argument("--steps", type=int, default=2)
    native_gate_e2e.add_argument("--warmup-steps", type=int, default=0)
    native_gate_e2e.add_argument("--anneal-steps", type=int, default=1)
    native_gate_e2e.add_argument("--batch-size", type=int, default=1)
    native_gate_e2e.add_argument("--learning-rate", type=float, default=1e-5)
    native_gate_e2e.add_argument("--local-weight", type=float, default=1.0)
    native_gate_e2e.add_argument("--dense-shadow-weight", type=float, default=0.25)
    native_gate_e2e.add_argument("--hidden-weight", type=float, default=0.25)
    native_gate_e2e.add_argument("--logit-weight", type=float, default=0.25)
    native_gate_e2e.add_argument("--utility-weight", type=float, default=0.1)
    native_gate_e2e.add_argument("--temperature", type=float, default=1.0)
    native_gate_e2e.add_argument("--max-train-records", type=int)
    native_gate_e2e.add_argument("--max-validation-records", type=int)
    native_gate_e2e.add_argument("--device", default="cpu")
    native_gate_e2e.add_argument("--no-artifact", action="store_true")
    native_gate_e2e.add_argument("--checkpoint-every", type=int, default=0)
    native_gate_e2e.add_argument("--resume", action="store_true")
    native_gate_e2e.add_argument("--utility-residual", type=Path)

    fully_sparse_e2e = commands.add_parser(
        "train-fully-sparse-student",
        help=(
            "distill all MLPs through exact hard Q-Sparse activation paths; "
            "CUDA is training-only and the saved artifact is CPU validated"
        ),
    )
    fully_sparse_e2e.add_argument("--model", required=True)
    fully_sparse_e2e.add_argument("--training-dataset", required=True, type=Path)
    fully_sparse_e2e.add_argument("--validation-dataset", required=True, type=Path)
    fully_sparse_e2e.add_argument("--out", required=True, type=Path)
    fully_sparse_e2e.add_argument("--input-fraction", type=float, default=0.49)
    fully_sparse_e2e.add_argument("--intermediate-fraction", type=float, default=0.34)
    fully_sparse_e2e.add_argument(
        "--input-counts",
        nargs="+",
        type=int,
        help="one hard input-coordinate count per transformer layer",
    )
    fully_sparse_e2e.add_argument(
        "--intermediate-counts",
        nargs="+",
        type=int,
        help="one hard intermediate-activation count per transformer layer",
    )
    fully_sparse_e2e.add_argument("--steps", type=int, default=8)
    fully_sparse_e2e.add_argument("--warmup-steps", type=int, default=1)
    fully_sparse_e2e.add_argument("--anneal-steps", type=int, default=6)
    fully_sparse_e2e.add_argument("--batch-size", type=int, default=1)
    fully_sparse_e2e.add_argument("--learning-rate", type=float, default=1e-5)
    fully_sparse_e2e.add_argument("--backbone-learning-rate", type=float, default=3e-6)
    fully_sparse_e2e.add_argument("--local-weight", type=float, default=0.25)
    fully_sparse_e2e.add_argument("--hidden-weight", type=float, default=0.5)
    fully_sparse_e2e.add_argument("--logit-weight", type=float, default=1.0)
    fully_sparse_e2e.add_argument("--label-weight", type=float, default=0.0)
    fully_sparse_e2e.add_argument("--max-train-records", type=int)
    fully_sparse_e2e.add_argument("--max-validation-records", type=int)
    fully_sparse_e2e.add_argument("--device", default="cuda")
    fully_sparse_e2e.add_argument("--checkpoint-every", type=int, default=0)
    fully_sparse_e2e.add_argument("--resume", action="store_true")
    fully_sparse_e2e.add_argument("--coadapt-backbone", action="store_true")
    fully_sparse_e2e.add_argument("--coadapt-embeddings-and-head", action="store_true")
    fully_sparse_e2e.add_argument("--residual-rank", type=int, default=0)

    width_train = commands.add_parser(
        "train-width-pruned-student",
        help="progressively distill contiguous fixed-width SwiGLU layers",
    )
    width_train.add_argument("--model", required=True)
    width_train.add_argument("--training-dataset", required=True, type=Path)
    width_train.add_argument("--validation-dataset", required=True, type=Path)
    width_train.add_argument("--calibration-traces")
    width_train.add_argument("--out", required=True, type=Path)
    width_train.add_argument("--target-width", type=int, default=672)
    width_train.add_argument(
        "--target-widths",
        nargs="+",
        type=int,
        help=(
            "one compact width per transformer layer; overrides the scalar "
            "--target-width"
        ),
    )
    width_train.add_argument("--steps", type=int, default=8)
    width_train.add_argument("--replacement-steps", type=int, default=0)
    width_train.add_argument("--batch-size", type=int, default=1)
    width_train.add_argument("--learning-rate", type=float, default=1e-5)
    width_train.add_argument("--local-weight", type=float, default=1.0)
    width_train.add_argument("--hidden-weight", type=float, default=0.25)
    width_train.add_argument("--logit-weight", type=float, default=0.25)
    width_train.add_argument("--initialization-records", type=int, default=512)
    width_train.add_argument("--local-warmup-steps", type=int, default=0)
    width_train.add_argument("--local-batch-size", type=int, default=32)
    width_train.add_argument("--local-learning-rate", type=float, default=3e-4)
    width_train.add_argument("--max-train-records", type=int)
    width_train.add_argument("--max-validation-records", type=int)
    width_train.add_argument("--device", default="cpu")
    width_train.add_argument("--no-artifact", action="store_true")
    width_train.add_argument(
        "--strict-q4-deployment",
        action="store_true",
        help=(
            "serialize/reload cache-aligned Q4 MLPs and validate causal quality "
            "on their decoded weights"
        ),
    )
    width_train.add_argument(
        "--fake-q4-training",
        action="store_true",
        help=(
            "train compact forwards through deployment-matched signed-Q4 STE; "
            "requires --strict-q4-deployment"
        ),
    )
    width_train.add_argument("--checkpoint-every", type=int, default=0)
    width_train.add_argument("--resume", action="store_true")
    width_train.add_argument("--initial-checkpoint", type=Path)
    width_train.add_argument(
        "--coadapt-backbone",
        action="store_true",
        help="jointly update already-resident attention and normalization weights",
    )
    width_train.add_argument(
        "--coadapt-embeddings-and-head",
        action="store_true",
        help="also update the already-resident input embedding and output head",
    )
    width_train.add_argument("--backbone-learning-rate", type=float, default=3e-5)

    ternary_train = commands.add_parser(
        "train-budget-native-ternary",
        help=(
            "continually distill full-width grouped-ternary MLPs through "
            "their serialized deployment representation"
        ),
    )
    ternary_train.add_argument("--model", required=True)
    ternary_train.add_argument("--training-dataset", required=True, type=Path)
    ternary_train.add_argument("--validation-dataset", required=True, type=Path)
    ternary_train.add_argument("--out", required=True, type=Path)
    ternary_train.add_argument("--group-size", type=int, default=128)
    ternary_train.add_argument("--steps", type=int, default=8)
    ternary_train.add_argument("--dense-warmup-steps", type=int, default=0)
    ternary_train.add_argument("--anneal-steps", type=int, default=4)
    ternary_train.add_argument(
        "--transition-mode",
        choices=("deepest_first", "global"),
        default="deepest_first",
        help=(
            "stagger quantization from the deepest MLP upward, or transition "
            "all MLPs together"
        ),
    )
    ternary_train.add_argument("--batch-size", type=int, default=1)
    ternary_train.add_argument("--learning-rate", type=float, default=1e-5)
    ternary_train.add_argument("--backbone-learning-rate", type=float, default=3e-5)
    ternary_train.add_argument("--local-weight", type=float, default=0.1)
    ternary_train.add_argument("--hidden-weight", type=float, default=0.5)
    ternary_train.add_argument("--final-hidden-weight", type=float, default=1.0)
    ternary_train.add_argument("--final-cka-weight", type=float, default=0.0)
    ternary_train.add_argument("--logit-weight", type=float, default=1.0)
    ternary_train.add_argument("--teacher-top1-weight", type=float, default=0.0)
    ternary_train.add_argument("--label-weight", type=float, default=0.1)
    ternary_train.add_argument("--temperature", type=float, default=1.0)
    ternary_train.add_argument("--confidence-weight", type=float, default=0.5)
    ternary_train.add_argument("--coadapt-backbone", action="store_true")
    ternary_train.add_argument(
        "--coadapt-embeddings-and-head",
        action="store_true",
    )
    ternary_train.add_argument("--backbone-start-step", type=int)
    ternary_train.add_argument("--max-train-records", type=int)
    ternary_train.add_argument("--training-record-offset", type=int, default=0)
    ternary_train.add_argument("--max-validation-records", type=int)
    ternary_train.add_argument("--device", default="cpu")
    ternary_train.add_argument("--no-artifact", action="store_true")
    ternary_train.add_argument("--checkpoint-every", type=int, default=0)
    ternary_train.add_argument("--resume", action="store_true")
    ternary_train.add_argument("--initial-checkpoint", type=Path)

    width_ceiling = commands.add_parser(
        "evaluate-width-local-ceiling",
        help="fit compact MLPs on cached teacher boundaries and screen their local ceiling",
    )
    width_ceiling.add_argument("--model", required=True)
    width_ceiling.add_argument("--training-traces", required=True)
    width_ceiling.add_argument("--validation-traces", required=True)
    width_ceiling.add_argument("--initial-checkpoint", required=True, type=Path)
    width_ceiling.add_argument("--out", required=True, type=Path)
    width_ceiling.add_argument("--layers", nargs="+", type=int, required=True)
    width_ceiling.add_argument("--target-width", type=int, default=672)
    width_ceiling.add_argument("--steps", type=int, default=64)
    width_ceiling.add_argument("--batch-size", type=int, default=32)
    width_ceiling.add_argument("--learning-rate", type=float, default=3e-4)
    width_ceiling.add_argument("--max-train-records", type=int, default=4096)
    width_ceiling.add_argument("--max-validation-records", type=int, default=2048)
    width_ceiling.add_argument("--maximum-mean-relative-l2", type=float, default=0.15)
    width_ceiling.add_argument(
        "--minimum-improvement-fraction", type=float, default=0.10
    )
    width_ceiling.add_argument("--device", default="cpu")

    gated_background = commands.add_parser(
        "evaluate-gated-background-ceiling",
        help="fit a small SwiGLU to the exact top-K semantic residual",
    )
    gated_background.add_argument("--model", required=True)
    gated_background.add_argument("--training-traces", required=True)
    gated_background.add_argument("--validation-traces", required=True)
    gated_background.add_argument("--out", required=True, type=Path)
    gated_background.add_argument("--layers", nargs="+", type=int, required=True)
    gated_background.add_argument("--top-k", type=int, default=512)
    gated_background.add_argument("--background-width", type=int, default=128)
    gated_background.add_argument("--router-rank", type=int, default=16)
    gated_background.add_argument("--steps", type=int, default=1024)
    gated_background.add_argument("--batch-size", type=int, default=32)
    gated_background.add_argument("--learning-rate", type=float, default=3e-4)
    gated_background.add_argument("--max-train-records", type=int, default=4096)
    gated_background.add_argument("--max-validation-records", type=int, default=2048)
    gated_background.add_argument(
        "--maximum-mean-relative-l2", type=float, default=0.10
    )
    gated_background.add_argument("--device", default="cpu")
    gated_background.add_argument("--seed", type=int, default=97)

    width_residual = commands.add_parser(
        "evaluate-width-residual-sweep",
        help="fit ridge/SVD residuals around a fixed-width compact MLP",
    )
    width_residual.add_argument("--model", required=True)
    width_residual.add_argument("--training-traces", required=True)
    width_residual.add_argument("--validation-traces", required=True)
    width_residual.add_argument("--initial-checkpoint", required=True, type=Path)
    width_residual.add_argument("--out", required=True, type=Path)
    width_residual.add_argument("--layers", nargs="+", type=int, required=True)
    width_residual.add_argument("--compact-width", type=int, default=672)
    width_residual.add_argument("--ranks", nargs="+", type=int, default=(8, 16, 24, 28))
    width_residual.add_argument("--ridge-factor", type=float, default=0.5)
    width_residual.add_argument("--max-train-records", type=int, default=4096)
    width_residual.add_argument("--max-validation-records", type=int, default=2048)
    width_residual.add_argument("--maximum-mean-relative-l2", type=float, default=0.10)

    oracle_residual = commands.add_parser(
        "evaluate-oracle-residual-ceiling",
        help="fit affine residuals around an exact sparse semantic read",
    )
    oracle_residual.add_argument("--model", required=True)
    oracle_residual.add_argument("--training-traces", required=True)
    oracle_residual.add_argument("--validation-traces", required=True)
    oracle_residual.add_argument("--out", required=True, type=Path)
    oracle_residual.add_argument("--layers", nargs="+", type=int, required=True)
    oracle_residual.add_argument("--top-k", type=int, default=640)
    oracle_residual.add_argument(
        "--ranks", nargs="+", type=int, default=(16, 32, 48, 64, 75)
    )
    oracle_residual.add_argument("--ridge-factor", type=float, default=0.5)
    oracle_residual.add_argument("--max-train-records", type=int, default=4096)
    oracle_residual.add_argument("--max-validation-records", type=int, default=2048)
    oracle_residual.add_argument("--maximum-mean-relative-l2", type=float, default=0.10)

    intervention_gate = commands.add_parser(
        "gate-mlp-intervention",
        help="apply declared go/no-go criteria to an existing intervention report",
    )
    intervention_gate.add_argument("--report", required=True, type=Path, nargs="+")
    intervention_gate.add_argument("--out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "inspect":
        result = inspect_model(
            args.model, hash_weights=not args.no_weight_hash
        ).to_dict()
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(payload, encoding="utf-8")
        print(payload, end="")
    elif args.command == "audit-native-bitnet":
        result = audit_native_bitnet_source(
            args.model,
            revision=args.revision,
            cache_dir=args.cache_dir,
        ).to_dict()
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.out:
            atomic_json(args.out, result)
        print(payload, end="")
        return 0 if result.get("decision") == "proceed_to_exact_weight_repack" else 2
    elif args.command == "audit-olmoe":
        result = audit_olmoe_source(
            args.model,
            revision=args.revision,
            cache_dir=args.cache_dir,
            verify_remote_shapes=args.verify_remote_shapes,
        ).to_dict()
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.out:
            atomic_json(args.out, result)
        print(payload, end="")
        return (
            0
            if result.get("decision")
            in {
                "proceed_to_exact_weight_shape_audit",
                "proceed_to_router_trace",
            }
            else 2
        )
    elif args.command == "repack-olmoe-q7":
        artifact = repack_olmoe_q7_model(
            args.model, args.out, group_size=args.group_size
        )
        result = inspect_olmoe_q7_artifact(artifact)
        if args.report:
            atomic_json(args.report, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "inspect-olmoe-q7":
        print(
            json.dumps(
                inspect_olmoe_q7_artifact(args.artifact),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "evaluate-native-olmoe-q7":
        result = evaluate_olmoe_q7_native_systems(
            args.artifact,
            args.library,
            args.out,
            layer=args.layer,
            states=args.states,
            threads=args.threads,
            seed=args.seed,
            maximum_relative_l2=args.maximum_relative_l2,
            maximum_traffic_fraction=args.maximum_traffic_fraction,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["gate_passed"] else 2
    elif args.command == "repack-olmoe-non-mlp":
        result = repack_olmoe_non_mlp_weights(args.model, args.out)
        if args.report:
            atomic_json(args.report, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "run-native-olmoe-token":
        tokenizer = None
        token_ids = args.token_ids
        eos_token_ids: tuple[int, ...] = ()
        if args.prompt is not None:
            if args.tokenizer is None:
                parser.error("run-native-olmoe-token --prompt requires --tokenizer")
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "install engram-lm[conversion] for text tokenization"
                ) from exc
            tokenizer_path = (
                args.tokenizer / "tokenizer.json"
                if args.tokenizer.is_dir()
                else args.tokenizer
            )
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            token_ids = tokenizer.encode(args.prompt).ids
            eos = tokenizer.token_to_id("<|endoftext|>")
            eos_token_ids = () if eos is None else (int(eos),)
        if not token_ids:
            parser.error("native OLMoE input tokenization produced no tokens")
        with OLMoENativeTokenRuntime(
            args.config,
            args.non_mlp,
            args.q7_artifact,
            args.library,
            threads=args.threads,
        ) as runtime:
            generated = runtime.generate(
                token_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos_token_ids,
            )
            position = runtime.position
            metrics = (
                runtime.last_result.metrics if runtime.last_result is not None else {}
            )
        print(
            json.dumps(
                {
                    "generated_token_ids": generated,
                    "completion": (
                        tokenizer.decode(generated) if tokenizer is not None else None
                    ),
                    "metrics": metrics,
                    "position": position,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "compile-native-olmoe":
        result = compile_olmoe_native_package(
            args.model,
            args.q7_artifact,
            args.non_mlp,
            args.out,
            kernel_threads=args.threads,
        )
        if args.report:
            atomic_json(args.report, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "generate-native-olmoe-package":
        with OLMoENativePackageRuntime(
            args.package,
            manifest_sha256=args.manifest_sha256,
            library=args.library,
            threads=args.threads,
        ) as runtime:
            result = runtime.generate(
                args.prompt,
                max_new_tokens=args.max_new_tokens,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "capture-olmoe-teacher-generation":
        result = capture_olmoe_teacher_generation(
            model=args.model,
            prompts=args.prompts,
            out=args.out,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            threads=args.threads,
        )
        print(
            json.dumps(
                {
                    "reference": result,
                    "reference_sha256": sha256_file(args.out),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "evaluate-native-olmoe-generation":
        result = evaluate_native_olmoe_generation(
            package=args.package,
            manifest_sha256=args.manifest_sha256,
            library=args.library,
            prompts=args.prompts,
            teacher_reference=args.teacher_reference,
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            out=args.out,
            threads=args.threads,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["gate_passed"] else 2
    elif args.command == "capture-olmoe-teacher-causal":
        result = capture_olmoe_teacher_causal_reference(
            model=args.model,
            dataset=args.dataset,
            out=args.out,
            arrays_out=args.arrays_out,
            sequences=args.sequences,
            tokens_per_sequence=args.tokens_per_sequence,
            device=args.device,
            threads=args.threads,
            batch_size=args.batch_size,
            expert_workers=args.expert_workers,
            sequence_workers=args.sequence_workers,
        )
        print(
            json.dumps(
                {
                    "reference": result,
                    "reference_sha256": sha256_file(args.out),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "evaluate-native-olmoe-causal":
        result = evaluate_native_olmoe_causal(
            package=args.package,
            manifest_sha256=args.manifest_sha256,
            library=args.library,
            dataset=args.dataset,
            teacher_reference=args.teacher_reference,
            teacher_arrays=args.teacher_arrays,
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            out=args.out,
            threads=args.threads,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["gate_passed"] else 2
    elif args.command == "freeze-olmoe-sustained-protocol":
        result = freeze_olmoe_sustained_context_protocol(
            package=args.package,
            manifest_sha256=args.manifest_sha256,
            library=args.library,
            dataset=args.dataset,
            corpus_manifest=args.corpus_manifest,
            teacher_reference=args.teacher_reference,
            teacher_arrays=args.teacher_arrays,
            out=args.out,
            threads=args.threads,
        )
        print(
            json.dumps(
                {
                    "protocol": result,
                    "protocol_sha256": sha256_file(args.out),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "evaluate-native-olmoe-sustained":
        result = evaluate_native_olmoe_sustained_context(
            package=args.package,
            manifest_sha256=args.manifest_sha256,
            library=args.library,
            dataset=args.dataset,
            corpus_manifest=args.corpus_manifest,
            teacher_reference=args.teacher_reference,
            teacher_arrays=args.teacher_arrays,
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            out=args.out,
            threads=args.threads,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["gate_passed"] else 2
    elif args.command == "repack-native-bitnet":
        result = repack_native_bitnet_model(
            args.model,
            args.out,
            revision=args.revision,
            cache_dir=args.cache_dir,
            report_path=args.report,
            verify_official_weight_hash=not args.skip_official_weight_hash,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "evaluate-native-bitnet-parity":
        result = evaluate_native_bitnet_parity(
            args.model,
            args.artifact,
            out=args.out,
            revision=args.revision,
            cache_dir=args.cache_dir,
            local_layers=args.local_layers,
            local_states=args.local_states,
            input_ids=args.input_ids,
            run_causal_substitution=not args.no_causal_substitution,
            expected_artifact_sha256=args.artifact_sha256,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("smoke_gate", {}).get("passed") else 2
    elif args.command == "evaluate-native-bitnet-kernel":
        result = evaluate_native_bitnet_kernel_confirmation(
            args.model,
            args.artifact,
            args.dataset,
            out=args.out,
            artifact_sha256=args.artifact_sha256,
            revision=args.revision,
            cache_dir=args.cache_dir,
            library=args.library,
            threads=args.threads,
            sequence_count=args.sequence_count,
            prediction_positions=args.prediction_positions,
            record_offset=args.record_offset,
            parity_layers=args.parity_layers,
            parity_states=args.parity_states,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("gate_passed") else 2
    elif args.command == "analyze-native-bitnet-oracle":
        result = evaluate_native_bitnet_oracle(
            args.model,
            args.dataset,
            out=args.out,
            layers=args.layers,
            samples=args.samples,
            max_tokens=args.max_tokens,
            record_offset=args.record_offset,
            fractions=args.fractions,
            library=args.library,
            threads=args.threads,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["progression_screen"]["passed"] else 2
    elif args.command == "evaluate-native-bitnet-oracle-causal":
        result = evaluate_native_bitnet_oracle_causal(
            args.model,
            args.dataset,
            out=args.out,
            fraction=args.fraction,
            layer_fractions=args.layer_fractions,
            sequence_count=args.sequence_count,
            predictions_per_sequence=args.predictions_per_sequence,
            record_offset=args.record_offset,
            library=args.library,
            threads=args.threads,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["quality_passed"] else 2
    elif args.command == "sweep-native-bitnet-oracle-layers":
        result = evaluate_native_bitnet_oracle_layer_sweep(
            args.model,
            args.dataset,
            out=args.out,
            fractions=args.fractions,
            mean_budget=args.mean_budget,
            sequence_count=args.sequence_count,
            tokens_per_sequence=args.tokens_per_sequence,
            record_offset=args.record_offset,
            library=args.library,
            threads=args.threads,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "evaluate-native-bitnet-router":
        result = evaluate_native_bitnet_low_rank_router(
            args.model,
            args.training_trace,
            args.validation_trace,
            out=args.out,
            layers=args.layers,
            top_ks=args.top_ks,
            rank=args.rank,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["eligible_for_all_layer_training"] else 2
    elif args.command == "evaluate-native-bitnet-dip-router":
        result = evaluate_native_bitnet_dip_router(
            args.model,
            args.validation_trace,
            out=args.out,
            layer=args.layer,
            top_k=args.top_k,
            input_fractions=args.input_fractions,
            candidate_multipliers=args.candidate_multipliers,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["best_arm"]["meets_joint_screen"] else 2
    elif args.command == "sweep-native-bitnet-dip-all-layers":
        result = evaluate_native_bitnet_dip_all_layers(
            args.model,
            args.validation_trace,
            args.oracle_schedule,
            out=args.out,
            input_fraction=args.input_fraction,
            candidate_multipliers=args.candidate_multipliers,
            maximum_traffic_fraction=args.maximum_traffic_fraction,
            recall_gate=args.recall_gate,
            tail_recall_preference=args.tail_recall_preference,
            worst_row_recall_preference=args.worst_row_recall_preference,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["progression_screen"]["passed"] else 2
    elif args.command == "sweep-native-bitnet-dip-adaptive-k":
        result = evaluate_native_bitnet_dip_adaptive_k(
            args.model,
            args.validation_trace,
            args.router_policy,
            out=args.out,
            energy_targets=args.energy_targets,
            minimum_fraction=args.minimum_fraction,
            maximum_fraction=args.maximum_fraction,
            mean_budget_fraction=args.mean_budget_fraction,
            device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return (
            0
            if result["selected_highest_energy_target_within_mean_budget"] is not None
            else 2
        )
    elif args.command == "optimize-native-bitnet-dip-joint-policy":
        result = evaluate_native_bitnet_dip_joint_policy(
            args.model,
            args.validation_trace,
            out=args.out,
            candidate_counts=args.candidate_counts,
            input_fraction=args.input_fraction,
            minimum_fraction=args.minimum_fraction,
            mean_budget_fraction=args.mean_budget_fraction,
            maximum_traffic_fraction=args.maximum_traffic_fraction,
            device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["progression_screen"]["passed"] else 2
    elif args.command == "create-fixture":
        print(create_tiny_fixture(args.out, seed=args.seed))
    elif args.command == "create-olmoe-fixture":
        print(create_tiny_olmoe_fixture(args.out, seed=args.seed))
    elif args.command == "trace-olmoe-fixture":
        capture_olmoe_fixture_router_traces(
            args.model,
            args.out,
            samples=args.samples,
            layers=args.layers,
            seed=args.seed,
        )
        print(args.out)
    elif args.command == "trace-olmoe-router":
        capture_olmoe_router_traces(
            args.model,
            args.dataset,
            args.out,
            samples=args.samples,
            layers=args.layers,
            tokens_per_sequence=args.tokens_per_sequence,
            seed=args.seed,
        )
        print(args.out)
    elif args.command == "evaluate-olmoe-q4-local":
        result = evaluate_olmoe_q4_local(
            args.model,
            args.trace,
            args.out,
            layer=args.layer,
            group_size=args.group_size,
            maximum_mean_relative_l2=args.maximum_mean_relative_l2,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["screen"]["passed"] else 2
    elif args.command in {
        "evaluate-olmoe-quantized-causal",
        "evaluate-olmoe-q4-causal",
    }:
        result = evaluate_olmoe_q4_causal(
            args.model,
            args.dataset,
            args.out,
            samples=args.samples,
            max_tokens=args.max_tokens,
            bits=args.bits,
            group_size=args.group_size,
            threads=args.threads,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["screen"]["quality_passed"] else 2
    elif args.command == "trace":
        if args.dry_run:
            if args.dataset is None:
                raise ValueError("--dataset is required for a trace dry run")
            plan = plan_teacher_trace_capture(
                args.model,
                args.dataset,
                samples=args.samples,
                include_attention=not args.mlp_only,
                tokens_per_sequence=args.tokens_per_sequence,
                layers=args.layers,
            )
            plan["planned_trace_output"] = str(args.out.resolve())
            plan["planned_split"] = args.split
            plan["planned_seed"] = args.seed
            if args.plan_out is not None:
                atomic_json(args.plan_out, plan)
            print(json.dumps(plan, indent=2, sort_keys=True))
        else:
            if args.plan_out is not None:
                raise ValueError("--plan-out requires --dry-run")
            capture_teacher_traces(
                args.model,
                args.out,
                dataset=args.dataset,
                split=args.split,
                seed=args.seed,
                samples=args.samples,
                include_attention=not args.mlp_only,
                tokens_per_sequence=args.tokens_per_sequence,
                layers=args.layers,
            )
            print(args.out)
    elif args.command == "trace-native-bitnet-controller":
        result = capture_native_bitnet_controller_traces(
            args.model,
            args.dataset,
            args.out,
            split=args.split,
            samples=args.samples,
            max_tokens=args.max_tokens,
            causal_top_k=args.causal_top_k,
            batch_size=args.batch_size,
            record_offset=args.record_offset,
            seed=args.seed,
            library=args.library,
            threads=args.threads,
            resume=args.resume,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "distill-controller":
        result = distill_factorized_controller(
            args.trace,
            args.out,
            validation_trace=args.validation_trace,
            initial_controller=args.initial_controller,
            device=args.device,
            rank=args.rank,
            adapter_rank=args.adapter_rank,
            input_adapter_rank=args.input_adapter_rank,
            operator_residual=args.operator_residual,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            teacher_forcing_schedule=args.teacher_forcing,
            causal_lm_head=args.causal_lm_head,
            causal_norm_weight=args.causal_norm_weight,
            causal_weight=args.causal_weight,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "analyze-mlp":
        report = analyze_magnitude_oracle(
            args.model, args.traces, max_records=args.max_records
        )
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
    elif args.command == "build-dip-package":
        print(
            build_serialized_dip_package(
                args.model,
                args.out,
                layers=args.layers,
                dual_layout=args.dual_layout_experimental,
            )
        )
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
    elif args.command == "evaluate-native-bitnet-attention":
        result = evaluate_native_bitnet_attention_substitution(
            args.model,
            args.dataset,
            out=args.out,
            library=args.library,
            threads=args.threads,
            native_projections=args.native_projections,
            sequence_count=args.sequence_count,
            prediction_positions=args.prediction_positions,
            record_offset=args.record_offset,
            modes=args.modes,
            layers=args.layers,
            local_window=args.local_window,
            recurrent_decay=args.recurrent_decay,
            retrieval_top_k=args.retrieval_top_k,
            older_weight=args.older_weight,
            retrieval_candidates=args.retrieval_candidates,
            lsh_tables=args.lsh_tables,
            lsh_bits=args.lsh_bits,
            lsh_radius=args.lsh_radius,
            lsh_seed=args.lsh_seed,
            page_size=args.page_size,
            page_bound=args.page_bound,
            sink_tokens=args.sink_tokens,
            native_attention_library=args.attention_library,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "evaluate-native-bitnet-controller":
        result = evaluate_native_bitnet_controller_substitution(
            args.model,
            args.dataset,
            args.controller,
            out=args.out,
            library=args.library,
            attention_library=args.attention_library,
            threads=args.threads,
            native_projections=not args.no_native_projections,
            sequence_count=args.sequence_count,
            prediction_positions=args.prediction_positions,
            record_offset=args.record_offset,
            local_window=args.local_window,
            retrieval_candidates=args.retrieval_candidates,
            retrieval_top_k=args.retrieval_top_k,
            sink_tokens=args.sink_tokens,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "evaluate-controller-only":
        result = evaluate_controller_only_trace(
            args.trace,
            args.controller,
            out=args.out,
            allow_correction=args.allow_correction,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "fit-operator-provider":
        result = fit_operator_stream_provider(
            args.trace,
            args.out,
            output_rank=args.output_rank,
            ridge=args.ridge,
            target=args.target,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "distill-operator-provider":
        result = joint_distill_operator_provider(
            args.provider,
            args.controller,
            args.trace,
            args.out,
            validation_trace=args.validation_trace,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "distill-state-space-provider":
        result = distill_state_space_operator_provider(
            args.provider,
            args.controller,
            args.trace,
            args.out,
            validation_trace=args.validation_trace,
            steps=args.steps,
            batch_size=args.batch_size,
            memory_dim=args.memory_dim,
            projection_width=args.projection_width,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "distill-state-space-residual-provider":
        result = distill_state_space_residual_provider(
            args.provider,
            args.controller,
            args.trace,
            args.out,
            validation_trace=args.validation_trace,
            steps=args.steps,
            batch_size=args.batch_size,
            memory_dim=args.memory_dim,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "adapt-controller-correction":
        result = adapt_controller_correction_for_provider(
            args.provider,
            args.controller,
            args.trace,
            args.out,
            validation_trace=args.validation_trace,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "dagger-refit-operator-provider":
        result = dagger_refit_operator_provider(
            args.provider,
            args.controller,
            args.trace,
            args.out,
            validation_trace=args.validation_trace,
            iterations=args.iterations,
            ridge=args.ridge,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "distill-nonlinear-residual-provider":
        result = distill_nonlinear_residual_provider(
            args.provider,
            args.controller,
            args.trace,
            args.out,
            validation_trace=args.validation_trace,
            steps=args.steps,
            teacher_forcing_steps=args.teacher_forcing_steps,
            teacher_forcing_decay_steps=args.teacher_forcing_decay_steps,
            batch_size=args.batch_size,
            hidden_width=args.hidden_width,
            stage_width=args.stage_width,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "evaluate-controller-provider":
        result = evaluate_controller_provider_trace(
            args.trace,
            args.provider,
            args.controller,
            out=args.out,
            allow_correction=args.allow_correction,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "evaluate-controller-sequence":
        result = evaluate_controller_sequence_replay(
            args.trace,
            args.provider,
            args.controller,
            out=args.out,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "benchmark-native-attention":
        result = benchmark_native_streaming_attention(
            out=args.out,
            library=args.library,
            lengths=args.lengths,
            local_window=args.local_window,
            older_candidates=args.candidates,
            older_top_k=args.top_k,
            sink_tokens=args.sink_tokens,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "benchmark-native-bitnet-generation":
        result = benchmark_native_bitnet_generation(
            package=args.model,
            out=args.out,
            prompt=args.prompt,
            lengths=args.lengths,
            max_new_tokens=args.max_tokens,
            mlp_library=args.mlp_library,
            attention_library=args.attention_library,
            threads=args.threads,
            native_projections=args.native_projections,
            local_window=args.local_window,
            older_candidates=args.candidates,
            older_top_k=args.top_k,
            sink_tokens=args.sink_tokens,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "evaluate-native-bitnet-generation":
        result = evaluate_native_bitnet_generation(
            package=args.model,
            prompts=args.prompts,
            out=args.out,
            max_new_tokens=args.max_tokens,
            mlp_library=args.mlp_library,
            attention_library=args.attention_library,
            threads=args.threads,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "evaluate-native-bitnet-dip-token-generation":
        result = evaluate_native_bitnet_dip_token_generation(
            package=args.model,
            executable=args.executable,
            prompts=args.prompts,
            reference_report=args.reference,
            out=args.out,
            package_manifest_sha256=args.package_manifest_sha256,
            executable_sha256=args.executable_sha256,
            max_new_tokens=args.max_tokens,
            threads=args.threads,
            verify_reset=not args.no_verify_reset,
            timeout_seconds=args.timeout,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "evaluate-native-bitnet-dip-attention":
        result = evaluate_native_bitnet_dip_attention_confirmation(
            package=args.model,
            library=args.library,
            out=args.out,
            lengths=args.lengths,
            prompt=args.prompt,
            threads=args.threads,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "evaluate-native-bitnet-controller-generation":
        result = evaluate_native_bitnet_controller_generation(
            package=args.model,
            controller=args.controller,
            prompts=args.prompts,
            out=args.out,
            max_new_tokens=args.max_tokens,
            mlp_library=args.mlp_library,
            attention_library=args.attention_library,
            threads=args.threads,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
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
                vocabulary_ivf_iterations=int(
                    configured("vocabulary_ivf_iterations", 20)
                ),
                local_window=int(configured("local_window", 16)),
                cycles=int(configured("cycles", 2)),
            )
        )
    elif args.command == "compile-native-bitnet":
        print(
            compile_native_bitnet_package(
                args.model,
                args.artifact,
                args.out,
                artifact_sha256=args.artifact_sha256,
                revision=args.revision,
                cache_dir=args.cache_dir,
                kernel_threads=args.threads,
            )
        )
    elif args.command == "install-native-bitnet-controller":
        print(
            install_native_bitnet_controller(
                args.model,
                args.controller,
            )
        )
    elif args.command == "install-native-bitnet-semantic-memory":
        print(
            install_native_bitnet_semantic_memory(
                args.model,
                args.index,
                args.policy,
                args.adjudication,
                args.out,
                coordinate_index_sha256=args.index_sha256,
                policy_manifest_sha256=args.policy_sha256,
                adjudication_sha256=args.adjudication_sha256,
            )
        )
    elif args.command == "generate":
        manifest = json.loads(
            (Path(args.model) / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("format") == "engram-native-bitnet":
            with NativeBitNetRuntime(args.model) as runtime:
                result = runtime.generate(
                    args.prompt,
                    max_new_tokens=args.max_tokens,
                )
            print(result.text)
            print(
                json.dumps(
                    {
                        "tokens": list(result.generated_tokens),
                        "elapsed_seconds": result.elapsed_seconds,
                        "mlp_calls": result.mlp_calls,
                        "scheduled_mlp_bytes": result.scheduled_mlp_bytes,
                    },
                    indent=2,
                )
            )
        else:
            runtime = EngramRuntime(args.model)
            prompt_tokens = runtime.tokenize(args.prompt)
            tokens, metrics = runtime.generate_tokens(
                prompt_tokens, max_tokens=args.max_tokens, exact_vocab=args.exact_vocab
            )
            print(runtime.detokenize(tokens))
            print(
                json.dumps(
                    {"tokens": tokens, "metrics": [item.__dict__ for item in metrics]},
                    indent=2,
                )
            )
    elif args.command == "generate-native-bitnet-controller":
        with NativeBitNetRuntime(
            args.model,
            library=args.library,
            threads=args.threads,
            native_projections=True,
        ) as runtime:
            result = runtime.generate_controller_bounded(
                args.prompt,
                max_new_tokens=args.max_tokens,
                attention_library=args.attention_library,
            )
        print(result.text)
        print(
            json.dumps(
                {
                    "tokens": list(result.generated_tokens),
                    "elapsed_seconds": result.elapsed_seconds,
                    "controller_mode": result.controller_mode,
                    "controller_seconds": result.controller_seconds,
                    "attention_state_bytes": result.attention_state_bytes,
                    "decoder_layer_forward_calls": (result.decoder_layer_forward_calls),
                },
                indent=2,
            )
        )
    elif args.command == "generate-native-bitnet":
        with NativeBitNetRuntime(
            args.model,
            library=args.library,
            threads=args.threads,
            native_projections=args.native_projections,
        ) as runtime:
            if args.bounded_attention:
                result = runtime.generate_bounded(
                    args.prompt,
                    max_new_tokens=args.max_tokens,
                    attention_library=args.attention_library,
                    local_window=args.local_window,
                    older_candidates=args.candidates,
                    older_top_k=args.top_k,
                    sink_tokens=args.sink_tokens,
                )
            else:
                result = runtime.generate(
                    args.prompt,
                    max_new_tokens=args.max_tokens,
                )
        print(result.text)
        print(
            json.dumps(
                {
                    "prompt_tokens": list(result.prompt_tokens),
                    "tokens": list(result.generated_tokens),
                    "elapsed_seconds": result.elapsed_seconds,
                    "prefill_seconds": result.prefill_seconds,
                    "decode_seconds": result.decode_seconds,
                    "mlp_calls": result.mlp_calls,
                    "mlp_elapsed_seconds": result.mlp_elapsed_seconds,
                    "scheduled_mlp_bytes": result.scheduled_mlp_bytes,
                    "maximum_scratch_bytes": result.maximum_scratch_bytes,
                    "attention_mode": result.attention_mode,
                    "attention_tokens_seen": result.attention_tokens_seen,
                    "attention_logical_read_bytes": (
                        result.attention_logical_read_bytes
                    ),
                    "attention_state_bytes": result.attention_state_bytes,
                    "attention_scratch_bytes": result.attention_scratch_bytes,
                    "qkv_projection_seconds": result.qkv_projection_seconds,
                    "rope_seconds": result.rope_seconds,
                    "native_attention_seconds": result.native_attention_seconds,
                    "output_projection_seconds": (result.output_projection_seconds),
                    "native_attention_calls": result.native_attention_calls,
                },
                indent=2,
            )
        )
    elif args.command == "chat-native-bitnet":
        with NativeBitNetDIPTokenRuntime(
            args.model,
            library=args.library,
            threads=args.threads,
        ) as runtime:
            try:
                run_native_bitnet_chat(
                    runtime,
                    max_new_tokens=args.max_tokens,
                    system_prompt=args.system,
                )
            except KeyboardInterrupt:
                print("\nChat interrupted.")
    elif args.command == "validate":
        manifest = json.loads(
            (Path(args.model) / "manifest.json").read_text(encoding="utf-8")
        )
        result = (
            validate_native_bitnet_package(args.model)
            if manifest.get("format") == "engram-native-bitnet"
            else validate_package(args.model)
        )
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
        report = evaluate_controller_gate(
            seed=args.seed, samples=args.samples, width=args.width
        )
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "controller_gate.json"
        atomic_json(path, report)
        print(path)
    elif args.command == "evaluate-mlp-intervention":
        report = evaluate_mlp_interventions(
            args.model,
            args.dataset,
            calibration_traces=args.calibration_traces,
            variants=args.variants,
            top_ks=args.top_k,
            candidate_counts=args.candidates,
            input_fractions=args.input_fractions,
            rank=args.rank,
            regularization=args.regularization,
            calibration_records=args.calibration_records,
            posting_groups=args.posting_groups,
            posting_size=args.posting_size,
            overlap_iterations=args.overlap_iterations,
            max_replication=args.max_replication,
            layers=args.layers,
            layer_mode=args.layer_mode,
            max_records=args.max_records,
            device=args.device,
            allow_calibration_overlap=args.allow_calibration_overlap,
            evaluation_role=args.evaluation_role,
            configuration_selection_traces=args.configuration_selection_traces,
            layer_top_ks=args.layer_top_k,
            paq_group_size=args.paq_group_size,
            paq_codebooks=args.paq_codebooks,
            paq_codebook_size=args.paq_codebook_size,
            paq_iterations=args.paq_iterations,
            paq_sample_limit=args.paq_sample_limit,
            paq_seed=args.paq_seed,
            paq_cacheline_amplification=args.paq_cacheline_amplification,
        )
        json_path, markdown_path = write_mlp_intervention_report(report, args.out)
        print(json_path)
        print(markdown_path)
    elif args.command == "sweep-dip":
        report = evaluate_dip_exact_completion_sweep(
            args.model,
            args.validation_traces,
            input_fractions=args.input_fractions,
            top_k=args.top_k,
            candidate_counts=args.candidates,
            validation_records=args.validation_records,
            input_block_size=args.input_block_size,
        )
        json_path, markdown_path = write_dip_sweep_report(report, args.out)
        print(json_path)
        print(markdown_path)
    elif args.command == "sweep-intrinsic-sparsity":
        report = evaluate_intrinsic_sparse_gate_sweep(
            args.model,
            args.calibration_traces,
            args.validation_traces,
            sparsities=args.sparsities,
            activations=args.activations,
            calibration_records=args.calibration_records,
            validation_records=args.validation_records,
            maximum_mean_relative_l2=args.maximum_mean_relative_l2,
            maximum_traffic_fraction=args.maximum_traffic_fraction,
        )
        json_path, markdown_path = write_intrinsic_sparse_gate_report(report, args.out)
        print(json_path)
        print(markdown_path)
        return 0 if report["eligible_arms"] else 2
    elif args.command == "train-intrinsic-sparse-boundaries":
        report = train_intrinsic_sparse_boundaries(
            args.model,
            args.training_traces,
            args.validation_traces,
            args.out,
            layers=args.layers,
            target_sparsity=args.target_sparsity,
            initial_artifact=args.initial_artifact,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            sparsity_weight=args.sparsity_weight,
            cosine_weight=args.cosine_weight,
            temperature_fraction=args.temperature_fraction,
            warmup_steps=args.warmup_steps,
            start_threshold_fraction=args.start_threshold_fraction,
            evaluation_interval=args.evaluation_interval,
            maximum_mean_relative_l2=args.maximum_mean_relative_l2,
            maximum_traffic_fraction=args.maximum_traffic_fraction,
            max_train_records=args.max_train_records,
            max_validation_records=args.max_validation_records,
            seed=args.seed,
            device=args.device,
        )
        print(args.out / "intrinsic_sparse_boundary_training.json")
        return 0 if report["screen"]["passed"] else 2
    elif args.command == "train-fully-sparse-boundaries":
        report = train_fully_sparse_boundaries(
            args.model,
            args.training_traces,
            args.validation_traces,
            args.out,
            layers=args.layers,
            input_fraction=args.input_fraction,
            intermediate_fraction=args.intermediate_fraction,
            initial_artifact=args.initial_artifact,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            start_sparse_fraction=args.start_sparse_fraction,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            cosine_weight=args.cosine_weight,
            evaluation_interval=args.evaluation_interval,
            maximum_mean_relative_l2=args.maximum_mean_relative_l2,
            maximum_traffic_fraction=args.maximum_traffic_fraction,
            max_train_records=args.max_train_records,
            max_validation_records=args.max_validation_records,
            seed=args.seed,
            device=args.device,
        )
        print(args.out / "fully_sparse_boundary_training.json")
        return 0 if report["screen"]["passed"] else 2
    elif args.command == "sweep-rank-router":
        report = evaluate_rank_router_regularization_sweep(
            args.model,
            args.calibration_traces,
            args.validation_traces,
            regularizations=args.regularization,
            top_k=args.top_k,
            candidate_counts=args.candidates,
            rank=args.rank,
            calibration_records=args.calibration_records,
            validation_records=args.validation_records,
            cache_dir=args.cache,
        )
        json_path, markdown_path = write_rank_router_sweep_report(report, args.out)
        print(json_path)
        print(markdown_path)
    elif args.command == "sweep-correction-capsules":
        report = evaluate_correction_capsule_sweep(
            args.model,
            args.calibration_traces,
            args.validation_traces,
            membership_cache=args.membership_cache,
            router_rank=args.router_rank,
            router_regularization=args.router_regularization,
            top_k=args.top_k,
            candidate_count=args.candidates,
            capsule_counts=args.capsules,
            capsule_ranks=args.capsule_ranks,
            capsule_ridge=args.capsule_ridge,
            capsule_iterations=args.capsule_iterations,
            radius_scale=args.radius_scale,
            priority_fractions=args.priority_fractions,
            radius_quantile=args.radius_quantile,
            calibration_records=args.calibration_records,
            validation_records=args.validation_records,
        )
        json_path, markdown_path = write_correction_capsule_sweep_report(
            report, args.out
        )
        print(json_path)
        print(markdown_path)
    elif args.command == "build-distillation-corpus":
        report = build_distillation_corpus(
            args.model,
            args.input,
            args.out,
            sequence_length=args.sequence_length,
            max_sequences=args.max_sequences,
            minimum_tokens=args.minimum_tokens,
        )
        print(report["dataset_path"])
    elif args.command == "build-distillation-holdout":
        report = build_distillation_tail_holdout(
            args.source,
            args.out,
            records=args.records,
        )
        print(report["holdout"]["path"])
    elif args.command == "train-sparse-student":
        report = train_sparse_student(
            args.model,
            args.calibration_dataset,
            args.validation_dataset,
            args.calibration_traces,
            args.out,
            top_k=args.top_k,
            candidate_count=args.candidates,
            router_rank=args.router_rank,
            router_regularization=args.router_regularization,
            adapter_rank=args.adapter_rank,
            residual_rank=args.residual_rank,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            router_learning_rate=args.router_learning_rate,
            local_weight=args.local_weight,
            hidden_weight=args.hidden_weight,
            logit_weight=args.logit_weight,
            teacher_forced_local_weight=args.teacher_forced_local_weight,
            router_weight=args.router_weight,
            locality_weight=args.locality_weight,
            routing_mode=args.routing_mode,
            input_fraction=args.input_fraction,
            temperature=args.temperature,
            cache_line_records=args.cache_line_records,
            batch_size=args.batch_size,
            gradient_diagnostics=args.gradient_diagnostics,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            coadapt_backbone=args.coadapt_backbone,
            coadapt_embeddings_and_head=args.coadapt_embeddings_and_head,
            layers=args.layers,
            exact_dense_start=args.exact_dense_start,
            dense_warmup_steps=args.dense_warmup_steps,
            start_top_k=args.start_top_k,
            start_candidate_count=args.start_candidates,
            start_input_fraction=args.start_input_fraction,
            anneal_steps=args.anneal_steps,
            start_temperature=args.start_temperature,
            checkpoint_selection_records=args.checkpoint_selection_records,
            checkpoint_selection_every=args.checkpoint_selection_every,
            router_group_size=args.router_group_size,
            train_full_mlp=args.train_full_mlp,
            training_dataset=args.training_dataset,
            max_train_records=args.max_train_records,
            max_validation_records=args.max_validation_records,
            device=args.device,
        )
        print(args.out / "sparse_teacher_training.json")
    elif args.command == "evaluate-structured-experts":
        report = evaluate_structured_expert_shadow(
            args.model,
            args.calibration_traces,
            args.validation_traces,
            args.out,
            experts=args.experts,
            active_experts=args.active_experts,
            regularization=args.regularization,
            grouping_iterations=args.grouping_iterations,
            calibration_records=args.calibration_records,
            validation_records=args.validation_records,
        )
        print(args.out / "structured_expert_shadow.json")
        return 0 if report["screen"]["passed"] else 2
    elif args.command == "train-grouped-sparse-boundaries":
        report = train_grouped_sparse_boundaries(
            args.model,
            args.training_traces,
            args.validation_traces,
            args.out,
            layers=args.layers,
            top_k=args.top_k,
            start_top_k=args.start_top_k,
            router_rank=args.router_rank,
            router_regularization=args.router_regularization,
            adapter_rank=args.adapter_rank,
            router_warmup_steps=args.router_warmup_steps,
            anneal_steps=args.anneal_steps,
            settle_steps=args.settle_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            router_learning_rate=args.router_learning_rate,
            route_weight=args.route_weight,
            cosine_weight=args.cosine_weight,
            dense_anchor_weight=args.dense_anchor_weight,
            start_temperature=args.start_temperature,
            temperature=args.temperature,
            evaluation_interval=args.evaluation_interval,
            maximum_mean_relative_l2=args.maximum_mean_relative_l2,
            max_train_records=args.max_train_records,
            max_validation_records=args.max_validation_records,
            learned_pairing=not args.identity_pairing,
            device=args.device,
        )
        print(args.out / "grouped_sparse_boundaries.json")
        return 0 if report["screen"]["passed"] else 2
    elif args.command == "train-shared-expert-boundaries":
        report = train_shared_expert_boundaries(
            args.model,
            args.training_traces,
            args.validation_traces,
            args.out,
            layers=args.layers,
            shared_records=args.shared_records,
            experts=args.experts,
            active_experts=args.active_experts,
            start_active_experts=args.start_active_experts,
            grouping_iterations=args.grouping_iterations,
            router_regularization=args.router_regularization,
            router_warmup_steps=args.router_warmup_steps,
            anneal_steps=args.anneal_steps,
            settle_steps=args.settle_steps,
            batch_size=args.batch_size,
            oracle_batch_size=args.oracle_batch_size,
            learning_rate=args.learning_rate,
            router_learning_rate=args.router_learning_rate,
            route_weight=args.route_weight,
            cosine_weight=args.cosine_weight,
            dense_anchor_weight=args.dense_anchor_weight,
            start_temperature=args.start_temperature,
            temperature=args.temperature,
            evaluation_interval=args.evaluation_interval,
            maximum_mean_relative_l2=args.maximum_mean_relative_l2,
            max_train_records=args.max_train_records,
            max_validation_records=args.max_validation_records,
            device=args.device,
        )
        print(args.out / "shared_expert_boundaries.json")
        return 0 if report["screen"]["passed"] else 2
    elif args.command == "train-activation-aware-aq":
        report = train_activation_aware_aq_boundaries(
            args.model,
            args.training_traces,
            args.validation_traces,
            args.out,
            layers=args.layers,
            fit_iterations=args.fit_iterations,
            fit_sample_limit=args.fit_sample_limit,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            cosine_loss_weight=args.cosine_loss_weight,
            projection_loss_weight=args.projection_loss_weight,
            checkpoint_interval=args.checkpoint_interval,
            maximum_mean_relative_l2=args.maximum_mean_relative_l2,
            max_train_records=args.max_train_records,
            max_validation_records=args.max_validation_records,
            seed=args.seed,
            device=args.device,
        )
        print(args.out / "activation_aware_aq_boundaries.json")
        return 0 if report["screen"]["passed"] else 2
    elif args.command == "train-projection-aq":
        report = train_projection_aq_layers(
            args.model,
            args.training_traces,
            args.validation_traces,
            args.out,
            layers=args.layers,
            p_steps_per_cycle=args.p_steps_per_cycle,
            v_cycles=args.v_cycles,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_interval=args.checkpoint_interval,
            fit_iterations=args.fit_iterations,
            fit_sample_limit=args.fit_sample_limit,
            v_max_records=args.v_max_records,
            v_change_fraction=args.v_change_fraction,
            selection_records=args.selection_records,
            maximum_mean_relative_l2=args.maximum_mean_relative_l2,
            maximum_p95_relative_l2=args.maximum_p95_relative_l2,
            minimum_mean_cosine=args.minimum_mean_cosine,
            max_train_records=args.max_train_records,
            max_validation_records=args.max_validation_records,
            seed=args.seed,
            device=args.device,
        )
        print(args.out / "projection_aq_layers.json")
        return 0 if report["screen"]["passed"] else 2
    elif args.command == "evaluate-native-gate-channels":
        report = evaluate_native_gate_channel_shadow(
            args.model,
            args.validation_traces,
            args.out,
            input_fractions=args.input_fractions,
            top_k=args.top_k,
            validation_records=args.validation_records,
        )
        print(args.out / "native_gate_channel_shadow.json")
        return 0 if report["screen"]["passed"] else 2
    elif args.command == "evaluate-native-gate-residual":
        report = evaluate_native_gate_residual_shadow(
            args.model,
            args.calibration_traces,
            args.validation_traces,
            args.out,
            ranks=args.ranks,
            blends=args.blends,
            input_fraction=args.input_fraction,
            top_k=args.top_k,
            regularization=args.regularization,
            calibration_records=args.calibration_records,
            validation_records=args.validation_records,
            active_record_limit=args.active_record_limit,
        )
        print(args.out / "native_gate_utility_residual.json")
        return 0 if report["screen"]["passed"] else 2
    elif args.command == "recalibrate-native-gate-residual":
        report = recalibrate_native_gate_residual(
            args.model,
            args.calibration_dataset,
            args.initial_residual,
            args.out,
            rank=args.rank,
            blends=args.blends,
            input_fraction=args.input_fraction,
            top_k=args.top_k,
            regularization=args.regularization,
            fit_fraction=args.fit_fraction,
            fit_states=args.fit_states,
            validation_states=args.validation_states,
            device=args.device,
        )
        print(args.out / "native_gate_on_policy_residual.json")
        return 0 if report["screen"]["passed"] else 2
    elif args.command == "train-native-gate-traces":
        report = train_native_gate_trace_student(
            args.model,
            args.calibration_traces,
            args.validation_traces,
            args.out,
            layers=args.layers,
            top_k=args.top_k,
            input_fraction=args.input_fraction,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dense_shadow_weight=args.dense_shadow_weight,
            utility_weight=args.utility_weight,
            temperature=args.temperature,
            calibration_records=args.calibration_records,
            validation_records=args.validation_records,
            device=args.device,
        )
        print(args.out / "native_gate_trace_training.json")
        return 0 if report["screen"]["passed"] else 2
    elif args.command == "train-native-gate-e2e":
        report = train_native_gate_end_to_end(
            args.model,
            args.training_dataset,
            args.validation_dataset,
            args.out,
            target_top_k=args.target_top_k,
            target_input_fraction=args.target_input_fraction,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            anneal_steps=args.anneal_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            local_weight=args.local_weight,
            dense_shadow_weight=args.dense_shadow_weight,
            hidden_weight=args.hidden_weight,
            logit_weight=args.logit_weight,
            utility_weight=args.utility_weight,
            temperature=args.temperature,
            max_train_records=args.max_train_records,
            max_validation_records=args.max_validation_records,
            device=args.device,
            save_artifact=not args.no_artifact,
            strict_q4_deployment=args.strict_q4_deployment,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            utility_residual=args.utility_residual,
        )
        print(args.out / "native_gate_end_to_end.json")
        return 0 if report["gate"]["passed"] else 2
    elif args.command == "train-fully-sparse-student":
        report = train_fully_sparse_student(
            args.model,
            args.training_dataset,
            args.validation_dataset,
            args.out,
            input_fraction=args.input_fraction,
            intermediate_fraction=args.intermediate_fraction,
            input_counts=args.input_counts,
            intermediate_counts=args.intermediate_counts,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            anneal_steps=args.anneal_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            backbone_learning_rate=args.backbone_learning_rate,
            local_weight=args.local_weight,
            hidden_weight=args.hidden_weight,
            logit_weight=args.logit_weight,
            label_weight=args.label_weight,
            max_train_records=args.max_train_records,
            max_validation_records=args.max_validation_records,
            device=args.device,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            coadapt_backbone=args.coadapt_backbone,
            coadapt_embeddings_and_head=args.coadapt_embeddings_and_head,
            residual_rank=args.residual_rank,
        )
        print(args.out / "fully_sparse_distillation.json")
        return 0 if report["gate"]["passed"] else 2
    elif args.command == "train-budget-native-ternary":
        report = train_budget_native_ternary_student(
            args.model,
            args.training_dataset,
            args.validation_dataset,
            args.out,
            group_size=args.group_size,
            steps=args.steps,
            dense_warmup_steps=args.dense_warmup_steps,
            anneal_steps=args.anneal_steps,
            transition_mode=args.transition_mode,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            backbone_learning_rate=args.backbone_learning_rate,
            local_weight=args.local_weight,
            hidden_weight=args.hidden_weight,
            final_hidden_weight=args.final_hidden_weight,
            final_cka_weight=args.final_cka_weight,
            logit_weight=args.logit_weight,
            teacher_top1_weight=args.teacher_top1_weight,
            label_weight=args.label_weight,
            temperature=args.temperature,
            confidence_weight=args.confidence_weight,
            coadapt_backbone=args.coadapt_backbone,
            coadapt_embeddings_and_head=args.coadapt_embeddings_and_head,
            backbone_start_step=args.backbone_start_step,
            max_train_records=args.max_train_records,
            training_record_offset=args.training_record_offset,
            max_validation_records=args.max_validation_records,
            device=args.device,
            save_artifact=not args.no_artifact,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            initial_checkpoint=args.initial_checkpoint,
        )
        print(args.out / "budget_native_ternary_training.json")
        return 0 if report["gate"]["passed"] else 2
    elif args.command == "train-width-pruned-student":
        report = train_width_pruned_student(
            args.model,
            args.training_dataset,
            args.validation_dataset,
            args.out,
            calibration_traces=args.calibration_traces,
            target_width=args.target_width,
            target_widths=args.target_widths,
            steps=args.steps,
            replacement_steps=args.replacement_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            local_weight=args.local_weight,
            hidden_weight=args.hidden_weight,
            logit_weight=args.logit_weight,
            initialization_records=args.initialization_records,
            local_warmup_steps=args.local_warmup_steps,
            local_batch_size=args.local_batch_size,
            local_learning_rate=args.local_learning_rate,
            max_train_records=args.max_train_records,
            max_validation_records=args.max_validation_records,
            device=args.device,
            save_artifact=not args.no_artifact,
            strict_q4_deployment=args.strict_q4_deployment,
            fake_q4_training=args.fake_q4_training,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            initial_checkpoint=args.initial_checkpoint,
            coadapt_backbone=args.coadapt_backbone,
            coadapt_embeddings_and_head=args.coadapt_embeddings_and_head,
            backbone_learning_rate=args.backbone_learning_rate,
        )
        print(args.out / "width_pruned_training.json")
        return 0 if report["gate"]["passed"] else 2
    elif args.command == "evaluate-width-local-ceiling":
        report = evaluate_width_pruned_local_ceiling(
            args.model,
            args.training_traces,
            args.validation_traces,
            args.initial_checkpoint,
            args.out,
            layers=args.layers,
            target_width=args.target_width,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_train_records=args.max_train_records,
            max_validation_records=args.max_validation_records,
            maximum_mean_relative_l2=args.maximum_mean_relative_l2,
            minimum_improvement_fraction=args.minimum_improvement_fraction,
            device=args.device,
        )
        print(args.out / "width_local_ceiling.json")
        return 0 if report["screen"]["passed"] else 2
    elif args.command == "evaluate-gated-background-ceiling":
        report = evaluate_gated_background_ceiling(
            args.model,
            args.training_traces,
            args.validation_traces,
            args.out,
            layers=args.layers,
            top_k=args.top_k,
            background_width=args.background_width,
            router_rank=args.router_rank,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_train_records=args.max_train_records,
            max_validation_records=args.max_validation_records,
            maximum_mean_relative_l2=args.maximum_mean_relative_l2,
            device=args.device,
            seed=args.seed,
        )
        print(args.out / "gated_background_ceiling.json")
        return 0 if report["screen"]["passed"] else 2
    elif args.command == "evaluate-width-residual-sweep":
        report = evaluate_width_residual_sweep(
            args.model,
            args.training_traces,
            args.validation_traces,
            args.initial_checkpoint,
            args.out,
            layers=args.layers,
            compact_width=args.compact_width,
            ranks=args.ranks,
            ridge_factor=args.ridge_factor,
            max_train_records=args.max_train_records,
            max_validation_records=args.max_validation_records,
            maximum_mean_relative_l2=args.maximum_mean_relative_l2,
        )
        print(args.out / "width_residual_sweep.json")
        return 0 if report["screen"]["passed"] else 2
    elif args.command == "evaluate-oracle-residual-ceiling":
        report = evaluate_oracle_residual_ceiling(
            args.model,
            args.training_traces,
            args.validation_traces,
            args.out,
            layers=args.layers,
            top_k=args.top_k,
            ranks=args.ranks,
            ridge_factor=args.ridge_factor,
            max_train_records=args.max_train_records,
            max_validation_records=args.max_validation_records,
            maximum_mean_relative_l2=args.maximum_mean_relative_l2,
        )
        print(args.out / "oracle_residual_ceiling.json")
        return 0 if report["screen"]["passed"] else 2
    elif args.command == "gate-mlp-intervention":
        if len(args.report) > 1 and args.out is None:
            raise ValueError("--out is required when composing multiple reports")
        if len(args.report) > 1:
            output_report = (args.out / "mlp_intervention.json").resolve()
            if output_report in {path.resolve() for path in args.report}:
                raise ValueError("composite --out must not overwrite an input report")
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.report]
        gated = (
            combine_mlp_intervention_reports(reports)
            if len(reports) > 1
            else apply_mlp_intervention_gates(reports[0])
        )
        if len(reports) > 1:
            gated["composite_sources"] = [
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for path in args.report
            ]
        target = args.out if args.out is not None else args.report[0].parent
        json_path, markdown_path = write_mlp_intervention_report(gated, target)
        print(json_path)
        print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
