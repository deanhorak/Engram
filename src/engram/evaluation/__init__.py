from .mlp_intervention import evaluate_mlp_interventions
from .native_bitnet_parity import evaluate_native_bitnet_parity
from .native_bitnet_kernel import (
    NativeBitNetCPUKernel,
    evaluate_native_bitnet_kernel_confirmation,
)
from .native_bitnet_dip_kernel import NativeBitNetDIPCPUKernel
from .native_attention_benchmark import benchmark_native_streaming_attention
from .router_sweep import evaluate_rank_router_regularization_sweep
from .dip_sweep import evaluate_dip_exact_completion_sweep
from .intrinsic_sparsity import (
    evaluate_intrinsic_sparse_gate_sweep,
    exact_gate_sparse_traffic,
    write_intrinsic_sparse_gate_report,
)
from .correction_sweep import evaluate_correction_capsule_sweep
from .gates import (
    apply_mlp_intervention_gates,
    combine_mlp_intervention_reports,
    evaluate_mlp_arm_gate,
)
from .report import (
    write_attention_report,
    write_correction_capsule_sweep_report,
    write_mlp_intervention_report,
    write_oracle_report,
    write_rank_router_sweep_report,
    write_dip_sweep_report,
    write_semantic_routing_report,
)

__all__ = [
    "evaluate_mlp_interventions",
    "evaluate_native_bitnet_parity",
    "NativeBitNetCPUKernel",
    "NativeBitNetDIPCPUKernel",
    "evaluate_native_bitnet_kernel_confirmation",
    "analyze_native_bitnet_layer_oracle",
    "evaluate_native_bitnet_oracle",
    "evaluate_native_bitnet_oracle_causal",
    "evaluate_native_bitnet_oracle_layer_sweep",
    "benchmark_native_streaming_attention",
    "evaluate_native_bitnet_attention_substitution",
    "evaluate_native_bitnet_controller_substitution",
    "evaluate_controller_only_trace",
    "evaluate_native_bitnet_controller_generation",
    "evaluate_rank_router_regularization_sweep",
    "evaluate_dip_exact_completion_sweep",
    "evaluate_intrinsic_sparse_gate_sweep",
    "exact_gate_sparse_traffic",
    "write_intrinsic_sparse_gate_report",
    "evaluate_correction_capsule_sweep",
    "apply_mlp_intervention_gates",
    "combine_mlp_intervention_reports",
    "evaluate_mlp_arm_gate",
    "write_attention_report",
    "write_correction_capsule_sweep_report",
    "write_mlp_intervention_report",
    "write_oracle_report",
    "write_rank_router_sweep_report",
    "write_dip_sweep_report",
    "write_semantic_routing_report",
]


def __getattr__(name):
    # Avoid a runtime.native_bitnet -> evaluation package -> attention evaluator
    # cycle. The evaluator itself depends on NativeBitNetRuntime.
    if name == "evaluate_native_bitnet_attention_substitution":
        from .native_bitnet_attention import (
            evaluate_native_bitnet_attention_substitution,
        )

        return evaluate_native_bitnet_attention_substitution
    if name == "evaluate_native_bitnet_controller_substitution":
        from .controller_substitution import (
            evaluate_native_bitnet_controller_substitution,
        )

        return evaluate_native_bitnet_controller_substitution
    if name == "evaluate_controller_only_trace":
        from .controller_only import evaluate_controller_only_trace

        return evaluate_controller_only_trace
    if name == "evaluate_native_bitnet_controller_generation":
        from .native_bitnet_controller_generation import (
            evaluate_native_bitnet_controller_generation,
        )

        return evaluate_native_bitnet_controller_generation
    if name in {
        "analyze_native_bitnet_layer_oracle",
        "evaluate_native_bitnet_oracle",
        "evaluate_native_bitnet_oracle_causal",
        "evaluate_native_bitnet_oracle_layer_sweep",
    }:
        from . import native_bitnet_oracle

        return getattr(native_bitnet_oracle, name)
    raise AttributeError(name)
