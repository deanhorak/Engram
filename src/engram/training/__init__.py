from .corpus import build_distillation_corpus, build_distillation_tail_holdout
from .controller_distillation import (
    capture_native_bitnet_controller_traces,
    distill_factorized_controller,
)
from .provider_distillation import joint_distill_operator_provider
from .activation_aware_aq import train_activation_aware_aq_boundaries
from .budget_native_ternary import (
    confidence_weighted_kl,
    grouped_ternary_mlp_class,
    layer_quantization_strengths_for_step,
    masked_linear_cka_loss,
    quantization_strength_for_step,
)
from .budget_native_ternary_codec import (
    BudgetNativeTernaryLayerWeights,
    budget_native_ternary_forward,
    budget_native_ternary_traffic,
    decode_budget_native_ternary_artifact,
    load_budget_native_ternary_artifact,
    save_budget_native_ternary_artifact,
)
from .budget_native_ternary_training import (
    train_budget_native_ternary_student,
)
from .entropy_q3_codec import (
    EntropyQ3LayerWeights,
    decode_entropy_q3_artifact,
    entropy_q3_dynamic_traffic,
    entropy_q3_forward,
    load_entropy_q3_artifact,
    save_entropy_q3_artifact,
)
from .gated_background import evaluate_gated_background_ceiling
from .fully_sparse import fully_sparse_mlp_traffic, train_fully_sparse_boundaries
from .fully_sparse_distillation import (
    fully_sparse_mlp_class,
    progressive_fully_sparse_counts,
    train_fully_sparse_student,
    validate_fully_sparse_artifact_cpu,
)
from .grouped_sparse_boundaries import train_grouped_sparse_boundaries
from .grouped_sparse_codec import (
    decode_grouped_sparse_artifact,
    grouped_sparse_forward,
    grouped_sparse_traffic,
    load_grouped_sparse_artifact,
    save_grouped_sparse_artifact,
)
from .interleaved_entropy_q3_codec import (
    decode_interleaved_entropy_q3_artifact,
    interleaved_entropy_q3_dynamic_traffic,
    interleaved_entropy_q3_forward,
    load_interleaved_entropy_q3_artifact,
    save_interleaved_entropy_q3_artifact,
)
from .interleaved_entropy_q4_codec import (
    EntropyQ4LayerWeights,
    decode_interleaved_entropy_q4_artifact,
    interleaved_entropy_q4_dynamic_traffic,
    interleaved_entropy_q4_forward,
    load_interleaved_entropy_q4_artifact,
    save_interleaved_entropy_q4_artifact,
)
from .intrinsic_sparsity import train_intrinsic_sparse_boundaries
from .linear_constrained_vq import (
    block_hadamard_function,
    linear_constrained_vq_mlp_class,
    linear_constrained_vq_traffic,
)
from .lifted_binary import (
    lifted_binary_mlp_class,
    lifted_binary_traffic,
)
from .codebook_vq import (
    unrestricted_codebook_vq_mlp_class,
    unrestricted_codebook_vq_traffic,
)
from .shared_expert_boundaries import train_shared_expert_boundaries
from .on_policy import recalibrate_native_gate_residual
from .projection_aq_pipeline import train_projection_aq_layers
from .projection_normalized_ternary import (
    projection_normalized_ternary_mlp_class,
    projection_normalized_ternary_traffic,
)
from .recurrent_compact import (
    recurrent_compact_mlp_class,
    recurrent_compact_q4_traffic,
)
from .oracle_residual import evaluate_oracle_residual_ceiling
from .sparse_teacher import train_sparse_student
from .structured_experts import (
    evaluate_native_gate_channel_shadow,
    evaluate_native_gate_residual_shadow,
    evaluate_structured_expert_shadow,
    load_native_gate_utility_residual,
    train_native_gate_end_to_end,
    train_native_gate_trace_student,
)
from .width_pruning import train_width_pruned_student
from .width_pruned_codec import (
    WidthPrunedQ4LayerWeights,
    decode_width_pruned_q4_artifact,
    load_width_pruned_q4_artifact,
    save_width_pruned_q4_artifact,
    width_pruned_q4_forward,
    width_pruned_q4_traffic,
)
from .width_pruned_q3_codec import (
    WidthPrunedQ3LayerWeights,
    decode_width_pruned_q3_artifact,
    load_width_pruned_q3_artifact,
    save_width_pruned_q3_artifact,
    width_pruned_q3_dynamic_traffic,
    width_pruned_q3_forward,
    width_pruned_q3_traffic,
)
from .width_residual import evaluate_width_residual_sweep
from .width_ceiling import evaluate_width_pruned_local_ceiling

__all__ = [
    "build_distillation_corpus",
    "build_distillation_tail_holdout",
    "capture_native_bitnet_controller_traces",
    "BudgetNativeTernaryLayerWeights",
    "budget_native_ternary_forward",
    "budget_native_ternary_traffic",
    "confidence_weighted_kl",
    "decode_budget_native_ternary_artifact",
    "evaluate_native_gate_channel_shadow",
    "evaluate_native_gate_residual_shadow",
    "evaluate_structured_expert_shadow",
    "evaluate_gated_background_ceiling",
    "fully_sparse_mlp_traffic",
    "fully_sparse_mlp_class",
    "EntropyQ3LayerWeights",
    "decode_entropy_q3_artifact",
    "entropy_q3_dynamic_traffic",
    "entropy_q3_forward",
    "load_entropy_q3_artifact",
    "save_entropy_q3_artifact",
    "decode_interleaved_entropy_q3_artifact",
    "interleaved_entropy_q3_dynamic_traffic",
    "interleaved_entropy_q3_forward",
    "load_interleaved_entropy_q3_artifact",
    "save_interleaved_entropy_q3_artifact",
    "EntropyQ4LayerWeights",
    "decode_interleaved_entropy_q4_artifact",
    "interleaved_entropy_q4_dynamic_traffic",
    "interleaved_entropy_q4_forward",
    "load_interleaved_entropy_q4_artifact",
    "save_interleaved_entropy_q4_artifact",
    "evaluate_oracle_residual_ceiling",
    "evaluate_width_pruned_local_ceiling",
    "evaluate_width_residual_sweep",
    "decode_grouped_sparse_artifact",
    "grouped_sparse_forward",
    "grouped_sparse_traffic",
    "load_grouped_sparse_artifact",
    "load_native_gate_utility_residual",
    "load_budget_native_ternary_artifact",
    "block_hadamard_function",
    "linear_constrained_vq_mlp_class",
    "linear_constrained_vq_traffic",
    "lifted_binary_mlp_class",
    "lifted_binary_traffic",
    "unrestricted_codebook_vq_mlp_class",
    "unrestricted_codebook_vq_traffic",
    "projection_normalized_ternary_mlp_class",
    "projection_normalized_ternary_traffic",
    "grouped_ternary_mlp_class",
    "layer_quantization_strengths_for_step",
    "masked_linear_cka_loss",
    "quantization_strength_for_step",
    "recalibrate_native_gate_residual",
    "recurrent_compact_mlp_class",
    "recurrent_compact_q4_traffic",
    "train_native_gate_end_to_end",
    "train_native_gate_trace_student",
    "train_grouped_sparse_boundaries",
    "train_fully_sparse_boundaries",
    "train_fully_sparse_student",
    "progressive_fully_sparse_counts",
    "validate_fully_sparse_artifact_cpu",
    "train_intrinsic_sparse_boundaries",
    "distill_factorized_controller",
    "joint_distill_operator_provider",
    "train_activation_aware_aq_boundaries",
    "train_budget_native_ternary_student",
    "train_projection_aq_layers",
    "train_shared_expert_boundaries",
    "train_sparse_student",
    "train_width_pruned_student",
    "save_budget_native_ternary_artifact",
    "WidthPrunedQ4LayerWeights",
    "decode_width_pruned_q4_artifact",
    "load_width_pruned_q4_artifact",
    "save_width_pruned_q4_artifact",
    "width_pruned_q4_forward",
    "width_pruned_q4_traffic",
    "WidthPrunedQ3LayerWeights",
    "decode_width_pruned_q3_artifact",
    "load_width_pruned_q3_artifact",
    "save_width_pruned_q3_artifact",
    "width_pruned_q3_dynamic_traffic",
    "width_pruned_q3_forward",
    "width_pruned_q3_traffic",
    "save_grouped_sparse_artifact",
]
