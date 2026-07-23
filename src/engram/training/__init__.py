from .corpus import build_distillation_corpus
from .activation_aware_aq import train_activation_aware_aq_boundaries
from .entropy_q3_codec import (
    EntropyQ3LayerWeights,
    decode_entropy_q3_artifact,
    entropy_q3_dynamic_traffic,
    entropy_q3_forward,
    load_entropy_q3_artifact,
    save_entropy_q3_artifact,
)
from .gated_background import evaluate_gated_background_ceiling
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
from .shared_expert_boundaries import train_shared_expert_boundaries
from .on_policy import recalibrate_native_gate_residual
from .projection_aq_pipeline import train_projection_aq_layers
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
    "evaluate_native_gate_channel_shadow",
    "evaluate_native_gate_residual_shadow",
    "evaluate_structured_expert_shadow",
    "evaluate_gated_background_ceiling",
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
    "recalibrate_native_gate_residual",
    "train_native_gate_end_to_end",
    "train_native_gate_trace_student",
    "train_grouped_sparse_boundaries",
    "train_activation_aware_aq_boundaries",
    "train_projection_aq_layers",
    "train_shared_expert_boundaries",
    "train_sparse_student",
    "train_width_pruned_student",
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
