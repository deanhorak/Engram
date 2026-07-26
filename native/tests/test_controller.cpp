#include "engram/controller.h"
#include "engram/native_shell_c.h"
#include "engram/operator_residual_c.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

namespace {

constexpr double kInputKernel[] = {
    0.2, -0.1, 0.3,  0.4,  -0.5, 0.2,
    0.1, 0.25, -0.2, 0.15, 0.35, -0.45,
};
constexpr double kRecurrentKernel[] = {
    0.05, -0.3, 0.2, 0.1, -0.25, 0.4,
    0.15, 0.2, -0.1, 0.3, 0.5, -0.2,
};
constexpr double kBias[] = {0.01, -0.02, 0.03, 0.04, -0.05, 0.06};
constexpr double kStageEmbeddings[] = {0.1, -0.1, -0.2, 0.3};
constexpr double kAdapterDown[] = {0.2, -0.1, -0.3, 0.25};
constexpr double kAdapterUp[] = {0.4, -0.2, 0.1, 0.35};
constexpr double kState[] = {0.3, -0.4};
constexpr double kInput[] = {0.7, -1.2};

engram::ControllerWeightsView weights(const std::size_t adapter_rank = 1) {
  return {
      .input_dimension = 2,
      .state_dimension = 2,
      .stage_count = 2,
      .adapter_rank = adapter_rank,
      .input_kernel = kInputKernel,
      .recurrent_kernel = kRecurrentKernel,
      .bias = kBias,
      .stage_embeddings = kStageEmbeddings,
      .adapter_down = adapter_rank == 0 ? nullptr : kAdapterDown,
      .adapter_up = adapter_rank == 0 ? nullptr : kAdapterUp,
  };
}

bool close(const double actual, const double expected,
           const double tolerance = 2e-15) {
  return std::abs(actual - expected) <= tolerance;
}

int fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}

}  // namespace

int main() {
  engram::ControllerWorkspace workspace(2, 1);
  if (workspace.state_dimension() != 2 || workspace.adapter_rank() != 1 ||
      workspace.persistent_elements() != 17 ||
      workspace.persistent_bytes() != 17 * sizeof(double)) {
    return fail("controller workspace metrics mismatch");
  }

  double stage_zero[2]{};
  engram::controller_step_scalar(weights(), kState, kInput, 0, workspace,
                                 stage_zero);
  if (!close(stage_zero[0], -0.18671565046157471) ||
      !close(stage_zero[1], -0.026312618737007326)) {
    return fail("stage-zero controller step differs from Python reference");
  }

  double stage_one[2]{};
  engram::controller_step_scalar(weights(), kState, kInput, 1, workspace,
                                 stage_one);
  if (!close(stage_one[0], -0.26098510812250686) ||
      !close(stage_one[1], 0.035311258129425982)) {
    return fail("stage-one adapter result differs from Python reference");
  }

  double no_adapter[2]{};
  engram::ControllerWorkspace no_adapter_workspace(2, 0);
  engram::controller_step_scalar(weights(0), kState, kInput, 1,
                                 no_adapter_workspace, no_adapter);
  if (!close(no_adapter[0], -0.25801229516332247) ||
      !close(no_adapter[1], 0.043660028969739306)) {
    return fail("rank-zero adapter result differs from Python reference");
  }

  double fixed[2]{};
  engram::controller_run_fixed_scalar(weights(), kState, kInput, 3, 0,
                                      workspace, fixed);
  if (!close(fixed[0], -0.51321222902654451, 4e-15) ||
      !close(fixed[1], 0.38281792524673441, 4e-15)) {
    return fail("fixed-cycle controller result differs from Python reference");
  }

  // A step may update the state in place after gates and candidate are staged.
  double in_place[] = {kState[0], kState[1]};
  engram::controller_step_scalar(weights(), in_place, kInput, 0, workspace,
                                 in_place);
  if (!close(in_place[0], stage_zero[0]) ||
      !close(in_place[1], stage_zero[1])) {
    return fail("in-place controller step mismatch");
  }

  bool rejected_stage = false;
  try {
    engram::controller_step_scalar(weights(), kState, kInput, 2, workspace,
                                   stage_zero);
  } catch (const std::out_of_range&) {
    rejected_stage = true;
  }
  if (!rejected_stage) {
    return fail("invalid controller stage was accepted");
  }

  const float residual_state[] = {1.0F, 2.0F, -1.0F, 0.5F};
  const float residual_semantic[] = {0.5F, -0.5F, 0.25F, 0.25F};
  const float residual_episodic[] = {-0.25F, 0.25F, 0.5F, -0.5F};
  float residual_output[4] = {};
  float residual_rms[2] = {};
  if (engram_operator_residual_step_f32(
          residual_state, residual_semantic, residual_episodic, 2, 2, 1.0F,
          1.0F, residual_output, residual_rms) != 0) {
    return fail("native operator residual rejected valid input");
  }
  for (std::size_t row = 0; row < 2; ++row) {
    const float mean_square =
        (residual_output[row * 2] * residual_output[row * 2] +
         residual_output[row * 2 + 1] * residual_output[row * 2 + 1]) /
        2.0F;
    if (std::abs(mean_square - 1.0F) > 2.0e-5F ||
        !(residual_rms[row] > 0.0F)) {
      return fail("native operator residual normalization mismatch");
    }
  }

  const std::uint16_t embedding_table[] = {
      0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0,
  };
  const std::int64_t token_ids[] = {2, 0};
  std::uint16_t embeddings[4] = {};
  if (engram_embedding_lookup_bf16(embedding_table, 3, 2, token_ids, 2,
                                   embeddings) != 0 ||
      embeddings[0] != 0x40A0 || embeddings[1] != 0x40C0 ||
      embeddings[2] != 0x3F80 || embeddings[3] != 0x4000) {
    return fail("native BF16 embedding lookup mismatch");
  }
  const float norm_input[] = {3.0F, 4.0F};
  const std::uint16_t norm_weight[] = {0x3F80, 0x3F80};
  std::uint16_t norm_output[2] = {};
  if (engram_rms_norm_f32_to_bf16(norm_input, norm_weight, 1, 2, 1.0e-6F,
                                  norm_output) != 0 ||
      norm_output[0] != 0x3F59 || norm_output[1] != 0x3F91) {
    return fail("native BF16 RMSNorm mismatch");
  }
  const std::uint16_t vocabulary[] = {
      0x3F80, 0x0000, 0x0000, 0x3F80, 0x3F80, 0x3F80,
  };
  const std::uint16_t vocabulary_input[] = {0x3F80, 0x4000};
  std::int64_t best_token = -1;
  float best_score = 0.0F;
  if (engram_vocab_argmax_bf16(vocabulary_input, vocabulary, 3, 2, 2,
                               &best_token, &best_score) != 0 ||
      best_token != 2 || std::abs(best_score - 3.0F) > 1.0e-6F) {
    return fail("native BF16 vocabulary argmax mismatch");
  }
  std::uint16_t rope_query[] = {0x3F80, 0x4000};
  std::uint16_t rope_key[] = {0x4040, 0x4080};
  const std::int64_t rope_position[] = {0};
  if (engram_rope_bf16(rope_query, 1, rope_key, 1, 1, 1, 2, rope_position, 1,
                       10000.0F) != 0 ||
      rope_query[0] != 0x3F80 || rope_query[1] != 0x4000 ||
      rope_key[0] != 0x4040 || rope_key[1] != 0x4080) {
    return fail("native BF16 RoPE zero-position mismatch");
  }
  return 0;
}
