#include "engram/controller.h"

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
  return 0;
}
