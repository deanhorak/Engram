#include "engram/controller.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engram {
namespace {

double stable_sigmoid(const double value) {
  const double exponential = std::exp(-std::abs(value));
  return value >= 0.0 ? 1.0 / (1.0 + exponential)
                      : exponential / (1.0 + exponential);
}

void validate(const ControllerWeightsView& weights, const double* state,
              const double* controller_input, const double* output) {
  if (weights.input_dimension == 0 || weights.state_dimension == 0 ||
      weights.stage_count == 0) {
    throw std::invalid_argument("controller dimensions must be positive");
  }
  if (weights.input_kernel == nullptr || weights.recurrent_kernel == nullptr ||
      weights.bias == nullptr || weights.stage_embeddings == nullptr ||
      state == nullptr || controller_input == nullptr || output == nullptr) {
    throw std::invalid_argument("controller tensor pointers must not be null");
  }
  if (weights.adapter_rank != 0 &&
      (weights.adapter_down == nullptr || weights.adapter_up == nullptr)) {
    throw std::invalid_argument(
        "nonzero adapter rank requires both adapter tensors");
  }
}

}  // namespace

ControllerWorkspace::ControllerWorkspace(const std::size_t state_dimension,
                                         const std::size_t adapter_rank) {
  prepare(state_dimension, adapter_rank);
}

void ControllerWorkspace::prepare(const std::size_t state_dimension,
                                  const std::size_t adapter_rank) {
  if (state_dimension == 0) {
    throw std::invalid_argument("workspace state dimension must be positive");
  }
  if (state_dimension_ == state_dimension && adapter_rank_ == adapter_rank) {
    return;
  }
  state_dimension_ = state_dimension;
  adapter_rank_ = adapter_rank;
  input_projection_.resize(3 * state_dimension);
  update_.resize(state_dimension);
  reset_.resize(state_dimension);
  candidate_.resize(state_dimension);
  adapter_hidden_.resize(adapter_rank);
  state_a_.resize(state_dimension);
  state_b_.resize(state_dimension);
}

std::size_t ControllerWorkspace::state_dimension() const noexcept {
  return state_dimension_;
}

std::size_t ControllerWorkspace::adapter_rank() const noexcept {
  return adapter_rank_;
}

std::size_t ControllerWorkspace::persistent_elements() const noexcept {
  return input_projection_.size() + update_.size() + reset_.size() +
         candidate_.size() + adapter_hidden_.size() + state_a_.size() +
         state_b_.size();
}

std::size_t ControllerWorkspace::persistent_bytes() const noexcept {
  return persistent_elements() * sizeof(double);
}

void controller_step_scalar(const ControllerWeightsView& weights,
                            const double* state,
                            const double* controller_input,
                            const std::size_t stage,
                            ControllerWorkspace& workspace, double* output) {
  validate(weights, state, controller_input, output);
  if (stage >= weights.stage_count) {
    throw std::out_of_range("controller stage is outside the stage table");
  }
  workspace.prepare(weights.state_dimension, weights.adapter_rank);
  const std::size_t input_dimension = weights.input_dimension;
  const std::size_t state_dimension = weights.state_dimension;
  const std::size_t kernel_width = 3 * state_dimension;

  std::fill(workspace.input_projection_.begin(),
            workspace.input_projection_.end(), 0.0);
  for (std::size_t input_index = 0; input_index < input_dimension;
       ++input_index) {
    const double input_value = controller_input[input_index];
    const double* kernel_row =
        weights.input_kernel + input_index * kernel_width;
    for (std::size_t column = 0; column < kernel_width; ++column) {
      workspace.input_projection_[column] += input_value * kernel_row[column];
    }
  }

  for (std::size_t column = 0; column < state_dimension; ++column) {
    double update_pre = workspace.input_projection_[column] +
                        weights.bias[column];
    double reset_pre = workspace.input_projection_[state_dimension + column] +
                       weights.bias[state_dimension + column];
    for (std::size_t row = 0; row < state_dimension; ++row) {
      const double* recurrent_row =
          weights.recurrent_kernel + row * kernel_width;
      update_pre += state[row] * recurrent_row[column];
      reset_pre += state[row] * recurrent_row[state_dimension + column];
    }
    workspace.update_[column] = stable_sigmoid(update_pre);
    workspace.reset_[column] = stable_sigmoid(reset_pre);
  }

  if (weights.adapter_rank != 0) {
    std::fill(workspace.adapter_hidden_.begin(),
              workspace.adapter_hidden_.end(), 0.0);
    const double* adapter_down =
        weights.adapter_down + stage * state_dimension * weights.adapter_rank;
    for (std::size_t row = 0; row < state_dimension; ++row) {
      for (std::size_t rank = 0; rank < weights.adapter_rank; ++rank) {
        workspace.adapter_hidden_[rank] +=
            state[row] * adapter_down[row * weights.adapter_rank + rank];
      }
    }
  }

  const double* stage_embedding =
      weights.stage_embeddings + stage * state_dimension;
  const double* adapter_up = weights.adapter_rank == 0
                                 ? nullptr
                                 : weights.adapter_up +
                                       stage * weights.adapter_rank *
                                           state_dimension;
  for (std::size_t column = 0; column < state_dimension; ++column) {
    double candidate_pre = workspace.input_projection_[2 * state_dimension +
                                                       column] +
                           weights.bias[2 * state_dimension + column] +
                           stage_embedding[column];
    for (std::size_t row = 0; row < state_dimension; ++row) {
      const double* recurrent_row =
          weights.recurrent_kernel + row * kernel_width;
      candidate_pre += workspace.reset_[row] * state[row] *
                       recurrent_row[2 * state_dimension + column];
    }
    for (std::size_t rank = 0; rank < weights.adapter_rank; ++rank) {
      candidate_pre += workspace.adapter_hidden_[rank] *
                       adapter_up[rank * state_dimension + column];
    }
    workspace.candidate_[column] = std::tanh(candidate_pre);
  }

  // All state-dependent intermediates have already been materialized, making
  // output == state safe without a temporary allocation.
  for (std::size_t column = 0; column < state_dimension; ++column) {
    output[column] = (1.0 - workspace.update_[column]) * state[column] +
                     workspace.update_[column] * workspace.candidate_[column];
  }
}

void controller_run_fixed_scalar(const ControllerWeightsView& weights,
                                 const double* initial_state,
                                 const double* controller_input,
                                 const std::size_t cycles,
                                 const std::size_t stage_offset,
                                 ControllerWorkspace& workspace,
                                 double* output) {
  validate(weights, initial_state, controller_input, output);
  if (cycles == 0) {
    throw std::invalid_argument("controller cycle count must be positive");
  }
  workspace.prepare(weights.state_dimension, weights.adapter_rank);
  std::copy(initial_state, initial_state + weights.state_dimension,
            workspace.state_a_.begin());
  double* current = workspace.state_a_.data();
  double* next = workspace.state_b_.data();
  for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
    const std::size_t stage = (stage_offset + cycle) % weights.stage_count;
    controller_step_scalar(weights, current, controller_input, stage, workspace,
                           next);
    std::swap(current, next);
  }
  std::copy(current, current + weights.state_dimension, output);
}

}  // namespace engram
