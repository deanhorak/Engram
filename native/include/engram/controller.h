#pragma once

#include <cstddef>
#include <vector>

namespace engram {

// Non-owning row-major views of the tensors emitted by SharedRecurrentController.
// Kernel columns are concatenated as [update, reset, candidate].
struct ControllerWeightsView {
  std::size_t input_dimension{};
  std::size_t state_dimension{};
  std::size_t stage_count{};
  std::size_t adapter_rank{};
  const double* input_kernel{};       // [input_dimension, 3 * state_dimension]
  const double* recurrent_kernel{};   // [state_dimension, 3 * state_dimension]
  const double* bias{};               // [3 * state_dimension]
  const double* stage_embeddings{};   // [stage_count, state_dimension]
  const double* adapter_down{};       // [stage_count, state_dimension, adapter_rank]
  const double* adapter_up{};         // [stage_count, adapter_rank, state_dimension]
};

// Scratch storage shared by incremental and fixed-cycle execution. prepare()
// only changes allocations when dimensions change; a steady-state step performs
// no heap allocation.
class ControllerWorkspace {
 public:
  ControllerWorkspace() = default;
  ControllerWorkspace(std::size_t state_dimension,
                      std::size_t adapter_rank);

  void prepare(std::size_t state_dimension, std::size_t adapter_rank);
  [[nodiscard]] std::size_t state_dimension() const noexcept;
  [[nodiscard]] std::size_t adapter_rank() const noexcept;
  [[nodiscard]] std::size_t persistent_elements() const noexcept;
  [[nodiscard]] std::size_t persistent_bytes() const noexcept;

 private:
  std::size_t state_dimension_{};
  std::size_t adapter_rank_{};
  std::vector<double> input_projection_;
  std::vector<double> update_;
  std::vector<double> reset_;
  std::vector<double> candidate_;
  std::vector<double> adapter_hidden_;
  std::vector<double> state_a_;
  std::vector<double> state_b_;

  friend void controller_step_scalar(const ControllerWeightsView&,
                                     const double*, const double*,
                                     std::size_t, ControllerWorkspace&,
                                     double*);
  friend void controller_run_fixed_scalar(const ControllerWeightsView&,
                                          const double*, const double*,
                                          std::size_t, std::size_t,
                                          ControllerWorkspace&, double*);
};

// One shared-GRU transition. output may alias state.
void controller_step_scalar(const ControllerWeightsView& weights,
                            const double* state,
                            const double* controller_input,
                            std::size_t stage,
                            ControllerWorkspace& workspace, double* output);

// Python fixed-mode parity helper. Stages wrap modulo stage_count.
void controller_run_fixed_scalar(const ControllerWeightsView& weights,
                                 const double* initial_state,
                                 const double* controller_input,
                                 std::size_t cycles,
                                 std::size_t stage_offset,
                                 ControllerWorkspace& workspace,
                                 double* output);

}  // namespace engram
