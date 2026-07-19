#include "engram/episodic.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

constexpr float kQueries[][2] = {
    {1.0F, 0.0F}, {0.0F, 1.0F}, {1.0F, 0.2F},
    {0.1F, 1.0F}, {1.0F, -0.4F}, {0.7F, 0.6F},
};
constexpr float kKeys[][2] = {
    {1.0F, 0.0F}, {0.0F, 1.0F}, {0.8F, 0.1F},
    {-0.2F, 1.0F}, {1.0F, -0.5F}, {0.6F, 0.7F},
};
constexpr float kValues[][2] = {
    {1.0F, 2.0F}, {3.0F, 4.0F}, {5.0F, -1.0F},
    {2.0F, 6.0F}, {-2.0F, 3.0F}, {4.0F, -3.0F},
};
constexpr float kPythonOutputs[][2] = {
    {1.0F, 2.0F},
    {2.3395228385925293F, 3.3395230770111084F},
    {2.931030035018921F, 1.3763624429702759F},
    {2.8915791511535645F, 3.519756317138672F},
    {0.18058595061302185F, 2.8404452800750732F},
    {2.449498414993286F, 0.11020729690790176F},
};

bool close(const float actual, const float expected,
           const float tolerance = 3e-5F) {
  return std::abs(actual - expected) <= tolerance;
}

int fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}

}  // namespace

int main() {
  const engram::EpisodicConfig config{
      .key_width = 2,
      .value_width = 2,
      .local_window = 2,
      .retrieval_capacity = 3,
      .retrieval_candidates = 2,
      .retrieval_top_k = 1,
      .decay = 0.8,
      .older_weight = 0.4,
      .epsilon = 1e-6,
  };
  engram::HybridEpisodicMemory memory(config);
  const std::size_t state_bytes = memory.allocated_state_bytes();
  const std::size_t scratch_bytes = memory.scratch_bytes();
  if (state_bytes != 156 || scratch_bytes == 0) {
    return fail("episodic allocation metrics mismatch");
  }

  for (std::size_t step = 0; step < 6; ++step) {
    float output[2]{};
    const engram::EpisodicStepMetrics metrics = memory.step(
        kQueries[step], kKeys[step], kValues[step], output);
    if (!close(output[0], kPythonOutputs[step][0]) ||
        !close(output[1], kPythonOutputs[step][1])) {
      return fail("native hybrid output differs from Python reference");
    }
    if (metrics.local_tokens != std::min(step + 1, std::size_t{2}) ||
        metrics.older_tokens !=
            std::min(step > 1 ? step - 1 : std::size_t{0}, std::size_t{3}) ||
        metrics.tokens_seen != step + 1 || metrics.state_bytes != state_bytes ||
        metrics.scratch_bytes != scratch_bytes) {
      return fail("episodic per-step metrics mismatch");
    }
    if (step < 2 &&
        (metrics.retrievals != 0 || metrics.bytes_read != 0 ||
         metrics.recurrent_steps != 0)) {
      return fail("older memory was read before local eviction");
    }
  }
  if (memory.local_count() != 2 || memory.older_count() != 3 ||
      memory.tokens_seen() != 6 || memory.allocated_state_bytes() != state_bytes ||
      memory.scratch_bytes() != scratch_bytes) {
    return fail("episodic memory did not remain bounded");
  }
  const std::span<const std::uint64_t> positions =
      memory.last_retrieved_positions();
  if (positions.size() != 1 || positions[0] != 2) {
    return fail("quantized retrieval selected the wrong older position");
  }

  memory.reset();
  if (memory.local_count() != 0 || memory.older_count() != 0 ||
      memory.tokens_seen() != 0 ||
      memory.allocated_state_bytes() != state_bytes ||
      memory.scratch_bytes() != scratch_bytes) {
    return fail("episodic reset changed capacity or retained active tokens");
  }
  float reset_output[2]{};
  const auto reset_metrics =
      memory.step(kQueries[0], kKeys[0], kValues[0], reset_output);
  if (!close(reset_output[0], 1.0F) || !close(reset_output[1], 2.0F) ||
      reset_metrics.local_tokens != 1) {
    return fail("episodic reuse after reset mismatch");
  }

  bool rejected = false;
  try {
    const engram::EpisodicConfig invalid{
        .key_width = 2,
        .value_width = 2,
        .local_window = 1,
        .retrieval_capacity = 2,
        .retrieval_candidates = 1,
        .retrieval_top_k = 2,
    };
    const engram::HybridEpisodicMemory unused(invalid);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) {
    return fail("invalid episodic retrieval configuration was accepted");
  }
  return 0;
}
