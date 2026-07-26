#include "engram/native_bitnet_token_runtime.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: engram-bitnet-token-generate PACKAGE MAX_NEW THREADS "
                 "[--verify-reset] TOKEN [TOKEN ...]\n";
    return 2;
  }
  try {
    const std::filesystem::path package(argv[1]);
    const std::size_t max_new = std::stoull(argv[2]);
    const std::size_t threads = std::stoull(argv[3]);
    int token_start = 4;
    const bool verify_reset =
        std::string(argv[token_start]) == "--verify-reset";
    if (verify_reset) ++token_start;
    if (token_start == argc) {
      throw std::invalid_argument("native generation prompt is empty");
    }
    std::vector<std::int64_t> prompt;
    for (int index = token_start; index < argc; ++index) {
      prompt.push_back(std::stoll(argv[index]));
    }
    // Version-1 native BitNet packages currently pin this architecture.
    engram::NativeBitNetTokenRuntime runtime(
        engram::NativeBitNetTokenConfig{
            .non_mlp_safetensors =
                package / "transformer/non_mlp.safetensors",
            .mlp_artifact = package / "mlp/model.bitnet-records.bin",
            .controller_directory = package / "controller",
            .layers = 30,
            .hidden_size = 2560,
            .query_heads = 20,
            .key_value_heads = 5,
            .head_dimension = 128,
            .threads = threads,
            .local_window = 16,
            .older_candidates = 8,
            .older_top_k = 4,
            .sink_tokens = 2,
            .rms_norm_epsilon = 1.0e-5F,
            .rope_theta = 500000.0F,
            .eos_token_ids = {128001},
        });
    const auto generated = runtime.generate(prompt, max_new);
    if (verify_reset) {
      runtime.reset();
      const auto repeated = runtime.generate(prompt, max_new);
      if (repeated != generated) {
        throw std::runtime_error("native reset generation is not deterministic");
      }
    }
    for (std::size_t index = 0; index < generated.size(); ++index) {
      if (index != 0) std::cout << ' ';
      std::cout << generated[index];
    }
    std::cout << '\n';
    const auto& metrics = runtime.metrics();
    std::cerr << "positions=" << metrics.positions_processed
              << " stage_calls=" << metrics.stage_calls
              << " semantic_seconds="
              << static_cast<double>(metrics.semantic_elapsed_ns) / 1.0e9
              << " attention_seconds="
              << static_cast<double>(metrics.attention_elapsed_ns) / 1.0e9
              << '\n';
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
