#include "engram/runtime.h"

#include <chrono>
#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3) {
      std::cerr << "usage: engram-bench MODEL [TOKENS]\n";
      return 2;
    }
    const std::size_t count = argc == 3 ? std::stoull(argv[2]) : 128;
    engram::NativeRuntime runtime(argv[1]);
    runtime.set_transition_cache_bypass(true);
    const auto started = std::chrono::steady_clock::now();
    const auto result = runtime.generate({1, 2, 3}, count, false);
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
    double records = 0.0;
    double proxy_records = 0.0;
    double probed_clusters = 0.0;
    double vocabulary_proxy_records = 0.0;
    double vocabulary_probed_clusters = 0.0;
    for (const auto& token : result) {
      records += token.semantic_records;
      proxy_records += token.semantic_proxy_records;
      probed_clusters += token.semantic_probed_clusters;
      vocabulary_proxy_records += token.vocabulary_proxy_records;
      vocabulary_probed_clusters += token.vocabulary_probed_clusters;
    }
    std::cout << "{\"measured_elapsed_ns\":" << elapsed
              << ",\"tokens\":" << count
              << ",\"decode_tokens_per_second\":"
              << (static_cast<double>(count) * 1.0e9 / elapsed)
              << ",\"mean_active_semantic_records\":"
              << records / static_cast<double>(count)
              << ",\"mean_semantic_proxy_records\":"
              << proxy_records / static_cast<double>(count)
              << ",\"mean_semantic_probed_clusters\":"
              << probed_clusters / static_cast<double>(count)
              << ",\"mean_vocabulary_proxy_records\":"
              << vocabulary_proxy_records / static_cast<double>(count)
              << ",\"mean_vocabulary_probed_clusters\":"
              << vocabulary_probed_clusters / static_cast<double>(count)
              << ",\"quality_claim\":null}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "engram-bench: " << error.what() << '\n';
    return 1;
  }
}
