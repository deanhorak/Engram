#include "engram/runtime.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::cerr << "usage: engram-run MODEL [--prompt TEXT|--tokens FILE] [--max-tokens N] [--greedy] [--exact-vocab]\n";
      return 2;
    }
    std::string prompt;
    std::size_t max_tokens = 16;
    bool exact = false;
    bool bypass_cache = false;
    std::size_t threads = 1;
    std::vector<unsigned> affinity;
    std::string token_file;
    for (int index = 2; index < argc; ++index) {
      const std::string option = argv[index];
      if (option == "--prompt" && index + 1 < argc) prompt = argv[++index];
      else if (option == "--tokens" && index + 1 < argc) token_file = argv[++index];
      else if (option == "--max-tokens" && index + 1 < argc) max_tokens = std::stoull(argv[++index]);
      else if (option == "--exact-vocab") exact = true;
      else if (option == "--greedy") {}
      else if (option == "--no-cache") bypass_cache = true;
      else if (option == "--threads" && index + 1 < argc) threads = std::stoull(argv[++index]);
      else if (option == "--affinity" && index + 1 < argc) {
        std::stringstream values(argv[++index]);
        std::string value;
        while (std::getline(values, value, ',')) affinity.push_back(static_cast<unsigned>(std::stoul(value)));
      }
      else throw std::invalid_argument("unknown or incomplete option: " + option);
    }
    engram::NativeRuntime runtime(argv[1], true, threads, affinity);
    runtime.set_transition_cache_bypass(bypass_cache);
    std::vector<std::uint32_t> prompt_tokens;
    if (!token_file.empty()) {
      std::ifstream input(token_file, std::ios::binary);
      if (!input) throw std::runtime_error("cannot open token file");
      std::uint32_t token;
      while (input.read(reinterpret_cast<char*>(&token), sizeof(token))) prompt_tokens.push_back(token);
      if (!input.eof()) throw std::runtime_error("token file is not packed uint32 data");
    } else {
      prompt_tokens = runtime.tokenize_fixture(prompt);
    }
    const auto results = runtime.generate(prompt_tokens, max_tokens, exact);
    for (const auto& result : results) std::cout << '<' << result.token << "> ";
    std::cout << '\n';
    for (const auto& result : results) {
      std::cout << result.token << ',' << result.cycles << ','
                << result.semantic_records << ',' << result.semantic_candidates
                << ',' << result.semantic_proxy_records << ','
                << result.semantic_probed_clusters
                << ',' << result.episodic_retrievals << ','
                << result.vocabulary_candidates << ','
                << result.vocabulary_proxy_records << ','
                << result.vocabulary_probed_clusters << ','
                << result.semantic_bytes_read << ','
                << result.episodic_bytes_read << ','
                << result.vocabulary_bytes_read << ',' << result.elapsed_ns
                << ',' << (result.transition_cache_hit ? 1 : 0) << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "engram-run: " << error.what() << '\n';
    return 1;
  }
}
