#include "engram/native_bitnet_weights.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: engram-bitnet-weights-info FILE LAYERS HIDDEN "
                 "QUERY_HEADS KV_HEADS HEAD_DIM THREADS\n";
    return 2;
  }
  try {
    engram::NativeBitNetWeights weights(
        std::filesystem::path(argv[1]), std::stoull(argv[2]),
        std::stoull(argv[3]), std::stoull(argv[4]), std::stoull(argv[5]),
        std::stoull(argv[6]), std::stoull(argv[7]));
    std::cout << "layers=" << weights.layers().size()
              << " vocabulary=" << weights.vocabulary_size()
              << " hidden=" << weights.hidden_size()
              << " mapped_bytes=" << weights.mapped_bytes()
              << " copied_projection_bytes=0\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
