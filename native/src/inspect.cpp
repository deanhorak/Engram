#include "engram/package.h"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: engram-inspect MODEL\n";
      return 2;
    }
    const auto package = engram::load_package(argv[1], true);
    std::cout << "format=" << package.format << "\nversion=" << package.version
              << "\nsource_hash=" << package.source_model_hash
              << "\nhidden_size=" << package.dimensions.hidden_size
              << "\nvocab_size=" << package.dimensions.vocabulary_size
              << "\nsemantic_layers=" << package.dimensions.semantic_layers
              << "\nfiles=" << package.files.size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "engram-inspect: " << error.what() << '\n';
    return 1;
  }
}
