#include "engram/safetensors.h"

#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: engram-safetensors-info FILE [TENSOR]\n";
    return 2;
  }
  try {
    const engram::SafetensorFile file =
        engram::load_safetensors(std::filesystem::path(argv[1]));
    std::cout << "tensors=" << file.tensor_count()
              << " mapped_bytes=" << file.mapped_bytes() << '\n';
    if (argc == 3) {
      const engram::SafetensorView tensor = file.tensor(std::string(argv[2]));
      std::cout << "tensor=" << argv[2] << " shape=";
      for (std::size_t index = 0; index < tensor.shape.size(); ++index) {
        if (index != 0) std::cout << 'x';
        std::cout << tensor.shape[index];
      }
      std::cout << " bytes=" << tensor.bytes.size() << '\n';
    }
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
