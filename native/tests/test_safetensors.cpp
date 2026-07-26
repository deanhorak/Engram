#include "engram/safetensors.h"
#include "engram/ternary_projection.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

int fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}

void write_u64(std::ofstream& output, std::uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    output.put(static_cast<char>((value >> (8 * byte)) & 0xFFU));
  }
}

}  // namespace

int main() {
  const auto path =
      std::filesystem::temp_directory_path() / "engram-safetensors-test.bin";
  const std::string header =
      R"({"embedding":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"codes":{"dtype":"U8","shape":[3],"data_offsets":[8,11]}})";
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    write_u64(output, header.size());
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    const std::uint16_t embedding[] = {0x3F80, 0x4000, 0x4040, 0x4080};
    output.write(reinterpret_cast<const char*>(embedding), sizeof(embedding));
    const std::uint8_t codes[] = {0, 1, 2};
    output.write(reinterpret_cast<const char*>(codes), sizeof(codes));
  }
  engram::SafetensorFile file = engram::load_safetensors(path);
  const auto embedding = file.tensor("embedding");
  const auto codes = file.tensor("codes");
  if (file.tensor_count() != 2 || embedding.shape != std::vector<std::size_t>{2, 2} ||
      embedding.bf16()[2] != 0x4040 || codes.uint8()[2] != 2) {
    std::filesystem::remove(path);
    return fail("safetensors mapped tensor mismatch");
  }
  engram::TernaryProjectionKernel projection(1);
  const std::size_t projection_index =
      projection.add_mapped(codes.uint8(), 3, 4, 1.0F);
  const std::uint16_t input[] = {0x3F80, 0x4000, 0x4040};
  std::uint16_t output[4] = {};
  projection.forward_bf16(projection_index, input, 1, output);
  if (output[0] != 0x4000 || output[1] != 0xC0C0 ||
      output[2] != 0xC0C0 || output[3] != 0xC0C0) {
    std::filesystem::remove(path);
    return fail("mapped ternary projection mismatch");
  }
  bool rejected_type = false;
  try {
    (void)codes.bf16();
  } catch (const engram::SafetensorError&) {
    rejected_type = true;
  }
  std::filesystem::remove(path);
  if (!rejected_type) return fail("safetensors accepted a wrong typed view");
  return 0;
}
