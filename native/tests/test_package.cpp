#include "engram/package.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

void write(const std::filesystem::path& path, const std::string& content) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary);
  stream << content;
  if (!stream) {
    throw std::runtime_error("test fixture write failed");
  }
}

std::string quote(const std::string& value) { return "\"" + value + "\""; }

int fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}

}  // namespace

int main() {
  const auto root = std::filesystem::temp_directory_path() /
                    "engram-native-package-test";
  std::filesystem::remove_all(root);
  try {
    const std::map<std::string, std::string> contents = {
        {"controller/metadata.json",
         R"({"format":"engram.controller.shared_gru","schema_version":1,"operator":"shared_gru_stage_adapter","input_dim":12,"state_dim":4,"num_stages":2,"adapter_rank":1,"storage_dtype":"float64","tensor_layout":{"input_kernel":[12,12],"recurrent_kernel":[4,12],"bias":[12],"stage_embeddings":[2,4],"adapter_down":[2,4,1],"adapter_up":[2,1,4]}})"},
        {"controller/input_kernel.npy", "input"},
        {"controller/recurrent_kernel.npy", "recurrent"},
        {"controller/bias.npy", "bias"},
        {"controller/stage_embeddings.npy", "stages"},
        {"controller/adapter_down.npy", "down"},
        {"controller/adapter_up.npy", "up"},
        {"embeddings/token_embeddings.npy", "tokens"},
        {"vocabulary/embeddings.npy", "vocabulary"},
        {"vocabulary/index.npy", "normalized"},
        {"vocabulary/ivf/centroids.npy", "vocab-centroids"},
        {"vocabulary/ivf/posting_offsets.npy", "vocab-offsets"},
        {"vocabulary/ivf/token_ids.npy", "vocab-tokens"},
        {"vocabulary/ivf/metadata.json", "{}"},
        {"semantic/manifest.json", R"({"format":"engram-semantic-memory"})"},
        {"semantic/layer-0000/quantized/ivf/centroids.npy", "centroids0"},
        {"semantic/layer-0000/quantized/ivf/posting_offsets.npy", "offsets0"},
        {"semantic/layer-0000/quantized/ivf/posting_indices.npy", "postings0"},
        {"semantic/layer-0000/quantized/ivf/metadata.json", "{}"},
        {"semantic/layer-0001/quantized/ivf/centroids.npy", "centroids1"},
        {"semantic/layer-0001/quantized/ivf/posting_offsets.npy", "offsets1"},
        {"semantic/layer-0001/quantized/ivf/posting_indices.npy", "postings1"},
        {"semantic/layer-0001/quantized/ivf/metadata.json", "{}"},
        {"episodic/config.json",
         R"({"local_window":16,"retrieval_capacity":1024,"retrieval_candidates":16,"retrieval_top_k":4,"decay":9.9e-1})"},
        {"transitions/config.json",
         R"({"capacity":4096,"similarity_radius":0.02})"},
        {"corrections/capsules.json",
         R"({"version":1,"capsules":[],"fallback":"extra_cycle_\u0026_search"})"},
    };
    for (const auto& [relative, content] : contents) {
      write(root / relative, content);
    }

    const auto abc = root / "abc.bin";
    write(abc, "abc");
    if (engram::sha256_file(abc) !=
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
      return fail("SHA-256 known vector mismatch");
    }
    std::filesystem::remove(abc);

    std::string files_json;
    for (const auto& [relative, content] : contents) {
      if (!files_json.empty()) {
        files_json += ',';
      }
      files_json += quote(relative) + ":{\"bytes\":" +
                    std::to_string(content.size()) + ",\"sha256\":" +
                    quote(engram::sha256_file(root / relative)) + "}";
    }
    const std::string manifest =
        R"({"format":"engram-model","version":1,"engram_version":"0.1.0","source_model_hash":"fixture-hash","source_architecture":"llama","fixture_only":true,"hidden_size":4,"vocab_size":32,"num_semantic_layers":2,"runtime":{"cycles":3,"semantic_top_k":4,"semantic_candidates":8,"semantic_ivf_clusters":4,"semantic_ivf_probes":2,"vocabulary_candidates":16,"vocabulary_ivf_clusters":8,"vocabulary_ivf_probes":2},"files":{)" +
        files_json + R"(},"does_not_require_source_transformer":true})";
    write(root / "manifest.json", manifest);

    const engram::PackageMetadata package = engram::load_package(root);
    if (package.format != "engram-model" || !package.fixture_only ||
        !package.source_transformer_independent ||
        package.dimensions.hidden_size != 4 ||
        package.dimensions.vocabulary_size != 32 ||
        package.dimensions.controller_input_size != 12 ||
        package.policies.cycles != 3 ||
        package.policies.retrieval_top_k != 4 ||
        package.policies.transition_capacity != 4096 ||
        package.policies.correction_fallback != "extra_cycle_&_search" ||
        package.files.size() != contents.size()) {
      std::cerr << package.format << ' ' << package.fixture_only << ' '
                << package.source_transformer_independent << ' '
                << package.dimensions.hidden_size << ' '
                << package.dimensions.vocabulary_size << ' '
                << package.dimensions.controller_input_size << ' '
                << package.policies.cycles << ' '
                << package.policies.retrieval_top_k << ' '
                << package.policies.transition_capacity << ' '
                << package.policies.correction_fallback << ' '
                << package.files.size() << '/' << contents.size() << '\n';
      return fail("parsed package metadata mismatch");
    }

    write(root / "controller/bias.npy", "tampered");
    try {
      static_cast<void>(engram::load_package(root));
      return fail("corrupt package file was accepted");
    } catch (const engram::PackageError&) {
    }
    // Even without hashing, declared size remains mandatory.
    try {
      static_cast<void>(engram::load_package(root, false));
      return fail("wrong declared file size was accepted without checksums");
    } catch (const engram::PackageError&) {
    }

    std::filesystem::remove_all(root);
  } catch (const std::exception& error) {
    std::filesystem::remove_all(root);
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
