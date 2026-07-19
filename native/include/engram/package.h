#pragma once

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace engram {

class PackageError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct PackageFile {
  std::string relative_path;
  std::size_t bytes = 0;
  std::string sha256;
};

struct RuntimeDimensions {
  std::size_t hidden_size = 0;
  std::size_t vocabulary_size = 0;
  std::size_t semantic_layers = 0;
  std::size_t controller_input_size = 0;
  std::size_t controller_stages = 0;
  std::size_t controller_adapter_rank = 0;
};

struct RuntimePolicies {
  std::size_t cycles = 0;
  std::size_t semantic_top_k = 0;
  std::size_t semantic_candidates = 0;
  std::size_t semantic_ivf_clusters = 0;
  std::size_t semantic_ivf_probes = 0;
  std::size_t vocabulary_candidates = 0;
  std::size_t vocabulary_ivf_clusters = 0;
  std::size_t vocabulary_ivf_probes = 0;
  std::size_t local_window = 0;
  std::size_t retrieval_capacity = 0;
  std::size_t retrieval_candidates = 0;
  std::size_t retrieval_top_k = 0;
  double recurrent_decay = 0.0;
  std::size_t transition_capacity = 0;
  double transition_similarity_radius = 0.0;
  std::string correction_fallback;
};

struct PackageMetadata {
  std::filesystem::path root;
  std::string format;
  std::size_t version = 0;
  std::string engram_version;
  std::string source_model_hash;
  std::string source_architecture;
  bool fixture_only = false;
  bool source_transformer_independent = false;
  RuntimeDimensions dimensions;
  RuntimePolicies policies;
  std::vector<PackageFile> files;
};

// Compute a lowercase SHA-256 digest without external dependencies.
[[nodiscard]] std::string sha256_file(const std::filesystem::path& path);

// Parse and validate a compiled package. When verify_checksums is false, file
// presence and declared byte sizes are still checked.
[[nodiscard]] PackageMetadata load_package(
    const std::filesystem::path& root, bool verify_checksums = true);

}  // namespace engram
