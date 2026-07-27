#pragma once

#include <cstddef>
#include <cstdint>
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

// Fully authenticated inputs for the CPU-only native BitNet DIP token
// runtime. Paths are resolved below root only after the package inventory and
// all nested descriptors have been validated.
struct NativeBitNetDIPPackageMetadata {
  std::filesystem::path root;
  std::string manifest_sha256;
  std::filesystem::path non_mlp_safetensors;
  std::filesystem::path mlp_artifact;
  std::filesystem::path dip_coordinate_index;
  std::filesystem::path controller_directory;
  std::size_t layers = 0;
  std::size_t hidden_size = 0;
  std::size_t intermediate_size = 0;
  std::size_t vocabulary_size = 0;
  std::size_t query_heads = 0;
  std::size_t key_value_heads = 0;
  std::size_t head_dimension = 0;
  std::size_t max_position_embeddings = 0;
  std::size_t kernel_threads = 0;
  std::size_t local_window = 0;
  std::size_t older_candidates = 0;
  std::size_t older_top_k = 0;
  std::size_t sink_tokens = 0;
  float rms_norm_epsilon = 0.0F;
  float rope_theta = 0.0F;
  std::vector<std::int64_t> eos_token_ids;
  std::vector<PackageFile> files;
};

// The local package manifest is not a signature. A native deployment must
// anchor promotion to externally reviewed immutable digests instead of
// trusting hashes that a rewritten manifest can self-assert.
struct NativeBitNetDIPTrustRoot {
  std::size_t package_manifest_bytes = 0;
  std::string package_manifest_sha256;
  std::string source_package_manifest_sha256;
  std::string source_artifact_sha256;
  std::string coordinate_index_sha256;
  std::string policy_manifest_sha256;
  std::string adjudication_sha256;
};

// Compute a lowercase SHA-256 digest without external dependencies.
[[nodiscard]] std::string sha256_file(const std::filesystem::path& path);

// Parse and validate a compiled package. When verify_checksums is false, file
// presence and declared byte sizes are still checked.
[[nodiscard]] PackageMetadata load_package(
    const std::filesystem::path& root, bool verify_checksums = true);

// Load the version-1 native BitNet package promoted to the adjudicated DIP-v2
// semantic backend. This is intentionally fail-closed: the inventory must be
// exact, every package file is hashed, no symlink is accepted, and all model,
// controller, runtime, attention, and EOS settings consumed by the native
// token runtime are authenticated and cross-checked.
[[nodiscard]] NativeBitNetDIPPackageMetadata load_native_bitnet_dip_package(
    const std::filesystem::path& root);

// Explicit trust-root overload for reproducible native tests and future
// separately adjudicated model tracks. Production callers should use the
// overload above, which is pinned to Engram's promoted BitNet package.
[[nodiscard]] NativeBitNetDIPPackageMetadata load_native_bitnet_dip_package(
    const std::filesystem::path& root,
    const NativeBitNetDIPTrustRoot& trust_root);

}  // namespace engram
