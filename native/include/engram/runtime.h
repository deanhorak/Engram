#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engram {

struct NativeTokenMetrics {
  std::uint32_t token{};
  std::size_t cycles{};
  std::size_t semantic_records{};
  std::size_t semantic_candidates{};
  std::size_t semantic_proxy_records{};
  std::size_t semantic_probed_clusters{};
  std::size_t episodic_retrievals{};
  std::size_t vocabulary_candidates{};
  std::size_t vocabulary_proxy_records{};
  std::size_t vocabulary_probed_clusters{};
  std::size_t semantic_bytes_read{};
  std::size_t episodic_bytes_read{};
  std::size_t vocabulary_bytes_read{};
  std::uint64_t elapsed_ns{};
  bool transition_cache_hit{};
};

class NativeRuntime {
 public:
  explicit NativeRuntime(const std::filesystem::path& package,
                         bool verify_checksums = true,
                         std::size_t thread_count = 1,
                         std::vector<unsigned> affinity = {});
  ~NativeRuntime();
  NativeRuntime(NativeRuntime&&) noexcept;
  NativeRuntime& operator=(NativeRuntime&&) noexcept;
  NativeRuntime(const NativeRuntime&) = delete;
  NativeRuntime& operator=(const NativeRuntime&) = delete;

  void reset();
  void set_transition_cache_bypass(bool enabled);
  [[nodiscard]] std::vector<std::uint32_t> tokenize_fixture(
      const std::string& prompt) const;
  [[nodiscard]] NativeTokenMetrics step(std::uint32_t input_token,
                                        bool exact_vocabulary = false);
  [[nodiscard]] std::vector<NativeTokenMetrics> generate(
      const std::vector<std::uint32_t>& prompt_tokens, std::size_t max_tokens,
      bool exact_vocabulary = false);

 private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

}  // namespace engram
