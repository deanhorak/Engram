#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace engram {

enum class TransitionSource { kOffline, kOnline };

enum class TransitionLookupReason {
  kHit,
  kMiss,
  kBypass,
  kRadiusRejection,
};

struct TransitionCandidate {
  std::uint32_t token = 0;
  float score = 0.0F;
};

struct StateFingerprint {
  std::uint32_t input_token = 0;
  std::vector<std::int16_t> codes;

  bool operator==(const StateFingerprint&) const = default;
};

struct CachedTransitionView {
  std::span<const float> next_state;
  std::span<const TransitionCandidate> output_candidates;
  float confidence = 0.0F;
};

struct TransitionLookup {
  bool hit = false;
  TransitionLookupReason reason = TransitionLookupReason::kMiss;
  CachedTransitionView transition;
  double state_distance = 0.0;
};

struct TransitionCacheMetrics {
  std::size_t entries = 0;
  std::size_t capacity = 0;
  std::uint64_t lookups = 0;
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  double hit_rate = 0.0;
  std::uint64_t radius_rejections = 0;
  std::uint64_t collisions = 0;
  std::uint64_t evictions = 0;
  std::uint64_t offline_puts = 0;
  std::uint64_t online_puts = 0;
  std::uint64_t bypassed_lookups = 0;
  std::uint64_t bypassed_puts = 0;
  std::uint64_t approximation_error_samples = 0;
  double mean_approximation_error = 0.0;
  double max_approximation_error = 0.0;
};

// Fixed-capacity, non-thread-safe transition LRU. Returned views remain valid
// until the next non-const cache operation.
class TransitionCache {
 public:
  TransitionCache(std::size_t state_width, std::size_t capacity = 1024,
                  float quantization_step = 0.125F,
                  float similarity_radius = 0.05F,
                  std::size_t max_output_candidates = 64,
                  bool bypass = false);
  ~TransitionCache();

  TransitionCache(TransitionCache&&) noexcept;
  TransitionCache& operator=(TransitionCache&&) noexcept;
  TransitionCache(const TransitionCache&) = delete;
  TransitionCache& operator=(const TransitionCache&) = delete;

  [[nodiscard]] StateFingerprint fingerprint(
      std::span<const float> state, std::uint32_t input_token) const;

  bool put(std::span<const float> state, std::uint32_t input_token,
           std::span<const float> next_state,
           std::span<const TransitionCandidate> output_candidates,
           float confidence, TransitionSource source = TransitionSource::kOnline);

  bool put_offline(
      std::span<const float> state, std::uint32_t input_token,
      std::span<const float> next_state,
      std::span<const TransitionCandidate> output_candidates, float confidence);

  bool put_online(
      std::span<const float> state, std::uint32_t input_token,
      std::span<const float> next_state,
      std::span<const TransitionCandidate> output_candidates, float confidence);

  [[nodiscard]] TransitionLookup lookup(
      std::span<const float> state, std::uint32_t input_token,
      std::span<const float> actual_next_state = {});

  void set_bypass(bool enabled) noexcept;
  [[nodiscard]] bool bypass() const noexcept;
  void clear() noexcept;
  [[nodiscard]] TransitionCacheMetrics metrics() const noexcept;
  [[nodiscard]] std::size_t state_width() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace engram
