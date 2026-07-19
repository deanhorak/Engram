#include "engram/transition_cache.h"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::array<engram::TransitionCandidate, 2> kCandidates = {
    engram::TransitionCandidate{7, 2.5F},
    engram::TransitionCandidate{8, 1.0F}};

int fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}

bool close(const double left, const double right) {
  return std::abs(left - right) < 1e-6;
}

}  // namespace

int main() {
  {
    engram::TransitionCache cache(4, 4, 1.0F, 1.0F, 2);
    const std::array<float, 4> state = {0.5F, 1.5F, 2.5F, -1.5F};
    const auto fingerprint = cache.fingerprint(state, 9);
    const std::array<std::int16_t, 4> expected = {0, 2, 2, -2};
    if (fingerprint.input_token != 9 ||
        !std::equal(fingerprint.codes.begin(), fingerprint.codes.end(),
                    expected.begin()) ||
        fingerprint == cache.fingerprint(state, 10)) {
      return fail("deterministic ties-to-even fingerprint mismatch");
    }
  }

  {
    engram::TransitionCache cache(2, 4, 0.5F, 0.5F, 2);
    const std::array<float, 2> state = {0.1F, -0.1F};
    const std::array<float, 2> next = {1.0F, 2.0F};
    if (!cache.put_offline(state, 3, next, kCandidates, 0.9F)) {
      return fail("offline cache population failed");
    }
    const std::array<float, 2> nearby = {0.11F, -0.1F};
    const auto hit = cache.lookup(nearby, 3);
    if (!hit.hit || hit.reason != engram::TransitionLookupReason::kHit ||
        hit.transition.output_candidates[0].token != 7 ||
        hit.transition.next_state[1] != 2.0F ||
        cache.lookup(nearby, 4).hit) {
      return fail("validated transition lookup mismatch");
    }
  }

  {
    engram::TransitionCache cache(2, 2, 10.0F, 0.1F, 2);
    const std::array<float, 2> stored = {1.0F, 0.0F};
    const std::array<float, 2> query = {2.0F, 0.0F};
    const std::array<float, 2> next = {0.0F, 1.0F};
    cache.put_online(stored, 5, next, kCandidates, 1.0F);
    const auto rejected = cache.lookup(query, 5);
    const auto metrics = cache.metrics();
    if (rejected.hit ||
        rejected.reason != engram::TransitionLookupReason::kRadiusRejection ||
        !close(rejected.state_distance, 0.5) ||
        metrics.radius_rejections != 1 || metrics.collisions != 1) {
      return fail("fingerprint collision radius rejection mismatch");
    }
  }

  {
    engram::TransitionCache cache(2, 2, 0.1F, 0.01F, 2);
    const std::array<float, 2> a = {1.0F, 0.0F};
    const std::array<float, 2> b = {0.0F, 1.0F};
    const std::array<float, 2> c = {-1.0F, 0.0F};
    const std::array<float, 2> next_a = {2.0F, 1.0F};
    const std::array<float, 2> next_b = {1.0F, 2.0F};
    const std::array<float, 2> next_c = {0.0F, 1.0F};
    cache.put_offline(a, 1, next_a, kCandidates, 0.8F);
    cache.put_online(b, 2, next_b, kCandidates, 0.8F);
    static_cast<void>(cache.lookup(a, 1));  // Refresh A.
    cache.put_online(c, 3, next_c, kCandidates, 0.8F);
    if (cache.lookup(b, 2).hit) {
      return fail("LRU did not evict the oldest entry");
    }
    const std::array<float, 2> actual = {2.1F, 1.0F};
    if (!cache.lookup(a, 1, actual).hit) {
      return fail("LRU evicted refreshed entry");
    }
    const auto metrics = cache.metrics();
    if (metrics.entries != 2 || metrics.evictions != 1 ||
        metrics.offline_puts != 1 || metrics.online_puts != 2 ||
        metrics.approximation_error_samples != 1 ||
        metrics.mean_approximation_error <= 0.0 ||
        !close(metrics.hit_rate, 2.0 / 3.0)) {
      return fail("LRU or transition metrics mismatch");
    }
  }

  {
    engram::TransitionCache cache(2, 2, 0.1F, 0.1F, 2, true);
    const std::array<float, 2> state = {1.0F, 0.0F};
    const std::array<float, 2> next = {0.0F, 1.0F};
    if (cache.put_online(state, 1, next, kCandidates, 1.0F)) {
      return fail("bypass accepted cache population");
    }
    const auto result = cache.lookup(state, 1);
    const auto metrics = cache.metrics();
    if (result.hit || result.reason != engram::TransitionLookupReason::kBypass ||
        metrics.entries != 0 || metrics.bypassed_puts != 1 ||
        metrics.bypassed_lookups != 1) {
      return fail("transition bypass metrics mismatch");
    }
  }

  try {
    engram::TransitionCache cache(2, 2, 0.1F, 0.1F, 1);
    const std::array<float, 2> state = {1.0F, 0.0F};
    cache.put_online(state, 1, state, kCandidates, 1.0F);
    return fail("oversized output candidate list was accepted");
  } catch (const std::invalid_argument&) {
  }

  return 0;
}
