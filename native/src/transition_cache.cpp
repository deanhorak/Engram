#include "engram/transition_cache.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engram {
namespace {

constexpr std::size_t kEmpty = std::numeric_limits<std::size_t>::max();
constexpr std::size_t kTombstone = kEmpty - 1;

std::int16_t quantize_scalar(const float value, const double step) noexcept {
  const double scaled = static_cast<double>(value) / step;
  constexpr double lower =
      static_cast<double>(std::numeric_limits<std::int16_t>::min());
  constexpr double upper =
      static_cast<double>(std::numeric_limits<std::int16_t>::max());
  if (scaled <= lower) {
    return std::numeric_limits<std::int16_t>::min();
  }
  if (scaled >= upper) {
    return std::numeric_limits<std::int16_t>::max();
  }
  const double floor_value = std::floor(scaled);
  const double fraction = scaled - floor_value;
  double rounded = floor_value;
  if (fraction > 0.5 ||
      (fraction == 0.5 &&
       std::fmod(std::abs(floor_value), 2.0) == 1.0)) {
    rounded += 1.0;
  }
  return static_cast<std::int16_t>(rounded);
}

void validate_state(const std::span<const float> state,
                    const std::size_t width, const char* name) {
  if (state.size() != width) {
    throw std::invalid_argument(std::string(name) +
                                " width does not match transition cache");
  }
  if (!std::all_of(state.begin(), state.end(),
                   [](const float value) { return std::isfinite(value); })) {
    throw std::invalid_argument(std::string(name) +
                                " must contain finite values");
  }
}

double relative_distance(const std::span<const float> left,
                         const std::span<const float> right) noexcept {
  double squared_error = 0.0;
  double squared_left = 0.0;
  double squared_right = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const double left_value = left[index];
    const double right_value = right[index];
    const double difference = left_value - right_value;
    squared_error += difference * difference;
    squared_left += left_value * left_value;
    squared_right += right_value * right_value;
  }
  const double denominator =
      std::max({std::sqrt(squared_left), std::sqrt(squared_right), 1e-12});
  return std::sqrt(squared_error) / denominator;
}

std::size_t table_capacity(const std::size_t capacity) {
  if (capacity > std::numeric_limits<std::size_t>::max() / 2) {
    throw std::invalid_argument("transition cache capacity is too large");
  }
  const std::size_t requested = capacity * 2;
  std::size_t result = 1;
  while (result < requested) {
    if (result > std::numeric_limits<std::size_t>::max() / 2) {
      throw std::invalid_argument("transition cache capacity is too large");
    }
    result *= 2;
  }
  return result;
}

std::uint64_t fingerprint_hash(
    const std::uint32_t token,
    const std::span<const std::int16_t> codes) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto mix = [&hash](const std::uint8_t byte) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  };
  for (unsigned shift = 0; shift < 32; shift += 8) {
    mix(static_cast<std::uint8_t>(token >> shift));
  }
  for (const std::int16_t code : codes) {
    const std::uint16_t bits = static_cast<std::uint16_t>(code);
    mix(static_cast<std::uint8_t>(bits));
    mix(static_cast<std::uint8_t>(bits >> 8U));
  }
  return hash;
}

}  // namespace

struct TransitionCache::Impl {
  struct Entry {
    bool occupied = false;
    std::uint32_t input_token = 0;
    std::uint64_t hash = 0;
    std::vector<std::int16_t> codes;
    std::vector<float> reference_state;
    std::vector<float> next_state;
    std::vector<TransitionCandidate> candidates;
    float confidence = 0.0F;
    TransitionSource source = TransitionSource::kOnline;
    std::size_t previous = kEmpty;
    std::size_t next = kEmpty;
  };

  Impl(const std::size_t width, const std::size_t requested_capacity,
       const float step, const float radius,
       const std::size_t candidate_capacity, const bool bypass_enabled)
      : state_width(width),
        capacity(requested_capacity),
        quantization_step(step),
        similarity_radius(radius),
        max_output_candidates(candidate_capacity),
        bypass(bypass_enabled),
        entries(requested_capacity),
        table(table_capacity(requested_capacity), kEmpty),
        scratch_codes(width),
        table_mask(table.size() - 1) {
    if (state_width == 0 || capacity == 0 || max_output_candidates == 0) {
      throw std::invalid_argument(
          "state width, capacity, and output candidate capacity must be positive");
    }
    if (!std::isfinite(quantization_step) || quantization_step <= 0.0F) {
      throw std::invalid_argument(
          "transition quantization step must be finite and positive");
    }
    if (!std::isfinite(similarity_radius) || similarity_radius < 0.0F) {
      throw std::invalid_argument(
          "transition similarity radius must be finite and non-negative");
    }
    free_slots.reserve(capacity);
    for (std::size_t index = capacity; index-- > 0;) {
      free_slots.push_back(index);
    }
    for (Entry& entry : entries) {
      entry.codes.resize(state_width);
      entry.reference_state.resize(state_width);
      entry.next_state.resize(state_width);
      entry.candidates.reserve(max_output_candidates);
    }
  }

  void quantize(const std::span<const float> state) noexcept {
    for (std::size_t index = 0; index < state_width; ++index) {
      scratch_codes[index] =
          quantize_scalar(state[index], quantization_step);
    }
  }

  bool matches(const Entry& entry, const std::uint32_t token,
               const std::span<const std::int16_t> codes) const noexcept {
    return entry.occupied && entry.input_token == token &&
           std::equal(entry.codes.begin(), entry.codes.end(), codes.begin());
  }

  std::size_t find(const std::uint32_t token,
                   const std::span<const std::int16_t> codes,
                   const std::uint64_t hash) const noexcept {
    std::size_t bucket = static_cast<std::size_t>(hash) & table_mask;
    for (std::size_t probe = 0; probe < table.size(); ++probe) {
      const std::size_t slot = table[bucket];
      if (slot == kEmpty) {
        return kEmpty;
      }
      if (slot != kTombstone && entries[slot].hash == hash &&
          matches(entries[slot], token, codes)) {
        return slot;
      }
      bucket = (bucket + 1) & table_mask;
    }
    return kEmpty;
  }

  void insert_table(const std::size_t slot) noexcept {
    std::size_t bucket =
        static_cast<std::size_t>(entries[slot].hash) & table_mask;
    std::size_t first_tombstone = kEmpty;
    for (std::size_t probe = 0; probe < table.size(); ++probe) {
      if (table[bucket] == kEmpty) {
        table[first_tombstone == kEmpty ? bucket : first_tombstone] = slot;
        return;
      }
      if (table[bucket] == kTombstone && first_tombstone == kEmpty) {
        first_tombstone = bucket;
      }
      bucket = (bucket + 1) & table_mask;
    }
    // The table is provisioned at <= 50% load, so this is unreachable.
    table[first_tombstone] = slot;
  }

  void erase_table(const std::size_t slot) noexcept {
    std::size_t bucket =
        static_cast<std::size_t>(entries[slot].hash) & table_mask;
    for (std::size_t probe = 0; probe < table.size(); ++probe) {
      if (table[bucket] == kEmpty) {
        return;
      }
      if (table[bucket] == slot) {
        table[bucket] = kTombstone;
        return;
      }
      bucket = (bucket + 1) & table_mask;
    }
  }

  void detach(const std::size_t slot) noexcept {
    Entry& entry = entries[slot];
    if (entry.previous != kEmpty) {
      entries[entry.previous].next = entry.next;
    } else {
      oldest = entry.next;
    }
    if (entry.next != kEmpty) {
      entries[entry.next].previous = entry.previous;
    } else {
      newest = entry.previous;
    }
    entry.previous = kEmpty;
    entry.next = kEmpty;
  }

  void make_newest(const std::size_t slot) noexcept {
    if (newest == slot) {
      return;
    }
    if (entries[slot].previous != kEmpty || entries[slot].next != kEmpty ||
        oldest == slot) {
      detach(slot);
    }
    entries[slot].previous = newest;
    entries[slot].next = kEmpty;
    if (newest != kEmpty) {
      entries[newest].next = slot;
    } else {
      oldest = slot;
    }
    newest = slot;
  }

  std::size_t state_width;
  std::size_t capacity;
  double quantization_step;
  double similarity_radius;
  std::size_t max_output_candidates;
  bool bypass;
  std::vector<Entry> entries;
  std::vector<std::size_t> table;
  std::vector<std::size_t> free_slots;
  std::vector<std::int16_t> scratch_codes;
  std::size_t table_mask;
  std::size_t size = 0;
  std::size_t oldest = kEmpty;
  std::size_t newest = kEmpty;
  TransitionCacheMetrics counters;
  double approximation_error_sum = 0.0;
};

TransitionCache::TransitionCache(
    const std::size_t state_width, const std::size_t capacity,
    const float quantization_step, const float similarity_radius,
    const std::size_t max_output_candidates, const bool bypass)
    : impl_(std::make_unique<Impl>(state_width, capacity, quantization_step,
                                  similarity_radius, max_output_candidates,
                                  bypass)) {}

TransitionCache::~TransitionCache() = default;
TransitionCache::TransitionCache(TransitionCache&&) noexcept = default;
TransitionCache& TransitionCache::operator=(TransitionCache&&) noexcept =
    default;

StateFingerprint TransitionCache::fingerprint(
    const std::span<const float> state,
    const std::uint32_t input_token) const {
  validate_state(state, impl_->state_width, "state");
  StateFingerprint result;
  result.input_token = input_token;
  result.codes.resize(impl_->state_width);
  for (std::size_t index = 0; index < impl_->state_width; ++index) {
    result.codes[index] =
        quantize_scalar(state[index], impl_->quantization_step);
  }
  return result;
}

bool TransitionCache::put(
    const std::span<const float> state, const std::uint32_t input_token,
    const std::span<const float> next_state,
    const std::span<const TransitionCandidate> output_candidates,
    const float confidence, const TransitionSource source) {
  validate_state(state, impl_->state_width, "state");
  validate_state(next_state, impl_->state_width, "next state");
  if (source != TransitionSource::kOffline &&
      source != TransitionSource::kOnline) {
    throw std::invalid_argument("invalid transition population source");
  }
  if (output_candidates.empty() ||
      output_candidates.size() > impl_->max_output_candidates) {
    throw std::invalid_argument(
        "output candidates must be non-empty and fit configured capacity");
  }
  if (!std::isfinite(confidence) || confidence < 0.0F || confidence > 1.0F) {
    throw std::invalid_argument("transition confidence must lie in [0, 1]");
  }
  if (!std::all_of(output_candidates.begin(), output_candidates.end(),
                   [](const TransitionCandidate candidate) {
                     return std::isfinite(candidate.score);
                   })) {
    throw std::invalid_argument("transition candidate scores must be finite");
  }
  if (impl_->bypass) {
    ++impl_->counters.bypassed_puts;
    return false;
  }

  impl_->quantize(state);
  const std::uint64_t hash = fingerprint_hash(input_token, impl_->scratch_codes);
  std::size_t slot = impl_->find(input_token, impl_->scratch_codes, hash);
  if (slot != kEmpty) {
    const double distance = relative_distance(
        state, impl_->entries[slot].reference_state);
    if (distance > impl_->similarity_radius) {
      ++impl_->counters.collisions;
    }
  } else if (impl_->size < impl_->capacity) {
    slot = impl_->free_slots.back();
    impl_->free_slots.pop_back();
    ++impl_->size;
  } else {
    slot = impl_->oldest;
    impl_->erase_table(slot);
    impl_->detach(slot);
    ++impl_->counters.evictions;
  }

  Impl::Entry& entry = impl_->entries[slot];
  entry.occupied = true;
  entry.input_token = input_token;
  entry.hash = hash;
  std::copy(impl_->scratch_codes.begin(), impl_->scratch_codes.end(),
            entry.codes.begin());
  std::copy(state.begin(), state.end(), entry.reference_state.begin());
  std::copy(next_state.begin(), next_state.end(), entry.next_state.begin());
  entry.candidates.assign(output_candidates.begin(), output_candidates.end());
  entry.confidence = confidence;
  entry.source = source;
  impl_->make_newest(slot);
  if (impl_->find(input_token, entry.codes, hash) == kEmpty) {
    impl_->insert_table(slot);
  }
  if (source == TransitionSource::kOffline) {
    ++impl_->counters.offline_puts;
  } else {
    ++impl_->counters.online_puts;
  }
  return true;
}

bool TransitionCache::put_offline(
    const std::span<const float> state, const std::uint32_t input_token,
    const std::span<const float> next_state,
    const std::span<const TransitionCandidate> output_candidates,
    const float confidence) {
  return put(state, input_token, next_state, output_candidates, confidence,
             TransitionSource::kOffline);
}

bool TransitionCache::put_online(
    const std::span<const float> state, const std::uint32_t input_token,
    const std::span<const float> next_state,
    const std::span<const TransitionCandidate> output_candidates,
    const float confidence) {
  return put(state, input_token, next_state, output_candidates, confidence,
             TransitionSource::kOnline);
}

TransitionLookup TransitionCache::lookup(
    const std::span<const float> state, const std::uint32_t input_token,
    const std::span<const float> actual_next_state) {
  validate_state(state, impl_->state_width, "state");
  if (!actual_next_state.empty()) {
    validate_state(actual_next_state, impl_->state_width, "actual next state");
  }
  ++impl_->counters.lookups;
  if (impl_->bypass) {
    ++impl_->counters.misses;
    ++impl_->counters.bypassed_lookups;
    return TransitionLookup{false, TransitionLookupReason::kBypass, {}, 0.0};
  }
  impl_->quantize(state);
  const std::uint64_t hash = fingerprint_hash(input_token, impl_->scratch_codes);
  const std::size_t slot =
      impl_->find(input_token, impl_->scratch_codes, hash);
  if (slot == kEmpty) {
    ++impl_->counters.misses;
    return TransitionLookup{false, TransitionLookupReason::kMiss, {}, 0.0};
  }
  Impl::Entry& entry = impl_->entries[slot];
  const double distance = relative_distance(state, entry.reference_state);
  if (distance > impl_->similarity_radius) {
    ++impl_->counters.misses;
    ++impl_->counters.radius_rejections;
    ++impl_->counters.collisions;
    return TransitionLookup{false, TransitionLookupReason::kRadiusRejection,
                            {}, distance};
  }
  impl_->make_newest(slot);
  ++impl_->counters.hits;
  if (!actual_next_state.empty()) {
    const double error =
        relative_distance(actual_next_state, entry.next_state);
    ++impl_->counters.approximation_error_samples;
    impl_->approximation_error_sum += error;
    impl_->counters.max_approximation_error =
        std::max(impl_->counters.max_approximation_error, error);
  }
  return TransitionLookup{
      true,
      TransitionLookupReason::kHit,
      {entry.next_state, entry.candidates, entry.confidence},
      distance};
}

void TransitionCache::set_bypass(const bool enabled) noexcept {
  impl_->bypass = enabled;
}

bool TransitionCache::bypass() const noexcept { return impl_->bypass; }

void TransitionCache::clear() noexcept {
  std::fill(impl_->table.begin(), impl_->table.end(), kEmpty);
  impl_->free_slots.clear();
  for (std::size_t index = impl_->capacity; index-- > 0;) {
    impl_->free_slots.push_back(index);
    Impl::Entry& entry = impl_->entries[index];
    entry.occupied = false;
    entry.previous = kEmpty;
    entry.next = kEmpty;
    entry.candidates.clear();
  }
  impl_->size = 0;
  impl_->oldest = kEmpty;
  impl_->newest = kEmpty;
}

TransitionCacheMetrics TransitionCache::metrics() const noexcept {
  TransitionCacheMetrics result = impl_->counters;
  result.entries = impl_->size;
  result.capacity = impl_->capacity;
  result.hit_rate = result.lookups == 0
                        ? 0.0
                        : static_cast<double>(result.hits) /
                              static_cast<double>(result.lookups);
  result.mean_approximation_error =
      result.approximation_error_samples == 0
          ? 0.0
          : impl_->approximation_error_sum /
                static_cast<double>(result.approximation_error_samples);
  return result;
}

std::size_t TransitionCache::state_width() const noexcept {
  return impl_->state_width;
}

std::size_t TransitionCache::capacity() const noexcept {
  return impl_->capacity;
}

}  // namespace engram
