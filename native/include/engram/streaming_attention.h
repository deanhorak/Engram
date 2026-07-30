#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace engram {

struct StreamingAttentionConfig {
  std::size_t query_heads{};
  std::size_t key_value_heads{};
  std::size_t head_dimension{};
  std::size_t local_window{16};
  std::size_t older_candidates{8};
  std::size_t older_top_k{4};
  std::size_t sink_tokens{2};
  // Optional bounded BF16 episodic K/V storage. Both fields must be zero to
  // disable episodic memory, or positive with slots divisible by span size.
  std::size_t episodic_slots{};
  std::size_t episodic_span_size{};
  // Empty preserves the legacy behavior in which every query head reads the
  // configured episodic span. Otherwise this must contain one 0/1 entry per
  // query head, with at least one selected head.
  std::vector<std::uint8_t> episodic_head_mask;
  float scale{};
  // Additive logit bias applied only to selected episodic entries before the
  // joint local/older/episodic softmax. Zero preserves the original route.
  float episodic_logit_bias{};
};

struct StreamingAttentionMetrics {
  std::uint64_t tokens_seen{};
  std::uint64_t local_entries{};
  std::uint64_t active_older_entries{};
  std::uint64_t candidate_key_bytes{};
  std::uint64_t selected_value_bytes{};
  std::uint64_t local_kv_bytes{};
  std::uint64_t eviction_events{};
  std::uint64_t older_candidate_entries_scored{};
  std::uint64_t older_selected_entries{};
  std::uint64_t sink_insertions{};
  std::uint64_t heavy_hitter_updates{};
  std::uint64_t state_bytes{};
  std::uint64_t scratch_bytes{};
  std::uint64_t episodic_slots_written{};
  std::uint64_t episodic_active_slots{};
  std::uint64_t episodic_read_events{};
  std::uint64_t episodic_entries_read{};
  std::uint64_t episodic_write_bytes{};
  std::uint64_t episodic_key_read_bytes{};
  std::uint64_t episodic_value_read_bytes{};
  std::uint64_t episodic_duplicate_older_entries_suppressed{};
};

// Stateful causal attention with an exact local window, fixed attention sinks,
// and a bounded per-query-head cumulative-attention heavy-hitter cache.
class StreamingAttention {
 public:
  static constexpr std::size_t kNoEpisodicDirective =
      std::numeric_limits<std::size_t>::max();
  static constexpr std::size_t kRegularTraceLocalEntries = 16;
  static constexpr std::size_t kRegularTraceOlderEntries = 4;
  static constexpr std::size_t kRegularTraceEntries =
      kRegularTraceLocalEntries + kRegularTraceOlderEntries;
  static constexpr std::size_t kC28TraceEpisodicEntries = 8;
  static constexpr std::size_t kC28TraceEntries =
      kRegularTraceEntries + kC28TraceEpisodicEntries;
  static constexpr std::size_t kQKTraceHeadDimension = 128;
  static constexpr std::size_t kQKPartialBands = 8;
  static constexpr std::size_t kQKPartialBandHalfWidth = 8;
  static constexpr std::size_t kQKPartialBandDimensions =
      2 * kQKPartialBandHalfWidth;
  static constexpr std::uint8_t kRegularTraceInvalid = 0;
  static constexpr std::uint8_t kRegularTraceLocal = 1;
  static constexpr std::uint8_t kRegularTraceOlder = 2;

  explicit StreamingAttention(StreamingAttentionConfig config);

  void reset() noexcept;
  StreamingAttentionMetrics step(std::span<const float> query,
                                 std::span<const float> key,
                                 std::span<const float> value,
                                 std::span<float> output);
  // Optionally capture the current K/V in one fixed BF16 slot and/or make one
  // contiguous episodic span visible in the same softmax as local and
  // heavy-hitter entries. The sentinel above disables either directive.
  StreamingAttentionMetrics step_episodic(
      std::span<const float> query, std::span<const float> key,
      std::span<const float> value, std::size_t write_slot,
      std::size_t read_span, std::span<float> output);
  // Evaluator-only view of the exact joint episodic softmax. The regular and
  // episodic value components are weighted by the joint denominator, so their
  // sum reconstructs the ordinary output up to floating-point regrouping.
  // Mass arrays contain one entry per query head. Existing output/state
  // arithmetic is completed before these trace-only values are derived.
  StreamingAttentionMetrics step_episodic_traced(
      std::span<const float> query, std::span<const float> key,
      std::span<const float> value, std::size_t write_slot,
      std::size_t read_span, std::span<float> output,
      std::span<float> regular_component,
      std::span<float> episodic_component,
      std::span<float> regular_mass,
      std::span<float> episodic_mass);
  // Evaluator-only extension of the partition trace for a real episodic
  // read. Slot mass is query-head-major/span-minor and slot values are
  // query-head-major/span-minor/dimension-minor. Selected heads receive the
  // exact normalized joint-softmax weights and BF16-decoded values in
  // read-span order. Both arrays are zero for masked heads.
  StreamingAttentionMetrics step_episodic_slots_traced(
      std::span<const float> query, std::span<const float> key,
      std::span<const float> value, std::size_t write_slot,
      std::size_t read_span, std::span<float> output,
      std::span<float> regular_component,
      std::span<float> episodic_component,
      std::span<float> regular_mass,
      std::span<float> episodic_mass,
      std::span<float> slot_mass,
      std::span<float> slot_values);
  // Evaluator-only fixed-layout trace for the already-read regular values in
  // the frozen W16/K4 policy. Entries [0, 16) are valid local values in
  // chronological oldest-to-newest order, followed by zero/sentinel padding.
  // Entries [16, 20) are selected older values in native score order (with
  // the existing absolute-position tie break), followed by padding. Kind 0
  // marks padding, 1 local, and 2 selected older. Entry masses are normalized
  // by the exact ordinary attention denominator.
  StreamingAttentionMetrics step_regular_entries_traced(
      std::span<const float> query, std::span<const float> key,
      std::span<const float> value, std::span<float> output,
      std::span<float> entry_mass,
      std::span<float> entry_values,
      std::span<std::uint8_t> valid_kind,
      std::span<std::uint64_t> positions);
  // Evaluator-only extension of the fixed regular-entry trace. Q/K partials
  // are query-head-major/C28-entry-major/block-major raw dot-product sums.
  // C28 entries [0, 20) match the regular trace and entries [20, 28) name
  // episodic read-span offsets. This regular-only route leaves the episodic
  // tail exactly zero. Each of the eight bands is RoPE-closed: band r owns
  // dimensions [8r, 8r+8) and [64+8r, 64+8r+8). This evaluator fails closed
  // unless the production head dimension is exactly 128.
  StreamingAttentionMetrics step_regular_entries_qk_traced(
      std::span<const float> query, std::span<const float> key,
      std::span<const float> value, std::span<float> output,
      std::span<float> entry_mass,
      std::span<float> entry_values,
      std::span<std::uint8_t> valid_kind,
      std::span<std::uint64_t> positions,
      std::span<float> qk_partials);
  // Minimal non-intervening C28 Q/K capture. Either episodic directive may
  // use kNoEpisodicDirective. The fixed regular ordering is still used even
  // when the companion mass/value trace is not requested.
  StreamingAttentionMetrics step_c28_qk_traced(
      std::span<const float> query, std::span<const float> key,
      std::span<const float> value, std::size_t write_slot,
      std::size_t read_span, std::span<float> output,
      std::span<float> qk_partials);
  // Combined evaluator-only episodic partition, per-slot, and fixed regular-
  // entry trace. Existing trace entry points remain unchanged.
  StreamingAttentionMetrics step_episodic_full_traced(
      std::span<const float> query, std::span<const float> key,
      std::span<const float> value, std::size_t write_slot,
      std::size_t read_span, std::span<float> output,
      std::span<float> regular_component,
      std::span<float> episodic_component,
      std::span<float> regular_mass,
      std::span<float> episodic_mass,
      std::span<float> slot_mass,
      std::span<float> slot_values,
      std::span<float> regular_entry_mass,
      std::span<float> regular_entry_values,
      std::span<std::uint8_t> regular_entry_valid_kind,
      std::span<std::uint64_t> regular_entry_positions);
  // Evaluator-only C28 Q/K extension of the combined trace above. The first
  // 20 entries have exactly the regular trace's order; the last eight have
  // read-span offset order. Invalid regular, masked episodic, and absent
  // episodic entries are canonical zero. Raw partials exclude scale and
  // episodic bias so summing blocks reconstructs the legacy scalar Q dot K
  // up to ordinary regrouping roundoff. Each stored band is scaled by the
  // ordinary attention scale; the additive episodic bias is excluded.
  StreamingAttentionMetrics step_episodic_full_qk_traced(
      std::span<const float> query, std::span<const float> key,
      std::span<const float> value, std::size_t write_slot,
      std::size_t read_span, std::span<float> output,
      std::span<float> regular_component,
      std::span<float> episodic_component,
      std::span<float> regular_mass,
      std::span<float> episodic_mass,
      std::span<float> slot_mass,
      std::span<float> slot_values,
      std::span<float> regular_entry_mass,
      std::span<float> regular_entry_values,
      std::span<std::uint8_t> regular_entry_valid_kind,
      std::span<std::uint64_t> regular_entry_positions,
      std::span<float> qk_partials);
  // Evaluator-only mass of a fixed set of strictly earlier positions. All
  // tracked positions must still be in the exact local window before the
  // current row is inserted. This keeps validation fail-closed and is the
  // required route for the W128 same-state shadow.
  StreamingAttentionMetrics step_tracked_positions(
      std::span<const float> query, std::span<const float> key,
      std::span<const float> value,
      std::span<const std::uint64_t> tracked_positions,
      std::span<float> tracked_mass, std::span<float> output);

  [[nodiscard]] const StreamingAttentionConfig& config() const noexcept;
  [[nodiscard]] std::size_t tokens_seen() const noexcept;
  [[nodiscard]] std::size_t active_older_entries() const noexcept;
  [[nodiscard]] std::size_t active_episodic_slots() const noexcept;
  [[nodiscard]] std::size_t qk_partial_blocks() const noexcept {
    return kQKPartialBands;
  }
  [[nodiscard]] std::size_t allocated_state_bytes() const noexcept;
  [[nodiscard]] std::size_t scratch_bytes() const noexcept;

 private:
  [[nodiscard]] std::size_t recent_offset(std::size_t slot,
                                          std::size_t kv_head) const noexcept;
  [[nodiscard]] std::size_t older_offset(std::size_t head,
                                         std::size_t slot) const noexcept;
  [[nodiscard]] std::size_t episodic_offset(
      std::size_t slot, std::size_t kv_head) const noexcept;
  void evict_recent(std::size_t slot, std::uint64_t& sink_insertions,
                    std::uint64_t& heavy_hitter_updates);
  void validate_inputs(std::span<const float> query,
                       std::span<const float> key,
                       std::span<const float> value,
                       std::span<float> output) const;
  StreamingAttentionMetrics step_episodic_impl(
      std::span<const float> query, std::span<const float> key,
      std::span<const float> value, std::size_t write_slot,
      std::size_t read_span, std::span<float> output,
      std::span<float> regular_component,
      std::span<float> episodic_component,
      std::span<float> regular_mass,
      std::span<float> episodic_mass,
      std::span<const std::uint64_t> tracked_positions,
      std::span<float> tracked_mass,
      std::span<float> episodic_slot_mass,
      std::span<float> episodic_slot_values,
      std::span<float> regular_entry_mass,
      std::span<float> regular_entry_values,
      std::span<std::uint8_t> regular_entry_valid_kind,
      std::span<std::uint64_t> regular_entry_positions,
      std::span<float> qk_partials);

  StreamingAttentionConfig config_;
  std::size_t groups_{};
  std::size_t tokens_seen_{};
  std::size_t recent_start_{};
  std::size_t recent_size_{};

  std::vector<float> recent_keys_;
  std::vector<float> recent_values_;
  std::vector<float> recent_mass_;
  std::vector<std::uint64_t> recent_positions_;

  std::vector<float> older_keys_;
  std::vector<float> older_values_;
  std::vector<float> older_scores_;
  std::vector<std::uint64_t> older_positions_;
  std::vector<std::uint8_t> older_active_;

  std::vector<std::uint16_t> episodic_keys_;
  std::vector<std::uint16_t> episodic_values_;
  std::vector<std::uint64_t> episodic_positions_;
  std::size_t episodic_active_slots_{};

  std::vector<float> score_scratch_;
  std::vector<float> candidate_score_scratch_;
  std::vector<float> weight_scratch_;
  std::vector<std::size_t> selected_scratch_;
};

}  // namespace engram
