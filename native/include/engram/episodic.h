#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace engram {

struct EpisodicConfig {
  std::size_t key_width{};
  std::size_t value_width{};
  std::size_t local_window{16};
  std::size_t retrieval_capacity{1024};
  std::size_t retrieval_candidates{16};
  std::size_t retrieval_top_k{4};
  double decay{0.99};
  double older_weight{0.5};
  double epsilon{1e-6};
};

struct EpisodicStepMetrics {
  std::size_t tokens_seen{};
  std::size_t local_tokens{};
  std::size_t older_tokens{};
  std::size_t recurrent_steps{};
  std::size_t retrieval_candidates{};
  std::size_t retrievals{};
  std::size_t bytes_read{};
  std::size_t state_bytes{};
  std::size_t scratch_bytes{};
};

// Exact bounded local attention plus a normalized recurrent summary and a
// bounded, symmetrically-int8-quantized older-context retrieval ring.
class HybridEpisodicMemory {
 public:
  explicit HybridEpisodicMemory(EpisodicConfig config);

  void reset() noexcept;
  EpisodicStepMetrics step(const float* query, const float* key,
                           const float* value, float* output);

  [[nodiscard]] const EpisodicConfig& config() const noexcept;
  [[nodiscard]] std::size_t local_count() const noexcept;
  [[nodiscard]] std::size_t older_count() const noexcept;
  [[nodiscard]] std::size_t tokens_seen() const noexcept;
  [[nodiscard]] std::size_t allocated_state_bytes() const noexcept;
  [[nodiscard]] std::size_t scratch_bytes() const noexcept;
  [[nodiscard]] std::span<const float> last_local_output() const noexcept;
  [[nodiscard]] std::span<const float> last_recurrent_output() const noexcept;
  [[nodiscard]] std::span<const float> last_retrieval_output() const noexcept;
  [[nodiscard]] std::span<const std::uint64_t>
  last_retrieved_positions() const noexcept;

 private:
  void validate_vector(const float* vector, std::size_t width,
                       const char* name) const;
  void update_recurrent(const float* query, const float* key,
                        const float* value);
  void read_recurrent(const float* query);
  void append_older(std::uint64_t position, const float* key,
                    const float* value);
  void compute_local(const float* query);
  std::size_t retrieve(const float* query, std::size_t& bytes_read);

  EpisodicConfig config_;
  std::size_t tokens_seen_{};
  std::size_t recent_start_{};
  std::size_t recent_size_{};
  std::size_t older_start_{};
  std::size_t older_size_{};
  std::size_t recurrent_steps_{};
  std::size_t last_retrieval_count_{};

  std::vector<float> recent_keys_;
  std::vector<float> recent_values_;
  std::vector<std::uint64_t> recent_positions_;
  std::vector<std::int8_t> older_key_codes_;
  std::vector<float> older_key_scales_;
  std::vector<std::int8_t> older_value_codes_;
  std::vector<float> older_value_scales_;
  std::vector<std::uint64_t> older_positions_;
  std::vector<double> recurrent_numerator_;
  std::vector<double> recurrent_normalizer_;

  // Fixed-size scratch and component-output buffers.
  std::vector<double> query_features_;
  std::vector<double> key_features_;
  std::vector<double> local_scores_;
  std::vector<std::size_t> candidate_slots_;
  std::vector<double> candidate_scores_;
  std::vector<std::size_t> selected_slots_;
  std::vector<double> selected_scores_;
  std::vector<std::uint64_t> selected_positions_;
  std::vector<float> local_output_;
  std::vector<float> recurrent_output_;
  std::vector<float> retrieval_output_;
};

}  // namespace engram
