#include "engram/dip.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace {

template <typename Function>
long long measure(const std::size_t iterations, Function&& function) {
  const auto started = std::chrono::steady_clock::now();
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    function(iteration);
  }
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now() - started)
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 6 || argc > 8) {
      std::cerr << "usage: engram-dip-bench LAYER_OR_PACKAGE_DIR ITERATIONS INPUT_COORDINATES CANDIDATES TOP_K [--dense-first] [--stream-completion|--record-completion|--record-stream-completion]\n";
      return 2;
    }
    const std::size_t iterations = std::stoull(argv[2]);
    const std::size_t input_coordinates = std::stoull(argv[3]);
    const std::size_t candidates = std::stoull(argv[4]);
    const std::size_t top_k = std::stoull(argv[5]);
    bool dense_first = false;
    bool stream_completion = false;
    bool record_completion = false;
    bool record_stream_completion = false;
    for (int index = 6; index < argc; ++index) {
      const std::string_view option(argv[index]);
      if (option == "--dense-first") dense_first = true;
      else if (option == "--stream-completion") stream_completion = true;
      else if (option == "--record-completion") record_completion = true;
      else if (option == "--record-stream-completion") record_stream_completion = true;
      else throw std::invalid_argument("unknown benchmark option");
    }
    if (static_cast<int>(stream_completion) + static_cast<int>(record_completion) +
            static_cast<int>(record_stream_completion) >
        1) {
      throw std::invalid_argument("completion modes are mutually exclusive");
    }
    const auto completion_mode =
        record_stream_completion
            ? engram::DIPCompletionMode::RecordMajorStream
            : record_completion
            ? engram::DIPCompletionMode::RecordMajorGather
            : (stream_completion
                   ? engram::DIPCompletionMode::FullCoordinateStream
                   : engram::DIPCompletionMode::CandidateGather);
    if (iterations == 0) throw std::invalid_argument("iterations must be positive");
    const std::filesystem::path input_path(argv[1]);
    std::vector<std::filesystem::path> layer_paths;
    if (std::filesystem::is_regular_file(input_path / "config.npy")) {
      layer_paths.push_back(input_path);
    } else {
      for (const auto& entry : std::filesystem::directory_iterator(input_path)) {
        if (entry.is_directory() &&
            entry.path().filename().string().starts_with("layer-") &&
            std::filesystem::is_regular_file(entry.path() / "config.npy")) {
          layer_paths.push_back(entry.path());
        }
      }
      std::sort(layer_paths.begin(), layer_paths.end());
    }
    if (layer_paths.empty()) {
      throw std::invalid_argument("no serialized DIP layers found");
    }
    std::vector<std::unique_ptr<engram::DIPLayerStorage>> storage;
    std::vector<engram::DIPLayerView> layers;
    for (const auto& path : layer_paths) {
      storage.push_back(std::make_unique<engram::DIPLayerStorage>(path));
      layers.push_back(storage.back()->view());
    }
    const auto layer = layers.front();
    for (const auto& current : layers) {
      if (current.hidden_size != layer.hidden_size ||
          current.records != layer.records) {
        throw std::invalid_argument("serialized DIP layer dimensions differ");
      }
    }
    engram::DIPScratch scratch(layer.hidden_size, layer.records);
    std::vector<float> hidden(layer.hidden_size);
    std::vector<float> sparse_output(layer.hidden_size);
    std::vector<float> dense_output(layer.hidden_size);
    std::vector<engram::DIPRecordResult> selected(top_k);
    for (std::size_t index = 0; index < hidden.size(); ++index) {
      hidden[index] = static_cast<float>((static_cast<int>(index % 29) - 14) * 0.03125);
    }
    engram::DIPReadMetrics metrics;
    for (const auto& current : layers) {
      engram::dip_read_scalar(current, hidden, input_coordinates, candidates,
                              top_k, sparse_output, selected, scratch, &metrics,
                              completion_mode);
      engram::dip_dense_scalar(current, hidden, dense_output, scratch);
    }
    engram::dip_read_scalar(layer, hidden, input_coordinates, candidates, top_k,
                            sparse_output, selected, scratch, &metrics,
                            completion_mode);
    double reference_output_checksum = 0.0;
    for (const float value : sparse_output) reference_output_checksum += value;
    std::vector<std::uint32_t> reference_selected;
    reference_selected.reserve(selected.size());
    for (const auto& item : selected) reference_selected.push_back(item.index);

    volatile float checksum = 0.0F;
    const auto sparse_body = [&](const std::size_t iteration) {
      hidden[iteration % hidden.size()] += 1.0e-7F;
      for (const auto& current : layers) {
        engram::dip_read_scalar(current, hidden, input_coordinates, candidates,
                                top_k, sparse_output, selected, scratch, nullptr,
                                completion_mode);
      }
      checksum = checksum + sparse_output[iteration % sparse_output.size()];
    };
    const auto dense_body = [&](const std::size_t iteration) {
      hidden[iteration % hidden.size()] -= 1.0e-7F;
      for (const auto& current : layers) {
        engram::dip_dense_scalar(current, hidden, dense_output, scratch);
      }
      checksum = checksum + dense_output[iteration % dense_output.size()];
    };
    long long sparse_ns = 0;
    long long dense_ns = 0;
    if (dense_first) {
      dense_ns = measure(iterations, dense_body);
      sparse_ns = measure(iterations, sparse_body);
    } else {
      sparse_ns = measure(iterations, sparse_body);
      dense_ns = measure(iterations, dense_body);
    }
    const double reads = static_cast<double>(iterations * layers.size());
    std::cout << "{\"iterations\":" << iterations
              << ",\"layers\":" << layers.size()
              << ",\"dense_first\":" << (dense_first ? "true" : "false")
              << ",\"stream_completion\":"
              << (stream_completion ? "true" : "false")
              << ",\"record_completion\":"
              << (record_completion ? "true" : "false")
              << ",\"record_stream_completion\":"
              << (record_stream_completion ? "true" : "false")
              << ",\"hidden_source\":\"deterministic_synthetic_period_29\""
              << ",\"phase_timing_scope\":\"single_untimed_reference_layer\""
              << ",\"sparse_ns_per_pass\":"
              << static_cast<double>(sparse_ns) / iterations
              << ",\"dense_ns_per_pass\":"
              << static_cast<double>(dense_ns) / iterations
              << ",\"sparse_ns_per_read\":"
              << static_cast<double>(sparse_ns) / reads
              << ",\"dense_ns_per_read\":"
              << static_cast<double>(dense_ns) / reads
              << ",\"wall_time_speedup\":"
              << static_cast<double>(dense_ns) / sparse_ns
              << ",\"logical_weight_bytes\":" << metrics.logical_weight_bytes
              << ",\"executed_weight_bytes\":"
              << metrics.executed_weight_bytes
              << ",\"cache_line_weight_bytes\":"
              << metrics.cache_line_weight_bytes
              << ",\"dense_weight_bytes\":" << metrics.dense_weight_bytes
              << ",\"logical_fraction_of_dense\":"
              << static_cast<double>(metrics.logical_weight_bytes) /
                     metrics.dense_weight_bytes
              << ",\"executed_fraction_of_dense\":"
              << static_cast<double>(metrics.executed_weight_bytes) /
                     metrics.dense_weight_bytes
              << ",\"cache_line_fraction_of_dense\":"
              << static_cast<double>(metrics.cache_line_weight_bytes) /
                     metrics.dense_weight_bytes
              << ",\"phase_ns\":{\"coordinate_selection\":"
              << metrics.coordinate_selection_ns
              << ",\"partial_projection\":" << metrics.partial_projection_ns
              << ",\"proxy_scoring\":" << metrics.proxy_scoring_ns
              << ",\"candidate_selection\":"
              << metrics.candidate_selection_ns
              << ",\"candidate_completion\":"
              << metrics.candidate_completion_ns
              << ",\"exact_scoring\":" << metrics.exact_scoring_ns
              << ",\"exact_selection\":" << metrics.exact_selection_ns
              << ",\"down_accumulation\":"
              << metrics.down_accumulation_ns << '}'
              << ",\"reference_output_checksum\":"
              << reference_output_checksum << ",\"reference_selected\":[";
    for (std::size_t index = 0; index < reference_selected.size(); ++index) {
      if (index != 0) std::cout << ',';
      std::cout << reference_selected[index];
    }
    std::cout << ']'
              << ",\"checksum\":" << checksum
              << ",\"dram_bytes_measured\":false}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "engram-dip-bench: " << error.what() << '\n';
    return 1;
  }
}
