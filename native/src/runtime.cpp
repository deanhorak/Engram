#include "engram/runtime.h"

#include "engram/controller.h"
#include "engram/episodic.h"
#include "engram/npy.h"
#include "engram/package.h"
#include "engram/semantic.h"
#include "engram/semantic_ivf.h"
#include "engram/semantic_quantized.h"
#include "engram/transition_cache.h"
#include "engram/thread_pool.h"
#include "engram/vocabulary.h"
#include "engram/vocabulary_ivf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace engram {
namespace {

void require_shape(const NpyArray& array,
                   const std::vector<std::size_t>& expected,
                   const char* name) {
  if (array.shape() != expected) {
    throw std::runtime_error(std::string("unexpected shape for ") + name);
  }
}

struct SemanticStorage {
  NpyArray gate_codes;
  NpyArray gate_offsets;
  NpyArray gate_scales;
  NpyArray up_codes;
  NpyArray up_offsets;
  NpyArray up_scales;
  NpyArray value_codes;
  NpyArray value_codebooks;
  NpyArray ivf_centroids;
  NpyArray ivf_offsets;
  NpyArray ivf_postings;
  QuantizedSemanticScratch scratch;
  SemanticIvfScratch ivf_scratch;
  std::vector<float> output;
  std::vector<SemanticRecordResult> selected;
  std::vector<std::uint32_t> probed_clusters;
  std::vector<SemanticIvfCandidate> ivf_candidates;
  std::vector<std::uint32_t> candidate_indices;
  std::vector<float> candidate_proxy_scores;
  SemanticIvfSearchMetrics ivf_metrics;

  SemanticStorage(const std::filesystem::path& root, std::size_t layer,
                  std::size_t records, std::size_t hidden, std::size_t top_k,
                  std::size_t candidate_count, std::size_t clusters)
      : gate_codes(load_npy(root / ("semantic/layer-" + [&] {
                         char value[5];
                         std::snprintf(value, sizeof(value), "%04zu", layer);
                         return std::string(value);
                       }() + "/quantized/gate_codes.npy"))),
        gate_offsets(load_npy(root / ("semantic/layer-" + [&] {
                         char value[5];
                         std::snprintf(value, sizeof(value), "%04zu", layer);
                         return std::string(value);
                       }() + "/quantized/gate_offsets.npy"))),
        gate_scales(load_npy(root / ("semantic/layer-" + [&] {
                         char value[5];
                         std::snprintf(value, sizeof(value), "%04zu", layer);
                         return std::string(value);
                       }() + "/quantized/gate_scales.npy"))),
        up_codes(load_npy(root / ("semantic/layer-" + [&] {
                       char value[5];
                       std::snprintf(value, sizeof(value), "%04zu", layer);
                       return std::string(value);
                     }() + "/quantized/up_codes.npy"))),
        up_offsets(load_npy(root / ("semantic/layer-" + [&] {
                         char value[5];
                         std::snprintf(value, sizeof(value), "%04zu", layer);
                         return std::string(value);
                       }() + "/quantized/up_offsets.npy"))),
        up_scales(load_npy(root / ("semantic/layer-" + [&] {
                         char value[5];
                         std::snprintf(value, sizeof(value), "%04zu", layer);
                         return std::string(value);
                       }() + "/quantized/up_scales.npy"))),
        value_codes(load_npy(root / ("semantic/layer-" + [&] {
                           char value[5];
                           std::snprintf(value, sizeof(value), "%04zu", layer);
                           return std::string(value);
                         }() + "/quantized/value_codes.npy"))),
        value_codebooks(load_npy(root / ("semantic/layer-" + [&] {
                           char value[5];
                           std::snprintf(value, sizeof(value), "%04zu", layer);
                           return std::string(value);
                           }() + "/quantized/value_codebooks.npy"))),
        ivf_centroids(load_npy(root / ("semantic/layer-" + [&] {
                          char value[5];
                          std::snprintf(value, sizeof(value), "%04zu", layer);
                          return std::string(value);
                        }() + "/quantized/ivf/centroids.npy"))),
        ivf_offsets(load_npy(root / ("semantic/layer-" + [&] {
                        char value[5];
                        std::snprintf(value, sizeof(value), "%04zu", layer);
                        return std::string(value);
                      }() + "/quantized/ivf/posting_offsets.npy"))),
        ivf_postings(load_npy(root / ("semantic/layer-" + [&] {
                         char value[5];
                         std::snprintf(value, sizeof(value), "%04zu", layer);
                         return std::string(value);
                       }() + "/quantized/ivf/posting_indices.npy"))),
        scratch(records),
        ivf_scratch(clusters, records),
        output(hidden),
        selected(top_k),
        probed_clusters(clusters),
        ivf_candidates(candidate_count),
        candidate_indices(candidate_count),
        candidate_proxy_scores(candidate_count) {
    require_shape(gate_codes, {records, hidden}, "semantic gate codes");
    require_shape(gate_offsets, {hidden}, "semantic gate offsets");
    require_shape(gate_scales, {hidden}, "semantic gate scales");
    require_shape(up_codes, {records, hidden}, "semantic up codes");
    require_shape(up_offsets, {hidden}, "semantic up offsets");
    require_shape(up_scales, {hidden}, "semantic up scales");
    if (value_codes.shape().size() != 2 || value_codes.shape()[0] != records ||
        value_codebooks.shape().size() != 3 ||
        value_codebooks.shape()[0] != value_codes.shape()[1] ||
        value_codebooks.shape()[2] != hidden) {
      throw std::runtime_error("unexpected quantized semantic value shapes");
    }
    require_shape(ivf_centroids, {clusters, 2 * hidden},
                  "semantic IVF centroids");
    require_shape(ivf_offsets, {clusters + 1}, "semantic IVF offsets");
    require_shape(ivf_postings, {records}, "semantic IVF postings");
    validate_semantic_ivf_index(ivf_view(), view(), ivf_scratch);
  }

  QuantizedSemanticLayerView view() const {
    return {gate_codes.shape()[0], gate_codes.shape()[1],
            value_codebooks.shape()[2], value_codes.shape()[1],
            value_codebooks.shape()[1], gate_codes.uint8().data(),
            gate_offsets.float32().data(), gate_scales.float32().data(),
            up_codes.uint8().data(), up_offsets.float32().data(),
            up_scales.float32().data(), value_codes.uint8().data(),
            value_codebooks.float32().data()};
  }

  SemanticIvfIndexView ivf_view() const {
    return {ivf_centroids.shape()[0],
            ivf_centroids.shape()[1] / 2,
            ivf_postings.shape()[0],
            ivf_centroids.float32().data(),
            ivf_offsets.uint32().data(),
            ivf_postings.uint32().data()};
  }
};

}  // namespace

struct NativeRuntime::Implementation {
  PackageMetadata metadata;
  ThreadPool thread_pool;
  NpyArray token_embeddings;
  NpyArray vocabulary_embeddings;
  NpyArray vocabulary_index;
  NpyArray vocabulary_ivf_centroids;
  NpyArray vocabulary_ivf_offsets;
  NpyArray vocabulary_ivf_tokens;
  NpyArray input_kernel;
  NpyArray recurrent_kernel;
  NpyArray bias;
  NpyArray stage_embeddings;
  NpyArray adapter_down;
  NpyArray adapter_up;
  std::vector<SemanticStorage> semantic;
  std::vector<SemanticReadMetrics> semantic_metrics;
  VocabularyIndex vocabulary;
  VocabularyScratch vocabulary_scratch;
  VocabularyIvfScratch vocabulary_ivf_scratch;
  std::optional<PreparedVocabularyIvfIndex> vocabulary_ivf;
  std::vector<TokenScore> vocabulary_ivf_candidates;
  VocabularyIvfSearchMetrics vocabulary_ivf_metrics;
  ControllerWorkspace controller_workspace;
  HybridEpisodicMemory episodic;
  TransitionCache transition_cache;
  std::vector<double> state;
  std::vector<double> controller_input;
  std::vector<double> next_state;
  std::vector<float> state_float;
  std::vector<float> previous_state_float;
  std::vector<float> semantic_average;
  std::vector<float> episodic_output;

  explicit Implementation(const std::filesystem::path& root, bool verify,
                          std::size_t thread_count,
                          std::vector<unsigned> affinity)
      : metadata(load_package(root, verify)),
        thread_pool(thread_count, std::move(affinity)),
        token_embeddings(load_npy(root / "embeddings/token_embeddings.npy")),
        vocabulary_embeddings(load_npy(root / "vocabulary/embeddings.npy")),
        vocabulary_index(load_npy(root / "vocabulary/index.npy")),
        vocabulary_ivf_centroids(
            load_npy(root / "vocabulary/ivf/centroids.npy")),
        vocabulary_ivf_offsets(
            load_npy(root / "vocabulary/ivf/posting_offsets.npy")),
        vocabulary_ivf_tokens(load_npy(root / "vocabulary/ivf/token_ids.npy")),
        input_kernel(load_npy(root / "controller/input_kernel.npy")),
        recurrent_kernel(load_npy(root / "controller/recurrent_kernel.npy")),
        bias(load_npy(root / "controller/bias.npy")),
        stage_embeddings(load_npy(root / "controller/stage_embeddings.npy")),
        adapter_down(load_npy(root / "controller/adapter_down.npy")),
        adapter_up(load_npy(root / "controller/adapter_up.npy")),
        vocabulary(std::vector<float>(vocabulary_embeddings.float32().begin(),
                                      vocabulary_embeddings.float32().end()),
                   metadata.dimensions.vocabulary_size,
                   metadata.dimensions.hidden_size),
        vocabulary_scratch(metadata.dimensions.vocabulary_size),
        vocabulary_ivf_scratch(metadata.policies.vocabulary_ivf_clusters,
                               metadata.dimensions.vocabulary_size),
        vocabulary_ivf_candidates(metadata.policies.vocabulary_candidates),
        controller_workspace(metadata.dimensions.hidden_size,
                             metadata.dimensions.controller_adapter_rank),
        episodic(EpisodicConfig{metadata.dimensions.hidden_size,
                                metadata.dimensions.hidden_size,
                                metadata.policies.local_window,
                                metadata.policies.retrieval_capacity,
                                metadata.policies.retrieval_candidates,
                                metadata.policies.retrieval_top_k,
                                metadata.policies.recurrent_decay,
                                0.5,
                                1e-6}),
        transition_cache(metadata.dimensions.hidden_size,
                         metadata.policies.transition_capacity, 0.125F,
                         static_cast<float>(metadata.policies.transition_similarity_radius)),
        state(metadata.dimensions.hidden_size),
        controller_input(metadata.dimensions.controller_input_size),
        next_state(metadata.dimensions.hidden_size),
        state_float(metadata.dimensions.hidden_size),
        previous_state_float(metadata.dimensions.hidden_size),
        semantic_average(metadata.dimensions.hidden_size),
        episodic_output(metadata.dimensions.hidden_size) {
    const auto hidden = metadata.dimensions.hidden_size;
    require_shape(token_embeddings,
                  {metadata.dimensions.vocabulary_size, hidden}, "token embeddings");
    require_shape(vocabulary_embeddings,
                  {metadata.dimensions.vocabulary_size, hidden}, "vocabulary embeddings");
    require_shape(vocabulary_index,
                  {metadata.dimensions.vocabulary_size, hidden},
                  "normalized vocabulary index");
    require_shape(vocabulary_ivf_centroids,
                  {metadata.policies.vocabulary_ivf_clusters, hidden},
                  "vocabulary IVF centroids");
    require_shape(vocabulary_ivf_offsets,
                  {metadata.policies.vocabulary_ivf_clusters + 1},
                  "vocabulary IVF offsets");
    require_shape(vocabulary_ivf_tokens,
                  {metadata.dimensions.vocabulary_size},
                  "vocabulary IVF token IDs");
    vocabulary_ivf.emplace(prepare_vocabulary_ivf_index(
        VocabularyIvfIndexView{
            metadata.dimensions.vocabulary_size,
            hidden,
            metadata.policies.vocabulary_ivf_clusters,
            metadata.dimensions.vocabulary_size,
            vocabulary_index.float32().data(),
            vocabulary_ivf_centroids.float32().data(),
            vocabulary_ivf_offsets.uint32().data(),
            vocabulary_ivf_tokens.uint32().data()},
        vocabulary_ivf_scratch));
    const std::size_t records = [&] {
      NpyArray probe = load_npy(root / "semantic/layer-0000/quantized/gate_codes.npy");
      return probe.shape().at(0);
    }();
    semantic.reserve(metadata.dimensions.semantic_layers);
    for (std::size_t layer = 0; layer < metadata.dimensions.semantic_layers; ++layer) {
      semantic.emplace_back(root, layer, records, hidden,
                            metadata.policies.semantic_top_k,
                            metadata.policies.semantic_candidates,
                            metadata.policies.semantic_ivf_clusters);
    }
    semantic_metrics.resize(metadata.dimensions.semantic_layers);
  }

  ControllerWeightsView controller_view() const {
    return {metadata.dimensions.controller_input_size,
            metadata.dimensions.hidden_size,
            metadata.dimensions.controller_stages,
            metadata.dimensions.controller_adapter_rank,
            input_kernel.float64().data(), recurrent_kernel.float64().data(),
            bias.float64().data(), stage_embeddings.float64().data(),
            adapter_down.float64().data(), adapter_up.float64().data()};
  }
};

NativeRuntime::NativeRuntime(const std::filesystem::path& package,
                             bool verify_checksums,
                             std::size_t thread_count,
                             std::vector<unsigned> affinity)
    : implementation_(
          std::make_unique<Implementation>(package, verify_checksums,
                                           thread_count, std::move(affinity))) {}
NativeRuntime::~NativeRuntime() = default;
NativeRuntime::NativeRuntime(NativeRuntime&&) noexcept = default;
NativeRuntime& NativeRuntime::operator=(NativeRuntime&&) noexcept = default;

void NativeRuntime::reset() {
  std::fill(implementation_->state.begin(), implementation_->state.end(), 0.0);
  implementation_->episodic.reset();
}

void NativeRuntime::set_transition_cache_bypass(bool enabled) {
  implementation_->transition_cache.set_bypass(enabled);
}

std::vector<std::uint32_t> NativeRuntime::tokenize_fixture(
    const std::string& prompt) const {
  if (!implementation_->metadata.fixture_only) {
    throw std::runtime_error(
        "native text tokenization is available only for fixtures; pass --tokens for real packages");
  }
  std::vector<std::uint32_t> tokens;
  const auto vocabulary = implementation_->metadata.dimensions.vocabulary_size;
  for (const unsigned char byte : prompt) {
    tokens.push_back(static_cast<std::uint32_t>(3 + byte % std::max<std::size_t>(vocabulary - 3, 1)));
  }
  if (tokens.empty()) tokens.push_back(1);
  return tokens;
}

NativeTokenMetrics NativeRuntime::step(std::uint32_t input_token,
                                       bool exact_vocabulary) {
  const auto started = std::chrono::steady_clock::now();
  auto& runtime = *implementation_;
  const auto hidden = runtime.metadata.dimensions.hidden_size;
  if (input_token >= runtime.metadata.dimensions.vocabulary_size) {
    throw std::out_of_range("input token exceeds vocabulary");
  }
  std::transform(runtime.state.begin(), runtime.state.end(),
                 runtime.state_float.begin(), [](double value) {
                   return static_cast<float>(value);
                 });
  std::copy(runtime.state_float.begin(), runtime.state_float.end(),
            runtime.previous_state_float.begin());
  const TransitionLookup cached =
      runtime.transition_cache.lookup(runtime.previous_state_float, input_token);
  if (cached.hit && !cached.transition.output_candidates.empty()) {
    std::transform(cached.transition.next_state.begin(),
                   cached.transition.next_state.end(), runtime.state.begin(),
                   [](float value) { return static_cast<double>(value); });
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
    return {cached.transition.output_candidates.front().token, 0, 0, 0, 0,
            0, 0, cached.transition.output_candidates.size(), 0, 0, 0, 0, 0,
            static_cast<std::uint64_t>(elapsed), true};
  }
  std::fill(runtime.semantic_average.begin(), runtime.semantic_average.end(), 0.0F);
  std::size_t active = 0;
  std::size_t candidates = 0;
  std::size_t proxy_records = 0;
  std::size_t probed_clusters = 0;
  std::size_t semantic_bytes = 0;
  runtime.thread_pool.parallel_for(0, runtime.semantic.size(), 1,
                                   [&](const std::size_t index) {
    auto& layer = runtime.semantic[index];
    const std::size_t candidate_count =
        runtime.metadata.policies.semantic_candidates;
    const bool zero_query = std::all_of(
        runtime.state_float.begin(), runtime.state_float.end(),
        [](const float value) { return value == 0.0F; });
    if (zero_query) {
      layer.ivf_metrics = SemanticIvfSearchMetrics{};
      layer.ivf_metrics.zero_norm_query = true;
      layer.ivf_metrics.scored_records = candidate_count;
      for (std::size_t candidate = 0; candidate < candidate_count;
           ++candidate) {
        layer.candidate_indices[candidate] =
            static_cast<std::uint32_t>(candidate);
        layer.candidate_proxy_scores[candidate] = 0.0F;
      }
    } else {
      semantic_ivf_search_scalar(
          layer.ivf_view(), layer.view(), runtime.state_float,
          runtime.metadata.policies.semantic_ivf_probes, candidate_count,
          layer.probed_clusters, layer.ivf_candidates, layer.ivf_scratch,
          &layer.ivf_metrics);
      for (std::size_t candidate = 0; candidate < candidate_count;
           ++candidate) {
        layer.candidate_indices[candidate] =
            layer.ivf_candidates[candidate].index;
        layer.candidate_proxy_scores[candidate] =
            layer.ivf_candidates[candidate].proxy_score;
      }
    }
    semantic_read_quantized_candidates_scalar(
        layer.view(), runtime.state_float, layer.candidate_indices,
        layer.candidate_proxy_scores,
        runtime.metadata.policies.semantic_top_k, layer.output,
        layer.selected, layer.scratch, &runtime.semantic_metrics[index]);
    runtime.semantic_metrics[index].proxy_records =
        layer.ivf_metrics.scored_records;
    runtime.semantic_metrics[index].proxy_key_bytes =
        layer.ivf_metrics.index_bytes_read;
    runtime.semantic_metrics[index].total_bytes_read +=
        layer.ivf_metrics.index_bytes_read;
  });
  for (std::size_t index = 0; index < runtime.semantic.size(); ++index) {
    auto& layer = runtime.semantic[index];
    const auto& metrics = runtime.semantic_metrics[index];
    for (std::size_t column = 0; column < hidden; ++column) {
      runtime.semantic_average[column] +=
          layer.output[column] / static_cast<float>(runtime.semantic.size());
    }
    active += metrics.active_records;
    candidates += metrics.candidate_records;
    proxy_records += layer.ivf_metrics.scored_records;
    probed_clusters += layer.ivf_metrics.probed_clusters;
    semantic_bytes += metrics.total_bytes_read;
  }
  const auto episodic_metrics = runtime.episodic.step(
      runtime.state_float.data(), runtime.state_float.data(),
      runtime.state_float.data(), runtime.episodic_output.data());
  const auto embeddings = runtime.token_embeddings.float32();
  const std::size_t offset = static_cast<std::size_t>(input_token) * hidden;
  for (std::size_t column = 0; column < hidden; ++column) {
    runtime.controller_input[column] = embeddings[offset + column];
    runtime.controller_input[hidden + column] = runtime.semantic_average[column];
    runtime.controller_input[2 * hidden + column] = runtime.episodic_output[column];
  }
  controller_run_fixed_scalar(runtime.controller_view(), runtime.state.data(),
                              runtime.controller_input.data(),
                              runtime.metadata.policies.cycles, 0,
                              runtime.controller_workspace,
                              runtime.next_state.data());
  runtime.state.swap(runtime.next_state);
  std::transform(runtime.state.begin(), runtime.state.end(),
                 runtime.state_float.begin(), [](double value) {
                   return static_cast<float>(value);
                 });
  VocabularySearchMetrics vocabulary_metrics;
  TokenScore predicted;
  if (exact_vocabulary) {
    runtime.vocabulary_ivf_metrics = VocabularyIvfSearchMetrics{};
    predicted = runtime.vocabulary.exact_greedy(runtime.state_float,
                                                &vocabulary_metrics);
  } else {
    vocabulary_ivf_search_scalar(
        *runtime.vocabulary_ivf, runtime.state_float,
        runtime.metadata.policies.vocabulary_ivf_probes,
        runtime.metadata.policies.vocabulary_candidates,
        runtime.vocabulary_ivf_candidates, runtime.vocabulary_ivf_scratch,
        &runtime.vocabulary_ivf_metrics);
    predicted = runtime.vocabulary.rescore_greedy(
        runtime.state_float, runtime.vocabulary_ivf_candidates,
        runtime.vocabulary_scratch, &vocabulary_metrics);
    vocabulary_metrics.proxy_scores =
        runtime.vocabulary_ivf_metrics.embedding_proxy_scores;
    vocabulary_metrics.embedding_bytes_read +=
        runtime.vocabulary_ivf_metrics.index_bytes_read;
  }
  const TransitionCandidate candidate{predicted.token, predicted.score};
  runtime.transition_cache.put_online(
      runtime.previous_state_float, input_token, runtime.state_float,
      std::span<const TransitionCandidate>(&candidate, 1), 1.0F);
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - started).count();
  return {predicted.token,
          runtime.metadata.policies.cycles,
          active,
          candidates,
          proxy_records,
          probed_clusters,
          episodic_metrics.retrievals,
          vocabulary_metrics.exact_scores,
          runtime.vocabulary_ivf_metrics.embedding_proxy_scores,
          runtime.vocabulary_ivf_metrics.probed_clusters,
          semantic_bytes,
          episodic_metrics.bytes_read,
          vocabulary_metrics.embedding_bytes_read,
          static_cast<std::uint64_t>(elapsed),
          false};
}

std::vector<NativeTokenMetrics> NativeRuntime::generate(
    const std::vector<std::uint32_t>& prompt_tokens, std::size_t max_tokens,
    bool exact_vocabulary) {
  if (prompt_tokens.empty()) throw std::invalid_argument("prompt tokens are empty");
  reset();
  std::uint32_t current = prompt_tokens.front();
  for (std::size_t index = 1; index < prompt_tokens.size(); ++index) {
    static_cast<void>(step(current, exact_vocabulary));
    current = prompt_tokens[index];
  }
  std::vector<NativeTokenMetrics> output;
  output.reserve(max_tokens);
  for (std::size_t index = 0; index < max_tokens; ++index) {
    output.push_back(step(current, exact_vocabulary));
    current = output.back().token;
  }
  return output;
}

}  // namespace engram
