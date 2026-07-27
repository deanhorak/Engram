#include "engram/package.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

void write(const std::filesystem::path& path, const std::string& content) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary);
  stream << content;
  if (!stream) {
    throw std::runtime_error("native BitNet package fixture write failed");
  }
}

std::string quote(const std::string& value) { return "\"" + value + "\""; }

int fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}

struct Fixture {
  std::filesystem::path root;
  engram::NativeBitNetDIPTrustRoot trust;
};

Fixture make_fixture(const std::filesystem::path& root,
                     const bool dense_fallback = false,
                     const bool include_second_eos = true,
                     const std::size_t local_window = 16,
                     const std::string& mlp_mode =
                         "native_bitnet_dynamic_input_pruning_v2") {
  std::filesystem::remove_all(root);
  const std::string controller_metadata =
      R"({"format":"engram.controller.factorized_residual","schema_version":3,"operator":"operator_residual_with_factorized_correction","input_dim":24,"state_dim":8,"num_stages":2,"serialized_bytes":16,"tensor_layout":{"operator_residual_scale":[2,2],"step_scale":[2]}})";
  const std::string model_config =
      R"({"architectures":["BitNetForCausalLM"],"eos_token_id":128001,"hidden_act":"relu2","hidden_size":8,"intermediate_size":12,"max_position_embeddings":64,"model_type":"bitnet","rms_norm_eps":1e-5,"num_attention_heads":2,"num_hidden_layers":2,"num_key_value_heads":1,"rope_theta":500000.0,"tie_word_embeddings":true,"torch_dtype":"bfloat16","vocab_size":128256,"quantization_config":{"quant_method":"bitnet","quantization_mode":"offline"}})";
  const std::string generation_config =
      include_second_eos
          ? R"({"eos_token_id":[128001,128009]})"
          : R"({"eos_token_id":[128001]})";
  const std::map<std::string, std::string> contents = {
      {"config/config.json", model_config},
      {"controller/metadata.json", controller_metadata},
      {"controller/operator_residual_scale.npy", "operator"},
      {"controller/step_scale.npy", "steps"},
      {"mlp/model.bitnet-dip-index.bin", "coordinate-index"},
      {"mlp/model.bitnet-records.bin", "record-artifact"},
      {"tokenizer/generation_config.json", generation_config},
      {"transformer/non_mlp.safetensors", "non-mlp"},
  };
  for (const auto& [relative, content] : contents) {
    write(root / relative, content);
  }

  std::string files_json;
  for (const auto& [relative, content] : contents) {
    if (!files_json.empty()) files_json += ',';
    files_json += quote(relative) + ":{\"bytes\":" +
                  std::to_string(content.size()) + ",\"sha256\":" +
                  quote(engram::sha256_file(root / relative)) + "}";
  }
  const std::string base_sha =
      engram::sha256_file(root / "mlp/model.bitnet-records.bin");
  const std::string index_sha =
      engram::sha256_file(root / "mlp/model.bitnet-dip-index.bin");
  const std::string controller_sha =
      engram::sha256_file(root / "controller/metadata.json");
  const std::string source_manifest_sha(64, '1');
  const std::string policy_sha(64, '2');
  const std::string adjudication_sha(64, '3');
  const std::string manifest =
      R"({"format":"engram-native-bitnet","version":1,"engram_version":"0.1.0","does_not_require_source_transformer":true,)"
      R"("model":{"hidden_size":8,"intermediate_size":12,"num_hidden_layers":2,"rms_norm_eps":1e-5},)"
      R"("runtime":{"device":"cpu","dtype":"bfloat16","kernel_threads":4,"mlp_mode":)" +
      quote(mlp_mode) +
      R"(,"attention_mode":"native_streaming_w16_c8_k4_sinks2","attention_policy":{"local_window":)" +
      std::to_string(local_window) +
      R"(,"older_candidates":8,"older_top_k":4,"sink_tokens":2}},)"
      R"("mlp":{"path":"mlp/model.bitnet-records.bin","encoding":"native_bitnet_phase_base3_v1","serialized_bytes":)" +
      std::to_string(contents.at("mlp/model.bitnet-records.bin").size()) +
      R"(,"sha256":)" + quote(base_sha) +
      R"(,"dense_weight_materialization_bytes":0},)"
      R"("semantic_memory":{"operator":"native_bitnet_dynamic_input_pruning_v2","runtime_scope":"native_token_runtime","path":"mlp/model.bitnet-dip-index.bin","format":"engram-native-bitnet-dip-index","version":2,"sha256":)" +
      quote(index_sha) + R"(,"serialized_bytes":)" +
      std::to_string(contents.at("mlp/model.bitnet-dip-index.bin").size()) +
      R"(,"source_artifact_sha256":)" + quote(base_sha) +
      R"(,"source_package_manifest_sha256":)" + quote(source_manifest_sha) +
      R"(,"runtime_policy":"embedded_authenticated_layer_headers","policy_manifest_sha256":)" +
      quote(policy_sha) + R"(,"adjudication_sha256":)" +
      quote(adjudication_sha) +
      R"(,"adjudication_decision":"milestone_2_semantic_gate_passed_by_postmortem_adjudication","all_mlp_layers_substituted":true,"dense_fallback":)" +
      (dense_fallback ? "true" : "false") +
      R"(,"cpu_only":true,"traffic_accounting":"modelled_cache_line_v2"},)"
      R"("transformer":{"non_mlp_path":"transformer/non_mlp.safetensors","packaged_bytes":)" +
      std::to_string(contents.at("transformer/non_mlp.safetensors").size()) +
      R"(},)"
      R"("controller":{"path":"controller","format":"engram.controller.factorized_residual","schema_version":3,"operator":"operator_residual_with_factorized_correction","metadata_sha256":)" +
      quote(controller_sha) +
      R"(,"serialized_bytes":16,"correction_enabled":false},)"
      R"("tokenizer":{"path":"tokenizer","files":["generation_config.json"]},"files":{)" +
      files_json + "}}";
  write(root / "manifest.json", manifest);
  return Fixture{
      .root = root,
      .trust =
          {
              .package_manifest_bytes =
                  std::filesystem::file_size(root / "manifest.json"),
              .package_manifest_sha256 =
                  engram::sha256_file(root / "manifest.json"),
              .source_package_manifest_sha256 = source_manifest_sha,
              .source_artifact_sha256 = base_sha,
              .coordinate_index_sha256 = index_sha,
              .policy_manifest_sha256 = policy_sha,
              .adjudication_sha256 = adjudication_sha,
          },
  };
}

bool rejected(const std::function<void()>& operation) {
  try {
    operation();
  } catch (const engram::PackageError&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "engram-native-bitnet-dip-package-test";
  std::filesystem::remove_all(root);
  try {
    Fixture fixture = make_fixture(root);
    const engram::NativeBitNetDIPPackageMetadata package =
        engram::load_native_bitnet_dip_package(fixture.root, fixture.trust);
    if (package.layers != 2 || package.hidden_size != 8 ||
        package.intermediate_size != 12 || package.vocabulary_size != 128256 ||
        package.query_heads != 2 || package.key_value_heads != 1 ||
        package.head_dimension != 4 || package.max_position_embeddings != 64 ||
        package.kernel_threads != 4 ||
        package.local_window != 16 || package.older_candidates != 8 ||
        package.older_top_k != 4 || package.sink_tokens != 2 ||
        package.rope_theta != 500000.0F ||
        package.eos_token_ids !=
            std::vector<std::int64_t>({128001, 128009}) ||
        package.files.size() != 8) {
      return fail("native BitNet package metadata mismatch");
    }

    write(root / "unlisted.bin", "not authenticated");
    if (!rejected([&] {
          static_cast<void>(engram::load_native_bitnet_dip_package(
              fixture.root, fixture.trust));
        })) {
      return fail("unlisted native BitNet package file was accepted");
    }

    fixture = make_fixture(root);
    std::filesystem::remove(root / "transformer/non_mlp.safetensors");
    std::filesystem::create_symlink("../mlp/model.bitnet-records.bin",
                                    root / "transformer/non_mlp.safetensors");
    if (!rejected([&] {
          static_cast<void>(engram::load_native_bitnet_dip_package(
              fixture.root, fixture.trust));
        })) {
      return fail("symlinked native BitNet package file was accepted");
    }

    fixture = make_fixture(root, true);
    if (!rejected([&] {
          static_cast<void>(engram::load_native_bitnet_dip_package(
              fixture.root, fixture.trust));
        })) {
      return fail("dense semantic fallback was accepted");
    }

    fixture = make_fixture(root, false, false);
    if (!rejected([&] {
          static_cast<void>(engram::load_native_bitnet_dip_package(
              fixture.root, fixture.trust));
        })) {
      return fail("incomplete authenticated EOS set was accepted");
    }

    fixture = make_fixture(root, false, true, 15);
    if (!rejected([&] {
          static_cast<void>(engram::load_native_bitnet_dip_package(
              fixture.root, fixture.trust));
        })) {
      return fail("unsupported authenticated attention policy was accepted");
    }

    fixture = make_fixture(root, false, true, 16, "dense_reference");
    if (!rejected([&] {
          static_cast<void>(engram::load_native_bitnet_dip_package(
              fixture.root, fixture.trust));
        })) {
      return fail("unsupported authenticated MLP mode was accepted");
    }

    fixture = make_fixture(root);
    write(root / "config/config.json",
          std::string(std::filesystem::file_size(root / "config/config.json"),
                      'x'));
    if (!rejected([&] {
          static_cast<void>(engram::load_native_bitnet_dip_package(
              fixture.root, fixture.trust));
        })) {
      return fail("tampered authenticated package file was accepted");
    }

    fixture = make_fixture(root);
    engram::NativeBitNetDIPTrustRoot wrong_trust = fixture.trust;
    wrong_trust.adjudication_sha256 = std::string(64, '4');
    if (!rejected([&] {
          static_cast<void>(engram::load_native_bitnet_dip_package(
              fixture.root, wrong_trust));
        })) {
      return fail("untrusted semantic promotion was accepted");
    }
    if (!rejected([&] {
          static_cast<void>(
              engram::load_native_bitnet_dip_package(fixture.root));
        })) {
      return fail("fixture bypassed production semantic trust root");
    }

    std::filesystem::remove_all(root);
  } catch (const std::exception& error) {
    std::filesystem::remove_all(root);
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
