#include "engram/package.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string_view>
#include <variant>

namespace engram {
namespace {

using JsonObject = std::map<std::string, struct Json>;
using JsonArray = std::vector<struct Json>;

struct Json {
  using Value = std::variant<std::nullptr_t, bool, double, std::string,
                             JsonArray, JsonObject>;
  Value value;
};

class JsonParser {
 public:
  explicit JsonParser(std::string_view input) : input_(input) {}

  Json parse() {
    skip_space();
    Json result = parse_value();
    skip_space();
    if (position_ != input_.size()) {
      fail("trailing content");
    }
    return result;
  }

 private:
  [[noreturn]] void fail(const std::string& message) const {
    throw PackageError("JSON parse error at byte " +
                       std::to_string(position_) + ": " + message);
  }

  void skip_space() {
    while (position_ < input_.size() &&
           (input_[position_] == ' ' || input_[position_] == '\n' ||
            input_[position_] == '\r' || input_[position_] == '\t')) {
      ++position_;
    }
  }

  bool consume(const char value) {
    if (position_ < input_.size() && input_[position_] == value) {
      ++position_;
      return true;
    }
    return false;
  }

  Json parse_value() {
    if (position_ >= input_.size()) {
      fail("expected a value");
    }
    switch (input_[position_]) {
      case '{':
        return Json{parse_object()};
      case '[':
        return Json{parse_array()};
      case '"':
        return Json{parse_string()};
      case 't':
        parse_literal("true");
        return Json{true};
      case 'f':
        parse_literal("false");
        return Json{false};
      case 'n':
        parse_literal("null");
        return Json{nullptr};
      default:
        if (input_[position_] == '-' ||
            (input_[position_] >= '0' && input_[position_] <= '9')) {
          return Json{parse_number()};
        }
        fail("unexpected character");
    }
  }

  void parse_literal(const std::string_view literal) {
    if (input_.substr(position_, literal.size()) != literal) {
      fail("invalid literal");
    }
    position_ += literal.size();
  }

  JsonObject parse_object() {
    consume('{');
    skip_space();
    JsonObject result;
    if (consume('}')) {
      return result;
    }
    while (true) {
      if (position_ >= input_.size() || input_[position_] != '"') {
        fail("object key must be a string");
      }
      std::string key = parse_string();
      skip_space();
      if (!consume(':')) {
        fail("expected ':' after object key");
      }
      skip_space();
      Json value = parse_value();
      if (!result.emplace(std::move(key), std::move(value)).second) {
        fail("duplicate object key");
      }
      skip_space();
      if (consume('}')) {
        return result;
      }
      if (!consume(',')) {
        fail("expected ',' or '}'");
      }
      skip_space();
    }
  }

  JsonArray parse_array() {
    consume('[');
    skip_space();
    JsonArray result;
    if (consume(']')) {
      return result;
    }
    while (true) {
      result.push_back(parse_value());
      skip_space();
      if (consume(']')) {
        return result;
      }
      if (!consume(',')) {
        fail("expected ',' or ']'");
      }
      skip_space();
    }
  }

  static void append_utf8(std::string& output, const std::uint32_t codepoint) {
    if (codepoint <= 0x7FU) {
      output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FFU) {
      output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0xFFFFU) {
      output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
      output.push_back(
          static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
      output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
      output.push_back(
          static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
      output.push_back(
          static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
  }

  std::uint32_t parse_hex_quad() {
    if (position_ + 4 > input_.size()) {
      fail("truncated unicode escape");
    }
    std::uint32_t value = 0;
    for (int index = 0; index < 4; ++index) {
      const char digit = input_[position_++];
      value <<= 4U;
      if (digit >= '0' && digit <= '9') {
        value |= static_cast<std::uint32_t>(digit - '0');
      } else if (digit >= 'a' && digit <= 'f') {
        value |= static_cast<std::uint32_t>(digit - 'a' + 10);
      } else if (digit >= 'A' && digit <= 'F') {
        value |= static_cast<std::uint32_t>(digit - 'A' + 10);
      } else {
        fail("invalid unicode escape");
      }
    }
    return value;
  }

  std::string parse_string() {
    consume('"');
    std::string result;
    while (position_ < input_.size()) {
      const unsigned char character =
          static_cast<unsigned char>(input_[position_++]);
      if (character == '"') {
        return result;
      }
      if (character < 0x20U) {
        fail("unescaped control character in string");
      }
      if (character != '\\') {
        result.push_back(static_cast<char>(character));
        continue;
      }
      if (position_ >= input_.size()) {
        fail("truncated string escape");
      }
      const char escaped = input_[position_++];
      switch (escaped) {
        case '"':
        case '\\':
        case '/':
          result.push_back(escaped);
          break;
        case 'b':
          result.push_back('\b');
          break;
        case 'f':
          result.push_back('\f');
          break;
        case 'n':
          result.push_back('\n');
          break;
        case 'r':
          result.push_back('\r');
          break;
        case 't':
          result.push_back('\t');
          break;
        case 'u': {
          std::uint32_t codepoint = parse_hex_quad();
          if (codepoint >= 0xD800U && codepoint <= 0xDBFFU) {
            if (position_ + 2 > input_.size() || input_[position_] != '\\' ||
                input_[position_ + 1] != 'u') {
              fail("high surrogate lacks low surrogate");
            }
            position_ += 2;
            const std::uint32_t low = parse_hex_quad();
            if (low < 0xDC00U || low > 0xDFFFU) {
              fail("invalid low surrogate");
            }
            codepoint = 0x10000U + ((codepoint - 0xD800U) << 10U) +
                        (low - 0xDC00U);
          } else if (codepoint >= 0xDC00U && codepoint <= 0xDFFFU) {
            fail("unexpected low surrogate");
          }
          append_utf8(result, codepoint);
          break;
        }
        default:
          fail("invalid string escape");
      }
    }
    fail("unterminated string");
  }

  double parse_number() {
    const std::size_t start = position_;
    consume('-');
    if (position_ >= input_.size()) {
      fail("truncated number");
    }
    if (consume('0')) {
      if (position_ < input_.size() && input_[position_] >= '0' &&
          input_[position_] <= '9') {
        fail("leading zero in number");
      }
    } else {
      if (input_[position_] < '1' || input_[position_] > '9') {
        fail("invalid integer part");
      }
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
    }
    if (consume('.')) {
      const std::size_t fraction = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
      if (fraction == position_) {
        fail("fraction has no digits");
      }
    }
    if (position_ < input_.size() &&
        (input_[position_] == 'e' || input_[position_] == 'E')) {
      ++position_;
      if (position_ < input_.size() &&
          (input_[position_] == '+' || input_[position_] == '-')) {
        ++position_;
      }
      const std::size_t exponent = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
      if (exponent == position_) {
        fail("exponent has no digits");
      }
    }
    const std::string text(input_.substr(start, position_ - start));
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(text.c_str(), &end);
    if (errno == ERANGE || end != text.c_str() + text.size() ||
        !std::isfinite(value)) {
      fail("number is outside finite range");
    }
    return value;
  }

  std::string_view input_;
  std::size_t position_ = 0;
};

const JsonObject& object(const Json& value, const std::string_view context) {
  const auto* result = std::get_if<JsonObject>(&value.value);
  if (result == nullptr) {
    throw PackageError(std::string(context) + " must be an object");
  }
  return *result;
}

const Json& field(const JsonObject& value, const std::string_view name,
                  const std::string_view context) {
  const auto found = value.find(std::string(name));
  if (found == value.end()) {
    throw PackageError(std::string(context) + " is missing field '" +
                       std::string(name) + "'");
  }
  return found->second;
}

std::string string_field(const JsonObject& value, const std::string_view name,
                         const std::string_view context) {
  const auto* result =
      std::get_if<std::string>(&field(value, name, context).value);
  if (result == nullptr) {
    throw PackageError(std::string(context) + "." + std::string(name) +
                       " must be a string");
  }
  return *result;
}

bool bool_field(const JsonObject& value, const std::string_view name,
                const std::string_view context) {
  const auto* result = std::get_if<bool>(&field(value, name, context).value);
  if (result == nullptr) {
    throw PackageError(std::string(context) + "." + std::string(name) +
                       " must be a boolean");
  }
  return *result;
}

double number_field(const JsonObject& value, const std::string_view name,
                    const std::string_view context) {
  const auto* result = std::get_if<double>(&field(value, name, context).value);
  if (result == nullptr) {
    throw PackageError(std::string(context) + "." + std::string(name) +
                       " must be numeric");
  }
  return *result;
}

std::size_t size_field(const JsonObject& value, const std::string_view name,
                       const std::string_view context,
                       const bool allow_zero = false) {
  const double number = number_field(value, name, context);
  if (number < 0.0 || std::floor(number) != number ||
      static_cast<long double>(number) >
          static_cast<long double>(std::numeric_limits<std::size_t>::max()) ||
      (!allow_zero && number == 0.0)) {
    throw PackageError(std::string(context) + "." + std::string(name) +
                       " must be a representable " +
                       (allow_zero ? "non-negative" : "positive") +
                       " integer");
  }
  return static_cast<std::size_t>(number);
}

Json read_json(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw PackageError("cannot open JSON file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  if (!stream.good() && !stream.eof()) {
    throw PackageError("cannot read JSON file: " + path.string());
  }
  return JsonParser(buffer.str()).parse();
}

bool valid_digest(const std::string& digest) {
  if (digest.size() != 64) {
    return false;
  }
  return std::all_of(digest.begin(), digest.end(), [](const char value) {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
  });
}

bool safe_relative_path(const std::string& text) {
  const std::filesystem::path path(text);
  if (text.empty() || path.is_absolute()) {
    return false;
  }
  for (const auto& component : path) {
    if (component == "." || component == ".." || component.empty()) {
      return false;
    }
  }
  return path.generic_string() == text;
}

constexpr std::array<std::uint32_t, 64> kShaConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

std::uint32_t rotate_right(const std::uint32_t value,
                           const unsigned count) noexcept {
  return (value >> count) | (value << (32U - count));
}

class Sha256 {
 public:
  void update(const std::uint8_t* data, std::size_t size) {
    total_bytes_ += size;
    while (size > 0) {
      const std::size_t count = std::min(size, block_.size() - buffered_);
      std::copy_n(data, count, block_.begin() + buffered_);
      buffered_ += count;
      data += count;
      size -= count;
      if (buffered_ == block_.size()) {
        transform(block_.data());
        buffered_ = 0;
      }
    }
  }

  std::array<std::uint8_t, 32> finish() {
    const std::uint64_t bit_count = total_bytes_ * 8U;
    block_[buffered_++] = 0x80U;
    if (buffered_ > 56) {
      std::fill(block_.begin() + buffered_, block_.end(), 0U);
      transform(block_.data());
      buffered_ = 0;
    }
    std::fill(block_.begin() + buffered_, block_.begin() + 56, 0U);
    for (std::size_t index = 0; index < 8; ++index) {
      block_[63 - index] =
          static_cast<std::uint8_t>(bit_count >> (index * 8U));
    }
    transform(block_.data());
    std::array<std::uint8_t, 32> digest{};
    for (std::size_t word = 0; word < state_.size(); ++word) {
      for (std::size_t byte = 0; byte < 4; ++byte) {
        digest[word * 4 + byte] = static_cast<std::uint8_t>(
            state_[word] >> ((3U - byte) * 8U));
      }
    }
    return digest;
  }

 private:
  void transform(const std::uint8_t* block) {
    std::array<std::uint32_t, 64> schedule{};
    for (std::size_t index = 0; index < 16; ++index) {
      schedule[index] =
          (static_cast<std::uint32_t>(block[index * 4]) << 24U) |
          (static_cast<std::uint32_t>(block[index * 4 + 1]) << 16U) |
          (static_cast<std::uint32_t>(block[index * 4 + 2]) << 8U) |
          static_cast<std::uint32_t>(block[index * 4 + 3]);
    }
    for (std::size_t index = 16; index < schedule.size(); ++index) {
      const std::uint32_t s0 =
          rotate_right(schedule[index - 15], 7) ^
          rotate_right(schedule[index - 15], 18) ^
          (schedule[index - 15] >> 3U);
      const std::uint32_t s1 =
          rotate_right(schedule[index - 2], 17) ^
          rotate_right(schedule[index - 2], 19) ^
          (schedule[index - 2] >> 10U);
      schedule[index] = schedule[index - 16] + s0 +
                        schedule[index - 7] + s1;
    }
    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t index = 0; index < schedule.size(); ++index) {
      const std::uint32_t sigma1 =
          rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
      const std::uint32_t choice = (e & f) ^ (~e & g);
      const std::uint32_t temporary1 =
          h + sigma1 + choice + kShaConstants[index] + schedule[index];
      const std::uint32_t sigma0 =
          rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = sigma0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_ = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<std::uint8_t, 64> block_{};
  std::size_t buffered_ = 0;
  std::uint64_t total_bytes_ = 0;
};

std::string digest_hex(const std::array<std::uint8_t, 32>& digest) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const std::uint8_t byte : digest) {
    output << std::setw(2) << static_cast<unsigned>(byte);
  }
  return output.str();
}

void require_file(const std::set<std::string>& files,
                  const std::string& required) {
  if (!files.contains(required)) {
    throw PackageError("required package file is absent from manifest: " +
                       required);
  }
}

}  // namespace

std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw PackageError("cannot open file for checksum: " + path.string());
  }
  Sha256 hash;
  std::array<char, 64 * 1024> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0) {
      hash.update(reinterpret_cast<const std::uint8_t*>(buffer.data()),
                  static_cast<std::size_t>(count));
    }
  }
  if (!input.eof()) {
    throw PackageError("failed while checksumming file: " + path.string());
  }
  return digest_hex(hash.finish());
}

PackageMetadata load_package(const std::filesystem::path& root,
                             const bool verify_checksums) {
  if (!std::filesystem::is_directory(root)) {
    throw PackageError("package root is not a directory: " + root.string());
  }
  const Json manifest_json = read_json(root / "manifest.json");
  const JsonObject& manifest = object(manifest_json, "manifest");
  PackageMetadata result;
  result.root = root;
  result.format = string_field(manifest, "format", "manifest");
  result.version = size_field(manifest, "version", "manifest");
  result.engram_version =
      string_field(manifest, "engram_version", "manifest");
  result.source_model_hash =
      string_field(manifest, "source_model_hash", "manifest");
  result.source_architecture =
      string_field(manifest, "source_architecture", "manifest");
  result.fixture_only = bool_field(manifest, "fixture_only", "manifest");
  result.source_transformer_independent = bool_field(
      manifest, "does_not_require_source_transformer", "manifest");
  result.dimensions.hidden_size =
      size_field(manifest, "hidden_size", "manifest");
  result.dimensions.vocabulary_size =
      size_field(manifest, "vocab_size", "manifest");
  result.dimensions.semantic_layers =
      size_field(manifest, "num_semantic_layers", "manifest");
  if (result.format != "engram-model" || result.version != 1) {
    throw PackageError("unsupported Engram package format or version");
  }
  if (result.engram_version.empty() || result.source_model_hash.empty() ||
      result.source_architecture.empty()) {
    throw PackageError("package identity strings must not be empty");
  }
  if (!result.source_transformer_independent) {
    throw PackageError("native runtime requires a source-independent package");
  }

  const JsonObject& runtime =
      object(field(manifest, "runtime", "manifest"), "manifest.runtime");
  result.policies.cycles = size_field(runtime, "cycles", "manifest.runtime");
  result.policies.semantic_top_k =
      size_field(runtime, "semantic_top_k", "manifest.runtime");
  result.policies.semantic_candidates =
      size_field(runtime, "semantic_candidates", "manifest.runtime");
  result.policies.semantic_ivf_clusters =
      size_field(runtime, "semantic_ivf_clusters", "manifest.runtime");
  result.policies.semantic_ivf_probes =
      size_field(runtime, "semantic_ivf_probes", "manifest.runtime");
  result.policies.vocabulary_candidates =
      size_field(runtime, "vocabulary_candidates", "manifest.runtime");
  result.policies.vocabulary_ivf_clusters =
      size_field(runtime, "vocabulary_ivf_clusters", "manifest.runtime");
  result.policies.vocabulary_ivf_probes =
      size_field(runtime, "vocabulary_ivf_probes", "manifest.runtime");
  if (result.policies.semantic_top_k >
          result.policies.semantic_candidates ||
      result.policies.semantic_ivf_probes == 0 ||
      result.policies.semantic_ivf_probes >
          result.policies.semantic_ivf_clusters ||
      result.policies.vocabulary_candidates == 0 ||
      result.policies.vocabulary_ivf_probes == 0 ||
      result.policies.vocabulary_ivf_probes >
          result.policies.vocabulary_ivf_clusters) {
    throw PackageError("invalid semantic candidate or IVF policy");
  }

  const JsonObject& file_object =
      object(field(manifest, "files", "manifest"), "manifest.files");
  std::set<std::string> listed_files;
  result.files.reserve(file_object.size());
  for (const auto& [relative, descriptor_json] : file_object) {
    if (!safe_relative_path(relative) || !listed_files.insert(relative).second) {
      throw PackageError("unsafe or duplicate package path: " + relative);
    }
    const JsonObject& descriptor =
        object(descriptor_json, "manifest.files." + relative);
    PackageFile package_file{
        relative,
        size_field(descriptor, "bytes", "manifest.files." + relative, true),
        string_field(descriptor, "sha256", "manifest.files." + relative)};
    if (!valid_digest(package_file.sha256)) {
      throw PackageError("invalid SHA-256 descriptor: " + relative);
    }
    const std::filesystem::path file_path = root / relative;
    if (!std::filesystem::is_regular_file(file_path)) {
      throw PackageError("listed package file is missing: " + relative);
    }
    std::error_code size_error;
    const std::uintmax_t actual_bytes =
        std::filesystem::file_size(file_path, size_error);
    if (size_error || actual_bytes != package_file.bytes) {
      throw PackageError("package file size mismatch: " + relative);
    }
    if (verify_checksums && sha256_file(file_path) != package_file.sha256) {
      throw PackageError("package file checksum mismatch: " + relative);
    }
    result.files.push_back(std::move(package_file));
  }

  constexpr std::array<std::string_view, 7> required = {
      "controller/metadata.json", "embeddings/token_embeddings.npy",
      "vocabulary/embeddings.npy", "semantic/manifest.json",
      "episodic/config.json", "transitions/config.json",
      "corrections/capsules.json"};
  for (const std::string_view path : required) {
    require_file(listed_files, std::string(path));
  }
  require_file(listed_files, "vocabulary/index.npy");
  require_file(listed_files, "vocabulary/ivf/centroids.npy");
  require_file(listed_files, "vocabulary/ivf/posting_offsets.npy");
  require_file(listed_files, "vocabulary/ivf/token_ids.npy");
  require_file(listed_files, "vocabulary/ivf/metadata.json");
  for (std::size_t layer = 0; layer < result.dimensions.semantic_layers;
       ++layer) {
    std::ostringstream prefix;
    prefix << "semantic/layer-" << std::setw(4) << std::setfill('0') << layer
           << "/quantized/ivf/";
    require_file(listed_files, prefix.str() + "centroids.npy");
    require_file(listed_files, prefix.str() + "posting_offsets.npy");
    require_file(listed_files, prefix.str() + "posting_indices.npy");
    require_file(listed_files, prefix.str() + "metadata.json");
  }

  const Json controller_json = read_json(root / "controller/metadata.json");
  const JsonObject& controller =
      object(controller_json, "controller metadata");
  if (string_field(controller, "format", "controller metadata") !=
          "engram.controller.shared_gru" ||
      size_field(controller, "schema_version", "controller metadata") != 1 ||
      string_field(controller, "operator", "controller metadata") !=
          "shared_gru_stage_adapter") {
    throw PackageError("unsupported controller metadata");
  }
  result.dimensions.controller_input_size =
      size_field(controller, "input_dim", "controller metadata");
  const std::size_t state_size =
      size_field(controller, "state_dim", "controller metadata");
  result.dimensions.controller_stages =
      size_field(controller, "num_stages", "controller metadata");
  result.dimensions.controller_adapter_rank =
      size_field(controller, "adapter_rank", "controller metadata");
  if (state_size > std::numeric_limits<std::size_t>::max() / 3 ||
      state_size != result.dimensions.hidden_size ||
      result.dimensions.controller_input_size != 3 * state_size ||
      result.dimensions.controller_stages !=
          result.dimensions.semantic_layers) {
    throw PackageError("controller and package dimensions disagree");
  }
  const JsonObject& tensor_layout = object(
      field(controller, "tensor_layout", "controller metadata"),
      "controller metadata.tensor_layout");
  if (tensor_layout.empty()) {
    throw PackageError("controller tensor layout must not be empty");
  }
  for (const auto& [name, unused] : tensor_layout) {
    static_cast<void>(unused);
    require_file(listed_files, "controller/" + name + ".npy");
  }

  const Json episodic_json = read_json(root / "episodic/config.json");
  const JsonObject& episodic = object(episodic_json, "episodic config");
  result.policies.local_window =
      size_field(episodic, "local_window", "episodic config", true);
  result.policies.retrieval_capacity =
      size_field(episodic, "retrieval_capacity", "episodic config", true);
  result.policies.retrieval_candidates =
      size_field(episodic, "retrieval_candidates", "episodic config");
  result.policies.retrieval_top_k =
      size_field(episodic, "retrieval_top_k", "episodic config");
  result.policies.recurrent_decay =
      number_field(episodic, "decay", "episodic config");
  if (result.policies.retrieval_top_k >
          result.policies.retrieval_candidates ||
      result.policies.recurrent_decay < 0.0 ||
      result.policies.recurrent_decay > 1.0) {
    throw PackageError("invalid episodic runtime policy");
  }

  const Json transitions_json = read_json(root / "transitions/config.json");
  const JsonObject& transitions =
      object(transitions_json, "transition config");
  result.policies.transition_capacity =
      size_field(transitions, "capacity", "transition config", true);
  result.policies.transition_similarity_radius =
      number_field(transitions, "similarity_radius", "transition config");
  if (result.policies.transition_similarity_radius < 0.0 ||
      !std::isfinite(result.policies.transition_similarity_radius)) {
    throw PackageError("transition similarity radius must be finite and non-negative");
  }

  const Json corrections_json = read_json(root / "corrections/capsules.json");
  const JsonObject& corrections =
      object(corrections_json, "correction config");
  static_cast<void>(size_field(corrections, "version", "correction config"));
  result.policies.correction_fallback =
      string_field(corrections, "fallback", "correction config");
  if (result.policies.correction_fallback.empty()) {
    throw PackageError("correction fallback must not be empty");
  }
  return result;
}

}  // namespace engram
