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

const JsonArray& array(const Json& value, const std::string_view context) {
  const auto* result = std::get_if<JsonArray>(&value.value);
  if (result == nullptr) {
    throw PackageError(std::string(context) + " must be an array");
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

std::int64_t int64_value(const Json& value, const std::string_view context,
                         const bool allow_negative = false) {
  const auto* number = std::get_if<double>(&value.value);
  if (number == nullptr || std::floor(*number) != *number ||
      *number < -9223372036854775808.0 ||
      *number >= 9223372036854775808.0 ||
      (!allow_negative && *number < 0.0)) {
    throw PackageError(std::string(context) +
                       " must be a representable non-negative integer");
  }
  return static_cast<std::int64_t>(*number);
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

const PackageFile& package_file(
    const std::map<std::string, PackageFile>& files,
    const std::string& relative, const std::string_view context) {
  const auto found = files.find(relative);
  if (found == files.end()) {
    throw PackageError(std::string(context) +
                       " names a file absent from the package inventory: " +
                       relative);
  }
  return found->second;
}

std::filesystem::path safe_package_path(
    const std::filesystem::path& root, const std::string& relative,
    const std::string_view context) {
  if (!safe_relative_path(relative)) {
    throw PackageError(std::string(context) +
                       " contains an unsafe package path: " + relative);
  }
  return root / std::filesystem::path(relative);
}

std::map<std::string, PackageFile> load_exact_inventory(
    const std::filesystem::path& root, const JsonObject& manifest) {
  const JsonObject& file_object =
      object(field(manifest, "files", "manifest"), "manifest.files");
  if (file_object.empty()) {
    throw PackageError("native BitNet package inventory must not be empty");
  }

  std::map<std::string, PackageFile> listed;
  for (const auto& [relative, descriptor_json] : file_object) {
    if (!safe_relative_path(relative) || relative == "manifest.json") {
      throw PackageError("unsafe native BitNet package path: " + relative);
    }
    const JsonObject& descriptor =
        object(descriptor_json, "manifest.files." + relative);
    PackageFile package_file_descriptor{
        relative,
        size_field(descriptor, "bytes", "manifest.files." + relative, true),
        string_field(descriptor, "sha256", "manifest.files." + relative)};
    if (!valid_digest(package_file_descriptor.sha256) ||
        !listed.emplace(relative, package_file_descriptor).second) {
      throw PackageError("invalid native BitNet file descriptor: " +
                         relative);
    }

    const std::filesystem::path path = root / relative;
    std::error_code status_error;
    const std::filesystem::file_status status =
        std::filesystem::symlink_status(path, status_error);
    if (status_error || status.type() != std::filesystem::file_type::regular) {
      throw PackageError(
          "listed native BitNet package file is missing or not regular: " +
          relative);
    }
    std::error_code size_error;
    const std::uintmax_t actual_bytes =
        std::filesystem::file_size(path, size_error);
    if (size_error || actual_bytes != package_file_descriptor.bytes) {
      throw PackageError("native BitNet package file size mismatch: " +
                         relative);
    }
    if (sha256_file(path) != package_file_descriptor.sha256) {
      throw PackageError("native BitNet package file checksum mismatch: " +
                         relative);
    }
  }

  std::set<std::string> actual;
  try {
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(root)) {
      const std::filesystem::file_status status = entry.symlink_status();
      const std::string relative =
          entry.path().lexically_relative(root).generic_string();
      if (status.type() == std::filesystem::file_type::symlink) {
        throw PackageError(
            "native BitNet package contains a symlink: " + relative);
      }
      if (status.type() == std::filesystem::file_type::directory) {
        continue;
      }
      if (status.type() != std::filesystem::file_type::regular) {
        throw PackageError(
            "native BitNet package contains a non-regular entry: " +
            relative);
      }
      if (relative != "manifest.json") {
        actual.insert(relative);
      }
    }
  } catch (const std::filesystem::filesystem_error& error) {
    throw PackageError("cannot enumerate native BitNet package: " +
                       std::string(error.what()));
  }
  std::set<std::string> expected;
  for (const auto& [relative, unused] : listed) {
    static_cast<void>(unused);
    expected.insert(relative);
  }
  if (actual != expected) {
    throw PackageError("native BitNet package inventory is not exact");
  }
  return listed;
}

std::string authenticated_file_path(
    const std::filesystem::path& root,
    const std::map<std::string, PackageFile>& files,
    const JsonObject& descriptor, const std::string_view path_name,
    const std::string_view context) {
  const std::string relative = string_field(descriptor, path_name, context);
  static_cast<void>(safe_package_path(root, relative, context));
  static_cast<void>(package_file(files, relative, context));
  return relative;
}

void require_file_binding(const std::map<std::string, PackageFile>& files,
                          const std::string& relative,
                          const JsonObject& descriptor,
                          const std::string_view context) {
  const PackageFile& inventory = package_file(files, relative, context);
  if (size_field(descriptor, "serialized_bytes", context, true) !=
          inventory.bytes ||
      string_field(descriptor, "sha256", context) != inventory.sha256) {
    throw PackageError(std::string(context) +
                       " disagrees with the package inventory");
  }
}

float positive_float_field(const JsonObject& object_value,
                           const std::string_view name,
                           const std::string_view context) {
  const double value = number_field(object_value, name, context);
  if (!(value > 0.0) ||
      value > static_cast<double>(std::numeric_limits<float>::max())) {
    throw PackageError(std::string(context) + "." + std::string(name) +
                       " must be a positive finite float");
  }
  return static_cast<float>(value);
}

std::vector<std::string> string_array_field(
    const JsonObject& object_value, const std::string_view name,
    const std::string_view context) {
  const JsonArray& values =
      array(field(object_value, name, context),
            std::string(context) + "." + std::string(name));
  std::vector<std::string> result;
  result.reserve(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    const auto* value = std::get_if<std::string>(&values[index].value);
    if (value == nullptr) {
      throw PackageError(std::string(context) + "." + std::string(name) +
                         "[" + std::to_string(index) +
                         "] must be a string");
    }
    result.push_back(*value);
  }
  return result;
}

std::vector<std::int64_t> eos_token_ids(
    const JsonObject& generation_config) {
  const Json& eos =
      field(generation_config, "eos_token_id", "generation config");
  std::vector<std::int64_t> result;
  if (const auto* values = std::get_if<JsonArray>(&eos.value)) {
    result.reserve(values->size());
    for (std::size_t index = 0; index < values->size(); ++index) {
      result.push_back(int64_value(
          (*values)[index],
          "generation config.eos_token_id[" + std::to_string(index) + "]"));
    }
  } else {
    result.push_back(
        int64_value(eos, "generation config.eos_token_id"));
  }
  std::set<std::int64_t> unique(result.begin(), result.end());
  if (result.empty() || unique.size() != result.size() ||
      !unique.contains(128001) || !unique.contains(128009)) {
    throw PackageError(
        "generation config EOS ids must uniquely include 128001 and 128009");
  }
  return result;
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

NativeBitNetDIPPackageMetadata load_native_bitnet_dip_package(
    const std::filesystem::path& root,
    const NativeBitNetDIPTrustRoot& trust_root) {
  for (const auto* digest :
       {&trust_root.package_manifest_sha256,
        &trust_root.source_package_manifest_sha256,
        &trust_root.source_artifact_sha256,
        &trust_root.coordinate_index_sha256,
        &trust_root.policy_manifest_sha256,
        &trust_root.adjudication_sha256}) {
    if (!valid_digest(*digest)) {
      throw PackageError("native BitNet DIP trust root is malformed");
    }
  }

  std::error_code root_error;
  const std::filesystem::file_status root_status =
      std::filesystem::symlink_status(root, root_error);
  if (root_error ||
      root_status.type() != std::filesystem::file_type::directory) {
    throw PackageError(
        "native BitNet package root is missing, a symlink, or not a directory: " +
        root.string());
  }
  const std::filesystem::path manifest_path = root / "manifest.json";
  std::error_code manifest_error;
  const std::filesystem::file_status manifest_status =
      std::filesystem::symlink_status(manifest_path, manifest_error);
  if (manifest_error ||
      manifest_status.type() != std::filesystem::file_type::regular) {
    throw PackageError(
        "native BitNet package manifest is missing, a symlink, or not regular");
  }
  std::error_code manifest_size_error;
  if (trust_root.package_manifest_bytes == 0 ||
      std::filesystem::file_size(manifest_path, manifest_size_error) !=
          trust_root.package_manifest_bytes ||
      manifest_size_error ||
      sha256_file(manifest_path) != trust_root.package_manifest_sha256) {
    throw PackageError(
        "native BitNet package manifest does not match the deployment trust "
        "root");
  }

  const Json manifest_json = read_json(manifest_path);
  const JsonObject& manifest = object(manifest_json, "manifest");
  if (string_field(manifest, "format", "manifest") !=
          "engram-native-bitnet" ||
      size_field(manifest, "version", "manifest") != 1 ||
      !bool_field(manifest, "does_not_require_source_transformer",
                  "manifest")) {
    throw PackageError(
        "unsupported or source-dependent native BitNet package");
  }
  if (string_field(manifest, "engram_version", "manifest").empty()) {
    throw PackageError("native BitNet package has no Engram version");
  }
  const std::map<std::string, PackageFile> files =
      load_exact_inventory(root, manifest);

  const JsonObject& runtime =
      object(field(manifest, "runtime", "manifest"), "manifest.runtime");
  const JsonObject& attention_policy =
      object(field(runtime, "attention_policy", "manifest.runtime"),
             "manifest.runtime.attention_policy");
  constexpr std::size_t kLocalWindow = 16;
  constexpr std::size_t kOlderCandidates = 8;
  constexpr std::size_t kOlderTopK = 4;
  constexpr std::size_t kSinkTokens = 2;
  if (string_field(runtime, "device", "manifest.runtime") != "cpu" ||
      string_field(runtime, "dtype", "manifest.runtime") != "bfloat16" ||
      string_field(runtime, "mlp_mode", "manifest.runtime") !=
          "native_bitnet_dynamic_input_pruning_v2" ||
      string_field(runtime, "attention_mode", "manifest.runtime") !=
          "native_streaming_w16_c8_k4_sinks2" ||
      size_field(attention_policy, "local_window",
                 "manifest.runtime.attention_policy") != kLocalWindow ||
      size_field(attention_policy, "older_candidates",
                 "manifest.runtime.attention_policy") != kOlderCandidates ||
      size_field(attention_policy, "older_top_k",
                 "manifest.runtime.attention_policy") != kOlderTopK ||
      size_field(attention_policy, "sink_tokens",
                 "manifest.runtime.attention_policy", true) != kSinkTokens) {
    throw PackageError(
        "native BitNet runtime or bounded-attention policy is unsupported");
  }

  const JsonObject& mlp =
      object(field(manifest, "mlp", "manifest"), "manifest.mlp");
  const std::string mlp_relative =
      authenticated_file_path(root, files, mlp, "path", "manifest.mlp");
  require_file_binding(files, mlp_relative, mlp, "manifest.mlp");
  if (string_field(mlp, "encoding", "manifest.mlp") !=
          "native_bitnet_phase_base3_v1" ||
      size_field(mlp, "dense_weight_materialization_bytes", "manifest.mlp",
                 true) != 0) {
    throw PackageError("unsupported native BitNet MLP artifact");
  }

  const JsonObject& semantic_memory =
      object(field(manifest, "semantic_memory", "manifest"),
             "manifest.semantic_memory");
  const std::string index_relative = authenticated_file_path(
      root, files, semantic_memory, "path", "manifest.semantic_memory");
  require_file_binding(files, index_relative, semantic_memory,
                       "manifest.semantic_memory");
  const std::string index_sha =
      string_field(semantic_memory, "sha256", "manifest.semantic_memory");
  const std::string source_artifact_sha = string_field(
      semantic_memory, "source_artifact_sha256", "manifest.semantic_memory");
  const std::string source_package_manifest_sha =
      string_field(semantic_memory, "source_package_manifest_sha256",
                   "manifest.semantic_memory");
  const std::string policy_manifest_sha =
      string_field(semantic_memory, "policy_manifest_sha256",
                   "manifest.semantic_memory");
  const std::string adjudication_sha =
      string_field(semantic_memory, "adjudication_sha256",
                   "manifest.semantic_memory");
  if (string_field(semantic_memory, "operator",
                   "manifest.semantic_memory") !=
          "native_bitnet_dynamic_input_pruning_v2" ||
      string_field(semantic_memory, "runtime_scope",
                   "manifest.semantic_memory") != "native_token_runtime" ||
      string_field(semantic_memory, "format", "manifest.semantic_memory") !=
          "engram-native-bitnet-dip-index" ||
      size_field(semantic_memory, "version", "manifest.semantic_memory") != 2 ||
      string_field(semantic_memory, "runtime_policy",
                   "manifest.semantic_memory") !=
          "embedded_authenticated_layer_headers" ||
      string_field(semantic_memory, "traffic_accounting",
                   "manifest.semantic_memory") !=
          "modelled_cache_line_v2" ||
      string_field(semantic_memory, "adjudication_decision",
                   "manifest.semantic_memory") !=
          "milestone_2_semantic_gate_passed_by_postmortem_adjudication" ||
      !bool_field(semantic_memory, "all_mlp_layers_substituted",
                  "manifest.semantic_memory") ||
      bool_field(semantic_memory, "dense_fallback",
                 "manifest.semantic_memory") ||
      !bool_field(semantic_memory, "cpu_only", "manifest.semantic_memory") ||
      source_artifact_sha !=
          string_field(mlp, "sha256", "manifest.mlp") ||
      source_package_manifest_sha !=
          trust_root.source_package_manifest_sha256 ||
      source_artifact_sha != trust_root.source_artifact_sha256 ||
      index_sha != trust_root.coordinate_index_sha256 ||
      policy_manifest_sha != trust_root.policy_manifest_sha256 ||
      adjudication_sha != trust_root.adjudication_sha256) {
    throw PackageError(
        "native BitNet semantic-memory promotion binding is unsupported");
  }

  const JsonObject& transformer =
      object(field(manifest, "transformer", "manifest"),
             "manifest.transformer");
  const std::string non_mlp_relative = authenticated_file_path(
      root, files, transformer, "non_mlp_path", "manifest.transformer");
  if (size_field(transformer, "packaged_bytes", "manifest.transformer",
                 true) !=
      package_file(files, non_mlp_relative, "manifest.transformer").bytes) {
    throw PackageError(
        "native BitNet non-MLP descriptor disagrees with inventory");
  }

  const JsonObject& controller =
      object(field(manifest, "controller", "manifest"),
             "manifest.controller");
  const std::string controller_relative =
      string_field(controller, "path", "manifest.controller");
  const std::filesystem::path controller_path = safe_package_path(
      root, controller_relative, "manifest.controller");
  std::error_code controller_error;
  if (std::filesystem::symlink_status(controller_path, controller_error)
          .type() != std::filesystem::file_type::directory ||
      controller_error ||
      string_field(controller, "format", "manifest.controller") !=
          "engram.controller.factorized_residual" ||
      size_field(controller, "schema_version", "manifest.controller") != 3 ||
      string_field(controller, "operator", "manifest.controller") !=
          "operator_residual_with_factorized_correction" ||
      bool_field(controller, "correction_enabled", "manifest.controller")) {
    throw PackageError("unsupported native BitNet controller descriptor");
  }
  const std::string controller_metadata_relative =
      (std::filesystem::path(controller_relative) / "metadata.json")
          .generic_string();
  const PackageFile& controller_metadata_file =
      package_file(files, controller_metadata_relative,
                   "manifest.controller");
  if (string_field(controller, "metadata_sha256", "manifest.controller") !=
      controller_metadata_file.sha256) {
    throw PackageError(
        "native BitNet controller metadata binding disagrees with inventory");
  }
  const Json controller_json = read_json(root / controller_metadata_relative);
  const JsonObject& controller_metadata =
      object(controller_json, "controller metadata");
  if (string_field(controller_metadata, "format", "controller metadata") !=
          "engram.controller.factorized_residual" ||
      size_field(controller_metadata, "schema_version",
                 "controller metadata") != 3 ||
      string_field(controller_metadata, "operator", "controller metadata") !=
          "operator_residual_with_factorized_correction" ||
      size_field(controller_metadata, "serialized_bytes",
                 "controller metadata", true) !=
          size_field(controller, "serialized_bytes", "manifest.controller",
                     true)) {
    throw PackageError("controller metadata and package descriptor disagree");
  }

  const JsonObject& model =
      object(field(manifest, "model", "manifest"), "manifest.model");
  const std::size_t hidden_size =
      size_field(model, "hidden_size", "manifest.model");
  const std::size_t intermediate_size =
      size_field(model, "intermediate_size", "manifest.model");
  const std::size_t layers =
      size_field(model, "num_hidden_layers", "manifest.model");
  const float manifest_rms =
      positive_float_field(model, "rms_norm_eps", "manifest.model");
  if (hidden_size > std::numeric_limits<std::size_t>::max() / 3 ||
      size_field(controller_metadata, "state_dim", "controller metadata") !=
          hidden_size ||
      size_field(controller_metadata, "input_dim", "controller metadata") !=
          3 * hidden_size ||
      size_field(controller_metadata, "num_stages",
                 "controller metadata") != layers) {
    throw PackageError("controller and model dimensions disagree");
  }
  const JsonObject& tensor_layout =
      object(field(controller_metadata, "tensor_layout", "controller metadata"),
             "controller metadata.tensor_layout");
  for (const auto& [name, unused] : tensor_layout) {
    static_cast<void>(unused);
    const std::string relative =
        (std::filesystem::path(controller_relative) / (name + ".npy"))
            .generic_string();
    static_cast<void>(
        package_file(files, relative, "controller metadata.tensor_layout"));
  }
  static_cast<void>(field(tensor_layout, "operator_residual_scale",
                          "controller metadata.tensor_layout"));
  static_cast<void>(
      field(tensor_layout, "step_scale", "controller metadata.tensor_layout"));

  constexpr std::string_view kConfigRelative = "config/config.json";
  static_cast<void>(
      package_file(files, std::string(kConfigRelative), "model config"));
  const Json config_json = read_json(root / kConfigRelative);
  const JsonObject& config = object(config_json, "model config");
  const std::size_t config_hidden =
      size_field(config, "hidden_size", "model config");
  const std::size_t config_intermediate =
      size_field(config, "intermediate_size", "model config");
  const std::size_t config_layers =
      size_field(config, "num_hidden_layers", "model config");
  const std::size_t vocabulary_size =
      size_field(config, "vocab_size", "model config");
  const std::size_t query_heads =
      size_field(config, "num_attention_heads", "model config");
  const std::size_t key_value_heads =
      size_field(config, "num_key_value_heads", "model config");
  const std::size_t max_position_embeddings =
      size_field(config, "max_position_embeddings", "model config");
  const float config_rms =
      positive_float_field(config, "rms_norm_eps", "model config");
  const float rope_theta =
      positive_float_field(config, "rope_theta", "model config");
  const std::vector<std::string> architectures =
      string_array_field(config, "architectures", "model config");
  const JsonObject& quantization = object(
      field(config, "quantization_config", "model config"),
      "model config.quantization_config");
  if (config_hidden != hidden_size ||
      config_intermediate != intermediate_size || config_layers != layers ||
      config_rms != manifest_rms || query_heads == 0 ||
      hidden_size % query_heads != 0 || key_value_heads == 0 ||
      query_heads % key_value_heads != 0 ||
      std::find(architectures.begin(), architectures.end(),
                "BitNetForCausalLM") == architectures.end() ||
      string_field(config, "model_type", "model config") != "bitnet" ||
      string_field(config, "hidden_act", "model config") != "relu2" ||
      string_field(config, "torch_dtype", "model config") != "bfloat16" ||
      !bool_field(config, "tie_word_embeddings", "model config") ||
      string_field(quantization, "quant_method",
                   "model config.quantization_config") != "bitnet" ||
      string_field(quantization, "quantization_mode",
                   "model config.quantization_config") != "offline") {
    throw PackageError(
        "authenticated BitNet model config is unsupported or disagrees with "
        "the manifest");
  }

  const JsonObject& tokenizer =
      object(field(manifest, "tokenizer", "manifest"),
             "manifest.tokenizer");
  const std::string tokenizer_relative =
      string_field(tokenizer, "path", "manifest.tokenizer");
  static_cast<void>(
      safe_package_path(root, tokenizer_relative, "manifest.tokenizer"));
  const std::vector<std::string> tokenizer_files =
      string_array_field(tokenizer, "files", "manifest.tokenizer");
  bool generation_config_listed = false;
  for (const std::string& name : tokenizer_files) {
    const std::string relative =
        (std::filesystem::path(tokenizer_relative) / name).generic_string();
    static_cast<void>(
        safe_package_path(root, relative, "manifest.tokenizer.files"));
    static_cast<void>(
        package_file(files, relative, "manifest.tokenizer.files"));
    generation_config_listed =
        generation_config_listed || name == "generation_config.json";
  }
  if (!generation_config_listed) {
    throw PackageError(
        "authenticated tokenizer does not list generation_config.json");
  }
  const std::string generation_config_relative =
      (std::filesystem::path(tokenizer_relative) / "generation_config.json")
          .generic_string();
  const Json generation_json =
      read_json(root / generation_config_relative);
  const JsonObject& generation_config =
      object(generation_json, "generation config");
  std::vector<std::int64_t> eos = eos_token_ids(generation_config);
  const std::int64_t config_eos =
      int64_value(field(config, "eos_token_id", "model config"),
                  "model config.eos_token_id");
  if (std::find(eos.begin(), eos.end(), config_eos) == eos.end() ||
      std::any_of(eos.begin(), eos.end(), [vocabulary_size](const auto token) {
        return static_cast<std::uint64_t>(token) >= vocabulary_size;
      })) {
    throw PackageError(
        "authenticated EOS token ids disagree with model config or vocabulary");
  }

  NativeBitNetDIPPackageMetadata result{
      .root = root,
      .non_mlp_safetensors = root / non_mlp_relative,
      .mlp_artifact = root / mlp_relative,
      .dip_coordinate_index = root / index_relative,
      .controller_directory = controller_path,
      .layers = layers,
      .hidden_size = hidden_size,
      .intermediate_size = intermediate_size,
      .vocabulary_size = vocabulary_size,
      .query_heads = query_heads,
      .key_value_heads = key_value_heads,
      .head_dimension = hidden_size / query_heads,
      .max_position_embeddings = max_position_embeddings,
      .local_window = kLocalWindow,
      .older_candidates = kOlderCandidates,
      .older_top_k = kOlderTopK,
      .sink_tokens = kSinkTokens,
      .rms_norm_epsilon = config_rms,
      .rope_theta = rope_theta,
      .eos_token_ids = std::move(eos),
      .files = {},
  };
  result.files.reserve(files.size());
  for (const auto& [unused, descriptor] : files) {
    static_cast<void>(unused);
    result.files.push_back(descriptor);
  }
  return result;
}

NativeBitNetDIPPackageMetadata load_native_bitnet_dip_package(
    const std::filesystem::path& root) {
  static const NativeBitNetDIPTrustRoot trust_root{
      .package_manifest_bytes = 5787,
      .package_manifest_sha256 =
          "707bbe069ef6892ce9bfe98258f3289e28af15a400922e950c4386f56dd26926",
      .source_package_manifest_sha256 =
          "cddd96a01ff03bd565c108ab58925e7463aad35ebd8b1cc315eb7b050030cd35",
      .source_artifact_sha256 =
          "4fcf598af4346d5391ba428e32ba1629daae2768b73ab6bf872d3f9fb300ab55",
      .coordinate_index_sha256 =
          "b98ce4e46c8ae67d9d92d4d13f5de3d4fe45ef2c76400bd9d50be08b2bd60e15",
      .policy_manifest_sha256 =
          "c572754e597a760bc5ea6ba337bdaaf092e4ae1d5b5e90b6a2a14cbfbea3768e",
      .adjudication_sha256 =
          "ebb5ca9568387ffd3c5b187f8e17f3ce706aaee86f4bbe9e314bf1760a7da5cc",
  };
  return load_native_bitnet_dip_package(root, trust_root);
}

}  // namespace engram
