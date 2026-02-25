//===- dim_specs_parser.cc -----------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Parses IREE EP dim specialization constraints from session options.
//
//===----------------------------------------------------------------------===//

#include <charconv>
#include <format>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "iree_ep.h"

namespace onnxruntime::iree {
namespace {

struct DimSpecParseCursor {
  std::string_view input;
  size_t pos = 0;
};

static OrtStatus* DimSpecError(std::string msg) {
  return Ort::Status(msg.c_str(), ORT_INVALID_ARGUMENT).release();
}

static std::string_view Trim(std::string_view sv) {
  while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) {
    sv.remove_prefix(1);
  }
  while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) {
    sv.remove_suffix(1);
  }
  return sv;
}

static bool IsAtEnd(const DimSpecParseCursor& cursor) {
  return cursor.pos >= cursor.input.size();
}

static char Peek(const DimSpecParseCursor& cursor) {
  return cursor.input[cursor.pos];
}

static void SkipWhitespace(DimSpecParseCursor& cursor) {
  while (!IsAtEnd(cursor) &&
         std::isspace(static_cast<unsigned char>(Peek(cursor)))) {
    ++cursor.pos;
  }
}

static bool ParseInt(std::string_view sv, int64_t& result) {
  auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), result);
  return ec == std::errc{} && ptr == sv.data() + sv.size();
}

static OrtStatus* ParseIntArg(DimSpecParseCursor& cursor,
                              const std::string& symbolic_name,
                              const char* arg_name, int64_t& result) {
  SkipWhitespace(cursor);
  size_t start = cursor.pos;
  while (!IsAtEnd(cursor) && Peek(cursor) != ',' && Peek(cursor) != ')') {
    ++cursor.pos;
  }
  std::string_view token = Trim(cursor.input.substr(start, cursor.pos - start));
  if (!ParseInt(token, result)) {
    return DimSpecError(std::format("dim_specs: invalid {} \"{}\" for \"{}\"",
                                    arg_name, std::string(token),
                                    symbolic_name));
  }
  return nullptr;
}

static OrtStatus* ParseSpec(DimSpecParseCursor& cursor, DimSpec& dim) {
  SkipWhitespace(cursor);
  size_t spec_start = cursor.pos;

  // Parse name (everything up to '(' with surrounding whitespace ignored).
  while (!IsAtEnd(cursor) && Peek(cursor) != '(' && Peek(cursor) != ',' &&
         Peek(cursor) != ';') {
    ++cursor.pos;
  }
  std::string_view name =
      Trim(cursor.input.substr(spec_start, cursor.pos - spec_start));
  if (IsAtEnd(cursor) || Peek(cursor) != '(') {
    return DimSpecError(
        std::format("dim_specs: missing '(' in \"{}\"", std::string(name)));
  }

  if (name.empty()) {
    return DimSpecError(
        std::format("dim_specs: empty name in \"{}\"",
                    std::string(cursor.input.substr(
                        spec_start, cursor.pos - spec_start + 1))));
  }

  std::string symbolic_name(name);
  ++cursor.pos;  // Consume '('.

  int64_t min_val = 0;
  ORT_RETURN_IF_ERROR(ParseIntArg(cursor, symbolic_name, "min", min_val));
  if (IsAtEnd(cursor) || Peek(cursor) != ',') {
    return DimSpecError(std::format(
        "dim_specs: expected ',' after min for \"{}\"", symbolic_name));
  }
  ++cursor.pos;  // Consume ','.

  int64_t max_val = 0;
  ORT_RETURN_IF_ERROR(ParseIntArg(cursor, symbolic_name, "max", max_val));

  int64_t div_val = 0;
  bool has_div_arg = false;
  if (!IsAtEnd(cursor) && Peek(cursor) == ',') {
    ++cursor.pos;  // Consume ',' before div.
    has_div_arg = true;
    ORT_RETURN_IF_ERROR(ParseIntArg(cursor, symbolic_name, "div", div_val));
  }

  if (IsAtEnd(cursor) || Peek(cursor) != ')') {
    return DimSpecError(
        std::format("dim_specs: missing ')' for \"{}\"", symbolic_name));
  }
  ++cursor.pos;  // Consume ')'.

  if (min_val < 0) {
    return DimSpecError(
        std::format("dim_specs: min must be >= 0, got {} for \"{}\"", min_val,
                    symbolic_name));
  }
  if (max_val < min_val) {
    return DimSpecError(std::format(
        "dim_specs: max must be >= min, got min={} max={} for \"{}\"", min_val,
        max_val, symbolic_name));
  }
  if (has_div_arg && div_val <= 0) {
    return DimSpecError(
        std::format("dim_specs: div must be > 0, got {} for \"{}\"", div_val,
                    symbolic_name));
  }

  dim = {symbolic_name, min_val, max_val, div_val};
  return nullptr;
}

static OrtStatus* ParseVariant(DimSpecParseCursor& cursor,
                               DimSpecVariant& variant) {
  std::unordered_set<std::string> seen_symbolic_names;

  while (SkipWhitespace(cursor), !IsAtEnd(cursor) && Peek(cursor) != ';') {
    // Ignore empty entries like ",,".
    if (Peek(cursor) == ',') {
      ++cursor.pos;
      continue;
    }

    DimSpec dim;
    ORT_RETURN_IF_ERROR(ParseSpec(cursor, dim));
    if (!seen_symbolic_names.insert(dim.symbolic_name).second) {
      return DimSpecError(std::format(
          "dim_specs: duplicate key \"{}\" in one variant", dim.symbolic_name));
    }
    variant.push_back(std::move(dim));

    SkipWhitespace(cursor);
    if (IsAtEnd(cursor) || Peek(cursor) == ';') {
      return nullptr;
    }
    if (Peek(cursor) != ',') {
      return DimSpecError("dim_specs: expected ',' between specs");
    }
    ++cursor.pos;  // Consume comma between specs.
  }
  return nullptr;
}

static OrtStatus* ParseDimSpecsImpl(std::string_view spec_str,
                                    std::vector<DimSpecVariant>& out) {
  out.clear();
  DimSpecParseCursor cursor{Trim(spec_str), 0};

  while (!IsAtEnd(cursor)) {
    DimSpecVariant variant;
    ORT_RETURN_IF_ERROR(ParseVariant(cursor, variant));
    if (!variant.empty()) out.push_back(std::move(variant));

    SkipWhitespace(cursor);
    if (IsAtEnd(cursor)) break;
    if (Peek(cursor) != ';') {
      return DimSpecError("dim_specs: expected ';' between variants");
    }
    ++cursor.pos;  // Consume ';'.
  }

  return nullptr;
}

}  // namespace

// Parses dim_specs in the format "batch(1,1), seq(0,131072,16)".
// Semicolons separate variants. Commas outside parentheses separate specs.
// Each spec is name(min, max) or name(min, max, div).
OrtStatus* ParseDimSpecs(const std::string& spec_str,
                         std::vector<DimSpecVariant>& out) {
  return ParseDimSpecsImpl(spec_str, out);
}

}  // namespace onnxruntime::iree
