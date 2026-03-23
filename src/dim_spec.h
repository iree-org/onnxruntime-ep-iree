//===- dim_spec.h ---------------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Dimension specialization types and parser for the IREE EP.
//
//===----------------------------------------------------------------------===//

#ifndef ONNXRUNTIME_EP_IREE_SRC_DIM_SPEC_H_
#define ONNXRUNTIME_EP_IREE_SRC_DIM_SPEC_H_

#include <cstdint>
#include <string>
#include <vector>

#include "ort_import.h"

namespace onnxruntime::iree {

// A single dimension constraint for specialization.
struct DimSpec {
  std::string symbolic_name;
  int64_t min;  // Minimum value (inclusive), >= 1.
  int64_t max;  // Maximum value (inclusive), >= min.
  int64_t div;  // Divisibility constraint. 1 = none.
};

// A set of dimension constraints forming one specialization variant.
using DimSpecVariant = std::vector<DimSpec>;

// Parses the "ep.iree.dim_specs" session option string.
// Format: "batch(1,1), seq(1,131072,16); batch(1,1), seq(1,65536,8)"
//   - name(min, max): range constraint. name(min, max, div): range +
//     divisibility.
//   - Static dims: min == max (e.g., batch(1,1)).
//   - Semicolons separate variants; commas outside parentheses separate specs.
// Returns nullptr on success (results written to `out`), or an OrtStatus* on
// parse failure (e.g., invalid syntax, divisor <= 0).
OrtStatus* ParseDimSpecs(const std::string& spec_str,
                         std::vector<DimSpecVariant>& out);

}  // namespace onnxruntime::iree

#endif  // ONNXRUNTIME_EP_IREE_SRC_DIM_SPEC_H_
