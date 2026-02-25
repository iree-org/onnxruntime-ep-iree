//===- dim_spec_dispatch.cc ----------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Runtime dispatch helpers for dim specialization variants.
//
//===----------------------------------------------------------------------===//

#include "dim_spec_dispatch.h"

namespace onnxruntime::iree {
namespace {

static bool VariantMatchesDimSpecs(
    const IreeNodeComputeInfo::Variant& variant,
    const std::unordered_map<std::string, int64_t>& dim_values,
    bool has_conflict) {
  // If inputs disagree on a shared symbolic dim, skip specialized variants
  // (they apply tie_shape which assumes all occurrences are equal).
  if (has_conflict && !variant.dim_specs.empty()) return false;

  for (const auto& spec : variant.dim_specs) {
    auto it = dim_values.find(spec.symbolic_name);
    if (it == dim_values.end()) continue;

    int64_t actual = it->second;
    if (actual < spec.min || actual > spec.max) return false;
    if (spec.div > 0 && (actual <= 0 || actual % spec.div != 0)) return false;
  }
  return true;
}

}  // namespace

DispatchDimState CollectDimValuesAndConflicts(
    const Ort::KernelContext& ctx,
    const std::vector<IreeNodeComputeInfo::SymbolicDimMapping>& dim_mappings) {
  DispatchDimState state;
  for (const auto& mapping : dim_mappings) {
    if (mapping.input_index >= ctx.GetInputCount()) continue;
    auto shape = ctx.GetInput(mapping.input_index)
                     .GetTensorTypeAndShapeInfo()
                     .GetShape();
    if (mapping.dim_index >= shape.size()) continue;
    int64_t actual_value = shape[mapping.dim_index];
    auto [it, inserted] =
        state.dim_values.try_emplace(mapping.symbolic_name, actual_value);
    if (!inserted && it->second != actual_value) {
      state.has_conflict = true;
    }
  }
  return state;
}

const IreeNodeComputeInfo::Variant* SelectMatchingVariant(
    const std::vector<IreeNodeComputeInfo::Variant>& variants,
    const std::unordered_map<std::string, int64_t>& dim_values,
    bool has_conflict) {
  // Variants are in user-specified order (first match wins).
  for (const auto& variant : variants) {
    if (VariantMatchesDimSpecs(variant, dim_values, has_conflict)) {
      return &variant;
    }
  }
  return nullptr;
}

}  // namespace onnxruntime::iree
