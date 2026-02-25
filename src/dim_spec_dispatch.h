//===- dim_spec_dispatch.h -----------------------------------------------===//
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

#ifndef ONNXRUNTIME_EP_IREE_SRC_DIM_SPEC_DISPATCH_H_
#define ONNXRUNTIME_EP_IREE_SRC_DIM_SPEC_DISPATCH_H_

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "iree_ep.h"

namespace onnxruntime::iree {

struct DispatchDimState {
  std::unordered_map<std::string, int64_t> dim_values;
  bool has_conflict = false;
};

DispatchDimState CollectDimValuesAndConflicts(
    const Ort::KernelContext& ctx,
    const std::vector<IreeNodeComputeInfo::SymbolicDimMapping>& dim_mappings);

const IreeNodeComputeInfo::Variant* SelectMatchingVariant(
    const std::vector<IreeNodeComputeInfo::Variant>& variants,
    const std::unordered_map<std::string, int64_t>& dim_values,
    bool has_conflict);

}  // namespace onnxruntime::iree

#endif  // ONNXRUNTIME_EP_IREE_SRC_DIM_SPEC_DISPATCH_H_
