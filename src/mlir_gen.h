//===- mlir_gen.h ---------------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Generates torch-mlir ONNX dialect MLIR text directly from an OrtGraph,
// bypassing the need for ONNX protobuf dependencies.
//
// Note that we make a conscious decision to avoid using ONNX protobuf here.
// There are some utilities to convert OrtGraph to ONNX protobuf implemented in
// https://github.com/microsoft/onnxruntime/pull/25292, but even they recommend
// that EP's should directly convert OrtGraph to their own internal
// representation. We also want to be light on dependencies and don't want to
// take a dependency on protobuf and onnx. Since we don't use ONNX protobuf, we
// have to directly emit torch onnx dialect instead of using torch-mlir's onnx
// importer.
//
//===----------------------------------------------------------------------===//

#ifndef ONNXRUNTIME_EP_IREE_SRC_MLIR_GEN_H_
#define ONNXRUNTIME_EP_IREE_SRC_MLIR_GEN_H_

#include <string>

#include "iree_wrappers.h"
#include "ort_import.h"

namespace onnxruntime::iree {

// Target configuration passed to the MLIR generator. Always populated with the
// EP's main target arch and backend. When the graph contains
// com.iree:ExternDispatch nodes, this info is used to emit hal.dispatch.extern
// ops with the correct executable target.
struct TargetConfig {
  std::string target_arch;  // e.g., "gfx1100"
  std::string backend;      // e.g., "hip"
  std::string hal_backend;  // e.g., "rocm" (HAL executable target backend)
  std::string hal_format;   // e.g., "rocm-hsaco-fb" (HAL executable format)

  // Creates a TargetConfig, resolving hal_backend/hal_format from backend.
  static TargetConfig Create(const std::string& target_arch,
                             const std::string& backend);
};

// Generates MLIR text from an OrtGraph and writes it to the specified file.
// Small initializers are inlined in the MLIR. Large initializers are emitted as
// parameter references and their data is written to an IRPA archive at
// irpa_path. out_index and out_provider are populated with the parameter index
// and provider for the archive. They remain null if no parameters are needed.
// com.iree:ExternDispatch nodes are emitted as hal.dispatch.extern ops using
// the target info from target_config.
OrtStatus* GenerateMlir(const Ort::ConstGraph& graph, const OrtApi& ort_api,
                        const std::string& mlir_path,
                        const std::string& irpa_path,
                        ParameterIndexPtr& out_index,
                        ParameterProviderPtr& out_provider,
                        TargetConfig target_config);

}  // namespace onnxruntime::iree

#endif  // ONNXRUNTIME_EP_IREE_SRC_MLIR_GEN_H_
