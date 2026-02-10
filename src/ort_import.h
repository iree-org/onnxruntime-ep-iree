//===- ort_import.h -------------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This header wraps the ONNX Runtime C++ API include with the necessary
// ORT_API_MANUAL_INIT define to prevent static initialization issues.
// Include this file instead of including onnxruntime_cxx_api.h directly.
//
//===----------------------------------------------------------------------===//

#ifndef ONNXRUNTIME_EP_IREE_SRC_ORT_IMPORT_H_
#define ONNXRUNTIME_EP_IREE_SRC_ORT_IMPORT_H_

// ORT_API_MANUAL_INIT prevents static initialization of the C++ API.
// We must call Ort::InitApi() explicitly before using any C++ API wrappers.
#define ORT_API_MANUAL_INIT
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

// Helper macro for ORT status checking (returns OrtStatus* on failure).
// Use this macro to propagate OrtStatus* errors from called functions.
#define ORT_RETURN_IF_ERROR(expr)    \
  do {                               \
    OrtStatus* _ort_status = (expr); \
    if (_ort_status != nullptr) {    \
      return _ort_status;            \
    }                                \
  } while (0)

#endif  // ONNXRUNTIME_EP_IREE_SRC_ORT_IMPORT_H_
