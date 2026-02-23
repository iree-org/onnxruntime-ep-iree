//===- plugin_entry.cc ----------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Plugin entry points for IREE ONNX Runtime EP.
//
//===----------------------------------------------------------------------===//

#include "iree_ep_factory.h"
#include "ort_import.h"

// ORT_EXPORT is a no-op on Linux. Define EP_EXPORT with explicit visibility
// so that only the plugin entry points are exported from the shared library.
// Combined with CXX_VISIBILITY_PRESET=hidden in CMake, this ensures all
// statically-linked IREE runtime symbols remain hidden.
#if defined(_WIN32)
#define EP_EXPORT __declspec(dllexport)
#else
#define EP_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

// Plugin entry point - creates EP factories.
EP_EXPORT OrtStatus* CreateEpFactories(const char* registration_name,
                                        const OrtApiBase* ort_api_base,
                                        const OrtLogger* default_logger,
                                        OrtEpFactory** factories,
                                        size_t max_factories,
                                        size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ep_api = ort_api->GetEpApi();
  const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();

  // Initialize C++ API (required when using ORT_API_MANUAL_INIT).
  Ort::InitApi(ort_api);

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Need at least one factory slot");
  }

  // Create factory (use registration_name or "IREE").
  auto factory = std::make_unique<onnxruntime::iree::IreeEpFactory>(
      registration_name ? registration_name : "IREE",
      onnxruntime::iree::ApiPtrs{*ort_api, *ep_api, *model_editor_api},
      default_logger);

  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;
}

// Plugin cleanup
EP_EXPORT OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<onnxruntime::iree::IreeEpFactory*>(factory);
  return nullptr;
}

}  // extern "C"
