//===- iree_ep.h ----------------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file defines the IreeEp class which handles graph partitioning,
// compilation, and execution using IREE as the backend.
//
//===----------------------------------------------------------------------===//

#ifndef ONNXRUNTIME_EP_IREE_SRC_IREE_EP_H_
#define ONNXRUNTIME_EP_IREE_SRC_IREE_EP_H_

#include <string>

#include "iree_ep_factory.h"
#include "iree_wrappers.h"
#include "ort_import.h"

namespace onnxruntime::iree {

// Forward declarations
class IreeEpFactory;

// IREE Execution Provider.
// Handles graph partitioning, compilation, and execution using IREE runtime.
// Each EP instance owns an IREE HAL device created from the factory's instance.
class IreeEp : public OrtEp, public ApiPtrs {
 public:
  struct Config {
    bool enable_ep_context = false;
    // Target architecture to compile for.
    // TODO: Ideally, we want to get this from the device. I'm not sure how
    // to do this in IREE.
    std::string target_arch = "";  // e.g., "gfx1100", "mi300x"
    // Optimization level for the IREE compiler: O0, O1, O2, O3
    std::string opt_level = "O0";
    // Backend to use for the device (derived from driver name).
    std::string backend = "";
    // Save intermediate compilation artifacts (MLIR, VMFB) for debugging.
    bool save_intermediates = false;
  };

  IreeEp(IreeEpFactory& factory, const std::string& name, const Config& config,
         const OrtLogger& logger, uint32_t device_id);

  ~IreeEp();

  // Accessor for the IREE device (from factory's device cache).
  [[nodiscard]] iree_hal_device_t* IreeDevice() const;

  // Accessor for the logger.
  [[nodiscard]] const Ort::Logger& Logger() const { return logger_; }

 private:
  // EP interface implementations (called via function pointers).

  // Returns the EP name.
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;

  // Determines which nodes the EP can execute.
  // For now: claims ALL nodes (compile mode).
  static OrtStatus* ORT_API_CALL
  GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                    OrtEpGraphSupportInfo* graph_support_info) noexcept;

  // Compiles fused subgraphs into executable code.
  static OrtStatus* ORT_API_CALL CompileImpl(
      OrtEp* this_ptr, const OrtGraph** graphs, const OrtNode** fused_nodes,
      size_t count, OrtNodeComputeInfo** node_compute_infos,
      OrtNode** ep_context_nodes) noexcept;

  // Releases node compute infos created in Compile.
  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(
      OrtEp* this_ptr, OrtNodeComputeInfo** node_compute_infos,
      size_t num_node_compute_infos) noexcept;

  IreeEpFactory& factory_;
  std::string name_;
  Config config_;
  Ort::Logger logger_;

  // Device ID for this EP. The actual HAL device is managed by the factory's
  // device_cache_ and accessed via GetDeviceForId(). This ensures the EP and
  // allocator use the same device instance.
  uint32_t device_id_;
};

// Compute kernel for compiled nodes.
// Holds the IREE session and function for a compiled subgraph.
struct IreeNodeComputeInfo : OrtNodeComputeInfo {
  // Constructor takes ownership of session and stores function reference.
  // Session is created in CompileImpl and passed here.
  IreeNodeComputeInfo(IreeEp& ep, RuntimeSessionPtr session,
                      iree_vm_function_t function);

  ~IreeNodeComputeInfo();

  // Creates per-node computation state.
  static OrtStatus* ORT_API_CALL CreateStateImpl(
      OrtNodeComputeInfo* this_ptr, OrtNodeComputeContext* compute_context,
      void** compute_state) noexcept;

  // Executes the computation.
  static OrtStatus* ORT_API_CALL
  ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
              OrtKernelContext* kernel_context) noexcept;

  // Releases computation state.
  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* this_ptr,
                                            void* compute_state) noexcept;

  // Non-owning reference to parent EP. The EP must outlive this compute info.
  IreeEp& ep;

  // IREE runtime state for this compiled subgraph.
  RuntimeSessionPtr session_;
  iree_vm_function_t function_;
};

}  // namespace onnxruntime::iree

#endif  // ONNXRUNTIME_EP_IREE_SRC_IREE_EP_H_
