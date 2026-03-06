//===- iree_ep_factory.h --------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ONNXRUNTIME_EP_IREE_SRC_IREE_EP_FACTORY_H_
#define ONNXRUNTIME_EP_IREE_SRC_IREE_EP_FACTORY_H_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "iree_wrappers.h"
#include "ort_import.h"

namespace onnxruntime::iree {

// Forward declarations.
class IreeAllocator;
class IreeDataTransfer;

// EP configuration constants
inline constexpr const char* kEpVendor = "IREE";
inline constexpr uint32_t kEpVendorId = 0x1EEE;  // "IREE" in hex-ish
inline constexpr const char* kEpVersion = "0.1.0";

// Hardware vendor IDs for device matching.
// These match OrtDevice::VendorIds from onnxruntime/core/framework/ortdevice.h.
namespace VendorIds {
inline constexpr uint32_t kAmd = 0x1002;     // AMD: ROCm, MIGraphX EPs
inline constexpr uint32_t kNvidia = 0x10DE;  // NVIDIA: CUDA/TensorRT
inline constexpr uint32_t kIntel = 0x8086;   // Intel: OpenVINO
}  // namespace VendorIds

// Helper struct to pass API pointers
struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
  const OrtModelEditorApi& model_editor_api;
};

// IREE Execution Provider Factory
// Inherits from OrtEpFactory and sets up function pointers in constructor.
// Manages a shared IREE runtime instance that is used by all EPs created
// by this factory.
class IreeEpFactory : public OrtEpFactory, public ApiPtrs {
 public:
  IreeEpFactory(const char* ep_name, ApiPtrs apis,
                const OrtLogger* default_logger);
  ~IreeEpFactory();

  // Accessor for the shared IREE runtime instance.
  iree_runtime_instance_t* IreeInstance() const { return instance_.Get(); }

  // Gets or creates a HAL device for the given device_id.
  // Returns nullptr if device_id is invalid.
  iree_hal_device_t* GetDeviceForId(uint32_t device_id);

  // Accessor for the logger.
  const Ort::Logger& Logger() const { return logger_; }

 private:
  // Internal version of GetDeviceForId that assumes factory_mutex_ is already
  // held.
  iree_hal_device_t* GetDeviceForIdLocked(uint32_t device_id);

  // Factory interface implementations (called via function pointers)
  static const char* ORT_API_CALL
  GetNameImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL
  GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;
  static uint32_t ORT_API_CALL
  GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL
  GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(
      OrtEpFactory* this_ptr, const OrtHardwareDevice* const* devices,
      size_t num_devices, OrtEpDevice** ep_devices, size_t max_ep_devices,
      size_t* p_num_ep_devices) noexcept;

  static OrtStatus* ORT_API_CALL
  CreateEpImpl(OrtEpFactory* this_ptr, const OrtHardwareDevice* const* devices,
               const OrtKeyValuePairs* const* ep_metadata, size_t num_devices,
               const OrtSessionOptions* session_options,
               const OrtLogger* logger, OrtEp** ep) noexcept;

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* this_ptr,
                                         OrtEp* ep) noexcept;

  static OrtStatus* ORT_API_CALL
  CreateAllocatorImpl(OrtEpFactory* this_ptr, const OrtMemoryInfo* memory_info,
                      const OrtKeyValuePairs* allocator_options,
                      OrtAllocator** allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(
      OrtEpFactory* this_ptr, OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(
      OrtEpFactory* this_ptr, OrtDataTransferImpl** data_transfer) noexcept;

  static bool ORT_API_CALL
  IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(
      OrtEpFactory* this_ptr, const OrtMemoryDevice* memory_device,
      const OrtKeyValuePairs* stream_options,
      OrtSyncStreamImpl** stream) noexcept;

  // Custom op domain support for com.iree:ExternDispatch.
  static OrtStatus* ORT_API_CALL GetNumCustomOpDomainsImpl(
      OrtEpFactory* this_ptr, size_t* num_domains) noexcept;
  static OrtStatus* ORT_API_CALL
  GetCustomOpDomainsImpl(OrtEpFactory* this_ptr, OrtCustomOpDomain** domains,
                         size_t num_domains) noexcept;

  // Enumerate IREE devices and create OrtHardwareDevice instances.
  void CreateIreeHwDevices();

  // Member variables
  Ort::Logger logger_;
  const std::string ep_name_;

  // IREE runtime instance (shared across all EPs created by this factory).
  RuntimeInstancePtr instance_;

  // OrtHardwareDevice instances created by this factory.
  // Owned by the factory and released in the destructor.
  std::vector<OrtHardwareDevice*> hw_devices_;

  // Protects device_cache_, allocators_, and data_transfer_ for thread safety.
  // ORT may call factory methods concurrently from multiple sessions.
  mutable std::mutex factory_mutex_;

  // Cache of HAL devices by device_id. Created lazily when allocator is needed.
  std::unordered_map<uint32_t, HalDevicePtr> device_cache_;

  // Memory info objects for device-local memory, indexed by device_id.
  // These must be kept alive for the lifetime of the factory since
  // EpDevice_AddAllocatorInfo does not copy the OrtMemoryInfo.
  std::vector<Ort::MemoryInfo> device_memory_infos_;

  // Allocators by device_id. Created lazily when requested.
  std::unordered_map<uint32_t, std::unique_ptr<IreeAllocator>> allocators_;

  // Shared data transfer instance. Created lazily when requested.
  std::unique_ptr<IreeDataTransfer> data_transfer_;

  // Custom op domain for com.iree:ExternDispatch nodes.
  // Must outlive all sessions using it.
  Ort::CustomOpDomain extern_dispatch_domain_{nullptr};
};

}  // namespace onnxruntime::iree

#endif  // ONNXRUNTIME_EP_IREE_SRC_IREE_EP_FACTORY_H_
