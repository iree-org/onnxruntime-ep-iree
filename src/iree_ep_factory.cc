//===- iree_ep_factory.cc -------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// IREE Execution Provider Factory Implementation.
//
//===----------------------------------------------------------------------===//

#include "iree_ep_factory.h"

#include <algorithm>
#include <memory>
#include <mutex>

#include "iree_allocator.h"
#include "iree_data_transfer.h"
#include "iree_ep.h"

namespace onnxruntime::iree {

IreeEpFactory::IreeEpFactory(const char* ep_name, ApiPtrs apis,
                             const OrtLogger* default_logger)
    : OrtEpFactory{},
      ApiPtrs(apis),
      logger_(default_logger),
      ep_name_(ep_name) {
  // Set ORT version we support.
  ort_version_supported = ORT_API_VERSION;

  // Set function pointers.
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;
  GetSupportedDevices = GetSupportedDevicesImpl;
  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;
  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;
  CreateDataTransfer = CreateDataTransferImpl;
  IsStreamAware = IsStreamAwareImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;

  // Initialize IREE runtime instance (shared across all EPs).
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);

  iree_status_t status = iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), instance_.ForOutput());
  if (!iree_status_is_ok(status)) {
    // Log error but continue - EP creation will fail later if needed.
    ORT_CXX_LOG_NOEXCEPT(logger_, ORT_LOGGING_LEVEL_ERROR,
                         "IREE EP: Failed to create IREE runtime instance");
    iree_status_ignore(status);
    return;
  }

  CreateIreeHwDevices();
}

void IreeEpFactory::CreateIreeHwDevices() {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_driver_registry_t* registry =
      iree_runtime_instance_driver_registry(instance_.Get());

  iree_host_size_t driver_count = 0;
  IreeAllocatedPtr<iree_hal_driver_info_t> driver_infos(allocator);
  iree_status_t status = iree_hal_driver_registry_enumerate(
      registry, allocator, &driver_count, driver_infos.ForOutput());
  if (!iree_status_is_ok(status)) {
    ORT_CXX_LOG_NOEXCEPT(logger_, ORT_LOGGING_LEVEL_WARNING,
                         "IREE EP: Failed to enumerate drivers");
    iree_status_ignore(status);
    return;
  }

  size_t device_index = 0;
  for (iree_host_size_t i = 0; i < driver_count; ++i) {
    iree_string_view_t driver_name = driver_infos.Get()[i].driver_name;
    std::string driver_name_str(driver_name.data, driver_name.size);

    HalDriverPtr driver;
    status = iree_hal_driver_registry_try_create(registry, driver_name,
                                                 allocator, driver.ForOutput());
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      continue;
    }

    iree_host_size_t device_count = 0;
    IreeAllocatedPtr<iree_hal_device_info_t> device_infos(allocator);
    status = iree_hal_driver_query_available_devices(
        driver.Get(), allocator, &device_count, device_infos.ForOutput());
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      continue;
    }

    // Create OrtHardwareDevice for each IREE device available.
    //
    // Given an OrtHardwareDevice, ORT can find which hardware device it is by
    // checking its vendor id (should be kEpVendorId) and the device id (the
    // index it's stored in hw_devices_).
    //
    // IREE can identify the device using the driver and device path stored in
    // the hardware device metadata.
    for (iree_host_size_t j = 0; j < device_count; ++j) {
      iree_string_view_t device_path = device_infos.Get()[j].path;
      std::string device_path_str(device_path.data, device_path.size);

      OrtKeyValuePairs* hw_metadata = nullptr;
      ort_api.CreateKeyValuePairs(&hw_metadata);
      ort_api.AddKeyValuePair(hw_metadata, "iree.driver",
                              driver_name_str.c_str());
      ort_api.AddKeyValuePair(hw_metadata, "iree.device_path",
                              device_path_str.c_str());
      // Store device_id for later retrieval in CreateEpImpl.
      ort_api.AddKeyValuePair(hw_metadata, "iree.device_id",
                              std::to_string(device_index).c_str());

      // TODO: We are pretending here that all IREE devices are GPUs. This is
      // because we follow a host <-> device model in IREE. How does this
      // matter in practice? If we are using IREE APIs anyway, does it matter
      // if we set this to CPU?
      OrtHardwareDeviceType device_type = OrtHardwareDeviceType_GPU;

      OrtHardwareDevice* hw_device = nullptr;
      OrtStatus* ort_status = ep_api.CreateHardwareDevice(
          device_type, kEpVendorId, static_cast<uint32_t>(device_index++),
          kEpVendor, hw_metadata, &hw_device);
      ort_api.ReleaseKeyValuePairs(hw_metadata);
      if (ort_status != nullptr) {
        ort_api.ReleaseStatus(ort_status);
        continue;
      }

      hw_devices_.push_back(hw_device);
    }
  }
}

IreeEpFactory::~IreeEpFactory() {
  // Note: Avoid excessive logging during cleanup - ORT logging infrastructure
  // may be torn down before our destructors run during Python interpreter
  // shutdown.

  // Clean up in proper order: allocators first (they reference devices),
  // then data transfer, then devices, finally hardware devices.
  allocators_.clear();
  data_transfer_.reset();
  device_cache_.clear();

  // Release hardware devices registered with ORT.
  for (size_t i = 0; i < hw_devices_.size(); ++i) {
    ep_api.ReleaseHardwareDevice(hw_devices_[i]);
  }
}

iree_hal_device_t* IreeEpFactory::GetDeviceForId(uint32_t device_id) {
  std::lock_guard<std::mutex> lock(factory_mutex_);
  return GetDeviceForIdLocked(device_id);
}

iree_hal_device_t* IreeEpFactory::GetDeviceForIdLocked(uint32_t device_id) {
  // Check if device is already cached.
  auto it = device_cache_.find(device_id);
  if (it != device_cache_.end()) {
    return it->second.Get();
  }

  // Validate device_id.
  if (device_id >= hw_devices_.size()) {
    ORT_CXX_LOGF_NOEXCEPT(logger_, ORT_LOGGING_LEVEL_ERROR,
                          "IREE EP: Invalid device_id %u (max=%zu)", device_id,
                          hw_devices_.size());
    return nullptr;
  }

  // Get driver and device path from hardware device metadata.
  const OrtHardwareDevice* hw_device = hw_devices_[device_id];
  const OrtKeyValuePairs* hw_metadata =
      ort_api.HardwareDevice_Metadata(hw_device);
  const char* driver_name = ort_api.GetKeyValue(hw_metadata, "iree.driver");
  const char* device_path =
      ort_api.GetKeyValue(hw_metadata, "iree.device_path");

  if (!driver_name || !device_path) {
    ORT_CXX_LOGF_NOEXCEPT(logger_, ORT_LOGGING_LEVEL_ERROR,
                          "IREE EP: Missing driver/device_path for device %u",
                          device_id);
    return nullptr;
  }

  // Build device URI and create HAL device.
  std::string device_uri = std::string(driver_name) + "://" + device_path;
  HalDevicePtr hal_device;
  iree_status_t status = iree_hal_create_device(
      iree_runtime_instance_driver_registry(IreeInstance()),
      iree_make_string_view(device_uri.data(), device_uri.size()),
      iree_allocator_system(), hal_device.ForOutput());

  if (!iree_status_is_ok(status)) {
    ORT_CXX_LOGF_NOEXCEPT(logger_, ORT_LOGGING_LEVEL_ERROR,
                          "IREE EP: Failed to create HAL device for %s",
                          device_uri.c_str());
    iree_status_ignore(status);
    return nullptr;
  }

  ORT_CXX_LOGF_NOEXCEPT(logger_, ORT_LOGGING_LEVEL_INFO,
                        "IREE EP: Created HAL device for device_id %u (%s)",
                        device_id, device_uri.c_str());

  iree_hal_device_t* device_ptr = hal_device.Get();
  device_cache_[device_id] = std::move(hal_device);
  return device_ptr;
}

// Factory interface implementations

/*static*/
const char* ORT_API_CALL
IreeEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const IreeEpFactory*>(this_ptr);
  return factory->ep_name_.c_str();
}

/*static*/
const char* ORT_API_CALL
IreeEpFactory::GetVendorImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return kEpVendor;
}

/*static*/
uint32_t ORT_API_CALL
IreeEpFactory::GetVendorIdImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return kEpVendorId;
}

/*static*/
const char* ORT_API_CALL
IreeEpFactory::GetVersionImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return kEpVersion;
}

/*static*/
OrtStatus* ORT_API_CALL IreeEpFactory::GetSupportedDevicesImpl(
    OrtEpFactory* this_ptr, const OrtHardwareDevice* const* /*devices*/,
    size_t /*num_devices*/, OrtEpDevice** ep_devices, size_t max_ep_devices,
    size_t* p_num_ep_devices) noexcept {
  auto* factory = static_cast<IreeEpFactory*>(this_ptr);

  size_t& num_ep_devices = *p_num_ep_devices;
  num_ep_devices = 0;

  // Reserve capacity for memory_info objects to prevent reallocation during
  // the loop. This is critical because EpDevice_AddAllocatorInfo stores
  // pointers and reallocation would invalidate them.
  size_t num_devices_to_add =
      std::min(factory->hw_devices_.size(), max_ep_devices);
  factory->device_memory_infos_.reserve(num_devices_to_add);

  // Create OrtEpDevice for each hardware device enumerated in constructor.
  for (size_t i = 0;
       i < factory->hw_devices_.size() && num_ep_devices < max_ep_devices;
       ++i) {
    OrtEpDevice* ep_device = nullptr;
    ORT_RETURN_IF_ERROR(factory->ep_api.CreateEpDevice(
        factory, factory->hw_devices_[i], nullptr, nullptr, &ep_device));

    // Register allocator info for device-local memory.
    // This tells ORT that this device has an allocator available for
    // device-local memory allocations.
    //
    // IMPORTANT: EpDevice_AddAllocatorInfo does NOT copy the OrtMemoryInfo,
    // it just stores the pointer. We must keep the memory_info alive for
    // the lifetime of the factory by storing it in device_memory_infos_.
    factory->device_memory_infos_.emplace_back(
        "IREE",                       // name
        OrtMemoryInfoDeviceType_GPU,  // device_type
        kEpVendorId,                  // vendor_id
        static_cast<uint32_t>(i),     // device_id
        OrtDeviceMemoryType_DEFAULT,  // mem_type
        0,                            // alignment (default)
        OrtDeviceAllocator);          // allocator_type

    // Get raw pointer to pass to ORT.
    const OrtMemoryInfo* mem_info_ptr = factory->device_memory_infos_.back();

    OrtStatus* add_alloc_status =
        factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, mem_info_ptr);
    if (add_alloc_status != nullptr) {
      factory->device_memory_infos_.pop_back();
      factory->ep_api.ReleaseEpDevice(ep_device);
      return add_alloc_status;
    }

    ep_devices[num_ep_devices++] = ep_device;
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL IreeEpFactory::CreateEpImpl(
    OrtEpFactory* this_ptr, const OrtHardwareDevice* const* devices,
    const OrtKeyValuePairs* const* /*ep_options*/, size_t num_devices,
    const OrtSessionOptions* session_options, const OrtLogger* logger,
    OrtEp** ep) noexcept {
  auto* factory = static_cast<IreeEpFactory*>(this_ptr);
  *ep = nullptr;

  // Validate device selection.
  if (num_devices != 1) {
    // TODO: Can we even do multi-device work with onnxruntime? Afaik, there is
    // multi-device graph partitioning support:
    // https://github.com/microsoft/onnxruntime/issues/16382
    //
    // The other option would be to partition the graph ourselves in IREE. That
    // seems like a whole other line of work which is very out-of-scope for now.
    return Ort::Status("NYI: Multi-device IREE EP", ORT_NOT_IMPLEMENTED)
        .release();
  }

  // Get driver, device path, and device_id from hardware device metadata.
  const OrtHardwareDevice& hardware_device = *devices[0];
  const OrtKeyValuePairs* hw_metadata =
      factory->ort_api.HardwareDevice_Metadata(&hardware_device);
  const char* driver_name =
      factory->ort_api.GetKeyValue(hw_metadata, "iree.driver");
  const char* device_path =
      factory->ort_api.GetKeyValue(hw_metadata, "iree.device_path");
  const char* device_id_str =
      factory->ort_api.GetKeyValue(hw_metadata, "iree.device_id");
  if (driver_name == nullptr || device_path == nullptr ||
      device_id_str == nullptr) {
    return Ort::Status(
               "IREE EP: Driver, device path, or device_id not found in "
               "hardware device metadata",
               ORT_INVALID_ARGUMENT)
        .release();
  }
  uint32_t device_id = static_cast<uint32_t>(std::stoul(device_id_str));

  // Parse configuration from session options config entries.
  // The Python API stores options with "ep.iree." prefix.
  IreeEp::Config config = {};
  if (session_options != nullptr) {
    Ort::ConstSessionOptions sess_opts(session_options);
    config.target_arch =
        sess_opts.GetConfigEntryOrDefault("ep.iree.target_arch", "");
    config.opt_level =
        sess_opts.GetConfigEntryOrDefault("ep.iree.opt_level", "O0");
    config.save_intermediates = sess_opts.GetConfigEntryOrDefault(
                                    "ep.iree.save_intermediates", "0") == "1";
    config.enable_ep_context_cache = sess_opts.GetConfigEntryOrDefault(
                                    "ep.iree.enable_ep_context_cache", "0") == "1";
  }

  // Select backend based on driver.
  std::string driver_str(driver_name);
  if (driver_str == "vulkan") {
    config.backend = "vulkan";
  } else if (driver_str == "cuda") {
    config.backend = "cuda";
  } else if (driver_str == "hip") {
    config.backend = "hip";
  } else if (driver_str == "local-task" || driver_str == "local-sync") {
    config.backend = "llvm-cpu";
  } else {
    return Ort::Status(("IREE EP: Unknown driver: " + driver_str).c_str(),
                       ORT_INVALID_ARGUMENT)
        .release();
  }

  // Require target_arch for non-CPU devices.
  OrtHardwareDeviceType device_type =
      factory->ort_api.HardwareDevice_Type(&hardware_device);
  if (device_type != OrtHardwareDeviceType_CPU && config.target_arch.empty()) {
    return Ort::Status(
               "IREE EP: 'target_arch' option must be specified for non-CPU "
               "devices",
               ORT_INVALID_ARGUMENT)
        .release();
  }

  ORT_CXX_LOGF_NOEXCEPT(factory->logger_, ORT_LOGGING_LEVEL_INFO,
                        "IREE EP: Creating EP with driver='%s', "
                        "device_path='%s', device_id=%u, target_arch='%s', "
                        "opt_level='%s'",
                        driver_name, device_path, device_id,
                        config.target_arch.c_str(), config.opt_level.c_str());

  // Get HAL device from factory's device cache. This ensures the EP uses
  // the same device instance as the allocator, which is critical for
  // buffer operations to work correctly.
  iree_hal_device_t* hal_device = factory->GetDeviceForId(device_id);
  if (!hal_device) {
    return Ort::Status("IREE EP: Failed to get device from factory",
                       ORT_INVALID_ARGUMENT)
        .release();
  }

  // Create the EP instance with the device_id (EP gets device from factory).
  auto ep_instance = std::make_unique<IreeEp>(*factory, factory->ep_name_,
                                              config, *logger, device_id);

  *ep = ep_instance.release();
  return nullptr;
}

/*static*/
void ORT_API_CALL IreeEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/,
                                               OrtEp* ep) noexcept {
  delete static_cast<IreeEp*>(ep);
}

/*static*/
OrtStatus* ORT_API_CALL IreeEpFactory::CreateAllocatorImpl(
    OrtEpFactory* this_ptr, const OrtMemoryInfo* memory_info,
    const OrtKeyValuePairs* /*allocator_options*/,
    OrtAllocator** allocator) noexcept {
  auto* factory = static_cast<IreeEpFactory*>(this_ptr);

  *allocator = nullptr;

  if (memory_info == nullptr) {
    // No memory info means use default CPU allocator.
    return nullptr;
  }

  // Get device_id from memory_info.
  int32_t device_id_signed = 0;
  OrtStatus* id_status =
      factory->ort_api.MemoryInfoGetId(memory_info, &device_id_signed);
  if (id_status != nullptr) {
    return id_status;
  }
  uint32_t device_id = static_cast<uint32_t>(device_id_signed);

  std::lock_guard<std::mutex> lock(factory->factory_mutex_);

  // Check if allocator already exists for this device.
  auto it = factory->allocators_.find(device_id);
  if (it != factory->allocators_.end()) {
    ORT_CXX_LOGF_NOEXCEPT(factory->logger_, ORT_LOGGING_LEVEL_INFO,
                          "IREE EP: Returning existing allocator for device %u",
                          device_id);
    *allocator = it->second.get();
    return nullptr;
  }

  // Validate device_id.
  if (device_id >= factory->device_memory_infos_.size()) {
    return Ort::Status("IREE EP: Invalid device_id for allocator",
                       ORT_INVALID_ARGUMENT)
        .release();
  }

  // Get or create the HAL device for this device_id.
  // Use locked variant since we already hold factory_mutex_.
  iree_hal_device_t* device = factory->GetDeviceForIdLocked(device_id);
  if (device == nullptr) {
    return Ort::Status("IREE EP: Failed to get device for allocator",
                       ORT_INVALID_ARGUMENT)
        .release();
  }

  // Use the memory_info we created and stored in GetSupportedDevicesImpl.
  // The memory_info is owned by the factory and kept alive for its lifetime.
  const OrtMemoryInfo* allocator_memory_info =
      factory->device_memory_infos_[device_id];

  // Create the allocator.
  auto alloc = std::make_unique<IreeAllocator>(
      device_id, device, allocator_memory_info, factory->logger_);

  ORT_CXX_LOGF_NOEXCEPT(factory->logger_, ORT_LOGGING_LEVEL_INFO,
                        "IREE EP: Created allocator for device %u", device_id);

  *allocator = alloc.get();
  factory->allocators_[device_id] = std::move(alloc);
  return nullptr;
}

/*static*/
void ORT_API_CALL IreeEpFactory::ReleaseAllocatorImpl(
    OrtEpFactory* /*this_ptr*/, OrtAllocator* /*allocator*/) noexcept {
  // Allocators are owned by factory and cleaned up in destructor.
  // Do not delete here.
}

/*static*/
OrtStatus* ORT_API_CALL IreeEpFactory::CreateDataTransferImpl(
    OrtEpFactory* this_ptr, OrtDataTransferImpl** data_transfer) noexcept {
  auto* factory = static_cast<IreeEpFactory*>(this_ptr);

  std::lock_guard<std::mutex> lock(factory->factory_mutex_);

  // Create data transfer if not already created.
  if (!factory->data_transfer_) {
    factory->data_transfer_ = std::make_unique<IreeDataTransfer>(*factory);
    ORT_CXX_LOG_NOEXCEPT(factory->logger_, ORT_LOGGING_LEVEL_INFO,
                         "IREE EP: Created data transfer instance");
  }

  *data_transfer = factory->data_transfer_.get();
  return nullptr;
}

/*static*/
bool ORT_API_CALL
IreeEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  // Not stream aware yet (synchronous execution only)
  return false;
}

/*static*/
OrtStatus* ORT_API_CALL IreeEpFactory::CreateSyncStreamForDeviceImpl(
    OrtEpFactory* /*this_ptr*/, const OrtMemoryDevice* /*memory_device*/,
    const OrtKeyValuePairs* /*stream_options*/,
    OrtSyncStreamImpl** stream) noexcept {
  // Not stream aware
  *stream = nullptr;
  return nullptr;
}

}  // namespace onnxruntime::iree
