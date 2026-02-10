//===- iree_data_transfer.cc ----------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Data transfer implementation for IREE EP.
//
//===----------------------------------------------------------------------===//

#include "iree_data_transfer.h"

#include "iree/hal/buffer_transfer.h"
#include "iree_ort_utils.h"

namespace onnxruntime::iree {

IreeDataTransfer::IreeDataTransfer(IreeEpFactory& factory) : factory_(factory) {
  // Initialize OrtDataTransferImpl base struct.
  ort_version_supported = ORT_API_VERSION;
  Release = ReleaseImpl;
  CanCopy = CanCopyImpl;
  CopyTensors = CopyTensorsImpl;
}

/*static*/
void ORT_API_CALL
IreeDataTransfer::ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept {
  // Data transfer is owned by factory, so we don't delete it here.
  // The factory will clean it up in its destructor.
  (void)this_ptr;
}

/*static*/
bool ORT_API_CALL IreeDataTransfer::CanCopyImpl(
    const OrtDataTransferImpl* this_ptr,
    const OrtMemoryDevice* src_memory_device,
    const OrtMemoryDevice* dst_memory_device) noexcept {
  const auto* self = static_cast<const IreeDataTransfer*>(this_ptr);

  // Get vendor IDs from both memory devices.
  uint32_t src_vendor_id =
      self->factory_.ep_api.MemoryDevice_GetVendorId(src_memory_device);
  uint32_t dst_vendor_id =
      self->factory_.ep_api.MemoryDevice_GetVendorId(dst_memory_device);

  // We can copy if either source or destination is our IREE device,
  // and the other is either our IREE device or CPU (vendor_id == 0).
  // TODO: Is the vendor id check for CPU actually correct? Maybe we can check
  // the hardware device for and determine if it's HOST/CPU.
  bool src_is_iree = (src_vendor_id == kEpVendorId);
  bool dst_is_iree = (dst_vendor_id == kEpVendorId);
  bool src_is_cpu = (src_vendor_id == 0);
  bool dst_is_cpu = (dst_vendor_id == 0);

  // Supported transfers:
  // - IREE <-> CPU (H2D, D2H)
  // - IREE <-> IREE (D2D on same device)
  return (src_is_iree && (dst_is_iree || dst_is_cpu)) ||
         (dst_is_iree && (src_is_iree || src_is_cpu));
}

/*static*/
OrtStatus* ORT_API_CALL IreeDataTransfer::CopyTensorsImpl(
    OrtDataTransferImpl* this_ptr, const OrtValue** src_tensors,
    OrtValue** dst_tensors, OrtSyncStream** streams,
    size_t num_tensors) noexcept {
  auto* self = static_cast<IreeDataTransfer*>(this_ptr);

  // TODO(async): Use streams parameter for async copy when streams != nullptr.
  (void)streams;

  for (size_t i = 0; i < num_tensors; ++i) {
    const OrtValue* src_value = src_tensors[i];
    OrtValue* dst_value = dst_tensors[i];

    // Get memory device info for source and destination.
    const OrtMemoryDevice* src_mem_device =
        self->factory_.ep_api.Value_GetMemoryDevice(src_value);
    const OrtMemoryDevice* dst_mem_device =
        self->factory_.ep_api.Value_GetMemoryDevice(dst_value);

    if (!src_mem_device || !dst_mem_device) {
      return Ort::Status("IREE EP: Cannot get memory device from tensor",
                         ORT_INVALID_ARGUMENT)
          .release();
    }

    uint32_t src_vendor_id =
        self->factory_.ep_api.MemoryDevice_GetVendorId(src_mem_device);
    uint32_t dst_vendor_id =
        self->factory_.ep_api.MemoryDevice_GetVendorId(dst_mem_device);
    uint32_t src_device_id =
        self->factory_.ep_api.MemoryDevice_GetDeviceId(src_mem_device);
    uint32_t dst_device_id =
        self->factory_.ep_api.MemoryDevice_GetDeviceId(dst_mem_device);

    bool src_is_iree = (src_vendor_id == kEpVendorId);
    bool dst_is_iree = (dst_vendor_id == kEpVendorId);

    // Get tensor data pointers and size.
    // IMPORTANT: Use Unowned wrappers to avoid releasing ORT-owned values.
    // The Unowned wrapper doesn't call OrtRelease in its destructor.
    Ort::UnownedValue src_ort_value{const_cast<OrtValue*>(src_value)};
    Ort::UnownedValue dst_ort_value{dst_value};

    auto src_type_info = src_ort_value.GetTensorTypeAndShapeInfo();
    size_t element_count = src_type_info.GetElementCount();
    ONNXTensorElementDataType dtype = src_type_info.GetElementType();

    // Calculate byte size.
    size_t element_size = OnnxElementTypeSize(dtype);
    if (element_size == 0) {
      return Ort::Status("IREE EP: Unsupported tensor element type",
                         ORT_NOT_IMPLEMENTED)
          .release();
    }

    size_t byte_size = element_count * element_size;

    ORT_CXX_LOGF_NOEXCEPT(
        self->factory_.Logger(), ORT_LOGGING_LEVEL_INFO,
        "IREE EP: Data transfer %zu/%zu: %s, %zu elements, %zu bytes, "
        "src_device=%u, dst_device=%u",
        i + 1, num_tensors, src_is_iree ? (dst_is_iree ? "D2D" : "D2H") : "H2D",
        element_count, byte_size, src_device_id, dst_device_id);

    if (src_is_iree && !dst_is_iree) {
      // Device to Host (D2H).
      // Source is an IREE buffer handle, destination is host memory.
      auto* src_buffer = static_cast<iree_hal_buffer_t*>(
          const_cast<void*>(src_ort_value.GetTensorRawData()));
      void* dst_ptr = dst_ort_value.GetTensorMutableRawData();

      // Get the HAL device for the source.
      iree_hal_device_t* device = self->factory_.GetDeviceForId(src_device_id);
      if (!device) {
        return Ort::Status("IREE EP: No device found for D2H transfer",
                           ORT_INVALID_ARGUMENT)
            .release();
      }

      IREE_ORT_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
          device, src_buffer, /*source_offset=*/0, dst_ptr,
          static_cast<iree_device_size_t>(byte_size),
          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

    } else if (!src_is_iree && dst_is_iree) {
      // Host to Device (H2D).
      // Source is host memory, destination is an IREE buffer handle.
      const void* src_ptr = src_ort_value.GetTensorRawData();
      auto* dst_buffer = static_cast<iree_hal_buffer_t*>(
          dst_ort_value.GetTensorMutableRawData());

      // Get the HAL device for the destination.
      iree_hal_device_t* device = self->factory_.GetDeviceForId(dst_device_id);
      if (!device) {
        return Ort::Status("IREE EP: No device found for H2D transfer",
                           ORT_INVALID_ARGUMENT)
            .release();
      }

      IREE_ORT_RETURN_IF_ERROR(iree_hal_device_transfer_h2d(
          device, src_ptr, dst_buffer, /*target_offset=*/0,
          static_cast<iree_device_size_t>(byte_size),
          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

    } else if (src_is_iree && dst_is_iree) {
      // Device to Device (D2D).
      // Both are IREE buffer handles.
      auto* src_buffer = static_cast<iree_hal_buffer_t*>(
          const_cast<void*>(src_ort_value.GetTensorRawData()));
      auto* dst_buffer = static_cast<iree_hal_buffer_t*>(
          dst_ort_value.GetTensorMutableRawData());

      // Use source device for the transfer.
      iree_hal_device_t* device = self->factory_.GetDeviceForId(src_device_id);
      if (!device) {
        return Ort::Status("IREE EP: No device found for D2D transfer",
                           ORT_INVALID_ARGUMENT)
            .release();
      }

      IREE_ORT_RETURN_IF_ERROR(iree_hal_device_transfer_d2d(
          device, src_buffer, /*source_offset=*/0, dst_buffer,
          /*target_offset=*/0, static_cast<iree_device_size_t>(byte_size),
          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

    } else {
      // CPU to CPU should be handled by ORT's default data transfer.
      return Ort::Status("IREE EP: CPU to CPU transfer not supported",
                         ORT_NOT_IMPLEMENTED)
          .release();
    }
  }

  return nullptr;
}

}  // namespace onnxruntime::iree
