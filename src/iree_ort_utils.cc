//===- iree_ort_utils.cc --------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ORT/IREE type conversion and tensor utilities.
//
//===----------------------------------------------------------------------===//

#include "iree_ort_utils.h"

#include <numeric>

#include "iree/hal/buffer_transfer.h"
#include "iree/hal/buffer_view_util.h"
#include "iree_ep_factory.h"

namespace onnxruntime::iree {

// ============================================================================
// Element Type Mapping
// ============================================================================

iree_hal_element_type_t OnnxToIreeElementType(
    ONNXTensorElementDataType onnx_type) {
  switch (onnx_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return IREE_HAL_ELEMENT_TYPE_SINT_8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return IREE_HAL_ELEMENT_TYPE_SINT_16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return IREE_HAL_ELEMENT_TYPE_SINT_32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return IREE_HAL_ELEMENT_TYPE_SINT_64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return IREE_HAL_ELEMENT_TYPE_UINT_8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return IREE_HAL_ELEMENT_TYPE_UINT_16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return IREE_HAL_ELEMENT_TYPE_UINT_32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return IREE_HAL_ELEMENT_TYPE_UINT_64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return IREE_HAL_ELEMENT_TYPE_BOOL_8;
    default:
      return IREE_HAL_ELEMENT_TYPE_NONE;
  }
}

ONNXTensorElementDataType IreeToOnnxElementType(
    iree_hal_element_type_t iree_type) {
  switch (iree_type) {
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case IREE_HAL_ELEMENT_TYPE_BFLOAT_16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
    case IREE_HAL_ELEMENT_TYPE_INT_8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
    case IREE_HAL_ELEMENT_TYPE_INT_16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
    case IREE_HAL_ELEMENT_TYPE_INT_32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
    case IREE_HAL_ELEMENT_TYPE_INT_64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    case IREE_HAL_ELEMENT_TYPE_BOOL_8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    default:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

size_t OnnxElementTypeSize(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return 8;
    default:
      return 0;
  }
}

// ============================================================================
// Buffer/Tensor Conversion
// ============================================================================

OrtStatus* OrtTensorToIreeBufferView(const Ort::ConstValue& ort_value,
                                     iree_hal_device_t* device,
                                     iree_hal_allocator_t* allocator,
                                     iree_allocator_t /*host_allocator*/,
                                     iree_hal_buffer_view_t** out_buffer_view,
                                     const OrtEpApi& ep_api,
                                     const Ort::Logger& logger) {
  *out_buffer_view = nullptr;

  // Get tensor info from ORT.
  auto type_info = ort_value.GetTensorTypeAndShapeInfo();
  auto onnx_dtype = type_info.GetElementType();
  auto shape = type_info.GetShape();

  // Convert element type.
  iree_hal_element_type_t iree_dtype = OnnxToIreeElementType(onnx_dtype);
  if (iree_dtype == IREE_HAL_ELEMENT_TYPE_NONE) {
    return Ort::Status("IREE EP: Unsupported element type", ORT_NOT_IMPLEMENTED)
        .release();
  }

  // Convert shape to IREE format.
  std::vector<iree_hal_dim_t> iree_shape(shape.begin(), shape.end());
  size_t byte_size = CalculateTensorByteSize(shape, onnx_dtype);

  // Check if tensor is already on an IREE device.
  const OrtMemoryDevice* mem_device =
      ep_api.Value_GetMemoryDevice(ort_value.operator const OrtValue*());
  if (mem_device) {
    uint32_t vendor_id = ep_api.MemoryDevice_GetVendorId(mem_device);
    if (vendor_id == kEpVendorId) {
      // Tensor already on IREE device - wrap existing buffer without copy.
      ORT_CXX_LOGF_NOEXCEPT(logger, ORT_LOGGING_LEVEL_INFO,
                            "IREE EP: Input tensor already on device, "
                            "wrapping existing buffer (%zu bytes)",
                            byte_size);

      const void* data_ptr = ort_value.GetTensorRawData();
      iree_hal_buffer_t* buffer =
          static_cast<iree_hal_buffer_t*>(const_cast<void*>(data_ptr));

      // Create buffer view wrapping existing buffer (no copy).
      IREE_ORT_RETURN_IF_ERROR(iree_hal_buffer_view_create(
          buffer, iree_shape.size(), iree_shape.data(), iree_dtype,
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
          iree_hal_device_host_allocator(device), out_buffer_view));

      return nullptr;
    }
  }

  // Tensor is on host - copy to device.
  ORT_CXX_LOGF_NOEXCEPT(logger, ORT_LOGGING_LEVEL_INFO,
                        "IREE EP: Copying input tensor to device (%zu bytes)",
                        byte_size);

  // Get raw data pointer.
  const void* data = ort_value.GetTensorRawData();

  // Set up buffer parameters for device-local memory.
  iree_hal_buffer_params_t buffer_params = {};
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;

  // Allocate buffer and copy data.
  IREE_ORT_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      device, allocator, iree_shape.size(), iree_shape.data(), iree_dtype,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, buffer_params,
      iree_make_const_byte_span(data, byte_size), out_buffer_view));

  return nullptr;
}

OrtStatus* IreeBufferViewToOrtTensor(iree_hal_buffer_view_t* buffer_view,
                                     Ort::UnownedValue ort_value,
                                     iree_hal_device_t* device,
                                     const OrtEpApi& ep_api,
                                     const Ort::Logger& logger) {
  // Get buffer from view.
  iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
  iree_device_size_t byte_length =
      iree_hal_buffer_view_byte_length(buffer_view);

  // Check if output tensor is on an IREE device.
  const OrtMemoryDevice* mem_device =
      ep_api.Value_GetMemoryDevice(ort_value.operator OrtValue*());
  if (mem_device) {
    uint32_t vendor_id = ep_api.MemoryDevice_GetVendorId(mem_device);
    if (vendor_id == kEpVendorId) {
      // Output is on device - copy buffer directly (D2D).
      ORT_CXX_LOGF_NOEXCEPT(logger, ORT_LOGGING_LEVEL_INFO,
                            "IREE EP: Output tensor on device, performing D2D "
                            "copy (%zu bytes)",
                            static_cast<size_t>(byte_length));

      iree_hal_buffer_t* dst_buffer =
          static_cast<iree_hal_buffer_t*>(ort_value.GetTensorMutableRawData());

      IREE_ORT_RETURN_IF_ERROR(iree_hal_device_transfer_d2d(
          device, buffer, /*source_offset=*/0, dst_buffer, /*target_offset=*/0,
          byte_length, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
          iree_infinite_timeout()));

      return nullptr;
    }
  }

  // Output tensor is on host - transfer from device to host.
  ORT_CXX_LOGF_NOEXCEPT(
      logger, ORT_LOGGING_LEVEL_INFO,
      "IREE EP: Copying output tensor from device (%zu bytes)",
      static_cast<size_t>(byte_length));

  // Get destination pointer from ORT tensor.
  void* dest_data = ort_value.GetTensorMutableRawData();

  // Transfer data from device to host.
  IREE_ORT_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
      device, buffer,
      /*source_offset=*/0, dest_data, byte_length,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  return nullptr;
}

std::vector<int64_t> GetBufferViewShape(iree_hal_buffer_view_t* buffer_view) {
  iree_host_size_t rank = iree_hal_buffer_view_shape_rank(buffer_view);
  const iree_hal_dim_t* dims = iree_hal_buffer_view_shape_dims(buffer_view);

  std::vector<int64_t> shape(rank);
  for (iree_host_size_t i = 0; i < rank; ++i) {
    shape[i] = static_cast<int64_t>(dims[i]);
  }
  return shape;
}

size_t CalculateTensorByteSize(const std::vector<int64_t>& shape,
                               ONNXTensorElementDataType element_type) {
  if (shape.empty()) {
    return OnnxElementTypeSize(element_type);  // Scalar.
  }

  size_t num_elements = std::accumulate(shape.begin(), shape.end(), size_t{1},
                                        std::multiplies<size_t>());
  return num_elements * OnnxElementTypeSize(element_type);
}

}  // namespace onnxruntime::iree
