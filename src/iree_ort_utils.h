//===- iree_ort_utils.h ---------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Provides element type mapping between ONNX and IREE, and buffer/tensor
// conversion utilities for data transfer between ORT and IREE runtime.
//
//===----------------------------------------------------------------------===//

#ifndef ONNXRUNTIME_EP_IREE_SRC_IREE_ORT_UTILS_H_
#define ONNXRUNTIME_EP_IREE_SRC_IREE_ORT_UTILS_H_

#include <format>
#include <vector>

#include "iree/hal/api.h"
#include "iree_wrappers.h"
#include "ort_import.h"

namespace onnxruntime::iree {

// ============================================================================
// Error Helpers
// ============================================================================

// Creates an ORT error status. Accepts std::format arguments directly.
// Usage: return MakeError("expected {}, got {}", expected, actual);
template <typename... Args>
OrtStatus* MakeError(std::format_string<Args...> fmt, Args&&... args) {
  return Ort::Status(std::format(fmt, std::forward<Args>(args)...).c_str(),
                     ORT_FAIL)
      .release();
}

// ============================================================================
// Element Type Mapping
// ============================================================================

// Converts ONNX tensor element data type to IREE HAL element type.
// Returns IREE_HAL_ELEMENT_TYPE_NONE for unsupported types.
iree_hal_element_type_t OnnxToIreeElementType(
    ONNXTensorElementDataType onnx_type);

// Converts IREE HAL element type to ONNX tensor element data type.
// Returns ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED for unsupported types.
ONNXTensorElementDataType IreeToOnnxElementType(
    iree_hal_element_type_t iree_type);

// Returns the byte size of an ONNX element type.
// Returns 0 for unsupported types.
size_t OnnxElementTypeSize(ONNXTensorElementDataType type);

// ============================================================================
// Buffer/Tensor Conversion
// ============================================================================

// Converts an ORT input tensor to an IREE buffer view.
//
// If the tensor is already on an IREE device (detected via vendor_id), the
// existing buffer is wrapped in a view without copying data. Otherwise, a new
// buffer is allocated on the device and data is copied from host.
//
// Parameters:
//   ort_value - Input ORT tensor (const, read-only)
//   device - IREE HAL device for allocation
//   allocator - Device allocator from session
//   host_allocator - Host allocator for metadata
//   out_buffer_view - Output buffer view (caller must release)
//   ep_api - EP API for checking memory device info
//   logger - Logger for tracing
//
// Returns nullptr on success, OrtStatus* on error.
OrtStatus* OrtTensorToIreeBufferView(const Ort::ConstValue& ort_value,
                                     iree_hal_device_t* device,
                                     iree_hal_allocator_t* allocator,
                                     iree_allocator_t host_allocator,
                                     iree_hal_buffer_view_t** out_buffer_view,
                                     const OrtEpApi& ep_api,
                                     const Ort::Logger& logger);

// Copies data from an IREE buffer view to an ORT output tensor.
//
// If the output tensor is on an IREE device (detected via vendor_id), the
// buffer is copied directly without device-to-host transfer. Otherwise, data
// is transferred from device to host memory.
//
// Parameters:
//   buffer_view - Input IREE buffer view
//   ort_value - Output ORT tensor (mutable)
//   device - IREE HAL device for transfer
//   ep_api - EP API for checking memory device info
//   logger - Logger for tracing
//
// Returns nullptr on success, OrtStatus* on error.
OrtStatus* IreeBufferViewToOrtTensor(iree_hal_buffer_view_t* buffer_view,
                                     Ort::UnownedValue ort_value,
                                     iree_hal_device_t* device,
                                     const OrtEpApi& ep_api,
                                     const Ort::Logger& logger);

// Extracts shape from an IREE buffer view as int64_t vector for ORT.
std::vector<int64_t> GetBufferViewShape(iree_hal_buffer_view_t* buffer_view);

// Calculates total byte size from shape and element type.
size_t CalculateTensorByteSize(const std::vector<int64_t>& shape,
                               ONNXTensorElementDataType element_type);

}  // namespace onnxruntime::iree

#endif  // ONNXRUNTIME_EP_IREE_SRC_IREE_ORT_UTILS_H_
