//===- iree_ort_utils.h ---------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Error conversion utilities between OrtStatus*, iree_status_t, and the
// internal ErrorObject/ErrorCode types; element type mapping between ONNX and
// IREE; and buffer/tensor conversion for data transfer between ORT and IREE.
//
//===----------------------------------------------------------------------===//

#ifndef ONNXRUNTIME_EP_IREE_SRC_IREE_ORT_UTILS_H_
#define ONNXRUNTIME_EP_IREE_SRC_IREE_ORT_UTILS_H_

#include <format>
#include <vector>

#include "iree/hal/api.h"
#include "iree_wrappers.h"
#include "ort_import.h"
#include "support.h"

namespace onnxruntime::iree {

// ============================================================================
// Error Helpers
// ============================================================================

// Creates an ORT error status with ORT_FAIL. Accepts std::format arguments.
// Usage: return MakeError("expected {}, got {}", expected, actual);
template <typename... Args>
OrtStatus* MakeError(std::format_string<Args...> fmt, Args&&... args) {
  return Ort::Status(std::format(fmt, std::forward<Args>(args)...).c_str(),
                     ORT_FAIL)
      .release();
}

// Maps OrtErrorCode to the nearest internal ErrorCode. This mapping is
// semantic and deliberately lossy: distinct ORT codes may collapse to the same
// category (e.g. ORT_INVALID_GRAPH -> kInvalidArgument). That is intentional;
// see the ErrorCode comment in support.h.
inline ErrorCode FromOrtErrorCode(OrtErrorCode code) {
  switch (code) {
    case ORT_INVALID_ARGUMENT:
    case ORT_INVALID_PROTOBUF:
    case ORT_INVALID_GRAPH:
      return ErrorCode::kInvalidArgument;
    case ORT_NOT_FOUND:
    case ORT_NO_MODEL:  // model not loaded, not a file-system error
      return ErrorCode::kNotFound;
    case ORT_NO_SUCHFILE:
      return ErrorCode::kNoSuchFile;
    case ORT_NOT_IMPLEMENTED:
      return ErrorCode::kNotImplemented;
    default:
      return ErrorCode::kUnknown;
  }
}

// Maps internal ErrorCode to OrtErrorCode. Only called at the ORT boundary.
inline OrtErrorCode ToOrtErrorCode(ErrorCode code) {
  switch (code) {
    case ErrorCode::kInvalidArgument:
      return ORT_INVALID_ARGUMENT;
    case ErrorCode::kNotFound:
      return ORT_NOT_FOUND;
    case ErrorCode::kNoSuchFile:
      return ORT_NO_SUCHFILE;
    case ErrorCode::kNotImplemented:
      return ORT_NOT_IMPLEMENTED;
    default:
      return ORT_FAIL;
  }
}

// Converts an ErrorObject to OrtStatus*, preserving error category.
inline OrtStatus* ToOrtStatus(const ErrorObject& err) {
  return Ort::Status(err.message.c_str(), ToOrtErrorCode(err.code)).release();
}

// Converts an Ort::Status to ErrorObject, consuming the status.
// Only call when !status.IsOK().
inline ErrorObject OrtStatusToErrorObject(Ort::Status status) {
  return ErrorObject{status.GetErrorMessage(),
                     FromOrtErrorCode(status.GetErrorCode())};
}

// Converts an iree_status_t to ErrorObject, consuming the status.
inline ErrorObject IreeStatusToErrorObject(iree_status_t status) {
  iree_allocator_t allocator = iree_allocator_system();
  char* buf = nullptr;
  iree_host_size_t buf_size = 0;
  iree_status_to_string(status, &allocator, &buf, &buf_size);
  std::string message = buf ? std::string(buf, buf_size) : "Unknown IREE error";
  if (buf) {
    iree_allocator_free(allocator, buf);
  }
  iree_status_code_t code = iree_status_code(status);
  iree_status_ignore(status);
  // Map IREE status codes to the nearest internal ErrorCode.
  ErrorCode error_code = ErrorCode::kUnknown;
  switch (code) {
    case IREE_STATUS_INVALID_ARGUMENT:
    case IREE_STATUS_OUT_OF_RANGE:
      error_code = ErrorCode::kInvalidArgument;
      break;
    case IREE_STATUS_NOT_FOUND:
      error_code = ErrorCode::kNoSuchFile;
      break;
    case IREE_STATUS_UNIMPLEMENTED:
      error_code = ErrorCode::kNotImplemented;
      break;
    default:
      break;
  }
  return ErrorObject{std::move(message), error_code};
}

// Evaluates `expr` (which must return iree_status_t), propagates the error as
// MaybeError if in error state. For use in MaybeError-returning functions.
//
// Usage (in a function returning MaybeError):
//   IREE_EP_RETURN_IF_IREE_ERROR(iree_fn(...));
#define IREE_EP_RETURN_IF_IREE_ERROR(expr)     \
  do {                                         \
    iree_status_t _iree_s = (expr);            \
    if (!iree_status_is_ok(_iree_s)) {         \
      return IreeStatusToErrorObject(_iree_s); \
    }                                          \
  } while (0)

// Evaluates `expr` (which must return OrtStatus*), propagates the error as
// MaybeError if non-null. For use in MaybeError-returning functions at points
// that call ORT APIs returning OrtStatus*.
//
// Usage (in a function returning MaybeError):
//   IREE_EP_RETURN_IF_ORT_STATUS(ort_fn(...).release());
#define IREE_EP_RETURN_IF_ORT_STATUS(expr)                \
  do {                                                    \
    if (OrtStatus* _ort_s = (expr))                       \
      return OrtStatusToErrorObject(Ort::Status(_ort_s)); \
  } while (0)

// Evaluates `expr` (which must return MaybeError), converts and returns as
// OrtStatus* if in error state. For use in OrtStatus*-returning functions at
// the ORT API boundary.
//
// Usage (in a function returning OrtStatus*):
//   IREE_ORT_RETURN_IF_MAYBE_ERROR(GenerateMlir(...));
#define IREE_ORT_RETURN_IF_MAYBE_ERROR(expr)         \
  do {                                               \
    auto _iree_ort_err_ = (expr);                    \
    if (isError(_iree_ort_err_))                     \
      return ToOrtStatus(_iree_ort_err_.getError()); \
  } while (0)

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

// ============================================================================
// Name Sanitization
// ============================================================================

// Sanitizes an ONNX name to be a valid MLIR SSA identifier.
// MLIR identifiers must match [a-zA-Z_][a-zA-Z0-9_$]*.
// Uses '$' as an escape delimiter with hex encoding so the mapping is
// injective. Alphanumerics and '_' pass through unchanged. '$' itself,
// leading digits, and all other characters are escaped as $XX$.
// Examples: "input-1" -> "input$2D$1", "input_1" -> "input_1",
//           "0abc" -> "$30$abc".
std::string SanitizeName(const std::string& name);

}  // namespace onnxruntime::iree

#endif  // ONNXRUNTIME_EP_IREE_SRC_IREE_ORT_UTILS_H_
