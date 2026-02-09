// iree_wrappers.h - RAII wrappers for IREE C objects.
//
// Following the coding style requirement that all IREE C objects must use RAII.

#ifndef ONNXRUNTIME_EP_IREE_SRC_IREE_WRAPPERS_H_
#define ONNXRUNTIME_EP_IREE_SRC_IREE_WRAPPERS_H_

#include <string>

#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_provider.h"
#include "iree/runtime/api.h"
#include "ort_import.h"

namespace onnxruntime::iree {

// Generic RAII wrapper for IREE retain/release style objects.
template <typename T, void (*RetainFn)(T*), void (*ReleaseFn)(T*)>
class IreePtr {
 public:
  IreePtr() : ptr_(nullptr) {}
  explicit IreePtr(T* ptr) : ptr_(ptr) {}  // Takes ownership.

  ~IreePtr() { Reset(); }

  // Move-only semantics.
  IreePtr(IreePtr&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }

  IreePtr& operator=(IreePtr&& other) noexcept {
    if (this != &other) {
      Reset();
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  IreePtr(const IreePtr&) = delete;
  IreePtr& operator=(const IreePtr&) = delete;

  // Accessors.
  T* Get() const { return ptr_; }
  T* operator->() const { return ptr_; }
  explicit operator bool() const { return ptr_ != nullptr; }

  // For passing to IREE API out parameters.
  T** ForOutput() {
    Reset();
    return &ptr_;
  }

  // Release ownership without calling release function.
  T* Release() {
    T* tmp = ptr_;
    ptr_ = nullptr;
    return tmp;
  }

  void Reset(T* new_ptr = nullptr) {
    if (ptr_) {
      ReleaseFn(ptr_);
    }
    ptr_ = new_ptr;
  }

  // Create a new reference (retains).
  static IreePtr Borrow(T* ptr) {
    if (ptr) {
      RetainFn(ptr);
    }
    return IreePtr(ptr);
  }

 private:
  T* ptr_;
};

// Type aliases for IREE runtime types.
using RuntimeInstancePtr =
    IreePtr<iree_runtime_instance_t, iree_runtime_instance_retain,
            iree_runtime_instance_release>;

using RuntimeSessionPtr =
    IreePtr<iree_runtime_session_t, iree_runtime_session_retain,
            iree_runtime_session_release>;

// Type aliases for IREE HAL types.
using HalDevicePtr =
    IreePtr<iree_hal_device_t, iree_hal_device_retain, iree_hal_device_release>;

using HalDriverPtr =
    IreePtr<iree_hal_driver_t, iree_hal_driver_retain, iree_hal_driver_release>;

// RAII wrapper for memory allocated by IREE that needs iree_allocator_free.
template <typename T>
class IreeAllocatedPtr {
 public:
  explicit IreeAllocatedPtr(iree_allocator_t allocator)
      : allocator_(allocator), ptr_(nullptr) {}

  ~IreeAllocatedPtr() {
    if (ptr_) {
      iree_allocator_free(allocator_, ptr_);
    }
  }

  // Move-only semantics.
  IreeAllocatedPtr(IreeAllocatedPtr&& other) noexcept
      : allocator_(other.allocator_), ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }

  IreeAllocatedPtr& operator=(IreeAllocatedPtr&& other) noexcept {
    if (this != &other) {
      if (ptr_) {
        iree_allocator_free(allocator_, ptr_);
      }
      allocator_ = other.allocator_;
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  IreeAllocatedPtr(const IreeAllocatedPtr&) = delete;
  IreeAllocatedPtr& operator=(const IreeAllocatedPtr&) = delete;

  // For passing to IREE API out parameters.
  T** ForOutput() {
    if (ptr_) {
      iree_allocator_free(allocator_, ptr_);
      ptr_ = nullptr;
    }
    return &ptr_;
  }

  T* Get() const { return ptr_; }

 private:
  iree_allocator_t allocator_;
  T* ptr_;
};

using HalBufferViewPtr =
    IreePtr<iree_hal_buffer_view_t, iree_hal_buffer_view_retain,
            iree_hal_buffer_view_release>;

using HalBufferPtr =
    IreePtr<iree_hal_buffer_t, iree_hal_buffer_retain, iree_hal_buffer_release>;

// Type aliases for IREE IO / parameter types.
using ParameterIndexPtr =
    IreePtr<iree_io_parameter_index_t, iree_io_parameter_index_retain,
            iree_io_parameter_index_release>;

using ParameterProviderPtr =
    IreePtr<iree_io_parameter_provider_t, iree_io_parameter_provider_retain,
            iree_io_parameter_provider_release>;

using FileHandlePtr = IreePtr<iree_io_file_handle_t, iree_io_file_handle_retain,
                              iree_io_file_handle_release>;

using VmModulePtr =
    IreePtr<iree_vm_module_t, iree_vm_module_retain, iree_vm_module_release>;

// Wrapper for iree_runtime_call_t (stack-allocated, needs deinitialize).
class RuntimeCall {
 public:
  RuntimeCall() : initialized_(false) { std::memset(&call_, 0, sizeof(call_)); }

  ~RuntimeCall() { Deinitialize(); }

  RuntimeCall(RuntimeCall&&) = delete;
  RuntimeCall(const RuntimeCall&) = delete;
  RuntimeCall& operator=(RuntimeCall&&) = delete;
  RuntimeCall& operator=(const RuntimeCall&) = delete;

  iree_runtime_call_t* Get() { return &call_; }
  const iree_runtime_call_t* Get() const { return &call_; }

  void MarkInitialized() { initialized_ = true; }

  void Deinitialize() {
    if (initialized_) {
      iree_runtime_call_deinitialize(&call_);
      initialized_ = false;
    }
  }

 private:
  iree_runtime_call_t call_;
  bool initialized_;
};

// Converts iree_status_t to OrtStatus*.
// Returns nullptr on success, OrtStatus* on error.
inline OrtStatus* IreeStatusToOrtStatus(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return nullptr;
  }

  // Extract error message using IREE's allocator.
  iree_allocator_t allocator = iree_allocator_system();
  char* message_buffer = nullptr;
  iree_host_size_t message_size = 0;
  iree_status_to_string(status, &allocator, &message_buffer, &message_size);

  std::string message = message_buffer
                            ? std::string(message_buffer, message_size)
                            : "Unknown IREE error";
  if (message_buffer) {
    iree_allocator_free(allocator, message_buffer);
  }

  iree_status_code_t code = iree_status_code(status);
  iree_status_ignore(status);

  // Map IREE status codes to ORT status codes.
  OrtErrorCode ort_code = ORT_FAIL;
  switch (code) {
    case IREE_STATUS_INVALID_ARGUMENT:
      ort_code = ORT_INVALID_ARGUMENT;
      break;
    case IREE_STATUS_NOT_FOUND:
      ort_code = ORT_NO_SUCHFILE;
      break;
    case IREE_STATUS_OUT_OF_RANGE:
      ort_code = ORT_INVALID_ARGUMENT;
      break;
    case IREE_STATUS_UNIMPLEMENTED:
      ort_code = ORT_NOT_IMPLEMENTED;
      break;
    default:
      ort_code = ORT_FAIL;
      break;
  }

  return Ort::Status(message.c_str(), ort_code).release();
}

// Helper macro for IREE status checking (returns OrtStatus* on failure).
#define IREE_ORT_RETURN_IF_ERROR(expr)       \
  do {                                       \
    iree_status_t _status = (expr);          \
    if (!iree_status_is_ok(_status)) {       \
      return IreeStatusToOrtStatus(_status); \
    }                                        \
  } while (0)

}  // namespace onnxruntime::iree

#endif  // ONNXRUNTIME_EP_IREE_SRC_IREE_WRAPPERS_H_
