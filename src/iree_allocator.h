//===- iree_allocator.h ---------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements OrtAllocator interface for allocating device-local memory using
// IREE's HAL allocator. The allocator returns buffer handles
// (iree_hal_buffer_t*) as opaque pointers, which ORT passes back to Free() when
// memory is released.
//
//===----------------------------------------------------------------------===//

#ifndef ONNXRUNTIME_EP_IREE_SRC_IREE_ALLOCATOR_H_
#define ONNXRUNTIME_EP_IREE_SRC_IREE_ALLOCATOR_H_

#include <mutex>
#include <unordered_map>

#include "iree_ep_factory.h"
#include "iree_wrappers.h"
#include "ort_import.h"

namespace onnxruntime::iree {

// Device memory allocator for IREE execution provider.
//
// Allocates device-local memory via IREE's HAL allocator. The Alloc() function
// returns a pointer to an iree_hal_buffer_t (cast to void*), not the actual
// device memory address. This handle is passed back to Free() when the memory
// should be released. This is similar to what the WebGPU EP does (passing
// handles to memory instead of raw pointers like CUDA EP).
//
// The allocator owns all allocated buffers and tracks them internally for
// proper cleanup. Buffers are automatically released when the allocator is
// destroyed.
class IreeAllocator : public OrtAllocator {
 public:
  // Creates an allocator for the specified device.
  //
  // Args:
  //   device_id: The IREE device ID (index in hw_devices_ list).
  //   device: Non-owning pointer to the HAL device. Must outlive this
  //   allocator.
  //   memory_info: ORT memory info describing this allocator. Takes ownership.
  //   logger: Logger for allocation tracing.
  IreeAllocator(uint32_t device_id, iree_hal_device_t* device,
                const OrtMemoryInfo* memory_info, const Ort::Logger& logger);

  ~IreeAllocator();

  // Non-copyable, non-movable (ORT holds pointer to this object).
  IreeAllocator(const IreeAllocator&) = delete;
  IreeAllocator& operator=(const IreeAllocator&) = delete;
  IreeAllocator(IreeAllocator&&) = delete;
  IreeAllocator& operator=(IreeAllocator&&) = delete;

 private:
  // OrtAllocator function pointer implementations.

  // Allocates device-local memory.
  // Returns iree_hal_buffer_t* cast to void* on success, nullptr on failure.
  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_, size_t size);

  // Frees memory previously allocated by AllocImpl.
  static void ORT_API_CALL FreeImpl(OrtAllocator* this_, void* p);

  // Returns the memory info describing this allocator.
  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_);

  uint32_t device_id_;
  iree_hal_device_t* device_;         // Non-owning (owned by factory/EP).
  const OrtMemoryInfo* memory_info_;  // Non-owning (owned by factory).
  Ort::Logger logger_;

  // Protects allocations_ for concurrent Alloc/Free from parallel inference.
  mutable std::mutex allocations_mutex_;

  // Tracks allocated buffers for ownership management.
  // Key: iree_hal_buffer_t* (the value returned by Alloc)
  // Value: RAII wrapper that releases buffer on destruction.
  std::unordered_map<void*, HalBufferPtr> allocations_;
};

}  // namespace onnxruntime::iree

#endif  // ONNXRUNTIME_EP_IREE_SRC_IREE_ALLOCATOR_H_
