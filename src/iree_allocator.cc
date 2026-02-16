//===- iree_allocator.cc --------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// IREE device memory allocator implementation.
//
//===----------------------------------------------------------------------===//

#include "iree_allocator.h"

#include <mutex>

#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"

namespace onnxruntime::iree {

IreeAllocator::IreeAllocator(uint32_t device_id, iree_hal_device_t* device,
                             const OrtMemoryInfo* memory_info,
                             const Ort::Logger& logger)
    : device_id_(device_id),
      device_(device),
      memory_info_(memory_info),
      logger_(logger) {
  // Initialize OrtAllocator base struct.
  version = ORT_API_VERSION;
  Alloc = AllocImpl;
  Free = FreeImpl;
  Info = InfoImpl;
  Reserve = AllocImpl;      // Use same implementation for Reserve.
  GetStats = nullptr;       // Optional, not implemented.
  AllocOnStream = nullptr;  // TODO(async): Implement async allocation.
}

IreeAllocator::~IreeAllocator() {
  // Note: Avoid logging during cleanup - ORT logging infrastructure may be torn
  // down before our destructors run during Python interpreter shutdown.
  //
  // Note: memory_info_ is owned by the factory (device_memory_infos_), not this
  // allocator. We just store a pointer to it. The factory keeps it alive.
  //
  // Allocations are automatically released via HalBufferPtr RAII.
}

/*static*/
void* ORT_API_CALL IreeAllocator::AllocImpl(OrtAllocator* this_, size_t size) {
  auto* self = static_cast<IreeAllocator*>(this_);

  if (size == 0) {
    return nullptr;
  }

  // Get the HAL allocator from the device.
  iree_hal_allocator_t* hal_allocator =
      iree_hal_device_allocator(self->device_);

  // Set up buffer parameters for device-local memory.
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_TRANSFER;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  // Allocate the buffer. IREE HAL allocators are thread-safe.
  // TODO(async): Track allocation semaphore for async deallocation.
  HalBufferPtr buffer;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      hal_allocator, params, static_cast<iree_device_size_t>(size),
      buffer.ForOutput());

  if (!iree_status_is_ok(status)) {
    // Log allocation failure.
    char* message = nullptr;
    iree_host_size_t message_size = 0;
    iree_allocator_t allocator = iree_allocator_system();
    iree_status_to_string(status, &allocator, &message, &message_size);
    ORT_CXX_LOGF_NOEXCEPT(
        self->logger_, ORT_LOGGING_LEVEL_ERROR,
        "IREE EP: Failed to allocate %zu bytes on device %u: %s", size,
        self->device_id_, message ? message : "unknown");
    if (message) {
      iree_allocator_free(allocator, message);
    }
    iree_status_ignore(status);
    return nullptr;
  }

  // Store in allocations map and return the buffer pointer.
  // TODO: We probably don't need to do this. This just adds an extra layer of
  // validation that when the pointer is freed, we actually owned it and free
  // any unfreed memory on teardown.
  iree_hal_buffer_t* buffer_ptr = buffer.Get();
  {
    std::lock_guard<std::mutex> lock(self->allocations_mutex_);
    self->allocations_[buffer_ptr] = std::move(buffer);
  }

  ORT_CXX_LOGF_NOEXCEPT(self->logger_, ORT_LOGGING_LEVEL_INFO,
                        "IREE EP: Allocated %zu bytes on device %u (buffer=%p)",
                        size, self->device_id_, static_cast<void*>(buffer_ptr));

  return buffer_ptr;
}

/*static*/
void ORT_API_CALL IreeAllocator::FreeImpl(OrtAllocator* this_, void* p) {
  auto* self = static_cast<IreeAllocator*>(this_);

  if (p == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> lock(self->allocations_mutex_);

  auto it = self->allocations_.find(p);
  if (it == self->allocations_.end()) {
    ORT_CXX_LOGF_NOEXCEPT(self->logger_, ORT_LOGGING_LEVEL_WARNING,
                          "IREE EP: Attempted to free unknown buffer %p", p);
    return;
  }

  ORT_CXX_LOGF_NOEXCEPT(self->logger_, ORT_LOGGING_LEVEL_INFO,
                        "IREE EP: Freeing buffer %p on device %u", p,
                        self->device_id_);

  // Erase from map - HalBufferPtr destructor releases the buffer.
  self->allocations_.erase(it);
}

/*static*/
const OrtMemoryInfo* ORT_API_CALL
IreeAllocator::InfoImpl(const OrtAllocator* this_) {
  const auto* self = static_cast<const IreeAllocator*>(this_);
  return self->memory_info_;
}

}  // namespace onnxruntime::iree
