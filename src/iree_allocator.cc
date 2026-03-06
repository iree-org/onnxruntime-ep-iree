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

thread_local IreeAllocator::ActiveReuse IreeAllocator::active_reuse_ = {};

void IreeAllocator::SetActiveReuseQueue(ReuseQueue* queue) {
  active_reuse_.owner = queue ? this : nullptr;
  active_reuse_.queue = queue;
}

/*static*/
void IreeAllocator::DrainReuseQueue(ReuseQueue& queue) {
  while (!queue.empty()) {
    iree_hal_buffer_release(queue.front().buffer);
    queue.pop();
  }
}

/*static*/
void* ORT_API_CALL IreeAllocator::AllocImpl(OrtAllocator* this_, size_t size) {
  auto* self = static_cast<IreeAllocator*>(this_);

  if (size == 0) {
    return nullptr;
  }

  // Check the thread-local reuse queue. If ComputeImpl queued an IREE output
  // buffer on THIS thread and THIS allocator, reuse it directly (zero-copy).
  // The queue is thread-local so no lock is needed for the queue itself.
  if (active_reuse_.owner == self && active_reuse_.queue &&
      !active_reuse_.queue->empty()) {
    PendingBuffer pending = active_reuse_.queue->front();
    active_reuse_.queue->pop();

    // Validate byte size. On mismatch, the queue is desynchronized —
    // release this buffer, drain remaining entries, and fall through
    // to a fresh allocation.
    if (pending.byte_size != size) {
      ORT_CXX_LOGF_NOEXCEPT(
          self->logger_, ORT_LOGGING_LEVEL_WARNING,
          "IREE EP: Reuse queue size mismatch (expected %zu, got %zu) — "
          "draining queue and falling back to fresh alloc",
          size, pending.byte_size);
      iree_hal_buffer_release(pending.buffer);
      DrainReuseQueue(*active_reuse_.queue);
      // Fall through to fresh allocation below.
    } else {
      iree_hal_buffer_t* buffer = pending.buffer;

      std::lock_guard<std::mutex> lock(self->allocations_mutex_);
      auto it = self->allocations_.find(buffer);
      if (it != self->allocations_.end()) {
        // Buffer already tracked — IREE performed an in-place op (output
        // buffer IS the input buffer). ORT will call Free once for the input
        // tensor and once for the output tensor, so bump the ref count.
        // Release the extra retain from ComputeImpl (we don't need a new
        // HalBufferPtr, the existing one already owns the buffer).
        iree_hal_buffer_release(buffer);
        it->second.ref_count++;
        ORT_CXX_LOGF_NOEXCEPT(
            self->logger_, ORT_LOGGING_LEVEL_INFO,
            "IREE EP: Output zero-copy in-place reuse (%zu bytes, buffer=%p, "
            "ref_count=%d)",
            size, static_cast<void*>(buffer), it->second.ref_count);
        return buffer;
      } else {
        // New buffer from IREE — take ownership via HalBufferPtr.
        // The buffer was already retained by ComputeImpl.
        self->allocations_[buffer] = {HalBufferPtr(buffer), 1};
        ORT_CXX_LOGF_NOEXCEPT(
            self->logger_, ORT_LOGGING_LEVEL_INFO,
            "IREE EP: Output zero-copy reuse (%zu bytes, buffer=%p)", size,
            static_cast<void*>(buffer));
        return buffer;
      }
    }
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
  iree_hal_buffer_t* buffer_ptr = buffer.Get();
  {
    std::lock_guard<std::mutex> lock(self->allocations_mutex_);
    self->allocations_[buffer_ptr] = {std::move(buffer), 1};
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

  // Decrement ref count. Only release when all ORT tensors sharing this
  // buffer have been freed (ref_count reaches 0).
  if (--it->second.ref_count > 0) {
    ORT_CXX_LOGF_NOEXCEPT(
        self->logger_, ORT_LOGGING_LEVEL_INFO,
        "IREE EP: Decremented ref_count for buffer %p on device %u "
        "(remaining=%d)",
        p, self->device_id_, it->second.ref_count);
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
