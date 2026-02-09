// iree_data_transfer.h - Data transfer implementation for IREE EP.
//
// Implements OrtDataTransferImpl interface for copying tensors between
// host memory and IREE device memory, and between IREE devices.

#ifndef ONNXRUNTIME_EP_IREE_SRC_IREE_DATA_TRANSFER_H_
#define ONNXRUNTIME_EP_IREE_SRC_IREE_DATA_TRANSFER_H_

#include "iree_ep_factory.h"
#include "iree_wrappers.h"
#include "ort_import.h"

namespace onnxruntime::iree {

class IreeEpFactory;

// Data transfer implementation for IREE execution provider.
//
// Handles copying tensor data between:
// - Host (CPU) memory and IREE device memory (H2D, D2H)
// - IREE device memory on the same device (D2D)
//
// TODO: Move to async implementation.
class IreeDataTransfer : public OrtDataTransferImpl {
 public:
  // Creates a data transfer instance.
  //
  // Args:
  //   factory: Reference to the factory that owns device information.
  //            Must outlive this data transfer instance.
  explicit IreeDataTransfer(IreeEpFactory& factory);

  ~IreeDataTransfer() = default;

  // Non-copyable, non-movable.
  IreeDataTransfer(const IreeDataTransfer&) = delete;
  IreeDataTransfer& operator=(const IreeDataTransfer&) = delete;
  IreeDataTransfer(IreeDataTransfer&&) = delete;
  IreeDataTransfer& operator=(IreeDataTransfer&&) = delete;

 private:
  // OrtDataTransferImpl function pointer implementations.

  // Releases this data transfer instance.
  static void ORT_API_CALL ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept;

  // Checks if this implementation can copy between the specified devices.
  // Returns true if either device is an IREE device or host.
  static bool ORT_API_CALL
  CanCopyImpl(const OrtDataTransferImpl* this_ptr,
              const OrtMemoryDevice* src_memory_device,
              const OrtMemoryDevice* dst_memory_device) noexcept;

  // Copies tensors from src_tensors to dst_tensors.
  // Determines copy direction based on memory device info and uses
  // appropriate IREE transfer function (h2d, d2h, or d2d).
  static OrtStatus* ORT_API_CALL CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                 const OrtValue** src_tensors,
                                                 OrtValue** dst_tensors,
                                                 OrtSyncStream** streams,
                                                 size_t num_tensors) noexcept;

  IreeEpFactory& factory_;
};

}  // namespace onnxruntime::iree

#endif  // ONNXRUNTIME_EP_IREE_SRC_IREE_DATA_TRANSFER_H_
