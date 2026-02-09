// iree_compile.h - MLIR to VMFB compilation via iree-compile CLI.
//
// Provides a function to compile MLIR files to IREE VMFB bytecode by invoking
// the iree-compile tool as a subprocess.

#ifndef ONNXRUNTIME_EP_IREE_SRC_IREE_COMPILE_H_
#define ONNXRUNTIME_EP_IREE_SRC_IREE_COMPILE_H_

#include <string>

#include "ort_import.h"

namespace onnxruntime::iree {

// Compiles MLIR to VMFB using iree-compile CLI.
//
// Args:
//   mlir_path: Path to the input MLIR file.
//   vmfb_path: Path where the output VMFB should be written.
//   flags: Additional flags to pass to iree-compile (e.g.,
//          ["--iree-hal-target-device=local", "--iree-input-type=onnx"]).
//   ort_api: ORT API for creating status objects.
//
// Returns:
//   nullptr on success, OrtStatus* with error message on failure.
OrtStatus* CompileToVmfb(const std::string& mlir_path,
                         const std::string& vmfb_path,
                         const std::vector<std::string>& flags,
                         const OrtApi& ort_api);

}  // namespace onnxruntime::iree

#endif  // ONNXRUNTIME_EP_IREE_SRC_IREE_COMPILE_H_
