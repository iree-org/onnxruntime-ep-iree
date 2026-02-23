//===- iree_compile.h -----------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Provides in-process MLIR-to-VMFB compilation via the IREE compiler C API.
// The compiler shared library (libIREECompiler.so) is loaded dynamically at
// runtime — no compile-time linking to the compiler is required.
//
//===----------------------------------------------------------------------===//

#ifndef ONNXRUNTIME_EP_IREE_SRC_IREE_COMPILE_H_
#define ONNXRUNTIME_EP_IREE_SRC_IREE_COMPILE_H_

#include <string>
#include <vector>

#include "ort_import.h"

namespace onnxruntime::iree {

// Manages the IREE compiler lifecycle and provides compilation.
//
// The compiler is a process-global singleton — the first successful
// Initialize() loads the shared library and that library is used for the
// lifetime of the process.
//
// Thread safety:
//   Initialize() — Thread-safe. Multiple threads may call concurrently;
//     exactly one will perform the actual initialization. If initialization
//     fails (e.g. bad library path), subsequent calls MAY retry with a
//     different path.
//   CompileToVmfb() — Thread-safe (each call creates its own compiler
//     session/invocation). Requires successful Initialize() first.
//   Shutdown() — NOT thread-safe. Must not be called concurrently with
//     Initialize() or CompileToVmfb(). Intended for explicit process-exit
//     cleanup only; do NOT call from factory destructors.
class IreeCompiler {
 public:
  // Load the compiler shared library and initialize global state.
  //
  // library_path: absolute path to libIREECompiler.so/.dylib/IREECompiler.dll.
  //   If empty, auto-discovers the library using this search order:
  //     1. Session option ep.iree.compiler_lib_path (passed as library_path)
  //     2. Environment variable IREE_EP_COMPILER_LIB (full path to library)
  //     3. Environment variable IREE_EP_COMPILER_LIB_DIR (directory to search)
  //     4. Build-time default directory (-DIREE_COMPILER_LIB_DIR)
  //     5. pip-installed iree-base-compiler (relative to EP shared library)
  //     6. Bare dlopen("libIREECompiler.so") via standard search paths
  //
  // Thread-safe. First successful call wins; subsequent calls are no-ops.
  // If initialization fails, subsequent calls with a different path may retry.
  static OrtStatus* Initialize(const std::string& library_path);

  // Shut down the compiler and release global state.
  // After shutdown, the compiler CANNOT be re-initialized in the same process.
  // Only call at process exit for explicit cleanup. Safe to call if not
  // initialized (no-op). NOT safe to call concurrently with other methods.
  static void Shutdown();

  // Returns true if Initialize() has been called successfully.
  static bool IsInitialized();

  // Compile an MLIR file to VMFB bytecode using the IREE compiler C API.
  //
  // Args:
  //   mlir_path: Path to the input MLIR file.
  //   vmfb_path: Path where the output VMFB should be written.
  //   flags: Compiler flags (e.g., ["--iree-hal-target-device=hip",
  //          "--iree-hip-target=gfx1100"]).
  //
  // Returns nullptr on success, OrtStatus* with error message on failure.
  // Requires successful Initialize() first.
  static OrtStatus* CompileToVmfb(const std::string& mlir_path,
                                   const std::string& vmfb_path,
                                   const std::vector<std::string>& flags);
};

}  // namespace onnxruntime::iree

#endif  // ONNXRUNTIME_EP_IREE_SRC_IREE_COMPILE_H_
